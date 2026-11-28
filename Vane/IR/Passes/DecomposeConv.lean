import Vane.IR.Rewrite
import Vane.IR.Passes.DCE
import Vane.IR.Passes.Quantize
import Vane.Tiling

namespace Vane.IR

open Vane.Binary
open Vane.Float16
open Vane.Tiling
open Vane.Quant
open Vane.Conv

-- ============================================================================
-- Pass 6b: Pure CoreOp Conv Decomposition (Grouped Conv, Conv1d, 5x5, 7x7, ...)
-- ============================================================================
--
-- Lowers non-native `CoreOp.conv` nodes into strictly hardware-legal `1x1` and
-- `3x3` `CoreOp.conv` primitives (verified by untouched `Vane.Verify`):
-- 1. Grouped conv (`1 < groups`, not native 3x3 depthwise):
--    decomposes into per-group C-axis `slice` + dense `conv` + C-axis `concat`
--    (leveraging L3 zero-copy subview aliasing for zero extra copy tasks!).
-- 2. Non-native kernel shapes (`5x5`, `7x7`, `1xK`, `Kx1`, `1x1 s=2`, explicit
--    padding): decomposes mathematically EXACTLY via sign-monotone depthwise
--    `3x3` shifts + `3x3` sub-convs summed via `elemBinary .add`, folding bias
--    into the first sub-conv, post-activation onto the final sum, and optional
--    zero-offset crop `slice` when `Ho < subH` or `Wo < subW`.

/-- Pure sign-monotone `(shift, tap)` assignment for a single spatial axis.
    Given hardware `3x3` top/left pad `basePad ∈ {0, 1}`, kernel offset bounds
    `[minD, maxD]`, and tap offset `d = k - padBefore`, returns `(shift, tap)` with:
    - `tap ∈ {0, 1, 2}`
    - `shift + (tap - basePad) = d` (exact offset match)
    - `shift < 0 → tap ≤ basePad` and `shift > 0 → basePad ≤ tap`
      (exact zero-padded boundary preservation under shift composition!). -/
def assignConvAxisShiftTap (basePad : Nat) (minD maxD d : Int) : Int × Nat :=
  if basePad == 1 then
    if -1 ≤ d && d ≤ 1 then
      (0, (d + 1).toNat)
    else if d < -1 then
      if minD == -2 && d == -2 then
        (-1, 0)
      else
        let m : Nat := (-d).toNat
        let sMag : Nat := 2 * (m / 2)
        (-(sMag : Int), if m % 2 == 0 then 1 else 0)
    else
      if maxD == 2 && d == 2 then
        (1, 2)
      else
        let m : Nat := d.toNat
        let sMag : Nat := 2 * (m / 2)
        ((sMag : Int), if m % 2 == 0 then 1 else 2)
  else
    -- `basePad == 0` (stride-2 `pad=same` on even spatial extent: relTap ∈ {0, 1, 2})
    if 0 ≤ d && d ≤ 2 then
      (0, d.toNat)
    else if d < 0 then
      (d, 0)
    else
      let m : Nat := d.toNat
      ((3 * (m / 3) : Int), m % 3)

/-- Check whether `pads` on input `(h, w)` with `3x3` stride `s ∈ {1, 2}` matches
    native hardware `valid` or `same` padding. -/
def isNativeConv3x3Pads (h w s : Nat) (pads : Array Nat) : Bool :=
  if pads.size != 4 || (s != 1 && s != 2) then
    false
  else
    let pt := pads[0]!; let pl := pads[1]!; let pb := pads[2]!; let pr := pads[3]!
    if h + pt + pb < 3 || w + pl + pr < 3 then
      false
    else
      let onnxHo := (h + pt + pb - 3) / s + 1
      let onnxWo := (w + pl + pr - 3) / s + 1
      let validOk :=
        h >= 3 && w >= 3 &&
        pt == 0 && pl == 0 && pb == 0 && pr == 0 &&
        onnxHo == (h - 3) / s + 1 && onnxWo == (w - 3) / s + 1
      let sameHo := ceilDiv h s
      let sameWo := ceilDiv w s
      let padTotalH := if (sameHo - 1) * s + 3 > h then (sameHo - 1) * s + 3 - h else 0
      let padTotalW := if (sameWo - 1) * s + 3 > w then (sameWo - 1) * s + 3 - w else 0
      let ptS := padTotalH / 2
      let plS := padTotalW / 2
      let sameOk :=
        onnxHo == sameHo && onnxWo == sameWo &&
        pt == ptS && pl == plS && pb == padTotalH - ptS && pr == padTotalW - plS
      validOk || sameOk

/-- Return `true` iff a `CoreOp.conv` node is already natively supported by `Vane.Verify`
    (`1x1` dense stride-1 unpadded, or `3x3` dense/depthwise symmetric stride-1/2 valid/same). -/
def isNativeConvOp (xShape : Array Nat) (kh kw sh sw : Nat) (pads : Array Nat) (groups : Nat)
    (cout : Nat) : Bool :=
  if xShape.size == 2 then
    kh == 1 && kw == 1 && sh == 1 && sw == 1 && groups == 1 && pads == #[0, 0, 0, 0]
  else if xShape.size == 4 then
    let cin := xShape[1]!; let h := xShape[2]!; let w := xShape[3]!
    if kh == 1 && kw == 1 then
      sh == 1 && sw == 1 && groups == 1 && pads == #[0, 0, 0, 0]
    else if kh == 3 && kw == 3 then
      sh == sw && (sh == 1 || sh == 2) &&
      (groups == 1 || (groups == cin && cout == cin && cin > 1)) &&
      isNativeConv3x3Pads h w sh pads
    else
      false
  else
    false

private def lookupConstCoreMap (cm : Array (ValueId × TensorType × ConstPayload)) (v : ValueId) :
    Option (TensorType × ConstPayload) :=
  (cm.find? fun e => e.1 == v).map fun e => (e.2.1, e.2.2)

private def lookupDeqCoreMap (dm : Array (ValueId × QuantFormat × ValueId)) (v : ValueId) :
    Option (QuantFormat × ValueId) :=
  (dm.find? fun e => e.1 == v).map fun e => (e.2.1, e.2.2)

private def lookupTyCoreMap (tm : Array (ValueId × TensorType)) (v : ValueId) : Option TensorType :=
  (tm.find? fun e => e.1 == v).map Prod.snd

private structure ConvDecompState where
  nodes : Array (IRNode CoreOp)
  constMap : Array (ValueId × TensorType × ConstPayload)
  deqMap : Array (ValueId × QuantFormat × ValueId)
  tyMap : Array (ValueId × TensorType)
  nextId : ValueId
  deriving Inhabited

private def pushCoreNode (st : ConvDecompState) (nd : IRNode CoreOp) : ConvDecompState := Id.run do
  let mut tm := st.tyMap
  let mut cm := st.constMap
  let mut dm := st.deqMap
  for (v, ty) in nd.outputs do
    tm := tm.push (v, ty)
  match nd.op, nd.inputs.toList, nd.outputs.toList with
  | .const p, [], [(v, ty)] => cm := cm.push (v, ty, p)
  | .dequantize fmt, [qId], [(v, _)] => dm := dm.push (v, fmt, qId)
  | _, _, _ => pure ()
  return { st with nodes := st.nodes.push nd, constMap := cm, deqMap := dm, tyMap := tm }

private def extractConvWeightF64 (st : ConvDecompState) (wId : ValueId) (kh kw : Nat) (ctx : String) :
    Except String (TensorType × Array Float) := do
  match lookupDeqCoreMap st.deqMap wId with
  | some (_, qId) =>
    let some (qTy, .quant qSpec) := lookupConstCoreMap st.constMap qId
      | throw s!"{ctx}: dequantize input %{qId} must be a quantized constant"
    if qTy.shape.size != 4 then
      throw s!"{ctx}: conv weight must be 4D, got {qTy.shape.toList}"
    let cout := qTy.shape[0]!; let cinPerG := qTy.shape[1]!
    let fp16Bytes ← dequantizeSpecToFp16 qSpec cout (cinPerG * kh * kw)
    let some vals := (ConstPayload.fp16 fp16Bytes).toFloat64?
      | throw s!"{ctx}: failed to convert dequantized weights to float64"
    pure (qTy, vals)
  | none =>
    let some (wTy, wPay) := lookupConstCoreMap st.constMap wId
      | throw s!"{ctx}: conv weight %{wId} must be a constant"
    let some vals := wPay.toFloat64?
      | throw s!"{ctx}: conv weight %{wId} must be float"
    pure (wTy, vals)

private def intSign (x : Int) : Int :=
  if x < 0 then -1 else if x > 0 then 1 else 0

/-- Emit a single 1-pixel depthwise (or `Cin=1` dense) `3x3 stride=1 pad=same` shift node
    computing `Y[c, y, x] = X[c, y + stepY, x + stepX]` (0 outside bounds) for `stepY, stepX ∈ {-1, 0, 1}`. -/
private def emitUnitShift3x3 (st : ConvDecompState) (baseName : String)
    (srcVid : ValueId) (cin h w : Nat) (stepY stepX : Int) : ConvDecompState × ValueId := Id.run do
  let tapY : Nat := (1 + stepY).toNat
  let tapX : Nat := (1 + stepX).toNat
  let tapIdx : Nat := tapY * 3 + tapX
  let wId := st.nextId
  let outId := st.nextId + 1
  let st0 := { st with nextId := st.nextId + 2 }
  let xTy : TensorType := { shape := #[1, cin, h, w], dtype := .fp16 }
  let (wTy, groups, wVals) :=
    if cin > 1 then
      let arr := Id.run do
        let mut a := Array.replicate (cin * 9) 0.0
        for c in [:cin] do
          a := a.set! (c * 9 + tapIdx) 1.0
        return a
      ({ shape := #[cin, 1, 3, 3], dtype := .f32 : TensorType }, cin, arr)
    else
      let arr := (Array.replicate 9 0.0).set! tapIdx 1.0
      ({ shape := #[1, 1, 3, 3], dtype := .f32 : TensorType }, 1, arr)
  let st1 := pushCoreNode st0 {
    name := s!"{baseName}_w"
    op := .const (.f32 wVals)
    inputs := #[]
    outputs := #[(wId, wTy)]
  }
  let st2 := pushCoreNode st1 {
    name := baseName
    op := .conv 3 3 1 1 #[1, 1, 1, 1] groups false .none
    inputs := #[srcVid, wId]
    outputs := #[(outId, xTy)]
  }
  return (st2, outId)

/-- Decompose a single dense (`groups == 1`) non-native `CoreOp.conv` on 4D `[1, Cin, H, W]`
    into an exact sum of sign-monotone shifted `3x3` sub-convolutions. -/
private def decomposeDenseNonNativeConv
    (st : ConvDecompState) (ndName ctx : String)
    (xId : ValueId) (xShape : Array Nat)
    (wVals : Array Float) (bId? : Option ValueId)
    (kh kw sh sw : Nat) (pads : Array Nat) (act : ConvActivation)
    (outId : ValueId) (outTy : TensorType) : Except String ConvDecompState := do
  if xShape.size != 4 || outTy.shape.size != 4 then
    throw s!"{ctx}: non-native conv requires 4D input and output (got {xShape.toList} -> {outTy.shape.toList})"
  if pads.size != 4 then
    throw s!"{ctx}: conv pads must have length 4, got {pads.size}"
  if kh == 0 || kw == 0 then
    throw s!"{ctx}: conv kernel dims ({kh}x{kw}) must be >= 1"

  let cin := xShape[1]!; let h := xShape[2]!; let w := xShape[3]!
  let cout := outTy.shape[1]!; let ho := outTy.shape[2]!; let wo := outTy.shape[3]!
  if wVals.size != cout * cin * kh * kw then
    throw s!"{ctx}: conv weight size {wVals.size} != expected {cout * cin * kh * kw}"

  let pt := pads[0]!; let pl := pads[1]!
  let sHw : Nat ←
    if sh == 1 && sw == 1 then pure 1
    else if sh == 2 && sw == 2 then pure 2
    else if h == 1 && sh == 1 && sw == 2 then pure 2
    else
      throw s!"{ctx}: conv({kh}x{kw}, stride={sh}x{sw}) unsupported stride combination (supported: 1x1, 2x2, or 1x2 when H=1)"

  let subH := ceilDiv h sHw
  let subW := ceilDiv w sHw
  if ho > subH || wo > subW then
    throw s!"{ctx}: conv({kh}x{kw}, stride={sh}x{sw}, pads={pads.toList}) output {ho}x{wo} exceeds {subH}x{subW}"

  let padTotalH := if (subH - 1) * sHw + 3 > h then (subH - 1) * sHw + 3 - h else 0
  let padTotalW := if (subW - 1) * sHw + 3 > w then (subW - 1) * sHw + 3 - w else 0
  let basePt := padTotalH / 2
  let basePl := padTotalW / 2
  let subPads : Array Nat := #[basePt, basePl, padTotalH - basePt, padTotalW - basePl]

  let minDy : Int := -(pt : Int)
  let maxDy : Int := (kh : Int) - 1 - (pt : Int)
  let minDx : Int := -(pl : Int)
  let maxDx : Int := (kw : Int) - 1 - (pl : Int)

  -- Collect distinct `(sy, sx)` buckets with their `3x3` weight tensors `[cout, cin, 3, 3]`
  let mut buckets : Array ((Int × Int) × Array Float) := #[]
  let w3Size := cout * cin * 9
  for ky in [:kh] do
    let dy : Int := (ky : Int) - (pt : Int)
    let (sy, ty) := assignConvAxisShiftTap basePt minDy maxDy dy
    for kx in [:kw] do
      let dx : Int := (kx : Int) - (pl : Int)
      let (sx, tx) := assignConvAxisShiftTap basePl minDx maxDx dx
      let tap3 := ty * 3 + tx
      let key := (sy, sx)
      let bIdx ← match buckets.findIdx? (fun b => b.1 == key) with
        | some i => pure i
        | none =>
          let i := buckets.size
          buckets := buckets.push (key, Array.replicate w3Size 0.0)
          pure i
      let mut arr := buckets[bIdx]!.2
      for o in [:cout] do
        for c in [:cin] do
          let srcIdx := ((o * cin + c) * kh + ky) * kw + kx
          let dstIdx := (o * cin + c) * 9 + tap3
          arr := arr.set! dstIdx (arr[dstIdx]! + wVals[srcIdx]!)
      buckets := buckets.set! bIdx (key, arr)

  -- Filter out identically zero buckets (keep one `(0, 0)` zero bucket if all weights are zero)
  let mut active : Array ((Int × Int) × Array Float) := #[]
  for b in buckets do
    if b.2.any (fun v => v != 0.0) then
      active := active.push b
  if active.isEmpty then
    active := #[((0, 0), Array.replicate w3Size 0.0)]

  -- Move `(0, 0)` bucket to index 0 when present so bias folds into unshifted center tile
  if let some zIdx := active.findIdx? (fun b => b.1 == (0, 0)) then
    if zIdx > 0 then
      let zEntry := active[zIdx]!
      let mut reordered : Array ((Int × Int) × Array Float) := #[zEntry]
      for i in [:active.size] do
        if i != zIdx then
          reordered := reordered.push active[i]!
      active := reordered

  let mut curSt := st
  let mut shiftCache : Array ((Int × Int) × ValueId) := #[((0, 0), xId)]

  let ensureShift := fun (stIn : ConvDecompState) (cacheIn : Array ((Int × Int) × ValueId)) (sy sx : Int) => Id.run do
    let mut s := stIn
    let mut cache := cacheIn
    let mut cy : Int := 0
    let mut cx : Int := 0
    let mut curVid : ValueId := xId
    let steps := max sy.natAbs sx.natAbs
    for _ in [:steps] do
      let stepY := intSign (sy - cy)
      let stepX := intSign (sx - cx)
      let ny := cy + stepY
      let nx := cx + stepX
      match cache.find? (fun e => e.1 == (ny, nx)) with
      | some (_, v) =>
        curVid := v
      | none =>
        let tag := s!"{ndName}_sh_{if ny < 0 then s!"m{ny.natAbs}" else s!"p{ny.toNat}"}_{if nx < 0 then s!"m{nx.natAbs}" else s!"p{nx.toNat}"}"
        let (s', v') := emitUnitShift3x3 s tag curVid cin h w stepY stepX
        s := s'
        curVid := v'
        cache := cache.push ((ny, nx), v')
      cy := ny
      cx := nx
    return (s, cache, curVid)

  let subOutTy : TensorType := { shape := #[1, cout, subH, subW], dtype := outTy.dtype }
  let needsCrop := ho < subH || wo < subW
  let numTiles := active.size
  let mut tileOutVids : Array ValueId := Array.mkEmpty numTiles

  for ti in [:numTiles] do
    let ((sy, sx), wArr) := active[ti]!
    let (s1, cache1, shiftedVid) := ensureShift curSt shiftCache sy sx
    curSt := s1; shiftCache := cache1

    let wTileId := curSt.nextId
    curSt := { curSt with nextId := curSt.nextId + 1 }
    curSt := pushCoreNode curSt {
      name := s!"{ndName}_t{ti}_w"
      op := .const (.f32 wArr)
      inputs := #[]
      outputs := #[(wTileId, { shape := #[cout, cin, 3, 3], dtype := .f32 })]
    }
    let tileHasBias := ti == 0 && bId?.isSome
    let tileInps : Array ValueId := match tileHasBias, bId? with
      | true, some bId => #[shiftedVid, wTileId, bId]
      | _, _ => #[shiftedVid, wTileId]
    let isFinalSingle := numTiles == 1 && !needsCrop
    let tileAct := if numTiles == 1 then act else .none
    let tileOutId ←
      if isFinalSingle then pure outId
      else
        let v := curSt.nextId
        curSt := { curSt with nextId := curSt.nextId + 1 }
        pure v
    curSt := pushCoreNode curSt {
      name := if isFinalSingle then ndName else s!"{ndName}_t{ti}"
      op := .conv 3 3 sHw sHw subPads 1 tileHasBias tileAct
      inputs := tileInps
      outputs := #[(tileOutId, subOutTy)]
    }
    tileOutVids := tileOutVids.push tileOutId

  -- Sum sub-conv outputs if `numTiles > 1`
  let mut accVid := tileOutVids[0]!
  if numTiles > 1 then
    let hasPostAct := act != .none
    for ti in [1 : numTiles] do
      let isLastAdd := ti + 1 == numTiles
      let isFinalNode := isLastAdd && !hasPostAct && !needsCrop
      let sumOutId ←
        if isFinalNode then pure outId
        else
          let v := curSt.nextId
          curSt := { curSt with nextId := curSt.nextId + 1 }
          pure v
      curSt := pushCoreNode curSt {
        name := if isFinalNode then ndName else s!"{ndName}_sum{ti}"
        op := .elemBinary .add
        inputs := #[accVid, tileOutVids[ti]!]
        outputs := #[(sumOutId, subOutTy)]
      }
      accVid := sumOutId

    -- Apply post-activation once onto the final sum
    match act with
    | .none => pure ()
    | .relu =>
      let isFinalNode := !needsCrop
      let actOutId ←
        if isFinalNode then pure outId
        else
          let v := curSt.nextId
          curSt := { curSt with nextId := curSt.nextId + 1 }
          pure v
      curSt := pushCoreNode curSt {
        name := if isFinalNode then ndName else s!"{ndName}_relu"
        op := .lut .relu
        inputs := #[accVid]
        outputs := #[(actOutId, subOutTy)]
      }
      accVid := actOutId
    | .lut lspec =>
      let isFinalNode := !needsCrop
      let actOutId ←
        if isFinalNode then pure outId
        else
          let v := curSt.nextId
          curSt := { curSt with nextId := curSt.nextId + 1 }
          pure v
      curSt := pushCoreNode curSt {
        name := if isFinalNode then ndName else s!"{ndName}_act"
        op := .lut (.lut lspec)
        inputs := #[accVid]
        outputs := #[(actOutId, subOutTy)]
      }
      accVid := actOutId

  -- Optional zero-offset crop slice when `Ho < subH` or `Wo < subW` (64B-aligned `beg=[0,0,0,0]`!)
  if needsCrop then
    curSt := pushCoreNode curSt {
      name := ndName
      op := .slice #[0, 0, 0, 0] #[1, cout, ho, wo] #[1, 1, 1, 1]
      inputs := #[accVid]
      outputs := #[(outId, outTy)]
    }

  pure curSt

/-- Decompose all non-native convolutions in an `SSAGraph CoreOp`:
    - Grouped convolutions (`1 < groups`, not native 3x3 depthwise) → C-slice + dense conv + C-concat.
    - Non-native kernel shapes (`5x5`, `7x7`, `1xK`, `Kx1`, `1x1 s=2`, explicit pad) → exact
      sign-monotone depthwise `3x3` shift DAG + `3x3` sub-convs + PE add + activation + optional crop. -/
def decomposeConvPass (g : SSAGraph CoreOp) : Except String (SSAGraph CoreOp) := do
  let mut st : ConvDecompState := {
    nodes := Array.mkEmpty g.nodes.size
    constMap := #[]
    deqMap := #[]
    tyMap := g.inputs.map fun (v, _, ty) => (v, ty)
    nextId := g.nextValueId
  }

  for nd in g.nodes do
    let ctx := if nd.name.isEmpty then "decomposeConv" else s!"node '{nd.name}'"
    match nd.op, nd.outputs.toList with
    | .conv kh kw sh sw pads groups hasBias act, [(outId, outTy)] =>
      if nd.inputs.size < 2 then
        throw s!"{ctx}: conv expects >= 2 inputs"
      let xId := nd.inputs[0]!
      let wId := nd.inputs[1]!
      let bId? := if hasBias && nd.inputs.size >= 3 then some nd.inputs[2]! else none
      let some xTy := lookupTyCoreMap st.tyMap xId
        | throw s!"{ctx}: unknown input %{xId}"
      let cout := outTy.shape.getD 1 0

      if isNativeConvOp xTy.shape kh kw sh sw pads groups cout then
        st := pushCoreNode st nd
      else if xTy.shape.size != 4 || outTy.shape.size != 4 then
        -- Let downstream LayoutSelect validate/report any unexpected rank
        st := pushCoreNode st nd
      else
        let cin := xTy.shape[1]!; let h := xTy.shape[2]!; let w := xTy.shape[3]!
        let ho := outTy.shape[2]!; let wo := outTy.shape[3]!
        if groups == 0 || cin % groups != 0 || cout % groups != 0 then
          throw s!"{ctx}: incompatible grouped Conv channels cin={cin} cout={cout} groups={groups}"

        if groups > 1 then
          -- Grouped conv decomposition: per-group C-Slice + dense conv (+ kernel decomp if needed) + C-Concat
          let cinG := cin / groups
          let coutG := cout / groups
          let bVals? ← match bId? with
            | none => pure none
            | some bId =>
              let some (_, bPay) := lookupConstCoreMap st.constMap bId
                | throw s!"{ctx}: grouped conv bias %{bId} must be constant"
              let some bv := bPay.toFloat64?
                | throw s!"{ctx}: grouped conv bias must be float"
              if bv.size != cout then
                throw s!"{ctx}: grouped conv bias length {bv.size} != Cout={cout}"
              pure (some bv)

          -- Check whether each group conv is natively quant-capable and weight is `dequantize(const .quant)`
          let nativePerGroup := isNativeConvOp #[1, cinG, h, w] kh kw sh sw pads 1 coutG
          let quantPerGroup? : Option (QuantFormat × QuantWeightSpec) :=
            if nativePerGroup && sh == 1 && sw == 1 && (act == .none || act == .relu) then
              match lookupDeqCoreMap st.deqMap wId with
              | some (fmt, qId) =>
                match lookupConstCoreMap st.constMap qId with
                | some (_, .quant qSpec) => some (fmt, qSpec)
                | _ => none
              | none => none
            else none

          let wValsF64? ← match quantPerGroup? with
            | some _ => pure none
            | none =>
              let (wTy, vals) ← extractConvWeightF64 st wId kh kw ctx
              if wTy.shape != #[cout, cinG, kh, kw] then
                throw s!"{ctx}: grouped conv weight shape {wTy.shape.toList} != expected [{cout},{cinG},{kh},{kw}]"
              pure (some vals)

          let mut groupOutVids : Array ValueId := Array.mkEmpty groups
          let grpInTy : TensorType := { shape := #[1, cinG, h, w], dtype := xTy.dtype }
          let grpOutTy : TensorType := { shape := #[1, coutG, ho, wo], dtype := outTy.dtype }

          for gi in [:groups] do
            let xGId := st.nextId
            st := { st with nextId := st.nextId + 1 }
            st := pushCoreNode st {
              name := s!"{nd.name}_g{gi}_in"
              op := .slice #[0, gi * cinG, 0, 0] #[1, (gi + 1) * cinG, h, w] #[1, 1, 1, 1]
              inputs := #[xId]
              outputs := #[(xGId, grpInTy)]
            }
            let bGId? ← match bVals? with
              | none => pure none
              | some bv =>
                let bid := st.nextId
                st := { st with nextId := st.nextId + 1 }
                let bSlice := bv.extract (gi * coutG) ((gi + 1) * coutG)
                st := pushCoreNode st {
                  name := s!"{nd.name}_g{gi}_b"
                  op := .const (.f32 bSlice)
                  inputs := #[]
                  outputs := #[(bid, { shape := #[coutG], dtype := .f32 })]
                }
                pure (some bid)

            let yGId := st.nextId
            st := { st with nextId := st.nextId + 1 }
            match quantPerGroup? with
            | some (fmt, qSpec) =>
              let qSlice ← sliceQuantWeightSpec qSpec (gi * coutG) coutG cinG (kh * kw)
              let qGId := st.nextId
              let dqGId := st.nextId + 1
              st := { st with nextId := st.nextId + 2 }
              st := pushCoreNode st {
                name := s!"{nd.name}_g{gi}_qw"
                op := .const (.quant qSlice)
                inputs := #[]
                outputs := #[(qGId, { shape := #[coutG, cinG, kh, kw], dtype := DType.ofQuantFormat fmt })]
              }
              st := pushCoreNode st {
                name := s!"{nd.name}_g{gi}_dq"
                op := .dequantize fmt
                inputs := #[qGId]
                outputs := #[(dqGId, { shape := #[coutG, cinG, kh, kw], dtype := .fp16 })]
              }
              let gInps := match bGId? with | some b => #[xGId, dqGId, b] | none => #[xGId, dqGId]
              st := pushCoreNode st {
                name := s!"{nd.name}_g{gi}"
                op := .conv kh kw sh sw pads 1 bGId?.isSome act
                inputs := gInps
                outputs := #[(yGId, grpOutTy)]
              }
            | none =>
              let vals := wValsF64?.getD #[]
              let elemsPerGrp := coutG * cinG * kh * kw
              let wSlice := vals.extract (gi * elemsPerGrp) ((gi + 1) * elemsPerGrp)
              if nativePerGroup then
                let wGId := st.nextId
                st := { st with nextId := st.nextId + 1 }
                st := pushCoreNode st {
                  name := s!"{nd.name}_g{gi}_w"
                  op := .const (.f32 wSlice)
                  inputs := #[]
                  outputs := #[(wGId, { shape := #[coutG, cinG, kh, kw], dtype := .f32 })]
                }
                let gInps := match bGId? with | some b => #[xGId, wGId, b] | none => #[xGId, wGId]
                st := pushCoreNode st {
                  name := s!"{nd.name}_g{gi}"
                  op := .conv kh kw sh sw pads 1 bGId?.isSome act
                  inputs := gInps
                  outputs := #[(yGId, grpOutTy)]
                }
              else
                st ← decomposeDenseNonNativeConv st s!"{nd.name}_g{gi}" ctx
                  xGId #[1, cinG, h, w] wSlice bGId? kh kw sh sw pads act yGId grpOutTy
            groupOutVids := groupOutVids.push yGId

          st := pushCoreNode st {
            name := nd.name
            op := .concat 1
            inputs := groupOutVids
            outputs := #[(outId, outTy)]
          }
        else
          -- Dense (`groups == 1`) non-native kernel/stride/padding decomposition
          let (wTy, wVals) ← extractConvWeightF64 st wId kh kw ctx
          if wTy.shape != #[cout, cin, kh, kw] then
            throw s!"{ctx}: conv weight shape {wTy.shape.toList} != expected [{cout},{cin},{kh},{kw}]"
          st ← decomposeDenseNonNativeConv st nd.name ctx
            xId xTy.shape wVals bId? kh kw sh sw pads act outId outTy

    | _, _ =>
      st := pushCoreNode st nd

  pure (dce { g with nodes := st.nodes })

end Vane.IR
