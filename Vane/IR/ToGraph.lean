import Vane.IR.Pipeline
import Vane.Graph
import Vane.View
import Vane.Matmul
import Vane.Frontend.Onnx

namespace Vane.IR.ToGraph

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Tiling
open Vane.Quant
open Vane.Pe
open Vane.Lut
open Vane.Pool
open Vane.Conv
open Vane.View
open Vane.Matmul
open Vane.Graph
open Vane.Frontend.Onnx
open Vane.IR

-- ============================================================================
-- Temporary L1b `SSAGraph CoreOp → Vane.Graph.GraphSpec` Bridge (Until Phase C2)
-- ============================================================================

/-- Map `(headIdx, rowIdx)` in 4D `[1, H, M, K]` into head-stacked channel index `headIdx * M + rowIdx` in `[1, H*M, 1, K]`. -/
def packHeadRow (m hIdx mIdx : Nat) : Nat :=
  hIdx * m + mIdx

/-- Inverse of `packHeadRow`: recover `(headIdx, rowIdx)` from head-stacked channel index. -/
def unpackHeadRow (m idx : Nat) : Nat × Nat :=
  (idx / m, idx % m)

/-- Standard ONNX 4D multi-head attention tensor shape `[1, H, M, K]`. -/
def matmul4DShape (h m k : Nat) : Shape4D :=
  { n := 1, c := h, h := m, w := k }

/-- Vane head-stacked multi-head attention tensor shape `[1, H*M, 1, K]`. -/
def matmulStackedShape (h m k : Nat) : Shape4D :=
  { n := 1, c := h * m, h := 1, w := k }

/-- Compute right-aligned ONNX numpy-style broadcast shape of `a` and `b`. -/
def broadcastOnnxShapes (a b : Array Nat) : Except String (Array Nat) := do
  let outRank := max a.size b.size
  let mut out : Array Nat := Array.mkEmpty outRank
  for i in [:outRank] do
    let da := if i + a.size >= outRank then a[i + a.size - outRank]! else 1
    let db := if i + b.size >= outRank then b[i + b.size - outRank]! else 1
    if da != db && da != 1 && db != 1 then
      throw s!"incompatible ONNX broadcast shapes {a.toList} and {b.toList} at axis {i} ({da} vs {db})"
    out := out.push (max da db)
  pure out

/-- Result of bridging a `SSAGraph CoreOp` into a single-output `Vane.Graph.GraphSpec`. -/
structure BridgeResult where
  spec : GraphSpec
  outputId : String
  outputShape : Shape4D
  outputLogicalShape : Array Nat
  weightCount : Nat
  weightBytes : Nat
  deriving BEq

private def sanitizeGraphId (raw : String) (used : Array String) : String × Array String :=
  let cleaned := String.ofList <| raw.toList.map fun c =>
    if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
       c == '_' || c == '-' then c else '_'
  let chars := cleaned.toList
  let trimmed := ((chars.dropWhile (· == '_')).reverse.dropWhile (· == '_')).reverse
  let base0 := String.ofList trimmed
  let base := if base0.isEmpty then "node" else base0
  if !used.contains base then
    (base, used.push base)
  else Id.run do
    for idx in [2 : used.size + 4] do
      let cand := s!"{base}_{idx}"
      if !used.contains cand then
        return (cand, used.push cand)
    let fallback := s!"{base}_{used.size + 4}"
    return (fallback, used.push fallback)

/-- Convert a logical rank-N tensor shape (2D `[1,C]`, 3D `[H,M,K]`, or 4D `[1,C,H,W]`)
    into its ANE physical `Shape4D`. -/
def logicalToShape4D (dims : Array Nat) (ctx : String) : Except String Shape4D := do
  if dims.size == 4 then
    if dims[0]! != 1 then
      throw s!"{ctx}: batch dimension N={dims[0]!} != 1"
    pure { n := 1, c := dims[1]!, h := dims[2]!, w := dims[3]! }
  else if dims.size == 3 then
    pure { n := 1, c := dims[0]! * dims[1]!, h := 1, w := dims[2]! }
  else if dims.size == 2 then
    if dims[0]! != 1 then
      throw s!"{ctx}: batch dimension N={dims[0]!} != 1"
    pure { n := 1, c := dims[1]!, h := 1, w := 1 }
  else
    throw s!"{ctx}: unsupported logical tensor rank {dims.size} ({dims.toList})"

private def resolveBroadcastBridge (shapeA shapeB : Array Nat) (ctx : String) :
    Except String (Array Nat × Shape4D) := do
  let outLogical ← match broadcastOnnxShapes shapeA shapeB with
    | .ok s => pure s
    | .error e => throw s!"{ctx}: {e}"
  let expectedVane ← logicalToShape4D outLogical ctx
  let vA ← logicalToShape4D shapeA ctx
  let vB ← logicalToShape4D shapeB ctx
  let dA : TensorDims := { c := vA.c, h := vA.h, w := vA.w }
  let dB : TensorDims := { c := vB.c, h := vB.h, w := vB.w }
  let outD ← broadcastDims dA dB
  let outVane : Shape4D := { n := 1, c := outD.c, h := outD.h, w := outD.w }
  if outVane != expectedVane then
    throw s!"{ctx}: ONNX broadcast {shapeA.toList} x {shapeB.toList} -> {outLogical.toList} is incompatible with ANE physical layout {outVane.toArray.toList}"
  let checkOp := fun (sh : Array Nat) (v : Shape4D) (side : String) => do
    if v.numElems == 1 then pure ()
    else if sh.size != outLogical.size then
      throw s!"{ctx}: {side} shape {sh.toList} cannot broadcast to {outLogical.toList} under ANE physical layout"
    else if outLogical.size == 3 then
      let hX := sh[0]!; let mX := sh[1]!
      let hO := outLogical[0]!; let mO := outLogical[1]!
      if !((hX == hO && mX == mO) || (hX == 1 && mX == 1)) then
        throw s!"{ctx}: 3D broadcast {shapeA.toList} x {shapeB.toList} requires matching (H, M) or (1, 1) on {side}"
    else pure ()
  checkOp shapeA vA "LHS"
  checkOp shapeB vB "RHS"
  pure (outLogical, outVane)

private def resolveConv3x3PadMode (h w stride : Nat) (pads : Array Nat) (ctx : String) :
    Except String (PadMode × Nat × Nat) := do
  if pads.size != 4 then
    throw s!"{ctx}: conv3x3 pads must have length 4, got {pads.size}"
  let pt := pads[0]!; let pl := pads[1]!; let pb := pads[2]!; let pr := pads[3]!
  if h + pt + pb < 3 || w + pl + pr < 3 then
    throw s!"{ctx}: conv3x3 padded spatial dims ({h + pt + pb},{w + pl + pr}) < 3"
  let onnxHo := (h + pt + pb - 3) / stride + 1
  let onnxWo := (w + pl + pr - 3) / stride + 1
  if h >= 3 && w >= 3 then
    let shValid : ConvSpec := { kernelSize := 3, mode := .dense, h := h, w := w, cin := 1, cout := 1, stride := stride, pad := .valid }
    if onnxHo == shValid.outH && onnxWo == shValid.outW && pt == 0 && pl == 0 && pb == 0 && pr == 0 then
      return (.valid, shValid.outH, shValid.outW)
  let shSame : ConvSpec := { kernelSize := 3, mode := .dense, h := h, w := w, cin := 1, cout := 1, stride := stride, pad := .same }
  let padTotalH := if (shSame.outH - 1) * stride + 3 > h then (shSame.outH - 1) * stride + 3 - h else 0
  let padTotalW := if (shSame.outW - 1) * stride + 3 > w then (shSame.outW - 1) * stride + 3 - w else 0
  let ptS := shSame.padTop
  let plS := shSame.padLeft
  let pbS := padTotalH - ptS
  let prS := padTotalW - plS
  if onnxHo == shSame.outH && onnxWo == shSame.outW && pt == ptS && pl == plS && pb == pbS && pr == prS then
    return (.same, shSame.outH, shSame.outW)
  throw s!"{ctx}: conv3x3 padding (pads=[{pt},{pl},{pb},{pr}], out={onnxHo}x{onnxWo}) for {h}x{w} stride={stride} does not match Vane conv3x3 'valid' or 'same'"

private def validateFiniteFp16ScalarBridge (v : Float) (ctx : String) : Except String UInt16 := do
  if v.isNaN || v.isInf then
    throw s!"{ctx}: scalar {v} must be finite"
  let h := floatToF16CanonicalZero v
  if !f16IsFinite h then
    throw s!"{ctx}: scalar {v} overflows finite fp16 range"
  pure h

private def ln2ConstBridge : Float := 0.6931471805599453

/-- Lower a validated `SSAGraph CoreOp` into a `Vane.Graph.GraphSpec`, absorbing
    `dequantize(const (.quant spec))` directly into M8A quantized conv nodes. -/
def coreGraphToGraphSpec
    (g : SSAGraph CoreOp)
    (placement : EdgePlacementMode := .auto)
    (stripMode : StripMode := .off)
    (allowViewOps : Bool := true) : Except String BridgeResult := do
  validateGraph g
  if g.outputs.size != 1 then
    throw s!"ToGraph: GraphSpec currently supports 1 output, got {g.outputs.size}"
  if g.inputs.isEmpty then
    throw "ToGraph: graph has no runtime inputs"
  if g.inputs.size > 3 then
    throw s!"ToGraph: graph has {g.inputs.size} runtime inputs (Vane supports 1..3 inputs)"

  let mut usedIds : Array String := #[]
  let mut graphInputs : Array GraphInput := Array.mkEmpty g.inputs.size
  let mut idMap : Array (ValueId × String) := #[]
  let mut typeMap : Array (ValueId × TensorType) := #[]
  let mut constMap : Array (ValueId × TensorType × ConstPayload) := #[]
  let mut deqMap : Array (ValueId × QuantFormat × ValueId) := #[]

  let lookupId := fun (m : Array (ValueId × String)) (v : ValueId) (ctx : String) => do
    match m.find? fun e => e.1 == v with
    | some (_, s) => pure s
    | none => throw s!"{ctx}: unknown runtime SSA value %{v}"

  let lookupTy := fun (m : Array (ValueId × TensorType)) (v : ValueId) (ctx : String) => do
    match m.find? fun e => e.1 == v with
    | some (_, ty) => pure ty
    | none => throw s!"{ctx}: unknown SSA type for %{v}"

  let lookupConst := fun (m : Array (ValueId × TensorType × ConstPayload)) (v : ValueId) =>
    (m.find? fun e => e.1 == v).map fun e => (e.2.1, e.2.2)

  let lookupDeq := fun (m : Array (ValueId × QuantFormat × ValueId)) (v : ValueId) =>
    m.find? fun e => e.1 == v

  for (vid, name, ty) in g.inputs do
    let sh4 ← logicalToShape4D ty.shape s!"input '{name}'"
    let (gid, used') := sanitizeGraphId name usedIds
    usedIds := used'
    graphInputs := graphInputs.push { name := gid, shape := { c := sh4.c, h := sh4.h, w := sh4.w } }
    idMap := idMap.push (vid, gid)
    typeMap := typeMap.push (vid, ty)

  let mut vaneNodes : Array GraphNode := #[]
  let mut weightCount : Nat := 0
  let mut weightBytes : Nat := 0

  for idx in [:g.nodes.size] do
    let nd := g.nodes[idx]!
    let ctx := if nd.name.isEmpty then s!"node[{idx}]" else s!"node '{nd.name}'"
    let (baseId, used1) := sanitizeGraphId (if nd.name.isEmpty then s!"n{idx}" else nd.name) usedIds
    usedIds := used1

    match nd.op with
    | .const payload =>
      if nd.outputs.size != 1 then
        throw s!"{ctx}: const must have 1 output"
      let (vid, ty) := nd.outputs[0]!
      constMap := constMap.push (vid, ty, payload)
      typeMap := typeMap.push (vid, ty)

    | .dequantize fmt =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: dequantize must have 1 input and 1 output"
      let (vid, ty) := nd.outputs[0]!
      deqMap := deqMap.push (vid, fmt, nd.inputs[0]!)
      typeMap := typeMap.push (vid, ty)

    | .conv kh kw sh sw pads groups hasBias act =>
      if nd.inputs.size < 2 || nd.outputs.size != 1 then
        throw s!"{ctx}: conv expects >= 2 inputs and 1 output"
      let xId := nd.inputs[0]!
      let wId := nd.inputs[1]!
      let (outVid, outTy) := nd.outputs[0]!
      let xGid ← lookupId idMap xId ctx
      let xTy ← lookupTy typeMap xId ctx
      if xTy.shape.size != 4 && xTy.shape.size != 2 then
        throw s!"{ctx}: conv requires 4D [1,C,H,W] or 2D [1,C] input, got rank {xTy.shape.size} ({xTy.shape.toList})"
      let xSh4 ← logicalToShape4D xTy.shape ctx
      let cin := xSh4.c
      let h := xSh4.h
      let w := xSh4.w

      -- Optional bias extraction
      let bBytes? ←
        if hasBias then
          if nd.inputs.size < 3 then
            throw s!"{ctx}: conv hasBias=true but missing bias input"
          let bId := nd.inputs[2]!
          let some (_, bPay) := lookupConst constMap bId
            | throw s!"{ctx}: conv bias %{bId} must be a constant"
          let bb ← bPay.toFp16Bytes s!"{baseId}_bias"
          weightCount := weightCount + 1
          weightBytes := weightBytes + bb.size
          pure (some bb)
        else
          pure none

      -- Weight extraction: either absorbed M8A quantized `dequantize(const quant)` OR fp16 `const`
      let (wTy, fp16WeightBytes, quantSpec?) ←
        match lookupDeq deqMap wId with
        | some (_, _, qId) =>
          let some (qTy, .quant qSpec) := lookupConst constMap qId
            | throw s!"{ctx}: dequantize input %{qId} must be a quantized constant"
          if qTy.shape.size != 4 then
            throw s!"{ctx}: quantized weight must be 4D, got {qTy.shape.toList}"
          let cout := qTy.shape[0]!
          let cinW := qTy.shape[1]!
          if groups == 1 && sh == 1 && sw == 1 && ((kh == 1 && kw == 1) || (kh == 3 && kw == 3)) &&
             (act == .none || act == .relu) then
            validateQuantWeightSpec qSpec cout cinW (kh * kw)
            weightCount := weightCount + 1
            weightBytes := weightBytes + qSpec.codes.size
            pure (qTy, ByteArray.empty, some qSpec)
          else
            -- Fallback dequantization to fp16 if conv is not natively quant-capable
            let bs ← dequantizeSpecToFp16 qSpec cout (cinW * kh * kw)
            weightCount := weightCount + 1
            weightBytes := weightBytes + bs.size
            pure (qTy, bs, none)
        | none =>
          let some (wTy, wPay) := lookupConst constMap wId
            | throw s!"{ctx}: conv weight %{wId} must be a constant or dequantize(const)"
          let bs ← wPay.toFp16Bytes s!"{baseId}_weights"
          weightCount := weightCount + 1
          weightBytes := weightBytes + bs.size
          pure (wTy, bs, none)

      if wTy.shape.size != 4 then
        throw s!"{ctx}: conv weight must be 4D [Cout,Cin/G,kH,kW], got {wTy.shape.toList}"
      let cout := wTy.shape[0]!
      let cinPerG := wTy.shape[1]!
      if groups == 0 || cin % groups != 0 || cout % groups != 0 || cin / groups != cinPerG then
        throw s!"{ctx}: incompatible Conv channels cin={cin} cout={cout} group={groups} weight_cin_per_g={cinPerG}"

      if kh == 1 && kw == 1 then
        if sh != 1 || sw != 1 then
          throw s!"{ctx}: conv1x1 requires strides [1,1], got [{sh},{sw}]"
        if groups != 1 then
          throw s!"{ctx}: conv1x1 requires group=1, got {groups}"
        if pads != #[0, 0, 0, 0] then
          throw s!"{ctx}: conv1x1 requires pads [0,0,0,0], got {pads.toList}"
        vaneNodes := vaneNodes.push {
          id := baseId
          inputs := #[xGid]
          op := .conv1x1 cin cout fp16WeightBytes bBytes? act quantSpec?
        }
      else if kh == 3 && kw == 3 then
        if sh != sw || (sh != 1 && sh != 2) then
          throw s!"{ctx}: conv3x3 requires symmetric stride 1 or 2, got [{sh},{sw}]"
        let mode ←
          if groups == 1 then pure Conv3x3Mode.dense
          else if groups == cin && cout == cin then pure Conv3x3Mode.depthwise
          else throw s!"{ctx}: grouped conv3x3 (group={groups}, cin={cin}, cout={cout}) is not supported"
        let (padMode, _, _) ← resolveConv3x3PadMode h w sh pads ctx
        vaneNodes := vaneNodes.push {
          id := baseId
          inputs := #[xGid]
          op := .conv3x3 mode cin cout sh padMode fp16WeightBytes bBytes? act quantSpec?
        }
      else
        throw s!"{ctx}: unsupported Conv kernel size {kh}x{kw} (supported: 1x1, 3x3)"

      idMap := idMap.push (outVid, baseId)
      typeMap := typeMap.push (outVid, outTy)

    | .matmul trA trB scaleBits =>
      if nd.inputs.size != 2 || nd.outputs.size != 1 then
        throw s!"{ctx}: matmul expects 2 inputs and 1 output"
      let aId := nd.inputs[0]!
      let bId := nd.inputs[1]!
      let (outVid, outTy) := nd.outputs[0]!
      if (lookupConst constMap aId).isSome then
        throw s!"{ctx}: constant LHS with dynamic RHS in MatMul is not supported"
      let aGid ← lookupId idMap aId ctx
      let bGid ← lookupId idMap bId ctx
      let aTy ← lookupTy typeMap aId ctx
      let bTy ← lookupTy typeMap bId ctx
      if !f16IsFinite scaleBits || (scaleBits &&& 0x7fff) == 0 then
        throw s!"{ctx}: dynamic MatMul scale must be finite and non-zero in fp16"

      let is4DMatchingOut : Bool :=
        aTy.shape.size == 4 && bTy.shape.size == 4 && outTy.shape.size == 4 &&
        aTy.shape[0]! == 1 && bTy.shape[0]! == 1 && aTy.shape[1]! == bTy.shape[1]! &&
        (let (m4, ka4) := if trA then (aTy.shape[3]!, aTy.shape[2]!) else (aTy.shape[2]!, aTy.shape[3]!)
         let (kb4, n4) := if trB then (bTy.shape[3]!, bTy.shape[2]!) else (bTy.shape[2]!, bTy.shape[3]!)
         ka4 == kb4 && outTy.shape == #[1, aTy.shape[1]!, m4, n4])
      if aTy.shape.size == 3 && bTy.shape.size == 3 then
        let hA := aTy.shape[0]!; let rA := aTy.shape[1]!; let cA := aTy.shape[2]!
        let hB := bTy.shape[0]!; let rB := bTy.shape[1]!; let cB := bTy.shape[2]!
        if hA != hB then
          throw s!"{ctx}: 3D MatMul head count mismatch ({hA} != {hB})"
        let heads := hA
        let (m, ka) := if trA then (cA, rA) else (rA, cA)
        let (kb, n) := if trB then (cB, rB) else (rB, cB)
        if ka != kb then
          throw s!"{ctx}: 3D MatMul inner dimension mismatch A.K={ka} != B.K={kb}"
        validateMatmulSpec { b := heads, m := m, k := ka, n := n, transposeA := trA, transposeB := trB, scaleBits := scaleBits }
        vaneNodes := vaneNodes.push {
          id := baseId, inputs := #[aGid, bGid], op := .matmul heads trA trB scaleBits
        }
        idMap := idMap.push (outVid, baseId)
        typeMap := typeMap.push (outVid, { shape := #[heads, m, n], dtype := outTy.dtype })
      else if aTy.shape.size == 4 && bTy.shape.size == 4 &&
              (aTy.shape[2]! > 1 || bTy.shape[2]! > 1 || is4DMatchingOut) then
        let aSh ← logicalToShape4D aTy.shape ctx
        let bSh ← logicalToShape4D bTy.shape ctx
        let hA := aSh.c; let rA := aSh.h; let cA := aSh.w
        let hB := bSh.c; let rB := bSh.h; let cB := bSh.w
        if hA != hB then
          throw s!"{ctx}: 4D MatMul head count mismatch ({hA} != {hB})"
        let heads := hA
        let (m, ka) := if trA then (cA, rA) else (rA, cA)
        let (kb, n) := if trB then (cB, rB) else (rB, cB)
        if ka != kb then
          throw s!"{ctx}: 4D MatMul inner dimension mismatch A.K={ka} != B.K={kb}"
        validateMatmulSpec { b := heads, m := m, k := ka, n := n, transposeA := trA, transposeB := trB, scaleBits := scaleBits }
        let needPackA := rA != 1
        let needPackB := rB != 1
        let needUnpack := m != 1
        if (needPackA || needPackB || needUnpack) && !allowViewOps then
          throw s!"{ctx}: 4D MatMul [1,{heads},{rA},{cA}] x [1,{heads},{rB},{cB}] requires view reshape ops (--allow-view-ops=true)"

        let mut effA := aGid
        if needPackA then
          let stA := matmulStackedShape heads rA cA
          let _ ← planReshape aSh stA
          let (pA, u1) := sanitizeGraphId s!"{baseId}_packA" usedIds
          usedIds := u1
          vaneNodes := vaneNodes.push { id := pA, inputs := #[aGid], op := .reshape { c := stA.c, h := 1, w := stA.w } }
          effA := pA

        let mut effB := bGid
        if needPackB then
          let stB := matmulStackedShape heads rB cB
          let _ ← planReshape bSh stB
          let (pB, u2) := sanitizeGraphId s!"{baseId}_packB" usedIds
          usedIds := u2
          vaneNodes := vaneNodes.push { id := pB, inputs := #[bGid], op := .reshape { c := stB.c, h := 1, w := stB.w } }
          effB := pB

        let stackedOut := matmulStackedShape heads m n
        let out4D := matmul4DShape heads m n
        if needUnpack then
          let _ ← planReshape stackedOut out4D
          let (mmCoreId, u3) := sanitizeGraphId s!"{baseId}_mm" usedIds
          usedIds := u3
          vaneNodes := vaneNodes.push { id := mmCoreId, inputs := #[effA, effB], op := .matmul heads trA trB scaleBits }
          vaneNodes := vaneNodes.push { id := baseId, inputs := #[mmCoreId], op := .reshape { c := out4D.c, h := out4D.h, w := out4D.w } }
        else
          vaneNodes := vaneNodes.push { id := baseId, inputs := #[effA, effB], op := .matmul heads trA trB scaleBits }
        idMap := idMap.push (outVid, baseId)
        typeMap := typeMap.push (outVid, { shape := #[1, heads, m, n], dtype := outTy.dtype })
      else
        let aSh ← logicalToShape4D aTy.shape ctx
        let bSh ← logicalToShape4D bTy.shape ctx
        if aTy.shape.size != 4 || bTy.shape.size != 4 || aSh.h != 1 || bSh.h != 1 then
          throw s!"{ctx}: dynamic MatMul requires matching 4D [1,H,M,K], 3D [H,M,K], or head-stacked [1,H*M,1,K] inputs (got {aTy.shape.toList} and {bTy.shape.toList})"
        let ca := aSh.c; let wa := aSh.w
        let cb := bSh.c; let wb := bSh.w
        let (heads, m, k, n) ←
          if !trA && !trB then
            if wa == 0 || cb % wa != 0 then
              throw s!"{ctx}: head-stacked MatMul B.C={cb} not divisible by A.W={wa}"
            let h := cb / wa
            if h == 0 || ca % h != 0 then
              throw s!"{ctx}: head-stacked MatMul A.C={ca} not divisible by inferred heads={h}"
            pure (h, ca / h, wa, wb)
          else if trA && !trB then
            if ca != cb then
              throw s!"{ctx}: head-stacked MatMul (transposeA) requires A.C={ca} == B.C={cb}"
            pure (1, wa, ca, wb)
          else if !trA && trB then
            if wa != wb then
              throw s!"{ctx}: head-stacked MatMul (transposeB) requires A.W={wa} == B.W={wb}"
            pure (1, ca, wa, cb)
          else
            if ca != wb then
              throw s!"{ctx}: head-stacked MatMul (transposeA+B) requires A.C={ca} == B.W={wb} (heads=1)"
            pure (1, wa, ca, cb)
        validateMatmulSpec { b := heads, m := m, k := k, n := n, transposeA := trA, transposeB := trB, scaleBits := scaleBits }
        let outVane := matmulStackedShape heads m n
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[aGid, bGid], op := .matmul heads trA trB scaleBits }
        idMap := idMap.push (outVid, baseId)
        typeMap := typeMap.push (outVid, { shape := #[1, outVane.c, 1, outVane.w], dtype := outTy.dtype })

    | .elemUnary uop =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: elemUnary expects 1 input and 1 output"
      let xGid ← lookupId idMap nd.inputs[0]! ctx
      let (outVid, outTy) := nd.outputs[0]!
      match uop with
      | .abs =>
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .pe1 .abs }
      | .neg =>
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .pe1 (.scalarMul f16NegOne) }
      | .square =>
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .pe1 .square }
      | .log =>
        let (log2Id, u1) := sanitizeGraphId s!"{baseId}_log2" usedIds
        usedIds := u1
        let ln2Bits := floatToF16CanonicalZero ln2ConstBridge
        vaneNodes := vaneNodes.push { id := log2Id, inputs := #[xGid], op := .lut (.lut .log2) }
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[log2Id], op := .pe1 (.scalarMul ln2Bits) }
      | .softplus =>
        let (expId, u1) := sanitizeGraphId s!"{baseId}_exp" usedIds
        let (addId, u2) := sanitizeGraphId s!"{baseId}_add1" u1
        let (log2Id, u3) := sanitizeGraphId s!"{baseId}_log2" u2
        usedIds := u3
        let ln2Bits := floatToF16CanonicalZero ln2ConstBridge
        vaneNodes := vaneNodes.push { id := expId, inputs := #[xGid], op := .lut (.lut .exp) }
        vaneNodes := vaneNodes.push { id := addId, inputs := #[expId], op := .pe1 (.scalarAdd f16PosOne) }
        vaneNodes := vaneNodes.push { id := log2Id, inputs := #[addId], op := .lut (.lut .log2) }
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[log2Id], op := .pe1 (.scalarMul ln2Bits) }
      | .scalarAdd b =>
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .pe1 (.scalarAdd b) }
      | .scalarSub b =>
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .pe1 (.scalarSub b) }
      | .scalarRSub b =>
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .pe1 (.scalarRSub b) }
      | .scalarMul b =>
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .pe1 (.scalarMul b) }
      | .scalarDiv b =>
        let peOp ← PeOp.scalarDiv b
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .pe1 peOp }
      | .linearAct a b =>
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .pe1 (.linearActivation a b) }
      idMap := idMap.push (outVid, baseId)
      typeMap := typeMap.push (outVid, outTy)

    | .elemBinary bop =>
      if nd.inputs.size != 2 || nd.outputs.size != 1 then
        throw s!"{ctx}: elemBinary expects 2 inputs and 1 output"
      let aId := nd.inputs[0]!
      let bId := nd.inputs[1]!
      let (outVid, outTy) := nd.outputs[0]!
      let aConst? := lookupConst constMap aId
      let bConst? := lookupConst constMap bId
      if aConst?.isSome && bConst?.isSome then
        throw s!"{ctx}: both inputs are constants; constant folding for non-scalar binary ops is not supported"

      if (aConst?.isSome || bConst?.isSome) &&
         (bop == .add || bop == .sub || bop == .mul || bop == .div) then
        let (cTy, cPay, tensorVid, constOnLeft) := match aConst?, bConst? with
          | some (ty, pay), _ => (ty, pay, bId, true)
          | _, some (ty, pay) => (ty, pay, aId, false)
          | none, none => ({ shape := #[], dtype := .fp16 }, .f32 #[], aId, false)
        if cTy.numElems != 1 then
          throw s!"{ctx}: non-scalar constant operand (shape {cTy.shape.toList}) is not supported for {bop.name}"
        let some vals := cPay.toFloat64?
          | throw s!"{ctx}: constant scalar operand must be float"
        if vals.size != 1 then
          throw s!"{ctx}: constant scalar operand length {vals.size} != 1"
        let scBits ← validateFiniteFp16ScalarBridge vals[0]! ctx
        let xGid ← lookupId idMap tensorVid ctx
        let xTy ← lookupTy typeMap tensorVid ctx
        let xSh4 ← logicalToShape4D xTy.shape ctx
        let outSh4 ← logicalToShape4D outTy.shape ctx
        if outSh4 != xSh4 then
          throw s!"{ctx}: scalar-constant {bop.name} changes physical layout ({xSh4.toArray.toList} -> {outSh4.toArray.toList}; rank-increasing broadcast is not supported)"
        match bop with
        | .add =>
          vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .pe1 (.scalarAdd scBits) }
        | .mul =>
          vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .pe1 (.scalarMul scBits) }
        | .sub =>
          let peOp := if constOnLeft then PeOp.scalarRSub scBits else PeOp.scalarSub scBits
          vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .pe1 peOp }
        | .div =>
          if !constOnLeft then
            let peOp ← PeOp.scalarDiv scBits
            vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .pe1 peOp }
          else
            let (invId, u1) := sanitizeGraphId s!"{baseId}_inv" usedIds
            usedIds := u1
            vaneNodes := vaneNodes.push { id := invId, inputs := #[xGid], op := .lut (.lut .inverse) }
            vaneNodes := vaneNodes.push { id := baseId, inputs := #[invId], op := .pe1 (.scalarMul scBits) }
        | _ => pure ()
        idMap := idMap.push (outVid, baseId)
        typeMap := typeMap.push (outVid, outTy)
      else
        if aConst?.isSome || bConst?.isSome then
          throw s!"{ctx}: non-scalar constant operand is not supported for {bop.name}"
        let aGid ← lookupId idMap aId ctx
        let bGid ← lookupId idMap bId ctx
        let aTy ← lookupTy typeMap aId ctx
        let bTy ← lookupTy typeMap bId ctx
        let (outLog, _) ← resolveBroadcastBridge aTy.shape bTy.shape ctx
        if bop == .div then
          let (invId, u1) := sanitizeGraphId s!"{baseId}_inv" usedIds
          usedIds := u1
          vaneNodes := vaneNodes.push { id := invId, inputs := #[bGid], op := .lut (.lut .inverse) }
          vaneNodes := vaneNodes.push { id := baseId, inputs := #[aGid, invId], op := .pe2 .mul false }
        else
          let peBin : PeBinOp := match bop with
            | .add => .add
            | .sub => .sub
            | .mul => .mul
            | .max => .max
            | _ => .min
          vaneNodes := vaneNodes.push { id := baseId, inputs := #[aGid, bGid], op := .pe2 peBin false }
        idMap := idMap.push (outVid, baseId)
        typeMap := typeMap.push (outVid, { shape := outLog, dtype := outTy.dtype })

    | .lut act =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: lut expects 1 input and 1 output"
      let xGid ← lookupId idMap nd.inputs[0]! ctx
      let (outVid, outTy) := nd.outputs[0]!
      vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .lut act }
      idMap := idMap.push (outVid, baseId)
      typeMap := typeMap.push (outVid, outTy)

    | .pool kind kh kw sh sw pad excl =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: pool expects 1 input and 1 output"
      let xId := nd.inputs[0]!
      let xGid ← lookupId idMap xId ctx
      let xTy ← lookupTy typeMap xId ctx
      if xTy.shape.size != 4 then
        throw s!"{ctx}: pool requires 4D [1,C,H,W] input, got rank {xTy.shape.size}"
      let xSh ← logicalToShape4D xTy.shape ctx
      let pspec : PoolSpec := {
        target := g.target, op := kind
        dims := { c := xSh.c, h := xSh.h, w := xSh.w }
        kh := kh, kw := kw, sh := sh, sw := sw, pad := pad, excludePad := excl
      }
      let (_, _, _) ← resolvePoolBackend pspec
      let (outVid, outTy) := nd.outputs[0]!
      vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .pool kind kh kw sh sw pad excl }
      idMap := idMap.push (outVid, baseId)
      typeMap := typeMap.push (outVid, outTy)

    | .reduce op axes keepDims eps =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: reduce expects 1 input and 1 output"
      if !keepDims then
        throw s!"{ctx}: keepdims=0 is not supported (Vane reduce preserves 4D rank with keepdims=1)"
      let xId := nd.inputs[0]!
      let xGid ← lookupId idMap xId ctx
      let xTy ← lookupTy typeMap xId ctx
      let xSh ← logicalToShape4D xTy.shape ctx
      let rank := xTy.shape.size
      let mut rc := false
      let mut rh := false
      let mut rw := false
      for ax in axes do
        if rank == 3 then
          if ax == 2 then rw := true
          else throw s!"{ctx}: 3D reduce axis {ax} not supported (expected axis -1 or 2)"
        else if rank == 2 then
          if ax == 1 then rc := true
          else throw s!"{ctx}: 2D reduce axis {ax} not supported (expected axis -1 or 1)"
        else
          if ax == 1 then rc := true
          else if ax == 2 then rh := true
          else if ax == 3 then rw := true
          else throw s!"{ctx}: reduce axis {ax} not supported (expected subset of \{1,2,3})"
      let rAxes : ReduceAxes := { reduceC := rc, reduceH := rh, reduceW := rw }
      if rAxes.isEmpty then
        throw s!"{ctx}: reduce resolved to empty axes"
      let (outVid, outTy) := nd.outputs[0]!
      if rAxes.reduceC && !rAxes.reduceH && !rAxes.reduceW && xSh.w > 1 && !op.isNeChannelSupported then
        match op with
        | .rsqrtMeanSquare | .rsqrtSumSquare =>
          if !f16IsFinite eps || ((eps &&& 0x8000) != 0 && (eps &&& 0x7fff) != 0) then
            throw s!"{ctx}: reduce {op.name} eps must be finite and >= 0"
          let redOp : ReduceOp := if op == .rsqrtMeanSquare then .mean else .sum
          let (sqId, u1) := sanitizeGraphId s!"{baseId}_sq" usedIds
          let (rdId, u2) := sanitizeGraphId s!"{baseId}_red" u1
          let (epId, u3) := sanitizeGraphId s!"{baseId}_eps" u2
          usedIds := u3
          vaneNodes := vaneNodes.push { id := sqId, inputs := #[xGid], op := .pe1 .square }
          vaneNodes := vaneNodes.push { id := rdId, inputs := #[sqId], op := .reduce redOp rAxes 0 }
          let preId := if (eps &&& 0x7fff) == 0 then rdId else epId
          if (eps &&& 0x7fff) != 0 then
            vaneNodes := vaneNodes.push { id := epId, inputs := #[rdId], op := .pe1 (.scalarAdd eps) }
          vaneNodes := vaneNodes.push { id := baseId, inputs := #[preId], op := .lut (.lut (.rsqrt 0.0)) }
        | .sumSquare =>
          let (sqId, u1) := sanitizeGraphId s!"{baseId}_sq" usedIds; usedIds := u1
          vaneNodes := vaneNodes.push { id := sqId, inputs := #[xGid], op := .pe1 .square }
          vaneNodes := vaneNodes.push { id := baseId, inputs := #[sqId], op := .reduce .sum rAxes 0 }
        | .l2 =>
          let (sqId, u1) := sanitizeGraphId s!"{baseId}_sq" usedIds
          let (rdId, u2) := sanitizeGraphId s!"{baseId}_sum" u1
          usedIds := u2
          vaneNodes := vaneNodes.push { id := sqId, inputs := #[xGid], op := .pe1 .square }
          vaneNodes := vaneNodes.push { id := rdId, inputs := #[sqId], op := .reduce .sum rAxes 0 }
          vaneNodes := vaneNodes.push { id := baseId, inputs := #[rdId], op := .lut (.lut .sqrt) }
        | .invSum =>
          let (rdId, u1) := sanitizeGraphId s!"{baseId}_sum" usedIds; usedIds := u1
          vaneNodes := vaneNodes.push { id := rdId, inputs := #[xGid], op := .reduce .sum rAxes 0 }
          vaneNodes := vaneNodes.push { id := baseId, inputs := #[rdId], op := .lut (.lut .inverse) }
        | _ =>
          let rspec : ReduceSpec := {
            target := g.target, op := op
            dims := { c := xSh.c, h := xSh.h, w := xSh.w }
            axes := rAxes, eps := eps
          }
          let _ ← resolveReduceBackend rspec
          throw s!"{ctx}: unsupported channel reduce op {op.name}"
      else
        let rspec : ReduceSpec := {
          target := g.target, op := op
          dims := { c := xSh.c, h := xSh.h, w := xSh.w }
          axes := rAxes, eps := eps
        }
        let _ ← resolveReduceBackend rspec
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .reduce op rAxes eps }
      idMap := idMap.push (outVid, baseId)
      typeMap := typeMap.push (outVid, outTy)

    | .reshape targetShape =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: reshape expects 1 input and 1 output"
      let xId := nd.inputs[0]!
      let xGid ← lookupId idMap xId ctx
      let xTy ← lookupTy typeMap xId ctx
      let inSh4 ← logicalToShape4D xTy.shape ctx
      let outSh4 ← logicalToShape4D targetShape ctx
      let (outVid, outTy) := nd.outputs[0]!
      if outSh4 == inSh4 then
        -- Physical no-op (e.g. [1,8,1] -> [1,8] both map to [1,8,1,1])
        idMap := idMap.push (outVid, xGid)
        typeMap := typeMap.push (outVid, outTy)
      else
        if !allowViewOps then
          throw s!"{ctx}: non-no-op view op Reshape ({inSh4.toArray.toList} -> {outSh4.toArray.toList}) requires --allow-view-ops=true"
        let _ ← planReshape inSh4 outSh4
        vaneNodes := vaneNodes.push {
          id := baseId, inputs := #[xGid]
          op := .reshape { c := outSh4.c, h := outSh4.h, w := outSh4.w }
        }
        idMap := idMap.push (outVid, baseId)
        typeMap := typeMap.push (outVid, outTy)

    | .transpose perm =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: transpose expects 1 input and 1 output"
      let xId := nd.inputs[0]!
      let xGid ← lookupId idMap xId ctx
      let xTy ← lookupTy typeMap xId ctx
      let inSh4 ← logicalToShape4D xTy.shape ctx
      let (outVid, outTy) := nd.outputs[0]!
      if isIdentityPerm perm then
        idMap := idMap.push (outVid, xGid)
        typeMap := typeMap.push (outVid, outTy)
      else if xTy.shape.size == 3 then
        if perm == #[0, 2, 1] then
          if !allowViewOps then
            throw s!"{ctx}: non-no-op view op Transpose (perm={perm.toList}) requires --allow-view-ops=true"
          let h0 := xTy.shape[0]!; let r0 := xTy.shape[1]!; let c0 := xTy.shape[2]!
          let outVane := matmulStackedShape h0 c0 r0
          if h0 == 1 then
            let _ ← planTranspose inSh4 #[0, 3, 2, 1]
            vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .transpose #[0, 3, 2, 1] }
          else
            let sh4In := matmul4DShape h0 r0 c0
            let sh4Out := matmul4DShape h0 c0 r0
            let _ ← planReshape inSh4 sh4In
            let _ ← planTranspose sh4In #[0, 1, 3, 2]
            let _ ← planReshape sh4Out outVane
            let (uId, u1) := sanitizeGraphId s!"{baseId}_unpack" usedIds
            let (tId, u2) := sanitizeGraphId s!"{baseId}_tr" u1
            usedIds := u2
            vaneNodes := vaneNodes.push { id := uId, inputs := #[xGid], op := .reshape { c := h0, h := r0, w := c0 } }
            vaneNodes := vaneNodes.push { id := tId, inputs := #[uId], op := .transpose #[0, 1, 3, 2] }
            vaneNodes := vaneNodes.push { id := baseId, inputs := #[tId], op := .reshape { c := outVane.c, h := 1, w := outVane.w } }
          idMap := idMap.push (outVid, baseId)
          typeMap := typeMap.push (outVid, outTy)
        else
          throw s!"{ctx}: unsupported 3D Transpose perm {perm.toList} (supported: [0,1,2] or [0,2,1])"
      else
        if perm.size != 4 then
          throw s!"{ctx}: perm must have length 4, got {perm.toList}"
        if !allowViewOps then
          throw s!"{ctx}: non-no-op view op Transpose (perm={perm.toList}) requires --allow-view-ops=true"
        let vp ← planTranspose inSh4 perm
        if vp.outShape.n != 1 then
          throw s!"{ctx}: Transpose output batch dimension must be 1, got {vp.outShape.n}"
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .transpose perm }
        idMap := idMap.push (outVid, baseId)
        typeMap := typeMap.push (outVid, outTy)

    | .slice beg en st =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: slice expects 1 input and 1 output"
      let xId := nd.inputs[0]!
      let xGid ← lookupId idMap xId ctx
      let xTy ← lookupTy typeMap xId ctx
      let inSh4 ← logicalToShape4D xTy.shape ctx
      let rank := xTy.shape.size
      let (outVid, outTy) := nd.outputs[0]!
      if beg.size != rank || en.size != rank || st.size != rank then
        throw s!"{ctx}: slice params length mismatch (beg={beg.size}, end={en.size}, step={st.size}) != input rank {rank}"
      if beg == Array.replicate rank 0 && en == xTy.shape && st == Array.replicate rank 1 then
        idMap := idMap.push (outVid, xGid)
        typeMap := typeMap.push (outVid, outTy)
      else
        if !allowViewOps then
          throw s!"{ctx}: non-no-op view op Slice requires --allow-view-ops=true"
        let (beg4, en4, st4) ←
          if rank == 4 then
            pure (beg, en, st)
          else if rank == 3 then
            if beg[0]! != 0 || en[0]! != xTy.shape[0]! || st[0]! != 1 ||
               beg[1]! != 0 || en[1]! != xTy.shape[1]! || st[1]! != 1 then
              throw s!"{ctx}: 3D Slice only supports slicing along last axis 2/-1 on 3D [H,M,K]"
            pure (#[0, 0, 0, beg[2]!], #[1, inSh4.c, 1, en[2]!], #[1, 1, 1, st[2]!])
          else if rank == 2 then
            pure (#[beg[0]!, beg[1]!, 0, 0], #[en[0]!, en[1]!, 1, 1], #[st[0]!, st[1]!, 1, 1])
          else
            throw s!"{ctx}: Slice unsupported rank {rank}"
        if beg4[3]! % 32 != 0 then
          throw s!"{ctx}: Slice along W: start offset {beg4[3]!} is not a multiple of 32 (64-byte DMA alignment)"
        let vp ← planSlice inSh4 beg4 en4 st4
        if vp.outShape.n != 1 then
          throw s!"{ctx}: Slice output batch dimension must be 1, got {vp.outShape.n}"
        vaneNodes := vaneNodes.push { id := baseId, inputs := #[xGid], op := .slice beg4 en4 st4 }
        idMap := idMap.push (outVid, baseId)
        typeMap := typeMap.push (outVid, outTy)

    | .concat axis =>
      if nd.inputs.size < 2 || nd.outputs.size != 1 then
        throw s!"{ctx}: concat expects >= 2 inputs and 1 output"
      let mut inGids : Array String := Array.mkEmpty nd.inputs.size
      let mut inTys : Array TensorType := Array.mkEmpty nd.inputs.size
      let mut inSh4s : Array Shape4D := Array.mkEmpty nd.inputs.size
      for inpVid in nd.inputs do
        let gid ← lookupId idMap inpVid ctx
        let ty ← lookupTy typeMap inpVid ctx
        let sh4 ← logicalToShape4D ty.shape ctx
        inGids := inGids.push gid
        inTys := inTys.push ty
        inSh4s := inSh4s.push sh4
      let rank := inTys[0]!.shape.size
      for i in [1:inTys.size] do
        if inTys[i]!.shape.size != rank then
          throw s!"{ctx}: Concat input[{i}] rank {inTys[i]!.shape.size} != input[0] rank {rank}"
      let vaneAxis : Nat ←
        if rank == 3 then
          if axis != 2 then
            throw s!"{ctx}: 3D Concat axis {axis} not supported (only last axis 2/-1 is supported on 3D [H,M,K])"
          let h0 := inTys[0]!.shape[0]!; let m0 := inTys[0]!.shape[1]!
          for i in [1:inTys.size] do
            if inTys[i]!.shape[0]! != h0 || inTys[i]!.shape[1]! != m0 then
              throw s!"{ctx}: 3D Concat input[{i}] shape {inTys[i]!.shape.toList} has mismatched (H,M) != ({h0},{m0})"
          pure 3
        else if rank == 2 then
          if axis != 1 then
            throw s!"{ctx}: 2D Concat axis {axis} not supported (expected axis 1 or -1)"
          pure 1
        else if rank == 4 then
          if axis < 1 || axis > 3 then
            throw s!"{ctx}: Concat axis {axis} must be in \{1,2,3}"
          pure axis
        else
          throw s!"{ctx}: Concat requires 4D, 3D, or 2D inputs, got rank {rank}"
      let _ ← planConcat inSh4s vaneAxis
      if !allowViewOps then
        throw s!"{ctx}: non-no-op view op Concat (axis={vaneAxis}) requires --allow-view-ops=true"
      let (outVid, outTy) := nd.outputs[0]!
      vaneNodes := vaneNodes.push { id := baseId, inputs := inGids, op := .concat vaneAxis }
      idMap := idMap.push (outVid, baseId)
      typeMap := typeMap.push (outVid, outTy)

  -- Final output resolution
  let outVid := g.outputs[0]!
  let outGid0 ← lookupId idMap outVid "graph output"
  let outTy ← lookupTy typeMap outVid "graph output"
  let outSh4 ← logicalToShape4D outTy.shape "graph output"

  let mut outGid := outGid0
  if vaneNodes.isEmpty || vaneNodes.back!.id != outGid then
    let (copyId, _) := sanitizeGraphId s!"{outGid}_out" usedIds
    vaneNodes := vaneNodes.push { id := copyId, inputs := #[outGid], op := .lut .copy }
    outGid := copyId

  let spec : GraphSpec := {
    target := g.target
    inputs := graphInputs
    nodes := vaneNodes
    defaultPlacement := placement
    stripMode := stripMode
  }

  pure {
    spec := spec
    outputId := outGid
    outputShape := outSh4
    outputLogicalShape := outTy.shape
    weightCount := weightCount
    weightBytes := weightBytes
  }

end Vane.IR.ToGraph
