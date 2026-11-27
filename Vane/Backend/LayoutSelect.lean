import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Tiling
import Vane.Quant
import Vane.Pe
import Vane.Lut
import Vane.Pool
import Vane.Conv
import Vane.View
import Vane.Matmul
import Vane.IR.Ops
import Vane.IR.Graph
import Vane.IR.Validate
import Vane.IR.Passes.Quantize
import Vane.Ane.Types

namespace Vane.Backend.LayoutSelect

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
open Vane.IR
open Vane.Ane

-- ============================================================================
-- Options & 4D Canonicalization Helpers
-- ============================================================================

/-- Configuration options for L1b (`CoreOp`) → L2 (`AneOp`) layout selection. -/
structure LayoutSelectOptions where
  placement : EdgePlacementMode := .auto
  allowViewOps : Bool := true
  outputNames : Array String := #[]
  donations : Array (Nat × Nat) := #[]
  deriving BEq, Inhabited

/-- Compute right-aligned ONNX numpy-style broadcast shape of `a` and `b`. -/
def broadcastShapes (a b : Array Nat) : Except String (Array Nat) := do
  let outRank := max a.size b.size
  let mut out : Array Nat := Array.mkEmpty outRank
  for i in [:outRank] do
    let da := if i + a.size >= outRank then a[i + a.size - outRank]! else 1
    let db := if i + b.size >= outRank then b[i + b.size - outRank]! else 1
    if da != db && da != 1 && db != 1 then
      throw s!"incompatible ONNX broadcast shapes {a.toList} and {b.toList} at axis {i} ({da} vs {db})"
    out := out.push (max da db)
  pure out

/-- Convert a logical rank-N tensor shape (2D `[1,C]`, 3D `[H,M,K]`, or 4D `[1,C,H,W]`)
    into its canonical ANE 4D physical layout `[1, C, H, W]`. -/
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

private def matmul4DShape (h m k : Nat) : Shape4D :=
  { n := 1, c := h, h := m, w := k }

private def matmulStackedShape (h m k : Nat) : Shape4D :=
  { n := 1, c := h * m, h := 1, w := k }

private def sanitizeName (raw : String) (used : Array String) : String × Array String :=
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

private def resolveBroadcast (shapeA shapeB : Array Nat) (ctx : String) :
    Except String (Array Nat × Shape4D) := do
  let outLogical ← match broadcastShapes shapeA shapeB with
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

private def validateFiniteFp16Scalar (v : Float) (ctx : String) : Except String UInt16 := do
  if v.isNaN || v.isInf then
    throw s!"{ctx}: scalar {v} must be finite"
  let h := floatToF16CanonicalZero v
  if !f16IsFinite h then
    throw s!"{ctx}: scalar {v} overflows finite fp16 range"
  pure h

private def ln2Const : Float := 0.6931471805599453

-- ============================================================================
-- L1b (`SSAGraph CoreOp`) → L2 (`AneProgram` / `SSAGraph AneOp`)
-- ============================================================================

/-- Lower a validated rank-N `SSAGraph CoreOp` into a 4D `[1, C, H, W]` value-semantic
    `AneProgram` (`SSAGraph AneOp`).
    - Absorbs `CoreOp.const` and `CoreOp.dequantize(const (.quant spec))` into `AneOp.neConv`.
    - Canonicalizes `Linear` → `neConv 1x1`, batched 3D/4D/head-stacked `MatMul` → `neMatmul`.
    - Inserts `dmaView` ONLY when a layout/transpose cannot be satisfied zero-copy.
    - Preserves ALL graph inputs (`1..3`) and MULTIPLE graph outputs (`1..3`, `in + out <= 4`). -/
def lowerCoreToAne (g : SSAGraph CoreOp) (opts : LayoutSelectOptions := {}) :
    Except String AneProgram := do
  validateGraph g
  if g.inputs.isEmpty then
    throw "LayoutSelect: graph has no runtime inputs"
  if g.outputs.isEmpty then
    throw "LayoutSelect: graph has no outputs"
  if g.inputs.size > 3 then
    throw s!"LayoutSelect: graph has {g.inputs.size} runtime inputs (Vane supports 1..3 inputs)"

  let mut usedNames : Array String := #[]
  let mut aneInputs : Array (ValueId × String × TensorType) := Array.mkEmpty g.inputs.size
  let mut valMap : Array (ValueId × ValueId) := #[]
  let mut logTyMap : Array (ValueId × TensorType) := #[]
  let mut sh4Map : Array (ValueId × Shape4D) := #[]
  let mut constMap : Array (ValueId × TensorType × ConstPayload) := #[]
  let mut deqMap : Array (ValueId × QuantFormat × ValueId) := #[]
  let mut nextId : ValueId := 0

  let lookupValChecked := fun (m : Array (ValueId × ValueId)) (dm : Array (ValueId × QuantFormat × ValueId))
      (v : ValueId) (ctx : String) => do
    match m.find? fun e => e.1 == v with
    | some (_, av) => pure av
    | none =>
      if (dm.find? fun e => e.1 == v).isSome then
        throw s!"{ctx}: CoreOp.dequantize value %{v} can only be consumed as a Conv weight input"
      throw s!"{ctx}: unknown runtime SSA value %{v}"

  let lookupLogTy := fun (m : Array (ValueId × TensorType)) (v : ValueId) (ctx : String) => do
    match m.find? fun e => e.1 == v with
    | some (_, ty) => pure ty
    | none => throw s!"{ctx}: unknown SSA type for %{v}"

  let lookupConst := fun (m : Array (ValueId × TensorType × ConstPayload)) (v : ValueId) =>
    (m.find? fun e => e.1 == v).map fun e => (e.2.1, e.2.2)

  let lookupDeq := fun (m : Array (ValueId × QuantFormat × ValueId)) (v : ValueId) =>
    m.find? fun e => e.1 == v

  for (vid, name, ty) in g.inputs do
    let sh4 ← logicalToShape4D ty.shape s!"input '{name}'"
    let (gid, u') := sanitizeName name usedNames
    usedNames := u'
    let aVid := nextId; nextId := nextId + 1
    aneInputs := aneInputs.push (aVid, gid, shape4DToTensorType sh4)
    valMap := valMap.push (vid, aVid)
    logTyMap := logTyMap.push (vid, ty)
    sh4Map := sh4Map.push (aVid, sh4)

  let mut aneNodes : Array (IRNode AneOp) := #[]
  let mut nodeMeta : Array AneNodeMeta := #[]
  let mut weightCount : Nat := 0
  let mut weightBytes : Nat := 0
  let mut nextWgId : Nat := 1

  let pushAneNode (nds : Array (IRNode AneOp)) (ms : Array AneNodeMeta)
      (shM : Array (ValueId × Shape4D)) (nid : ValueId)
      (nm logicalNm : String) (op : AneOp) (inps : Array ValueId) (outSh : Shape4D)
      (outPlace? : Option EdgePlacementMode := none) :=
    let vid := nid
    let nd : IRNode AneOp := {
      name := nm
      op := op
      inputs := inps
      outputs := #[(vid, shape4DToTensorType outSh)]
    }
    let ndMeta : AneNodeMeta := {
      logicalId := if logicalNm.isEmpty then nm else logicalNm
      outPlacement := outPlace?
    }
    (vid, nds.push nd, ms.push ndMeta, shM.push (vid, outSh), nid + 1)

  for idx in [:g.nodes.size] do
    let nd := g.nodes[idx]!
    let ctx := if nd.name.isEmpty then s!"node[{idx}]" else s!"node '{nd.name}'"
    let (baseName, used1) := sanitizeName (if nd.name.isEmpty then s!"n{idx}" else nd.name) usedNames
    usedNames := used1

    match nd.op with
    | .const payload =>
      if nd.outputs.size != 1 then
        throw s!"{ctx}: const must have 1 output"
      let (vid, ty) := nd.outputs[0]!
      constMap := constMap.push (vid, ty, payload)
      logTyMap := logTyMap.push (vid, ty)

    | .dequantize fmt =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: dequantize must have 1 input and 1 output"
      let (vid, ty) := nd.outputs[0]!
      deqMap := deqMap.push (vid, fmt, nd.inputs[0]!)
      logTyMap := logTyMap.push (vid, ty)

    | .conv kh kw sh sw pads groups hasBias act =>
      if nd.inputs.size < 2 || nd.outputs.size != 1 then
        throw s!"{ctx}: conv expects >= 2 inputs and 1 output"
      let xId := nd.inputs[0]!
      let wId := nd.inputs[1]!
      let (outVid, outTy) := nd.outputs[0]!
      let xAVid ← lookupValChecked valMap deqMap xId ctx
      let xTy ← lookupLogTy logTyMap xId ctx
      if xTy.shape.size != 4 && xTy.shape.size != 2 then
        throw s!"{ctx}: conv requires 4D [1,C,H,W] or 2D [1,C] input, got rank {xTy.shape.size} ({xTy.shape.toList})"
      let xSh4 ← logicalToShape4D xTy.shape ctx
      let cin := xSh4.c; let h := xSh4.h; let w := xSh4.w

      let bBytes? ←
        if hasBias then
          if nd.inputs.size < 3 then
            throw s!"{ctx}: conv hasBias=true but missing bias input"
          let bId := nd.inputs[2]!
          let some (_, bPay) := lookupConst constMap bId
            | throw s!"{ctx}: conv bias %{bId} must be a constant"
          let bb ← bPay.toFp16Bytes s!"{baseName}_bias"
          weightCount := weightCount + 1
          weightBytes := weightBytes + bb.size
          pure (some bb)
        else
          pure none

      let (wTy, weightData) ←
        match lookupDeq deqMap wId with
        | some (_, _, qId) =>
          let some (qTy, .quant qSpec) := lookupConst constMap qId
            | throw s!"{ctx}: dequantize input %{qId} must be a quantized constant"
          if qTy.shape.size != 4 then
            throw s!"{ctx}: quantized weight must be 4D, got {qTy.shape.toList}"
          if qTy.shape[2]! != kh || qTy.shape[3]! != kw then
            throw s!"{ctx}: quantized conv weight kernel dims [{qTy.shape[2]!},{qTy.shape[3]!}] != expected [{kh},{kw}]"
          let cout := qTy.shape[0]!
          let cinW := qTy.shape[1]!
          if groups == 1 && sh == 1 && sw == 1 && ((kh == 1 && kw == 1) || (kh == 3 && kw == 3)) &&
             (act == .none || act == .relu) then
            validateQuantWeightSpec qSpec cout cinW (kh * kw)
            weightCount := weightCount + 1
            weightBytes := weightBytes + qSpec.codes.size
            pure (qTy, WeightData.quant qSpec)
          else
            let bs ← dequantizeSpecToFp16 qSpec cout (cinW * kh * kw)
            weightCount := weightCount + 1
            weightBytes := weightBytes + bs.size
            pure (qTy, WeightData.fp16 bs)
        | none =>
          let some (wTy, wPay) := lookupConst constMap wId
            | throw s!"{ctx}: conv weight %{wId} must be a constant or dequantize(const)"
          let bs ← wPay.toFp16Bytes s!"{baseName}_weights"
          weightCount := weightCount + 1
          weightBytes := weightBytes + bs.size
          pure (wTy, WeightData.fp16 bs)

      if wTy.shape.size != 4 then
        throw s!"{ctx}: conv weight must be 4D [Cout,Cin/G,kH,kW], got {wTy.shape.toList}"
      if wTy.shape[2]! != kh || wTy.shape[3]! != kw then
        throw s!"{ctx}: conv weight kernel dims [{wTy.shape[2]!},{wTy.shape[3]!}] != expected [{kh},{kw}]"
      let cout := wTy.shape[0]!
      let cinPerG := wTy.shape[1]!
      if groups == 0 || cin % groups != 0 || cout % groups != 0 || cin / groups != cinPerG then
        throw s!"{ctx}: incompatible Conv channels cin={cin} cout={cout} group={groups} weight_cin_per_g={cinPerG}"

      let wgId := nextWgId; nextWgId := nextWgId + 1
      if kh == 1 && kw == 1 then
        if sh != 1 || sw != 1 then
          throw s!"{ctx}: conv1x1 requires strides [1,1], got [{sh},{sw}]"
        if groups != 1 then
          throw s!"{ctx}: conv1x1 requires group=1, got {groups}"
        if pads != #[0, 0, 0, 0] then
          throw s!"{ctx}: conv1x1 requires pads [0,0,0,0], got {pads.toList}"
        let outSh : Shape4D := { n := 1, c := cout, h := h, w := w }
        let lbl := if xTy.shape.size == 2 then "linear" else "conv1x1"
        let (aOut, nds', ms', shM', nid') := pushAneNode aneNodes nodeMeta sh4Map nextId
          baseName baseName (.neConv 1 1 1 1 .same .dense cin cout weightData bBytes? act none wgId lbl) #[xAVid] outSh
        aneNodes := nds'; nodeMeta := ms'; sh4Map := shM'; nextId := nid'
        valMap := valMap.push (outVid, aOut)
        logTyMap := logTyMap.push (outVid, outTy)
      else if kh == 3 && kw == 3 then
        if sh != sw || (sh != 1 && sh != 2) then
          throw s!"{ctx}: conv3x3 requires symmetric stride 1 or 2, got [{sh},{sw}]"
        let mode ←
          if groups == 1 then pure ConvMode.dense
          else if groups == cin && cout == cin then pure ConvMode.depthwise
          else throw s!"{ctx}: grouped conv3x3 (group={groups}, cin={cin}, cout={cout}) is not supported"
        let (padMode, ho, wo) ← resolveConv3x3PadMode h w sh pads ctx
        let outSh : Shape4D := { n := 1, c := cout, h := ho, w := wo }
        let (aOut, nds', ms', shM', nid') := pushAneNode aneNodes nodeMeta sh4Map nextId
          baseName baseName (.neConv 3 3 sh sw padMode mode cin cout weightData bBytes? act none wgId "conv3x3") #[xAVid] outSh
        aneNodes := nds'; nodeMeta := ms'; sh4Map := shM'; nextId := nid'
        valMap := valMap.push (outVid, aOut)
        logTyMap := logTyMap.push (outVid, outTy)
      else
        throw s!"{ctx}: unsupported Conv kernel size {kh}x{kw} (supported: 1x1, 3x3)"

    | .matmul trA trB scaleBits =>
      if nd.inputs.size != 2 || nd.outputs.size != 1 then
        throw s!"{ctx}: matmul expects 2 inputs and 1 output"
      let aId := nd.inputs[0]!
      let bId := nd.inputs[1]!
      let (outVid, outTy) := nd.outputs[0]!
      if (lookupConst constMap aId).isSome then
        throw s!"{ctx}: constant LHS with dynamic RHS in MatMul is not supported"
      let aAVid ← lookupValChecked valMap deqMap aId ctx
      let bAVid ← lookupValChecked valMap deqMap bId ctx
      let aTy ← lookupLogTy logTyMap aId ctx
      let bTy ← lookupLogTy logTyMap bId ctx
      if !f16IsFinite scaleBits || (scaleBits &&& 0x7fff) == 0 then
        throw s!"{ctx}: dynamic MatMul scale must be finite and non-zero in fp16"

      let emitMatmulCore := fun (nds : Array (IRNode AneOp)) (ms : Array AneNodeMeta)
          (shM : Array (ValueId × Shape4D)) (nid : ValueId) (uNames : Array String)
          (mmBase : String) (heads m k n : Nat) (effA effB : ValueId) => do
        let mut ndsC := nds
        let mut msC := ms
        let mut shMC := shM
        let mut nidC := nid
        let mut uC := uNames
        let mut curA := effA
        let mut curB := effB
        if trA then
          let (cpA, szA) ← planMatmulScratchTranspose heads k m
          let (trName, u1) := sanitizeName s!"{mmBase}_trA" uC; uC := u1
          let trInSh : Shape4D := { n := 1, c := heads * k, h := 1, w := m }
          let trOutSh : Shape4D := { n := 1, c := heads * m, h := 1, w := k }
          let desc : DmaViewSpec := {
            inShape := trInSh, outShape := trOutSh, transposeCW := cpA.transposeCW, batchN := heads,
            minDramBytes := szA, explicitCopy? := some cpA
          }
          let (v, n1, m1, s1, id1) := pushAneNode ndsC msC shMC nidC trName trName
            (.dmaView desc "matmul_trA") #[curA] trOutSh (some .dram)
          ndsC := n1; msC := m1; shMC := s1; nidC := id1; curA := v
        if trB then
          let (cpB, szB) ← planMatmulScratchTranspose heads n k
          let (trName, u1) := sanitizeName s!"{mmBase}_trB" uC; uC := u1
          let trInSh : Shape4D := { n := 1, c := heads * n, h := 1, w := k }
          let trOutSh : Shape4D := { n := 1, c := heads * k, h := 1, w := n }
          let desc : DmaViewSpec := {
            inShape := trInSh, outShape := trOutSh, transposeCW := cpB.transposeCW, batchN := heads,
            minDramBytes := szB, explicitCopy? := some cpB
          }
          let (v, n1, m1, s1, id1) := pushAneNode ndsC msC shMC nidC trName trName
            (.dmaView desc "matmul_trB") #[curB] trOutSh (some .dram)
          ndsC := n1; msC := m1; shMC := s1; nidC := id1; curB := v
        let fullOut := matmulStackedShape heads m n
        let numSlices := ceilDiv m 16
        if numSlices == 1 then
          let (v, n1, m1, s1, id1) := pushAneNode ndsC msC shMC nidC mmBase mmBase
            (.neMatmul heads m k n false false scaleBits 0 m) #[curA, curB] fullOut
          pure (v, n1, m1, s1, id1, uC)
        else
          let mut sliceVids : Array ValueId := Array.mkEmpty numSlices
          let mut sliceShapes : Array Shape4D := Array.mkEmpty numSlices
          for si in [:numSlices] do
            let m0 := si * 16
            let mi := min 16 (m - m0)
            let slSh := matmulStackedShape heads mi n
            let slNm := s!"{mmBase}_s{si}"
            let (sv, n1, m1, s1, id1) := pushAneNode ndsC msC shMC nidC slNm mmBase
              (.neMatmul heads mi k n false false scaleBits m0 m) #[curA, curB] slSh
            ndsC := n1; msC := m1; shMC := s1; nidC := id1
            sliceVids := sliceVids.push sv
            sliceShapes := sliceShapes.push slSh
          let concatDesc := dmaViewSpecFromConcat sliceShapes 1 fullOut (aliasable := true)
          let (v, n1, m1, s1, id1) := pushAneNode ndsC msC shMC nidC s!"{mmBase}_join" mmBase
            (.dmaView concatDesc "matmul_join") sliceVids fullOut
          pure (v, n1, m1, s1, id1, uC)

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
        let (aOut, n1, m1, s1, id1, u1) ← emitMatmulCore aneNodes nodeMeta sh4Map nextId usedNames baseName heads m ka n aAVid bAVid
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; usedNames := u1
        valMap := valMap.push (outVid, aOut)
        logTyMap := logTyMap.push (outVid, { shape := #[heads, m, n], dtype := outTy.dtype })
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
        if (needPackA || needPackB || needUnpack) && !opts.allowViewOps then
          throw s!"{ctx}: 4D MatMul [1,{heads},{rA},{cA}] x [1,{heads},{rB},{cB}] requires view reshape ops (--allow-view-ops=true)"

        let mut effA := aAVid
        if needPackA then
          let stA := matmulStackedShape heads rA cA
          let vp ← planReshape aSh stA
          let (pA, u1) := sanitizeName s!"{baseName}_packA" usedNames; usedNames := u1
          let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId pA pA
            (.dmaView (dmaViewSpecFromPlan aSh vp) "reshape") #[aAVid] stA
          aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; effA := v

        let mut effB := bAVid
        if needPackB then
          let stB := matmulStackedShape heads rB cB
          let vp ← planReshape bSh stB
          let (pB, u2) := sanitizeName s!"{baseName}_packB" usedNames; usedNames := u2
          let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId pB pB
            (.dmaView (dmaViewSpecFromPlan bSh vp) "reshape") #[bAVid] stB
          aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; effB := v

        let stackedOut := matmulStackedShape heads m n
        let out4D := matmul4DShape heads m n
        if needUnpack then
          let vp ← planReshape stackedOut out4D
          let (mmCore, u3) := sanitizeName s!"{baseName}_mm" usedNames; usedNames := u3
          let (mmVid, n1, m1, s1, id1, u4) ← emitMatmulCore aneNodes nodeMeta sh4Map nextId usedNames mmCore heads m ka n effA effB
          aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; usedNames := u4
          let (aOut, n2, m2, s2, id2) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName
            (.dmaView (dmaViewSpecFromPlan stackedOut vp) "reshape") #[mmVid] out4D
          aneNodes := n2; nodeMeta := m2; sh4Map := s2; nextId := id2
          valMap := valMap.push (outVid, aOut)
        else
          let (aOut, n1, m1, s1, id1, u4) ← emitMatmulCore aneNodes nodeMeta sh4Map nextId usedNames baseName heads m ka n effA effB
          aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; usedNames := u4
          valMap := valMap.push (outVid, aOut)
        logTyMap := logTyMap.push (outVid, { shape := #[1, heads, m, n], dtype := outTy.dtype })
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
        let (aOut, n1, m1, s1, id1, u4) ← emitMatmulCore aneNodes nodeMeta sh4Map nextId usedNames baseName heads m k n aAVid bAVid
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; usedNames := u4
        valMap := valMap.push (outVid, aOut)
        logTyMap := logTyMap.push (outVid, { shape := #[1, outVane.c, 1, outVane.w], dtype := outTy.dtype })

    | .elemUnary uop =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: elemUnary expects 1 input and 1 output"
      let xAVid ← lookupValChecked valMap deqMap nd.inputs[0]! ctx
      let (outVid, outTy) := nd.outputs[0]!
      let outSh ← logicalToShape4D outTy.shape ctx
      match uop with
      | .abs =>
        let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.pe (.unary .abs)) #[xAVid] outSh
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v)
      | .neg =>
        let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.pe (.unary (.scalarMul f16NegOne))) #[xAVid] outSh
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v)
      | .square =>
        let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.pe (.unary .square)) #[xAVid] outSh
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v)
      | .log =>
        let (log2Nm, u1) := sanitizeName s!"{baseName}_log2" usedNames; usedNames := u1
        let wg := nextWgId; nextWgId := nextWgId + 1
        let ln2Bits := floatToF16CanonicalZero ln2Const
        let (v0, n0, m0, s0, id0) := pushAneNode aneNodes nodeMeta sh4Map nextId log2Nm log2Nm (.neAct (.act (.lut .log2)) none wg) #[xAVid] outSh
        let (v1, n1, m1, s1, id1) := pushAneNode n0 m0 s0 id0 baseName baseName (.pe (.unary (.scalarMul ln2Bits))) #[v0] outSh
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v1)
      | .softplus =>
        let (expNm, u1) := sanitizeName s!"{baseName}_exp" usedNames
        let (addNm, u2) := sanitizeName s!"{baseName}_add1" u1
        let (log2Nm, u3) := sanitizeName s!"{baseName}_log2" u2
        usedNames := u3
        let wg1 := nextWgId; let wg2 := nextWgId + 1; nextWgId := nextWgId + 2
        let ln2Bits := floatToF16CanonicalZero ln2Const
        let (v0, n0, m0, s0, id0) := pushAneNode aneNodes nodeMeta sh4Map nextId expNm expNm (.neAct (.act (.lut .exp)) none wg1) #[xAVid] outSh
        let (v1, n1, m1, s1, id1) := pushAneNode n0 m0 s0 id0 addNm addNm (.pe (.unary (.scalarAdd f16PosOne))) #[v0] outSh
        let (v2, n2, m2, s2, id2) := pushAneNode n1 m1 s1 id1 log2Nm log2Nm (.neAct (.act (.lut .log2)) none wg2) #[v1] outSh
        let (v3, n3, m3, s3, id3) := pushAneNode n2 m2 s2 id2 baseName baseName (.pe (.unary (.scalarMul ln2Bits))) #[v2] outSh
        aneNodes := n3; nodeMeta := m3; sh4Map := s3; nextId := id3; valMap := valMap.push (outVid, v3)
      | .scalarAdd b =>
        let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.pe (.unary (.scalarAdd b))) #[xAVid] outSh
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v)
      | .scalarSub b =>
        let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.pe (.unary (.scalarSub b))) #[xAVid] outSh
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v)
      | .scalarRSub b =>
        let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.pe (.unary (.scalarRSub b))) #[xAVid] outSh
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v)
      | .scalarMul b =>
        let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.pe (.unary (.scalarMul b))) #[xAVid] outSh
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v)
      | .scalarDiv b =>
        let peOp ← PeOp.scalarDiv b
        let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.pe (.unary peOp)) #[xAVid] outSh
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v)
      | .linearAct a b =>
        let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.pe (.unary (.linearActivation a b))) #[xAVid] outSh
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v)
      logTyMap := logTyMap.push (outVid, outTy)

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
        let scBits ← validateFiniteFp16Scalar vals[0]! ctx
        let xAVid ← lookupValChecked valMap deqMap tensorVid ctx
        let xTy ← lookupLogTy logTyMap tensorVid ctx
        let xSh4 ← logicalToShape4D xTy.shape ctx
        let outSh4 ← logicalToShape4D outTy.shape ctx
        if outSh4 != xSh4 then
          throw s!"{ctx}: scalar-constant {bop.name} changes physical layout ({xSh4.toArray.toList} -> {outSh4.toArray.toList}; rank-increasing broadcast is not supported)"
        match bop with
        | .add =>
          let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.pe (.unary (.scalarAdd scBits))) #[xAVid] outSh4
          aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v)
        | .mul =>
          let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.pe (.unary (.scalarMul scBits))) #[xAVid] outSh4
          aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v)
        | .sub =>
          let peOp := if constOnLeft then PeOp.scalarRSub scBits else PeOp.scalarSub scBits
          let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.pe (.unary peOp)) #[xAVid] outSh4
          aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v)
        | .div =>
          if !constOnLeft then
            let peOp ← PeOp.scalarDiv scBits
            let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.pe (.unary peOp)) #[xAVid] outSh4
            aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v)
          else
            let (invNm, u1) := sanitizeName s!"{baseName}_inv" usedNames; usedNames := u1
            let wg := nextWgId; nextWgId := nextWgId + 1
            let (v0, n0, m0, s0, id0) := pushAneNode aneNodes nodeMeta sh4Map nextId invNm invNm (.neAct (.act (.lut .inverse)) none wg) #[xAVid] outSh4
            let (v1, n1, m1, s1, id1) := pushAneNode n0 m0 s0 id0 baseName baseName (.pe (.unary (.scalarMul scBits))) #[v0] outSh4
            aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v1)
        | _ => pure ()
        logTyMap := logTyMap.push (outVid, outTy)
      else
        if aConst?.isSome || bConst?.isSome then
          throw s!"{ctx}: non-scalar constant operand is not supported for {bop.name}"
        let aAVid ← lookupValChecked valMap deqMap aId ctx
        let bAVid ← lookupValChecked valMap deqMap bId ctx
        let aTy ← lookupLogTy logTyMap aId ctx
        let bTy ← lookupLogTy logTyMap bId ctx
        let (outLog, outSh4) ← resolveBroadcast aTy.shape bTy.shape ctx
        if bop == .div then
          let (invNm, u1) := sanitizeName s!"{baseName}_inv" usedNames; usedNames := u1
          let wg := nextWgId; nextWgId := nextWgId + 1
          let bSh4 ← logicalToShape4D bTy.shape ctx
          let (v0, n0, m0, s0, id0) := pushAneNode aneNodes nodeMeta sh4Map nextId invNm invNm (.neAct (.act (.lut .inverse)) none wg) #[bAVid] bSh4
          let (v1, n1, m1, s1, id1) := pushAneNode n0 m0 s0 id0 baseName baseName (.pe (.binary .mul false)) #[aAVid, v0] outSh4
          aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v1)
        else
          let peBin : PeBinOp := match bop with
            | .add => .add
            | .sub => .sub
            | .mul => .mul
            | .max => .max
            | _ => .min
          let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.pe (.binary peBin false)) #[aAVid, bAVid] outSh4
          aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1; valMap := valMap.push (outVid, v)
        logTyMap := logTyMap.push (outVid, { shape := outLog, dtype := outTy.dtype })

    | .lut act =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: lut expects 1 input and 1 output"
      let xAVid ← lookupValChecked valMap deqMap nd.inputs[0]! ctx
      let (outVid, outTy) := nd.outputs[0]!
      let outSh ← logicalToShape4D outTy.shape ctx
      let wg := nextWgId; nextWgId := nextWgId + 1
      let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName (.neAct (.act act) none wg) #[xAVid] outSh
      aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1
      valMap := valMap.push (outVid, v)
      logTyMap := logTyMap.push (outVid, outTy)

    | .pool kind kh kw sh sw pad excl =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: pool expects 1 input and 1 output"
      let xId := nd.inputs[0]!
      let xAVid ← lookupValChecked valMap deqMap xId ctx
      let xTy ← lookupLogTy logTyMap xId ctx
      if xTy.shape.size != 4 then
        throw s!"{ctx}: pool requires 4D [1,C,H,W] input, got rank {xTy.shape.size}"
      let xSh ← logicalToShape4D xTy.shape ctx
      let pspec : PoolSpec := {
        target := g.target, op := kind, dims := shape4DToDims xSh,
        kh := kh, kw := kw, sh := sh, sw := sw, pad := pad, excludePad := excl
      }
      let (backend, gh, gw) ← resolvePoolBackend pspec
      let outSh : Shape4D := { n := 1, c := xSh.c, h := gh.out, w := gw.out }
      let (outVid, outTy) := nd.outputs[0]!
      let aneOp : AneOp := match backend with
        | .pePool => .pe (.pool kind kh kw sh sw pad excl)
        | .nePool => .neAct (.nePool kind kh kw sh sw pad excl)
        | .peReduceGlobal =>
          let redOp : ReduceOp := match kind with | .max => .max | .avg => .mean
          .pe (.reduce redOp { reduceH := true, reduceW := true } 0)
      let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName aneOp #[xAVid] outSh
      aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1
      valMap := valMap.push (outVid, v)
      logTyMap := logTyMap.push (outVid, outTy)

    | .reduce op axes keepDims eps =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: reduce expects 1 input and 1 output"
      if !keepDims then
        throw s!"{ctx}: keepdims=0 is not supported (Vane reduce preserves 4D rank with keepdims=1)"
      let xId := nd.inputs[0]!
      let xAVid ← lookupValChecked valMap deqMap xId ctx
      let xTy ← lookupLogTy logTyMap xId ctx
      let xSh ← logicalToShape4D xTy.shape ctx
      let rank := xTy.shape.size
      let mut rc := false; let mut rh := false; let mut rw := false
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
      let rspec : ReduceSpec := {
        target := g.target, op := op, dims := shape4DToDims xSh, axes := rAxes, eps := eps
      }
      let (backend, od) ← resolveReduceBackend rspec
      let outSh : Shape4D := { n := 1, c := od.c, h := od.h, w := od.w }
      let (outVid, outTy) := nd.outputs[0]!
      let aneOp : AneOp := match backend with
        | .peReduce => .pe (.reduce op rAxes eps)
        | .neChannelReduce => .neAct (.channelReduce op)
      let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName aneOp #[xAVid] outSh
      aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1
      valMap := valMap.push (outVid, v)
      logTyMap := logTyMap.push (outVid, outTy)

    | .reshape targetShape =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: reshape expects 1 input and 1 output"
      let xId := nd.inputs[0]!
      let xAVid ← lookupValChecked valMap deqMap xId ctx
      let xTy ← lookupLogTy logTyMap xId ctx
      let inSh4 ← logicalToShape4D xTy.shape ctx
      let outSh4 ← logicalToShape4D targetShape ctx
      let (outVid, outTy) := nd.outputs[0]!
      if outSh4 == inSh4 then
        -- Zero-copy physical no-op
        valMap := valMap.push (outVid, xAVid)
        logTyMap := logTyMap.push (outVid, outTy)
      else
        if !opts.allowViewOps then
          throw s!"{ctx}: non-no-op view op Reshape ({inSh4.toArray.toList} -> {outSh4.toArray.toList}) requires --allow-view-ops=true"
        let vp ← planReshape inSh4 outSh4
        let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName
          (.dmaView (dmaViewSpecFromPlan inSh4 vp) "reshape") #[xAVid] outSh4
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1
        valMap := valMap.push (outVid, v)
        logTyMap := logTyMap.push (outVid, outTy)

    | .transpose perm =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: transpose expects 1 input and 1 output"
      let xId := nd.inputs[0]!
      let xAVid ← lookupValChecked valMap deqMap xId ctx
      let xTy ← lookupLogTy logTyMap xId ctx
      let inSh4 ← logicalToShape4D xTy.shape ctx
      let (outVid, outTy) := nd.outputs[0]!
      if isIdentityPerm perm then
        valMap := valMap.push (outVid, xAVid)
        logTyMap := logTyMap.push (outVid, outTy)
      else if xTy.shape.size == 3 then
        if perm == #[0, 2, 1] then
          if !opts.allowViewOps then
            throw s!"{ctx}: non-no-op view op Transpose (perm={perm.toList}) requires --allow-view-ops=true"
          let h0 := xTy.shape[0]!; let r0 := xTy.shape[1]!; let c0 := xTy.shape[2]!
          let outVane := matmulStackedShape h0 c0 r0
          if h0 == 1 then
            let vp ← planTranspose inSh4 #[0, 3, 2, 1]
            let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName
              (.dmaView (dmaViewSpecFromPlan inSh4 vp) "transpose") #[xAVid] vp.outShape
            aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1
            valMap := valMap.push (outVid, v)
          else
            let sh4In := matmul4DShape h0 r0 c0
            let sh4Out := matmul4DShape h0 c0 r0
            let vp1 ← planReshape inSh4 sh4In
            let vp2 ← planTranspose sh4In #[0, 1, 3, 2]
            let vp3 ← planReshape sh4Out outVane
            let (uNm, u1) := sanitizeName s!"{baseName}_unpack" usedNames
            let (tNm, u2) := sanitizeName s!"{baseName}_tr" u1
            usedNames := u2
            let (v1, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId uNm uNm
              (.dmaView (dmaViewSpecFromPlan inSh4 vp1) "reshape") #[xAVid] sh4In
            let (v2, n2, m2, s2, id2) := pushAneNode n1 m1 s1 id1 tNm tNm
              (.dmaView (dmaViewSpecFromPlan sh4In vp2) "transpose") #[v1] sh4Out
            let (v3, n3, m3, s3, id3) := pushAneNode n2 m2 s2 id2 baseName baseName
              (.dmaView (dmaViewSpecFromPlan sh4Out vp3) "reshape") #[v2] outVane
            aneNodes := n3; nodeMeta := m3; sh4Map := s3; nextId := id3
            valMap := valMap.push (outVid, v3)
          logTyMap := logTyMap.push (outVid, outTy)
        else
          throw s!"{ctx}: unsupported 3D Transpose perm {perm.toList} (supported: [0,1,2] or [0,2,1])"
      else
        if perm.size != 4 then
          throw s!"{ctx}: perm must have length 4, got {perm.toList}"
        if !opts.allowViewOps then
          throw s!"{ctx}: non-no-op view op Transpose (perm={perm.toList}) requires --allow-view-ops=true"
        let vp ← planTranspose inSh4 perm
        if vp.outShape.n != 1 then
          throw s!"{ctx}: Transpose output batch dimension must be 1, got {vp.outShape.n}"
        let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName
          (.dmaView (dmaViewSpecFromPlan inSh4 vp) "transpose") #[xAVid] vp.outShape
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1
        valMap := valMap.push (outVid, v)
        logTyMap := logTyMap.push (outVid, outTy)

    | .slice beg en st =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"{ctx}: slice expects 1 input and 1 output"
      let xId := nd.inputs[0]!
      let xAVid ← lookupValChecked valMap deqMap xId ctx
      let xTy ← lookupLogTy logTyMap xId ctx
      let inSh4 ← logicalToShape4D xTy.shape ctx
      let rank := xTy.shape.size
      let (outVid, outTy) := nd.outputs[0]!
      if beg.size != rank || en.size != rank || st.size != rank then
        throw s!"{ctx}: slice params length mismatch (beg={beg.size}, end={en.size}, step={st.size}) != input rank {rank}"
      if beg == Array.replicate rank 0 && en == xTy.shape && st == Array.replicate rank 1 then
        valMap := valMap.push (outVid, xAVid)
        logTyMap := logTyMap.push (outVid, outTy)
      else
        if !opts.allowViewOps then
          throw s!"{ctx}: non-no-op view op Slice requires --allow-view-ops=true"
        let (beg4, en4, st4) ←
          if rank == 4 then pure (beg, en, st)
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
        -- Contiguous C-axis slice (only axis 1 sliced, step=1) is eligible for zero-copy subview aliasing!
        let isContigCSlice :=
          st4 == #[1, 1, 1, 1] &&
          beg4[0]! == 0 && en4[0]! == inSh4.n &&
          beg4[2]! == 0 && en4[2]! == inSh4.h &&
          beg4[3]! == 0 && en4[3]! == inSh4.w
        let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName
          (.dmaView (dmaViewSpecFromPlan inSh4 vp (aliasable := isContigCSlice)) "slice") #[xAVid] vp.outShape
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1
        valMap := valMap.push (outVid, v)
        logTyMap := logTyMap.push (outVid, outTy)

    | .concat axis =>
      if nd.inputs.size < 2 || nd.outputs.size != 1 then
        throw s!"{ctx}: concat expects >= 2 inputs and 1 output"
      let mut inAVids : Array ValueId := Array.mkEmpty nd.inputs.size
      let mut inTys : Array TensorType := Array.mkEmpty nd.inputs.size
      let mut inSh4s : Array Shape4D := Array.mkEmpty nd.inputs.size
      for inpVid in nd.inputs do
        let av ← lookupValChecked valMap deqMap inpVid ctx
        let ty ← lookupLogTy logTyMap inpVid ctx
        let sh4 ← logicalToShape4D ty.shape ctx
        inAVids := inAVids.push av
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
      let vp ← planConcat inSh4s vaneAxis
      if !opts.allowViewOps then
        throw s!"{ctx}: non-no-op view op Concat (axis={vaneAxis}) requires --allow-view-ops=true"
      let (outVid, outTy) := nd.outputs[0]!
      let desc := dmaViewSpecFromConcat inSh4s vaneAxis vp.outShape (aliasable := vaneAxis == 1)
      let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId baseName baseName
        (.dmaView desc "concat") inAVids vp.outShape
      aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1
      valMap := valMap.push (outVid, v)
      logTyMap := logTyMap.push (outVid, outTy)

  -- Resolve ALL graph outputs (multi-output!)
  let inputAVids := aneInputs.map (·.1)
  let mut aneOutputs : Array ValueId := Array.mkEmpty g.outputs.size
  let mut outNames : Array String := Array.mkEmpty g.outputs.size
  let mut outLogShapes : Array (Array Nat) := Array.mkEmpty g.outputs.size
  let mut seenOutVids : Array ValueId := #[]

  for oi in [:g.outputs.size] do
    let outVid := g.outputs[oi]!
    let aVid0 ← lookupValChecked valMap deqMap outVid s!"graph output[{oi}]"
    let outTy ← lookupLogTy logTyMap outVid s!"graph output[{oi}]"
    let outSh ← logicalToShape4D outTy.shape s!"graph output[{oi}]"
    let outNm := opts.outputNames.getD oi (if g.outputs.size == 1 then "y" else s!"y{oi}")
    -- For single-output graphs where the output aliases a graph input, insert an explicit `neAct .copy`.
    -- For multi-output graphs (`g.outputs.size >= 2`), L3 `MemoryPlan` packs all logical outputs into
    -- the single ABI v3 HWX output surface (emitting zero-copy writes or `pack_out` DMA stores as needed).
    let needsCopy := g.outputs.size == 1 && inputAVids.contains aVid0
    let finalVid ←
      if needsCopy then
        let (copyNm, u1) := sanitizeName s!"{outNm}_out" usedNames; usedNames := u1
        let (v, n1, m1, s1, id1) := pushAneNode aneNodes nodeMeta sh4Map nextId copyNm copyNm
          (.neAct (.act .copy)) #[aVid0] outSh
        aneNodes := n1; nodeMeta := m1; sh4Map := s1; nextId := id1
        pure v
      else
        pure aVid0
    seenOutVids := seenOutVids.push finalVid
    aneOutputs := aneOutputs.push finalVid
    outNames := outNames.push outNm
    outLogShapes := outLogShapes.push outTy.shape

  -- Prune any dead AneOp nodes (e.g. from unused splits/branches) via generic DCE on `SSAGraph AneOp`
  let rawGraph : SSAGraph AneOp := {
    target := g.target
    inputs := aneInputs
    nodes := aneNodes
    outputs := aneOutputs
  }
  -- Keep nodeMeta aligned with kept nodes
  let mut live : Array Bool := Array.replicate (nextId + 1) false
  for ov in rawGraph.outputs do
    if ov < live.size then live := live.set! ov true
  for step in [:rawGraph.nodes.size] do
    let idx := rawGraph.nodes.size - 1 - step
    let nd := rawGraph.nodes[idx]!
    if nd.outputs.any (fun (v, _) => v < live.size && live[v]!) then
      for iv in nd.inputs do
        if iv < live.size then live := live.set! iv true
  let mut prunedNodes : Array (IRNode AneOp) := #[]
  let mut prunedMeta : Array AneNodeMeta := #[]
  for idx in [:rawGraph.nodes.size] do
    let nd := rawGraph.nodes[idx]!
    if nd.outputs.any (fun (v, _) => v < live.size && live[v]!) then
      prunedNodes := prunedNodes.push nd
      prunedMeta := prunedMeta.push (nodeMeta.getD idx {})

  pure {
    graph := { rawGraph with nodes := prunedNodes }
    outputNames := outNames
    defaultPlacement := opts.placement
    nodeMeta := prunedMeta
    donations := opts.donations
    weightCount := weightCount
    weightBytes := weightBytes
    outputLogicalShapes := outLogShapes
  }

end Vane.Backend.LayoutSelect
