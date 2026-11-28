import Lean.Data.Json
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
import Vane.Graph
import Vane.Frontend.SafeTensors
import Vane.Frontend.Onnx
import Vane.IR.ToGraph
import Vane.Ane.Types
import Vane.Machine.Types
import Vane.Backend.LayoutSelect
import Vane.Backend.SyncAndEmit

namespace Vane.Frontend.Lower

open Lean (Json)
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
open Vane.Frontend.SafeTensors
open Vane.Frontend.Onnx
open Vane.IR
open Vane.IR.ToGraph
open Vane.Ane
open Vane.Machine
open Vane.Backend.SyncAndEmit

-- Re-export pure classifiers & helpers used by existing proofs and tests
export Vane.IR (
  PowKind powKindToF64Bits classifyPowBits classifyPowExponent
  GeluApprox classifyGeluApprox
  isValidLayerNormParamShape
)
export Vane.IR.ToGraph (
  broadcastOnnxShapes packHeadRow unpackHeadRow matmul4DShape matmulStackedShape
)

/-- Compute the ONNX 2D output shape `(d0, d1)` of `Flatten(dims, axis=ax0)`. -/
def computeFlattenShape (dims : Array Nat) (ax0 : Int) : Except String (Nat × Nat) := do
  let rank := dims.size
  let axInt := if ax0 < 0 then ax0 + (rank : Int) else ax0
  if axInt < 0 || axInt > (rank : Int) then
    throw s!"Flatten axis {ax0} out of range [{-(rank : Int)},{rank}]"
  let ax := axInt.toNat
  let mut d0 : Nat := 1
  for i in [:ax] do
    d0 := d0 * dims[i]!
  let mut d1 : Nat := 1
  for i in [ax:rank] do
    d1 := d1 * dims[i]!
  pure (d0, d1)

/-- Compute default ONNX `Split` partition sizes for `dim` across `numOutputs` outputs
    (equal chunks when divisible, otherwise opset-18 `ceilDiv dim numOutputs` chunks with positive remainder). -/
def computeDefaultSplitSizes (dim numOutputs : Nat) : Except String (List Nat) :=
  if numOutputs == 0 then
    .error "Split: num_outputs must be >= 1"
  else if dim < numOutputs then
    .error s!"Split: cannot split dimension {dim} into {numOutputs} non-empty outputs"
  else if dim % numOutputs == 0 then
    .ok (List.replicate numOutputs (dim / numOutputs))
  else
    let chunk := ceilDiv dim numOutputs
    let prefixSum := chunk * (numOutputs - 1)
    if prefixSum < dim then
      .ok (List.replicate (numOutputs - 1) chunk ++ [dim - prefixSum])
    else
      .error s!"Split: cannot split dimension {dim} into {numOutputs} non-empty chunks using chunk size {chunk}"

-- ============================================================================
-- Public Lowering Options, Summary, Result & CLI Input-Shape Parser
-- ============================================================================

/-- Configuration options for lowering an ONNX model through L1 `TensorIR` into `Vane.Graph.GraphSpec`. -/
structure LowerOptions where
  target : Target := .h17
  placement : EdgePlacementMode := .auto
  stripMode : StripMode := .off
  inputShapes : Array (String × TensorDims) := #[]
  safeTensors : Option SafeTensorsFile := none
  noFuse : Bool := false
  /-- Enable emitting non-no-op M5C view ops (`transpose`, `slice`, `concat`, `reshape`, `split`).
      Enabled by default; pass `--allow-view-ops=false` to opt out. -/
  allowViewOps : Bool := true
  /-- Optional post-training weight quantization format (`int8`, `int4`, `pal4`, `pal2`). -/
  quantize : Option QuantFormat := none
  deriving BEq, Inhabited

/-- Summary statistics and diagnostics produced by `lowerOnnx`. -/
structure LowerSummary where
  target : Target
  placement : EdgePlacementMode
  inputs : Array GraphInput
  output : String
  outputShape : Shape4D
  outputs : Array String := #[]
  outputShapes : Array Shape4D := #[]
  onnxNodeCount : Nat
  nodeCount : Nat
  fusedBnCount : Nat
  fusedActCount : Nat
  fusedMatmulCount : Nat
  elidedCount : Nat
  weightCount : Nat
  weightBytes : Nat
  safeTensorsUsed : Nat
  quantizedConvCount : Nat := 0
  warnings : Array String
  deriving Repr, BEq

/-- Result of pure Lean 4 ONNX lowering: `HighOp`/`CoreOp` SSA graphs, L2 `AneProgram`,
    backwards-compatible `GraphSpec`, and `LowerSummary`. -/
structure LowerResult where
  spec : GraphSpec
  summary : LowerSummary
  highGraph : SSAGraph HighOp
  coreGraph : SSAGraph CoreOp
  aneProgram : AneProgram
  deriving BEq

/-- Result of full end-to-end ONNX compilation (L1 `TensorIR` → L2 `AneIR` → L3 `MachineIR` → L4 `SyncAndEmit`). -/
structure CompiledOnnx where
  lowered : LowerResult
  compiled : CompiledGraph
  aneProgram : AneProgram
  machineProgram : MachineProgram

/-- Parse a CLI `--input-shape` override of the form `NAME=C,H,W` or `NAME=1,C,H,W`. -/
def parseInputShapeOverride (raw : String) : Except String (String × TensorDims) := do
  let parts := raw.splitOn "="
  if parts.length != 2 then
    throw s!"invalid --input-shape '{raw}' (expected NAME=C,H,W)"
  let name := parts[0]!.trimAscii.toString
  let rhs := parts[1]!.trimAscii.toString
  if name.isEmpty || rhs.isEmpty then
    throw s!"invalid --input-shape '{raw}' (expected NAME=C,H,W)"
  let dimStrs := rhs.splitOn ","
  let mut dims : Array Nat := #[]
  for ds in dimStrs do
    let t := ds.trimAscii.toString
    match t.toNat? with
    | some n =>
      if n == 0 then
        throw s!"invalid --input-shape '{raw}': dimension '{t}' must be a positive integer"
      dims := dims.push n
    | none =>
      throw s!"invalid --input-shape '{raw}': dimension '{t}' must be a positive integer"
  if dims.size == 3 then
    pure (name, { c := dims[0]!, h := dims[1]!, w := dims[2]! })
  else if dims.size == 4 then
    if dims[0]! != 1 then
      throw s!"invalid --input-shape '{raw}': batch dimension N must be 1, got {dims[0]!}"
    pure (name, { c := dims[1]!, h := dims[2]!, w := dims[3]! })
  else
    throw s!"invalid --input-shape '{raw}' (expected NAME=C,H,W or NAME=1,C,H,W)"

-- ============================================================================
-- Mechanical ONNX + SafeTensors -> `SSAGraph HighOp` Decoder
-- ============================================================================

private inductive RawConstPayload where
  | onnx (t : OnnxTensor)
  | safetensors (sf : SafeTensorsFile)
  | inlineF64 (vals : Array Float)
  | inlineI64 (vals : Array Int)
  deriving Inhabited

private structure RawConst where
  name : String
  dims : Array Nat
  payload : RawConstPayload
  deriving Inhabited

private def RawConst.isFromSafeTensors (c : RawConst) : Bool :=
  match c.payload with
  | .safetensors _ => true
  | _ => false

private def RawConst.float64s (c : RawConst) : Except String (Array Float) :=
  match c.payload with
  | .inlineF64 vals => pure vals
  | .safetensors sf => sf.toFloat64 c.name
  | .onnx t => t.toFloat64
  | .inlineI64 _ => throw s!"constant '{c.name}' has no float data"

private def RawConst.int64s (c : RawConst) : Except String (Array Int) :=
  match c.payload with
  | .inlineI64 vals => pure vals
  | .safetensors sf => sf.toInt64 c.name
  | .onnx t => t.toInt64
  | .inlineF64 _ => throw s!"constant '{c.name}' has no integer data"

private def RawConst.isScalar (c : RawConst) : Except String Bool := do
  let n ← shapeProductChecked c.dims s!"constant '{c.name}'"
  pure (n == 1)

private def lookupAssoc {α : Type} (arr : Array (String × α)) (key : String) : Option α :=
  (arr.find? fun p => p.1 == key).map Prod.snd

private def setAssoc {α : Type} (arr : Array (String × α)) (key : String) (val : α) : Array (String × α) :=
  match arr.findIdx? fun p => p.1 == key with
  | some idx => arr.set! idx (key, val)
  | none => arr.push (key, val)

private def resolveAliasStr (aliases : Array (String × String)) (name : String) : String := Id.run do
  let mut cur := name
  for _ in [:256] do
    match lookupAssoc aliases cur with
    | some nxt =>
      if nxt == cur then return cur
      cur := nxt
    | none => return cur
  return cur

private def extractConstantNode (nd : OnnxNode) : Except String RawConst := do
  if nd.outputs.size != 1 || nd.outputs[0]!.isEmpty then
    throw s!"Constant node '{nd.name}': expected 1 non-empty output"
  let outName := nd.outputs[0]!
  if let some t := nd.attrTensor? "value" then
    let tNamed := if t.name.isEmpty then { t with name := outName } else t
    let dims ← tNamed.dimsToNat
    return { name := outName, dims := dims, payload := .onnx tNamed }
  if let some a := nd.attr? "value_float" then
    return { name := outName, dims := #[], payload := .inlineF64 #[a.f] }
  if let some a := nd.attr? "value_int" then
    return { name := outName, dims := #[], payload := .inlineI64 #[a.i] }
  if let some fs := nd.attrFloats? "value_floats" then
    return { name := outName, dims := #[fs.size], payload := .inlineF64 fs }
  if let some is := nd.attrInts? "value_ints" then
    return { name := outName, dims := #[is.size], payload := .inlineI64 is }
  throw s!"Constant node '{nd.name}': missing supported value attribute"

private def validateFiniteFp16ScalarLower (v : Float) (ctx : String) : Except String UInt16 := do
  if v.isNaN || v.isInf then
    throw s!"{ctx}: scalar {v} must be finite"
  let h := floatToF16CanonicalZero v
  if !f16IsFinite h then
    throw s!"{ctx}: scalar {v} overflows finite fp16 range"
  pure h

private def readOptionalScalarParam
    (nd : OnnxNode) (inps : Array String) (consts : Array (String × RawConst))
    (inputIdx : Nat) (attrName ctx : String) : Except String (Option Float) := do
  if h : inputIdx < inps.size then
    let nm := inps[inputIdx]
    if !nm.isEmpty then
      let some cs := lookupAssoc consts nm
        | throw s!"{ctx}: input '{nm}' ({attrName}) must be a constant scalar"
      let vals ← cs.float64s
      if vals.size != 1 then
        throw s!"{ctx}: {attrName} must be scalar, got {vals.size} elements"
      return some vals[0]!
  match nd.attr? attrName with
  | some a => pure (some a.f)
  | none => pure none

private def computeConcreteConvPads
    (h w kh kw sh sw : Nat) (autoPad : String) (rawPads : Array Int) (ctx : String) :
    Except String (Array Nat × Nat × Nat) := do
  match autoPad with
  | "VALID" =>
    if h < kh || w < kw then
      throw s!"{ctx}: Conv VALID requires H>={kh} and W>={kw}, got ({h},{w})"
    pure (#[0, 0, 0, 0], (h - kh) / sh + 1, (w - kw) / sw + 1)
  | "SAME_UPPER" =>
    let ho := ceilDiv h sh
    let wo := ceilDiv w sw
    let padH := if (ho - 1) * sh + kh > h then (ho - 1) * sh + kh - h else 0
    let padW := if (wo - 1) * sw + kw > w then (wo - 1) * sw + kw - w else 0
    let pt := padH / 2
    let pl := padW / 2
    pure (#[pt, pl, padH - pt, padW - pl], ho, wo)
  | "SAME_LOWER" =>
    let ho := ceilDiv h sh
    let wo := ceilDiv w sw
    let padH := if (ho - 1) * sh + kh > h then (ho - 1) * sh + kh - h else 0
    let padW := if (wo - 1) * sw + kw > w then (wo - 1) * sw + kw - w else 0
    let pt := (padH + 1) / 2
    let pl := (padW + 1) / 2
    pure (#[pt, pl, padH - pt, padW - pl], ho, wo)
  | "" | "NOTSET" =>
    if rawPads.size != 4 then
      throw s!"{ctx}: Conv pads must have length 4, got {rawPads.size}"
    for p in rawPads do
      if p < 0 then
        throw s!"{ctx}: Conv negative pad {p}"
    let pt := rawPads[0]!.toNat; let pl := rawPads[1]!.toNat
    let pb := rawPads[2]!.toNat; let pr := rawPads[3]!.toNat
    if h + pt + pb < kh || w + pl + pr < kw then
      throw s!"{ctx}: Conv padded spatial dims ({h + pt + pb},{w + pl + pr}) < kernel ({kh},{kw})"
    pure (#[pt, pl, pb, pr], (h + pt + pb - kh) / sh + 1, (w + pl + pr - kw) / sw + 1)
  | other =>
    throw s!"{ctx}: unsupported auto_pad '{other}'"

private def resolveReshapeShapeIR (inDims : Array Nat) (raw : Array Int) (allowZero : Bool) (ctx : String) :
    Except String (Array Nat) := do
  if raw.isEmpty then
    throw s!"{ctx}: empty Reshape target shape"
  let total := inDims.foldl (· * ·) 1
  let mut resolved : Array Nat := Array.replicate raw.size 0
  let mut negIdx : Option Nat := none
  let mut knownProd : Nat := 1
  for i in [:raw.size] do
    let d0 := raw[i]!
    if d0 == -1 then
      if negIdx.isSome then
        throw s!"{ctx}: Reshape target {raw.toList} has multiple -1 dimensions"
      negIdx := some i
    else
      let d ←
        if d0 == 0 && !allowZero then
          if h : i < inDims.size then pure inDims[i]
          else throw s!"{ctx}: Reshape dim 0 at index {i} out of range for input rank {inDims.size}"
        else if d0 <= 0 then
          throw s!"{ctx}: Reshape target {raw.toList} has non-positive dim[{i}]={d0}"
        else
          pure d0.toNat
      resolved := resolved.set! i d
      let nextProd := knownProd * d
      if nextProd > maxU64Nat then
        throw s!"{ctx}: Reshape element count overflow"
      knownProd := nextProd
  if let some ni := negIdx then
    if knownProd == 0 || total % knownProd != 0 then
      throw s!"{ctx}: cannot infer -1 in Reshape target {raw.toList} for input size {total}"
    resolved := resolved.set! ni (total / knownProd)
  else if knownProd != total then
    throw s!"{ctx}: Reshape target {resolved.toList} element count {knownProd} != input element count {total}"
  if resolved.size == 4 then
    if resolved[0]! != 1 then
      throw s!"{ctx}: Reshape output batch dimension N must be 1, got {resolved[0]!}"
  else if resolved.size == 2 then
    if resolved[0]! != 1 then
      throw s!"{ctx}: 2D Reshape output batch dimension N must be 1, got {resolved[0]!}"
  else if resolved.size != 3 then
    throw s!"{ctx}: Reshape target rank {resolved.size} not supported (expected 4D [1,C,H,W], 3D [H,M,K], or 2D [1,C])"
  pure resolved

private def nodeNameOrOut (nd : OnnxNode) (idx : Nat) : String :=
  if !nd.name.isEmpty then nd.name
  else if !nd.outputs.isEmpty && !nd.outputs[0]!.isEmpty then nd.outputs[0]!
  else s!"{nd.opType.toLower}_{idx}"

/-- Decode an `OnnxModel` + optional `SafeTensorsFile` mechanically into a typed
    `SSAGraph HighOp` (zero lookahead fusion or composite op decomposition!). -/
def decodeOnnxToHighGraph (model : OnnxModel) (opts : LowerOptions := {}) :
    Except String (SSAGraph HighOp × Nat × Nat) := do
  if !model.hasGraph then
    throw "onnx model has no graph"
  let g := model.graph
  if g.outputs.isEmpty then
    throw s!"graph '{g.name}': expected at least 1 output, got 0"

  -- 1. Collect initializers & SafeTensors overrides into a raw constant table.
  let mut consts : Array (String × RawConst) := #[]
  let mut initNames : Array String := #[]
  let mut sfUsed : Nat := 0
  let mut sfCounted : Array String := #[]

  for t in g.initializers do
    initNames := initNames.push t.name
    let tDims ← t.dimsToNat
    match opts.safeTensors with
    | some sf =>
      match sf.find? t.name with
      | some sfInfo =>
        if sfInfo.shape != tDims then
          throw s!"safetensors override for initializer '{t.name}': shape {sfInfo.shape.toList} != ONNX shape {tDims.toList}"
        consts := setAssoc consts t.name { name := t.name, dims := sfInfo.shape, payload := .safetensors sf }
        sfUsed := sfUsed + 1
        sfCounted := sfCounted.push t.name
      | none =>
        consts := setAssoc consts t.name { name := t.name, dims := tDims, payload := .onnx t }
    | none =>
      consts := setAssoc consts t.name { name := t.name, dims := tDims, payload := .onnx t }

  if let some sf := opts.safeTensors then
    for info in sf.tensors do
      if (lookupAssoc consts info.name).isNone then
        consts := setAssoc consts info.name { name := info.name, dims := info.shape, payload := .safetensors sf }

  -- 2. Resolve runtime graph inputs (occupying dense ValueIds `0 .. inputs.size - 1`).
  let mut ssaInputs : Array (ValueId × String × TensorType) := #[]
  let mut valOf : Array (String × ValueId) := #[]
  let mut tyOf : Array (ValueId × TensorType) := #[]
  let mut nextId : ValueId := 0

  for inp in g.inputs do
    if initNames.contains inp.name || (lookupAssoc consts inp.name).isSome then
      pure ()
    else
      let dims : Array Nat ←
        match lookupAssoc opts.inputShapes inp.name with
        | some ov => pure #[1, ov.c, ov.h, ov.w]
        | none =>
          if !inp.hasShape || inp.dims.isEmpty then
            throw s!"input '{inp.name}' has unknown shape (provide --input-shape {inp.name}=C,H,W)"
          let mut dimVals : Array Nat := Array.mkEmpty inp.dims.size
          for axis in [:inp.dims.size] do
            let d := inp.dims[axis]!
            if !d.hasValue || d.value <= 0 then
              throw s!"input '{inp.name}' has dynamic/invalid dim[{axis}] ('{d.param}') (provide --input-shape {inp.name}=C,H,W)"
            dimVals := dimVals.push d.value.toNat
          if dimVals.size == 4 then
            if dimVals[0]! != 1 then
              throw s!"input '{inp.name}' batch dimension N={dimVals[0]!} != 1 (only N=1 is supported; override with --input-shape {inp.name}=C,H,W)"
            pure dimVals
          else if dimVals.size == 3 then
            pure dimVals
          else if dimVals.size == 2 then
            if dimVals[0]! != 1 then
              throw s!"input '{inp.name}' batch dimension N={dimVals[0]!} != 1"
            pure dimVals
          else
            throw s!"input '{inp.name}' has unsupported rank {dimVals.size} (expected 4D [1,C,H,W], 3D [H,M,K], or override with --input-shape {inp.name}=C,H,W)"
      let vid := nextId
      nextId := nextId + 1
      let ty : TensorType := { shape := dims, dtype := .fp16 }
      ssaInputs := ssaInputs.push (vid, inp.name, ty)
      valOf := setAssoc valOf inp.name vid
      tyOf := tyOf.push (vid, ty)

  if ssaInputs.isEmpty then
    throw s!"graph '{g.name}' has no runtime inputs"
  if ssaInputs.size > 3 then
    throw s!"graph '{g.name}' has {ssaInputs.size} runtime inputs (Vane supports 1..3 inputs)"

  -- 3. Mechanical 1:1 node decoding into `SSAGraph HighOp`
  let opset := model.defaultOpsetVersion
  let legacySoftmax := opset > 0 && opset < 13
  let mut aliases : Array (String × String) := #[]
  -- Virtual 5D GQA tensors: `(name, base4DVid, base4DShape, shape5D, unsqAxis, repeatFactor)`
  let mut virt5D : Array (String × ValueId × Array Nat × Array Nat × Nat × Nat) := #[]
  let mut nodes : Array (IRNode HighOp) := #[]
  let mut elidedCount : Nat := 0

  let getTy := fun (tyM : Array (ValueId × TensorType)) (v : ValueId) (ctx : String) => do
    match tyM.find? fun p => p.1 == v with
    | some (_, ty) => pure ty
    | none => throw s!"{ctx}: internal error: missing type for SSA value %{v}"

  -- Materialize a float constant as a first-class SSA `CoreOp.const` node on demand
  let materializeFloatConst := fun (name : String) (consts' : Array (String × RawConst))
      (valM : Array (String × ValueId)) (tyM : Array (ValueId × TensorType))
      (nds : Array (IRNode HighOp)) (nid : ValueId)
      (sfCnt : Nat) (sfSeen : Array String) (ctx : String) => do
    match lookupAssoc valM name with
    | some existingVid => pure (existingVid, valM, tyM, nds, nid, sfCnt, sfSeen)
    | none =>
      let some cs := lookupAssoc consts' name
        | throw s!"{ctx}: unknown constant '{name}'"
      let vals ← cs.float64s
      let vid := nid
      let ty : TensorType := { shape := cs.dims, dtype := .f32 }
      let nd : IRNode HighOp := {
        name := name
        op := .core (.const (.f32 vals))
        inputs := #[]
        outputs := #[(vid, ty)]
      }
      let (sfCnt', sfSeen') :=
        if cs.isFromSafeTensors && !sfSeen.contains name then (sfCnt + 1, sfSeen.push name)
        else (sfCnt, sfSeen)
      pure (vid, setAssoc valM name vid, tyM.push (vid, ty), nds.push nd, nid + 1, sfCnt', sfSeen')

  let lookupRuntime := fun (valM : Array (String × ValueId)) (tyM : Array (ValueId × TensorType))
      (name ctx : String) => do
    let some vid := lookupAssoc valM name
      | throw s!"{ctx}: unknown runtime tensor '{name}' (missing producer or constant)"
    let ty ← getTy tyM vid ctx
    pure (vid, ty)

  for idx in [:g.nodes.size] do
    let nd := g.nodes[idx]!
    let lbl := nodeNameOrOut nd idx
    let ctx := s!"node '{lbl}' ({nd.opType})"

    if nd.opType == "Constant" then
      let cs ← extractConstantNode nd
      match lookupAssoc consts cs.name with
      | some existing =>
        if existing.isFromSafeTensors then
          if existing.dims != cs.dims then
            throw s!"safetensors override for Constant '{cs.name}': shape {existing.dims.toList} != ONNX Constant shape {cs.dims.toList}"
        else
          consts := setAssoc consts cs.name cs
      | none =>
        consts := setAssoc consts cs.name cs
      elidedCount := elidedCount + 1
    else if nd.opType == "Identity" then
      if nd.inputs.size < 1 || nd.outputs.size < 1 then
        throw s!"{ctx}: expected 1 input and 1 output"
      aliases := setAssoc aliases nd.outputs[0]! (resolveAliasStr aliases nd.inputs[0]!)
      elidedCount := elidedCount + 1
    else if nd.opType == "Dropout" then
      if nd.inputs.size < 1 || nd.outputs.size < 1 then
        throw s!"{ctx}: expected at least 1 input and 1 output"
      aliases := setAssoc aliases nd.outputs[0]! (resolveAliasStr aliases nd.inputs[0]!)
      elidedCount := elidedCount + 1
    else
      if nd.outputs.isEmpty || nd.outputs[0]!.isEmpty then
        throw s!"{ctx}: missing output tensor name"
      let inps := nd.inputs.map fun s => if s.isEmpty then "" else resolveAliasStr aliases s
      let out0 := nd.outputs[0]!

      match nd.opType with
      | "DequantizeLinear" =>
        -- ONNX DequantizeLinear(x, x_scale, x_zero_point?) -> const(.quant spec) + CoreOp.dequantize
        if inps.size < 2 then
          throw s!"{ctx}: DequantizeLinear requires at least 2 inputs (x, x_scale[, x_zero_point])"
        let blockSize := nd.attrInt "block_size" 0
        if blockSize != 0 then
          throw s!"{ctx}: DequantizeLinear block_size={blockSize} is not supported (only block_size=0)"
        let some xCs := lookupAssoc consts inps[0]!
          | throw s!"{ctx}: DequantizeLinear x '{inps[0]!}' must be a constant quantized initializer"
        let some sCs := lookupAssoc consts inps[1]!
          | throw s!"{ctx}: DequantizeLinear x_scale '{inps[1]!}' must be a constant initializer"
        if xCs.dims.isEmpty then
          throw s!"{ctx}: DequantizeLinear x must have rank >= 1"
        let (fmt, xDataType, codesBytes) ← match xCs.payload with
          | .onnx t =>
            let fmt ←
              if t.dataType == onnxInt8 then pure QuantFormat.int8
              else if t.dataType == onnxUint8 then pure QuantFormat.uint8
              else throw s!"{ctx}: DequantizeLinear x '{inps[0]!}' unsupported dataType {t.dataType} (expected INT8 or UINT8)"
            let bs ← t.toByteCodes
            pure (fmt, t.dataType, bs)
          | _ => throw s!"{ctx}: DequantizeLinear x '{inps[0]!}' must be an ONNX INT8/UINT8 initializer"
        let scales ← sCs.float64s
        let cout := xCs.dims[0]!
        let isScalarScale := sCs.dims.isEmpty || sCs.dims == #[1]
        if isScalarScale then
          if scales.size != 1 then
            throw s!"{ctx}: DequantizeLinear scalar x_scale length {scales.size} != 1"
        else
          if sCs.dims.size != 1 then
            throw s!"{ctx}: DequantizeLinear x_scale shape {sCs.dims.toList} must be scalar []/[1] or 1D [{cout}]"
          let axisAttr := nd.attrInt "axis" 1
          let axisNorm := if axisAttr < 0 then axisAttr + (xCs.dims.size : Int) else axisAttr
          if axisNorm < 0 || axisNorm >= (xCs.dims.size : Int) then
            throw s!"{ctx}: DequantizeLinear axis={axisAttr} out of range for rank {xCs.dims.size}"
          if axisNorm != 0 then
            throw s!"{ctx}: DequantizeLinear per-channel axis={axisAttr} (normalized {axisNorm}) is not supported (only axis=0 per-output-channel is supported)"
          if sCs.dims[0]! != cout || scales.size != cout then
            throw s!"{ctx}: DequantizeLinear x_scale shape {sCs.dims.toList} (len={scales.size}) != expected [{cout}] along axis=0"
        let zps? ←
          if inps.size >= 3 && !inps[2]!.isEmpty then
            let some zCs := lookupAssoc consts inps[2]!
              | throw s!"{ctx}: DequantizeLinear x_zero_point '{inps[2]!}' must be a constant initializer"
            match zCs.payload with
            | .onnx zt =>
              if zt.dataType != xDataType then
                throw s!"{ctx}: DequantizeLinear x_zero_point dataType {zt.dataType} != x dataType {xDataType}"
              if zCs.dims != sCs.dims then
                throw s!"{ctx}: DequantizeLinear x_zero_point shape {zCs.dims.toList} != x_scale shape {sCs.dims.toList}"
              let zb ← zt.toByteCodes
              if zb.size > 1 then
                let axisAttr := nd.attrInt "axis" 1
                let axisNorm := if axisAttr < 0 then axisAttr + (xCs.dims.size : Int) else axisAttr
                if axisNorm != 0 then
                  throw s!"{ctx}: DequantizeLinear per-channel axis={axisAttr} (normalized {axisNorm}) is not supported (only axis=0 per-output-channel is supported)"
              let expanded : Array UInt8 ←
                if zb.size == 1 then pure (Array.replicate cout zb.data[0]!)
                else if zb.size == cout then pure zb.data
                else throw s!"{ctx}: DequantizeLinear zero_point length {zb.size} != 1 or Cout={cout}"
              if fmt == .int8 && expanded.all (· == 0) then pure none
              else pure (some expanded)
            | _ => throw s!"{ctx}: DequantizeLinear zero_point must be an ONNX byte initializer"
          else pure none
        let qSpec : QuantWeightSpec := {
          format := fmt
          codes := codesBytes
          scales := scales
          zeroPoints := zps?
          palette := #[]
        }
        let qVid := nextId
        let dqVid := nextId + 1
        nextId := nextId + 2
        let qTy : TensorType := { shape := xCs.dims, dtype := DType.ofQuantFormat fmt }
        let dqTy : TensorType := { shape := xCs.dims, dtype := .fp16 }
        nodes := nodes.push {
          name := s!"{lbl}_qconst"
          op := .core (.const (.quant qSpec))
          inputs := #[]
          outputs := #[(qVid, qTy)]
        }
        nodes := nodes.push {
          name := lbl
          op := .core (.dequantize fmt)
          inputs := #[qVid]
          outputs := #[(dqVid, dqTy)]
        }
        tyOf := tyOf.push (qVid, qTy) |>.push (dqVid, dqTy)
        valOf := setAssoc valOf out0 dqVid

      | "Conv" =>
        if inps.size < 2 then throw s!"{ctx}: expected at least 2 inputs (X, W)"
        let (xVid0, xTy0) ← lookupRuntime valOf tyOf inps[0]! ctx
        if xTy0.shape.size != 4 && xTy0.shape.size != 3 then
          throw s!"{ctx}: Conv requires 4D [1,C,H,W] or 3D [1,C,L] input, got rank {xTy0.shape.size} ({xTy0.shape.toList})"
        let isConv1d := xTy0.shape.size == 3
        if xTy0.shape[0]! != 1 then
          throw s!"{ctx}: Conv batch dimension N={xTy0.shape[0]!} != 1"

        -- Weight may already be an SSA value (e.g. from DequantizeLinear) or a raw constant
        let (wVid0, wTy0) ← match lookupAssoc valOf inps[1]! with
          | some existingW =>
            let wt ← getTy tyOf existingW ctx
            pure (existingW, wt)
          | none =>
            let (wv, vm, tm, nds, nid, sfc, sfs) ←
              materializeFloatConst inps[1]! consts valOf tyOf nodes nextId sfUsed sfCounted ctx
            valOf := vm; tyOf := tm; nodes := nds; nextId := nid; sfUsed := sfc; sfCounted := sfs
            let wt ← getTy tyOf wv ctx
            pure (wv, wt)

        -- Lift 3D Conv1d `[1,C,L]` + `[Cout,Cin/G,K]` cleanly to 4D `[1,C,1,L]` + `[Cout,Cin/G,1,K]`
        let (xVid, wVid, cin, h, w, cout, cinPerG, kh, kw, sh, sw, pads, ho, wo) ←
          if isConv1d then
            if wTy0.shape.size != 3 then
              throw s!"{ctx}: Conv1d weight must be 3D [Cout,Cin/G,K], got shape {wTy0.shape.toList}"
            let cin1 := xTy0.shape[1]!; let w1 := xTy0.shape[2]!
            let cout1 := wTy0.shape[0]!; let cinPerG1 := wTy0.shape[1]!; let kw1 := wTy0.shape[2]!
            if let some dils := nd.attrInts? "dilations" then
              if dils.size != 1 || dils[0]! != 1 then
                throw s!"{ctx}: Conv1d unsupported dilations {dils.toList} (only [1] is supported)"
            let sw1 ← match nd.attrInts? "strides" with
              | some st =>
                if st.size != 1 || st[0]! <= 0 then
                  throw s!"{ctx}: Conv1d strides must be 1 positive int, got {st.toList}"
                pure st[0]!.toNat
              | none => pure 1
            let autoPad := (← nd.attrStringChecked "auto_pad" "NOTSET").toUpper
            let rawPads4 : Array Int ← match nd.attrInts? "pads" with
              | some ps =>
                if ps.size != 2 then throw s!"{ctx}: Conv1d pads must have length 2, got {ps.toList}"
                pure #[0, ps[0]!, 0, ps[1]!]
              | none => pure #[0, 0, 0, 0]
            let (pads4, ho1, wo1) ← computeConcreteConvPads 1 w1 1 kw1 1 sw1 autoPad rawPads4 ctx
            let x4Vid := nextId; nextId := nextId + 1
            let x4Ty : TensorType := { shape := #[1, cin1, 1, w1], dtype := xTy0.dtype }
            nodes := nodes.push {
              name := s!"{lbl}_in4d", op := .core (.reshape x4Ty.shape), inputs := #[xVid0], outputs := #[(x4Vid, x4Ty)]
            }
            tyOf := tyOf.push (x4Vid, x4Ty)
            let w4Vid := nextId; nextId := nextId + 1
            let w4Ty : TensorType := { shape := #[cout1, cinPerG1, 1, kw1], dtype := wTy0.dtype }
            nodes := nodes.push {
              name := s!"{lbl}_w4d", op := .core (.reshape w4Ty.shape), inputs := #[wVid0], outputs := #[(w4Vid, w4Ty)]
            }
            tyOf := tyOf.push (w4Vid, w4Ty)
            pure (x4Vid, w4Vid, cin1, 1, w1, cout1, cinPerG1, 1, kw1, 1, sw1, pads4, ho1, wo1)
          else
            if wTy0.shape.size != 4 then
              throw s!"{ctx}: weight must be 4D [Cout,Cin/G,kH,kW], got shape {wTy0.shape.toList}"
            let cin2 := xTy0.shape[1]!; let h2 := xTy0.shape[2]!; let w2 := xTy0.shape[3]!
            let cout2 := wTy0.shape[0]!; let cinPerG2 := wTy0.shape[1]!
            let kh2 := wTy0.shape[2]!; let kw2 := wTy0.shape[3]!
            if let some dils := nd.attrInts? "dilations" then
              if dils.size != 2 || dils[0]! != 1 || dils[1]! != 1 then
                throw s!"{ctx}: unsupported dilations {dils.toList} (only [1,1] is supported)"
            let (sh2, sw2) ← match nd.attrInts? "strides" with
              | some st =>
                if st.size != 2 || st[0]! <= 0 || st[1]! <= 0 then
                  throw s!"{ctx}: strides must be 2 positive ints, got {st.toList}"
                pure (st[0]!.toNat, st[1]!.toNat)
              | none => pure (1, 1)
            let autoPad := (← nd.attrStringChecked "auto_pad" "NOTSET").toUpper
            let rawPads ← match nd.attrInts? "pads" with
              | some ps =>
                if ps.size != 4 then throw s!"{ctx}: pads must have length 4, got {ps.toList}"
                pure ps
              | none => pure #[0, 0, 0, 0]
            let (pads4, ho2, wo2) ← computeConcreteConvPads h2 w2 kh2 kw2 sh2 sw2 autoPad rawPads ctx
            pure (xVid0, wVid0, cin2, h2, w2, cout2, cinPerG2, kh2, kw2, sh2, sw2, pads4, ho2, wo2)

        let _ := (h, w)
        let groupInt := nd.attrInt "group" 1
        if groupInt <= 0 then throw s!"{ctx}: invalid group={groupInt}"
        let group := groupInt.toNat
        if cin % group != 0 || cout % group != 0 || cin / group != cinPerG then
          throw s!"{ctx}: incompatible Conv channels cin={cin} cout={cout} group={group} weight_cin_per_g={cinPerG}"

        let mut convInps : Array ValueId := #[xVid, wVid]
        let mut hasBias := false
        if inps.size >= 3 && !inps[2]!.isEmpty then
          let (bVid, vm, tm, nds, nid, sfc, sfs) ←
            materializeFloatConst inps[2]! consts valOf tyOf nodes nextId sfUsed sfCounted ctx
          valOf := vm; tyOf := tm; nodes := nds; nextId := nid; sfUsed := sfc; sfCounted := sfs
          let bTy ← getTy tyOf bVid ctx
          if bTy.shape.size != 1 || bTy.shape[0]! != cout then
            throw s!"{ctx}: bias '{inps[2]!}' must have shape [{cout}], got {bTy.shape.toList}"
          convInps := convInps.push bVid
          hasBias := true

        let convOutVid := nextId; nextId := nextId + 1
        let convOutTy : TensorType := { shape := #[1, cout, ho, wo], dtype := .fp16 }
        nodes := nodes.push {
          name := if isConv1d then s!"{lbl}_conv" else lbl
          op := .core (.conv kh kw sh sw pads group hasBias .none)
          inputs := convInps
          outputs := #[(convOutVid, convOutTy)]
        }
        tyOf := tyOf.push (convOutVid, convOutTy)
        if isConv1d then
          let out3Vid := nextId; nextId := nextId + 1
          let out3Ty : TensorType := { shape := #[1, cout, wo], dtype := .fp16 }
          nodes := nodes.push {
            name := lbl, op := .core (.reshape out3Ty.shape), inputs := #[convOutVid], outputs := #[(out3Vid, out3Ty)]
          }
          tyOf := tyOf.push (out3Vid, out3Ty)
          valOf := setAssoc valOf out0 out3Vid
        else
          valOf := setAssoc valOf out0 convOutVid

      | "BatchNormalization" =>
        if inps.size < 5 then throw s!"{ctx}: expected 5 inputs (X, scale, B, mean, var)"
        if nd.attrInt "training_mode" 0 != 0 then
          throw s!"{ctx}: BatchNormalization training_mode != 0 is not supported (inference mode only)"
        let (xVid0, xTy0) ← lookupRuntime valOf tyOf inps[0]! ctx
        let is3D := xTy0.shape.size == 3
        let (xVid, xTy) ←
          if is3D then
            if xTy0.shape[0]! != 1 then
              throw s!"{ctx}: 3D BatchNormalization batch dimension N={xTy0.shape[0]!} != 1"
            let x4Shape := #[1, xTy0.shape[1]!, 1, xTy0.shape[2]!]
            -- Bypass round-trip 4D->3D->4D reshape so HighFuse folds Conv1d+BN1d in exact f64
            let bypassed? : Option (ValueId × TensorType) := Id.run do
              for prevNd in nodes do
                if prevNd.outputs.size == 1 && prevNd.outputs[0]!.1 == xVid0 then
                  match prevNd.op with
                  | .core (.reshape _) =>
                    if prevNd.inputs.size == 1 then
                      let origVid := prevNd.inputs[0]!
                      match tyOf.find? (fun p => p.1 == origVid) with
                      | some (_, origTy) =>
                        if origTy.shape == x4Shape then
                          return some (origVid, origTy)
                      | none => pure ()
                  | _ => pure ()
              return none
            match bypassed? with
            | some pair => pure pair
            | none =>
              let x4Vid := nextId; nextId := nextId + 1
              let x4Ty : TensorType := { shape := x4Shape, dtype := xTy0.dtype }
              nodes := nodes.push {
                name := s!"{lbl}_in4d", op := .core (.reshape x4Ty.shape), inputs := #[xVid0], outputs := #[(x4Vid, x4Ty)]
              }
              tyOf := tyOf.push (x4Vid, x4Ty)
              pure (x4Vid, x4Ty)
          else
            pure (xVid0, xTy0)
        let mut bnInps : Array ValueId := #[xVid]
        for slot in [1:5] do
          let (cv, vm, tm, nds, nid, sfc, sfs) ←
            materializeFloatConst inps[slot]! consts valOf tyOf nodes nextId sfUsed sfCounted ctx
          valOf := vm; tyOf := tm; nodes := nds; nextId := nid; sfUsed := sfc; sfCounted := sfs
          bnInps := bnInps.push cv
        let eps := nd.attrFloat "epsilon" 1e-5
        let bnOutVid := nextId; nextId := nextId + 1
        nodes := nodes.push {
          name := if is3D then s!"{lbl}_bn" else lbl
          op := .batchNorm eps
          inputs := bnInps
          outputs := #[(bnOutVid, xTy)]
        }
        tyOf := tyOf.push (bnOutVid, xTy)
        if is3D then
          let out3Vid := nextId; nextId := nextId + 1
          nodes := nodes.push {
            name := lbl, op := .core (.reshape xTy0.shape), inputs := #[bnOutVid], outputs := #[(out3Vid, xTy0)]
          }
          tyOf := tyOf.push (out3Vid, xTy0)
          valOf := setAssoc valOf out0 out3Vid
        else
          valOf := setAssoc valOf out0 bnOutVid

      | "Gemm" =>
        if inps.size < 2 then throw s!"{ctx}: expected at least 2 inputs"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        if xTy.shape.size != 2 && xTy.shape.size != 4 then
          throw s!"{ctx}: Gemm requires 2D [1,Cin] or 4D [1,Cin,1,1] input, got rank {xTy.shape.size} ({xTy.shape.toList})"
        let xSh4 ← logicalToShape4D xTy.shape ctx
        let cin := xSh4.c
        if xSh4.h != 1 || xSh4.w != 1 then
          throw s!"{ctx}: Gemm requires 2D input [1, Cin] (normalized [1, Cin, 1, 1]), got [1, {cin}, {xSh4.h}, {xSh4.w}]"
        let some bc := lookupAssoc consts inps[1]!
          | throw s!"{ctx}: weight input '{inps[1]!}' must be a constant initializer"
        if bc.dims.size != 2 then
          throw s!"{ctx}: weight '{inps[1]!}' must be 2D, got shape {bc.dims.toList}"
        let bRaw ← bc.float64s
        if bc.isFromSafeTensors && !sfCounted.contains inps[1]! then
          sfUsed := sfUsed + 1; sfCounted := sfCounted.push inps[1]!
        let transA := nd.attrInt "transA" 0
        let transB := nd.attrInt "transB" 0
        let alpha := nd.attrFloat "alpha" 1.0
        let beta := nd.attrFloat "beta" 1.0
        if transA != 0 then throw s!"{ctx}: transA={transA} is not supported (expected 0)"
        let (cout, wF64) ←
          if transB != 0 then
            let co := bc.dims[0]!
            if bc.dims[1]! != cin then
              throw s!"{ctx}: weight shape [{bc.dims[0]!},{bc.dims[1]!}] (transB=1) incompatible with Cin={cin}"
            pure (co, bRaw.map fun v => alpha * v)
          else
            if bc.dims[0]! != cin then
              throw s!"{ctx}: weight shape [{bc.dims[0]!},{bc.dims[1]!}] (transB=0) incompatible with Cin={cin}"
            let co := bc.dims[1]!
            let mut wArr : Array Float := Array.replicate (co * cin) 0.0
            for c in [:cin] do
              for o in [:co] do
                wArr := wArr.set! (o * cin + c) (alpha * bRaw[c * co + o]!)
            pure (co, wArr)
        let wVid := nextId; nextId := nextId + 1
        let wTy : TensorType := { shape := #[cout, cin, 1, 1], dtype := .f32 }
        nodes := nodes.push {
          name := s!"{lbl}_w", op := .core (.const (.f32 wF64)), inputs := #[], outputs := #[(wVid, wTy)]
        }
        tyOf := tyOf.push (wVid, wTy)
        let mut convInps : Array ValueId := #[xVid, wVid]
        let mut hasBias := false
        if inps.size >= 3 && !inps[2]!.isEmpty then
          let some cc := lookupAssoc consts inps[2]!
            | throw s!"{ctx}: bias input '{inps[2]!}' must be a constant initializer"
          let cRaw ← cc.float64s
          if cc.isFromSafeTensors && !sfCounted.contains inps[2]! then
            sfUsed := sfUsed + 1; sfCounted := sfCounted.push inps[2]!
          let bF64 ←
            if cRaw.size == 1 then pure (Array.replicate cout (beta * cRaw[0]!))
            else if cRaw.size == cout then pure (cRaw.map fun v => beta * v)
            else throw s!"{ctx}: bias length {cRaw.size} incompatible with Cout={cout}"
          let bVid := nextId; nextId := nextId + 1
          let bTy : TensorType := { shape := #[cout], dtype := .f32 }
          nodes := nodes.push {
            name := s!"{lbl}_b", op := .core (.const (.f32 bF64)), inputs := #[], outputs := #[(bVid, bTy)]
          }
          tyOf := tyOf.push (bVid, bTy)
          convInps := convInps.push bVid
          hasBias := true
        let outVid := nextId; nextId := nextId + 1
        let outShape := if xTy.shape.size == 2 then #[1, cout] else #[1, cout, 1, 1]
        let outTy : TensorType := { shape := outShape, dtype := .fp16 }
        nodes := nodes.push {
          name := lbl
          op := .core (.conv 1 1 1 1 #[0, 0, 0, 0] 1 hasBias .none)
          inputs := convInps
          outputs := #[(outVid, outTy)]
        }
        tyOf := tyOf.push (outVid, outTy)
        valOf := setAssoc valOf out0 outVid

      | "MatMul" =>
        if inps.size != 2 then throw s!"{ctx}: expected 2 inputs, got {inps.size}"
        match lookupAssoc consts inps[1]! with
        | some bc =>
          -- Static-weight MatMul(A, const B) -> 1x1 conv with transposed weight constant [Cout, Cin, 1, 1]
          let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
          if xTy.shape.size != 2 && xTy.shape.size != 4 then
            throw s!"{ctx}: static-weight MatMul requires 2D [1,C] or 4D [1,C,H,W] input, got rank {xTy.shape.size} ({xTy.shape.toList})"
          let cin := xTy.shape[1]!
          if bc.dims.size != 2 then
            throw s!"{ctx}: static MatMul weight '{inps[1]!}' must be 2D, got shape {bc.dims.toList}"
          if bc.dims[0]! != cin then
            throw s!"{ctx}: weight shape [{bc.dims[0]!},{bc.dims[1]!}] incompatible with Cin={cin}"
          let cout := bc.dims[1]!
          let bRaw ← bc.float64s
          if bc.isFromSafeTensors && !sfCounted.contains inps[1]! then
            sfUsed := sfUsed + 1; sfCounted := sfCounted.push inps[1]!
          let mut wF64 : Array Float := Array.replicate (cout * cin) 0.0
          for c in [:cin] do
            for o in [:cout] do
              wF64 := wF64.set! (o * cin + c) bRaw[c * cout + o]!
          let wVid := nextId; nextId := nextId + 1
          let wTy : TensorType := { shape := #[cout, cin, 1, 1], dtype := .f32 }
          nodes := nodes.push {
            name := s!"{lbl}_w", op := .core (.const (.f32 wF64)), inputs := #[], outputs := #[(wVid, wTy)]
          }
          tyOf := tyOf.push (wVid, wTy)
          let outVid := nextId; nextId := nextId + 1
          let outShape := if xTy.shape.size == 2 then #[1, cout] else #[1, cout, xTy.shape[2]!, xTy.shape[3]!]
          let outTy : TensorType := { shape := outShape, dtype := .fp16 }
          nodes := nodes.push {
            name := lbl
            op := .core (.conv 1 1 1 1 #[0, 0, 0, 0] 1 false .none)
            inputs := #[xVid, wVid]
            outputs := #[(outVid, outTy)]
          }
          tyOf := tyOf.push (outVid, outTy)
          valOf := setAssoc valOf out0 outVid
        | none =>
          -- Dynamic MatMul(A, B) -> CoreOp.matmul (transpose & scale fusions happen in CoreFuse!)
          if (lookupAssoc consts inps[0]!).isSome then
            throw s!"{ctx}: constant LHS with dynamic RHS in MatMul is not supported"
          let (aVid, aTy) ← lookupRuntime valOf tyOf inps[0]! ctx
          let (bVid, bTy) ← lookupRuntime valOf tyOf inps[1]! ctx
          let outShape : Array Nat :=
            if aTy.shape.size == 3 && bTy.shape.size == 3 then
              #[aTy.shape[0]!, aTy.shape[1]!, bTy.shape[2]!]
            else if aTy.shape.size == 4 && bTy.shape.size == 4 &&
                    (aTy.shape[2]! > 1 || bTy.shape[2]! > 1) then
              #[1, aTy.shape[1]!, aTy.shape[2]!, bTy.shape[3]!]
            else if aTy.shape.size == 4 && bTy.shape.size == 4 then
              #[1, aTy.shape[1]!, 1, bTy.shape[3]!]
            else
              aTy.shape
          let outVid := nextId; nextId := nextId + 1
          let outTy : TensorType := { shape := outShape, dtype := .fp16 }
          nodes := nodes.push {
            name := lbl
            op := .core (.matmul false false f16PosOne)
            inputs := #[aVid, bVid]
            outputs := #[(outVid, outTy)]
          }
          tyOf := tyOf.push (outVid, outTy)
          valOf := setAssoc valOf out0 outVid

      | "Add" | "Sub" | "Mul" | "Div" | "Max" | "Min" =>
        if inps.size != 2 then throw s!"{ctx}: expected 2 inputs, got {inps.size}"
        let bop : ElemBinaryOp := match nd.opType with
          | "Add" => .add | "Sub" => .sub | "Mul" => .mul
          | "Div" => .div | "Max" => .max | _ => .min
        let resolveOperand := fun (nm : String) (valM : Array (String × ValueId))
            (tyM : Array (ValueId × TensorType)) (nds : Array (IRNode HighOp)) (nid : ValueId)
            (sfc : Nat) (sfs : Array String) => do
          match lookupAssoc valM nm with
          | some vid =>
            let ty ← getTy tyM vid ctx
            pure (vid, ty, valM, tyM, nds, nid, sfc, sfs)
          | none =>
            let (vid, vm, tm, nds', nid', sfc', sfs') ←
              materializeFloatConst nm consts valM tyM nds nid sfc sfs ctx
            let ty ← getTy tm vid ctx
            pure (vid, ty, vm, tm, nds', nid', sfc', sfs')
        let (aVid, aTy, vm1, tm1, nds1, nid1, sfc1, sfs1) ←
          resolveOperand inps[0]! valOf tyOf nodes nextId sfUsed sfCounted
        valOf := vm1; tyOf := tm1; nodes := nds1; nextId := nid1; sfUsed := sfc1; sfCounted := sfs1
        let (bVid, bTy, vm2, tm2, nds2, nid2, sfc2, sfs2) ←
          resolveOperand inps[1]! valOf tyOf nodes nextId sfUsed sfCounted
        valOf := vm2; tyOf := tm2; nodes := nds2; nextId := nid2; sfUsed := sfc2; sfCounted := sfs2
        let outShape ← match broadcastOnnxShapes aTy.shape bTy.shape with
          | .ok s => pure s
          | .error e => throw s!"{ctx}: {e}"
        let outVid := nextId; nextId := nextId + 1
        let outTy : TensorType := { shape := outShape, dtype := .fp16 }
        nodes := nodes.push {
          name := lbl
          op := .core (.elemBinary bop)
          inputs := #[aVid, bVid]
          outputs := #[(outVid, outTy)]
        }
        tyOf := tyOf.push (outVid, outTy)
        valOf := setAssoc valOf out0 outVid

      | "Pow" =>
        if inps.size != 2 then throw s!"{ctx}: Pow requires 2 inputs (X, exponent)"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let some expCs := lookupAssoc consts inps[1]!
          | throw s!"{ctx}: Pow exponent '{inps[1]!}' must be a constant scalar"
        if !(← expCs.isScalar) then
          throw s!"{ctx}: Pow exponent '{inps[1]!}' must be scalar, got shape {expCs.dims.toList}"
        let expVals ← expCs.float64s
        let powKind ← classifyPowExponent expVals[0]! ctx
        let outVid := nextId; nextId := nextId + 1
        nodes := nodes.push {
          name := lbl
          op := .pow powKind
          inputs := #[xVid]
          outputs := #[(outVid, xTy)]
        }
        tyOf := tyOf.push (outVid, xTy)
        valOf := setAssoc valOf out0 outVid

      | "Abs" | "Neg" | "Log" | "Softplus" =>
        if inps.size != 1 then throw s!"{ctx}: expected 1 input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let uop : ElemUnaryOp := match nd.opType with
          | "Abs" => .abs | "Neg" => .neg | "Log" => .log | _ => .softplus
        let outVid := nextId; nextId := nextId + 1
        nodes := nodes.push {
          name := lbl
          op := .core (.elemUnary uop)
          inputs := #[xVid]
          outputs := #[(outVid, xTy)]
        }
        tyOf := tyOf.push (outVid, xTy)
        valOf := setAssoc valOf out0 outVid

      | "Gelu" =>
        if inps.size != 1 then throw s!"{ctx}: expected 1 input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let approxStr ← nd.attrStringChecked "approximate" ""
        let approx ← match classifyGeluApprox approxStr with
          | .ok a => pure a
          | .error e => throw s!"{ctx}: {e}"
        let outVid := nextId; nextId := nextId + 1
        nodes := nodes.push {
          name := lbl
          op := .gelu approx
          inputs := #[xVid]
          outputs := #[(outVid, xTy)]
        }
        tyOf := tyOf.push (outVid, xTy)
        valOf := setAssoc valOf out0 outVid

      | "Relu" | "Sigmoid" | "Tanh" | "Exp" | "Erf" | "Sqrt" | "Sign" | "Silu" | "Swish" | "Reciprocal" =>
        if inps.size != 1 then throw s!"{ctx}: expected 1 input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let act : NeActSpec := match nd.opType with
          | "Relu" => .relu | "Sigmoid" => .lut .sigmoid | "Tanh" => .lut .tanh
          | "Exp" => .lut .exp | "Erf" => .lut .erf | "Sqrt" => .lut .sqrt
          | "Sign" => .lut .sign | "Silu" | "Swish" => .lut .silu | _ => .lut .inverse
        let outVid := nextId; nextId := nextId + 1
        nodes := nodes.push {
          name := lbl, op := .core (.lut act), inputs := #[xVid], outputs := #[(outVid, xTy)]
        }
        tyOf := tyOf.push (outVid, xTy)
        valOf := setAssoc valOf out0 outVid

      | "Elu" =>
        if inps.size != 1 then throw s!"{ctx}: expected 1 input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let alpha := nd.attrFloat "alpha" 1.0
        let _ ← validateFiniteFp16ScalarLower alpha ctx
        let outVid := nextId; nextId := nextId + 1
        nodes := nodes.push {
          name := lbl, op := .core (.lut (.lut (.elu alpha))), inputs := #[xVid], outputs := #[(outVid, xTy)]
        }
        tyOf := tyOf.push (outVid, xTy)
        valOf := setAssoc valOf out0 outVid

      | "LeakyRelu" =>
        if inps.size != 1 then throw s!"{ctx}: expected 1 input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let alpha := nd.attrFloat "alpha" 0.01
        let _ ← validateFiniteFp16ScalarLower alpha ctx
        let outVid := nextId; nextId := nextId + 1
        nodes := nodes.push {
          name := lbl, op := .core (.lut (.lut (.leakyRelu alpha))), inputs := #[xVid], outputs := #[(outVid, xTy)]
        }
        tyOf := tyOf.push (outVid, xTy)
        valOf := setAssoc valOf out0 outVid

      | "HardSigmoid" =>
        if inps.size != 1 then throw s!"{ctx}: expected 1 input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let alpha := nd.attrFloat "alpha" 0.2
        let beta := nd.attrFloat "beta" 0.5
        if alpha.isNaN || alpha.isInf || alpha <= 0.0 then
          throw s!"{ctx}: HardSigmoid alpha must be finite and > 0 (got {alpha})"
        let _ ← validateFiniteFp16ScalarLower alpha ctx
        let _ ← validateFiniteFp16ScalarLower beta ctx
        let preBits ← validateFiniteFp16ScalarLower (alpha / 0.2) ctx
        if (preBits &&& 0x7fff) == 0 then
          throw s!"{ctx}: HardSigmoid alpha ({alpha}) underflows pre-LUT scale to 0"
        let outVid := nextId; nextId := nextId + 1
        nodes := nodes.push {
          name := lbl, op := .core (.lut (.lut (.hardSigmoid alpha beta))), inputs := #[xVid], outputs := #[(outVid, xTy)]
        }
        tyOf := tyOf.push (outVid, xTy)
        valOf := setAssoc valOf out0 outVid

      | "Clip" =>
        if inps.size < 1 then throw s!"{ctx}: expected at least 1 input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let some lo ← readOptionalScalarParam nd inps consts 1 "min" ctx
          | throw s!"{ctx}: Clip requires both constant min and max bounds"
        let some hi ← readOptionalScalarParam nd inps consts 2 "max" ctx
          | throw s!"{ctx}: Clip requires both constant min and max bounds"
        let _ ← validateFiniteFp16ScalarLower lo ctx
        let _ ← validateFiniteFp16ScalarLower hi ctx
        if lo > hi then throw s!"{ctx}: Clip requires min <= max (got min={lo}, max={hi})"
        let lspec := if lo == 0.0 && hi == 6.0 then LutSpec.relu6 else LutSpec.clip lo hi
        let outVid := nextId; nextId := nextId + 1
        nodes := nodes.push {
          name := lbl, op := .core (.lut (.lut lspec)), inputs := #[xVid], outputs := #[(outVid, xTy)]
        }
        tyOf := tyOf.push (outVid, xTy)
        valOf := setAssoc valOf out0 outVid

      | "MaxPool" | "AveragePool" =>
        if inps.size != 1 then throw s!"{ctx}: expected 1 input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        if xTy.shape.size != 4 then
          throw s!"{ctx}: {nd.opType} requires 4D [1,C,H,W] input, got rank {xTy.shape.size} ({xTy.shape.toList})"
        let some kShape := nd.attrInts? "kernel_shape"
          | throw s!"{ctx}: missing kernel_shape attribute"
        if kShape.size != 2 || kShape[0]! <= 0 || kShape[1]! <= 0 then
          throw s!"{ctx}: kernel_shape attribute must be 2 positive ints, got {kShape.toList}"
        let kh := kShape[0]!.toNat; let kw := kShape[1]!.toNat
        if nd.attrInt "ceil_mode" 0 != 0 then throw s!"{ctx}: ceil_mode != 0 is not supported"
        if let some dils := nd.attrInts? "dilations" then
          if dils.size != 2 || dils[0]! != 1 || dils[1]! != 1 then
            throw s!"{ctx}: unsupported dilations {dils.toList} (only [1,1] is supported)"
        let (sh, sw) ← match nd.attrInts? "strides" with
          | some st =>
            if st.size != 2 || st[0]! <= 0 || st[1]! <= 0 then
              throw s!"{ctx}: pool strides must be 2 positive ints, got {st.toList}"
            pure (st[0]!.toNat, st[1]!.toNat)
          | none => pure (1, 1)
        let autoPad := (← nd.attrStringChecked "auto_pad" "NOTSET").toUpper
        let pads ← match nd.attrInts? "pads" with
          | some ps =>
            if ps.size != 4 then throw s!"{ctx}: pads must have length 4, got {ps.toList}"
            for p in ps do if p < 0 then throw s!"{ctx}: negative pool pad {p}"
            pure ps
          | none => pure #[0, 0, 0, 0]
        let h := xTy.shape[2]!; let w := xTy.shape[3]!
        let padMode : PoolPad ← match autoPad with
          | "VALID" => pure .valid
          | "SAME_UPPER" => pure .same
          | "SAME_LOWER" =>
            let ohTmp := ceilDiv h sh
            let padH := if (ohTmp - 1) * sh + kh > h then (ohTmp - 1) * sh + kh - h else 0
            let owTmp := ceilDiv w sw
            let padW := if (owTmp - 1) * sw + kw > w then (owTmp - 1) * sw + kw - w else 0
            let pt := (padH + 1) / 2; let pb := padH - pt
            let pl := (padW + 1) / 2; let pr := padW - pl
            if pt == 0 && pb == 0 && pl == 0 && pr == 0 then pure .valid else pure (.custom pt pb pl pr)
          | "" | "NOTSET" =>
            let pt := pads[0]!.toNat; let pl := pads[1]!.toNat
            let pb := pads[2]!.toNat; let pr := pads[3]!.toNat
            if pt == 0 && pb == 0 && pl == 0 && pr == 0 then pure .valid else pure (.custom pt pb pl pr)
          | other => throw s!"{ctx}: unsupported auto_pad '{other}'"
        let kind : PoolKind := if nd.opType == "MaxPool" then .max else .avg
        let excl := if nd.opType == "AveragePool" then nd.attrInt "count_include_pad" 0 == 0 else true
        let pspec : PoolSpec := {
          target := opts.target, op := kind
          dims := { c := xTy.shape[1]!, h := h, w := w }
          kh := kh, kw := kw, sh := sh, sw := sw, pad := padMode, excludePad := excl
        }
        let (_, gh, gw) ← resolvePoolBackend pspec
        let outVid := nextId; nextId := nextId + 1
        let outTy : TensorType := { shape := #[1, xTy.shape[1]!, gh.out, gw.out], dtype := .fp16 }
        nodes := nodes.push {
          name := lbl, op := .core (.pool kind kh kw sh sw padMode excl)
          inputs := #[xVid], outputs := #[(outVid, outTy)]
        }
        tyOf := tyOf.push (outVid, outTy)
        valOf := setAssoc valOf out0 outVid

      | "GlobalAveragePool" =>
        if inps.size != 1 then throw s!"{ctx}: expected 1 input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        if xTy.shape.size != 4 then
          throw s!"{ctx}: GlobalAveragePool requires 4D [1,C,H,W] input, got rank {xTy.shape.size} ({xTy.shape.toList})"
        let outVid := nextId; nextId := nextId + 1
        let outTy : TensorType := { shape := #[1, xTy.shape[1]!, 1, 1], dtype := .fp16 }
        nodes := nodes.push {
          name := lbl, op := .core (.reduce .mean #[2, 3] true 0)
          inputs := #[xVid], outputs := #[(outVid, outTy)]
        }
        tyOf := tyOf.push (outVid, outTy)
        valOf := setAssoc valOf out0 outVid

      | "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceL1" | "ReduceL2" | "ReduceSumSquare" =>
        if inps.size < 1 then throw s!"{ctx}: expected at least 1 input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        if nd.attrInt "keepdims" 1 == 0 then
          throw s!"{ctx}: keepdims=0 is not supported (Vane reduce preserves 4D rank with keepdims=1)"
        let rawAxes ← match nd.attrInts? "axes" with
          | some ax => pure ax
          | none =>
            if inps.size >= 2 && !inps[1]!.isEmpty then
              let some ac := lookupAssoc consts inps[1]!
                | throw s!"{ctx}: axes input '{inps[1]!}' must be a constant int64 tensor"
              ac.int64s
            else pure #[]
        if rawAxes.isEmpty && nd.attrInt "noop_with_empty_axes" 0 != 0 then
          aliases := setAssoc aliases out0 inps[0]!
          elidedCount := elidedCount + 1
        else
          let rank := xTy.shape.size
          let effRaw :=
            if rawAxes.isEmpty then
              if rank == 3 then #[0, 1, 2]
              else if rank == 2 then #[1]
              else #[1, 2, 3]
            else rawAxes
          let mut axes : Array Nat := #[]
          let mut outShape := xTy.shape
          for ax0 in effRaw do
            let ax := if ax0 < 0 then ax0 + (rank : Int) else ax0
            if ax < 0 || ax >= (rank : Int) then
              throw s!"{ctx}: reduce axis {ax0} out of range for rank {rank}"
            let aNat := ax.toNat
            if !axes.contains aNat then
              axes := axes.push aNat
              outShape := outShape.set! aNat 1
          let redOp : ReduceOp := match nd.opType with
            | "ReduceMean" => .mean | "ReduceSum" => .sum | "ReduceMax" => .max
            | "ReduceMin" => .min | "ReduceL1" => .l1 | "ReduceL2" => .l2 | _ => .sumSquare
          let outVid := nextId; nextId := nextId + 1
          let outTy : TensorType := { shape := outShape, dtype := .fp16 }
          nodes := nodes.push {
            name := lbl, op := .core (.reduce redOp axes true 0)
            inputs := #[xVid], outputs := #[(outVid, outTy)]
          }
          tyOf := tyOf.push (outVid, outTy)
          valOf := setAssoc valOf out0 outVid

      | "Softmax" =>
        if inps.size != 1 then throw s!"{ctx}: expected 1 input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let axis := if legacySoftmax then nd.attrInt "axis" 1 else nd.attrInt "axis" (-1)
        let outVid := nextId; nextId := nextId + 1
        nodes := nodes.push {
          name := lbl, op := .softmax axis legacySoftmax, inputs := #[xVid], outputs := #[(outVid, xTy)]
        }
        tyOf := tyOf.push (outVid, xTy)
        valOf := setAssoc valOf out0 outVid

      | "RMSNormalization" =>
        if inps.size < 1 then throw s!"{ctx}: RMSNormalization requires at least 1 input (X[, scale])"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let axis := nd.attrInt "axis" (-1)
        let eps := nd.attrFloat "epsilon" 1e-5
        let rank := xTy.shape.size
        let axNat ← resolveRmsNormAxis rank axis ctx
        -- Match ONNX opset-23 suffix semantics (`X.shape[axis:]`), identical to LayerNormalization:
        -- only the last axis (`rank - 1`) is supported in general, plus 4D `axis = 1` when `H = 1, W = 1`.
        if rank == 4 then
          if axNat == 1 then
            if xTy.shape[2]! != 1 || xTy.shape[3]! != 1 then
              throw s!"{ctx}: 4D RMSNormalization with axis=1 normalizes over [C,H,W]={xTy.shape.extract 1 4 |>.toList}, which is only supported when H=1 and W=1 (got H={xTy.shape[2]!}, W={xTy.shape[3]!}); use axis=-1 (W) or reshape/transpose so the normalized axis is last"
          else if axNat != 3 then
            throw s!"{ctx}: 4D RMSNormalization axis {axis} not supported (supported: C [axis=1 with H=W=1] or W [axis=3/-1])"
        else if rank == 3 then
          if axNat != 2 then
            throw s!"{ctx}: 3D RMSNormalization axis {axis} not supported (expected -1 or 2)"
        else if rank == 2 then
          if axNat != 1 then
            throw s!"{ctx}: 2D RMSNormalization axis {axis} not supported (expected -1 or 1)"
        else
          throw s!"{ctx}: RMSNormalization requires 4D, 3D, or 2D input, got rank {rank}"
        let normDim := xTy.shape[axNat]!
        let mut rmsInps : Array ValueId := #[xVid]
        let mut hasScale := false
        if inps.size >= 2 && !inps[1]!.isEmpty then
          let scName := inps[1]!
          let scVid ← match lookupAssoc valOf scName with
            | some sv =>
              let svTy ← getTy tyOf sv ctx
              if !isValidRmsNormScaleShape svTy.shape axNat normDim rank then
                throw s!"{ctx}: RMSNorm Scale shape {svTy.shape.toList} incompatible with normalized axis {axNat} (normDim={normDim}, input rank {rank})"
              -- Canonicalize shared 1D scale `[normDim]` onto `axNat` so it never right-aligns to the wrong axis!
              let canonShape : Array Nat :=
                if svTy.numElems == 1 then #[]
                else (Array.replicate rank 1).set! axNat normDim
              if svTy.numElems > 1 && svTy.shape != canonShape then
                let rsVid := nextId; nextId := nextId + 1
                let rsTy : TensorType := { shape := canonShape, dtype := svTy.dtype }
                nodes := nodes.push {
                  name := s!"{lbl}_scale_canon"
                  op := .core (.reshape canonShape)
                  inputs := #[sv]
                  outputs := #[(rsVid, rsTy)]
                }
                tyOf := tyOf.push (rsVid, rsTy)
                pure rsVid
              else
                pure sv
            | none =>
              let some cs := lookupAssoc consts scName
                | throw s!"{ctx}: unknown RMSNormalization scale '{scName}'"
              if !isValidRmsNormScaleShape cs.dims axNat normDim rank then
                throw s!"{ctx}: RMSNorm Scale shape {cs.dims.toList} incompatible with normalized axis {axNat} (normDim={normDim}, input rank {rank})"
              let vals ← cs.float64s
              if cs.isFromSafeTensors && !sfCounted.contains scName then
                sfUsed := sfUsed + 1; sfCounted := sfCounted.push scName
              -- Canonicalize non-scalar scale shape onto `axNat` so numpy broadcasting aligns with `axNat`!
              let scShape : Array Nat :=
                if vals.size == 1 then #[]
                else (Array.replicate rank 1).set! axNat normDim
              let sv := nextId; nextId := nextId + 1
              let scTy : TensorType := { shape := scShape, dtype := .f32 }
              nodes := nodes.push {
                name := scName, op := .core (.const (.f32 vals)), inputs := #[], outputs := #[(sv, scTy)]
              }
              tyOf := tyOf.push (sv, scTy)
              pure sv
          rmsInps := rmsInps.push scVid
          hasScale := true
        let yVid := nextId; nextId := nextId + 1
        nodes := nodes.push {
          name := lbl
          op := .rmsNorm axis eps hasScale
          inputs := rmsInps
          outputs := #[(yVid, xTy)]
        }
        tyOf := tyOf.push (yVid, xTy)
        valOf := setAssoc valOf out0 yVid

      | "LayerNormalization" =>
        if inps.size < 2 then throw s!"{ctx}: LayerNormalization requires at least 2 inputs (X, Scale[, B])"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let axis := nd.attrInt "axis" (-1)
        let eps := nd.attrFloat "epsilon" 1e-5
        let mut lnInps : Array ValueId := #[xVid]
        let (scVid, vm1, tm1, nds1, nid1, sfc1, sfs1) ←
          match lookupAssoc valOf inps[1]! with
          | some sv => pure (sv, valOf, tyOf, nodes, nextId, sfUsed, sfCounted)
          | none => materializeFloatConst inps[1]! consts valOf tyOf nodes nextId sfUsed sfCounted ctx
        valOf := vm1; tyOf := tm1; nodes := nds1; nextId := nid1; sfUsed := sfc1; sfCounted := sfs1
        lnInps := lnInps.push scVid
        let mut hasBias := false
        if inps.size >= 3 && !inps[2]!.isEmpty then
          let (bVid, vm2, tm2, nds2, nid2, sfc2, sfs2) ←
            match lookupAssoc valOf inps[2]! with
            | some bv => pure (bv, valOf, tyOf, nodes, nextId, sfUsed, sfCounted)
            | none => materializeFloatConst inps[2]! consts valOf tyOf nodes nextId sfUsed sfCounted ctx
          valOf := vm2; tyOf := tm2; nodes := nds2; nextId := nid2; sfUsed := sfc2; sfCounted := sfs2
          lnInps := lnInps.push bVid
          hasBias := true

        let yVid := nextId; nextId := nextId + 1
        let mut lnOuts : Array (ValueId × TensorType) := #[(yVid, xTy)]
        tyOf := tyOf.push (yVid, xTy)
        valOf := setAssoc valOf out0 yVid
        -- Optional Mean and InvStdDev outputs
        let rank := xTy.shape.size
        let ax := if axis < 0 then axis + (rank : Int) else axis
        let redShape : Array Nat :=
          if ax >= 0 && ax < (rank : Int) then
            Array.ofFn (n := rank) fun i => if i.val >= ax.toNat then 1 else xTy.shape[i.val]!
          else xTy.shape
        let redTy : TensorType := { shape := redShape, dtype := .fp16 }
        let hasMean := nd.outputs.size >= 2 && !nd.outputs[1]!.isEmpty
        let hasInvStd := nd.outputs.size >= 3 && !nd.outputs[2]!.isEmpty
        if hasMean || hasInvStd then
          let mVid := nextId; nextId := nextId + 1
          lnOuts := lnOuts.push (mVid, redTy)
          tyOf := tyOf.push (mVid, redTy)
          if hasMean then
            valOf := setAssoc valOf nd.outputs[1]! mVid
        if hasInvStd then
          let isVid := nextId; nextId := nextId + 1
          lnOuts := lnOuts.push (isVid, redTy)
          tyOf := tyOf.push (isVid, redTy)
          valOf := setAssoc valOf nd.outputs[2]! isVid
        nodes := nodes.push {
          name := lbl
          op := .layerNorm axis eps true hasBias
          inputs := lnInps
          outputs := lnOuts
        }

      | "Split" =>
        if inps.size < 1 then throw s!"{ctx}: Split missing data input"
        if nd.outputs.isEmpty then throw s!"{ctx}: Split requires at least 1 output"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let rank := xTy.shape.size
        if rank != 4 && rank != 3 then
          throw s!"{ctx}: Split requires 4D or 3D input, got rank {rank}"
        let ax0 := nd.attrInt "axis" 0
        let axNorm := if ax0 < 0 then ax0 + (rank : Int) else ax0
        let axis : Nat ←
          if rank == 3 then
            if axNorm == 2 then pure 2
            else throw s!"{ctx}: 3D Split axis {ax0} not supported (expected -1 or 2)"
          else
            if axNorm == 1 || axNorm == 2 || axNorm == 3 then pure axNorm.toNat
            else throw s!"{ctx}: Split axis {ax0} must be in \{1,2,3}"
        let dim := xTy.shape[axis]!
        if let some numOutAttr := nd.attr? "num_outputs" then
          if numOutAttr.i <= 0 || numOutAttr.i.toNat != nd.outputs.size then
            throw s!"{ctx}: Split num_outputs={numOutAttr.i} != node outputs count {nd.outputs.size}"
        let sizesList : List Nat ←
          if inps.size >= 2 && !inps[1]!.isEmpty then
            let some sc := lookupAssoc consts inps[1]!
              | throw s!"{ctx}: Split sizes input '{inps[1]!}' must be a constant int64 tensor"
            let raw ← sc.int64s
            let mut ls : List Nat := []
            for v in raw do
              if v <= 0 then throw s!"{ctx}: Split size {v} must be >= 1"
              ls := ls ++ [v.toNat]
            pure ls
          else if let some raw := nd.attrInts? "split" then
            let mut ls : List Nat := []
            for v in raw do
              if v <= 0 then throw s!"{ctx}: Split size {v} must be >= 1"
              ls := ls ++ [v.toNat]
            pure ls
          else
            match computeDefaultSplitSizes dim nd.outputs.size with
            | .ok ls => pure ls
            | .error e => throw s!"{ctx}: {e}"
        let sizes := sizesList.toArray
        if sizes.size != nd.outputs.size then
          throw s!"{ctx}: Split sizes length {sizes.size} != outputs count {nd.outputs.size}"
        let total := sizes.foldl (· + ·) 0
        if total != dim then
          throw s!"{ctx}: Split sum(sizes)={total} != axis {axis} dim {dim}"
        let mut spOuts : Array (ValueId × TensorType) := Array.mkEmpty sizes.size
        for i in [:sizes.size] do
          let oVid := nextId; nextId := nextId + 1
          let oTy : TensorType := { shape := xTy.shape.set! axis sizes[i]!, dtype := .fp16 }
          spOuts := spOuts.push (oVid, oTy)
          tyOf := tyOf.push (oVid, oTy)
          if !nd.outputs[i]!.isEmpty then
            valOf := setAssoc valOf nd.outputs[i]! oVid
        nodes := nodes.push {
          name := lbl, op := .split axis sizes, inputs := #[xVid], outputs := spOuts
        }

      | "Transpose" =>
        if inps.size != 1 then throw s!"{ctx}: expected 1 input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let rank := xTy.shape.size
        let defaultPerm : Array Int := Array.ofFn (n := rank) fun i => ((rank - 1 - i.val) : Int)
        let perm64 := (nd.attrInts? "perm").getD defaultPerm
        if perm64.size != rank then
          throw s!"{ctx}: perm must have length {rank}, got {perm64.toList}"
        let mut perm : Array Nat := Array.mkEmpty rank
        let mut seen : Array Bool := Array.replicate rank false
        for p in perm64 do
          if p < 0 || p >= (rank : Int) || seen[p.toNat]! then
            throw s!"{ctx}: invalid permutation {perm64.toList}"
          seen := seen.set! p.toNat true
          perm := perm.push p.toNat
        let outShape := perm.map fun a => xTy.shape[a]!
        let outVid := nextId; nextId := nextId + 1
        let outTy : TensorType := { shape := outShape, dtype := .fp16 }
        nodes := nodes.push {
          name := lbl, op := .core (.transpose perm), inputs := #[xVid], outputs := #[(outVid, outTy)]
        }
        tyOf := tyOf.push (outVid, outTy)
        valOf := setAssoc valOf out0 outVid

      | "Concat" =>
        if inps.size < 2 then throw s!"{ctx}: Concat requires at least 2 inputs, got {inps.size}"
        let some axAttr := nd.attr? "axis"
          | throw s!"{ctx}: Concat missing required axis attribute"
        let mut cInps : Array ValueId := Array.mkEmpty inps.size
        let mut cTys : Array TensorType := Array.mkEmpty inps.size
        for nm in inps do
          let (v, ty) ← lookupRuntime valOf tyOf nm ctx
          cInps := cInps.push v
          cTys := cTys.push ty
        let rank := cTys[0]!.shape.size
        for i in [1:cTys.size] do
          if cTys[i]!.shape.size != rank then
            throw s!"{ctx}: Concat input[{i}] rank {cTys[i]!.shape.size} != input[0] rank {rank}"
        if rank != 4 && rank != 3 && rank != 2 then
          throw s!"{ctx}: Concat requires 4D, 3D, or 2D inputs, got rank {rank}"
        let axInt := if axAttr.i < 0 then axAttr.i + (rank : Int) else axAttr.i
        if axInt < 0 || axInt >= (rank : Int) then
          throw s!"{ctx}: Concat axis {axAttr.i} out of range for rank {rank}"
        let axis := axInt.toNat
        if rank == 3 && axis != 2 then
          throw s!"{ctx}: 3D Concat axis {axAttr.i} not supported (only last axis 2/-1 is supported on 3D [H,M,K])"
        let mut sumDim : Nat := 0
        for ty in cTys do sumDim := sumDim + ty.shape[axis]!
        let outShape := cTys[0]!.shape.set! axis sumDim
        let outVid := nextId; nextId := nextId + 1
        let outTy : TensorType := { shape := outShape, dtype := .fp16 }
        nodes := nodes.push {
          name := lbl, op := .core (.concat axis), inputs := cInps, outputs := #[(outVid, outTy)]
        }
        tyOf := tyOf.push (outVid, outTy)
        valOf := setAssoc valOf out0 outVid

      | "Unsqueeze" =>
        if inps.size < 1 then throw s!"{ctx}: Unsqueeze missing data input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let rawAxes ← match nd.attrInts? "axes" with
          | some ax => pure ax
          | none =>
            if inps.size >= 2 && !inps[1]!.isEmpty then
              let some ac := lookupAssoc consts inps[1]!
                | throw s!"{ctx}: Unsqueeze axes '{inps[1]!}' must be a constant int64 tensor"
              ac.int64s
            else throw s!"{ctx}: Unsqueeze missing axes attribute/input"
        let outRank := xTy.shape.size + rawAxes.size
        let mut normAxes : Array Nat := Array.mkEmpty rawAxes.size
        for a0 in rawAxes do
          let aInt := if a0 < 0 then a0 + (outRank : Int) else a0
          if aInt < 0 || aInt >= (outRank : Int) then
            throw s!"{ctx}: Unsqueeze axis {a0} out of range for output rank {outRank}"
          let aNat := aInt.toNat
          if normAxes.contains aNat then
            throw s!"{ctx}: Unsqueeze duplicate axis {aNat}"
          normAxes := normAxes.push aNat
        let mut outShape : Array Nat := Array.mkEmpty outRank
        let mut xIdx : Nat := 0
        for i in [:outRank] do
          if normAxes.contains i then
            outShape := outShape.push 1
          else
            outShape := outShape.push xTy.shape[xIdx]!
            xIdx := xIdx + 1
        if outRank == 5 && xTy.shape.size == 4 && xTy.shape[0]! == 1 && normAxes == #[2] then
          -- Virtual 5D GQA `Unsqueeze(K[1, n_kv, L, D], axes=[2]) -> [1, n_kv, 1, L, D]`
          virt5D := virt5D.push (out0, xVid, xTy.shape, outShape, 2, 1)
          elidedCount := elidedCount + 1
        else if outRank == 2 || outRank == 3 || outRank == 4 then
          if (outRank == 2 || outRank == 4) && outShape[0]! != 1 then
            throw s!"{ctx}: Unsqueeze batch dimension N={outShape[0]!} != 1"
          let outVid := nextId; nextId := nextId + 1
          let outTy : TensorType := { shape := outShape, dtype := xTy.dtype }
          nodes := nodes.push {
            name := lbl, op := .core (.reshape outShape), inputs := #[xVid], outputs := #[(outVid, outTy)]
          }
          tyOf := tyOf.push (outVid, outTy)
          valOf := setAssoc valOf out0 outVid
        else
          throw s!"{ctx}: Unsqueeze to rank {outRank} ({outShape.toList}) is not supported"

      | "Squeeze" =>
        if inps.size < 1 then throw s!"{ctx}: Squeeze missing data input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let rank := xTy.shape.size
        let rawAxes ← match nd.attrInts? "axes" with
          | some ax => pure ax
          | none =>
            if inps.size >= 2 && !inps[1]!.isEmpty then
              let some ac := lookupAssoc consts inps[1]!
                | throw s!"{ctx}: Squeeze axes '{inps[1]!}' must be a constant int64 tensor"
              ac.int64s
            else pure #[]
        let mut sqAxes : Array Nat := #[]
        if rawAxes.isEmpty then
          for i in [:rank] do
            if xTy.shape[i]! == 1 then sqAxes := sqAxes.push i
        else
          for a0 in rawAxes do
            let aInt := if a0 < 0 then a0 + (rank : Int) else a0
            if aInt < 0 || aInt >= (rank : Int) then
              throw s!"{ctx}: Squeeze axis {a0} out of range for rank {rank}"
            let aNat := aInt.toNat
            if sqAxes.contains aNat then
              throw s!"{ctx}: Squeeze duplicate axis {aNat}"
            if xTy.shape[aNat]! != 1 then
              throw s!"{ctx}: Squeeze axis {aNat} has dim {xTy.shape[aNat]!} != 1"
            sqAxes := sqAxes.push aNat
        let mut outShape : Array Nat := #[]
        for i in [:rank] do
          if !sqAxes.contains i then outShape := outShape.push xTy.shape[i]!
        if outShape.size != 2 && outShape.size != 3 && outShape.size != 4 then
          throw s!"{ctx}: Squeeze output rank {outShape.size} ({outShape.toList}) not supported"
        if (outShape.size == 2 || outShape.size == 4) && outShape[0]! != 1 then
          throw s!"{ctx}: Squeeze output batch dimension N={outShape[0]!} != 1"
        let outVid := nextId; nextId := nextId + 1
        let outTy : TensorType := { shape := outShape, dtype := xTy.dtype }
        nodes := nodes.push {
          name := lbl, op := .core (.reshape outShape), inputs := #[xVid], outputs := #[(outVid, outTy)]
        }
        tyOf := tyOf.push (outVid, outTy)
        valOf := setAssoc valOf out0 outVid

      | "Expand" =>
        if inps.size != 2 then throw s!"{ctx}: Expand requires 2 inputs (input, shape)"
        let some sc := lookupAssoc consts inps[1]!
          | throw s!"{ctx}: Expand target shape '{inps[1]!}' must be a constant int64 tensor"
        let rawShape ← sc.int64s
        let mut targetShape : Array Nat := Array.mkEmpty rawShape.size
        for d in rawShape do
          if d <= 0 then throw s!"{ctx}: Expand target shape has non-positive dim {d}"
          targetShape := targetShape.push d.toNat
        match virt5D.find? fun e => e.1 == inps[0]! with
        | some (_, base4DVid, base4DShape, shape5D, unsqAx, _) =>
          let out5D ← match broadcastOnnxShapes shape5D targetShape with
            | .ok s => pure s
            | .error e => throw s!"{ctx}: {e}"
          if out5D.size != 5 then
            throw s!"{ctx}: 5D GQA Expand output rank {out5D.size} != 5"
          for i in [:5] do
            if i != unsqAx && out5D[i]! != shape5D[i]! then
              throw s!"{ctx}: 5D GQA Expand only supports expanding the unsqueezed repeat axis {unsqAx} (got {shape5D.toList} -> {out5D.toList})"
          virt5D := virt5D.push (out0, base4DVid, base4DShape, out5D, unsqAx, out5D[unsqAx]!)
          elidedCount := elidedCount + 1
        | none =>
          let (xVid0, xTy0) ← lookupRuntime valOf tyOf inps[0]! ctx
          let outShape ← match broadcastOnnxShapes xTy0.shape targetShape with
            | .ok s => pure s
            | .error e => throw s!"{ctx}: {e}"
          if outShape.size != 2 && outShape.size != 3 && outShape.size != 4 then
            throw s!"{ctx}: Expand output rank {outShape.size} ({outShape.toList}) not supported (expected 2D/3D/4D)"
          if (outShape.size == 2 || outShape.size == 4) && outShape[0]! != 1 then
            throw s!"{ctx}: Expand output batch dimension N={outShape[0]!} != 1"
          if outShape == xTy0.shape then
            aliases := setAssoc aliases out0 inps[0]!
            elidedCount := elidedCount + 1
          else
            let mut curVid := xVid0
            let mut curShape := xTy0.shape
            if curShape.size < outShape.size then
              let liftedShape := Array.replicate (outShape.size - curShape.size) 1 ++ curShape
              let lv := nextId; nextId := nextId + 1
              let lTy : TensorType := { shape := liftedShape, dtype := xTy0.dtype }
              nodes := nodes.push {
                name := s!"{lbl}_lift", op := .core (.reshape liftedShape), inputs := #[curVid], outputs := #[(lv, lTy)]
              }
              tyOf := tyOf.push (lv, lTy)
              curVid := lv; curShape := liftedShape
            let mut expandedCount : Nat := 0
            for ax in [:outShape.size] do
              if curShape[ax]! != outShape[ax]! then
                if curShape[ax]! != 1 then
                  throw s!"{ctx}: Expand cannot broadcast axis {ax} from {curShape[ax]!} to {outShape[ax]!}"
                let rep := outShape[ax]!
                let nextShape := curShape.set! ax rep
                let nv := nextId; nextId := nextId + 1
                let nTy : TensorType := { shape := nextShape, dtype := xTy0.dtype }
                nodes := nodes.push {
                  name := if expandedCount == 0 then lbl else s!"{lbl}_ax{ax}"
                  op := .core (.concat ax)
                  inputs := Array.replicate rep curVid
                  outputs := #[(nv, nTy)]
                }
                tyOf := tyOf.push (nv, nTy)
                curVid := nv; curShape := nextShape; expandedCount := expandedCount + 1
            valOf := setAssoc valOf out0 curVid

      | "Tile" =>
        if inps.size != 2 then throw s!"{ctx}: Tile requires 2 inputs (input, repeats)"
        let (xVid0, xTy0) ← lookupRuntime valOf tyOf inps[0]! ctx
        let some rc := lookupAssoc consts inps[1]!
          | throw s!"{ctx}: Tile repeats '{inps[1]!}' must be a constant int64 tensor"
        let rawReps ← rc.int64s
        if rawReps.size != xTy0.shape.size then
          throw s!"{ctx}: Tile repeats length {rawReps.size} != input rank {xTy0.shape.size}"
        let mut reps : Array Nat := Array.mkEmpty rawReps.size
        for r in rawReps do
          if r <= 0 then throw s!"{ctx}: Tile repeat {r} must be >= 1"
          reps := reps.push r.toNat
        if reps.all (· == 1) then
          aliases := setAssoc aliases out0 inps[0]!
          elidedCount := elidedCount + 1
        else
          let mut curVid := xVid0
          let mut curShape := xTy0.shape
          let mut stepCnt : Nat := 0
          for ax in [:reps.size] do
            let r := reps[ax]!
            if r > 1 then
              if (curShape.size == 2 || curShape.size == 4) && ax == 0 then
                throw s!"{ctx}: Tile cannot repeat batch axis 0 (N must stay 1)"
              let nextShape := curShape.set! ax (curShape[ax]! * r)
              let nv := nextId; nextId := nextId + 1
              let nTy : TensorType := { shape := nextShape, dtype := xTy0.dtype }
              nodes := nodes.push {
                name := if stepCnt == 0 then lbl else s!"{lbl}_ax{ax}"
                op := .core (.concat ax)
                inputs := Array.replicate r curVid
                outputs := #[(nv, nTy)]
              }
              tyOf := tyOf.push (nv, nTy)
              curVid := nv; curShape := nextShape; stepCnt := stepCnt + 1
          valOf := setAssoc valOf out0 curVid

      | "Resize" | "Upsample" =>
        if inps.size < 1 then throw s!"{ctx}: {nd.opType} missing data input"
        let (xVid0, xTy0) ← lookupRuntime valOf tyOf inps[0]! ctx
        if xTy0.shape.size != 4 || xTy0.shape[0]! != 1 then
          throw s!"{ctx}: {nd.opType} requires 4D [1,C,H,W] input, got {xTy0.shape.toList}"
        let mode := (← nd.attrStringChecked "mode" "nearest").toLower
        if mode != "nearest" then
          throw s!"{ctx}: {nd.opType} mode '{mode}' is not supported (only mode='nearest' is supported)"
        if nd.opType == "Resize" then
          if inps.size >= 2 && !inps[1]!.isEmpty then
            throw s!"{ctx}: Resize non-empty roi input '{inps[1]!}' is not supported"
          if let some axAttr := nd.attrInts? "axes" then
            if !axAttr.isEmpty then
              throw s!"{ctx}: Resize non-empty axes attribute {axAttr.toList} is not supported"
          if nd.attrInt "exclude_outside" 0 != 0 then
            throw s!"{ctx}: Resize exclude_outside != 0 is not supported"
          let policy := (← nd.attrStringChecked "keep_aspect_ratio_policy" "").toLower
          if !policy.isEmpty && policy != "stretch" then
            throw s!"{ctx}: Resize keep_aspect_ratio_policy '{policy}' is not supported (only 'stretch' is supported)"
          let coordMode := (← nd.attrStringChecked "coordinate_transformation_mode" "half_pixel").toLower
          let nearMode := (← nd.attrStringChecked "nearest_mode" "round_prefer_floor").toLower
          let exactFloorDup : Bool :=
            ((coordMode == "half_pixel" || coordMode == "pytorch_half_pixel") &&
              (nearMode == "round_prefer_floor" || nearMode == "round_prefer_ceil")) ||
            (coordMode == "asymmetric" && nearMode == "floor")
          if !exactFloorDup then
            throw s!"{ctx}: Resize coordinate_transformation_mode='{coordMode}' with nearest_mode='{nearMode}' is not supported (only half_pixel/pytorch_half_pixel + round_prefer_floor/round_prefer_ceil or asymmetric + floor match exact integer pixel duplication)"
        let c := xTy0.shape[1]!; let h := xTy0.shape[2]!; let w := xTy0.shape[3]!
        let parseIntScale := fun (f : Float) (axLbl : String) => do
          if f.isNaN || f.isInf || f < 1.0 then
            throw s!"{ctx}: {nd.opType} {axLbl} scale {f} must be a positive integer >= 1"
          let n := (f + 0.5).toUInt64.toNat
          if Float.abs (f - Float.ofNat n) > 1e-4 || n == 0 then
            throw s!"{ctx}: {nd.opType} {axLbl} scale {f} must be an exact positive integer"
          pure n
        let (sh, sw) ←
          if nd.opType == "Upsample" then
            let scVals ←
              if inps.size >= 2 && !inps[1]!.isEmpty then
                let some cs := lookupAssoc consts inps[1]!
                  | throw s!"{ctx}: Upsample scales must be constant"
                cs.float64s
              else if let some fs := nd.attrFloats? "scales" then pure fs
              else throw s!"{ctx}: Upsample missing scales"
            if scVals.size != 4 then throw s!"{ctx}: Upsample scales length {scVals.size} != 4"
            let sn ← parseIntScale scVals[0]! "N"
            let sc ← parseIntScale scVals[1]! "C"
            let sh1 ← parseIntScale scVals[2]! "H"
            let sw1 ← parseIntScale scVals[3]! "W"
            if sn != 1 || sc != 1 then throw s!"{ctx}: Upsample only supports spatial (H,W) scaling (got N={sn}, C={sc})"
            pure (sh1, sw1)
          else
            if inps.size >= 4 && !inps[3]!.isEmpty then
              let some szCs := lookupAssoc consts inps[3]!
                | throw s!"{ctx}: Resize sizes '{inps[3]!}' must be a constant int64 tensor"
              let szs ← szCs.int64s
              if szs.size != 4 then throw s!"{ctx}: Resize sizes length {szs.size} != 4"
              if szs[0]! != 1 || szs[1]! != (c : Int) || szs[2]! <= 0 || szs[3]! <= 0 then
                throw s!"{ctx}: Resize sizes {szs.toList} incompatible with input [1,{c},{h},{w}]"
              let oh := szs[2]!.toNat; let ow := szs[3]!.toNat
              if oh % h != 0 || ow % w != 0 then
                throw s!"{ctx}: Resize nearest sizes ({oh}x{ow}) must be integer multiples of input ({h}x{w})"
              pure (oh / h, ow / w)
            else if inps.size >= 3 && !inps[2]!.isEmpty then
              let some scCs := lookupAssoc consts inps[2]!
                | throw s!"{ctx}: Resize scales '{inps[2]!}' must be a constant float tensor"
              let scVals ← scCs.float64s
              if scVals.size != 4 then throw s!"{ctx}: Resize scales length {scVals.size} != 4"
              let sn ← parseIntScale scVals[0]! "N"
              let sc ← parseIntScale scVals[1]! "C"
              let sh1 ← parseIntScale scVals[2]! "H"
              let sw1 ← parseIntScale scVals[3]! "W"
              if sn != 1 || sc != 1 then throw s!"{ctx}: Resize only supports spatial (H,W) scaling (got N={sn}, C={sc})"
              pure (sh1, sw1)
            else
              throw s!"{ctx}: Resize requires either constant 'scales' (input 2) or 'sizes' (input 3)"
        if sh == 1 && sw == 1 then
          aliases := setAssoc aliases out0 inps[0]!
          elidedCount := elidedCount + 1
        else
          -- Exact pure-CoreOp nearest-neighbor upsampling (`W -> sw*W` first while H is small, then `H -> sh*H`)!
          let emitWUpsampleBlock := fun (inV : ValueId) (bh bw : Nat) (tag : String)
              (nds : Array (IRNode HighOp)) (tm : Array (ValueId × TensorType)) (nid : ValueId) =>
            let t1 := nid; let t2 := nid + 1; let t3 := nid + 2; let t4 := nid + 3; let t5 := nid + 4
            let ty1 : TensorType := { shape := #[1, c, bw, bh], dtype := xTy0.dtype }
            let ty2 : TensorType := { shape := #[1, c * bw, 1, bh], dtype := xTy0.dtype }
            let ty3 : TensorType := { shape := #[1, c * bw, sw, bh], dtype := xTy0.dtype }
            let ty4 : TensorType := { shape := #[1, c, sw * bw, bh], dtype := xTy0.dtype }
            let ty5 : TensorType := { shape := #[1, c, bh, sw * bw], dtype := xTy0.dtype }
            let nds' := nds
              |>.push { name := s!"{tag}_tr1", op := .core (.transpose #[0, 1, 3, 2]), inputs := #[inV], outputs := #[(t1, ty1)] }
              |>.push { name := s!"{tag}_rs1", op := .core (.reshape ty2.shape), inputs := #[t1], outputs := #[(t2, ty2)] }
              |>.push { name := s!"{tag}_cat", op := .core (.concat 2), inputs := Array.replicate sw t2, outputs := #[(t3, ty3)] }
              |>.push { name := s!"{tag}_rs2", op := .core (.reshape ty4.shape), inputs := #[t3], outputs := #[(t4, ty4)] }
              |>.push { name := s!"{tag}_tr2", op := .core (.transpose #[0, 1, 3, 2]), inputs := #[t4], outputs := #[(t5, ty5)] }
            let tm' := tm.push (t1, ty1) |>.push (t2, ty2) |>.push (t3, ty3) |>.push (t4, ty4) |>.push (t5, ty5)
            (t5, nds', tm', nid + 5)

          let mut curVid := xVid0
          let mut curW := w
          if sw > 1 then
            if h <= 128 && sw * w <= 128 then
              let (vW, nds', tm', nid') := emitWUpsampleBlock curVid h w s!"{lbl}_w" nodes tyOf nextId
              curVid := vW; nodes := nds'; tyOf := tm'; nextId := nid'; curW := sw * w
            else
              -- Chunk H into <=128 bands and W into <=maxChunkW (multiple of 32) bands so every T-mode transpose has tc <= 128
              let hStep := min h 128
              let rawMaxW := 128 / sw
              let wStep := if sw * w <= 128 then w else (rawMaxW / 32) * 32
              if wStep == 0 || sw * wStep > 128 || (w > wStep && wStep % 32 != 0) then
                throw s!"{ctx}: Resize W scale {sw} with W={w} cannot be tiled into 64B-aligned <=128 chunks"
              let nH := ceilDiv h hStep
              let nW := ceilDiv w wStep
              let mut hBandVids : Array ValueId := Array.mkEmpty nH
              for hi in [:nH] do
                let y0 := hi * hStep; let y1 := min h (y0 + hStep); let bh := y1 - y0
                let mut wBandVids : Array ValueId := Array.mkEmpty nW
                for wi in [:nW] do
                  let x0 := wi * wStep; let x1 := min w (x0 + wStep); let bw := x1 - x0
                  let slV := nextId; nextId := nextId + 1
                  let slTy : TensorType := { shape := #[1, c, bh, bw], dtype := xTy0.dtype }
                  nodes := nodes.push {
                    name := s!"{lbl}_sl_{hi}_{wi}", op := .core (.slice #[0, 0, y0, x0] #[1, c, y1, x1] #[1, 1, 1, 1]),
                    inputs := #[curVid], outputs := #[(slV, slTy)]
                  }
                  tyOf := tyOf.push (slV, slTy)
                  let (vB, nds', tm', nid') := emitWUpsampleBlock slV bh bw s!"{lbl}_w_{hi}_{wi}" nodes tyOf nextId
                  nodes := nds'; tyOf := tm'; nextId := nid'
                  wBandVids := wBandVids.push vB
                let rowBandV ←
                  if wBandVids.size == 1 then pure wBandVids[0]!
                  else
                    let rv := nextId; nextId := nextId + 1
                    let rTy : TensorType := { shape := #[1, c, bh, sw * w], dtype := xTy0.dtype }
                    nodes := nodes.push {
                      name := s!"{lbl}_wcat_{hi}", op := .core (.concat 3), inputs := wBandVids, outputs := #[(rv, rTy)]
                    }
                    tyOf := tyOf.push (rv, rTy)
                    pure rv
                hBandVids := hBandVids.push rowBandV
              curVid ←
                if hBandVids.size == 1 then pure hBandVids[0]!
                else
                  let fv := nextId; nextId := nextId + 1
                  let fTy : TensorType := { shape := #[1, c, h, sw * w], dtype := xTy0.dtype }
                  nodes := nodes.push {
                    name := s!"{lbl}_hcat", op := .core (.concat 2), inputs := hBandVids, outputs := #[(fv, fTy)]
                  }
                  tyOf := tyOf.push (fv, fTy)
                  pure fv
              curW := sw * w
          if sh > 1 then
            let u1 := nextId; let u2 := nextId + 1; let u3 := nextId + 2; nextId := nextId + 3
            let ty1 : TensorType := { shape := #[1, c * h, 1, curW], dtype := xTy0.dtype }
            let ty2 : TensorType := { shape := #[1, c * h, sh, curW], dtype := xTy0.dtype }
            let ty3 : TensorType := { shape := #[1, c, sh * h, curW], dtype := xTy0.dtype }
            nodes := nodes
              |>.push { name := s!"{lbl}_hrs1", op := .core (.reshape ty1.shape), inputs := #[curVid], outputs := #[(u1, ty1)] }
              |>.push { name := s!"{lbl}_hdup", op := .core (.concat 2), inputs := Array.replicate sh u1, outputs := #[(u2, ty2)] }
              |>.push { name := lbl, op := .core (.reshape ty3.shape), inputs := #[u2], outputs := #[(u3, ty3)] }
            tyOf := tyOf.push (u1, ty1) |>.push (u2, ty2) |>.push (u3, ty3)
            curVid := u3
          valOf := setAssoc valOf out0 curVid

      | "Reshape" =>
        if inps.size != 2 then throw s!"{ctx}: Reshape requires 2 inputs (data, shape)"
        let some sc := lookupAssoc consts inps[1]!
          | throw s!"{ctx}: Reshape target shape '{inps[1]!}' must be a constant int64 tensor"
        let rawTarget ← sc.int64s
        let allowZero := nd.attrInt "allowzero" 0 != 0
        match virt5D.find? fun e => e.1 == inps[0]! with
        | some (_, base4DVid, base4DShape, shape5D, _, rep) =>
          let outShape ← resolveReshapeShapeIR shape5D rawTarget allowZero ctx
          let nKv := base4DShape[1]!; let lDim := base4DShape[2]!; let dDim := base4DShape[3]!
          if outShape != #[1, nKv * rep, lDim, dDim] then
            throw s!"{ctx}: 5D GQA Reshape target {outShape.toList} != expected [1,{nKv * rep},{lDim},{dDim}]"
          if rep == 1 then
            valOf := setAssoc valOf out0 base4DVid
            elidedCount := elidedCount + 1
          else if nKv == 1 then
            let outVid := nextId; nextId := nextId + 1
            let outTy : TensorType := { shape := outShape, dtype := .fp16 }
            nodes := nodes.push {
              name := lbl, op := .core (.concat 1), inputs := Array.replicate rep base4DVid, outputs := #[(outVid, outTy)]
            }
            tyOf := tyOf.push (outVid, outTy)
            valOf := setAssoc valOf out0 outVid
          else
            -- Interleaved GQA head repeat `[h0, ..., h0, h1, ..., h1, ...]` via per-head `slice` + single `concat(axis=1)`
            let mut catInps : Array ValueId := Array.mkEmpty (nKv * rep)
            let headTy : TensorType := { shape := #[1, 1, lDim, dDim], dtype := .fp16 }
            for hi in [:nKv] do
              let hVid := nextId; nextId := nextId + 1
              nodes := nodes.push {
                name := s!"{lbl}_kv{hi}"
                op := .core (.slice #[0, hi, 0, 0] #[1, hi + 1, lDim, dDim] #[1, 1, 1, 1])
                inputs := #[base4DVid]
                outputs := #[(hVid, headTy)]
              }
              tyOf := tyOf.push (hVid, headTy)
              for _ in [:rep] do
                catInps := catInps.push hVid
            let outVid := nextId; nextId := nextId + 1
            let outTy : TensorType := { shape := outShape, dtype := .fp16 }
            nodes := nodes.push {
              name := lbl, op := .core (.concat 1), inputs := catInps, outputs := #[(outVid, outTy)]
            }
            tyOf := tyOf.push (outVid, outTy)
            valOf := setAssoc valOf out0 outVid
        | none =>
          let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
          let outShape ← resolveReshapeShapeIR xTy.shape rawTarget allowZero ctx
          let outVid := nextId; nextId := nextId + 1
          let outTy : TensorType := { shape := outShape, dtype := .fp16 }
          nodes := nodes.push {
            name := lbl, op := .core (.reshape outShape), inputs := #[xVid], outputs := #[(outVid, outTy)]
          }
          tyOf := tyOf.push (outVid, outTy)
          valOf := setAssoc valOf out0 outVid

      | "Flatten" =>
        if inps.size != 1 then throw s!"{ctx}: Flatten requires 1 input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let ax0 := nd.attrInt "axis" 1
        let (d0, d1) ← match computeFlattenShape xTy.shape ax0 with
          | .ok p => pure p
          | .error e => throw s!"{ctx}: {e}"
        if d0 != 1 then
          throw s!"{ctx}: Flatten on {xTy.shape.toList} with axis={ax0} produces batch dimension {d0} != 1 (only [1, C] output is supported)"
        let outShape := #[1, d1]
        let outVid := nextId; nextId := nextId + 1
        let outTy : TensorType := { shape := outShape, dtype := .fp16 }
        nodes := nodes.push {
          name := lbl, op := .core (.reshape outShape), inputs := #[xVid], outputs := #[(outVid, outTy)]
        }
        tyOf := tyOf.push (outVid, outTy)
        valOf := setAssoc valOf out0 outVid

      | "Slice" =>
        if inps.size < 1 then throw s!"{ctx}: Slice missing data input"
        let (xVid, xTy) ← lookupRuntime valOf tyOf inps[0]! ctx
        let rank := xTy.shape.size
        if rank != 4 && rank != 3 && rank != 2 then
          throw s!"{ctx}: Slice requires 4D, 3D, or 2D input, got rank {rank}"
        let (starts, ends, axes0, steps0) ←
          if inps.size >= 3 then
            let readI64 := fun (slot : Nat) (label : String) => do
              let nm := inps[slot]!
              let some cs := lookupAssoc consts nm
                | throw s!"{ctx}: Slice {label} '{nm}' must be a constant int64 tensor"
              cs.int64s
            let st ← readI64 1 "starts"
            let en ← readI64 2 "ends"
            let ax ← if inps.size >= 4 && !inps[3]!.isEmpty then readI64 3 "axes" else pure #[]
            let sp ← if inps.size >= 5 && !inps[4]!.isEmpty then readI64 4 "steps" else pure #[]
            pure (st, en, ax, sp)
          else
            let some st := nd.attrInts? "starts" | throw s!"{ctx}: Slice missing starts"
            let some en := nd.attrInts? "ends" | throw s!"{ctx}: Slice missing ends"
            let ax := (nd.attrInts? "axes").getD #[]
            pure (st, en, ax, #[])
        if starts.size != ends.size || starts.isEmpty then
          throw s!"{ctx}: Slice starts ({starts.size}) and ends ({ends.size}) length mismatch"
        let axes : Array Int ←
          if axes0.isEmpty then pure (Array.ofFn (n := starts.size) fun i => (i.val : Int))
          else if axes0.size != starts.size then throw s!"{ctx}: Slice axes length {axes0.size} != starts length {starts.size}"
          else pure axes0
        let steps : Array Int ←
          if steps0.isEmpty then pure (Array.replicate starts.size (1 : Int))
          else if steps0.size != starts.size then throw s!"{ctx}: Slice steps length {steps0.size} != starts length {starts.size}"
          else pure steps0

        let mut beg := Array.replicate rank 0
        let mut en := xTy.shape
        let mut st := Array.replicate rank 1
        let mut seen := Array.replicate rank false
        for i in [:starts.size] do
          let axRaw := axes[i]!
          let axInt := if axRaw < 0 then axRaw + (rank : Int) else axRaw
          if axInt < 0 || axInt >= (rank : Int) then
            throw s!"{ctx}: Slice axis {axRaw} out of range [0,{rank - 1}] for rank-{rank} input"
          let onnxAx := axInt.toNat
          if seen[onnxAx]! then throw s!"{ctx}: Slice duplicate axis {onnxAx}"
          seen := seen.set! onnxAx true
          if rank == 3 && onnxAx != 2 && onnxAx != 0 then
            throw s!"{ctx}: 3D Slice axis {axRaw} not supported (only axis 0 or last axis 2/-1 is supported on 3D [H,M,K])"
          let stInt := steps[i]!
          if stInt <= 0 then throw s!"{ctx}: Slice step {stInt} must be > 0"
          let dim : Int := (xTy.shape[onnxAx]! : Int)
          let mut s := starts[i]!
          let mut e := ends[i]!
          if s < 0 then s := s + dim
          if e < 0 then e := e + dim
          if s < 0 then s := 0 else if s > dim then s := dim
          if e < 0 then e := 0 else if e > dim then e := dim
          if e <= s then throw s!"{ctx}: Slice on axis {onnxAx} produces empty range [{s},{e})"
          beg := beg.set! onnxAx s.toNat
          en := en.set! onnxAx e.toNat
          st := st.set! onnxAx stInt.toNat
        let outShape := (sliceShape xTy.shape.toList beg en st).toArray
        let outVid := nextId; nextId := nextId + 1
        let outTy : TensorType := { shape := outShape, dtype := .fp16 }
        nodes := nodes.push {
          name := lbl, op := .core (.slice beg en st), inputs := #[xVid], outputs := #[(outVid, outTy)]
        }
        tyOf := tyOf.push (outVid, outTy)
        valOf := setAssoc valOf out0 outVid

      | other =>
        throw s!"{ctx}: unsupported ONNX operator '{other}'"

  let mut outVids : Array ValueId := Array.mkEmpty g.outputs.size
  for oi in [:g.outputs.size] do
    let graphOutputName := resolveAliasStr aliases g.outputs[oi]!.name
    let (ov, _) ← lookupRuntime valOf tyOf graphOutputName s!"graph output[{oi}]"
    outVids := outVids.push ov

  let highGraph : SSAGraph HighOp := {
    target := opts.target
    inputs := ssaInputs
    nodes := nodes
    outputs := outVids
  }
  pure (highGraph, elidedCount, sfUsed)

-- ============================================================================
-- End-to-End ONNX -> L1 TensorIR -> L2 AneIR -> L3 MachineIR -> L4 SyncAndEmit
-- ============================================================================

/-- Internal lowering driver: when `requireCompatSpec = true`, single-output models must also
    lower cleanly to legacy `GraphSpec` (`lowerOnnx` compatibility); when `false` (`compileOnnx`),
    models using L2-only capabilities (constant materialization, scratch transpose, multi-output)
    compile directly through `AneProgram` (L2 -> L3 -> L4). -/
private def lowerOnnxInternal (model : OnnxModel) (opts : LowerOptions := {})
    (requireCompatSpec : Bool := true) : Except String LowerResult := do
  let (highGraph, decodeElided, sfUsed) ← decodeOnnxToHighGraph model opts
  let pipeCfg : PassConfig := {
    enableFusion := !opts.noFuse
    allowViewOps := opts.allowViewOps
    quantize := opts.quantize
  }
  let pipeRes ← runStandardPipeline highGraph pipeCfg
  let outNames := model.graph.outputs.map fun o => o.name
  let aneProg ← Vane.Backend.LayoutSelect.lowerCoreToAne pipeRes.coreGraph {
    placement := opts.placement
    allowViewOps := opts.allowViewOps
    outputNames := outNames
  }

  -- Validate lowered output ONNX shapes against declared static output dimensions for ALL outputs.
  let mut outShapes4D : Array Shape4D := Array.mkEmpty model.graph.outputs.size
  for oi in [:model.graph.outputs.size] do
    let declOut := model.graph.outputs[oi]!
    let outLogical := aneProg.outputLogicalShapes[oi]!
    if declOut.hasShape && !declOut.dims.isEmpty then
      if outLogical.size != declOut.dims.size then
        throw s!"graph output '{declOut.name}': lowered ONNX shape {outLogical.toList} (rank {outLogical.size}) != declared output rank {declOut.dims.size}"
      for i in [:declOut.dims.size] do
        let d := declOut.dims[i]!
        if d.hasValue && d.value > 0 then
          if outLogical[i]! != d.value.toNat then
            throw s!"graph output '{declOut.name}': lowered ONNX shape {outLogical.toList} != declared output dim[{i}]={d.value}"
    let oSh4 ← logicalToShape4D outLogical s!"graph output '{declOut.name}'"
    outShapes4D := outShapes4D.push oSh4

  -- Build backwards-compatible `GraphSpec` for single-output models (`--emit-graph-dir` & legacy tests);
  -- multi-output models (and L2-only features such as constant materialization or scratch transpose)
  -- compile directly through `aneProgram` (L2 -> L3 -> L4).
  let mkAneCompatSpec := fun () =>
    let gInps : Array GraphInput := aneProg.graph.inputs.map fun (_, nm, ty) =>
      let sh := (tensorTypeToShape4D ty nm).toOption.getD { n := 1, c := 1, h := 1, w := 1 }
      { name := nm, shape := shape4DToDims sh }
    let dummySpec : GraphSpec := {
      target := opts.target
      inputs := gInps
      nodes := #[]
      defaultPlacement := opts.placement
      stripMode := opts.stripMode
      outputs := outNames
    }
    (dummySpec, gInps, outNames[0]!)

  let (compatSpec, compatInputs, primaryOutId, hasCompatSpec) ←
    if pipeRes.coreGraph.outputs.size == 1 then
      if requireCompatSpec then
        let bridged ← coreGraphToGraphSpec pipeRes.coreGraph opts.placement opts.stripMode opts.allowViewOps
        pure (bridged.spec, bridged.spec.inputs, bridged.outputId, true)
      else
        match coreGraphToGraphSpec pipeRes.coreGraph opts.placement opts.stripMode opts.allowViewOps with
        | .ok bridged => pure (bridged.spec, bridged.spec.inputs, bridged.outputId, true)
        | .error _ =>
          let (ds, gi, po) := mkAneCompatSpec ()
          pure (ds, gi, po, false)
    else
      let (ds, gi, po) := mkAneCompatSpec ()
      pure (ds, gi, po, false)

  let summary : LowerSummary := {
    target := opts.target
    placement := opts.placement
    inputs := compatInputs
    output := primaryOutId
    outputShape := outShapes4D[0]!
    outputs := outNames
    outputShapes := outShapes4D
    onnxNodeCount := model.graph.nodes.size
    nodeCount := if hasCompatSpec then compatSpec.nodes.size else aneProg.graph.nodes.size
    fusedBnCount := pipeRes.stats.fusedBnCount
    fusedActCount := pipeRes.stats.fusedActCount
    fusedMatmulCount := pipeRes.stats.fusedMatmulCount
    elidedCount := decodeElided + pipeRes.stats.elidedCount
    weightCount := aneProg.weightCount
    weightBytes := aneProg.weightBytes
    safeTensorsUsed := sfUsed
    quantizedConvCount := pipeRes.stats.quantizedConvCount
    warnings := #[]
  }

  pure {
    spec := compatSpec
    summary := summary
    highGraph := pipeRes.highGraph
    coreGraph := pipeRes.coreGraph
    aneProgram := aneProg
  }

/-- Lower an `OnnxModel` into `LowerResult` (including legacy `GraphSpec` for single-output models). -/
def lowerOnnx (model : OnnxModel) (opts : LowerOptions := {}) : Except String LowerResult :=
  lowerOnnxInternal model opts true

/-- Lower an `OnnxModel` directly through L1 (`TensorIR`) → L2 (`AneIR` + SOK + rolling cascades) →
    L3 (`MachineIR` + `MemoryPlan`) → L4 (`SyncAndEmit`) into a verified HWX + 4 LOAD sections.
    Supports both single-output and multi-output (`1..3` outputs) ONNX models natively! -/
def compileOnnx (model : OnnxModel) (opts : LowerOptions := {}) : Except String CompiledOnnx := do
  let lowered ← lowerOnnxInternal model opts false
  let (finalAne, cm) ← compileAneProgram lowered.aneProgram opts.stripMode
  let compiled := compiledMachineToGraph cm (some finalAne)
  pure {
    lowered := lowered
    compiled := compiled
    aneProgram := finalAne
    machineProgram := cm.machine
  }

-- ============================================================================
-- Optional `graph.json` + fp16 Weight `.bin` Exporter (`--emit-graph-dir`)
-- ============================================================================

private def floatToJson (f : Float) : Json :=
  Lean.toJson f

private def lutSpecToJson (spec : LutSpec) : Json :=
  match spec with
  | .clip lo hi =>
    Json.mkObj [("name", Json.str "clip"), ("lo", floatToJson lo), ("hi", floatToJson hi)]
  | .leakyRelu alpha =>
    Json.mkObj [("name", Json.str "leaky_relu"), ("alpha", floatToJson alpha)]
  | .threshold a =>
    Json.mkObj [("name", Json.str "threshold"), ("scalar", floatToJson a)]
  | .thresholdedRelu a =>
    Json.mkObj [("name", Json.str "thresholded_relu"), ("alpha", floatToJson a)]
  | .hardSigmoid alpha beta =>
    Json.mkObj [("name", Json.str "hard_sigmoid"), ("alpha", floatToJson alpha), ("beta", floatToJson beta)]
  | .clampMax v =>
    Json.mkObj [("name", Json.str "clamp_max"), ("scalar", floatToJson v)]
  | .clampMin v =>
    Json.mkObj [("name", Json.str "clamp_min"), ("scalar", floatToJson v)]
  | .elu alpha =>
    Json.mkObj [("name", Json.str "elu"), ("alpha", floatToJson alpha)]
  | .scaledTanh alpha beta =>
    Json.mkObj [("name", Json.str "scaled_tanh"), ("alpha", floatToJson alpha), ("beta", floatToJson beta)]
  | .rsqrt eps =>
    if eps == 0.0 then Json.str "rsqrt"
    else Json.mkObj [("name", Json.str "rsqrt"), ("eps", floatToJson eps)]
  | other => Json.str other.name

private def neActSpecToJson : NeActSpec → Json
  | .relu => Json.str "relu"
  | .copy => Json.str "copy"
  | .lut s => lutSpecToJson s

private def convActFields (act : ConvActivation) : List (String × Json) :=
  match act with
  | .none => []
  | .relu => [("relu", Json.bool true)]
  | .lut s => [("act", lutSpecToJson s)]

/-- Write a lowered `LowerResult` to `outDir` as `graph.json` + little-endian fp16 weight `.bin` files
    compatible with both `vane compile-graph` and `vane-runner gen-graph`. -/
def emitGraphDir (outDir : System.FilePath) (res : LowerResult) : IO (Except String Unit) := do
  if res.summary.outputs.size > 1 then
    return .error "emitGraphDir does not yet support multi-output ONNX models"
  if res.spec.nodes.isEmpty && !res.aneProgram.graph.nodes.isEmpty then
    return .error "emitGraphDir does not support models requiring L2 constant materialization or scratch transpose"
  try
    IO.FS.createDirAll outDir
  catch e =>
    return .error s!"create graph dir '{outDir}': {e}"

  let spec := res.spec
  let mut nodeJsons : Array Json := Array.mkEmpty spec.nodes.size

  for nd in spec.nodes do
    let baseFields : List (String × Json) := [
      ("id", Json.str nd.id),
      ("inputs", Json.arr (nd.inputs.map Json.str))
    ]
    let placeFields : List (String × Json) := match nd.outPlacement with
      | some p => [("out_placement", Json.str p.name)]
      | none => []

    let opFields : List (String × Json) ← match nd.op with
      | .conv1x1 cin cout weights bias? act quant? | .linear cin cout weights bias? act quant? =>
        if quant?.isSome then
          return .error s!"emitGraphDir: quantized node '{nd.id}' is not supported by fp16 graph export"
        let opNm := match nd.op with | .linear .. => "linear" | _ => "conv1x1"
        let wFile := s!"{nd.id}_weights.bin"
        try IO.FS.writeBinFile (outDir / wFile) weights
        catch e => return .error s!"write '{wFile}': {e}"
        let bFields ← match bias? with
          | some bb =>
            let bFile := s!"{nd.id}_bias.bin"
            try IO.FS.writeBinFile (outDir / bFile) bb
            catch e => return .error s!"write '{bFile}': {e}"
            pure [("bias", Json.str bFile)]
          | none => pure []
        pure ([
          ("op", Json.str opNm),
          ("cin", Json.num cin),
          ("cout", Json.num cout),
          ("weights", Json.str wFile)
        ] ++ bFields ++ convActFields act)
      | .conv3x3 mode cin cout stride pad weights bias? act quant? =>
        if quant?.isSome then
          return .error s!"emitGraphDir: quantized node '{nd.id}' is not supported by fp16 graph export"
        let wFile := s!"{nd.id}_weights.bin"
        try IO.FS.writeBinFile (outDir / wFile) weights
        catch e => return .error s!"write '{wFile}': {e}"
        let bFields ← match bias? with
          | some bb =>
            let bFile := s!"{nd.id}_bias.bin"
            try IO.FS.writeBinFile (outDir / bFile) bb
            catch e => return .error s!"write '{bFile}': {e}"
            pure [("bias", Json.str bFile)]
          | none => pure []
        pure ([
          ("op", Json.str "conv3x3"),
          ("mode", Json.str mode.name),
          ("cin", Json.num cin),
          ("cout", Json.num cout),
          ("stride", Json.num stride),
          ("pad", Json.str pad.name),
          ("weights", Json.str wFile)
        ] ++ bFields ++ convActFields act)
      | .pe1 op =>
        match op with
        | .abs => pure [("op", Json.str "pe"), ("pe_op", Json.str "abs")]
        | .square => pure [("op", Json.str "pe"), ("pe_op", Json.str "square")]
        | .chain none (some b) none =>
          pure [("op", Json.str "pe"), ("pe_op", Json.str "sadd"), ("scalar", floatToJson (f16ToFloat b))]
        | .chain none none (some sc) =>
          pure [("op", Json.str "pe"), ("pe_op", Json.str "smul"), ("scalar", floatToJson (f16ToFloat sc))]
        | .chain ins b sc =>
          let f1 := match ins with | some v => [("in_scale", floatToJson (f16ToFloat v))] | none => []
          let f2 := match b with | some v => [("bias", floatToJson (f16ToFloat v))] | none => []
          let f3 := match sc with | some v => [("scale", floatToJson (f16ToFloat v))] | none => []
          pure ([("op", Json.str "pe"), ("pe_op", Json.str "chain")] ++ f1 ++ f2 ++ f3)
        | .binary _ _ _ _ => pure [("op", Json.str "pe"), ("pe_op", Json.str op.opName)]
      | .pe2 binOp rev =>
        let peNm := if rev && binOp == .sub then "rsub" else binOp.name
        pure [("op", Json.str "pe"), ("pe_op", Json.str peNm)]
      | .lut act =>
        pure [("op", Json.str "lut"), ("act", neActSpecToJson act)]
      | .pool kind kh kw sh sw pad excl =>
        let (padStr, customFields) := match pad with
          | .valid => ("valid", [])
          | .same => ("same", [])
          | .custom pt pb pl pr =>
            ("custom", [("pt", Json.num pt), ("pb", Json.num pb), ("pl", Json.num pl), ("pr", Json.num pr)])
        let exclFields := if kind == .avg then [("exclude_pad", Json.bool excl)] else []
        pure ([
          ("op", Json.str kind.opName),
          ("kh", Json.num kh), ("kw", Json.num kw),
          ("sh", Json.num sh), ("sw", Json.num sw),
          ("pad", Json.str padStr)
        ] ++ customFields ++ exclFields)
      | .reduce op axes eps =>
        let epsFields := if eps != 0 then [("eps", floatToJson (f16ToFloat eps))] else []
        pure ([
          ("op", Json.str "reduce"),
          ("reduce_op", Json.str op.name),
          ("axes", Json.str axes.name)
        ] ++ epsFields)
      | .transpose perm =>
        pure [("op", Json.str "transpose"), ("perm", Json.arr (perm.map fun (v : Nat) => Json.num v))]
      | .slice beg en step =>
        pure [
          ("op", Json.str "slice"),
          ("begin", Json.arr (beg.map fun (v : Nat) => Json.num v)),
          ("end", Json.arr (en.map fun (v : Nat) => Json.num v)),
          ("step", Json.arr (step.map fun (v : Nat) => Json.num v))
        ]
      | .concat axis =>
        pure [("op", Json.str "concat"), ("axis", Json.num axis)]
      | .reshape dims =>
        pure [("op", Json.str "reshape"), ("out_shape", Json.arr #[Json.num 1, Json.num dims.c, Json.num dims.h, Json.num dims.w])]
      | .matmul heads trA trB scaleBits =>
        pure [
          ("op", Json.str "matmul"),
          ("heads", Json.num heads),
          ("transpose_a", Json.bool trA),
          ("transpose_b", Json.bool trB),
          ("scale", floatToJson (f16ToFloat scaleBits))
        ]

    nodeJsons := nodeJsons.push (Json.mkObj (baseFields ++ opFields ++ placeFields))

  let inputJsons : Array Json := spec.inputs.map fun gi =>
    let snFields := if gi.shortName.isEmpty then [] else [("short_name", Json.str gi.shortName)]
    Json.mkObj ([
      ("name", Json.str gi.name),
      ("c", Json.num gi.shape.c),
      ("h", Json.num gi.shape.h),
      ("w", Json.num gi.shape.w)
    ] ++ snFields)

  let rootJson := Json.mkObj [
    ("target", Json.str spec.target.name),
    ("placement", Json.str spec.defaultPlacement.name),
    ("inputs", Json.arr inputJsons),
    ("nodes", Json.arr nodeJsons),
    ("output", Json.str res.summary.output)
  ]

  try
    IO.FS.writeFile (outDir / "graph.json") (rootJson.pretty ++ "\n")
  catch e =>
    return .error s!"write graph.json: {e}"
  return .ok ()

end Vane.Frontend.Lower
