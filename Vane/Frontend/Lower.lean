import Lean.Data.Json
import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Tiling
import Vane.Pe
import Vane.Lut
import Vane.Pool
import Vane.Conv
import Vane.View
import Vane.Matmul
import Vane.Graph
import Vane.Frontend.SafeTensors
import Vane.Frontend.Onnx

namespace Vane.Frontend.Lower

open Lean (Json)
open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Tiling
open Vane.Pe
open Vane.Lut
open Vane.Pool
open Vane.Conv
open Vane.View
open Vane.Matmul
open Vane.Graph
open Vane.Frontend.SafeTensors
open Vane.Frontend.Onnx

-- ============================================================================
-- Pure Classifiers & Layout Helpers (Exported for Formal Proofs)
-- ============================================================================

/-- Supported constant scalar exponent classifications for ONNX `Pow(x, exp)`. -/
inductive PowKind where
  | square      -- exp == 2.0  -> pe(mul, x, x)
  | identity    -- exp == 1.0  -> elided
  | sqrt        -- exp == 0.5  -> lut(sqrt)
  | rsqrt       -- exp == -0.5 -> lut(rsqrt)
  | reciprocal  -- exp == -1.0 -> lut(inverse)
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Exact IEEE-754 binary64 bit pattern for each supported `PowKind` exponent. -/
def powKindToF64Bits : PowKind → UInt64
  | .square     => 0x4000000000000000 --  2.0
  | .identity   => 0x3ff0000000000000 --  1.0
  | .sqrt       => 0x3fe0000000000000 --  0.5
  | .rsqrt      => 0xbfe0000000000000 -- -0.5
  | .reciprocal => 0xbff0000000000000 -- -1.0

/-- Classify an IEEE-754 binary64 exponent bit pattern into a supported `PowKind`. -/
def classifyPowBits (bits : UInt64) : Option PowKind :=
  if bits == 0x4000000000000000 then some .square
  else if bits == 0x3ff0000000000000 then some .identity
  else if bits == 0x3fe0000000000000 then some .sqrt
  else if bits == 0xbfe0000000000000 then some .rsqrt
  else if bits == 0xbff0000000000000 then some .reciprocal
  else none

/-- Validate and classify a floating-point `Pow` scalar exponent. -/
def classifyPowExponent (expVal : Float) (ctx : String) : Except String PowKind :=
  match classifyPowBits expVal.toBits with
  | some k => .ok k
  | none => .error s!"{ctx}: unsupported Pow exponent {expVal} (supported: 2.0, 1.0, 0.5, -0.5, -1.0)"

/-- Validated `Gelu` `approximate` attribute mode (only exact erf GELU is supported). -/
inductive GeluApprox where
  | exact       -- "" or "none"
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Validate the ONNX `Gelu` `approximate` attribute value.
    Rejects `"tanh"` and any unknown value instead of silently approximating. -/
def classifyGeluApprox (raw : String) : Except String GeluApprox :=
  match raw with
  | "" | "none" => .ok .exact
  | "tanh" => .error "Gelu: approximate='tanh' is not supported (only exact erf GELU with approximate='' or 'none' is supported)"
  | other => .error s!"Gelu: unsupported approximate='{other}' (expected '' or 'none')"

/-- Validate a `LayerNormalization` `Scale` or `B` tensor shape against the normalized suffix `X.shape[axis:]`.
    For `normAxis == 1` (with `H=W=1` on 4D or 2D `[1, C]`): accepts scalar all-1s, `[C]`,
    `[C, 1, 1]`, `[1, C, 1, 1]` (4D), or `[1, C]` (2D).
    For `normAxis == 3` (last axis `W` on 4D or `K` on 3D): accepts scalar all-1s, `[W]`,
    or `[1, ..., 1, W]` up to `inRank`. -/
def isValidLayerNormParamShape (dims : Array Nat) (normAxis normDim inRank : Nat) : Bool :=
  if dims.size > inRank then
    false
  else if dims.all (· == 1) then
    true
  else if normAxis == 1 then
    dims == #[normDim] ||
    (inRank == 4 && (dims == #[normDim, 1, 1] || dims == #[1, normDim, 1, 1])) ||
    (inRank == 2 && dims == #[1, normDim])
  else
    dims.size >= 1 && dims[dims.size - 1]! == normDim &&
    (dims.extract 0 (dims.size - 1)).all (· == 1)

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

-- ============================================================================
-- Public Lowering Options, Summary, Result & CLI Input-Shape Parser
-- ============================================================================

/-- Configuration options for lowering an ONNX model into `Vane.Graph.GraphSpec`. -/
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
  deriving BEq, Inhabited

/-- Summary statistics and diagnostics produced by `lowerOnnx`. -/
structure LowerSummary where
  target : Target
  placement : EdgePlacementMode
  inputs : Array GraphInput
  output : String
  outputShape : Shape4D
  onnxNodeCount : Nat
  nodeCount : Nat
  fusedBnCount : Nat
  fusedActCount : Nat
  fusedMatmulCount : Nat
  elidedCount : Nat
  weightCount : Nat
  weightBytes : Nat
  safeTensorsUsed : Nat
  warnings : Array String
  deriving Repr, BEq

/-- Result of pure Lean 4 ONNX lowering: in-memory `GraphSpec` plus `LowerSummary`. -/
structure LowerResult where
  spec : GraphSpec
  summary : LowerSummary
  deriving BEq

/-- Result of full end-to-end ONNX compilation (`lowerOnnx` + `compileGraph`). -/
structure CompiledOnnx where
  lowered : LowerResult
  compiled : CompiledGraph

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
-- Internal Constant & IR Representation
-- ============================================================================

private inductive ConstPayload where
  | onnx (t : OnnxTensor)
  | safetensors (sf : SafeTensorsFile)
  | inlineF64 (vals : Array Float)
  | inlineI64 (vals : Array Int)
  deriving Inhabited

private structure ConstSource where
  name : String
  dims : Array Nat
  payload : ConstPayload
  deriving Inhabited

private def ConstSource.isFromSafeTensors (c : ConstSource) : Bool :=
  match c.payload with
  | .safetensors _ => true
  | _ => false

private def ConstSource.float64s (c : ConstSource) : Except String (Array Float) :=
  match c.payload with
  | .inlineF64 vals => pure vals
  | .safetensors sf => sf.toFloat64 c.name
  | .onnx t => t.toFloat64
  | .inlineI64 _ => throw s!"constant '{c.name}' has no float data"

private def ConstSource.int64s (c : ConstSource) : Except String (Array Int) :=
  match c.payload with
  | .inlineI64 vals => pure vals
  | .safetensors sf => sf.toInt64 c.name
  | .onnx t => t.toInt64
  | .inlineF64 _ => throw s!"constant '{c.name}' has no integer data"

private def ConstSource.isScalar (c : ConstSource) : Except String Bool := do
  let n ← shapeProductChecked c.dims s!"constant '{c.name}'"
  pure (n == 1)

private structure TensorShapeInfo where
  vane : Shape4D
  onnx : Array Nat
  deriving Repr, BEq, Inhabited

private structure IRNode where
  orig : OnnxNode
  opType : String
  name : String
  inputs : Array String
  outputs : Array String
  foldedW : Option (Array Float) := none
  foldedWSh : Array Nat := #[]
  foldedB : Option (Array Float) := none
  fusedAct : ConvActivation := .none
  mmTransposeA : Bool := false
  mmTransposeB : Bool := false
  mmFoldedTrAPerm : Option (Array Int) := none
  mmFoldedTrBPerm : Option (Array Int) := none
  mmScale : Float := 1.0
  mmFrom4D : Bool := false
  deriving Inhabited

private def IRNode.primaryOutput (nd : IRNode) : String :=
  nd.outputs[0]?.getD ""

private def sanitizeNodeID (raw : String) (used : Array String) : String × Array String :=
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

private def lookupAssoc {α : Type} (arr : Array (String × α)) (key : String) : Option α :=
  (arr.find? fun p => p.1 == key).map Prod.snd

private def setAssoc {α : Type} (arr : Array (String × α)) (key : String) (val : α) : Array (String × α) :=
  match arr.findIdx? fun p => p.1 == key with
  | some idx => arr.set! idx (key, val)
  | none => arr.push (key, val)

private def resolveAlias (aliases : Array (String × String)) (name : String) : String := Id.run do
  let mut cur := name
  for _ in [:256] do
    match lookupAssoc aliases cur with
    | some nxt =>
      if nxt == cur then return cur
      cur := nxt
    | none => return cur
  return cur

private def nodeLabel (nd : IRNode) (idx : Nat) : String :=
  if !nd.name.isEmpty then nd.name
  else if !nd.primaryOutput.isEmpty then nd.primaryOutput
  else s!"{nd.opType.toLower}_{idx}"

private def extractConstantNode (nd : OnnxNode) : Except String ConstSource := do
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

-- ============================================================================
-- Graph Reachability & Consumer Counting (Multi-Output Aware)
-- ============================================================================

private def buildProducerMap (nodes : Array IRNode) : Array (String × Nat) := Id.run do
  let mut m : Array (String × Nat) := Array.mkEmpty nodes.size
  for i in [:nodes.size] do
    for o in nodes[i]!.outputs do
      if !o.isEmpty then
        m := setAssoc m o i
  return m

private def pruneUnreachable (nodes : Array IRNode) (graphOutput : String) : Array IRNode := Id.run do
  let prodMap := buildProducerMap nodes
  let mut live : Array Bool := Array.replicate nodes.size false
  let mut stack : Array String := #[graphOutput]
  -- Each node is visited at most once, so at most `nodes.size * maxInputs + 1` pops.
  let maxSteps := nodes.foldl (fun acc nd => acc + nd.inputs.size + 1) 1
  for _ in [:maxSteps] do
    if stack.isEmpty then break
    let cur := stack.back!
    stack := stack.pop
    if let some pIdx := lookupAssoc prodMap cur then
      if !live[pIdx]! then
        live := live.set! pIdx true
        for inp in nodes[pIdx]!.inputs do
          if !inp.isEmpty then
            stack := stack.push inp
  let mut out : Array IRNode := #[]
  for i in [:nodes.size] do
    if live[i]! then
      out := out.push nodes[i]!
  return out

private def countConsumers (nodes : Array IRNode) (graphOutput : String) : Array (String × Nat) := Id.run do
  let mut counts : Array (String × Nat) := #[]
  let bump := fun (m : Array (String × Nat)) (k : String) =>
    if k.isEmpty then m
    else
      match lookupAssoc m k with
      | some c => setAssoc m k (c + 1)
      | none => m.push (k, 1)
  for nd in nodes do
    for inp in nd.inputs do
      counts := bump counts inp
  counts := bump counts graphOutput
  return counts

private def consumerCount (counts : Array (String × Nat)) (name : String) : Nat :=
  (lookupAssoc counts name).getD 0

-- ============================================================================
-- Validation & Geometry Helpers
-- ============================================================================

private def validateFiniteFp16Scalar (v : Float) (ctx : String) : Except String UInt16 := do
  if v.isNaN || v.isInf then
    throw s!"{ctx}: scalar {v} must be finite"
  let h := floatToF16CanonicalZero v
  if !f16IsFinite h then
    throw s!"{ctx}: scalar {v} overflows finite fp16 range"
  pure h

private def readOptionalScalarParam
    (nd : IRNode) (consts : Array (String × ConstSource))
    (inputIdx : Nat) (attrName ctx : String) : Except String (Option Float) := do
  if h : inputIdx < nd.inputs.size then
    let nm := nd.inputs[inputIdx]
    if !nm.isEmpty then
      let some cs := lookupAssoc consts nm
        | throw s!"{ctx}: input '{nm}' ({attrName}) must be a constant scalar"
      let vals ← cs.float64s
      if vals.size != 1 then
        throw s!"{ctx}: {attrName} must be scalar, got {vals.size} elements"
      return some vals[0]!
  match nd.orig.attr? attrName with
  | some a => pure (some a.f)
  | none => pure none

private def isChannelBiasShape (dims : Array Nat) (cout : Nat) : Bool :=
  if cout == 0 then false
  else if dims.size == 1 then dims[0]! == cout
  else if dims.size == 2 then dims[0]! == 1 && dims[1]! == cout
  else if dims.size == 4 then dims[0]! == 1 && dims[1]! == cout && dims[2]! == 1 && dims[3]! == 1
  else false

private def computeBNAffine
    (bn : IRNode) (consts : Array (String × ConstSource)) (cout : Nat) (ctx : String) :
    Except String (Array Float × Array Float) := do
  if bn.inputs.size < 5 then
    throw s!"{ctx}: BatchNormalization requires 5 inputs"
  if bn.orig.attrInt "training_mode" 0 != 0 then
    throw s!"{ctx}: BatchNormalization training_mode != 0 is not supported (inference mode only)"
  let getVec := fun (idx : Nat) (label : String) => do
    let nm := bn.inputs[idx]!
    let some cs := lookupAssoc consts nm
      | throw s!"{ctx}: BatchNormalization {label} '{nm}' must be a constant initializer"
    let v ← cs.float64s
    if v.size != cout then
      throw s!"{ctx}: BatchNormalization {label} length {v.size} != channels {cout}"
    pure v
  let scale ← getVec 1 "scale"
  let bVec ← getVec 2 "B"
  let mean ← getVec 3 "mean"
  let variance ← getVec 4 "var"
  let eps := bn.orig.attrFloat "epsilon" 1e-5
  if eps.isNaN || eps.isInf || eps < 0.0 then
    throw s!"{ctx}: BatchNormalization invalid epsilon {eps}"

  let mut factor : Array Float := Array.mkEmpty cout
  let mut bias : Array Float := Array.mkEmpty cout
  for o in [:cout] do
    let v := variance[o]!
    if v < 0.0 then
      throw s!"{ctx}: BatchNormalization negative variance[{o}]={v}"
    let denom := v + eps
    if denom <= 0.0 || denom.isNaN || denom.isInf then
      throw s!"{ctx}: BatchNormalization invalid variance[{o}]+eps={denom}"
    let f := scale[o]! / Float.sqrt denom
    if f.isNaN || f.isInf then
      throw s!"{ctx}: BatchNormalization non-finite scale factor at channel {o}"
    factor := factor.push f
    bias := bias.push (bVec[o]! - mean[o]! * f)
  pure (factor, bias)

private def resolveConv3x3Padding
    (h w stride : Nat) (autoPad : String) (pads : Array Int) (ctx : String) :
    Except String (PadMode × Nat × Nat) := do
  let (onnxHo, onnxWo, pt, pl, pb, pr) ← match autoPad with
    | "VALID" =>
      if h < 3 || w < 3 then
        throw s!"{ctx}: conv3x3 VALID requires H>=3 and W>=3, got ({h},{w})"
      pure ((h - 3) / stride + 1, (w - 3) / stride + 1, 0, 0, 0, 0)
    | "SAME_UPPER" =>
      let ho := ceilDiv h stride
      let wo := ceilDiv w stride
      let padH := if (ho - 1) * stride + 3 > h then (ho - 1) * stride + 3 - h else 0
      let padW := if (wo - 1) * stride + 3 > w then (wo - 1) * stride + 3 - w else 0
      let pt := padH / 2
      let pl := padW / 2
      pure (ho, wo, pt, pl, padH - pt, padW - pl)
    | "SAME_LOWER" =>
      let ho := ceilDiv h stride
      let wo := ceilDiv w stride
      let padH := if (ho - 1) * stride + 3 > h then (ho - 1) * stride + 3 - h else 0
      let padW := if (wo - 1) * stride + 3 > w then (wo - 1) * stride + 3 - w else 0
      let pt := (padH + 1) / 2
      let pl := (padW + 1) / 2
      pure (ho, wo, pt, pl, padH - pt, padW - pl)
    | "" | "NOTSET" =>
      if pads.size != 4 then
        throw s!"{ctx}: conv3x3 pads must have length 4, got {pads.size}"
      for p in pads do
        if p < 0 then
          throw s!"{ctx}: conv3x3 negative pad {p}"
      let pt := pads[0]!.toNat
      let pl := pads[1]!.toNat
      let pb := pads[2]!.toNat
      let pr := pads[3]!.toNat
      if h + pt + pb < 3 || w + pl + pr < 3 then
        throw s!"{ctx}: conv3x3 padded spatial dims ({h + pt + pb},{w + pl + pr}) < 3"
      pure ((h + pt + pb - 3) / stride + 1, (w + pl + pr - 3) / stride + 1, pt, pl, pb, pr)
    | other =>
      throw s!"{ctx}: unsupported auto_pad '{other}'"

  -- Match Vane's `valid` mode
  if h >= 3 && w >= 3 then
    let shValid : ConvSpec := { kernelSize := 3, mode := .dense, h := h, w := w, cin := 1, cout := 1, stride := stride, pad := .valid }
    if onnxHo == shValid.outH && onnxWo == shValid.outW && pt == 0 && pl == 0 && pb == 0 && pr == 0 then
      return (.valid, shValid.outH, shValid.outW)

  -- Match Vane's `same` mode (SAME_UPPER)
  let shSame : ConvSpec := { kernelSize := 3, mode := .dense, h := h, w := w, cin := 1, cout := 1, stride := stride, pad := .same }
  let padTotalH := if (shSame.outH - 1) * stride + 3 > h then (shSame.outH - 1) * stride + 3 - h else 0
  let padTotalW := if (shSame.outW - 1) * stride + 3 > w then (shSame.outW - 1) * stride + 3 - w else 0
  let ptS := shSame.padTop
  let plS := shSame.padLeft
  let pbS := padTotalH - ptS
  let prS := padTotalW - plS
  if onnxHo == shSame.outH && onnxWo == shSame.outW && pt == ptS && pl == plS && pb == pbS && pr == prS then
    return (.same, shSame.outH, shSame.outW)

  throw s!"{ctx}: conv3x3 padding (auto_pad='{autoPad}', pads=[{pt},{pl},{pb},{pr}], out={onnxHo}x{onnxWo}) for {h}x{w} stride={stride} does not match Vane conv3x3 'valid' or 'same'"

private def resolveReduceAxes4D (rawAxes : Array Int) (inSh : Shape4D) (rank : Nat) (ctx : String) :
    Except String (ReduceAxes × Shape4D) := do
  if rawAxes.isEmpty then
    throw s!"{ctx}: reduce requires at least one axis"
  let mut rc := false
  let mut rh := false
  let mut rw := false
  for ax0 in rawAxes do
    let ax := if ax0 < 0 then ax0 + (rank : Int) else ax0
    if rank == 3 then
      -- 3D `[H, M, K]` is represented as `[1, H*M, 1, K]`: axis 2 maps to W.
      if ax == 2 then rw := true
      else throw s!"{ctx}: 3D reduce axis {ax0} not supported (expected axis -1 or 2)"
    else if rank == 2 then
      if ax == 1 then rc := true
      else throw s!"{ctx}: 2D reduce axis {ax0} not supported (expected axis -1 or 1)"
    else
      if ax == 1 then rc := true
      else if ax == 2 then rh := true
      else if ax == 3 then rw := true
      else throw s!"{ctx}: reduce axis {ax0} not supported (expected subset of \{1,2,3})"
  let axes : ReduceAxes := { reduceC := rc, reduceH := rh, reduceW := rw }
  if axes.isEmpty then
    throw s!"{ctx}: reduce resolved to empty axes"
  let outSh : Shape4D := {
    n := 1
    c := if rc then 1 else inSh.c
    h := if rh then 1 else inSh.h
    w := if rw then 1 else inSh.w
  }
  pure (axes, outSh)

private def validateReduceBackendForNode (target : Target) (redOp : ReduceOp) (axes : ReduceAxes) (inSh : Shape4D) (ctx : String) : Except String Unit := do
  let rspec : ReduceSpec := {
    target := target
    op := redOp
    dims := { c := inSh.c, h := inSh.h, w := inSh.w }
    axes := axes
    eps := 0
  }
  match resolveReduceBackend rspec with
  | .ok _ => pure ()
  | .error e => throw s!"{ctx}: {e}"

private def resolveReshapeTarget (inInfo : TensorShapeInfo) (raw : Array Int) (allowZero : Bool) (ctx : String) :
    Except String TensorShapeInfo := do
  if raw.isEmpty then
    throw s!"{ctx}: empty Reshape target shape"
  let total := inInfo.vane.numElems
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
          if h : i < inInfo.onnx.size then
            pure inInfo.onnx[i]
          else
            throw s!"{ctx}: Reshape dim 0 at index {i} out of range for input rank {inInfo.onnx.size}"
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
    pure {
      vane := { n := 1, c := resolved[1]!, h := resolved[2]!, w := resolved[3]! }
      onnx := resolved
    }
  else if resolved.size == 3 then
    pure {
      vane := { n := 1, c := resolved[0]! * resolved[1]!, h := 1, w := resolved[2]! }
      onnx := resolved
    }
  else if resolved.size == 2 then
    if resolved[0]! != 1 then
      throw s!"{ctx}: 2D Reshape output batch dimension N must be 1, got {resolved[0]!}"
    pure {
      vane := { n := 1, c := resolved[1]!, h := 1, w := 1 }
      onnx := resolved
    }
  else
    throw s!"{ctx}: Reshape target rank {resolved.size} not supported (expected 4D [1,C,H,W], 3D [H,M,K], or 2D [1,C])"

private def resolveSliceParams
    (nd : IRNode) (consts : Array (String × ConstSource)) (inInfo : TensorShapeInfo) (ctx : String) :
    Except String (Array Nat × Array Nat × Array Nat × TensorShapeInfo) := do
  let rank := inInfo.onnx.size
  if rank != 4 && rank != 3 && rank != 2 then
    throw s!"{ctx}: Slice requires 4D, 3D, or 2D input, got rank {rank}"
  let inSh := inInfo.vane
  let (starts, ends, axes0, steps0) ←
    if nd.inputs.size >= 3 then
      let readI64 := fun (idx : Nat) (label : String) => do
        let nm := nd.inputs[idx]!
        let some cs := lookupAssoc consts nm
          | throw s!"{ctx}: Slice {label} '{nm}' must be a constant int64 tensor"
        cs.int64s
      let st ← readI64 1 "starts"
      let en ← readI64 2 "ends"
      let ax ← if nd.inputs.size >= 4 && !nd.inputs[3]!.isEmpty then readI64 3 "axes" else pure #[]
      let sp ← if nd.inputs.size >= 5 && !nd.inputs[4]!.isEmpty then readI64 4 "steps" else pure #[]
      pure (st, en, ax, sp)
    else
      let some st := nd.orig.attrInts? "starts" | throw s!"{ctx}: Slice missing starts"
      let some en := nd.orig.attrInts? "ends" | throw s!"{ctx}: Slice missing ends"
      let ax := (nd.orig.attrInts? "axes").getD #[]
      pure (st, en, ax, #[])

  if starts.size != ends.size || starts.isEmpty then
    throw s!"{ctx}: Slice starts ({starts.size}) and ends ({ends.size}) length mismatch"
  let axes : Array Int ←
    if axes0.isEmpty then
      pure (Array.ofFn (n := starts.size) fun i => (i.val : Int))
    else if axes0.size != starts.size then
      throw s!"{ctx}: Slice axes length {axes0.size} != starts length {starts.size}"
    else
      pure axes0
  let steps : Array Int ←
    if steps0.isEmpty then
      pure (Array.replicate starts.size (1 : Int))
    else if steps0.size != starts.size then
      throw s!"{ctx}: Slice steps length {steps0.size} != starts length {starts.size}"
    else
      pure steps0

  let mut beg : Array Nat := #[0, 0, 0, 0]
  let mut en : Array Nat := #[inSh.n, inSh.c, inSh.h, inSh.w]
  let mut step : Array Nat := #[1, 1, 1, 1]
  let mut seen : Array Bool := Array.replicate rank false

  for i in [:starts.size] do
    let axRaw := axes[i]!
    let axInt := if axRaw < 0 then axRaw + (rank : Int) else axRaw
    if axInt < 0 || axInt >= (rank : Int) then
      throw s!"{ctx}: Slice axis {axRaw} out of range [0,{rank - 1}] for rank-{rank} input"
    let onnxAx := axInt.toNat
    if seen[onnxAx]! then
      throw s!"{ctx}: Slice duplicate axis {onnxAx}"
    seen := seen.set! onnxAx true
    let vaneAx : Nat ←
      if rank == 3 then
        if onnxAx == 2 then pure 3
        else throw s!"{ctx}: 3D Slice axis {axRaw} not supported (only last axis 2/-1 is supported on 3D [H,M,K])"
      else if rank == 2 then
        if onnxAx == 0 || onnxAx == 1 then pure onnxAx
        else throw s!"{ctx}: 2D Slice axis {axRaw} out of range"
      else
        pure onnxAx
    let stInt := steps[i]!
    if stInt <= 0 then
      throw s!"{ctx}: Slice step {stInt} must be > 0"
    let dim : Int := (inInfo.onnx[onnxAx]! : Int)
    let mut s := starts[i]!
    let mut e := ends[i]!
    if s < 0 then s := s + dim
    if e < 0 then e := e + dim
    if s < 0 then s := 0 else if s > dim then s := dim
    if e < 0 then e := 0 else if e > dim then e := dim
    if e <= s then
      throw s!"{ctx}: Slice on axis {onnxAx} produces empty range [{s},{e})"
    beg := beg.set! vaneAx s.toNat
    en := en.set! vaneAx e.toNat
    step := step.set! vaneAx stInt.toNat

  let vp ← planSlice inSh beg en step
  if vp.outShape.n != 1 then
    throw s!"{ctx}: Slice output batch dimension must be 1, got {vp.outShape.n}"
  let outOnnx : Array Nat :=
    if rank == 3 then #[inInfo.onnx[0]!, inInfo.onnx[1]!, vp.outShape.w]
    else if rank == 2 then #[1, vp.outShape.c]
    else vp.outShape.toArray
  pure (beg, en, step, { vane := vp.outShape, onnx := outOnnx })

/-- Validate ONNX right-aligned broadcasting against ANE physical `[1, C, H, W]` broadcast semantics. -/
private def resolveElementwiseBroadcast
    (infoA infoB : TensorShapeInfo) (ctx : String) : Except String TensorShapeInfo := do
  let outOnnx ← match broadcastOnnxShapes infoA.onnx infoB.onnx with
    | .ok s => pure s
    | .error e => throw s!"{ctx}: {e}"
  let expectedVane : Shape4D ←
    if outOnnx.size == 4 then
      if outOnnx[0]! != 1 then
        throw s!"{ctx}: broadcast output batch dimension N={outOnnx[0]!} != 1"
      pure { n := 1, c := outOnnx[1]!, h := outOnnx[2]!, w := outOnnx[3]! }
    else if outOnnx.size == 3 then
      pure { n := 1, c := outOnnx[0]! * outOnnx[1]!, h := 1, w := outOnnx[2]! }
    else if outOnnx.size == 2 then
      if outOnnx[0]! != 1 then
        throw s!"{ctx}: broadcast output batch dimension N={outOnnx[0]!} != 1"
      pure { n := 1, c := outOnnx[1]!, h := 1, w := 1 }
    else
      throw s!"{ctx}: unsupported broadcast output rank {outOnnx.size} ({outOnnx.toList})"
  let dA : TensorDims := { c := infoA.vane.c, h := infoA.vane.h, w := infoA.vane.w }
  let dB : TensorDims := { c := infoB.vane.c, h := infoB.vane.h, w := infoB.vane.w }
  let outD ← broadcastDims dA dB
  let outVane : Shape4D := { n := 1, c := outD.c, h := outD.h, w := outD.w }
  if outVane != expectedVane then
    throw s!"{ctx}: ONNX broadcast {infoA.onnx.toList} x {infoB.onnx.toList} -> {outOnnx.toList} is incompatible with ANE physical layout {outVane.toArray.toList}"
  let checkOperand := fun (opInfo : TensorShapeInfo) (side : String) => do
    if opInfo.vane.numElems == 1 then
      pure ()
    else if opInfo.onnx.size != outOnnx.size then
      throw s!"{ctx}: {side} shape {opInfo.onnx.toList} cannot broadcast to {outOnnx.toList} under ANE physical layout"
    else if outOnnx.size == 3 then
      let hX := opInfo.onnx[0]!; let mX := opInfo.onnx[1]!
      let hO := outOnnx[0]!; let mO := outOnnx[1]!
      if !((hX == hO && mX == mO) || (hX == 1 && mX == 1)) then
        throw s!"{ctx}: 3D broadcast {infoA.onnx.toList} x {infoB.onnx.toList} requires matching (H, M) or (1, 1) on {side}"
    else
      pure ()
  checkOperand infoA "LHS"
  checkOperand infoB "RHS"
  pure { vane := outVane, onnx := outOnnx }

-- ============================================================================
-- Fusion Passes (Skipped when `--no-fuse`)
-- ============================================================================

/-- Collapse `Mul(x, Sigmoid(x))` / `Mul(Sigmoid(x), x)` with single-consumer `Sigmoid` -> `Silu(x)`. -/
private def collapseSiluPatterns (nodes : Array IRNode) (graphOutput : String) : Array IRNode := Id.run do
  let counts := countConsumers nodes graphOutput
  let prodMap := buildProducerMap nodes
  let mut removed : Array Bool := Array.replicate nodes.size false
  let mut outNodes := nodes
  for i in [:outNodes.size] do
    let nd := outNodes[i]!
    if nd.opType == "Mul" && nd.inputs.size == 2 then
      let a := nd.inputs[0]!
      let b := nd.inputs[1]!
      let matchRight :=
        match lookupAssoc prodMap b with
        | some pIdx =>
          let p := outNodes[pIdx]!
          p.opType == "Sigmoid" && p.inputs.size == 1 && p.inputs[0]! == a && consumerCount counts b == 1
        | none => false
      let matchLeft :=
        match lookupAssoc prodMap a with
        | some pIdx =>
          let p := outNodes[pIdx]!
          p.opType == "Sigmoid" && p.inputs.size == 1 && p.inputs[0]! == b && consumerCount counts a == 1
        | none => false
      if matchRight then
        let pIdx := (lookupAssoc prodMap b).get!
        removed := removed.set! pIdx true
        outNodes := outNodes.set! i { nd with opType := "Silu", inputs := #[a] }
      else if matchLeft then
        let pIdx := (lookupAssoc prodMap a).get!
        removed := removed.set! pIdx true
        outNodes := outNodes.set! i { nd with opType := "Silu", inputs := #[b] }
  let mut kept : Array IRNode := #[]
  for i in [:outNodes.size] do
    if !removed[i]! then
      kept := kept.push outNodes[i]!
  return kept

/-- Algebraically fold single-consumer `Conv -> BatchNormalization` in float64 before fp16 quantization. -/
private def foldConvBatchNorm
    (nodes : Array IRNode) (consts : Array (String × ConstSource)) (graphOutput : String) :
    Except String (Array IRNode × Nat) := do
  let counts := countConsumers nodes graphOutput
  let mut prodMap := buildProducerMap nodes
  let mut removed : Array Bool := Array.replicate nodes.size false
  let mut outNodes := nodes
  let mut fused : Nat := 0

  for i in [:outNodes.size] do
    let bn := outNodes[i]!
    if bn.opType == "BatchNormalization" && bn.inputs.size >= 5 then
      if let some convIdx := lookupAssoc prodMap bn.inputs[0]! then
        let conv := outNodes[convIdx]!
        if conv.opType == "Conv" && consumerCount counts conv.primaryOutput == 1 && conv.inputs.size >= 2 then
          if let some wc := lookupAssoc consts conv.inputs[1]! then
            let ctx := s!"node '{nodeLabel bn i}' (Conv+BatchNormalization)"
            if wc.dims.size != 4 then
              throw s!"{ctx}: Conv weight '{conv.inputs[1]!}' must be 4D, got {wc.dims.toList}"
            let cout := wc.dims[0]!
            let mut wF64 ← match conv.foldedW with
              | some fw => pure fw
              | none => wc.float64s
            if cout == 0 || wF64.isEmpty || wF64.size % cout != 0 then
              throw s!"{ctx}: invalid Conv weight length {wF64.size} for Cout={cout}"

            let mut convBias ← match conv.foldedB with
              | some fb => pure fb
              | none =>
                if conv.inputs.size >= 3 && !conv.inputs[2]!.isEmpty then
                  let some bc := lookupAssoc consts conv.inputs[2]!
                    | throw s!"{ctx}: Conv bias '{conv.inputs[2]!}' must be constant"
                  let bv ← bc.float64s
                  if bv.size != cout then
                    throw s!"{ctx}: Conv bias length {bv.size} != Cout={cout}"
                  pure bv
                else
                  pure (Array.replicate cout 0.0)

            let (factor, bnBiasShift) ← computeBNAffine bn consts cout ctx
            let perOut := wF64.size / cout
            for o in [:cout] do
              let f := factor[o]!
              let base := o * perOut
              for k in [:perOut] do
                wF64 := wF64.set! (base + k) (wF64[base + k]! * f)
              convBias := convBias.set! o (convBias[o]! * f + bnBiasShift[o]!)

            let updatedConv : IRNode := {
              conv with
              foldedW := some wF64
              foldedWSh := wc.dims
              foldedB := some convBias
              outputs := bn.outputs
            }
            outNodes := outNodes.set! convIdx updatedConv
            for o in bn.outputs do
              if !o.isEmpty then
                prodMap := setAssoc prodMap o convIdx
            removed := removed.set! i true
            fused := fused + 1

  let mut kept : Array IRNode := #[]
  for i in [:outNodes.size] do
    if !removed[i]! then
      kept := kept.push outNodes[i]!
  pure (kept, fused)

/-- Fold single-consumer static-weight `MatMul(A, const B) -> Add(_, const bias)` into a biased linear op. -/
private def foldMatMulAddBias
    (nodes : Array IRNode) (consts : Array (String × ConstSource)) (graphOutput : String) :
    Except String (Array IRNode) := do
  let counts := countConsumers nodes graphOutput
  let mut prodMap := buildProducerMap nodes
  let mut removed : Array Bool := Array.replicate nodes.size false
  let mut outNodes := nodes

  for i in [:outNodes.size] do
    let addNd := outNodes[i]!
    if addNd.opType == "Add" && addNd.inputs.size == 2 then
      let a := addNd.inputs[0]!
      let b := addNd.inputs[1]!
      let cand? : Option (Nat × String) :=
        match lookupAssoc prodMap a with
        | some pIdx =>
          let p := outNodes[pIdx]!
          if p.opType == "MatMul" && p.foldedB.isNone && consumerCount counts a == 1 && (lookupAssoc consts b).isSome then
            some (pIdx, b)
          else none
        | none =>
          match lookupAssoc prodMap b with
          | some pIdx =>
            let p := outNodes[pIdx]!
            if p.opType == "MatMul" && p.foldedB.isNone && consumerCount counts b == 1 && (lookupAssoc consts a).isSome then
              some (pIdx, a)
            else none
          | none => none
      if let some (mmIdx, biasName) := cand? then
        let mm := outNodes[mmIdx]!
        if mm.inputs.size >= 2 then
          if let some wc := lookupAssoc consts mm.inputs[1]! then
            if wc.dims.size == 2 then
              let cout := wc.dims[1]!
              let bc := (lookupAssoc consts biasName).get!
              if isChannelBiasShape bc.dims cout then
                let bVals ← bc.float64s
                if bVals.size == cout then
                  let updatedMm : IRNode := { mm with foldedB := some bVals, outputs := addNd.outputs }
                  outNodes := outNodes.set! mmIdx updatedMm
                  for o in addNd.outputs do
                    if !o.isEmpty then
                      prodMap := setAssoc prodMap o mmIdx
                  removed := removed.set! i true

  let mut kept : Array IRNode := #[]
  for i in [:outNodes.size] do
    if !removed[i]! then
      kept := kept.push outNodes[i]!
  pure kept

/-- Try to extract a fusable `ConvActivation` from a unary activation `IRNode` (validating params & recording warnings). -/
private def extractFusableAct
    (act : IRNode) (consts : Array (String × ConstSource)) (ctx : String) :
    Except String (Option (ConvActivation × Option String)) := do
  match act.opType with
  | "Relu" =>
    if act.inputs.size != 1 then pure none
    else pure (some (.relu, none))
  | "Gelu" =>
    if act.inputs.size != 1 then pure none
    else
      let approxStr ← act.orig.attrStringChecked "approximate" ""
      let _ ← match classifyGeluApprox approxStr with
        | .ok a => pure a
        | .error e => throw s!"{ctx}: {e}"
      pure (some (.lut .gelu, none))
  | "Silu" | "Swish" =>
    if act.inputs.size != 1 then pure none
    else pure (some (.lut .silu, none))
  | "Sigmoid" =>
    if act.inputs.size != 1 then pure none
    else pure (some (.lut .sigmoid, none))
  | "Tanh" =>
    if act.inputs.size != 1 then pure none
    else pure (some (.lut .tanh, none))
  | "Erf" =>
    if act.inputs.size != 1 then pure none
    else pure (some (.lut .erf, none))
  | "LeakyRelu" =>
    if act.inputs.size != 1 then pure none
    else
      let alpha := act.orig.attrFloat "alpha" 0.01
      let _ ← validateFiniteFp16Scalar alpha ctx
      pure (some (.lut (.leakyRelu alpha), none))
  | "Elu" =>
    if act.inputs.size != 1 then pure none
    else
      let alpha := act.orig.attrFloat "alpha" 1.0
      let _ ← validateFiniteFp16Scalar alpha ctx
      pure (some (.lut (.elu alpha), none))
  | "HardSigmoid" =>
    if act.inputs.size != 1 then pure none
    else
      let alpha := act.orig.attrFloat "alpha" 0.2
      let beta := act.orig.attrFloat "beta" 0.5
      if alpha.isNaN || alpha.isInf || alpha <= 0.0 then
        throw s!"{ctx}: HardSigmoid alpha must be finite and > 0 (got {alpha})"
      let _ ← validateFiniteFp16Scalar alpha ctx
      let _ ← validateFiniteFp16Scalar beta ctx
      let preBits ← validateFiniteFp16Scalar (alpha / 0.2) ctx
      if (preBits &&& 0x7fff) == 0 then
        throw s!"{ctx}: HardSigmoid alpha ({alpha}) underflows pre-LUT scale to 0"
      pure (some (.lut (.hardSigmoid alpha beta), none))
  | "Clip" =>
    if act.inputs.size < 1 then pure none
    else
      let lo? ← readOptionalScalarParam act consts 1 "min" ctx
      let hi? ← readOptionalScalarParam act consts 2 "max" ctx
      match lo?, hi? with
      | some lo, some hi =>
        let _ ← validateFiniteFp16Scalar lo ctx
        let _ ← validateFiniteFp16Scalar hi ctx
        if lo > hi then
          throw s!"{ctx}: Clip requires min <= max (got min={lo}, max={hi})"
        let spec := if lo == 0.0 && hi == 6.0 then LutSpec.relu6 else LutSpec.clip lo hi
        pure (some (.lut spec, none))
      | _, _ => pure none
  | _ => pure none

/-- Fuse single-consumer activations onto preceding `Conv`, `Gemm`, or static-weight `MatMul`. -/
private def fuseConvActivations
    (nodes : Array IRNode) (consts : Array (String × ConstSource)) (graphOutput : String) :
    Except String (Array IRNode × Nat × Array String) := do
  let counts := countConsumers nodes graphOutput
  let mut prodMap := buildProducerMap nodes
  let mut removed : Array Bool := Array.replicate nodes.size false
  let mut outNodes := nodes
  let mut fused : Nat := 0
  let mut warnings : Array String := #[]

  for i in [:outNodes.size] do
    let act := outNodes[i]!
    if act.inputs.size >= 1 then
      if let some prodIdx := lookupAssoc prodMap act.inputs[0]! then
        let prod := outNodes[prodIdx]!
        let isStaticMatMul := prod.opType == "MatMul" && prod.inputs.size >= 2 && (lookupAssoc consts prod.inputs[1]!).isSome
        let canFuseProd :=
          (prod.opType == "Conv" || prod.opType == "Gemm" || isStaticMatMul) &&
          prod.fusedAct == .none &&
          consumerCount counts prod.primaryOutput == 1
        if canFuseProd then
          let ctx := s!"node '{nodeLabel act i}' ({act.opType})"
          if let some (convAct, warn?) ← extractFusableAct act consts ctx then
            if let some w := warn? then
              warnings := warnings.push w
            let updatedProd : IRNode := { prod with fusedAct := convAct, outputs := act.outputs }
            outNodes := outNodes.set! prodIdx updatedProd
            for o in act.outputs do
              if !o.isEmpty then
                prodMap := setAssoc prodMap o prodIdx
            removed := removed.set! i true
            fused := fused + 1

  let mut kept : Array IRNode := #[]
  for i in [:outNodes.size] do
    if !removed[i]! then
      kept := kept.push outNodes[i]!
  pure (kept, fused, warnings)

private def isMatmulFoldableTransposePerm (perm : Array Int) : Bool :=
  perm == #[0, 1, 3, 2] || perm == #[0, 2, 1] || perm == #[0, 3, 2, 1]

/-- Fold single-consumer `Transpose` inputs (`transposeA`/`transposeB`) and post-`MatMul` scalar
    `Mul`/`Div` (`scale`) onto dynamic `MatMul(A, B)` nodes. -/
private def foldDynamicMatMulPatterns
    (nodes : Array IRNode) (consts : Array (String × ConstSource)) (graphOutput : String) :
    Except String (Array IRNode × Nat) := do
  let counts := countConsumers nodes graphOutput
  let mut prodMap := buildProducerMap nodes
  let mut removed : Array Bool := Array.replicate nodes.size false
  let mut outNodes := nodes
  let mut folded : Nat := 0

  -- 1. Fold single-consumer Transpose on A and/or B into dynamic MatMul.
  for i in [:outNodes.size] do
    let nd := outNodes[i]!
    let isDynMM :=
      nd.opType == "MatMul" && nd.inputs.size == 2 &&
      (lookupAssoc consts nd.inputs[0]!).isNone && (lookupAssoc consts nd.inputs[1]!).isNone
    if isDynMM then
      let mut updated := nd
      -- Check A (input 0)
      if !updated.mmTransposeA then
        if let some trIdx := lookupAssoc prodMap updated.inputs[0]! then
          let tr := outNodes[trIdx]!
          if !removed[trIdx]! && tr.opType == "Transpose" && tr.inputs.size == 1 &&
             consumerCount counts tr.primaryOutput == 1 then
            let perm := (tr.orig.attrInts? "perm").getD #[3, 2, 1, 0]
            if isMatmulFoldableTransposePerm perm then
              updated := {
                updated with
                inputs := updated.inputs.set! 0 tr.inputs[0]!
                mmTransposeA := true
                mmFoldedTrAPerm := some perm
                mmFrom4D := updated.mmFrom4D || perm == #[0, 1, 3, 2]
              }
              removed := removed.set! trIdx true
              folded := folded + 1
      -- Check B (input 1)
      if !updated.mmTransposeB then
        if let some trIdx := lookupAssoc prodMap updated.inputs[1]! then
          let tr := outNodes[trIdx]!
          if !removed[trIdx]! && tr.opType == "Transpose" && tr.inputs.size == 1 &&
             consumerCount counts tr.primaryOutput == 1 then
            let perm := (tr.orig.attrInts? "perm").getD #[3, 2, 1, 0]
            if isMatmulFoldableTransposePerm perm then
              updated := {
                updated with
                inputs := updated.inputs.set! 1 tr.inputs[0]!
                mmTransposeB := true
                mmFoldedTrBPerm := some perm
                mmFrom4D := updated.mmFrom4D || perm == #[0, 1, 3, 2]
              }
              removed := removed.set! trIdx true
              folded := folded + 1
      outNodes := outNodes.set! i updated

  -- 2. Fold single-consumer post-MatMul scalar Mul / Div into `mmScale`.
  let counts2 := countConsumers (Id.run do
    let mut tmp : Array IRNode := #[]
    for i in [:outNodes.size] do
      if !removed[i]! then tmp := tmp.push outNodes[i]!
    return tmp) graphOutput
  for i in [:outNodes.size] do
    if !removed[i]! then
      let nd := outNodes[i]!
      if (nd.opType == "Mul" || nd.opType == "Div") && nd.inputs.size == 2 then
        let a := nd.inputs[0]!
        let b := nd.inputs[1]!
        let tryFold? : Option (Nat × ConstSource × Bool) :=
          if nd.opType == "Div" then
            match lookupAssoc prodMap a, lookupAssoc consts b with
            | some mmIdx, some sc =>
              let mm := outNodes[mmIdx]!
              let isDyn := mm.opType == "MatMul" && mm.inputs.size == 2 &&
                (lookupAssoc consts mm.inputs[0]!).isNone && (lookupAssoc consts mm.inputs[1]!).isNone
              if !removed[mmIdx]! && isDyn && consumerCount counts2 a == 1 then
                some (mmIdx, sc, true)
              else none
            | _, _ => none
          else
            match lookupAssoc prodMap a, lookupAssoc consts b with
            | some mmIdx, some sc =>
              let mm := outNodes[mmIdx]!
              let isDyn := mm.opType == "MatMul" && mm.inputs.size == 2 &&
                (lookupAssoc consts mm.inputs[0]!).isNone && (lookupAssoc consts mm.inputs[1]!).isNone
              if !removed[mmIdx]! && isDyn && consumerCount counts2 a == 1 then
                some (mmIdx, sc, false)
              else none
            | _, _ =>
              match lookupAssoc prodMap b, lookupAssoc consts a with
              | some mmIdx, some sc =>
                let mm := outNodes[mmIdx]!
                let isDyn := mm.opType == "MatMul" && mm.inputs.size == 2 &&
                  (lookupAssoc consts mm.inputs[0]!).isNone && (lookupAssoc consts mm.inputs[1]!).isNone
                if !removed[mmIdx]! && isDyn && consumerCount counts2 b == 1 then
                  some (mmIdx, sc, false)
                else none
              | _, _ => none
        if let some (mmIdx, cs, isDiv) := tryFold? then
          if ← cs.isScalar then
            let vals ← cs.float64s
            let sc := vals[0]!
            let ctx := s!"node '{nodeLabel nd i}' (MatMul+{nd.opType})"
            if isDiv && sc == 0.0 then
              throw s!"{ctx}: scalar division by zero"
            let mm := outNodes[mmIdx]!
            let newScale := if isDiv then mm.mmScale / sc else mm.mmScale * sc
            let newBits := floatToF16CanonicalZero newScale
            -- Only fold if the resulting fp16 scale is finite and non-zero.
            if !newScale.isNaN && !newScale.isInf && f16IsFinite newBits && (newBits &&& 0x7fff) != 0 then
              let updatedMm : IRNode := { mm with mmScale := newScale, outputs := nd.outputs }
              outNodes := outNodes.set! mmIdx updatedMm
              for o in nd.outputs do
                if !o.isEmpty then
                  prodMap := setAssoc prodMap o mmIdx
              removed := removed.set! i true
              folded := folded + 1

  let mut kept : Array IRNode := #[]
  for i in [:outNodes.size] do
    if !removed[i]! then
      kept := kept.push outNodes[i]!
  pure (kept, folded)

-- ============================================================================
-- Main ONNX -> Vane.Graph.GraphSpec Lowering Engine
-- ============================================================================

private def arrayUniformFloat (arr : Array Float) : Option Float :=
  if arr.isEmpty then none
  else
    let v0 := arr[0]!
    if arr.all (fun v => v == v0) then some v0 else none

private def ln2Const : Float := 0.6931471805599453

/-- Lower a decoded `OnnxModel` directly into an in-memory `Vane.Graph.GraphSpec` and `LowerSummary`. -/
def lowerOnnx (model : OnnxModel) (opts : LowerOptions := {}) : Except String LowerResult := do
  if !model.hasGraph then
    throw "onnx model has no graph"
  let g := model.graph
  if g.outputs.size != 1 then
    throw s!"graph '{g.name}': expected exactly 1 output, got {g.outputs.size}"

  -- 1. Collect initializers (with optional SafeTensors overrides).
  let mut consts : Array (String × ConstSource) := #[]
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

  -- Expose SafeTensors tensors not listed in initializers (for external weight archives).
  if let some sf := opts.safeTensors then
    for info in sf.tensors do
      if (lookupAssoc consts info.name).isNone then
        consts := setAssoc consts info.name { name := info.name, dims := info.shape, payload := .safetensors sf }

  -- 2. Resolve runtime graph inputs (excluding initializers and SafeTensors constants).
  let mut graphInputs : Array GraphInput := #[]
  let mut shapes : Array (String × TensorShapeInfo) := #[]
  let mut vaneIDOf : Array (String × String) := #[]
  let mut usedIDs : Array String := #[]

  for inp in g.inputs do
    if initNames.contains inp.name || (lookupAssoc consts inp.name).isSome then
      pure ()
    else
      let shapeInfo : TensorShapeInfo ←
        match lookupAssoc opts.inputShapes inp.name with
        | some ov =>
          pure { vane := { n := 1, c := ov.c, h := ov.h, w := ov.w }, onnx := #[1, ov.c, ov.h, ov.w] }
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
            pure { vane := { n := 1, c := dimVals[1]!, h := dimVals[2]!, w := dimVals[3]! }, onnx := dimVals }
          else if dimVals.size == 3 then
            -- 3D `[H, M, K]` maps directly to head-stacked Vane `[1, H*M, 1, K]`.
            pure { vane := { n := 1, c := dimVals[0]! * dimVals[1]!, h := 1, w := dimVals[2]! }, onnx := dimVals }
          else if dimVals.size == 2 then
            if dimVals[0]! != 1 then
              throw s!"input '{inp.name}' batch dimension N={dimVals[0]!} != 1"
            pure { vane := { n := 1, c := dimVals[1]!, h := 1, w := 1 }, onnx := dimVals }
          else
            throw s!"input '{inp.name}' has unsupported rank {dimVals.size} (expected 4D [1,C,H,W], 3D [H,M,K], or override with --input-shape {inp.name}=C,H,W)"

      let (id, used') := sanitizeNodeID inp.name usedIDs
      usedIDs := used'
      graphInputs := graphInputs.push {
        name := id
        shape := { c := shapeInfo.vane.c, h := shapeInfo.vane.h, w := shapeInfo.vane.w }
      }
      shapes := setAssoc shapes inp.name shapeInfo
      vaneIDOf := setAssoc vaneIDOf inp.name id

  if graphInputs.isEmpty then
    throw s!"graph '{g.name}' has no runtime inputs"
  if graphInputs.size > 3 then
    throw s!"graph '{g.name}' has {graphInputs.size} runtime inputs (Vane supports 1..3 inputs)"

  -- 3. Stage 1: Elide Constant, Identity, Dropout and build IR node list.
  let mut aliases : Array (String × String) := #[]
  let mut irNodes : Array IRNode := #[]
  let mut elidedCount : Nat := 0

  for nd in g.nodes do
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
        throw s!"Identity node '{nd.name}': expected 1 input and 1 output"
      aliases := setAssoc aliases nd.outputs[0]! (resolveAlias aliases nd.inputs[0]!)
      elidedCount := elidedCount + 1
    else if nd.opType == "Dropout" then
      if nd.inputs.size < 1 || nd.outputs.size < 1 then
        throw s!"Dropout node '{nd.name}': expected at least 1 input and 1 output"
      aliases := setAssoc aliases nd.outputs[0]! (resolveAlias aliases nd.inputs[0]!)
      elidedCount := elidedCount + 1
    else
      if nd.outputs.isEmpty || nd.outputs[0]!.isEmpty then
        throw s!"node '{nd.name}' ({nd.opType}): missing output tensor name"
      let resolvedInps := nd.inputs.map fun inp => if inp.isEmpty then "" else resolveAlias aliases inp
      irNodes := irNodes.push {
        orig := nd
        opType := nd.opType
        name := nd.name
        inputs := resolvedInps
        outputs := nd.outputs
      }

  let graphOutput := resolveAlias aliases g.outputs[0]!.name

  -- 4. Prune nodes unreachable from graphOutput while preserving topological order.
  irNodes := pruneUnreachable irNodes graphOutput

  -- 5. Pattern fusion passes (unless --no-fuse).
  let mut fusedBnCount : Nat := 0
  let mut fusedActCount : Nat := 0
  let mut fusedMatmulCount : Nat := 0
  let mut warnings : Array String := #[]

  if !opts.noFuse then
    irNodes := collapseSiluPatterns irNodes graphOutput
    let (nBn, cBn) ← foldConvBatchNorm irNodes consts graphOutput
    irNodes := nBn
    fusedBnCount := cBn
    irNodes ← foldMatMulAddBias irNodes consts graphOutput
    let (nAct, cAct, actWarns) ← fuseConvActivations irNodes consts graphOutput
    irNodes := nAct
    fusedActCount := cAct
    warnings := warnings ++ actWarns
    let (nMm, cMm) ← foldDynamicMatMulPatterns irNodes consts graphOutput
    irNodes := nMm
    fusedMatmulCount := cMm

  -- Count additional SafeTensors entries referenced by live nodes.
  if let some sf := opts.safeTensors then
    for nd in irNodes do
      for inp in nd.inputs do
        if !inp.isEmpty && !sfCounted.contains inp then
          if (sf.find? inp).isSome then
            sfCounted := sfCounted.push inp
            sfUsed := sfUsed + 1

  -- Compute consumer counts on the final IR (needed for multi-output nodes like Split).
  let finalCounts := countConsumers irNodes graphOutput

  -- 6. Emit in-memory `Vane.Graph.GraphNode` list + static shape inference.
  let opset := model.defaultOpsetVersion
  let mut vaneNodes : Array GraphNode := #[]
  let mut weightCount : Nat := 0
  let mut weightBytes : Nat := 0

  let lookupShape := fun (shMap : Array (String × TensorShapeInfo)) (idMap : Array (String × String))
      (tensorName ctx : String) => do
    let some sh := lookupAssoc shMap tensorName
      | throw s!"{ctx}: unknown runtime tensor '{tensorName}' (missing producer or constant)"
    let some vid := lookupAssoc idMap tensorName
      | throw s!"{ctx}: internal error: no Vane ID for tensor '{tensorName}'"
    pure (sh, vid)

  for idx in [:irNodes.size] do
    let nd := irNodes[idx]!
    let ctx := s!"node '{nodeLabel nd idx}' ({nd.opType})"
    let (baseID, used1) := sanitizeNodeID (nodeLabel nd idx) usedIDs
    usedIDs := used1

    let emit := fun (vNodes : Array GraphNode) (shMap : Array (String × TensorShapeInfo))
        (idMap : Array (String × String)) (gn : GraphNode) (onnxOut : String) (outInfo : TensorShapeInfo) =>
      let vNodes' := vNodes.push gn
      if onnxOut.isEmpty then (vNodes', shMap, idMap)
      else (vNodes', setAssoc shMap onnxOut outInfo, setAssoc idMap onnxOut gn.id)

    match nd.opType with
    | "Conv" =>
      if nd.inputs.size < 2 then
        throw s!"{ctx}: expected at least 2 inputs (X, W)"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      if inInfo.onnx.size != 4 then
        throw s!"{ctx}: Conv requires 4D [1,C,H,W] input, got rank {inInfo.onnx.size} ({inInfo.onnx.toList})"
      let cin := inInfo.vane.c
      let h := inInfo.vane.h
      let w := inInfo.vane.w

      let (wF64, wDims) ← match nd.foldedW with
        | some fw => pure (fw, nd.foldedWSh)
        | none =>
          let some wc := lookupAssoc consts nd.inputs[1]!
            | throw s!"{ctx}: weight input '{nd.inputs[1]!}' must be a constant initializer"
          let vals ← wc.float64s
          pure (vals, wc.dims)
      if wDims.size != 4 then
        throw s!"{ctx}: weight must be 4D [Cout,Cin/G,kH,kW], got shape {wDims.toList}"
      let cout := wDims[0]!
      let cinPerG := wDims[1]!
      let kh := wDims[2]!
      let kw := wDims[3]!
      let groupInt := nd.orig.attrInt "group" 1
      if groupInt <= 0 then
        throw s!"{ctx}: invalid group={groupInt}"
      let group := groupInt.toNat
      if cin % group != 0 || cout % group != 0 || cin / group != cinPerG then
        throw s!"{ctx}: incompatible Conv channels cin={cin} cout={cout} group={group} weight_cin_per_g={cinPerG}"

      if let some dils := nd.orig.attrInts? "dilations" then
        if dils.size != 2 || dils[0]! != 1 || dils[1]! != 1 then
          throw s!"{ctx}: unsupported dilations {dils.toList} (only [1,1] is supported)"

      let (sh, sw) ← match nd.orig.attrInts? "strides" with
        | some st =>
          if st.size != 2 || st[0]! <= 0 || st[1]! <= 0 then
            throw s!"{ctx}: strides must be 2 positive ints, got {st.toList}"
          pure (st[0]!.toNat, st[1]!.toNat)
        | none => pure (1, 1)

      let autoPad := (← nd.orig.attrStringChecked "auto_pad" "NOTSET").toUpper
      let pads ← match nd.orig.attrInts? "pads" with
        | some ps =>
          if ps.size != 4 then
            throw s!"{ctx}: pads must have length 4, got {ps.toList}"
          pure ps
        | none => pure #[0, 0, 0, 0]

      let biasF64? ← match nd.foldedB with
        | some fb => pure (some fb)
        | none =>
          if nd.inputs.size >= 3 && !nd.inputs[2]!.isEmpty then
            let some bc := lookupAssoc consts nd.inputs[2]!
              | throw s!"{ctx}: bias input '{nd.inputs[2]!}' must be a constant initializer"
            if bc.dims.size != 1 || bc.dims[0]! != cout then
              throw s!"{ctx}: bias '{nd.inputs[2]!}' must have shape [{cout}], got {bc.dims.toList}"
            some <$> bc.float64s
          else
            pure none

      let wFP16 ← float64ArrayToFp16 wF64 s!"{baseID}_weights"
      let wBytes := u16ArrayToBytes wFP16
      weightCount := weightCount + 1
      weightBytes := weightBytes + wBytes.size

      let bBytes? ← match biasF64? with
        | some bf =>
          let bFP16 ← float64ArrayToFp16 bf s!"{baseID}_bias"
          let bb := u16ArrayToBytes bFP16
          weightCount := weightCount + 1
          weightBytes := weightBytes + bb.size
          pure (some bb)
        | none => pure none

      if kh == 1 && kw == 1 then
        if sh != 1 || sw != 1 then
          throw s!"{ctx}: conv1x1 requires strides [1,1], got [{sh},{sw}]"
        if group != 1 then
          throw s!"{ctx}: conv1x1 requires group=1, got {group}"
        match autoPad with
        | "" | "NOTSET" =>
          if pads[0]! != 0 || pads[1]! != 0 || pads[2]! != 0 || pads[3]! != 0 then
            throw s!"{ctx}: conv1x1 requires pads [0,0,0,0], got {pads.toList}"
        | "VALID" | "SAME_UPPER" | "SAME_LOWER" => pure ()
        | other => throw s!"{ctx}: conv1x1 unsupported auto_pad '{other}'"
        let outInfo : TensorShapeInfo := {
          vane := { n := 1, c := cout, h := h, w := w }
          onnx := #[1, cout, h, w]
        }
        let (v', s', i') := emit vaneNodes shapes vaneIDOf {
          id := baseID, inputs := #[inID], op := .conv1x1 cin cout wBytes bBytes? nd.fusedAct
        } nd.primaryOutput outInfo
        vaneNodes := v'; shapes := s'; vaneIDOf := i'
      else if kh == 3 && kw == 3 then
        if sh != sw || (sh != 1 && sh != 2) then
          throw s!"{ctx}: conv3x3 requires symmetric stride 1 or 2, got [{sh},{sw}]"
        let mode ←
          if group == 1 then pure Conv3x3Mode.dense
          else if group == cin && cout == cin then pure Conv3x3Mode.depthwise
          else throw s!"{ctx}: grouped conv3x3 (group={group}, cin={cin}, cout={cout}) is not supported (only dense group=1 or depthwise group=cin=cout)"
        let (padMode, ho, wo) ← resolveConv3x3Padding h w sh autoPad pads ctx
        let outInfo : TensorShapeInfo := {
          vane := { n := 1, c := cout, h := ho, w := wo }
          onnx := #[1, cout, ho, wo]
        }
        let (v', s', i') := emit vaneNodes shapes vaneIDOf {
          id := baseID, inputs := #[inID], op := .conv3x3 mode cin cout sh padMode wBytes bBytes? nd.fusedAct
        } nd.primaryOutput outInfo
        vaneNodes := v'; shapes := s'; vaneIDOf := i'
      else
        throw s!"{ctx}: unsupported Conv kernel size {kh}x{kw} (supported: 1x1, 3x3)"

    | "BatchNormalization" =>
      if nd.inputs.size < 5 then
        throw s!"{ctx}: expected 5 inputs (X, scale, B, mean, var)"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      if inInfo.onnx.size != 4 && inInfo.onnx.size != 2 then
        throw s!"{ctx}: BatchNormalization requires 4D [1,C,H,W] or 2D [1,C] input, got rank {inInfo.onnx.size} ({inInfo.onnx.toList})"
      let c := inInfo.vane.c
      let (factor, biasF64) ← computeBNAffine nd consts c ctx
      let mut diagW : Array Float := Array.replicate (c * c) 0.0
      for ch in [:c] do
        diagW := diagW.set! (ch * c + ch) factor[ch]!
      let wBytes := u16ArrayToBytes (← float64ArrayToFp16 diagW s!"{baseID}_weights")
      let bBytes := u16ArrayToBytes (← float64ArrayToFp16 biasF64 s!"{baseID}_bias")
      weightCount := weightCount + 2
      weightBytes := weightBytes + wBytes.size + bBytes.size
      let (v', s', i') := emit vaneNodes shapes vaneIDOf {
        id := baseID, inputs := #[inID], op := .conv1x1 c c wBytes (some bBytes) nd.fusedAct
      } nd.primaryOutput inInfo
      vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Gemm" =>
      if nd.inputs.size < 2 then
        throw s!"{ctx}: expected at least 2 inputs"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      if inInfo.onnx.size != 2 && inInfo.onnx.size != 4 then
        throw s!"{ctx}: Gemm requires 2D [1,Cin] or 4D [1,Cin,1,1] input, got rank {inInfo.onnx.size} ({inInfo.onnx.toList})"
      let cin := inInfo.vane.c
      if inInfo.vane.h != 1 || inInfo.vane.w != 1 then
        throw s!"{ctx}: Gemm requires 2D input [1, Cin] (normalized [1, Cin, 1, 1]), got [1, {cin}, {inInfo.vane.h}, {inInfo.vane.w}]"
      let some bc := lookupAssoc consts nd.inputs[1]!
        | throw s!"{ctx}: weight input '{nd.inputs[1]!}' must be a constant initializer"
      if bc.dims.size != 2 then
        throw s!"{ctx}: weight '{nd.inputs[1]!}' must be 2D, got shape {bc.dims.toList}"
      let bRaw ← bc.float64s
      let transA := nd.orig.attrInt "transA" 0
      let transB := nd.orig.attrInt "transB" 0
      let alpha := nd.orig.attrFloat "alpha" 1.0
      let beta := nd.orig.attrFloat "beta" 1.0
      if transA != 0 then
        throw s!"{ctx}: transA={transA} is not supported (expected 0)"
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

      let biasF64? ← match nd.foldedB with
        | some fb =>
          if fb.size != cout then throw s!"{ctx}: folded bias length {fb.size} != Cout={cout}"
          pure (some fb)
        | none =>
          if nd.inputs.size >= 3 && !nd.inputs[2]!.isEmpty then
            let some cc := lookupAssoc consts nd.inputs[2]!
              | throw s!"{ctx}: bias input '{nd.inputs[2]!}' must be a constant initializer"
            let cRaw ← cc.float64s
            if cRaw.size == 1 then
              pure (some (Array.replicate cout (beta * cRaw[0]!)))
            else if cRaw.size == cout then
              pure (some (cRaw.map fun v => beta * v))
            else
              throw s!"{ctx}: bias length {cRaw.size} incompatible with Cout={cout}"
          else
            pure none

      let wBytes := u16ArrayToBytes (← float64ArrayToFp16 wF64 s!"{baseID}_weights")
      weightCount := weightCount + 1
      weightBytes := weightBytes + wBytes.size
      let bBytes? ← match biasF64? with
        | some bf =>
          let bb := u16ArrayToBytes (← float64ArrayToFp16 bf s!"{baseID}_bias")
          weightCount := weightCount + 1
          weightBytes := weightBytes + bb.size
          pure (some bb)
        | none => pure none
      let outInfo : TensorShapeInfo := {
        vane := { n := 1, c := cout, h := 1, w := 1 }
        onnx := if inInfo.onnx.size == 2 then #[1, cout] else #[1, cout, 1, 1]
      }
      let (v', s', i') := emit vaneNodes shapes vaneIDOf {
        id := baseID, inputs := #[inID], op := .conv1x1 cin cout wBytes bBytes? nd.fusedAct
      } nd.primaryOutput outInfo
      vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "MatMul" =>
      if nd.inputs.size != 2 then
        throw s!"{ctx}: expected 2 inputs, got {nd.inputs.size}"
      match lookupAssoc consts nd.inputs[1]! with
      | some bc =>
        -- Static-weight MatMul(A, const B) -> linear / conv1x1 projection along C
        let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
        if inInfo.onnx.size != 2 && inInfo.onnx.size != 4 then
          throw s!"{ctx}: static-weight MatMul requires 2D [1,C] or 4D [1,C,H,W] input, got rank {inInfo.onnx.size} ({inInfo.onnx.toList})"
        let cin := inInfo.vane.c
        let h := inInfo.vane.h
        let w := inInfo.vane.w
        if bc.dims.size != 2 then
          throw s!"{ctx}: static MatMul weight '{nd.inputs[1]!}' must be 2D, got shape {bc.dims.toList}"
        if bc.dims[0]! != cin then
          throw s!"{ctx}: weight shape [{bc.dims[0]!},{bc.dims[1]!}] incompatible with Cin={cin}"
        let cout := bc.dims[1]!
        let bRaw ← bc.float64s
        let mut wF64 : Array Float := Array.replicate (cout * cin) 0.0
        for c in [:cin] do
          for o in [:cout] do
            wF64 := wF64.set! (o * cin + c) bRaw[c * cout + o]!
        let wBytes := u16ArrayToBytes (← float64ArrayToFp16 wF64 s!"{baseID}_weights")
        weightCount := weightCount + 1
        weightBytes := weightBytes + wBytes.size
        let bBytes? ← match nd.foldedB with
          | some fb =>
            if fb.size != cout then throw s!"{ctx}: folded bias length {fb.size} != Cout={cout}"
            let bb := u16ArrayToBytes (← float64ArrayToFp16 fb s!"{baseID}_bias")
            weightCount := weightCount + 1
            weightBytes := weightBytes + bb.size
            pure (some bb)
          | none => pure none
        let outInfo : TensorShapeInfo := {
          vane := { n := 1, c := cout, h := h, w := w }
          onnx := if inInfo.onnx.size == 2 then #[1, cout] else #[1, cout, h, w]
        }
        let (v', s', i') := emit vaneNodes shapes vaneIDOf {
          id := baseID, inputs := #[inID], op := .conv1x1 cin cout wBytes bBytes? nd.fusedAct
        } nd.primaryOutput outInfo
        vaneNodes := v'; shapes := s'; vaneIDOf := i'
      | none =>
        -- Dynamic runtime-kernel MatMul(A, B) -> NodeOp.matmul!
        if (lookupAssoc consts nd.inputs[0]!).isSome then
          throw s!"{ctx}: constant LHS with dynamic RHS in MatMul is not supported"
        let (infoA, idA) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
        let (infoB, idB) ← lookupShape shapes vaneIDOf nd.inputs[1]! ctx

        let scaleBits ← validateFiniteFp16Scalar nd.mmScale ctx
        if (scaleBits &&& 0x7fff) == 0 then
          throw s!"{ctx}: dynamic MatMul scale must be non-zero in fp16 (got {nd.mmScale})"

        let checkFoldedPerm := fun (side : String) (got? : Option (Array Int)) (expected : Array Int) (branchLabel : String) => do
          match got? with
          | none => pure ()
          | some p =>
            if p != expected then
              throw s!"{ctx}: {branchLabel} MatMul {side} folded Transpose perm {p.toList} != required {expected.toList}"

        let trA := nd.mmTransposeA
        let trB := nd.mmTransposeB

        if infoA.onnx.size == 3 && infoB.onnx.size == 3 then
          -- 3D `[H, M, K]` x `[H, K, N]` (already stored head-stacked `[1, H*M, 1, K]`)
          checkFoldedPerm "LHS" nd.mmFoldedTrAPerm #[0, 2, 1] "3D"
          checkFoldedPerm "RHS" nd.mmFoldedTrBPerm #[0, 2, 1] "3D"
          let hA := infoA.onnx[0]!; let rA := infoA.onnx[1]!; let cA := infoA.onnx[2]!
          let hB := infoB.onnx[0]!; let rB := infoB.onnx[1]!; let cB := infoB.onnx[2]!
          if hA != hB then
            throw s!"{ctx}: 3D MatMul head count mismatch ({hA} != {hB})"
          let heads := hA
          let (m, ka) := if trA then (cA, rA) else (rA, cA)
          let (kb, n) := if trB then (cB, rB) else (rB, cB)
          if ka != kb then
            throw s!"{ctx}: 3D MatMul inner dimension mismatch A.K={ka} != B.K={kb}"
          validateMatmulSpec { b := heads, m := m, k := ka, n := n, transposeA := trA, transposeB := trB, scaleBits := scaleBits }
          let outInfo : TensorShapeInfo := {
            vane := matmulStackedShape heads m n
            onnx := #[heads, m, n]
          }
          let (v', s', i') := emit vaneNodes shapes vaneIDOf {
            id := baseID, inputs := #[idA, idB], op := .matmul heads trA trB scaleBits
          } nd.primaryOutput outInfo
          vaneNodes := v'; shapes := s'; vaneIDOf := i'
        else if infoA.onnx.size == 4 && infoB.onnx.size == 4 &&
                (infoA.vane.h > 1 || infoB.vane.h > 1 || nd.mmFrom4D) then
          -- Standard ONNX 4D `[1, H, M, K]` x `[1, H, K, N]` packed/unpacked via legal M5C reshape!
          checkFoldedPerm "LHS" nd.mmFoldedTrAPerm #[0, 1, 3, 2] "4D"
          checkFoldedPerm "RHS" nd.mmFoldedTrBPerm #[0, 1, 3, 2] "4D"
          let hA := infoA.vane.c; let rA := infoA.vane.h; let cA := infoA.vane.w
          let hB := infoB.vane.c; let rB := infoB.vane.h; let cB := infoB.vane.w
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

          let mut effIdA := idA
          if needPackA then
            let stackedA := matmulStackedShape heads rA cA
            let _ ← planReshape infoA.vane stackedA
            let (packAId, u1) := sanitizeNodeID s!"{baseID}_packA" usedIDs
            usedIDs := u1
            let (v', s', i') := emit vaneNodes shapes vaneIDOf {
              id := packAId, inputs := #[idA], op := .reshape { c := stackedA.c, h := 1, w := stackedA.w }
            } "" { vane := stackedA, onnx := #[1, stackedA.c, 1, stackedA.w] }
            vaneNodes := v'; shapes := s'; vaneIDOf := i'
            effIdA := packAId

          let mut effIdB := idB
          if needPackB then
            let stackedB := matmulStackedShape heads rB cB
            let _ ← planReshape infoB.vane stackedB
            let (packBId, u2) := sanitizeNodeID s!"{baseID}_packB" usedIDs
            usedIDs := u2
            let (v', s', i') := emit vaneNodes shapes vaneIDOf {
              id := packBId, inputs := #[idB], op := .reshape { c := stackedB.c, h := 1, w := stackedB.w }
            } "" { vane := stackedB, onnx := #[1, stackedB.c, 1, stackedB.w] }
            vaneNodes := v'; shapes := s'; vaneIDOf := i'
            effIdB := packBId

          let stackedOut := matmulStackedShape heads m n
          let out4D := matmul4DShape heads m n
          if needUnpack then
            let _ ← planReshape stackedOut out4D
            let (mmCoreId, u3) := sanitizeNodeID s!"{baseID}_mm" usedIDs
            usedIDs := u3
            let (v1, s1, i1) := emit vaneNodes shapes vaneIDOf {
              id := mmCoreId, inputs := #[effIdA, effIdB], op := .matmul heads trA trB scaleBits
            } "" { vane := stackedOut, onnx := #[1, stackedOut.c, 1, stackedOut.w] }
            let (v2, s2, i2) := emit v1 s1 i1 {
              id := baseID, inputs := #[mmCoreId], op := .reshape { c := out4D.c, h := out4D.h, w := out4D.w }
            } nd.primaryOutput { vane := out4D, onnx := #[1, heads, m, n] }
            vaneNodes := v2; shapes := s2; vaneIDOf := i2
          else
            let (v', s', i') := emit vaneNodes shapes vaneIDOf {
              id := baseID, inputs := #[effIdA, effIdB], op := .matmul heads trA trB scaleBits
            } nd.primaryOutput { vane := out4D, onnx := #[1, heads, m, n] }
            vaneNodes := v'; shapes := s'; vaneIDOf := i'
        else
          -- Vane head-stacked `[1, H*M, 1, K]` x `[1, H*K, 1, N]` (H=1 on both inputs)
          if infoA.onnx.size != 4 || infoB.onnx.size != 4 || infoA.vane.h != 1 || infoB.vane.h != 1 then
            throw s!"{ctx}: dynamic MatMul requires matching 4D [1,H,M,K], 3D [H,M,K], or head-stacked [1,H*M,1,K] inputs (got {infoA.onnx.toList} and {infoB.onnx.toList})"
          checkFoldedPerm "LHS" nd.mmFoldedTrAPerm #[0, 3, 2, 1] "head-stacked"
          checkFoldedPerm "RHS" nd.mmFoldedTrBPerm #[0, 3, 2, 1] "head-stacked"
          let ca := infoA.vane.c; let wa := infoA.vane.w
          let cb := infoB.vane.c; let wb := infoB.vane.w
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
          let (v', s', i') := emit vaneNodes shapes vaneIDOf {
            id := baseID, inputs := #[idA, idB], op := .matmul heads trA trB scaleBits
          } nd.primaryOutput { vane := outVane, onnx := #[1, outVane.c, 1, outVane.w] }
          vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Add" | "Sub" | "Mul" | "Div" | "Max" | "Min" =>
      if nd.inputs.size != 2 then
        throw s!"{ctx}: expected 2 inputs, got {nd.inputs.size}"
      let lhsName := nd.inputs[0]!
      let rhsName := nd.inputs[1]!
      let lhsC? := lookupAssoc consts lhsName
      let rhsC? := lookupAssoc consts rhsName
      if lhsC?.isSome && rhsC?.isSome then
        throw s!"{ctx}: both inputs ('{lhsName}', '{rhsName}') are constants; constant folding for binary ops is not supported"

      if (lhsC?.isSome || rhsC?.isSome) &&
         (nd.opType == "Add" || nd.opType == "Sub" || nd.opType == "Mul" || nd.opType == "Div") then
        let (cs, tensorName, constOnLeft) := match lhsC?, rhsC? with
          | some c, _ => (c, rhsName, true)
          | _, some c => (c, lhsName, false)
          | none, none => ({ name := "", dims := #[], payload := .inlineF64 #[] }, "", false)
        if !(← cs.isScalar) then
          throw s!"{ctx}: non-scalar constant operand '{cs.name}' (shape {cs.dims.toList}) is not supported for {nd.opType}"
        let vals ← cs.float64s
        let sc := vals[0]!
        let scBits ← validateFiniteFp16Scalar sc ctx
        let (inInfo, inID) ← lookupShape shapes vaneIDOf tensorName ctx
        match nd.opType with
        | "Add" =>
          let (v', s', i') := emit vaneNodes shapes vaneIDOf {
            id := baseID, inputs := #[inID], op := .pe1 (.scalarAdd scBits)
          } nd.primaryOutput inInfo
          vaneNodes := v'; shapes := s'; vaneIDOf := i'
        | "Mul" =>
          let (v', s', i') := emit vaneNodes shapes vaneIDOf {
            id := baseID, inputs := #[inID], op := .pe1 (.scalarMul scBits)
          } nd.primaryOutput inInfo
          vaneNodes := v'; shapes := s'; vaneIDOf := i'
        | "Sub" =>
          let peOp := if constOnLeft then PeOp.scalarRSub scBits else PeOp.scalarSub scBits
          let (v', s', i') := emit vaneNodes shapes vaneIDOf {
            id := baseID, inputs := #[inID], op := .pe1 peOp
          } nd.primaryOutput inInfo
          vaneNodes := v'; shapes := s'; vaneIDOf := i'
        | "Div" =>
          if !constOnLeft then
            let peOp ← PeOp.scalarDiv scBits
            let (v', s', i') := emit vaneNodes shapes vaneIDOf {
              id := baseID, inputs := #[inID], op := .pe1 peOp
            } nd.primaryOutput inInfo
            vaneNodes := v'; shapes := s'; vaneIDOf := i'
          else
            -- c / x = smul(inverse(x), c)
            let (invID, u1) := sanitizeNodeID s!"{baseID}_inv" usedIDs
            usedIDs := u1
            let (v1, s1, i1) := emit vaneNodes shapes vaneIDOf {
              id := invID, inputs := #[inID], op := .lut (.lut .inverse)
            } "" inInfo
            let (v2, s2, i2) := emit v1 s1 i1 {
              id := baseID, inputs := #[invID], op := .pe1 (.scalarMul scBits)
            } nd.primaryOutput inInfo
            vaneNodes := v2; shapes := s2; vaneIDOf := i2
        | _ => pure ()
      else
        let (infoA, idA) ← lookupShape shapes vaneIDOf lhsName ctx
        let (infoB, idB) ← lookupShape shapes vaneIDOf rhsName ctx
        let outInfo ← resolveElementwiseBroadcast infoA infoB ctx
        if nd.opType == "Div" then
          let (invID, u1) := sanitizeNodeID s!"{baseID}_inv" usedIDs
          usedIDs := u1
          let (v1, s1, i1) := emit vaneNodes shapes vaneIDOf {
            id := invID, inputs := #[idB], op := .lut (.lut .inverse)
          } "" infoB
          let (v2, s2, i2) := emit v1 s1 i1 {
            id := baseID, inputs := #[idA, invID], op := .pe2 .mul false
          } nd.primaryOutput outInfo
          vaneNodes := v2; shapes := s2; vaneIDOf := i2
        else
          let binOp : PeBinOp := match nd.opType with
            | "Add" => .add
            | "Sub" => .sub
            | "Mul" => .mul
            | "Max" => .max
            | _ => .min
          let (v', s', i') := emit vaneNodes shapes vaneIDOf {
            id := baseID, inputs := #[idA, idB], op := .pe2 binOp false
          } nd.primaryOutput outInfo
          vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Pow" =>
      if nd.inputs.size != 2 then
        throw s!"{ctx}: Pow requires 2 inputs (X, exponent)"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let some expCs := lookupAssoc consts nd.inputs[1]!
        | throw s!"{ctx}: Pow exponent '{nd.inputs[1]!}' must be a constant scalar"
      if !(← expCs.isScalar) then
        throw s!"{ctx}: Pow exponent '{nd.inputs[1]!}' must be scalar, got shape {expCs.dims.toList}"
      let expVals ← expCs.float64s
      let powKind ← classifyPowExponent expVals[0]! ctx
      match powKind with
      | .square =>
        let (v', s', i') := emit vaneNodes shapes vaneIDOf {
          id := baseID, inputs := #[inID, inID], op := .pe2 .mul false
        } nd.primaryOutput inInfo
        vaneNodes := v'; shapes := s'; vaneIDOf := i'
      | .sqrt =>
        let (v', s', i') := emit vaneNodes shapes vaneIDOf {
          id := baseID, inputs := #[inID], op := .lut (.lut .sqrt)
        } nd.primaryOutput inInfo
        vaneNodes := v'; shapes := s'; vaneIDOf := i'
      | .rsqrt =>
        let (v', s', i') := emit vaneNodes shapes vaneIDOf {
          id := baseID, inputs := #[inID], op := .lut (.lut (.rsqrt 0.0))
        } nd.primaryOutput inInfo
        vaneNodes := v'; shapes := s'; vaneIDOf := i'
      | .reciprocal =>
        let (v', s', i') := emit vaneNodes shapes vaneIDOf {
          id := baseID, inputs := #[inID], op := .lut (.lut .inverse)
        } nd.primaryOutput inInfo
        vaneNodes := v'; shapes := s'; vaneIDOf := i'
      | .identity =>
        shapes := setAssoc shapes nd.primaryOutput inInfo
        vaneIDOf := setAssoc vaneIDOf nd.primaryOutput inID
        elidedCount := elidedCount + 1

    | "Abs" =>
      if nd.inputs.size != 1 then throw s!"{ctx}: expected 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let (v', s', i') := emit vaneNodes shapes vaneIDOf {
        id := baseID, inputs := #[inID], op := .pe1 .abs
      } nd.primaryOutput inInfo
      vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Neg" =>
      if nd.inputs.size != 1 then throw s!"{ctx}: expected 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let (v', s', i') := emit vaneNodes shapes vaneIDOf {
        id := baseID, inputs := #[inID], op := .pe1 (.scalarMul f16NegOne)
      } nd.primaryOutput inInfo
      vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Gelu" =>
      if nd.inputs.size != 1 then throw s!"{ctx}: expected 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let approxStr ← nd.orig.attrStringChecked "approximate" ""
      let _ ← match classifyGeluApprox approxStr with
        | .ok a => pure a
        | .error e => throw s!"{ctx}: {e}"
      let (v', s', i') := emit vaneNodes shapes vaneIDOf {
        id := baseID, inputs := #[inID], op := .lut (.lut .gelu)
      } nd.primaryOutput inInfo
      vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Relu" | "Sigmoid" | "Tanh" | "Exp" | "Erf" | "Sqrt" | "Sign" | "Silu" | "Swish" | "Reciprocal" =>
      if nd.inputs.size != 1 then throw s!"{ctx}: expected 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let act : NeActSpec := match nd.opType with
        | "Relu" => .relu
        | "Sigmoid" => .lut .sigmoid
        | "Tanh" => .lut .tanh
        | "Exp" => .lut .exp
        | "Erf" => .lut .erf
        | "Sqrt" => .lut .sqrt
        | "Sign" => .lut .sign
        | "Silu" | "Swish" => .lut .silu
        | _ => .lut .inverse
      let (v', s', i') := emit vaneNodes shapes vaneIDOf {
        id := baseID, inputs := #[inID], op := .lut act
      } nd.primaryOutput inInfo
      vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Log" =>
      -- Log(x) = ln(2) * log2(x)
      if nd.inputs.size != 1 then throw s!"{ctx}: expected 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let (log2ID, u1) := sanitizeNodeID s!"{baseID}_log2" usedIDs
      usedIDs := u1
      let ln2Bits := floatToF16CanonicalZero ln2Const
      let (v1, s1, i1) := emit vaneNodes shapes vaneIDOf {
        id := log2ID, inputs := #[inID], op := .lut (.lut .log2)
      } "" inInfo
      let (v2, s2, i2) := emit v1 s1 i1 {
        id := baseID, inputs := #[log2ID], op := .pe1 (.scalarMul ln2Bits)
      } nd.primaryOutput inInfo
      vaneNodes := v2; shapes := s2; vaneIDOf := i2

    | "Elu" =>
      if nd.inputs.size != 1 then throw s!"{ctx}: expected 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let alpha := nd.orig.attrFloat "alpha" 1.0
      let _ ← validateFiniteFp16Scalar alpha ctx
      let (v', s', i') := emit vaneNodes shapes vaneIDOf {
        id := baseID, inputs := #[inID], op := .lut (.lut (.elu alpha))
      } nd.primaryOutput inInfo
      vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "LeakyRelu" =>
      if nd.inputs.size != 1 then throw s!"{ctx}: expected 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let alpha := nd.orig.attrFloat "alpha" 0.01
      let _ ← validateFiniteFp16Scalar alpha ctx
      let (v', s', i') := emit vaneNodes shapes vaneIDOf {
        id := baseID, inputs := #[inID], op := .lut (.lut (.leakyRelu alpha))
      } nd.primaryOutput inInfo
      vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "HardSigmoid" =>
      if nd.inputs.size != 1 then throw s!"{ctx}: expected 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let alpha := nd.orig.attrFloat "alpha" 0.2
      let beta := nd.orig.attrFloat "beta" 0.5
      if alpha.isNaN || alpha.isInf || alpha <= 0.0 then
        throw s!"{ctx}: HardSigmoid alpha must be finite and > 0 (got {alpha})"
      let _ ← validateFiniteFp16Scalar alpha ctx
      let _ ← validateFiniteFp16Scalar beta ctx
      let preBits ← validateFiniteFp16Scalar (alpha / 0.2) ctx
      if (preBits &&& 0x7fff) == 0 then
        throw s!"{ctx}: HardSigmoid alpha ({alpha}) underflows pre-LUT scale to 0"
      let (v', s', i') := emit vaneNodes shapes vaneIDOf {
        id := baseID, inputs := #[inID], op := .lut (.lut (.hardSigmoid alpha beta))
      } nd.primaryOutput inInfo
      vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Clip" =>
      if nd.inputs.size < 1 then throw s!"{ctx}: expected at least 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let some lo ← readOptionalScalarParam nd consts 1 "min" ctx
        | throw s!"{ctx}: Clip requires both constant min and max bounds"
      let some hi ← readOptionalScalarParam nd consts 2 "max" ctx
        | throw s!"{ctx}: Clip requires both constant min and max bounds"
      let _ ← validateFiniteFp16Scalar lo ctx
      let _ ← validateFiniteFp16Scalar hi ctx
      if lo > hi then
        throw s!"{ctx}: Clip requires min <= max (got min={lo}, max={hi})"
      let lspec := if lo == 0.0 && hi == 6.0 then LutSpec.relu6 else LutSpec.clip lo hi
      let (v', s', i') := emit vaneNodes shapes vaneIDOf {
        id := baseID, inputs := #[inID], op := .lut (.lut lspec)
      } nd.primaryOutput inInfo
      vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Softplus" =>
      -- Softplus(x) = ln(2) * log2(1 + exp(x))
      if nd.inputs.size != 1 then throw s!"{ctx}: expected 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let (expID, u1) := sanitizeNodeID s!"{baseID}_exp" usedIDs
      let (addID, u2) := sanitizeNodeID s!"{baseID}_add1" u1
      let (log2ID, u3) := sanitizeNodeID s!"{baseID}_log2" u2
      usedIDs := u3
      let ln2Bits := floatToF16CanonicalZero ln2Const
      let (v1, s1, i1) := emit vaneNodes shapes vaneIDOf { id := expID, inputs := #[inID], op := .lut (.lut .exp) } "" inInfo
      let (v2, s2, i2) := emit v1 s1 i1 { id := addID, inputs := #[expID], op := .pe1 (.scalarAdd f16PosOne) } "" inInfo
      let (v3, s3, i3) := emit v2 s2 i2 { id := log2ID, inputs := #[addID], op := .lut (.lut .log2) } "" inInfo
      let (v4, s4, i4) := emit v3 s3 i3 { id := baseID, inputs := #[log2ID], op := .pe1 (.scalarMul ln2Bits) } nd.primaryOutput inInfo
      vaneNodes := v4; shapes := s4; vaneIDOf := i4

    | "MaxPool" | "AveragePool" =>
      if nd.inputs.size != 1 then throw s!"{ctx}: expected 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      if inInfo.onnx.size != 4 then
        throw s!"{ctx}: {nd.opType} requires 4D [1,C,H,W] input, got rank {inInfo.onnx.size} ({inInfo.onnx.toList})"
      let some kShape := nd.orig.attrInts? "kernel_shape"
        | throw s!"{ctx}: missing kernel_shape attribute"
      if kShape.size != 2 || kShape[0]! <= 0 || kShape[1]! <= 0 then
        throw s!"{ctx}: kernel_shape attribute must be 2 positive ints, got {kShape.toList}"
      let kh := kShape[0]!.toNat
      let kw := kShape[1]!.toNat
      if nd.orig.attrInt "ceil_mode" 0 != 0 then
        throw s!"{ctx}: ceil_mode != 0 is not supported"
      if let some dils := nd.orig.attrInts? "dilations" then
        if dils.size != 2 || dils[0]! != 1 || dils[1]! != 1 then
          throw s!"{ctx}: unsupported dilations {dils.toList} (only [1,1] is supported)"
      let (sh, sw) ← match nd.orig.attrInts? "strides" with
        | some st =>
          if st.size != 2 || st[0]! <= 0 || st[1]! <= 0 then
            throw s!"{ctx}: pool strides must be 2 positive ints, got {st.toList}"
          pure (st[0]!.toNat, st[1]!.toNat)
        | none => pure (1, 1)
      let autoPad := (← nd.orig.attrStringChecked "auto_pad" "NOTSET").toUpper
      let pads ← match nd.orig.attrInts? "pads" with
        | some ps =>
          if ps.size != 4 then throw s!"{ctx}: pads must have length 4, got {ps.toList}"
          for p in ps do
            if p < 0 then throw s!"{ctx}: negative pool pad {p}"
          pure ps
        | none => pure #[0, 0, 0, 0]
      let padMode : PoolPad ← match autoPad with
        | "VALID" => pure .valid
        | "SAME_UPPER" => pure .same
        | "SAME_LOWER" =>
          let ohTmp := ceilDiv inInfo.vane.h sh
          let padH := if (ohTmp - 1) * sh + kh > inInfo.vane.h then (ohTmp - 1) * sh + kh - inInfo.vane.h else 0
          let owTmp := ceilDiv inInfo.vane.w sw
          let padW := if (owTmp - 1) * sw + kw > inInfo.vane.w then (owTmp - 1) * sw + kw - inInfo.vane.w else 0
          let pt := (padH + 1) / 2
          let pb := padH - pt
          let pl := (padW + 1) / 2
          let pr := padW - pl
          if pt == 0 && pb == 0 && pl == 0 && pr == 0 then pure .valid else pure (.custom pt pb pl pr)
        | "" | "NOTSET" =>
          let pt := pads[0]!.toNat; let pl := pads[1]!.toNat
          let pb := pads[2]!.toNat; let pr := pads[3]!.toNat
          if pt == 0 && pb == 0 && pl == 0 && pr == 0 then pure .valid else pure (.custom pt pb pl pr)
        | other => throw s!"{ctx}: unsupported auto_pad '{other}'"
      let kind : PoolKind := if nd.opType == "MaxPool" then .max else .avg
      let excl := if nd.opType == "AveragePool" then nd.orig.attrInt "count_include_pad" 0 == 0 else true
      let pspec : PoolSpec := {
        target := opts.target, op := kind
        dims := { c := inInfo.vane.c, h := inInfo.vane.h, w := inInfo.vane.w }
        kh := kh, kw := kw, sh := sh, sw := sw, pad := padMode, excludePad := excl
      }
      let (_, gh, gw) ← resolvePoolBackend pspec
      let outInfo : TensorShapeInfo := {
        vane := { n := 1, c := inInfo.vane.c, h := gh.out, w := gw.out }
        onnx := #[1, inInfo.vane.c, gh.out, gw.out]
      }
      let (v', s', i') := emit vaneNodes shapes vaneIDOf {
        id := baseID, inputs := #[inID], op := .pool kind kh kw sh sw padMode excl
      } nd.primaryOutput outInfo
      vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "GlobalAveragePool" =>
      if nd.inputs.size != 1 then throw s!"{ctx}: expected 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      if inInfo.onnx.size != 4 then
        throw s!"{ctx}: GlobalAveragePool requires 4D [1,C,H,W] input, got rank {inInfo.onnx.size} ({inInfo.onnx.toList})"
      let outInfo : TensorShapeInfo := {
        vane := { n := 1, c := inInfo.vane.c, h := 1, w := 1 }
        onnx := #[1, inInfo.vane.c, 1, 1]
      }
      let (v', s', i') := emit vaneNodes shapes vaneIDOf {
        id := baseID, inputs := #[inID], op := .reduce .mean { reduceH := true, reduceW := true } 0
      } nd.primaryOutput outInfo
      vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceL1" | "ReduceL2" | "ReduceSumSquare" =>
      if nd.inputs.size < 1 then throw s!"{ctx}: expected at least 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      if nd.orig.attrInt "keepdims" 1 == 0 then
        throw s!"{ctx}: keepdims=0 is not supported (Vane reduce preserves 4D rank with keepdims=1)"
      let rawAxes ← match nd.orig.attrInts? "axes" with
        | some ax => pure ax
        | none =>
          if nd.inputs.size >= 2 && !nd.inputs[1]!.isEmpty then
            let some ac := lookupAssoc consts nd.inputs[1]!
              | throw s!"{ctx}: axes input '{nd.inputs[1]!}' must be a constant int64 tensor"
            ac.int64s
          else pure #[]
      if rawAxes.isEmpty && nd.orig.attrInt "noop_with_empty_axes" 0 != 0 then
        shapes := setAssoc shapes nd.primaryOutput inInfo
        vaneIDOf := setAssoc vaneIDOf nd.primaryOutput inID
        elidedCount := elidedCount + 1
      else
        let effAxes :=
          if rawAxes.isEmpty then
            if inInfo.onnx.size == 3 then #[0, 1, 2]
            else if inInfo.onnx.size == 2 then #[1]
            else #[1, 2, 3]
          else rawAxes
        let (axes, outVane) ← resolveReduceAxes4D effAxes inInfo.vane inInfo.onnx.size ctx
        let redOp : ReduceOp := match nd.opType with
          | "ReduceMean" => .mean
          | "ReduceSum" => .sum
          | "ReduceMax" => .max
          | "ReduceMin" => .min
          | "ReduceL1" => .l1
          | "ReduceL2" => .l2
          | _ => .sumSquare
        validateReduceBackendForNode opts.target redOp axes inInfo.vane ctx
        let outOnnx : Array Nat :=
          if inInfo.onnx.size == 3 && axes.reduceW then #[inInfo.onnx[0]!, inInfo.onnx[1]!, 1]
          else if inInfo.onnx.size == 2 && axes.reduceC then #[1, 1]
          else #[1, outVane.c, outVane.h, outVane.w]
        let (v', s', i') := emit vaneNodes shapes vaneIDOf {
          id := baseID, inputs := #[inID], op := .reduce redOp axes 0
        } nd.primaryOutput { vane := outVane, onnx := outOnnx }
        vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Softmax" =>
      if nd.inputs.size != 1 then throw s!"{ctx}: expected 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let rank := inInfo.onnx.size
      let (axes, redVane) ←
        if opset > 0 && opset < 13 then
          let ax0 := nd.orig.attrInt "axis" 1
          let ax := if ax0 < 0 then ax0 + (rank : Int) else ax0
          if rank == 3 then
            if ax == 2 then pure ({ reduceW := true : ReduceAxes }, { n := 1, c := inInfo.vane.c, h := 1, w := 1 : Shape4D })
            else throw s!"{ctx}: 3D Softmax (opset {opset}) axis {ax0} not supported (expected -1 or 2)"
          else
            match ax with
            | 0 | 1 => pure ({ reduceC := true, reduceH := true, reduceW := true : ReduceAxes }, { n := 1, c := 1, h := 1, w := 1 : Shape4D })
            | 2 => pure ({ reduceH := true, reduceW := true : ReduceAxes }, { n := 1, c := inInfo.vane.c, h := 1, w := 1 : Shape4D })
            | 3 => pure ({ reduceW := true : ReduceAxes }, { n := 1, c := inInfo.vane.c, h := inInfo.vane.h, w := 1 : Shape4D })
            | _ => throw s!"{ctx}: Softmax (opset {opset}) axis {ax0} out of range [0,3]"
        else
          let ax0 := nd.orig.attrInt "axis" (-1)
          let ax := if ax0 < 0 then ax0 + (rank : Int) else ax0
          if rank == 3 then
            if ax == 2 then pure ({ reduceW := true : ReduceAxes }, { n := 1, c := inInfo.vane.c, h := 1, w := 1 : Shape4D })
            else throw s!"{ctx}: 3D Softmax axis {ax0} not supported (expected -1 or 2)"
          else
            match ax with
            | 1 => pure ({ reduceC := true : ReduceAxes }, { n := 1, c := 1, h := inInfo.vane.h, w := inInfo.vane.w : Shape4D })
            | 2 => pure ({ reduceH := true : ReduceAxes }, { n := 1, c := inInfo.vane.c, h := 1, w := inInfo.vane.w : Shape4D })
            | 3 => pure ({ reduceW := true : ReduceAxes }, { n := 1, c := inInfo.vane.c, h := inInfo.vane.h, w := 1 : Shape4D })
            | _ => throw s!"{ctx}: Softmax axis {ax0} not in \{1,2,3}"
      validateReduceBackendForNode opts.target .max axes inInfo.vane ctx
      let redInfo : TensorShapeInfo := { vane := redVane, onnx := #[1, redVane.c, redVane.h, redVane.w] }
      let (maxID, u1) := sanitizeNodeID s!"{baseID}_max" usedIDs
      let (subID, u2) := sanitizeNodeID s!"{baseID}_sub" u1
      let (expID, u3) := sanitizeNodeID s!"{baseID}_exp" u2
      usedIDs := u3
      let (v1, s1, i1) := emit vaneNodes shapes vaneIDOf { id := maxID, inputs := #[inID], op := .reduce .max axes 0 } "" redInfo
      let (v2, s2, i2) := emit v1 s1 i1 { id := subID, inputs := #[inID, maxID], op := .pe2 .sub false } "" inInfo
      let (v3, s3, i3) := emit v2 s2 i2 { id := expID, inputs := #[subID], op := .lut (.lut .exp) } "" inInfo
      let (v4, s4, i4, invSumID) ←
        if axes.reduceC && !axes.reduceH && !axes.reduceW && inInfo.vane.w > 1 then
          let (sumID, u4) := sanitizeNodeID s!"{baseID}_sum" usedIDs
          let (invID, u5) := sanitizeNodeID s!"{baseID}_invsum" u4
          usedIDs := u5
          let (va, sa, ia) := emit v3 s3 i3 { id := sumID, inputs := #[expID], op := .reduce .sum axes 0 } "" redInfo
          let (vb, sb, ib) := emit va sa ia { id := invID, inputs := #[sumID], op := .lut (.lut .inverse) } "" redInfo
          pure (vb, sb, ib, invID)
        else
          let (invID, u4) := sanitizeNodeID s!"{baseID}_invsum" usedIDs
          usedIDs := u4
          let (va, sa, ia) := emit v3 s3 i3 { id := invID, inputs := #[expID], op := .reduce .invSum axes 0 } "" redInfo
          pure (va, sa, ia, invID)
      let (v5, s5, i5) := emit v4 s4 i4 { id := baseID, inputs := #[expID, invSumID], op := .pe2 .mul false } nd.primaryOutput inInfo
      vaneNodes := v5; shapes := s5; vaneIDOf := i5

    | "LayerNormalization" =>
      -- ONNX opset 17+: Y = ((X - mean) / sqrt(var + eps)) * Scale [+ B]
      -- Normalization reduces over EVERY axis from `axis` to the last (`X.shape[axis:]`).
      if nd.inputs.size < 2 then
        throw s!"{ctx}: LayerNormalization requires at least 2 inputs (X, Scale[, B])"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let rank := inInfo.onnx.size
      let ax0 := nd.orig.attrInt "axis" (-1)
      let ax := if ax0 < 0 then ax0 + (rank : Int) else ax0
      let (axes, redVane, redOnnx, normAxis, normDim) ←
        if rank == 3 then
          if ax == 2 then
            pure ({ reduceW := true : ReduceAxes }, { n := 1, c := inInfo.vane.c, h := 1, w := 1 : Shape4D },
                  #[inInfo.onnx[0]!, inInfo.onnx[1]!, 1], 3, inInfo.vane.w)
          else
            throw s!"{ctx}: 3D LayerNormalization axis {ax0} not supported (expected -1 or 2)"
        else if rank == 2 then
          if ax == 1 then
            pure ({ reduceC := true : ReduceAxes }, { n := 1, c := 1, h := 1, w := 1 : Shape4D },
                  #[1, 1], 1, inInfo.vane.c)
          else
            throw s!"{ctx}: 2D LayerNormalization axis {ax0} not supported (expected -1 or 1)"
        else if rank == 4 then
          match ax with
          | 1 =>
            if inInfo.vane.h != 1 || inInfo.vane.w != 1 then
              throw s!"{ctx}: 4D LayerNormalization with axis=1 normalizes over [C,H,W]={inInfo.onnx.extract 1 4 |>.toList}, which is only supported when H=1 and W=1 (got H={inInfo.vane.h}, W={inInfo.vane.w}); use axis=-1 (W) or reshape/transpose so the normalized axis is last"
            pure ({ reduceC := true : ReduceAxes }, { n := 1, c := 1, h := 1, w := 1 : Shape4D },
                  #[1, 1, 1, 1], 1, inInfo.vane.c)
          | 3 =>
            pure ({ reduceW := true : ReduceAxes }, { n := 1, c := inInfo.vane.c, h := inInfo.vane.h, w := 1 : Shape4D },
                  #[1, inInfo.vane.c, inInfo.vane.h, 1], 3, inInfo.vane.w)
          | _ =>
            throw s!"{ctx}: LayerNormalization axis {ax0} not supported (supported: C [axis=1 with H=W=1] or W [axis=3/-1])"
        else
          throw s!"{ctx}: LayerNormalization requires 4D, 3D, or 2D input, got rank {rank}"
      let eps := nd.orig.attrFloat "epsilon" 1e-5
      if eps.isNaN || eps.isInf || eps < 0.0 then
        throw s!"{ctx}: LayerNormalization invalid epsilon {eps}"
      let epsBits ← validateFiniteFp16Scalar eps ctx
      validateReduceBackendForNode opts.target .mean axes inInfo.vane ctx

      let redInfo : TensorShapeInfo := { vane := redVane, onnx := redOnnx }
      let (meanID, u1) := sanitizeNodeID s!"{baseID}_mean" usedIDs
      let (diffID, u2) := sanitizeNodeID s!"{baseID}_diff" u1
      let (sqID, u3) := sanitizeNodeID s!"{baseID}_sq" u2
      let (varID, u4) := sanitizeNodeID s!"{baseID}_var" u3
      let (varEpsID, u5) := sanitizeNodeID s!"{baseID}_vareps" u4
      let (invStdID, u6) := sanitizeNodeID s!"{baseID}_invstd" u5
      usedIDs := u6

      let (v1, s1, i1) := emit vaneNodes shapes vaneIDOf { id := meanID, inputs := #[inID], op := .reduce .mean axes 0 } "" redInfo
      let (v2, s2, i2) := emit v1 s1 i1 { id := diffID, inputs := #[inID, meanID], op := .pe2 .sub false } "" inInfo
      let (v3, s3, i3) := emit v2 s2 i2 { id := sqID, inputs := #[diffID, diffID], op := .pe2 .mul false } "" inInfo
      let (v4, s4, i4) := emit v3 s3 i3 { id := varID, inputs := #[sqID], op := .reduce .mean axes 0 } "" redInfo
      let (v5, s5, i5) := emit v4 s4 i4 { id := varEpsID, inputs := #[varID], op := .pe1 (.scalarAdd epsBits) } "" redInfo
      let (v6, s6, i6) := emit v5 s5 i5 { id := invStdID, inputs := #[varEpsID], op := .lut (.lut (.rsqrt 0.0)) } "" redInfo
      vaneNodes := v6; shapes := s6; vaneIDOf := i6

      -- Register optional Mean / InvStdDev outputs if present in ONNX node
      if nd.outputs.size >= 2 && !nd.outputs[1]!.isEmpty then
        shapes := setAssoc shapes nd.outputs[1]! redInfo
        vaneIDOf := setAssoc vaneIDOf nd.outputs[1]! meanID
      if nd.outputs.size >= 3 && !nd.outputs[2]!.isEmpty then
        shapes := setAssoc shapes nd.outputs[2]! redInfo
        vaneIDOf := setAssoc vaneIDOf nd.outputs[2]! invStdID

      -- Inspect Scale (input 1) and optional B (input 2)
      let scaleName := nd.inputs[1]!
      let biasName? := if nd.inputs.size >= 3 && !nd.inputs[2]!.isEmpty then some nd.inputs[2]! else none
      let scaleConst? := lookupAssoc consts scaleName
      let biasConst? := biasName?.bind (lookupAssoc consts)

      match scaleConst? with
      | some scCs =>
        if !isValidLayerNormParamShape scCs.dims normAxis normDim rank then
          throw s!"{ctx}: LayerNormalization Scale shape {scCs.dims.toList} incompatible with normalized suffix X.shape[{ax}:] (normDim={normDim})"
        let scVals ← scCs.float64s
        if scVals.size != 1 && scVals.size != normDim then
          throw s!"{ctx}: LayerNormalization Scale length {scVals.size} incompatible with normalized dim {normDim}"
        let biasVals? ← match biasName?, biasConst? with
          | none, _ => pure none
          | some _, some bc =>
            if !isValidLayerNormParamShape bc.dims normAxis normDim rank then
              throw s!"{ctx}: LayerNormalization B shape {bc.dims.toList} incompatible with normalized suffix X.shape[{ax}:] (normDim={normDim})"
            let bv ← bc.float64s
            if bv.size != 1 && bv.size != normDim then
              throw s!"{ctx}: LayerNormalization B length {bv.size} incompatible with normalized dim {normDim}"
            pure (some bv)
          | some bn, none =>
            throw s!"{ctx}: LayerNormalization bias '{bn}' must be constant when Scale is constant"

        let scUnif? := arrayUniformFloat scVals
        let bUnif? := match biasVals? with
          | none => some 0.0
          | some bv => arrayUniformFloat bv

        match scUnif?, bUnif? with
        | some sVal, some bVal =>
          if sVal == 1.0 && bVal == 0.0 then
            let (v', s', i') := emit vaneNodes shapes vaneIDOf {
              id := baseID, inputs := #[diffID, invStdID], op := .pe2 .mul false
            } nd.primaryOutput inInfo
            vaneNodes := v'; shapes := s'; vaneIDOf := i'
          else
            let (normID, u7) := sanitizeNodeID s!"{baseID}_norm" usedIDs
            usedIDs := u7
            let sBits ← validateFiniteFp16Scalar sVal ctx
            let bBits ← validateFiniteFp16Scalar bVal ctx
            let peOp :=
              if bVal == 0.0 then PeOp.scalarMul sBits
              else if sVal == 1.0 then PeOp.scalarAdd bBits
              else PeOp.linearActivation sBits bBits
            let (va, sa, ia) := emit vaneNodes shapes vaneIDOf { id := normID, inputs := #[diffID, invStdID], op := .pe2 .mul false } "" inInfo
            let (vb, sb, ib) := emit va sa ia { id := baseID, inputs := #[normID], op := .pe1 peOp } nd.primaryOutput inInfo
            vaneNodes := vb; shapes := sb; vaneIDOf := ib
        | _, _ =>
          -- Non-uniform constant per-element Scale/B
          let fullSc := if scVals.size == 1 then Array.replicate normDim scVals[0]! else scVals
          let fullB? := biasVals?.map fun bv => if bv.size == 1 then Array.replicate normDim bv[0]! else bv
          let mut diagW : Array Float := Array.replicate (normDim * normDim) 0.0
          for k in [:normDim] do
            diagW := diagW.set! (k * normDim + k) fullSc[k]!
          let wBytes := u16ArrayToBytes (← float64ArrayToFp16 diagW s!"{baseID}_weights")
          weightCount := weightCount + 1
          weightBytes := weightBytes + wBytes.size
          let bBytes? ← match fullB? with
            | some fb =>
              let bb := u16ArrayToBytes (← float64ArrayToFp16 fb s!"{baseID}_bias")
              weightCount := weightCount + 1
              weightBytes := weightBytes + bb.size
              pure (some bb)
            | none => pure none
          let (normID, u7) := sanitizeNodeID s!"{baseID}_norm" usedIDs
          usedIDs := u7
          let (va, sa, ia) := emit vaneNodes shapes vaneIDOf { id := normID, inputs := #[diffID, invStdID], op := .pe2 .mul false } "" inInfo
          if normAxis == 1 then
            let (vb, sb, ib) := emit va sa ia {
              id := baseID, inputs := #[normID], op := .conv1x1 normDim normDim wBytes bBytes? .none
            } nd.primaryOutput inInfo
            vaneNodes := vb; shapes := sb; vaneIDOf := ib
          else
            -- Axis W with non-uniform constant affine: supported via transpose C<->W when C == W <= 128 and H == 1
            if !opts.allowViewOps then
              throw s!"{ctx}: LayerNormalization along W with non-uniform constant scale/bias requires --allow-view-ops=true (or runtime scale/bias inputs)"
            if inInfo.vane.h != 1 || inInfo.vane.c != inInfo.vane.w || normDim > 128 then
              throw s!"{ctx}: LayerNormalization along W with non-uniform constant scale/bias requires C==W<=128 and H==1 (got {inInfo.vane.toArray.toList}); use runtime scale/bias inputs or normalize along C"
            let (tr1ID, u8) := sanitizeNodeID s!"{baseID}_tr1" usedIDs
            let (affID, u9) := sanitizeNodeID s!"{baseID}_aff" u8
            usedIDs := u9
            let (vb, sb, ib) := emit va sa ia { id := tr1ID, inputs := #[normID], op := .transpose #[0, 3, 2, 1] } "" inInfo
            let (vc, sc, ic) := emit vb sb ib { id := affID, inputs := #[tr1ID], op := .conv1x1 normDim normDim wBytes bBytes? .none } "" inInfo
            let (vd, sd, id_) := emit vc sc ic { id := baseID, inputs := #[affID], op := .transpose #[0, 3, 2, 1] } nd.primaryOutput inInfo
            vaneNodes := vd; shapes := sd; vaneIDOf := id_
      | none =>
        -- Runtime Scale (and optional runtime Bias)
        let (scInfo, scID) ← lookupShape shapes vaneIDOf scaleName ctx
        if !isValidLayerNormParamShape scInfo.onnx normAxis normDim rank then
          throw s!"{ctx}: LayerNormalization runtime Scale shape {scInfo.onnx.toList} incompatible with normalized suffix X.shape[{ax}:] (normDim={normDim})"
        let _ ← resolveElementwiseBroadcast inInfo scInfo ctx
        let (normID, u7) := sanitizeNodeID s!"{baseID}_norm" usedIDs
        usedIDs := u7
        let (va, sa, ia) := emit vaneNodes shapes vaneIDOf { id := normID, inputs := #[diffID, invStdID], op := .pe2 .mul false } "" inInfo
        match biasName? with
        | none =>
          let (vb, sb, ib) := emit va sa ia { id := baseID, inputs := #[normID, scID], op := .pe2 .mul false } nd.primaryOutput inInfo
          vaneNodes := vb; shapes := sb; vaneIDOf := ib
        | some bn =>
          let (bInfo, bID) ← lookupShape shapes vaneIDOf bn ctx
          if !isValidLayerNormParamShape bInfo.onnx normAxis normDim rank then
            throw s!"{ctx}: LayerNormalization runtime B shape {bInfo.onnx.toList} incompatible with normalized suffix X.shape[{ax}:] (normDim={normDim})"
          let _ ← resolveElementwiseBroadcast inInfo bInfo ctx
          let (scOutID, u8) := sanitizeNodeID s!"{baseID}_scaled" usedIDs
          usedIDs := u8
          let (vb, sb, ib) := emit va sa ia { id := scOutID, inputs := #[normID, scID], op := .pe2 .mul false } "" inInfo
          let (vc, sc, ic) := emit vb sb ib { id := baseID, inputs := #[scOutID, bID], op := .pe2 .add false } nd.primaryOutput inInfo
          vaneNodes := vc; shapes := sc; vaneIDOf := ic

    | "Split" =>
      if nd.inputs.size < 1 then throw s!"{ctx}: Split missing data input"
      if nd.outputs.isEmpty then throw s!"{ctx}: Split requires at least 1 output"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let rank := inInfo.onnx.size
      if rank != 4 && rank != 3 then
        throw s!"{ctx}: Split requires 4D or 3D input, got rank {rank}"
      let ax0 := nd.orig.attrInt "axis" 0
      let axNorm := if ax0 < 0 then ax0 + (rank : Int) else ax0
      let axis : Nat ←
        if rank == 3 then
          if axNorm == 2 then pure 3
          else throw s!"{ctx}: 3D Split axis {ax0} not supported (expected -1 or 2)"
        else
          if axNorm == 1 || axNorm == 2 || axNorm == 3 then pure axNorm.toNat
          else throw s!"{ctx}: Split axis {ax0} must be in \{1,2,3}"
      let dim := inInfo.vane.axis axis

      if let some numOutAttr := nd.orig.attr? "num_outputs" then
        if numOutAttr.i <= 0 || numOutAttr.i.toNat != nd.outputs.size then
          throw s!"{ctx}: Split num_outputs={numOutAttr.i} != node outputs count {nd.outputs.size}"

      let sizes : List Nat ←
        if nd.inputs.size >= 2 && !nd.inputs[1]!.isEmpty then
          -- Opset >= 13: constant split tensor input
          let some sc := lookupAssoc consts nd.inputs[1]!
            | throw s!"{ctx}: Split sizes input '{nd.inputs[1]!}' must be a constant int64 tensor"
          let raw ← sc.int64s
          let mut ls : List Nat := []
          for v in raw do
            if v <= 0 then throw s!"{ctx}: Split size {v} must be >= 1"
            ls := ls ++ [v.toNat]
          pure ls
        else if let some raw := nd.orig.attrInts? "split" then
          -- Opset < 13: split attribute
          let mut ls : List Nat := []
          for v in raw do
            if v <= 0 then throw s!"{ctx}: Split size {v} must be >= 1"
            ls := ls ++ [v.toNat]
          pure ls
        else
          -- Equal / opset-18 default split
          match computeDefaultSplitSizes dim nd.outputs.size with
          | .ok ls => pure ls
          | .error e => throw s!"{ctx}: {e}"

      if sizes.length != nd.outputs.size then
        throw s!"{ctx}: Split sizes length {sizes.length} != outputs count {nd.outputs.size}"
      let total := sumSizes sizes
      if total != dim then
        throw s!"{ctx}: Split sum(sizes)={total} != axis {axis} dim {dim}"
      let starts := prefixStarts sizes
      if axis == 3 then
        for i in [:starts.length] do
          let st := starts[i]!
          if st % 32 != 0 then
            throw s!"{ctx}: Split along W: output[{i}] start offset {st} is not a multiple of 32 (64-byte DMA alignment)"

      if sizes.length == 1 then
        shapes := setAssoc shapes nd.outputs[0]! inInfo
        vaneIDOf := setAssoc vaneIDOf nd.outputs[0]! inID
        elidedCount := elidedCount + 1
      else
        if !opts.allowViewOps then
          throw s!"{ctx}: non-no-op view op Split (axis={axis}, sizes={sizes}) requires --allow-view-ops=true"
        for i in [:nd.outputs.size] do
          let outName := nd.outputs[i]!
          let st := starts[i]!
          let sz := sizes[i]!
          let beg : Array Nat := (#[0, 0, 0, 0] : Array Nat).set! axis st
          let en : Array Nat := inInfo.vane.toArray.set! axis (st + sz)
          let step : Array Nat := #[1, 1, 1, 1]
          let vp ← planSlice inInfo.vane beg en step
          let outOnnx : Array Nat :=
            if rank == 3 then #[inInfo.onnx[0]!, inInfo.onnx[1]!, sz]
            else vp.outShape.toArray
          let outInfo : TensorShapeInfo := { vane := vp.outShape, onnx := outOnnx }
          if !outName.isEmpty && consumerCount finalCounts outName > 0 then
            let (partID, uNext) := sanitizeNodeID s!"{baseID}_p{i}" usedIDs
            usedIDs := uNext
            let (v', s', i') := emit vaneNodes shapes vaneIDOf {
              id := partID, inputs := #[inID], op := .slice beg en step
            } outName outInfo
            vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Transpose" =>
      if nd.inputs.size != 1 then throw s!"{ctx}: expected 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      if inInfo.onnx.size == 3 then
        let perm64 := (nd.orig.attrInts? "perm").getD #[2, 1, 0]
        if perm64.size != 3 then
          throw s!"{ctx}: 3D Transpose perm must have length 3, got {perm64.toList}"
        if perm64 == #[0, 1, 2] then
          shapes := setAssoc shapes nd.primaryOutput inInfo
          vaneIDOf := setAssoc vaneIDOf nd.primaryOutput inID
          elidedCount := elidedCount + 1
        else if perm64 == #[0, 2, 1] then
          if !opts.allowViewOps then
            throw s!"{ctx}: non-no-op view op Transpose (perm={perm64.toList}) requires --allow-view-ops=true"
          let h0 := inInfo.onnx[0]!; let r0 := inInfo.onnx[1]!; let c0 := inInfo.onnx[2]!
          let outVane := matmulStackedShape h0 c0 r0
          let outInfo : TensorShapeInfo := { vane := outVane, onnx := #[h0, c0, r0] }
          if h0 == 1 then
            let _ ← planTranspose inInfo.vane #[0, 3, 2, 1]
            let (v', s', i') := emit vaneNodes shapes vaneIDOf {
              id := baseID, inputs := #[inID], op := .transpose #[0, 3, 2, 1]
            } nd.primaryOutput outInfo
            vaneNodes := v'; shapes := s'; vaneIDOf := i'
          else
            -- Unpack [1, H*R, 1, C] -> [1, H, R, C] -> transpose [0, 1, 3, 2] -> [1, H, C, R] -> [1, H*C, 1, R]
            let sh4In := matmul4DShape h0 r0 c0
            let sh4Out := matmul4DShape h0 c0 r0
            let _ ← planReshape inInfo.vane sh4In
            let _ ← planTranspose sh4In #[0, 1, 3, 2]
            let _ ← planReshape sh4Out outVane
            let (uId, u1) := sanitizeNodeID s!"{baseID}_unpack" usedIDs
            let (tId, u2) := sanitizeNodeID s!"{baseID}_tr" u1
            usedIDs := u2
            let (v1, s1, i1) := emit vaneNodes shapes vaneIDOf { id := uId, inputs := #[inID], op := .reshape { c := h0, h := r0, w := c0 } } "" { vane := sh4In, onnx := sh4In.toArray }
            let (v2, s2, i2) := emit v1 s1 i1 { id := tId, inputs := #[uId], op := .transpose #[0, 1, 3, 2] } "" { vane := sh4Out, onnx := sh4Out.toArray }
            let (v3, s3, i3) := emit v2 s2 i2 { id := baseID, inputs := #[tId], op := .reshape { c := outVane.c, h := 1, w := outVane.w } } nd.primaryOutput outInfo
            vaneNodes := v3; shapes := s3; vaneIDOf := i3
        else
          throw s!"{ctx}: unsupported 3D Transpose perm {perm64.toList} (supported: [0,1,2] or [0,2,1])"
      else
        let perm64 := (nd.orig.attrInts? "perm").getD #[3, 2, 1, 0]
        if perm64.size != 4 then
          throw s!"{ctx}: perm must have length 4, got {perm64.toList}"
        let mut perm : Array Nat := Array.mkEmpty 4
        let mut seen : Array Bool := Array.replicate 4 false
        for p in perm64 do
          if p < 0 || p >= 4 || seen[p.toNat]! then
            throw s!"{ctx}: invalid permutation {perm64.toList}"
          seen := seen.set! p.toNat true
          perm := perm.push p.toNat
        if perm == #[0, 1, 2, 3] then
          shapes := setAssoc shapes nd.primaryOutput inInfo
          vaneIDOf := setAssoc vaneIDOf nd.primaryOutput inID
          elidedCount := elidedCount + 1
        else
          if !opts.allowViewOps then
            throw s!"{ctx}: non-no-op view op Transpose (perm={perm.toList}) requires --allow-view-ops=true"
          let vp ← planTranspose inInfo.vane perm
          if vp.outShape.n != 1 then
            throw s!"{ctx}: Transpose output batch dimension must be 1, got {vp.outShape.n}"
          let outInfo : TensorShapeInfo := { vane := vp.outShape, onnx := vp.outShape.toArray }
          let (v', s', i') := emit vaneNodes shapes vaneIDOf {
            id := baseID, inputs := #[inID], op := .transpose perm
          } nd.primaryOutput outInfo
          vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Concat" =>
      if nd.inputs.size < 2 then
        throw s!"{ctx}: Concat requires at least 2 inputs, got {nd.inputs.size}"
      let some axAttr := nd.orig.attr? "axis"
        | throw s!"{ctx}: Concat missing required axis attribute"
      let mut inIDs : Array String := Array.mkEmpty nd.inputs.size
      let mut inInfos : Array TensorShapeInfo := Array.mkEmpty nd.inputs.size
      let mut inShapes4 : Array Shape4D := Array.mkEmpty nd.inputs.size
      for inpName in nd.inputs do
        let (sh, vid) ← lookupShape shapes vaneIDOf inpName ctx
        inIDs := inIDs.push vid
        inInfos := inInfos.push sh
        inShapes4 := inShapes4.push sh.vane
      let rank := inInfos[0]!.onnx.size
      for i in [1:inInfos.size] do
        if inInfos[i]!.onnx.size != rank then
          throw s!"{ctx}: Concat input[{i}] rank {inInfos[i]!.onnx.size} != input[0] rank {rank}"
      if rank != 4 && rank != 3 && rank != 2 then
        throw s!"{ctx}: Concat requires 4D, 3D, or 2D inputs, got rank {rank}"
      let axInt := if axAttr.i < 0 then axAttr.i + (rank : Int) else axAttr.i
      let axis : Nat ←
        if rank == 3 then
          if axInt != 2 then
            throw s!"{ctx}: 3D Concat axis {axAttr.i} not supported (only last axis 2/-1 is supported on 3D [H,M,K])"
          let h0 := inInfos[0]!.onnx[0]!
          let m0 := inInfos[0]!.onnx[1]!
          for i in [1:inInfos.size] do
            if inInfos[i]!.onnx[0]! != h0 || inInfos[i]!.onnx[1]! != m0 then
              throw s!"{ctx}: 3D Concat input[{i}] shape {inInfos[i]!.onnx.toList} has mismatched (H,M) != ({h0},{m0})"
          pure 3
        else if rank == 2 then
          if axInt != 1 then
            throw s!"{ctx}: 2D Concat axis {axAttr.i} not supported (expected axis 1 or -1)"
          pure 1
        else
          if axInt < 1 || axInt > 3 then
            throw s!"{ctx}: Concat axis {axAttr.i} must be in \{1,2,3}"
          pure axInt.toNat
      let vp ← planConcat inShapes4 axis
      if !opts.allowViewOps then
        throw s!"{ctx}: non-no-op view op Concat (axis={axis}) requires --allow-view-ops=true"
      let outOnnx : Array Nat :=
        if rank == 3 then #[inInfos[0]!.onnx[0]!, inInfos[0]!.onnx[1]!, vp.outShape.w]
        else if rank == 2 then #[1, vp.outShape.c]
        else vp.outShape.toArray
      let outInfo : TensorShapeInfo := { vane := vp.outShape, onnx := outOnnx }
      let (v', s', i') := emit vaneNodes shapes vaneIDOf {
        id := baseID, inputs := inIDs, op := .concat axis
      } nd.primaryOutput outInfo
      vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Reshape" =>
      if nd.inputs.size != 2 then
        throw s!"{ctx}: Reshape requires 2 inputs (data, shape)"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let some sc := lookupAssoc consts nd.inputs[1]!
        | throw s!"{ctx}: Reshape target shape '{nd.inputs[1]!}' must be a constant int64 tensor"
      let rawTarget ← sc.int64s
      let allowZero := nd.orig.attrInt "allowzero" 0 != 0
      let outInfo ← resolveReshapeTarget inInfo rawTarget allowZero ctx
      if outInfo.vane == inInfo.vane then
        shapes := setAssoc shapes nd.primaryOutput outInfo
        vaneIDOf := setAssoc vaneIDOf nd.primaryOutput inID
        elidedCount := elidedCount + 1
      else
        if !opts.allowViewOps then
          throw s!"{ctx}: non-no-op view op Reshape ({inInfo.vane.toArray.toList} -> {outInfo.vane.toArray.toList}) requires --allow-view-ops=true"
        let _ ← planReshape inInfo.vane outInfo.vane
        let (v', s', i') := emit vaneNodes shapes vaneIDOf {
          id := baseID, inputs := #[inID]
          op := .reshape { c := outInfo.vane.c, h := outInfo.vane.h, w := outInfo.vane.w }
        } nd.primaryOutput outInfo
        vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Flatten" =>
      if nd.inputs.size != 1 then throw s!"{ctx}: Flatten requires 1 input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let ax0 := nd.orig.attrInt "axis" 1
      let (d0, d1) ← match computeFlattenShape inInfo.onnx ax0 with
        | .ok p => pure p
        | .error e => throw s!"{ctx}: {e}"
      if d0 != 1 then
        throw s!"{ctx}: Flatten on {inInfo.onnx.toList} with axis={ax0} produces batch dimension {d0} != 1 (only [1, C] output is supported)"
      let outVane : Shape4D := { n := 1, c := d1, h := 1, w := 1 }
      let outInfo : TensorShapeInfo := { vane := outVane, onnx := #[1, d1] }
      if outVane == inInfo.vane then
        shapes := setAssoc shapes nd.primaryOutput outInfo
        vaneIDOf := setAssoc vaneIDOf nd.primaryOutput inID
        elidedCount := elidedCount + 1
      else
        if !opts.allowViewOps then
          throw s!"{ctx}: non-no-op view op Flatten ({inInfo.vane.toArray.toList} -> {outVane.toArray.toList}) requires --allow-view-ops=true"
        let _ ← planReshape inInfo.vane outVane
        let (v', s', i') := emit vaneNodes shapes vaneIDOf {
          id := baseID, inputs := #[inID], op := .reshape { c := outVane.c, h := 1, w := 1 }
        } nd.primaryOutput outInfo
        vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | "Slice" =>
      if nd.inputs.size < 1 then throw s!"{ctx}: Slice missing data input"
      let (inInfo, inID) ← lookupShape shapes vaneIDOf nd.inputs[0]! ctx
      let (beg, en, step, outInfo) ← resolveSliceParams nd consts inInfo ctx
      if !opts.allowViewOps then
        throw s!"{ctx}: non-no-op view op Slice ({inInfo.vane.toArray.toList} -> {outInfo.vane.toArray.toList}) requires --allow-view-ops=true"
      let (v', s', i') := emit vaneNodes shapes vaneIDOf {
        id := baseID, inputs := #[inID], op := .slice beg en step
      } nd.primaryOutput outInfo
      vaneNodes := v'; shapes := s'; vaneIDOf := i'

    | other =>
      throw s!"{ctx}: unsupported ONNX operator '{other}'"

  -- Ensure graph has at least one node and ends at graphOutput.
  let (outInfo, outID0) ← lookupShape shapes vaneIDOf graphOutput "graph output"
  -- Validate lowered output ONNX shape against declared static output dimensions (if any).
  let declOut := g.outputs[0]!
  if declOut.hasShape && !declOut.dims.isEmpty then
    if outInfo.onnx.size != declOut.dims.size then
      throw s!"graph output '{declOut.name}': lowered ONNX shape {outInfo.onnx.toList} (rank {outInfo.onnx.size}) != declared output rank {declOut.dims.size}"
    for i in [:declOut.dims.size] do
      let d := declOut.dims[i]!
      if d.hasValue && d.value > 0 then
        if outInfo.onnx[i]! != d.value.toNat then
          throw s!"graph output '{declOut.name}': lowered ONNX shape {outInfo.onnx.toList} != declared output dim[{i}]={d.value}"
  let mut outID := outID0
  if vaneNodes.isEmpty || vaneNodes.back!.id != outID then
    let (copyID, uLast) := sanitizeNodeID s!"{outID}_out" usedIDs
    usedIDs := uLast
    vaneNodes := vaneNodes.push { id := copyID, inputs := #[outID], op := .lut .copy }
    outID := copyID

  let spec : GraphSpec := {
    target := opts.target
    inputs := graphInputs
    nodes := vaneNodes
    defaultPlacement := opts.placement
    stripMode := opts.stripMode
  }

  let summary : LowerSummary := {
    target := opts.target
    placement := opts.placement
    inputs := graphInputs
    output := outID
    outputShape := outInfo.vane
    onnxNodeCount := g.nodes.size
    nodeCount := vaneNodes.size
    fusedBnCount := fusedBnCount
    fusedActCount := fusedActCount
    fusedMatmulCount := fusedMatmulCount
    elidedCount := elidedCount
    weightCount := weightCount
    weightBytes := weightBytes
    safeTensorsUsed := sfUsed
    warnings := warnings
  }

  pure { spec := spec, summary := summary }

/-- Lower an `OnnxModel` in memory and compile it directly into a verified HWX + 4 LOAD sections. -/
def compileOnnx (model : OnnxModel) (opts : LowerOptions := {}) : Except String CompiledOnnx := do
  let lowered ← lowerOnnx model opts
  let compiled ← compileGraph lowered.spec
  pure { lowered := lowered, compiled := compiled }

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
