import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Hwx
import Vane.Sections
import Vane.Register
import Vane.Reach
import Vane.Alloc
import Vane.Tiling
import Vane.Quant
import Vane.Program
import Vane.Verify
import Vane.Pe
import Vane.Lut
import Vane.Pool
import Vane.Conv
import Vane.View
import Vane.Matmul
import Vane.IR.Ops
import Vane.IR.Graph
import Vane.Ane.Types
import Vane.Machine.Types
import Vane.Backend.SyncAndEmit

namespace Vane.Graph

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Tiling
open Vane.Quant
open Vane.Verify
open Vane.Pe
open Vane.Lut
open Vane.Pool
open Vane.Conv
open Vane.View
open Vane.Matmul
open Vane.IR
open Vane.Ane
open Vane.Machine
open Vane.Backend.SyncAndEmit

-- Re-export `EdgePlacementMode` and `StripMode` (with constructors & methods) from `Vane.Ane`
-- so all existing callers using `Vane.Graph.EdgePlacementMode.*` / `StripMode.*` remain source-compatible.
export Vane.Ane (EdgePlacementMode StripMode)
namespace EdgePlacementMode
export Vane.Ane.EdgePlacementMode (auto l2 dram name parse?)
end EdgePlacementMode

namespace StripMode
export Vane.Ane.StripMode (off auto fixed name parse?)
end StripMode

-- ============================================================================
-- Multi-Task DAG Graph IR (Thin Adapter over L2 AneIR -> L3 MachineIR -> L4 SyncAndEmit)
-- ============================================================================

/-- Network input surface specification in a multi-task graph. -/
structure GraphInput where
  name : String
  shape : TensorDims
  shortName : String := ""
  deriving Repr, BEq, Inhabited

/-- Supported node operations in a multi-task ANE DAG. -/
inductive NodeOp where
  | conv1x1 (cin cout : Nat) (weights : ByteArray) (bias : Option ByteArray := none) (act : ConvActivation := .none) (quant : Option QuantWeightSpec := none)
  | conv3x3 (mode : Conv3x3Mode) (cin cout stride : Nat) (padding : PadMode) (weights : ByteArray) (bias : Option ByteArray := none) (act : ConvActivation := .none) (quant : Option QuantWeightSpec := none)
  | linear (cin cout : Nat) (weights : ByteArray) (bias : Option ByteArray := none) (act : ConvActivation := .none) (quant : Option QuantWeightSpec := none)
  | pe1 (op : PeOp)
  | pe2 (op : PeBinOp) (rev : Bool := false)
  | lut (act : NeActSpec)
  | pool (op : PoolKind) (kh kw sh sw : Nat) (pad : PoolPad) (excludePad : Bool := true)
  | reduce (op : ReduceOp) (axes : ReduceAxes) (eps : UInt16 := 0)
  | transpose (perm : Array Nat)
  | slice (beginIdx endIdx step : Array Nat)
  | concat (axis : Nat)
  | reshape (outDims : TensorDims)
  | matmul (heads : Nat := 1) (transposeA : Bool := false) (transposeB : Bool := false) (scaleBits : UInt16 := f16PosOne)
  deriving BEq

instance : Inhabited NodeOp := ⟨.pe2 .add false⟩

/-- One node in topological order inside a multi-task DAG. -/
structure GraphNode where
  id : String
  inputs : Array String
  op : NodeOp
  outPlacement : Option EdgePlacementMode := none
  deriving BEq, Inhabited

/-- Complete multi-task DAG specification for compilation.
    When `outputs` is empty, the final node in `nodes` is the sole graph output (backwards-compatible default).
    When `outputs` is non-empty (`1..3` node IDs, with `inputs.size + outputs.size <= 4`), each listed node
    is emitted to a distinct HWX output surface. -/
structure GraphSpec where
  target : Target
  inputs : Array GraphInput
  nodes : Array GraphNode
  defaultPlacement : EdgePlacementMode := .auto
  stripMode : StripMode := .off
  outputs : Array String := #[]
  donations : Array (Nat × Nat) := #[]
  deriving BEq

/-- Concrete storage location assigned to a graph tensor. -/
inductive TensorLocation where
  | surface (surfIndex : Nat) (barSlot : UInt8)
  | l2 (base : Nat) (bytes : Nat)
  | dram (offset : Nat) (bytes : Nat)
  deriving Repr, BEq, Inhabited

def TensorLocation.placementName : TensorLocation → String
  | .surface _ _ => "surface"
  | .l2 _ _ => "l2"
  | .dram _ _ => "dram"

/-- Compilation metadata for one lowered node in a compiled multi-task graph. -/
structure CompiledGraphNode where
  id : String
  taskId : Nat
  engine : Engine
  opKind : String
  outDims : TensorDims
  outLocation : TensorLocation
  direct : Bool
  l2UseBytes : Nat
  deriving Repr, BEq, Inhabited

/-- Complete compiled multi-task graph artifact, verified by `Vane.Verify.verifyHwxTasksSummary`. -/
structure CompiledGraph where
  target : Target
  taskCount : Nat
  scratchBytes : Nat
  l2PeakBytes : Nat
  kernelBytes : Nat
  stripCount : Nat := 1
  stripHeight : Nat := 0
  nodes : Array CompiledGraphNode
  hwx : ByteArray
  info : HwxInfo
  sections : LoadSections
  aneProgram : Option AneProgram := none
  machineProgram : Option MachineProgram := none

/-- Convert an `Array UInt16` of fp16 bit patterns into little-endian bytes. -/
def u16ArrayToBytes (arr : Array UInt16) : ByteArray := Id.run do
  let mut bs := ByteArray.emptyWithCapacity (2 * arr.size)
  for w in arr do
    bs := pushU16LE bs w
  return bs

-- ============================================================================
-- Adapter: `GraphSpec` -> `AneProgram` (L2 Value-Semantic ANE Engine IR)
-- ============================================================================

private def lookupAssoc {α : Type} (arr : Array (String × α)) (k : String) : Option α :=
  (arr.find? fun p => p.1 == k).map Prod.snd

private def validateGraphConvWeightData (ctx : String) (cs : ConvSpec)
    (weights : ByteArray) (bias : Option ByteArray) (quant? : Option QuantWeightSpec) :
    Except String WeightData := do
  if let some b := bias then
    if b.size != 2 * cs.cout then
      throw s!"{ctx}: bias byte size {b.size} != expected {2 * cs.cout}"
  let cinPerG := if cs.effMode == Conv3x3Mode.depthwise then 1 else cs.cin
  let kArea := cs.kernelSize * cs.kernelSize
  match quant? with
  | none =>
    let expBytes := 2 * cs.cout * cinPerG * kArea
    if weights.size != expBytes then
      throw s!"{ctx}: weights byte size {weights.size} != expected {expBytes}"
    pure (.fp16 weights)
  | some q =>
    validateQuantWeightSpec q cs.cout cinPerG kArea
    pure (.quant q)

/-- Translate a `GraphSpec` JSON/struct DAG into a typed `AneProgram` (`SSAGraph AneOp`). -/
def graphSpecToAne (spec : GraphSpec) : Except String AneProgram := do
  if spec.inputs.isEmpty || spec.inputs.size > 3 then
    throw s!"graph must have 1..3 inputs, got {spec.inputs.size}"
  if (spec.nodes.isEmpty && spec.outputs.size < 2) || spec.nodes.size > 256 then
    throw s!"graph must have 1..256 nodes, got {spec.nodes.size}"

  let mut nameToVid : Array (String × ValueId) := #[]
  let mut nameToDims : Array (String × TensorDims) := #[]
  let mut ssaInputs : Array (ValueId × String × TensorType) := Array.mkEmpty spec.inputs.size
  let mut inShorts : Array String := Array.mkEmpty spec.inputs.size
  let mut nextVid : ValueId := 0

  for i in [:spec.inputs.size] do
    let inp := spec.inputs[i]!
    if inp.name.isEmpty then
      throw s!"graph input[{i}] has empty name"
    if (lookupAssoc nameToVid inp.name).isSome then
      throw s!"duplicate graph input name '{inp.name}'"
    if inp.shape.c == 0 || inp.shape.h == 0 || inp.shape.w == 0 then
      throw s!"graph input '{inp.name}' has zero dimension ({inp.shape.c}x{inp.shape.h}x{inp.shape.w})"
    let vid := nextVid; nextVid := nextVid + 1
    let sh : Shape4D := { n := 1, c := inp.shape.c, h := inp.shape.h, w := inp.shape.w }
    ssaInputs := ssaInputs.push (vid, inp.name, shape4DToTensorType sh)
    inShorts := inShorts.push inp.shortName
    nameToVid := nameToVid.push (inp.name, vid)
    nameToDims := nameToDims.push (inp.name, inp.shape)

  let mut aneNodes : Array (IRNode AneOp) := Array.mkEmpty spec.nodes.size
  let mut nodeMeta : Array AneNodeMeta := Array.mkEmpty spec.nodes.size

  let lookupInput (vidMap : Array (String × ValueId)) (dimMap : Array (String × TensorDims))
      (nodeId inpName : String) := do
    match lookupAssoc vidMap inpName, lookupAssoc dimMap inpName with
    | some v, some d => pure (v, d)
    | _, _ => throw s!"node '{nodeId}' references unknown input '{inpName}'"

  for ni in [:spec.nodes.size] do
    let nd := spec.nodes[ni]!
    if nd.id.isEmpty then
      throw s!"graph node[{ni}] has empty id"
    if (lookupAssoc nameToVid nd.id).isSome then
      throw s!"duplicate node/input id '{nd.id}'"

    let place := nd.outPlacement
    let wgId := ni + 1
    let lookup := lookupInput nameToVid nameToDims

    let (outVid, outDims) : ValueId × TensorDims ← match nd.op with
      | .conv1x1 cin cout weights bias act quant? | .linear cin cout weights bias act quant? =>
        let opNm := match nd.op with | .linear .. => "linear" | _ => "conv1x1"
        if nd.inputs.size != 1 then
          throw s!"node '{nd.id}' ({opNm}) expects 1 input, got {nd.inputs.size}"
        let (xVid, xDims) ← lookup nd.id nd.inputs[0]!
        if xDims.c != cin then
          throw s!"node '{nd.id}' ({opNm}): input channels {xDims.c} != cin={cin}"
        let cs : ConvSpec := {
          kernelSize := 1, mode := .dense, h := xDims.h, w := xDims.w,
          cin := cin, cout := cout, stride := 1, pad := .same
        }
        validateConvShape cs
        let wd ← validateGraphConvWeightData s!"node '{nd.id}' ({opNm})" cs weights bias quant?
        let oDims : TensorDims := { c := cout, h := xDims.h, w := xDims.w }
        let ov := nextVid; nextVid := nextVid + 1
        aneNodes := aneNodes.push {
          name := nd.id
          op := .neConv 1 1 1 1 .same .dense cin cout wd bias act none wgId opNm
          inputs := #[xVid]
          outputs := #[(ov, shape4DToTensorType { n := 1, c := oDims.c, h := oDims.h, w := oDims.w })]
        }
        nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := place }
        pure (ov, oDims)

      | .conv3x3 mode cin cout stride pad weights bias act quant? =>
        if nd.inputs.size != 1 then
          throw s!"node '{nd.id}' (conv3x3) expects 1 input, got {nd.inputs.size}"
        let (xVid, xDims) ← lookup nd.id nd.inputs[0]!
        if xDims.c != cin then
          throw s!"node '{nd.id}' (conv3x3): input channels {xDims.c} != cin={cin}"
        let cs : ConvSpec := {
          kernelSize := 3, mode := mode, h := xDims.h, w := xDims.w,
          cin := cin, cout := cout, stride := stride, pad := pad
        }
        validateConvShape cs
        let wd ← validateGraphConvWeightData s!"node '{nd.id}' (conv3x3)" cs weights bias quant?
        let oDims : TensorDims := { c := cout, h := cs.outH, w := cs.outW }
        let ov := nextVid; nextVid := nextVid + 1
        let cMode : ConvMode := match mode with | .dense => .dense | .depthwise => .depthwise
        aneNodes := aneNodes.push {
          name := nd.id
          op := .neConv 3 3 stride stride pad cMode cin cout wd bias act none wgId "conv3x3"
          inputs := #[xVid]
          outputs := #[(ov, shape4DToTensorType { n := 1, c := oDims.c, h := oDims.h, w := oDims.w })]
        }
        nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := place }
        pure (ov, oDims)

      | .pe1 op =>
        if nd.inputs.size != 1 then
          throw s!"node '{nd.id}' (pe1) expects 1 input, got {nd.inputs.size}"
        if op.isTwoSource then
          throw s!"node '{nd.id}' (pe1) cannot use two-source op"
        let (xVid, xDims) ← lookup nd.id nd.inputs[0]!
        let ov := nextVid; nextVid := nextVid + 1
        aneNodes := aneNodes.push {
          name := nd.id
          op := .pe (.unary op) none
          inputs := #[xVid]
          outputs := #[(ov, shape4DToTensorType { n := 1, c := xDims.c, h := xDims.h, w := xDims.w })]
        }
        nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := place }
        pure (ov, xDims)

      | .pe2 binOp rev =>
        if nd.inputs.size != 2 then
          throw s!"node '{nd.id}' (pe2) expects 2 inputs, got {nd.inputs.size}"
        let (xVid, d1) ← lookup nd.id nd.inputs[0]!
        let (yVid, d2) ← lookup nd.id nd.inputs[1]!
        let oDims ← broadcastDims d1 d2
        let ov := nextVid; nextVid := nextVid + 1
        aneNodes := aneNodes.push {
          name := nd.id
          op := .pe (.binary binOp rev) none
          inputs := #[xVid, yVid]
          outputs := #[(ov, shape4DToTensorType { n := 1, c := oDims.c, h := oDims.h, w := oDims.w })]
        }
        nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := place }
        pure (ov, oDims)

      | .lut act =>
        if nd.inputs.size != 1 then
          throw s!"node '{nd.id}' (lut) expects 1 input, got {nd.inputs.size}"
        let (xVid, xDims) ← lookup nd.id nd.inputs[0]!
        let ov := nextVid; nextVid := nextVid + 1
        aneNodes := aneNodes.push {
          name := nd.id
          op := .neAct (.act act) none wgId
          inputs := #[xVid]
          outputs := #[(ov, shape4DToTensorType { n := 1, c := xDims.c, h := xDims.h, w := xDims.w })]
        }
        nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := place }
        pure (ov, xDims)

      | .pool op kh kw sh sw pad excl =>
        if nd.inputs.size != 1 then
          throw s!"node '{nd.id}' (pool) expects 1 input, got {nd.inputs.size}"
        let (xVid, xDims) ← lookup nd.id nd.inputs[0]!
        let pspec : PoolSpec := {
          target := spec.target, op := op, dims := xDims,
          kh := kh, kw := kw, sh := sh, sw := sw, pad := pad, excludePad := excl
        }
        let (backend, gh, gw) ← resolvePoolBackend pspec
        let oDims : TensorDims := { c := xDims.c, h := gh.out, w := gw.out }
        let ov := nextVid; nextVid := nextVid + 1
        let aOp : AneOp := match backend with
          | .pePool => .pe (.pool op kh kw sh sw pad excl) none
          | .nePool => .neAct (.nePool op kh kw sh sw pad excl) none wgId
          | .peReduceGlobal =>
            let rOp : ReduceOp := match op with | .max => .max | .avg => .mean
            .pe (.reduce rOp { reduceC := false, reduceH := true, reduceW := true } 0) none
        aneNodes := aneNodes.push {
          name := nd.id
          op := aOp
          inputs := #[xVid]
          outputs := #[(ov, shape4DToTensorType { n := 1, c := oDims.c, h := oDims.h, w := oDims.w })]
        }
        nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := place }
        pure (ov, oDims)

      | .reduce op axes eps =>
        if nd.inputs.size != 1 then
          throw s!"node '{nd.id}' (reduce) expects 1 input, got {nd.inputs.size}"
        let (xVid, xDims) ← lookup nd.id nd.inputs[0]!
        let rspec : ReduceSpec := {
          target := spec.target, op := op, dims := xDims, axes := axes, eps := eps
        }
        let (backend, oDims) ← resolveReduceBackend rspec
        let ov := nextVid; nextVid := nextVid + 1
        let aOp : AneOp := match backend with
          | .peReduce => .pe (.reduce op axes eps) none
          | .neChannelReduce => .neAct (.channelReduce op) none wgId
        aneNodes := aneNodes.push {
          name := nd.id
          op := aOp
          inputs := #[xVid]
          outputs := #[(ov, shape4DToTensorType { n := 1, c := oDims.c, h := oDims.h, w := oDims.w })]
        }
        nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := place }
        pure (ov, oDims)

      | .transpose perm =>
        if nd.inputs.size != 1 then
          throw s!"node '{nd.id}' (transpose) expects 1 input, got {nd.inputs.size}"
        let (xVid, xDims) ← lookup nd.id nd.inputs[0]!
        let inSh : Shape4D := { n := 1, c := xDims.c, h := xDims.h, w := xDims.w }
        let vp ← planTranspose inSh perm
        if vp.outShape.n != 1 then
          throw s!"node '{nd.id}' (transpose): output requires N=1, got {vp.outShape.n}"
        let oDims : TensorDims := { c := vp.outShape.c, h := vp.outShape.h, w := vp.outShape.w }
        let ov := nextVid; nextVid := nextVid + 1
        aneNodes := aneNodes.push {
          name := nd.id
          op := .dmaView (dmaViewSpecFromPlan inSh vp) "transpose"
          inputs := #[xVid]
          outputs := #[(ov, shape4DToTensorType vp.outShape)]
        }
        nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := place }
        pure (ov, oDims)

      | .slice beg en st =>
        if nd.inputs.size != 1 then
          throw s!"node '{nd.id}' (slice) expects 1 input, got {nd.inputs.size}"
        let (xVid, xDims) ← lookup nd.id nd.inputs[0]!
        let inSh : Shape4D := { n := 1, c := xDims.c, h := xDims.h, w := xDims.w }
        let vp ← planSlice inSh beg en st
        if vp.outShape.n != 1 then
          throw s!"node '{nd.id}' (slice): output requires N=1, got {vp.outShape.n}"
        let isContigC := beg.size == 4 && en.size == 4 && st.size == 4 &&
          beg[0]! == 0 && en[0]! == 1 && beg[2]! == 0 && en[2]! == xDims.h &&
          beg[3]! == 0 && en[3]! == xDims.w &&
          st[0]! == 1 && st[1]! == 1 && st[2]! == 1 && st[3]! == 1
        let oDims : TensorDims := { c := vp.outShape.c, h := vp.outShape.h, w := vp.outShape.w }
        let ov := nextVid; nextVid := nextVid + 1
        aneNodes := aneNodes.push {
          name := nd.id
          op := .dmaView (dmaViewSpecFromPlan inSh vp (aliasable := isContigC)) "slice"
          inputs := #[xVid]
          outputs := #[(ov, shape4DToTensorType vp.outShape)]
        }
        nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := place }
        pure (ov, oDims)

      | .concat axis =>
        if nd.inputs.size < 2 then
          throw s!"node '{nd.id}' (concat) expects >= 2 inputs, got {nd.inputs.size}"
        let mut inVids : Array ValueId := Array.mkEmpty nd.inputs.size
        let mut inShapes : Array Shape4D := Array.mkEmpty nd.inputs.size
        for inpNm in nd.inputs do
          let (v, d) ← lookup nd.id inpNm
          inVids := inVids.push v
          inShapes := inShapes.push { n := 1, c := d.c, h := d.h, w := d.w }
        let vp ← planConcat inShapes axis
        let oDims : TensorDims := { c := vp.outShape.c, h := vp.outShape.h, w := vp.outShape.w }
        let ov := nextVid; nextVid := nextVid + 1
        let desc : DmaViewSpec := {
          inShape := inShapes[0]!
          outShape := vp.outShape
          concatAxis? := some axis
          concatInShapes := inShapes
          aliasable := axis == 1
        }
        aneNodes := aneNodes.push {
          name := nd.id
          op := .dmaView desc "concat"
          inputs := inVids
          outputs := #[(ov, shape4DToTensorType vp.outShape)]
        }
        nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := place }
        pure (ov, oDims)

      | .reshape targetDims =>
        if nd.inputs.size != 1 then
          throw s!"node '{nd.id}' (reshape) expects 1 input, got {nd.inputs.size}"
        let (xVid, xDims) ← lookup nd.id nd.inputs[0]!
        let inSh : Shape4D := { n := 1, c := xDims.c, h := xDims.h, w := xDims.w }
        let outSh : Shape4D := { n := 1, c := targetDims.c, h := targetDims.h, w := targetDims.w }
        let vp ← planReshape inSh outSh
        let ov := nextVid; nextVid := nextVid + 1
        aneNodes := aneNodes.push {
          name := nd.id
          op := .dmaView (dmaViewSpecFromPlan inSh vp) "reshape"
          inputs := #[xVid]
          outputs := #[(ov, shape4DToTensorType vp.outShape)]
        }
        nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := place }
        pure (ov, targetDims)

      | .matmul heads trA trB scaleBits =>
        if nd.inputs.size != 2 then
          throw s!"node '{nd.id}' (matmul) expects 2 inputs, got {nd.inputs.size}"
        let (aVid0, da) ← lookup nd.id nd.inputs[0]!
        let (bVid0, db) ← lookup nd.id nd.inputs[1]!
        if heads == 0 then
          throw s!"node '{nd.id}' (matmul): heads must be >= 1"
        if da.h != 1 || db.h != 1 then
          throw s!"node '{nd.id}' (matmul): requires H=1 for both inputs (got A.h={da.h}, B.h={db.h})"
        if da.c % heads != 0 || db.c % heads != 0 then
          throw s!"node '{nd.id}' (matmul): input channels ({da.c}, {db.c}) not divisible by heads={heads}"
        let (m, ka) := if trA then (da.w, da.c / heads) else (da.c / heads, da.w)
        let (kb, nCols) := if trB then (db.w, db.c / heads) else (db.c / heads, db.w)
        if ka != kb then
          throw s!"node '{nd.id}' (matmul): inner dimension mismatch A.K={ka} != B.K={kb}"
        let mmSpec : MatmulSpec := {
          b := heads, m := m, k := ka, n := nCols,
          transposeA := trA, transposeB := trB, scaleBits := scaleBits
        }
        validateMatmulSpec mmSpec

        -- Emit transpose dmaView if trA / trB
        let aVid ←
          if trA then
            let (trCp, minBytes) ← planMatmulScratchTranspose heads ka m
            let tv := nextVid; nextVid := nextVid + 1
            let trOutSh : Shape4D := { n := 1, c := heads * m, h := 1, w := ka }
            let aInSh : Shape4D := { n := 1, c := da.c, h := 1, w := da.w }
            aneNodes := aneNodes.push {
              name := s!"{nd.id}_trA"
              op := .dmaView { inShape := aInSh, outShape := trOutSh, transposeCW := trCp.transposeCW, batchN := heads, explicitCopy? := some trCp, minDramBytes := minBytes } "matmul_trA"
              inputs := #[aVid0]
              outputs := #[(tv, shape4DToTensorType trOutSh)]
            }
            nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := some .dram }
            pure tv
          else pure aVid0

        let bVid ←
          if trB then
            let (trCp, minBytes) ← planMatmulScratchTranspose heads nCols ka
            let tv := nextVid; nextVid := nextVid + 1
            let trOutSh : Shape4D := { n := 1, c := heads * ka, h := 1, w := nCols }
            let bInSh : Shape4D := { n := 1, c := db.c, h := 1, w := db.w }
            aneNodes := aneNodes.push {
              name := s!"{nd.id}_trB"
              op := .dmaView { inShape := bInSh, outShape := trOutSh, transposeCW := trCp.transposeCW, batchN := heads, explicitCopy? := some trCp, minDramBytes := minBytes } "matmul_trB"
              inputs := #[bVid0]
              outputs := #[(tv, shape4DToTensorType trOutSh)]
            }
            nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := some .dram }
            pure tv
          else pure bVid0

        let oDims : TensorDims := { c := heads * m, h := 1, w := nCols }
        let fullOutSh : Shape4D := { n := 1, c := oDims.c, h := 1, w := oDims.w }
        if m <= 16 then
          let ov := nextVid; nextVid := nextVid + 1
          aneNodes := aneNodes.push {
            name := nd.id
            op := .neMatmul heads m ka nCols false false scaleBits 0 m
            inputs := #[aVid, bVid]
            outputs := #[(ov, shape4DToTensorType fullOutSh)]
          }
          nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := place }
          pure (ov, oDims)
        else
          let numSlices := ceilDiv m 16
          let mut slVids : Array ValueId := Array.mkEmpty numSlices
          let mut slShapes : Array Shape4D := Array.mkEmpty numSlices
          for si in [:numSlices] do
            let m0 := si * 16
            let curM := min 16 (m - m0)
            let slSh : Shape4D := { n := 1, c := heads * curM, h := 1, w := nCols }
            let sv := nextVid; nextVid := nextVid + 1
            aneNodes := aneNodes.push {
              name := s!"{nd.id}_m{si}"
              op := .neMatmul heads curM ka nCols false false scaleBits m0 m
              inputs := #[aVid, bVid]
              outputs := #[(sv, shape4DToTensorType slSh)]
            }
            nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := place }
            slVids := slVids.push sv
            slShapes := slShapes.push slSh
          let ov := nextVid; nextVid := nextVid + 1
          aneNodes := aneNodes.push {
            name := s!"{nd.id}_join"
            op := .dmaView { inShape := slShapes[0]!, outShape := fullOutSh, concatAxis? := some 1, concatInShapes := slShapes, aliasable := true } "matmul_join"
            inputs := slVids
            outputs := #[(ov, shape4DToTensorType fullOutSh)]
          }
          nodeMeta := nodeMeta.push { logicalId := nd.id, outPlacement := place }
          pure (ov, oDims)

    nameToVid := nameToVid.push (nd.id, outVid)
    nameToDims := nameToDims.push (nd.id, outDims)

  -- Resolve graph output ValueIds (default: final node; or explicit `spec.outputs` for multi-output!)
  let outNames : Array String :=
    if spec.outputs.isEmpty then #[spec.nodes[spec.nodes.size - 1]!.id]
    else spec.outputs
  if outNames.isEmpty then
    throw "graph must have at least 1 output"

  let mut outVids : Array ValueId := Array.mkEmpty outNames.size
  for oi in [:outNames.size] do
    let oNm := outNames[oi]!
    let some ov := lookupAssoc nameToVid oNm
      | throw s!"graph output[{oi}] '{oNm}' not found among graph nodes"
    -- For single-output graphs where the output aliases a graph input, insert an explicit copy.
    -- For multi-output graphs (`outNames.size >= 2`), `MemoryPlan` packs all logical outputs into
    -- the single ABI v3 HWX output surface (handling direct zero-copy or `pack_out` DMA stores).
    if outNames.size == 1 && spec.inputs.any (fun inp => inp.name == oNm) then
      let some d := lookupAssoc nameToDims oNm
        | throw s!"graph output[{oi}] '{oNm}' missing dimensions"
      let cv := nextVid; nextVid := nextVid + 1
      aneNodes := aneNodes.push {
        name := s!"out_copy_{oi}"
        op := .neAct (.act .copy) none (spec.nodes.size + oi + 1)
        inputs := #[ov]
        outputs := #[(cv, shape4DToTensorType { n := 1, c := d.c, h := d.h, w := d.w })]
      }
      nodeMeta := nodeMeta.push { logicalId := oNm, outPlacement := none }
      outVids := outVids.push cv
    else
      outVids := outVids.push ov

  -- When explicit `spec.outputs` is provided, prune any dead/unused trailing nodes via DCE
  -- (preserving aligned `nodeMeta`) so unused nodes do not trip MemoryPlan dead-intermediate checks.
  let (finalNodes, finalMeta) :=
    if spec.outputs.isEmpty then
      (aneNodes, nodeMeta)
    else Id.run do
      let mut live : Array Bool := Array.replicate (nextVid + 1) false
      for ov in outVids do
        if ov < live.size then live := live.set! ov true
      for step in [:aneNodes.size] do
        let idx := aneNodes.size - 1 - step
        let nd := aneNodes[idx]!
        if nd.outputs.any (fun (v, _) => v < live.size && live[v]!) then
          for iv in nd.inputs do
            if iv < live.size then live := live.set! iv true
      let mut keptNodes : Array (IRNode AneOp) := #[]
      let mut keptMeta : Array AneNodeMeta := #[]
      for idx in [:aneNodes.size] do
        let nd := aneNodes[idx]!
        if nd.outputs.any (fun (v, _) => v < live.size && live[v]!) then
          keptNodes := keptNodes.push nd
          keptMeta := keptMeta.push (nodeMeta.getD idx {})
      return (keptNodes, keptMeta)

  pure {
    graph := {
      target := spec.target
      inputs := ssaInputs
      nodes := finalNodes
      outputs := outVids
    }
    defaultPlacement := spec.defaultPlacement
    nodeMeta := finalMeta
    inputShortNames := inShorts
    outputNames := outNames
    donations := spec.donations
  }

/-- Convert a `CompiledMachineProgram` (L4 output) into the public `CompiledGraph` record. -/
def compiledMachineToGraph
    (cm : CompiledMachineProgram)
    (ane? : Option AneProgram := none) : CompiledGraph :=
  let mp := cm.machine
  let compiledNodes : Array CompiledGraphNode := mp.tasks.map fun mt =>
    let outLoc : TensorLocation := match resolveBufferStorage mp.buffers mt.output with
      | .ok (.surface surfIdx _ barSlot _ _ _) => .surface surfIdx barSlot
      | .ok (.l2 base bytes) => .l2 base bytes
      | .ok (.dramScratch off bytes) => .dram off bytes
      | .error _ => .dram 0 0
    {
      id := mt.name
      taskId := mt.id
      engine := mt.engine
      opKind := mt.opKind
      outDims := shape4DToDims mt.outShape
      outLocation := outLoc
      direct := mt.tilePlan.direct
      l2UseBytes := mt.tilePlan.l2Peak
    }
  {
    target := mp.target
    taskCount := mp.tasks.size
    scratchBytes := mp.scratchBytes
    l2PeakBytes := mp.l2PeakBytes
    kernelBytes := mp.kernelBytes
    stripCount := mp.stripCount
    stripHeight := mp.stripHeight
    nodes := compiledNodes
    hwx := cm.hwx
    info := cm.info
    sections := cm.sections
    aneProgram := ane?
    machineProgram := some mp
  }

/-- Compile an arbitrary multi-task DAG (`1..256` tasks, `1..3` inputs, `1..3` outputs, `inputs + outputs <= 4`)
    through L2 (`AneIR`) → L3 (`MachineIR` + `MemoryPlan`) → L4 (`SyncAndEmit`) into a statically
    verified HWX binary and its 4 host-built Linux LOAD sections. -/
def compileGraph (spec : GraphSpec) : Except String CompiledGraph := do
  let aneProg ← graphSpecToAne spec
  let (finalAne, cm) ← compileAneProgram aneProg spec.stripMode
  pure (compiledMachineToGraph cm (some finalAne))

end Vane.Graph
