import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Hwx
import Vane.Sections
import Vane.Register
import Vane.Reach
import Vane.Alloc
import Vane.Tiling
import Vane.Program
import Vane.Verify
import Vane.Pe
import Vane.Lut
import Vane.Conv

namespace Vane.Graph

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Reach
open Vane.Alloc
open Vane.Tiling
open Vane.Program
open Vane.Verify
open Vane.Pe
open Vane.Lut
open Vane.Conv

-- ============================================================================
-- Multi-Task DAG Graph IR
-- ============================================================================

/-- Network input surface specification in a multi-task graph. -/
structure GraphInput where
  name : String
  shape : TensorDims
  shortName : String := ""
  deriving Repr, BEq, Inhabited

/-- Placement strategy for intermediate edges between tasks. -/
inductive EdgePlacementMode where
  | auto
  | l2
  | dram
  deriving Repr, BEq, DecidableEq, Inhabited

namespace EdgePlacementMode

def name : EdgePlacementMode → String
  | .auto => "auto"
  | .l2 => "l2"
  | .dram => "dram"

def parse? (s : String) : Option EdgePlacementMode :=
  match s.toLower with
  | "auto" => some .auto
  | "l2" => some .l2
  | "dram" => some .dram
  | _ => none

end EdgePlacementMode

/-- Supported node operations in a multi-task ANE DAG. -/
inductive NodeOp where
  | conv1x1 (cin cout : Nat) (weights : ByteArray) (bias : Option ByteArray := none) (act : ConvActivation := .none)
  | conv3x3 (mode : Conv3x3Mode) (cin cout stride : Nat) (padding : PadMode) (weights : ByteArray) (bias : Option ByteArray := none) (act : ConvActivation := .none)
  | pe1 (op : PeOp)
  | pe2 (op : PeBinOp) (rev : Bool := false)
  | lut (act : NeActSpec)
  deriving BEq

instance : Inhabited NodeOp := ⟨.pe2 .add false⟩

/-- One node in topological order inside a multi-task DAG. -/
structure GraphNode where
  id : String
  inputs : Array String
  op : NodeOp
  outPlacement : Option EdgePlacementMode := none
  deriving BEq, Inhabited

/-- Complete multi-task DAG specification for compilation. -/
structure GraphSpec where
  target : Target
  inputs : Array GraphInput
  nodes : Array GraphNode
  defaultPlacement : EdgePlacementMode := .auto
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
  nodes : Array CompiledGraphNode
  hwx : ByteArray
  info : HwxInfo
  sections : LoadSections

/-- Convert an `Array UInt16` of fp16 bit patterns into little-endian bytes. -/
def u16ArrayToBytes (arr : Array UInt16) : ByteArray := Id.run do
  let mut bs := ByteArray.emptyWithCapacity (2 * arr.size)
  for w in arr do
    bs := pushU16LE bs w
  return bs

-- ============================================================================
-- Geometry & L2 Staging Placement Helpers
-- ============================================================================

private def l2ResidentBytes (d : TensorDims) : Nat :=
  let chs := 2 * alignUp d.w 8
  let row := d.c * chs
  alignUp (d.h * row) dmaLineBytes

private def dramScratchBytes (d : TensorDims) : Nat :=
  let row := alignUp (2 * d.w) dmaLineBytes
  let plane := d.h * row
  alignUp (d.c * plane) dmaLineBytes

private def dramSpanBytes (d : TensorDims) : Nat :=
  let row := alignUp (2 * d.w) dmaLineBytes
  let plane := d.h * row
  d.c * plane

private def placeTempBuffers (occupied : List Range) : List Nat → Option (List Nat)
  | [] => some []
  | sz :: rest =>
    let alignedSz := alignUp sz dmaLineBytes
    match findFreeOffset occupied alignedSz l2CapacityBytes with
    | none => none
    | some off =>
      let nextOcc := { base := off, bytes := alignedSz } :: occupied
      match placeTempBuffers nextOcc rest with
      | none => none
      | some restOffs => some (off :: restOffs)

private def maxEndOfRanges (rs : List Range) : Nat :=
  rs.foldl (fun acc r => max acc r.hi) 0

-- ============================================================================
-- Budget-Aware Per-Task Tile Choosers (respecting live L2-resident buffers)
-- ============================================================================

private structure ConvTaskPlan where
  shape : ConvSpec
  tiling : Tiling
  direct : Bool
  srcStageBase : Nat
  resStageBase : Nat
  l2TaskPeak : Nat

private def convStageBasesForTh
    (target : Target) (sh : ConvSpec) (probe : Tiling)
    (srcIsL2 resIsL2 : Bool) (occupied : List Range) (th : Nat) : Option (Nat × Nat × Nat) :=
  let cand : Tiling := { probe with tileHeight := th }
  let direct := if resIsL2 then false else isDirectConv target sh cand
  let srcSz := if srcIsL2 then 0 else srcL2BytesConv sh cand
  let resSz :=
    if resIsL2 then 0
    else if direct then ((1 : Nat) <<< (cand.f + effOcgConv sh cand + cand.fat)) * 512
    else th * resRowStrideConv sh cand
  if srcIsL2 && resIsL2 then
    some (0, 0, maxEndOfRanges occupied)
  else if !srcIsL2 && resIsL2 then
    match placeTempBuffers occupied [srcSz] with
    | some [sBase] =>
      let peak := max (maxEndOfRanges occupied) (sBase + srcSz)
      some (sBase, 0, peak)
    | _ => none
  else if srcIsL2 && !resIsL2 then
    match placeTempBuffers occupied [resSz] with
    | some [rBase] =>
      let peak := max (maxEndOfRanges occupied) (rBase + resSz)
      some (0, rBase, peak)
    | _ => none
  else
    match placeTempBuffers occupied [srcSz, resSz] with
    | some [sBase, rBase] =>
      let peak := max (maxEndOfRanges occupied) (max (sBase + srcSz) (rBase + resSz))
      some (sBase, rBase, peak)
    | _ => none

private def maxConvThForOccupied
    (target : Target) (sh : ConvSpec) (probe : Tiling)
    (srcIsL2 resIsL2 : Bool) (occupied : List Range) : Option (Nat × Nat × Nat × Nat) := Id.run do
  let ho := sh.outH
  if ho == 0 then
    return none
  if let some (sb, rb, pk) := convStageBasesForTh target sh probe srcIsL2 resIsL2 occupied ho then
    return some (ho, sb, rb, pk)
  if ho <= 1 then
    return none
  -- Binary search over 1..(ho - 1)
  let mut lo : Nat := 1
  let mut hi : Nat := ho - 1
  let mut best : Option (Nat × Nat × Nat × Nat) := none
  while lo <= hi do
    let mid := (lo + hi) / 2
    match convStageBasesForTh target sh probe srcIsL2 resIsL2 occupied mid with
    | some (sb, rb, pk) =>
      best := some (mid, sb, rb, pk)
      lo := mid + 1
    | none =>
      if mid == 0 then break
      hi := mid - 1
  return best

private def chooseConvTilingForGraph
    (target : Target) (sh : ConvSpec)
    (extraLutBytes : Nat) (hasBiasInKernel : Bool)
    (srcIsL2 resIsL2 : Bool) (occupied : List Range) : Except String ConvTaskPlan := do
  validateConvShape sh
  let eff := sh.effMode
  let wo := sh.outW
  let maxF := min 4 (ceilLog2 sh.cout)
  let fCandidates : Array Nat :=
    if eff == .depthwise && sh.cout >= 16 then
      #[4]
    else
      Id.run do
        let mut fs : Array Nat := #[]
        for step in [:maxF + 1] do
          fs := fs.push (maxF - step)
        return fs

  let logNEs : Array Nat := #[4, 3, 2, 1, 0]
  let fats : Array Nat := #[0, 1]

  for logNE in logNEs do
    for fat in fats do
      for f in fCandidates do
        if f + fat <= logNE && (logNE - f - fat) <= 3 then
          if let some ocg := chooseConvOcg? target sh f then
            for shPref in [:5] do
              let stripW := (128 <<< fat) >>> shPref
              if stripW >= 8 && stripW <= 256 then
                let cMin := coverMin wo stripW fat
                let shMaxStart := max shPref cMin
                if shMaxStart <= 4 then
                  for shMax in [shMaxStart : 5] do
                    let probe : Tiling := {
                      logNE := logNE
                      f := f
                      fat := fat
                      ocg := ocg
                      shPref := shPref
                      shMax := shMax
                      tileHeight := 1
                    }
                    let chunkOk := (checkConvDenseChunks64K sh probe extraLutBytes hasBiasInKernel).isOk
                    let ocgOk := (validateConvOcgSafety target sh probe).isOk
                    let covOk := (validateConvCoverageSafety target sh probe).isOk
                    if chunkOk && ocgOk && covOk then
                      if let some (maxTh, sb, rb, pk) := maxConvThForOccupied target sh probe srcIsL2 resIsL2 occupied then
                        let cand : Tiling := { probe with tileHeight := maxTh }
                        if (validateConvChooserPolicy sh cand).isOk then
                          let direct := if resIsL2 then false else isDirectConv target sh cand
                          return {
                            shape := sh
                            tiling := cand
                            direct := direct
                            srcStageBase := sb
                            resStageBase := rb
                            l2TaskPeak := pk
                          }
  throw s!"no legal {target.name} conv tiling fits available L2 budget for ({sh.h}x{sh.w}, {sh.cin}->{sh.cout})"

private structure PeTaskPlan where
  outDims : TensorDims
  hwSrc1 : TensorDims
  hwSrc2? : Option TensorDims
  tiling : PeTiling
  direct : Bool
  src1StageBase : Nat
  src2StageBase : Nat
  resStageBase : Nat
  l2TaskPeak : Nat

private def peStageBasesForTh
    (target : Target) (outDims hwSrc1 : TensorDims) (hwSrc2? : Option TensorDims)
    (src1IsL2 src2IsL2 resIsL2 : Bool) (occupied : List Range) (k th : Nat) :
    Option (Nat × Nat × Nat × Nat) := Id.run do
  let twoSource := hwSrc2?.isSome
  let stripW := peStripWidth k
  let effStripW := max stripW 32
  let direct := if resIsL2 then false else peIsDirect target outDims.w k
  let whole1 := twoSource && hwSrc1.c == 1 && hwSrc1.h == 1 && hwSrc1.w > 1 && outDims.h > 1 && th < outDims.h
  let row1 := 2 * (if whole1 then alignUp hwSrc1.w 8 else min (alignUp hwSrc1.w 8) effStripW)
  let size1 := if src1IsL2 then 0 else (if hwSrc1.h > 1 then th else 1) * (hwSrc1.c * row1)
  let size2 :=
    match hwSrc2? with
    | none => 0
    | some hwSrc2 =>
      if src2IsL2 then 0
      else
        let whole2 := hwSrc2.c == 1 && hwSrc2.h == 1 && hwSrc2.w > 1 && outDims.h > 1 && th < outDims.h
        let row2 := 2 * (if whole2 then alignUp hwSrc2.w 8 else min (alignUp hwSrc2.w 8) effStripW)
        (if hwSrc2.h > 1 then th else 1) * (hwSrc2.c * row2)
  let drow := 2 * min (alignUp outDims.w 8) effStripW
  let du := outDims.c * drow
  let sizeRes :=
    if resIsL2 then 0
    else if direct then peDirectResBytes outDims.c drow th
    else th * du

  -- Place non-zero temporary staging buffers sequentially
  let mut occ := occupied
  let mut b1 := 0
  let mut b2 := 0
  let mut bRes := 0
  let mut peak := maxEndOfRanges occupied
  if !src1IsL2 then
    match findFreeOffset occ (alignUp size1 dmaLineBytes) l2CapacityBytes with
    | none => return none
    | some off =>
      b1 := off
      peak := max peak (off + size1)
      occ := { base := off, bytes := alignUp size1 dmaLineBytes } :: occ
  if hwSrc2?.isSome && !src2IsL2 then
    match findFreeOffset occ (alignUp size2 dmaLineBytes) l2CapacityBytes with
    | none => return none
    | some off =>
      b2 := off
      peak := max peak (off + size2)
      occ := { base := off, bytes := alignUp size2 dmaLineBytes } :: occ
  if !resIsL2 then
    match findFreeOffset occ (alignUp sizeRes dmaLineBytes) l2CapacityBytes with
    | none => return none
    | some off =>
      bRes := off
      peak := max peak (off + sizeRes)
  some (b1, b2, bRes, peak)

private def maxPeThForOccupied
    (target : Target) (outDims hwSrc1 : TensorDims) (hwSrc2? : Option TensorDims)
    (src1IsL2 src2IsL2 resIsL2 : Bool) (occupied : List Range) (k : Nat) :
    Option (Nat × Nat × Nat × Nat × Nat) := Id.run do
  if outDims.h == 0 then
    return none
  if let some (b1, b2, br, pk) := peStageBasesForTh target outDims hwSrc1 hwSrc2? src1IsL2 src2IsL2 resIsL2 occupied k outDims.h then
    return some (outDims.h, b1, b2, br, pk)
  if outDims.h <= 1 then
    return none
  let mut lo : Nat := 1
  let mut hi : Nat := outDims.h - 1
  let mut best : Option (Nat × Nat × Nat × Nat × Nat) := none
  while lo <= hi do
    let mid := (lo + hi) / 2
    match peStageBasesForTh target outDims hwSrc1 hwSrc2? src1IsL2 src2IsL2 resIsL2 occupied k mid with
    | some (b1, b2, br, pk) =>
      best := some (mid, b1, b2, br, pk)
      lo := mid + 1
    | none =>
      if mid == 0 then break
      hi := mid - 1
  return best

private def choosePeTilingForGraph
    (target : Target) (outDims hwSrc1 : TensorDims) (hwSrc2? : Option TensorDims)
    (src1IsL2 src2IsL2 resIsL2 : Bool) (occupied : List Range) : Except String PeTaskPlan := do
  let twoSource := hwSrc2?.isSome
  let maxK := peMaxK twoSource
  let outWAligned := alignUp outDims.w 8
  let mut best? : Option (PeTaskPlan × Nat × Nat × Nat) := none

  -- Search k from maxK down to 5 (k = 4 never saves L2 under the max(T, 32) DMA line floor and forces [DL] staging)
  for step in [:maxK - 4] do
    let k := maxK - step
    if let some (th, b1, b2, br, pk) := maxPeThForOccupied target outDims hwSrc1 hwSrc2? src1IsL2 src2IsL2 resIsL2 occupied k then
      let stripW := peStripWidth k
      let direct := if resIsL2 then false else peIsDirect target outDims.w k
      let strips := ceilDiv outWAligned stripW * ceilDiv outDims.h th
      let area := stripW * th
      let plan : PeTaskPlan := {
        outDims := outDims
        hwSrc1 := hwSrc1
        hwSrc2? := hwSrc2?
        tiling := { k := k, tileHeight := th }
        direct := direct
        src1StageBase := b1
        src2StageBase := b2
        resStageBase := br
        l2TaskPeak := pk
      }
      let isBetter := match best? with
        | none => true
        | some (bPlan, bStrips, bArea, bStripW) =>
          if plan.direct != bPlan.direct then plan.direct
          else if strips != bStrips then strips < bStrips
          else if area != bArea then area > bArea
          else stripW < bStripW
      if isBetter then
        best? := some (plan, strips, area, stripW)

  match best? with
  | some (winner, _, _, _) => pure winner
  | none => throw s!"no legal {target.name} PE tiling fits available L2 budget for out={repr outDims}"

private structure NeActTaskPlan where
  tiling : NeActTiling
  direct : Bool
  srcStageBase : Nat
  resStageBase : Nat
  l2TaskPeak : Nat

private def neActStageBasesForTh
    (target : Target) (c _h w : Nat) (probe : NeActTiling)
    (srcIsL2 resIsL2 : Bool) (occupied : List Range) (th : Nat) : Option (Nat × Nat × Nat) :=
  let stripW := probe.stripWidth
  let direct := if resIsL2 then false else neActIsDirect target w probe
  let row := 2 * min (alignUp w 8) stripW
  let u := c * row
  let srcSz := if srcIsL2 then 0 else th * u
  let drow := 2 * min (alignUp w 8) (max stripW 32)
  let du := c * drow
  let resSz :=
    if resIsL2 then 0
    else if direct then neActDirectResBytes probe.f probe.fat
    else th * du
  if srcIsL2 && resIsL2 then
    some (0, 0, maxEndOfRanges occupied)
  else if !srcIsL2 && resIsL2 then
    match placeTempBuffers occupied [srcSz] with
    | some [sb] => some (sb, 0, max (maxEndOfRanges occupied) (sb + srcSz))
    | _ => none
  else if srcIsL2 && !resIsL2 then
    match placeTempBuffers occupied [resSz] with
    | some [rb] => some (0, rb, max (maxEndOfRanges occupied) (rb + resSz))
    | _ => none
  else
    match placeTempBuffers occupied [srcSz, resSz] with
    | some [sb, rb] => some (sb, rb, max (maxEndOfRanges occupied) (max (sb + srcSz) (rb + resSz)))
    | _ => none

private def chooseNeActTilingForGraph
    (target : Target) (c h w : Nat)
    (srcIsL2 resIsL2 : Bool) (occupied : List Range) : Except String NeActTaskPlan := do
  for shPref in [:5] do
    let probe : NeActTiling := {
      logNE := 0, f := 0, fat := 0, shPref := shPref, shMax := shPref, tileHeight := 1
    }
    if let some (sb, rb, pk) := neActStageBasesForTh target c h w probe srcIsL2 resIsL2 occupied h then
      let direct := if resIsL2 then false else neActIsDirect target w probe
      return {
        tiling := { probe with tileHeight := h }
        direct := direct, srcStageBase := sb, resStageBase := rb, l2TaskPeak := pk
      }
    if h > 1 then
      let mut lo : Nat := 1
      let mut hi : Nat := h - 1
      let mut best : Option (Nat × Nat × Nat × Nat) := none
      while lo <= hi do
        let mid := (lo + hi) / 2
        match neActStageBasesForTh target c h w probe srcIsL2 resIsL2 occupied mid with
        | some (sb, rb, pk) =>
          best := some (mid, sb, rb, pk)
          lo := mid + 1
        | none =>
          if mid == 0 then break
          hi := mid - 1
      if let some (th, sb, rb, pk) := best then
        let direct := if resIsL2 then false else neActIsDirect target w probe
        return {
          tiling := { probe with tileHeight := th }
          direct := direct, srcStageBase := sb, resStageBase := rb, l2TaskPeak := pk
        }
  throw s!"no legal {target.name} NE activation tiling fits available L2 budget for ({c}x{h}x{w})"

-- ============================================================================
-- Internal Validated Graph & Lowering Structures
-- ============================================================================

private inductive TensorRef where
  | input (idx : Nat)
  | node (idx : Nat)
  deriving Repr, BEq, DecidableEq, Inhabited

private inductive PlannedOp where
  | conv (shape : ConvSpec) (weights : ByteArray) (bias : Option ByteArray) (act : ConvActivation) (lutPayload? : Option LutPayload) (biasInKernel : Bool)
  | pe1 (op : PeOp) (dims : TensorDims)
  | pe2 (op : PeBinOp) (rev : Bool) (outDims hwSrc1 hwSrc2 : TensorDims) (hwSrc1Ref hwSrc2Ref : TensorRef)
  | lut (act : NeActSpec) (dims : TensorDims) (lutPayload? : Option LutPayload)

private instance : Inhabited PlannedOp := ⟨.pe2 .add false default default default default default⟩

private structure ValidatedNode where
  id : String
  nodeIdx : Nat
  inputs : Array TensorRef
  op : PlannedOp
  outDims : TensorDims
  outPlacement : Option EdgePlacementMode
  engine : Engine
  isNeCompute : Bool
  usesKernel : Bool
  opKind : String
  deriving Inhabited

private inductive TaskPlanKind where
  | conv (p : ConvTaskPlan)
  | pe (p : PeTaskPlan)
  | lut (p : NeActTaskPlan)

private instance : Inhabited TaskPlanKind :=
  ⟨.pe {
    outDims := default
    hwSrc1 := default
    hwSrc2? := none
    tiling := { k := 1, tileHeight := 1 }
    direct := false
    src1StageBase := 0
    src2StageBase := 0
    resStageBase := 0
    l2TaskPeak := 0
  }⟩

private def TaskPlanKind.direct : TaskPlanKind → Bool
  | .conv p => p.direct
  | .pe p => p.direct
  | .lut p => p.direct

private def TaskPlanKind.l2Peak : TaskPlanKind → Nat
  | .conv p => p.l2TaskPeak
  | .pe p => p.l2TaskPeak
  | .lut p => p.l2TaskPeak

private def deriveInputShortNames (inputs : Array GraphInput) : Except String (Array String) := do
  let mut names : Array String := #[]
  for i in [:inputs.size] do
    let inp := inputs[i]!
    let sn :=
      if !inp.shortName.isEmpty then inp.shortName
      else if inputs.size == 1 then "x"
      else s!"in{i}"
    if sn == "y" then
      throw s!"input '{inp.name}': shortName 'y' is reserved for the output surface"
    if sn.utf8ByteSize < 1 || sn.utf8ByteSize > 7 then
      throw s!"input '{inp.name}': shortName '{sn}' length must be 1..7"
    if names.contains sn then
      throw s!"duplicate input shortName '{sn}'"
    names := names.push sn
  pure names

private def nodeUsesKernel : NodeOp → Bool
  | .conv1x1 .. => true
  | .conv3x3 .. => true
  | .lut (.lut _) => true
  | _ => false

/-- Validate graph structure, DAG topological ordering, tensor shapes, and op contracts. -/
private def validateGraph (spec : GraphSpec) :
    Except String (Array String × Array ValidatedNode × Array Nat) := do
  if spec.inputs.isEmpty || spec.inputs.size > 3 then
    throw s!"graph must have 1..3 network inputs, got {spec.inputs.size}"
  if spec.nodes.isEmpty || spec.nodes.size > 256 then
    throw s!"graph must have 1..256 nodes, got {spec.nodes.size}"

  let shortNames ← deriveInputShortNames spec.inputs

  let mut nameMap : Array (String × TensorRef × TensorDims) := #[]
  for i in [:spec.inputs.size] do
    let inp := spec.inputs[i]!
    if inp.name.isEmpty then
      throw s!"graph input {i} has empty name"
    if nameMap.any (fun (nm, _, _) => nm == inp.name) then
      throw s!"duplicate graph input name '{inp.name}'"
    validateTensorDims inp.shape s!"input '{inp.name}'"
    nameMap := nameMap.push (inp.name, .input i, inp.shape)

  let lookup := fun (m : Array (String × TensorRef × TensorDims)) (nm : String) =>
    m.find? (fun (k, _, _) => k == nm)

  let mut vNodes : Array ValidatedNode := Array.mkEmpty spec.nodes.size
  let mut lastConsumer : Array Nat := Array.replicate spec.nodes.size 0

  for i in [:spec.nodes.size] do
    let nd := spec.nodes[i]!
    if nd.id.isEmpty then
      throw s!"graph node {i} has empty id"
    if nameMap.any (fun (nm, _, _) => nm == nd.id) then
      throw s!"duplicate graph tensor/node id '{nd.id}'"
    if nodeUsesKernel nd.op && i >= 255 then
      throw s!"node '{nd.id}' (taskId={i}) uses __KERN_0, which requires taskId < 255 (stage = taskId + 1 <= 255)"

    let mut inRefs : Array TensorRef := #[]
    let mut inDims : Array TensorDims := #[]
    for inName in nd.inputs do
      match lookup nameMap inName with
      | none =>
        throw s!"node '{nd.id}': input '{inName}' is undefined or not in topological order"
      | some (_, ref, dims) =>
        inRefs := inRefs.push ref
        inDims := inDims.push dims
        match ref with
        | .node prodIdx =>
          lastConsumer := lastConsumer.set! prodIdx (max lastConsumer[prodIdx]! i)
        | .input _ => pure ()

    let (plannedOp, outDims, eng, isNeComp, usesKern, opKind) ← match nd.op with
      | .conv1x1 cin cout weights bias act =>
        if inRefs.size != 1 then
          throw s!"node '{nd.id}' (conv1x1): expected 1 input, got {inRefs.size}"
        let d := inDims[0]!
        if d.c != cin then
          throw s!"node '{nd.id}' (conv1x1): input C={d.c} != cin={cin}"
        let sh : ConvSpec := { kernelSize := 1, mode := .dense, h := d.h, w := d.w, cin := cin, cout := cout }
        validateConvShape sh
        let lutPayload? ← match act with
          | .lut lspec => some <$> buildLutTable lspec
          | _ => pure none
        let shiftBits? := lutPayload?.bind (fun lp => lp.inputShiftBits)
        let scalarBiasMode := sh.effMode == .dense && sh.cout == 1
        let hasEffBias := bias.isSome || shiftBits?.isSome
        let biasInKernel := hasEffBias && !scalarBiasMode
        let od : TensorDims := { c := cout, h := sh.outH, w := sh.outW }
        pure (PlannedOp.conv sh weights bias act lutPayload? biasInKernel, od, Engine.ne, true, true, "conv1x1")
      | .conv3x3 mode cin cout stride padding weights bias act =>
        if inRefs.size != 1 then
          throw s!"node '{nd.id}' (conv3x3): expected 1 input, got {inRefs.size}"
        let d := inDims[0]!
        if d.c != cin then
          throw s!"node '{nd.id}' (conv3x3): input C={d.c} != cin={cin}"
        let sh : ConvSpec := {
          kernelSize := 3, mode := mode, h := d.h, w := d.w, cin := cin, cout := cout,
          stride := stride, pad := padding
        }
        validateConvShape sh
        let lutPayload? ← match act with
          | .lut lspec => some <$> buildLutTable lspec
          | _ => pure none
        let shiftBits? := lutPayload?.bind (fun lp => lp.inputShiftBits)
        let scalarBiasMode := sh.effMode == .dense && sh.cout == 1
        let hasEffBias := bias.isSome || shiftBits?.isSome
        let biasInKernel := hasEffBias && !scalarBiasMode
        let od : TensorDims := { c := cout, h := sh.outH, w := sh.outW }
        pure (PlannedOp.conv sh weights bias act lutPayload? biasInKernel, od, Engine.ne, true, true, "conv3x3")
      | .pe1 op =>
        if inRefs.size != 1 then
          throw s!"node '{nd.id}' (pe1): expected 1 input, got {inRefs.size}"
        if op.isTwoSource then
          throw s!"node '{nd.id}' (pe1): expected 1-source PE op, got binary '{op.opName}'"
        if let .chain ins b sc := op then
          if ins.isNone && b.isNone && sc.isNone then
            throw s!"node '{nd.id}' (pe1): PE chain requires at least one of inScale, bias, or scale"
        let d := inDims[0]!
        validateTensorDims d s!"node '{nd.id}'"
        pure (PlannedOp.pe1 op d, d, Engine.pe, false, false, s!"pe_{op.opName}")
      | .pe2 binOp rev =>
        if inRefs.size != 2 then
          throw s!"node '{nd.id}' (pe2): expected 2 inputs, got {inRefs.size}"
        let d1 := inDims[0]!
        let d2 := inDims[1]!
        let od ← broadcastDims d1 d2
        let (hw1, hw2, r1, r2) := if rev then (d2, d1, inRefs[1]!, inRefs[0]!) else (d1, d2, inRefs[0]!, inRefs[1]!)
        let lbl := if rev && binOp == .sub then "pe_rsub" else s!"pe_{binOp.name}"
        pure (PlannedOp.pe2 binOp rev od hw1 hw2 r1 r2, od, Engine.pe, false, false, lbl)
      | .lut act =>
        if inRefs.size != 1 then
          throw s!"node '{nd.id}' (lut): expected 1 input, got {inRefs.size}"
        let d := inDims[0]!
        validateTensorDims d s!"node '{nd.id}'"
        let lutPayload? ← match act with
          | .lut lspec => some <$> buildLutTable lspec
          | _ => pure none
        let isLut := lutPayload?.isSome
        pure (PlannedOp.lut act d lutPayload?, d, Engine.ne, isLut, isLut, s!"lut_{act.name}")

    vNodes := vNodes.push {
      id := nd.id
      nodeIdx := i
      inputs := inRefs
      op := plannedOp
      outDims := outDims
      outPlacement := nd.outPlacement
      engine := eng
      isNeCompute := isNeComp
      usesKernel := usesKern
      opKind := opKind
    }
    nameMap := nameMap.push (nd.id, .node i, outDims)

  -- Ensure every non-final intermediate node is consumed at least once
  for i in [:spec.nodes.size - 1] do
    if lastConsumer[i]! <= i then
      throw s!"intermediate node '{vNodes[i]!.id}' (task {i}) is never consumed by any later node"

  pure (shortNames, vNodes, lastConsumer)

-- ============================================================================
-- L2 Eligibility, Placement Planner & Auto-Spill Loop
-- ============================================================================

private def isNodeL2Eligible (vNodes : Array ValidatedNode) (i : Nat) : Bool := Id.run do
  let nd := vNodes[i]!
  if l2ResidentBytes nd.outDims >= l2CapacityBytes then
    return false
  for j in [i + 1 : vNodes.size] do
    let cons := vNodes[j]!
    match cons.op with
    | .pe2 _ _ outDims hwSrc1 hwSrc2 r1 r2 =>
      if r1 == .node i && hwSrc1 != outDims then
        return false
      if r2 == .node i && hwSrc2 != outDims then
        return false
    | .conv sh _ _ _ _ _ =>
      if cons.inputs[0]! == .node i && sh.kernelSize == 3 && sh.stride == 2 then
        if sh.winReg != sh.w || sh.hinReg != sh.h then
          return false
    | _ => pure ()
  return true

private def lookupPlaced (ps : List PlacedBuf) (id : Nat) : Option PlacedBuf :=
  ps.find? (fun p => p.id == id)

private def tryPlanAllTasks
    (target : Target) (vNodes : Array ValidatedNode)
    (placedL2 : List PlacedBuf) : Except (Nat × String) (Array TaskPlanKind × Nat) := do
  let mut plans : Array TaskPlanKind := Array.mkEmpty vNodes.size
  let mut peakL2 : Nat := 0
  let isNodeL2 := fun (ref : TensorRef) =>
    match ref with
    | .input _ => false
    | .node idx => (lookupPlaced placedL2 idx).isSome

  for t in [:vNodes.size] do
    let nd := vNodes[t]!
    let liveRanges := liveAt placedL2 t
    let resIsL2 := (lookupPlaced placedL2 t).isSome
    let planRes : Except String TaskPlanKind :=
      match nd.op with
      | .conv sh _ _ _ lutPayload? biasInKernel =>
        let srcIsL2 := isNodeL2 nd.inputs[0]!
        let extraLut := if lutPayload?.isSome then lutTableBytes else 0
        TaskPlanKind.conv <$> chooseConvTilingForGraph target sh extraLut biasInKernel srcIsL2 resIsL2 liveRanges
      | .pe1 _ dims =>
        let srcIsL2 := isNodeL2 nd.inputs[0]!
        TaskPlanKind.pe <$> choosePeTilingForGraph target dims dims none srcIsL2 false resIsL2 liveRanges
      | .pe2 _ _ outDims hw1 hw2 r1 r2 =>
        let s1L2 := isNodeL2 r1
        let s2L2 := isNodeL2 r2
        TaskPlanKind.pe <$> choosePeTilingForGraph target outDims hw1 (some hw2) s1L2 s2L2 resIsL2 liveRanges
      | .lut _ dims _ =>
        let srcIsL2 := isNodeL2 nd.inputs[0]!
        TaskPlanKind.lut <$> chooseNeActTilingForGraph target dims.c dims.h dims.w srcIsL2 resIsL2 liveRanges
    match planRes with
    | .ok p =>
      peakL2 := max peakL2 p.l2Peak
      plans := plans.push p
    | .error msg =>
      throw (t, msg)
  pure (plans, peakL2)

-- ============================================================================
-- Multi-Task Register & Sync Bit Encoders
-- ============================================================================

private def encodeSrcPortCfgMulti
    (name : String) (isFirst isL2 waitNe5 waitPe22 handoffNe handoffPe : Bool) : Except String UInt32 :=
  encodeWord (portCfgMultiLayout name) #[
    ("mode", if isL2 then portModeL2 else portModeDma),
    ("handoffNe", if !isFirst && isL2 && handoffNe then 1 else 0),
    ("handoffPe", if !isFirst && isL2 && handoffPe then 1 else 0),
    ("waitNe4", if isFirst then 1 else 0),
    ("waitNe5", if isFirst || waitNe5 then 1 else 0),
    ("dmaFlag", if isL2 then 0 else 1),
    ("P", 1),
    ("waitPe20", if isFirst then 1 else 0),
    ("waitPe22", if isFirst || waitPe22 then 1 else 0)
  ]

private def encodeResPortCfgMulti (isFirst isL2 direct : Bool) : Except String UInt32 :=
  encodeWord resCfgMultiLayout #[
    ("mode", if isL2 then portModeL2 else portModeDma),
    ("direct", if !isL2 && direct then 1 else 0),
    ("waitNe4", if isFirst then 1 else 0),
    ("waitNe5", if isFirst then 1 else 0),
    ("dmaFlag", 1),
    ("P", 1),
    ("waitPe20", if isFirst then 1 else 0),
    ("waitPe22", if isFirst then 1 else 0)
  ]

private def encodeSrcDepMulti (name : String) (taskId depKind depDist : Nat) : Except String UInt32 := do
  let tU32 ← natToU32Checked taskId s!"{name} task"
  let kU32 ← natToU32Checked depKind s!"{name} depKind"
  let dU32 ← natToU32Checked depDist s!"{name} depDist"
  encodeWord (srcDepMultiLayout name) #[
    ("code", srcDepCodeDma),
    ("task", tU32),
    ("depDist", dU32),
    ("depKind", kU32)
  ]

private def encodeKernelStageMulti (taskId : Nat) (isLast isSparse : Bool) : Except String UInt32 := do
  let stageU32 ← natToU32Checked (taskId + 1) "kernelStage.stage"
  encodeWord kernelStageMultiLayout #[
    ("sparse", if isSparse then 1 else 0),
    ("kernEn", 1),
    ("isLast", if isLast then 1 else 0),
    ("kernCfg", 1),
    ("stage", stageU32)
  ]

private def outputStageWrites (taskId : Nat) : Except String (Array RegWrite) := do
  let taskIdU32 ← natToU32Checked taskId "out2 taskId"
  let wOut2 ← encodeWord out2Layout #[("form", out2FormValue), ("task", taskIdU32)]
  pure #[
    { byteAddr := regOut0, value := out0Value },
    { byteAddr := regOut1, value := out1Value },
    { byteAddr := regOut2, value := wOut2 },
    { byteAddr := regOut3, value := out3Value },
    { byteAddr := regOut4, value := out4Value },
    { byteAddr := regOut5, value := out5Value }
  ]

-- ============================================================================
-- Per-Task Lowering with Exact L2/DRAM Access Footprint Construction
-- ============================================================================

private structure PreLoweredTask where
  node : ValidatedNode
  plan : TaskPlanKind
  src1Loc : TensorLocation
  src2Loc? : Option TensorLocation
  resLoc : TensorLocation
  liveL2Mask : UInt32
  l2Accesses : Array VerifiedL2Access
  dramAccesses : Array VerifiedDramAccess
  hasSrc1Dma : Bool
  hasSrc2Dma : Bool
  packedKernel? : Option PackedKernel
  kernelBaseOff : Nat
  src1Producer? : Option Nat
  deriving Inhabited

private def locToBarResource (loc : TensorLocation) (inputShortNames : Array String) (outSurfIdx : Nat) :
    Option BarResource :=
  match loc with
  | .surface idx _ =>
    if idx < inputShortNames.size then
      some (.surface idx inputShortNames[idx]!)
    else
      some (.surface outSurfIdx "y")
  | .dram _ _ => some .scratch
  | .l2 _ _ => none

private def buildPreLoweredTasks
    (spec : GraphSpec)
    (shortNames : Array String)
    (vNodes : Array ValidatedNode)
    (plans : Array TaskPlanKind)
    (placedL2 : List PlacedBuf)
    (nodeLocs : Array TensorLocation) : Except String (Array PreLoweredTask × ByteArray) := do
  let n := vNodes.size
  let outSurfIdx := spec.inputs.size
  let mut kernBuf := ByteArray.empty
  let mut out : Array PreLoweredTask := Array.mkEmpty n

  let resolveRefLoc := fun (ref : TensorRef) =>
    match ref with
    | .input idx => TensorLocation.surface idx (barSlotFirstSurface + idx.toUInt8)
    | .node idx => nodeLocs[idx]!

  let resolveRefProd := fun (ref : TensorRef) =>
    match ref with
    | .input _ => none
    | .node idx => some idx

  for i in [:n] do
    let nd := vNodes[i]!
    let plan := plans[i]!
    let isLast := i + 1 == n
    let resLoc := nodeLocs[i]!

    -- Compute header[7] live-L2 64 KiB block mask for all L2-resident tensors live at task i
    let mut l2Mask : UInt32 := 0
    if !isLast then
      for r in liveAt placedL2 i do
        l2Mask := l2Mask ||| l2BlockMask r.base r.hi

    let mut l2Accs : Array VerifiedL2Access := #[]
    let mut dramAccs : Array VerifiedDramAccess := #[]
    let mut packedKern? : Option PackedKernel := none
    let mut kernBaseOff : Nat := 0
    let mut src1Loc : TensorLocation := resolveRefLoc nd.inputs[0]!
    let mut src2Loc? : Option TensorLocation := none
    let mut src1Prod? : Option Nat := resolveRefProd nd.inputs[0]!

    match nd.op, plan with
    | .conv sh weights bias _ lutPayload? _, .conv cp =>
      -- Pack weights/bias/LUT into shared __KERN_0
      kernBuf := padToAlign kernBuf dmaLineBytes
      kernBaseOff := kernBuf.size
      let packed ← packConvKernelChunks spec.target sh cp.tiling weights bias lutPayload?
      kernBuf := kernBuf ++ packed.bytes
      packedKern? := some packed
      for cs in packed.slots do
        dramAccs := dramAccs.push {
          resource := .kernel
          lo := kernBaseOff + cs.offset
          hi := kernBaseOff + cs.offset + cs.size
          isWrite := false
          port := .kernel
        }
      match src1Loc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dram off _ =>
        let sz := srcL2BytesConv sh cp.tiling
        l2Accs := l2Accs.push { lo := cp.srcStageBase, hi := cp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        let span := dramSpanBytes { c := sh.cin, h := sh.h, w := sh.w }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + span, isWrite := false, port := .src1Dma }
      | .surface idx _ =>
        let sz := srcL2BytesConv sh cp.tiling
        l2Accs := l2Accs.push { lo := cp.srcStageBase, hi := cp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        let span := dramSpanBytes { c := sh.cin, h := sh.h, w := sh.w }
        dramAccs := dramAccs.push { resource := .surface idx shortNames[idx]!, lo := 0, hi := span, isWrite := false, port := .src1Dma }

      match resLoc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dram off _ =>
        let sz :=
          if cp.direct then ((1 : Nat) <<< (cp.tiling.f + effOcgConv sh cp.tiling + cp.tiling.fat)) * 512
          else cp.tiling.tileHeight * resRowStrideConv sh cp.tiling
        l2Accs := l2Accs.push { lo := cp.resStageBase, hi := cp.resStageBase + sz, isWrite := true, role := .resStage }
        let span := dramSpanBytes nd.outDims
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + span, isWrite := true, port := .resDma }
      | .surface _ _ =>
        let sz :=
          if cp.direct then ((1 : Nat) <<< (cp.tiling.f + effOcgConv sh cp.tiling + cp.tiling.fat)) * 512
          else cp.tiling.tileHeight * resRowStrideConv sh cp.tiling
        l2Accs := l2Accs.push { lo := cp.resStageBase, hi := cp.resStageBase + sz, isWrite := true, role := .resStage }
        let span := dramSpanBytes nd.outDims
        dramAccs := dramAccs.push { resource := .surface outSurfIdx "y", lo := 0, hi := span, isWrite := true, port := .resDma }

    | .pe1 _ dims, .pe pp =>
      let stripW := peStripWidth pp.tiling.k
      let effStripW := max stripW 32
      match src1Loc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | _ =>
        let row1 := 2 * min (alignUp dims.w 8) effStripW
        let sz1 := (if dims.h > 1 then pp.tiling.tileHeight else 1) * (dims.c * row1)
        l2Accs := l2Accs.push { lo := pp.src1StageBase, hi := pp.src1StageBase + sz1, isWrite := true, role := .src1Stage }
        let off := match src1Loc with | .dram o _ => o | _ => 0
        let resId := (locToBarResource src1Loc shortNames outSurfIdx).get!
        dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + dramSpanBytes dims, isWrite := false, port := .src1Dma }

      match resLoc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | _ =>
        let drow := 2 * min (alignUp dims.w 8) effStripW
        let szRes := if pp.direct then peDirectResBytes dims.c drow pp.tiling.tileHeight else pp.tiling.tileHeight * (dims.c * drow)
        l2Accs := l2Accs.push { lo := pp.resStageBase, hi := pp.resStageBase + szRes, isWrite := true, role := .resStage }
        let off := match resLoc with | .dram o _ => o | _ => 0
        let resId := (locToBarResource resLoc shortNames outSurfIdx).get!
        dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + dramSpanBytes dims, isWrite := true, port := .resDma }

    | .pe2 _ _ outDims hw1 hw2 r1 r2, .pe pp =>
      src1Loc := resolveRefLoc r1
      src2Loc? := some (resolveRefLoc r2)
      src1Prod? := resolveRefProd r1
      let stripW := peStripWidth pp.tiling.k
      let effStripW := max stripW 32
      let th := pp.tiling.tileHeight
      match src1Loc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | _ =>
        let whole1 := hw1.c == 1 && hw1.h == 1 && hw1.w > 1 && outDims.h > 1 && th < outDims.h
        let row1 := 2 * (if whole1 then alignUp hw1.w 8 else min (alignUp hw1.w 8) effStripW)
        let sz1 := (if hw1.h > 1 then th else 1) * (hw1.c * row1)
        l2Accs := l2Accs.push { lo := pp.src1StageBase, hi := pp.src1StageBase + sz1, isWrite := true, role := .src1Stage }
        let off := match src1Loc with | .dram o _ => o | _ => 0
        let resId := (locToBarResource src1Loc shortNames outSurfIdx).get!
        dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + dramSpanBytes hw1, isWrite := false, port := .src1Dma }

      let s2Loc := src2Loc?.get!
      match s2Loc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src2L2 }
      | _ =>
        let whole2 := hw2.c == 1 && hw2.h == 1 && hw2.w > 1 && outDims.h > 1 && th < outDims.h
        let row2 := 2 * (if whole2 then alignUp hw2.w 8 else min (alignUp hw2.w 8) effStripW)
        let sz2 := (if hw2.h > 1 then th else 1) * (hw2.c * row2)
        l2Accs := l2Accs.push { lo := pp.src2StageBase, hi := pp.src2StageBase + sz2, isWrite := true, role := .src2Stage }
        let off := match s2Loc with | .dram o _ => o | _ => 0
        let resId := (locToBarResource s2Loc shortNames outSurfIdx).get!
        dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + dramSpanBytes hw2, isWrite := false, port := .src2Dma }

      match resLoc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | _ =>
        let drow := 2 * min (alignUp outDims.w 8) effStripW
        let szRes := if pp.direct then peDirectResBytes outDims.c drow th else th * (outDims.c * drow)
        l2Accs := l2Accs.push { lo := pp.resStageBase, hi := pp.resStageBase + szRes, isWrite := true, role := .resStage }
        let off := match resLoc with | .dram o _ => o | _ => 0
        let resId := (locToBarResource resLoc shortNames outSurfIdx).get!
        dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + dramSpanBytes outDims, isWrite := true, port := .resDma }

    | .lut _ dims lutPayload?, .lut lp =>
      if let some payload := lutPayload? then
        kernBuf := padToAlign kernBuf dmaLineBytes
        kernBaseOff := kernBuf.size
        kernBuf := kernBuf ++ payload.bytes
        dramAccs := dramAccs.push {
          resource := .kernel, lo := kernBaseOff, hi := kernBaseOff + lutTableBytes, isWrite := false, port := .lut
        }
      let stripW := lp.tiling.stripWidth
      match src1Loc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | _ =>
        let row := 2 * min (alignUp dims.w 8) stripW
        let sz := lp.tiling.tileHeight * (dims.c * row)
        l2Accs := l2Accs.push { lo := lp.srcStageBase, hi := lp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        let off := match src1Loc with | .dram o _ => o | _ => 0
        let resId := (locToBarResource src1Loc shortNames outSurfIdx).get!
        dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + dramSpanBytes dims, isWrite := false, port := .src1Dma }

      match resLoc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | _ =>
        let drow := 2 * min (alignUp dims.w 8) (max stripW 32)
        let szRes := if lp.direct then neActDirectResBytes lp.tiling.f lp.tiling.fat else lp.tiling.tileHeight * (dims.c * drow)
        l2Accs := l2Accs.push { lo := lp.resStageBase, hi := lp.resStageBase + szRes, isWrite := true, role := .resStage }
        let off := match resLoc with | .dram o _ => o | _ => 0
        let resId := (locToBarResource resLoc shortNames outSurfIdx).get!
        dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + dramSpanBytes dims, isWrite := true, port := .resDma }
    | _, _ =>
      throw s!"internal error: op/plan mismatch on node '{nd.id}'"

    let hasSrc1Dma := match src1Loc with | .l2 _ _ => false | _ => true
    let hasSrc2Dma := match src2Loc? with | some (.l2 _ _) => false | some _ => true | none => false
    out := out.push {
      node := nd
      plan := plan
      src1Loc := src1Loc
      src2Loc? := src2Loc?
      resLoc := resLoc
      liveL2Mask := l2Mask
      l2Accesses := l2Accs
      dramAccesses := dramAccs
      hasSrc1Dma := hasSrc1Dma
      hasSrc2Dma := hasSrc2Dma
      packedKernel? := packedKern?
      kernelBaseOff := kernBaseOff
      src1Producer? := src1Prod?
    }

  pure (out, kernBuf)

-- ============================================================================
-- Synchronization Pass (§7.4) — Fixpoint Hazard Ordering
-- ============================================================================

private def intervalsOverlap (lo1 hi1 lo2 hi2 : Nat) : Bool :=
  lo1 < hi2 && lo2 < hi1

private def latestEngineBefore (pre : Array PreLoweredTask) (eng : Engine) (i : Nat) : Option Nat := Id.run do
  let mut best : Option Nat := none
  for j in [:i] do
    if pre[j]!.node.engine == eng then
      best := some j
  return best

private def computeSynchronization (pre : Array PreLoweredTask) : Array TaskSyncInfo := Id.run do
  let n := pre.size
  let mut syncs : Array TaskSyncInfo := Array.ofFn (n := n) fun i => {
    hasSrc1Dma := pre[i.val]!.hasSrc1Dma
    hasSrc2Dma := pre[i.val]!.hasSrc2Dma
  }

  -- Seed L2 handoff bits on L2-resident src1 reads from immediately latest same/other engine producer
  for i in [1 : n] do
    if let .l2 _ _ := pre[i]!.src1Loc then
      if let some prod := pre[i]!.src1Producer? then
        let prodEng := pre[prod]!.node.engine
        if prodEng == .ne && latestEngineBefore pre .ne i == some prod then
          syncs := syncs.set! i { syncs[i]! with src1HandoffNe := true }
        else if prodEng == .pe && latestEngineBefore pre .pe i == some prod then
          syncs := syncs.set! i { syncs[i]! with src1HandoffPe := true }

  -- Seed BAR 3 DRAM-scratch read coherence (depKind = 2, depDist = min 15 (i - j - 1)) on every PE src1/src2
  -- reading a scratch buffer written by an earlier task j < i (latest writer j).
  -- Never credit depDist > 0 as ordering: the fixpoint loop below adds HB / src2Wait5 / src2Wait22 whenever j < i - 1.
  for i in [1 : n] do
    if pre[i]!.node.engine == .pe then
      for b in pre[i]!.dramAccesses do
        if b.resource == .scratch && !b.isWrite then
          let mut latestW? : Option Nat := none
          for j in [:i] do
            for a in pre[j]!.dramAccesses do
              if a.resource == .scratch && a.isWrite && intervalsOverlap a.lo a.hi b.lo b.hi then
                latestW? := some j
          if let some j := latestW? then
            let dist := min 15 (i - j - 1)
            if b.port == .src1Dma then
              syncs := syncs.set! i { syncs[i]! with src1DepKind := 2, src1DepDist := dist }
            else if b.port == .src2Dma then
              syncs := syncs.set! i { syncs[i]! with src2DepKind := 2, src2DepDist := dist }

  -- Fixpoint loop: converge whenever no new wait/dep bits are added (monotone, <= 8*n steps)
  let maxIters := 8 * n + 2
  for _ in [:maxIters] do
    let fps : Array VerifiedTaskFootprint := Array.ofFn (n := n) fun i => {
      taskIndex := i.val
      kind := match pre[i.val]!.node.op with
        | .conv .. => .neConv
        | .lut .. => .neAct
        | _ => .peOp
      engine := pre[i.val]!.node.engine
      isNeCompute := pre[i.val]!.node.isNeCompute
      l2Accesses := pre[i.val]!.l2Accesses
      dramAccesses := pre[i.val]!.dramAccesses
      sync := syncs[i.val]!
    }
    let hbTbl := reachTable (stepHB fps) n
    let hblTbl := reachTable (stepHBL fps) n
    let hb := fun a b => reachB hbTbl a b
    let hbl := fun a b => reachB hblTbl a b
    let hbEq := fun a b => a == b || hb a b
    let hblEq := fun a b => a == b || hbl a b

    let mut changed := false

    for i in [1 : n] do
      let ti := fps[i]!
      let latestNe := latestEngineBefore pre .ne i
      let latestPe := latestEngineBefore pre .pe i

      for k in [:i] do
        let tk := fps[k]!

        let dramRawKind : Bool :=
          syncs[i]!.hasSrc1Dma &&
          (syncs[i]!.src1DepKind == 1 || syncs[i]!.src1DepKind == 2) &&
          syncs[i]!.src1DepDist == 0 &&
          hbEq k (i - 1)

        let src2Kind : Bool :=
          pre[i - 1]!.node.engine == .pe &&
          syncs[i]!.hasSrc2Dma &&
          (syncs[i]!.src2DepKind == 1 || syncs[i]!.src2DepKind == 2) &&
          syncs[i]!.src2DepDist == 0 &&
          hbEq k (i - 1)

        let src2Wait22 : Bool :=
          (syncs[i]!.src1WaitPe22 || syncs[i]!.src2WaitPe22) &&
          match latestPe with | some j => hbEq k j | none => false

        let src2Wait5 : Bool :=
          (syncs[i]!.src1WaitNe5 || syncs[i]!.src2WaitNe5) &&
          (Id.run do
            let mut ok := false
            for j in [:i] do
              if pre[j]!.node.engine == .ne && hbEq k j then ok := true
            return ok)

        -- 1. DRAM conflicts
        for a in tk.dramAccesses do
          for b in ti.dramAccesses do
            if a.resource == b.resource && (a.isWrite || b.isWrite) && intervalsOverlap a.lo a.hi b.lo b.hi then
              if ti.engine == .ne && b.port == .src1Dma then
                -- srcReadNE: set 0x4d00 depKind=2, depDist=0 and ensure HB(k, i - 1)
                if syncs[i]!.src1DepKind != 2 || syncs[i]!.src1DepDist != 0 then
                  syncs := syncs.set! i { syncs[i]! with src1DepKind := 2, src1DepDist := 0 }
                  changed := true
                if !hbEq k (i - 1) then
                  let prev := i - 1
                  if tk.engine == .ne then
                    if !syncs[prev]!.src1WaitNe5 then
                      syncs := syncs.set! prev { syncs[prev]! with src1WaitNe5 := true, src1HandoffNe := false }
                      changed := true
                  else
                    if !syncs[prev]!.src1WaitPe22 then
                      syncs := syncs.set! prev { syncs[prev]! with src1WaitPe22 := true, src1HandoffPe := false }
                      changed := true
              else if ti.engine == .pe && b.port == .src2Dma then
                if !(src2Kind || src2Wait22 || src2Wait5 || dramRawKind) then
                  if tk.engine == .ne then
                    if !syncs[i]!.src2WaitNe5 then
                      syncs := syncs.set! i { syncs[i]! with src2WaitNe5 := true }
                      changed := true
                  else
                    if !syncs[i]!.src2WaitPe22 then
                      syncs := syncs.set! i { syncs[i]! with src2WaitPe22 := true }
                      changed := true
              else if a.isWrite && !b.isWrite then
                if !(hb k i || dramRawKind) then
                  if tk.engine == .ne then
                    if !syncs[i]!.src1WaitNe5 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitNe5 := true, src1HandoffNe := false }
                      changed := true
                  else
                    if !syncs[i]!.src1WaitPe22 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitPe22 := true, src1HandoffPe := false }
                      changed := true
              else
                -- DRAM WAR / WAW
                if !hb k i then
                  if tk.engine == .ne then
                    if !syncs[i]!.src1WaitNe5 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitNe5 := true, src1HandoffNe := false }
                      changed := true
                  else
                    if !syncs[i]!.src1WaitPe22 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitPe22 := true, src1HandoffPe := false }
                      changed := true

        -- 2. L2 conflicts
        for a in tk.l2Accesses do
          for b in ti.l2Accesses do
            if (a.isWrite || b.isWrite) && intervalsOverlap a.lo a.hi b.lo b.hi then
              let handoff2 : Bool :=
                a.isWrite &&
                ((b.role == .src1L2 && syncs[i]!.src1HandoffNe) ||
                 (b.role == .src2L2 && syncs[i]!.src2HandoffNe)) &&
                match latestNe with | some j => hblEq k j | none => false
              let handoff3 : Bool :=
                a.isWrite && b.role == .src1L2 && syncs[i]!.src1HandoffPe &&
                match latestPe with | some j => hblEq k j | none => false

              if ti.engine == .pe && (b.role == .src2Stage || b.role == .src2L2) then
                if !(src2Wait22 || src2Wait5 || src2Kind || (b.role == .src2L2 && handoff2)) then
                  if tk.engine == .ne then
                    if !syncs[i]!.src2WaitNe5 then
                      syncs := syncs.set! i { syncs[i]! with src2WaitNe5 := true }
                      changed := true
                  else
                    if !syncs[i]!.src2WaitPe22 then
                      syncs := syncs.set! i { syncs[i]! with src2WaitPe22 := true }
                      changed := true
              else
                if !(hbl k i || handoff2 || handoff3) then
                  if tk.engine == .ne then
                    if !syncs[i]!.src1WaitNe5 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitNe5 := true, src1HandoffNe := false }
                      changed := true
                  else
                    if !syncs[i]!.src1WaitPe22 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitPe22 := true, src1HandoffPe := false }
                      changed := true

    if !changed then
      break

  -- Clean up redundant handoff bits when stronger whole-task wait bits are set
  for i in [:n] do
    let s := syncs[i]!
    let s' : TaskSyncInfo := {
      s with
      src1HandoffNe := s.src1HandoffNe && !s.src1WaitNe5
      src1HandoffPe := s.src1HandoffPe && !s.src1WaitPe22
      src2HandoffNe := s.src2HandoffNe && !s.src2WaitNe5
      src2HandoffPe := s.src2HandoffPe && !s.src2WaitPe22
    }
    syncs := syncs.set! i s'

  return syncs

-- ============================================================================
-- Multi-Task Register Lowering
-- ============================================================================

private def emitTaskDesc
    (target : Target) (totalTasks : Nat) (kernelBarSlot : UInt8)
    (pt : PreLoweredTask) (sync : TaskSyncInfo) : Except String TaskDesc := do
  let nd := pt.node
  let taskId := nd.nodeIdx
  let isFirst := taskId == 0
  let isLast := taskId + 1 == totalTasks
  let taskIdU32 ← natToU32Checked taskId "taskId"

  let src1IsL2 := match pt.src1Loc with | .l2 _ _ => true | _ => false
  let resIsL2 := match pt.resLoc with | .l2 _ _ => true | _ => false

  let wSrc1Cfg ← encodeSrcPortCfgMulti "src1Cfg" isFirst src1IsL2
    sync.src1WaitNe5 sync.src1WaitPe22 sync.src1HandoffNe sync.src1HandoffPe

  match nd.op, pt.plan with
  | .conv sh _ _ act _ _, .conv cp =>
    let some packed := pt.packedKernel?
      | throw s!"internal error: missing packed kernel on conv node '{nd.id}'"
    let sparse := sh.isSparse
    let wide3x3 := computeWide3x3Conv target sh cp.tiling
    let ho := sh.outH
    let wo := sh.outW
    let chs := chsBytesConv sh cp.tiling
    let dchs := dchsBytesConv sh cp.tiling
    let srcRow := srcRowStrideConv sh cp.tiling
    let resRow := resRowStrideConv sh cp.tiling

    let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked sh.winReg "Win")]
    let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked sh.hinReg "Hin")]
    let cinU32 ← natToU32Checked sh.cin "Cin"
    let wWout ← encodeWord (dim16Layout "Wout") #[("val", ← natToU32Checked wo "Wout")]
    let wHout ← encodeWord (dim16Layout "Hout") #[("val", ← natToU32Checked ho "Hout")]
    let coutU32 ← natToU32Checked sh.cout "Cout"
    let wConvCfg ← encodeWord convCfgLayout #[
      ("kw", ← natToU32Checked sh.kernelSize "kw"), ("kh", ← natToU32Checked sh.kernelSize "kh"),
      ("sx", ← natToU32Checked sh.stride "sx"), ("sy", ← natToU32Checked sh.stride "sy"),
      ("pl", ← natToU32Checked sh.padLeft "pl"), ("pt", ← natToU32Checked sh.padTop "pt"),
      ("hx", 1), ("vy", 1)
    ]
    let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked cp.tiling.tileHeight "TH")]
    let wNeCommon ← encodeWord neCommonLayout #[
      ("kind", 0),
      ("shPref", ← natToU32Checked cp.tiling.shPref "shPref"),
      ("shMin", ← natToU32Checked cp.tiling.shPref "shMin"),
      ("shMax", ← natToU32Checked cp.tiling.shMax "shMax"),
      ("F", ← natToU32Checked cp.tiling.f "F"),
      ("wide3x3", if wide3x3 then 1 else 0)
    ]
    let wNeTile ← encodeWord neTileLayout #[
      ("ocg", ← natToU32Checked (effOcgConv sh cp.tiling) "ocg"),
      ("fat", ← natToU32Checked cp.tiling.fat "fat"),
      ("WU", ← natToU32Checked cp.tiling.wu "WU")
    ]
    let wResCfg ← encodeResPortCfgMulti isFirst resIsL2 cp.direct
    let hasLut := packed.lutPayload.isSome
    let neScaleVal : UInt32 := match packed.lutPayload with
      | some lp => lp.preScaleBits.toUInt32
      | none => fp16OneBits
    let wNeCfg ← encodeWord (neCfgLayout target) #[("sparse", if sparse then 1 else 0)]
    let wNeOp ← encodeWord neOpLayout #[
      ("biasInKernel", if packed.biasInKernel then 1 else 0),
      ("relu", if act.isRelu && !hasLut then 1 else 0),
      ("lut", if hasLut then 1 else 0)
    ]
    let wNeScale ← encodeWord neScaleLayout #[("val", neScaleVal)]
    let wKernStage ← encodeKernelStageMulti taskId isLast sparse

    let mut plain : Array RegWrite := #[
      { byteAddr := regWin, value := wWin },
      { byteAddr := regHin, value := wHin },
      { byteAddr := regCin, value := cinU32 },
      { byteAddr := regWout, value := wWout },
      { byteAddr := regHout, value := wHout },
      { byteAddr := regCout, value := coutU32 },
      { byteAddr := regConvCfg, value := wConvCfg },
      { byteAddr := regTH, value := wTH },
      { byteAddr := regNeCommon, value := wNeCommon },
      { byteAddr := regNeTile, value := wNeTile },
      { byteAddr := regTargetCfg, value := targetCfgValue target },
      { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
      { byteAddr := regResCfg, value := wResCfg },
      { byteAddr := regNeCfg, value := wNeCfg },
      { byteAddr := regNeOp, value := wNeOp },
      { byteAddr := regNeScale, value := wNeScale },
      { byteAddr := regKernelStage, value := wKernStage }
    ]
    if isFirst then
      plain := plain.push { byteAddr := regSrc2Cfg, value := src2CfgValue }
    if sh.effMode == .depthwise then
      plain := plain.push { byteAddr := regDwGroups, value := ← encodeWord dwGroupsLayout #[("perGroupIn", 1)] }
    if sh.effMode == .dense && sh.cout == 1 then
      plain := plain.push { byteAddr := regScalarBias, value := ← encodeWord scalarBiasLayout #[("val", packed.scalarBiasBits.toUInt32)] }

    let mut barRecs : Array BarRegWrite := #[
      { baseByteAddr := regKernelBaseLo, barSlot := kernelBarSlot, values := #[← natToU32Checked pt.kernelBaseOff "kernelBaseLo", 0] }
    ]

    -- Source 1 L2 & optional DMA
    match pt.src1Loc with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp sh.winReg 8
      let l2Row := sh.cin * l2Chs
      plain := plain ++ #[
        { byteAddr := regSrc1Base, value := ← natToU32Checked base "src1Base" },
        { byteAddr := regSrc1ChanStride, value := ← natToU32Checked l2Chs "src1ChanStride" },
        { byteAddr := regSrc1RowStride, value := ← natToU32Checked l2Row "src1RowStride" }
      ]
    | loc =>
      let inRowStride := alignUp (2 * sh.w) dmaLineBytes
      let inPlaneStride := sh.h * inRowStride
      let (slot, off) := match loc with
        | .surface _ s => (s, 0)
        | .dram o _ => (barSlotScratch, o)
        | .l2 _ _ => (0, 0)
      plain := plain ++ #[
        { byteAddr := regSrc1Base, value := ← natToU32Checked cp.srcStageBase "src1Base" },
        { byteAddr := regSrc1ChanStride, value := ← natToU32Checked chs "src1ChanStride" },
        { byteAddr := regSrc1RowStride, value := ← natToU32Checked srcRow "src1RowStride" },
        { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked (sh.cin * chs) "src1RowBytesA" },
        { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked (sh.cin * chs) "src1RowBytesB" },
        { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
        { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked inRowStride "src1DmaRowStride" },
        { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked inPlaneStride "src1DmaPlaneStride" },
        { byteAddr := regSrc1Fmt, value := srcFmtWordP1 }
      ]
      let wSrc1Dep ← encodeSrcDepMulti "src1Dep" taskId sync.src1DepKind sync.src1DepDist
      barRecs := barRecs ++ #[
        { baseByteAddr := regSrc1BaseLo, barSlot := slot, values := #[← natToU32Checked off "src1BaseLo", 0] },
        { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrc1Dep] }
      ]

    -- Result L2 & optional DMA
    match pt.resLoc with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp wo 8
      let l2Row := sh.cout * l2Chs
      plain := plain ++ #[
        { byteAddr := regResBase, value := ← natToU32Checked base "resBase" },
        { byteAddr := regResChanStride, value := ← natToU32Checked l2Chs "resChanStride" },
        { byteAddr := regResRowStride, value := ← natToU32Checked l2Row "resRowStride" }
      ]
    | loc =>
      let outRowStride := alignUp (2 * wo) dmaLineBytes
      let outPlaneStride := ho * outRowStride
      let (slot, off, depCode) := match loc with
        | .surface _ s => (s, 0, resDepCode target)
        | .dram o _ => (barSlotScratch, o, resDepCode target ||| resDepScratchFlags)
        | .l2 _ _ => (0, 0, resDepCode target)
      plain := plain ++ #[
        { byteAddr := regResBase, value := ← natToU32Checked cp.resStageBase "resBase" },
        { byteAddr := regResDmaRowStride, value := ← natToU32Checked outRowStride "resDmaRowStride" },
        { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked outPlaneStride "resDmaPlaneStride" },
        { byteAddr := regResFmt, value := resFmtWordP1 }
      ]
      if !cp.direct then
        let resRowBytesU32 ← natToU32Checked (sh.cout * dchs) "resRowBytes"
        plain := plain ++ #[
          { byteAddr := regResChanStride, value := ← natToU32Checked dchs "resChanStride" },
          { byteAddr := regResRowStride, value := ← natToU32Checked resRow "resRowStride" },
          { byteAddr := regResRowBytesA, value := resRowBytesU32 },
          { byteAddr := regResRowBytesB, value := resRowBytesU32 }
        ]
      let wResDep ← encodeWord resDepLayout #[
        ("code", depCode),
        ("task", taskIdU32),
        ("direct", if cp.direct then 1 else 0)
      ]
      barRecs := barRecs ++ #[
        { baseByteAddr := regResBaseLo, barSlot := slot, values := #[← natToU32Checked off "resBaseLo", 0] },
        { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
      ]

    -- Kernel chunk slots: active slots enabled; on taskId > 0, ALL unused slots explicitly disabled (§3)!
    let wSlotOn ← encodeWord chunkSlotLayout #[("enable", 1), ("on20", 1), ("task", taskIdU32)]
    let wSlotOff ← encodeWord chunkSlotLayout #[("enable", 0), ("on20", 1), ("task", taskIdU32)]
    for s in [:16] do
      match packed.slots.find? (fun cs => cs.slot == s) with
      | some cs =>
        plain := plain ++ #[
          { byteAddr := regChunkSlot s, value := wSlotOn },
          { byteAddr := regChunkOffset s, value := ← natToU32Checked cs.offset s!"chunkOffset[{s}]" },
          { byteAddr := regChunkSize s, value := ← natToU32Checked cs.size s!"chunkSize[{s}]" }
        ]
      | none =>
        if !isFirst then
          plain := plain.push { byteAddr := regChunkSlot s, value := wSlotOff }

    if isLast then
      plain := plain ++ (← outputStageWrites taskId)

    pure {
      taskId := taskId
      logNE := cp.tiling.logNE
      eneOverride := some (cp.tiling.logNE + 1)
      isFirst := isFirst
      isLast := isLast
      liveL2Mask := pt.liveL2Mask
      plainWrites := plain
      barWrites := barRecs
      kernelRelocs := #[{ barRecIndex := 0, payloadWordIndex := 0 }]
    }

  | .pe1 op dims, .pe pp =>
    let stripW := peStripWidth pp.tiling.k
    let th := pp.tiling.tileHeight
    let m := peTileM false pp.tiling.k th
    let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked dims.w "W")]
    let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked dims.h "H")]
    let cU32 ← natToU32Checked dims.c "C"
    let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked th "TH")]
    let wPeCommon ← encodeWord peCommonLayout #[("mode", peCommonOneSrcValue)]
    let wPeTile ← encodeWord peTileLayout #[("k", ← natToU32Checked pp.tiling.k "k"), ("m", ← natToU32Checked m "m")]
    let wResCfg ← encodeResPortCfgMulti isFirst resIsL2 pp.direct

    let mut plain : Array RegWrite := #[
      { byteAddr := regWin, value := wWin },
      { byteAddr := regHin, value := wHin },
      { byteAddr := regCin, value := cU32 },
      { byteAddr := regWout, value := wWin },
      { byteAddr := regHout, value := wHin },
      { byteAddr := regCout, value := cU32 },
      { byteAddr := regBatchN, value := 1 },
      { byteAddr := regTH, value := wTH },
      { byteAddr := regPeCommon, value := wPeCommon },
      { byteAddr := regPeTile, value := wPeTile },
      { byteAddr := regTargetCfg, value := 0 },
      { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
      { byteAddr := regResCfg, value := wResCfg }
    ]
    if isFirst then
      plain := plain.push { byteAddr := regSrc2Cfg, value := src2CfgValue }

    match op with
    | .abs => plain := plain.push { byteAddr := regPeOp, value := peOpAbsValue }
    | .square => plain := plain.push { byteAddr := regPeOp, value := peOpSquareValue }
    | .chain inScale bias scale =>
      if let some b := bias then plain := plain.push { byteAddr := regPeBias, value := f16ToF32Bits b }
      if let some s := scale then plain := plain.push { byteAddr := regPeScale, value := f16ToF32Bits s }
      if let some ins := inScale then plain := plain.push { byteAddr := regPeInScale, value := f16ToF32Bits ins }
    | .binary .. => throw "internal error: binary op in pe1 lowering"

    let mut barRecs : Array BarRegWrite := #[]
    match pt.src1Loc with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp dims.w 8
      let l2Row := dims.c * l2Chs
      plain := plain ++ #[
        { byteAddr := regSrc1Base, value := ← natToU32Checked base "src1Base" },
        { byteAddr := regSrc1ChanStride, value := ← natToU32Checked l2Chs "src1ChanStride" },
        { byteAddr := regSrc1RowStride, value := ← natToU32Checked l2Row "src1RowStride" }
      ]
    | loc =>
      let row1 := 2 * min (alignUp dims.w 8) (max stripW 32)
      let u1 := dims.c * row1
      let dmaRow := alignUp (2 * dims.w) dmaLineBytes
      let dmaPlane := dims.h * dmaRow
      let (slot, off) := match loc with | .surface _ s => (s, 0) | .dram o _ => (barSlotScratch, o) | .l2 _ _ => (0, 0)
      plain := plain ++ #[
        { byteAddr := regSrc1Base, value := ← natToU32Checked pp.src1StageBase "src1Base" },
        { byteAddr := regSrc1ChanStride, value := ← natToU32Checked row1 "src1ChanStride" },
        { byteAddr := regSrc1RowStride, value := ← natToU32Checked u1 "src1RowStride" },
        { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked u1 "src1RowBytesA" },
        { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked u1 "src1RowBytesB" },
        { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
        { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked dmaRow "src1DmaRowStride" },
        { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked dmaPlane "src1DmaPlaneStride" },
        { byteAddr := regSrc1Fmt, value := srcFmtWordP1 }
      ]
      let wSrc1Dep ← encodeSrcDepMulti "src1Dep" taskId sync.src1DepKind sync.src1DepDist
      barRecs := barRecs ++ #[
        { baseByteAddr := regSrc1BaseLo, barSlot := slot, values := #[← natToU32Checked off "src1BaseLo", 0] },
        { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrc1Dep] }
      ]

    match pt.resLoc with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp dims.w 8
      let l2Row := dims.c * l2Chs
      plain := plain ++ #[
        { byteAddr := regResBase, value := ← natToU32Checked base "resBase" },
        { byteAddr := regResChanStride, value := ← natToU32Checked l2Chs "resChanStride" },
        { byteAddr := regResRowStride, value := ← natToU32Checked l2Row "resRowStride" }
      ]
    | loc =>
      let dmaRow := alignUp (2 * dims.w) dmaLineBytes
      let dmaPlane := dims.h * dmaRow
      let (slot, off, depCode) := match loc with
        | .surface _ s => (s, 0, resDepCode target)
        | .dram o _ => (barSlotScratch, o, resDepCode target ||| resDepScratchFlags)
        | .l2 _ _ => (0, 0, resDepCode target)
      plain := plain ++ #[
        { byteAddr := regResBase, value := ← natToU32Checked pp.resStageBase "resBase" },
        { byteAddr := regResDmaRowStride, value := ← natToU32Checked dmaRow "resDmaRowStride" },
        { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked dmaPlane "resDmaPlaneStride" },
        { byteAddr := regResFmt, value := resFmtWordP1 }
      ]
      if !pp.direct then
        let drow := 2 * min (alignUp dims.w 8) (max stripW 32)
        let du := dims.c * drow
        plain := plain ++ #[
          { byteAddr := regResChanStride, value := ← natToU32Checked drow "resChanStride" },
          { byteAddr := regResRowStride, value := ← natToU32Checked du "resRowStride" },
          { byteAddr := regResRowBytesA, value := ← natToU32Checked du "resRowBytesA" },
          { byteAddr := regResRowBytesB, value := ← natToU32Checked du "resRowBytesB" }
        ]
      let wResDep ← encodeWord resDepLayout #[
        ("code", depCode), ("task", taskIdU32), ("direct", if pp.direct then 1 else 0)
      ]
      barRecs := barRecs ++ #[
        { baseByteAddr := regResBaseLo, barSlot := slot, values := #[← natToU32Checked off "resBaseLo", 0] },
        { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
      ]

    if isLast then
      plain := plain.push { byteAddr := regKernelStage, value := kernelStageNoKernValue }
      plain := plain ++ (← outputStageWrites taskId)

    pure {
      taskId := taskId, logNE := 0, eneOverride := some 0,
      isFirst := isFirst, isLast := isLast, liveL2Mask := pt.liveL2Mask,
      plainWrites := plain, barWrites := barRecs, kernelRelocs := #[]
    }

  | .pe2 binOp _ outDims hw1 hw2 _ _, .pe pp =>
    let stripW := peStripWidth pp.tiling.k
    let th := pp.tiling.tileHeight
    let m := peTileM true pp.tiling.k th
    let s2Loc := pt.src2Loc?.get!
    let src2IsL2 := match s2Loc with | .l2 _ _ => true | _ => false
    let wSrc2Cfg ← encodeSrcPortCfgMulti "src2Cfg" isFirst src2IsL2
      sync.src2WaitNe5 sync.src2WaitPe22 sync.src2HandoffNe sync.src2HandoffPe
    let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked outDims.w "W")]
    let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked outDims.h "H")]
    let cU32 ← natToU32Checked outDims.c "C"
    let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked th "TH")]
    let wPeCommon ← encodeWord peCommonLayout #[("mode", peCommonTwoSrcValue)]
    let wPeTile ← encodeWord peTileLayout #[("k", ← natToU32Checked pp.tiling.k "k"), ("m", ← natToU32Checked m "m")]
    let wPeBcast ← encodeWord peBcastLayout #[
      ("src1W", if hw1.w != outDims.w then 1 else 0),
      ("src1H", if hw1.h != outDims.h then 1 else 0),
      ("src1C", if hw1.c != outDims.c then 1 else 0),
      ("src2W", if hw2.w != outDims.w then 1 else 0),
      ("src2H", if hw2.h != outDims.h then 1 else 0),
      ("src2C", if hw2.c != outDims.c then 1 else 0)
    ]
    let wResCfg ← encodeResPortCfgMulti isFirst resIsL2 pp.direct

    let mut plain : Array RegWrite := #[
      { byteAddr := regWin, value := wWin },
      { byteAddr := regHin, value := wHin },
      { byteAddr := regCin, value := cU32 },
      { byteAddr := regWout, value := wWin },
      { byteAddr := regHout, value := wHin },
      { byteAddr := regCout, value := cU32 },
      { byteAddr := regBatchN, value := 1 },
      { byteAddr := regTH, value := wTH },
      { byteAddr := regPeCommon, value := wPeCommon },
      { byteAddr := regPeTile, value := wPeTile },
      { byteAddr := regPeBcast, value := wPeBcast },
      { byteAddr := regTargetCfg, value := 0 },
      { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
      { byteAddr := regSrc2Cfg, value := wSrc2Cfg },
      { byteAddr := regResCfg, value := wResCfg },
      { byteAddr := regPeOp, value := binOp.regValue }
    ]

    let mut barRecs : Array BarRegWrite := #[]
    -- Source 1
    match pt.src1Loc with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp hw1.w 8
      let l2Row := hw1.c * l2Chs
      plain := plain ++ #[
        { byteAddr := regSrc1Base, value := ← natToU32Checked base "src1Base" },
        { byteAddr := regSrc1ChanStride, value := ← natToU32Checked l2Chs "src1ChanStride" },
        { byteAddr := regSrc1RowStride, value := ← natToU32Checked l2Row "src1RowStride" }
      ]
    | loc =>
      let whole1 := hw1.c == 1 && hw1.h == 1 && hw1.w > 1 && outDims.h > 1 && th < outDims.h
      let row1 := 2 * (if whole1 then alignUp hw1.w 8 else min (alignUp hw1.w 8) (max stripW 32))
      let u1 := hw1.c * row1
      let dmaRow := alignUp (2 * hw1.w) dmaLineBytes
      let dmaPlane := hw1.h * dmaRow
      let (slot, off) := match loc with | .surface _ s => (s, 0) | .dram o _ => (barSlotScratch, o) | .l2 _ _ => (0, 0)
      plain := plain ++ #[
        { byteAddr := regSrc1Base, value := ← natToU32Checked pp.src1StageBase "src1Base" },
        { byteAddr := regSrc1ChanStride, value := ← natToU32Checked row1 "src1ChanStride" },
        { byteAddr := regSrc1RowStride, value := ← natToU32Checked u1 "src1RowStride" },
        { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked u1 "src1RowBytesA" },
        { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked u1 "src1RowBytesB" },
        { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
        { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked dmaRow "src1DmaRowStride" },
        { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked dmaPlane "src1DmaPlaneStride" },
        { byteAddr := regSrc1Fmt, value := srcFmtWordP1 }
      ]
      let wSrc1Dep ← encodeSrcDepMulti "src1Dep" taskId sync.src1DepKind sync.src1DepDist
      barRecs := barRecs ++ #[
        { baseByteAddr := regSrc1BaseLo, barSlot := slot, values := #[← natToU32Checked off "src1BaseLo", 0] },
        { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrc1Dep] }
      ]

    -- Source 2
    match s2Loc with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp hw2.w 8
      let l2Row := hw2.c * l2Chs
      plain := plain ++ #[
        { byteAddr := regSrc2Base, value := ← natToU32Checked base "src2Base" },
        { byteAddr := regSrc2ChanStride, value := ← natToU32Checked l2Chs "src2ChanStride" },
        { byteAddr := regSrc2RowStride, value := ← natToU32Checked l2Row "src2RowStride" }
      ]
    | loc =>
      let whole2 := hw2.c == 1 && hw2.h == 1 && hw2.w > 1 && outDims.h > 1 && th < outDims.h
      let row2 := 2 * (if whole2 then alignUp hw2.w 8 else min (alignUp hw2.w 8) (max stripW 32))
      let u2 := hw2.c * row2
      let dmaRow := alignUp (2 * hw2.w) dmaLineBytes
      let dmaPlane := hw2.h * dmaRow
      let (slot, off) := match loc with | .surface _ s => (s, 0) | .dram o _ => (barSlotScratch, o) | .l2 _ _ => (0, 0)
      plain := plain ++ #[
        { byteAddr := regSrc2Base, value := ← natToU32Checked pp.src2StageBase "src2Base" },
        { byteAddr := regSrc2ChanStride, value := ← natToU32Checked row2 "src2ChanStride" },
        { byteAddr := regSrc2RowStride, value := ← natToU32Checked u2 "src2RowStride" },
        { byteAddr := regSrc2RowBytesA, value := ← natToU32Checked u2 "src2RowBytesA" },
        { byteAddr := regSrc2RowBytesB, value := ← natToU32Checked u2 "src2RowBytesB" },
        { byteAddr := regSrc2DmaCfg, value := src1DmaCfgValue },
        { byteAddr := regSrc2DmaRowStride, value := ← natToU32Checked dmaRow "src2DmaRowStride" },
        { byteAddr := regSrc2DmaPlaneStride, value := ← natToU32Checked dmaPlane "src2DmaPlaneStride" },
        { byteAddr := regSrc2Fmt, value := srcFmtWordP1 }
      ]
      let wSrc2Dep ← encodeSrcDepMulti "src2Dep" taskId sync.src2DepKind sync.src2DepDist
      barRecs := barRecs ++ #[
        { baseByteAddr := regSrc2BaseLo, barSlot := slot, values := #[← natToU32Checked off "src2BaseLo", 0] },
        { baseByteAddr := regSrc2Dep, barSlot := barSlotConfigZero, values := #[wSrc2Dep] }
      ]

    -- Result
    match pt.resLoc with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp outDims.w 8
      let l2Row := outDims.c * l2Chs
      plain := plain ++ #[
        { byteAddr := regResBase, value := ← natToU32Checked base "resBase" },
        { byteAddr := regResChanStride, value := ← natToU32Checked l2Chs "resChanStride" },
        { byteAddr := regResRowStride, value := ← natToU32Checked l2Row "resRowStride" }
      ]
    | loc =>
      let dmaRow := alignUp (2 * outDims.w) dmaLineBytes
      let dmaPlane := outDims.h * dmaRow
      let (slot, off, depCode) := match loc with
        | .surface _ s => (s, 0, resDepCode target)
        | .dram o _ => (barSlotScratch, o, resDepCode target ||| resDepScratchFlags)
        | .l2 _ _ => (0, 0, resDepCode target)
      plain := plain ++ #[
        { byteAddr := regResBase, value := ← natToU32Checked pp.resStageBase "resBase" },
        { byteAddr := regResDmaRowStride, value := ← natToU32Checked dmaRow "resDmaRowStride" },
        { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked dmaPlane "resDmaPlaneStride" },
        { byteAddr := regResFmt, value := resFmtWordP1 }
      ]
      if !pp.direct then
        let drow := 2 * min (alignUp outDims.w 8) (max stripW 32)
        let du := outDims.c * drow
        plain := plain ++ #[
          { byteAddr := regResChanStride, value := ← natToU32Checked drow "resChanStride" },
          { byteAddr := regResRowStride, value := ← natToU32Checked du "resRowStride" },
          { byteAddr := regResRowBytesA, value := ← natToU32Checked du "resRowBytesA" },
          { byteAddr := regResRowBytesB, value := ← natToU32Checked du "resRowBytesB" }
        ]
      let wResDep ← encodeWord resDepLayout #[
        ("code", depCode), ("task", taskIdU32), ("direct", if pp.direct then 1 else 0)
      ]
      barRecs := barRecs ++ #[
        { baseByteAddr := regResBaseLo, barSlot := slot, values := #[← natToU32Checked off "resBaseLo", 0] },
        { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
      ]

    if isLast then
      plain := plain.push { byteAddr := regKernelStage, value := kernelStageNoKernValue }
      plain := plain ++ (← outputStageWrites taskId)

    pure {
      taskId := taskId, logNE := 0, eneOverride := some 0,
      isFirst := isFirst, isLast := isLast, liveL2Mask := pt.liveL2Mask,
      plainWrites := plain, barWrites := barRecs, kernelRelocs := #[]
    }

  | .lut act dims lutPayload?, .lut lp =>
    let (isLut, isRelu, preScaleBits, shiftBits?) ← match act with
      | .relu => pure (false, true, f16PosOne, none)
      | .copy => pure (false, false, f16PosOne, none)
      | .lut _ =>
        let some payload := lutPayload? | throw "internal error: missing LutPayload"
        pure (true, false, payload.preScaleBits, payload.inputShiftBits)
    let t := lp.tiling
    let stripW := t.stripWidth
    let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked dims.w "W")]
    let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked dims.h "H")]
    let cU32 ← natToU32Checked dims.c "C"
    let wDwGroups ← encodeWord dwGroupsLayout #[("perGroupIn", 1)]
    let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked t.tileHeight "TH")]
    let wNeCommon ← encodeWord neCommonLayout #[
      ("kind", 0),
      ("shPref", ← natToU32Checked t.shPref "shPref"),
      ("shMin", ← natToU32Checked t.shPref "shMin"),
      ("shMax", ← natToU32Checked t.shMax "shMax"),
      ("F", ← natToU32Checked t.f "F"),
      ("wide3x3", 0)
    ]
    let wNeTile ← encodeWord neTileLayout #[
      ("ocg", 0),
      ("fat", ← natToU32Checked t.fat "fat"),
      ("WU", ← natToU32Checked t.wu "WU")
    ]
    let wResCfg ← encodeResPortCfgMulti isFirst resIsL2 lp.direct
    let wNeActOp ← encodeWord neActOpLayout #[
      ("noShift", if shiftBits?.isSome then 0 else 1),
      ("relu", if isRelu then 1 else 0),
      ("lut", if isLut then 1 else 0)
    ]
    let wNeScale ← encodeWord neScaleLayout #[("val", preScaleBits.toUInt32)]

    let mut plain : Array RegWrite := #[
      { byteAddr := regWin, value := wWin },
      { byteAddr := regHin, value := wHin },
      { byteAddr := regCin, value := cU32 },
      { byteAddr := regWout, value := wWin },
      { byteAddr := regHout, value := wHin },
      { byteAddr := regCout, value := cU32 },
      { byteAddr := regBatchN, value := 1 },
      { byteAddr := regDwGroups, value := wDwGroups },
      { byteAddr := regTH, value := wTH },
      { byteAddr := regNeCommon, value := wNeCommon },
      { byteAddr := regNeTile, value := wNeTile },
      { byteAddr := regTargetCfg, value := targetCfgValue target },
      { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
      { byteAddr := regResCfg, value := wResCfg },
      { byteAddr := regNeOp, value := wNeActOp },
      { byteAddr := regNeScale, value := wNeScale }
    ]
    if isFirst then
      plain := plain.push { byteAddr := regSrc2Cfg, value := src2CfgValue }
    if let some shiftBits := shiftBits? then
      plain := plain ++ #[
        { byteAddr := regNeActShiftEn, value := ← encodeWord neActShiftEnLayout #[("enable", 1)] },
        { byteAddr := regScalarBias, value := ← encodeWord scalarBiasLayout #[("val", shiftBits.toUInt32)] }
      ]

    let mut barRecs : Array BarRegWrite := #[]
    let mut relocs : Array TaskReloc := #[]
    if isLut then
      let wKernStage ← encodeKernelStageMulti taskId isLast false
      let wLutCfg ← encodeWord chunkSlotLayout #[("enable", 1), ("on20", 1), ("task", taskIdU32)]
      plain := plain ++ #[
        { byteAddr := regNeCfg, value := neActCfgValue },
        { byteAddr := regKernelStage, value := wKernStage },
        { byteAddr := regLutCfg, value := wLutCfg },
        { byteAddr := regLutSize, value := lutTableBytes.toUInt32 }
      ]
      barRecs := barRecs.push {
        baseByteAddr := regLutBaseLo, barSlot := kernelBarSlot,
        values := #[← natToU32Checked pt.kernelBaseOff "lutBaseLo", 0]
      }
      relocs := #[{ barRecIndex := 0, payloadWordIndex := 0 }]
    else if isLast then
      plain := plain.push { byteAddr := regKernelStage, value := kernelStageNoKernValue }

    match pt.src1Loc with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp dims.w 8
      let l2Row := dims.c * l2Chs
      plain := plain ++ #[
        { byteAddr := regSrc1Base, value := ← natToU32Checked base "src1Base" },
        { byteAddr := regSrc1ChanStride, value := ← natToU32Checked l2Chs "src1ChanStride" },
        { byteAddr := regSrc1RowStride, value := ← natToU32Checked l2Row "src1RowStride" }
      ]
    | loc =>
      let row := 2 * min (alignUp dims.w 8) stripW
      let u := dims.c * row
      let dmaRow := alignUp (2 * dims.w) dmaLineBytes
      let dmaPlane := dims.h * dmaRow
      let (slot, off) := match loc with | .surface _ s => (s, 0) | .dram o _ => (barSlotScratch, o) | .l2 _ _ => (0, 0)
      plain := plain ++ #[
        { byteAddr := regSrc1Base, value := ← natToU32Checked lp.srcStageBase "src1Base" },
        { byteAddr := regSrc1ChanStride, value := ← natToU32Checked row "src1ChanStride" },
        { byteAddr := regSrc1RowStride, value := ← natToU32Checked u "src1RowStride" },
        { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked u "src1RowBytesA" },
        { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked u "src1RowBytesB" },
        { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
        { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked dmaRow "src1DmaRowStride" },
        { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked dmaPlane "src1DmaPlaneStride" },
        { byteAddr := regSrc1Fmt, value := srcFmtWordP1 }
      ]
      let wSrc1Dep ← encodeSrcDepMulti "src1Dep" taskId sync.src1DepKind sync.src1DepDist
      barRecs := barRecs ++ #[
        { baseByteAddr := regSrc1BaseLo, barSlot := slot, values := #[← natToU32Checked off "src1BaseLo", 0] },
        { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrc1Dep] }
      ]

    match pt.resLoc with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp dims.w 8
      let l2Row := dims.c * l2Chs
      plain := plain ++ #[
        { byteAddr := regResBase, value := ← natToU32Checked base "resBase" },
        { byteAddr := regResChanStride, value := ← natToU32Checked l2Chs "resChanStride" },
        { byteAddr := regResRowStride, value := ← natToU32Checked l2Row "resRowStride" }
      ]
    | loc =>
      let dmaRow := alignUp (2 * dims.w) dmaLineBytes
      let dmaPlane := dims.h * dmaRow
      let (slot, off, depCode) := match loc with
        | .surface _ s => (s, 0, resDepCode target)
        | .dram o _ => (barSlotScratch, o, resDepCode target ||| resDepScratchFlags)
        | .l2 _ _ => (0, 0, resDepCode target)
      plain := plain ++ #[
        { byteAddr := regResBase, value := ← natToU32Checked lp.resStageBase "resBase" },
        { byteAddr := regResDmaRowStride, value := ← natToU32Checked dmaRow "resDmaRowStride" },
        { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked dmaPlane "resDmaPlaneStride" },
        { byteAddr := regResFmt, value := resFmtWordP1 }
      ]
      if !lp.direct then
        let drow := 2 * min (alignUp dims.w 8) (max stripW 32)
        let du := dims.c * drow
        plain := plain ++ #[
          { byteAddr := regResChanStride, value := ← natToU32Checked drow "resChanStride" },
          { byteAddr := regResRowStride, value := ← natToU32Checked du "resRowStride" },
          { byteAddr := regResRowBytesA, value := ← natToU32Checked du "resRowBytesA" },
          { byteAddr := regResRowBytesB, value := ← natToU32Checked du "resRowBytesB" }
        ]
      let wResDep ← encodeWord resDepLayout #[
        ("code", depCode), ("task", taskIdU32), ("direct", if lp.direct then 1 else 0)
      ]
      barRecs := barRecs ++ #[
        { baseByteAddr := regResBaseLo, barSlot := slot, values := #[← natToU32Checked off "resBaseLo", 0] },
        { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
      ]

    if isLast then
      plain := plain ++ (← outputStageWrites taskId)

    let h4Override : Option UInt32 :=
      if isFirst && isLast && act == .copy then some taskHeaderWord4CopyConst else none

    pure {
      taskId := taskId, logNE := t.logNE, eneOverride := some (t.logNE + 1),
      header4Override := h4Override,
      isFirst := isFirst, isLast := isLast, liveL2Mask := pt.liveL2Mask,
      plainWrites := plain, barWrites := barRecs, kernelRelocs := relocs
    }

  | _, _ =>
    throw s!"internal error: op/plan mismatch when lowering node '{nd.id}'"

-- ============================================================================
-- Top-Level Multi-Task Graph Compiler
-- ============================================================================

/-- Compile an arbitrary multi-task DAG (`1..256` tasks, `1..3` inputs, `1` output) into a
    statically verified HWX binary (checked by independent `Vane.Verify.verifyHwxTasksSummary`)
    and its 4 host-built Linux LOAD sections. -/
def compileGraph (spec : GraphSpec) : Except String CompiledGraph := do
  let (shortNames, vNodes, lastConsumer) ← validateGraph spec
  let n := vNodes.size

  -- 1. Partition intermediate nodes (0 .. n - 2) into pinned L2, auto L2 candidates, and DRAM
  let mut pinnedL2 : Array Nat := #[]
  let mut autoL2 : Array Nat := #[]
  let mut dramIds : Array Nat := #[]

  for i in [:n - 1] do
    let nd := vNodes[i]!
    let effMode := nd.outPlacement.getD spec.defaultPlacement
    let eligible := isNodeL2Eligible vNodes i
    match effMode with
    | .dram =>
      dramIds := dramIds.push i
    | .l2 =>
      if !eligible then
        throw s!"node '{nd.id}': forced L2 placement is ineligible (size >= 2 MiB or broadcast/cropped consumer)"
      pinnedL2 := pinnedL2.push i
    | .auto =>
      if eligible then
        autoL2 := autoL2.push i
      else
        dramIds := dramIds.push i

  -- 2. Iterative L2 allocation & deterministic auto-spill loop
  let mut curAutoL2 := autoL2
  let mut placedL2 : List PlacedBuf := []
  let mut plans : Array TaskPlanKind := #[]
  let mut peakL2 : Nat := 0
  let mut solved := false

  while !solved do
    let l2Ids := pinnedL2 ++ curAutoL2
    let liveBufs : List LiveBuf := l2Ids.toList.map fun i => {
      id := i
      bytes := l2ResidentBytes vNodes[i]!.outDims
      first := i
      last := lastConsumer[i]!
      priority := lastConsumer[i]! - i
    }
    let trialPlaced := allocate true liveBufs
    let l2Top := topOf trialPlaced
    let planAttempt : Except (Option Nat × String) (Array TaskPlanKind × Nat) :=
      if l2Top > l2CapacityBytes then
        .error (none, s!"live L2-resident buffers reach {l2Top}B > {l2CapacityBytes}B (2 MiB)")
      else
        match tryPlanAllTasks spec.target vNodes trialPlaced with
        | .ok res => .ok res
        | .error (failTask, msg) => .error (some failTask, msg)

    match planAttempt with
    | .ok (ps, pk) =>
      placedL2 := trialPlaced
      plans := ps
      peakL2 := pk
      solved := true
    | .error (failTask?, msg) =>
      if curAutoL2.isEmpty then
        if !pinnedL2.isEmpty then
          throw s!"forced L2 placement exceeds 2 MiB capacity: {msg}"
        else
          throw s!"graph compilation failed even with all intermediates in DRAM: {msg}"
      -- Deterministically spill the largest auto-L2 tensor (preferring one live at failTask?)
      let liveAuto : Array Nat := match failTask? with
        | some ft => curAutoL2.filter fun i => i <= ft && ft <= lastConsumer[i]!
        | none => #[]
      let pool : Array Nat := if liveAuto.isEmpty then curAutoL2 else liveAuto
      let mut victim := pool[0]!
      let mut bestBytes := l2ResidentBytes vNodes[victim]!.outDims
      for cand in pool do
        let b := l2ResidentBytes vNodes[cand]!.outDims
        if b > bestBytes || (b == bestBytes && cand > victim) then
          victim := cand
          bestBytes := b
      curAutoL2 := curAutoL2.filter (fun x => x != victim)
      dramIds := dramIds.push victim

  -- 3. Allocate DRAM scratch buffers with liveness-based reuse (first-fit 64B aligned)
  let dramLiveBufs : List LiveBuf := dramIds.toList.map fun i => {
    id := i
    bytes := dramScratchBytes vNodes[i]!.outDims
    first := i
    last := lastConsumer[i]!
    priority := lastConsumer[i]! - i
  }
  let placedDram := allocate true dramLiveBufs
  let scratchBytes := topOf placedDram

  -- 4. Build final TensorLocation table for every node
  let outSurfIdx := spec.inputs.size
  let outBarSlot : UInt8 := barSlotFirstSurface + outSurfIdx.toUInt8
  let mut nodeLocs : Array TensorLocation := Array.mkEmpty n
  for i in [:n] do
    if i + 1 == n then
      nodeLocs := nodeLocs.push (.surface outSurfIdx outBarSlot)
    else
      match lookupPlaced placedL2 i with
      | some pb => nodeLocs := nodeLocs.push (.l2 pb.base pb.bytes)
      | none =>
        match lookupPlaced placedDram i with
        | some pb => nodeLocs := nodeLocs.push (.dram pb.base pb.bytes)
        | none => throw s!"internal error: node {i} missing from both L2 and DRAM allocations"

  -- 5. Assemble shared __KERN_0 & per-task memory footprints
  let (preTasks, kernBuf) ← buildPreLoweredTasks spec shortNames vNodes plans placedL2 nodeLocs

  -- 6. Compute silicon-proven synchronization (HB/HBL/srcReadNE/src2AheadPE/[XE]/DRAM WAR+WAW)
  let syncs := computeSynchronization preTasks

  -- 7. Lower all tasks to TaskDesc
  let totalSurfs := spec.inputs.size + 1
  let kernelBarSlot : UInt8 := barSlotFirstSurface + totalSurfs.toUInt8
  let mut taskDescs : Array TaskDesc := Array.mkEmpty n
  for i in [:n] do
    let td ← emitTaskDesc spec.target n kernelBarSlot preTasks[i]! syncs[i]!
    taskDescs := taskDescs.push td

  -- 8. Build network surfaces, optional scratch, optional kernel, and emit HWX
  let mut surfaces : Array SurfaceSpec := Array.mkEmpty totalSurfs
  for i in [:spec.inputs.size] do
    let inp := spec.inputs[i]!
    let sn := shortNames[i]!
    let sym := s!"input_{inp.name}"
    let s ← makeFp16Surface spec.target sn sym .input inp.shape.c inp.shape.h inp.shape.w
    surfaces := surfaces.push s
  let outNode := vNodes[n - 1]!
  let outSurf ← makeFp16Surface spec.target "y" s!"output_{outNode.id}" .output
    outNode.outDims.c outNode.outDims.h outNode.outDims.w
  surfaces := surfaces.push outSurf

  let scratchSpec? : Option ScratchSpec ←
    if scratchBytes == 0 then pure none
    else
      let szU64 ← natToU64Checked scratchBytes "scratch size"
      let usU32 ← natToU32Checked scratchBytes "scratch usable"
      pure (some { size := szU64, usable := usU32 })

  let kernelOpt? : Option ByteArray := if kernBuf.isEmpty then none else some kernBuf

  let emitted ← emitProgram {
    target := spec.target
    surfaces := surfaces
    tasks := taskDescs
    kernel := kernelOpt?
    scratch := scratchSpec?
    constants := zeroBytes textConstBytes
    ident := some "vane"
  }

  -- 9. Independent whole-program static safety verification (S1–S5)
  let summary ← verifyHwxTasksSummary emitted.hwx emitted.info (requireKnownTask := true)
  if summary.total != n then
    throw s!"internal error: verifier checked {summary.total} tasks != {n}"

  let compiledNodes : Array CompiledGraphNode := Array.ofFn (n := n) fun i =>
    let nd := vNodes[i.val]!
    {
      id := nd.id
      taskId := i.val
      engine := nd.engine
      opKind := nd.opKind
      outDims := nd.outDims
      outLocation := nodeLocs[i.val]!
      direct := plans[i.val]!.direct
      l2UseBytes := plans[i.val]!.l2Peak
    }

  pure {
    target := spec.target
    taskCount := n
    scratchBytes := scratchBytes
    l2PeakBytes := peakL2
    kernelBytes := kernBuf.size
    nodes := compiledNodes
    hwx := emitted.hwx
    info := emitted.info
    sections := emitted.sections
  }

end Vane.Graph
