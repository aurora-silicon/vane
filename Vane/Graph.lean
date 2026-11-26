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
import Vane.Pool
import Vane.Conv
import Vane.View
import Vane.Matmul

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
open Vane.Pool
open Vane.Conv
open Vane.View
open Vane.Matmul

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

/-- Spatial H-strip fusion mode for multi-task graphs. -/
inductive StripMode where
  | off
  | auto
  | fixed (height : Nat)
  deriving Repr, BEq, DecidableEq, Inhabited

namespace StripMode

def name : StripMode → String
  | .off => "off"
  | .auto => "auto"
  | .fixed h => toString h

def parse? (s : String) : Option StripMode :=
  match s.toLower with
  | "off" => some .off
  | "auto" => some .auto
  | raw =>
    match raw.toNat? with
    | some n => if n >= 1 then some (.fixed n) else none
    | none => none

end StripMode

/-- Supported node operations in a multi-task ANE DAG. -/
inductive NodeOp where
  | conv1x1 (cin cout : Nat) (weights : ByteArray) (bias : Option ByteArray := none) (act : ConvActivation := .none)
  | conv3x3 (mode : Conv3x3Mode) (cin cout stride : Nat) (padding : PadMode) (weights : ByteArray) (bias : Option ByteArray := none) (act : ConvActivation := .none)
  | linear (cin cout : Nat) (weights : ByteArray) (bias : Option ByteArray := none) (act : ConvActivation := .none)
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

/-- Complete multi-task DAG specification for compilation. -/
structure GraphSpec where
  target : Target
  inputs : Array GraphInput
  nodes : Array GraphNode
  defaultPlacement : EdgePlacementMode := .auto
  stripMode : StripMode := .off
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

/-- Optional horizontal row-band window on a full-frame network surface (`fullH == 0` means full frame). -/
private structure SurfaceWindow where
  rowOffset : Nat := 0
  fullH : Nat := 0
  deriving Repr, BEq, Inhabited

/-- Compute `(baseOff, rowStride, planeStride, exactExtent)` for a `P = 1` surface DMA access. -/
private def surfaceDmaWindow (win : SurfaceWindow) (c h w : Nat) : Nat × Nat × Nat × Nat :=
  let effFullH := if win.fullH == 0 then h else win.fullH
  let rowStride := alignUp (2 * w) dmaLineBytes
  let planeStride := effFullH * rowStride
  let baseOff := win.rowOffset * rowStride
  let extent := if c == 0 || h == 0 then 0 else (c - 1) * planeStride + h * rowStride
  (baseOff, rowStride, planeStride, extent)

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
  -- L2-resident PE2 broadcast operands require th == outDims.h (otherwise spill to DRAM)
  let hasL2Bcast :=
    (src1IsL2 && hwSrc1 != outDims) ||
    (match hwSrc2? with | some hw2 => src2IsL2 && hw2 != outDims | none => false)
  if hasL2Bcast || outDims.h <= 1 then
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
    (srcIsL2 resIsL2 : Bool) (occupied : List Range) (th : Nat) (srcFloor32 : Bool := false) : Option (Nat × Nat × Nat) :=
  let stripW := probe.stripWidth
  let direct := if resIsL2 then false else neActIsDirect target w probe
  let effSrcStrip := if srcFloor32 then max stripW 32 else stripW
  let row := 2 * min (alignUp w 8) effSrcStrip
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
    (srcIsL2 resIsL2 : Bool) (occupied : List Range) (srcFloor32 : Bool := false) : Except String NeActTaskPlan := do
  for shPref in [:5] do
    let probe : NeActTiling := {
      logNE := 0, f := 0, fat := 0, shPref := shPref, shMax := shPref, tileHeight := 1
    }
    if let some (sb, rb, pk) := neActStageBasesForTh target c h w probe srcIsL2 resIsL2 occupied h srcFloor32 then
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
        match neActStageBasesForTh target c h w probe srcIsL2 resIsL2 occupied mid srcFloor32 with
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
  | pePool (op : PoolKind) (excludePad : Bool) (dims : TensorDims) (gh gw : PoolAxisGeom)
  | nePool (op : PoolKind) (dims : TensorDims) (gh gw : PoolAxisGeom)
  | peReduce (op : ReduceOp) (axes : ReduceAxes) (eps : UInt16) (dims : TensorDims)
  | neChannelReduce (op : ReduceOp) (dims : TensorDims)
  | neView (copy : ViewCopy)
  | neMatmul (heads m0 m totalM k n : Nat) (scaleBits : UInt16) (refA refB : TensorRef)

private instance : Inhabited PlannedOp := ⟨.pe2 .add false default default default default default⟩

private structure ValidatedNode where
  id : String
  logicalId : String := ""
  nodeIdx : Nat
  origNodeIdx : Nat := 0
  inputs : Array TensorRef
  op : PlannedOp
  outDims : TensorDims
  groupOutDims : TensorDims := default
  outByteOffset : Nat := 0
  minDramBytes : Nat := 0
  sharedOutGroup : Option Nat := none
  isFinalGroup : Bool := false
  outPlacement : Option EdgePlacementMode
  engine : Engine
  isNeCompute : Bool
  usesKernel : Bool
  opKind : String
  src1Win : SurfaceWindow := {}
  src2Win : SurfaceWindow := {}
  resWin : SurfaceWindow := {}
  deriving Inhabited

private inductive TaskPlanKind where
  | conv (p : ConvTaskPlan)
  | pe (p : PeTaskPlan)
  | lut (p : NeActTaskPlan)
  | pePool (p : PePoolPlan)
  | peReduce (p : PeReducePlan)
  | nePool (p : NeWeightlessPlan)
  | neChannelReduce (p : NeWeightlessPlan)
  | neView (p : ViewTaskPlan)
  | neMatmul (p : MatmulTaskPlan)

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
  | .pePool p => p.direct
  | .peReduce p => p.direct
  | .nePool p => p.direct
  | .neChannelReduce p => p.direct
  | .neView p => p.direct
  | .neMatmul p => p.direct

private def TaskPlanKind.l2Peak : TaskPlanKind → Nat
  | .conv p => p.l2TaskPeak
  | .pe p => p.l2TaskPeak
  | .lut p => p.l2TaskPeak
  | .pePool p => p.l2TaskPeak
  | .peReduce p => p.l2TaskPeak
  | .nePool p => p.l2TaskPeak
  | .neChannelReduce p => p.l2TaskPeak
  | .neView p => p.l2TaskPeak
  | .neMatmul p => p.l2TaskPeak

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
  | .linear .. => true
  | .lut (.lut _) => true
  | _ => false

/-- Validate graph structure, DAG topological ordering, tensor shapes, and op contracts,
    expanding multi-task ops (`concat` and output-channel-split `conv1x1` / `linear`). -/
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
  let mut lastConsumer : Array Nat := #[]
  -- Track logical node representatives `(repIdx, endIdx, isFinal)` for unconsumed-node checking
  let mut logicalReps : Array (Nat × Nat × Bool) := Array.mkEmpty spec.nodes.size

  for logIdx in [:spec.nodes.size] do
    let nd := spec.nodes[logIdx]!
    let isLastLogical := logIdx + 1 == spec.nodes.size
    if nd.id.isEmpty then
      throw s!"graph node {logIdx} has empty id"
    if nameMap.any (fun (nm, _, _) => nm == nd.id) then
      throw s!"duplicate graph tensor/node id '{nd.id}'"

    let mut inRefs : Array TensorRef := #[]
    let mut inDims : Array TensorDims := #[]
    for inName in nd.inputs do
      match lookup nameMap inName with
      | none =>
        throw s!"node '{nd.id}': input '{inName}' is undefined or not in topological order"
      | some (_, ref, dims) =>
        inRefs := inRefs.push ref
        inDims := inDims.push dims

    let repIdx := vNodes.size

    let pushExpandedConv1x1 := fun (opLabel : String) (cin cout : Nat) (weights : ByteArray)
        (bias : Option ByteArray) (act : ConvActivation) (d : TensorDims) => do
      if d.c != cin then
        throw s!"node '{nd.id}' ({opLabel}): input C={d.c} != cin={cin}"
      let fullSh : ConvSpec := { kernelSize := 1, mode := .dense, h := d.h, w := d.w, cin := cin, cout := cout }
      validateConvShape fullSh
      let lutPayload? ← match act with
        | .lut lspec => some <$> buildLutTable lspec
        | _ => pure none
      let shiftBits? := lutPayload?.bind (fun lp => lp.inputShiftBits)
      let hasEffBias := bias.isSome || shiftBits?.isSome
      let extraLutBytes := if lutPayload?.isSome then lutTableBytes else 0
      let slices ← chooseConv1x1OutputSlices spec.target d.h d.w cin cout extraLutBytes hasEffBias
      let fullOutDims : TensorDims := { c := cout, h := d.h, w := d.w }
      if slices.size == 1 then
        let scalarBiasMode := cout == 1
        let biasInKernel := hasEffBias && !scalarBiasMode
        let tIdx := vNodes.size
        if tIdx >= 255 then
          throw s!"node '{nd.id}' (taskId={tIdx}) uses __KERN_0, which requires taskId < 255"
        pure (#[{
          id := nd.id
          logicalId := nd.id
          nodeIdx := tIdx
          origNodeIdx := tIdx
          inputs := inRefs
          op := PlannedOp.conv fullSh weights bias act lutPayload? biasInKernel
          outDims := fullOutDims
          groupOutDims := fullOutDims
          outByteOffset := 0
          sharedOutGroup := none
          isFinalGroup := isLastLogical
          outPlacement := nd.outPlacement
          engine := Engine.ne
          isNeCompute := true
          usesKernel := true
          opKind := opLabel
        }], fullOutDims)
      else
        if weights.size != 2 * cout * cin then
          throw s!"node '{nd.id}' ({opLabel}): weights byte size {weights.size} != expected {2 * cout * cin}"
        if let some b := bias then
          if b.size != 2 * cout then
            throw s!"node '{nd.id}' ({opLabel}): bias byte size {b.size} != expected {2 * cout}"
        let starts := prefixStarts slices.toList
        let outRowStride := alignUp (2 * d.w) dmaLineBytes
        let outPlaneStride := d.h * outRowStride
        let mut sliceNodes : Array ValidatedNode := Array.mkEmpty slices.size
        for si in [:slices.size] do
          let tIdx := repIdx + si
          if tIdx >= 255 then
            throw s!"node '{nd.id}' slice {si} (taskId={tIdx}) uses __KERN_0, which requires taskId < 255"
          let c0 := starts[si]!
          let dc := slices[si]!
          let wSlice := weights.extract (2 * c0 * cin) (2 * (c0 + dc) * cin)
          let bSlice? := bias.map (fun b => b.extract (2 * c0) (2 * (c0 + dc)))
          let sh_i : ConvSpec := { kernelSize := 1, mode := .dense, h := d.h, w := d.w, cin := cin, cout := dc }
          let biasInKernel_i := hasEffBias && dc > 1
          sliceNodes := sliceNodes.push {
            id := s!"{nd.id}_s{si}"
            logicalId := nd.id
            nodeIdx := tIdx
            origNodeIdx := tIdx
            inputs := inRefs
            op := PlannedOp.conv sh_i wSlice bSlice? act lutPayload? biasInKernel_i
            outDims := { c := dc, h := d.h, w := d.w }
            groupOutDims := fullOutDims
            outByteOffset := c0 * outPlaneStride
            sharedOutGroup := some repIdx
            isFinalGroup := isLastLogical
            outPlacement := nd.outPlacement
            engine := Engine.ne
            isNeCompute := true
            usesKernel := true
            opKind := opLabel
          }
        pure (sliceNodes, fullOutDims)

    let (expandedNodes, fullOutDims) : Array ValidatedNode × TensorDims ← match nd.op with
      | .conv1x1 cin cout weights bias act =>
        if inRefs.size != 1 then
          throw s!"node '{nd.id}' (conv1x1): expected 1 input, got {inRefs.size}"
        pushExpandedConv1x1 "conv1x1" cin cout weights bias act inDims[0]!
      | .linear cin cout weights bias act =>
        if inRefs.size != 1 then
          throw s!"node '{nd.id}' (linear): expected 1 input, got {inRefs.size}"
        pushExpandedConv1x1 "linear" cin cout weights bias act inDims[0]!
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
        if repIdx >= 255 then
          throw s!"node '{nd.id}' (taskId={repIdx}) uses __KERN_0, which requires taskId < 255"
        pure (#[{
          id := nd.id, logicalId := nd.id, nodeIdx := repIdx, origNodeIdx := repIdx,
          inputs := inRefs, op := PlannedOp.conv sh weights bias act lutPayload? biasInKernel,
          outDims := od, groupOutDims := od, outByteOffset := 0, sharedOutGroup := none,
          isFinalGroup := isLastLogical, outPlacement := nd.outPlacement,
          engine := Engine.ne, isNeCompute := true, usesKernel := true, opKind := "conv3x3"
        }], od)
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
        pure (#[{
          id := nd.id, logicalId := nd.id, nodeIdx := repIdx, origNodeIdx := repIdx,
          inputs := inRefs, op := PlannedOp.pe1 op d,
          outDims := d, groupOutDims := d, outByteOffset := 0, sharedOutGroup := none,
          isFinalGroup := isLastLogical, outPlacement := nd.outPlacement,
          engine := Engine.pe, isNeCompute := false, usesKernel := false, opKind := s!"pe_{op.opName}"
        }], d)
      | .pe2 binOp rev =>
        if inRefs.size != 2 then
          throw s!"node '{nd.id}' (pe2): expected 2 inputs, got {inRefs.size}"
        let d1 := inDims[0]!
        let d2 := inDims[1]!
        let od ← broadcastDims d1 d2
        let (hw1, hw2, r1, r2) := if rev then (d2, d1, inRefs[1]!, inRefs[0]!) else (d1, d2, inRefs[0]!, inRefs[1]!)
        let lbl := if rev && binOp == .sub then "pe_rsub" else s!"pe_{binOp.name}"
        pure (#[{
          id := nd.id, logicalId := nd.id, nodeIdx := repIdx, origNodeIdx := repIdx,
          inputs := inRefs, op := PlannedOp.pe2 binOp rev od hw1 hw2 r1 r2,
          outDims := od, groupOutDims := od, outByteOffset := 0, sharedOutGroup := none,
          isFinalGroup := isLastLogical, outPlacement := nd.outPlacement,
          engine := Engine.pe, isNeCompute := false, usesKernel := false, opKind := lbl
        }], od)
      | .lut act =>
        if inRefs.size != 1 then
          throw s!"node '{nd.id}' (lut): expected 1 input, got {inRefs.size}"
        let d := inDims[0]!
        validateTensorDims d s!"node '{nd.id}'"
        let lutPayload? ← match act with
          | .lut lspec => some <$> buildLutTable lspec
          | _ => pure none
        let isLut := lutPayload?.isSome
        if isLut && repIdx >= 255 then
          throw s!"node '{nd.id}' (taskId={repIdx}) uses __KERN_0, which requires taskId < 255"
        pure (#[{
          id := nd.id, logicalId := nd.id, nodeIdx := repIdx, origNodeIdx := repIdx,
          inputs := inRefs, op := PlannedOp.lut act d lutPayload?,
          outDims := d, groupOutDims := d, outByteOffset := 0, sharedOutGroup := none,
          isFinalGroup := isLastLogical, outPlacement := nd.outPlacement,
          engine := Engine.ne, isNeCompute := isLut, usesKernel := isLut, opKind := s!"lut_{act.name}"
        }], d)
      | .pool op kh kw sh sw pad excl =>
        if inRefs.size != 1 then
          throw s!"node '{nd.id}' ({op.opName}): expected 1 input, got {inRefs.size}"
        let d := inDims[0]!
        let pspec : PoolSpec := {
          target := spec.target, op := op, dims := d, kh := kh, kw := kw, sh := sh, sw := sw, pad := pad, excludePad := excl
        }
        let (backend, gh, gw) ← resolvePoolBackend pspec
        let od : TensorDims := { c := d.c, h := gh.out, w := gw.out }
        let (pop, eng) := match backend with
          | .pePool => (PlannedOp.pePool op excl d gh gw, Engine.pe)
          | .nePool => (PlannedOp.nePool op d gh gw, Engine.ne)
          | .peReduceGlobal =>
            let redOp : ReduceOp := match op with | .max => .max | .avg => .mean
            (PlannedOp.peReduce redOp { reduceH := true, reduceW := true } 0 d, Engine.pe)
        pure (#[{
          id := nd.id, logicalId := nd.id, nodeIdx := repIdx, origNodeIdx := repIdx,
          inputs := inRefs, op := pop,
          outDims := od, groupOutDims := od, outByteOffset := 0, sharedOutGroup := none,
          isFinalGroup := isLastLogical, outPlacement := nd.outPlacement,
          engine := eng, isNeCompute := false, usesKernel := false, opKind := op.opName
        }], od)
      | .reduce op axes eps =>
        if inRefs.size != 1 then
          throw s!"node '{nd.id}' (reduce_{op.name}): expected 1 input, got {inRefs.size}"
        let d := inDims[0]!
        let rspec : ReduceSpec := {
          target := spec.target, op := op, dims := d, axes := axes, eps := eps
        }
        let (backend, od) ← resolveReduceBackend rspec
        let (pop, eng) := match backend with
          | .peReduce => (PlannedOp.peReduce op axes eps d, Engine.pe)
          | .neChannelReduce => (PlannedOp.neChannelReduce op d, Engine.ne)
        pure (#[{
          id := nd.id, logicalId := nd.id, nodeIdx := repIdx, origNodeIdx := repIdx,
          inputs := inRefs, op := pop,
          outDims := od, groupOutDims := od, outByteOffset := 0, sharedOutGroup := none,
          isFinalGroup := isLastLogical, outPlacement := nd.outPlacement,
          engine := eng, isNeCompute := false, usesKernel := false, opKind := s!"reduce_{op.name}"
        }], od)
      | .transpose perm =>
        if inRefs.size != 1 then
          throw s!"node '{nd.id}' (transpose): expected 1 input, got {inRefs.size}"
        let d := inDims[0]!
        let vp ← planTranspose { n := 1, c := d.c, h := d.h, w := d.w } perm
        if vp.outShape.n != 1 then
          throw s!"node '{nd.id}' (transpose): graph tensors require output N=1, got N={vp.outShape.n}"
        let od : TensorDims := { c := vp.outShape.c, h := vp.outShape.h, w := vp.outShape.w }
        pure (#[{
          id := nd.id, logicalId := nd.id, nodeIdx := repIdx, origNodeIdx := repIdx,
          inputs := inRefs, op := PlannedOp.neView vp.copies[0]!,
          outDims := od, groupOutDims := od, outByteOffset := 0, sharedOutGroup := none,
          isFinalGroup := isLastLogical, outPlacement := nd.outPlacement,
          engine := Engine.ne, isNeCompute := false, usesKernel := false, opKind := "transpose"
        }], od)
      | .slice beginIdx endIdx step =>
        if inRefs.size != 1 then
          throw s!"node '{nd.id}' (slice): expected 1 input, got {inRefs.size}"
        let d := inDims[0]!
        let vp ← planSlice { n := 1, c := d.c, h := d.h, w := d.w } beginIdx endIdx step
        if vp.outShape.n != 1 then
          throw s!"node '{nd.id}' (slice): graph tensors require output N=1, got N={vp.outShape.n}"
        let od : TensorDims := { c := vp.outShape.c, h := vp.outShape.h, w := vp.outShape.w }
        pure (#[{
          id := nd.id, logicalId := nd.id, nodeIdx := repIdx, origNodeIdx := repIdx,
          inputs := inRefs, op := PlannedOp.neView vp.copies[0]!,
          outDims := od, groupOutDims := od, outByteOffset := 0, sharedOutGroup := none,
          isFinalGroup := isLastLogical, outPlacement := nd.outPlacement,
          engine := Engine.ne, isNeCompute := false, usesKernel := false, opKind := "slice"
        }], od)
      | .reshape targetDims =>
        if inRefs.size != 1 then
          throw s!"node '{nd.id}' (reshape): expected 1 input, got {inRefs.size}"
        let d := inDims[0]!
        let vp ← planReshape { n := 1, c := d.c, h := d.h, w := d.w }
          { n := 1, c := targetDims.c, h := targetDims.h, w := targetDims.w }
        let od : TensorDims := { c := vp.outShape.c, h := vp.outShape.h, w := vp.outShape.w }
        pure (#[{
          id := nd.id, logicalId := nd.id, nodeIdx := repIdx, origNodeIdx := repIdx,
          inputs := inRefs, op := PlannedOp.neView vp.copies[0]!,
          outDims := od, groupOutDims := od, outByteOffset := 0, sharedOutGroup := none,
          isFinalGroup := isLastLogical, outPlacement := nd.outPlacement,
          engine := Engine.ne, isNeCompute := false, usesKernel := false, opKind := "reshape"
        }], od)
      | .concat axis =>
        if inRefs.size < 2 then
          throw s!"node '{nd.id}' (concat): expected >= 2 inputs, got {inRefs.size}"
        let inShapes4 : Array Shape4D := inDims.map fun d => { n := 1, c := d.c, h := d.h, w := d.w }
        let vp ← planConcat inShapes4 axis
        let fullOd : TensorDims := { c := vp.outShape.c, h := vp.outShape.h, w := vp.outShape.w }
        let mut concatNodes : Array ValidatedNode := Array.mkEmpty vp.copies.size
        for ci in [:vp.copies.size] do
          let cp := vp.copies[ci]!
          let tIdx := repIdx + ci
          let sliceD := inDims[ci]!
          concatNodes := concatNodes.push {
            id := s!"{nd.id}_c{ci}"
            logicalId := nd.id
            nodeIdx := tIdx
            origNodeIdx := tIdx
            inputs := #[inRefs[ci]!]
            op := PlannedOp.neView cp
            outDims := sliceD
            groupOutDims := fullOd
            outByteOffset := cp.outOff
            sharedOutGroup := some repIdx
            isFinalGroup := isLastLogical
            outPlacement := nd.outPlacement
            engine := Engine.ne
            isNeCompute := false
            usesKernel := false
            opKind := "concat"
          }
        pure (concatNodes, fullOd)
      | .matmul heads transposeA transposeB scaleBits =>
        if inRefs.size != 2 then
          throw s!"node '{nd.id}' (matmul): expected 2 inputs [a, b], got {inRefs.size}"
        if heads == 0 then
          throw s!"node '{nd.id}' (matmul): heads must be >= 1"
        let da := inDims[0]!
        let db := inDims[1]!
        if da.h != 1 || db.h != 1 then
          throw s!"node '{nd.id}' (matmul): requires H=1 for both inputs (got A.h={da.h}, B.h={db.h})"
        if da.c % heads != 0 then
          throw s!"node '{nd.id}' (matmul): A channels C={da.c} not divisible by heads={heads}"
        if db.c % heads != 0 then
          throw s!"node '{nd.id}' (matmul): B channels C={db.c} not divisible by heads={heads}"
        let (m, ka) := if transposeA then (da.w, da.c / heads) else (da.c / heads, da.w)
        let (kb, n) := if transposeB then (db.w, db.c / heads) else (db.c / heads, db.w)
        if ka != kb then
          throw s!"node '{nd.id}' (matmul): inner dimension mismatch A.K={ka} != B.K={kb}"
        let k := ka
        validateMatmulSpec {
          b := heads, m := m, k := k, n := n,
          transposeA := transposeA, transposeB := transposeB, scaleBits := scaleBits
        }
        let mut mmNodes : Array ValidatedNode := #[]
        let mut effRefA := inRefs[0]!
        let mut effRefB := inRefs[1]!
        if transposeA then
          let (cpA, szA) ← planMatmulScratchTranspose heads k m
          let trIdx := repIdx + mmNodes.size
          let trDims : TensorDims := { c := heads * m, h := 1, w := k }
          mmNodes := mmNodes.push {
            id := s!"{nd.id}_trA"
            logicalId := s!"{nd.id}_trA"
            nodeIdx := trIdx
            origNodeIdx := trIdx
            inputs := #[inRefs[0]!]
            op := PlannedOp.neView cpA
            outDims := trDims
            groupOutDims := trDims
            outByteOffset := 0
            minDramBytes := szA
            sharedOutGroup := none
            isFinalGroup := false
            outPlacement := some .dram
            engine := Engine.ne
            isNeCompute := false
            usesKernel := false
            opKind := "matmul_trA"
          }
          effRefA := .node trIdx
        if transposeB then
          let (cpB, szB) ← planMatmulScratchTranspose heads n k
          let trIdx := repIdx + mmNodes.size
          let trDims : TensorDims := { c := heads * k, h := 1, w := n }
          mmNodes := mmNodes.push {
            id := s!"{nd.id}_trB"
            logicalId := s!"{nd.id}_trB"
            nodeIdx := trIdx
            origNodeIdx := trIdx
            inputs := #[inRefs[1]!]
            op := PlannedOp.neView cpB
            outDims := trDims
            groupOutDims := trDims
            outByteOffset := 0
            minDramBytes := szB
            sharedOutGroup := none
            isFinalGroup := false
            outPlacement := some .dram
            engine := Engine.ne
            isNeCompute := false
            usesKernel := false
            opKind := "matmul_trB"
          }
          effRefB := .node trIdx
        let mmRepIdx := repIdx + mmNodes.size
        let numSlices := ceilDiv m 16
        let fullOd : TensorDims := { c := heads * m, h := 1, w := n }
        let rowBN := alignUp (2 * n) dmaLineBytes
        for si in [:numSlices] do
          let mmIdx := mmRepIdx + si
          if mmIdx >= 255 then
            throw s!"node '{nd.id}' (matmul slice {si}, taskId={mmIdx}) exceeds kernelStage 8-bit limit 254"
          let m0 := si * 16
          let mi := min 16 (m - m0)
          mmNodes := mmNodes.push {
            id := if numSlices == 1 then nd.id else s!"{nd.id}_s{si}"
            logicalId := nd.id
            nodeIdx := mmIdx
            origNodeIdx := mmIdx
            inputs := #[effRefA, effRefB]
            op := PlannedOp.neMatmul heads m0 mi m k n scaleBits effRefA effRefB
            outDims := { c := heads * mi, h := 1, w := n }
            groupOutDims := fullOd
            outByteOffset := m0 * rowBN
            sharedOutGroup := if numSlices > 1 then some mmRepIdx else none
            isFinalGroup := isLastLogical
            outPlacement := nd.outPlacement
            engine := Engine.ne
            isNeCompute := true
            usesKernel := false
            opKind := "matmul"
          }
        pure (mmNodes, fullOd)

    for en in expandedNodes do
      let tIdx := vNodes.size
      for ref in en.inputs do
        match ref with
        | .node prodIdx =>
          lastConsumer := lastConsumer.set! prodIdx (max lastConsumer[prodIdx]! tIdx)
        | .input _ => pure ()
      vNodes := vNodes.push en
      lastConsumer := lastConsumer.push 0

    let endIdx := vNodes.size - 1
    let outRefIdx := match nd.op with
      | .matmul .. =>
        match expandedNodes.findIdx? (fun en => match en.op with | .neMatmul .. => true | _ => false) with
        | some offIdx => repIdx + offIdx
        | none => repIdx
      | _ => repIdx
    logicalReps := logicalReps.push (outRefIdx, endIdx, isLastLogical)
    nameMap := nameMap.push (nd.id, .node outRefIdx, fullOutDims)

  if vNodes.size > 256 then
    throw s!"expanded graph has {vNodes.size} tasks, exceeding hardware limit 256"

  -- Ensure every non-final logical node is consumed at least once
  for (repIdx, endIdx, isFinal) in logicalReps do
    if !isFinal && lastConsumer[repIdx]! <= endIdx then
      throw s!"intermediate node '{vNodes[repIdx]!.logicalId}' (task {repIdx}) is never consumed by any later node"

  pure (shortNames, vNodes, lastConsumer)

-- ============================================================================
-- L2 Eligibility, Placement Planner & Auto-Spill Loop
-- ============================================================================

private def isNodeL2Eligible (vNodes : Array ValidatedNode) (i : Nat) : Bool := Id.run do
  let nd := vNodes[i]!
  if nd.sharedOutGroup.isSome then
    return false
  if let .neView _ := nd.op then
    return false
  if let .neMatmul heads _ _ totalM _ _ _ _ _ := nd.op then
    if heads > 1 || totalM > 16 then
      return false
  if l2ResidentBytes nd.outDims >= l2CapacityBytes then
    return false
  for j in [i + 1 : vNodes.size] do
    let cons := vNodes[j]!
    match cons.op with
    | .conv sh _ _ _ _ _ =>
      if cons.inputs[0]! == .node i && sh.kernelSize == 3 && sh.stride == 2 then
        if sh.winReg != sh.w || sh.hinReg != sh.h then
          return false
    | .pePool _ _ inDims gh _ | .nePool _ inDims gh _ =>
      if cons.inputs[0]! == .node i && gh.used != inDims.h then
        return false
    | .neView _ =>
      if cons.inputs.any (fun r => r == .node i) then
        return false
    | .neMatmul heads _ _ _ _ _ _ refA refB =>
      if refA == .node i || (refB == .node i && heads > 1) then
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
      | .lut act dims _ =>
        let srcIsL2 := isNodeL2 nd.inputs[0]!
        let srcFloor32 := act == .copy && vNodes.size > 1
        TaskPlanKind.lut <$> chooseNeActTilingForGraph target dims.c dims.h dims.w srcIsL2 resIsL2 liveRanges srcFloor32
      | .pePool _ _ dims gh gw =>
        let srcIsL2 := isNodeL2 nd.inputs[0]!
        TaskPlanKind.pePool <$> choosePePoolTilingForL2 target dims.c gh gw srcIsL2 resIsL2 liveRanges
      | .nePool _ dims gh gw =>
        let srcIsL2 := isNodeL2 nd.inputs[0]!
        TaskPlanKind.nePool <$> chooseNePoolTilingForL2 target dims.c gh gw srcIsL2 resIsL2 liveRanges
      | .peReduce _ axes _ dims =>
        let srcIsL2 := isNodeL2 nd.inputs[0]!
        let ax := peReduceAxes dims.c dims.h axes.reduceC axes.reduceH
        TaskPlanKind.peReduce <$> choosePeReduceTilingForL2 target dims.w ax srcIsL2 resIsL2 liveRanges
      | .neChannelReduce _ dims =>
        let srcIsL2 := isNodeL2 nd.inputs[0]!
        TaskPlanKind.neChannelReduce <$> chooseNeChannelReduceTilingForL2 target dims srcIsL2 resIsL2 liveRanges
      | .neView cp =>
        TaskPlanKind.neView <$> chooseViewTaskPlan target cp liveRanges
      | .neMatmul _ _ m _ k n _ _ refB =>
        let srcIsL2 := isNodeL2 refB
        TaskPlanKind.neMatmul <$> chooseMatmulTaskPlan target m k n srcIsL2 resIsL2 liveRanges
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
  kernLoc? : Option TensorLocation := none
  extProducerTask? : Option Nat := none
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

private structure ConvKernCacheKey where
  origNodeIdx : Nat
  logNE : Nat
  f : Nat
  fat : Nat
  ocg : Nat
  deriving BEq, DecidableEq

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
  let mut convKernCache : Array (ConvKernCacheKey × Nat × PackedKernel) := #[]
  let mut lutKernCache : Array (Nat × Nat) := #[]
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
    let mut kernLoc? : Option TensorLocation := none
    let mut extProd? : Option Nat := none

    match nd.op, plan with
    | .conv sh weights bias _ lutPayload? _, .conv cp =>
      -- Pack weights/bias/LUT into shared __KERN_0, deduplicating identical per-node payloads across strips
      let cKey : ConvKernCacheKey := {
        origNodeIdx := nd.origNodeIdx
        logNE := cp.tiling.logNE
        f := cp.tiling.f
        fat := cp.tiling.fat
        ocg := cp.tiling.ocg
      }
      let (kOff, packed) ← match convKernCache.find? (fun (k, _, _) => k == cKey) with
        | some (_, cachedOff, cachedPacked) => pure (cachedOff, cachedPacked)
        | none =>
          kernBuf := padToAlign kernBuf dmaLineBytes
          let newOff := kernBuf.size
          let p ← packConvKernelChunks spec.target sh cp.tiling weights bias lutPayload?
          kernBuf := kernBuf ++ p.bytes
          convKernCache := convKernCache.push (cKey, newOff, p)
          pure (newOff, p)
      kernBaseOff := kOff
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
        let (baseOff, _, _, ext) := surfaceDmaWindow nd.src1Win sh.cin sh.h sh.w
        dramAccs := dramAccs.push { resource := .surface idx shortNames[idx]!, lo := baseOff, hi := baseOff + ext, isWrite := false, port := .src1Dma }

      match resLoc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dram off _ =>
        let sz :=
          if cp.direct then ((1 : Nat) <<< (cp.tiling.f + effOcgConv sh cp.tiling + cp.tiling.fat)) * 512
          else cp.tiling.tileHeight * resRowStrideConv sh cp.tiling
        l2Accs := l2Accs.push { lo := cp.resStageBase, hi := cp.resStageBase + sz, isWrite := true, role := .resStage }
        let span := dramSpanBytes nd.outDims
        let actOff := off + nd.outByteOffset
        dramAccs := dramAccs.push { resource := .scratch, lo := actOff, hi := actOff + span, isWrite := true, port := .resDma }
      | .surface _ _ =>
        let sz :=
          if cp.direct then ((1 : Nat) <<< (cp.tiling.f + effOcgConv sh cp.tiling + cp.tiling.fat)) * 512
          else cp.tiling.tileHeight * resRowStrideConv sh cp.tiling
        l2Accs := l2Accs.push { lo := cp.resStageBase, hi := cp.resStageBase + sz, isWrite := true, role := .resStage }
        let (baseOff, _, _, ext) := surfaceDmaWindow nd.resWin nd.outDims.c nd.outDims.h nd.outDims.w
        let actOff := baseOff + nd.outByteOffset
        dramAccs := dramAccs.push { resource := .surface outSurfIdx "y", lo := actOff, hi := actOff + ext, isWrite := true, port := .resDma }

    | .pe1 _ dims, .pe pp =>
      let stripW := peStripWidth pp.tiling.k
      let effStripW := max stripW 32
      match src1Loc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dram off _ =>
        let row1 := 2 * min (alignUp dims.w 8) effStripW
        let sz1 := (if dims.h > 1 then pp.tiling.tileHeight else 1) * (dims.c * row1)
        l2Accs := l2Accs.push { lo := pp.src1StageBase, hi := pp.src1StageBase + sz1, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanBytes dims, isWrite := false, port := .src1Dma }
      | .surface idx _ =>
        let row1 := 2 * min (alignUp dims.w 8) effStripW
        let sz1 := (if dims.h > 1 then pp.tiling.tileHeight else 1) * (dims.c * row1)
        l2Accs := l2Accs.push { lo := pp.src1StageBase, hi := pp.src1StageBase + sz1, isWrite := true, role := .src1Stage }
        let (baseOff, _, _, ext) := surfaceDmaWindow nd.src1Win dims.c dims.h dims.w
        dramAccs := dramAccs.push { resource := .surface idx shortNames[idx]!, lo := baseOff, hi := baseOff + ext, isWrite := false, port := .src1Dma }

      match resLoc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dram off _ =>
        let drow := 2 * min (alignUp dims.w 8) effStripW
        let szRes := if pp.direct then peDirectResBytes dims.c drow pp.tiling.tileHeight else pp.tiling.tileHeight * (dims.c * drow)
        l2Accs := l2Accs.push { lo := pp.resStageBase, hi := pp.resStageBase + szRes, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanBytes dims, isWrite := true, port := .resDma }
      | .surface _ _ =>
        let drow := 2 * min (alignUp dims.w 8) effStripW
        let szRes := if pp.direct then peDirectResBytes dims.c drow pp.tiling.tileHeight else pp.tiling.tileHeight * (dims.c * drow)
        l2Accs := l2Accs.push { lo := pp.resStageBase, hi := pp.resStageBase + szRes, isWrite := true, role := .resStage }
        let (baseOff, _, _, ext) := surfaceDmaWindow nd.resWin dims.c dims.h dims.w
        dramAccs := dramAccs.push { resource := .surface outSurfIdx "y", lo := baseOff, hi := baseOff + ext, isWrite := true, port := .resDma }

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
      | .dram off _ =>
        let whole1 := hw1.c == 1 && hw1.h == 1 && hw1.w > 1 && outDims.h > 1 && th < outDims.h
        let row1 := 2 * (if whole1 then alignUp hw1.w 8 else min (alignUp hw1.w 8) effStripW)
        let sz1 := (if hw1.h > 1 then th else 1) * (hw1.c * row1)
        l2Accs := l2Accs.push { lo := pp.src1StageBase, hi := pp.src1StageBase + sz1, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanBytes hw1, isWrite := false, port := .src1Dma }
      | .surface idx _ =>
        let whole1 := hw1.c == 1 && hw1.h == 1 && hw1.w > 1 && outDims.h > 1 && th < outDims.h
        let row1 := 2 * (if whole1 then alignUp hw1.w 8 else min (alignUp hw1.w 8) effStripW)
        let sz1 := (if hw1.h > 1 then th else 1) * (hw1.c * row1)
        l2Accs := l2Accs.push { lo := pp.src1StageBase, hi := pp.src1StageBase + sz1, isWrite := true, role := .src1Stage }
        let (baseOff, _, _, ext) := surfaceDmaWindow nd.src1Win hw1.c hw1.h hw1.w
        dramAccs := dramAccs.push { resource := .surface idx shortNames[idx]!, lo := baseOff, hi := baseOff + ext, isWrite := false, port := .src1Dma }

      let s2Loc := src2Loc?.get!
      match s2Loc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src2L2 }
      | .dram off _ =>
        let whole2 := hw2.c == 1 && hw2.h == 1 && hw2.w > 1 && outDims.h > 1 && th < outDims.h
        let row2 := 2 * (if whole2 then alignUp hw2.w 8 else min (alignUp hw2.w 8) effStripW)
        let sz2 := (if hw2.h > 1 then th else 1) * (hw2.c * row2)
        l2Accs := l2Accs.push { lo := pp.src2StageBase, hi := pp.src2StageBase + sz2, isWrite := true, role := .src2Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanBytes hw2, isWrite := false, port := .src2Dma }
      | .surface idx _ =>
        let whole2 := hw2.c == 1 && hw2.h == 1 && hw2.w > 1 && outDims.h > 1 && th < outDims.h
        let row2 := 2 * (if whole2 then alignUp hw2.w 8 else min (alignUp hw2.w 8) effStripW)
        let sz2 := (if hw2.h > 1 then th else 1) * (hw2.c * row2)
        l2Accs := l2Accs.push { lo := pp.src2StageBase, hi := pp.src2StageBase + sz2, isWrite := true, role := .src2Stage }
        let (baseOff, _, _, ext) := surfaceDmaWindow nd.src2Win hw2.c hw2.h hw2.w
        dramAccs := dramAccs.push { resource := .surface idx shortNames[idx]!, lo := baseOff, hi := baseOff + ext, isWrite := false, port := .src2Dma }

      match resLoc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dram off _ =>
        let drow := 2 * min (alignUp outDims.w 8) effStripW
        let szRes := if pp.direct then peDirectResBytes outDims.c drow th else th * (outDims.c * drow)
        l2Accs := l2Accs.push { lo := pp.resStageBase, hi := pp.resStageBase + szRes, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanBytes outDims, isWrite := true, port := .resDma }
      | .surface _ _ =>
        let drow := 2 * min (alignUp outDims.w 8) effStripW
        let szRes := if pp.direct then peDirectResBytes outDims.c drow th else th * (outDims.c * drow)
        l2Accs := l2Accs.push { lo := pp.resStageBase, hi := pp.resStageBase + szRes, isWrite := true, role := .resStage }
        let (baseOff, _, _, ext) := surfaceDmaWindow nd.resWin outDims.c outDims.h outDims.w
        dramAccs := dramAccs.push { resource := .surface outSurfIdx "y", lo := baseOff, hi := baseOff + ext, isWrite := true, port := .resDma }

    | .lut act dims lutPayload?, .lut lp =>
      if let some payload := lutPayload? then
        let lutOff ← match lutKernCache.find? (fun (nid, _) => nid == nd.origNodeIdx) with
          | some (_, cachedOff) => pure cachedOff
          | none =>
            kernBuf := padToAlign kernBuf dmaLineBytes
            let newOff := kernBuf.size
            kernBuf := kernBuf ++ payload.bytes
            lutKernCache := lutKernCache.push (nd.origNodeIdx, newOff)
            pure newOff
        kernBaseOff := lutOff
        dramAccs := dramAccs.push {
          resource := .kernel, lo := kernBaseOff, hi := kernBaseOff + lutTableBytes, isWrite := false, port := .lut
        }
      let stripW := lp.tiling.stripWidth
      let effSrcStrip := if act == .copy && n > 1 then max stripW 32 else stripW
      match src1Loc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dram off _ =>
        let row := 2 * min (alignUp dims.w 8) effSrcStrip
        let sz := lp.tiling.tileHeight * (dims.c * row)
        l2Accs := l2Accs.push { lo := lp.srcStageBase, hi := lp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanBytes dims, isWrite := false, port := .src1Dma }
      | .surface idx _ =>
        let row := 2 * min (alignUp dims.w 8) effSrcStrip
        let sz := lp.tiling.tileHeight * (dims.c * row)
        l2Accs := l2Accs.push { lo := lp.srcStageBase, hi := lp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        let (baseOff, _, _, ext) := surfaceDmaWindow nd.src1Win dims.c dims.h dims.w
        dramAccs := dramAccs.push { resource := .surface idx shortNames[idx]!, lo := baseOff, hi := baseOff + ext, isWrite := false, port := .src1Dma }

      match resLoc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dram off _ =>
        let drow := 2 * min (alignUp dims.w 8) (max stripW 32)
        let szRes := if lp.direct then neActDirectResBytes lp.tiling.f lp.tiling.fat else lp.tiling.tileHeight * (dims.c * drow)
        l2Accs := l2Accs.push { lo := lp.resStageBase, hi := lp.resStageBase + szRes, isWrite := true, role := .resStage }
        let span := dramSpanBytes dims
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + span, isWrite := true, port := .resDma, coverage := .dense off (off + span) }
      | .surface _ _ =>
        let drow := 2 * min (alignUp dims.w 8) (max stripW 32)
        let szRes := if lp.direct then neActDirectResBytes lp.tiling.f lp.tiling.fat else lp.tiling.tileHeight * (dims.c * drow)
        l2Accs := l2Accs.push { lo := lp.resStageBase, hi := lp.resStageBase + szRes, isWrite := true, role := .resStage }
        let (baseOff, _, _, ext) := surfaceDmaWindow nd.resWin dims.c dims.h dims.w
        dramAccs := dramAccs.push { resource := .surface outSurfIdx "y", lo := baseOff, hi := baseOff + ext, isWrite := true, port := .resDma }

    | .pePool _ _ dims gh gw, .pePool pp =>
      let stripW := (1 : Nat) <<< pp.k
      let px := poolStripPx gw.used gw.k gw.sEff gw.padBefore stripW gw.out
      let row := 2 * px
      let u := dims.c * row
      match src1Loc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dram off _ =>
        let sz := alignUp (poolInRows gh.used gh.k gh.sEff gh.padBefore gh.out pp.tileHeight false * u) dmaLineBytes
        l2Accs := l2Accs.push { lo := pp.srcStageBase, hi := pp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanBytes dims, isWrite := false, port := .src1Dma }
      | .surface idx _ =>
        let sz := alignUp (poolInRows gh.used gh.k gh.sEff gh.padBefore gh.out pp.tileHeight false * u) dmaLineBytes
        l2Accs := l2Accs.push { lo := pp.srcStageBase, hi := pp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .surface idx shortNames[idx]!, lo := 0, hi := dramSpanBytes dims, isWrite := false, port := .src1Dma }

      let drow := 2 * min (alignUp gw.out 8) (max stripW 32)
      let resSz := if pp.direct then peDirectResBytes dims.c drow pp.tileHeight else alignUp (pp.tileHeight * dims.c * drow) dmaLineBytes
      match resLoc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dram off _ =>
        l2Accs := l2Accs.push { lo := pp.resStageBase, hi := pp.resStageBase + resSz, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanBytes nd.outDims, isWrite := true, port := .resDma }
      | .surface _ _ =>
        l2Accs := l2Accs.push { lo := pp.resStageBase, hi := pp.resStageBase + resSz, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .surface outSurfIdx "y", lo := 0, hi := dramSpanBytes nd.outDims, isWrite := true, port := .resDma }

    | .peReduce _ axes _ dims, .peReduce pr =>
      let ax := peReduceAxes dims.c dims.h axes.reduceC axes.reduceH
      let stripW := (1 : Nat) <<< pr.k
      let px := min (alignUp dims.w 8) (max stripW 32)
      let u := ax.sC * ax.sD * 2 * px
      match src1Loc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dram off _ =>
        let sz := alignUp (pr.tileHeight * u) dmaLineBytes
        l2Accs := l2Accs.push { lo := pr.srcStageBase, hi := pr.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanBytes dims, isWrite := false, port := .src1Dma }
      | .surface idx _ =>
        let sz := alignUp (pr.tileHeight * u) dmaLineBytes
        l2Accs := l2Accs.push { lo := pr.srcStageBase, hi := pr.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .surface idx shortNames[idx]!, lo := 0, hi := dramSpanBytes dims, isWrite := false, port := .src1Dma }

      match resLoc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dram off _ =>
        l2Accs := l2Accs.push { lo := pr.resStageBase, hi := pr.resStageBase + 2048, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanBytes nd.outDims, isWrite := true, port := .resDma }
      | .surface _ _ =>
        l2Accs := l2Accs.push { lo := pr.resStageBase, hi := pr.resStageBase + 2048, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .surface outSurfIdx "y", lo := 0, hi := dramSpanBytes nd.outDims, isWrite := true, port := .resDma }

    | .nePool _ dims gh gw, .nePool np =>
      let stripW := np.stripWidth
      let px := poolStripPx gw.used gw.k gw.sEff gw.padBefore stripW gw.out
      let u := dims.c * 2 * px
      match src1Loc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dram off _ =>
        let sz := alignUp (poolInRows gh.used gh.k gh.sEff gh.padBefore gh.out np.tileHeight true * u) dmaLineBytes
        l2Accs := l2Accs.push { lo := np.srcStageBase, hi := np.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanBytes dims, isWrite := false, port := .src1Dma }
      | .surface idx _ =>
        let sz := alignUp (poolInRows gh.used gh.k gh.sEff gh.padBefore gh.out np.tileHeight true * u) dmaLineBytes
        l2Accs := l2Accs.push { lo := np.srcStageBase, hi := np.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .surface idx shortNames[idx]!, lo := 0, hi := dramSpanBytes dims, isWrite := false, port := .src1Dma }

      let drow := 2 * min (alignUp gw.out 8) (max stripW 32)
      let resSz := if np.direct then neActDirectResBytes np.f np.fat else alignUp (np.tileHeight * dims.c * drow) dmaLineBytes
      match resLoc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dram off _ =>
        l2Accs := l2Accs.push { lo := np.resStageBase, hi := np.resStageBase + resSz, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanBytes nd.outDims, isWrite := true, port := .resDma }
      | .surface _ _ =>
        l2Accs := l2Accs.push { lo := np.resStageBase, hi := np.resStageBase + resSz, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .surface outSurfIdx "y", lo := 0, hi := dramSpanBytes nd.outDims, isWrite := true, port := .resDma }

    | .neChannelReduce _ dims, .neChannelReduce nc =>
      let stripW := nc.stripWidth
      let px := min (alignUp dims.w 8) stripW
      let u := dims.c * 2 * px
      match src1Loc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dram off _ =>
        let sz := alignUp (nc.tileHeight * u) dmaLineBytes
        l2Accs := l2Accs.push { lo := nc.srcStageBase, hi := nc.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanBytes dims, isWrite := false, port := .src1Dma }
      | .surface idx _ =>
        let sz := alignUp (nc.tileHeight * u) dmaLineBytes
        l2Accs := l2Accs.push { lo := nc.srcStageBase, hi := nc.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .surface idx shortNames[idx]!, lo := 0, hi := dramSpanBytes dims, isWrite := false, port := .src1Dma }

      let drow := 2 * min (alignUp dims.w 8) (max stripW 32)
      let resSz := if nc.direct then neActDirectResBytes nc.f nc.fat else alignUp (nc.tileHeight * drow) dmaLineBytes
      match resLoc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dram off _ =>
        l2Accs := l2Accs.push { lo := nc.resStageBase, hi := nc.resStageBase + resSz, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanBytes nd.outDims, isWrite := true, port := .resDma }
      | .surface _ _ =>
        l2Accs := l2Accs.push { lo := nc.resStageBase, hi := nc.resStageBase + resSz, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .surface outSurfIdx "y", lo := 0, hi := dramSpanBytes nd.outDims, isWrite := true, port := .resDma }

    | .neView cp, .neView vp =>
      let srcSz := viewSrcL2Bytes cp vp.tiling
      let resSz := viewResL2Bytes spec.target cp vp.tiling
      l2Accs := l2Accs.push { lo := vp.srcStageBase, hi := vp.srcStageBase + srcSz, isWrite := true, role := .src1Stage }
      l2Accs := l2Accs.push { lo := vp.resStageBase, hi := vp.resStageBase + resSz, isWrite := true, role := .resStage }
      let rawInExt := 2 * cp.w + (cp.h - 1) * cp.inRowStride + (cp.c - 1) * cp.inPlaneStride +
        (cp.d - 1) * cp.inDStride + (cp.n - 1) * cp.inNStride
      match src1Loc with
      | .dram off _ =>
        let actInOff := off + cp.inOff
        let actInEnd := alignUp (actInOff + rawInExt) dmaLineBytes
        dramAccs := dramAccs.push { resource := .scratch, lo := actInOff, hi := actInEnd, isWrite := false, port := .src1Dma }
      | .surface idx _ =>
        let actInOff := cp.inOff
        let actInEnd := alignUp (actInOff + rawInExt) dmaLineBytes
        dramAccs := dramAccs.push { resource := .surface idx shortNames[idx]!, lo := actInOff, hi := actInEnd, isWrite := false, port := .src1Dma }
      | .l2 _ _ =>
        throw s!"internal error: neView node '{nd.id}' received L2-resident input"
      let (rw, rch) := if cp.transposeCW then (cp.c, cp.w) else (cp.w, cp.c)
      let rawOutExt := 2 * rw + (cp.h - 1) * cp.outRowStride + (rch - 1) * cp.outPlaneStride +
        (cp.d - 1) * cp.outDStride + (cp.n - 1) * cp.outNStride
      match resLoc with
      | .dram off _ =>
        let actOutOff := off + cp.outOff
        let actOutEnd := alignUp (actOutOff + rawOutExt) dmaLineBytes
        let cov := classifyStridedWriteCoverage cp.n cp.d rch cp.h rw actOutOff cp.outNStride cp.outDStride cp.outPlaneStride cp.outRowStride
        dramAccs := dramAccs.push { resource := .scratch, lo := actOutOff, hi := actOutEnd, isWrite := true, port := .resDma, coverage := cov }
      | .surface _ _ =>
        let actOutOff := cp.outOff
        let actOutEnd := alignUp (actOutOff + rawOutExt) dmaLineBytes
        dramAccs := dramAccs.push { resource := .surface outSurfIdx "y", lo := actOutOff, hi := actOutEnd, isWrite := true, port := .resDma }
      | .l2 _ _ =>
        throw s!"internal error: neView node '{nd.id}' received L2-resident output"

    | .neMatmul heads m0 m totalM k n _ refA refB, .neMatmul mp =>
      let aLoc := resolveRefLoc refA
      let bLoc := resolveRefLoc refB
      src1Loc := bLoc
      src1Prod? := resolveRefProd refB
      kernLoc? := some aLoc
      let chunkSize := matmulChunkSize k
      let kBaseOff := m0 * chunkSize
      let kBatchSt := totalM * chunkSize
      match aLoc with
      | .surface idx _ =>
        if heads > 1 && totalM > m then
          for bIdx in [:heads] do
            let lo := kBaseOff + bIdx * kBatchSt
            dramAccs := dramAccs.push {
              resource := .surface idx shortNames[idx]!, lo := lo, hi := lo + m * chunkSize, isWrite := false, port := .kernel
            }
        else
          dramAccs := dramAccs.push {
            resource := .surface idx shortNames[idx]!, lo := kBaseOff, hi := kBaseOff + heads * m * chunkSize, isWrite := false, port := .kernel
          }
      | .dram off _ =>
        let actKOff := off + kBaseOff
        if heads > 1 && totalM > m then
          for bIdx in [:heads] do
            let lo := actKOff + bIdx * kBatchSt
            dramAccs := dramAccs.push {
              resource := .scratch, lo := lo, hi := lo + m * chunkSize, isWrite := false, port := .kernel
            }
        else
          dramAccs := dramAccs.push {
            resource := .scratch, lo := actKOff, hi := actKOff + heads * m * chunkSize, isWrite := false, port := .kernel
          }
        let mut latestW? : Option Nat := none
        for j in [:out.size] do
          for a in out[j]!.dramAccesses do
            if a.resource == .scratch && a.isWrite then
              if dramAccs.any (fun kb => kb.port == .kernel && a.lo < kb.hi && kb.lo < a.hi) then
                latestW? := some j
        match latestW? with
        | some p => extProd? := some p
        | none => throw s!"internal error: matmul node '{nd.id}' scratch kernel at {toHex actKOff} has no prior writer"
      | .l2 _ _ =>
        throw s!"internal error: matmul node '{nd.id}' kernel operand A cannot be L2-resident"

      let stripW := (128 : Nat) >>> mp.tiling.shPref
      let rowBN := alignUp (2 * n) dmaLineBytes
      match bLoc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dram off _ =>
        let chs := 2 * min (alignUp n 8) (max stripW 32)
        let sz := k * chs
        l2Accs := l2Accs.push { lo := mp.srcStageBase, hi := mp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        let bSpan := heads * k * rowBN
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + bSpan, isWrite := false, port := .src1Dma }
      | .surface idx _ =>
        let chs := 2 * min (alignUp n 8) (max stripW 32)
        let sz := k * chs
        l2Accs := l2Accs.push { lo := mp.srcStageBase, hi := mp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        let bSpan := heads * k * rowBN
        dramAccs := dramAccs.push { resource := .surface idx shortNames[idx]!, lo := 0, hi := bSpan, isWrite := false, port := .src1Dma }

      let yBaseOff := m0 * rowBN
      let yBatchSt := totalM * rowBN
      match resLoc with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dram off _ =>
        let dchs := 2 * min (alignUp n 8) (max stripW 32)
        let szRes := if mp.direct then ((1 : Nat) <<< mp.tiling.f) * 512 else m * dchs
        l2Accs := l2Accs.push { lo := mp.resStageBase, hi := mp.resStageBase + szRes, isWrite := true, role := .resStage }
        let actYOff := off + yBaseOff
        if heads > 1 && totalM > m then
          for bIdx in [:heads] do
            let lo := actYOff + bIdx * yBatchSt
            let hi := lo + m * rowBN
            dramAccs := dramAccs.push { resource := .scratch, lo := lo, hi := hi, isWrite := true, port := .resDma, coverage := .dense lo hi }
        else
          let ySpan := heads * m * rowBN
          dramAccs := dramAccs.push { resource := .scratch, lo := actYOff, hi := actYOff + ySpan, isWrite := true, port := .resDma, coverage := .dense actYOff (actYOff + ySpan) }
      | .surface _ _ =>
        let dchs := 2 * min (alignUp n 8) (max stripW 32)
        let szRes := if mp.direct then ((1 : Nat) <<< mp.tiling.f) * 512 else m * dchs
        l2Accs := l2Accs.push { lo := mp.resStageBase, hi := mp.resStageBase + szRes, isWrite := true, role := .resStage }
        if heads > 1 && totalM > m then
          for bIdx in [:heads] do
            let lo := yBaseOff + bIdx * yBatchSt
            let hi := lo + m * rowBN
            dramAccs := dramAccs.push { resource := .surface outSurfIdx "y", lo := lo, hi := hi, isWrite := true, port := .resDma, coverage := .dense lo hi }
        else
          let ySpan := heads * m * rowBN
          dramAccs := dramAccs.push { resource := .surface outSurfIdx "y", lo := yBaseOff, hi := yBaseOff + ySpan, isWrite := true, port := .resDma, coverage := .dense yBaseOff (yBaseOff + ySpan) }

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
      kernLoc? := kernLoc?
      extProducerTask? := extProd?
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

private def isMultiBatchOp (op : PlannedOp) : Bool :=
  match op with
  | .peReduce _ axes _ dims => (peReduceAxes dims.c dims.h axes.reduceC axes.reduceH).sN > 1
  | .neView cp => cp.n > 1
  | .neMatmul heads .. => heads > 1
  | _ => false

private def latestScratchWriterBefore (pre : Array PreLoweredTask) (i lo hi : Nat) : Option Nat := Id.run do
  let mut latestW? : Option Nat := none
  for j in [:i] do
    for a in pre[j]!.dramAccesses do
      if a.resource == .scratch && a.isWrite && intervalsOverlap a.lo a.hi lo hi then
        latestW? := some j
  return latestW?

private def anyActiveMultiBatchScratchWriterBefore (pre : Array PreLoweredTask) (i lo hi : Nat) : Bool := Id.run do
  let mut writers : Array (Nat × VerifiedDramAccess) := #[]
  for j in [:i] do
    for a in pre[j]!.dramAccesses do
      if a.resource == .scratch && a.isWrite && intervalsOverlap a.lo a.hi lo hi then
        writers := writers.push (j, a)
  for idx in [:writers.size] do
    let (j, a) := writers[idx]!
    if isMultiBatchOp pre[j]!.node.op then
      let ovLo := max a.lo lo
      let ovHi := min a.hi hi
      let mut laterCovs : List WriteCoverage := []
      for idx2 in [idx + 1 : writers.size] do
        let (j2, a2) := writers[idx2]!
        if j2 > j then
          let cov2 := match a2.coverage with
            | .sparse =>
              match pre[j2]!.node.op with
              | .neView _ => .sparse
              | _ => .dense a2.lo a2.hi
            | other => other
          laterCovs := cov2 :: laterCovs
      let laterIvs := effectiveCoveredIntervals laterCovs
      if !intervalsCover laterIvs ovLo ovHi then
        return true
  return false

private def computeSynchronization (pre : Array PreLoweredTask) : Array TaskSyncInfo := Id.run do
  let n := pre.size
  let mut syncs : Array TaskSyncInfo := Array.ofFn (n := n) fun i => {
    hasSrc1Dma := pre[i.val]!.hasSrc1Dma
    hasSrc2Dma := pre[i.val]!.hasSrc2Dma
  }

  -- Seed L2 handoff bits on non-broadcast L2-resident src1 reads from immediately latest producer;
  -- for L2-resident PE2 broadcast src1 reads, use whole-task wait (bit 22 for PE, bit 5 for NE).
  for i in [1 : n] do
    if let .l2 _ _ := pre[i]!.src1Loc then
      let isBcastSrc1 : Bool := match pre[i]!.node.op with
        | .pe2 _ _ outDims hw1 _ _ _ => hw1 != outDims
        | _ => false
      if let some prod := pre[i]!.src1Producer? then
        let prodEng := pre[prod]!.node.engine
        if isBcastSrc1 then
          if prodEng == .ne then
            syncs := syncs.set! i { syncs[i]! with src1WaitNe5 := true }
          else
            syncs := syncs.set! i { syncs[i]! with src1WaitPe22 := true }
        else if prodEng == .ne && latestEngineBefore pre .ne i == some prod then
          syncs := syncs.set! i { syncs[i]! with src1HandoffNe := true }
        else if prodEng == .pe && latestEngineBefore pre .pe i == some prod then
          syncs := syncs.set! i { syncs[i]! with src1HandoffPe := true }

  -- Seed BAR 3 DRAM-scratch read coherence (depKind = 1 if latest writer has BatchN > 1 else 2,
  -- depDist = min 15 (i - j - 1)) on every PE src1/src2 reading a scratch buffer written by an earlier task j < i.
  -- Never credit depDist > 0 as ordering: the fixpoint loop below adds HB / src2Wait5 / src2Wait22 whenever j < i - 1.
  for i in [1 : n] do
    if pre[i]!.node.engine == .pe then
      for b in pre[i]!.dramAccesses do
        if b.resource == .scratch && !b.isWrite then
          if let some j := latestScratchWriterBefore pre i b.lo b.hi then
            let depKind := if isMultiBatchOp pre[i]!.node.op || anyActiveMultiBatchScratchWriterBefore pre i b.lo b.hi then 1 else 2
            let dist := min 15 (i - j - 1)
            if b.port == .src1Dma then
              syncs := syncs.set! i { syncs[i]! with src1DepKind := depKind, src1DepDist := dist }
            else if b.port == .src2Dma then
              syncs := syncs.set! i { syncs[i]! with src2DepKind := depKind, src2DepDist := dist }

  let mut latestNeArr : Array (Option Nat) := Array.mkEmpty n
  let mut latestPeArr : Array (Option Nat) := Array.mkEmpty n
  let mut lastNe : Option Nat := none
  let mut lastPe : Option Nat := none
  for i in [:n] do
    latestNeArr := latestNeArr.push lastNe
    latestPeArr := latestPeArr.push lastPe
    match pre[i]!.node.engine with
    | .ne => lastNe := some i
    | .pe => lastPe := some i

  -- Fixpoint loop: converge whenever no new wait/dep bits are added (monotone, <= 8*n steps)
  let maxIters := 8 * n + 2
  for _ in [:maxIters] do
    let fps : Array VerifiedTaskFootprint := Array.ofFn (n := n) fun i => {
      taskIndex := i.val
      kind := match pre[i.val]!.node.op with
        | .conv .. => .neConv
        | .lut .. => .neAct
        | .pePool .. => .pePool
        | .peReduce .. => .peReduce
        | .nePool .. => .nePool
        | .neChannelReduce .. => .neChannelReduce
        | .neView .. => .neView
        | .neMatmul .. => .neMatmul
        | _ => .peOp
      engine := pre[i.val]!.node.engine
      isNeCompute := pre[i.val]!.node.isNeCompute
      isMultiBatch := isMultiBatchOp pre[i.val]!.node.op
      extProducerTask := pre[i.val]!.extProducerTask?
      l2Accesses := pre[i.val]!.l2Accesses
      dramAccesses := pre[i.val]!.dramAccesses
      sync := syncs[i.val]!
    }
    let (hbTbl, hblTbl) := buildHBReachTables fps
    let hb := fun a b => reachB hbTbl a b
    let hbl := fun a b => reachB hblTbl a b
    let hbEq := fun a b => a == b || hb a b
    let hblEq := fun a b => a == b || hbl a b

    let mut changed := false

    for i in [1 : n] do
      let ti := fps[i]!
      let latestNe := latestNeArr[i]!
      let latestPe := latestPeArr[i]!

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
          (if tk.engine == .ne then true
           else Id.run do
            let mut ok := false
            for j in [:i] do
              if pre[j]!.node.engine == .ne && hbEq k j then
                ok := true
                break
            return ok)

        -- 1. DRAM conflicts
        for a in tk.dramAccesses do
          for b in ti.dramAccesses do
            if a.resource == b.resource && (a.isWrite || b.isWrite) && intervalsOverlap a.lo a.hi b.lo b.hi then
              if b.port == .kernel then
                match ti.extProducerTask with
                | some p =>
                  if !hbEq k p then
                    if tk.engine == .ne then
                      if !syncs[p]!.src1WaitNe5 then
                        syncs := syncs.set! p { syncs[p]! with src1WaitNe5 := true, src1HandoffNe := false }
                        changed := true
                    else
                      if !syncs[p]!.src1WaitPe22 then
                        syncs := syncs.set! p { syncs[p]! with src1WaitPe22 := true, src1HandoffPe := false }
                        changed := true
                | none => pure ()
              else if ti.engine == .ne && b.port == .src1Dma then
                -- srcReadNE: set 0x4d00 depKind (1 if consumer or active scratch writer has BatchN > 1, else 2), depDist=0 and ensure HB(k, i - 1)
                let wantDepKind :=
                  if b.resource == .scratch then
                    if ti.isMultiBatch || anyActiveMultiBatchScratchWriterBefore pre i b.lo b.hi then 1 else 2
                  else 2
                if syncs[i]!.src1DepKind != wantDepKind || syncs[i]!.src1DepDist != 0 then
                  syncs := syncs.set! i { syncs[i]! with src1DepKind := wantDepKind, src1DepDist := 0 }
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
              else if b.role == .src1Stage && a.role == .resStage then
                -- srcStageAhead: plain HBL/HB does NOT order NE or PE DMA src1Stage fill over draining resStage
                if !(src2Wait5 || src2Wait22) then
                  if tk.engine == .ne then
                    if !syncs[i]!.src1WaitNe5 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitNe5 := true, src1HandoffNe := false }
                      changed := true
                  else
                    if !syncs[i]!.src1WaitPe22 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitPe22 := true, src1HandoffPe := false }
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
    (pt : PreLoweredTask) (sync : TaskSyncInfo) (kernKeepLive : Bool := false) : Except String TaskDesc := do
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
      let (slot, off, inRowStride, inPlaneStride) := match loc with
        | .surface _ s =>
          let (bOff, rSt, pSt, _) := surfaceDmaWindow nd.src1Win sh.cin sh.h sh.w
          (s, bOff, rSt, pSt)
        | .dram o _ =>
          let rSt := alignUp (2 * sh.w) dmaLineBytes
          (barSlotScratch, o, rSt, sh.h * rSt)
        | .l2 _ _ => (0, 0, 0, 0)
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
      let (slot, off, depCode, outRowStride, outPlaneStride) := match loc with
        | .surface _ s =>
          let (bOff, rSt, pSt, _) := surfaceDmaWindow nd.resWin sh.cout ho wo
          (s, bOff + nd.outByteOffset, resDepCode target, rSt, pSt)
        | .dram o _ =>
          let rSt := alignUp (2 * wo) dmaLineBytes
          (barSlotScratch, o + nd.outByteOffset, resDepCode target ||| resDepScratchFlags, rSt, ho * rSt)
        | .l2 _ _ => (0, 0, resDepCode target, 0, 0)
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
      let (slot, off, dmaRow, dmaPlane) := match loc with
        | .surface _ s =>
          let (bOff, rSt, pSt, _) := surfaceDmaWindow nd.src1Win dims.c dims.h dims.w
          (s, bOff, rSt, pSt)
        | .dram o _ =>
          let rSt := alignUp (2 * dims.w) dmaLineBytes
          (barSlotScratch, o, rSt, dims.h * rSt)
        | .l2 _ _ => (0, 0, 0, 0)
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
      let (slot, off, depCode, dmaRow, dmaPlane) := match loc with
        | .surface _ s =>
          let (bOff, rSt, pSt, _) := surfaceDmaWindow nd.resWin dims.c dims.h dims.w
          (s, bOff, resDepCode target, rSt, pSt)
        | .dram o _ =>
          let rSt := alignUp (2 * dims.w) dmaLineBytes
          (barSlotScratch, o, resDepCode target ||| resDepScratchFlags, rSt, dims.h * rSt)
        | .l2 _ _ => (0, 0, resDepCode target, 0, 0)
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
      let (slot, off, dmaRow, dmaPlane) := match loc with
        | .surface _ s =>
          let (bOff, rSt, pSt, _) := surfaceDmaWindow nd.src1Win hw1.c hw1.h hw1.w
          (s, bOff, rSt, pSt)
        | .dram o _ =>
          let rSt := alignUp (2 * hw1.w) dmaLineBytes
          (barSlotScratch, o, rSt, hw1.h * rSt)
        | .l2 _ _ => (0, 0, 0, 0)
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
      let (slot, off, dmaRow, dmaPlane) := match loc with
        | .surface _ s =>
          let (bOff, rSt, pSt, _) := surfaceDmaWindow nd.src2Win hw2.c hw2.h hw2.w
          (s, bOff, rSt, pSt)
        | .dram o _ =>
          let rSt := alignUp (2 * hw2.w) dmaLineBytes
          (barSlotScratch, o, rSt, hw2.h * rSt)
        | .l2 _ _ => (0, 0, 0, 0)
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
      let (slot, off, depCode, dmaRow, dmaPlane) := match loc with
        | .surface _ s =>
          let (bOff, rSt, pSt, _) := surfaceDmaWindow nd.resWin outDims.c outDims.h outDims.w
          (s, bOff, resDepCode target, rSt, pSt)
        | .dram o _ =>
          let rSt := alignUp (2 * outDims.w) dmaLineBytes
          (barSlotScratch, o, resDepCode target ||| resDepScratchFlags, rSt, outDims.h * rSt)
        | .l2 _ _ => (0, 0, resDepCode target, 0, 0)
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
      let effSrcStrip := if act == .copy && totalTasks > 1 then max stripW 32 else stripW
      let row := 2 * min (alignUp dims.w 8) effSrcStrip
      let u := dims.c * row
      let (slot, off, dmaRow, dmaPlane) := match loc with
        | .surface _ s =>
          let (bOff, rSt, pSt, _) := surfaceDmaWindow nd.src1Win dims.c dims.h dims.w
          (s, bOff, rSt, pSt)
        | .dram o _ =>
          let rSt := alignUp (2 * dims.w) dmaLineBytes
          (barSlotScratch, o, rSt, dims.h * rSt)
        | .l2 _ _ => (0, 0, 0, 0)
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
      let (slot, off, depCode, dmaRow, dmaPlane) := match loc with
        | .surface _ s =>
          let (bOff, rSt, pSt, _) := surfaceDmaWindow nd.resWin dims.c dims.h dims.w
          (s, bOff, resDepCode target, rSt, pSt)
        | .dram o _ =>
          let rSt := alignUp (2 * dims.w) dmaLineBytes
          (barSlotScratch, o, resDepCode target ||| resDepScratchFlags, rSt, dims.h * rSt)
        | .l2 _ _ => (0, 0, resDepCode target, 0, 0)
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

  | .pePool op excl dims gh gw, .pePool pp =>
    let toPortLoc := fun
      | TensorLocation.surface _ slot => PoolPortLoc.surface slot
      | TensorLocation.dram off _ => PoolPortLoc.dram off
      | TensorLocation.l2 base _ => PoolPortLoc.l2 base
    buildPePoolTaskMulti target op excl dims gh gw pp taskId totalTasks
      (toPortLoc pt.src1Loc) (toPortLoc pt.resLoc) sync pt.liveL2Mask

  | .peReduce op axes eps dims, .peReduce pr =>
    let toPortLoc := fun
      | TensorLocation.surface _ slot => PoolPortLoc.surface slot
      | TensorLocation.dram off _ => PoolPortLoc.dram off
      | TensorLocation.l2 base _ => PoolPortLoc.l2 base
    buildPeReduceTaskMulti target op axes eps dims pr taskId totalTasks
      (toPortLoc pt.src1Loc) (toPortLoc pt.resLoc) sync pt.liveL2Mask

  | .nePool op dims gh gw, .nePool np =>
    let toPortLoc := fun
      | TensorLocation.surface _ slot => PoolPortLoc.surface slot
      | TensorLocation.dram off _ => PoolPortLoc.dram off
      | TensorLocation.l2 base _ => PoolPortLoc.l2 base
    buildNePoolTaskMulti target op dims gh gw np taskId totalTasks
      (toPortLoc pt.src1Loc) (toPortLoc pt.resLoc) sync pt.liveL2Mask

  | .neChannelReduce op dims, .neChannelReduce nc =>
    let toPortLoc := fun
      | TensorLocation.surface _ slot => PoolPortLoc.surface slot
      | TensorLocation.dram off _ => PoolPortLoc.dram off
      | TensorLocation.l2 base _ => PoolPortLoc.l2 base
    buildNeChannelReduceTaskMulti target op dims nc taskId totalTasks
      (toPortLoc pt.src1Loc) (toPortLoc pt.resLoc) sync pt.liveL2Mask

  | .neView cp, .neView vp =>
    let (inSlot, inBase) ← match pt.src1Loc with
      | .surface _ slot => pure (slot, 0)
      | .dram off _ => pure (barSlotScratch, off)
      | .l2 _ _ => throw s!"internal error: neView node '{nd.id}' cannot use L2-resident input"
    let (outSlot, outBase, outIsScratch) ← match pt.resLoc with
      | .surface _ slot => pure (slot, 0, false)
      | .dram off _ => pure (barSlotScratch, off, true)
      | .l2 _ _ => throw s!"internal error: neView node '{nd.id}' cannot use L2-resident output"
    let cp' : ViewCopy := { cp with inOff := inBase + cp.inOff, outOff := outBase + cp.outOff }
    buildViewTaskMulti target cp' vp taskId totalTasks inSlot outSlot outIsScratch sync pt.liveL2Mask

  | .neMatmul heads m0 m totalM k n scaleBits _ _, .neMatmul mp =>
    let some aLoc := pt.kernLoc?
      | throw s!"internal error: neMatmul node '{nd.id}' missing kernel location"
    let chunkSize := matmulChunkSize k
    let rowBN := alignUp (2 * n) dmaLineBytes
    let (kernSlot, kernOff, kernIsScratch) ← match aLoc with
      | .surface _ slot => pure (slot, m0 * chunkSize, false)
      | .dram off _ => pure (barSlotScratch, off + m0 * chunkSize, true)
      | .l2 _ _ => throw s!"internal error: neMatmul node '{nd.id}' kernel cannot be L2-resident"
    let toMmSrcLoc := fun
      | TensorLocation.surface _ slot => MatmulPortLoc.surface slot 0
      | TensorLocation.dram off _ => MatmulPortLoc.dram off
      | TensorLocation.l2 base _ => MatmulPortLoc.l2 base
    let toMmResLoc := fun
      | TensorLocation.surface _ slot => MatmulPortLoc.surface slot (m0 * rowBN)
      | TensorLocation.dram off _ => MatmulPortLoc.dram (off + m0 * rowBN)
      | TensorLocation.l2 base _ => MatmulPortLoc.l2 base
    buildMatmulTaskMulti target heads m k n scaleBits mp taskId totalTasks
      kernSlot kernOff kernIsScratch pt.extProducerTask?
      (toMmSrcLoc pt.src1Loc) (toMmResLoc pt.resLoc) sync pt.liveL2Mask kernKeepLive totalM

  | _, _ =>
    throw s!"internal error: op/plan mismatch when lowering node '{nd.id}'"

-- ============================================================================
-- L2 Placement Solver & Spatial H-Strip Graph Fusion
-- ============================================================================

/-- Partition intermediate nodes `(0 .. n - 2)` into L2-resident and DRAM-scratch sets
    using deterministic iterative auto-spill until all `n` tasks fit inside 2 MiB L2. -/
private def solveL2Placement
    (target : Target) (defaultPlacement : EdgePlacementMode)
    (vNodes : Array ValidatedNode) (lastConsumer : Array Nat) :
    Except String (List PlacedBuf × Array TaskPlanKind × Nat × Array Nat) := do
  let n := vNodes.size
  let mut pinnedL2 : Array Nat := #[]
  let mut autoL2 : Array Nat := #[]
  let mut dramIds : Array Nat := #[]

  for i in [:n] do
    let nd := vNodes[i]!
    if nd.isFinalGroup then
      continue
    let skipNonRep := match nd.sharedOutGroup with
      | some rep => i != rep
      | none => false
    if skipNonRep then
      continue
    let effMode := nd.outPlacement.getD defaultPlacement
    let eligible := isNodeL2Eligible vNodes i
    match effMode with
    | .dram =>
      dramIds := dramIds.push i
    | .l2 =>
      if !eligible then
        throw s!"node '{nd.id}': forced L2 placement is ineligible (size >= 2 MiB, view/shared-output op, or broadcast/cropped consumer)"
      pinnedL2 := pinnedL2.push i
    | .auto =>
      if eligible then
        autoL2 := autoL2.push i
      else
        dramIds := dramIds.push i

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
        match tryPlanAllTasks target vNodes trialPlaced with
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
      -- Deterministically spill the highest-impact auto-L2 tensor:
      -- prefer candidates live at `failTask?`; when spilling due to aggregate L2 overflow (`failTask? = none`),
      -- maximize the number of over-capacity tasks covered by the candidate so a single spill resolves
      -- multiple adjacent conflicts (breaking ties by largest byte footprint, then highest node index).
      let liveAuto : Array Nat := match failTask? with
        | some ft => curAutoL2.filter fun i => i <= ft && ft <= lastConsumer[i]!
        | none => #[]
      let pool : Array Nat := if liveAuto.isEmpty then curAutoL2 else liveAuto
      let overCapCount : Nat → Nat :=
        if failTask?.isSome then
          fun _ => 0
        else
          fun cand => Id.run do
            let mut cnt := 0
            for t in [cand : lastConsumer[cand]! + 1] do
              let mut sumBytes := 0
              for j in l2Ids do
                if j <= t && t <= lastConsumer[j]! then
                  sumBytes := sumBytes + l2ResidentBytes vNodes[j]!.outDims
              if sumBytes > l2CapacityBytes then
                cnt := cnt + 1
            return cnt
      let mut victim := pool[0]!
      let mut bestHits := overCapCount victim
      let mut bestBytes := l2ResidentBytes vNodes[victim]!.outDims
      for cand in pool do
        let hits := overCapCount cand
        let b := l2ResidentBytes vNodes[cand]!.outDims
        if hits > bestHits || (hits == bestHits && (b > bestBytes || (b == bestBytes && cand > victim))) then
          victim := cand
          bestHits := hits
          bestBytes := b
      curAutoL2 := curAutoL2.filter (fun x => x != victim)
      dramIds := dramIds.push victim

  pure (placedL2, plans, peakL2, dramIds)

/-- Reverse halo propagation for a single output row band `[y0, y1)` of node `M - 1`.
    Verifies exact internal-edge interval matching for every multi-consumer internal node,
    returning `(outIntervals, inIntervals)` per node `0..M-1`. -/
private def propagateStripIntervals
    (vNodes : Array ValidatedNode) (radii : Array Nat) (hFull y0 y1 : Nat) :
    Except String (Array (Nat × Nat) × Array (Nat × Nat)) := do
  let m := vNodes.size
  if y0 >= y1 || y1 > hFull then
    throw s!"invalid strip row band [{y0}, {y1}) for H={hFull}"
  let mut reqOut : Array (Option (Nat × Nat)) := Array.replicate m none
  let mut reqIn : Array (Nat × Nat) := Array.replicate m (0, 0)
  reqOut := reqOut.set! (m - 1) (some (y0, y1))

  for step in [:m] do
    let u := m - 1 - step
    let some (lo, hi) := reqOut[u]!
      | throw s!"spatial strip fusion: node '{vNodes[u]!.id}' is unreachable from output"
    let rad := radii[u]!
    let inLo := if lo >= rad then lo - rad else 0
    let inHi := min hFull (hi + rad)
    if inLo >= inHi then
      throw s!"spatial strip fusion: empty input interval [{inLo}, {inHi}) on node '{vNodes[u]!.id}'"
    reqIn := reqIn.set! u (inLo, inHi)
    for ref in vNodes[u]!.inputs do
      match ref with
      | .input _ => pure ()
      | .node prodIdx =>
        match reqOut[prodIdx]! with
        | none =>
          reqOut := reqOut.set! prodIdx (some (inLo, inHi))
        | some (prevLo, prevHi) =>
          if prevLo != inLo || prevHi != inHi then
            throw s!"spatial strip fusion ineligible: skip-halo interval mismatch on internal node '{vNodes[prodIdx]!.id}' (consumer '{vNodes[u]!.id}' requires [{inLo}, {inHi}), another consumer requires [{prevLo}, {prevHi}))"

  let mut outIntervals : Array (Nat × Nat) := Array.mkEmpty m
  for u in [:m] do
    match reqOut[u]! with
    | some iv => outIntervals := outIntervals.push iv
    | none => throw s!"spatial strip fusion: node '{vNodes[u]!.id}' missing interval"
  pure (outIntervals, reqIn)

/-- Check whether a validated graph is eligible for spatial H-strip fusion.
    Returns `(hFull, wFull, radii, maxTasks)`. -/
private def checkStripEligible
    (spec : GraphSpec) (vNodes : Array ValidatedNode) :
    Except String (Nat × Nat × Array Nat × Nat) := do
  let m := vNodes.size
  if m == 0 then
    throw "spatial strip fusion ineligible: graph has 0 nodes"
  let hFull := spec.inputs[0]!.shape.h
  let wFull := spec.inputs[0]!.shape.w
  if hFull <= 1 then
    throw s!"spatial strip fusion ineligible: input height H={hFull} must be > 1"
  for inp in spec.inputs do
    if inp.shape.h != hFull || inp.shape.w != wFull then
      throw s!"spatial strip fusion ineligible: input '{inp.name}' shape ({inp.shape.h}x{inp.shape.w}) != ({hFull}x{wFull})"

  let mut radii : Array Nat := Array.mkEmpty m
  for nd in vNodes do
    if nd.sharedOutGroup.isSome then
      throw s!"spatial strip fusion ineligible: node '{nd.logicalId}' uses multi-task shared output"
    let rad ← match nd.op with
      | .conv sh _ _ _ _ _ =>
        if sh.kernelSize == 1 then
          if sh.w != wFull then
            throw s!"spatial strip fusion ineligible: node '{nd.id}' width {sh.w} != {wFull}"
          pure 0
        else if sh.kernelSize == 3 && sh.stride == 1 && sh.pad == .same then
          if sh.w != wFull then
            throw s!"spatial strip fusion ineligible: node '{nd.id}' width {sh.w} != {wFull}"
          pure 1
        else
          throw s!"spatial strip fusion ineligible: node '{nd.id}' conv3x3 requires stride=1 and pad=same (got stride={sh.stride}, pad={sh.pad.name})"
      | .pe1 _ dims =>
        if dims.w != wFull then
          throw s!"spatial strip fusion ineligible: node '{nd.id}' width {dims.w} != {wFull}"
        pure 0
      | .pe2 _ _ outDims hw1 hw2 _ _ =>
        if hw1.h != hFull || hw1.w != wFull || hw2.h != hFull || hw2.w != wFull ||
           outDims.h != hFull || outDims.w != wFull then
          throw s!"spatial strip fusion ineligible: node '{nd.id}' pe2 broadcast inputs are not supported"
        pure 0
      | .lut _ dims _ =>
        if dims.w != wFull then
          throw s!"spatial strip fusion ineligible: node '{nd.id}' width {dims.w} != {wFull}"
        pure 0
      | .pePool .. | .nePool .. =>
        throw s!"spatial strip fusion ineligible: node '{nd.id}' pool operation is not supported by spatial H-strip fusion"
      | .peReduce .. | .neChannelReduce .. =>
        throw s!"spatial strip fusion ineligible: node '{nd.id}' reduce operation is not supported by spatial H-strip fusion"
      | .neView .. =>
        throw s!"spatial strip fusion ineligible: node '{nd.id}' view operation is not supported by spatial H-strip fusion"
      | .neMatmul .. =>
        throw s!"spatial strip fusion ineligible: node '{nd.id}' matmul operation is not supported by spatial H-strip fusion"
    radii := radii.push rad

  -- Probe skip-halo exact interval match on top and interior 1-row bands
  let _ ← propagateStripIntervals vNodes radii hFull 0 1
  let mid := hFull / 2
  let _ ← propagateStripIntervals vNodes radii hFull mid (mid + 1)

  let anyKern := vNodes.any (fun nd => nd.usesKernel)
  let maxTasks := if anyKern then 255 else 256
  pure (hFull, wFull, radii, maxTasks)

/-- Materialize the `M` strip-local `ValidatedNode`s for output row band `[y0, y1)`. -/
private def buildStripNodes
    (vNodes : Array ValidatedNode) (radii : Array Nat)
    (hFull y0 y1 : Nat) (stripSuffix : String := "") : Except String (Array ValidatedNode) := do
  let m := vNodes.size
  let (outIntervals, inIntervals) ← propagateStripIntervals vNodes radii hFull y0 y1
  let mut out : Array ValidatedNode := Array.mkEmpty m
  for l in [:m] do
    let nd := vNodes[l]!
    let (outLo, outHi) := outIntervals[l]!
    let (inLo, inHi) := inIntervals[l]!
    let outH := outHi - outLo
    let inH := inHi - inLo
    let (newOp, s1Ref, s2Ref?) := match nd.op with
      | .conv sh w b act lp bk =>
        let sh' : ConvSpec :=
          if sh.kernelSize == 1 then
            { sh with h := inH }
          else
            let pt := if outLo == 0 then 1 else 0
            let pb := if outHi == hFull then 1 else 0
            { sh with h := inH, pad := .strip pt pb }
        (PlannedOp.conv sh' w b act lp bk, nd.inputs[0]!, none)
      | .pe1 op dims =>
        (PlannedOp.pe1 op { dims with h := outH }, nd.inputs[0]!, none)
      | .pe2 binOp rev outDims hw1 hw2 r1 r2 =>
        (PlannedOp.pe2 binOp rev { outDims with h := outH } { hw1 with h := inH } { hw2 with h := inH } r1 r2,
         r1, some r2)
      | .lut act dims lp =>
        (PlannedOp.lut act { dims with h := outH } lp, nd.inputs[0]!, none)
      | other =>
        (other, nd.inputs[0]!, none)
    let src1Win : SurfaceWindow := match s1Ref with
      | .input _ => { rowOffset := inLo, fullH := hFull }
      | .node _ => {}
    let src2Win : SurfaceWindow := match s2Ref? with
      | some (.input _) => { rowOffset := inLo, fullH := hFull }
      | _ => {}
    let resWin : SurfaceWindow :=
      if l + 1 == m then { rowOffset := outLo, fullH := hFull } else {}
    let nodeId := if stripSuffix.isEmpty then nd.id else s!"{nd.id}{stripSuffix}"
    out := out.push {
      nd with
      id := nodeId
      nodeIdx := l
      origNodeIdx := l
      op := newOp
      outDims := { nd.outDims with h := outH }
      src1Win := src1Win
      src2Win := src2Win
      resWin := resWin
    }
  pure out

private def shiftTensorRef (base : Nat) : TensorRef → TensorRef
  | .input idx => .input idx
  | .node idx => .node (base + idx)

private def shiftValidatedNode (base : Nat) (nd : ValidatedNode) : ValidatedNode :=
  let newOp := match nd.op with
    | .pe2 binOp rev od hw1 hw2 r1 r2 =>
      .pe2 binOp rev od hw1 hw2 (shiftTensorRef base r1) (shiftTensorRef base r2)
    | other => other
  {
    nd with
    nodeIdx := base + nd.nodeIdx
    inputs := nd.inputs.map (shiftTensorRef base)
    op := newOp
  }

/-- Assemble, synchronize, emit, and statically verify a flat `N`-task stream
    (used by both unstripped and strip-fused graph compilation). -/
private def finalizeCompiledGraph
    (spec : GraphSpec)
    (shortNames : Array String)
    (fullOutDims : TensorDims)
    (fullOutId : String)
    (flatNodes : Array ValidatedNode)
    (flatPlans : Array TaskPlanKind)
    (placedL2 : List PlacedBuf)
    (peakL2 : Nat)
    (nodeLocs : Array TensorLocation)
    (scratchBytes : Nat)
    (stripCount stripHeight : Nat) : Except String CompiledGraph := do
  let n := flatNodes.size
  let totalSurfs := spec.inputs.size + 1
  let kernelBarSlot : UInt8 := barSlotFirstSurface + totalSurfs.toUInt8

  let (preTasks, kernBuf) ← buildPreLoweredTasks spec shortNames flatNodes flatPlans placedL2 nodeLocs
  let syncs := computeSynchronization preTasks

  let mut taskDescs : Array TaskDesc := Array.mkEmpty n
  for i in [:n] do
    let kernKeepLive : Bool :=
      match preTasks[i]!.kernLoc? with
      | some (.dram off bytes) =>
        Id.run do
          for j in [i + 1 : n] do
            if preTasks[j]!.dramAccesses.any (fun a => a.resource == .scratch && a.port == .kernel && intervalsOverlap off (off + bytes) a.lo a.hi) then
              return true
            if preTasks[j]!.dramAccesses.any (fun a => a.resource == .scratch && a.isWrite && intervalsOverlap off (off + bytes) a.lo a.hi) then
              return false
          return false
      | _ => false
    let td ← emitTaskDesc spec.target n kernelBarSlot preTasks[i]! syncs[i]! kernKeepLive
    taskDescs := taskDescs.push td

  let mut surfaces : Array SurfaceSpec := Array.mkEmpty totalSurfs
  for i in [:spec.inputs.size] do
    let inp := spec.inputs[i]!
    let sn := shortNames[i]!
    let sym := s!"input_{inp.name}"
    let s ← makeFp16Surface spec.target sn sym .input inp.shape.c inp.shape.h inp.shape.w
    surfaces := surfaces.push s
  let outSurf ← makeFp16Surface spec.target "y" s!"output_{fullOutId}" .output
    fullOutDims.c fullOutDims.h fullOutDims.w
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

  let summary ← verifyHwxTasksSummary emitted.hwx emitted.info (requireKnownTask := true)
  if summary.total != n then
    throw s!"internal error: verifier checked {summary.total} tasks != {n}"

  let compiledNodes : Array CompiledGraphNode := Array.ofFn (n := n) fun i =>
    let nd := flatNodes[i.val]!
    {
      id := nd.id
      taskId := i.val
      engine := nd.engine
      opKind := nd.opKind
      outDims := nd.outDims
      outLocation := nodeLocs[i.val]!
      direct := flatPlans[i.val]!.direct
      l2UseBytes := flatPlans[i.val]!.l2Peak
    }

  pure {
    target := spec.target
    taskCount := n
    scratchBytes := scratchBytes
    l2PeakBytes := peakL2
    kernelBytes := kernBuf.size
    stripCount := stripCount
    stripHeight := stripHeight
    nodes := compiledNodes
    hwx := emitted.hwx
    info := emitted.info
    sections := emitted.sections
  }

private def finishUnstrippedGraph
    (spec : GraphSpec) (shortNames : Array String)
    (vNodes : Array ValidatedNode) (lastConsumer : Array Nat)
    (placedL2 : List PlacedBuf) (plans : Array TaskPlanKind)
    (peakL2 : Nat) (dramIds : Array Nat) : Except String CompiledGraph := do
  let n := vNodes.size
  let dramLiveBufs : List LiveBuf := dramIds.toList.map fun i => {
    id := i
    bytes := max (dramScratchBytes vNodes[i]!.groupOutDims) vNodes[i]!.minDramBytes
    first := i
    last := lastConsumer[i]!
    priority := lastConsumer[i]! - i
  }
  let placedDram := allocate true dramLiveBufs
  let scratchBytes := topOf placedDram

  let outSurfIdx := spec.inputs.size
  let outBarSlot : UInt8 := barSlotFirstSurface + outSurfIdx.toUInt8
  let mut nodeLocs : Array TensorLocation := Array.mkEmpty n
  for i in [:n] do
    let nd := vNodes[i]!
    if nd.isFinalGroup then
      nodeLocs := nodeLocs.push (.surface outSurfIdx outBarSlot)
    else
      match nd.sharedOutGroup with
      | some rep =>
        match lookupPlaced placedDram rep with
        | some pb => nodeLocs := nodeLocs.push (.dram pb.base pb.bytes)
        | none => throw s!"internal error: shared-output group node {i} (rep {rep}) missing from DRAM allocations"
      | none =>
        match lookupPlaced placedL2 i with
        | some pb => nodeLocs := nodeLocs.push (.l2 pb.base pb.bytes)
        | none =>
          match lookupPlaced placedDram i with
          | some pb => nodeLocs := nodeLocs.push (.dram pb.base pb.bytes)
          | none => throw s!"internal error: node {i} missing from both L2 and DRAM allocations"

  let outNode := vNodes[n - 1]!
  let outId := if outNode.logicalId.isEmpty then outNode.id else outNode.logicalId
  finalizeCompiledGraph spec shortNames outNode.groupOutDims outId vNodes plans placedL2 peakL2 nodeLocs scratchBytes 1 0

private def compileStrippedAtHeight
    (spec : GraphSpec) (shortNames : Array String)
    (vNodes : Array ValidatedNode) (lastConsumer : Array Nat)
    (hFull : Nat) (radii : Array Nat) (maxTasks hOut : Nat) : Except String CompiledGraph := do
  if hOut == 0 then
    throw "strip height must be >= 1"
  let m := vNodes.size
  let sCount := ceilDiv hFull hOut
  let totalTasks := sCount * m
  if totalTasks > maxTasks then
    throw s!"strip height {hOut} expands {m} nodes across {sCount} strips to {totalTasks} tasks, exceeding limit {maxTasks}"

  let mut flatNodes : Array ValidatedNode := Array.mkEmpty totalTasks
  let mut flatPlans : Array TaskPlanKind := Array.mkEmpty totalTasks
  let mut allPlacedL2 : List PlacedBuf := []
  let mut allDramLiveBufs : List LiveBuf := []
  let mut overallPeakL2 : Nat := 0

  for s in [:sCount] do
    let y0 := s * hOut
    let y1 := min hFull (y0 + hOut)
    let suffix := if sCount > 1 then s!"_s{s}" else ""
    let stripNodes ← buildStripNodes vNodes radii hFull y0 y1 suffix
    let (sPlacedL2, sPlans, sPeakL2, sDramIds) ← solveL2Placement spec.target spec.defaultPlacement stripNodes lastConsumer
    let base := s * m
    overallPeakL2 := max overallPeakL2 sPeakL2
    for l in [:m] do
      flatNodes := flatNodes.push (shiftValidatedNode base stripNodes[l]!)
      flatPlans := flatPlans.push sPlans[l]!
    let shiftedL2 := sPlacedL2.map fun pb => {
      pb with
      id := base + pb.id
      first := base + pb.first
      last := base + pb.last
    }
    allPlacedL2 := allPlacedL2 ++ shiftedL2
    let shiftedDram : List LiveBuf := sDramIds.toList.map fun l => {
      id := base + l
      bytes := dramScratchBytes stripNodes[l]!.outDims
      first := base + l
      last := base + lastConsumer[l]!
      priority := lastConsumer[l]! - l
    }
    allDramLiveBufs := allDramLiveBufs ++ shiftedDram

  let placedDram := allocate true allDramLiveBufs
  let scratchBytes := topOf placedDram

  let outSurfIdx := spec.inputs.size
  let outBarSlot : UInt8 := barSlotFirstSurface + outSurfIdx.toUInt8
  let mut nodeLocs : Array TensorLocation := Array.mkEmpty totalTasks
  for i in [:totalTasks] do
    if i % m == m - 1 then
      nodeLocs := nodeLocs.push (.surface outSurfIdx outBarSlot)
    else
      match lookupPlaced allPlacedL2 i with
      | some pb => nodeLocs := nodeLocs.push (.l2 pb.base pb.bytes)
      | none =>
        match lookupPlaced placedDram i with
        | some pb => nodeLocs := nodeLocs.push (.dram pb.base pb.bytes)
        | none => throw s!"internal error: stripped task {i} missing from both L2 and DRAM allocations"

  let outNode := vNodes[m - 1]!
  let effStripHeight := min hOut hFull
  finalizeCompiledGraph spec shortNames outNode.outDims outNode.id
    flatNodes flatPlans allPlacedL2 overallPeakL2 nodeLocs scratchBytes sCount effStripHeight

private def isTextSizeExceededError (msg : String) : Bool :=
  (msg.splitOn "task stream size").length > 1

private def evalInteriorStripMetrics
    (spec : GraphSpec) (vNodes : Array ValidatedNode) (lastConsumer : Array Nat)
    (hFull : Nat) (radii : Array Nat) (h : Nat) : Option (Nat × Nat) :=
  if h == 0 || h > hFull then none
  else
    let y0 := (hFull - h) / 2
    let y1 := y0 + h
    match buildStripNodes vNodes radii hFull y0 y1 with
    | .error _ => none
    | .ok stripNodes =>
      match solveL2Placement spec.target spec.defaultPlacement stripNodes lastConsumer with
      | .error _ => none
      | .ok (_, _, _, dramIds) =>
        if dramIds.isEmpty then
          some (0, 0)
        else
          let dramBufs : List LiveBuf := dramIds.toList.map fun l => {
            id := l
            bytes := dramScratchBytes stripNodes[l]!.outDims
            first := l
            last := lastConsumer[l]!
            priority := lastConsumer[l]! - l
          }
          some (dramIds.size, topOf (allocate true dramBufs))

private def chooseAutoStripHeight
    (spec : GraphSpec) (vNodes : Array ValidatedNode) (lastConsumer : Array Nat)
    (hFull : Nat) (radii : Array Nat) (minH : Nat) : Option Nat := Id.run do
  if minH == 0 || minH >= hFull then
    return none
  -- Fast path: if minH achieves 0 DRAM scratch, binary search for the largest h achieving 0 scratch.
  if evalInteriorStripMetrics spec vNodes lastConsumer hFull radii minH == some (0, 0) then
    let mut lo := minH
    let mut hi := hFull - 1
    let mut best := minH
    while lo <= hi do
      let mid := (lo + hi) / 2
      if evalInteriorStripMetrics spec vNodes lastConsumer hFull radii mid == some (0, 0) then
        best := mid
        lo := mid + 1
      else
        if mid == 0 then break
        hi := mid - 1
    return some best

  -- Fallback: minimize (stripScratchEdges, stripScratchBytes), breaking ties by largest h.
  -- Since intermediate L2 and DRAM footprints grow monotonically with strip height h,
  -- once consecutive heights strictly worsen (edges, bytes) beyond the best seen, stop early.
  let mut best? : Option (Nat × Nat × Nat) := none
  let mut worseStreak : Nat := 0
  for h in [minH : hFull] do
    match evalInteriorStripMetrics spec vNodes lastConsumer hFull radii h with
    | none =>
      worseStreak := worseStreak + 1
      if best?.isSome && worseStreak >= 4 then
        break
    | some (edges, bytes) =>
      let isBetter := match best? with
        | none => true
        | some (bh, bEdges, bBytes) =>
          if edges != bEdges then edges < bEdges
          else if bytes != bBytes then bytes < bBytes
          else h > bh
      if isBetter then
        best? := some (h, edges, bytes)
        worseStreak := 0
      else
        worseStreak := worseStreak + 1
        if worseStreak >= 4 then
          break

  return best?.map (fun (h, _, _) => h)

-- ============================================================================
-- Top-Level Multi-Task Graph Compiler
-- ============================================================================

/-- Compile an arbitrary multi-task DAG (`1..256` tasks, `1..3` inputs, `1` output) into a
    statically verified HWX binary (checked by independent `Vane.Verify.verifyHwxTasksSummary`)
    and its 4 host-built Linux LOAD sections. Supports optional spatial H-strip fusion via `spec.stripMode`. -/
def compileGraph (spec : GraphSpec) : Except String CompiledGraph := do
  let (shortNames, vNodes, lastConsumer) ← validateGraph spec

  match spec.stripMode with
  | .off =>
    let (placedL2, plans, peakL2, dramIds) ← solveL2Placement spec.target spec.defaultPlacement vNodes lastConsumer
    finishUnstrippedGraph spec shortNames vNodes lastConsumer placedL2 plans peakL2 dramIds

  | .fixed hOut =>
    let (hFull, _, radii, maxTasks) ← checkStripEligible spec vNodes
    compileStrippedAtHeight spec shortNames vNodes lastConsumer hFull radii maxTasks hOut

  | .auto =>
    -- 1. If unstripped already fits 100% in L2 (0 B DRAM scratch), keep it unstripped.
    let unstrippedSolve := solveL2Placement spec.target spec.defaultPlacement vNodes lastConsumer
    if let .ok (placedL2, plans, peakL2, dramIds) := unstrippedSolve then
      if dramIds.isEmpty then
        return (← finishUnstrippedGraph spec shortNames vNodes lastConsumer placedL2 plans peakL2 dramIds)

    -- 2. Otherwise, attempt automatic spatial H-strip fusion if eligible.
    match checkStripEligible spec vNodes with
    | .error _ =>
      let (placedL2, plans, peakL2, dramIds) ← unstrippedSolve
      finishUnstrippedGraph spec shortNames vNodes lastConsumer placedL2 plans peakL2 dramIds
    | .ok (hFull, _, radii, maxTasks) =>
      let m := vNodes.size
      let maxStrips := maxTasks / m
      if maxStrips <= 1 then
        let (placedL2, plans, peakL2, dramIds) ← unstrippedSolve
        finishUnstrippedGraph spec shortNames vNodes lastConsumer placedL2 plans peakL2 dramIds
      else
        let minH := ceilDiv hFull maxStrips
        match chooseAutoStripHeight spec vNodes lastConsumer hFull radii minH with
        | none =>
          let (placedL2, plans, peakL2, dramIds) ← unstrippedSolve
          finishUnstrippedGraph spec shortNames vNodes lastConsumer placedL2 plans peakL2 dramIds
        | some startH =>
          let mut curH := startH
          let mut result? : Option CompiledGraph := none
          while result?.isNone && curH < hFull do
            match compileStrippedAtHeight spec shortNames vNodes lastConsumer hFull radii maxTasks curH with
            | .ok compiled =>
              result? := some compiled
            | .error msg =>
              if isTextSizeExceededError msg then
                let curS := ceilDiv hFull curH
                if curS <= 2 then
                  break
                curH := ceilDiv hFull (curS - 1)
              else
                throw msg
          match result? with
          | some compiled => pure compiled
          | none =>
            let (placedL2, plans, peakL2, dramIds) ← unstrippedSolve
            finishUnstrippedGraph spec shortNames vNodes lastConsumer placedL2 plans peakL2 dramIds

end Vane.Graph
