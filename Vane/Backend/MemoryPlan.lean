import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Hwx
import Vane.Register
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
import Vane.Ane.Types
import Vane.Machine.Types

namespace Vane.Backend.MemoryPlan

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Hwx
open Vane.Register
open Vane.Alloc
open Vane.Tiling
open Vane.Quant
open Vane.Program
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

-- ============================================================================
-- Budget-Aware Per-Task Tile & L2 Staging Choosers (5 Unified AneOp Families)
-- ============================================================================

def placeTempBuffers (occupied : List Range) : List Nat → Option (List Nat)
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

def maxEndOfRanges (rs : List Range) : Nat :=
  rs.foldl (fun acc r => max acc r.hi) 0

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
    | some [sBase] => some (sBase, 0, max (maxEndOfRanges occupied) (sBase + alignUp srcSz dmaLineBytes))
    | _ => none
  else if srcIsL2 && !resIsL2 then
    match placeTempBuffers occupied [resSz] with
    | some [rBase] => some (0, rBase, max (maxEndOfRanges occupied) (rBase + alignUp resSz dmaLineBytes))
    | _ => none
  else
    match placeTempBuffers occupied [srcSz, resSz] with
    | some [sBase, rBase] =>
      some (sBase, rBase, max (maxEndOfRanges occupied) (max (sBase + alignUp srcSz dmaLineBytes) (rBase + alignUp resSz dmaLineBytes)))
    | _ => none

private def maxConvThForOccupied
    (target : Target) (sh : ConvSpec) (probe : Tiling)
    (srcIsL2 resIsL2 : Bool) (occupied : List Range) : Option (Nat × Nat × Nat × Nat) := Id.run do
  let ho := sh.outH
  if ho == 0 then return none
  if let some (sb, rb, pk) := convStageBasesForTh target sh probe srcIsL2 resIsL2 occupied ho then
    return some (ho, sb, rb, pk)
  if ho <= 1 then return none
  let mut lo : Nat := 1
  let mut hi : Nat := ho - 1
  let mut best : Option (Nat × Nat × Nat × Nat) := none
  for _ in [:20] do
    if lo > hi then break
    let mid := (lo + hi) / 2
    match convStageBasesForTh target sh probe srcIsL2 resIsL2 occupied mid with
    | some (sb, rb, pk) =>
      best := some (mid, sb, rb, pk)
      lo := mid + 1
    | none =>
      if mid == 0 then break
      hi := mid - 1
  return best

def chooseConvTaskPlan
    (target : Target) (sh : ConvSpec)
    (extraLutBytes : Nat) (hasBiasInKernel : Bool)
    (srcIsL2 resIsL2 : Bool) (occupied : List Range)
    (quantParams? : Option QuantChunkParams := none) : Except String ConvTaskPlan := do
  validateConvShape sh
  let eff := sh.effMode
  let wo := sh.outW
  let maxF := min 4 (ceilLog2 sh.cout)
  let fCandidates : Array Nat :=
    if eff == .depthwise && sh.cout >= 16 then #[4]
    else Array.ofFn (n := maxF + 1) fun step => maxF - step.val

  for logNE in #[4, 3, 2, 1, 0] do
    for fat in #[0, 1] do
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
                      logNE := logNE, f := f, fat := fat, ocg := ocg,
                      shPref := shPref, shMax := shMax, tileHeight := 1
                    }
                    let chunkOk := match quantParams? with
                      | some qp => (checkConvQuantChunks64K sh probe qp).isOk
                      | none => (checkConvDenseChunks64K sh probe extraLutBytes hasBiasInKernel).isOk
                    let ocgOk := (validateConvOcgSafety target sh probe).isOk
                    let covOk := (validateConvCoverageSafety target sh probe).isOk
                    if chunkOk && ocgOk && covOk then
                      if let some (maxTh, sb, rb, pk) := maxConvThForOccupied target sh probe srcIsL2 resIsL2 occupied then
                        let cand : Tiling := { probe with tileHeight := maxTh }
                        if (validateConvChooserPolicy sh cand).isOk then
                          let direct := if resIsL2 then false else isDirectConv target sh cand
                          return {
                            shape := sh, tiling := cand, direct := direct,
                            srcStageBase := sb, resStageBase := rb, l2TaskPeak := pk
                          }
  throw s!"no legal {target.name} conv tiling fits available L2 budget for ({sh.h}x{sh.w}, {sh.cin}->{sh.cout})"

/-- Place all live L2-resident buffers via `Vane.Alloc.allocate` (first-fit interval reuse). -/
def allocateL2Buffers (liveBufs : List LiveBuf) : List PlacedBuf :=
  allocate true liveBufs

/-- Place all spilled BAR 3 DRAM scratch buffers via `Vane.Alloc.allocate` and return `(placed, totalBytes)`. -/
def allocateScratchBuffers (dramLiveBufs : List LiveBuf) : List PlacedBuf × Nat :=
  let placed := allocate true dramLiveBufs
  (placed, topOf placed)

private def peStageBasesForTh
    (target : Target) (outDims hwSrc1 : TensorDims) (hwSrc2? : Option TensorDims)
    (src1IsL2 src2IsL2 resIsL2 : Bool) (occupied : List Range) (k th : Nat) :
    Option (Nat × Nat × Nat × Nat) :=
  let twoSource := hwSrc2?.isSome
  let stripW := peStripWidth k
  let effStripW := max stripW 32
  let direct := if resIsL2 then false else peIsDirect target outDims.w k
  let whole1 := twoSource && hwSrc1.c == 1 && hwSrc1.h == 1 && hwSrc1.w > 1 && outDims.h > 1 && th < outDims.h
  let row1 := 2 * (if whole1 then alignUp hwSrc1.w 8 else min (alignUp hwSrc1.w 8) effStripW)
  let size1 := if src1IsL2 then 0 else (if hwSrc1.h > 1 then th else 1) * (hwSrc1.c * row1)
  let size2 := match hwSrc2? with
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

  let need1 := !src1IsL2
  let need2 := hwSrc2?.isSome && !src2IsL2
  let needR := !resIsL2
  let reqSizes : List Nat :=
    (if need1 then [size1] else []) ++
    (if need2 then [size2] else []) ++
    (if needR then [sizeRes] else [])
  match placeTempBuffers occupied reqSizes with
  | none => none
  | some offs =>
    let basePeak := maxEndOfRanges occupied
    let (b1, rest1, pk1) :=
      if need1 then
        match offs with
        | o :: r => (o, r, max basePeak (o + alignUp size1 dmaLineBytes))
        | [] => (0, [], basePeak)
      else (0, offs, basePeak)
    let (b2, rest2, pk2) :=
      if need2 then
        match rest1 with
        | o :: r => (o, r, max pk1 (o + alignUp size2 dmaLineBytes))
        | [] => (0, [], pk1)
      else (0, rest1, pk1)
    let (bRes, pk3) :=
      if needR then
        match rest2 with
        | o :: _ => (o, max pk2 (o + alignUp sizeRes dmaLineBytes))
        | [] => (0, pk2)
      else (0, pk2)
    some (b1, b2, bRes, pk3)

private def maxPeThForOccupied
    (target : Target) (outDims hwSrc1 : TensorDims) (hwSrc2? : Option TensorDims)
    (src1IsL2 src2IsL2 resIsL2 : Bool) (occupied : List Range) (k : Nat) :
    Option (Nat × Nat × Nat × Nat × Nat) := Id.run do
  if outDims.h == 0 then return none
  if let some (b1, b2, br, pk) := peStageBasesForTh target outDims hwSrc1 hwSrc2? src1IsL2 src2IsL2 resIsL2 occupied k outDims.h then
    return some (outDims.h, b1, b2, br, pk)
  let hasL2Bcast :=
    (src1IsL2 && hwSrc1 != outDims) ||
    (match hwSrc2? with | some hw2 => src2IsL2 && hw2 != outDims | none => false)
  if hasL2Bcast || outDims.h <= 1 then return none
  let mut lo : Nat := 1
  let mut hi : Nat := outDims.h - 1
  let mut best : Option (Nat × Nat × Nat × Nat × Nat) := none
  for _ in [:20] do
    if lo > hi then break
    let mid := (lo + hi) / 2
    match peStageBasesForTh target outDims hwSrc1 hwSrc2? src1IsL2 src2IsL2 resIsL2 occupied k mid with
    | some (b1, b2, br, pk) =>
      best := some (mid, b1, b2, br, pk)
      lo := mid + 1
    | none =>
      if mid == 0 then break
      hi := mid - 1
  return best

def choosePeTaskPlan
    (target : Target) (outDims hwSrc1 : TensorDims) (hwSrc2? : Option TensorDims)
    (src1IsL2 src2IsL2 resIsL2 : Bool) (occupied : List Range) : Except String PeTaskPlan := do
  let twoSource := hwSrc2?.isSome
  let maxK := peMaxK twoSource
  let outWAligned := alignUp outDims.w 8
  let mut best? : Option (PeTaskPlan × Nat × Nat × Nat) := none
  for step in [:maxK - 4] do
    let k := maxK - step
    if let some (th, b1, b2, br, pk) := maxPeThForOccupied target outDims hwSrc1 hwSrc2? src1IsL2 src2IsL2 resIsL2 occupied k then
      let stripW := peStripWidth k
      let direct := if resIsL2 then false else peIsDirect target outDims.w k
      let strips := ceilDiv outWAligned stripW * ceilDiv outDims.h th
      let area := stripW * th
      let plan : PeTaskPlan := {
        outDims := outDims, hwSrc1 := hwSrc1, hwSrc2? := hwSrc2?,
        tiling := { k := k, tileHeight := th }, direct := direct,
        src1StageBase := b1, src2StageBase := b2, resStageBase := br, l2TaskPeak := pk
      }
      let isBetter := match best? with
        | none => true
        | some (bPlan, bStrips, bArea, bStripW) =>
          if plan.direct != bPlan.direct then plan.direct
          else if strips != bStrips then strips < bStrips
          else if area != bArea then area > bArea
          else stripW < bStripW
      if isBetter then best? := some (plan, strips, area, stripW)
  match best? with
  | some (winner, _, _, _) => pure winner
  | none => throw s!"no legal {target.name} PE tiling fits available L2 budget for out={repr outDims}"

private def neActStageBasesForTh
    (target : Target) (c _h w : Nat) (probe : NeActTiling)
    (srcIsL2 resIsL2 : Bool) (occupied : List Range) (th : Nat) (srcFloor32 : Bool := false) :
    Option (Nat × Nat × Nat) :=
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
    | some [sb] => some (sb, 0, max (maxEndOfRanges occupied) (sb + alignUp srcSz dmaLineBytes))
    | _ => none
  else if srcIsL2 && !resIsL2 then
    match placeTempBuffers occupied [resSz] with
    | some [rb] => some (0, rb, max (maxEndOfRanges occupied) (rb + alignUp resSz dmaLineBytes))
    | _ => none
  else
    match placeTempBuffers occupied [srcSz, resSz] with
    | some [sb, rb] => some (sb, rb, max (maxEndOfRanges occupied) (max (sb + alignUp srcSz dmaLineBytes) (rb + alignUp resSz dmaLineBytes)))
    | _ => none

def chooseNeActTaskPlan
    (target : Target) (c h w : Nat)
    (srcIsL2 resIsL2 : Bool) (occupied : List Range) (srcFloor32 : Bool := false) :
    Except String NeActTaskPlan := do
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
      for _ in [:20] do
        if lo > hi then break
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

/-- Budget-aware tiling chooser for `dmaView` supporting DMA↔DMA, Surface→L2, L2→Surface, and L2→L2. -/
def chooseDmaViewTaskPlan
    (target : Target) (cp : ViewCopy) (srcIsL2 resIsL2 : Bool) (occupied : List Range) :
    Except String ViewTaskPlan := do
  if !srcIsL2 && !resIsL2 then
    return ← chooseViewTaskPlan target cp occupied
  if cp.n == 0 || cp.c == 0 || cp.h == 0 || cp.w == 0 || cp.d == 0 then
    throw s!"dmaView: invalid dims ({cp.n}x{cp.c}x{cp.h}x{cp.w})"
  if (srcIsL2 || resIsL2) && (cp.n != 1 || cp.d != 1) then
    throw "dmaView: L2-resident view requires BatchN=1 and D=1"
  if resIsL2 && cp.transposeCW then
    throw "dmaView: L2-resident view output requires !transposeCW"
  let tryProbe := fun (probe : Tiling) =>
    let cand : Tiling := { probe with tileHeight := cp.h }
    let direct := if resIsL2 then false else viewIsDirect target cp cand
    let srcSz := if srcIsL2 then 0 else viewSrcL2Bytes cp cand
    let resSz := if resIsL2 then 0 else viewResL2Bytes target cp cand
    if srcIsL2 && resIsL2 then
      some ({ tiling := cand, direct := false, srcStageBase := 0, resStageBase := 0, l2TaskPeak := maxEndOfRanges occupied } : ViewTaskPlan)
    else if !srcIsL2 && resIsL2 then
      match placeTempBuffers occupied [srcSz] with
      | some [sb] => some { tiling := cand, direct := false, srcStageBase := sb, resStageBase := 0, l2TaskPeak := max (maxEndOfRanges occupied) (sb + alignUp srcSz dmaLineBytes) }
      | _ => none
    else
      match placeTempBuffers occupied [resSz] with
      | some [rb] => some { tiling := cand, direct := direct, srcStageBase := 0, resStageBase := rb, l2TaskPeak := max (maxEndOfRanges occupied) (rb + alignUp resSz dmaLineBytes) }
      | _ => none
  for fat in #[0, 1] do
    for shPref in [:5] do
      let stripW := (128 <<< fat) >>> shPref
      if stripW >= 8 && stripW <= 256 then
        let shMax := max shPref (coverMin cp.w stripW fat)
        if shMax <= 4 then
          let probe : Tiling := { logNE := fat, f := 0, fat := fat, ocg := 0, shPref := shPref, shMax := shMax, tileHeight := cp.h }
          if let some plan := tryProbe probe then
            return plan
  throw s!"no legal {target.name} L2-assisted dmaView tiling fits 2 MiB L2"

-- ============================================================================
-- Zero-Copy Subview Expansion & Flat Task Preparation
-- ============================================================================

private structure PrepTask where
  name : String
  logicalId : String
  op : AneOp
  inputs : Array ValueId
  outVid : ValueId
  outShape : Shape4D
  /-- Parent value ID when this task writes a zero-copy subview into a shared parent buffer. -/
  parentVid? : Option ValueId := none
  subviewByteOff : Nat := 0
  minDramBytes : Nat := 0
  outPlacement : Option EdgePlacementMode := none
  src1Win : SurfaceWindow := {}
  src2Win : SurfaceWindow := {}
  resWin : SurfaceWindow := {}
  stripOutputIdx? : Option Nat := none
  /-- Explicit `ViewCopy` when expanded from non-aliased concat/view. -/
  explicitViewCopy? : Option ViewCopy := none
  deriving Inhabited

private def lookupAssoc {α : Type} (arr : Array (ValueId × α)) (v : ValueId) : Option α :=
  (arr.find? fun p => p.1 == v).map Prod.snd

/-- Derive validated distinct 1..7-byte short names for all input and output surfaces. -/
private def deriveSurfaceShortNames
    (inputs : Array (ValueId × String × TensorType))
    (inShortOverrides : Array String)
    (outCount : Nat)
    (outShortOverrides : Array String) : Except String (Array String × Array String) := do
  let mut outShorts : Array String := Array.mkEmpty outCount
  for oi in [:outCount] do
    let ov := outShortOverrides.getD oi ""
    let sn := if !ov.isEmpty then ov else if outCount == 1 then "y" else s!"y{oi}"
    if sn.utf8ByteSize < 1 || sn.utf8ByteSize > 7 then
      throw s!"output[{oi}]: shortName '{sn}' length must be 1..7"
    if outShorts.contains sn then
      throw s!"duplicate output shortName '{sn}'"
    outShorts := outShorts.push sn

  let mut inShorts : Array String := Array.mkEmpty inputs.size
  for ii in [:inputs.size] do
    let (_, nm, _) := inputs[ii]!
    let ov := inShortOverrides.getD ii ""
    let sn := if !ov.isEmpty then ov else if inputs.size == 1 then "x" else s!"in{ii}"
    if outShorts.contains sn then
      throw s!"input '{nm}': shortName '{sn}' is reserved for an output surface"
    if sn.utf8ByteSize < 1 || sn.utf8ByteSize > 7 then
      throw s!"input '{nm}': shortName '{sn}' length must be 1..7"
    if inShorts.contains sn then
      throw s!"duplicate input shortName '{sn}'"
    inShorts := inShorts.push sn
  pure (inShorts, outShorts)

/-- Pack logical output tensors sequentially at 64B-aligned byte offsets inside the single
    ABI v3 HWX output surface, returning `(slots, totalPackedBytes)`. -/
def packOutputSlots (shapes : Array Shape4D) (names : Array String) : Array OutputSlot × Nat :=
  let rec go (i packOffset : Nat) (acc : Array OutputSlot) : Array OutputSlot × Nat :=
    if h : i < shapes.size then
      let sh := shapes[i]
      let rSt := alignUp (2 * sh.w) dmaLineBytes
      let pSt := sh.h * rSt
      let bSt := sh.c * pSt
      let sz := alignUp (sh.n * bSt) dmaLineBytes
      let nm := names.getD i (if shapes.size == 1 then "y" else s!"y{i}")
      let slot : OutputSlot := {
        name := nm, shape := sh, byteOffset := packOffset, byteSize := sz,
        rowStride := rSt, planeStride := pSt, batchStride := bSt
      }
      go (i + 1) (packOffset + sz) (acc.push slot)
    else
      (acc, packOffset)
  termination_by shapes.size - i
  go 0 0 (Array.mkEmpty shapes.size)

/-- Pure helper computing the packed multi-output HWX output surface shape `(hwxOutShape, isHomogeneous)`. -/
def computeHwxOutShape (outSlots : Array OutputSlot) (packOffset : Nat) : Except String (Shape4D × Bool) :=
  if outSlots.size == 1 then
    .ok (outSlots[0]!.shape, true)
  else
    let sh0 := outSlots[0]!.shape
    let allSameHW := outSlots.all fun s => s.shape.n == 1 && s.shape.h == sh0.h && s.shape.w == sh0.w
    let totalC := outSlots.foldl (fun acc s => acc + s.shape.c) 0
    if allSameHW && totalC <= 65535 then
      .ok ({ n := 1, c := totalC, h := sh0.h, w := sh0.w }, true)
    else
      let totalLines := packOffset / dmaLineBytes
      let (cPack, hPack) :=
        if totalLines <= 65535 then (totalLines, 1)
        else
          let hp := (totalLines + 65534) / 65535
          ((totalLines + hp - 1) / hp, hp)
      if cPack == 0 || cPack > 65535 || hPack == 0 || hPack > 65535 then
        .error s!"MemoryPlan: packed multi-output size {packOffset}B exceeds HWX output surface limits"
      else
        .ok ({ n := 1, c := cPack, h := hPack, w := 32 }, false)

-- ============================================================================
-- Main L3 Memory Planner (`AneProgram → Except String MachineProgram`)
-- ============================================================================

private def lookupPlaced (ps : List PlacedBuf) (id : Nat) : Option PlacedBuf :=
  ps.find? fun p => p.id == id

/-- Resolve whether an intermediate logical buffer `bufKey` (identified by representative task index)
    is legally eligible for L2 SRAM residence under all `Vane.Verify` hardware rules. -/
private def isLogicalBufL2Eligible
    (target : Target)
    (tasks : Array PrepTask)
    (vidToOwner : Array (ValueId × Sum Nat Nat))
    (sharedParents : Array ValueId)
    (repIdx : Nat) : Bool := Id.run do
  let pt := tasks[repIdx]!
  if pt.parentVid?.isSome || sharedParents.contains pt.outVid then
    return false
  if l2ResidentBytes4D pt.outShape >= l2CapacityBytes then
    return false
  match pt.op with
  | .dmaView desc lbl =>
    -- Only `w_slice` (non-transpose, N=1, D=1, explicit L2 placement) writes directly to L2
    if lbl != "w_slice" || desc.transposeCW || desc.batchN != 1 || desc.d != 1 then
      return false
  | .neMatmul heads _ _ _ _ _ _ _ totalM =>
    if heads > 1 || totalM > 16 then
      return false
  | _ => pure ()

  let outVid := pt.outVid
  for j in [repIdx + 1 : tasks.size] do
    let cons := tasks[j]!
    if cons.inputs.contains outVid then
      match cons.op with
      | .neConv 3 3 sh _ pad mode cin cout _ _ _ _ _ _ =>
        if sh == 2 then
          let cMode := mode.toConv3x3Mode?.getD .dense
          let cs : ConvSpec := {
            kernelSize := 3, mode := cMode, h := pt.outShape.h, w := pt.outShape.w,
            cin := cin, cout := cout, stride := sh, pad := pad
          }
          if cs.winReg != cs.w || cs.hinReg != cs.h then
            return false
      | .pe (.pool op kh kw sh sw pad excl) _ | .neAct (.nePool op kh kw sh sw pad excl) _ _ =>
        let pspec : PoolSpec := {
          target := target, op := op, dims := shape4DToDims pt.outShape,
          kh := kh, kw := kw, sh := sh, sw := sw, pad := pad, excludePad := excl
        }
        match resolvePoolBackend pspec with
        | .ok (_, gh, gw) =>
          if gh.used != pt.outShape.h || gw.used != pt.outShape.w then return false
        | .error _ => return false
      | .dmaView desc lbl =>
        if (lbl != "w_stitch" && lbl != "pack_out") || desc.transposeCW || desc.batchN != 1 || desc.d != 1 then
          return false
      | .neMatmul heads .. =>
        -- Operand A (inputs[0]) is dynamic kernel (must be DRAM/Surface); operand B (inputs[1]) with heads>1 must be DRAM/Surface
        if cons.inputs[0]? == some outVid then return false
        if cons.inputs[1]? == some outVid && heads > 1 then return false
      | _ => pure ()
  let _ := vidToOwner
  return true

/-- Compute `(lastReadTask, firstWriteTask)` bounds for a candidate `(inIdx -> outIdx)` donation. -/
def computeDonationTaskBounds
    (buffers : Array BufferStorage)
    (machineTasks : Array MachineTask)
    (taskOutSurfIdx : Array (Option Nat))
    (inIdx outIdx : Nat) : Except String (Nat × Nat) := do
  let mut lastRead? : Option Nat := none
  let mut firstWrite? : Option Nat := none
  for ti in [:machineTasks.size] do
    let mt := machineTasks[ti]!
    for inBid in mt.inputs do
      match ← resolveBufferStorage buffers inBid with
      | .surface sIdx .input _ _ _ _ =>
        if sIdx == inIdx then
          lastRead? := some (match lastRead? with | some prev => max prev ti | none => ti)
      | _ => pure ()
    if taskOutSurfIdx[ti]! == some outIdx then
      firstWrite? := some (match firstWrite? with | some prev => min prev ti | none => ti)
  let some lr := lastRead?
    | throw s!"MemoryPlan: buffer donation ({inIdx} -> {outIdx}) has no task reading input surface {inIdx}"
  let some fw := firstWrite?
    | throw s!"MemoryPlan: buffer donation ({inIdx} -> {outIdx}) has no task writing output {outIdx}"
  pure (lr, fw)

/-- Total recursive validator for buffer donation lists, enforcing bounds, uniqueness,
    exact shape & span match, and strict `lastReadTask < firstWriteTask` separation. -/
def validateDonationsAux
    (inputSurfs : Array SurfaceSpec)
    (outSlots : Array OutputSlot)
    (hwxOutSpanBytes : Nat)
    (inSurfBufIds : Array BufferId)
    (outSurfBufId : BufferId)
    (buffers : Array BufferStorage)
    (machineTasks : Array MachineTask)
    (taskOutSurfIdx : Array (Option Nat))
    (seenIn seenOut : List Nat)
    (acc : Array BufferDonation) : List (Nat × Nat) → Except String (Array BufferDonation)
  | [] => .ok acc
  | (inIdx, outIdx) :: rest => do
    if inIdx >= inputSurfs.size || outIdx >= outSlots.size then
      throw s!"MemoryPlan: buffer donation ({inIdx} -> {outIdx}) out of bounds"
    if seenIn.contains inIdx then
      throw s!"MemoryPlan: duplicate buffer donation input index {inIdx}"
    if seenOut.contains outIdx then
      throw s!"MemoryPlan: duplicate buffer donation output index {outIdx}"
    let inS := inputSurfs[inIdx]!
    let outSl := outSlots[outIdx]!
    if inS.channels.toNat != outSl.shape.c || inS.height.toNat != outSl.shape.h ||
       inS.width.toNat != outSl.shape.w || inS.n.toNat != outSl.shape.n then
      throw s!"MemoryPlan: buffer donation ({inIdx} -> {outIdx}) shape mismatch"
    let inSpanBytes := inS.batchStride.toNat * inS.n.toNat
    if inSpanBytes != hwxOutSpanBytes then
      throw s!"MemoryPlan: buffer donation ({inIdx} -> {outIdx}) surface span mismatch ({inSpanBytes}B != {hwxOutSpanBytes}B)"
    let (lr, fw) ← computeDonationTaskBounds buffers machineTasks taskOutSurfIdx inIdx outIdx
    if lr >= fw then
      throw s!"MemoryPlan: unsafe buffer donation ({inIdx} -> {outIdx}): input last read at task {lr} is not strictly before output first write at task {fw}"
    let d : BufferDonation := {
      inputSurfIdx := inIdx
      outputSurfIdx := outIdx
      inputBuf := inSurfBufIds[inIdx]!
      outputBuf := outSurfBufId
      lastReadTask := lr
      firstWriteTask := fw
    }
    validateDonationsAux inputSurfs outSlots hwxOutSpanBytes inSurfBufIds outSurfBufId
      buffers machineTasks taskOutSurfIdx (inIdx :: seenIn) (outIdx :: seenOut) (acc.push d) rest

/-- Validate optional buffer donations with strict liveness safety:
    - Reject on packed multi-output programs (`outSlots.size > 1`)
    - Reject duplicate `inIdx` or duplicate `outIdx`
    - Shape and surface span must match exactly
    - `lastReadTask(inIdx) < firstWriteTask(outIdx)` strictly (never same-task or post-write read). -/
def validateDonations
    (rawDonations : Array (Nat × Nat))
    (inputSurfs : Array SurfaceSpec)
    (outSlots : Array OutputSlot)
    (hwxOutSpanBytes : Nat)
    (inSurfBufIds : Array BufferId)
    (outSurfBufId : BufferId)
    (buffers : Array BufferStorage)
    (machineTasks : Array MachineTask)
    (taskOutSurfIdx : Array (Option Nat)) : Except String (Array BufferDonation) := do
  if !rawDonations.isEmpty && outSlots.size > 1 then
    throw s!"MemoryPlan: buffer donation is unsupported on packed multi-output programs (outputs={outSlots.size})"
  validateDonationsAux inputSurfs outSlots hwxOutSpanBytes inSurfBufIds outSurfBufId
    buffers machineTasks taskOutSurfIdx [] [] #[] rawDonations.toList

/-- Run L3 bufferization, zero-copy subview resolution, exact live-interval analysis,
    `Vane.Alloc.allocate` L2 rolling/ping-pong slot packing, selective L2↔DRAM BAR 3 spilling,
    and ABI v3 single-output-surface multi-output packing on a value-semantic `AneProgram`,
    emitting a `MachineProgram` with explicit `MemoryPlanWitness`. -/
def planMemory (prog : AneProgram) : Except String MachineProgram := do
  let g := prog.graph
  if g.inputs.isEmpty || g.inputs.size > 3 then
    throw s!"MemoryPlan: graph must have 1..3 network inputs, got {g.inputs.size}"
  if g.outputs.isEmpty then
    throw "MemoryPlan: graph must have at least 1 output"
  if g.nodes.isEmpty && g.outputs.size < 2 then
    throw "MemoryPlan: graph must have at least 1 node"

  -- ABI v3 requires exactly 1 HWX output surface (`buf_id = 4`), so 1..3 inputs + 1 output = 2..4 HWX surfaces.
  let (inShorts, outShorts) ← deriveSurfaceShortNames g.inputs prog.inputShortNames 1 prog.outputShortNames

  -- Collect shapes of all SSA values in `g`
  let mut shMap : Array (ValueId × Shape4D) := #[]
  for (vid, nm, ty) in g.inputs do
    shMap := shMap.push (vid, ← tensorTypeToShape4D ty s!"input '{nm}'")
  for nd in g.nodes do
    for (ov, oty) in nd.outputs do
      shMap := shMap.push (ov, ← tensorTypeToShape4D oty s!"node '{nd.name}'")

  let getSh := fun (v : ValueId) (ctx : String) => do
    match lookupAssoc shMap v with
    | some s => pure s
    | none => throw s!"MemoryPlan ({ctx}): missing shape for %{v}"

  -- Count consumers of every SSA value in `g`
  let mut useCount : Array (ValueId × Nat) := #[]
  let bumpUse := fun (m : Array (ValueId × Nat)) (v : ValueId) =>
    match m.findIdx? fun p => p.1 == v with
    | some idx => m.set! idx (v, m[idx]!.2 + 1)
    | none => m.push (v, 1)
  for nd in g.nodes do
    for iv in nd.inputs do useCount := bumpUse useCount iv
  for ov in g.outputs do useCount := bumpUse useCount ov

  let getUses := fun (v : ValueId) => (lookupAssoc useCount v).getD 0

  -- Build map from ValueId -> producer node index (for alias analysis)
  let mut prodNodeOf : Array (ValueId × Nat) := #[]
  for ni in [:g.nodes.size] do
    for (ov, _) in g.nodes[ni]!.outputs do
      prodNodeOf := prodNodeOf.push (ov, ni)

  -- Detect zero-copy aliasable `concat(axis=1)` joins (e.g., SOK channel split, matmul m>16 join)
  -- and zero-copy aliasable contiguous C-axis `slice` views!
  let mut skippedNodes : Array Bool := Array.replicate g.nodes.size false
  -- Producer node `ni` redirected to write a subview `(parentOutVid, byteOffset)`
  let mut prodSubviewOf : Array (Option (ValueId × Nat)) := Array.replicate g.nodes.size none
  -- Slice output ValueId redirected to `(parentInVid, byteOffset, sliceShape)`
  let mut readSubviewOf : Array (ValueId × ValueId × Nat × Shape4D) := #[]
  let mut sharedParents : Array ValueId := #[]

  for ni in [:g.nodes.size] do
    let nd := g.nodes[ni]!
    match nd.op with
    | .dmaView desc _ =>
      if desc.aliasable then
        match desc.concatAxis? with
        | some 1 =>
          -- Zero-copy concat(axis=1) if every input has a single consumer (this concat)
          -- and is produced by an op that supports writing at an explicit `outByteOffset`
          -- (`neConv 1x1`, `neMatmul`, or `neConv 3x3` when writing to scratch).
          let mut canAlias : Bool := nd.inputs.size >= 2
          let mut offsets : Array Nat := Array.mkEmpty nd.inputs.size
          let mut prodIdxs : Array Nat := Array.mkEmpty nd.inputs.size
          let outVid := nd.outputs[0]!.1
          let isOutSurf := g.outputs.contains outVid
          let outSh ← getSh outVid nd.name
          let outPlaneSt := outSh.planeStride
          let mut cCursor : Nat := 0
          for si in [:nd.inputs.size] do
            let iv := nd.inputs[si]!
            if getUses iv != 1 then
              canAlias := false
            match lookupAssoc prodNodeOf iv with
            | none => canAlias := false
            | some pIdx =>
              let pNd := g.nodes[pIdx]!
              let pSh ← getSh iv pNd.name
              match pNd.op with
              | .neConv 1 1 .. =>
                offsets := offsets.push (cCursor * outPlaneSt)
                prodIdxs := prodIdxs.push pIdx
                cCursor := cCursor + pSh.c
              | .neConv 3 3 .. =>
                if isOutSurf then canAlias := false
                else
                  offsets := offsets.push (cCursor * outPlaneSt)
                  prodIdxs := prodIdxs.push pIdx
                  cCursor := cCursor + pSh.c
              | .neMatmul _ _ _ n _ _ _ m0 _ =>
                let rowBN := alignUp (2 * n) dmaLineBytes
                offsets := offsets.push (m0 * rowBN)
                prodIdxs := prodIdxs.push pIdx
                cCursor := cCursor + pSh.c
              | _ => canAlias := false
          if canAlias then
            skippedNodes := skippedNodes.set! ni true
            sharedParents := sharedParents.push outVid
            for k in [:prodIdxs.size] do
              prodSubviewOf := prodSubviewOf.set! prodIdxs[k]! (some (outVid, offsets[k]!))
        | none =>
          -- Zero-copy contiguous C-axis slice if NOT a graph output, input is a graph input surface,
          -- and ALL consumers are `neConv 1x1`!
          let outVid := nd.outputs[0]!.1
          let inVid := nd.inputs[0]!
          let inIsInputSurf := g.inputs.any fun e => e.1 == inVid
          if !g.outputs.contains outVid && inIsInputSurf && desc.srcBaseOff % dmaLineBytes == 0 then
            let mut allConsumersConv1x1 := true
            let mut hasAnyConsumer := false
            for cj in [ni + 1 : g.nodes.size] do
              let cNd := g.nodes[cj]!
              if cNd.inputs.contains outVid then
                hasAnyConsumer := true
                match cNd.op with
                | .neConv 1 1 .. => pure ()
                | _ => allConsumersConv1x1 := false
            if hasAnyConsumer && allConsumersConv1x1 then
              skippedNodes := skippedNodes.set! ni true
              readSubviewOf := readSubviewOf.push (outVid, inVid, desc.srcBaseOff, desc.outShape)
        | _ => pure ()
    | _ => pure ()

  -- Expand remaining `g.nodes` into flat `PrepTask`s (expanding non-aliased `concat` into `m` `dmaView` copy tasks)
  let mut prepTasks : Array PrepTask := #[]
  let mut nextVirtVid := g.nextValueId + 1
  for ni in [:g.nodes.size] do
    if skippedNodes[ni]! then continue
    let nd := g.nodes[ni]!
    let ndMeta := prog.getNodeMeta ni
    let outVid := nd.outputs[0]!.1
    let outSh ← getSh outVid nd.name
    let logicalId := if ndMeta.logicalId.isEmpty then nd.name else ndMeta.logicalId
    let s1Win : SurfaceWindow := { rowOffset := ndMeta.src1Win.rowOffset, fullH := ndMeta.src1Win.fullH }
    let s2Win : SurfaceWindow := { rowOffset := ndMeta.src2Win.rowOffset, fullH := ndMeta.src2Win.fullH }
    let rWin : SurfaceWindow := { rowOffset := ndMeta.resWin.rowOffset, fullH := ndMeta.resWin.fullH }

    match nd.op with
    | .dmaView desc lbl =>
      match desc.concatAxis? with
      | some ax =>
        -- Non-aliased concat: expand into `nd.inputs.size` `dmaView` copy tasks writing into shared `outVid`
        let inShapes := desc.concatInShapes
        let vp ← planConcat inShapes ax
        sharedParents := sharedParents.push outVid
        for ci in [:vp.copies.size] do
          let cp := vp.copies[ci]!
          let cpZeroOut : ViewCopy := { cp with outOff := 0 }
          let slVid := nextVirtVid; nextVirtVid := nextVirtVid + 1
          let slSh := inShapes[ci]!
          prepTasks := prepTasks.push {
            name := s!"{nd.name}_c{ci}"
            logicalId := logicalId
            op := .dmaView { desc with inShape := slSh, outShape := slSh, concatAxis? := none, explicitCopy? := some cpZeroOut } lbl
            inputs := #[nd.inputs[ci]!]
            outVid := slVid
            outShape := slSh
            parentVid? := some outVid
            subviewByteOff := cp.outOff
            outPlacement := ndMeta.outPlacement
            src1Win := s1Win
            resWin := rWin
            stripOutputIdx? := ndMeta.stripOutputIdx?
            explicitViewCopy? := some cpZeroOut
          }
      | none =>
        let (pVid?, subOff) := match prodSubviewOf[ni]! with
          | some (pv, off) => (some pv, off)
          | none => (none, 0)
        prepTasks := prepTasks.push {
          name := nd.name
          logicalId := logicalId
          op := nd.op
          inputs := nd.inputs
          outVid := outVid
          outShape := outSh
          parentVid? := pVid?
          subviewByteOff := subOff
          minDramBytes := desc.minDramBytes
          outPlacement := ndMeta.outPlacement
          src1Win := s1Win
          src2Win := s2Win
          resWin := rWin
          stripOutputIdx? := ndMeta.stripOutputIdx?
          explicitViewCopy? := some desc.toViewCopy
        }
    | _ =>
      let (pVid?, subOff) := match prodSubviewOf[ni]! with
        | some (pv, off) => (some pv, off)
        | none => (none, 0)
      prepTasks := prepTasks.push {
        name := nd.name
        logicalId := logicalId
        op := nd.op
        inputs := nd.inputs
        outVid := outVid
        outShape := outSh
        parentVid? := pVid?
        subviewByteOff := subOff
        outPlacement := ndMeta.outPlacement
        src1Win := s1Win
        src2Win := s2Win
        resWin := rWin
        stripOutputIdx? := ndMeta.stripOutputIdx?
      }

  -- Determine full logical output shapes and canonical 64B-strided `OutputSlot` packing
  -- inside the SINGLE ABI v3 HWX output surface (`buf_id = 4`).
  let mut outLogicalShapes : Array Shape4D := Array.replicate g.outputs.size { n := 1, c := 1, h := 1, w := 1 }
  for oi in [:g.outputs.size] do
    let ov := g.outputs[oi]!
    let osh ← getSh ov s!"graph output[{oi}]"
    let mut fullSh := osh
    for ti in [:prepTasks.size] do
      let pt := prepTasks[ti]!
      if pt.stripOutputIdx? == some oi then
        if pt.resWin.fullH > 0 then
          fullSh := { pt.outShape with h := pt.resWin.fullH }
        else if let .dmaView desc "w_stitch" := pt.op then
          fullSh := desc.outShape
    outLogicalShapes := outLogicalShapes.set! oi fullSh

  let (outSlots, packOffset) := packOutputSlots outLogicalShapes prog.outputNames
  for oi in [:outSlots.size] do
    if outSlots[oi]!.byteSize == 0 then
      throw s!"MemoryPlan: logical output[{oi}] has zero byte size"

  -- Determine the single ABI v3 HWX output surface shape:
  -- - Single output: exact legacy `[1, C, H, W]` shape (zero overhead).
  -- - Multi-output homogeneous `(H, W)` with `sum(C) <= 65535`: `[1, sum(C), H, W]` (zero padding; direct zero-copy for conv1x1 & dmaView).
  -- - Multi-output heterogeneous `(H, W)`: `[1, cPack, hPack, 32]` (`rowStride=64B`, `span >= packOffset`).
  let (hwxOutShape, isHomogeneousPack) ← computeHwxOutShape outSlots packOffset

  -- For multi-output programs (`g.outputs.size >= 2`), route each logical output `oi` either:
  -- 1. Zero-copy directly into `outSlots[oi].byteOffset` of the packed output surface (when homogeneous,
  --    not an input, not duplicated, not read downstream, and every writer is `neConv 1x1` or `dmaView`), OR
  -- 2. For H-stripped outputs (`resWin.fullH > 0`), insert a per-strip `pack_out_{oi}_s{si}` `dmaView` store
  --    immediately after each strip writer so L2 strip buffers die immediately for rolling reuse, OR
  -- 3. For unstripped outputs, append a `pack_out_{oi}` `dmaView` store task (supporting `n >= 1`).
  let mut directMultiOutMap : Array (Option Nat) := Array.replicate prepTasks.size none
  if g.outputs.size >= 2 then
    let origTaskCount := prepTasks.size
    let mut origTasks := prepTasks
    let mut origDirectOut : Array (Option Nat) := Array.replicate origTaskCount none
    let mut insertAfter : Array (Array (PrepTask × Nat)) := Array.replicate origTaskCount #[]
    let mut appendEnd : Array (PrepTask × Nat) := #[]
    for oi in [:g.outputs.size] do
      let ov := g.outputs[oi]!
      let slot := outSlots[oi]!
      let mut writerIdxs : Array Nat := #[]
      for ti in [:origTaskCount] do
        let pt := origTasks[ti]!
        if pt.stripOutputIdx? == some oi || (pt.stripOutputIdx?.isNone && (pt.parentVid?.getD pt.outVid) == ov) then
          writerIdxs := writerIdxs.push ti
      let writerOutVids := writerIdxs.map fun ti => origTasks[ti]!.outVid
      let isInput := g.inputs.any fun e => e.1 == ov
      let isDupOut := (g.outputs.filter (· == ov)).size > 1
      let hasDownstream := (Array.range origTaskCount).any fun ti =>
        origTasks[ti]!.inputs.any fun iv => iv == ov || writerOutVids.contains iv
      let isStripped := writerIdxs.any fun ti => origTasks[ti]!.resWin.fullH > 0
      let allWritersDirectCapable := !writerIdxs.isEmpty && writerIdxs.all fun ti =>
        match origTasks[ti]!.op with
        | .neConv 1 1 .. => true
        | .dmaView .. => true
        | _ => false
      let canWriteDirect := isHomogeneousPack && !isInput && !isDupOut && !hasDownstream && allWritersDirectCapable
      if canWriteDirect then
        for ti in writerIdxs do
          let pt := origTasks[ti]!
          origTasks := origTasks.set! ti { pt with subviewByteOff := slot.byteOffset + pt.subviewByteOff }
          origDirectOut := origDirectOut.set! ti (some oi)
      else if isStripped then
        for si in [:writerIdxs.size] do
          let ti := writerIdxs[si]!
          let pt := origTasks[ti]!
          let rowOff := pt.resWin.rowOffset
          -- Strip producer now writes its strip to L2/DRAM; clear `resWin` & `stripOutputIdx?`
          origTasks := origTasks.set! ti { pt with stripOutputIdx? := none, resWin := {} }
          let stripSh := pt.outShape
          let inPSt := stripSh.h * slot.rowStride
          let inBSt := stripSh.c * inPSt
          let cp : ViewCopy := {
            n := stripSh.n, c := stripSh.c, h := stripSh.h, w := stripSh.w, d := 1
            inOff := 0
            inRowStride := if stripSh.h > 1 then slot.rowStride else 0
            inPlaneStride := if stripSh.c > 1 then inPSt else 0
            inNStride := if stripSh.n > 1 then inBSt else 0
            outOff := 0
            outRowStride := if stripSh.h > 1 then slot.rowStride else 0
            outPlaneStride := if stripSh.c > 1 then slot.planeStride else 0
            outNStride := if stripSh.n > 1 then slot.batchStride else 0
            transposeCW := false
          }
          let desc : DmaViewSpec := {
            inShape := stripSh, outShape := stripSh
            srcStrides := stripSh.dramStrides, dstStrides := slot.shape.dramStrides
            batchN := stripSh.n
            explicitCopy? := some cp
          }
          let pv := nextVirtVid; nextVirtVid := nextVirtVid + 1
          shMap := shMap.push (pv, stripSh)
          let packTask : PrepTask := {
            name := s!"pack_out_{oi}_s{si}"
            logicalId := slot.name
            op := .dmaView desc "pack_out"
            inputs := #[pt.outVid]
            outVid := pv
            outShape := stripSh
            subviewByteOff := slot.byteOffset + rowOff * slot.rowStride
            stripOutputIdx? := some oi
            explicitViewCopy? := some cp
          }
          insertAfter := insertAfter.set! ti (insertAfter[ti]!.push (packTask, oi))
      else
        -- Clear stripOutputIdx? on producer tasks if any, and append `pack_out_{oi}` dmaView store task
        for ti in writerIdxs do
          let pt := origTasks[ti]!
          if pt.stripOutputIdx? == some oi then
            origTasks := origTasks.set! ti { pt with stripOutputIdx? := none }
        let cp : ViewCopy := {
          n := slot.shape.n, c := slot.shape.c, h := slot.shape.h, w := slot.shape.w, d := 1
          inOff := 0
          inRowStride := if slot.shape.h > 1 then slot.rowStride else 0
          inPlaneStride := if slot.shape.c > 1 then slot.planeStride else 0
          inNStride := if slot.shape.n > 1 then slot.batchStride else 0
          outOff := 0
          outRowStride := if slot.shape.h > 1 then slot.rowStride else 0
          outPlaneStride := if slot.shape.c > 1 then slot.planeStride else 0
          outNStride := if slot.shape.n > 1 then slot.batchStride else 0
          transposeCW := false
        }
        let desc : DmaViewSpec := {
          inShape := slot.shape, outShape := slot.shape
          srcStrides := slot.shape.dramStrides, dstStrides := slot.shape.dramStrides
          batchN := slot.shape.n
          explicitCopy? := some cp
        }
        let pv := nextVirtVid; nextVirtVid := nextVirtVid + 1
        shMap := shMap.push (pv, slot.shape)
        appendEnd := appendEnd.push ({
          name := s!"pack_out_{oi}"
          logicalId := slot.name
          op := .dmaView desc "pack_out"
          inputs := #[ov]
          outVid := pv
          outShape := slot.shape
          subviewByteOff := slot.byteOffset
          stripOutputIdx? := some oi
          explicitViewCopy? := some cp
        }, oi)
    -- Rebuild `prepTasks` and `directMultiOutMap` preserving exact interleaved strip order
    let mut rebuiltTasks : Array PrepTask := #[]
    let mut rebuiltOutMap : Array (Option Nat) := #[]
    for ti in [:origTaskCount] do
      rebuiltTasks := rebuiltTasks.push origTasks[ti]!
      rebuiltOutMap := rebuiltOutMap.push origDirectOut[ti]!
      for (ptPack, oiPack) in insertAfter[ti]! do
        rebuiltTasks := rebuiltTasks.push ptPack
        rebuiltOutMap := rebuiltOutMap.push (some oiPack)
    for (ptPack, oiPack) in appendEnd do
      rebuiltTasks := rebuiltTasks.push ptPack
      rebuiltOutMap := rebuiltOutMap.push (some oiPack)
    prepTasks := rebuiltTasks
    directMultiOutMap := rebuiltOutMap

  let n := prepTasks.size
  if n == 0 || n > 256 then
    throw s!"MemoryPlan: expanded task count {n} out of hardware range 1..256"
  for ti in [:n] do
    if prepTasks[ti]!.op.usesKernel && ti >= 255 then
      throw s!"node '{prepTasks[ti]!.name}' (taskId={ti}) uses __KERN_0, which requires taskId < 255"

  -- Map each logical ValueId to either `.inl inputSurfIdx` or `.inr repTaskIdx`
  -- (for shared parents written by multiple subview tasks, `repTaskIdx` is the first writer's task index)
  let mut vidToOwner : Array (ValueId × Sum Nat Nat) := #[]
  for ii in [:g.inputs.size] do
    vidToOwner := vidToOwner.push (g.inputs[ii]!.1, .inl ii)
  for ti in [:n] do
    let pt := prepTasks[ti]!
    let keyVid := pt.parentVid?.getD pt.outVid
    if (lookupAssoc vidToOwner keyVid).isNone then
      vidToOwner := vidToOwner.push (keyVid, .inr ti)
    if pt.parentVid?.isSome && (lookupAssoc vidToOwner pt.outVid).isNone then
      vidToOwner := vidToOwner.push (pt.outVid, .inr ti)
  for (slVid, parentInVid, _, _) in readSubviewOf do
    if let some owner := lookupAssoc vidToOwner parentInVid then
      vidToOwner := vidToOwner.push (slVid, owner)

  -- Map each task `ti` to its logical output index `0..g.outputs.size-1` (when writing to the HWX output surface)
  let mut taskOutSurfIdx : Array (Option Nat) := Array.replicate n none
  if g.outputs.size == 1 then
    for ti in [:n] do
      let pt := prepTasks[ti]!
      if let some oi := pt.stripOutputIdx? then
        taskOutSurfIdx := taskOutSurfIdx.set! ti (some oi)
      else
        let keyVid := pt.parentVid?.getD pt.outVid
        if g.outputs[0]! == keyVid then
          taskOutSurfIdx := taskOutSurfIdx.set! ti (some 0)
  else
    taskOutSurfIdx := directMultiOutMap

  -- Verify every declared logical output slot has at least one writer task
  for oi in [:g.outputs.size] do
    if !taskOutSurfIdx.any (· == some oi) then
      throw s!"MemoryPlan: logical output[{oi}] ('{outSlots[oi]!.name}') has no writer task"

  -- Ensure final task writes to the HWX output surface (required by Vane.Verify S1/S2)
  if taskOutSurfIdx[n - 1]!.isNone then
    throw s!"MemoryPlan: final task {n - 1} ('{prepTasks[n - 1]!.name}') does not write to an output surface"

  -- Compute exact live intervals `[firstTask, lastTask]` for every non-output intermediate logical buffer
  let mut firstTaskOf : Array Nat := Array.ofFn (n := n) fun i => i.val
  let mut lastTaskOf : Array Nat := Array.replicate n 0
  let mut isRepBuf : Array Bool := Array.replicate n false
  let mut repOfTask : Array Nat := Array.ofFn (n := n) fun i => i.val
  let mut bufShapeOf : Array Shape4D := Array.ofFn (n := n) fun i => prepTasks[i.val]!.outShape
  let mut bufMinDramOf : Array Nat := Array.ofFn (n := n) fun i => prepTasks[i.val]!.minDramBytes

  for ti in [:n] do
    let pt := prepTasks[ti]!
    let keyVid := pt.parentVid?.getD pt.outVid
    let rep := match lookupAssoc vidToOwner keyVid with
      | some (.inr r) => r
      | _ => ti
    repOfTask := repOfTask.set! ti rep
    if rep == ti && taskOutSurfIdx[ti]!.isNone then
      isRepBuf := isRepBuf.set! rep true
      let fullSh ← getSh keyVid pt.name
      bufShapeOf := bufShapeOf.set! rep fullSh
    if rep != ti then
      firstTaskOf := firstTaskOf.set! rep (min firstTaskOf[rep]! ti)
      lastTaskOf := lastTaskOf.set! rep (max lastTaskOf[rep]! ti)
    bufMinDramOf := bufMinDramOf.set! rep (max bufMinDramOf[rep]! pt.minDramBytes)

    for iv in pt.inputs do
      -- Follow readSubviewOf if present
      let effIv := match readSubviewOf.find? fun e => e.1 == iv with
        | some (_, par, _, _) => par
        | none => iv
      if let some (.inr prodRep) := lookupAssoc vidToOwner effIv then
        lastTaskOf := lastTaskOf.set! prodRep (max lastTaskOf[prodRep]! ti)

  -- Check every non-output intermediate logical buffer is consumed (no dead L2 writes!)
  for rep in [:n] do
    if isRepBuf[rep]! && lastTaskOf[rep]! <= rep then
      throw s!"intermediate node '{prepTasks[rep]!.logicalId}' (task {rep}) is never consumed by any later node"

  -- Partition intermediate buffers into pinnedL2, autoL2, dramIds
  let mut pinnedL2 : Array Nat := #[]
  let mut autoL2 : Array Nat := #[]
  let mut dramIds : Array Nat := #[]

  for rep in [:n] do
    if !isRepBuf[rep]! then continue
    let pt := prepTasks[rep]!
    let effMode := pt.outPlacement.getD prog.defaultPlacement
    let eligible := isLogicalBufL2Eligible g.target prepTasks vidToOwner sharedParents rep
    match effMode with
    | .dram => dramIds := dramIds.push rep
    | .l2 =>
      if !eligible then
        throw s!"node '{pt.name}': forced L2 placement is ineligible (size >= 2 MiB, view/shared-output op, or broadcast/cropped consumer)"
      pinnedL2 := pinnedL2.push rep
    | .auto =>
      if eligible then autoL2 := autoL2.push rep
      else dramIds := dramIds.push rep

  -- Pre-validate conv LUT payloads once (deterministic)
  let mut taskLutPayloads : Array (Option LutPayload) := Array.mkEmpty n
  for ti in [:n] do
    let lp? ← match prepTasks[ti]!.op with
      | .neConv _ _ _ _ _ _ _ _ (.fp16 _) _ (.lut lspec) _ _ _ => some <$> buildLutTable lspec
      | .neAct (.act (.lut lspec)) _ _ => some <$> buildLutTable lspec
      | _ => pure none
    taskLutPayloads := taskLutPayloads.push lp?

  -- Helper: try planning all `n` tasks against a candidate `placedL2`
  let tryPlanAll (placedL2 : List PlacedBuf) : Except (Nat × String) (Array MachineTilePlan × Nat) := do
    let mut plans : Array MachineTilePlan := Array.mkEmpty n
    let mut peakL2 : Nat := topOf placedL2
    let isVidL2 := fun (v : ValueId) =>
      if readSubviewOf.any (fun e => e.1 == v) then false
      else match lookupAssoc vidToOwner v with
        | some (.inr r) => (lookupPlaced placedL2 r).isSome
        | _ => false

    for t in [:n] do
      let pt := prepTasks[t]!
      let liveRanges := liveAt placedL2 t
      let resIsL2 := taskOutSurfIdx[t]!.isNone && (lookupPlaced placedL2 repOfTask[t]!).isSome
      let lp? := taskLutPayloads[t]!
      let planRes : Except String MachineTilePlan := match pt.op with
        | .neConv kh _ sh _ pad mode cin cout weight bias _ _ _ _ => do
          let inSh ← getSh pt.inputs[0]! pt.name
          let cMode ← match mode.toConv3x3Mode? with
            | some cm => pure cm
            | none => throw s!"node '{pt.name}': grouped conv mode '{mode.name}' is not supported"
          let inH := if pt.src1Win.fullH > 0 then
            -- Input surface window height equals expected input height for this strip
            if kh == 1 then pt.outShape.h
            else match pad with
              | .strip ptop pbot => pt.outShape.h + 2 - ptop - pbot
              | _ => inSh.h
            else inSh.h
          let cs : ConvSpec := {
            kernelSize := kh, mode := cMode, h := inH, w := inSh.w,
            cin := cin, cout := cout, stride := sh, pad := pad
          }
          let srcIsL2 := isVidL2 pt.inputs[0]!
          let extraLut := if lp?.isSome then lutTableBytes else 0
          let shiftBits? := lp?.bind (·.inputShiftBits)
          let scalarBiasMode := cs.effMode == Conv3x3Mode.dense && cs.cout == 1
          let hasEffBias := bias.isSome || shiftBits?.isSome
          let biasInKernel := match weight with
            | .quant _ => bias.isSome
            | .fp16 _ => hasEffBias && !scalarBiasMode
          let qp? := match weight with
            | .quant q => some (quantChunkParamsFor q cs.cout bias.isSome)
            | .fp16 _ => none
          let cp ← chooseConvTaskPlan g.target cs extraLut biasInKernel srcIsL2 resIsL2 liveRanges qp?
          pure (.neConv cp lp? biasInKernel)
        | .neAct kind _ _ => do
          let inSh ← getSh pt.inputs[0]! pt.name
          let inDims : TensorDims := { c := inSh.c, h := pt.outShape.h, w := pt.outShape.w }
          let srcIsL2 := isVidL2 pt.inputs[0]!
          match kind with
          | .act actSpec =>
            let srcFloor32 := actSpec == NeActSpec.copy && n > 1
            let ap ← chooseNeActTaskPlan g.target inDims.c inDims.h inDims.w srcIsL2 resIsL2 liveRanges srcFloor32
            pure (.neAct ap lp?)
          | .channelReduce _ =>
            let fullInDims := shape4DToDims inSh
            let np ← chooseNeChannelReduceTilingForL2 g.target fullInDims srcIsL2 resIsL2 liveRanges
            pure (.neChannelReduce np)
          | .nePool op kh kw sh sw pad excl =>
            let pspec : PoolSpec := {
              target := g.target, op := op, dims := shape4DToDims inSh,
              kh := kh, kw := kw, sh := sh, sw := sw, pad := pad, excludePad := excl
            }
            let (_, gh, gw) ← resolvePoolBackend pspec
            let np ← chooseNePoolTilingForL2 g.target inSh.c gh gw srcIsL2 resIsL2 liveRanges
            pure (.nePool np gh gw)
        | .neMatmul _ m k nCols _ _ _ _ _ => do
          let srcIsL2 := isVidL2 pt.inputs[1]!
          let mp ← chooseMatmulTaskPlan g.target m k nCols srcIsL2 resIsL2 liveRanges
          pure (.neMatmul mp)
        | .dmaView desc _ => do
          let cp := pt.explicitViewCopy?.getD desc.toViewCopy
          let srcIsL2 := isVidL2 pt.inputs[0]!
          let vp ← chooseDmaViewTaskPlan g.target cp srcIsL2 resIsL2 liveRanges
          pure (.dmaView vp cp srcIsL2 resIsL2)
        | .pe peOp _ => do
          match peOp with
          | .unary _ =>
            let outDims := shape4DToDims pt.outShape
            let srcIsL2 := isVidL2 pt.inputs[0]!
            let pp ← choosePeTaskPlan g.target outDims outDims none srcIsL2 false resIsL2 liveRanges
            pure (.pe pp)
          | .binary _ rev =>
            let d1Full ← getSh pt.inputs[0]! pt.name
            let d2Full ← getSh pt.inputs[1]! pt.name
            let outDims := shape4DToDims pt.outShape
            -- Respect strip height when reading surface window on stripped DAG
            let d1 : TensorDims := { c := d1Full.c, h := if pt.src1Win.fullH > 0 then outDims.h else d1Full.h, w := d1Full.w }
            let d2 : TensorDims := { c := d2Full.c, h := if pt.src2Win.fullH > 0 then outDims.h else d2Full.h, w := d2Full.w }
            let (hw1, hw2, v1, v2) := if rev then (d2, d1, pt.inputs[1]!, pt.inputs[0]!) else (d1, d2, pt.inputs[0]!, pt.inputs[1]!)
            let s1L2 := isVidL2 v1
            let s2L2 := isVidL2 v2
            let pp ← choosePeTaskPlan g.target outDims hw1 (some hw2) s1L2 s2L2 resIsL2 liveRanges
            pure (.pe pp)
          | .pool op kh kw sh sw pad excl =>
            let inSh ← getSh pt.inputs[0]! pt.name
            let pspec : PoolSpec := {
              target := g.target, op := op, dims := shape4DToDims inSh,
              kh := kh, kw := kw, sh := sh, sw := sw, pad := pad, excludePad := excl
            }
            let (_, gh, gw) ← resolvePoolBackend pspec
            let srcIsL2 := isVidL2 pt.inputs[0]!
            let pp ← choosePePoolTilingForL2 g.target inSh.c gh gw srcIsL2 resIsL2 liveRanges
            pure (.pePool pp gh gw)
          | .reduce _ axes _ =>
            let inSh ← getSh pt.inputs[0]! pt.name
            let ax := peReduceAxes inSh.c inSh.h axes.reduceC axes.reduceH
            let srcIsL2 := isVidL2 pt.inputs[0]!
            let pr ← choosePeReduceTilingForL2 g.target inSh.w ax srcIsL2 resIsL2 liveRanges
            pure (.peReduce pr)
      match planRes with
      | .ok pl =>
        peakL2 := max peakL2 pl.l2Peak
        plans := plans.push pl
      | .error msg =>
        throw (t, msg)
    pure (plans, peakL2)

  -- Selective L2 <-> DRAM spilling loop (total, bounded by `autoL2.size + 1` iterations!)
  let mut curAutoL2 := autoL2
  let mut placedL2 : List PlacedBuf := []
  let mut taskPlans : Array MachineTilePlan := #[]
  let mut peakL2 : Nat := 0
  let mut solved := false
  let mut lastErr : String := "memory planning failed"

  for _ in [:autoL2.size + 1] do
    let l2Ids := pinnedL2 ++ curAutoL2
    let liveBufs : List LiveBuf := l2Ids.toList.map fun rep => {
      id := rep
      bytes := l2ResidentBytes4D bufShapeOf[rep]!
      first := firstTaskOf[rep]!
      last := lastTaskOf[rep]!
      priority := lastTaskOf[rep]! - firstTaskOf[rep]!
    }
    let trialPlaced := allocateL2Buffers liveBufs
    let l2Top := topOf trialPlaced
    let attempt : Except (Option Nat × String) (Array MachineTilePlan × Nat) :=
      if l2Top > l2CapacityBytes then
        .error (none, s!"live L2-resident buffers reach {l2Top}B > {l2CapacityBytes}B (2 MiB)")
      else
        match tryPlanAll trialPlaced with
        | .ok res => .ok res
        | .error (failTask, msg) => .error (some failTask, msg)

    match attempt with
    | .ok (ps, pk) =>
      placedL2 := trialPlaced
      taskPlans := ps
      peakL2 := pk
      solved := true
      break
    | .error (failTask?, msg) =>
      lastErr := msg
      if curAutoL2.isEmpty then
        if !pinnedL2.isEmpty then
          throw s!"forced L2 placement exceeds 2 MiB capacity: {msg}"
        else
          throw s!"graph compilation failed even with all intermediates in DRAM: {msg}"
      -- Selective spill: pick highest-cost (`bytes * lifetime`, weighted by over-cap hits on aggregate overflow)
      -- among live auto-L2 candidates at `failTask?` so long skips spill first while hot ops stay in L2!
      let liveAuto : Array Nat := match failTask? with
        | some ft => curAutoL2.filter fun rep => firstTaskOf[rep]! <= ft && ft <= lastTaskOf[rep]!
        | none => #[]
      let pool : Array Nat := if liveAuto.isEmpty then curAutoL2 else liveAuto
      let overCapHits : Nat → Nat :=
        if failTask?.isSome then fun _ => 0
        else fun cand => Id.run do
          let mut cnt := 0
          for t in [firstTaskOf[cand]! : lastTaskOf[cand]! + 1] do
            let mut sumB := 0
            for j in l2Ids do
              if firstTaskOf[j]! <= t && t <= lastTaskOf[j]! then
                sumB := sumB + l2ResidentBytes4D bufShapeOf[j]!
            if sumB > l2CapacityBytes then cnt := cnt + 1
          return cnt
      let spillCost := fun (rep : Nat) =>
        let bytes := l2ResidentBytes4D bufShapeOf[rep]!
        let life := max 1 (lastTaskOf[rep]! - firstTaskOf[rep]!)
        bytes * life
      let mut victim := pool[0]!
      let mut bestHits := overCapHits victim
      let mut bestCost := spillCost victim
      for cand in pool do
        let h := overCapHits cand
        let c := spillCost cand
        if h > bestHits || (h == bestHits && (c > bestCost || (c == bestCost && cand > victim))) then
          victim := cand
          bestHits := h
          bestCost := c
      curAutoL2 := curAutoL2.filter fun x => x != victim
      dramIds := dramIds.push victim

  if !solved then
    throw s!"MemoryPlan: failed to fit within 2 MiB L2: {lastErr}"

  -- Pack spilled DRAM intermediates into BAR 3 scratch via `Vane.Alloc.allocate` for interval reuse!
  let dramLiveBufs : List LiveBuf := dramIds.toList.map fun rep => {
    id := rep
    bytes := max (dramScratchBytes4D bufShapeOf[rep]!) bufMinDramOf[rep]!
    first := firstTaskOf[rep]!
    last := lastTaskOf[rep]!
    priority := lastTaskOf[rep]! - firstTaskOf[rep]!
  }
  let (placedScratch, scratchBytes) := allocateScratchBuffers dramLiveBufs

  -- Build SurfaceSpecs: 1..3 input surfaces + SINGLE ABI v3 HWX output surface (`buf_id = 4`)
  let mut inputSurfs : Array SurfaceSpec := Array.mkEmpty g.inputs.size
  for ii in [:g.inputs.size] do
    let (vid, nm, _) := g.inputs[ii]!
    let sh ← getSh vid nm
    let s ← makeFp16Surface g.target inShorts[ii]! s!"input_{nm}" .input sh.c sh.h sh.w sh.n
    inputSurfs := inputSurfs.push s

  let hwxOutSym :=
    if g.outputs.size == 1 then s!"output_{outSlots[0]!.name}"
    else "output_packed"
  let hwxOutSurf ← makeFp16Surface g.target outShorts[0]! hwxOutSym .output
    hwxOutShape.c hwxOutShape.h hwxOutShape.w hwxOutShape.n
  let outputSurfs : Array SurfaceSpec := #[hwxOutSurf]

  -- Verify all packed logical output slots fit cleanly and pairwise non-overlapping inside `hwxOutSurf`
  let hwxOutSpanBytes := hwxOutSurf.batchStride.toNat * hwxOutSurf.n.toNat
  for oi in [:outSlots.size] do
    let sl := outSlots[oi]!
    if sl.byteOffset % dmaLineBytes != 0 || sl.byteOffset + sl.byteSize > hwxOutSpanBytes then
      throw s!"MemoryPlan: packed output[{oi}] [{sl.byteOffset}, {sl.byteOffset + sl.byteSize}) exceeds HWX output surface span {hwxOutSpanBytes}"
    for oj in [oi + 1 : outSlots.size] do
      let sj := outSlots[oj]!
      if sl.byteOffset < sj.byteOffset + sj.byteSize && sj.byteOffset < sl.byteOffset + sl.byteSize then
        throw s!"MemoryPlan: packed outputs {oi} and {oj} overlap in output surface"

  -- Allocate `BufferId`s in `buffers : Array BufferStorage`
  let mut buffers : Array BufferStorage := #[]
  let mut vidToBuf : Array (ValueId × BufferId) := #[]

  -- 1. Input surface buffers (BAR slots `6 .. 6 + inputs.size - 1`)
  let mut inSurfBufIds : Array BufferId := Array.mkEmpty g.inputs.size
  for ii in [:g.inputs.size] do
    let (vid, nm, _) := g.inputs[ii]!
    let sh ← getSh vid nm
    let bid := buffers.size
    buffers := buffers.push (.surface ii .input (barSlotFirstSurface + ii.toUInt8) sh {})
    vidToBuf := vidToBuf.push (vid, bid)
    inSurfBufIds := inSurfBufIds.push bid

  -- 2. Single ABI v3 HWX output surface buffer (BAR slot `6 + inputs.size`)
  let outSurfIdx := g.inputs.size
  let outSurfBufId := buffers.size
  buffers := buffers.push (.surface outSurfIdx .output (barSlotFirstSurface + outSurfIdx.toUInt8) hwxOutShape {})
  if g.outputs.size == 1 && prog.stripCount == 1 && prog.stripHeight == 0 then
    vidToBuf := vidToBuf.push (g.outputs[0]!, outSurfBufId)

  -- 3. Intermediate L2 and BAR 3 DRAM scratch buffers (one per representative `rep`)
  let mut repBufIds : Array (Option BufferId) := Array.replicate n none
  for rep in [:n] do
    if !isRepBuf[rep]! then continue
    let sh := bufShapeOf[rep]!
    let bid := buffers.size
    match lookupPlaced placedL2 rep with
    | some pb =>
      buffers := buffers.push (.l2 pb.base pb.bytes sh)
    | none =>
      match lookupPlaced placedScratch rep with
      | some pb =>
        buffers := buffers.push (.dramScratch pb.base pb.bytes sh)
      | none =>
        throw s!"internal error: representative buffer {rep} missing from L2 and scratch placements"
    repBufIds := repBufIds.set! rep (some bid)
    let keyVid := prepTasks[rep]!.parentVid?.getD prepTasks[rep]!.outVid
    vidToBuf := vidToBuf.push (keyVid, bid)

  -- 4. Zero-copy read subviews (contiguous C-axis slice aliases!)
  for (slVid, parentInVid, byteOff, slSh) in readSubviewOf do
    let some parentBid := lookupAssoc vidToBuf parentInVid
      | throw s!"internal error: read subview parent %{parentInVid} missing buffer"
    let bid := buffers.size
    buffers := buffers.push (.subview parentBid byteOff slSh {})
    vidToBuf := vidToBuf.push (slVid, bid)

  -- 5. Build `MachineTask`s with explicit input/output `BufferId`s (using zero-copy subviews for strips, SOK, & packed multi-outputs!)
  let mut machineTasks : Array MachineTask := Array.mkEmpty n
  for ti in [:n] do
    let pt := prepTasks[ti]!
    let pl := taskPlans[ti]!

    -- Resolve input BufferIds (adding zero-copy SurfaceWindow subviews when reading a strip band of an input surface)
    let mut inBufs : Array BufferId := Array.mkEmpty pt.inputs.size
    for slot in [:pt.inputs.size] do
      let iv := pt.inputs[slot]!
      let some baseBid := lookupAssoc vidToBuf iv
        | throw s!"internal error: task '{pt.name}' input %{iv} has no allocated BufferId"
      let win := if slot == 0 then pt.src1Win else if slot == 1 then pt.src2Win else {}
      if win.fullH > 0 then
        let inSh ← getSh iv pt.name
        let winH := match pt.op with
          | .neConv kh _ _ _ (.strip ptop pbot) .. =>
            if kh == 3 then pt.outShape.h + 2 - ptop - pbot else pt.outShape.h
          | _ => pt.outShape.h
        let winSh : Shape4D := { inSh with h := winH }
        let subBid := buffers.size
        buffers := buffers.push (.subview baseBid 0 winSh win)
        inBufs := inBufs.push subBid
      else
        inBufs := inBufs.push baseBid

    -- Resolve output BufferId (single HWX output surface, subview into packed output surface, or intermediate L2/DRAM)
    let outBid ← match taskOutSurfIdx[ti]! with
      | some _ =>
        if pt.subviewByteOff > 0 || pt.resWin.fullH > 0 then
          let subBid := buffers.size
          buffers := buffers.push (.subview outSurfBufId pt.subviewByteOff pt.outShape pt.resWin)
          pure subBid
        else
          pure outSurfBufId
      | none =>
        let rep := repOfTask[ti]!
        let some parBid := repBufIds[rep]!
          | throw s!"internal error: task '{pt.name}' missing intermediate BufferId for rep {rep}"
        if pt.subviewByteOff > 0 then
          let subBid := buffers.size
          buffers := buffers.push (.subview parBid pt.subviewByteOff pt.outShape {})
          pure subBid
        else
          pure parBid

    let (eng, isNeComp, isMb) : Engine × Bool × Bool := match pt.op, pl with
      | .neConv .., _ => (.ne, true, false)
      | .neAct (.act (.lut _)) _ _, _ => (.ne, true, false)
      | .neAct .., _ => (.ne, false, false)
      | .neMatmul heads .., _ => (.ne, true, heads > 1)
      | .dmaView .., .dmaView _ cp _ _ => (.ne, false, cp.n > 1)
      | .dmaView desc _, _ => (.ne, false, desc.batchN > 1)
      | .pe (.reduce _ axes _) _, _ =>
        let inSh := (lookupAssoc shMap pt.inputs[0]!).getD pt.outShape
        (.pe, false, (peReduceAxes inSh.c inSh.h axes.reduceC axes.reduceH).sN > 1)
      | .pe .., _ => (.pe, false, false)

    -- Compute header[7] liveL2Mask (exact 64 KiB blocks for all L2 buffers live at task `ti`)
    let mut l2Mask : UInt32 := 0
    if ti + 1 < n then
      for r in liveAt placedL2 ti do
        l2Mask := l2Mask ||| l2BlockMask r.base r.hi

    -- Record task temporary L2 staging ranges for ALL MachineTilePlan variants (self-describing MachineIR!)
    let src1IsL2Buf := isBufferL2 buffers inBufs[0]!
    let resIsL2Buf := isBufferL2 buffers outBid
    let tempL2 : List Range := match pl with
      | .neConv cp _ _ =>
        let rSrc := if !src1IsL2Buf then [{ base := cp.srcStageBase, bytes := alignUp (srcL2BytesConv cp.shape cp.tiling) dmaLineBytes }] else []
        let rRes := if !resIsL2Buf then
          let rsz := if cp.direct then ((1 : Nat) <<< (cp.tiling.f + effOcgConv cp.shape cp.tiling + cp.tiling.fat)) * 512
                     else cp.tiling.tileHeight * resRowStrideConv cp.shape cp.tiling
          [{ base := cp.resStageBase, bytes := alignUp rsz dmaLineBytes }]
        else []
        rSrc ++ rRes
      | .neAct ap _ =>
        let stripW := ap.tiling.stripWidth
        let effSrcStrip := match pt.op with | .neAct (.act .copy) _ _ => if n > 1 then max stripW 32 else stripW | _ => stripW
        let u := pt.outShape.c * (2 * min (alignUp pt.outShape.w 8) effSrcStrip)
        let rSrc := if !src1IsL2Buf then [{ base := ap.srcStageBase, bytes := alignUp (ap.tiling.tileHeight * u) dmaLineBytes }] else []
        let du := pt.outShape.c * (2 * min (alignUp pt.outShape.w 8) (max stripW 32))
        let rRes := if !resIsL2Buf then
          let rsz := if ap.direct then neActDirectResBytes ap.tiling.f ap.tiling.fat else ap.tiling.tileHeight * du
          [{ base := ap.resStageBase, bytes := alignUp rsz dmaLineBytes }]
        else []
        rSrc ++ rRes
      | .nePool np gh gw =>
        let inSh := buffers[inBufs[0]!]!.shape
        let stripW := np.stripWidth
        let px := poolStripPx gw.used gw.k gw.sEff gw.padBefore stripW gw.out
        let rSrc := if !src1IsL2Buf then
          [{ base := np.srcStageBase, bytes := alignUp (poolInRows gh.used gh.k gh.sEff gh.padBefore gh.out np.tileHeight true * (inSh.c * 2 * px)) dmaLineBytes }]
        else []
        let drow := 2 * min (alignUp gw.out 8) (max stripW 32)
        let rRes := if !resIsL2Buf then
          let rsz := if np.direct then neActDirectResBytes np.f np.fat else alignUp (np.tileHeight * inSh.c * drow) dmaLineBytes
          [{ base := np.resStageBase, bytes := alignUp rsz dmaLineBytes }]
        else []
        rSrc ++ rRes
      | .neChannelReduce nc =>
        let inSh := buffers[inBufs[0]!]!.shape
        let stripW := nc.stripWidth
        let px := min (alignUp inSh.w 8) stripW
        let rSrc := if !src1IsL2Buf then [{ base := nc.srcStageBase, bytes := alignUp (nc.tileHeight * inSh.c * 2 * px) dmaLineBytes }] else []
        let drow := 2 * min (alignUp inSh.w 8) (max stripW 32)
        let rRes := if !resIsL2Buf then
          let rsz := if nc.direct then neActDirectResBytes nc.f nc.fat else alignUp (nc.tileHeight * drow) dmaLineBytes
          [{ base := nc.resStageBase, bytes := alignUp rsz dmaLineBytes }]
        else []
        rSrc ++ rRes
      | .neMatmul mp =>
        match pt.op with
        | .neMatmul _ m k nCols .. =>
          let stripW := (128 : Nat) >>> mp.tiling.shPref
          let bIsL2 := isBufferL2 buffers inBufs[1]!
          let rSrc := if !bIsL2 then [{ base := mp.srcStageBase, bytes := alignUp (k * (2 * min (alignUp nCols 8) (max stripW 32))) dmaLineBytes }] else []
          let dchs := 2 * min (alignUp nCols 8) (max stripW 32)
          let rRes := if !resIsL2Buf then
            let rsz := if mp.direct then ((1 : Nat) <<< mp.tiling.f) * 512 else m * dchs
            [{ base := mp.resStageBase, bytes := alignUp rsz dmaLineBytes }]
          else []
          rSrc ++ rRes
        | _ => []
      | .dmaView vp cp _ _ =>
        let rSrc := if !src1IsL2Buf then [{ base := vp.srcStageBase, bytes := alignUp (viewSrcL2Bytes cp vp.tiling) dmaLineBytes }] else []
        let rRes := if !resIsL2Buf then [{ base := vp.resStageBase, bytes := alignUp (viewResL2Bytes g.target cp vp.tiling) dmaLineBytes }] else []
        rSrc ++ rRes
      | .pe pp =>
        let effStripW := max (peStripWidth pp.tiling.k) 32
        let th := pp.tiling.tileHeight
        let whole1 := pp.hwSrc2?.isSome && pp.hwSrc1.c == 1 && pp.hwSrc1.h == 1 && pp.hwSrc1.w > 1 && pp.outDims.h > 1 && th < pp.outDims.h
        let row1 := 2 * (if whole1 then alignUp pp.hwSrc1.w 8 else min (alignUp pp.hwSrc1.w 8) effStripW)
        let sz1 := (if pp.hwSrc1.h > 1 then th else 1) * (pp.hwSrc1.c * row1)
        let rSrc1 := if !src1IsL2Buf then [{ base := pp.src1StageBase, bytes := alignUp sz1 dmaLineBytes }] else []
        let rSrc2 := match pp.hwSrc2?, inBufs[1]? with
          | some hw2, some b2 =>
            if !isBufferL2 buffers b2 then
              let whole2 := hw2.c == 1 && hw2.h == 1 && hw2.w > 1 && pp.outDims.h > 1 && th < pp.outDims.h
              let row2 := 2 * (if whole2 then alignUp hw2.w 8 else min (alignUp hw2.w 8) effStripW)
              let sz2 := (if hw2.h > 1 then th else 1) * (hw2.c * row2)
              [{ base := pp.src2StageBase, bytes := alignUp sz2 dmaLineBytes }]
            else []
          | _, _ => []
        let drow := 2 * min (alignUp pp.outDims.w 8) effStripW
        let rRes := if !resIsL2Buf then
          let rsz := if pp.direct then peDirectResBytes pp.outDims.c drow th else th * (pp.outDims.c * drow)
          [{ base := pp.resStageBase, bytes := alignUp rsz dmaLineBytes }]
        else []
        rSrc1 ++ rSrc2 ++ rRes
      | .pePool pp gh gw =>
        let inSh := buffers[inBufs[0]!]!.shape
        let stripW := (1 : Nat) <<< pp.k
        let px := poolStripPx gw.used gw.k gw.sEff gw.padBefore stripW gw.out
        let rSrc := if !src1IsL2Buf then
          [{ base := pp.srcStageBase, bytes := alignUp (poolInRows gh.used gh.k gh.sEff gh.padBefore gh.out pp.tileHeight false * (inSh.c * 2 * px)) dmaLineBytes }]
        else []
        let drow := 2 * min (alignUp gw.out 8) (max stripW 32)
        let rRes := if !resIsL2Buf then
          let rsz := if pp.direct then peDirectResBytes inSh.c drow pp.tileHeight else alignUp (pp.tileHeight * inSh.c * drow) dmaLineBytes
          [{ base := pp.resStageBase, bytes := alignUp rsz dmaLineBytes }]
        else []
        rSrc ++ rRes
      | .peReduce pr =>
        let inSh := buffers[inBufs[0]!]!.shape
        let ax := match pt.op with
          | .pe (.reduce _ axes _) _ => peReduceAxes inSh.c inSh.h axes.reduceC axes.reduceH
          | _ => peReduceAxes inSh.c inSh.h false false
        let stripW := (1 : Nat) <<< pr.k
        let px := min (alignUp inSh.w 8) (max stripW 32)
        let rSrc := if !src1IsL2Buf then [{ base := pr.srcStageBase, bytes := alignUp (pr.tileHeight * ax.sC * ax.sD * 2 * px) dmaLineBytes }] else []
        let rRes := if !resIsL2Buf then [{ base := pr.resStageBase, bytes := 2048 }] else []
        rSrc ++ rRes

    machineTasks := machineTasks.push {
      id := ti
      name := pt.name
      logicalId := pt.logicalId
      opKind := pt.op.opKindLabel
      engine := eng
      isNeCompute := isNeComp
      isMultiBatch := isMb
      op := pt.op
      tilePlan := pl
      inputs := inBufs
      output := outBid
      outShape := pt.outShape
      tempL2 := tempL2
      liveL2Mask := l2Mask
    }

  -- 6. Validate optional buffer donations with strict liveness safety
  let donations ← validateDonations prog.donations inputSurfs outSlots hwxOutSpanBytes
    inSurfBufIds outSurfBufId buffers machineTasks taskOutSurfIdx

  let witness : MemoryPlanWitness := {
    l2Placed := placedL2
    scratchPlaced := placedScratch
    l2Peak := peakL2
    l2Capacity := l2CapacityBytes
  }
  if !witness.l2WithinCapacity then
    throw s!"MemoryPlan witness violation: l2Peak={peakL2}B exceeds {l2CapacityBytes}B"

  pure {
    target := g.target
    inputSurfaces := inputSurfs
    outputSurfaces := outputSurfs
    outputs := outSlots
    buffers := buffers
    tasks := machineTasks
    kernelBytes := 0
    scratchBytes := scratchBytes
    l2PeakBytes := peakL2
    stripCount := prog.stripCount
    stripHeight := prog.stripHeight
    donations := donations
    witness := witness
  }

end Vane.Backend.MemoryPlan
