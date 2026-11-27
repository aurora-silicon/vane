import Vane.Binary
import Vane.TaskStream
import Vane.Hwx
import Vane.Alloc
import Vane.Tiling
import Vane.Verify
import Vane.Pe
import Vane.Lut
import Vane.Pool
import Vane.Conv
import Vane.View
import Vane.Matmul
import Vane.Ane.Types

namespace Vane.Machine

open Vane.Binary
open Vane.TaskStream
open Vane.Hwx
open Vane.Alloc
open Vane.Tiling
open Vane.Verify
open Vane.Pe
open Vane.Lut
open Vane.Pool
open Vane.Conv
open Vane.View
open Vane.Matmul
open Vane.Ane

-- ============================================================================
-- L3 Bufferized Machine IR (`BufferStorage`, `MachineTask`, `MachineProgram`)
-- ============================================================================

/-- Dense buffer identifier in `MachineProgram.buffers`. -/
abbrev BufferId := Nat

/-- Optional horizontal row-band window on a full-frame network surface (`fullH == 0` means full frame). -/
structure SurfaceWindow where
  rowOffset : Nat := 0
  fullH : Nat := 0
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Compute `(baseOff, rowStride, planeStride, exactExtent)` for a `P = 1` surface DMA access. -/
def surfaceDmaWindow (win : SurfaceWindow) (c h w : Nat) : Nat × Nat × Nat × Nat :=
  let effFullH := if win.fullH == 0 then h else win.fullH
  let rowStride := alignUp (2 * w) 64
  let planeStride := effFullH * rowStride
  let baseOff := win.rowOffset * rowStride
  let extent := if c == 0 || h == 0 then 0 else (c - 1) * planeStride + h * rowStride
  (baseOff, rowStride, planeStride, extent)

/-- Explicit physical memory space and optional zero-copy subview aliasing for a buffer. -/
inductive BufferStorage where
  | surface (surfIdx : Nat) (dir : SurfaceDirection) (barSlot : UInt8) (shape : Shape4D) (window : SurfaceWindow := {})
  | l2 (base : Nat) (bytes : Nat) (shape : Shape4D)
  | dramScratch (offset : Nat) (bytes : Nat) (shape : Shape4D)
  | subview (parent : BufferId) (byteOffset : Nat) (shape : Shape4D) (window : SurfaceWindow := {})
  deriving Repr, BEq, Inhabited

namespace BufferStorage

def shape : BufferStorage → Shape4D
  | .surface _ _ _ sh _ => sh
  | .l2 _ _ sh => sh
  | .dramScratch _ _ sh => sh
  | .subview _ _ sh _ => sh

def pretty : BufferStorage → String
  | .surface idx dir slot sh win =>
    let wStr := if win.fullH > 0 then s!",rows={win.rowOffset}..{win.rowOffset + sh.h}/{win.fullH}" else ""
    s!"@Surface(idx={idx},{dir.name},bar={slot.toNat},shape={sh.toArray.toList}{wStr})"
  | .l2 base bytes sh =>
    s!"@L2(base={toHex base},bytes={bytes},shape={sh.toArray.toList})"
  | .dramScratch off bytes sh =>
    s!"@Scratch(off={toHex off},bytes={bytes},shape={sh.toArray.toList})"
  | .subview parent off sh win =>
    let wStr := if win.fullH > 0 then s!",rows={win.rowOffset}..{win.rowOffset + sh.h}/{win.fullH}" else ""
    s!"@Subview(parent=%b{parent},off={off},shape={sh.toArray.toList}{wStr})"

end BufferStorage

/-- Resolved physical root storage behind any `BufferId` (following zero-copy `subview` chains). -/
inductive ResolvedStorage where
  | surface (surfIdx : Nat) (dir : SurfaceDirection) (barSlot : UInt8) (byteOffset : Nat) (window : SurfaceWindow) (rootShape : Shape4D)
  | l2 (base : Nat) (bytes : Nat)
  | dramScratch (offset : Nat) (bytes : Nat)
  deriving Repr, BEq, Inhabited

/-- Fuel-bounded total resolution of `BufferStorage.subview` chains to concrete physical storage. -/
def resolveBufferStorageAux (buffers : Array BufferStorage) (fuel : Nat) (bid : BufferId)
    (accOff : Nat) (winOverride : SurfaceWindow) : Except String ResolvedStorage :=
  match fuel with
  | 0 => .error s!"buffer %{bid}: cyclic subview alias chain"
  | fuel' + 1 =>
    match buffers[bid]? with
    | none => .error s!"buffer %{bid}: out of bounds (buffer count = {buffers.size})"
    | some (.surface idx dir slot sh win) =>
      let effWin := if winOverride.fullH > 0 then winOverride else win
      .ok (.surface idx dir slot accOff effWin sh)
    | some (.l2 base bytes _) =>
      if accOff != 0 then
        .error s!"buffer %{bid}: non-zero subview offset {accOff} into L2 buffer is unsupported"
      else
        .ok (.l2 base bytes)
    | some (.dramScratch off bytes _) =>
      .ok (.dramScratch (off + accOff) bytes)
    | some (.subview parent subOff _ subWin) =>
      let effWin := if winOverride.fullH > 0 then winOverride else subWin
      resolveBufferStorageAux buffers fuel' parent (accOff + subOff) effWin

/-- Resolve a `BufferId` through zero-copy `subview` aliases to its concrete physical storage. -/
def resolveBufferStorage (buffers : Array BufferStorage) (bid : BufferId) : Except String ResolvedStorage :=
  resolveBufferStorageAux buffers (buffers.size + 1) bid 0 {}

/-- Return `true` iff `bid` resolves to an L2-resident buffer. -/
def isBufferL2 (buffers : Array BufferStorage) (bid : BufferId) : Bool :=
  match resolveBufferStorage buffers bid with
  | .ok (.l2 _ _) => true
  | _ => false

/-- Logical output tensor slice packed inside the single ABI v3 HWX output surface (`buf_id = 4`). -/
structure OutputSlot where
  name : String
  shape : Shape4D
  byteOffset : Nat
  byteSize : Nat
  rowStride : Nat
  planeStride : Nat
  batchStride : Nat := 0
  deriving Repr, BEq, Inhabited

/-- Optional buffer donation record pairing an input surface with a matching logical output. -/
structure BufferDonation where
  inputSurfIdx : Nat
  outputSurfIdx : Nat
  inputBuf : BufferId
  outputBuf : BufferId
  lastReadTask : Nat := 0
  firstWriteTask : Nat := 0
  deriving Repr, BEq, Inhabited

-- ============================================================================
-- Concrete Per-Task Hardware Tiling & Staging Plans
-- ============================================================================

/-- Tiling & L2 staging plan for `AneOp.neConv`. -/
structure ConvTaskPlan where
  shape : ConvSpec
  tiling : Tiling
  direct : Bool
  srcStageBase : Nat
  resStageBase : Nat
  l2TaskPeak : Nat
  deriving Repr, BEq, Inhabited

/-- Tiling & L2 staging plan for `AneOp.pe` unary/binary. -/
structure PeTaskPlan where
  outDims : TensorDims
  hwSrc1 : TensorDims
  hwSrc2? : Option TensorDims
  tiling : PeTiling
  direct : Bool
  src1StageBase : Nat
  src2StageBase : Nat
  resStageBase : Nat
  l2TaskPeak : Nat
  deriving Repr, BEq, Inhabited

/-- Tiling & L2 staging plan for `AneOp.neAct` (`relu`, `copy`, `lut`). -/
structure NeActTaskPlan where
  tiling : NeActTiling
  direct : Bool
  srcStageBase : Nat
  resStageBase : Nat
  l2TaskPeak : Nat
  deriving Repr, BEq, Inhabited

/-- Concrete hardware task plan for each of the 5 `AneOp` families. -/
inductive MachineTilePlan where
  | neConv (plan : ConvTaskPlan) (lutPayload? : Option LutPayload) (biasInKernel : Bool)
  | neAct (plan : NeActTaskPlan) (lutPayload? : Option LutPayload)
  | nePool (plan : NeWeightlessPlan) (gh gw : PoolAxisGeom)
  | neChannelReduce (plan : NeWeightlessPlan)
  | neMatmul (plan : MatmulTaskPlan)
  | dmaView (plan : ViewTaskPlan) (copy : ViewCopy) (srcIsL2 resIsL2 : Bool)
  | pe (plan : PeTaskPlan)
  | pePool (plan : PePoolPlan) (gh gw : PoolAxisGeom)
  | peReduce (plan : PeReducePlan)
  deriving Inhabited

namespace MachineTilePlan

def direct : MachineTilePlan → Bool
  | .neConv p _ _ => p.direct
  | .neAct p _ => p.direct
  | .nePool p _ _ => p.direct
  | .neChannelReduce p => p.direct
  | .neMatmul p => p.direct
  | .dmaView p _ _ _ => p.direct
  | .pe p => p.direct
  | .pePool p _ _ => p.direct
  | .peReduce p => p.direct

def l2Peak : MachineTilePlan → Nat
  | .neConv p _ _ => p.l2TaskPeak
  | .neAct p _ => p.l2TaskPeak
  | .nePool p _ _ => p.l2TaskPeak
  | .neChannelReduce p => p.l2TaskPeak
  | .neMatmul p => p.l2TaskPeak
  | .dmaView p _ _ _ => p.l2TaskPeak
  | .pe p => p.l2TaskPeak
  | .pePool p _ _ => p.l2TaskPeak
  | .peReduce p => p.l2TaskPeak

end MachineTilePlan

/-- One bufferized hardware task in `MachineProgram`. -/
structure MachineTask where
  id : Nat
  name : String
  logicalId : String := ""
  opKind : String
  engine : Engine
  isNeCompute : Bool
  isMultiBatch : Bool := false
  op : AneOp
  tilePlan : MachineTilePlan
  inputs : Array BufferId
  output : BufferId
  outShape : Shape4D
  tempL2 : List Range := []
  liveL2Mask : UInt32 := 0
  sync : TaskSyncInfo := {}
  extProducerTask : Option Nat := none
  deriving Inhabited

/-- Explicit auditable memory-plan witness produced by `Backend/MemoryPlan.lean`. -/
structure MemoryPlanWitness where
  l2Placed : List PlacedBuf
  scratchPlaced : List PlacedBuf
  l2Peak : Nat
  l2Capacity : Nat := l2CapacityBytes
  deriving Repr, BEq, Inhabited

def MemoryPlanWitness.l2WithinCapacity (w : MemoryPlanWitness) : Bool :=
  w.l2Peak <= w.l2Capacity && topOf w.l2Placed <= w.l2Capacity

/-- Complete bufferized L3 `MachineIR` program.
    `outputSurfaces` always contains the single ABI v3 HWX output surface (`buf_id = 4`);
    `outputs` describes all `1..N` logical output tensors packed inside it at 64B-aligned byte offsets. -/
structure MachineProgram where
  target : Target
  inputSurfaces : Array SurfaceSpec
  outputSurfaces : Array SurfaceSpec
  outputs : Array OutputSlot := #[]
  buffers : Array BufferStorage
  tasks : Array MachineTask
  kernelBytes : Nat := 0
  scratchBytes : Nat := 0
  l2PeakBytes : Nat := 0
  stripCount : Nat := 1
  stripHeight : Nat := 0
  donations : Array BufferDonation := #[]
  witness : MemoryPlanWitness := { l2Placed := [], scratchPlaced := [], l2Peak := 0 }

namespace MachineProgram

def allSurfaces (p : MachineProgram) : Array SurfaceSpec :=
  p.inputSurfaces ++ p.outputSurfaces

/-- Render a human-readable textual dump of `MachineProgram` (`--dump-ir=machine`). -/
def pretty (p : MachineProgram) : String := Id.run do
  let mut lines : Array String := #[]
  let logicalOutCount := if p.outputs.isEmpty then p.outputSurfaces.size else p.outputs.size
  lines := lines.push (
    s!"machine_program (target={p.target.name}, inputs={p.inputSurfaces.size}, " ++
    s!"outputs={logicalOutCount}, hwx_out_surfs={p.outputSurfaces.size}, tasks={p.tasks.size}, " ++
    s!"l2_peak={p.l2PeakBytes}B, scratch={p.scratchBytes}B, kernel={p.kernelBytes}B) \{"
  )
  if p.outputs.size > 1 then
    lines := lines.push "  // Packed Logical Outputs (in single ABI v3 HWX output surface):"
    for oi in [:p.outputs.size] do
      let o := p.outputs[oi]!
      let bNote := if o.shape.n > 1 then s!" batchSt={o.batchStride}B" else ""
      lines := lines.push s!"  out[{oi}] \"{o.name}\" shape={o.shape.toArray.toList} off={o.byteOffset}B bytes={o.byteSize}B rowSt={o.rowStride}B planeSt={o.planeStride}B{bNote}"
  if !p.donations.isEmpty then
    for d in p.donations do
      lines := lines.push s!"  donate in_surf[{d.inputSurfIdx}](%b{d.inputBuf}) -> out[{d.outputSurfIdx}](%b{d.outputBuf}) [lastRead=t{d.lastReadTask} < firstWrite=t{d.firstWriteTask}]"
  lines := lines.push "  // Buffers:"
  for i in [:p.buffers.size] do
    lines := lines.push s!"  %b{i} = {p.buffers[i]!.pretty}"
  lines := lines.push "  // Tasks:"
  for t in p.tasks do
    let ins := String.intercalate ", " (t.inputs.toList.map fun b => s!"%b{b}")
    let eng := match t.engine with | .ne => "NE" | .pe => "PE"
    let syncStr :=
      s!"w5={t.sync.src1WaitNe5},w22={t.sync.src1WaitPe22},dk1={t.sync.src1DepKind}/{t.sync.src1DepDist},l2m={toHex t.liveL2Mask.toNat}"
    lines := lines.push s!"  task[{t.id}] ({eng}) %b{t.output} = {t.op.pretty}({ins}) [{syncStr}] // {t.name}"
  lines := lines.push (
    s!"  // Witness: l2_placed={p.witness.l2Placed.length}, scratch_placed={p.witness.scratchPlaced.length}, " ++
    s!"l2_peak={p.witness.l2Peak}B <= {p.witness.l2Capacity}B ({p.witness.l2WithinCapacity})"
  )
  lines := lines.push "}"
  return String.intercalate "\n" lines.toList

end MachineProgram

end Vane.Machine
