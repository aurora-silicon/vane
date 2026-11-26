import Vane.Binary
import Vane.TaskStream
import Vane.Hwx
import Vane.Sections
import Vane.Register

namespace Vane.Program

open Vane.Binary
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register

/-- Size of `__TEXT,__const` for compiled programs (16 KiB of zero bytes). -/
def textConstBytes : Nat := 0x4000

/-- DMA cache-line alignment in bytes for surface row strides and kernel chunks. -/
def dmaLineBytes : Nat := 64

/-- Identifies a BAR-record payload word inside a `TaskDesc` that requires a Mach-O
    `__KERN_0` relocation (`barRecIndex` indexes into `TaskDesc.barWrites`). -/
structure TaskReloc where
  barRecIndex : Nat
  payloadWordIndex : Nat := 0
  deriving Repr, BEq, Inhabited

/-- High-level task descriptor reusable across NE conv, NE act, PE ops, and multi-task graphs. -/
structure TaskDesc where
  taskId : Nat := 0
  logNE : Nat := 0
  eneOverride : Option Nat := none
  header4 : UInt32 := taskHeaderWord4Const
  header4Override : Option UInt32 := none
  isFirst : Bool := true
  isLast : Bool := true
  liveL2Mask : UInt32 := 0
  plainWrites : Array RegWrite
  barWrites : Array BarRegWrite
  kernelRelocs : Array TaskReloc := #[]
  deriving Repr, BEq, Inhabited

/-- Generic multi-task ANE program specification. -/
structure ProgramSpec where
  target : Target
  surfaces : Array SurfaceSpec
  tasks : Array TaskDesc
  kernel : Option ByteArray := none
  scratch : Option ScratchSpec := none
  constants : ByteArray := zeroBytes textConstBytes
  ident : Option String := some "vane"
  deriving BEq

/-- Emitted HWX container, validated container metadata, and 4 Linux LOAD sections. -/
structure EmittedProgram where
  hwx : ByteArray
  info : HwxInfo
  sections : LoadSections

/-- Build a 64-byte row-strided fp16 `SurfaceSpec` with target-appropriate section sizing
    (16 KiB rounded on H16, exact span on H17), checking all `Nat -> UInt32/UInt64` conversions. -/
def makeFp16Surface
    (target : Target)
    (shortName symbolName : String)
    (direction : SurfaceDirection)
    (channels height width : Nat)
    (n : Nat := 1) : Except String SurfaceSpec := do
  if n == 0 || channels == 0 || height == 0 || width == 0 then
    throw s!"surface '{shortName}': dimensions (N={n}, C={channels}, H={height}, W={width}) must all be >= 1"
  let nU32 ← natToU32Checked n s!"surface '{shortName}' batch N"
  let cU32 ← natToU32Checked channels s!"surface '{shortName}' channels"
  let hU32 ← natToU32Checked height s!"surface '{shortName}' height"
  let wU32 ← natToU32Checked width s!"surface '{shortName}' width"
  let rowStride := alignUp (2 * width) dmaLineBytes
  let planeStride := height * rowStride
  let batchStride := channels * planeStride
  let span := n * batchStride
  let alloc := alignUp span pageSize
  let rowStrideU64 ← natToU64Checked rowStride s!"surface '{shortName}' rowStride"
  let planeStrideU64 ← natToU64Checked planeStride s!"surface '{shortName}' planeStride"
  let batchStrideU64 ← natToU64Checked batchStride s!"surface '{shortName}' batchStride"
  let spanU64 ← natToU64Checked span s!"surface '{shortName}' span"
  let allocU64 ← natToU64Checked alloc s!"surface '{shortName}' allocSize"
  let sectSize : UInt64 := if target.profile.fvmlibUseSpanSize then spanU64 else allocU64
  pure {
    shortName := shortName
    symbolName := symbolName
    direction := direction
    n := nU32
    channels := cU32
    height := hU32
    width := wU32
    rowStride := rowStrideU64
    planeStride := planeStrideU64
    batchStride := batchStrideU64
    sectionSize := some sectSize
    allocSize := some allocU64
  }

/-- Emit a complete HWX Mach-O binary and its 4 Linux LOAD sections from a generic `ProgramSpec`,
    self-validating both the HWX container (`validateHwx`) and the derived LOAD sections. -/
def emitProgram (spec : ProgramSpec) : Except String EmittedProgram := do
  if spec.tasks.isEmpty then
    throw "emitProgram: program must contain at least 1 task"

  let mut tasks : Array Task := #[]
  let mut relocTargets : Array (Nat × Nat × Nat) := #[]

  for ti in [:spec.tasks.size] do
    let td := spec.tasks[ti]!
    let (plainRecs, barRecs) ← packTaskRecordsSplit td.plainWrites td.barWrites
    let allRecs := plainRecs ++ barRecs
    let totalTaskWords := 9 + recordWordsCount allRecs
    let ene ← match td.eneOverride with
      | some e => pure e
      | none =>
        if td.logNE > 4 then
          throw s!"task header: logNE {td.logNE} exceeds maximum 4"
        pure (td.logNE + 1)
    let h4Override : Option UInt32 :=
      match td.header4Override with
      | some v => some v
      | none =>
        if td.isFirst && td.isLast && td.header4 != taskHeaderWord4Const then
          some td.header4
        else
          none
    let header ← encodeTaskHeader9Multi spec.target td.taskId totalTaskWords ene
      td.isFirst td.isLast td.liveL2Mask h4Override
    tasks := tasks.push { header := header, records := allRecs }
    for rel in td.kernelRelocs do
      if rel.barRecIndex >= td.barWrites.size then
        throw s!"emitProgram: task {ti} relocation barRecIndex {rel.barRecIndex} >= {td.barWrites.size}"
      relocTargets := relocTargets.push (ti, plainRecs.size + rel.barRecIndex, rel.payloadWordIndex)

  let prologue : UInt32 := spec.target.profile.defaultPrologue
  let stream : TaskStream := {
    prologue := prologue
    tasks := tasks
    trailingZeroBlocks := 0
  }

  let encStream ← encodeTaskStream spec.target stream
  let mut relocs : Array RelocationSpec := #[]
  for (ti, ri, pi) in relocTargets do
    let some (_, _, _, patchOff) :=
      encStream.payloadSites.find? (fun (tIdx, rIdx, pIdx, _) => tIdx == ti && rIdx == ri && pIdx == pi)
      | throw s!"emitProgram: failed to locate relocation payload offset for task {ti}, record {ri}, word {pi}"
    let patchOffU32 ← natToU32Checked patchOff s!"emitProgram task {ti} relocation patchOffset"
    relocs := relocs.push { patchOffset := patchOffU32 }

  let prog : HwxProgram := {
    target := spec.target
    surfaces := spec.surfaces
    constants := spec.constants
    kernel := spec.kernel
    scratch := spec.scratch
    taskStream := stream
    relocations := relocs
    ident := spec.ident
  }

  let hwxBytes ← emitHwx prog
  let info ← validateHwx hwxBytes
  let secs ← buildSectionsFromInfo info
  validateSectionsWithInfo info secs
  pure { hwx := hwxBytes, info := info, sections := secs }

end Vane.Program
