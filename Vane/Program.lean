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

/-- High-level single-task descriptor reusable across NE conv, PE ops, and multi-task graphs. -/
structure TaskDesc where
  taskId : Nat := 0
  logNE : Nat
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
    (16 KiB rounded on H16, exact span on H17). -/
def makeFp16Surface
    (target : Target)
    (shortName symbolName : String)
    (direction : SurfaceDirection)
    (channels height width : Nat) : SurfaceSpec :=
  let rowStride := alignUp (2 * width) dmaLineBytes
  let planeStride := height * rowStride
  let span := channels * planeStride
  let alloc := alignUp span pageSize
  let sectSize : UInt64 := match target with
    | .h16 => alloc.toUInt64
    | .h17 => span.toUInt64
  {
    shortName := shortName
    symbolName := symbolName
    direction := direction
    n := 1
    channels := channels.toUInt32
    height := height.toUInt32
    width := width.toUInt32
    rowStride := rowStride.toUInt64
    planeStride := planeStride.toUInt64
    batchStride := span.toUInt64
    sectionSize := some sectSize
    allocSize := some alloc.toUInt64
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
    let plainRecs ← packPlainWrites td.plainWrites
    let allRecs ← packTaskRecords td.plainWrites td.barWrites
    let totalTaskWords := 9 + recordWordsCount allRecs
    let header ← encodeTaskHeader9 td.taskId totalTaskWords td.logNE
    tasks := tasks.push { header := header, records := allRecs }
    for rel in td.kernelRelocs do
      if rel.barRecIndex >= td.barWrites.size then
        throw s!"emitProgram: task {ti} relocation barRecIndex {rel.barRecIndex} >= {td.barWrites.size}"
      relocTargets := relocTargets.push (ti, plainRecs.size + rel.barRecIndex, rel.payloadWordIndex)

  let prologue : UInt32 := match spec.target with
    | .h16 => 1
    | .h17 => 0
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
    relocs := relocs.push { patchOffset := patchOff.toUInt32 }

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
