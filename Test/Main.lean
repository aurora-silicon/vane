import Vane

open Vane.Binary
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections

structure TestResult where
  name : String
  passed : Bool
  detail : String

private def checkOk {α : Type} (name : String) (res : Except String α) : TestResult :=
  match res with
  | .ok _ => { name := name, passed := true, detail := "ok" }
  | .error e => { name := name, passed := false, detail := s!"unexpected error: {e}" }

private def checkErr {α : Type} (name : String) (res : Except String α) (needle : String := "") : TestResult :=
  match res with
  | .ok _ => { name := name, passed := false, detail := "expected error, but succeeded" }
  | .error e =>
    if needle.isEmpty || (e.splitOn needle).length > 1 then
      { name := name, passed := true, detail := s!"rejected as expected: {e}" }
    else
      { name := name, passed := false, detail := s!"error '{e}' did not mention '{needle}'" }

private def roundTrip (prog : HwxProgram) : Except String (ByteArray × HwxInfo × LoadSections) := do
  let hwx ← emitHwx prog
  let info ← validateHwx hwx
  let secs ← buildSections hwx
  validateSections hwx secs
  pure (hwx, info, secs)

-- Helper: build baseline H16 linear program (2 inputs, 1 output, seq + bitmask records)
private def h16LinearProg : HwxProgram :=
  let in0 := SurfaceSpec.packed "in0" "tensor_in_0" .input 1 8 4 4
  let in1 := SurfaceSpec.packed "in1" "tensor_in_1" .input 1 8 4 4
  let out0 := SurfaceSpec.packed "out0" "tensor_out_0" .output 1 8 4 4
  let recs : Array RegRecord := #[
    .sequential 0x0000 (some 6) #[0x1, 0x2],
    .sequential 0x1040 (some 7) #[0x3, 0x4],
    .sequential 0x1140 (some 8) #[0x5],
    .bitmask 0x1240 0x0003 #[0x10, 0x20, 0x30],
    -- Config-word BAR slot 15 (program descriptor base is 0; excluded from OPERATION)
    .sequential 0x1134 (some 15) #[0x99]
  ]
  {
    target := .h16
    surfaces := #[in0, in1, out0]
    constants := zeroBytes 32
    taskStream := { prologue := 1, tasks := #[Task.mk9 recs] }
    ident := some "vane"
  }

-- Helper: build H16 kernel-table program with relocations
private def h16KernelProg : Except String HwxProgram := do
  let in0 := SurfaceSpec.packed "x" "input_x" .input 1 16 4 4
  let out0 := SurfaceSpec.packed "y" "output_y" .output 1 16 4 4
  let mut kernel := zeroBytes 512
  kernel ← putU32LE kernel 0 0x11223344
  kernel ← putU32LE kernel 128 0x55667788
  let recs : Array RegRecord := #[
    .sequential 0x0000 (some 6) #[0x1],
    .sequential 0x1040 (some 7) #[0x2],
    -- Kernel is at BAR slot 8 (2 surfaces => slots 6, 7; kernel => slot 8)
    .sequential 0x1340 (some 8) #[0x00000080, 0x00000004]
  ]
  let stream : TaskStream := { prologue := 1, tasks := #[Task.mk9 recs] }
  let enc ← encodeTaskStream .h16 stream
  let some (_, _, _, off0) := enc.payloadSites.find? (fun (t, r, p, _) => t == 0 && r == 2 && p == 0)
    | throw "missing payload site 0"
  let some (_, _, _, off1) := enc.payloadSites.find? (fun (t, r, p, _) => t == 0 && r == 2 && p == 1)
    | throw "missing payload site 1"
  pure {
    target := .h16
    surfaces := #[in0, out0]
    constants := zeroBytes 16
    kernel := some kernel
    taskStream := stream
    relocations := #[{ patchOffset := off0.toUInt32 }, { patchOffset := off1.toUInt32 }]
    ident := none
  }

-- Helper: build H17 scratch-backed multi-task program with 10-word task header and __KERN_0
private def h17ScratchMultiTaskProg : Except String HwxProgram := do
  let in0 := SurfaceSpec.packed "a" "input_a" .input 1 8 8 8
  let out0 := SurfaceSpec.packed "b" "output_b" .output 1 8 8 8
  let kernel := zeroBytes 256
  let t0 := Task.mk9 #[
    .sequential 0x0000 (some 6) #[0x1, 0x2],
    .sequential 0x1240 (some 3) #[0x40] -- scratch BAR slot 3
  ]
  let t1 := Task.mk10 #[
    .sequential 0x1340 (some 8) #[0x20], -- kernel BAR slot 8
    .bitmask 0x1440 0x0005 #[0x10, 0x20, 0x30],
    .sequential 0x1540 (some 7) #[0x99] -- output BAR slot 7
  ] (leadingZeroBlocks := 1)
  let stream : TaskStream := { prologue := 1, tasks := #[t0, t1], trailingZeroBlocks := 1 }
  let enc ← encodeTaskStream .h17 stream
  let some (_, _, _, relOff) := enc.payloadSites.find? (fun (t, r, p, _) => t == 1 && r == 0 && p == 0)
    | throw "missing task 1 kernel payload site"
  pure {
    target := .h17
    surfaces := #[in0, out0]
    constants := zeroBytes 64
    kernel := some kernel
    scratch := some { size := 0x4000, usable := 0x3000 }
    taskStream := stream
    relocations := #[{ patchOffset := relOff.toUInt32 }]
    ident := some "vane-h17"
    noteVersion := some 1
  }

-- Helper: find offset of first load command matching predicate
private def findLoadCmd (bs : ByteArray) (pred : UInt32 → Nat → Bool) : Except String (Nat × Nat) := do
  let ncmds ← readU32LE bs 0x10
  let mut cursor := 32
  for _ in [:ncmds.toNat] do
    let cmd ← readU32LE bs cursor
    let cmdsize ← readU32LE bs (cursor + 4)
    if pred cmd cursor then
      return (cursor, cmdsize.toNat)
    cursor := cursor + cmdsize.toNat
  throw "load command not found"

private def runAllTests : IO (Array TestResult) := do
  let mut results : Array TestResult := #[]

  -- ========================
  -- Positive round-trip tests
  -- ========================

  -- 1. H16 linear program (2 inputs, 1 output, sequential + bitmask records)
  results := results.push (checkOk "pos: H16 linear 2-in 1-out (seq + bitmask)" (do
    let (_, info, secs) ← roundTrip h16LinearProg
    if info.target != .h16 then throw "expected target h16"
    if info.surfaces.size != 3 then throw "expected 3 surfaces"
    if info.programDescSize != 0x8a0 then throw "expected 0x8a0 program descriptor"
    if secs.generic.size != 0x208 + 3 * 0x30 then throw "unexpected generic size"
    pure ()
  ))

  -- 2a. H17 linear program with prologue 0
  results := results.push (checkOk "pos: H17 linear prologue=0" (do
    let prog : HwxProgram := {
      h16LinearProg with
      target := .h17
      taskStream := { h16LinearProg.taskStream with prologue := 0 }
    }
    let (_, info, _) ← roundTrip prog
    if info.taskStream.prologue != 0 then throw "expected prologue 0"
    pure ()
  ))

  -- 2b. H17 linear program with prologue 1 and BAR table at slot 5 (+0x60)
  results := results.push (checkOk "pos: H17 linear prologue=1 barTableSlot=5" (do
    let in0 := SurfaceSpec.packed "in0" "input_0" .input 1 4 4 4
    let out0 := SurfaceSpec.packed "out0" "output_0" .output 1 4 4 4
    let prog : HwxProgram := {
      target := .h17
      surfaces := #[in0, out0]
      taskStream := {
        prologue := 1
        tasks := #[Task.mk9 #[
          .sequential 0x0000 (some 5) #[0x1],
          .sequential 0x1040 (some 6) #[0x2]
        ]]
      }
      barTableSlot := 5
      noteVersion := some 42
    }
    let (_, info, _) ← roundTrip prog
    if info.barTableOffset != 0x60 then throw "expected barTableOffset 0x60"
    if info.noteVersion != some 42 then throw "expected noteVersion 42"
    pure ()
  ))

  -- 3. H16 kernel-table program (__KERN_0 + relocations into __text payload words)
  results := results.push (checkOk "pos: H16 kernel-table + relocations" (do
    let prog ← h16KernelProg
    let (_, info, _) ← roundTrip prog
    if info.relocCount != 2 then throw s!"expected 2 relocs, got {info.relocCount}"
    if info.kernelSection.isNone then throw "expected kernel section"
    pure ()
  ))

  -- 4. H17 scratch-backed multi-task program (0x8c0, __KERN_0, 10-word task header)
  results := results.push (checkOk "pos: H17 scratch-backed multi-task + 10-word header" (do
    let prog ← h17ScratchMultiTaskProg
    let (_, info, secs) ← roundTrip prog
    if info.programDescSize != 0x8c0 then throw "expected 0x8c0 program descriptor"
    if info.taskStream.taskCount != 2 then throw "expected 2 tasks"
    if info.scratchSection.isNone then throw "expected scratch section"
    -- 2 surfaces + 1 scratch = 3 GENERIC entries
    if secs.generic.size != 0x208 + 3 * 0x30 then throw "expected 3 GENERIC entries"
    pure ()
  ))

  -- 5. Boundary shapes & non-trivial surface strides
  results := results.push (checkOk "pos: boundary shapes & non-trivial strides" (do
    let sIn : SurfaceSpec := {
      shortName := "in_s"
      symbolName := "strided_input_surface"
      direction := .input
      n := 2
      channels := 3
      height := 7
      width := 5
      rowStride := 64 -- > 5 * 2 = 10
      planeStride := 512 -- > 7 * 64 = 448
      batchStride := 2048 -- > 3 * 512 = 1536
      sectionSize := some 8192 -- > span 4096
      allocSize := some 0x8000 -- 32 KiB
    }
    let sOut : SurfaceSpec := {
      shortName := "out_s"
      symbolName := "strided_output_surface"
      direction := .output
      n := 1
      channels := 1
      height := 1
      width := 1
      rowStride := 64
      planeStride := 64
      batchStride := 64
    }
    let prog : HwxProgram := {
      target := .h16
      surfaces := #[sIn, sOut]
      taskStream := {
        prologue := 1
        tasks := #[Task.mk9 #[
          .sequential 0x0016 (some 6) #[0x1], -- last word of block 0 (0x0058)
          .sequential 0x164b (some 7) #[0x2]  -- last word of block 8 (0x592c)
        ]]
      }
    }
    let (_, info, _) ← roundTrip prog
    if info.surfaces[0]!.span != 4096 then throw "unexpected span"
    if info.surfaces[0]!.vmSize != 0x8000 then throw "unexpected vmSize"
    pure ()
  ))

  -- 6. Canonical ascending BAR slot ordering in OPERATION regardless of encounter order
  results := results.push (checkOk "pos: OPERATION BAR bindings sorted ascending" (do
    let in0 := SurfaceSpec.packed "in0" "input_0" .input 1 8 4 4
    let out0 := SurfaceSpec.packed "out0" "output_0" .output 1 8 4 4
    -- Reference BAR slots intentionally out of order: 8 (kernel), 6 (input), 3 (scratch), 7 (output)
    let recs : Array RegRecord := #[
      .sequential 0x1340 (some 8) #[0x00000010],
      .sequential 0x0000 (some 6) #[0x00000001],
      .sequential 0x1240 (some 3) #[0x00000000],
      .sequential 0x1040 (some 7) #[0x00000002]
    ]
    let prog : HwxProgram := {
      target := .h17
      surfaces := #[in0, out0]
      constants := zeroBytes 16
      kernel := some (zeroBytes 128)
      scratch := some { size := 0x4000, usable := 0x2000 }
      taskStream := { prologue := 1, tasks := #[Task.mk9 recs] }
    }
    let (_, _, secs) ← roundTrip prog
    let refCount ← readU32LE secs.operation 0x0c
    if refCount != 4 then throw s!"expected 4 BAR bindings, got {refCount}"
    let mut emittedSlots : Array UInt32 := #[]
    let mut emittedTags : Array UInt32 := #[]
    for j in [:4] do
      emittedSlots := emittedSlots.push (← readU32LE secs.operation (0x10 + 8 * j))
      emittedTags := emittedTags.push (← readU32LE secs.operation (0x14 + 8 * j))
    if emittedSlots != #[3, 6, 7, 8] then
      throw s!"expected ascending BAR slots [3, 6, 7, 8], got {emittedSlots.toList}"
    -- Expected tags: slot 3 -> scratch (6), slot 6 -> input (5), slot 7 -> output (4), slot 8 -> kernel (1007)
    if emittedTags != #[6, 5, 4, 1007] then
      throw s!"expected tags [6, 5, 4, 1007], got {emittedTags.toList}"
    pure ()
  ))

  -- Prepare valid baseline binaries for mutation-based negative tests
  let unwrap {α : Type} (label : String) (res : Except String α) : IO α :=
    match res with
    | .ok v => pure v
    | .error e => throw (IO.userError s!"baseline fixture '{label}' failed: {e}")

  let baseLinearHwx ← unwrap "baseLinearHwx" (emitHwx h16LinearProg)
  let baseLinearInfo ← unwrap "baseLinearInfo" (validateHwx baseLinearHwx)
  let baseLinearSecs ← unwrap "baseLinearSecs" (buildSections baseLinearHwx)

  let baseKernProg ← unwrap "baseKernProg" h16KernelProg
  let baseKernHwx ← unwrap "baseKernHwx" (emitHwx baseKernProg)
  let baseKernInfo ← unwrap "baseKernInfo" (validateHwx baseKernHwx)
  let baseKernSecs ← unwrap "baseKernSecs" (buildSections baseKernHwx)

  -- ========================
  -- Negative validator tests
  -- ========================

  -- Bad magic
  results := results.push (checkErr "neg: bad magic" (do
    let bad ← putU32LE baseLinearHwx 0x00 0xdeadbeef
    validateHwx bad
  ) "magic")

  -- Bad CPU subtype
  results := results.push (checkErr "neg: bad CPU subtype" (do
    let bad ← putU32LE baseLinearHwx 0x08 8
    validateHwx bad
  ) "subtype")

  -- Bad header flags
  results := results.push (checkErr "neg: bad header flags" (do
    let bad ← putU32LE baseLinearHwx 0x18 0x100000
    validateHwx bad
  ) "flags")

  -- Missing __PAGEZERO (rename __PAGEZERO to __BADSEG)
  results := results.push (checkErr "neg: missing __PAGEZERO" (do
    let (off, _) ← findLoadCmd baseLinearHwx (fun c _ => c == lcSegment64)
    let badName ← encodeFixedAscii "__BADSEG" 16 "badseg"
    let bad ← putBytes baseLinearHwx (off + 8) badName
    validateHwx bad
  ))

  -- Overlapping segment VM ranges (point __TEXT vmaddr to 0x0 overlapping __PAGEZERO)
  results := results.push (checkErr "neg: overlapping segment VM ranges" (do
    let (off, _) ← findLoadCmd baseLinearHwx fun c cur =>
      c == lcSegment64 && (readFixedAscii baseLinearHwx (cur + 8) 16 "seg").toOption == some "__TEXT"
    -- Set __TEXT vmaddr = 0x0000 (overlaps __PAGEZERO [0, 0x4000)) and update its sections' addr
    let origVm ← readU64LE baseLinearHwx (off + 0x18)
    let delta := origVm
    let mut bad ← putU64LE baseLinearHwx (off + 0x18) 0
    -- Adjust __text and __const addr down by delta so per-segment checks pass and global VM overlap fires
    let s0Addr ← readU64LE bad (off + 72 + 0x20)
    let s1Addr ← readU64LE bad (off + 72 + 80 + 0x20)
    bad ← putU64LE bad (off + 72 + 0x20) (s0Addr - delta)
    bad ← putU64LE bad (off + 72 + 80 + 0x20) (s1Addr - delta)
    validateHwx bad
  ) "overlapping")

  -- Overlapping file ranges (point symoff inside __TEXT file range)
  results := results.push (checkErr "neg: overlapping file ranges" (do
    let (symOff, _) ← findLoadCmd baseLinearHwx (fun c _ => c == lcSymtab)
    let bad ← putU32LE baseLinearHwx (symOff + 8) baseLinearInfo.textSection.offset
    validateHwx bad
  ) "overlapping")

  -- Wrong output surface count: 0 outputs
  results := results.push (checkErr "neg: 0 output surfaces" (do
    let s0 := SurfaceSpec.packed "in0" "input_0" .input 1 4 4 4
    let s1 := SurfaceSpec.packed "in1" "input_1" .input 1 4 4 4
    let prog : HwxProgram := { h16LinearProg with surfaces := #[s0, s1] }
    emitHwx prog
  ) "output")

  -- Wrong output surface count: 2 outputs (also verify validateHwx rejects mutated HWX with 2 outputs)
  results := results.push (checkErr "neg: 2 output surfaces (validator)" (do
    let (off, _) ← findLoadCmd baseLinearHwx fun c cur =>
      c == lcSegment64 &&
      (readFixedAscii baseLinearHwx (cur + 8) 16 "seg").toOption == some "__FVMLIB" &&
      (readU32LE baseLinearHwx (cur + 0x3c)).toOption == some 1
    -- Flip first input __FVMLIB to output (prot=2, sect=__data, flags=0x23)
    let mut bad ← putU32LE baseLinearHwx (off + 0x38) 2
    bad ← putU32LE bad (off + 0x3c) 2
    let dataName ← encodeFixedAscii "__data" 16 "sect"
    bad ← putBytes bad (off + 72) dataName
    bad ← putU32LE bad (off + 72 + 0x40) 0x23
    validateHwx bad
  ) "output")

  -- Invalid register address outside the 9 allowed blocks (sequential form)
  results := results.push (checkErr "neg: invalid register address (sequential)" (do
    -- Word 0x0017 => byte 0x005c (first byte past block 0 [0x0000, 0x005c))
    let badText : ByteArray := Id.run do
      let hdr := pushZeros (pushU32LE ByteArray.empty 1) 12
      let taskWords : UInt32 := 11 -- 9 header + 1 rec hdr + 1 payload
      let mut t := pushU32LE hdr (taskWords <<< 16)
      t := pushZeros t (8 * 4)
      t := pushU32LE t 0x0017 -- seq base=0x0017, count=1
      t := pushU32LE t 0xdeadbeef
      return t
    validateTaskStream .h16 badText
  ) "outside allowed blocks")

  -- Invalid register address outside the 9 allowed blocks (bitmask form)
  results := results.push (checkErr "neg: invalid register address (bitmask)" (do
    -- Base 0x0016 (valid, 0x0058) + mask bit 0 => 0x0017 (invalid, 0x005c)
    let badText : ByteArray := Id.run do
      let hdr := pushZeros (pushU32LE ByteArray.empty 1) 12
      let taskWords : UInt32 := 12 -- 9 header + 1 rec hdr + 2 payload
      let mut t := pushU32LE hdr (taskWords <<< 16)
      t := pushZeros t (8 * 4)
      let recHdr : UInt32 := ((1 : UInt32) <<< 31) ||| ((1 : UInt32) <<< 15) ||| 0x0016
      t := pushU32LE t recHdr
      t := pushU32LE t 0x1
      t := pushU32LE t 0x2
      return t
    validateTaskStream .h16 badText
  ) "outside allowed blocks")

  -- H16 trailing zero padding after final task
  results := results.push (checkErr "neg: H16 trailing zero padding after final task" (do
    let enc ← encodeTaskStream .h16 h16LinearProg.taskStream
    let padded := padToAlign enc.bytes 16 ++ zeroBytes 16
    validateTaskStream .h16 padded
  ) "trailing")

  -- Relocation targeting task header instead of payload word
  results := results.push (checkErr "neg: relocation targeting task header" (do
    -- Task header starts at offset 16 inside __text; patch relocation 0 to offset 16
    let reloff := baseKernInfo.textSection.reloff.toNat
    let bad ← putU32LE baseKernHwx reloff 16
    validateHwx bad
  ) "payload")

  -- Relocation targeting register-record header instead of payload word
  results := results.push (checkErr "neg: relocation targeting register-record header" (do
    -- First register record header is at offset 16 + 9*4 = 52
    let reloff := baseKernInfo.textSection.reloff.toNat
    let bad ← putU32LE baseKernHwx reloff 52
    validateHwx bad
  ) "payload")

  -- Out-of-bounds relocation addend (>= kernel.size)
  results := results.push (checkErr "neg: out-of-bounds relocation addend" (do
    let reloff := baseKernInfo.textSection.reloff.toNat
    let patchOff ← readU32LE baseKernHwx reloff
    let textFileOff := baseKernInfo.textSection.offset.toNat
    let kSize := baseKernInfo.kernelSection.get!.size.toUInt32
    let bad ← putU32LE baseKernHwx (textFileOff + patchOff.toNat) kSize
    validateHwx bad
  ) "addend")

  -- Non-zero reserved byte inside program descriptor
  results := results.push (checkErr "neg: non-zero reserved byte in program descriptor" (do
    let pdOff := baseLinearInfo.programDescOffset
    let bad ← putU8 baseLinearHwx (pdOff + 0x100) 1
    validateHwx bad
  ) "program descriptor")

  -- Mismatched tensor span > section.size
  results := results.push (checkErr "neg: tensor span > section.size" (do
    -- Shrink first __FVMLIB section size to 4 bytes (< span)
    let (off, _) ← findLoadCmd baseLinearHwx fun c cur =>
      c == lcSegment64 &&
      (readFixedAscii baseLinearHwx (cur + 8) 16 "seg").toOption == some "__FVMLIB"
    let bad ← putU64LE baseLinearHwx (off + 72 + 0x28) 4
    validateHwx bad
  ) "span")

  -- GENERIC section with reserved buffer ID 2
  results := results.push (checkErr "neg: GENERIC reserved buffer ID 2" (do
    let badGen ← putU32LE baseLinearSecs.generic (0x208 + 0x04) 2
    validateSections baseLinearHwx { baseLinearSecs with generic := badGen }
  ) "reserved")

  -- GENERIC section with reserved buffer ID 3
  results := results.push (checkErr "neg: GENERIC reserved buffer ID 3" (do
    let badGen ← putU32LE baseLinearSecs.generic (0x208 + 0x04) 3
    validateSections baseLinearHwx { baseLinearSecs with generic := badGen }
  ) "reserved")

  -- GENERIC section with duplicate buffer IDs
  results := results.push (checkErr "neg: GENERIC duplicate buffer ID" (do
    let bid0 ← readU32LE baseLinearSecs.generic (0x208 + 0x04)
    let badGen ← putU32LE baseLinearSecs.generic (0x208 + 0x30 + 0x04) bid0
    validateSections baseLinearHwx { baseLinearSecs with generic := badGen }
  ) "duplicate")

  -- OPERATION section missing kernel binding (1007) when __KERN_0 exists
  results := results.push (checkErr "neg: OPERATION missing kernel binding 1007" (do
    -- baseKernSecs has 3 bindings: slots 6, 7, 8 (where slot 8 is tag 1007). Drop the 3rd binding.
    let mut badOp ← putU32LE baseKernSecs.operation 0x0c 2
    badOp ← putU32LE badOp (0x10 + 8 * 2) 0
    badOp ← putU32LE badOp (0x14 + 8 * 2) 0
    validateSections baseKernHwx { baseKernSecs with operation := badOp }
  ))

  -- OPERATION section binding a BAR slot to the wrong VM base
  results := results.push (checkErr "neg: OPERATION BAR slot bound to wrong VM base" (do
    -- Swap tag of binding 0 (slot 6 -> input0, bid 5) to bid 4 (output0's buffer ID)
    let badOp ← putU32LE baseLinearSecs.operation 0x14 4
    validateSections baseLinearHwx { baseLinearSecs with operation := badOp }
  ) "VM base")

  pure results

def main : IO UInt32 := do
  let results ← runAllTests
  let mut passed := 0
  let mut failed := 0
  for r in results do
    if r.passed then
      passed := passed + 1
      IO.println s!"[PASS] {r.name}"
    else
      failed := failed + 1
      IO.eprintln s!"[FAIL] {r.name}: {r.detail}"
  IO.println s!"\nLean tests: {passed} passed, {failed} failed (total {results.size})"
  return if failed == 0 then 0 else 1
