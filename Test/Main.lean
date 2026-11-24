import Vane

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Tiling
open Vane.Program
open Vane.Verify
open Vane.Pe
open Vane.Lut
open Vane.Conv
open Vane.Conv1x1
open Vane.Conv3x3

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

-- Helper: synthesize deterministic tagged 16-bit weight & bias buffers for testing
private def mkTaggedWeights (cout cin : Nat) : ByteArray := Id.run do
  let mut bs := ByteArray.empty
  for o in [:cout] do
    for c in [:cin] do
      let tag : UInt16 := ((o * 257 + c * 17 + 1) % 65535).toUInt16
      bs := pushU16LE bs tag
  return bs

private def mkTaggedBias (cout : Nat) : ByteArray := Id.run do
  let mut bs := ByteArray.empty
  for o in [:cout] do
    let tag : UInt16 := (0x8000 + ((o * 13 + 5) % 0x7fff)).toUInt16
    bs := pushU16LE bs tag
  return bs

-- Helper: locate a register's payload file offset inside a compiled single-task HWX
private def findRegFileOffset (hwx : ByteArray) (info : HwxInfo) (regByteAddr : Nat) : Except String Nat := do
  let textBytes ← readSlice hwx info.textSection.offset.toNat info.textSection.size.toNat
  let tasks ← decodeTaskStream info.target textBytes
  let some task := tasks[0]?
    | throw "no task in HWX"
  let some w := task.writes.find? (fun wr => wr.byteAddr == regByteAddr)
    | throw s!"register {toHex regByteAddr} not found in task 0"
  pure (info.textSection.offset.toNat + w.payloadOffset)

-- Helper: tamper a single register payload word in a compiled HWX and re-run validateHwx + verifyHwxTasks
private def tamperRegAndVerify (compiled : CompiledConv1x1) (regByteAddr : Nat) (newVal : UInt32) : Except String Unit := do
  let fOff ← findRegFileOffset compiled.hwx compiled.info regByteAddr
  let badHwx ← putU32LE compiled.hwx fOff newVal
  let info ← validateHwx badHwx
  let _ ← verifyConvHwxTasks badHwx info (requireConvTask := true)
  pure ()

-- Helper: synthesize deterministic tagged 3x3 fp16 weights (`cout * cinW * 9` halfwords)
private def mkTaggedWeights3x3 (cout cinW : Nat) (withZeros : Bool := false) : ByteArray := Id.run do
  let mut bs := ByteArray.empty
  let mut idx := 0
  for o in [:cout] do
    for ci in [:cinW] do
      for tp in [:9] do
        let tag : UInt16 :=
          if withZeros && idx % 5 == 0 then 0x0000
          else if withZeros && idx % 7 == 0 then 0x8000
          else
            -- Non-zero fp16-safe value (avoid 0x0000 and 0x8000)
            let raw := ((o * 257 + ci * 31 + tp * 13 + 1) % 0x3bff) + 1
            raw.toUInt16
        bs := pushU16LE bs tag
        idx := idx + 1
  return bs

-- Helper: brute-force tile scan reference for inRows3x3
private def inRows3x3Brute (sh : Conv3x3Shape) (th : Nat) : Nat := Id.run do
  let ho := sh.outH
  let pt := sh.padTop
  if th == 0 || ho == 0 then return 0
  let nTiles := ceilDiv ho th
  let mut best := 0
  for i in [:nTiles] do
    let r := tileInRows3x3 sh.h sh.stride pt ho th i
    if r > best then best := r
  return best

-- Helper: tamper a single register payload word in a compiled 3x3 HWX and verify with unified verifier
private def tamperReg3x3AndVerify (compiled : CompiledConv3x3) (regByteAddr : Nat) (newVal : UInt32) : Except String Unit := do
  let fOff ← findRegFileOffset compiled.hwx compiled.info regByteAddr
  let badHwx ← putU32LE compiled.hwx fOff newVal
  let info ← validateHwx badHwx
  let _ ← verifyConvHwxTasks badHwx info (requireConvTask := true)
  pure ()

private def runAllTests : IO (Array TestResult) := do
  let mut results : Array TestResult := #[]

  -- =========================================================================
  -- M0 Positive round-trip tests
  -- =========================================================================

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
      rowStride := 64
      planeStride := 512
      batchStride := 2048
      sectionSize := some 8192
      allocSize := some 0x8000
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
          .sequential 0x0016 (some 6) #[0x1],
          .sequential 0x164b (some 7) #[0x2]
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
    if emittedTags != #[6, 5, 4, 1007] then
      throw s!"expected tags [6, 5, 4, 1007], got {emittedTags.toList}"
    pure ()
  ))

  -- Prepare valid baseline binaries for mutation-based M0 negative tests
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

  -- =========================================================================
  -- M0 Negative validator tests
  -- =========================================================================

  results := results.push (checkErr "neg: bad magic" (do
    let bad ← putU32LE baseLinearHwx 0x00 0xdeadbeef
    validateHwx bad
  ) "magic")

  results := results.push (checkErr "neg: bad CPU subtype" (do
    let bad ← putU32LE baseLinearHwx 0x08 8
    validateHwx bad
  ) "subtype")

  results := results.push (checkErr "neg: bad header flags" (do
    let bad ← putU32LE baseLinearHwx 0x18 0x100000
    validateHwx bad
  ) "flags")

  results := results.push (checkErr "neg: missing __PAGEZERO" (do
    let (off, _) ← findLoadCmd baseLinearHwx (fun c _ => c == lcSegment64)
    let badName ← encodeFixedAscii "__BADSEG" 16 "badseg"
    let bad ← putBytes baseLinearHwx (off + 8) badName
    validateHwx bad
  ))

  results := results.push (checkErr "neg: overlapping segment VM ranges" (do
    let (off, _) ← findLoadCmd baseLinearHwx fun c cur =>
      c == lcSegment64 && (readFixedAscii baseLinearHwx (cur + 8) 16 "seg").toOption == some "__TEXT"
    let origVm ← readU64LE baseLinearHwx (off + 0x18)
    let delta := origVm
    let mut bad ← putU64LE baseLinearHwx (off + 0x18) 0
    let s0Addr ← readU64LE bad (off + 72 + 0x20)
    let s1Addr ← readU64LE bad (off + 72 + 80 + 0x20)
    bad ← putU64LE bad (off + 72 + 0x20) (s0Addr - delta)
    bad ← putU64LE bad (off + 72 + 80 + 0x20) (s1Addr - delta)
    validateHwx bad
  ) "overlapping")

  results := results.push (checkErr "neg: overlapping file ranges" (do
    let (symOff, _) ← findLoadCmd baseLinearHwx (fun c _ => c == lcSymtab)
    let bad ← putU32LE baseLinearHwx (symOff + 8) baseLinearInfo.textSection.offset
    validateHwx bad
  ) "overlapping")

  results := results.push (checkErr "neg: 0 output surfaces" (do
    let s0 := SurfaceSpec.packed "in0" "input_0" .input 1 4 4 4
    let s1 := SurfaceSpec.packed "in1" "input_1" .input 1 4 4 4
    let prog : HwxProgram := { h16LinearProg with surfaces := #[s0, s1] }
    emitHwx prog
  ) "output")

  results := results.push (checkErr "neg: 2 output surfaces (validator)" (do
    let (off, _) ← findLoadCmd baseLinearHwx fun c cur =>
      c == lcSegment64 &&
      (readFixedAscii baseLinearHwx (cur + 8) 16 "seg").toOption == some "__FVMLIB" &&
      (readU32LE baseLinearHwx (cur + 0x3c)).toOption == some 1
    let mut bad ← putU32LE baseLinearHwx (off + 0x38) 2
    bad ← putU32LE bad (off + 0x3c) 2
    let dataName ← encodeFixedAscii "__data" 16 "sect"
    bad ← putBytes bad (off + 72) dataName
    bad ← putU32LE bad (off + 72 + 0x40) 0x23
    validateHwx bad
  ) "output")

  results := results.push (checkErr "neg: invalid register address (sequential)" (do
    let badText : ByteArray := Id.run do
      let hdr := pushZeros (pushU32LE ByteArray.empty 1) 12
      let taskWords : UInt32 := 11
      let mut t := pushU32LE hdr (taskWords <<< 16)
      t := pushZeros t (8 * 4)
      t := pushU32LE t 0x0017
      t := pushU32LE t 0xdeadbeef
      return t
    validateTaskStream .h16 badText
  ) "outside allowed blocks")

  results := results.push (checkErr "neg: invalid register address (bitmask)" (do
    let badText : ByteArray := Id.run do
      let hdr := pushZeros (pushU32LE ByteArray.empty 1) 12
      let taskWords : UInt32 := 12
      let mut t := pushU32LE hdr (taskWords <<< 16)
      t := pushZeros t (8 * 4)
      let recHdr : UInt32 := ((1 : UInt32) <<< 31) ||| ((1 : UInt32) <<< 15) ||| 0x0016
      t := pushU32LE t recHdr
      t := pushU32LE t 0x1
      t := pushU32LE t 0x2
      return t
    validateTaskStream .h16 badText
  ) "outside allowed blocks")

  results := results.push (checkErr "neg: H16 trailing zero padding after final task" (do
    let enc ← encodeTaskStream .h16 h16LinearProg.taskStream
    let padded := padToAlign enc.bytes 16 ++ zeroBytes 16
    validateTaskStream .h16 padded
  ) "trailing")

  results := results.push (checkErr "neg: relocation targeting task header" (do
    let reloff := baseKernInfo.textSection.reloff.toNat
    let bad ← putU32LE baseKernHwx reloff 16
    validateHwx bad
  ) "payload")

  results := results.push (checkErr "neg: relocation targeting register-record header" (do
    let reloff := baseKernInfo.textSection.reloff.toNat
    let bad ← putU32LE baseKernHwx reloff 52
    validateHwx bad
  ) "payload")

  results := results.push (checkErr "neg: out-of-bounds relocation addend" (do
    let reloff := baseKernInfo.textSection.reloff.toNat
    let patchOff ← readU32LE baseKernHwx reloff
    let textFileOff := baseKernInfo.textSection.offset.toNat
    let kSize := baseKernInfo.kernelSection.get!.size.toUInt32
    let bad ← putU32LE baseKernHwx (textFileOff + patchOff.toNat) kSize
    validateHwx bad
  ) "addend")

  results := results.push (checkErr "neg: non-zero reserved byte in program descriptor" (do
    let pdOff := baseLinearInfo.programDescOffset
    let bad ← putU8 baseLinearHwx (pdOff + 0x100) 1
    validateHwx bad
  ) "program descriptor")

  results := results.push (checkErr "neg: tensor span > section.size" (do
    let (off, _) ← findLoadCmd baseLinearHwx fun c cur =>
      c == lcSegment64 &&
      (readFixedAscii baseLinearHwx (cur + 8) 16 "seg").toOption == some "__FVMLIB"
    let bad ← putU64LE baseLinearHwx (off + 72 + 0x28) 4
    validateHwx bad
  ) "span")

  results := results.push (checkErr "neg: GENERIC reserved buffer ID 2" (do
    let badGen ← putU32LE baseLinearSecs.generic (0x208 + 0x04) 2
    validateSections baseLinearHwx { baseLinearSecs with generic := badGen }
  ) "reserved")

  results := results.push (checkErr "neg: GENERIC reserved buffer ID 3" (do
    let badGen ← putU32LE baseLinearSecs.generic (0x208 + 0x04) 3
    validateSections baseLinearHwx { baseLinearSecs with generic := badGen }
  ) "reserved")

  results := results.push (checkErr "neg: GENERIC duplicate buffer ID" (do
    let bid0 ← readU32LE baseLinearSecs.generic (0x208 + 0x04)
    let badGen ← putU32LE baseLinearSecs.generic (0x208 + 0x30 + 0x04) bid0
    validateSections baseLinearHwx { baseLinearSecs with generic := badGen }
  ) "duplicate")

  results := results.push (checkErr "neg: OPERATION missing kernel binding 1007" (do
    let mut badOp ← putU32LE baseKernSecs.operation 0x0c 2
    badOp ← putU32LE badOp (0x10 + 8 * 2) 0
    badOp ← putU32LE badOp (0x14 + 8 * 2) 0
    validateSections baseKernHwx { baseKernSecs with operation := badOp }
  ))

  results := results.push (checkErr "neg: OPERATION BAR slot bound to wrong VM base" (do
    let badOp ← putU32LE baseLinearSecs.operation 0x14 4
    validateSections baseLinearHwx { baseLinearSecs with operation := badOp }
  ) "VM base")

  -- =========================================================================
  -- M1 Register Schema & Bitfield Encoder Tests
  -- =========================================================================

  results := results.push (checkOk "m1: register bitfield encode/decode round-trip" (do
    let wConv ← encodeWord convCfgLayout #[
      ("kw", 1), ("kh", 1), ("sx", 1), ("sy", 1),
      ("pl", 0), ("pt", 0), ("hx", 1), ("vy", 1)
    ]
    if wConv != 0x5000a041 then
      throw s!"convCfg={toHex wConv.toNat} != 0x5000a041"
    if (← decodeFieldChecked convCfgLayout wConv "hx") != 1 then
      throw "hx decode mismatch"

    let wTile ← encodeWord neTileLayout #[("ocg", 4), ("fat", 1), ("WU", 3)]
    if wTile != 0x3c then
      throw s!"neTile={toHex wTile.toNat} != 0x3c"
    if (← decodeFieldChecked neTileLayout wTile "ocg") != 4 then throw "ocg mismatch"
    if (← decodeFieldChecked neTileLayout wTile "fat") != 1 then throw "fat mismatch"
    if (← decodeFieldChecked neTileLayout wTile "WU") != 3 then throw "WU mismatch"

    let wSrc1 ← encodeWord src1CfgLayout #[("mode", 2), ("P", 1)]
    if wSrc1 != 0x00500172 then throw s!"src1Cfg={toHex wSrc1.toNat}"
    let wResDir ← encodeWord resCfgLayout #[("mode", 2), ("direct", 1), ("P", 1)]
    if wResDir != 0x0050017a then throw s!"resCfg(direct)={toHex wResDir.toNat}"
    let wResStg ← encodeWord resCfgLayout #[("mode", 2), ("direct", 0), ("P", 1)]
    if wResStg != 0x00500172 then throw s!"resCfg(staged)={toHex wResStg.toNat}"

    let wOp ← encodeWord neOpLayout #[("biasInKernel", 1), ("relu", 1), ("lut", 0)]
    if wOp != 0x00113c10 then throw s!"neOp={toHex wOp.toNat}"

    let hdr ← encodeTaskHeader9 0 64 4
    if hdr[0]! != ((64 : UInt32) <<< 16) then throw s!"hdr[0]={toHex hdr[0]!.toNat}"
    if hdr[8]! != 0x00050009 then throw s!"hdr[8]={toHex hdr[8]!.toNat}"
    pure ()
  ))

  results := results.push (checkErr "m1 neg: bitfield WU=4 overflow rejected" (do
    encodeWord neTileLayout #[("ocg", 0), ("fat", 0), ("WU", 4)]
  ) "exceeds 2-bit")

  results := results.push (checkErr "m1 neg: bitfield overlapping layout rejected" (do
    let badLayout : WordLayout := {
      name := "overlap"
      fields := #[
        { name := "a", lo := 0, width := 4 },
        { name := "b", lo := 3, width := 2 }
      ]
    }
    encodeWord badLayout #[("a", 1), ("b", 1)]
  ) "overlaps")

  results := results.push (checkErr "m1 neg: bitfield >32 bits rejected" (do
    let badLayout : WordLayout := {
      name := "wide"
      fields := #[{ name := "a", lo := 30, width := 4 }]
    }
    encodeWord badLayout #[("a", 1)]
  ) "exceeds 32")

  results := results.push (checkErr "m1 neg: bitfield overlapping fixedMask rejected" (do
    let badLayout : WordLayout := {
      name := "fixedOverlap"
      fixedMask := 0x00000010
      fixedValue := 0x00000010
      fields := #[{ name := "a", lo := 4, width := 2 }]
    }
    encodeWord badLayout #[("a", 1)]
  ) "overlaps fixed")

  results := results.push (checkErr "m1 neg: non-zero reserved bits in neTile rejected on decode" (do
    -- Bit 6 set (as if WU=4 overflowed into bit 6)
    decodeFieldChecked neTileLayout 0x00000040 "WU"
  ) "reserved")

  -- =========================================================================
  -- M1 Tiling Legality & Independent Tile Chooser Tests
  -- =========================================================================

  results := results.push (checkOk "m1: legal H16 & H17 tilings accepted" (do
    let sh : Conv1x1Shape := { h := 16, w := 32, cin := 16, cout := 32 }
    let t16 : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 1, shPref := 0, shMax := 2, tileHeight := 16 }
    validateTiling .h16 sh t16
    validateTiling .h17 sh t16
    -- H17 2-round covering Cout=512 at F=4, ocg=4, fat=0
    let sh512 : Conv1x1Shape := { h := 8, w := 16, cin := 16, cout := 512 }
    let t512 : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 4, shPref := 0, shMax := 3, tileHeight := 8 }
    validateTiling .h17 sh512 t512
    pure ()
  ))

  results := results.push (checkErr "m1 neg: tiling rejects WU=4" (do
    let sh : Conv1x1Shape := { h := 8, w := 16, cin := 8, cout := 1 }
    let t : Tiling := { logNE := 4, f := 0, fat := 0, ocg := 0, shPref := 0, shMax := 3, tileHeight := 8 }
    validateTiling .h17 sh t
  ) "WU=4")

  results := results.push (checkErr "m1 neg: tiling rejects ocg=5 at F=0, fat=1 on H17" (do
    let sh : Conv1x1Shape := { h := 8, w := 16, cin := 8, cout := 32 }
    let t : Tiling := { logNE := 4, f := 0, fat := 1, ocg := 5, shPref := 0, shMax := 4, tileHeight := 8 }
    validateTiling .h17 sh t
  ) "ocg=5")

  results := results.push (checkErr "m1 neg: tiling rejects ocg=5 at F=4, fat=0 on H17" (do
    let sh : Conv1x1Shape := { h := 8, w := 8, cin := 64, cout := 600 }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 5, shPref := 0, shMax := 4, tileHeight := 8 }
    validateTiling .h17 sh t
  ) "ocg=5")

  results := results.push (checkErr "m1 neg: tiling rejects ocg=5 on H16" (do
    let sh : Conv1x1Shape := { h := 8, w := 16, cin := 8, cout := 256 }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 5, shPref := 0, shMax := 3, tileHeight := 8 }
    validateTiling .h16 sh t
  ) "ocg=5")

  results := results.push (checkErr "m1 neg: tiling rejects shPref > shMax" (do
    let sh : Conv1x1Shape := { h := 8, w := 16, cin := 8, cout := 16 }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 3, shMax := 2, tileHeight := 8 }
    validateTiling .h17 sh t
  ) "shPref")

  results := results.push (checkErr "m1 neg: tiling rejects coverMin > shMax" (do
    -- W = 24 => aligned multiples 24, 32, 32, 64 => with T=128, fat=0: coverMin = 1 (from 64 = 128>>1)
    let sh : Conv1x1Shape := { h := 8, w := 24, cin := 8, cout := 16 }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 0, shMax := 0, tileHeight := 8 }
    validateTiling .h17 sh t
  ) "coverMin")

  results := results.push (checkErr "m1 neg: tiling rejects L2 > 2 MiB" (do
    -- srcRowStride = 1024 * 2 * 128 = 262144 bytes; TH=9 => 2,359,296 > 2 MiB
    let sh : Conv1x1Shape := { h := 16, w := 128, cin := 1024, cout := 16 }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 0, shMax := 0, tileHeight := 9 }
    validateTiling .h17 sh t
  ) "2 MiB")

  results := results.push (checkErr "m1 neg: tiling rejects TH=0" (do
    let sh : Conv1x1Shape := { h := 8, w := 16, cin := 8, cout := 16 }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 0, shMax := 3, tileHeight := 0 }
    validateTiling .h17 sh t
  ) "tileHeight")

  results := results.push (checkErr "m1 neg: tiling rejects TH > H" (do
    let sh : Conv1x1Shape := { h := 8, w := 16, cin := 8, cout := 16 }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 0, shMax := 3, tileHeight := 9 }
    validateTiling .h17 sh t
  ) "tileHeight")

  results := results.push (checkErr "m1 neg: tiling rejects Cout exceeding H16 coverage" (do
    let sh : Conv1x1Shape := { h := 8, w := 16, cin := 8, cout := 257 }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 4, shPref := 0, shMax := 3, tileHeight := 8 }
    validateTiling .h16 sh t
  ) "coverage")

  results := results.push (checkErr "m1 neg: tiling rejects Cout exceeding H17 coverage" (do
    let sh : Conv1x1Shape := { h := 8, w := 16, cin := 8, cout := 513 }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 4, shPref := 0, shMax := 3, tileHeight := 8 }
    validateTiling .h17 sh t
  ) "coverage")

  results := results.push (checkErr "m1 neg: chooseTiling rejects H17 Cout=600 (> 512)" (do
    chooseTiling .h17 { h := 8, w := 8, cin := 64, cout := 600 }
  ) "no legal")

  -- Tile chooser across diverse shapes on H16 and H17
  results := results.push (checkOk "m1: chooseTiling across diverse shapes on H16 and H17" (do
    let commonShapes : Array Conv1x1Shape := #[
      { h := 1, w := 1, cin := 1, cout := 1 },
      { h := 4, w := 4, cin := 8, cout := 8 },
      { h := 13, w := 17, cin := 7, cout := 11 },
      { h := 7, w := 37, cin := 19, cout := 23 },
      { h := 4, w := 300, cin := 16, cout := 16 },
      { h := 512, w := 16, cin := 16, cout := 16 },
      { h := 8, w := 8, cin := 16, cout := 1 },
      { h := 8, w := 8, cin := 1, cout := 16 },
      { h := 16, w := 64, cin := 256, cout := 256 },
      { h := 128, w := 128, cin := 128, cout := 64 },
      { h := 32, w := 25, cin := 32, cout := 32 } -- odd W exercising coverMin
    ]
    for sh in commonShapes do
      let t16 ← chooseTiling .h16 sh
      validateTiling .h16 sh t16
      let t17 ← chooseTiling .h17 sh
      validateTiling .h17 sh t17

    -- H17 2-round shapes (256 < Cout <= 512)
    let h17OnlyShapes : Array Conv1x1Shape := #[
      { h := 8, w := 16, cin := 64, cout := 300 },
      { h := 16, w := 32, cin := 256, cout := 512 }
    ]
    for sh in h17OnlyShapes do
      let t17 ← chooseTiling .h17 sh
      validateTiling .h17 sh t17
      if t17.ocg > 4 then throw s!"chosen ocg={t17.ocg} exceeds 4"
      -- Verify H16 properly rejects Cout > 256
      match chooseTiling .h16 sh with
      | .ok _ => throw s!"expected H16 to reject Cout={sh.cout}"
      | .error _ => pure ()

    -- Verify H17 rejects Cout > 512 in single-task chooser
    for badCout in #[513, 600, 1024] do
      match chooseTiling .h17 { h := 8, w := 8, cin := 64, cout := badCout } with
      | .ok _ => throw s!"expected H17 to reject single-task Cout={badCout}"
      | .error _ => pure ()

    -- Verify a shape that forces staged mode (shMax=4, fat=0, W=40 => (128>>4)=8 < min(32,40)=32)
    let stagedShape : Conv1x1Shape := { h := 8, w := 40, cin := 16, cout := 16 }
    let stagedT : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 4, shMax := 4, tileHeight := 8 }
    if isDirect .h16 stagedShape stagedT then throw "expected staged mode on H16"
    if isDirect .h17 stagedShape stagedT then throw "expected staged mode on H17"
    validateTiling .h16 stagedShape stagedT
    validateTiling .h17 stagedShape stagedT

    pure ()
  ))

  -- =========================================================================
  -- M1 Kernel Packer Tests
  -- =========================================================================

  results := results.push (checkOk "m1: kernel packer exact placement (Cout=1, indivisible, multi-round, H16 rep vs H17)" (do
    -- Case 1: Cout=1 with bias -> scalarBiasBits set, biasInKernel=false, weights only in chunk
    let sh1 : Conv1x1Shape := { h := 4, w := 4, cin := 5, cout := 1 }
    let t1 ← chooseTiling .h17 sh1
    let w1 := mkTaggedWeights 1 5
    let b1 := mkTaggedBias 1
    let pk1 ← packKernelChunks .h17 sh1 t1 w1 (some b1)
    if pk1.biasInKernel then throw "Cout=1 must not put bias in kernel"
    let expectedScalar ← readU16LE b1 0
    if pk1.scalarBiasBits != expectedScalar then throw "Cout=1 scalarBiasBits mismatch"
    if pk1.slots.size != 1 then throw s!"H17 Cout=1 expected 1 slot, got {pk1.slots.size}"
    if pk1.slots[0]!.size != 64 then throw s!"Cout=1 chunk size {pk1.slots[0]!.size} != 64"
    for c in [:5] do
      let got ← readU16LE pk1.bytes (2 * c)
      let exp ← readU16LE w1 (2 * c)
      if got != exp then throw s!"Cout=1 weight c={c} mismatch"

    -- Case 2: Cout=11 indivisible by 2^F=16 with bias on H17 (11 active slots, 5 omitted)
    let sh2 : Conv1x1Shape := { h := 4, w := 4, cin := 7, cout := 11 }
    let t2 : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 0, shMax := 4, tileHeight := 4 }
    let w2 := mkTaggedWeights 11 7
    let b2 := mkTaggedBias 11
    let pk2 ← packKernelChunks .h17 sh2 t2 w2 (some b2)
    if !pk2.biasInKernel then throw "Cout=11 with bias must embed bias in kernel"
    if pk2.slots.size != 11 then throw s!"expected 11 active slots for Cout=11/F=4, got {pk2.slots.size}"
    for g in [:11] do
      let cs := pk2.slots[g]!
      if cs.slot != g then throw s!"expected slot {g}, got {cs.slot}"
      if cs.size % 64 != 0 then throw "chunk size not 64-byte padded"
      -- Bias for output g at word 0
      let gotB ← readU16LE pk2.bytes cs.offset
      let expB ← readU16LE b2 (2 * g)
      if gotB != expB then throw s!"Cout=11 group {g} bias mismatch"
      for c in [:7] do
        let gotW ← readU16LE pk2.bytes (cs.offset + 2 * (1 + c))
        let expW ← readU16LE w2 (2 * (g * 7 + c))
        if gotW != expW then throw s!"Cout=11 group {g} c={c} weight mismatch"

    -- Case 3: Multi-round Cout=300 on H17 (F=4, ocg=4 -> round=256, full=1, rem=44)
    let sh3 : Conv1x1Shape := { h := 4, w := 16, cin := 3, cout := 300 }
    let t3 ← chooseTiling .h17 sh3
    let w3 := mkTaggedWeights 300 3
    let b3 := mkTaggedBias 300
    let pk3 ← packKernelChunks .h17 sh3 t3 w3 (some b3)
    let groups3 := (1 : Nat) <<< t3.f
    if pk3.slots.size != groups3 then throw s!"expected {groups3} slots, got {pk3.slots.size}"
    -- Verify every single (o, c) weight and bias B[o] appears at its exact spec position
    let mut seenOut : Array Bool := Array.replicate 300 false
    for g in [:groups3] do
      let cs := pk3.slots[g]!
      let blocks := groupOutputBlocks 300 t3.f t3.ocg g
      let mut pos := cs.offset
      for (o0, n) in blocks do
        for k in [:n] do
          let o := o0 + k
          let gotB ← readU16LE pk3.bytes (pos + 2 * k)
          let expB ← readU16LE b3 (2 * o)
          if gotB != expB then throw s!"multi-round bias mismatch at o={o}"
          if seenOut[o]! then throw s!"duplicate output channel {o}"
          seenOut := seenOut.set! o true
        pos := pos + 2 * n
        for c in [:3] do
          for k in [:n] do
            let o := o0 + k
            let gotW ← readU16LE pk3.bytes (pos + 2 * k)
            let expW ← readU16LE w3 (2 * (o * 3 + c))
            if gotW != expW then throw s!"multi-round weight mismatch at o={o}, c={c}"
          pos := pos + 2 * n
      -- Trailing padding must be zero
      for p in [pos : cs.offset + cs.size] do
        if pk3.bytes.get! p != 0 then throw s!"non-zero padding byte at {p}"
    if seenOut.any (!·) then throw "not all 300 outputs covered by kernel chunks"

    -- Case 4: H17 single-slot vs H16 rep-replicated chunks (Cout=4, logNE=4, F=2 => rep=4)
    let sh4 : Conv1x1Shape := { h := 4, w := 16, cin := 4, cout := 4 }
    let t4 : Tiling := { logNE := 4, f := 2, fat := 0, ocg := 0, shPref := 0, shMax := 3, tileHeight := 4 }
    let w4 := mkTaggedWeights 4 4
    let pk17 ← packKernelChunks .h17 sh4 t4 w4 none
    let pk16 ← packKernelChunks .h16 sh4 t4 w4 none
    let slots17 := pk17.slots.map (·.slot)
    if slots17 != #[0, 4, 8, 12] then throw s!"H17 expected slots [0,4,8,12], got {slots17.toList}"
    if pk16.slots.size != 16 then throw s!"H16 expected 16 replicated slots, got {pk16.slots.size}"
    for g in [:4] do
      let refChunk := pk17.bytes.extract pk17.slots[g]!.offset (pk17.slots[g]!.offset + pk17.slots[g]!.size)
      for j in [:4] do
        let cs := pk16.slots[g * 4 + j]!
        if cs.slot != g * 4 + j then throw s!"H16 slot order mismatch at {g*4+j}"
        let gotChunk := pk16.bytes.extract cs.offset (cs.offset + cs.size)
        if gotChunk != refChunk then throw s!"H16 replicated chunk mismatch at slot {cs.slot}"

    pure ()
  ))

  -- =========================================================================
  -- M1 Full compileConv1x1 Round-Trip & Static Verifier Tests
  -- =========================================================================

  results := results.push (checkOk "m1: compileConv1x1 round-trip matrix (H16 & H17, bias/relu, Cout=1, staged, 2-round)" (do
    let cases : Array (Target × Conv1x1Shape × Bool × Bool × Option Tiling) := #[
      (.h16, { h := 8, w := 16, cin := 16, cout := 16 }, false, false, none),
      (.h16, { h := 7, w := 13, cin := 9, cout := 11 }, true, true, none),
      (.h16, { h := 4, w := 8, cin := 16, cout := 1 }, true, false, none),
      (.h16, { h := 4, w := 8, cin := 16, cout := 1 }, false, true, none),
      (.h16, { h := 8, w := 40, cin := 16, cout := 16 }, true, false,
        some { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 4, shMax := 4, tileHeight := 8 }),
      (.h17, { h := 8, w := 16, cin := 16, cout := 16 }, false, false, none),
      (.h17, { h := 7, w := 13, cin := 9, cout := 11 }, true, true, none),
      (.h17, { h := 4, w := 8, cin := 16, cout := 1 }, true, false, none),
      (.h17, { h := 4, w := 8, cin := 1, cout := 16 }, true, true, none),
      (.h17, { h := 8, w := 16, cin := 32, cout := 300 }, true, true, none),
      (.h17, { h := 8, w := 40, cin := 16, cout := 16 }, true, true,
        some { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 4, shMax := 4, tileHeight := 8 })
    ]
    for (target, sh, withBias, relu, tOverride) in cases do
      let ws := mkTaggedWeights sh.cout sh.cin
      let bs? := if withBias then some (mkTaggedBias sh.cout) else none
      let compiled ← compileConv1x1 target sh ws bs? relu tOverride
      let info ← validateHwx compiled.hwx
      validateSectionsWithInfo info compiled.sections
      let nVerified ← verifyHwxTasks compiled.hwx info (requireConvTask := true)
      if nVerified != 1 then throw "expected 1 verified conv task"

      -- Verify target-specific prologue, __const size/alignment, and surface section sizing
      let expPrologue : UInt32 := match target with | .h16 => 1 | .h17 => 0
      if info.taskStream.prologue != expPrologue then
        throw s!"{target.name}: prologue {info.taskStream.prologue} != {expPrologue}"
      if info.constSection.size != 0x4000 then
        throw s!"__TEXT,__const size {info.constSection.size} != 0x4000"
      if info.constSection.align != 6 then
        throw s!"__TEXT,__const align {info.constSection.align} != 6"
      for surf in info.surfaces do
        match target with
        | .h16 =>
          if surf.sect.size != surf.vmSize || surf.sect.size.toNat % 0x4000 != 0 then
            throw s!"H16 surface '{surf.shortName}' section size {surf.sect.size} not 16 KiB rounded"
        | .h17 =>
          if surf.sect.size != surf.span then
            throw s!"H17 surface '{surf.shortName}' section size {surf.sect.size} != span {surf.span}"
    pure ()
  ))

  -- Negative tampering tests on emitted conv1x1 HWX binaries
  let baseConvShape : Conv1x1Shape := { h := 8, w := 16, cin := 16, cout := 16 }
  let baseConv ← unwrap "baseConv" (compileConv1x1 .h17 baseConvShape (mkTaggedWeights 16 16) (some (mkTaggedBias 16)) true)

  results := results.push (checkErr "m1 neg verifier: tampered neTile WU=4 (bit 6 set)" (do
    -- Set bit 6 on 0x40 (neTile)
    tamperRegAndVerify baseConv regNeTile 0x00000040
  ) "reserved")

  results := results.push (checkErr "m1 neg verifier: tampered neTile ocg=5 (F=4, fat=0)" (do
    -- Set ocg=5, fat=0, WU=0 on 0x40 (neTile)
    tamperRegAndVerify baseConv regNeTile 0x00000005
  ) "ocg=5")

  results := results.push (checkErr "m1 neg verifier: tampered chunk slot disabled while size written" (do
    -- Clear enable bit 0 on chunk slot 0 (0x5520 = 0x20 instead of 0x21)
    tamperRegAndVerify baseConv (regChunkSlot 0) 0x00000020
  ) "enable bit 0")

  results := results.push (checkErr "m1 neg verifier: tampered chunk size exceeds kernel" (do
    tamperRegAndVerify baseConv (regChunkSize 0) (baseConv.kernelBytes + 64).toUInt32
  ) "exceeds kernel")

  results := results.push (checkErr "m1 neg verifier: tampered L2 resBase overlapping source" (do
    -- Set resBase = 0 (overlaps src1Base = 0)
    tamperRegAndVerify baseConv regResBase 0
  ) "overlapping L2")

  results := results.push (checkErr "m1 neg verifier: tampered L2 resBase exceeding 2 MiB" (do
    -- Set resBase near 2 MiB so resBase + resBytes > 2 MiB
    tamperRegAndVerify baseConv regResBase 0x001fffe0
  ) "exceeds 2 MiB")

  results := results.push (checkErr "m1 neg verifier: tampered TH > H" (do
    tamperRegAndVerify baseConv regTH 9
  ) "tileHeight")

  results := results.push (checkErr "m1 neg verifier: tampered input DMA stride exceeds surface" (do
    tamperRegAndVerify baseConv regSrc1DmaRowStride 0x10000
  ) "DMA")

  -- Verify M0 demo program still passes unified verifyConvHwxTasks when requireConvTask := false
  results := results.push (checkOk "m1: M0 demo passes non-forced task verification" (do
    let checked ← verifyConvHwxTasks baseLinearHwx baseLinearInfo (requireConvTask := false)
    if checked != 0 then throw s!"expected 0 conv tasks in M0 demo, got {checked}"
    pure ()
  ))

  -- =========================================================================
  -- M2 Register Schema & Bitfield Tests (0x0030, 0x003c[27], 0x4900[8], 0x5500[5])
  -- =========================================================================

  results := results.push (checkOk "m2: bitfields round-trip (dwGroups, neCommon wide3x3, neCfg sparse, kernelStage sparse)" (do
    let wDw ← encodeWord dwGroupsLayout #[("perGroupIn", 1)]
    if wDw != 0x00014000 then throw s!"dwGroups={toHex wDw.toNat} != 0x00014000"
    if (← decodeFieldChecked dwGroupsLayout wDw "perGroupIn") != 1 then
      throw "dwGroups perGroupIn decode mismatch"

    let wConv3Same ← encodeWord convCfgLayout #[
      ("kw", 3), ("kh", 3), ("sx", 1), ("sy", 1),
      ("pl", 1), ("pt", 1), ("hx", 1), ("vy", 1)
    ]
    if wConv3Same != 0x5042a0c3 then
      throw s!"convCfg(3x3 s=1 same)={toHex wConv3Same.toNat} != 0x5042a0c3"

    let wCom0 ← encodeWord neCommonLayout #[
      ("kind", 0), ("shPref", 1), ("shMin", 1), ("shMax", 3), ("F", 4), ("wide3x3", 0)
    ]
    let wCom1 ← encodeWord neCommonLayout #[
      ("kind", 0), ("shPref", 1), ("shMin", 1), ("shMax", 3), ("F", 4), ("wide3x3", 1)
    ]
    if wCom1 != (wCom0 ||| ((1 : UInt32) <<< 27)) then
      throw s!"wide3x3 bit 27 mismatch: {toHex wCom0.toNat} vs {toHex wCom1.toNat}"
    if (← decodeFieldChecked neCommonLayout wCom0 "wide3x3") != 0 then throw "wide3x3=0 decode mismatch"
    if (← decodeFieldChecked neCommonLayout wCom1 "wide3x3") != 1 then throw "wide3x3=1 decode mismatch"

    let wNeCfg16Dense ← encodeWord (neCfgLayout .h16) #[("sparse", 0)]
    let wNeCfg16Sparse ← encodeWord (neCfgLayout .h16) #[("sparse", 1)]
    let wNeCfg17Dense ← encodeWord (neCfgLayout .h17) #[("sparse", 0)]
    let wNeCfg17Sparse ← encodeWord (neCfgLayout .h17) #[("sparse", 1)]
    if wNeCfg16Dense != 0x00000082 then throw s!"h16 neCfg dense={toHex wNeCfg16Dense.toNat}"
    if wNeCfg16Sparse != 0x00000182 then throw s!"h16 neCfg sparse={toHex wNeCfg16Sparse.toNat}"
    if wNeCfg17Dense != 0x10000082 then throw s!"h17 neCfg dense={toHex wNeCfg17Dense.toNat}"
    if wNeCfg17Sparse != 0x10000182 then throw s!"h17 neCfg sparse={toHex wNeCfg17Sparse.toNat}"
    if (← decodeFieldChecked (neCfgLayout .h17) wNeCfg17Sparse "sparse") != 1 then throw "neCfg sparse decode mismatch"

    let wKsDense ← encodeWord kernelStageLayout #[("sparse", 0), ("stage", 1)]
    let wKsSparse ← encodeWord kernelStageLayout #[("sparse", 1), ("stage", 1)]
    if wKsDense != 0x000102c0 then throw s!"kernelStage dense={toHex wKsDense.toNat}"
    if wKsSparse != 0x000102e0 then throw s!"kernelStage sparse={toHex wKsSparse.toNat}"
    if (← decodeFieldChecked kernelStageLayout wKsSparse "sparse") != 1 then throw "kernelStage sparse decode mismatch"
    if (← decodeFieldChecked kernelStageLayout wKsSparse "stage") != 1 then throw "kernelStage stage decode mismatch"
    pure ()
  ))

  results := results.push (checkErr "m2 neg: dwGroups bad fixed low bits rejected" (do
    decodeFieldChecked dwGroupsLayout 0x00014001 "perGroupIn"
  ) "fixed bits")

  results := results.push (checkErr "m2 neg: neCommon wide3x3=2 overflow rejected" (do
    encodeWord neCommonLayout #[
      ("kind", 0), ("shPref", 0), ("shMin", 0), ("shMax", 0), ("F", 0), ("wide3x3", 2)
    ]
  ) "exceeds 1-bit")

  results := results.push (checkErr "m2 neg: neCommon reserved bit 28 rejected on decode" (do
    decodeFieldChecked neCommonLayout ((1 : UInt32) <<< 28) "wide3x3"
  ) "reserved")

  results := results.push (checkErr "m2 neg: neCfg reserved bit rejected on decode" (do
    decodeFieldChecked (neCfgLayout .h17) 0x10000282 "sparse"
  ) "fixed bits")

  results := results.push (checkErr "m2 neg: kernelStage bad form bit rejected on decode" (do
    decodeFieldChecked kernelStageLayout 0x000102c1 "sparse"
  ) "fixed bits")

  -- =========================================================================
  -- M2 inRows3x3 O(1) vs Brute-Force Equivalence Tests
  -- =========================================================================

  results := results.push (checkOk "m2: inRows3x3 O(1) matches brute-force across diverse (H, s, pad, TH)" (do
    let heights : Array Nat := #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 23, 32, 63, 64, 127, 128, 255]
    let strides : Array Nat := #[1, 2]
    let pads : Array PadMode := #[.same, .valid]
    for h in heights do
      for s in strides do
        for pad in pads do
          if !(pad == .valid && h < 3) then
            let sh : Conv3x3Shape := {
              mode := .dense, h := h, w := max h 3, cin := 1, cout := 1, stride := s, pad := pad
            }
            let ho := sh.outH
            let thList : Array Nat :=
              if ho <= 20 then
                Id.run do
                  let mut arr : Array Nat := #[]
                  for th in [1 : ho + 1] do arr := arr.push th
                  return arr
              else
                #[1, 2, 3, 4, 5, 7, 8, 13, 16, ho / 3, ho / 2, ho - 2, ho - 1, ho]
            for th in thList do
              if th >= 1 && th <= ho then
                let fast := inRows3x3 sh th
                let brute := inRows3x3Brute sh th
                if fast != brute then
                  throw s!"inRows3x3 mismatch for H={h}, s={s}, pad={pad.name}, TH={th}: O(1)={fast} != brute={brute}"
    pure ()
  ))

  -- =========================================================================
  -- M2 Sparse Encoder / Decoder Round-Trip Tests
  -- =========================================================================

  results := results.push (checkOk "m2: sparse stream encode/decode round-trip (zeros, -0.0, mixed, non-16, LCG)" (do
    let checkRoundTrip (label : String) (slots : Array (Option UInt16)) : Except String Unit := do
      let enc ← encodeSparseStream slots
      let (dec, consumed) ← decodeSparseStream enc 0
      if consumed != enc.size then
        throw s!"{label}: consumed {consumed} != encoded size {enc.size}"
      let expLen := alignUp slots.size 16
      if dec.size != expLen then
        throw s!"{label}: decoded len {dec.size} != alignUp({slots.size}, 16)={expLen}"
      -- Verify u24 byte-length prefix
      let b0 := (enc.get! 0).toNat
      let b1 := (enc.get! 1).toNat
      let b2 := (enc.get! 2).toNat
      let u24 := b0 ||| (b1 <<< 8) ||| (b2 <<< 16)
      if u24 + 3 != enc.size then
        throw s!"{label}: u24 prefix {u24} + 3 != encoded size {enc.size}"
      for i in [:expLen] do
        let exp : UInt16 :=
          if i < slots.size then
            match slots[i]! with
            | some w => if (w &&& 0x7fff) != 0 then w else 0
            | none => 0
          else 0
        if dec[i]! != exp then
          throw s!"{label}: slot {i} got {toHex dec[i]!.toNat} != expected {toHex exp.toNat}"
      pure ()

    -- All zeros (including +0.0, -0.0, and none)
    let allZero : Array (Option UInt16) := #[
      some 0x0000, some 0x8000, none, some 0x0000,
      some 0x8000, none, some 0x0000, some 0x8000,
      none, none, some 0x0000, some 0x8000,
      some 0x0000, none, some 0x8000, some 0x0000
    ]
    let encZero ← encodeSparseStream allZero
    -- 16 all-zero slots => 3-byte u24 (= 2) + 1 mask u16 (0x0000) = 5 bytes
    if encZero.size != 5 then throw s!"allZero encoded size {encZero.size} != 5"
    checkRoundTrip "allZero" allZero

    -- Non-multiple-of-16 mixed stream with -0.0 skipped
    let mixed : Array (Option UInt16) := #[
      some 0x3c00, some 0x8000, none, some 0xbc00,
      some 0x0000, some 0x4000, some 0x0001, some 0x8001,
      none, some 0x7bff, some 0x8000, some 0x3800,
      some 0x0000, some 0x4200, none, some 0x4400,
      some 0x3c00, some 0x8000, some 0x4500 -- 19 entries -> padded to 32
    ]
    checkRoundTrip "mixed19" mixed

    -- Boundary 16 exact
    let boundary16 : Array (Option UInt16) := Id.run do
      let mut arr : Array (Option UInt16) := #[]
      for i in [:16] do
        arr := arr.push (some (0x3c00 + i.toUInt16))
      return arr
    let encFull ← encodeSparseStream boundary16
    -- 16 non-zero slots => 3 + 2*(1 + 16) = 37 bytes
    if encFull.size != 37 then throw s!"boundary16 encoded size {encFull.size} != 37"
    checkRoundTrip "boundary16" boundary16

    -- Deterministic LCG pseudorandom stream (length 137)
    let randStream : Array (Option UInt16) := Id.run do
      let mut arr : Array (Option UInt16) := #[]
      let mut state : Nat := 0x1234567
      for _ in [:137] do
        state := (state * 1103515245 + 12345) % 0x80000000
        let kind := state % 5
        if kind == 0 then arr := arr.push none
        else if kind == 1 then arr := arr.push (some 0x0000)
        else if kind == 2 then arr := arr.push (some 0x8000)
        else arr := arr.push (some ((state / 5) % 65536).toUInt16)
      return arr
    checkRoundTrip "lcg137" randStream

    pure ()
  ))

  -- =========================================================================
  -- M2 3x3 Tiling Validator & Independent O(1) Chooser Tests
  -- =========================================================================

  results := results.push (checkErr "m2 neg: 3x3 validator rejects stride=3" (do
    let sh : Conv3x3Shape := { mode := .dense, h := 8, w := 8, cin := 8, cout := 8, stride := 3, pad := .same }
    let t : Tiling := { logNE := 4, f := 3, fat := 0, ocg := 0, shPref := 0, shMax := 4, tileHeight := 1 }
    validateConv3x3Tiling .h17 sh t
  ) "stride")

  results := results.push (checkErr "m2 neg: 3x3 validator rejects depthwise Cin != Cout" (do
    let sh : Conv3x3Shape := { mode := .depthwise, h := 8, w := 8, cin := 8, cout := 16, stride := 1, pad := .same }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 0, shMax := 4, tileHeight := 8 }
    validateConv3x3Tiling .h17 sh t
  ) "depthwise")

  results := results.push (checkErr "m2 neg: 3x3 validator rejects depthwise ocg != 0" (do
    let sh : Conv3x3Shape := { mode := .depthwise, h := 8, w := 8, cin := 16, cout := 16, stride := 1, pad := .same }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 1, shPref := 0, shMax := 4, tileHeight := 8 }
    validateConv3x3Tiling .h17 sh t
  ) "ocg=0")

  results := results.push (checkErr "m2 neg: 3x3 validator rejects valid pad with H < 3" (do
    let sh : Conv3x3Shape := { mode := .dense, h := 2, w := 8, cin := 8, cout := 8, stride := 1, pad := .valid }
    let t : Tiling := { logNE := 4, f := 3, fat := 0, ocg := 0, shPref := 0, shMax := 4, tileHeight := 1 }
    validateConv3x3Tiling .h17 sh t
  ) "valid-padded")

  results := results.push (checkErr "m2 neg: 3x3 validator rejects WU > 3" (do
    let sh : Conv3x3Shape := { mode := .dense, h := 8, w := 8, cin := 8, cout := 1, stride := 1, pad := .same }
    let t : Tiling := { logNE := 4, f := 0, fat := 0, ocg := 0, shPref := 0, shMax := 4, tileHeight := 8 }
    validateConv3x3Tiling .h17 sh t
  ) "WU=4")

  results := results.push (checkErr "m2 neg: 3x3 validator rejects ocg > 4" (do
    let sh : Conv3x3Shape := { mode := .dense, h := 8, w := 8, cin := 8, cout := 32, stride := 1, pad := .same }
    let t : Tiling := { logNE := 4, f := 0, fat := 1, ocg := 5, shPref := 0, shMax := 4, tileHeight := 8 }
    validateConv3x3Tiling .h17 sh t
  ) "ocg=5")

  results := results.push (checkErr "m2 neg: 3x3 validator rejects shPref > shMax" (do
    let sh : Conv3x3Shape := { mode := .dense, h := 8, w := 8, cin := 8, cout := 16, stride := 1, pad := .same }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 3, shMax := 2, tileHeight := 8 }
    validateConv3x3Tiling .h17 sh t
  ) "shPref")

  results := results.push (checkErr "m2 neg: 3x3 validator rejects coverMin(Wo) > shMax" (do
    -- W=24, s=1, same => Wo=24 => coverMin(24, 128, 0) = 1 > shMax=0
    let sh : Conv3x3Shape := { mode := .dense, h := 8, w := 24, cin := 8, cout := 16, stride := 1, pad := .same }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 0, shMax := 0, tileHeight := 8 }
    validateConv3x3Tiling .h17 sh t
  ) "coverMin")

  results := results.push (checkErr "m2 neg: 3x3 validator rejects TH > Ho" (do
    let sh : Conv3x3Shape := { mode := .dense, h := 8, w := 8, cin := 8, cout := 16, stride := 2, pad := .same }
    -- Ho = 4; TH = 5 rejected
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 0, shMax := 4, tileHeight := 5 }
    validateConv3x3Tiling .h17 sh t
  ) "tileHeight")

  results := results.push (checkErr "m2 neg: 3x3 validator rejects L2 > 2 MiB" (do
    -- W=128, s=1, same, Cin=1024, T=128 => srcPx=128, chs=256, srcRowStride=262144; TH=8 => inRows=8 => 2097152 + resBytes > 2 MiB
    let sh : Conv3x3Shape := { mode := .dense, h := 16, w := 128, cin := 1024, cout := 16, stride := 1, pad := .same }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 0, shMax := 0, tileHeight := 8 }
    validateConv3x3Tiling .h17 sh t
  ) "2 MiB")

  results := results.push (checkErr "m2 neg: 3x3 validator rejects stride-2 Cout > 256" (do
    let sh : Conv3x3Shape := { mode := .dense, h := 8, w := 8, cin := 8, cout := 257, stride := 2, pad := .same }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 4, shPref := 0, shMax := 4, tileHeight := 4 }
    validateConv3x3Tiling .h17 sh t
  ) "coverage")

  results := results.push (checkErr "m2 neg: 3x3 validator rejects H17 dense s=1 ocg=4 (Cout=256)" (do
    let sh : Conv3x3Shape := { mode := .dense, h := 16, w := 16, cin := 64, cout := 256, stride := 1, pad := .same }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 4, shPref := 0, shMax := 4, tileHeight := 16 }
    validateConv3x3Tiling .h17 sh t
  ) "ocg=4")

  results := results.push (checkErr "m2 neg: 3x3 validator rejects H17 Cout=320 with ocg=4 (must use ocg=3)" (do
    let sh : Conv3x3Shape := { mode := .dense, h := 8, w := 8, cin := 8, cout := 320, stride := 1, pad := .same }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 4, shPref := 0, shMax := 4, tileHeight := 8 }
    validateConv3x3Tiling .h17 sh t
  ) "ocg=4")

  results := results.push (checkOk "m2: H16 dense s=1 and H17 dense s=2 still accept ocg=4" (do
    let shS1 : Conv3x3Shape := { mode := .dense, h := 16, w := 16, cin := 64, cout := 256, stride := 1, pad := .same }
    let tS1 : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 4, shPref := 0, shMax := 4, tileHeight := 16 }
    validateConv3x3Tiling .h16 shS1 tS1
    let shS2 : Conv3x3Shape := { mode := .dense, h := 16, w := 16, cin := 64, cout := 256, stride := 2, pad := .same }
    let tS2 : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 4, shPref := 0, shMax := 4, tileHeight := 8 }
    validateConv3x3Tiling .h17 shS2 tS2
    pure ()
  ))

  results := results.push (checkOk "m2: chooseConv3x3Tiling matrix (H16 & H17, dense/dw, s=1/2, same/valid, boundary, multi-tile, Cout=320)" (do
    let commonShapes : Array Conv3x3Shape := #[
      -- Dense stride 1 & 2, same & valid, even/odd/prime, Cout=1, Cin=1, Cout=256
      { mode := .dense, h := 1, w := 1, cin := 1, cout := 1, stride := 1, pad := .same },
      { mode := .dense, h := 1, w := 1, cin := 1, cout := 1, stride := 2, pad := .same },
      { mode := .dense, h := 8, w := 8, cin := 16, cout := 1, stride := 1, pad := .same },
      { mode := .dense, h := 9, w := 13, cin := 7, cout := 11, stride := 1, pad := .same },
      { mode := .dense, h := 9, w := 13, cin := 7, cout := 11, stride := 1, pad := .valid },
      { mode := .dense, h := 16, w := 16, cin := 16, cout := 32, stride := 2, pad := .same },
      { mode := .dense, h := 15, w := 17, cin := 9, cout := 19, stride := 2, pad := .same },
      { mode := .dense, h := 16, w := 16, cin := 16, cout := 32, stride := 2, pad := .valid },
      { mode := .dense, h := 15, w := 17, cin := 9, cout := 1, stride := 2, pad := .valid },
      { mode := .dense, h := 16, w := 32, cin := 64, cout := 256, stride := 1, pad := .same },
      { mode := .dense, h := 16, w := 32, cin := 64, cout := 256, stride := 2, pad := .same },
      -- Tall multi-tile 128x128 forcing TH < Ho
      { mode := .dense, h := 128, w := 128, cin := 64, cout := 64, stride := 1, pad := .same },
      { mode := .dense, h := 128, w := 128, cin := 64, cout := 64, stride := 2, pad := .same },
      -- Depthwise C=1 (normalized to dense), C=7, C=16, C=29, C=64, stride 1 & 2, same & valid
      { mode := .depthwise, h := 8, w := 8, cin := 1, cout := 1, stride := 1, pad := .same },
      { mode := .depthwise, h := 8, w := 8, cin := 1, cout := 1, stride := 2, pad := .same },
      { mode := .depthwise, h := 9, w := 11, cin := 7, cout := 7, stride := 1, pad := .same },
      { mode := .depthwise, h := 9, w := 11, cin := 7, cout := 7, stride := 2, pad := .valid },
      { mode := .depthwise, h := 16, w := 16, cin := 16, cout := 16, stride := 1, pad := .same },
      { mode := .depthwise, h := 16, w := 16, cin := 16, cout := 16, stride := 2, pad := .same },
      { mode := .depthwise, h := 13, w := 19, cin := 29, cout := 29, stride := 1, pad := .valid },
      { mode := .depthwise, h := 13, w := 19, cin := 29, cout := 29, stride := 2, pad := .same },
      { mode := .depthwise, h := 128, w := 128, cin := 64, cout := 64, stride := 1, pad := .same }
    ]
    for sh in commonShapes do
      let t16 ← chooseConv3x3Tiling .h16 sh
      validateConv3x3Tiling .h16 sh t16
      let t17 ← chooseConv3x3Tiling .h17 sh
      validateConv3x3Tiling .h17 sh t17
      -- Verify maxTileHeight3x3 is truly maximal (th + 1 must exceed L2 whenever th < Ho)
      if t17.tileHeight < sh.outH then
        let nextCand : Tiling := { t17 with tileHeight := t17.tileHeight + 1 }
        if l2UseBytes3x3 .h17 sh nextCand <= l2CapacityBytes then
          throw s!"chooser did not find maximal TH for {repr sh}: chose {t17.tileHeight} but {t17.tileHeight + 1} fits"

    -- Verify 128x128 Cin=64 actually triggers multi-tile TH < Ho
    let tallSh : Conv3x3Shape := { mode := .dense, h := 128, w := 128, cin := 64, cout := 64, stride := 1, pad := .same }
    let tTall ← chooseConv3x3Tiling .h17 tallSh
    if tTall.tileHeight >= tallSh.outH then
      throw s!"expected multi-tile TH < 128 for 128x128 Cin=64, got TH={tTall.tileHeight}"

    -- H17 dense stride-1 Cout=256 picks F=4, ocg=3 (while H16 s=1 and H17 s=2 pick F=4, ocg=4)
    let sh256s1 : Conv3x3Shape := { mode := .dense, h := 16, w := 16, cin := 64, cout := 256, stride := 1, pad := .same }
    let t256H17 ← chooseConv3x3Tiling .h17 sh256s1
    if t256H17.f != 4 || t256H17.ocg != 3 then
      throw s!"expected H17 dense s=1 Cout=256 to pick F=4, ocg=3, got F={t256H17.f}, ocg={t256H17.ocg}"
    let t256H16 ← chooseConv3x3Tiling .h16 sh256s1
    if t256H16.f != 4 || t256H16.ocg != 4 then
      throw s!"expected H16 dense s=1 Cout=256 to pick F=4, ocg=4, got F={t256H16.f}, ocg={t256H16.ocg}"
    let sh256s2 : Conv3x3Shape := { sh256s1 with stride := 2 }
    let t256H17s2 ← chooseConv3x3Tiling .h17 sh256s2
    if t256H17s2.f != 4 || t256H17s2.ocg != 4 then
      throw s!"expected H17 dense s=2 Cout=256 to pick F=4, ocg=4, got F={t256H17s2.f}, ocg={t256H17s2.ocg}"

    -- H17 dense stride-1 Cout=320 uses F=4, ocg=3; H16 rejects Cout=320
    let sh320 : Conv3x3Shape := { mode := .dense, h := 8, w := 16, cin := 16, cout := 320, stride := 1, pad := .same }
    let t320 ← chooseConv3x3Tiling .h17 sh320
    validateConv3x3Tiling .h17 sh320 t320
    if t320.f != 4 || t320.ocg != 3 then
      throw s!"expected H17 Cout=320 to pick F=4, ocg=3, got F={t320.f}, ocg={t320.ocg}"
    match chooseConv3x3Tiling .h16 sh320 with
    | .ok _ => throw "expected H16 to reject dense 3x3 Cout=320"
    | .error _ => pure ()

    -- Both H16 and H17 reject dense stride-2 Cout=320
    let sh320s2 : Conv3x3Shape := { sh320 with stride := 2 }
    match chooseConv3x3Tiling .h17 sh320s2 with
    | .ok _ => throw "expected H17 to reject stride-2 dense Cout=320"
    | .error _ => pure ()

    pure ()
  ))

  -- =========================================================================
  -- M2 3x3 Kernel Packer Exact Layout & Sparse Decode Tests
  -- =========================================================================

  results := results.push (checkOk "m2: 3x3 kernel packer (dense s=1/s=2, depthwise F=4/F<4/C=1, H16 rep vs H17)" (do
    -- 1. Depthwise C=1 normalized to dense: scalarBiasMode=true, biasInKernel=false
    let shDw1 : Conv3x3Shape := { mode := .depthwise, h := 6, w := 6, cin := 1, cout := 1, stride := 1, pad := .same }
    let tDw1 ← chooseConv3x3Tiling .h17 shDw1
    let wDw1 := mkTaggedWeights3x3 1 1
    let bDw1 := mkTaggedBias 1
    let pkDw1 ← packConv3x3KernelChunks .h17 shDw1 tDw1 wDw1 (some bDw1)
    if pkDw1.biasInKernel then throw "depthwise C=1 must use scalarBias, not biasInKernel"
    if pkDw1.scalarBiasBits != (← readU16LE bDw1 0) then throw "depthwise C=1 scalarBiasBits mismatch"
    for tp in [:9] do
      let got ← readU16LE pkDw1.bytes (2 * tp)
      let exp ← readU16LE wDw1 (2 * tp)
      if got != exp then throw s!"depthwise C=1 tap {tp} mismatch"

    -- 2. Depthwise C=24 stride-1 (F=4 => group g gets channels g, g+16 as (o,1) blocks)
    let shDw24 : Conv3x3Shape := { mode := .depthwise, h := 8, w := 8, cin := 24, cout := 24, stride := 1, pad := .same }
    let tDw24 ← chooseConv3x3Tiling .h17 shDw24
    let wDw24 := mkTaggedWeights3x3 24 1
    let bDw24 := mkTaggedBias 24
    let pkDw24 ← packConv3x3KernelChunks .h17 shDw24 tDw24 wDw24 (some bDw24)
    if !pkDw24.biasInKernel then throw "depthwise C=24 must embed bias in kernel"
    if pkDw24.slots.size != 16 then throw s!"depthwise C=24 expected 16 slots, got {pkDw24.slots.size}"
    for g in [:16] do
      let cs := pkDw24.slots[g]!
      let blocks := groupOutputBlocks3x3 shDw24 tDw24 g
      let mut pos := cs.offset
      for (o0, n) in blocks do
        if n != 1 then throw s!"depthwise F=4 expected block size 1, got {n}"
        let gotB ← readU16LE pkDw24.bytes pos
        let expB ← readU16LE bDw24 (2 * o0)
        if gotB != expB then throw s!"depthwise C=24 o={o0} bias mismatch"
        pos := pos + 2
        for tp in [:9] do
          let gotW ← readU16LE pkDw24.bytes (pos + 2 * tp)
          let expW ← readU16LE wDw24 (2 * (o0 * 9 + tp))
          if gotW != expW then throw s!"depthwise C=24 o={o0} tp={tp} weight mismatch"
        pos := pos + 18

    -- 3. Dense stride-2 with bias + zeros/-0.0: verify raw bias before u24 + sparse tap-slot decode
    let shS2 : Conv3x3Shape := { mode := .dense, h := 10, w := 10, cin := 3, cout := 5, stride := 2, pad := .same }
    let tS2 : Tiling := { logNE := 4, f := 2, fat := 0, ocg := 1, shPref := 0, shMax := 4, tileHeight := 5 }
    let wS2 := mkTaggedWeights3x3 5 3 (withZeros := true)
    let bS2 := mkTaggedBias 5
    let pkS2 ← packConv3x3KernelChunks .h17 shS2 tS2 wS2 (some bS2)
    if !pkS2.biasInKernel then throw "dense Cout=5 stride-2 must embed bias in kernel"
    let groups := (1 : Nat) <<< tS2.f
    for g in [:groups] do
      let cs := pkS2.slots[g]!
      let blocks := groupOutputBlocks3x3 shS2 tS2 g
      let mut pos := cs.offset
      for (o0, n) in blocks do
        -- Raw uncompressed bias comes immediately before u24
        for k in [:n] do
          let gotB ← readU16LE pkS2.bytes (pos + 2 * k)
          let expB ← readU16LE bS2 (2 * (o0 + k))
          if gotB != expB then throw s!"stride-2 bias mismatch at g={g}, o={o0 + k}"
        pos := pos + 2 * n
        let (dec, consumed) ← decodeSparseStream pkS2.bytes pos
        pos := pos + consumed
        let ev := alignUp n 2
        let mut sIdx := 0
        for ci in [:3] do
          for tpOpt in stride2TapSlots do
            for l in [:ev] do
              let exp : UInt16 :=
                match tpOpt with
                | some tp =>
                  if l < n then
                    let raw := (readU16LE wS2 (2 * (((o0 + l) * 3 + ci) * 9 + tp))).toOption.get!
                    if (raw &&& 0x7fff) != 0 then raw else 0
                  else 0
                | none => 0
              if dec[sIdx]! != exp then
                throw s!"stride-2 sparse slot {sIdx} mismatch at g={g}, ci={ci}, l={l}"
              sIdx := sIdx + 1

    -- 4. H16 rep-replicated chunks vs H17 single-slot chunks (Cout=4, logNE=4, F=2 => rep=4)
    let shRep : Conv3x3Shape := { mode := .dense, h := 8, w := 8, cin := 4, cout := 4, stride := 2, pad := .same }
    let tRep : Tiling := { logNE := 4, f := 2, fat := 0, ocg := 0, shPref := 0, shMax := 4, tileHeight := 4 }
    let wRep := mkTaggedWeights3x3 4 4 (withZeros := true)
    let pk17 ← packConv3x3KernelChunks .h17 shRep tRep wRep none
    let pk16 ← packConv3x3KernelChunks .h16 shRep tRep wRep none
    if pk17.slots.map (·.slot) != #[0, 4, 8, 12] then throw "H17 3x3 slots mismatch"
    if pk16.slots.size != 16 then throw s!"H16 3x3 expected 16 replicated slots, got {pk16.slots.size}"
    for g in [:4] do
      let refChunk := pk17.bytes.extract pk17.slots[g]!.offset (pk17.slots[g]!.offset + pk17.slots[g]!.size)
      for j in [:4] do
        let cs := pk16.slots[g * 4 + j]!
        let gotChunk := pk16.bytes.extract cs.offset (cs.offset + cs.size)
        if gotChunk != refChunk then throw s!"H16 3x3 replicated chunk mismatch at slot {cs.slot}"

    pure ()
  ))

  -- =========================================================================
  -- M2 Full compileConv3x3 Round-Trip & Unified Static Verifier Tests
  -- =========================================================================

  results := results.push (checkOk "m2: compileConv3x3 round-trip matrix (H16 & H17, dense/dw, s=1/2, same/valid, bias/relu, wide3x3)" (do
    let cases : Array (Target × Conv3x3Shape × Bool × Bool) := #[
      -- H17 dense s=1 & s=2, same & valid, Cout=1, Cin=1, Cout=320, wide3x3=true & false
      (.h17, { mode := .dense, h := 8, w := 16, cin := 16, cout := 16, stride := 1, pad := .same }, true, true),
      (.h17, { mode := .dense, h := 9, w := 13, cin := 7, cout := 11, stride := 1, pad := .valid }, false, false),
      (.h17, { mode := .dense, h := 8, w := 16, cin := 16, cout := 1, stride := 1, pad := .same }, true, false),
      (.h17, { mode := .dense, h := 8, w := 16, cin := 16, cout := 1, stride := 2, pad := .same }, true, true),
      (.h17, { mode := .dense, h := 16, w := 16, cin := 8, cout := 16, stride := 2, pad := .same }, true, false),
      (.h17, { mode := .dense, h := 15, w := 17, cin := 9, cout := 13, stride := 2, pad := .same }, false, true),
      (.h17, { mode := .dense, h := 14, w := 18, cin := 6, cout := 12, stride := 2, pad := .valid }, true, true),
      (.h17, { mode := .dense, h := 8, w := 16, cin := 16, cout := 320, stride := 1, pad := .same }, true, true),
      (.h17, { mode := .dense, h := 16, w := 16, cin := 64, cout := 256, stride := 1, pad := .same }, true, false),
      -- H17 depthwise C=1 (lowered to dense), C=7, C=16, C=29, s=1 & s=2, same & valid
      (.h17, { mode := .depthwise, h := 8, w := 8, cin := 1, cout := 1, stride := 1, pad := .same }, true, true),
      (.h17, { mode := .depthwise, h := 8, w := 8, cin := 1, cout := 1, stride := 2, pad := .same }, true, false),
      (.h17, { mode := .depthwise, h := 9, w := 11, cin := 7, cout := 7, stride := 1, pad := .same }, true, false),
      (.h17, { mode := .depthwise, h := 12, w := 14, cin := 16, cout := 16, stride := 2, pad := .same }, true, true),
      (.h17, { mode := .depthwise, h := 13, w := 15, cin := 29, cout := 29, stride := 1, pad := .valid }, false, true),
      (.h17, { mode := .depthwise, h := 13, w := 15, cin := 29, cout := 29, stride := 2, pad := .valid }, true, false),
      -- H16 dense & depthwise, s=1 & s=2, same & valid, C=1, Cout=1
      (.h16, { mode := .dense, h := 8, w := 16, cin := 16, cout := 16, stride := 1, pad := .same }, true, true),
      (.h16, { mode := .dense, h := 11, w := 13, cin := 5, cout := 9, stride := 2, pad := .same }, true, false),
      (.h16, { mode := .dense, h := 10, w := 12, cin := 8, cout := 1, stride := 2, pad := .valid }, false, true),
      (.h16, { mode := .depthwise, h := 8, w := 8, cin := 1, cout := 1, stride := 1, pad := .same }, true, false),
      (.h16, { mode := .depthwise, h := 10, w := 12, cin := 24, cout := 24, stride := 1, pad := .same }, true, true),
      (.h16, { mode := .depthwise, h := 10, w := 12, cin := 24, cout := 24, stride := 2, pad := .valid }, false, false)
    ]
    for (target, sh, withBias, relu) in cases do
      let cinW := if sh.mode == .depthwise then 1 else sh.cin
      let ws := mkTaggedWeights3x3 sh.cout cinW (withZeros := sh.stride == 2)
      let bs? := if withBias then some (mkTaggedBias sh.cout) else none
      let compiled ← compileConv3x3 target sh ws bs? relu
      let info ← validateHwx compiled.hwx
      validateSectionsWithInfo info compiled.sections
      let nVerified ← verifyConvHwxTasks compiled.hwx info (requireConvTask := true)
      if nVerified != 1 then throw "expected 1 verified 3x3 conv task"

      -- Check wide3x3 flag exact semantics
      let expWide := computeWide3x3 target sh compiled.tiling
      if compiled.wide3x3 != expWide then
        throw s!"wide3x3 mismatch for {target.name} {repr sh}: got {compiled.wide3x3}, expected {expWide}"
      if target == .h16 && compiled.wide3x3 then throw "wide3x3 must be false on H16"
      if sh.effMode == .depthwise && compiled.wide3x3 then throw "wide3x3 must be false for depthwise"
      if sh.stride == 2 && compiled.wide3x3 then throw "wide3x3 must be false for stride 2"

      -- Check H16 16 KiB surface section rounding vs H17 exact span
      for surf in info.surfaces do
        match target with
        | .h16 =>
          if surf.sect.size != surf.vmSize || surf.sect.size.toNat % 0x4000 != 0 then
            throw s!"H16 surface '{surf.shortName}' section size {surf.sect.size} not 16 KiB rounded"
        | .h17 =>
          if surf.sect.size != surf.span then
            throw s!"H17 surface '{surf.shortName}' section size {surf.sect.size} != span {surf.span}"

    -- Assert H17 dense s=1 (16x16, Cin=64, Cout=256) compiles with F=4, ocg=3
    let sh256 : Conv3x3Shape := { mode := .dense, h := 16, w := 16, cin := 64, cout := 256, stride := 1, pad := .same }
    let c256 ← compileConv3x3 .h17 sh256 (mkTaggedWeights3x3 256 64) none false
    if c256.tiling.f != 4 || c256.tiling.ocg != 3 then
      throw s!"expected H17 dense s=1 Cout=256 to compile with F=4, ocg=3, got F={c256.tiling.f}, ocg={c256.tiling.ocg}"

    -- Exact 64 KiB chunk boundary (F=4): (4x8, Cin=455, Cout=128, s=1, same, bias+relu)
    let sh455 : Conv3x3Shape := { mode := .dense, h := 4, w := 8, cin := 455, cout := 128, stride := 1, pad := .same }
    let c455 ← compileConv3x3 .h17 sh455 (mkTaggedWeights3x3 128 455) (some (mkTaggedBias 128)) true
    if !c455.wide3x3 then throw "expected wide3x3=true for (4x8, 455->128)"
    if c455.tiling.f != 4 || c455.tiling.ocg != 3 then
      throw s!"expected F=4, ocg=3 for (4x8, 455->128), got F={c455.tiling.f}, ocg={c455.tiling.ocg}"
    if c455.kernelBytes != 16 * maxKernelChunkBytes then
      throw s!"expected 16*65536B kernel for (4x8, 455->128), got {c455.kernelBytes}"

    -- Exact 64 KiB chunk boundary (F=0): (4x8, Cin=3640, Cout=1, s=1, same, bias+relu)
    let sh3640 : Conv3x3Shape := { mode := .dense, h := 4, w := 8, cin := 3640, cout := 1, stride := 1, pad := .same }
    let c3640 ← compileConv3x3 .h17 sh3640 (mkTaggedWeights3x3 1 3640) (some (mkTaggedBias 1)) true
    if !c3640.wide3x3 then throw "expected wide3x3=true for (4x8, 3640->1)"
    if c3640.tiling.f != 0 then throw s!"expected F=0 for (4x8, 3640->1), got F={c3640.tiling.f}"
    if c3640.kernelBytes != maxKernelChunkBytes then
      throw s!"expected exact 65536B chunk for (4x8, 3640->1), got {c3640.kernelBytes}"

    -- Staged-mode (direct=false) depthwise 3x3: (3x128, C=5500, s=1, valid, bias+relu)
    let shStgDw : Conv3x3Shape := { mode := .depthwise, h := 3, w := 128, cin := 5500, cout := 5500, stride := 1, pad := .valid }
    let cStgDw ← compileConv3x3 .h17 shStgDw (mkTaggedWeights3x3 5500 1) (some (mkTaggedBias 5500)) true
    if cStgDw.direct then throw "expected direct=false (staged mode) for depthwise (3x128, C=5500)"
    if cStgDw.l2UseBytes > l2CapacityBytes then
      throw s!"expected l2 <= 2 MiB for depthwise (3x128, C=5500), got {cStgDw.l2UseBytes}"
    if cStgDw.wide3x3 then throw "expected wide3x3=false for depthwise (3x128, C=5500)"

    -- Multi-round boundary: (16x16, Cin=227, Cout=256, s=1, same, bias+relu) => 65,408 B/chunk <= 64 KiB
    let sh227 : Conv3x3Shape := { mode := .dense, h := 16, w := 16, cin := 227, cout := 256, stride := 1, pad := .same }
    let c227 ← compileConv3x3 .h17 sh227 (mkTaggedWeights3x3 256 227) (some (mkTaggedBias 256)) true
    if !c227.wide3x3 then throw "expected wide3x3=true for (16x16, 227->256)"
    if c227.tiling.f != 4 || c227.tiling.ocg != 3 then
      throw s!"expected F=4, ocg=3 for (16x16, 227->256), got F={c227.tiling.f}, ocg={c227.tiling.ocg}"
    if c227.kernelBytes != 16 * 65408 then
      throw s!"expected 16*65408B kernel for (16x16, 227->256), got {c227.kernelBytes}"

    -- Legal H17 dense 3x3 stride-1 with wide3x3=false: (4x8, Cin=2000, Cout=16, s=1, same, bias) => 36,032 B/chunk
    let sh2000 : Conv3x3Shape := { mode := .dense, h := 4, w := 8, cin := 2000, cout := 16, stride := 1, pad := .same }
    let c2000 ← compileConv3x3 .h17 sh2000 (mkTaggedWeights3x3 16 2000) (some (mkTaggedBias 16)) false
    if c2000.wide3x3 then throw "expected wide3x3=false for (4x8, 2000->16) (maxBlockSlots=36000 > 32768)"
    if c2000.kernelBytes != 16 * 36032 then
      throw s!"expected 16*36032B kernel for (4x8, 2000->16), got {c2000.kernelBytes}"
    pure ()
  ))

  -- Per-chunk 64 KiB cap negative tests (3x3 and 1x1)
  results := results.push (checkErr "m2 neg: H17 rejects (4x8, Cin=460, Cout=128, s=1, same) (> 64 KiB chunk)" (do
    let sh460 : Conv3x3Shape := { mode := .dense, h := 4, w := 8, cin := 460, cout := 128, stride := 1, pad := .same }
    compileConv3x3 .h17 sh460 (mkTaggedWeights3x3 128 460) none false
  ) "no legal")

  results := results.push (checkErr "m2 neg: H17 rejects (16x16, Cin=256, Cout=256, s=1, same) (73,792 B > 64 KiB chunk)" (do
    let sh256x256 : Conv3x3Shape := { mode := .dense, h := 16, w := 16, cin := 256, cout := 256, stride := 1, pad := .same }
    compileConv3x3 .h17 sh256x256 (mkTaggedWeights3x3 256 256) (some (mkTaggedBias 256)) false
  ) "no legal")

  results := results.push (checkErr "m2 neg: H17 rejects (4x8, Cin=4900, Cout=1, s=1, same) (> 64 KiB chunk)" (do
    let sh4900 : Conv3x3Shape := { mode := .dense, h := 4, w := 8, cin := 4900, cout := 1, stride := 1, pad := .same }
    compileConv3x3 .h17 sh4900 (mkTaggedWeights3x3 1 4900) none false
  ) "no legal")

  results := results.push (checkErr "m2 neg: 3x3 validator rejects Rule 13 (> 64 KiB dense chunk)" (do
    let sh4900 : Conv3x3Shape := { mode := .dense, h := 4, w := 8, cin := 4900, cout := 1, stride := 1, pad := .same }
    let t : Tiling := { logNE := 4, f := 0, fat := 1, ocg := 0, shPref := 0, shMax := 4, tileHeight := 4 }
    validateConv3x3Tiling .h17 sh4900 t
  ) "64 KiB")

  results := results.push (checkErr "m2 neg: 1x1 validator rejects chunk > 64 KiB" (do
    let sh1x1Big : Conv1x1Shape := { h := 4, w := 8, cin := 4096, cout := 256 }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 4, shPref := 0, shMax := 4, tileHeight := 4 }
    validateTiling .h17 sh1x1Big t
  ) "64 KiB")

  results := results.push (checkErr "m2 neg: 1x1 packer rejects bias pushing chunk > 64 KiB" (do
    -- 2 * 2048 * 16 = 65536 (weights alone fit), + 16*2 = 32B bias -> 65568 -> padded 65600 > 65536
    let sh1x1Bias : Conv1x1Shape := { h := 4, w := 8, cin := 2048, cout := 256 }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 4, shPref := 0, shMax := 4, tileHeight := 4 }
    packKernelChunks .h17 sh1x1Bias t (mkTaggedWeights 256 2048) (some (mkTaggedBias 256))
  ) "64 KiB")

  -- =========================================================================
  -- M2 Tampered-Task Negative Verifier Tests
  -- =========================================================================

  let base3DenseS1 ← unwrap "base3DenseS1" (compileConv3x3 .h17
    { mode := .dense, h := 8, w := 16, cin := 16, cout := 16, stride := 1, pad := .same }
    (mkTaggedWeights3x3 16 16) (some (mkTaggedBias 16)) true)

  let base3DenseS2 ← unwrap "base3DenseS2" (compileConv3x3 .h17
    { mode := .dense, h := 16, w := 16, cin := 8, cout := 16, stride := 2, pad := .same }
    (mkTaggedWeights3x3 16 8 true) (some (mkTaggedBias 16)) false)

  let base3Dw ← unwrap "base3Dw" (compileConv3x3 .h17
    { mode := .depthwise, h := 8, w := 16, cin := 16, cout := 16, stride := 1, pad := .same }
    (mkTaggedWeights3x3 16 1) (some (mkTaggedBias 16)) false)

  let base3H16 ← unwrap "base3H16" (compileConv3x3 .h16
    { mode := .dense, h := 8, w := 16, cin := 16, cout := 16, stride := 1, pad := .same }
    (mkTaggedWeights3x3 16 16) none false)

  let base3Boundary455 ← unwrap "base3Boundary455" (compileConv3x3 .h17
    { mode := .dense, h := 4, w := 8, cin := 455, cout := 128, stride := 1, pad := .same }
    (mkTaggedWeights3x3 128 455) (some (mkTaggedBias 128)) true)

  let base3Wide0 ← unwrap "base3Wide0" (compileConv3x3 .h17
    { mode := .dense, h := 4, w := 8, cin := 2000, cout := 16, stride := 1, pad := .same }
    (mkTaggedWeights3x3 16 2000) (some (mkTaggedBias 16)) false)

  let base1x1Boundary ← unwrap "base1x1Boundary" (compileConv1x1 .h17
    { h := 4, w := 8, cin := 2048, cout := 256 }
    (mkTaggedWeights 256 2048) none false)

  results := results.push (checkErr "m2 neg verifier: 1x1 chunkSize > 65536 rejected (even within __KERN_0 bounds)" (do
    tamperRegAndVerify base1x1Boundary (regChunkSize 0) (maxKernelChunkBytes + 64).toUInt32
  ) "64 KiB")

  results := results.push (checkErr "m2 neg verifier: sparse mismatch (neCfg=1 vs kernelStage=0 on s=1)" (do
    tamperReg3x3AndVerify base3DenseS1 regNeCfg 0x10000182
  ) "sparse")

  results := results.push (checkErr "m2 neg verifier: sparse mismatch (neCfg=1 vs kernelStage=0 on s=2)" (do
    tamperReg3x3AndVerify base3DenseS2 regKernelStage 0x000102c0
  ) "sparse")

  results := results.push (checkErr "m2 neg verifier: wide3x3=1 on H16 rejected" (do
    let fOff ← findRegFileOffset base3H16.hwx base3H16.info regNeCommon
    let cur ← readU32LE base3H16.hwx fOff
    tamperReg3x3AndVerify base3H16 regNeCommon (cur ||| ((1 : UInt32) <<< 27))
  ) "wide3x3")

  results := results.push (checkErr "m2 neg verifier: wide3x3=1 on stride-2 rejected" (do
    let fOff ← findRegFileOffset base3DenseS2.hwx base3DenseS2.info regNeCommon
    let cur ← readU32LE base3DenseS2.hwx fOff
    tamperReg3x3AndVerify base3DenseS2 regNeCommon (cur ||| ((1 : UInt32) <<< 27))
  ) "wide3x3")

  results := results.push (checkErr "m2 neg verifier: wide3x3=1 on depthwise rejected" (do
    let fOff ← findRegFileOffset base3Dw.hwx base3Dw.info regNeCommon
    let cur ← readU32LE base3Dw.hwx fOff
    tamperReg3x3AndVerify base3Dw regNeCommon (cur ||| ((1 : UInt32) <<< 27))
  ) "wide3x3")

  results := results.push (checkErr "m2 neg verifier: wide3x3=1 exceeding maxBlockSlots rejected" (do
    let fOff ← findRegFileOffset base3Wide0.hwx base3Wide0.info regNeCommon
    let cur ← readU32LE base3Wide0.hwx fOff
    tamperReg3x3AndVerify base3Wide0 regNeCommon (cur ||| ((1 : UInt32) <<< 27))
  ) "maxBlockSlots")

  results := results.push (checkOk "m2 pos verifier: wide3x3=0 accepted on H17 dense stride-1 (4x8, 2000->16)" (do
    let fOff ← findRegFileOffset base3DenseS1.hwx base3DenseS1.info regNeCommon
    let cur ← readU32LE base3DenseS1.hwx fOff
    tamperReg3x3AndVerify base3DenseS1 regNeCommon (cur &&& (~~~((1 : UInt32) <<< 27)))
  ))

  results := results.push (checkErr "m2 neg verifier: chunkSize > 65536 rejected (even within __KERN_0 bounds)" (do
    -- base3Boundary455 has kernelBytes = 16 * 65536, so chunk 0 at offset 0 with size 65600 fits in __KERN_0
    -- yet violates the 64 KiB per-chunk hardware cap.
    tamperReg3x3AndVerify base3Boundary455 (regChunkSize 0) (maxKernelChunkBytes + 64).toUInt32
  ) "64 KiB")

  results := results.push (checkErr "m2 neg verifier: H17 dense s=1 ocg=4 rejected" (do
    let fOff ← findRegFileOffset base3DenseS1.hwx base3DenseS1.info regNeTile
    let cur ← readU32LE base3DenseS1.hwx fOff
    -- Replace low 3 bits (ocg) with 4
    tamperReg3x3AndVerify base3DenseS1 regNeTile ((cur &&& 0xfffffff8) ||| 0x4)
  ) "ocg=4")

  results := results.push (checkErr "m2 neg verifier: dwGroups with ocg != 0 rejected" (do
    let fOff ← findRegFileOffset base3Dw.hwx base3Dw.info regNeTile
    let cur ← readU32LE base3Dw.hwx fOff
    tamperReg3x3AndVerify base3Dw regNeTile (cur ||| 0x1)
  ) "ocg=0")

  results := results.push (checkErr "m2 neg verifier: L2 overflow rejected" (do
    tamperReg3x3AndVerify base3DenseS1 regResBase 0x001fffe0
  ) "exceeds 2 MiB")

  results := results.push (checkErr "m2 neg verifier: L2 overlap rejected" (do
    tamperReg3x3AndVerify base3DenseS1 regResBase 0
  ) "overlapping L2")

  results := results.push (checkErr "m2 neg verifier: disabled chunk slot with size written rejected" (do
    tamperReg3x3AndVerify base3DenseS1 (regChunkSlot 0) 0x00000020
  ) "enable bit 0")

  results := results.push (checkErr "m2 neg verifier: wrong srcPx narrowing rejected" (do
    let fOff ← findRegFileOffset base3DenseS1.hwx base3DenseS1.info regSrc1ChanStride
    let cur ← readU32LE base3DenseS1.hwx fOff
    tamperReg3x3AndVerify base3DenseS1 regSrc1ChanStride (cur - 16)
  ) "srcPx")

  -- =========================================================================
  -- M2.5 Architecture Refactor: Silicon Safety vs Chooser Policy & Program IR
  -- =========================================================================

  results := results.push (checkOk "m2.5: Vane.Verify accepts silicon-safe task with relaxed chooser policy (F > ceil_log2(Cout))" (do
    -- Cout = 4 has ceil_log2(4) = 2, so F = 3 violates chooser policy (F > ceil_log2(Cout))
    -- yet is 100% silicon-safe (logNE=4, F=3, fat=0, WU=1 <= 3, rep=2, 4 active groups + 4 empty groups).
    let sh3 : ConvSpec := {
      kernelSize := 3, mode := .dense, h := 8, w := 16, cin := 8, cout := 4, stride := 1, pad := .same
    }
    let tRelaxed : Tiling := {
      logNE := 4, f := 3, fat := 0, ocg := 0, shPref := 0, shMax := 4, tileHeight := 8
    }
    -- 1. Safety validator accepts; chooser policy rejects
    validateConvSafety .h17 sh3 tRelaxed
    match validateConvTiling .h17 sh3 tRelaxed with
    | .ok () => throw "expected validateConvTiling to reject F=3 > ceil_log2(Cout=4)=2"
    | .error _ => pure ()
    -- 2. Lower via lowerConvTask + emitProgram and verify with standalone Vane.Verify
    let ws := mkTaggedWeights3x3 4 8
    let bs := mkTaggedBias 4
    let packed ← packConvKernelChunks .h17 sh3 tRelaxed ws (some bs)
    let td ← lowerConvTask .h17 sh3 tRelaxed packed true
    let inSurf ← makeFp16Surface .h17 "x" "input_x" .input sh3.cin sh3.h sh3.w
    let outSurf ← makeFp16Surface .h17 "y" "output_y" .output sh3.cout sh3.outH sh3.outW
    let emitted ← emitProgram {
      target := .h17
      surfaces := #[inSurf, outSurf]
      tasks := #[td]
      kernel := some packed.bytes
    }
    let nChecked ← verifyConvHwxTasks emitted.hwx emitted.info (requireConvTask := true)
    if nChecked != 1 then throw s!"expected 1 verified task, got {nChecked}"

    -- Same check for 1x1 conv with F=3 > ceil_log2(Cout=4)=2
    let sh1 : ConvSpec := {
      kernelSize := 1, mode := .dense, h := 8, w := 16, cin := 8, cout := 4, stride := 1, pad := .same
    }
    validateConvSafety .h17 sh1 tRelaxed
    match validateConvTiling .h17 sh1 tRelaxed with
    | .ok () => throw "expected validateConvTiling to reject 1x1 F=3 > ceil_log2(Cout=4)=2"
    | .error _ => pure ()
    let ws1 := mkTaggedWeights 4 8
    let packed1 ← packConvKernelChunks .h17 sh1 tRelaxed ws1 none
    let td1 ← lowerConvTask .h17 sh1 tRelaxed packed1 false
    let emitted1 ← emitProgram {
      target := .h17
      surfaces := #[inSurf, outSurf]
      tasks := #[td1]
      kernel := some packed1.bytes
    }
    let nChecked1 ← verifyConvHwxTasks emitted1.hwx emitted1.info (requireConvTask := true)
    if nChecked1 != 1 then throw s!"expected 1 verified 1x1 task, got {nChecked1}"
    pure ()
  ))

  results := results.push (checkOk "m2.5: generic emitProgram supports multi-task ProgramSpec (1x1 + 3x3 verified)" (do
    let spec : Vane.Graph.GraphSpec := {
      target := .h17
      inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
      nodes := #[
        { id := "c1", inputs := #["x"], op := .conv1x1 16 16 (mkTaggedWeights 16 16) none .none },
        { id := "c2", inputs := #["c1"], op := .conv3x3 .dense 16 16 1 .same (mkTaggedWeights3x3 16 16) none .relu }
      ]
    }
    let compiled ← Vane.Graph.compileGraph spec
    if compiled.info.taskStream.taskCount != 2 then
      throw s!"expected 2 tasks, got {compiled.info.taskStream.taskCount}"
    let nChecked ← verifyConvHwxTasks compiled.hwx compiled.info (requireConvTask := true)
    if nChecked != 2 then throw s!"expected 2 verified conv tasks, got {nChecked}"
    pure ()
  ))

  -- =========================================================================
  -- M3 Checked Nat -> UInt32/UInt64 Overflow Tests
  -- =========================================================================

  results := results.push (checkErr "m3 neg: natToU32Checked rejects 1 <<< 32" (do
    natToU32Checked ((1 : Nat) <<< 32) "test"
  ) "overflows UInt32")

  results := results.push (checkErr "m3 neg: natToU64Checked rejects 1 <<< 64" (do
    natToU64Checked ((1 : Nat) <<< 64) "test"
  ) "overflows UInt64")

  results := results.push (checkErr "m3 neg: encodeTaskHeader9 rejects taskId = 1 <<< 32 (no silent wrap to 0)" (do
    encodeTaskHeader9 ((1 : Nat) <<< 32) 64 4
  ) "taskId")

  results := results.push (checkErr "m3 neg: encodeTaskHeader9 rejects taskId = 65536" (do
    encodeTaskHeader9 65536 64 4
  ) "taskId")

  results := results.push (checkErr "m3 neg: makeFp16Surface rejects huge H = 1 <<< 32" (do
    makeFp16Surface .h17 "x" "input_x" .input 1 ((1 : Nat) <<< 32) 8
  ) "overflows UInt32")

  results := results.push (checkErr "m3 neg: makeFp16Surface rejects huge W = 1 <<< 32" (do
    makeFp16Surface .h17 "x" "input_x" .input 1 8 ((1 : Nat) <<< 32)
  ) "overflows UInt32")

  results := results.push (checkErr "m3 neg: makeFp16Surface rejects UInt64 span overflow" (do
    makeFp16Surface .h17 "x" "input_x" .input 0xffffffff 0xffffffff 0x80000000
  ) "overflows UInt64")

  results := results.push (checkErr "m3 neg: lowerConvTask rejects taskId = 1 <<< 32" (do
    let sh : ConvSpec := { kernelSize := 1, h := 8, w := 16, cin := 16, cout := 16 }
    let t ← chooseConvTiling .h17 sh
    let pk ← packConvKernelChunks .h17 sh t (mkTaggedWeights 16 16) none
    lowerConvTask .h17 sh t pk false (taskId := (1 : Nat) <<< 32)
  ) "overflows UInt32")

  -- =========================================================================
  -- B1-B4 / M1 / M2 / N5 Comprehensive Negative Verifier & Safety Tests
  -- =========================================================================

  -- B2: shMax > 4 and (fat=1, shPref=5) rejected by validateConvSafety directly
  results := results.push (checkErr "b2 neg: validateConvSafety rejects shMax=5" (do
    let sh : ConvSpec := { kernelSize := 1, h := 8, w := 16, cin := 16, cout := 16 }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 0, shMax := 5, tileHeight := 8 }
    validateConvSafety .h17 sh t
  ) "shMax=5")

  results := results.push (checkErr "b2 neg: validateConvSafety rejects (fat=1, shPref=5, shMax=5)" (do
    let sh : ConvSpec := { kernelSize := 1, h := 8, w := 16, cin := 16, cout := 16 }
    let t : Tiling := { logNE := 4, f := 3, fat := 1, ocg := 1, shPref := 5, shMax := 5, tileHeight := 8 }
    validateConvSafety .h17 sh t
  ) "shMax=5")

  -- Helper: emit a custom single-task HWX from a modified TaskDesc/surfaces/reloc flag and run verifyConvHwxTasks
  let emitTaskAndVerify
      (target : Target) (inSurf outSurf : SurfaceSpec) (kernel : ByteArray)
      (td : TaskDesc) (emitReloc : Bool := true)
      (extraRecs : Array RegRecord := #[])
      (headerPatch? : Option (Nat × UInt32) := none) : Except String Unit := do
    let (plainRecs, barRecs) ← packTaskRecordsSplit td.plainWrites td.barWrites
    let allRecs := plainRecs ++ barRecs ++ extraRecs
    let totalTaskWords := 9 + recordWordsCount allRecs
    let mut hdr ← encodeTaskHeader9 td.taskId totalTaskWords td.logNE
    if let some (idx, val) := headerPatch? then
      hdr := hdr.set! idx val
    let prologue : UInt32 := match target with | .h16 => 1 | .h17 => 0
    let stream : TaskStream := { prologue := prologue, tasks := #[{ header := hdr, records := allRecs }] }
    let encStream ← encodeTaskStream target stream
    let mut relocs : Array RelocationSpec := #[]
    if emitReloc then
      for rel in td.kernelRelocs do
        let some (_, _, _, patchOff) :=
          encStream.payloadSites.find? (fun (tIdx, rIdx, pIdx, _) =>
            tIdx == 0 && rIdx == plainRecs.size + rel.barRecIndex && pIdx == rel.payloadWordIndex)
          | throw "missing reloc site"
        relocs := relocs.push { patchOffset := patchOff.toUInt32 }
    let hwx ← emitHwx {
      target := target
      surfaces := #[inSurf, outSurf]
      constants := zeroBytes textConstBytes
      kernel := some kernel
      taskStream := stream
      relocations := relocs
      ident := some "vane"
    }
    let info ← validateHwx hwx
    let _ ← verifyConvHwxTasks hwx info (requireConvTask := true)
    pure ()

  -- Baseline components for structural negative tests
  let baseSh1 : ConvSpec := { kernelSize := 1, h := 8, w := 16, cin := 16, cout := 16 }
  let baseT1 ← unwrap "baseT1" (chooseConvTiling .h17 baseSh1)
  let basePk1 ← unwrap "basePk1" (packConvKernelChunks .h17 baseSh1 baseT1 (mkTaggedWeights 16 16) (some (mkTaggedBias 16)))
  let baseTd1 ← unwrap "baseTd1" (lowerConvTask .h17 baseSh1 baseT1 basePk1 true)
  let baseIn1 ← unwrap "baseIn1" (makeFp16Surface .h17 "x" "input_x" .input 16 8 16)
  let baseOut1 ← unwrap "baseOut1" (makeFp16Surface .h17 "y" "output_y" .output 16 8 16)

  let baseSh1Cout1 : ConvSpec := { kernelSize := 1, h := 4, w := 8, cin := 16, cout := 1 }
  let baseT1C1 ← unwrap "baseT1C1" (chooseConvTiling .h17 baseSh1Cout1)
  let basePk1C1 ← unwrap "basePk1C1" (packConvKernelChunks .h17 baseSh1Cout1 baseT1C1 (mkTaggedWeights 1 16) (some (mkTaggedBias 1)))
  let baseTd1C1 ← unwrap "baseTd1C1" (lowerConvTask .h17 baseSh1Cout1 baseT1C1 basePk1C1 false)
  let baseIn1C1 ← unwrap "baseIn1C1" (makeFp16Surface .h17 "x" "input_x" .input 16 4 8)
  let baseOut1C1 ← unwrap "baseOut1C1" (makeFp16Surface .h17 "y" "output_y" .output 1 4 8)

  let baseSh3Dw1 : ConvSpec := { kernelSize := 3, mode := .depthwise, h := 8, w := 8, cin := 1, cout := 1, stride := 1, pad := .same }
  let baseT3Dw1 ← unwrap "baseT3Dw1" (chooseConvTiling .h17 baseSh3Dw1)
  let basePk3Dw1 ← unwrap "basePk3Dw1" (packConvKernelChunks .h17 baseSh3Dw1 baseT3Dw1 (mkTaggedWeights3x3 1 1) none)
  let baseTd3Dw1 ← unwrap "baseTd3Dw1" (lowerConvTask .h17 baseSh3Dw1 baseT3Dw1 basePk3Dw1 false)
  let baseIn3Dw1 ← unwrap "baseIn3Dw1" (makeFp16Surface .h17 "x" "input_x" .input 1 8 8)
  let baseOut3Dw1 ← unwrap "baseOut3Dw1" (makeFp16Surface .h17 "y" "output_y" .output 1 8 8)

  -- B1: non-zero src1BaseLo / resBaseLo, missing/mismatched/nonzero *BaseHi, kernelBaseLo overflow
  results := results.push (checkErr "m2 neg verifier: non-zero misaligned src1BaseLo rejected" (do
    tamperRegAndVerify baseConv regSrc1BaseLo 32
  ) "64-byte aligned")

  results := results.push (checkErr "m2 neg verifier: non-zero aligned src1BaseLo overflowing surface rejected" (do
    tamperRegAndVerify baseConv regSrc1BaseLo 64
  ) "input DMA range")

  results := results.push (checkErr "m2 neg verifier: non-zero aligned resBaseLo overflowing surface rejected" (do
    tamperRegAndVerify baseConv regResBaseLo 64
  ) "output DMA range")

  results := results.push (checkErr "m2 neg verifier: non-zero src1BaseHi rejected" (do
    tamperRegAndVerify baseConv regSrc1BaseHi 1
  ) "input DMA range")

  results := results.push (checkErr "m2 neg verifier: non-zero resBaseHi rejected" (do
    tamperRegAndVerify baseConv regResBaseHi 1
  ) "output DMA range")

  results := results.push (checkErr "m2 neg verifier: non-zero kernelBaseHi rejected" (do
    tamperRegAndVerify baseConv regKernelBaseHi 1
  ) "exceeds kernel size")

  results := results.push (checkErr "m2 neg verifier: kernelBaseLo relocation addend overflowing __KERN_0 rejected" (do
    tamperRegAndVerify baseConv regKernelBaseLo 64
  ) "exceeds kernel size")

  results := results.push (checkErr "m2 neg verifier: missing src1BaseHi rejected" (do
    let barWrites := baseTd1.barWrites.map fun br =>
      if br.baseByteAddr == regSrc1BaseLo then { br with values := #[0] } else br
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes { baseTd1 with barWrites := barWrites }
  ) "src1BaseHi")

  results := results.push (checkErr "m2 neg verifier: missing resBaseHi rejected" (do
    let barWrites := baseTd1.barWrites.map fun br =>
      if br.baseByteAddr == regResBaseLo then { br with values := #[0] } else br
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes { baseTd1 with barWrites := barWrites }
  ) "resBaseHi")

  results := results.push (checkErr "m2 neg verifier: missing kernelBaseHi rejected" (do
    let barWrites := baseTd1.barWrites.map fun br =>
      if br.baseByteAddr == regKernelBaseLo then { br with values := #[0] } else br
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes { baseTd1 with barWrites := barWrites }
  ) "kernelBaseHi")

  results := results.push (checkErr "m2 neg verifier: mismatched BAR slot between src1BaseLo and src1BaseHi rejected" (do
    let mut barWrites : Array BarRegWrite := #[]
    for br in baseTd1.barWrites do
      if br.baseByteAddr == regSrc1BaseLo then
        barWrites := barWrites.push { baseByteAddr := regSrc1BaseLo, barSlot := 6, values := #[0] }
        barWrites := barWrites.push { baseByteAddr := regSrc1BaseHi, barSlot := 7, values := #[0] }
      else
        barWrites := barWrites.push br
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes { baseTd1 with barWrites := barWrites }
  ) "BAR slot mismatch")

  -- B2: shMax=5, (fat=1, shPref=5, shMax=5), shMin!=shPref, WU+F+fat!=logNE
  results := results.push (checkErr "m2 neg verifier: shMax=5 rejected" (do
    let wBad ← encodeWord neCommonLayout #[
      ("kind", 0), ("shPref", 0), ("shMin", 0), ("shMax", 5), ("F", 4), ("wide3x3", 0)
    ]
    tamperRegAndVerify baseConv regNeCommon wBad
  ) "shMax=5")

  results := results.push (checkErr "m2 neg verifier: (fat=1, shPref=5, shMax=5) rejected" (do
    let fOffCom ← findRegFileOffset baseConv.hwx baseConv.info regNeCommon
    let fOffTile ← findRegFileOffset baseConv.hwx baseConv.info regNeTile
    let wCom ← encodeWord neCommonLayout #[
      ("kind", 0), ("shPref", 5), ("shMin", 5), ("shMax", 5), ("F", 3), ("wide3x3", 0)
    ]
    let wTile ← encodeWord neTileLayout #[("ocg", 1), ("fat", 1), ("WU", 0)]
    let hwx1 ← putU32LE baseConv.hwx fOffCom wCom
    let hwx2 ← putU32LE hwx1 fOffTile wTile
    let info ← validateHwx hwx2
    let _ ← verifyConvHwxTasks hwx2 info (requireConvTask := true)
    pure ()
  ) "shMax=5")

  results := results.push (checkErr "m2 neg verifier: shMin != shPref rejected" (do
    let wBad ← encodeWord neCommonLayout #[
      ("kind", 0), ("shPref", 0), ("shMin", 1), ("shMax", 4), ("F", 4), ("wide3x3", 0)
    ]
    tamperRegAndVerify baseConv regNeCommon wBad
  ) "shMin")

  results := results.push (checkErr "m2 neg verifier: WU + F + fat != logNE rejected" (do
    let wBad ← encodeWord neTileLayout #[("ocg", 0), ("fat", 0), ("WU", 1)]
    tamperRegAndVerify baseConv regNeTile wBad
  ) "WU(1) + F(4) + fat(0) != logNE(4)")

  -- 1x1 with wide3x3=1 or dwGroups
  results := results.push (checkErr "m2 neg verifier: 1x1 with wide3x3=1 rejected" (do
    let fOff ← findRegFileOffset baseConv.hwx baseConv.info regNeCommon
    let cur ← readU32LE baseConv.hwx fOff
    tamperRegAndVerify baseConv regNeCommon (cur ||| ((1 : UInt32) <<< 27))
  ) "wide3x3")

  results := results.push (checkErr "m2 neg verifier: 1x1 with dwGroups (0x0030) rejected" (do
    let wDw ← encodeWord dwGroupsLayout #[("perGroupIn", 1)]
    let plain := baseTd1.plainWrites.push { byteAddr := regDwGroups, value := wDw }
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes { baseTd1 with plainWrites := plain }
  ) "dwGroups")

  -- Invalid dwGroups (perGroupIn != 1, Cin != Cout, Cin == 1)
  results := results.push (checkErr "m2 neg verifier: dwGroups perGroupIn != 1 rejected" (do
    let wDwBad ← encodeWord dwGroupsLayout #[("perGroupIn", 2)]
    tamperReg3x3AndVerify base3Dw regDwGroups wDwBad
  ) "perGroupIn")

  results := results.push (checkErr "m2 neg verifier: dwGroups with Cin != Cout rejected" (do
    let shUnequal : ConvSpec := { kernelSize := 3, mode := .dense, h := 8, w := 16, cin := 8, cout := 16, stride := 1, pad := .same }
    let tU ← chooseConvTiling .h17 shUnequal
    let pkU ← packConvKernelChunks .h17 shUnequal tU (mkTaggedWeights3x3 16 8) none
    let tdU ← lowerConvTask .h17 shUnequal tU pkU false
    let inU ← makeFp16Surface .h17 "x" "input_x" .input 8 8 16
    let outU ← makeFp16Surface .h17 "y" "output_y" .output 16 8 16
    let wDw ← encodeWord dwGroupsLayout #[("perGroupIn", 1)]
    let plain := tdU.plainWrites.push { byteAddr := regDwGroups, value := wDw }
    emitTaskAndVerify .h17 inU outU pkU.bytes { tdU with plainWrites := plain }
  ) "Cin == Cout")

  results := results.push (checkErr "m2 neg verifier: dwGroups with Cin == Cout == 1 rejected" (do
    let wDw ← encodeWord dwGroupsLayout #[("perGroupIn", 1)]
    let plain := baseTd3Dw1.plainWrites.push { byteAddr := regDwGroups, value := wDw }
    emitTaskAndVerify .h17 baseIn3Dw1 baseOut3Dw1 basePk3Dw1.bytes { baseTd3Dw1 with plainWrites := plain }
  ) "Cin == Cout > 1")

  -- Stride-2 Win/Hin mismatch
  results := results.push (checkErr "m2 neg verifier: stride-2 Win mismatch rejected" (do
    tamperReg3x3AndVerify base3DenseS2 regWin 15
  ) "Win=")

  results := results.push (checkErr "m2 neg verifier: stride-2 Hin mismatch rejected" (do
    tamperReg3x3AndVerify base3DenseS2 regHin 15
  ) "Hin=")

  -- resCfg.direct mismatch vs isDirect, and resDep.direct mismatch vs resCfg.direct
  results := results.push (checkErr "m2 neg verifier: resCfg.direct mismatch vs target/tiling rejected" (do
    let wBad ← encodeWord resCfgLayout #[("mode", 2), ("direct", 0), ("P", 1)]
    tamperRegAndVerify baseConv regResCfg wBad
  ) "resCfg direct")

  results := results.push (checkErr "m2 neg verifier: resDep.direct mismatch vs resCfg.direct rejected" (do
    let wBad ← encodeWord resDepLayout #[("code", resDepCode .h17), ("task", 0), ("direct", 0)]
    tamperRegAndVerify baseConv regResDep wBad
  ) "resDep direct")

  -- scalarBias missing / spurious / combined with biasInKernel=1
  results := results.push (checkErr "m2 neg verifier: missing scalarBias for dense Cout=1 rejected" (do
    let plain := baseTd1C1.plainWrites.filter (fun w => w.byteAddr != regScalarBias)
    emitTaskAndVerify .h17 baseIn1C1 baseOut1C1 basePk1C1.bytes { baseTd1C1 with plainWrites := plain }
  ) "scalarBias")

  results := results.push (checkErr "m2 neg verifier: spurious scalarBias for Cout > 1 rejected" (do
    let plain := baseTd1.plainWrites.push { byteAddr := regScalarBias, value := 0x3c00 }
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes { baseTd1 with plainWrites := plain }
  ) "scalarBias")

  results := results.push (checkErr "m2 neg verifier: dense Cout=1 with biasInKernel=1 rejected" (do
    let wOpBad ← encodeWord neOpLayout #[("biasInKernel", 1), ("relu", 0), ("lut", 0)]
    let plain := baseTd1C1.plainWrites.map fun w =>
      if w.byteAddr == regNeOp then { w with value := wOpBad } else w
    emitTaskAndVerify .h17 baseIn1C1 baseOut1C1 basePk1C1.bytes { baseTd1C1 with plainWrites := plain }
  ) "biasInKernel=0")

  -- Missing Mach-O kernel relocation
  results := results.push (checkErr "m2 neg verifier: missing Mach-O kernel relocation rejected" (do
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes baseTd1 (emitReloc := false)
  ) "relocation")

  -- DMA stride / alignment / span violations
  results := results.push (checkErr "m2 neg verifier: unaligned input DMA row stride rejected" (do
    tamperRegAndVerify baseConv regSrc1DmaRowStride 32
  ) "64-byte aligned")

  results := results.push (checkErr "m2 neg verifier: input DMA plane stride < H*rowStride rejected" (do
    tamperRegAndVerify baseConv regSrc1DmaPlaneStride 64
  ) "plane stride")

  results := results.push (checkErr "m2 neg verifier: output DMA plane stride exceeding surface rejected" (do
    tamperRegAndVerify baseConv regResDmaPlaneStride 0x10000
  ) "output DMA range")

  -- Duplicate register write
  results := results.push (checkErr "m2 neg verifier: duplicate register write rejected" (do
    let dupRec := RegRecord.sequential (regWin / 4).toUInt16 none #[16]
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes baseTd1 (extraRecs := #[dupRec])
  ) "duplicate write")

  -- B3: chunkSlot on20==0, task!=0, missing active chunk slot, spurious extra chunk slot,
  --     chunkOffset % 64 != 0, under-sized stride-1 chunkSize, truncated sparse chunk
  results := results.push (checkErr "m2 neg verifier: chunkSlot on20==0 rejected" (do
    let wBad ← encodeWord chunkSlotLayout #[("enable", 1), ("on20", 0), ("task", 0)]
    tamperRegAndVerify baseConv (regChunkSlot 0) wBad
  ) "on20=1")

  results := results.push (checkErr "m2 neg verifier: chunkSlot task != taskId rejected" (do
    let wBad ← encodeWord chunkSlotLayout #[("enable", 1), ("on20", 1), ("task", 1)]
    tamperRegAndVerify baseConv (regChunkSlot 0) wBad
  ) "taskId")

  results := results.push (checkErr "m2 neg verifier: missing required active chunk slot rejected" (do
    let plain := baseTd1.plainWrites.filter fun w =>
      w.byteAddr != regChunkSlot 15 && w.byteAddr != regChunkOffset 15 && w.byteAddr != regChunkSize 15
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes { baseTd1 with plainWrites := plain }
  ) "missing required active chunk slot")

  results := results.push (checkErr "m2 neg verifier: spurious extra chunk slot rejected" (do
    let wSlot ← encodeWord chunkSlotLayout #[("enable", 1), ("on20", 1), ("task", 0)]
    let plain := baseTd1C1.plainWrites ++ #[
      { byteAddr := regChunkSlot 1, value := wSlot },
      { byteAddr := regChunkOffset 1, value := 0 },
      { byteAddr := regChunkSize 1, value := 64 }
    ]
    emitTaskAndVerify .h17 baseIn1C1 baseOut1C1 basePk1C1.bytes { baseTd1C1 with plainWrites := plain }
  ) "spurious chunk slot")

  results := results.push (checkErr "m2 neg verifier: misaligned chunkOffset % 64 != 0 rejected" (do
    tamperRegAndVerify baseConv (regChunkOffset 0) 32
  ) "64-byte aligned")

  results := results.push (checkErr "m2 neg verifier: under-sized stride-1 chunkSize rejected" (do
    -- base3DenseS1 has Cin=16, 1 output/group, 9 taps + 1 bias = 290B -> min 320B; 64B is too small
    tamperReg3x3AndVerify base3DenseS1 (regChunkSize 0) 64
  ) "minimum required")

  results := results.push (checkErr "m2 neg verifier: truncated stride-2 sparse chunk u24 length rejected" (do
    -- Overwrite first block's u24 length in __KERN_0 (immediately after 2B bias of chunk 0) to exceed chunkSize
    let some kSec := base3DenseS2.info.kernelSection
      | throw "missing __KERN_0"
    let u24FileOff := kSec.offset.toNat + 2
    let hwx1 ← putU8 base3DenseS2.hwx u24FileOff 0x00
    let hwx2 ← putU8 hwx1 (u24FileOff + 1) 0x80
    let hwx3 ← putU8 hwx2 (u24FileOff + 2) 0x00
    let info ← validateHwx hwx3
    let _ ← verifyConvHwxTasks hwx3 info (requireConvTask := true)
    pure ()
  ) "exceeds chunk end")

  -- B4: tampered task header words 0..7
  results := results.push (checkErr "m2 neg verifier: header[0] reserved bits [31:27] != 0 rejected" (do
    let textBytes ← readSlice baseConv.hwx baseConv.info.textSection.offset.toNat baseConv.info.textSection.size.toNat
    let tasks ← decodeTaskStream baseConv.info.target textBytes
    let some t0 := tasks[0]? | throw "no task 0"
    let h0FileOff := baseConv.info.textSection.offset.toNat + t0.startOffset
    let h0 ← readU32LE baseConv.hwx h0FileOff
    let badHwx ← putU32LE baseConv.hwx h0FileOff (h0 ||| ((1 : UInt32) <<< 27))
    let info ← validateHwx badHwx
    let _ ← verifyConvHwxTasks badHwx info (requireConvTask := true)
    pure ()
  ) "reserved")

  results := results.push (checkErr "m2 neg verifier: header[1] != 0 rejected" (do
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes baseTd1 (headerPatch? := some (1, 1))
  ) "header[1]")

  results := results.push (checkErr "m2 neg verifier: header[2] tampered rejected" (do
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes baseTd1 (headerPatch? := some (2, 0x04000069))
  ) "header[2]")

  results := results.push (checkErr "m2 neg verifier: header[4] tampered rejected" (do
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes baseTd1 (headerPatch? := some (4, 0x00fff869))
  ) "header[4]")

  results := results.push (checkErr "m2 neg verifier: header[5] != 0 rejected" (do
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes baseTd1 (headerPatch? := some (5, 1))
  ) "header[5..7]")

  -- B4: tampered src1Fmt / resFmt / src1DmaCfg / neScale / out0..out5
  results := results.push (checkErr "m2 neg verifier: tampered src1DmaCfg rejected" (do
    tamperRegAndVerify baseConv regSrc1DmaCfg 0x000000cf
  ) "src1DmaCfg")

  results := results.push (checkErr "m2 neg verifier: tampered src1Fmt rejected" (do
    tamperRegAndVerify baseConv regSrc1Fmt 0x01002030
  ) "src1Fmt")

  results := results.push (checkErr "m2 neg verifier: tampered resFmt rejected" (do
    tamperRegAndVerify baseConv regResFmt 0x01302030
  ) "resFmt")

  results := results.push (checkErr "m2 neg verifier: tampered neScale rejected" (do
    tamperRegAndVerify baseConv regNeScale 0x00004000
  ) "neScale")

  results := results.push (checkErr "m2 neg verifier: tampered out0 rejected" (do
    tamperRegAndVerify baseConv regOut0 0x00000013
  ) "out0")

  results := results.push (checkErr "m2 neg verifier: tampered out2 form rejected" (do
    tamperRegAndVerify baseConv regOut2 0x000000e1
  ) "out2")

  results := results.push (checkErr "m2 neg verifier: tampered out5 rejected" (do
    tamperRegAndVerify baseConv regOut5 0x20010700
  ) "out5")

  -- B4: missing BAR-4 on src1Dep / resDep & bad code/task
  results := results.push (checkErr "m2 neg verifier: src1Dep without BAR-4 rejected" (do
    let barWrites := baseTd1.barWrites.filter (fun br => br.baseByteAddr != regSrc1Dep)
    let wSrcDep ← encodeWord src1DepLayout #[("code", 0x21), ("task", 0)]
    let plain := baseTd1.plainWrites.push { byteAddr := regSrc1Dep, value := wSrcDep }
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes { baseTd1 with plainWrites := plain, barWrites := barWrites }
  ) "BAR slot 4")

  results := results.push (checkErr "m2 neg verifier: resDep without BAR-4 rejected" (do
    let barWrites := baseTd1.barWrites.filter (fun br => br.baseByteAddr != regResDep)
    let wResDep ← encodeWord resDepLayout #[("code", resDepCode .h17), ("task", 0), ("direct", 1)]
    let plain := baseTd1.plainWrites.push { byteAddr := regResDep, value := wResDep }
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes { baseTd1 with plainWrites := plain, barWrites := barWrites }
  ) "BAR slot 4")

  results := results.push (checkErr "m2 neg verifier: src1Dep bad code rejected" (do
    tamperRegAndVerify baseConv regSrc1Dep 0x00000022
  ) "src1Dep code")

  -- B4: direct-mode staged register 0x4150 present
  results := results.push (checkErr "m2 neg verifier: direct-mode 0x4150 (resChanStride) present rejected" (do
    let plain := baseTd1.plainWrites.push { byteAddr := regResChanStride, value := 32 }
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes { baseTd1 with plainWrites := plain }
  ) "direct mode must omit")

  -- N5: inSurf.n != 1 and unaligned L2 bases
  results := results.push (checkErr "m2 neg verifier: inSurf.n != 1 rejected" (do
    let badSpan := 2 * baseIn1.batchStride
    let badAlloc := (alignUp badSpan.toNat pageSize).toUInt64
    let badIn : SurfaceSpec := {
      baseIn1 with
      n := 2
      sectionSize := some badSpan
      allocSize := some badAlloc
    }
    emitTaskAndVerify .h17 badIn baseOut1 basePk1.bytes baseTd1
  ) "single-batch")

  results := results.push (checkErr "m2 neg verifier: unaligned L2 src1Base % 64 != 0 rejected" (do
    let fOffSrc ← findRegFileOffset baseConv.hwx baseConv.info regSrc1Base
    let fOffRes ← findRegFileOffset baseConv.hwx baseConv.info regResBase
    let curRes ← readU32LE baseConv.hwx fOffRes
    let hwx1 ← putU32LE baseConv.hwx fOffSrc 32
    let hwx2 ← putU32LE hwx1 fOffRes (curRes + 64)
    let info ← validateHwx hwx2
    let _ ← verifyConvHwxTasks hwx2 info (requireConvTask := true)
    pure ()
  ) "L2 src1Base")

  results := results.push (checkErr "m2 neg verifier: unaligned L2 resBase % 64 != 0 rejected" (do
    let fOff ← findRegFileOffset baseConv.hwx baseConv.info regResBase
    let cur ← readU32LE baseConv.hwx fOff
    tamperRegAndVerify baseConv regResBase (cur + 32)
  ) "64-byte aligned")

  -- =========================================================================
  -- Milestone 3 Tests: Float16, PE, 128B LUTs, Fused Conv+LUT, & M3 Verifier
  -- =========================================================================

  -- 1. Float16 / Float32 / math helpers
  results := results.push (checkOk "m3: Float16 exhaustive 65536 bit-pattern round-trip & f16ToF32Bits" (do
    for i in [:65536] do
      let h := i.toUInt16
      let f := f16ToFloat h
      let f32 := f16ToF32Bits h
      if f16IsNaN h then
        if !f.isNaN then throw s!"NaN 0x{Nat.toDigits 16 h.toNat} did not convert to NaN"
        let rt := floatToF16 f
        if rt != f16QNaN then throw s!"NaN round-trip gave {toHex rt.toNat}"
        if (f32 &&& 0x7f800000) != 0x7f800000 || (f32 &&& 0x007fffff) == 0 then
          throw s!"f16ToF32Bits NaN failed for {toHex h.toNat}: {toHex f32.toNat}"
      else
        let rt := floatToF16 f
        if rt != h then
          throw s!"round-trip mismatch for {toHex h.toNat}: got {toHex rt.toNat}"
        let cz := floatToF16CanonicalZero f
        if h == f16NegZero then
          if cz != f16PosZero then throw "canonicalZero failed on -0.0"
        else if cz != h then
          throw s!"canonicalZero mismatch for {toHex h.toNat}"
    -- Spot check f16ToF32Bits exact IEEE-754 binary32 values
    if f16ToF32Bits 0x3c00 != 0x3f800000 then throw "f16ToF32Bits(1.0) mismatch"
    if f16ToF32Bits 0xbc00 != 0xbf800000 then throw "f16ToF32Bits(-1.0) mismatch"
    if f16ToF32Bits 0x3e00 != 0x3fc00000 then throw "f16ToF32Bits(1.5) mismatch"
    if f16ToF32Bits 0x0001 != 0x33800000 then throw "f16ToF32Bits(2^-24) mismatch"
    if f16ToF32Bits 0x8000 != 0x80000000 then throw "f16ToF32Bits(-0.0) mismatch"
    if f16ToF32Bits 0x7c00 != 0x7f800000 then throw "f16ToF32Bits(+inf) mismatch"
    if f16ToF32Bits 0xfc00 != 0xff800000 then throw "f16ToF32Bits(-inf) mismatch"
    -- prevF16 checks
    if prevF16 0x0000 != 0x8001 then throw "prevF16(+0) != -2^-24"
    if prevF16 0x8000 != 0x8001 then throw "prevF16(-0) != -2^-24"
    if prevF16 0x0001 != 0x0000 then throw "prevF16(+2^-24) != +0"
    if prevF16 0x3c00 != 0x3bff then throw "prevF16(1.0) != 0x3bff"
    if prevF16 0x8001 != 0x8002 then throw "prevF16(-2^-24) != -2*2^-24"
    if prevF16 0xfbff != 0xfc00 then throw "prevF16(-65504) != -inf"
    if prevF16 0xfc00 != 0xfc00 then throw "prevF16(-inf) != -inf"
    -- erfApprox & expm1 accuracy checks
    if Float.abs (erfApprox 0.0) > 1e-12 then throw "erfApprox(0) != 0"
    if Float.abs (erfApprox 1.0 - 0.8427007929) > 2e-6 then throw s!"erfApprox(1.0)={erfApprox 1.0}"
    if Float.abs (erfApprox (-1.0) + 0.8427007929) > 2e-6 then throw "erfApprox(-1.0) oddness"
    if Float.abs (erfApprox 2.0 - 0.9953222650) > 2e-6 then throw s!"erfApprox(2.0)={erfApprox 2.0}"
    if Float.abs (expm1 1e-6 - 1.0000005e-6) > 1e-12 then throw "expm1(1e-6) inaccurate"
    if Float.abs (expm1 (-1.0) - (Float.exp (-1.0) - 1.0)) > 1e-12 then throw "expm1(-1) inaccurate"
    -- rna16 rounding checks (ties away, -0 -> +0, NaN -> +0, overflow at 65520)
    if rna16 (1.0 + ldexp 1.0 (-11)) != 0x3c01 then throw "rna16 ties-away failed"
    if rna16 (-0.0) != 0x0000 then throw "rna16(-0.0) != +0.0"
    if rna16 (0.0 / 0.0) != 0x0000 then throw "rna16(NaN) != +0.0"
    if rna16 65520.0 != 0x7c00 then throw "rna16(65520) != +inf"
    if rna16 (-65520.0) != 0xfc00 then throw "rna16(-65520) != -inf"
    if rna16 65519.0 != 0x7bff then throw "rna16(65519) != 65504"
    -- parseFloat?
    match parseFloat? "-1.25e-2" with
    | some v => if Float.abs (v + 0.0125) > 1e-12 then throw "parseFloat? -1.25e-2 mismatch"
    | none => throw "parseFloat? failed on -1.25e-2"
    pure ()
  ))

  -- 2. PE bitfields, tiling, 1-source & 2-source & broadcasts & H16 staged
  results := results.push (checkOk "m3: PE 1-source ops (abs, square, linear, 3-step chain, scalars) on H16 & H17" (do
    let d : TensorDims := { c := 8, h := 16, w := 24 }
    let divOp ← PeOp.scalarDiv (floatToF16 4.0)
    let ops : Array PeOp := #[
      .abs,
      .square,
      PeOp.linearActivation (floatToF16 1.5) (floatToF16 (-0.25)),
      .chain (some (floatToF16 1.5)) (some (floatToF16 (-0.25))) (some (floatToF16 0.5)),
      PeOp.scalarAdd (floatToF16 2.0),
      PeOp.scalarSub (floatToF16 3.0),
      PeOp.scalarRSub (floatToF16 5.0),
      PeOp.scalarMul (floatToF16 (-0.5)),
      divOp
    ]
    for tgt in #[Target.h16, Target.h17] do
      for op in ops do
        let c ← compilePe tgt op d
        let s ← verifyHwxTasksSummary c.hwx c.info (requireKnownTask := true)
        if s.peOp != 1 || s.total != 1 then
          throw s!"expected 1 verified PE task for {tgt.name}/{op.opName}, got {repr s}"
    pure ()
  ))

  results := results.push (checkErr "m3 neg: PE scalarDiv rejects zero divisor" (do
    PeOp.scalarDiv 0x0000
  ) "non-zero")

  results := results.push (checkErr "m3 neg: PE empty chain rejected" (do
    compilePe .h17 (.chain none none none) { c := 4, h := 4, w := 4 }
  ) "at least one")

  results := results.push (checkOk "m3: PE 2-source binary ops + broadcasts + whole2 + large multi-tile on H16 & H17" (do
    let dFull : TensorDims := { c := 4, h := 8, w := 16 }
    let binOps : Array (PeBinOp × Bool) := #[
      (.add, false), (.sub, false), (.sub, true), (.mul, false), (.max, false), (.min, false)
    ]
    for tgt in #[Target.h16, Target.h17] do
      for (bop, rev) in binOps do
        let c ← compilePe tgt (.binary bop dFull dFull rev) dFull
        let s ← verifyHwxTasksSummary c.hwx c.info (requireKnownTask := true)
        if s.peOp != 1 then throw s!"expected 1 PE task for {bop.name}"
    -- Broadcast shapes: [C,1,1], [1,H,W], [1,1,W] with th < H (whole2!), [C,H,1], [1,1,1]
    let bcastPairs : Array (TensorDims × TensorDims × Bool) := #[
      ({ c := 8, h := 8, w := 16 }, { c := 8, h := 1, w := 1 }, false),
      ({ c := 8, h := 8, w := 16 }, { c := 1, h := 8, w := 16 }, false),
      ({ c := 8, h := 8, w := 16 }, { c := 8, h := 8, w := 1 }, false),
      ({ c := 8, h := 8, w := 16 }, { c := 1, h := 1, w := 1 }, false),
      ({ c := 1, h := 8, w := 16 }, { c := 8, h := 1, w := 1 }, true),
      -- Large multi-tile triggering th < H with [1,1,W] row broadcast (whole2 = true)
      ({ c := 256, h := 64, w := 128 }, { c := 1, h := 1, w := 128 }, false)
    ]
    for (s1, s2, rev) in bcastPairs do
      for tgt in #[Target.h16, Target.h17] do
        let c ← compilePe tgt (.binary .add s1 s2 rev) s1
        let s ← verifyHwxTasksSummary c.hwx c.info (requireKnownTask := true)
        if s.peOp != 1 then throw "expected 1 PE task for broadcast"
    -- Also explicit whole2 forced tiling check: [1,4,8,16] + [1,1,1,16] with th=4 < H=8
    let cWhole2 ← compilePe .h17 (.binary .add { c := 4, h := 8, w := 16 } { c := 1, h := 1, w := 16 } false)
      { c := 4, h := 8, w := 16 } (tilingOverride := some { k := 4, tileHeight := 4 })
    let sW2 ← verifyHwxTasksSummary cWhole2.hwx cWhole2.info (requireKnownTask := true)
    if sW2.peOp != 1 then throw "whole2 explicit tiling failed"
    pure ()
  ))

  results := results.push (checkOk "m3: PE H16 staged-result edge case (k=5, W=56)" (do
    let d : TensorDims := { c := 4, h := 8, w := 56 }
    if peIsDirect .h16 56 5 then throw "expected peIsDirect(.h16, 56, 5) == false"
    if !peIsDirect .h17 56 5 then throw "expected peIsDirect(.h17, 56, 5) == true"
    let cH16 ← compilePe .h16 .abs d (tilingOverride := some { k := 5, tileHeight := 8 })
    if cH16.direct then throw "expected H16 k=5 W=56 to use staged result mode"
    let s ← verifyHwxTasksSummary cH16.hwx cH16.info (requireKnownTask := true)
    if s.peOp != 1 then throw "H16 staged PE verification failed"
    -- Auto chooser on H16 should prefer direct mode (e.g. k=6)
    let cAuto ← compilePe .h16 .abs d
    if !cAuto.direct then throw "expected H16 auto PE chooser to prefer direct mode"
    pure ()
  ))

  -- 3. LUT table generation, invariants, parameter validation, & mathematical accuracy
  results := results.push (checkOk "m3: all 23 LUT specs generate valid 128B tables (zero reserved, +0 canon, accurate eval)" (do
    let allSpecs : Array LutSpec := #[
      .relu6,
      .clip (-1.5) 2.5,
      .leakyRelu 0.25,
      .abs,
      .sign,
      .threshold 0.5,
      .thresholdedRelu 1.0,
      .hardSigmoid 0.2 0.5,
      .hardSigmoid 0.5 0.25,
      .clampMax (-0.5),
      .clampMin 2.0,
      .elu 1.0,
      .elu 0.5,
      .tanh,
      .scaledTanh 2.0 0.5,
      .scaledTanh (-2.0) 0.5,
      .erf,
      .sigmoid,
      .gelu,
      .silu,
      .exp2,
      .exp,
      .sqrt,
      .rsqrt 0.0,
      .rsqrt 0.0625,
      .inverse,
      .log2
    ]
    for spec in allSpecs do
      let lp ← buildLutTable spec
      if lp.words.size != 64 || lp.bytes.size != 128 then
        throw s!"{spec.name}: expected 64 words / 128 bytes"
      for wi in [43 : 64] do
        if lp.words[wi]! != 0 then
          throw s!"{spec.name}: reserved word {wi} non-zero"
      for wi in [:41] do
        if lp.words[wi]! == f16NegZero then
          throw s!"{spec.name}: table word {wi} contains -0.0 (0x8000)"

    -- Verify mathematical accuracy of evalLutTable against analytical formulas
    let checkEval (label : String) (spec : LutSpec) (xs : Array Float) (exactFn : Float → Float) (tol : Float) : Except String Unit := do
      let lp ← buildLutTable spec
      for x in xs do
        let xh := floatToF16 x
        let xf := f16ToFloat xh
        let gotBits := evalLutTable lp.words lp.preScaleBits lp.inputShiftBits xh
        let got := f16ToFloat gotBits
        let want := exactFn xf
        let err := Float.abs (got - want)
        if err > tol then
          throw s!"{label} at x={xf}: got={got}, want={want}, err={err} > tol={tol}"

    let grid : Array Float := #[-6.0, -3.5, -2.0, -1.25, -0.5, -0.125, 0.0, 0.125, 0.5, 1.25, 2.0, 3.5, 7.0]
    let posGrid : Array Float := #[0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 9.0, 16.0]
    let invSqrt2 := 0.707106781186547524400844362104849039

    checkEval "relu6" .relu6 grid (fun x => max 0.0 (min 6.0 x)) 0.0
    checkEval "clip" (.clip (-1.5) 2.5) grid (fun x => max (-1.5) (min 2.5 x)) 0.0
    checkEval "leaky_relu" (.leakyRelu 0.25) grid (fun x => if x < 0.0 then 0.25 * x else x) 1e-3
    checkEval "abs" .abs grid Float.abs 0.0
    checkEval "sign" .sign grid (fun x => if x < 0.0 then -1.0 else if x > 0.0 then 1.0 else 0.0) 0.0
    checkEval "threshold" (.threshold 0.5) grid (fun x => max x 0.5) 0.0
    checkEval "thresholded_relu" (.thresholdedRelu 1.0) #[-2.0, 0.0, 0.5, 0.9, 1.25, 3.0] (fun x => if x > 1.0 then x else 0.0) 0.0
    checkEval "hard_sigmoid_std" (.hardSigmoid 0.2 0.5) grid (fun x => max 0.0 (min 1.0 (0.2 * x + 0.5))) 2e-3
    checkEval "hard_sigmoid_shift" (.hardSigmoid 0.5 0.25) grid (fun x => max 0.0 (min 1.0 (0.5 * x + 0.25))) 2e-3
    checkEval "clamp_max" (.clampMax (-0.5)) grid (fun x => max x (-0.5)) 0.0
    checkEval "clamp_min" (.clampMin 2.0) grid (fun x => min x 2.0) 0.0
    checkEval "elu" (.elu 1.0) grid (fun x => if x >= 0.0 then x else expm1 x) 1e-2
    checkEval "tanh" .tanh grid Float.tanh 3e-3
    checkEval "scaled_tanh_pos" (.scaledTanh 2.0 0.5) grid (fun x => 2.0 * Float.tanh (0.5 * x)) 5e-3
    checkEval "scaled_tanh_neg" (.scaledTanh (-2.0) 0.5) grid (fun x => (-2.0) * Float.tanh (0.5 * x)) 5e-3
    checkEval "erf" .erf grid erfApprox 4e-3
    checkEval "sigmoid" .sigmoid grid (fun x => 1.0 / (1.0 + Float.exp (-x))) 5e-3
    checkEval "gelu" .gelu grid (fun x => 0.5 * x * (1.0 + erfApprox (x * invSqrt2))) 1e-2
    checkEval "silu" .silu grid (fun x => x / (1.0 + Float.exp (-x))) 2e-2
    checkEval "exp2" .exp2 #[-4.0, -2.5, -1.0, -0.25, 0.0, 0.5, 1.25, 3.0] exp2Float 2e-2
    checkEval "exp" .exp #[-4.0, -2.0, -0.5, 0.0, 0.5, 1.5, 2.5] Float.exp 0.1
    checkEval "sqrt" .sqrt posGrid Float.sqrt 1e-2
    checkEval "rsqrt0" (.rsqrt 0.0) posGrid (fun x => 1.0 / Float.sqrt x) 1e-2
    checkEval "rsqrt_eps" (.rsqrt 0.25) #[0.0, 0.25, 1.0, 3.75] (fun x => 1.0 / Float.sqrt (x + 0.25)) 1e-2
    checkEval "inverse" .inverse #[-4.0, -2.0, -0.5, 0.5, 2.0, 4.0] (fun x => 1.0 / x) 1e-2
    checkEval "log2" .log2 posGrid log2Float 1e-2
    pure ()
  ))

  results := results.push (checkErr "m3 neg: LUT clip rejects lo > hi" (do
    buildLutTable (.clip 2.0 1.0)
  ) "lo")

  results := results.push (checkErr "m3 neg: LUT hard_sigmoid rejects alpha <= 0" (do
    buildLutTable (.hardSigmoid 0.0 0.5)
  ) "alpha")

  results := results.push (checkErr "m3 neg: LUT rsqrt rejects eps < 0" (do
    buildLutTable (.rsqrt (-0.1))
  ) "eps")

  -- 4. Standalone compileNeAct across H16 & H17
  results := results.push (checkOk "m3: standalone compileNeAct (relu, copy, LUTs, rsqrt+eps, exp, hard_sigmoid) on H16 & H17" (do
    let acts : Array NeActSpec := #[
      .relu,
      .copy,
      .lut .relu6,
      .lut .tanh,
      .lut .sigmoid,
      .lut .gelu,
      .lut .silu,
      .lut .exp,
      .lut (.rsqrt 0.0625),
      .lut (.hardSigmoid 0.5 0.25),
      .lut .sqrt,
      .lut .inverse,
      .lut .log2
    ]
    for tgt in #[Target.h16, Target.h17] do
      for act in acts do
        let c ← compileNeAct tgt act 8 16 24
        let s ← verifyHwxTasksSummary c.hwx c.info (requireKnownTask := true)
        if s.neAct != 1 || s.total != 1 then
          throw s!"expected 1 verified NE act task for {tgt.name}/{act.name}"
    -- Large C test forcing shPref >= 2 to leave 512B direct-mode output ring headroom
    let cBig ← compileNeAct .h17 (.lut .tanh) 16384 4 128
    if cBig.tiling.shPref < 2 then
      throw s!"expected C=16384 W=128 to shrink shPref >= 2 for direct-ring headroom, got {cBig.tiling.shPref}"
    if cBig.l2UseBytes > l2CapacityBytes then
      throw s!"expected l2UseBytes <= 2 MiB, got {cBig.l2UseBytes}"
    let sBig ← verifyHwxTasksSummary cBig.hwx cBig.info (requireKnownTask := true)
    if sBig.neAct != 1 then throw "large C standalone NE act failed"
    pure ()
  ))

  -- 5. Fused conv + LUT activation (1x1 dense, 3x3 dense s1/s2, 3x3 dw s1/s2, Cout=1 & Cout>1, 64KiB boundary)
  results := results.push (checkOk "m3: fused 1x1 & 3x3 (dense, depthwise, stride-2 sparse) conv + LUT on H16 & H17" (do
    let lutSpecs : Array LutSpec := #[
      .tanh, .sigmoid, .gelu, .silu, .exp, .elu 1.0, .relu6,
      .hardSigmoid 0.5 0.25, .rsqrt 0.0625
    ]
    -- 1x1 dense + LUTs (with and without bias)
    for spec in lutSpecs do
      let sh1 : Conv1x1Shape := { h := 8, w := 16, cin := 16, cout := 16 }
      let c1 ← compileConv1x1 .h17 sh1 (mkTaggedWeights 16 16) (some (mkTaggedBias 16)) false none (.lut spec)
      let s1 ← verifyHwxTasksSummary c1.hwx c1.info (requireKnownTask := true)
      if s1.neConv != 1 then throw s!"fused 1x1 {spec.name} failed"
      -- No user bias with shifted LUT (rsqrt/hard_sigmoid) synthesizes kernel bias cleanly
      let c1NoB ← compileConv1x1 .h16 sh1 (mkTaggedWeights 16 16) none false none (.lut spec)
      let s1NoB ← verifyHwxTasksSummary c1NoB.hwx c1NoB.info (requireKnownTask := true)
      if s1NoB.neConv != 1 then throw s!"fused 1x1 h16 no-bias {spec.name} failed"

    -- 3x3 dense stride-1 + LUT
    let sh3D1 : Conv3x3Shape := { mode := .dense, h := 8, w := 16, cin := 8, cout := 16, stride := 1, pad := .same }
    let c3D1 ← compileConv3x3 .h17 sh3D1 (mkTaggedWeights3x3 16 8) (some (mkTaggedBias 16)) false none (.lut .gelu)
    if (← verifyHwxTasksSummary c3D1.hwx c3D1.info).neConv != 1 then throw "fused 3x3 dense s1 failed"

    -- 3x3 depthwise stride-1 + LUT
    let sh3Dw1 : Conv3x3Shape := { mode := .depthwise, h := 8, w := 16, cin := 16, cout := 16, stride := 1, pad := .same }
    let c3Dw1 ← compileConv3x3 .h17 sh3Dw1 (mkTaggedWeights3x3 16 1) (some (mkTaggedBias 16)) false none (.lut .silu)
    if (← verifyHwxTasksSummary c3Dw1.hwx c3Dw1.info).neConv != 1 then throw "fused 3x3 dw s1 failed"

    -- 3x3 stride-2 sparse + LUT (Cout > 1 with kernel bias, Cout == 1 with scalar bias, and shift without bias)
    for tgt in #[Target.h16, Target.h17] do
      let sh3S2 : Conv3x3Shape := { mode := .dense, h := 12, w := 14, cin := 8, cout := 16, stride := 2, pad := .same }
      let c3S2 ← compileConv3x3 tgt sh3S2 (mkTaggedWeights3x3 16 8) (some (mkTaggedBias 16)) false none (.lut .tanh)
      if (← verifyHwxTasksSummary c3S2.hwx c3S2.info).neConv != 1 then throw "fused 3x3 s2 Cout=16 failed"

      let sh3S2C1 : Conv3x3Shape := { mode := .dense, h := 12, w := 14, cin := 8, cout := 1, stride := 2, pad := .valid }
      let c3S2C1 ← compileConv3x3 tgt sh3S2C1 (mkTaggedWeights3x3 1 8) none false none (.lut (.rsqrt 0.0625))
      if (← verifyHwxTasksSummary c3S2C1.hwx c3S2C1.info).neConv != 1 then throw "fused 3x3 s2 Cout=1 rsqrt failed"

      let sh3DwS2 : Conv3x3Shape := { mode := .depthwise, h := 12, w := 14, cin := 16, cout := 16, stride := 2, pad := .same }
      let c3DwS2 ← compileConv3x3 tgt sh3DwS2 (mkTaggedWeights3x3 16 1) none false none (.lut (.hardSigmoid 0.5 0.25))
      if (← verifyHwxTasksSummary c3DwS2.hwx c3DwS2.info).neConv != 1 then throw "fused 3x3 dw s2 hard_sigmoid failed"
    pure ()
  ))

  results := results.push (checkOk "m3: 64 KiB chunk boundary with +128B fused LUT forces re-tiling cleanly" (do
    -- 1x1 dense Cin=2048, Cout=16:
    -- Without LUT: F=0 puts all 16 outs in 1 chunk = 2 * 16 * 2048 = 65536B (fits 64 KiB).
    -- With LUT: F=0 chunk would be 65536 + 128 = 65664B > 65536B -> must reject F=0 and pick F >= 1!
    let sh : ConvSpec := { kernelSize := 1, h := 4, w := 8, cin := 2048, cout := 16 }
    let tF0 : Tiling := { logNE := 0, f := 0, fat := 0, ocg := 4, shPref := 0, shMax := 4, tileHeight := 4 }
    validateConvSafety .h17 sh tF0 0 false
    match validateConvSafety .h17 sh tF0 128 false with
    | .ok () => throw "expected F=0 65536B + 128B LUT to exceed 64 KiB"
    | .error _ => pure ()
    let tChosen ← chooseConvTiling .h17 sh 128 false
    if tChosen.f == 0 then throw "expected chooseConvTiling with +128B LUT to choose F >= 1"
    let c ← compileConv .h17 sh (zeroBytes (2 * 16 * 2048)) none false none (.lut .tanh)
    if (← verifyHwxTasksSummary c.hwx c.info).neConv != 1 then throw "re-tiled fused conv verification failed"
    pure ()
  ))

  results := results.push (checkErr "m3 neg: 64 KiB chunk cap rejects impossible Cout=1 Cin=32768 + fused LUT" (do
    let sh : ConvSpec := { kernelSize := 1, h := 2, w := 8, cin := 32768, cout := 1 }
    chooseConvTiling .h17 sh 128 false
  ) "no legal")

  results := results.push (checkErr "m3 neg: compileConv rejects relu=true combined with LUT activation" (do
    let sh : ConvSpec := { kernelSize := 1, h := 4, w := 8, cin := 8, cout := 8 }
    compileConv .h17 sh (mkTaggedWeights 8 8) none true none (.lut .tanh)
  ) "cannot pass both")

  -- 6. Tampered-task negative verifier tests for M3 (PE, NE Act, Fused Conv)
  let basePe1 ← unwrap "basePe1" (compilePe .h17 .abs { c := 8, h := 8, w := 16 })
  let basePe2 ← unwrap "basePe2" (compilePe .h17 (.binary .add { c := 4, h := 8, w := 16 } { c := 4, h := 1, w := 1 } false) { c := 4, h := 8, w := 16 })
  let baseNeLut ← unwrap "baseNeLut" (compileNeAct .h17 (.lut .tanh) 8 8 16)
  let baseFusedS1 ← unwrap "baseFusedS1" (compileConv1x1 .h17 { h := 8, w := 16, cin := 16, cout := 16 } (mkTaggedWeights 16 16) (some (mkTaggedBias 16)) false none (.lut .tanh))
  let baseFusedS2 ← unwrap "baseFusedS2" (compileConv3x3 .h17 { mode := .dense, h := 12, w := 14, cin := 8, cout := 16, stride := 2, pad := .same } (mkTaggedWeights3x3 16 8) (some (mkTaggedBias 16)) false none (.lut .tanh))

  let tamperGenericAndVerify (hwx : ByteArray) (info : HwxInfo) (regAddr : Nat) (newVal : UInt32) : Except String Unit := do
    let fOff ← findRegFileOffset hwx info regAddr
    let badHwx ← putU32LE hwx fOff newVal
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasks badHwx badInfo (requireKnownTask := true)
    pure ()

  -- PE negative tests
  results := results.push (checkErr "m3 neg verifier (PE): ene != 0 rejected" (do
    let textBytes ← readSlice basePe1.hwx basePe1.info.textSection.offset.toNat basePe1.info.textSection.size.toNat
    let tasks ← decodeTaskStream basePe1.info.target textBytes
    let some t0 := tasks[0]? | throw "no task 0"
    let h8FileOff := basePe1.info.textSection.offset.toNat + t0.startOffset + 32
    let badW8 ← encodeWord taskHeader8Layout #[("ene", 1)]
    let badHwx ← putU32LE basePe1.hwx h8FileOff badW8
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasks badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "ene=0")

  results := results.push (checkErr "m3 neg verifier (PE): anti-hang src1ChanStride < 2*min(W,T) rejected" (do
    -- basePe1 has C=8 > 1, W=16, T >= 16 -> min anti-hang is 32; tamper to 16
    tamperGenericAndVerify basePe1.hwx basePe1.info regSrc1ChanStride 16
  ) "anti-hang")

  results := results.push (checkErr "m3 neg verifier (PE): wrong peBcast bits rejected" (do
    tamperGenericAndVerify basePe2.hwx basePe2.info regPeBcast 0
  ) "peBcast")

  results := results.push (checkErr "m3 neg verifier (PE): peTile m mismatch rejected" (do
    let curM := peTileM false basePe1.tiling.k basePe1.tiling.tileHeight
    let badM : UInt32 := if curM == 0 then 1 else 0
    let wBad ← encodeWord peTileLayout #[("k", basePe1.tiling.k.toUInt32), ("m", badM)]
    tamperGenericAndVerify basePe1.hwx basePe1.info regPeTile wBad
  ) "peTile m=")

  results := results.push (checkErr "m3 neg verifier (PE): targetCfg != 0 rejected" (do
    tamperGenericAndVerify basePe1.hwx basePe1.info regTargetCfg 0x00200000
  ) "targetCfg")

  results := results.push (checkErr "m3 neg verifier (PE): 2-source writing 0x4504 (peBias) rejected" (do
    let d1 : TensorDims := { c := 4, h := 8, w := 16 }
    let d2 : TensorDims := { c := 4, h := 1, w := 1 }
    let t ← choosePeTiling .h17 d1 d1 (some d2)
    let td ← lowerPeTask .h17 (.binary .add d1 d2 false) d1 d1 (some d2) 6 (some 7) t
    let tdBad : TaskDesc := { td with plainWrites := td.plainWrites.push { byteAddr := regPeBias, value := 0x3f800000 } }
    let sX ← makeFp16Surface .h17 "x" "input_x" .input 4 8 16
    let sZ ← makeFp16Surface .h17 "z" "input_z" .input 4 1 1
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 4 8 16
    let em ← emitProgram { target := .h17, surfaces := #[sX, sZ, sY], tasks := #[tdBad] }
    let _ ← verifyHwxTasks em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "chain constant")

  results := results.push (checkErr "m3 neg verifier (PE): 2-source missing 0x4d28 (src2BaseLo) rejected" (do
    let d1 : TensorDims := { c := 4, h := 8, w := 16 }
    let d2 : TensorDims := { c := 4, h := 1, w := 1 }
    let t ← choosePeTiling .h17 d1 d1 (some d2)
    let td ← lowerPeTask .h17 (.binary .add d1 d2 false) d1 d1 (some d2) 6 (some 7) t
    let tdBad : TaskDesc := { td with barWrites := td.barWrites.filter (fun b => b.baseByteAddr != regSrc2BaseLo) }
    let sX ← makeFp16Surface .h17 "x" "input_x" .input 4 8 16
    let sZ ← makeFp16Surface .h17 "z" "input_z" .input 4 1 1
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 4 8 16
    let em ← emitProgram { target := .h17, surfaces := #[sX, sZ, sY], tasks := #[tdBad] }
    let _ ← verifyHwxTasks em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "src2BaseLo")

  results := results.push (checkErr "m3 neg verifier (PE): L2 overlap between src1 and src2 rejected" (do
    tamperGenericAndVerify basePe2.hwx basePe2.info regSrc2Base 0
  ) "overlapping")

  results := results.push (checkErr "m3 neg verifier (PE): L2 overflow > 2 MiB rejected" (do
    tamperGenericAndVerify basePe1.hwx basePe1.info regSrc1Base 0x200000
  ) "exceeds limit")

  -- NE Act negative tests
  results := results.push (checkErr "m3 neg verifier (NE act): relu == 1 && lut == 1 rejected" (do
    let wBad ← encodeWord neActOpLayout #[("noShift", 1), ("relu", 1), ("lut", 1)]
    tamperGenericAndVerify baseNeLut.hwx baseNeLut.info regNeOp wBad
  ) "both relu")

  results := results.push (checkErr "m3 neg verifier (NE act): lut == 1 with lutSize != 128 rejected" (do
    tamperGenericAndVerify baseNeLut.hwx baseNeLut.info regLutSize 64
  ) "lutSize")

  results := results.push (checkErr "m3 neg verifier (NE act): non-zero reserved LUT word rejected" (do
    let some kSec := baseNeLut.info.kernelSection | throw "missing __KERN_0"
    let badHwx ← putU16LE baseNeLut.hwx (kSec.offset.toNat + 2 * 50) 0x0001
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasks badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "reserved word")

  results := results.push (checkErr "m3 neg verifier (NE act): dwGroups != 0x14000 rejected" (do
    let wBad ← encodeWord dwGroupsLayout #[("perGroupIn", 2)]
    tamperGenericAndVerify baseNeLut.hwx baseNeLut.info regDwGroups wBad
  ) "dwGroups")

  results := results.push (checkErr "m3 neg verifier (NE act): missing LUT relocation rejected" (do
    let lp ← buildLutTable .tanh
    let t ← chooseNeActTiling .h17 8 8 16
    let td ← lowerNeActTask .h17 (.lut .tanh) (some lp) 8 8 16 t
    let tdNoReloc : TaskDesc := { td with kernelRelocs := #[] }
    let sX ← makeFp16Surface .h17 "x" "input_x" .input 8 8 16
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 8 8 16
    let em ← emitProgram { target := .h17, surfaces := #[sX, sY], tasks := #[tdNoReloc], kernel := some lp.bytes }
    let _ ← verifyHwxTasks em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "relocation")

  results := results.push (checkErr "m3 neg verifier (NE act): L2 overflow > 2 MiB rejected" (do
    tamperGenericAndVerify baseNeLut.hwx baseNeLut.info regSrc1Base 0x200000
  ) "exceeds limit")

  -- Fused Conv + LUT negative tests
  results := results.push (checkErr "m3 neg verifier (fused conv): relu == 1 && lut == 1 rejected" (do
    let wBad ← encodeWord neOpLayout #[("biasInKernel", 1), ("relu", 1), ("lut", 1)]
    tamperGenericAndVerify baseFusedS1.hwx baseFusedS1.info regNeOp wBad
  ) "both relu")

  results := results.push (checkErr "m3 neg verifier (fused conv): chunkSize < 128 with lut == 1 rejected" (do
    tamperGenericAndVerify baseFusedS1.hwx baseFusedS1.info (regChunkSize 0) 64
  ) "128B required")

  results := results.push (checkErr "m3 neg verifier (fused conv): non-zero reserved LUT word in fused chunk rejected" (do
    let some kSec := baseFusedS1.info.kernelSection | throw "missing __KERN_0"
    let badHwx ← putU16LE baseFusedS1.hwx (kSec.offset.toNat + 2 * 60) 0x1234
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasks badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "reserved word")

  results := results.push (checkErr "m3 neg verifier (fused sparse conv): corrupted sparse stream after 128B LUT rejected" (do
    let some kSec := baseFusedS2.info.kernelSection | throw "missing __KERN_0"
    -- Chunk 0 starts with 128B LUT, then 2B bias, then u24 sparse length header at +130
    let u24Off := kSec.offset.toNat + 128 + 2
    let h1 ← putU8 baseFusedS2.hwx u24Off 0x00
    let h2 ← putU8 h1 (u24Off + 1) 0x80
    let h3 ← putU8 h2 (u24Off + 2) 0x00
    let badInfo ← validateHwx h3
    let _ ← verifyHwxTasks h3 badInfo (requireKnownTask := true)
    pure ()
  ) "exceeds chunk end")

  -- =========================================================================
  -- Milestone 3 Review Hardening Tests (B1+Q1, B2, Q4, Q3, M1, M2, M3, N1-N5)
  -- =========================================================================

  -- B1+Q1: PE & standalone NE-act direct-mode L2 output ring reservation & disjointness
  results := results.push (checkErr "m3 rev [B1/Q1] neg (PE): direct resBase = 0 overlapping src1 rejected" (do
    tamperGenericAndVerify basePe1.hwx basePe1.info regResBase 0
  ) "overlapping")

  results := results.push (checkErr "m3 rev [B1/Q1] neg (PE): direct resBase overlapping src2 rejected" (do
    let fOffS2 ← findRegFileOffset basePe2.hwx basePe2.info regSrc2Base
    let s2Base ← readU32LE basePe2.hwx fOffS2
    tamperGenericAndVerify basePe2.hwx basePe2.info regResBase s2Base
  ) "overlapping")

  results := results.push (checkErr "m3 rev [B1/Q1] neg (PE): direct resBase = 0x200000 (insufficient headroom) rejected" (do
    tamperGenericAndVerify basePe1.hwx basePe1.info regResBase 0x200000
  ) "exceeds limit")

  results := results.push (checkErr "m3 rev [B1/Q1] neg (PE): direct resBase = 0x1ffe00 (512B < 2048B headroom) rejected" (do
    tamperGenericAndVerify basePe1.hwx basePe1.info regResBase 0x1ffe00
  ) "exceeds limit")

  results := results.push (checkErr "m3 rev [B1/Q1] neg (NE act): direct resBase = 0 overlapping src1 rejected" (do
    tamperGenericAndVerify baseNeLut.hwx baseNeLut.info regResBase 0
  ) "overlapping")

  results := results.push (checkErr "m3 rev [B1/Q1] neg (NE act): direct resBase = 0x200000 (0B headroom) rejected" (do
    tamperGenericAndVerify baseNeLut.hwx baseNeLut.info regResBase 0x200000
  ) "exceeds limit")

  results := results.push (checkErr "m3 rev [B1/Q1] neg (NE act): direct resBase = 0x1fff00 (256B < 512B headroom) rejected" (do
    tamperGenericAndVerify baseNeLut.hwx baseNeLut.info regResBase 0x1fff00
  ) "exceeds limit")

  -- B2: Strict per-task-kind register allowlist (reject smuggled cross-engine & unknown regs)
  results := results.push (checkErr "m3 rev [B2] neg (PE): smuggled regNeOp (0x4904) rejected" (do
    let d : TensorDims := { c := 4, h := 8, w := 16 }
    let t ← choosePeTiling .h17 d d none
    let td ← lowerPeTask .h17 .abs d d none 6 none t
    let tdBad : TaskDesc := { td with plainWrites := td.plainWrites.push { byteAddr := regNeOp, value := 0x00103c0c } }
    let sX ← makeFp16Surface .h17 "x" "input_x" .input 4 8 16
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 4 8 16
    let em ← emitProgram { target := .h17, surfaces := #[sX, sY], tasks := #[tdBad] }
    let _ ← verifyHwxTasks em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "disallowed register")

  results := results.push (checkErr "m3 rev [B2] neg (PE): smuggled regLutBaseLo (0x5618) rejected" (do
    let d : TensorDims := { c := 4, h := 8, w := 16 }
    let t ← choosePeTiling .h17 d d none
    let td ← lowerPeTask .h17 .abs d d none 6 none t
    let tdBad : TaskDesc := { td with plainWrites := td.plainWrites.push { byteAddr := regLutBaseLo, value := 0 } }
    let sX ← makeFp16Surface .h17 "x" "input_x" .input 4 8 16
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 4 8 16
    let em ← emitProgram { target := .h17, surfaces := #[sX, sY], tasks := #[tdBad] }
    let _ ← verifyHwxTasks em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "disallowed register")

  results := results.push (checkErr "m3 rev [B2] neg (NE act): smuggled regPeOp (0x4500) rejected" (do
    let lp ← buildLutTable .tanh
    let t ← chooseNeActTiling .h17 4 8 16
    let td ← lowerNeActTask .h17 (.lut .tanh) (some lp) 4 8 16 t
    let tdBad : TaskDesc := { td with plainWrites := td.plainWrites.push { byteAddr := regPeOp, value := peOpAbsValue } }
    let sX ← makeFp16Surface .h17 "x" "input_x" .input 4 8 16
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 4 8 16
    let em ← emitProgram { target := .h17, surfaces := #[sX, sY], tasks := #[tdBad], kernel := some lp.bytes }
    let _ ← verifyHwxTasks em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "disallowed register")

  results := results.push (checkErr "m3 rev [B2] neg (NE act): smuggled regSrc2BaseLo (0x4d28) rejected" (do
    let lp ← buildLutTable .tanh
    let t ← chooseNeActTiling .h17 4 8 16
    let td ← lowerNeActTask .h17 (.lut .tanh) (some lp) 4 8 16 t
    let tdBad : TaskDesc := { td with plainWrites := td.plainWrites.push { byteAddr := regSrc2BaseLo, value := 0 } }
    let sX ← makeFp16Surface .h17 "x" "input_x" .input 4 8 16
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 4 8 16
    let em ← emitProgram { target := .h17, surfaces := #[sX, sY], tasks := #[tdBad], kernel := some lp.bytes }
    let _ ← verifyHwxTasks em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "disallowed register")

  results := results.push (checkErr "m3 rev [B2] neg (conv): smuggled unknown block offset 0x4920 rejected" (do
    let plain := baseTd1.plainWrites.push { byteAddr := 0x4920, value := 0 }
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes { baseTd1 with plainWrites := plain }
  ) "disallowed register")

  results := results.push (checkErr "m3 rev [B2] neg (conv): smuggled regPeOp (0x4500) rejected" (do
    let plain := baseTd1.plainWrites.push { byteAddr := regPeOp, value := peOpAbsValue }
    emitTaskAndVerify .h17 baseIn1 baseOut1 basePk1.bytes { baseTd1 with plainWrites := plain }
  ) "disallowed register")

  -- Q4: Symmetric whole1 full-row staging on hwSrc1 in 2-source PE (rsub with out=[128,64,512], z=[1,1,512])
  results := results.push (checkOk "m3 rev [Q4]: PE rsub [128,64,512] - [1,1,512] activates whole1 (src1ChanStride = src1Row = 1024)" (do
    let xDims : TensorDims := { c := 128, h := 64, w := 512 }
    let zDims : TensorDims := { c := 1, h := 1, w := 512 }
    let compiled ← compilePe .h17 (.binary .sub xDims zDims true) xDims
    -- For rsub, hwSrc1 = z ([1,1,512]), hwSrc2 = x ([128,64,512]), T <= 256 < 512, th < 64 -> whole1 MUST be true!
    let l2 := computePeL2Layout .h17 xDims zDims (some xDims) compiled.tiling
    if !l2.whole1 then throw s!"expected whole1=true, got tiling={repr compiled.tiling}"
    if l2.row1 != 1024 || l2.u1 != 1024 then
      throw s!"expected whole1 src1ChanStride=src1Row=1024, got row1={l2.row1}, u1={l2.u1}"
    let fOffChs ← findRegFileOffset compiled.hwx compiled.info regSrc1ChanStride
    let chsVal ← readU32LE compiled.hwx fOffChs
    if chsVal != 1024 then throw s!"encoded regSrc1ChanStride={chsVal} != 1024"
    let s ← verifyHwxTasksSummary compiled.hwx compiled.info (requireKnownTask := true)
    if s.peOp != 1 then throw "expected 1 verified PE task"
    pure ()
  ))

  results := results.push (checkErr "m3 rev [Q4] neg (PE): tampering whole1 src1ChanStride/RowStride down to 512 rejected" (do
    let xDims : TensorDims := { c := 128, h := 64, w := 512 }
    let zDims : TensorDims := { c := 1, h := 1, w := 512 }
    let compiled ← compilePe .h17 (.binary .sub xDims zDims true) xDims
    let fChs ← findRegFileOffset compiled.hwx compiled.info regSrc1ChanStride
    let fRow ← findRegFileOffset compiled.hwx compiled.info regSrc1RowStride
    let fRowA ← findRegFileOffset compiled.hwx compiled.info regSrc1RowBytesA
    let fRowB ← findRegFileOffset compiled.hwx compiled.info regSrc1RowBytesB
    let h1 ← putU32LE compiled.hwx fChs 512
    let h2 ← putU32LE h1 fRow 512
    let h3 ← putU32LE h2 fRowA 512
    let h4 ← putU32LE h3 fRowB 512
    let badInfo ← validateHwx h4
    let _ ← verifyHwxTasks h4 badInfo (requireKnownTask := true)
    pure ()
  ) "whole1=true")

  -- Q3: thresholded_relu(a) exact boundary lo = a (prevF16(a) -> 0, a -> 0, nextF16(a) -> nextF16(a))
  results := results.push (checkOk "m3 rev [Q3]: thresholded_relu(a) exact boundary lo = a" (do
    for aVal in #[0.0, 1.0, -0.5, 2.25] do
      let aBits := floatToF16CanonicalZero aVal
      let lp ← buildLutTable (.thresholdedRelu aVal)
      if lp.words[0]! != aBits then
        throw s!"thresholded_relu({aVal}): lo word {toHex lp.words[0]!.toNat} != aBits {toHex aBits.toNat}"
      let yPrev := evalLutTable lp.words lp.preScaleBits lp.inputShiftBits (prevF16 aBits)
      let yAt := evalLutTable lp.words lp.preScaleBits lp.inputShiftBits aBits
      let nextBits := nextF16 aBits
      let yNext := evalLutTable lp.words lp.preScaleBits lp.inputShiftBits nextBits
      if yPrev != f16PosZero then
        throw s!"thresholded_relu({aVal}) at prevF16: got {toHex yPrev.toNat}, want +0"
      if yAt != f16PosZero then
        throw s!"thresholded_relu({aVal}) at a: got {toHex yAt.toNat}, want +0"
      if yNext != nextBits then
        throw s!"thresholded_relu({aVal}) at nextF16: got {toHex yNext.toNat}, want {toHex nextBits.toNat}"
    pure ()
  ))

  -- M1: LUT control words 41 (k) and 42 (mode) validation
  results := results.push (checkErr "m3 rev [M1] neg (LUT): k = 0xffff rejected" (do
    let some kSec := baseNeLut.info.kernelSection | throw "missing __KERN_0"
    let badHwx ← putU16LE baseNeLut.hwx (kSec.offset.toNat + 2 * 41) 0xffff
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasks badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "scale word 41")

  results := results.push (checkErr "m3 rev [M1] neg (LUT): k = 0x40 with base != 0x01 rejected" (do
    -- baseNeLut is tanh (mode = 0x0060, base = 0x00 != 0x01)
    let some kSec := baseNeLut.info.kernelSection | throw "missing __KERN_0"
    let badHwx ← putU16LE baseNeLut.hwx (kSec.offset.toNat + 2 * 41) 0x0040
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasks badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "scale word 41")

  results := results.push (checkErr "m3 rev [M1] neg (LUT): mode = 0xffff rejected" (do
    let some kSec := baseNeLut.info.kernelSection | throw "missing __KERN_0"
    let badHwx ← putU16LE baseNeLut.hwx (kSec.offset.toNat + 2 * 42) 0xffff
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasks badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "high byte")

  results := results.push (checkErr "m3 rev [M1] neg (LUT): unsupported base mode 0x1f rejected" (do
    let some kSec := baseNeLut.info.kernelSection | throw "missing __KERN_0"
    let badHwx ← putU16LE baseNeLut.hwx (kSec.offset.toNat + 2 * 42) 0x001f
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasks badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "unsupported base mode")

  results := results.push (checkErr "m3 rev [M1] neg (LUT): unsupported symmetry flag 0x40 rejected" (do
    let some kSec := baseNeLut.info.kernelSection | throw "missing __KERN_0"
    let badHwx ← putU16LE baseNeLut.hwx (kSec.offset.toNat + 2 * 42) 0x0040
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasks badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "unsupported symmetry flag")

  -- M2: Fused conv + bias + shifted LUT overflow guard
  results := results.push (checkErr "m3 rev [M2] neg (fused conv): bias + LUT shift fp16 overflow rejected" (do
    let sh : Conv1x1Shape := { h := 4, w := 8, cin := 4, cout := 4 }
    let mut bigBias := ByteArray.empty
    for _ in [:4] do
      bigBias := pushU16LE bigBias 0x7bff -- 65504 (max finite fp16)
    -- rsqrt(eps = 32.0) shifts bias by +32 -> 65536 -> overflows fp16!
    compileConv1x1 .h17 sh (mkTaggedWeights 4 4) (some bigBias) false none (.lut (.rsqrt 32.0))
  ) "overflowed fp16")

  -- M3: Finiteness & fp16-range validation for LUT parameters
  results := results.push (checkErr "m3 rev [M3] neg: clip rejects Inf" (do
    buildLutTable (.clip (-1.0) (Float.ofBits 0x7ff0000000000000))
  ) "finite")

  results := results.push (checkErr "m3 rev [M3] neg: hard_sigmoid rejects alpha = 20000 (preScale overflows fp16)" (do
    buildLutTable (.hardSigmoid 20000.0 0.5)
  ) "overflows fp16")

  results := results.push (checkErr "m3 rev [M3] neg: hard_sigmoid rejects alpha = 1e-9 (preScale underflows to 0)" (do
    buildLutTable (.hardSigmoid 1e-9 0.5)
  ) "underflows")

  results := results.push (checkErr "m3 rev [M3] neg: hard_sigmoid rejects beta = 1e5" (do
    buildLutTable (.hardSigmoid 0.2 1e5)
  ) "overflows fp16")

  results := results.push (checkErr "m3 rev [M3] neg: elu(70000) rejected" (do
    buildLutTable (.elu 70000.0)
  ) "overflows fp16")

  results := results.push (checkErr "m3 rev [M3] neg: leaky_relu(NaN) rejected" (do
    buildLutTable (.leakyRelu (0.0 / 0.0))
  ) "finite")

  results := results.push (checkErr "m3 rev [M3] neg: scaled_tanh(70000, 1) rejected" (do
    buildLutTable (.scaledTanh 70000.0 1.0)
  ) "overflows fp16")

  -- N1, N2, N3, N4 checks
  results := results.push (checkOk "m3 rev [N1/N4]: parseFloat? 0e400/huge-exp & PeOp.scalarSub(+-0) canonicalization" (do
    match parseFloat? "0e400" with
    | some v => if v.isNaN || v != 0.0 then throw s!"parseFloat? 0e400 = {v}, want 0.0"
    | none => throw "parseFloat? 0e400 returned none"
    match parseFloat? "1e9999999999" with
    | some v => if !v.isInf then throw s!"parseFloat? 1e9999999999 = {v}, want +Inf"
    | none => throw "parseFloat? 1e9999999999 returned none"
    if PeOp.scalarSub f16PosZero != .chain none (some f16PosZero) none then
      throw "PeOp.scalarSub(+0) did not canonicalize to +0"
    if PeOp.scalarSub f16NegZero != .chain none (some f16PosZero) none then
      throw "PeOp.scalarSub(-0) did not canonicalize to +0"
    pure ()
  ))

  results := results.push (checkErr "m3 rev [N2] neg (PE): over-wide src1ChanStride rejected (exact match)" (do
    -- basePe1 has W=16, T>=16 -> expRow1 = 32; tamper to 64 (and matching rowStride/rowBytes)
    let fChs ← findRegFileOffset basePe1.hwx basePe1.info regSrc1ChanStride
    let fRow ← findRegFileOffset basePe1.hwx basePe1.info regSrc1RowStride
    let fRowA ← findRegFileOffset basePe1.hwx basePe1.info regSrc1RowBytesA
    let fRowB ← findRegFileOffset basePe1.hwx basePe1.info regSrc1RowBytesB
    let fRes ← findRegFileOffset basePe1.hwx basePe1.info regResBase
    let h1 ← putU32LE basePe1.hwx fChs 64
    let h2 ← putU32LE h1 fRow (8 * 64)
    let h3 ← putU32LE h2 fRowA (8 * 64)
    let h4 ← putU32LE h3 fRowB (8 * 64)
    let h5 ← putU32LE h4 fRes (8 * 8 * 64)
    let badInfo ← validateHwx h5
    let _ ← verifyHwxTasks h5 badInfo (requireKnownTask := true)
    pure ()
  ) "src1ChanStride (64) != expected 32")

  results := results.push (checkErr "m3 rev [N3] neg (PE): scalarDiv rejects tiny subnormal whose reciprocal overflows fp16" (do
    PeOp.scalarDiv f16MinSubnormal
  ) "overflows fp16")

  pure results

private def runM4Tests : IO (Array TestResult) := do
  let mut results : Array TestResult := #[]

  -- =========================================================================
  -- Milestone 4: Vane.Alloc & Vane.Reach Unit Tests
  -- =========================================================================

  results := results.push (checkOk "m4 unit: Vane.Alloc overlapping disjointness, reuse, capacity, 64B alignment" (do
    -- 4 buffers: b0 [0,1], b1 [1,2], b2 [2,3], b3 [0,3] (skip connection live across all)
    let bufs : List Vane.Alloc.LiveBuf := [
      { id := 0, bytes := 1024, first := 0, last := 1, priority := 1 },
      { id := 1, bytes := 1024, first := 1, last := 2, priority := 1 },
      { id := 2, bytes := 1024, first := 2, last := 3, priority := 1 },
      { id := 3, bytes := 2048, first := 0, last := 3, priority := 3 }
    ]
    let placed := Vane.Alloc.allocate true bufs
    if placed.length != 4 then throw s!"expected 4 placed buffers, got {placed.length}"
    for p in placed do
      if p.base % 64 != 0 then throw s!"buffer {p.id} base {p.base} not 64B aligned"
    -- Total sum is 5120B, but b0 [0,1] and b2 [2,3] don't overlap so topOf must reuse space (<= 4096B)
    let top := Vane.Alloc.topOf placed
    if top > 4096 then throw s!"expected reuse top <= 4096, got {top}"
    -- Pairwise disjoint whenever live intervals overlap
    for a in placed do
      for b in placed do
        if a.id != b.id && a.first <= b.last && b.first <= a.last then
          if a.base < b.hi && b.base < a.hi then
            throw s!"overlapping live buffers {a.id} and {b.id}"
    -- findFreeOffset places cleanly and returns none when full
    let occ := Vane.Alloc.liveAt placed 1
    match Vane.Alloc.findFreeOffset occ 512 (2 * 1024 * 1024) with
    | none => throw "findFreeOffset unexpectedly returned none"
    | some off =>
      if off % 64 != 0 then throw s!"findFreeOffset {off} not 64B aligned"
      for r in occ do
        if off < r.hi && r.base < off + 512 then
          throw s!"findFreeOffset {off} overlapped live range [{r.base}, {r.hi})"
    match Vane.Alloc.findFreeOffset [{ base := 0, bytes := 2 * 1024 * 1024 }] 64 (2 * 1024 * 1024) with
    | some off => throw s!"expected none when L2 full, got {off}"
    | none => pure ()
  ))

  results := results.push (checkOk "m4 unit: Vane.Reach DAG transitive closure (diamond, chain, disconnected)" (do
    -- 5-node DAG: 0->1, 0->2, 1->3, 2->3; node 4 isolated
    let step : Nat → Nat → Bool
      | 0, 1 => true
      | 0, 2 => true
      | 1, 3 => true
      | 2, 3 => true
      | _, _ => false
    let tbl := Vane.Reach.reachTable step 5
    if !(Vane.Reach.reachB tbl 0 3) then throw "expected reachB 0 3 = true"
    if !(Vane.Reach.reachB tbl 0 1) then throw "expected reachB 0 1 = true"
    if Vane.Reach.reachB tbl 1 2 then throw "expected reachB 1 2 = false"
    if Vane.Reach.reachB tbl 3 0 then throw "expected reachB 3 0 = false"
    if Vane.Reach.reachB tbl 0 4 then throw "expected reachB 0 4 = false"
    if Vane.Reach.reachB tbl 2 2 then throw "expected reachB 2 2 = false"
    pure ()
  ))

  -- =========================================================================
  -- Milestone 4: End-to-End Multi-Task Graph Compilation (H17 & H16)
  -- =========================================================================

  let mkGraphW1x1 (cin cout : Nat) (seed : Nat := 1) : ByteArray := Id.run do
    let mut b := ByteArray.empty
    for o in [:cout] do
      for i in [:cin] do
        let v := if (o + i + seed) % 3 == 0 then 0x3800 else 0x3400 -- 0.5 or 0.25
        b := pushU16LE b v
    return b

  let mkGraphW3x3 (cin cout : Nat) (dw : Bool := false) : ByteArray := Id.run do
    let k := if dw then cout * 9 else cout * cin * 9
    let mut b := ByteArray.empty
    for i in [:k] do
      b := pushU16LE b (if i % 9 == 4 then 0x3c00 else 0x3000)
    return b

  let mkGraphBias (cout : Nat) : ByteArray := Id.run do
    let mut b := ByteArray.empty
    for i in [:cout] do
      b := pushU16LE b (if i % 2 == 0 then 0x3400 else 0xb400)
    return b

  let verifyCompiledGraph (g : Vane.Graph.CompiledGraph) (expTasks : Nat) : Except String TaskVerifySummary := do
    if g.taskCount != expTasks then
      throw s!"expected {expTasks} tasks, got {g.taskCount}"
    if g.info.taskStream.taskCount != expTasks then
      throw s!"HWX stream taskCount {g.info.taskStream.taskCount} != {expTasks}"
    validateSections g.hwx g.sections
    let s ← verifyHwxTasksSummary g.hwx g.info (requireKnownTask := true)
    if s.total != expTasks then
      throw s!"verified task count {s.total} != {expTasks}"
    pure s

  -- M4 E2E 1: 2-layer & 4-layer conv chains in L2, DRAM (verifying scratch reuse!), and auto on H17 & H16
  for tgt in #[Target.h17, Target.h16] do
    results := results.push (checkOk s!"m4 ({tgt.name}): 4-layer conv chain (1x1+3x3 dense+dw+stride2) in L2, DRAM (with reuse), and auto" (do
      let makeChain (mode : Vane.Graph.EdgePlacementMode) : Vane.Graph.GraphSpec := {
        target := tgt
        defaultPlacement := mode
        inputs := #[{ name := "x", shape := { c := 16, h := 16, w := 16 } }]
        nodes := #[
          { id := "c1", inputs := #["x"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 1) (some (mkGraphBias 16)) .relu },
          { id := "c2", inputs := #["c1"], op := .conv3x3 .depthwise 16 16 1 .same (mkGraphW3x3 16 16 true) none .none },
          { id := "c3", inputs := #["c2"], op := .conv3x3 .dense 16 32 2 .same (mkGraphW3x3 16 32 false) (some (mkGraphBias 32)) .relu },
          { id := "c4", inputs := #["c3"], op := .conv1x1 32 16 (mkGraphW1x1 32 16 2) none .none }
        ]
      }
      -- L2 mode
      let gL2 ← Vane.Graph.compileGraph (makeChain .l2)
      let sL2 ← verifyCompiledGraph gL2 4
      if sL2.neConv != 4 then throw s!"L2 chain: expected 4 conv tasks, got {sL2.neConv}"
      if gL2.scratchBytes != 0 then throw s!"L2 chain: expected 0 scratchBytes, got {gL2.scratchBytes}"

      -- DRAM mode: verify scratch reuse!
      -- Intermediates: c1 [16,16,16] = 16*16*64 = 16384B, c2 [16,16,16] = 16384B, c3 [32,8,8] = 32*8*64 = 16384B
      -- Sum = 49152B, but only 2 buffers are ever live simultaneously (max live = 32768B)!
      let gDram ← Vane.Graph.compileGraph (makeChain .dram)
      let _ ← verifyCompiledGraph gDram 4
      let sumInterm := 16384 + 16384 + 16384
      if gDram.scratchBytes == 0 then throw "DRAM chain: expected scratchBytes > 0"
      if gDram.scratchBytes >= sumInterm then
        throw s!"DRAM chain: expected scratch reuse (< {sumInterm}B), got {gDram.scratchBytes}B"
      if gDram.info.scratchSection.isNone then throw "DRAM chain: missing __DATA,__bss scratch section"

      -- Auto mode
      let gAuto ← Vane.Graph.compileGraph (makeChain .auto)
      let _ ← verifyCompiledGraph gAuto 4
      pure ()
    ))

    -- M4 E2E 2: Mixed NE + PE + standalone LUT pipelines + fused LUT conv
    results := results.push (checkOk s!"m4 ({tgt.name}): mixed conv -> pe1 -> lut(silu) -> conv & fused conv+gelu -> pe2(add)" (do
      let spec1 : Vane.Graph.GraphSpec := {
        target := tgt
        defaultPlacement := .auto
        inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
        nodes := #[
          { id := "c1", inputs := #["x"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 3) none .none },
          { id := "p1", inputs := #["c1"], op := .pe1 (.chain (some 0x3800) (some 0x3400) none) },
          { id := "l1", inputs := #["p1"], op := .lut (.lut .silu) },
          { id := "c2", inputs := #["l1"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 4) none .relu }
        ]
      }
      let g1 ← Vane.Graph.compileGraph spec1
      let s1 ← verifyCompiledGraph g1 4
      if s1.neConv != 2 || s1.peOp != 1 || s1.neAct != 1 then
        throw s!"unexpected summary {repr s1}"

      -- Also run same mixed pipeline in DRAM mode (exercises cross-engine DRAM RAW + WAR)
      let g1Dram ← Vane.Graph.compileGraph { spec1 with defaultPlacement := .dram }
      let _ ← verifyCompiledGraph g1Dram 4

      -- Fused conv+LUT -> PE add with skip
      let spec2 : Vane.Graph.GraphSpec := {
        target := tgt
        defaultPlacement := .auto
        inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
        nodes := #[
          { id := "c1", inputs := #["x"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 5) (some (mkGraphBias 16)) (.lut .gelu) },
          { id := "add", inputs := #["c1", "x"], op := .pe2 .add false }
        ]
      }
      let g2 ← Vane.Graph.compileGraph spec2
      let s2 ← verifyCompiledGraph g2 2
      if s2.neConv != 1 || s2.peOp != 1 then
        throw s!"unexpected fused+pe summary {repr s2}"
      pure ()
    ))

    -- M4 E2E 3: Residual blocks (L2, DRAM, hybrid L2+DRAM), 2-input graph, and broadcast skip
    results := results.push (checkOk s!"m4 ({tgt.name}): residual block (L2, DRAM, hybrid), 2-input graph, and broadcast skip" (do
      -- Residual: conv3x3 -> conv1x1 -> pe add(c2, x) in L2, DRAM, and hybrid (c1=l2, c2=dram)
      let mkRes (p1 p2 : Option Vane.Graph.EdgePlacementMode) (defP : Vane.Graph.EdgePlacementMode) : Vane.Graph.GraphSpec := {
        target := tgt
        defaultPlacement := defP
        inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
        nodes := #[
          { id := "c1", inputs := #["x"], op := .conv3x3 .dense 16 16 1 .same (mkGraphW3x3 16 16 false) none .relu, outPlacement := p1 },
          { id := "c2", inputs := #["c1"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 6) none .none, outPlacement := p2 },
          { id := "res", inputs := #["c2", "x"], op := .pe2 .add false }
        ]
      }
      let _ ← verifyCompiledGraph (← Vane.Graph.compileGraph (mkRes none none .l2)) 3
      let _ ← verifyCompiledGraph (← Vane.Graph.compileGraph (mkRes none none .dram)) 3
      let gHyb ← Vane.Graph.compileGraph (mkRes (some .l2) (some .dram) .auto)
      let _ ← verifyCompiledGraph gHyb 3
      if gHyb.nodes[0]!.outLocation.placementName != "l2" || gHyb.nodes[1]!.outLocation.placementName != "dram" then
        throw "hybrid residual placement mismatch"

      -- Residual where skip is an intermediate node (c0 -> c1 -> add(c1, c0))
      let specSkipInterm : Vane.Graph.GraphSpec := {
        target := tgt
        defaultPlacement := .l2
        inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
        nodes := #[
          { id := "c0", inputs := #["x"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 7) none .relu },
          { id := "c1", inputs := #["c0"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 8) none .none },
          { id := "add", inputs := #["c1", "c0"], op := .pe2 .add false }
        ]
      }
      let _ ← verifyCompiledGraph (← Vane.Graph.compileGraph specSkipInterm) 3
      let _ ← verifyCompiledGraph (← Vane.Graph.compileGraph { specSkipInterm with defaultPlacement := .dram }) 3

      -- 2-network-input graph + broadcast skip [C, 1, 1]
      let spec2In : Vane.Graph.GraphSpec := {
        target := tgt
        defaultPlacement := .auto
        inputs := #[
          { name := "x", shape := { c := 16, h := 8, w := 16 } },
          { name := "z", shape := { c := 16, h := 1, w := 1 } }
        ]
        nodes := #[
          { id := "c1", inputs := #["x"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 9) none .relu },
          { id := "add", inputs := #["c1", "z"], op := .pe2 .add false }
        ]
      }
      let g2In ← Vane.Graph.compileGraph spec2In
      let _ ← verifyCompiledGraph g2In 2
      if g2In.info.surfaces.size != 3 then throw s!"expected 3 surfaces, got {g2In.info.surfaces.size}"
      pure ()
    ))

  -- M4 E2E 4: Auto-spill from L2 to DRAM when simultaneous live tensors exceed 2 MiB
  results := results.push (checkOk "m4: auto-spill moves oversized live intermediate from L2 to DRAM scratch" (do
    -- Shape [64, 128, 128]: 1 tensor = 64 * 128 * 256 = 2,097,152B (2 MiB exactly -> ineligible/spilled!)
    -- Use [32, 128, 128]: 1 tensor = 32 * 128 * 256 = 1,048,576B (1 MiB) -> 2 simultaneous live tensors (c0 + c1)
    -- plus staging exceed 2 MiB, forcing auto-spill of at least one to DRAM!
    let specSpill : Vane.Graph.GraphSpec := {
      target := .h17
      defaultPlacement := .auto
      inputs := #[{ name := "x", shape := { c := 32, h := 128, w := 128 } }]
      nodes := #[
        { id := "p0", inputs := #["x"], op := .pe1 .abs },
        { id := "p1", inputs := #["p0"], op := .pe1 (.chain (some 0x3800) none none) },
        { id := "add", inputs := #["p1", "p0"], op := .pe2 .add false }
      ]
    }
    let gSpill ← Vane.Graph.compileGraph specSpill
    let _ ← verifyCompiledGraph gSpill 3
    if gSpill.scratchBytes == 0 then
      throw "expected auto-spill to allocate DRAM scratch > 0"
    let spilledCount := (gSpill.nodes.filter fun nd => nd.outLocation.placementName == "dram").size
    if spilledCount == 0 then
      throw "expected at least one spilled DRAM intermediate"
    pure ()
  ))

  -- M4 E2E 5: PE [DL] rule forces staged mode when k = 4, W > 16 (e.g. W = 64, stripW = 16)
  results := results.push (checkOk "m4: PE [DL] forces staged mode for W=64 (k=4, stripW=16 % 32 != 0)" (do
    -- Shape [16, 4, 64] on H17/H16: W=64, 64 > 16 and stripW(k=4)=16 % 32 != 0 -> direct MUST be false!
    -- Wait: choosePeTiling prefers k=5 (stripW=32) when it fits! To hit k=4 (stripW=16) naturally,
    -- make C large enough that k=5 staging exceeds 2 MiB or use compilePe with W=64 and C=32768?
    -- Actually check peIsDirect directly AND verify a multi-task graph where PE has W=64, C=64, H=4.
    if peIsDirect .h17 64 4 then
      throw "peIsDirect .h17 64 4 should be false under [DL]"
    if peIsDirect .h16 64 4 then
      throw "peIsDirect .h16 64 4 should be false under [DL]"
    if !(peIsDirect .h17 16 4) then
      throw "peIsDirect .h17 16 4 should be true (outW <= stripW)"
    if !(peIsDirect .h17 64 5) then
      throw "peIsDirect .h17 64 5 should be true (stripW=32 % 32 == 0)"
    -- Also test large C where k=4 is forced (or any multi-task PE with W=64 passes verifyHwxTasksSummary)
    let specW64 : Vane.Graph.GraphSpec := {
      target := .h17
      defaultPlacement := .auto
      inputs := #[{ name := "x", shape := { c := 8, h := 4, w := 64 } }]
      nodes := #[
        { id := "p1", inputs := #["x"], op := .pe1 .abs },
        { id := "p2", inputs := #["p1"], op := .pe1 (.chain (some 0x3800) none none) }
      ]
    }
    let gW64 ← Vane.Graph.compileGraph specW64
    let _ ← verifyCompiledGraph gW64 2
    pure ()
  ))

  -- =========================================================================
  -- Milestone 4: All 11 Mandatory Negative Multi-Task Verifier Tests
  -- =========================================================================

  let findTaskRegFileOffset (hwx : ByteArray) (info : HwxInfo) (taskIdx regAddr : Nat) : Except String Nat := do
    let textBytes ← readSlice hwx info.textSection.offset.toNat info.textSection.size.toNat
    let tasks ← decodeTaskStream info.target textBytes
    let some t := tasks[taskIdx]? | throw s!"no task {taskIdx}"
    let some w := t.writes.find? (fun w => w.byteAddr == regAddr)
      | throw s!"task {taskIdx}: register {toHex regAddr} not found"
    pure (info.textSection.offset.toNat + w.payloadOffset)

  let tamperTaskRegAndVerify (hwx : ByteArray) (info : HwxInfo) (taskIdx regAddr : Nat) (newVal : UInt32) :
      Except String Unit := do
    let fOff ← findTaskRegFileOffset hwx info taskIdx regAddr
    let badHwx ← putU32LE hwx fOff newVal
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasks badHwx badInfo (requireKnownTask := true)
    pure ()

  let tamperTaskHeaderAndVerify (hwx : ByteArray) (info : HwxInfo) (taskIdx wordIdx : Nat) (newVal : UInt32) :
      Except String Unit := do
    let textBytes ← readSlice hwx info.textSection.offset.toNat info.textSection.size.toNat
    let tasks ← decodeTaskStream info.target textBytes
    let some t := tasks[taskIdx]? | throw s!"no task {taskIdx}"
    let fOff := info.textSection.offset.toNat + t.startOffset + 4 * wordIdx
    let badHwx ← putU32LE hwx fOff newVal
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasks badHwx badInfo (requireKnownTask := true)
    pure ()

  -- Baseline compiled multi-task fixtures for the 11 negative tests
  let base2DramConv ← match Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .dram
      inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
      nodes := #[
        { id := "c1", inputs := #["x"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 1) none .none },
        { id := "c2", inputs := #["c1"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 2) none .none }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"base2DramConv setup failed: {e}")

  let base3DramConv ← match Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .dram
      inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
      nodes := #[
        { id := "c1", inputs := #["x"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 1) none .none },
        { id := "c2", inputs := #["c1"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 2) none .none },
        { id := "c3", inputs := #["c2"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 3) none .none }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"base3DramConv setup failed: {e}")

  let baseNeToPe ← match Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .dram
      inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
      nodes := #[
        { id := "c1", inputs := #["x"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 1) none .none },
        { id := "p1", inputs := #["c1"], op := .pe1 .abs }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"baseNeToPe setup failed: {e}")

  let basePeToNe ← match Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .l2
      inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
      nodes := #[
        { id := "p1", inputs := #["x"], op := .pe1 .abs },
        { id := "c1", inputs := #["p1"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 1) none .none }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"basePeToNe setup failed: {e}")

  let basePeSrc2 ← match Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .dram
      inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
      nodes := #[
        { id := "p0", inputs := #["x"], op := .pe1 .abs },
        { id := "add", inputs := #["x", "p0"], op := .pe2 .add false }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"basePeSrc2 setup failed: {e}")

  let mkSrcDepWord (name : String) (code taskId depDist depKind : UInt32) : Except String UInt32 :=
    encodeWord (srcDepMultiLayout name) #[
      ("code", code),
      ("task", taskId),
      ("depDist", depDist),
      ("depKind", depKind)
    ]

  -- Neg #1: Missing 0x4d00 depKind on NE task reading DRAM scratch written by i-1 (srcReadNE)
  results := results.push (checkErr "m4 neg #1: missing 0x4d00 depKind on NE DRAM RAW (srcReadNE) rejected" (do
    let fDep ← findTaskRegFileOffset base2DramConv.hwx base2DramConv.info 1 regSrc1Dep
    let origCode := (← readU32LE base2DramConv.hwx fDep) &&& 0xff
    let wNoDep ← mkSrcDepWord "src1Dep" origCode 1 0 0
    tamperTaskRegAndVerify base2DramConv.hwx base2DramConv.info 1 regSrc1Dep wNoDep
  ) "srcReadNE")

  -- Neg #2: Non-zero depDist = 1 on 0x4d00 without intervening HB
  results := results.push (checkErr "m4 neg #2: non-zero depDist=1 on 0x4d00 without intervening HB rejected" (do
    let fDep ← findTaskRegFileOffset base3DramConv.hwx base3DramConv.info 2 regSrc1Dep
    let origCode := (← readU32LE base3DramConv.hwx fDep) &&& 0xff
    let wBad ← mkSrcDepWord "src1Dep" origCode 2 1 2
    tamperTaskRegAndVerify base3DramConv.hwx base3DramConv.info 2 regSrc1Dep wBad
  ) "srcReadNE")

  -- Neg #3: Missing bit 5 on PE task reading NE's DRAM scratch / overlapping NE L2 ([XE] / DRAM RAW)
  results := results.push (checkErr "m4 neg #3: missing bit 5 on PE task after NE ([XE] / DRAM RAW) rejected" (do
    let fCfg ← findTaskRegFileOffset baseNeToPe.hwx baseNeToPe.info 1 regSrc1Cfg
    let orig ← readU32LE baseNeToPe.hwx fCfg
    let stripped := orig &&& (~~~(cfgBitWaitNe5 ||| cfgBitWaitNe4))
    let h1 ← putU32LE baseNeToPe.hwx fCfg stripped
    let h2 ← match findTaskRegFileOffset h1 baseNeToPe.info 1 regSrc1Dep with
      | .ok fDep =>
        let origCode := (← readU32LE h1 fDep) &&& 0xff
        let wNoDep ← mkSrcDepWord "src1Dep" origCode 1 0 0
        putU32LE h1 fDep wNoDep
      | .error _ => pure h1
    let badInfo ← validateHwx h2
    let _ ← verifyHwxTasks h2 badInfo (requireKnownTask := true)
    pure ()
  ) "S5")

  -- Neg #4: Missing bit 22 on NE task reading PE's L2 / overlapping PE L2 ([XE] / L2 RAW)
  results := results.push (checkErr "m4 neg #4: missing bit 22 on NE task after PE ([XE] / L2 RAW) rejected" (do
    let fCfg ← findTaskRegFileOffset basePeToNe.hwx basePeToNe.info 1 regSrc1Cfg
    let orig ← readU32LE basePeToNe.hwx fCfg
    let stripped := orig &&& (~~~(cfgBitWaitPe22 ||| cfgBitWaitPe20 ||| cfgBitHandoffPe))
    tamperTaskRegAndVerify basePeToNe.hwx basePeToNe.info 1 regSrc1Cfg stripped
  ) "S5 L2 hazard")

  -- Neg #5: Missing src2 wait (bit 22 / 0x4d04 depKind) on 2-source PE whose src2 reads scratch (src2AheadPE)
  results := results.push (checkErr "m4 neg #5: missing src2 wait on 2-source PE reading scratch (src2AheadPE) rejected" (do
    let fCfg1 ← findTaskRegFileOffset basePeSrc2.hwx basePeSrc2.info 1 regSrc1Cfg
    let fCfg2 ← findTaskRegFileOffset basePeSrc2.hwx basePeSrc2.info 1 regSrc2Cfg
    let v1 ← readU32LE basePeSrc2.hwx fCfg1
    let v2 ← readU32LE basePeSrc2.hwx fCfg2
    let mask := ~~~(cfgBitWaitPe22 ||| cfgBitWaitPe20 ||| cfgBitWaitNe5 ||| cfgBitWaitNe4)
    let h1 ← putU32LE basePeSrc2.hwx fCfg1 (v1 &&& mask)
    let h2 ← putU32LE h1 fCfg2 (v2 &&& mask)
    let h3 ← match findTaskRegFileOffset h2 basePeSrc2.info 1 regSrc2Dep with
      | .ok fDep2 =>
        let origCode := (← readU32LE h2 fDep2) &&& 0xff
        let wNoDep2 ← mkSrcDepWord "src2Dep" origCode 1 0 0
        putU32LE h2 fDep2 wNoDep2
      | .error _ => pure h2
    let h4 ← match findTaskRegFileOffset h3 basePeSrc2.info 1 regSrc1Dep with
      | .ok fDep1 =>
        let origCode := (← readU32LE h3 fDep1) &&& 0xff
        let wNoDep1 ← mkSrcDepWord "src1Dep" origCode 1 0 0
        putU32LE h3 fDep1 wNoDep1
      | .error _ => pure h3
    let badInfo ← validateHwx h4
    let _ ← verifyHwxTasks h4 badInfo (requireKnownTask := true)
    pure ()
  ) "src2AheadPE")

  -- Neg #6: Missing whole-task wait on task reusing DRAM scratch read by earlier task (DRAM WAR)
  results := results.push (checkErr "m4 neg #6: missing whole-task wait on DRAM scratch reuse (DRAM WAR) rejected" (do
    -- c0 (NE conv) writes slot 0, p1 (PE abs) reads slot 0 & writes slot 1,
    -- c2 (NE conv) reads slot 1 (via 0x4d00) & REUSES slot 0!
    -- programOrderNEcompute gives HB(0, 2) (so WAW between c0 & c2 is ordered),
    -- shifting c2's L2 bases to +0x40000 avoids L2 overlap with p1,
    -- 0x4d00 orders DRAM RAW on slot 1, so clearing bit 22 on c2 isolates DRAM WAR with p1!
    let gWar ← Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .dram
      inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
      nodes := #[
        { id := "c0", inputs := #["x"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 1) none .none },
        { id := "p1", inputs := #["c0"], op := .pe1 .abs },
        { id := "c2", inputs := #["p1"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 2) none .none },
        { id := "c3", inputs := #["c2"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 3) none .none }
      ]
    }
    let fCfg ← findTaskRegFileOffset gWar.hwx gWar.info 2 regSrc1Cfg
    let origCfg ← readU32LE gWar.hwx fCfg
    let stripped := origCfg &&& (~~~(cfgBitWaitPe22 ||| cfgBitWaitPe20))
    let fSrc1 ← findTaskRegFileOffset gWar.hwx gWar.info 2 regSrc1Base
    let fRes ← findTaskRegFileOffset gWar.hwx gWar.info 2 regResBase
    let s1Base ← readU32LE gWar.hwx fSrc1
    let rBase ← readU32LE gWar.hwx fRes
    let h1 ← putU32LE gWar.hwx fCfg stripped
    let h2 ← putU32LE h1 fSrc1 (s1Base + 0x40000)
    let h3 ← putU32LE h2 fRes (rBase + 0x40000)
    let badInfo ← validateHwx h3
    let _ ← verifyHwxTasks h3 badInfo (requireKnownTask := true)
    pure ()
  ) "DRAM WAR")

  -- Helper to rebuild a TaskDesc from a DecodedTask (with optional plain-write filter/append)
  let decodedTaskToDesc (t : DecodedTask) (ws : Array DecodedWrite) (isFirst isLast : Bool) : Except String TaskDesc := do
    let mut plain : Array RegWrite := #[]
    let mut barW : Array BarRegWrite := #[]
    let mut relocs : Array TaskReloc := #[]
    for w in ws do
      match w.barSlot with
      | none => plain := plain.push { byteAddr := w.byteAddr, value := w.value }
      | some s =>
        let recIdx := barW.size
        barW := barW.push { baseByteAddr := w.byteAddr, barSlot := s, values := #[w.value] }
        if w.byteAddr == regKernelBaseLo || w.byteAddr == regLutBaseLo then
          relocs := relocs.push { barRecIndex := recIdx, payloadWordIndex := 0 }
    let ene ← decodeFieldChecked taskHeader8MultiLayout t.header[8]! "ene"
    pure {
      taskId := t.taskIndex
      eneOverride := some ene.toNat
      plainWrites := plain
      barWrites := barW
      kernelRelocs := relocs
      isFirst := isFirst
      isLast := isLast
      liveL2Mask := t.header[7]!
    }

  -- Neg #7: Omitted disabled chunk slot 0x5520+4*s on NE conv task with taskId > 0 (chunk-slot carry-over)
  results := results.push (checkErr "m4 neg #7: omitted disabled chunk slot on NE conv taskId > 0 rejected" (do
    let sh0 : ConvSpec := { kernelSize := 1, h := 8, w := 16, cin := 16, cout := 16 }
    let sh1 : ConvSpec := { kernelSize := 1, h := 8, w := 16, cin := 16, cout := 1 }
    let g2 ← Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .l2
      inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
      nodes := #[
        { id := "c1", inputs := #["x"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 1) none .none },
        { id := "c2", inputs := #["c1"], op := .conv1x1 16 1 (mkGraphW1x1 16 1 2) none .none }
      ]
    }
    let textBytes ← readSlice g2.hwx g2.info.textSection.offset.toNat g2.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let t0 ← match tasks[0]? with | some t => pure t | none => throw "missing task 0"
    let t1 ← match tasks[1]? with | some t => pure t | none => throw "missing task 1"
    let td0 ← decodedTaskToDesc t0 t0.writes true false
    let w1NoSlot15 := t1.writes.filter (fun w => w.byteAddr != regChunkSlot 15)
    let td1 ← decodedTaskToDesc t1 w1NoSlot15 false true
    let sIn ← makeFp16Surface .h17 "x" "input_x" .input sh0.cin sh0.h sh0.w
    let sOut ← makeFp16Surface .h17 "y" "output_c2" .output sh1.cout sh1.h sh1.w
    let some kSec := g2.info.kernelSection | throw "missing __KERN_0"
    let kernBytes ← readSlice g2.hwx kSec.offset.toNat kSec.size.toNat
    let em ← emitProgram { target := .h17, surfaces := #[sIn, sOut], kernel := some kernBytes, tasks := #[td0, td1] }
    let _ ← verifyHwxTasks em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "carry-over")

  -- Neg #8: Scratch DMA offset out of bounds of __DATA,__bss section size
  results := results.push (checkErr "m4 neg #8: scratch DMA offset out of bounds of __DATA,__bss rejected" (do
    let some scSec := base2DramConv.info.scratchSection | throw "missing scratch section"
    tamperTaskRegAndVerify base2DramConv.hwx base2DramConv.info 0 regResBaseLo scSec.size.toUInt32
  ) "exceeds")

  -- Neg #9: L2-resident tensor overwritten by an unrelated intermediate task while still live
  results := results.push (checkErr "m4 neg #9: live L2-resident tensor overwritten by intermediate task rejected" (do
    let gLive ← Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .l2
      inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
      nodes := #[
        { id := "c0", inputs := #["x"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 1) none .none },
        { id := "c1", inputs := #["x"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 2) none .none },
        { id := "add", inputs := #["c0", "c1"], op := .pe2 .add false }
      ]
    }
    let fC0Res ← findTaskRegFileOffset gLive.hwx gLive.info 0 regResBase
    let c0Base ← readU32LE gLive.hwx fC0Res
    tamperTaskRegAndVerify gLive.hwx gLive.info 1 regResBase c0Base
  ) "overwritten while live")

  -- Neg #10: Wrong multi-task header[2]/header[8] flags, 0x5900 on non-last task, dangling bit 22
  results := results.push (checkErr "m4 neg #10a: wrong header[8] isLast=1 on non-last task rejected" (do
    let textBytes ← readSlice base2DramConv.hwx base2DramConv.info.textSection.offset.toNat base2DramConv.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let h8 ← match tasks[0]? with | some t => pure t.header[8]! | none => throw "missing task 0"
    let badH8 := h8 ||| ((1 : UInt32) <<< 3)
    tamperTaskHeaderAndVerify base2DramConv.hwx base2DramConv.info 0 8 badH8
  ) "isLast")

  results := results.push (checkErr "m4 neg #10b: 0x5900 output stage emitted on non-last task rejected" (do
    let textBytes ← readSlice base2DramConv.hwx base2DramConv.info.textSection.offset.toNat base2DramConv.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let t0 ← match tasks[0]? with | some t => pure t | none => throw "missing task 0"
    let t1 ← match tasks[1]? with | some t => pure t | none => throw "missing task 1"
    let td0Base ← decodedTaskToDesc t0 t0.writes true false
    let td0Bad := { td0Base with plainWrites := td0Base.plainWrites.push { byteAddr := regOut0, value := out0Value } }
    let td1 ← decodedTaskToDesc t1 t1.writes false true
    let sIn ← makeFp16Surface .h17 "x" "input_x" .input 16 8 16
    let sOut ← makeFp16Surface .h17 "y" "output_c2" .output 16 8 16
    let some kSec := base2DramConv.info.kernelSection | throw "missing __KERN_0"
    let some scSec := base2DramConv.info.scratchSection | throw "missing __DATA,__bss"
    let kernBytes ← readSlice base2DramConv.hwx kSec.offset.toNat kSec.size.toNat
    let em ← emitProgram {
      target := .h17, surfaces := #[sIn, sOut], kernel := some kernBytes
      scratch := some { size := scSec.size, usable := scSec.size.toUInt32 }, tasks := #[td0Bad, td1]
    }
    let _ ← verifyHwxTasks em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "non-last task")

  results := results.push (checkErr "m4 neg #10c: dangling bit 22 (waitPe22) when no prior PE task exists rejected" (do
    let fCfg ← findTaskRegFileOffset base2DramConv.hwx base2DramConv.info 1 regSrc1Cfg
    let orig ← readU32LE base2DramConv.hwx fCfg
    tamperTaskRegAndVerify base2DramConv.hwx base2DramConv.info 1 regSrc1Cfg (orig ||| cfgBitWaitPe22)
  ) "bit 22")

  -- Neg #11: PE direct write with k=4, outW > 16 ([DL] violation)
  results := results.push (checkErr "m4 neg #11: PE direct write with k=4, outW=64 > 16 ([DL]) rejected" (do
    let d : TensorDims := { c := 4, h := 4, w := 64 }
    let t : PeTiling := { k := 4, tileHeight := 4 }
    let td ← lowerPeTask .h17 .abs d d none 6 none t
    let sX ← makeFp16Surface .h17 "x" "input_x" .input 4 4 64
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 4 4 64
    let em ← emitProgram { target := .h17, surfaces := #[sX, sY], tasks := #[td] }
    let _ ← verifyHwxTasks em.hwx em.info (requireKnownTask := true)
    let fResCfg ← findTaskRegFileOffset em.hwx em.info 0 regResCfg
    let origCfg ← readU32LE em.hwx fResCfg
    tamperTaskRegAndVerify em.hwx em.info 0 regResCfg (origCfg ||| ((1 : UInt32) <<< 3))
  ) "[DL]")

  -- =========================================================================
  -- Milestone 4 Silicon-Contract Fixes:
  -- 1. BAR 3 DRAM-scratch read coherence (depKind = 2 on PE 0x4d00 / 0x4d04)
  -- 2. PE P=1 DMA-fed L2 channel-row stride 64-byte DMA line floor max(T, 32)
  -- =========================================================================

  -- Fix 1a: Compiled DRAM-scratch PE src1 & src2 readers (adjacent j = i-1 and skip j < i-1)
  --         emit depKind = 2 with exact depDist = min 15 (i - j - 1) and pass verifyHwxTasks.
  let basePeScratchCoherence ← match Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .dram
      inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
      nodes := #[
        { id := "p0", inputs := #["x"], op := .pe1 .abs },
        { id := "p1", inputs := #["p0"], op := .pe1 (.chain (some 0x3800) none none) },
        { id := "add", inputs := #["p0", "p1"], op := .pe2 .add false }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"basePeScratchCoherence setup failed: {e}")

  results := results.push (checkOk "m4 fix1: BAR 3 DRAM-scratch PE src1 & src2 readers emit depKind=2 (adjacent + skip) and pass verifier" (do
    let _ ← verifyCompiledGraph basePeScratchCoherence 3
    -- Task 1 (p1): PE src1 reads p0 (j=0, i=1) -> 0x4d00 depKind=2, depDist=0
    let f1 ← findTaskRegFileOffset basePeScratchCoherence.hwx basePeScratchCoherence.info 1 regSrc1Dep
    let w1 ← readU32LE basePeScratchCoherence.hwx f1
    let dk1 ← decodeFieldChecked (srcDepMultiLayout "src1Dep") w1 "depKind"
    let dd1 ← decodeFieldChecked (srcDepMultiLayout "src1Dep") w1 "depDist"
    if dk1 != 2 || dd1 != 0 then
      throw s!"task 1 src1Dep: expected depKind=2, depDist=0; got depKind={dk1}, depDist={dd1}"
    -- Task 2 (add): PE src1 reads p0 (j=0, i=2 -> depKind=2, depDist=1); src2 reads p1 (j=1, i=2 -> depKind=2, depDist=0)
    let f2s1 ← findTaskRegFileOffset basePeScratchCoherence.hwx basePeScratchCoherence.info 2 regSrc1Dep
    let w2s1 ← readU32LE basePeScratchCoherence.hwx f2s1
    let dk2s1 ← decodeFieldChecked (srcDepMultiLayout "src1Dep") w2s1 "depKind"
    let dd2s1 ← decodeFieldChecked (srcDepMultiLayout "src1Dep") w2s1 "depDist"
    if dk2s1 != 2 || dd2s1 != 1 then
      throw s!"task 2 src1Dep: expected depKind=2, depDist=1; got depKind={dk2s1}, depDist={dd2s1}"
    let f2s2 ← findTaskRegFileOffset basePeScratchCoherence.hwx basePeScratchCoherence.info 2 regSrc2Dep
    let w2s2 ← readU32LE basePeScratchCoherence.hwx f2s2
    let dk2s2 ← decodeFieldChecked (srcDepMultiLayout "src2Dep") w2s2 "depKind"
    let dd2s2 ← decodeFieldChecked (srcDepMultiLayout "src2Dep") w2s2 "depDist"
    if dk2s2 != 2 || dd2s2 != 0 then
      throw s!"task 2 src2Dep: expected depKind=2, depDist=0; got depKind={dk2s2}, depDist={dd2s2}"
    -- Also test NE -> PE src1 & NE -> conv -> PE src2 (skip j=0, i=2) in DRAM
    let gNePe ← Vane.Graph.compileGraph {
      target := .h16, defaultPlacement := .dram
      inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
      nodes := #[
        { id := "c0", inputs := #["x"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 1) none .none },
        { id := "c1", inputs := #["c0"], op := .conv1x1 16 16 (mkGraphW1x1 16 16 2) none .none },
        { id := "add", inputs := #["c1", "c0"], op := .pe2 .add false }
      ]
    }
    let _ ← verifyCompiledGraph gNePe 3
    let fNeS1 ← findTaskRegFileOffset gNePe.hwx gNePe.info 2 regSrc1Dep
    let wNeS1 ← readU32LE gNePe.hwx fNeS1
    let dkNeS1 ← decodeFieldChecked (srcDepMultiLayout "src1Dep") wNeS1 "depKind"
    let ddNeS1 ← decodeFieldChecked (srcDepMultiLayout "src1Dep") wNeS1 "depDist"
    if dkNeS1 != 2 || ddNeS1 != 0 then
      throw s!"gNePe task 2 src1Dep: expected depKind=2, depDist=0; got depKind={dkNeS1}, depDist={ddNeS1}"
    let fNeS2 ← findTaskRegFileOffset gNePe.hwx gNePe.info 2 regSrc2Dep
    let wNeS2 ← readU32LE gNePe.hwx fNeS2
    let dkNeS2 ← decodeFieldChecked (srcDepMultiLayout "src2Dep") wNeS2 "depKind"
    let ddNeS2 ← decodeFieldChecked (srcDepMultiLayout "src2Dep") wNeS2 "depDist"
    if dkNeS2 != 2 || ddNeS2 != 1 then
      throw s!"gNePe task 2 src2Dep: expected depKind=2, depDist=1; got depKind={dkNeS2}, depDist={ddNeS2}"
    pure ()
  ))

  -- Fix 1b: Mutating BAR 3 PE src1 reader depKind to 0 is rejected by verifier (S5) even with programOrderPE HB intact
  results := results.push (checkErr "m4 fix1 neg: BAR 3 PE src1 reader depKind=0 rejected by S5 scratch coherence" (do
    let f1 ← findTaskRegFileOffset basePeScratchCoherence.hwx basePeScratchCoherence.info 1 regSrc1Dep
    let origCode := (← readU32LE basePeScratchCoherence.hwx f1) &&& 0xff
    let wBad ← mkSrcDepWord "src1Dep" origCode 1 0 0
    tamperTaskRegAndVerify basePeScratchCoherence.hwx basePeScratchCoherence.info 1 regSrc1Dep wBad
  ) "scratch coherence")

  -- Fix 1c: Mutating BAR 3 PE src2 reader depKind to 0 is rejected by verifier (S5) even with bit 22 intact
  results := results.push (checkErr "m4 fix1 neg: BAR 3 PE src2 reader depKind=0 rejected by S5 scratch coherence" (do
    let fCfg2 ← findTaskRegFileOffset basePeScratchCoherence.hwx basePeScratchCoherence.info 2 regSrc2Cfg
    let origCfg2 ← readU32LE basePeScratchCoherence.hwx fCfg2
    let h1 ← putU32LE basePeScratchCoherence.hwx fCfg2 (origCfg2 ||| cfgBitWaitPe22)
    let f2 ← findTaskRegFileOffset h1 basePeScratchCoherence.info 2 regSrc2Dep
    let origCode := (← readU32LE h1 f2) &&& 0xff
    let wBad ← mkSrcDepWord "src2Dep" origCode 2 0 0
    let h2 ← putU32LE h1 f2 wBad
    let badInfo ← validateHwx h2
    let _ ← verifyHwxTasks h2 badInfo (requireKnownTask := true)
    pure ()
  ) "scratch coherence")

  -- Fix 2a: Natural staged PE case on H16 (C=9600, H=1, W=56 -> k=5, direct=false, 0x4114=0x4128=0x4150=0x40)
  results := results.push (checkOk "m4 fix2: H16 natural staged PE (C=9600, H=1, W=56) selects k=5, direct=false, strides=0x40" (do
    let d : TensorDims := { c := 9600, h := 1, w := 56 }
    let comp ← compilePe .h16 (.binary .add d d false) d
    if comp.tiling.k != 5 then
      throw s!"expected k=5, got k={comp.tiling.k}"
    if comp.direct then
      throw "expected direct=false (staged) on H16 W=56 k=5"
    let v1 ← readU32LE comp.hwx (← findTaskRegFileOffset comp.hwx comp.info 0 regSrc1ChanStride)
    let v2 ← readU32LE comp.hwx (← findTaskRegFileOffset comp.hwx comp.info 0 regSrc2ChanStride)
    let vr ← readU32LE comp.hwx (← findTaskRegFileOffset comp.hwx comp.info 0 regResChanStride)
    if v1 != 0x40 || v2 != 0x40 || vr != 0x40 then
      throw s!"expected 0x4114=0x4128=0x4150=0x40, got ({toHex v1.toNat}, {toHex v2.toNat}, {toHex vr.toNat})"
    let _ ← verifyHwxTasksSummary comp.hwx comp.info (requireKnownTask := true)
    pure ()
  ))

  -- Fix 2b: Explicit k=4 with W=32, direct=false, 0x4114=0x4128=0x4150=0x40 passes verifier;
  --         patching any of 0x4114, 0x4128, or 0x4150 down to 0x20 is rejected (S3).
  let basePeK4W32 ← match (do
      let d : TensorDims := { c := 4, h := 4, w := 32 }
      compilePe .h16 (.binary .add d d false) d (tilingOverride := some { k := 4, tileHeight := 4 })) with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"basePeK4W32 setup failed: {e}")

  results := results.push (checkOk "m4 fix2: explicit k=4 W=32 staged PE emits 0x4114=0x4128=0x4150=0x40 and passes verifier" (do
    if basePeK4W32.tiling.k != 4 || basePeK4W32.direct then
      throw s!"expected k=4 direct=false, got k={basePeK4W32.tiling.k} direct={basePeK4W32.direct}"
    let v1 ← readU32LE basePeK4W32.hwx (← findTaskRegFileOffset basePeK4W32.hwx basePeK4W32.info 0 regSrc1ChanStride)
    let v2 ← readU32LE basePeK4W32.hwx (← findTaskRegFileOffset basePeK4W32.hwx basePeK4W32.info 0 regSrc2ChanStride)
    let vr ← readU32LE basePeK4W32.hwx (← findTaskRegFileOffset basePeK4W32.hwx basePeK4W32.info 0 regResChanStride)
    if v1 != 0x40 || v2 != 0x40 || vr != 0x40 then
      throw s!"expected 0x4114=0x4128=0x4150=0x40, got ({toHex v1.toNat}, {toHex v2.toNat}, {toHex vr.toNat})"
    let _ ← verifyHwxTasksSummary basePeK4W32.hwx basePeK4W32.info (requireKnownTask := true)
    pure ()
  ))

  results := results.push (checkErr "m4 fix2 neg: k=4 W=32 patching 0x4114 (src1ChanStride) to 0x20 rejected by S3" (do
    let h1 ← putU32LE basePeK4W32.hwx (← findTaskRegFileOffset basePeK4W32.hwx basePeK4W32.info 0 regSrc1ChanStride) 0x20
    let h2 ← putU32LE h1 (← findTaskRegFileOffset h1 basePeK4W32.info 0 regSrc1RowStride) (4 * 0x20)
    let h3 ← putU32LE h2 (← findTaskRegFileOffset h2 basePeK4W32.info 0 regSrc1RowBytesA) (4 * 0x20)
    let h4 ← putU32LE h3 (← findTaskRegFileOffset h3 basePeK4W32.info 0 regSrc1RowBytesB) (4 * 0x20)
    let badInfo ← validateHwx h4
    let _ ← verifyHwxTasks h4 badInfo (requireKnownTask := true)
    pure ()
  ) "64B DMA line floor")

  results := results.push (checkErr "m4 fix2 neg: k=4 W=32 patching 0x4128 (src2ChanStride) to 0x20 rejected by S3" (do
    let h1 ← putU32LE basePeK4W32.hwx (← findTaskRegFileOffset basePeK4W32.hwx basePeK4W32.info 0 regSrc2ChanStride) 0x20
    let h2 ← putU32LE h1 (← findTaskRegFileOffset h1 basePeK4W32.info 0 regSrc2RowStride) (4 * 0x20)
    let h3 ← putU32LE h2 (← findTaskRegFileOffset h2 basePeK4W32.info 0 regSrc2RowBytesA) (4 * 0x20)
    let h4 ← putU32LE h3 (← findTaskRegFileOffset h3 basePeK4W32.info 0 regSrc2RowBytesB) (4 * 0x20)
    let badInfo ← validateHwx h4
    let _ ← verifyHwxTasks h4 badInfo (requireKnownTask := true)
    pure ()
  ) "64B DMA line floor")

  results := results.push (checkErr "m4 fix2 neg: k=4 W=32 patching 0x4150 (resChanStride) to 0x20 rejected by S3" (do
    let h1 ← putU32LE basePeK4W32.hwx (← findTaskRegFileOffset basePeK4W32.hwx basePeK4W32.info 0 regResChanStride) 0x20
    let h2 ← putU32LE h1 (← findTaskRegFileOffset h1 basePeK4W32.info 0 regResRowStride) (4 * 0x20)
    let h3 ← putU32LE h2 (← findTaskRegFileOffset h2 basePeK4W32.info 0 regResRowBytesA) (4 * 0x20)
    let h4 ← putU32LE h3 (← findTaskRegFileOffset h3 basePeK4W32.info 0 regResRowBytesB) (4 * 0x20)
    let badInfo ← validateHwx h4
    let _ ← verifyHwxTasks h4 badInfo (requireKnownTask := true)
    pure ()
  ) "64B DMA line floor")

  pure results

def main : IO UInt32 := do
  let results := (← runAllTests) ++ (← runM4Tests)
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
