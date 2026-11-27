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
open Vane.Pool
open Vane.Conv
open Vane.Conv1x1
open Vane.Conv3x3
open Vane.View
open Vane.Matmul
open Vane.Frontend.SafeTensors
open Vane.Frontend.Onnx
open Vane.Frontend.Lower

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

  -- =========================================================================
  -- Milestone 4 Review Hardening Tests (M1 header[7] liveL2Mask, M2 uninit scratch)
  -- =========================================================================

  -- M1a: Clearing a required bit in header[7] (to 0) on a non-last L2-chained task is rejected
  results := results.push (checkErr "m4 rev [M1] neg: clearing header[7] liveL2Mask to 0 on L2-chained task rejected" (do
    tamperTaskHeaderAndVerify basePeToNe.hwx basePeToNe.info 0 7 0
  ) "header[7] liveL2Mask")

  -- M1b: Setting an extra bit in header[7] on a non-last L2-chained task is rejected
  results := results.push (checkErr "m4 rev [M1] neg: extra bit in header[7] liveL2Mask on L2-chained task rejected" (do
    let textBytes ← readSlice basePeToNe.hwx basePeToNe.info.textSection.offset.toNat basePeToNe.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let h7 ← match tasks[0]? with | some t => pure t.header[7]! | none => throw "missing task 0"
    tamperTaskHeaderAndVerify basePeToNe.hwx basePeToNe.info 0 7 (h7 ||| ((1 : UInt32) <<< 31))
  ) "header[7] liveL2Mask")

  -- M1c: Non-zero header[7] (e.g. 0xdeadbeef) on a non-last pure-DRAM task is rejected
  results := results.push (checkErr "m4 rev [M1] neg: non-zero header[7] on pure-DRAM non-last task rejected" (do
    tamperTaskHeaderAndVerify base2DramConv.hwx base2DramConv.info 0 7 0xdeadbeef
  ) "header[7] liveL2Mask")

  -- M2a: Reading uninitialized BAR 3 scratch offset (never written by prior task) with depKind=2 rejected
  results := results.push (checkErr "m4 rev [M2] neg: uninitialized BAR 3 scratch read (depKind=2) rejected" (do
    -- In base3DramConv (c1 -> c2 -> c3, all DRAM [16,8,16], slot size 0x2000):
    -- scratch size is 0x4000; task 0 writes [0, 0x2000), task 1 writes [0x2000, 0x4000).
    -- Tamper task 1's src1BaseLo from 0 to 0x2000 -> reads [0x2000, 0x4000) which task 0 never wrote!
    let fT1Res ← findTaskRegFileOffset base3DramConv.hwx base3DramConv.info 1 regResBaseLo
    let slot1Off ← readU32LE base3DramConv.hwx fT1Res
    tamperTaskRegAndVerify base3DramConv.hwx base3DramConv.info 1 regSrc1BaseLo slot1Off
  ) "uninitialized scratch")

  -- M2b: Reading uninitialized BAR 3 scratch offset with depKind=0 rejected by unconditional coherence check
  results := results.push (checkErr "m4 rev [M2] neg: uninitialized BAR 3 scratch read (depKind=0) rejected" (do
    let fT1Res ← findTaskRegFileOffset base3DramConv.hwx base3DramConv.info 1 regResBaseLo
    let slot1Off ← readU32LE base3DramConv.hwx fT1Res
    let fT1Src ← findTaskRegFileOffset base3DramConv.hwx base3DramConv.info 1 regSrc1BaseLo
    let fT1Dep ← findTaskRegFileOffset base3DramConv.hwx base3DramConv.info 1 regSrc1Dep
    let origCode := (← readU32LE base3DramConv.hwx fT1Dep) &&& 0xff
    let wNoDep ← mkSrcDepWord "src1Dep" origCode 1 0 0
    let h1 ← putU32LE base3DramConv.hwx fT1Src slot1Off
    let h2 ← putU32LE h1 fT1Dep wNoDep
    let badInfo ← validateHwx h2
    let _ ← verifyHwxTasks h2 badInfo (requireKnownTask := true)
    pure ()
  ) "scratch coherence")

  -- =========================================================================
  -- Milestone 4 neSrcStageAhead: NE DMA src1Stage fill over draining resStage
  -- =========================================================================

  let baseNeSrcStageAhead ← match Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .dram
      inputs := #[{ name := "x", shape := { c := 3, h := 32, w := 64 } }]
      nodes := #[
        { id := "c0", inputs := #["x"], op := .conv3x3 .dense 3 64 1 .same (mkGraphW3x3 3 64 false) none .relu },
        { id := "c1", inputs := #["c0"], op := .conv3x3 .dense 64 3 1 .same (mkGraphW3x3 64 3 false) none .none }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"baseNeSrcStageAhead setup failed: {e}")

  results := results.push (checkOk "m4 neSrcStageAhead: NE->NE DRAM conv3x3 (3->64->3) sets 0x4104 bit 5 and passes verifier" (do
    let s ← verifyCompiledGraph baseNeSrcStageAhead 2
    if s.neConv != 2 then
      throw s!"expected 2 conv tasks, got {s.neConv}"
    let fCfg1 ← findTaskRegFileOffset baseNeSrcStageAhead.hwx baseNeSrcStageAhead.info 1 regSrc1Cfg
    let wCfg1 ← readU32LE baseNeSrcStageAhead.hwx fCfg1
    if (wCfg1 &&& cfgBitWaitNe5) == 0 then
      throw s!"expected task 1 0x4104 to set bit 5 (0x20), got {toHex wCfg1.toNat}"
    pure ()
  ))

  results := results.push (checkErr "m4 neSrcStageAhead neg: clearing 0x4104 bit 5 on NE->NE overlapping src1Stage/resStage rejected" (do
    let fCfg1 ← findTaskRegFileOffset baseNeSrcStageAhead.hwx baseNeSrcStageAhead.info 1 regSrc1Cfg
    let orig ← readU32LE baseNeSrcStageAhead.hwx fCfg1
    let stripped := orig &&& (~~~cfgBitWaitNe5)
    tamperTaskRegAndVerify baseNeSrcStageAhead.hwx baseNeSrcStageAhead.info 1 regSrc1Cfg stripped
  ) "srcStageAhead")

  pure results

private def mkM5aW1x1 (cin cout seed : Nat) : ByteArray :=
  Vane.Graph.u16ArrayToBytes (Array.ofFn (n := cout * cin) fun i =>
    if (i.val + seed) % 3 == 0 then 0x3400 else 0x3800)

private def mkM5aW3x3 (cin cout : Nat) (isDw : Bool) : ByteArray :=
  let cinW := if isDw then 1 else cin
  Vane.Graph.u16ArrayToBytes (Array.ofFn (n := cout * cinW * 9) fun i =>
    if i.val % 5 == 0 then 0x3800 else 0x3000)

private def mkM5aBias (cout : Nat) : ByteArray :=
  Vane.Graph.u16ArrayToBytes (Array.replicate cout (0x3400 : UInt16))

private def verifyM5aGraph (g : Vane.Graph.CompiledGraph) (expTasks : Nat) : Except String TaskVerifySummary := do
  let info ← validateHwx g.hwx
  validateSectionsWithInfo info g.sections
  let s ← verifyHwxTasksSummary g.hwx info (requireKnownTask := true)
  if s.total != expTasks then
    throw s!"expected {expTasks} verified tasks, got {s.total}"
  pure s

private def findM5aTaskRegOffset (hwx : ByteArray) (info : HwxInfo) (taskIdx regAddr : Nat) : Except String Nat := do
  let textBytes ← readSlice hwx info.textSection.offset.toNat info.textSection.size.toNat
  let tasks ← decodeTaskStream info.target textBytes
  let some task := tasks[taskIdx]?
    | throw s!"missing task {taskIdx} (only {tasks.size} tasks)"
  match task.writes.find? (fun w => w.byteAddr == regAddr) with
  | some w => pure (info.textSection.offset.toNat + w.payloadOffset)
  | none => throw s!"task {taskIdx} does not write register {toHex regAddr}"

private def tamperM5aTaskReg (hwx : ByteArray) (info : HwxInfo) (taskIdx regAddr : Nat) (newVal : UInt32) :
    Except String Unit := do
  let fileOff ← findM5aTaskRegOffset hwx info taskIdx regAddr
  let badHwx ← putU32LE hwx fileOff newVal
  let badInfo ← validateHwx badHwx
  let _ ← verifyHwxTasks badHwx badInfo (requireKnownTask := true)
  pure ()

/-- Build a 6-layer camera-denoise-shaped CNN (`3 -> 32 -> 32 -> 32 -> 32 -> 32 -> 3`,
    five 3x3 stride-1 conv+ReLU layers followed by a 1x1 projection) for spatial H-strip tests. -/
private def mkDenoiseGraphSpec (target : Target) (h w : Nat) (stripMode : Vane.Graph.StripMode) :
    Vane.Graph.GraphSpec := {
  target := target
  defaultPlacement := .auto
  stripMode := stripMode
  inputs := #[{ name := "x", shape := { c := 3, h := h, w := w } }]
  nodes := #[
    { id := "c0", inputs := #["x"],  op := .conv3x3 .dense 3 32 1 .same (mkM5aW3x3 3 32 false) (some (mkM5aBias 32)) .relu },
    { id := "c1", inputs := #["c0"], op := .conv3x3 .dense 32 32 1 .same (mkM5aW3x3 32 32 false) (some (mkM5aBias 32)) .relu },
    { id := "c2", inputs := #["c1"], op := .conv3x3 .dense 32 32 1 .same (mkM5aW3x3 32 32 false) (some (mkM5aBias 32)) .relu },
    { id := "c3", inputs := #["c2"], op := .conv3x3 .dense 32 32 1 .same (mkM5aW3x3 32 32 false) (some (mkM5aBias 32)) .relu },
    { id := "c4", inputs := #["c3"], op := .conv3x3 .dense 32 32 1 .same (mkM5aW3x3 32 32 false) (some (mkM5aBias 32)) .relu },
    { id := "c5", inputs := #["c4"], op := .conv1x1 32 3 (mkM5aW1x1 32 3 7) (some (mkM5aBias 3)) .none }
  ]
}

private def runM5aTests : IO (Array TestResult) := do
  let mut results : Array TestResult := #[]

  -- 1. Positive: all 4 asymmetric vertical 3x3 strip padding modes (dense + depthwise, H16 + H17)
  results := results.push (checkOk "m5a: PadMode.strip (full/top/mid/bot) single-task conv3x3 on H16 & H17 passes verifier" (do
    let stripCases : Array (Nat × Nat × Nat × Nat) := #[
      (8, 1, 1, 8), -- full frame
      (9, 1, 0, 8), -- top strip
      (10, 0, 0, 8), -- middle strip
      (9, 0, 1, 8)  -- bottom strip
    ]
    for tgt in #[Target.h16, Target.h17] do
      for mode in #[Conv3x3Mode.dense, Conv3x3Mode.depthwise] do
        for (hIn, pt, pb, expHo) in stripCases do
          let sh : ConvSpec := {
            kernelSize := 3, mode := mode, h := hIn, w := 32, cin := 16, cout := 16,
            stride := 1, pad := .strip pt pb
          }
          if sh.outH != expHo || sh.outW != 32 || sh.padTop != pt || sh.padLeft != 1 then
            throw s!"strip({pt},{pb}) geometry mismatch: out=({sh.outH}x{sh.outW}), pad=({sh.padTop},{sh.padLeft})"
          let w := mkM5aW3x3 16 16 (mode == .depthwise)
          let c ← compileConv tgt sh w none true
          let s ← verifyHwxTasksSummary c.hwx c.info (requireKnownTask := true)
          if s.neConv != 1 then
            throw s!"expected 1 verified conv task, got {s.neConv}"
    pure ()
  ))

  -- 2. Positive: 6-layer camera-denoise CNN (VGA 240x320) on H16 & H17 with stripMode = .auto -> 0 B scratch & shared __KERN_0
  for tgt in #[Target.h16, Target.h17] do
    results := results.push (checkOk s!"m5a: {tgt.name} 6-layer denoise VGA (240x320) auto-strip achieves 0B scratch & shared __KERN_0" (do
      let unstripped ← Vane.Graph.compileGraph (mkDenoiseGraphSpec tgt 240 320 .off)
      if unstripped.scratchBytes == 0 then
        throw "expected unstripped VGA denoise CNN to spill to DRAM scratch"
      let stripped ← Vane.Graph.compileGraph (mkDenoiseGraphSpec tgt 240 320 .auto)
      let _ ← verifyM5aGraph stripped stripped.taskCount
      if stripped.stripCount <= 1 || stripped.stripHeight == 0 then
        throw s!"expected stripped VGA denoise CNN (stripCount={stripped.stripCount}, stripHeight={stripped.stripHeight})"
      if stripped.taskCount != stripped.stripCount * 6 then
        throw s!"taskCount {stripped.taskCount} != stripCount({stripped.stripCount}) * 6"
      if stripped.scratchBytes != 0 then
        throw s!"expected 0 B DRAM scratch for stripped VGA denoise CNN, got {stripped.scratchBytes}B"
      if stripped.kernelBytes != unstripped.kernelBytes then
        throw s!"expected shared __KERN_0 size {unstripped.kernelBytes}B, got {stripped.kernelBytes}B (no Sx duplication)"
      pure ()
    ))

  -- 3. Positive: 6-layer camera-denoise CNN (1080p 540x960) on H16 & H17 with stripMode = .auto -> <= 2 MiB scratch & shared __KERN_0
  for tgt in #[Target.h16, Target.h17] do
    results := results.push (checkOk s!"m5a: {tgt.name} 6-layer denoise 1080p (540x960) auto-strip <= 2 MiB scratch & shared __KERN_0" (do
      let singleStrip ← Vane.Graph.compileGraph (mkDenoiseGraphSpec tgt 16 960 .off)
      let stripped ← Vane.Graph.compileGraph (mkDenoiseGraphSpec tgt 540 960 .auto)
      let _ ← verifyM5aGraph stripped stripped.taskCount
      if stripped.stripCount <= 1 || stripped.taskCount > 255 then
        throw s!"expected stripped 1080p with 2..255 tasks, got strips={stripped.stripCount}, tasks={stripped.taskCount}"
      if stripped.scratchBytes > 2 * 1024 * 1024 then
        throw s!"expected 1080p stripped scratch <= 2 MiB, got {stripped.scratchBytes}B"
      if stripped.kernelBytes != singleStrip.kernelBytes then
        throw s!"expected shared __KERN_0 size {singleStrip.kernelBytes}B, got {stripped.kernelBytes}B"
      pure ()
    ))

  -- 4. Positive: uneven remainder strip (H % h_out != 0) + mixed NE/LUT/PE input-skip + single-strip edge case
  let unevenStripped ← match Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .auto, stripMode := .fixed 8
      inputs := #[{ name := "x", shape := { c := 8, h := 25, w := 32 } }]
      nodes := #[
        { id := "c0", inputs := #["x"], op := .conv3x3 .dense 8 8 1 .same (mkM5aW3x3 8 8 false) none .relu },
        { id := "l0", inputs := #["c0"], op := .lut (.lut .silu) },
        { id := "p0", inputs := #["l0"], op := .pe1 (.chain (some 0x3800) none none) },
        { id := "res", inputs := #["p0", "x"], op := .pe2 .add false }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"unevenStripped setup failed: {e}")

  results := results.push (checkOk "m5a: uneven remainder strip (H=25, h_out=8 -> 4 strips) mixed conv3x3/lut/pe+input-skip passes S1-S5 & shares __KERN_0" (do
    let single ← Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .auto, stripMode := .fixed 25
      inputs := #[{ name := "x", shape := { c := 8, h := 25, w := 32 } }]
      nodes := #[
        { id := "c0", inputs := #["x"], op := .conv3x3 .dense 8 8 1 .same (mkM5aW3x3 8 8 false) none .relu },
        { id := "l0", inputs := #["c0"], op := .lut (.lut .silu) },
        { id := "p0", inputs := #["l0"], op := .pe1 (.chain (some 0x3800) none none) },
        { id := "res", inputs := #["p0", "x"], op := .pe2 .add false }
      ]
    }
    let _ ← verifyM5aGraph single 4
    if single.stripCount != 1 || single.stripHeight != 25 then
      throw s!"expected single-strip edge case stripCount=1 stripHeight=25, got count={single.stripCount} h={single.stripHeight}"
    -- [N1] Oversized fixed strip height (> H_full) clamps reported stripHeight to H_full
    let clamped ← Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .auto, stripMode := .fixed 100
      inputs := #[{ name := "x", shape := { c := 8, h := 25, w := 32 } }]
      nodes := #[
        { id := "c0", inputs := #["x"], op := .conv3x3 .dense 8 8 1 .same (mkM5aW3x3 8 8 false) none .relu }
      ]
    }
    let _ ← verifyM5aGraph clamped 1
    if clamped.stripCount != 1 || clamped.stripHeight != 25 then
      throw s!"expected clamped stripCount=1 stripHeight=25 for fixed 100 on H=25, got count={clamped.stripCount} h={clamped.stripHeight}"
    let _ ← verifyM5aGraph unevenStripped 16
    if unevenStripped.stripCount != 4 || unevenStripped.stripHeight != 8 then
      throw s!"expected 4 strips of height 8, got strips={unevenStripped.stripCount}, h={unevenStripped.stripHeight}"
    if unevenStripped.kernelBytes != single.kernelBytes then
      throw s!"expected shared __KERN_0 (conv+lut) {single.kernelBytes}B, got {unevenStripped.kernelBytes}B"
    -- Final remainder strip (strip 3, task 15) must have output height 1 (25 - 3*8)
    if unevenStripped.nodes[15]!.outDims.h != 1 then
      throw s!"expected remainder strip height 1, got {unevenStripped.nodes[15]!.outDims.h}"
    pure ()
  ))

  -- 5. Negative tamper: illegal strip padding pb = 2 (Hin=5, pt=0, Hout=5) rejected by verifier
  let baseStripPadConv ← match compileConv .h17 {
      kernelSize := 3, mode := .dense, h := 5, w := 32, cin := 4, cout := 4, stride := 1, pad := .strip 1 1
    } (mkM5aW3x3 4 4 false) none false with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseStripPadConv setup failed: {e}")

  results := results.push (checkErr "m5a neg: illegal strip padding pb=2 (Hin=5, pt=0, Hout=5) rejected by verifier" (do
    let fCfg ← findM5aTaskRegOffset baseStripPadConv.hwx baseStripPadConv.info 0 regConvCfg
    let origCfg ← readU32LE baseStripPadConv.hwx fCfg
    -- Clear pt bits [27:22] from 1 to 0 while keeping Hin=5, Hout=5 -> implies pb = 5+2-5-0 = 2
    let badCfg := origCfg &&& (~~~((0x3f : UInt32) <<< 22))
    tamperM5aTaskReg baseStripPadConv.hwx baseStripPadConv.info 0 regConvCfg badCfg
  ) "3x3 spatial geometry mismatch")

  -- 6. Negative tamper: illegal strip padding pb < 0 (Hin=5, pt=1, Hout=3) rejected by verifier
  results := results.push (checkErr "m5a neg: illegal strip padding pb<0 (Hin=5, pt=1, Hout=3) rejected by verifier" (do
    let fHo ← findM5aTaskRegOffset baseStripPadConv.hwx baseStripPadConv.info 0 regHout
    let fTh ← findM5aTaskRegOffset baseStripPadConv.hwx baseStripPadConv.info 0 regTH
    let h1 ← putU32LE baseStripPadConv.hwx fHo 3
    let h2 ← putU32LE h1 fTh 3
    let badInfo ← validateHwx h2
    let _ ← verifyHwxTasks h2 badInfo (requireKnownTask := true)
    pure ()
  ) "3x3 spatial geometry mismatch")

  -- 7. Negative tamper: surface DMA row offset causing inter-channel plane bleed rejected by verifier
  let base2StripConv ← match Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .auto, stripMode := .fixed 4
      inputs := #[{ name := "x", shape := { c := 4, h := 16, w := 32 } }]
      nodes := #[
        { id := "c0", inputs := #["x"], op := .conv1x1 4 4 (mkM5aW1x1 4 4 1) none .none }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"base2StripConv setup failed: {e}")

  results := results.push (checkErr "m5a neg: sub-height surface DMA offset causing plane bleed rejected by verifier" (do
    -- Surface is C=4, H_full=16, W=32 -> rowStride=64, span=4096.
    -- Task 0 reads h=4 rows; shrink planeStride to 4*64=256 and set src1BaseLo=64:
    -- baseOff + extent = 64 + 3*256 + 256 = 1088 <= 4096 (fits in surface capacity),
    -- yet baseOff + h*rowStride = 64 + 256 = 320 > 256 = planeStride (inter-channel plane bleed!).
    let fBase ← findM5aTaskRegOffset base2StripConv.hwx base2StripConv.info 0 regSrc1BaseLo
    let fPlane ← findM5aTaskRegOffset base2StripConv.hwx base2StripConv.info 0 regSrc1DmaPlaneStride
    let h1 ← putU32LE base2StripConv.hwx fPlane (4 * 64)
    let h2 ← putU32LE h1 fBase 64
    let badInfo ← validateHwx h2
    let _ ← verifyHwxTasks h2 badInfo (requireKnownTask := true)
    pure ()
  ) "plane bleed")

  -- 8. Negative tamper: non-last strip setting 0x5500 kernelStage isLast=1 rejected by verifier
  results := results.push (checkErr "m5a neg: non-last strip output-surface write setting 0x5500 isLast=1 rejected by verifier" (do
    let fKs ← findM5aTaskRegOffset base2StripConv.hwx base2StripConv.info 0 regKernelStage
    let origKs ← readU32LE base2StripConv.hwx fKs
    tamperM5aTaskReg base2StripConv.hwx base2StripConv.info 0 regKernelStage (origKs ||| ((1 : UInt32) <<< 7))
  ) "kernelStage isLast=1 != expected 0")

  -- 9. Negative tamper: non-last strip emitting 0x5900 output stage on multi-strip output-surface write rejected
  results := results.push (checkErr "m5a neg: non-last strip output-surface write emitting 0x5900 output stage rejected by verifier" (do
    let textBytes ← readSlice base2StripConv.hwx base2StripConv.info.textSection.offset.toNat base2StripConv.info.textSection.size.toNat
    let decoded ← decodeTaskStream .h17 textBytes
    let t0 := decoded[0]!
    let t1 := decoded[1]!
    let rebuildTask := fun (t : DecodedTask) (extraPlain : Array RegWrite) => do
      let ene ← decodeFieldChecked taskHeader8MultiLayout t.header[8]! "ene"
      let mut plain : Array RegWrite := #[]
      let mut bars : Array BarRegWrite := #[]
      let mut kRelocs : Array TaskReloc := #[]
      let mut i := 0
      while i < t.writes.size do
        let w := t.writes[i]!
        match w.barSlot with
        | none =>
          plain := plain.push { byteAddr := w.byteAddr, value := w.value }
          i := i + 1
        | some slot =>
          if i + 1 < t.writes.size && t.writes[i + 1]!.barSlot == some slot && t.writes[i + 1]!.byteAddr == w.byteAddr + 4 then
            let bIdx := bars.size
            bars := bars.push { baseByteAddr := w.byteAddr, barSlot := slot, values := #[w.value, t.writes[i + 1]!.value] }
            if w.byteAddr == regKernelBaseLo then
              kRelocs := kRelocs.push { barRecIndex := bIdx, payloadWordIndex := 0 }
            i := i + 2
          else
            bars := bars.push { baseByteAddr := w.byteAddr, barSlot := slot, values := #[w.value] }
            i := i + 1
      pure ({
        taskId := t.taskIndex
        logNE := 0
        eneOverride := some ene.toNat
        isFirst := t.taskIndex == 0
        isLast := t.taskIndex == 1
        liveL2Mask := t.header[7]!
        plainWrites := plain ++ extraPlain
        barWrites := bars
        kernelRelocs := kRelocs
      } : TaskDesc)
    let badOutStage : Array RegWrite := #[
      { byteAddr := regOut0, value := out0Value },
      { byteAddr := regOut1, value := out1Value },
      { byteAddr := regOut2, value := (out2FormValue <<< 24) },
      { byteAddr := regOut3, value := out3Value },
      { byteAddr := regOut4, value := out4Value },
      { byteAddr := regOut5, value := out5Value }
    ]
    let kSec := base2StripConv.info.kernelSection.get!
    let kBytes ← readSlice base2StripConv.hwx kSec.offset.toNat kSec.size.toNat
    let inSurf ← makeFp16Surface .h17 "x" "input_x" .input 4 16 32
    let outSurf ← makeFp16Surface .h17 "y" "output_c0" .output 4 16 32
    let td0 ← rebuildTask t0 badOutStage
    let td1 ← rebuildTask t1 #[]
    let em ← emitProgram {
      target := .h17
      surfaces := #[inSurf, outSurf]
      tasks := #[td0, td1]
      kernel := some kBytes
      scratch := none
      constants := zeroBytes textConstBytes
      ident := some "vane"
    }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "non-last task must omit out0")

  -- 10. Negative compiler checks: over-limit (>255 tasks) and ineligible graphs (stride-2, skip-halo mismatch, broadcast pe2)
  results := results.push (checkErr "m5a neg: fixed strip height exceeding 255 kernel-bearing tasks rejected" (do
    let _ ← Vane.Graph.compileGraph (mkDenoiseGraphSpec .h17 64 32 (.fixed 1))
    pure ()
  ) "exceeding limit 255")

  results := results.push (checkErr "m5a neg: fixed strip height on stride-2 conv3x3 graph rejected" (do
    let _ ← Vane.Graph.compileGraph {
      target := .h17, stripMode := .fixed 4
      inputs := #[{ name := "x", shape := { c := 8, h := 16, w := 16 } }]
      nodes := #[
        { id := "c0", inputs := #["x"], op := .conv3x3 .dense 8 8 2 .same (mkM5aW3x3 8 8 false) none .none }
      ]
    }
    pure ()
  ) "requires stride=1 and pad=same")

  results := results.push (checkErr "m5a neg: fixed strip height on skip-halo mismatch graph rejected" (do
    let _ ← Vane.Graph.compileGraph {
      target := .h17, stripMode := .fixed 4
      inputs := #[{ name := "x", shape := { c := 8, h := 16, w := 16 } }]
      nodes := #[
        { id := "c0", inputs := #["x"], op := .conv3x3 .dense 8 8 1 .same (mkM5aW3x3 8 8 false) none .relu },
        { id := "c1", inputs := #["c0"], op := .conv3x3 .dense 8 8 1 .same (mkM5aW3x3 8 8 false) none .relu },
        { id := "add", inputs := #["c1", "c0"], op := .pe2 .add false }
      ]
    }
    pure ()
  ) "skip-halo interval mismatch")

  results := results.push (checkErr "m5a neg: fixed strip height on broadcast pe2 graph rejected" (do
    let _ ← Vane.Graph.compileGraph {
      target := .h17, stripMode := .fixed 4
      inputs := #[
        { name := "x", shape := { c := 8, h := 16, w := 16 } },
        { name := "b", shape := { c := 8, h := 1, w := 1 } }
      ]
      nodes := #[
        { id := "add", inputs := #["x", "b"], op := .pe2 .add false }
      ]
    }
    pure ()
  ) "spatial strip fusion ineligible")

  -- 11. [M2] Negative unit tests for validateConvShape / validateConvSafety on illegal PadMode.strip specs
  let dummyTiling : Tiling := {
    logNE := 4, f := 4, fat := 0, ocg := 0, shPref := 0, shMax := 0, tileHeight := 1
  }
  results := results.push (checkErr "m5a [M2] neg: PadMode.strip(1,0) with stride=2 rejected by validateConvShape" (do
    validateConvShape { kernelSize := 3, mode := .dense, h := 8, w := 32, cin := 4, cout := 4, stride := 2, pad := .strip 1 0 }
  ) "requires stride=1")

  results := results.push (checkErr "m5a [M2] neg: PadMode.strip(1,0) with stride=2 rejected by validateConvSafety" (do
    validateConvSafety .h17 { kernelSize := 3, mode := .dense, h := 8, w := 32, cin := 4, cout := 4, stride := 2, pad := .strip 1 0 } dummyTiling
  ) "requires stride=1")

  results := results.push (checkErr "m5a [M2] neg: PadMode.strip(2,0) rejected by validateConvShape" (do
    validateConvShape { kernelSize := 3, mode := .dense, h := 8, w := 32, cin := 4, cout := 4, stride := 1, pad := .strip 2 0 }
  ) "requires pt <= 1 and pb <= 1")

  results := results.push (checkErr "m5a [M2] neg: PadMode.strip(0,2) rejected by validateConvSafety" (do
    validateConvSafety .h17 { kernelSize := 3, mode := .dense, h := 8, w := 32, cin := 4, cout := 4, stride := 1, pad := .strip 0 2 } dummyTiling
  ) "requires pt <= 1 and pb <= 1")

  results := results.push (checkErr "m5a [M2] neg: PadMode.strip(1,0) with h=1 (h+pt+pb=2 < 3) rejected by validateConvShape" (do
    validateConvShape { kernelSize := 3, mode := .dense, h := 1, w := 32, cin := 4, cout := 4, stride := 1, pad := .strip 1 0 }
  ) "requires H + pt + pb >= 3")

  results := results.push (checkErr "m5a [M2] neg: PadMode.strip(0,0) with h=2 (h+pt+pb=2 < 3) rejected by validateConvSafety" (do
    validateConvSafety .h17 { kernelSize := 3, mode := .dense, h := 2, w := 32, cin := 4, cout := 4, stride := 1, pad := .strip 0 0 } dummyTiling
  ) "requires H + pt + pb >= 3")

  -- 12. [M2] Negative tamper: single-channel (C=1) sub-height surface window with baseOff + extent > cap rejected
  let baseC1StripConv ← match Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .auto, stripMode := .fixed 4
      inputs := #[{ name := "x", shape := { c := 1, h := 16, w := 32 } }]
      nodes := #[
        { id := "c0", inputs := #["x"], op := .conv1x1 1 1 (mkM5aW1x1 1 1 3) none .none }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"baseC1StripConv setup failed: {e}")

  results := results.push (checkErr "m5a [M2] neg: C=1 sub-height surface window baseOff pushed past remaining rows (baseOff+extent > cap) rejected" (do
    -- Surface is C=1, H_full=16, W=32 -> rowStride=64, span=cap=1024.
    -- Task 0 reads h=4 rows (extent=256); setting src1BaseLo = 13*64 = 832 gives 832 + 256 = 1088 > 1024.
    tamperM5aTaskReg baseC1StripConv.hwx baseC1StripConv.info 0 regSrc1BaseLo (13 * 64)
  ) "exceeds")

  pure results

-- ===========================================================================
-- Milestone 5B Tests: Pooling, Reductions, L2 PE2 Broadcast, Verifier Tampers
-- ===========================================================================

private def runM5bTests : IO (Array TestResult) := do
  let mut results : Array TestResult := #[]

  -- 1. Positive: PE window pooling (H16 & H17) across max/avg, excl=true/false, valid/same/custom, multi-strip, H16 staged
  results := results.push (checkOk "m5b: pePool H16 & H17 (max/avg, excl, valid/same/custom, multi-strip, H16 staged)" (do
    let cases : Array (PoolKind × TensorDims × Nat × Nat × Nat × Nat × PoolPad × Bool) := #[
      (.max, { c := 16, h := 16, w := 16 }, 2, 2, 2, 2, .valid, true),
      (.avg, { c := 16, h := 16, w := 16 }, 3, 3, 1, 1, .same, true),
      (.avg, { c := 16, h := 16, w := 16 }, 3, 3, 1, 1, .same, false),
      (.max, { c := 8, h := 13, w := 17 }, 5, 5, 2, 2, .custom 2 1 1 2, true),
      (.max, { c := 8, h := 12, w := 160 }, 2, 2, 2, 2, .valid, true), -- multi-strip Ow=80
      (.avg, { c := 4, h := 2, w := 2 }, 2, 2, 2, 2, .valid, true)     -- sEff=1 edge case
    ]
    for tgt in #[Target.h16, Target.h17] do
      for (op, dims, kh, kw, sh, sw, pad, excl) in cases do
        let c ← compilePool { target := tgt, op := op, dims := dims, kh := kh, kw := kw, sh := sh, sw := sw, pad := pad, excludePad := excl }
        if c.backend != .pePool then
          throw s!"expected pePool backend, got {c.backend.name}"
        let s ← verifyHwxTasksSummary c.hwx c.info (requireKnownTask := true)
        if s.pePool != 1 then
          throw s!"expected 1 pePool task, got {repr s}"
    -- H16 staged multi-strip pePool: large C forces k=5 (T=32) with Ow=56 > 32 -> direct=false on H16
    let stagedH16 ← compilePool {
      target := .h16, op := .max, dims := { c := 5120, h := 2, w := 112 },
      kh := 2, kw := 2, sh := 2, sw := 2, pad := .valid
    }
    if stagedH16.direct then
      throw "expected H16 multi-strip k=5 pePool to use staged mode (direct=false)"
    let sStg ← verifyHwxTasksSummary stagedH16.hwx stagedH16.info (requireKnownTask := true)
    if sStg.pePool != 1 then
      throw s!"expected 1 verified H16 staged pePool task, got {repr sStg}"
    pure ()
  ))

  -- 2. Positive: PE tensor reductions (H16 & H17) across all 10 ops, all 7 axis subsets, multi-strip/multi-tile, sD > 1
  results := results.push (checkOk "m5b: peReduce H16 & H17 (all 10 ops, all 7 axis subsets, multi-strip/multi-tile, sD>1)" (do
    let allOps : Array ReduceOp := #[
      .sum, .mean, .max, .min, .l1, .sumSquare, .l2, .invSum, .rsqrtSumSquare, .rsqrtMeanSquare
    ]
    for tgt in #[Target.h16, Target.h17] do
      for op in allOps do
        let eps : UInt16 := if op.usesEps then floatToF16CanonicalZero 0.001 else 0
        let c ← compileReduce { target := tgt, op := op, dims := { c := 8, h := 6, w := 32 }, axes := { reduceW := true }, eps := eps }
        if c.backend != .peReduce || c.outDims != { c := 8, h := 6, w := 1 } then
          throw s!"peReduce {op.name} unexpected backend/outDims"
        let s ← verifyHwxTasksSummary c.hwx c.info (requireKnownTask := true)
        if s.peReduce != 1 then
          throw s!"expected 1 peReduce task for {op.name}"
      -- All 7 axis subsets (W-containing on W=16, and W=1 for h/c/ch)
      let axisCases : Array (TensorDims × ReduceAxes × TensorDims) := #[
        ({ c := 1, h := 6, w := 16 }, { reduceW := true }, { c := 1, h := 6, w := 1 }),   -- [W] C=1
        ({ c := 4, h := 6, w := 16 }, { reduceW := true }, { c := 4, h := 6, w := 1 }),   -- [W] C>1
        ({ c := 4, h := 6, w := 16 }, { reduceH := true, reduceW := true }, { c := 4, h := 1, w := 1 }),
        ({ c := 4, h := 6, w := 16 }, { reduceC := true, reduceW := true }, { c := 1, h := 6, w := 1 }),
        ({ c := 8, h := 4, w := 16 }, { reduceC := true, reduceH := true, reduceW := true }, { c := 1, h := 1, w := 1 }), -- C > H (sD > 1)
        ({ c := 4, h := 8, w := 16 }, { reduceC := true, reduceH := true, reduceW := true }, { c := 1, h := 1, w := 1 }), -- C <= H (sD > 1)
        ({ c := 4, h := 6, w := 1 },  { reduceH := true }, { c := 4, h := 1, w := 1 }),
        ({ c := 4, h := 6, w := 1 },  { reduceC := true }, { c := 1, h := 6, w := 1 }),
        ({ c := 4, h := 6, w := 1 },  { reduceC := true, reduceH := true }, { c := 1, h := 1, w := 1 })
      ]
      for (dims, axes, expOut) in axisCases do
        let c ← compileReduce { target := tgt, op := .mean, dims := dims, axes := axes }
        if c.outDims != expOut then
          throw s!"peReduce axes={axes.name} outDims={repr c.outDims} != {repr expOut}"
        let _ ← verifyHwxTasksSummary c.hwx c.info (requireKnownTask := true)
      -- Multi-strip + multi-tile (W=512, C=64, H=64 -> exceeds 2 MiB single tile -> th < sH)
      let big ← compileReduce { target := tgt, op := .sumSquare, dims := { c := 64, h := 64, w := 512 }, axes := { reduceH := true, reduceW := true } }
      let _ ← verifyHwxTasksSummary big.hwx big.info (requireKnownTask := true)
    pure ()
  ))

  -- 3. Positive: Weightless NE window pooling (7x7, 15x15, max/avg, s1/s2) + global pool fast-path
  results := results.push (checkOk "m5b: nePool H16 & H17 (7x7, 15x15, max/avg, s1/s2) + global pool fast-path" (do
    let cases : Array (PoolKind × TensorDims × Nat × Nat × Nat × Nat × PoolPad × Bool) := #[
      (.max, { c := 16, h := 16, w := 16 }, 7, 7, 1, 1, .same, false),
      (.avg, { c := 16, h := 16, w := 16 }, 7, 7, 2, 2, .same, false),
      (.max, { c := 8, h := 32, w := 32 }, 15, 15, 2, 2, .valid, false),
      (.avg, { c := 8, h := 32, w := 32 }, 15, 15, 2, 2, .same, false),
      (.avg, { c := 8, h := 24, w := 24 }, 10, 10, 1, 1, .valid, true) -- even k=10 s1 valid
    ]
    for tgt in #[Target.h16, Target.h17] do
      for (op, dims, kh, kw, sh, sw, pad, excl) in cases do
        let c ← compilePool { target := tgt, op := op, dims := dims, kh := kh, kw := kw, sh := sh, sw := sw, pad := pad, excludePad := excl }
        if c.backend != .nePool then
          throw s!"expected nePool backend, got {c.backend.name}"
        if c.info.kernelSection.isSome then
          throw "weightless nePool must have zero __KERN_0"
        let s ← verifyHwxTasksSummary c.hwx c.info (requireKnownTask := true)
        if s.nePool != 1 then
          throw s!"expected 1 nePool task, got {repr s}"
      -- Global pool fast-path: 8x8 valid on 8x8 input lowers to peReduce [H, W]
      let gap ← compilePool { target := tgt, op := .avg, dims := { c := 32, h := 8, w := 8 }, kh := 8, kw := 8, sh := 1, sw := 1, pad := .valid }
      if gap.backend != .peReduceGlobal || gap.outDims != { c := 32, h := 1, w := 1 } then
        throw s!"expected peReduceGlobal 32x1x1, got {gap.backend.name} {repr gap.outDims}"
      let sGap ← verifyHwxTasksSummary gap.hwx gap.info (requireKnownTask := true)
      if sGap.peReduce != 1 then
        throw s!"expected 1 peReduce task for global avg_pool, got {repr sGap}"
    pure ()
  ))

  -- 4. Positive: Weightless NE channel reduction (max/min/sum/mean, odd C, multi-strip)
  results := results.push (checkOk "m5b: neChannelReduce H16 & H17 (max/min/sum/mean, odd C=17, multi-strip W=160)" (do
    for tgt in #[Target.h16, Target.h17] do
      for op in #[ReduceOp.max, .min, .sum, .mean] do
        for dims in #[({ c := 17, h := 8, w := 23 } : TensorDims), { c := 64, h := 12, w := 160 }] do
          let c ← compileReduce { target := tgt, op := op, dims := dims, axes := { reduceC := true } }
          if c.backend != .neChannelReduce || c.outDims != { c := 1, h := dims.h, w := dims.w } then
            throw s!"neChannelReduce {op.name} unexpected backend/outDims"
          if c.info.kernelSection.isSome then
            throw "weightless neChannelReduce must have zero __KERN_0"
          let s ← verifyHwxTasksSummary c.hwx c.info (requireKnownTask := true)
          if s.neChannelReduce != 1 then
            throw s!"expected 1 neChannelReduce task, got {repr s}"
    pure ()
  ))

  -- 5. Positive: 100% L2-resident DAGs (rms_norm, layer_norm, softmax, channel-softmax prefix, ResNet block) on H16 & H17
  for tgt in #[Target.h16, Target.h17] do
    results := results.push (checkOk s!"m5b: {tgt.name} 100% L2-resident rms_norm, layer_norm, softmax, channel-softmax & ResNet DAGs" (do
      -- (a) RMSNorm along W (0 B scratch, 0 B kernel)
      let rms ← Vane.Graph.compileGraph {
        target := tgt, defaultPlacement := .auto
        inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 64 } }]
        nodes := #[
          { id := "inv_rms", inputs := #["x"], op := .reduce .rsqrtMeanSquare { reduceW := true } (floatToF16CanonicalZero 0.001) },
          { id := "out", inputs := #["x", "inv_rms"], op := .pe2 .mul false }
        ]
      }
      let sRms ← verifyM5aGraph rms 2
      if rms.scratchBytes != 0 || rms.kernelBytes != 0 || sRms.peReduce != 1 || sRms.peOp != 1 then
        throw s!"rms_norm unexpected scratch={rms.scratchBytes} kern={rms.kernelBytes} summary={repr sRms}"

      -- (b) LayerNorm along W (4 tasks, 0 B scratch, 0 B kernel)
      let ln ← Vane.Graph.compileGraph {
        target := tgt, defaultPlacement := .auto
        inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 64 } }]
        nodes := #[
          { id := "mu", inputs := #["x"], op := .reduce .mean { reduceW := true } },
          { id := "xc", inputs := #["x", "mu"], op := .pe2 .sub false },
          { id := "inv_std", inputs := #["xc"], op := .reduce .rsqrtMeanSquare { reduceW := true } (floatToF16CanonicalZero 0.001) },
          { id := "out", inputs := #["xc", "inv_std"], op := .pe2 .mul false }
        ]
      }
      let sLn ← verifyM5aGraph ln 4
      if ln.scratchBytes != 0 || ln.kernelBytes != 0 || sLn.peReduce != 2 || sLn.peOp != 2 then
        throw s!"layer_norm unexpected scratch={ln.scratchBytes} kern={ln.kernelBytes} summary={repr sLn}"

      -- (c) Softmax along W (6 tasks, 0 B scratch)
      let sm ← Vane.Graph.compileGraph {
        target := tgt, defaultPlacement := .auto
        inputs := #[{ name := "x", shape := { c := 8, h := 8, w := 64 } }]
        nodes := #[
          { id := "mx", inputs := #["x"], op := .reduce .max { reduceW := true } },
          { id := "shifted", inputs := #["x", "mx"], op := .pe2 .sub false },
          { id := "scaled", inputs := #["shifted"], op := .pe1 (PeOp.scalarMul fp16InvLn2Bits) },
          { id := "ex", inputs := #["scaled"], op := .lut (.lut .exp2) },
          { id := "inv_s", inputs := #["ex"], op := .reduce .invSum { reduceW := true } },
          { id := "out", inputs := #["ex", "inv_s"], op := .pe2 .mul false }
        ]
      }
      let sSm ← verifyM5aGraph sm 6
      if sm.scratchBytes != 0 || sSm.peReduce != 2 || sSm.peOp != 3 || sSm.neAct != 1 then
        throw s!"softmax unexpected scratch={sm.scratchBytes} summary={repr sSm}"

      -- (d) Channel-axis softmax prefix (neChannelReduce -> pe2 C-broadcast sub -> smul -> lut exp2 -> neChannelReduce sum, 0 B scratch)
      let csm ← Vane.Graph.compileGraph {
        target := tgt, defaultPlacement := .auto
        inputs := #[{ name := "x", shape := { c := 32, h := 8, w := 32 } }]
        nodes := #[
          { id := "mx", inputs := #["x"], op := .reduce .max { reduceC := true } },
          { id := "shifted", inputs := #["x", "mx"], op := .pe2 .sub false },
          { id := "scaled", inputs := #["shifted"], op := .pe1 (PeOp.scalarMul fp16InvLn2Bits) },
          { id := "ex", inputs := #["scaled"], op := .lut (.lut .exp2) },
          { id := "sum_ex", inputs := #["ex"], op := .reduce .sum { reduceC := true } }
        ]
      }
      let sCsm ← verifyM5aGraph csm 5
      if csm.scratchBytes != 0 || sCsm.neChannelReduce != 2 || sCsm.peOp != 2 || sCsm.neAct != 1 then
        throw s!"channel-softmax unexpected scratch={csm.scratchBytes} summary={repr sCsm}"

      -- (e) ResNet block: conv3x3 -> max_pool 3x3 s2 -> conv1x1 + skip -> global avg_pool -> conv1x1 head
      let resnet ← Vane.Graph.compileGraph {
        target := tgt, defaultPlacement := .auto
        inputs := #[{ name := "x", shape := { c := 16, h := 16, w := 16 } }]
        nodes := #[
          { id := "stem", inputs := #["x"], op := .conv3x3 .dense 16 16 1 .same (mkM5aW3x3 16 16 false) (some (mkM5aBias 16)) .relu },
          { id := "pool", inputs := #["stem"], op := .pool .max 3 3 2 2 .same true },
          { id := "proj", inputs := #["pool"], op := .conv1x1 16 16 (mkM5aW1x1 16 16 3) (some (mkM5aBias 16)) .relu },
          { id := "skip", inputs := #["proj", "pool"], op := .pe2 .add false },
          { id := "gap", inputs := #["skip"], op := .pool .avg 8 8 1 1 .valid true },
          { id := "head", inputs := #["gap"], op := .conv1x1 16 8 (mkM5aW1x1 16 8 5) (some (mkM5aBias 8)) .none }
        ]
      }
      let sRes ← verifyM5aGraph resnet 6
      if resnet.scratchBytes != 0 || sRes.neConv != 3 || sRes.pePool != 1 || sRes.peOp != 1 || sRes.peReduce != 1 then
        throw s!"resnet unexpected scratch={resnet.scratchBytes} summary={repr sRes}"
      pure ()
    ))

  -- 6. Positive: DRAM fallback when L2-resident PE2 broadcast cannot fit at th == out.h (auto-spill)
  results := results.push (checkOk "m5b: auto-spill fallback to DRAM when L2 PE2 broadcast cannot fit th == out.h" (do
    -- x is 128x256x64 (4 MiB); inv_rms is 128x256x1 (512 KiB, fits in L2 alone),
    -- but at th == out.h = 256 even k=5 staging of x needs 2 MiB -> exceeds L2 -> spills inv_rms to DRAM!
    let bigRms ← Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .auto
      inputs := #[{ name := "x", shape := { c := 128, h := 256, w := 64 } }]
      nodes := #[
        { id := "inv_rms", inputs := #["x"], op := .reduce .rsqrtMeanSquare { reduceW := true } (floatToF16CanonicalZero 0.001) },
        { id := "out", inputs := #["x", "inv_rms"], op := .pe2 .mul false }
      ]
    }
    let _ ← verifyM5aGraph bigRms 2
    if bigRms.scratchBytes == 0 then
      throw "expected auto-spill to DRAM when full tensor + broadcast cannot fit th == out.h in 2 MiB L2"
    pure ()
  ))

  -- 7. Negative compiler checks: StripMode rejects graphs containing pool or reduce; unsupported reduce/pool configs rejected
  results := results.push (checkErr "m5b neg: StripMode.fixed rejects graph containing pool" (do
    let _ ← Vane.Graph.compileGraph {
      target := .h17, stripMode := .fixed 4
      inputs := #[{ name := "x", shape := { c := 8, h := 16, w := 16 } }]
      nodes := #[{ id := "p", inputs := #["x"], op := .pool .max 2 2 2 2 .valid true }]
    }
    pure ()
  ) "pool operation is not supported")

  results := results.push (checkErr "m5b neg: StripMode.fixed rejects graph containing reduce" (do
    let _ ← Vane.Graph.compileGraph {
      target := .h17, stripMode := .fixed 4
      inputs := #[{ name := "x", shape := { c := 8, h := 16, w := 16 } }]
      nodes := #[{ id := "r", inputs := #["x"], op := .reduce .mean { reduceW := true } }]
    }
    pure ()
  ) "reduce operation is not supported")

  results := results.push (checkErr "m5b neg: compiler rejects NE stride-1 max_pool k=9" (do
    compilePool { target := .h17, op := .max, dims := { c := 8, h := 16, w := 16 }, kh := 9, kw := 9, sh := 1, sw := 1, pad := .same }
  ) "stride-1 max_pool")

  results := results.push (checkErr "m5b neg: compiler rejects NE stride-1 avg_pool odd k=9" (do
    compilePool { target := .h17, op := .avg, dims := { c := 8, h := 16, w := 16 }, kh := 9, kw := 9, sh := 1, sw := 1, pad := .same, excludePad := false }
  ) "stride-1 avg_pool")

  results := results.push (checkErr "m5b neg: compiler rejects reduce over H alone when W > 1" (do
    compileReduce { target := .h17, op := .sum, dims := { c := 8, h := 8, w := 16 }, axes := { reduceH := true } }
  ) "transpose")

  -- 8. Negative static-verifier tamper tests (per m5b.md §2.8)
  let basePePool ← match compilePool {
      target := .h17, op := .max, dims := { c := 8, h := 16, w := 16 }, kh := 3, kw := 3, sh := 2, sw := 2, pad := .same
    } with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"basePePool setup failed: {e}")

  results := results.push (checkErr "m5b neg tamper: PE pool kh=6 (> 5) rejected by verifier" (do
    let fCfg ← findM5aTaskRegOffset basePePool.hwx basePePool.info 0 regConvCfg
    let origCfg ← readU32LE basePePool.hwx fCfg
    -- Replace kh[12:6] with 6
    let badCfg := (origCfg &&& (~~~((0x7f : UInt32) <<< 6))) ||| ((6 : UInt32) <<< 6)
    tamperM5aTaskReg basePePool.hwx basePePool.info 0 regConvCfg badCfg
  ) "must be in 1..5")

  results := results.push (checkErr "m5b neg tamper: PE pool illegal 0x4500 opcode rejected by verifier" (do
    tamperM5aTaskReg basePePool.hwx basePePool.info 0 regPeOp 0x00000005
  ) "illegal PE pool 0x4500")

  results := results.push (checkErr "m5b neg tamper: PE pool 0x4100 cmpFlags mismatch rejected by verifier" (do
    tamperM5aTaskReg basePePool.hwx basePePool.info 0 regCmpFlags cmpFlagMinValue
  ) "cmpFlags")

  results := results.push (checkErr "m5b neg tamper: PE pool L2 overlap (resBase = src1Base) rejected by verifier" (do
    tamperM5aTaskReg basePePool.hwx basePePool.info 0 regResBase 0
  ) "overlapping L2")

  let baseNeMaxPool ← match compilePool {
      target := .h17, op := .max, dims := { c := 8, h := 24, w := 24 }, kh := 7, kw := 7, sh := 1, sw := 1, pad := .same, excludePad := false
    } with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseNeMaxPool setup failed: {e}")

  results := results.push (checkErr "m5b neg tamper: NE pool max_pool s1 k=9 rejected by verifier" (do
    let fCfg ← findM5aTaskRegOffset baseNeMaxPool.hwx baseNeMaxPool.info 0 regConvCfg
    let origCfg ← readU32LE baseNeMaxPool.hwx fCfg
    let badCfg := (origCfg &&& (~~~((0x7f : UInt32) <<< 6))) ||| ((9 : UInt32) <<< 6)
    tamperM5aTaskReg baseNeMaxPool.hwx baseNeMaxPool.info 0 regConvCfg badCfg
  ) "stride-1 max_pool unsupported")

  results := results.push (checkErr "m5b neg tamper: NE pool kh=16 (> 15) rejected by verifier" (do
    let fCfg ← findM5aTaskRegOffset baseNeMaxPool.hwx baseNeMaxPool.info 0 regConvCfg
    let origCfg ← readU32LE baseNeMaxPool.hwx fCfg
    let badCfg := (origCfg &&& (~~~((0x7f : UInt32) <<< 6))) ||| ((16 : UInt32) <<< 6)
    tamperM5aTaskReg baseNeMaxPool.hwx baseNeMaxPool.info 0 regConvCfg badCfg
  ) "must be in 1..15")

  let baseNeAvgPool ← match compilePool {
      target := .h17, op := .avg, dims := { c := 8, h := 24, w := 24 }, kh := 7, kw := 7, sh := 1, sw := 1, pad := .same, excludePad := false
    } with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseNeAvgPool setup failed: {e}")

  results := results.push (checkErr "m5b neg tamper: NE pool avg_pool s1 odd k=9 rejected by verifier" (do
    let fCfg ← findM5aTaskRegOffset baseNeAvgPool.hwx baseNeAvgPool.info 0 regConvCfg
    let origCfg ← readU32LE baseNeAvgPool.hwx fCfg
    let badCfg := (origCfg &&& (~~~((0x7f : UInt32) <<< 6))) ||| ((9 : UInt32) <<< 6)
    tamperM5aTaskReg baseNeAvgPool.hwx baseNeAvgPool.info 0 regConvCfg badCfg
  ) "stride-1 avg_pool unsupported")

  let basePeRed ← match compileReduce {
      target := .h17, op := .max, dims := { c := 8, h := 6, w := 32 }, axes := { reduceH := true, reduceW := true }
    } with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"basePeRed setup failed: {e}")

  results := results.push (checkErr "m5b neg tamper: PE reduce mismatched Wout != Win rejected by verifier" (do
    tamperM5aTaskReg basePeRed.hwx basePeRed.info 0 regWout 16
  ) "!= input dims")

  results := results.push (checkErr "m5b neg tamper: PE reduce illegal 0x4500 opcode rejected by verifier" (do
    tamperM5aTaskReg basePeRed.hwx basePeRed.info 0 regPeOp 0x00000001
  ) "illegal PE reduce 0x4500")

  results := results.push (checkErr "m5b neg tamper: PE reduce 0x4100 mismatch rejected by verifier" (do
    tamperM5aTaskReg basePeRed.hwx basePeRed.info 0 regCmpFlags cmpFlagMinValue
  ) "cmpFlags")

  results := results.push (checkErr "m5b neg tamper: PE reduce th > sH rejected by verifier" (do
    tamperM5aTaskReg basePeRed.hwx basePeRed.info 0 regTH 7
  ) "tileHeight=7 out of range")

  let baseNeChanRed ← match compileReduce {
      target := .h17, op := .max, dims := { c := 16, h := 8, w := 32 }, axes := { reduceC := true }
    } with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseNeChanRed setup failed: {e}")

  results := results.push (checkErr "m5b neg tamper: NE channel reduce Cout != 1 rejected by verifier" (do
    tamperM5aTaskReg baseNeChanRed.hwx baseNeChanRed.info 0 regCout 2
  ) "requires Cout=1")

  results := results.push (checkErr "m5b neg tamper: NE channel reduce 0x0030 != (Cin << 16) rejected by verifier" (do
    tamperM5aTaskReg baseNeChanRed.hwx baseNeChanRed.info 0 regDwGroups ((17 : UInt32) <<< 16)
  ) "chanCount=17 != Cin=16")

  results := results.push (checkErr "m5b neg tamper: NE channel reduce unknown 0x4904 rejected by verifier" (do
    tamperM5aTaskReg baseNeChanRed.hwx baseNeChanRed.info 0 regNeOp 0x00103c0f
  ) "unknown NE channel reduce neOp")

  -- Reject L2-resident PE2 broadcast consumer when th < outH
  let baseL2Bcast ← match Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .auto
      inputs := #[{ name := "x", shape := { c := 8, h := 8, w := 32 } }]
      nodes := #[
        { id := "mu", inputs := #["x"], op := .reduce .mean { reduceW := true } },
        { id := "xc", inputs := #["x", "mu"], op := .pe2 .sub false }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"baseL2Bcast setup failed: {e}")

  results := results.push (checkErr "m5b neg tamper: L2-resident PE2 broadcast consumer with th < outH rejected by verifier" (do
    let fTh ← findM5aTaskRegOffset baseL2Bcast.hwx baseL2Bcast.info 1 regTH
    let fTile ← findM5aTaskRegOffset baseL2Bcast.hwx baseL2Bcast.info 1 regPeTile
    let origTile ← readU32LE baseL2Bcast.hwx fTile
    let k := (origTile &&& 0xf).toNat
    let newTh := 4
    let newM := min (8 - k) (ceilLog2 newTh)
    let newTile : UInt32 := (k.toUInt32) ||| ((newM.toUInt32) <<< 4)
    let h1 ← putU32LE baseL2Bcast.hwx fTh newTh.toUInt32
    let h2 ← putU32LE h1 fTile newTile
    let badInfo ← validateHwx h2
    let _ ← verifyHwxTasks h2 badInfo (requireKnownTask := true)
    pure ()
  ) "L2-resident PE2 broadcast")

  -- 9. Silicon-discovered DRAM-scratch sync rules: general PE/NE srcStageAhead & multi-batch (BatchN > 1) depKind=1
  let baseLnDram ← match Vane.Graph.compileGraph {
      target := .h16, defaultPlacement := .dram
      inputs := #[{ name := "x", shape := { c := 8, h := 6, w := 32 } }]
      nodes := #[
        { id := "mu", inputs := #["x"], op := .reduce .mean { reduceW := true } },
        { id := "xc", inputs := #["x", "mu"], op := .pe2 .sub false },
        { id := "inv_std", inputs := #["xc"], op := .reduce .rsqrtMeanSquare { reduceW := true } (floatToF16CanonicalZero 0.001) },
        { id := "out", inputs := #["xc", "inv_std"], op := .pe2 .mul false }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"baseLnDram setup failed: {e}")

  let baseRedToNeDram ← match Vane.Graph.compileGraph {
      target := .h16, defaultPlacement := .dram
      inputs := #[{ name := "x", shape := { c := 8, h := 6, w := 32 } }]
      nodes := #[
        { id := "mu", inputs := #["x"], op := .reduce .mean { reduceW := true } },
        { id := "proj", inputs := #["mu"], op := .conv1x1 8 8 (mkM5aW1x1 8 8 1) none .none }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"baseRedToNeDram setup failed: {e}")

  let baseRedToPe1Dram ← match Vane.Graph.compileGraph {
      target := .h16, defaultPlacement := .dram
      inputs := #[{ name := "x", shape := { c := 8, h := 6, w := 32 } }]
      nodes := #[
        { id := "mu", inputs := #["x"], op := .reduce .mean { reduceW := true } },
        { id := "scaled", inputs := #["mu"], op := .pe1 (PeOp.scalarMul 0x3800) }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"baseRedToPe1Dram setup failed: {e}")

  results := results.push (checkOk "m5b: LayerNorm(axis=W) DRAM + peReduce(w)->PE/NE emit depKind=1 (BatchN>1) & waitPe22 (srcStageAhead)" (do
    for tgt in #[Target.h16, Target.h17] do
      let lnDram ← Vane.Graph.compileGraph {
        target := tgt, defaultPlacement := .dram
        inputs := #[{ name := "x", shape := { c := 8, h := 6, w := 32 } }]
        nodes := #[
          { id := "mu", inputs := #["x"], op := .reduce .mean { reduceW := true } },
          { id := "xc", inputs := #["x", "mu"], op := .pe2 .sub false },
          { id := "inv_std", inputs := #["xc"], op := .reduce .rsqrtMeanSquare { reduceW := true } (floatToF16CanonicalZero 0.001) },
          { id := "out", inputs := #["xc", "inv_std"], op := .pe2 .mul false }
        ]
      }
      let s ← verifyM5aGraph lnDram 4
      if s.peReduce != 2 || s.peOp != 2 || lnDram.scratchBytes == 0 then
        throw s!"{tgt.name} lnDram unexpected summary={repr s} scratch={lnDram.scratchBytes}"
      -- Task 1 (xc): src2Dep reads task 0 (BatchN=6 > 1) -> depKind=1; src1Cfg sets bit 22 (srcStageAhead)
      let w1s2 ← readU32LE lnDram.hwx (← findM5aTaskRegOffset lnDram.hwx lnDram.info 1 regSrc2Dep)
      let dk1s2 ← decodeFieldChecked (srcDepMultiLayout "src2Dep") w1s2 "depKind"
      if dk1s2 != 1 then
        throw s!"{tgt.name} lnDram task 1 src2Dep depKind={dk1s2} != 1"
      let w1c1 ← readU32LE lnDram.hwx (← findM5aTaskRegOffset lnDram.hwx lnDram.info 1 regSrc1Cfg)
      if (w1c1 &&& cfgBitWaitPe22) == 0 then
        throw s!"{tgt.name} lnDram task 1 src1Cfg {toHex w1c1.toNat} missing bit 22 (waitPe22 for srcStageAhead)"
      -- Task 2 (inv_std): multi-batch (BatchN=6 > 1) consumer reading scratch -> depKind=1
      let w2s1 ← readU32LE lnDram.hwx (← findM5aTaskRegOffset lnDram.hwx lnDram.info 2 regSrc1Dep)
      let dk2s1 ← decodeFieldChecked (srcDepMultiLayout "src1Dep") w2s1 "depKind"
      if dk2s1 != 1 then
        throw s!"{tgt.name} lnDram task 2 src1Dep depKind={dk2s1} != 1"
      -- Task 3 (out): src1Dep reads task 1 (BatchN=1) -> depKind=2; src2Dep reads task 2 (BatchN=6 > 1) -> depKind=1; src1Cfg sets bit 22
      let w3s1 ← readU32LE lnDram.hwx (← findM5aTaskRegOffset lnDram.hwx lnDram.info 3 regSrc1Dep)
      let dk3s1 ← decodeFieldChecked (srcDepMultiLayout "src1Dep") w3s1 "depKind"
      if dk3s1 != 2 then
        throw s!"{tgt.name} lnDram task 3 src1Dep depKind={dk3s1} != 2"
      let w3s2 ← readU32LE lnDram.hwx (← findM5aTaskRegOffset lnDram.hwx lnDram.info 3 regSrc2Dep)
      let dk3s2 ← decodeFieldChecked (srcDepMultiLayout "src2Dep") w3s2 "depKind"
      if dk3s2 != 1 then
        throw s!"{tgt.name} lnDram task 3 src2Dep depKind={dk3s2} != 1"
      let w3c1 ← readU32LE lnDram.hwx (← findM5aTaskRegOffset lnDram.hwx lnDram.info 3 regSrc1Cfg)
      if (w3c1 &&& cfgBitWaitPe22) == 0 then
        throw s!"{tgt.name} lnDram task 3 src1Cfg {toHex w3c1.toNat} missing bit 22 (waitPe22 for srcStageAhead)"
    -- Also verify NE consumer (peReduce(w) -> DRAM -> neConv1x1) emits depKind=1, depDist=0
    let _ ← verifyM5aGraph baseRedToNeDram 2
    let wNe ← readU32LE baseRedToNeDram.hwx (← findM5aTaskRegOffset baseRedToNeDram.hwx baseRedToNeDram.info 1 regSrc1Dep)
    let dkNe ← decodeFieldChecked (srcDepMultiLayout "src1Dep") wNe "depKind"
    let ddNe ← decodeFieldChecked (srcDepMultiLayout "src1Dep") wNe "depDist"
    if dkNe != 1 || ddNe != 0 then
      throw s!"peReduce(w)->neConv1x1 DRAM task 1 src1Dep expected (depKind=1, depDist=0), got ({dkNe}, {ddNe})"
    -- Also verify PE src1 consumer (peReduce(w) -> DRAM -> pe1) emits depKind=1
    let _ ← verifyM5aGraph baseRedToPe1Dram 2
    let wPe1 ← readU32LE baseRedToPe1Dram.hwx (← findM5aTaskRegOffset baseRedToPe1Dram.hwx baseRedToPe1Dram.info 1 regSrc1Dep)
    let dkPe1 ← decodeFieldChecked (srcDepMultiLayout "src1Dep") wPe1 "depKind"
    if dkPe1 != 1 then
      throw s!"peReduce(w)->pe1 DRAM task 1 src1Dep expected depKind=1, got {dkPe1}"
    pure ()
  ))

  results := results.push (checkErr "m5b neg tamper: clearing PE src1Cfg bit 22 when src1Stage overlaps prior PE resStage rejected ([srcStageAhead])" (do
    let fCfg1 ← findM5aTaskRegOffset baseLnDram.hwx baseLnDram.info 1 regSrc1Cfg
    let origCfg1 ← readU32LE baseLnDram.hwx fCfg1
    let stripped := origCfg1 &&& (~~~cfgBitWaitPe22)
    tamperM5aTaskReg baseLnDram.hwx baseLnDram.info 1 regSrc1Cfg stripped
  ) "[srcStageAhead]")

  results := results.push (checkErr "m5b neg tamper: PE src2Dep reading BatchN>1 peReduce scratch with depKind=2 rejected ([multiBatchRaw])" (do
    let fDep ← findM5aTaskRegOffset baseLnDram.hwx baseLnDram.info 1 regSrc2Dep
    let origDep ← readU32LE baseLnDram.hwx fDep
    let badDep := (origDep &&& (~~~((0x3 : UInt32) <<< 28))) ||| ((2 : UInt32) <<< 28)
    tamperM5aTaskReg baseLnDram.hwx baseLnDram.info 1 regSrc2Dep badDep
  ) "[multiBatchRaw]")

  results := results.push (checkErr "m5b neg tamper: NE src1Dep reading BatchN>1 peReduce scratch with depKind=2 rejected ([multiBatchRaw])" (do
    let fDep ← findM5aTaskRegOffset baseRedToNeDram.hwx baseRedToNeDram.info 1 regSrc1Dep
    let origDep ← readU32LE baseRedToNeDram.hwx fDep
    let badDep := (origDep &&& (~~~((0x3 : UInt32) <<< 28))) ||| ((2 : UInt32) <<< 28)
    tamperM5aTaskReg baseRedToNeDram.hwx baseRedToNeDram.info 1 regSrc1Dep badDep
  ) "[multiBatchRaw]")

  results := results.push (checkErr "m5b neg tamper: PE src1Dep reading BatchN>1 peReduce scratch with depKind=2 rejected ([multiBatchRaw])" (do
    let fDep ← findM5aTaskRegOffset baseRedToPe1Dram.hwx baseRedToPe1Dram.info 1 regSrc1Dep
    let origDep ← readU32LE baseRedToPe1Dram.hwx fDep
    let badDep := (origDep &&& (~~~((0x3 : UInt32) <<< 28))) ||| ((2 : UInt32) <<< 28)
    tamperM5aTaskReg baseRedToPe1Dram.hwx baseRedToPe1Dram.info 1 regSrc1Dep badDep
  ) "[multiBatchRaw]")

  -- 10. [M2] Additional negative verifier tamper tests (H16 staged PE pool direct=1, peReduce 4D DMA OOB, L2 PE2 broadcast handoff)
  let baseStagedPePoolH16 ← match compilePool {
      target := .h16, op := .max, dims := { c := 5120, h := 2, w := 112 },
      kh := 2, kw := 2, sh := 2, sw := 2, pad := .valid
    } with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseStagedPePoolH16 setup failed: {e}")

  results := results.push (checkErr "m5b [M2] neg tamper: H16 multi-strip k=5 pePool resCfg/resDep direct=1 rejected by verifier" (do
    let fCfg ← findM5aTaskRegOffset baseStagedPePoolH16.hwx baseStagedPePoolH16.info 0 regResCfg
    let fDep ← findM5aTaskRegOffset baseStagedPePoolH16.hwx baseStagedPePoolH16.info 0 regResDep
    let origCfg ← readU32LE baseStagedPePoolH16.hwx fCfg
    let origDep ← readU32LE baseStagedPePoolH16.hwx fDep
    let h1 ← putU32LE baseStagedPePoolH16.hwx fCfg (origCfg ||| ((1 : UInt32) <<< 3))
    let h2 ← putU32LE h1 fDep (origDep ||| ((1 : UInt32) <<< 24))
    let badInfo ← validateHwx h2
    let _ ← verifyHwxTasks h2 badInfo (requireKnownTask := true)
    pure ()
  ) "PE pool direct")

  let basePeRed4dChw ← match compileReduce {
      target := .h17, op := .sum, dims := { c := 8, h := 4, w := 32 },
      axes := { reduceC := true, reduceH := true, reduceW := true }
    } with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"basePeRed4dChw setup failed: {e}")

  let basePeRed4dW ← match compileReduce {
      target := .h17, op := .mean, dims := { c := 8, h := 6, w := 32 },
      axes := { reduceW := true }
    } with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"basePeRed4dW setup failed: {e}")

  results := results.push (checkErr "m5b [M2] neg tamper: peReduce input 4D DMA stride (sD>1 0x4d20) inflated past input surface capacity rejected" (do
    let fD ← findM5aTaskRegOffset basePeRed4dChw.hwx basePeRed4dChw.info 0 regSrc1DmaDStride
    let origD ← readU32LE basePeRed4dChw.hwx fD
    tamperM5aTaskReg basePeRed4dChw.hwx basePeRed4dChw.info 0 regSrc1DmaDStride (origD + 0x10000)
  ) "exceeds")

  results := results.push (checkErr "m5b [M2] neg tamper: peReduce output 4D DMA stride (sN>1 0x511c) inflated past output surface capacity rejected" (do
    let fN ← findM5aTaskRegOffset basePeRed4dW.hwx basePeRed4dW.info 0 regResDmaNStride
    let origN ← readU32LE basePeRed4dW.hwx fN
    tamperM5aTaskReg basePeRed4dW.hwx basePeRed4dW.info 0 regResDmaNStride (origN + 0x10000)
  ) "exceeds")

  results := results.push (checkErr "m5b [M2] neg tamper: L2-resident PE2 broadcast src2Cfg setting L2 handoff bit rejected by verifier" (do
    let fCfg2 ← findM5aTaskRegOffset baseL2Bcast.hwx baseL2Bcast.info 1 regSrc2Cfg
    let origCfg2 ← readU32LE baseL2Bcast.hwx fCfg2
    tamperM5aTaskReg baseL2Bcast.hwx baseL2Bcast.info 1 regSrc2Cfg (origCfg2 ||| cfgBitHandoffPe)
  ) "must use whole-task wait, not L2 handoff")

  pure results

/-- Re-validate an emitted program from its raw bytes (header, LOAD sections, static
    verifier) and check the verified task kinds. -/
private def matrixVerify (tgt : Target) (hwx : ByteArray) (secs : LoadSections)
    (expected : TaskVerifySummary) : Except String Unit := do
  let info ← validateHwx hwx
  if info.target != tgt then
    throw s!"HWX target {info.target.name} != {tgt.name}"
  validateSectionsWithInfo info secs
  validateSections hwx secs
  let s ← verifyHwxTasksSummary hwx info (requireKnownTask := true)
  if s != expected then
    throw s!"verifier summary {repr s} != expected {repr expected}"

/-- Cross-target smoke matrix: representative programs compiled and independently
    verified for every target in `Target.all`, plus a tamper rejection per target. -/
private def runTargetMatrixTests : IO (Array TestResult) := do
  let mut results : Array TestResult := #[]

  results := results.push (checkOk "matrix: Target.all is non-empty with distinct names and CPU subtypes" (do
    if Target.all.isEmpty then
      throw "Target.all is empty"
    for t in Target.all do
      let p := t.profile
      if p.name != t.name then
        throw s!"profile.name mismatch for {t.name}"
      if p.cpuSubtype != t.cpuSubtype then
        throw s!"profile.cpuSubtype mismatch for {t.name}"
      if Target.parse? p.name != some t then
        throw s!"Target.parse? does not round-trip {p.name}"
      if Target.ofCpuSubtype? p.cpuSubtype != some t then
        throw s!"Target.ofCpuSubtype? does not round-trip {p.name}"
      if !p.isValidPrologue p.defaultPrologue then
        throw s!"defaultPrologue {p.defaultPrologue} invalid for {p.name}"
      if !p.isValidPrologue 1 then
        throw s!"prologue 1 must always be valid for {p.name}"
      if p.isValidPrologue 2 then
        throw s!"prologue 2 must be invalid for {p.name}"
      if p.isValidPrologue 0 != p.allowZeroPrologue then
        throw s!"isValidPrologue 0 disagrees with allowZeroPrologue for {p.name}"
      if p.dense3x3Stride1MaxOcg > 4 then
        throw s!"dense3x3Stride1MaxOcg {p.dense3x3Stride1MaxOcg} > 4 for {p.name}"
      if p.poolMultiStripMinW == 0 then
        throw s!"poolMultiStripMinW must be > 0 for {p.name}"
      if p.abiVersion == 0 || p.chipFamily == 0 then
        throw s!"invalid abiVersion/chipFamily for {p.name}"
      if (p.neCfgValue &&& ((1 : UInt32) <<< 8)) != 0 then
        throw s!"neCfgValue bit 8 (sparse) must be 0 for {p.name}"
      if (p.taskHeader4Base &&& taskHeader4FirstBit) != 0 || (p.taskHeader4EdgeBits &&& taskHeader4FirstBit) != 0 then
        throw s!"taskHeader4 base/edge overlaps firstBit for {p.name}"
    if (Target.all.map Target.name).eraseDups.length != Target.all.length then
      throw "duplicate target names in Target.all"
    -- Concrete H16 vs H17 profile helper checks
    let p16 := Target.h16.profile
    let p17 := Target.h17.profile
    if p16.chipFamily != 16 || p16.abiVersion != 3 || p16.cpuSubtype != 7 then
      throw "H16 identity metadata mismatch"
    if p17.chipFamily != 17 || p17.abiVersion != 3 || p17.cpuSubtype != 9 then
      throw "H17 identity metadata mismatch"
    if p16.directThreshold 1 2 != 32 || p17.directThreshold 1 2 != 64 then
      throw s!"directThreshold mismatch: h16={p16.directThreshold 1 2}, h17={p17.directThreshold 1 2}"
    if p16.conv1x1MaxCout 4 4 != 256 || p17.conv1x1MaxCout 4 4 != 512 then
      throw s!"conv1x1MaxCout mismatch: h16={p16.conv1x1MaxCout 4 4}, h17={p17.conv1x1MaxCout 4 4}"
    if p16.supportsWide3x3 || !p17.supportsWide3x3 then
      throw "supportsWide3x3 mismatch"
    if !p16.replicateKernelChunkSlots || p17.replicateKernelChunkSlots then
      throw "replicateKernelChunkSlots mismatch"
    if !p16.peForbidK5LastStrip24 || p17.peForbidK5LastStrip24 then
      throw "peForbidK5LastStrip24 mismatch"
    if p16.poolMultiStripMinW != 64 || p17.poolMultiStripMinW != 32 then
      throw "poolMultiStripMinW mismatch"
  ))

  for tgt in Target.all do
    results := results.push (checkOk s!"matrix ({tgt.name}): conv1x1 bias+relu compiles and verifies" (do
      let sh : Conv1x1Shape := { h := 8, w := 16, cin := 16, cout := 16 }
      let c ← compileConv1x1 tgt sh (mkTaggedWeights 16 16) (some (mkTaggedBias 16)) true
      matrixVerify tgt c.hwx c.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }
    ))

    results := results.push (checkOk s!"matrix ({tgt.name}): conv3x3 dense s=2 and depthwise compile and verify" (do
      let dense : Conv3x3Shape := { mode := .dense, h := 11, w := 13, cin := 5, cout := 9, stride := 2, pad := .same }
      let cd ← compileConv3x3 tgt dense (mkTaggedWeights3x3 9 5 (withZeros := true)) (some (mkTaggedBias 9)) true
      matrixVerify tgt cd.hwx cd.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }
      let dw : Conv3x3Shape := { mode := .depthwise, h := 10, w := 12, cin := 24, cout := 24, stride := 1, pad := .same }
      let cw ← compileConv3x3 tgt dw (mkTaggedWeights3x3 24 1) (some (mkTaggedBias 24)) false
      matrixVerify tgt cw.hwx cw.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }
    ))

    results := results.push (checkOk s!"matrix ({tgt.name}): NE LUT activations compile and verify" (do
      for act in #[NeActSpec.lut .tanh, .lut .silu, .lut (.rsqrt 0.0625)] do
        let c ← compileNeAct tgt act 8 16 24
        matrixVerify tgt c.hwx c.sections { total := 1, neConv := 0, neAct := 1, peOp := 0 }
    ))

    results := results.push (checkOk s!"matrix ({tgt.name}): PE unary and binary ops compile and verify" (do
      let d : TensorDims := { c := 4, h := 8, w := 16 }
      for op in #[PeOp.abs, PeOp.scalarMul (floatToF16 (-0.5)), .binary .add d d false] do
        let c ← compilePe tgt op d
        matrixVerify tgt c.hwx c.sections { total := 1, neConv := 0, neAct := 0, peOp := 1 }
    ))

    results := results.push (checkOk s!"matrix ({tgt.name}): multi-task graph conv -> pe1 -> lut -> conv -> pe2 verifies" (do
      let g ← Vane.Graph.compileGraph {
        target := tgt
        defaultPlacement := .auto
        inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
        nodes := #[
          { id := "c1", inputs := #["x"], op := .conv1x1 16 16 (mkM5aW1x1 16 16 3) (some (mkM5aBias 16)) .relu },
          { id := "p1", inputs := #["c1"], op := .pe1 .abs },
          { id := "l1", inputs := #["p1"], op := .lut (.lut .silu) },
          { id := "c2", inputs := #["l1"], op := .conv1x1 16 16 (mkM5aW1x1 16 16 4) none .none },
          { id := "add", inputs := #["c2", "c1"], op := .pe2 .add false }
        ]
      }
      if g.taskCount != 5 then
        throw s!"expected 5 tasks, got {g.taskCount}"
      matrixVerify tgt g.hwx g.sections { total := 5, neConv := 2, neAct := 1, peOp := 2 }
    ))

    let baseConv ← match compileConv1x1 tgt { h := 8, w := 16, cin := 16, cout := 16 }
        (mkTaggedWeights 16 16) (some (mkTaggedBias 16)) true with
      | .ok c => pure c
      | .error e => throw (IO.userError s!"matrix ({tgt.name}) base conv setup failed: {e}")
    results := results.push (checkErr s!"matrix ({tgt.name}) neg: tampered L2 resBase past 2 MiB rejected by verifier" (do
      tamperRegAndVerify baseConv regResBase 0x001fffe0
    ) "exceeds 2 MiB")

  return results

-- ===========================================================================
-- Milestone 5C Tests: Views, Large-Channel Output-Split Conv1x1/Linear, S5 Multi-Writer
-- ===========================================================================

private def rebuildTaskWithExtraWrites
    (t : DecodedTask) (totalTasks : Nat) (extraPlain : Array RegWrite) : Except String TaskDesc := do
  let ene ← decodeFieldChecked taskHeader8MultiLayout t.header[8]! "ene"
  let mut plain : Array RegWrite := #[]
  let mut bars : Array BarRegWrite := #[]
  let mut kRelocs : Array TaskReloc := #[]
  let mut i := 0
  while i < t.writes.size do
    let w := t.writes[i]!
    match w.barSlot with
    | none =>
      plain := plain.push { byteAddr := w.byteAddr, value := w.value }
      i := i + 1
    | some slot =>
      if i + 1 < t.writes.size && t.writes[i + 1]!.barSlot == some slot && t.writes[i + 1]!.byteAddr == w.byteAddr + 4 then
        let bIdx := bars.size
        bars := bars.push { baseByteAddr := w.byteAddr, barSlot := slot, values := #[w.value, t.writes[i + 1]!.value] }
        if w.byteAddr == regKernelBaseLo then
          kRelocs := kRelocs.push { barRecIndex := bIdx, payloadWordIndex := 0 }
        i := i + 2
      else
        bars := bars.push { baseByteAddr := w.byteAddr, barSlot := slot, values := #[w.value] }
        i := i + 1
  pure {
    taskId := t.taskIndex
    logNE := 0
    eneOverride := some ene.toNat
    isFirst := t.taskIndex == 0
    isLast := t.taskIndex + 1 == totalTasks
    liveL2Mask := t.header[7]!
    plainWrites := plain ++ extraPlain
    barWrites := bars
    kernelRelocs := kRelocs
  }

private def runM5cTests : IO (Array TestResult) := do
  let mut results : Array TestResult := #[]

  -- 1. Positive: transpose W-preserving (C<->H, N<->H, N<->C) on H16 & H17
  results := results.push (checkOk "m5c: transpose W-preserving (C<->H, N<->H, N<->C) on H16 & H17" (do
    let cases : Array (Shape4D × Array Nat) := #[
      ({ n := 1, c := 16, h := 8, w := 32 }, #[0, 2, 1, 3]),
      ({ n := 1, c := 13, h := 7, w := 19 }, #[0, 2, 1, 3]),
      ({ n := 2, c := 8, h := 6, w := 32 }, #[2, 1, 0, 3]),
      ({ n := 2, c := 8, h := 6, w := 32 }, #[1, 0, 2, 3]),
      ({ n := 1, c := 8, h := 8, w := 16 }, #[0, 1, 2, 3])
    ]
    for tgt in #[Target.h16, Target.h17] do
      for (sh, perm) in cases do
        let vp ← planTranspose sh perm
        let c ← compileView tgt vp
        matrixVerify tgt c.hwx c.sections { total := 1, neConv := 0, neAct := 0, peOp := 0, neView := 1 }
    pure ()
  ))

  -- 2. Positive: transpose T-mode (C<->W, c <= min(w, 128)) across direct & staged shapes checking F=4, logNE=4, wu=0, c <= 2^(F+ocg), shMax >= coverMin on H16 & H17
  results := results.push (checkOk "m5c: transpose T-mode C<->W (direct c=4..128 & staged c=12 w=16, F=4/logNE=4/WU=0/coverMin) on H16 & H17" (do
    let cases : Array (Shape4D × Array Nat × Bool) := #[
      ({ n := 1, c := 4,   h := 4, w := 32  }, #[0, 3, 2, 1], true),  -- groupCode=1, direct=true
      ({ n := 1, c := 12,  h := 4, w := 16  }, #[0, 3, 2, 1], false), -- groupCode=1, 9<=c<=16 & w<32 -> staged direct=false
      ({ n := 1, c := 16,  h := 4, w := 32  }, #[0, 3, 2, 1], true),  -- groupCode=1, direct=true
      ({ n := 1, c := 24,  h := 4, w := 32  }, #[0, 3, 2, 1], true),  -- groupCode=2, wideGroup=1
      ({ n := 1, c := 32,  h := 4, w := 32  }, #[0, 3, 2, 1], true),  -- groupCode=2, square 32x32
      ({ n := 1, c := 48,  h := 4, w := 64  }, #[0, 3, 2, 1], true),  -- groupCode=4, wideGroup=1
      ({ n := 1, c := 64,  h := 4, w := 64  }, #[0, 3, 2, 1], true),  -- groupCode=4, square 64x64
      ({ n := 1, c := 80,  h := 2, w := 96  }, #[0, 3, 2, 1], true),  -- groupCode=8, ocg=3 (must classify as neView!)
      ({ n := 1, c := 128, h := 2, w := 128 }, #[0, 3, 2, 1], true),  -- groupCode=8, max 128-channel T-mode
      ({ n := 2, c := 16,  h := 4, w := 32  }, #[0, 3, 2, 1], true)   -- 4D batch N=2 T-mode
    ]
    for tgt in #[Target.h16, Target.h17] do
      for (sh, perm, expDirect) in cases do
        let vp ← planTranspose sh perm
        let c ← compileView tgt vp
        let tp := c.taskPlans[0]!
        let t := tp.tiling
        if t.f != 4 || t.logNE != 4 || t.fat != 0 || t.wu != 0 then
          throw s!"{tgt.name} T-mode C={sh.c} W={sh.w}: expected F=4,logNE=4,fat=0,WU=0, got F={t.f},logNE={t.logNE},fat={t.fat},WU={t.wu}"
        if sh.c > ((1 : Nat) <<< (t.f + t.ocg)) then
          throw s!"{tgt.name} T-mode C={sh.c}: violates C <= 2^(F+ocg)={((1 : Nat) <<< (t.f + t.ocg))}"
        let minSh := coverMin sh.w (stripWidth t) t.fat
        if t.shMax < minSh then
          throw s!"{tgt.name} T-mode C={sh.c} W={sh.w}: shMax={t.shMax} < coverMin={minSh}"
        if tp.direct != expDirect then
          throw s!"{tgt.name} T-mode C={sh.c} W={sh.w}: direct={tp.direct} != expected {expDirect}"
        matrixVerify tgt c.hwx c.sections { total := 1, neConv := 0, neAct := 0, peOp := 0, neView := 1 }
    pure ()
  ))

  -- 3. Positive: slice (N/C/H/32-aligned W + strided H) on H16 & H17
  results := results.push (checkOk "m5c: slice (N, C, H, 32-aligned W, strided H) on H16 & H17" (do
    let cases : Array (Shape4D × Array Nat × Array Nat × Array Nat) := #[
      ({ n := 1, c := 32, h := 8, w := 32 }, #[0, 8, 0, 0],  #[1, 24, 8, 32], #[1, 1, 1, 1]),
      ({ n := 1, c := 16, h := 16, w := 32 }, #[0, 0, 4, 0], #[1, 16, 12, 32], #[1, 1, 1, 1]),
      ({ n := 1, c := 8,  h := 8, w := 96 }, #[0, 0, 0, 32], #[1, 8, 8, 64],  #[1, 1, 1, 1]),
      ({ n := 1, c := 16, h := 16, w := 32 }, #[0, 0, 1, 0], #[1, 16, 15, 32], #[1, 1, 2, 1]),
      ({ n := 4, c := 8,  h := 6, w := 32 }, #[1, 0, 0, 0],  #[3, 8, 6, 32],  #[1, 1, 1, 1])
    ]
    for tgt in #[Target.h16, Target.h17] do
      for (sh, beg, en, st) in cases do
        let vp ← planSlice sh beg en st
        let c ← compileView tgt vp
        matrixVerify tgt c.hwx c.sections { total := 1, neConv := 0, neAct := 0, peOp := 0, neView := 1 }
    pure ()
  ))

  -- 4. Positive: concat (C 2-input & 3-input, H, 32-aligned W) on H16 & H17
  results := results.push (checkOk "m5c: concat (C 2/3-input, H, 32-aligned W) on H16 & H17" (do
    let cases : Array (Array Shape4D × Nat) := #[
      (#[( { n := 1, c := 16, h := 8, w := 32 } ), ( { n := 1, c := 24, h := 8, w := 32 } )], 1),
      (#[( { n := 1, c := 8, h := 6, w := 16 } ), ( { n := 1, c := 12, h := 6, w := 16 } ), ( { n := 1, c := 4, h := 6, w := 16 } )], 1),
      (#[( { n := 1, c := 8, h := 4, w := 32 } ), ( { n := 1, c := 8, h := 6, w := 32 } )], 2),
      (#[( { n := 1, c := 8, h := 4, w := 32 } ), ( { n := 1, c := 8, h := 4, w := 64 } )], 3)
    ]
    for tgt in #[Target.h16, Target.h17] do
      for (shs, ax) in cases do
        let vp ← planConcat shs ax
        let c ← compileView tgt vp
        matrixVerify tgt c.hwx c.sections { total := shs.size, neConv := 0, neAct := 0, peOp := 0, neView := shs.size }
    pure ()
  ))

  -- 5. Positive: split (C, H, 32-aligned W) & reshape (viewStrides) on H16 & H17
  results := results.push (checkOk "m5c: split (C, H, 32-aligned W) & reshape (viewStrides) on H16 & H17" (do
    for tgt in #[Target.h16, Target.h17] do
      for part in [:3] do
        let vp ← planSplitPart { n := 1, c := 24, h := 8, w := 32 } 1 #[8, 8, 8] part
        let c ← compileView tgt vp
        matrixVerify tgt c.hwx c.sections { total := 1, neConv := 0, neAct := 0, peOp := 0, neView := 1 }
      let vpH ← planSplitPart { n := 1, c := 8, h := 12, w := 32 } 2 #[4, 8] 1
      let cH ← compileView tgt vpH
      matrixVerify tgt cH.hwx cH.sections { total := 1, neConv := 0, neAct := 0, peOp := 0, neView := 1 }
      let vpW ← planSplitPart { n := 1, c := 8, h := 4, w := 96 } 3 #[32, 64] 1
      let cW ← compileView tgt vpW
      matrixVerify tgt cW.hwx cW.sections { total := 1, neConv := 0, neAct := 0, peOp := 0, neView := 1 }

      let reshapes : Array (Shape4D × Shape4D) := #[
        ({ n := 1, c := 16, h := 8, w := 32 }, { n := 1, c := 32, h := 4, w := 32 }),
        ({ n := 1, c := 4,  h := 4, w := 64 }, { n := 1, c := 4,  h := 8, w := 32 }),
        ({ n := 1, c := 8,  h := 4, w := 16 }, { n := 1, c := 32, h := 1, w := 16 }),
        ({ n := 2, c := 8,  h := 4, w := 32 }, { n := 4, c := 4,  h := 4, w := 32 })
      ]
      for (xSh, ySh) in reshapes do
        let vpR ← planReshape xSh ySh
        let cR ← compileView tgt vpR
        matrixVerify tgt cR.hwx cR.sections { total := 1, neConv := 0, neAct := 0, peOp := 0, neView := 1 }
    pure ()
  ))

  -- 6. Positive: single-task conv1x1 bit-for-bit parity (chooseConv1x1OutputSlices == #[cout])
  results := results.push (checkOk "m5c: small conv1x1 preserves single-task bit-for-bit parity with compileConv" (do
    for tgt in #[Target.h16, Target.h17] do
      let sh : Conv1x1Shape := { h := 8, w := 16, cin := 32, cout := 64 }
      let slices ← chooseConv1x1OutputSlices tgt sh.h sh.w sh.cin sh.cout 0 true
      if slices != #[64] then
        throw s!"{tgt.name}: expected single slice #[64], got {slices}"
      let w := mkM5aW1x1 32 64 5
      let b := mkM5aBias 64
      let cNew ← compileConv1x1 tgt sh w (some b) true
      let cRef ← compileConv tgt sh.toConvSpec w (some b) true
      if cNew.hwx != cRef.hwx then
        throw s!"{tgt.name}: single-task conv1x1 HWX diverged from compileConv!"
    pure ()
  ))

  -- 7. Positive: large-channel auto-split conv1x1 & linear on H16 & H17
  results := results.push (checkOk "m5c: large-channel auto-split conv1x1 & linear (256->512, 512->512, 1024->512, 512->2048) on H16 & H17" (do
    let cases : Array (Nat × Nat × Nat × Nat × ConvActivation × Bool) := #[
      (4, 16, 256, 512, .relu, true),
      (4, 16, 512, 512, .lut .gelu, true),
      (4, 16, 1024, 512, .lut .gelu, true), -- > 64 KiB/chunk on both H16 & H17
      (1, 16, 512, 2048, .none, false)      -- > 512 Cout on both H16 & H17
    ]
    for tgt in #[Target.h16, Target.h17] do
      for (h, w, cin, cout, act, withBias) in cases do
        let wts := mkM5aW1x1 cin cout 3
        let bias? := if withBias then some (mkM5aBias cout) else none
        let c ← compileLinear tgt w cin cout wts bias? h false act
        if c.slices.foldl (· + ·) 0 != cout then
          throw s!"{tgt.name} ({cin}->{cout}): slice sum != {cout}"
        if ((cin == 1024 && cout == 512) || cout == 2048) && c.slices.size < 2 then
          throw s!"{tgt.name} ({cin}->{cout}): expected multi-task split (>= 2 slices), got {c.slices}"
        if tgt == .h16 && cout == 512 && c.slices.size < 2 then
          throw s!"h16 ({cin}->512): expected multi-task split (> 256 Cout cap), got {c.slices}"
        matrixVerify tgt c.hwx c.sections { total := c.slices.size, neConv := c.slices.size, neAct := 0, peOp := 0 }
    pure ()
  ))

  -- 8. Positive: U-Net skip-connection graph (conv3x3 -> conv1x1 -> concat(C) -> conv1x1) on H16 & H17
  let mkUnetSkipGraph := fun (tgt : Target) => ({
    target := tgt
    defaultPlacement := .auto
    inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 32 } }]
    nodes := #[
      { id := "enc", inputs := #["x"], op := .conv3x3 .dense 16 16 1 .same (mkM5aW3x3 16 16 false) (some (mkM5aBias 16)) .relu },
      { id := "mid", inputs := #["enc"], op := .conv1x1 16 16 (mkM5aW1x1 16 16 2) none .relu },
      { id := "cat", inputs := #["mid", "enc"], op := .concat 1 },
      { id := "dec", inputs := #["cat"], op := .conv1x1 32 16 (mkM5aW1x1 32 16 9) (some (mkM5aBias 16)) .none }
    ]
  } : Vane.Graph.GraphSpec)

  for tgt in #[Target.h16, Target.h17] do
    results := results.push (checkOk s!"m5c: {tgt.name} U-Net skip-concat graph (conv3x3 -> conv1x1 -> concat -> conv1x1) verifies S1-S5" (do
      let g ← Vane.Graph.compileGraph (mkUnetSkipGraph tgt)
      if g.taskCount != 5 then
        throw s!"expected 5 tasks (conv3x3 + conv1x1 + 2 concat copies + conv1x1), got {g.taskCount}"
      matrixVerify tgt g.hwx g.sections { total := 5, neConv := 3, neAct := 0, peOp := 0, neView := 2 }
    ))

  -- 9. Positive: Large-hidden Transformer FFN graph (RMSNorm -> linear(256->1024)+GELU -> linear(1024->256) + residual) on H16 & H17
  for tgt in #[Target.h16, Target.h17] do
    results := results.push (checkOk s!"m5c: {tgt.name} large-hidden Transformer FFN graph (RMSNorm -> linear 256->1024 GELU -> linear 1024->256 + add) verifies S1-S5" (do
      let g ← Vane.Graph.compileGraph {
        target := tgt
        defaultPlacement := .auto
        inputs := #[{ name := "x", shape := { c := 256, h := 1, w := 32 } }]
        nodes := #[
          { id := "sq",      inputs := #["x"], op := .pe1 .square },
          { id := "ms",      inputs := #["sq"], op := .reduce .mean { reduceC := true } 0 },
          { id := "ms_eps",  inputs := #["ms"], op := .pe1 (.scalarAdd (floatToF16 0.001)) },
          { id := "inv_rms", inputs := #["ms_eps"], op := .lut (.lut .rsqrt) },
          { id := "norm",    inputs := #["x", "inv_rms"], op := .pe2 .mul false },
          { id := "up",      inputs := #["norm"], op := .linear 256 1024 (mkM5aW1x1 256 1024 1) (some (mkM5aBias 1024)) (.lut .gelu) },
          { id := "down",    inputs := #["up"],   op := .linear 1024 256 (mkM5aW1x1 1024 256 2) (some (mkM5aBias 256)) .none },
          { id := "out",     inputs := #["down", "x"], op := .pe2 .add false }
        ]
      }
      if g.taskCount < 9 then
        throw s!"expected >= 9 tasks due to output-split linear(256->1024), got {g.taskCount}"
      let s ← verifyM5aGraph g g.taskCount
      if s.neConv < 3 || s.neChannelReduce != 1 || s.neAct != 1 || s.peOp != 4 then
        throw s!"unexpected task mix: {repr s}"
      pure ()
    ))

  -- 10. Positive: View-chained graphs (transpose -> slice -> reshape -> conv1x1, plus H/W concat -> conv1x1) on H16 & H17
  results := results.push (checkOk "m5c: view-chained graphs (transpose->slice->reshape->conv1x1 & H/W concat->conv1x1) on H16 & H17" (do
    for tgt in #[Target.h16, Target.h17] do
      let g1 ← Vane.Graph.compileGraph {
        target := tgt
        inputs := #[{ name := "x", shape := { c := 8, h := 16, w := 32 } }]
        nodes := #[
          { id := "tr", inputs := #["x"], op := .transpose #[0, 2, 1, 3] },
          { id := "sl", inputs := #["tr"], op := .slice #[0, 0, 0, 0] #[1, 8, 8, 32] #[1, 1, 1, 1] },
          { id := "rs", inputs := #["sl"], op := .reshape { c := 16, h := 4, w := 32 } },
          { id := "c0", inputs := #["rs"], op := .conv1x1 16 8 (mkM5aW1x1 16 8 3) none .relu }
        ]
      }
      matrixVerify tgt g1.hwx g1.sections { total := 4, neConv := 1, neAct := 0, peOp := 0, neView := 3 }

      let gH ← Vane.Graph.compileGraph {
        target := tgt
        inputs := #[
          { name := "a", shape := { c := 8, h := 4, w := 32 } },
          { name := "b", shape := { c := 8, h := 4, w := 32 } }
        ]
        nodes := #[
          { id := "cat", inputs := #["a", "b"], op := .concat 2 },
          { id := "c0",  inputs := #["cat"], op := .conv1x1 8 8 (mkM5aW1x1 8 8 4) none .none }
        ]
      }
      matrixVerify tgt gH.hwx gH.sections { total := 3, neConv := 1, neAct := 0, peOp := 0, neView := 2 }

      let gW ← Vane.Graph.compileGraph {
        target := tgt
        inputs := #[
          { name := "a", shape := { c := 8, h := 4, w := 32 } },
          { name := "b", shape := { c := 8, h := 4, w := 32 } }
        ]
        nodes := #[
          { id := "cat", inputs := #["a", "b"], op := .concat 3 },
          { id := "c0",  inputs := #["cat"], op := .conv1x1 8 8 (mkM5aW1x1 8 8 5) none .none }
        ]
      }
      matrixVerify tgt gW.hwx gW.sections { total := 3, neConv := 1, neAct := 0, peOp := 0, neView := 2 }
    pure ()
  ))

  -- 11. Negative compiler checks: unsupported T-mode C > 128, C > W, non-32-aligned W views, non-contiguous reshape
  results := results.push (checkErr "m5c neg: T-mode transpose C=129 (> 128) rejected by planTranspose" (do
    let _ ← planTranspose { n := 1, c := 129, h := 2, w := 129 } #[0, 3, 2, 1]
    pure ()
  ) "128")

  results := results.push (checkErr "m5c neg: T-mode transpose C=192 (> 128) rejected by planTranspose" (do
    let _ ← planTranspose { n := 1, c := 192, h := 2, w := 192 } #[0, 3, 2, 1]
    pure ()
  ) "128")

  results := results.push (checkErr "m5c neg: T-mode chooseViewTaskPlan rejects C=129 (> 128)" (do
    let cp : ViewCopy := { n := 1, c := 129, h := 2, w := 129, transposeCW := true }
    let _ ← chooseViewTaskPlan .h17 cp []
    pure ()
  ) "128")

  results := results.push (checkErr "m5c neg: T-mode transpose C > W exceeding ane_hx_dma_tile rejected by planTranspose" (do
    let _ ← planTranspose { n := 1, c := 64, h := 8, w := 16 } #[0, 3, 2, 1]
    pure ()
  ) "ane_hx_dma_tile")

  results := results.push (checkErr "m5c neg: slice with non-32-aligned W begin rejected" (do
    let _ ← planSlice { n := 1, c := 8, h := 8, w := 64 } #[0, 0, 0, 16] #[1, 8, 8, 48] #[1, 1, 1, 1]
    pure ()
  ) "multiple of 32")

  results := results.push (checkErr "m5c neg: concat along W with non-32-aligned input width rejected" (do
    let _ ← planConcat #[{ n := 1, c := 8, h := 4, w := 16 }, { n := 1, c := 8, h := 4, w := 32 }] 3
    pure ()
  ) "multiple of 32")

  results := results.push (checkErr "m5c neg: split along W with non-32-aligned part start rejected" (do
    let _ ← planSplitPart { n := 1, c := 8, h := 4, w := 48 } 3 #[16, 32] 1
    pure ()
  ) "multiple of 32")

  results := results.push (checkErr "m5c neg: reshape crossing 64B padded W boundary rejected" (do
    let _ ← planReshape { n := 1, c := 4, h := 4, w := 24 } { n := 1, c := 4, h := 2, w := 48 }
    pure ()
  ) "non-contiguous")

  -- 12. Negative verifier tamper tests: unmodelled DMA registers (0x4d84, 0x5150, 0x5104)
  let baseViewCopy ← match (do
      let vp ← planTranspose { n := 1, c := 8, h := 8, w := 32 } #[0, 2, 1, 3]
      compileView .h17 vp) with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseViewCopy setup failed: {e}")

  for (regAddr, regName) in #[(regSrc1DmaCrop, "0x4d84"), (regResDmaViewOff, "0x5150"), (regResDmaWarWait, "0x5104")] do
    results := results.push (checkErr s!"m5c neg tamper: unmodelled DMA register {regName} rejected by verifier" (do
      let textBytes ← readSlice baseViewCopy.hwx baseViewCopy.info.textSection.offset.toNat baseViewCopy.info.textSection.size.toNat
      let decoded ← decodeTaskStream .h17 textBytes
      let td ← rebuildTaskWithExtraWrites decoded[0]! 1 #[{ byteAddr := regAddr, value := 0x00000040 }]
      let inSurf ← makeFp16Surface .h17 "x" "input_x" .input 8 8 32
      let outSurf ← makeFp16Surface .h17 "y" "output_y" .output 8 8 32
      let em ← emitProgram {
        target := .h17, surfaces := #[inSurf, outSurf], tasks := #[td],
        kernel := none, scratch := none, constants := zeroBytes textConstBytes, ident := some "vane"
      }
      let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
      pure ()
    ) "unmodelled DMA register")

  -- 13. Negative verifier tamper tests on neView: out-of-bounds inOff/outOff, groupCode, transposeWideGroup, ocg, neScale, T-mode tile overflow
  let baseTModeView ← match (do
      let vp ← planTranspose { n := 1, c := 24, h := 4, w := 32 } #[0, 3, 2, 1]
      compileView .h17 vp) with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseTModeView setup failed: {e}")

  results := results.push (checkErr "m5c neg tamper: neView out-of-bounds src1BaseLo (0x4d10) rejected by verifier" (do
    tamperM5aTaskReg baseViewCopy.hwx baseViewCopy.info 0 regSrc1BaseLo 0x1000
  ) "ane_hx_dma_tile bound")

  results := results.push (checkErr "m5c neg tamper: neView out-of-bounds resBaseLo (0x5108) rejected by verifier" (do
    tamperM5aTaskReg baseViewCopy.hwx baseViewCopy.info 0 regResBaseLo 0x1000
  ) "ane_hx_dma_tile bound")

  results := results.push (checkErr "m5c neg tamper: neView wrong dwGroups/groupCode rejected by verifier" (do
    let badDw ← encodeWord dwGroupsLayout #[("perGroupIn", 1)]
    tamperM5aTaskReg baseTModeView.hwx baseTModeView.info 0 regDwGroups badDw
  ) "groupCode=1 != expected 2")

  results := results.push (checkErr "m5c neg tamper: neView missing transposeWideGroup bit 29 for C=24 rejected by verifier" (do
    let fNc ← findM5aTaskRegOffset baseTModeView.hwx baseTModeView.info 0 regNeCommon
    let orig ← readU32LE baseTModeView.hwx fNc
    tamperM5aTaskReg baseTModeView.hwx baseTModeView.info 0 regNeCommon (orig &&& (~~~((1 : UInt32) <<< 29)))
  ) "transposeWideGroup")

  results := results.push (checkErr "m5c neg tamper: neView wrong neTile.ocg for T-mode rejected by verifier" (do
    let fNt ← findM5aTaskRegOffset baseTModeView.hwx baseTModeView.info 0 regNeTile
    let orig ← readU32LE baseTModeView.hwx fNt
    -- clear ocg bits [2:0] to 0 instead of 1
    tamperM5aTaskReg baseTModeView.hwx baseTModeView.info 0 regNeTile (orig &&& (~~~(0x7 : UInt32)))
  ) "neTile ocg")

  results := results.push (checkErr "m5c neg tamper: neView wrong neScale (!= 0x3c00) rejected by verifier" (do
    tamperM5aTaskReg baseViewCopy.hwx baseViewCopy.info 0 regNeScale 0x00004000
  ) "neScale")

  results := results.push (checkErr "m5c neg tamper: T-mode neView inflated resDmaPlaneStride exceeding ane_hx_dma_tile rejected" (do
    let fPs ← findM5aTaskRegOffset baseTModeView.hwx baseTModeView.info 0 regResDmaPlaneStride
    let orig ← readU32LE baseTModeView.hwx fPs
    tamperM5aTaskReg baseTModeView.hwx baseTModeView.info 0 regResDmaPlaneStride (orig + 0x400)
  ) "ane_hx_dma_tile bound")

  results := results.push (checkErr "m5c neg tamper: T-mode neView F tampered down (C > 2^(F+ocg)) rejected by verifier" (do
    -- baseTModeView has C=24, W=32, ocg=1, F=4, logNE=4, fat=0, WU=0.
    -- Tamper F from 4 to 2 and WU from 0 to 2 (preserving WU+F+fat == logNE=4), so 2^(2+1)=8 < 24.
    let fNc ← findM5aTaskRegOffset baseTModeView.hwx baseTModeView.info 0 regNeCommon
    let fNt ← findM5aTaskRegOffset baseTModeView.hwx baseTModeView.info 0 regNeTile
    let origNc ← readU32LE baseTModeView.hwx fNc
    let origNt ← readU32LE baseTModeView.hwx fNt
    let badNc := (origNc &&& (~~~((0x7 : UInt32) <<< 19))) ||| ((2 : UInt32) <<< 19)
    let badNt := (origNt &&& (~~~((0x3 : UInt32) <<< 4))) ||| ((2 : UInt32) <<< 4)
    let h1 ← putU32LE baseTModeView.hwx fNc badNc
    let h2 ← putU32LE h1 fNt badNt
    let badInfo ← validateHwx h2
    let _ ← verifyHwxTasksSummary h2 badInfo (requireKnownTask := true)
    pure ()
  ) "channel-group coverage")

  results := results.push (checkErr "m5c neg tamper: T-mode neView shMax tampered below coverMin rejected by verifier" (do
    -- baseTModeView has C=24, W=32, fat=0, shPref=0, shMin=0, shMax=1 (coverMin(32, 128, 0) = 1).
    -- Tamper shMax bits [18:16] from 1 down to 0.
    let fNc ← findM5aTaskRegOffset baseTModeView.hwx baseTModeView.info 0 regNeCommon
    let origNc ← readU32LE baseTModeView.hwx fNc
    let badNc := origNc &&& (~~~((0x7 : UInt32) <<< 16))
    tamperM5aTaskReg baseTModeView.hwx baseTModeView.info 0 regNeCommon badNc
  ) "coverMin")

  -- 14. Negative verifier tamper: out-of-bounds resBaseLo on split conv1x1
  let baseSplitConv ← match compileLinear .h16 16 256 512 (mkM5aW1x1 256 512 1) none 4 false .none with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseSplitConv setup failed: {e}")

  results := results.push (checkErr "m5c neg tamper: split conv1x1 out-of-bounds resBaseLo (0x5108) rejected by verifier" (do
    let fRes ← findM5aTaskRegOffset baseSplitConv.hwx baseSplitConv.info 1 regResBaseLo
    let orig ← readU32LE baseSplitConv.hwx fRes
    tamperM5aTaskReg baseSplitConv.hwx baseSplitConv.info 1 regResBaseLo (orig + 0x1000)
  ) "exceeds")

  -- 15. Negative verifier tamper (S5 multi-writer): scratch read with gap between two writers rejected
  let baseUnetH17 ← match Vane.Graph.compileGraph (mkUnetSkipGraph .h17) with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"baseUnetH17 setup failed: {e}")

  results := results.push (checkErr "m5c neg tamper: S5 multi-writer scratch read with gap between concat writers rejected" (do
    -- In mkUnetSkipGraph: task 2 writes concat slice 0 ([0, 16*512)), task 3 writes slice 1 ([16*512, 32*512)),
    -- task 4 (dec) reads [0, 32*512). Shrink task 2's Cin/Cout from 16 to 15 so [15*512, 16*512) is an unwritten gap!
    let fCin ← findM5aTaskRegOffset baseUnetH17.hwx baseUnetH17.info 2 regCin
    let fCout ← findM5aTaskRegOffset baseUnetH17.hwx baseUnetH17.info 2 regCout
    let fRow ← findM5aTaskRegOffset baseUnetH17.hwx baseUnetH17.info 2 regSrc1RowStride
    let fRowA ← findM5aTaskRegOffset baseUnetH17.hwx baseUnetH17.info 2 regSrc1RowBytesA
    let fRowB ← findM5aTaskRegOffset baseUnetH17.hwx baseUnetH17.info 2 regSrc1RowBytesB
    let chs := (← readU32LE baseUnetH17.hwx (← findM5aTaskRegOffset baseUnetH17.hwx baseUnetH17.info 2 regSrc1ChanStride)).toNat
    let newU := (15 * chs).toUInt32
    let h1 ← putU32LE baseUnetH17.hwx fCin 15
    let h2 ← putU32LE h1 fCout 15
    let h3 ← putU32LE h2 fRow newU
    let h4 ← putU32LE h3 fRowA newU
    let h5 ← putU32LE h4 fRowB newU
    let badInfo ← validateHwx h5
    let _ ← verifyHwxTasksSummary h5 badInfo (requireKnownTask := true)
    pure ()
  ) "partially covered")

  -- 16. Negative verifier tamper (S5 multi-writer): unordered overlapping H-concat writers rejected
  let baseConcatH ← match (do
      let vp ← planConcat #[{ n := 1, c := 8, h := 4, w := 32 }, { n := 1, c := 8, h := 4, w := 32 }] 2
      compileView .h17 vp) with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseConcatH setup failed: {e}")

  results := results.push (checkErr "m5c neg tamper: clearing waitNe5 on overlapping multi-writer H-concat rejected by S5" (do
    let fCfg1 ← findM5aTaskRegOffset baseConcatH.hwx baseConcatH.info 1 regSrc1Cfg
    let orig ← readU32LE baseConcatH.hwx fCfg1
    tamperM5aTaskReg baseConcatH.hwx baseConcatH.info 1 regSrc1Cfg (orig &&& (~~~cfgBitWaitNe5))
  ) "S5 ")

  -- 17. Positive multi-task graph with T-mode transpose (C<->W, perm #[0, 3, 2, 1]) on H16 & H17
  results := results.push (checkOk "m5c: multi-task graph conv1x1 -> T-mode transpose [0,3,2,1] -> conv1x1 on H16 & H17 (auto & dram)" (do
    for tgt in #[Target.h16, Target.h17] do
      for pl in #[Vane.Graph.EdgePlacementMode.auto, Vane.Graph.EdgePlacementMode.dram] do
        let g ← Vane.Graph.compileGraph {
          target := tgt
          defaultPlacement := pl
          inputs := #[{ name := "x", shape := { c := 16, h := 4, w := 32 } }]
          nodes := #[
            { id := "c0", inputs := #["x"],  op := .conv1x1 16 32 (mkM5aW1x1 16 32 11) none .none },
            { id := "tr", inputs := #["c0"], op := .transpose #[0, 3, 2, 1] },
            { id := "c1", inputs := #["tr"], op := .conv1x1 32 16 (mkM5aW1x1 32 16 12) none .none }
          ]
        }
        matrixVerify tgt g.hwx g.sections { total := 3, neConv := 2, neAct := 0, peOp := 0, neView := 1 }
    pure ()
  ))

  -- 18. [M2] Regression: multi-task NodeOp.lut .copy under tight L2 budget uses 64B src1ChanStride floor matching verifyNeViewTaskMulti
  results := results.push (checkOk "m5c [M2]: multi-task lut .copy under tight L2 budget compiles & verifies on H16 & H17" (do
    for tgt in #[Target.h16, Target.h17] do
      let g ← Vane.Graph.compileGraph {
        target := tgt
        defaultPlacement := .auto
        inputs := #[{ name := "x", shape := { c := 12288, h := 1, w := 64 } }]
        nodes := #[
          { id := "cp", inputs := #["x"],  op := .lut .copy },
          { id := "ab", inputs := #["cp"], op := .pe1 .abs }
        ]
      }
      matrixVerify tgt g.hwx g.sections { total := 2, neConv := 0, neAct := 0, peOp := 1, neView := 1 }
    pure ()
  ))

  -- 19. [B1] Negative tamper (a): single strided-hole scratch write with full bounding box rejected
  let baseSingleViewGraph ← match Vane.Graph.compileGraph {
      target := .h17
      defaultPlacement := .dram
      inputs := #[{ name := "x", shape := { c := 4, h := 4, w := 32 } }]
      nodes := #[
        { id := "v0", inputs := #["x"],  op := .transpose #[0, 1, 2, 3] },
        { id := "c0", inputs := #["v0"], op := .conv1x1 4 4 (mkM5aW1x1 4 4 13) none .none }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"baseSingleViewGraph setup failed: {e}")

  results := results.push (checkErr "m5c [B1] neg tamper: single strided-hole scratch write with full bounding box rejected" (do
    -- Task 0 originally writes C=4, H=4, W=32, outRowSt=64, outPlaneSt=256 (dense [0, 1024)).
    -- Tamper to C=3, outPlaneSt=384: bounding box is still [0, 2*384+256) = [0, 1024),
    -- yet [256, 384) and [640, 768) are unwritten holes!
    let fCin ← findM5aTaskRegOffset baseSingleViewGraph.hwx baseSingleViewGraph.info 0 regCin
    let fCout ← findM5aTaskRegOffset baseSingleViewGraph.hwx baseSingleViewGraph.info 0 regCout
    let fRow ← findM5aTaskRegOffset baseSingleViewGraph.hwx baseSingleViewGraph.info 0 regSrc1RowStride
    let fRowA ← findM5aTaskRegOffset baseSingleViewGraph.hwx baseSingleViewGraph.info 0 regSrc1RowBytesA
    let fRowB ← findM5aTaskRegOffset baseSingleViewGraph.hwx baseSingleViewGraph.info 0 regSrc1RowBytesB
    let fPl ← findM5aTaskRegOffset baseSingleViewGraph.hwx baseSingleViewGraph.info 0 regResDmaPlaneStride
    let chs := (← readU32LE baseSingleViewGraph.hwx (← findM5aTaskRegOffset baseSingleViewGraph.hwx baseSingleViewGraph.info 0 regSrc1ChanStride)).toNat
    let newU := (3 * chs).toUInt32
    let h1 ← putU32LE baseSingleViewGraph.hwx fCin 3
    let h2 ← putU32LE h1 fCout 3
    let h3 ← putU32LE h2 fRow newU
    let h4 ← putU32LE h3 fRowA newU
    let h5 ← putU32LE h4 fRowB newU
    let h6 ← putU32LE h5 fPl 384
    let badInfo ← validateHwx h6
    let _ ← verifyHwxTasksSummary h6 badInfo (requireKnownTask := true)
    pure ()
  ) "partially covered")

  -- 20. [B1] Negative tamper (b): partial H-concat leaving interior per-channel row gaps despite full bounding-box union
  let baseConcatHGraph ← match Vane.Graph.compileGraph {
      target := .h17
      defaultPlacement := .dram
      inputs := #[
        { name := "a", shape := { c := 8, h := 4, w := 32 } },
        { name := "b", shape := { c := 8, h := 4, w := 32 } }
      ]
      nodes := #[
        { id := "cat", inputs := #["a", "b"], op := .concat 2 },
        { id := "c0",  inputs := #["cat"], op := .conv1x1 8 8 (mkM5aW1x1 8 8 14) none .none }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"baseConcatHGraph setup failed: {e}")

  results := results.push (checkErr "m5c [B1] neg tamper: partial H-concat with interior channel gaps (full bbox union) rejected" (do
    -- Shrink task 0 Hin/Hout/TH from 4 to 3: bounding boxes [0, 3776) U [256, 4096) == [0, 4096),
    -- yet row 3 ([192, 256)) of every channel is unwritten!
    let fHin ← findM5aTaskRegOffset baseConcatHGraph.hwx baseConcatHGraph.info 0 regHin
    let fHout ← findM5aTaskRegOffset baseConcatHGraph.hwx baseConcatHGraph.info 0 regHout
    let fTh ← findM5aTaskRegOffset baseConcatHGraph.hwx baseConcatHGraph.info 0 regTH
    let h1 ← putU32LE baseConcatHGraph.hwx fHin 3
    let h2 ← putU32LE h1 fHout 3
    let h3 ← putU32LE h2 fTh 3
    let badInfo ← validateHwx h3
    let _ ← verifyHwxTasksSummary h3 badInfo (requireKnownTask := true)
    pure ()
  ) "partially covered")

  -- 21. [B1+M3] Multi-writer [multiBatchRaw] superseding: full dense supersede passes, partial dense & strided bbox-full rejected
  let baseMbSupersede ← match Vane.Graph.compileGraph {
      target := .h17
      defaultPlacement := .dram
      inputs := #[{ name := "x", shape := { c := 4, h := 4, w := 16 } }]
      nodes := #[
        { id := "red", inputs := #["x"],          op := .reduce .sum { reduceW := true } },
        { id := "cpy", inputs := #["x"],          op := .transpose #[0, 1, 2, 3] },
        { id := "out", inputs := #["cpy", "red"], op := .pe2 .add false }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"baseMbSupersede setup failed: {e}")

  let mkMbSupersedeHwx := fun (cpyMode : Nat) => do
    -- Task 0 (red, PE, BatchN=4) writes scratch [0, 1024).
    -- Task 1 (cpy, NE, BatchN=1) originally writes [1024, 2048); move its resBaseLo -> 0 and add waitPe22 (task 0 -> task 1 WAW).
    -- Task 2 (out, PE2) reads [0, 1024) on both src1 (depKind=2) and src2 (tampered depKind 1 -> 2).
    -- cpyMode = 0: full dense [0, 1024) (completely supersedes task 0 -> MUST PASS)
    -- cpyMode = 1: partial dense [0, 512) (Cin=Cout=2 -> MUST FAIL [multiBatchRaw])
    -- cpyMode = 2: strided holes C=3, outPlaneSt=384, bbox [0, 1024) -> MUST FAIL [multiBatchRaw]
    let fCfg1 ← findM5aTaskRegOffset baseMbSupersede.hwx baseMbSupersede.info 1 regSrc1Cfg
    let origCfg1 ← readU32LE baseMbSupersede.hwx fCfg1
    let fRes1 ← findM5aTaskRegOffset baseMbSupersede.hwx baseMbSupersede.info 1 regResBaseLo
    let fSrc12 ← findM5aTaskRegOffset baseMbSupersede.hwx baseMbSupersede.info 2 regSrc1BaseLo
    let fDep22 ← findM5aTaskRegOffset baseMbSupersede.hwx baseMbSupersede.info 2 regSrc2Dep
    let origDep22 ← readU32LE baseMbSupersede.hwx fDep22
    let badDep22 := (origDep22 &&& (~~~((0x3 : UInt32) <<< 28))) ||| ((2 : UInt32) <<< 28)
    let mut h ← putU32LE baseMbSupersede.hwx fCfg1 (origCfg1 ||| cfgBitWaitPe22)
    h ← putU32LE h fRes1 0
    h ← putU32LE h fSrc12 0
    h ← putU32LE h fDep22 badDep22
    if cpyMode == 1 || cpyMode == 2 then
      let newC : Nat := if cpyMode == 1 then 2 else 3
      let fCin ← findM5aTaskRegOffset h baseMbSupersede.info 1 regCin
      let fCout ← findM5aTaskRegOffset h baseMbSupersede.info 1 regCout
      let fRow ← findM5aTaskRegOffset h baseMbSupersede.info 1 regSrc1RowStride
      let fRowA ← findM5aTaskRegOffset h baseMbSupersede.info 1 regSrc1RowBytesA
      let fRowB ← findM5aTaskRegOffset h baseMbSupersede.info 1 regSrc1RowBytesB
      let chs := (← readU32LE h (← findM5aTaskRegOffset h baseMbSupersede.info 1 regSrc1ChanStride)).toNat
      let newU := (newC * chs).toUInt32
      h ← putU32LE h fCin newC.toUInt32
      h ← putU32LE h fCout newC.toUInt32
      h ← putU32LE h fRow newU
      h ← putU32LE h fRowA newU
      h ← putU32LE h fRowB newU
      if cpyMode == 2 then
        let fPl ← findM5aTaskRegOffset h baseMbSupersede.info 1 regResDmaPlaneStride
        h ← putU32LE h fPl 384
    let info ← validateHwx h
    let _ ← verifyHwxTasksSummary h info (requireKnownTask := true)
    pure ()

  results := results.push (checkOk "m5c [B1] pos: full dense BatchN=1 overwrite supersedes prior BatchN>1 scratch writer" (mkMbSupersedeHwx 0))
  results := results.push (checkErr "m5c [B1+M3] neg tamper: partial dense overwrite fails to supersede BatchN>1 scratch writer ([multiBatchRaw])" (mkMbSupersedeHwx 1) "[multiBatchRaw]")
  results := results.push (checkErr "m5c [B1+M3] neg tamper: strided-hole overwrite (full bbox) fails to supersede BatchN>1 scratch writer ([multiBatchRaw])" (mkMbSupersedeHwx 2) "[multiBatchRaw]")

  -- 22. [M3] Negative tamper: T-mode alignedActualOutEnd > cap while alignedKernelOutEnd <= cap
  let baseTModeSmallC ← match (do
      let vp ← planTranspose { n := 1, c := 8, h := 4, w := 32 } #[0, 3, 2, 1]
      compileView .h17 vp) with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseTModeSmallC setup failed: {e}")

  results := results.push (checkErr "m5c [M3] neg tamper: T-mode actual output write range > cap while kernel ane_hx_dma_tile <= cap rejected" (do
    -- Output surface cap = 32*4*64 = 8192. Kernel bound = 64 + 2048 = 2112 <= 8192,
    -- actual hardware write bound = alignUp(64 + 8144, 64) = 8256 > 8192.
    tamperM5aTaskReg baseTModeSmallC.hwx baseTModeSmallC.info 0 regResBaseLo 64
  ) "output DMA write range")

  -- 23. [M3] Negative tamper: neView resCfg.direct mismatch (T-mode staged rule 9<=C<=16, shMin==0, W<32)
  let baseTModeStaged ← match (do
      let vp ← planTranspose { n := 1, c := 12, h := 4, w := 16 } #[0, 3, 2, 1]
      compileView .h17 vp) with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseTModeStaged setup failed: {e}")

  results := results.push (checkErr "m5c [M3] neg tamper: T-mode staged neView resCfg/resDep direct=1 rejected by verifier" (do
    let fCfg ← findM5aTaskRegOffset baseTModeStaged.hwx baseTModeStaged.info 0 regResCfg
    let fDep ← findM5aTaskRegOffset baseTModeStaged.hwx baseTModeStaged.info 0 regResDep
    let origCfg ← readU32LE baseTModeStaged.hwx fCfg
    let origDep ← readU32LE baseTModeStaged.hwx fDep
    let h1 ← putU32LE baseTModeStaged.hwx fCfg (origCfg ||| ((1 : UInt32) <<< 3))
    let h2 ← putU32LE h1 fDep (origDep ||| ((1 : UInt32) <<< 24))
    let badInfo ← validateHwx h2
    let _ ← verifyHwxTasksSummary h2 badInfo (requireKnownTask := true)
    pure ()
  ) "resCfg direct=true disagrees with expected direct=false")

  -- 24. [M3] Negative tamper: neView unaligned src1BaseLo/resBaseLo & unaligned DMA strides (% 64 != 0)
  results := results.push (checkErr "m5c [M3] neg tamper: neView unaligned src1BaseLo (32 % 64 != 0) rejected" (do
    tamperM5aTaskReg baseViewCopy.hwx baseViewCopy.info 0 regSrc1BaseLo 32
  ) "not 64-byte aligned")

  results := results.push (checkErr "m5c [M3] neg tamper: neView unaligned resBaseLo (32 % 64 != 0) rejected" (do
    tamperM5aTaskReg baseViewCopy.hwx baseViewCopy.info 0 regResBaseLo 32
  ) "not 64-byte aligned")

  results := results.push (checkErr "m5c [M3] neg tamper: neView unaligned src1DmaRowStride (% 64 != 0) rejected" (do
    let fSt ← findM5aTaskRegOffset baseViewCopy.hwx baseViewCopy.info 0 regSrc1DmaRowStride
    let orig ← readU32LE baseViewCopy.hwx fSt
    tamperM5aTaskReg baseViewCopy.hwx baseViewCopy.info 0 regSrc1DmaRowStride (orig + 32)
  ) "not 64-byte aligned")

  results := results.push (checkErr "m5c [M3] neg tamper: neView unaligned resDmaRowStride (% 64 != 0) rejected" (do
    let fSt ← findM5aTaskRegOffset baseViewCopy.hwx baseViewCopy.info 0 regResDmaRowStride
    let orig ← readU32LE baseViewCopy.hwx fSt
    tamperM5aTaskReg baseViewCopy.hwx baseViewCopy.info 0 regResDmaRowStride (orig + 32)
  ) "not 64-byte aligned")

  -- 25. [M3] Negative tamper: neView L2 staging overflow (> 2 MiB)
  results := results.push (checkErr "m5c [M3] neg tamper: neView L2 resBase past 2 MiB rejected" (do
    tamperM5aTaskReg baseViewCopy.hwx baseViewCopy.info 0 regResBase 0x001fffc0
  ) "exceeds 2 MiB")

  -- 26. [Q1] Staged neView with D > 1 rejected by both compiler (viewStageBasesForTh) and verifier
  results := results.push (checkOk "m5c [Q1]: chooseViewTaskPlan uses staged shPref=0 for D=1 but forbids staged for D=2 (promotes shPref=1 direct=true)" (do
    let cp1 : ViewCopy := { n := 1, c := 12, h := 4, d := 1, w := 16, transposeCW := true }
    let p1 ← chooseViewTaskPlan .h17 cp1 []
    if p1.direct || p1.tiling.shPref != 0 then
      throw s!"expected D=1 to choose staged shPref=0, got direct={p1.direct} shPref={p1.tiling.shPref}"
    let cp2 : ViewCopy := { cp1 with d := 2 }
    let p2 ← chooseViewTaskPlan .h17 cp2 []
    if !p2.direct || p2.tiling.shPref == 0 then
      throw s!"expected D=2 to reject staged shPref=0 and promote to direct=true, got direct={p2.direct} shPref={p2.tiling.shPref}"
    pure ()
  ))

  results := results.push (checkErr "m5c [Q1] neg tamper: staged neView with D=2 rejected by verifier" (do
    let fRow ← findM5aTaskRegOffset baseTModeStaged.hwx baseTModeStaged.info 0 regSrc1RowStride
    let fRowB ← findM5aTaskRegOffset baseTModeStaged.hwx baseTModeStaged.info 0 regSrc1RowBytesB
    let origRowA := (← readU32LE baseTModeStaged.hwx (← findM5aTaskRegOffset baseTModeStaged.hwx baseTModeStaged.info 0 regSrc1RowBytesA)).toNat
    let newDU := (2 * origRowA).toUInt32
    let h1 ← putU32LE baseTModeStaged.hwx fRow newDU
    let h2 ← putU32LE h1 fRowB newDU
    let textBytes ← readSlice h2 baseTModeStaged.info.textSection.offset.toNat baseTModeStaged.info.textSection.size.toNat
    let decoded ← decodeTaskStream .h17 textBytes
    let td ← rebuildTaskWithExtraWrites decoded[0]! 1 #[
      { byteAddr := regDin, value := 2 },
      { byteAddr := regDout, value := 2 }
    ]
    let inSurf ← makeFp16Surface .h17 "x" "input_x" .input 12 4 16
    let outSurf ← makeFp16Surface .h17 "y" "output_y" .output 16 4 12
    let em ← emitProgram {
      target := .h17, surfaces := #[inSurf, outSurf], tasks := #[td],
      kernel := none, scratch := none, constants := zeroBytes textConstBytes, ident := some "vane"
    }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "staged mode (direct=false) requires D=1")

  pure results

-- ===========================================================================
-- Milestone 5D Tests: Dynamic MatMul, Multi-Head SDPA, Full Transformer Block,
-- and Dedicated Negative Verifier Tamper Tests
-- ===========================================================================

private def rebuildDynTask
    (t : DecodedTask) (totalTasks : Nat)
    (extProdOverride : Option (Option Nat) := none)
    (transformWrites : Array DecodedWrite → Array DecodedWrite := id) : Except String TaskDesc := do
  let ws := transformWrites t.writes
  let ene ←
    if t.header.size == 10 then
      decodeFieldChecked taskHeader8ExtLayout t.header[8]! "ene"
    else
      decodeFieldChecked taskHeader8MultiLayout t.header[8]! "ene"
  let origExt : Option Nat := if t.header.size == 10 then some t.header[9]!.toNat else none
  let effExt := extProdOverride.getD origExt
  let mut plain : Array RegWrite := #[]
  let mut bars : Array BarRegWrite := #[]
  let mut i := 0
  while i < ws.size do
    let w := ws[i]!
    match w.barSlot with
    | none =>
      plain := plain.push { byteAddr := w.byteAddr, value := w.value }
      i := i + 1
    | some slot =>
      if slot != barSlotConfigZero && i + 1 < ws.size && ws[i + 1]!.barSlot == some slot && ws[i + 1]!.byteAddr == w.byteAddr + 4 then
        bars := bars.push { baseByteAddr := w.byteAddr, barSlot := slot, values := #[w.value, ws[i + 1]!.value] }
        i := i + 2
      else
        bars := bars.push { baseByteAddr := w.byteAddr, barSlot := slot, values := #[w.value] }
        i := i + 1
  pure {
    taskId := t.taskIndex
    logNE := 0
    eneOverride := some ene.toNat
    isFirst := t.taskIndex == 0
    isLast := t.taskIndex + 1 == totalTasks
    liveL2Mask := t.header[7]!
    extProducerTask := effExt
    plainWrites := plain
    barWrites := bars
    kernelRelocs := #[]
  }

/-- Build a complete Multi-Head Scaled Dot-Product Attention graph (`H=4, L=32, S=32, D=32`):
    `Q @ K^T * (1/sqrt(D)) -> Softmax(axes=w) -> P @ V`. -/
private def mkMultiHeadSdpaGraph (target : Target) (heads l s d : Nat) : Vane.Graph.GraphSpec :=
  let scaleBits := floatToF16CanonicalZero (1.0 / Float.sqrt (Float.ofNat d))
  {
    target := target
    defaultPlacement := .auto
    inputs := #[
      { name := "q", shape := { c := heads * l, h := 1, w := d }, shortName := "q" },
      { name := "k", shape := { c := heads * s, h := 1, w := d }, shortName := "k" },
      { name := "v", shape := { c := heads * s, h := 1, w := d }, shortName := "v" }
    ]
    nodes := #[
      { id := "scores", inputs := #["q", "k"], op := .matmul heads false true scaleBits },
      { id := "rmax",   inputs := #["scores"], op := .reduce .max { reduceW := true } },
      { id := "shift",  inputs := #["scores", "rmax"], op := .pe2 .sub false },
      { id := "exps",   inputs := #["shift"],  op := .lut (.lut .exp) },
      { id := "rsum",   inputs := #["exps"],   op := .reduce .sum { reduceW := true } },
      { id := "inv",    inputs := #["rsum"],   op := .lut (.lut .inverse) },
      { id := "probs",  inputs := #["exps", "inv"], op := .pe2 .mul false },
      { id := "out",    inputs := #["probs", "v"], op := .matmul heads false false f16PosOne }
    ]
  }

/-- Build a complete Transformer Block graph on `[1, D, 1, L]` (`L=32, D=64, H=2, Dh=32, FFN=256`):
    `RMSNorm -> Q/K/V linear -> transpose to [1, H*L, 1, Dh] -> Multi-Head SDPA -> transpose back ->
     O linear + residual -> RMSNorm -> wide FFN (linear + GELU -> linear) + residual`. -/
private def mkTransformerBlockGraph (target : Target) : Vane.Graph.GraphSpec :=
  let l := 32
  let d := 64
  let heads := 2
  let dh := 32
  let ffn := 256
  let scaleBits := floatToF16CanonicalZero (1.0 / Float.sqrt (Float.ofNat dh))
  {
    target := target
    defaultPlacement := .auto
    inputs := #[
      { name := "x", shape := { c := d, h := 1, w := l }, shortName := "x" }
    ]
    nodes := #[
      -- 1. Pre-Attention RMSNorm over channels C (`axes = c`)
      { id := "sq1",    inputs := #["x"], op := .pe1 .square },
      { id := "ms1",    inputs := #["sq1"], op := .reduce .mean { reduceC := true } },
      { id := "ms1_e",  inputs := #["ms1"], op := .pe1 (.chain none (some (floatToF16CanonicalZero 1e-3)) none) },
      { id := "rms1",   inputs := #["ms1_e"], op := .lut (.lut (.rsqrt 0.0)) },
      { id := "norm1",  inputs := #["x", "rms1"], op := .pe2 .mul false },
      -- 2. Q, K, V linear projections: `[1, D, 1, L] -> [1, D, 1, L]`
      { id := "q_proj", inputs := #["norm1"], op := .linear d d (mkM5aW1x1 d d 1) none .none },
      { id := "k_proj", inputs := #["norm1"], op := .linear d d (mkM5aW1x1 d d 2) none .none },
      { id := "v_proj", inputs := #["norm1"], op := .linear d d (mkM5aW1x1 d d 3) none .none },
      -- 3. Reshape `[1, H*Dh, 1, L] -> [1, H, Dh, L]` then transpose `[0, 1, 3, 2]` -> `[1, H, L, Dh]` -> reshape `[1, H*L, 1, Dh]`
      --    (Here using 1-head or C<->W transpose directly: `[1, D=64, 1, L=32]` -> split per head `[1, 32, 1, 32]` -> T-mode transpose -> concat!)
      { id := "q_h0",   inputs := #["q_proj"], op := .slice #[0, 0, 0, 0] #[1, dh, 1, l] #[1, 1, 1, 1] },
      { id := "q_h1",   inputs := #["q_proj"], op := .slice #[0, dh, 0, 0] #[1, d, 1, l] #[1, 1, 1, 1] },
      { id := "q_h0_t", inputs := #["q_h0"],   op := .transpose #[0, 3, 2, 1] },
      { id := "q_h1_t", inputs := #["q_h1"],   op := .transpose #[0, 3, 2, 1] },
      { id := "q_mh",   inputs := #["q_h0_t", "q_h1_t"], op := .concat 1 },

      { id := "k_h0",   inputs := #["k_proj"], op := .slice #[0, 0, 0, 0] #[1, dh, 1, l] #[1, 1, 1, 1] },
      { id := "k_h1",   inputs := #["k_proj"], op := .slice #[0, dh, 0, 0] #[1, d, 1, l] #[1, 1, 1, 1] },
      { id := "k_h0_t", inputs := #["k_h0"],   op := .transpose #[0, 3, 2, 1] },
      { id := "k_h1_t", inputs := #["k_h1"],   op := .transpose #[0, 3, 2, 1] },
      { id := "k_mh",   inputs := #["k_h0_t", "k_h1_t"], op := .concat 1 },

      { id := "v_h0",   inputs := #["v_proj"], op := .slice #[0, 0, 0, 0] #[1, dh, 1, l] #[1, 1, 1, 1] },
      { id := "v_h1",   inputs := #["v_proj"], op := .slice #[0, dh, 0, 0] #[1, d, 1, l] #[1, 1, 1, 1] },
      { id := "v_h0_t", inputs := #["v_h0"],   op := .transpose #[0, 3, 2, 1] },
      { id := "v_h1_t", inputs := #["v_h1"],   op := .transpose #[0, 3, 2, 1] },
      { id := "v_mh",   inputs := #["v_h0_t", "v_h1_t"], op := .concat 1 },

      -- 4. Multi-Head Scaled Dot-Product Attention (`heads = 2`, `Q @ K^T * scale -> Softmax -> P @ V`)
      { id := "scores", inputs := #["q_mh", "k_mh"], op := .matmul heads false true scaleBits },
      { id := "rmax",   inputs := #["scores"], op := .reduce .max { reduceW := true } },
      { id := "shift",  inputs := #["scores", "rmax"], op := .pe2 .sub false },
      { id := "exps",   inputs := #["shift"],  op := .lut (.lut .exp) },
      { id := "rsum",   inputs := #["exps"],   op := .reduce .sum { reduceW := true } },
      { id := "inv",    inputs := #["rsum"],   op := .lut (.lut .inverse) },
      { id := "probs",  inputs := #["exps", "inv"], op := .pe2 .mul false },
      { id := "attn",   inputs := #["probs", "v_mh"], op := .matmul heads false false f16PosOne },

      -- 5. Merge heads back to `[1, D, 1, L]`
      { id := "a_h0",   inputs := #["attn"], op := .slice #[0, 0, 0, 0] #[1, l, 1, dh] #[1, 1, 1, 1] },
      { id := "a_h1",   inputs := #["attn"], op := .slice #[0, l, 0, 0] #[1, heads * l, 1, dh] #[1, 1, 1, 1] },
      { id := "a_h0_t", inputs := #["a_h0"], op := .transpose #[0, 3, 2, 1] },
      { id := "a_h1_t", inputs := #["a_h1"], op := .transpose #[0, 3, 2, 1] },
      { id := "a_cat",  inputs := #["a_h0_t", "a_h1_t"], op := .concat 1 },

      -- 6. Output projection + residual
      { id := "o_proj", inputs := #["a_cat"], op := .linear d d (mkM5aW1x1 d d 4) none .none },
      { id := "res1",   inputs := #["x", "o_proj"], op := .pe2 .add false },

      -- 7. Post-Attention RMSNorm + wide FFN (GELU) + residual
      { id := "sq2",    inputs := #["res1"], op := .pe1 .square },
      { id := "ms2",    inputs := #["sq2"], op := .reduce .mean { reduceC := true } },
      { id := "ms2_e",  inputs := #["ms2"], op := .pe1 (.chain none (some (floatToF16CanonicalZero 1e-3)) none) },
      { id := "rms2",   inputs := #["ms2_e"], op := .lut (.lut (.rsqrt 0.0)) },
      { id := "norm2",  inputs := #["res1", "rms2"], op := .pe2 .mul false },
      { id := "ffn1",   inputs := #["norm2"], op := .linear d ffn (mkM5aW1x1 d ffn 5) (some (mkM5aBias ffn)) (.lut .gelu) },
      { id := "ffn2",   inputs := #["ffn1"], op := .linear ffn d (mkM5aW1x1 ffn d 6) (some (mkM5aBias d)) .none },
      { id := "out",    inputs := #["res1", "ffn2"], op := .pe2 .add false }
    ]
  }

private def runM5dTests : IO (Array TestResult) := do
  let mut results : Array TestResult := #[]

  -- 1. Positive: standalone compileMatmul across H16 & H17 covering all modes (including Mi <= 16 row-slicing for M > 16)
  for tgt in Target.all do
    results := results.push (checkOk s!"m5d ({tgt.name}): standalone compileMatmul matrix (B=1/4, M=1/8/16/20/32, trA/trB/both, scale)" (do
      let scaleInvSqrt32 := floatToF16CanonicalZero (1.0 / Float.sqrt 32.0)
      let cases : Array (MatmulSpec × Nat × Nat) := #[
        ({ b := 1, m := 1,  k := 16, n := 16, transposeA := false, transposeB := false, scaleBits := f16PosOne }, 1, 0),
        ({ b := 1, m := 8,  k := 32, n := 32, transposeA := false, transposeB := false, scaleBits := f16PosOne }, 1, 0),
        ({ b := 1, m := 32, k := 32, n := 128, transposeA := false, transposeB := false, scaleBits := scaleInvSqrt32 }, 2, 0),
        ({ b := 4, m := 16, k := 32, n := 32, transposeA := false, transposeB := false, scaleBits := f16PosOne }, 1, 0),
        ({ b := 4, m := 32, k := 32, n := 32, transposeA := false, transposeB := false, scaleBits := scaleInvSqrt32 }, 2, 0),
        ({ b := 1, m := 32, k := 32, n := 64, transposeA := false, transposeB := true,  scaleBits := scaleInvSqrt32 }, 2, 1),
        ({ b := 4, m := 16, k := 32, n := 32, transposeA := false, transposeB := true,  scaleBits := scaleInvSqrt32 }, 1, 1),
        ({ b := 4, m := 32, k := 32, n := 32, transposeA := false, transposeB := true,  scaleBits := scaleInvSqrt32 }, 2, 1),
        ({ b := 1, m := 16, k := 32, n := 32, transposeA := true,  transposeB := false, scaleBits := f16PosOne }, 1, 1),
        ({ b := 4, m := 20, k := 32, n := 32, transposeA := true,  transposeB := true,  scaleBits := scaleInvSqrt32 }, 2, 2)
      ]
      for (sp, expMm, expView) in cases do
        let expTasks := expMm + expView
        let c ← compileMatmul tgt sp
        if c.taskCount != expTasks then
          throw s!"{tgt.name} {repr sp}: taskCount={c.taskCount} != {expTasks}"
        if c.info.kernelSection.isSome then
          throw s!"{tgt.name} {repr sp}: dynamic matmul should not emit __KERN_0"
        let info ← validateHwx c.hwx
        validateSectionsWithInfo info c.sections
        let s ← verifyHwxTasksSummary c.hwx info (requireKnownTask := true)
        if s.neMatmul != expMm || s.neView != expView || s.total != expTasks then
          throw s!"{tgt.name} {repr sp}: summary mismatch {repr s} (expected mm={expMm}, view={expView})"
      pure ()
    ))

  -- 2. Positive: Multi-Head Scaled Dot-Product Attention graph (`H=4, L=32, S=32, D=32`) on H16 & H17
  for tgt in Target.all do
    results := results.push (checkOk s!"m5d ({tgt.name}): Multi-Head SDPA graph (H=4, L=32, S=32, D=32) compiles & passes S1-S5" (do
      let g ← Vane.Graph.compileGraph (mkMultiHeadSdpaGraph tgt 4 32 32 32)
      let info ← validateHwx g.hwx
      validateSectionsWithInfo info g.sections
      let s ← verifyHwxTasksSummary g.hwx info (requireKnownTask := true)
      if s.neMatmul != 4 || s.neView != 1 || s.peReduce != 2 || s.neAct != 2 || s.peOp != 2 || s.total != 11 then
        throw s!"{tgt.name} SDPA summary mismatch: {repr s}"
      -- Verify both P @ V slices use 10-word extended headers and slice 0 sets keepLive=1 (0xc1)
      let textBytes ← readSlice g.hwx info.textSection.offset.toNat info.textSection.size.toNat
      let tasks ← decodeTaskStream tgt textBytes
      let prevTask := tasks[tasks.size - 2]!
      let lastTask := tasks[tasks.size - 1]!
      if prevTask.header.size != 10 || lastTask.header.size != 10 then
        throw s!"expected 10-word extended headers on P @ V slices, got {prevTask.header.size} and {lastTask.header.size} words"
      match prevTask.writes.find? (fun w => w.byteAddr == regChunkSlot 0) with
      | some w0 =>
        let kl ← decodeFieldChecked chunkSlotDynLayout w0.value "keepLive"
        if kl != 1 then
          throw s!"expected keepLive=1 on P @ V slice 0 chunkSlot[0], got {toHex w0.value.toNat}"
      | none => throw "missing chunkSlot[0] on P @ V slice 0"
      pure ()
    ))

  -- 3. Positive: Full Transformer Block graph (RMSNorm + QKV + Multi-Head SDPA + O + Residual + RMSNorm + Wide FFN + Residual) on H16 & H17
  for tgt in Target.all do
    results := results.push (checkOk s!"m5d ({tgt.name}): Full Transformer Block graph compiles & passes S1-S5" (do
      let g ← Vane.Graph.compileGraph (mkTransformerBlockGraph tgt)
      let info ← validateHwx g.hwx
      validateSectionsWithInfo info g.sections
      let s ← verifyHwxTasksSummary g.hwx info (requireKnownTask := true)
      if s.neMatmul != 4 || s.total != g.taskCount then
        throw s!"{tgt.name} Transformer block summary mismatch: {repr s} (expected {g.taskCount} tasks, 4 neMatmul)"
      pure ()
    ))

  -- 4. Positive: single-head matmul in graph IR with L2-resident B & Y (auto) and DRAM mode, plus multi-slice linear -> matmul A
  for tgt in Target.all do
    results := results.push (checkOk s!"m5d ({tgt.name}): single-head matmul L2/DRAM graph & multi-slice linear -> matmul kernel" (do
      for pl in #[Vane.Graph.EdgePlacementMode.auto, Vane.Graph.EdgePlacementMode.dram] do
        let g1 ← Vane.Graph.compileGraph {
          target := tgt
          defaultPlacement := pl
          inputs := #[
            { name := "a", shape := { c := 16, h := 1, w := 32 }, shortName := "a" },
            { name := "b", shape := { c := 32, h := 1, w := 32 }, shortName := "b" }
          ]
          nodes := #[
            { id := "b_abs", inputs := #["b"], op := .pe1 .abs },
            { id := "mm",    inputs := #["a", "b_abs"], op := .matmul 1 false false f16PosOne },
            { id := "out",   inputs := #["mm"], op := .pe1 .abs }
          ]
        }
        let s1 ← verifyHwxTasksSummary g1.hwx g1.info (requireKnownTask := true)
        if s1.neMatmul != 1 || s1.peOp != 2 || s1.total != 3 then
          throw s!"{tgt.name} ({pl.name}) summary mismatch: {repr s1}"
      -- Multi-slice output-split linear (512 channels on H16 splits into 2 tasks) feeding 32-slice matmul kernel A (512 / 16)
      let gSplit ← Vane.Graph.compileGraph {
        target := tgt
        defaultPlacement := .auto
        inputs := #[
          { name := "x", shape := { c := 32, h := 1, w := 32 }, shortName := "x" },
          { name := "b", shape := { c := 32, h := 1, w := 32 }, shortName := "b" }
        ]
        nodes := #[
          { id := "lin", inputs := #["x"], op := .linear 32 512 (mkM5aW1x1 32 512 9) none .none },
          { id := "mm",  inputs := #["lin", "b"], op := .matmul 1 false false f16PosOne }
        ]
      }
      let s2 ← verifyHwxTasksSummary gSplit.hwx gSplit.info (requireKnownTask := true)
      if s2.neMatmul != 32 || s2.total != gSplit.taskCount then
        throw s!"{tgt.name} multi-slice linear->matmul summary mismatch: {repr s2}"
      pure ()
    ))

  -- 5. Negative compiler rejections: StripMode with matmul, M > 2048, K > 32768, non-finite scale, transposeA K > 128, transposeB N > 128
  results := results.push (checkErr "m5d neg compiler: StripMode rejected on graph containing matmul" (do
    let _ ← Vane.Graph.compileGraph {
      target := .h17
      stripMode := .fixed 1
      inputs := #[
        { name := "a", shape := { c := 16, h := 1, w := 16 }, shortName := "a" },
        { name := "b", shape := { c := 16, h := 1, w := 16 }, shortName := "b" }
      ]
      nodes := #[
        { id := "mm", inputs := #["a", "b"], op := .matmul 1 false false f16PosOne }
      ]
    }
    pure ()
  ) "spatial strip fusion ineligible")

  results := results.push (checkErr "m5d neg compiler: M > 2048 rejected" (do
    let _ ← compileMatmul .h17 { b := 1, m := 2049, k := 16, n := 16 }
    pure ()
  ) "1..2048")

  results := results.push (checkErr "m5d neg compiler: chunkSize > 65536 (K=32769) rejected" (do
    let _ ← compileMatmul .h17 { b := 1, m := 16, k := 32769, n := 16 }
    pure ()
  ) "64 KiB")

  results := results.push (checkErr "m5d neg compiler: non-finite scale (+Inf) rejected" (do
    let _ ← compileMatmul .h17 { b := 1, m := 16, k := 16, n := 16, scaleBits := f16PosInf }
    pure ()
  ) "finite")

  results := results.push (checkErr "m5d neg compiler: transposeA with K=129 > 128 rejected" (do
    let _ ← compileMatmul .h17 { b := 1, m := 16, k := 129, n := 16, transposeA := true }
    pure ()
  ) "128")

  results := results.push (checkErr "m5d neg compiler: transposeB with N=129 > 128 rejected" (do
    let _ ← compileMatmul .h17 { b := 1, m := 16, k := 16, n := 129, transposeB := true }
    pure ()
  ) "128")

  -- =========================================================================
  -- Dedicated Negative Verifier Tamper Tests for EVERY New Verifier Rule
  -- =========================================================================
  let baseExtH16 ← match compileMatmul .h16 { b := 4, m := 16, k := 32, n := 32 } with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseExtH16 setup failed: {e}")

  let baseExtH17 ← match compileMatmul .h17 { b := 4, m := 16, k := 32, n := 32 } with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseExtH17 setup failed: {e}")

  let baseProdH17 ← match compileMatmul .h17 { b := 1, m := 8, k := 32, n := 32, transposeA := true } with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseProdH17 setup failed: {e}")

  -- Tamper 1: Bad neCfg (H17 bit set on H16)
  results := results.push (checkErr "m5d neg tamper #1: H17 neCfg (0x10010082) on H16 rejected" (do
    tamperM5aTaskReg baseExtH16.hwx baseExtH16.info 0 regNeCfg 0x10010082
  ) "neMatmulCfg")

  -- Tamper 2: Bad neCfg low bit (0x00010083)
  results := results.push (checkErr "m5d neg tamper #2: corrupted neCfg (0x00010083) rejected" (do
    tamperM5aTaskReg baseExtH16.hwx baseExtH16.info 0 regNeCfg 0x00010083
  ) "neMatmulCfg")

  -- Tamper 3: Missing H17 0x400 (h17Dyn) bit in kernelStage when M*B > 1
  results := results.push (checkErr "m5d neg tamper #3: missing H17 0x400 bit in kernelStage (0x5500) rejected" (do
    let fKs ← findM5aTaskRegOffset baseExtH17.hwx baseExtH17.info 0 regKernelStage
    let orig ← readU32LE baseExtH17.hwx fKs
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regKernelStage (orig &&& (~~~kernelStageH17DynBit))
  ) "h17Dyn")

  -- Tamper 4: Spurious H17 0x400 bit on H16 kernelStage
  results := results.push (checkErr "m5d neg tamper #4: spurious H17 0x400 bit on H16 kernelStage rejected" (do
    let fKs ← findM5aTaskRegOffset baseExtH16.hwx baseExtH16.info 0 regKernelStage
    let orig ← readU32LE baseExtH16.hwx fKs
    tamperM5aTaskReg baseExtH16.hwx baseExtH16.info 0 regKernelStage (orig ||| kernelStageH17DynBit)
  ) "h17Dyn")

  -- Tamper 5: Wrong common bit 29 (dynKern cleared when M*B > 1)
  results := results.push (checkErr "m5d neg tamper #5: clearing common bit 29 (dynKern) when M*B > 1 rejected" (do
    let fNc ← findM5aTaskRegOffset baseExtH17.hwx baseExtH17.info 0 regNeCommon
    let orig ← readU32LE baseExtH17.hwx fNc
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regNeCommon (orig &&& (~~~((1 : UInt32) <<< 29)))
  ) "dynKern")

  -- Tamper 6: Wrong common bit 2 (lo2 cleared when alignUp(N, 8) <= 64)
  results := results.push (checkErr "m5d neg tamper #6: wrong common lo2 bit rejected" (do
    let fNc ← findM5aTaskRegOffset baseExtH17.hwx baseExtH17.info 0 regNeCommon
    let orig ← readU32LE baseExtH17.hwx fNc
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regNeCommon (orig &&& (~~~((1 : UInt32) <<< 2)))
  ) "lo2")

  -- Tamper 7: Missing BAR slot-4 relocation on external-input chunkSlot[0] (plain 0x21 instead of slot-4)
  results := results.push (checkErr "m5d neg tamper #7: external-input chunkSlot[0] without BAR slot-4 relocation rejected" (do
    let textBytes ← readSlice baseExtH17.hwx baseExtH17.info.textSection.offset.toNat baseExtH17.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let td ← rebuildDynTask tasks[0]! 1 none fun ws =>
      ws.map fun w => if w.byteAddr == regChunkSlot 0 then { w with barSlot := none } else w
    let sA ← makeFp16Surface .h17 "a" "input_a" .input 16 1 32 4
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 4
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 16 1 32 4
    let em ← emitProgram { target := .h17, surfaces := #[sA, sB, sY], tasks := #[td] }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "BAR slot 4 config relocation")

  -- Tamper 8: Spurious BAR slot-4 relocation on scratch-produced chunkSlot[0] (0x41 with slot 4)
  results := results.push (checkErr "m5d neg tamper #8: scratch-produced chunkSlot[0] with BAR slot-4 relocation rejected" (do
    let textBytes ← readSlice baseProdH17.hwx baseProdH17.info.textSection.offset.toNat baseProdH17.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let td0 ← rebuildDynTask tasks[0]! 2
    let td1 ← rebuildDynTask tasks[1]! 2 none fun ws =>
      ws.map fun w => if w.byteAddr == regChunkSlot 0 then { w with barSlot := some barSlotConfigZero } else w
    let sA ← makeFp16Surface .h17 "a" "input_a" .input 32 1 8 1
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 1
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 8 1 32 1
    let scSec := baseProdH17.info.scratchSection.get!
    let em ← emitProgram {
      target := .h17, surfaces := #[sA, sB, sY], tasks := #[td0, td1],
      scratch := some { size := scSec.size, usable := scSec.size.toUInt32 }
    }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "plain register write")

  -- Tamper 9: Missing 10-word extended header on scratch-produced kernel (9-word header with 0x41)
  results := results.push (checkErr "m5d neg tamper #9: scratch-produced kernel missing 10-word extended header rejected" (do
    let textBytes ← readSlice baseProdH17.hwx baseProdH17.info.textSection.offset.toNat baseProdH17.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let td0 ← rebuildDynTask tasks[0]! 2
    let td1 ← rebuildDynTask tasks[1]! 2 (extProdOverride := some none)
    let sA ← makeFp16Surface .h17 "a" "input_a" .input 32 1 8 1
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 1
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 8 1 32 1
    let scSec := baseProdH17.info.scratchSection.get!
    let em ← emitProgram {
      target := .h17, surfaces := #[sA, sB, sY], tasks := #[td0, td1],
      scratch := some { size := scSec.size, usable := scSec.size.toUInt32 }
    }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "10-word extended header")

  -- Tamper 10: Spurious 10-word extended header on external-input kernel
  results := results.push (checkErr "m5d neg tamper #10: external-input kernel with spurious 10-word extended header rejected" (do
    let g2 ← Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .dram
      inputs := #[
        { name := "a", shape := { c := 8, h := 1, w := 32 }, shortName := "a" },
        { name := "b", shape := { c := 32, h := 1, w := 32 }, shortName := "b" }
      ]
      nodes := #[
        { id := "b_abs", inputs := #["b"], op := .pe1 .abs },
        { id := "mm",    inputs := #["a", "b_abs"], op := .matmul 1 false false f16PosOne }
      ]
    }
    let textBytes ← readSlice g2.hwx g2.info.textSection.offset.toNat g2.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let td0 ← rebuildDynTask tasks[0]! 2
    let td1 ← rebuildDynTask tasks[1]! 2 (extProdOverride := some (some 0))
    let sA ← makeFp16Surface .h17 "a" "input_a" .input 8 1 32 1
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 1
    let sY ← makeFp16Surface .h17 "y" "output_mm" .output 8 1 32 1
    let scSec := g2.info.scratchSection.get!
    let em ← emitProgram {
      target := .h17, surfaces := #[sA, sB, sY], tasks := #[td0, td1],
      scratch := some { size := scSec.size, usable := scSec.size.toUInt32 }
    }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "forbids 10-word extended header")

  -- Tamper 11: Spurious 10-word extended header on non-matmul task (neView)
  results := results.push (checkErr "m5d neg tamper #11: 10-word extended header on non-matmul (neView) task rejected" (do
    let textBytes ← readSlice baseProdH17.hwx baseProdH17.info.textSection.offset.toNat baseProdH17.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let td0 ← rebuildDynTask tasks[0]! 2 (extProdOverride := some (some 0))
    let td1 ← rebuildDynTask tasks[1]! 2
    let sA ← makeFp16Surface .h17 "a" "input_a" .input 32 1 8 1
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 1
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 8 1 32 1
    let scSec := baseProdH17.info.scratchSection.get!
    let em ← emitProgram {
      target := .h17, surfaces := #[sA, sB, sY], tasks := #[td0, td1],
      scratch := some { size := scSec.size, usable := scSec.size.toUInt32 }
    }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "expected 9-word header")

  -- Tamper 12: Invalid producerTaskId >= taskId in 10-word extended header (`hdr[9] = taskId`)
  results := results.push (checkErr "m5d neg tamper #12: extended header producerTaskId >= taskId rejected" (do
    let textBytes ← readSlice baseProdH17.hwx baseProdH17.info.textSection.offset.toNat baseProdH17.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let fHdr9 := baseProdH17.info.textSection.offset.toNat + tasks[1]!.startOffset + 4 * 9
    let badHwx ← putU32LE baseProdH17.hwx fHdr9 1
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasksSummary badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "must be < taskId")

  -- Tamper 13: Extended header producerTaskId pointing to a non-ordering earlier task (S5 [kernelRaw] hazard)
  results := results.push (checkErr "m5d neg tamper #13: wrong producerTaskId in extended header rejected by S5 [kernelRaw]" (do
    -- Task 0: pe1 abs on B (writes scratch slot 0), Task 1: pe1 abs on A (writes scratch slot 1),
    -- Task 2: matmul reads A from slot 1 (producer=1) and B from slot 0.
    -- Tamper task 2's extProducerTask from 1 down to 0: task 1 writes slot 1 AFTER task 0, so hbEq 1 0 is false!
    let g3 ← Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .dram
      inputs := #[
        { name := "a", shape := { c := 8, h := 1, w := 32 }, shortName := "a" },
        { name := "b", shape := { c := 32, h := 1, w := 32 }, shortName := "b" }
      ]
      nodes := #[
        { id := "b1", inputs := #["b"], op := .pe1 .abs },
        { id := "a1", inputs := #["a"], op := .pe1 .abs },
        { id := "mm", inputs := #["a1", "b1"], op := .matmul 1 false false f16PosOne }
      ]
    }
    let textBytes ← readSlice g3.hwx g3.info.textSection.offset.toNat g3.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let fHdr9 := g3.info.textSection.offset.toNat + tasks[2]!.startOffset + 4 * 9
    let badHwx ← putU32LE g3.hwx fHdr9 0
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasksSummary badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "[kernelRaw]")

  -- Tamper 14: Kernel DMA out-of-bounds (kernelBaseLo shifted past surface span)
  results := results.push (checkErr "m5d neg tamper #14: kernel DMA out-of-bounds (kernelBaseLo shifted) rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regKernelBaseLo 64
  ) "kernel DMA range")

  -- Tamper 15: Uninitialized BAR 3 scratch kernel read (kernelBaseLo pointed to unwritten scratch region)
  results := results.push (checkErr "m5d neg tamper #15: uninitialized BAR 3 scratch kernel read rejected by S5" (do
    -- In g3 (b1 writes [0, 2048), a1 writes [2048, 2560)): compile a 3-task graph with an extra unused
    -- tail in scratch by shifting mm's kernelBaseLo to a region never written by any task!
    let textBytes ← readSlice baseProdH17.hwx baseProdH17.info.textSection.offset.toNat baseProdH17.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let scSec := baseProdH17.info.scratchSection.get!
    -- Grow scratch by 0x1000 and point task 1 kernelBaseLo to scSec.size (unwritten!)
    let td0 ← rebuildDynTask tasks[0]! 2
    let td1 ← rebuildDynTask tasks[1]! 2 none fun ws =>
      ws.map fun w => if w.byteAddr == regKernelBaseLo then { w with value := scSec.size.toUInt32 } else w
    let sA ← makeFp16Surface .h17 "a" "input_a" .input 32 1 8 1
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 1
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 8 1 32 1
    let bigSc := scSec.size + 0x1000
    let em ← emitProgram {
      target := .h17, surfaces := #[sA, sB, sY], tasks := #[td0, td1],
      scratch := some { size := bigSc, usable := bigSc.toUInt32 }
    }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "uninitialized scratch")

  -- Tamper 16: Partially covered BAR 3 scratch kernel read (uninitialized gap in scratch kernel)
  results := results.push (checkErr "m5d neg tamper #16: partially covered BAR 3 scratch kernel read rejected by S5" (do
    let textBytes ← readSlice baseProdH17.hwx baseProdH17.info.textSection.offset.toNat baseProdH17.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let scSec := baseProdH17.info.scratchSection.get!
    -- Shift kernelBaseLo by +64B inside larger scratch so tail 64B is unwritten!
    let td0 ← rebuildDynTask tasks[0]! 2
    let td1 ← rebuildDynTask tasks[1]! 2 none fun ws =>
      ws.map fun w => if w.byteAddr == regKernelBaseLo then { w with value := 64 } else w
    let sA ← makeFp16Surface .h17 "a" "input_a" .input 32 1 8 1
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 1
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 8 1 32 1
    let bigSc := scSec.size + 0x1000
    let em ← emitProgram {
      target := .h17, surfaces := #[sA, sB, sY], tasks := #[td0, td1],
      scratch := some { size := bigSc, usable := bigSc.toUInt32 }
    }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "partially covered")

  -- Tamper 17: chunkSize > 65536 (64 KiB cap) rejected by verifier
  results := results.push (checkErr "m5d neg tamper #17: chunkSize > 65536 (Cin=32769) rejected by verifier" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regCin 32769
  ) "exceeds hardware limit 65536B")

  -- Tamper 18: Non-finite neScale (+Inf and NaN) rejected by verifier
  results := results.push (checkErr "m5d neg tamper #18a: non-finite neScale (+Inf 0x7c00) rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regNeScale 0x7c00
  ) "finite fp16")

  results := results.push (checkErr "m5d neg tamper #18b: non-finite neScale (NaN 0x7e00) rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regNeScale 0x7e00
  ) "finite fp16")

  -- Tamper 19: Wrong kernelBatchStride (0x5518) and kernelRoundStride (0x551c)
  results := results.push (checkErr "m5d neg tamper #19a: wrong kernelBatchStride (0x5518) rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regKernelBatchStride 64
  ) "kernelBatchStride")

  results := results.push (checkErr "m5d neg tamper #19b: wrong kernelRoundStride (0x551c) rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regKernelRoundStride 64
  ) "kernelRoundStride")

  -- Tamper 20: Missing disabled chunkSlot[15] (0x20) on taskId > 0 (carry-over check)
  results := results.push (checkErr "m5d neg tamper #20: omitted disabled chunkSlot[15] on matmul taskId > 0 rejected" (do
    let textBytes ← readSlice baseProdH17.hwx baseProdH17.info.textSection.offset.toNat baseProdH17.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let td0 ← rebuildDynTask tasks[0]! 2
    let td1 ← rebuildDynTask tasks[1]! 2 none fun ws =>
      ws.filter fun w => w.byteAddr != regChunkSlot 15
    let sA ← makeFp16Surface .h17 "a" "input_a" .input 32 1 8 1
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 1
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 8 1 32 1
    let scSec := baseProdH17.info.scratchSection.get!
    let em ← emitProgram {
      target := .h17, surfaces := #[sA, sB, sY], tasks := #[td0, td1],
      scratch := some { size := scSec.size, usable := scSec.size.toUInt32 }
    }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "carry-over")

  -- Tamper 21: kernelBase pointing to output surface rejected
  results := results.push (checkErr "m5d neg tamper #21: kernelBase pointing to output surface rejected" (do
    let textBytes ← readSlice baseExtH17.hwx baseExtH17.info.textSection.offset.toNat baseExtH17.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    -- Output surface y is BAR slot 8 (barSlotFirstSurface + 2)
    let td ← rebuildDynTask tasks[0]! 1 none fun ws =>
      ws.map fun w =>
        if w.byteAddr == regKernelBaseLo || w.byteAddr == regKernelBaseHi then
          { w with barSlot := some 8 }
        else w
    let sA ← makeFp16Surface .h17 "a" "input_a" .input 16 1 32 4
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 4
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 16 1 32 4
    let em ← emitProgram { target := .h17, surfaces := #[sA, sB, sY], tasks := #[td] }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "non-input surface")

  -- Tamper 22: Wrong chunkSlot encoding (0x41 on external input)
  results := results.push (checkErr "m5d neg tamper #22: chunkSlot 0x41 on external-input kernel rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 (regChunkSlot 0) 0x0041
  ) "must encode 0x21")

  -- Tamper 23: Cout > 16 on ocg = 0 neMatmul rejected (silicon 16-slot limit)
  results := results.push (checkErr "m5d neg tamper #23a: ocg=0 neMatmul Cout=17 (> 16) rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regCout 17
  ) "exceeds 16 chunk slots")

  results := results.push (checkErr "m5d neg tamper #23b: ocg=0 neMatmul Cout=32 (> 16) rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regCout 32
  ) "exceeds 16 chunk slots")

  -- Tamper 24: Bad neOp rejected
  results := results.push (checkErr "m5d neg tamper #24: bad neOp (0x00103c01) rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regNeOp 0x00103c01
  ) "neOp")

  -- Tamper 25: Invalid neTile (ocg != 0, fat != 0, WU != 0) rejected
  results := results.push (checkErr "m5d neg tamper #25a: neTile ocg=1 rejected" (do
    let w ← encodeWord neTileLayout #[("ocg", 1), ("fat", 0), ("WU", 0)]
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regNeTile w
  ) "ocg=0, fat=0, WU=0")

  results := results.push (checkErr "m5d neg tamper #25b: neTile fat=1 rejected" (do
    let w ← encodeWord neTileLayout #[("ocg", 0), ("fat", 1), ("WU", 0)]
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regNeTile w
  ) "ocg=0, fat=0, WU=0")

  results := results.push (checkErr "m5d neg tamper #25c: neTile WU=1 rejected" (do
    let w ← encodeWord neTileLayout #[("ocg", 0), ("fat", 0), ("WU", 1)]
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regNeTile w
  ) "ocg=0, fat=0, WU=0")

  -- Tamper 26: Under-provisioned 2^F < Mi rejected (M=16 with F=3, logNE=3 => 2^3=8 < 16)
  results := results.push (checkErr "m5d neg tamper #26: under-provisioned 2^F < Mi (F=3, M=16) rejected" (do
    let textBytes ← readSlice baseExtH17.hwx baseExtH17.info.textSection.offset.toNat baseExtH17.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let td0 ← rebuildDynTask tasks[0]! 1 none fun ws =>
      ws.map fun w =>
        if w.byteAddr == regNeCommon then
          let cleared := w.value &&& (~~~((7 : UInt32) <<< 19))
          { w with value := cleared ||| ((3 : UInt32) <<< 19) }
        else w
    let td := { td0 with eneOverride := some 4 }
    let sA ← makeFp16Surface .h17 "a" "input_a" .input 16 1 32 4
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 4
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 16 1 32 4
    let em ← emitProgram { target := .h17, surfaces := #[sA, sB, sY], tasks := #[td] }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "2^F=")

  -- Tamper 27: kernelBase targeting static __KERN_0 rejected
  results := results.push (checkErr "m5d neg tamper #27: kernelBase targeting __KERN_0 rejected" (do
    let textBytes ← readSlice baseExtH17.hwx baseExtH17.info.textSection.offset.toNat baseExtH17.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    -- With 3 surfaces (BAR slots 6, 7, 8) and a static kernel section, __KERN_0 gets BAR slot 9.
    let td ← rebuildDynTask tasks[0]! 1 none fun ws =>
      ws.map fun w =>
        if w.byteAddr == regKernelBaseLo || w.byteAddr == regKernelBaseHi then
          { w with barSlot := some 9 }
        else w
    let sA ← makeFp16Surface .h17 "a" "input_a" .input 16 1 32 4
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 4
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 16 1 32 4
    let em ← emitProgram {
      target := .h17, surfaces := #[sA, sB, sY], tasks := #[td],
      kernel := some (zeroBytes 64)
    }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "static __KERN_0")

  -- Tamper 28a & 28b: Mach-O __KERN_0 relocation on neMatmul (kernelBaseLo and non-kernelBaseLo reg)
  -- Use a 2-task conv1x1 -> matmul graph so __KERN_0 is legitimately present & referenced by task 0,
  -- then attach a spurious Mach-O __KERN_0 reloc onto task 1 (neMatmul).
  let baseConvMm ← match Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .dram
      inputs := #[
        { name := "x", shape := { c := 8, h := 1, w := 32 }, shortName := "x" },
        { name := "b", shape := { c := 32, h := 1, w := 32 }, shortName := "b" }
      ]
      nodes := #[
        { id := "c0", inputs := #["x"], op := .conv1x1 8 8 (mkM5aW1x1 8 8 1) none .none },
        { id := "mm", inputs := #["c0", "b"], op := .matmul 1 false false f16PosOne }
      ]
    } with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"baseConvMm setup failed: {e}")

  let rebuildConvMmWithMmReloc (targetReg : Nat) : Except String Unit := do
    let textBytes ← readSlice baseConvMm.hwx baseConvMm.info.textSection.offset.toNat baseConvMm.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let td0Base ← rebuildDynTask tasks[0]! 2
    let some c0KernIdx := td0Base.barWrites.findIdx? (fun br => br.baseByteAddr == regKernelBaseLo)
      | throw "missing conv task kernelBaseLo bar write"
    let td0 := { td0Base with kernelRelocs := #[{ barRecIndex := c0KernIdx, payloadWordIndex := 0 }] }
    let td1Base ← rebuildDynTask tasks[1]! 2
    let some relIdx := td1Base.barWrites.findIdx? (fun br => br.baseByteAddr == targetReg)
      | throw s!"missing matmul target reg {toHex targetReg} bar write"
    let td1 := { td1Base with kernelRelocs := #[{ barRecIndex := relIdx, payloadWordIndex := 0 }] }
    let sX ← makeFp16Surface .h17 "x" "input_x" .input 8 1 32 1
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 1
    let sY ← makeFp16Surface .h17 "y" "output_mm" .output 8 1 32 1
    let kSec := baseConvMm.info.kernelSection.get!
    let kBytes ← readSlice baseConvMm.hwx kSec.offset.toNat kSec.size.toNat
    let scSec := baseConvMm.info.scratchSection.get!
    let em ← emitProgram {
      target := .h17, surfaces := #[sX, sB, sY], tasks := #[td0, td1],
      kernel := some kBytes,
      scratch := some { size := scSec.size, usable := scSec.size.toUInt32 }
    }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()

  results := results.push (checkErr "m5d neg tamper #28a: Mach-O __KERN_0 reloc on neMatmul kernelBaseLo rejected" (
    rebuildConvMmWithMmReloc regKernelBaseLo
  ) "Mach-O __KERN_0 relocation")

  results := results.push (checkErr "m5d neg tamper #28b: Mach-O __KERN_0 reloc on neMatmul non-kernelBaseLo reg rejected" (
    rebuildConvMmWithMmReloc regSrc1BaseLo
  ) "Mach-O __KERN_0 relocation")

  -- Tamper 29: Misaligned kernelBaseLo % 64 != 0 rejected
  results := results.push (checkErr "m5d neg tamper #29: misaligned kernelBaseLo (32 % 64 != 0) rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regKernelBaseLo 32
  ) "not 64-byte aligned")

  -- Tamper 30: Spurious barSlot on non-BAR register rejected
  results := results.push (checkErr "m5d neg tamper #30: spurious barSlot on non-BAR reg (neScale) rejected" (do
    let textBytes ← readSlice baseExtH17.hwx baseExtH17.info.textSection.offset.toNat baseExtH17.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let td ← rebuildDynTask tasks[0]! 1 none fun ws =>
      ws.map fun w => if w.byteAddr == regNeScale then { w with barSlot := some barSlotConfigZero } else w
    let sA ← makeFp16Surface .h17 "a" "input_a" .input 16 1 32 4
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 4
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 16 1 32 4
    let em ← emitProgram { target := .h17, surfaces := #[sA, sB, sY], tasks := #[td] }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "unexpected BAR slot")

  -- Tamper 31: keepLive=1 (0xc1) on final task rejected
  results := results.push (checkErr "m5d neg tamper #31: keepLive=1 (0xc1) on final scratch-produced task rejected" (do
    let fSlot ← findM5aTaskRegOffset baseProdH17.hwx baseProdH17.info 1 (regChunkSlot 0)
    let orig ← readU32LE baseProdH17.hwx fSlot
    tamperM5aTaskReg baseProdH17.hwx baseProdH17.info 1 (regChunkSlot 0) (orig ||| ((1 : UInt32) <<< 7))
  ) "cannot set keepLive")

  -- Tamper 32a & 32b: Wrong chunkOffset[s] and wrong chunkSize[s]
  results := results.push (checkErr "m5d neg tamper #32a: wrong chunkOffset[1] rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 (regChunkOffset 1) 128
  ) "chunkOffset[1]")

  results := results.push (checkErr "m5d neg tamper #32b: wrong chunkSize[0] rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 (regChunkSize 0) 128
  ) "chunkSize[0]")

  -- Tamper 33: Offset/size written on inactive chunk slot (slot 8 when M=8) rejected
  results := results.push (checkErr "m5d neg tamper #33: chunkOffset on inactive slot 8 (M=8) rejected" (do
    let textBytes ← readSlice baseProdH17.hwx baseProdH17.info.textSection.offset.toNat baseProdH17.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let td0 ← rebuildDynTask tasks[0]! 2
    let td1 ← rebuildDynTask tasks[1]! 2 none fun ws =>
      ws.push { byteAddr := regChunkOffset 8, value := 512, barSlot := none, payloadOffset := 0 }
    let sA ← makeFp16Surface .h17 "a" "input_a" .input 32 1 8 1
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 1
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 8 1 32 1
    let scSec := baseProdH17.info.scratchSection.get!
    let em ← emitProgram {
      target := .h17, surfaces := #[sA, sB, sY], tasks := #[td0, td1],
      scratch := some { size := scSec.size, usable := scSec.size.toUInt32 }
    }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "inactive chunk slot 8")

  -- Tamper 34: neMatmul L2 overflow (> 2 MiB) rejected
  results := results.push (checkErr "m5d neg tamper #34: neMatmul L2 resBase past 2 MiB rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regResBase 0x001fffc0
  ) "exceeds 2 MiB")

  -- Tamper 35a & 35b: src1 DMA OOB and res DMA OOB rejected
  results := results.push (checkErr "m5d neg tamper #35a: neMatmul src1 DMA out-of-bounds rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regSrc1BaseLo 64
  ) "input DMA range")

  results := results.push (checkErr "m5d neg tamper #35b: neMatmul res DMA out-of-bounds rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regResBaseLo 64
  ) "output DMA range")

  -- Tamper 36: Intra-task BAR 3 scratch read/write overlap rejected
  results := results.push (checkErr "m5d neg tamper #36: intra-task BAR 3 scratch kernel/res overlap rejected" (do
    let gIntra ← Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .dram
      inputs := #[
        { name := "a", shape := { c := 8, h := 1, w := 32 }, shortName := "a" },
        { name := "b", shape := { c := 32, h := 1, w := 32 }, shortName := "b" }
      ]
      nodes := #[
        { id := "a1",  inputs := #["a"], op := .pe1 .abs },
        { id := "mm",  inputs := #["a1", "b"], op := .matmul 1 false false f16PosOne },
        { id := "out", inputs := #["mm"], op := .pe1 .abs }
      ]
    }
    let fKern ← findM5aTaskRegOffset gIntra.hwx gIntra.info 1 regKernelBaseLo
    let kernOff ← readU32LE gIntra.hwx fKern
    tamperM5aTaskReg gIntra.hwx gIntra.info 1 regResBaseLo kernOff
  ) "intra-task BAR 3 scratch overlap")

  -- Tamper 37: Multi-batch consumer reading single-batch scratch producer with depKind=2 rejected ([multiBatchRaw])
  results := results.push (checkErr "m5d neg tamper #37: multi-batch matmul consumer depKind=2 rejected by [multiBatchRaw]" (do
    let gMbCons ← Vane.Graph.compileGraph {
      target := .h17, defaultPlacement := .dram
      inputs := #[
        { name := "a", shape := { c := 32, h := 1, w := 32 }, shortName := "a" },
        { name := "b", shape := { c := 128, h := 1, w := 32 }, shortName := "b" }
      ]
      nodes := #[
        { id := "b1", inputs := #["b"], op := .pe1 .abs },
        { id := "mm", inputs := #["a", "b1"], op := .matmul 4 false false f16PosOne }
      ]
    }
    let wBadDep ← encodeWord (srcDepMultiLayout "src1Dep") #[
      ("code", srcDepCodeDma), ("task", 1), ("depDist", 0), ("depKind", 2)
    ]
    tamperM5aTaskReg gMbCons.hwx gMbCons.info 1 regSrc1Dep wBadDep
  ) "[multiBatchRaw]")

  -- Tamper 38: Coalesced 2-word BAR-4 record for chunkSlot[0..1] rejected (single-word granularity)
  results := results.push (checkErr "m5d neg tamper #38: multi-word BAR-4 chunkSlot[0..1] record rejected" (do
    let textBytes ← readSlice baseExtH17.hwx baseExtH17.info.textSection.offset.toNat baseExtH17.info.textSection.size.toNat
    let tasks ← decodeTaskStream .h17 textBytes
    let td0 ← rebuildDynTask tasks[0]! 1
    let some w0 := td0.barWrites.find? (fun br => br.baseByteAddr == regChunkSlot 0)
      | throw "missing chunkSlot[0]"
    let some w1 := td0.barWrites.find? (fun br => br.baseByteAddr == regChunkSlot 1)
      | throw "missing chunkSlot[1]"
    let filtered := td0.barWrites.filter fun br =>
      br.baseByteAddr != regChunkSlot 0 && br.baseByteAddr != regChunkSlot 1
    let coalesced : BarRegWrite := {
      baseByteAddr := regChunkSlot 0
      barSlot := barSlotConfigZero
      values := #[w0.values[0]!, w1.values[0]!]
    }
    let td := { td0 with barWrites := filtered.push coalesced }
    let sA ← makeFp16Surface .h17 "a" "input_a" .input 16 1 32 4
    let sB ← makeFp16Surface .h17 "b" "input_b" .input 32 1 32 4
    let sY ← makeFp16Surface .h17 "y" "output_y" .output 16 1 32 4
    let em ← emitProgram { target := .h17, surfaces := #[sA, sB, sY], tasks := #[td] }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()
  ) "single-word BAR slot 4")

  -- Tamper 39: Invalid resDmaNStride (< M * rowBN) rejected
  results := results.push (checkErr "m5d neg tamper #39: invalid resDmaNStride (< M * rowBN) rejected" (do
    tamperM5aTaskReg baseExtH17.hwx baseExtH17.info 0 regResDmaNStride 64
  ) "resDmaNStride")

  pure results

-- ============================================================================
-- Milestone 7B: Lean-Native SafeTensors, ONNX Proto3 & Transformer Frontend Tests
-- ============================================================================

private def pbVI (name : String) (dims : Array Int) (symBatch : Bool := false) : ProtoBuilder := Id.run do
  let mut sh := ProtoBuilder.empty
  for i in [:dims.size] do
    let mut db := ProtoBuilder.empty
    if i == 0 && symBatch then
      db := db.writeString 2 "batch"
    else
      db := db.writeInt64 1 dims[i]!
    sh := sh.writeMessage 1 db
  let tt := (ProtoBuilder.empty.writeInt64 1 onnxFloat).writeMessage 2 sh
  let tp := ProtoBuilder.empty.writeMessage 1 tt
  return (ProtoBuilder.empty.writeString 1 name).writeMessage 2 tp

private def pbF32Tensor (name : String) (dims : Array Int) (vals : Array Float) (useRaw : Bool := false) : ProtoBuilder := Id.run do
  let mut tb := (ProtoBuilder.empty.writeString 8 name).writeInt64 2 onnxFloat
  tb := tb.writePackedInt64 1 dims
  if useRaw then
    let mut raw := ByteArray.emptyWithCapacity (4 * vals.size)
    for v in vals do
      raw := pushU32LE raw (floatToF32Bits v)
    tb := tb.writeBytes 9 raw
  else
    tb := tb.writePackedFloat32 4 vals
  return tb

private def pbI64Tensor (name : String) (dims : Array Int) (vals : Array Int) : ProtoBuilder :=
  (((ProtoBuilder.empty.writeString 8 name).writeInt64 2 onnxInt64).writePackedInt64 1 dims).writePackedInt64 7 vals

private def pbF16Int32Tensor (name : String) (dims : Array Int) (vals : Array UInt16) : ProtoBuilder :=
  (((ProtoBuilder.empty.writeString 8 name).writeInt64 2 onnxFloat16).writePackedInt64 1 dims).writePackedInt32 5
    (vals.map fun u => (u.toNat : Int))

private def pbExternalTensor (name : String) (dims : Array Int) : ProtoBuilder :=
  (((ProtoBuilder.empty.writeString 8 name).writeInt64 2 onnxFloat).writePackedInt64 1 dims).writeInt64 14 onnxDataExternal

private def pbAttrInt (name : String) (v : Int) : ProtoBuilder :=
  ((ProtoBuilder.empty.writeString 1 name).writeInt64 3 v).writeInt64 20 attrInt

private def pbAttrInts (name : String) (vals : Array Int) : ProtoBuilder :=
  ((ProtoBuilder.empty.writeString 1 name).writePackedInt64 8 vals).writeInt64 20 attrInts

private def pbAttrFloat (name : String) (v : Float) : ProtoBuilder :=
  ((ProtoBuilder.empty.writeString 1 name).writeFloat32 2 v).writeInt64 20 attrFloat

private def pbAttrString (name : String) (s : String) : ProtoBuilder :=
  ((ProtoBuilder.empty.writeString 1 name).writeString 4 s).writeInt64 20 attrString

private def pbAttrTensor (name : String) (tb : ProtoBuilder) : ProtoBuilder :=
  ((ProtoBuilder.empty.writeString 1 name).writeMessage 5 tb).writeInt64 20 attrTensor

private def pbNode (op name : String) (ins outs : Array String) (attrs : Array ProtoBuilder := #[]) : ProtoBuilder := Id.run do
  let mut nb := (ProtoBuilder.empty.writeString 4 op).writeString 3 name
  for s in ins do nb := nb.writeString 1 s
  for s in outs do nb := nb.writeString 2 s
  for a in attrs do nb := nb.writeMessage 5 a
  return nb

private def pbWrapModel (opset : Int) (gb : ProtoBuilder) : ByteArray :=
  let op := (ProtoBuilder.empty.writeString 1 "").writeInt64 2 opset
  let mb := ((ProtoBuilder.empty.writeInt64 1 9).writeMessage 8 op).writeMessage 7 gb
  mb.bytes

private def pbGraph (name : String) (inputs outputs : Array ProtoBuilder)
    (inits nodes : Array ProtoBuilder) : ProtoBuilder := Id.run do
  let mut gb := ProtoBuilder.empty.writeString 2 name
  for vi in inputs do gb := gb.writeMessage 11 vi
  for vi in outputs do gb := gb.writeMessage 12 vi
  for t in inits do gb := gb.writeMessage 5 t
  for nd in nodes do gb := gb.writeMessage 1 nd
  return gb

private def mkF32Bytes (vals : Array Float) : ByteArray := Id.run do
  let mut bs := ByteArray.emptyWithCapacity (4 * vals.size)
  for v in vals do
    bs := pushU32LE bs (floatToF32Bits v)
  return bs

private def mkF16Bytes (vals : Array Float) : ByteArray := Id.run do
  let mut bs := ByteArray.emptyWithCapacity (2 * vals.size)
  for v in vals do
    bs := pushU16LE bs (floatToF16 v)
  return bs

private def mkBF16Bytes (vals : Array Float) : ByteArray := Id.run do
  let mut bs := ByteArray.emptyWithCapacity (2 * vals.size)
  for v in vals do
    let u32 := floatToF32Bits v
    bs := pushU16LE bs ((u32 >>> 16) &&& 0xffff).toUInt16
  return bs

private def mkF64Bytes (vals : Array Float) : ByteArray := Id.run do
  let mut bs := ByteArray.emptyWithCapacity (8 * vals.size)
  for v in vals do
    bs := pushU64LE bs v.toBits
  return bs

private def mkI64Bytes (vals : Array Int) : ByteArray := Id.run do
  let mut bs := ByteArray.emptyWithCapacity (8 * vals.size)
  for v in vals do
    bs := pushU64LE bs (int64ToU64 v)
  return bs

private def mkI32Bytes (vals : Array Int) : ByteArray := Id.run do
  let mut bs := ByteArray.emptyWithCapacity (4 * vals.size)
  for v in vals do
    bs := pushU32LE bs (int32ToU32 v)
  return bs

private def mkDetF32 (count : Nat) (scale : Float := 0.05) : Array Float :=
  Array.ofFn (n := count) fun i =>
    let k := (i.val % 7 : Nat)
    (Float.ofNat k - 3.0) * scale

private def verifyCompiledOnnx (co : CompiledOnnx) : Except String TaskVerifySummary := do
  validateSections co.compiled.hwx co.compiled.sections
  let s ← verifyHwxTasksSummary co.compiled.hwx co.compiled.info (requireKnownTask := true)
  if s.total != co.compiled.taskCount then
    throw s!"verified task count {s.total} != compiled taskCount {co.compiled.taskCount}"
  pure s

private def runM7bTests : IO (Array TestResult) := do
  let mut results : Array TestResult := #[]

  -- 1. SafeTensors round-trip & dtype decoding (F16, BF16, F32, F64, I64, I32, BOOL)
  results := results.push (checkOk "m7b safetensors: write/parse round-trip across F16, BF16, F32, F64, I64, I32, BOOL" (do
    let blob ← writeSafeTensors #[
      { name := "w_f16", dtype := "F16", shape := #[2, 2], data := mkF16Bytes #[0.5, -1.0, 2.0, 0.25] },
      { name := "w_bf16", dtype := "BF16", shape := #[2], data := mkBF16Bytes #[1.5, -0.5] },
      { name := "w_f32", dtype := "F32", shape := #[2], data := mkF32Bytes #[0.25, -3.5] },
      { name := "w_f64", dtype := "F64", shape := #[2], data := mkF64Bytes #[4.0, -0.125] },
      { name := "idx_i64", dtype := "I64", shape := #[3], data := mkI64Bytes #[-1, 0, 42] },
      { name := "idx_i32", dtype := "I32", shape := #[2], data := mkI32Bytes #[-7, 13] },
      { name := "mask_bool", dtype := "BOOL", shape := #[2], data := ByteArray.mk #[0, 1] }
    ] #[("format", "pt")]
    let sf ← parseSafeTensors blob { strict := true }
    if sf.tensors.size != 7 then throw s!"expected 7 tensors, got {sf.tensors.size}"
    let f16v ← sf.toFp16 "w_f16"
    if f16v != #[floatToF16 0.5, floatToF16 (-1.0), floatToF16 2.0, floatToF16 0.25] then
      throw "w_f16 mismatch"
    let bf16v ← sf.toFp16 "w_bf16"
    if bf16v != #[floatToF16 1.5, floatToF16 (-0.5)] then throw "w_bf16 mismatch"
    let f32v ← sf.toFp16 "w_f32"
    if f32v != #[floatToF16 0.25, floatToF16 (-3.5)] then throw "w_f32 mismatch"
    let f64v ← sf.toFp16 "w_f64"
    if f64v != #[floatToF16 4.0, floatToF16 (-0.125)] then throw "w_f64 mismatch"
    let i64v ← sf.toInt64 "idx_i64"
    if i64v != #[-1, 0, 42] then throw s!"i64 mismatch: {i64v.toList}"
    let i32v ← sf.toInt64 "idx_i32"
    if i32v != #[-7, 13] then throw s!"i32 mismatch: {i32v.toList}"
    let bv ← sf.toBool "mask_bool"
    if bv != #[false, true] then throw "bool mismatch"
    pure ()
  ))

  -- 2. SafeTensors negative tests
  results := results.push (checkErr "m7b neg safetensors: truncated header (< 8B) rejected" (
    parseSafeTensors (ByteArray.mk #[1, 2, 3])
  ) "truncated")

  results := results.push (checkErr "m7b neg safetensors: header_len == 0 rejected" (
    parseSafeTensors (pushU64LE ByteArray.empty 0)
  ) "header_len must be > 0")

  results := results.push (checkErr "m7b neg safetensors: header_len > 100 MiB rejected" (
    parseSafeTensors (pushU64LE ByteArray.empty (maxHeaderBytes + 1).toUInt64)
  ) "exceeds limit")

  results := results.push (checkErr "m7b neg safetensors: strict mode rejects overlapping spans" (do
    let hdr := "{\"a\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[2,6]}}"
    let hBytes := hdr.toUTF8
    let buf := (pushU64LE ByteArray.empty hBytes.size.toUInt64) ++ hBytes ++ zeroBytes 6
    parseSafeTensors buf { strict := true }
  ) "overlapping")

  results := results.push (checkErr "m7b neg safetensors: span != shape * dtypeBytes rejected" (do
    let hdr := "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}"
    let hBytes := hdr.toUTF8
    let buf := (pushU64LE ByteArray.empty hBytes.size.toUInt64) ++ hBytes ++ zeroBytes 4
    parseSafeTensors buf
  ) "byte span")

  results := results.push (checkErr "m7b neg safetensors: NaN weight rejected on toFp16" (do
    let mut badF16 := ByteArray.empty
    badF16 := pushU16LE badF16 f16QNaN
    let blob ← writeSafeTensors #[{ name := "nan_w", dtype := "F16", shape := #[1], data := badF16 }]
    let sf ← parseSafeTensors blob
    sf.toFp16 "nan_w"
  ) "non-finite")

  results := results.push (checkErr "m7b neg safetensors: F32 overflow (> 65504) rejected on toFp16" (do
    let blob ← writeSafeTensors #[{ name := "big_w", dtype := "F32", shape := #[1], data := mkF32Bytes #[100000.0] }]
    let sf ← parseSafeTensors blob
    sf.toFp16 "big_w"
  ) "overflows fp16")

  -- 3. ONNX Proto3 decoder unit & negative tests (including FLOAT16 in int32_data, depth > 32, bad wire types)
  results := results.push (checkOk "m7b onnx proto3: FLOAT16 int32_data + raw_data + packed/unpacked fields" (do
    let tF16 := pbF16Int32Tensor "w16" #[2] #[floatToF16 1.5, floatToF16 (-0.25)]
    let tRaw := pbF32Tensor "w32" #[2] #[0.5, 2.0] (useRaw := true)
    let gb := pbGraph "g" #[pbVI "x" #[1, 2, 1, 1]] #[pbVI "y" #[1, 2, 1, 1]] #[tF16, tRaw] #[]
    let m ← parseOnnx (pbWrapModel 17 gb)
    if m.defaultOpsetVersion != 17 then throw "opset mismatch"
    let f16Vals ← m.graph.initializers[0]!.toFloat64
    if f16Vals != #[1.5, -0.25] then throw s!"FLOAT16 int32_data mismatch: {f16Vals.toList}"
    let f32Vals ← m.graph.initializers[1]!.toFloat64
    if f32Vals != #[0.5, 2.0] then throw s!"FLOAT raw_data mismatch: {f32Vals.toList}"
    pure ()
  ))

  results := results.push (checkErr "m7b neg onnx proto3: unsupported wire type 3 rejected" (
    parseOnnx (ByteArray.mk #[((1 : UInt8) <<< 3) ||| 3])
  ) "unsupported wire type")

  results := results.push (checkErr "m7b neg onnx proto3: truncated varint rejected" (
    parseOnnx (ByteArray.mk #[0x08, 0x80])
  ) "truncated varint")

  results := results.push (checkErr "m7b neg onnx proto3: missing graph rejected" (
    parseOnnx (ProtoBuilder.empty.writeInt64 1 9).bytes
  ) "missing graph")

  results := results.push (checkErr "m7b neg onnx proto3: nesting depth > 32 rejected" (do
    -- Nest GraphProto -> NodeProto -> AttributeProto.g -> GraphProto 15 times (> 32 message levels)
    let mut g := ProtoBuilder.empty.writeString 2 "leaf"
    for _ in [:15] do
      let attr := (ProtoBuilder.empty.writeString 1 "body").writeMessage 6 g
      let nd := (ProtoBuilder.empty.writeString 4 "If").writeMessage 5 attr
      g := ProtoBuilder.empty.writeMessage 1 nd
    let m := (ProtoBuilder.empty.writeInt64 1 9).writeMessage 7 g
    parseOnnx m.bytes
  ) "recursion depth limit")

  results := results.push (checkErr "m7b neg onnx: EXTERNAL data_location without SafeTensors override rejected" (do
    let gb := pbGraph "ext_g" #[pbVI "x" #[1, 4, 4, 4]] #[pbVI "y" #[1, 4, 4, 4]]
      #[pbExternalTensor "w" #[4, 4, 1, 1]]
      #[pbNode "Conv" "c0" #["x", "w"] #["y"]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "EXTERNAL")

  -- 4. ResNet residual block (Conv3x3 + BN + Relu + Conv1x1 + Add + Identity + Dropout + symbolic input shape)
  let resnetOnnxBytes : ByteArray :=
    let w1 := pbF32Tensor "w1" #[16, 16, 3, 3] (mkDetF32 (16 * 16 * 9) 0.03)
    let bnS := pbF32Tensor "bn_s" #[16] (Array.replicate 16 1.25)
    let bnB := pbF32Tensor "bn_b" #[16] (Array.replicate 16 0.1)
    let bnM := pbF32Tensor "bn_m" #[16] (Array.replicate 16 0.05)
    let bnV := pbF32Tensor "bn_v" #[16] (Array.replicate 16 0.9)
    let w2 := pbF32Tensor "w2" #[16, 16, 1, 1] (mkDetF32 (16 * 16) 0.04)
    let b2 := pbF32Tensor "b2" #[16] (Array.replicate 16 0.02)
    let gb := pbGraph "resnet"
      #[pbVI "x" #[1, 16, 8, 8] (symBatch := true)]
      #[pbVI "y" #[1, 16, 8, 8]]
      #[w1, bnS, bnB, bnM, bnV, w2, b2]
      #[
        pbNode "Identity" "id0" #["x"] #["x_id"],
        pbNode "Conv" "c1" #["x_id", "w1"] #["c1_out"] #[pbAttrInts "pads" #[1, 1, 1, 1]],
        pbNode "BatchNormalization" "bn1" #["c1_out", "bn_s", "bn_b", "bn_m", "bn_v"] #["bn1_out"] #[pbAttrFloat "epsilon" 1e-5],
        pbNode "Relu" "r1" #["bn1_out"] #["r1_out"],
        pbNode "Dropout" "dr1" #["r1_out"] #["dr1_out", "dr1_mask"],
        pbNode "Conv" "c2" #["dr1_out", "w2", "b2"] #["c2_out"],
        pbNode "Add" "add0" #["c2_out", "x_id"] #["y"]
      ]
    pbWrapModel 17 gb

  for tgt in #[Target.h16, Target.h17] do
    results := results.push (checkOk s!"m7b ({tgt.name}): ResNet residual (Conv3x3+BN+Relu+Conv1x1+Add, fused & --no-fuse)" (do
      let m ← parseOnnx resnetOnnxBytes
      let co ← compileOnnx m { target := tgt, inputShapes := #[("x", { c := 16, h := 8, w := 8 })] }
      let _ ← verifyCompiledOnnx co
      if co.lowered.summary.fusedBnCount != 1 then throw s!"expected fusedBnCount=1, got {co.lowered.summary.fusedBnCount}"
      if co.lowered.summary.fusedActCount != 1 then throw s!"expected fusedActCount=1, got {co.lowered.summary.fusedActCount}"
      if co.lowered.summary.elidedCount != 2 then throw s!"expected elidedCount=2, got {co.lowered.summary.elidedCount}"
      let coNoFuse ← compileOnnx m { target := tgt, inputShapes := #[("x", { c := 16, h := 8, w := 8 })], noFuse := true }
      let _ ← verifyCompiledOnnx coNoFuse
      if coNoFuse.lowered.summary.fusedBnCount != 0 || coNoFuse.lowered.summary.fusedActCount != 0 then
        throw "expected 0 fusions with noFuse=true"
      pure ()
    ))

  -- 5. Pool + Reduce + Softmax + Log + Mul(x,Sigmoid(x)) -> Silu
  let poolRedSoftBytes : ByteArray :=
    let gb := pbGraph "pool_red_soft"
      #[pbVI "x" #[1, 8, 8, 8]]
      #[pbVI "y" #[1, 8, 1, 1]]
      #[]
      #[
        pbNode "Sigmoid" "sig" #["x"] #["x_sig"],
        pbNode "Mul" "silu_mul" #["x", "x_sig"] #["x_silu"],
        pbNode "MaxPool" "mp" #["x_silu"] #["mp_out"] #[pbAttrInts "kernel_shape" #[2, 2], pbAttrInts "strides" #[2, 2]],
        pbNode "AveragePool" "ap" #["mp_out"] #["ap_out"] #[pbAttrInts "kernel_shape" #[2, 2], pbAttrInts "strides" #[2, 2]],
        pbNode "GlobalAveragePool" "gap" #["ap_out"] #["gap_out"],
        pbNode "Softmax" "sm" #["gap_out"] #["y"] #[pbAttrInt "axis" 1]
      ]
    pbWrapModel 17 gb

  for tgt in #[Target.h16, Target.h17] do
    results := results.push (checkOk s!"m7b ({tgt.name}): Silu-pattern + MaxPool + AvgPool + GAP + Softmax" (do
      let m ← parseOnnx poolRedSoftBytes
      let co ← compileOnnx m { target := tgt }
      let _ ← verifyCompiledOnnx co
      if co.lowered.summary.outputShape != { n := 1, c := 8, h := 1, w := 1 } then
        throw s!"unexpected output shape {repr co.lowered.summary.outputShape}"
      pure ()
    ))

  -- 6. MLP: Gemm + static MatMul + Add bias + Relu fusion + Log
  let mlpBytes : ByteArray :=
    let w1 := pbF32Tensor "w1" #[16, 8] (mkDetF32 (16 * 8) 0.05)
    let b1 := pbF32Tensor "b1" #[16] (Array.replicate 16 0.1)
    let w2 := pbF32Tensor "w2" #[16, 8] (mkDetF32 (16 * 8) 0.05)
    let b2 := pbF32Tensor "b2" #[8] (Array.replicate 8 0.05)
    let gb := pbGraph "mlp"
      #[pbVI "x" #[1, 8]]
      #[pbVI "y" #[1, 8]]
      #[w1, b1, w2, b2]
      #[
        pbNode "Gemm" "g1" #["x", "w1", "b1"] #["h1"] #[pbAttrInt "transB" 1],
        pbNode "Relu" "r1" #["h1"] #["h1r"],
        pbNode "MatMul" "mm2" #["h1r", "w2"] #["h2"],
        pbNode "Add" "a2" #["h2", "b2"] #["h2b"],
        pbNode "Sigmoid" "s2" #["h2b"] #["h2s"],
        pbNode "Log" "log_out" #["h2s"] #["y"]
      ]
    pbWrapModel 17 gb

  for tgt in #[Target.h16, Target.h17] do
    results := results.push (checkOk s!"m7b ({tgt.name}): MLP (Gemm transB=1 + Relu + static MatMul+Add+Sigmoid + Log)" (do
      let m ← parseOnnx mlpBytes
      let co ← compileOnnx m { target := tgt }
      let _ ← verifyCompiledOnnx co
      if co.lowered.summary.fusedActCount != 2 then
        throw s!"expected 2 fused activations, got {co.lowered.summary.fusedActCount}"
      pure ()
    ))

  -- 7. Multi-Head SDPA: 4D (with Transpose + Div scale fusion & --no-fuse), 3D, and head-stacked
  let sdpa4DBytes : ByteArray :=
    let scaleT := pbF32Tensor "sqrt_d" #[] #[4.0]
    let gb := pbGraph "sdpa4d"
      #[pbVI "q" #[1, 4, 16, 16], pbVI "k" #[1, 4, 16, 16], pbVI "v" #[1, 4, 16, 16]]
      #[pbVI "y" #[1, 4, 16, 16]]
      #[scaleT]
      #[
        pbNode "Transpose" "kt" #["k"] #["k_t"] #[pbAttrInts "perm" #[0, 1, 3, 2]],
        pbNode "MatMul" "qk" #["q", "k_t"] #["scores_raw"],
        pbNode "Div" "scale" #["scores_raw", "sqrt_d"] #["scores"],
        pbNode "Softmax" "sm" #["scores"] #["probs"] #[pbAttrInt "axis" (-1)],
        pbNode "MatMul" "pv" #["probs", "v"] #["y"]
      ]
    pbWrapModel 17 gb

  let sdpa3DBytes : ByteArray :=
    let invScaleT := pbF32Tensor "inv_sqrt_d" #[] #[0.25]
    let gb := pbGraph "sdpa3d"
      #[pbVI "q" #[4, 16, 16], pbVI "k" #[4, 16, 16], pbVI "v" #[4, 16, 16]]
      #[pbVI "y" #[4, 16, 16]]
      #[invScaleT]
      #[
        pbNode "Transpose" "kt" #["k"] #["k_t"] #[pbAttrInts "perm" #[0, 2, 1]],
        pbNode "MatMul" "qk" #["q", "k_t"] #["scores_raw"],
        pbNode "Mul" "scale" #["scores_raw", "inv_sqrt_d"] #["scores"],
        pbNode "Softmax" "sm" #["scores"] #["probs"] #[pbAttrInt "axis" (-1)],
        pbNode "MatMul" "pv" #["probs", "v"] #["y"]
      ]
    pbWrapModel 17 gb

  let sdpaStackedBytes : ByteArray :=
    let gb := pbGraph "sdpa_stacked"
      #[pbVI "q" #[1, 16, 1, 16], pbVI "k" #[1, 16, 1, 16], pbVI "v" #[1, 16, 1, 16]]
      #[pbVI "y" #[1, 16, 1, 16]]
      #[]
      #[
        pbNode "Transpose" "kt" #["k"] #["k_t"] #[pbAttrInts "perm" #[0, 3, 2, 1]],
        pbNode "MatMul" "qk" #["q", "k_t"] #["scores"],
        pbNode "Softmax" "sm" #["scores"] #["probs"] #[pbAttrInt "axis" 3],
        pbNode "MatMul" "pv" #["probs", "v"] #["y"]
      ]
    pbWrapModel 17 gb

  for tgt in #[Target.h16, Target.h17] do
    results := results.push (checkOk s!"m7b ({tgt.name}): Multi-Head SDPA 4D (fused Transpose+Div scale & --no-fuse), 3D, and head-stacked" (do
      let m4 ← parseOnnx sdpa4DBytes
      let co4 ← compileOnnx m4 { target := tgt }
      let s4 ← verifyCompiledOnnx co4
      if co4.lowered.summary.fusedMatmulCount != 2 then
        throw s!"4D SDPA: expected fusedMatmulCount=2, got {co4.lowered.summary.fusedMatmulCount}"
      if s4.neMatmul < 2 then throw s!"4D SDPA: expected >= 2 matmul tasks, got {s4.neMatmul}"
      let co4NoFuse ← compileOnnx m4 { target := tgt, noFuse := true }
      let _ ← verifyCompiledOnnx co4NoFuse
      if co4NoFuse.lowered.summary.fusedMatmulCount != 0 then
        throw "4D SDPA --no-fuse: expected fusedMatmulCount=0"

      let m3 ← parseOnnx sdpa3DBytes
      let co3 ← compileOnnx m3 { target := tgt }
      let _ ← verifyCompiledOnnx co3
      if co3.lowered.summary.fusedMatmulCount != 2 then
        throw s!"3D SDPA: expected fusedMatmulCount=2, got {co3.lowered.summary.fusedMatmulCount}"

      let mSt ← parseOnnx sdpaStackedBytes
      let coSt ← compileOnnx mSt { target := tgt, allowViewOps := false }
      let _ ← verifyCompiledOnnx coSt
      pure ()
    ))

  -- 8. Complete Pre-LN Multi-Head Transformer Block with external `.safetensors` weights!
  let transformerBlockTest (tgt : Target) : Except String Unit := do
    -- C = 32, L = 16, 2 heads of D = 16, FFN = 64
    let sfBlob ← writeSafeTensors #[
      { name := "ln1_g", dtype := "F32", shape := #[16], data := mkF32Bytes (Array.replicate 16 1.0) },
      { name := "ln1_b", dtype := "F32", shape := #[16], data := mkF32Bytes (Array.replicate 16 0.0) },
      { name := "wq", dtype := "F16", shape := #[32, 32, 1, 1], data := mkF16Bytes (mkDetF32 (32 * 32) 0.04) },
      { name := "wk", dtype := "F16", shape := #[32, 32, 1, 1], data := mkF16Bytes (mkDetF32 (32 * 32) 0.03) },
      { name := "wv", dtype := "BF16", shape := #[32, 32, 1, 1], data := mkBF16Bytes (mkDetF32 (32 * 32) 0.03) },
      { name := "wo", dtype := "F32", shape := #[32, 32, 1, 1], data := mkF32Bytes (mkDetF32 (32 * 32) 0.03) },
      { name := "ln2_g", dtype := "F32", shape := #[16], data := mkF32Bytes (Array.replicate 16 1.0) },
      { name := "ln2_b", dtype := "F32", shape := #[16], data := mkF32Bytes (Array.replicate 16 0.0) },
      { name := "w_up", dtype := "F16", shape := #[64, 32, 1, 1], data := mkF16Bytes (mkDetF32 (64 * 32) 0.03) },
      { name := "w_dn", dtype := "F16", shape := #[32, 64, 1, 1], data := mkF16Bytes (mkDetF32 (32 * 64) 0.03) }
    ] #[("arch", "transformer_block")]
    let sf ← parseSafeTensors sfBlob { strict := true }

    let gb := pbGraph "transformer_block"
      #[pbVI "x" #[1, 32, 1, 16]]
      #[pbVI "y" #[1, 32, 1, 16]]
      #[
        pbExternalTensor "ln1_g" #[16], pbExternalTensor "ln1_b" #[16],
        pbExternalTensor "wq" #[32, 32, 1, 1], pbExternalTensor "wk" #[32, 32, 1, 1],
        pbExternalTensor "wv" #[32, 32, 1, 1], pbExternalTensor "wo" #[32, 32, 1, 1],
        pbExternalTensor "ln2_g" #[16], pbExternalTensor "ln2_b" #[16],
        pbExternalTensor "w_up" #[64, 32, 1, 1], pbExternalTensor "w_dn" #[32, 64, 1, 1],
        pbI64Tensor "sh_heads" #[4] #[1, 2, 16, 16],
        pbI64Tensor "sh_flat" #[4] #[1, 32, 1, 16],
        pbF32Tensor "sqrt_d" #[] #[4.0]
      ]
      #[
        pbNode "LayerNormalization" "ln1" #["x", "ln1_g", "ln1_b"] #["x1"] #[pbAttrInt "axis" (-1), pbAttrFloat "epsilon" 1e-5],
        pbNode "Conv" "q_proj" #["x1", "wq"] #["q0"],
        pbNode "Conv" "k_proj" #["x1", "wk"] #["k0"],
        pbNode "Conv" "v_proj" #["x1", "wv"] #["v0"],
        pbNode "Reshape" "q_rs" #["q0", "sh_heads"] #["q1"],
        pbNode "Reshape" "k_rs" #["k0", "sh_heads"] #["k1"],
        pbNode "Reshape" "v_rs" #["v0", "sh_heads"] #["v1"],
        -- [1, 2, D=16, L=16] -> [1, 2, L=16, D=16]
        pbNode "Transpose" "q_tr" #["q1"] #["q2"] #[pbAttrInts "perm" #[0, 1, 3, 2]],
        pbNode "Transpose" "v_tr" #["v1"] #["v2"] #[pbAttrInts "perm" #[0, 1, 3, 2]],
        -- Q[1,2,L,D] @ K[1,2,D,L] -> scores[1,2,L,L]
        pbNode "MatMul" "qk" #["q2", "k1"] #["sc0"],
        pbNode "Div" "sc_div" #["sc0", "sqrt_d"] #["sc1"],
        pbNode "Softmax" "sm" #["sc1"] #["probs"] #[pbAttrInt "axis" (-1)],
        -- probs[1,2,L,L] @ V[1,2,L,D] -> ctx[1,2,L,D]
        pbNode "MatMul" "pv" #["probs", "v2"] #["ctx0"],
        pbNode "Transpose" "ctx_tr" #["ctx0"] #["ctx1"] #[pbAttrInts "perm" #[0, 1, 3, 2]],
        pbNode "Reshape" "ctx_rs" #["ctx1", "sh_flat"] #["ctx2"],
        pbNode "Conv" "o_proj" #["ctx2", "wo"] #["attn_out"],
        pbNode "Add" "res1" #["x", "attn_out"] #["h1"],
        pbNode "LayerNormalization" "ln2" #["h1", "ln2_g", "ln2_b"] #["h2"] #[pbAttrInt "axis" (-1), pbAttrFloat "epsilon" 1e-5],
        pbNode "Conv" "ffn_up" #["h2", "w_up"] #["u0"],
        pbNode "Gelu" "ffn_gelu" #["u0"] #["u1"],
        pbNode "Conv" "ffn_dn" #["u1", "w_dn"] #["ffn_out"],
        pbNode "Add" "res2" #["h1", "ffn_out"] #["y"]
      ]
    let m ← parseOnnx (pbWrapModel 17 gb)
    let co ← compileOnnx m { target := tgt, safeTensors := some sf }
    let summary ← verifyCompiledOnnx co
    if co.lowered.summary.safeTensorsUsed != 10 then
      throw s!"expected safeTensorsUsed=10, got {co.lowered.summary.safeTensorsUsed}"
    if summary.neMatmul < 2 then
      throw s!"expected >= 2 matmul tasks in Transformer block, got {summary.neMatmul}"
    pure ()

  for tgt in #[Target.h16, Target.h17] do
    results := results.push (checkOk s!"m7b ({tgt.name}): Pre-LN Multi-Head Transformer Block with SafeTensors weights" (
      transformerBlockTest tgt
    ))

  -- 9. Unfused RMSNorm (Pow(2.0), Pow(-0.5)) + Pow(0.5), Pow(-1.0), Pow(1.0) + negative Pow(3.0)
  let rmsNormBytes : ByteArray :=
    let gb := pbGraph "rmsnorm"
      #[pbVI "x" #[1, 16, 1, 32]]
      #[pbVI "y" #[1, 16, 1, 32]]
      #[pbF32Tensor "two" #[] #[2.0], pbF32Tensor "eps" #[] #[1e-5], pbF32Tensor "neg_half" #[] #[-0.5], pbF32Tensor "one" #[] #[1.0]]
      #[
        pbNode "Pow" "id1" #["x", "one"] #["x1"],
        pbNode "Pow" "sq" #["x1", "two"] #["x2"],
        pbNode "ReduceMean" "ms" #["x2"] #["mean_sq"] #[pbAttrInts "axes" #[-1], pbAttrInt "keepdims" 1],
        pbNode "Add" "ms_eps" #["mean_sq", "eps"] #["ms_e"],
        pbNode "Pow" "inv_rms" #["ms_e", "neg_half"] #["r"],
        pbNode "Mul" "norm" #["x1", "r"] #["y"]
      ]
    pbWrapModel 17 gb

  for tgt in #[Target.h16, Target.h17] do
    results := results.push (checkOk s!"m7b ({tgt.name}): Unfused RMSNorm (Pow 2.0, -0.5, 1.0 elision)" (do
      let m ← parseOnnx rmsNormBytes
      let co ← compileOnnx m { target := tgt }
      let _ ← verifyCompiledOnnx co
      if co.lowered.summary.elidedCount != 1 then
        throw s!"expected Pow(1.0) elidedCount=1, got {co.lowered.summary.elidedCount}"
      pure ()
    ))

  results := results.push (checkOk "m7b (h17): Pow(0.5) -> sqrt and Pow(-1.0) -> inverse" (do
    let gb := pbGraph "pow_misc"
      #[pbVI "x" #[1, 8, 4, 8]]
      #[pbVI "y" #[1, 8, 4, 8]]
      #[pbF32Tensor "half" #[] #[0.5], pbF32Tensor "neg1" #[] #[-1.0]]
      #[
        pbNode "Pow" "p1" #["x", "half"] #["h"],
        pbNode "Pow" "p2" #["h", "neg1"] #["y"]
      ]
    let m ← parseOnnx (pbWrapModel 17 gb)
    let co ← compileOnnx m { target := .h17 }
    let _ ← verifyCompiledOnnx co
    pure ()
  ))

  results := results.push (checkErr "m7b neg: unsupported Pow(x, 3.0) rejected" (do
    let gb := pbGraph "pow_bad"
      #[pbVI "x" #[1, 8, 4, 8]]
      #[pbVI "y" #[1, 8, 4, 8]]
      #[pbF32Tensor "three" #[] #[3.0]]
      #[pbNode "Pow" "pbad" #["x", "three"] #["y"]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "unsupported Pow exponent")

  -- 10. U-Net Conv -> Split -> Conv -> Concat view graph (default views, opset <13, >=13, 18, and --allow-view-ops=false)
  let mkUNetSplitBytes (opset : Int) (splitMode : Nat) : ByteArray :=
    -- splitMode: 0 = opset <13 attr, 1 = opset >=13 const input, 2 = opset 18 default equal split
    let wIn := pbF32Tensor "w_in" #[16, 16, 1, 1] (mkDetF32 (16 * 16) 0.05)
    let wA := pbF32Tensor "w_a" #[8, 8, 3, 3] (mkDetF32 (8 * 8 * 9) 0.04)
    let wB := pbF32Tensor "w_b" #[8, 8, 1, 1] (mkDetF32 (8 * 8) 0.05)
    let inits :=
      if splitMode == 1 then #[wIn, wA, wB, pbI64Tensor "sp_sz" #[2] #[8, 8]]
      else #[wIn, wA, wB]
    let splitNode :=
      if splitMode == 0 then
        pbNode "Split" "sp" #["c0"] #["s0", "s1"] #[pbAttrInt "axis" 1, pbAttrInts "split" #[8, 8]]
      else if splitMode == 1 then
        pbNode "Split" "sp" #["c0", "sp_sz"] #["s0", "s1"] #[pbAttrInt "axis" 1]
      else
        pbNode "Split" "sp" #["c0"] #["s0", "s1"] #[pbAttrInt "axis" 1, pbAttrInt "num_outputs" 2]
    let gb := pbGraph "unet_split"
      #[pbVI "x" #[1, 16, 8, 8]]
      #[pbVI "y" #[1, 16, 8, 8]]
      inits
      #[
        pbNode "Conv" "c_in" #["x", "w_in"] #["c0"],
        splitNode,
        pbNode "Conv" "c_a" #["s0", "w_a"] #["a_out"] #[pbAttrInts "pads" #[1, 1, 1, 1]],
        pbNode "Conv" "c_b" #["s1", "w_b"] #["b_out"],
        pbNode "Concat" "cat" #["a_out", "b_out"] #["y"] #[pbAttrInt "axis" 1]
      ]
    pbWrapModel opset gb

  for tgt in #[Target.h16, Target.h17] do
    results := results.push (checkOk s!"m7b ({tgt.name}): U-Net Conv->Split->Conv->Concat (opset 11, 13, 18, default views)" (do
      for (opset, mode) in #[(11, 0), (13, 1), (18, 2)] do
        let m ← parseOnnx (mkUNetSplitBytes opset mode)
        let co ← compileOnnx m { target := tgt } -- allowViewOps := true by default!
        let s ← verifyCompiledOnnx co
        if s.neView < 4 then throw s!"expected >= 4 view tasks (2 split slices + 2 concat copies), got {s.neView}"
      pure ()
    ))

  results := results.push (checkErr "m7b neg: U-Net Split/Concat rejected when --allow-view-ops=false" (do
    let m ← parseOnnx (mkUNetSplitBytes 18 2)
    lowerOnnx m { allowViewOps := false }
  ) "--allow-view-ops=true")

  results := results.push (checkErr "m7b neg: W-axis Split with misaligned start offset (16 % 32 != 0) rejected" (do
    let gb := pbGraph "w_split_bad"
      #[pbVI "x" #[1, 4, 4, 32]]
      #[pbVI "y" #[1, 4, 4, 32]]
      #[]
      #[
        pbNode "Split" "sp" #["x"] #["s0", "s1"] #[pbAttrInt "axis" 3],
        pbNode "Concat" "cat" #["s0", "s1"] #["y"] #[pbAttrInt "axis" 3]
      ]
    let m ← parseOnnx (pbWrapModel 18 gb)
    lowerOnnx m {}
  ) "multiple of 32")

  -- 11. LayerNormalization along C (H=W=1, non-uniform affine) and along W (uniform & runtime affine)
  for tgt in #[Target.h16, Target.h17] do
    results := results.push (checkOk s!"m7b ({tgt.name}): LayerNormalization along C (H=W=1, non-uniform) and W (uniform & runtime)" (do
      -- Along C on [1, 16, 1, 1] (true ONNX suffix [16, 1, 1] semantics) with non-uniform constant scale & bias
      let scaleC := Array.ofFn (n := 16) fun i => 0.8 + 0.02 * Float.ofNat i.val
      let biasC := Array.ofFn (n := 16) fun i => -0.1 + 0.01 * Float.ofNat i.val
      for paramSh in #[#[16], #[16, 1, 1], #[1, 16, 1, 1]] do
        let gbC := pbGraph "ln_c"
          #[pbVI "x" #[1, 16, 1, 1]] #[pbVI "y" #[1, 16, 1, 1]]
          #[pbF32Tensor "g" paramSh scaleC, pbF32Tensor "b" paramSh biasC]
          #[pbNode "LayerNormalization" "ln" #["x", "g", "b"] #["y"] #[pbAttrInt "axis" 1, pbAttrFloat "epsilon" 1e-5]]
        let mC ← parseOnnx (pbWrapModel 17 gbC)
        let coC ← compileOnnx mC { target := tgt }
        let _ ← verifyCompiledOnnx coC

      -- Along W with uniform constant scale & bias
      let gbW := pbGraph "ln_w"
        #[pbVI "x" #[1, 8, 4, 16]] #[pbVI "y" #[1, 8, 4, 16]]
        #[pbF32Tensor "g" #[16] (Array.replicate 16 1.5), pbF32Tensor "b" #[16] (Array.replicate 16 0.25)]
        #[pbNode "LayerNormalization" "ln" #["x", "g", "b"] #["y"] #[pbAttrInt "axis" (-1), pbAttrFloat "epsilon" 1e-5]]
      let mW ← parseOnnx (pbWrapModel 17 gbW)
      let coW ← compileOnnx mW { target := tgt }
      let _ ← verifyCompiledOnnx coW

      -- Along W with runtime scale & bias inputs
      let gbWR := pbGraph "ln_w_rt"
        #[pbVI "x" #[1, 8, 4, 16], pbVI "g" #[1, 1, 1, 16], pbVI "b" #[1, 1, 1, 16]]
        #[pbVI "y" #[1, 8, 4, 16]] #[]
        #[pbNode "LayerNormalization" "ln" #["x", "g", "b"] #["y"] #[pbAttrInt "axis" (-1), pbAttrFloat "epsilon" 1e-5]]
      let mWR ← parseOnnx (pbWrapModel 17 gbWR)
      let coWR ← compileOnnx mWR { target := tgt }
      let _ ← verifyCompiledOnnx coWR
      pure ()
    ))

  -- 12. Gelu(approximate): "", "none" accepted; "tanh" (standalone & fused) and invalid explicitly rejected
  results := results.push (checkOk "m7b: Gelu(approximate) '' and 'none' (exact erf)" (do
    for approx in #["", "none"] do
      let attrs := if approx.isEmpty then #[] else #[pbAttrString "approximate" approx]
      let gb := pbGraph "gelu_exact" #[pbVI "x" #[1, 8, 4, 8]] #[pbVI "y" #[1, 8, 4, 8]] #[]
        #[pbNode "Gelu" "g0" #["x"] #["y"] attrs]
      let m ← parseOnnx (pbWrapModel 20 gb)
      let co ← compileOnnx m { target := .h17 }
      let _ ← verifyCompiledOnnx co
      if !co.lowered.summary.warnings.isEmpty then throw s!"expected 0 warnings for approximate='{approx}'"
    pure ()
  ))

  results := results.push (checkErr "m7b neg: standalone Gelu(approximate='tanh') rejected" (do
    let gb := pbGraph "gelu_tanh" #[pbVI "x" #[1, 8, 4, 8]] #[pbVI "y" #[1, 8, 4, 8]] #[]
      #[pbNode "Gelu" "g0" #["x"] #["y"] #[pbAttrString "approximate" "tanh"]]
    let m ← parseOnnx (pbWrapModel 20 gb)
    lowerOnnx m {}
  ) "approximate='tanh' is not supported")

  results := results.push (checkErr "m7b neg: fused Conv+Gelu(approximate='tanh') rejected" (do
    let gb := pbGraph "conv_gelu_tanh" #[pbVI "x" #[1, 8, 4, 8]] #[pbVI "y" #[1, 8, 4, 8]]
      #[pbF32Tensor "w" #[8, 8, 1, 1] (mkDetF32 64 0.05)]
      #[
        pbNode "Conv" "c0" #["x", "w"] #["c0_out"],
        pbNode "Gelu" "g0" #["c0_out"] #["y"] #[pbAttrString "approximate" "tanh"]
      ]
    let m ← parseOnnx (pbWrapModel 20 gb)
    lowerOnnx m {}
  ) "approximate='tanh' is not supported")

  results := results.push (checkErr "m7b neg: Gelu(approximate='bad') rejected" (do
    let gb := pbGraph "gelu_bad" #[pbVI "x" #[1, 8, 4, 8]] #[pbVI "y" #[1, 8, 4, 8]] #[]
      #[pbNode "Gelu" "g0" #["x"] #["y"] #[pbAttrString "approximate" "bad"]]
    let m ← parseOnnx (pbWrapModel 20 gb)
    lowerOnnx m {}
  ) "unsupported approximate")

  -- 13. Negative MatMul constraint checks (inner K mismatch, M > 2048, transposeB N > 128)
  results := results.push (checkErr "m7b neg matmul: inner K mismatch rejected" (do
    let gb := pbGraph "mm_bad_k" #[pbVI "a" #[1, 2, 16, 32], pbVI "b" #[1, 2, 16, 32]] #[pbVI "y" #[1, 2, 16, 32]] #[]
      #[pbNode "MatMul" "mm" #["a", "b"] #["y"]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "inner dimension mismatch")

  results := results.push (checkErr "m7b neg matmul: M > 2048 rejected" (do
    let gb := pbGraph "mm_big_m" #[pbVI "a" #[1, 1, 2049, 16], pbVI "b" #[1, 1, 16, 16]] #[pbVI "y" #[1, 1, 2049, 16]] #[]
      #[pbNode "MatMul" "mm" #["a", "b"] #["y"]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "M=2049")

  results := results.push (checkErr "m7b neg matmul: transposeB with N > 128 rejected" (do
    let gb := pbGraph "mm_big_trb" #[pbVI "a" #[1, 2, 16, 32], pbVI "b" #[1, 2, 129, 32]] #[pbVI "y" #[1, 2, 16, 129]] #[]
      #[
        pbNode "Transpose" "bt" #["b"] #["b_t"] #[pbAttrInts "perm" #[0, 1, 3, 2]],
        pbNode "MatMul" "mm" #["a", "b_t"] #["y"]
      ]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "transposeB requires N=129 <= 128")

  -- Review regression #1: 4D MatMul must reject folded Transpose with [0,3,2,1] perm
  results := results.push (checkErr "m7b neg matmul: 4D MatMul rejects folded [0,3,2,1] Transpose perm on RHS" (do
    let gb := pbGraph "mm_4d_bad_perm"
      #[pbVI "a" #[1, 4, 16, 4], pbVI "b" #[1, 4, 1, 4]]
      #[pbVI "y" #[1, 4, 16, 4]] #[]
      #[
        pbNode "Transpose" "bt" #["b"] #["b_t"] #[pbAttrInts "perm" #[0, 3, 2, 1]],
        pbNode "MatMul" "mm" #["a", "b_t"] #["y"]
      ]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "4D MatMul RHS folded Transpose perm")

  -- Review regression #2: head-stacked MatMul (trA && trB) forces heads=1 (rejects A.C != B.W)
  results := results.push (checkErr "m7b neg matmul: head-stacked MatMul (trA+trB) rejects multi-head A.C != B.W" (do
    let gb := pbGraph "mm_hs_bad_trab"
      #[pbVI "a" #[1, 16, 1, 4], pbVI "b" #[1, 8, 1, 8]]
      #[pbVI "y" #[1, 4, 1, 8]] #[]
      #[
        pbNode "Transpose" "at" #["a"] #["a_t"] #[pbAttrInts "perm" #[0, 3, 2, 1]],
        pbNode "Transpose" "bt" #["b"] #["b_t"] #[pbAttrInts "perm" #[0, 3, 2, 1]],
        pbNode "MatMul" "mm" #["a_t", "b_t"] #["y"]
      ]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "heads=1")

  -- Review regression #3: static-weight MatMul rejects 3D [H, M, K] input
  results := results.push (checkErr "m7b neg matmul: static-weight MatMul rejects 3D [H,M,K] input" (do
    let w := pbF32Tensor "w" #[8, 4] (mkDetF32 32 0.05)
    let gb := pbGraph "mm_static_3d" #[pbVI "x" #[2, 4, 8]] #[pbVI "y" #[2, 4, 4]] #[w]
      #[pbNode "MatMul" "mm" #["x", "w"] #["y"]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "requires 2D [1,C] or 4D [1,C,H,W]")

  -- Review regression #4: Gelu invalid UTF-8 bytes in approximate attribute rejected
  results := results.push (checkErr "m7b neg gelu: invalid UTF-8 in approximate attribute rejected" (do
    let badAttr := ((ProtoBuilder.empty.writeString 1 "approximate").writeBytes 4 (ByteArray.mk #[0xff, 0xfe])).writeInt64 20 attrString
    let gb := pbGraph "gelu_bad_utf8" #[pbVI "x" #[1, 8, 4, 8]] #[pbVI "y" #[1, 8, 4, 8]] #[]
      #[pbNode "Gelu" "g0" #["x"] #["y"] #[badAttr]]
    let m ← parseOnnx (pbWrapModel 20 gb)
    lowerOnnx m {}
  ) "invalid UTF-8")

  -- 14. External review Blocker 1 regressions: LayerNormalization axis=1 on 4D & Scale/B suffix validation
  results := results.push (checkErr "m7b neg ln: LayerNormalization(axis=1) on 4D [1,16,4,8] rejected" (do
    let gb := pbGraph "ln_c_4d_bad"
      #[pbVI "x" #[1, 16, 4, 8]] #[pbVI "y" #[1, 16, 4, 8]]
      #[pbF32Tensor "g" #[16] (Array.replicate 16 1.0), pbF32Tensor "b" #[16] (Array.replicate 16 0.0)]
      #[pbNode "LayerNormalization" "ln" #["x", "g", "b"] #["y"] #[pbAttrInt "axis" 1, pbAttrFloat "epsilon" 1e-5]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "only supported when H=1 and W=1")

  results := results.push (checkErr "m7b neg ln: LayerNormalization Scale shape [2,8] rejected for C=16" (do
    let gb := pbGraph "ln_bad_scale_sh"
      #[pbVI "x" #[1, 16, 1, 1]] #[pbVI "y" #[1, 16, 1, 1]]
      #[pbF32Tensor "g" #[2, 8] (Array.replicate 16 1.0)]
      #[pbNode "LayerNormalization" "ln" #["x", "g"] #["y"] #[pbAttrInt "axis" 1]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "incompatible with normalized suffix")

  results := results.push (checkErr "m7b neg ln: LayerNormalization runtime Scale [1,16,1,1] rejected for axis=-1" (do
    let gb := pbGraph "ln_bad_rt_scale"
      #[pbVI "x" #[1, 8, 4, 16], pbVI "g" #[1, 16, 1, 1]] #[pbVI "y" #[1, 8, 4, 16]] #[]
      #[pbNode "LayerNormalization" "ln" #["x", "g"] #["y"] #[pbAttrInt "axis" (-1)]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "incompatible with normalized suffix")

  -- 15. External review Blocker 2 regressions: Rank-3 [H,M,K] ops (Concat, Slice, Broadcast, Flatten, Output shape)
  results := results.push (checkErr "m7b neg rank3: Concat rank-3 axis=1 rejected" (do
    let gb := pbGraph "cat3_ax1"
      #[pbVI "a" #[2, 4, 8], pbVI "b" #[2, 4, 8]] #[pbVI "y" #[2, 8, 8]] #[]
      #[pbNode "Concat" "cat" #["a", "b"] #["y"] #[pbAttrInt "axis" 1]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "3D Concat axis 1 not supported")

  results := results.push (checkErr "m7b neg rank3: Concat rank-3 axis=0 rejected" (do
    let gb := pbGraph "cat3_ax0"
      #[pbVI "a" #[2, 4, 8], pbVI "b" #[2, 4, 8]] #[pbVI "y" #[4, 4, 8]] #[]
      #[pbNode "Concat" "cat" #["a", "b"] #["y"] #[pbAttrInt "axis" 0]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "3D Concat axis 0 not supported")

  results := results.push (checkErr "m7b neg rank3: Concat rank-3 mismatched (H,M) [2,6,8]+[3,4,8] rejected" (do
    let gb := pbGraph "cat3_bad_hm"
      #[pbVI "a" #[2, 6, 8], pbVI "b" #[3, 4, 8]] #[pbVI "y" #[2, 6, 16]] #[]
      #[pbNode "Concat" "cat" #["a", "b"] #["y"] #[pbAttrInt "axis" 2]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "mismatched (H,M)")

  results := results.push (checkOk "m7b rank3: Concat axis=2 and Slice axes=[2] on [2,4,64] compile and verify" (do
    let gbCat := pbGraph "cat3_ax2"
      #[pbVI "a" #[2, 4, 32], pbVI "b" #[2, 4, 32]] #[pbVI "y" #[2, 4, 64]] #[]
      #[pbNode "Concat" "cat" #["a", "b"] #["y"] #[pbAttrInt "axis" 2]]
    let mCat ← parseOnnx (pbWrapModel 17 gbCat)
    let coCat ← compileOnnx mCat { target := .h17 }
    let _ ← verifyCompiledOnnx coCat

    let gbSl := pbGraph "slice3_ax2"
      #[pbVI "x" #[2, 4, 64]] #[pbVI "y" #[2, 4, 32]]
      #[pbI64Tensor "st" #[1] #[0], pbI64Tensor "en" #[1] #[32], pbI64Tensor "ax" #[1] #[2]]
      #[pbNode "Slice" "sl" #["x", "st", "en", "ax"] #["y"]]
    let mSl ← parseOnnx (pbWrapModel 17 gbSl)
    let coSl ← compileOnnx mSl { target := .h17 }
    let _ ← verifyCompiledOnnx coSl
    if coSl.lowered.summary.outputShape != { n := 1, c := 8, h := 1, w := 32 } then
      throw s!"unexpected 3D Slice output shape {repr coSl.lowered.summary.outputShape}"
    pure ()
  ))

  results := results.push (checkErr "m7b neg rank3: Slice rank-3 axes=[1] rejected" (do
    let gb := pbGraph "slice3_ax1"
      #[pbVI "x" #[2, 4, 64]] #[pbVI "y" #[2, 2, 64]]
      #[pbI64Tensor "st" #[1] #[0], pbI64Tensor "en" #[1] #[2], pbI64Tensor "ax" #[1] #[1]]
      #[pbNode "Slice" "sl" #["x", "st", "en", "ax"] #["y"]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "3D Slice axis 1 not supported")

  results := results.push (checkErr "m7b neg rank3: Slice rank-3 axes=[-2] rejected" (do
    let gb := pbGraph "slice3_axneg2"
      #[pbVI "x" #[2, 4, 64]] #[pbVI "y" #[2, 2, 64]]
      #[pbI64Tensor "st" #[1] #[0], pbI64Tensor "en" #[1] #[2], pbI64Tensor "ax" #[1] #[-2]]
      #[pbNode "Slice" "sl" #["x", "st", "en", "ax"] #["y"]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "3D Slice axis -2 not supported")

  results := results.push (checkErr "m7b neg rank3: Slice rank-3 axes=[3] out of range rejected" (do
    let gb := pbGraph "slice3_ax3"
      #[pbVI "x" #[2, 4, 64]] #[pbVI "y" #[2, 4, 32]]
      #[pbI64Tensor "st" #[1] #[0], pbI64Tensor "en" #[1] #[32], pbI64Tensor "ax" #[1] #[3]]
      #[pbNode "Slice" "sl" #["x", "st", "en", "ax"] #["y"]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "out of range [0,2]")

  results := results.push (checkErr "m7b neg rank3: elementwise broadcast [3,1,4]+[1,3,4] rejected" (do
    let gb := pbGraph "bcast3_bad"
      #[pbVI "a" #[3, 1, 4], pbVI "b" #[1, 3, 4]] #[pbVI "y" #[3, 3, 4]] #[]
      #[pbNode "Add" "add0" #["a", "b"] #["y"]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "incompatible with ANE physical layout")

  results := results.push (checkErr "m7b neg rank3: elementwise [2,3,4]+[3,2,4] rejected" (do
    let gb := pbGraph "bcast3_bad2"
      #[pbVI "a" #[2, 3, 4], pbVI "b" #[3, 2, 4]] #[pbVI "y" #[2, 3, 4]] #[]
      #[pbNode "Add" "add0" #["a", "b"] #["y"]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "incompatible ONNX broadcast shapes")

  results := results.push (checkOk "m7b rank3: elementwise [2,3,4]+[1,1,4] compiles and verifies" (do
    let gb := pbGraph "bcast3_ok"
      #[pbVI "a" #[2, 3, 4], pbVI "b" #[1, 1, 4]] #[pbVI "y" #[2, 3, 4]] #[]
      #[pbNode "Add" "add0" #["a", "b"] #["y"]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    let co ← compileOnnx m { target := .h17 }
    let _ ← verifyCompiledOnnx co
    pure ()
  ))

  results := results.push (checkErr "m7b neg rank3: Flatten rank-3 [2,4,8] axis=1 rejected" (do
    let gb := pbGraph "flat3_bad"
      #[pbVI "x" #[2, 4, 8]] #[pbVI "y" #[2, 32]] #[]
      #[pbNode "Flatten" "fl" #["x"] #["y"] #[pbAttrInt "axis" 1]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "batch dimension 2 != 1")

  results := results.push (checkOk "m7b rank3: Flatten [1,8,1] axis=1 -> [1,8] compiles and verifies" (do
    let gb := pbGraph "flat3_ok"
      #[pbVI "x" #[1, 8, 1]] #[pbVI "y" #[1, 8]] #[]
      #[pbNode "Flatten" "fl" #["x"] #["y"] #[pbAttrInt "axis" 1]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    let co ← compileOnnx m { target := .h17 }
    let _ ← verifyCompiledOnnx co
    pure ()
  ))

  results := results.push (checkErr "m7b neg output shape: declared output dim mismatch rejected" (do
    let gb := pbGraph "out_sh_bad"
      #[pbVI "x" #[1, 8, 4, 4]] #[pbVI "y" #[1, 4, 4, 4]] #[]
      #[pbNode "Relu" "r0" #["x"] #["y"]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "declared output dim[1]=4")

  results := results.push (checkErr "m7b neg output shape: declared output rank mismatch rejected" (do
    let gb := pbGraph "out_rk_bad"
      #[pbVI "x" #[1, 8, 1, 1]] #[pbVI "y" #[1, 8]] #[]
      #[pbNode "Relu" "r0" #["x"] #["y"]]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "declared output rank 2")

  -- 16. Ported parser & lowering regression guards
  results := results.push (checkErr "m7b neg onnx proto3: varint overflow (10th byte = 0x02) rejected" (
    readVarint { buf := ByteArray.mk #[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02], pos := 0 }
  ) "varint overflow")

  results := results.push (checkErr "m7b neg onnx proto3: truncated length-delimited field rejected" (
    -- Field 2 (producer_name, wire type 2 = bytes), length 10, only 2 bytes present
    parseOnnx (ByteArray.mk #[0x12, 0x0a, 0x01, 0x02])
  ) "exceeds remaining")

  results := results.push (checkErr "m7b neg safetensors: header_len = 2^64 - 1 rejected" (
    parseSafeTensors ((pushU64LE ByteArray.empty 0xffffffffffffffff) ++ zeroBytes 8)
  ) "exceeds limit")

  results := results.push (checkErr "m7b neg safetensors: header_len past EOF rejected" (
    parseSafeTensors ((pushU64LE ByteArray.empty 64) ++ zeroBytes 8)
  ) "exceeds remaining file size")

  results := results.push (checkErr "m7b neg safetensors: data offset past EOF rejected" (do
    let hdr := "{\"w\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,8]}}"
    let hBytes := hdr.toUTF8
    parseSafeTensors ((pushU64LE ByteArray.empty hBytes.size.toUInt64) ++ hBytes ++ zeroBytes 4)
  ) "exceeds data section size")

  results := results.push (checkErr "m7b neg safetensors: begin > end rejected" (do
    let hdr := "{\"w\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[4,2]}}"
    let hBytes := hdr.toUTF8
    parseSafeTensors ((pushU64LE ByteArray.empty hBytes.size.toUInt64) ++ hBytes ++ zeroBytes 8)
  ) "begin 4 > end 2")

  results := results.push (checkErr "m7b neg safetensors: shape-product overflow rejected" (do
    let hdr := "{\"w\":{\"dtype\":\"F32\",\"shape\":[3037000500,3037000500],\"data_offsets\":[0,0]}}"
    let hBytes := hdr.toUTF8
    parseSafeTensors ((pushU64LE ByteArray.empty hBytes.size.toUInt64) ++ hBytes)
  ) "overflow")

  results := results.push (checkErr "m7b neg safetensors: trailing non-JSON garbage rejected" (do
    let hdr := "{\"w\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}} GARBAGE"
    let hBytes := hdr.toUTF8
    parseSafeTensors ((pushU64LE ByteArray.empty hBytes.size.toUInt64) ++ hBytes ++ zeroBytes 2)
  ) "invalid JSON")

  results := results.push (checkErr "m7b neg safetensors: trailing JSON object rejected" (do
    let hdr := "{\"w\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}{\"extra\":1}"
    let hBytes := hdr.toUTF8
    parseSafeTensors ((pushU64LE ByteArray.empty hBytes.size.toUInt64) ++ hBytes ++ zeroBytes 2)
  ) "invalid JSON")

  results := results.push (checkOk "m7b fusion guard: multi-consumer Conv+Relu, Conv+BN, and graph-output Conv not fused" (do
    let w := pbF32Tensor "w" #[2, 2, 1, 1] #[1.0, 0.0, 0.0, 1.0]
    -- Multi-consumer Conv -> Relu + Add(r_out, c_out)
    let gbMcRelu := pbGraph "mc_relu"
      #[pbVI "x" #[1, 2, 2, 2]] #[pbVI "y" #[1, 2, 2, 2]] #[w]
      #[
        pbNode "Conv" "c1" #["x", "w"] #["c_out"],
        pbNode "Relu" "r1" #["c_out"] #["r_out"],
        pbNode "Add" "a1" #["r_out", "c_out"] #["y"]
      ]
    let m1 ← parseOnnx (pbWrapModel 17 gbMcRelu)
    let r1 ← lowerOnnx m1 {}
    if r1.summary.fusedActCount != 0 || r1.spec.nodes.size != 3 then
      throw s!"expected multi-consumer Conv+Relu not fused, got fusedAct={r1.summary.fusedActCount} nodes={r1.spec.nodes.size}"

    -- Multi-consumer Conv -> BN + Add(bn_out, c_out)
    let gbMcBn := pbGraph "mc_bn"
      #[pbVI "x" #[1, 2, 2, 2]] #[pbVI "y" #[1, 2, 2, 2]]
      #[w, pbF32Tensor "s" #[2] #[1.0, 1.0], pbF32Tensor "b" #[2] #[0.0, 0.0],
        pbF32Tensor "m" #[2] #[0.0, 0.0], pbF32Tensor "v" #[2] #[1.0, 1.0]]
      #[
        pbNode "Conv" "c1" #["x", "w"] #["c_out"],
        pbNode "BatchNormalization" "bn1" #["c_out", "s", "b", "m", "v"] #["bn_out"],
        pbNode "Add" "a1" #["bn_out", "c_out"] #["y"]
      ]
    let m2 ← parseOnnx (pbWrapModel 17 gbMcBn)
    let r2 ← lowerOnnx m2 {}
    if r2.summary.fusedBnCount != 0 || r2.spec.nodes.size != 3 then
      throw s!"expected multi-consumer Conv+BN not folded, got fusedBn={r2.summary.fusedBnCount} nodes={r2.spec.nodes.size}"

    -- Graph-output Conv with dead Relu consumer not fused
    let gbOut := pbGraph "conv_out"
      #[pbVI "x" #[1, 2, 2, 2]] #[pbVI "c_out" #[1, 2, 2, 2]] #[w]
      #[
        pbNode "Conv" "c1" #["x", "w"] #["c_out"],
        pbNode "Relu" "dead_r" #["c_out"] #["dead_y"]
      ]
    let m3 ← parseOnnx (pbWrapModel 17 gbOut)
    let r3 ← lowerOnnx m3 {}
    if r3.summary.fusedActCount != 0 || r3.spec.nodes.size != 1 then
      throw s!"expected graph-output Conv not fused, got fusedAct={r3.summary.fusedActCount} nodes={r3.spec.nodes.size}"
    pure ()
  ))

  results := results.push (checkErr "m7b neg fusion guard: MatMul+Add spatial [1,1,2,2] constant not misfolded as channel bias" (do
    let gb := pbGraph "mm_spatial_add"
      #[pbVI "x" #[1, 2, 2, 2]] #[pbVI "y" #[1, 4, 2, 2]]
      #[
        pbF32Tensor "w" #[2, 4] (Array.replicate 8 0.0),
        pbF32Tensor "spatial_c" #[1, 1, 2, 2] #[1.0, 2.0, 3.0, 4.0]
      ]
      #[
        pbNode "MatMul" "mm" #["x", "w"] #["h"],
        pbNode "Add" "add_sp" #["h", "spatial_c"] #["y"]
      ]
    let m ← parseOnnx (pbWrapModel 17 gb)
    lowerOnnx m {}
  ) "non-scalar constant")

  pure results

-- ===========================================================================
-- Milestone 8A Tests: Native NE Weight Quantization (int8, uint8, int4, uint4, pal*)
-- ===========================================================================

private def mkQuantCodes (n : Nat) (modVal : Nat) (seed : Nat := 1) : ByteArray :=
  ByteArray.mk (Array.ofFn (n := n) fun i => ((i.val * 7 + seed) % modVal).toUInt8)

private def mkQuantScales (n : Nat) (base : Float := 0.0625) : Array Float :=
  Array.ofFn (n := n) fun i => base * Float.ofNat (1 + i.val % 4)

private def mkQuantZps (n : Nat) (modVal : Nat) : Array UInt8 :=
  Array.ofFn (n := n) fun i => ((i.val * 3 + 1) % modVal).toUInt8

private def mkQuantPalette (bits : Nat) : Array UInt16 :=
  let entries := (1 : Nat) <<< bits
  Array.ofFn (n := entries) fun i =>
    let centered : Float := (Float.ofNat i.val / Float.ofNat entries - 0.5) * 2.0
    floatToF16CanonicalZero centered

private def mkConstBiasBytes (cout : Nat) (val : Float) : ByteArray := Id.run do
  let h := floatToF16CanonicalZero val
  let mut bs := ByteArray.emptyWithCapacity (2 * cout)
  for _ in [:cout] do
    bs := pushU16LE bs h
  return bs

private def runM8aTests : IO (Array TestResult) := do
  let mut results : Array TestResult := #[]

  -- 1. Unit tests: packBitsLSB / unpackBitsLSB round-trip across widths {1, 2, 4, 6, 8}
  results := results.push (checkOk "m8a: packBitsLSB / unpackBitsLSB lossless round-trip for widths {1, 2, 4, 6, 8}" (do
    for w in #[1, 2, 4, 6, 8] do
      let mask : UInt8 := (((1 : Nat) <<< w) - 1).toUInt8
      for cnt in #[0, 1, 2, 3, 5, 7, 8, 9, 13, 16, 33] do
        let codes := Array.ofFn (n := cnt) fun i => ((i.val * 19 + 5) % 256).toUInt8
        let packed := Vane.Quant.packBitsLSB w codes
        if packed.size != packedByteCount w cnt then
          throw s!"w={w} cnt={cnt}: packed size {packed.size} != {packedByteCount w cnt}"
        let unpacked := Vane.Quant.unpackBitsLSB w packed cnt
        let expected := codes.map (fun c => c &&& mask)
        if unpacked != expected then
          throw s!"w={w} cnt={cnt}: unpacked mismatch"
    -- Concrete LSB-first 4-bit and 6-bit byte checks
    let p4 := Vane.Quant.packBitsLSB 4 #[0x03, 0x0a, 0x05]
    if p4.data != #[0xa3, 0x05] then
      throw s!"4-bit LSB packing mismatch: got {p4.data}"
    let p6 := Vane.Quant.packBitsLSB 6 #[0x3f, 0x01, 0x2a, 0x15]
    -- c0=0b111111, c1=0b000001 -> b0 = 0x3f | (1<<6) = 0x7f, b1 = 0 | (0x0a<<4) = 0xa0, b2 = 0x02 | (0x15<<2) = 0x56
    if p6.data != #[0x7f, 0xa0, 0x56] then
      throw s!"6-bit LSB packing mismatch: got {p6.data}"
    pure ()
  ))

  -- 2. Unit tests: computePexp & computeBiasShift exact thresholds
  results := results.push (checkOk "m8a: computePexp and computeBiasShift exact thresholds" (do
    if Vane.Quant.computePexp #[0.0625] != 0 then
      throw "computePexp(0.0625) != 0"
    if Vane.Quant.computePexp #[ldexp 1.0 (-16)] != 0 then
      throw "computePexp(2^-16) != 0"
    if Vane.Quant.computePexp #[ldexp 1.0 (-18)] != 2 then
      throw "computePexp(2^-18) != 2"
    if Vane.Quant.computePexp #[0.5, ldexp 1.0 (-20)] != 4 then
      throw "computePexp([0.5, 2^-20]) != 4"
    let s15 := ldexp 1.0 (-15)
    if Vane.Quant.computeBiasShift #[s15] #[1.5] != 1 then
      throw "computeBiasShift(2^-15, 1.5) != 1"
    if Vane.Quant.computeBiasShift #[0.125] #[1.5] != 0 then
      throw "computeBiasShift(0.125, 1.5) != 0"
    pure ()
  ))

  -- 3. Positive H16 & H17: int8 (per-tensor, per-channel, per-channel+ZP, tiny scale pexp>0, biasShift>0)
  for tgt in Target.all do
    results := results.push (checkOk s!"m8a ({tgt.name}): int8 per-tensor, per-channel, per-channel+ZP, pexp>0, biasShift>0 compile & verify" (do
      let sh : Conv1x1Shape := { h := 8, w := 16, cin := 32, cout := 32 }
      let bias32 := mkM5aBias 32
      -- per-tensor
      let qPt : Vane.Quant.QuantWeightSpec := {
        format := .int8, codes := mkQuantCodes (32 * 32) 256 1, scales := #[0.0625]
      }
      let cPt ← compileConv1x1Quant tgt sh qPt (some bias32) true
      matrixVerify tgt cPt.hwx cPt.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }

      -- per-channel: verify 0x003c bit 27 == 0 and explicit 0x490c == 0 on ALL 1x1 quantized convs (both H16 & H17!)
      let qPc : Vane.Quant.QuantWeightSpec := {
        format := .int8, codes := mkQuantCodes (32 * 32) 256 2, scales := mkQuantScales 32 0.03125
      }
      let cPc ← compileConv1x1Quant tgt sh qPc (some bias32) false
      matrixVerify tgt cPc.hwx cPc.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }
      let fComPc ← findM5aTaskRegOffset cPc.hwx cPc.info 0 regNeCommon
      let wComPc ← readU32LE cPc.hwx fComPc
      if (wComPc &&& ((1 : UInt32) <<< 27)) != 0 then
        throw s!"{tgt.name} 1x1 int8: 0x003c bit 27 (wide3x3) must be 0 on 1x1 conv"
      let fSbPc ← findM5aTaskRegOffset cPc.hwx cPc.info 0 regScalarBias
      let wSbPc ← readU32LE cPc.hwx fSbPc
      if wSbPc != 0 then
        throw s!"{tgt.name} 1x1 int8: expected explicit 0x490c == 0 when biasShift=0, got {toHex wSbPc.toNat}"

      -- per-channel + ZP: verify H17 0x4900 bit 28 == 0 and 0x003c bit 27 == 0
      let qZp : Vane.Quant.QuantWeightSpec := {
        format := .int8, codes := mkQuantCodes (32 * 32) 256 3,
        scales := mkQuantScales 32 0.03125, zeroPoints := some (mkQuantZps 32 120)
      }
      let cZp ← compileConv1x1Quant tgt sh qZp (some bias32) false
      matrixVerify tgt cZp.hwx cZp.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }
      let fCfg ← findM5aTaskRegOffset cZp.hwx cZp.info 0 regNeCfg
      let wCfg ← readU32LE cZp.hwx fCfg
      if (wCfg &&& ((1 : UInt32) <<< 28)) != 0 then
        throw s!"{tgt.name} int8+ZP: 0x4900 bit 28 must be 0 when hasZP=1"
      let fCom ← findM5aTaskRegOffset cZp.hwx cZp.info 0 regNeCommon
      let wCom ← readU32LE cZp.hwx fCom
      if (wCom &&& ((1 : UInt32) <<< 27)) != 0 then
        throw s!"{tgt.name} int8+ZP: 0x003c bit 27 must be 0 when hasZP=1"

      -- tiny scale s = 2^-18 -> pexp = 2
      let qTiny : Vane.Quant.QuantWeightSpec := {
        format := .int8, codes := mkQuantCodes (32 * 32) 256 4, scales := #[ldexp 1.0 (-18)]
      }
      let cTiny ← compileConv1x1Quant tgt sh qTiny none false
      matrixVerify tgt cTiny.hwx cTiny.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }
      let fSc ← findM5aTaskRegOffset cTiny.hwx cTiny.info 0 regNeScale
      let wSc ← readU32LE cTiny.hwx fSc
      if (wSc >>> 16) != 2 then
        throw s!"{tgt.name} int8 tiny scale: expected pexp=2 in 0x4910[31:16], got {wSc >>> 16}"

      -- large bias/scale -> biasShift = 1 on Cout=1 and Cout=16
      let sh1 : Conv1x1Shape := { h := 8, w := 16, cin := 32, cout := 1 }
      let qBs1 : Vane.Quant.QuantWeightSpec := {
        format := .int8, codes := mkQuantCodes 32 256 5, scales := #[ldexp 1.0 (-15)]
      }
      let cBs1 ← compileConv1x1Quant tgt sh1 qBs1 (some (mkConstBiasBytes 1 1.5)) false
      matrixVerify tgt cBs1.hwx cBs1.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }
      let fSb ← findM5aTaskRegOffset cBs1.hwx cBs1.info 0 regScalarBias
      let wSb ← readU32LE cBs1.hwx fSb
      if wSb != 0x00010000 then
        throw s!"{tgt.name} int8 biasShift: expected 0x490c == 0x00010000, got {toHex wSb.toNat}"
      pure ()
    ))

    -- 4. Positive H16 & H17: uint8+ZP, int4 (bias, nobias, 3x3 same & valid), uint4+ZP, pal1/2/4/6/8
    results := results.push (checkOk s!"m8a ({tgt.name}): uint8+ZP, int4 (1x1/3x3), uint4+ZP, and pal1/pal2/pal4/pal6/pal8 compile & verify" (do
      let sh : Conv1x1Shape := { h := 8, w := 16, cin := 32, cout := 32 }
      let bias32 := mkM5aBias 32

      -- uint8 per-channel + ZP
      let qU8 : Vane.Quant.QuantWeightSpec := {
        format := .uint8, codes := mkQuantCodes (32 * 32) 256 6,
        scales := mkQuantScales 32 0.03125, zeroPoints := some (mkQuantZps 32 200)
      }
      let cU8 ← compileConv1x1Quant tgt sh qU8 (some bias32) true
      matrixVerify tgt cU8.hwx cU8.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }

      -- int4 per-channel (with bias & bias-free)
      let qI4 : Vane.Quant.QuantWeightSpec := {
        format := .int4, codes := mkQuantCodes (32 * 32) 16 7, scales := mkQuantScales 32 0.125
      }
      let cI4 ← compileConv1x1Quant tgt sh qI4 (some bias32) false
      matrixVerify tgt cI4.hwx cI4.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }
      let cI4Nb ← compileConv1x1Quant tgt sh qI4 none true
      matrixVerify tgt cI4Nb.hwx cI4Nb.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }

      -- int4 dense 3x3 (same & valid): check wide3x3 bit 27 (1 on H17 without ZP, 0 on H16)
      let sh3Same : Conv3x3Shape := { mode := .dense, h := 8, w := 16, cin := 16, cout := 16, stride := 1, pad := .same }
      let q3 : Vane.Quant.QuantWeightSpec := {
        format := .int4, codes := mkQuantCodes (16 * 16 * 9) 16 8, scales := mkQuantScales 16 0.125
      }
      let c3Same ← compileConv3x3DenseQuant tgt sh3Same q3 (some (mkM5aBias 16)) true
      matrixVerify tgt c3Same.hwx c3Same.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }
      let expW3 := tgt == .h17
      if c3Same.wide3x3 != expW3 then
        throw s!"{tgt.name} 3x3 int4: wide3x3={c3Same.wide3x3} != expected {expW3}"
      let sh3Valid : Conv3x3Shape := { sh3Same with pad := .valid }
      let c3Valid ← compileConv3x3DenseQuant tgt sh3Valid q3 none false
      matrixVerify tgt c3Valid.hwx c3Valid.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }

      -- uint4 per-channel + ZP (and uint4 without ZP)
      let qU4Zp : Vane.Quant.QuantWeightSpec := {
        format := .uint4, codes := mkQuantCodes (32 * 32) 16 9,
        scales := mkQuantScales 32 0.125, zeroPoints := some (mkQuantZps 32 16)
      }
      let cU4Zp ← compileConv1x1Quant tgt sh qU4Zp (some bias32) false
      matrixVerify tgt cU4Zp.hwx cU4Zp.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }
      let qU4NoZp : Vane.Quant.QuantWeightSpec := { qU4Zp with zeroPoints := none }
      let cU4NoZp ← compileConv1x1Quant tgt sh qU4NoZp none false
      matrixVerify tgt cU4NoZp.hwx cU4NoZp.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }

      -- pal1, pal2, pal4, pal6, pal8
      for b in #[1, 2, 4, 6, 8] do
        let qPal : Vane.Quant.QuantWeightSpec := {
          format := .pal b,
          codes := mkQuantCodes (32 * 32) ((1 : Nat) <<< b) (10 + b),
          palette := mkQuantPalette b
        }
        let cPal ← compileConv1x1Quant tgt sh qPal (some bias32) false
        matrixVerify tgt cPal.hwx cPal.sections { total := 1, neConv := 1, neAct := 0, peOp := 0 }
      pure ()
    ))

    -- 5. Positive H16 & H17: wide-Cout auto-output-split quantized linear & weight compression ratio
    results := results.push (checkOk s!"m8a ({tgt.name}): wide-Cout auto-split linear (256->640 int4) & compression ratio" (do
      let qWide : Vane.Quant.QuantWeightSpec := {
        format := .int4, codes := mkQuantCodes (640 * 256) 16 12, scales := mkQuantScales 640 0.0625
      }
      let cWide ← compileLinearQuant tgt 16 256 640 qWide (some (mkM5aBias 640)) 1 true
      if cWide.slices.size < 2 then
        throw s!"{tgt.name}: expected wide 256->640 to split into >= 2 slices, got {cWide.slices}"
      if cWide.slices.foldl (· + ·) 0 != 640 then
        throw s!"{tgt.name}: split slice sum != 640"
      matrixVerify tgt cWide.hwx cWide.sections { total := cWide.slices.size, neConv := cWide.slices.size, neAct := 0, peOp := 0 }
      -- Verify mixed-biasShift split linear: slice 0 has biasShift=1, later slices explicitly reset 0x490c=0
      let mixedScales : Array Float := Array.ofFn (n := 640) fun i =>
        if i.val < cWide.slices[0]! then ldexp 1.0 (-15) else 0.0625
      let cWideMix ← compileLinearQuant tgt 16 256 640 { qWide with scales := mixedScales }
        (some (mkConstBiasBytes 640 1.5)) 1 false
      matrixVerify tgt cWideMix.hwx cWideMix.sections { total := cWideMix.slices.size, neConv := cWideMix.slices.size, neAct := 0, peOp := 0 }
      let fSb0 ← findM5aTaskRegOffset cWideMix.hwx cWideMix.info 0 regScalarBias
      let wSb0 ← readU32LE cWideMix.hwx fSb0
      if wSb0 != 0x00010000 then
        throw s!"{tgt.name}: mixed-biasShift slice 0 expected 0x490c=0x00010000, got {toHex wSb0.toNat}"
      for si in [1 : cWideMix.slices.size] do
        let fSbi ← findM5aTaskRegOffset cWideMix.hwx cWideMix.info si regScalarBias
        let wSbi ← readU32LE cWideMix.hwx fSbi
        if wSbi != 0 then
          throw s!"{tgt.name}: mixed-biasShift slice {si} expected explicit 0x490c=0, got {toHex wSbi.toNat}"

      -- Check compression ratios on 256->256 linear
      let cFp16 ← compileLinear tgt 16 256 256 (mkM5aW1x1 256 256 1) none 1 false
      let cI8 ← compileLinearQuant tgt 16 256 256 {
        format := .int8, codes := mkQuantCodes (256 * 256) 256 1, scales := mkQuantScales 256 0.0625
      } none 1 false
      let cI4 ← compileLinearQuant tgt 16 256 256 {
        format := .int4, codes := mkQuantCodes (256 * 256) 16 1, scales := mkQuantScales 256 0.0625
      } none 1 false
      let cP2 ← compileLinearQuant tgt 16 256 256 {
        format := .pal 2, codes := mkQuantCodes (256 * 256) 4 1, palette := mkQuantPalette 2
      } none 1 false
      if !(cP2.kernelBytes < cI4.kernelBytes && cI4.kernelBytes < cI8.kernelBytes && cI8.kernelBytes < cFp16.kernelBytes) then
        throw s!"{tgt.name}: expected pal2({cP2.kernelBytes}) < int4({cI4.kernelBytes}) < int8({cI8.kernelBytes}) < fp16({cFp16.kernelBytes})"
      pure ()
    ))

    -- 6. Positive H16 & H17: multi-layer mixed-precision quantized graph + residual
    results := results.push (checkOk s!"m8a ({tgt.name}): multi-layer quantized graph (int8 conv3x3 -> ReLU -> int4 conv1x1 -> pal4 linear + fp16 residual)" (do
      let qC1 : Vane.Quant.QuantWeightSpec := {
        format := .int8, codes := mkQuantCodes (16 * 16 * 9) 256 1,
        scales := mkQuantScales 16 0.0625, zeroPoints := some (mkQuantZps 16 64)
      }
      let qC2 : Vane.Quant.QuantWeightSpec := {
        format := .int4, codes := mkQuantCodes (16 * 16) 16 2, scales := mkQuantScales 16 0.125
      }
      let qC3 : Vane.Quant.QuantWeightSpec := {
        format := .pal 4, codes := mkQuantCodes (16 * 16) 16 3, palette := mkQuantPalette 4
      }
      let g ← Vane.Graph.compileGraph {
        target := tgt
        defaultPlacement := .auto
        inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
        nodes := #[
          { id := "c1", inputs := #["x"], op := .conv3x3 .dense 16 16 1 .same ByteArray.empty (some (mkM5aBias 16)) .relu (some qC1) },
          { id := "c2", inputs := #["c1"], op := .conv1x1 16 16 ByteArray.empty (some (mkM5aBias 16)) .none (some qC2) },
          { id := "c3", inputs := #["c2"], op := .linear 16 16 ByteArray.empty (some (mkM5aBias 16)) .none (some qC3) },
          { id := "out", inputs := #["c3", "x"], op := .pe2 .add false }
        ]
      }
      if g.taskCount != 4 then
        throw s!"expected 4 tasks, got {g.taskCount}"
      matrixVerify tgt g.hwx g.sections { total := 4, neConv := 3, neAct := 0, peOp := 1 }
    ))

  -- 7. Dedicated Negative Verifier Tamper Tests (every new M8A verifier rule)
  let shBase : Conv1x1Shape := { h := 8, w := 16, cin := 32, cout := 32 }
  let baseI8H17 ← match compileConv1x1Quant .h17 shBase {
      format := .int8, codes := mkQuantCodes (32 * 32) 256 1, scales := mkQuantScales 32 0.0625
    } (some (mkM5aBias 32)) false with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseI8H17 setup failed: {e}")

  let baseI8PtH17 ← match compileConv1x1Quant .h17 shBase {
      format := .int8, codes := mkQuantCodes (32 * 32) 256 2, scales := #[0.0625]
    } (some (mkM5aBias 32)) false with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseI8PtH17 setup failed: {e}")

  let baseI8ZpH17 ← match compileConv1x1Quant .h17 shBase {
      format := .int8, codes := mkQuantCodes (32 * 32) 256 3,
      scales := mkQuantScales 32 0.0625, zeroPoints := some (mkQuantZps 32 64)
    } (some (mkM5aBias 32)) false with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseI8ZpH17 setup failed: {e}")

  let baseI8H16 ← match compileConv1x1Quant .h16 shBase {
      format := .int8, codes := mkQuantCodes (32 * 32) 256 4, scales := mkQuantScales 32 0.0625
    } (some (mkM5aBias 32)) false with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseI8H16 setup failed: {e}")

  let baseI4H17 ← match compileConv1x1Quant .h17 shBase {
      format := .int4, codes := mkQuantCodes (32 * 32) 16 5, scales := mkQuantScales 32 0.125
    } (some (mkM5aBias 32)) false with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseI4H17 setup failed: {e}")

  let baseU4H17 ← match compileConv1x1Quant .h17 shBase {
      format := .uint4, codes := mkQuantCodes (32 * 32) 16 6, scales := mkQuantScales 32 0.125
    } none false with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseU4H17 setup failed: {e}")

  let basePal4H17 ← match compileConv1x1Quant .h17 shBase {
      format := .pal 4, codes := mkQuantCodes (32 * 32) 16 7, palette := mkQuantPalette 4
    } (some (mkM5aBias 32)) false with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"basePal4H17 setup failed: {e}")

  let baseBsH17 ← match compileConv1x1Quant .h17 { h := 8, w := 16, cin := 32, cout := 1 } {
      format := .int8, codes := mkQuantCodes 32 256 8, scales := #[ldexp 1.0 (-15)]
    } (some (mkConstBiasBytes 1 1.5)) false with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseBsH17 setup failed: {e}")

  -- Helper to verify a quantized tamper test matches a specific non-empty error substring
  let checkQuantTamper (name : String) (act : Except String Unit) (expectedSubstr : String) : TestResult :=
    if expectedSubstr.isEmpty then
      { name := name, passed := false, detail := "internal test bug: expectedSubstr must be non-empty" }
    else
      checkErr name act expectedSubstr

  -- Helper to drop a single plain register write from a single-task HWX and re-emit + verify
  let stripRegAndVerify (hwx : ByteArray) (info : HwxInfo) (cIn cOut h w dropAddr : Nat) : Except String Unit := do
    let textBytes ← readSlice hwx info.textSection.offset.toNat info.textSection.size.toNat
    let decoded ← decodeTaskStream .h17 textBytes
    let td0 ← rebuildTaskWithExtraWrites decoded[0]! 1 #[]
    let td := { td0 with plainWrites := td0.plainWrites.filter (fun wr => wr.byteAddr != dropAddr) }
    let inSurf ← makeFp16Surface .h17 "x" "input_x" .input cIn h w
    let outSurf ← makeFp16Surface .h17 "y" "output_y" .output cOut h w
    let some kSec := info.kernelSection | throw "missing kernel section"
    let kBytes ← readSlice hwx kSec.offset.toNat kSec.size.toNat
    let em ← emitProgram {
      target := .h17, surfaces := #[inSurf, outSurf], tasks := #[td],
      kernel := some kBytes, scratch := none, constants := zeroBytes textConstBytes, ident := some "vane"
    }
    let _ ← verifyHwxTasksSummary em.hwx em.info (requireKnownTask := true)
    pure ()

  -- Tamper 1a: Illegal (wKind, wWidth) = 0x31
  results := results.push (checkQuantTamper "m8a neg tamper: illegal (wKind=1, wWidth=3) 0x31 in 0x4900 rejected" (do
    tamperM5aTaskReg baseI8H17.hwx baseI8H17.info 0 regNeCfg 0x10000031
  ) "illegal weight format in neCfg (wKind=1, wWidth=3)")

  -- Tamper 1b: Illegal wKind = 7
  results := results.push (checkQuantTamper "m8a neg tamper: illegal wKind=7 in 0x4900 rejected" (do
    tamperM5aTaskReg baseI8H17.hwx baseI8H17.info 0 regNeCfg 0x10000087
  ) "illegal weight format in neCfg (wKind=7, wWidth=8)")

  -- Tamper 1c: sparse = 1 on quantized int8 rejected
  results := results.push (checkQuantTamper "m8a neg tamper: sparse=1 on quantized int8 rejected" (do
    let fCfg ← findM5aTaskRegOffset baseI8H17.hwx baseI8H17.info 0 regNeCfg
    let orig ← readU32LE baseI8H17.hwx fCfg
    tamperM5aTaskReg baseI8H17.hwx baseI8H17.info 0 regNeCfg (orig ||| ((1 : UInt32) <<< 8))
  ) "neCfg sparse=1 is forbidden for quantized weights")

  -- Tamper 2: H17 0x4900 bit 28 left set when hasZP = 1
  results := results.push (checkQuantTamper "m8a neg tamper: H17 0x4900 bit 28 set with hasZP=1 rejected" (do
    let fCfg ← findM5aTaskRegOffset baseI8ZpH17.hwx baseI8ZpH17.info 0 regNeCfg
    let orig ← readU32LE baseI8ZpH17.hwx fCfg
    tamperM5aTaskReg baseI8ZpH17.hwx baseI8ZpH17.info 0 regNeCfg (orig ||| ((1 : UInt32) <<< 28))
  ) "H17 quantized neCfg bit 28 (h17Flag=1) != expected 0")

  -- Tamper 3: H16 0x4900 bit 28 erroneously set
  results := results.push (checkQuantTamper "m8a neg tamper: H16 0x4900 bit 28 set rejected" (do
    let fCfg ← findM5aTaskRegOffset baseI8H16.hwx baseI8H16.info 0 regNeCfg
    let orig ← readU32LE baseI8H16.hwx fCfg
    tamperM5aTaskReg baseI8H16.hwx baseI8H16.info 0 regNeCfg (orig ||| ((1 : UInt32) <<< 28))
  ) "H16 neCfg bit 28 (h17Flag) must be 0")

  -- Tamper 4a: hasZP = 1 on pal4 rejected
  results := results.push (checkQuantTamper "m8a neg tamper: hasZP=1 on pal4 rejected" (do
    let fCfg ← findM5aTaskRegOffset basePal4H17.hwx basePal4H17.info 0 regNeCfg
    let orig ← readU32LE basePal4H17.hwx fCfg
    tamperM5aTaskReg basePal4H17.hwx basePal4H17.info 0 regNeCfg ((orig &&& (~~~((1 : UInt32) <<< 28))) ||| ((1 : UInt32) <<< 24))
  ) "neCfg hasZeroPoint=1 is forbidden for fp16 and palette weights")

  -- Tamper 4b: 0x003c bit 27 set on H17 quantized conv1x1 (without ZP) rejected (live H17 quarantine guard!)
  results := results.push (checkQuantTamper "m8a neg tamper: 0x003c bit 27 (wide3x3=1) on H17 quantized conv1x1 rejected" (do
    let fCom ← findM5aTaskRegOffset baseI8H17.hwx baseI8H17.info 0 regNeCommon
    let orig ← readU32LE baseI8H17.hwx fCom
    tamperM5aTaskReg baseI8H17.hwx baseI8H17.info 0 regNeCommon (orig ||| ((1 : UInt32) <<< 27))
  ) "neCommon wide3x3=1 must be 0 for 1x1 conv")

  -- Tamper 4c: 0x003c bit 27 set on H16 quantized conv3x3 rejected
  let baseI43x3H16 ← match compileConv3x3DenseQuant .h16
      { mode := .dense, h := 8, w := 16, cin := 16, cout := 16, stride := 1, pad := .same }
      { format := .int4, codes := mkQuantCodes (16 * 16 * 9) 16 11, scales := mkQuantScales 16 0.125 }
      (some (mkM5aBias 16)) false with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseI43x3H16 setup failed: {e}")

  results := results.push (checkQuantTamper "m8a neg tamper: 0x003c bit 27 (wide3x3=1) on H16 quantized conv3x3 rejected" (do
    let fCom ← findM5aTaskRegOffset baseI43x3H16.hwx baseI43x3H16.info 0 regNeCommon
    let orig ← readU32LE baseI43x3H16.hwx fCom
    tamperM5aTaskReg baseI43x3H16.hwx baseI43x3H16.info 0 regNeCommon (orig ||| ((1 : UInt32) <<< 27))
  ) "neCommon wide3x3=1 is illegal on h16")

  -- Tamper 4d: 0x003c bit 27 left set on H17 3x3 with hasZP = 1 rejected
  let baseI8Zp3x3H17 ← match compileConv3x3DenseQuant .h17
      { mode := .dense, h := 8, w := 16, cin := 16, cout := 16, stride := 1, pad := .same }
      { format := .int8, codes := mkQuantCodes (16 * 16 * 9) 256 12,
        scales := mkQuantScales 16 0.0625, zeroPoints := some (mkQuantZps 16 64) }
      (some (mkM5aBias 16)) false with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseI8Zp3x3H17 setup failed: {e}")

  results := results.push (checkQuantTamper "m8a neg tamper: 0x003c bit 27 set on H17 3x3 with hasZP=1 rejected" (do
    let fCom ← findM5aTaskRegOffset baseI8Zp3x3H17.hwx baseI8Zp3x3H17.info 0 regNeCommon
    let orig ← readU32LE baseI8Zp3x3H17.hwx fCom
    tamperM5aTaskReg baseI8Zp3x3H17.hwx baseI8Zp3x3H17.info 0 regNeCommon (orig ||| ((1 : UInt32) <<< 27))
  ) "neCommon bit 27 must be 0 when hasZeroPoint=true")

  -- Tamper 4e: 0x003c bit 27 left set on H17 unsigned uint4 3x3 rejected
  let baseU43x3H17 ← match compileConv3x3DenseQuant .h17
      { mode := .dense, h := 8, w := 16, cin := 16, cout := 16, stride := 1, pad := .same }
      { format := .uint4, codes := mkQuantCodes (16 * 16 * 9) 16 13, scales := mkQuantScales 16 0.125 }
      none false with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"baseU43x3H17 setup failed: {e}")

  results := results.push (checkQuantTamper "m8a neg tamper: 0x003c bit 27 set on H17 unsigned uint4 3x3 rejected" (do
    let fCom ← findM5aTaskRegOffset baseU43x3H17.hwx baseU43x3H17.info 0 regNeCommon
    let orig ← readU32LE baseU43x3H17.hwx fCom
    tamperM5aTaskReg baseU43x3H17.hwx baseU43x3H17.info 0 regNeCommon (orig ||| ((1 : UInt32) <<< 27))
  ) "neCommon bit 27 must be 0 when hasZeroPoint=false or unsigned weight kind (wKind=4)")

  -- Tamper 4f: 0x003c bit 27 left set on 1x1 with hasZP = 1 or unsigned uint4 rejected
  results := results.push (checkQuantTamper "m8a neg tamper: 0x003c bit 27 set on 1x1 with hasZP=1 rejected" (do
    let fCom ← findM5aTaskRegOffset baseI8ZpH17.hwx baseI8ZpH17.info 0 regNeCommon
    let orig ← readU32LE baseI8ZpH17.hwx fCom
    tamperM5aTaskReg baseI8ZpH17.hwx baseI8ZpH17.info 0 regNeCommon (orig ||| ((1 : UInt32) <<< 27))
  ) "neCommon wide3x3=1 must be 0 for 1x1 conv")

  results := results.push (checkQuantTamper "m8a neg tamper: 0x003c bit 27 set on 1x1 unsigned uint4 rejected" (do
    let fCom ← findM5aTaskRegOffset baseU4H17.hwx baseU4H17.info 0 regNeCommon
    let orig ← readU32LE baseU4H17.hwx fCom
    tamperM5aTaskReg baseU4H17.hwx baseU4H17.info 0 regNeCommon (orig ||| ((1 : UInt32) <<< 27))
  ) "neCommon wide3x3=1 must be 0 for 1x1 conv")

  -- Tamper 4g: quantized weights combined with depthwise 3x3 conv rejected
  results := results.push (checkQuantTamper "m8a neg tamper: quantized int8 on depthwise 3x3 conv rejected" (do
    let cDw ← compileConv3x3 .h17 { mode := .depthwise, h := 8, w := 16, cin := 16, cout := 16, stride := 1, pad := .same }
      (mkTaggedWeights3x3 16 1) none false
    tamperM5aTaskReg cDw.hwx cDw.info 0 regNeCfg 0x10000081
  ) "quantized weights (wKind=1) are only supported for dense convolutions")

  -- Tamper 4h: quantized weights combined with stride-2 sparse 3x3 conv rejected
  results := results.push (checkQuantTamper "m8a neg tamper: quantized int8 on stride-2 sparse 3x3 conv rejected" (do
    let cS2 ← compileConv3x3 .h17 { mode := .dense, h := 12, w := 14, cin := 8, cout := 16, stride := 2, pad := .same }
      (mkTaggedWeights3x3 16 8) none false
    tamperM5aTaskReg cS2.hwx cS2.info 0 regNeCfg 0x10000081
  ) "quantized weights (wKind=1) are not supported for stride-2 sparse convolutions")

  -- Tamper 5a0: neOp base20 (bit 20) cleared rejected
  results := results.push (checkQuantTamper "m8a neg tamper: 0x4904 base20 (bit 20) cleared on quantized int8 rejected" (do
    let fOp ← findM5aTaskRegOffset baseI8H17.hwx baseI8H17.info 0 regNeOp
    let orig ← readU32LE baseI8H17.hwx fOp
    tamperM5aTaskReg baseI8H17.hwx baseI8H17.info 0 regNeOp (orig &&& (~~~((1 : UInt32) <<< 20)))
  ) "neOp base20 (bit 20) must be 1")

  -- Tamper 5a1: fused LUT (0x4904 bit 17) on quantized weights rejected
  results := results.push (checkQuantTamper "m8a neg tamper: 0x4904 fused LUT bit 17 set on quantized int8 rejected" (do
    let fOp ← findM5aTaskRegOffset baseI8H17.hwx baseI8H17.info 0 regNeOp
    let orig ← readU32LE baseI8H17.hwx fOp
    tamperM5aTaskReg baseI8H17.hwx baseI8H17.info 0 regNeOp (orig ||| ((1 : UInt32) <<< 17))
  ) "fused LUT activation (0x4904 bit 17) is not supported for quantized weights")

  -- Tamper 5a: 0x4904 0x3c00 (unscaled) left set on scaled int8
  results := results.push (checkQuantTamper "m8a neg tamper: 0x4904 0x3c00 bits set on scaled int8 rejected" (do
    let fOp ← findM5aTaskRegOffset baseI8H17.hwx baseI8H17.info 0 regNeOp
    let orig ← readU32LE baseI8H17.hwx fOp
    tamperM5aTaskReg baseI8H17.hwx baseI8H17.info 0 regNeOp (orig ||| 0x00003c00)
  ) "neOp unscaledBits (0xf) must be 0 for scaled integer weights")

  -- Tamper 5b: 0x4904 perLaneScale=1 on pal4 rejected
  results := results.push (checkQuantTamper "m8a neg tamper: 0x4904 perLaneScale=1 on pal4 rejected" (do
    let fOp ← findM5aTaskRegOffset basePal4H17.hwx basePal4H17.info 0 regNeOp
    let orig ← readU32LE basePal4H17.hwx fOp
    tamperM5aTaskReg basePal4H17.hwx basePal4H17.info 0 regNeOp (orig ||| 0x00004000)
  ) "neOp perLaneScale (bit 14) must be 0 for fp16 and palette weights")

  -- Tamper 5c: 0x4904 unscaled bits cleared on pal4 rejected
  results := results.push (checkQuantTamper "m8a neg tamper: 0x4904 unscaled bits cleared on pal4 rejected" (do
    let fOp ← findM5aTaskRegOffset basePal4H17.hwx basePal4H17.info 0 regNeOp
    let orig ← readU32LE basePal4H17.hwx fOp
    tamperM5aTaskReg basePal4H17.hwx basePal4H17.info 0 regNeOp (orig &&& (~~~0x00003c00))
  ) "neOp unscaledBits (0x0) must be 0xf (0x3c00) for fp16 and palette weights")

  -- Tamper 6a: 0x4910 non-zero low 16 bits when perLaneScale = 1
  results := results.push (checkQuantTamper "m8a neg tamper: 0x4910 non-zero low 16 bits with perLaneScale=1 rejected" (do
    tamperM5aTaskReg baseI8H17.hwx baseI8H17.info 0 regNeScale 0x00003c00
  ) "neScale low 16 bits (0x3c00) must be 0 when perLaneScale=1")

  -- Tamper 6b: 0x4910 zero per-tensor scale rejected
  results := results.push (checkQuantTamper "m8a neg tamper: 0x4910 zero per-tensor scale rejected" (do
    tamperM5aTaskReg baseI8PtH17.hwx baseI8PtH17.info 0 regNeScale 0x00000000
  ) "per-tensor neScale (0x0) must be a finite positive fp16 value")

  -- Tamper 6c: 0x4910 Inf / NaN / negative per-tensor scale rejected
  for (lbl, badVal, hexStr) in #[("+Inf", 0x00007c00, "0x7c00"), ("NaN", 0x00007e00, "0x7e00"), ("negative", 0x0000bc00, "0xbc00")] do
    results := results.push (checkQuantTamper s!"m8a neg tamper: 0x4910 {lbl} per-tensor scale rejected" (do
      tamperM5aTaskReg baseI8PtH17.hwx baseI8PtH17.info 0 regNeScale badVal
    ) s!"per-tensor neScale ({hexStr}) must be a finite positive fp16 value")

  -- Tamper 6d: 0x4910 pexp = 31 (> 30) rejected
  results := results.push (checkQuantTamper "m8a neg tamper: 0x4910 pexp=31 (> 30) rejected" (do
    tamperM5aTaskReg baseI8H17.hwx baseI8H17.info 0 regNeScale ((31 : UInt32) <<< 16)
  ) "neScale pexp (31) exceeds maximum 30")

  -- Tamper 7a0: Missing 0x490c on quantized int8 rejected (prevents biasShift carry-over!)
  results := results.push (checkQuantTamper "m8a neg tamper: missing 0x490c on quantized int8 rejected" (do
    stripRegAndVerify baseI8H17.hwx baseI8H17.info 32 32 8 16 regScalarBias
  ) "missing required register scalarBias (0x490c)")

  -- Tamper 7a1: Missing 0x490c on quantized pal4 rejected (prevents biasShift carry-over!)
  results := results.push (checkQuantTamper "m8a neg tamper: missing 0x490c on quantized pal4 rejected" (do
    stripRegAndVerify basePal4H17.hwx basePal4H17.info 32 32 8 16 regScalarBias
  ) "missing required register scalarBias (0x490c)")

  -- Tamper 7a: 0x490c biasShift = 16 (> 15) rejected
  results := results.push (checkQuantTamper "m8a neg tamper: 0x490c biasShift=16 (> 15) rejected" (do
    tamperM5aTaskReg baseBsH17.hwx baseBsH17.info 0 regScalarBias ((16 : UInt32) <<< 16)
  ) "scalarBias biasShift (16) exceeds maximum 15")

  -- Tamper 7a2: Integer 0x490c low 16 bits != 0 when biasInKernel=1 rejected
  results := results.push (checkQuantTamper "m8a neg tamper: 0x490c low 16 bits != 0 with biasInKernel=1 on int8 rejected" (do
    tamperM5aTaskReg baseI8H17.hwx baseI8H17.info 0 regScalarBias 0x00003c00
  ) "scalarBias low 16 bits (0x3c00) must be 0 when biasInKernel=1")

  -- Tamper 7a3: Integer 0x490c low 16 bits != 0 when biasInKernel=0 and Cout != 1 rejected
  results := results.push (checkQuantTamper "m8a neg tamper: 0x490c low 16 bits != 0 with biasInKernel=0 and Cout=32 rejected" (do
    tamperM5aTaskReg baseU4H17.hwx baseU4H17.info 0 regScalarBias 0x00003c00
  ) "scalarBias low 16 bits only allowed for Cout=1 when biasInKernel=0")

  -- Tamper 7a4: Integer 0x490c non-finite low 16 bits when biasInKernel=0 and Cout=1 rejected
  results := results.push (checkQuantTamper "m8a neg tamper: non-finite 0x490c scalarBias on Cout=1 bias-free int8 rejected" (do
    let cI8C1 ← compileConv1x1Quant .h17 { h := 8, w := 16, cin := 32, cout := 1 } {
      format := .int8, codes := mkQuantCodes 32 256 15, scales := #[0.0625]
    } none false
    tamperM5aTaskReg cI8C1.hwx cI8C1.info 0 regScalarBias f16QNaN.toUInt32
  ) "scalarBias low 16 bits (0x7e00) must be finite fp16")

  -- Tamper 7b: 0x490c biasShift = 1 on pal4 rejected
  results := results.push (checkQuantTamper "m8a neg tamper: 0x490c biasShift=1 on pal4 rejected" (do
    tamperM5aTaskReg basePal4H17.hwx basePal4H17.info 0 regScalarBias 0x00010000
  ) "scalarBias biasShift (1) must be 0 for palette weights")

  -- Tamper 7b1: 0x490c low 16 bits != 0 on pal4 when biasInKernel=1 or Cout != 1 rejected
  results := results.push (checkQuantTamper "m8a neg tamper: 0x490c low 16 bits != 0 on pal4 Cout=32 rejected" (do
    tamperM5aTaskReg basePal4H17.hwx basePal4H17.info 0 regScalarBias 0x00003c00
  ) "scalarBias low 16 bits (0x3c00) invalid for palette weights when biasInKernel=true or Cout=32")

  -- Tamper 7c: 0x4910 != 0x3c00 on pal4 rejected
  results := results.push (checkQuantTamper "m8a neg tamper: 0x4910 != 0x3c00 on pal4 rejected" (do
    tamperM5aTaskReg basePal4H17.hwx basePal4H17.info 0 regNeScale 0x00004000
  ) "!= expected fp16 1.0 (0x3c00) for palette weights")

  -- Tamper 7d: 0x4910 pexp = 1 on pal4 rejected
  results := results.push (checkQuantTamper "m8a neg tamper: 0x4910 pexp=1 on pal4 rejected" (do
    tamperM5aTaskReg basePal4H17.hwx basePal4H17.info 0 regNeScale 0x00013c00
  ) "neScale pexp (1) must be 0 for palette weights")

  -- Tamper 8a: Corrupted int4 decode table byte in __KERN_0
  results := results.push (checkQuantTamper "m8a neg tamper: corrupted int4 decode table byte in __KERN_0 rejected" (do
    let some kSec := baseI4H17.info.kernelSection | throw "missing kernel section"
    let badHwx ← putU8 baseI4H17.hwx (kSec.offset.toNat + 8) 0x77
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasksSummary badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "corrupted int4 decode table byte[8]")

  -- Tamper 8b: Non-zero padding byte in int4 64B table pad (byte 20)
  results := results.push (checkQuantTamper "m8a neg tamper: non-zero int4 decode table padding byte rejected" (do
    let some kSec := baseI4H17.info.kernelSection | throw "missing kernel section"
    let badHwx ← putU8 baseI4H17.hwx (kSec.offset.toNat + 20) 0x01
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasksSummary badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "non-zero int4 decode table padding byte[20]")

  -- Tamper 8a2: Corrupted uint4 decode table byte in __KERN_0
  results := results.push (checkQuantTamper "m8a neg tamper: corrupted uint4 decode table byte in __KERN_0 rejected" (do
    let some kSec := baseU4H17.info.kernelSection | throw "missing kernel section"
    let badHwx ← putU8 baseU4H17.hwx (kSec.offset.toNat + 8) 0x77
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasksSummary badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "corrupted uint4 decode table byte[8]")

  -- Tamper 8b2: Non-zero padding byte in uint4 64B table pad (byte 20)
  results := results.push (checkQuantTamper "m8a neg tamper: non-zero uint4 decode table padding byte rejected" (do
    let some kSec := baseU4H17.info.kernelSection | throw "missing kernel section"
    let badHwx ← putU8 baseU4H17.hwx (kSec.offset.toNat + 20) 0x01
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasksSummary badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "non-zero uint4 decode table padding byte[20]")

  -- Tamper 8c: NaN entry in pal4 palette table in __KERN_0
  results := results.push (checkQuantTamper "m8a neg tamper: NaN entry in pal4 table rejected" (do
    let some kSec := basePal4H17.info.kernelSection | throw "missing kernel section"
    let badHwx ← putU16LE basePal4H17.hwx (kSec.offset.toNat + 4) f16QNaN
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasksSummary badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "pal4 palette entry[2] (0x7e00) is non-finite (NaN/Inf)")

  -- Tamper 8d: Non-zero padding byte in pal4 64B table pad (byte 40)
  results := results.push (checkQuantTamper "m8a neg tamper: non-zero pal4 table padding byte rejected" (do
    let some kSec := basePal4H17.info.kernelSection | throw "missing kernel section"
    let badHwx ← putU8 basePal4H17.hwx (kSec.offset.toNat + 40) 0xff
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasksSummary badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "non-zero pal4 table padding byte[40]")

  -- Tamper 8e: Per-lane fp16 scale entry tampered to NaN / 0 / negative in __KERN_0 rejected
  for (lbl, badScale, hexStr) in #[("NaN", f16QNaN, "0x7e00"), ("zero", f16PosZero, "0x0"), ("negative", f16NegOne, "0xbc00")] do
    results := results.push (checkQuantTamper s!"m8a neg tamper: per-lane fp16 scale {lbl} in __KERN_0 rejected" (do
      let some kSec := baseI8H17.info.kernelSection | throw "missing kernel section"
      -- int8 has tableBytes=0, so block 0 per-lane scale[0] is at chunk offset 0!
      let badHwx ← putU16LE baseI8H17.hwx kSec.offset.toNat badScale
      let badInfo ← validateHwx badHwx
      let _ ← verifyHwxTasksSummary badHwx badInfo (requireKnownTask := true)
      pure ()
    ) s!"block 0 per-lane scale[0] ({hexStr}) must be a finite positive fp16 value")

  -- Tamper 8f: Non-finite scalarBias on Cout=1 bias-free pal4 rejected
  results := results.push (checkQuantTamper "m8a neg tamper: non-finite 0x490c scalarBias on Cout=1 pal4 rejected" (do
    let cPal1 ← compileConv1x1Quant .h17 { h := 8, w := 16, cin := 32, cout := 1 } {
      format := .pal 4, codes := mkQuantCodes 32 16 14, palette := mkQuantPalette 4
    } none false
    tamperM5aTaskReg cPal1.hwx cPal1.info 0 regScalarBias f16PosInf.toUInt32
  ) "scalarBias low 16 bits (0x7c00) must be finite fp16")

  -- Tamper 8g: Non-zero quantized chunk tail padding byte in __KERN_0 rejected
  results := results.push (checkQuantTamper "m8a neg tamper: non-zero quantized chunk tail padding byte rejected" (do
    -- Cout=1, Cin=32, int4 per-tensor no-bias: table=64B + 16B codes = 80B unpadded, 128B chunk -> tail pad is [80..127]
    let cTail ← compileConv1x1Quant .h17 { h := 8, w := 16, cin := 32, cout := 1 } {
      format := .int4, codes := mkQuantCodes 32 16 16, scales := #[0.125]
    } none false
    let some kSec := cTail.info.kernelSection | throw "missing kernel section"
    let badHwx ← putU8 cTail.hwx (kSec.offset.toNat + 80) 0x42
    let badInfo ← validateHwx badHwx
    let _ ← verifyHwxTasksSummary badHwx badInfo (requireKnownTask := true)
    pure ()
  ) "non-zero quantized chunk tail padding byte[80]=66")

  -- Tamper 9: Truncated/mismatched chunkSize[0] on quantized conv
  results := results.push (checkQuantTamper "m8a neg tamper: mismatched chunkSize[0] on quantized conv rejected" (do
    let fSz ← findM5aTaskRegOffset baseI4H17.hwx baseI4H17.info 0 (regChunkSize 0)
    let orig ← readU32LE baseI4H17.hwx fSz
    tamperM5aTaskReg baseI4H17.hwx baseI4H17.info 0 (regChunkSize 0) (orig - 64)
  ) "quantized size")

  pure results

-- ============================================================================
-- Milestone C1: Parameterized SSA TensorIR (HighOp -> CoreOp), Rewriter & PTQ
-- ============================================================================

namespace C1Tests

open Vane.IR
open Vane.Quant

private def pbByteTensor (name : String) (dtype : Int) (dims : Array Int) (bytes : ByteArray) : ProtoBuilder :=
  (((ProtoBuilder.empty.writeString 8 name).writeInt64 2 dtype).writePackedInt64 1 dims).writeBytes 9 bytes

private def lowerOnnxBytes (bytes : ByteArray) (opts : LowerOptions := {}) : Except String LowerResult := do
  lowerOnnx (← parseOnnx bytes) opts

private def compileOnnxBytes (bytes : ByteArray) (opts : LowerOptions := {}) : Except String CompiledOnnx := do
  compileOnnx (← parseOnnx bytes) opts

private def mkTestIntTensor (sh : List Nat) (f : Nat → Int) : TensorVal Int :=
  { shape := sh, data := fun idx => f idx.val }

private def evalTensorToList (t : TensorVal Int) : List Int :=
  (List.range (shapeProd t.shape)).map fun i =>
    if h : i < shapeProd t.shape then t.data ⟨i, h⟩ else 0

private def runC1Tests : IO (Array TestResult) := do
  let mut results : Array TestResult := #[]

  -- 1. SSA Graph Validation: positive multi-input/multi-output + negative rejection rules
  results := results.push (checkOk "c1 validate: valid multi-input multi-output SSAGraph CoreOp & HighOp" (do
    let ty : TensorType := { shape := #[1, 8, 4, 4], dtype := .fp16 }
    let gCore : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "a", ty), (1, "b", ty)]
      nodes := #[
        { name := "add", op := .elemBinary .add, inputs := #[0, 1], outputs := #[(2, ty)] },
        { name := "sub", op := .elemBinary .sub, inputs := #[0, 1], outputs := #[(3, ty)] }
      ]
      outputs := #[2, 3]
    }
    validateGraph gCore
    let gHigh : SSAGraph HighOp := {
      target := .h16
      inputs := #[(0, "x", ty)]
      nodes := #[
        { name := "sp", op := .split 1 #[4, 4], inputs := #[0],
          outputs := #[(1, { shape := #[1, 4, 4, 4], dtype := .fp16 }),
                       (2, { shape := #[1, 4, 4, 4], dtype := .fp16 })] }
      ]
      outputs := #[1, 2]
    }
    validateGraph gHigh
  ))

  results := results.push (checkErr "c1 validate neg: non-dense input ValueIds rejected" (do
    let ty : TensorType := { shape := #[1, 4, 4, 4], dtype := .fp16 }
    let g : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(1, "x", ty)]
      nodes := #[]
      outputs := #[1]
    }
    validateGraph g
  ) "dense")

  results := results.push (checkErr "c1 validate neg: zero dimension in input shape rejected" (do
    let g : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", { shape := #[1, 0, 4, 4], dtype := .fp16 })]
      nodes := #[]
      outputs := #[0]
    }
    validateGraph g
  ) "positive")

  results := results.push (checkErr "c1 validate neg: duplicate output ValueId rejected" (do
    let ty : TensorType := { shape := #[1, 4, 4, 4], dtype := .fp16 }
    let g : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", ty)]
      nodes := #[
        { name := "n0", op := .elemUnary .abs, inputs := #[0], outputs := #[(1, ty)] },
        { name := "n1", op := .elemUnary .abs, inputs := #[0], outputs := #[(1, ty)] }
      ]
      outputs := #[1]
    }
    validateGraph g
  ) "duplicate")

  results := results.push (checkErr "c1 validate neg: use-before-def / cycle rejected" (do
    let ty : TensorType := { shape := #[1, 4, 4, 4], dtype := .fp16 }
    let g : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", ty)]
      nodes := #[
        { name := "bad", op := .elemBinary .add, inputs := #[0, 2], outputs := #[(1, ty)] },
        { name := "late", op := .elemUnary .abs, inputs := #[1], outputs := #[(2, ty)] }
      ]
      outputs := #[2]
    }
    validateGraph g
  ) "not defined before node")

  results := results.push (checkErr "c1 validate neg: undefined graph output rejected" (do
    let ty : TensorType := { shape := #[1, 4, 4, 4], dtype := .fp16 }
    let g : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", ty)]
      nodes := #[]
      outputs := #[99]
    }
    validateGraph g
  ) "not defined in graph")

  -- 2. Polymorphic DCE on multi-output SSA graph
  results := results.push (checkOk "c1 dce: multi-output SSA graph drops dead branches and preserves live outputs" (do
    let ty : TensorType := { shape := #[1, 8, 4, 4], dtype := .fp16 }
    let g : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", ty)]
      nodes := #[
        { name := "shared", op := .elemUnary .abs, inputs := #[0], outputs := #[(1, ty)] },
        { name := "dead1", op := .elemUnary .square, inputs := #[1], outputs := #[(2, ty)] },
        { name := "dead2", op := .elemBinary .add, inputs := #[2, 0], outputs := #[(3, ty)] },
        { name := "liveA", op := .lut .relu, inputs := #[1], outputs := #[(4, ty)] },
        { name := "liveB", op := .elemBinary .mul, inputs := #[1, 0], outputs := #[(5, ty)] }
      ]
      outputs := #[4, 5]
    }
    validateGraph g
    let gDce := dce g
    validateGraph gDce
    if gDce.nodes.size != 3 then
      throw s!"expected 3 live nodes after DCE, got {gDce.nodes.size}"
    if gDce.outputs != #[4, 5] then
      throw s!"unexpected outputs after DCE: {repr gDce.outputs}"
    let names := gDce.nodes.map (·.name)
    if names != #["shared", "liveA", "liveB"] then
      throw s!"unexpected kept node names: {repr names}"
  ))

  -- 3. Polymorphic ViewSimplify: Transpose(Transpose) / Reshape(Reshape) / full-span Slice
  results := results.push (checkOk "c1 viewSimplify: ONNX inverse Transpose, round-trip Reshape & full-span Slice compile to 0 DMA view tasks" (do
    let w1 := mkDetF32 (16 * 16) 0.05
    let b1 := mkDetF32 16 0.01
    for tgt in #[Target.h16, Target.h17] do
      -- (a) Transpose([0,2,3,1]) followed by inverse Transpose([0,3,1,2]) -> Conv1x1
      let gTr := pbGraph "tr_cancel"
        #[pbVI "x" #[1, 16, 8, 8]]
        #[pbVI "y" #[1, 16, 8, 8]]
        #[pbF32Tensor "w" #[16, 16, 1, 1] w1, pbF32Tensor "b" #[16] b1]
        #[pbNode "Transpose" "t1" #["x"] #["h1"] #[pbAttrInts "perm" #[0, 2, 3, 1]],
          pbNode "Transpose" "t2" #["h1"] #["h2"] #[pbAttrInts "perm" #[0, 3, 1, 2]],
          pbNode "Conv" "conv" #["h2", "w", "b"] #["y"]]
      let coTr ← compileOnnxBytes (pbWrapModel 18 gTr) { target := tgt }
      let sTr ← verifyCompiledOnnx coTr
      if sTr.neView != 0 || sTr.neConv != 1 || sTr.total != 1 then
        throw s!"{tgt.name} inverse Transpose not eliminated: summary={repr sTr}"

      -- (b) Reshape([1,8,16,8]) followed by Reshape([1,16,8,8]) -> Conv1x1
      let gRs := pbGraph "rs_cancel"
        #[pbVI "x" #[1, 16, 8, 8]]
        #[pbVI "y" #[1, 16, 8, 8]]
        #[pbI64Tensor "s1" #[4] #[1, 8, 16, 8],
          pbI64Tensor "s2" #[4] #[1, 16, 8, 8],
          pbF32Tensor "w" #[16, 16, 1, 1] w1, pbF32Tensor "b" #[16] b1]
        #[pbNode "Reshape" "r1" #["x", "s1"] #["h1"],
          pbNode "Reshape" "r2" #["h1", "s2"] #["h2"],
          pbNode "Conv" "conv" #["h2", "w", "b"] #["y"]]
      let coRs ← compileOnnxBytes (pbWrapModel 18 gRs) { target := tgt }
      let sRs ← verifyCompiledOnnx coRs
      if sRs.neView != 0 || sRs.neConv != 1 || sRs.total != 1 then
        throw s!"{tgt.name} round-trip Reshape not eliminated: summary={repr sRs}"

      -- (c) Full-span Slice([0,0,0,0]..[1,16,8,8], step=1) -> Conv1x1
      let gSl := pbGraph "sl_cancel"
        #[pbVI "x" #[1, 16, 8, 8]]
        #[pbVI "y" #[1, 16, 8, 8]]
        #[pbI64Tensor "starts" #[4] #[0, 0, 0, 0],
          pbI64Tensor "ends" #[4] #[1, 16, 8, 8],
          pbI64Tensor "axes" #[4] #[0, 1, 2, 3],
          pbI64Tensor "steps" #[4] #[1, 1, 1, 1],
          pbF32Tensor "w" #[16, 16, 1, 1] w1, pbF32Tensor "b" #[16] b1]
        #[pbNode "Slice" "sl" #["x", "starts", "ends", "axes", "steps"] #["h1"],
          pbNode "Conv" "conv" #["h1", "w", "b"] #["y"]]
      let coSl ← compileOnnxBytes (pbWrapModel 18 gSl) { target := tgt }
      let sSl ← verifyCompiledOnnx coSl
      if sSl.neView != 0 || sSl.neConv != 1 || sSl.total != 1 then
        throw s!"{tgt.name} full-span Slice not eliminated: summary={repr sSl}"
  ))

  results := results.push (checkOk "c1 viewSimplify: non-identity Transpose(Transpose(x, p1), p2) composes to single Transpose" (do
    let t0 : TensorType := { shape := #[1, 8, 4, 6], dtype := .fp16 }
    let t1 : TensorType := { shape := #[1, 4, 6, 8], dtype := .fp16 }
    let t2 : TensorType := { shape := #[1, 6, 8, 4], dtype := .fp16 }
    let g : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", t0)]
      nodes := #[
        { name := "tr1", op := .transpose #[0, 2, 3, 1], inputs := #[0], outputs := #[(1, t1)] },
        { name := "tr2", op := .transpose #[0, 2, 3, 1], inputs := #[1], outputs := #[(2, t2)] }
      ]
      outputs := #[2]
    }
    validateGraph g
    let gSimp := viewSimplify g
    validateGraph gSimp
    if gSimp.nodes.size != 1 then
      throw s!"expected 1 composed transpose node, got {gSimp.nodes.size}"
    match gSimp.nodes[0]!.op with
    | .transpose perm =>
      if perm != #[0, 3, 1, 2] then
        throw s!"expected composed perm #[0, 3, 1, 2], got {repr perm}"
    | other => throw s!"expected transpose op, got {other.pretty}"
  ))

  -- 4. Polymorphic ConstFold: Transpose(const), Reshape(const), scalar fold & fallback dequantize
  results := results.push (checkOk "c1 constFold: Transpose(const), Reshape(const), scalar math & fallback dequantize" (do
    let vals : Array Float := #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let g1 : SSAGraph CoreOp := {
      target := .h17
      inputs := #[]
      nodes := #[
        { name := "c", op := .const (.f32 vals), inputs := #[],
          outputs := #[(0, { shape := #[2, 3], dtype := .f32 })] },
        { name := "tr", op := .transpose #[1, 0], inputs := #[0],
          outputs := #[(1, { shape := #[3, 2], dtype := .f32 })] },
        { name := "rs", op := .reshape #[1, 6], inputs := #[1],
          outputs := #[(2, { shape := #[1, 6], dtype := .f32 })] }
      ]
      outputs := #[2]
    }
    validateGraph g1
    let g1f ← constFold g1
    let g1d := dce g1f
    validateGraph g1d
    if g1d.nodes.size != 1 then
      throw s!"expected 1 folded const node, got {g1d.nodes.size}"
    match g1d.nodes[0]!.op with
    | .const (.f32 folded) =>
      let expected : Array Float := #[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
      if folded != expected then
        throw s!"unexpected folded transpose/reshape values: {repr folded}"
    | other => throw s!"expected folded f32 const, got {other.pretty}"

    -- Scalar binary + unary constant folding
    let sTy : TensorType := { shape := #[1], dtype := .f32 }
    let g2 : SSAGraph CoreOp := {
      target := .h17
      inputs := #[]
      nodes := #[
        { name := "a", op := .const (.f32 #[1.5]), inputs := #[], outputs := #[(0, sTy)] },
        { name := "b", op := .const (.f32 #[2.5]), inputs := #[], outputs := #[(1, sTy)] },
        { name := "sum", op := .elemBinary .add, inputs := #[0, 1], outputs := #[(2, sTy)] },
        { name := "neg", op := .elemUnary .neg, inputs := #[2], outputs := #[(3, sTy)] }
      ]
      outputs := #[3]
    }
    let g2d := dce (← constFold g2)
    validateGraph g2d
    if g2d.nodes.size != 1 then
      throw s!"expected 1 scalar folded const node, got {g2d.nodes.size}"
    match g2d.nodes[0]!.op with
    | .const (.f32 #[v]) =>
      if Float.abs (v - (-4.0)) > 1.0e-5 then
        throw s!"expected folded scalar -4.0, got {v}"
    | other => throw s!"expected folded scalar -4.0, got {other.pretty}"

    -- Fallback dequantize(const quant) feeding non-quant-capable op folds into fp16 const
    let qSpec : QuantWeightSpec := {
      format := .int8
      codes := ByteArray.mk #[2, 254, 4, 0] -- signed: [2, -2, 4, 0]
      scales := #[0.5, 0.25]
    }
    let xTy : TensorType := { shape := #[2, 1, 1, 2], dtype := .fp16 }
    let g3 : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", xTy)]
      nodes := #[
        { name := "qw", op := .const (.quant qSpec), inputs := #[],
          outputs := #[(1, { shape := #[2, 1, 1, 2], dtype := .i8 })] },
        { name := "dq", op := .dequantize .int8, inputs := #[1],
          outputs := #[(2, xTy)] },
        { name := "add", op := .elemBinary .add, inputs := #[0, 2], outputs := #[(3, xTy)] }
      ]
      outputs := #[3]
    }
    let g3d := dce (← constFold g3)
    validateGraph g3d
    if g3d.nodes.any (fun nd => match nd.op with | .dequantize _ => true | _ => false) then
      throw "expected fallback dequantize feeding elemBinary.add to fold into fp16 const"
    match g3d.nodes[0]!.op with
    | .const (.fp16 bs) =>
      let got := (List.range 4).map fun i =>
        f16ToFloat ((bs.get! (2 * i)).toUInt16 ||| ((bs.get! (2 * i + 1)).toUInt16 <<< 8))
      if got != [1.0, -1.0, 1.0, 0.0] then
        throw s!"unexpected fallback dequantized fp16 values: {repr got}"
    | other => throw s!"expected folded fp16 const, got {other.pretty}"
  ))

  -- 5. HighOp.rmsNorm decomposition + HighFuse + CoreFuse end-to-end
  results := results.push (checkOk "c1 pipeline: HighOp.rmsNorm decomposes cleanly and compiles to verified HWX on H16 & H17" (do
    let ty : TensorType := { shape := #[1, 8, 4, 32], dtype := .fp16 }
    let scaleTy : TensorType := { shape := #[1, 1, 1, 32], dtype := .fp16 }
    let scaleVals : Array Float := Array.replicate 32 1.25
    for tgt in #[Target.h16, Target.h17] do
      let gHigh : SSAGraph HighOp := {
        target := tgt
        inputs := #[(0, "x", ty)]
        nodes := #[
          { name := "gamma", op := .core (.const (.f32 scaleVals)), inputs := #[], outputs := #[(1, scaleTy)] },
          { name := "rms", op := .rmsNorm (-1) 1.0e-5 true, inputs := #[0, 1], outputs := #[(2, ty)] }
        ]
        outputs := #[2]
      }
      let pr ← runStandardPipeline gHigh {}
      validateGraph pr.coreGraph
      let bridge ← Vane.IR.ToGraph.coreGraphToGraphSpec pr.coreGraph
      let compiled ← Vane.Graph.compileGraph bridge.spec
      validateSections compiled.hwx compiled.sections
      let s ← verifyHwxTasksSummary compiled.hwx compiled.info (requireKnownTask := true)
      if s.peReduce != 1 || s.total != compiled.taskCount then
        throw s!"{tgt.name} HighOp.rmsNorm unexpected verified summary={repr s}"
  ))

  -- 6. ONNX DequantizeLinear (INT8 & UINT8 with ZP) -> native M8A quantized Conv1x1 & Conv3x3
  results := results.push (checkOk "c1 dequantizeLinear: ONNX INT8 & UINT8(zp) DequantizeLinear -> native M8A quantized HWX on H16 & H17" (do
    let wI8 := ByteArray.mk (Array.ofFn (n := 32 * 32) fun i => ((i.val * 7 + 3) % 251).toUInt8)
    let sc32 := Array.ofFn (n := 32) fun i => 0.03125 * Float.ofNat ((i.val % 4) + 1)
    let b32 := mkDetF32 32 0.01

    let wU8 := ByteArray.mk (Array.ofFn (n := 16 * 16 * 9) fun i => ((i.val * 5 + 17) % 256).toUInt8)
    let sc16 := Array.ofFn (n := 16) fun i => 0.0625 * Float.ofNat ((i.val % 3) + 1)
    let zp16 := ByteArray.mk (Array.ofFn (n := 16) fun i => (120 + (i.val % 15)).toUInt8)
    let b16 := mkDetF32 16 0.01

    for tgt in #[Target.h16, Target.h17] do
      -- (a) INT8 per-channel DequantizeLinear -> Conv1x1
      let gI8 := pbGraph "dq_int8_conv1x1"
        #[pbVI "x" #[1, 32, 8, 16]]
        #[pbVI "y" #[1, 32, 8, 16]]
        #[pbByteTensor "wq" onnxInt8 #[32, 32, 1, 1] wI8,
          pbF32Tensor "sc" #[32] sc32,
          pbF32Tensor "b" #[32] b32]
        #[pbNode "DequantizeLinear" "dq" #["wq", "sc"] #["w"] #[pbAttrInt "axis" 0],
          pbNode "Conv" "conv" #["x", "w", "b"] #["y"]]
      let coI8 ← compileOnnxBytes (pbWrapModel 18 gI8) { target := tgt }
      let sI8 ← verifyCompiledOnnx coI8
      if coI8.lowered.summary.quantizedConvCount != 1 || sI8.neConv != 1 then
        throw s!"{tgt.name} INT8 DequantizeLinear expected 1 quantized conv, got qCnt={coI8.lowered.summary.quantizedConvCount} s={repr sI8}"

      -- (b) UINT8 per-channel with zero-point DequantizeLinear -> Conv3x3
      let gU8 := pbGraph "dq_uint8_conv3x3"
        #[pbVI "x" #[1, 16, 8, 16]]
        #[pbVI "y" #[1, 16, 8, 16]]
        #[pbByteTensor "wq" onnxUint8 #[16, 16, 3, 3] wU8,
          pbF32Tensor "sc" #[16] sc16,
          pbByteTensor "zp" onnxUint8 #[16] zp16,
          pbF32Tensor "b" #[16] b16]
        #[pbNode "DequantizeLinear" "dq" #["wq", "sc", "zp"] #["w"] #[pbAttrInt "axis" 0],
          pbNode "Conv" "conv" #["x", "w", "b"] #["y"] #[pbAttrInts "pads" #[1, 1, 1, 1]]]
      let coU8 ← compileOnnxBytes (pbWrapModel 18 gU8) { target := tgt }
      let sU8 ← verifyCompiledOnnx coU8
      if coU8.lowered.summary.quantizedConvCount != 1 || sU8.neConv != 1 then
        throw s!"{tgt.name} UINT8 DequantizeLinear expected 1 quantized conv, got qCnt={coU8.lowered.summary.quantizedConvCount} s={repr sU8}"
  ))

  -- 7. Post-Training Quantization (--quantize int8/int4/pal4/pal2) + __KERN_0 compression + M8A LUT unfusion
  results := results.push (checkOk "c1 ptq: --quantize int8/int4/pal4/pal2 compiles verified HWX, unfuses LUT, compresses __KERN_0 on H16 & H17" (do
    let w1 := mkDetF32 (64 * 64) 0.05
    let b1 := mkDetF32 64 0.01
    let w2 := mkDetF32 (64 * 64 * 9) 0.03
    let b2 := mkDetF32 64 0.01
    let gPtq := pbGraph "ptq_two_conv"
      #[pbVI "x" #[1, 64, 8, 16]]
      #[pbVI "y" #[1, 64, 8, 16]]
      #[pbF32Tensor "w1" #[64, 64, 1, 1] w1, pbF32Tensor "b1" #[64] b1,
        pbF32Tensor "w2" #[64, 64, 3, 3] w2, pbF32Tensor "b2" #[64] b2]
      #[pbNode "Conv" "c1" #["x", "w1", "b1"] #["h1"],
        pbNode "Gelu" "g1" #["h1"] #["h2"],
        pbNode "Conv" "c2" #["h2", "w2", "b2"] #["h3"] #[pbAttrInts "pads" #[1, 1, 1, 1]],
        pbNode "Relu" "r2" #["h3"] #["y"]]
    let modelBytes := pbWrapModel 20 gPtq
    for tgt in #[Target.h16, Target.h17] do
      let coFp16 ← compileOnnxBytes modelBytes { target := tgt }
      let sFp16 ← verifyCompiledOnnx coFp16
      -- In FP16, c1+Gelu fuses into 1 neConv and c2+Relu fuses into 1 neConv (2 total tasks)
      if sFp16.neConv != 2 || sFp16.neAct != 0 then
        throw s!"{tgt.name} FP16 baseline expected 2 fused convs, got {repr sFp16}"

      let mut kernSizes : Array (String × Nat) := #[]
      for (lbl, fmt) in #[("int8", QuantFormat.int8), ("int4", .int4), ("pal4", .pal 4), ("pal2", .pal 2)] do
        let coQ ← compileOnnxBytes modelBytes { target := tgt, quantize := some fmt }
        let sQ ← verifyCompiledOnnx coQ
        -- M8A rule: Gelu LUT must be unfused into a separate neAct task on quantized conv, while Relu stays fused!
        if coQ.lowered.summary.quantizedConvCount != 2 || sQ.neConv != 2 || sQ.neAct != 1 then
          throw s!"{tgt.name} PTQ {lbl} expected 2 quantized convs + 1 unfused Gelu LUT, got qCnt={coQ.lowered.summary.quantizedConvCount} s={repr sQ}"
        kernSizes := kernSizes.push (lbl, coQ.compiled.kernelBytes)

      let kFp16 := coFp16.compiled.kernelBytes
      let kI8 := kernSizes[0]!.2
      let kI4 := kernSizes[1]!.2
      let kP4 := kernSizes[2]!.2
      let kP2 := kernSizes[3]!.2
      if !(kP2 < kI4 && kI4 < kI8 && kI8 < kFp16 && kP4 < kI8) then
        throw s!"{tgt.name} expected __KERN_0 compression pal2({kP2}) < int4({kI4}) < int8({kI8}) < fp16({kFp16}) and pal4({kP4}) < int8({kI8})"
  ))

  -- 8. IR Dump (dumpHigh / dumpCore) & Polymorphic evalCore on TensorVal Int
  results := results.push (checkOk "c1 semantics & dump-ir: dumpHigh/dumpCore formatting + exact TensorVal Int evalCore equivalence" (do
    let w1 := mkDetF32 (16 * 16) 0.05
    let gOnnx := pbGraph "dump_check"
      #[pbVI "x" #[1, 16, 4, 8]]
      #[pbVI "y" #[1, 16, 4, 8]]
      #[pbF32Tensor "w" #[16, 16, 1, 1] w1]
      #[pbNode "Conv" "c" #["x", "w"] #["h"], pbNode "Gelu" "g" #["h"] #["y"]]
    let lr ← lowerOnnxBytes (pbWrapModel 20 gOnnx) { target := .h17 }
    let hiDump := lr.highGraph.dumpHigh
    let coDump := lr.coreGraph.dumpCore
    if (hiDump.splitOn "gelu").length <= 1 || (hiDump.splitOn "graph (target=h17)").length <= 1 then
      throw s!"dumpHigh missing expected HighOp content:\n{hiDump}"
    if (coDump.splitOn "conv").length <= 1 || (coDump.splitOn "return").length <= 1 then
      throw s!"dumpCore missing expected CoreOp content:\n{coDump}"

    -- Executable polymorphic evalCore on TensorVal Int verifying exact equality across DCE + ViewSimplify
    let t0 : TensorType := { shape := #[2, 3, 2], dtype := .fp16 }
    let t1 : TensorType := { shape := #[3, 2, 2], dtype := .fp16 }
    let t2 : TensorType := { shape := #[6, 2], dtype := .fp16 }
    let gView : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", t0)]
      nodes := #[
        { name := "tr1", op := .transpose #[1, 2, 0], inputs := #[0], outputs := #[(1, t1)] },
        { name := "tr2", op := .transpose #[2, 0, 1], inputs := #[1], outputs := #[(2, t0)] },
        { name := "dead", op := .reshape #[12], inputs := #[1], outputs := #[(3, { shape := #[12], dtype := .fp16 })] },
        { name := "sl", op := .slice #[0, 0, 0] #[2, 3, 2] #[1, 1, 1], inputs := #[2], outputs := #[(4, t0)] },
        { name := "rs1", op := .reshape #[4, 3], inputs := #[4], outputs := #[(5, { shape := #[4, 3], dtype := .fp16 })] },
        { name := "rs2", op := .reshape #[6, 2], inputs := #[5], outputs := #[(6, t2)] }
      ]
      outputs := #[6]
    }
    validateGraph gView
    let gOpt := viewSimplify (dce gView)
    validateGraph gOpt
    if gOpt.nodes.size != 1 then
      throw s!"expected viewSimplify(dce) to collapse 6-node view chain to 1 reshape, got {gOpt.nodes.size}"
    let sem : OpSemantics CoreOp Int := mkCoreSemantics (fun _ _ _ => none) (fun _ _ _ _ h => by cases h)
    let xVal := mkTestIntTensor [2, 3, 2] (fun i => (i : Int) * 3 - 5)
    let env : Env Int := Env.set Env.empty 0 xVal
    let some [outOrig] := evalGraphOutputs sem gView env | throw "evalGraphOutputs failed on original graph"
    let some [outOpt] := evalGraphOutputs sem gOpt env | throw "evalGraphOutputs failed on optimized graph"
    if outOrig.shape != outOpt.shape || evalTensorToList outOrig != evalTensorToList outOpt then
      throw s!"evalCore mismatch: orig={outOrig.shape}/{evalTensorToList outOrig} opt={outOpt.shape}/{evalTensorToList outOpt}"
  ))

  -- 9. Regression tests for review Blockers 1–5 & cleanups

  -- Blocker 1: DequantizeLinear negative validation tests
  let wI8Dq := ByteArray.mk (Array.ofFn (n := 32 * 32) fun i => ((i.val * 7 + 3) % 251).toUInt8)
  let sc32Dq := Array.ofFn (n := 32) fun i => 0.03125 * Float.ofNat ((i.val % 4) + 1)

  results := results.push (checkErr "c1 reg neg: DequantizeLinear block_size != 0 rejected" (do
    let g := pbGraph "dq_bs" #[pbVI "x" #[1, 32, 8, 8]] #[pbVI "y" #[1, 32, 8, 8]]
      #[pbByteTensor "wq" onnxInt8 #[32, 32, 1, 1] wI8Dq, pbF32Tensor "sc" #[32] sc32Dq]
      #[pbNode "DequantizeLinear" "dq" #["wq", "sc"] #["w"] #[pbAttrInt "axis" 0, pbAttrInt "block_size" 16],
        pbNode "Conv" "conv" #["x", "w"] #["y"]]
    compileOnnxBytes (pbWrapModel 21 g) { target := .h17 }
  ) "block_size=16 is not supported")

  results := results.push (checkErr "c1 reg neg: DequantizeLinear per-channel default axis=1 rejected" (do
    let g := pbGraph "dq_axis_def" #[pbVI "x" #[1, 32, 8, 8]] #[pbVI "y" #[1, 32, 8, 8]]
      #[pbByteTensor "wq" onnxInt8 #[32, 32, 1, 1] wI8Dq, pbF32Tensor "sc" #[32] sc32Dq]
      #[pbNode "DequantizeLinear" "dq" #["wq", "sc"] #["w"],
        pbNode "Conv" "conv" #["x", "w"] #["y"]]
    compileOnnxBytes (pbWrapModel 18 g) { target := .h17 }
  ) "only axis=0")

  results := results.push (checkErr "c1 reg neg: DequantizeLinear explicit axis=1 rejected" (do
    let g := pbGraph "dq_axis1" #[pbVI "x" #[1, 32, 8, 8]] #[pbVI "y" #[1, 32, 8, 8]]
      #[pbByteTensor "wq" onnxInt8 #[32, 32, 1, 1] wI8Dq, pbF32Tensor "sc" #[32] sc32Dq]
      #[pbNode "DequantizeLinear" "dq" #["wq", "sc"] #["w"] #[pbAttrInt "axis" 1],
        pbNode "Conv" "conv" #["x", "w"] #["y"]]
    compileOnnxBytes (pbWrapModel 18 g) { target := .h17 }
  ) "only axis=0")

  results := results.push (checkErr "c1 reg neg: DequantizeLinear zero_point dataType mismatch rejected" (do
    let zpU8 := ByteArray.mk (Array.replicate 32 128)
    let g := pbGraph "dq_zp_dtype" #[pbVI "x" #[1, 32, 8, 8]] #[pbVI "y" #[1, 32, 8, 8]]
      #[pbByteTensor "wq" onnxInt8 #[32, 32, 1, 1] wI8Dq,
        pbF32Tensor "sc" #[32] sc32Dq,
        pbByteTensor "zp" onnxUint8 #[32] zpU8]
      #[pbNode "DequantizeLinear" "dq" #["wq", "sc", "zp"] #["w"] #[pbAttrInt "axis" 0],
        pbNode "Conv" "conv" #["x", "w"] #["y"]]
    compileOnnxBytes (pbWrapModel 18 g) { target := .h17 }
  ) "x_zero_point dataType")

  results := results.push (checkErr "c1 reg neg: DequantizeLinear zero_point shape != scale shape rejected" (do
    let zpI8 := ByteArray.mk (Array.replicate 32 0)
    let g := pbGraph "dq_zp_shape" #[pbVI "x" #[1, 32, 8, 8]] #[pbVI "y" #[1, 32, 8, 8]]
      #[pbByteTensor "wq" onnxInt8 #[32, 32, 1, 1] wI8Dq,
        pbF32Tensor "sc" #[1] #[0.0625],
        pbByteTensor "zp" onnxInt8 #[32] zpI8]
      #[pbNode "DequantizeLinear" "dq" #["wq", "sc", "zp"] #["w"] #[pbAttrInt "axis" 0],
        pbNode "Conv" "conv" #["x", "w"] #["y"]]
    compileOnnxBytes (pbWrapModel 18 g) { target := .h17 }
  ) "x_zero_point shape")

  results := results.push (checkErr "c1 reg neg: DequantizeLinear bad 2D x_scale shape rejected" (do
    let g := pbGraph "dq_sc_2d" #[pbVI "x" #[1, 32, 8, 8]] #[pbVI "y" #[1, 32, 8, 8]]
      #[pbByteTensor "wq" onnxInt8 #[32, 32, 1, 1] wI8Dq, pbF32Tensor "sc" #[32, 1] sc32Dq]
      #[pbNode "DequantizeLinear" "dq" #["wq", "sc"] #["w"] #[pbAttrInt "axis" 0],
        pbNode "Conv" "conv" #["x", "w"] #["y"]]
    compileOnnxBytes (pbWrapModel 18 g) { target := .h17 }
  ) "x_scale shape")

  -- Blocker 2: 4D MatMul with folded Transpose([0,1,3,2]) when M=1, N=1 preserves exact [1,8,1,1] output
  results := results.push (checkOk "c1 reg: 4D MatMul [1,8,1,16] @ Transpose([1,8,1,16],[0,1,3,2]) compiles with exact shape [1,8,1,1]" (do
    let gMm := pbGraph "mm4d_m1n1"
      #[pbVI "q" #[1, 8, 1, 16], pbVI "k" #[1, 8, 1, 16]]
      #[pbVI "y" #[1, 8, 1, 1]]
      #[]
      #[pbNode "Transpose" "kt" #["k"] #["k_t"] #[pbAttrInts "perm" #[0, 1, 3, 2]],
        pbNode "MatMul" "qk" #["q", "k_t"] #["y"]]
    for tgt in #[Target.h16, Target.h17] do
      let co ← compileOnnxBytes (pbWrapModel 18 gMm) { target := tgt }
      let s ← verifyCompiledOnnx co
      if co.lowered.summary.outputShape != { n := 1, c := 8, h := 1, w := 1 } then
        throw s!"{tgt.name} expected outputShape [1,8,1,1], got {repr co.lowered.summary.outputShape}"
      if co.lowered.coreGraph.nodes.size != 1 || co.lowered.summary.fusedMatmulCount != 1 || s.neMatmul != 1 then
        throw s!"{tgt.name} expected 1 folded coreGraph matmul node & 1 verified neMatmul task, got nodes={co.lowered.coreGraph.nodes.size} fused={co.lowered.summary.fusedMatmulCount} s={repr s}"
  ))

  -- Blocker 3: 4D Conv + 1D/2D Add constant (W-axis broadcast) must NOT misfold into Conv channel bias
  results := results.push (checkOk "c1 reg: 4D Conv + 1D/2D Add constant is never misfolded into Conv bias (while [1,C,1,1] and 2D [C] do fold)" (do
    let wTy : TensorType := { shape := #[16, 16, 1, 1], dtype := .f32 }
    let wPay := ConstPayload.f32 (mkDetF32 (16 * 16) 0.05)
    let bPay := ConstPayload.f32 (mkDetF32 16 0.01)
    let x4Ty : TensorType := { shape := #[1, 16, 4, 16], dtype := .fp16 }
    -- (a) 4D Conv + 1D [16] must NOT fold
    let g4d1d : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", x4Ty)]
      nodes := #[
        { name := "w", op := .const wPay, inputs := #[], outputs := #[(1, wTy)] },
        { name := "b", op := .const bPay, inputs := #[], outputs := #[(2, { shape := #[16], dtype := .f32 })] },
        { name := "c", op := .conv 1 1 1 1 #[0, 0, 0, 0] 1 false .none, inputs := #[0, 1], outputs := #[(3, x4Ty)] },
        { name := "a", op := .elemBinary .add, inputs := #[3, 2], outputs := #[(4, x4Ty)] }
      ]
      outputs := #[4]
    }
    let (g4d1dF, _, _) ← coreFuse g4d1d
    if !g4d1dF.nodes.any (fun nd => nd.op == .elemBinary .add) then
      throw "4D Conv + 1D [16] Add was erroneously folded into Conv bias!"

    -- (b) 4D Conv + 2D [1, 16] must NOT fold
    let g4d2d : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", x4Ty)]
      nodes := #[
        { name := "w", op := .const wPay, inputs := #[], outputs := #[(1, wTy)] },
        { name := "b", op := .const bPay, inputs := #[], outputs := #[(2, { shape := #[1, 16], dtype := .f32 })] },
        { name := "c", op := .conv 1 1 1 1 #[0, 0, 0, 0] 1 false .none, inputs := #[0, 1], outputs := #[(3, x4Ty)] },
        { name := "a", op := .elemBinary .add, inputs := #[3, 2], outputs := #[(4, x4Ty)] }
      ]
      outputs := #[4]
    }
    let (g4d2dF, _, _) ← coreFuse g4d2d
    if !g4d2dF.nodes.any (fun nd => nd.op == .elemBinary .add) then
      throw "4D Conv + 2D [1, 16] Add was erroneously folded into Conv bias!"

    -- (c) 4D Conv + [1, 16, 1, 1] DOES fold; 2D Conv [1, 16] + [16] DOES fold
    let g4dOk : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", x4Ty)]
      nodes := #[
        { name := "w", op := .const wPay, inputs := #[], outputs := #[(1, wTy)] },
        { name := "b", op := .const bPay, inputs := #[], outputs := #[(2, { shape := #[1, 16, 1, 1], dtype := .f32 })] },
        { name := "c", op := .conv 1 1 1 1 #[0, 0, 0, 0] 1 false .none, inputs := #[0, 1], outputs := #[(3, x4Ty)] },
        { name := "a", op := .elemBinary .add, inputs := #[3, 2], outputs := #[(4, x4Ty)] }
      ]
      outputs := #[4]
    }
    let (g4dOkF, _, _) ← coreFuse g4dOk
    if g4dOkF.nodes.any (fun nd => nd.op == .elemBinary .add) then
      throw "4D Conv + [1, 16, 1, 1] Add should have folded into Conv bias"

    let x2Ty : TensorType := { shape := #[1, 16], dtype := .fp16 }
    let g2dOk : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", x2Ty)]
      nodes := #[
        { name := "w", op := .const wPay, inputs := #[], outputs := #[(1, wTy)] },
        { name := "b", op := .const bPay, inputs := #[], outputs := #[(2, { shape := #[16], dtype := .f32 })] },
        { name := "c", op := .conv 1 1 1 1 #[0, 0, 0, 0] 1 false .none, inputs := #[0, 1], outputs := #[(3, x2Ty)] },
        { name := "a", op := .elemBinary .add, inputs := #[3, 2], outputs := #[(4, x2Ty)] }
      ]
      outputs := #[4]
    }
    let (g2dOkF, _, _) ← coreFuse g2dOk
    if g2dOkF.nodes.any (fun nd => nd.op == .elemBinary .add) then
      throw "2D Conv [1, 16] + [16] Add should have folded into Conv bias"
  ))

  -- Blocker 4: LayerNormalization with omitted Mean output ["Y", "", "InvStdDev"]
  results := results.push (checkOk "c1 reg: LayerNormalization [\"Y\", \"\", \"InvStdDev\"] binds slot 2 to InvStdDev (not Mean)" (do
    let gLn := pbGraph "ln_skip_mean"
      #[pbVI "x" #[1, 8, 4, 32]]
      #[pbVI "out" #[1, 8, 4, 1]]
      #[pbF32Tensor "sc" #[32] (Array.replicate 32 1.0)]
      #[pbNode "LayerNormalization" "ln" #["x", "sc"] #["y", "", "inv_std"] #[pbAttrInt "axis" (-1), pbAttrFloat "epsilon" 1.0e-5],
        pbNode "Relu" "out_op" #["inv_std"] #["out"]]
    for tgt in #[Target.h16, Target.h17] do
      let co ← compileOnnxBytes (pbWrapModel 18 gLn) { target := tgt }
      let s ← verifyCompiledOnnx co
      -- Check that the producer feeding out_op in coreGraph is invStd (lut rsqrt), NOT mean (reduce mean)
      let some outRelu := co.lowered.coreGraph.nodes.find? (fun nd => nd.name == "out_op")
        | throw "missing out_op in coreGraph"
      let some (_, prodNd, _) := co.lowered.coreGraph.producerOf? outRelu.inputs[0]!
        | throw "missing producer of inv_std"
      match prodNd.op with
      | .lut (.lut (.rsqrt _)) => pure ()
      | other => throw s!"expected InvStdDev to be produced by lut rsqrt, got {other.pretty}"
      if s.total != co.compiled.taskCount then
        throw s!"unexpected task count mismatch {repr s}"
  ))

  -- Blocker 5: Rank-increasing scalar-constant elemBinary layout desync rejected
  results := results.push (checkErr "c1 reg neg: rank-increasing scalar-constant elemBinary layout desync rejected" (do
    let g : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", { shape := #[1, 8], dtype := .fp16 })]
      nodes := #[
        { name := "c", op := .const (.f32 #[1.0]), inputs := #[], outputs := #[(1, { shape := #[1, 1, 1, 1], dtype := .f32 })] },
        { name := "add", op := .elemBinary .add, inputs := #[0, 1], outputs := #[(2, { shape := #[1, 1, 1, 8], dtype := .fp16 })] }
      ]
      outputs := #[2]
    }
    Vane.IR.ToGraph.coreGraphToGraphSpec g
  ) "changes physical layout")

  -- Additional cleanups & coverage:
  -- (a) DequantizeLinear fallback on non-quant-capable Conv (stride-2, depthwise) + Conv+Gelu unfusion
  results := results.push (checkOk "c1 reg: DequantizeLinear fallback on stride-2 & depthwise Conv3x3 + Conv+Gelu LUT unfusion" (do
    let w3Dense := ByteArray.mk (Array.ofFn (n := 16 * 16 * 9) fun i => ((i.val * 3 + 11) % 251).toUInt8)
    let w3Dw := ByteArray.mk (Array.ofFn (n := 16 * 1 * 9) fun i => ((i.val * 5 + 7) % 251).toUInt8)
    let sc16 := Array.ofFn (n := 16) fun i => 0.0625 * Float.ofNat ((i.val % 3) + 1)
    let w1 := ByteArray.mk (Array.ofFn (n := 16 * 16) fun i => ((i.val * 7 + 5) % 251).toUInt8)

    -- Stride-2 dense 3x3 falls back to FP16
    let gS2 := pbGraph "dq_s2_fallback" #[pbVI "x" #[1, 16, 11, 11]] #[pbVI "y" #[1, 16, 6, 6]]
      #[pbByteTensor "wq" onnxInt8 #[16, 16, 3, 3] w3Dense, pbF32Tensor "sc" #[16] sc16]
      #[pbNode "DequantizeLinear" "dq" #["wq", "sc"] #["w"] #[pbAttrInt "axis" 0],
        pbNode "Conv" "conv" #["x", "w"] #["y"] #[pbAttrInts "strides" #[2, 2], pbAttrInts "pads" #[1, 1, 1, 1]]]
    let coS2 ← compileOnnxBytes (pbWrapModel 18 gS2) { target := .h17 }
    let sS2 ← verifyCompiledOnnx coS2
    if coS2.lowered.summary.quantizedConvCount != 0 || sS2.neConv != 1 then
      throw s!"stride-2 DequantizeLinear expected FP16 fallback (qCnt=0, neConv=1), got qCnt={coS2.lowered.summary.quantizedConvCount} s={repr sS2}"

    -- Depthwise 3x3 falls back to FP16
    let gDw := pbGraph "dq_dw_fallback" #[pbVI "x" #[1, 16, 8, 8]] #[pbVI "y" #[1, 16, 8, 8]]
      #[pbByteTensor "wq" onnxInt8 #[16, 1, 3, 3] w3Dw, pbF32Tensor "sc" #[16] sc16]
      #[pbNode "DequantizeLinear" "dq" #["wq", "sc"] #["w"] #[pbAttrInt "axis" 0],
        pbNode "Conv" "conv" #["x", "w"] #["y"] #[pbAttrInt "group" 16, pbAttrInts "pads" #[1, 1, 1, 1]]]
    let coDw ← compileOnnxBytes (pbWrapModel 18 gDw) { target := .h17 }
    let sDw ← verifyCompiledOnnx coDw
    if coDw.lowered.summary.quantizedConvCount != 0 || sDw.neConv != 1 then
      throw s!"depthwise DequantizeLinear expected FP16 fallback (qCnt=0, neConv=1), got qCnt={coDw.lowered.summary.quantizedConvCount} s={repr sDw}"

    -- Quantized Conv1x1 + Gelu keeps native quantized conv and unfuses Gelu onto separate neAct task
    let gGelu := pbGraph "dq_conv_gelu" #[pbVI "x" #[1, 16, 8, 8]] #[pbVI "y" #[1, 16, 8, 8]]
      #[pbByteTensor "wq" onnxInt8 #[16, 16, 1, 1] w1, pbF32Tensor "sc" #[16] sc16]
      #[pbNode "DequantizeLinear" "dq" #["wq", "sc"] #["w"] #[pbAttrInt "axis" 0],
        pbNode "Conv" "conv" #["x", "w"] #["h"],
        pbNode "Gelu" "g" #["h"] #["y"]]
    let coGelu ← compileOnnxBytes (pbWrapModel 20 gGelu) { target := .h17 }
    let sGelu ← verifyCompiledOnnx coGelu
    if coGelu.lowered.summary.quantizedConvCount != 1 || sGelu.neConv != 1 || sGelu.neAct != 1 then
      throw s!"DequantizeLinear Conv+Gelu expected qCnt=1, neConv=1, neAct=1, got qCnt={coGelu.lowered.summary.quantizedConvCount} s={repr sGelu}"
  ))

  -- (b) Multi-consumer non-fusion in HighFuse & CoreFuse + --quantize PTQ with Conv + BatchNorm
  results := results.push (checkOk "c1 reg: multi-consumer Conv non-fusion in HighFuse/CoreFuse + PTQ on Conv+BatchNorm" (do
    let w1 := mkDetF32 (16 * 16) 0.05
    let gamma := Array.replicate 16 1.2
    let beta := Array.replicate 16 0.1
    let mean := Array.replicate 16 0.05
    let var := Array.replicate 16 1.0

    -- Multi-consumer Conv feeding both BatchNorm and Relu -> neither may fuse into Conv
    let gMc := pbGraph "mc_no_fuse" #[pbVI "x" #[1, 16, 8, 8]] #[pbVI "y" #[1, 16, 8, 8]]
      #[pbF32Tensor "w" #[16, 16, 1, 1] w1,
        pbF32Tensor "g" #[16] gamma, pbF32Tensor "b" #[16] beta,
        pbF32Tensor "m" #[16] mean, pbF32Tensor "v" #[16] var]
      #[pbNode "Conv" "conv" #["x", "w"] #["h"],
        pbNode "BatchNormalization" "bn" #["h", "g", "b", "m", "v"] #["h_bn"],
        pbNode "Relu" "rl" #["h"] #["h_rl"],
        pbNode "Add" "sum" #["h_bn", "h_rl"] #["y"]]
    let coMc ← compileOnnxBytes (pbWrapModel 18 gMc) { target := .h17 }
    let _ ← verifyCompiledOnnx coMc
    if coMc.lowered.summary.fusedBnCount != 0 || coMc.lowered.summary.fusedActCount != 0 then
      throw s!"multi-consumer Conv must not fuse BN or Relu, got bn={coMc.lowered.summary.fusedBnCount} act={coMc.lowered.summary.fusedActCount}"

    -- Conv + BatchNorm with --quantize int4 folds BN then quantizes folded weights
    let gBnQ := pbGraph "conv_bn_ptq" #[pbVI "x" #[1, 16, 8, 8]] #[pbVI "y" #[1, 16, 8, 8]]
      #[pbF32Tensor "w" #[16, 16, 1, 1] w1,
        pbF32Tensor "g" #[16] gamma, pbF32Tensor "b" #[16] beta,
        pbF32Tensor "m" #[16] mean, pbF32Tensor "v" #[16] var]
      #[pbNode "Conv" "conv" #["x", "w"] #["h"],
        pbNode "BatchNormalization" "bn" #["h", "g", "b", "m", "v"] #["y"]]
    for tgt in #[Target.h16, Target.h17] do
      let coBnQ ← compileOnnxBytes (pbWrapModel 18 gBnQ) { target := tgt, quantize := some .int4 }
      let sBnQ ← verifyCompiledOnnx coBnQ
      if coBnQ.lowered.summary.fusedBnCount != 1 || coBnQ.lowered.summary.quantizedConvCount != 1 || sBnQ.neConv != 1 then
        throw s!"{tgt.name} Conv+BN PTQ expected fusedBn=1, qCnt=1, neConv=1, got bn={coBnQ.lowered.summary.fusedBnCount} q={coBnQ.lowered.summary.quantizedConvCount} s={repr sBnQ}"
  ))

  -- (c) Cleanups: CoreFuse phase 4b skips overflowing MatMul scale; rmsNorm bad scale; ToGraph rank-3 conv & bad slice len; ruleTransposeId rank mismatch
  results := results.push (checkOk "c1 reg: CoreFuse skips overflowing MatMul scale & continues; ViewSimplify preserves rank-mismatched Transpose" (do
    let mTy : TensorType := { shape := #[2, 4, 4], dtype := .fp16 }
    let g2Mm : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "a", mTy), (1, "b", mTy)]
      nodes := #[
        { name := "mm1", op := .matmul false false f16PosOne, inputs := #[0, 1], outputs := #[(2, mTy)] },
        { name := "c_ovf", op := .const (.f32 #[1.0e10]), inputs := #[], outputs := #[(3, { shape := #[1], dtype := .f32 })] },
        { name := "mul1", op := .elemBinary .mul, inputs := #[2, 3], outputs := #[(4, mTy)] },
        { name := "mm2", op := .matmul false false f16PosOne, inputs := #[0, 1], outputs := #[(5, mTy)] },
        { name := "c_ok", op := .const (.f32 #[0.25]), inputs := #[], outputs := #[(6, { shape := #[1], dtype := .f32 })] },
        { name := "mul2", op := .elemBinary .mul, inputs := #[5, 6], outputs := #[(7, mTy)] },
        { name := "sum", op := .elemBinary .add, inputs := #[4, 7], outputs := #[(8, mTy)] }
      ]
      outputs := #[8]
    }
    let (_, _, mmCnt) ← coreFuse g2Mm
    if mmCnt != 1 then
      throw s!"expected second MatMul to fold after skipping overflowing first MatMul scale, got mmCnt={mmCnt}"

    -- ruleTransposeId must NOT fire when perm rank != input rank
    let gBadRank : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", { shape := #[1, 8, 4, 4], dtype := .fp16 })]
      nodes := #[
        { name := "tr_bad", op := .transpose #[0, 1], inputs := #[0], outputs := #[(1, { shape := #[1, 8, 4, 4], dtype := .fp16 })] }
      ]
      outputs := #[1]
    }
    if (ruleTransposeId.apply gBadRank).isSome then
      throw "ruleTransposeId must not fire when perm.size != input rank"
  ))

  results := results.push (checkErr "c1 reg neg: HighOp.rmsNorm rejects incompatible constant Scale shape" (do
    let ty : TensorType := { shape := #[1, 8, 4, 32], dtype := .fp16 }
    let gHigh : SSAGraph HighOp := {
      target := .h17
      inputs := #[(0, "x", ty)]
      nodes := #[
        { name := "bad_gamma", op := .core (.const (.f32 (Array.replicate 16 1.0))), inputs := #[],
          outputs := #[(1, { shape := #[16], dtype := .fp16 })] },
        { name := "rms", op := .rmsNorm (-1) 1.0e-5 true, inputs := #[0, 1], outputs := #[(2, ty)] }
      ]
      outputs := #[2]
    }
    runStandardPipeline gHigh {}
  ) "RMSNorm Scale shape")

  results := results.push (checkErr "c1 reg neg: ToGraph rejects rank-3 input on CoreOp.conv" (do
    let g : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", { shape := #[4, 4, 16], dtype := .fp16 })]
      nodes := #[
        { name := "w", op := .const (.f32 (mkDetF32 (16 * 16) 0.05)), inputs := #[],
          outputs := #[(1, { shape := #[16, 16, 1, 1], dtype := .f32 })] },
        { name := "c", op := .conv 1 1 1 1 #[0, 0, 0, 0] 1 false .none, inputs := #[0, 1],
          outputs := #[(2, { shape := #[4, 4, 16], dtype := .fp16 })] }
      ]
      outputs := #[2]
    }
    Vane.IR.ToGraph.coreGraphToGraphSpec g
  ) "conv requires 4D")

  results := results.push (checkErr "c1 reg neg: ToGraph rejects CoreOp.slice with mismatched param length" (do
    let g : SSAGraph CoreOp := {
      target := .h17
      inputs := #[(0, "x", { shape := #[1, 8, 4, 4], dtype := .fp16 })]
      nodes := #[
        { name := "sl", op := .slice #[0, 0] #[1, 4] #[1, 1], inputs := #[0],
          outputs := #[(1, { shape := #[1, 4, 4, 4], dtype := .fp16 })] }
      ]
      outputs := #[1]
    }
    Vane.IR.ToGraph.coreGraphToGraphSpec g
  ) "slice params length mismatch")

  -- PR #15 review blocker: rank-increasing [1,1,1,1] Mul/Div after 3D MatMul([1,1,16], [1,16,1]) -> [1,1,1]
  -- must preserve logical output shape [1,1,1,1] with both fusion ON and OFF, while same-rank scalar Mul still folds.
  results := results.push (checkOk "c1 reg: 3D MatMul + [1,1,1,1] Mul/Div preserves [1,1,1,1] output (fusion ON & OFF); same-rank scalar Mul folds" (do
    for opName in #["Mul", "Div"] do
      let gBcast := pbGraph s!"mm3d_bcast4d_{opName}"
        #[pbVI "a" #[1, 1, 16], pbVI "b" #[1, 16, 1]]
        #[pbVI "y" #[1, 1, 1, 1]]
        #[pbF32Tensor "sc4d" #[1, 1, 1, 1] #[0.25]]
        #[pbNode "MatMul" "mm" #["a", "b"] #["h"],
          pbNode opName "sc_op" #["h", "sc4d"] #["y"]]
      let modelBytes := pbWrapModel 18 gBcast
      for tgt in #[Target.h16, Target.h17] do
        for noFuse in #[false, true] do
          let co ← compileOnnxBytes modelBytes { target := tgt, noFuse := noFuse }
          let s ← verifyCompiledOnnx co
          if co.lowered.summary.outputShape != { n := 1, c := 1, h := 1, w := 1 } then
            throw s!"{tgt.name} {opName} (noFuse={noFuse}) expected outputShape [1,1,1,1], got {repr co.lowered.summary.outputShape}"
          if co.lowered.summary.fusedMatmulCount != 0 then
            throw s!"{tgt.name} {opName} (noFuse={noFuse}) must not fold rank-changing [1,1,1,1] broadcast into 3D MatMul, got fusedMatmulCount={co.lowered.summary.fusedMatmulCount}"
          if s.neMatmul != 1 || s.peOp != 1 || s.total != 2 then
            throw s!"{tgt.name} {opName} (noFuse={noFuse}) expected 1 neMatmul + 1 peOp task, got {repr s}"

    -- Same-rank scalar Mul (constant shape [] or [1] -> logical [1,1,1]) MUST still fold when fusion is ON
    for scDims in #[(#[] : Array Int), #[1]] do
      let gSame := pbGraph "mm3d_same_rank_mul"
        #[pbVI "a" #[1, 1, 16], pbVI "b" #[1, 16, 1]]
        #[pbVI "y" #[1, 1, 1]]
        #[pbF32Tensor "sc" scDims #[0.25]]
        #[pbNode "MatMul" "mm" #["a", "b"] #["h"],
          pbNode "Mul" "sc_op" #["h", "sc"] #["y"]]
      let modelBytes := pbWrapModel 18 gSame
      for tgt in #[Target.h16, Target.h17] do
        let coOn ← compileOnnxBytes modelBytes { target := tgt, noFuse := false }
        let sOn ← verifyCompiledOnnx coOn
        if coOn.lowered.summary.fusedMatmulCount != 1 || sOn.neMatmul != 1 || sOn.total != 1 then
          throw s!"{tgt.name} same-rank scalar Mul (dims={scDims.toList}) failed to fold: fused={coOn.lowered.summary.fusedMatmulCount} s={repr sOn}"
        let coOff ← compileOnnxBytes modelBytes { target := tgt, noFuse := true }
        let sOff ← verifyCompiledOnnx coOff
        if coOff.lowered.summary.fusedMatmulCount != 0 || sOff.neMatmul != 1 || sOff.peOp != 1 then
          throw s!"{tgt.name} same-rank scalar Mul (noFuse=true) unexpected: fused={coOff.lowered.summary.fusedMatmulCount} s={repr sOff}"
  ))

  pure results

end C1Tests

-- ============================================================================
-- Milestone C2: AneIR, Rolling L2 Cascades, MachineIR, Selective Spill & Multi-I/O
-- ============================================================================

namespace C2Tests

open Vane.IR
open Vane.Ane
open Vane.Machine
open Vane.Backend.LayoutSelect
open Vane.Backend.ChannelSplit
open Vane.Backend.Cascade
open Vane.Backend.MemoryPlan
open Vane.Backend.SyncAndEmit

private def compileOnnxBytes (bytes : ByteArray) (opts : LowerOptions := {}) : Except String CompiledOnnx := do
  compileOnnx (← parseOnnx bytes) opts

private def mkFp16Weights (n : Nat) (val : UInt16 := 0x3400) : ByteArray :=
  Vane.Graph.u16ArrayToBytes (Array.replicate n val)

private def runC2Tests : IO (Array TestResult) := do
  let mut results : Array TestResult := #[]

  -- 1. Multi-output ONNX models (homogeneous & heterogeneous shapes) packed into single ABI v3 HWX output surface (H16 & H17)
  results := results.push (checkOk "c2 multi-output onnx: 1-in 2-out (homogeneous & heterogeneous) pack into single ABI v3 output surface (H16 & H17)" (do
    let w1 := Array.replicate (16 * 16 * 1 * 1) 0.125
    let w2 := Array.replicate (8 * 16 * 3 * 3) 0.0625
    -- 1a. Homogeneous (H=8, W=8): [1,16,8,8] + [1,8,8,8] -> packed [1,24,8,8]
    let gHomo := pbGraph "multi_out_onnx_homo"
      #[pbVI "x" #[1, 16, 8, 8]]
      #[pbVI "y0" #[1, 16, 8, 8], pbVI "y1" #[1, 8, 8, 8]]
      #[pbF32Tensor "w1" #[16, 16, 1, 1] w1, pbF32Tensor "w2" #[8, 16, 3, 3] w2]
      #[pbNode "Conv" "conv1" #["x", "w1"] #["c1"],
        pbNode "Relu" "relu0" #["c1"] #["y0"],
        pbNode "Conv" "conv2" #["x", "w2"] #["y1"] #[pbAttrInts "pads" #[1, 1, 1, 1]]]
    let mHomo := pbWrapModel 17 gHomo
    -- 1b. Heterogeneous (H,W): [1,16,8,8] + [1,8,3,3] (stride-2 valid conv3x3) -> packed [1,cPack,hPack,32]
    let gHetero := pbGraph "multi_out_onnx_hetero"
      #[pbVI "x" #[1, 16, 8, 8]]
      #[pbVI "y0" #[1, 16, 8, 8], pbVI "y1" #[1, 8, 3, 3]]
      #[pbF32Tensor "w1" #[16, 16, 1, 1] w1, pbF32Tensor "w2" #[8, 16, 3, 3] w2]
      #[pbNode "Conv" "conv1" #["x", "w1"] #["c1"],
        pbNode "Relu" "relu0" #["c1"] #["y0"],
        pbNode "Conv" "conv2" #["x", "w2"] #["y1"] #[pbAttrInts "pads" #[0, 0, 0, 0], pbAttrInts "strides" #[2, 2]]]
    let mHetero := pbWrapModel 17 gHetero
    for (tag, mBytes) in #[("homo", mHomo), ("hetero", mHetero)] do
      for tgt in #[Target.h16, Target.h17] do
        let co ← compileOnnxBytes mBytes { target := tgt }
        if co.lowered.summary.outputs.size != 2 then
          throw s!"{tag}/{tgt.name}: expected 2 lowered outputs, got {co.lowered.summary.outputs.size}"
        if co.machineProgram.outputs.size != 2 then
          throw s!"{tag}/{tgt.name}: expected 2 logical OutputSlots, got {co.machineProgram.outputs.size}"
        if co.machineProgram.outputSurfaces.size != 1 then
          throw s!"{tag}/{tgt.name}: expected 1 ABI v3 HWX output surface, got {co.machineProgram.outputSurfaces.size}"
        let outSurfs := co.compiled.info.surfaces.filter fun s => s.direction == .output
        if outSurfs.size != 1 || co.compiled.info.surfaces.size != 2 then
          throw s!"{tag}/{tgt.name}: expected 1 HWX output surface & 2 total surfaces, got out={outSurfs.size} total={co.compiled.info.surfaces.size}"
        let sl0 := co.machineProgram.outputs[0]!
        let sl1 := co.machineProgram.outputs[1]!
        if sl0.byteOffset != 0 || sl1.byteOffset < sl0.byteSize || sl1.byteOffset % 64 != 0 ||
            sl1.byteOffset + sl1.byteSize > outSurfs[0]!.span.toNat then
          throw s!"{tag}/{tgt.name}: invalid packed output layout {repr sl0} / {repr sl1} vs span {outSurfs[0]!.span}"
        let summary ← verifyHwxTasksSummary co.compiled.hwx co.compiled.info (requireKnownTask := true)
        if summary.total != co.compiled.taskCount then
          throw s!"{tag}/{tgt.name}: verified task count mismatch"
        validateSectionsWithInfo co.compiled.info co.compiled.sections
  ))

  -- 2. Multi-output GraphSpec: 2-in 2-out + 3-in 3-out KV-cache-style block (4 ABI v3 surfaces!) + negative 4-input rejection
  results := results.push (checkOk "c2 multi-output graph: 2-in 2-out & 3-in 3-out KV-cache GraphSpecs verify on H16 & H17" (do
    let w1 := mkFp16Weights (16 * 16)
    let w8 := mkFp16Weights (8 * 8)
    for tgt in #[Target.h16, Target.h17] do
      -- 2a. 2-in 2-out (3 HWX surfaces: 2 inputs + 1 packed output)
      let spec2 : Vane.Graph.GraphSpec := {
        target := tgt
        inputs := #[
          { name := "a", shape := { c := 16, h := 8, w := 8 } },
          { name := "b", shape := { c := 16, h := 8, w := 8 } }
        ]
        nodes := #[
          { id := "c1", inputs := #["a"], op := .conv1x1 16 16 w1 none .relu },
          { id := "sum", inputs := #["c1", "b"], op := .pe2 .add false },
          { id := "out2", inputs := #["sum"], op := .lut (.lut .gelu) }
        ]
        outputs := #["c1", "out2"]
      }
      let g2 ← Vane.Graph.compileGraph spec2
      if g2.info.surfaces.size != 3 then
        throw s!"{tgt.name}: expected 3 ABI v3 surfaces for 2-in 2-out, got {g2.info.surfaces.size}"
      let outSurfs2 := g2.info.surfaces.filter fun s => s.direction == .output
      let some mp2 := g2.machineProgram | throw s!"{tgt.name}: missing MachineProgram"
      if outSurfs2.size != 1 || mp2.outputs.size != 2 then
        throw s!"{tgt.name}: expected 1 HWX output surface & 2 logical outputs"
      let _ ← verifyHwxTasksSummary g2.hwx g2.info (requireKnownTask := true)
      validateSectionsWithInfo g2.info g2.sections

      -- 2b. 3-in 3-out KV-cache block (x, k_in, v_in) -> (y, k_out, v_out) = 4 ABI v3 surfaces!
      let specKV : Vane.Graph.GraphSpec := {
        target := tgt
        inputs := #[
          { name := "x", shape := { c := 8, h := 4, w := 8 } },
          { name := "k_in", shape := { c := 8, h := 4, w := 8 } },
          { name := "v_in", shape := { c := 8, h := 4, w := 8 } }
        ]
        nodes := #[
          { id := "q", inputs := #["x"], op := .conv1x1 8 8 w8 none .relu },
          { id := "k_out", inputs := #["q", "k_in"], op := .pe2 .add false },
          { id := "v_out", inputs := #["q", "v_in"], op := .pe2 .add false },
          { id := "y", inputs := #["k_out", "v_out"], op := .pe2 .add false }
        ]
        outputs := #["y", "k_out", "v_out"]
      }
      let gKV ← Vane.Graph.compileGraph specKV
      if gKV.info.surfaces.size != 4 then
        throw s!"{tgt.name}: expected 4 ABI v3 surfaces for 3-in 3-out KV block, got {gKV.info.surfaces.size}"
      let outSurfsKV := gKV.info.surfaces.filter fun s => s.direction == .output
      let some mpKV := gKV.machineProgram | throw s!"{tgt.name}: missing MachineProgram"
      if outSurfsKV.size != 1 || mpKV.outputs.size != 3 then
        throw s!"{tgt.name}: expected 1 HWX output surface & 3 logical outputs for KV block"
      let _ ← verifyHwxTasksSummary gKV.hwx gKV.info (requireKnownTask := true)
      validateSectionsWithInfo gKV.info gKV.sections
  ))

  results := results.push (checkErr "c2 multi-output neg: 4 inputs (>3 ABI v3 input limit) rejected" (do
    let spec : Vane.Graph.GraphSpec := {
      target := .h17
      inputs := #[
        { name := "a", shape := { c := 8, h := 4, w := 8 } },
        { name := "b", shape := { c := 8, h := 4, w := 8 } },
        { name := "c", shape := { c := 8, h := 4, w := 8 } },
        { name := "d", shape := { c := 8, h := 4, w := 8 } }
      ]
      nodes := #[
        { id := "s1", inputs := #["a", "b"], op := .pe2 .add false },
        { id := "s2", inputs := #["c", "d"], op := .pe2 .add false },
        { id := "out", inputs := #["s1", "s2"], op := .pe2 .add false }
      ]
    }
    Vane.Graph.compileGraph spec
  ) "1..3 inputs")

  -- 3. Multi-output duplicate output inserts explicit pack_out copy tasks into single ABI v3 output surface
  results := results.push (checkOk "c2 multi-output duplicate: packs aliased outputs via 2 pack_out stores into single output surface" (do
    let w1 := mkFp16Weights (8 * 8)
    let spec : Vane.Graph.GraphSpec := {
      target := .h17
      inputs := #[{ name := "x", shape := { c := 8, h := 4, w := 8 } }]
      nodes := #[{ id := "c1", inputs := #["x"], op := .conv1x1 8 8 w1 none .relu }]
      outputs := #["c1", "c1"]
    }
    let g ← Vane.Graph.compileGraph spec
    let some mpDup := g.machineProgram | throw "missing MachineProgram"
    if g.taskCount != 3 || g.info.surfaces.size != 2 || mpDup.outputs.size != 2 then
      throw s!"expected 3 tasks (conv + 2 pack_out), 2 surfaces, 2 logical outputs; got tasks={g.taskCount} surfs={g.info.surfaces.size} outs={mpDup.outputs.size}"
    let _ ← verifyHwxTasksSummary g.hwx g.info (requireKnownTask := true)
    validateSectionsWithInfo g.info g.sections
  ))

  -- 4. Vela-style rolling L2 cascade on a deep 10-layer 3x3 stack exceeding 2 MiB unstripped (H16 & H17)
  results := results.push (checkOk "c2 rolling cascade: 10-layer 3x3 stack (>2 MiB unstripped) achieves 0 B scratch on H16 & H17" (do
    let c := 64; let h := 96; let w := 128
    let w3 := mkFp16Weights (c * c * 9)
    let mut nodes : Array Vane.Graph.GraphNode := Array.mkEmpty 10
    for i in [:10] do
      let inp := if i == 0 then "x" else s!"conv{i - 1}"
      nodes := nodes.push {
        id := s!"conv{i}"
        inputs := #[inp]
        op := .conv3x3 .dense c c 1 .same w3 none .relu
      }
    for tgt in #[Target.h16, Target.h17] do
      let unstripped ← Vane.Graph.compileGraph {
        target := tgt
        inputs := #[{ name := "x", shape := { c := c, h := h, w := w } }]
        nodes := nodes
        defaultPlacement := .auto
        stripMode := .off
      }
      if unstripped.scratchBytes == 0 then
        throw s!"{tgt.name}: expected unstripped 10-layer 3x3 stack to exceed 2 MiB and spill to scratch"
      let cascaded ← Vane.Graph.compileGraph {
        target := tgt
        inputs := #[{ name := "x", shape := { c := c, h := h, w := w } }]
        nodes := nodes
        defaultPlacement := .auto
        stripMode := .auto
      }
      if cascaded.stripCount <= 1 then
        throw s!"{tgt.name}: expected stripCount > 1, got {cascaded.stripCount}"
      if cascaded.scratchBytes != 0 then
        throw s!"{tgt.name}: expected cascaded scratchBytes == 0, got {cascaded.scratchBytes}"
      if cascaded.l2PeakBytes > l2CapacityBytes then
        throw s!"{tgt.name}: cascaded l2PeakBytes {cascaded.l2PeakBytes} > 2 MiB"
      if cascaded.kernelBytes != unstripped.kernelBytes then
        throw s!"{tgt.name}: expected shared __KERN_0 weights ({unstripped.kernelBytes}B), got {cascaded.kernelBytes}B"
      let _ ← verifyHwxTasksSummary cascaded.hwx cascaded.info (requireKnownTask := true)
      validateSectionsWithInfo cascaded.info cascaded.sections
  ))

  -- 5. Zero-recompute rolling halo-stitch cascade via `cascadePass` (0 B scratch on halo=0 interior) & direct 3x3->LUT->3x3 (L2 c0!)
  results := results.push (checkOk "c2 rolling halo-stitch: cascadePass achieves 0 B scratch & micro-DMA halo reuse keeps c0 in L2 (H16 & H17)" (do
    let w3Big := mkFp16Weights (64 * 64 * 9)
    let w1Big := mkFp16Weights (64 * 64)
    let w3Small := mkFp16Weights (16 * 16 * 9)
    for tgt in #[Target.h16, Target.h17] do
      -- 5a. Reachable through `cascadePass { stripMode := .auto, allowRollingHaloStitch := true }`:
      -- conv3x3 -> lut(silu) -> conv1x1 (>2 MiB unstripped) achieves 0 B scratch!
      let specAuto : Vane.Graph.GraphSpec := {
        target := tgt
        inputs := #[{ name := "x", shape := { c := 64, h := 64, w := 128 } }]
        nodes := #[
          { id := "c0", inputs := #["x"], op := .conv3x3 .dense 64 64 1 .same w3Big none .none },
          { id := "act1", inputs := #["c0"], op := .lut (.lut .silu) },
          { id := "c2", inputs := #["act1"], op := .conv1x1 64 64 w1Big none .relu }
        ]
      }
      let aneAuto ← Vane.Graph.graphSpecToAne specAuto
      let rolledAuto ← cascadePass aneAuto { stripMode := .auto, allowRollingHaloStitch := true }
      if rolledAuto.stripCount <= 1 then
        throw s!"{tgt.name}: expected cascadePass(allowRollingHaloStitch=true) to strip >1 strips"
      let mpAuto ← planMemory rolledAuto
      if mpAuto.scratchBytes != 0 then
        throw s!"{tgt.name}: expected 0 B scratch on rolling halo-stitch conv3x3->lut->conv1x1, got {mpAuto.scratchBytes}B"
      if !mpAuto.witness.l2WithinCapacity then
        throw s!"{tgt.name}: rolling halo auto witness exceeded L2 capacity"
      let cmAuto ← syncAndEmit mpAuto
      let _ ← verifyHwxTasksSummary cmAuto.hwx cmAuto.info (requireKnownTask := true)
      validateSectionsWithInfo cmAuto.info cmAuto.sections

      -- 5b. Interior 3x3 consumer: c0 stays L2-resident (no forced DRAM!), zero halo DMA on act1, micro-DMA halo stitch on c2
      let spec3 : Vane.Graph.GraphSpec := {
        target := tgt
        inputs := #[{ name := "x", shape := { c := 16, h := 32, w := 32 } }]
        nodes := #[
          { id := "c0", inputs := #["x"], op := .conv3x3 .dense 16 16 1 .same w3Small none .none },
          { id := "act1", inputs := #["c0"], op := .lut (.lut .silu) },
          { id := "c2", inputs := #["act1"], op := .conv3x3 .dense 16 16 1 .same w3Small none .relu }
        ]
      }
      let ane3 ← Vane.Graph.graphSpecToAne spec3
      let rolled ← unrollRollingHaloCascade ane3 8
      let act1HaloCount := rolled.graph.nodes.foldl (fun acc nd =>
        if (nd.name.splitOn "act1").length > 1 && (nd.name.splitOn "halo").length > 1 then acc + 1 else acc) 0
      let c2HaloCount := rolled.graph.nodes.foldl (fun acc nd =>
        if (nd.name.splitOn "c2").length > 1 && (nd.name.splitOn "halo").length > 1 then acc + 1 else acc) 0
      if act1HaloCount != 0 then
        throw s!"{tgt.name}: expected 0 halo DMA tasks for halo=0 LUT layer, got {act1HaloCount}"
      if c2HaloCount == 0 then
        throw s!"{tgt.name}: expected halo DMA stitching tasks for second 3x3 conv layer"
      let mp ← planMemory rolled
      -- Verify c0 strips are placed in L2 (not forced to DRAM!)
      let c0DramCount := mp.tasks.foldl (fun acc mt =>
        if (mt.name.splitOn "c0_s").length > 1 then
          match mp.buffers[mt.output]! with
          | .l2 .. => acc
          | _ => acc + 1
        else acc) 0
      if c0DramCount != 0 then
        throw s!"{tgt.name}: expected all c0 rolling strips to remain L2-resident, found {c0DramCount} in DRAM"
      if !mp.witness.l2WithinCapacity then
        throw s!"{tgt.name}: rolling halo MemoryPlanWitness exceeded L2 capacity"
      let cm ← syncAndEmit mp
      let _ ← verifyHwxTasksSummary cm.hwx cm.info (requireKnownTask := true)
      validateSectionsWithInfo cm.info cm.sections
  ))

  -- 6. W/sequence stripping on H=1, W=seq_len block (0 B scratch, verified HWX on H16 & H17)
  results := results.push (checkOk "c2 sequence W-strip: H=1 W=seq_len block cascades with 0 B scratch on H16 & H17" (do
    let c := 64; let wFull := 128
    let w1 := mkFp16Weights (c * c)
    for tgt in #[Target.h16, Target.h17] do
      let spec : Vane.Graph.GraphSpec := {
        target := tgt
        inputs := #[{ name := "x", shape := { c := c, h := 1, w := wFull } }]
        nodes := #[
          { id := "fc1", inputs := #["x"], op := .conv1x1 c c w1 none .relu },
          { id := "fc2", inputs := #["fc1"], op := .conv1x1 c c w1 none .none }
        ]
      }
      let ane ← Vane.Graph.graphSpecToAne spec
      let wStripped ← unrollWStripCascade ane 64
      if wStripped.stripCount != 2 || wStripped.stripAxis != 3 then
        throw s!"{tgt.name}: expected 2 W-strips on axis 3, got count={wStripped.stripCount} axis={wStripped.stripAxis}"
      let mp ← planMemory wStripped
      if mp.scratchBytes != 0 then
        throw s!"{tgt.name}: expected 0 B scratch for W-stripped sequence cascade, got {mp.scratchBytes}B"
      if !mp.witness.l2WithinCapacity then
        throw s!"{tgt.name}: W-strip witness exceeded 2 MiB L2"
      let cm ← syncAndEmit mp
      let _ ← verifyHwxTasksSummary cm.hwx cm.info (requireKnownTask := true)
      validateSectionsWithInfo cm.info cm.sections
  ))

  -- 7. Selective L2<->DRAM BAR 3 spilling on U-Net-style long-skip DAG (>2 MiB live) on H16 & H17
  results := results.push (checkOk "c2 selective spill: U-Net long skip spills to BAR 3 while hot bottleneck stays in L2 (H16 & H17)" (do
    -- enc: 48x128x128 (~1.5 MiB L2) spans 4 bottleneck convs (16x128x128 ~512 KiB each) before residual add.
    -- Peak live = 1.5 + 0.5 + 0.5 = 2.5 MiB > 2 MiB -> selective spill must spill `enc` (highest bytes*lifetime)
    -- to BAR 3 DRAM scratch while keeping hot short-lived bottleneck activations in L2!
    let h := 128; let w := 128
    let wEnc := mkFp16Weights (48 * 16)
    let wDown := mkFp16Weights (16 * 48)
    let wMid := mkFp16Weights (16 * 16)
    let wUp := mkFp16Weights (48 * 16)
    for tgt in #[Target.h16, Target.h17] do
      let spec : Vane.Graph.GraphSpec := {
        target := tgt
        inputs := #[{ name := "x", shape := { c := 16, h := h, w := w } }]
        nodes := #[
          { id := "enc", inputs := #["x"], op := .conv1x1 16 48 wEnc none .relu },
          { id := "b0", inputs := #["enc"], op := .conv1x1 48 16 wDown none .relu },
          { id := "b1", inputs := #["b0"], op := .conv1x1 16 16 wMid none .relu },
          { id := "b2", inputs := #["b1"], op := .conv1x1 16 16 wMid none .relu },
          { id := "up", inputs := #["b2"], op := .conv1x1 16 48 wUp none .none },
          { id := "out", inputs := #["enc", "up"], op := .pe2 .add false }
        ]
        defaultPlacement := .auto
        stripMode := .off
      }
      let g ← Vane.Graph.compileGraph spec
      if g.scratchBytes == 0 then
        throw s!"{tgt.name}: expected selective spill scratchBytes > 0"
      if g.l2PeakBytes > l2CapacityBytes then
        throw s!"{tgt.name}: selective spill l2PeakBytes {g.l2PeakBytes} > 2 MiB"
      let some mp := g.machineProgram
        | throw s!"{tgt.name}: missing MachineProgram"
      if !mp.witness.l2WithinCapacity then
        throw s!"{tgt.name}: MemoryPlanWitness not within L2 capacity"
      let encNode? := g.nodes.find? fun n => n.id == "enc"
      let b0Node? := g.nodes.find? fun n => n.id == "b0"
      let b1Node? := g.nodes.find? fun n => n.id == "b1"
      match encNode?, b0Node?, b1Node? with
      | some encN, some b0N, some b1N =>
        if encN.outLocation.placementName != "dram" then
          throw s!"{tgt.name}: expected long-skip 'enc' to spill to dram, got {encN.outLocation.placementName}"
        if b0N.outLocation.placementName != "l2" || b1N.outLocation.placementName != "l2" then
          throw s!"{tgt.name}: expected hot bottleneck 'b0'/'b1' to stay in l2, got {b0N.outLocation.placementName}/{b1N.outLocation.placementName}"
      | _, _, _ => throw s!"{tgt.name}: missing expected nodes in compiled U-Net graph"
      let _ ← verifyHwxTasksSummary g.hwx g.info (requireKnownTask := true)
      validateSectionsWithInfo g.info g.sections
  ))

  -- 8. Zero-copy subview aliasing (contiguous C-axis slice & dual conv1x1 concat(axis=1))
  results := results.push (checkOk "c2 zero-copy subview: contiguous C-slice and dual conv1x1 concat emit zero copy tasks" (do
    let w8 := mkFp16Weights (8 * 8)
    -- 8a. Contiguous C-axis slice [0..1, 8..16, 0..8, 0..16] -> conv1x1 => 1 task (slice aliased!)
    let specSlice : Vane.Graph.GraphSpec := {
      target := .h17
      inputs := #[{ name := "x", shape := { c := 16, h := 8, w := 16 } }]
      nodes := #[
        { id := "sl", inputs := #["x"], op := .slice #[0, 8, 0, 0] #[1, 16, 8, 16] #[1, 1, 1, 1] },
        { id := "c1", inputs := #["sl"], op := .conv1x1 8 8 w8 none .relu }
      ]
    }
    let gSl ← Vane.Graph.compileGraph specSlice
    if gSl.taskCount != 1 then
      throw s!"expected zero-copy aliased slice to emit 1 task, got {gSl.taskCount}"
    let _ ← verifyHwxTasksSummary gSl.hwx gSl.info (requireKnownTask := true)

    -- 8b. Dual conv1x1 -> concat(axis=1) => 2 tasks (concat zero-copy aliased into output surface!)
    let specCat : Vane.Graph.GraphSpec := {
      target := .h17
      inputs := #[{ name := "x", shape := { c := 8, h := 8, w := 16 } }]
      nodes := #[
        { id := "b0", inputs := #["x"], op := .conv1x1 8 8 w8 none .relu },
        { id := "b1", inputs := #["x"], op := .conv1x1 8 8 w8 none .relu },
        { id := "cat", inputs := #["b0", "b1"], op := .concat 1 }
      ]
    }
    let gCat ← Vane.Graph.compileGraph specCat
    if gCat.taskCount != 2 then
      throw s!"expected zero-copy aliased concat to emit 2 tasks, got {gCat.taskCount}"
    let _ ← verifyHwxTasksSummary gCat.hwx gCat.info (requireKnownTask := true)
    validateSectionsWithInfo gCat.info gCat.sections
  ))

  -- 9. Buffer donation validation (positive 2-task chain + negative shape mismatch / intra-task / post-write / duplicate)
  results := results.push (checkOk "c2 buffer donation: valid 2-task chain donation recorded with lr < fw and sync barrier" (do
    let w8 := mkFp16Weights (8 * 8)
    let spec : Vane.Graph.GraphSpec := {
      target := .h17
      inputs := #[{ name := "x", shape := { c := 8, h := 4, w := 8 } }]
      nodes := #[
        { id := "h", inputs := #["x"], op := .conv1x1 8 8 w8 none .relu },
        { id := "y", inputs := #["h"], op := .conv1x1 8 8 w8 none .relu }
      ]
      donations := #[(0, 0)]
    }
    let g ← Vane.Graph.compileGraph spec
    let some mp := g.machineProgram | throw "missing MachineProgram"
    if mp.donations.size != 1 then
      throw s!"expected 1 recorded buffer donation, got {mp.donations.size}"
    let d := mp.donations[0]!
    if d.lastReadTask != 0 || d.firstWriteTask != 1 then
      throw s!"expected donation lr=0 < fw=1, got lr={d.lastReadTask} fw={d.firstWriteTask}"
    let _ ← verifyHwxTasksSummary g.hwx g.info (requireKnownTask := true)
  ))

  results := results.push (checkErr "c2 buffer donation neg: mismatched input/output shapes rejected" (do
    let w8 := mkFp16Weights (8 * 8)
    let w16 := mkFp16Weights (16 * 8)
    let spec : Vane.Graph.GraphSpec := {
      target := .h17
      inputs := #[{ name := "x", shape := { c := 8, h := 4, w := 8 } }]
      nodes := #[
        { id := "h", inputs := #["x"], op := .conv1x1 8 8 w8 none .relu },
        { id := "y", inputs := #["h"], op := .conv1x1 8 16 w16 none .relu }
      ]
      donations := #[(0, 0)]
    }
    Vane.Graph.compileGraph spec
  ) "shape mismatch")

  results := results.push (checkErr "c2 buffer donation neg: single-task intra-task donation (lr == fw) rejected" (do
    let w8 := mkFp16Weights (8 * 8)
    let spec : Vane.Graph.GraphSpec := {
      target := .h17
      inputs := #[{ name := "x", shape := { c := 8, h := 4, w := 8 } }]
      nodes := #[{ id := "y", inputs := #["x"], op := .conv1x1 8 8 w8 none .relu }]
      donations := #[(0, 0)]
    }
    Vane.Graph.compileGraph spec
  ) "not strictly before")

  results := results.push (checkErr "c2 buffer donation neg: post-write input read (lr > fw) rejected" (do
    let w8 := mkFp16Weights (8 * 16)
    let spec : Vane.Graph.GraphSpec := {
      target := .h17
      inputs := #[{ name := "x", shape := { c := 16, h := 4, w := 8 } }]
      nodes := #[
        { id := "b0", inputs := #["x"], op := .conv1x1 16 8 w8 none .relu },
        { id := "b1", inputs := #["x"], op := .conv1x1 16 8 w8 none .relu },
        { id := "y", inputs := #["b0", "b1"], op := .concat 1 }
      ]
      donations := #[(0, 0)]
    }
    Vane.Graph.compileGraph spec
  ) "not strictly before")

  results := results.push (checkErr "c2 buffer donation neg: duplicate output donation index rejected" (do
    let w8 := mkFp16Weights (8 * 8)
    let spec : Vane.Graph.GraphSpec := {
      target := .h17
      inputs := #[
        { name := "x", shape := { c := 8, h := 4, w := 8 } },
        { name := "z", shape := { c := 8, h := 4, w := 8 } }
      ]
      nodes := #[
        { id := "h", inputs := #["x", "z"], op := .pe2 .add false },
        { id := "y", inputs := #["h"], op := .conv1x1 8 8 w8 none .relu }
      ]
      donations := #[(0, 0), (1, 0)]
    }
    Vane.Graph.compileGraph spec
  ) "duplicate buffer donation")

  results := results.push (checkErr "c2 reg R3-B3: buffer donation on packed multi-output rejected" (do
    let w8 := mkFp16Weights (8 * 8)
    let spec : Vane.Graph.GraphSpec := {
      target := .h17
      inputs := #[{ name := "x", shape := { c := 8, h := 4, w := 8 } }]
      nodes := #[
        { id := "h", inputs := #["x"], op := .conv1x1 8 8 w8 none .relu },
        { id := "y0", inputs := #["h"], op := .conv1x1 8 8 w8 none .relu },
        { id := "y1", inputs := #["h"], op := .conv1x1 8 8 w8 none .none }
      ]
      outputs := #["y0", "y1"]
      donations := #[(0, 0)]
    }
    Vane.Graph.compileGraph spec
  ) "unsupported on packed multi-output")

  -- 10. SOK ChannelSplit pass on wide-Cout fp16 1x1, fp16 3x3 & quantized int8 3x3 convs (H16 & H17)
  results := results.push (checkOk "c2 channelSplit (SOK): wide-Cout fp16 1x1, fp16 3x3 & int8 3x3 convs split cleanly on H16 & H17" (do
    let wWide1x1 := mkFp16Weights (1024 * 128)
    let wWide3x3 := mkFp16Weights (256 * 256 * 9)
    let q3x3 : Vane.Quant.QuantWeightSpec := {
      format := .int8
      codes := ByteArray.mk (Array.replicate (512 * 256 * 9) (1 : UInt8))
      scales := Array.replicate 512 (0.125 : Float)
      zeroPoints := none
    }
    for tgt in #[Target.h16, Target.h17] do
      let g1 ← Vane.Graph.compileGraph {
        target := tgt
        inputs := #[{ name := "x", shape := { c := 128, h := 4, w := 8 } }]
        nodes := #[{ id := "wide1x1", inputs := #["x"], op := .conv1x1 128 1024 wWide1x1 none .relu }]
      }
      if g1.taskCount < 2 then
        throw s!"{tgt.name}: expected SOK split on 128->1024 conv1x1 (>1 task), got {g1.taskCount}"
      let _ ← verifyHwxTasksSummary g1.hwx g1.info (requireKnownTask := true)

      let g3 ← Vane.Graph.compileGraph {
        target := tgt
        inputs := #[{ name := "x", shape := { c := 256, h := 8, w := 8 } }]
        nodes := #[{ id := "wide3x3", inputs := #["x"], op := .conv3x3 .dense 256 256 1 .same wWide3x3 none .relu }]
      }
      if g3.taskCount < 2 then
        throw s!"{tgt.name}: expected SOK split on 256->256 fp16 conv3x3 (>1 task), got {g3.taskCount}"
      let _ ← verifyHwxTasksSummary g3.hwx g3.info (requireKnownTask := true)

      let gQ3 ← Vane.Graph.compileGraph {
        target := tgt
        inputs := #[{ name := "x", shape := { c := 256, h := 8, w := 8 } }]
        nodes := #[{ id := "wideQ3x3", inputs := #["x"], op := .conv3x3 .dense 256 512 1 .same ByteArray.empty none .relu (some q3x3) }]
      }
      if gQ3.taskCount < 2 then
        throw s!"{tgt.name}: expected SOK split on 256->512 int8 conv3x3 (>1 task), got {gQ3.taskCount}"
      let _ ← verifyHwxTasksSummary gQ3.hwx gQ3.info (requireKnownTask := true)
  ))

  -- 11. Textual IR dumps (`--dump-ir=ane` and `--dump-ir=machine`)
  results := results.push (checkOk "c2 dump-ir: AneProgram.dumpAne & MachineProgram.pretty format cleanly" (do
    let w8 := mkFp16Weights (8 * 8)
    let g ← Vane.Graph.compileGraph {
      target := .h17
      inputs := #[{ name := "x", shape := { c := 8, h := 4, w := 8 } }]
      nodes := #[
        { id := "c1", inputs := #["x"], op := .conv1x1 8 8 w8 none .relu },
        { id := "y", inputs := #["c1", "x"], op := .pe2 .add false }
      ]
    }
    let some ane := g.aneProgram | throw "missing AneProgram"
    let some mp := g.machineProgram | throw "missing MachineProgram"
    let aneStr := ane.dumpAne
    let mpStr := mp.pretty
    if (aneStr.splitOn "// AneIR:").length <= 1 || (aneStr.splitOn "ane.ne_conv").length <= 1 then
      throw s!"unexpected AneIR dump:\n{aneStr}"
    if (mpStr.splitOn "machine_program").length <= 1 || (mpStr.splitOn "l2_peak=").length <= 1 then
      throw s!"unexpected MachineIR dump:\n{mpStr}"
  ))

  -- 12. Blocker #1 regression: SOK-split node 0 + separate node 1 with distinct weights must NOT alias in __KERN_0
  results := results.push (checkOk "c2 reg B1: SOK-split wide conv1x1 + second conv1x1 with distinct weights never collide in __KERN_0 (H16 & H17)" (do
    let w0 := mkFp16Weights (1024 * 128) (val := 0x3800)
    let w1 := mkFp16Weights (256 * 128) (val := 0x3C00)
    for tgt in #[Target.h16, Target.h17] do
      let gSingle ← Vane.Graph.compileGraph {
        target := tgt
        inputs := #[{ name := "x", shape := { c := 128, h := 4, w := 8 } }]
        nodes := #[{ id := "n0", inputs := #["x"], op := .conv1x1 128 1024 w0 none .relu }]
      }
      let gBoth ← Vane.Graph.compileGraph {
        target := tgt
        inputs := #[{ name := "x", shape := { c := 128, h := 4, w := 8 } }]
        nodes := #[
          { id := "n0", inputs := #["x"], op := .conv1x1 128 1024 w0 none .relu },
          { id := "n1", inputs := #["x"], op := .conv1x1 128 256 w1 none .relu }
        ]
        outputs := #["n0", "n1"]
      }
      if gBoth.kernelBytes <= gSingle.kernelBytes then
        throw s!"{tgt.name}: __KERN_0 weight collision detected! combined={gBoth.kernelBytes}B <= single={gSingle.kernelBytes}B"
      let _ ← verifyHwxTasksSummary gBoth.hwx gBoth.info (requireKnownTask := true)
      validateSectionsWithInfo gBoth.info gBoth.sections
  ))

  -- 13. Blocker #4 regression: dmaView with BOTH outBase > 0 and cp.outOff > 0 adds offsets cleanly
  results := results.push (checkOk "c2 reg B4: dmaView writing into packed output slot (outBase > 0) with cp.outOff > 0 verifies on H16 & H17" (do
    let w8 := mkFp16Weights (8 * 8)
    for tgt in #[Target.h16, Target.h17] do
      -- y0: [1, 8, 4, 64] at offset 0; y1 = concat(axis=3, [a, b]): [1, 8, 4, 64] at outBase = 4096,
      -- where concat slice 1 has cp.outOff = 64 > 0 -> effOutOff = 4096 + 64 = 4160!
      let spec : Vane.Graph.GraphSpec := {
        target := tgt
        inputs := #[
          { name := "x", shape := { c := 8, h := 4, w := 64 } },
          { name := "a", shape := { c := 8, h := 4, w := 32 } },
          { name := "b", shape := { c := 8, h := 4, w := 32 } }
        ]
        nodes := #[
          { id := "y0", inputs := #["x"], op := .conv1x1 8 8 w8 none .relu },
          { id := "y1", inputs := #["a", "b"], op := .concat 3 }
        ]
        outputs := #["y0", "y1"]
      }
      let g ← Vane.Graph.compileGraph spec
      let _ ← verifyHwxTasksSummary g.hwx g.info (requireKnownTask := true)
      validateSectionsWithInfo g.info g.sections
  ))

  -- 14. Blocker #5 regression: conv1x1 [1,8,8,9] -> pool(max, 2x2, stride=2, valid) avoids illegal L2 input placement
  results := results.push (checkOk "c2 reg B5: conv1x1 [1,8,8,9] -> pool(max,2x2,s=2,valid) spills to DRAM and verifies on H16 & H17" (do
    let w8 := mkFp16Weights (8 * 8)
    for tgt in #[Target.h16, Target.h17] do
      let spec : Vane.Graph.GraphSpec := {
        target := tgt
        inputs := #[{ name := "x", shape := { c := 8, h := 8, w := 9 } }]
        nodes := #[
          { id := "c0", inputs := #["x"], op := .conv1x1 8 8 w8 none .relu },
          { id := "p0", inputs := #["c0"], op := .pool .max 2 2 2 2 .valid false }
        ]
      }
      let g ← Vane.Graph.compileGraph spec
      let some c0N := g.nodes.find? fun n => n.id == "c0" | throw s!"{tgt.name}: missing c0"
      if c0N.outLocation.placementName != "dram" then
        throw s!"{tgt.name}: expected c0 feeding odd-width stride-2 valid pool to spill to dram, got {c0N.outLocation.placementName}"
      let _ ← verifyHwxTasksSummary g.hwx g.info (requireKnownTask := true)
      validateSectionsWithInfo g.info g.sections
  ))

  -- 15. Blocker #7 regressions: Conv weight kernel-dim mismatch & non-conv CoreOp.dequantize rejection
  results := results.push (checkErr "c2 reg B7a: CoreOp.conv mismatched weight kernel dims rejected" (do
    let wBad := mkFp16Weights (8 * 8)
    let xTy : Vane.IR.TensorType := { shape := #[1, 8, 4, 8], dtype := .fp16 }
    let wTy : Vane.IR.TensorType := { shape := #[8, 8, 1, 1], dtype := .fp16 }
    let cg : Vane.IR.SSAGraph Vane.IR.CoreOp := {
      target := .h17
      inputs := #[(0, "x", xTy)]
      nodes := #[
        { name := "w_1x1", op := .const (.fp16 wBad), inputs := #[], outputs := #[(1, wTy)] },
        { name := "conv3", op := .conv 3 3 1 1 #[1, 1, 1, 1] 1 false .none, inputs := #[0, 1], outputs := #[(2, xTy)] }
      ]
      outputs := #[2]
    }
    Vane.Backend.LayoutSelect.lowerCoreToAne cg
  ) "kernel dims")

  results := results.push (checkErr "c2 reg B7b: CoreOp.dequantize consumed by non-conv op rejected cleanly" (do
    let q : Vane.Quant.QuantWeightSpec := {
      format := .int8
      codes := ByteArray.mk (Array.replicate (8 * 8) (1 : UInt8))
      scales := Array.replicate 8 (0.125 : Float)
      zeroPoints := none
    }
    let wTy : Vane.IR.TensorType := { shape := #[8, 8, 1, 1], dtype := .fp16 }
    let cg : Vane.IR.SSAGraph Vane.IR.CoreOp := {
      target := .h17
      inputs := #[(0, "x", { shape := #[1, 8, 4, 8], dtype := .fp16 })]
      nodes := #[
        { name := "q_w", op := .const (.quant q), inputs := #[], outputs := #[(1, wTy)] },
        { name := "deq_w", op := .dequantize .int8, inputs := #[1], outputs := #[(2, wTy)] },
        { name := "bad_act", op := .elemUnary .abs, inputs := #[2], outputs := #[(3, wTy)] }
      ]
      outputs := #[3]
    }
    Vane.Backend.LayoutSelect.lowerCoreToAne cg
  ) "can only be consumed as a Conv weight input")

  -- 16. Blocker #8 regression: emitGraphDir rejects multi-output ONNX explicitly
  results := results.push (← do
    let name := "c2 reg B8: emitGraphDir rejects multi-output ONNX cleanly"
    let w1 := Array.replicate (8 * 8) 0.125
    let g := pbGraph "multi_out_emit_dir"
      #[pbVI "x" #[1, 8, 4, 8]]
      #[pbVI "y0" #[1, 8, 4, 8], pbVI "y1" #[1, 8, 4, 8]]
      #[pbF32Tensor "w1" #[8, 8, 1, 1] w1]
      #[pbNode "Conv" "c1" #["x", "w1"] #["y0"], pbNode "Relu" "r1" #["x"] #["y1"]]
    match (do
      let m ← parseOnnx (pbWrapModel 17 g)
      lowerOnnx m { target := .h17 }) with
    | .error e => pure { name := name, passed := false, detail := s!"lowerOnnx failed: {e}" }
    | .ok lr =>
      match ← emitGraphDir (System.FilePath.mk "/tmp/vane_test_b8_unused") lr with
      | .ok () => pure { name := name, passed := false, detail := "expected emitGraphDir error on multi-output" }
      | .error e =>
        if (e.splitOn "does not yet support multi-output").length > 1 then
          pure { name := name, passed := true, detail := e }
        else
          pure { name := name, passed := false, detail := s!"unexpected error: {e}" })

  -- 17. Minor #4 regression: GraphSpec DCE prunes dead trailing nodes when explicit `outputs` are given
  results := results.push (checkOk "c2 reg M4: GraphSpec DCE prunes unused trailing nodes with explicit outputs" (do
    let w8 := mkFp16Weights (8 * 8)
    let spec : Vane.Graph.GraphSpec := {
      target := .h17
      inputs := #[{ name := "x", shape := { c := 8, h := 4, w := 8 } }]
      nodes := #[
        { id := "y0", inputs := #["x"], op := .conv1x1 8 8 w8 none .relu },
        { id := "dead_branch", inputs := #["x"], op := .conv1x1 8 8 w8 none .relu }
      ]
      outputs := #["y0"]
    }
    let g ← Vane.Graph.compileGraph spec
    if g.taskCount != 1 then
      throw s!"expected DCE to prune dead_branch leaving 1 task, got {g.taskCount}"
    let _ ← verifyHwxTasksSummary g.hwx g.info (requireKnownTask := true)
  ))

  -- 18. Round 3 Blocker #1: H-stripped packed multi-output (non-direct conv3x3 + conv1x1 AND downstream-consumed y0 -> y1 + dup outputs)
  results := results.push (checkOk "c2 reg R3-B1: H-stripped packed multi-output (conv3x3+conv1x1 & downstream y0->y1) packs per-strip with 0 B scratch (H16 & H17)" (do
    let w3 := mkFp16Weights (64 * 64 * 9)
    let w1 := mkFp16Weights (64 * 64)
    for tgt in #[Target.h16, Target.h17] do
      -- 18a. Non-direct conv3x3 + direct conv1x1 on 64x128x128 (>2 MiB unstripped -> forced H-strip!)
      let g1 ← Vane.Graph.compileGraph {
        target := tgt
        inputs := #[{ name := "x", shape := { c := 64, h := 128, w := 128 } }]
        nodes := #[
          { id := "b0", inputs := #["x"], op := .conv3x3 .dense 64 64 1 .same w3 none .relu },
          { id := "y0", inputs := #["b0"], op := .conv3x3 .dense 64 64 1 .same w3 none .relu },
          { id := "b1", inputs := #["x"], op := .conv1x1 64 64 w1 none .relu },
          { id := "y1", inputs := #["b1"], op := .conv1x1 64 64 w1 none .relu }
        ]
        outputs := #["y0", "y1"]
        stripMode := .auto
      }
      if g1.stripCount <= 1 then
        throw s!"{tgt.name} (18a): expected stripCount > 1, got {g1.stripCount}"
      if g1.scratchBytes != 0 then
        throw s!"{tgt.name} (18a): expected 0 B scratch with per-strip pack_out, got {g1.scratchBytes}B"
      let _ ← verifyHwxTasksSummary g1.hwx g1.info (requireKnownTask := true)
      validateSectionsWithInfo g1.info g1.sections

      -- 18b. Output 0 also consumed downstream by output 1 (y0 = conv1x1(h0), y1 = relu(y0))
      let g2 ← Vane.Graph.compileGraph {
        target := tgt
        inputs := #[{ name := "x", shape := { c := 64, h := 128, w := 128 } }]
        nodes := #[
          { id := "h0", inputs := #["x"], op := .conv1x1 64 64 w1 none .relu },
          { id := "y0", inputs := #["h0"], op := .conv1x1 64 64 w1 none .none },
          { id := "y1", inputs := #["y0"], op := .lut .relu }
        ]
        outputs := #["y0", "y1"]
        stripMode := .auto
      }
      if g2.stripCount <= 1 then
        throw s!"{tgt.name} (18b): expected stripCount > 1, got {g2.stripCount}"
      if g2.scratchBytes != 0 then
        throw s!"{tgt.name} (18b): expected 0 B scratch for downstream-consumed y0->y1, got {g2.scratchBytes}B"
      let _ ← verifyHwxTasksSummary g2.hwx g2.info (requireKnownTask := true)
      validateSectionsWithInfo g2.info g2.sections

    -- 18c. Duplicate outputs under stripMode=.auto cleanly skips H-stripping without corrupting nodeOutIdx
    let gDup ← Vane.Graph.compileGraph {
      target := .h17
      inputs := #[{ name := "x", shape := { c := 16, h := 16, w := 16 } }]
      nodes := #[{ id := "y0", inputs := #["x"], op := .conv1x1 16 16 (mkFp16Weights (16 * 16)) none .relu }]
      outputs := #["y0", "y0"]
      stripMode := .auto
    }
    let _ ← verifyHwxTasksSummary gDup.hwx gDup.info (requireKnownTask := true)
  ))

  -- 19. Round 3 Blocker #2: Multi-head (n = 4) packed multi-output preserves all 4 batches & exact byteSize/batchStride
  results := results.push (checkOk "c2 reg R3-B2: multi-head (n=4) 2-output neMatmul preserves full 4-head byteSize/batchStride (H16 & H17)" (do
    let aSh : Vane.View.Shape4D := { n := 1, c := 4, h := 8, w := 16 }
    let bSh : Vane.View.Shape4D := { n := 4, c := 16, h := 1, w := 16 }
    let outSh : Vane.View.Shape4D := { n := 4, c := 8, h := 1, w := 16 }
    let aTy := Vane.Ane.shape4DToTensorType aSh
    let bTy := Vane.Ane.shape4DToTensorType bSh
    let outTy := Vane.Ane.shape4DToTensorType outSh
    for tgt in #[Target.h16, Target.h17] do
      let aneProg : Vane.Ane.AneProgram := {
        graph := {
          target := tgt
          inputs := #[(0, "a", aTy), (1, "b", bTy)]
          nodes := #[
            { name := "mm0", op := .neMatmul 4 8 16 16 false false (0x3C00 : UInt16) 0 8, inputs := #[0, 1], outputs := #[(2, outTy)] },
            { name := "mm1", op := .neMatmul 4 8 16 16 false false (0x3C00 : UInt16) 0 8, inputs := #[0, 1], outputs := #[(3, outTy)] }
          ]
          outputs := #[2, 3]
        }
        outputNames := #["y0", "y1"]
      }
      let mp ← planMemory aneProg
      if mp.outputs.size != 2 then
        throw s!"{tgt.name}: expected 2 OutputSlots, got {mp.outputs.size}"
      let expBSt := 8 * 64
      let expSz := 4 * expBSt
      let sl0 := mp.outputs[0]!
      let sl1 := mp.outputs[1]!
      if sl0.shape.n != 4 || sl0.batchStride != expBSt || sl0.byteSize != expSz || sl0.byteOffset != 0 then
        throw s!"{tgt.name}: bad output[0] multi-head slot {repr sl0}"
      if sl1.shape.n != 4 || sl1.batchStride != expBSt || sl1.byteSize != expSz || sl1.byteOffset != expSz then
        throw s!"{tgt.name}: bad output[1] multi-head slot {repr sl1}"
      let cm ← syncAndEmit mp
      let _ ← verifyHwxTasksSummary cm.hwx cm.info (requireKnownTask := true)
      validateSectionsWithInfo cm.info cm.sections
  ))

  -- 20. Round 3 Minor #1: All-passthrough multi-output graph (nodes = #[], outputs = #["x", "y"]) compiles via 2 pack_out copies
  results := results.push (checkOk "c2 reg R3-M1: all-passthrough multi-output (outputs=[x, y]) compiles and verifies on H16 & H17" (do
    for tgt in #[Target.h16, Target.h17] do
      let spec : Vane.Graph.GraphSpec := {
        target := tgt
        inputs := #[
          { name := "x", shape := { c := 8, h := 4, w := 8 } },
          { name := "y", shape := { c := 8, h := 4, w := 8 } }
        ]
        nodes := #[]
        outputs := #["x", "y"]
      }
      let g ← Vane.Graph.compileGraph spec
      if g.taskCount != 2 || g.info.surfaces.size != 3 then
        throw s!"{tgt.name}: expected 2 pack_out tasks & 3 ABI v3 surfaces, got tasks={g.taskCount} surfs={g.info.surfaces.size}"
      let _ ← verifyHwxTasksSummary g.hwx g.info (requireKnownTask := true)
      validateSectionsWithInfo g.info g.sections
  ))

  -- 21. C3 soundness helpers & witnesses: SOK balancedSlices, Cascade strip/halo, packOutputSlots + computeHwxOutShape, cleanupRedundantHandoffs
  results := results.push (checkOk "c3 witnesses: SOK slices, Cascade strips/halo, multi-output packing, and handoff cleanup invariants" (do
    -- 1. SOK balancedSlices conservation & cap
    for (cout, bestDc) in #[(64, 64), (128, 48), (255, 32), (513, 64)] do
      let k := Vane.Tiling.ceilDiv cout bestDc
      let sl := Vane.Backend.ChannelSplit.balancedSlices cout k
      let total := sl.foldl (· + ·) 0
      if total != cout then
        throw s!"balancedSlices({cout}, {k}) sum={total} != {cout}"
      if sl.any (fun dc => dc == 0 || dc > bestDc) then
        throw s!"balancedSlices({cout}, {k}) out of bounds (0, {bestDc}]: {repr sl}"
    -- 2. Cascade stripOutBand + haloInBand exact 3x3 height invariant
    for (hFull, step) in #[(16, 4), (17, 6), (64, 10), (33, 7)] do
      let nStrips := Vane.Tiling.ceilDiv hFull step
      let mut prevHi := 0
      for s in [0:nStrips] do
        let (lo, hi) := Vane.Backend.Cascade.stripOutBand hFull step s
        if lo != prevHi || lo >= hi || hi > hFull then
          throw s!"stripOutBand({hFull}, {step}, {s}) = ({lo}, {hi}), expected lo={prevHi}"
        let (inLo, inHi) := Vane.Backend.Cascade.haloInBand hFull 1 lo hi
        let pt := Vane.Backend.Cascade.stripPadTop lo
        let pb := Vane.Backend.Cascade.stripPadBot hi hFull
        if inLo > lo || hi > inHi || inHi > hFull || (inHi - inLo) + pt + pb != (hi - lo) + 2 then
          throw s!"haloInBand3x3({hFull}, {lo}, {hi}) failed exact height invariant"
        prevHi := hi
      if prevHi != hFull then
        throw s!"stripOutBand({hFull}, {step}) did not cover full H (end={prevHi})"
    -- 3. packOutputSlots + computeHwxOutShape alignment, disjointness & coverage
    let shapes : Array Vane.View.Shape4D := #[
      { n := 2, c := 16, h := 8, w := 16 },
      { n := 1, c := 32, h := 4, w := 32 },
      { n := 4, c := 8, h := 2, w := 8 }
    ]
    let (slots, totalBytes) := Vane.Backend.MemoryPlan.packOutputSlots shapes #["a", "b", "c"]
    let (hwxShape, _) ← Vane.Backend.MemoryPlan.computeHwxOutShape slots totalBytes
    if totalBytes % 64 != 0 || totalBytes > hwxShape.span then
      throw s!"packOutputSlots totalBytes={totalBytes} failed 64B align or hwxShape.span={hwxShape.span}"
    for i in [0:slots.size] do
      let sl := slots[i]!
      if sl.byteOffset % 64 != 0 || sl.byteSize % 64 != 0 || sl.byteOffset + sl.byteSize > hwxShape.span then
        throw s!"slot {i} alignment/bounds failed: {repr sl}"
      if i + 1 < slots.size && sl.byteOffset + sl.byteSize > slots[i + 1]!.byteOffset then
        throw s!"slots {i} and {i+1} overlap"
    -- 4. cleanupRedundantHandoffs preserves wait flags and clears dominated handoff flags
    let s0 : Vane.Verify.TaskSyncInfo := {
      src1WaitNe5 := true, src1WaitPe22 := true,
      src1HandoffNe := true, src1HandoffPe := true,
      src2WaitNe5 := true, src2HandoffNe := true
    }
    let s1 := Vane.Backend.SyncAndEmit.cleanupRedundantHandoffs s0
    if !s1.src1WaitNe5 || !s1.src1WaitPe22 || s1.src1HandoffNe || s1.src1HandoffPe || s1.src2HandoffNe then
      throw s!"cleanupRedundantHandoffs did not preserve waits / clear dominated handoffs: {repr s1}"
    let s2 := Vane.Backend.SyncAndEmit.cleanupRedundantHandoffs s1
    if s2.src1WaitNe5 != s1.src1WaitNe5 || s2.src1WaitPe22 != s1.src1WaitPe22 ||
       s2.src1HandoffNe != s1.src1HandoffNe || s2.src1HandoffPe != s1.src1HandoffPe ||
       s2.src2WaitNe5 != s1.src2WaitNe5 || s2.src2HandoffNe != s1.src2HandoffNe then
      throw s!"cleanupRedundantHandoffs not idempotent"
  ))

  pure results

end C2Tests

def main : IO UInt32 := do
  let results := (← runAllTests) ++ (← runM4Tests) ++ (← runM5aTests) ++ (← runM5bTests) ++ (← runM5cTests) ++ (← runM5dTests) ++ (← runM7bTests) ++ (← runM8aTests) ++ (← C1Tests.runC1Tests) ++ (← C2Tests.runC2Tests) ++ (← runTargetMatrixTests)
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
