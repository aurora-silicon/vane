import Vane

open Vane.Binary
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Tiling
open Vane.Conv1x1

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
  let _ ← verifyHwxTasks badHwx info (requireConvTask := true)
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

    let wOp ← encodeWord neOpLayout #[("biasInKernel", 1), ("relu", 1)]
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
    -- H17 ocg=5 at F=4, fat=0 covering Cout=1024
    let sh1024 : Conv1x1Shape := { h := 8, w := 16, cin := 16, cout := 1024 }
    let t1024 : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 5, shPref := 0, shMax := 3, tileHeight := 8 }
    validateTiling .h17 sh1024 t1024
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
    let sh : Conv1x1Shape := { h := 8, w := 16, cin := 8, cout := 1025 }
    let t : Tiling := { logNE := 4, f := 4, fat := 0, ocg := 5, shPref := 0, shMax := 3, tileHeight := 8 }
    validateTiling .h17 sh t
  ) "coverage")

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

    -- H17 2-round shapes (Cout > 256)
    let h17OnlyShapes : Array Conv1x1Shape := #[
      { h := 8, w := 16, cin := 64, cout := 300 },
      { h := 16, w := 32, cin := 256, cout := 512 },
      { h := 8, w := 16, cin := 32, cout := 1024 }
    ]
    for sh in h17OnlyShapes do
      let t17 ← chooseTiling .h17 sh
      validateTiling .h17 sh t17
      -- Verify H16 properly rejects Cout > 256
      match chooseTiling .h16 sh with
      | .ok _ => throw s!"expected H16 to reject Cout={sh.cout}"
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

  -- Verify M0 demo program still passes verifyHwxTasks when requireConvTask := false
  results := results.push (checkOk "m1: M0 demo passes non-forced task verification" (do
    let checked ← verifyHwxTasks baseLinearHwx baseLinearInfo (requireConvTask := false)
    if checked != 0 then throw s!"expected 0 conv tasks in M0 demo, got {checked}"
    pure ()
  ))

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
