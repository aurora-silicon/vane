import Vane.Binary

namespace Vane.TaskStream

open Vane.Binary

/-- Target ANE hardware family. -/
inductive Target where
  | h16
  | h17
  deriving Repr, BEq, DecidableEq

/-- Declarative hardware target profile capturing container rules, register constants,
    and execution/tiling capabilities for an ANE generation. -/
structure TargetProfile where
  -- Identity & metadata
  name : String
  chipFamily : Nat
  abiVersion : Nat
  cpuSubtype : UInt32
  -- Task stream & HWX container rules
  defaultPrologue : UInt32
  allowZeroPrologue : Bool
  requireTightEnd : Bool
  allowNoteVersion : Bool
  allowSlot5BarTable : Bool
  allowZeroScratchTextFlags : Bool
  allowEndSymbolValue : Bool
  allowStabSymbols : Bool
  fvmlibUseSpanSize : Bool
  -- Task header & register constants
  taskHeader4Base : UInt32
  taskHeader4EdgeBits : UInt32
  targetCfgValue : UInt32
  neCfgValue : UInt32
  resDepCode : UInt32
  -- Execution & tiling capabilities
  directModeUsesFat : Bool
  conv1x1CoutShiftBonus : Nat
  dense3x3Stride1MaxOcg : Nat
  dense3x3Stride1MultiRoundF4Ocg3 : Bool
  supportsWide3x3 : Bool
  replicateKernelChunkSlots : Bool
  peForbidK5LastStrip24 : Bool
  poolMultiStripMinW : Nat
  deriving Repr, BEq, DecidableEq

namespace TargetProfile

/-- Return `true` iff `prologue` is legal for this target (`1` is always legal; `0` when `allowZeroPrologue`). -/
def isValidPrologue (p : TargetProfile) (prologue : UInt32) : Bool :=
  prologue == 1 || (p.allowZeroPrologue && prologue == 0)

/-- Direct result mode width threshold `(if directModeUsesFat then 128 <<< fat else 128) >>> shMax`. -/
def directThreshold (p : TargetProfile) (fat shMax : Nat) : Nat :=
  (if p.directModeUsesFat then (128 : Nat) <<< fat else 128) >>> shMax

/-- Maximum legal output channels for dense 1x1 convolution at `(f, ocg)`. -/
def conv1x1MaxCout (p : TargetProfile) (f ocg : Nat) : Nat :=
  (1 : Nat) <<< (f + ocg + p.conv1x1CoutShiftBonus)

end TargetProfile

namespace Target

/-- Declarative hardware profile for each supported ANE target. -/
def profile : Target → TargetProfile
  | .h16 => {
      name := "h16"
      chipFamily := 16
      abiVersion := 3
      cpuSubtype := 7
      defaultPrologue := 1
      allowZeroPrologue := false
      requireTightEnd := true
      allowNoteVersion := false
      allowSlot5BarTable := false
      allowZeroScratchTextFlags := false
      allowEndSymbolValue := false
      allowStabSymbols := false
      fvmlibUseSpanSize := false
      taskHeader4Base := 0x00fff820
      taskHeader4EdgeBits := 0x00000040
      targetCfgValue := 0x00100000
      neCfgValue := 0x00000082
      resDepCode := 0x31
      directModeUsesFat := false
      conv1x1CoutShiftBonus := 0
      dense3x3Stride1MaxOcg := 4
      dense3x3Stride1MultiRoundF4Ocg3 := false
      supportsWide3x3 := false
      replicateKernelChunkSlots := true
      peForbidK5LastStrip24 := true
      poolMultiStripMinW := 64
    }
  | .h17 => {
      name := "h17"
      chipFamily := 17
      abiVersion := 3
      cpuSubtype := 9
      defaultPrologue := 0
      allowZeroPrologue := true
      requireTightEnd := false
      allowNoteVersion := true
      allowSlot5BarTable := true
      allowZeroScratchTextFlags := true
      allowEndSymbolValue := true
      allowStabSymbols := true
      fvmlibUseSpanSize := true
      taskHeader4Base := 0x00fff800
      taskHeader4EdgeBits := 0x00000060
      targetCfgValue := 0x00200000
      neCfgValue := 0x10000082
      resDepCode := 0x21
      directModeUsesFat := true
      conv1x1CoutShiftBonus := 1
      dense3x3Stride1MaxOcg := 3
      dense3x3Stride1MultiRoundF4Ocg3 := true
      supportsWide3x3 := true
      replicateKernelChunkSlots := false
      peForbidK5LastStrip24 := false
      poolMultiStripMinW := 32
    }

def cpuSubtype (t : Target) : UInt32 :=
  t.profile.cpuSubtype

def ofCpuSubtype? : UInt32 → Option Target
  | 7 => some .h16
  | 9 => some .h17
  | _ => none

def name (t : Target) : String :=
  t.profile.name

def parse? (s : String) : Option Target :=
  match s with
  | "h16" => some .h16
  | "h17" => some .h17
  | _ =>
    match s.toLower with
    | "h16" => some .h16
    | "h17" => some .h17
    | _ => none

/-- Every supported target, in a fixed order. Tests iterate this list, so a new
    constructor is exercised by CI once it is added here. -/
def all : List Target := [.h16, .h17]

theorem mem_all (t : Target) : t ∈ all := by
  cases t <;> simp [all]

end Target

/-- Byte size of the task-stream opening prologue frame. -/
def prologueFrameBytes : Nat := 16

/-- Byte alignment granularity for task starts and zero-block padding frames. -/
def taskFrameAlignBytes : Nat := 16

/-- Maximum byte length of `__TEXT,__text` (384 KiB). -/
def maxTextSize : Nat := 0x60000

/-- Allowed register byte-address half-open ranges `[start, end)`. -/
def allowedRegBlocks : Array (Nat × Nat) := #[
  (0x0000, 0x005c),
  (0x4100, 0x41a4),
  (0x44d0, 0x44d4),
  (0x4500, 0x453c),
  (0x4900, 0x4930),
  (0x4d00, 0x4e44),
  (0x5100, 0x5154),
  (0x5500, 0x5620),
  (0x5900, 0x5930)
]

/-- Return `true` iff register word index `w` falls inside one of the 9 allowed byte-address blocks. -/
def isAllowedRegWord (w : Nat) : Bool :=
  let byteAddr := w * 4
  allowedRegBlocks.any fun (lo, hi) => lo <= byteAddr && byteAddr < hi

/-- Population count of a 16-bit mask. -/
def popcount16 (mask : UInt16) : Nat :=
  let m := mask.toNat
  Id.run do
    let mut c : Nat := 0
    for i in [:16] do
      if ((m >>> i) &&& 1) == 1 then
        c := c + 1
    return c

/-- A register-write record inside a task body. -/
inductive RegRecord where
  /-- Sequential register record writing `values` starting at word index `base`,
      with optional BAR slot tag (bit 29 + 6-bit slot index). -/
  | sequential (base : UInt16) (barSlot : Option UInt8) (values : Array UInt32)
  /-- Bitmask register record writing `base` and `base + i + 1` for each set bit `i` in `mask`. -/
  | bitmask (base : UInt16) (mask : UInt16) (values : Array UInt32)
  deriving Repr, BEq, Inhabited

/-- One task within `__TEXT,__text`. -/
structure Task where
  /-- Task header words: length 9 (with `(header[8] &&& 3) != 3`) or
      length 10 (with `(header[8] &&& 3) == 3`).
      The encoder automatically fills bits `[26:16]` of `header[0]` with the total task word count. -/
  header : Array UInt32
  /-- Register records following the task header. -/
  records : Array RegRecord
  /-- Optional number of 16-byte zero blocks emitted immediately before this task. -/
  leadingZeroBlocks : Nat := 0
  deriving Repr, BEq, Inhabited

namespace Task

/-- Construct a task with a standard 9-word zeroed header (word 8 low 2 bits = 0). -/
def mk9 (records : Array RegRecord) (header : Array UInt32 := Array.replicate 9 0)
    (leadingZeroBlocks : Nat := 0) : Task :=
  { header := header, records := records, leadingZeroBlocks := leadingZeroBlocks }

/-- Construct a task with a 10-word header (word 8 low 2 bits forced to 3). -/
def mk10 (records : Array RegRecord) (header : Array UInt32 := (Array.replicate 10 0).set! 8 3)
    (leadingZeroBlocks : Nat := 0) : Task :=
  { header := header, records := records, leadingZeroBlocks := leadingZeroBlocks }

end Task

/-- High-level task stream representation for `__TEXT,__text`. -/
structure TaskStream where
  /-- Opening frame word 0 (`1` on H16; `0` or `1` on H17). -/
  prologue : UInt32 := 1
  /-- Non-empty sequence of tasks. -/
  tasks : Array Task
  /-- Optional 16-byte zero blocks after the final task (must be `0` on H16). -/
  trailingZeroBlocks : Nat := 0
  deriving Repr, BEq

/-- Encoded task stream together with structural metadata and payload word byte offsets. -/
structure EncodedTaskStream where
  bytes : ByteArray
  taskCount : Nat
  recordCount : Nat
  recordLimit : Nat
  /-- Byte offsets of every register-record payload word, grouped as `(taskIdx, recordIdx, payloadIdx, byteOffset)`. -/
  payloadSites : Array (Nat × Nat × Nat × Nat)

/-- Summary produced by `validateTaskStream`. -/
structure TaskStreamInfo where
  prologue : UInt32
  taskCount : Nat
  recordCount : Nat
  recordLimit : Nat
  /-- Distinct BAR slots referenced by sequential register records (bit 29 set), in order of first appearance. -/
  barSlots : Array UInt8
  /-- Sorted byte offsets in `__text` of all register-record payload words (valid relocation sites). -/
  payloadSites : Array Nat
  deriving Repr, BEq

/-- Check whether `offset` is in a sorted array of payload byte offsets. -/
def TaskStreamInfo.isPayloadSite (info : TaskStreamInfo) (offset : Nat) : Bool :=
  let arr := info.payloadSites
  if arr.isEmpty then false
  else
    Id.run do
      let mut lo := 0
      let mut hi := arr.size
      while lo < hi do
        let mid := (lo + hi) / 2
        let v := arr[mid]!
        if v == offset then
          return true
        else if v < offset then
          lo := mid + 1
        else
          hi := mid
      return false

/-- Encode a single register record into its header word and payload words, validating register blocks. -/
def encodeRegRecord (rec : RegRecord) : Except String (UInt32 × Array UInt32) := do
  match rec with
  | .sequential base barSlot values =>
    if base.toNat > 0x7fff then
      throw s!"sequential record base {toHex base.toNat} exceeds 15-bit range"
    if values.isEmpty || values.size > 64 then
      throw s!"sequential record value count {values.size} must be in 1..64"
    for i in [:values.size] do
      let reg := base.toNat + i
      if !isAllowedRegWord reg then
        throw s!"sequential record register word {toHex reg} (byte {toHex (reg * 4)}) outside allowed blocks"
    let barBits ← match barSlot with
      | none => pure (0 : UInt32)
      | some slot =>
        if slot.toNat > 63 then
          throw s!"sequential record BAR slot {slot} exceeds 0..63"
        pure (((1 : UInt32) <<< 29) ||| (slot.toUInt32 <<< 23))
    let countField : UInt32 := ((values.size - 1).toUInt32 &&& 0x3f) <<< 15
    let header : UInt32 := barBits ||| countField ||| (base.toUInt32 &&& 0x7fff)
    pure (header, values)
  | .bitmask base mask values =>
    if base.toNat > 0x7fff then
      throw s!"bitmask record base {toHex base.toNat} exceeds 15-bit range"
    let expectedCount := 1 + popcount16 mask
    if values.size != expectedCount then
      throw s!"bitmask record value count {values.size} != 1 + popcount({toHex mask.toNat}) = {expectedCount}"
    if !isAllowedRegWord base.toNat then
      throw s!"bitmask record base register word {toHex base.toNat} (byte {toHex (base.toNat * 4)}) outside allowed blocks"
    for i in [:16] do
      if ((mask.toNat >>> i) &&& 1) == 1 then
        let reg := base.toNat + i + 1
        if !isAllowedRegWord reg then
          throw s!"bitmask record register word {toHex reg} (byte {toHex (reg * 4)}) outside allowed blocks"
    let header : UInt32 := ((1 : UInt32) <<< 31) ||| (mask.toUInt32 <<< 15) ||| (base.toUInt32 &&& 0x7fff)
    pure (header, values)

/-- Encode a `TaskStream` into raw `__TEXT,__text` bytes. -/
def encodeTaskStream (target : Target) (stream : TaskStream) : Except String EncodedTaskStream := do
  let prof := target.profile
  if !prof.isValidPrologue stream.prologue then
    throw s!"{prof.name} task stream prologue {stream.prologue} is invalid"
  if prof.requireTightEnd && stream.trailingZeroBlocks != 0 then
    throw s!"{prof.name} task stream forbids trailing zero blocks after final task"

  if stream.tasks.isEmpty then
    throw "task stream must contain at least 1 task"

  -- 16-byte opening frame
  let mut out := pushZeros (pushU32LE ByteArray.empty stream.prologue) 12
  let mut totalRecords := 0
  let mut totalRecordLimit := 0
  let mut sites : Array (Nat × Nat × Nat × Nat) := #[]

  for ti in [:stream.tasks.size] do
    let task := stream.tasks[ti]!
    if task.leadingZeroBlocks > 0 then
      out := pushZeros out (task.leadingZeroBlocks * 16)

    if task.header.size != 9 && task.header.size != 10 then
      throw s!"task {ti}: header must have 9 or 10 words, got {task.header.size}"
    let w8 := task.header[8]!
    let is10 := (w8 &&& 3) == 3
    if task.header.size == 9 && is10 then
      throw s!"task {ti}: 9-word header must not have (word[8] & 3) == 3"
    if task.header.size == 10 && !is10 then
      throw s!"task {ti}: 10-word header must have (word[8] & 3) == 3"

    let mut encodedRecs : Array (UInt32 × Array UInt32) := #[]
    let mut recWords := 0
    for ri in [:task.records.size] do
      let enc ← encodeRegRecord task.records[ri]!
      encodedRecs := encodedRecs.push enc
      recWords := recWords + 1 + enc.2.size

    let totalWords := task.header.size + recWords
    if totalWords < 9 || totalWords > 0x7ff then
      throw s!"task {ti}: total word count {totalWords} out of range 9..{0x7ff}"

    -- Stamp word count into bits [26:16] of header[0]
    let w0Cleared := task.header[0]! &&& (~~~((0x7ff : UInt32) <<< 16))
    let w0 := w0Cleared ||| ((totalWords.toUInt32 &&& 0x7ff) <<< 16)
    out := pushU32LE out w0
    for hi in [1 : task.header.size] do
      out := pushU32LE out task.header[hi]!

    let recStart := out.size
    for ri in [:encodedRecs.size] do
      let (hdr, vals) := encodedRecs[ri]!
      out := pushU32LE out hdr
      for pi in [:vals.size] do
        let payloadOff := out.size
        sites := sites.push (ti, ri, pi, payloadOff)
        out := pushU32LE out vals[pi]!
      totalRecords := totalRecords + 1

    let taskEnd := out.size
    totalRecordLimit := totalRecordLimit + (taskEnd - recStart) / 8

    -- Align to 16 bytes between tasks (or before trailing zero blocks on H17).
    -- Never pad after the final task unless trailingZeroBlocks > 0 (H17 only).
    let isLast := ti + 1 == stream.tasks.size
    if !isLast || stream.trailingZeroBlocks > 0 then
      out := padToAlign out 16

  if stream.trailingZeroBlocks > 0 then
    out := pushZeros out (stream.trailingZeroBlocks * 16)

  if out.size < 16 || out.size > maxTextSize then
    throw s!"encoded task stream size {out.size} out of range 16..{maxTextSize}"
  if out.size % 4 != 0 then
    throw s!"encoded task stream size {out.size} is not 4-byte aligned"

  pure {
    bytes := out
    taskCount := stream.tasks.size
    recordCount := totalRecords
    recordLimit := totalRecordLimit
    payloadSites := sites
  }

/-- Validate raw `__TEXT,__text` bytes according to all structural task-stream invariants. -/
def validateTaskStream (target : Target) (bs : ByteArray) : Except String TaskStreamInfo := do
  if bs.size < prologueFrameBytes || bs.size > maxTextSize then
    throw s!"task stream size {bs.size} out of range [{prologueFrameBytes}, {maxTextSize}]"
  if bs.size % 4 != 0 then
    throw s!"task stream size {bs.size} is not a multiple of 4"

  let prof := target.profile
  let prologue ← readU32LE bs 0
  if !prof.isValidPrologue prologue then
    throw s!"{prof.name} task stream prologue {prologue} is invalid"
  checkAllZero bs 4 (prologueFrameBytes - 4) "task stream opening frame"

  let mut cursor := prologueFrameBytes
  let mut taskCount := 0
  let mut recordCount := 0
  let mut recordLimit := 0
  let mut finalTaskEnd := 0
  let mut barSlots : Array UInt8 := #[]
  let mut payloadSites : Array Nat := #[]

  while cursor < bs.size do
    if cursor + taskFrameAlignBytes > bs.size then
      throw s!"task stream at offset {toHex cursor}: fewer than {taskFrameAlignBytes} bytes remain ({bs.size - cursor})"
    let w0 ← readU32LE bs cursor
    let words := ((w0 >>> 16) &&& 0x7ff).toNat
    if words == 0 then
      checkAllZero bs cursor taskFrameAlignBytes s!"task stream zero block at {toHex cursor}"
      cursor := cursor + taskFrameAlignBytes
    else
      if words < 9 then
        throw s!"task at {toHex cursor}: word count {words} < 9"
      let taskBytes := 4 * words
      if cursor + taskBytes > bs.size then
        throw s!"task at {toHex cursor}: extends past end of __text ({cursor + taskBytes} > {bs.size})"
      let taskEnd := cursor + taskBytes
      let w8 ← readU32LE bs (cursor + 32)
      let headerWords := if (w8 &&& 3) == 3 then 10 else 9
      if words < headerWords then
        throw s!"task at {toHex cursor}: word count {words} < header words {headerWords}"

      let recStart := cursor + 4 * headerWords
      recordLimit := recordLimit + (taskEnd - recStart) / 8

      let mut recPos := recStart
      while recPos < taskEnd do
        if recPos + 4 > taskEnd then
          throw s!"task at {toHex cursor}: truncated register record header at {toHex recPos}"
        let h ← readU32LE bs recPos
        recPos := recPos + 4
        let base := (h &&& 0x7fff).toNat
        let isBitmask := ((h >>> 31) &&& 1) == 1
        let count ←
          if isBitmask then
            let mask := ((h >>> 15) &&& 0xffff).toUInt16
            if !isAllowedRegWord base then
              throw s!"task at {toHex cursor}, bitmask record: base register word {toHex base} (byte {toHex (base * 4)}) outside allowed blocks"
            for i in [:16] do
              if ((mask.toNat >>> i) &&& 1) == 1 then
                let reg := base + i + 1
                if !isAllowedRegWord reg then
                  throw s!"task at {toHex cursor}, bitmask record: register word {toHex reg} (byte {toHex (reg * 4)}) outside allowed blocks"
            pure (1 + popcount16 mask)
          else
            let cnt := (((h >>> 15) &&& 0x3f).toNat) + 1
            for i in [:cnt] do
              let reg := base + i
              if !isAllowedRegWord reg then
                throw s!"task at {toHex cursor}, sequential record: register word {toHex reg} (byte {toHex (reg * 4)}) outside allowed blocks"
            if ((h >>> 29) &&& 1) == 1 then
              let slot := ((h >>> 23) &&& 0x3f).toUInt8
              if !barSlots.contains slot then
                barSlots := barSlots.push slot
            pure cnt

        if recPos + 4 * count > taskEnd then
          throw s!"task at {toHex cursor}: register record at {toHex (recPos - 4)} payload ({count} words) overruns task end {toHex taskEnd}"
        for i in [:count] do
          payloadSites := payloadSites.push (recPos + 4 * i)
        recPos := recPos + 4 * count
        recordCount := recordCount + 1

      if recPos != taskEnd then
        throw s!"task at {toHex cursor}: register records ended at {toHex recPos} != task end {toHex taskEnd}"

      taskCount := taskCount + 1
      finalTaskEnd := taskEnd

      if taskEnd < bs.size then
        let nextCursor := alignUp taskEnd taskFrameAlignBytes
        if nextCursor > bs.size then
          throw s!"task at {toHex cursor}: inter-task alignment target {toHex nextCursor} exceeds __text size {toHex bs.size}"
        checkAllZero bs taskEnd (nextCursor - taskEnd) s!"inter-task zero padding after {toHex taskEnd}"
        cursor := nextCursor
      else
        cursor := taskEnd

  if taskCount == 0 then
    throw "task stream contains 0 tasks"

  if prof.requireTightEnd && finalTaskEnd != bs.size then
    throw s!"{prof.name} task stream forbids trailing zero padding/blocks after final task (final_task_end={toHex finalTaskEnd}, text.size={toHex bs.size})"

  pure {
    prologue := prologue
    taskCount := taskCount
    recordCount := recordCount
    recordLimit := recordLimit
    barSlots := barSlots
    payloadSites := payloadSites
  }

end Vane.TaskStream
