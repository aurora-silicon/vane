import Vane.Binary

namespace Vane.TaskStream

open Vane.Binary

/-- Target ANE hardware family. -/
inductive Target where
  | h16
  | h17
  deriving Repr, BEq, DecidableEq

namespace Target

def cpuSubtype : Target → UInt32
  | .h16 => 7
  | .h17 => 9

def ofCpuSubtype? : UInt32 → Option Target
  | 7 => some .h16
  | 9 => some .h17
  | _ => none

def name : Target → String
  | .h16 => "h16"
  | .h17 => "h17"

def parse? (s : String) : Option Target :=
  match s.toLower with
  | "h16" => some .h16
  | "h17" => some .h17
  | _ => none

end Target

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
  match target with
  | .h16 =>
    if stream.prologue != 1 then
      throw s!"H16 task stream prologue must be 1, got {stream.prologue}"
    if stream.trailingZeroBlocks != 0 then
      throw s!"H16 task stream forbids trailing zero blocks after final task"
  | .h17 =>
    if stream.prologue != 0 && stream.prologue != 1 then
      throw s!"H17 task stream prologue must be 0 or 1, got {stream.prologue}"

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
  if bs.size < 16 || bs.size > maxTextSize then
    throw s!"task stream size {bs.size} out of range [16, {maxTextSize}]"
  if bs.size % 4 != 0 then
    throw s!"task stream size {bs.size} is not a multiple of 4"

  let prologue ← readU32LE bs 0
  match target with
  | .h16 =>
    if prologue != 1 then
      throw s!"H16 task stream prologue must be 1, got {prologue}"
  | .h17 =>
    if prologue != 0 && prologue != 1 then
      throw s!"H17 task stream prologue must be 0 or 1, got {prologue}"
  checkAllZero bs 4 12 "task stream opening frame"

  let mut pos := 16
  let mut taskCount := 0
  let mut recordCount := 0
  let mut recordLimit := 0
  let mut finalTaskEnd := 0
  let mut barSlots : Array UInt8 := #[]
  let mut payloadSites : Array Nat := #[]

  while pos < bs.size do
    if pos + 16 > bs.size then
      throw s!"task stream at offset {toHex pos}: fewer than 16 bytes remain ({bs.size - pos})"
    let w0 ← readU32LE bs pos
    let words := ((w0 >>> 16) &&& 0x7ff).toNat
    if words == 0 then
      checkAllZero bs pos 16 s!"task stream zero block at {toHex pos}"
      pos := pos + 16
    else
      if words < 9 then
        throw s!"task at {toHex pos}: word count {words} < 9"
      let taskBytes := 4 * words
      if pos + taskBytes > bs.size then
        throw s!"task at {toHex pos}: extends past end of __text ({pos + taskBytes} > {bs.size})"
      let taskEnd := pos + taskBytes
      let w8 ← readU32LE bs (pos + 32)
      let headerWords := if (w8 &&& 3) == 3 then 10 else 9
      if words < headerWords then
        throw s!"task at {toHex pos}: word count {words} < header words {headerWords}"

      let recStart := pos + 4 * headerWords
      recordLimit := recordLimit + (taskEnd - recStart) / 8

      let mut recPos := recStart
      while recPos < taskEnd do
        if recPos + 4 > taskEnd then
          throw s!"task at {toHex pos}: truncated register record header at {toHex recPos}"
        let h ← readU32LE bs recPos
        recPos := recPos + 4
        let base := (h &&& 0x7fff).toNat
        let isBitmask := ((h >>> 31) &&& 1) == 1
        let count ←
          if isBitmask then
            let mask := ((h >>> 15) &&& 0xffff).toUInt16
            if !isAllowedRegWord base then
              throw s!"task at {toHex pos}, bitmask record: base register word {toHex base} (byte {toHex (base * 4)}) outside allowed blocks"
            for i in [:16] do
              if ((mask.toNat >>> i) &&& 1) == 1 then
                let reg := base + i + 1
                if !isAllowedRegWord reg then
                  throw s!"task at {toHex pos}, bitmask record: register word {toHex reg} (byte {toHex (reg * 4)}) outside allowed blocks"
            pure (1 + popcount16 mask)
          else
            let cnt := (((h >>> 15) &&& 0x3f).toNat) + 1
            for i in [:cnt] do
              let reg := base + i
              if !isAllowedRegWord reg then
                throw s!"task at {toHex pos}, sequential record: register word {toHex reg} (byte {toHex (reg * 4)}) outside allowed blocks"
            if ((h >>> 29) &&& 1) == 1 then
              let slot := ((h >>> 23) &&& 0x3f).toUInt8
              if !barSlots.contains slot then
                barSlots := barSlots.push slot
            pure cnt

        if recPos + 4 * count > taskEnd then
          throw s!"task at {toHex pos}: register record at {toHex (recPos - 4)} payload ({count} words) overruns task end {toHex taskEnd}"
        for i in [:count] do
          payloadSites := payloadSites.push (recPos + 4 * i)
        recPos := recPos + 4 * count
        recordCount := recordCount + 1

      if recPos != taskEnd then
        throw s!"task at {toHex pos}: register records ended at {toHex recPos} != task end {toHex taskEnd}"

      taskCount := taskCount + 1
      finalTaskEnd := taskEnd

      if taskEnd < bs.size then
        let nextPos := alignUp taskEnd 16
        if nextPos > bs.size then
          throw s!"task at {toHex pos}: inter-task alignment target {toHex nextPos} exceeds __text size {toHex bs.size}"
        checkAllZero bs taskEnd (nextPos - taskEnd) s!"inter-task zero padding after {toHex taskEnd}"
        pos := nextPos
      else
        pos := taskEnd

  if taskCount == 0 then
    throw "task stream contains 0 tasks"

  if target == .h16 && finalTaskEnd != bs.size then
    throw s!"H16 task stream forbids trailing zero padding/blocks after final task (final_task_end={toHex finalTaskEnd}, text.size={toHex bs.size})"

  pure {
    prologue := prologue
    taskCount := taskCount
    recordCount := recordCount
    recordLimit := recordLimit
    barSlots := barSlots
    payloadSites := payloadSites
  }

end Vane.TaskStream
