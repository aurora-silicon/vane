import Vane.Binary
import Vane.TaskStream
import Vane.Hwx

namespace Vane.Sections

open Vane.Binary
open Vane.TaskStream
open Vane.Hwx

def minSectionSize : Nat := 0x28
def maxSectionSize : Nat := 0x10000
def maxBufferId : UInt32 := 0x3c
def kernelResourceTag : UInt32 := 1007

/-- Four host-built LOAD sections required by `accel/ane` ABI v3. -/
structure LoadSections where
  generic : ByteArray
  operation : ByteArray
  procedure : ByteArray
  textProperties : ByteArray
  deriving BEq

/-- Deterministic buffer ID assignment for surfaces and optional scratch. -/
structure BufferAssignment where
  surfaceIds : Array UInt32
  scratchId : Option UInt32
  deriving Repr, BEq

/-- Assign deterministic buffer IDs: output = 4, inputs = 5, 6, ..., optional scratch = next ID. -/
def assignBufferIds (info : HwxInfo) : Except String BufferAssignment := do
  let mut nextInputId : UInt32 := 5
  let mut surfIds : Array UInt32 := #[]
  for s in info.surfaces do
    match s.direction with
    | .output =>
      surfIds := surfIds.push 4
    | .input =>
      if nextInputId > maxBufferId then
        throw s!"input buffer ID {nextInputId} exceeds {maxBufferId}"
      surfIds := surfIds.push nextInputId
      nextInputId := nextInputId + 1
  let scId ←
    if info.scratchSection.isSome then
      if nextInputId > maxBufferId then
        throw s!"scratch buffer ID {nextInputId} exceeds {maxBufferId}"
      pure (some nextInputId)
    else
      pure none
  pure { surfaceIds := surfIds, scratchId := scId }

private def buildProcedureSection : ByteArray := Id.run do
  let mut buf := zeroBytes 0x38
  buf := (putU32LE buf 0x00 1).toOption.get!
  buf := (putU32LE buf 0x04 0).toOption.get!
  buf := (putU64LE buf 0x08 0x18).toOption.get!
  buf := (putU64LE buf 0x10 0x20).toOption.get!
  let words : Array UInt32 := #[1, 3, 0, 0, 1, 0, 0xffffffff, 4]
  for i in [:words.size] do
    buf := (putU32LE buf (0x18 + 4 * i) words[i]!).toOption.get!
  buf

private def buildTextPropertiesSection (taskCount : UInt32) (textSize : UInt64) : Except String ByteArray := do
  let mut buf := zeroBytes 0x28
  buf ← putU32LE buf 0x00 1
  buf ← putU32LE buf 0x04 0
  buf ← putU32LE buf 0x08 0
  buf ← putU32LE buf 0x0c taskCount
  buf ← putU64LE buf 0x10 0
  buf ← putU64LE buf 0x18 0
  buf ← putU64LE buf 0x20 textSize
  pure buf

private def resolveBarSlotBase (info : HwxInfo) (slot : UInt8) : Except String UInt64 := do
  if slot.toNat > 63 then
    throw s!"BAR slot {slot} exceeds 0..63"
  let off := 0x10 * (slot.toNat + 1)
  readU64LE info.programDescBytes off

/-- Build all four host-built LOAD sections deterministically from a validated HWX. -/
def buildSectionsFromInfo (info : HwxInfo) : Except String LoadSections := do
  let ids ← assignBufferIds info
  let entries := info.surfaces.size + (if info.scratchSection.isSome then 1 else 0)

  -- 1. GENERIC section
  let genSize := 0x208 + entries * 0x30
  let mut gen := zeroBytes genSize
  gen ← putU32LE gen 0x000 1
  gen ← putU32LE gen 0x004 0x10
  gen ← putU32LE gen 0x204 entries.toUInt32

  for i in [:info.surfaces.size] do
    let s := info.surfaces[i]!
    let bid := ids.surfaceIds[i]!
    let dir := s.direction.toUInt32
    let base := 0x208 + i * 0x30
    gen ← putU32LE gen (base + 0x00) 1
    gen ← putU32LE gen (base + 0x04) bid
    gen ← putU32LE gen (base + 0x08) (dir - 1)
    gen ← putU32LE gen (base + 0x10) dir
    gen ← putU64LE gen (base + 0x20) s.vmSize
    gen ← putU32LE gen (base + 0x28) 0xffff

  if let some sc := info.scratchSection then
    let bid := ids.scratchId.get!
    let base := 0x208 + info.surfaces.size * 0x30
    gen ← putU32LE gen (base + 0x00) 1
    gen ← putU32LE gen (base + 0x04) bid
    gen ← putU32LE gen (base + 0x08) 0
    gen ← putU32LE gen (base + 0x10) 1
    gen ← putU64LE gen (base + 0x20) sc.size
    gen ← putU32LE gen (base + 0x28) 0xffff

  -- 2. OPERATION section (single-record form: 4 + 0x40c = 0x410 bytes)
  let mut bindings : Array (UInt32 × UInt32) := #[]
  let mut hasKernelBinding := false
  for slot in info.taskStream.barSlots do
    let vmBase ← resolveBarSlotBase info slot
    if vmBase != 0 then
      let mut matched : Option UInt32 := none
      for i in [:info.surfaces.size] do
        if info.surfaces[i]!.sect.addr == vmBase then
          if matched.isSome then
            throw s!"BAR slot {slot} base {toHex vmBase.toNat} matches multiple resources"
          matched := some ids.surfaceIds[i]!
      if let some sc := info.scratchSection then
        if sc.addr == vmBase then
          if matched.isSome then
            throw s!"BAR slot {slot} base {toHex vmBase.toNat} matches multiple resources"
          matched := some ids.scratchId.get!
      if let some ks := info.kernelSection then
        if ks.addr == vmBase then
          if matched.isSome then
            throw s!"BAR slot {slot} base {toHex vmBase.toNat} matches multiple resources"
          matched := some kernelResourceTag
          hasKernelBinding := true
      match matched with
      | some tag => bindings := bindings.push (slot.toUInt32, tag)
      | none =>
        throw s!"BAR slot {slot} references non-zero base {toHex vmBase.toNat} that does not match any surface, scratch, or kernel"

  if bindings.isEmpty then
    throw "cannot build OPERATION section: task stream references no non-zero BAR slots"
  if bindings.size > 64 then
    throw s!"OPERATION section binding count {bindings.size} exceeds 64"
  if info.kernelSection.isSome && !hasKernelBinding then
    throw "HWX contains __KERN_0 but task stream never references the kernel BAR slot"

  let mut op := zeroBytes 0x410
  op ← putU32LE op 0x00 1
  op ← putU32LE op 0x0c bindings.size.toUInt32
  for j in [:bindings.size] do
    let (slot, tag) := bindings[j]!
    op ← putU32LE op (0x10 + 8 * j) slot
    op ← putU32LE op (0x14 + 8 * j) tag

  -- 3. PROCEDURE section
  let proc := buildProcedureSection

  -- 4. TEXT_PROPERTIES section
  let tprop ← buildTextPropertiesSection info.taskStream.taskCount.toUInt32 info.textSection.size

  pure {
    generic := gen
    operation := op
    procedure := proc
    textProperties := tprop
  }

/-- Build all four host-built LOAD sections from raw HWX bytes. -/
def buildSections (hwxBytes : ByteArray) : Except String LoadSections := do
  let info ← validateHwx hwxBytes
  buildSectionsFromInfo info

private def checkSectionSize (bs : ByteArray) (name : String) : Except String Unit := do
  if bs.size < minSectionSize || bs.size > maxSectionSize then
    throw s!"{name} section size {bs.size} out of range [{minSectionSize}, {maxSectionSize}]"
  if bs.size % 4 != 0 then
    throw s!"{name} section size {bs.size} is not a multiple of 4"

/-- Validate all four host-built LOAD sections against a validated `HwxInfo`. -/
def validateSectionsWithInfo (info : HwxInfo) (secs : LoadSections) : Except String Unit := do
  checkSectionSize secs.generic "GENERIC"
  checkSectionSize secs.operation "OPERATION"
  checkSectionSize secs.procedure "PROCEDURE"
  checkSectionSize secs.textProperties "TEXT_PROPERTIES"

  -- 1. Validate GENERIC
  let expectedEntries := info.surfaces.size + (if info.scratchSection.isSome then 1 else 0)
  let expectedGenSize := 0x208 + expectedEntries * 0x30
  if secs.generic.size != expectedGenSize then
    throw s!"GENERIC section size {secs.generic.size} != expected {expectedGenSize}"
  let magic ← readU32LE secs.generic 0x000
  if magic != 1 then
    throw s!"GENERIC magic {magic} != 1"
  let ver ← readU32LE secs.generic 0x004
  if ver < 1 || ver > 0x10 then
    throw s!"GENERIC version {ver} out of range 1..0x10"
  checkAllZero secs.generic 0x008 0x1fc "GENERIC header reserved"
  let entries ← readU32LE secs.generic 0x204
  if entries.toNat != expectedEntries then
    throw s!"GENERIC entries {entries} != expected {expectedEntries}"

  let mut usedBufIds : Array UInt32 := #[]
  let mut bufIdToAddr : Array (UInt32 × UInt64) := #[]

  for i in [:expectedEntries] do
    let base := 0x208 + i * 0x30
    let eKind ← readU32LE secs.generic (base + 0x00)
    let bid ← readU32LE secs.generic (base + 0x04)
    let dirMinus1 ← readU32LE secs.generic (base + 0x08)
    let pad0c ← readU32LE secs.generic (base + 0x0c)
    let dir ← readU32LE secs.generic (base + 0x10)
    checkAllZero secs.generic (base + 0x14) 0x0c s!"GENERIC entry {i} reserved +0x14"
    let sz ← readU64LE secs.generic (base + 0x20)
    let tail ← readU32LE secs.generic (base + 0x28)
    let pad2c ← readU32LE secs.generic (base + 0x2c)

    if eKind != 1 then
      throw s!"GENERIC entry {i}: field +0x00 is {eKind}, expected 1"
    if bid == 2 || bid == 3 then
      throw s!"GENERIC entry {i}: buffer ID {bid} is reserved (2 and 3 are forbidden)"
    if bid > maxBufferId then
      throw s!"GENERIC entry {i}: buffer ID {bid} exceeds maximum {maxBufferId}"
    if usedBufIds.contains bid then
      throw s!"GENERIC entry {i}: duplicate buffer ID {bid}"
    usedBufIds := usedBufIds.push bid

    if pad0c != 0 || tail != 0xffff || pad2c != 0 then
      throw s!"GENERIC entry {i}: invalid constant/padding fields"

    if i < info.surfaces.size then
      let s := info.surfaces[i]!
      let expDir := s.direction.toUInt32
      if dir != expDir || dirMinus1 != expDir - 1 then
        throw s!"GENERIC entry {i}: direction ({dir}, {dirMinus1}) does not match surface '{s.shortName}' ({expDir})"
      if sz < s.span || sz > s.vmSize then
        throw s!"GENERIC entry {i}: size {sz} outside [{s.span}, {s.vmSize}] for surface '{s.shortName}'"
      bufIdToAddr := bufIdToAddr.push (bid, s.sect.addr)
    else
      let sc := info.scratchSection.get!
      if dir != 1 || dirMinus1 != 0 then
        throw s!"GENERIC scratch entry {i}: direction must be 1 (dir-1 = 0)"
      let maxScSize ← alignUpU64 sc.size pageSize.toUInt64
      if sz < sc.size || sz > maxScSize then
        throw s!"GENERIC scratch entry {i}: size {sz} outside [{sc.size}, {maxScSize}]"
      bufIdToAddr := bufIdToAddr.push (bid, sc.addr)

  -- 2. Validate OPERATION
  let opRecs ← readU32LE secs.operation 0x00
  if opRecs < 1 || opRecs > 64 then
    throw s!"OPERATION record count {opRecs} out of range 1..64"
  let expectedOpSize := 4 + opRecs.toNat * 0x40c
  if secs.operation.size != expectedOpSize then
    throw s!"OPERATION section size {secs.operation.size} != expected {expectedOpSize}"

  let mut boundSlots : Array (UInt8 × UInt32) := #[]
  let mut sawKernelBinding := false

  for r in [:opRecs.toNat] do
    let rBase := 4 + r * 0x40c
    let f0 ← readU32LE secs.operation (rBase + 0x00)
    let f4 ← readU32LE secs.operation (rBase + 0x04)
    let refCount ← readU32LE secs.operation (rBase + 0x08)
    if f0 != 0 || f4 != 0 then
      throw s!"OPERATION record {r}: header words +0x00/+0x04 must be 0"
    if refCount > 64 then
      throw s!"OPERATION record {r}: ref_count {refCount} exceeds 64"

    let mut seenInRec : Array UInt32 := #[]
    for j in [:refCount.toNat] do
      let slot ← readU32LE secs.operation (rBase + 0x0c + 8 * j)
      let tag ← readU32LE secs.operation (rBase + 0x10 + 8 * j)
      if slot > 63 then
        throw s!"OPERATION record {r} binding {j}: BAR slot {slot} exceeds 0..63"
      if seenInRec.contains slot then
        throw s!"OPERATION record {r}: duplicate BAR slot {slot}"
      seenInRec := seenInRec.push slot

      let vmBase ← resolveBarSlotBase info slot.toUInt8
      if vmBase == 0 then
        throw s!"OPERATION record {r} binding {j}: BAR slot {slot} has zero VM base (config slot must not be bound)"

      if tag == kernelResourceTag then
        let kSec ← match info.kernelSection with
          | some k => pure k
          | none => throw s!"OPERATION record {r} binding {j}: tag 1007 used but HWX has no __KERN_0"
        if vmBase != kSec.addr then
          throw s!"OPERATION record {r} binding {j}: BAR slot {slot} VM base {toHex vmBase.toNat} != kernel addr {toHex kSec.addr.toNat}"
        sawKernelBinding := true
      else
        let expectedAddr ← match bufIdToAddr.find? (fun (b, _) => b == tag) with
          | some (_, a) => pure a
          | none => throw s!"OPERATION record {r} binding {j}: unknown buffer ID tag {tag}"
        if vmBase != expectedAddr then
          throw s!"OPERATION record {r} binding {j}: BAR slot {slot} VM base {toHex vmBase.toNat} != buffer {tag} addr {toHex expectedAddr.toNat}"

      boundSlots := boundSlots.push (slot.toUInt8, tag)

    let usedBytes := 0x0c + refCount.toNat * 8
    checkAllZero secs.operation (rBase + usedBytes) (0x40c - usedBytes) s!"OPERATION record {r} trailing bytes"

  -- Every non-zero-base BAR slot referenced by the task stream must be bound in OPERATION
  for slot in info.taskStream.barSlots do
    let vmBase ← resolveBarSlotBase info slot
    if vmBase != 0 then
      if !boundSlots.any (fun (s, _) => s == slot) then
        throw s!"OPERATION missing binding for task-stream BAR slot {slot} (VM base {toHex vmBase.toNat})"

  if info.kernelSection.isSome && !sawKernelBinding then
    throw "OPERATION missing required kernel binding (tag 1007) for __KERN_0"

  -- 3. Validate PROCEDURE
  checkBytesEqual secs.procedure buildProcedureSection "PROCEDURE section"

  -- 4. Validate TEXT_PROPERTIES
  let expectedTProp ← buildTextPropertiesSection info.taskStream.taskCount.toUInt32 info.textSection.size
  checkBytesEqual secs.textProperties expectedTProp "TEXT_PROPERTIES section"

  pure ()

/-- Validate all four host-built LOAD sections against raw HWX bytes. -/
def validateSections (hwxBytes : ByteArray) (secs : LoadSections) : Except String Unit := do
  let info ← validateHwx hwxBytes
  validateSectionsWithInfo info secs

end Vane.Sections
