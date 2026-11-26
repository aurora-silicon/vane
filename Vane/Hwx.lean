import Vane.Binary
import Vane.TaskStream

namespace Vane.Hwx

open Vane.Binary
open Vane.TaskStream

def hwxMagic : UInt32 := 0xbeefface
def hwxCpuType : UInt32 := 0x80
def hwxFileType : UInt32 := 2
def hwxHeaderFlags : UInt32 := 0x200000

def pageSize : Nat := 0x4000
def maxFileSize : Nat := 0x10100000
def maxSegmentVmSize : Nat := 0x10000000
def maxTotalNonGuardVmSize : Nat := 0x20000000
def maxSymtabSyms : Nat := 4096
def maxStrtabSize : Nat := 0x40000

def lcSymtab : UInt32 := 2
def lcDescriptor : UInt32 := 4
def lcIdent : UInt32 := 8
def lcSegment64 : UInt32 := 0x19
def lcNote : UInt32 := 0x31
def lcBufRef : UInt32 := 0x40

def descSubtypeTensor : UInt32 := 3
def descSubtypeProgram : UInt32 := 4

def progDescLinearSize : Nat := 0x8a0
def progDescScratchSize : Nat := 0x8c0

/-- Direction of an I/O surface (`1` = input, `2` = output). -/
inductive SurfaceDirection where
  | input
  | output
  deriving Repr, BEq, DecidableEq, Inhabited

namespace SurfaceDirection

def toUInt32 : SurfaceDirection → UInt32
  | .input => 1
  | .output => 2

def ofUInt32? : UInt32 → Option SurfaceDirection
  | 1 => some .input
  | 2 => some .output
  | _ => none

def name : SurfaceDirection → String
  | .input => "input"
  | .output => "output"

end SurfaceDirection

/-- High-level surface specification for HWX emission. -/
structure SurfaceSpec where
  shortName : String
  symbolName : String
  direction : SurfaceDirection
  n : UInt32 := 1
  channels : UInt32
  height : UInt32
  width : UInt32
  rowStride : UInt64
  planeStride : UInt64
  batchStride : UInt64
  sectionSize : Option UInt64 := none
  allocSize : Option UInt64 := none
  deriving Repr, BEq, Inhabited

namespace SurfaceSpec

/-- Create a tightly packed fp16 surface specification for `(n, channels, height, width)`. -/
def packed (shortName symbolName : String) (direction : SurfaceDirection)
    (n channels height width : UInt32) : SurfaceSpec :=
  let rowStride : UInt64 := width.toUInt64 * 2
  let planeStride : UInt64 := height.toUInt64 * rowStride
  let batchStride : UInt64 := channels.toUInt64 * planeStride
  {
    shortName := shortName
    symbolName := symbolName
    direction := direction
    n := n
    channels := channels
    height := height
    width := width
    rowStride := rowStride
    planeStride := planeStride
    batchStride := batchStride
  }

end SurfaceSpec

/-- High-level `__DATA,__bss` scratch specification. -/
structure ScratchSpec where
  size : UInt64
  usable : UInt32
  vmSize : Option UInt64 := none
  deriving Repr, BEq

/-- Relocation pointing at a 4-byte payload word inside `__TEXT,__text`. -/
structure RelocationSpec where
  patchOffset : UInt32
  deriving Repr, BEq

/-- Complete high-level HWX program container specification. -/
structure HwxProgram where
  target : Target
  surfaces : Array SurfaceSpec
  constants : ByteArray := zeroBytes 16
  kernel : Option ByteArray := none
  scratch : Option ScratchSpec := none
  taskStream : TaskStream
  relocations : Array RelocationSpec := #[]
  workHint : Option UInt32 := none
  ident : Option String := some "vane"
  noteVersion : Option UInt32 := none
  /-- Starting BAR slot index for the surface/kernel base table (`6` = `+0x70` on H16/H17, `5` = `+0x60` on H17). -/
  barTableSlot : UInt8 := 6

/-- Parsed and validated section metadata. -/
structure SectionInfo where
  ordinal : Nat
  sectName : String
  segName : String
  addr : UInt64
  size : UInt64
  offset : UInt32
  align : UInt32
  reloff : UInt32
  nreloc : UInt32
  flags : UInt32
  deriving Repr, BEq, Inhabited

/-- Validated surface metadata extracted from an HWX container. -/
structure ValidatedSurface where
  shortName : String
  symbolName : String
  direction : SurfaceDirection
  n : UInt32
  channels : UInt32
  height : UInt32
  width : UInt32
  rowStride : UInt64
  planeStride : UInt64
  batchStride : UInt64
  span : UInt64
  sect : SectionInfo
  vmSize : UInt64
  deriving Repr, BEq, Inhabited

/-- Complete validated summary of an HWX binary. -/
structure HwxInfo where
  target : Target
  textSection : SectionInfo
  constSection : SectionInfo
  kernelSection : Option SectionInfo
  scratchSection : Option SectionInfo
  scratchVmSize : Option UInt64
  scratchUsable : Option UInt32
  surfaces : Array ValidatedSurface
  programDescOffset : Nat
  programDescSize : Nat
  programDescBytes : ByteArray
  barTableOffset : Nat
  taskStream : TaskStreamInfo
  records : UInt32
  workHint : UInt32
  relocCount : Nat
  relocPatchOffsets : Array UInt32 := #[]
  ident : Option String
  noteVersion : Option UInt32

/-- Build the exact program descriptor bytes (linear `0x8a0` or scratch-backed `0x8c0`). -/
def buildProgramDescriptor
    (textAddr constAddr : UInt64)
    (scratchAddr : Option UInt64)
    (surfaceAddrs : Array UInt64)
    (kernelAddr : Option UInt64)
    (barTableOffset : Nat)
    (textWords taskCount records workHint : UInt32)
    (scratchUsable : Option UInt32) : Except String ByteArray := do
  let hasScratch := scratchAddr.isSome
  let cmdsize := if hasScratch then progDescScratchSize else progDescLinearSize
  let mut buf := zeroBytes cmdsize

  buf ← putU32LE buf 0x00 lcDescriptor
  buf ← putU32LE buf 0x04 cmdsize.toUInt32
  buf ← putU32LE buf 0x08 descSubtypeProgram
  buf ← putU32LE buf 0x0c (if hasScratch then 0x22a else 0x222)
  buf ← putU64LE buf 0x10 textAddr
  buf ← putU64LE buf 0x20 constAddr
  if let some sAddr := scratchAddr then
    buf ← putU64LE buf 0x40 sAddr

  -- BAR base table (5 slots at barTableOffset + 16*i)
  for i in [:5] do
    let val : UInt64 :=
      if i < surfaceAddrs.size then
        surfaceAddrs[i]!
      else if i == surfaceAddrs.size then
        kernelAddr.getD 0
      else
        0
    buf ← putU64LE buf (barTableOffset + 16 * i) val

  -- Common body at +0x810..+0x86f
  buf ← putU64LE buf 0x810 textAddr
  buf ← putU32LE buf 0x820 4
  buf ← putU32LE buf 0x824 textWords
  buf ← putU32LE buf 0x830 taskCount
  buf ← putU32LE buf 0x838 0xffffffff
  buf ← putU32LE buf 0x83c 0xffffffff
  buf ← putU32LE buf 0x844 0xffff
  buf ← putU32LE buf 0x848 (if hasScratch then 0x8b8 else 0x898)
  let usableVal := scratchUsable.getD 0
  buf ← putU32LE buf 0x858 (if hasScratch then usableVal else 0)
  buf ← putU32LE buf 0x860 records
  buf ← putU32LE buf 0x868 4

  let mainBytes := "main".toUTF8
  if !hasScratch then
    buf ← putU32LE buf 0x870 2
    buf ← putU32LE buf 0x878 7
    buf ← putU32LE buf 0x87c 8
    buf ← putU32LE buf 0x880 0x461c4000
    buf ← putU32LE buf 0x888 9
    buf ← putU32LE buf 0x88c 8
    buf ← putU32LE buf 0x890 workHint
    buf ← putU32LE buf 0x894 0
    buf ← putBytes buf 0x898 mainBytes
  else
    buf ← putU32LE buf 0x870 4
    buf ← putU32LE buf 0x878 2
    buf ← putU32LE buf 0x87c 8
    buf ← putU32LE buf 0x880 usableVal
    buf ← putU32LE buf 0x888 3
    buf ← putU32LE buf 0x88c 8
    buf ← putU32LE buf 0x890 0x08020004
    buf ← putU32LE buf 0x894 8
    buf ← putU32LE buf 0x898 7
    buf ← putU32LE buf 0x89c 8
    buf ← putU32LE buf 0x8a0 0x461c4000
    buf ← putU32LE buf 0x8a8 9
    buf ← putU32LE buf 0x8ac 8
    buf ← putU32LE buf 0x8b0 workHint
    buf ← putBytes buf 0x8b8 mainBytes

  pure buf

/-- Build the exact tensor descriptor bytes for a surface. -/
def buildTensorDescriptor
    (direction : SurfaceDirection)
    (n channels height width : UInt32)
    (batchStride planeStride rowStride : UInt64)
    (symbolName shortName : String)
    (cmdsizeOverride : Option Nat := none) : Except String ByteArray := do
  let symBytes ← validateAsciiNonNul symbolName "tensor symbol_name"
  let shortBytes ← validateAsciiNonNul shortName "tensor short_name"
  if symBytes.size < 1 || symBytes.size > 255 then
    throw s!"tensor symbol_name '{symbolName}' length must be 1..255"
  if shortBytes.size < 1 || shortBytes.size > 7 then
    throw s!"tensor short_name '{shortName}' length must be 1..7"

  let shortOff := 0xd3d + symBytes.size + 1
  let minSize := alignUp (shortOff + shortBytes.size + 1) 8
  let minCmdSize := max 0xd48 minSize
  let cmdsize := cmdsizeOverride.getD minCmdSize
  if cmdsize < 0xd48 || cmdsize > 0x4000 || cmdsize % 8 != 0 then
    throw s!"tensor descriptor cmdsize {toHex cmdsize} invalid"
  if shortOff + shortBytes.size + 1 > cmdsize then
    throw s!"tensor descriptor strings exceed cmdsize {toHex cmdsize}"

  let mut buf := zeroBytes cmdsize
  buf ← putU32LE buf 0x00 lcDescriptor
  buf ← putU32LE buf 0x04 cmdsize.toUInt32
  buf ← putU32LE buf 0x08 descSubtypeTensor
  buf ← putU32LE buf 0x0c 0x34a
  buf ← putU32LE buf 0x10 3
  buf ← putU32LE buf 0x14 direction.toUInt32
  buf ← putU32LE buf 0x18 0xd38
  buf ← putU32LE buf 0x1c 0
  buf ← putU32LE buf 0x20 0xd3d
  buf ← putU32LE buf 0x24 5
  buf ← putU32LE buf 0x28 n
  buf ← putU32LE buf 0x2c channels
  buf ← putU32LE buf 0x30 height
  buf ← putU32LE buf 0x34 width
  buf ← putU32LE buf 0x38 1
  buf ← putU64LE buf 0x50 batchStride
  buf ← putU64LE buf 0x58 planeStride
  buf ← putU64LE buf 0x60 rowStride
  buf ← putU64LE buf 0x68 2
  buf ← putU64LE buf 0x70 batchStride
  buf ← putU32LE buf 0x78 1
  buf ← putU32LE buf 0x7c shortOff.toUInt32
  buf ← putU64LE buf 0x80 batchStride
  buf ← putBytes buf 0xd38 "main".toUTF8
  buf ← putBytes buf 0xd3d symBytes
  buf ← putBytes buf shortOff shortBytes
  pure buf

private def encodeSegmentHeader
    (segName : String)
    (vmaddr vmsize fileoff filesize : UInt64)
    (prot nsects flags : UInt32) : Except String ByteArray := do
  let nameBytes ← encodeFixedAscii segName 16 "segment name"
  let cmdsize : UInt32 := (72 + nsects.toNat * 80).toUInt32
  let mut buf := ByteArray.empty
  buf := pushU32LE buf lcSegment64
  buf := pushU32LE buf cmdsize
  buf := buf ++ nameBytes
  buf := pushU64LE buf vmaddr
  buf := pushU64LE buf vmsize
  buf := pushU64LE buf fileoff
  buf := pushU64LE buf filesize
  buf := pushU32LE buf prot
  buf := pushU32LE buf prot
  buf := pushU32LE buf nsects
  buf := pushU32LE buf flags
  pure buf

private def encodeSectionHeader
    (sectName segName : String)
    (addr size : UInt64)
    (offset align reloff nreloc flags : UInt32) : Except String ByteArray := do
  let sectBytes ← encodeFixedAscii sectName 16 "section name"
  let segBytes ← encodeFixedAscii segName 16 "section segment name"
  let mut buf := ByteArray.empty
  buf := buf ++ sectBytes
  buf := buf ++ segBytes
  buf := pushU64LE buf addr
  buf := pushU64LE buf size
  buf := pushU32LE buf offset
  buf := pushU32LE buf align
  buf := pushU32LE buf reloff
  buf := pushU32LE buf nreloc
  buf := pushU32LE buf flags
  buf := pushZeros buf 12
  pure buf

private def encodeBufferRef (addr : UInt64) (shortName : String) : Except String ByteArray := do
  let shortBytes ← validateAsciiNonNul shortName "buffer ref short_name"
  if shortBytes.size < 1 || shortBytes.size > 7 then
    throw s!"buffer ref short_name '{shortName}' length must be 1..7"
  let nameSlot ← encodeFixedAscii shortName 8 "buffer ref short_name"
  let mut buf := ByteArray.empty
  buf := pushU32LE buf lcBufRef
  buf := pushU32LE buf 32
  buf := pushU64LE buf 0x18
  buf := pushU64LE buf addr
  buf := buf ++ nameSlot
  pure buf

private def encodeIdentCommand (ident : String) : Except String ByteArray := do
  let bytes ← validateAsciiNonNul ident "LC_IDENT"
  let cmdsize := alignUp (8 + bytes.size + 1) 8
  if cmdsize <= 8 || cmdsize > 4096 then
    throw s!"LC_IDENT cmdsize {cmdsize} out of range (8, 4096]"
  let mut buf := zeroBytes cmdsize
  buf ← putU32LE buf 0 lcIdent
  buf ← putU32LE buf 4 cmdsize.toUInt32
  buf ← putBytes buf 8 bytes
  pure buf

private def encodeNoteCommand (version : UInt32) : Except String ByteArray := do
  let nameSlot ← encodeFixedAscii "rt.version" 16 "LC_NOTE name"
  let mut buf := zeroBytes 48
  buf ← putU32LE buf 0 lcNote
  buf ← putU32LE buf 4 48
  buf ← putBytes buf 8 nameSlot
  buf ← putU64LE buf 24 40
  buf ← putU64LE buf 32 4
  buf ← putU32LE buf 40 version
  buf ← putU32LE buf 44 0
  pure buf

/-- Emit a complete valid HWX container binary from a high-level `HwxProgram`. -/
def emitHwx (prog : HwxProgram) : Except String ByteArray := do
  -- Basic target & metadata checks
  let prof := prog.target.profile
  if prog.noteVersion.isSome && !prof.allowNoteVersion then
    throw s!"LC_NOTE is not supported on {prof.name}"
  let barTableOffset ← match prog.barTableSlot with
    | 6 => pure 0x70
    | 5 =>
      if !prof.allowSlot5BarTable then
        throw s!"BAR table slot 5 (+0x60) is not valid on {prof.name}"
      pure 0x60
    | s => throw s!"unsupported barTableSlot {s}; expected 6 or 5 (when allowed by target)"

  if prog.constants.isEmpty then
    throw "__TEXT,__const payload must be non-empty"

  -- Validate surface counts and geometry
  if prog.surfaces.size < 2 || prog.surfaces.size > 4 then
    throw s!"surface count {prog.surfaces.size} must be in 2..4"
  let mut inputCount := 0
  let mut outputCount := 0
  let mut seenShort : Array String := #[]
  let mut seenSym : Array String := #[]
  let mut surfacePlans : Array (SurfaceSpec × UInt64 × UInt64 × UInt64) := #[]

  for i in [:prog.surfaces.size] do
    let s := prog.surfaces[i]!
    match s.direction with
    | .input => inputCount := inputCount + 1
    | .output => outputCount := outputCount + 1
    if seenShort.contains s.shortName then
      throw s!"duplicate surface shortName '{s.shortName}'"
    if seenSym.contains s.symbolName then
      throw s!"duplicate surface symbolName '{s.symbolName}'"
    seenShort := seenShort.push s.shortName
    seenSym := seenSym.push s.symbolName

    if s.n == 0 || s.channels == 0 || s.height == 0 || s.width == 0 then
      throw s!"surface '{s.shortName}': dimensions must all be > 0"
    let minRow ← mulU64 s.width.toUInt64 2 s!"surface '{s.shortName}' minRow"
    if s.rowStride < minRow then
      throw s!"surface '{s.shortName}': rowStride {s.rowStride} < width * 2 ({minRow})"
    let minPlane ← mulU64 s.height.toUInt64 s.rowStride s!"surface '{s.shortName}' minPlane"
    if s.planeStride < minPlane then
      throw s!"surface '{s.shortName}': planeStride {s.planeStride} < height * rowStride ({minPlane})"
    let minBatch ← mulU64 s.channels.toUInt64 s.planeStride s!"surface '{s.shortName}' minBatch"
    if s.batchStride < minBatch then
      throw s!"surface '{s.shortName}': batchStride {s.batchStride} < channels * planeStride ({minBatch})"
    let span ← mulU64 s.n.toUInt64 s.batchStride s!"surface '{s.shortName}' span"
    if span == 0 then
      throw s!"surface '{s.shortName}': span must be > 0"
    let sectSize := s.sectionSize.getD span
    if sectSize < span then
      throw s!"surface '{s.shortName}': sectionSize {sectSize} < span {span}"
    let defaultAlloc ← alignUpU64 sectSize pageSize.toUInt64
    let allocSize := s.allocSize.getD defaultAlloc
    if allocSize == 0 || allocSize.toNat % pageSize != 0 || allocSize.toNat > maxSegmentVmSize then
      throw s!"surface '{s.shortName}': allocSize {allocSize} must be 16 KiB aligned and <= 256 MiB"
    if sectSize > allocSize then
      throw s!"surface '{s.shortName}': sectionSize {sectSize} > allocSize {allocSize}"
    surfacePlans := surfacePlans.push (s, span, sectSize, allocSize)

  if inputCount < 1 then
    throw "program must have at least 1 input surface"
  if outputCount != 1 then
    throw s!"program must have exactly 1 output surface, got {outputCount}"

  -- Encode task stream
  let encText ← encodeTaskStream prog.target prog.taskStream
  if encText.recordCount < 1 || encText.recordCount > encText.recordLimit then
    throw s!"task stream must emit at least 1 register record (got {encText.recordCount})"

  -- Validate kernel & scratch specs
  let kernelBytes? ← match prog.kernel with
    | none => pure none
    | some kb =>
      if kb.isEmpty || kb.size > maxSegmentVmSize then
        throw s!"kernel size {kb.size} must be in 1..{maxSegmentVmSize}"
      pure (some kb)

  let scratchPlan? ← match prog.scratch with
    | none => pure none
    | some sc =>
      if sc.size == 0 || sc.size.toNat > maxSegmentVmSize then
        throw s!"scratch size {sc.size} must be in 1..{maxSegmentVmSize}"
      if sc.usable == 0 || sc.usable.toUInt64 > sc.size then
        throw s!"scratch usable {sc.usable} must satisfy 0 < usable <= {sc.size}"
      let defaultVm ← alignUpU64 sc.size pageSize.toUInt64
      let vmSize := sc.vmSize.getD defaultVm
      if vmSize == 0 || vmSize.toNat % pageSize != 0 || vmSize.toNat > maxSegmentVmSize || sc.size > vmSize then
        throw s!"scratch vmSize {vmSize} must be 16 KiB aligned, >= size, and <= 256 MiB"
      pure (some (sc, vmSize))

  -- Validate segment/section limits
  let segCount := 2 + (if scratchPlan?.isSome then 1 else 0) + (if kernelBytes?.isSome then 1 else 0) + prog.surfaces.size
  if segCount > 8 then
    throw s!"total segment count {segCount} exceeds maximum 8"
  let sectCount := 2 + (if scratchPlan?.isSome then 1 else 0) + (if kernelBytes?.isSome then 1 else 0) + prog.surfaces.size
  if sectCount > 8 then
    throw s!"total section count {sectCount} exceeds maximum 8"

  -- Assign VM addresses (canonical order: __PAGEZERO, __DATA?, __TEXT, __KERN_0?, __FVMLIB...)
  let mut vmCursor : UInt64 := pageSize.toUInt64

  let mut scratchVmAddr : Option UInt64 := none
  let mut nextOrdinal : Nat := 1
  if let some (_, scVmSize) := scratchPlan? then
    scratchVmAddr := some vmCursor
    vmCursor ← addU64 vmCursor scVmSize "scratch VM"
    nextOrdinal := nextOrdinal + 1

  -- __TEXT layout
  let textVmAddr := vmCursor
  let constRelOff := alignUp encText.bytes.size 64
  let textFileSize := constRelOff + prog.constants.size
  let textVmSize := alignUp textFileSize pageSize
  if textVmSize > maxSegmentVmSize then
    throw s!"__TEXT vmsize {textVmSize} exceeds 256 MiB"
  let constVmAddr ← addU64 textVmAddr constRelOff.toUInt64 "__const VM"
  vmCursor ← addU64 vmCursor textVmSize.toUInt64 "__TEXT VM"
  nextOrdinal := nextOrdinal + 2 -- __text, __const

  -- __KERN_0 layout
  let mut kernVmAddr : Option UInt64 := none
  let mut kernOrdinal : Option Nat := none
  let mut kernVmSize : Nat := 0
  if let some kb := kernelBytes? then
    kernVmAddr := some vmCursor
    kernOrdinal := some nextOrdinal
    kernVmSize := alignUp kb.size pageSize
    if kernVmSize > maxSegmentVmSize then
      throw s!"__KERN_0 vmsize {kernVmSize} exceeds 256 MiB"
    vmCursor ← addU64 vmCursor kernVmSize.toUInt64 "__KERN_0 VM"
    nextOrdinal := nextOrdinal + 1

  -- __FVMLIB surfaces layout
  let mut surfVmAddrs : Array UInt64 := #[]
  let mut surfOrdinals : Array Nat := #[]
  let mut totalSurfVmSize : Nat := 0
  for (_, _, _, allocSize) in surfacePlans do
    surfVmAddrs := surfVmAddrs.push vmCursor
    surfOrdinals := surfOrdinals.push nextOrdinal
    totalSurfVmSize := totalSurfVmSize + allocSize.toNat
    vmCursor ← addU64 vmCursor allocSize "surface VM"
    nextOrdinal := nextOrdinal + 1

  -- Check total non-guard rounded VM size <= 512 MiB
  let nonGuardVm :=
    (match scratchPlan? with | some (_, v) => alignUp v.toNat pageSize | none => 0) +
    alignUp textVmSize pageSize +
    (if kernelBytes?.isSome then alignUp kernVmSize pageSize else 0) +
    totalSurfVmSize
  if nonGuardVm > maxTotalNonGuardVmSize then
    throw s!"total non-guard VM size {nonGuardVm} exceeds 512 MiB"

  -- Validate relocations against encoded task stream and kernel
  if !prog.relocations.isEmpty then
    if kernelBytes?.isNone then
      throw "relocations require a __KERN_0 kernel section"
    let kb := kernelBytes?.get!
    let kAddr := kernVmAddr.get!
    if prog.relocations.size > encText.bytes.size / 4 then
      throw s!"relocation count {prog.relocations.size} exceeds __text word count"
    let payloadSet := encText.payloadSites.map (fun (_, _, _, off) => off)
    let mut seenPatches : Array UInt32 := #[]
    for rel in prog.relocations do
      if rel.patchOffset.toNat % 4 != 0 || rel.patchOffset.toNat + 4 > encText.bytes.size then
        throw s!"relocation patch offset {toHex rel.patchOffset.toNat} misaligned or out of bounds"
      if seenPatches.contains rel.patchOffset then
        throw s!"duplicate relocation patch offset {toHex rel.patchOffset.toNat}"
      if !payloadSet.contains rel.patchOffset.toNat then
        throw s!"relocation patch offset {toHex rel.patchOffset.toNat} does not target a register-record payload word"
      let addend ← readU32LE encText.bytes rel.patchOffset.toNat
      if addend.toNat >= kb.size then
        throw s!"relocation addend {toHex addend.toNat} at {toHex rel.patchOffset.toNat} >= kernel size {toHex kb.size}"
      let _ ← addU64 kAddr addend.toUInt64 "relocation target"
      seenPatches := seenPatches.push rel.patchOffset

  -- Build variable-size load commands first so we know exact sizeofcmds
  let workHint := prog.workHint.getD (min totalSurfVmSize maxU32Nat).toUInt32
  if workHint.toNat > totalSurfVmSize then
    throw s!"workHint {workHint} exceeds sum of surface vmsizes {totalSurfVmSize}"

  let progDesc ← buildProgramDescriptor
    textVmAddr constVmAddr scratchVmAddr surfVmAddrs kernVmAddr
    barTableOffset
    (encText.bytes.size / 4).toUInt32
    encText.taskCount.toUInt32
    encText.recordCount.toUInt32
    workHint
    (scratchPlan?.map fun (sc, _) => sc.usable)

  let mut bufRefCmds : Array ByteArray := #[]
  let mut tensorDescCmds : Array ByteArray := #[]
  for i in [:surfacePlans.size] do
    let (s, _, _, _) := surfacePlans[i]!
    let bcmd ← encodeBufferRef surfVmAddrs[i]! s.shortName
    bufRefCmds := bufRefCmds.push bcmd
    let tcmd ← buildTensorDescriptor s.direction s.n s.channels s.height s.width
      s.batchStride s.planeStride s.rowStride s.symbolName s.shortName
    tensorDescCmds := tensorDescCmds.push tcmd

  let identCmd? ← match prog.ident with
    | none => pure none
    | some idStr => pure (some (← encodeIdentCommand idStr))

  let noteCmd? ← match prog.noteVersion with
    | none => pure none
    | some ver => pure (some (← encodeNoteCommand ver))

  -- Compute sizeofcmds and ncmds
  let segCmdBytes :=
    72 + -- __PAGEZERO
    (if scratchPlan?.isSome then 72 + 80 else 0) +
    (72 + 2 * 80) + -- __TEXT
    (if kernelBytes?.isSome then 72 + 80 else 0) +
    surfacePlans.size * (72 + 80)
  let symtabCmdBytes := 24
  let bufRefBytes := bufRefCmds.foldl (fun acc c => acc + c.size) 0
  let tensorDescBytes := tensorDescCmds.foldl (fun acc c => acc + c.size) 0
  let identBytes := identCmd?.map (·.size) |>.getD 0
  let noteBytes := noteCmd?.map (·.size) |>.getD 0

  let sizeofcmds := segCmdBytes + symtabCmdBytes + bufRefBytes + tensorDescBytes + progDesc.size + identBytes + noteBytes
  let ncmds := segCount + 1 + bufRefCmds.size + tensorDescCmds.size + 1 +
    (if identCmd?.isSome then 1 else 0) + (if noteCmd?.isSome then 1 else 0)
  if ncmds > 64 then
    throw s!"load command count {ncmds} exceeds maximum 64"

  -- Compute file offsets
  let headerEnd := 32 + sizeofcmds
  let textFileOff := alignUp headerEnd pageSize
  let textFileEnd := textFileOff + textFileSize

  let (kernFileOff, afterKern) :=
    match kernelBytes? with
    | none => (0, textFileEnd)
    | some kb =>
      let off := alignUp textFileEnd pageSize
      (off, off + kb.size)

  let relBase := alignUp afterKern 8
  let (reloff, afterRel) :=
    if prog.relocations.isEmpty then (0, relBase)
    else (relBase, relBase + prog.relocations.size * 8)

  -- Build symbol table + string table (one symbol per surface shortName)
  let mut strtab := ByteArray.empty.push 0
  let mut symEntries := ByteArray.empty
  for i in [:surfacePlans.size] do
    let (s, _, _, _) := surfacePlans[i]!
    let strx := strtab.size.toUInt32
    strtab := (strtab ++ s.shortName.toUTF8).push 0
    symEntries := pushU32LE symEntries strx
    symEntries := pushU8 symEntries 0x0f
    symEntries := pushU8 symEntries surfOrdinals[i]!.toUInt8
    symEntries := pushU16LE symEntries 0
    symEntries := pushU64LE symEntries surfVmAddrs[i]!

  let nsyms := surfacePlans.size
  let symoff := afterRel
  let stroff := symoff + symEntries.size
  let totalFileSize := stroff + strtab.size
  if totalFileSize > maxFileSize then
    throw s!"emitted HWX size {totalFileSize} exceeds maximum {maxFileSize}"

  -- Assemble header + load commands
  let mut out := ByteArray.empty
  out := pushU32LE out hwxMagic
  out := pushU32LE out hwxCpuType
  out := pushU32LE out prog.target.cpuSubtype
  out := pushU32LE out hwxFileType
  out := pushU32LE out ncmds.toUInt32
  out := pushU32LE out sizeofcmds.toUInt32
  out := pushU32LE out hwxHeaderFlags
  out := pushU32LE out 0

  -- 1. __PAGEZERO
  out := out ++ (← encodeSegmentHeader "__PAGEZERO" 0 pageSize.toUInt64 0 0 0 0 4)

  -- 2. __DATA (optional, emitted before __TEXT)
  if let some (sc, scVmSize) := scratchPlan? then
    let scAddr := scratchVmAddr.get!
    out := out ++ (← encodeSegmentHeader "__DATA" scAddr scVmSize 0 0 3 1 4)
    out := out ++ (← encodeSectionHeader "__bss" "__DATA" scAddr sc.size 0 14 0 0 0x25)

  -- 3. __TEXT
  let textSegFlags : UInt32 := if scratchPlan?.isSome then 4 else 0
  let textSectFlags : UInt32 := if prog.relocations.isEmpty then 0x28 else 0x128
  out := out ++ (← encodeSegmentHeader "__TEXT" textVmAddr textVmSize.toUInt64
    textFileOff.toUInt64 textFileSize.toUInt64 5 2 textSegFlags)
  out := out ++ (← encodeSectionHeader "__text" "__TEXT" textVmAddr encText.bytes.size.toUInt64
    textFileOff.toUInt32 4 reloff.toUInt32 prog.relocations.size.toUInt32 textSectFlags)
  out := out ++ (← encodeSectionHeader "__const" "__TEXT" constVmAddr prog.constants.size.toUInt64
    (textFileOff + constRelOff).toUInt32 6 0 0 0x26)

  -- 4. __KERN_0 (optional)
  if let some kb := kernelBytes? then
    let kAddr := kernVmAddr.get!
    out := out ++ (← encodeSegmentHeader "__KERN_0" kAddr kernVmSize.toUInt64
      kernFileOff.toUInt64 kb.size.toUInt64 1 1 4)
    out := out ++ (← encodeSectionHeader "__kern_0" "__KERN_0" kAddr kb.size.toUInt64
      kernFileOff.toUInt32 6 0 0 0x26)

  -- 5. __FVMLIB surfaces
  for i in [:surfacePlans.size] do
    let (s, _, sectSize, allocSize) := surfacePlans[i]!
    let addr := surfVmAddrs[i]!
    let (sectName, sectFlags) := match s.direction with
      | .input => ("__const", (0x21 : UInt32))
      | .output => ("__data", (0x23 : UInt32))
    out := out ++ (← encodeSegmentHeader "__FVMLIB" addr allocSize 0 0 s.direction.toUInt32 1 6)
    out := out ++ (← encodeSectionHeader sectName "__FVMLIB" addr sectSize 0 14 0 0 sectFlags)

  -- 6. LC_SYMTAB
  out := pushU32LE out lcSymtab
  out := pushU32LE out 24
  out := pushU32LE out symoff.toUInt32
  out := pushU32LE out nsyms.toUInt32
  out := pushU32LE out stroff.toUInt32
  out := pushU32LE out strtab.size.toUInt32

  -- 7. Buffer references
  for c in bufRefCmds do
    out := out ++ c

  -- 8. Tensor descriptors
  for c in tensorDescCmds do
    out := out ++ c

  -- 9. Program descriptor
  out := out ++ progDesc

  -- 10. Optional LC_IDENT & LC_NOTE
  if let some c := identCmd? then
    out := out ++ c
  if let some c := noteCmd? then
    out := out ++ c

  if out.size != headerEnd then
    throw s!"internal error: load command stream size {out.size} != expected {headerEnd}"

  -- File payloads
  out := pushZeros out (textFileOff - out.size)
  out := out ++ encText.bytes
  out := pushZeros out (constRelOff - encText.bytes.size)
  out := out ++ prog.constants

  if let some kb := kernelBytes? then
    out := pushZeros out (kernFileOff - out.size)
    out := out ++ kb

  if !prog.relocations.isEmpty then
    out := pushZeros out (reloff - out.size)
    let kOrd := kernOrdinal.get!.toUInt32
    let info : UInt32 := ((7 : UInt32) <<< 24) ||| (kOrd &&& 0xffffff)
    for rel in prog.relocations do
      out := pushU32LE out rel.patchOffset
      out := pushU32LE out info

  out := pushZeros out (symoff - out.size)
  out := out ++ symEntries
  out := out ++ strtab

  pure out

private structure RawBufRef where
  addr : UInt64
  shortName : String
  deriving Inhabited

private structure RawTensorDesc where
  offset : Nat
  cmdsize : Nat
  direction : SurfaceDirection
  n : UInt32
  channels : UInt32
  height : UInt32
  width : UInt32
  batchStride : UInt64
  planeStride : UInt64
  rowStride : UInt64
  symbolName : String
  shortName : String
  deriving Inhabited

private structure RawSurfaceSeg where
  prot : UInt32
  vmSize : UInt64
  sect : SectionInfo
  deriving Inhabited

/-- Full pure-Lean validator verifying every HWX container, task-stream, descriptor, symbol, and relocation invariant. -/
def validateHwx (bs : ByteArray) : Except String HwxInfo := do
  -- §1 File Header
  if bs.size < 32 || bs.size > maxFileSize then
    throw s!"HWX file size {bs.size} out of range [32, {maxFileSize}]"
  let magic ← readU32LE bs 0x00
  if magic != hwxMagic then
    throw s!"bad HWX magic: got {toHex magic.toNat}, expected {toHex hwxMagic.toNat}"
  let cpuType ← readU32LE bs 0x04
  if cpuType != hwxCpuType then
    throw s!"bad CPU type: got {toHex cpuType.toNat}, expected {toHex hwxCpuType.toNat}"
  let cpuSubtype ← readU32LE bs 0x08
  let target ← match Target.ofCpuSubtype? cpuSubtype with
    | some t => pure t
    | none => throw s!"unsupported CPU subtype {cpuSubtype} (expected 7 for H16 or 9 for H17)"
  let prof := target.profile
  let fileType ← readU32LE bs 0x0c
  if fileType != hwxFileType then
    throw s!"bad file type: got {fileType}, expected {hwxFileType}"
  let ncmds ← readU32LE bs 0x10
  if ncmds < 1 || ncmds > 64 then
    throw s!"ncmds {ncmds} out of range 1..64"
  let sizeofcmds ← readU32LE bs 0x14
  let cmdEnd := 32 + sizeofcmds.toNat
  if cmdEnd > bs.size then
    throw s!"load command stream end {cmdEnd} exceeds file size {bs.size}"
  let headerFlags ← readU32LE bs 0x18
  if headerFlags != hwxHeaderFlags then
    throw s!"bad header flags: got {toHex headerFlags.toNat}, expected {toHex hwxHeaderFlags.toNat}"
  let reserved ← readU32LE bs 0x1c
  if reserved != 0 then
    throw s!"non-zero header reserved word {toHex reserved.toNat}"

  -- §2 & §3 Walk load commands
  let mut cursor := 32
  let mut segCount := 0
  let mut nonGuardCount := 0
  let mut totalNonGuardVm : Nat := 0
  let mut allSections : Array SectionInfo := #[]
  let mut vmRanges : Array ByteRange := #[]
  let mut sectVmRanges : Array ByteRange := #[]
  let mut fileRanges : Array ByteRange := #[]

  let mut pageZeroCount := 0
  let mut dataCount := 0
  let mut textCount := 0
  let mut kernCount := 0
  let mut textSegFlags : UInt32 := 0
  let mut textSection? : Option SectionInfo := none
  let mut constSection? : Option SectionInfo := none
  let mut kernSection? : Option SectionInfo := none
  let mut scratchSection? : Option SectionInfo := none
  let mut scratchVmSize? : Option UInt64 := none
  let mut surfaceSegs : Array RawSurfaceSeg := #[]

  let mut symtab? : Option (UInt32 × UInt32 × UInt32 × UInt32) := none
  let mut bufRefs : Array RawBufRef := #[]
  let mut tensorDescs : Array RawTensorDesc := #[]
  let mut progDescLoc? : Option (Nat × Nat) := none
  let mut ident? : Option String := none
  let mut noteVer? : Option UInt32 := none

  for ci in [:ncmds.toNat] do
    if cursor + 8 > cmdEnd then
      throw s!"load command {ci} truncated at offset {toHex cursor}"
    let cmd ← readU32LE bs cursor
    let cmdsize32 ← readU32LE bs (cursor + 4)
    let cmdsize := cmdsize32.toNat
    if cmdsize < 8 || cmdsize % 8 != 0 || cursor + cmdsize > cmdEnd then
      throw s!"load command {ci} (cmd={toHex cmd.toNat}): invalid cmdsize {toHex cmdsize} at {toHex cursor}"

    if cmd == lcSegment64 then
      if cmdsize < 72 then
        throw s!"LC_SEGMENT_64 at {toHex cursor}: cmdsize {cmdsize} < 72"
      let segName ← readFixedAscii bs (cursor + 8) 16 s!"segment name at {toHex cursor}"
      let vmaddr ← readU64LE bs (cursor + 0x18)
      let vmsize ← readU64LE bs (cursor + 0x20)
      let fileoff ← readU64LE bs (cursor + 0x28)
      let filesize ← readU64LE bs (cursor + 0x30)
      let maxprot ← readU32LE bs (cursor + 0x38)
      let initprot ← readU32LE bs (cursor + 0x3c)
      let nsects ← readU32LE bs (cursor + 0x40)
      let flags ← readU32LE bs (cursor + 0x44)

      if cmdsize != 72 + nsects.toNat * 80 then
        throw s!"segment '{segName}': cmdsize {cmdsize} != 72 + {nsects} * 80"
      if maxprot != initprot then
        throw s!"segment '{segName}': maxprot ({maxprot}) != initprot ({initprot})"
      if vmsize == 0 then
        throw s!"segment '{segName}': vmsize must be > 0"
      let _ ← addU64 vmaddr vmsize s!"segment '{segName}' VM range"

      segCount := segCount + 1
      if segCount > 8 then
        throw s!"segment count exceeds maximum 8"
      vmRanges := vmRanges.push { label := s!"seg:{segName}#{segCount}", start := vmaddr.toNat, size := vmsize.toNat }

      if segName != "__PAGEZERO" then
        nonGuardCount := nonGuardCount + 1
        if nonGuardCount > 7 then
          throw s!"non-guard segment count exceeds maximum 7"
        totalNonGuardVm := totalNonGuardVm + alignUp vmsize.toNat pageSize

      if filesize > 0 then
        let _ ← addU64 fileoff filesize s!"segment '{segName}' file range"
        fileRanges := fileRanges.push { label := s!"segfile:{segName}", start := fileoff.toNat, size := filesize.toNat }

      -- Parse sections belonging to this segment
      let mut segSects : Array SectionInfo := #[]
      for si in [:nsects.toNat] do
        let sOff := cursor + 72 + si * 80
        let sectName ← readFixedAscii bs sOff 16 s!"section name in '{segName}'"
        -- Check segname matches parent segment's segname byte-for-byte
        let parentSegRaw ← readSlice bs (cursor + 8) 16
        let sectSegRaw ← readSlice bs (sOff + 16) 16
        checkBytesEqual sectSegRaw parentSegRaw s!"section '{sectName}' segname vs parent '{segName}'"
        let addr ← readU64LE bs (sOff + 0x20)
        let size ← readU64LE bs (sOff + 0x28)
        let offset ← readU32LE bs (sOff + 0x30)
        let align ← readU32LE bs (sOff + 0x34)
        let reloff ← readU32LE bs (sOff + 0x38)
        let nreloc ← readU32LE bs (sOff + 0x3c)
        let sFlags ← readU32LE bs (sOff + 0x40)
        checkAllZero bs (sOff + 0x44) 12 s!"section '{segName},{sectName}' reserved bytes"

        if align > 31 then
          throw s!"section '{segName},{sectName}': align {align} > 31"
        let alignBytes := 1 <<< align.toNat
        if addr.toNat % alignBytes != 0 then
          throw s!"section '{segName},{sectName}': addr {toHex addr.toNat} not aligned to 2^{align}"
        if size == 0 then
          throw s!"section '{segName},{sectName}': size must be > 0"
        let sectEnd ← addU64 addr size s!"section '{segName},{sectName}' end"
        let segEnd ← addU64 vmaddr vmsize s!"segment '{segName}' end"
        if addr < vmaddr || sectEnd > segEnd then
          throw s!"section '{segName},{sectName}' [{toHex addr.toNat}, {toHex sectEnd.toNat}) outside parent segment"

        let ord := allSections.size + 1
        if ord > 8 then
          throw "total section count exceeds maximum 8"
        let sInfo : SectionInfo := {
          ordinal := ord
          sectName := sectName
          segName := segName
          addr := addr
          size := size
          offset := offset
          align := align
          reloff := reloff
          nreloc := nreloc
          flags := sFlags
        }
        allSections := allSections.push sInfo
        segSects := segSects.push sInfo
        sectVmRanges := sectVmRanges.push { label := s!"sect:{segName},{sectName}#{ord}", start := addr.toNat, size := size.toNat }

      -- Per-segment validation
      if segName == "__PAGEZERO" then
        pageZeroCount := pageZeroCount + 1
        if pageZeroCount > 1 then throw "duplicate __PAGEZERO segment"
        if vmaddr != 0 || vmsize != pageSize.toUInt64 || fileoff != 0 || filesize != 0 ||
            initprot != 0 || nsects != 0 || flags != 4 then
          throw "__PAGEZERO segment header fields do not match required values"
      else if segName == "__DATA" then
        dataCount := dataCount + 1
        if dataCount > 1 then throw "duplicate __DATA segment"
        if vmaddr.toNat % pageSize != 0 || vmsize.toNat % pageSize != 0 || vmsize.toNat > maxSegmentVmSize then
          throw "__DATA segment VM address/size invalid"
        if fileoff != 0 || filesize != 0 || initprot != 3 || nsects != 1 || flags != 4 then
          throw "__DATA segment header fields invalid"
        let sec := segSects[0]!
        if sec.sectName != "__bss" || sec.flags != 0x25 || sec.addr != vmaddr ||
            sec.offset != 0 || sec.reloff != 0 || sec.nreloc != 0 then
          throw "__DATA section must be __bss with flags=0x25, addr=vmaddr, offset=reloff=nreloc=0"
        scratchSection? := some sec
        scratchVmSize? := some vmsize
      else if segName == "__TEXT" then
        textCount := textCount + 1
        if textCount > 1 then throw "duplicate __TEXT segment"
        if vmaddr.toNat % pageSize != 0 || vmsize.toNat > maxSegmentVmSize then
          throw "__TEXT segment VM address/size invalid"
        if fileoff.toNat % pageSize != 0 || filesize == 0 || filesize > vmsize then
          throw "__TEXT segment fileoff/filesize invalid"
        if initprot != 5 || nsects != 2 then
          throw "__TEXT segment must have prot=5 and nsects=2"
        if flags != 0 && flags != 4 then
          throw s!"__TEXT segment invalid flags {toHex flags.toNat}"
        if flags == 4 && dataCount == 0 then
          throw "__TEXT flags=4 requires __DATA scratch segment to appear before __TEXT"
        textSegFlags := flags

        for sec in segSects do
          let alignBytes := 1 <<< sec.align.toNat
          if sec.offset.toNat % alignBytes != 0 then
            throw s!"__TEXT,{sec.sectName}: offset {toHex sec.offset.toNat} not aligned to 2^{sec.align}"
          if sec.offset.toUInt64 < fileoff ||
              sec.offset.toNat - fileoff.toNat != sec.addr.toNat - vmaddr.toNat ||
              sec.offset.toNat + sec.size.toNat > fileoff.toNat + filesize.toNat then
            throw s!"__TEXT,{sec.sectName}: file offset mismatch with parent __TEXT segment"
          if sec.sectName == "__text" then
            if textSection?.isSome then throw "duplicate __TEXT,__text section"
            let expectedFlags : UInt32 := if sec.nreloc > 0 then 0x128 else 0x28
            if sec.flags != expectedFlags then
              throw s!"__TEXT,__text flags {toHex sec.flags.toNat} != expected {toHex expectedFlags.toNat}"
            if sec.nreloc.toNat > maxTextSize / 4 then
              throw s!"__TEXT,__text nreloc {sec.nreloc} exceeds maximum"
            if sec.reloff.toNat > bs.size then
              throw s!"__TEXT,__text reloff {sec.reloff} exceeds file size"
            if sec.nreloc > 0 then
              fileRanges := fileRanges.push {
                label := "relocations"
                start := sec.reloff.toNat
                size := sec.nreloc.toNat * 8
              }
            textSection? := some sec
          else if sec.sectName == "__const" then
            if constSection?.isSome then throw "duplicate __TEXT,__const section"
            if sec.flags != 0x26 || sec.reloff != 0 || sec.nreloc != 0 then
              throw "__TEXT,__const must have flags=0x26 and reloff=nreloc=0"
            constSection? := some sec
          else
            throw s!"unexpected section '{sec.sectName}' in __TEXT"
        if textSection?.isNone || constSection?.isNone then
          throw "__TEXT must contain both __text and __const sections"
      else if segName == "__KERN_0" then
        kernCount := kernCount + 1
        if kernCount > 1 then throw "duplicate __KERN_0 segment"
        if vmaddr.toNat % pageSize != 0 || vmsize.toNat > maxSegmentVmSize then
          throw "__KERN_0 segment VM address/size invalid"
        if fileoff.toNat % pageSize != 0 || filesize == 0 || filesize > vmsize then
          throw "__KERN_0 segment fileoff/filesize invalid"
        if initprot != 1 || nsects != 1 || flags != 4 then
          throw "__KERN_0 segment must have prot=1, nsects=1, flags=4"
        let sec := segSects[0]!
        let alignBytes := 1 <<< sec.align.toNat
        if sec.sectName != "__kern_0" || sec.flags != 0x26 || sec.addr.toNat % 64 != 0 ||
            sec.size.toNat > maxSegmentVmSize || sec.reloff != 0 || sec.nreloc != 0 then
          throw "__KERN_0,__kern_0 section constraints violated"
        if sec.offset.toNat % alignBytes != 0 || sec.offset.toUInt64 < fileoff ||
            sec.offset.toNat - fileoff.toNat != sec.addr.toNat - vmaddr.toNat ||
            sec.offset.toNat + sec.size.toNat > fileoff.toNat + filesize.toNat then
          throw "__KERN_0,__kern_0 file offset mismatch"
        kernSection? := some sec
      else if segName == "__FVMLIB" then
        if vmaddr.toNat % pageSize != 0 || vmsize.toNat % pageSize != 0 || vmsize.toNat > maxSegmentVmSize then
          throw "__FVMLIB segment VM address/size invalid"
        if fileoff != 0 || filesize != 0 || (initprot != 1 && initprot != 2) || nsects != 1 || flags != 6 then
          throw "__FVMLIB segment header fields invalid"
        let sec := segSects[0]!
        if sec.addr != vmaddr || sec.offset != 0 || sec.reloff != 0 || sec.nreloc != 0 then
          throw "__FVMLIB section must have addr=vmaddr and offset=reloff=nreloc=0"
        if initprot == 1 then
          if sec.sectName != "__const" || sec.flags != 0x21 then
            throw "input __FVMLIB section must be __const with flags=0x21"
        else
          if sec.sectName != "__data" || sec.flags != 0x23 then
            throw "output __FVMLIB section must be __data with flags=0x23"
        surfaceSegs := surfaceSegs.push { prot := initprot, vmSize := vmsize, sect := sec }
      else
        throw s!"unsupported segment name '{segName}'"

    else if cmd == lcSymtab then
      if symtab?.isSome then throw "duplicate LC_SYMTAB command"
      if cmdsize != 24 then throw s!"LC_SYMTAB cmdsize {cmdsize} != 24"
      let symoff ← readU32LE bs (cursor + 8)
      let nsyms ← readU32LE bs (cursor + 12)
      let stroff ← readU32LE bs (cursor + 16)
      let strsize ← readU32LE bs (cursor + 20)
      symtab? := some (symoff, nsyms, stroff, strsize)

    else if cmd == lcBufRef then
      if cmdsize != 32 then throw s!"buffer reference cmdsize {cmdsize} != 32"
      let offField ← readU64LE bs (cursor + 8)
      if offField != 0x18 then throw s!"buffer reference offset field {offField} != 0x18"
      let addr ← readU64LE bs (cursor + 16)
      let shortName ← readFixedAscii bs (cursor + 24) 8 "buffer ref short_name"
      if shortName.utf8ByteSize < 1 || shortName.utf8ByteSize > 7 then
        throw s!"buffer reference short_name '{shortName}' length must be 1..7"
      bufRefs := bufRefs.push { addr := addr, shortName := shortName }

    else if cmd == lcDescriptor then
      if cmdsize < 12 then throw s!"descriptor command at {toHex cursor} too small"
      let subtype ← readU32LE bs (cursor + 8)
      if subtype == descSubtypeTensor then
        if cmdsize < 0xd48 || cmdsize > 0x4000 then
          throw s!"tensor descriptor cmdsize {toHex cmdsize} out of range [0xd48, 0x4000]"
        let dirVal ← readU32LE bs (cursor + 0x14)
        let dir ← match SurfaceDirection.ofUInt32? dirVal with
          | some d => pure d
          | none => throw s!"tensor descriptor invalid direction {dirVal}"
        let n ← readU32LE bs (cursor + 0x28)
        let c ← readU32LE bs (cursor + 0x2c)
        let h ← readU32LE bs (cursor + 0x30)
        let w ← readU32LE bs (cursor + 0x34)
        let batchStride ← readU64LE bs (cursor + 0x50)
        let planeStride ← readU64LE bs (cursor + 0x58)
        let rowStride ← readU64LE bs (cursor + 0x60)
        let shortOff ← readU32LE bs (cursor + 0x7c)
        let (symName, symLen) ← readCString bs (cursor + 0xd3d) (cursor + cmdsize) "tensor symbol_name"
        if symLen < 1 || symLen > 255 then
          throw s!"tensor symbol_name '{symName}' length must be 1..255"
        if shortOff.toNat != 0xd3d + symLen + 1 then
          throw s!"tensor shortoff {toHex shortOff.toNat} != {toHex (0xd3d + symLen + 1)}"
        let (shortName, shortLen) ← readCString bs (cursor + shortOff.toNat) (cursor + cmdsize) "tensor short_name"
        if shortLen < 1 || shortLen > 7 then
          throw s!"tensor short_name '{shortName}' length must be 1..7"
        -- Exact byte-for-byte check (verifies all fixed constants and zeroed reserved bytes)
        let expected ← buildTensorDescriptor dir n c h w batchStride planeStride rowStride symName shortName (some cmdsize)
        let actual ← readSlice bs cursor cmdsize
        checkBytesEqual actual expected s!"tensor descriptor '{shortName}'"
        tensorDescs := tensorDescs.push {
          offset := cursor
          cmdsize := cmdsize
          direction := dir
          n := n
          channels := c
          height := h
          width := w
          batchStride := batchStride
          planeStride := planeStride
          rowStride := rowStride
          symbolName := symName
          shortName := shortName
        }
      else if subtype == descSubtypeProgram then
        if progDescLoc?.isSome then throw "duplicate program descriptor"
        if cmdsize != progDescLinearSize && cmdsize != progDescScratchSize then
          throw s!"program descriptor cmdsize {toHex cmdsize} must be 0x8a0 or 0x8c0"
        progDescLoc? := some (cursor, cmdsize)
      else
        throw s!"unsupported descriptor subtype {subtype}"

    else if cmd == lcIdent then
      if ident?.isSome then throw "duplicate LC_IDENT command"
      if cmdsize <= 8 || cmdsize > 4096 then
        throw s!"LC_IDENT cmdsize {cmdsize} out of range (8, 4096]"
      let s ← readFixedAscii bs (cursor + 8) (cmdsize - 8) "LC_IDENT"
      ident? := some s

    else if cmd == lcNote then
      if !prof.allowNoteVersion then
        throw s!"LC_NOTE is not allowed on {prof.name}"
      if noteVer?.isSome then throw "duplicate LC_NOTE command"
      if cmdsize != 48 then throw s!"LC_NOTE cmdsize {cmdsize} != 48"
      let noteName ← readFixedAscii bs (cursor + 8) 16 "LC_NOTE name"
      let noteOff ← readU64LE bs (cursor + 24)
      let noteSize ← readU64LE bs (cursor + 32)
      let ver ← readU32LE bs (cursor + 40)
      let pad ← readU32LE bs (cursor + 44)
      if noteName != "rt.version" || noteOff != 40 || noteSize != 4 || pad != 0 then
        throw "LC_NOTE payload does not match required rt.version format"
      noteVer? := some ver

    else
      throw s!"unsupported load command {toHex cmd.toNat} at {toHex cursor}"

    cursor := cursor + cmdsize

  if cursor != cmdEnd then
    throw s!"load command cursor {toHex cursor} != expected end {toHex cmdEnd}"

  -- Required segment checks
  if pageZeroCount != 1 then throw "missing required __PAGEZERO segment"
  if textCount != 1 then throw "missing required __TEXT segment"
  let textSec := textSection?.get!
  let constSec := constSection?.get!

  -- Check __TEXT flags against presence of __DATA scratch
  let hasScratch := scratchSection?.isSome
  if hasScratch then
    if textSegFlags != 4 && !(prof.allowZeroScratchTextFlags && textSegFlags == 0) then
      throw s!"{prof.name} __TEXT flags {textSegFlags} invalid when __DATA scratch is present"
  else
    if textSegFlags != 0 then
      throw s!"__TEXT flags must be 0 when no __DATA scratch is present, got {textSegFlags}"

  -- Surface segment count checks
  if surfaceSegs.size < 2 || surfaceSegs.size > 4 then
    throw s!"surface segment count {surfaceSegs.size} must be in 2..4"
  let mut inCount := 0
  let mut outCount := 0
  for s in surfaceSegs do
    if s.prot == 1 then inCount := inCount + 1
    else if s.prot == 2 then outCount := outCount + 1
  if inCount < 1 then
    throw "HWX must contain at least 1 input surface"
  if outCount != 1 then
    throw s!"HWX must contain exactly 1 output surface, got {outCount}"

  -- Total non-guard VM size check
  if totalNonGuardVm > maxTotalNonGuardVmSize then
    throw s!"sum of 16 KiB-rounded non-guard segment sizes ({totalNonGuardVm}) exceeds 512 MiB"

  -- Symbol table range checks
  let (symoff, nsyms, stroff, strsize) ← match symtab? with
    | some st => pure st
    | none => throw "missing required LC_SYMTAB command"
  if nsyms < 1 || nsyms.toNat > maxSymtabSyms then
    throw s!"LC_SYMTAB nsyms {nsyms} out of range 1..{maxSymtabSyms}"
  if strsize < 1 || strsize.toNat > maxStrtabSize then
    throw s!"LC_SYMTAB strsize {strsize} out of range 1..{maxStrtabSize}"
  fileRanges := fileRanges.push { label := "symtab", start := symoff.toNat, size := nsyms.toNat * 16 }
  fileRanges := fileRanges.push { label := "strtab", start := stroff.toNat, size := strsize.toNat }

  -- Global disjointness checks
  checkRangesDisjoint vmRanges "segment VM"
  checkRangesDisjoint sectVmRanges "section VM"
  checkRangesDisjoint fileRanges "file range" (minStart := cmdEnd) (maxEnd := some bs.size)

  -- §5 Match buffer references and tensor descriptors to surfaces
  if bufRefs.size != surfaceSegs.size then
    throw s!"buffer ref count {bufRefs.size} != surface count {surfaceSegs.size}"
  if tensorDescs.size != surfaceSegs.size then
    throw s!"tensor descriptor count {tensorDescs.size} != surface count {surfaceSegs.size}"

  let mut seenBufShorts : Array String := #[]
  for i in [:surfaceSegs.size] do
    let br := bufRefs[i]!
    let ss := surfaceSegs[i]!
    if br.addr != ss.sect.addr then
      throw s!"buffer ref {i} addr {toHex br.addr.toNat} != surface[{i}].addr {toHex ss.sect.addr.toNat}"
    if seenBufShorts.contains br.shortName then
      throw s!"duplicate buffer ref short_name '{br.shortName}'"
    seenBufShorts := seenBufShorts.push br.shortName

  let mut seenTensorShorts : Array String := #[]
  let mut seenTensorSyms : Array String := #[]
  for td in tensorDescs do
    if seenTensorShorts.contains td.shortName then
      throw s!"duplicate tensor descriptor short_name '{td.shortName}'"
    if seenTensorSyms.contains td.symbolName then
      throw s!"duplicate tensor descriptor symbol_name '{td.symbolName}'"
    seenTensorShorts := seenTensorShorts.push td.shortName
    seenTensorSyms := seenTensorSyms.push td.symbolName

  let mut validatedSurfaces : Array ValidatedSurface := #[]
  for i in [:surfaceSegs.size] do
    let br := bufRefs[i]!
    let ss := surfaceSegs[i]!
    let td ← match tensorDescs.find? (fun t => t.shortName == br.shortName) with
      | some t => pure t
      | none => throw s!"no tensor descriptor matching buffer ref '{br.shortName}'"
    if td.direction.toUInt32 != ss.prot then
      throw s!"surface '{br.shortName}': tensor direction {td.direction.toUInt32} != segment prot {ss.prot}"
    if td.n == 0 || td.channels == 0 || td.height == 0 || td.width == 0 then
      throw s!"surface '{br.shortName}': zero tensor dimension"
    let minRow ← mulU64 td.width.toUInt64 2 s!"surface '{br.shortName}' minRow"
    if td.rowStride < minRow then
      throw s!"surface '{br.shortName}': row_stride {td.rowStride} < width * 2 ({minRow})"
    let minPlane ← mulU64 td.height.toUInt64 td.rowStride s!"surface '{br.shortName}' minPlane"
    if td.planeStride < minPlane then
      throw s!"surface '{br.shortName}': plane_stride {td.planeStride} < height * row_stride ({minPlane})"
    let minBatch ← mulU64 td.channels.toUInt64 td.planeStride s!"surface '{br.shortName}' minBatch"
    if td.batchStride < minBatch then
      throw s!"surface '{br.shortName}': batch_stride {td.batchStride} < channels * plane_stride ({minBatch})"
    let span ← mulU64 td.n.toUInt64 td.batchStride s!"surface '{br.shortName}' span"
    if span == 0 || span > ss.sect.size || ss.sect.size > ss.vmSize then
      throw s!"surface '{br.shortName}': span ({span}) > section.size ({ss.sect.size}) or vmsize ({ss.vmSize})"
    validatedSurfaces := validatedSurfaces.push {
      shortName := br.shortName
      symbolName := td.symbolName
      direction := td.direction
      n := td.n
      channels := td.channels
      height := td.height
      width := td.width
      rowStride := td.rowStride
      planeStride := td.planeStride
      batchStride := td.batchStride
      span := span
      sect := ss.sect
      vmSize := ss.vmSize
    }

  -- §6 Validate symbol table & string table
  let firstStrByte ← readU8 bs stroff.toNat
  if firstStrByte != 0 then
    throw s!"string table byte at stroff ({toHex stroff.toNat}) must be 0x00"
  let strEnd := stroff.toNat + strsize.toNat
  let mut symMap : Array (String × UInt8 × UInt8 × UInt64) := #[]
  let mut seenSymNames : Array String := #[]
  for si in [:nsyms.toNat] do
    let eOff := symoff.toNat + si * 16
    let nStrx ← readU32LE bs eOff
    let nType ← readU8 bs (eOff + 4)
    let nSect ← readU8 bs (eOff + 5)
    let nValue ← readU64LE bs (eOff + 8)
    if nStrx >= strsize then
      throw s!"symbol {si}: n_strx {nStrx} >= strsize {strsize}"
    let (name, len) ← readCString bs (stroff.toNat + nStrx.toNat) strEnd s!"symbol {si}"
    if len < 1 || len > 255 then
      throw s!"symbol {si}: string length {len} not in 1..255"
    if seenSymNames.contains name then
      throw s!"duplicate symbol name '{name}'"
    seenSymNames := seenSymNames.push name

    if (nType &&& 0xfe) == 0x0e then
      if nSect == 0 || nSect.toNat > allSections.size then
        throw s!"symbol '{name}': n_sect {nSect} out of range 1..{allSections.size}"
      let sec := allSections[nSect.toNat - 1]!
      let secEnd ← addU64 sec.addr sec.size s!"symbol '{name}' section end"
      let inRange :=
        if prof.allowEndSymbolValue then
          sec.addr <= nValue && nValue <= secEnd
        else
          sec.addr <= nValue && nValue < secEnd
      if !inRange then
        throw s!"symbol '{name}': n_value {toHex nValue.toNat} outside section {nSect}"
    else if prof.allowStabSymbols && (nType == 0x20 || nType == 0x80) then
      if nSect != 0 || nValue != 0 then
        throw s!"{prof.name} stab symbol '{name}' must have n_sect=0 and n_value=0"
    else
      throw s!"symbol '{name}': invalid n_type {toHex nType.toNat}"

    symMap := symMap.push (name, nType, nSect, nValue)

  for surf in validatedSurfaces do
    match symMap.find? (fun (nm, _, _, _) => nm == surf.shortName) with
    | none => throw s!"surface short_name '{surf.shortName}' missing from symbol table"
    | some (_, nType, nSect, nVal) =>
      if (nType &&& 0xfe) != 0x0e || nSect.toNat != surf.sect.ordinal || nVal != surf.sect.addr then
        throw s!"surface symbol '{surf.shortName}' does not match section ordinal {surf.sect.ordinal} and addr {toHex surf.sect.addr.toNat}"

  -- §8 Validate task stream
  let textBytes ← readSlice bs textSec.offset.toNat textSec.size.toNat
  let tsInfo ← validateTaskStream target textBytes

  -- §4 Validate program descriptor exact byte-for-byte
  let (pdOff, pdSize) ← match progDescLoc? with
    | some loc => pure loc
    | none => throw "missing required program descriptor"
  if hasScratch && pdSize != progDescScratchSize then
    throw s!"scratch-backed HWX must use 0x8c0 program descriptor, got {toHex pdSize}"
  if !hasScratch && pdSize != progDescLinearSize then
    throw s!"linear HWX must use 0x8a0 program descriptor, got {toHex pdSize}"
  let pdActual ← readSlice bs pdOff pdSize
  let barTableOff ←
    if prof.allowSlot5BarTable then
      let slot5 ← readU64LE pdActual 0x60
      pure (if slot5 != 0 then 0x60 else 0x70)
    else
      pure 0x70

  let records ← readU32LE pdActual 0x860
  if records.toNat < 1 || records.toNat > tsInfo.recordLimit then
    throw s!"program descriptor records {records} out of range 1..{tsInfo.recordLimit}"

  let totalSurfVm := validatedSurfaces.foldl (fun acc s => acc + s.vmSize.toNat) 0
  let (usable?, workHint) ←
    if hasScratch then
      let u ← readU32LE pdActual 0x858
      let scSec := scratchSection?.get!
      if u == 0 || u.toUInt64 > scSec.size then
        throw s!"program descriptor scratch usable {u} out of range 1..{scSec.size}"
      let wh ← readU32LE pdActual 0x8b0
      pure (some u, wh)
    else
      let wh ← readU32LE pdActual 0x890
      pure (none, wh)
  if workHint.toNat > totalSurfVm then
    throw s!"program descriptor work_hint {workHint} exceeds sum of surface vmsizes {totalSurfVm}"

  let pdExpected ← buildProgramDescriptor
    textSec.addr constSec.addr
    (scratchSection?.map (·.addr))
    (validatedSurfaces.map (·.sect.addr))
    (kernSection?.map (·.addr))
    barTableOff
    (textSec.size / 4).toUInt32
    tsInfo.taskCount.toUInt32
    records
    workHint
    usable?
  checkBytesEqual pdActual pdExpected "program descriptor"

  -- §9 Validate relocations
  let mut relocPatches : Array UInt32 := #[]
  if textSec.nreloc > 0 then
    let kSec ← match kernSection? with
      | some k => pure k
      | none => throw "__TEXT,__text has relocations but __KERN_0,__kern_0 is missing"
    if kSec.size == 0 then
      throw "__KERN_0,__kern_0 size must be > 0 for relocations"
    if textSec.nreloc.toNat > textSec.size.toNat / 4 then
      throw s!"relocation count {textSec.nreloc} exceeds text.size / 4"
    for ri in [:textSec.nreloc.toNat] do
      let rOff := textSec.reloff.toNat + ri * 8
      let patch ← readU32LE bs rOff
      let info ← readU32LE bs (rOff + 4)
      if patch.toNat % 4 != 0 || patch.toNat + 4 > textSec.size.toNat then
        throw s!"relocation {ri}: patch offset {toHex patch.toNat} misaligned or out of bounds"
      if relocPatches.contains patch then
        throw s!"relocation {ri}: duplicate patch offset {toHex patch.toNat}"
      if !tsInfo.isPayloadSite patch.toNat then
        throw s!"relocation {ri}: patch offset {toHex patch.toNat} does not target a register-record payload word"
      let rType := info >>> 24
      let rOrd := (info &&& 0xffffff).toNat
      if rType != 7 || rOrd != kSec.ordinal then
        throw s!"relocation {ri}: info {toHex info.toNat} does not match type=7, ordinal={kSec.ordinal}"
      let addend ← readU32LE textBytes patch.toNat
      if addend.toUInt64 >= kSec.size then
        throw s!"relocation {ri}: addend {toHex addend.toNat} >= kernel size {toHex kSec.size.toNat}"
      let _ ← addU64 kSec.addr addend.toUInt64 s!"relocation {ri} target"
      relocPatches := relocPatches.push patch

  pure {
    target := target
    textSection := textSec
    constSection := constSec
    kernelSection := kernSection?
    scratchSection := scratchSection?
    scratchVmSize := scratchVmSize?
    scratchUsable := usable?
    surfaces := validatedSurfaces
    programDescOffset := pdOff
    programDescSize := pdSize
    programDescBytes := pdActual
    barTableOffset := barTableOff
    taskStream := tsInfo
    records := records
    workHint := workHint
    relocCount := textSec.nreloc.toNat
    relocPatchOffsets := relocPatches
    ident := ident?
    noteVersion := noteVer?
  }

end Vane.Hwx
