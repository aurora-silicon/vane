import Vane.Binary
import Vane.TaskStream

namespace Vane.Register

open Vane.Binary
open Vane.TaskStream

/-- A single contiguous bitfield `[lo + width - 1 : lo]` within a 32-bit word. -/
structure BitField where
  name : String
  lo : Nat
  width : Nat
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Bit-level schema for a 32-bit machine word or register. -/
structure WordLayout where
  name : String
  fixedMask : UInt32 := 0
  fixedValue : UInt32 := 0
  fields : Array BitField := #[]
  deriving Repr, BEq, Inhabited

/-- Compute the 32-bit mask of a valid bitfield (`1 <= width`, `lo + width <= 32`). -/
def BitField.maskNat (f : BitField) : Nat :=
  (((1 : Nat) <<< f.width) - 1) <<< f.lo

/-- Compute the 32-bit `UInt32` mask of a valid bitfield. -/
def BitField.mask (f : BitField) : UInt32 :=
  f.maskNat.toUInt32

/-- Recursive field-list validator for `WordLayout` (kernel-reducible for formal layout proofs). -/
private def validateWordLayoutFields
    (layoutName : String) (fixedMask : UInt32) :
    List BitField → UInt32 → List String → Except String UInt32
  | [], usedMask, _ => .ok usedMask
  | f :: rest, usedMask, seenNames =>
    if f.width == 0 then
      .error s!"layout '{layoutName}.{f.name}': width must be >= 1"
    else if f.lo + f.width > 32 then
      .error s!"layout '{layoutName}.{f.name}': bit range [{f.lo + f.width - 1}:{f.lo}] exceeds 32 bits"
    else if seenNames.any (fun nm => decide (nm = f.name)) then
      .error s!"layout '{layoutName}': duplicate field name '{f.name}'"
    else
      let m := f.mask
      if (m &&& fixedMask) != 0 then
        .error s!"layout '{layoutName}.{f.name}': overlaps fixed constant mask {toHex fixedMask.toNat}"
      else if (m &&& usedMask) != 0 then
        .error s!"layout '{layoutName}.{f.name}': overlaps another field"
      else
        validateWordLayoutFields layoutName fixedMask rest (usedMask ||| m) (f.name :: seenNames)

/-- Validate that a `WordLayout` has well-formed, non-overlapping fields within 32 bits
    and that `fixedValue` has no bits outside `fixedMask`. -/
def validateWordLayout (layout : WordLayout) : Except String UInt32 :=
  if (layout.fixedValue &&& (~~~layout.fixedMask)) != 0 then
    .error s!"layout '{layout.name}': fixedValue {toHex layout.fixedValue.toNat} sets bits outside fixedMask {toHex layout.fixedMask.toNat}"
  else
    validateWordLayoutFields layout.name layout.fixedMask layout.fields.toList layout.fixedMask []

/-- Width-checked bitfield encoder: validates layout disjointness, checks that every field
    is supplied exactly once and fits strictly in `< 2^width`, and combines with fixed bits. -/
def encodeWord (layout : WordLayout) (values : Array (String × UInt32)) : Except String UInt32 := do
  let _ ← validateWordLayout layout
  let mut seenVals : Array String := #[]
  for (fname, _) in values do
    if seenVals.contains fname then
      throw s!"layout '{layout.name}': duplicate value for field '{fname}'"
    if !layout.fields.any (fun f => f.name == fname) then
      throw s!"layout '{layout.name}': unknown field '{fname}'"
    seenVals := seenVals.push fname

  let mut word : UInt32 := layout.fixedValue
  for f in layout.fields do
    let some (_, val) := values.find? (fun (nm, _) => nm == f.name)
      | throw s!"layout '{layout.name}': missing required field '{f.name}'"
    let limit : Nat := (1 : Nat) <<< f.width
    if val.toNat >= limit then
      throw s!"layout '{layout.name}.{f.name}': value {val} ({toHex val.toNat}) exceeds {f.width}-bit field max {limit - 1}"
    word := word ||| (val <<< f.lo.toUInt32)
  pure word

/-- Extract a bitfield value from a 32-bit word without bounds checking the layout. -/
def decodeField (word : UInt32) (f : BitField) : UInt32 :=
  let rawMask : UInt32 := (((1 : Nat) <<< f.width) - 1).toUInt32
  (word >>> f.lo.toUInt32) &&& rawMask

/-- Check that `word` matches `layout`'s fixed bits and has zero in all unused bits. -/
def validateWordBits (layout : WordLayout) (word : UInt32) : Except String Unit := do
  let usedMask ← validateWordLayout layout
  if (word &&& layout.fixedMask) != layout.fixedValue then
    throw s!"layout '{layout.name}': word {toHex word.toNat} violates fixed bits (got {toHex (word &&& layout.fixedMask).toNat}, expected {toHex layout.fixedValue.toNat})"
  let unused := word &&& (~~~usedMask)
  if unused != 0 then
    throw s!"layout '{layout.name}': word {toHex word.toNat} has non-zero reserved/unused bits {toHex unused.toNat}"
  pure ()

/-- Validate fixed/unused bits of `word` against `layout` and extract the named field. -/
def decodeFieldChecked (layout : WordLayout) (word : UInt32) (fieldName : String) : Except String UInt32 := do
  validateWordBits layout word
  match layout.fields.find? (fun f => f.name == fieldName) with
  | some f => pure (decodeField word f)
  | none => throw s!"layout '{layout.name}': unknown field '{fieldName}'"

-- ============================================================================
-- Task header layouts (9-word header for NE 1x1 conv)
-- ============================================================================

/-- Task header word 0: `task_id[15:0]`, `task_words[26:16]`, reserved `[31:27] = 0`. -/
def taskHeader0Layout : WordLayout := {
  name := "taskHeader[0]"
  fields := #[
    { name := "taskId", lo := 0, width := 16 },
    { name := "taskWords", lo := 16, width := 11 }
  ]
}

/-- Task header word 8: `flags[3:0] = 0x9`, `ene[18:16] = logNE + 1`, other bits 0. -/
def taskHeader8Layout : WordLayout := {
  name := "taskHeader[8]"
  fixedMask := 0x0000000f
  fixedValue := 0x00000009
  fields := #[
    { name := "ene", lo := 16, width := 3 }
  ]
}

/-- Fixed 9-word task header constants. -/
def taskHeaderWord2Const : UInt32 := 0x04000068
def taskHeaderWord4Const : UInt32 := 0x00fff868

/-- Encode the 9-word NE task header. Checks `logNE <= 4` and `9 <= taskWords <= 0x7ff`. -/
def encodeTaskHeader9 (taskId taskWords logNE : Nat) : Except String (Array UInt32) := do
  if logNE > 4 then
    throw s!"task header: logNE {logNE} exceeds maximum 4"
  if taskWords < 9 || taskWords > 0x7ff then
    throw s!"task header: taskWords {taskWords} out of range 9..{0x7ff}"
  let w0 ← encodeWord taskHeader0Layout #[
    ("taskId", taskId.toUInt32),
    ("taskWords", taskWords.toUInt32)
  ]
  let ene := logNE + 1
  let w8 ← encodeWord taskHeader8Layout #[("ene", ene.toUInt32)]
  pure #[w0, 0, taskHeaderWord2Const, 0, taskHeaderWord4Const, 0, 0, 0, w8]

-- ============================================================================
-- Machine register addresses (byte offsets in ANE register space)
-- ============================================================================

-- Common block (0x0000..0x0058)
def regWin : Nat := 0x0004
def regHin : Nat := 0x0008
def regCin : Nat := 0x000c
def regWout : Nat := 0x0014
def regHout : Nat := 0x0018
def regCout : Nat := 0x001c
def regConvCfg : Nat := 0x0028
def regDwGroups : Nat := 0x0030
def regTH : Nat := 0x0034
def regNeCommon : Nat := 0x003c
def regNeTile : Nat := 0x0040
def regTargetCfg : Nat := 0x004c

-- L2 staging block (0x4100..0x4160)
def regSrc1Cfg : Nat := 0x4104
def regSrc2Cfg : Nat := 0x4108
def regSrc1Base : Nat := 0x4110
def regSrc1ChanStride : Nat := 0x4114
def regSrc1RowStride : Nat := 0x4118
def regSrc1RowBytesA : Nat := 0x411c
def regSrc1RowBytesB : Nat := 0x4120
def regResCfg : Nat := 0x4148
def regResBase : Nat := 0x414c
def regResChanStride : Nat := 0x4150
def regResRowStride : Nat := 0x4154
def regResRowBytesA : Nat := 0x4158
def regResRowBytesB : Nat := 0x415c

-- Neural Engine block (0x4900..0x4914)
def regNeCfg : Nat := 0x4900
def regNeOp : Nat := 0x4904
def regScalarBias : Nat := 0x490c
def regNeScale : Nat := 0x4910

-- Tile DMA (Source) block (0x4d00..0x4d6c)
def regSrc1Dep : Nat := 0x4d00
def regSrc1DmaCfg : Nat := 0x4d08
def regSrc1BaseLo : Nat := 0x4d10
def regSrc1BaseHi : Nat := 0x4d14
def regSrc1DmaRowStride : Nat := 0x4d18
def regSrc1DmaPlaneStride : Nat := 0x4d1c
def regSrc1Fmt : Nat := 0x4d68

-- Result DMA block (0x5100..0x513c)
def regResDep : Nat := 0x5100
def regResBaseLo : Nat := 0x5108
def regResBaseHi : Nat := 0x510c
def regResDmaRowStride : Nat := 0x5110
def regResDmaPlaneStride : Nat := 0x5114
def regResFmt : Nat := 0x5138

-- Kernel DMA and chunk table block (0x5500..0x55dc)
def regKernelStage : Nat := 0x5500
def regKernelBaseLo : Nat := 0x5510
def regKernelBaseHi : Nat := 0x5514
def regChunkSlot (s : Nat) : Nat := 0x5520 + 4 * s
def regChunkOffset (s : Nat) : Nat := 0x5560 + 4 * s
def regChunkSize (s : Nat) : Nat := 0x55a0 + 4 * s

-- Output / cache stage block (0x5900..0x592c)
def regOut0 : Nat := 0x5900
def regOut1 : Nat := 0x5904
def regOut2 : Nat := 0x5908
def regOut3 : Nat := 0x5918
def regOut4 : Nat := 0x5928
def regOut5 : Nat := 0x592c

-- ============================================================================
-- Machine register bitfield layouts
-- ============================================================================

def dim16Layout (name : String) : WordLayout := {
  name := name
  fields := #[{ name := "val", lo := 0, width := 16 }]
}

def u32Layout (name : String) : WordLayout := {
  name := name
  fields := #[{ name := "val", lo := 0, width := 32 }]
}

/-- `0x0028` convCfg: `kw[5:0]`, `kh[12:6]`, `sx[14:13]`, `sy[16:15]`,
    `pl[21:17]`, `pt[27:22]`, `hx[29:28]`, `vy[31:30]`. -/
def convCfgLayout : WordLayout := {
  name := "convCfg"
  fields := #[
    { name := "kw", lo := 0, width := 6 },
    { name := "kh", lo := 6, width := 7 },
    { name := "sx", lo := 13, width := 2 },
    { name := "sy", lo := 15, width := 2 },
    { name := "pl", lo := 17, width := 5 },
    { name := "pt", lo := 22, width := 6 },
    { name := "hx", lo := 28, width := 2 },
    { name := "vy", lo := 30, width := 2 }
  ]
}

/-- `0x0030` dwGroups: `fixed[15:0] = 0x4000`, `perGroupIn[31:16]`. -/
def dwGroupsLayout : WordLayout := {
  name := "dwGroups"
  fixedMask := 0x0000ffff
  fixedValue := 0x00004000
  fields := #[
    { name := "perGroupIn", lo := 16, width := 16 }
  ]
}

/-- `0x003c` neCommon: `kind[7:0]`, `shPref[10:8]`, `shMin[14:12]`, `shMax[18:16]`,
    `F[21:19]`, `wide3x3[27:27]` (all other bits must be 0). -/
def neCommonLayout : WordLayout := {
  name := "neCommon"
  fields := #[
    { name := "kind", lo := 0, width := 8 },
    { name := "shPref", lo := 8, width := 3 },
    { name := "shMin", lo := 12, width := 3 },
    { name := "shMax", lo := 16, width := 3 },
    { name := "F", lo := 19, width := 3 },
    { name := "wide3x3", lo := 27, width := 1 }
  ]
}

/-- `0x0040` neTile: `ocg[2:0]`, `fat[3:3]`, `WU[5:4]`, bits `[31:6]` must be 0. -/
def neTileLayout : WordLayout := {
  name := "neTile"
  fields := #[
    { name := "ocg", lo := 0, width := 3 },
    { name := "fat", lo := 3, width := 1 },
    { name := "WU", lo := 4, width := 2 }
  ]
}

/-- `0x4104` src1Cfg: `mode[1:0]`, `P[11:8]`, fixed flags `0x00500070`. -/
def src1CfgLayout : WordLayout := {
  name := "src1Cfg"
  fixedMask := 0xfff0f0fc
  fixedValue := 0x00500070
  fields := #[
    { name := "mode", lo := 0, width := 2 },
    { name := "P", lo := 8, width := 4 }
  ]
}

/-- `0x4148` resCfg: `mode[1:0]`, `direct[3:3]`, `P[11:8]`, fixed flags `0x00500070`. -/
def resCfgLayout : WordLayout := {
  name := "resCfg"
  fixedMask := 0xfff0f0f4
  fixedValue := 0x00500070
  fields := #[
    { name := "mode", lo := 0, width := 2 },
    { name := "direct", lo := 3, width := 1 },
    { name := "P", lo := 8, width := 4 }
  ]
}

/-- `0x4904` neOp: base `0x00103c00`, `biasInKernel[4:4]`, `relu[16:16]`. -/
def neOpLayout : WordLayout := {
  name := "neOp"
  fixedMask := 0xfffeffef
  fixedValue := 0x00103c00
  fields := #[
    { name := "biasInKernel", lo := 4, width := 1 },
    { name := "relu", lo := 16, width := 1 }
  ]
}

/-- `0x490c` scalarBias: IEEE fp16 bits in `[15:0]`. -/
def scalarBiasLayout : WordLayout := dim16Layout "scalarBias"

/-- `0x4910` neScale: IEEE fp16 bits in `[15:0]`. -/
def neScaleLayout : WordLayout := dim16Layout "neScale"

/-- `0x4d00` src1Dep: `code[7:0]`, `task[23:16]`. -/
def src1DepLayout : WordLayout := {
  name := "src1Dep"
  fields := #[
    { name := "code", lo := 0, width := 8 },
    { name := "task", lo := 16, width := 8 }
  ]
}

/-- `0x4d68` src1Fmt: `format[23:0]`, `P[31:24]`. -/
def dmaFmtLayout (name : String) : WordLayout := {
  name := name
  fields := #[
    { name := "format", lo := 0, width := 24 },
    { name := "P", lo := 24, width := 8 }
  ]
}

/-- `0x5100` resDep: `code[7:0]`, `task[23:16]`, `direct[24:24]`. -/
def resDepLayout : WordLayout := {
  name := "resDep"
  fields := #[
    { name := "code", lo := 0, width := 8 },
    { name := "task", lo := 16, width := 8 },
    { name := "direct", lo := 24, width := 1 }
  ]
}

/-- `0x5500` kernelStage: fixed `0x000002c0`, `sparse[5:5]`, `stage[23:16]` (`0x000102c0 | (sparse << 5)`). -/
def kernelStageLayout : WordLayout := {
  name := "kernelStage"
  fixedMask := 0xff00ffdf
  fixedValue := 0x000002c0
  fields := #[
    { name := "sparse", lo := 5, width := 1 },
    { name := "stage", lo := 16, width := 8 }
  ]
}

/-- `0x5520 + 4*s` chunkSlot[s]: `enable[0:0]`, `on20[5:5]`, `task[23:16]`. -/
def chunkSlotLayout : WordLayout := {
  name := "chunkSlot"
  fields := #[
    { name := "enable", lo := 0, width := 1 },
    { name := "on20", lo := 5, width := 1 },
    { name := "task", lo := 16, width := 8 }
  ]
}

/-- `0x5908` out2: `form[15:0]`, `task[23:16]`. -/
def out2Layout : WordLayout := {
  name := "out2"
  fields := #[
    { name := "form", lo := 0, width := 16 },
    { name := "task", lo := 16, width := 8 }
  ]
}

-- Target-dependent fixed constants
def targetCfgValue : Target → UInt32
  | .h16 => 0x00100000
  | .h17 => 0x00200000

def neCfgValue : Target → UInt32
  | .h16 => 0x00000082
  | .h17 => 0x10000082

/-- `0x4900` neCfg: target base (`0x00000082` on H16, `0x10000082` on H17), `sparse[8:8]`. -/
def neCfgLayout (target : Target) : WordLayout := {
  name := "neCfg"
  fixedMask := 0xfffffeff
  fixedValue := neCfgValue target
  fields := #[
    { name := "sparse", lo := 8, width := 1 }
  ]
}

def resDepCode : Target → UInt32
  | .h16 => 0x31
  | .h17 => 0x21

-- Fixed register constants
def src2CfgValue : UInt32 := 0x00500130
def src1DmaCfgValue : UInt32 := 0x000000ce
def fp16OneBits : UInt32 := 0x3c00
def out0Value : UInt32 := 0x00000012
def out1Value : UInt32 := 0x00a000a0
def out3Value : UInt32 := 0x00000000
def out4Value : UInt32 := 0x007f0000
def out5Value : UInt32 := 0x20010701

-- ============================================================================
-- Register record packer (plain writes + BAR-tagged sequential records)
-- ============================================================================

/-- A single plain (non-BAR) register write at a 4-byte aligned byte address. -/
structure RegWrite where
  byteAddr : Nat
  value : UInt32
  deriving Repr, BEq, Inhabited

/-- A sequential BAR-tagged register record at a 4-byte aligned base byte address. -/
structure BarRegWrite where
  baseByteAddr : Nat
  barSlot : UInt8
  values : Array UInt32
  deriving Repr, BEq, Inhabited

/-- Pack plain register writes (sorted ascending by byte address, checking alignment,
    uniqueness, and the 9 allowed register blocks) into valid sequential/bitmask `RegRecord`s. -/
def packPlainWrites (writes : Array RegWrite) : Except String (Array RegRecord) := do
  -- Validate every address first
  for w in writes do
    if w.byteAddr % 4 != 0 then
      throw s!"register address {toHex w.byteAddr} is not 4-byte aligned"
    let wordIdx := w.byteAddr / 4
    if wordIdx > 0x7fff then
      throw s!"register word index {toHex wordIdx} (byte {toHex w.byteAddr}) exceeds 15-bit range"
    if !isAllowedRegWord wordIdx then
      throw s!"register address {toHex w.byteAddr} (word {toHex wordIdx}) outside allowed blocks"

  let sorted := writes.qsort fun a b => a.byteAddr < b.byteAddr
  for i in [1 : sorted.size] do
    if sorted[i - 1]!.byteAddr == sorted[i]!.byteAddr then
      throw s!"duplicate register write at byte address {toHex sorted[i]!.byteAddr}"

  let mut out : Array RegRecord := #[]
  let mut idx := 0
  while idx < sorted.size do
    let baseWord := sorted[idx]!.byteAddr / 4

    -- Count strictly contiguous run starting at idx (up to 64 words)
    let mut seqLen := 1
    while idx + seqLen < sorted.size && seqLen < 64 do
      let nextWord := sorted[idx + seqLen]!.byteAddr / 4
      if nextWord == baseWord + seqLen then
        seqLen := seqLen + 1
      else
        break

    -- Count words covered by a bitmask record at baseWord (span [baseWord, baseWord + 16])
    let mut bmLen := 1
    while idx + bmLen < sorted.size do
      let nextWord := sorted[idx + bmLen]!.byteAddr / 4
      if nextWord <= baseWord + 16 then
        bmLen := bmLen + 1
      else
        break

    if bmLen > seqLen then
      let mut mask : UInt16 := 0
      let mut vals : Array UInt32 := #[]
      vals := vals.push sorted[idx]!.value
      for k in [1 : bmLen] do
        let w := sorted[idx + k]!.byteAddr / 4
        let bit := w - baseWord - 1
        mask := mask ||| ((1 : UInt16) <<< bit.toUInt16)
        vals := vals.push sorted[idx + k]!.value
      let regRec := RegRecord.bitmask baseWord.toUInt16 mask vals
      let _ ← encodeRegRecord regRec
      out := out.push regRec
      idx := idx + bmLen
    else
      let mut vals : Array UInt32 := #[]
      for k in [:seqLen] do
        vals := vals.push sorted[idx + k]!.value
      let regRec := RegRecord.sequential baseWord.toUInt16 none vals
      let _ ← encodeRegRecord regRec
      out := out.push regRec
      idx := idx + seqLen

  pure out

/-- Pack plain writes followed by explicit BAR-tagged sequential records in caller order,
    ensuring no byte address is written by both plain and BAR records. -/
def packTaskRecords (plain : Array RegWrite) (barRecs : Array BarRegWrite) : Except String (Array RegRecord) := do
  let mut barAddrs : Array Nat := #[]
  let mut barEncoded : Array RegRecord := #[]
  for br in barRecs do
    if br.baseByteAddr % 4 != 0 then
      throw s!"BAR record base address {toHex br.baseByteAddr} is not 4-byte aligned"
    if br.values.isEmpty || br.values.size > 64 then
      throw s!"BAR record at {toHex br.baseByteAddr}: value count {br.values.size} must be in 1..64"
    let baseWord := br.baseByteAddr / 4
    if baseWord > 0x7fff then
      throw s!"BAR record base word {toHex baseWord} exceeds 15-bit range"
    for i in [:br.values.size] do
      let addr := br.baseByteAddr + 4 * i
      if barAddrs.contains addr then
        throw s!"duplicate BAR register write at byte address {toHex addr}"
      barAddrs := barAddrs.push addr
    let regRec := RegRecord.sequential baseWord.toUInt16 (some br.barSlot) br.values
    let _ ← encodeRegRecord regRec
    barEncoded := barEncoded.push regRec

  for pw in plain do
    if barAddrs.contains pw.byteAddr then
      throw s!"register address {toHex pw.byteAddr} written by both plain and BAR records"

  let plainRecs ← packPlainWrites plain
  pure (plainRecs ++ barEncoded)

/-- Count total 32-bit words consumed by an array of register records (header + payload). -/
def recordWordsCount (recs : Array RegRecord) : Nat :=
  recs.foldl (fun acc r =>
    match r with
    | .sequential _ _ vals => acc + 1 + vals.size
    | .bitmask _ _ vals => acc + 1 + vals.size
  ) 0

-- ============================================================================
-- Task stream decoder (raw `__TEXT,__text` bytes -> decoded register writes)
-- ============================================================================

/-- A single decoded register write from a `__TEXT,__text` task stream. -/
structure DecodedWrite where
  byteAddr : Nat
  value : UInt32
  barSlot : Option UInt8
  payloadOffset : Nat
  deriving Repr, BEq, Inhabited

/-- A single decoded task from `__TEXT,__text`. -/
structure DecodedTask where
  taskIndex : Nat
  startOffset : Nat
  header : Array UInt32
  writes : Array DecodedWrite
  deriving Repr, BEq, Inhabited

/-- Decode all tasks and register writes from raw `__TEXT,__text` bytes.
    Runs full structural `validateTaskStream` before extracting headers and writes. -/
def decodeTaskStream (target : Target) (bs : ByteArray) : Except String (Array DecodedTask) := do
  let _ ← validateTaskStream target bs
  let mut cursor := prologueFrameBytes
  let mut tasks : Array DecodedTask := #[]
  let mut taskIdx := 0

  while cursor < bs.size do
    let w0 ← readU32LE bs cursor
    let words := ((w0 >>> 16) &&& 0x7ff).toNat
    if words == 0 then
      cursor := cursor + taskFrameAlignBytes
    else
      let taskEnd := cursor + 4 * words
      let w8 ← readU32LE bs (cursor + 32)
      let headerWords := if (w8 &&& 3) == 3 then 10 else 9
      let mut hdr : Array UInt32 := #[]
      for hi in [:headerWords] do
        hdr := hdr.push (← readU32LE bs (cursor + 4 * hi))

      let mut writes : Array DecodedWrite := #[]
      let mut recPos := cursor + 4 * headerWords
      while recPos < taskEnd do
        let h ← readU32LE bs recPos
        recPos := recPos + 4
        let base := (h &&& 0x7fff).toNat
        let isBitmask := ((h >>> 31) &&& 1) == 1
        if isBitmask then
          let mask := ((h >>> 15) &&& 0xffff).toNat
          let mut addrs : Array Nat := #[base * 4]
          for i in [:16] do
            if ((mask >>> i) &&& 1) == 1 then
              addrs := addrs.push ((base + i + 1) * 4)
          for i in [:addrs.size] do
            let pOff := recPos + 4 * i
            let v ← readU32LE bs pOff
            writes := writes.push {
              byteAddr := addrs[i]!
              value := v
              barSlot := none
              payloadOffset := pOff
            }
          recPos := recPos + 4 * addrs.size
        else
          let cnt := (((h >>> 15) &&& 0x3f).toNat) + 1
          let slot? : Option UInt8 :=
            if ((h >>> 29) &&& 1) == 1 then
              some ((h >>> 23) &&& 0x3f).toUInt8
            else
              none
          for i in [:cnt] do
            let pOff := recPos + 4 * i
            let v ← readU32LE bs pOff
            writes := writes.push {
              byteAddr := (base + i) * 4
              value := v
              barSlot := slot?
              payloadOffset := pOff
            }
          recPos := recPos + 4 * cnt

      tasks := tasks.push {
        taskIndex := taskIdx
        startOffset := cursor
        header := hdr
        writes := writes
      }
      taskIdx := taskIdx + 1
      if taskEnd < bs.size then
        cursor := alignUp taskEnd taskFrameAlignBytes
      else
        cursor := taskEnd

  pure tasks

end Vane.Register
