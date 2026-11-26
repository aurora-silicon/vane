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
def validateWordLayoutFields
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

/-- Validate that all supplied `(fieldName, value)` pairs have unique, known field names. -/
def validateEncodeValues
    (layoutName : String) (fields : List BitField) :
    List (String × UInt32) → List String → Except String Unit
  | [], _ => .ok ()
  | (fname, _) :: rest, seenVals =>
    if seenVals.any (fun nm => decide (nm = fname)) then
      .error s!"layout '{layoutName}': duplicate value for field '{fname}'"
    else if !fields.any (fun f => decide (f.name = fname)) then
      .error s!"layout '{layoutName}': unknown field '{fname}'"
    else
      validateEncodeValues layoutName fields rest (fname :: seenVals)

/-- Recursively width-check and OR all fields of a `WordLayout` into `word`. -/
def encodeWordFields
    (layoutName : String) (values : List (String × UInt32)) :
    List BitField → UInt32 → Except String UInt32
  | [], word => .ok word
  | f :: rest, word =>
    match values.find? (fun p => decide (p.1 = f.name)) with
    | none => .error s!"layout '{layoutName}': missing required field '{f.name}'"
    | some (_, val) =>
      let limit : Nat := (1 : Nat) <<< f.width
      if val.toNat >= limit then
        .error s!"layout '{layoutName}.{f.name}': value {val} ({toHex val.toNat}) exceeds {f.width}-bit field max {limit - 1}"
      else
        encodeWordFields layoutName values rest (word ||| (val <<< f.lo.toUInt32))

/-- Width-checked bitfield encoder: validates layout disjointness, checks that every field
    is supplied exactly once and fits strictly in `< 2^width`, and combines with fixed bits. -/
def encodeWord (layout : WordLayout) (values : Array (String × UInt32)) : Except String UInt32 :=
  validateWordLayout layout >>= fun _ =>
  validateEncodeValues layout.name layout.fields.toList values.toList [] >>= fun () =>
  encodeWordFields layout.name values.toList layout.fields.toList layout.fixedValue

/-- Extract a bitfield value from a 32-bit word without bounds checking the layout. -/
def decodeField (word : UInt32) (f : BitField) : UInt32 :=
  let rawMask : UInt32 := (((1 : Nat) <<< f.width) - 1).toUInt32
  (word >>> f.lo.toUInt32) &&& rawMask

/-- Look up a named bitfield in a `WordLayout` field list. -/
def findBitField? (fields : List BitField) (fieldName : String) : Option BitField :=
  fields.find? (fun f => decide (f.name = fieldName))

/-- Check that `word` matches `layout`'s fixed bits and has zero in all unused bits. -/
def validateWordBits (layout : WordLayout) (word : UInt32) : Except String Unit :=
  validateWordLayout layout >>= fun usedMask =>
  if (word &&& layout.fixedMask) != layout.fixedValue then
    .error s!"layout '{layout.name}': word {toHex word.toNat} violates fixed bits (got {toHex (word &&& layout.fixedMask).toNat}, expected {toHex layout.fixedValue.toNat})"
  else
    let unused := word &&& (~~~usedMask)
    if unused != 0 then
      .error s!"layout '{layout.name}': word {toHex word.toNat} has non-zero reserved/unused bits {toHex unused.toNat}"
    else
      .ok ()

/-- Validate fixed/unused bits of `word` against `layout` and extract the named field. -/
def decodeFieldChecked (layout : WordLayout) (word : UInt32) (fieldName : String) : Except String UInt32 :=
  validateWordBits layout word >>= fun () =>
  match findBitField? layout.fields.toList fieldName with
  | some f => .ok (decodeField word f)
  | none => .error s!"layout '{layout.name}': unknown field '{fieldName}'"

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

/-- Task header word 2: `firstFlags[7:0]` (`0x68` when first), `lastFlag[26:26]` (`1` when last). -/
def taskHeader2Layout : WordLayout := {
  name := "taskHeader[2]"
  fields := #[
    { name := "firstFlags", lo := 0, width := 8 },
    { name := "lastFlag", lo := 26, width := 1 }
  ]
}

/-- Task header word 8 (single-task): `flags[3:0] = 0x9`, `ene[18:16] = 0` (PE) or `logNE + 1` (NE), other bits 0. -/
def taskHeader8Layout : WordLayout := {
  name := "taskHeader[8]"
  fixedMask := 0x0000000f
  fixedValue := 0x00000009
  fields := #[
    { name := "ene", lo := 16, width := 3 }
  ]
}

/-- Task header word 8 (multi-task): `active[0:0] = 1`, `isLast[3:3]`, `ene[18:16]`, other bits 0. -/
def taskHeader8MultiLayout : WordLayout := {
  name := "taskHeader[8]Multi"
  fields := #[
    { name := "active", lo := 0, width := 1 },
    { name := "isLast", lo := 3, width := 1 },
    { name := "ene", lo := 16, width := 3 }
  ]
}

/-- Task header word 8 (extended 10-word header): `active[0:0] = 1`, `hasExtHdr[1:1]`,
    `isLast[3:3]`, `ene[18:16]`, other bits 0. -/
def taskHeader8ExtLayout : WordLayout := {
  name := "taskHeader[8]Ext"
  fields := #[
    { name := "active", lo := 0, width := 1 },
    { name := "hasExtHdr", lo := 1, width := 1 },
    { name := "isLast", lo := 3, width := 1 },
    { name := "ene", lo := 16, width := 3 }
  ]
}

/-- Bit 1 (`0x2`) of task header word 8 indicating a 10-word extended task header. -/
def taskHeader8ExtBit : UInt32 := 0x00000002

/-- Named constants for 9-word task header words 2 and 4. -/
def taskHeader2FirstFlags : UInt32 := 0x68
def taskHeader4FirstBit : UInt32 := 0x00000008
def taskHeaderWord2Const : UInt32 := ((1 : UInt32) <<< 26) ||| taskHeader2FirstFlags
def taskHeaderWord4CopyConst : UInt32 := taskHeader2FirstFlags

/-- Target- and position-specific `header[4]` word for multi-task streams. -/
def taskHeaderWord4Multi (target : Target) (isFirst isLast : Bool) : UInt32 :=
  let prof := target.profile
  let firstBit := if isFirst then taskHeader4FirstBit else 0
  let edgeBits := if isFirst || isLast then prof.taskHeader4EdgeBits else 0
  prof.taskHeader4Base ||| firstBit ||| edgeBits

def taskHeaderWord4Const : UInt32 := taskHeaderWord4Multi .h17 true true

/-- 64 KiB L2 block size for `header[7]` live-L2 bitmasks (32 blocks cover 2 MiB). -/
def l2MaskBlockBytes : Nat := 0x10000

/-- Compute the 32-bit `header[7]` live-L2 64 KiB block bitmask for a byte range `[lo, hi)`. -/
def l2BlockMask (lo hi : Nat) : UInt32 :=
  if lo >= hi then 0
  else
    let b0 := lo / l2MaskBlockBytes
    let b1 := min 31 ((hi - 1) / l2MaskBlockBytes)
    Id.run do
      let mut mask : UInt32 := 0
      for b in [b0 : b1 + 1] do
        if b < 32 then
          mask := mask ||| ((1 : UInt32) <<< b.toUInt32)
      return mask

/-- Encode a 9-word or 10-word extended task header for single-task or multi-task streams.
    When `extProducerTask = some prodId`, sets `header[8]` bit 1 (`hasExtHdr = 1`) and appends
    `header[9] = prodId` (requiring `taskWords >= 10`). -/
def encodeTaskHeaderMulti
    (target : Target)
    (taskId taskWords ene : Nat)
    (isFirst isLast : Bool)
    (liveL2Mask : UInt32 := 0)
    (header4Override : Option UInt32 := none)
    (extProducerTask : Option Nat := none) : Except String (Array UInt32) := do
  if taskId > 0xffff then
    throw s!"task header: taskId {taskId} exceeds 16-bit maximum 65535"
  if ene > 5 then
    throw s!"task header: ene {ene} exceeds maximum 5"
  let minWords := if extProducerTask.isSome then 10 else 9
  if taskWords < minWords || taskWords > 0x7ff then
    throw s!"task header: taskWords {taskWords} out of range {minWords}..{0x7ff}"
  let taskIdU32 ← natToU32Checked taskId "task header taskId"
  let taskWordsU32 ← natToU32Checked taskWords "task header taskWords"
  let eneU32 ← natToU32Checked ene "task header ene"
  let w0 ← encodeWord taskHeader0Layout #[
    ("taskId", taskIdU32),
    ("taskWords", taskWordsU32)
  ]
  let w2 ← encodeWord taskHeader2Layout #[
    ("firstFlags", if isFirst then taskHeader2FirstFlags else 0),
    ("lastFlag", if isLast then 1 else 0)
  ]
  let w4 := header4Override.getD (taskHeaderWord4Multi target isFirst isLast)
  let w7 := if isLast then 0 else liveL2Mask
  match extProducerTask with
  | none =>
    let w8 ← encodeWord taskHeader8MultiLayout #[
      ("active", 1),
      ("isLast", if isLast then 1 else 0),
      ("ene", eneU32)
    ]
    pure #[w0, 0, w2, 0, w4, 0, 0, w7, w8]
  | some prodTask =>
    let prodU32 ← natToU32Checked prodTask "task header extProducerTask"
    let w8 ← encodeWord taskHeader8ExtLayout #[
      ("active", 1),
      ("hasExtHdr", 1),
      ("isLast", if isLast then 1 else 0),
      ("ene", eneU32)
    ]
    pure #[w0, 0, w2, 0, w4, 0, 0, w7, w8, prodU32]

/-- Encode a 9-word task header for single-task or multi-task streams.
    Sets `header[2]`, `header[4]`, `header[7]` (live-L2 64 KiB block bitmask, forced 0 on `isLast`),
    and `header[8]` (`active = 1`, `isLast`, `ene`). -/
def encodeTaskHeader9Multi
    (target : Target)
    (taskId taskWords ene : Nat)
    (isFirst isLast : Bool)
    (liveL2Mask : UInt32 := 0)
    (header4Override : Option UInt32 := none) : Except String (Array UInt32) :=
  encodeTaskHeaderMulti target taskId taskWords ene isFirst isLast liveL2Mask header4Override none

/-- Encode the 9-word task header with explicit `ene` (`0` for PE, `1..5` for NE) and
    optional `header4` override (default `0x00fff868`). -/
def encodeTaskHeader9Ex
    (taskId taskWords ene : Nat)
    (header4 : UInt32 := taskHeaderWord4Const) : Except String (Array UInt32) :=
  encodeTaskHeader9Multi .h17 taskId taskWords ene true true 0 (some header4)

/-- Encode the 9-word NE task header (`ene = logNE + 1`).
    Checks `taskId < 65536`, `logNE <= 4`, and `9 <= taskWords <= 0x7ff`. -/
def encodeTaskHeader9 (taskId taskWords logNE : Nat) : Except String (Array UInt32) := do
  if logNE > 4 then
    throw s!"task header: logNE {logNE} exceeds maximum 4"
  encodeTaskHeader9Ex taskId taskWords (logNE + 1) taskHeaderWord4Const

-- ============================================================================
-- Machine register addresses (byte offsets in ANE register space)
-- ============================================================================

-- Common block (0x0000..0x0058)
def regWin : Nat := 0x0004
def regHin : Nat := 0x0008
def regCin : Nat := 0x000c
def regDin : Nat := 0x0010
def regWout : Nat := 0x0014
def regHout : Nat := 0x0018
def regCout : Nat := 0x001c
def regDout : Nat := 0x0020
def regBatchN : Nat := 0x0024
def regConvCfg : Nat := 0x0028
def regDwGroups : Nat := 0x0030
def regTH : Nat := 0x0034
def regNeCommon : Nat := 0x003c
def regPeCommon : Nat := 0x003c
def regNeTile : Nat := 0x0040
def regPeTile : Nat := 0x0044
def regPeBcast : Nat := 0x0048
def regTargetCfg : Nat := 0x004c
def regNeActShiftEn : Nat := 0x0050

-- L2 staging block (0x4100..0x4160)
def regCmpFlags : Nat := 0x4100
def regSrc1Cfg : Nat := 0x4104
def regSrc2Cfg : Nat := 0x4108
def regSrc1Base : Nat := 0x4110
def regSrc1ChanStride : Nat := 0x4114
def regSrc1RowStride : Nat := 0x4118
def regSrc1RowBytesA : Nat := 0x411c
def regSrc1RowBytesB : Nat := 0x4120
def regSrc2Base : Nat := 0x4124
def regSrc2ChanStride : Nat := 0x4128
def regSrc2RowStride : Nat := 0x412c
def regSrc2RowBytesA : Nat := 0x4130
def regSrc2RowBytesB : Nat := 0x4134
def regResCfg : Nat := 0x4148
def regResBase : Nat := 0x414c
def regResChanStride : Nat := 0x4150
def regResRowStride : Nat := 0x4154
def regResRowBytesA : Nat := 0x4158
def regResRowBytesB : Nat := 0x415c

-- Planar Engine (PE) arithmetic block (0x4500..0x4514)
def regPeOp : Nat := 0x4500
def regPeBias : Nat := 0x4504
def regPeScale : Nat := 0x4508
def regPeEps : Nat := 0x450c
def regPeInScale : Nat := 0x4510
def regPeRedScale : Nat := 0x4514

-- Neural Engine block (0x4900..0x4914)
def regNeCfg : Nat := 0x4900
def regNeOp : Nat := 0x4904
def regScalarBias : Nat := 0x490c
def regNeScale : Nat := 0x4910

-- Tile DMA (Source) block (0x4d00..0x4d6c)
def regSrc1Dep : Nat := 0x4d00
def regSrc2Dep : Nat := 0x4d04
def regSrc1DmaCfg : Nat := 0x4d08
def regSrc2DmaCfg : Nat := 0x4d0c
def regSrc1BaseLo : Nat := 0x4d10
def regSrc1BaseHi : Nat := 0x4d14
def regSrc1DmaRowStride : Nat := 0x4d18
def regSrc1DmaPlaneStride : Nat := 0x4d1c
def regSrc1DmaDStride : Nat := 0x4d20
def regSrc1DmaNStride : Nat := 0x4d24
def regSrc2BaseLo : Nat := 0x4d28
def regSrc2BaseHi : Nat := 0x4d2c
def regSrc2DmaRowStride : Nat := 0x4d30
def regSrc2DmaPlaneStride : Nat := 0x4d34
def regSrc1Fmt : Nat := 0x4d68
def regSrc2Fmt : Nat := 0x4d6c

-- Result DMA block (0x5100..0x513c)
def regResDep : Nat := 0x5100
def regResDmaWarWait : Nat := 0x5104
def regResBaseLo : Nat := 0x5108
def regResBaseHi : Nat := 0x510c
def regResDmaRowStride : Nat := 0x5110
def regResDmaPlaneStride : Nat := 0x5114
def regResDmaDStride : Nat := 0x5118
def regResDmaNStride : Nat := 0x511c
def regResFmt : Nat := 0x5138
def regResDmaViewOff : Nat := 0x5150
def regSrc1DmaCrop : Nat := 0x4d84

-- Kernel DMA, chunk table, and standalone LUT descriptor block (0x5500..0x561c)
def regKernelStage : Nat := 0x5500
def regKernelBaseLo : Nat := 0x5510
def regKernelBaseHi : Nat := 0x5514
def regKernelBatchStride : Nat := 0x5518
def regKernelRoundStride : Nat := 0x551c
def regChunkSlot (s : Nat) : Nat := 0x5520 + 4 * s
def regChunkOffset (s : Nat) : Nat := 0x5560 + 4 * s
def regChunkSize (s : Nat) : Nat := 0x55a0 + 4 * s
def regLutCfg : Nat := 0x5610
def regLutSize : Nat := 0x5614
def regLutBaseLo : Nat := 0x5618
def regLutBaseHi : Nat := 0x561c

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

/-- `0x0030` neChanReduceGroups (channel-reduce mode): `fixed[15:0] = 0`, `chanCount[31:16]`. -/
def neChanReduceGroupsLayout : WordLayout := {
  name := "neChanReduceGroups"
  fixedMask := 0x0000ffff
  fixedValue := 0x00000000
  fields := #[
    { name := "chanCount", lo := 16, width := 16 }
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

/-- `0x003c` neViewCommon (weightless NE copy/view): `kind[7:0]`, `shPref[10:8]`,
    `shMin[14:12]`, `shMax[18:16]`, `F[21:19]`, `transposeOut[28:28]`,
    `transposeWideGroup[29:29]` (all other bits must be 0). -/
def neViewCommonLayout : WordLayout := {
  name := "neViewCommon"
  fields := #[
    { name := "kind", lo := 0, width := 8 },
    { name := "shPref", lo := 8, width := 3 },
    { name := "shMin", lo := 12, width := 3 },
    { name := "shMax", lo := 16, width := 3 },
    { name := "F", lo := 19, width := 3 },
    { name := "transposeOut", lo := 28, width := 1 },
    { name := "transposeWideGroup", lo := 29, width := 1 }
  ]
}

/-- `0x003c` neMatmulCommon (dynamic runtime-kernel NE matmul): `lo2[2:2]` (`alignUp(N, 8) <= 64`),
    `shPref[10:8]`, `shMin[14:12]`, `shMax[18:16]`, `F[21:19]`, `dynKern[29:29]` (`M * B > 1`). -/
def neMatmulCommonLayout : WordLayout := {
  name := "neMatmulCommon"
  fields := #[
    { name := "lo2", lo := 2, width := 1 },
    { name := "shPref", lo := 8, width := 3 },
    { name := "shMin", lo := 12, width := 3 },
    { name := "shMax", lo := 16, width := 3 },
    { name := "F", lo := 19, width := 3 },
    { name := "dynKern", lo := 29, width := 1 }
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

/-- `0x003c` peCommon: `mode[7:0]` (`0x10` pool, `0x20` 1-source, `0x30` 2-source, `0x50` reduce), bits `[31:8] = 0`. -/
def peCommonLayout : WordLayout := {
  name := "peCommon"
  fields := #[
    { name := "mode", lo := 0, width := 8 }
  ]
}

/-- `0x4500` peReduceOp (PE reduction mode): `reducer[1:0]`, `preOp[8:4]`, `postOp[13:12]`. -/
def peReduceOpLayout : WordLayout := {
  name := "peReduceOp"
  fields := #[
    { name := "reducer", lo := 0, width := 2 },
    { name := "preOp", lo := 4, width := 5 },
    { name := "postOp", lo := 12, width := 2 }
  ]
}

/-- `0x0044` peTile: `k[3:0]`, `m[7:4]`, bits `[31:8] = 0`. -/
def peTileLayout : WordLayout := {
  name := "peTile"
  fields := #[
    { name := "k", lo := 0, width := 4 },
    { name := "m", lo := 4, width := 4 }
  ]
}

/-- `0x0048` peBcast: 2-source broadcast flags (`src1` in `[3:0]`, `src2` in `[7:4]`),
    reserved bits 2, 6, `[31:8]` must be 0. -/
def peBcastLayout : WordLayout := {
  name := "peBcast"
  fields := #[
    { name := "src1W", lo := 0, width := 1 },
    { name := "src1H", lo := 1, width := 1 },
    { name := "src1C", lo := 3, width := 1 },
    { name := "src2W", lo := 4, width := 1 },
    { name := "src2H", lo := 5, width := 1 },
    { name := "src2C", lo := 7, width := 1 }
  ]
}

/-- `0x0050` neActShiftEn: `enable[0:0]`, bits `[31:1] = 0`. -/
def neActShiftEnLayout : WordLayout := {
  name := "neActShiftEn"
  fields := #[
    { name := "enable", lo := 0, width := 1 }
  ]
}

/-- `0x4104` src1Cfg (single-task): `mode[1:0]`, `P[11:8]`, fixed flags `0x00500070`. -/
def src1CfgLayout : WordLayout := {
  name := "src1Cfg"
  fixedMask := 0xfff0f0fc
  fixedValue := 0x00500070
  fields := #[
    { name := "mode", lo := 0, width := 2 },
    { name := "P", lo := 8, width := 4 }
  ]
}

/-- `0x4104`/`0x4108` multi-task source port config layout:
    `mode[1:0]`, `handoffNe[2:2]`, `handoffPe[3:3]`, `waitNe4[4:4]`, `waitNe5[5:5]`,
    `dmaFlag[6:6]`, `P[11:8]`, `waitPe20[20:20]`, `waitPe22[22:22]` (all other bits 0). -/
def portCfgMultiLayout (name : String) : WordLayout := {
  name := name
  fields := #[
    { name := "mode", lo := 0, width := 2 },
    { name := "handoffNe", lo := 2, width := 1 },
    { name := "handoffPe", lo := 3, width := 1 },
    { name := "waitNe4", lo := 4, width := 1 },
    { name := "waitNe5", lo := 5, width := 1 },
    { name := "dmaFlag", lo := 6, width := 1 },
    { name := "P", lo := 8, width := 4 },
    { name := "waitPe20", lo := 20, width := 1 },
    { name := "waitPe22", lo := 22, width := 1 }
  ]
}

/-- `0x4148` resCfg (single-task): `mode[1:0]`, `direct[3:3]`, `P[11:8]`, fixed flags `0x00500070`. -/
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

/-- `0x4148` resCfg (multi-task): `mode[1:0]`, `direct[3:3]`, `waitNe4[4:4]`, `waitNe5[5:5]`,
    `dmaFlag[6:6]`, `P[11:8]`, `waitPe20[20:20]`, `waitPe22[22:22]` (all other bits 0). -/
def resCfgMultiLayout : WordLayout := {
  name := "resCfgMulti"
  fields := #[
    { name := "mode", lo := 0, width := 2 },
    { name := "direct", lo := 3, width := 1 },
    { name := "waitNe4", lo := 4, width := 1 },
    { name := "waitNe5", lo := 5, width := 1 },
    { name := "dmaFlag", lo := 6, width := 1 },
    { name := "P", lo := 8, width := 4 },
    { name := "waitPe20", lo := 20, width := 1 },
    { name := "waitPe22", lo := 22, width := 1 }
  ]
}

/-- `0x4904` neOp (NE conv): base `0x00103c00`, `biasInKernel[4:4]`, `relu[16:16]`, `lut[17:17]`. -/
def neOpLayout : WordLayout := {
  name := "neOp"
  fixedMask := 0xfffcffef
  fixedValue := 0x00103c00
  fields := #[
    { name := "biasInKernel", lo := 4, width := 1 },
    { name := "relu", lo := 16, width := 1 },
    { name := "lut", lo := 17, width := 1 }
  ]
}

/-- `0x4904` neActOp (standalone NE activation): base `0x00103c08`,
    `noShift[2:2]`, `relu[16:16]`, `lut[17:17]`. -/
def neActOpLayout : WordLayout := {
  name := "neActOp"
  fixedMask := 0xfffcfffb
  fixedValue := 0x00103c08
  fields := #[
    { name := "noShift", lo := 2, width := 1 },
    { name := "relu", lo := 16, width := 1 },
    { name := "lut", lo := 17, width := 1 }
  ]
}

/-- `0x490c` scalarBias: IEEE fp16 bits in `[15:0]`. -/
def scalarBiasLayout : WordLayout := dim16Layout "scalarBias"

/-- `0x490c` scalarBiasExt: IEEE fp16 scalar bias in `[15:0]`, unsigned `biasShift` in `[31:16]`. -/
def scalarBiasExtLayout : WordLayout := {
  name := "scalarBiasExt"
  fields := #[
    { name := "val", lo := 0, width := 16 },
    { name := "biasShift", lo := 16, width := 16 }
  ]
}

/-- `0x4910` neScale: IEEE fp16 bits in `[15:0]`. -/
def neScaleLayout : WordLayout := dim16Layout "neScale"

/-- `0x4910` neScaleExt: IEEE fp16 tensor scale in `[15:0]`, power-of-two scale exponent `pexp` in `[31:16]`. -/
def neScaleExtLayout : WordLayout := {
  name := "neScaleExt"
  fields := #[
    { name := "val", lo := 0, width := 16 },
    { name := "pexp", lo := 16, width := 16 }
  ]
}

/-- `0x4900` neCfgQuant: `wKind[3:0]`, `wWidth[7:4]`, `sparse[8:8]`, `dynKern[16:16]`,
    `hasZP[24:24]`, `h17Flag[28:28]` (all other bits must be 0). -/
def neCfgQuantLayout : WordLayout := {
  name := "neCfgQuant"
  fields := #[
    { name := "wKind", lo := 0, width := 4 },
    { name := "wWidth", lo := 4, width := 4 },
    { name := "sparse", lo := 8, width := 1 },
    { name := "dynKern", lo := 16, width := 1 },
    { name := "hasZP", lo := 24, width := 1 },
    { name := "h17Flag", lo := 28, width := 1 }
  ]
}

/-- `0x4904` neOpQuant: `biasInKernel[4:4]`, `unscaled[13:10]`, `perLaneScale[14:14]`,
    `relu[16:16]`, `lut[17:17]`, `base20[20:20]` (all other bits must be 0). -/
def neOpQuantLayout : WordLayout := {
  name := "neOpQuant"
  fields := #[
    { name := "biasInKernel", lo := 4, width := 1 },
    { name := "unscaled", lo := 10, width := 4 },
    { name := "perLaneScale", lo := 14, width := 1 },
    { name := "relu", lo := 16, width := 1 },
    { name := "lut", lo := 17, width := 1 },
    { name := "base20", lo := 20, width := 1 }
  ]
}

-- Named quantized weight kind & control constants
def wKindUint8 : UInt32 := 0
def wKindInt8 : UInt32 := 1
def wKindFp16 : UInt32 := 2
def wKindUint4 : UInt32 := 4
def wKindInt4 : UInt32 := 5
def wKindPal : UInt32 := 6

/-- `0x4904[13:10]` unscaled-format nibble (`0xf`, i.e., `0x3c00`) for `fp16` and `pal*`. -/
def neOpUnscaledBits : UInt32 := 0xf

/-- Maximum legal `0x4910[31:16]` scale power-of-two exponent `pexp`. -/
def maxQuantScalePexp : Nat := 30

/-- Maximum legal `0x490c[31:16]` folded-bias shift `biasShift`. -/
def maxQuantBiasShift : Nat := 15

/-- Return `true` iff `(wKind, wWidth)` is a hardware-legal NE weight format. -/
def isLegalNeWeightFormat (wKind : UInt32) (wWidth : Nat) : Bool :=
  (wKind == wKindFp16 && wWidth == 8) ||
  (wKind == wKindInt8 && wWidth == 8) ||
  (wKind == wKindUint8 && wWidth == 8) ||
  (wKind == wKindInt4 && wWidth == 4) ||
  (wKind == wKindUint4 && wWidth == 4) ||
  (wKind == wKindPal && (wWidth == 1 || wWidth == 2 || wWidth == 4 || wWidth == 6 || wWidth == 8))

/-- Return `true` iff `wKind` is a scaled integer weight format (`uint8`, `int8`, `uint4`, `int4`). -/
def isIntegerWeightKind (wKind : UInt32) : Bool :=
  wKind == wKindUint8 || wKind == wKindInt8 || wKind == wKindUint4 || wKind == wKindInt4

/-- Return `true` iff `wKind` is an unsigned integer weight format (`uint8` or `uint4`). -/
def isUnsignedWeightKind (wKind : UInt32) : Bool :=
  wKind == wKindUint8 || wKind == wKindUint4

/-- `0x003c` (`neCommon`) bit 27 (`wide3x3`) control for quantized NE convolutions:
    `1` iff `wide3x3` is true (H17 dense 3x3 stride-1 within `maxBlockSlots`),
    `hasZP` is false, and `wKind` is not unsigned (`uint8` / `uint4`); `0` otherwise. -/
def neCommonQuantBit27 (wide3x3 hasZP : Bool) (wKind : UInt32) : UInt32 :=
  if wide3x3 && !hasZP && !isUnsignedWeightKind wKind then 1 else 0

/-- Canonical 16-byte signed 4-bit two's-complement decode table for `int4` (`wKind = 5`). -/
def int4DecodeTable : Array UInt8 := #[
  0, 1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255
]

/-- Canonical 16-byte unsigned 4-bit decode table for `uint4` (`wKind = 4`). -/
def uint4DecodeTable : Array UInt8 := #[
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
]

/-- Byte count required to LSB-first bit-pack `count` codes of bit-width `width`. -/
def packedByteCount (width count : Nat) : Nat :=
  (count * width + 7) / 8

/-- 64-byte-padded decode/palette table byte size at offset 0 of each quantized kernel chunk. -/
def quantTableBytes (wKind : UInt32) (wWidth : Nat) : Nat :=
  if wKind == wKindInt4 || wKind == wKindUint4 then
    64
  else if wKind == wKindPal then
    alignUp (2 * ((1 : Nat) <<< wWidth)) 64
  else
    0

/-- Unpadded byte size of one output-channel block of `n` lanes inside a quantized kernel chunk. -/
def quantBlockBytes
    (wWidth cin taps : Nat)
    (perLaneScale biasInKernel hasZeroPoint : Bool)
    (n : Nat) : Nat :=
  (if perLaneScale then 2 * n else 0) +
  (if biasInKernel then 2 * n else 0) +
  (if hasZeroPoint then n else 0) +
  packedByteCount wWidth (cin * taps * n)

/-- Exact 64-byte-padded byte size of a quantized kernel chunk for a group's output-channel blocks. -/
def expectedQuantChunkSize
    (wKind : UInt32) (wWidth cin taps : Nat)
    (perLaneScale biasInKernel hasZeroPoint : Bool)
    (blocks : Array (Nat × Nat)) : Nat :=
  let body := blocks.foldl
    (fun acc (_, n) => acc + quantBlockBytes wWidth cin taps perLaneScale biasInKernel hasZeroPoint n)
    (quantTableBytes wKind wWidth)
  alignUp body 64

/-- `0x4d00` src1Dep (single-task): `code[7:0]`, `task[23:16]`. -/
def src1DepLayout : WordLayout := {
  name := "src1Dep"
  fields := #[
    { name := "code", lo := 0, width := 8 },
    { name := "task", lo := 16, width := 8 }
  ]
}

/-- `0x4d00`/`0x4d04` multi-task source DMA dependency layout:
    `code[7:0]`, `task[23:16]`, `depDist[27:24]`, `depKind[29:28]` (all other bits 0). -/
def srcDepMultiLayout (name : String) : WordLayout := {
  name := name
  fields := #[
    { name := "code", lo := 0, width := 8 },
    { name := "task", lo := 16, width := 8 },
    { name := "depDist", lo := 24, width := 4 },
    { name := "depKind", lo := 28, width := 2 }
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

/-- `0x5500` kernelStage (single-task): fixed `0x000002c0`, `sparse[5:5]`, `stage[23:16]`. -/
def kernelStageLayout : WordLayout := {
  name := "kernelStage"
  fixedMask := 0xff00ffdf
  fixedValue := 0x000002c0
  fields := #[
    { name := "sparse", lo := 5, width := 1 },
    { name := "stage", lo := 16, width := 8 }
  ]
}

/-- `0x5500` kernelStage (multi-task): `sparse[5:5]`, `kernEn[6:6]`, `isLast[7:7]`,
    `kernCfg[9:9]`, `stage[23:16]` (all other bits 0). -/
def kernelStageMultiLayout : WordLayout := {
  name := "kernelStageMulti"
  fields := #[
    { name := "sparse", lo := 5, width := 1 },
    { name := "kernEn", lo := 6, width := 1 },
    { name := "isLast", lo := 7, width := 1 },
    { name := "kernCfg", lo := 9, width := 1 },
    { name := "stage", lo := 16, width := 8 }
  ]
}

/-- `0x5500` kernelStage (dynamic runtime-kernel): adds H17 `h17Dyn[10:10]` (`0x400`). -/
def kernelStageDynLayout : WordLayout := {
  name := "kernelStageDyn"
  fields := #[
    { name := "sparse", lo := 5, width := 1 },
    { name := "kernEn", lo := 6, width := 1 },
    { name := "isLast", lo := 7, width := 1 },
    { name := "kernCfg", lo := 9, width := 1 },
    { name := "h17Dyn", lo := 10, width := 1 },
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

/-- `0x5520 + 4*s` chunkSlot[s] (dynamic runtime-kernel): `enable[0:0]`, `on20[5:5]`,
    `produced[6:6]` (BAR 3 scratch kernel), `keepLive[7:7]`, `task[23:16]`. -/
def chunkSlotDynLayout : WordLayout := {
  name := "chunkSlotDyn"
  fields := #[
    { name := "enable", lo := 0, width := 1 },
    { name := "on20", lo := 5, width := 1 },
    { name := "produced", lo := 6, width := 1 },
    { name := "keepLive", lo := 7, width := 1 },
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

-- Target-dependent fixed constants (delegated to Target.profile)
def targetCfgValue (target : Target) : UInt32 :=
  target.profile.targetCfgValue

def neCfgValue (target : Target) : UInt32 :=
  target.profile.neCfgValue

/-- `0x4900` neCfg: target base (`0x00000082` on H16, `0x10000082` on H17), `sparse[8:8]`. -/
def neCfgLayout (target : Target) : WordLayout := {
  name := "neCfg"
  fixedMask := 0xfffffeff
  fixedValue := neCfgValue target
  fields := #[
    { name := "sparse", lo := 8, width := 1 }
  ]
}

/-- Dynamic runtime-kernel bit 16 (`0x00010000`) in `0x4900` (`neCfg`). -/
def neCfgDynKernBit : UInt32 := (1 : UInt32) <<< 16

/-- Dynamic runtime-kernel `0x4900` (`neCfg`) constant (`0x00010082` on H16, `0x10010082` on H17). -/
def neCfgDynValue (target : Target) : UInt32 :=
  neCfgValue target ||| neCfgDynKernBit

/-- `0x4900` neMatmulCfg: target base (`0x00000082` on H16, `0x10000082` on H17), `dynKern[16:16]`. -/
def neMatmulCfgLayout (target : Target) : WordLayout := {
  name := "neMatmulCfg"
  fixedMask := 0xfffeffff
  fixedValue := neCfgValue target
  fields := #[
    { name := "dynKern", lo := 16, width := 1 }
  ]
}

def kernelStageH17DynBit : UInt32 := (1 : UInt32) <<< 10
def neOpMatmul : UInt32 := 0x00103c00

def resDepCode (target : Target) : UInt32 :=
  target.profile.resDepCode

-- Named multi-task port config, dependency, and BAR constants
def portModeL2 : UInt32 := 0
def portModeDma : UInt32 := 2
def cfgBitHandoffNe : UInt32 := (1 : UInt32) <<< 2
def cfgBitHandoffPe : UInt32 := (1 : UInt32) <<< 3
def cfgBitWaitNe4 : UInt32 := (1 : UInt32) <<< 4
def cfgBitWaitNe5 : UInt32 := (1 : UInt32) <<< 5
def cfgDmaFlag : UInt32 := (1 : UInt32) <<< 6
def cfgBitWaitPe20 : UInt32 := (1 : UInt32) <<< 20
def cfgBitWaitPe22 : UInt32 := (1 : UInt32) <<< 22
def cfgFirstTaskMask : UInt32 := cfgBitWaitNe4 ||| cfgBitWaitNe5 ||| cfgBitWaitPe20 ||| cfgBitWaitPe22
def resDepScratchFlags : UInt32 := 0xc0
def srcDepCodeDma : UInt32 := 0x21
def srcDepCodePrev : UInt32 := 0x41
def srcDmaCfgPrev : UInt32 := 0x0000004e
def srcFmtPayloadP1 : UInt32 := 0x002031
def resFmtPayloadP1 : UInt32 := 0x302031
def srcFmtWordP1 : UInt32 := ((1 : UInt32) <<< 24) ||| srcFmtPayloadP1
def resFmtWordP1 : UInt32 := ((1 : UInt32) <<< 24) ||| resFmtPayloadP1
def out2FormValue : UInt32 := 0xe0
def barSlotScratch : UInt8 := 3
def barSlotConfigZero : UInt8 := 4
def barSlotFirstSurface : UInt8 := 6

-- Fixed register constants
def src2CfgValue : UInt32 := cfgFirstTaskMask ||| ((1 : UInt32) <<< 8) ||| portModeL2
def src2CfgEnabledValue : UInt32 := cfgFirstTaskMask ||| ((1 : UInt32) <<< 8) ||| cfgDmaFlag ||| portModeDma
def src1DmaCfgValue : UInt32 := 0x000000ce
def fp16OneBits : UInt32 := 0x3c00
def neScaleOne : UInt32 := 0x00003c00
def neOpViewCopy : UInt32 := 0x00103c0c
def fp16InvLn2Bits : UInt16 := 0x3dc5
def neActCfgValue : UInt32 := 0x00010082
def kernelStageNoKernValue : UInt32 := 0x00000080
def peCommonPoolValue : UInt32 := 0x00000010
def peCommonOneSrcValue : UInt32 := 0x00000020
def peCommonTwoSrcValue : UInt32 := 0x00000030
def peCommonReduceValue : UInt32 := 0x00000050
def cmpFlagMaxValue : UInt32 := 0x00000004
def cmpFlagMinValue : UInt32 := 0x0000000c
def nePoolMaxOpValue : UInt32 := 0x00103c0b
def nePoolMinOpValue : UInt32 := 0x00183c0b
def nePoolSumOpValue : UInt32 := 0x00103c08
def peOpAbsValue : UInt32 := 0x000001c0
def peOpSquareValue : UInt32 := 0x00000010
def peOpAddValue : UInt32 := 0x00080000
def peOpMulValue : UInt32 := 0x00080004
def peOpMaxValue : UInt32 := 0x00080008
def peOpMinValue : UInt32 := 0x0008000c
def peOpSubValue : UInt32 := 0x000c0000
def lutTableBytes : Nat := 128
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

/-- Pack plain writes and explicit BAR-tagged sequential records separately in a single pass,
    ensuring no byte address is written by both plain and BAR records. -/
def packTaskRecordsSplit (plain : Array RegWrite) (barRecs : Array BarRegWrite) :
    Except String (Array RegRecord × Array RegRecord) := do
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
  pure (plainRecs, barEncoded)

/-- Pack plain writes followed by explicit BAR-tagged sequential records in caller order,
    ensuring no byte address is written by both plain and BAR records. -/
def packTaskRecords (plain : Array RegWrite) (barRecs : Array BarRegWrite) : Except String (Array RegRecord) := do
  let (plainRecs, barEncoded) ← packTaskRecordsSplit plain barRecs
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
  recordLen : Nat := 1
  deriving Repr, BEq, Inhabited

/-- A single decoded task from `__TEXT,__text`. -/
structure DecodedTask where
  taskIndex : Nat
  startOffset : Nat
  header : Array UInt32
  writes : Array DecodedWrite
  deriving Repr, BEq, Inhabited

/-- Push a list of little-endian 32-bit words onto `bs`. -/
def pushU32LEs (bs : ByteArray) : List UInt32 → ByteArray
  | [] => bs
  | v :: rest => pushU32LEs (pushU32LE bs v) rest

/-- Decode the set-bit register byte addresses `(base + i + 1) * 4` for `i < k` in ascending order. -/
def decodeBitmaskAddrsAux (base mask : Nat) : Nat → List Nat → List Nat
  | 0, acc => acc
  | i + 1, acc =>
    let acc' := if ((mask >>> i) &&& 1) == 1 then ((base + i + 1) * 4) :: acc else acc
    decodeBitmaskAddrsAux base mask i acc'

/-- Decode all register byte addresses targeted by a bitmask record header `(base, mask)`. -/
def decodeBitmaskAddrs (base mask : Nat) : List Nat :=
  (base * 4) :: decodeBitmaskAddrsAux base mask 16 []

/-- Decode payload words for a bitmask register record into `acc`. -/
def decodeBitmaskWrites (bs : ByteArray) (recPos recLen : Nat) :
    Nat → List Nat → Array DecodedWrite → Except String (Array DecodedWrite)
  | _, [], acc => .ok acc
  | i, addr :: rest, acc => do
    let pOff := recPos + 4 * i
    let v ← readU32LE bs pOff
    decodeBitmaskWrites bs recPos recLen (i + 1) rest (acc.push {
      byteAddr := addr
      value := v
      barSlot := none
      payloadOffset := pOff
      recordLen := recLen
    })

/-- Decode `rem` payload words (starting at index `i`) for a sequential register record into `acc`. -/
def decodeSequentialWrites (bs : ByteArray) (recPos base cnt : Nat) (slot? : Option UInt8) :
    Nat → Nat → Array DecodedWrite → Except String (Array DecodedWrite)
  | _, 0, acc => .ok acc
  | i, rem + 1, acc => do
    let pOff := recPos + 4 * i
    let v ← readU32LE bs pOff
    decodeSequentialWrites bs recPos base cnt slot? (i + 1) rem (acc.push {
      byteAddr := (base + i) * 4
      value := v
      barSlot := slot?
      payloadOffset := pOff
      recordLen := cnt
    })

/-- Decode one register record starting at byte offset `recPos` in `bs`, appending its
    `DecodedWrite`s to `acc` and returning `(acc', nextRecPos)`. -/
def decodeRegRecordAt (bs : ByteArray) (recPos : Nat) (acc : Array DecodedWrite) :
    Except String (Array DecodedWrite × Nat) := do
  let h ← readU32LE bs recPos
  let payloadStart := recPos + 4
  let base := (h &&& 0x7fff).toNat
  let isBitmask := ((h >>> 31) &&& 1) == 1
  if isBitmask then
    let mask := ((h >>> 15) &&& 0xffff).toNat
    let addrs := decodeBitmaskAddrs base mask
    let recLen := addrs.length
    let acc' ← decodeBitmaskWrites bs payloadStart recLen 0 addrs acc
    pure (acc', payloadStart + 4 * recLen)
  else
    let cnt := (((h >>> 15) &&& 0x3f).toNat) + 1
    let slot? : Option UInt8 :=
      if ((h >>> 29) &&& 1) == 1 then
        some ((h >>> 23) &&& 0x3f).toUInt8
      else
        none
    let acc' ← decodeSequentialWrites bs payloadStart base cnt slot? 0 cnt acc
    pure (acc', payloadStart + 4 * cnt)

/-- Decode all register records in `[recPos, taskEnd)` of `bs` into `acc`. -/
def decodeTaskRecords (bs : ByteArray) (recPos taskEnd : Nat) (acc : Array DecodedWrite) :
    Except String (Array DecodedWrite) :=
  if _ : recPos < taskEnd then
    match decodeRegRecordAt bs recPos acc with
    | .error e => .error e
    | .ok (acc', nextPos) =>
      if _ : recPos < nextPos then
        decodeTaskRecords bs nextPos taskEnd acc'
      else
        .error s!"decoder internal error: non-advancing register record at {toHex recPos}"
  else
    .ok acc
termination_by taskEnd - recPos

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

      let writes ← decodeTaskRecords bs (cursor + 4 * headerWords) taskEnd #[]

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
