import Vane.Binary
import Vane.TaskStream
import Vane.Hwx
import Vane.Sections
import Vane.Register
import Vane.Tiling
import Vane.Program
import Vane.Verify

namespace Vane.Conv

open Vane.Binary
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Tiling
open Vane.Program
open Vane.Verify

/-- Packed chunk slot metadata in `__KERN_0`. -/
structure KernelChunkSlot where
  slot : Nat
  offset : Nat
  size : Nat
  deriving Repr, BEq, Inhabited

/-- Result of packing fp16 weights and optional bias into `__KERN_0`. -/
structure PackedKernel where
  bytes : ByteArray
  slots : Array KernelChunkSlot
  biasInKernel : Bool
  scalarBiasBits : UInt16
  deriving BEq

/-- Complete compilation artifact for a single-task fp16 NE convolution (`1x1` or `3x3`). -/
structure CompiledConv where
  target : Target
  shape : ConvSpec
  tiling : Tiling
  sparse : Bool
  wide3x3 : Bool
  hasBias : Bool
  relu : Bool
  direct : Bool
  l2UseBytes : Nat
  kernelBytes : Nat
  hwx : ByteArray
  info : HwxInfo
  sections : LoadSections

-- ============================================================================
-- Stride-2 tap-slot order & zero-skipping sparse stream encoder / decoder
-- ============================================================================

/-- Flattened 16-entry stride-2 3x3 tap-slot list in hardware `(p, q, row, col)` phase order. -/
def stride2TapSlots : Array (Option Nat) := #[
  some 0, some 2,
  some 6, some 8,
  some 1, none,
  some 7, none,
  some 3, some 5,
  none, none,
  some 4, none,
  none, none
]

/-- Return `true` iff fp16 bit pattern `w` is non-zero (skips both `+0.0 = 0x0000` and `-0.0 = 0x8000`). -/
def isNonZeroFp16 (w : UInt16) : Bool :=
  (w &&& 0x7fff) != 0

/-- Maximum byte length representable in the 3-byte little-endian `u24` sparse stream prefix. -/
def maxSparseBodyBytes : Nat := 0xffffff

/-- Append a zero-skipping sparse encoded stream directly onto `dst` (O(N), avoiding quadratic `++`). -/
def appendSparseStream (dst : ByteArray) (slots : Array (Option UInt16)) : Except String ByteArray := do
  let paddedLen := alignUp slots.size 16
  let numVecs := paddedLen / 16
  let mut bodyHalfwords : Array UInt16 := Array.mkEmpty (numVecs * 2)

  for v in [:numVecs] do
    let base := v * 16
    let mut mask : UInt16 := 0
    let mut kept : Array UInt16 := Array.mkEmpty 16
    for i in [:16] do
      let idx := base + i
      if h : idx < slots.size then
        match slots[idx] with
        | some w =>
          if isNonZeroFp16 w then
            mask := mask ||| ((1 : UInt16) <<< i.toUInt16)
            kept := kept.push w
        | none => pure ()
    bodyHalfwords := bodyHalfwords.push mask
    for w in kept do
      bodyHalfwords := bodyHalfwords.push w

  let bodyBytes := 2 * bodyHalfwords.size
  if bodyBytes > maxSparseBodyBytes then
    throw s!"sparse kernel body length {bodyBytes} exceeds 24-bit limit {maxSparseBodyBytes}"

  let mut out := dst
  out := out.push (bodyBytes &&& 0xff).toUInt8
  out := out.push ((bodyBytes >>> 8) &&& 0xff).toUInt8
  out := out.push ((bodyBytes >>> 16) &&& 0xff).toUInt8
  for hw in bodyHalfwords do
    out := pushU16LE out hw
  pure out

/-- Encode a slot stream (`Array (Option UInt16)`) into the zero-skipping sparse binary format:
    pads to a multiple of 16 slots, encodes each 16-slot vector as `(mask : u16, kept weights...)`,
    and prefixes with the 3-byte little-endian `u24` body byte length. -/
def encodeSparseStream (slots : Array (Option UInt16)) : Except String ByteArray :=
  appendSparseStream ByteArray.empty slots

/-- Decode a zero-skipping sparse stream starting at `offset` in `bs`.
    Returns `(decodedSlots, totalBytesConsumed)` where `decodedSlots.size` is a multiple of 16
    and skipped/padding slots decode to `0x0000`. -/
def decodeSparseStream (bs : ByteArray) (offset : Nat := 0) : Except String (Array UInt16 × Nat) := do
  if offset + 3 > bs.size then
    throw s!"sparse stream truncated at u24 length header (offset={offset}, size={bs.size})"
  let b0 := (← readU8 bs offset).toNat
  let b1 := (← readU8 bs (offset + 1)).toNat
  let b2 := (← readU8 bs (offset + 2)).toNat
  let bodyBytes := b0 ||| (b1 <<< 8) ||| (b2 <<< 16)
  if bodyBytes % 2 != 0 then
    throw s!"sparse stream body length {bodyBytes} is not a multiple of 2"
  let bodyEnd := offset + 3 + bodyBytes
  if bodyEnd > bs.size then
    throw s!"sparse stream body [{offset + 3}, {bodyEnd}) exceeds buffer size {bs.size}"

  let mut decoded : Array UInt16 := #[]
  let mut pos := offset + 3
  while pos < bodyEnd do
    let mask ← readU16LE bs pos
    pos := pos + 2
    for i in [:16] do
      if ((mask.toNat >>> i) &&& 1) == 1 then
        if pos + 2 > bodyEnd then
          throw s!"sparse stream truncated inside 16-slot vector at offset {pos} (bodyEnd={bodyEnd})"
        let w ← readU16LE bs pos
        pos := pos + 2
        decoded := decoded.push w
      else
        decoded := decoded.push 0
  pure (decoded, 3 + bodyBytes)

-- ============================================================================
-- Unified wide3x3 condition, kernel chunk packer, register lowering, compiler
-- ============================================================================

/-- Compute `neCommon` bit 27 (`wide3x3`): `true` iff 3x3, `target == h17`, `effMode == dense`,
    `stride == 1`, and `maxBlockSlots <= 32768 * rep`. -/
def computeWide3x3Conv (target : Target) (sh : ConvSpec) (t : Tiling) : Bool :=
  if sh.kernelSize != 3 || target != .h17 || sh.effMode != .dense || sh.stride != 1 then
    false
  else
    let rep := ((1 : Nat) <<< t.logNE) >>> t.f
    maxBlockSlotsConv sh t <= 32768 * rep

/-- Unified fp16 kernel weight & optional bias packer for 1x1 dense, 3x3 dense stride-1,
    3x3 depthwise stride-1, and 3x3 stride-2 zero-skipping sparse convolutions.
    Enforces the hardware 64 KiB per-chunk size cap on every active chunk. -/
def packConvKernelChunks
    (target : Target)
    (sh : ConvSpec)
    (tiling : Tiling)
    (weights : ByteArray)
    (bias : Option ByteArray) : Except String PackedKernel := do
  validateConvSafety target sh tiling

  let cinW := sh.cinW
  let taps := sh.taps
  let expectedWeightBytes := 2 * sh.cout * cinW * taps
  if weights.size != expectedWeightBytes then
    throw s!"weights byte size {weights.size} != expected {expectedWeightBytes} (2 * Cout={sh.cout} * cinW={cinW} * taps={taps})"

  if let some b := bias then
    let expectedBiasBytes := 2 * sh.cout
    if b.size != expectedBiasBytes then
      throw s!"bias byte size {b.size} != expected {expectedBiasBytes} (2 * Cout={sh.cout})"

  let hasBias := bias.isSome
  let scalarBiasMode := sh.effMode == .dense && sh.cout == 1
  let biasInKernel := hasBias && !scalarBiasMode
  let scalarBiasBits ←
    if scalarBiasMode then
      match bias with
      | some b => readU16LE b 0
      | none => pure (0 : UInt16)
    else
      pure (0 : UInt16)

  let groups := (1 : Nat) <<< tiling.f
  let rep := ((1 : Nat) <<< tiling.logNE) >>> tiling.f
  if rep == 0 || groups * rep > 16 then
    throw s!"invalid chunk replication geometry: groups={groups}, rep={rep}"

  let sparse := sh.isSparse
  let mut slotChunks : Array (Nat × ByteArray) := Array.mkEmpty 16
  let mut totalKernelBytes : Nat := 0

  for g in [:groups] do
    let blocks := groupOutputBlocksConv sh tiling g
    if !blocks.isEmpty then
      let mut chunk := ByteArray.empty
      if !sparse then
        -- Dense uncompressed kernel encoding (1x1 or 3x3 stride-1)
        for (o0, n) in blocks do
          if biasInKernel then
            match bias with
            | some b =>
              for k in [:n] do
                let bOff := 2 * (o0 + k)
                chunk := chunk.push (← readU8 b bOff)
                chunk := chunk.push (← readU8 b (bOff + 1))
            | none =>
              throw "internal error: biasInKernel set without bias bytes"
          for ci in [:cinW] do
            for tp in [:taps] do
              for k in [:n] do
                let o := o0 + k
                let wOff := 2 * ((o * cinW + ci) * taps + tp)
                chunk := chunk.push (← readU8 weights wOff)
                chunk := chunk.push (← readU8 weights (wOff + 1))
      else
        -- Stride-2 zero-skipping sparse kernel encoding (3x3)
        for (o0, n) in blocks do
          if biasInKernel then
            match bias with
            | some b =>
              for k in [:n] do
                let bOff := 2 * (o0 + k)
                chunk := chunk.push (← readU8 b bOff)
                chunk := chunk.push (← readU8 b (bOff + 1))
            | none =>
              throw "internal error: biasInKernel set without bias bytes"
          let ev := alignUp n 2
          let mut slotStream : Array (Option UInt16) := Array.mkEmpty (cinW * 16 * ev)
          for ci in [:cinW] do
            for tpOpt in stride2TapSlots do
              match tpOpt with
              | some tp =>
                for l in [:ev] do
                  if l < n then
                    let o := o0 + l
                    let wOff := 2 * ((o * cinW + ci) * 9 + tp)
                    let w ← readU16LE weights wOff
                    slotStream := slotStream.push (some w)
                  else
                    slotStream := slotStream.push none
              | none =>
                for _ in [:ev] do
                  slotStream := slotStream.push none
          chunk ← appendSparseStream chunk slotStream

      chunk := padToAlign chunk dmaLineBytes
      if chunk.size > maxKernelChunkBytes then
        throw s!"{sh.kernelSize}x{sh.kernelSize} kernel group {g} chunk size {chunk.size}B exceeds maximum {maxKernelChunkBytes}B (64 KiB)"

      match target with
      | .h17 =>
        let s := g * rep
        if s >= 16 then
          throw s!"chunk slot index {s} exceeds 0..15"
        slotChunks := slotChunks.push (s, chunk)
        totalKernelBytes := totalKernelBytes + chunk.size
      | .h16 =>
        for j in [:rep] do
          let s := g * rep + j
          if s >= 16 then
            throw s!"chunk slot index {s} exceeds 0..15"
          slotChunks := slotChunks.push (s, chunk)
          totalKernelBytes := totalKernelBytes + chunk.size

  if slotChunks.isEmpty || totalKernelBytes == 0 then
    throw "kernel packer produced 0 active chunk slots"

  let mut kernelBuf := ByteArray.emptyWithCapacity totalKernelBytes
  let mut slots : Array KernelChunkSlot := Array.mkEmpty slotChunks.size
  for (s, chunk) in slotChunks do
    let off := kernelBuf.size
    kernelBuf := kernelBuf ++ chunk
    slots := slots.push { slot := s, offset := off, size := chunk.size }

  pure {
    bytes := kernelBuf
    slots := slots
    biasInKernel := biasInKernel
    scalarBiasBits := scalarBiasBits
  }

/-- Lower a single-task fp16 NE convolution (`1x1` or `3x3`) into a generic `TaskDesc`
    with explicit, width-checked machine register writes and BAR-slot records. -/
def lowerConvTask
    (target : Target)
    (shape : ConvSpec)
    (tiling : Tiling)
    (packed : PackedKernel)
    (relu : Bool)
    (taskId : Nat := 0) : Except String TaskDesc := do
  let sparse := shape.isSparse
  let wide3x3 := computeWide3x3Conv target shape tiling
  let direct := isDirectConv target shape tiling
  let chs := chsBytesConv shape tiling
  let dchs := dchsBytesConv shape tiling
  let srcRow := srcRowStrideConv shape tiling
  let resRow := resRowStrideConv shape tiling
  let resBase := resBaseBytesConv shape tiling

  let ho := shape.outH
  let wo := shape.outW
  let inRowStride := alignUp (2 * shape.w) dmaLineBytes
  let inPlaneStride := shape.h * inRowStride
  let outRowStride := alignUp (2 * wo) dmaLineBytes
  let outPlaneStride := ho * outRowStride

  let winU32 ← natToU32Checked shape.winReg "Win"
  let hinU32 ← natToU32Checked shape.hinReg "Hin"
  let cinU32 ← natToU32Checked shape.cin "Cin"
  let woU32 ← natToU32Checked wo "Wout"
  let hoU32 ← natToU32Checked ho "Hout"
  let coutU32 ← natToU32Checked shape.cout "Cout"
  let kU32 ← natToU32Checked shape.kernelSize "kernelSize"
  let sU32 ← natToU32Checked shape.stride "stride"
  let plU32 ← natToU32Checked shape.padLeft "padLeft"
  let ptU32 ← natToU32Checked shape.padTop "padTop"
  let thU32 ← natToU32Checked tiling.tileHeight "TH"
  let shPrefU32 ← natToU32Checked tiling.shPref "shPref"
  let shMaxU32 ← natToU32Checked tiling.shMax "shMax"
  let fU32 ← natToU32Checked tiling.f "F"
  let effOcgU32 ← natToU32Checked (effOcgConv shape tiling) "ocg"
  let fatU32 ← natToU32Checked tiling.fat "fat"
  let wuU32 ← natToU32Checked tiling.wu "WU"
  let taskIdU32 ← natToU32Checked taskId "taskId"
  let chsU32 ← natToU32Checked chs "src1ChanStride"
  let srcRowU32 ← natToU32Checked srcRow "src1RowStride"
  let srcRowBytesU32 ← natToU32Checked (shape.cin * chs) "src1RowBytes"
  let resBaseU32 ← natToU32Checked resBase "resBase"
  let inRowStrideU32 ← natToU32Checked inRowStride "src1DmaRowStride"
  let inPlaneStrideU32 ← natToU32Checked inPlaneStride "src1DmaPlaneStride"
  let outRowStrideU32 ← natToU32Checked outRowStride "resDmaRowStride"
  let outPlaneStrideU32 ← natToU32Checked outPlaneStride "resDmaPlaneStride"

  let wWin ← encodeWord (dim16Layout "Win") #[("val", winU32)]
  let wHin ← encodeWord (dim16Layout "Hin") #[("val", hinU32)]
  let wWout ← encodeWord (dim16Layout "Wout") #[("val", woU32)]
  let wHout ← encodeWord (dim16Layout "Hout") #[("val", hoU32)]
  let wConvCfg ← encodeWord convCfgLayout #[
    ("kw", kU32), ("kh", kU32),
    ("sx", sU32), ("sy", sU32),
    ("pl", plU32), ("pt", ptU32),
    ("hx", 1), ("vy", 1)
  ]
  let wTH ← encodeWord (dim16Layout "TH") #[("val", thU32)]
  let wNeCommon ← encodeWord neCommonLayout #[
    ("kind", 0),
    ("shPref", shPrefU32),
    ("shMin", shPrefU32),
    ("shMax", shMaxU32),
    ("F", fU32),
    ("wide3x3", if wide3x3 then 1 else 0)
  ]
  let wNeTile ← encodeWord neTileLayout #[
    ("ocg", effOcgU32),
    ("fat", fatU32),
    ("WU", wuU32)
  ]
  let wSrc1Cfg ← encodeWord src1CfgLayout #[("mode", 2), ("P", 1)]
  let wResCfg ← encodeWord resCfgLayout #[
    ("mode", 2),
    ("direct", if direct then 1 else 0),
    ("P", 1)
  ]
  let wNeCfg ← encodeWord (neCfgLayout target) #[("sparse", if sparse then 1 else 0)]
  let wNeOp ← encodeWord neOpLayout #[
    ("biasInKernel", if packed.biasInKernel then 1 else 0),
    ("relu", if relu then 1 else 0)
  ]
  let wNeScale ← encodeWord neScaleLayout #[("val", fp16OneBits)]
  let wSrc1Fmt ← encodeWord (dmaFmtLayout "src1Fmt") #[("format", 0x002031), ("P", 1)]
  let wResFmt ← encodeWord (dmaFmtLayout "resFmt") #[("format", 0x302031), ("P", 1)]
  let wKernStage ← encodeWord kernelStageLayout #[
    ("sparse", if sparse then 1 else 0),
    ("stage", 1)
  ]
  let wChunkSlotVal ← encodeWord chunkSlotLayout #[("enable", 1), ("on20", 1), ("task", taskIdU32)]
  let wOut2 ← encodeWord out2Layout #[("form", 0xe0), ("task", taskIdU32)]

  let mut plain : Array RegWrite := #[
    -- Common block
    { byteAddr := regWin, value := wWin },
    { byteAddr := regHin, value := wHin },
    { byteAddr := regCin, value := cinU32 },
    { byteAddr := regWout, value := wWout },
    { byteAddr := regHout, value := wHout },
    { byteAddr := regCout, value := coutU32 },
    { byteAddr := regConvCfg, value := wConvCfg },
    { byteAddr := regTH, value := wTH },
    { byteAddr := regNeCommon, value := wNeCommon },
    { byteAddr := regNeTile, value := wNeTile },
    { byteAddr := regTargetCfg, value := targetCfgValue target },
    -- L2 staging block
    { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
    { byteAddr := regSrc2Cfg, value := src2CfgValue },
    { byteAddr := regSrc1Base, value := 0 },
    { byteAddr := regSrc1ChanStride, value := chsU32 },
    { byteAddr := regSrc1RowStride, value := srcRowU32 },
    { byteAddr := regSrc1RowBytesA, value := srcRowBytesU32 },
    { byteAddr := regSrc1RowBytesB, value := srcRowBytesU32 },
    { byteAddr := regResCfg, value := wResCfg },
    { byteAddr := regResBase, value := resBaseU32 },
    -- Neural Engine block
    { byteAddr := regNeCfg, value := wNeCfg },
    { byteAddr := regNeOp, value := wNeOp },
    { byteAddr := regNeScale, value := wNeScale },
    -- Tile DMA (Source) block
    { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
    { byteAddr := regSrc1DmaRowStride, value := inRowStrideU32 },
    { byteAddr := regSrc1DmaPlaneStride, value := inPlaneStrideU32 },
    { byteAddr := regSrc1Fmt, value := wSrc1Fmt },
    -- Result DMA block
    { byteAddr := regResDmaRowStride, value := outRowStrideU32 },
    { byteAddr := regResDmaPlaneStride, value := outPlaneStrideU32 },
    { byteAddr := regResFmt, value := wResFmt },
    -- Kernel DMA stage
    { byteAddr := regKernelStage, value := wKernStage },
    -- Output / cache stage block
    { byteAddr := regOut0, value := out0Value },
    { byteAddr := regOut1, value := out1Value },
    { byteAddr := regOut2, value := wOut2 },
    { byteAddr := regOut3, value := out3Value },
    { byteAddr := regOut4, value := out4Value },
    { byteAddr := regOut5, value := out5Value }
  ]

  -- True depthwise group register (written only when effMode == depthwise)
  if shape.effMode == .depthwise then
    let wDwGroups ← encodeWord dwGroupsLayout #[("perGroupIn", 1)]
    plain := plain.push { byteAddr := regDwGroups, value := wDwGroups }

  -- Staged mode result L2 registers (omitted in direct mode)
  if !direct then
    let dchsU32 ← natToU32Checked dchs "resChanStride"
    let resRowU32 ← natToU32Checked resRow "resRowStride"
    let resRowBytesU32 ← natToU32Checked (shape.cout * dchs) "resRowBytes"
    plain := plain ++ #[
      { byteAddr := regResChanStride, value := dchsU32 },
      { byteAddr := regResRowStride, value := resRowU32 },
      { byteAddr := regResRowBytesA, value := resRowBytesU32 },
      { byteAddr := regResRowBytesB, value := resRowBytesU32 }
    ]

  -- Scalar bias register (written only when effMode == dense && Cout == 1)
  if shape.effMode == .dense && shape.cout == 1 then
    let wScalarBias ← encodeWord scalarBiasLayout #[("val", packed.scalarBiasBits.toUInt32)]
    plain := plain.push { byteAddr := regScalarBias, value := wScalarBias }

  -- Active kernel chunk table slots (inactive slots omitted)
  for cs in packed.slots do
    let cOffU32 ← natToU32Checked cs.offset s!"chunkOffset[{cs.slot}]"
    let cSzU32 ← natToU32Checked cs.size s!"chunkSize[{cs.slot}]"
    plain := plain ++ #[
      { byteAddr := regChunkSlot cs.slot, value := wChunkSlotVal },
      { byteAddr := regChunkOffset cs.slot, value := cOffU32 },
      { byteAddr := regChunkSize cs.slot, value := cSzU32 }
    ]

  -- BAR-tagged sequential records in canonical order
  let wSrc1Dep ← encodeWord src1DepLayout #[("code", 0x21), ("task", taskIdU32)]
  let wResDep ← encodeWord resDepLayout #[
    ("code", resDepCode target),
    ("task", taskIdU32),
    ("direct", if direct then 1 else 0)
  ]
  let barRecs : Array BarRegWrite := #[
    { baseByteAddr := regKernelBaseLo, barSlot := 8, values := #[0, 0] },
    { baseByteAddr := regSrc1BaseLo, barSlot := 6, values := #[0, 0] },
    { baseByteAddr := regResBaseLo, barSlot := 7, values := #[0, 0] },
    { baseByteAddr := regSrc1Dep, barSlot := 4, values := #[wSrc1Dep] },
    { baseByteAddr := regResDep, barSlot := 4, values := #[wResDep] }
  ]

  pure {
    taskId := taskId
    logNE := tiling.logNE
    plainWrites := plain
    barWrites := barRecs
    kernelRelocs := #[{ barRecIndex := 0, payloadWordIndex := 0 }]
  }

/-- Unified compiler for single-task fp16 NE convolutions (`1x1` and `3x3`, dense and depthwise).
    Lowers registers via `lowerConvTask`, assembles the HWX and 4 LOAD sections via `emitProgram`,
    and runs the standalone decoded static safety verifier `Vane.Verify.verifyConvHwxTasks`. -/
def compileConv
    (target : Target)
    (shape : ConvSpec)
    (weights : ByteArray)
    (bias : Option ByteArray := none)
    (relu : Bool := false)
    (tilingOverride : Option Tiling := none) : Except String CompiledConv := do
  validateConvShape shape
  let tiling ← match tilingOverride with
    | some t => validateConvTiling target shape t; pure t
    | none => chooseConvTiling target shape

  let packed ← packConvKernelChunks target shape tiling weights bias
  let taskDesc ← lowerConvTask target shape tiling packed relu

  let inSurf ← makeFp16Surface target "x" "input_x" .input shape.cin shape.h shape.w
  let outSurf ← makeFp16Surface target "y" "output_y" .output shape.cout shape.outH shape.outW

  let emitted ← emitProgram {
    target := target
    surfaces := #[inSurf, outSurf]
    tasks := #[taskDesc]
    kernel := some packed.bytes
    scratch := none
    constants := zeroBytes textConstBytes
    ident := some "vane"
  }

  let _ ← verifyConvHwxTasks emitted.hwx emitted.info (requireConvTask := true)

  pure {
    target := target
    shape := shape
    tiling := tiling
    sparse := shape.isSparse
    wide3x3 := computeWide3x3Conv target shape tiling
    hasBias := bias.isSome
    relu := relu
    direct := isDirectConv target shape tiling
    l2UseBytes := l2UseBytesConv target shape tiling
    kernelBytes := packed.bytes.size
    hwx := emitted.hwx
    info := emitted.info
    sections := emitted.sections
  }

end Vane.Conv
