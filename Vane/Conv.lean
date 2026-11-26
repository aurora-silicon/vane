import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Hwx
import Vane.Sections
import Vane.Register
import Vane.Tiling
import Vane.Program
import Vane.Verify
import Vane.Lut
import Vane.View

namespace Vane.Conv

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Tiling
open Vane.Program
open Vane.Verify
open Vane.Lut
open Vane.View

/-- Fused post-convolution activation (`none`, hardware `relu`, or any 128-byte NE LUT). -/
inductive ConvActivation where
  | none
  | relu
  | lut (spec : LutSpec)
  deriving Repr, BEq, Inhabited

namespace ConvActivation

def name : ConvActivation → String
  | .none => "none"
  | .relu => "relu"
  | .lut spec => spec.name

def isRelu : ConvActivation → Bool
  | .relu => true
  | _ => false

def isLut : ConvActivation → Bool
  | .lut _ => true
  | _ => false

end ConvActivation

/-- Packed chunk slot metadata in `__KERN_0`. -/
structure KernelChunkSlot where
  slot : Nat
  offset : Nat
  size : Nat
  deriving Repr, BEq, Inhabited

/-- Result of packing fp16 weights, optional bias, and optional fused 128-byte LUT into `__KERN_0`. -/
structure PackedKernel where
  bytes : ByteArray
  slots : Array KernelChunkSlot
  biasInKernel : Bool
  scalarBiasBits : UInt16
  lutPayload : Option LutPayload := none
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
  activation : ConvActivation := .none
  lutBytes : Option ByteArray := none
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

/-- Compute `neCommon` bit 27 (`wide3x3`): `true` iff 3x3, target supports wide3x3, `effMode == dense`,
    `stride == 1`, and `maxBlockSlots <= 32768 * rep`. -/
def computeWide3x3Conv (target : Target) (sh : ConvSpec) (t : Tiling) : Bool :=
  if sh.kernelSize != 3 || !target.profile.supportsWide3x3 || sh.effMode != .dense || sh.stride != 1 then
    false
  else
    let rep := ((1 : Nat) <<< t.logNE) >>> t.f
    maxBlockSlotsConv sh t <= 32768 * rep

/-- Apply an optional LUT input shift (`rsqrt(eps)` / `hard_sigmoid(alpha, beta != 0.5)`)
    to convolution bias bytes, returning `(effectiveBias?, biasInKernel, scalarBiasBits)`. -/
private def resolveConvBias
    (sh : ConvSpec)
    (bias : Option ByteArray)
    (lutPayload? : Option LutPayload) : Except String (Option ByteArray × Bool × UInt16) := do
  if let some b := bias then
    let expectedBiasBytes := 2 * sh.cout
    if b.size != expectedBiasBytes then
      throw s!"bias byte size {b.size} != expected {expectedBiasBytes} (2 * Cout={sh.cout})"

  let scalarBiasMode := sh.effMode == .dense && sh.cout == 1
  let shiftBits? := lutPayload?.bind (fun lp => lp.inputShiftBits)

  let effBias? ← match shiftBits? with
    | none => pure bias
    | some shiftBits =>
      if !f16IsFinite shiftBits then
        throw s!"fused LUT input shift {toHex shiftBits.toNat} is not finite fp16"
      let shiftF := f16ToFloat shiftBits
      match bias with
      | some b =>
        let mut shifted := ByteArray.emptyWithCapacity b.size
        for o in [:sh.cout] do
          let orig ← readU16LE b (2 * o)
          let updated := floatToF16CanonicalZero (f16ToFloat orig + shiftF)
          if (updated &&& 0x7c00) == 0x7c00 then
            throw s!"fused LUT input shift overflowed fp16 on bias[{o}] (orig={toHex orig.toNat}, shift={toHex shiftBits.toNat})"
          shifted := pushU16LE shifted updated
        pure (some shifted)
      | none =>
        -- Synthesize constant shift across all channels (scalarBias if dense Cout=1, kernel bias otherwise)
        let mut synth := ByteArray.emptyWithCapacity (2 * sh.cout)
        for _ in [:sh.cout] do
          synth := pushU16LE synth shiftBits
        pure (some synth)

  let hasEffBias := effBias?.isSome
  let biasInKernel := hasEffBias && !scalarBiasMode
  let scalarBiasBits ←
    if scalarBiasMode then
      match effBias? with
      | some b => readU16LE b 0
      | none => pure (0 : UInt16)
    else
      pure (0 : UInt16)
  pure (effBias?, biasInKernel, scalarBiasBits)

/-- Unified fp16 kernel weight, optional bias, and optional fused 128-byte LUT packer
    for 1x1 dense, 3x3 dense stride-1, 3x3 depthwise stride-1, and 3x3 stride-2 sparse convolutions.
    Enforces the hardware 64 KiB per-chunk size cap on every active chunk. -/
def packConvKernelChunks
    (target : Target)
    (sh : ConvSpec)
    (tiling : Tiling)
    (weights : ByteArray)
    (bias : Option ByteArray)
    (lutPayload? : Option LutPayload := none) : Except String PackedKernel := do
  if let some lp := lutPayload? then
    if lp.bytes.size != lutTableBytes then
      throw s!"fused LUT table byte size {lp.bytes.size} != expected {lutTableBytes}"

  let (effBias?, biasInKernel, scalarBiasBits) ← resolveConvBias sh bias lutPayload?
  let extraLutBytes := if lutPayload?.isSome then lutTableBytes else 0
  validateConvSafety target sh tiling extraLutBytes biasInKernel

  let cinW := sh.cinW
  let taps := sh.taps
  let expectedWeightBytes := 2 * sh.cout * cinW * taps
  if weights.size != expectedWeightBytes then
    throw s!"weights byte size {weights.size} != expected {expectedWeightBytes} (2 * Cout={sh.cout} * cinW={cinW} * taps={taps})"

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
      -- Prepend the exact 128-byte LUT table in front of every active kernel chunk when fused
      let mut chunk := match lutPayload? with
        | some lp => lp.bytes
        | none => ByteArray.empty
      if !sparse then
        -- Dense uncompressed kernel encoding (1x1 or 3x3 stride-1)
        for (o0, n) in blocks do
          if biasInKernel then
            match effBias? with
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
            match effBias? with
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

      if target.profile.replicateKernelChunkSlots then
        for j in [:rep] do
          let s := g * rep + j
          if s >= 16 then
            throw s!"chunk slot index {s} exceeds 0..15"
          slotChunks := slotChunks.push (s, chunk)
          totalKernelBytes := totalKernelBytes + chunk.size
      else
        let s := g * rep
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
    lutPayload := lutPayload?
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
  let hasLut := packed.lutPayload.isSome
  if hasLut && relu then
    throw "lowerConvTask: cannot enable both relu and fused LUT activation"
  let neScaleVal : UInt32 := match packed.lutPayload with
    | some lp => lp.preScaleBits.toUInt32
    | none => fp16OneBits

  let wNeCfg ← encodeWord (neCfgLayout target) #[("sparse", if sparse then 1 else 0)]
  let wNeOp ← encodeWord neOpLayout #[
    ("biasInKernel", if packed.biasInKernel then 1 else 0),
    ("relu", if relu && !hasLut then 1 else 0),
    ("lut", if hasLut then 1 else 0)
  ]
  let wNeScale ← encodeWord neScaleLayout #[("val", neScaleVal)]
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

/-- Unified compiler for single-task fp16 NE convolutions (`1x1` and `3x3`, dense and depthwise),
    with optional fused ReLU or 128-byte NE LUT activation.
    Lowers registers via `lowerConvTask`, assembles the HWX and 4 LOAD sections via `emitProgram`,
    and runs the standalone decoded static safety verifier `Vane.Verify.verifyHwxTasks`. -/
def compileConv
    (target : Target)
    (shape : ConvSpec)
    (weights : ByteArray)
    (bias : Option ByteArray := none)
    (relu : Bool := false)
    (tilingOverride : Option Tiling := none)
    (act : ConvActivation := .none) : Except String CompiledConv := do
  validateConvShape shape

  let effectiveAct ← match act with
    | .none => pure (if relu then ConvActivation.relu else ConvActivation.none)
    | .relu => pure ConvActivation.relu
    | .lut spec =>
      if relu then
        throw s!"compileConv: cannot pass both relu=true and fused LUT activation '{spec.name}'"
      pure (ConvActivation.lut spec)

  let lutPayload? ← match effectiveAct with
    | .lut spec => some <$> buildLutTable spec
    | _ => pure none

  let (_, willHaveBiasInKernel, _) ← resolveConvBias shape bias lutPayload?
  let extraLutBytes := if lutPayload?.isSome then lutTableBytes else 0

  let tiling ← match tilingOverride with
    | some t => validateConvTiling target shape t extraLutBytes willHaveBiasInKernel; pure t
    | none => chooseConvTiling target shape extraLutBytes willHaveBiasInKernel

  let packed ← packConvKernelChunks target shape tiling weights bias lutPayload?
  let taskDesc ← lowerConvTask target shape tiling packed effectiveAct.isRelu

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

  let _ ← verifyHwxTasks emitted.hwx emitted.info (requireKnownTask := true)

  pure {
    target := target
    shape := shape
    tiling := tiling
    sparse := shape.isSparse
    wide3x3 := computeWide3x3Conv target shape tiling
    hasBias := bias.isSome
    relu := effectiveAct.isRelu
    activation := effectiveAct
    lutBytes := lutPayload?.map (fun lp => lp.bytes)
    direct := isDirectConv target shape tiling
    l2UseBytes := l2UseBytesConv target shape tiling
    kernelBytes := packed.bytes.size
    hwx := emitted.hwx
    info := emitted.info
    sections := emitted.sections
  }

-- ============================================================================
-- Large-Channel Output-Split 1x1 Convolution / Linear
-- ============================================================================

/-- Complete compilation artifact for single-task or output-channel-split `conv1x1` / `linear`. -/
structure CompiledConv1x1Split where
  target : Target
  shape : Conv1x1Shape
  slices : Array Nat
  tilings : Array Tiling
  hasBias : Bool
  relu : Bool
  activation : ConvActivation := .none
  lutBytes : Option ByteArray := none
  direct : Bool
  l2UseBytes : Nat
  kernelBytes : Nat
  hwx : ByteArray
  info : HwxInfo
  sections : LoadSections

/-- Choose contiguous output-channel slice widths partitioning `cout` for `conv1x1` / `linear`.
    Returns `#[cout]` whenever single-task `chooseConvTiling` succeeds (preserving 100% bit-for-bit
    parity with existing single-task `conv1x1`). Otherwise finds the largest legal per-slice width
    (preferring multiples of 32, then 16, then 8, then any `1 <= s < cout`) such that every slice
    has a legal single-task tiling and `<= 64 KiB` kernel chunks, within the 255-task kernel limit. -/
def chooseConv1x1OutputSlices
    (target : Target) (h w cin cout : Nat)
    (extraChunkBytes : Nat := 0) (hasBias : Bool := false) : Except String (Array Nat) := do
  let fullSpec : ConvSpec := { kernelSize := 1, mode := .dense, h := h, w := w, cin := cin, cout := cout }
  validateConvShape fullSpec
  let canFit := fun (dc : Nat) =>
    let sh : ConvSpec := { kernelSize := 1, mode := .dense, h := h, w := w, cin := cin, cout := dc }
    let biasInK := hasBias && dc > 1
    (chooseConvTiling target sh extraChunkBytes biasInK).isOk

  if canFit cout then
    return #[cout]

  let maxSingle := min (cout - 1) (target.profile.conv1x1MaxCout 4 4)
  if maxSingle == 0 then
    throw s!"no legal {target.name} output-channel split for conv1x1 ({h}x{w}, {cin}->{cout})"

  let tryWidth := fun (s : Nat) =>
    if s == 0 || s >= cout then none
    else
      let numSlices := ceilDiv cout s
      if numSlices > 255 then none
      else
        let rem := cout % s
        if canFit s && (rem == 0 || canFit rem) then
          let fullCount := cout / s
          let base := Array.replicate fullCount s
          some (if rem == 0 then base else base.push rem)
        else
          none

  -- Prefer multiples of 32, then 16, then 8, then any 1 <= s <= maxSingle
  let chosen? : Option (Array Nat) :=
    findFirstSome [32, 16, 8, 1] fun alignStep =>
      let top := (maxSingle / alignStep) * alignStep
      let count := top / alignStep
      findFirstSome (natRange 0 count) fun i =>
        let s := top - i * alignStep
        tryWidth s

  match chosen? with
  | some slices => pure slices
  | none => throw s!"no legal {target.name} output-channel split for conv1x1 ({h}x{w}, {cin}->{cout})"

/-- Lower one slice of a multi-task output-channel-split `conv1x1` into a `TaskDesc`. -/
private def lowerConv1x1SliceTask
    (target : Target)
    (shape : ConvSpec)
    (tiling : Tiling)
    (packed : PackedKernel)
    (relu : Bool)
    (taskId totalTasks : Nat)
    (kernBaseOff outOff : Nat)
    (waitNe5 : Bool) : Except String TaskDesc := do
  let isFirst := taskId == 0
  let isLast := taskId + 1 == totalTasks
  let direct := isDirectConv target shape tiling
  let chs := chsBytesConv shape tiling
  let dchs := dchsBytesConv shape tiling
  let srcRow := srcRowStrideConv shape tiling
  let resRow := resRowStrideConv shape tiling
  let resBase := resBaseBytesConv shape tiling

  let rowStride := alignUp (2 * shape.w) dmaLineBytes
  let planeStride := shape.h * rowStride

  let taskIdU32 ← natToU32Checked taskId "taskId"
  let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked shape.w "Win")]
  let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked shape.h "Hin")]
  let cinU32 ← natToU32Checked shape.cin "Cin"
  let coutU32 ← natToU32Checked shape.cout "Cout"
  let wConvCfg ← encodeWord convCfgLayout #[
    ("kw", 1), ("kh", 1), ("sx", 1), ("sy", 1),
    ("pl", 0), ("pt", 0), ("hx", 1), ("vy", 1)
  ]
  let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked tiling.tileHeight "TH")]
  let wNeCommon ← encodeWord neCommonLayout #[
    ("kind", 0),
    ("shPref", ← natToU32Checked tiling.shPref "shPref"),
    ("shMin", ← natToU32Checked tiling.shPref "shMin"),
    ("shMax", ← natToU32Checked tiling.shMax "shMax"),
    ("F", ← natToU32Checked tiling.f "F"),
    ("wide3x3", 0)
  ]
  let wNeTile ← encodeWord neTileLayout #[
    ("ocg", ← natToU32Checked tiling.ocg "ocg"),
    ("fat", ← natToU32Checked tiling.fat "fat"),
    ("WU", ← natToU32Checked tiling.wu "WU")
  ]
  let wSrc1Cfg ← encodeWord (portCfgMultiLayout "src1Cfg") #[
    ("mode", portModeDma),
    ("handoffNe", 0),
    ("handoffPe", 0),
    ("waitNe4", if isFirst then 1 else 0),
    ("waitNe5", if isFirst || waitNe5 then 1 else 0),
    ("dmaFlag", 1),
    ("P", 1),
    ("waitPe20", if isFirst then 1 else 0),
    ("waitPe22", if isFirst then 1 else 0)
  ]
  let wResCfg ← encodeWord resCfgMultiLayout #[
    ("mode", portModeDma),
    ("direct", if direct then 1 else 0),
    ("waitNe4", if isFirst then 1 else 0),
    ("waitNe5", if isFirst then 1 else 0),
    ("dmaFlag", 1),
    ("P", 1),
    ("waitPe20", if isFirst then 1 else 0),
    ("waitPe22", if isFirst then 1 else 0)
  ]
  let hasLut := packed.lutPayload.isSome
  if hasLut && relu then
    throw "lowerConv1x1SliceTask: cannot enable both relu and fused LUT activation"
  let neScaleVal : UInt32 := match packed.lutPayload with
    | some lp => lp.preScaleBits.toUInt32
    | none => fp16OneBits
  let wNeCfg ← encodeWord (neCfgLayout target) #[("sparse", 0)]
  let wNeOp ← encodeWord neOpLayout #[
    ("biasInKernel", if packed.biasInKernel then 1 else 0),
    ("relu", if relu && !hasLut then 1 else 0),
    ("lut", if hasLut then 1 else 0)
  ]
  let wNeScale ← encodeWord neScaleLayout #[("val", neScaleVal)]
  let wKernStage ← encodeWord kernelStageMultiLayout #[
    ("sparse", 0),
    ("kernEn", 1),
    ("isLast", if isLast then 1 else 0),
    ("kernCfg", 1),
    ("stage", ← natToU32Checked (taskId + 1) "kernelStage.stage")
  ]

  let mut plain : Array RegWrite := #[
    { byteAddr := regWin, value := wWin },
    { byteAddr := regHin, value := wHin },
    { byteAddr := regCin, value := cinU32 },
    { byteAddr := regWout, value := wWin },
    { byteAddr := regHout, value := wHin },
    { byteAddr := regCout, value := coutU32 },
    { byteAddr := regConvCfg, value := wConvCfg },
    { byteAddr := regTH, value := wTH },
    { byteAddr := regNeCommon, value := wNeCommon },
    { byteAddr := regNeTile, value := wNeTile },
    { byteAddr := regTargetCfg, value := targetCfgValue target },
    { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
    { byteAddr := regSrc1Base, value := 0 },
    { byteAddr := regSrc1ChanStride, value := ← natToU32Checked chs "src1ChanStride" },
    { byteAddr := regSrc1RowStride, value := ← natToU32Checked srcRow "src1RowStride" },
    { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked (shape.cin * chs) "src1RowBytesA" },
    { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked (shape.cin * chs) "src1RowBytesB" },
    { byteAddr := regResCfg, value := wResCfg },
    { byteAddr := regResBase, value := ← natToU32Checked resBase "resBase" },
    { byteAddr := regNeCfg, value := wNeCfg },
    { byteAddr := regNeOp, value := wNeOp },
    { byteAddr := regNeScale, value := wNeScale },
    { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
    { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked rowStride "src1DmaRowStride" },
    { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked planeStride "src1DmaPlaneStride" },
    { byteAddr := regSrc1Fmt, value := srcFmtWordP1 },
    { byteAddr := regResDmaRowStride, value := ← natToU32Checked rowStride "resDmaRowStride" },
    { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked planeStride "resDmaPlaneStride" },
    { byteAddr := regResFmt, value := resFmtWordP1 },
    { byteAddr := regKernelStage, value := wKernStage }
  ]

  if isFirst then
    plain := plain.push { byteAddr := regSrc2Cfg, value := src2CfgValue }

  if !direct then
    let resRowBytesU32 ← natToU32Checked (shape.cout * dchs) "resRowBytes"
    plain := plain ++ #[
      { byteAddr := regResChanStride, value := ← natToU32Checked dchs "resChanStride" },
      { byteAddr := regResRowStride, value := ← natToU32Checked resRow "resRowStride" },
      { byteAddr := regResRowBytesA, value := resRowBytesU32 },
      { byteAddr := regResRowBytesB, value := resRowBytesU32 }
    ]

  if shape.cout == 1 then
    let wScalarBias ← encodeWord scalarBiasLayout #[("val", packed.scalarBiasBits.toUInt32)]
    plain := plain.push { byteAddr := regScalarBias, value := wScalarBias }

  let wSlotOn ← encodeWord chunkSlotLayout #[("enable", 1), ("on20", 1), ("task", taskIdU32)]
  let wSlotOff ← encodeWord chunkSlotLayout #[("enable", 0), ("on20", 1), ("task", taskIdU32)]
  for s in [:16] do
    match packed.slots.find? (fun cs => cs.slot == s) with
    | some cs =>
      plain := plain ++ #[
        { byteAddr := regChunkSlot s, value := wSlotOn },
        { byteAddr := regChunkOffset s, value := ← natToU32Checked cs.offset s!"chunkOffset[{s}]" },
        { byteAddr := regChunkSize s, value := ← natToU32Checked cs.size s!"chunkSize[{s}]" }
      ]
    | none =>
      if !isFirst then
        plain := plain.push { byteAddr := regChunkSlot s, value := wSlotOff }

  if isLast then
    let wOut2 ← encodeWord out2Layout #[("form", out2FormValue), ("task", taskIdU32)]
    plain := plain ++ #[
      { byteAddr := regOut0, value := out0Value },
      { byteAddr := regOut1, value := out1Value },
      { byteAddr := regOut2, value := wOut2 },
      { byteAddr := regOut3, value := out3Value },
      { byteAddr := regOut4, value := out4Value },
      { byteAddr := regOut5, value := out5Value }
    ]

  let wSrc1Dep ← encodeWord (srcDepMultiLayout "src1Dep") #[
    ("code", srcDepCodeDma), ("task", taskIdU32), ("depDist", 0), ("depKind", 0)
  ]
  let wResDep ← encodeWord resDepLayout #[
    ("code", resDepCode target),
    ("task", taskIdU32),
    ("direct", if direct then 1 else 0)
  ]
  let barRecs : Array BarRegWrite := #[
    { baseByteAddr := regKernelBaseLo, barSlot := 8, values := #[← natToU32Checked kernBaseOff "kernelBaseLo", 0] },
    { baseByteAddr := regSrc1BaseLo, barSlot := 6, values := #[0, 0] },
    { baseByteAddr := regResBaseLo, barSlot := 7, values := #[← natToU32Checked outOff "resBaseLo", 0] },
    { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrc1Dep] },
    { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
  ]

  pure {
    taskId := taskId
    logNE := tiling.logNE
    eneOverride := some (tiling.logNE + 1)
    isFirst := isFirst
    isLast := isLast
    liveL2Mask := 0
    plainWrites := plain
    barWrites := barRecs
    kernelRelocs := #[{ barRecIndex := 0, payloadWordIndex := 0 }]
  }

/-- Compile a standalone `conv1x1` / `linear` layer, automatically partitioning `Cout` across
    multiple NE tasks writing disjoint output-channel slices into the shared output surface
    whenever single-task `Cout` or 64 KiB kernel-chunk limits are exceeded.
    Preserves 100% bit-for-bit parity with `compileConv` when a single task suffices. -/
def compileConv1x1Split
    (target : Target)
    (shape : Conv1x1Shape)
    (weights : ByteArray)
    (bias : Option ByteArray := none)
    (relu : Bool := false)
    (tilingOverride : Option Tiling := none)
    (act : ConvActivation := .none) : Except String CompiledConv1x1Split := do
  let fullSpec := shape.toConvSpec
  validateConvShape fullSpec

  let effectiveAct ← match act with
    | .none => pure (if relu then ConvActivation.relu else ConvActivation.none)
    | .relu => pure ConvActivation.relu
    | .lut spec =>
      if relu then
        throw s!"compileConv1x1Split: cannot pass both relu=true and fused LUT activation '{spec.name}'"
      pure (ConvActivation.lut spec)

  let lutPayload? ← match effectiveAct with
    | .lut spec => some <$> buildLutTable spec
    | _ => pure none

  let extraLutBytes := if lutPayload?.isSome then lutTableBytes else 0
  let hasEffBias := bias.isSome || (lutPayload?.bind (fun lp => lp.inputShiftBits)).isSome

  if tilingOverride.isSome then
    let c ← compileConv target fullSpec weights bias relu tilingOverride act
    return {
      target := c.target, shape := shape, slices := #[shape.cout], tilings := #[c.tiling],
      hasBias := c.hasBias, relu := c.relu, activation := c.activation, lutBytes := c.lutBytes,
      direct := c.direct, l2UseBytes := c.l2UseBytes, kernelBytes := c.kernelBytes,
      hwx := c.hwx, info := c.info, sections := c.sections
    }

  let slices ← chooseConv1x1OutputSlices target shape.h shape.w shape.cin shape.cout extraLutBytes hasEffBias
  if slices.size == 1 then
    let c ← compileConv target fullSpec weights bias relu none act
    return {
      target := c.target, shape := shape, slices := slices, tilings := #[c.tiling],
      hasBias := c.hasBias, relu := c.relu, activation := c.activation, lutBytes := c.lutBytes,
      direct := c.direct, l2UseBytes := c.l2UseBytes, kernelBytes := c.kernelBytes,
      hwx := c.hwx, info := c.info, sections := c.sections
    }

  -- Multi-task output-channel split
  let expWBytes := 2 * shape.cout * shape.cin
  if weights.size != expWBytes then
    throw s!"weights byte size {weights.size} != expected {expWBytes} (2 * Cout={shape.cout} * Cin={shape.cin})"
  if let some b := bias then
    if b.size != 2 * shape.cout then
      throw s!"bias byte size {b.size} != expected {2 * shape.cout} (2 * Cout={shape.cout})"

  let k := slices.size
  let starts := prefixStarts slices.toList
  let outRowStride := alignUp (2 * shape.w) dmaLineBytes
  let outPlaneStride := shape.h * outRowStride

  let mut packedSlices : Array (Nat × PackedKernel) := Array.mkEmpty k
  let mut tilings : Array Tiling := Array.mkEmpty k
  let mut taskDescs : Array TaskDesc := Array.mkEmpty k
  let mut prevResRanges : Array (Nat × Nat) := Array.mkEmpty k
  let mut peakL2 : Nat := 0
  let mut allDirect := true
  let mut totalKernBytes : Nat := 0

  for i in [:k] do
    let c0 := starts[i]!
    let dc := slices[i]!
    let sh_i : ConvSpec := {
      kernelSize := 1, mode := .dense, h := shape.h, w := shape.w, cin := shape.cin, cout := dc
    }
    let wSlice := weights.extract (2 * c0 * shape.cin) (2 * (c0 + dc) * shape.cin)
    let bSlice? := bias.map (fun b => b.extract (2 * c0) (2 * (c0 + dc)))
    let biasInK := hasEffBias && dc > 1
    let t_i ← chooseConvTiling target sh_i extraLutBytes biasInK
    let packed_i ← packConvKernelChunks target sh_i t_i wSlice bSlice? lutPayload?

    let kernOff := alignUp totalKernBytes dmaLineBytes
    totalKernBytes := kernOff + packed_i.bytes.size
    packedSlices := packedSlices.push (kernOff, packed_i)

    let srcEnd := srcL2BytesConv sh_i t_i
    let resLo := resBaseBytesConv sh_i t_i
    let resHi := resLo + resL2BytesConv target sh_i t_i
    peakL2 := max peakL2 resHi
    if !isDirectConv target sh_i t_i then
      allDirect := false

    -- Set bit 5 (waitNe5) whenever src1Stage [0, srcEnd) overlaps any earlier slice's resStage [rLo, rHi)
    let haz := i > 0 && prevResRanges.any (fun (rLo, rHi) => rLo < srcEnd && 0 < rHi)
    prevResRanges := prevResRanges.push (resLo, resHi)

    let outOff := c0 * outPlaneStride
    let td ← lowerConv1x1SliceTask target sh_i t_i packed_i effectiveAct.isRelu i k kernOff outOff haz
    tilings := tilings.push t_i
    taskDescs := taskDescs.push td

  -- Single linear pass copy into pre-sized kernel buffer (avoids O(k^2) ByteArray.append)
  let mut kernBuf := zeroBytes totalKernBytes
  for (kernOff, packed_i) in packedSlices do
    kernBuf := ByteArray.copySlice packed_i.bytes 0 kernBuf kernOff packed_i.bytes.size true

  let inSurf ← makeFp16Surface target "x" "input_x" .input shape.cin shape.h shape.w
  let outSurf ← makeFp16Surface target "y" "output_y" .output shape.cout shape.h shape.w

  let emitted ← emitProgram {
    target := target
    surfaces := #[inSurf, outSurf]
    tasks := taskDescs
    kernel := some kernBuf
    scratch := none
    constants := zeroBytes textConstBytes
    ident := some "vane"
  }

  let summary ← verifyHwxTasksSummary emitted.hwx emitted.info (requireKnownTask := true)
  if summary.neConv != k || summary.total != k then
    throw s!"internal error: verifier checked {summary.neConv}/{summary.total} split conv1x1 tasks != {k}"

  pure {
    target := target
    shape := shape
    slices := slices
    tilings := tilings
    hasBias := bias.isSome
    relu := effectiveAct.isRelu
    activation := effectiveAct
    lutBytes := lutPayload?.map (fun lp => lp.bytes)
    direct := allDirect
    l2UseBytes := peakL2
    kernelBytes := kernBuf.size
    hwx := emitted.hwx
    info := emitted.info
    sections := emitted.sections
  }

end Vane.Conv
