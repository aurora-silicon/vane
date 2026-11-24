import Vane.Binary
import Vane.TaskStream
import Vane.Hwx
import Vane.Sections
import Vane.Register
import Vane.Tiling
import Vane.Conv1x1

namespace Vane.Conv3x3

open Vane.Binary
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Tiling
open Vane.Conv1x1

/-- Complete compilation artifact for a single-task fp16 3x3 convolution (dense or depthwise). -/
structure CompiledConv3x3 where
  target : Target
  shape : Conv3x3Shape
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

/-- Encode a slot stream (`Array (Option UInt16)`) into the zero-skipping sparse binary format:
    pads to a multiple of 16 slots, encodes each 16-slot vector as `(mask : u16, kept weights...)`,
    and prefixes with the 3-byte little-endian `u24` body byte length. -/
def encodeSparseStream (slots : Array (Option UInt16)) : Except String ByteArray := do
  let paddedLen := alignUp slots.size 16
  let numVecs := paddedLen / 16
  let mut bodyHalfwords : Array UInt16 := #[]

  for v in [:numVecs] do
    let base := v * 16
    let mut mask : UInt16 := 0
    let mut kept : Array UInt16 := #[]
    for i in [:16] do
      let idx := base + i
      if idx < slots.size then
        match slots[idx]! with
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

  let mut out := ByteArray.empty
  out := out.push (bodyBytes &&& 0xff).toUInt8
  out := out.push ((bodyBytes >>> 8) &&& 0xff).toUInt8
  out := out.push ((bodyBytes >>> 16) &&& 0xff).toUInt8
  for hw in bodyHalfwords do
    out := pushU16LE out hw
  pure out

/-- Decode a zero-skipping sparse stream starting at `offset` in `bs`.
    Returns `(decodedSlots, totalBytesConsumed)` where `decodedSlots.size` is a multiple of 16
    and skipped/padding slots decode to `0x0000`. -/
def decodeSparseStream (bs : ByteArray) (offset : Nat := 0) : Except String (Array UInt16 × Nat) := do
  if offset + 3 > bs.size then
    throw s!"sparse stream truncated at u24 length header (offset={offset}, size={bs.size})"
  let b0 := (bs.get! offset).toNat
  let b1 := (bs.get! (offset + 1)).toNat
  let b2 := (bs.get! (offset + 2)).toNat
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
-- wide3x3 condition and 3x3 kernel chunk packer
-- ============================================================================

/-- Compute `neCommon` bit 27 (`wide3x3`): `true` iff `target == h17`, `effMode == dense`,
    `stride == 1`, and `maxBlockSlots <= 32768 * rep`. -/
def computeWide3x3 (target : Target) (sh : Conv3x3Shape) (t : Tiling) : Bool :=
  if target != .h17 || sh.effMode != .dense || sh.stride != 1 then
    false
  else
    let rep := ((1 : Nat) <<< t.logNE) >>> t.f
    maxBlockSlots3x3 sh t <= 32768 * rep

/-- Pack fp16 3x3 weights (`(Cout, Cin, 3, 3)` for dense or `(C, 1, 3, 3)` for depthwise)
    and optional bias `(Cout,)` into `__KERN_0` chunk slots. -/
def packConv3x3KernelChunks
    (target : Target)
    (sh : Conv3x3Shape)
    (tiling : Tiling)
    (weights : ByteArray)
    (bias : Option ByteArray) : Except String PackedKernel := do
  validateConv3x3Tiling target sh tiling

  let cinW := if sh.effMode == .depthwise then 1 else sh.cin
  let expectedWeightBytes := 2 * sh.cout * cinW * 9
  if weights.size != expectedWeightBytes then
    throw s!"weights byte size {weights.size} != expected {expectedWeightBytes} (2 * Cout={sh.cout} * cinW={cinW} * 9)"

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
  let mut kernelBuf := ByteArray.empty
  let mut slots : Array KernelChunkSlot := #[]

  for g in [:groups] do
    let blocks := groupOutputBlocks3x3 sh tiling g
    if !blocks.isEmpty then
      let mut chunk := ByteArray.empty
      if !sparse then
        -- Stride 1: dense uncompressed kernel encoding
        for (o0, n) in blocks do
          if biasInKernel then
            let b := bias.get!
            for k in [:n] do
              let bOff := 2 * (o0 + k)
              chunk := chunk.push (b.get! bOff)
              chunk := chunk.push (b.get! (bOff + 1))
          for ci in [:cinW] do
            for tp in [:9] do
              for k in [:n] do
                let o := o0 + k
                let wOff := 2 * ((o * cinW + ci) * 9 + tp)
                chunk := chunk.push (weights.get! wOff)
                chunk := chunk.push (weights.get! (wOff + 1))
      else
        -- Stride 2: zero-skipping sparse kernel encoding
        for (o0, n) in blocks do
          if biasInKernel then
            let b := bias.get!
            for k in [:n] do
              let bOff := 2 * (o0 + k)
              chunk := chunk.push (b.get! bOff)
              chunk := chunk.push (b.get! (bOff + 1))
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
          let encoded ← encodeSparseStream slotStream
          chunk := chunk ++ encoded

      chunk := padToAlign chunk dmaLineBytes
      if chunk.size > maxKernelChunkBytes then
        throw s!"3x3 kernel group {g} chunk size {chunk.size}B exceeds maximum {maxKernelChunkBytes}B (64 KiB)"

      match target with
      | .h17 =>
        let s := g * rep
        if s >= 16 then
          throw s!"chunk slot index {s} exceeds 0..15"
        let off := kernelBuf.size
        kernelBuf := kernelBuf ++ chunk
        slots := slots.push { slot := s, offset := off, size := chunk.size }
      | .h16 =>
        for j in [:rep] do
          let s := g * rep + j
          if s >= 16 then
            throw s!"chunk slot index {s} exceeds 0..15"
          let off := kernelBuf.size
          kernelBuf := kernelBuf ++ chunk
          slots := slots.push { slot := s, offset := off, size := chunk.size }

  if slots.isEmpty || kernelBuf.isEmpty then
    throw "3x3 kernel packer produced 0 active chunk slots"

  pure {
    bytes := kernelBuf
    slots := slots
    biasInKernel := biasInKernel
    scalarBiasBits := scalarBiasBits
  }

-- ============================================================================
-- Post-emission static task-legality verifier (1x1 and 3x3 unified)
-- ============================================================================

private def findDecodedWrite (writes : Array DecodedWrite) (addr : Nat) : Option DecodedWrite :=
  writes.find? (fun w => w.byteAddr == addr)

private def requireDecodedWrite (writes : Array DecodedWrite) (addr : Nat) (label : String) : Except String DecodedWrite :=
  match findDecodedWrite writes addr with
  | some w => pure w
  | none => throw s!"missing required register {label} ({toHex addr})"

private def resolveBarSlotVmBase (info : HwxInfo) (slot : UInt8) : Except String UInt64 := do
  if slot.toNat > 63 then
    throw s!"BAR slot {slot} out of range 0..63"
  readU64LE info.programDescBytes (0x10 * (slot.toNat + 1))

/-- Determine `PadMode` uniquely from full uncropped `(H, W)`, `stride`, and output `(Ho, Wo)`. -/
private def inferPadMode (h w stride hout wout pt pl : Nat) : Except String PadMode := do
  let sameShape : Conv3x3Shape := {
    mode := .dense, h := h, w := w, cin := 1, cout := 1, stride := stride, pad := .same
  }
  if sameShape.outH == hout && sameShape.outW == wout && sameShape.padTop == pt && sameShape.padLeft == pl then
    return .same
  if h >= 3 && w >= 3 then
    let validShape : Conv3x3Shape := {
      mode := .dense, h := h, w := w, cin := 1, cout := 1, stride := stride, pad := .valid
    }
    if validShape.outH == hout && validShape.outW == wout && validShape.padTop == pt && validShape.padLeft == pl then
      return .valid
  throw s!"3x3 spatial geometry mismatch: input=({h}x{w}), stride={stride}, output=({hout}x{wout}), pad=(pt={pt}, pl={pl}) matches neither same nor valid"

/-- Statically verify one decoded NE 3x3 conv task (dense or depthwise, stride 1 or 2)
    against hardware register bitfields, geometry, tiling legality, 2 MiB L2 capacity/disjointness,
    sparse/wide3x3/dwGroups invariants, chunk-slot invariants, and DMA surface bounds. -/
def verifyConv3x3Task (info : HwxInfo) (task : DecodedTask) : Except String Unit := do
  let target := info.target
  let prefixMsg := s!"task {task.taskIndex}"

  -- 1. Verify 9-word header and extract logNE
  if task.header.size != 9 then
    throw s!"{prefixMsg}: expected 9-word header, got {task.header.size}"
  let ene ← decodeFieldChecked taskHeader8Layout task.header[8]! "ene"
  if ene < 1 || ene > 5 then
    throw s!"{prefixMsg}: header[8] ene={ene} out of range 1..5"
  let logNE := ene.toNat - 1

  -- 2. Reject duplicate register writes within the same task
  let mut seenAddrs : Array Nat := #[]
  for w in task.writes do
    if seenAddrs.contains w.byteAddr then
      throw s!"{prefixMsg}: duplicate write to register {toHex w.byteAddr}"
    seenAddrs := seenAddrs.push w.byteAddr

  -- 3. Decode common shape & convCfg registers
  let wWin ← requireDecodedWrite task.writes regWin "Win"
  let wHin ← requireDecodedWrite task.writes regHin "Hin"
  let wCin ← requireDecodedWrite task.writes regCin "Cin"
  let wWout ← requireDecodedWrite task.writes regWout "Wout"
  let wHout ← requireDecodedWrite task.writes regHout "Hout"
  let wCout ← requireDecodedWrite task.writes regCout "Cout"
  let wConvCfg ← requireDecodedWrite task.writes regConvCfg "convCfg"
  let wTH ← requireDecodedWrite task.writes regTH "TH"
  let wNeCommon ← requireDecodedWrite task.writes regNeCommon "neCommon"
  let wNeTile ← requireDecodedWrite task.writes regNeTile "neTile"
  let wTargetCfg ← requireDecodedWrite task.writes regTargetCfg "targetCfg"

  let winReg := (← decodeFieldChecked (dim16Layout "Win") wWin.value "val").toNat
  let hinReg := (← decodeFieldChecked (dim16Layout "Hin") wHin.value "val").toNat
  let cin := wCin.value.toNat
  let wout := (← decodeFieldChecked (dim16Layout "Wout") wWout.value "val").toNat
  let hout := (← decodeFieldChecked (dim16Layout "Hout") wHout.value "val").toNat
  let cout := wCout.value.toNat
  let th := (← decodeFieldChecked (dim16Layout "TH") wTH.value "val").toNat

  let kw := (← decodeFieldChecked convCfgLayout wConvCfg.value "kw").toNat
  let kh := (← decodeFieldChecked convCfgLayout wConvCfg.value "kh").toNat
  let sx := (← decodeFieldChecked convCfgLayout wConvCfg.value "sx").toNat
  let sy := (← decodeFieldChecked convCfgLayout wConvCfg.value "sy").toNat
  let pl := (← decodeFieldChecked convCfgLayout wConvCfg.value "pl").toNat
  let pt := (← decodeFieldChecked convCfgLayout wConvCfg.value "pt").toNat
  let hx := (← decodeFieldChecked convCfgLayout wConvCfg.value "hx").toNat
  let vy := (← decodeFieldChecked convCfgLayout wConvCfg.value "vy").toNat

  if kw != 3 || kh != 3 then
    throw s!"{prefixMsg}: convCfg kw={kw}, kh={kh} is not a 3x3 convolution"
  if sx != sy || (sx != 1 && sx != 2) then
    throw s!"{prefixMsg}: convCfg strides sx={sx}, sy={sy} must be equal and 1 or 2"
  if hx != 1 || vy != 1 then
    throw s!"{prefixMsg}: convCfg hx={hx}, vy={vy} must both be 1"
  let stride := sx

  if wTargetCfg.value != targetCfgValue target then
    throw s!"{prefixMsg}: targetCfg {toHex wTargetCfg.value.toNat} != expected {toHex (targetCfgValue target).toNat}"

  -- 4. Resolve bound input/output surfaces early to verify uncropped vs cropped spatial extents
  let wSrcBaseLo ← requireDecodedWrite task.writes regSrc1BaseLo "src1BaseLo"
  let some inBarSlot := wSrcBaseLo.barSlot
    | throw s!"{prefixMsg}: src1BaseLo (0x4d10) must carry a BAR slot"
  let inBarVm ← resolveBarSlotVmBase info inBarSlot
  let some inSurf := info.surfaces.find? (fun s => s.sect.addr == inBarVm)
    | throw s!"{prefixMsg}: src1BaseLo BAR slot {inBarSlot} VM base {toHex inBarVm.toNat} matches no surface"
  if inSurf.direction != .input then
    throw s!"{prefixMsg}: src1BaseLo BAR slot {inBarSlot} bound to non-input surface '{inSurf.shortName}'"

  let wResBaseLo ← requireDecodedWrite task.writes regResBaseLo "resBaseLo"
  let some outBarSlot := wResBaseLo.barSlot
    | throw s!"{prefixMsg}: resBaseLo (0x5108) must carry a BAR slot"
  let outBarVm ← resolveBarSlotVmBase info outBarSlot
  let some outSurf := info.surfaces.find? (fun s => s.sect.addr == outBarVm)
    | throw s!"{prefixMsg}: resBaseLo BAR slot {outBarSlot} VM base {toHex outBarVm.toNat} matches no surface"
  if outSurf.direction != .output then
    throw s!"{prefixMsg}: resBaseLo BAR slot {outBarSlot} bound to non-output surface '{outSurf.shortName}'"

  let fullH := inSurf.height.toNat
  let fullW := inSurf.width.toNat
  if inSurf.channels.toNat != cin then
    throw s!"{prefixMsg}: Cin={cin} != bound input surface channels {inSurf.channels}"
  if outSurf.channels.toNat != cout then
    throw s!"{prefixMsg}: Cout={cout} != bound output surface channels {outSurf.channels}"
  if outSurf.height.toNat != hout || outSurf.width.toNat != wout then
    throw s!"{prefixMsg}: Wout/Hout=({hout}x{wout}) != bound output surface ({outSurf.height}x{outSurf.width})"

  let padMode ← inferPadMode fullH fullW stride hout wout pt pl

  -- 5. Decode dwGroups (0x0030), neCommon (0x003c), and neTile (0x0040)
  let mode : Conv3x3Mode ← match findDecodedWrite task.writes regDwGroups with
    | none => pure .dense
    | some wDw =>
      let perGroupIn ← decodeFieldChecked dwGroupsLayout wDw.value "perGroupIn"
      if perGroupIn != 1 then
        throw s!"{prefixMsg}: dwGroups perGroupIn={perGroupIn} must be 1"
      if cin != cout || cin <= 1 then
        throw s!"{prefixMsg}: dwGroups (0x0030) requires Cin == Cout > 1 (got Cin={cin}, Cout={cout})"
      pure .depthwise

  let shape : Conv3x3Shape := {
    mode := mode
    h := fullH
    w := fullW
    cin := cin
    cout := cout
    stride := stride
    pad := padMode
  }

  if winReg != shape.winReg then
    throw s!"{prefixMsg}: Win={winReg} != expected {shape.winReg} (W={fullW}, cropW={shape.cropW}, s={stride})"
  if hinReg != shape.hinReg then
    throw s!"{prefixMsg}: Hin={hinReg} != expected {shape.hinReg} (H={fullH}, cropH={shape.cropH}, s={stride})"

  let kind := (← decodeFieldChecked neCommonLayout wNeCommon.value "kind").toNat
  let shPref := (← decodeFieldChecked neCommonLayout wNeCommon.value "shPref").toNat
  let shMin := (← decodeFieldChecked neCommonLayout wNeCommon.value "shMin").toNat
  let shMax := (← decodeFieldChecked neCommonLayout wNeCommon.value "shMax").toNat
  let f := (← decodeFieldChecked neCommonLayout wNeCommon.value "F").toNat
  let wide3x3Bit := (← decodeFieldChecked neCommonLayout wNeCommon.value "wide3x3").toNat
  if kind != 0 then
    throw s!"{prefixMsg}: neCommon kind={kind} != 0"
  if shMin != shPref then
    throw s!"{prefixMsg}: neCommon shMin={shMin} != shPref={shPref}"

  let ocg := (← decodeFieldChecked neTileLayout wNeTile.value "ocg").toNat
  let fat := (← decodeFieldChecked neTileLayout wNeTile.value "fat").toNat
  let wu := (← decodeFieldChecked neTileLayout wNeTile.value "WU").toNat
  if mode == .depthwise && ocg != 0 then
    throw s!"{prefixMsg}: dwGroups (0x0030) depthwise mode requires ocg=0, got ocg={ocg}"
  if target == .h17 && shape.effMode == .dense && stride == 1 && ocg > 3 then
    throw s!"{prefixMsg}: h17 stride-1 dense ocg={ocg} exceeds maximum 3"
  if ocg > 4 then
    throw s!"{prefixMsg}: ocg={ocg} exceeds maximum 4"
  if wu > 3 then
    throw s!"{prefixMsg}: WU={wu} exceeds 3"
  if f + fat > logNE || wu + f + fat != logNE then
    throw s!"{prefixMsg}: WU({wu}) + F({f}) + fat({fat}) != logNE({logNE})"

  let probeTiling : Tiling := {
    logNE := logNE
    f := f
    fat := fat
    ocg := ocg
    shPref := shPref
    shMax := shMax
    tileHeight := th
    srcSkew := 0
    resSkew := 0
  }

  -- Verify wide3x3 bit 27 safety precondition (wide3x3=0 is legal; wide3x3=1 requires H17 dense s=1 and maxBlockSlots <= 32768*rep)
  if wide3x3Bit == 1 then
    if target != .h17 then
      throw s!"{prefixMsg}: neCommon wide3x3=1 is illegal on {target.name}"
    if shape.effMode != .dense then
      throw s!"{prefixMsg}: neCommon wide3x3=1 is illegal for depthwise 3x3 conv"
    if stride != 1 then
      throw s!"{prefixMsg}: neCommon wide3x3=1 is illegal for stride-{stride} 3x3 conv"
    let rep := ((1 : Nat) <<< logNE) >>> f
    let mbs := maxBlockSlots3x3 shape probeTiling
    if mbs > 32768 * rep then
      throw s!"{prefixMsg}: neCommon wide3x3=1 illegal because maxBlockSlots={mbs} > 32768*rep={32768 * rep}"

  -- 6. Decode L2 staging registers & verify exact srcPx and non-overlapping [0, 2 MiB) ranges
  let wSrc1Cfg ← requireDecodedWrite task.writes regSrc1Cfg "src1Cfg"
  let src1Mode ← decodeFieldChecked src1CfgLayout wSrc1Cfg.value "mode"
  let src1P ← decodeFieldChecked src1CfgLayout wSrc1Cfg.value "P"
  if src1Mode != 2 || src1P != 1 then
    throw s!"{prefixMsg}: src1Cfg mode={src1Mode}, P={src1P} invalid"
  let wSrc2Cfg ← requireDecodedWrite task.writes regSrc2Cfg "src2Cfg"
  if wSrc2Cfg.value != src2CfgValue then
    throw s!"{prefixMsg}: src2Cfg {toHex wSrc2Cfg.value.toNat} != {toHex src2CfgValue.toNat}"

  let wSrc1Base ← requireDecodedWrite task.writes regSrc1Base "src1Base"
  let wSrc1Chs ← requireDecodedWrite task.writes regSrc1ChanStride "src1ChanStride"
  let wSrc1Row ← requireDecodedWrite task.writes regSrc1RowStride "src1RowStride"
  let wSrc1RowA ← requireDecodedWrite task.writes regSrc1RowBytesA "src1RowBytesA"
  let wSrc1RowB ← requireDecodedWrite task.writes regSrc1RowBytesB "src1RowBytesB"
  let wResCfg ← requireDecodedWrite task.writes regResCfg "resCfg"
  let wResBase ← requireDecodedWrite task.writes regResBase "resBase"

  let resMode ← decodeFieldChecked resCfgLayout wResCfg.value "mode"
  let resP ← decodeFieldChecked resCfgLayout wResCfg.value "P"
  let directBit ← decodeFieldChecked resCfgLayout wResCfg.value "direct"
  if resMode != 2 || resP != 1 then
    throw s!"{prefixMsg}: resCfg mode={resMode}, P={resP} invalid"
  let direct := directBit == 1

  let expectedChs := chsBytes3x3 shape probeTiling
  if wSrc1Chs.value.toNat != expectedChs then
    throw s!"{prefixMsg}: src1ChanStride {wSrc1Chs.value} != expected 3x3 chs {expectedChs} (srcPx narrowing forbidden)"

  let minSrcRow := cin * expectedChs
  if wSrc1Row.value.toNat < minSrcRow || (wSrc1Row.value.toNat - minSrcRow) % 16 != 0 then
    throw s!"{prefixMsg}: src1RowStride {wSrc1Row.value} incompatible with Cin*chs={minSrcRow} + 16*srcSkew"
  if wSrc1RowA.value.toNat != minSrcRow || wSrc1RowB.value.toNat != minSrcRow then
    throw s!"{prefixMsg}: src1RowBytesA/B ({wSrc1RowA.value}, {wSrc1RowB.value}) != Cin*chs ({minSrcRow})"
  let srcSkew := (wSrc1Row.value.toNat - minSrcRow) / 16

  let expectedDirect := isDirect3x3 target shape probeTiling
  if direct != expectedDirect then
    throw s!"{prefixMsg}: resCfg direct={direct} disagrees with target/tiling direct={expectedDirect}"

  let resSkew ←
    if direct then
      pure 0
    else
      let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
      let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
      let wResRowA ← requireDecodedWrite task.writes regResRowBytesA "resRowBytesA"
      let wResRowB ← requireDecodedWrite task.writes regResRowBytesB "resRowBytesB"
      let expDchs := dchsBytes3x3 shape probeTiling
      if wResChs.value.toNat != expDchs then
        throw s!"{prefixMsg}: resChanStride {wResChs.value} != expected dchs {expDchs}"
      let minResRow := cout * expDchs
      if wResRow.value.toNat < minResRow || (wResRow.value.toNat - minResRow) % 16 != 0 then
        throw s!"{prefixMsg}: resRowStride {wResRow.value} incompatible with Cout*dchs={minResRow} + 16*resSkew"
      if wResRowA.value.toNat != minResRow || wResRowB.value.toNat != minResRow then
        throw s!"{prefixMsg}: resRowBytesA/B ({wResRowA.value}, {wResRowB.value}) != Cout*dchs ({minResRow})"
      pure ((wResRow.value.toNat - minResRow) / 16)

  let tiling : Tiling := { probeTiling with srcSkew := srcSkew, resSkew := resSkew }
  validateConv3x3Tiling target shape tiling

  let srcBase := wSrc1Base.value.toNat
  let srcBytes := inRows3x3 shape th * wSrc1Row.value.toNat
  let resBase := wResBase.value.toNat
  let resBytes :=
    if direct then
      ((1 : Nat) <<< (f + effOcg shape tiling + fat)) * 512
    else
      let wResRow := (findDecodedWrite task.writes regResRowStride).get!
      th * wResRow.value.toNat

  if srcBytes == 0 || resBytes == 0 then
    throw s!"{prefixMsg}: L2 source ({srcBytes}B) and result ({resBytes}B) footprints must be non-empty"
  let srcEnd := srcBase + srcBytes
  let resEnd := resBase + resBytes
  if srcEnd > l2CapacityBytes then
    throw s!"{prefixMsg}: L2 source range [{toHex srcBase}, {toHex srcEnd}) exceeds 2 MiB ({toHex l2CapacityBytes})"
  if resEnd > l2CapacityBytes then
    throw s!"{prefixMsg}: L2 result range [{toHex resBase}, {toHex resEnd}) exceeds 2 MiB ({toHex l2CapacityBytes})"
  if srcBase < resEnd && resBase < srcEnd then
    throw s!"{prefixMsg}: overlapping L2 ranges src [{toHex srcBase}, {toHex srcEnd}) and res [{toHex resBase}, {toHex resEnd})"

  -- 7. Verify NE block, sparse consistency across 0x4900 and 0x5500, and scalarBias rules
  let expectedSparse : UInt32 := if stride == 2 then 1 else 0
  let wNeCfg ← requireDecodedWrite task.writes regNeCfg "neCfg"
  let neCfgSparse ← decodeFieldChecked (neCfgLayout target) wNeCfg.value "sparse"
  let wKernStage ← requireDecodedWrite task.writes regKernelStage "kernelStage"
  let kernStageSparse ← decodeFieldChecked kernelStageLayout wKernStage.value "sparse"
  let kernStageNum ← decodeFieldChecked kernelStageLayout wKernStage.value "stage"
  if kernStageNum != 1 then
    throw s!"{prefixMsg}: kernelStage stage={kernStageNum} != 1"
  if neCfgSparse != kernStageSparse then
    throw s!"{prefixMsg}: sparse flag mismatch between neCfg (0x4900: {neCfgSparse}) and kernelStage (0x5500: {kernStageSparse})"
  if neCfgSparse != expectedSparse then
    throw s!"{prefixMsg}: sparse flag {neCfgSparse} disagrees with stride {stride} (expected {expectedSparse})"

  let wNeOp ← requireDecodedWrite task.writes regNeOp "neOp"
  let biasInKernelBit ← decodeFieldChecked neOpLayout wNeOp.value "biasInKernel"
  let scalarBiasMode := shape.effMode == .dense && cout == 1
  if scalarBiasMode then
    let wScalarBias ← requireDecodedWrite task.writes regScalarBias "scalarBias"
    validateWordBits scalarBiasLayout wScalarBias.value
    if biasInKernelBit != 0 then
      throw s!"{prefixMsg}: dense Cout=1 uses scalarBias (0x490c) and must have biasInKernel=0"
  else
    if (findDecodedWrite task.writes regScalarBias).isSome then
      throw s!"{prefixMsg}: scalarBias (0x490c) must only be written when effMode == dense && Cout == 1"

  -- 8. Verify active kernel chunks & relocation
  let some kSec := info.kernelSection
    | throw s!"{prefixMsg}: NE conv task requires __KERN_0 kernel section"
  let wKernLo ← requireDecodedWrite task.writes regKernelBaseLo "kernelBaseLo"
  let some kSlot := wKernLo.barSlot
    | throw s!"{prefixMsg}: kernelBaseLo (0x5510) must carry a BAR slot"
  let kBarVm ← resolveBarSlotVmBase info kSlot
  if kBarVm != kSec.addr then
    throw s!"{prefixMsg}: kernelBaseLo BAR slot {kSlot} VM base {toHex kBarVm.toNat} != __KERN_0 addr {toHex kSec.addr.toNat}"
  if !info.relocPatchOffsets.contains wKernLo.payloadOffset.toUInt32 then
    throw s!"{prefixMsg}: missing Mach-O relocation targeting kernelBaseLo payload offset {toHex wKernLo.payloadOffset}"

  let mut activeChunks := 0
  for s in [:16] do
    let slotW? := findDecodedWrite task.writes (regChunkSlot s)
    let offW? := findDecodedWrite task.writes (regChunkOffset s)
    let sizeW? := findDecodedWrite task.writes (regChunkSize s)

    let slotEnabled ← match slotW? with
      | none => pure false
      | some sw =>
        let en ← decodeFieldChecked chunkSlotLayout sw.value "enable"
        pure (en == 1)

    if sizeW?.isSome && !slotEnabled then
      throw s!"{prefixMsg}: chunk slot {s} writes size ({toHex (regChunkSize s)}) without enable bit 0 in {toHex (regChunkSlot s)}"
    if slotEnabled then
      let some ow := offW?
        | throw s!"{prefixMsg}: enabled chunk slot {s} missing offset register {toHex (regChunkOffset s)}"
      let some szw := sizeW?
        | throw s!"{prefixMsg}: enabled chunk slot {s} missing size register {toHex (regChunkSize s)}"
      let cOff := ow.value.toNat
      let cSz := szw.value.toNat
      if cSz == 0 || cSz % dmaLineBytes != 0 then
        throw s!"{prefixMsg}: chunk slot {s} size {cSz} must be a positive multiple of {dmaLineBytes}"
      if cSz > maxKernelChunkBytes then
        throw s!"{prefixMsg}: chunk slot {s} size {cSz}B exceeds maximum {maxKernelChunkBytes}B (64 KiB)"
      if cOff + cSz > kSec.size.toNat then
        throw s!"{prefixMsg}: chunk slot {s} range [{cOff}, {cOff + cSz}) exceeds kernel size {kSec.size}"
      activeChunks := activeChunks + 1
    else if offW?.isSome then
      throw s!"{prefixMsg}: chunk slot {s} writes offset without enable bit 0 in {toHex (regChunkSlot s)}"

  if activeChunks == 0 then
    throw s!"{prefixMsg}: no active kernel chunk slots enabled"

  -- 9. Verify DMA strides & surface span bounds (using full uncropped input H, W and output Ho, Wo)
  let wSrcDep ← requireDecodedWrite task.writes regSrc1Dep "src1Dep"
  validateWordBits src1DepLayout wSrcDep.value
  let wResDep ← requireDecodedWrite task.writes regResDep "resDep"
  let resDepDirect ← decodeFieldChecked resDepLayout wResDep.value "direct"
  let resDepC ← decodeFieldChecked resDepLayout wResDep.value "code"
  if resDepC != resDepCode target then
    throw s!"{prefixMsg}: resDep code {toHex resDepC.toNat} != expected {toHex (resDepCode target).toNat}"
  if (resDepDirect == 1) != direct then
    throw s!"{prefixMsg}: resDep direct bit {resDepDirect} != resCfg direct={direct}"

  let wInRow ← requireDecodedWrite task.writes regSrc1DmaRowStride "src1DmaRowStride"
  let inRowStride := wInRow.value.toNat
  if inRowStride < 2 * fullW || inRowStride % dmaLineBytes != 0 then
    throw s!"{prefixMsg}: input DMA row stride {inRowStride} must be >= {2 * fullW} and {dmaLineBytes}-byte aligned"
  let inExtent ←
    match findDecodedWrite task.writes regSrc1DmaPlaneStride with
    | some wInPlane =>
      let inPlaneStride := wInPlane.value.toNat
      if inPlaneStride < fullH * inRowStride then
        throw s!"{prefixMsg}: input DMA plane stride {inPlaneStride} < H * rowStride ({fullH * inRowStride})"
      pure (cin * inPlaneStride)
    | none =>
      if cin == 1 then
        pure (fullH * inRowStride)
      else
        throw s!"{prefixMsg}: missing required register src1DmaPlaneStride (0x4d1c) for Cin={cin} > 1"
  if inExtent > inSurf.span.toNat || inExtent > inSurf.sect.size.toNat then
    throw s!"{prefixMsg}: input DMA extent {inExtent} exceeds bound surface '{inSurf.shortName}' (span={inSurf.span}, section={inSurf.sect.size})"

  let wOutRow ← requireDecodedWrite task.writes regResDmaRowStride "resDmaRowStride"
  let outRowStride := wOutRow.value.toNat
  if outRowStride < 2 * wout || outRowStride % dmaLineBytes != 0 then
    throw s!"{prefixMsg}: output DMA row stride {outRowStride} must be >= {2 * wout} and {dmaLineBytes}-byte aligned"
  let outExtent ←
    match findDecodedWrite task.writes regResDmaPlaneStride with
    | some wOutPlane =>
      let outPlaneStride := wOutPlane.value.toNat
      if outPlaneStride < hout * outRowStride then
        throw s!"{prefixMsg}: output DMA plane stride {outPlaneStride} < Ho * rowStride ({hout * outRowStride})"
      pure (cout * outPlaneStride)
    | none =>
      if cout == 1 then
        pure (hout * outRowStride)
      else
        throw s!"{prefixMsg}: missing required register resDmaPlaneStride (0x5114) for Cout={cout} > 1"
  if outExtent > outSurf.span.toNat || outExtent > outSurf.sect.size.toNat then
    throw s!"{prefixMsg}: output DMA extent {outExtent} exceeds bound surface '{outSurf.shortName}' (span={outSurf.span}, section={outSurf.sect.size})"

  pure ()

/-- Unified static task verifier dispatching decoded NE conv tasks by kernel size (`1x1` or `3x3`). -/
def verifyConvTask (info : HwxInfo) (task : DecodedTask) : Except String Unit := do
  let wConvCfg ← requireDecodedWrite task.writes regConvCfg "convCfg"
  let kw := (← decodeFieldChecked convCfgLayout wConvCfg.value "kw").toNat
  let kh := (← decodeFieldChecked convCfgLayout wConvCfg.value "kh").toNat
  if kw == 1 && kh == 1 then
    verifyConv1x1Task info task
  else if kw == 3 && kh == 3 then
    verifyConv3x3Task info task
  else
    throw s!"task {task.taskIndex}: unsupported conv kernel size kh={kh}, kw={kw} (expected 1x1 or 3x3)"

/-- Verify all NE 1x1 and 3x3 conv tasks decoded from a validated HWX binary.
    When `requireConvTask` is `true`, fails if any task is not a verified NE conv task or if none exist. -/
def verifyConvHwxTasks (hwxBytes : ByteArray) (info : HwxInfo) (requireConvTask : Bool := true) : Except String Nat := do
  let textBytes ← readSlice hwxBytes info.textSection.offset.toNat info.textSection.size.toNat
  let tasks ← decodeTaskStream info.target textBytes
  let mut checked := 0
  for task in tasks do
    if isNeConvTask task then
      verifyConvTask info task
      checked := checked + 1
    else if requireConvTask then
      throw s!"task {task.taskIndex}: missing NE conv configuration registers"
  if requireConvTask && checked == 0 then
    throw "HWX contains no NE conv tasks"
  pure checked

-- ============================================================================
-- Single-task fp16 3x3 convolution compiler (dense & depthwise, stride 1 & 2)
-- ============================================================================

/-- Compile a single-task fp16 3x3 convolution (dense or depthwise, stride 1 or 2,
    same or valid padding) into a statically verified HWX binary and 4 LOAD sections. -/
def compileConv3x3
    (target : Target)
    (shape : Conv3x3Shape)
    (weights : ByteArray)
    (bias : Option ByteArray := none)
    (relu : Bool := false)
    (tilingOverride : Option Tiling := none) : Except String CompiledConv3x3 := do
  validateConv3x3Shape shape
  let tiling ← match tilingOverride with
    | some t => validateConv3x3Tiling target shape t; pure t
    | none => chooseConv3x3Tiling target shape

  let packed ← packConv3x3KernelChunks target shape tiling weights bias
  let sparse := shape.isSparse
  let wide3x3 := computeWide3x3 target shape tiling
  let direct := isDirect3x3 target shape tiling
  let chs := chsBytes3x3 shape tiling
  let dchs := dchsBytes3x3 shape tiling
  let srcRow := srcRowStride3x3 shape tiling
  let resRow := resRowStride3x3 shape tiling
  let srcL2 := srcL2Bytes3x3 shape tiling
  let l2Use := l2UseBytes3x3 target shape tiling

  let ho := shape.outH
  let wo := shape.outW
  let inRowStride := alignUp (2 * shape.w) dmaLineBytes
  let inPlaneStride := shape.h * inRowStride
  let inSpan := shape.cin * inPlaneStride
  let outRowStride := alignUp (2 * wo) dmaLineBytes
  let outPlaneStride := ho * outRowStride
  let outSpan := shape.cout * outPlaneStride

  -- Encode every active register explicitly through width-checked bitfield layouts
  let wWin ← encodeWord (dim16Layout "Win") #[("val", shape.winReg.toUInt32)]
  let wHin ← encodeWord (dim16Layout "Hin") #[("val", shape.hinReg.toUInt32)]
  let wWout ← encodeWord (dim16Layout "Wout") #[("val", wo.toUInt32)]
  let wHout ← encodeWord (dim16Layout "Hout") #[("val", ho.toUInt32)]
  let wConvCfg ← encodeWord convCfgLayout #[
    ("kw", 3), ("kh", 3),
    ("sx", shape.stride.toUInt32), ("sy", shape.stride.toUInt32),
    ("pl", shape.padLeft.toUInt32), ("pt", shape.padTop.toUInt32),
    ("hx", 1), ("vy", 1)
  ]
  let wTH ← encodeWord (dim16Layout "TH") #[("val", tiling.tileHeight.toUInt32)]
  let wNeCommon ← encodeWord neCommonLayout #[
    ("kind", 0),
    ("shPref", tiling.shPref.toUInt32),
    ("shMin", tiling.shPref.toUInt32),
    ("shMax", tiling.shMax.toUInt32),
    ("F", tiling.f.toUInt32),
    ("wide3x3", if wide3x3 then 1 else 0)
  ]
  let wNeTile ← encodeWord neTileLayout #[
    ("ocg", (effOcg shape tiling).toUInt32),
    ("fat", tiling.fat.toUInt32),
    ("WU", tiling.wu.toUInt32)
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
  let wChunkSlotVal ← encodeWord chunkSlotLayout #[("enable", 1), ("on20", 1), ("task", 0)]
  let wOut2 ← encodeWord out2Layout #[("form", 0xe0), ("task", 0)]

  let mut plain : Array RegWrite := #[
    -- Common block
    { byteAddr := regWin, value := wWin },
    { byteAddr := regHin, value := wHin },
    { byteAddr := regCin, value := shape.cin.toUInt32 },
    { byteAddr := regWout, value := wWout },
    { byteAddr := regHout, value := wHout },
    { byteAddr := regCout, value := shape.cout.toUInt32 },
    { byteAddr := regConvCfg, value := wConvCfg },
    { byteAddr := regTH, value := wTH },
    { byteAddr := regNeCommon, value := wNeCommon },
    { byteAddr := regNeTile, value := wNeTile },
    { byteAddr := regTargetCfg, value := targetCfgValue target },
    -- L2 staging block
    { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
    { byteAddr := regSrc2Cfg, value := src2CfgValue },
    { byteAddr := regSrc1Base, value := 0 },
    { byteAddr := regSrc1ChanStride, value := chs.toUInt32 },
    { byteAddr := regSrc1RowStride, value := srcRow.toUInt32 },
    { byteAddr := regSrc1RowBytesA, value := (shape.cin * chs).toUInt32 },
    { byteAddr := regSrc1RowBytesB, value := (shape.cin * chs).toUInt32 },
    { byteAddr := regResCfg, value := wResCfg },
    { byteAddr := regResBase, value := srcL2.toUInt32 },
    -- Neural Engine block
    { byteAddr := regNeCfg, value := wNeCfg },
    { byteAddr := regNeOp, value := wNeOp },
    { byteAddr := regNeScale, value := wNeScale },
    -- Tile DMA (Source) block (non-BAR; uses full uncropped input strides)
    { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
    { byteAddr := regSrc1DmaRowStride, value := inRowStride.toUInt32 },
    { byteAddr := regSrc1DmaPlaneStride, value := inPlaneStride.toUInt32 },
    { byteAddr := regSrc1Fmt, value := wSrc1Fmt },
    -- Result DMA block (non-BAR)
    { byteAddr := regResDmaRowStride, value := outRowStride.toUInt32 },
    { byteAddr := regResDmaPlaneStride, value := outPlaneStride.toUInt32 },
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

  -- Staged mode result L2 registers (omitted only in direct mode)
  if !direct then
    plain := plain ++ #[
      { byteAddr := regResChanStride, value := dchs.toUInt32 },
      { byteAddr := regResRowStride, value := resRow.toUInt32 },
      { byteAddr := regResRowBytesA, value := (shape.cout * dchs).toUInt32 },
      { byteAddr := regResRowBytesB, value := (shape.cout * dchs).toUInt32 }
    ]

  -- Scalar bias register (written only when effMode == dense && Cout == 1)
  if shape.effMode == .dense && shape.cout == 1 then
    let wScalarBias ← encodeWord scalarBiasLayout #[("val", packed.scalarBiasBits.toUInt32)]
    plain := plain.push { byteAddr := regScalarBias, value := wScalarBias }

  -- Active kernel chunk table slots (inactive slots omitted)
  for cs in packed.slots do
    plain := plain ++ #[
      { byteAddr := regChunkSlot cs.slot, value := wChunkSlotVal },
      { byteAddr := regChunkOffset cs.slot, value := cs.offset.toUInt32 },
      { byteAddr := regChunkSize cs.slot, value := cs.size.toUInt32 }
    ]

  -- BAR-tagged sequential records in canonical order
  let wSrc1Dep ← encodeWord src1DepLayout #[("code", 0x21), ("task", 0)]
  let wResDep ← encodeWord resDepLayout #[
    ("code", resDepCode target),
    ("task", 0),
    ("direct", if direct then 1 else 0)
  ]
  let barRecs : Array BarRegWrite := #[
    { baseByteAddr := regKernelBaseLo, barSlot := 8, values := #[0, 0] },
    { baseByteAddr := regSrc1BaseLo, barSlot := 6, values := #[0, 0] },
    { baseByteAddr := regResBaseLo, barSlot := 7, values := #[0, 0] },
    { baseByteAddr := regSrc1Dep, barSlot := 4, values := #[wSrc1Dep] },
    { baseByteAddr := regResDep, barSlot := 4, values := #[wResDep] }
  ]

  let plainRecs ← packPlainWrites plain
  let allRecs ← packTaskRecords plain barRecs
  let kernelRecIdx := plainRecs.size
  let totalTaskWords := 9 + recordWordsCount allRecs
  let header ← encodeTaskHeader9 0 totalTaskWords tiling.logNE

  let prologue : UInt32 := match target with
    | .h16 => 1
    | .h17 => 0
  let stream : TaskStream := {
    prologue := prologue
    tasks := #[{ header := header, records := allRecs }]
    trailingZeroBlocks := 0
  }

  let encStream ← encodeTaskStream target stream
  let some (_, _, _, kernRelocOff) :=
    encStream.payloadSites.find? (fun (ti, ri, pi, _) => ti == 0 && ri == kernelRecIdx && pi == 0)
    | throw "internal error: failed to locate 0x5510 payload word offset for kernel relocation"

  let inAlloc := alignUp inSpan pageSize
  let outAlloc := alignUp outSpan pageSize
  let inSectSize : UInt64 := match target with
    | .h16 => inAlloc.toUInt64
    | .h17 => inSpan.toUInt64
  let outSectSize : UInt64 := match target with
    | .h16 => outAlloc.toUInt64
    | .h17 => outSpan.toUInt64

  let inSurf : SurfaceSpec := {
    shortName := "x"
    symbolName := "input_x"
    direction := .input
    n := 1
    channels := shape.cin.toUInt32
    height := shape.h.toUInt32
    width := shape.w.toUInt32
    rowStride := inRowStride.toUInt64
    planeStride := inPlaneStride.toUInt64
    batchStride := inSpan.toUInt64
    sectionSize := some inSectSize
    allocSize := some inAlloc.toUInt64
  }
  let outSurf : SurfaceSpec := {
    shortName := "y"
    symbolName := "output_y"
    direction := .output
    n := 1
    channels := shape.cout.toUInt32
    height := ho.toUInt32
    width := wo.toUInt32
    rowStride := outRowStride.toUInt64
    planeStride := outPlaneStride.toUInt64
    batchStride := outSpan.toUInt64
    sectionSize := some outSectSize
    allocSize := some outAlloc.toUInt64
  }

  let prog : HwxProgram := {
    target := target
    surfaces := #[inSurf, outSurf]
    constants := zeroBytes textConstBytes
    kernel := some packed.bytes
    scratch := none
    taskStream := stream
    relocations := #[{ patchOffset := kernRelocOff.toUInt32 }]
    ident := some "vane"
  }

  let hwxBytes ← emitHwx prog
  let info ← validateHwx hwxBytes
  let secs ← buildSectionsFromInfo info
  validateSectionsWithInfo info secs
  let _ ← verifyConvHwxTasks hwxBytes info (requireConvTask := true)

  pure {
    target := target
    shape := shape
    tiling := tiling
    sparse := sparse
    wide3x3 := wide3x3
    hasBias := bias.isSome
    relu := relu
    direct := direct
    l2UseBytes := l2Use
    kernelBytes := packed.bytes.size
    hwx := hwxBytes
    info := info
    sections := secs
  }

end Vane.Conv3x3
