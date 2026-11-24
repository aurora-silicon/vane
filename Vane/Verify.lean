import Vane.Binary
import Vane.TaskStream
import Vane.Hwx
import Vane.Register

namespace Vane.Verify

open Vane.Binary
open Vane.TaskStream
open Vane.Hwx
open Vane.Register

-- ============================================================================
-- Independent TCB hardware constants & geometry helpers (no Vane.Tiling import)
-- ============================================================================

private def vL2CapacityBytes : Nat := 0x200000
private def vMaxKernelChunkBytes : Nat := 0x10000
private def vDmaLineBytes : Nat := 64

private def vCeilDiv (a b : Nat) : Nat :=
  if b == 0 then 0 else (a + b - 1) / b

private def vStripW (fat shPref : Nat) : Nat :=
  (128 <<< fat) >>> shPref

private def vSrcPx (kernelSize stride winReg padLeft stripW : Nat) : Nat :=
  if kernelSize == 1 then
    min (alignUp winReg 8) (max stripW 32)
  else
    let ti := stride * stripW
    let r := alignUp winReg 8
    let halo : Int := if padLeft > 0 then 8 else 0
    let flr : Int := if ti > 8 then ((min ti 32 : Nat) : Int) else 24
    let lookahead : Int := halo + min ((r : Int) - 2 * ((max ti 16 : Nat) : Int)) 32
    let extra : Nat := (max flr lookahead).toNat
    min r (ti + extra)

private def vChs (kernelSize stride winReg padLeft stripW : Nat) : Nat :=
  2 * vSrcPx kernelSize stride winReg padLeft stripW

private def vDchs (wout stripW : Nat) : Nat :=
  2 * min (alignUp wout 8) (max stripW 32)

private def vIsDirect (target : Target) (fat shMax wout : Nat) : Bool :=
  let lhs := match target with
    | .h17 => (128 <<< fat) >>> shMax
    | .h16 => 128 >>> shMax
  lhs >= min 32 wout

/-- Independent linear tile scan for maximum input rows buffered per output row tile. -/
private def vInRows (kernelSize h stride pt hout th : Nat) : Nat := Id.run do
  if th == 0 || hout == 0 then
    return 0
  if kernelSize == 1 then
    return min th h
  let rk := alignUp 3 stride
  let nTiles := vCeilDiv hout th
  let mut best := 0
  for i in [:nTiles] do
    let y0 := i * th
    let y1 := min hout (y0 + th)
    if y1 > y0 then
      let hi := min h ((y1 - 1) * stride + rk - pt)
      let lo := y0 * stride - pt
      let rows := hi - lo
      if rows > best then
        best := rows
  return best

/-- Independent output channel block assignment for group `g in 0..2^F-1`. -/
private def vGroupBlocks (isDepthwise : Bool) (cout f effOcg g : Nat) : Array (Nat × Nat) := Id.run do
  if isDepthwise && f == 4 then
    let mut blocks : Array (Nat × Nat) := #[]
    let mut o := g
    while o < cout do
      blocks := blocks.push (o, 1)
      o := o + 16
    return blocks
  let groups := (1 : Nat) <<< f
  let blk := (1 : Nat) <<< effOcg
  let round := groups * blk
  let full := cout / round
  let rem := cout % round
  let mut blocks : Array (Nat × Nat) := Array.mkEmpty (full + 1)
  for r in [:full] do
    blocks := blocks.push (r * round + g * blk, blk)
  if rem > 0 then
    let base := rem / groups
    let extra := rem % groups
    let lastSize := base + (if g < extra then 1 else 0)
    let lastStart := g * base + min g extra
    if lastSize > 0 then
      blocks := blocks.push (full * round + lastStart, lastSize)
  return blocks

private def vGroupOuts (blocks : Array (Nat × Nat)) : Nat :=
  blocks.foldl (fun acc (_, n) => acc + n) 0

private def vMaxBlockSlots (isDepthwise : Bool) (cin cout f effOcg : Nat) : Nat := Id.run do
  let groups := (1 : Nat) <<< f
  let mut best := 0
  for g in [:groups] do
    for (_, n) in vGroupBlocks isDepthwise cout f effOcg g do
      let slots := cin * 9 * alignUp n 2
      if slots > best then
        best := slots
  return best

/-- Independently verify 3x3 `(same | valid)` spatial geometry and return expected `(Win, Hin)`. -/
private def vVerify3x3Geometry (fullH fullW stride hout wout pt pl : Nat) : Except String (Nat × Nat) := do
  let sameHo := vCeilDiv fullH stride
  let sameWo := vCeilDiv fullW stride
  let samePt := if sameHo == 0 then 0 else ((sameHo - 1) * stride + 3 - fullH) / 2
  let samePl := if sameWo == 0 then 0 else ((sameWo - 1) * stride + 3 - fullW) / 2
  let matchesSame := sameHo == hout && sameWo == wout && samePt == pt && samePl == pl
  let matchesValid :=
    fullH >= 3 && fullW >= 3 &&
    (fullH - 3) / stride + 1 == hout &&
    (fullW - 3) / stride + 1 == wout &&
    pt == 0 && pl == 0
  if !matchesSame && !matchesValid then
    throw s!"3x3 spatial geometry mismatch: input=({fullH}x{fullW}), stride={stride}, output=({hout}x{wout}), pad=(pt={pt}, pl={pl}) matches neither same nor valid"
  if stride == 1 then
    pure (fullW, fullH)
  else
    let rk := alignUp 3 stride
    let cropW := min fullW ((wout - 1) * stride + rk - pl)
    let cropH := min fullH ((hout - 1) * stride + rk - pt)
    pure (cropW, cropH)

private def vPopcount16 (m : UInt16) : Nat := Id.run do
  let mut cnt := 0
  for i in [:16] do
    if ((m.toNat >>> i) &&& 1) == 1 then
      cnt := cnt + 1
  return cnt

/-- Independently verify stride-2 sparse kernel chunk block headers and 16-slot vectors inside
    `__KERN_0` bytes `[chunkStart, chunkEnd)`. -/
private def vVerifySparseChunkBytes
    (kBytes : ByteArray)
    (chunkStart chunkEnd : Nat)
    (blocks : Array (Nat × Nat))
    (biasInKernel : Bool)
    (label : String) : Except String Unit := do
  if chunkEnd > kBytes.size then
    throw s!"{label}: sparse chunk range [{chunkStart}, {chunkEnd}) exceeds __KERN_0 size {kBytes.size}"
  let mut pos := chunkStart
  let mut bi := 0
  for (_, n) in blocks do
    if biasInKernel then
      let biasBytes := 2 * n
      if pos + biasBytes > chunkEnd then
        throw s!"{label}: sparse block {bi} bias [{pos}, {pos + biasBytes}) exceeds chunk end {chunkEnd}"
      pos := pos + biasBytes
    if pos + 3 > chunkEnd then
      throw s!"{label}: sparse block {bi} u24 length header truncated at {pos} (chunkEnd={chunkEnd})"
    let b0 := (← readU8 kBytes pos).toNat
    let b1 := (← readU8 kBytes (pos + 1)).toNat
    let b2 := (← readU8 kBytes (pos + 2)).toNat
    let bodyBytes := b0 ||| (b1 <<< 8) ||| (b2 <<< 16)
    if bodyBytes % 2 != 0 then
      throw s!"{label}: sparse block {bi} body length {bodyBytes} is not a multiple of 2"
    let bodyEnd := pos + 3 + bodyBytes
    if bodyEnd > chunkEnd then
      throw s!"{label}: sparse block {bi} body [{pos + 3}, {bodyEnd}) exceeds chunk end {chunkEnd}"
    let mut vPos := pos + 3
    while vPos < bodyEnd do
      let mask ← readU16LE kBytes vPos
      let nextPos := vPos + 2 + 2 * vPopcount16 mask
      if nextPos > bodyEnd then
        throw s!"{label}: sparse block {bi} vector at {vPos} overruns body end {bodyEnd}"
      vPos := nextPos
    pos := bodyEnd
    bi := bi + 1
  pure ()

-- ============================================================================
-- Task classification & BAR-slot resource binding resolver (extensible for M3/M4)
-- ============================================================================

/-- High-level task kind recognized by the static safety verifier (extensible for M3 PE ops). -/
inductive TaskKind where
  | neConv
  | unknown
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Resolved resource target referenced by a BAR slot in the HWX program descriptor. -/
inductive BarTarget where
  | surface (surf : ValidatedSurface)
  | kernel (sect : SectionInfo)
  | scratch (sect : SectionInfo)
  | configZero
  deriving Repr, BEq

private def findDecodedWrite (writes : Array DecodedWrite) (addr : Nat) : Option DecodedWrite :=
  writes.find? (fun w => w.byteAddr == addr)

private def requireDecodedWrite (writes : Array DecodedWrite) (addr : Nat) (label : String) : Except String DecodedWrite :=
  match findDecodedWrite writes addr with
  | some w => pure w
  | none => throw s!"missing required register {label} ({toHex addr})"

/-- Classify a decoded task by its emitted machine registers. -/
def classifyTask (task : DecodedTask) : TaskKind :=
  if (findDecodedWrite task.writes regConvCfg).isSome ||
     (findDecodedWrite task.writes regNeCommon).isSome ||
     (findDecodedWrite task.writes regNeTile).isSome then
    .neConv
  else
    .unknown

/-- Return `true` iff a decoded task writes NE convolution configuration registers. -/
def isNeConvTask (task : DecodedTask) : Bool :=
  classifyTask task == .neConv

private def resolveBarSlotVmBase (info : HwxInfo) (slot : UInt8) : Except String UInt64 := do
  if slot.toNat > 63 then
    throw s!"BAR slot {slot} out of range 0..63"
  readU64LE info.programDescBytes (0x10 * (slot.toNat + 1))

/-- Resolve a BAR slot index to its bound HWX resource (`surface`, `kernel`, `scratch`, or zero config slot). -/
def resolveBarTarget (info : HwxInfo) (slot : UInt8) : Except String BarTarget := do
  let vm ← resolveBarSlotVmBase info slot
  if vm == 0 then
    return .configZero
  if let some surf := info.surfaces.find? (fun s => s.sect.addr == vm) then
    return .surface surf
  if let some kSec := info.kernelSection then
    if kSec.addr == vm then
      return .kernel kSec
  if let some sSec := info.scratchSection then
    if sSec.addr == vm then
      return .scratch sSec
  throw s!"BAR slot {slot} VM base {toHex vm.toNat} matches no HWX surface, kernel, or scratch section"

/-- Require a `(BaseLo, BaseHi)` register pair with matching BAR slots and return
    `(barSlot, resolvedTarget, full64BitOffset, loPayloadOffset)`. -/
private def resolveBasePair
    (info : HwxInfo) (writes : Array DecodedWrite)
    (loAddr hiAddr : Nat) (loName hiName prefixMsg : String) :
    Except String (UInt8 × BarTarget × Nat × Nat) := do
  let wLo ← requireDecodedWrite writes loAddr loName
  let wHi ← requireDecodedWrite writes hiAddr hiName
  let some loSlot := wLo.barSlot
    | throw s!"{prefixMsg}: {loName} ({toHex loAddr}) must carry a BAR slot"
  let some hiSlot := wHi.barSlot
    | throw s!"{prefixMsg}: {hiName} ({toHex hiAddr}) must carry a BAR slot"
  if loSlot != hiSlot then
    throw s!"{prefixMsg}: {loName}/{hiName} BAR slot mismatch ({loSlot} != {hiSlot})"
  let target ← resolveBarTarget info loSlot
  let fullOff : Nat := (wHi.value.toNat <<< 32) ||| wLo.value.toNat
  pure (loSlot, target, fullOff, wLo.payloadOffset)

-- ============================================================================
-- Unified standalone NE Conv static safety verifier
-- ============================================================================

/-- Unified standalone static safety verifier for decoded NE conv tasks (`1x1` and `3x3`,
    dense and depthwise, stride 1 and 2). Operates purely on decoded HWX bytes with
    independent geometry and safety checks (not reusing `Vane.Tiling` lowering helpers). -/
def verifyConvTask (info : HwxInfo) (task : DecodedTask) (kernelBytes? : Option ByteArray := none) :
    Except String Unit := do
  let target := info.target
  let prefixMsg := s!"task {task.taskIndex}"

  -- 1. Verify all 9 task header words and extract taskId, taskWords, logNE
  let (h0, h1, h2, h3, h4, h5, h6, h7, h8) ← match task.header.toList with
    | [w0, w1, w2, w3, w4, w5, w6, w7, w8] => pure (w0, w1, w2, w3, w4, w5, w6, w7, w8)
    | _ => throw s!"{prefixMsg}: expected 9-word header, got {task.header.size}"

  let taskId := (← decodeFieldChecked taskHeader0Layout h0 "taskId").toNat
  let taskWords := (← decodeFieldChecked taskHeader0Layout h0 "taskWords").toNat
  if taskWords < 9 || taskWords > 0x7ff then
    throw s!"{prefixMsg}: header[0] taskWords={taskWords} out of range 9..{0x7ff}"
  if h1 != 0 then
    throw s!"{prefixMsg}: header[1] must be 0, got {toHex h1.toNat}"
  if h2 != taskHeaderWord2Const then
    throw s!"{prefixMsg}: header[2] {toHex h2.toNat} != expected {toHex taskHeaderWord2Const.toNat}"
  if h3 != 0 then
    throw s!"{prefixMsg}: header[3] must be 0, got {toHex h3.toNat}"
  if h4 != taskHeaderWord4Const then
    throw s!"{prefixMsg}: header[4] {toHex h4.toNat} != expected {toHex taskHeaderWord4Const.toNat}"
  if h5 != 0 || h6 != 0 || h7 != 0 then
    throw s!"{prefixMsg}: header[5..7] must all be 0 (got {toHex h5.toNat}, {toHex h6.toNat}, {toHex h7.toNat})"
  let ene ← decodeFieldChecked taskHeader8Layout h8 "ene"
  if ene < 1 || ene > 5 then
    throw s!"{prefixMsg}: header[8] ene={ene} out of range 1..5"
  let logNE := ene.toNat - 1

  -- 2. Reject duplicate register writes within the same task
  let mut seenAddrs : Array Nat := #[]
  for w in task.writes do
    if seenAddrs.contains w.byteAddr then
      throw s!"{prefixMsg}: duplicate write to register {toHex w.byteAddr}"
    seenAddrs := seenAddrs.push w.byteAddr

  -- 3. Decode common shape & control registers with strict bitfield validation
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

  if wTargetCfg.value != targetCfgValue target then
    throw s!"{prefixMsg}: targetCfg {toHex wTargetCfg.value.toNat} != expected {toHex (targetCfgValue target).toNat}"

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
  if shPref > shMax then
    throw s!"{prefixMsg}: neCommon shPref={shPref} > shMax={shMax}"
  if shMax > 4 then
    throw s!"{prefixMsg}: neCommon shMax={shMax} exceeds maximum 4"

  -- Checks 0x40[31:6] == 0 via validateWordBits inside decodeFieldChecked
  let ocg := (← decodeFieldChecked neTileLayout wNeTile.value "ocg").toNat
  let fat := (← decodeFieldChecked neTileLayout wNeTile.value "fat").toNat
  let wu := (← decodeFieldChecked neTileLayout wNeTile.value "WU").toNat
  if fat > 1 then
    throw s!"{prefixMsg}: fat={fat} must be 0 or 1"
  if wu > 3 then
    throw s!"{prefixMsg}: WU={wu} exceeds 3"
  if f + fat > logNE || wu + f + fat != logNE then
    throw s!"{prefixMsg}: WU({wu}) + F({f}) + fat({fat}) != logNE({logNE})"

  let stripW := vStripW fat shPref
  if stripW < 8 || stripW > 256 then
    throw s!"{prefixMsg}: strip width T={stripW} outside [8, 256]"

  -- 4. Resolve bound input and output HWX surfaces via (BaseLo, BaseHi) BAR pairs
  let (inBarSlot, inTarget, srcBaseOff, _) ←
    resolveBasePair info task.writes regSrc1BaseLo regSrc1BaseHi "src1BaseLo" "src1BaseHi" prefixMsg
  let inSurf ← match inTarget with
    | .surface s =>
      if s.direction != .input then
        throw s!"{prefixMsg}: src1Base BAR slot {inBarSlot} bound to non-input surface '{s.shortName}'"
      pure s
    | _ =>
      throw s!"{prefixMsg}: src1Base BAR slot {inBarSlot} does not reference an input surface"
  if srcBaseOff % vDmaLineBytes != 0 then
    throw s!"{prefixMsg}: src1Base offset {toHex srcBaseOff} is not {vDmaLineBytes}-byte aligned"

  let (outBarSlot, outTarget, resBaseOff, _) ←
    resolveBasePair info task.writes regResBaseLo regResBaseHi "resBaseLo" "resBaseHi" prefixMsg
  let outSurf ← match outTarget with
    | .surface s =>
      if s.direction != .output then
        throw s!"{prefixMsg}: resBase BAR slot {outBarSlot} bound to non-output surface '{s.shortName}'"
      pure s
    | _ =>
      throw s!"{prefixMsg}: resBase BAR slot {outBarSlot} does not reference an output surface"
  if resBaseOff % vDmaLineBytes != 0 then
    throw s!"{prefixMsg}: resBase offset {toHex resBaseOff} is not {vDmaLineBytes}-byte aligned"

  if inSurf.n != 1 || outSurf.n != 1 then
    throw s!"{prefixMsg}: single-batch surfaces required (got in.n={inSurf.n}, out.n={outSurf.n})"

  let fullH := inSurf.height.toNat
  let fullW := inSurf.width.toNat
  if fullH == 0 || fullH > 65535 || fullW == 0 || fullW > 65535 then
    throw s!"{prefixMsg}: bound input surface dims ({fullH}x{fullW}) outside 1..65535"
  if cin == 0 || cout == 0 then
    throw s!"{prefixMsg}: Cin={cin} and Cout={cout} must be >= 1"
  if inSurf.channels.toNat != cin then
    throw s!"{prefixMsg}: Cin={cin} != bound input surface channels {inSurf.channels}"
  if outSurf.channels.toNat != cout then
    throw s!"{prefixMsg}: Cout={cout} != bound output surface channels {outSurf.channels}"
  if outSurf.height.toNat != hout || outSurf.width.toNat != wout then
    throw s!"{prefixMsg}: Wout/Hout=({hout}x{wout}) != bound output surface ({outSurf.height}x{outSurf.width})"
  if hout == 0 || wout == 0 then
    throw s!"{prefixMsg}: output dims ({hout}x{wout}) must be >= 1"

  -- 5. Kernel-specific geometry, dwGroups, ocg/coverage safety, and wide3x3 precondition
  let (kernelSize, stride, isDepthwise) ←
    if kw == 1 && kh == 1 then
      if sx != 1 || sy != 1 || pl != 0 || pt != 0 || hx != 1 || vy != 1 then
        throw s!"{prefixMsg}: convCfg {toHex wConvCfg.value.toNat} is not a stride-1 1x1 convolution"
      if wout != winReg || hout != hinReg || winReg != fullW || hinReg != fullH then
        throw s!"{prefixMsg}: output/input spatial dims ({hout}x{wout}, {hinReg}x{winReg}) != surface ({fullH}x{fullW}) for 1x1 conv"
      if wide3x3Bit != 0 then
        throw s!"{prefixMsg}: neCommon wide3x3={wide3x3Bit} must be 0 for 1x1 conv"
      if (findDecodedWrite task.writes regDwGroups).isSome then
        throw s!"{prefixMsg}: dwGroups (0x0030) must not be written for 1x1 conv"
      if ocg > 4 then
        throw s!"{prefixMsg}: ocg={ocg} exceeds maximum 4"
      let maxCout := match target with
        | .h16 => (1 : Nat) <<< (f + ocg)
        | .h17 => (1 : Nat) <<< (f + ocg + 1)
      if cout > maxCout then
        throw s!"{prefixMsg}: 1x1 Cout={cout} exceeds {target.name} coverage {maxCout} (F={f}, ocg={ocg})"
      pure (1, 1, false)
    else if kw == 3 && kh == 3 then
      if sx != sy || (sx != 1 && sx != 2) then
        throw s!"{prefixMsg}: convCfg strides sx={sx}, sy={sy} must be equal and 1 or 2"
      if hx != 1 || vy != 1 then
        throw s!"{prefixMsg}: convCfg hx={hx}, vy={vy} must both be 1"
      let s := sx
      let (expWin, expHin) ← vVerify3x3Geometry fullH fullW s hout wout pt pl
      let isDw ← match findDecodedWrite task.writes regDwGroups with
        | none => pure false
        | some wDw =>
          let perGroupIn ← decodeFieldChecked dwGroupsLayout wDw.value "perGroupIn"
          if perGroupIn != 1 then
            throw s!"{prefixMsg}: dwGroups perGroupIn={perGroupIn} must be 1"
          if cin != cout || cin <= 1 then
            throw s!"{prefixMsg}: dwGroups (0x0030) requires Cin == Cout > 1 (got Cin={cin}, Cout={cout})"
          pure true
      if winReg != expWin then
        throw s!"{prefixMsg}: Win={winReg} != expected {expWin} (W={fullW}, s={s})"
      if hinReg != expHin then
        throw s!"{prefixMsg}: Hin={hinReg} != expected {expHin} (H={fullH}, s={s})"
      if isDw then
        if ocg != 0 then
          throw s!"{prefixMsg}: dwGroups (0x0030) depthwise mode requires ocg=0, got ocg={ocg}"
      else
        if target == .h17 && s == 1 then
          if ocg > 3 then
            throw s!"{prefixMsg}: h17 stride-1 dense ocg={ocg} exceeds maximum 3"
          let singleCap := (1 : Nat) <<< (f + ocg)
          if cout > singleCap && !(f == 4 && ocg == 3) then
            throw s!"{prefixMsg}: h17 stride-1 dense Cout={cout} exceeds coverage {singleCap} (F={f}, ocg={ocg})"
        else
          if ocg > 4 then
            throw s!"{prefixMsg}: ocg={ocg} exceeds maximum 4"
          let singleCap := (1 : Nat) <<< (f + ocg)
          if cout > singleCap then
            throw s!"{prefixMsg}: dense 3x3 Cout={cout} exceeds single-round coverage {singleCap} (F={f}, ocg={ocg})"
      pure (3, s, isDw)
    else
      throw s!"{prefixMsg}: unsupported conv kernel size kh={kh}, kw={kw} (expected 1x1 or 3x3)"

  let effOcg := if isDepthwise then 0 else ocg
  let isSparse := kernelSize == 3 && stride == 2

  -- Verify wide3x3 bit 27 safety precondition (wide3x3=0 is legal; wide3x3=1 requires H17 dense 3x3 s=1 and maxBlockSlots <= 32768*rep)
  if kernelSize == 3 && wide3x3Bit == 1 then
    if target != .h17 then
      throw s!"{prefixMsg}: neCommon wide3x3=1 is illegal on {target.name}"
    if isDepthwise then
      throw s!"{prefixMsg}: neCommon wide3x3=1 is illegal for depthwise 3x3 conv"
    if stride != 1 then
      throw s!"{prefixMsg}: neCommon wide3x3=1 is illegal for stride-{stride} 3x3 conv"
    let rep := ((1 : Nat) <<< logNE) >>> f
    let mbs := vMaxBlockSlots isDepthwise cin cout f effOcg
    if mbs > 32768 * rep then
      throw s!"{prefixMsg}: neCommon wide3x3=1 illegal because maxBlockSlots={mbs} > 32768*rep={32768 * rep}"

  -- 6. Decode L2 staging registers & verify exact srcPx, direct-mode cleanliness, and [0, 2 MiB) ranges
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

  let expectedChs := vChs kernelSize stride winReg pl stripW
  if wSrc1Chs.value.toNat != expectedChs then
    throw s!"{prefixMsg}: src1ChanStride {wSrc1Chs.value} != expected chs {expectedChs} (srcPx narrowing forbidden)"

  let minSrcRow := cin * expectedChs
  if wSrc1Row.value.toNat < minSrcRow || (wSrc1Row.value.toNat - minSrcRow) % 16 != 0 then
    throw s!"{prefixMsg}: src1RowStride {wSrc1Row.value} incompatible with Cin*chs={minSrcRow} + 16*srcSkew"
  if wSrc1RowA.value.toNat != minSrcRow || wSrc1RowB.value.toNat != minSrcRow then
    throw s!"{prefixMsg}: src1RowBytesA/B ({wSrc1RowA.value}, {wSrc1RowB.value}) != Cin*chs ({minSrcRow})"

  let expectedDirect := vIsDirect target fat shMax wout
  if direct != expectedDirect then
    throw s!"{prefixMsg}: resCfg direct={direct} disagrees with target/tiling direct={expectedDirect}"

  let stagedResRowStride? : Option Nat ←
    if direct then
      -- Direct mode: staged output stride registers 0x4150..0x415c must be absent
      for (addr, name) in #[
        (regResChanStride, "resChanStride"),
        (regResRowStride, "resRowStride"),
        (regResRowBytesA, "resRowBytesA"),
        (regResRowBytesB, "resRowBytesB")
      ] do
        if (findDecodedWrite task.writes addr).isSome then
          throw s!"{prefixMsg}: direct mode must omit staged register {name} ({toHex addr})"
      pure none
    else
      let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
      let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
      let wResRowA ← requireDecodedWrite task.writes regResRowBytesA "resRowBytesA"
      let wResRowB ← requireDecodedWrite task.writes regResRowBytesB "resRowBytesB"
      let expDchs := vDchs wout stripW
      if wResChs.value.toNat != expDchs then
        throw s!"{prefixMsg}: resChanStride {wResChs.value} != expected dchs {expDchs}"
      let minResRow := cout * expDchs
      if wResRow.value.toNat < minResRow || (wResRow.value.toNat - minResRow) % 16 != 0 then
        throw s!"{prefixMsg}: resRowStride {wResRow.value} incompatible with Cout*dchs={minResRow} + 16*resSkew"
      if wResRowA.value.toNat != minResRow || wResRowB.value.toNat != minResRow then
        throw s!"{prefixMsg}: resRowBytesA/B ({wResRowA.value}, {wResRowB.value}) != Cout*dchs ({minResRow})"
      pure (some wResRow.value.toNat)

  if th == 0 || th > hout then
    throw s!"{prefixMsg}: tileHeight={th} outside 1..{hout}"

  let srcBase := wSrc1Base.value.toNat
  let inRows := vInRows kernelSize fullH stride pt hout th
  let srcBytes := inRows * wSrc1Row.value.toNat
  let resBase := wResBase.value.toNat
  let resBytes ← match stagedResRowStride? with
    | none => pure (((1 : Nat) <<< (f + effOcg + fat)) * 512)
    | some resRowStride => pure (th * resRowStride)

  if srcBytes == 0 || resBytes == 0 then
    throw s!"{prefixMsg}: L2 source ({srcBytes}B) and result ({resBytes}B) footprints must be non-empty"
  let srcEnd := srcBase + srcBytes
  let resEnd := resBase + resBytes
  if srcEnd > vL2CapacityBytes then
    throw s!"{prefixMsg}: L2 source range [{toHex srcBase}, {toHex srcEnd}) exceeds 2 MiB ({toHex vL2CapacityBytes})"
  if resEnd > vL2CapacityBytes then
    throw s!"{prefixMsg}: L2 result range [{toHex resBase}, {toHex resEnd}) exceeds 2 MiB ({toHex vL2CapacityBytes})"
  if srcBase < resEnd && resBase < srcEnd then
    throw s!"{prefixMsg}: overlapping L2 ranges src [{toHex srcBase}, {toHex srcEnd}) and res [{toHex resBase}, {toHex resEnd})"
  if srcBase % vDmaLineBytes != 0 then
    throw s!"{prefixMsg}: L2 src1Base {toHex srcBase} is not {vDmaLineBytes}-byte aligned"
  if resBase % vDmaLineBytes != 0 then
    throw s!"{prefixMsg}: L2 resBase {toHex resBase} is not {vDmaLineBytes}-byte aligned"

  -- 7. Verify NE block, neScale, sparse consistency across 0x4900 and 0x5500, and scalarBias rules
  let expectedSparse : UInt32 := if isSparse then 1 else 0
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
  let biasInKernel := biasInKernelBit == 1
  let scalarBiasMode := !isDepthwise && cout == 1
  if scalarBiasMode then
    let wScalarBias ← requireDecodedWrite task.writes regScalarBias "scalarBias"
    validateWordBits scalarBiasLayout wScalarBias.value
    if biasInKernel then
      throw s!"{prefixMsg}: dense Cout=1 uses scalarBias (0x490c) and must have biasInKernel=0"
  else
    if (findDecodedWrite task.writes regScalarBias).isSome then
      throw s!"{prefixMsg}: scalarBias (0x490c) must only be written when effMode == dense && Cout == 1"

  let wNeScale ← requireDecodedWrite task.writes regNeScale "neScale"
  let neScaleVal ← decodeFieldChecked neScaleLayout wNeScale.value "val"
  if neScaleVal != fp16OneBits then
    throw s!"{prefixMsg}: neScale {toHex neScaleVal.toNat} != expected fp16 1.0 ({toHex fp16OneBits.toNat})"

  -- 8. Verify kernel BaseLo/Hi pair, relocation, exact active chunk slot set, control bits,
  --    64B alignment, min/max bytes, and sparse block u24 headers inside __KERN_0
  let (kSlot, kTarget, kernBaseOff, kLoPayloadOff) ←
    resolveBasePair info task.writes regKernelBaseLo regKernelBaseHi "kernelBaseLo" "kernelBaseHi" prefixMsg
  let kSec ← match kTarget with
    | .kernel s => pure s
    | _ => throw s!"{prefixMsg}: kernelBase BAR slot {kSlot} does not reference __KERN_0"
  if !info.relocPatchOffsets.contains kLoPayloadOff.toUInt32 then
    throw s!"{prefixMsg}: missing Mach-O relocation targeting kernelBaseLo payload offset {toHex kLoPayloadOff}"

  let groups := (1 : Nat) <<< f
  let rep := ((1 : Nat) <<< logNE) >>> f
  if rep == 0 || groups * rep > 16 then
    throw s!"{prefixMsg}: invalid chunk replication geometry groups={groups}, rep={rep}"

  let mut expectedSlotMap : Array (Option (Array (Nat × Nat))) := Array.replicate 16 none
  for g in [:groups] do
    let blocks := vGroupBlocks isDepthwise cout f effOcg g
    if !blocks.isEmpty then
      match target with
      | .h17 =>
        let s := g * rep
        if s >= 16 then
          throw s!"{prefixMsg}: expected chunk slot {s} exceeds 0..15"
        expectedSlotMap := expectedSlotMap.set! s (some blocks)
      | .h16 =>
        for j in [:rep] do
          let s := g * rep + j
          if s >= 16 then
            throw s!"{prefixMsg}: expected chunk slot {s} exceeds 0..15"
          expectedSlotMap := expectedSlotMap.set! s (some blocks)

  let cinW := if isDepthwise then 1 else cin
  let taps := kernelSize * kernelSize
  let mut activeChunks := 0

  for s in [:16] do
    let slotW? := findDecodedWrite task.writes (regChunkSlot s)
    let offW? := findDecodedWrite task.writes (regChunkOffset s)
    let sizeW? := findDecodedWrite task.writes (regChunkSize s)
    let expBlocks? := if h : s < expectedSlotMap.size then expectedSlotMap[s] else none

    let slotEnabled ← match slotW? with
      | none => pure false
      | some sw =>
        let en ← decodeFieldChecked chunkSlotLayout sw.value "enable"
        if en == 1 then
          let on20 ← decodeFieldChecked chunkSlotLayout sw.value "on20"
          let slotTask ← decodeFieldChecked chunkSlotLayout sw.value "task"
          if on20 != 1 then
            throw s!"{prefixMsg}: chunk slot {s} missing on20=1 control bit"
          if slotTask.toNat != taskId then
            throw s!"{prefixMsg}: chunk slot {s} task={slotTask} != header taskId={taskId}"
          pure true
        else
          pure false

    if sizeW?.isSome && !slotEnabled then
      throw s!"{prefixMsg}: chunk slot {s} writes size ({toHex (regChunkSize s)}) without enable bit 0 in {toHex (regChunkSlot s)}"
    if slotEnabled then
      let some ow := offW?
        | throw s!"{prefixMsg}: enabled chunk slot {s} missing offset register {toHex (regChunkOffset s)}"
      let some szw := sizeW?
        | throw s!"{prefixMsg}: enabled chunk slot {s} missing size register {toHex (regChunkSize s)}"
      let some blocks := expBlocks?
        | throw s!"{prefixMsg}: spurious chunk slot {s} enabled (not in expected active slot set)"
      let cOff := ow.value.toNat
      let cSz := szw.value.toNat
      if cOff % vDmaLineBytes != 0 then
        throw s!"{prefixMsg}: chunk slot {s} offset {toHex cOff} is not {vDmaLineBytes}-byte aligned"
      if cSz == 0 || cSz % vDmaLineBytes != 0 then
        throw s!"{prefixMsg}: chunk slot {s} size {cSz} must be a positive multiple of {vDmaLineBytes}"
      if cSz > vMaxKernelChunkBytes then
        throw s!"{prefixMsg}: chunk slot {s} size {cSz}B exceeds maximum {vMaxKernelChunkBytes}B (64 KiB)"
      if kernBaseOff + cOff + cSz > kSec.size.toNat then
        throw s!"{prefixMsg}: chunk slot {s} range [{kernBaseOff + cOff}, {kernBaseOff + cOff + cSz}) exceeds kernel size {kSec.size}"

      let groupOuts := vGroupOuts blocks
      if !isSparse then
        let minBytes := alignUp (2 * groupOuts * (cinW * taps + (if biasInKernel then 1 else 0))) vDmaLineBytes
        if cSz < minBytes then
          throw s!"{prefixMsg}: chunk slot {s} size {cSz}B < minimum required {minBytes}B for stride-1 kernel"
      else
        let minBytes := alignUp (2 * (if biasInKernel then groupOuts else 0) + 3 * blocks.size) vDmaLineBytes
        if cSz < minBytes then
          throw s!"{prefixMsg}: chunk slot {s} size {cSz}B < minimum required {minBytes}B for stride-2 sparse kernel"
        if let some kBytes := kernelBytes? then
          vVerifySparseChunkBytes kBytes (kernBaseOff + cOff) (kernBaseOff + cOff + cSz)
            blocks biasInKernel s!"{prefixMsg} chunk slot {s}"

      activeChunks := activeChunks + 1
    else
      if offW?.isSome then
        throw s!"{prefixMsg}: chunk slot {s} writes offset without enable bit 0 in {toHex (regChunkSlot s)}"
      if expBlocks?.isSome then
        throw s!"{prefixMsg}: missing required active chunk slot {s}"

  if activeChunks == 0 then
    throw s!"{prefixMsg}: no active kernel chunk slots enabled"

  -- 9. Verify src1Dep/resDep (BAR-4, code, task, direct), DMA format/control regs, and surface bounds
  let wSrcDep ← requireDecodedWrite task.writes regSrc1Dep "src1Dep"
  if wSrcDep.barSlot != some 4 then
    throw s!"{prefixMsg}: src1Dep (0x4d00) must use BAR slot 4 (got {repr wSrcDep.barSlot})"
  match ← resolveBarTarget info 4 with
  | .configZero => pure ()
  | _ => throw s!"{prefixMsg}: BAR slot 4 must have VM base 0"
  let srcDepC ← decodeFieldChecked src1DepLayout wSrcDep.value "code"
  let srcDepTask ← decodeFieldChecked src1DepLayout wSrcDep.value "task"
  if srcDepC != 0x21 then
    throw s!"{prefixMsg}: src1Dep code {toHex srcDepC.toNat} != expected 0x21"
  if srcDepTask.toNat != taskId then
    throw s!"{prefixMsg}: src1Dep task={srcDepTask} != header taskId={taskId}"

  let wResDep ← requireDecodedWrite task.writes regResDep "resDep"
  if wResDep.barSlot != some 4 then
    throw s!"{prefixMsg}: resDep (0x5100) must use BAR slot 4 (got {repr wResDep.barSlot})"
  let resDepC ← decodeFieldChecked resDepLayout wResDep.value "code"
  let resDepTask ← decodeFieldChecked resDepLayout wResDep.value "task"
  let resDepDirect ← decodeFieldChecked resDepLayout wResDep.value "direct"
  if resDepC != resDepCode target then
    throw s!"{prefixMsg}: resDep code {toHex resDepC.toNat} != expected {toHex (resDepCode target).toNat}"
  if resDepTask.toNat != taskId then
    throw s!"{prefixMsg}: resDep task={resDepTask} != header taskId={taskId}"
  if (resDepDirect == 1) != direct then
    throw s!"{prefixMsg}: resDep direct bit {resDepDirect} != resCfg direct={direct}"

  let wSrc1DmaCfg ← requireDecodedWrite task.writes regSrc1DmaCfg "src1DmaCfg"
  if wSrc1DmaCfg.value != src1DmaCfgValue then
    throw s!"{prefixMsg}: src1DmaCfg {toHex wSrc1DmaCfg.value.toNat} != expected {toHex src1DmaCfgValue.toNat}"
  let wSrc1Fmt ← requireDecodedWrite task.writes regSrc1Fmt "src1Fmt"
  if wSrc1Fmt.value != 0x01002031 then
    throw s!"{prefixMsg}: src1Fmt {toHex wSrc1Fmt.value.toNat} != expected 0x1002031"
  let wResFmt ← requireDecodedWrite task.writes regResFmt "resFmt"
  if wResFmt.value != 0x01302031 then
    throw s!"{prefixMsg}: resFmt {toHex wResFmt.value.toNat} != expected 0x1302031"

  let wInRow ← requireDecodedWrite task.writes regSrc1DmaRowStride "src1DmaRowStride"
  let inRowStride := wInRow.value.toNat
  if inRowStride < 2 * fullW || inRowStride % vDmaLineBytes != 0 then
    throw s!"{prefixMsg}: input DMA row stride {inRowStride} must be >= {2 * fullW} and {vDmaLineBytes}-byte aligned"
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
  let inCap := min inSurf.span.toNat inSurf.sect.size.toNat
  if srcBaseOff + inExtent > inCap then
    throw s!"{prefixMsg}: input DMA range [{toHex srcBaseOff}, {toHex (srcBaseOff + inExtent)}) exceeds bound surface '{inSurf.shortName}' (span={inSurf.span}, section={inSurf.sect.size})"

  let wOutRow ← requireDecodedWrite task.writes regResDmaRowStride "resDmaRowStride"
  let outRowStride := wOutRow.value.toNat
  if outRowStride < 2 * wout || outRowStride % vDmaLineBytes != 0 then
    throw s!"{prefixMsg}: output DMA row stride {outRowStride} must be >= {2 * wout} and {vDmaLineBytes}-byte aligned"
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
  let outCap := min outSurf.span.toNat outSurf.sect.size.toNat
  if resBaseOff + outExtent > outCap then
    throw s!"{prefixMsg}: output DMA range [{toHex resBaseOff}, {toHex (resBaseOff + outExtent)}) exceeds bound surface '{outSurf.shortName}' (span={outSurf.span}, section={outSurf.sect.size})"

  -- 10. Verify output / cache stage block (0x5900..0x592c)
  let wOut0 ← requireDecodedWrite task.writes regOut0 "out0"
  if wOut0.value != out0Value then
    throw s!"{prefixMsg}: out0 {toHex wOut0.value.toNat} != expected {toHex out0Value.toNat}"
  let wOut1 ← requireDecodedWrite task.writes regOut1 "out1"
  if wOut1.value != out1Value then
    throw s!"{prefixMsg}: out1 {toHex wOut1.value.toNat} != expected {toHex out1Value.toNat}"
  let wOut2 ← requireDecodedWrite task.writes regOut2 "out2"
  let out2Form ← decodeFieldChecked out2Layout wOut2.value "form"
  let out2Task ← decodeFieldChecked out2Layout wOut2.value "task"
  if out2Form != 0xe0 then
    throw s!"{prefixMsg}: out2 form {toHex out2Form.toNat} != expected 0xe0"
  if out2Task.toNat != taskId then
    throw s!"{prefixMsg}: out2 task={out2Task} != header taskId={taskId}"
  let wOut3 ← requireDecodedWrite task.writes regOut3 "out3"
  if wOut3.value != out3Value then
    throw s!"{prefixMsg}: out3 {toHex wOut3.value.toNat} != expected {toHex out3Value.toNat}"
  let wOut4 ← requireDecodedWrite task.writes regOut4 "out4"
  if wOut4.value != out4Value then
    throw s!"{prefixMsg}: out4 {toHex wOut4.value.toNat} != expected {toHex out4Value.toNat}"
  let wOut5 ← requireDecodedWrite task.writes regOut5 "out5"
  if wOut5.value != out5Value then
    throw s!"{prefixMsg}: out5 {toHex wOut5.value.toNat} != expected {toHex out5Value.toNat}"

  pure ()

/-- Verify a decoded task is specifically a legal 1x1 conv task. -/
def verifyConv1x1Task (info : HwxInfo) (task : DecodedTask) (kernelBytes? : Option ByteArray := none) :
    Except String Unit := do
  let wConvCfg ← requireDecodedWrite task.writes regConvCfg "convCfg"
  let kw := (← decodeFieldChecked convCfgLayout wConvCfg.value "kw").toNat
  let kh := (← decodeFieldChecked convCfgLayout wConvCfg.value "kh").toNat
  if kw != 1 || kh != 1 then
    throw s!"task {task.taskIndex}: expected 1x1 conv, got {kh}x{kw}"
  verifyConvTask info task kernelBytes?

/-- Verify a decoded task is specifically a legal 3x3 conv task. -/
def verifyConv3x3Task (info : HwxInfo) (task : DecodedTask) (kernelBytes? : Option ByteArray := none) :
    Except String Unit := do
  let wConvCfg ← requireDecodedWrite task.writes regConvCfg "convCfg"
  let kw := (← decodeFieldChecked convCfgLayout wConvCfg.value "kw").toNat
  let kh := (← decodeFieldChecked convCfgLayout wConvCfg.value "kh").toNat
  if kw != 3 || kh != 3 then
    throw s!"task {task.taskIndex}: expected 3x3 conv, got {kh}x{kw}"
  verifyConvTask info task kernelBytes?

/-- Verify all recognized tasks decoded from a validated HWX binary via the `TaskKind` dispatcher.
    When `requireConvTask` is `true`, fails if any task is unknown or if no tasks were verified. -/
def verifyConvHwxTasks (hwxBytes : ByteArray) (info : HwxInfo) (requireConvTask : Bool := true) :
    Except String Nat := do
  let textBytes ← readSlice hwxBytes info.textSection.offset.toNat info.textSection.size.toNat
  let tasks ← decodeTaskStream info.target textBytes
  let kernelBytes? ← match info.kernelSection with
    | some kSec => some <$> readSlice hwxBytes kSec.offset.toNat kSec.size.toNat
    | none => pure none
  let mut checked := 0
  for task in tasks do
    match classifyTask task with
    | .neConv =>
      verifyConvTask info task kernelBytes?
      checked := checked + 1
    | .unknown =>
      if requireConvTask then
        throw s!"task {task.taskIndex}: missing NE conv configuration registers"
  if requireConvTask && checked == 0 then
    throw "HWX contains no NE conv tasks"
  pure checked

end Vane.Verify
