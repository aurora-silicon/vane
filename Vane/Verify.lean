import Vane.Binary
import Vane.TaskStream
import Vane.Hwx
import Vane.Register
import Vane.Reach

namespace Vane.Verify

open Vane.Binary
open Vane.TaskStream
open Vane.Hwx
open Vane.Register
open Vane.Reach

-- ============================================================================
-- Independent TCB hardware constants & geometry helpers (no Vane.Tiling import)
-- ============================================================================

private def vL2CapacityBytes : Nat := 0x200000
private def vMaxKernelChunkBytes : Nat := 0x10000
private def vDmaLineBytes : Nat := 64
private def vLutBytes : Nat := 128
private def vMaxTasks : Nat := 256

private def vCeilDiv (a b : Nat) : Nat :=
  if b == 0 then 0 else (a + b - 1) / b

private def vCeilLog2 (n : Nat) : Nat :=
  if n <= 1 then 0 else Nat.log2 (n - 1) + 1

private def vPeIsDirect (target : Target) (outW k : Nat) : Bool :=
  let stripW := (1 : Nat) <<< k
  if outW > stripW && stripW % 32 != 0 then
    false
  else if target.profile.peForbidK5LastStrip24 then
    let r := alignUp outW 8
    let lastStrip := if r == 0 then 0 else r - 32 * ((r - 1) / 32)
    !(k == 5 && r > 32 && lastStrip == 24)
  else
    true

private def vPeDirectResBytes (outC drow th : Nat) : Nat :=
  alignUp (max 2048 (min (outC * drow * th) (4 * drow * th))) vDmaLineBytes

private def vPePoolDirect (target : Target) (ow k : Nat) : Bool :=
  let stripW := (1 : Nat) <<< k
  let multi := ow > stripW
  !multi || stripW >= target.profile.poolMultiStripMinW

private def vPoolInRows (usedH kh sh pt oh th : Nat) (isNe : Bool) : Nat := Id.run do
  if th == 0 || oh == 0 then
    return 0
  let ke := if isNe then alignUp kh sh else max kh sh
  let nTiles := vCeilDiv oh th
  let mut best := 0
  for i in [:nTiles] do
    let o0 := i * th
    let o1 := min oh (o0 + th)
    if o1 > o0 then
      let lo := o0 * sh - pt
      let hi := min usedH ((o1 - 1) * sh + ke - pt)
      let rows := hi - lo
      if rows > best then
        best := rows
  return best

private def vPoolStripPx (usedW kw sw pl stripW ow : Nat) : Nat :=
  let r := alignUp usedW 8
  if ow <= stripW then r
  else
    let tin := stripW * sw
    let left := pl > 0
    let right := kw - pl > sw
    let extraL := if left then 8 else 0
    let l := max 24 (min 32 (r - 2 * tin))
    let extraR := if right then max l (32 - extraL) else 0
    min r (tin + extraL + extraR)

private def vVerifyPoolAxisGeom
    (fullExt k s padBefore usedExt outExt : Nat) (hasExactFull : Bool)
    (axisLabel prefixMsg : String) : Except String Unit := do
  if k == 0 then
    throw s!"{prefixMsg}: pool {axisLabel} kernel size must be >= 1"
  if s != 1 && s != 2 then
    throw s!"{prefixMsg}: pool {axisLabel} stride {s} must be 1 or 2"
  if padBefore >= k then
    throw s!"{prefixMsg}: pool {axisLabel} padBefore ({padBefore}) >= kernel ({k})"
  if usedExt == 0 || outExt == 0 then
    throw s!"{prefixMsg}: pool {axisLabel} used={usedExt}, out={outExt} must be >= 1"
  if hasExactFull then
    if usedExt > fullExt then
      throw s!"{prefixMsg}: pool {axisLabel} used={usedExt} exceeds input extent {fullExt}"
    if (outExt - 1) * s >= fullExt + padBefore then
      throw s!"{prefixMsg}: pool {axisLabel} output extent {outExt} out of bounds for input {fullExt}, stride {s}, pad {padBefore}"
    let expUsed := min fullExt ((outExt - 1) * s + k - padBefore)
    if usedExt != expUsed then
      throw s!"{prefixMsg}: pool {axisLabel} used={usedExt} != expected {expUsed}"
  else
    if usedExt > (outExt - 1) * s + k - padBefore then
      throw s!"{prefixMsg}: pool {axisLabel} used={usedExt} exceeds window reach {(outExt - 1) * s + k - padBefore}"
    if (outExt - 1) * s >= usedExt + padBefore then
      throw s!"{prefixMsg}: pool {axisLabel} output extent {outExt} out of bounds for used={usedExt}, stride={s}, pad={padBefore}"
  pure ()

private def vNeActDirectResBytes (f fat : Nat) : Nat :=
  ((1 : Nat) <<< (f + fat)) * 512

private def vGroupCode (c : Nat) : Nat :=
  if c <= 16 then 1
  else if c <= 32 then 2
  else if c <= 64 then 4
  else 8

private def vGroupCodeLog2 (c : Nat) : Nat :=
  if c <= 16 then 0
  else if c <= 32 then 1
  else if c <= 64 then 2
  else 3

/-- Explicitly reject unmodelled Linux `accel/ane` DMA registers (`0x4d84`, `0x5150`, `0x5104`)
    that are rejected with `-ENOEXEC` by `ane_hx_dma_modelled`. -/
private def vCheckNoUnmodelledDmaRegs (writes : Array DecodedWrite) (prefixMsg : String) :
    Except String Unit := do
  for w in writes do
    if w.byteAddr == regSrc1DmaCrop || w.byteAddr == regResDmaViewOff || w.byteAddr == regResDmaWarWait then
      throw s!"{prefixMsg}: forbidden unmodelled DMA register {toHex w.byteAddr} (rejected by ane_hx_dma_modelled)"
  pure ()

private def vL2ResidentGeometry (c h w : Nat) : Nat × Nat × Nat :=
  let chs := 2 * alignUp w 8
  let row := c * chs
  let bytes := alignUp (h * row) vDmaLineBytes
  (chs, row, bytes)

/-- Enforce that every decoded register write falls inside the mode-conditional allowlist
    for the verified task kind. -/
private def vCheckAllowedRegs (writes : Array DecodedWrite) (allowed : Array Nat) (prefixMsg : String) :
    Except String Unit := do
  for w in writes do
    if !allowed.contains w.byteAddr then
      throw s!"{prefixMsg}: disallowed register write at {toHex w.byteAddr}"
  pure ()

/-- Independently verify a 128-byte NE LUT table in `__KERN_0`: fits in bounds,
    control words 41 (`k`) and 42 (`mode`) match hardware-documented modes/symmetry flags,
    and reserved words `43..63` (byte offsets `0x56..0x7f`) are all zero. -/
private def vVerifyLutBytes (kBytes : ByteArray) (lutStart : Nat) (label : String) : Except String Unit := do
  if lutStart + vLutBytes > kBytes.size then
    throw s!"{label}: 128-byte LUT range [{lutStart}, {lutStart + vLutBytes}) exceeds __KERN_0 size {kBytes.size}"
  let kWord ← readU16LE kBytes (lutStart + 2 * 41)
  let modeWord ← readU16LE kBytes (lutStart + 2 * 42)
  if (modeWord >>> 8) != 0 then
    throw s!"{label}: LUT mode word 42 ({toHex modeWord.toNat}) has non-zero high byte"
  let base := modeWord &&& 0x1f
  let validBase :=
    base == 0x00 || base == 0x01 || base == 0x02 || base == 0x05 ||
    base == 0x06 || base == 0x07 || base == 0x09 || base == 0x0a || base == 0x11
  if !validBase then
    throw s!"{label}: LUT mode word 42 ({toHex modeWord.toNat}) has unsupported base mode {toHex base.toNat}"
  let sym := modeWord &&& 0xe0
  let validSym :=
    sym == 0x00 || sym == 0x20 || sym == 0x60 ||
    sym == 0x80 || sym == 0xa0 || sym == 0xc0
  if !validSym then
    throw s!"{label}: LUT mode word 42 ({toHex modeWord.toNat}) has unsupported symmetry flag {toHex sym.toNat}"
  let validK := kWord <= 15 || (base == 0x01 && kWord == 0x40)
  if !validK then
    throw s!"{label}: LUT scale word 41 (k={toHex kWord.toNat}) invalid for base mode {toHex base.toNat}"
  for wi in [43 : 64] do
    let w ← readU16LE kBytes (lutStart + 2 * wi)
    if w != 0 then
      throw s!"{label}: LUT reserved word {wi} (offset +{toHex (2 * wi)}) is non-zero ({toHex w.toNat})"
  pure ()

private def vStripW (fat shPref : Nat) : Nat :=
  (128 <<< fat) >>> shPref

private def vFirstDiv (wd base : Nat) : Nat := Id.run do
  for i in [:6] do
    let d := base >>> i
    if d > 0 && wd % d == 0 then
      return i
  return 6

private def vCover (wAligned stripW base : Nat) : Nat :=
  if stripW == 0 || wAligned == 0 then
    6
  else if wAligned <= stripW then
    min 4 (vFirstDiv wAligned base)
  else
    let fullDiv := vFirstDiv stripW base
    let rem := wAligned % stripW
    let mx := if rem == 0 then fullDiv else max fullDiv (vFirstDiv rem base)
    min 4 mx

private def vCoverMin (w stripW fat : Nat) : Nat :=
  let wuBase : Nat := (128 : Nat) <<< fat
  let c8 := vCover (alignUp w 8) stripW wuBase
  let c16 := vCover (alignUp w 16) stripW wuBase
  let c32 := vCover (alignUp w 32) stripW wuBase
  let c64 := vCover (alignUp w 64) stripW wuBase
  min (min c8 c16) (min c32 c64)

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
  target.profile.directThreshold fat shMax >= min 32 wout

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

/-- Independently verify 3x3 `(same | valid | strip)` spatial geometry and return expected `(Win, Hin)`. -/
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
  let matchesStrip :=
    stride == 1 && pl == 1 && wout == fullW && pt <= 1 && hout >= 1 &&
    hout + 2 >= fullH + pt && hout + 2 - fullH - pt <= 1
  if !matchesSame && !matchesValid && !matchesStrip then
    throw s!"3x3 spatial geometry mismatch: input=({fullH}x{fullW}), stride={stride}, output=({hout}x{wout}), pad=(pt={pt}, pl={pl}) matches neither same, valid, nor strip"
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
-- Public task classification, BAR resolution, and multi-task footprint types
-- ============================================================================

/-- High-level task kind recognized by the static safety verifier. -/
inductive TaskKind where
  | neConv
  | neAct
  | neView
  | neMatmul
  | peOp
  | pePool
  | peReduce
  | nePool
  | neChannelReduce
  | unknown
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Hardware execution engine (`ne` = Neural Engine, `pe` = Planar Engine). -/
inductive Engine where
  | ne
  | pe
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Role of a decoded L2 SRAM access within a task. -/
inductive L2AccessRole where
  | src1Stage
  | src2Stage
  | src1L2
  | src2L2
  | resStage
  | resL2
  deriving Repr, BEq, DecidableEq, Inhabited

def L2AccessRole.name : L2AccessRole → String
  | .src1Stage => "src1Stage"
  | .src2Stage => "src2Stage"
  | .src1L2 => "src1L2"
  | .src2L2 => "src2L2"
  | .resStage => "resStage"
  | .resL2 => "resL2"

/-- A verified half-open L2 SRAM access `[lo, hi)` within `[0, 2 MiB)`. -/
structure VerifiedL2Access where
  lo : Nat
  hi : Nat
  isWrite : Bool
  role : L2AccessRole
  deriving Repr, BEq, Inhabited

/-- Resolved DRAM resource identity behind a BAR slot. -/
inductive BarResource where
  | surface (index : Nat) (shortName : String)
  | scratch
  | kernel
  deriving Repr, BEq, DecidableEq, Inhabited

def BarResource.name : BarResource → String
  | .surface idx nm => s!"surface[{idx}]({nm})"
  | .scratch => "scratch(__DATA,__bss)"
  | .kernel => "kernel(__KERN_0)"

/-- Hardware DMA port associated with a DRAM access. -/
inductive DramAccessPort where
  | src1Dma
  | src2Dma
  | resDma
  | kernel
  | lut
  deriving Repr, BEq, DecidableEq, Inhabited

def DramAccessPort.name : DramAccessPort → String
  | .src1Dma => "src1Dma"
  | .src2Dma => "src2Dma"
  | .resDma => "resDma"
  | .kernel => "kernel"
  | .lut => "lut"

/-- Exact written-byte coverage descriptor for a DRAM `resDma` write, used by S5 scratch
    read coverage and `[multiBatchRaw]` superseding. Bounding box `[lo, hi)` remains on
    `VerifiedDramAccess` for conservative hazard detection. -/
inductive WriteCoverage where
  | dense (lo hi : Nat)
  | periodic (start innerLen outerCount outerStride : Nat)
  | sparse
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Classify the exact written byte set of a 4D/5D strided fp16 `resDma` write
    `(batchN, dOut, rch, h, rw)` at `outOff` with strides `(outNSt, outDSt, outPlaneSt, outRowSt)`. -/
def classifyStridedWriteCoverage
    (batchN dOut rch h rw outOff outNSt outDSt outPlaneSt outRowSt : Nat) : WriteCoverage :=
  let rowBytes := alignUp (2 * rw) vDmaLineBytes
  let rowTight := h <= 1 || outRowSt == rowBytes
  let planeTight := rch <= 1 || outPlaneSt == h * rowBytes
  let dTight := dOut <= 1 || outDSt == rch * h * rowBytes
  let nTight := batchN <= 1 || outNSt == dOut * rch * h * rowBytes
  if batchN == 0 || dOut == 0 || rch == 0 || h == 0 || rw == 0 then
    .sparse
  else if rowTight && planeTight && dTight && nTight then
    let totalLen := batchN * dOut * rch * h * rowBytes
    .dense outOff (outOff + totalLen)
  else if batchN == 1 && dOut == 1 && rowTight &&
          outPlaneSt > 0 && h * rowBytes <= outPlaneSt then
    let innerLen := h * rowBytes
    if rch == 1 then .dense outOff (outOff + innerLen)
    else .periodic outOff innerLen rch outPlaneSt
  else if batchN == 1 && dOut == 1 && (rch <= 1 || outPlaneSt == h * outRowSt) &&
          outRowSt > 0 && rowBytes <= outRowSt then
    let outerCount := rch * h
    if outerCount == 1 then .dense outOff (outOff + rowBytes)
    else .periodic outOff rowBytes outerCount outRowSt
  else
    .sparse

/-- A verified half-open DRAM access `[lo, hi)` on a resolved `BarResource`. -/
structure VerifiedDramAccess where
  resource : BarResource
  lo : Nat
  hi : Nat
  isWrite : Bool
  port : DramAccessPort
  coverage : WriteCoverage := .sparse
  deriving Repr, BEq, Inhabited

/-- Decoded synchronization bits and DMA dependency fields for one task. -/
structure TaskSyncInfo where
  src1WaitNe5 : Bool := false
  src1WaitPe22 : Bool := false
  src1HandoffNe : Bool := false
  src1HandoffPe : Bool := false
  src2WaitNe5 : Bool := false
  src2WaitPe22 : Bool := false
  src2HandoffNe : Bool := false
  src2HandoffPe : Bool := false
  src1DepKind : Nat := 0
  src1DepDist : Nat := 0
  src2DepKind : Nat := 0
  src2DepDist : Nat := 0
  hasSrc1Dma : Bool := false
  hasSrc2Dma : Bool := false
  deriving Repr, BEq, Inhabited

/-- Decoded per-task memory footprint and synchronization metadata used by S5 hazard checking. -/
structure VerifiedTaskFootprint where
  taskIndex : Nat
  kind : TaskKind
  engine : Engine
  isNeCompute : Bool
  isMultiBatch : Bool := false
  liveL2Mask : UInt32 := 0
  extProducerTask : Option Nat := none
  l2Accesses : Array VerifiedL2Access
  dramAccesses : Array VerifiedDramAccess
  sync : TaskSyncInfo
  deriving Repr, BEq, Inhabited

/-- Summary counts of verified tasks across all supported HWX task kinds. -/
structure TaskVerifySummary where
  total : Nat
  neConv : Nat
  neAct : Nat
  peOp : Nat
  pePool : Nat := 0
  peReduce : Nat := 0
  nePool : Nat := 0
  neChannelReduce : Nat := 0
  neView : Nat := 0
  neMatmul : Nat := 0
  deriving Repr, BEq, Inhabited

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

/-- Classify a decoded task unambiguously from its emitted machine registers. -/
def classifyTask (task : DecodedTask) : TaskKind :=
  let hasPeTile := (findDecodedWrite task.writes regPeTile).isSome
  let hasNeTile := (findDecodedWrite task.writes regNeTile).isSome
  let hasNeOp := (findDecodedWrite task.writes regNeOp).isSome
  let peCommon? := findDecodedWrite task.writes regPeCommon
  if hasPeTile || (!hasNeTile && !hasNeOp && match peCommon? with
      | some w => w.value == peCommonPoolValue || w.value == peCommonOneSrcValue ||
                  w.value == peCommonTwoSrcValue || w.value == peCommonReduceValue
      | none => false) then
    match peCommon? with
    | some w =>
      if w.value == peCommonPoolValue then .pePool
      else if w.value == peCommonReduceValue then .peReduce
      else .peOp
    | none => .peOp
  else if (findDecodedWrite task.writes regConvCfg).isSome then
    if (findDecodedWrite task.writes regKernelBaseLo).isSome ||
       (findDecodedWrite task.writes regNeCfg).isSome then
      let isDynMatmul := match findDecodedWrite task.writes regNeCfg with
        | some w => (w.value &&& neCfgDynKernBit) != 0
        | none => false
      if isDynMatmul then .neMatmul else .neConv
    else
      .nePool
  else if (findDecodedWrite task.writes regDwGroups).isSome || hasNeTile ||
          (peCommon?.isSome && hasNeOp) then
    let isViewCopyOp := match findDecodedWrite task.writes regNeOp with
      | some w => w.value == neOpViewCopy
      | none => false
    let noKernNeCfg := (findDecodedWrite task.writes regLutCfg).isNone &&
      (findDecodedWrite task.writes regNeCfg).isNone
    if isViewCopyOp && noKernNeCfg then
      let hasTransposeBits := match findDecodedWrite task.writes regNeCommon with
        | some w => (w.value &&& ((3 : UInt32) <<< 28)) != 0
        | none => false
      let has4dViewRegs :=
        (findDecodedWrite task.writes regDin).isSome ||
        (findDecodedWrite task.writes regDout).isSome ||
        (findDecodedWrite task.writes regSrc1DmaDStride).isSome ||
        (findDecodedWrite task.writes regSrc1DmaNStride).isSome ||
        (findDecodedWrite task.writes regResDmaDStride).isSome ||
        (findDecodedWrite task.writes regResDmaNStride).isSome ||
        (match findDecodedWrite task.writes regBatchN with
          | some w => w.value != 1
          | none => false)
      let isLegacyCopy :=
        task.header.getD 4 0 == taskHeaderWord4CopyConst && !hasTransposeBits && !has4dViewRegs
      if isLegacyCopy then .neAct else .neView
    else
      let dwLowZero := match findDecodedWrite task.writes regDwGroups with
        | some w => (w.value &&& 0xffff) == 0
        | none => false
      let isOcg3 := match findDecodedWrite task.writes regNeTile with
        | some w => (w.value &&& 7) == 3
        | none => false
      if dwLowZero || (isOcg3 && noKernNeCfg) then
        .neChannelReduce
      else
        .neAct
  else
    .unknown

/-- Return `true` iff a decoded task is an NE convolution task. -/
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

private def findSurfaceIndex (info : HwxInfo) (surf : ValidatedSurface) : Nat := Id.run do
  for i in [:info.surfaces.size] do
    if info.surfaces[i]!.sect.addr == surf.sect.addr then
      return i
  return 0

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
-- Shared multi-task header, port-config, DMA, L2, and stage verifiers (S1-S4)
-- ============================================================================

private def vCheckNoDuplicateWrites (writes : Array DecodedWrite) (prefixMsg : String) : Except String Unit := do
  let mut seen : Array Nat := #[]
  for w in writes do
    if seen.contains w.byteAddr then
      throw s!"{prefixMsg}: duplicate write to register {toHex w.byteAddr}"
    seen := seen.push w.byteAddr
  pure ()

/-- Verify 9-word single-task or multi-task header and return `(taskId, ene, h7)`. -/
private def vVerifyTaskHeader9
    (target : Target)
    (task : DecodedTask)
    (taskCount : Nat)
    (expectPe : Bool)
    (allowCopyH4 : Bool)
    (prefixMsg : String) : Except String (Nat × Nat × UInt32) := do
  if taskCount == 0 || taskCount > vMaxTasks then
    throw s!"{prefixMsg}: taskCount={taskCount} out of range 1..{vMaxTasks}"
  let isFirst := task.taskIndex == 0
  let isLast := task.taskIndex + 1 == taskCount

  let (h0, h1, h2, h3, h4, h5, h6, h7, h8) ← match task.header.toList with
    | [w0, w1, w2, w3, w4, w5, w6, w7, w8] => pure (w0, w1, w2, w3, w4, w5, w6, w7, w8)
    | _ => throw s!"{prefixMsg}: expected 9-word header, got {task.header.size}"

  let taskId := (← decodeFieldChecked taskHeader0Layout h0 "taskId").toNat
  let taskWords := (← decodeFieldChecked taskHeader0Layout h0 "taskWords").toNat
  if taskId != task.taskIndex then
    throw s!"{prefixMsg}: header[0] taskId={taskId} != task index {task.taskIndex}"
  if taskWords < 9 || taskWords > 0x7ff then
    throw s!"{prefixMsg}: header[0] taskWords={taskWords} out of range 9..{0x7ff}"
  if h1 != 0 then
    throw s!"{prefixMsg}: header[1] must be 0, got {toHex h1.toNat}"

  let h2First ← decodeFieldChecked taskHeader2Layout h2 "firstFlags"
  let h2Last ← decodeFieldChecked taskHeader2Layout h2 "lastFlag"
  let expFirstFlags : UInt32 := if isFirst then taskHeader2FirstFlags else 0
  let expLastFlag : UInt32 := if isLast then 1 else 0
  if h2First != expFirstFlags || h2Last != expLastFlag then
    throw s!"{prefixMsg}: header[2] {toHex h2.toNat} mismatches isFirst={isFirst}, isLast={isLast}"

  if h3 != 0 then
    throw s!"{prefixMsg}: header[3] must be 0, got {toHex h3.toNat}"

  let expH4 := taskHeaderWord4Multi target isFirst isLast
  let h4Ok := h4 == expH4 || (allowCopyH4 && isFirst && isLast && h4 == taskHeaderWord4CopyConst)
  if !h4Ok then
    throw s!"{prefixMsg}: header[4] {toHex h4.toNat} != expected {toHex expH4.toNat}"

  if h5 != 0 || h6 != 0 then
    throw s!"{prefixMsg}: header[5..7] reserved words header[5..6] must be 0 (got {toHex h5.toNat}, {toHex h6.toNat})"
  if isLast && h7 != 0 then
    throw s!"{prefixMsg}: header[7] liveL2Mask must be 0 on final task, got {toHex h7.toNat}"

  let h8Active ← decodeFieldChecked taskHeader8MultiLayout h8 "active"
  let h8Last ← decodeFieldChecked taskHeader8MultiLayout h8 "isLast"
  let ene ← decodeFieldChecked taskHeader8MultiLayout h8 "ene"
  if h8Active != 1 then
    throw s!"{prefixMsg}: header[8] active bit must be 1, got {h8Active}"
  if h8Last != expLastFlag then
    throw s!"{prefixMsg}: header[8] isLast bit {h8Last} mismatches isLast={isLast}"
  if expectPe then
    if ene != 0 then
      throw s!"{prefixMsg}: PE task requires header[8] ene=0, got ene={ene}"
  else
    if ene < 1 || ene > 5 then
      throw s!"{prefixMsg}: header[8] ene={ene} out of range 1..5"
  pure (taskId, ene.toNat, h7)

/-- Verify 9-word or 10-word extended task header for dynamic runtime-kernel NE matmul tasks.
    Returns `(taskId, ene, h7, extProducerTask?)`. -/
private def vVerifyTaskHeaderMatmul
    (target : Target)
    (task : DecodedTask)
    (taskCount : Nat)
    (prefixMsg : String) : Except String (Nat × Nat × UInt32 × Option Nat) := do
  if task.header.size == 9 then
    let (taskId, ene, h7) ← vVerifyTaskHeader9 target task taskCount false false prefixMsg
    pure (taskId, ene, h7, none)
  else if task.header.size == 10 then
    if taskCount == 0 || taskCount > vMaxTasks then
      throw s!"{prefixMsg}: taskCount={taskCount} out of range 1..{vMaxTasks}"
    let isFirst := task.taskIndex == 0
    let isLast := task.taskIndex + 1 == taskCount
    let (h0, h1, h2, h3, h4, h5, h6, h7, h8, h9) ← match task.header.toList with
      | [w0, w1, w2, w3, w4, w5, w6, w7, w8, w9] => pure (w0, w1, w2, w3, w4, w5, w6, w7, w8, w9)
      | _ => throw s!"{prefixMsg}: expected 10-word extended header, got {task.header.size}"
    let taskId := (← decodeFieldChecked taskHeader0Layout h0 "taskId").toNat
    let taskWords := (← decodeFieldChecked taskHeader0Layout h0 "taskWords").toNat
    if taskId != task.taskIndex then
      throw s!"{prefixMsg}: header[0] taskId={taskId} != task index {task.taskIndex}"
    if taskWords < 10 || taskWords > 0x7ff then
      throw s!"{prefixMsg}: header[0] taskWords={taskWords} out of range 10..{0x7ff}"
    if h1 != 0 then
      throw s!"{prefixMsg}: header[1] must be 0, got {toHex h1.toNat}"
    let h2First ← decodeFieldChecked taskHeader2Layout h2 "firstFlags"
    let h2Last ← decodeFieldChecked taskHeader2Layout h2 "lastFlag"
    let expFirstFlags : UInt32 := if isFirst then taskHeader2FirstFlags else 0
    let expLastFlag : UInt32 := if isLast then 1 else 0
    if h2First != expFirstFlags || h2Last != expLastFlag then
      throw s!"{prefixMsg}: header[2] {toHex h2.toNat} mismatches isFirst={isFirst}, isLast={isLast}"
    if h3 != 0 then
      throw s!"{prefixMsg}: header[3] must be 0, got {toHex h3.toNat}"
    let expH4 := taskHeaderWord4Multi target isFirst isLast
    if h4 != expH4 then
      throw s!"{prefixMsg}: header[4] {toHex h4.toNat} != expected {toHex expH4.toNat}"
    if h5 != 0 || h6 != 0 then
      throw s!"{prefixMsg}: header[5..6] reserved words must be 0 (got {toHex h5.toNat}, {toHex h6.toNat})"
    if isLast && h7 != 0 then
      throw s!"{prefixMsg}: header[7] liveL2Mask must be 0 on final task, got {toHex h7.toNat}"
    let h8Active ← decodeFieldChecked taskHeader8ExtLayout h8 "active"
    let h8Ext ← decodeFieldChecked taskHeader8ExtLayout h8 "hasExtHdr"
    let h8Last ← decodeFieldChecked taskHeader8ExtLayout h8 "isLast"
    let ene ← decodeFieldChecked taskHeader8ExtLayout h8 "ene"
    if h8Active != 1 || h8Ext != 1 then
      throw s!"{prefixMsg}: 10-word header[8] must set active=1 and hasExtHdr=1 (got {toHex h8.toNat})"
    if h8Last != expLastFlag then
      throw s!"{prefixMsg}: header[8] isLast bit {h8Last} mismatches isLast={isLast}"
    if ene < 1 || ene > 5 then
      throw s!"{prefixMsg}: header[8] ene={ene} out of range 1..5"
    let prodTaskId := h9.toNat
    if prodTaskId >= taskId then
      throw s!"{prefixMsg}: extended header[9] producerTaskId={prodTaskId} must be < taskId={taskId}"
    pure (taskId, ene.toNat, h7, some prodTaskId)
  else
    throw s!"{prefixMsg}: expected 9-word or 10-word header, got {task.header.size}"

private structure DecodedSrcPortCfg where
  isL2 : Bool
  waitNe5 : Bool
  waitPe22 : Bool
  handoffNe : Bool
  handoffPe : Bool
  deriving Inhabited

/-- Decode and strictly validate `0x4104` (`src1Cfg`) or `0x4108` (`src2Cfg`) in single- or multi-task mode. -/
private def vDecodeSrcPortCfg
    (regName : String) (word : UInt32)
    (isFirst hasPriorNe hasPriorPe : Bool)
    (prefixMsg : String) : Except String DecodedSrcPortCfg := do
  let layout := portCfgMultiLayout regName
  let mode ← decodeFieldChecked layout word "mode"
  let handoffNe ← decodeFieldChecked layout word "handoffNe"
  let handoffPe ← decodeFieldChecked layout word "handoffPe"
  let waitNe4 ← decodeFieldChecked layout word "waitNe4"
  let waitNe5 ← decodeFieldChecked layout word "waitNe5"
  let dmaFlag ← decodeFieldChecked layout word "dmaFlag"
  let p ← decodeFieldChecked layout word "P"
  let waitPe20 ← decodeFieldChecked layout word "waitPe20"
  let waitPe22 ← decodeFieldChecked layout word "waitPe22"
  if p != 1 then
    throw s!"{prefixMsg}: {regName} P={p} must be 1"
  if mode != portModeL2 && mode != portModeDma then
    throw s!"{prefixMsg}: {regName} mode={mode} must be 0 (L2) or 2 (DMA)"
  let isL2 := mode == portModeL2
  if isFirst then
    if isL2 then
      throw s!"{prefixMsg}: {regName} on first task cannot use L2-resident mode=0"
    if dmaFlag != 1 || waitNe4 != 1 || waitNe5 != 1 || waitPe20 != 1 || waitPe22 != 1 ||
       handoffNe != 0 || handoffPe != 0 then
      throw s!"{prefixMsg}: {regName} {toHex word.toNat} violates first-task DMA flags"
    pure { isL2 := false, waitNe5 := false, waitPe22 := false, handoffNe := false, handoffPe := false }
  else
    if waitNe4 != 0 || waitPe20 != 0 then
      throw s!"{prefixMsg}: {regName} on taskId > 0 must not set weak wait bits 4/20 ({toHex word.toNat})"
    if isL2 then
      if dmaFlag != 0 then
        throw s!"{prefixMsg}: {regName} in L2 mode=0 must have dmaFlag=0 ({toHex word.toNat})"
    else
      if dmaFlag != 1 then
        throw s!"{prefixMsg}: {regName} in DMA mode=2 must have dmaFlag=1 ({toHex word.toNat})"
      if handoffNe != 0 || handoffPe != 0 then
        throw s!"{prefixMsg}: {regName} in DMA mode=2 must not set L2 handoff bits ({toHex word.toNat})"
    if waitNe5 == 1 && !hasPriorNe then
      throw s!"{prefixMsg}: {regName} sets dangling wait bit 5 with no prior NE task"
    if waitPe22 == 1 && !hasPriorPe then
      throw s!"{prefixMsg}: {regName} sets dangling wait bit 22 with no prior PE task"
    if handoffNe == 1 && !hasPriorNe then
      throw s!"{prefixMsg}: {regName} sets dangling L2 handoff bit 2 with no prior NE task"
    if handoffPe == 1 && !hasPriorPe then
      throw s!"{prefixMsg}: {regName} sets dangling L2 handoff bit 3 with no prior PE task"
    pure {
      isL2 := isL2
      waitNe5 := waitNe5 == 1
      waitPe22 := waitPe22 == 1
      handoffNe := handoffNe == 1
      handoffPe := handoffPe == 1
    }

/-- Decode and validate `0x4148` (`resCfg`) in single- or multi-task mode. Returns `(isL2, direct)`. -/
private def vDecodeResPortCfg
    (word : UInt32) (isFirst isLast : Bool) (prefixMsg : String) : Except String (Bool × Bool) := do
  let mode ← decodeFieldChecked resCfgMultiLayout word "mode"
  let direct ← decodeFieldChecked resCfgMultiLayout word "direct"
  let waitNe4 ← decodeFieldChecked resCfgMultiLayout word "waitNe4"
  let waitNe5 ← decodeFieldChecked resCfgMultiLayout word "waitNe5"
  let dmaFlag ← decodeFieldChecked resCfgMultiLayout word "dmaFlag"
  let p ← decodeFieldChecked resCfgMultiLayout word "P"
  let waitPe20 ← decodeFieldChecked resCfgMultiLayout word "waitPe20"
  let waitPe22 ← decodeFieldChecked resCfgMultiLayout word "waitPe22"
  if p != 1 then
    throw s!"{prefixMsg}: resCfg P={p} must be 1"
  if dmaFlag != 1 then
    throw s!"{prefixMsg}: resCfg dmaFlag must be 1 ({toHex word.toNat})"
  if mode != portModeL2 && mode != portModeDma then
    throw s!"{prefixMsg}: resCfg mode={mode} must be 0 (L2) or 2 (DMA)"
  let isL2 := mode == portModeL2
  if isLast && isL2 then
    throw s!"{prefixMsg}: final task resCfg must use DMA mode=2 (cannot leave output L2-resident)"
  if isL2 && direct != 0 then
    throw s!"{prefixMsg}: resCfg in L2-resident mode=0 must have direct=0"
  if isFirst then
    if waitNe4 != 1 || waitNe5 != 1 || waitPe20 != 1 || waitPe22 != 1 then
      throw s!"{prefixMsg}: first-task resCfg {toHex word.toNat} missing first-task mask bits"
  else
    if waitNe4 != 0 || waitNe5 != 0 || waitPe20 != 0 || waitPe22 != 0 then
      throw s!"{prefixMsg}: resCfg on taskId > 0 must clear wait bits ({toHex word.toNat})"
  pure (isL2, direct == 1)

/-- Decode and validate `0x4d00` (`src1Dep`) or `0x4d04` (`src2Dep`) and matching `dmaCfg`.
    Returns `(depKind, depDist)`. -/
private def vDecodeSrcDep
    (info : HwxInfo) (writes : Array DecodedWrite)
    (depAddr cfgAddr : Nat) (depName cfgName : String)
    (taskId : Nat) (isFirst : Bool) (prefixMsg : String) : Except String (Nat × Nat) := do
  match ← resolveBarTarget info barSlotConfigZero with
  | .configZero => pure ()
  | _ => throw s!"{prefixMsg}: BAR slot {barSlotConfigZero} must have VM base 0"
  let wDep ← requireDecodedWrite writes depAddr depName
  if wDep.barSlot != some barSlotConfigZero then
    throw s!"{prefixMsg}: {depName} ({toHex depAddr}) must use BAR slot {barSlotConfigZero}"
  let layout := srcDepMultiLayout depName
  let code ← decodeFieldChecked layout wDep.value "code"
  let depTask ← decodeFieldChecked layout wDep.value "task"
  let depDist := (← decodeFieldChecked layout wDep.value "depDist").toNat
  let depKind := (← decodeFieldChecked layout wDep.value "depKind").toNat
  if depTask.toNat != taskId then
    throw s!"{prefixMsg}: {depName} task={depTask} != header taskId={taskId}"
  if depKind > 2 then
    throw s!"{prefixMsg}: {depName} depKind={depKind} must be 0, 1, or 2"
  if isFirst && depKind != 0 then
    throw s!"{prefixMsg}: {depName} on first task must have depKind=0, got {depKind}"
  let wCfg ← requireDecodedWrite writes cfgAddr cfgName
  if code == srcDepCodeDma then
    if wCfg.value != src1DmaCfgValue then
      throw s!"{prefixMsg}: {cfgName} {toHex wCfg.value.toNat} != expected {toHex src1DmaCfgValue.toNat}"
  else if code == srcDepCodePrev && !isFirst then
    if wCfg.value != srcDmaCfgPrev then
      throw s!"{prefixMsg}: {cfgName} {toHex wCfg.value.toNat} != expected {toHex srcDmaCfgPrev.toNat} for code=0x41"
  else
    throw s!"{prefixMsg}: {depName} code {toHex code.toNat} invalid"
  pure (depKind, depDist)

/-- Verify a DMA stride pair `(rowStride, planeStride)` for shape `(c, h, w)` against a
    capacity bound `cap` and return the verified exact strided byte `(extent, coverage)`.
    When `isSurface` and `c > 1`, additionally enforces `(baseOff % planeStride) + h * rowStride <= planeStride`
    to prevent inter-channel plane bleed. -/
private def vVerifyDmaStridesAndBounds
    (writes : Array DecodedWrite)
    (c h w baseOff cap : Nat)
    (isSurface : Bool)
    (rowAddr planeAddr : Nat)
    (rowName planeName dirLabel targetLabel prefixMsg : String) : Except String (Nat × WriteCoverage) := do
  if baseOff % vDmaLineBytes != 0 then
    throw s!"{prefixMsg}: {dirLabel} {targetLabel} base offset {toHex baseOff} is not {vDmaLineBytes}-byte aligned"
  let rowStride ← match findDecodedWrite writes rowAddr with
    | some wr =>
      let rs := wr.value.toNat
      if rs < 2 * w || rs % vDmaLineBytes != 0 then
        throw s!"{prefixMsg}: {dirLabel} row stride {rowName} ({rs}) must be >= {2 * w} and {vDmaLineBytes}-byte aligned"
      pure rs
    | none =>
      if h == 1 then pure (alignUp (2 * w) vDmaLineBytes)
      else throw s!"{prefixMsg}: missing required register {rowName} ({toHex rowAddr}) for H={h} > 1"
  let planeStride ← match findDecodedWrite writes planeAddr with
    | some wp =>
      let ps := wp.value.toNat
      if ps < h * rowStride then
        throw s!"{prefixMsg}: {dirLabel} plane stride ({planeName}={ps}) < H * rowStride ({h * rowStride})"
      pure ps
    | none =>
      if c == 1 then pure (h * rowStride)
      else throw s!"{prefixMsg}: missing required register {planeName} ({toHex planeAddr}) for C={c} > 1"
  let extent := if c == 0 || h == 0 then 0 else (c - 1) * planeStride + h * rowStride
  if extent == 0 || baseOff + extent > cap then
    throw s!"{prefixMsg}: {dirLabel} range [{toHex baseOff}, {toHex (baseOff + extent)}) exceeds {targetLabel} capacity {cap}"
  if isSurface && c > 1 && planeStride > 0 && (baseOff % planeStride) + h * rowStride > planeStride then
    throw s!"{prefixMsg}: {dirLabel} {targetLabel} plane bleed: (baseOff % planeStride) ({toHex (baseOff % planeStride)}) + H * rowStride ({h * rowStride}) > planeStride ({planeStride})"
  let cov := classifyStridedWriteCoverage 1 1 c h w baseOff 0 0 planeStride rowStride
  pure (extent, cov)

/-- Verify that a set of register addresses are all absent (omitted). -/
private def vCheckOmitted (writes : Array DecodedWrite) (addrs : Array (Nat × String)) (reason prefixMsg : String) :
    Except String Unit := do
  for (addr, nm) in addrs do
    if (findDecodedWrite writes addr).isSome then
      throw s!"{prefixMsg}: {reason} must omit {nm} ({toHex addr})"
  pure ()

/-- Verify the shared output / cache stage block `0x5900..0x592c` (present iff `isLast`). -/
private def vVerifyOutputStageMulti
    (writes : Array DecodedWrite) (taskId : Nat) (isLast : Bool) (prefixMsg : String) : Except String Unit := do
  if !isLast then
    vCheckOmitted writes #[
      (regOut0, "out0"), (regOut1, "out1"), (regOut2, "out2"),
      (regOut3, "out3"), (regOut4, "out4"), (regOut5, "out5")
    ] "non-last task" prefixMsg
    return ()
  let wOut0 ← requireDecodedWrite writes regOut0 "out0"
  if wOut0.value != out0Value then
    throw s!"{prefixMsg}: out0 {toHex wOut0.value.toNat} != expected {toHex out0Value.toNat}"
  let wOut1 ← requireDecodedWrite writes regOut1 "out1"
  if wOut1.value != out1Value then
    throw s!"{prefixMsg}: out1 {toHex wOut1.value.toNat} != expected {toHex out1Value.toNat}"
  let wOut2 ← requireDecodedWrite writes regOut2 "out2"
  let out2Form ← decodeFieldChecked out2Layout wOut2.value "form"
  let out2Task ← decodeFieldChecked out2Layout wOut2.value "task"
  if out2Form != out2FormValue then
    throw s!"{prefixMsg}: out2 form {toHex out2Form.toNat} != expected {toHex out2FormValue.toNat}"
  if out2Task.toNat != taskId then
    throw s!"{prefixMsg}: out2 task={out2Task} != header taskId={taskId}"
  let wOut3 ← requireDecodedWrite writes regOut3 "out3"
  if wOut3.value != out3Value then
    throw s!"{prefixMsg}: out3 {toHex wOut3.value.toNat} != expected {toHex out3Value.toNat}"
  let wOut4 ← requireDecodedWrite writes regOut4 "out4"
  if wOut4.value != out4Value then
    throw s!"{prefixMsg}: out4 {toHex wOut4.value.toNat} != expected {toHex out4Value.toNat}"
  let wOut5 ← requireDecodedWrite writes regOut5 "out5"
  if wOut5.value != out5Value then
    throw s!"{prefixMsg}: out5 {toHex wOut5.value.toNat} != expected {toHex out5Value.toNat}"
  pure ()

/-- Verify `0x5500` (`kernelStage`) in single- or multi-task mode. -/
private def vVerifyKernelStageMulti
    (writes : Array DecodedWrite) (taskId : Nat) (isLast usesKernel isSparse : Bool)
    (prefixMsg : String) : Except String Unit := do
  if usesKernel then
    let wKs ← requireDecodedWrite writes regKernelStage "kernelStage"
    let sp ← decodeFieldChecked kernelStageMultiLayout wKs.value "sparse"
    let ke ← decodeFieldChecked kernelStageMultiLayout wKs.value "kernEn"
    let il ← decodeFieldChecked kernelStageMultiLayout wKs.value "isLast"
    let kc ← decodeFieldChecked kernelStageMultiLayout wKs.value "kernCfg"
    let st ← decodeFieldChecked kernelStageMultiLayout wKs.value "stage"
    let expSp : UInt32 := if isSparse then 1 else 0
    let expIl : UInt32 := if isLast then 1 else 0
    if ke != 1 || kc != 1 then
      throw s!"{prefixMsg}: kernelStage (0x5500) must set kernEn=1 and kernCfg=1 ({toHex wKs.value.toNat})"
    if sp != expSp then
      throw s!"{prefixMsg}: kernelStage sparse={sp} != expected {expSp}"
    if il != expIl then
      throw s!"{prefixMsg}: kernelStage isLast={il} != expected {expIl}"
    if taskId + 1 > 255 then
      throw s!"{prefixMsg}: kernel-bearing taskId={taskId} overflows 8-bit stage={taskId + 1}"
    if st.toNat != taskId + 1 then
      throw s!"{prefixMsg}: kernelStage stage={st} != expected {taskId + 1}"
  else
    if isLast then
      let wKs ← requireDecodedWrite writes regKernelStage "kernelStage"
      if wKs.value != kernelStageNoKernValue then
        throw s!"{prefixMsg}: non-kernel final task kernelStage (0x5500) must be {toHex kernelStageNoKernValue.toNat}, got {toHex wKs.value.toNat}"
    else
      vCheckOmitted writes #[(regKernelStage, "kernelStage")] "non-last weightless task" prefixMsg
  pure ()

/-- Verify intra-task L2 SRAM bounds (`[0, 2 MiB)`), 64-byte alignment, and disjointness.
    Every writing access (`src1Stage`, `src2Stage`, `resStage`, `resL2`) must be disjoint from
    every other L2 access in the same task. -/
private def vCheckTaskL2Accesses (accs : Array VerifiedL2Access) (prefixMsg : String) : Except String Unit := do
  for a in accs do
    if a.lo >= a.hi then
      throw s!"{prefixMsg}: L2 access {a.role.name} is empty ([{toHex a.lo}, {toHex a.hi}))"
    if a.hi > vL2CapacityBytes then
      throw s!"{prefixMsg}: L2 SRAM range '{a.role.name}' [{toHex a.lo}, {toHex a.hi}) exceeds 2 MiB (exceeds limit {vL2CapacityBytes})"
    if a.lo % vDmaLineBytes != 0 then
      let regLbl := match a.role with
        | .src1Stage | .src1L2 => "src1Base"
        | .src2Stage | .src2L2 => "src2Base"
        | .resStage | .resL2 => "resBase"
      throw s!"{prefixMsg}: L2 {regLbl} ({a.role.name}) {toHex a.lo} is not {vDmaLineBytes}-byte aligned"
  for i in [:accs.size] do
    let a := accs[i]!
    for j in [i + 1 : accs.size] do
      let b := accs[j]!
      if (a.isWrite || b.isWrite) && a.lo < b.hi && b.lo < a.hi then
        throw s!"{prefixMsg}: overlapping L2 ranges '{a.role.name}' [{toHex a.lo}, {toHex a.hi}) and '{b.role.name}' [{toHex b.lo}, {toHex b.hi})"
  pure ()

-- ============================================================================
-- Multi-Task NE Conv static safety verifier (S1-S4)
-- ============================================================================

/-- Independent static safety verifier for decoded NE conv tasks (`1x1` and `3x3`,
    dense and depthwise, stride 1 and 2) in both single-task and multi-task streams. -/
def verifyConvTaskMulti
    (taskCount : Nat) (hasPriorNe hasPriorPe : Bool)
    (info : HwxInfo) (task : DecodedTask) (kernelBytes? : Option ByteArray := none) :
    Except String VerifiedTaskFootprint := do
  let target := info.target
  let prefixMsg := s!"task {task.taskIndex}"
  let isFirst := task.taskIndex == 0
  let isLast := task.taskIndex + 1 == taskCount

  -- 1. Verify 9-word multi-task header
  let (taskId, ene, h7) ← vVerifyTaskHeader9 target task taskCount false false prefixMsg
  let logNE := ene - 1

  -- 2. Reject duplicate register writes and unmodelled DMA registers
  vCheckNoDuplicateWrites task.writes prefixMsg
  vCheckNoUnmodelledDmaRegs task.writes prefixMsg

  -- 3. Decode common shape & control registers
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

  if cin == 0 || cout == 0 || hout == 0 || wout == 0 || hinReg == 0 || winReg == 0 then
    throw s!"{prefixMsg}: conv dimensions must all be >= 1"

  -- 4. Decode L2/DMA port modes (`src1Cfg`, `src2Cfg`, `resCfg`)
  let wSrc1Cfg ← requireDecodedWrite task.writes regSrc1Cfg "src1Cfg"
  let src1Cfg ← vDecodeSrcPortCfg "src1Cfg" wSrc1Cfg.value isFirst hasPriorNe hasPriorPe prefixMsg
  if isFirst then
    let wSrc2Cfg ← requireDecodedWrite task.writes regSrc2Cfg "src2Cfg"
    if wSrc2Cfg.value != src2CfgValue then
      throw s!"{prefixMsg}: src2Cfg {toHex wSrc2Cfg.value.toNat} != {toHex src2CfgValue.toNat}"
  else
    vCheckOmitted task.writes #[(regSrc2Cfg, "src2Cfg")] "NE conv taskId > 0" prefixMsg

  let wResCfg ← requireDecodedWrite task.writes regResCfg "resCfg"
  let (resIsL2, direct) ← vDecodeResPortCfg wResCfg.value isFirst isLast prefixMsg

  -- 5. Resolve DMA endpoints (when DMA-backed) and determine full input dims `(fullH, fullW)`
  let mut dramAccs : Array VerifiedDramAccess := #[]
  let mut sync : TaskSyncInfo := {
    src1WaitNe5 := src1Cfg.waitNe5
    src1WaitPe22 := src1Cfg.waitPe22
    src1HandoffNe := src1Cfg.handoffNe
    src1HandoffPe := src1Cfg.handoffPe
    hasSrc1Dma := !src1Cfg.isL2
  }

  let (fullH, fullW, src1DmaTarget?) ←
    if src1Cfg.isL2 then
      vCheckOmitted task.writes #[
        (regSrc1Dep, "src1Dep"), (regSrc1DmaCfg, "src1DmaCfg"),
        (regSrc1BaseLo, "src1BaseLo"), (regSrc1BaseHi, "src1BaseHi"),
        (regSrc1DmaRowStride, "src1DmaRowStride"), (regSrc1DmaPlaneStride, "src1DmaPlaneStride"),
        (regSrc1Fmt, "src1Fmt")
      ] "L2-resident src1" prefixMsg
      pure (hinReg, winReg, none)
    else
      let (inBarSlot, inTarget, srcBaseOff, _) ←
        resolveBasePair info task.writes regSrc1BaseLo regSrc1BaseHi "src1BaseLo" "src1BaseHi" prefixMsg
      match inTarget with
      | .surface s =>
        if s.direction != .input then
          throw s!"{prefixMsg}: src1Base BAR slot {inBarSlot} bound to non-input surface '{s.shortName}'"
        if s.n != 1 then
          throw s!"{prefixMsg}: single-batch input surface required (got n={s.n})"
        let inChanOk := if kw == 1 && kh == 1 then cin <= s.channels.toNat else s.channels.toNat == cin
        if !inChanOk then
          throw s!"{prefixMsg}: Cin={cin} incompatible with bound input surface channels {s.channels}"
        let sfh := s.height.toNat
        let sfw := s.width.toNat
        if sfh == 0 || sfh > 65535 || sfw == 0 || sfw > 65535 then
          throw s!"{prefixMsg}: bound input surface dims ({sfh}x{sfw}) outside 1..65535"
        let fh ←
          if kw == 1 || (kw == 3 && sx == 1) then
            if hinReg > sfh then
              throw s!"{prefixMsg}: Hin={hinReg} exceeds bound input surface height {sfh}"
            pure hinReg
          else
            pure sfh
        pure (fh, sfw, some (inBarSlot, inTarget, srcBaseOff))
      | .scratch _ =>
        if inBarSlot != barSlotScratch then
          throw s!"{prefixMsg}: scratch src1Base must use BAR slot {barSlotScratch}, got {inBarSlot}"
        if isFirst then
          throw s!"{prefixMsg}: first task cannot read uninitialized DRAM scratch"
        pure (hinReg, winReg, some (inBarSlot, inTarget, srcBaseOff))
      | _ =>
        throw s!"{prefixMsg}: src1Base BAR slot {inBarSlot} does not reference an input surface or scratch"

  let resDmaTarget? ←
    if resIsL2 then
      vCheckOmitted task.writes #[
        (regResDep, "resDep"), (regResBaseLo, "resBaseLo"), (regResBaseHi, "resBaseHi"),
        (regResDmaRowStride, "resDmaRowStride"), (regResDmaPlaneStride, "resDmaPlaneStride"),
        (regResFmt, "resFmt")
      ] "L2-resident res" prefixMsg
      pure none
    else
      let (outBarSlot, outTarget, resBaseOff, _) ←
        resolveBasePair info task.writes regResBaseLo regResBaseHi "resBaseLo" "resBaseHi" prefixMsg
      match outTarget with
      | .surface s =>
        if s.direction != .output then
          throw s!"{prefixMsg}: resBase BAR slot {outBarSlot} bound to non-output surface '{s.shortName}'"
        if s.n != 1 then
          throw s!"{prefixMsg}: single-batch output surface required (got n={s.n})"
        let outHeightOk :=
          if kw == 3 && sx == 2 then s.height.toNat == hout
          else hout <= s.height.toNat
        let outChanOk := if kw == 1 && kh == 1 then cout <= s.channels.toNat else s.channels.toNat == cout
        if !outChanOk || !outHeightOk || s.width.toNat != wout then
          throw s!"{prefixMsg}: output dims ({cout}x{hout}x{wout}) incompatible with bound output surface ({s.channels}x{s.height}x{s.width})"
        pure (some (outBarSlot, outTarget, resBaseOff))
      | .scratch _ =>
        if outBarSlot != barSlotScratch then
          throw s!"{prefixMsg}: scratch resBase must use BAR slot {barSlotScratch}, got {outBarSlot}"
        if isLast then
          throw s!"{prefixMsg}: final task cannot write to DRAM scratch"
        pure (some (outBarSlot, outTarget, resBaseOff))
      | _ =>
        throw s!"{prefixMsg}: resBase BAR slot {outBarSlot} does not reference an output surface or scratch"

  -- 6. Kernel-specific geometry, dwGroups, ocg/coverage safety, and wide3x3 precondition
  let (kernelSize, stride, isDepthwise) ←
    if kw == 1 && kh == 1 then
      if sx != 1 || sy != 1 || pl != 0 || pt != 0 || hx != 1 || vy != 1 then
        throw s!"{prefixMsg}: convCfg {toHex wConvCfg.value.toNat} is not a stride-1 1x1 convolution"
      if wout != winReg || hout != hinReg || winReg != fullW || hinReg != fullH then
        throw s!"{prefixMsg}: output/input spatial dims ({hout}x{wout}, {hinReg}x{winReg}) != input ({fullH}x{fullW}) for 1x1 conv"
      if wide3x3Bit != 0 then
        throw s!"{prefixMsg}: neCommon wide3x3={wide3x3Bit} must be 0 for 1x1 conv"
      if (findDecodedWrite task.writes regDwGroups).isSome then
        throw s!"{prefixMsg}: dwGroups (0x0030) must not be written for 1x1 conv"
      if ocg > 4 then
        throw s!"{prefixMsg}: ocg={ocg} exceeds maximum 4"
      let maxCout := target.profile.conv1x1MaxCout f ocg
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
        let prof := target.profile
        let maxOcg := if s == 1 then prof.dense3x3Stride1MaxOcg else 4
        if ocg > maxOcg then
          throw s!"{prefixMsg}: {prof.name} dense 3x3 stride-{s} ocg={ocg} exceeds maximum {maxOcg}"
        let singleCap := (1 : Nat) <<< (f + ocg)
        let multiRoundOk := s == 1 && prof.dense3x3Stride1MultiRoundF4Ocg3 && f == 4 && ocg == 3
        if cout > singleCap && !multiRoundOk then
          throw s!"{prefixMsg}: {prof.name} dense 3x3 stride-{s} Cout={cout} exceeds coverage {singleCap} (F={f}, ocg={ocg})"
      pure (3, s, isDw)
    else
      throw s!"{prefixMsg}: unsupported conv kernel size kh={kh}, kw={kw} (expected 1x1 or 3x3)"

  let effOcg := if isDepthwise then 0 else ocg
  let isSparse := kernelSize == 3 && stride == 2

  if kernelSize == 3 && wide3x3Bit == 1 then
    if !target.profile.supportsWide3x3 then
      throw s!"{prefixMsg}: neCommon wide3x3=1 is illegal on {target.name}"
    if isDepthwise then
      throw s!"{prefixMsg}: neCommon wide3x3=1 is illegal for depthwise 3x3 conv"
    if stride != 1 then
      throw s!"{prefixMsg}: neCommon wide3x3=1 is illegal for stride-{stride} 3x3 conv"
    let rep := ((1 : Nat) <<< logNE) >>> f
    let mbs := vMaxBlockSlots isDepthwise cin cout f effOcg
    if mbs > 32768 * rep then
      throw s!"{prefixMsg}: neCommon wide3x3=1 illegal because maxBlockSlots={mbs} > 32768*rep={32768 * rep}"

  if th == 0 || th > hout then
    throw s!"{prefixMsg}: tileHeight={th} outside 1..{hout}"

  -- 7. Verify L2 source & result geometry (L2-resident vs DMA-staged)
  let mut l2Accs : Array VerifiedL2Access := #[]

  let wSrc1Base ← requireDecodedWrite task.writes regSrc1Base "src1Base"
  let wSrc1Chs ← requireDecodedWrite task.writes regSrc1ChanStride "src1ChanStride"
  let wSrc1Row ← requireDecodedWrite task.writes regSrc1RowStride "src1RowStride"
  let srcBase := wSrc1Base.value.toNat

  if src1Cfg.isL2 then
    vCheckOmitted task.writes #[
      (regSrc1RowBytesA, "src1RowBytesA"), (regSrc1RowBytesB, "src1RowBytesB")
    ] "L2-resident src1" prefixMsg
    let (expChs, expRow, l2Bytes) := vL2ResidentGeometry cin hinReg winReg
    if wSrc1Chs.value.toNat != expChs then
      throw s!"{prefixMsg}: L2 src1ChanStride {wSrc1Chs.value} != expected {expChs}"
    if wSrc1Row.value.toNat != expRow then
      throw s!"{prefixMsg}: L2 src1RowStride {wSrc1Row.value} != expected {expRow}"
    l2Accs := l2Accs.push { lo := srcBase, hi := srcBase + l2Bytes, isWrite := false, role := .src1L2 }
  else
    let wSrc1RowA ← requireDecodedWrite task.writes regSrc1RowBytesA "src1RowBytesA"
    let wSrc1RowB ← requireDecodedWrite task.writes regSrc1RowBytesB "src1RowBytesB"
    let expectedChs := vChs kernelSize stride winReg pl stripW
    if wSrc1Chs.value.toNat != expectedChs then
      throw s!"{prefixMsg}: src1ChanStride {wSrc1Chs.value} != expected chs {expectedChs} (srcPx narrowing forbidden)"
    let minSrcRow := cin * expectedChs
    if wSrc1Row.value.toNat < minSrcRow || (wSrc1Row.value.toNat - minSrcRow) % 16 != 0 then
      throw s!"{prefixMsg}: src1RowStride {wSrc1Row.value} incompatible with Cin*chs={minSrcRow} + 16*srcSkew"
    if wSrc1RowA.value.toNat != minSrcRow || wSrc1RowB.value.toNat != minSrcRow then
      throw s!"{prefixMsg}: src1RowBytesA/B ({wSrc1RowA.value}, {wSrc1RowB.value}) != Cin*chs ({minSrcRow})"
    let inRows := vInRows kernelSize fullH stride pt hout th
    let srcBytes := inRows * wSrc1Row.value.toNat
    l2Accs := l2Accs.push { lo := srcBase, hi := srcBase + srcBytes, isWrite := true, role := .src1Stage }

  let wResBase ← requireDecodedWrite task.writes regResBase "resBase"
  let resBase := wResBase.value.toNat
  if resIsL2 then
    vCheckOmitted task.writes #[
      (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
    ] "L2-resident res" prefixMsg
    let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
    let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
    let (expChs, expRow, l2Bytes) := vL2ResidentGeometry cout hout wout
    if wResChs.value.toNat != expChs then
      throw s!"{prefixMsg}: L2 resChanStride {wResChs.value} != expected {expChs}"
    if wResRow.value.toNat != expRow then
      throw s!"{prefixMsg}: L2 resRowStride {wResRow.value} != expected {expRow}"
    l2Accs := l2Accs.push { lo := resBase, hi := resBase + l2Bytes, isWrite := true, role := .resL2 }
  else
    let expectedDirect := vIsDirect target fat shMax wout
    if direct != expectedDirect then
      throw s!"{prefixMsg}: resCfg direct={direct} disagrees with target/tiling direct={expectedDirect}"
    if direct then
      vCheckOmitted task.writes #[
        (regResChanStride, "resChanStride"), (regResRowStride, "resRowStride"),
        (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
      ] "direct mode" prefixMsg
      let resBytes := ((1 : Nat) <<< (f + effOcg + fat)) * 512
      l2Accs := l2Accs.push { lo := resBase, hi := resBase + resBytes, isWrite := true, role := .resStage }
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
      let resBytes := th * wResRow.value.toNat
      l2Accs := l2Accs.push { lo := resBase, hi := resBase + resBytes, isWrite := true, role := .resStage }

  vCheckTaskL2Accesses l2Accs prefixMsg

  -- 8. Verify NE block, neScale, sparse consistency across 0x4900 and 0x5500, and scalarBias rules
  let expectedSparse : UInt32 := if isSparse then 1 else 0
  let wNeCfg ← requireDecodedWrite task.writes regNeCfg "neCfg"
  let neCfgSparse ← decodeFieldChecked (neCfgLayout target) wNeCfg.value "sparse"
  if neCfgSparse != expectedSparse then
    throw s!"{prefixMsg}: sparse flag {neCfgSparse} disagrees with stride {stride} (expected {expectedSparse})"
  vVerifyKernelStageMulti task.writes taskId isLast true isSparse prefixMsg

  let wNeOp ← requireDecodedWrite task.writes regNeOp "neOp"
  let biasInKernelBit ← decodeFieldChecked neOpLayout wNeOp.value "biasInKernel"
  let reluBit ← decodeFieldChecked neOpLayout wNeOp.value "relu"
  let lutBit ← decodeFieldChecked neOpLayout wNeOp.value "lut"
  if reluBit == 1 && lutBit == 1 then
    throw s!"{prefixMsg}: neOp cannot enable both relu (bit 16) and lut (bit 17)"
  let biasInKernel := biasInKernelBit == 1
  let hasLut := lutBit == 1
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
  if !hasLut && neScaleVal != fp16OneBits then
    throw s!"{prefixMsg}: neScale {toHex neScaleVal.toNat} != expected fp16 1.0 ({toHex fp16OneBits.toNat})"

  -- 9. Verify kernel BaseLo/Hi, relocation, active & disabled chunk slots (S1 carry-over on taskId > 0)
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
      if target.profile.replicateKernelChunkSlots then
        for j in [:rep] do
          let s := g * rep + j
          if s >= 16 then
            throw s!"{prefixMsg}: expected chunk slot {s} exceeds 0..15"
          expectedSlotMap := expectedSlotMap.set! s (some blocks)
      else
        let s := g * rep
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
      | none =>
        if !isFirst then
          throw s!"{prefixMsg}: NE conv taskId > 0 must explicitly write chunk slot {s} ({toHex (regChunkSlot s)}) to prevent carry-over"
        pure false
      | some sw =>
        let en ← decodeFieldChecked chunkSlotLayout sw.value "enable"
        let on20 ← decodeFieldChecked chunkSlotLayout sw.value "on20"
        let slotTask ← decodeFieldChecked chunkSlotLayout sw.value "task"
        if on20 != 1 then
          throw s!"{prefixMsg}: chunk slot {s} missing on20=1 control bit"
        if slotTask.toNat != taskId then
          throw s!"{prefixMsg}: chunk slot {s} task={slotTask} != header taskId={taskId}"
        pure (en == 1)

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

      let lutOverhead := if hasLut then vLutBytes else 0
      if hasLut && cSz < vLutBytes then
        throw s!"{prefixMsg}: chunk slot {s} size {cSz}B < {vLutBytes}B required for fused LUT"
      if hasLut then
        if let some kBytes := kernelBytes? then
          vVerifyLutBytes kBytes (kernBaseOff + cOff) s!"{prefixMsg} chunk slot {s} fused LUT"

      let groupOuts := vGroupOuts blocks
      if !isSparse then
        let minBytes := alignUp (lutOverhead + 2 * groupOuts * (cinW * taps + (if biasInKernel then 1 else 0))) vDmaLineBytes
        if cSz < minBytes then
          throw s!"{prefixMsg}: chunk slot {s} size {cSz}B < minimum required {minBytes}B for stride-1 kernel"
      else
        let minBytes := alignUp (lutOverhead + 2 * (if biasInKernel then groupOuts else 0) + 3 * blocks.size) vDmaLineBytes
        if cSz < minBytes then
          throw s!"{prefixMsg}: chunk slot {s} size {cSz}B < minimum required {minBytes}B for stride-2 sparse kernel"
        if let some kBytes := kernelBytes? then
          vVerifySparseChunkBytes kBytes (kernBaseOff + cOff + lutOverhead) (kernBaseOff + cOff + cSz)
            blocks biasInKernel s!"{prefixMsg} chunk slot {s}"

      dramAccs := dramAccs.push {
        resource := .kernel
        lo := kernBaseOff + cOff
        hi := kernBaseOff + cOff + cSz
        isWrite := false
        port := .kernel
      }
      activeChunks := activeChunks + 1
    else
      if offW?.isSome then
        throw s!"{prefixMsg}: chunk slot {s} writes offset without enable bit 0 in {toHex (regChunkSlot s)}"
      if expBlocks?.isSome then
        throw s!"{prefixMsg}: missing required active chunk slot {s}"

  if activeChunks == 0 then
    throw s!"{prefixMsg}: no active kernel chunk slots enabled"

  -- 10. Verify DMA deps, formats, and bounds for DMA-backed src1 / res
  if let some (_, inTarget, srcBaseOff) := src1DmaTarget? then
    let (dk, dd) ← vDecodeSrcDep info task.writes regSrc1Dep regSrc1DmaCfg "src1Dep" "src1DmaCfg" taskId isFirst prefixMsg
    sync := { sync with src1DepKind := dk, src1DepDist := dd }
    let wSrc1Fmt ← requireDecodedWrite task.writes regSrc1Fmt "src1Fmt"
    if wSrc1Fmt.value != srcFmtWordP1 then
      throw s!"{prefixMsg}: src1Fmt {toHex wSrc1Fmt.value.toNat} != expected {toHex srcFmtWordP1.toNat}"
    let (resId, cap, isSurf, lbl) := match inTarget with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, true, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, false, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, false, "invalid")
    let (inExtent, _) ← vVerifyDmaStridesAndBounds task.writes cin fullH fullW srcBaseOff cap isSurf
      regSrc1DmaRowStride regSrc1DmaPlaneStride "src1DmaRowStride" "src1DmaPlaneStride" "input DMA" lbl prefixMsg
    dramAccs := dramAccs.push { resource := resId, lo := srcBaseOff, hi := srcBaseOff + inExtent, isWrite := false, port := .src1Dma }

  if let some (_, outTarget, resBaseOff) := resDmaTarget? then
    match ← resolveBarTarget info barSlotConfigZero with
    | .configZero => pure ()
    | _ => throw s!"{prefixMsg}: BAR slot {barSlotConfigZero} must have VM base 0"
    let wResDep ← requireDecodedWrite task.writes regResDep "resDep"
    if wResDep.barSlot != some barSlotConfigZero then
      throw s!"{prefixMsg}: resDep (0x5100) must use BAR slot {barSlotConfigZero}"
    let resDepC ← decodeFieldChecked resDepLayout wResDep.value "code"
    let resDepTask ← decodeFieldChecked resDepLayout wResDep.value "task"
    let resDepDirect ← decodeFieldChecked resDepLayout wResDep.value "direct"
    let expCode := match outTarget with
      | .scratch _ => resDepCode target ||| resDepScratchFlags
      | _ => resDepCode target
    if resDepC != expCode then
      throw s!"{prefixMsg}: resDep code {toHex resDepC.toNat} != expected {toHex expCode.toNat}"
    if resDepTask.toNat != taskId then
      throw s!"{prefixMsg}: resDep task={resDepTask} != header taskId={taskId}"
    if (resDepDirect == 1) != direct then
      throw s!"{prefixMsg}: resDep direct bit {resDepDirect} != resCfg direct={direct}"
    let wResFmt ← requireDecodedWrite task.writes regResFmt "resFmt"
    if wResFmt.value != resFmtWordP1 then
      throw s!"{prefixMsg}: resFmt {toHex wResFmt.value.toNat} != expected {toHex resFmtWordP1.toNat}"
    let (resId, cap, isSurf, lbl) := match outTarget with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, true, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, false, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, false, "invalid")
    let (outExtent, outCov) ← vVerifyDmaStridesAndBounds task.writes cout hout wout resBaseOff cap isSurf
      regResDmaRowStride regResDmaPlaneStride "resDmaRowStride" "resDmaPlaneStride" "output DMA" lbl prefixMsg
    dramAccs := dramAccs.push { resource := resId, lo := resBaseOff, hi := resBaseOff + outExtent, isWrite := true, port := .resDma, coverage := outCov }

  -- 11. Output stage block (only on final task)
  vVerifyOutputStageMulti task.writes taskId isLast prefixMsg

  -- 12. Strict mode-conditional register allowlist
  if let some wN := findDecodedWrite task.writes regBatchN then
    if wN.value != 1 then
      throw s!"{prefixMsg}: batch N (0x0024) must be 1, got {wN.value}"
  let mut allowedConvRegs : Array Nat := #[
    regWin, regHin, regCin, regWout, regHout, regCout, regBatchN,
    regConvCfg, regTH, regNeCommon, regNeTile, regTargetCfg,
    regSrc1Cfg, regSrc1Base, regSrc1ChanStride, regSrc1RowStride,
    regResCfg, regResBase,
    regNeCfg, regNeOp, regNeScale,
    regKernelStage, regKernelBaseLo, regKernelBaseHi
  ]
  if isFirst then
    allowedConvRegs := allowedConvRegs.push regSrc2Cfg
  if !src1Cfg.isL2 then
    allowedConvRegs := allowedConvRegs ++ #[
      regSrc1RowBytesA, regSrc1RowBytesB,
      regSrc1Dep, regSrc1DmaCfg, regSrc1BaseLo, regSrc1BaseHi,
      regSrc1DmaRowStride, regSrc1DmaPlaneStride, regSrc1Fmt
    ]
  if resIsL2 then
    allowedConvRegs := allowedConvRegs ++ #[regResChanStride, regResRowStride]
  else
    allowedConvRegs := allowedConvRegs ++ #[
      regResDep, regResBaseLo, regResBaseHi,
      regResDmaRowStride, regResDmaPlaneStride, regResFmt
    ]
    if !direct then
      allowedConvRegs := allowedConvRegs ++ #[
        regResChanStride, regResRowStride, regResRowBytesA, regResRowBytesB
      ]
  if isLast then
    allowedConvRegs := allowedConvRegs ++ #[regOut0, regOut1, regOut2, regOut3, regOut4, regOut5]
  if isDepthwise then
    allowedConvRegs := allowedConvRegs.push regDwGroups
  if scalarBiasMode then
    allowedConvRegs := allowedConvRegs.push regScalarBias
  for s in [:16] do
    if h : s < expectedSlotMap.size then
      if expectedSlotMap[s].isSome then
        allowedConvRegs := allowedConvRegs ++ #[regChunkSlot s, regChunkOffset s, regChunkSize s]
      else if !isFirst || (findDecodedWrite task.writes (regChunkSlot s)).isSome then
        allowedConvRegs := allowedConvRegs.push (regChunkSlot s)
  vCheckAllowedRegs task.writes allowedConvRegs prefixMsg

  pure {
    taskIndex := task.taskIndex
    kind := .neConv
    engine := .ne
    isNeCompute := true
    liveL2Mask := h7
    l2Accesses := l2Accs
    dramAccesses := dramAccs
    sync := sync
  }

/-- Single-task compatibility wrapper for `verifyConvTaskMulti`. -/
def verifyConvTask (info : HwxInfo) (task : DecodedTask) (kernelBytes? : Option ByteArray := none) :
    Except String Unit := do
  let _ ← verifyConvTaskMulti 1 false false info task kernelBytes?
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

-- ============================================================================
-- Multi-Task Planar Engine (PE) static safety verifier (S1-S4, [DL])
-- ============================================================================

/-- Independent static safety verifier for decoded Planar Engine (PE) tasks (1-source and 2-source)
    in single-task or multi-task streams. Enforces `[DL]`, anti-hang channel strides, broadcast flags,
    L2-resident & DRAM-scratch bounds, and strict allowlists. -/
def verifyPeTaskMulti
    (taskCount : Nat) (hasPriorNe hasPriorPe : Bool)
    (info : HwxInfo) (task : DecodedTask) : Except String VerifiedTaskFootprint := do
  let target := info.target
  let prefixMsg := s!"task {task.taskIndex} (PE)"
  let isFirst := task.taskIndex == 0
  let isLast := task.taskIndex + 1 == taskCount

  -- 1. Verify 9-word header with ene == 0
  let (taskId, _, h7) ← vVerifyTaskHeader9 target task taskCount true false prefixMsg

  -- 2. Reject duplicate register writes and unmodelled DMA registers
  vCheckNoDuplicateWrites task.writes prefixMsg
  vCheckNoUnmodelledDmaRegs task.writes prefixMsg

  -- 3. Common block & PE control registers
  vCheckOmitted task.writes #[
    (regConvCfg, "convCfg"), (regDwGroups, "dwGroups"), (regNeTile, "neTile")
  ] "PE task" prefixMsg

  let wWin ← requireDecodedWrite task.writes regWin "Win"
  let wHin ← requireDecodedWrite task.writes regHin "Hin"
  let wCin ← requireDecodedWrite task.writes regCin "Cin"
  let wWout ← requireDecodedWrite task.writes regWout "Wout"
  let wHout ← requireDecodedWrite task.writes regHout "Hout"
  let wCout ← requireDecodedWrite task.writes regCout "Cout"
  let wTH ← requireDecodedWrite task.writes regTH "TH"
  let wPeCommon ← requireDecodedWrite task.writes regPeCommon "peCommon"
  let wPeTile ← requireDecodedWrite task.writes regPeTile "peTile"
  let wTargetCfg ← requireDecodedWrite task.writes regTargetCfg "targetCfg"

  if let some wN := findDecodedWrite task.writes regBatchN then
    if wN.value != 1 then
      throw s!"{prefixMsg}: batch N (0x0024) must be 1, got {wN.value}"

  let outW := (← decodeFieldChecked (dim16Layout "Win") wWin.value "val").toNat
  let outH := (← decodeFieldChecked (dim16Layout "Hin") wHin.value "val").toNat
  let outC := wCin.value.toNat
  let wout := (← decodeFieldChecked (dim16Layout "Wout") wWout.value "val").toNat
  let hout := (← decodeFieldChecked (dim16Layout "Hout") wHout.value "val").toNat
  let cout := wCout.value.toNat
  let th := (← decodeFieldChecked (dim16Layout "TH") wTH.value "val").toNat

  if outW == 0 || outH == 0 || outC == 0 || outC > 65535 then
    throw s!"{prefixMsg}: invalid PE dims (C={outC}, H={outH}, W={outW})"
  if wout != outW || hout != outH || cout != outC then
    throw s!"{prefixMsg}: Win/Hin/Cin ({outC}x{outH}x{outW}) != Wout/Hout/Cout ({cout}x{hout}x{wout})"
  if wTargetCfg.value != 0 then
    throw s!"{prefixMsg}: PE targetCfg (0x004c) must be 0, got {toHex wTargetCfg.value.toNat}"

  let peMode ← decodeFieldChecked peCommonLayout wPeCommon.value "mode"
  if peMode != peCommonOneSrcValue && peMode != peCommonTwoSrcValue then
    throw s!"{prefixMsg}: peCommon mode {toHex peMode.toNat} must be 0x20 (1-src) or 0x30 (2-src)"
  let twoSource := peMode == peCommonTwoSrcValue

  let k := (← decodeFieldChecked peTileLayout wPeTile.value "k").toNat
  let m := (← decodeFieldChecked peTileLayout wPeTile.value "m").toNat
  let maxK := if twoSource then 8 else 9
  if k < 4 || k > maxK then
    throw s!"{prefixMsg}: peTile k={k} out of range 4..{maxK}"
  if th == 0 || th > outH then
    throw s!"{prefixMsg}: tileHeight={th} out of range 1..{outH}"
  let expM := min (maxK - k) (vCeilLog2 th)
  if m != expM then
    throw s!"{prefixMsg}: peTile m={m} != expected min({maxK - k}, ceilLog2({th}))={expM}"
  let stripW := (1 : Nat) <<< k

  -- 4. Decode port configs (`src1Cfg`, `src2Cfg`, `resCfg`)
  let wSrc1Cfg ← requireDecodedWrite task.writes regSrc1Cfg "src1Cfg"
  let src1Cfg ← vDecodeSrcPortCfg "src1Cfg" wSrc1Cfg.value isFirst hasPriorNe hasPriorPe prefixMsg
  let src2Cfg? : Option DecodedSrcPortCfg ←
    if twoSource then
      let wSrc2Cfg ← requireDecodedWrite task.writes regSrc2Cfg "src2Cfg"
      some <$> vDecodeSrcPortCfg "src2Cfg" wSrc2Cfg.value isFirst hasPriorNe hasPriorPe prefixMsg
    else
      if isFirst then
        let wSrc2Cfg ← requireDecodedWrite task.writes regSrc2Cfg "src2Cfg"
        if wSrc2Cfg.value != src2CfgValue then
          throw s!"{prefixMsg}: src2Cfg {toHex wSrc2Cfg.value.toNat} != expected {toHex src2CfgValue.toNat}"
      else
        vCheckOmitted task.writes #[(regSrc2Cfg, "src2Cfg")] "1-source PE taskId > 0" prefixMsg
      pure none

  let wResCfg ← requireDecodedWrite task.writes regResCfg "resCfg"
  let (resIsL2, direct) ← vDecodeResPortCfg wResCfg.value isFirst isLast prefixMsg

  -- 5. Decode broadcast flags & derive operand shapes `(s1C, s1H, s1W)` and `(s2C, s2H, s2W)`
  let (s1C, s1H, s1W, s2Dims?) ←
    if twoSource then
      let bcastWord : UInt32 := match findDecodedWrite task.writes regPeBcast with
        | some wb => wb.value
        | none => 0
      let b1W ← decodeFieldChecked peBcastLayout bcastWord "src1W"
      let b1H ← decodeFieldChecked peBcastLayout bcastWord "src1H"
      let b1C ← decodeFieldChecked peBcastLayout bcastWord "src1C"
      let b2W ← decodeFieldChecked peBcastLayout bcastWord "src2W"
      let b2H ← decodeFieldChecked peBcastLayout bcastWord "src2H"
      let b2C ← decodeFieldChecked peBcastLayout bcastWord "src2C"
      if (b1W == 1 && outW == 1) || (b1H == 1 && outH == 1) || (b1C == 1 && outC == 1) ||
         (b2W == 1 && outW == 1) || (b2H == 1 && outH == 1) || (b2C == 1 && outC == 1) then
        throw s!"{prefixMsg}: peBcast {toHex bcastWord.toNat} sets broadcast flag on axis of size 1"
      if (b1W == 1 && b2W == 1) || (b1H == 1 && b2H == 1) || (b1C == 1 && b2C == 1) then
        throw s!"{prefixMsg}: peBcast {toHex bcastWord.toNat} broadcasts both sources on the same axis"
      let c1 := if b1C == 1 then 1 else outC
      let h1 := if b1H == 1 then 1 else outH
      let w1 := if b1W == 1 then 1 else outW
      let c2 := if b2C == 1 then 1 else outC
      let h2 := if b2H == 1 then 1 else outH
      let w2 := if b2W == 1 then 1 else outW
      if src1Cfg.isL2 && (b1W != 0 || b1H != 0 || b1C != 0) then
        if th != outH then
          throw s!"{prefixMsg}: L2-resident PE2 broadcast src1 requires th == outH (got th={th}, outH={outH})"
        if src1Cfg.handoffNe || src1Cfg.handoffPe then
          throw s!"{prefixMsg}: L2-resident PE2 broadcast src1 must use whole-task wait, not L2 handoff"
      if src2Cfg?.get!.isL2 && (b2W != 0 || b2H != 0 || b2C != 0) then
        if th != outH then
          throw s!"{prefixMsg}: L2-resident PE2 broadcast src2 requires th == outH (got th={th}, outH={outH})"
        if src2Cfg?.get!.handoffNe || src2Cfg?.get!.handoffPe then
          throw s!"{prefixMsg}: L2-resident PE2 broadcast src2 must use whole-task wait, not L2 handoff"
      pure (c1, h1, w1, some (c2, h2, w2))
    else
      if let some wb := findDecodedWrite task.writes regPeBcast then
        if wb.value != 0 then
          throw s!"{prefixMsg}: 1-source PE task must not set peBcast ({toHex wb.value.toNat})"
      vCheckOmitted task.writes #[
        (regSrc2Base, "src2Base"), (regSrc2ChanStride, "src2ChanStride"),
        (regSrc2RowStride, "src2RowStride"), (regSrc2RowBytesA, "src2RowBytesA"),
        (regSrc2RowBytesB, "src2RowBytesB"), (regSrc2Dep, "src2Dep"),
        (regSrc2DmaCfg, "src2DmaCfg"), (regSrc2BaseLo, "src2BaseLo"),
        (regSrc2BaseHi, "src2BaseHi"), (regSrc2DmaRowStride, "src2DmaRowStride"),
        (regSrc2DmaPlaneStride, "src2DmaPlaneStride"), (regSrc2Fmt, "src2Fmt")
      ] "1-source PE task" prefixMsg
      pure (outC, outH, outW, none)

  -- 6. Resolve DMA endpoints (checking surface shapes & broadcast consistency when surface-backed)
  let mut dramAccs : Array VerifiedDramAccess := #[]
  let mut sync : TaskSyncInfo := {
    src1WaitNe5 := src1Cfg.waitNe5
    src1WaitPe22 := src1Cfg.waitPe22
    src1HandoffNe := src1Cfg.handoffNe
    src1HandoffPe := src1Cfg.handoffPe
    src2WaitNe5 := match src2Cfg? with | some c => c.waitNe5 | none => false
    src2WaitPe22 := match src2Cfg? with | some c => c.waitPe22 | none => false
    src2HandoffNe := match src2Cfg? with | some c => c.handoffNe | none => false
    src2HandoffPe := match src2Cfg? with | some c => c.handoffPe | none => false
    hasSrc1Dma := !src1Cfg.isL2
    hasSrc2Dma := match src2Cfg? with | some c => !c.isL2 | none => false
  }

  let src1DmaTarget? ←
    if src1Cfg.isL2 then
      vCheckOmitted task.writes #[
        (regSrc1Dep, "src1Dep"), (regSrc1DmaCfg, "src1DmaCfg"),
        (regSrc1BaseLo, "src1BaseLo"), (regSrc1BaseHi, "src1BaseHi"),
        (regSrc1DmaRowStride, "src1DmaRowStride"), (regSrc1DmaPlaneStride, "src1DmaPlaneStride"),
        (regSrc1Fmt, "src1Fmt")
      ] "L2-resident PE src1" prefixMsg
      pure none
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regSrc1BaseLo regSrc1BaseHi "src1BaseLo" "src1BaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .input then
          throw s!"{prefixMsg}: src1Base BAR slot {slot} bound to non-input surface '{s.shortName}'"
        if s.n != 1 then
          throw s!"{prefixMsg}: src1 surface must be single-batch (n=1), got {s.n}"
        let s1HOk := if s1H != outH then s.height.toNat == s1H else s1H <= s.height.toNat
        if s.channels.toNat != s1C || !s1HOk || s.width.toNat != s1W then
          throw s!"{prefixMsg}: peBcast mismatches src1 surface ({s.channels}x{s.height}x{s.width}) vs expected ({s1C}x{s1H}x{s1W})"
        pure (some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: scratch src1Base must use BAR slot {barSlotScratch}, got {slot}"
        if isFirst then
          throw s!"{prefixMsg}: first task cannot read uninitialized DRAM scratch"
        pure (some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: src1Base BAR slot {slot} does not reference an input surface or scratch"

  let src2DmaTarget? ←
    match s2Dims?, src2Cfg? with
    | some (s2C, s2H, s2W), some s2Cfg =>
      if s2Cfg.isL2 then
        vCheckOmitted task.writes #[
          (regSrc2Dep, "src2Dep"), (regSrc2DmaCfg, "src2DmaCfg"),
          (regSrc2BaseLo, "src2BaseLo"), (regSrc2BaseHi, "src2BaseHi"),
          (regSrc2DmaRowStride, "src2DmaRowStride"), (regSrc2DmaPlaneStride, "src2DmaPlaneStride"),
          (regSrc2Fmt, "src2Fmt")
        ] "L2-resident PE src2" prefixMsg
        pure none
      else
        let (slot, tgt, off, _) ← resolveBasePair info task.writes regSrc2BaseLo regSrc2BaseHi "src2BaseLo" "src2BaseHi" prefixMsg
        match tgt with
        | .surface s =>
          if s.direction != .input then
            throw s!"{prefixMsg}: src2Base BAR slot {slot} bound to non-input surface '{s.shortName}'"
          if s.n != 1 then
            throw s!"{prefixMsg}: src2 surface must be single-batch (n=1), got {s.n}"
          let s2HOk := if s2H != outH then s.height.toNat == s2H else s2H <= s.height.toNat
          if s.channels.toNat != s2C || !s2HOk || s.width.toNat != s2W then
            throw s!"{prefixMsg}: peBcast mismatches src2 surface ({s.channels}x{s.height}x{s.width}) vs expected ({s2C}x{s2H}x{s2W})"
          pure (some (slot, tgt, off))
        | .scratch _ =>
          if slot != barSlotScratch then
            throw s!"{prefixMsg}: scratch src2Base must use BAR slot {barSlotScratch}, got {slot}"
          if isFirst then
            throw s!"{prefixMsg}: first task cannot read uninitialized DRAM scratch"
          pure (some (slot, tgt, off))
        | _ => throw s!"{prefixMsg}: src2Base BAR slot {slot} does not reference an input surface or scratch"
    | _, _ => pure none

  let resDmaTarget? ←
    if resIsL2 then
      vCheckOmitted task.writes #[
        (regResDep, "resDep"), (regResBaseLo, "resBaseLo"), (regResBaseHi, "resBaseHi"),
        (regResDmaRowStride, "resDmaRowStride"), (regResDmaPlaneStride, "resDmaPlaneStride"),
        (regResFmt, "resFmt")
      ] "L2-resident PE res" prefixMsg
      pure none
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regResBaseLo regResBaseHi "resBaseLo" "resBaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .output then
          throw s!"{prefixMsg}: resBase BAR slot {slot} bound to non-output surface '{s.shortName}'"
        if s.n != 1 then
          throw s!"{prefixMsg}: output surface must be single-batch (n=1), got {s.n}"
        if s.channels.toNat != outC || outH > s.height.toNat || s.width.toNat != outW then
          throw s!"{prefixMsg}: output surface ({s.channels}x{s.height}x{s.width}) incompatible with PE dims ({outC}x{outH}x{outW})"
        if let some (s1Slot, .surface _, _) := src1DmaTarget? then
          if slot == s1Slot then throw s!"{prefixMsg}: output BAR slot {slot} aliases src1 BAR slot"
        if let some (s2Slot, .surface _, _) := src2DmaTarget? then
          if slot == s2Slot then throw s!"{prefixMsg}: output BAR slot {slot} aliases src2 BAR slot"
        pure (some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: scratch resBase must use BAR slot {barSlotScratch}, got {slot}"
        if isLast then
          throw s!"{prefixMsg}: final task cannot write to DRAM scratch"
        pure (some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: resBase BAR slot {slot} does not reference an output surface or scratch"

  -- 7. Verify PE arithmetic block (0x4500..0x4510)
  let peOpW? := findDecodedWrite task.writes regPeOp
  let peBiasW? := findDecodedWrite task.writes regPeBias
  let peScaleW? := findDecodedWrite task.writes regPeScale
  let peInScaleW? := findDecodedWrite task.writes regPeInScale
  if twoSource then
    let some wOp := peOpW?
      | throw s!"{prefixMsg}: 2-source PE task requires peOp (0x4500)"
    let v := wOp.value
    if v != peOpAddValue && v != peOpMulValue && v != peOpMaxValue && v != peOpMinValue && v != peOpSubValue then
      throw s!"{prefixMsg}: illegal 2-source peOp opcode {toHex v.toNat}"
    if peBiasW?.isSome || peScaleW?.isSome || peInScaleW?.isSome then
      throw s!"{prefixMsg}: 2-source PE task must not write chain constant registers (0x4504/0x4508/0x4510)"
  else
    match peOpW? with
    | some wOp =>
      if wOp.value != peOpAbsValue && wOp.value != peOpSquareValue then
        throw s!"{prefixMsg}: illegal 1-source peOp opcode {toHex wOp.value.toNat}"
      if peBiasW?.isSome || peScaleW?.isSome || peInScaleW?.isSome then
        throw s!"{prefixMsg}: 1-source unary peOp must not write chain constant registers (0x4504/0x4508/0x4510)"
    | none =>
      if peBiasW?.isNone && peScaleW?.isNone && peInScaleW?.isNone then
        throw s!"{prefixMsg}: 1-source PE chain without peOp (0x4500) must write at least one of 0x4504/0x4508/0x4510"

  -- 8. Verify L2 SRAM staging, anti-hang invariant, whole1/whole2, [DL] direct rule, and disjointness
  let mut l2Accs : Array VerifiedL2Access := #[]

  let wSrc1Base ← requireDecodedWrite task.writes regSrc1Base "src1Base"
  let wSrc1Chs ← requireDecodedWrite task.writes regSrc1ChanStride "src1ChanStride"
  let wSrc1Row ← requireDecodedWrite task.writes regSrc1RowStride "src1RowStride"
  let src1L2Base := wSrc1Base.value.toNat
  let src1Chs := wSrc1Chs.value.toNat
  let src1Row := wSrc1Row.value.toNat
  if src1Cfg.isL2 then
    vCheckOmitted task.writes #[
      (regSrc1RowBytesA, "src1RowBytesA"), (regSrc1RowBytesB, "src1RowBytesB")
    ] "L2-resident PE src1" prefixMsg
    let (expChs, expRow, l2Bytes) := vL2ResidentGeometry s1C s1H s1W
    if src1Chs != expChs then
      throw s!"{prefixMsg}: L2 src1ChanStride ({src1Chs}) != expected {expChs}"
    if src1Row != expRow then
      throw s!"{prefixMsg}: L2 src1RowStride ({src1Row}) != expected {expRow}"
    l2Accs := l2Accs.push { lo := src1L2Base, hi := src1L2Base + l2Bytes, isWrite := false, role := .src1L2 }
  else
    let wSrc1RowA ← requireDecodedWrite task.writes regSrc1RowBytesA "src1RowBytesA"
    let wSrc1RowB ← requireDecodedWrite task.writes regSrc1RowBytesB "src1RowBytesB"
    let effStripW := max stripW 32
    let minAntiHang1 := 2 * min (alignUp s1W 8) effStripW
    if src1Chs < minAntiHang1 then
      throw s!"{prefixMsg}: anti-hang / 64B DMA line floor violation: src1ChanStride ({src1Chs}) < 2 * min(alignUp(W1={s1W}, 8), max(T={stripW}, 32)) = {minAntiHang1}"
    let whole1 := twoSource && s1C == 1 && s1H == 1 && s1W > 1 && outH > 1 && th < outH
    let expRow1 := 2 * (if whole1 then alignUp s1W 8 else min (alignUp s1W 8) effStripW)
    if src1Chs != expRow1 then
      throw s!"{prefixMsg}: src1ChanStride ({src1Chs}) != expected {expRow1} (whole1={whole1})"
    let minU1 := s1C * src1Chs
    if src1Row < minU1 || (src1Row - minU1) % 16 != 0 then
      throw s!"{prefixMsg}: src1RowStride ({src1Row}) incompatible with C1*chs={minU1}"
    if wSrc1RowA.value.toNat != minU1 || wSrc1RowB.value.toNat != minU1 then
      throw s!"{prefixMsg}: src1RowBytesA/B ({wSrc1RowA.value}, {wSrc1RowB.value}) != C1*chs ({minU1})"
    let size1 := (if s1H > 1 then th else 1) * src1Row
    l2Accs := l2Accs.push { lo := src1L2Base, hi := src1L2Base + size1, isWrite := true, role := .src1Stage }

  if let (some (s2C, s2H, s2W), some s2Cfg) := (s2Dims?, src2Cfg?) then
    let wSrc2Base ← requireDecodedWrite task.writes regSrc2Base "src2Base"
    let wSrc2Chs ← requireDecodedWrite task.writes regSrc2ChanStride "src2ChanStride"
    let wSrc2Row ← requireDecodedWrite task.writes regSrc2RowStride "src2RowStride"
    let src2L2Base := wSrc2Base.value.toNat
    let src2Chs := wSrc2Chs.value.toNat
    let src2Row := wSrc2Row.value.toNat
    if s2Cfg.isL2 then
      vCheckOmitted task.writes #[
        (regSrc2RowBytesA, "src2RowBytesA"), (regSrc2RowBytesB, "src2RowBytesB")
      ] "L2-resident PE src2" prefixMsg
      let (expChs, expRow, l2Bytes) := vL2ResidentGeometry s2C s2H s2W
      if src2Chs != expChs then
        throw s!"{prefixMsg}: L2 src2ChanStride ({src2Chs}) != expected {expChs}"
      if src2Row != expRow then
        throw s!"{prefixMsg}: L2 src2RowStride ({src2Row}) != expected {expRow}"
      l2Accs := l2Accs.push { lo := src2L2Base, hi := src2L2Base + l2Bytes, isWrite := false, role := .src2L2 }
    else
      let wSrc2RowA ← requireDecodedWrite task.writes regSrc2RowBytesA "src2RowBytesA"
      let wSrc2RowB ← requireDecodedWrite task.writes regSrc2RowBytesB "src2RowBytesB"
      let effStripW := max stripW 32
      let minAntiHang2 := 2 * min (alignUp s2W 8) effStripW
      if src2Chs < minAntiHang2 then
        throw s!"{prefixMsg}: anti-hang / 64B DMA line floor violation: src2ChanStride ({src2Chs}) < 2 * min(alignUp(W2={s2W}, 8), max(T={stripW}, 32)) = {minAntiHang2}"
      let whole2 := s2C == 1 && s2H == 1 && s2W > 1 && outH > 1 && th < outH
      let expRow2 := 2 * (if whole2 then alignUp s2W 8 else min (alignUp s2W 8) effStripW)
      if src2Chs != expRow2 then
        throw s!"{prefixMsg}: src2ChanStride ({src2Chs}) != expected {expRow2} (whole2={whole2})"
      let minU2 := s2C * src2Chs
      if src2Row < minU2 || (src2Row - minU2) % 16 != 0 then
        throw s!"{prefixMsg}: src2RowStride ({src2Row}) incompatible with C2*chs={minU2}"
      if wSrc2RowA.value.toNat != minU2 || wSrc2RowB.value.toNat != minU2 then
        throw s!"{prefixMsg}: src2RowBytesA/B ({wSrc2RowA.value}, {wSrc2RowB.value}) != C2*chs ({minU2})"
      let size2 := (if s2H > 1 then th else 1) * src2Row
      l2Accs := l2Accs.push { lo := src2L2Base, hi := src2L2Base + size2, isWrite := true, role := .src2Stage }

  let wResBase ← requireDecodedWrite task.writes regResBase "resBase"
  let resL2Base := wResBase.value.toNat
  if resIsL2 then
    vCheckOmitted task.writes #[
      (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
    ] "L2-resident PE res" prefixMsg
    let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
    let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
    let (expChs, expRow, l2Bytes) := vL2ResidentGeometry outC outH outW
    if wResChs.value.toNat != expChs then
      throw s!"{prefixMsg}: L2 resChanStride {wResChs.value} != expected {expChs}"
    if wResRow.value.toNat != expRow then
      throw s!"{prefixMsg}: L2 resRowStride {wResRow.value} != expected {expRow}"
    l2Accs := l2Accs.push { lo := resL2Base, hi := resL2Base + l2Bytes, isWrite := true, role := .resL2 }
  else
    -- Enforce [DL] direct PE DRAM write line rule
    if direct && outW > stripW && stripW % 32 != 0 then
      throw s!"{prefixMsg}: [DL] direct PE DRAM write illegal for W={outW} > 2^{k}={stripW} (strip width not a multiple of 32 pixels)"
    let expDirect := vPeIsDirect target outW k
    if direct != expDirect then
      throw s!"{prefixMsg}: resCfg direct={direct} disagrees with {target.name} PE direct={expDirect} (W={outW}, k={k})"
    let expDrow := 2 * min (alignUp outW 8) (max stripW 32)
    if direct then
      vCheckOmitted task.writes #[
        (regResChanStride, "resChanStride"), (regResRowStride, "resRowStride"),
        (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
      ] "direct mode" prefixMsg
      let resDirectBytes := vPeDirectResBytes outC expDrow th
      l2Accs := l2Accs.push { lo := resL2Base, hi := resL2Base + resDirectBytes, isWrite := true, role := .resStage }
    else
      let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
      let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
      let wResRowA ← requireDecodedWrite task.writes regResRowBytesA "resRowBytesA"
      let wResRowB ← requireDecodedWrite task.writes regResRowBytesB "resRowBytesB"
      if wResChs.value.toNat < expDrow then
        throw s!"{prefixMsg}: 64B DMA line floor violation: staged resChanStride ({wResChs.value}) < {expDrow}"
      if wResChs.value.toNat != expDrow then
        throw s!"{prefixMsg}: staged resChanStride {wResChs.value} != expected {expDrow}"
      let minDu := outC * expDrow
      let resRow := wResRow.value.toNat
      if resRow < minDu || (resRow - minDu) % 16 != 0 then
        throw s!"{prefixMsg}: staged resRowStride {resRow} incompatible with C*drow={minDu}"
      if wResRowA.value.toNat != minDu || wResRowB.value.toNat != minDu then
        throw s!"{prefixMsg}: staged resRowBytesA/B ({wResRowA.value}, {wResRowB.value}) != {minDu}"
      l2Accs := l2Accs.push { lo := resL2Base, hi := resL2Base + th * resRow, isWrite := true, role := .resStage }

  vCheckTaskL2Accesses l2Accs prefixMsg

  -- 9. Verify DMA deps, configs, formats, and bounds
  if let some (_, tgt, off) := src1DmaTarget? then
    let (dk, dd) ← vDecodeSrcDep info task.writes regSrc1Dep regSrc1DmaCfg "src1Dep" "src1DmaCfg" taskId isFirst prefixMsg
    sync := { sync with src1DepKind := dk, src1DepDist := dd }
    let wFmt ← requireDecodedWrite task.writes regSrc1Fmt "src1Fmt"
    if wFmt.value != srcFmtWordP1 then
      throw s!"{prefixMsg}: src1Fmt {toHex wFmt.value.toNat} != {toHex srcFmtWordP1.toNat}"
    let (resId, cap, isSurf, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, true, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, false, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, false, "invalid")
    let (ext, _) ← vVerifyDmaStridesAndBounds task.writes s1C s1H s1W off cap isSurf
      regSrc1DmaRowStride regSrc1DmaPlaneStride "src1DmaRowStride" "src1DmaPlaneStride" "input DMA" lbl prefixMsg
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + ext, isWrite := false, port := .src1Dma }

  if let (some (s2C, s2H, s2W), some (_, tgt, off)) := (s2Dims?, src2DmaTarget?) then
    let (dk, dd) ← vDecodeSrcDep info task.writes regSrc2Dep regSrc2DmaCfg "src2Dep" "src2DmaCfg" taskId isFirst prefixMsg
    sync := { sync with src2DepKind := dk, src2DepDist := dd }
    let wFmt ← requireDecodedWrite task.writes regSrc2Fmt "src2Fmt"
    if wFmt.value != srcFmtWordP1 then
      throw s!"{prefixMsg}: src2Fmt {toHex wFmt.value.toNat} != {toHex srcFmtWordP1.toNat}"
    let (resId, cap, isSurf, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, true, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, false, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, false, "invalid")
    let (ext, _) ← vVerifyDmaStridesAndBounds task.writes s2C s2H s2W off cap isSurf
      regSrc2DmaRowStride regSrc2DmaPlaneStride "src2DmaRowStride" "src2DmaPlaneStride" "input DMA" lbl prefixMsg
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + ext, isWrite := false, port := .src2Dma }

  if let some (_, tgt, off) := resDmaTarget? then
    match ← resolveBarTarget info barSlotConfigZero with
    | .configZero => pure ()
    | _ => throw s!"{prefixMsg}: BAR slot {barSlotConfigZero} must have VM base 0"
    let wResDep ← requireDecodedWrite task.writes regResDep "resDep"
    if wResDep.barSlot != some barSlotConfigZero then
      throw s!"{prefixMsg}: resDep (0x5100) must use BAR slot {barSlotConfigZero}"
    let expCode := match tgt with
      | .scratch _ => resDepCode target ||| resDepScratchFlags
      | _ => resDepCode target
    if (← decodeFieldChecked resDepLayout wResDep.value "code") != expCode then
      throw s!"{prefixMsg}: resDep code mismatch"
    if (← decodeFieldChecked resDepLayout wResDep.value "task").toNat != taskId then
      throw s!"{prefixMsg}: resDep task != header taskId={taskId}"
    if ((← decodeFieldChecked resDepLayout wResDep.value "direct") == 1) != direct then
      throw s!"{prefixMsg}: resDep direct bit != resCfg direct={direct}"
    let wResFmt ← requireDecodedWrite task.writes regResFmt "resFmt"
    if wResFmt.value != resFmtWordP1 then
      throw s!"{prefixMsg}: resFmt {toHex wResFmt.value.toNat} != {toHex resFmtWordP1.toNat}"
    let (resId, cap, isSurf, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, true, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, false, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, false, "invalid")
    let (ext, outCov) ← vVerifyDmaStridesAndBounds task.writes outC outH outW off cap isSurf
      regResDmaRowStride regResDmaPlaneStride "resDmaRowStride" "resDmaPlaneStride" "output DMA" lbl prefixMsg
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + ext, isWrite := true, port := .resDma, coverage := outCov }

  -- 10. Kernel stage, output stage, and strict mode-conditional register allowlist
  vVerifyKernelStageMulti task.writes taskId isLast false false prefixMsg
  vVerifyOutputStageMulti task.writes taskId isLast prefixMsg

  let mut allowedPeRegs : Array Nat := #[
    regWin, regHin, regCin, regWout, regHout, regCout, regBatchN,
    regTH, regPeCommon, regPeTile, regTargetCfg,
    regSrc1Cfg, regSrc1Base, regSrc1ChanStride, regSrc1RowStride,
    regResCfg, regResBase
  ]
  if !src1Cfg.isL2 then
    allowedPeRegs := allowedPeRegs ++ #[
      regSrc1RowBytesA, regSrc1RowBytesB,
      regSrc1Dep, regSrc1DmaCfg, regSrc1BaseLo, regSrc1BaseHi,
      regSrc1DmaRowStride, regSrc1DmaPlaneStride, regSrc1Fmt
    ]
  if resIsL2 then
    allowedPeRegs := allowedPeRegs ++ #[regResChanStride, regResRowStride]
  else
    allowedPeRegs := allowedPeRegs ++ #[
      regResDep, regResBaseLo, regResBaseHi,
      regResDmaRowStride, regResDmaPlaneStride, regResFmt
    ]
    if !direct then
      allowedPeRegs := allowedPeRegs ++ #[
        regResChanStride, regResRowStride, regResRowBytesA, regResRowBytesB
      ]
  if isLast then
    allowedPeRegs := allowedPeRegs ++ #[regKernelStage, regOut0, regOut1, regOut2, regOut3, regOut4, regOut5]
  if twoSource then
    allowedPeRegs := allowedPeRegs ++ #[
      regPeBcast, regPeOp, regSrc2Cfg,
      regSrc2Base, regSrc2ChanStride, regSrc2RowStride
    ]
    if !src2Cfg?.get!.isL2 then
      allowedPeRegs := allowedPeRegs ++ #[
        regSrc2RowBytesA, regSrc2RowBytesB,
        regSrc2Dep, regSrc2DmaCfg, regSrc2BaseLo, regSrc2BaseHi,
        regSrc2DmaRowStride, regSrc2DmaPlaneStride, regSrc2Fmt
      ]
  else
    if isFirst then
      allowedPeRegs := allowedPeRegs.push regSrc2Cfg
    match peOpW? with
    | some _ => allowedPeRegs := allowedPeRegs.push regPeOp
    | none => allowedPeRegs := allowedPeRegs ++ #[regPeBias, regPeScale, regPeInScale]
  vCheckAllowedRegs task.writes allowedPeRegs prefixMsg

  pure {
    taskIndex := task.taskIndex
    kind := .peOp
    engine := .pe
    isNeCompute := false
    liveL2Mask := h7
    l2Accesses := l2Accs
    dramAccesses := dramAccs
    sync := sync
  }

/-- Single-task compatibility wrapper for `verifyPeTaskMulti`. -/
def verifyPeTask (info : HwxInfo) (task : DecodedTask) : Except String Unit := do
  let _ ← verifyPeTaskMulti 1 false false info task
  pure ()

-- ============================================================================
-- Multi-Task Standalone NE Activation / LUT static safety verifier (S1-S4)
-- ============================================================================

/-- Independent static safety verifier for decoded standalone NE activation / 128-byte LUT tasks
    in single-task or multi-task streams. -/
def verifyNeActTaskMulti
    (taskCount : Nat) (hasPriorNe hasPriorPe : Bool)
    (info : HwxInfo) (task : DecodedTask) (kernelBytes? : Option ByteArray := none) :
    Except String VerifiedTaskFootprint := do
  let target := info.target
  let prefixMsg := s!"task {task.taskIndex} (NE act)"
  let isFirst := task.taskIndex == 0
  let isLast := task.taskIndex + 1 == taskCount

  -- 1. Verify 9-word header
  let (taskId, ene, h7) ← vVerifyTaskHeader9 target task taskCount false true prefixMsg
  let logNE := ene - 1

  -- 2. Reject duplicate writes
  vCheckNoDuplicateWrites task.writes prefixMsg

  -- 3. Common block
  if (findDecodedWrite task.writes regConvCfg).isSome then
    throw s!"{prefixMsg}: standalone NE act must omit convCfg (0x0028)"
  let wDw ← requireDecodedWrite task.writes regDwGroups "dwGroups"
  let perGroupIn ← decodeFieldChecked dwGroupsLayout wDw.value "perGroupIn"
  if perGroupIn != 1 then
    throw s!"{prefixMsg}: dwGroups (0x0030) must be 0x00014000 (perGroupIn=1)"

  let wWin ← requireDecodedWrite task.writes regWin "Win"
  let wHin ← requireDecodedWrite task.writes regHin "Hin"
  let wCin ← requireDecodedWrite task.writes regCin "Cin"
  let wWout ← requireDecodedWrite task.writes regWout "Wout"
  let wHout ← requireDecodedWrite task.writes regHout "Hout"
  let wCout ← requireDecodedWrite task.writes regCout "Cout"
  let wTH ← requireDecodedWrite task.writes regTH "TH"
  let wNeCommon ← requireDecodedWrite task.writes regNeCommon "neCommon"
  let wNeTile ← requireDecodedWrite task.writes regNeTile "neTile"
  let wTargetCfg ← requireDecodedWrite task.writes regTargetCfg "targetCfg"

  if let some wN := findDecodedWrite task.writes regBatchN then
    if wN.value != 1 then
      throw s!"{prefixMsg}: batch N (0x0024) must be 1, got {wN.value}"

  let w := (← decodeFieldChecked (dim16Layout "Win") wWin.value "val").toNat
  let h := (← decodeFieldChecked (dim16Layout "Hin") wHin.value "val").toNat
  let c := wCin.value.toNat
  let wout := (← decodeFieldChecked (dim16Layout "Wout") wWout.value "val").toNat
  let hout := (← decodeFieldChecked (dim16Layout "Hout") wHout.value "val").toNat
  let cout := wCout.value.toNat
  let th := (← decodeFieldChecked (dim16Layout "TH") wTH.value "val").toNat

  if w == 0 || h == 0 || c == 0 then
    throw s!"{prefixMsg}: invalid dims (C={c}, H={h}, W={w})"
  if wout != w || hout != h || cout != c then
    throw s!"{prefixMsg}: input dims ({c}x{h}x{w}) != output dims ({cout}x{hout}x{wout})"
  if th == 0 || th > h then
    throw s!"{prefixMsg}: tileHeight={th} out of range 1..{h}"
  if wTargetCfg.value != targetCfgValue target then
    throw s!"{prefixMsg}: targetCfg {toHex wTargetCfg.value.toNat} != expected {toHex (targetCfgValue target).toNat}"

  let kind := (← decodeFieldChecked neCommonLayout wNeCommon.value "kind").toNat
  let shPref := (← decodeFieldChecked neCommonLayout wNeCommon.value "shPref").toNat
  let shMin := (← decodeFieldChecked neCommonLayout wNeCommon.value "shMin").toNat
  let shMax := (← decodeFieldChecked neCommonLayout wNeCommon.value "shMax").toNat
  let f := (← decodeFieldChecked neCommonLayout wNeCommon.value "F").toNat
  let wide3x3Bit := (← decodeFieldChecked neCommonLayout wNeCommon.value "wide3x3").toNat
  if kind != 0 || wide3x3Bit != 0 then
    throw s!"{prefixMsg}: neCommon kind={kind}, wide3x3={wide3x3Bit} must both be 0"
  if shMin != shPref || shPref > shMax || shMax > 4 then
    throw s!"{prefixMsg}: invalid shifts shPref={shPref}, shMin={shMin}, shMax={shMax}"

  let ocg := (← decodeFieldChecked neTileLayout wNeTile.value "ocg").toNat
  let fat := (← decodeFieldChecked neTileLayout wNeTile.value "fat").toNat
  let wu := (← decodeFieldChecked neTileLayout wNeTile.value "WU").toNat
  if ocg != 0 then
    throw s!"{prefixMsg}: standalone NE act requires ocg=0, got {ocg}"
  if fat > 1 || wu > 3 || wu + f + fat != logNE then
    throw s!"{prefixMsg}: invalid neTile (WU={wu}, F={f}, fat={fat}, logNE={logNE})"
  let stripW := vStripW fat shPref
  if stripW < 8 || stripW > 256 then
    throw s!"{prefixMsg}: strip width T={stripW} outside [8, 256]"

  -- 4. Decode port configs (`src1Cfg`, `src2Cfg`, `resCfg`)
  let wSrc1Cfg ← requireDecodedWrite task.writes regSrc1Cfg "src1Cfg"
  let src1Cfg ← vDecodeSrcPortCfg "src1Cfg" wSrc1Cfg.value isFirst hasPriorNe hasPriorPe prefixMsg
  if isFirst then
    let wSrc2Cfg ← requireDecodedWrite task.writes regSrc2Cfg "src2Cfg"
    if wSrc2Cfg.value != src2CfgValue then
      throw s!"{prefixMsg}: src2Cfg {toHex wSrc2Cfg.value.toNat} != {toHex src2CfgValue.toNat}"
  else
    vCheckOmitted task.writes #[(regSrc2Cfg, "src2Cfg")] "NE act taskId > 0" prefixMsg

  let wResCfg ← requireDecodedWrite task.writes regResCfg "resCfg"
  let (resIsL2, direct) ← vDecodeResPortCfg wResCfg.value isFirst isLast prefixMsg

  -- 5. Resolve DMA endpoints (when DMA-backed)
  let mut dramAccs : Array VerifiedDramAccess := #[]
  let mut sync : TaskSyncInfo := {
    src1WaitNe5 := src1Cfg.waitNe5
    src1WaitPe22 := src1Cfg.waitPe22
    src1HandoffNe := src1Cfg.handoffNe
    src1HandoffPe := src1Cfg.handoffPe
    hasSrc1Dma := !src1Cfg.isL2
  }

  let src1DmaTarget? ←
    if src1Cfg.isL2 then
      vCheckOmitted task.writes #[
        (regSrc1Dep, "src1Dep"), (regSrc1DmaCfg, "src1DmaCfg"),
        (regSrc1BaseLo, "src1BaseLo"), (regSrc1BaseHi, "src1BaseHi"),
        (regSrc1DmaRowStride, "src1DmaRowStride"), (regSrc1DmaPlaneStride, "src1DmaPlaneStride"),
        (regSrc1Fmt, "src1Fmt")
      ] "L2-resident NE act src1" prefixMsg
      pure none
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regSrc1BaseLo regSrc1BaseHi "src1BaseLo" "src1BaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .input then
          throw s!"{prefixMsg}: src1Base BAR slot {slot} bound to non-input surface '{s.shortName}'"
        if s.n != 1 then throw s!"{prefixMsg}: single-batch input surface required"
        if s.channels.toNat != c || h > s.height.toNat || s.width.toNat != w then
          throw s!"{prefixMsg}: input surface dims mismatch"
        pure (some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: scratch src1Base must use BAR slot {barSlotScratch}, got {slot}"
        if isFirst then throw s!"{prefixMsg}: first task cannot read uninitialized DRAM scratch"
        pure (some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: src1Base BAR slot {slot} does not reference an input surface or scratch"

  let resDmaTarget? ←
    if resIsL2 then
      vCheckOmitted task.writes #[
        (regResDep, "resDep"), (regResBaseLo, "resBaseLo"), (regResBaseHi, "resBaseHi"),
        (regResDmaRowStride, "resDmaRowStride"), (regResDmaPlaneStride, "resDmaPlaneStride"),
        (regResFmt, "resFmt")
      ] "L2-resident NE act res" prefixMsg
      pure none
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regResBaseLo regResBaseHi "resBaseLo" "resBaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .output then
          throw s!"{prefixMsg}: resBase BAR slot {slot} bound to non-output surface '{s.shortName}'"
        if s.n != 1 then throw s!"{prefixMsg}: single-batch output surface required"
        if s.channels.toNat != c || h > s.height.toNat || s.width.toNat != w then
          throw s!"{prefixMsg}: output surface dims mismatch"
        pure (some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: scratch resBase must use BAR slot {barSlotScratch}, got {slot}"
        if isLast then throw s!"{prefixMsg}: final task cannot write to DRAM scratch"
        pure (some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: resBase BAR slot {slot} does not reference an output surface or scratch"

  -- 6. NE activation mode, input shift, kernelStage, and standalone LUT descriptor
  let wNeOp ← requireDecodedWrite task.writes regNeOp "neOp"
  let noShiftBit ← decodeFieldChecked neActOpLayout wNeOp.value "noShift"
  let reluBit ← decodeFieldChecked neActOpLayout wNeOp.value "relu"
  let lutBit ← decodeFieldChecked neActOpLayout wNeOp.value "lut"
  if reluBit == 1 && lutBit == 1 then
    throw s!"{prefixMsg}: neActOp cannot enable both relu (bit 16) and lut (bit 17)"
  let hasLut := lutBit == 1

  let wNeScale ← requireDecodedWrite task.writes regNeScale "neScale"
  let neScaleVal ← decodeFieldChecked neScaleLayout wNeScale.value "val"
  if !hasLut && neScaleVal != fp16OneBits then
    throw s!"{prefixMsg}: non-LUT neScale {toHex neScaleVal.toNat} != fp16 1.0 ({toHex fp16OneBits.toNat})"

  if noShiftBit == 0 then
    if !hasLut then
      throw s!"{prefixMsg}: input shift (noShift=0) is only valid with lut=1"
    let wShiftEn ← requireDecodedWrite task.writes regNeActShiftEn "neActShiftEn"
    if (← decodeFieldChecked neActShiftEnLayout wShiftEn.value "enable") != 1 then
      throw s!"{prefixMsg}: neActShiftEn (0x0050) must be 1 when noShift=0"
    let wBias ← requireDecodedWrite task.writes regScalarBias "scalarBias"
    validateWordBits scalarBiasLayout wBias.value
  else
    vCheckOmitted task.writes #[
      (regNeActShiftEn, "neActShiftEn"), (regScalarBias, "scalarBias")
    ] "noShift=1" prefixMsg

  vVerifyKernelStageMulti task.writes taskId isLast hasLut false prefixMsg

  if hasLut then
    let wNeCfg ← requireDecodedWrite task.writes regNeCfg "neCfg"
    if wNeCfg.value != neActCfgValue then
      throw s!"{prefixMsg}: standalone LUT neCfg {toHex wNeCfg.value.toNat} != expected {toHex neActCfgValue.toNat}"
    let wLutCfg ← requireDecodedWrite task.writes regLutCfg "lutCfg"
    let lEn ← decodeFieldChecked chunkSlotLayout wLutCfg.value "enable"
    let lOn20 ← decodeFieldChecked chunkSlotLayout wLutCfg.value "on20"
    let lTask ← decodeFieldChecked chunkSlotLayout wLutCfg.value "task"
    if lEn != 1 || lOn20 != 1 || lTask.toNat != taskId then
      throw s!"{prefixMsg}: lutCfg (0x5610) invalid ({toHex wLutCfg.value.toNat})"
    let wLutSize ← requireDecodedWrite task.writes regLutSize "lutSize"
    if wLutSize.value.toNat != vLutBytes then
      throw s!"{prefixMsg}: lutSize (0x5614) = {wLutSize.value}, expected {vLutBytes}"
    let (lutSlot, lutTarget, lutOff, lutLoPayloadOff) ←
      resolveBasePair info task.writes regLutBaseLo regLutBaseHi "lutBaseLo" "lutBaseHi" prefixMsg
    let kSec ← match lutTarget with
      | .kernel s => pure s
      | _ => throw s!"{prefixMsg}: lutBase BAR slot {lutSlot} does not reference __KERN_0"
    if !info.relocPatchOffsets.contains lutLoPayloadOff.toUInt32 then
      throw s!"{prefixMsg}: missing Mach-O relocation targeting lutBaseLo payload offset {toHex lutLoPayloadOff}"
    if lutOff % vDmaLineBytes != 0 then
      throw s!"{prefixMsg}: lutBase offset {toHex lutOff} is not {vDmaLineBytes}-byte aligned"
    if lutOff + vLutBytes > kSec.size.toNat then
      throw s!"{prefixMsg}: LUT range [{lutOff}, {lutOff + vLutBytes}) exceeds __KERN_0 size {kSec.size}"
    if let some kBytes := kernelBytes? then
      vVerifyLutBytes kBytes lutOff s!"{prefixMsg} LUT"
    dramAccs := dramAccs.push {
      resource := .kernel
      lo := lutOff
      hi := lutOff + vLutBytes
      isWrite := false
      port := .lut
    }
  else
    vCheckOmitted task.writes #[
      (regNeCfg, "neCfg"), (regLutCfg, "lutCfg"), (regLutSize, "lutSize"),
      (regLutBaseLo, "lutBaseLo"), (regLutBaseHi, "lutBaseHi")
    ] "non-LUT NE act" prefixMsg

  -- 7. L2 staging / resident bounds
  let mut l2Accs : Array VerifiedL2Access := #[]

  let wSrc1Base ← requireDecodedWrite task.writes regSrc1Base "src1Base"
  let wSrc1Chs ← requireDecodedWrite task.writes regSrc1ChanStride "src1ChanStride"
  let wSrc1Row ← requireDecodedWrite task.writes regSrc1RowStride "src1RowStride"
  let srcL2Base := wSrc1Base.value.toNat
  if src1Cfg.isL2 then
    vCheckOmitted task.writes #[
      (regSrc1RowBytesA, "src1RowBytesA"), (regSrc1RowBytesB, "src1RowBytesB")
    ] "L2-resident NE act src1" prefixMsg
    let (expChs, expRow, l2Bytes) := vL2ResidentGeometry c h w
    if wSrc1Chs.value.toNat != expChs then
      throw s!"{prefixMsg}: L2 src1ChanStride {wSrc1Chs.value} != expected {expChs}"
    if wSrc1Row.value.toNat != expRow then
      throw s!"{prefixMsg}: L2 src1RowStride {wSrc1Row.value} != expected {expRow}"
    l2Accs := l2Accs.push { lo := srcL2Base, hi := srcL2Base + l2Bytes, isWrite := false, role := .src1L2 }
  else
    let wSrc1RowA ← requireDecodedWrite task.writes regSrc1RowBytesA "src1RowBytesA"
    let wSrc1RowB ← requireDecodedWrite task.writes regSrc1RowBytesB "src1RowBytesB"
    let expRow := 2 * min (alignUp w 8) stripW
    if wSrc1Chs.value.toNat != expRow then
      throw s!"{prefixMsg}: src1ChanStride {wSrc1Chs.value} != expected {expRow}"
    let minU := c * expRow
    let src1Row := wSrc1Row.value.toNat
    if src1Row < minU || (src1Row - minU) % 16 != 0 then
      throw s!"{prefixMsg}: src1RowStride {src1Row} incompatible with C*row={minU}"
    if wSrc1RowA.value.toNat != minU || wSrc1RowB.value.toNat != minU then
      throw s!"{prefixMsg}: src1RowBytesA/B mismatch"
    l2Accs := l2Accs.push { lo := srcL2Base, hi := srcL2Base + th * src1Row, isWrite := true, role := .src1Stage }

  let wResBase ← requireDecodedWrite task.writes regResBase "resBase"
  let resL2Base := wResBase.value.toNat
  if resIsL2 then
    vCheckOmitted task.writes #[
      (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
    ] "L2-resident NE act res" prefixMsg
    let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
    let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
    let (expChs, expRow, l2Bytes) := vL2ResidentGeometry c h w
    if wResChs.value.toNat != expChs then
      throw s!"{prefixMsg}: L2 resChanStride {wResChs.value} != expected {expChs}"
    if wResRow.value.toNat != expRow then
      throw s!"{prefixMsg}: L2 resRowStride {wResRow.value} != expected {expRow}"
    l2Accs := l2Accs.push { lo := resL2Base, hi := resL2Base + l2Bytes, isWrite := true, role := .resL2 }
  else
    let expDirect := vIsDirect target fat shMax w
    if direct != expDirect then
      throw s!"{prefixMsg}: resCfg direct={direct} disagrees with target/tiling direct={expDirect}"
    if direct then
      vCheckOmitted task.writes #[
        (regResChanStride, "resChanStride"), (regResRowStride, "resRowStride"),
        (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
      ] "direct mode" prefixMsg
      let resDirectBytes := vNeActDirectResBytes f fat
      l2Accs := l2Accs.push { lo := resL2Base, hi := resL2Base + resDirectBytes, isWrite := true, role := .resStage }
    else
      let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
      let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
      let wResRowA ← requireDecodedWrite task.writes regResRowBytesA "resRowBytesA"
      let wResRowB ← requireDecodedWrite task.writes regResRowBytesB "resRowBytesB"
      let expDrow := vDchs w stripW
      if wResChs.value.toNat != expDrow then
        throw s!"{prefixMsg}: resChanStride {wResChs.value} != expected {expDrow}"
      let minDu := c * expDrow
      let resRow := wResRow.value.toNat
      if resRow < minDu || (resRow - minDu) % 16 != 0 then
        throw s!"{prefixMsg}: resRowStride {resRow} incompatible with C*drow={minDu}"
      if wResRowA.value.toNat != minDu || wResRowB.value.toNat != minDu then
        throw s!"{prefixMsg}: resRowBytesA/B mismatch"
      l2Accs := l2Accs.push { lo := resL2Base, hi := resL2Base + th * resRow, isWrite := true, role := .resStage }

  vCheckTaskL2Accesses l2Accs prefixMsg

  -- 8. DMA deps, formats, bounds, output stage, and allowlist
  if let some (_, tgt, off) := src1DmaTarget? then
    let (dk, dd) ← vDecodeSrcDep info task.writes regSrc1Dep regSrc1DmaCfg "src1Dep" "src1DmaCfg" taskId isFirst prefixMsg
    sync := { sync with src1DepKind := dk, src1DepDist := dd }
    let wFmt ← requireDecodedWrite task.writes regSrc1Fmt "src1Fmt"
    if wFmt.value != srcFmtWordP1 then
      throw s!"{prefixMsg}: src1Fmt mismatch"
    let (resId, cap, isSurf, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, true, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, false, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, false, "invalid")
    let (ext, _) ← vVerifyDmaStridesAndBounds task.writes c h w off cap isSurf
      regSrc1DmaRowStride regSrc1DmaPlaneStride "src1DmaRowStride" "src1DmaPlaneStride" "input DMA" lbl prefixMsg
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + ext, isWrite := false, port := .src1Dma }

  if let some (_, tgt, off) := resDmaTarget? then
    match ← resolveBarTarget info barSlotConfigZero with
    | .configZero => pure ()
    | _ => throw s!"{prefixMsg}: BAR slot {barSlotConfigZero} must have VM base 0"
    let wResDep ← requireDecodedWrite task.writes regResDep "resDep"
    if wResDep.barSlot != some barSlotConfigZero then
      throw s!"{prefixMsg}: resDep must use BAR slot {barSlotConfigZero}"
    let expCode := match tgt with
      | .scratch _ => resDepCode target ||| resDepScratchFlags
      | _ => resDepCode target
    if (← decodeFieldChecked resDepLayout wResDep.value "code") != expCode then
      throw s!"{prefixMsg}: resDep code mismatch"
    if (← decodeFieldChecked resDepLayout wResDep.value "task").toNat != taskId then
      throw s!"{prefixMsg}: resDep task != header taskId={taskId}"
    if ((← decodeFieldChecked resDepLayout wResDep.value "direct") == 1) != direct then
      throw s!"{prefixMsg}: resDep direct bit != resCfg direct={direct}"
    let wResFmt ← requireDecodedWrite task.writes regResFmt "resFmt"
    if wResFmt.value != resFmtWordP1 then
      throw s!"{prefixMsg}: resFmt mismatch"
    let (resId, cap, isSurf, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, true, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, false, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, false, "invalid")
    let (ext, outCov) ← vVerifyDmaStridesAndBounds task.writes c h w off cap isSurf
      regResDmaRowStride regResDmaPlaneStride "resDmaRowStride" "resDmaPlaneStride" "output DMA" lbl prefixMsg
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + ext, isWrite := true, port := .resDma, coverage := outCov }

  vVerifyOutputStageMulti task.writes taskId isLast prefixMsg

  let mut allowedActRegs : Array Nat := #[
    regWin, regHin, regCin, regWout, regHout, regCout, regBatchN,
    regDwGroups, regTH, regNeCommon, regNeTile, regTargetCfg,
    regSrc1Cfg, regSrc1Base, regSrc1ChanStride, regSrc1RowStride,
    regResCfg, regResBase,
    regNeOp, regNeScale
  ]
  if isFirst then
    allowedActRegs := allowedActRegs.push regSrc2Cfg
  if !src1Cfg.isL2 then
    allowedActRegs := allowedActRegs ++ #[
      regSrc1RowBytesA, regSrc1RowBytesB,
      regSrc1Dep, regSrc1DmaCfg, regSrc1BaseLo, regSrc1BaseHi,
      regSrc1DmaRowStride, regSrc1DmaPlaneStride, regSrc1Fmt
    ]
  if resIsL2 then
    allowedActRegs := allowedActRegs ++ #[regResChanStride, regResRowStride]
  else
    allowedActRegs := allowedActRegs ++ #[
      regResDep, regResBaseLo, regResBaseHi,
      regResDmaRowStride, regResDmaPlaneStride, regResFmt
    ]
    if !direct then
      allowedActRegs := allowedActRegs ++ #[
        regResChanStride, regResRowStride, regResRowBytesA, regResRowBytesB
      ]
  if hasLut then
    allowedActRegs := allowedActRegs ++ #[
      regNeCfg, regKernelStage, regLutCfg, regLutSize, regLutBaseLo, regLutBaseHi
    ]
  else if isLast then
    allowedActRegs := allowedActRegs.push regKernelStage
  if isLast then
    allowedActRegs := allowedActRegs ++ #[regOut0, regOut1, regOut2, regOut3, regOut4, regOut5]
  if noShiftBit == 0 then
    allowedActRegs := allowedActRegs ++ #[regNeActShiftEn, regScalarBias]
  vCheckAllowedRegs task.writes allowedActRegs prefixMsg

  pure {
    taskIndex := task.taskIndex
    kind := .neAct
    engine := .ne
    isNeCompute := hasLut
    liveL2Mask := h7
    l2Accesses := l2Accs
    dramAccesses := dramAccs
    sync := sync
  }

/-- Single-task compatibility wrapper for `verifyNeActTaskMulti`. -/
def verifyNeActTask (info : HwxInfo) (task : DecodedTask) (kernelBytes? : Option ByteArray := none) :
    Except String Unit := do
  let _ ← verifyNeActTaskMulti 1 false false info task kernelBytes?
  pure ()

-- ============================================================================
-- Multi-Task PE Pool, PE Reduce, Weightless NE Pool & NE Channel Reduce (S1-S4)
-- ============================================================================

/-- Independent static safety verifier for PE Window Pooling tasks (`0x003c = 0x10`, `ene = 0`). -/
def verifyPePoolTaskMulti
    (taskCount : Nat) (hasPriorNe hasPriorPe : Bool)
    (info : HwxInfo) (task : DecodedTask) : Except String VerifiedTaskFootprint := do
  let target := info.target
  let prefixMsg := s!"task {task.taskIndex} (PE pool)"
  let isFirst := task.taskIndex == 0
  let isLast := task.taskIndex + 1 == taskCount

  let (taskId, _, h7) ← vVerifyTaskHeader9 target task taskCount true false prefixMsg
  vCheckNoDuplicateWrites task.writes prefixMsg

  vCheckOmitted task.writes #[
    (regDwGroups, "dwGroups"), (regNeTile, "neTile"), (regPeBcast, "peBcast")
  ] "PE pool task" prefixMsg

  let wWin ← requireDecodedWrite task.writes regWin "Win"
  let wHin ← requireDecodedWrite task.writes regHin "Hin"
  let wCin ← requireDecodedWrite task.writes regCin "Cin"
  let wWout ← requireDecodedWrite task.writes regWout "Wout"
  let wHout ← requireDecodedWrite task.writes regHout "Hout"
  let wCout ← requireDecodedWrite task.writes regCout "Cout"
  let wConvCfg ← requireDecodedWrite task.writes regConvCfg "convCfg"
  let wTH ← requireDecodedWrite task.writes regTH "TH"
  let wPeCommon ← requireDecodedWrite task.writes regPeCommon "peCommon"
  let wPeTile ← requireDecodedWrite task.writes regPeTile "peTile"
  let wTargetCfg ← requireDecodedWrite task.writes regTargetCfg "targetCfg"

  if let some wN := findDecodedWrite task.writes regBatchN then
    if wN.value != 1 then
      throw s!"{prefixMsg}: batch N (0x0024) must be 1, got {wN.value}"

  let winReg := (← decodeFieldChecked (dim16Layout "Win") wWin.value "val").toNat
  let hinReg := (← decodeFieldChecked (dim16Layout "Hin") wHin.value "val").toNat
  let cin := wCin.value.toNat
  let wout := (← decodeFieldChecked (dim16Layout "Wout") wWout.value "val").toNat
  let hout := (← decodeFieldChecked (dim16Layout "Hout") wHout.value "val").toNat
  let cout := wCout.value.toNat
  let th := (← decodeFieldChecked (dim16Layout "TH") wTH.value "val").toNat

  if winReg == 0 || hinReg == 0 || cin == 0 || cin > 65535 || wout == 0 || hout == 0 || cout != cin then
    throw s!"{prefixMsg}: invalid PE pool dims (in={cin}x{hinReg}x{winReg}, out={cout}x{hout}x{wout})"
  if wTargetCfg.value != 0 then
    throw s!"{prefixMsg}: PE pool targetCfg (0x004c) must be 0, got {toHex wTargetCfg.value.toNat}"
  let peMode ← decodeFieldChecked peCommonLayout wPeCommon.value "mode"
  if peMode != peCommonPoolValue then
    throw s!"{prefixMsg}: peCommon mode {toHex peMode.toNat} != expected 0x10"

  let kw := (← decodeFieldChecked convCfgLayout wConvCfg.value "kw").toNat
  let kh := (← decodeFieldChecked convCfgLayout wConvCfg.value "kh").toNat
  let sx := (← decodeFieldChecked convCfgLayout wConvCfg.value "sx").toNat
  let sy := (← decodeFieldChecked convCfgLayout wConvCfg.value "sy").toNat
  let pl := (← decodeFieldChecked convCfgLayout wConvCfg.value "pl").toNat
  let pt := (← decodeFieldChecked convCfgLayout wConvCfg.value "pt").toNat
  let hx := (← decodeFieldChecked convCfgLayout wConvCfg.value "hx").toNat
  let vy := (← decodeFieldChecked convCfgLayout wConvCfg.value "vy").toNat
  if hx != 1 || vy != 1 then
    throw s!"{prefixMsg}: convCfg hx={hx}, vy={vy} must both be 1"
  if kh == 0 || kh > 5 || kw == 0 || kw > 5 then
    throw s!"{prefixMsg}: PE pool kernel ({kh}x{kw}) must be in 1..5"
  if (sx != 1 && sx != 2) || (sy != 1 && sy != 2) then
    throw s!"{prefixMsg}: PE pool strides (sy={sy}, sx={sx}) must be 1 or 2"
  if pl >= kw || pt >= kh then
    throw s!"{prefixMsg}: PE pool padding (pt={pt}, pl={pl}) must be < kernel ({kh}, {kw})"

  let k := (← decodeFieldChecked peTileLayout wPeTile.value "k").toNat
  let m := (← decodeFieldChecked peTileLayout wPeTile.value "m").toNat
  if k < 5 || k > 9 then
    throw s!"{prefixMsg}: peTile k={k} out of range 5..9"
  if th == 0 || th > hout then
    throw s!"{prefixMsg}: tileHeight={th} out of range 1..{hout}"
  let expM := min (9 - k) (vCeilLog2 th)
  if m != expM then
    throw s!"{prefixMsg}: peTile m={m} != expected {expM}"
  let stripW := (1 : Nat) <<< k

  -- Verify compare & PE pool op registers (0x4100, 0x4500)
  let peOpVal : UInt32 := match findDecodedWrite task.writes regPeOp with
    | some w => w.value
    | none => 0
  let cmpW? := findDecodedWrite task.writes regCmpFlags
  if peOpVal == 0x2 then
    let some wc := cmpW?
      | throw s!"{prefixMsg}: PE max_pool (0x4500=0x2) requires cmpFlags (0x4100=0x4)"
    if wc.value != cmpFlagMaxValue then
      throw s!"{prefixMsg}: PE max_pool cmpFlags {toHex wc.value.toNat} != expected 0x4"
  else if peOpVal == 0x0 || peOpVal == 0x1 then
    if let some wc := cmpW? then
      throw s!"{prefixMsg}: PE avg_pool (0x4500={toHex peOpVal.toNat}) must omit cmpFlags (0x4100), got {toHex wc.value.toNat}"
  else
    throw s!"{prefixMsg}: illegal PE pool 0x4500 opcode {toHex peOpVal.toNat}"

  -- Port configs
  let wSrc1Cfg ← requireDecodedWrite task.writes regSrc1Cfg "src1Cfg"
  let src1Cfg ← vDecodeSrcPortCfg "src1Cfg" wSrc1Cfg.value isFirst hasPriorNe hasPriorPe prefixMsg
  if isFirst then
    let wSrc2Cfg ← requireDecodedWrite task.writes regSrc2Cfg "src2Cfg"
    if wSrc2Cfg.value != src2CfgValue then
      throw s!"{prefixMsg}: src2Cfg {toHex wSrc2Cfg.value.toNat} != {toHex src2CfgValue.toNat}"
  else
    vCheckOmitted task.writes #[(regSrc2Cfg, "src2Cfg")] "PE pool taskId > 0" prefixMsg

  let wResCfg ← requireDecodedWrite task.writes regResCfg "resCfg"
  let (resIsL2, direct) ← vDecodeResPortCfg wResCfg.value isFirst isLast prefixMsg

  let mut dramAccs : Array VerifiedDramAccess := #[]
  let mut sync : TaskSyncInfo := {
    src1WaitNe5 := src1Cfg.waitNe5
    src1WaitPe22 := src1Cfg.waitPe22
    src1HandoffNe := src1Cfg.handoffNe
    src1HandoffPe := src1Cfg.handoffPe
    hasSrc1Dma := !src1Cfg.isL2
  }

  let (fullH, fullW, hasExactFull, src1DmaTarget?) ←
    if src1Cfg.isL2 then
      vCheckOmitted task.writes #[
        (regSrc1Dep, "src1Dep"), (regSrc1DmaCfg, "src1DmaCfg"),
        (regSrc1BaseLo, "src1BaseLo"), (regSrc1BaseHi, "src1BaseHi"),
        (regSrc1DmaRowStride, "src1DmaRowStride"), (regSrc1DmaPlaneStride, "src1DmaPlaneStride"),
        (regSrc1Fmt, "src1Fmt")
      ] "L2-resident PE pool src1" prefixMsg
      pure (hinReg, winReg, false, none)
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regSrc1BaseLo regSrc1BaseHi "src1BaseLo" "src1BaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .input then
          throw s!"{prefixMsg}: src1Base BAR slot {slot} bound to non-input surface '{s.shortName}'"
        if s.n != 1 then throw s!"{prefixMsg}: single-batch input surface required"
        if s.channels.toNat != cin then
          throw s!"{prefixMsg}: Cin={cin} != input surface channels {s.channels}"
        pure (s.height.toNat, s.width.toNat, true, some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: scratch src1Base must use BAR slot {barSlotScratch}, got {slot}"
        if isFirst then throw s!"{prefixMsg}: first task cannot read uninitialized DRAM scratch"
        pure (hinReg, winReg, false, some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: src1Base BAR slot {slot} does not reference an input surface or scratch"

  vVerifyPoolAxisGeom fullH kh sy pt hinReg hout hasExactFull "H" prefixMsg
  vVerifyPoolAxisGeom fullW kw sx pl winReg wout hasExactFull "W" prefixMsg

  let resDmaTarget? ←
    if resIsL2 then
      vCheckOmitted task.writes #[
        (regResDep, "resDep"), (regResBaseLo, "resBaseLo"), (regResBaseHi, "resBaseHi"),
        (regResDmaRowStride, "resDmaRowStride"), (regResDmaPlaneStride, "resDmaPlaneStride"),
        (regResFmt, "resFmt")
      ] "L2-resident PE pool res" prefixMsg
      pure none
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regResBaseLo regResBaseHi "resBaseLo" "resBaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .output then
          throw s!"{prefixMsg}: resBase BAR slot {slot} bound to non-output surface '{s.shortName}'"
        if s.n != 1 then throw s!"{prefixMsg}: single-batch output surface required"
        if s.channels.toNat != cout || s.height.toNat != hout || s.width.toNat != wout then
          throw s!"{prefixMsg}: output surface ({s.channels}x{s.height}x{s.width}) != pool output ({cout}x{hout}x{wout})"
        pure (some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: scratch resBase must use BAR slot {barSlotScratch}, got {slot}"
        if isLast then throw s!"{prefixMsg}: final task cannot write to DRAM scratch"
        pure (some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: resBase BAR slot {slot} does not reference an output surface or scratch"

  -- L2 bounds
  let mut l2Accs : Array VerifiedL2Access := #[]
  let wSrc1Base ← requireDecodedWrite task.writes regSrc1Base "src1Base"
  let wSrc1Chs ← requireDecodedWrite task.writes regSrc1ChanStride "src1ChanStride"
  let wSrc1Row ← requireDecodedWrite task.writes regSrc1RowStride "src1RowStride"
  let srcBase := wSrc1Base.value.toNat
  if src1Cfg.isL2 then
    vCheckOmitted task.writes #[
      (regSrc1RowBytesA, "src1RowBytesA"), (regSrc1RowBytesB, "src1RowBytesB")
    ] "L2-resident PE pool src1" prefixMsg
    let minChs := 2 * alignUp winReg 8
    let gotChs := wSrc1Chs.value.toNat
    if gotChs < minChs || gotChs % 16 != 0 then
      throw s!"{prefixMsg}: L2 src1ChanStride {gotChs} invalid (must be >= {minChs} and multiple of 16)"
    let expRow := cin * gotChs
    if wSrc1Row.value.toNat != expRow then
      throw s!"{prefixMsg}: L2 src1RowStride {wSrc1Row.value} != expected {expRow}"
    let l2Bytes := alignUp (hinReg * expRow) vDmaLineBytes
    l2Accs := l2Accs.push { lo := srcBase, hi := srcBase + l2Bytes, isWrite := false, role := .src1L2 }
  else
    let wSrc1RowA ← requireDecodedWrite task.writes regSrc1RowBytesA "src1RowBytesA"
    let wSrc1RowB ← requireDecodedWrite task.writes regSrc1RowBytesB "src1RowBytesB"
    let expPx := vPoolStripPx winReg kw sx pl stripW wout
    let expRow := 2 * expPx
    if wSrc1Chs.value.toNat != expRow then
      throw s!"{prefixMsg}: src1ChanStride {wSrc1Chs.value} != expected {expRow}"
    let minU := cin * expRow
    let src1Row := wSrc1Row.value.toNat
    if src1Row < minU || (src1Row - minU) % 16 != 0 then
      throw s!"{prefixMsg}: src1RowStride {src1Row} incompatible with C*row={minU}"
    if wSrc1RowA.value.toNat != minU || wSrc1RowB.value.toNat != minU then
      throw s!"{prefixMsg}: src1RowBytesA/B mismatch"
    let inRows := vPoolInRows hinReg kh sy pt hout th false
    let srcBytes := alignUp (inRows * src1Row) vDmaLineBytes
    l2Accs := l2Accs.push { lo := srcBase, hi := srcBase + srcBytes, isWrite := true, role := .src1Stage }

  let wResBase ← requireDecodedWrite task.writes regResBase "resBase"
  let resBase := wResBase.value.toNat
  if resIsL2 then
    vCheckOmitted task.writes #[
      (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
    ] "L2-resident PE pool res" prefixMsg
    let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
    let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
    let (expChs, expRow, l2Bytes) := vL2ResidentGeometry cout hout wout
    if wResChs.value.toNat != expChs then
      throw s!"{prefixMsg}: L2 resChanStride {wResChs.value} != expected {expChs}"
    if wResRow.value.toNat != expRow then
      throw s!"{prefixMsg}: L2 resRowStride {wResRow.value} != expected {expRow}"
    l2Accs := l2Accs.push { lo := resBase, hi := resBase + l2Bytes, isWrite := true, role := .resL2 }
  else
    let expDirect := vPePoolDirect target wout k
    if direct != expDirect then
      throw s!"{prefixMsg}: resCfg direct={direct} disagrees with {target.name} PE pool direct={expDirect} (Ow={wout}, k={k})"
    let expDrow := 2 * min (alignUp wout 8) (max stripW 32)
    if direct then
      vCheckOmitted task.writes #[
        (regResChanStride, "resChanStride"), (regResRowStride, "resRowStride"),
        (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
      ] "direct mode" prefixMsg
      let resDirectBytes := vPeDirectResBytes cout expDrow th
      l2Accs := l2Accs.push { lo := resBase, hi := resBase + resDirectBytes, isWrite := true, role := .resStage }
    else
      let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
      let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
      let wResRowA ← requireDecodedWrite task.writes regResRowBytesA "resRowBytesA"
      let wResRowB ← requireDecodedWrite task.writes regResRowBytesB "resRowBytesB"
      if wResChs.value.toNat != expDrow then
        throw s!"{prefixMsg}: staged resChanStride {wResChs.value} != expected {expDrow}"
      let minDu := cout * expDrow
      let resRow := wResRow.value.toNat
      if resRow < minDu || (resRow - minDu) % 16 != 0 then
        throw s!"{prefixMsg}: staged resRowStride {resRow} incompatible with C*drow={minDu}"
      if wResRowA.value.toNat != minDu || wResRowB.value.toNat != minDu then
        throw s!"{prefixMsg}: staged resRowBytesA/B mismatch"
      l2Accs := l2Accs.push { lo := resBase, hi := resBase + alignUp (th * resRow) vDmaLineBytes, isWrite := true, role := .resStage }

  vCheckTaskL2Accesses l2Accs prefixMsg

  -- DMA deps, formats, and bounds
  if let some (_, tgt, off) := src1DmaTarget? then
    let (dk, dd) ← vDecodeSrcDep info task.writes regSrc1Dep regSrc1DmaCfg "src1Dep" "src1DmaCfg" taskId isFirst prefixMsg
    sync := { sync with src1DepKind := dk, src1DepDist := dd }
    let wFmt ← requireDecodedWrite task.writes regSrc1Fmt "src1Fmt"
    if wFmt.value != srcFmtWordP1 then
      throw s!"{prefixMsg}: src1Fmt {toHex wFmt.value.toNat} != {toHex srcFmtWordP1.toNat}"
    let (resId, cap, isSurf, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, true, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, false, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, false, "invalid")
    let (ext, _) ← vVerifyDmaStridesAndBounds task.writes cin fullH fullW off cap isSurf
      regSrc1DmaRowStride regSrc1DmaPlaneStride "src1DmaRowStride" "src1DmaPlaneStride" "input DMA" lbl prefixMsg
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + ext, isWrite := false, port := .src1Dma }

  if let some (_, tgt, off) := resDmaTarget? then
    match ← resolveBarTarget info barSlotConfigZero with
    | .configZero => pure ()
    | _ => throw s!"{prefixMsg}: BAR slot {barSlotConfigZero} must have VM base 0"
    let wResDep ← requireDecodedWrite task.writes regResDep "resDep"
    if wResDep.barSlot != some barSlotConfigZero then
      throw s!"{prefixMsg}: resDep must use BAR slot {barSlotConfigZero}"
    let expCode := match tgt with
      | .scratch _ => resDepCode target ||| resDepScratchFlags
      | _ => resDepCode target
    if (← decodeFieldChecked resDepLayout wResDep.value "code") != expCode then
      throw s!"{prefixMsg}: resDep code mismatch"
    if (← decodeFieldChecked resDepLayout wResDep.value "task").toNat != taskId then
      throw s!"{prefixMsg}: resDep task != header taskId={taskId}"
    if ((← decodeFieldChecked resDepLayout wResDep.value "direct") == 1) != direct then
      throw s!"{prefixMsg}: resDep direct bit != resCfg direct={direct}"
    let wResFmt ← requireDecodedWrite task.writes regResFmt "resFmt"
    if wResFmt.value != resFmtWordP1 then
      throw s!"{prefixMsg}: resFmt {toHex wResFmt.value.toNat} != {toHex resFmtWordP1.toNat}"
    let (resId, cap, isSurf, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, true, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, false, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, false, "invalid")
    let (ext, outCov) ← vVerifyDmaStridesAndBounds task.writes cout hout wout off cap isSurf
      regResDmaRowStride regResDmaPlaneStride "resDmaRowStride" "resDmaPlaneStride" "output DMA" lbl prefixMsg
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + ext, isWrite := true, port := .resDma, coverage := outCov }

  vVerifyKernelStageMulti task.writes taskId isLast false false prefixMsg
  vVerifyOutputStageMulti task.writes taskId isLast prefixMsg

  let mut allowed : Array Nat := #[
    regWin, regHin, regCin, regWout, regHout, regCout, regBatchN,
    regConvCfg, regTH, regPeCommon, regPeTile, regTargetCfg,
    regSrc1Cfg, regSrc1Base, regSrc1ChanStride, regSrc1RowStride,
    regResCfg, regResBase, regPeOp
  ]
  if peOpVal == 0x2 then allowed := allowed.push regCmpFlags
  if isFirst then allowed := allowed.push regSrc2Cfg
  if !src1Cfg.isL2 then
    allowed := allowed ++ #[
      regSrc1RowBytesA, regSrc1RowBytesB,
      regSrc1Dep, regSrc1DmaCfg, regSrc1BaseLo, regSrc1BaseHi,
      regSrc1DmaRowStride, regSrc1DmaPlaneStride, regSrc1Fmt
    ]
  if resIsL2 then
    allowed := allowed ++ #[regResChanStride, regResRowStride]
  else
    allowed := allowed ++ #[
      regResDep, regResBaseLo, regResBaseHi,
      regResDmaRowStride, regResDmaPlaneStride, regResFmt
    ]
    if !direct then
      allowed := allowed ++ #[
        regResChanStride, regResRowStride, regResRowBytesA, regResRowBytesB
      ]
  if isLast then
    allowed := allowed ++ #[regKernelStage, regOut0, regOut1, regOut2, regOut3, regOut4, regOut5]
  vCheckAllowedRegs task.writes allowed prefixMsg

  pure {
    taskIndex := task.taskIndex
    kind := .pePool
    engine := .pe
    isNeCompute := false
    liveL2Mask := h7
    l2Accesses := l2Accs
    dramAccesses := dramAccs
    sync := sync
  }

/-- Independent static safety verifier for PE Tensor Reduction tasks (`0x003c = 0x50`, `ene = 0`). -/
def verifyPeReduceTaskMulti
    (taskCount : Nat) (hasPriorNe hasPriorPe : Bool)
    (info : HwxInfo) (task : DecodedTask) : Except String VerifiedTaskFootprint := do
  let target := info.target
  let prefixMsg := s!"task {task.taskIndex} (PE reduce)"
  let isFirst := task.taskIndex == 0
  let isLast := task.taskIndex + 1 == taskCount

  let (taskId, _, h7) ← vVerifyTaskHeader9 target task taskCount true false prefixMsg
  vCheckNoDuplicateWrites task.writes prefixMsg

  vCheckOmitted task.writes #[
    (regConvCfg, "convCfg"), (regDwGroups, "dwGroups"), (regNeTile, "neTile"), (regPeBcast, "peBcast")
  ] "PE reduce task" prefixMsg

  let wWin ← requireDecodedWrite task.writes regWin "Win"
  let wHin ← requireDecodedWrite task.writes regHin "Hin"
  let wCin ← requireDecodedWrite task.writes regCin "Cin"
  let wDin ← requireDecodedWrite task.writes regDin "Din"
  let wBatchN ← requireDecodedWrite task.writes regBatchN "BatchN"
  let wWout ← requireDecodedWrite task.writes regWout "Wout"
  let wHout ← requireDecodedWrite task.writes regHout "Hout"
  let wCout ← requireDecodedWrite task.writes regCout "Cout"
  let wDout ← requireDecodedWrite task.writes regDout "Dout"
  let wTH ← requireDecodedWrite task.writes regTH "TH"
  let wPeCommon ← requireDecodedWrite task.writes regPeCommon "peCommon"
  let wPeTile ← requireDecodedWrite task.writes regPeTile "peTile"
  let wTargetCfg ← requireDecodedWrite task.writes regTargetCfg "targetCfg"

  let w := (← decodeFieldChecked (dim16Layout "Win") wWin.value "val").toNat
  let sH := (← decodeFieldChecked (dim16Layout "Hin") wHin.value "val").toNat
  let sC := wCin.value.toNat
  let sD := (← decodeFieldChecked (dim16Layout "Din") wDin.value "val").toNat
  let sN := (← decodeFieldChecked (dim16Layout "BatchN") wBatchN.value "val").toNat
  let wout := (← decodeFieldChecked (dim16Layout "Wout") wWout.value "val").toNat
  let hout := (← decodeFieldChecked (dim16Layout "Hout") wHout.value "val").toNat
  let cout := wCout.value.toNat
  let dout := (← decodeFieldChecked (dim16Layout "Dout") wDout.value "val").toNat
  let th := (← decodeFieldChecked (dim16Layout "TH") wTH.value "val").toNat

  if w == 0 || sH == 0 || sC == 0 || sC > 65535 || sD == 0 || sN == 0 then
    throw s!"{prefixMsg}: invalid PE reduce dims (N={sN}, C={sC}, H={sH}, D={sD}, W={w})"
  if wout != w || hout != sH || cout != sC || dout != sD then
    throw s!"{prefixMsg}: PE reduce output dims ({cout}x{hout}x{dout}x{wout}) != input dims ({sC}x{sH}x{sD}x{w})"
  if wTargetCfg.value != 0 then
    throw s!"{prefixMsg}: PE reduce targetCfg (0x004c) must be 0, got {toHex wTargetCfg.value.toNat}"
  let peMode ← decodeFieldChecked peCommonLayout wPeCommon.value "mode"
  if peMode != peCommonReduceValue then
    throw s!"{prefixMsg}: peCommon mode {toHex peMode.toNat} != expected 0x50"

  let k := (← decodeFieldChecked peTileLayout wPeTile.value "k").toNat
  let m := (← decodeFieldChecked peTileLayout wPeTile.value "m").toNat
  if k < 5 || k > 9 then
    throw s!"{prefixMsg}: peTile k={k} out of range 5..9"
  if th == 0 || th > sH then
    throw s!"{prefixMsg}: tileHeight={th} out of range 1..{sH}"
  let expM := min (9 - k) (vCeilLog2 th)
  if m != expM then
    throw s!"{prefixMsg}: peTile m={m} != expected {expM}"
  let stripW := (1 : Nat) <<< k

  -- Verify 0x4500 reduction opcode, 0x4100 compare flags, 0x450c eps, 0x4514 scale
  let opVal : UInt32 := match findDecodedWrite task.writes regPeOp with
    | some wr => wr.value
    | none => 0
  validateWordBits peReduceOpLayout opVal
  let legalOp :=
    opVal == 0x0000 || opVal == 0x0002 || opVal == 0x0003 || opVal == 0x0010 ||
    opVal == 0x01c0 || opVal == 0x1000 || opVal == 0x2010 || opVal == 0x3010
  if !legalOp then
    throw s!"{prefixMsg}: illegal PE reduce 0x4500 opcode {toHex opVal.toNat}"

  let cmpW? := findDecodedWrite task.writes regCmpFlags
  if opVal == 0x0002 then
    let some wc := cmpW? | throw s!"{prefixMsg}: PE reduce max requires cmpFlags (0x4100=0x4)"
    if wc.value != cmpFlagMaxValue then
      throw s!"{prefixMsg}: PE reduce max cmpFlags {toHex wc.value.toNat} != expected 0x4"
  else if opVal == 0x0003 then
    let some wc := cmpW? | throw s!"{prefixMsg}: PE reduce min requires cmpFlags (0x4100=0xc)"
    if wc.value != cmpFlagMinValue then
      throw s!"{prefixMsg}: PE reduce min cmpFlags {toHex wc.value.toNat} != expected 0xc"
  else
    if let some wc := cmpW? then
      throw s!"{prefixMsg}: PE reduce opcode {toHex opVal.toNat} must omit cmpFlags (0x4100), got {toHex wc.value.toNat}"

  if (findDecodedWrite task.writes regPeEps).isSome && opVal != 0x2010 then
    throw s!"{prefixMsg}: peEps (0x450c) is only allowed for rsqrt ops (0x4500=0x2010)"
  if (findDecodedWrite task.writes regPeRedScale).isSome && opVal != 0x0000 && opVal != 0x2010 then
    throw s!"{prefixMsg}: peRedScale (0x4514) is only allowed for mean / rsqrt_mean_square"

  -- Port configs
  let wSrc1Cfg ← requireDecodedWrite task.writes regSrc1Cfg "src1Cfg"
  let src1Cfg ← vDecodeSrcPortCfg "src1Cfg" wSrc1Cfg.value isFirst hasPriorNe hasPriorPe prefixMsg
  if isFirst then
    let wSrc2Cfg ← requireDecodedWrite task.writes regSrc2Cfg "src2Cfg"
    if wSrc2Cfg.value != src2CfgValue then
      throw s!"{prefixMsg}: src2Cfg {toHex wSrc2Cfg.value.toNat} != {toHex src2CfgValue.toNat}"
  else
    vCheckOmitted task.writes #[(regSrc2Cfg, "src2Cfg")] "PE reduce taskId > 0" prefixMsg

  let wResCfg ← requireDecodedWrite task.writes regResCfg "resCfg"
  let (resIsL2, direct) ← vDecodeResPortCfg wResCfg.value isFirst isLast prefixMsg

  let mut dramAccs : Array VerifiedDramAccess := #[]
  let mut sync : TaskSyncInfo := {
    src1WaitNe5 := src1Cfg.waitNe5
    src1WaitPe22 := src1Cfg.waitPe22
    src1HandoffNe := src1Cfg.handoffNe
    src1HandoffPe := src1Cfg.handoffPe
    hasSrc1Dma := !src1Cfg.isL2
  }

  let src1DmaTarget? ←
    if src1Cfg.isL2 then
      vCheckOmitted task.writes #[
        (regSrc1Dep, "src1Dep"), (regSrc1DmaCfg, "src1DmaCfg"),
        (regSrc1BaseLo, "src1BaseLo"), (regSrc1BaseHi, "src1BaseHi"),
        (regSrc1DmaRowStride, "src1DmaRowStride"), (regSrc1DmaPlaneStride, "src1DmaPlaneStride"),
        (regSrc1DmaDStride, "src1DmaDStride"), (regSrc1DmaNStride, "src1DmaNStride"),
        (regSrc1Fmt, "src1Fmt")
      ] "L2-resident PE reduce src1" prefixMsg
      pure none
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regSrc1BaseLo regSrc1BaseHi "src1BaseLo" "src1BaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .input then
          throw s!"{prefixMsg}: src1Base BAR slot {slot} bound to non-input surface '{s.shortName}'"
        if s.n != 1 then throw s!"{prefixMsg}: single-batch input surface required"
        if s.width.toNat != w || s.channels.toNat * s.height.toNat != sN * sC * sH * sD then
          throw s!"{prefixMsg}: input surface ({s.channels}x{s.height}x{s.width}) incompatible with PE reduce ({sN}x{sC}x{sH}x{sD}x{w})"
        pure (some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: scratch src1Base must use BAR slot {barSlotScratch}, got {slot}"
        if isFirst then throw s!"{prefixMsg}: first task cannot read uninitialized DRAM scratch"
        pure (some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: src1Base BAR slot {slot} does not reference an input surface or scratch"

  let resDmaTarget? ←
    if resIsL2 then
      vCheckOmitted task.writes #[
        (regResDep, "resDep"), (regResBaseLo, "resBaseLo"), (regResBaseHi, "resBaseHi"),
        (regResDmaRowStride, "resDmaRowStride"), (regResDmaPlaneStride, "resDmaPlaneStride"),
        (regResDmaNStride, "resDmaNStride"), (regResFmt, "resFmt")
      ] "L2-resident PE reduce res" prefixMsg
      pure none
    else
      if !direct then
        throw s!"{prefixMsg}: DRAM-backed PE reduce requires direct=1 in resCfg"
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regResBaseLo regResBaseHi "resBaseLo" "resBaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .output then
          throw s!"{prefixMsg}: resBase BAR slot {slot} bound to non-output surface '{s.shortName}'"
        if s.n != 1 then throw s!"{prefixMsg}: single-batch output surface required"
        if s.width.toNat != 1 || s.channels.toNat * s.height.toNat != sN * sC then
          throw s!"{prefixMsg}: output surface ({s.channels}x{s.height}x{s.width}) incompatible with PE reduce output ({sN}x{sC}x1)"
        pure (some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: scratch resBase must use BAR slot {barSlotScratch}, got {slot}"
        if isLast then throw s!"{prefixMsg}: final task cannot write to DRAM scratch"
        pure (some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: resBase BAR slot {slot} does not reference an output surface or scratch"

  -- L2 bounds
  let mut l2Accs : Array VerifiedL2Access := #[]
  let wSrc1Base ← requireDecodedWrite task.writes regSrc1Base "src1Base"
  let srcBase := wSrc1Base.value.toNat
  if src1Cfg.isL2 then
    let inCh := 2 * alignUp w 8
    let checkL2Axis := fun (ext : Nat) (addr : Nat) (nm : String) => do
      if ext > 1 then
        let wr ← requireDecodedWrite task.writes addr nm
        let sv := wr.value.toNat
        if sv < inCh || sv % 16 != 0 then
          throw s!"{prefixMsg}: L2 {nm} ({sv}) must be >= {inCh} and multiple of 16"
        pure sv
      else
        if (findDecodedWrite task.writes addr).isSome then
          throw s!"{prefixMsg}: L2 PE reduce with extent=1 must omit {nm} ({toHex addr})"
        pure 0
    let strC ← checkL2Axis sC regSrc1ChanStride "src1ChanStride"
    let strH ← checkL2Axis sH regSrc1RowStride "src1RowStride"
    let strD ← checkL2Axis sD regSrc1RowBytesA "src1RowBytesA"
    let strN ← checkL2Axis sN regSrc1RowBytesB "src1RowBytesB"
    let rawSpan := (sC - 1) * strC + (sH - 1) * strH + (sD - 1) * strD + (sN - 1) * strN + inCh
    let l2Bytes := alignUp rawSpan vDmaLineBytes
    l2Accs := l2Accs.push { lo := srcBase, hi := srcBase + l2Bytes, isWrite := false, role := .src1L2 }
  else
    let wSrc1Chs ← requireDecodedWrite task.writes regSrc1ChanStride "src1ChanStride"
    let wSrc1Row ← requireDecodedWrite task.writes regSrc1RowStride "src1RowStride"
    let wSrc1RowA ← requireDecodedWrite task.writes regSrc1RowBytesA "src1RowBytesA"
    let wSrc1RowB ← requireDecodedWrite task.writes regSrc1RowBytesB "src1RowBytesB"
    let px := min (alignUp w 8) (max stripW 32)
    let expRow := 2 * px
    let expU := sC * sD * expRow
    let expD := if sD > 1 then sC * expRow else expU
    if wSrc1Chs.value.toNat != expRow then
      throw s!"{prefixMsg}: src1ChanStride {wSrc1Chs.value} != expected {expRow}"
    if wSrc1Row.value.toNat != expU then
      throw s!"{prefixMsg}: src1RowStride {wSrc1Row.value} != expected {expU}"
    if wSrc1RowA.value.toNat != expD then
      throw s!"{prefixMsg}: src1RowBytesA {wSrc1RowA.value} != expected {expD}"
    if wSrc1RowB.value.toNat != expU then
      throw s!"{prefixMsg}: src1RowBytesB {wSrc1RowB.value} != expected {expU}"
    let srcBytes := alignUp (th * expU) vDmaLineBytes
    l2Accs := l2Accs.push { lo := srcBase, hi := srcBase + srcBytes, isWrite := true, role := .src1Stage }

  let wResBase ← requireDecodedWrite task.writes regResBase "resBase"
  let resBase := wResBase.value.toNat
  if resIsL2 then
    vCheckOmitted task.writes #[
      (regResRowStride, "resRowStride"), (regResRowBytesA, "resRowBytesA")
    ] "L2-resident PE reduce res" prefixMsg
    if sC > 1 then
      let wRc ← requireDecodedWrite task.writes regResChanStride "resChanStride"
      if wRc.value.toNat != 16 then
        throw s!"{prefixMsg}: L2 resChanStride {wRc.value} != expected 16"
    else
      vCheckOmitted task.writes #[(regResChanStride, "resChanStride")] "L2 PE reduce sC=1" prefixMsg
    if sN > 1 then
      let wRn ← requireDecodedWrite task.writes regResRowBytesB "resRowBytesB"
      if wRn.value.toNat != sC * 16 then
        throw s!"{prefixMsg}: L2 resRowBytesB {wRn.value} != expected {sC * 16}"
    else
      vCheckOmitted task.writes #[(regResRowBytesB, "resRowBytesB")] "L2 PE reduce sN=1" prefixMsg
    let l2Bytes := alignUp (sN * sC * 16) vDmaLineBytes
    l2Accs := l2Accs.push { lo := resBase, hi := resBase + l2Bytes, isWrite := true, role := .resL2 }
  else
    vCheckOmitted task.writes #[
      (regResChanStride, "resChanStride"), (regResRowStride, "resRowStride"),
      (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
    ] "direct PE reduce res" prefixMsg
    l2Accs := l2Accs.push { lo := resBase, hi := resBase + 2048, isWrite := true, role := .resStage }

  vCheckTaskL2Accesses l2Accs prefixMsg

  -- 4D strided DMA verification
  if let some (_, tgt, off) := src1DmaTarget? then
    let (dk, dd) ← vDecodeSrcDep info task.writes regSrc1Dep regSrc1DmaCfg "src1Dep" "src1DmaCfg" taskId isFirst prefixMsg
    sync := { sync with src1DepKind := dk, src1DepDist := dd }
    let wFmt ← requireDecodedWrite task.writes regSrc1Fmt "src1Fmt"
    if wFmt.value != srcFmtWordP1 then
      throw s!"{prefixMsg}: src1Fmt {toHex wFmt.value.toNat} != {toHex srcFmtWordP1.toNat}"
    let (resId, cap, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, "invalid")
    if off % vDmaLineBytes != 0 then
      throw s!"{prefixMsg}: input DMA {lbl} base offset {toHex off} is not {vDmaLineBytes}-byte aligned"
    let rs := alignUp (2 * w) vDmaLineBytes
    let checkDmaAxis := fun (ext : Nat) (addr : Nat) (nm : String) => do
      if ext > 1 then
        let wr ← requireDecodedWrite task.writes addr nm
        let sv := wr.value.toNat
        if sv < rs || sv % vDmaLineBytes != 0 then
          throw s!"{prefixMsg}: input DMA {nm} ({sv}) must be >= {rs} and {vDmaLineBytes}-byte aligned"
        pure sv
      else
        if (findDecodedWrite task.writes addr).isSome then
          throw s!"{prefixMsg}: PE reduce input DMA with extent=1 must omit {nm} ({toHex addr})"
        pure 0
    let dH ← checkDmaAxis sH regSrc1DmaRowStride "src1DmaRowStride"
    let dC ← checkDmaAxis sC regSrc1DmaPlaneStride "src1DmaPlaneStride"
    let dD ← checkDmaAxis sD regSrc1DmaDStride "src1DmaDStride"
    let dN ← checkDmaAxis sN regSrc1DmaNStride "src1DmaNStride"
    let ext := (sH - 1) * dH + (sC - 1) * dC + (sD - 1) * dD + (sN - 1) * dN + rs
    if off + ext > cap then
      throw s!"{prefixMsg}: input DMA range [{toHex off}, {toHex (off + ext)}) exceeds {lbl} capacity {cap}"
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + ext, isWrite := false, port := .src1Dma }

  if let some (_, tgt, off) := resDmaTarget? then
    match ← resolveBarTarget info barSlotConfigZero with
    | .configZero => pure ()
    | _ => throw s!"{prefixMsg}: BAR slot {barSlotConfigZero} must have VM base 0"
    let wResDep ← requireDecodedWrite task.writes regResDep "resDep"
    if wResDep.barSlot != some barSlotConfigZero then
      throw s!"{prefixMsg}: resDep must use BAR slot {barSlotConfigZero}"
    let expCode := match tgt with
      | .scratch _ => resDepCode target ||| resDepScratchFlags
      | _ => resDepCode target
    if (← decodeFieldChecked resDepLayout wResDep.value "code") != expCode then
      throw s!"{prefixMsg}: resDep code mismatch"
    if (← decodeFieldChecked resDepLayout wResDep.value "task").toNat != taskId then
      throw s!"{prefixMsg}: resDep task != header taskId={taskId}"
    if (← decodeFieldChecked resDepLayout wResDep.value "direct") != 1 then
      throw s!"{prefixMsg}: resDep direct bit must be 1 for PE reduce"
    let wResFmt ← requireDecodedWrite task.writes regResFmt "resFmt"
    if wResFmt.value != resFmtWordP1 then
      throw s!"{prefixMsg}: resFmt {toHex wResFmt.value.toNat} != {toHex resFmtWordP1.toNat}"
    let (resId, cap, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, "invalid")
    if off % vDmaLineBytes != 0 then
      throw s!"{prefixMsg}: output DMA {lbl} base offset {toHex off} is not {vDmaLineBytes}-byte aligned"
    vCheckOmitted task.writes #[(regResDmaRowStride, "resDmaRowStride")] "PE reduce output DMA" prefixMsg
    let outC ←
      if sC > 1 then
        let wr ← requireDecodedWrite task.writes regResDmaPlaneStride "resDmaPlaneStride"
        let sv := wr.value.toNat
        if sv < vDmaLineBytes || sv % vDmaLineBytes != 0 then
          throw s!"{prefixMsg}: output DMA resDmaPlaneStride ({sv}) must be >= 64 and 64-byte aligned"
        pure sv
      else
        vCheckOmitted task.writes #[(regResDmaPlaneStride, "resDmaPlaneStride")] "PE reduce output sC=1" prefixMsg
        pure 0
    let outN ←
      if sN > 1 then
        let wr ← requireDecodedWrite task.writes regResDmaNStride "resDmaNStride"
        let sv := wr.value.toNat
        if sv < vDmaLineBytes || sv % vDmaLineBytes != 0 then
          throw s!"{prefixMsg}: output DMA resDmaNStride ({sv}) must be >= 64 and 64-byte aligned"
        pure sv
      else
        vCheckOmitted task.writes #[(regResDmaNStride, "resDmaNStride")] "PE reduce output sN=1" prefixMsg
        pure 0
    let ext := (sC - 1) * outC + (sN - 1) * outN + vDmaLineBytes
    if off + ext > cap then
      throw s!"{prefixMsg}: output DMA range [{toHex off}, {toHex (off + ext)}) exceeds {lbl} capacity {cap}"
    let stdTight := (sC <= 1 || outC == vDmaLineBytes) && (sN <= 1 || outN == sC * vDmaLineBytes)
    let swapTight := (sN <= 1 || outN == vDmaLineBytes) && (sC <= 1 || outC == sN * vDmaLineBytes)
    let outCov : WriteCoverage :=
      if stdTight || swapTight then .dense off (off + sN * sC * vDmaLineBytes)
      else .sparse
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + ext, isWrite := true, port := .resDma, coverage := outCov }

  vVerifyKernelStageMulti task.writes taskId isLast false false prefixMsg
  vVerifyOutputStageMulti task.writes taskId isLast prefixMsg

  let mut allowed : Array Nat := #[
    regWin, regHin, regCin, regDin, regWout, regHout, regCout, regDout, regBatchN,
    regTH, regPeCommon, regPeTile, regTargetCfg,
    regSrc1Cfg, regSrc1Base, regResCfg, regResBase, regPeOp
  ]
  if opVal == 0x0002 || opVal == 0x0003 then allowed := allowed.push regCmpFlags
  if opVal == 0x2010 then allowed := allowed.push regPeEps
  if opVal == 0x0000 || opVal == 0x2010 then allowed := allowed.push regPeRedScale
  if isFirst then allowed := allowed.push regSrc2Cfg
  if src1Cfg.isL2 then
    if sC > 1 then allowed := allowed.push regSrc1ChanStride
    if sH > 1 then allowed := allowed.push regSrc1RowStride
    if sD > 1 then allowed := allowed.push regSrc1RowBytesA
    if sN > 1 then allowed := allowed.push regSrc1RowBytesB
  else
    allowed := allowed ++ #[
      regSrc1ChanStride, regSrc1RowStride, regSrc1RowBytesA, regSrc1RowBytesB,
      regSrc1Dep, regSrc1DmaCfg, regSrc1BaseLo, regSrc1BaseHi, regSrc1Fmt
    ]
    if sH > 1 then allowed := allowed.push regSrc1DmaRowStride
    if sC > 1 then allowed := allowed.push regSrc1DmaPlaneStride
    if sD > 1 then allowed := allowed.push regSrc1DmaDStride
    if sN > 1 then allowed := allowed.push regSrc1DmaNStride
  if resIsL2 then
    if sC > 1 then allowed := allowed.push regResChanStride
    if sN > 1 then allowed := allowed.push regResRowBytesB
  else
    allowed := allowed ++ #[regResDep, regResBaseLo, regResBaseHi, regResFmt]
    if sC > 1 then allowed := allowed.push regResDmaPlaneStride
    if sN > 1 then allowed := allowed.push regResDmaNStride
  if isLast then
    allowed := allowed ++ #[regKernelStage, regOut0, regOut1, regOut2, regOut3, regOut4, regOut5]
  vCheckAllowedRegs task.writes allowed prefixMsg

  pure {
    taskIndex := task.taskIndex
    kind := .peReduce
    engine := .pe
    isNeCompute := false
    isMultiBatch := sN > 1
    liveL2Mask := h7
    l2Accesses := l2Accs
    dramAccesses := dramAccs
    sync := sync
  }

/-- Independent static safety verifier for Weightless NE Window Pooling tasks (`ene in 1..5`, no `__KERN_0`). -/
def verifyNePoolTaskMulti
    (taskCount : Nat) (hasPriorNe hasPriorPe : Bool)
    (info : HwxInfo) (task : DecodedTask) : Except String VerifiedTaskFootprint := do
  let target := info.target
  let prefixMsg := s!"task {task.taskIndex} (NE pool)"
  let isFirst := task.taskIndex == 0
  let isLast := task.taskIndex + 1 == taskCount

  let (taskId, ene, h7) ← vVerifyTaskHeader9 target task taskCount false false prefixMsg
  let logNE := ene - 1
  vCheckNoDuplicateWrites task.writes prefixMsg

  vCheckOmitted task.writes #[
    (regNeCfg, "neCfg"), (regKernelBaseLo, "kernelBaseLo"), (regKernelBaseHi, "kernelBaseHi"),
    (regLutCfg, "lutCfg"), (regLutSize, "lutSize"), (regLutBaseLo, "lutBaseLo"), (regLutBaseHi, "lutBaseHi")
  ] "weightless NE pool" prefixMsg

  let wDw ← requireDecodedWrite task.writes regDwGroups "dwGroups"
  let perGroupIn ← decodeFieldChecked dwGroupsLayout wDw.value "perGroupIn"
  if perGroupIn != 1 then
    throw s!"{prefixMsg}: dwGroups (0x0030) must be 0x00014000 (perGroupIn=1)"

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

  if let some wN := findDecodedWrite task.writes regBatchN then
    if wN.value != 1 then
      throw s!"{prefixMsg}: batch N (0x0024) must be 1, got {wN.value}"

  let winReg := (← decodeFieldChecked (dim16Layout "Win") wWin.value "val").toNat
  let hinReg := (← decodeFieldChecked (dim16Layout "Hin") wHin.value "val").toNat
  let cin := wCin.value.toNat
  let wout := (← decodeFieldChecked (dim16Layout "Wout") wWout.value "val").toNat
  let hout := (← decodeFieldChecked (dim16Layout "Hout") wHout.value "val").toNat
  let cout := wCout.value.toNat
  let th := (← decodeFieldChecked (dim16Layout "TH") wTH.value "val").toNat

  if winReg == 0 || hinReg == 0 || cin == 0 || cin > 65535 || wout == 0 || hout == 0 || cout != cin then
    throw s!"{prefixMsg}: invalid NE pool dims (in={cin}x{hinReg}x{winReg}, out={cout}x{hout}x{wout})"
  if th == 0 || th > hout then
    throw s!"{prefixMsg}: tileHeight={th} out of range 1..{hout}"
  if wTargetCfg.value != targetCfgValue target then
    throw s!"{prefixMsg}: targetCfg {toHex wTargetCfg.value.toNat} != expected {toHex (targetCfgValue target).toNat}"

  let kw := (← decodeFieldChecked convCfgLayout wConvCfg.value "kw").toNat
  let kh := (← decodeFieldChecked convCfgLayout wConvCfg.value "kh").toNat
  let sx := (← decodeFieldChecked convCfgLayout wConvCfg.value "sx").toNat
  let sy := (← decodeFieldChecked convCfgLayout wConvCfg.value "sy").toNat
  let pl := (← decodeFieldChecked convCfgLayout wConvCfg.value "pl").toNat
  let pt := (← decodeFieldChecked convCfgLayout wConvCfg.value "pt").toNat
  let hx := (← decodeFieldChecked convCfgLayout wConvCfg.value "hx").toNat
  let vy := (← decodeFieldChecked convCfgLayout wConvCfg.value "vy").toNat
  if hx != 1 || vy != 1 then
    throw s!"{prefixMsg}: convCfg hx={hx}, vy={vy} must both be 1"
  if kh == 0 || kh > 15 || kw == 0 || kw > 15 then
    throw s!"{prefixMsg}: NE pool kernel ({kh}x{kw}) must be in 1..15"
  if (sx != 1 && sx != 2) || (sy != 1 && sy != 2) then
    throw s!"{prefixMsg}: NE pool strides (sy={sy}, sx={sx}) must be 1 or 2"
  if pl >= kw || pt >= kh then
    throw s!"{prefixMsg}: NE pool padding (pt={pt}, pl={pl}) must be < kernel ({kh}, {kw})"

  let kind := (← decodeFieldChecked neCommonLayout wNeCommon.value "kind").toNat
  let shPref := (← decodeFieldChecked neCommonLayout wNeCommon.value "shPref").toNat
  let shMin := (← decodeFieldChecked neCommonLayout wNeCommon.value "shMin").toNat
  let shMax := (← decodeFieldChecked neCommonLayout wNeCommon.value "shMax").toNat
  let f := (← decodeFieldChecked neCommonLayout wNeCommon.value "F").toNat
  let wide3x3Bit := (← decodeFieldChecked neCommonLayout wNeCommon.value "wide3x3").toNat
  if kind != 0 || wide3x3Bit != 0 then
    throw s!"{prefixMsg}: neCommon kind={kind}, wide3x3={wide3x3Bit} must both be 0"
  if shPref > shMin || shMin > shMax || shMax > 5 then
    throw s!"{prefixMsg}: invalid shifts shPref={shPref}, shMin={shMin}, shMax={shMax}"

  let ocg := (← decodeFieldChecked neTileLayout wNeTile.value "ocg").toNat
  let fat := (← decodeFieldChecked neTileLayout wNeTile.value "fat").toNat
  let wu := (← decodeFieldChecked neTileLayout wNeTile.value "WU").toNat
  if ocg != 3 then
    throw s!"{prefixMsg}: NE pool requires neTile ocg=3, got {ocg}"
  if fat > 1 || wu > 3 || wu + f + fat != logNE then
    throw s!"{prefixMsg}: invalid neTile (WU={wu}, F={f}, fat={fat}, logNE={logNE})"
  let stripW := vStripW fat shPref
  if stripW < 8 || stripW > 256 then
    throw s!"{prefixMsg}: strip width T={stripW} outside [8, 256]"

  -- Verify NE pool op registers and hardware window limits
  let wNeOp ← requireDecodedWrite task.writes regNeOp "neOp"
  let wNeScale ← requireDecodedWrite task.writes regNeScale "neScale"
  let neScaleVal ← decodeFieldChecked neScaleLayout wNeScale.value "val"
  let isMaxPool : Bool ←
    if wNeOp.value == nePoolMaxOpValue then
      let wCmp ← requireDecodedWrite task.writes regCmpFlags "cmpFlags"
      if wCmp.value != cmpFlagMaxValue then
        throw s!"{prefixMsg}: NE max_pool cmpFlags {toHex wCmp.value.toNat} != 0x4"
      vCheckOmitted task.writes #[(regNeActShiftEn, "neActShiftEn")] "NE max_pool" prefixMsg
      if neScaleVal != fp16OneBits then
        throw s!"{prefixMsg}: NE max_pool neScale {toHex neScaleVal.toNat} != fp16 1.0"
      if sx == 1 && sy == 1 && max kh kw >= 9 then
        throw s!"{prefixMsg}: weightless NE stride-1 max_pool unsupported for max(kh, kw)={max kh kw} >= 9"
      pure true
    else if wNeOp.value == nePoolSumOpValue then
      vCheckOmitted task.writes #[(regCmpFlags, "cmpFlags")] "NE avg_pool" prefixMsg
      let wEn ← requireDecodedWrite task.writes regNeActShiftEn "neActShiftEn"
      if (← decodeFieldChecked neActShiftEnLayout wEn.value "enable") != 1 then
        throw s!"{prefixMsg}: NE avg_pool requires 0x0050=1"
      if sx == 1 && sy == 1 && max kh kw >= 9 && max kh kw % 2 == 1 then
        throw s!"{prefixMsg}: weightless NE stride-1 avg_pool unsupported for odd max(kh, kw)={max kh kw} >= 9"
      pure false
    else
      throw s!"{prefixMsg}: illegal NE pool neOp (0x4904) {toHex wNeOp.value.toNat}"

  -- Port configs
  let wSrc1Cfg ← requireDecodedWrite task.writes regSrc1Cfg "src1Cfg"
  let src1Cfg ← vDecodeSrcPortCfg "src1Cfg" wSrc1Cfg.value isFirst hasPriorNe hasPriorPe prefixMsg
  if isFirst then
    let wSrc2Cfg ← requireDecodedWrite task.writes regSrc2Cfg "src2Cfg"
    if wSrc2Cfg.value != src2CfgValue then
      throw s!"{prefixMsg}: src2Cfg {toHex wSrc2Cfg.value.toNat} != {toHex src2CfgValue.toNat}"
  else
    vCheckOmitted task.writes #[(regSrc2Cfg, "src2Cfg")] "NE pool taskId > 0" prefixMsg

  let wResCfg ← requireDecodedWrite task.writes regResCfg "resCfg"
  let (resIsL2, direct) ← vDecodeResPortCfg wResCfg.value isFirst isLast prefixMsg

  let mut dramAccs : Array VerifiedDramAccess := #[]
  let mut sync : TaskSyncInfo := {
    src1WaitNe5 := src1Cfg.waitNe5
    src1WaitPe22 := src1Cfg.waitPe22
    src1HandoffNe := src1Cfg.handoffNe
    src1HandoffPe := src1Cfg.handoffPe
    hasSrc1Dma := !src1Cfg.isL2
  }

  let (fullH, fullW, hasExactFull, src1DmaTarget?) ←
    if src1Cfg.isL2 then
      vCheckOmitted task.writes #[
        (regSrc1Dep, "src1Dep"), (regSrc1DmaCfg, "src1DmaCfg"),
        (regSrc1BaseLo, "src1BaseLo"), (regSrc1BaseHi, "src1BaseHi"),
        (regSrc1DmaRowStride, "src1DmaRowStride"), (regSrc1DmaPlaneStride, "src1DmaPlaneStride"),
        (regSrc1Fmt, "src1Fmt")
      ] "L2-resident NE pool src1" prefixMsg
      pure (hinReg, winReg, false, none)
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regSrc1BaseLo regSrc1BaseHi "src1BaseLo" "src1BaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .input then
          throw s!"{prefixMsg}: src1Base BAR slot {slot} bound to non-input surface '{s.shortName}'"
        if s.n != 1 then throw s!"{prefixMsg}: single-batch input surface required"
        if s.channels.toNat != cin then
          throw s!"{prefixMsg}: Cin={cin} != input surface channels {s.channels}"
        pure (s.height.toNat, s.width.toNat, true, some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: scratch src1Base must use BAR slot {barSlotScratch}, got {slot}"
        if isFirst then throw s!"{prefixMsg}: first task cannot read uninitialized DRAM scratch"
        pure (hinReg, winReg, false, some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: src1Base BAR slot {slot} does not reference an input surface or scratch"

  vVerifyPoolAxisGeom fullH kh sy pt hinReg hout hasExactFull "H" prefixMsg
  vVerifyPoolAxisGeom fullW kw sx pl winReg wout hasExactFull "W" prefixMsg

  let resDmaTarget? ←
    if resIsL2 then
      vCheckOmitted task.writes #[
        (regResDep, "resDep"), (regResBaseLo, "resBaseLo"), (regResBaseHi, "resBaseHi"),
        (regResDmaRowStride, "resDmaRowStride"), (regResDmaPlaneStride, "resDmaPlaneStride"),
        (regResFmt, "resFmt")
      ] "L2-resident NE pool res" prefixMsg
      pure none
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regResBaseLo regResBaseHi "resBaseLo" "resBaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .output then
          throw s!"{prefixMsg}: resBase BAR slot {slot} bound to non-output surface '{s.shortName}'"
        if s.n != 1 then throw s!"{prefixMsg}: single-batch output surface required"
        if s.channels.toNat != cout || s.height.toNat != hout || s.width.toNat != wout then
          throw s!"{prefixMsg}: output surface ({s.channels}x{s.height}x{s.width}) != NE pool output ({cout}x{hout}x{wout})"
        pure (some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: scratch resBase must use BAR slot {barSlotScratch}, got {slot}"
        if isLast then throw s!"{prefixMsg}: final task cannot write to DRAM scratch"
        pure (some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: resBase BAR slot {slot} does not reference an output surface or scratch"

  -- L2 bounds
  let mut l2Accs : Array VerifiedL2Access := #[]
  let wSrc1Base ← requireDecodedWrite task.writes regSrc1Base "src1Base"
  let wSrc1Chs ← requireDecodedWrite task.writes regSrc1ChanStride "src1ChanStride"
  let wSrc1Row ← requireDecodedWrite task.writes regSrc1RowStride "src1RowStride"
  let srcBase := wSrc1Base.value.toNat
  if src1Cfg.isL2 then
    vCheckOmitted task.writes #[
      (regSrc1RowBytesA, "src1RowBytesA"), (regSrc1RowBytesB, "src1RowBytesB")
    ] "L2-resident NE pool src1" prefixMsg
    let minChs := 2 * alignUp winReg 8
    let gotChs := wSrc1Chs.value.toNat
    if gotChs < minChs || gotChs % 16 != 0 then
      throw s!"{prefixMsg}: L2 src1ChanStride {gotChs} invalid (must be >= {minChs} and multiple of 16)"
    let expRow := cin * gotChs
    if wSrc1Row.value.toNat != expRow then
      throw s!"{prefixMsg}: L2 src1RowStride {wSrc1Row.value} != expected {expRow}"
    let l2Bytes := alignUp (hinReg * expRow) vDmaLineBytes
    l2Accs := l2Accs.push { lo := srcBase, hi := srcBase + l2Bytes, isWrite := false, role := .src1L2 }
  else
    let wSrc1RowA ← requireDecodedWrite task.writes regSrc1RowBytesA "src1RowBytesA"
    let wSrc1RowB ← requireDecodedWrite task.writes regSrc1RowBytesB "src1RowBytesB"
    let expPx := vPoolStripPx winReg kw sx pl stripW wout
    let expRow := 2 * expPx
    if wSrc1Chs.value.toNat != expRow then
      throw s!"{prefixMsg}: src1ChanStride {wSrc1Chs.value} != expected {expRow}"
    let minU := cin * expRow
    let src1Row := wSrc1Row.value.toNat
    if src1Row < minU || (src1Row - minU) % 16 != 0 then
      throw s!"{prefixMsg}: src1RowStride {src1Row} incompatible with C*row={minU}"
    if wSrc1RowA.value.toNat != minU || wSrc1RowB.value.toNat != minU then
      throw s!"{prefixMsg}: src1RowBytesA/B mismatch"
    let inRows := vPoolInRows hinReg kh sy pt hout th true
    let srcBytes := alignUp (inRows * src1Row) vDmaLineBytes
    l2Accs := l2Accs.push { lo := srcBase, hi := srcBase + srcBytes, isWrite := true, role := .src1Stage }

  let wResBase ← requireDecodedWrite task.writes regResBase "resBase"
  let resBase := wResBase.value.toNat
  if resIsL2 then
    vCheckOmitted task.writes #[
      (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
    ] "L2-resident NE pool res" prefixMsg
    let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
    let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
    let (expChs, expRow, l2Bytes) := vL2ResidentGeometry cout hout wout
    if wResChs.value.toNat != expChs then
      throw s!"{prefixMsg}: L2 resChanStride {wResChs.value} != expected {expChs}"
    if wResRow.value.toNat != expRow then
      throw s!"{prefixMsg}: L2 resRowStride {wResRow.value} != expected {expRow}"
    l2Accs := l2Accs.push { lo := resBase, hi := resBase + l2Bytes, isWrite := true, role := .resL2 }
  else
    let expDirect := vIsDirect target fat shMax wout
    if direct != expDirect then
      throw s!"{prefixMsg}: resCfg direct={direct} disagrees with target/tiling direct={expDirect}"
    if direct then
      vCheckOmitted task.writes #[
        (regResChanStride, "resChanStride"), (regResRowStride, "resRowStride"),
        (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
      ] "direct mode" prefixMsg
      let resDirectBytes := vNeActDirectResBytes f fat
      l2Accs := l2Accs.push { lo := resBase, hi := resBase + resDirectBytes, isWrite := true, role := .resStage }
    else
      let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
      let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
      let wResRowA ← requireDecodedWrite task.writes regResRowBytesA "resRowBytesA"
      let wResRowB ← requireDecodedWrite task.writes regResRowBytesB "resRowBytesB"
      let expDrow := vDchs wout stripW
      if wResChs.value.toNat != expDrow then
        throw s!"{prefixMsg}: staged resChanStride {wResChs.value} != expected {expDrow}"
      let minDu := cout * expDrow
      let resRow := wResRow.value.toNat
      if resRow < minDu || (resRow - minDu) % 16 != 0 then
        throw s!"{prefixMsg}: staged resRowStride {resRow} incompatible with C*drow={minDu}"
      if wResRowA.value.toNat != minDu || wResRowB.value.toNat != minDu then
        throw s!"{prefixMsg}: staged resRowBytesA/B mismatch"
      l2Accs := l2Accs.push { lo := resBase, hi := resBase + alignUp (th * resRow) vDmaLineBytes, isWrite := true, role := .resStage }

  vCheckTaskL2Accesses l2Accs prefixMsg

  -- DMA bounds
  if let some (_, tgt, off) := src1DmaTarget? then
    let (dk, dd) ← vDecodeSrcDep info task.writes regSrc1Dep regSrc1DmaCfg "src1Dep" "src1DmaCfg" taskId isFirst prefixMsg
    sync := { sync with src1DepKind := dk, src1DepDist := dd }
    let wFmt ← requireDecodedWrite task.writes regSrc1Fmt "src1Fmt"
    if wFmt.value != srcFmtWordP1 then
      throw s!"{prefixMsg}: src1Fmt mismatch"
    let (resId, cap, isSurf, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, true, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, false, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, false, "invalid")
    let (ext, _) ← vVerifyDmaStridesAndBounds task.writes cin fullH fullW off cap isSurf
      regSrc1DmaRowStride regSrc1DmaPlaneStride "src1DmaRowStride" "src1DmaPlaneStride" "input DMA" lbl prefixMsg
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + ext, isWrite := false, port := .src1Dma }

  if let some (_, tgt, off) := resDmaTarget? then
    match ← resolveBarTarget info barSlotConfigZero with
    | .configZero => pure ()
    | _ => throw s!"{prefixMsg}: BAR slot {barSlotConfigZero} must have VM base 0"
    let wResDep ← requireDecodedWrite task.writes regResDep "resDep"
    if wResDep.barSlot != some barSlotConfigZero then
      throw s!"{prefixMsg}: resDep must use BAR slot {barSlotConfigZero}"
    let expCode := match tgt with
      | .scratch _ => resDepCode target ||| resDepScratchFlags
      | _ => resDepCode target
    if (← decodeFieldChecked resDepLayout wResDep.value "code") != expCode then
      throw s!"{prefixMsg}: resDep code mismatch"
    if (← decodeFieldChecked resDepLayout wResDep.value "task").toNat != taskId then
      throw s!"{prefixMsg}: resDep task != header taskId={taskId}"
    if ((← decodeFieldChecked resDepLayout wResDep.value "direct") == 1) != direct then
      throw s!"{prefixMsg}: resDep direct bit != resCfg direct={direct}"
    let wResFmt ← requireDecodedWrite task.writes regResFmt "resFmt"
    if wResFmt.value != resFmtWordP1 then
      throw s!"{prefixMsg}: resFmt mismatch"
    let (resId, cap, isSurf, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, true, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, false, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, false, "invalid")
    let (ext, outCov) ← vVerifyDmaStridesAndBounds task.writes cout hout wout off cap isSurf
      regResDmaRowStride regResDmaPlaneStride "resDmaRowStride" "resDmaPlaneStride" "output DMA" lbl prefixMsg
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + ext, isWrite := true, port := .resDma, coverage := outCov }

  vVerifyKernelStageMulti task.writes taskId isLast false false prefixMsg
  vVerifyOutputStageMulti task.writes taskId isLast prefixMsg

  let mut allowed : Array Nat := #[
    regWin, regHin, regCin, regWout, regHout, regCout, regBatchN,
    regConvCfg, regDwGroups, regTH, regNeCommon, regNeTile, regTargetCfg,
    regSrc1Cfg, regSrc1Base, regSrc1ChanStride, regSrc1RowStride,
    regResCfg, regResBase, regNeOp, regNeScale
  ]
  if isMaxPool then allowed := allowed.push regCmpFlags
  else allowed := allowed.push regNeActShiftEn
  if isFirst then allowed := allowed.push regSrc2Cfg
  if !src1Cfg.isL2 then
    allowed := allowed ++ #[
      regSrc1RowBytesA, regSrc1RowBytesB,
      regSrc1Dep, regSrc1DmaCfg, regSrc1BaseLo, regSrc1BaseHi,
      regSrc1DmaRowStride, regSrc1DmaPlaneStride, regSrc1Fmt
    ]
  if resIsL2 then
    allowed := allowed ++ #[regResChanStride, regResRowStride]
  else
    allowed := allowed ++ #[
      regResDep, regResBaseLo, regResBaseHi,
      regResDmaRowStride, regResDmaPlaneStride, regResFmt
    ]
    if !direct then
      allowed := allowed ++ #[
        regResChanStride, regResRowStride, regResRowBytesA, regResRowBytesB
      ]
  if isLast then
    allowed := allowed ++ #[regKernelStage, regOut0, regOut1, regOut2, regOut3, regOut4, regOut5]
  vCheckAllowedRegs task.writes allowed prefixMsg

  pure {
    taskIndex := task.taskIndex
    kind := .nePool
    engine := .ne
    isNeCompute := false
    liveL2Mask := h7
    l2Accesses := l2Accs
    dramAccesses := dramAccs
    sync := sync
  }

/-- Independent static safety verifier for Weightless NE Channel Reduction tasks (`Cout = 1`, `0x0030 = Cin << 16`). -/
def verifyNeChannelReduceTaskMulti
    (taskCount : Nat) (hasPriorNe hasPriorPe : Bool)
    (info : HwxInfo) (task : DecodedTask) : Except String VerifiedTaskFootprint := do
  let target := info.target
  let prefixMsg := s!"task {task.taskIndex} (NE channel reduce)"
  let isFirst := task.taskIndex == 0
  let isLast := task.taskIndex + 1 == taskCount

  let (taskId, ene, h7) ← vVerifyTaskHeader9 target task taskCount false false prefixMsg
  let logNE := ene - 1
  vCheckNoDuplicateWrites task.writes prefixMsg

  vCheckOmitted task.writes #[
    (regConvCfg, "convCfg"), (regNeCfg, "neCfg"),
    (regKernelBaseLo, "kernelBaseLo"), (regKernelBaseHi, "kernelBaseHi"),
    (regLutCfg, "lutCfg"), (regLutSize, "lutSize"), (regLutBaseLo, "lutBaseLo"), (regLutBaseHi, "lutBaseHi")
  ] "weightless NE channel reduce" prefixMsg

  let wWin ← requireDecodedWrite task.writes regWin "Win"
  let wHin ← requireDecodedWrite task.writes regHin "Hin"
  let wCin ← requireDecodedWrite task.writes regCin "Cin"
  let wWout ← requireDecodedWrite task.writes regWout "Wout"
  let wHout ← requireDecodedWrite task.writes regHout "Hout"
  let wCout ← requireDecodedWrite task.writes regCout "Cout"
  let wDw ← requireDecodedWrite task.writes regDwGroups "dwGroups"
  let wTH ← requireDecodedWrite task.writes regTH "TH"
  let wNeCommon ← requireDecodedWrite task.writes regNeCommon "neCommon"
  let wNeTile ← requireDecodedWrite task.writes regNeTile "neTile"
  let wTargetCfg ← requireDecodedWrite task.writes regTargetCfg "targetCfg"

  if let some wN := findDecodedWrite task.writes regBatchN then
    if wN.value != 1 then
      throw s!"{prefixMsg}: batch N (0x0024) must be 1, got {wN.value}"

  let w := (← decodeFieldChecked (dim16Layout "Win") wWin.value "val").toNat
  let h := (← decodeFieldChecked (dim16Layout "Hin") wHin.value "val").toNat
  let cin := wCin.value.toNat
  let wout := (← decodeFieldChecked (dim16Layout "Wout") wWout.value "val").toNat
  let hout := (← decodeFieldChecked (dim16Layout "Hout") wHout.value "val").toNat
  let cout := wCout.value.toNat
  let th := (← decodeFieldChecked (dim16Layout "TH") wTH.value "val").toNat

  if w == 0 || h == 0 || cin == 0 || cin > 65535 then
    throw s!"{prefixMsg}: invalid NE channel reduce dims (C={cin}, H={h}, W={w})"
  if cout != 1 then
    throw s!"{prefixMsg}: NE channel reduce requires Cout=1, got Cout={cout}"
  if wout != w || hout != h then
    throw s!"{prefixMsg}: output spatial dims ({hout}x{wout}) != input ({h}x{w})"
  let chanCount := (← decodeFieldChecked neChanReduceGroupsLayout wDw.value "chanCount").toNat
  if chanCount != cin then
    throw s!"{prefixMsg}: dwGroups (0x0030) chanCount={chanCount} != Cin={cin}"
  if th == 0 || th > h then
    throw s!"{prefixMsg}: tileHeight={th} out of range 1..{h}"
  if wTargetCfg.value != targetCfgValue target then
    throw s!"{prefixMsg}: targetCfg {toHex wTargetCfg.value.toNat} != expected {toHex (targetCfgValue target).toNat}"

  let kind := (← decodeFieldChecked neCommonLayout wNeCommon.value "kind").toNat
  let shPref := (← decodeFieldChecked neCommonLayout wNeCommon.value "shPref").toNat
  let shMin := (← decodeFieldChecked neCommonLayout wNeCommon.value "shMin").toNat
  let shMax := (← decodeFieldChecked neCommonLayout wNeCommon.value "shMax").toNat
  let f := (← decodeFieldChecked neCommonLayout wNeCommon.value "F").toNat
  let wide3x3Bit := (← decodeFieldChecked neCommonLayout wNeCommon.value "wide3x3").toNat
  if kind != 0 || wide3x3Bit != 0 then
    throw s!"{prefixMsg}: neCommon kind={kind}, wide3x3={wide3x3Bit} must both be 0"
  if shPref > shMin || shMin > shMax || shMax > 5 then
    throw s!"{prefixMsg}: invalid shifts shPref={shPref}, shMin={shMin}, shMax={shMax}"

  let ocg := (← decodeFieldChecked neTileLayout wNeTile.value "ocg").toNat
  let fat := (← decodeFieldChecked neTileLayout wNeTile.value "fat").toNat
  let wu := (← decodeFieldChecked neTileLayout wNeTile.value "WU").toNat
  if ocg != 3 then
    throw s!"{prefixMsg}: NE channel reduce requires neTile ocg=3, got {ocg}"
  if fat > 1 || wu > 3 || wu + f + fat != logNE then
    throw s!"{prefixMsg}: invalid neTile (WU={wu}, F={f}, fat={fat}, logNE={logNE})"
  let stripW := vStripW fat shPref
  if stripW < 8 || stripW > 256 then
    throw s!"{prefixMsg}: strip width T={stripW} outside [8, 256]"

  -- Verify allowed (0x0050, 0x4100, 0x4904, 0x4910) tuples
  let wNeOp ← requireDecodedWrite task.writes regNeOp "neOp"
  let wNeScale ← requireDecodedWrite task.writes regNeScale "neScale"
  let neScaleVal ← decodeFieldChecked neScaleLayout wNeScale.value "val"
  let usesCmp : Bool ←
    if wNeOp.value == nePoolMaxOpValue then
      let wCmp ← requireDecodedWrite task.writes regCmpFlags "cmpFlags"
      if wCmp.value != cmpFlagMaxValue then
        throw s!"{prefixMsg}: NE channel reduce max cmpFlags {toHex wCmp.value.toNat} != 0x4"
      vCheckOmitted task.writes #[(regNeActShiftEn, "neActShiftEn")] "NE channel reduce max" prefixMsg
      if neScaleVal != fp16OneBits then
        throw s!"{prefixMsg}: NE channel reduce max neScale {toHex neScaleVal.toNat} != fp16 1.0"
      pure true
    else if wNeOp.value == nePoolMinOpValue then
      let wCmp ← requireDecodedWrite task.writes regCmpFlags "cmpFlags"
      if wCmp.value != cmpFlagMinValue then
        throw s!"{prefixMsg}: NE channel reduce min cmpFlags {toHex wCmp.value.toNat} != 0xc"
      vCheckOmitted task.writes #[(regNeActShiftEn, "neActShiftEn")] "NE channel reduce min" prefixMsg
      if neScaleVal != fp16OneBits then
        throw s!"{prefixMsg}: NE channel reduce min neScale {toHex neScaleVal.toNat} != fp16 1.0"
      pure true
    else if wNeOp.value == nePoolSumOpValue then
      vCheckOmitted task.writes #[(regCmpFlags, "cmpFlags")] "NE channel reduce sum/mean" prefixMsg
      let wEn ← requireDecodedWrite task.writes regNeActShiftEn "neActShiftEn"
      if (← decodeFieldChecked neActShiftEnLayout wEn.value "enable") != 1 then
        throw s!"{prefixMsg}: NE channel reduce sum/mean requires 0x0050=1"
      pure false
    else
      throw s!"{prefixMsg}: unknown NE channel reduce neOp (0x4904) {toHex wNeOp.value.toNat}"

  -- Port configs
  let wSrc1Cfg ← requireDecodedWrite task.writes regSrc1Cfg "src1Cfg"
  let src1Cfg ← vDecodeSrcPortCfg "src1Cfg" wSrc1Cfg.value isFirst hasPriorNe hasPriorPe prefixMsg
  if isFirst then
    let wSrc2Cfg ← requireDecodedWrite task.writes regSrc2Cfg "src2Cfg"
    if wSrc2Cfg.value != src2CfgValue then
      throw s!"{prefixMsg}: src2Cfg {toHex wSrc2Cfg.value.toNat} != {toHex src2CfgValue.toNat}"
  else
    vCheckOmitted task.writes #[(regSrc2Cfg, "src2Cfg")] "NE channel reduce taskId > 0" prefixMsg

  let wResCfg ← requireDecodedWrite task.writes regResCfg "resCfg"
  let (resIsL2, direct) ← vDecodeResPortCfg wResCfg.value isFirst isLast prefixMsg

  let mut dramAccs : Array VerifiedDramAccess := #[]
  let mut sync : TaskSyncInfo := {
    src1WaitNe5 := src1Cfg.waitNe5
    src1WaitPe22 := src1Cfg.waitPe22
    src1HandoffNe := src1Cfg.handoffNe
    src1HandoffPe := src1Cfg.handoffPe
    hasSrc1Dma := !src1Cfg.isL2
  }

  let src1DmaTarget? ←
    if src1Cfg.isL2 then
      vCheckOmitted task.writes #[
        (regSrc1Dep, "src1Dep"), (regSrc1DmaCfg, "src1DmaCfg"),
        (regSrc1BaseLo, "src1BaseLo"), (regSrc1BaseHi, "src1BaseHi"),
        (regSrc1DmaRowStride, "src1DmaRowStride"), (regSrc1DmaPlaneStride, "src1DmaPlaneStride"),
        (regSrc1Fmt, "src1Fmt")
      ] "L2-resident NE channel reduce src1" prefixMsg
      pure none
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regSrc1BaseLo regSrc1BaseHi "src1BaseLo" "src1BaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .input then
          throw s!"{prefixMsg}: src1Base BAR slot {slot} bound to non-input surface '{s.shortName}'"
        if s.n != 1 then throw s!"{prefixMsg}: single-batch input surface required"
        if s.channels.toNat != cin || s.height.toNat != h || s.width.toNat != w then
          throw s!"{prefixMsg}: input surface ({s.channels}x{s.height}x{s.width}) != ({cin}x{h}x{w})"
        pure (some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: scratch src1Base must use BAR slot {barSlotScratch}, got {slot}"
        if isFirst then throw s!"{prefixMsg}: first task cannot read uninitialized DRAM scratch"
        pure (some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: src1Base BAR slot {slot} does not reference an input surface or scratch"

  let resDmaTarget? ←
    if resIsL2 then
      vCheckOmitted task.writes #[
        (regResDep, "resDep"), (regResBaseLo, "resBaseLo"), (regResBaseHi, "resBaseHi"),
        (regResDmaRowStride, "resDmaRowStride"), (regResDmaPlaneStride, "resDmaPlaneStride"),
        (regResFmt, "resFmt")
      ] "L2-resident NE channel reduce res" prefixMsg
      pure none
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regResBaseLo regResBaseHi "resBaseLo" "resBaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .output then
          throw s!"{prefixMsg}: resBase BAR slot {slot} bound to non-output surface '{s.shortName}'"
        if s.n != 1 then throw s!"{prefixMsg}: single-batch output surface required"
        if s.channels.toNat != 1 || s.height.toNat != h || s.width.toNat != w then
          throw s!"{prefixMsg}: output surface ({s.channels}x{s.height}x{s.width}) != (1x{h}x{w})"
        pure (some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: scratch resBase must use BAR slot {barSlotScratch}, got {slot}"
        if isLast then throw s!"{prefixMsg}: final task cannot write to DRAM scratch"
        pure (some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: resBase BAR slot {slot} does not reference an output surface or scratch"

  -- L2 bounds
  let mut l2Accs : Array VerifiedL2Access := #[]
  let wSrc1Base ← requireDecodedWrite task.writes regSrc1Base "src1Base"
  let wSrc1Chs ← requireDecodedWrite task.writes regSrc1ChanStride "src1ChanStride"
  let wSrc1Row ← requireDecodedWrite task.writes regSrc1RowStride "src1RowStride"
  let srcBase := wSrc1Base.value.toNat
  if src1Cfg.isL2 then
    vCheckOmitted task.writes #[
      (regSrc1RowBytesA, "src1RowBytesA"), (regSrc1RowBytesB, "src1RowBytesB")
    ] "L2-resident NE channel reduce src1" prefixMsg
    let (expChs, expRow, l2Bytes) := vL2ResidentGeometry cin h w
    if wSrc1Chs.value.toNat != expChs then
      throw s!"{prefixMsg}: L2 src1ChanStride {wSrc1Chs.value} != expected {expChs}"
    if wSrc1Row.value.toNat != expRow then
      throw s!"{prefixMsg}: L2 src1RowStride {wSrc1Row.value} != expected {expRow}"
    l2Accs := l2Accs.push { lo := srcBase, hi := srcBase + l2Bytes, isWrite := false, role := .src1L2 }
  else
    let wSrc1RowA ← requireDecodedWrite task.writes regSrc1RowBytesA "src1RowBytesA"
    let wSrc1RowB ← requireDecodedWrite task.writes regSrc1RowBytesB "src1RowBytesB"
    let expRow := 2 * min (alignUp w 8) stripW
    if wSrc1Chs.value.toNat != expRow then
      throw s!"{prefixMsg}: src1ChanStride {wSrc1Chs.value} != expected {expRow}"
    let minU := cin * expRow
    let src1Row := wSrc1Row.value.toNat
    if src1Row < minU || (src1Row - minU) % 16 != 0 then
      throw s!"{prefixMsg}: src1RowStride {src1Row} incompatible with C*row={minU}"
    if wSrc1RowA.value.toNat != minU || wSrc1RowB.value.toNat != minU then
      throw s!"{prefixMsg}: src1RowBytesA/B mismatch"
    let srcBytes := alignUp (th * src1Row) vDmaLineBytes
    l2Accs := l2Accs.push { lo := srcBase, hi := srcBase + srcBytes, isWrite := true, role := .src1Stage }

  let wResBase ← requireDecodedWrite task.writes regResBase "resBase"
  let resBase := wResBase.value.toNat
  if resIsL2 then
    vCheckOmitted task.writes #[
      (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
    ] "L2-resident NE channel reduce res" prefixMsg
    let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
    let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
    let (expChs, expRow, l2Bytes) := vL2ResidentGeometry 1 h w
    if wResChs.value.toNat != expChs then
      throw s!"{prefixMsg}: L2 resChanStride {wResChs.value} != expected {expChs}"
    if wResRow.value.toNat != expRow then
      throw s!"{prefixMsg}: L2 resRowStride {wResRow.value} != expected {expRow}"
    l2Accs := l2Accs.push { lo := resBase, hi := resBase + l2Bytes, isWrite := true, role := .resL2 }
  else
    let expDirect := vIsDirect target fat shMax w
    if direct != expDirect then
      throw s!"{prefixMsg}: resCfg direct={direct} disagrees with target/tiling direct={expDirect}"
    if direct then
      vCheckOmitted task.writes #[
        (regResChanStride, "resChanStride"), (regResRowStride, "resRowStride"),
        (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
      ] "direct mode" prefixMsg
      let resDirectBytes := vNeActDirectResBytes f fat
      l2Accs := l2Accs.push { lo := resBase, hi := resBase + resDirectBytes, isWrite := true, role := .resStage }
    else
      let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
      let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
      let wResRowA ← requireDecodedWrite task.writes regResRowBytesA "resRowBytesA"
      let wResRowB ← requireDecodedWrite task.writes regResRowBytesB "resRowBytesB"
      let expDrow := vDchs w stripW
      if wResChs.value.toNat != expDrow then
        throw s!"{prefixMsg}: staged resChanStride {wResChs.value} != expected {expDrow}"
      let resRow := wResRow.value.toNat
      if resRow < expDrow || (resRow - expDrow) % 16 != 0 then
        throw s!"{prefixMsg}: staged resRowStride {resRow} incompatible with drow={expDrow}"
      if wResRowA.value.toNat != expDrow || wResRowB.value.toNat != expDrow then
        throw s!"{prefixMsg}: staged resRowBytesA/B mismatch"
      l2Accs := l2Accs.push { lo := resBase, hi := resBase + alignUp (th * resRow) vDmaLineBytes, isWrite := true, role := .resStage }

  vCheckTaskL2Accesses l2Accs prefixMsg

  -- DMA bounds
  if let some (_, tgt, off) := src1DmaTarget? then
    let (dk, dd) ← vDecodeSrcDep info task.writes regSrc1Dep regSrc1DmaCfg "src1Dep" "src1DmaCfg" taskId isFirst prefixMsg
    sync := { sync with src1DepKind := dk, src1DepDist := dd }
    let wFmt ← requireDecodedWrite task.writes regSrc1Fmt "src1Fmt"
    if wFmt.value != srcFmtWordP1 then
      throw s!"{prefixMsg}: src1Fmt mismatch"
    let (resId, cap, isSurf, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, true, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, false, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, false, "invalid")
    let (ext, _) ← vVerifyDmaStridesAndBounds task.writes cin h w off cap isSurf
      regSrc1DmaRowStride regSrc1DmaPlaneStride "src1DmaRowStride" "src1DmaPlaneStride" "input DMA" lbl prefixMsg
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + ext, isWrite := false, port := .src1Dma }

  if let some (_, tgt, off) := resDmaTarget? then
    match ← resolveBarTarget info barSlotConfigZero with
    | .configZero => pure ()
    | _ => throw s!"{prefixMsg}: BAR slot {barSlotConfigZero} must have VM base 0"
    let wResDep ← requireDecodedWrite task.writes regResDep "resDep"
    if wResDep.barSlot != some barSlotConfigZero then
      throw s!"{prefixMsg}: resDep must use BAR slot {barSlotConfigZero}"
    let expCode := match tgt with
      | .scratch _ => resDepCode target ||| resDepScratchFlags
      | _ => resDepCode target
    if (← decodeFieldChecked resDepLayout wResDep.value "code") != expCode then
      throw s!"{prefixMsg}: resDep code mismatch"
    if (← decodeFieldChecked resDepLayout wResDep.value "task").toNat != taskId then
      throw s!"{prefixMsg}: resDep task != header taskId={taskId}"
    if ((← decodeFieldChecked resDepLayout wResDep.value "direct") == 1) != direct then
      throw s!"{prefixMsg}: resDep direct bit != resCfg direct={direct}"
    let wResFmt ← requireDecodedWrite task.writes regResFmt "resFmt"
    if wResFmt.value != resFmtWordP1 then
      throw s!"{prefixMsg}: resFmt mismatch"
    let (resId, cap, isSurf, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, true, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, false, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, false, "invalid")
    let (ext, outCov) ← vVerifyDmaStridesAndBounds task.writes 1 h w off cap isSurf
      regResDmaRowStride regResDmaPlaneStride "resDmaRowStride" "resDmaPlaneStride" "output DMA" lbl prefixMsg
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + ext, isWrite := true, port := .resDma, coverage := outCov }

  vVerifyKernelStageMulti task.writes taskId isLast false false prefixMsg
  vVerifyOutputStageMulti task.writes taskId isLast prefixMsg

  let mut allowed : Array Nat := #[
    regWin, regHin, regCin, regWout, regHout, regCout, regBatchN,
    regDwGroups, regTH, regNeCommon, regNeTile, regTargetCfg,
    regSrc1Cfg, regSrc1Base, regSrc1ChanStride, regSrc1RowStride,
    regResCfg, regResBase, regNeOp, regNeScale
  ]
  if usesCmp then allowed := allowed.push regCmpFlags
  else allowed := allowed.push regNeActShiftEn
  if isFirst then allowed := allowed.push regSrc2Cfg
  if !src1Cfg.isL2 then
    allowed := allowed ++ #[
      regSrc1RowBytesA, regSrc1RowBytesB,
      regSrc1Dep, regSrc1DmaCfg, regSrc1BaseLo, regSrc1BaseHi, regSrc1Fmt
    ]
    if h > 1 then allowed := allowed.push regSrc1DmaRowStride
    if cin > 1 then allowed := allowed.push regSrc1DmaPlaneStride
  if resIsL2 then
    allowed := allowed ++ #[regResChanStride, regResRowStride]
  else
    allowed := allowed ++ #[regResDep, regResBaseLo, regResBaseHi, regResFmt]
    if h > 1 then allowed := allowed.push regResDmaRowStride
    if !direct then
      allowed := allowed ++ #[
        regResChanStride, regResRowStride, regResRowBytesA, regResRowBytesB
      ]
  if isLast then
    allowed := allowed ++ #[regKernelStage, regOut0, regOut1, regOut2, regOut3, regOut4, regOut5]
  vCheckAllowedRegs task.writes allowed prefixMsg

  pure {
    taskIndex := task.taskIndex
    kind := .neChannelReduce
    engine := .ne
    isNeCompute := false
    liveL2Mask := h7
    l2Accesses := l2Accs
    dramAccesses := dramAccs
    sync := sync
  }

/-- Independent static safety verifier for Weightless NE Copy / View tasks (`neView`:
    `transpose`, `slice`, `concat`, `split`, `reshape`).
    Enforces strict `ane_hx_dma_modelled` compliance (forbidding `0x4d84`, `0x5150`, `0x5104`),
    64-byte alignment, exact `ane_hx_dma_tile` bounds on both input and output BAR slots,
    `groupCode`/`ocg`/`transposeOut`/`transposeWideGroup` consistency, and `<= 2 MiB` L2 bounds. -/
def verifyNeViewTaskMulti
    (taskCount : Nat) (hasPriorNe hasPriorPe : Bool)
    (info : HwxInfo) (task : DecodedTask) : Except String VerifiedTaskFootprint := do
  let target := info.target
  let prefixMsg := s!"task {task.taskIndex} (NE view)"
  let isFirst := task.taskIndex == 0
  let isLast := task.taskIndex + 1 == taskCount

  -- 1. Verify 9-word header, no duplicate writes, no unmodelled DMA registers
  let (taskId, ene, h7) ← vVerifyTaskHeader9 target task taskCount false false prefixMsg
  let logNE := ene - 1
  vCheckNoDuplicateWrites task.writes prefixMsg
  vCheckNoUnmodelledDmaRegs task.writes prefixMsg

  vCheckOmitted task.writes #[
    (regConvCfg, "convCfg"), (regNeCfg, "neCfg"),
    (regKernelBaseLo, "kernelBaseLo"), (regKernelBaseHi, "kernelBaseHi"),
    (regLutCfg, "lutCfg"), (regLutSize, "lutSize"), (regLutBaseLo, "lutBaseLo"), (regLutBaseHi, "lutBaseHi")
  ] "weightless NE view" prefixMsg

  -- 2. Common block dims, groupCode, neViewCommon, neTile, targetCfg
  let wWin ← requireDecodedWrite task.writes regWin "Win"
  let wHin ← requireDecodedWrite task.writes regHin "Hin"
  let wCin ← requireDecodedWrite task.writes regCin "Cin"
  let wWout ← requireDecodedWrite task.writes regWout "Wout"
  let wHout ← requireDecodedWrite task.writes regHout "Hout"
  let wCout ← requireDecodedWrite task.writes regCout "Cout"
  let wDw ← requireDecodedWrite task.writes regDwGroups "dwGroups"
  let wTH ← requireDecodedWrite task.writes regTH "TH"
  let wNeCommon ← requireDecodedWrite task.writes regNeCommon "neCommon"
  let wNeTile ← requireDecodedWrite task.writes regNeTile "neTile"
  let wTargetCfg ← requireDecodedWrite task.writes regTargetCfg "targetCfg"

  let w := (← decodeFieldChecked (dim16Layout "Win") wWin.value "val").toNat
  let h := (← decodeFieldChecked (dim16Layout "Hin") wHin.value "val").toNat
  let c := wCin.value.toNat
  let wout := (← decodeFieldChecked (dim16Layout "Wout") wWout.value "val").toNat
  let hout := (← decodeFieldChecked (dim16Layout "Hout") wHout.value "val").toNat
  let cout := wCout.value.toNat
  let dIn ← match findDecodedWrite task.writes regDin with
    | some wd => pure (← decodeFieldChecked (dim16Layout "Din") wd.value "val").toNat
    | none => pure 1
  let dOut ← match findDecodedWrite task.writes regDout with
    | some wd => pure (← decodeFieldChecked (dim16Layout "Dout") wd.value "val").toNat
    | none => pure 1
  let batchN ← match findDecodedWrite task.writes regBatchN with
    | some wn => pure (← decodeFieldChecked (dim16Layout "BatchN") wn.value "val").toNat
    | none => pure 1
  let th := (← decodeFieldChecked (dim16Layout "TH") wTH.value "val").toNat

  if w == 0 || h == 0 || c == 0 || dIn == 0 || batchN == 0 then
    throw s!"{prefixMsg}: invalid view dims (N={batchN}, C={c}, H={h}, D={dIn}, W={w})"
  if wout != w || hout != h || cout != c || dOut != dIn then
    throw s!"{prefixMsg}: view output dims ({cout}x{hout}x{dOut}x{wout}) != input dims ({c}x{h}x{dIn}x{w})"
  if th == 0 || th > h then
    throw s!"{prefixMsg}: tileHeight={th} out of range 1..{h}"
  if wTargetCfg.value != targetCfgValue target then
    throw s!"{prefixMsg}: targetCfg {toHex wTargetCfg.value.toNat} != expected {toHex (targetCfgValue target).toNat}"

  let kind := (← decodeFieldChecked neViewCommonLayout wNeCommon.value "kind").toNat
  let shPref := (← decodeFieldChecked neViewCommonLayout wNeCommon.value "shPref").toNat
  let shMin := (← decodeFieldChecked neViewCommonLayout wNeCommon.value "shMin").toNat
  let shMax := (← decodeFieldChecked neViewCommonLayout wNeCommon.value "shMax").toNat
  let f := (← decodeFieldChecked neViewCommonLayout wNeCommon.value "F").toNat
  let tOutBit := (← decodeFieldChecked neViewCommonLayout wNeCommon.value "transposeOut").toNat
  let tWideBit := (← decodeFieldChecked neViewCommonLayout wNeCommon.value "transposeWideGroup").toNat
  if kind != 0 then
    throw s!"{prefixMsg}: neViewCommon kind={kind} != 0"
  if shPref > shMin || shMin > shMax || shMax > 4 then
    throw s!"{prefixMsg}: invalid shifts shPref={shPref}, shMin={shMin}, shMax={shMax}"

  let transposeCW := tOutBit == 1
  let expCode := if transposeCW then vGroupCode c else 1
  let expOcg := if transposeCW then vGroupCodeLog2 c else 0
  let expWide : Nat := if transposeCW && expCode >= 2 then 1 else 0

  let gotCode := (← decodeFieldChecked dwGroupsLayout wDw.value "perGroupIn").toNat
  if gotCode != expCode then
    throw s!"{prefixMsg}: dwGroups (0x0030) groupCode={gotCode} != expected {expCode} (transposeCW={transposeCW}, C={c})"
  if tWideBit != expWide then
    throw s!"{prefixMsg}: neViewCommon transposeWideGroup={tWideBit} != expected {expWide} (transposeCW={transposeCW}, groupCode={expCode})"

  let ocg := (← decodeFieldChecked neTileLayout wNeTile.value "ocg").toNat
  let fat := (← decodeFieldChecked neTileLayout wNeTile.value "fat").toNat
  let wu := (← decodeFieldChecked neTileLayout wNeTile.value "WU").toNat
  if ocg != expOcg then
    throw s!"{prefixMsg}: neTile ocg={ocg} != expected {expOcg} (transposeCW={transposeCW}, C={c})"
  if fat > 1 || wu > 3 || wu + f + fat != logNE then
    throw s!"{prefixMsg}: invalid neTile (WU={wu}, F={f}, fat={fat}, logNE={logNE})"

  let stripW := vStripW fat shPref
  let tmin := vStripW fat shMin
  if stripW < 8 || stripW > 256 || tmin < 8 || tmin > 256 then
    throw s!"{prefixMsg}: strip width T={stripW}, tmin={tmin} outside [8, 256]"

  if transposeCW then
    if f > 4 then
      throw s!"{prefixMsg}: T-mode F={f} exceeds maximum 4"
    let maxCov := (1 : Nat) <<< (f + ocg)
    if c > maxCov then
      throw s!"{prefixMsg}: T-mode C={c} exceeds channel-group coverage 2^(F+ocg)={maxCov} (F={f}, ocg={ocg})"
    let minShMax := vCoverMin w stripW fat
    if shMax < minShMax then
      throw s!"{prefixMsg}: T-mode shMax={shMax} < required coverMin(W={w}, T={stripW}, fat={fat})={minShMax}"

  -- 3. NE copy op & scale
  let wNeOp ← requireDecodedWrite task.writes regNeOp "neOp"
  if wNeOp.value != neOpViewCopy then
    throw s!"{prefixMsg}: neOp {toHex wNeOp.value.toNat} != expected neOpViewCopy ({toHex neOpViewCopy.toNat})"
  let wNeScale ← requireDecodedWrite task.writes regNeScale "neScale"
  if wNeScale.value != neScaleOne then
    throw s!"{prefixMsg}: neScale {toHex wNeScale.value.toNat} != expected neScaleOne ({toHex neScaleOne.toNat})"

  -- 4. Port configs
  let wSrc1Cfg ← requireDecodedWrite task.writes regSrc1Cfg "src1Cfg"
  let src1Cfg ← vDecodeSrcPortCfg "src1Cfg" wSrc1Cfg.value isFirst hasPriorNe hasPriorPe prefixMsg
  if isFirst then
    let wSrc2Cfg ← requireDecodedWrite task.writes regSrc2Cfg "src2Cfg"
    if wSrc2Cfg.value != src2CfgValue then
      throw s!"{prefixMsg}: src2Cfg {toHex wSrc2Cfg.value.toNat} != {toHex src2CfgValue.toNat}"
  else
    vCheckOmitted task.writes #[(regSrc2Cfg, "src2Cfg")] "NE view taskId > 0" prefixMsg

  let wResCfg ← requireDecodedWrite task.writes regResCfg "resCfg"
  let (resIsL2, direct) ← vDecodeResPortCfg wResCfg.value isFirst isLast prefixMsg

  -- 5. Resolve DMA endpoints
  let mut dramAccs : Array VerifiedDramAccess := #[]
  let mut sync : TaskSyncInfo := {
    src1WaitNe5 := src1Cfg.waitNe5
    src1WaitPe22 := src1Cfg.waitPe22
    src1HandoffNe := src1Cfg.handoffNe
    src1HandoffPe := src1Cfg.handoffPe
    hasSrc1Dma := !src1Cfg.isL2
  }

  let src1DmaTarget? ←
    if src1Cfg.isL2 then
      if batchN != 1 || dIn != 1 then
        throw s!"{prefixMsg}: L2-resident view src1 requires BatchN=1 and Din=1"
      vCheckOmitted task.writes #[
        (regSrc1Dep, "src1Dep"), (regSrc1DmaCfg, "src1DmaCfg"),
        (regSrc1BaseLo, "src1BaseLo"), (regSrc1BaseHi, "src1BaseHi"),
        (regSrc1DmaRowStride, "src1DmaRowStride"), (regSrc1DmaPlaneStride, "src1DmaPlaneStride"),
        (regSrc1DmaDStride, "src1DmaDStride"), (regSrc1DmaNStride, "src1DmaNStride"),
        (regSrc1Fmt, "src1Fmt")
      ] "L2-resident NE view src1" prefixMsg
      pure none
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regSrc1BaseLo regSrc1BaseHi "src1BaseLo" "src1BaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .input then
          throw s!"{prefixMsg}: src1Base BAR slot {slot} bound to non-input surface '{s.shortName}'"
        pure (some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: scratch src1Base must use BAR slot {barSlotScratch}, got {slot}"
        if isFirst then
          throw s!"{prefixMsg}: first task cannot read uninitialized DRAM scratch"
        pure (some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: src1Base BAR slot {slot} does not reference an input surface or scratch"

  let resDmaTarget? ←
    if resIsL2 then
      if transposeCW || batchN != 1 || dIn != 1 then
        throw s!"{prefixMsg}: L2-resident view res requires !transposeCW, BatchN=1, and Din=1"
      vCheckOmitted task.writes #[
        (regResDep, "resDep"), (regResBaseLo, "resBaseLo"), (regResBaseHi, "resBaseHi"),
        (regResDmaRowStride, "resDmaRowStride"), (regResDmaPlaneStride, "resDmaPlaneStride"),
        (regResDmaDStride, "resDmaDStride"), (regResDmaNStride, "resDmaNStride"),
        (regResFmt, "resFmt")
      ] "L2-resident NE view res" prefixMsg
      pure none
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regResBaseLo regResBaseHi "resBaseLo" "resBaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .output then
          throw s!"{prefixMsg}: resBase BAR slot {slot} bound to non-output surface '{s.shortName}'"
        if let some (s1Slot, .surface _, _) := src1DmaTarget? then
          if slot == s1Slot then
            throw s!"{prefixMsg}: output BAR slot {slot} aliases src1 BAR slot"
        pure (some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: scratch resBase must use BAR slot {barSlotScratch}, got {slot}"
        if isLast then
          throw s!"{prefixMsg}: final task cannot write to DRAM scratch"
        pure (some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: resBase BAR slot {slot} does not reference an output surface or scratch"

  -- 6. L2 staging geometry & <= 2 MiB capacity check
  let mut l2Accs : Array VerifiedL2Access := #[]
  let wSrc1Base ← requireDecodedWrite task.writes regSrc1Base "src1Base"
  let wSrc1Chs ← requireDecodedWrite task.writes regSrc1ChanStride "src1ChanStride"
  let wSrc1Row ← requireDecodedWrite task.writes regSrc1RowStride "src1RowStride"
  let srcL2Base := wSrc1Base.value.toNat
  if src1Cfg.isL2 then
    vCheckOmitted task.writes #[
      (regSrc1RowBytesA, "src1RowBytesA"), (regSrc1RowBytesB, "src1RowBytesB")
    ] "L2-resident NE view src1" prefixMsg
    let (expChs, expRow, l2Bytes) := vL2ResidentGeometry c h w
    if wSrc1Chs.value.toNat != expChs then
      throw s!"{prefixMsg}: L2 src1ChanStride {wSrc1Chs.value} != expected {expChs}"
    if wSrc1Row.value.toNat != expRow then
      throw s!"{prefixMsg}: L2 src1RowStride {wSrc1Row.value} != expected {expRow}"
    l2Accs := l2Accs.push { lo := srcL2Base, hi := srcL2Base + l2Bytes, isWrite := false, role := .src1L2 }
  else
    let wSrc1RowA ← requireDecodedWrite task.writes regSrc1RowBytesA "src1RowBytesA"
    let wSrc1RowB ← requireDecodedWrite task.writes regSrc1RowBytesB "src1RowBytesB"
    let effT := if transposeCW then tmin else stripW
    let expPx := min (alignUp w 8) (max effT 32)
    let expChs := 2 * expPx
    if wSrc1Chs.value.toNat != expChs then
      throw s!"{prefixMsg}: src1ChanStride {wSrc1Chs.value} != expected {expChs}"
    let expU := c * expChs
    let expDU := dIn * expU
    let src1Row := wSrc1Row.value.toNat
    if src1Row < expDU || (src1Row - expDU) % 16 != 0 then
      throw s!"{prefixMsg}: src1RowStride {src1Row} incompatible with D*C*chs={expDU} + 16*srcSkew"
    if wSrc1RowA.value.toNat != expU || wSrc1RowB.value.toNat != expDU then
      throw s!"{prefixMsg}: src1RowBytesA/B ({wSrc1RowA.value}, {wSrc1RowB.value}) != ({expU}, {expDU})"
    l2Accs := l2Accs.push { lo := srcL2Base, hi := srcL2Base + th * src1Row, isWrite := true, role := .src1Stage }

  let wResBase ← requireDecodedWrite task.writes regResBase "resBase"
  let resL2Base := wResBase.value.toNat
  if resIsL2 then
    vCheckOmitted task.writes #[
      (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
    ] "L2-resident NE view res" prefixMsg
    let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
    let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
    let (expChs, expRow, l2Bytes) := vL2ResidentGeometry c h w
    if wResChs.value.toNat != expChs then
      throw s!"{prefixMsg}: L2 resChanStride {wResChs.value} != expected {expChs}"
    if wResRow.value.toNat != expRow then
      throw s!"{prefixMsg}: L2 resRowStride {wResRow.value} != expected {expRow}"
    l2Accs := l2Accs.push { lo := resL2Base, hi := resL2Base + l2Bytes, isWrite := true, role := .resL2 }
  else
    let expDirect :=
      if transposeCW then
        !(9 <= c && c <= 16 && shMin == 0 && w < 32)
      else
        vIsDirect target fat shMax w
    if direct != expDirect then
      throw s!"{prefixMsg}: resCfg direct={direct} disagrees with expected direct={expDirect} (transposeCW={transposeCW})"
    if direct then
      vCheckOmitted task.writes #[
        (regResChanStride, "resChanStride"), (regResRowStride, "resRowStride"),
        (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
      ] "direct mode" prefixMsg
      let resDirectBytes := ((1 : Nat) <<< (f + ocg + fat)) * 512
      l2Accs := l2Accs.push { lo := resL2Base, hi := resL2Base + resDirectBytes, isWrite := true, role := .resStage }
    else
      if dOut > 1 then
        throw s!"{prefixMsg}: staged mode (direct=false) requires D=1, got D={dOut}"
      let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
      let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
      let wResRowA ← requireDecodedWrite task.writes regResRowBytesA "resRowBytesA"
      let wResRowB ← requireDecodedWrite task.writes regResRowBytesB "resRowBytesB"
      let (rw, rch) := if transposeCW then (c, w) else (w, c)
      let expResChs := 2 * min (alignUp rw 8) (max stripW 32)
      if wResChs.value.toNat != expResChs then
        throw s!"{prefixMsg}: staged resChanStride {wResChs.value} != expected {expResChs}"
      let expResRowBytes := rch * expResChs
      let resRow := wResRow.value.toNat
      if resRow < expResRowBytes || (resRow - expResRowBytes) % 16 != 0 then
        throw s!"{prefixMsg}: staged resRowStride {resRow} incompatible with rch*resChs={expResRowBytes} + 16*resSkew"
      if wResRowA.value.toNat != expResRowBytes || wResRowB.value.toNat != expResRowBytes then
        throw s!"{prefixMsg}: staged resRowBytesA/B ({wResRowA.value}, {wResRowB.value}) != {expResRowBytes}"
      l2Accs := l2Accs.push { lo := resL2Base, hi := resL2Base + th * resRow, isWrite := true, role := .resStage }

  vCheckTaskL2Accesses l2Accs prefixMsg

  -- 7. Verify DMA deps, formats, 64B alignment, and exact ane_hx_dma_tile bounds
  let checkViewStride := fun (dim : Nat) (addr : Nat) (nm dirLbl : String) => do
    match findDecodedWrite task.writes addr with
    | some wr =>
      let sv := wr.value.toNat
      if sv % vDmaLineBytes != 0 then
        throw s!"{prefixMsg}: {dirLbl} {nm} ({sv}) is not {vDmaLineBytes}-byte aligned"
      if dim > 1 && sv == 0 then
        throw s!"{prefixMsg}: {dirLbl} {nm} must be > 0 for dim={dim} > 1"
      pure sv
    | none =>
      if dim > 1 then
        throw s!"{prefixMsg}: {dirLbl} missing required stride {nm} ({toHex addr}) for dim={dim} > 1"
      pure 0

  if let some (_, tgt, off) := src1DmaTarget? then
    let (dk, dd) ← vDecodeSrcDep info task.writes regSrc1Dep regSrc1DmaCfg "src1Dep" "src1DmaCfg" taskId isFirst prefixMsg
    sync := { sync with src1DepKind := dk, src1DepDist := dd }
    let wFmt ← requireDecodedWrite task.writes regSrc1Fmt "src1Fmt"
    if wFmt.value != srcFmtWordP1 then
      throw s!"{prefixMsg}: src1Fmt {toHex wFmt.value.toNat} != {toHex srcFmtWordP1.toNat}"
    let (resId, cap, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, "invalid")
    if off % vDmaLineBytes != 0 then
      throw s!"{prefixMsg}: input DMA {lbl} base offset {toHex off} is not {vDmaLineBytes}-byte aligned"
    let inRowSt ← checkViewStride h regSrc1DmaRowStride "src1DmaRowStride" "input DMA"
    let inPlaneSt ← checkViewStride c regSrc1DmaPlaneStride "src1DmaPlaneStride" "input DMA"
    let inDSt ← checkViewStride dIn regSrc1DmaDStride "src1DmaDStride" "input DMA"
    let inNSt ← checkViewStride batchN regSrc1DmaNStride "src1DmaNStride" "input DMA"
    let rawInExt := 2 * w + (h - 1) * inRowSt + (c - 1) * inPlaneSt + (dIn - 1) * inDSt + (batchN - 1) * inNSt
    let alignedInEnd := alignUp (off + rawInExt) vDmaLineBytes
    if alignedInEnd > cap then
      throw s!"{prefixMsg}: input DMA ane_hx_dma_tile bound [{toHex off}, {toHex alignedInEnd}) exceeds {lbl} capacity {cap}"
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := alignedInEnd, isWrite := false, port := .src1Dma }

  if let some (_, tgt, off) := resDmaTarget? then
    match ← resolveBarTarget info barSlotConfigZero with
    | .configZero => pure ()
    | _ => throw s!"{prefixMsg}: BAR slot {barSlotConfigZero} must have VM base 0"
    let wResDep ← requireDecodedWrite task.writes regResDep "resDep"
    if wResDep.barSlot != some barSlotConfigZero then
      throw s!"{prefixMsg}: resDep must use BAR slot {barSlotConfigZero}"
    let expCode := match tgt with
      | .scratch _ => resDepCode target ||| resDepScratchFlags
      | _ => resDepCode target
    if (← decodeFieldChecked resDepLayout wResDep.value "code") != expCode then
      throw s!"{prefixMsg}: resDep code mismatch"
    if (← decodeFieldChecked resDepLayout wResDep.value "task").toNat != taskId then
      throw s!"{prefixMsg}: resDep task != header taskId={taskId}"
    if ((← decodeFieldChecked resDepLayout wResDep.value "direct") == 1) != direct then
      throw s!"{prefixMsg}: resDep direct bit != resCfg direct={direct}"
    let wResFmt ← requireDecodedWrite task.writes regResFmt "resFmt"
    if wResFmt.value != resFmtWordP1 then
      throw s!"{prefixMsg}: resFmt {toHex wResFmt.value.toNat} != {toHex resFmtWordP1.toNat}"
    let (resId, cap, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, "invalid")
    if off % vDmaLineBytes != 0 then
      throw s!"{prefixMsg}: output DMA {lbl} base offset {toHex off} is not {vDmaLineBytes}-byte aligned"
    let resPlaneDim := if transposeCW then w else c
    let outRowSt ← checkViewStride h regResDmaRowStride "resDmaRowStride" "output DMA"
    let outPlaneSt ← checkViewStride resPlaneDim regResDmaPlaneStride "resDmaPlaneStride" "output DMA"
    let outDSt ← checkViewStride dOut regResDmaDStride "resDmaDStride" "output DMA"
    let outNSt ← checkViewStride batchN regResDmaNStride "resDmaNStride" "output DMA"
    -- Exact kernel ane_hx_dma_tile bound (evaluates using 0x14=Wout, 0x18=Hout, 0x1c=Cout, 0x20=Dout, 0x24=BatchN)
    let rawKernelOutExt := 2 * w + (h - 1) * outRowSt + (c - 1) * outPlaneSt + (dOut - 1) * outDSt + (batchN - 1) * outNSt
    let alignedKernelOutEnd := alignUp (off + rawKernelOutExt) vDmaLineBytes
    if alignedKernelOutEnd > cap then
      throw s!"{prefixMsg}: output DMA ane_hx_dma_tile bound [{toHex off}, {toHex alignedKernelOutEnd}) exceeds {lbl} capacity {cap}"
    -- Actual hardware write bound (exchanges C and W when transposeCW == true)
    let (rw, rch) := if transposeCW then (c, w) else (w, c)
    let rawActualOutExt := 2 * rw + (h - 1) * outRowSt + (rch - 1) * outPlaneSt + (dOut - 1) * outDSt + (batchN - 1) * outNSt
    let alignedActualOutEnd := alignUp (off + rawActualOutExt) vDmaLineBytes
    if alignedActualOutEnd > cap then
      throw s!"{prefixMsg}: output DMA write range [{toHex off}, {toHex alignedActualOutEnd}) exceeds {lbl} capacity {cap}"
    let outCov := classifyStridedWriteCoverage batchN dOut rch h rw off outNSt outDSt outPlaneSt outRowSt
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := alignedActualOutEnd, isWrite := true, port := .resDma, coverage := outCov }

  -- 8. Kernel stage, output stage, and strict mode-conditional register allowlist
  vVerifyKernelStageMulti task.writes taskId isLast false false prefixMsg
  vVerifyOutputStageMulti task.writes taskId isLast prefixMsg

  let mut allowed : Array Nat := #[
    regWin, regHin, regCin, regDin, regWout, regHout, regCout, regDout, regBatchN,
    regDwGroups, regTH, regNeCommon, regNeTile, regTargetCfg,
    regSrc1Cfg, regSrc1Base, regSrc1ChanStride, regSrc1RowStride,
    regResCfg, regResBase, regNeOp, regNeScale
  ]
  if isFirst then allowed := allowed.push regSrc2Cfg
  if !src1Cfg.isL2 then
    allowed := allowed ++ #[
      regSrc1RowBytesA, regSrc1RowBytesB,
      regSrc1Dep, regSrc1DmaCfg, regSrc1BaseLo, regSrc1BaseHi,
      regSrc1DmaRowStride, regSrc1DmaPlaneStride, regSrc1DmaDStride, regSrc1DmaNStride,
      regSrc1Fmt
    ]
  if resIsL2 then
    allowed := allowed ++ #[regResChanStride, regResRowStride]
  else
    allowed := allowed ++ #[
      regResDep, regResBaseLo, regResBaseHi,
      regResDmaRowStride, regResDmaPlaneStride, regResDmaDStride, regResDmaNStride,
      regResFmt
    ]
    if !direct then
      allowed := allowed ++ #[
        regResChanStride, regResRowStride, regResRowBytesA, regResRowBytesB
      ]
  if isLast then
    allowed := allowed ++ #[regKernelStage, regOut0, regOut1, regOut2, regOut3, regOut4, regOut5]
  vCheckAllowedRegs task.writes allowed prefixMsg

  pure {
    taskIndex := task.taskIndex
    kind := .neView
    engine := .ne
    isNeCompute := false
    isMultiBatch := batchN > 1
    liveL2Mask := h7
    l2Accesses := l2Accs
    dramAccesses := dramAccs
    sync := sync
  }

-- ============================================================================
-- Multi-Task Dynamic Runtime-Kernel NE MatMul Static Safety Verifier (S1-S4)
-- ============================================================================

/-- Independent static safety verifier for dynamic runtime-kernel NE `MatMul` tasks
    (`0x4900` bit 16 = `1`, `0x4904 = 0x00103c00`) reading runtime kernel `A` from an
    input surface or `BAR 3` scratch and activation `B` from DRAM or L2. -/
def verifyNeMatmulTaskMulti
    (taskCount : Nat) (hasPriorNe hasPriorPe : Bool)
    (info : HwxInfo) (task : DecodedTask) : Except String VerifiedTaskFootprint := do
  let target := info.target
  let prefixMsg := s!"task {task.taskIndex}"
  let isFirst := task.taskIndex == 0
  let isLast := task.taskIndex + 1 == taskCount

  -- 1. Verify 9-word or 10-word extended task header
  let (taskId, ene, h7, extProd?) ← vVerifyTaskHeaderMatmul target task taskCount prefixMsg
  let logNE := ene - 1

  -- 2. Reject duplicate writes, unmodelled DMA registers, and Mach-O __KERN_0 relocations on neMatmul
  vCheckNoDuplicateWrites task.writes prefixMsg
  vCheckNoUnmodelledDmaRegs task.writes prefixMsg
  for w in task.writes do
    if info.relocPatchOffsets.contains w.payloadOffset.toUInt32 then
      throw s!"{prefixMsg}: neMatmul register {toHex w.byteAddr} must not carry a Mach-O __KERN_0 relocation"

  -- 3. Decode common dimensions, convCfg, TH, neCommon, neTile, targetCfg
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

  let n := (← decodeFieldChecked (dim16Layout "Win") wWin.value "val").toNat
  let hin := (← decodeFieldChecked (dim16Layout "Hin") wHin.value "val").toNat
  let k := wCin.value.toNat
  let wout := (← decodeFieldChecked (dim16Layout "Wout") wWout.value "val").toNat
  let hout := (← decodeFieldChecked (dim16Layout "Hout") wHout.value "val").toNat
  let m := wCout.value.toNat
  let th := (← decodeFieldChecked (dim16Layout "TH") wTH.value "val").toNat
  let b ← match findDecodedWrite task.writes regBatchN with
    | some wB => pure (← decodeFieldChecked (dim16Layout "BatchN") wB.value "val").toNat
    | none => pure 1

  if hin != 1 || hout != 1 then
    throw s!"{prefixMsg}: neMatmul requires Hin=1 and Hout=1, got Hin={hin}, Hout={hout}"
  if n == 0 || wout != n then
    throw s!"{prefixMsg}: neMatmul requires Win == Wout >= 1 (got Win={n}, Wout={wout})"
  if k == 0 || m == 0 || b == 0 then
    throw s!"{prefixMsg}: neMatmul requires B >= 1, M >= 1, K >= 1 (got B={b}, M={m}, K={k})"
  if m > 16 then
    throw s!"{prefixMsg}: neMatmul ocg=0 task Cout={m} exceeds 16 chunk slots (slice M > 16 into Mi <= 16 tasks)"
  if th != 1 then
    throw s!"{prefixMsg}: neMatmul requires TH=1 (Hin=1), got TH={th}"

  let kw := (← decodeFieldChecked convCfgLayout wConvCfg.value "kw").toNat
  let kh := (← decodeFieldChecked convCfgLayout wConvCfg.value "kh").toNat
  let sx := (← decodeFieldChecked convCfgLayout wConvCfg.value "sx").toNat
  let sy := (← decodeFieldChecked convCfgLayout wConvCfg.value "sy").toNat
  let pl := (← decodeFieldChecked convCfgLayout wConvCfg.value "pl").toNat
  let pt := (← decodeFieldChecked convCfgLayout wConvCfg.value "pt").toNat
  let hx := (← decodeFieldChecked convCfgLayout wConvCfg.value "hx").toNat
  let vy := (← decodeFieldChecked convCfgLayout wConvCfg.value "vy").toNat
  if kw != 1 || kh != 1 || sx != 1 || sy != 1 || pl != 0 || pt != 0 || hx != 1 || vy != 1 then
    throw s!"{prefixMsg}: neMatmul convCfg {toHex wConvCfg.value.toNat} must encode 1x1 stride-1"

  if wTargetCfg.value != targetCfgValue target then
    throw s!"{prefixMsg}: targetCfg {toHex wTargetCfg.value.toNat} != expected {toHex (targetCfgValue target).toNat}"

  let isDyn : Bool := m * b > 1
  let lo2 := (← decodeFieldChecked neMatmulCommonLayout wNeCommon.value "lo2").toNat
  let shPref := (← decodeFieldChecked neMatmulCommonLayout wNeCommon.value "shPref").toNat
  let shMin := (← decodeFieldChecked neMatmulCommonLayout wNeCommon.value "shMin").toNat
  let shMax := (← decodeFieldChecked neMatmulCommonLayout wNeCommon.value "shMax").toNat
  let f := (← decodeFieldChecked neMatmulCommonLayout wNeCommon.value "F").toNat
  let dynBit := (← decodeFieldChecked neMatmulCommonLayout wNeCommon.value "dynKern").toNat

  let expLo2 := if alignUp n 8 <= 64 then 1 else 0
  if lo2 != expLo2 then
    throw s!"{prefixMsg}: neMatmul common lo2={lo2} != expected {expLo2} for N={n}"
  let expDyn := if isDyn then 1 else 0
  if dynBit != expDyn then
    throw s!"{prefixMsg}: neMatmul common dynKern={dynBit} != expected {expDyn} (M={m}, B={b})"
  if shMin != shPref || shPref > shMax || shMax > 4 then
    throw s!"{prefixMsg}: neMatmul invalid shifts (shPref={shPref}, shMin={shMin}, shMax={shMax})"

  let ocg := (← decodeFieldChecked neTileLayout wNeTile.value "ocg").toNat
  let fat := (← decodeFieldChecked neTileLayout wNeTile.value "fat").toNat
  let wu := (← decodeFieldChecked neTileLayout wNeTile.value "WU").toNat
  if ocg != 0 || fat != 0 || wu != 0 then
    throw s!"{prefixMsg}: neMatmul requires ocg=0, fat=0, WU=0 (got ocg={ocg}, fat={fat}, WU={wu})"
  if f > 4 || logNE != f then
    throw s!"{prefixMsg}: neMatmul requires F <= 4 and logNE == F (got F={f}, logNE={logNE})"
  if ((1 : Nat) <<< f) < m then
    throw s!"{prefixMsg}: neMatmul 2^F={(1 : Nat) <<< f} < M={m}"

  let stripW := vStripW 0 shPref
  if stripW < 8 || stripW > 256 then
    throw s!"{prefixMsg}: neMatmul strip width T={stripW} outside [8, 256]"
  let minShMax := vCoverMin n stripW 0
  if shMax < minShMax then
    throw s!"{prefixMsg}: neMatmul shMax={shMax} < required coverMin(N={n}, T={stripW}, fat=0)={minShMax}"

  -- 4. Verify NE config, op, and finite fp16 scale
  let wNeCfg ← requireDecodedWrite task.writes regNeCfg "neCfg"
  let neCfgDyn ← decodeFieldChecked (neMatmulCfgLayout target) wNeCfg.value "dynKern"
  if neCfgDyn != 1 then
    throw s!"{prefixMsg}: neMatmul neCfg {toHex wNeCfg.value.toNat} must set dynKern=1 ({toHex (neCfgDynValue target).toNat})"

  let wNeOp ← requireDecodedWrite task.writes regNeOp "neOp"
  if wNeOp.value != neOpMatmul then
    throw s!"{prefixMsg}: neMatmul neOp {toHex wNeOp.value.toNat} != expected {toHex neOpMatmul.toNat}"

  let wNeScale ← requireDecodedWrite task.writes regNeScale "neScale"
  let scaleVal ← decodeFieldChecked neScaleLayout wNeScale.value "val"
  if (scaleVal &&& 0x7c00) == 0x7c00 then
    throw s!"{prefixMsg}: neMatmul neScale {toHex scaleVal.toNat} must be finite fp16"

  -- 5. Verify kernelStage (0x5500), including H17 bit 10 (h17Dyn)
  let wKs ← requireDecodedWrite task.writes regKernelStage "kernelStage"
  let ksSparse ← decodeFieldChecked kernelStageDynLayout wKs.value "sparse"
  let ksEn ← decodeFieldChecked kernelStageDynLayout wKs.value "kernEn"
  let ksLast ← decodeFieldChecked kernelStageDynLayout wKs.value "isLast"
  let ksCfg ← decodeFieldChecked kernelStageDynLayout wKs.value "kernCfg"
  let ksH17Dyn ← decodeFieldChecked kernelStageDynLayout wKs.value "h17Dyn"
  let ksStage ← decodeFieldChecked kernelStageDynLayout wKs.value "stage"
  if ksSparse != 0 then
    throw s!"{prefixMsg}: neMatmul kernelStage sparse={ksSparse} must be 0"
  if ksEn != 1 || ksCfg != 1 then
    throw s!"{prefixMsg}: neMatmul kernelStage must set kernEn=1 and kernCfg=1 ({toHex wKs.value.toNat})"
  let expKsLast : UInt32 := if isLast then 1 else 0
  if ksLast != expKsLast then
    throw s!"{prefixMsg}: neMatmul kernelStage isLast={ksLast} != expected {expKsLast}"
  if taskId + 1 > 255 then
    throw s!"{prefixMsg}: neMatmul taskId={taskId} overflows 8-bit kernelStage stage={taskId + 1}"
  if ksStage.toNat != taskId + 1 then
    throw s!"{prefixMsg}: neMatmul kernelStage stage={ksStage} != expected {taskId + 1}"
  let expH17Dyn : UInt32 := if target.profile.dynKernelStageBit && isDyn then 1 else 0
  if ksH17Dyn != expH17Dyn then
    throw s!"{prefixMsg}: neMatmul kernelStage h17Dyn={ksH17Dyn} != expected {expH17Dyn} for {target.name} (dynKern={isDyn})"

  -- 6. Verify kernelBaseLo/Hi (0x5510/0x5514), produced vs external-input mode, and extended header consistency
  let (kSlot, kTarget, kOff, _) ←
    resolveBasePair info task.writes regKernelBaseLo regKernelBaseHi "kernelBaseLo" "kernelBaseHi" prefixMsg
  if kOff % vDmaLineBytes != 0 then
    throw s!"{prefixMsg}: neMatmul kernelBaseLo offset {toHex kOff} is not {vDmaLineBytes}-byte aligned"

  let (produced, kRes, kCap, kLbl) ← match kTarget with
    | .surface s =>
      if s.direction != .input then
        throw s!"{prefixMsg}: neMatmul kernelBase BAR slot {kSlot} bound to non-input surface '{s.shortName}'"
      pure (false, BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, s!"surface '{s.shortName}'")
    | .scratch sc =>
      if kSlot != barSlotScratch then
        throw s!"{prefixMsg}: neMatmul scratch kernelBase must use BAR slot {barSlotScratch}, got {kSlot}"
      if isFirst then
        throw s!"{prefixMsg}: first task cannot read uninitialized BAR 3 scratch kernel"
      pure (true, BarResource.scratch, sc.size.toNat, "scratch (__DATA,__bss)")
    | .kernel _ =>
      throw s!"{prefixMsg}: neMatmul kernelBase cannot reference static __KERN_0"
    | .configZero =>
      throw s!"{prefixMsg}: neMatmul kernelBase BAR slot {kSlot} has zero VM base"

  if produced && extProd?.isNone then
    throw s!"{prefixMsg}: scratch-produced neMatmul kernel requires 10-word extended header (hdr8 bit 1)"
  if !produced && extProd?.isSome then
    throw s!"{prefixMsg}: external-input neMatmul kernel forbids 10-word extended header (hdr8 bit 1)"

  -- 7. Verify chunkSize <= 64 KiB, kernelBatchStride, kernelRoundStride, chunk slot table, and kernel DMA bounds
  let chunkSize := alignUp (2 * k) vDmaLineBytes
  if chunkSize == 0 || chunkSize > vMaxKernelChunkBytes then
    throw s!"{prefixMsg}: neMatmul chunkSize={chunkSize}B exceeds hardware limit {vMaxKernelChunkBytes}B (64 KiB)"

  let kBatchSt ←
    if b > 1 then
      let wBs ← requireDecodedWrite task.writes regKernelBatchStride "kernelBatchStride"
      let bs := wBs.value.toNat
      if bs < m * chunkSize || bs % chunkSize != 0 then
        throw s!"{prefixMsg}: neMatmul kernelBatchStride={bs} must be >= M*chunkSize ({m * chunkSize}) and a multiple of chunkSize ({chunkSize})"
      pure bs
    else
      vCheckOmitted task.writes #[(regKernelBatchStride, "kernelBatchStride")] "neMatmul B=1" prefixMsg
      pure (m * chunkSize)

  if isDyn then
    let wRs ← requireDecodedWrite task.writes regKernelRoundStride "kernelRoundStride"
    let expRs := (if m > 1 then m else b) * chunkSize
    if wRs.value.toNat != expRs then
      throw s!"{prefixMsg}: neMatmul kernelRoundStride={wRs.value} != expected {expRs}"
  else
    vCheckOmitted task.writes #[(regKernelRoundStride, "kernelRoundStride")] "neMatmul M*B=1" prefixMsg

  if !produced then
    match ← resolveBarTarget info barSlotConfigZero with
    | .configZero => pure ()
    | _ => throw s!"{prefixMsg}: BAR slot {barSlotConfigZero} must have VM base 0"

  let numSlots := m
  for s in [:16] do
    let slotW? := findDecodedWrite task.writes (regChunkSlot s)
    let offW? := findDecodedWrite task.writes (regChunkOffset s)
    let sizeW? := findDecodedWrite task.writes (regChunkSize s)
    if s < numSlots then
      let some sw := slotW?
        | throw s!"{prefixMsg}: neMatmul missing required active chunk slot {s} ({toHex (regChunkSlot s)})"
      let some ow := offW?
        | throw s!"{prefixMsg}: neMatmul missing chunkOffset[{s}] ({toHex (regChunkOffset s)})"
      let some szw := sizeW?
        | throw s!"{prefixMsg}: neMatmul missing chunkSize[{s}] ({toHex (regChunkSize s)})"
      let en ← decodeFieldChecked chunkSlotDynLayout sw.value "enable"
      let on20 ← decodeFieldChecked chunkSlotDynLayout sw.value "on20"
      let prodBit ← decodeFieldChecked chunkSlotDynLayout sw.value "produced"
      let keepLiveBit ← decodeFieldChecked chunkSlotDynLayout sw.value "keepLive"
      let slotTask ← decodeFieldChecked chunkSlotDynLayout sw.value "task"
      if en != 1 then
        throw s!"{prefixMsg}: neMatmul active chunk slot {s} must set enable=1"
      if slotTask.toNat != taskId then
        throw s!"{prefixMsg}: neMatmul chunk slot {s} task={slotTask} != header taskId={taskId}"
      if !produced then
        if on20 != 1 || prodBit != 0 || keepLiveBit != 0 then
          throw s!"{prefixMsg}: external-input neMatmul chunk slot {s} must encode 0x21 (got {toHex sw.value.toNat})"
        if sw.barSlot != some barSlotConfigZero then
          throw s!"{prefixMsg}: external-input neMatmul chunk slot {s} ({toHex (regChunkSlot s)}) must use BAR slot {barSlotConfigZero} config relocation"
        if sw.recordLen != 1 then
          throw s!"{prefixMsg}: external-input neMatmul chunk slot {s} ({toHex (regChunkSlot s)}) must be a single-word BAR slot {barSlotConfigZero} record (got {sw.recordLen} words)"
      else
        if on20 != 0 || prodBit != 1 then
          throw s!"{prefixMsg}: scratch-produced neMatmul chunk slot {s} must encode 0x41/0xc1 (got {toHex sw.value.toNat})"
        if keepLiveBit == 1 && isLast then
          throw s!"{prefixMsg}: scratch-produced neMatmul chunk slot {s} cannot set keepLive (0x80) on final task"
        if sw.barSlot.isSome then
          throw s!"{prefixMsg}: scratch-produced neMatmul chunk slot {s} must be a plain register write (not BAR relocated)"
      let expOff := s * chunkSize
      if ow.value.toNat != expOff then
        throw s!"{prefixMsg}: neMatmul chunkOffset[{s}]={ow.value} != expected {expOff}"
      if szw.value.toNat != chunkSize then
        throw s!"{prefixMsg}: neMatmul chunkSize[{s}]={szw.value} != expected {chunkSize}"
    else
      if offW?.isSome || sizeW?.isSome then
        throw s!"{prefixMsg}: neMatmul inactive chunk slot {s} must omit offset/size registers"
      match slotW? with
      | none =>
        if !isFirst then
          throw s!"{prefixMsg}: neMatmul taskId > 0 must explicitly disable unused chunk slot {s} ({toHex (regChunkSlot s)}) to prevent carry-over"
      | some sw =>
        if sw.barSlot.isSome then
          throw s!"{prefixMsg}: neMatmul disabled chunk slot {s} must be a plain register write"
        let en ← decodeFieldChecked chunkSlotDynLayout sw.value "enable"
        let on20 ← decodeFieldChecked chunkSlotDynLayout sw.value "on20"
        let prodBit ← decodeFieldChecked chunkSlotDynLayout sw.value "produced"
        let keepLiveBit ← decodeFieldChecked chunkSlotDynLayout sw.value "keepLive"
        let slotTask ← decodeFieldChecked chunkSlotDynLayout sw.value "task"
        if en != 0 || on20 != 1 || prodBit != 0 || keepLiveBit != 0 || slotTask.toNat != taskId then
          throw s!"{prefixMsg}: neMatmul disabled chunk slot {s} invalid encoding {toHex sw.value.toNat}"

  let kEnd := kOff + (if b > 1 then (b - 1) * kBatchSt else 0) + m * chunkSize
  if m * chunkSize == 0 || kEnd > kCap then
    throw s!"{prefixMsg}: neMatmul kernel DMA range [{toHex kOff}, {toHex kEnd}) exceeds {kLbl} capacity {kCap}"

  let mut dramAccs : Array VerifiedDramAccess := #[]
  if b > 1 && kBatchSt > m * chunkSize then
    for bIdx in [:b] do
      let lo := kOff + bIdx * kBatchSt
      let hi := lo + m * chunkSize
      dramAccs := dramAccs.push {
        resource := kRes
        lo := lo
        hi := hi
        isWrite := false
        port := .kernel
      }
  else
    dramAccs := dramAccs.push {
      resource := kRes
      lo := kOff
      hi := kOff + b * m * chunkSize
      isWrite := false
      port := .kernel
    }

  -- 8. Decode L2/DMA port configs for src1 (B) and res (Y)
  let wSrc1Cfg ← requireDecodedWrite task.writes regSrc1Cfg "src1Cfg"
  let src1Cfg ← vDecodeSrcPortCfg "src1Cfg" wSrc1Cfg.value isFirst hasPriorNe hasPriorPe prefixMsg
  if isFirst then
    let wSrc2Cfg ← requireDecodedWrite task.writes regSrc2Cfg "src2Cfg"
    if wSrc2Cfg.value != src2CfgValue then
      throw s!"{prefixMsg}: src2Cfg {toHex wSrc2Cfg.value.toNat} != {toHex src2CfgValue.toNat}"
  else
    vCheckOmitted task.writes #[(regSrc2Cfg, "src2Cfg")] "neMatmul taskId > 0" prefixMsg

  let wResCfg ← requireDecodedWrite task.writes regResCfg "resCfg"
  let (resIsL2, direct) ← vDecodeResPortCfg wResCfg.value isFirst isLast prefixMsg

  let mut sync : TaskSyncInfo := {
    src1WaitNe5 := src1Cfg.waitNe5
    src1WaitPe22 := src1Cfg.waitPe22
    src1HandoffNe := src1Cfg.handoffNe
    src1HandoffPe := src1Cfg.handoffPe
    hasSrc1Dma := !src1Cfg.isL2
  }

  -- 9. Resolve DMA targets for src1 (B) and res (Y)
  let src1DmaTarget? ←
    if src1Cfg.isL2 then
      vCheckOmitted task.writes #[
        (regSrc1Dep, "src1Dep"), (regSrc1DmaCfg, "src1DmaCfg"),
        (regSrc1BaseLo, "src1BaseLo"), (regSrc1BaseHi, "src1BaseHi"),
        (regSrc1DmaRowStride, "src1DmaRowStride"), (regSrc1DmaPlaneStride, "src1DmaPlaneStride"),
        (regSrc1DmaNStride, "src1DmaNStride"), (regSrc1Fmt, "src1Fmt")
      ] "L2-resident neMatmul src1" prefixMsg
      pure none
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regSrc1BaseLo regSrc1BaseHi "src1BaseLo" "src1BaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .input then
          throw s!"{prefixMsg}: neMatmul src1Base BAR slot {slot} bound to non-input surface '{s.shortName}'"
        if s.height.toNat != 1 || s.width.toNat != n then
          throw s!"{prefixMsg}: neMatmul src1 surface '{s.shortName}' dims (H={s.height}, W={s.width}) incompatible with (H=1, W={n})"
        pure (some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: neMatmul scratch src1Base must use BAR slot {barSlotScratch}, got {slot}"
        if isFirst then
          throw s!"{prefixMsg}: first task cannot read uninitialized DRAM scratch"
        pure (some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: neMatmul src1Base BAR slot {slot} does not reference an input surface or scratch"

  let resDmaTarget? ←
    if resIsL2 then
      vCheckOmitted task.writes #[
        (regResDep, "resDep"), (regResBaseLo, "resBaseLo"), (regResBaseHi, "resBaseHi"),
        (regResDmaRowStride, "resDmaRowStride"), (regResDmaPlaneStride, "resDmaPlaneStride"),
        (regResDmaNStride, "resDmaNStride"), (regResFmt, "resFmt")
      ] "L2-resident neMatmul res" prefixMsg
      pure none
    else
      let (slot, tgt, off, _) ← resolveBasePair info task.writes regResBaseLo regResBaseHi "resBaseLo" "resBaseHi" prefixMsg
      match tgt with
      | .surface s =>
        if s.direction != .output then
          throw s!"{prefixMsg}: neMatmul resBase BAR slot {slot} bound to non-output surface '{s.shortName}'"
        if s.height.toNat != 1 || s.width.toNat != n then
          throw s!"{prefixMsg}: neMatmul res surface '{s.shortName}' dims (H={s.height}, W={s.width}) incompatible with (H=1, W={n})"
        pure (some (slot, tgt, off))
      | .scratch _ =>
        if slot != barSlotScratch then
          throw s!"{prefixMsg}: neMatmul scratch resBase must use BAR slot {barSlotScratch}, got {slot}"
        if isLast then
          throw s!"{prefixMsg}: final task cannot write to DRAM scratch"
        pure (some (slot, tgt, off))
      | _ => throw s!"{prefixMsg}: neMatmul resBase BAR slot {slot} does not reference an output surface or scratch"

  -- 10. L2 staging geometry & <= 2 MiB capacity check
  let mut l2Accs : Array VerifiedL2Access := #[]
  let wSrc1Base ← requireDecodedWrite task.writes regSrc1Base "src1Base"
  let srcL2Base := wSrc1Base.value.toNat
  if src1Cfg.isL2 then
    let expChs := 2 * alignUp n 8
    let expBatch := k * expChs
    let l2Bytes := alignUp (b * expBatch) vDmaLineBytes
    if let some wChs := findDecodedWrite task.writes regSrc1ChanStride then
      if wChs.value.toNat != expChs then
        throw s!"{prefixMsg}: neMatmul L2 src1ChanStride {wChs.value} != expected {expChs}"
    else if k > 1 then
      throw s!"{prefixMsg}: neMatmul L2 src1 missing src1ChanStride for K={k} > 1"
    if let some wRow := findDecodedWrite task.writes regSrc1RowStride then
      if wRow.value.toNat != expBatch then
        throw s!"{prefixMsg}: neMatmul L2 src1RowStride {wRow.value} != expected {expBatch}"
    vCheckOmitted task.writes #[(regSrc1RowBytesA, "src1RowBytesA")] "L2-resident neMatmul src1" prefixMsg
    if b > 1 then
      let wB ← requireDecodedWrite task.writes regSrc1RowBytesB "src1RowBytesB"
      if wB.value.toNat != expBatch then
        throw s!"{prefixMsg}: neMatmul L2 src1RowBytesB {wB.value} != expected {expBatch}"
    else
      vCheckOmitted task.writes #[(regSrc1RowBytesB, "src1RowBytesB")] "L2-resident neMatmul src1 B=1" prefixMsg
    l2Accs := l2Accs.push { lo := srcL2Base, hi := srcL2Base + l2Bytes, isWrite := false, role := .src1L2 }
  else
    let wSrc1Chs ← requireDecodedWrite task.writes regSrc1ChanStride "src1ChanStride"
    let wSrc1Row ← requireDecodedWrite task.writes regSrc1RowStride "src1RowStride"
    let wSrc1RowA ← requireDecodedWrite task.writes regSrc1RowBytesA "src1RowBytesA"
    let wSrc1RowB ← requireDecodedWrite task.writes regSrc1RowBytesB "src1RowBytesB"
    let expChs := 2 * min (alignUp n 8) (max stripW 32)
    if wSrc1Chs.value.toNat != expChs then
      throw s!"{prefixMsg}: neMatmul src1ChanStride {wSrc1Chs.value} != expected {expChs}"
    let minSrcRow := k * expChs
    let src1Row := wSrc1Row.value.toNat
    if src1Row < minSrcRow || (src1Row - minSrcRow) % 16 != 0 then
      throw s!"{prefixMsg}: neMatmul src1RowStride {src1Row} incompatible with K*chs={minSrcRow} + 16*srcSkew"
    if wSrc1RowA.value.toNat != minSrcRow || wSrc1RowB.value.toNat != minSrcRow then
      throw s!"{prefixMsg}: neMatmul src1RowBytesA/B ({wSrc1RowA.value}, {wSrc1RowB.value}) != K*chs ({minSrcRow})"
    l2Accs := l2Accs.push { lo := srcL2Base, hi := srcL2Base + src1Row, isWrite := true, role := .src1Stage }

  let wResBase ← requireDecodedWrite task.writes regResBase "resBase"
  let resL2Base := wResBase.value.toNat
  if resIsL2 then
    let expChs := 2 * alignUp n 8
    let expBatch := m * expChs
    let l2Bytes := alignUp (b * expBatch) vDmaLineBytes
    if let some wChs := findDecodedWrite task.writes regResChanStride then
      if wChs.value.toNat != expChs then
        throw s!"{prefixMsg}: neMatmul L2 resChanStride {wChs.value} != expected {expChs}"
    else if m > 1 then
      throw s!"{prefixMsg}: neMatmul L2 res missing resChanStride for M={m} > 1"
    if let some wRow := findDecodedWrite task.writes regResRowStride then
      if wRow.value.toNat != expBatch then
        throw s!"{prefixMsg}: neMatmul L2 resRowStride {wRow.value} != expected {expBatch}"
    vCheckOmitted task.writes #[(regResRowBytesA, "resRowBytesA")] "L2-resident neMatmul res" prefixMsg
    if b > 1 then
      let wB ← requireDecodedWrite task.writes regResRowBytesB "resRowBytesB"
      if wB.value.toNat != expBatch then
        throw s!"{prefixMsg}: neMatmul L2 resRowBytesB {wB.value} != expected {expBatch}"
    else
      vCheckOmitted task.writes #[(regResRowBytesB, "resRowBytesB")] "L2-resident neMatmul res B=1" prefixMsg
    l2Accs := l2Accs.push { lo := resL2Base, hi := resL2Base + l2Bytes, isWrite := true, role := .resL2 }
  else
    let expDirect := vIsDirect target 0 shMax n
    if direct != expDirect then
      throw s!"{prefixMsg}: neMatmul resCfg direct={direct} disagrees with expected direct={expDirect}"
    if direct then
      vCheckOmitted task.writes #[
        (regResChanStride, "resChanStride"), (regResRowStride, "resRowStride"),
        (regResRowBytesA, "resRowBytesA"), (regResRowBytesB, "resRowBytesB")
      ] "direct mode" prefixMsg
      let resBytes := ((1 : Nat) <<< f) * 512
      l2Accs := l2Accs.push { lo := resL2Base, hi := resL2Base + resBytes, isWrite := true, role := .resStage }
    else
      let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
      let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
      let wResRowA ← requireDecodedWrite task.writes regResRowBytesA "resRowBytesA"
      let wResRowB ← requireDecodedWrite task.writes regResRowBytesB "resRowBytesB"
      let expDchs := vDchs n stripW
      if wResChs.value.toNat != expDchs then
        throw s!"{prefixMsg}: neMatmul staged resChanStride {wResChs.value} != expected {expDchs}"
      let minResRow := m * expDchs
      let resRow := wResRow.value.toNat
      if resRow < minResRow || (resRow - minResRow) % 16 != 0 then
        throw s!"{prefixMsg}: neMatmul staged resRowStride {resRow} incompatible with M*dchs={minResRow} + 16*resSkew"
      if wResRowA.value.toNat != minResRow || wResRowB.value.toNat != minResRow then
        throw s!"{prefixMsg}: neMatmul staged resRowBytesA/B ({wResRowA.value}, {wResRowB.value}) != M*dchs ({minResRow})"
      l2Accs := l2Accs.push { lo := resL2Base, hi := resL2Base + resRow, isWrite := true, role := .resStage }

  vCheckTaskL2Accesses l2Accs prefixMsg

  -- 11. Verify DMA deps, formats, strides, and exact bounds for src1 (B) and res (Y)
  let rowBN := alignUp (2 * n) vDmaLineBytes
  if let some (_, tgt, off) := src1DmaTarget? then
    let (dk, dd) ← vDecodeSrcDep info task.writes regSrc1Dep regSrc1DmaCfg "src1Dep" "src1DmaCfg" taskId isFirst prefixMsg
    sync := { sync with src1DepKind := dk, src1DepDist := dd }
    let wFmt ← requireDecodedWrite task.writes regSrc1Fmt "src1Fmt"
    if wFmt.value != srcFmtWordP1 then
      throw s!"{prefixMsg}: src1Fmt {toHex wFmt.value.toNat} != {toHex srcFmtWordP1.toNat}"
    let (resId, cap, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, "invalid")
    if off % vDmaLineBytes != 0 then
      throw s!"{prefixMsg}: neMatmul input DMA {lbl} base offset {toHex off} is not {vDmaLineBytes}-byte aligned"
    if let some wr := findDecodedWrite task.writes regSrc1DmaRowStride then
      if wr.value.toNat < 2 * n || wr.value.toNat % vDmaLineBytes != 0 then
        throw s!"{prefixMsg}: neMatmul src1DmaRowStride ({wr.value}) invalid for W={n}"
    let inPlaneSt ← match findDecodedWrite task.writes regSrc1DmaPlaneStride with
      | some wp =>
        let ps := wp.value.toNat
        if ps < rowBN || ps % vDmaLineBytes != 0 then
          throw s!"{prefixMsg}: neMatmul src1DmaPlaneStride ({ps}) must be >= {rowBN} and 64B-aligned"
        pure ps
      | none =>
        if k > 1 then
          throw s!"{prefixMsg}: neMatmul missing src1DmaPlaneStride for K={k} > 1"
        pure rowBN
    let inNSt ← match findDecodedWrite task.writes regSrc1DmaNStride with
      | some wn =>
        let ns := wn.value.toNat
        if ns < k * inPlaneSt || ns % vDmaLineBytes != 0 then
          throw s!"{prefixMsg}: neMatmul src1DmaNStride ({ns}) must be >= K*planeStride ({k * inPlaneSt}) and 64B-aligned"
        pure ns
      | none =>
        if b > 1 then
          throw s!"{prefixMsg}: neMatmul missing src1DmaNStride for B={b} > 1"
        pure (k * inPlaneSt)
    let inExt := (b - 1) * inNSt + (k - 1) * inPlaneSt + rowBN
    if off + inExt > cap then
      throw s!"{prefixMsg}: neMatmul input DMA range [{toHex off}, {toHex (off + inExt)}) exceeds {lbl} capacity {cap}"
    dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + inExt, isWrite := false, port := .src1Dma }

  if let some (_, tgt, off) := resDmaTarget? then
    match ← resolveBarTarget info barSlotConfigZero with
    | .configZero => pure ()
    | _ => throw s!"{prefixMsg}: BAR slot {barSlotConfigZero} must have VM base 0"
    let wResDep ← requireDecodedWrite task.writes regResDep "resDep"
    if wResDep.barSlot != some barSlotConfigZero then
      throw s!"{prefixMsg}: resDep must use BAR slot {barSlotConfigZero}"
    let expCode := match tgt with
      | .scratch _ => resDepCode target ||| resDepScratchFlags
      | _ => resDepCode target
    if (← decodeFieldChecked resDepLayout wResDep.value "code") != expCode then
      throw s!"{prefixMsg}: resDep code mismatch"
    if (← decodeFieldChecked resDepLayout wResDep.value "task").toNat != taskId then
      throw s!"{prefixMsg}: resDep task != header taskId={taskId}"
    if ((← decodeFieldChecked resDepLayout wResDep.value "direct") == 1) != direct then
      throw s!"{prefixMsg}: resDep direct bit != resCfg direct={direct}"
    let wResFmt ← requireDecodedWrite task.writes regResFmt "resFmt"
    if wResFmt.value != resFmtWordP1 then
      throw s!"{prefixMsg}: resFmt {toHex wResFmt.value.toNat} != {toHex resFmtWordP1.toNat}"
    let (resId, cap, lbl) := match tgt with
      | .surface s => (BarResource.surface (findSurfaceIndex info s) s.shortName, min s.span.toNat s.sect.size.toNat, s!"surface '{s.shortName}'")
      | .scratch sc => (BarResource.scratch, sc.size.toNat, "scratch (__DATA,__bss)")
      | _ => (BarResource.scratch, 0, "invalid")
    if off % vDmaLineBytes != 0 then
      throw s!"{prefixMsg}: neMatmul output DMA {lbl} base offset {toHex off} is not {vDmaLineBytes}-byte aligned"
    if let some wr := findDecodedWrite task.writes regResDmaRowStride then
      if wr.value.toNat < 2 * n || wr.value.toNat % vDmaLineBytes != 0 then
        throw s!"{prefixMsg}: neMatmul resDmaRowStride ({wr.value}) invalid for W={n}"
    let outPlaneSt ← match findDecodedWrite task.writes regResDmaPlaneStride with
      | some wp =>
        let ps := wp.value.toNat
        if ps < rowBN || ps % vDmaLineBytes != 0 then
          throw s!"{prefixMsg}: neMatmul resDmaPlaneStride ({ps}) must be >= {rowBN} and 64B-aligned"
        pure ps
      | none =>
        if m > 1 then
          throw s!"{prefixMsg}: neMatmul missing resDmaPlaneStride for M={m} > 1"
        pure rowBN
    let outNSt ← match findDecodedWrite task.writes regResDmaNStride with
      | some wn =>
        let ns := wn.value.toNat
        if ns < m * outPlaneSt || ns % rowBN != 0 then
          throw s!"{prefixMsg}: neMatmul resDmaNStride ({ns}) must be >= M*planeStride ({m * outPlaneSt}) and a multiple of rowBN ({rowBN})"
        pure ns
      | none =>
        if b > 1 then
          throw s!"{prefixMsg}: neMatmul missing resDmaNStride for B={b} > 1"
        pure (m * outPlaneSt)
    let outExt := (if b > 1 then (b - 1) * outNSt else 0) + (m - 1) * outPlaneSt + rowBN
    if off + outExt > cap then
      throw s!"{prefixMsg}: neMatmul output DMA range [{toHex off}, {toHex (off + outExt)}) exceeds {lbl} capacity {cap}"
    if b > 1 && outPlaneSt == rowBN && outNSt > m * rowBN then
      for bIdx in [:b] do
        let lo := off + bIdx * outNSt
        let hi := lo + m * rowBN
        dramAccs := dramAccs.push {
          resource := resId
          lo := lo
          hi := hi
          isWrite := true
          port := .resDma
          coverage := .dense lo hi
        }
    else
      let outCov := classifyStridedWriteCoverage b 1 m 1 n off outNSt 0 outPlaneSt rowBN
      dramAccs := dramAccs.push { resource := resId, lo := off, hi := off + outExt, isWrite := true, port := .resDma, coverage := outCov }

  -- Reject pairwise intra-task BAR 3 scratch read/write or write/write overlap
  for idx1 in [:dramAccs.size] do
    for idx2 in [idx1 + 1 : dramAccs.size] do
      let a := dramAccs[idx1]!
      let bAcc := dramAccs[idx2]!
      if a.resource == .scratch && bAcc.resource == .scratch && (a.isWrite || bAcc.isWrite) &&
         a.lo < bAcc.hi && bAcc.lo < a.hi then
        throw s!"{prefixMsg}: intra-task BAR 3 scratch overlap between {a.port.name} [{toHex a.lo}, {toHex a.hi}) and {bAcc.port.name} [{toHex bAcc.lo}, {toHex bAcc.hi})"

  -- 12. Output stage and strict mode-conditional register allowlist
  vVerifyOutputStageMulti task.writes taskId isLast prefixMsg

  let mut allowed : Array Nat := #[
    regWin, regHin, regCin, regWout, regHout, regCout, regBatchN,
    regConvCfg, regTH, regNeCommon, regNeTile, regTargetCfg,
    regSrc1Cfg, regSrc1Base,
    regResCfg, regResBase,
    regNeCfg, regNeOp, regNeScale,
    regKernelStage, regKernelBaseLo, regKernelBaseHi
  ]
  if b > 1 then
    allowed := allowed.push regKernelBatchStride
  if isDyn then
    allowed := allowed.push regKernelRoundStride
  if isFirst then
    allowed := allowed.push regSrc2Cfg
  if src1Cfg.isL2 then
    allowed := allowed ++ #[regSrc1ChanStride, regSrc1RowStride]
    if b > 1 then
      allowed := allowed.push regSrc1RowBytesB
  else
    allowed := allowed ++ #[
      regSrc1ChanStride, regSrc1RowStride, regSrc1RowBytesA, regSrc1RowBytesB,
      regSrc1Dep, regSrc1DmaCfg, regSrc1BaseLo, regSrc1BaseHi,
      regSrc1DmaRowStride, regSrc1DmaPlaneStride, regSrc1DmaNStride, regSrc1Fmt
    ]
  if resIsL2 then
    allowed := allowed ++ #[regResChanStride, regResRowStride]
    if b > 1 then
      allowed := allowed.push regResRowBytesB
  else
    allowed := allowed ++ #[
      regResDep, regResBaseLo, regResBaseHi,
      regResDmaRowStride, regResDmaPlaneStride, regResDmaNStride, regResFmt
    ]
    if !direct then
      allowed := allowed ++ #[
        regResChanStride, regResRowStride, regResRowBytesA, regResRowBytesB
      ]
  if isLast then
    allowed := allowed ++ #[regOut0, regOut1, regOut2, regOut3, regOut4, regOut5]
  for s in [:16] do
    if s < numSlots then
      allowed := allowed ++ #[regChunkSlot s, regChunkOffset s, regChunkSize s]
    else if !isFirst || (findDecodedWrite task.writes (regChunkSlot s)).isSome then
      allowed := allowed.push (regChunkSlot s)
  vCheckAllowedRegs task.writes allowed prefixMsg

  let mut barAllowed : Array Nat := #[regKernelBaseLo, regKernelBaseHi]
  if !src1Cfg.isL2 then
    barAllowed := barAllowed ++ #[regSrc1BaseLo, regSrc1BaseHi, regSrc1Dep]
  if !resIsL2 then
    barAllowed := barAllowed ++ #[regResBaseLo, regResBaseHi, regResDep]
  if !produced then
    for s in [:numSlots] do
      barAllowed := barAllowed.push (regChunkSlot s)
  for w in task.writes do
    if let some slot := w.barSlot then
      if !barAllowed.contains w.byteAddr then
        throw s!"{prefixMsg}: neMatmul register {toHex w.byteAddr} has unexpected BAR slot {slot}"

  pure {
    taskIndex := task.taskIndex
    kind := .neMatmul
    engine := .ne
    isNeCompute := true
    isMultiBatch := b > 1
    liveL2Mask := h7
    extProducerTask := extProd?
    l2Accesses := l2Accs
    dramAccesses := dramAccs
    sync := sync
  }

-- ============================================================================
-- S5: Independent Whole-Program Multi-Task Happens-Before & Hazard Checker
-- ============================================================================

private def vIntervalsOverlap (lo1 hi1 lo2 hi2 : Nat) : Bool :=
  lo1 < hi2 && lo2 < hi1

/-- Insert a half-open interval `(lo, hi)` into a list sorted ascending by `lo`. -/
def insertInterval (iv : Nat × Nat) : List (Nat × Nat) → List (Nat × Nat)
  | [] => [iv]
  | hd :: tl =>
    if iv.1 <= hd.1 then iv :: hd :: tl
    else hd :: insertInterval iv tl

/-- Insertion-sort a list of half-open intervals ascending by start address `lo`. -/
def sortIntervals : List (Nat × Nat) → List (Nat × Nat)
  | [] => []
  | hd :: tl => insertInterval hd (sortIntervals tl)

/-- Sweep contiguous coverage frontier starting from `cur` across intervals `ivs`. -/
def sweepCover : List (Nat × Nat) → Nat → Nat
  | [], cur => cur
  | (a, b) :: rest, cur =>
    if a <= cur then
      sweepCover rest (max cur b)
    else
      cur

/-- Return `true` iff the union of half-open intervals `ivs` covers `[lo, hi)` completely. -/
def intervalsCover (ivs : List (Nat × Nat)) (lo hi : Nat) : Bool :=
  if lo >= hi then true
  else sweepCover (sortIntervals ivs) lo >= hi

/-- Check whether periodic inner-block intervals `innerIvs` cover `[0, outerStride)` with
    `outerCount > 0` and `outerStride > 0`, yielding the dense combined byte range
    `[outerBase, outerBase + outerCount * outerStride)`. -/
def periodicGroupCovered
    (outerBase outerCount outerStride : Nat) (innerIvs : List (Nat × Nat)) : Option (Nat × Nat) :=
  if outerCount > 0 && outerStride > 0 && intervalsCover innerIvs 0 outerStride then
    some (outerBase, outerBase + outerCount * outerStride)
  else
    none

/-- Convert exact `WriteCoverage` descriptors into guaranteed-written dense byte intervals:
    - `.dense lo hi` emits `(lo, hi)` directly;
    - `.periodic start innerLen outerCount outerStride` entries with matching `(outerCount, outerStride)`
      are clustered by `start` within one `outerStride` window and emit `(clusterBase, clusterBase + outerCount * outerStride)`
      iff `periodicGroupCovered` proves `[0, outerStride)` is completely covered;
    - `.sparse` contributes nothing. -/
def effectiveCoveredIntervals (covs : List WriteCoverage) : List (Nat × Nat) := Id.run do
  let mut denseIvs : List (Nat × Nat) := []
  let mut periodicKeys : Array (Nat × Nat) := #[]
  let mut periodicEntries : Array (Nat × Nat × Nat × Nat) := #[]
  for c in covs do
    match c with
    | .dense lo hi =>
      if lo < hi then
        denseIvs := (lo, hi) :: denseIvs
    | .periodic start innerLen outerCount outerStride =>
      if innerLen > 0 && outerCount > 0 && outerStride > 0 then
        periodicEntries := periodicEntries.push (start, innerLen, outerCount, outerStride)
        let key := (outerCount, outerStride)
        if !periodicKeys.contains key then
          periodicKeys := periodicKeys.push key
    | .sparse => pure ()

  for (outerCount, outerStride) in periodicKeys do
    let mut group : List (Nat × Nat) := []
    for (st, len, oc, os) in periodicEntries do
      if oc == outerCount && os == outerStride then
        group := insertInterval (st, len) group
    -- Cluster sorted `(start, innerLen)` entries by contiguous `outerStride` windows
    let mut curBase? : Option Nat := none
    let mut curInner : List (Nat × Nat) := []
    for (st, len) in group do
      match curBase? with
      | none =>
        if len <= outerStride then
          curBase? := some st
          curInner := [(0, len)]
      | some base =>
        if st >= base + outerStride then
          if let some iv := periodicGroupCovered base outerCount outerStride curInner then
            denseIvs := iv :: denseIvs
          if len <= outerStride then
            curBase? := some st
            curInner := [(0, len)]
          else
            curBase? := none
            curInner := []
        else
          let rel := st - base
          if rel + len <= outerStride then
            curInner := (rel, rel + len) :: curInner
    if let some base := curBase? then
      if let some iv := periodicGroupCovered base outerCount outerStride curInner then
        denseIvs := iv :: denseIvs

  return denseIvs

private def vLatestEngBefore (fps : Array VerifiedTaskFootprint) (eng : Engine) (i : Nat) : Option Nat := Id.run do
  let mut best : Option Nat := none
  for j in [:i] do
    if h : j < fps.size then
      if fps[j].engine == eng then
        best := some j
  return best

private def vSrc1DramReadsFromWrite (tj ti : VerifiedTaskFootprint) : Bool :=
  ti.dramAccesses.any fun b =>
    b.port == .src1Dma &&
      tj.dramAccesses.any fun a =>
        a.isWrite && a.resource == b.resource && vIntervalsOverlap a.lo a.hi b.lo b.hi

/-- Direct whole-task happens-before edge `stepHB(j, i)` (`j < i`), strictly forward. -/
def stepHB (fps : Array VerifiedTaskFootprint) (j i : Nat) : Bool :=
  if !(j < i && i < fps.size) then
    false
  else
    let tj := fps[j]!
    let ti := fps[i]!
    let latestPe := vLatestEngBefore fps .pe i
    let latestNe := vLatestEngBefore fps .ne i
    -- 1. programOrderPE
    let poPe := ti.engine == .pe && latestPe == some j
    -- 2. programOrderNEcompute (both j and i write 0x4900)
    let poNeComp := ti.engine == .ne && tj.engine == .ne && latestNe == some j && tj.isNeCompute && ti.isNeCompute
    -- 3. wait5 (whole-task wait for all earlier NE tasks)
    let w5 := tj.engine == .ne && (ti.sync.src1WaitNe5 || (ti.engine == .pe && ti.sync.src2WaitNe5))
    -- 4. waitPE (whole-task wait for latest earlier PE task via bit 22)
    let wPe := tj.engine == .pe && latestPe == some j && (ti.sync.src1WaitPe22 || (ti.engine == .pe && ti.sync.src2WaitPe22))
    -- 5. depKindNonRaw (0x4d00 depKind in {1, 2} at depDist == 0 when src1 does NOT read j's DRAM write)
    let depNonRaw :=
      j + 1 == i && ti.sync.hasSrc1Dma &&
      (ti.sync.src1DepKind == 1 || ti.sync.src1DepKind == 2) &&
      ti.sync.src1DepDist == 0 &&
      !vSrc1DramReadsFromWrite tj ti
    poPe || poNeComp || w5 || wPe || depNonRaw

/-- Direct L2-only happens-before edge `stepHBL(j, i)` (`j < i`), strictly forward. -/
def stepHBL (fps : Array VerifiedTaskFootprint) (j i : Nat) : Bool :=
  if !(j < i && i < fps.size) then
    false
  else
    stepHB fps j i ||
      (fps[i]!.engine == .ne && vLatestEngBefore fps .ne i == some j)

/-- Precompute `(latestNeBefore, latestPeBefore)` in O(N) time for all `i < fps.size`. -/
private def vLatestEngArrays (fps : Array VerifiedTaskFootprint) : Array (Option Nat) × Array (Option Nat) := Id.run do
  let n := fps.size
  let mut neArr : Array (Option Nat) := Array.mkEmpty n
  let mut peArr : Array (Option Nat) := Array.mkEmpty n
  let mut lastNe : Option Nat := none
  let mut lastPe : Option Nat := none
  for i in [:n] do
    neArr := neArr.push lastNe
    peArr := peArr.push lastPe
    match fps[i]!.engine with
    | .ne => lastNe := some i
    | .pe => lastPe := some i
  return (neArr, peArr)

/-- Build transitive-closure reachability tables `(hbTbl, hblTbl)` for `stepHB` and `stepHBL`
    by precomputing the direct-edge boolean matrices in O(N^2) time before running `reachTable`. -/
def buildHBReachTables (fps : Array VerifiedTaskFootprint) : Array (Array Bool) × Array (Array Bool) :=
  let n := fps.size
  let (latestNeArr, latestPeArr) := vLatestEngArrays fps
  let hbMat : Array (Array Bool) := Array.ofFn (n := n) fun iFin =>
    let i := iFin.val
    let ti := fps[i]!
    let latestPe := latestPeArr[i]!
    let latestNe := latestNeArr[i]!
    Array.ofFn (n := i) fun jFin =>
      let j := jFin.val
      let tj := fps[j]!
      let poPe := ti.engine == .pe && latestPe == some j
      let poNeComp := ti.engine == .ne && tj.engine == .ne && latestNe == some j && tj.isNeCompute && ti.isNeCompute
      let w5 := tj.engine == .ne && (ti.sync.src1WaitNe5 || (ti.engine == .pe && ti.sync.src2WaitNe5))
      let wPe := tj.engine == .pe && latestPe == some j && (ti.sync.src1WaitPe22 || (ti.engine == .pe && ti.sync.src2WaitPe22))
      let depNonRaw :=
        j + 1 == i && ti.sync.hasSrc1Dma &&
        (ti.sync.src1DepKind == 1 || ti.sync.src1DepKind == 2) &&
        ti.sync.src1DepDist == 0 &&
        !vSrc1DramReadsFromWrite tj ti
      poPe || poNeComp || w5 || wPe || depNonRaw
  let hblMat : Array (Array Bool) := Array.ofFn (n := n) fun iFin =>
    let i := iFin.val
    let ti := fps[i]!
    let latestNe := latestNeArr[i]!
    let hbRow := hbMat[i]!
    Array.ofFn (n := i) fun jFin =>
      let j := jFin.val
      hbRow[j]! || (ti.engine == .ne && latestNe == some j)
  let hbStep := fun j i => if j < i && i < n then (hbMat[i]!)[j]! else false
  let hblStep := fun j i => if j < i && i < n then (hblMat[i]!)[j]! else false
  (reachTable hbStep n, reachTable hblStep n)

/-- Verify Clause S5 (whole-program hazard freedom + live L2-resident integrity) across all `N` tasks. -/
def verifyProgramHazards (fps : Array VerifiedTaskFootprint) : Except String Unit := do
  let n := fps.size
  let (hbTbl, hblTbl) := buildHBReachTables fps
  let (latestNeArr, latestPeArr) := vLatestEngArrays fps
  let hb := fun a b => reachB hbTbl a b
  let hbl := fun a b => reachB hblTbl a b
  let hbEq := fun a b => a == b || hb a b
  let hblEq := fun a b => a == b || hbl a b

  -- 1. Live L2-resident integrity & exact header[7] (liveL2Mask) validation (Clauses S2 & S5):
  --    Every L2-resident read must match an intact prior resL2 write that was not clobbered by any
  --    intermediate task's L2 write; every resL2 write must have at least one matching consumer;
  --    and every non-last task's header[7] must equal the exact 64 KiB block mask of live L2 tensors.
  let mut resL2Spans : Array (Nat × Nat × Nat × Option Nat) := #[]
  for i in [:n] do
    for a in fps[i]!.l2Accesses do
      if a.role == .resL2 then
        resL2Spans := resL2Spans.push (i, a.lo, a.hi, none)

  for i in [:n] do
    let ti := fps[i]!
    for b in ti.l2Accesses do
      if b.role == .src1L2 || b.role == .src2L2 then
        let mut latestWriter? : Option (Nat × VerifiedL2Access) := none
        for k in [:i] do
          let tk := fps[k]!
          for a in tk.l2Accesses do
            if a.isWrite && vIntervalsOverlap a.lo a.hi b.lo b.hi then
              latestWriter? := some (k, a)
        match latestWriter? with
        | none =>
          throw s!"S5 L2 integrity: task {i} {b.role.name} [{toHex b.lo}, {toHex b.hi}) has no prior L2 producer"
        | some (p, a) =>
          if a.role != .resL2 || a.lo != b.lo || a.hi != b.hi then
            throw s!"S5 L2 integrity: task {i} {b.role.name} [{toHex b.lo}, {toHex b.hi}) was overwritten while live by task {p} {a.role.name} [{toHex a.lo}, {toHex a.hi})"
          for idx in [:resL2Spans.size] do
            let (rp, rlo, rhi, rlast?) := resL2Spans[idx]!
            if rp == p && rlo == a.lo && rhi == a.hi then
              let newLast := match rlast? with | some prev => max prev i | none => i
              resL2Spans := resL2Spans.set! idx (rp, rlo, rhi, some newLast)

  let mut liveL2Intervals : Array (Nat × Nat × Nat × Nat) := #[]
  for (rp, rlo, rhi, rlast?) in resL2Spans do
    match rlast? with
    | none =>
      throw s!"S2/S5 dead L2-resident write: task {rp} resL2 [{toHex rlo}, {toHex rhi}) has no matching L2 consumer"
    | some lastC =>
      liveL2Intervals := liveL2Intervals.push (rp, rlo, rhi, lastC)

  if n > 1 then
    for i in [:n - 1] do
      let mut expMask : UInt32 := 0
      for (rp, rlo, rhi, lastC) in liveL2Intervals do
        if rp <= i && i <= lastC then
          expMask := expMask ||| l2BlockMask rlo rhi
      let gotMask := fps[i]!.liveL2Mask
      if gotMask != expMask then
        throw s!"S2 header[7] liveL2Mask mismatch on task {i}: got {toHex gotMask.toNat}, expected {toHex expMask.toNat}"

  -- 2. Pairwise hazard verification across all 0 <= k < i < n
  for i in [1 : n] do
    let ti := fps[i]!
    let latestNe := latestNeArr[i]!
    let latestPe := latestPeArr[i]!

    for k in [:i] do
      let tk := fps[k]!

      -- Common access-level ordering predicates for pair (k, i)
      let dramRawKind : Bool :=
        ti.sync.hasSrc1Dma &&
        (ti.sync.src1DepKind == 1 || ti.sync.src1DepKind == 2) &&
        ti.sync.src1DepDist == 0 &&
        hbEq k (i - 1)

      let src2Kind : Bool :=
        fps[i - 1]!.engine == .pe &&
        ti.sync.hasSrc2Dma &&
        (ti.sync.src2DepKind == 1 || ti.sync.src2DepKind == 2) &&
        ti.sync.src2DepDist == 0 &&
        hbEq k (i - 1)

      let src2Wait22 : Bool :=
        (ti.sync.src1WaitPe22 || ti.sync.src2WaitPe22) &&
        match latestPe with
        | some j => hbEq k j
        | none => false

      let src2Wait5 : Bool :=
        (ti.sync.src1WaitNe5 || ti.sync.src2WaitNe5) &&
        (if tk.engine == .ne then true
         else Id.run do
          let mut ok := false
          for j in [:i] do
            if fps[j]!.engine == .ne && hbEq k j then
              ok := true
              break
          return ok)

      -- A. Check all conflicting L2 pairs
      for a in tk.l2Accesses do
        for b in ti.l2Accesses do
          if (a.isWrite || b.isWrite) && vIntervalsOverlap a.lo a.hi b.lo b.hi then
            let handoff2 : Bool :=
              a.isWrite &&
              ((b.role == .src1L2 && ti.sync.src1HandoffNe) ||
               (b.role == .src2L2 && ti.sync.src2HandoffNe)) &&
              match latestNe with
              | some j => hblEq k j
              | none => false

            let handoff3 : Bool :=
              a.isWrite &&
              b.role == .src1L2 &&
              ti.sync.src1HandoffPe &&
              match latestPe with
              | some j => hblEq k j
              | none => false

            let ordered : Bool :=
              if ti.engine == .pe && (b.role == .src2Stage || b.role == .src2L2) then
                -- src2AheadPE: plain HB/HBL does NOT order PE src2 staging fill or L2 read
                src2Wait22 || src2Wait5 || src2Kind ||
                  (b.role == .src2L2 && handoff2)
              else if b.role == .src1Stage && a.role == .resStage then
                -- srcStageAhead: plain HBL/HB does NOT order NE or PE DMA src1Stage fill over draining resStage
                src2Wait5 || src2Wait22
              else
                hbl k i || handoff2 || handoff3

            if !ordered then
              let xeTag := if tk.engine != ti.engine then " [XE cross-engine]" else ""
              let src2Tag := if ti.engine == .pe && (b.role == .src2Stage || b.role == .src2L2) then " [src2AheadPE]" else ""
              let stageAheadTag := if b.role == .src1Stage && a.role == .resStage then " [srcStageAhead]" else ""
              throw s!"S5 L2 hazard{xeTag}{src2Tag}{stageAheadTag}: unordered conflict between task {k} ({a.role.name} [{toHex a.lo}, {toHex a.hi})) and task {i} ({b.role.name} [{toHex b.lo}, {toHex b.hi}))"

      -- B. Check all conflicting DRAM pairs
      for a in tk.dramAccesses do
        for b in ti.dramAccesses do
          if a.resource == b.resource && (a.isWrite || b.isWrite) && vIntervalsOverlap a.lo a.hi b.lo b.hi then
            let ordered : Bool :=
              if b.port == .kernel then
                -- Scratch-produced dynamic kernel RAW: ordered via 10-word extended header [producerTaskId]
                a.isWrite && match ti.extProducerTask with
                  | some p => hbEq k p
                  | none => false
              else if ti.engine == .ne && b.port == .src1Dma then
                -- srcReadNE: plain HB does NOT order NE src1 DRAM read; requires depKind at depDist == 0
                a.isWrite && dramRawKind
              else if ti.engine == .pe && b.port == .src2Dma then
                -- src2AheadPE: plain HB does NOT order PE src2 DRAM read
                src2Kind || src2Wait22 || src2Wait5 || dramRawKind
              else if a.isWrite && !b.isWrite then
                -- PE src1Dma RAW
                hb k i || dramRawKind
              else
                -- DRAM WAR or WAW (scratch reuse): requires full whole-task HB
                hb k i

            if !ordered then
              let kindTag :=
                if b.port == .kernel then " [kernelRaw]"
                else if ti.engine == .ne && b.port == .src1Dma then " [srcReadNE]"
                else if ti.engine == .pe && b.port == .src2Dma then " [src2AheadPE]"
                else if !a.isWrite && b.isWrite then " [DRAM WAR]"
                else if a.isWrite && b.isWrite then " [DRAM WAW]"
                else " [DRAM RAW]"
              throw s!"S5 DRAM hazard{kindTag}: unordered conflict on {a.resource.name} between task {k} ({a.port.name} [{toHex a.lo}, {toHex a.hi})) and task {i} ({b.port.name} [{toHex b.lo}, {toHex b.hi}))"

            -- BAR 3 DRAM-scratch read coherence: every scratch RAW reader (NE or PE, src1Dma or src2Dma)
            -- must set depKind in {1, 2} on its own reading port (0x4d00 / 0x4d04).
            if a.resource == .scratch && a.isWrite && !b.isWrite then
              if b.port == .src1Dma && !(ti.sync.src1DepKind == 1 || ti.sync.src1DepKind == 2) then
                throw s!"S5 DRAM scratch coherence: task {i} src1Dma reading scratch [{toHex b.lo}, {toHex b.hi}) written by task {k} must set 0x4d00 depKind to 1 or 2 (got {ti.sync.src1DepKind})"
              if b.port == .src2Dma && !(ti.sync.src2DepKind == 1 || ti.sync.src2DepKind == 2) then
                throw s!"S5 DRAM scratch coherence: task {i} src2Dma reading scratch [{toHex b.lo}, {toHex b.hi}) written by task {k} must set 0x4d04 depKind to 1 or 2 (got {ti.sync.src2DepKind})"

  -- 3. BAR 3 DRAM-scratch read integrity (Clause S5):
  --    Every task reading BAR 3 scratch must (a) unconditionally set depKind in {1, 2} on its reading
  --    port (even with no prior overlapping write), and (b) have prior ordered overlapping resDma
  --    writes whose union covers [b.lo, b.hi) completely (rejecting uninitialized or gap reads),
  --    and set depKind == 1 whenever any active (non-superseded) contributing writer has BatchN > 1.
  for i in [:n] do
    let ti := fps[i]!
    for b in ti.dramAccesses do
      if b.resource == .scratch && !b.isWrite then
        if b.port == .src1Dma && !(ti.sync.src1DepKind == 1 || ti.sync.src1DepKind == 2) then
          throw s!"S5 DRAM scratch coherence: task {i} src1Dma reading scratch [{toHex b.lo}, {toHex b.hi}) must set 0x4d00 depKind to 1 or 2 (got {ti.sync.src1DepKind})"
        if b.port == .src2Dma && !(ti.sync.src2DepKind == 1 || ti.sync.src2DepKind == 2) then
          throw s!"S5 DRAM scratch coherence: task {i} src2Dma reading scratch [{toHex b.lo}, {toHex b.hi}) must set 0x4d04 depKind to 1 or 2 (got {ti.sync.src2DepKind})"
        if b.port == .kernel && ti.extProducerTask.isNone then
          throw s!"S5 DRAM scratch coherence: task {i} kernel reading scratch [{toHex b.lo}, {toHex b.hi}) requires extended header extProducerTask"
        let mut priorWriters : Array (Nat × VerifiedDramAccess) := #[]
        for k in [:i] do
          for a in fps[k]!.dramAccesses do
            if a.resource == .scratch && a.port == .resDma && a.isWrite && vIntervalsOverlap a.lo a.hi b.lo b.hi then
              priorWriters := priorWriters.push (k, a)
        if priorWriters.isEmpty then
          throw s!"S5 DRAM scratch integrity: task {i} {b.port.name} reads uninitialized scratch [{toHex b.lo}, {toHex b.hi}) with no prior overlapping resDma write"
        let writerIvs : List (Nat × Nat) := effectiveCoveredIntervals (priorWriters.toList.map (fun (_, a) => a.coverage))
        if !intervalsCover writerIvs b.lo b.hi then
          throw s!"S5 DRAM scratch integrity: task {i} {b.port.name} scratch read [{toHex b.lo}, {toHex b.hi}) is only partially covered by prior resDma writers (uninitialized gap)"
        -- Check [multiBatchRaw] whenever the consumer itself is multi-batch or any active
        -- (not completely superseded by later dense/periodic writers) contributing writer has BatchN > 1.
        if ti.isMultiBatch then
          if b.port == .src1Dma && ti.sync.src1DepKind != 1 then
            throw s!"S5 DRAM scratch coherence [multiBatchRaw]: multi-batch task {i} src1Dma reading scratch [{toHex b.lo}, {toHex b.hi}) must set 0x4d00 depKind to 1 (got {ti.sync.src1DepKind})"
          if b.port == .src2Dma && ti.sync.src2DepKind != 1 then
            throw s!"S5 DRAM scratch coherence [multiBatchRaw]: multi-batch task {i} src2Dma reading scratch [{toHex b.lo}, {toHex b.hi}) must set 0x4d04 depKind to 1 (got {ti.sync.src2DepKind})"
        for idx in [:priorWriters.size] do
          let (p, a) := priorWriters[idx]!
          if fps[p]!.isMultiBatch then
            let mut laterCovs : List WriteCoverage := []
            for idx2 in [idx + 1 : priorWriters.size] do
              let (p2, a2) := priorWriters[idx2]!
              if p2 > p then
                laterCovs := a2.coverage :: laterCovs
            let laterIvs := effectiveCoveredIntervals laterCovs
            let clipLo := max a.lo b.lo
            let clipHi := min a.hi b.hi
            if !intervalsCover laterIvs clipLo clipHi then
              if b.port == .src1Dma && ti.sync.src1DepKind != 1 then
                throw s!"S5 DRAM scratch coherence [multiBatchRaw]: task {i} src1Dma reading multi-batch (BatchN > 1) scratch [{toHex b.lo}, {toHex b.hi}) written by task {p} must set 0x4d00 depKind to 1 (got {ti.sync.src1DepKind})"
              if b.port == .src2Dma && ti.sync.src2DepKind != 1 then
                throw s!"S5 DRAM scratch coherence [multiBatchRaw]: task {i} src2Dma reading multi-batch (BatchN > 1) scratch [{toHex b.lo}, {toHex b.hi}) written by task {p} must set 0x4d04 depKind to 1 (got {ti.sync.src2DepKind})"

  -- 4. Reject pairwise intra-task BAR 3 scratch read/write or write/write overlap on any task
  for i in [:n] do
    let accs := fps[i]!.dramAccesses
    for idx1 in [:accs.size] do
      for idx2 in [idx1 + 1 : accs.size] do
        let a := accs[idx1]!
        let b := accs[idx2]!
        if a.resource == .scratch && b.resource == .scratch && (a.isWrite || b.isWrite) &&
           vIntervalsOverlap a.lo a.hi b.lo b.hi then
          throw s!"S5 intra-task scratch hazard: task {i} has overlapping BAR 3 scratch accesses {a.port.name} [{toHex a.lo}, {toHex a.hi}) and {b.port.name} [{toHex b.lo}, {toHex b.hi})"

  pure ()

-- ============================================================================
-- Unified Whole-Program Entry Points
-- ============================================================================

/-- Verify all recognized tasks decoded from a validated HWX binary (`neConv`, `neAct`, `peOp`)
    against clauses S1–S5 and return per-kind counts.
    When `requireKnownTask` is `true`, rejects any `.unknown` task or empty verified count. -/
def verifyHwxTasksSummary (hwxBytes : ByteArray) (info : HwxInfo) (requireKnownTask : Bool := true) :
    Except String TaskVerifySummary := do
  let textBytes ← readSlice hwxBytes info.textSection.offset.toNat info.textSection.size.toNat
  let tasks ← decodeTaskStream info.target textBytes
  if tasks.size > vMaxTasks then
    throw s!"task count {tasks.size} exceeds hardware limit {vMaxTasks}"
  let kernelBytes? ← match info.kernelSection with
    | some kSec => some <$> readSlice hwxBytes kSec.offset.toNat kSec.size.toNat
    | none => pure none
  let mut nConv := 0
  let mut nAct := 0
  let mut nView := 0
  let mut nMatmul := 0
  let mut nPe := 0
  let mut nPePool := 0
  let mut nPeReduce := 0
  let mut nNePool := 0
  let mut nNeChanRed := 0
  let mut hasPriorNe := false
  let mut hasPriorPe := false
  let mut footprints : Array VerifiedTaskFootprint := Array.mkEmpty tasks.size
  for task in tasks do
    match classifyTask task with
    | .neConv =>
      let fp ← verifyConvTaskMulti tasks.size hasPriorNe hasPriorPe info task kernelBytes?
      footprints := footprints.push fp
      hasPriorNe := true
      nConv := nConv + 1
    | .neAct =>
      let fp ← verifyNeActTaskMulti tasks.size hasPriorNe hasPriorPe info task kernelBytes?
      footprints := footprints.push fp
      hasPriorNe := true
      nAct := nAct + 1
    | .neView =>
      let fp ← verifyNeViewTaskMulti tasks.size hasPriorNe hasPriorPe info task
      footprints := footprints.push fp
      hasPriorNe := true
      nView := nView + 1
    | .neMatmul =>
      let fp ← verifyNeMatmulTaskMulti tasks.size hasPriorNe hasPriorPe info task
      footprints := footprints.push fp
      hasPriorNe := true
      nMatmul := nMatmul + 1
    | .peOp =>
      let fp ← verifyPeTaskMulti tasks.size hasPriorNe hasPriorPe info task
      footprints := footprints.push fp
      hasPriorPe := true
      nPe := nPe + 1
    | .pePool =>
      let fp ← verifyPePoolTaskMulti tasks.size hasPriorNe hasPriorPe info task
      footprints := footprints.push fp
      hasPriorPe := true
      nPePool := nPePool + 1
    | .peReduce =>
      let fp ← verifyPeReduceTaskMulti tasks.size hasPriorNe hasPriorPe info task
      footprints := footprints.push fp
      hasPriorPe := true
      nPeReduce := nPeReduce + 1
    | .nePool =>
      let fp ← verifyNePoolTaskMulti tasks.size hasPriorNe hasPriorPe info task
      footprints := footprints.push fp
      hasPriorNe := true
      nNePool := nNePool + 1
    | .neChannelReduce =>
      let fp ← verifyNeChannelReduceTaskMulti tasks.size hasPriorNe hasPriorPe info task
      footprints := footprints.push fp
      hasPriorNe := true
      nNeChanRed := nNeChanRed + 1
    | .unknown =>
      if requireKnownTask then
        throw s!"task {task.taskIndex}: unrecognized task kind (missing conv, NE activation, view, matmul, pooling/reduction, and PE configuration registers)"
  let total := nConv + nAct + nView + nMatmul + nPe + nPePool + nPeReduce + nNePool + nNeChanRed
  if requireKnownTask && total == 0 then
    throw "HWX contains no recognized tasks"
  if footprints.size == tasks.size && footprints.size > 1 then
    verifyProgramHazards footprints
  pure {
    total := total
    neConv := nConv
    neAct := nAct
    peOp := nPe
    pePool := nPePool
    peReduce := nPeReduce
    nePool := nNePool
    neChannelReduce := nNeChanRed
    neView := nView
    neMatmul := nMatmul
  }

/-- Verify all recognized tasks (`neConv`, `neAct`, `peOp`) decoded from a validated HWX binary. -/
def verifyHwxTasks (hwxBytes : ByteArray) (info : HwxInfo) (requireKnownTask : Bool := true) :
    Except String Nat := do
  let s ← verifyHwxTasksSummary hwxBytes info (requireKnownTask := requireKnownTask)
  pure s.total

/-- Compatibility wrapper verifying all recognized tasks decoded from a validated HWX binary. -/
def verifyConvHwxTasks (hwxBytes : ByteArray) (info : HwxInfo) (requireConvTask : Bool := true) :
    Except String Nat :=
  verifyHwxTasks hwxBytes info (requireKnownTask := requireConvTask)

end Vane.Verify
