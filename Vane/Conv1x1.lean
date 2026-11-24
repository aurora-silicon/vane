import Vane.Binary
import Vane.TaskStream
import Vane.Hwx
import Vane.Sections
import Vane.Register
import Vane.Tiling

namespace Vane.Conv1x1

open Vane.Binary
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Tiling

/-- Size of `__TEXT,__const` for compiled programs (16 KiB of zero bytes). -/
def textConstBytes : Nat := 0x4000

/-- DMA cache-line alignment in bytes for surface row strides and kernel chunks. -/
def dmaLineBytes : Nat := 64

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

/-- Complete compilation artifact for a single-task fp16 1x1 convolution. -/
structure CompiledConv1x1 where
  target : Target
  shape : Conv1x1Shape
  tiling : Tiling
  hasBias : Bool
  relu : Bool
  direct : Bool
  l2UseBytes : Nat
  kernelBytes : Nat
  hwx : ByteArray
  info : HwxInfo
  sections : LoadSections

/-- Compute the output channel blocks `(o0, count)` assigned to group `g` (`0 <= g < 2^F`). -/
def groupOutputBlocks (cout f ocg g : Nat) : Array (Nat × Nat) := Id.run do
  let groups := (1 : Nat) <<< f
  let blk := (1 : Nat) <<< ocg
  let round := groups * blk
  let full := cout / round
  let rem := cout % round
  let mut blocks : Array (Nat × Nat) := #[]
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

/-- Pack little-endian fp16 weights `(Cout, Cin)` row-major and optional bias `(Cout,)` into `__KERN_0`. -/
def packKernelChunks
    (target : Target)
    (shape : Conv1x1Shape)
    (tiling : Tiling)
    (weights : ByteArray)
    (bias : Option ByteArray) : Except String PackedKernel := do
  validateTiling target shape tiling

  let expectedWeightBytes := 2 * shape.cout * shape.cin
  if weights.size != expectedWeightBytes then
    throw s!"weights byte size {weights.size} != expected {expectedWeightBytes} (2 * Cout={shape.cout} * Cin={shape.cin})"

  if let some b := bias then
    let expectedBiasBytes := 2 * shape.cout
    if b.size != expectedBiasBytes then
      throw s!"bias byte size {b.size} != expected {expectedBiasBytes} (2 * Cout={shape.cout})"

  let hasBias := bias.isSome
  let biasInKernel := hasBias && (shape.cout > 1)
  let scalarBiasBits ←
    if shape.cout == 1 then
      match bias with
      | some b => readU16LE b 0
      | none => pure (0 : UInt16)
    else
      pure (0 : UInt16)

  let groups := (1 : Nat) <<< tiling.f
  let rep := ((1 : Nat) <<< tiling.logNE) >>> tiling.f
  if rep == 0 || groups * rep > 16 then
    throw s!"invalid chunk replication geometry: groups={groups}, rep={rep}"

  let rowsK := shape.cin + (if biasInKernel then 1 else 0)
  let mut kernelBuf := ByteArray.empty
  let mut slots : Array KernelChunkSlot := #[]

  for g in [:groups] do
    let blocks := groupOutputBlocks shape.cout tiling.f tiling.ocg g
    if !blocks.isEmpty then
      let totalOuts := blocks.foldl (fun acc (_, n) => acc + n) 0
      let expectedPadded := alignUp (2 * rowsK * totalOuts) dmaLineBytes
      let mut chunk := ByteArray.empty
      for (o0, n) in blocks do
        if biasInKernel then
          let b := bias.get!
          for k in [:n] do
            let bOff := 2 * (o0 + k)
            chunk := chunk.push (b.get! bOff)
            chunk := chunk.push (b.get! (bOff + 1))
        for c in [:shape.cin] do
          for k in [:n] do
            let o := o0 + k
            let wOff := 2 * (o * shape.cin + c)
            chunk := chunk.push (weights.get! wOff)
            chunk := chunk.push (weights.get! (wOff + 1))
      chunk := padToAlign chunk dmaLineBytes
      if chunk.size != expectedPadded then
        throw s!"internal error: group {g} chunk size {chunk.size} != expected {expectedPadded}"

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
    throw "kernel packer produced 0 active chunk slots"

  pure {
    bytes := kernelBuf
    slots := slots
    biasInKernel := biasInKernel
    scalarBiasBits := scalarBiasBits
  }

-- ============================================================================
-- Post-emission static task-legality verifier (operates on decoded HWX bytes)
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

/-- Return `true` iff a decoded task writes NE convolution configuration registers. -/
def isNeConvTask (task : DecodedTask) : Bool :=
  (findDecodedWrite task.writes regConvCfg).isSome ||
  (findDecodedWrite task.writes regNeCommon).isSome ||
  (findDecodedWrite task.writes regNeTile).isSome

/-- Statically verify one decoded NE 1x1 conv task against hardware register bounds,
    tiling legality, 2 MiB L2 capacity/disjointness, chunk-slot invariants, and DMA surface bounds. -/
def verifyConv1x1Task (info : HwxInfo) (task : DecodedTask) : Except String Unit := do
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

  -- 3. Decode common shape & tiling registers with full bitfield checks
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

  let win := (← decodeFieldChecked (dim16Layout "Win") wWin.value "val").toNat
  let hin := (← decodeFieldChecked (dim16Layout "Hin") wHin.value "val").toNat
  let cin := wCin.value.toNat
  let wout := (← decodeFieldChecked (dim16Layout "Wout") wWout.value "val").toNat
  let hout := (← decodeFieldChecked (dim16Layout "Hout") wHout.value "val").toNat
  let cout := wCout.value.toNat
  let th := (← decodeFieldChecked (dim16Layout "TH") wTH.value "val").toNat

  if wout != win || hout != hin then
    throw s!"{prefixMsg}: output spatial dims ({hout}x{wout}) != input ({hin}x{win}) for 1x1 conv"

  -- Validate convCfg fields for 1x1 stride-1 conv
  let kw ← decodeFieldChecked convCfgLayout wConvCfg.value "kw"
  let kh ← decodeFieldChecked convCfgLayout wConvCfg.value "kh"
  let sx ← decodeFieldChecked convCfgLayout wConvCfg.value "sx"
  let sy ← decodeFieldChecked convCfgLayout wConvCfg.value "sy"
  let pl ← decodeFieldChecked convCfgLayout wConvCfg.value "pl"
  let pt ← decodeFieldChecked convCfgLayout wConvCfg.value "pt"
  let hx ← decodeFieldChecked convCfgLayout wConvCfg.value "hx"
  let vy ← decodeFieldChecked convCfgLayout wConvCfg.value "vy"
  if kw != 1 || kh != 1 || sx != 1 || sy != 1 || pl != 0 || pt != 0 || hx != 1 || vy != 1 then
    throw s!"{prefixMsg}: convCfg {toHex wConvCfg.value.toNat} is not a stride-1 1x1 convolution"

  let kind := (← decodeFieldChecked neCommonLayout wNeCommon.value "kind").toNat
  let shPref := (← decodeFieldChecked neCommonLayout wNeCommon.value "shPref").toNat
  let shMin := (← decodeFieldChecked neCommonLayout wNeCommon.value "shMin").toNat
  let shMax := (← decodeFieldChecked neCommonLayout wNeCommon.value "shMax").toNat
  let f := (← decodeFieldChecked neCommonLayout wNeCommon.value "F").toNat
  if kind != 0 then
    throw s!"{prefixMsg}: neCommon kind={kind} != 0"
  if shMin != shPref then
    throw s!"{prefixMsg}: neCommon shMin={shMin} != shPref={shPref}"

  -- Checks 0x40[31:6] == 0 via validateWordBits inside decodeFieldChecked
  let ocg := (← decodeFieldChecked neTileLayout wNeTile.value "ocg").toNat
  let fat := (← decodeFieldChecked neTileLayout wNeTile.value "fat").toNat
  let wu := (← decodeFieldChecked neTileLayout wNeTile.value "WU").toNat
  if wu > 3 then
    throw s!"{prefixMsg}: WU={wu} exceeds 3"
  if f + fat > logNE || wu + f + fat != logNE then
    throw s!"{prefixMsg}: WU({wu}) + F({f}) + fat({fat}) != logNE({logNE})"

  if wTargetCfg.value != targetCfgValue target then
    throw s!"{prefixMsg}: targetCfg {toHex wTargetCfg.value.toNat} != expected {toHex (targetCfgValue target).toNat}"

  let shape : Conv1x1Shape := { h := hin, w := win, cin := cin, cout := cout }

  -- 4. Decode L2 staging registers & check non-overlapping [0, 2 MiB) ranges
  let wSrc1Base ← requireDecodedWrite task.writes regSrc1Base "src1Base"
  let wSrc1Chs ← requireDecodedWrite task.writes regSrc1ChanStride "src1ChanStride"
  let wSrc1Row ← requireDecodedWrite task.writes regSrc1RowStride "src1RowStride"
  let wResCfg ← requireDecodedWrite task.writes regResCfg "resCfg"
  let wResBase ← requireDecodedWrite task.writes regResBase "resBase"

  let directBit ← decodeFieldChecked resCfgLayout wResCfg.value "direct"
  let direct := directBit == 1

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

  let expectedChs := chsBytes shape probeTiling
  if wSrc1Chs.value.toNat != expectedChs then
    throw s!"{prefixMsg}: src1ChanStride {wSrc1Chs.value} != expected chs {expectedChs}"

  let minSrcRow := cin * expectedChs
  if wSrc1Row.value.toNat < minSrcRow || (wSrc1Row.value.toNat - minSrcRow) % 16 != 0 then
    throw s!"{prefixMsg}: src1RowStride {wSrc1Row.value} incompatible with Cin*chs={minSrcRow} + 16*srcSkew"
  let srcSkew := (wSrc1Row.value.toNat - minSrcRow) / 16

  let expectedDirect := isDirect target shape probeTiling
  if direct != expectedDirect then
    throw s!"{prefixMsg}: resCfg direct={direct} disagrees with target/tiling direct={expectedDirect}"

  let resSkew ←
    if direct then
      pure 0
    else
      let wResChs ← requireDecodedWrite task.writes regResChanStride "resChanStride"
      let wResRow ← requireDecodedWrite task.writes regResRowStride "resRowStride"
      let _ ← requireDecodedWrite task.writes regResRowBytesA "resRowBytesA"
      let _ ← requireDecodedWrite task.writes regResRowBytesB "resRowBytesB"
      let expDchs := dchsBytes shape probeTiling
      if wResChs.value.toNat != expDchs then
        throw s!"{prefixMsg}: resChanStride {wResChs.value} != expected dchs {expDchs}"
      let minResRow := cout * expDchs
      if wResRow.value.toNat < minResRow || (wResRow.value.toNat - minResRow) % 16 != 0 then
        throw s!"{prefixMsg}: resRowStride {wResRow.value} incompatible with Cout*dchs={minResRow} + 16*resSkew"
      pure ((wResRow.value.toNat - minResRow) / 16)

  let tiling : Tiling := { probeTiling with srcSkew := srcSkew, resSkew := resSkew }
  validateTiling target shape tiling

  let srcBase := wSrc1Base.value.toNat
  let srcBytes := th * wSrc1Row.value.toNat
  let resBase := wResBase.value.toNat
  let resBytes :=
    if direct then
      ((1 : Nat) <<< (f + ocg + fat)) * 512
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

  -- 5. Verify NE block config
  let wNeCfg ← requireDecodedWrite task.writes regNeCfg "neCfg"
  if wNeCfg.value != neCfgValue target then
    throw s!"{prefixMsg}: neCfg {toHex wNeCfg.value.toNat} != expected {toHex (neCfgValue target).toNat}"
  let wNeOp ← requireDecodedWrite task.writes regNeOp "neOp"
  validateWordBits neOpLayout wNeOp.value

  -- 6. Verify active kernel chunks & relocation
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
      if cOff + cSz > kSec.size.toNat then
        throw s!"{prefixMsg}: chunk slot {s} range [{cOff}, {cOff + cSz}) exceeds kernel size {kSec.size}"
      activeChunks := activeChunks + 1
    else if offW?.isSome then
      throw s!"{prefixMsg}: chunk slot {s} writes offset without enable bit 0 in {toHex (regChunkSlot s)}"

  if activeChunks == 0 then
    throw s!"{prefixMsg}: no active kernel chunk slots enabled"

  -- 7. Verify DMA extents fit inside bound input and output surfaces
  let wSrcDep ← requireDecodedWrite task.writes regSrc1Dep "src1Dep"
  validateWordBits src1DepLayout wSrcDep.value
  let wResDep ← requireDecodedWrite task.writes regResDep "resDep"
  let resDepDirect ← decodeFieldChecked resDepLayout wResDep.value "direct"
  if (resDepDirect == 1) != direct then
    throw s!"{prefixMsg}: resDep direct bit {resDepDirect} != resCfg direct={direct}"

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

  let wInRow ← requireDecodedWrite task.writes regSrc1DmaRowStride "src1DmaRowStride"
  let inRowStride := wInRow.value.toNat
  if inRowStride < 2 * win || inRowStride % dmaLineBytes != 0 then
    throw s!"{prefixMsg}: input DMA row stride {inRowStride} must be >= {2 * win} and {dmaLineBytes}-byte aligned"
  let inExtent ←
    if cin == 1 then
      pure (hin * inRowStride)
    else
      let wInPlane ← requireDecodedWrite task.writes regSrc1DmaPlaneStride "src1DmaPlaneStride"
      let inPlaneStride := wInPlane.value.toNat
      if inPlaneStride < hin * inRowStride then
        throw s!"{prefixMsg}: input DMA plane stride {inPlaneStride} < H * rowStride ({hin * inRowStride})"
      pure (cin * inPlaneStride)
  if inExtent > inSurf.span.toNat || inExtent > inSurf.sect.size.toNat then
    throw s!"{prefixMsg}: input DMA extent {inExtent} exceeds bound surface '{inSurf.shortName}' (span={inSurf.span}, section={inSurf.sect.size})"

  let wOutRow ← requireDecodedWrite task.writes regResDmaRowStride "resDmaRowStride"
  let outRowStride := wOutRow.value.toNat
  if outRowStride < 2 * wout || outRowStride % dmaLineBytes != 0 then
    throw s!"{prefixMsg}: output DMA row stride {outRowStride} must be >= {2 * wout} and {dmaLineBytes}-byte aligned"
  let outExtent ←
    if cout == 1 then
      pure (hout * outRowStride)
    else
      let wOutPlane ← requireDecodedWrite task.writes regResDmaPlaneStride "resDmaPlaneStride"
      let outPlaneStride := wOutPlane.value.toNat
      if outPlaneStride < hout * outRowStride then
        throw s!"{prefixMsg}: output DMA plane stride {outPlaneStride} < H * rowStride ({hout * outRowStride})"
      pure (cout * outPlaneStride)
  if outExtent > outSurf.span.toNat || outExtent > outSurf.sect.size.toNat then
    throw s!"{prefixMsg}: output DMA extent {outExtent} exceeds bound surface '{outSurf.shortName}' (span={outSurf.span}, section={outSurf.sect.size})"

  pure ()

/-- Verify all NE 1x1 conv tasks decoded from a validated HWX binary.
    When `requireConvTask` is `true`, fails if no NE conv task is present. -/
def verifyHwxTasks (hwxBytes : ByteArray) (info : HwxInfo) (requireConvTask : Bool := true) : Except String Nat := do
  let textBytes ← readSlice hwxBytes info.textSection.offset.toNat info.textSection.size.toNat
  let tasks ← decodeTaskStream info.target textBytes
  let mut checked := 0
  for task in tasks do
    if isNeConvTask task then
      verifyConv1x1Task info task
      checked := checked + 1
    else if requireConvTask then
      throw s!"task {task.taskIndex}: missing NE 1x1 conv configuration registers"
  if requireConvTask && checked == 0 then
    throw "HWX contains no NE 1x1 conv tasks"
  pure checked

-- ============================================================================
-- Single-task fp16 1x1 convolution compiler
-- ============================================================================

/-- Compile a single-task fp16 1x1 convolution into a statically verified HWX binary
    and its 4 host-built LOAD sections. -/
def compileConv1x1
    (target : Target)
    (shape : Conv1x1Shape)
    (weights : ByteArray)
    (bias : Option ByteArray := none)
    (relu : Bool := false)
    (tilingOverride : Option Tiling := none) : Except String CompiledConv1x1 := do
  validateShape shape
  let tiling ← match tilingOverride with
    | some t => validateTiling target shape t; pure t
    | none => chooseTiling target shape

  let packed ← packKernelChunks target shape tiling weights bias
  let direct := isDirect target shape tiling
  let chs := chsBytes shape tiling
  let dchs := dchsBytes shape tiling
  let srcRow := srcRowStride shape tiling
  let resRow := resRowStride shape tiling
  let srcL2 := srcL2Bytes shape tiling
  let l2Use := l2UseBytes target shape tiling

  let rowStride := alignUp (2 * shape.w) dmaLineBytes
  let planeStride := shape.h * rowStride
  let inSpan := shape.cin * planeStride
  let outSpan := shape.cout * planeStride

  -- Encode every active register explicitly (no omission of reset-valued active registers)
  let wWin ← encodeWord (dim16Layout "Win") #[("val", shape.w.toUInt32)]
  let wHin ← encodeWord (dim16Layout "Hin") #[("val", shape.h.toUInt32)]
  let wWout ← encodeWord (dim16Layout "Wout") #[("val", shape.w.toUInt32)]
  let wHout ← encodeWord (dim16Layout "Hout") #[("val", shape.h.toUInt32)]
  let wConvCfg ← encodeWord convCfgLayout #[
    ("kw", 1), ("kh", 1), ("sx", 1), ("sy", 1),
    ("pl", 0), ("pt", 0), ("hx", 1), ("vy", 1)
  ]
  let wTH ← encodeWord (dim16Layout "TH") #[("val", tiling.tileHeight.toUInt32)]
  let wNeCommon ← encodeWord neCommonLayout #[
    ("kind", 0),
    ("shPref", tiling.shPref.toUInt32),
    ("shMin", tiling.shPref.toUInt32),
    ("shMax", tiling.shMax.toUInt32),
    ("F", tiling.f.toUInt32)
  ]
  let wNeTile ← encodeWord neTileLayout #[
    ("ocg", tiling.ocg.toUInt32),
    ("fat", tiling.fat.toUInt32),
    ("WU", tiling.wu.toUInt32)
  ]
  let wSrc1Cfg ← encodeWord src1CfgLayout #[("mode", 2), ("P", 1)]
  let wResCfg ← encodeWord resCfgLayout #[
    ("mode", 2),
    ("direct", if direct then 1 else 0),
    ("P", 1)
  ]
  let wNeOp ← encodeWord neOpLayout #[
    ("biasInKernel", if packed.biasInKernel then 1 else 0),
    ("relu", if relu then 1 else 0)
  ]
  let wNeScale ← encodeWord neScaleLayout #[("val", fp16OneBits)]
  let wSrc1Fmt ← encodeWord (dmaFmtLayout "src1Fmt") #[("format", 0x002031), ("P", 1)]
  let wResFmt ← encodeWord (dmaFmtLayout "resFmt") #[("format", 0x302031), ("P", 1)]
  let wKernStage ← encodeWord kernelStageLayout #[("form", 0x2c0), ("stage", 1)]
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
    { byteAddr := regNeCfg, value := neCfgValue target },
    { byteAddr := regNeOp, value := wNeOp },
    { byteAddr := regNeScale, value := wNeScale },
    -- Tile DMA (Source) block (non-BAR)
    { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
    { byteAddr := regSrc1DmaRowStride, value := rowStride.toUInt32 },
    { byteAddr := regSrc1DmaPlaneStride, value := planeStride.toUInt32 },
    { byteAddr := regSrc1Fmt, value := wSrc1Fmt },
    -- Result DMA block (non-BAR)
    { byteAddr := regResDmaRowStride, value := rowStride.toUInt32 },
    { byteAddr := regResDmaPlaneStride, value := planeStride.toUInt32 },
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

  -- Staged mode result L2 registers (omitted only in direct mode)
  if !direct then
    plain := plain ++ #[
      { byteAddr := regResChanStride, value := dchs.toUInt32 },
      { byteAddr := regResRowStride, value := resRow.toUInt32 },
      { byteAddr := regResRowBytesA, value := (shape.cout * dchs).toUInt32 },
      { byteAddr := regResRowBytesB, value := (shape.cout * dchs).toUInt32 }
    ]

  -- Scalar bias register (written when Cout == 1, omitted when Cout > 1)
  if shape.cout == 1 then
    let wScalarBias ← encodeWord scalarBiasLayout #[("val", packed.scalarBiasBits.toUInt32)]
    plain := plain.push { byteAddr := regScalarBias, value := wScalarBias }

  -- Active kernel chunk table slots (inactive slots omitted)
  for cs in packed.slots do
    plain := plain ++ #[
      { byteAddr := regChunkSlot cs.slot, value := wChunkSlotVal },
      { byteAddr := regChunkOffset cs.slot, value := cs.offset.toUInt32 },
      { byteAddr := regChunkSize cs.slot, value := cs.size.toUInt32 }
    ]

  -- BAR-tagged sequential records in canonical spec order
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
  let kernelRecIdx := plainRecs.size -- BAR-8 [0x5510, 0x5514] is first BAR record
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
    rowStride := rowStride.toUInt64
    planeStride := planeStride.toUInt64
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
    height := shape.h.toUInt32
    width := shape.w.toUInt32
    rowStride := rowStride.toUInt64
    planeStride := planeStride.toUInt64
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
  let _ ← verifyHwxTasks hwxBytes info (requireConvTask := true)

  pure {
    target := target
    shape := shape
    tiling := tiling
    hasBias := bias.isSome
    relu := relu
    direct := direct
    l2UseBytes := l2Use
    kernelBytes := packed.bytes.size
    hwx := hwxBytes
    info := info
    sections := secs
  }

end Vane.Conv1x1
