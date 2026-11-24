import Vane.Binary
import Vane.TaskStream
import Vane.Hwx
import Vane.Register
import Vane.Tiling

namespace Vane.Verify

open Vane.Binary
open Vane.TaskStream
open Vane.Hwx
open Vane.Register
open Vane.Tiling

/-- DMA cache-line alignment in bytes for surface row strides and kernel chunks. -/
private def dmaLineBytes : Nat := 64

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
  let sameSpec : ConvSpec := {
    kernelSize := 3, mode := .dense, h := h, w := w, cin := 1, cout := 1, stride := stride, pad := .same
  }
  if sameSpec.outH == hout && sameSpec.outW == wout && sameSpec.padTop == pt && sameSpec.padLeft == pl then
    return .same
  if h >= 3 && w >= 3 then
    let validSpec : ConvSpec := {
      kernelSize := 3, mode := .dense, h := h, w := w, cin := 1, cout := 1, stride := stride, pad := .valid
    }
    if validSpec.outH == hout && validSpec.outW == wout && validSpec.padTop == pt && validSpec.padLeft == pl then
      return .valid
  throw s!"3x3 spatial geometry mismatch: input=({h}x{w}), stride={stride}, output=({hout}x{wout}), pad=(pt={pt}, pl={pl}) matches neither same nor valid"

/-- Return `true` iff a decoded task writes NE convolution configuration registers. -/
def isNeConvTask (task : DecodedTask) : Bool :=
  (findDecodedWrite task.writes regConvCfg).isSome ||
  (findDecodedWrite task.writes regNeCommon).isSome ||
  (findDecodedWrite task.writes regNeTile).isSome

/-- Unified standalone static safety verifier for decoded NE conv tasks (`1x1` and `3x3`,
    dense and depthwise, stride 1 and 2). Enforces hardware silicon safety invariants directly
    from decoded HWX bytes (without enforcing chooser-only policy rules such as `F <= ceil_log2(Cout)`). -/
def verifyConvTask (info : HwxInfo) (task : DecodedTask) : Except String Unit := do
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

  -- Checks 0x40[31:6] == 0 via validateWordBits inside decodeFieldChecked
  let ocg := (← decodeFieldChecked neTileLayout wNeTile.value "ocg").toNat
  let fat := (← decodeFieldChecked neTileLayout wNeTile.value "fat").toNat
  let wu := (← decodeFieldChecked neTileLayout wNeTile.value "WU").toNat
  if wu > 3 then
    throw s!"{prefixMsg}: WU={wu} exceeds 3"
  if f + fat > logNE || wu + f + fat != logNE then
    throw s!"{prefixMsg}: WU({wu}) + F({f}) + fat({fat}) != logNE({logNE})"

  -- 4. Resolve bound input and output HWX surfaces via BAR slots
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

  -- 5. Kernel-specific geometry, dwGroups, and wide3x3 safety precondition
  let shape : ConvSpec ←
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
      pure {
        kernelSize := 1, mode := .dense, h := fullH, w := fullW,
        cin := cin, cout := cout, stride := 1, pad := .same
      }
    else if kw == 3 && kh == 3 then
      if sx != sy || (sx != 1 && sx != 2) then
        throw s!"{prefixMsg}: convCfg strides sx={sx}, sy={sy} must be equal and 1 or 2"
      if hx != 1 || vy != 1 then
        throw s!"{prefixMsg}: convCfg hx={hx}, vy={vy} must both be 1"
      let stride := sx
      let padMode ← inferPadMode fullH fullW stride hout wout pt pl
      let mode : Conv3x3Mode ← match findDecodedWrite task.writes regDwGroups with
        | none => pure .dense
        | some wDw =>
          let perGroupIn ← decodeFieldChecked dwGroupsLayout wDw.value "perGroupIn"
          if perGroupIn != 1 then
            throw s!"{prefixMsg}: dwGroups perGroupIn={perGroupIn} must be 1"
          if cin != cout || cin <= 1 then
            throw s!"{prefixMsg}: dwGroups (0x0030) requires Cin == Cout > 1 (got Cin={cin}, Cout={cout})"
          pure .depthwise
      let sh : ConvSpec := {
        kernelSize := 3, mode := mode, h := fullH, w := fullW,
        cin := cin, cout := cout, stride := stride, pad := padMode
      }
      if winReg != sh.winReg then
        throw s!"{prefixMsg}: Win={winReg} != expected {sh.winReg} (W={fullW}, cropW={sh.cropW}, s={stride})"
      if hinReg != sh.hinReg then
        throw s!"{prefixMsg}: Hin={hinReg} != expected {sh.hinReg} (H={fullH}, cropH={sh.cropH}, s={stride})"
      if mode == .depthwise && ocg != 0 then
        throw s!"{prefixMsg}: dwGroups (0x0030) depthwise mode requires ocg=0, got ocg={ocg}"
      if target == .h17 && sh.effMode == .dense && stride == 1 && ocg > 3 then
        throw s!"{prefixMsg}: h17 stride-1 dense ocg={ocg} exceeds maximum 3"
      if ocg > 4 then
        throw s!"{prefixMsg}: ocg={ocg} exceeds maximum 4"
      pure sh
    else
      throw s!"{prefixMsg}: unsupported conv kernel size kh={kh}, kw={kw} (expected 1x1 or 3x3)"

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

  -- Verify wide3x3 bit 27 safety precondition (wide3x3=0 is legal; wide3x3=1 requires H17 dense 3x3 s=1 and maxBlockSlots <= 32768*rep)
  if shape.kernelSize == 3 && wide3x3Bit == 1 then
    if target != .h17 then
      throw s!"{prefixMsg}: neCommon wide3x3=1 is illegal on {target.name}"
    if shape.effMode != .dense then
      throw s!"{prefixMsg}: neCommon wide3x3=1 is illegal for depthwise 3x3 conv"
    if shape.stride != 1 then
      throw s!"{prefixMsg}: neCommon wide3x3=1 is illegal for stride-{shape.stride} 3x3 conv"
    let rep := ((1 : Nat) <<< logNE) >>> f
    let mbs := maxBlockSlotsConv shape probeTiling
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

  let expectedChs := chsBytesConv shape probeTiling
  if wSrc1Chs.value.toNat != expectedChs then
    throw s!"{prefixMsg}: src1ChanStride {wSrc1Chs.value} != expected chs {expectedChs} (srcPx narrowing forbidden)"

  let minSrcRow := cin * expectedChs
  if wSrc1Row.value.toNat < minSrcRow || (wSrc1Row.value.toNat - minSrcRow) % 16 != 0 then
    throw s!"{prefixMsg}: src1RowStride {wSrc1Row.value} incompatible with Cin*chs={minSrcRow} + 16*srcSkew"
  if wSrc1RowA.value.toNat != minSrcRow || wSrc1RowB.value.toNat != minSrcRow then
    throw s!"{prefixMsg}: src1RowBytesA/B ({wSrc1RowA.value}, {wSrc1RowB.value}) != Cin*chs ({minSrcRow})"
  let srcSkew := (wSrc1Row.value.toNat - minSrcRow) / 16

  let expectedDirect := isDirectConv target shape probeTiling
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
      let expDchs := dchsBytesConv shape probeTiling
      if wResChs.value.toNat != expDchs then
        throw s!"{prefixMsg}: resChanStride {wResChs.value} != expected dchs {expDchs}"
      let minResRow := cout * expDchs
      if wResRow.value.toNat < minResRow || (wResRow.value.toNat - minResRow) % 16 != 0 then
        throw s!"{prefixMsg}: resRowStride {wResRow.value} incompatible with Cout*dchs={minResRow} + 16*resSkew"
      if wResRowA.value.toNat != minResRow || wResRowB.value.toNat != minResRow then
        throw s!"{prefixMsg}: resRowBytesA/B ({wResRowA.value}, {wResRowB.value}) != Cout*dchs ({minResRow})"
      pure ((wResRow.value.toNat - minResRow) / 16)

  -- Enforce silicon safety invariants (not chooser-only policy rules)
  if th == 0 || th > hout then
    throw s!"{prefixMsg}: tileHeight={th} outside 1..{hout}"
  let tiling : Tiling := { probeTiling with srcSkew := srcSkew, resSkew := resSkew }
  validateConvSafety target shape tiling

  let srcBase := wSrc1Base.value.toNat
  let srcBytes := inRowsConv shape th * wSrc1Row.value.toNat
  let resBase := wResBase.value.toNat
  let resBytes :=
    if direct then
      ((1 : Nat) <<< (f + effOcgConv shape tiling + fat)) * 512
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
  let expectedSparse : UInt32 := if shape.isSparse then 1 else 0
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
    throw s!"{prefixMsg}: sparse flag {neCfgSparse} disagrees with stride {shape.stride} (expected {expectedSparse})"

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

  -- 8. Verify active kernel chunks, 64 KiB cap, and Mach-O relocation
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

  -- 9. Verify DMA strides & surface span bounds (full uncropped input H, W and output Ho, Wo)
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

/-- Verify a decoded task is specifically a legal 1x1 conv task. -/
def verifyConv1x1Task (info : HwxInfo) (task : DecodedTask) : Except String Unit := do
  let wConvCfg ← requireDecodedWrite task.writes regConvCfg "convCfg"
  let kw := (← decodeFieldChecked convCfgLayout wConvCfg.value "kw").toNat
  let kh := (← decodeFieldChecked convCfgLayout wConvCfg.value "kh").toNat
  if kw != 1 || kh != 1 then
    throw s!"task {task.taskIndex}: expected 1x1 conv, got {kh}x{kw}"
  verifyConvTask info task

/-- Verify a decoded task is specifically a legal 3x3 conv task. -/
def verifyConv3x3Task (info : HwxInfo) (task : DecodedTask) : Except String Unit := do
  let wConvCfg ← requireDecodedWrite task.writes regConvCfg "convCfg"
  let kw := (← decodeFieldChecked convCfgLayout wConvCfg.value "kw").toNat
  let kh := (← decodeFieldChecked convCfgLayout wConvCfg.value "kh").toNat
  if kw != 3 || kh != 3 then
    throw s!"task {task.taskIndex}: expected 3x3 conv, got {kh}x{kw}"
  verifyConvTask info task

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

end Vane.Verify
