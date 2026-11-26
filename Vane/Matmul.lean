import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Hwx
import Vane.Sections
import Vane.Register
import Vane.Alloc
import Vane.Tiling
import Vane.Program
import Vane.Verify
import Vane.View

namespace Vane.Matmul

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Alloc
open Vane.Tiling
open Vane.Program
open Vane.Verify
open Vane.View

/-- Specification for batched dynamic runtime-kernel matrix multiplication:
    `Y[b, m, n] = scale * sum_{k=0}^{K-1} A_eff[b, m, k] * B_eff[b, k, n]`
    on 4D ANE fp16 tensors (`A` in `[B, M, 1, K]` or `[B, K, 1, M]` if `transposeA`,
    `B` in `[B, K, 1, N]` or `[B, N, 1, K]` if `transposeB`, `Y` in `[B, M, 1, N]`). -/
structure MatmulSpec where
  b : Nat := 1
  m : Nat
  k : Nat
  n : Nat
  transposeA : Bool := false
  transposeB : Bool := false
  scaleBits : UInt16 := f16PosOne
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Storage location for a `neMatmul` activation (`B`) or output (`Y`) port. -/
inductive MatmulPortLoc where
  | surface (barSlot : UInt8) (offset : Nat := 0)
  | dram (offset : Nat)
  | l2 (base : Nat)
  deriving Repr, BEq, Inhabited

/-- Tiling and L2 staging plan for a single dynamic runtime-kernel `neMatmul` task. -/
structure MatmulTaskPlan where
  tiling : Tiling
  direct : Bool
  srcStageBase : Nat
  resStageBase : Nat
  l2TaskPeak : Nat
  deriving Repr, BEq, Inhabited

/-- Complete compiled artifact for standalone `compileMatmul`. -/
structure CompiledMatmul where
  target : Target
  spec : MatmulSpec
  taskCount : Nat
  tiling : Tiling
  direct : Bool
  l2PeakBytes : Nat
  scratchBytes : Nat
  hwx : ByteArray
  info : HwxInfo
  sections : LoadSections

/-- Dynamic kernel chunk size in bytes for inner dimension `K` (`G = 1`). -/
def matmulChunkSize (k : Nat) : Nat :=
  alignUp (2 * k) dmaLineBytes

/-- Validate `MatmulSpec` dimensions, scale finiteness, 64 KiB chunk cap, and T-mode transpose limits. -/
def validateMatmulSpec (spec : MatmulSpec) : Except String Unit := do
  if spec.b == 0 || spec.b > 65535 then
    throw s!"matmul: B={spec.b} must be in 1..65535"
  if spec.m == 0 || spec.m > 2048 then
    throw s!"matmul: M={spec.m} must be in 1..2048"
  if spec.k == 0 || spec.k > 65535 then
    throw s!"matmul: K={spec.k} must be in 1..65535"
  if spec.n == 0 || spec.n > 65535 then
    throw s!"matmul: N={spec.n} must be in 1..65535"
  if !f16IsFinite spec.scaleBits then
    throw s!"matmul: scale ({toHex spec.scaleBits.toNat}) must be a finite IEEE-754 fp16 value"
  let chunkSize := matmulChunkSize spec.k
  if chunkSize > maxKernelChunkBytes then
    throw s!"matmul: chunkSize={chunkSize}B for K={spec.k} exceeds hardware 64 KiB cap ({maxKernelChunkBytes}B)"
  if spec.transposeA && spec.k > 128 then
    throw s!"matmul: transposeA requires K={spec.k} <= 128 for single-task T-mode transpose"
  if spec.transposeB && spec.n > 128 then
    throw s!"matmul: transposeB requires N={spec.n} <= 128 for single-task T-mode transpose"
  pure ()

/-- Plan a T-mode `neView` scratch transpose `[B, rows, 1, cols] -> [B, cols, 1, rows]`
    (with `rows <= 128`), returning `(copyDesc, requiredScratchBytes)` where `requiredScratchBytes`
    satisfies both actual hardware output span and the Linux kernel `ane_hx_dma_tile` destination bound. -/
def planMatmulScratchTranspose (b rows cols : Nat) : Except String (ViewCopy × Nat) := do
  if b == 0 || b > 65535 || rows == 0 || cols == 0 || rows > 65535 || cols > 65535 then
    throw s!"matmul transpose: invalid dims (B={b}, rows={rows}, cols={cols})"
  if rows == 1 && cols == 1 then
    let rowBytes := dmaLineBytes
    let cp : ViewCopy := {
      n := b, c := 1, h := 1, w := 1, d := 1
      inOff := 0
      inNStride := if b > 1 then rowBytes else 0
      outOff := 0
      outNStride := if b > 1 then rowBytes else 0
      transposeCW := false
    }
    return (cp, alignUp (b * rowBytes) dmaLineBytes)
  if rows > 128 then
    throw s!"matmul transpose: C={rows} exceeds single-task T-mode coverage bound 128"
  let inRow := alignUp (2 * cols) dmaLineBytes
  let inNSt := rows * inRow
  let outRow := alignUp (2 * rows) dmaLineBytes
  let outPSt := if cols > 1 then outRow else 0
  let outNSt := cols * outRow
  let actualSpan := b * cols * outRow
  let kernelOutRaw := 2 * cols + (rows - 1) * outPSt + (b - 1) * (if b > 1 then outNSt else 0)
  let kernelOutEnd := alignUp kernelOutRaw dmaLineBytes
  let reqScratch := alignUp (max actualSpan kernelOutEnd) dmaLineBytes
  let cp : ViewCopy := {
    n := b, c := rows, h := 1, w := cols, d := 1
    inOff := 0
    inRowStride := 0
    inPlaneStride := if rows > 1 then inRow else 0
    inDStride := 0
    inNStride := if b > 1 then inNSt else 0
    outOff := 0
    outRowStride := 0
    outPlaneStride := outPSt
    outDStride := 0
    outNStride := if b > 1 then outNSt else 0
    transposeCW := true
  }
  pure (cp, reqScratch)

private def maxEndOf (rs : List Range) : Nat :=
  rs.foldl (fun acc r => max acc r.hi) 0

/-- Choose a legal tiling and L2 staging placement for a dynamic runtime-kernel `neMatmul` task
    with `M` output rows, `K` inner dimension, and `N` output columns. -/
def chooseMatmulTaskPlan
    (target : Target) (m k n : Nat)
    (srcIsL2 resIsL2 : Bool) (occupied : List Range := []) : Except String MatmulTaskPlan := do
  if m == 0 || m > 2048 || k == 0 || n == 0 || n > 65535 then
    throw s!"matmul tiling: invalid shape (M={m}, K={k}, N={n})"
  let f := min 4 (ceilLog2 (min m 16))
  for wantDirect in #[true, false] do
    for shPref in [:5] do
      let stripW := (128 : Nat) >>> shPref
      if stripW >= 8 && stripW <= 256 then
        let shMaxStart := max shPref (coverMin n stripW 0)
        if shMaxStart <= 4 then
          for shMax in [shMaxStart : 5] do
            let direct := if resIsL2 then false else (target.profile.directThreshold 0 shMax >= min 32 n)
            if resIsL2 || direct == wantDirect then
              let chs := 2 * min (alignUp n 8) (max stripW 32)
              let srcSz := if srcIsL2 then 0 else k * chs
              let resSz :=
                if resIsL2 then 0
                else if direct then ((1 : Nat) <<< f) * 512
                else m * chs
              let placed? : Option (Nat × Nat × Nat) :=
                if srcIsL2 && resIsL2 then
                  some (0, 0, maxEndOf occupied)
                else if !srcIsL2 && resIsL2 then
                  let srcAligned := alignUp srcSz dmaLineBytes
                  (findFreeOffset occupied srcAligned l2CapacityBytes).map fun sb =>
                    (sb, 0, max (maxEndOf occupied) (sb + srcSz))
                else if srcIsL2 && !resIsL2 then
                  let resAligned := alignUp resSz dmaLineBytes
                  (findFreeOffset occupied resAligned l2CapacityBytes).map fun rb =>
                    (0, rb, max (maxEndOf occupied) (rb + resSz))
                else
                  let srcAligned := alignUp srcSz dmaLineBytes
                  let resAligned := alignUp resSz dmaLineBytes
                  match findFreeOffset occupied srcAligned l2CapacityBytes with
                  | none => none
                  | some sb =>
                    let occ2 := { base := sb, bytes := srcAligned } :: occupied
                    match findFreeOffset occ2 resAligned l2CapacityBytes with
                    | none => none
                    | some rb =>
                      some (sb, rb, max (maxEndOf occupied) (max (sb + srcSz) (rb + resSz)))
              if let some (sb, rb, pk) := placed? then
                return {
                  tiling := {
                    logNE := f
                    f := f
                    fat := 0
                    ocg := 0
                    shPref := shPref
                    shMax := shMax
                    tileHeight := 1
                  }
                  direct := direct
                  srcStageBase := sb
                  resStageBase := rb
                  l2TaskPeak := pk
                }
  throw s!"no legal {target.name} dynamic matmul tiling fits 2 MiB L2 for (M={m}, K={k}, N={n})"

/-- Lower one dynamic runtime-kernel NE `MatMul` task into a multi-task `TaskDesc`. -/
def buildMatmulTaskMulti
    (target : Target)
    (b m k n : Nat)
    (scaleBits : UInt16)
    (mp : MatmulTaskPlan)
    (taskId totalTasks : Nat)
    (kernBarSlot : UInt8)
    (kernOff : Nat)
    (kernIsScratch : Bool)
    (kernProducerTask? : Option Nat)
    (src1Loc resLoc : MatmulPortLoc)
    (sync : TaskSyncInfo)
    (liveL2Mask : UInt32 := 0)
    (kernKeepLive : Bool := false) : Except String TaskDesc := do
  if !f16IsFinite scaleBits then
    throw s!"matmul task {taskId}: scale ({toHex scaleBits.toNat}) must be finite fp16"
  if kernIsScratch && kernProducerTask?.isNone then
    throw s!"matmul task {taskId}: scratch-produced kernel requires kernProducerTask"
  if taskId + 1 > 255 then
    throw s!"matmul task {taskId}: exceeds 8-bit kernelStage limit 254"

  let isFirst := taskId == 0
  let isLast := taskId + 1 == totalTasks
  let isDyn : Bool := m * b > 1
  let t := mp.tiling
  let stripW := (128 : Nat) >>> t.shPref
  let taskIdU32 ← natToU32Checked taskId "matmul taskId"
  let chunkSize := matmulChunkSize k
  if chunkSize > maxKernelChunkBytes then
    throw s!"matmul task {taskId}: chunkSize {chunkSize}B exceeds 64 KiB"

  let src1IsL2 := match src1Loc with | .l2 _ => true | _ => false
  let resIsL2 := match resLoc with | .l2 _ => true | _ => false

  let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked n "Win")]
  let wHin ← encodeWord (dim16Layout "Hin") #[("val", 1)]
  let kU32 ← natToU32Checked k "Cin"
  let mU32 ← natToU32Checked m "Cout"
  let wConvCfg ← encodeWord convCfgLayout #[
    ("kw", 1), ("kh", 1), ("sx", 1), ("sy", 1), ("pl", 0), ("pt", 0), ("hx", 1), ("vy", 1)
  ]
  let wTH ← encodeWord (dim16Layout "TH") #[("val", 1)]
  let wNeCommon ← encodeWord neMatmulCommonLayout #[
    ("lo2", if alignUp n 8 <= 64 then 1 else 0),
    ("shPref", ← natToU32Checked t.shPref "shPref"),
    ("shMin", ← natToU32Checked t.shPref "shMin"),
    ("shMax", ← natToU32Checked t.shMax "shMax"),
    ("F", ← natToU32Checked t.f "F"),
    ("dynKern", if isDyn then 1 else 0)
  ]
  let wNeTile ← encodeWord neTileLayout #[("ocg", 0), ("fat", 0), ("WU", 0)]
  let wNeCfg ← encodeWord (neMatmulCfgLayout target) #[("dynKern", 1)]
  let wNeScale ← encodeWord neScaleLayout #[("val", scaleBits.toUInt32)]
  let wKernStage ← encodeWord kernelStageDynLayout #[
    ("sparse", 0),
    ("kernEn", 1),
    ("isLast", if isLast then 1 else 0),
    ("kernCfg", 1),
    ("h17Dyn", if target.profile.dynKernelStageBit && isDyn then 1 else 0),
    ("stage", ← natToU32Checked (taskId + 1) "kernelStage.stage")
  ]
  let wSrc1Cfg ← encodeWord (portCfgMultiLayout "src1Cfg") #[
    ("mode", if src1IsL2 then portModeL2 else portModeDma),
    ("handoffNe", if !isFirst && src1IsL2 && sync.src1HandoffNe then 1 else 0),
    ("handoffPe", if !isFirst && src1IsL2 && sync.src1HandoffPe then 1 else 0),
    ("waitNe4", if isFirst then 1 else 0),
    ("waitNe5", if isFirst || sync.src1WaitNe5 then 1 else 0),
    ("dmaFlag", if src1IsL2 then 0 else 1),
    ("P", 1),
    ("waitPe20", if isFirst then 1 else 0),
    ("waitPe22", if isFirst || sync.src1WaitPe22 then 1 else 0)
  ]
  let wResCfg ← encodeWord resCfgMultiLayout #[
    ("mode", if resIsL2 then portModeL2 else portModeDma),
    ("direct", if !resIsL2 && mp.direct then 1 else 0),
    ("waitNe4", if isFirst then 1 else 0),
    ("waitNe5", if isFirst then 1 else 0),
    ("dmaFlag", 1),
    ("P", 1),
    ("waitPe20", if isFirst then 1 else 0),
    ("waitPe22", if isFirst then 1 else 0)
  ]

  let mut plain : Array RegWrite := #[
    { byteAddr := regWin, value := wWin },
    { byteAddr := regHin, value := wHin },
    { byteAddr := regCin, value := kU32 },
    { byteAddr := regWout, value := wWin },
    { byteAddr := regHout, value := wHin },
    { byteAddr := regCout, value := mU32 },
    { byteAddr := regConvCfg, value := wConvCfg },
    { byteAddr := regTH, value := wTH },
    { byteAddr := regNeCommon, value := wNeCommon },
    { byteAddr := regNeTile, value := wNeTile },
    { byteAddr := regTargetCfg, value := targetCfgValue target },
    { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
    { byteAddr := regResCfg, value := wResCfg },
    { byteAddr := regNeCfg, value := wNeCfg },
    { byteAddr := regNeOp, value := neOpMatmul },
    { byteAddr := regNeScale, value := wNeScale },
    { byteAddr := regKernelStage, value := wKernStage }
  ]

  if b > 1 then
    let wBatch ← encodeWord (dim16Layout "BatchN") #[("val", ← natToU32Checked b "BatchN")]
    plain := plain.push { byteAddr := regBatchN, value := wBatch }
    plain := plain.push { byteAddr := regKernelBatchStride, value := ← natToU32Checked (m * chunkSize) "kernelBatchStride" }

  if isDyn then
    let roundSpan := (if m > 1 then m else b) * chunkSize
    plain := plain.push { byteAddr := regKernelRoundStride, value := ← natToU32Checked roundSpan "kernelRoundStride" }

  if isFirst then
    plain := plain.push { byteAddr := regSrc2Cfg, value := src2CfgValue }

  let kOffLo ← natToU32Checked (kernOff &&& 0xffffffff) "kernelBaseLo"
  let kOffHi ← natToU32Checked (kernOff >>> 32) "kernelBaseHi"
  let mut barRecs : Array BarRegWrite := #[
    { baseByteAddr := regKernelBaseLo, barSlot := kernBarSlot, values := #[kOffLo, kOffHi] }
  ]

  -- Chunk slots 0..15:
  -- External-input enabled slots MUST be single-word BAR slot-4 relocations with 0x21.
  -- Scratch-produced enabled slots MUST be plain writes with 0x41 (or 0xc1 if keepLive && !isLast).
  -- Unused slots on taskId > 0 MUST be plain writes with 0x20.
  let numSlots := min 16 m
  let wSlotExt ← encodeWord chunkSlotDynLayout #[
    ("enable", 1), ("on20", 1), ("produced", 0), ("keepLive", 0), ("task", taskIdU32)
  ]
  let wSlotProd ← encodeWord chunkSlotDynLayout #[
    ("enable", 1), ("on20", 0), ("produced", 1),
    ("keepLive", if kernKeepLive && !isLast then 1 else 0),
    ("task", taskIdU32)
  ]
  let wSlotOff ← encodeWord chunkSlotDynLayout #[
    ("enable", 0), ("on20", 1), ("produced", 0), ("keepLive", 0), ("task", taskIdU32)
  ]
  for s in [:16] do
    if s < numSlots then
      if kernIsScratch then
        plain := plain.push { byteAddr := regChunkSlot s, value := wSlotProd }
      else
        barRecs := barRecs.push {
          baseByteAddr := regChunkSlot s
          barSlot := barSlotConfigZero
          values := #[wSlotExt]
        }
      plain := plain ++ #[
        { byteAddr := regChunkOffset s, value := ← natToU32Checked (s * chunkSize) s!"chunkOffset[{s}]" },
        { byteAddr := regChunkSize s, value := ← natToU32Checked chunkSize s!"chunkSize[{s}]" }
      ]
    else if !isFirst then
      plain := plain.push { byteAddr := regChunkSlot s, value := wSlotOff }

  -- Source 1 (B in [B, K, 1, N])
  let rowBN := alignUp (2 * n) dmaLineBytes
  match src1Loc with
  | .l2 base =>
    let l2Chs := 2 * alignUp n 8
    let l2Batch := k * l2Chs
    plain := plain ++ #[
      { byteAddr := regSrc1Base, value := ← natToU32Checked base "src1Base" },
      { byteAddr := regSrc1ChanStride, value := ← natToU32Checked l2Chs "src1ChanStride" },
      { byteAddr := regSrc1RowStride, value := ← natToU32Checked l2Batch "src1RowStride" }
    ]
    if b > 1 then
      plain := plain.push { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked l2Batch "src1RowBytesB" }
  | loc =>
    let (slot, off) := match loc with
      | .surface s o => (s, o)
      | .dram o => (barSlotScratch, o)
      | .l2 _ => (0, 0)
    let chs := 2 * min (alignUp n 8) (max stripW 32)
    let u := k * chs
    let uU32 ← natToU32Checked u "src1RowBytes"
    plain := plain ++ #[
      { byteAddr := regSrc1Base, value := ← natToU32Checked mp.srcStageBase "src1Base" },
      { byteAddr := regSrc1ChanStride, value := ← natToU32Checked chs "src1ChanStride" },
      { byteAddr := regSrc1RowStride, value := uU32 },
      { byteAddr := regSrc1RowBytesA, value := uU32 },
      { byteAddr := regSrc1RowBytesB, value := uU32 },
      { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
      { byteAddr := regSrc1Fmt, value := srcFmtWordP1 }
    ]
    if k > 1 then
      plain := plain.push { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked rowBN "src1DmaPlaneStride" }
    if b > 1 then
      plain := plain.push { byteAddr := regSrc1DmaNStride, value := ← natToU32Checked (k * rowBN) "src1DmaNStride" }
    let wSrc1Dep ← encodeWord (srcDepMultiLayout "src1Dep") #[
      ("code", srcDepCodeDma),
      ("task", taskIdU32),
      ("depDist", ← natToU32Checked sync.src1DepDist "src1DepDist"),
      ("depKind", ← natToU32Checked sync.src1DepKind "src1DepKind")
    ]
    let offLo ← natToU32Checked (off &&& 0xffffffff) "src1BaseLo"
    let offHi ← natToU32Checked (off >>> 32) "src1BaseHi"
    barRecs := barRecs ++ #[
      { baseByteAddr := regSrc1BaseLo, barSlot := slot, values := #[offLo, offHi] },
      { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrc1Dep] }
    ]

  -- Result (Y in [B, M, 1, N])
  match resLoc with
  | .l2 base =>
    let l2Chs := 2 * alignUp n 8
    let l2Batch := m * l2Chs
    plain := plain ++ #[
      { byteAddr := regResBase, value := ← natToU32Checked base "resBase" },
      { byteAddr := regResChanStride, value := ← natToU32Checked l2Chs "resChanStride" },
      { byteAddr := regResRowStride, value := ← natToU32Checked l2Batch "resRowStride" }
    ]
    if b > 1 then
      plain := plain.push { byteAddr := regResRowBytesB, value := ← natToU32Checked l2Batch "resRowBytesB" }
  | loc =>
    let (slot, off, depCode) := match loc with
      | .surface s o => (s, o, resDepCode target)
      | .dram o => (barSlotScratch, o, resDepCode target ||| resDepScratchFlags)
      | .l2 _ => (0, 0, resDepCode target)
    plain := plain ++ #[
      { byteAddr := regResBase, value := ← natToU32Checked mp.resStageBase "resBase" },
      { byteAddr := regResFmt, value := resFmtWordP1 }
    ]
    if !mp.direct then
      let dchs := 2 * min (alignUp n 8) (max stripW 32)
      let du := m * dchs
      let duU32 ← natToU32Checked du "resRowBytes"
      plain := plain ++ #[
        { byteAddr := regResChanStride, value := ← natToU32Checked dchs "resChanStride" },
        { byteAddr := regResRowStride, value := duU32 },
        { byteAddr := regResRowBytesA, value := duU32 },
        { byteAddr := regResRowBytesB, value := duU32 }
      ]
    if m > 1 then
      plain := plain.push { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked rowBN "resDmaPlaneStride" }
    if b > 1 then
      plain := plain.push { byteAddr := regResDmaNStride, value := ← natToU32Checked (m * rowBN) "resDmaNStride" }
    let wResDep ← encodeWord resDepLayout #[
      ("code", depCode),
      ("task", taskIdU32),
      ("direct", if mp.direct then 1 else 0)
    ]
    let offLo ← natToU32Checked (off &&& 0xffffffff) "resBaseLo"
    let offHi ← natToU32Checked (off >>> 32) "resBaseHi"
    barRecs := barRecs ++ #[
      { baseByteAddr := regResBaseLo, barSlot := slot, values := #[offLo, offHi] },
      { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
    ]

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

  pure {
    taskId := taskId
    logNE := t.logNE
    eneOverride := some (t.logNE + 1)
    isFirst := isFirst
    isLast := isLast
    liveL2Mask := liveL2Mask
    extProducerTask := if kernIsScratch then kernProducerTask? else none
    plainWrites := plain
    barWrites := barRecs
    kernelRelocs := #[]
  }

/-- Compile a standalone dynamic runtime-kernel `MatMul` program into a statically verified
    HWX container and its 4 Linux LOAD sections.
    Supports optional `transposeA` and/or `transposeB` (lowering via T-mode `neView` into
    `BAR 3` scratch) and arbitrary finite fp16 post-accumulation `scaleBits`. -/
def compileMatmul (target : Target) (spec : MatmulSpec) : Except String CompiledMatmul := do
  validateMatmulSpec spec

  let aSlot : UInt8 := barSlotFirstSurface
  let bSlot : UInt8 := barSlotFirstSurface + 1
  let ySlot : UInt8 := barSlotFirstSurface + 2

  let totalTasks : Nat :=
    1 + (if spec.transposeA then 1 else 0) + (if spec.transposeB then 1 else 0)

  let mut tasks : Array TaskDesc := Array.mkEmpty totalTasks
  let mut scratchCursor : Nat := 0
  let mut peakL2 : Nat := 0

  let mut kernSlot : UInt8 := aSlot
  let mut kernOff : Nat := 0
  let mut kernIsScratch : Bool := false
  let mut kernProd? : Option Nat := none

  if spec.transposeA then
    let (cpA, szA) ← planMatmulScratchTranspose spec.b spec.k spec.m
    let vpA ← chooseViewTaskPlan target cpA []
    peakL2 := max peakL2 vpA.l2TaskPeak
    let tid := tasks.size
    let syncA : TaskSyncInfo := { src1WaitNe5 := tid > 0, hasSrc1Dma := true }
    let tdA ← buildViewTaskMulti target { cpA with outOff := scratchCursor } vpA tid totalTasks aSlot barSlotScratch true syncA 0
    tasks := tasks.push tdA
    kernSlot := barSlotScratch
    kernOff := scratchCursor
    kernIsScratch := true
    kernProd? := some tid
    scratchCursor := scratchCursor + szA

  let mut src1Loc : MatmulPortLoc := .surface bSlot 0
  let mut src1DepKind : Nat := 0
  if spec.transposeB then
    let (cpB, szB) ← planMatmulScratchTranspose spec.b spec.n spec.k
    let vpB ← chooseViewTaskPlan target cpB []
    peakL2 := max peakL2 vpB.l2TaskPeak
    let tid := tasks.size
    let syncB : TaskSyncInfo := { src1WaitNe5 := tid > 0, hasSrc1Dma := true }
    let tdB ← buildViewTaskMulti target { cpB with outOff := scratchCursor } vpB tid totalTasks bSlot barSlotScratch true syncB 0
    tasks := tasks.push tdB
    src1Loc := .dram scratchCursor
    src1DepKind := if spec.b > 1 then 1 else 2
    scratchCursor := scratchCursor + szB

  let mp ← chooseMatmulTaskPlan target spec.m spec.k spec.n false false []
  peakL2 := max peakL2 mp.l2TaskPeak
  let mmTid := tasks.size
  let syncMM : TaskSyncInfo := {
    src1WaitNe5 := mmTid > 0
    src1DepKind := src1DepKind
    src1DepDist := 0
    hasSrc1Dma := true
  }
  let tdMM ← buildMatmulTaskMulti target spec.b spec.m spec.k spec.n spec.scaleBits mp
    mmTid totalTasks kernSlot kernOff kernIsScratch kernProd? src1Loc (.surface ySlot 0) syncMM 0
  tasks := tasks.push tdMM

  let (aC, aW) := if spec.transposeA then (spec.k, spec.m) else (spec.m, spec.k)
  let (bC, bW) := if spec.transposeB then (spec.n, spec.k) else (spec.k, spec.n)
  let sA ← makeFp16Surface target "a" "input_a" .input aC 1 aW spec.b
  let sB ← makeFp16Surface target "b" "input_b" .input bC 1 bW spec.b
  let sY ← makeFp16Surface target "y" "output_y" .output spec.m 1 spec.n spec.b

  let scratchSpec? : Option ScratchSpec ←
    if scratchCursor == 0 then pure none
    else
      let szU64 ← natToU64Checked scratchCursor "matmul scratch size"
      let usU32 ← natToU32Checked scratchCursor "matmul scratch usable"
      pure (some { size := szU64, usable := usU32 })

  let emitted ← emitProgram {
    target := target
    surfaces := #[sA, sB, sY]
    tasks := tasks
    kernel := none
    scratch := scratchSpec?
    constants := zeroBytes textConstBytes
    ident := some "vane"
  }

  let summary ← verifyHwxTasksSummary emitted.hwx emitted.info (requireKnownTask := true)
  if summary.neMatmul != 1 || summary.total != totalTasks then
    throw s!"internal error: verifier checked {summary.neMatmul} matmul / {summary.total} total tasks != 1/{totalTasks}"

  pure {
    target := target
    spec := spec
    taskCount := totalTasks
    tiling := mp.tiling
    direct := mp.direct
    l2PeakBytes := peakL2
    scratchBytes := scratchCursor
    hwx := emitted.hwx
    info := emitted.info
    sections := emitted.sections
  }

end Vane.Matmul
