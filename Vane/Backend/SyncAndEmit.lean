import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Hwx
import Vane.Sections
import Vane.Register
import Vane.Reach
import Vane.Alloc
import Vane.Tiling
import Vane.Quant
import Vane.Program
import Vane.Verify
import Vane.Pe
import Vane.Lut
import Vane.Pool
import Vane.Conv
import Vane.View
import Vane.Matmul
import Vane.Ane.Types
import Vane.Machine.Types
import Vane.Backend.ChannelSplit
import Vane.Backend.Cascade
import Vane.Backend.MemoryPlan

namespace Vane.Backend.SyncAndEmit

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Reach
open Vane.Alloc
open Vane.Tiling
open Vane.Quant
open Vane.Program
open Vane.Verify
open Vane.Pe
open Vane.Lut
open Vane.Pool
open Vane.Conv
open Vane.View
open Vane.Matmul
open Vane.Ane
open Vane.Machine

-- ============================================================================
-- Compiled Machine Artifact
-- ============================================================================

/-- Complete verified compilation artifact emitted from L3 `MachineProgram`. -/
structure CompiledMachineProgram where
  machine : MachineProgram
  hwx : ByteArray
  info : HwxInfo
  sections : LoadSections
  verifySummary : TaskVerifySummary

-- ============================================================================
-- Register & Sync Bit Encoders (Single Source of Truth)
-- ============================================================================

private def encodeSrcPortCfgMulti
    (name : String) (isFirst isL2 waitNe5 waitPe22 handoffNe handoffPe : Bool) : Except String UInt32 :=
  encodeWord (portCfgMultiLayout name) #[
    ("mode", if isL2 then portModeL2 else portModeDma),
    ("handoffNe", if !isFirst && isL2 && handoffNe then 1 else 0),
    ("handoffPe", if !isFirst && isL2 && handoffPe then 1 else 0),
    ("waitNe4", if isFirst then 1 else 0),
    ("waitNe5", if isFirst || waitNe5 then 1 else 0),
    ("dmaFlag", if isL2 then 0 else 1),
    ("P", 1),
    ("waitPe20", if isFirst then 1 else 0),
    ("waitPe22", if isFirst || waitPe22 then 1 else 0)
  ]

private def encodeResPortCfgMulti (isFirst isL2 direct : Bool) : Except String UInt32 :=
  encodeWord resCfgMultiLayout #[
    ("mode", if isL2 then portModeL2 else portModeDma),
    ("direct", if !isL2 && direct then 1 else 0),
    ("waitNe4", if isFirst then 1 else 0),
    ("waitNe5", if isFirst then 1 else 0),
    ("dmaFlag", 1),
    ("P", 1),
    ("waitPe20", if isFirst then 1 else 0),
    ("waitPe22", if isFirst then 1 else 0)
  ]

private def encodeSrcDepMulti (name : String) (taskId depKind depDist : Nat) : Except String UInt32 := do
  let tU32 ← natToU32Checked taskId s!"{name} task"
  let kU32 ← natToU32Checked depKind s!"{name} depKind"
  let dU32 ← natToU32Checked depDist s!"{name} depDist"
  encodeWord (srcDepMultiLayout name) #[
    ("code", srcDepCodeDma), ("task", tU32), ("depDist", dU32), ("depKind", kU32)
  ]

private def encodeKernelStageMulti (taskId : Nat) (isLast isSparse : Bool) : Except String UInt32 := do
  let stageU32 ← natToU32Checked (taskId + 1) "kernelStage.stage"
  encodeWord kernelStageMultiLayout #[
    ("sparse", if isSparse then 1 else 0), ("kernEn", 1),
    ("isLast", if isLast then 1 else 0), ("kernCfg", 1), ("stage", stageU32)
  ]

private def outputStageWrites (taskId : Nat) : Except String (Array RegWrite) := do
  let taskIdU32 ← natToU32Checked taskId "out2 taskId"
  let wOut2 ← encodeWord out2Layout #[("form", out2FormValue), ("task", taskIdU32)]
  pure #[
    { byteAddr := regOut0, value := out0Value },
    { byteAddr := regOut1, value := out1Value },
    { byteAddr := regOut2, value := wOut2 },
    { byteAddr := regOut3, value := out3Value },
    { byteAddr := regOut4, value := out4Value },
    { byteAddr := regOut5, value := out5Value }
  ]

private def dramSpanDims (c h w : Nat) : Nat :=
  let row := alignUp (2 * w) dmaLineBytes
  c * h * row

-- ============================================================================
-- Phase 1: Kernel Packing & Exact L2/DRAM Access Footprint Extraction
-- ============================================================================

private structure ConvKernCacheKey where
  weightGroupId : Nat
  kernelSize : Nat
  mode : Conv3x3Mode
  cin : Nat
  cout : Nat
  stride : Nat
  act : ConvActivation
  hasBias : Bool
  quantFmt? : Option QuantFormat := none
  logNE : Nat
  f : Nat
  fat : Nat
  ocg : Nat
  deriving BEq

private structure PreparedEmitTask where
  task : MachineTask
  src1Storage : ResolvedStorage
  src2Storage? : Option ResolvedStorage
  resStorage : ResolvedStorage
  l2Accesses : Array VerifiedL2Access
  dramAccesses : Array VerifiedDramAccess
  hasSrc1Dma : Bool
  hasSrc2Dma : Bool
  packedKernel? : Option PackedKernel := none
  packedQuantKernel? : Option PackedQuantKernel := none
  kernelBaseOff : Nat := 0
  src1L2Producer? : Option Nat := none
  kernStorage? : Option ResolvedStorage := none
  extProducerTask? : Option Nat := none
  deriving Inhabited

private def surfNameAt (allSurfs : Array SurfaceSpec) (idx : Nat) : String :=
  match allSurfs[idx]? with
  | some s => s.shortName
  | none => s!"s{idx}"

private def buildPreparedEmitTasks (mp : MachineProgram) :
    Except String (Array PreparedEmitTask × ByteArray) := do
  let allSurfs := mp.allSurfaces
  let n := mp.tasks.size
  let mut kernBuf := ByteArray.empty
  let mut convKernCache : Array (ConvKernCacheKey × ByteArray × Option ByteArray × Option ByteArray × Nat × PackedKernel) := #[]
  let mut quantKernCache : Array (ConvKernCacheKey × QuantWeightSpec × Option ByteArray × Nat × PackedQuantKernel) := #[]
  let mut lutKernCache : Array (Nat × ByteArray × Nat) := #[]
  let mut out : Array PreparedEmitTask := Array.mkEmpty n

  -- Track latest L2 writer task index for each L2 base range `(base, bytes)`
  let mut l2Writers : Array (Nat × Nat × Nat) := #[] -- (base, bytes, taskIdx)
  let findL2Prod := fun (ws : Array (Nat × Nat × Nat)) (base bytes : Nat) => Id.run do
    let mut best : Option Nat := none
    for (b, sz, tid) in ws do
      if b == base && sz == bytes then best := some tid
    return best

  for i in [:n] do
    let mt := mp.tasks[i]!
    let resSt ← resolveBufferStorage mp.buffers mt.output
    let mut src1St ← resolveBufferStorage mp.buffers mt.inputs[0]!
    let mut src2St? : Option ResolvedStorage := none
    let mut l2Accs : Array VerifiedL2Access := #[]
    let mut dramAccs : Array VerifiedDramAccess := #[]
    let mut packedKern? : Option PackedKernel := none
    let mut packedQuantKern? : Option PackedQuantKernel := none
    let mut kernBaseOff : Nat := 0
    let mut kernSt? : Option ResolvedStorage := none
    let mut extProd? : Option Nat := none

    match mt.op, mt.tilePlan with
    | .neConv _ _ _ _ _ _ _ _ weight bias act _ wgId _, .neConv cp lp? _ =>
      let qFmt? : Option QuantFormat := match weight with | .quant q => some q.format | .fp16 _ => none
      let cKey : ConvKernCacheKey := {
        weightGroupId := wgId
        kernelSize := cp.shape.kernelSize
        mode := cp.shape.effMode
        cin := cp.shape.cin
        cout := cp.shape.cout
        stride := cp.shape.stride
        act := act
        hasBias := bias.isSome
        quantFmt? := qFmt?
        logNE := cp.tiling.logNE
        f := cp.tiling.f
        fat := cp.tiling.fat
        ocg := cp.tiling.ocg
      }
      match weight with
      | .quant q =>
        let cached? :=
          if wgId == 0 then none
          else quantKernCache.find? fun (k, cq, cb, _, _) => k == cKey && cq == q && cb == bias
        let (kOff, packed) ← match cached? with
          | some (_, _, _, off, p) => pure (off, p)
          | none =>
            kernBuf := padToAlign kernBuf dmaLineBytes
            let newOff := kernBuf.size
            let p ← packQuantKernel mp.target cp.shape cp.tiling q bias
            kernBuf := kernBuf ++ p.bytes
            if wgId != 0 then
              quantKernCache := quantKernCache.push (cKey, q, bias, newOff, p)
            pure (newOff, p)
        kernBaseOff := kOff
        packedQuantKern? := some packed
        for cs in packed.slots do
          dramAccs := dramAccs.push { resource := .kernel, lo := kOff + cs.offset, hi := kOff + cs.offset + cs.size, isWrite := false, port := .kernel }
      | .fp16 wBytes =>
        let lutBytes? := lp?.map (·.bytes)
        let cached? :=
          if wgId == 0 then none
          else convKernCache.find? fun (k, cw, cb, cl, _, _) =>
            k == cKey && cw == wBytes && cb == bias && cl == lutBytes?
        let (kOff, packed) ← match cached? with
          | some (_, _, _, _, off, p) => pure (off, p)
          | none =>
            kernBuf := padToAlign kernBuf dmaLineBytes
            let newOff := kernBuf.size
            let p ← packConvKernelChunks mp.target cp.shape cp.tiling wBytes bias lp?
            kernBuf := kernBuf ++ p.bytes
            if wgId != 0 then
              convKernCache := convKernCache.push (cKey, wBytes, bias, lutBytes?, newOff, p)
            pure (newOff, p)
        kernBaseOff := kOff
        packedKern? := some packed
        for cs in packed.slots do
          dramAccs := dramAccs.push { resource := .kernel, lo := kOff + cs.offset, hi := kOff + cs.offset + cs.size, isWrite := false, port := .kernel }

      let sh := cp.shape
      match src1St with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dramScratch off _ =>
        let sz := srcL2BytesConv sh cp.tiling
        l2Accs := l2Accs.push { lo := cp.srcStageBase, hi := cp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        let span := dramSpanDims sh.cin sh.h sh.w
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + span, isWrite := false, port := .src1Dma }
      | .surface idx _ _ bOff win _ =>
        let sz := srcL2BytesConv sh cp.tiling
        l2Accs := l2Accs.push { lo := cp.srcStageBase, hi := cp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        let (wOff, _, _, ext) := surfaceDmaWindow win sh.cin sh.h sh.w
        let actOff := bOff + wOff
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := actOff, hi := actOff + ext, isWrite := false, port := .src1Dma }

      match resSt with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dramScratch off _ =>
        let sz := if cp.direct then ((1 : Nat) <<< (cp.tiling.f + effOcgConv sh cp.tiling + cp.tiling.fat)) * 512
                  else cp.tiling.tileHeight * resRowStrideConv sh cp.tiling
        l2Accs := l2Accs.push { lo := cp.resStageBase, hi := cp.resStageBase + sz, isWrite := true, role := .resStage }
        let span := dramSpanDims sh.cout sh.outH sh.outW
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + span, isWrite := true, port := .resDma, coverage := .dense off (off + span) }
      | .surface idx _ _ bOff win _ =>
        let sz := if cp.direct then ((1 : Nat) <<< (cp.tiling.f + effOcgConv sh cp.tiling + cp.tiling.fat)) * 512
                  else cp.tiling.tileHeight * resRowStrideConv sh cp.tiling
        l2Accs := l2Accs.push { lo := cp.resStageBase, hi := cp.resStageBase + sz, isWrite := true, role := .resStage }
        let (wOff, _, _, ext) := surfaceDmaWindow win sh.cout sh.outH sh.outW
        let actOff := bOff + wOff
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := actOff, hi := actOff + ext, isWrite := true, port := .resDma }

    | .pe (.unary _) _, .pe pp =>
      let dims := pp.outDims
      let effStripW := max (peStripWidth pp.tiling.k) 32
      let row1 := 2 * min (alignUp dims.w 8) effStripW
      let sz1 := (if dims.h > 1 then pp.tiling.tileHeight else 1) * (dims.c * row1)
      match src1St with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := pp.src1StageBase, hi := pp.src1StageBase + sz1, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanDims dims.c dims.h dims.w, isWrite := false, port := .src1Dma }
      | .surface idx _ _ bOff win _ =>
        l2Accs := l2Accs.push { lo := pp.src1StageBase, hi := pp.src1StageBase + sz1, isWrite := true, role := .src1Stage }
        let (wOff, _, _, ext) := surfaceDmaWindow win dims.c dims.h dims.w
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := bOff + wOff, hi := bOff + wOff + ext, isWrite := false, port := .src1Dma }

      let szRes := if pp.direct then peDirectResBytes dims.c row1 pp.tiling.tileHeight else pp.tiling.tileHeight * (dims.c * row1)
      match resSt with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := pp.resStageBase, hi := pp.resStageBase + szRes, isWrite := true, role := .resStage }
        let span := dramSpanDims dims.c dims.h dims.w
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + span, isWrite := true, port := .resDma, coverage := .dense off (off + span) }
      | .surface idx _ _ bOff win _ =>
        l2Accs := l2Accs.push { lo := pp.resStageBase, hi := pp.resStageBase + szRes, isWrite := true, role := .resStage }
        let (wOff, _, _, ext) := surfaceDmaWindow win dims.c dims.h dims.w
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := bOff + wOff, hi := bOff + wOff + ext, isWrite := true, port := .resDma }

    | .pe (.binary _ rev) _, .pe pp =>
      let b1 := if rev then mt.inputs[1]! else mt.inputs[0]!
      let b2 := if rev then mt.inputs[0]! else mt.inputs[1]!
      src1St ← resolveBufferStorage mp.buffers b1
      let s2St ← resolveBufferStorage mp.buffers b2
      src2St? := some s2St
      let outDims := pp.outDims
      let hw1 := pp.hwSrc1
      let hw2 := pp.hwSrc2?.getD outDims
      let effStripW := max (peStripWidth pp.tiling.k) 32
      let th := pp.tiling.tileHeight
      let whole1 := hw1.c == 1 && hw1.h == 1 && hw1.w > 1 && outDims.h > 1 && th < outDims.h
      let row1 := 2 * (if whole1 then alignUp hw1.w 8 else min (alignUp hw1.w 8) effStripW)
      let sz1 := (if hw1.h > 1 then th else 1) * (hw1.c * row1)
      match src1St with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := pp.src1StageBase, hi := pp.src1StageBase + sz1, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanDims hw1.c hw1.h hw1.w, isWrite := false, port := .src1Dma }
      | .surface idx _ _ bOff win _ =>
        l2Accs := l2Accs.push { lo := pp.src1StageBase, hi := pp.src1StageBase + sz1, isWrite := true, role := .src1Stage }
        let (wOff, _, _, ext) := surfaceDmaWindow win hw1.c hw1.h hw1.w
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := bOff + wOff, hi := bOff + wOff + ext, isWrite := false, port := .src1Dma }

      let whole2 := hw2.c == 1 && hw2.h == 1 && hw2.w > 1 && outDims.h > 1 && th < outDims.h
      let row2 := 2 * (if whole2 then alignUp hw2.w 8 else min (alignUp hw2.w 8) effStripW)
      let sz2 := (if hw2.h > 1 then th else 1) * (hw2.c * row2)
      match s2St with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src2L2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := pp.src2StageBase, hi := pp.src2StageBase + sz2, isWrite := true, role := .src2Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanDims hw2.c hw2.h hw2.w, isWrite := false, port := .src2Dma }
      | .surface idx _ _ bOff win _ =>
        l2Accs := l2Accs.push { lo := pp.src2StageBase, hi := pp.src2StageBase + sz2, isWrite := true, role := .src2Stage }
        let (wOff, _, _, ext) := surfaceDmaWindow win hw2.c hw2.h hw2.w
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := bOff + wOff, hi := bOff + wOff + ext, isWrite := false, port := .src2Dma }

      let drow := 2 * min (alignUp outDims.w 8) effStripW
      let szRes := if pp.direct then peDirectResBytes outDims.c drow th else th * (outDims.c * drow)
      match resSt with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := pp.resStageBase, hi := pp.resStageBase + szRes, isWrite := true, role := .resStage }
        let span := dramSpanDims outDims.c outDims.h outDims.w
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + span, isWrite := true, port := .resDma, coverage := .dense off (off + span) }
      | .surface idx _ _ bOff win _ =>
        l2Accs := l2Accs.push { lo := pp.resStageBase, hi := pp.resStageBase + szRes, isWrite := true, role := .resStage }
        let (wOff, _, _, ext) := surfaceDmaWindow win outDims.c outDims.h outDims.w
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := bOff + wOff, hi := bOff + wOff + ext, isWrite := true, port := .resDma }

    | .neAct (.act actSpec) _ wgId, .neAct lp lutPayload? =>
      let dims := shape4DToDims mt.outShape
      if let some payload := lutPayload? then
        let cachedLut? :=
          if wgId == 0 then none
          else lutKernCache.find? fun (wid, cBytes, _) => wid == wgId && cBytes == payload.bytes
        let lutOff ← match cachedLut? with
          | some (_, _, off) => pure off
          | none =>
            kernBuf := padToAlign kernBuf dmaLineBytes
            let newOff := kernBuf.size
            kernBuf := kernBuf ++ payload.bytes
            if wgId != 0 then
              lutKernCache := lutKernCache.push (wgId, payload.bytes, newOff)
            pure newOff
        kernBaseOff := lutOff
        dramAccs := dramAccs.push { resource := .kernel, lo := lutOff, hi := lutOff + lutTableBytes, isWrite := false, port := .lut }
      let stripW := lp.tiling.stripWidth
      let effSrcStrip := if actSpec == .copy && n > 1 then max stripW 32 else stripW
      let row := 2 * min (alignUp dims.w 8) effSrcStrip
      let sz := lp.tiling.tileHeight * (dims.c * row)
      match src1St with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := lp.srcStageBase, hi := lp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanDims dims.c dims.h dims.w, isWrite := false, port := .src1Dma }
      | .surface idx _ _ bOff win _ =>
        l2Accs := l2Accs.push { lo := lp.srcStageBase, hi := lp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        let (wOff, _, _, ext) := surfaceDmaWindow win dims.c dims.h dims.w
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := bOff + wOff, hi := bOff + wOff + ext, isWrite := false, port := .src1Dma }

      let drow := 2 * min (alignUp dims.w 8) (max stripW 32)
      let szRes := if lp.direct then neActDirectResBytes lp.tiling.f lp.tiling.fat else lp.tiling.tileHeight * (dims.c * drow)
      match resSt with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := lp.resStageBase, hi := lp.resStageBase + szRes, isWrite := true, role := .resStage }
        let span := dramSpanDims dims.c dims.h dims.w
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + span, isWrite := true, port := .resDma, coverage := .dense off (off + span) }
      | .surface idx _ _ bOff win _ =>
        l2Accs := l2Accs.push { lo := lp.resStageBase, hi := lp.resStageBase + szRes, isWrite := true, role := .resStage }
        let (wOff, _, _, ext) := surfaceDmaWindow win dims.c dims.h dims.w
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := bOff + wOff, hi := bOff + wOff + ext, isWrite := true, port := .resDma }

    | .pe (.pool ..) _, .pePool pp gh gw =>
      let inSh := mp.buffers[mt.inputs[0]!]!.shape
      let dims := shape4DToDims inSh
      let stripW := (1 : Nat) <<< pp.k
      let px := poolStripPx gw.used gw.k gw.sEff gw.padBefore stripW gw.out
      let sz := alignUp (poolInRows gh.used gh.k gh.sEff gh.padBefore gh.out pp.tileHeight false * (dims.c * 2 * px)) dmaLineBytes
      match src1St with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := pp.srcStageBase, hi := pp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanDims dims.c dims.h dims.w, isWrite := false, port := .src1Dma }
      | .surface idx _ _ _ _ _ =>
        l2Accs := l2Accs.push { lo := pp.srcStageBase, hi := pp.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := 0, hi := dramSpanDims dims.c dims.h dims.w, isWrite := false, port := .src1Dma }
      let drow := 2 * min (alignUp gw.out 8) (max stripW 32)
      let resSz := if pp.direct then peDirectResBytes dims.c drow pp.tileHeight else alignUp (pp.tileHeight * dims.c * drow) dmaLineBytes
      let outSpan := dramSpanDims mt.outShape.c mt.outShape.h mt.outShape.w
      match resSt with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := pp.resStageBase, hi := pp.resStageBase + resSz, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + outSpan, isWrite := true, port := .resDma, coverage := .dense off (off + outSpan) }
      | .surface idx _ _ _ _ _ =>
        l2Accs := l2Accs.push { lo := pp.resStageBase, hi := pp.resStageBase + resSz, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := 0, hi := outSpan, isWrite := true, port := .resDma }

    | .pe (.reduce _ axes _) _, .peReduce pr =>
      let inSh := mp.buffers[mt.inputs[0]!]!.shape
      let dims := shape4DToDims inSh
      let ax := peReduceAxes dims.c dims.h axes.reduceC axes.reduceH
      let stripW := (1 : Nat) <<< pr.k
      let px := min (alignUp dims.w 8) (max stripW 32)
      let sz := alignUp (pr.tileHeight * ax.sC * ax.sD * 2 * px) dmaLineBytes
      match src1St with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := pr.srcStageBase, hi := pr.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanDims dims.c dims.h dims.w, isWrite := false, port := .src1Dma }
      | .surface idx _ _ _ _ _ =>
        l2Accs := l2Accs.push { lo := pr.srcStageBase, hi := pr.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := 0, hi := dramSpanDims dims.c dims.h dims.w, isWrite := false, port := .src1Dma }
      let outSpan := dramSpanDims mt.outShape.c mt.outShape.h mt.outShape.w
      match resSt with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := pr.resStageBase, hi := pr.resStageBase + 2048, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + outSpan, isWrite := true, port := .resDma, coverage := .dense off (off + outSpan) }
      | .surface idx _ _ _ _ _ =>
        l2Accs := l2Accs.push { lo := pr.resStageBase, hi := pr.resStageBase + 2048, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := 0, hi := outSpan, isWrite := true, port := .resDma }

    | .neAct (.nePool ..) _ _, .nePool np gh gw =>
      let inSh := mp.buffers[mt.inputs[0]!]!.shape
      let dims := shape4DToDims inSh
      let stripW := np.stripWidth
      let px := poolStripPx gw.used gw.k gw.sEff gw.padBefore stripW gw.out
      let sz := alignUp (poolInRows gh.used gh.k gh.sEff gh.padBefore gh.out np.tileHeight true * (dims.c * 2 * px)) dmaLineBytes
      match src1St with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := np.srcStageBase, hi := np.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanDims dims.c dims.h dims.w, isWrite := false, port := .src1Dma }
      | .surface idx _ _ _ _ _ =>
        l2Accs := l2Accs.push { lo := np.srcStageBase, hi := np.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := 0, hi := dramSpanDims dims.c dims.h dims.w, isWrite := false, port := .src1Dma }
      let drow := 2 * min (alignUp gw.out 8) (max stripW 32)
      let resSz := if np.direct then neActDirectResBytes np.f np.fat else alignUp (np.tileHeight * dims.c * drow) dmaLineBytes
      let outSpan := dramSpanDims mt.outShape.c mt.outShape.h mt.outShape.w
      match resSt with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := np.resStageBase, hi := np.resStageBase + resSz, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + outSpan, isWrite := true, port := .resDma, coverage := .dense off (off + outSpan) }
      | .surface idx _ _ _ _ _ =>
        l2Accs := l2Accs.push { lo := np.resStageBase, hi := np.resStageBase + resSz, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := 0, hi := outSpan, isWrite := true, port := .resDma }

    | .neAct (.channelReduce _) _ _, .neChannelReduce nc =>
      let inSh := mp.buffers[mt.inputs[0]!]!.shape
      let dims := shape4DToDims inSh
      let stripW := nc.stripWidth
      let px := min (alignUp dims.w 8) stripW
      let sz := alignUp (nc.tileHeight * dims.c * 2 * px) dmaLineBytes
      match src1St with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := nc.srcStageBase, hi := nc.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + dramSpanDims dims.c dims.h dims.w, isWrite := false, port := .src1Dma }
      | .surface idx _ _ _ _ _ =>
        l2Accs := l2Accs.push { lo := nc.srcStageBase, hi := nc.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := 0, hi := dramSpanDims dims.c dims.h dims.w, isWrite := false, port := .src1Dma }
      let drow := 2 * min (alignUp dims.w 8) (max stripW 32)
      let resSz := if nc.direct then neActDirectResBytes nc.f nc.fat else alignUp (nc.tileHeight * drow) dmaLineBytes
      let outSpan := dramSpanDims mt.outShape.c mt.outShape.h mt.outShape.w
      match resSt with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := nc.resStageBase, hi := nc.resStageBase + resSz, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + outSpan, isWrite := true, port := .resDma, coverage := .dense off (off + outSpan) }
      | .surface idx _ _ _ _ _ =>
        l2Accs := l2Accs.push { lo := nc.resStageBase, hi := nc.resStageBase + resSz, isWrite := true, role := .resStage }
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := 0, hi := outSpan, isWrite := true, port := .resDma }

    | .dmaView _ _, .dmaView vp cp _ _ =>
      let rawInExt := 2 * cp.w + (cp.h - 1) * cp.inRowStride + (cp.c - 1) * cp.inPlaneStride +
        (cp.d - 1) * cp.inDStride + (cp.n - 1) * cp.inNStride
      match src1St with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dramScratch off _ =>
        let srcSz := viewSrcL2Bytes cp vp.tiling
        l2Accs := l2Accs.push { lo := vp.srcStageBase, hi := vp.srcStageBase + srcSz, isWrite := true, role := .src1Stage }
        let actInOff := off + cp.inOff
        dramAccs := dramAccs.push { resource := .scratch, lo := actInOff, hi := alignUp (actInOff + rawInExt) dmaLineBytes, isWrite := false, port := .src1Dma }
      | .surface idx _ _ bOff _ _ =>
        let srcSz := viewSrcL2Bytes cp vp.tiling
        l2Accs := l2Accs.push { lo := vp.srcStageBase, hi := vp.srcStageBase + srcSz, isWrite := true, role := .src1Stage }
        let actInOff := bOff + cp.inOff
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := actInOff, hi := alignUp (actInOff + rawInExt) dmaLineBytes, isWrite := false, port := .src1Dma }

      let (rw, rch) := if cp.transposeCW then (cp.c, cp.w) else (cp.w, cp.c)
      let rawOutExt := 2 * rw + (cp.h - 1) * cp.outRowStride + (rch - 1) * cp.outPlaneStride +
        (cp.d - 1) * cp.outDStride + (cp.n - 1) * cp.outNStride
      match resSt with
      | .l2 base bytes =>
        l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dramScratch off _ =>
        let resSz := viewResL2Bytes mp.target cp vp.tiling
        l2Accs := l2Accs.push { lo := vp.resStageBase, hi := vp.resStageBase + resSz, isWrite := true, role := .resStage }
        let actOutOff := off + cp.outOff
        let cov := classifyStridedWriteCoverage cp.n cp.d rch cp.h rw actOutOff cp.outNStride cp.outDStride cp.outPlaneStride cp.outRowStride
        dramAccs := dramAccs.push { resource := .scratch, lo := actOutOff, hi := alignUp (actOutOff + rawOutExt) dmaLineBytes, isWrite := true, port := .resDma, coverage := cov }
      | .surface idx _ _ bOff _ _ =>
        let resSz := viewResL2Bytes mp.target cp vp.tiling
        l2Accs := l2Accs.push { lo := vp.resStageBase, hi := vp.resStageBase + resSz, isWrite := true, role := .resStage }
        let actOutOff := bOff + cp.outOff
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := actOutOff, hi := alignUp (actOutOff + rawOutExt) dmaLineBytes, isWrite := true, port := .resDma }

    | .neMatmul heads m k nCols _ _ _ m0 totalM, .neMatmul mmPlan =>
      let effTotalM := if totalM == 0 then m else totalM
      let aSt ← resolveBufferStorage mp.buffers mt.inputs[0]!
      let bSt ← resolveBufferStorage mp.buffers mt.inputs[1]!
      src1St := bSt
      kernSt? := some aSt
      let chunkSize := matmulChunkSize k
      let kBaseOff := m0 * chunkSize
      let kBatchSt := effTotalM * chunkSize
      match aSt with
      | .surface idx _ _ bOff _ _ =>
        if heads > 1 && effTotalM > m then
          for bIdx in [:heads] do
            let lo := bOff + kBaseOff + bIdx * kBatchSt
            dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := lo, hi := lo + m * chunkSize, isWrite := false, port := .kernel }
        else
          dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := bOff + kBaseOff, hi := bOff + kBaseOff + heads * m * chunkSize, isWrite := false, port := .kernel }
      | .dramScratch off _ =>
        let actKOff := off + kBaseOff
        if heads > 1 && effTotalM > m then
          for bIdx in [:heads] do
            let lo := actKOff + bIdx * kBatchSt
            dramAccs := dramAccs.push { resource := .scratch, lo := lo, hi := lo + m * chunkSize, isWrite := false, port := .kernel }
        else
          dramAccs := dramAccs.push { resource := .scratch, lo := actKOff, hi := actKOff + heads * m * chunkSize, isWrite := false, port := .kernel }
        let mut latestW? : Option Nat := none
        for j in [:out.size] do
          for a in out[j]!.dramAccesses do
            if a.resource == .scratch && a.isWrite then
              if dramAccs.any (fun kb => kb.port == .kernel && a.lo < kb.hi && kb.lo < a.hi) then
                latestW? := some j
        match latestW? with
        | some p => extProd? := some p
        | none => throw s!"internal error: matmul task '{mt.name}' scratch kernel at {toHex actKOff} has no prior writer"
      | .l2 _ _ => throw s!"internal error: matmul task '{mt.name}' kernel operand A cannot be L2-resident"

      let stripW := (128 : Nat) >>> mmPlan.tiling.shPref
      let rowBN := alignUp (2 * nCols) dmaLineBytes
      match bSt with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := false, role := .src1L2 }
      | .dramScratch off _ =>
        let sz := k * (2 * min (alignUp nCols 8) (max stripW 32))
        l2Accs := l2Accs.push { lo := mmPlan.srcStageBase, hi := mmPlan.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + heads * k * rowBN, isWrite := false, port := .src1Dma }
      | .surface idx _ _ bOff _ _ =>
        let sz := k * (2 * min (alignUp nCols 8) (max stripW 32))
        l2Accs := l2Accs.push { lo := mmPlan.srcStageBase, hi := mmPlan.srcStageBase + sz, isWrite := true, role := .src1Stage }
        dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := bOff, hi := bOff + heads * k * rowBN, isWrite := false, port := .src1Dma }

      let yBatchSt := effTotalM * rowBN
      let dchs := 2 * min (alignUp nCols 8) (max stripW 32)
      let szRes := if mmPlan.direct then ((1 : Nat) <<< mmPlan.tiling.f) * 512 else m * dchs
      match resSt with
      | .l2 base bytes => l2Accs := l2Accs.push { lo := base, hi := base + bytes, isWrite := true, role := .resL2 }
      | .dramScratch off _ =>
        l2Accs := l2Accs.push { lo := mmPlan.resStageBase, hi := mmPlan.resStageBase + szRes, isWrite := true, role := .resStage }
        if heads > 1 && effTotalM > m then
          for bIdx in [:heads] do
            let lo := off + bIdx * yBatchSt
            dramAccs := dramAccs.push { resource := .scratch, lo := lo, hi := lo + m * rowBN, isWrite := true, port := .resDma, coverage := .dense lo (lo + m * rowBN) }
        else
          dramAccs := dramAccs.push { resource := .scratch, lo := off, hi := off + heads * m * rowBN, isWrite := true, port := .resDma, coverage := .dense off (off + heads * m * rowBN) }
      | .surface idx _ _ bOff _ _ =>
        l2Accs := l2Accs.push { lo := mmPlan.resStageBase, hi := mmPlan.resStageBase + szRes, isWrite := true, role := .resStage }
        if heads > 1 && effTotalM > m then
          for bIdx in [:heads] do
            let lo := bOff + bIdx * yBatchSt
            dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := lo, hi := lo + m * rowBN, isWrite := true, port := .resDma, coverage := .dense lo (lo + m * rowBN) }
        else
          dramAccs := dramAccs.push { resource := .surface idx (surfNameAt allSurfs idx), lo := bOff, hi := bOff + heads * m * rowBN, isWrite := true, port := .resDma, coverage := .dense bOff (bOff + heads * m * rowBN) }

    | _, _ =>
      throw s!"internal error: op/tilePlan mismatch on task '{mt.name}'"

    let src1L2Prod? := match src1St with
      | .l2 base bytes => findL2Prod l2Writers base bytes
      | _ => none
    if let .l2 base bytes := resSt then
      l2Writers := l2Writers.push (base, bytes, i)

    let hasSrc1Dma := match src1St with | .l2 _ _ => false | _ => true
    let hasSrc2Dma := match src2St? with | some (.l2 _ _) => false | some _ => true | none => false

    out := out.push {
      task := mt
      src1Storage := src1St
      src2Storage? := src2St?
      resStorage := resSt
      l2Accesses := l2Accs
      dramAccesses := dramAccs
      hasSrc1Dma := hasSrc1Dma
      hasSrc2Dma := hasSrc2Dma
      packedKernel? := packedKern?
      packedQuantKernel? := packedQuantKern?
      kernelBaseOff := kernBaseOff
      src1L2Producer? := src1L2Prod?
      kernStorage? := kernSt?
      extProducerTask? := extProd?
    }
  pure (out, kernBuf)

-- ============================================================================
-- Phase 2: Unified S1–S5 Synchronization & Barrier Fixpoint Solver
-- ============================================================================

private def intervalsOverlap (lo1 hi1 lo2 hi2 : Nat) : Bool :=
  lo1 < hi2 && lo2 < hi1

private def latestEngineBefore (pre : Array PreparedEmitTask) (eng : Engine) (i : Nat) : Option Nat := Id.run do
  let mut best : Option Nat := none
  for j in [:i] do
    if pre[j]!.task.engine == eng then best := some j
  return best

private def latestScratchWriterBefore (pre : Array PreparedEmitTask) (i lo hi : Nat) : Option Nat := Id.run do
  let mut latestW? : Option Nat := none
  for j in [:i] do
    for a in pre[j]!.dramAccesses do
      if a.resource == .scratch && a.isWrite && intervalsOverlap a.lo a.hi lo hi then
        latestW? := some j
  return latestW?

private def anyActiveMultiBatchScratchWriterBefore (pre : Array PreparedEmitTask) (i lo hi : Nat) : Bool := Id.run do
  let mut writers : Array (Nat × VerifiedDramAccess) := #[]
  for j in [:i] do
    for a in pre[j]!.dramAccesses do
      if a.resource == .scratch && a.isWrite && intervalsOverlap a.lo a.hi lo hi then
        writers := writers.push (j, a)
  for idx in [:writers.size] do
    let (j, a) := writers[idx]!
    if pre[j]!.task.isMultiBatch then
      let ovLo := max a.lo lo
      let ovHi := min a.hi hi
      let mut laterCovs : List WriteCoverage := []
      for idx2 in [idx + 1 : writers.size] do
        let (j2, a2) := writers[idx2]!
        if j2 > j then
          let cov2 := match a2.coverage with
            | .sparse =>
              match pre[j2]!.task.op with
              | .dmaView .. => .sparse
              | _ => .dense a2.lo a2.hi
            | other => other
          laterCovs := cov2 :: laterCovs
      let laterIvs := effectiveCoveredIntervals laterCovs
      if !intervalsCover laterIvs ovLo ovHi then return true
  return false

private def taskKindOf (mt : MachineTask) : TaskKind :=
  match mt.tilePlan with
  | .neConv .. => .neConv
  | .neAct .. => .neAct
  | .nePool .. => .nePool
  | .neChannelReduce .. => .neChannelReduce
  | .neMatmul .. => .neMatmul
  | .dmaView .. => .neView
  | .pe .. => .peOp
  | .pePool .. => .pePool
  | .peReduce .. => .peReduce

/-- Compute exact monotone S1–S5 barrier/dependency bits across all `MachineTask`s, uniformly
    covering DRAM RAW/WAR/WAW, L2 RAW/WAR/WAW (including cross-strip slot reuse!),
    multi-batch `depKind = 1`, BAR 3 scratch coherence, and multi-output surface writes. -/
private def computeSynchronization (pre : Array PreparedEmitTask)
    (donations : Array BufferDonation := #[]) : Array TaskSyncInfo := Id.run do
  let n := pre.size
  let mut syncs : Array TaskSyncInfo := Array.ofFn (n := n) fun i => {
    hasSrc1Dma := pre[i.val]!.hasSrc1Dma
    hasSrc2Dma := pre[i.val]!.hasSrc2Dma
  }

  -- 1. Seed L2 handoff bits (or whole-task wait on PE2 L2 broadcast src1)
  for i in [1 : n] do
    if let .l2 _ _ := pre[i]!.src1Storage then
      let isBcastSrc1 : Bool := match pre[i]!.task.tilePlan with
        | .pe pp => pp.hwSrc2?.isSome && pp.hwSrc1 != pp.outDims
        | _ => false
      if let some prod := pre[i]!.src1L2Producer? then
        let prodEng := pre[prod]!.task.engine
        if isBcastSrc1 then
          if prodEng == .ne then syncs := syncs.set! i { syncs[i]! with src1WaitNe5 := true }
          else syncs := syncs.set! i { syncs[i]! with src1WaitPe22 := true }
        else if prodEng == .ne && latestEngineBefore pre .ne i == some prod then
          syncs := syncs.set! i { syncs[i]! with src1HandoffNe := true }
        else if prodEng == .pe && latestEngineBefore pre .pe i == some prod then
          syncs := syncs.set! i { syncs[i]! with src1HandoffPe := true }

  -- 2. Seed BAR 3 scratch read coherence on PE tasks (depKind = 1 for multi-batch, else 2)
  for i in [1 : n] do
    if pre[i]!.task.engine == .pe then
      for b in pre[i]!.dramAccesses do
        if b.resource == .scratch && !b.isWrite then
          if let some j := latestScratchWriterBefore pre i b.lo b.hi then
            let depKind := if pre[i]!.task.isMultiBatch || anyActiveMultiBatchScratchWriterBefore pre i b.lo b.hi then 1 else 2
            let dist := min 15 (i - j - 1)
            if b.port == .src1Dma then
              syncs := syncs.set! i { syncs[i]! with src1DepKind := depKind, src1DepDist := dist }
            else if b.port == .src2Dma then
              syncs := syncs.set! i { syncs[i]! with src2DepKind := depKind, src2DepDist := dist }

  let mut latestNeArr : Array (Option Nat) := Array.mkEmpty n
  let mut latestPeArr : Array (Option Nat) := Array.mkEmpty n
  let mut lastNe : Option Nat := none
  let mut lastPe : Option Nat := none
  for i in [:n] do
    latestNeArr := latestNeArr.push lastNe
    latestPeArr := latestPeArr.push lastPe
    match pre[i]!.task.engine with
    | .ne => lastNe := some i
    | .pe => lastPe := some i

  -- 3. Monotone fixpoint loop (guaranteed termination in <= 8*n + 2 steps)
  for _ in [:8 * n + 2] do
    let fps : Array VerifiedTaskFootprint := Array.ofFn (n := n) fun i => {
      taskIndex := i.val
      kind := taskKindOf pre[i.val]!.task
      engine := pre[i.val]!.task.engine
      isNeCompute := pre[i.val]!.task.isNeCompute
      isMultiBatch := pre[i.val]!.task.isMultiBatch
      extProducerTask := pre[i.val]!.extProducerTask?
      l2Accesses := pre[i.val]!.l2Accesses
      dramAccesses := pre[i.val]!.dramAccesses
      sync := syncs[i.val]!
    }
    let (hbTbl, hblTbl) := buildHBReachTables fps
    let hb := fun a b => reachB hbTbl a b
    let hbl := fun a b => reachB hblTbl a b
    let hbEq := fun a b => a == b || hb a b
    let hblEq := fun a b => a == b || hbl a b
    let mut changed := false

    for i in [1 : n] do
      let ti := fps[i]!
      let latestNe := latestNeArr[i]!
      let latestPe := latestPeArr[i]!
      for k in [:i] do
        let tk := fps[k]!
        let dramRawKind : Bool :=
          syncs[i]!.hasSrc1Dma && (syncs[i]!.src1DepKind == 1 || syncs[i]!.src1DepKind == 2) &&
          syncs[i]!.src1DepDist == 0 && hbEq k (i - 1)
        let src2Kind : Bool :=
          pre[i - 1]!.task.engine == .pe && syncs[i]!.hasSrc2Dma &&
          (syncs[i]!.src2DepKind == 1 || syncs[i]!.src2DepKind == 2) &&
          syncs[i]!.src2DepDist == 0 && hbEq k (i - 1)
        let src2Wait22 : Bool :=
          (syncs[i]!.src1WaitPe22 || syncs[i]!.src2WaitPe22) &&
          match latestPe with | some j => hbEq k j | none => false
        let src2Wait5 : Bool :=
          (syncs[i]!.src1WaitNe5 || syncs[i]!.src2WaitNe5) &&
          (if tk.engine == .ne then true
           else Id.run do
            let mut ok := false
            for j in [:i] do
              if pre[j]!.task.engine == .ne && hbEq k j then ok := true; break
            return ok)

        -- A. DRAM conflicts
        for a in tk.dramAccesses do
          for b in ti.dramAccesses do
            if a.resource == b.resource && (a.isWrite || b.isWrite) && intervalsOverlap a.lo a.hi b.lo b.hi then
              if b.port == .kernel then
                if let some p := ti.extProducerTask then
                  if !hbEq k p then
                    if tk.engine == .ne then
                      if !syncs[p]!.src1WaitNe5 then
                        syncs := syncs.set! p { syncs[p]! with src1WaitNe5 := true, src1HandoffNe := false }; changed := true
                    else
                      if !syncs[p]!.src1WaitPe22 then
                        syncs := syncs.set! p { syncs[p]! with src1WaitPe22 := true, src1HandoffPe := false }; changed := true
              else if ti.engine == .ne && b.port == .src1Dma then
                let wantDepKind :=
                  if b.resource == .scratch then
                    if ti.isMultiBatch || anyActiveMultiBatchScratchWriterBefore pre i b.lo b.hi then 1 else 2
                  else 2
                if syncs[i]!.src1DepKind != wantDepKind || syncs[i]!.src1DepDist != 0 then
                  syncs := syncs.set! i { syncs[i]! with src1DepKind := wantDepKind, src1DepDist := 0 }; changed := true
                if !hbEq k (i - 1) then
                  let prev := i - 1
                  if tk.engine == .ne then
                    if !syncs[prev]!.src1WaitNe5 then
                      syncs := syncs.set! prev { syncs[prev]! with src1WaitNe5 := true, src1HandoffNe := false }; changed := true
                  else
                    if !syncs[prev]!.src1WaitPe22 then
                      syncs := syncs.set! prev { syncs[prev]! with src1WaitPe22 := true, src1HandoffPe := false }; changed := true
              else if ti.engine == .pe && b.port == .src2Dma then
                if !(src2Kind || src2Wait22 || src2Wait5 || dramRawKind) then
                  if tk.engine == .ne then
                    if !syncs[i]!.src2WaitNe5 then syncs := syncs.set! i { syncs[i]! with src2WaitNe5 := true }; changed := true
                  else
                    if !syncs[i]!.src2WaitPe22 then syncs := syncs.set! i { syncs[i]! with src2WaitPe22 := true }; changed := true
              else if a.isWrite && !b.isWrite then
                if !(hb k i || dramRawKind) then
                  if tk.engine == .ne then
                    if !syncs[i]!.src1WaitNe5 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitNe5 := true, src1HandoffNe := false }; changed := true
                  else
                    if !syncs[i]!.src1WaitPe22 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitPe22 := true, src1HandoffPe := false }; changed := true
              else
                if !hb k i then
                  if tk.engine == .ne then
                    if !syncs[i]!.src1WaitNe5 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitNe5 := true, src1HandoffNe := false }; changed := true
                  else
                    if !syncs[i]!.src1WaitPe22 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitPe22 := true, src1HandoffPe := false }; changed := true

        -- B. L2 conflicts (including cross-strip WAR/WAW when Vane.Alloc reuses rolling L2 slots!)
        for a in tk.l2Accesses do
          for b in ti.l2Accesses do
            if (a.isWrite || b.isWrite) && intervalsOverlap a.lo a.hi b.lo b.hi then
              let handoff2 : Bool :=
                a.isWrite &&
                ((b.role == .src1L2 && syncs[i]!.src1HandoffNe) || (b.role == .src2L2 && syncs[i]!.src2HandoffNe)) &&
                match latestNe with | some j => hblEq k j | none => false
              let handoff3 : Bool :=
                a.isWrite && b.role == .src1L2 && syncs[i]!.src1HandoffPe &&
                match latestPe with | some j => hblEq k j | none => false
              if ti.engine == .pe && (b.role == .src2Stage || b.role == .src2L2) then
                if !(src2Wait22 || src2Wait5 || src2Kind || (b.role == .src2L2 && handoff2)) then
                  if tk.engine == .ne then
                    if !syncs[i]!.src2WaitNe5 then syncs := syncs.set! i { syncs[i]! with src2WaitNe5 := true }; changed := true
                  else
                    if !syncs[i]!.src2WaitPe22 then syncs := syncs.set! i { syncs[i]! with src2WaitPe22 := true }; changed := true
              else if b.role == .src1Stage && a.role == .resStage then
                if !(src2Wait5 || src2Wait22) then
                  if tk.engine == .ne then
                    if !syncs[i]!.src1WaitNe5 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitNe5 := true, src1HandoffNe := false }; changed := true
                  else
                    if !syncs[i]!.src1WaitPe22 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitPe22 := true, src1HandoffPe := false }; changed := true
              else
                if !(hbl k i || handoff2 || handoff3) then
                  if tk.engine == .ne then
                    if !syncs[i]!.src1WaitNe5 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitNe5 := true, src1HandoffNe := false }; changed := true
                  else
                    if !syncs[i]!.src1WaitPe22 then
                      syncs := syncs.set! i { syncs[i]! with src1WaitPe22 := true, src1HandoffPe := false }; changed := true

      -- C. Explicit happens-before edge for every active BufferDonation (`lastReadTask < firstWriteTask`)
      for d in donations do
        if d.firstWriteTask == i && d.lastReadTask < i then
          if !hb d.lastReadTask i then
            if pre[d.lastReadTask]!.task.engine == .ne then
              if !syncs[i]!.src1WaitNe5 then
                syncs := syncs.set! i { syncs[i]! with src1WaitNe5 := true, src1HandoffNe := false }; changed := true
            else
              if !syncs[i]!.src1WaitPe22 then
                syncs := syncs.set! i { syncs[i]! with src1WaitPe22 := true, src1HandoffPe := false }; changed := true
    if !changed then break

  -- Clean up redundant handoff bits when stronger whole-task wait bits are set
  for i in [:n] do
    let s := syncs[i]!
    syncs := syncs.set! i {
      s with
      src1HandoffNe := s.src1HandoffNe && !s.src1WaitNe5
      src1HandoffPe := s.src1HandoffPe && !s.src1WaitPe22
      src2HandoffNe := s.src2HandoffNe && !s.src2WaitNe5
      src2HandoffPe := s.src2HandoffPe && !s.src2WaitPe22
    }
  return syncs

-- ============================================================================
-- Phase 3: Unified Emitters — One Clean Emitter per `AneOp` Family
-- ============================================================================

private def resolvedToPoolPortLoc (ctx : String) : ResolvedStorage → Except String PoolPortLoc
  | .surface _ _ slot bOff win _ =>
    if bOff != 0 || win.fullH > 0 then
      .error s!"{ctx}: pool/reduce does not support non-zero surface offset ({bOff}) or row window (fullH={win.fullH})"
    else
      .ok (.surface slot)
  | .dramScratch off _ => .ok (.dram off)
  | .l2 base _ => .ok (.l2 base)

/-- Emitter 1: `AneOp.neConv` (fp16 1x1, 3x3 dense/depthwise, stride-1/2, quantized M8A, fused ReLU/LUT). -/
private def emitNeConvTask
    (target : Target) (totalTasks : Nat) (kernelBarSlot : UInt8)
    (pt : PreparedEmitTask) (sync : TaskSyncInfo) (act : ConvActivation) (cp : ConvTaskPlan) :
    Except String TaskDesc := do
  let taskId := pt.task.id
  let isFirst := taskId == 0
  let isLast := taskId + 1 == totalTasks
  let taskIdU32 ← natToU32Checked taskId "taskId"
  let src1IsL2 := match pt.src1Storage with | .l2 _ _ => true | _ => false
  let resIsL2 := match pt.resStorage with | .l2 _ _ => true | _ => false
  let wSrc1Cfg ← encodeSrcPortCfgMulti "src1Cfg" isFirst src1IsL2
    sync.src1WaitNe5 sync.src1WaitPe22 sync.src1HandoffNe sync.src1HandoffPe

  let sh := cp.shape
  let sparse := sh.isSparse
  let ho := sh.outH; let wo := sh.outW
  let chs := chsBytesConv sh cp.tiling
  let dchs := dchsBytesConv sh cp.tiling
  let srcRow := srcRowStrideConv sh cp.tiling
  let resRow := resRowStrideConv sh cp.tiling

  let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked sh.winReg "Win")]
  let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked sh.hinReg "Hin")]
  let cinU32 ← natToU32Checked sh.cin "Cin"
  let wWout ← encodeWord (dim16Layout "Wout") #[("val", ← natToU32Checked wo "Wout")]
  let wHout ← encodeWord (dim16Layout "Hout") #[("val", ← natToU32Checked ho "Hout")]
  let coutU32 ← natToU32Checked sh.cout "Cout"
  let wConvCfg ← encodeWord convCfgLayout #[
    ("kw", ← natToU32Checked sh.kernelSize "kw"), ("kh", ← natToU32Checked sh.kernelSize "kh"),
    ("sx", ← natToU32Checked sh.stride "sx"), ("sy", ← natToU32Checked sh.stride "sy"),
    ("pl", ← natToU32Checked sh.padLeft "pl"), ("pt", ← natToU32Checked sh.padTop "pt"),
    ("hx", 1), ("vy", 1)
  ]
  let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked cp.tiling.tileHeight "TH")]

  let (bit27Val, wNeCfg, wNeOp, wNeScale, scalarBiasReg?, chunkSlots) ←
    match pt.packedQuantKernel? with
    | some pq =>
      let wide3x3 := computeWide3x3Conv target sh cp.tiling
      let b27 := neCommonQuantBit27 wide3x3 pq.hasZeroPoint pq.format.wKind
      let wCfg ← encodeWord neCfgQuantLayout #[
        ("wKind", pq.format.wKind), ("wWidth", ← natToU32Checked pq.format.wWidth "wWidth"),
        ("sparse", 0), ("dynKern", 0), ("hasZP", if pq.hasZeroPoint then 1 else 0),
        ("h17Flag", if target == .h17 && !pq.hasZeroPoint then 1 else 0)
      ]
      let wOp ← encodeWord neOpQuantLayout #[
        ("biasInKernel", if pq.biasInKernel then 1 else 0),
        ("unscaled", if pq.format.isPalette then neOpUnscaledBits else 0),
        ("perLaneScale", if pq.perLaneScale then 1 else 0),
        ("relu", if act.isRelu then 1 else 0), ("lut", 0), ("base20", 1)
      ]
      let wSc ← encodeWord neScaleExtLayout #[
        ("val", pq.tensorScaleFp16.toUInt32), ("pexp", ← natToU32Checked pq.scalePexp "scalePexp")
      ]
      let sb? ← some <$> encodeWord scalarBiasExtLayout #[
        ("val", 0), ("biasShift", ← natToU32Checked pq.biasShift "biasShift")
      ]
      pure (b27, wCfg, wOp, wSc, sb?, pq.slots)
    | none =>
      let some packed := pt.packedKernel? | throw s!"internal error: missing packed kernel on '{pt.task.name}'"
      let wide3x3 := computeWide3x3Conv target sh cp.tiling
      let hasLut := packed.lutPayload.isSome
      let neScaleVal : UInt32 := match packed.lutPayload with | some lp => lp.preScaleBits.toUInt32 | none => fp16OneBits
      let wCfg ← encodeWord (neCfgLayout target) #[("sparse", if sparse then 1 else 0)]
      let wOp ← encodeWord neOpLayout #[
        ("biasInKernel", if packed.biasInKernel then 1 else 0),
        ("relu", if act.isRelu && !hasLut then 1 else 0),
        ("lut", if hasLut then 1 else 0)
      ]
      let wSc ← encodeWord neScaleLayout #[("val", neScaleVal)]
      let sb? ← if sh.effMode == .dense && sh.cout == 1 then
          some <$> encodeWord scalarBiasLayout #[("val", packed.scalarBiasBits.toUInt32)]
        else pure none
      pure (if wide3x3 then 1 else 0, wCfg, wOp, wSc, sb?, packed.slots)

  let wNeCommon ← encodeWord neCommonLayout #[
    ("kind", 0), ("shPref", ← natToU32Checked cp.tiling.shPref "shPref"),
    ("shMin", ← natToU32Checked cp.tiling.shPref "shMin"), ("shMax", ← natToU32Checked cp.tiling.shMax "shMax"),
    ("F", ← natToU32Checked cp.tiling.f "F"), ("wide3x3", bit27Val)
  ]
  let wNeTile ← encodeWord neTileLayout #[
    ("ocg", ← natToU32Checked (effOcgConv sh cp.tiling) "ocg"),
    ("fat", ← natToU32Checked cp.tiling.fat "fat"),
    ("WU", ← natToU32Checked cp.tiling.wu "WU")
  ]
  let wResCfg ← encodeResPortCfgMulti isFirst resIsL2 cp.direct
  let wKernStage ← encodeKernelStageMulti taskId isLast sparse

  let mut plain : Array RegWrite := #[
    { byteAddr := regWin, value := wWin }, { byteAddr := regHin, value := wHin }, { byteAddr := regCin, value := cinU32 },
    { byteAddr := regWout, value := wWout }, { byteAddr := regHout, value := wHout }, { byteAddr := regCout, value := coutU32 },
    { byteAddr := regBatchN, value := 1 }, { byteAddr := regConvCfg, value := wConvCfg },
    { byteAddr := regTH, value := wTH }, { byteAddr := regNeCommon, value := wNeCommon },
    { byteAddr := regNeTile, value := wNeTile }, { byteAddr := regTargetCfg, value := targetCfgValue target },
    { byteAddr := regSrc1Cfg, value := wSrc1Cfg }, { byteAddr := regResCfg, value := wResCfg },
    { byteAddr := regNeCfg, value := wNeCfg }, { byteAddr := regNeOp, value := wNeOp },
    { byteAddr := regNeScale, value := wNeScale }, { byteAddr := regKernelStage, value := wKernStage }
  ]
  if sh.effMode == .depthwise then
    plain := plain.push { byteAddr := regDwGroups, value := ← encodeWord dwGroupsLayout #[("perGroupIn", 1)] }
  if isFirst then
    plain := plain.push { byteAddr := regSrc2Cfg, value := src2CfgValue }
  if let some sb := scalarBiasReg? then
    plain := plain.push { byteAddr := regScalarBias, value := sb }

  let mut barRecs : Array BarRegWrite := #[
    { baseByteAddr := regKernelBaseLo, barSlot := kernelBarSlot, values := #[← natToU32Checked pt.kernelBaseOff "kernelBaseLo", 0] }
  ]

  match pt.src1Storage with
  | .l2 base _ =>
    let l2Chs := 2 * alignUp sh.winReg 8
    plain := plain ++ #[
      { byteAddr := regSrc1Base, value := ← natToU32Checked base "src1Base" },
      { byteAddr := regSrc1ChanStride, value := ← natToU32Checked l2Chs "src1ChanStride" },
      { byteAddr := regSrc1RowStride, value := ← natToU32Checked (sh.cin * l2Chs) "src1RowStride" }
    ]
  | st =>
    let (slot, off, inRowSt, inPlaneSt) := match st with
      | .surface _ _ s bOff win _ =>
        let (wOff, rSt, pSt, _) := surfaceDmaWindow win sh.cin sh.h sh.w
        (s, bOff + wOff, rSt, pSt)
      | .dramScratch o _ =>
        let rSt := alignUp (2 * sh.w) dmaLineBytes
        (barSlotScratch, o, rSt, sh.h * rSt)
      | .l2 _ _ => (0, 0, 0, 0)
    plain := plain ++ #[
      { byteAddr := regSrc1Base, value := ← natToU32Checked cp.srcStageBase "src1Base" },
      { byteAddr := regSrc1ChanStride, value := ← natToU32Checked chs "src1ChanStride" },
      { byteAddr := regSrc1RowStride, value := ← natToU32Checked srcRow "src1RowStride" },
      { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked (sh.cin * chs) "src1RowBytesA" },
      { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked (sh.cin * chs) "src1RowBytesB" },
      { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
      { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked inRowSt "src1DmaRowStride" },
      { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked inPlaneSt "src1DmaPlaneStride" },
      { byteAddr := regSrc1Fmt, value := srcFmtWordP1 }
    ]
    let wSrc1Dep ← encodeSrcDepMulti "src1Dep" taskId sync.src1DepKind sync.src1DepDist
    barRecs := barRecs ++ #[
      { baseByteAddr := regSrc1BaseLo, barSlot := slot, values := #[← natToU32Checked off "src1BaseLo", 0] },
      { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrc1Dep] }
    ]

  match pt.resStorage with
  | .l2 base _ =>
    let l2Chs := 2 * alignUp wo 8
    plain := plain ++ #[
      { byteAddr := regResBase, value := ← natToU32Checked base "resBase" },
      { byteAddr := regResChanStride, value := ← natToU32Checked l2Chs "resChanStride" },
      { byteAddr := regResRowStride, value := ← natToU32Checked (sh.cout * l2Chs) "resRowStride" }
    ]
  | st =>
    let (slot, off, depCode, outRowSt, outPlaneSt) := match st with
      | .surface _ _ s bOff win _ =>
        let (wOff, rSt, pSt, _) := surfaceDmaWindow win sh.cout ho wo
        (s, bOff + wOff, resDepCode target, rSt, pSt)
      | .dramScratch o _ =>
        let rSt := alignUp (2 * wo) dmaLineBytes
        (barSlotScratch, o, resDepCode target ||| resDepScratchFlags, rSt, ho * rSt)
      | .l2 _ _ => (0, 0, resDepCode target, 0, 0)
    plain := plain ++ #[
      { byteAddr := regResBase, value := ← natToU32Checked cp.resStageBase "resBase" },
      { byteAddr := regResDmaRowStride, value := ← natToU32Checked outRowSt "resDmaRowStride" },
      { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked outPlaneSt "resDmaPlaneStride" },
      { byteAddr := regResFmt, value := resFmtWordP1 }
    ]
    if !cp.direct then
      let resRowBytesU32 ← natToU32Checked (sh.cout * dchs) "resRowBytes"
      plain := plain ++ #[
        { byteAddr := regResChanStride, value := ← natToU32Checked dchs "resChanStride" },
        { byteAddr := regResRowStride, value := ← natToU32Checked resRow "resRowStride" },
        { byteAddr := regResRowBytesA, value := resRowBytesU32 },
        { byteAddr := regResRowBytesB, value := resRowBytesU32 }
      ]
    let wResDep ← encodeWord resDepLayout #[("code", depCode), ("task", taskIdU32), ("direct", if cp.direct then 1 else 0)]
    barRecs := barRecs ++ #[
      { baseByteAddr := regResBaseLo, barSlot := slot, values := #[← natToU32Checked off "resBaseLo", 0] },
      { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
    ]

  let wSlotOn ← encodeWord chunkSlotLayout #[("enable", 1), ("on20", 1), ("task", taskIdU32)]
  let wSlotOff ← encodeWord chunkSlotLayout #[("enable", 0), ("on20", 1), ("task", taskIdU32)]
  for s in [:16] do
    match chunkSlots.find? (fun cs => cs.slot == s) with
    | some cs =>
      plain := plain ++ #[
        { byteAddr := regChunkSlot s, value := wSlotOn },
        { byteAddr := regChunkOffset s, value := ← natToU32Checked cs.offset s!"chunkOffset[{s}]" },
        { byteAddr := regChunkSize s, value := ← natToU32Checked cs.size s!"chunkSize[{s}]" }
      ]
    | none =>
      if !isFirst then plain := plain.push { byteAddr := regChunkSlot s, value := wSlotOff }

  if isLast then plain := plain ++ (← outputStageWrites taskId)
  pure {
    taskId := taskId, logNE := cp.tiling.logNE, eneOverride := some (cp.tiling.logNE + 1),
    isFirst := isFirst, isLast := isLast, liveL2Mask := pt.task.liveL2Mask,
    plainWrites := plain, barWrites := barRecs,
    kernelRelocs := #[{ barRecIndex := 0, payloadWordIndex := 0 }]
  }

/-- Emitter 2: `AneOp.neAct` (`relu`, `copy`, LUT activations, NE pool, NE channel reduce). -/
private def emitNeActTask
    (target : Target) (totalTasks : Nat) (kernelBarSlot : UInt8) (buffers : Array BufferStorage)
    (pt : PreparedEmitTask) (sync : TaskSyncInfo) (kind : NeActKind) : Except String TaskDesc := do
  let taskId := pt.task.id
  let isFirst := taskId == 0
  let isLast := taskId + 1 == totalTasks
  let taskIdU32 ← natToU32Checked taskId "taskId"
  let src1IsL2 := match pt.src1Storage with | .l2 _ _ => true | _ => false
  let resIsL2 := match pt.resStorage with | .l2 _ _ => true | _ => false
  let wSrc1Cfg ← encodeSrcPortCfgMulti "src1Cfg" isFirst src1IsL2
    sync.src1WaitNe5 sync.src1WaitPe22 sync.src1HandoffNe sync.src1HandoffPe

  match kind, pt.task.tilePlan with
  | .act actSpec, .neAct lp lutPayload? =>
    let dims := shape4DToDims pt.task.outShape
    let (isLut, isRelu, preScaleBits, shiftBits?) ← match actSpec with
      | .relu => pure (false, true, f16PosOne, none)
      | .copy => pure (false, false, f16PosOne, none)
      | .lut _ =>
        let some payload := lutPayload? | throw "internal error: missing LutPayload"
        pure (true, false, payload.preScaleBits, payload.inputShiftBits)
    let t := lp.tiling
    let stripW := t.stripWidth
    let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked dims.w "W")]
    let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked dims.h "H")]
    let cU32 ← natToU32Checked dims.c "C"
    let wDwGroups ← encodeWord dwGroupsLayout #[("perGroupIn", 1)]
    let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked t.tileHeight "TH")]
    let wNeCommon ← encodeWord neCommonLayout #[
      ("kind", 0), ("shPref", ← natToU32Checked t.shPref "shPref"), ("shMin", ← natToU32Checked t.shPref "shMin"),
      ("shMax", ← natToU32Checked t.shMax "shMax"), ("F", ← natToU32Checked t.f "F"), ("wide3x3", 0)
    ]
    let wNeTile ← encodeWord neTileLayout #[("ocg", 0), ("fat", ← natToU32Checked t.fat "fat"), ("WU", ← natToU32Checked t.wu "WU")]
    let wResCfg ← encodeResPortCfgMulti isFirst resIsL2 lp.direct
    let wNeActOp ← encodeWord neActOpLayout #[
      ("noShift", if shiftBits?.isSome then 0 else 1), ("relu", if isRelu then 1 else 0), ("lut", if isLut then 1 else 0)
    ]
    let wNeScale ← encodeWord neScaleLayout #[("val", preScaleBits.toUInt32)]

    let mut plain : Array RegWrite := #[
      { byteAddr := regWin, value := wWin }, { byteAddr := regHin, value := wHin }, { byteAddr := regCin, value := cU32 },
      { byteAddr := regWout, value := wWin }, { byteAddr := regHout, value := wHin }, { byteAddr := regCout, value := cU32 },
      { byteAddr := regBatchN, value := 1 }, { byteAddr := regDwGroups, value := wDwGroups },
      { byteAddr := regTH, value := wTH }, { byteAddr := regNeCommon, value := wNeCommon },
      { byteAddr := regNeTile, value := wNeTile }, { byteAddr := regTargetCfg, value := targetCfgValue target },
      { byteAddr := regSrc1Cfg, value := wSrc1Cfg }, { byteAddr := regResCfg, value := wResCfg },
      { byteAddr := regNeOp, value := wNeActOp }, { byteAddr := regNeScale, value := wNeScale }
    ]
    if isFirst then plain := plain.push { byteAddr := regSrc2Cfg, value := src2CfgValue }
    if let some shiftBits := shiftBits? then
      plain := plain ++ #[
        { byteAddr := regNeActShiftEn, value := ← encodeWord neActShiftEnLayout #[("enable", 1)] },
        { byteAddr := regScalarBias, value := ← encodeWord scalarBiasLayout #[("val", shiftBits.toUInt32)] }
      ]

    let mut barRecs : Array BarRegWrite := #[]
    let mut relocs : Array TaskReloc := #[]
    if isLut then
      let wKernStage ← encodeKernelStageMulti taskId isLast false
      let wLutCfg ← encodeWord chunkSlotLayout #[("enable", 1), ("on20", 1), ("task", taskIdU32)]
      plain := plain ++ #[
        { byteAddr := regNeCfg, value := neActCfgValue }, { byteAddr := regKernelStage, value := wKernStage },
        { byteAddr := regLutCfg, value := wLutCfg }, { byteAddr := regLutSize, value := lutTableBytes.toUInt32 }
      ]
      barRecs := barRecs.push { baseByteAddr := regLutBaseLo, barSlot := kernelBarSlot, values := #[← natToU32Checked pt.kernelBaseOff "lutBaseLo", 0] }
      relocs := #[{ barRecIndex := 0, payloadWordIndex := 0 }]
    else if isLast then
      plain := plain.push { byteAddr := regKernelStage, value := kernelStageNoKernValue }

    match pt.src1Storage with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp dims.w 8
      plain := plain ++ #[
        { byteAddr := regSrc1Base, value := ← natToU32Checked base "src1Base" },
        { byteAddr := regSrc1ChanStride, value := ← natToU32Checked l2Chs "src1ChanStride" },
        { byteAddr := regSrc1RowStride, value := ← natToU32Checked (dims.c * l2Chs) "src1RowStride" }
      ]
    | st =>
      let effSrcStrip := if actSpec == .copy && totalTasks > 1 then max stripW 32 else stripW
      let row := 2 * min (alignUp dims.w 8) effSrcStrip
      let u := dims.c * row
      let (slot, off, dmaRow, dmaPlane) := match st with
        | .surface _ _ s bOff win _ =>
          let (wOff, rSt, pSt, _) := surfaceDmaWindow win dims.c dims.h dims.w
          (s, bOff + wOff, rSt, pSt)
        | .dramScratch o _ =>
          let rSt := alignUp (2 * dims.w) dmaLineBytes
          (barSlotScratch, o, rSt, dims.h * rSt)
        | .l2 _ _ => (0, 0, 0, 0)
      plain := plain ++ #[
        { byteAddr := regSrc1Base, value := ← natToU32Checked lp.srcStageBase "src1Base" },
        { byteAddr := regSrc1ChanStride, value := ← natToU32Checked row "src1ChanStride" },
        { byteAddr := regSrc1RowStride, value := ← natToU32Checked u "src1RowStride" },
        { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked u "src1RowBytesA" },
        { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked u "src1RowBytesB" },
        { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
        { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked dmaRow "src1DmaRowStride" },
        { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked dmaPlane "src1DmaPlaneStride" },
        { byteAddr := regSrc1Fmt, value := srcFmtWordP1 }
      ]
      let wSrc1Dep ← encodeSrcDepMulti "src1Dep" taskId sync.src1DepKind sync.src1DepDist
      barRecs := barRecs ++ #[
        { baseByteAddr := regSrc1BaseLo, barSlot := slot, values := #[← natToU32Checked off "src1BaseLo", 0] },
        { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrc1Dep] }
      ]

    match pt.resStorage with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp dims.w 8
      plain := plain ++ #[
        { byteAddr := regResBase, value := ← natToU32Checked base "resBase" },
        { byteAddr := regResChanStride, value := ← natToU32Checked l2Chs "resChanStride" },
        { byteAddr := regResRowStride, value := ← natToU32Checked (dims.c * l2Chs) "resRowStride" }
      ]
    | st =>
      let (slot, off, depCode, dmaRow, dmaPlane) := match st with
        | .surface _ _ s bOff win _ =>
          let (wOff, rSt, pSt, _) := surfaceDmaWindow win dims.c dims.h dims.w
          (s, bOff + wOff, resDepCode target, rSt, pSt)
        | .dramScratch o _ =>
          let rSt := alignUp (2 * dims.w) dmaLineBytes
          (barSlotScratch, o, resDepCode target ||| resDepScratchFlags, rSt, dims.h * rSt)
        | .l2 _ _ => (0, 0, resDepCode target, 0, 0)
      plain := plain ++ #[
        { byteAddr := regResBase, value := ← natToU32Checked lp.resStageBase "resBase" },
        { byteAddr := regResDmaRowStride, value := ← natToU32Checked dmaRow "resDmaRowStride" },
        { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked dmaPlane "resDmaPlaneStride" },
        { byteAddr := regResFmt, value := resFmtWordP1 }
      ]
      if !lp.direct then
        let drow := 2 * min (alignUp dims.w 8) (max stripW 32)
        let du := dims.c * drow
        plain := plain ++ #[
          { byteAddr := regResChanStride, value := ← natToU32Checked drow "resChanStride" },
          { byteAddr := regResRowStride, value := ← natToU32Checked du "resRowStride" },
          { byteAddr := regResRowBytesA, value := ← natToU32Checked du "resRowBytesA" },
          { byteAddr := regResRowBytesB, value := ← natToU32Checked du "resRowBytesB" }
        ]
      let wResDep ← encodeWord resDepLayout #[("code", depCode), ("task", taskIdU32), ("direct", if lp.direct then 1 else 0)]
      barRecs := barRecs ++ #[
        { baseByteAddr := regResBaseLo, barSlot := slot, values := #[← natToU32Checked off "resBaseLo", 0] },
        { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
      ]

    if isLast then plain := plain ++ (← outputStageWrites taskId)
    let h4Override : Option UInt32 :=
      if isFirst && isLast && actSpec == .copy then some taskHeaderWord4CopyConst else none
    pure {
      taskId := taskId, logNE := t.logNE, eneOverride := some (t.logNE + 1),
      header4Override := h4Override, isFirst := isFirst, isLast := isLast,
      liveL2Mask := pt.task.liveL2Mask, plainWrites := plain, barWrites := barRecs, kernelRelocs := relocs
    }

  | .nePool op .., .nePool np gh gw =>
    let dims := shape4DToDims buffers[pt.task.inputs[0]!]!.shape
    let srcLoc ← resolvedToPoolPortLoc s!"nePool '{pt.task.name}' src1" pt.src1Storage
    let resLoc ← resolvedToPoolPortLoc s!"nePool '{pt.task.name}' res" pt.resStorage
    buildNePoolTaskMulti target op dims gh gw np taskId totalTasks
      srcLoc resLoc sync pt.task.liveL2Mask

  | .channelReduce op, .neChannelReduce nc =>
    let dims := shape4DToDims buffers[pt.task.inputs[0]!]!.shape
    let srcLoc ← resolvedToPoolPortLoc s!"channelReduce '{pt.task.name}' src1" pt.src1Storage
    let resLoc ← resolvedToPoolPortLoc s!"channelReduce '{pt.task.name}' res" pt.resStorage
    buildNeChannelReduceTaskMulti target op dims nc taskId totalTasks
      srcLoc resLoc sync pt.task.liveL2Mask

  | _, _ => throw s!"internal error: neAct op/plan mismatch on '{pt.task.name}'"

/-- Emitter 3: `AneOp.neMatmul` (batched dynamic runtime-kernel NE matmul). -/
private def emitNeMatmulTask
    (target : Target) (totalTasks : Nat) (pt : PreparedEmitTask) (sync : TaskSyncInfo)
    (heads m k nCols : Nat) (scaleBits : UInt16) (m0 totalM : Nat) (mmPlan : MatmulTaskPlan)
    (kernKeepLive : Bool) : Except String TaskDesc := do
  let effTotalM := if totalM == 0 then m else totalM
  let some aSt := pt.kernStorage? | throw s!"internal error: matmul '{pt.task.name}' missing kernel storage"
  let chunkSize := matmulChunkSize k
  let (kernSlot, kernOff, kernIsScratch) ← match aSt with
    | .surface _ _ slot bOff _ _ => pure (slot, bOff + m0 * chunkSize, false)
    | .dramScratch off _ => pure (barSlotScratch, off + m0 * chunkSize, true)
    | .l2 _ _ => throw s!"internal error: matmul '{pt.task.name}' kernel cannot be L2-resident"
  let toMmSrcLoc := fun
    | ResolvedStorage.surface _ _ slot bOff _ _ => MatmulPortLoc.surface slot bOff
    | ResolvedStorage.dramScratch off _ => MatmulPortLoc.dram off
    | ResolvedStorage.l2 base _ => MatmulPortLoc.l2 base
  let toMmResLoc := fun
    | ResolvedStorage.surface _ _ slot bOff _ _ => MatmulPortLoc.surface slot bOff
    | ResolvedStorage.dramScratch off _ => MatmulPortLoc.dram off
    | ResolvedStorage.l2 base _ => MatmulPortLoc.l2 base
  buildMatmulTaskMulti target heads m k nCols scaleBits mmPlan pt.task.id totalTasks
    kernSlot kernOff kernIsScratch pt.extProducerTask?
    (toMmSrcLoc pt.src1Storage) (toMmResLoc pt.resStorage) sync pt.task.liveL2Mask kernKeepLive effTotalM

/-- Emitter 4: `AneOp.dmaView` (general strided 4D/batched DMA view — supports DMA↔DMA, Surface→L2, L2→Surface). -/
private def emitDmaViewTask
    (target : Target) (totalTasks : Nat) (pt : PreparedEmitTask) (sync : TaskSyncInfo)
    (vp : ViewTaskPlan) (cp : ViewCopy) : Except String TaskDesc := do
  let taskId := pt.task.id
  let isFirst := taskId == 0
  let isLast := taskId + 1 == totalTasks
  let srcIsL2 := match pt.src1Storage with | .l2 _ _ => true | _ => false
  let resIsL2 := match pt.resStorage with | .l2 _ _ => true | _ => false

  if !srcIsL2 && !resIsL2 then
    let (inSlot, inBase) ← match pt.src1Storage with
      | .surface _ _ slot bOff _ _ => pure (slot, bOff)
      | .dramScratch off _ => pure (barSlotScratch, off)
      | .l2 _ _ => throw "unreachable"
    let (outSlot, outBase, outIsScratch) ← match pt.resStorage with
      | .surface _ _ slot bOff _ _ => pure (slot, bOff, false)
      | .dramScratch off _ => pure (barSlotScratch, off, true)
      | .l2 _ _ => throw "unreachable"
    let effOutOff := outBase + cp.outOff
    let cp' : ViewCopy := { cp with inOff := inBase + cp.inOff, outOff := effOutOff }
    return ← buildViewTaskMulti target cp' vp taskId totalTasks inSlot outSlot outIsScratch sync pt.task.liveL2Mask

  -- L2-assisted `dmaView` (Surface->L2 `w_slice` or L2->Surface `w_stitch`, N=1, D=1, !transposeCW)
  let t := vp.tiling
  let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked cp.w "Win")]
  let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked cp.h "Hin")]
  let cU32 ← natToU32Checked cp.c "Cin"
  let wBatchN ← encodeWord (dim16Layout "BatchN") #[("val", 1)]
  let wDw ← encodeWord dwGroupsLayout #[("perGroupIn", 1)]
  let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked t.tileHeight "TH")]
  let wNeCommon ← encodeWord neViewCommonLayout #[
    ("kind", 0), ("shPref", ← natToU32Checked t.shPref "shPref"), ("shMin", ← natToU32Checked t.shPref "shMin"),
    ("shMax", ← natToU32Checked t.shMax "shMax"), ("F", ← natToU32Checked t.f "F"),
    ("transposeOut", 0), ("transposeWideGroup", 0)
  ]
  let wNeTile ← encodeWord neTileLayout #[
    ("ocg", ← natToU32Checked t.ocg "ocg"), ("fat", ← natToU32Checked t.fat "fat"), ("WU", ← natToU32Checked t.wu "WU")
  ]
  let wSrc1Cfg ← encodeSrcPortCfgMulti "src1Cfg" isFirst srcIsL2
    sync.src1WaitNe5 sync.src1WaitPe22 sync.src1HandoffNe sync.src1HandoffPe
  let wResCfg ← encodeResPortCfgMulti isFirst resIsL2 vp.direct

  let mut plain : Array RegWrite := #[
    { byteAddr := regWin, value := wWin }, { byteAddr := regHin, value := wHin }, { byteAddr := regCin, value := cU32 },
    { byteAddr := regWout, value := wWin }, { byteAddr := regHout, value := wHin }, { byteAddr := regCout, value := cU32 },
    { byteAddr := regBatchN, value := wBatchN }, { byteAddr := regDwGroups, value := wDw },
    { byteAddr := regTH, value := wTH }, { byteAddr := regNeCommon, value := wNeCommon },
    { byteAddr := regNeTile, value := wNeTile }, { byteAddr := regTargetCfg, value := targetCfgValue target },
    { byteAddr := regSrc1Cfg, value := wSrc1Cfg }, { byteAddr := regResCfg, value := wResCfg },
    { byteAddr := regNeOp, value := neOpViewCopy }, { byteAddr := regNeScale, value := neScaleOne }
  ]
  if isFirst then plain := plain.push { byteAddr := regSrc2Cfg, value := src2CfgValue }

  let mut barRecs : Array BarRegWrite := #[]
  let taskIdU32 ← natToU32Checked taskId "taskId"

  match pt.src1Storage with
  | .l2 base _ =>
    let l2Chs := 2 * alignUp cp.w 8
    plain := plain ++ #[
      { byteAddr := regSrc1Base, value := ← natToU32Checked base "src1Base" },
      { byteAddr := regSrc1ChanStride, value := ← natToU32Checked l2Chs "src1ChanStride" },
      { byteAddr := regSrc1RowStride, value := ← natToU32Checked (cp.c * l2Chs) "src1RowStride" }
    ]
  | st =>
    let (inSlot, inBase) := match st with
      | .surface _ _ s bOff _ _ => (s, bOff)
      | .dramScratch o _ => (barSlotScratch, o)
      | .l2 _ _ => (0, 0)
    let srcChs := viewSrcChanStride cp t
    let u := cp.c * srcChs
    let srcRow := viewSrcRowStride cp t
    plain := plain ++ #[
      { byteAddr := regSrc1Base, value := ← natToU32Checked vp.srcStageBase "src1Base" },
      { byteAddr := regSrc1ChanStride, value := ← natToU32Checked srcChs "src1ChanStride" },
      { byteAddr := regSrc1RowStride, value := ← natToU32Checked srcRow "src1RowStride" },
      { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked u "src1RowBytesA" },
      { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked u "src1RowBytesB" },
      { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
      { byteAddr := regSrc1Fmt, value := srcFmtWordP1 }
    ]
    if cp.h > 1 then plain := plain.push { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked cp.inRowStride "src1DmaRowStride" }
    if cp.c > 1 then plain := plain.push { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked cp.inPlaneStride "src1DmaPlaneStride" }
    let wSrc1Dep ← encodeSrcDepMulti "src1Dep" taskId sync.src1DepKind sync.src1DepDist
    barRecs := barRecs ++ #[
      { baseByteAddr := regSrc1BaseLo, barSlot := inSlot, values := #[← natToU32Checked (inBase + cp.inOff) "src1BaseLo", 0] },
      { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrc1Dep] }
    ]

  match pt.resStorage with
  | .l2 base _ =>
    let l2Chs := 2 * alignUp cp.w 8
    plain := plain ++ #[
      { byteAddr := regResBase, value := ← natToU32Checked base "resBase" },
      { byteAddr := regResChanStride, value := ← natToU32Checked l2Chs "resChanStride" },
      { byteAddr := regResRowStride, value := ← natToU32Checked (cp.c * l2Chs) "resRowStride" }
    ]
  | st =>
    let (outSlot, outOff, outIsScratch) := match st with
      | .surface _ _ s bOff _ _ => (s, bOff + cp.outOff, false)
      | .dramScratch o _ => (barSlotScratch, o + cp.outOff, true)
      | .l2 _ _ => (0, 0, false)
    plain := plain ++ #[
      { byteAddr := regResBase, value := ← natToU32Checked vp.resStageBase "resBase" },
      { byteAddr := regResFmt, value := resFmtWordP1 }
    ]
    if !vp.direct then
      let resChs := viewResChanStride cp t
      let resRowBytes := cp.c * resChs
      let resRow := viewResRowStride cp t
      plain := plain ++ #[
        { byteAddr := regResChanStride, value := ← natToU32Checked resChs "resChanStride" },
        { byteAddr := regResRowStride, value := ← natToU32Checked resRow "resRowStride" },
        { byteAddr := regResRowBytesA, value := ← natToU32Checked resRowBytes "resRowBytesA" },
        { byteAddr := regResRowBytesB, value := ← natToU32Checked resRowBytes "resRowBytesB" }
      ]
    if cp.h > 1 then plain := plain.push { byteAddr := regResDmaRowStride, value := ← natToU32Checked cp.outRowStride "resDmaRowStride" }
    if cp.c > 1 then plain := plain.push { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked cp.outPlaneStride "resDmaPlaneStride" }
    let resDepC := if outIsScratch then resDepCode target ||| resDepScratchFlags else resDepCode target
    let wResDep ← encodeWord resDepLayout #[("code", resDepC), ("task", taskIdU32), ("direct", if vp.direct then 1 else 0)]
    barRecs := barRecs ++ #[
      { baseByteAddr := regResBaseLo, barSlot := outSlot, values := #[← natToU32Checked outOff "resBaseLo", 0] },
      { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
    ]

  if isLast then
    plain := plain.push { byteAddr := regKernelStage, value := kernelStageNoKernValue }
    plain := plain ++ (← outputStageWrites taskId)

  pure {
    taskId := taskId, logNE := t.logNE, eneOverride := some (t.logNE + 1),
    isFirst := isFirst, isLast := isLast, liveL2Mask := pt.task.liveL2Mask,
    plainWrites := plain, barWrites := barRecs, kernelRelocs := #[]
  }

/-- Emitter 5: `AneOp.pe` (unary, binary broadcast, PE pool, PE reduce). -/
private def emitPeTask
    (target : Target) (totalTasks : Nat) (buffers : Array BufferStorage)
    (pt : PreparedEmitTask) (sync : TaskSyncInfo) (peOp : PeMachineOp) : Except String TaskDesc := do
  let taskId := pt.task.id
  let isFirst := taskId == 0
  let isLast := taskId + 1 == totalTasks
  let taskIdU32 ← natToU32Checked taskId "taskId"
  let src1IsL2 := match pt.src1Storage with | .l2 _ _ => true | _ => false
  let resIsL2 := match pt.resStorage with | .l2 _ _ => true | _ => false
  let wSrc1Cfg ← encodeSrcPortCfgMulti "src1Cfg" isFirst src1IsL2
    sync.src1WaitNe5 sync.src1WaitPe22 sync.src1HandoffNe sync.src1HandoffPe

  match peOp, pt.task.tilePlan with
  | .unary op, .pe pp =>
    let dims := pp.outDims
    let stripW := peStripWidth pp.tiling.k
    let th := pp.tiling.tileHeight
    let m := peTileM false pp.tiling.k th
    let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked dims.w "W")]
    let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked dims.h "H")]
    let cU32 ← natToU32Checked dims.c "C"
    let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked th "TH")]
    let wPeCommon ← encodeWord peCommonLayout #[("mode", peCommonOneSrcValue)]
    let wPeTile ← encodeWord peTileLayout #[("k", ← natToU32Checked pp.tiling.k "k"), ("m", ← natToU32Checked m "m")]
    let wResCfg ← encodeResPortCfgMulti isFirst resIsL2 pp.direct

    let mut plain : Array RegWrite := #[
      { byteAddr := regWin, value := wWin }, { byteAddr := regHin, value := wHin }, { byteAddr := regCin, value := cU32 },
      { byteAddr := regWout, value := wWin }, { byteAddr := regHout, value := wHin }, { byteAddr := regCout, value := cU32 },
      { byteAddr := regBatchN, value := 1 }, { byteAddr := regTH, value := wTH },
      { byteAddr := regPeCommon, value := wPeCommon }, { byteAddr := regPeTile, value := wPeTile },
      { byteAddr := regTargetCfg, value := 0 }, { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
      { byteAddr := regResCfg, value := wResCfg }
    ]
    if isFirst then plain := plain.push { byteAddr := regSrc2Cfg, value := src2CfgValue }
    match op with
    | .abs => plain := plain.push { byteAddr := regPeOp, value := peOpAbsValue }
    | .square => plain := plain.push { byteAddr := regPeOp, value := peOpSquareValue }
    | .chain inScale bias scale =>
      if let some b := bias then plain := plain.push { byteAddr := regPeBias, value := f16ToF32Bits b }
      if let some s := scale then plain := plain.push { byteAddr := regPeScale, value := f16ToF32Bits s }
      if let some ins := inScale then plain := plain.push { byteAddr := regPeInScale, value := f16ToF32Bits ins }
    | .binary .. => throw "internal error: binary op in pe unary emitter"

    let mut barRecs : Array BarRegWrite := #[]
    match pt.src1Storage with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp dims.w 8
      plain := plain ++ #[
        { byteAddr := regSrc1Base, value := ← natToU32Checked base "src1Base" },
        { byteAddr := regSrc1ChanStride, value := ← natToU32Checked l2Chs "src1ChanStride" },
        { byteAddr := regSrc1RowStride, value := ← natToU32Checked (dims.c * l2Chs) "src1RowStride" }
      ]
    | st =>
      let row1 := 2 * min (alignUp dims.w 8) (max stripW 32)
      let u1 := dims.c * row1
      let (slot, off, dmaRow, dmaPlane) := match st with
        | .surface _ _ s bOff win _ =>
          let (wOff, rSt, pSt, _) := surfaceDmaWindow win dims.c dims.h dims.w
          (s, bOff + wOff, rSt, pSt)
        | .dramScratch o _ =>
          let rSt := alignUp (2 * dims.w) dmaLineBytes
          (barSlotScratch, o, rSt, dims.h * rSt)
        | .l2 _ _ => (0, 0, 0, 0)
      plain := plain ++ #[
        { byteAddr := regSrc1Base, value := ← natToU32Checked pp.src1StageBase "src1Base" },
        { byteAddr := regSrc1ChanStride, value := ← natToU32Checked row1 "src1ChanStride" },
        { byteAddr := regSrc1RowStride, value := ← natToU32Checked u1 "src1RowStride" },
        { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked u1 "src1RowBytesA" },
        { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked u1 "src1RowBytesB" },
        { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
        { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked dmaRow "src1DmaRowStride" },
        { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked dmaPlane "src1DmaPlaneStride" },
        { byteAddr := regSrc1Fmt, value := srcFmtWordP1 }
      ]
      let wSrc1Dep ← encodeSrcDepMulti "src1Dep" taskId sync.src1DepKind sync.src1DepDist
      barRecs := barRecs ++ #[
        { baseByteAddr := regSrc1BaseLo, barSlot := slot, values := #[← natToU32Checked off "src1BaseLo", 0] },
        { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrc1Dep] }
      ]

    match pt.resStorage with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp dims.w 8
      plain := plain ++ #[
        { byteAddr := regResBase, value := ← natToU32Checked base "resBase" },
        { byteAddr := regResChanStride, value := ← natToU32Checked l2Chs "resChanStride" },
        { byteAddr := regResRowStride, value := ← natToU32Checked (dims.c * l2Chs) "resRowStride" }
      ]
    | st =>
      let (slot, off, depCode, dmaRow, dmaPlane) := match st with
        | .surface _ _ s bOff win _ =>
          let (wOff, rSt, pSt, _) := surfaceDmaWindow win dims.c dims.h dims.w
          (s, bOff + wOff, resDepCode target, rSt, pSt)
        | .dramScratch o _ =>
          let rSt := alignUp (2 * dims.w) dmaLineBytes
          (barSlotScratch, o, resDepCode target ||| resDepScratchFlags, rSt, dims.h * rSt)
        | .l2 _ _ => (0, 0, resDepCode target, 0, 0)
      plain := plain ++ #[
        { byteAddr := regResBase, value := ← natToU32Checked pp.resStageBase "resBase" },
        { byteAddr := regResDmaRowStride, value := ← natToU32Checked dmaRow "resDmaRowStride" },
        { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked dmaPlane "resDmaPlaneStride" },
        { byteAddr := regResFmt, value := resFmtWordP1 }
      ]
      if !pp.direct then
        let drow := 2 * min (alignUp dims.w 8) (max stripW 32)
        let du := dims.c * drow
        plain := plain ++ #[
          { byteAddr := regResChanStride, value := ← natToU32Checked drow "resChanStride" },
          { byteAddr := regResRowStride, value := ← natToU32Checked du "resRowStride" },
          { byteAddr := regResRowBytesA, value := ← natToU32Checked du "resRowBytesA" },
          { byteAddr := regResRowBytesB, value := ← natToU32Checked du "resRowBytesB" }
        ]
      let wResDep ← encodeWord resDepLayout #[("code", depCode), ("task", taskIdU32), ("direct", if pp.direct then 1 else 0)]
      barRecs := barRecs ++ #[
        { baseByteAddr := regResBaseLo, barSlot := slot, values := #[← natToU32Checked off "resBaseLo", 0] },
        { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
      ]

    if isLast then
      plain := plain.push { byteAddr := regKernelStage, value := kernelStageNoKernValue }
      plain := plain ++ (← outputStageWrites taskId)
    pure {
      taskId := taskId, logNE := 0, eneOverride := some 0,
      isFirst := isFirst, isLast := isLast, liveL2Mask := pt.task.liveL2Mask,
      plainWrites := plain, barWrites := barRecs, kernelRelocs := #[]
    }

  | .binary binOp _, .pe pp =>
    let outDims := pp.outDims
    let hw1 := pp.hwSrc1
    let hw2 := pp.hwSrc2?.getD outDims
    let stripW := peStripWidth pp.tiling.k
    let th := pp.tiling.tileHeight
    let m := peTileM true pp.tiling.k th
    let s2St := pt.src2Storage?.get!
    let src2IsL2 := match s2St with | .l2 _ _ => true | _ => false
    let wSrc2Cfg ← encodeSrcPortCfgMulti "src2Cfg" isFirst src2IsL2
      sync.src2WaitNe5 sync.src2WaitPe22 sync.src2HandoffNe sync.src2HandoffPe
    let wWin ← encodeWord (dim16Layout "Win") #[("val", ← natToU32Checked outDims.w "W")]
    let wHin ← encodeWord (dim16Layout "Hin") #[("val", ← natToU32Checked outDims.h "H")]
    let cU32 ← natToU32Checked outDims.c "C"
    let wTH ← encodeWord (dim16Layout "TH") #[("val", ← natToU32Checked th "TH")]
    let wPeCommon ← encodeWord peCommonLayout #[("mode", peCommonTwoSrcValue)]
    let wPeTile ← encodeWord peTileLayout #[("k", ← natToU32Checked pp.tiling.k "k"), ("m", ← natToU32Checked m "m")]
    let wPeBcast ← encodeWord peBcastLayout #[
      ("src1W", if hw1.w != outDims.w then 1 else 0), ("src1H", if hw1.h != outDims.h then 1 else 0), ("src1C", if hw1.c != outDims.c then 1 else 0),
      ("src2W", if hw2.w != outDims.w then 1 else 0), ("src2H", if hw2.h != outDims.h then 1 else 0), ("src2C", if hw2.c != outDims.c then 1 else 0)
    ]
    let wResCfg ← encodeResPortCfgMulti isFirst resIsL2 pp.direct

    let mut plain : Array RegWrite := #[
      { byteAddr := regWin, value := wWin }, { byteAddr := regHin, value := wHin }, { byteAddr := regCin, value := cU32 },
      { byteAddr := regWout, value := wWin }, { byteAddr := regHout, value := wHin }, { byteAddr := regCout, value := cU32 },
      { byteAddr := regBatchN, value := 1 }, { byteAddr := regTH, value := wTH },
      { byteAddr := regPeCommon, value := wPeCommon }, { byteAddr := regPeTile, value := wPeTile },
      { byteAddr := regPeBcast, value := wPeBcast }, { byteAddr := regTargetCfg, value := 0 },
      { byteAddr := regSrc1Cfg, value := wSrc1Cfg }, { byteAddr := regSrc2Cfg, value := wSrc2Cfg },
      { byteAddr := regResCfg, value := wResCfg }, { byteAddr := regPeOp, value := binOp.regValue }
    ]

    let mut barRecs : Array BarRegWrite := #[]
    match pt.src1Storage with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp hw1.w 8
      plain := plain ++ #[
        { byteAddr := regSrc1Base, value := ← natToU32Checked base "src1Base" },
        { byteAddr := regSrc1ChanStride, value := ← natToU32Checked l2Chs "src1ChanStride" },
        { byteAddr := regSrc1RowStride, value := ← natToU32Checked (hw1.c * l2Chs) "src1RowStride" }
      ]
    | st =>
      let whole1 := hw1.c == 1 && hw1.h == 1 && hw1.w > 1 && outDims.h > 1 && th < outDims.h
      let row1 := 2 * (if whole1 then alignUp hw1.w 8 else min (alignUp hw1.w 8) (max stripW 32))
      let u1 := hw1.c * row1
      let (slot, off, dmaRow, dmaPlane) := match st with
        | .surface _ _ s bOff win _ =>
          let (wOff, rSt, pSt, _) := surfaceDmaWindow win hw1.c hw1.h hw1.w
          (s, bOff + wOff, rSt, pSt)
        | .dramScratch o _ =>
          let rSt := alignUp (2 * hw1.w) dmaLineBytes
          (barSlotScratch, o, rSt, hw1.h * rSt)
        | .l2 _ _ => (0, 0, 0, 0)
      plain := plain ++ #[
        { byteAddr := regSrc1Base, value := ← natToU32Checked pp.src1StageBase "src1Base" },
        { byteAddr := regSrc1ChanStride, value := ← natToU32Checked row1 "src1ChanStride" },
        { byteAddr := regSrc1RowStride, value := ← natToU32Checked u1 "src1RowStride" },
        { byteAddr := regSrc1RowBytesA, value := ← natToU32Checked u1 "src1RowBytesA" },
        { byteAddr := regSrc1RowBytesB, value := ← natToU32Checked u1 "src1RowBytesB" },
        { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
        { byteAddr := regSrc1DmaRowStride, value := ← natToU32Checked dmaRow "src1DmaRowStride" },
        { byteAddr := regSrc1DmaPlaneStride, value := ← natToU32Checked dmaPlane "src1DmaPlaneStride" },
        { byteAddr := regSrc1Fmt, value := srcFmtWordP1 }
      ]
      let wSrc1Dep ← encodeSrcDepMulti "src1Dep" taskId sync.src1DepKind sync.src1DepDist
      barRecs := barRecs ++ #[
        { baseByteAddr := regSrc1BaseLo, barSlot := slot, values := #[← natToU32Checked off "src1BaseLo", 0] },
        { baseByteAddr := regSrc1Dep, barSlot := barSlotConfigZero, values := #[wSrc1Dep] }
      ]

    match s2St with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp hw2.w 8
      plain := plain ++ #[
        { byteAddr := regSrc2Base, value := ← natToU32Checked base "src2Base" },
        { byteAddr := regSrc2ChanStride, value := ← natToU32Checked l2Chs "src2ChanStride" },
        { byteAddr := regSrc2RowStride, value := ← natToU32Checked (hw2.c * l2Chs) "src2RowStride" }
      ]
    | st =>
      let whole2 := hw2.c == 1 && hw2.h == 1 && hw2.w > 1 && outDims.h > 1 && th < outDims.h
      let row2 := 2 * (if whole2 then alignUp hw2.w 8 else min (alignUp hw2.w 8) (max stripW 32))
      let u2 := hw2.c * row2
      let (slot, off, dmaRow, dmaPlane) := match st with
        | .surface _ _ s bOff win _ =>
          let (wOff, rSt, pSt, _) := surfaceDmaWindow win hw2.c hw2.h hw2.w
          (s, bOff + wOff, rSt, pSt)
        | .dramScratch o _ =>
          let rSt := alignUp (2 * hw2.w) dmaLineBytes
          (barSlotScratch, o, rSt, hw2.h * rSt)
        | .l2 _ _ => (0, 0, 0, 0)
      plain := plain ++ #[
        { byteAddr := regSrc2Base, value := ← natToU32Checked pp.src2StageBase "src2Base" },
        { byteAddr := regSrc2ChanStride, value := ← natToU32Checked row2 "src2ChanStride" },
        { byteAddr := regSrc2RowStride, value := ← natToU32Checked u2 "src2RowStride" },
        { byteAddr := regSrc2RowBytesA, value := ← natToU32Checked u2 "src2RowBytesA" },
        { byteAddr := regSrc2RowBytesB, value := ← natToU32Checked u2 "src2RowBytesB" },
        { byteAddr := regSrc2DmaCfg, value := src1DmaCfgValue },
        { byteAddr := regSrc2DmaRowStride, value := ← natToU32Checked dmaRow "src2DmaRowStride" },
        { byteAddr := regSrc2DmaPlaneStride, value := ← natToU32Checked dmaPlane "src2DmaPlaneStride" },
        { byteAddr := regSrc2Fmt, value := srcFmtWordP1 }
      ]
      let wSrc2Dep ← encodeSrcDepMulti "src2Dep" taskId sync.src2DepKind sync.src2DepDist
      barRecs := barRecs ++ #[
        { baseByteAddr := regSrc2BaseLo, barSlot := slot, values := #[← natToU32Checked off "src2BaseLo", 0] },
        { baseByteAddr := regSrc2Dep, barSlot := barSlotConfigZero, values := #[wSrc2Dep] }
      ]

    match pt.resStorage with
    | .l2 base _ =>
      let l2Chs := 2 * alignUp outDims.w 8
      plain := plain ++ #[
        { byteAddr := regResBase, value := ← natToU32Checked base "resBase" },
        { byteAddr := regResChanStride, value := ← natToU32Checked l2Chs "resChanStride" },
        { byteAddr := regResRowStride, value := ← natToU32Checked (outDims.c * l2Chs) "resRowStride" }
      ]
    | st =>
      let (slot, off, depCode, dmaRow, dmaPlane) := match st with
        | .surface _ _ s bOff win _ =>
          let (wOff, rSt, pSt, _) := surfaceDmaWindow win outDims.c outDims.h outDims.w
          (s, bOff + wOff, resDepCode target, rSt, pSt)
        | .dramScratch o _ =>
          let rSt := alignUp (2 * outDims.w) dmaLineBytes
          (barSlotScratch, o, resDepCode target ||| resDepScratchFlags, rSt, outDims.h * rSt)
        | .l2 _ _ => (0, 0, resDepCode target, 0, 0)
      plain := plain ++ #[
        { byteAddr := regResBase, value := ← natToU32Checked pp.resStageBase "resBase" },
        { byteAddr := regResDmaRowStride, value := ← natToU32Checked dmaRow "resDmaRowStride" },
        { byteAddr := regResDmaPlaneStride, value := ← natToU32Checked dmaPlane "resDmaPlaneStride" },
        { byteAddr := regResFmt, value := resFmtWordP1 }
      ]
      if !pp.direct then
        let drow := 2 * min (alignUp outDims.w 8) (max stripW 32)
        let du := outDims.c * drow
        plain := plain ++ #[
          { byteAddr := regResChanStride, value := ← natToU32Checked drow "resChanStride" },
          { byteAddr := regResRowStride, value := ← natToU32Checked du "resRowStride" },
          { byteAddr := regResRowBytesA, value := ← natToU32Checked du "resRowBytesA" },
          { byteAddr := regResRowBytesB, value := ← natToU32Checked du "resRowBytesB" }
        ]
      let wResDep ← encodeWord resDepLayout #[("code", depCode), ("task", taskIdU32), ("direct", if pp.direct then 1 else 0)]
      barRecs := barRecs ++ #[
        { baseByteAddr := regResBaseLo, barSlot := slot, values := #[← natToU32Checked off "resBaseLo", 0] },
        { baseByteAddr := regResDep, barSlot := barSlotConfigZero, values := #[wResDep] }
      ]

    if isLast then
      plain := plain.push { byteAddr := regKernelStage, value := kernelStageNoKernValue }
      plain := plain ++ (← outputStageWrites taskId)
    pure {
      taskId := taskId, logNE := 0, eneOverride := some 0,
      isFirst := isFirst, isLast := isLast, liveL2Mask := pt.task.liveL2Mask,
      plainWrites := plain, barWrites := barRecs, kernelRelocs := #[]
    }

  | .pool op _ _ _ _ _ excl, .pePool pp gh gw =>
    let dims := shape4DToDims buffers[pt.task.inputs[0]!]!.shape
    let srcLoc ← resolvedToPoolPortLoc s!"pePool '{pt.task.name}' src1" pt.src1Storage
    let resLoc ← resolvedToPoolPortLoc s!"pePool '{pt.task.name}' res" pt.resStorage
    buildPePoolTaskMulti target op excl dims gh gw pp taskId totalTasks
      srcLoc resLoc sync pt.task.liveL2Mask

  | .reduce op axes eps, .peReduce pr =>
    let dims := shape4DToDims buffers[pt.task.inputs[0]!]!.shape
    let srcLoc ← resolvedToPoolPortLoc s!"peReduce '{pt.task.name}' src1" pt.src1Storage
    let resLoc ← resolvedToPoolPortLoc s!"peReduce '{pt.task.name}' res" pt.resStorage
    buildPeReduceTaskMulti target op axes eps dims pr taskId totalTasks
      srcLoc resLoc sync pt.task.liveL2Mask

  | _, _ => throw s!"internal error: pe op/plan mismatch on '{pt.task.name}'"

-- ============================================================================
-- Top-Level L4 Entry Point (`MachineProgram → CompiledMachineProgram`)
-- ============================================================================

/-- Run unified S1–S5 synchronization, emit `ProgramSpec` via one clean emitter per `AneOp`
    constructor, invoke `Vane.Program.emitProgram`, and independently verify the resulting HWX
    with `Vane.Verify.verifyHwxTasksSummary`. -/
def syncAndEmit (mp : MachineProgram) : Except String CompiledMachineProgram := do
  let n := mp.tasks.size
  let allSurfs := mp.allSurfaces
  let kernelBarSlot : UInt8 := barSlotFirstSurface + allSurfs.size.toUInt8

  let (preTasks, kernBuf) ← buildPreparedEmitTasks mp
  let syncs := computeSynchronization preTasks mp.donations

  let mut taskDescs : Array TaskDesc := Array.mkEmpty n
  let mut syncedTasks : Array MachineTask := Array.mkEmpty n
  for i in [:n] do
    let pt := preTasks[i]!
    let sync := syncs[i]!
    let kernKeepLive : Bool := match pt.kernStorage? with
      | some (.dramScratch off bytes) => Id.run do
          for j in [i + 1 : n] do
            if preTasks[j]!.dramAccesses.any (fun a => a.resource == .scratch && a.port == .kernel && intervalsOverlap off (off + bytes) a.lo a.hi) then
              return true
            if preTasks[j]!.dramAccesses.any (fun a => a.resource == .scratch && a.isWrite && intervalsOverlap off (off + bytes) a.lo a.hi) then
              return false
          return false
      | _ => false

    let td ← match pt.task.op, pt.task.tilePlan with
      | .neConv _ _ _ _ _ _ _ _ _ _ act _ _ _, .neConv cp _ _ =>
        emitNeConvTask mp.target n kernelBarSlot pt sync act cp
      | .neAct kind _ _, _ =>
        emitNeActTask mp.target n kernelBarSlot mp.buffers pt sync kind
      | .neMatmul heads m k nCols _ _ scaleBits m0 totalM, .neMatmul mmPlan =>
        emitNeMatmulTask mp.target n pt sync heads m k nCols scaleBits m0 totalM mmPlan kernKeepLive
      | .dmaView _ _, .dmaView vp cp _ _ =>
        emitDmaViewTask mp.target n pt sync vp cp
      | .pe peOp _, _ =>
        emitPeTask mp.target n mp.buffers pt sync peOp
      | _, _ =>
        throw s!"internal error: op/tilePlan mismatch on task '{pt.task.name}'"

    taskDescs := taskDescs.push td
    syncedTasks := syncedTasks.push {
      pt.task with
      sync := sync
      extProducerTask := pt.extProducerTask?
    }

  let scratchSpec? : Option ScratchSpec ←
    if mp.scratchBytes == 0 then pure none
    else
      let szU64 ← natToU64Checked mp.scratchBytes "scratch size"
      let usU32 ← natToU32Checked mp.scratchBytes "scratch usable"
      pure (some { size := szU64, usable := usU32 })

  let kernelOpt? : Option ByteArray := if kernBuf.isEmpty then none else some kernBuf

  let emitted ← emitProgram {
    target := mp.target
    surfaces := allSurfs
    tasks := taskDescs
    kernel := kernelOpt?
    scratch := scratchSpec?
    constants := zeroBytes textConstBytes
    ident := some "vane"
  }

  let summary ← verifyHwxTasksSummary emitted.hwx emitted.info (requireKnownTask := true)
  if summary.total != n then
    throw s!"internal error: verifier checked {summary.total} tasks != {n}"

  let finalMachine : MachineProgram := {
    mp with
    tasks := syncedTasks
    kernelBytes := kernBuf.size
  }

  pure {
    machine := finalMachine
    hwx := emitted.hwx
    info := emitted.info
    sections := emitted.sections
    verifySummary := summary
  }

-- ============================================================================
-- Unified End-to-End L2 -> L3 -> L4 Backend Pipeline Driver
-- ============================================================================

private def isTextSizeExceededError (msg : String) : Bool :=
  (msg.splitOn "task stream size").length > 1

private def evalInteriorStripPlan
    (prog : AneProgram) (hFull : Nat) (radii : Array Nat) (h : Nat) : Option (Nat × Nat) :=
  if h == 0 || h > hFull then none
  else
    let y0 := (hFull - h) / 2
    let y1 := y0 + h
    match Vane.Backend.Cascade.buildSingleInteriorStripAne prog hFull radii y0 y1 with
    | .error _ => none
    | .ok stripProg =>
      match Vane.Backend.MemoryPlan.planMemory stripProg with
      | .error _ => none
      | .ok mp =>
        some (mp.witness.scratchPlaced.length, mp.scratchBytes)

private def chooseAutoStripHeightExact
    (prog : AneProgram) (hFull : Nat) (radii : Array Nat) (minH : Nat) : Option Nat := Id.run do
  if minH == 0 || minH >= hFull then
    return none
  if evalInteriorStripPlan prog hFull radii minH == some (0, 0) then
    let mut lo := minH
    let mut hi := hFull - 1
    let mut best := minH
    for _ in [:20] do
      if lo > hi then break
      let mid := (lo + hi) / 2
      if evalInteriorStripPlan prog hFull radii mid == some (0, 0) then
        best := mid
        lo := mid + 1
      else
        if mid == 0 then break
        hi := mid - 1
    return some best

  let mut best? : Option (Nat × Nat × Nat) := none
  let mut worseStreak : Nat := 0
  for h in [minH : hFull] do
    match evalInteriorStripPlan prog hFull radii h with
    | none =>
      worseStreak := worseStreak + 1
      if best?.isSome && worseStreak >= 4 then break
    | some (edges, bytes) =>
      let isBetter := match best? with
        | none => true
        | some (bh, bEdges, bBytes) =>
          if edges != bEdges then edges < bEdges
          else if bytes != bBytes then bytes < bBytes
          else h > bh
      if isBetter then
        best? := some (h, edges, bytes)
        worseStreak := 0
      else
        worseStreak := worseStreak + 1
        if worseStreak >= 4 then break
  return best?.map fun (h, _, _) => h

/-- Run the complete L2 → L3 → L4 backend pipeline:
    `channelSplitPass` → `cascadePass` (rolling L2 H-strip / W-strip cascades) →
    `planMemory` (L2 live-interval packing + selective BAR 3 DRAM spilling + zero-copy subviews) →
    `syncAndEmit` (unified S1–S5 fixpoint sync + HWX emission + independent `Vane.Verify`). -/
def compileAneProgram
    (aneIn : AneProgram)
    (stripMode : StripMode := .off)
    (allowSequenceWStrip : Bool := true) : Except String (AneProgram × CompiledMachineProgram) := do
  let aneSplit ← Vane.Backend.ChannelSplit.channelSplitPass aneIn

  match stripMode with
  | .off =>
    let mp ← Vane.Backend.MemoryPlan.planMemory aneSplit
    let cm ← syncAndEmit mp
    pure (aneSplit, cm)

  | .fixed hOut =>
    let aneCasc ← Vane.Backend.Cascade.cascadePass aneSplit {
      stripMode := .fixed hOut
      allowSequenceWStrip := false
    }
    let mp ← Vane.Backend.MemoryPlan.planMemory aneCasc
    let cm ← syncAndEmit mp
    pure (aneCasc, cm)

  | .auto =>
    -- 1. If unstripped already fits 100% in L2 (0 B DRAM scratch), keep it unstripped.
    let unstrippedMpRes := Vane.Backend.MemoryPlan.planMemory aneSplit
    if let .ok mp := unstrippedMpRes then
      if mp.scratchBytes == 0 then
        let cm ← syncAndEmit mp
        return (aneSplit, cm)

    -- 2. Try automatic spatial H-strip rolling L2 cascade if eligible.
    match Vane.Backend.Cascade.checkHStripEligible aneSplit with
    | .ok (hFull, _, radii, maxTasks) =>
      let m := aneSplit.graph.nodes.size
      let maxStrips := maxTasks / m
      if maxStrips > 1 then
        let minH := ceilDiv hFull maxStrips
        if let some startH := chooseAutoStripHeightExact aneSplit hFull radii minH then
          let mut curH := startH
          let mut result? : Option (AneProgram × CompiledMachineProgram) := none
          for _ in [:256] do
            if result?.isSome || curH >= hFull then break
            let attempt : Except String (AneProgram × CompiledMachineProgram) := do
              let aneCasc ← Vane.Backend.Cascade.unrollHStripCascade aneSplit hFull radii maxTasks curH
              let mp ← Vane.Backend.MemoryPlan.planMemory aneCasc
              let cm ← syncAndEmit mp
              pure (aneCasc, cm)
            match attempt with
            | .ok res => result? := some res
            | .error msg =>
              if isTextSizeExceededError msg then
                let curS := ceilDiv hFull curH
                if curS <= 2 then break
                curH := ceilDiv hFull (curS - 1)
              else
                throw msg
          if let some res := result? then
            return res
    | .error _ =>
      -- 3. Try W/sequence stripping for `H = 1, W = seq_len` blocks when enabled.
      if allowSequenceWStrip then
        if let .ok wCasc := Vane.Backend.Cascade.cascadePass aneSplit {
            stripMode := .auto, allowSequenceWStrip := true
          } then
          if wCasc.stripCount > 1 then
            if let .ok mp := Vane.Backend.MemoryPlan.planMemory wCasc then
              if let .ok cm := syncAndEmit mp then
                return (wCasc, cm)

    -- 4. Fallback: emit unstripped with selective L2 <-> DRAM BAR 3 spilling!
    let mp ← unstrippedMpRes
    let cm ← syncAndEmit mp
    pure (aneSplit, cm)

end Vane.Backend.SyncAndEmit
