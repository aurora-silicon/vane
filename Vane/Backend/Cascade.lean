import Vane.Binary
import Vane.TaskStream
import Vane.Alloc
import Vane.Tiling
import Vane.View
import Vane.Ane.Types

namespace Vane.Backend.Cascade

open Vane.Binary
open Vane.TaskStream
open Vane.Alloc
open Vane.Tiling
open Vane.View
open Vane.IR
open Vane.Ane

-- ============================================================================
-- L2 Pass: Vela-Style Rolling L2 Cascades in SSA (`SSAGraph AneOp`)
-- ============================================================================
--
-- Supports three pure SSA strip-cascade transformations before bufferization:
-- 1. Direct L2-only H-strip rolling cascade (`stripAxis = 2`):
--    Unrolls eligible spatial subgraphs (chains, input-skip DAGs, and multi-branch
--    DAGs with matching halo intervals, including multi-output graphs) into
--    interleaved strip-level `AneOp` tasks in strict topological order so L3
--    `Vane.Alloc.allocate` folds strip buffers into rolling/ping-pong physical L2
--    slots with 0 B DRAM scratch and shared `__KERN_0` (`weightGroupId`).
-- 2. Rolling halo-stitch H-strip cascade (`rollingHalo = true`):
--    Zero recomputed MACs on deep 3x3 stacks: each strip `s` of layer `L` computes
--    disjoint rows `[y0, y1)` exactly once, and micro-DMA `dmaView` halo stitching
--    concatenates strip `s-1`'s bottom halo row + strip `s` + strip `s+1`'s top row
--    only before 3x3 consumers (zero extra DMA on 1x1/PE/LUT where halo = 0).
-- 3. W/sequence stripping (`stripAxis = 3`) for `H = 1, W = seq_len` blocks:
--    Slices 64B-aligned W strips (`stripW % 32 == 0`) via `dmaView` (Surface->L2
--    input slice and L2->Surface output stitch), keeping all compute layers 100%
--    L2-resident (`scratchBytes == 0`).

/-- Configuration for the L2 rolling cascade pass. -/
structure CascadeConfig where
  stripMode : StripMode := .off
  /-- Enable rolling micro-DMA halo stitching (zero recomputed MACs) when direct
      halo overlap cannot fit or encounters an internal skip-halo mismatch. -/
  allowRollingHaloStitch : Bool := false
  /-- Enable W/sequence stripping for `H = 1, W = seq_len` blocks under `.auto`. -/
  allowSequenceWStrip : Bool := true
  deriving Repr, BEq, Inhabited

/-- Summary of a selected cascade schedule. -/
structure CascadePlan where
  axis : Nat            -- 2 = H, 3 = W
  stripSize : Nat
  numStrips : Nat
  fullExtent : Nat
  rollingHalo : Bool := false
  deriving Repr, BEq, Inhabited

-- ============================================================================
-- Pure Strip-Window & Halo Geometry Primitives (Shared with Formal Proofs)
-- ============================================================================

/-- Disjoint output row (or column) band `[lo, hi)` for strip `s` of extent `full` at strip size `step`. -/
def stripOutBand (full step s : Nat) : Nat × Nat :=
  let lo := s * step
  (lo, min full (lo + step))

/-- Clamped input halo interval `[inLo, inHi)` required to produce output rows `[lo, hi)` with radius `rad`. -/
def haloInBand (hFull rad lo hi : Nat) : Nat × Nat :=
  let inLo := if lo >= rad then lo - rad else 0
  let inHi := min hFull (hi + rad)
  (inLo, inHi)

/-- Top boundary zero-padding (`1` at top edge `outLo = 0`, `0` for interior/bottom strips). -/
def stripPadTop (outLo : Nat) : Nat :=
  if outLo == 0 then 1 else 0

/-- Bottom boundary zero-padding (`1` at bottom edge `outHi = hFull`, `0` for top/interior strips). -/
def stripPadBot (outHi hFull : Nat) : Nat :=
  if outHi == hFull then 1 else 0

/-- Compute cumulative wavefront lag per layer for rolling halo cascades:
    layer `0` has lag `0`, and each subsequent layer `l > 0` advances lag by `+1` iff `radii[l] > 0`. -/
def rollingHaloLayerLagsAux (radii : Array Nat) (l accLag : Nat) (acc : Array Nat) : Array Nat :=
  if h : l < radii.size then
    let nextLag := if l > 0 && radii[l] > 0 then accLag + 1 else accLag
    rollingHaloLayerLagsAux radii (l + 1) nextLag (acc.push nextLag)
  else
    acc
termination_by radii.size - l

/-- Cumulative wavefront lag array for `unrollRollingHaloCascade`. -/
def rollingHaloLayerLags (radii : Array Nat) : Array Nat :=
  rollingHaloLayerLagsAux radii 0 0 (Array.mkEmpty radii.size)

/-- Wavefront step at which layer `l` executes strip `s` under lag schedule `lags`. -/
def rollingHaloWave (lags : Array Nat) (l s : Nat) : Nat :=
  s + lags[l]!

-- ============================================================================
-- Geometry & Eligibility Helpers
-- ============================================================================

private def lookupShape (shMap : Array (ValueId × Shape4D)) (v : ValueId) (ctx : String) :
    Except String Shape4D := do
  match shMap.find? fun p => p.1 == v with
  | some (_, s) => pure s
  | none => throw s!"Cascade ({ctx}): missing shape for %{v}"

private def buildShapeMap (g : SSAGraph AneOp) : Except String (Array (ValueId × Shape4D)) := do
  let mut shMap : Array (ValueId × Shape4D) := #[]
  for (vid, nm, ty) in g.inputs do
    shMap := shMap.push (vid, ← tensorTypeToShape4D ty s!"input '{nm}'")
  for nd in g.nodes do
    for (ov, oty) in nd.outputs do
      shMap := shMap.push (ov, ← tensorTypeToShape4D oty s!"node '{nd.name}'")
  pure shMap

/-- Map each SSA value to either `.inl inputIdx` or `.inr nodeIdx`. -/
private def buildProducerMap (g : SSAGraph AneOp) : Array (ValueId × Sum Nat Nat) := Id.run do
  let mut m : Array (ValueId × Sum Nat Nat) := #[]
  for i in [:g.inputs.size] do
    m := m.push (g.inputs[i]!.1, .inl i)
  for ni in [:g.nodes.size] do
    for (ov, _) in g.nodes[ni]!.outputs do
      m := m.push (ov, .inr ni)
  return m

private def lookupProd (pm : Array (ValueId × Sum Nat Nat)) (v : ValueId) : Option (Sum Nat Nat) :=
  (pm.find? fun p => p.1 == v).map Prod.snd

/-- Reverse halo propagation across the DAG for an output row band `[y0, y1)` on ALL graph outputs.
    Returns `(outIntervals, inIntervals)` per node `0..M-1`. -/
def propagateStripIntervals
    (g : SSAGraph AneOp) (radii : Array Nat) (hFull y0 y1 : Nat) :
    Except String (Array (Nat × Nat) × Array (Nat × Nat)) := do
  let m := g.nodes.size
  if y0 >= y1 || y1 > hFull then
    throw s!"invalid strip row band [{y0}, {y1}) for H={hFull}"
  let pm := buildProducerMap g
  let mut reqOut : Array (Option (Nat × Nat)) := Array.replicate m none
  let mut reqIn : Array (Nat × Nat) := Array.replicate m (0, 0)

  for ov in g.outputs do
    match lookupProd pm ov with
    | some (.inr ni) =>
      reqOut := reqOut.set! ni (some (y0, y1))
    | _ =>
      throw "spatial strip fusion: graph output must be produced by a node"

  for step in [:m] do
    let u := m - 1 - step
    let some (lo, hi) := reqOut[u]!
      | throw s!"spatial strip fusion: node '{g.nodes[u]!.name}' is unreachable from output"
    let rad := radii[u]!
    let (inLo, inHi) := haloInBand hFull rad lo hi
    if inLo >= inHi then
      throw s!"spatial strip fusion: empty input interval [{inLo}, {inHi}) on node '{g.nodes[u]!.name}'"
    reqIn := reqIn.set! u (inLo, inHi)
    for iv in g.nodes[u]!.inputs do
      match lookupProd pm iv with
      | some (.inl _) => pure ()
      | some (.inr prodIdx) =>
        match reqOut[prodIdx]! with
        | none =>
          reqOut := reqOut.set! prodIdx (some (inLo, inHi))
        | some (prevLo, prevHi) =>
          if prevLo != inLo || prevHi != inHi then
            throw s!"spatial strip fusion ineligible: skip-halo interval mismatch on internal node '{g.nodes[prodIdx]!.name}' (consumer '{g.nodes[u]!.name}' requires [{inLo}, {inHi}), another consumer requires [{prevLo}, {prevHi}))"
      | none =>
        throw s!"spatial strip fusion: unknown input %{iv} on node '{g.nodes[u]!.name}'"

  let mut outIntervals : Array (Nat × Nat) := Array.mkEmpty m
  for u in [:m] do
    match reqOut[u]! with
    | some iv => outIntervals := outIntervals.push iv
    | none => throw s!"spatial strip fusion: node '{g.nodes[u]!.name}' missing interval"
  pure (outIntervals, reqIn)

/-- Check whether an `AneProgram` is eligible for spatial H-strip cascading.
    Returns `(hFull, wFull, radii, maxTasks)`. -/
def checkHStripEligible (prog : AneProgram) : Except String (Nat × Nat × Array Nat × Nat) := do
  let g := prog.graph
  let m := g.nodes.size
  if m == 0 then
    throw "spatial strip fusion ineligible: graph has 0 nodes"
  -- Reject duplicate graph output ValueIds so each node maps cleanly to at most one output slot
  for i in [:g.outputs.size] do
    for j in [i + 1 : g.outputs.size] do
      if g.outputs[i]! == g.outputs[j]! then
        throw "spatial strip fusion ineligible: duplicate graph output ValueIds"
  let shMap ← buildShapeMap g
  let in0Sh ← lookupShape shMap g.inputs[0]!.1 "in0"
  let hFull := in0Sh.h
  let wFull := in0Sh.w
  if hFull <= 1 then
    throw s!"spatial strip fusion ineligible: input height H={hFull} must be > 1"
  for (iv, nm, _) in g.inputs do
    let s ← lookupShape shMap iv nm
    if s.h != hFull || s.w != wFull then
      throw s!"spatial strip fusion ineligible: input '{nm}' shape ({s.h}x{s.w}) != ({hFull}x{wFull})"

  let mut radii : Array Nat := Array.mkEmpty m
  for ni in [:m] do
    let nd := g.nodes[ni]!
    let outSh ← lookupShape shMap nd.outputs[0]!.1 nd.name
    let rad ← match nd.op with
      | .neConv kh kw sh sw pad _ _ _ _ _ _ _ _ _ =>
        if kh == 1 && kw == 1 then
          if outSh.w != wFull then
            throw s!"spatial strip fusion ineligible: node '{nd.name}' width {outSh.w} != {wFull}"
          pure 0
        else if kh == 3 && kw == 3 && sh == 1 && sw == 1 && pad == .same then
          if outSh.w != wFull then
            throw s!"spatial strip fusion ineligible: node '{nd.name}' width {outSh.w} != {wFull}"
          pure 1
        else
          throw s!"spatial strip fusion ineligible: node '{nd.name}' conv3x3 requires stride=1 and pad=same (got stride={sh}, pad={pad.name})"
      | .pe (.unary _) _ =>
        if outSh.w != wFull then
          throw s!"spatial strip fusion ineligible: node '{nd.name}' width {outSh.w} != {wFull}"
        pure 0
      | .pe (.binary _ _) _ =>
        let s1 ← lookupShape shMap nd.inputs[0]! nd.name
        let s2 ← lookupShape shMap nd.inputs[1]! nd.name
        if s1.h != hFull || s1.w != wFull || s2.h != hFull || s2.w != wFull ||
           outSh.h != hFull || outSh.w != wFull then
          throw s!"spatial strip fusion ineligible: node '{nd.name}' pe2 broadcast inputs are not supported"
        pure 0
      | .neAct (.act _) _ _ =>
        if outSh.w != wFull then
          throw s!"spatial strip fusion ineligible: node '{nd.name}' width {outSh.w} != {wFull}"
        pure 0
      | .pe (.pool ..) _ | .neAct (.nePool ..) _ _ =>
        throw s!"spatial strip fusion ineligible: node '{nd.name}' pool operation is not supported by spatial H-strip fusion"
      | .pe (.reduce ..) _ | .neAct (.channelReduce ..) _ _ =>
        throw s!"spatial strip fusion ineligible: node '{nd.name}' reduce operation is not supported by spatial H-strip fusion"
      | .dmaView _ _ =>
        throw s!"spatial strip fusion ineligible: node '{nd.name}' view operation is not supported by spatial H-strip fusion"
      | .neMatmul .. =>
        throw s!"spatial strip fusion ineligible: node '{nd.name}' matmul operation is not supported by spatial H-strip fusion"
    radii := radii.push rad

  -- Probe skip-halo exact interval match on top and interior 1-row bands
  let _ ← propagateStripIntervals g radii hFull 0 1
  let mid := hFull / 2
  let _ ← propagateStripIntervals g radii hFull mid (mid + 1)

  let anyKern := g.nodes.any fun nd => nd.op.usesKernel
  let maxTasks := if anyKern then 255 else 256
  pure (hFull, wFull, radii, maxTasks)

/-- Estimate peak live L2 bytes across all tasks of a single strip (or unstripped graph)
    using `Vane.Alloc.allocate` over exact intermediate lifetimes plus conservative staging headroom. -/
def estimateGraphL2Bytes (g : SSAGraph AneOp) (shMap : Array (ValueId × Shape4D))
    (nodeHOverride? : Option (Array Nat) := none) : Nat := Id.run do
  let m := g.nodes.size
  if m == 0 then return 0
  let pm := buildProducerMap g
  let mut lastUse : Array Nat := Array.replicate m 0
  for ni in [:m] do
    for iv in g.nodes[ni]!.inputs do
      if let some (.inr p) := lookupProd pm iv then
        lastUse := lastUse.set! p (max lastUse[p]! ni)
  let mut isOut : Array Bool := Array.replicate m false
  for ov in g.outputs do
    if let some (.inr p) := lookupProd pm ov then
      isOut := isOut.set! p true

  let shapeAt := fun (ni : Nat) =>
    let baseSh := (shMap.find? fun p => p.1 == g.nodes[ni]!.outputs[0]!.1).map Prod.snd |>.getD { n := 1, c := 1, h := 1, w := 1 }
    match nodeHOverride? with
    | some hs => { baseSh with h := hs.getD ni baseSh.h }
    | none => baseSh

  let mut liveBufs : List LiveBuf := []
  for ni in [:m] do
    if !isOut[ni]! && lastUse[ni]! > ni then
      let sh := shapeAt ni
      liveBufs := {
        id := ni
        bytes := l2ResidentBytes4D sh
        first := ni
        last := lastUse[ni]!
        priority := lastUse[ni]! - ni
      } :: liveBufs
  let placed := allocate true liveBufs
  let residentTop := topOf placed
  -- Also estimate max staging for first/last tasks (surface DMA staging)
  let mut peak := residentTop
  for ni in [:m] do
    let liveAtT := (liveAt placed ni).foldl (fun acc r => max acc r.hi) 0
    let nd := g.nodes[ni]!
    let outSh := shapeAt ni
    -- Conservative staging estimate for surface-connected tasks (tileHeight >= 1, up to 3 rows for 3x3)
    let mut stageExtra : Nat := 0
    for iv in nd.inputs do
      if let some (.inl _) := lookupProd pm iv then
        let inSh := (shMap.find? fun p => p.1 == iv).map Prod.snd |>.getD outSh
        let rows := match nd.op with
          | .neConv 3 3 .. => min 3 (match nodeHOverride? with | some hs => hs.getD ni inSh.h | none => inSh.h)
          | _ => 1
        let chs := 2 * min (alignUp inSh.w 8) 128
        stageExtra := stageExtra + alignUp (rows * inSh.c * chs) 64
    if isOut[ni]! then
      stageExtra := stageExtra + 65536 -- conservative direct/staged output headroom
    peak := max peak (liveAtT + stageExtra)
  return peak

/-- Unroll an `AneProgram` into an interleaved strip-level `SSAGraph AneOp` at strip height `hOut`
    (direct L2-only H-strip rolling cascade). Every strip `s` emits layers `0..M-1` in strict
    topological order, preserving `weightGroupId` for zero kernel duplication! -/
def unrollHStripCascade (prog : AneProgram) (hFull : Nat) (radii : Array Nat)
    (maxTasks hOut : Nat) : Except String AneProgram := do
  if hOut == 0 then
    throw "strip height must be >= 1"
  let g := prog.graph
  let m := g.nodes.size
  let sCount := ceilDiv hFull hOut
  let pm := buildProducerMap g
  -- Account for per-strip `pack_out` tasks that `MemoryPlan` will insert for non-direct multi-outputs
  let extraPackPerStrip : Nat :=
    if g.outputs.size < 2 then 0
    else (Array.range g.outputs.size).foldl (fun (acc : Nat) oi =>
      let ov := g.outputs[oi]!
      let isDirect := match lookupProd pm ov with
        | some (.inr ni) =>
          let hasDown := g.nodes.any fun nd => nd.inputs.contains ov
          match g.nodes[ni]!.op with
          | .neConv 1 1 .. => !hasDown
          | .dmaView .. => !hasDown
          | _ => false
        | _ => false
      if isDirect then acc else acc + 1) 0
  let totalTasks := sCount * (m + extraPackPerStrip)
  if totalTasks > maxTasks then
    throw s!"strip height {hOut} expands {m} nodes (+{extraPackPerStrip} pack_out/strip) across {sCount} strips to {totalTasks} tasks, exceeding limit {maxTasks}"

  let shMap ← buildShapeMap g

  -- Map each node index to its graph output index (if it produces a graph output)
  let mut nodeOutIdx : Array (Option Nat) := Array.replicate m none
  for oi in [:g.outputs.size] do
    if let some (.inr ni) := lookupProd pm g.outputs[oi]! then
      nodeOutIdx := nodeOutIdx.set! ni (some oi)

  let mut nextVid : ValueId := g.inputs.foldl (fun acc e => max acc (e.1 + 1)) 0
  let mut newNodes : Array (IRNode AneOp) := Array.mkEmpty totalTasks
  let mut newMeta : Array AneNodeMeta := Array.mkEmpty totalTasks
  -- Keep the last strip's output ValueId per graph output (and mark ALL strips' output nodes via stripOutputIdx?)
  let mut finalOutVids : Array ValueId := Array.replicate g.outputs.size 0

  for s in [:sCount] do
    let (y0, y1) := stripOutBand hFull hOut s
    let suffix := if sCount > 1 then s!"_s{s}" else ""
    let (outIntervals, inIntervals) ← propagateStripIntervals g radii hFull y0 y1
    let mut stripVidMap : Array ValueId := Array.replicate m 0

    for l in [:m] do
      let nd := g.nodes[l]!
      let origMeta := prog.getNodeMeta l
      let (outLo, outHi) := outIntervals[l]!
      let (inLo, _inHi) := inIntervals[l]!
      let outH := outHi - outLo
      let origOutSh ← lookupShape shMap nd.outputs[0]!.1 nd.name
      let stripOutSh : Shape4D := { origOutSh with h := outH }

      let newOp : AneOp := match nd.op with
        | .neConv kh kw sh sw _ mode cin cout w b act til wgId lbl =>
          if kh == 1 && kw == 1 then
            .neConv 1 1 sh sw .same mode cin cout w b act til wgId lbl
          else
            let pt := stripPadTop outLo
            let pb := stripPadBot outHi hFull
            .neConv 3 3 sh sw (.strip pt pb) mode cin cout w b act til wgId lbl
        | other => other

      let newInps : Array ValueId := nd.inputs.map fun iv =>
        match lookupProd pm iv with
        | some (.inl _) => iv
        | some (.inr pIdx) => stripVidMap[pIdx]!
        | none => iv

      let isSrcInput := fun (slot : Nat) =>
        match nd.inputs[slot]? with
        | some iv => match lookupProd pm iv with | some (.inl _) => true | _ => false
        | none => false

      let src1Win : AneSurfaceWindow :=
        if isSrcInput 0 then { rowOffset := inLo, fullH := hFull } else {}
      let src2Win : AneSurfaceWindow :=
        if isSrcInput 1 then { rowOffset := inLo, fullH := hFull } else {}
      let resWin : AneSurfaceWindow :=
        if nodeOutIdx[l]!.isSome then { rowOffset := outLo, fullH := hFull } else {}

      let newVid := nextVid; nextVid := nextVid + 1
      stripVidMap := stripVidMap.set! l newVid

      let nodeName := if suffix.isEmpty then nd.name else s!"{nd.name}{suffix}"
      newNodes := newNodes.push {
        name := nodeName
        op := newOp
        inputs := newInps
        outputs := #[(newVid, shape4DToTensorType stripOutSh)]
      }
      newMeta := newMeta.push {
        logicalId := if origMeta.logicalId.isEmpty then nd.name else origMeta.logicalId
        outPlacement := origMeta.outPlacement
        src1Win := src1Win
        src2Win := src2Win
        resWin := resWin
        stripOutputIdx? := nodeOutIdx[l]!
      }
      if let some oi := nodeOutIdx[l]! then
        finalOutVids := finalOutVids.set! oi newVid

  pure {
    prog with
    graph := { g with nodes := newNodes, outputs := finalOutVids }
    nodeMeta := newMeta
    stripCount := sCount
    stripHeight := min hOut hFull
    stripAxis := 2
  }

/-- Build a single interior strip `AneProgram` covering output rows `[y0, y1)` of a `hFull`-row
    frame. Used by auto-strip selection to probe exact `planMemory` L2/DRAM behavior on one
    representative interior strip in O(M) time. -/
def buildSingleInteriorStripAne (prog : AneProgram) (hFull : Nat) (radii : Array Nat)
    (y0 y1 : Nat) : Except String AneProgram := do
  let g := prog.graph
  let m := g.nodes.size
  let shMap ← buildShapeMap g
  let pm := buildProducerMap g
  let (outIntervals, inIntervals) ← propagateStripIntervals g radii hFull y0 y1
  let mut nodeOutIdx : Array (Option Nat) := Array.replicate m none
  for oi in [:g.outputs.size] do
    if let some (.inr ni) := lookupProd pm g.outputs[oi]! then
      nodeOutIdx := nodeOutIdx.set! ni (some oi)

  let mut nextVid : ValueId := g.inputs.foldl (fun acc e => max acc (e.1 + 1)) 0
  let mut newNodes : Array (IRNode AneOp) := Array.mkEmpty m
  let mut newMeta : Array AneNodeMeta := Array.mkEmpty m
  let mut stripVidMap : Array ValueId := Array.replicate m 0
  let mut finalOutVids : Array ValueId := Array.replicate g.outputs.size 0

  for l in [:m] do
    let nd := g.nodes[l]!
    let origMeta := prog.getNodeMeta l
    let (outLo, outHi) := outIntervals[l]!
    let (inLo, _) := inIntervals[l]!
    let outH := outHi - outLo
    let origOutSh ← lookupShape shMap nd.outputs[0]!.1 nd.name
    let stripOutSh : Shape4D := { origOutSh with h := outH }
    let newOp : AneOp := match nd.op with
      | .neConv kh kw sh sw _ mode cin cout w b act til wgId lbl =>
        if kh == 1 && kw == 1 then
          .neConv 1 1 sh sw .same mode cin cout w b act til wgId lbl
        else
          let pt := if outLo == 0 then 1 else 0
          let pb := if outHi == hFull then 1 else 0
          .neConv 3 3 sh sw (.strip pt pb) mode cin cout w b act til wgId lbl
      | other => other
    let newInps : Array ValueId := nd.inputs.map fun iv =>
      match lookupProd pm iv with
      | some (.inl _) => iv
      | some (.inr pIdx) => stripVidMap[pIdx]!
      | none => iv
    let isSrcInput := fun (slot : Nat) =>
      match nd.inputs[slot]? with
      | some iv => match lookupProd pm iv with | some (.inl _) => true | _ => false
      | none => false
    let src1Win : AneSurfaceWindow := if isSrcInput 0 then { rowOffset := inLo, fullH := hFull } else {}
    let src2Win : AneSurfaceWindow := if isSrcInput 1 then { rowOffset := inLo, fullH := hFull } else {}
    let resWin : AneSurfaceWindow := if nodeOutIdx[l]!.isSome then { rowOffset := outLo, fullH := hFull } else {}
    let nv := nextVid; nextVid := nextVid + 1
    stripVidMap := stripVidMap.set! l nv
    newNodes := newNodes.push {
      name := nd.name, op := newOp, inputs := newInps, outputs := #[(nv, shape4DToTensorType stripOutSh)]
    }
    newMeta := newMeta.push {
      logicalId := if origMeta.logicalId.isEmpty then nd.name else origMeta.logicalId
      outPlacement := origMeta.outPlacement
      src1Win := src1Win, src2Win := src2Win, resWin := resWin
      stripOutputIdx? := nodeOutIdx[l]!
    }
    if let some oi := nodeOutIdx[l]! then
      finalOutVids := finalOutVids.set! oi nv

  pure {
    prog with
    graph := { g with nodes := newNodes, outputs := finalOutVids }
    nodeMeta := newMeta
    stripCount := 1
    stripHeight := y1 - y0
    stripAxis := 2
  }

-- ============================================================================
-- True Zero-Recompute Rolling Halo-Stitch Cascade (Micro-DMA Halo Reuse)
-- ============================================================================

/-- Check whether an `AneProgram` is a linear chain of H-preserving ops (`neConv` 1x1/3x3-s1-same,
    `pe` unary, `neAct`) suitable for zero-recompute rolling halo-stitch cascading. -/
def checkRollingHaloChainEligible (prog : AneProgram) : Except String (Nat × Nat × Array Nat) := do
  let g := prog.graph
  let m := g.nodes.size
  if m < 2 || g.inputs.size != 1 || g.outputs.size != 1 then
    throw "rolling halo cascade requires single-input single-output chain with >= 2 layers"
  let shMap ← buildShapeMap g
  let in0Sh ← lookupShape shMap g.inputs[0]!.1 "in0"
  let hFull := in0Sh.h; let wFull := in0Sh.w
  if hFull < 4 then
    throw s!"rolling halo cascade requires H >= 4, got H={hFull}"
  let mut radii : Array Nat := Array.mkEmpty m
  let mut prevVid := g.inputs[0]!.1
  for ni in [:m] do
    let nd := g.nodes[ni]!
    if nd.inputs.size != 1 || nd.inputs[0]! != prevVid then
      throw s!"rolling halo cascade: node '{nd.name}' breaks linear chain"
    let outSh ← lookupShape shMap nd.outputs[0]!.1 nd.name
    if outSh.h != hFull || outSh.w != wFull then
      throw s!"rolling halo cascade: node '{nd.name}' changes spatial dims"
    let rad ← match nd.op with
      | .neConv 1 1 1 1 .same _ _ _ _ _ _ _ _ _ => pure 0
      | .neConv 3 3 1 1 .same _ _ _ _ _ _ _ _ _ => pure 1
      | .pe (.unary _) _ => pure 0
      | .neAct (.act _) _ _ => pure 0
      | _ => throw s!"rolling halo cascade: unsupported op on node '{nd.name}'"
    radii := radii.push rad
    prevVid := nd.outputs[0]!.1
  if prevVid != g.outputs[0]! then
    throw "rolling halo cascade: final node does not match graph output"
  pure (hFull, wFull, radii)

/-- Unroll a chain with **zero recomputed MACs** on 3x3 layers:
    each strip `s` of layer `L` computes disjoint rows `[y0, y1)` exactly once, and when
    layer `L+1` is a 3x3 conv (`rad = 1`), emits micro-DMA `dmaView` halo stitch tasks
    combining strip `s-1`'s bottom row + strip `s` + strip `s+1`'s top row into a contiguous
    stitched buffer (zero extra DMA when `rad = 0` on 1x1/PE/LUT!). -/
def unrollRollingHaloCascade (prog : AneProgram) (hOut : Nat) : Except String AneProgram := do
  let (hFull, wFull, radii) ← checkRollingHaloChainEligible prog
  if hOut < 2 then
    throw "rolling halo cascade requires stripHeight >= 2"
  let g := prog.graph
  let m := g.nodes.size
  let shMap ← buildShapeMap g
  let sCount := ceilDiv hFull hOut
  if sCount <= 1 then
    return prog

  let mut nextVid : ValueId := g.inputs.foldl (fun acc e => max acc (e.1 + 1)) 0
  let mut newNodes : Array (IRNode AneOp) := #[]
  let mut newMeta : Array AneNodeMeta := #[]

  -- Compute per-strip disjoint row ranges `[(y0, y1)]`
  let stripBounds : Array (Nat × Nat) := Array.ofFn (n := sCount) fun s =>
    stripOutBand hFull hOut s.val

  -- `produced[l][s]` = SSA ValueId for strip `s` of layer `l` (valid once produced)
  -- Emit in wavefront order so strip `s` of layer `l` runs as soon as strips `s-1, s, s+1` of `l-1` exist!
  let mut produced : Array (Array ValueId) := Array.replicate m (Array.replicate sCount 0)
  let mut finalOutVid : ValueId := 0

  -- Layer-wavefront schedule: advance layer 0 by 1 strip ahead of each 3x3 layer
  -- Total wavefront steps = sCount + sum(radii)
  let lag : Array Nat := rollingHaloLayerLags radii
  let accLag : Nat := if m > 0 then lag[m - 1]! else 0
  let totalWaves := sCount + accLag

  for wave in [:totalWaves] do
    for l in [:m] do
      if wave >= lag[l]! && wave - lag[l]! < sCount then
        let s := wave - lag[l]!
        let (y0, y1) := stripBounds[s]!
        let stripH := y1 - y0
        let nd := g.nodes[l]!
        let origMeta := prog.getNodeMeta l
        let origOutSh ← lookupShape shMap nd.outputs[0]!.1 nd.name
        let stripOutSh : Shape4D := { origOutSh with h := stripH }
        let isLastLayer := l + 1 == m

        let (effInVid, src1Win) ←
          if l == 0 then
            let rad := radii[0]!
            let (inLo, _) := haloInBand hFull rad y0 y1
            pure (g.inputs[0]!.1, ({ rowOffset := inLo, fullH := hFull } : AneSurfaceWindow))
          else if radii[l]! == 0 then
            -- Zero halo -> ZERO extra DMA! Direct L2 feed from strip `s` of layer `l - 1`!
            pure (produced[l - 1]![s]!, ({} : AneSurfaceWindow))
          else
            -- 3x3 consumer reading from layer `l - 1`: stitch `[prev_bottom?, curr_strip, next_top?]`
            let prevOutSh ← lookupShape shMap g.nodes[l - 1]!.outputs[0]!.1 nd.name
            let cIn := prevOutSh.c
            let hasPrev := s > 0
            let hasNext := s + 1 < sCount
            let stitchH := (if hasPrev then 1 else 0) + stripH + (if hasNext then 1 else 0)
            let stitchSh : Shape4D := { n := 1, c := cIn, h := stitchH, w := wFull }
            let mut stitchInputs : Array ValueId := #[]
            let mut stitchShapes : Array Shape4D := #[]
            if hasPrev then
              let (_, prevY1) := stripBounds[s - 1]!
              let (prevY0, _) := stripBounds[s - 1]!
              let prevH := prevY1 - prevY0
              let prevFullSh : Shape4D := { n := 1, c := cIn, h := prevH, w := wFull }
              let vpSlice ← planSlice prevFullSh #[0, 0, prevH - 1, 0] #[1, cIn, prevH, wFull] #[1, 1, 1, 1]
              let haloTopVid := nextVid; nextVid := nextVid + 1
              newNodes := newNodes.push {
                name := s!"{nd.name}_s{s}_halo_top"
                op := .dmaView (dmaViewSpecFromPlan prevFullSh vpSlice) "halo_slice"
                inputs := #[produced[l - 1]![s - 1]!]
                outputs := #[(haloTopVid, shape4DToTensorType vpSlice.outShape)]
              }
              newMeta := newMeta.push { logicalId := nd.name, outPlacement := none }
              stitchInputs := stitchInputs.push haloTopVid
              stitchShapes := stitchShapes.push vpSlice.outShape
            stitchInputs := stitchInputs.push produced[l - 1]![s]!
            stitchShapes := stitchShapes.push { n := 1, c := cIn, h := stripH, w := wFull }
            if hasNext then
              let (nextY0, nextY1) := stripBounds[s + 1]!
              let nextH := nextY1 - nextY0
              let nextFullSh : Shape4D := { n := 1, c := cIn, h := nextH, w := wFull }
              let vpSlice ← planSlice nextFullSh #[0, 0, 0, 0] #[1, cIn, 1, wFull] #[1, 1, 1, 1]
              let haloBotVid := nextVid; nextVid := nextVid + 1
              newNodes := newNodes.push {
                name := s!"{nd.name}_s{s}_halo_bot"
                op := .dmaView (dmaViewSpecFromPlan nextFullSh vpSlice) "halo_slice"
                inputs := #[produced[l - 1]![s + 1]!]
                outputs := #[(haloBotVid, shape4DToTensorType vpSlice.outShape)]
              }
              newMeta := newMeta.push { logicalId := nd.name, outPlacement := none }
              stitchInputs := stitchInputs.push haloBotVid
              stitchShapes := stitchShapes.push vpSlice.outShape
            let stitchedVid := nextVid; nextVid := nextVid + 1
            let concatDesc := dmaViewSpecFromConcat stitchShapes 2 stitchSh (aliasable := false)
            newNodes := newNodes.push {
              name := s!"{nd.name}_s{s}_halo_stitch"
              op := .dmaView concatDesc "halo_stitch"
              inputs := stitchInputs
              outputs := #[(stitchedVid, shape4DToTensorType stitchSh)]
            }
            newMeta := newMeta.push { logicalId := nd.name, outPlacement := none }
            pure (stitchedVid, ({} : AneSurfaceWindow))

        let newOp : AneOp := match nd.op with
          | .neConv 3 3 sh sw _ mode cin cout w b act til wgId lbl =>
            let pt := stripPadTop y0
            let pb := stripPadBot y1 hFull
            .neConv 3 3 sh sw (.strip pt pb) mode cin cout w b act til wgId lbl
          | other => other

        let outVid := nextVid; nextVid := nextVid + 1
        produced := produced.set! l (produced[l]!.set! s outVid)
        let resWin : AneSurfaceWindow := if isLastLayer then { rowOffset := y0, fullH := hFull } else {}
        newNodes := newNodes.push {
          name := s!"{nd.name}_s{s}"
          op := newOp
          inputs := #[effInVid]
          outputs := #[(outVid, shape4DToTensorType stripOutSh)]
        }
        newMeta := newMeta.push {
          logicalId := if origMeta.logicalId.isEmpty then nd.name else origMeta.logicalId
          outPlacement := origMeta.outPlacement
          src1Win := src1Win
          resWin := resWin
          stripOutputIdx? := if isLastLayer then some 0 else none
        }
        if isLastLayer then
          finalOutVid := outVid

  -- Enforce hardware task-count limit (counting expanded `halo_stitch` concat copies!)
  let anyKern := g.nodes.any fun nd => nd.op.usesKernel
  let maxTasks := if anyKern then 255 else 256
  let expandedTasks : Nat := newNodes.foldl (fun (acc : Nat) nd =>
    match nd.op with
    | .dmaView desc _ =>
      if desc.concatAxis?.isSome && !desc.aliasable then acc + nd.inputs.size else acc + 1
    | _ => acc + 1) 0
  if expandedTasks > maxTasks then
    throw s!"rolling halo cascade at stripHeight={hOut} expands to {expandedTasks} tasks, exceeding limit {maxTasks}"

  pure {
    prog with
    graph := { g with nodes := newNodes, outputs := #[finalOutVid] }
    nodeMeta := newMeta
    stripCount := sCount
    stripHeight := min hOut hFull
    stripAxis := 2
  }

-- ============================================================================
-- W/Sequence Stripping for `H = 1, W = seq_len` Blocks (`stripAxis = 3`)
-- ============================================================================

/-- Check whether an `AneProgram` is a pointwise `H = 1, W = seq_len` block eligible for
    64B-aligned W/sequence stripping (`stripW % 32 == 0`). -/
def checkWStripEligible (prog : AneProgram) : Except String (Nat × Nat) := do
  let g := prog.graph
  let m := g.nodes.size
  if m == 0 then
    throw "W-strip ineligible: graph has 0 nodes"
  for i in [:g.outputs.size] do
    for j in [i + 1 : g.outputs.size] do
      if g.outputs[i]! == g.outputs[j]! then
        throw "W-strip ineligible: duplicate graph output ValueIds"
  let shMap ← buildShapeMap g
  let in0Sh ← lookupShape shMap g.inputs[0]!.1 "in0"
  if in0Sh.h != 1 || in0Sh.w <= 32 then
    throw s!"W-strip ineligible: requires H=1 and W>32, got {in0Sh.h}x{in0Sh.w}"
  let wFull := in0Sh.w
  for (iv, nm, _) in g.inputs do
    let s ← lookupShape shMap iv nm
    if s.h != 1 || s.w != wFull then
      throw s!"W-strip ineligible: input '{nm}' shape ({s.h}x{s.w}) != (1x{wFull})"
  for nd in g.nodes do
    let outSh ← lookupShape shMap nd.outputs[0]!.1 nd.name
    if outSh.h != 1 || outSh.w != wFull then
      throw s!"W-strip ineligible: node '{nd.name}' output shape ({outSh.h}x{outSh.w}) != (1x{wFull})"
    match nd.op with
    | .neConv 1 1 1 1 .same _ _ _ _ _ _ _ _ _ => pure ()
    | .pe (.unary _) _ => pure ()
    | .pe (.binary _ _) _ =>
      let s1 ← lookupShape shMap nd.inputs[0]! nd.name
      let s2 ← lookupShape shMap nd.inputs[1]! nd.name
      if s1.h != 1 || s1.w != wFull || s2.h != 1 || s2.w != wFull then
        throw s!"W-strip ineligible: node '{nd.name}' broadcast pe2 not supported"
    | .neAct (.act _) _ _ => pure ()
    | _ =>
      throw s!"W-strip ineligible: op on node '{nd.name}' not pointwise along W"
  let anyKern := g.nodes.any fun nd => nd.op.usesKernel
  let maxTasks := if anyKern then 255 else 256
  pure (wFull, maxTasks)

/-- Unroll a pointwise `H = 1, W = seq_len` block into interleaved 64B-aligned W strips
    using `dmaView` input slices (`Surface -> L2`) and output stitches (`L2 -> Surface`)
    so all intermediate compute layers stay 100% L2-resident (`scratchBytes == 0`)! -/
def unrollWStripCascade (prog : AneProgram) (stripW : Nat) : Except String AneProgram := do
  let (wFull, maxTasks) ← checkWStripEligible prog
  if stripW == 0 || stripW % 32 != 0 then
    throw s!"W-strip width {stripW} must be a positive multiple of 32 (64B DMA alignment)"
  let g := prog.graph
  let m := g.nodes.size
  let sCount := ceilDiv wFull stripW
  if sCount <= 1 then
    return prog
  -- Each strip emits: (#inputs slice views) + m compute nodes + (#outputs stitch views)
  let tasksPerStrip := g.inputs.size + m + g.outputs.size
  if sCount * tasksPerStrip > maxTasks then
    throw s!"W-strip width {stripW} expands to {sCount * tasksPerStrip} tasks, exceeding limit {maxTasks}"

  let shMap ← buildShapeMap g
  let pm := buildProducerMap g
  let mut nextVid : ValueId := g.inputs.foldl (fun acc e => max acc (e.1 + 1)) 0
  let mut newNodes : Array (IRNode AneOp) := Array.mkEmpty (sCount * tasksPerStrip)
  let mut newMeta : Array AneNodeMeta := Array.mkEmpty (sCount * tasksPerStrip)
  let mut finalOutVids : Array ValueId := Array.replicate g.outputs.size 0

  for s in [:sCount] do
    let (w0, w1) := stripOutBand wFull stripW s
    let curW := w1 - w0
    -- 1. Slice each graph input along W [w0, w1) via dmaView (Surface -> L2!)
    let mut slicedInVids : Array ValueId := Array.mkEmpty g.inputs.size
    for ii in [:g.inputs.size] do
      let (inVid, inNm, _) := g.inputs[ii]!
      let inSh ← lookupShape shMap inVid inNm
      let vpSlice ← planSlice inSh #[0, 0, 0, w0] #[1, inSh.c, 1, w1] #[1, 1, 1, 1]
      let sv := nextVid; nextVid := nextVid + 1
      newNodes := newNodes.push {
        name := s!"{inNm}_wslice_s{s}"
        op := .dmaView (dmaViewSpecFromPlan inSh vpSlice) "w_slice"
        inputs := #[inVid]
        outputs := #[(sv, shape4DToTensorType vpSlice.outShape)]
      }
      newMeta := newMeta.push { logicalId := inNm, outPlacement := some .l2 }
      slicedInVids := slicedInVids.push sv

    -- 2. Emit all m compute layers on width `curW` in L2
    let mut stripVidMap : Array ValueId := Array.replicate m 0
    for l in [:m] do
      let nd := g.nodes[l]!
      let origMeta := prog.getNodeMeta l
      let origOutSh ← lookupShape shMap nd.outputs[0]!.1 nd.name
      let stripOutSh : Shape4D := { origOutSh with w := curW }
      let newInps : Array ValueId := nd.inputs.map fun iv =>
        match lookupProd pm iv with
        | some (.inl ii) => slicedInVids[ii]!
        | some (.inr pIdx) => stripVidMap[pIdx]!
        | none => iv
      let nv := nextVid; nextVid := nextVid + 1
      stripVidMap := stripVidMap.set! l nv
      newNodes := newNodes.push {
        name := s!"{nd.name}_ws{s}"
        op := nd.op
        inputs := newInps
        outputs := #[(nv, shape4DToTensorType stripOutSh)]
      }
      newMeta := newMeta.push {
        logicalId := if origMeta.logicalId.isEmpty then nd.name else origMeta.logicalId
        outPlacement := origMeta.outPlacement
      }

    -- 3. Stitch each graph output into the output surface at column `w0` via dmaView (L2 -> Surface!)
    for oi in [:g.outputs.size] do
      let origOutVid := g.outputs[oi]!
      let origOutSh ← lookupShape shMap origOutVid s!"output[{oi}]"
      let prodIdx ← match lookupProd pm origOutVid with
        | some (.inr p) => pure p
        | _ => throw s!"W-strip cascade: graph output[{oi}] (%{origOutVid}) must be produced by a graph node"
      let stripProdVid := stripVidMap[prodIdx]!
      let stripOutSh : Shape4D := { origOutSh with w := curW }
      let ys := origOutSh.dramStrides
      let xs := stripOutSh.dramStrides
      let cp : ViewCopy := {
        n := 1, c := origOutSh.c, h := 1, w := curW, d := 1
        inOff := 0
        inPlaneStride := if origOutSh.c > 1 then xs[1]! else 0
        outOff := 2 * w0
        outPlaneStride := if origOutSh.c > 1 then ys[1]! else 0
        transposeCW := false
      }
      let desc : DmaViewSpec := {
        inShape := stripOutSh, outShape := origOutSh,
        dstBaseOff := 2 * w0, dstStrides := ys,
        explicitCopy? := some cp
      }
      let ov := nextVid; nextVid := nextVid + 1
      newNodes := newNodes.push {
        name := s!"out{oi}_wstitch_s{s}"
        op := .dmaView desc "w_stitch"
        inputs := #[stripProdVid]
        outputs := #[(ov, shape4DToTensorType stripOutSh)]
      }
      newMeta := newMeta.push {
        logicalId := prog.outputNames.getD oi s!"y{oi}"
        stripOutputIdx? := some oi
      }
      finalOutVids := finalOutVids.set! oi ov

  pure {
    prog with
    graph := { g with nodes := newNodes, outputs := finalOutVids }
    nodeMeta := newMeta
    stripCount := sCount
    stripHeight := stripW
    stripAxis := 3
  }

-- ============================================================================
-- Top-Level L2 Rolling Cascade Pass Driver
-- ============================================================================

/-- Estimate whether an interior strip of height `h` fits comfortably inside 2 MiB L2. -/
private def interiorStripFitsL2 (g : SSAGraph AneOp) (shMap : Array (ValueId × Shape4D))
    (radii : Array Nat) (hFull h : Nat) : Bool :=
  if h == 0 || h > hFull then false
  else
    let y0 := (hFull - h) / 2
    let y1 := y0 + h
    match propagateStripIntervals g radii hFull y0 y1 with
    | .error _ => false
    | .ok (outIvs, _) =>
      let hs := outIvs.map fun (lo, hi) => hi - lo
      estimateGraphL2Bytes g shMap (some hs) <= l2CapacityBytes

/-- Choose the largest legal strip height `h in [minH, hFull)` whose interior strip fits in 2 MiB L2,
    falling back to `minH` when even `minH` requires minimal DRAM spill (e.g. 1080p frames). -/
def chooseAutoHStripHeight (g : SSAGraph AneOp) (shMap : Array (ValueId × Shape4D))
    (radii : Array Nat) (hFull minH : Nat) : Option Nat := Id.run do
  if minH == 0 || minH >= hFull then
    return none
  if interiorStripFitsL2 g shMap radii hFull minH then
    let mut lo := minH
    let mut hi := hFull - 1
    let mut best := minH
    for _ in [:20] do
      if lo > hi then break
      let mid := (lo + hi) / 2
      if interiorStripFitsL2 g shMap radii hFull mid then
        best := mid
        lo := mid + 1
      else
        if mid == 0 then break
        hi := mid - 1
    return some best
  return some minH

/-- Pure SSA L2 rolling cascade pass on `AneProgram`.
    - `.off`: returns `prog` unchanged.
    - `.fixed hOut`: unrolls at fixed H-strip height `hOut` (or fails with exact eligibility diagnostics).
    - `.auto`: if unstripped estimated L2 exceeds 2 MiB, automatically selects optimal H-strip
      cascade (or W-strip sequence cascade for `H = 1, W = seq_len` blocks!). -/
def cascadePass (prog : AneProgram) (cfg : CascadeConfig := {}) : Except String AneProgram := do
  match cfg.stripMode with
  | .off => pure prog
  | .fixed hOut =>
    let (hFull, _, radii, maxTasks) ← checkHStripEligible prog
    unrollHStripCascade prog hFull radii maxTasks hOut
  | .auto =>
    let shMap ← buildShapeMap prog.graph
    let unstrippedPeak := estimateGraphL2Bytes prog.graph shMap none
    if unstrippedPeak <= l2CapacityBytes then
      return prog
    -- 1. If rolling halo stitching is explicitly requested and eligible, try it first (falling back to H-strip!)
    if cfg.allowRollingHaloStitch then
      if let .ok (hFull, _, _) := checkRollingHaloChainEligible prog then
        let hOut := max 2 (min 16 (hFull / 4))
        if let .ok rolled := unrollRollingHaloCascade prog hOut then
          return rolled
    -- 2. Try H-strip rolling cascade
    match checkHStripEligible prog with
    | .ok (hFull, _, radii, maxTasks) =>
      let m := prog.graph.nodes.size
      let maxStrips := maxTasks / m
      if maxStrips > 1 then
        let minH := ceilDiv hFull maxStrips
        if let some hOut := chooseAutoHStripHeight prog.graph shMap radii hFull minH then
          return ← unrollHStripCascade prog hFull radii maxTasks hOut
    | .error _ =>
      -- 3. Try W/sequence stripping for `H = 1, W = seq_len` blocks (accounting for `w_slice` L2 buffers!)
      if cfg.allowSequenceWStrip then
        if let .ok (wFull, maxTasks) := checkWStripEligible prog then
          let tasksPerStrip := prog.graph.inputs.size + prog.graph.nodes.size + prog.graph.outputs.size
          let maxStrips := maxTasks / max 1 tasksPerStrip
          if maxStrips > 1 then
            let minW := alignUp (ceilDiv wFull maxStrips) 32
            let mut bestW : Option Nat := none
            let mut curW := minW
            for _ in [:32] do
              if curW >= wFull then break
              let wShMap := shMap.map fun (v, s) => (v, { s with w := curW })
              let mut wSliceL2 : Nat := 0
              for (inVid, _, _) in prog.graph.inputs do
                if let some (_, s) := wShMap.find? (fun p => p.1 == inVid) then
                  wSliceL2 := wSliceL2 + l2ResidentBytes4D s
              if estimateGraphL2Bytes prog.graph wShMap none + wSliceL2 <= l2CapacityBytes then
                bestW := some curW
                curW := curW + 32
              else
                break
            if let some sw := bestW then
              if let .ok wStripped := unrollWStripCascade prog sw then
                return wStripped
    -- Otherwise leave unstripped so L3 MemoryPlan applies selective L2<->DRAM BAR 3 spilling!
    pure prog

end Vane.Backend.Cascade
