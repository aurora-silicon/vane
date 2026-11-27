import Vane.Binary
import Vane.TaskStream
import Vane.Register
import Vane.Tiling
import Vane.Quant
import Vane.Lut
import Vane.Conv
import Vane.View
import Vane.Ane.Types

namespace Vane.Backend.ChannelSplit

open Vane.Binary
open Vane.TaskStream
open Vane.Register
open Vane.Tiling
open Vane.Quant
open Vane.Lut
open Vane.Conv
open Vane.View
open Vane.IR
open Vane.Ane

-- ============================================================================
-- L2 Pass: General SOK Output-Channel Splitting on `SSAGraph AneOp`
-- ============================================================================

/-- Partition `cout` output channels into `numSlices` balanced contiguous slice widths
    differing by at most 1 (`base + 1` for the first `cout % numSlices` slices, then `base`). -/
def balancedSlices (cout numSlices : Nat) : List Nat :=
  let base := cout / numSlices
  let rem := cout % numSlices
  (List.range numSlices).map fun i => base + if i < rem then 1 else 0

/-- Extract the weight group identifier of an `AneOp` (`0` if weightless / ungrouped). -/
def opWeightGroupId : AneOp → Nat
  | .neConv _ _ _ _ _ _ _ _ _ _ _ _ wg _ => wg
  | .neAct _ _ wg => wg
  | _ => 0

/-- Maximum existing `weightGroupId` across all nodes in an `SSAGraph AneOp`. -/
def maxExistingWeightGroupId (g : SSAGraph AneOp) : Nat :=
  g.nodes.foldl (fun acc nd => max acc (opWeightGroupId nd.op)) 0

/-- Check whether a single-task `3x3` dense stride-1 fp16 conv fits within 64 KiB kernel chunks. -/
private def conv3x3SingleTaskFits (target : Target) (sh : ConvSpec)
    (extraLutBytes : Nat) (biasInKernel : Bool) : Bool := Id.run do
  let maxF := min 4 (ceilLog2 sh.cout)
  for logNE in #[4, 3, 2, 1, 0] do
    for fat in #[0, 1] do
      for step in [:maxF + 1] do
        let f := maxF - step
        if f + fat <= logNE && (logNE - f - fat) <= 3 then
          if let some ocg := chooseConvOcg? target sh f then
            let probe : Tiling := {
              logNE := logNE, f := f, fat := fat, ocg := ocg,
              shPref := 0, shMax := 0, tileHeight := 1
            }
            if (checkConvDenseChunks64K sh probe extraLutBytes biasInKernel).isOk &&
               (validateConvOcgSafety target sh probe).isOk then
              return true
  return false

/-- Compute SOK channel slices for a `3x3` dense stride-1 fp16 conv when single-task chunks exceed 64 KiB. -/
private def chooseConv3x3OutputSlices
    (target : Target) (h w cin cout extraLutBytes : Nat) (hasEffBias : Bool) :
    Except String (Array Nat) := do
  let fullSh : ConvSpec := { kernelSize := 3, mode := .dense, h := h, w := w, cin := cin, cout := cout, stride := 1, pad := .same }
  if conv3x3SingleTaskFits target fullSh extraLutBytes (hasEffBias && cout > 1) then
    return #[cout]
  -- Find largest chunk size `dc <= cout` that fits 64 KiB kernel chunks
  let mut bestDc : Nat := 0
  let mut lo : Nat := 1
  let mut hi : Nat := cout
  for _ in [:20] do
    if lo > hi then break
    let mid := (lo + hi) / 2
    let shM : ConvSpec := { fullSh with cout := mid }
    if conv3x3SingleTaskFits target shM extraLutBytes (hasEffBias && mid > 1) then
      bestDc := mid
      lo := mid + 1
    else
      if mid == 0 then break
      hi := mid - 1
  if bestDc == 0 then
    throw s!"SOK conv3x3 split: even Cout=1 exceeds 64 KiB kernel chunk cap for Cin={cin}"
  let numSlices := ceilDiv cout bestDc
  pure (balancedSlices cout numSlices).toArray

/-- Check whether a single-task quantized `3x3` dense stride-1 conv fits within 64 KiB kernel chunks. -/
private def conv3x3SingleTaskFitsQuant (target : Target) (sh : ConvSpec)
    (q : QuantWeightSpec) (hasBias : Bool) : Bool := Id.run do
  let qp := quantChunkParamsFor q sh.cout hasBias
  let maxF := min 4 (ceilLog2 sh.cout)
  for logNE in #[4, 3, 2, 1, 0] do
    for fat in #[0, 1] do
      for step in [:maxF + 1] do
        let f := maxF - step
        if f + fat <= logNE && (logNE - f - fat) <= 3 then
          if let some ocg := chooseConvOcg? target sh f then
            let probe : Tiling := {
              logNE := logNE, f := f, fat := fat, ocg := ocg,
              shPref := 0, shMax := 0, tileHeight := 1
            }
            if (checkConvQuantChunks64K sh probe qp).isOk &&
               (validateConvOcgSafety target sh probe).isOk then
              return true
  return false

/-- Compute SOK channel slices for a quantized `3x3` dense stride-1 conv when single-task chunks exceed 64 KiB. -/
private def chooseConv3x3OutputSlicesQuant
    (target : Target) (h w cin cout : Nat) (q : QuantWeightSpec) (hasBias : Bool) :
    Except String (Array Nat) := do
  let fullSh : ConvSpec := { kernelSize := 3, mode := .dense, h := h, w := w, cin := cin, cout := cout, stride := 1, pad := .same }
  if conv3x3SingleTaskFitsQuant target fullSh q hasBias then
    return #[cout]
  let mut bestDc : Nat := 0
  let mut lo : Nat := 1
  let mut hi : Nat := cout
  for _ in [:20] do
    if lo > hi then break
    let mid := (lo + hi) / 2
    let shM : ConvSpec := { fullSh with cout := mid }
    if conv3x3SingleTaskFitsQuant target shM q hasBias then
      bestDc := mid
      lo := mid + 1
    else
      if mid == 0 then break
      hi := mid - 1
  if bestDc == 0 then
    throw s!"SOK quantized conv3x3 split: even Cout=1 exceeds 64 KiB kernel chunk cap for Cin={cin}"
  let numSlices := ceilDiv cout bestDc
  pure (balancedSlices cout numSlices).toArray

/-- Pure SSA SOK (Split-Output-Channels) pass on `AneProgram` (`SSAGraph AneOp`).
    Splits any wide-`Cout` `neConv` node (1x1 fp16/quant, or 3x3 dense stride-1 fp16/quant)
    into parallel channel slices (each assigned a globally fresh `weightGroupId`) + an
    `AneOp.dmaView` `concat(axis=1)` join node (`aliasable := true` for 1x1 so L3 `MemoryPlan`
    lowers it with zero copies!). -/
def channelSplitPass (prog : AneProgram) : Except String AneProgram := do
  let g := prog.graph
  let mut nextVid := g.nextValueId
  let maxExistingWgId : Nat := maxExistingWeightGroupId g
  let mut nextWgId : Nat := maxExistingWgId + 1
  let mut newNodes : Array (IRNode AneOp) := Array.mkEmpty g.nodes.size
  let mut newMeta : Array AneNodeMeta := Array.mkEmpty g.nodes.size
  let mut shMap : Array (ValueId × Shape4D) := #[]
  for (vid, _, ty) in g.inputs do
    shMap := shMap.push (vid, ← tensorTypeToShape4D ty "ChannelSplit input")

  let lookupSh := fun (m : Array (ValueId × Shape4D)) (v : ValueId) => do
    match m.find? fun p => p.1 == v with
    | some (_, s) => pure s
    | none => throw s!"ChannelSplit: missing shape for SSA value %{v}"

  for idx in [:g.nodes.size] do
    let nd := g.nodes[idx]!
    let ndMeta := prog.getNodeMeta idx
    match nd.op with
    | .neConv kh kw sh sw pad mode cin cout weight bias act tiling _wgId opLbl =>
      if nd.inputs.size != 1 || nd.outputs.size != 1 then
        throw s!"ChannelSplit: neConv '{nd.name}' expects 1 input and 1 output"
      let inVid := nd.inputs[0]!
      let (outVid, outTy) := nd.outputs[0]!
      let inSh ← lookupSh shMap inVid
      let outSh ← tensorTypeToShape4D outTy s!"ChannelSplit '{nd.name}'"

      if kh == 1 && kw == 1 && mode == .dense then
        match weight with
        | .quant q =>
          if let .lut lspec := act then
            throw s!"node '{nd.name}' ({opLbl}): fused LUT activation '{lspec.name}' is not supported with quantized weights"
          validateQuantWeightSpec q cout cin 1
          let slices ← chooseConv1x1OutputSlicesQuant g.target inSh.h inSh.w cin cout q bias.isSome
          if slices.size <= 1 then
            newNodes := newNodes.push nd
            newMeta := newMeta.push ndMeta
            shMap := shMap.push (outVid, outSh)
          else
            if let some b := bias then
              if b.size != 2 * cout then
                throw s!"node '{nd.name}' ({opLbl}): bias byte size {b.size} != expected {2 * cout}"
            let starts := prefixStarts slices.toList
            let mut sliceVids : Array ValueId := Array.mkEmpty slices.size
            let mut sliceShapes : Array Shape4D := Array.mkEmpty slices.size
            for si in [:slices.size] do
              let c0 := starts[si]!
              let dc := slices[si]!
              let qSlice ← sliceQuantWeightSpec q c0 dc cin 1
              let bSlice? := bias.map fun b => b.extract (2 * c0) (2 * (c0 + dc))
              let slVid := nextVid; nextVid := nextVid + 1
              let slSh : Shape4D := { outSh with c := dc }
              let slWg := nextWgId; nextWgId := nextWgId + 1
              newNodes := newNodes.push {
                name := s!"{nd.name}_s{si}"
                op := .neConv 1 1 1 1 .same .dense cin dc (.quant qSlice) bSlice? act tiling slWg opLbl
                inputs := #[inVid]
                outputs := #[(slVid, shape4DToTensorType slSh)]
              }
              newMeta := newMeta.push { ndMeta with logicalId := nd.name }
              shMap := shMap.push (slVid, slSh)
              sliceVids := sliceVids.push slVid
              sliceShapes := sliceShapes.push slSh
            let joinDesc := dmaViewSpecFromConcat sliceShapes 1 outSh (aliasable := true)
            newNodes := newNodes.push {
              name := s!"{nd.name}_sok_join"
              op := .dmaView joinDesc "sok_concat"
              inputs := sliceVids
              outputs := #[(outVid, outTy)]
            }
            newMeta := newMeta.push { ndMeta with logicalId := nd.name }
            shMap := shMap.push (outVid, outSh)

        | .fp16 wBytes =>
          let lutPayload? ← match act with
            | .lut lspec => some <$> buildLutTable lspec
            | _ => pure none
          let shiftBits? := lutPayload?.bind (·.inputShiftBits)
          let hasEffBias := bias.isSome || shiftBits?.isSome
          let extraLutBytes := if lutPayload?.isSome then lutTableBytes else 0
          let slices ← chooseConv1x1OutputSlices g.target inSh.h inSh.w cin cout extraLutBytes hasEffBias
          if slices.size <= 1 then
            newNodes := newNodes.push nd
            newMeta := newMeta.push ndMeta
            shMap := shMap.push (outVid, outSh)
          else
            if wBytes.size != 2 * cout * cin then
              throw s!"node '{nd.name}' ({opLbl}): weights byte size {wBytes.size} != expected {2 * cout * cin}"
            if let some b := bias then
              if b.size != 2 * cout then
                throw s!"node '{nd.name}' ({opLbl}): bias byte size {b.size} != expected {2 * cout}"
            let starts := prefixStarts slices.toList
            let mut sliceVids : Array ValueId := Array.mkEmpty slices.size
            let mut sliceShapes : Array Shape4D := Array.mkEmpty slices.size
            for si in [:slices.size] do
              let c0 := starts[si]!
              let dc := slices[si]!
              let wSlice := wBytes.extract (2 * c0 * cin) (2 * (c0 + dc) * cin)
              let bSlice? := bias.map fun b => b.extract (2 * c0) (2 * (c0 + dc))
              let slVid := nextVid; nextVid := nextVid + 1
              let slSh : Shape4D := { outSh with c := dc }
              let slWg := nextWgId; nextWgId := nextWgId + 1
              newNodes := newNodes.push {
                name := s!"{nd.name}_s{si}"
                op := .neConv 1 1 1 1 .same .dense cin dc (.fp16 wSlice) bSlice? act tiling slWg opLbl
                inputs := #[inVid]
                outputs := #[(slVid, shape4DToTensorType slSh)]
              }
              newMeta := newMeta.push { ndMeta with logicalId := nd.name }
              shMap := shMap.push (slVid, slSh)
              sliceVids := sliceVids.push slVid
              sliceShapes := sliceShapes.push slSh
            let joinDesc := dmaViewSpecFromConcat sliceShapes 1 outSh (aliasable := true)
            newNodes := newNodes.push {
              name := s!"{nd.name}_sok_join"
              op := .dmaView joinDesc "sok_concat"
              inputs := sliceVids
              outputs := #[(outVid, outTy)]
            }
            newMeta := newMeta.push { ndMeta with logicalId := nd.name }
            shMap := shMap.push (outVid, outSh)

      else if kh == 3 && kw == 3 && mode == .dense && sh == 1 && sw == 1 && pad == .same then
        match weight with
        | .fp16 wBytes =>
          let lutPayload? ← match act with
            | .lut lspec => some <$> buildLutTable lspec
            | _ => pure none
          let shiftBits? := lutPayload?.bind (·.inputShiftBits)
          let hasEffBias := bias.isSome || shiftBits?.isSome
          let extraLutBytes := if lutPayload?.isSome then lutTableBytes else 0
          let slices ← chooseConv3x3OutputSlices g.target inSh.h inSh.w cin cout extraLutBytes hasEffBias
          if slices.size <= 1 then
            newNodes := newNodes.push nd
            newMeta := newMeta.push ndMeta
            shMap := shMap.push (outVid, outSh)
          else
            if wBytes.size != 2 * cout * cin * 9 then
              throw s!"node '{nd.name}' ({opLbl}): weights byte size {wBytes.size} != expected {2 * cout * cin * 9}"
            if let some b := bias then
              if b.size != 2 * cout then
                throw s!"node '{nd.name}' ({opLbl}): bias byte size {b.size} != expected {2 * cout}"
            let starts := prefixStarts slices.toList
            let mut sliceVids : Array ValueId := Array.mkEmpty slices.size
            let mut sliceShapes : Array Shape4D := Array.mkEmpty slices.size
            for si in [:slices.size] do
              let c0 := starts[si]!
              let dc := slices[si]!
              let wSlice := wBytes.extract (2 * c0 * cin * 9) (2 * (c0 + dc) * cin * 9)
              let bSlice? := bias.map fun b => b.extract (2 * c0) (2 * (c0 + dc))
              let slVid := nextVid; nextVid := nextVid + 1
              let slSh : Shape4D := { outSh with c := dc }
              let slWg := nextWgId; nextWgId := nextWgId + 1
              newNodes := newNodes.push {
                name := s!"{nd.name}_s{si}"
                op := .neConv 3 3 1 1 .same .dense cin dc (.fp16 wSlice) bSlice? act tiling slWg opLbl
                inputs := #[inVid]
                outputs := #[(slVid, shape4DToTensorType slSh)]
              }
              newMeta := newMeta.push { ndMeta with logicalId := nd.name }
              shMap := shMap.push (slVid, slSh)
              sliceVids := sliceVids.push slVid
              sliceShapes := sliceShapes.push slSh
            -- For 3x3, output surfaces require `cout == s.channels`, so aliasable is false when writing directly to an output surface
            let isGraphOut := g.outputs.contains outVid
            let joinDesc := dmaViewSpecFromConcat sliceShapes 1 outSh (aliasable := !isGraphOut)
            newNodes := newNodes.push {
              name := s!"{nd.name}_sok_join"
              op := .dmaView joinDesc "sok_concat"
              inputs := sliceVids
              outputs := #[(outVid, outTy)]
            }
            newMeta := newMeta.push { ndMeta with logicalId := nd.name }
            shMap := shMap.push (outVid, outSh)
        | .quant q =>
          if let .lut lspec := act then
            throw s!"node '{nd.name}' ({opLbl}): fused LUT activation '{lspec.name}' is not supported with quantized weights"
          validateQuantWeightSpec q cout cin 9
          let slices ← chooseConv3x3OutputSlicesQuant g.target inSh.h inSh.w cin cout q bias.isSome
          if slices.size <= 1 then
            newNodes := newNodes.push nd
            newMeta := newMeta.push ndMeta
            shMap := shMap.push (outVid, outSh)
          else
            if let some b := bias then
              if b.size != 2 * cout then
                throw s!"node '{nd.name}' ({opLbl}): bias byte size {b.size} != expected {2 * cout}"
            let starts := prefixStarts slices.toList
            let mut sliceVids : Array ValueId := Array.mkEmpty slices.size
            let mut sliceShapes : Array Shape4D := Array.mkEmpty slices.size
            for si in [:slices.size] do
              let c0 := starts[si]!
              let dc := slices[si]!
              let qSlice ← sliceQuantWeightSpec q c0 dc cin 9
              let bSlice? := bias.map fun b => b.extract (2 * c0) (2 * (c0 + dc))
              let slVid := nextVid; nextVid := nextVid + 1
              let slSh : Shape4D := { outSh with c := dc }
              let slWg := nextWgId; nextWgId := nextWgId + 1
              newNodes := newNodes.push {
                name := s!"{nd.name}_s{si}"
                op := .neConv 3 3 1 1 .same .dense cin dc (.quant qSlice) bSlice? act tiling slWg opLbl
                inputs := #[inVid]
                outputs := #[(slVid, shape4DToTensorType slSh)]
              }
              newMeta := newMeta.push { ndMeta with logicalId := nd.name }
              shMap := shMap.push (slVid, slSh)
              sliceVids := sliceVids.push slVid
              sliceShapes := sliceShapes.push slSh
            let isGraphOut := g.outputs.contains outVid
            let joinDesc := dmaViewSpecFromConcat sliceShapes 1 outSh (aliasable := !isGraphOut)
            newNodes := newNodes.push {
              name := s!"{nd.name}_sok_join"
              op := .dmaView joinDesc "sok_concat"
              inputs := sliceVids
              outputs := #[(outVid, outTy)]
            }
            newMeta := newMeta.push { ndMeta with logicalId := nd.name }
            shMap := shMap.push (outVid, outSh)
      else
        newNodes := newNodes.push nd
        newMeta := newMeta.push ndMeta
        shMap := shMap.push (outVid, outSh)

    | _ =>
      newNodes := newNodes.push nd
      newMeta := newMeta.push ndMeta
      for (ov, oty) in nd.outputs do
        shMap := shMap.push (ov, ← tensorTypeToShape4D oty s!"ChannelSplit '{nd.name}'")

  pure { prog with graph := { g with nodes := newNodes }, nodeMeta := newMeta }

end Vane.Backend.ChannelSplit
