import Vane.IR.Rewrite
import Vane.IR.Passes.DCE

namespace Vane.IR

open Vane.Conv

-- ============================================================================
-- Pass 4: High-Level Fusion on `SSAGraph HighOp` (Conv+BN f64, Conv+Gelu)
-- ============================================================================

/-- Compute float64 per-channel `(factor, biasShift)` for ONNX `BatchNormalization`. -/
def computeBNAffineF64
    (scale bias mean variance : Array Float) (eps : Float) (cout : Nat) (ctx : String) :
    Except String (Array Float × Array Float) := do
  if scale.size != cout then
    throw s!"{ctx}: BatchNormalization scale length {scale.size} != channels {cout}"
  if bias.size != cout then
    throw s!"{ctx}: BatchNormalization B length {bias.size} != channels {cout}"
  if mean.size != cout then
    throw s!"{ctx}: BatchNormalization mean length {mean.size} != channels {cout}"
  if variance.size != cout then
    throw s!"{ctx}: BatchNormalization var length {variance.size} != channels {cout}"
  if eps.isNaN || eps.isInf || eps < 0.0 then
    throw s!"{ctx}: BatchNormalization invalid epsilon {eps}"
  let mut factor : Array Float := Array.mkEmpty cout
  let mut bShift : Array Float := Array.mkEmpty cout
  for o in [:cout] do
    let v := variance[o]!
    if v < 0.0 then
      throw s!"{ctx}: BatchNormalization negative variance[{o}]={v}"
    let denom := v + eps
    if denom <= 0.0 || denom.isNaN || denom.isInf then
      throw s!"{ctx}: BatchNormalization invalid variance[{o}]+eps={denom}"
    let f := scale[o]! / Float.sqrt denom
    if f.isNaN || f.isInf then
      throw s!"{ctx}: BatchNormalization non-finite scale factor at channel {o}"
    factor := factor.push f
    bShift := bShift.push (bias[o]! - mean[o]! * f)
  pure (factor, bShift)

private def isDequantProducerHigh (g : SSAGraph HighOp) (v : ValueId) : Bool :=
  match g.producerOf? v with
  | some (_, nd, _) =>
    match nd.op with
    | .core (.dequantize _) => true
    | _ => false
  | none => false

private def lookupConstHigh (g : SSAGraph HighOp) (v : ValueId) : Option (TensorType × ConstPayload) :=
  match g.producerOf? v with
  | some (_, nd, 0) =>
    match nd.op, nd.outputs[0]? with
    | .core (.const p), some (_, ty) => some (ty, p)
    | _, _ => none
  | _ => none

/-- Single pass folding single-consumer `Conv + BatchNorm` in exact float64. -/
private def fuseConvBatchNormPass (g : SSAGraph HighOp) : Except String (SSAGraph HighOp × Nat) := do
  let mut cur := g
  let mut fused : Nat := 0
  let maxSteps := g.nodes.size + 1

  for _ in [:maxSteps] do
    let mut found? : Option (Nat × Nat × Float) := none
    for i in [:cur.nodes.size] do
      if found?.isNone then
        let bn := cur.nodes[i]!
        match bn.op, bn.inputs.toList, bn.outputs.toList with
        | .batchNorm eps, [xId, _, _, _, _], [(_, _)] =>
          if cur.consumerCount xId == 1 then
            match cur.producerOf? xId with
            | some (convIdx, convNd, 0) =>
              match convNd.op with
              | .core (.conv _ _ _ _ _ _ _ .none) =>
                if convNd.inputs.size >= 2 then
                  match lookupConstHigh cur convNd.inputs[1]! with
                  | some (_, .f32 _) | some (_, .fp16 _) =>
                    found? := some (convIdx, i, eps)
                  | _ => pure ()
              | _ => pure ()
            | _ => pure ()
        | _, _, _ => pure ()

    match found? with
    | none => break
    | some (convIdx, bnIdx, eps) =>
      let convNd := cur.nodes[convIdx]!
      let bnNd := cur.nodes[bnIdx]!
      let ctx := s!"node '{bnNd.name}' (Conv+BatchNormalization)"
      let (wTy, wPay) := (lookupConstHigh cur convNd.inputs[1]!).get!
      if wTy.shape.size != 4 then
        throw s!"{ctx}: Conv weight must be 4D, got {wTy.shape.toList}"
      let cout := wTy.shape[0]!
      let some wF64Orig := wPay.toFloat64?
        | throw s!"{ctx}: Conv weight must be float"
      if cout == 0 || wF64Orig.isEmpty || wF64Orig.size % cout != 0 then
        throw s!"{ctx}: invalid Conv weight length {wF64Orig.size} for Cout={cout}"

      let mut wF64 := wF64Orig
      let mut convBias : Array Float ←
        if convNd.inputs.size >= 3 then
          let some (_, bPay) := lookupConstHigh cur convNd.inputs[2]!
            | throw s!"{ctx}: Conv bias must be constant"
          let some bv := bPay.toFloat64?
            | throw s!"{ctx}: Conv bias must be float"
          if bv.size != cout then
            throw s!"{ctx}: Conv bias length {bv.size} != Cout={cout}"
          pure bv
        else
          pure (Array.replicate cout 0.0)

      let readBnVec := fun (slot : Nat) (label : String) => do
        let vid := bnNd.inputs[slot]!
        let some (_, p) := lookupConstHigh cur vid
          | throw s!"{ctx}: BatchNormalization {label} must be a constant initializer"
        let some vs := p.toFloat64?
          | throw s!"{ctx}: BatchNormalization {label} must be float"
        pure vs

      let scale ← readBnVec 1 "scale"
      let bVec ← readBnVec 2 "B"
      let mean ← readBnVec 3 "mean"
      let variance ← readBnVec 4 "var"
      let (factor, bnBiasShift) ← computeBNAffineF64 scale bVec mean variance eps cout ctx

      let perOut := wF64.size / cout
      for o in [:cout] do
        let f := factor[o]!
        let base := o * perOut
        for k in [:perOut] do
          wF64 := wF64.set! (base + k) (wF64[base + k]! * f)
        convBias := convBias.set! o (convBias[o]! * f + bnBiasShift[o]!)

      let newWId := cur.nextValueId
      let newBId := newWId + 1
      let convOutId := convNd.outputs[0]!.1
      let bnOutId := bnNd.outputs[0]!.1
      let newConvOp := match convNd.op with
        | .core (.conv kh kw sh sw pads groups _ act) =>
          HighOp.core (.conv kh kw sh sw pads groups true act)
        | other => other
      let newWNode : IRNode HighOp := {
        name := s!"{convNd.name}_bn_w"
        op := .core (.const (.f32 wF64))
        inputs := #[]
        outputs := #[(newWId, { shape := wTy.shape, dtype := .f32 })]
      }
      let newBNode : IRNode HighOp := {
        name := s!"{convNd.name}_bn_b"
        op := .core (.const (.f32 convBias))
        inputs := #[]
        outputs := #[(newBId, { shape := #[cout], dtype := .f32 })]
      }
      let updatedConv : IRNode HighOp := {
        convNd with
        op := newConvOp
        inputs := #[convNd.inputs[0]!, newWId, newBId]
      }

      -- Rebuild node list: insert folded consts right before convIdx, update convIdx, drop bnIdx
      let mut newNodes : Array (IRNode HighOp) := Array.mkEmpty (cur.nodes.size + 1)
      for idx in [:cur.nodes.size] do
        if idx == convIdx then
          newNodes := newNodes.push newWNode
          newNodes := newNodes.push newBNode
          newNodes := newNodes.push updatedConv
        else if idx == bnIdx then
          pure ()
        else
          newNodes := newNodes.push cur.nodes[idx]!
      cur := dce (({ cur with nodes := newNodes }).replaceUses bnOutId convOutId)
      fused := fused + 1

  pure (cur, fused)

/-- Fold single-consumer `Conv + Gelu` → `CoreOp.conv` with fused `.lut .gelu` epilogue
    (skipping quantized `dequantize` weights to respect M8A no-fused-LUT hardware legality). -/
private def fuseConvGeluPass (g : SSAGraph HighOp) : SSAGraph HighOp × Nat := Id.run do
  let mut cur := g
  let mut fused : Nat := 0
  let maxSteps := g.nodes.size + 1

  for _ in [:maxSteps] do
    let mut found? : Option (Nat × Nat) := none
    for i in [:cur.nodes.size] do
      if found?.isNone then
        let nd := cur.nodes[i]!
        match nd.op, nd.inputs.toList, nd.outputs.toList with
        | .gelu .exact, [inId], [(_, _)] =>
          if cur.consumerCount inId == 1 then
            match cur.producerOf? inId with
            | some (convIdx, convNd, 0) =>
              match convNd.op with
              | .core (.conv _ _ _ _ _ _ _ .none) =>
                if convNd.inputs.size >= 2 && !isDequantProducerHigh cur convNd.inputs[1]! then
                  found? := some (convIdx, i)
              | _ => pure ()
            | _ => pure ()
        | _, _, _ => pure ()

    match found? with
    | none => break
    | some (convIdx, geluIdx) =>
      let convNd := cur.nodes[convIdx]!
      let geluNd := cur.nodes[geluIdx]!
      let convOutId := convNd.outputs[0]!.1
      let geluOutId := geluNd.outputs[0]!.1
      let newConvOp := match convNd.op with
        | .core (.conv kh kw sh sw pads groups hasBias _) =>
          HighOp.core (.conv kh kw sh sw pads groups hasBias (.lut .gelu))
        | other => other
      let mut newNodes : Array (IRNode HighOp) := Array.mkEmpty cur.nodes.size
      for idx in [:cur.nodes.size] do
        if idx == convIdx then
          newNodes := newNodes.push { convNd with op := newConvOp }
        else if idx == geluIdx then
          pure ()
        else
          newNodes := newNodes.push cur.nodes[idx]!
      cur := dce (({ cur with nodes := newNodes }).replaceUses geluOutId convOutId)
      fused := fused + 1

  return (cur, fused)

/-- Run all high-level fusions (`Conv + BatchNorm` f64 folding, `Conv + Gelu` epilogue fusion). -/
def highFuse (g : SSAGraph HighOp) : Except String (SSAGraph HighOp × Nat × Nat) := do
  let (g1, bnCnt) ← fuseConvBatchNormPass g
  let (g2, geluCnt) := fuseConvGeluPass g1
  pure (g2, bnCnt, geluCnt)

end Vane.IR
