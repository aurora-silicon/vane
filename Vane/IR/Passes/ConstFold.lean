import Vane.IR.Rewrite
import Vane.IR.Passes.Quantize

namespace Vane.IR

open Vane.Binary
open Vane.Float16
open Vane.Quant

-- ============================================================================
-- Pass 3: Polymorphic Constant Folding (via HasCoreOps)
-- ============================================================================

/-- Transpose an array of elements of logical shape `shape` according to valid `perm`. -/
private def transposeFloatArray (shape : Array Nat) (perm : Array Nat) (vals : Array Float) :
    Option (Array Float) := Id.run do
  let shList := shape.toList
  if !isValidPerm perm || perm.size != shList.length || !shList.all (· > 0) then
    return none
  let total := shapeProd shList
  if vals.size != total then
    return none
  let outShape := transposeShape shList perm
  let inv := invPerm perm
  let outTotal := shapeProd outShape
  let mut out : Array Float := Array.mkEmpty outTotal
  for outIdx in [:outTotal] do
    let outCoords := unflatten outShape outIdx
    let inCoords := (List.range shList.length).map fun a =>
      outCoords.getD (inv.getD a 0) 0
    let fIdx := flatIndex shList inCoords
    out := out.push (vals.getD fIdx 0.0)
  return some out

/-- Transpose raw fp16 little-endian bytes of logical shape `shape` according to valid `perm`. -/
private def transposeFp16Bytes (shape : Array Nat) (perm : Array Nat) (bs : ByteArray) :
    Option ByteArray := Id.run do
  let shList := shape.toList
  if !isValidPerm perm || perm.size != shList.length || !shList.all (· > 0) then
    return none
  let total := shapeProd shList
  if bs.size != 2 * total then
    return none
  let outShape := transposeShape shList perm
  let inv := invPerm perm
  let outTotal := shapeProd outShape
  let mut out := ByteArray.emptyWithCapacity (2 * outTotal)
  for outIdx in [:outTotal] do
    let outCoords := unflatten outShape outIdx
    let inCoords := (List.range shList.length).map fun a =>
      outCoords.getD (inv.getD a 0) 0
    let fIdx := flatIndex shList inCoords
    let lo := bs.data.getD (2 * fIdx) 0
    let hi := bs.data.getD (2 * fIdx + 1) 0
    out := out.push lo |>.push hi
  return some out

private def evalScalarUnary (op : ElemUnaryOp) (x : Float) : Option Float :=
  let r? : Option Float := match op with
    | .abs => some (Float.abs x)
    | .neg => some (-x)
    | .square => some (x * x)
    | .log => if x > 0.0 then some (Float.log x) else none
    | .softplus => some (Float.log (1.0 + Float.exp x))
    | .scalarAdd b => some (x + f16ToFloat b)
    | .scalarSub b => some (x - f16ToFloat b)
    | .scalarRSub b => some (f16ToFloat b - x)
    | .scalarMul b => some (x * f16ToFloat b)
    | .scalarDiv b =>
      let d := f16ToFloat b
      if d == 0.0 then none else some (x / d)
    | .linearAct a b => some (x * f16ToFloat a + f16ToFloat b)
  match r? with
  | some r => if r.isNaN || r.isInf then none else some r
  | none => none

private def evalScalarBinary (op : ElemBinaryOp) (a b : Float) : Option Float :=
  let r? : Option Float := match op with
    | .add => some (a + b)
    | .sub => some (a - b)
    | .mul => some (a * b)
    | .div => if b == 0.0 then none else some (a / b)
    | .max => some (if a >= b then a else b)
    | .min => some (if a <= b then a else b)
  match r? with
  | some r => if r.isNaN || r.isInf then none else some r
  | none => none

/-- Check whether every consumer of `deqId` is a quant-capable `conv` (as its weight input 1)
    and `deqId` is not a graph output. -/
private def allConsumersQuantCapableConv {Op : Type} [HasCoreOps Op]
    (g : SSAGraph Op) (deqId : ValueId) : Bool :=
  if g.outputs.contains deqId then
    false
  else
    let consumers := g.nodes.filter fun nd => nd.inputs.contains deqId
    if consumers.isEmpty then
      true -- dead; leave for DCE
    else
      consumers.all fun nd =>
        match HasCoreOps.asCore? nd.op with
        | some coreOp =>
          isQuantCapableConvWithAct coreOp &&
          nd.inputs.size >= 2 &&
          nd.inputs[1]! == deqId &&
          nd.inputs[0]! != deqId &&
          (nd.inputs.size < 3 || nd.inputs[2]! != deqId)
        | none => false

/-- Polymorphic constant folding pass across `HighOp` and `CoreOp` (via `HasCoreOps Op`):
    - Folds `Transpose(const)` and `Reshape(const)`.
    - Folds scalar `elemUnary(const)` and `elemBinary(const, const)`.
    - Fallback-dequantizes `dequantize(const (.quant spec))` into `const (.fp16 bytes)`
      when any consumer is not a quant-capable `conv`. -/
def constFold {Op : Type} [HasCoreOps Op] (g : SSAGraph Op) : Except String (SSAGraph Op) := do
  let mut constMap : Array (ValueId × TensorType × ConstPayload) := #[]
  let mut outNodes : Array (IRNode Op) := Array.mkEmpty g.nodes.size

  let lookupConst := fun (cm : Array (ValueId × TensorType × ConstPayload)) (v : ValueId) =>
    cm.find? fun e => e.1 == v

  for nd in g.nodes do
    match HasCoreOps.asCore? nd.op, nd.inputs.toList, nd.outputs.toList with
    | some (.const payload), [], [(outId, outTy)] =>
      constMap := constMap.push (outId, outTy, payload)
      outNodes := outNodes.push nd

    | some (.reshape _), [inId], [(outId, outTy)] =>
      match lookupConst constMap inId with
      | some (_, _, payload) =>
        match payload with
        | .fp16 _ | .f32 _ =>
          constMap := constMap.push (outId, outTy, payload)
          outNodes := outNodes.push {
            nd with
            op := HasCoreOps.ofCore (.const payload)
            inputs := #[]
          }
        | .quant _ =>
          outNodes := outNodes.push nd
      | none =>
        outNodes := outNodes.push nd

    | some (.transpose perm), [inId], [(outId, outTy)] =>
      match lookupConst constMap inId with
      | some (_, inTy, .f32 vals) =>
        match transposeFloatArray inTy.shape perm vals with
        | some tVals =>
          let p := ConstPayload.f32 tVals
          constMap := constMap.push (outId, outTy, p)
          outNodes := outNodes.push { nd with op := HasCoreOps.ofCore (.const p), inputs := #[] }
        | none => outNodes := outNodes.push nd
      | some (_, inTy, .fp16 bs) =>
        match transposeFp16Bytes inTy.shape perm bs with
        | some tBs =>
          let p := ConstPayload.fp16 tBs
          constMap := constMap.push (outId, outTy, p)
          outNodes := outNodes.push { nd with op := HasCoreOps.ofCore (.const p), inputs := #[] }
        | none => outNodes := outNodes.push nd
      | _ =>
        outNodes := outNodes.push nd

    | some (.elemUnary uop), [inId], [(outId, outTy)] =>
      match lookupConst constMap inId with
      | some (_, inTy, payload) =>
        if inTy.numElems == 1 then
          match payload.toFloat64? with
          | some vals =>
            if vals.size == 1 then
              match evalScalarUnary uop vals[0]! with
              | some r =>
                let p := ConstPayload.f32 #[r]
                constMap := constMap.push (outId, outTy, p)
                outNodes := outNodes.push { nd with op := HasCoreOps.ofCore (.const p), inputs := #[] }
              | none => outNodes := outNodes.push nd
            else outNodes := outNodes.push nd
          | none => outNodes := outNodes.push nd
        else outNodes := outNodes.push nd
      | none => outNodes := outNodes.push nd

    | some (.elemBinary bop), [lhsId, rhsId], [(outId, outTy)] =>
      match lookupConst constMap lhsId, lookupConst constMap rhsId with
      | some (_, lTy, lPay), some (_, rTy, rPay) =>
        if lTy.numElems == 1 && rTy.numElems == 1 then
          match lPay.toFloat64?, rPay.toFloat64? with
          | some lv, some rv =>
            if lv.size == 1 && rv.size == 1 then
              match evalScalarBinary bop lv[0]! rv[0]! with
              | some r =>
                let p := ConstPayload.f32 #[r]
                constMap := constMap.push (outId, outTy, p)
                outNodes := outNodes.push { nd with op := HasCoreOps.ofCore (.const p), inputs := #[] }
              | none => outNodes := outNodes.push nd
            else outNodes := outNodes.push nd
          | _, _ => outNodes := outNodes.push nd
        else outNodes := outNodes.push nd
      | _, _ => outNodes := outNodes.push nd

    | some (.dequantize _), [qId], [(outId, outTy)] =>
      if allConsumersQuantCapableConv g outId then
        outNodes := outNodes.push nd
      else
        match lookupConst constMap qId with
        | some (_, qTy, .quant qSpec) =>
          if qTy.shape.size >= 1 then
            let cout := qTy.shape[0]!
            let total := qTy.numElems
            if cout > 0 && total % cout == 0 then
              let inner := total / cout
              let fp16Bytes ← dequantizeSpecToFp16 qSpec cout inner
              let p := ConstPayload.fp16 fp16Bytes
              let newTy := { outTy with dtype := .fp16 }
              constMap := constMap.push (outId, newTy, p)
              outNodes := outNodes.push {
                nd with
                op := HasCoreOps.ofCore (.const p)
                inputs := #[]
                outputs := #[(outId, newTy)]
              }
            else outNodes := outNodes.push nd
          else outNodes := outNodes.push nd
        | _ => outNodes := outNodes.push nd

    | _, _, _ =>
      outNodes := outNodes.push nd

  pure (dce { g with nodes := outNodes })

end Vane.IR
