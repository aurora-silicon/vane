import Vane.IR.Rewrite
import Vane.IR.Passes.DCE

namespace Vane.IR

open Vane.Float16
open Vane.Conv
open Vane.Lut

-- ============================================================================
-- Pass 6: CoreOp Pattern Fusion (SiLU, Conv+Bias, Conv+Act, MatMul+Tr+Scale)
-- ============================================================================

private def isDequantProducerCore (g : SSAGraph CoreOp) (v : ValueId) : Bool :=
  match g.producerOf? v with
  | some (_, nd, _) =>
    match nd.op with
    | .dequantize _ => true
    | _ => false
  | none => false

private def lookupConstCore (g : SSAGraph CoreOp) (v : ValueId) : Option (TensorType × ConstPayload) :=
  match g.producerOf? v with
  | some (_, nd, 0) =>
    match nd.op, nd.outputs[0]? with
    | .const p, some (_, ty) => some (ty, p)
    | _, _ => none
  | _ => none

/-- 1. Collapse single-consumer `Mul(x, Sigmoid(x))` / `Mul(Sigmoid(x), x)` → `lut (.lut .silu)`. -/
def fuseSiluCore (g : SSAGraph CoreOp) : SSAGraph CoreOp := Id.run do
  let mut cur := g
  let maxSteps := g.nodes.size + 1
  for _ in [:maxSteps] do
    let mut found? : Option (Nat × ValueId) := none
    for i in [:cur.nodes.size] do
      if found?.isNone then
        let nd := cur.nodes[i]!
        match nd.op, nd.inputs.toList, nd.outputs.toList with
        | .elemBinary .mul, [a, b], [(_, _)] =>
          let checkSide := fun (sigId xId : ValueId) =>
            if cur.consumerCount sigId == 1 then
              match cur.producerOf? sigId with
              | some (_, pnd, 0) =>
                pnd.op == .lut (.lut .sigmoid) && pnd.inputs == #[xId]
              | _ => false
            else false
          if checkSide b a then
            found? := some (i, a)
          else if checkSide a b then
            found? := some (i, b)
        | _, _, _ => pure ()
    match found? with
    | none => break
    | some (mulIdx, xId) =>
      let nd := cur.nodes[mulIdx]!
      let nodes' := cur.nodes.set! mulIdx { nd with op := .lut (.lut .silu), inputs := #[xId] }
      cur := dce { cur with nodes := nodes' }
  return cur

def isChannelBiasShapeCore (dims convOutShape : Array Nat) : Bool :=
  if convOutShape.size == 2 then
    let cout := convOutShape[1]!
    if cout == 0 then false
    else dims == #[cout] || dims == #[1, cout] || (cout == 1 && dims.isEmpty)
  else if convOutShape.size == 4 then
    let cout := convOutShape[1]!
    if cout == 0 then false
    else
      dims == #[1, cout, 1, 1] || dims == #[cout, 1, 1] ||
      (cout == 1 && (dims.isEmpty || dims == #[1] || dims == #[1, 1]))
  else
    false

/-- 2. Fold single-consumer bias-free `Conv(x, w) + Add(const channel bias)` into `Conv(x, w, bias)`.
       Rejects non-channel spatial constants (e.g. `[1, 1, 2, 2]`) and 1D/2D W-broadcast constants
       on 4D Conv (e.g. `[Cout]` or `[1, Cout]`) so they are never misfolded. -/
def fuseConvAddBiasCore (g : SSAGraph CoreOp) : SSAGraph CoreOp := Id.run do
  let mut cur := g
  let maxSteps := g.nodes.size + 1
  for _ in [:maxSteps] do
    let mut found? : Option (Nat × Nat × ValueId) := none
    for i in [:cur.nodes.size] do
      if found?.isNone then
        let addNd := cur.nodes[i]!
        match addNd.op, addNd.inputs.toList, addNd.outputs.toList with
        | .elemBinary .add, [a, b], [(_, addOutTy)] =>
          let checkSide := fun (convOutId biasId : ValueId) =>
            if cur.consumerCount convOutId == 1 then
              match cur.producerOf? convOutId, lookupConstCore cur biasId with
              | some (convIdx, convNd, 0), some (bTy, bPay) =>
                match convNd.op, convNd.outputs[0]? with
                | .conv _ _ _ _ _ _ false .none, some (_, convOutTy) =>
                  if convOutTy == addOutTy && isChannelBiasShapeCore bTy.shape convOutTy.shape && bPay.toFloat64?.isSome then
                    some (convIdx, i, biasId)
                  else none
                | _, _ => none
              | _, _ => none
            else none
          match checkSide a b with
          | some hit => found? := some hit
          | none => found? := checkSide b a
        | _, _, _ => pure ()
    match found? with
    | none => break
    | some (convIdx, addIdx, biasId) =>
      let convNd := cur.nodes[convIdx]!
      let addNd := cur.nodes[addIdx]!
      let convOutId := convNd.outputs[0]!.1
      let addOutId := addNd.outputs[0]!.1
      let newConvOp := match convNd.op with
        | .conv kh kw sh sw pads groups _ act => CoreOp.conv kh kw sh sw pads groups true act
        | other => other
      -- Ensure bias constant node appears BEFORE convIdx in topological order
      let mut newNodes : Array (IRNode CoreOp) := Array.mkEmpty cur.nodes.size
      let biasProd? := cur.producerOf? biasId
      let needHoistBias := match biasProd? with
        | some (bIdx, _, _) => bIdx > convIdx
        | none => false
      let biasNode? := biasProd?.map fun p => p.2.1
      for idx in [:cur.nodes.size] do
        if idx == convIdx then
          if needHoistBias then
            if let some bn := biasNode? then
              newNodes := newNodes.push bn
          newNodes := newNodes.push {
            convNd with
            op := newConvOp
            inputs := #[convNd.inputs[0]!, convNd.inputs[1]!, biasId]
          }
        else if idx == addIdx then
          pure ()
        else if needHoistBias && (match biasProd? with | some (bIdx, _, _) => idx == bIdx | none => false) then
          pure ()
        else
          newNodes := newNodes.push cur.nodes[idx]!
      cur := dce (({ cur with nodes := newNodes }).replaceUses addOutId convOutId)
  return cur

/-- 3. Fuse single-consumer `Conv + ReLU/LUT` epilogue (respecting M8A rule: do NOT fuse `.lut`
       onto a `dequantize`-weighted conv, only `.relu`). -/
def fuseConvActCore (g : SSAGraph CoreOp) : SSAGraph CoreOp × Nat := Id.run do
  let mut cur := g
  let mut fused : Nat := 0
  let maxSteps := g.nodes.size + 1
  for _ in [:maxSteps] do
    let mut found? : Option (Nat × Nat × ConvActivation) := none
    for i in [:cur.nodes.size] do
      if found?.isNone then
        let actNd := cur.nodes[i]!
        match actNd.op, actNd.inputs.toList, actNd.outputs.toList with
        | .lut neAct, [inId], [(_, _)] =>
          let convAct? : Option ConvActivation := match neAct with
            | .relu => some .relu
            | .lut lspec => some (.lut lspec)
            | .copy => none
          if let some convAct := convAct? then
            if cur.consumerCount inId == 1 then
              match cur.producerOf? inId with
              | some (convIdx, convNd, 0) =>
                match convNd.op with
                | .conv _ _ _ _ _ _ _ .none =>
                  let isQuantW := convNd.inputs.size >= 2 && isDequantProducerCore cur convNd.inputs[1]!
                  if !isQuantW || convAct == .relu then
                    found? := some (convIdx, i, convAct)
                | _ => pure ()
              | _ => pure ()
        | _, _, _ => pure ()
    match found? with
    | none => break
    | some (convIdx, actIdx, convAct) =>
      let convNd := cur.nodes[convIdx]!
      let actNd := cur.nodes[actIdx]!
      let convOutId := convNd.outputs[0]!.1
      let actOutId := actNd.outputs[0]!.1
      let newConvOp := match convNd.op with
        | .conv kh kw sh sw pads groups hasBias _ => CoreOp.conv kh kw sh sw pads groups hasBias convAct
        | other => other
      let mut newNodes : Array (IRNode CoreOp) := Array.mkEmpty cur.nodes.size
      for idx in [:cur.nodes.size] do
        if idx == convIdx then
          newNodes := newNodes.push { convNd with op := newConvOp }
        else if idx == actIdx then
          pure ()
        else
          newNodes := newNodes.push cur.nodes[idx]!
      cur := dce (({ cur with nodes := newNodes }).replaceUses actOutId convOutId)
      fused := fused + 1
  return (cur, fused)

/-- 4. Fuse single-consumer validated `Transpose` inputs and post-`MatMul` scalar `Mul`/`Div`
       into `CoreOp.matmul`. -/
def fuseMatmulCore (g : SSAGraph CoreOp) : Except String (SSAGraph CoreOp × Nat) := do
  let mut cur := g
  let mut folded : Nat := 0

  -- 4a. Fold single-consumer Transpose on A and/or B into `CoreOp.matmul`.
  let maxSteps := cur.nodes.size + 1
  for _ in [:maxSteps] do
    let mut changed := false
    for i in [:cur.nodes.size] do
      if !changed then
        let nd := cur.nodes[i]!
        match nd.op, nd.inputs.toList with
        | .matmul trA trB scaleBits, [aId, bId] =>
          let aTy? := cur.valueType? aId
          let bTy? := cur.valueType? bId
          match aTy?, bTy? with
          | some aTy, some bTy =>
            let is4DTrue := aTy.shape.size == 4 && bTy.shape.size == 4 &&
                            (aTy.shape[2]! > 1 || bTy.shape[2]! > 1)
            let checkTranspose := fun (side : String) (alreadyTr : Bool) (inpId : ValueId) (otherTy : TensorType) => do
              if alreadyTr || cur.consumerCount inpId != 1 then
                pure none
              else
                match cur.producerOf? inpId with
                | some (_, trNd, 0) =>
                  match trNd.op, trNd.inputs.toList with
                  | .transpose perm, [origInId] =>
                    if aTy.shape.size == 3 && bTy.shape.size == 3 then
                      if perm == #[0, 2, 1] then pure (some origInId)
                      else pure none
                    else if is4DTrue then
                      if perm == #[0, 1, 3, 2] then pure (some origInId)
                      else if perm == #[0, 3, 2, 1] then
                        -- Guard: reject mismatched [0,3,2,1] perm on true 4D MatMul when K mismatches
                        let myTy := (cur.valueType? inpId).get!
                        let kMismatch :=
                          if side == "LHS" then myTy.shape[3]! != otherTy.shape[2]!
                          else otherTy.shape[3]! != myTy.shape[2]!
                        if kMismatch then
                          throw s!"node '{nd.name}' (MatMul): 4D MatMul {side} folded Transpose perm {perm.toList} != required [0, 1, 3, 2]"
                        else
                          pure none
                      else pure none
                    else if aTy.shape.size == 4 && bTy.shape.size == 4 then
                      let isMultiHeadStacked := !trA && !trB && aTy.shape[3]! > 0 &&
                        bTy.shape[1]! > aTy.shape[3]! && bTy.shape[1]! % aTy.shape[3]! == 0
                      if !isMultiHeadStacked && perm == #[0, 3, 2, 1] then pure (some origInId)
                      else pure none
                    else
                      pure none
                  | _, _ => pure none
                | _ => pure none

            if let some newA ← checkTranspose "LHS" trA aId bTy then
              let nd' := { nd with op := .matmul true trB scaleBits, inputs := #[newA, bId] }
              cur := dce { cur with nodes := cur.nodes.set! i nd' }
              folded := folded + 1
              changed := true
            else if let some newB ← checkTranspose "RHS" trB bId aTy then
              let nd' := { nd with op := .matmul trA true scaleBits, inputs := #[aId, newB] }
              cur := dce { cur with nodes := cur.nodes.set! i nd' }
              folded := folded + 1
              changed := true
          | _, _ => pure ()
        | _, _ => pure ()
    if !changed then break

  -- 4b. Fold single-consumer post-MatMul scalar Mul / Div into `matmul.scaleBits`.
  let computeFoldedScale := fun (scName : String) (oldScaleBits : UInt16) (scVal : Float) (isDiv : Bool) => do
    if isDiv && scVal == 0.0 then
      throw s!"node '{scName}' (MatMul+Div): scalar division by zero"
    let oldScale := f16ToFloat oldScaleBits
    let newScale := if isDiv then oldScale / scVal else oldScale * scVal
    let newBits := floatToF16CanonicalZero newScale
    if !newScale.isNaN && !newScale.isInf && f16IsFinite newBits && (newBits &&& 0x7fff) != 0 then
      pure (some newBits)
    else
      pure none

  for _ in [:cur.nodes.size + 1] do
    let mut found? : Option (Nat × Nat × UInt16) := none
    for i in [:cur.nodes.size] do
      if found?.isNone then
        let nd := cur.nodes[i]!
        match nd.op, nd.inputs.toList, nd.outputs.toList with
        | .elemUnary (.scalarMul scBits), [mmId], [(_, scOutTy)] =>
          if cur.consumerCount mmId == 1 then
            if let some (mmIdx, mmNd, 0) := cur.producerOf? mmId then
              if let (.matmul _ _ oldBits, some (_, mmOutTy)) := (mmNd.op, mmNd.outputs[0]?) then
                if mmOutTy == scOutTy then
                  if let some newBits ← computeFoldedScale nd.name oldBits (f16ToFloat scBits) false then
                    found? := some (mmIdx, i, newBits)
        | .elemUnary (.scalarDiv scBits), [mmId], [(_, scOutTy)] =>
          if cur.consumerCount mmId == 1 then
            if let some (mmIdx, mmNd, 0) := cur.producerOf? mmId then
              if let (.matmul _ _ oldBits, some (_, mmOutTy)) := (mmNd.op, mmNd.outputs[0]?) then
                if mmOutTy == scOutTy then
                  if let some newBits ← computeFoldedScale nd.name oldBits (f16ToFloat scBits) true then
                    found? := some (mmIdx, i, newBits)
        | .elemBinary .div, [mmId, cId], [(_, scOutTy)] =>
          if cur.consumerCount mmId == 1 then
            match cur.producerOf? mmId, lookupConstCore cur cId with
            | some (mmIdx, mmNd, 0), some (cTy, cPay) =>
              if cTy.numElems == 1 then
                if let (.matmul _ _ oldBits, some (_, mmOutTy), some vs) := (mmNd.op, mmNd.outputs[0]?, cPay.toFloat64?) then
                  if mmOutTy == scOutTy && vs.size == 1 then
                    if let some newBits ← computeFoldedScale nd.name oldBits vs[0]! true then
                      found? := some (mmIdx, i, newBits)
            | _, _ => pure ()
        | .elemBinary .mul, [aId, bId], [(_, scOutTy)] =>
          let checkSide := fun (mmId cId : ValueId) => do
            if cur.consumerCount mmId == 1 then
              match cur.producerOf? mmId, lookupConstCore cur cId with
              | some (mmIdx, mmNd, 0), some (cTy, cPay) =>
                if cTy.numElems == 1 then
                  match mmNd.op, mmNd.outputs[0]?, cPay.toFloat64? with
                  | .matmul _ _ oldBits, some (_, mmOutTy), some vs =>
                    if mmOutTy == scOutTy && vs.size == 1 then
                      match ← computeFoldedScale nd.name oldBits vs[0]! false with
                      | some newBits => pure (some (mmIdx, i, newBits))
                      | none => pure none
                    else pure none
                  | _, _, _ => pure none
                else pure none
              | _, _ => pure none
            else pure none
          match ← checkSide aId bId with
          | some hit => found? := some hit
          | none => found? := (← checkSide bId aId)
        | _, _, _ => pure ()

    match found? with
    | none => break
    | some (mmIdx, scIdx, newBits) =>
      let mmNd := cur.nodes[mmIdx]!
      let scNd := cur.nodes[scIdx]!
      match mmNd.op with
      | .matmul trA trB _ =>
        let mmOutId := mmNd.outputs[0]!.1
        let scOutId := scNd.outputs[0]!.1
        let mut newNodes : Array (IRNode CoreOp) := Array.mkEmpty cur.nodes.size
        for idx in [:cur.nodes.size] do
          if idx == mmIdx then
            newNodes := newNodes.push { mmNd with op := .matmul trA trB newBits }
          else if idx == scIdx then
            pure ()
          else
            newNodes := newNodes.push cur.nodes[idx]!
        cur := dce (({ cur with nodes := newNodes }).replaceUses scOutId mmOutId)
        folded := folded + 1
      | _ => break

  pure (cur, folded)

/-- Run all CoreOp fusions: SiLU collapse, Conv+Bias fold, Conv+ReLU/LUT fusion,
    and MatMul+Transpose+Scale fusion. -/
def coreFuse (g : SSAGraph CoreOp) : Except String (SSAGraph CoreOp × Nat × Nat) := do
  let g1 := fuseSiluCore g
  let g2 := fuseConvAddBiasCore g1
  let (g3, actCnt) := fuseConvActCore g2
  let (g4, mmCnt) ← fuseMatmulCore g3
  pure (g4, actCnt, mmCnt)

end Vane.IR
