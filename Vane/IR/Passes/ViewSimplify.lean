import Vane.IR.Rewrite
import Vane.IR.Passes.DCE

namespace Vane.IR

-- ============================================================================
-- Pass 2: Polymorphic ViewSimplify (via HasCoreOps + applyRewrites)
-- ============================================================================

/-- Pure list-index search: find the first index `i` (offset by `idx`) where `f i x = some y`. -/
def findFirstNodeMatch {Op β : Type} (f : Nat → IRNode Op → Option β) (idx : Nat) :
    List (IRNode Op) → Option β
  | [] => none
  | nd :: rest =>
    match f idx nd with
    | some y => some y
    | none => findFirstNodeMatch f (idx + 1) rest

/-- Pure lookup of the first node index `< limit` that produces `targetId` as its sole output. -/
def findSoleProducerBefore {Op : Type} (nodes : List (IRNode Op)) (limit : Nat)
    (targetId : ValueId) : Option (IRNode Op) :=
  (nodes.take limit).find? fun nd =>
    match nd.outputs.toList with
    | [(vid, _)] => vid == targetId
    | _ => false

/-- Eliminate a single-input, single-output identity node at index `idx` by aliasing
    every use of `outId` to `inId` and removing node `idx`. -/
def eliminateUnaryIdentityNode {Op : Type} (g : SSAGraph Op) (idx : Nat)
    (inId outId : ValueId) : SSAGraph Op :=
  let subst := fun (v : ValueId) => if v == outId then inId else v
  let nodes' := (g.nodes.toList.eraseIdx idx).map fun nd =>
    { nd with inputs := nd.inputs.map subst }
  { g with
    nodes := nodes'.toArray
    outputs := g.outputs.map subst }

/-- Rewrite a single-input node at index `idx` in-place to use `newOp` and `newInId`,
    preserving its output ValueId and TensorType. -/
def rewriteUnaryNodeInPlace {Op : Type} (g : SSAGraph Op) (idx : Nat)
    (newOp : Op) (newInId : ValueId) : SSAGraph Op :=
  { g with
    nodes := g.nodes.modify idx fun nd => { nd with op := newOp, inputs := #[newInId] } }

/-- Rule 1: Identity `Transpose(x, [0, 1, ..., r-1])` (matching input rank `r`) → alias `x`. -/
def ruleTransposeId {Op : Type} [HasCoreOps Op] : RewriteRule Op where
  name := "view.transpose_id"
  apply g :=
    findFirstNodeMatch (fun idx nd =>
      match HasCoreOps.asCore? nd.op, nd.inputs.toList, nd.outputs.toList with
      | some (.transpose perm), [inId], [(outId, _)] =>
        match g.valueType? inId with
        | some inTy =>
          if isIdentityPerm perm && perm.size == inTy.shape.size then
            some (eliminateUnaryIdentityNode g idx inId outId)
          else none
        | none => none
      | _, _, _ => none) 0 g.nodes.toList

/-- Rule 2: Consecutive `Transpose(Transpose(x, p1), p2)` → `Transpose(x, composePerm p1 p2)`. -/
def ruleTransposeCompose {Op : Type} [HasCoreOps Op] : RewriteRule Op where
  name := "view.transpose_compose"
  apply g :=
    findFirstNodeMatch (fun idx nd =>
      match HasCoreOps.asCore? nd.op, nd.inputs.toList, nd.outputs.toList with
      | some (.transpose p2), [midId], [(_, _)] =>
        match findSoleProducerBefore g.nodes.toList idx midId with
        | some pnd =>
          match HasCoreOps.asCore? pnd.op, pnd.inputs.toList with
          | some (.transpose p1), [xId] =>
            if isValidPerm p1 && isValidPerm p2 && p1.size == p2.size then
              some (rewriteUnaryNodeInPlace g idx (HasCoreOps.ofCore (.transpose (composePerm p1 p2))) xId)
            else none
          | _, _ => none
        | none => none
      | _, _, _ => none) 0 g.nodes.toList

/-- Rule 3: Identity `Reshape(x, inShape)` → alias `x`. -/
def ruleReshapeId {Op : Type} [HasCoreOps Op] : RewriteRule Op where
  name := "view.reshape_id"
  apply g :=
    findFirstNodeMatch (fun idx nd =>
      match HasCoreOps.asCore? nd.op, nd.inputs.toList, nd.outputs.toList with
      | some (.reshape targetShape), [inId], [(outId, _)] =>
        match g.valueType? inId with
        | some inTy =>
          if targetShape == inTy.shape then
            some (eliminateUnaryIdentityNode g idx inId outId)
          else none
        | none => none
      | _, _, _ => none) 0 g.nodes.toList

/-- Rule 4: Consecutive `Reshape(Reshape(x, s1), s2)` → `Reshape(x, s2)`. -/
def ruleReshapeCompose {Op : Type} [HasCoreOps Op] : RewriteRule Op where
  name := "view.reshape_compose"
  apply g :=
    findFirstNodeMatch (fun idx nd =>
      match HasCoreOps.asCore? nd.op, nd.inputs.toList, nd.outputs.toList with
      | some (.reshape s2), [midId], [(_, _)] =>
        match findSoleProducerBefore g.nodes.toList idx midId with
        | some pnd =>
          match HasCoreOps.asCore? pnd.op, pnd.inputs.toList with
          | some (.reshape _), [xId] =>
            some (rewriteUnaryNodeInPlace g idx (HasCoreOps.ofCore (.reshape s2)) xId)
          | _, _ => none
        | none => none
      | _, _, _ => none) 0 g.nodes.toList

/-- Rule 5: Full-span identity `Slice(x, [0..0], inShape, [1..1])` → alias `x`. -/
def ruleSliceId {Op : Type} [HasCoreOps Op] : RewriteRule Op where
  name := "view.slice_id"
  apply g :=
    findFirstNodeMatch (fun idx nd =>
      match HasCoreOps.asCore? nd.op, nd.inputs.toList, nd.outputs.toList with
      | some (.slice beg en st), [inId], [(outId, _)] =>
        match g.valueType? inId with
        | some inTy =>
          let r := inTy.shape.size
          if beg == Array.replicate r 0 && en == inTy.shape && st == Array.replicate r 1 then
            some (eliminateUnaryIdentityNode g idx inId outId)
          else none
        | none => none
      | _, _, _ => none) 0 g.nodes.toList

/-- The complete list of formally verified `ViewSimplify` peephole rules. -/
def viewSimplifyRules {Op : Type} [HasCoreOps Op] : List (RewriteRule Op) :=
  [ruleTransposeId, ruleTransposeCompose, ruleReshapeId, ruleReshapeCompose, ruleSliceId]

/-- Run fuel-bounded `ViewSimplify` rewrite rules followed by `DCE` to drop any
    orphaned intermediate view nodes. -/
def viewSimplify {Op : Type} [HasCoreOps Op] (g : SSAGraph Op) (fuel : Nat := 0) : SSAGraph Op :=
  let effFuel := if fuel == 0 then 4 * g.nodes.size + 16 else fuel
  dce (applyRewrites effFuel viewSimplifyRules g)

end Vane.IR
