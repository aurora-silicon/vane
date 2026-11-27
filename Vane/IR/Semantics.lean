import Vane.IR.Validate

namespace Vane.IR

-- ============================================================================
-- Polymorphic Mathematical Tensor Values (`TensorVal α`)
-- ============================================================================

/-- Product of dimensions in a logical shape (`shapeProd [] = 1` for scalars). -/
def shapeProd : List Nat → Nat
  | [] => 1
  | d :: ds => d * shapeProd ds

/-- Pure mathematical rank-N tensor over an abstract scalar type `α`. -/
structure TensorVal (α : Type) where
  shape : List Nat
  data : Fin (shapeProd shape) → α

-- ============================================================================
-- Exact Row-Major Coordinate <-> Flat Index Bijection
-- ============================================================================

/-- Convert row-major multi-index `coords` within `shape` into a flat linear index. -/
def flatIndex : List Nat → List Nat → Nat
  | [], _ => 0
  | _ :: _, [] => 0
  | _ :: ds, c :: cs => c * shapeProd ds + flatIndex ds cs

/-- Inverse of `flatIndex`: convert a flat linear index into row-major coordinates for `shape`. -/
def unflatten : List Nat → Nat → List Nat
  | [], _ => []
  | _ :: ds, idx =>
    let stride := shapeProd ds
    (idx / stride) :: unflatten ds (idx % stride)

-- ============================================================================
-- Exact Polymorphic Denotational Semantics for Structural View Ops
-- ============================================================================

/-- Output shape of `transpose(shape, perm)`. -/
def transposeShape (shape : List Nat) (perm : Array Nat) : List Nat :=
  perm.toList.map fun a => shape.getD a 0

/-- Exact element-wise denotational semantics of `transpose(perm, t)` over any `α`. -/
def evalTranspose {α : Type} (perm : Array Nat) (t : TensorVal α) : Option (TensorVal α) :=
  if isValidPerm perm && perm.size == t.shape.length && t.shape.all (· > 0) then
    let outShape := transposeShape t.shape perm
    let inv := invPerm perm
    let nIn := shapeProd t.shape
    if hn : 0 < nIn then
      some {
        shape := outShape
        data := fun outIdx =>
          let outCoords := unflatten outShape outIdx.val
          let inCoords := (List.range t.shape.length).map fun a =>
            outCoords.getD (inv.getD a 0) 0
          let fIdx := flatIndex t.shape inCoords
          if hlt : fIdx < nIn then
            t.data ⟨fIdx, hlt⟩
          else
            t.data ⟨0, hn⟩
      }
    else
      none
  else
    none

/-- Exact element-wise denotational semantics of `reshape(targetShape, t)` over any `α`. -/
def evalReshape {α : Type} (targetShape : Array Nat) (t : TensorVal α) : Option (TensorVal α) :=
  if h : shapeProd targetShape.toList = shapeProd t.shape then
    if targetShape.all (· > 0) then
      some {
        shape := targetShape.toList
        data := fun idx => t.data ⟨idx.val, h ▸ idx.isLt⟩
      }
    else
      none
  else
    none

/-- Output dimension along one axis of `slice(beg, en, step)`. -/
def sliceAxisDim (beg en st : Nat) : Nat :=
  if st == 0 || en <= beg then 0
  else (en - beg + st - 1) / st

/-- Check validity of slice parameters against logical `shape`. -/
def validSliceParams (shape : List Nat) (beginIdx endIdx step : Array Nat) : Bool :=
  beginIdx.size == shape.length &&
  endIdx.size == shape.length &&
  step.size == shape.length &&
  (List.range shape.length).all fun i =>
    let b := beginIdx.getD i 0
    let e := endIdx.getD i 0
    let s := step.getD i 0
    let d := shape.getD i 0
    s > 0 && b < e && e <= d

/-- Output shape of `slice(shape, beginIdx, endIdx, step)`. -/
def sliceShape (shape : List Nat) (beginIdx endIdx step : Array Nat) : List Nat :=
  (List.range shape.length).map fun i =>
    sliceAxisDim (beginIdx.getD i 0) (endIdx.getD i 0) (step.getD i 0)

/-- Exact element-wise denotational semantics of `slice(beginIdx, endIdx, step, t)` over any `α`. -/
def evalSlice {α : Type} (beginIdx endIdx step : Array Nat) (t : TensorVal α) : Option (TensorVal α) :=
  if validSliceParams t.shape beginIdx endIdx step && t.shape.all (· > 0) then
    let outShape := sliceShape t.shape beginIdx endIdx step
    let nIn := shapeProd t.shape
    if hn : 0 < nIn then
      some {
        shape := outShape
        data := fun outIdx =>
          let outCoords := unflatten outShape outIdx.val
          let inCoords := (List.range t.shape.length).map fun i =>
            beginIdx.getD i 0 + outCoords.getD i 0 * step.getD i 0
          let fIdx := flatIndex t.shape inCoords
          if hlt : fIdx < nIn then
            t.data ⟨fIdx, hlt⟩
          else
            t.data ⟨0, hn⟩
      }
    else
      none
  else
    none

/-- Locate which input tensor and local axis offset correspond to `coord` along `axis`. -/
def concatLookup {α : Type} (axis : Nat) : List (TensorVal α) → Nat → Option (TensorVal α × Nat)
  | [], _ => none
  | t :: ts, coord =>
    let d := t.shape.getD axis 0
    if coord < d then some (t, coord)
    else concatLookup axis ts (coord - d)

/-- Exact element-wise denotational semantics of `concat(axis, ts)` over any `α`. -/
def evalConcat {α : Type} (axis : Nat) (ts : List (TensorVal α)) : Option (TensorVal α) :=
  match ts with
  | [] => none
  | t0 :: _ =>
    let r := t0.shape.length
    if axis >= r || !t0.shape.all (· > 0) then none
    else
      let compat := ts.all fun t =>
        t.shape.length == r &&
        t.shape.all (· > 0) &&
        (List.range r).all fun i => i == axis || t.shape.getD i 0 == t0.shape.getD i 0
      if !compat then none
      else
        let sumAxis := ts.foldl (fun acc t => acc + t.shape.getD axis 0) 0
        let outShape := (List.range r).map fun i =>
          if i == axis then sumAxis else t0.shape.getD i 0
        let n0 := shapeProd t0.shape
        if hn0 : 0 < n0 then
          some {
            shape := outShape
            data := fun outIdx =>
              let outCoords := unflatten outShape outIdx.val
              let axCoord := outCoords.getD axis 0
              match concatLookup axis ts axCoord with
              | some (srcT, localAx) =>
                let inCoords := (List.range r).map fun i =>
                  if i == axis then localAx else outCoords.getD i 0
                let fIdx := flatIndex srcT.shape inCoords
                if hlt : fIdx < shapeProd srcT.shape then
                  srcT.data ⟨fIdx, hlt⟩
                else
                  t0.data ⟨0, hn0⟩
              | none => t0.data ⟨0, hn0⟩
          }
        else
          none

-- ============================================================================
-- Value Environments, OpSemantics Interface & Graph Evaluator
-- ============================================================================

/-- Pure SSA value environment mapping `ValueId` to optional `TensorVal α`. -/
abbrev Env (α : Type) := ValueId → Option (TensorVal α)

def Env.empty {α : Type} : Env α := fun _ => none

def Env.set {α : Type} (env : Env α) (v : ValueId) (t : TensorVal α) : Env α :=
  fun x => if x == v then some t else env x

/-- Look up a list of SSA `ValueId`s in `env`, failing if any value is unbound. -/
def lookupAll {α : Type} (env : Env α) : List ValueId → Option (List (TensorVal α))
  | [] => some []
  | v :: vs =>
    match env v, lookupAll env vs with
    | some t, some ts => some (t :: ts)
    | _, _ => none

/-- Bind node output `(ValueId × TensorType)` pairs to evaluated `TensorVal α` results. -/
def bindOutputs {α : Type} (env : Env α) : List (ValueId × TensorType) → List (TensorVal α) → Env α
  | [], _ => env
  | _, [] => env
  | (v, _) :: os, t :: ts => bindOutputs (Env.set env v t) os ts

/-- Polymorphic op semantics interface for any dialect `Op` over abstract scalar type `α`. -/
structure OpSemantics (Op : Type) (α : Type) where
  evalOp : Op → List (TensorVal α) → List TensorType → Option (List (TensorVal α))
  evalOp_len : ∀ op ins outTys outs, evalOp op ins outTys = some outs → outs.length = outTys.length

/-- Lawfulness predicate ensuring a dialect `Op` with `HasCoreOps Op` interprets core
    structural view ops (`transpose`, `reshape`, `slice`, `concat`) identically to the
    exact polymorphic definitions above. -/
structure CoreViewLawful {Op α : Type} [HasCoreOps Op] (sem : OpSemantics Op α) : Prop where
  transpose_law : ∀ (op : Op) (perm : Array Nat) (t : TensorVal α) (outTy : TensorType),
    HasCoreOps.asCore? op = some (.transpose perm) →
    sem.evalOp op [t] [outTy] = (evalTranspose perm t).map (fun r => [r])
  reshape_law : ∀ (op : Op) (targetShape : Array Nat) (t : TensorVal α) (outTy : TensorType),
    HasCoreOps.asCore? op = some (.reshape targetShape) →
    sem.evalOp op [t] [outTy] = (evalReshape targetShape t).map (fun r => [r])
  slice_law : ∀ (op : Op) (beg en st : Array Nat) (t : TensorVal α) (outTy : TensorType),
    HasCoreOps.asCore? op = some (.slice beg en st) →
    sem.evalOp op [t] [outTy] = (evalSlice beg en st t).map (fun r => [r])
  concat_law : ∀ (op : Op) (axis : Nat) (ts : List (TensorVal α)) (outTy : TensorType),
    HasCoreOps.asCore? op = some (.concat axis) →
    sem.evalOp op ts [outTy] = (evalConcat axis ts).map (fun r => [r])

/-- Check that every output value's shape matches its declared `TensorType.shape`. -/
def shapesMatch {α : Type} : List (ValueId × TensorType) → List (TensorVal α) → Bool
  | [], [] => true
  | (_, ty) :: os, t :: ts => (t.shape == ty.shape.toList) && shapesMatch os ts
  | _, _ => false

/-- Check that every graph input in `env` is present and matches its declared `TensorType.shape`. -/
def inputShapesMatch {α : Type} (env : Env α) : List (ValueId × String × TensorType) → Bool
  | [] => true
  | (v, _, ty) :: rest =>
    match env v with
    | some t => (t.shape == ty.shape.toList) && inputShapesMatch env rest
    | none => false

/-- Evaluate a single SSA node in `env`, enforcing output arity and declared shape typing. -/
def evalNode {Op α : Type} (sem : OpSemantics Op α) (env : Env α) (nd : IRNode Op) : Option (Env α) :=
  match lookupAll env nd.inputs.toList with
  | none => none
  | some inVals =>
    let outTys := nd.outputs.toList.map Prod.snd
    match sem.evalOp nd.op inVals outTys with
    | none => none
    | some outVals =>
      if shapesMatch nd.outputs.toList outVals then
        some (bindOutputs env nd.outputs.toList outVals)
      else
        none

/-- Sequentially evaluate a list of SSA nodes in topological order. -/
def evalNodes {Op α : Type} (sem : OpSemantics Op α) (env : Env α) : List (IRNode Op) → Option (Env α)
  | [] => some env
  | nd :: rest =>
    match evalNode sem env nd with
    | none => none
    | some env' => evalNodes sem env' rest

/-- Denotational graph evaluator: checks input shapes against `g.inputs`, then runs
    `g.nodes` in topological order starting from `env`. -/
def evalGraph {Op α : Type} (sem : OpSemantics Op α) (g : SSAGraph Op) (env : Env α) : Option (Env α) :=
  if inputShapesMatch env g.inputs.toList then
    evalNodes sem env g.nodes.toList
  else
    none

/-- Evaluate `g` and project out its `g.outputs` tensor list. -/
def evalGraphOutputs {Op α : Type} (sem : OpSemantics Op α) (g : SSAGraph Op) (env : Env α) :
    Option (List (TensorVal α)) :=
  match evalGraph sem g env with
  | none => none
  | some finalEnv => lookupAll finalEnv g.outputs.toList

/-- Forward semantic refinement: whenever `g` evaluates to `outs` on `env`,
    `g'` also evaluates to the exact same `outs` on `env`.
    (For total/defined inputs, forward refinement implies output-observable equality.) -/
def SemRefines {Op α : Type} (sem : OpSemantics Op α) (g g' : SSAGraph Op) : Prop :=
  ∀ env outs, evalGraphOutputs sem g env = some outs → evalGraphOutputs sem g' env = some outs

/-- Exact output-observable semantic equivalence on every input environment `env`. -/
def SemEquiv {Op α : Type} (sem : OpSemantics Op α) (g g' : SSAGraph Op) : Prop :=
  ∀ env, evalGraphOutputs sem g' env = evalGraphOutputs sem g env

/-- Environment typing predicate: every SSA value `v` bound in `env` has the shape declared in `g`. -/
def EnvTyped {Op α : Type} (g : SSAGraph Op) (env : Env α) : Prop :=
  ∀ (v : ValueId) (ty : TensorType) (t : TensorVal α),
    g.valueType? v = some ty → env v = some t → t.shape = ty.shape.toList

/-- Canonical `evalCore` semantics over `CoreOp` for an abstract scalar type `α` equipped with
    a backend oracle for non-view compute ops. -/
def mkCoreSemantics {α : Type}
    (evalCompute : CoreOp → List (TensorVal α) → List TensorType → Option (List (TensorVal α)))
    (hComputeLen : ∀ op ins outTys outs, evalCompute op ins outTys = some outs → outs.length = outTys.length) :
    OpSemantics CoreOp α where
  evalOp op ins outTys :=
    match op, ins, outTys with
    | .transpose perm, [t], [_] => (evalTranspose perm t).map (fun r => [r])
    | .reshape sh, [t], [_] => (evalReshape sh t).map (fun r => [r])
    | .slice b e s, [t], [_] => (evalSlice b e s t).map (fun r => [r])
    | .concat ax, ts, [_] => (evalConcat ax ts).map (fun r => [r])
    | _, _, _ => evalCompute op ins outTys
  evalOp_len op ins outTys outs h := by
    cases op with
    | transpose _ =>
      cases ins with
      | nil => exact hComputeLen _ _ _ _ h
      | cons _ ts =>
        cases ts with
        | nil =>
          cases outTys with
          | nil => exact hComputeLen _ _ _ _ h
          | cons _ tys =>
            cases tys with
            | nil =>
              simp only [Option.map_eq_some_iff] at h
              obtain ⟨r, _, rfl⟩ := h
              rfl
            | cons _ _ => exact hComputeLen _ _ _ _ h
        | cons _ _ => exact hComputeLen _ _ _ _ h
    | reshape _ =>
      cases ins with
      | nil => exact hComputeLen _ _ _ _ h
      | cons _ ts =>
        cases ts with
        | nil =>
          cases outTys with
          | nil => exact hComputeLen _ _ _ _ h
          | cons _ tys =>
            cases tys with
            | nil =>
              simp only [Option.map_eq_some_iff] at h
              obtain ⟨r, _, rfl⟩ := h
              rfl
            | cons _ _ => exact hComputeLen _ _ _ _ h
        | cons _ _ => exact hComputeLen _ _ _ _ h
    | slice _ _ _ =>
      cases ins with
      | nil => exact hComputeLen _ _ _ _ h
      | cons _ ts =>
        cases ts with
        | nil =>
          cases outTys with
          | nil => exact hComputeLen _ _ _ _ h
          | cons _ tys =>
            cases tys with
            | nil =>
              simp only [Option.map_eq_some_iff] at h
              obtain ⟨r, _, rfl⟩ := h
              rfl
            | cons _ _ => exact hComputeLen _ _ _ _ h
        | cons _ _ => exact hComputeLen _ _ _ _ h
    | concat _ =>
      cases outTys with
      | nil => exact hComputeLen _ _ _ _ h
      | cons _ tys =>
        cases tys with
        | nil =>
          simp only [Option.map_eq_some_iff] at h
          obtain ⟨r, _, rfl⟩ := h
          rfl
        | cons _ _ => exact hComputeLen _ _ _ _ h
    | const _ => exact hComputeLen _ _ _ _ h
    | dequantize _ => exact hComputeLen _ _ _ _ h
    | conv _ _ _ _ _ _ _ _ => exact hComputeLen _ _ _ _ h
    | matmul _ _ _ => exact hComputeLen _ _ _ _ h
    | elemUnary _ => exact hComputeLen _ _ _ _ h
    | elemBinary _ => exact hComputeLen _ _ _ _ h
    | lut _ => exact hComputeLen _ _ _ _ h
    | pool _ _ _ _ _ _ _ => exact hComputeLen _ _ _ _ h
    | reduce _ _ _ _ => exact hComputeLen _ _ _ _ h

/-- Convenience alias matching the spec name `evalCore`. -/
def evalCore {α : Type} (sem : OpSemantics CoreOp α) (g : SSAGraph CoreOp) (env : Env α) :
    Option (Env α) :=
  evalGraph sem g env

end Vane.IR
