import Vane.IR.Graph

namespace Vane.IR

-- ============================================================================
-- Prop-Level SSA Well-Formedness & Executable Validator
-- ============================================================================

/-- Prop-level structural well-formedness for any `SSAGraph Op`:
    1. Graph inputs occupy dense sequential IDs `0 .. inputs.size - 1`.
    2. All input tensor shapes have strictly positive dimensions (`shapeProd > 0`).
    3. All defined `ValueId`s (graph inputs + all node outputs) are pairwise distinct.
    4. Strict def-before-use acyclicity: every input of node `i` is defined in
       `inputs` or in an earlier node `< i`.
    5. All node output tensor shapes have strictly positive dimensions.
    6. Every graph output is a defined `ValueId`. -/
structure WellFormed {Op : Type} (g : SSAGraph Op) : Prop where
  inputsDense : ∀ i (hi : i < g.inputs.size), g.inputs[i].1 = i
  inputsPositive : ∀ i (hi : i < g.inputs.size), ∀ d ∈ g.inputs[i].2.2.shape.toList, d > 0
  uniqueIds : g.allDefinedIds.Nodup
  defBeforeUse : ∀ i (hi : i < g.nodes.size), ∀ v ∈ g.nodes[i].inputs.toList,
    v ∈ g.allDefinedIdsBefore i
  nodeOutputsPositive : ∀ i (hi : i < g.nodes.size), ∀ o ∈ g.nodes[i].outputs.toList,
    ∀ d ∈ o.2.shape.toList, d > 0
  outputsValid : ∀ v ∈ g.outputs.toList, v ∈ g.allDefinedIds

-- ============================================================================
-- Pure Structural Helper Checks (Designed for Kernel-Clean Soundness Proof)
-- ============================================================================

/-- Check that every element of `xs` satisfies `p`, returning an error on the first failure. -/
def checkList {α : Type} (p : α → Bool) (err : α → String) : List α → Except String Unit
  | [] => .ok ()
  | x :: xs =>
    if p x then checkList p err xs
    else .error (err x)

/-- Check `f i` for every `i` from `lo` up to `hi - 1` (structurally total on `hi - lo`). -/
def checkIdxRange (lo hi : Nat) (f : Nat → Except String Unit) : Except String Unit :=
  if _ : lo < hi then
    match f lo with
    | .ok () => checkIdxRange (lo + 1) hi f
    | .error e => .error e
  else
    .ok ()
termination_by hi - lo

/-- Check that graph inputs occupy dense IDs `0 .. g.inputs.size - 1` with positive dims. -/
def validateGraphInputs {Op : Type} (g : SSAGraph Op) : Except String Unit :=
  checkIdxRange 0 g.inputs.size fun i =>
    if h : i < g.inputs.size then
      let entry := g.inputs[i]
      if entry.1 != i then
        .error s!"graph input[{i}] '{entry.2.1}' has ValueId {entry.1} != expected dense id {i}"
      else if !entry.2.2.hasPositiveDims then
        .error s!"graph input[{i}] '{entry.2.1}' has non-positive dimension in shape {entry.2.2.shape.toList}"
      else
        .ok ()
    else
      .ok ()

/-- Check that all defined ValueIds in `ids` are pairwise distinct. -/
def validateUniqueIds (ids : List ValueId) : Except String Unit :=
  match ids with
  | [] => .ok ()
  | x :: xs =>
    if xs.elem x then
      .error s!"duplicate SSA ValueId %{x} in graph"
    else
      validateUniqueIds xs

/-- Check strict def-before-use acyclicity and positive output shapes for all nodes. -/
def validateGraphNodes {Op : Type} (g : SSAGraph Op) : Except String Unit :=
  checkIdxRange 0 g.nodes.size fun i =>
    if h : i < g.nodes.size then
      let nd := g.nodes[i]
      let avail := g.allDefinedIdsBefore i
      match checkList (fun v => avail.elem v)
        (fun v => s!"node[{i}] '{nd.name}' uses SSA value %{v} which is not defined before node {i}")
        nd.inputs.toList with
      | .error e => .error e
      | .ok () =>
        checkList (fun o => o.2.hasPositiveDims)
          (fun o => s!"node[{i}] '{nd.name}' output %{o.1} has non-positive dimension in shape {o.2.shape.toList}")
          nd.outputs.toList
    else
      .ok ()

/-- Check that every graph output refers to a defined ValueId in `g`. -/
def validateGraphOutputs {Op : Type} (g : SSAGraph Op) : Except String Unit :=
  let defined := g.allDefinedIds
  checkList (fun v => defined.elem v)
    (fun v => s!"graph output %{v} is not defined in graph")
    g.outputs.toList

/-- Executable static validator for any `SSAGraph Op`.
    Soundness (`validateGraph g = .ok () → WellFormed g`) is proved kernel-clean in `Vane.Proofs`. -/
def validateGraph {Op : Type} (g : SSAGraph Op) : Except String Unit := do
  validateGraphInputs g
  validateUniqueIds g.allDefinedIds
  validateGraphNodes g
  validateGraphOutputs g

end Vane.IR
