import Vane.IR.Rewrite

namespace Vane.IR

-- ============================================================================
-- Pass 1: Polymorphic Multi-Output Dead Code Elimination (DCE)
-- ============================================================================

/-- Single right-to-left backwards reachability pass over a topologically ordered
    node list. Keeps every node producing at least one value in `needed` and adds
    all inputs of kept nodes to `needed`. Structurally total; supports arbitrary
    multi-output nodes and multi-output graphs. -/
def dceNodes {Op : Type} : List (IRNode Op) → List ValueId → List (IRNode Op) × List ValueId
  | [], needed => ([], needed)
  | nd :: rest, needed =>
    let (keptRest, neededAfter) := dceNodes rest needed
    if nd.outputs.toList.any (fun o => neededAfter.elem o.1) then
      (nd :: keptRest, nd.inputs.toList ++ neededAfter)
    else
      (keptRest, neededAfter)

/-- Polymorphic Dead Code Elimination over any `SSAGraph Op`.
    Preserves `WellFormed`, preserves `evalGraphOutputs` element-for-element,
    and preserves all graph inputs and multi-output roots. -/
def dce {Op : Type} (g : SSAGraph Op) : SSAGraph Op :=
  let (kept, _) := dceNodes g.nodes.toList g.outputs.toList
  { g with nodes := kept.toArray }

end Vane.IR
