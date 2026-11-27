import Vane.IR.Ops

namespace Vane.IR

open Vane.TaskStream

-- ============================================================================
-- Parameterized Multi-Input / Multi-Output SSA Graph Container
-- ============================================================================

/-- One typed multi-input, multi-output SSA node parameterized over dialect `Op`. -/
structure IRNode (Op : Type) where
  name : String := ""
  op : Op
  inputs : Array ValueId
  outputs : Array (ValueId × TensorType)
  deriving BEq, Inhabited

namespace IRNode

def primaryOutput? {Op : Type} (nd : IRNode Op) : Option (ValueId × TensorType) :=
  nd.outputs[0]?

def outputIds {Op : Type} (nd : IRNode Op) : List ValueId :=
  nd.outputs.toList.map Prod.fst

end IRNode

/-- Parameterized typed SSA graph container shared across dialects (`HighOp`, `CoreOp`,
    and future `AneOp`). Supports arbitrary multi-input and multi-output graphs. -/
structure SSAGraph (Op : Type) where
  target : Target
  inputs : Array (ValueId × String × TensorType)
  nodes : Array (IRNode Op)
  outputs : Array ValueId
  deriving BEq

instance {Op : Type} : Inhabited (SSAGraph Op) :=
  ⟨{ target := .h17, inputs := #[], nodes := #[], outputs := #[] }⟩

namespace SSAGraph

/-- List of all graph input `ValueId`s. -/
def inputIds {Op : Type} (g : SSAGraph Op) : List ValueId :=
  g.inputs.toList.map Prod.fst

/-- All `ValueId`s defined by `g.inputs` and the first `k` nodes of `g.nodes`. -/
def allDefinedIdsBefore {Op : Type} (g : SSAGraph Op) (k : Nat) : List ValueId :=
  g.inputIds ++ (g.nodes.toList.take k).flatMap IRNode.outputIds

/-- All `ValueId`s defined anywhere in `g` (inputs followed by all node outputs). -/
def allDefinedIds {Op : Type} (g : SSAGraph Op) : List ValueId :=
  g.inputIds ++ g.nodes.toList.flatMap IRNode.outputIds

/-- Next fresh `ValueId` strictly greater than every `ValueId` currently defined in `g`. -/
def nextValueId {Op : Type} (g : SSAGraph Op) : ValueId :=
  let maxIn := g.inputs.foldl (fun acc e => max acc (e.1 + 1)) 0
  g.nodes.foldl (fun acc nd =>
    nd.outputs.foldl (fun a o => max a (o.1 + 1)) acc) maxIn

/-- Look up the `TensorType` of SSA value `v` (checking graph inputs and all node outputs). -/
def valueType? {Op : Type} (g : SSAGraph Op) (v : ValueId) : Option TensorType :=
  match g.inputs.find? (fun e => e.1 == v) with
  | some (_, _, ty) => some ty
  | none =>
    g.nodes.findSome? fun nd =>
      (nd.outputs.find? fun o => o.1 == v).map Prod.snd

/-- Find the producing node `(nodeIdx, node, outputSlot)` for SSA value `v`, if not a graph input. -/
def producerOf? {Op : Type} (g : SSAGraph Op) (v : ValueId) : Option (Nat × IRNode Op × Nat) := Id.run do
  for h : i in [:g.nodes.size] do
    let nd := g.nodes[i]
    for h2 : j in [:nd.outputs.size] do
      if nd.outputs[j].1 == v then
        return some (i, nd, j)
  return none

/-- Count total uses of SSA value `v` across all node inputs and `g.outputs`. -/
def consumerCount {Op : Type} (g : SSAGraph Op) (v : ValueId) : Nat :=
  let inNodes := g.nodes.foldl (fun acc nd =>
    nd.inputs.foldl (fun a inp => if inp == v then a + 1 else a) acc) 0
  g.outputs.foldl (fun acc out => if out == v then acc + 1 else acc) inNodes

/-- Substitute every use of `oldId` with `newId` in node inputs and graph outputs. -/
def replaceUses {Op : Type} (g : SSAGraph Op) (oldId newId : ValueId) : SSAGraph Op :=
  let subst := fun (v : ValueId) => if v == oldId then newId else v
  { g with
    nodes := g.nodes.map fun nd => { nd with inputs := nd.inputs.map subst }
    outputs := g.outputs.map subst }

/-- Render an `SSAGraph Op` into a human-readable textual SSA listing (used by `--dump-ir`). -/
def pretty {Op : Type} (opStr : Op → String) (g : SSAGraph Op) : String := Id.run do
  let mut lines : Array String := #[]
  let inSig := g.inputs.toList.map fun (vid, nm, ty) => s!"%{vid} \"{nm}\" : {ty.pretty}"
  lines := lines.push s!"graph (target={g.target.name}) ({String.intercalate ", " inSig}) -> ({String.intercalate ", " (g.outputs.toList.map fun v => s!"%{v}")}) \{"
  for nd in g.nodes do
    let outs := nd.outputs.toList.map fun (vid, ty) => s!"%{vid} : {ty.pretty}"
    let ins := nd.inputs.toList.map fun vid => s!"%{vid}"
    let tag := if nd.name.isEmpty then "" else s!"  // {nd.name}"
    lines := lines.push s!"  {String.intercalate ", " outs} = {opStr nd.op}({String.intercalate ", " ins}){tag}"
  let outStr := String.intercalate ", " (g.outputs.toList.map fun v => s!"%{v}")
  lines := lines.push s!"  return ({outStr})"
  lines := lines.push "}"
  return String.intercalate "\n" lines.toList

def dumpHigh (g : SSAGraph HighOp) : String :=
  g.pretty HighOp.pretty

def dumpCore (g : SSAGraph CoreOp) : String :=
  g.pretty CoreOp.pretty

end SSAGraph

end Vane.IR
