import Vane.Binary
import Vane.TaskStream
import Vane.Tiling
import Vane.View
import Vane.IR.Ops
import Vane.IR.Graph
import Vane.IR.Validate
import Vane.IR.Semantics
import Vane.IR.Passes.Decompose
import Vane.Ane.Types
import Vane.Backend.ChannelSplit
import Vane.Backend.Cascade

namespace Vane.Proofs.C3Backend

open Vane.Binary
open Vane.TaskStream
open Vane.Tiling
open Vane.View
open Vane.IR
open Vane.Ane
open Vane.Backend.ChannelSplit
open Vane.Backend.Cascade

-- ============================================================================
-- Pillar 1: L1/L2 Semantic & Structural Preservation
-- ============================================================================

-- ============================================================================
-- 1A. HighOp → CoreOp Decompose Well-Formedness & SSA Monotonicity
-- ============================================================================

private theorem foldl_max_ge_init {α : Type} (f : α → Nat) (xs : List α) (init : Nat) :
    init ≤ xs.foldl (fun acc x => max acc (f x)) init := by
  induction xs generalizing init with
  | nil => exact Nat.le_refl _
  | cons x rest ih =>
    simp only [List.foldl_cons]
    exact Nat.le_trans (Nat.le_max_left _ _) (ih _)

private theorem foldl_max_ge_mem {α : Type} (f : α → Nat) (xs : List α) (init : Nat) :
    ∀ x ∈ xs, f x ≤ xs.foldl (fun acc y => max acc (f y)) init := by
  induction xs generalizing init with
  | nil => intro _ h; cases h
  | cons x rest ih =>
    intro y hy
    simp only [List.foldl_cons]
    rcases List.mem_cons.mp hy with rfl | hmem
    · exact Nat.le_trans (Nat.le_max_right _ _) (foldl_max_ge_init f rest _)
    · exact ih _ y hmem

private theorem foldl_max_nodes_ge_init {Op : Type} (ns : List (IRNode Op)) (init : Nat) :
    init ≤ ns.foldl (fun acc nd => nd.outputs.foldl (fun a o => max a (o.1 + 1)) acc) init := by
  induction ns generalizing init with
  | nil => exact Nat.le_refl _
  | cons nd rest ih =>
    simp only [List.foldl_cons]
    have h1 : init ≤ nd.outputs.foldl (fun a o => max a (o.1 + 1)) init := by
      rw [← Array.foldl_toList]
      exact foldl_max_ge_init (fun o => o.1 + 1) nd.outputs.toList init
    exact Nat.le_trans h1 (ih _)

private theorem foldl_max_nodes_ge_out {Op : Type} (ns : List (IRNode Op)) (init : Nat) :
    ∀ nd ∈ ns, ∀ o ∈ nd.outputs.toList,
      o.1 + 1 ≤ ns.foldl (fun acc n => n.outputs.foldl (fun a x => max a (x.1 + 1)) acc) init := by
  induction ns generalizing init with
  | nil => intro _ h; cases h
  | cons hd tl ih =>
    intro nd hnd o ho
    simp only [List.foldl_cons]
    rcases List.mem_cons.mp hnd with rfl | hmem
    · have h1 : o.1 + 1 ≤ nd.outputs.foldl (fun a x => max a (x.1 + 1)) init := by
        rw [← Array.foldl_toList]
        exact foldl_max_ge_mem (fun x => x.1 + 1) nd.outputs.toList init o ho
      exact Nat.le_trans h1 (foldl_max_nodes_ge_init tl _)
    · exact ih _ nd hmem o ho

/-- Every `ValueId` defined in an `SSAGraph` is strictly smaller than `g.nextValueId`. -/
theorem nextValueId_gt_allDefinedIds {Op : Type} (g : SSAGraph Op) :
    ∀ v ∈ g.allDefinedIds, v < g.nextValueId := by
  intro v hv
  unfold SSAGraph.allDefinedIds SSAGraph.inputIds at hv
  unfold SSAGraph.nextValueId
  dsimp only
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  rcases List.mem_append.mp hv with hIn | hNd
  · rw [List.mem_map] at hIn
    obtain ⟨e, he, rfl⟩ := hIn
    have h1 : e.1 + 1 ≤ g.inputs.toList.foldl (fun acc x => max acc (x.1 + 1)) 0 :=
      foldl_max_ge_mem (fun x => x.1 + 1) g.inputs.toList 0 e he
    have h2 := foldl_max_nodes_ge_init g.nodes.toList
      (g.inputs.toList.foldl (fun acc x => max acc (x.1 + 1)) 0)
    exact Nat.lt_of_succ_le (Nat.le_trans h1 h2)
  · rw [List.mem_flatMap] at hNd
    obtain ⟨nd, hnd, ho⟩ := hNd
    unfold IRNode.outputIds at ho
    rw [List.mem_map] at ho
    obtain ⟨o, hoMem, rfl⟩ := ho
    have h := foldl_max_nodes_ge_out g.nodes.toList
      (g.inputs.toList.foldl (fun acc x => max acc (x.1 + 1)) 0) nd hnd o hoMem
    exact Nat.lt_of_succ_le h

/-- Any `ValueId ≥ g.nextValueId` is guaranteed fresh (absent from `g.allDefinedIds`). -/
theorem fresh_id_not_mem_allDefinedIds {Op : Type} (g : SSAGraph Op) (vid : ValueId)
    (hge : g.nextValueId ≤ vid) : vid ∉ g.allDefinedIds := by
  intro hmem
  have hlt := nextValueId_gt_allDefinedIds g vid hmem
  exact Nat.not_lt_of_ge hge hlt

/-- Appending a node with fresh, pairwise-distinct output IDs, defined inputs, and positive
    output shapes preserves SSA `WellFormed`ness and monotonically extends `allDefinedIds`. -/
theorem pushNode_preservesWF {Op : Type} (g : SSAGraph Op) (nd : IRNode Op)
    (hwf : WellFormed g)
    (hOutNodup : nd.outputIds.Nodup)
    (hOutFresh : ∀ v ∈ nd.outputIds, v ∉ g.allDefinedIds)
    (hInDef : ∀ v ∈ nd.inputs.toList, v ∈ g.allDefinedIds)
    (hOutPos : ∀ o ∈ nd.outputs.toList, ∀ d ∈ o.2.shape.toList, d > 0) :
    WellFormed { g with nodes := g.nodes.push nd } := by
  let g' : SSAGraph Op := { g with nodes := g.nodes.push nd }
  have hNodesList : g'.nodes.toList = g.nodes.toList ++ [nd] := by
    simp [g', Array.toList_push]
  have hDefEq : g'.allDefinedIds = g.allDefinedIds ++ nd.outputIds := by
    unfold SSAGraph.allDefinedIds SSAGraph.inputIds
    rw [hNodesList, List.flatMap_append, List.flatMap_cons, List.flatMap_nil,
        List.append_nil, List.append_assoc]
  refine ⟨hwf.inputsDense, hwf.inputsPositive, ?_, ?_, ?_, ?_⟩
  · rw [hDefEq, List.nodup_append]
    exact ⟨hwf.uniqueIds, hOutNodup, fun a ha b hb hab => hOutFresh b hb (hab ▸ ha)⟩
  · intro i hi v hv
    have hiSz : i < g.nodes.size + 1 := by simpa [g'] using hi
    by_cases hlt : i < g.nodes.size
    · have hndEq : g'.nodes[i] = g.nodes[i] := Array.getElem_push_lt hlt
      rw [hndEq] at hv
      have hvBef := hwf.defBeforeUse i hlt v hv
      unfold SSAGraph.allDefinedIdsBefore at hvBef ⊢
      rw [hNodesList, List.take_append_of_le_length (by simpa using Nat.le_of_lt hlt)]
      exact hvBef
    · have hiEq : i = g.nodes.size := by omega
      subst hiEq
      have hndEq : g'.nodes[g.nodes.size] = nd := Array.getElem_push_eq
      rw [hndEq] at hv
      have hvAll := hInDef v hv
      unfold SSAGraph.allDefinedIdsBefore
      rw [hNodesList, List.take_left' (by simp)]
      exact hvAll
  · intro i hi o ho d hd
    by_cases hlt : i < g.nodes.size
    · have hndEq : g'.nodes[i] = g.nodes[i] := Array.getElem_push_lt hlt
      rw [hndEq] at ho
      exact hwf.nodeOutputsPositive i hlt o ho d hd
    · have hiEq : i = g.nodes.size := by
        have : i < g.nodes.size + 1 := by simpa [g'] using hi
        omega
      subst hiEq
      have hndEq : g'.nodes[g.nodes.size] = nd := Array.getElem_push_eq
      rw [hndEq] at ho
      exact hOutPos o ho d hd
  · intro v hv
    rw [hDefEq]
    exact List.mem_append_left _ (hwf.outputsValid v hv)

/-- Appending a single-output node with fresh ID `vid ≥ g.nextValueId` preserves `WellFormed`. -/
theorem pushSingleFreshNode_preservesWF {Op : Type} (g : SSAGraph Op)
    (name : String) (op : Op) (inputs : Array ValueId) (vid : ValueId) (ty : TensorType)
    (hwf : WellFormed g)
    (hFresh : g.nextValueId ≤ vid)
    (hInDef : ∀ v ∈ inputs.toList, v ∈ g.allDefinedIds)
    (hPos : ∀ d ∈ ty.shape.toList, d > 0) :
    WellFormed { g with nodes := g.nodes.push { name := name, op := op, inputs := inputs, outputs := #[(vid, ty)] } } := by
  apply pushNode_preservesWF g _ hwf
  · simp [IRNode.outputIds]
  · intro v hv
    simp [IRNode.outputIds] at hv
    rw [hv]
    exact fresh_id_not_mem_allDefinedIds g vid hFresh
  · exact hInDef
  · intro o ho d hd
    simp at ho
    subst ho
    exact hPos d hd

/-- Exact `nextValueId` recurrence when appending a single-output node:
    `(g.pushSingleNode vid).nextValueId = max g.nextValueId (vid + 1)`. -/
theorem pushSingleNode_nextValueId {Op : Type} (g : SSAGraph Op)
    (name : String) (op : Op) (inputs : Array ValueId) (vid : ValueId) (ty : TensorType) :
    ({ g with nodes := g.nodes.push { name := name, op := op, inputs := inputs, outputs := #[(vid, ty)] } } : SSAGraph Op).nextValueId =
    max g.nextValueId (vid + 1) := by
  unfold SSAGraph.nextValueId
  dsimp only
  rw [← Array.foldl_toList, Array.toList_push, List.foldl_append, List.foldl_cons, List.foldl_nil]
  rw [Array.foldl_toList]
  rfl

/-- Appending a single-output node makes its output ID available in `allDefinedIds` and
    preserves all previously defined IDs. -/
theorem pushSingleNode_mem_defined {Op : Type} (g : SSAGraph Op)
    (name : String) (op : Op) (inputs : Array ValueId) (vid : ValueId) (ty : TensorType) :
    vid ∈ ({ g with nodes := g.nodes.push { name := name, op := op, inputs := inputs, outputs := #[(vid, ty)] } } : SSAGraph Op).allDefinedIds ∧
    (∀ v ∈ g.allDefinedIds, v ∈ ({ g with nodes := g.nodes.push { name := name, op := op, inputs := inputs, outputs := #[(vid, ty)] } } : SSAGraph Op).allDefinedIds) := by
  have hDefEq : ({ g with nodes := g.nodes.push { name := name, op := op, inputs := inputs, outputs := #[(vid, ty)] } } : SSAGraph Op).allDefinedIds =
      g.allDefinedIds ++ [vid] := by
    unfold SSAGraph.allDefinedIds SSAGraph.inputIds
    simp [Array.toList_push, List.flatMap_append, IRNode.outputIds, List.append_assoc]
  rw [hDefEq]
  exact ⟨List.mem_append_right _ List.mem_cons_self, fun v hv => List.mem_append_left _ hv⟩

/-- Updating `g.outputs` to any list of already-defined `ValueId`s preserves `WellFormed`
    (models alias redirection at the end of `decomposeGraph`). -/
theorem redirectOutputs_preservesWF {Op : Type} (g : SSAGraph Op) (newOuts : Array ValueId)
    (hwf : WellFormed g) (hValid : ∀ v ∈ newOuts.toList, v ∈ g.allDefinedIds) :
    WellFormed { g with outputs := newOuts } :=
  ⟨hwf.inputsDense, hwf.inputsPositive, hwf.uniqueIds, hwf.defBeforeUse, hwf.nodeOutputsPositive, hValid⟩

private theorem array_find?_mem {α : Type} {xs : Array α} {p : α → Bool} {a : α}
    (h : xs.find? p = some a) : a ∈ xs.toList ∧ p a = true := by
  have hL : xs.toList.find? p = some a := by
    simpa [Array.find?_toList] using h
  exact ⟨List.mem_of_find?_eq_some hL, List.find?_some hL⟩

private theorem array_find?_eq_none_of_forall {α : Type} {xs : Array α} {p : α → Bool}
    (h : ∀ a ∈ xs.toList, p a = false) : xs.find? p = none := by
  have hL : xs.toList.find? p = none := List.find?_eq_none.mpr fun a ha => by simp [h a ha]
  simpa [Array.find?_toList] using hL

/-- If every alias pair `(oldId, newId)` maps into a closed set `S` of defined IDs, then
    `lookupAliasAux aliases fuel v` stays inside `S` for any `v ∈ S`. -/
theorem lookupAliasAux_closed (aliases : Array (ValueId × ValueId)) (S : ValueId → Prop)
    (hAlias : ∀ p ∈ aliases.toList, S p.2) :
    ∀ fuel v, S v → S (lookupAliasAux aliases fuel v) := by
  intro fuel
  induction fuel with
  | zero => intro v hv; exact hv
  | succ k ih =>
    intro v hv
    unfold lookupAliasAux
    cases hfind : aliases.find? (fun p => p.1 == v) with
    | none => exact hv
    | some p =>
      rcases p with ⟨a, nxt⟩
      have hpMem : (a, nxt) ∈ aliases.toList := (array_find?_mem hfind).1
      have hNxt : S nxt := hAlias (a, nxt) hpMem
      dsimp only
      split
      · exact hv
      · exact ih nxt hNxt

/-- `lookupAlias` maps any defined `ValueId` into the closed set `S` whenever all recorded
    alias targets lie in `S` (preventing dangling SSA references after identity elision). -/
theorem lookupAlias_closed (aliases : Array (ValueId × ValueId)) (S : ValueId → Prop)
    (hAlias : ∀ p ∈ aliases.toList, S p.2) (v : ValueId) (hv : S v) :
    S (lookupAlias aliases v) :=
  lookupAliasAux_closed aliases S hAlias 256 v hv

/-- `decomposeGraph` always preserves `target`, `inputs`, and output arity (`outputs.size`). -/
theorem decomposeGraph_preserves_signature (g : SSAGraph HighOp) (allowViewOps : Bool)
    (g' : SSAGraph CoreOp) (elided : Nat)
    (hok : decomposeGraph g allowViewOps = .ok (g', elided)) :
    g'.target = g.target ∧ g'.inputs = g.inputs ∧ g'.outputs.size = g.outputs.size := by
  unfold decomposeGraph at hok
  dsimp only [bind, Except.bind, pure, Except.pure] at hok
  cases hfold : g.nodes.foldlM (decomposeNode g.inputs allowViewOps)
      { outNodes := Array.mkEmpty g.nodes.size, aliases := #[], nextId := g.nextValueId, elidedCount := 0 } with
  | error e => rw [hfold] at hok; cases hok
  | ok finalSt =>
    rw [hfold] at hok
    simp only [Except.ok.injEq, Prod.mk.injEq] at hok
    rcases hok with ⟨rfl, _⟩
    exact ⟨rfl, rfl, Array.size_map⟩

-- ============================================================================
-- 1A-full. End-to-End `decomposeGraph` SSA `WellFormed` Preservation (All 8 HighOps)
-- ============================================================================

/-- Core graph view over the current emitted `outNodes` prefix during `decomposeGraph`. -/
private abbrev decompGraph (g : SSAGraph HighOp) (nodes : Array (IRNode CoreOp)) : SSAGraph CoreOp :=
  { target := g.target, inputs := g.inputs, nodes := nodes, outputs := #[] }

/-- Working state invariant maintained across pushes within a single `decomposeNode` call
    for node `nd = g.nodes[k]`, where `usedOrig` tracks original output IDs of `nd` already emitted. -/
private structure DecompStepInv
    (g : SSAGraph HighOp) (k : Nat) (usedOrig : List ValueId)
    (nodes : Array (IRNode CoreOp)) (aliases : Array (ValueId × ValueId)) (nextId : Nat) : Prop where
  nextGe : g.nextValueId ≤ nextId
  wf : WellFormed (decompGraph g nodes)
  defBound : ∀ v ∈ (decompGraph g nodes).allDefinedIds,
    v ∈ g.allDefinedIdsBefore k ∨ v ∈ usedOrig ∨ (g.nextValueId ≤ v ∧ v < nextId)
  aliasTargetDef : ∀ p ∈ aliases.toList, p.2 ∈ (decompGraph g nodes).allDefinedIds
  aliasKeyPrior : ∀ p ∈ aliases.toList, p.1 ∈ g.allDefinedIdsBefore k ∧ p.1 ∉ (decompGraph g nodes).allDefinedIds
  priorCovered : ∀ v ∈ g.allDefinedIdsBefore k, lookupAlias aliases v ∈ (decompGraph g nodes).allDefinedIds

private theorem lookupTypeInCore_pos (g : SSAGraph HighOp) (nodes : Array (IRNode CoreOp))
    (hwf : WellFormed (decompGraph g nodes)) (v : ValueId) (ty : TensorType)
    (hFind : lookupTypeInCore g.inputs nodes v = some ty) :
    ∀ d ∈ ty.shape.toList, d > 0 := by
  unfold lookupTypeInCore at hFind
  cases hIn : g.inputs.find? (fun e => e.1 == v) with
  | some e =>
    rw [hIn] at hFind
    rcases e with ⟨iv, nm, ity⟩
    simp only [Option.some.injEq] at hFind
    subst hFind
    have hMem := (array_find?_mem hIn).1
    rw [List.mem_iff_getElem] at hMem
    obtain ⟨idx, hIdx, hEq⟩ := hMem
    have hIdx' : idx < (decompGraph g nodes).inputs.size := by simpa using hIdx
    have hAt : (decompGraph g nodes).inputs[idx]'hIdx' = (iv, nm, ity) := by simpa using hEq
    have hPos := hwf.inputsPositive idx hIdx'
    rw [hAt] at hPos
    exact hPos
  | none =>
    rw [hIn] at hFind
    simp only at hFind
    have hList : nodes.toList.findSome? (fun nd => (nd.outputs.find? fun o => o.1 == v).map Prod.snd) = some ty := by
      simpa [Array.findSome?_toList] using hFind
    rw [List.findSome?_eq_some_iff] at hList
    obtain ⟨_, nd, _, hEqList, hNd, _⟩ := hList
    have hNdMem : nd ∈ nodes.toList := by rw [hEqList]; simp
    rw [List.mem_iff_getElem] at hNdMem
    obtain ⟨idx, hIdx, hNdEq⟩ := hNdMem
    cases hOut : nd.outputs.find? (fun o => o.1 == v) with
    | none => rw [hOut] at hNd; cases hNd
    | some o =>
      rw [hOut] at hNd
      simp only [Option.map_some, Option.some.injEq] at hNd
      subst hNd
      have hOMem := (array_find?_mem hOut).1
      have hIdx' : idx < (decompGraph g nodes).nodes.size := by simpa using hIdx
      have hNdAt : (decompGraph g nodes).nodes[idx]'hIdx' = nd := by
        simpa using hNdEq
      exact hwf.nodeOutputsPositive idx hIdx' o (hNdAt ▸ hOMem)

private theorem array_set!_pos (arr : Array Nat) (idx val : Nat)
    (hArr : ∀ d ∈ arr.toList, d > 0) (hVal : val > 0) :
    ∀ d ∈ (arr.set! idx val).toList, d > 0 := by
  intro d hd
  by_cases hlt : idx < arr.size
  · rw [Array.set!_eq_setIfInBounds, Array.setIfInBounds] at hd
    simp only [hlt, ↓reduceDIte] at hd
    rw [Array.toList_set] at hd
    rw [List.mem_iff_getElem] at hd
    obtain ⟨j, hj, rfl⟩ := hd
    by_cases hji : j = idx
    · subst hji; simp [hVal]
    · simp only [List.getElem_set_ne (Ne.symm hji)]
      exact hArr _ (List.getElem_mem _)
  · have hEq : arr.set! idx val = arr := by
      rw [Array.set!_eq_setIfInBounds, Array.setIfInBounds]
      simp [hlt]
    rw [hEq] at hd
    exact hArr d hd

private theorem softmaxRedShape_pos (sh redAxes : Array Nat)
    (hPos : ∀ d ∈ sh.toList, d > 0) :
    ∀ d ∈ (softmaxRedShape sh redAxes).toList, d > 0 := by
  unfold softmaxRedShape
  rw [← Array.foldl_toList]
  have hFold : ∀ (axs : List Nat) (cur : Array Nat),
      (∀ d ∈ cur.toList, d > 0) →
      ∀ d ∈ (axs.foldl (fun s a => if a < s.size then s.set! a 1 else s) cur).toList, d > 0 := by
    intro axs
    induction axs with
    | nil => intro cur hCur; exact hCur
    | cons a rest ih =>
      intro cur hCur
      simp only [List.foldl_cons]
      apply ih
      split
      · exact array_set!_pos cur a 1 hCur (by omega)
      · exact hCur
  exact hFold redAxes.toList sh hPos

private theorem array_getElem!_pos_of_all_pos (arr : Array Nat) (idx : Nat)
    (hArr : ∀ d ∈ arr.toList, d > 0) (hlt : idx < arr.size) : arr[idx]! > 0 := by
  rw [getElem!_pos arr idx hlt]
  exact hArr arr[idx] (Array.getElem_mem_toList hlt)

/-- Push a fresh helper node with ID `vid = nextId` (`≥ g.nextValueId`), advancing `nextId := nextId + 1`! -/
private theorem stepInv_pushFresh
    {g : SSAGraph HighOp} {k : Nat} {usedOrig : List ValueId}
    {nodes : Array (IRNode CoreOp)} {aliases : Array (ValueId × ValueId)} {nextId : Nat}
    (hInv : DecompStepInv g k usedOrig nodes aliases nextId)
    (hUsedOrig : ∀ v ∈ usedOrig, v < g.nextValueId)
    (name : String) (op : CoreOp) (inputs : Array ValueId) (ty : TensorType)
    (hInDef : ∀ v ∈ inputs.toList, v ∈ (decompGraph g nodes).allDefinedIds)
    (hPos : ∀ d ∈ ty.shape.toList, d > 0) :
    let nd' : IRNode CoreOp := { name := name, op := op, inputs := inputs, outputs := #[(nextId, ty)] }
    DecompStepInv g k usedOrig (nodes.push nd') aliases (nextId + 1) ∧
    nextId ∈ (decompGraph g (nodes.push nd')).allDefinedIds ∧
    (∀ v ∈ (decompGraph g nodes).allDefinedIds, v ∈ (decompGraph g (nodes.push nd')).allDefinedIds) := by
  intro nd'
  have hFreshNotDef : nextId ∉ (decompGraph g nodes).allDefinedIds := by
    intro hmem
    rcases hInv.defBound nextId hmem with hPrior | hUsed | ⟨_, hLt⟩
    · have hAll : nextId ∈ g.allDefinedIds := by
        unfold SSAGraph.allDefinedIdsBefore at hPrior
        unfold SSAGraph.allDefinedIds
        rcases List.mem_append.mp hPrior with hIn | hNd
        · exact List.mem_append_left _ hIn
        · rw [List.mem_flatMap] at hNd
          obtain ⟨x, hx, hv⟩ := hNd
          exact List.mem_append_right _ (List.mem_flatMap.mpr ⟨x, List.mem_of_mem_take hx, hv⟩)
      have hLtG := nextValueId_gt_allDefinedIds g nextId hAll
      exact Nat.not_lt_of_ge hInv.nextGe hLtG
    · exact Nat.not_lt_of_ge hInv.nextGe (hUsedOrig nextId hUsed)
    · exact Nat.lt_irrefl nextId hLt
  have hWf' : WellFormed (decompGraph g (nodes.push nd')) :=
    pushNode_preservesWF (decompGraph g nodes) nd' hInv.wf
      (by simp [nd', IRNode.outputIds])
      (fun v hv => by simp [nd', IRNode.outputIds] at hv; rw [hv]; exact hFreshNotDef)
      hInDef
      (fun o ho d hd => by simp [nd'] at ho; subst ho; exact hPos d hd)
  have hDefEq : (decompGraph g (nodes.push nd')).allDefinedIds =
      (decompGraph g nodes).allDefinedIds ++ [nextId] := by
    unfold SSAGraph.allDefinedIds SSAGraph.inputIds
    simp [nd', Array.toList_push, List.flatMap_append, IRNode.outputIds, List.append_assoc]
  have hMono : ∀ v ∈ (decompGraph g nodes).allDefinedIds, v ∈ (decompGraph g (nodes.push nd')).allDefinedIds := by
    intro v hv; rw [hDefEq]; exact List.mem_append_left _ hv
  refine ⟨⟨Nat.le_succ_of_le hInv.nextGe, hWf', ?_, ?_, ?_, ?_⟩,
          by rw [hDefEq]; exact List.mem_append_right _ List.mem_cons_self, hMono⟩
  · intro v hv
    rw [hDefEq, List.mem_append, List.mem_singleton] at hv
    rcases hv with hOld | rfl
    · rcases hInv.defBound v hOld with h1 | h2 | ⟨h3, h4⟩
      · exact Or.inl h1
      · exact Or.inr (Or.inl h2)
      · exact Or.inr (Or.inr ⟨h3, Nat.lt_succ_of_lt h4⟩)
    · exact Or.inr (Or.inr ⟨hInv.nextGe, Nat.lt_succ_self _⟩)
  · intro p hp; exact hMono _ (hInv.aliasTargetDef p hp)
  · intro p hp
    have ⟨hPr, hNot⟩ := hInv.aliasKeyPrior p hp
    refine ⟨hPr, ?_⟩
    rw [hDefEq, List.mem_append, List.mem_singleton]
    rintro (hOld | hEq)
    · exact hNot hOld
    · have hAll : p.1 ∈ g.allDefinedIds := by
        unfold SSAGraph.allDefinedIdsBefore at hPr
        unfold SSAGraph.allDefinedIds
        rcases List.mem_append.mp hPr with hIn | hNd
        · exact List.mem_append_left _ hIn
        · rw [List.mem_flatMap] at hNd
          obtain ⟨x, hx, hv⟩ := hNd
          exact List.mem_append_right _ (List.mem_flatMap.mpr ⟨x, List.mem_of_mem_take hx, hv⟩)
      have hLtG := nextValueId_gt_allDefinedIds g p.1 hAll
      exact Nat.not_lt_of_ge hInv.nextGe (hEq ▸ hLtG)
  · intro v hv; exact hMono _ (hInv.priorCovered v hv)

/-- Push a node emitting original output pairs `outs` from `g.nodes[k]`, all strictly `< g.nextValueId`
    and disjoint from `g.allDefinedIdsBefore k` and `usedOrig`! -/
private theorem stepInv_pushOrig
    {g : SSAGraph HighOp} {k : Nat} {usedOrig : List ValueId}
    {nodes : Array (IRNode CoreOp)} {aliases : Array (ValueId × ValueId)} {nextId : Nat}
    (hInv : DecompStepInv g k usedOrig nodes aliases nextId)
    (name : String) (op : CoreOp) (inputs : Array ValueId) (outs : Array (ValueId × TensorType))
    (hInDef : ∀ v ∈ inputs.toList, v ∈ (decompGraph g nodes).allDefinedIds)
    (hOutNodup : (outs.toList.map Prod.fst).Nodup)
    (hOutLt : ∀ o ∈ outs.toList, o.1 < g.nextValueId)
    (hOutNotPrior : ∀ o ∈ outs.toList, o.1 ∉ g.allDefinedIdsBefore k)
    (hOutNotUsed : ∀ o ∈ outs.toList, o.1 ∉ usedOrig)
    (hOutPos : ∀ o ∈ outs.toList, ∀ d ∈ o.2.shape.toList, d > 0) :
    let nd' : IRNode CoreOp := { name := name, op := op, inputs := inputs, outputs := outs }
    DecompStepInv g k (usedOrig ++ outs.toList.map Prod.fst) (nodes.push nd') aliases nextId ∧
    (∀ o ∈ outs.toList, o.1 ∈ (decompGraph g (nodes.push nd')).allDefinedIds) ∧
    (∀ v ∈ (decompGraph g nodes).allDefinedIds, v ∈ (decompGraph g (nodes.push nd')).allDefinedIds) := by
  intro nd'
  have hOutFresh : ∀ v ∈ nd'.outputIds, v ∉ (decompGraph g nodes).allDefinedIds := by
    intro v hv hmem
    unfold IRNode.outputIds at hv
    rw [List.mem_map] at hv
    obtain ⟨o, ho, rfl⟩ := hv
    rcases hInv.defBound o.1 hmem with h1 | h2 | ⟨h3, _⟩
    · exact hOutNotPrior o ho h1
    · exact hOutNotUsed o ho h2
    · exact Nat.not_lt_of_ge h3 (hOutLt o ho)
  have hWf' : WellFormed (decompGraph g (nodes.push nd')) :=
    pushNode_preservesWF (decompGraph g nodes) nd' hInv.wf hOutNodup hOutFresh hInDef hOutPos
  have hDefEq : (decompGraph g (nodes.push nd')).allDefinedIds =
      (decompGraph g nodes).allDefinedIds ++ outs.toList.map Prod.fst := by
    unfold SSAGraph.allDefinedIds SSAGraph.inputIds
    simp [nd', Array.toList_push, List.flatMap_append, IRNode.outputIds, List.append_assoc]
  have hMono : ∀ v ∈ (decompGraph g nodes).allDefinedIds, v ∈ (decompGraph g (nodes.push nd')).allDefinedIds := by
    intro v hv; rw [hDefEq]; exact List.mem_append_left _ hv
  refine ⟨⟨hInv.nextGe, hWf', ?_, ?_, ?_, ?_⟩, ?_, hMono⟩
  · intro v hv
    rw [hDefEq, List.mem_append] at hv
    rcases hv with hOld | hNew
    · rcases hInv.defBound v hOld with h1 | h2 | h3
      · exact Or.inl h1
      · exact Or.inr (Or.inl (List.mem_append_left _ h2))
      · exact Or.inr (Or.inr h3)
    · exact Or.inr (Or.inl (List.mem_append_right _ hNew))
  · intro p hp; exact hMono _ (hInv.aliasTargetDef p hp)
  · intro p hp
    have ⟨hPr, hNot⟩ := hInv.aliasKeyPrior p hp
    refine ⟨hPr, ?_⟩
    rw [hDefEq, List.mem_append]
    rintro (hOld | hNew)
    · exact hNot hOld
    · rw [List.mem_map] at hNew
      obtain ⟨o, ho, hEq⟩ := hNew
      exact hOutNotPrior o ho (hEq ▸ hPr)
  · intro v hv; exact hMono _ (hInv.priorCovered v hv)
  · intro o ho
    rw [hDefEq]
    exact List.mem_append_right _ (List.mem_map.mpr ⟨o, ho, rfl⟩)

/-- Helper: if `v` is NOT a key in `aliases`, then `lookupAlias aliases v = v`. -/
private theorem lookupAlias_of_not_key (aliases : Array (ValueId × ValueId)) (v : ValueId)
    (hNotKey : ∀ p ∈ aliases.toList, p.1 ≠ v) :
    lookupAlias aliases v = v := by
  unfold lookupAlias lookupAliasAux
  have hNone : aliases.find? (fun p => p.1 == v) = none :=
    array_find?_eq_none_of_forall fun p hp => by simp [hNotKey p hp]
  rw [hNone]

/-- Helper: appending a fresh alias `(outId, xId)` where `outId` is absent from `aliases`,
    `definedIds`, and `g.allDefinedIdsBefore k`, and `xId ∈ definedIds`, preserves `lookupAlias`
    for all prior values AND resolves `outId` to `xId` in 1 step! -/
private theorem lookupAlias_push_fresh
    (aliases : Array (ValueId × ValueId)) (outId xId : ValueId)
    (hOutNotKey : ∀ p ∈ aliases.toList, p.1 ≠ outId)
    (hXNotKey : ∀ p ∈ aliases.toList, p.1 ≠ xId)
    (hNe : outId ≠ xId) :
    lookupAlias (aliases.push (outId, xId)) outId = xId ∧
    ∀ v, v ≠ outId → (∀ p ∈ aliases.toList, p.2 ≠ outId) →
      lookupAlias (aliases.push (outId, xId)) v = lookupAlias aliases v := by
  have hFindOut : (aliases.push (outId, xId)).find? (fun p => p.1 == outId) = some (outId, xId) := by
    rw [← Array.find?_toList, Array.toList_push, List.find?_append]
    have hL : aliases.toList.find? (fun p => p.1 == outId) = none :=
      List.find?_eq_none.mpr fun p hp => by simp [hOutNotKey p hp]
    rw [hL]; simp
  have hFindX : (aliases.push (outId, xId)).find? (fun p => p.1 == xId) = none :=
    array_find?_eq_none_of_forall fun p hp => by
      rw [Array.toList_push, List.mem_append, List.mem_singleton] at hp
      rcases hp with hOld | rfl
      · simp [hXNotKey p hOld]
      · simp [hNe]
  constructor
  · unfold lookupAlias
    change lookupAliasAux (aliases.push (outId, xId)) 256 outId = xId
    unfold lookupAliasAux
    rw [hFindOut]
    have hNeB : (xId == outId) = false := by simp [Ne.symm hNe]
    simp only [hNeB, Bool.false_eq_true, ↓reduceIte]
    unfold lookupAliasAux
    rw [hFindX]
  · intro v hVne hTgtNe
    unfold lookupAlias
    have hGen : ∀ fuel u, u ≠ outId →
        lookupAliasAux (aliases.push (outId, xId)) fuel u = lookupAliasAux aliases fuel u := by
      intro fuel
      induction fuel with
      | zero => intro u _; rfl
      | succ f ih =>
        intro u hUne
        unfold lookupAliasAux
        have hFindEq : (aliases.push (outId, xId)).find? (fun p => p.1 == u) =
            aliases.find? (fun p => p.1 == u) := by
          rw [← Array.find?_toList, ← Array.find?_toList, Array.toList_push, List.find?_append]
          cases hF : aliases.toList.find? (fun p => p.1 == u) with
          | some p => simp
          | none => simp [Ne.symm hUne]
        rw [hFindEq]
        cases hF : aliases.find? (fun p => p.1 == u) with
        | none => rfl
        | some p =>
          rcases p with ⟨a, nxt⟩
          dsimp only
          split
          · rfl
          · exact ih nxt (hTgtNe (a, nxt) (array_find?_mem hF).1)
    exact hGen 256 v hVne

/-- Promote `DecompStepInv g k usedOrig` to `DecompStepInv g (k + 1) []` once all outputs of
    `nd = g.nodes[k]` are covered (either emitted in `nodes` or aliased). -/
private theorem stepInv_finish
    {g : SSAGraph HighOp} {k : Nat} (hk : k < g.nodes.size)
    {usedOrig : List ValueId} {nodes : Array (IRNode CoreOp)}
    {aliases : Array (ValueId × ValueId)} {nextId : Nat}
    (hInv : DecompStepInv g k usedOrig nodes aliases nextId)
    (hUsedSub : ∀ v ∈ usedOrig, v ∈ g.nodes[k].outputIds)
    (hNewCov : ∀ v ∈ g.nodes[k].outputIds, lookupAlias aliases v ∈ (decompGraph g nodes).allDefinedIds) :
    DecompStepInv g (k + 1) [] nodes aliases nextId := by
  have hBefSucc : g.allDefinedIdsBefore (k + 1) = g.allDefinedIdsBefore k ++ g.nodes[k].outputIds := by
    unfold SSAGraph.allDefinedIdsBefore
    have hTake : g.nodes.toList.take (k + 1) = g.nodes.toList.take k ++ [g.nodes[k]] := by
      rw [List.take_succ_eq_append_getElem (by simpa using hk)]; rfl
    rw [hTake, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.append_assoc]
  refine ⟨hInv.nextGe, hInv.wf, ?_, hInv.aliasTargetDef, ?_, ?_⟩
  · intro v hv
    rcases hInv.defBound v hv with h1 | h2 | h3
    · exact Or.inl (by rw [hBefSucc]; exact List.mem_append_left _ h1)
    · exact Or.inl (by rw [hBefSucc]; exact List.mem_append_right _ (hUsedSub v h2))
    · exact Or.inr (Or.inr h3)
  · intro p hp
    have ⟨hPr, hNot⟩ := hInv.aliasKeyPrior p hp
    exact ⟨by rw [hBefSucc]; exact List.mem_append_left _ hPr, hNot⟩
  · intro v hv
    rw [hBefSucc, List.mem_append] at hv
    rcases hv with hOld | hCur
    · exact hInv.priorCovered v hOld
    · exact hNewCov v hCur

/-- Every node output ID of `g.nodes[k]` in a `WellFormed` graph is `< g.nextValueId`,
    absent from `g.allDefinedIdsBefore k`, and pairwise distinct (`Nodup`). -/
private theorem wellFormed_node_outputs_fresh
    {Op : Type} {g : SSAGraph Op} (hwf : WellFormed g) {k : Nat} (hk : k < g.nodes.size) :
    g.nodes[k].outputIds.Nodup ∧
    (∀ o ∈ g.nodes[k].outputs.toList, o.1 < g.nextValueId) ∧
    (∀ o ∈ g.nodes[k].outputs.toList, o.1 ∉ g.allDefinedIdsBefore k) := by
  have hSplit : g.nodes.toList = g.nodes.toList.take k ++ g.nodes[k] :: g.nodes.toList.drop (k + 1) := by
    have h1 := List.take_append_drop k g.nodes.toList
    have h2 : g.nodes.toList.drop k = g.nodes[k] :: g.nodes.toList.drop (k + 1) := by
      rw [List.drop_eq_getElem_cons (by simpa using hk)]; rfl
    rw [← h2]; exact h1.symm
  have hDefSplit : g.allDefinedIds =
      g.allDefinedIdsBefore k ++ (g.nodes[k].outputIds ++ (g.nodes.toList.drop (k + 1)).flatMap IRNode.outputIds) := by
    unfold SSAGraph.allDefinedIds SSAGraph.allDefinedIdsBefore
    conv => lhs; rw [hSplit]
    rw [List.flatMap_append, List.flatMap_cons, List.append_assoc]
  have hNd := hwf.uniqueIds
  rw [hDefSplit, List.nodup_append] at hNd
  have hNdCur := (List.nodup_append.mp hNd.2.1).1
  refine ⟨hNdCur, ?_, ?_⟩
  · intro o ho
    apply nextValueId_gt_allDefinedIds g o.1
    rw [hDefSplit]
    exact List.mem_append_right _ (List.mem_append_left _ (List.mem_map.mpr ⟨o, ho, rfl⟩))
  · intro o ho hmem
    have hMemCur : o.1 ∈ g.nodes[k].outputIds := List.mem_map.mpr ⟨o, ho, rfl⟩
    exact hNd.2.2 o.1 hmem o.1 (List.mem_append_left _ hMemCur) rfl

private theorem decomposeSplitSlicesAux_preserves
    (g : SSAGraph HighOp) (k : Nat) (hk : k < g.nodes.size) (hwf : WellFormed g)
    (nodeName : String) (xId : ValueId) (xShape : Array Nat) (axis : Nat)
    (sizes : Array Nat) (hSzEq : sizes.size = g.nodes[k].outputs.size) :
    ∀ rem i curStart nodes aliases nextId,
      rem = sizes.size - i → i ≤ sizes.size →
      DecompStepInv g k ((g.nodes[k].outputs.toList.take i).map Prod.fst) nodes aliases nextId →
      xId ∈ (decompGraph g nodes).allDefinedIds →
      let outN := decomposeSplitSlicesAux nodeName xId xShape axis sizes g.nodes[k].outputs i curStart nodes
      DecompStepInv g k ((g.nodes[k].outputs.toList.take sizes.size).map Prod.fst) outN aliases nextId ∧
      (∀ o ∈ g.nodes[k].outputs.toList.drop i, o.1 ∈ (decompGraph g outN).allDefinedIds) ∧
      (∀ v ∈ (decompGraph g nodes).allDefinedIds, v ∈ (decompGraph g outN).allDefinedIds) := by
  intro rem
  induction rem with
  | zero =>
    intro i curStart nodes aliases nextId hrem hi hInv _
    have hiEq : i = sizes.size := by omega
    subst hiEq
    unfold decomposeSplitSlicesAux
    simp only [Nat.lt_irrefl, ↓reduceDIte]
    refine ⟨hInv, ?_, fun _ h => h⟩
    intro o ho
    have hDropNil : g.nodes[k].outputs.toList.drop sizes.size = [] :=
      List.drop_eq_nil_of_le (by simp [hSzEq])
    rw [hDropNil] at ho; cases ho
  | succ r ih =>
    intro i curStart nodes aliases nextId hrem hi hInv hxDef
    have hiLt : i < sizes.size := by omega
    have hiOut : i < g.nodes[k].outputs.size := by omega
    unfold decomposeSplitSlicesAux
    simp only [hiLt, ↓reduceDIte]
    have ⟨hOutNd, hOutLt, hOutNotPrior⟩ := wellFormed_node_outputs_fresh hwf hk
    have hGet : g.nodes[k].outputs[i]! = g.nodes[k].outputs[i] := getElem!_pos _ i hiOut
    rw [hGet]
    have hOMem : g.nodes[k].outputs[i] ∈ g.nodes[k].outputs.toList := Array.getElem_mem_toList hiOut
    have hNotUsed : g.nodes[k].outputs[i].1 ∉ (g.nodes[k].outputs.toList.take i).map Prod.fst := by
      intro hmem
      rw [List.mem_map] at hmem
      obtain ⟨o', ho', hEqId⟩ := hmem
      rw [List.mem_iff_getElem] at ho'
      obtain ⟨j, hj, hEqO⟩ := ho'
      have hjLtI : j < i := by simp only [List.length_take, Array.length_toList] at hj; omega
      have hSplitTake : g.nodes[k].outputIds =
          (g.nodes[k].outputs.toList.take i).map Prod.fst ++
          g.nodes[k].outputs[i].1 :: (g.nodes[k].outputs.toList.drop (i + 1)).map Prod.fst := by
        unfold IRNode.outputIds
        have h1 := List.take_append_drop i g.nodes[k].outputs.toList
        have h2 : g.nodes[k].outputs.toList.drop i = g.nodes[k].outputs[i] :: g.nodes[k].outputs.toList.drop (i + 1) := by
          rw [List.drop_eq_getElem_cons (by simpa using hiOut)]; rfl
        conv => lhs; rw [← h1, h2, List.map_append, List.map_cons]
      rw [hSplitTake, List.nodup_append] at hOutNd
      exact hOutNd.2.2 g.nodes[k].outputs[i].1
        (List.mem_map.mpr ⟨o', hEqO ▸ List.getElem_mem hj, hEqId⟩)
        g.nodes[k].outputs[i].1 List.mem_cons_self rfl
    have hPush := stepInv_pushOrig hInv s!"{nodeName}_p{i}"
      (.slice ((Array.replicate xShape.size 0).set! axis curStart)
              (xShape.set! axis (curStart + sizes[i]))
              (Array.replicate xShape.size 1))
      #[xId] #[g.nodes[k].outputs[i]]
      (fun v hv => by simp at hv; subst hv; exact hxDef)
      (by simp)
      (fun o ho => by simp at ho; subst ho; exact hOutLt _ hOMem)
      (fun o ho => by simp at ho; subst ho; exact hOutNotPrior _ hOMem)
      (fun o ho => by simp at ho; subst ho; exact hNotUsed)
      (fun o ho d hd => by simp at ho; subst ho; exact hwf.nodeOutputsPositive k hk _ hOMem d hd)
    have hTakeSucc : (g.nodes[k].outputs.toList.take (i + 1)).map Prod.fst =
        (g.nodes[k].outputs.toList.take i).map Prod.fst ++ #[g.nodes[k].outputs[i]].toList.map Prod.fst := by
      rw [List.take_succ_eq_append_getElem (by simpa using hiOut), List.map_append]; rfl
    have hRec := ih (i + 1) (curStart + sizes[i]) _ aliases nextId (by omega) (by omega)
      (hTakeSucc ▸ hPush.1) (hPush.2.2 _ hxDef)
    refine ⟨hRec.1, ?_, fun v hv => hRec.2.2 v (hPush.2.2 v hv)⟩
    intro o ho
    have hDropCons : g.nodes[k].outputs.toList.drop i = g.nodes[k].outputs[i] :: g.nodes[k].outputs.toList.drop (i + 1) := by
      rw [List.drop_eq_getElem_cons (by simpa using hiOut)]; rfl
    rw [hDropCons, List.mem_cons] at ho
    rcases ho with rfl | hRest
    · exact hRec.2.2 _ (hPush.2.1 _ (by simp))
    · exact hRec.2.1 o hRest

/-- Standard ONNX output-arity precondition on `HighOp.batchNorm` (1 output) and
    `HighOp.layerNorm` (`≤ 3` outputs: `[Y, Mean?, InvStdDev?]`). -/
def ValidHighOpOutputArity (g : SSAGraph HighOp) : Prop :=
  ∀ nd ∈ g.nodes.toList, match nd.op with
    | .batchNorm _ => nd.outputs.size = 1
    | .layerNorm .. => nd.outputs.size ≤ 3
    | _ => True

private theorem decomposeLayerNormNonUniformConst_preserves_inv
    {g : SSAGraph HighOp} {k : Nat} {usedOrig : List ValueId}
    {nodes : Array (IRNode CoreOp)} {aliases : Array (ValueId × ValueId)} {nextId : Nat}
    (hInv : DecompStepInv g k usedOrig nodes aliases nextId)
    (hUsedLt : ∀ v ∈ usedOrig, v < g.nextValueId)
    (allowViewOps : Bool) (ndName ctx : String) (xTy : TensorType)
    (hxPos : ∀ d ∈ xTy.shape.toList, d > 0)
    (normAxis normDim : Nat) (hNormPos : normDim > 0)
    (yOut : ValueId × TensorType)
    (hyLt : yOut.1 < g.nextValueId)
    (hyNotPrior : yOut.1 ∉ g.allDefinedIdsBefore k)
    (hyNotUsed : yOut.1 ∉ usedOrig)
    (hyPos : ∀ d ∈ yOut.2.shape.toList, d > 0)
    (diffId invStdId : ValueId)
    (hDiffDef : diffId ∈ (decompGraph g nodes).allDefinedIds)
    (hInvDef : invStdId ∈ (decompGraph g nodes).allDefinedIds)
    (scVals : Array Float) (biasVals? : Option (Array Float))
    (outNodes' : Array (IRNode CoreOp)) (nextId' : ValueId)
    (hOk : decomposeLayerNormNonUniformConst allowViewOps ndName ctx xTy normAxis normDim
      yOut diffId invStdId scVals biasVals? nodes nextId = .ok (outNodes', nextId')) :
    DecompStepInv g k (usedOrig ++ [yOut.1]) outNodes' aliases nextId' ∧
    yOut.1 ∈ (decompGraph g outNodes').allDefinedIds ∧
    (∀ v ∈ (decompGraph g nodes).allDefinedIds, v ∈ (decompGraph g outNodes').allDefinedIds) := by
  unfold decomposeLayerNormNonUniformConst at hOk
  dsimp only [bind, Except.bind, pure, Except.pure, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw] at hOk
  let fullSc := if scVals.size == 1 then Array.replicate normDim scVals[0]! else scVals
  have hPNorm := stepInv_pushFresh hInv hUsedLt s!"{ndName}_norm"
    (.elemBinary .mul) #[diffId, invStdId] xTy
    (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hDiffDef; exact hInvDef) hxPos
  have hPW := stepInv_pushFresh hPNorm.1 hUsedLt s!"{ndName}_w"
    (.const (.f32 (buildDiagWeights normDim fullSc))) #[]
    { shape := #[normDim, normDim, 1, 1], dtype := .f32 }
    (fun _ h => by simp at h)
    (fun d hd => by simp at hd; rcases hd with rfl | rfl | rfl | rfl <;> omega)
  have hPushY := fun {ns : Array (IRNode CoreOp)} {nid : Nat}
      (hI : DecompStepInv g k usedOrig ns aliases nid)
      (hL : ∀ v ∈ (decompGraph g nodes).allDefinedIds, v ∈ (decompGraph g ns).allDefinedIds)
      (opF : CoreOp) (inpsF : Array ValueId)
      (hInF : ∀ v ∈ inpsF.toList, v ∈ (decompGraph g ns).allDefinedIds) => by
    have hP := stepInv_pushOrig hI ndName opF inpsF #[yOut] hInF (by simp)
      (fun o ho => by simp at ho; subst ho; exact hyLt)
      (fun o ho => by simp at ho; subst ho; exact hyNotPrior)
      (fun o ho => by simp at ho; subst ho; exact hyNotUsed)
      (fun o ho d hd => by simp at ho; subst ho; exact hyPos d hd)
    exact And.intro hP.1 (And.intro (hP.2.1 yOut (by simp)) (fun v hv => hP.2.2 v (hL v hv)))
  cases hFB : biasVals?.map (fun bv => if bv.size == 1 then Array.replicate normDim bv[0]! else bv) with
  | some fb =>
    rw [hFB] at hOk; dsimp only at hOk
    have hPB := stepInv_pushFresh hPW.1 hUsedLt s!"{ndName}_b"
      (.const (.f32 fb)) #[] { shape := #[normDim], dtype := .f32 }
      (fun _ h => by simp at h) (fun d hd => by simp at hd; omega)
    cases hAx1 : (normAxis == 1) with
    | true =>
      rw [hAx1] at hOk; simp only [↓reduceIte, Except.ok.injEq, Prod.mk.injEq] at hOk
      rcases hOk with ⟨rfl, rfl⟩
      exact hPushY hPB.1 (fun v hv => hPB.2.2 _ (hPW.2.2 _ (hPNorm.2.2 _ hv)))
        (.conv 1 1 1 1 #[0, 0, 0, 0] 1 true .none) #[nextId, nextId + 1, nextId + 2]
        (fun v hv => by simp at hv; rcases hv with rfl | rfl | rfl
                        · exact hPB.2.2 _ (hPW.2.2 _ hPNorm.2.1)
                        · exact hPB.2.2 _ hPW.2.1
                        · exact hPB.2.1)
    | false =>
      rw [hAx1] at hOk; simp only [Bool.false_eq_true, ↓reduceIte] at hOk
      cases allowViewOps with
      | false => simp at hOk
      | true =>
        simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hOk
        cases hBad : (xTy.shape.size != 4 || xTy.shape[2]! != 1 || xTy.shape[1]! != xTy.shape[3]! || decide (normDim > 128)) with
        | true => rw [hBad] at hOk; cases hOk
        | false =>
          rw [hBad] at hOk; simp only [Bool.false_eq_true, ↓reduceIte, Except.ok.injEq, Prod.mk.injEq] at hOk
          rcases hOk with ⟨rfl, rfl⟩
          have hPTr1 := stepInv_pushFresh hPB.1 hUsedLt s!"{ndName}_tr1"
            (.transpose #[0, 3, 2, 1]) #[nextId] xTy
            (fun v hv => by simp at hv; subst hv; exact hPB.2.2 _ (hPW.2.2 _ hPNorm.2.1)) hxPos
          have hPAff := stepInv_pushFresh hPTr1.1 hUsedLt s!"{ndName}_aff"
            (.conv 1 1 1 1 #[0, 0, 0, 0] 1 true .none)
            (#[nextId, nextId + 1, nextId + 2].set! 0 (nextId + 3)) xTy
            (fun v hv => by
              simp at hv; rcases hv with rfl | rfl | rfl
              · exact hPTr1.2.1
              · exact hPTr1.2.2 _ (hPB.2.2 _ hPW.2.1)
              · exact hPTr1.2.2 _ hPB.2.1) hxPos
          exact hPushY hPAff.1 (fun v hv => hPAff.2.2 _ (hPTr1.2.2 _ (hPB.2.2 _ (hPW.2.2 _ (hPNorm.2.2 _ hv)))))
            (.transpose #[0, 3, 2, 1]) #[nextId + 4]
            (fun v hv => by simp at hv; subst hv; exact hPAff.2.1)
  | none =>
    rw [hFB] at hOk; dsimp only at hOk
    cases hAx1 : (normAxis == 1) with
    | true =>
      rw [hAx1] at hOk; simp only [↓reduceIte, Except.ok.injEq, Prod.mk.injEq] at hOk
      rcases hOk with ⟨rfl, rfl⟩
      exact hPushY hPW.1 (fun v hv => hPW.2.2 _ (hPNorm.2.2 _ hv))
        (.conv 1 1 1 1 #[0, 0, 0, 0] 1 false .none) #[nextId, nextId + 1]
        (fun v hv => by simp at hv; rcases hv with rfl | rfl
                        · exact hPW.2.2 _ hPNorm.2.1
                        · exact hPW.2.1)
    | false =>
      rw [hAx1] at hOk; simp only [Bool.false_eq_true, ↓reduceIte] at hOk
      cases allowViewOps with
      | false => simp at hOk
      | true =>
        simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hOk
        cases hBad : (xTy.shape.size != 4 || xTy.shape[2]! != 1 || xTy.shape[1]! != xTy.shape[3]! || decide (normDim > 128)) with
        | true => rw [hBad] at hOk; cases hOk
        | false =>
          rw [hBad] at hOk; simp only [Bool.false_eq_true, ↓reduceIte, Except.ok.injEq, Prod.mk.injEq] at hOk
          rcases hOk with ⟨rfl, rfl⟩
          have hPTr1 := stepInv_pushFresh hPW.1 hUsedLt s!"{ndName}_tr1"
            (.transpose #[0, 3, 2, 1]) #[nextId] xTy
            (fun v hv => by simp at hv; subst hv; exact hPW.2.2 _ hPNorm.2.1) hxPos
          have hPAff := stepInv_pushFresh hPTr1.1 hUsedLt s!"{ndName}_aff"
            (.conv 1 1 1 1 #[0, 0, 0, 0] 1 false .none)
            (#[nextId, nextId + 1].set! 0 (nextId + 2)) xTy
            (fun v hv => by
              simp at hv; rcases hv with rfl | rfl
              · exact hPTr1.2.1
              · exact hPTr1.2.2 _ hPW.2.1) hxPos
          exact hPushY hPAff.1 (fun v hv => hPAff.2.2 _ (hPTr1.2.2 _ (hPW.2.2 _ (hPNorm.2.2 _ hv))))
            (.transpose #[0, 3, 2, 1]) #[nextId + 3]
            (fun v hv => by simp at hv; subst hv; exact hPAff.2.1)

private theorem decomposeLayerNormTail_preserves_inv
    {g : SSAGraph HighOp} {k : Nat} {usedOrig : List ValueId}
    {nodes : Array (IRNode CoreOp)} {aliases : Array (ValueId × ValueId)} {nextId : Nat}
    (hInv : DecompStepInv g k usedOrig nodes aliases nextId)
    (hUsedLt : ∀ v ∈ usedOrig, v < g.nextValueId)
    (allowViewOps : Bool) (ndName ctx : String) (xTy : TensorType)
    (hxPos : ∀ d ∈ xTy.shape.toList, d > 0)
    (ax : Int) (normAxis normDim : Nat) (hNormPos : normDim > 0)
    (hasScale hasBias : Bool) (inps : Array ValueId)
    (hInpAt : ∀ idx, idx < inps.size → inps[idx]! ∈ (decompGraph g nodes).allDefinedIds)
    (yOut : ValueId × TensorType)
    (hyLt : yOut.1 < g.nextValueId)
    (hyNotPrior : yOut.1 ∉ g.allDefinedIdsBefore k)
    (hyNotUsed : yOut.1 ∉ usedOrig)
    (hyPos : ∀ d ∈ yOut.2.shape.toList, d > 0)
    (diffId invStdId : ValueId)
    (hDiffDef : diffId ∈ (decompGraph g nodes).allDefinedIds)
    (hInvDef : invStdId ∈ (decompGraph g nodes).allDefinedIds)
    (outNodes' : Array (IRNode CoreOp)) (nextId' : ValueId)
    (hOk : decomposeLayerNormTail g.inputs allowViewOps ndName ctx xTy ax normAxis normDim
      hasScale hasBias inps yOut diffId invStdId nodes nextId = .ok (outNodes', nextId')) :
    DecompStepInv g k (usedOrig ++ [yOut.1]) outNodes' aliases nextId' ∧
    yOut.1 ∈ (decompGraph g outNodes').allDefinedIds ∧
    (∀ v ∈ (decompGraph g nodes).allDefinedIds, v ∈ (decompGraph g outNodes').allDefinedIds) := by
  unfold decomposeLayerNormTail at hOk
  dsimp only [bind, Except.bind, pure, Except.pure, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw] at hOk
  have hPushY := fun {ns : Array (IRNode CoreOp)} {nid : Nat}
      (hI : DecompStepInv g k usedOrig ns aliases nid)
      (hL : ∀ v ∈ (decompGraph g nodes).allDefinedIds, v ∈ (decompGraph g ns).allDefinedIds)
      (opF : CoreOp) (inpsF : Array ValueId)
      (hInF : ∀ v ∈ inpsF.toList, v ∈ (decompGraph g ns).allDefinedIds) => by
    have hP := stepInv_pushOrig hI ndName opF inpsF #[yOut] hInF (by simp)
      (fun o ho => by simp at ho; subst ho; exact hyLt)
      (fun o ho => by simp at ho; subst ho; exact hyNotPrior)
      (fun o ho => by simp at ho; subst ho; exact hyNotUsed)
      (fun o ho d hd => by simp at ho; subst ho; exact hyPos d hd)
    exact And.intro hP.1 (And.intro (hP.2.1 yOut (by simp)) (fun v hv => hP.2.2 v (hL v hv)))
  cases hasScale with
  | false =>
    simp only [Bool.not_false, ↓reduceIte, Except.ok.injEq, Prod.mk.injEq] at hOk
    rcases hOk with ⟨rfl, rfl⟩
    exact hPushY hInv (fun _ h => h) (.elemBinary .mul) #[diffId, invStdId]
      (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hDiffDef; exact hInvDef)
  | true =>
    simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hOk
    by_cases hIn2 : inps.size < 2
    · simp only [hIn2, ↓reduceIte] at hOk; cases hOk
    · simp only [hIn2, ↓reduceIte] at hOk
      have hScaleDef : inps[1]! ∈ (decompGraph g nodes).allDefinedIds := hInpAt 1 (by omega)
      cases hScConst : lookupConstInCore nodes inps[1]! with
      | some scPair =>
        rw [hScConst] at hOk; dsimp only at hOk
        cases hScSh : isValidLayerNormParamShape scPair.1.shape normAxis normDim xTy.shape.size with
        | false => rw [hScSh] at hOk; cases hOk
        | true =>
          rw [hScSh] at hOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hOk
          cases hScF64 : scPair.2.toFloat64? with
          | none => rw [hScF64] at hOk; cases hOk
          | some scVals =>
            rw [hScF64] at hOk; dsimp only at hOk
            cases hScLen : (scVals.size != 1 && scVals.size != normDim) with
            | true => rw [hScLen] at hOk; cases hOk
            | false =>
              rw [hScLen] at hOk; simp only [Bool.false_eq_true, ↓reduceIte] at hOk
              have hConstAffine : ∀ (biasVals? : Option (Array Float)) (bUnif? : Option Float),
                  (match arrayUniformFloat scVals, bUnif? with
                   | some sVal, some bVal =>
                     if (sVal == 1.0 && bVal == 0.0) = true then
                       Except.ok (nodes.push { name := ndName, op := .elemBinary .mul, inputs := #[diffId, invStdId], outputs := #[yOut] }, nextId)
                     else
                       Except.bind (validateFiniteFp16ScalarIR sVal ctx) fun sBits =>
                       Except.bind (validateFiniteFp16ScalarIR bVal ctx) fun bBits =>
                         Except.ok ((nodes.push { name := s!"{ndName}_norm", op := .elemBinary .mul, inputs := #[diffId, invStdId], outputs := #[(nextId, xTy)] }).push
                           { name := ndName, op := .elemUnary (if (bVal == 0.0) = true then .scalarMul sBits else if (sVal == 1.0) = true then .scalarAdd bBits else .linearAct sBits bBits), inputs := #[nextId], outputs := #[yOut] }, nextId + 1)
                   | _, _ =>
                     decomposeLayerNormNonUniformConst allowViewOps ndName ctx xTy normAxis normDim
                       yOut diffId invStdId scVals biasVals? nodes nextId) = .ok (outNodes', nextId') →
                  DecompStepInv g k (usedOrig ++ [yOut.1]) outNodes' aliases nextId' ∧
                  yOut.1 ∈ (decompGraph g outNodes').allDefinedIds ∧
                  (∀ v ∈ (decompGraph g nodes).allDefinedIds, v ∈ (decompGraph g outNodes').allDefinedIds) := by
                intro biasVals? bUnif? hAff
                cases hScU : arrayUniformFloat scVals with
                | none =>
                  rw [hScU] at hAff
                  exact decomposeLayerNormNonUniformConst_preserves_inv hInv hUsedLt allowViewOps ndName ctx
                    xTy hxPos normAxis normDim hNormPos yOut hyLt hyNotPrior hyNotUsed hyPos
                    diffId invStdId hDiffDef hInvDef scVals biasVals? outNodes' nextId' hAff
                | some sVal =>
                  rw [hScU] at hAff
                  cases bUnif? with
                  | none =>
                    exact decomposeLayerNormNonUniformConst_preserves_inv hInv hUsedLt allowViewOps ndName ctx
                      xTy hxPos normAxis normDim hNormPos yOut hyLt hyNotPrior hyNotUsed hyPos
                      diffId invStdId hDiffDef hInvDef scVals biasVals? outNodes' nextId' hAff
                  | some bVal =>
                    dsimp only at hAff
                    cases hIdSB : (sVal == 1.0 && bVal == 0.0) with
                    | true =>
                      rw [hIdSB] at hAff; simp only [↓reduceIte, Except.ok.injEq, Prod.mk.injEq] at hAff
                      rcases hAff with ⟨rfl, rfl⟩
                      exact hPushY hInv (fun _ h => h) (.elemBinary .mul) #[diffId, invStdId]
                        (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hDiffDef; exact hInvDef)
                    | false =>
                      rw [hIdSB] at hAff; simp only [Bool.false_eq_true, ↓reduceIte] at hAff
                      cases hSB : validateFiniteFp16ScalarIR sVal ctx with
                      | error _ => rw [hSB] at hAff; cases hAff
                      | ok sBits =>
                        rw [hSB] at hAff; dsimp only [Except.bind] at hAff
                        cases hBB : validateFiniteFp16ScalarIR bVal ctx with
                        | error _ => rw [hBB] at hAff; cases hAff
                        | ok bBits =>
                          rw [hBB] at hAff; simp only [Except.ok.injEq, Prod.mk.injEq] at hAff
                          rcases hAff with ⟨rfl, rfl⟩
                          have hPNorm := stepInv_pushFresh hInv hUsedLt s!"{ndName}_norm"
                            (.elemBinary .mul) #[diffId, invStdId] xTy
                            (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hDiffDef; exact hInvDef) hxPos
                          exact hPushY hPNorm.1 hPNorm.2.2 _ #[nextId]
                            (fun v hv => by simp at hv; subst hv; exact hPNorm.2.1)
              cases hBiasId : (if hasBias && decide (inps.size ≥ 3) then some inps[2]! else none) with
              | none =>
                rw [hBiasId] at hOk; dsimp only at hOk
                exact hConstAffine none (some 0.0) (by
                  cases h1 : arrayUniformFloat scVals with
                  | none => simpa [h1] using hOk
                  | some sVal => simpa [h1, Except.bind] using hOk)
              | some bId =>
                rw [hBiasId] at hOk; dsimp only at hOk
                cases hBConst : lookupConstInCore nodes bId with
                | none => rw [hBConst] at hOk; cases hOk
                | some bPair =>
                  rw [hBConst] at hOk; dsimp only at hOk
                  cases hBSh : isValidLayerNormParamShape bPair.1.shape normAxis normDim xTy.shape.size with
                  | false => rw [hBSh] at hOk; cases hOk
                  | true =>
                    rw [hBSh] at hOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hOk
                    cases hBF64 : bPair.2.toFloat64? with
                    | none => rw [hBF64] at hOk; cases hOk
                    | some bv =>
                      rw [hBF64] at hOk; dsimp only at hOk
                      cases hBLen : (bv.size != 1 && bv.size != normDim) with
                      | true => rw [hBLen] at hOk; cases hOk
                      | false =>
                        rw [hBLen] at hOk; simp only [Bool.false_eq_true, ↓reduceIte] at hOk
                        exact hConstAffine (some bv) (arrayUniformFloat bv) (by
                          cases h1 : arrayUniformFloat scVals <;> cases h2 : arrayUniformFloat bv <;>
                            simpa [h1, h2, Except.bind] using hOk)
      | none =>
        rw [hScConst] at hOk; dsimp only at hOk
        cases hScTy : lookupTypeInCore g.inputs nodes inps[1]! with
        | none => rw [hScTy] at hOk; cases hOk
        | some scTy =>
          rw [hScTy] at hOk; dsimp only at hOk
          cases hScSh : isValidLayerNormParamShape scTy.shape normAxis normDim xTy.shape.size with
          | false => rw [hScSh] at hOk; cases hOk
          | true =>
            rw [hScSh] at hOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hOk
            have hPNorm := stepInv_pushFresh hInv hUsedLt s!"{ndName}_norm"
              (.elemBinary .mul) #[diffId, invStdId] xTy
              (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hDiffDef; exact hInvDef) hxPos
            by_cases hBiasRun : hasBias = true ∧ inps.size ≥ 3
            · have hCond : (hasBias && decide (inps.size ≥ 3)) = true := by simp [hBiasRun.1, hBiasRun.2]
              simp only [hCond, ↓reduceIte] at hOk
              have hBDef : inps[2]! ∈ (decompGraph g nodes).allDefinedIds := hInpAt 2 (by omega)
              cases hBTy : lookupTypeInCore g.inputs _ inps[2]! with
              | none => rw [hBTy] at hOk; cases hOk
              | some bTy =>
                rw [hBTy] at hOk; dsimp only at hOk
                cases hBSh : isValidLayerNormParamShape bTy.shape normAxis normDim xTy.shape.size with
                | false => rw [hBSh] at hOk; cases hOk
                | true =>
                  rw [hBSh] at hOk; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Except.ok.injEq, Prod.mk.injEq] at hOk
                  rcases hOk with ⟨rfl, rfl⟩
                  have hPScaled := stepInv_pushFresh hPNorm.1 hUsedLt s!"{ndName}_scaled"
                    (.elemBinary .mul) #[nextId, inps[1]!] xTy
                    (fun v hv => by simp at hv; rcases hv with rfl | rfl
                                    · exact hPNorm.2.1
                                    · exact hPNorm.2.2 _ hScaleDef) hxPos
                  exact hPushY hPScaled.1 (fun v hv => hPScaled.2.2 _ (hPNorm.2.2 _ hv))
                    (.elemBinary .add) #[nextId + 1, inps[2]!]
                    (fun v hv => by simp at hv; rcases hv with rfl | rfl
                                    · exact hPScaled.2.1
                                    · exact hPScaled.2.2 _ (hPNorm.2.2 _ hBDef))
            · have hCond : (hasBias && decide (inps.size ≥ 3)) = false := by cases hasBias <;> simp_all
              simp only [hCond, Bool.false_eq_true, ↓reduceIte, Except.ok.injEq, Prod.mk.injEq] at hOk
              rcases hOk with ⟨rfl, rfl⟩
              exact hPushY hPNorm.1 hPNorm.2.2 (.elemBinary .mul) #[nextId, inps[1]!]
                (fun v hv => by simp at hv; rcases hv with rfl | rfl
                                · exact hPNorm.2.1
                                · exact hPNorm.2.2 _ hScaleDef)

/-- Single-node invariant preservation for the REAL `Vane.IR.decomposeNode` across ALL 8 `HighOp`
    constructors (`.core`, `.gelu`, `.pow` incl. `.identity`, `.batchNorm`, `.softmax` incl.
    channel-only `W > 1`, `.layerNorm` across all 6 scale/bias branches & 1..3 outputs,
    `.rmsNorm` across all 4 scale branches, and `.split` 1-way & multi-way)! -/
private theorem decomposeNode_preserves_inv
    (g : SSAGraph HighOp) (allowViewOps : Bool)
    (hwf : WellFormed g) (hArity : ValidHighOpOutputArity g)
    (k : Nat) (hk : k < g.nodes.size) (st st' : DecomposeState)
    (hInv : DecompStepInv g k [] st.outNodes st.aliases st.nextId)
    (hStep : decomposeNode g.inputs allowViewOps st g.nodes[k] = .ok st') :
    DecompStepInv g (k + 1) [] st'.outNodes st'.aliases st'.nextId := by
  let nd := g.nodes[k]
  have hAr := hArity nd (Array.getElem_mem_toList hk)
  have ⟨hOutNd, hOutLt, hOutNotPrior⟩ := wellFormed_node_outputs_fresh hwf hk
  have hInpsDef : ∀ v ∈ (nd.inputs.map (lookupAlias st.aliases)).toList,
      v ∈ (decompGraph g st.outNodes).allDefinedIds := by
    intro v hv
    rw [Array.toList_map, List.mem_map] at hv
    obtain ⟨u, hu, rfl⟩ := hv
    exact hInv.priorCovered u (hwf.defBeforeUse k hk u hu)
  have hInpAt : ∀ idx (hIdx : idx < (nd.inputs.map (lookupAlias st.aliases)).size),
      (nd.inputs.map (lookupAlias st.aliases))[idx]! ∈ (decompGraph g st.outNodes).allDefinedIds := by
    intro idx hIdx
    rw [getElem!_pos _ idx hIdx]
    exact hInpsDef _ (Array.getElem_mem_toList hIdx)
  have hOrigNotAliasKey : ∀ o ∈ nd.outputs.toList, ∀ p ∈ st.aliases.toList, p.1 ≠ o.1 := by
    intro o ho p hp hEq
    exact hOutNotPrior o ho (hEq ▸ (hInv.aliasKeyPrior p hp).1)
  have hNotKeyLookup : ∀ o ∈ nd.outputs.toList, lookupAlias st.aliases o.1 = o.1 :=
    fun o ho => lookupAlias_of_not_key st.aliases o.1 (hOrigNotAliasKey o ho)
  -- Helper: pushing a single original output at index `idx`
  have hPushOrigAt := fun {usedOrig : List ValueId} {nodes : Array (IRNode CoreOp)} {nextId : Nat}
      (hI : DecompStepInv g k usedOrig nodes st.aliases nextId)
      (name : String) (op : CoreOp) (inputs : Array ValueId) (idx : Nat) (hIdx : idx < nd.outputs.size)
      (hIn : ∀ v ∈ inputs.toList, v ∈ (decompGraph g nodes).allDefinedIds)
      (hNotUsed : nd.outputs[idx].1 ∉ usedOrig) =>
    stepInv_pushOrig hI name op inputs #[nd.outputs[idx]] hIn (by simp)
      (fun o ho => by simp at ho; subst ho; exact hOutLt _ (Array.getElem_mem_toList hIdx))
      (fun o ho => by simp at ho; subst ho; exact hOutNotPrior _ (Array.getElem_mem_toList hIdx))
      (fun o ho => by simp at ho; subst ho; exact hNotUsed)
      (fun o ho d hd => by simp at ho; subst ho; exact hwf.nodeOutputsPositive k hk _ (Array.getElem_mem_toList hIdx) d hd)
  -- Unfold `decomposeNode` and split on `nd.op`
  change decomposeNode g.inputs allowViewOps st nd = .ok st' at hStep
  unfold decomposeNode at hStep
  dsimp only [bind, Except.bind, pure, Except.pure, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw] at hStep
  cases hOp : nd.op with
  | core c =>
    rw [hOp] at hStep; simp only [Except.ok.injEq] at hStep; subst hStep
    have hP := stepInv_pushOrig hInv nd.name c (nd.inputs.map (lookupAlias st.aliases)) nd.outputs
      hInpsDef hOutNd hOutLt hOutNotPrior (fun _ _ h => by cases h) (hwf.nodeOutputsPositive k hk)
    exact stepInv_finish hk hP.1 (fun v hv => by simpa [IRNode.outputIds] using hv) (fun v hv => by
      unfold IRNode.outputIds at hv; rw [List.mem_map] at hv; obtain ⟨o, ho, rfl⟩ := hv
      rw [hNotKeyLookup o ho]; exact hP.2.1 o ho)
  | gelu mode =>
    cases mode
    rw [hOp] at hStep; dsimp only at hStep
    split at hStep; · cases hStep
    simp only [Except.ok.injEq] at hStep; subst hStep
    have hP := stepInv_pushOrig hInv nd.name (.lut (.lut .gelu)) (nd.inputs.map (lookupAlias st.aliases)) nd.outputs
      hInpsDef hOutNd hOutLt hOutNotPrior (fun _ _ h => by cases h) (hwf.nodeOutputsPositive k hk)
    exact stepInv_finish hk hP.1 (fun v hv => by simpa [IRNode.outputIds] using hv) (fun v hv => by
      unfold IRNode.outputIds at hv; rw [List.mem_map] at hv; obtain ⟨o, ho, rfl⟩ := hv
      rw [hNotKeyLookup o ho]; exact hP.2.1 o ho)
  | pow kind =>
    rw [hOp] at hStep; dsimp only at hStep
    split at hStep; · cases hStep
    rename_i hInOut
    have hIn1 : (nd.inputs.map (lookupAlias st.aliases)).size = 1 := by
      simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Classical.not_not] at hInOut; exact hInOut.1
    have hOut1 : nd.outputs.size = 1 := by
      simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Classical.not_not] at hInOut; exact hInOut.2
    have hxDef : (nd.inputs.map (lookupAlias st.aliases))[0]! ∈ (decompGraph g st.outNodes).allDefinedIds :=
      hInpAt 0 (by omega)
    cases kind with
    | identity =>
      simp only [Except.ok.injEq] at hStep; subst hStep
      have h0 : 0 < nd.outputs.size := by omega
      have hGet0 : nd.outputs[0]! = nd.outputs[0] := getElem!_pos _ 0 h0
      rw [hGet0]
      have hO0Mem : nd.outputs[0] ∈ nd.outputs.toList := Array.getElem_mem_toList h0
      let outId := nd.outputs[0].1
      let xId := (nd.inputs.map (lookupAlias st.aliases))[0]!
      have hOutNotDef : outId ∉ (decompGraph g st.outNodes).allDefinedIds := by
        intro hmem
        rcases hInv.defBound outId hmem with h1 | h2 | ⟨h3, _⟩
        · exact hOutNotPrior _ hO0Mem h1
        · cases h2
        · exact Nat.not_lt_of_ge h3 (hOutLt _ hO0Mem)
      have hNe : outId ≠ xId := fun hEq => hOutNotDef (hEq ▸ hxDef)
      have hXNotKey : ∀ p ∈ st.aliases.toList, p.1 ≠ xId :=
        fun p hp hEq => (hInv.aliasKeyPrior p hp).2 (hEq ▸ hxDef)
      have hAliasRes := lookupAlias_push_fresh st.aliases outId xId
        (hOrigNotAliasKey _ hO0Mem) hXNotKey hNe
      have hBefSucc : g.allDefinedIdsBefore (k + 1) = g.allDefinedIdsBefore k ++ nd.outputIds := by
        unfold SSAGraph.allDefinedIdsBefore
        have hTake : g.nodes.toList.take (k + 1) = g.nodes.toList.take k ++ [nd] := by
          rw [List.take_succ_eq_append_getElem (by simpa using hk)]; rfl
        rw [hTake, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.append_assoc]
      have hOutList : nd.outputs.toList = [nd.outputs[0]] := by
        cases hl : nd.outputs.toList with
        | nil => have : nd.outputs.size = 0 := by rw [← Array.length_toList, hl]; rfl
                 omega
        | cons a tl =>
          cases tl with
          | nil =>
            have : nd.outputs[0] = a := by simp [← Array.getElem_toList, hl]
            rw [this]
          | cons b tl' =>
            have : nd.outputs.size ≥ 2 := by rw [← Array.length_toList, hl]; simp
            omega
      refine ⟨hInv.nextGe, hInv.wf, ?_, ?_, ?_, ?_⟩
      · intro v hv
        rcases hInv.defBound v hv with h1 | h2 | h3
        · exact Or.inl (by rw [hBefSucc]; exact List.mem_append_left _ h1)
        · cases h2
        · exact Or.inr (Or.inr h3)
      · intro p hp
        rw [Array.toList_push, List.mem_append, List.mem_singleton] at hp
        rcases hp with hOld | rfl
        · exact hInv.aliasTargetDef p hOld
        · exact hxDef
      · intro p hp
        rw [Array.toList_push, List.mem_append, List.mem_singleton] at hp
        rcases hp with hOld | rfl
        · have ⟨h1, h2⟩ := hInv.aliasKeyPrior p hOld
          exact ⟨by rw [hBefSucc]; exact List.mem_append_left _ h1, h2⟩
        · exact ⟨by rw [hBefSucc]; exact List.mem_append_right _ (by simp [IRNode.outputIds, hOutList]), hOutNotDef⟩
      · intro v hv
        rw [hBefSucc, List.mem_append] at hv
        rcases hv with hOld | hCur
        · have hVne : v ≠ outId := fun hEq => hOutNotPrior _ hO0Mem (by simpa [outId, hEq] using hOld)
          have hTgtNe : ∀ p ∈ st.aliases.toList, p.2 ≠ outId :=
            fun p hp hEq => hOutNotDef (hEq ▸ hInv.aliasTargetDef p hp)
          rw [hAliasRes.2 v hVne hTgtNe]
          exact hInv.priorCovered v hOld
        · simp only [IRNode.outputIds, hOutList, List.map_cons, List.map_nil, List.mem_singleton] at hCur
          rw [hCur, hAliasRes.1]
          exact hxDef
    | square =>
      simp only [Except.ok.injEq] at hStep; subst hStep
      have hP := stepInv_pushOrig hInv nd.name (.elemBinary .mul)
        #[(nd.inputs.map (lookupAlias st.aliases))[0]!, (nd.inputs.map (lookupAlias st.aliases))[0]!] nd.outputs
        (fun v hv => by simp at hv; rcases hv with rfl | rfl <;> exact hxDef)
        hOutNd hOutLt hOutNotPrior (fun _ _ h => by cases h) (hwf.nodeOutputsPositive k hk)
      exact stepInv_finish hk hP.1 (fun v hv => by simpa [IRNode.outputIds] using hv) (fun v hv => by
        unfold IRNode.outputIds at hv; rw [List.mem_map] at hv; obtain ⟨o, ho, rfl⟩ := hv
        rw [hNotKeyLookup o ho]; exact hP.2.1 o ho)
    | sqrt =>
      simp only [Except.ok.injEq] at hStep; subst hStep
      have hP := stepInv_pushOrig hInv nd.name (.lut (.lut .sqrt))
        #[(nd.inputs.map (lookupAlias st.aliases))[0]!] nd.outputs
        (fun v hv => by simp at hv; subst hv; exact hxDef)
        hOutNd hOutLt hOutNotPrior (fun _ _ h => by cases h) (hwf.nodeOutputsPositive k hk)
      exact stepInv_finish hk hP.1 (fun v hv => by simpa [IRNode.outputIds] using hv) (fun v hv => by
        unfold IRNode.outputIds at hv; rw [List.mem_map] at hv; obtain ⟨o, ho, rfl⟩ := hv
        rw [hNotKeyLookup o ho]; exact hP.2.1 o ho)
    | rsqrt =>
      simp only [Except.ok.injEq] at hStep; subst hStep
      have hP := stepInv_pushOrig hInv nd.name (.lut (.lut (.rsqrt 0.0)))
        #[(nd.inputs.map (lookupAlias st.aliases))[0]!] nd.outputs
        (fun v hv => by simp at hv; subst hv; exact hxDef)
        hOutNd hOutLt hOutNotPrior (fun _ _ h => by cases h) (hwf.nodeOutputsPositive k hk)
      exact stepInv_finish hk hP.1 (fun v hv => by simpa [IRNode.outputIds] using hv) (fun v hv => by
        unfold IRNode.outputIds at hv; rw [List.mem_map] at hv; obtain ⟨o, ho, rfl⟩ := hv
        rw [hNotKeyLookup o ho]; exact hP.2.1 o ho)
    | reciprocal =>
      simp only [Except.ok.injEq] at hStep; subst hStep
      have hP := stepInv_pushOrig hInv nd.name (.lut (.lut .inverse))
        #[(nd.inputs.map (lookupAlias st.aliases))[0]!] nd.outputs
        (fun v hv => by simp at hv; subst hv; exact hxDef)
        hOutNd hOutLt hOutNotPrior (fun _ _ h => by cases h) (hwf.nodeOutputsPositive k hk)
      exact stepInv_finish hk hP.1 (fun v hv => by simpa [IRNode.outputIds] using hv) (fun v hv => by
        unfold IRNode.outputIds at hv; rw [List.mem_map] at hv; obtain ⟨o, ho, rfl⟩ := hv
        rw [hNotKeyLookup o ho]; exact hP.2.1 o ho)
  | batchNorm eps =>
    rw [hOp] at hStep hAr; dsimp only at hStep hAr
    split at hStep; · cases hStep
    rename_i hInOut
    have hIn5 : (nd.inputs.map (lookupAlias st.aliases)).size ≥ 5 := by
      simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hInOut; omega
    have h0 : 0 < nd.outputs.size := by omega
    have hGet0 : nd.outputs[0]! = nd.outputs[0] := getElem!_pos _ 0 h0
    cases hTy : lookupTypeInCore g.inputs st.outNodes (nd.inputs.map (lookupAlias st.aliases))[0]! with
    | none => rw [hTy] at hStep; cases hStep
    | some xTy =>
      rw [hTy] at hStep; dsimp only at hStep
      split at hStep; · cases hStep
      rename_i hRank
      have hRank24 : xTy.shape.size = 4 ∨ xTy.shape.size = 2 := by
        simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, not_and, Classical.not_not] at hRank
        by_cases h4 : xTy.shape.size = 4
        · exact Or.inl h4
        · exact Or.inr (hRank h4)
      have hxTyPos := lookupTypeInCore_pos g st.outNodes hInv.wf _ xTy hTy
      have hcPos : xTy.shape[1]! > 0 := array_getElem!_pos_of_all_pos xTy.shape 1 hxTyPos (by omega)
      cases h1 : lookupConstInCore st.outNodes (nd.inputs.map (lookupAlias st.aliases))[1]! with
      | none => rw [h1] at hStep; cases hStep
      | some c1 =>
        rw [h1] at hStep; dsimp only at hStep
        cases hf1 : c1.2.toFloat64? with
        | none => rw [hf1] at hStep; cases hStep
        | some s1 =>
          rw [hf1] at hStep; dsimp only at hStep
          cases h2 : lookupConstInCore st.outNodes (nd.inputs.map (lookupAlias st.aliases))[2]! with
          | none => rw [h2] at hStep; cases hStep
          | some c2 =>
            rw [h2] at hStep; dsimp only at hStep
            cases hf2 : c2.2.toFloat64? with
            | none => rw [hf2] at hStep; cases hStep
            | some s2 =>
              rw [hf2] at hStep; dsimp only at hStep
              cases h3 : lookupConstInCore st.outNodes (nd.inputs.map (lookupAlias st.aliases))[3]! with
              | none => rw [h3] at hStep; cases hStep
              | some c3 =>
                rw [h3] at hStep; dsimp only at hStep
                cases hf3 : c3.2.toFloat64? with
                | none => rw [hf3] at hStep; cases hStep
                | some s3 =>
                  rw [hf3] at hStep; dsimp only at hStep
                  cases h4 : lookupConstInCore st.outNodes (nd.inputs.map (lookupAlias st.aliases))[4]! with
                  | none => rw [h4] at hStep; cases hStep
                  | some c4 =>
                    rw [h4] at hStep; dsimp only at hStep
                    cases hf4 : c4.2.toFloat64? with
                    | none => rw [hf4] at hStep; cases hStep
                    | some s4 =>
                      rw [hf4] at hStep; dsimp only at hStep
                      cases hBn : computeBNAffineF64 s1 s2 s3 s4 eps xTy.shape[1]! _ with
                      | error _ => rw [hBn] at hStep; cases hStep
                      | ok fb =>
                        rw [hBn] at hStep; dsimp only at hStep
                        simp only [Except.ok.injEq] at hStep; subst hStep
                        have hxDef := hInpAt 0 (by omega)
                        have hP1 := stepInv_pushFresh hInv (fun _ h => by cases h) s!"{nd.name}_w"
                          (.const (.f32 (buildDiagWeights xTy.shape[1]! fb.1))) #[]
                          { shape := #[xTy.shape[1]!, xTy.shape[1]!, 1, 1], dtype := .f32 }
                          (fun _ h => by simp at h)
                          (fun d hd => by simp at hd; rcases hd with rfl | rfl | rfl | rfl <;> omega)
                        have hP2 := stepInv_pushFresh hP1.1 (fun _ h => by cases h) s!"{nd.name}_b"
                          (.const (.f32 fb.2)) #[]
                          { shape := #[xTy.shape[1]!], dtype := .f32 }
                          (fun _ h => by simp at h)
                          (fun d hd => by simp at hd; omega)
                        have hP3 := hPushOrigAt hP2.1 nd.name (.conv 1 1 1 1 #[0, 0, 0, 0] 1 true .none)
                          #[(nd.inputs.map (lookupAlias st.aliases))[0]!, st.nextId, st.nextId + 1] 0 h0
                          (fun v hv => by
                            simp at hv; rcases hv with rfl | rfl | rfl
                            · exact hP2.2.2 _ (hP1.2.2 _ hxDef)
                            · exact hP2.2.2 _ hP1.2.1
                            · exact hP2.2.1)
                          (by simp)
                        rw [hGet0]
                        have hOutList : nd.outputs.toList = [nd.outputs[0]] := by
                          cases hl : nd.outputs.toList with
                          | nil => have : nd.outputs.size = 0 := by rw [← Array.length_toList, hl]; rfl
                                   omega
                          | cons a tl =>
                            cases tl with
                            | nil =>
                              have : nd.outputs[0] = a := by simp [← Array.getElem_toList, hl]
                              rw [this]
                            | cons b tl' =>
                              have : nd.outputs.size ≥ 2 := by rw [← Array.length_toList, hl]; simp
                              omega
                        exact stepInv_finish hk hP3.1
                          (fun v hv => by
                            simp only [List.nil_append, List.map_cons, List.map_nil, List.mem_singleton] at hv
                            subst hv
                            exact List.mem_map.mpr ⟨nd.outputs[0], Array.getElem_mem_toList h0, rfl⟩)
                          (fun v hv => by
                            unfold IRNode.outputIds at hv
                            change v ∈ nd.outputs.toList.map Prod.fst at hv
                            rw [hOutList] at hv
                            simp only [List.map_cons, List.map_nil, List.mem_singleton] at hv
                            subst hv
                            rw [hNotKeyLookup _ (Array.getElem_mem_toList h0)]
                            exact hP3.2.1 _ (by simp))
  | softmax axis legacyOpset =>
    rw [hOp] at hStep; dsimp only at hStep
    cases hInOut : ((nd.inputs.map (lookupAlias st.aliases)).size != 1 || nd.outputs.size != 1) with
    | true => rw [hInOut] at hStep; cases hStep
    | false =>
    rw [hInOut] at hStep; simp only [Bool.false_eq_true, ↓reduceIte] at hStep
    have hIn1 : (nd.inputs.map (lookupAlias st.aliases)).size = 1 := by
      simp only [Bool.or_eq_false_iff, bne_eq_false_iff_eq] at hInOut; exact hInOut.1
    have hOut1 : nd.outputs.size = 1 := by
      simp only [Bool.or_eq_false_iff, bne_eq_false_iff_eq] at hInOut; exact hInOut.2
    have h0 : 0 < nd.outputs.size := by omega
    have hGet0 : nd.outputs[0]! = nd.outputs[0] := getElem!_pos _ 0 h0
    cases hTy : lookupTypeInCore g.inputs st.outNodes (nd.inputs.map (lookupAlias st.aliases))[0]! with
    | none => rw [hTy] at hStep; cases hStep
    | some xTy =>
      rw [hTy] at hStep; dsimp only at hStep
      cases hRedRes : resolveSoftmaxRedAxes xTy.shape.size axis legacyOpset _ with
      | error _ => rw [hRedRes] at hStep; cases hStep
      | ok redAxes =>
        rw [hRedRes] at hStep; dsimp only at hStep
        have hxDef := hInpAt 0 (by omega)
        have hxPos := lookupTypeInCore_pos g st.outNodes hInv.wf _ xTy hTy
        have hRedPos := softmaxRedShape_pos xTy.shape redAxes hxPos
        have hP1 := stepInv_pushFresh hInv (fun _ h => by cases h) s!"{nd.name}_max"
          (.reduce .max redAxes true 0) #[(nd.inputs.map (lookupAlias st.aliases))[0]!]
          { shape := softmaxRedShape xTy.shape redAxes, dtype := xTy.dtype }
          (fun v hv => by simp at hv; subst hv; exact hxDef) hRedPos
        have hP2 := stepInv_pushFresh hP1.1 (fun _ h => by cases h) s!"{nd.name}_sub"
          (.elemBinary .sub) #[(nd.inputs.map (lookupAlias st.aliases))[0]!, st.nextId] xTy
          (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hP1.2.2 _ hxDef; exact hP1.2.1) hxPos
        have hP3 := stepInv_pushFresh hP2.1 (fun _ h => by cases h) s!"{nd.name}_exp"
          (.lut (.lut .exp)) #[st.nextId + 1] xTy
          (fun v hv => by simp at hv; subst hv; exact hP2.2.1) hxPos
        have hOutList : nd.outputs.toList = [nd.outputs[0]] := by
          cases hl : nd.outputs.toList with
          | nil => have : nd.outputs.size = 0 := by rw [← Array.length_toList, hl]; rfl
                   omega
          | cons a tl =>
            cases tl with
            | nil =>
              have : nd.outputs[0] = a := by simp [← Array.getElem_toList, hl]
              rw [this]
            | cons b tl' =>
              have : nd.outputs.size ≥ 2 := by rw [← Array.length_toList, hl]; simp
              omega
        split at hStep
        · simp only [Except.ok.injEq] at hStep; subst hStep
          have hP4 := stepInv_pushFresh hP3.1 (fun _ h => by cases h) s!"{nd.name}_sum"
            (.reduce .sum redAxes true 0) #[st.nextId + 2]
            { shape := softmaxRedShape xTy.shape redAxes, dtype := xTy.dtype }
            (fun v hv => by simp at hv; subst hv; exact hP3.2.1) hRedPos
          have hP5 := stepInv_pushFresh hP4.1 (fun _ h => by cases h) s!"{nd.name}_invsum"
            (.lut (.lut .inverse)) #[st.nextId + 3]
            { shape := softmaxRedShape xTy.shape redAxes, dtype := xTy.dtype }
            (fun v hv => by simp at hv; subst hv; exact hP4.2.1) hRedPos
          have hP6 := hPushOrigAt hP5.1 nd.name (.elemBinary .mul) #[st.nextId + 2, st.nextId + 4] 0 h0
            (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hP5.2.2 _ (hP4.2.2 _ hP3.2.1); exact hP5.2.1)
            (by simp)
          rw [hGet0]
          exact stepInv_finish hk hP6.1
            (fun v hv => by
              simp only [List.nil_append, List.map_cons, List.map_nil, List.mem_singleton] at hv
              subst hv
              exact List.mem_map.mpr ⟨nd.outputs[0], Array.getElem_mem_toList h0, rfl⟩)
            (fun v hv => by
              unfold IRNode.outputIds at hv
              change v ∈ nd.outputs.toList.map Prod.fst at hv
              rw [hOutList] at hv
              simp only [List.map_cons, List.map_nil, List.mem_singleton] at hv
              subst hv; rw [hNotKeyLookup _ (Array.getElem_mem_toList h0)]; exact hP6.2.1 _ (by simp))
        · simp only [Except.ok.injEq] at hStep; subst hStep
          have hP4 := stepInv_pushFresh hP3.1 (fun _ h => by cases h) s!"{nd.name}_invsum"
            (.reduce .invSum redAxes true 0) #[st.nextId + 2]
            { shape := softmaxRedShape xTy.shape redAxes, dtype := xTy.dtype }
            (fun v hv => by simp at hv; subst hv; exact hP3.2.1) hRedPos
          have hP5 := hPushOrigAt hP4.1 nd.name (.elemBinary .mul) #[st.nextId + 2, st.nextId + 3] 0 h0
            (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hP4.2.2 _ hP3.2.1; exact hP4.2.1)
            (by simp)
          rw [hGet0]
          exact stepInv_finish hk hP5.1
            (fun v hv => by
              simp only [List.nil_append, List.map_cons, List.map_nil, List.mem_singleton] at hv
              subst hv
              exact List.mem_map.mpr ⟨nd.outputs[0], Array.getElem_mem_toList h0, rfl⟩)
            (fun v hv => by
              unfold IRNode.outputIds at hv
              change v ∈ nd.outputs.toList.map Prod.fst at hv
              rw [hOutList] at hv
              simp only [List.map_cons, List.map_nil, List.mem_singleton] at hv
              subst hv; rw [hNotKeyLookup _ (Array.getElem_mem_toList h0)]; exact hP5.2.1 _ (by simp))
  | layerNorm axis eps hasScale hasBias =>
    rw [hOp] at hStep hAr; dsimp only at hStep hAr
    cases hInOut : (decide ((nd.inputs.map (lookupAlias st.aliases)).size < 1) || decide (nd.outputs.size < 1)) with
    | true => rw [hInOut] at hStep; cases hStep
    | false =>
    rw [hInOut] at hStep; simp only [Bool.false_eq_true, ↓reduceIte] at hStep
    have hIn1 : (nd.inputs.map (lookupAlias st.aliases)).size ≥ 1 := by
      simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not] at hInOut; omega
    have hOut1 : nd.outputs.size ≥ 1 := by
      simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not] at hInOut; omega
    have h0 : 0 < nd.outputs.size := by omega
    have hGet0 : nd.outputs[0]! = nd.outputs[0] := getElem!_pos _ 0 h0
    cases hTy : lookupTypeInCore g.inputs st.outNodes (nd.inputs.map (lookupAlias st.aliases))[0]! with
    | none => rw [hTy] at hStep; cases hStep
    | some xTy =>
      rw [hTy] at hStep; dsimp only at hStep
      have hxDef := hInpAt 0 (by omega)
      have hxPos := lookupTypeInCore_pos g st.outNodes hInv.wf _ xTy hTy
      have hLnCfgPos : ∀ ctx redAxes redShape normAxis normDim,
          resolveLayerNormConfig xTy.shape axis ctx = .ok (redAxes, redShape, normAxis, normDim) →
          (∀ d ∈ redShape.toList, d > 0) ∧ normDim > 0 := by
        intro ctx redAxes redShape normAxis normDim hCfg
        unfold resolveLayerNormConfig at hCfg
        dsimp only [bind, Except.bind, pure, Except.pure, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw] at hCfg
        by_cases h3 : xTy.shape.size = 3
        · simp only [h3, beq_self_eq_true, ↓reduceIte] at hCfg
          by_cases ha : (if axis < 0 then axis + ↑(3 : Nat) else axis) = 2
          · simp only [ha, beq_self_eq_true, ↓reduceIte, Except.ok.injEq, Prod.mk.injEq] at hCfg
            rcases hCfg with ⟨_, rfl, _, rfl⟩
            have h0p := array_getElem!_pos_of_all_pos xTy.shape 0 hxPos (by omega)
            have h1p := array_getElem!_pos_of_all_pos xTy.shape 1 hxPos (by omega)
            have h2p := array_getElem!_pos_of_all_pos xTy.shape 2 hxPos (by omega)
            exact ⟨fun d hd => by simp at hd; rcases hd with rfl | rfl | rfl <;> omega, h2p⟩
          · simp only [beq_eq_false_iff_ne.mpr ha, Bool.false_eq_true, ↓reduceIte] at hCfg
            cases hCfg
        · simp only [beq_eq_false_iff_ne.mpr h3, Bool.false_eq_true, ↓reduceIte] at hCfg
          by_cases h2 : xTy.shape.size = 2
          · simp only [h2, beq_self_eq_true, ↓reduceIte] at hCfg
            by_cases ha : (if axis < 0 then axis + ↑(2 : Nat) else axis) = 1
            · simp only [ha, beq_self_eq_true, ↓reduceIte, Except.ok.injEq, Prod.mk.injEq] at hCfg
              rcases hCfg with ⟨_, rfl, _, rfl⟩
              have h1p := array_getElem!_pos_of_all_pos xTy.shape 1 hxPos (by omega)
              exact ⟨fun d hd => by simp at hd; rcases hd with rfl | rfl <;> omega, h1p⟩
            · simp only [beq_eq_false_iff_ne.mpr ha, Bool.false_eq_true, ↓reduceIte] at hCfg
              cases hCfg
          · simp only [beq_eq_false_iff_ne.mpr h2, Bool.false_eq_true, ↓reduceIte] at hCfg
            by_cases h4 : xTy.shape.size = 4
            · simp only [h4, beq_self_eq_true, ↓reduceIte] at hCfg
              generalize (if axis < 0 then axis + ↑(4 : Nat) else axis) = ax4 at hCfg
              match ax4 with
              | 1 =>
                cases hBad : (xTy.shape[2]! != 1 || xTy.shape[3]! != 1) with
                | true => rw [hBad] at hCfg; cases hCfg
                | false =>
                  rw [hBad] at hCfg
                  simp only [Bool.false_eq_true, ↓reduceIte, Except.ok.injEq, Prod.mk.injEq] at hCfg
                  rcases hCfg with ⟨_, rfl, _, rfl⟩
                  have h1p := array_getElem!_pos_of_all_pos xTy.shape 1 hxPos (by omega)
                  exact ⟨fun d hd => by simp at hd; rcases hd with rfl | rfl | rfl | rfl <;> omega, h1p⟩
              | 3 =>
                simp only [Except.ok.injEq, Prod.mk.injEq] at hCfg
                rcases hCfg with ⟨_, rfl, _, rfl⟩
                have h1p := array_getElem!_pos_of_all_pos xTy.shape 1 hxPos (by omega)
                have h2p := array_getElem!_pos_of_all_pos xTy.shape 2 hxPos (by omega)
                have h3p := array_getElem!_pos_of_all_pos xTy.shape 3 hxPos (by omega)
                exact ⟨fun d hd => by simp at hd; rcases hd with rfl | rfl | rfl | rfl <;> omega, h3p⟩
              | Int.ofNat 0 | Int.ofNat 2 | Int.ofNat (_ + 4) | Int.negSucc _ => cases hCfg
            · simp only [beq_eq_false_iff_ne.mpr h4, Bool.false_eq_true, ↓reduceIte] at hCfg
              cases hCfg
      cases hLnCfg : resolveLayerNormConfig xTy.shape axis _ with
      | error _ => rw [hLnCfg] at hStep; cases hStep
      | ok lnCfg =>
        rw [hLnCfg] at hStep; dsimp only at hStep
        rcases lnCfg with ⟨redAxes, redShape, normAxis, normDim⟩
        have ⟨hRedPos, hNormPos⟩ := hLnCfgPos _ redAxes redShape normAxis normDim hLnCfg
        cases hEpsBad : (eps.isNaN || eps.isInf || decide (eps < 0.0)) with
        | true => rw [hEpsBad] at hStep; cases hStep
        | false =>
        rw [hEpsBad] at hStep; simp only [Bool.false_eq_true, ↓reduceIte] at hStep
        cases hEps : validateFiniteFp16ScalarIR eps _ with
        | error _ => rw [hEps] at hStep; cases hStep
        | ok epsBits =>
          rw [hEps] at hStep; dsimp only at hStep
          -- Track the 6 common preamble nodes (`mean`, `diff`, `sq`, `var`, `vareps`, `invstd`)
          let meanId := if nd.outputs.size ≥ 2 then nd.outputs[1]!.1 else st.nextId
          let nextAfterMean := if nd.outputs.size < 2 then st.nextId + 1 else st.nextId
          let diffId := nextAfterMean
          let sqId := nextAfterMean + 1
          let varId := nextAfterMean + 2
          let varEpsId := nextAfterMean + 3
          let nextAfterVarEps := nextAfterMean + 4
          let invStdId := if nd.outputs.size ≥ 3 then nd.outputs[2]!.1 else nextAfterVarEps
          let nextAfterInvStd := if nd.outputs.size < 3 then nextAfterVarEps + 1 else nextAfterVarEps
          let usedAfterMean : List ValueId := if nd.outputs.size ≥ 2 then [nd.outputs[1]!.1] else []
          let usedAfterInvStd : List ValueId :=
            usedAfterMean ++ (if nd.outputs.size ≥ 3 then [nd.outputs[2]!.1] else [])
          have hMeanUsedLt : ∀ v ∈ usedAfterMean, v < g.nextValueId := by
            intro v hv
            dsimp only [usedAfterMean] at hv
            split at hv
            · rename_i hge2
              have h1Lt : 1 < g.nodes[k].outputs.size := by dsimp only [nd] at hge2; omega
              simp only [List.mem_singleton] at hv
              subst hv
              rw [getElem!_pos nd.outputs 1 h1Lt]
              exact hOutLt g.nodes[k].outputs[1] (Array.getElem_mem_toList h1Lt)
            · cases hv
          have hInvUsedLt : ∀ v ∈ usedAfterInvStd, v < g.nextValueId := by
            intro v hv
            rw [List.mem_append] at hv
            rcases hv with h1 | h2
            · exact hMeanUsedLt v h1
            · split at h2
              · rename_i hge3
                have h2Lt : 2 < g.nodes[k].outputs.size := by dsimp only [nd] at hge3; omega
                simp only [List.mem_singleton] at h2
                subst h2
                rw [getElem!_pos nd.outputs 2 h2Lt]
                exact hOutLt g.nodes[k].outputs[2] (Array.getElem_mem_toList h2Lt)
              · cases h2
          -- Push node 1: `mean`
          let meanNd : IRNode CoreOp := {
            name := s!"{nd.name}_mean"
            op := .reduce .mean redAxes true 0
            inputs := #[(nd.inputs.map (lookupAlias st.aliases))[0]!]
            outputs := #[(meanId, { shape := redShape, dtype := xTy.dtype })]
          }
          have hPMean :
              DecompStepInv g k usedAfterMean (st.outNodes.push meanNd) st.aliases nextAfterMean ∧
              meanId ∈ (decompGraph g (st.outNodes.push meanNd)).allDefinedIds ∧
              (∀ v ∈ (decompGraph g st.outNodes).allDefinedIds,
                v ∈ (decompGraph g (st.outNodes.push meanNd)).allDefinedIds) := by
            by_cases hge2 : nd.outputs.size ≥ 2
            · have hNotLt2 : ¬ nd.outputs.size < 2 := by omega
              have h1Lt : 1 < g.nodes[k].outputs.size := by dsimp only [nd] at hge2; omega
              have hGet1 : nd.outputs[1]! = nd.outputs[1] := getElem!_pos _ 1 h1Lt
              dsimp only [meanNd, meanId, nextAfterMean, usedAfterMean]
              simp only [hge2, hNotLt2, ↓reduceIte, hGet1]
              have hP := stepInv_pushOrig hInv s!"{nd.name}_mean" (.reduce .mean redAxes true 0)
                #[(nd.inputs.map (lookupAlias st.aliases))[0]!]
                #[(nd.outputs[1].1, { shape := redShape, dtype := xTy.dtype })]
                (fun v hv => by simp at hv; subst hv; exact hxDef) (by simp)
                (fun o ho => by simp at ho; subst ho; exact hOutLt g.nodes[k].outputs[1] (Array.getElem_mem_toList h1Lt))
                (fun o ho => by simp at ho; subst ho; exact hOutNotPrior g.nodes[k].outputs[1] (Array.getElem_mem_toList h1Lt))
                (fun _ _ h => by cases h)
                (fun o ho d hd => by simp at ho; subst ho; exact hRedPos d hd)
              exact ⟨by simpa using hP.1,
                hP.2.1 (nd.outputs[1].1, { shape := redShape, dtype := xTy.dtype }) (by simp),
                hP.2.2⟩
            · have hLt2 : nd.outputs.size < 2 := by omega
              dsimp only [meanNd, meanId, nextAfterMean, usedAfterMean]
              simp only [hge2, hLt2, ↓reduceIte]
              exact stepInv_pushFresh hInv (fun _ h => by cases h) s!"{nd.name}_mean"
                (.reduce .mean redAxes true 0) #[(nd.inputs.map (lookupAlias st.aliases))[0]!]
                { shape := redShape, dtype := xTy.dtype }
                (fun v hv => by simp at hv; subst hv; exact hxDef) hRedPos
          -- Push nodes 2..5: `diff`, `sq`, `var`, `vareps` (all fresh)
          have hPDiff := stepInv_pushFresh hPMean.1 hMeanUsedLt s!"{nd.name}_diff"
            (.elemBinary .sub) #[(nd.inputs.map (lookupAlias st.aliases))[0]!, meanId] xTy
            (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hPMean.2.2 _ hxDef; exact hPMean.2.1) hxPos
          have hPSq := stepInv_pushFresh hPDiff.1 hMeanUsedLt s!"{nd.name}_sq"
            (.elemBinary .mul) #[diffId, diffId] xTy
            (fun v hv => by simp at hv; rcases hv with rfl | rfl <;> exact hPDiff.2.1) hxPos
          have hPVar := stepInv_pushFresh hPSq.1 hMeanUsedLt s!"{nd.name}_var"
            (.reduce .mean redAxes true 0) #[sqId] { shape := redShape, dtype := xTy.dtype }
            (fun v hv => by simp at hv; subst hv; exact hPSq.2.1) hRedPos
          have hPVarEps := stepInv_pushFresh hPVar.1 hMeanUsedLt s!"{nd.name}_vareps"
            (.elemUnary (.scalarAdd epsBits)) #[varId] { shape := redShape, dtype := xTy.dtype }
            (fun v hv => by simp at hv; subst hv; exact hPVar.2.1) hRedPos
          -- Push node 6: `invstd`
          let diffNd : IRNode CoreOp := {
            name := s!"{nd.name}_diff", op := .elemBinary .sub,
            inputs := #[(nd.inputs.map (lookupAlias st.aliases))[0]!, meanId],
            outputs := #[(diffId, xTy)]
          }
          let sqNd : IRNode CoreOp := {
            name := s!"{nd.name}_sq", op := .elemBinary .mul,
            inputs := #[diffId, diffId], outputs := #[(sqId, xTy)]
          }
          let varNd : IRNode CoreOp := {
            name := s!"{nd.name}_var", op := .reduce .mean redAxes true 0,
            inputs := #[sqId], outputs := #[(varId, { shape := redShape, dtype := xTy.dtype })]
          }
          let varEpsNd : IRNode CoreOp := {
            name := s!"{nd.name}_vareps", op := .elemUnary (.scalarAdd epsBits),
            inputs := #[varId], outputs := #[(varEpsId, { shape := redShape, dtype := xTy.dtype })]
          }
          let nodes5 := ((((st.outNodes.push meanNd).push diffNd).push sqNd).push varNd).push varEpsNd
          let invStdNd : IRNode CoreOp := {
            name := s!"{nd.name}_invstd", op := .lut (.lut (.rsqrt 0.0)),
            inputs := #[varEpsId], outputs := #[(invStdId, { shape := redShape, dtype := xTy.dtype })]
          }
          let nodes6 := nodes5.push invStdNd
          have hPInvStd :
              DecompStepInv g k usedAfterInvStd nodes6 st.aliases nextAfterInvStd ∧
              invStdId ∈ (decompGraph g nodes6).allDefinedIds ∧
              (∀ v ∈ (decompGraph g nodes5).allDefinedIds, v ∈ (decompGraph g nodes6).allDefinedIds) := by
            by_cases hge3 : nd.outputs.size ≥ 3
            · have hNotLt3 : ¬ nd.outputs.size < 3 := by omega
              have h2Lt : 2 < g.nodes[k].outputs.size := by dsimp only [nd] at hge3; omega
              have h1Lt : 1 < g.nodes[k].outputs.size := by omega
              have hGet2 : nd.outputs[2]! = nd.outputs[2] := getElem!_pos _ 2 h2Lt
              have hGet1 : nd.outputs[1]! = nd.outputs[1] := getElem!_pos _ 1 h1Lt
              have hNe21 : nd.outputs[2].1 ≠ nd.outputs[1].1 := by
                have hLen : g.nodes[k].outputIds.length = g.nodes[k].outputs.size := by simp [IRNode.outputIds]
                have hPair := List.pairwise_iff_getElem.mp hOutNd 1 2 (by omega) (by omega) (by omega)
                intro hEq; exact hPair (by simpa [IRNode.outputIds] using hEq.symm)
              dsimp only [nodes6, invStdNd, invStdId, nextAfterInvStd, usedAfterInvStd]
              simp only [hge3, hNotLt3, ↓reduceIte, hGet2]
              have hNotInMean : nd.outputs[2].1 ∉ usedAfterMean := by
                dsimp only [usedAfterMean]
                split
                · simp only [hGet1, List.mem_singleton]; exact hNe21
                · simp
              have hP := stepInv_pushOrig hPVarEps.1 s!"{nd.name}_invstd" (.lut (.lut (.rsqrt 0.0)))
                #[varEpsId] #[(nd.outputs[2].1, { shape := redShape, dtype := xTy.dtype })]
                (fun v hv => by simp at hv; subst hv; exact hPVarEps.2.1) (by simp)
                (fun o ho => by simp at ho; subst ho; exact hOutLt g.nodes[k].outputs[2] (Array.getElem_mem_toList h2Lt))
                (fun o ho => by simp at ho; subst ho; exact hOutNotPrior g.nodes[k].outputs[2] (Array.getElem_mem_toList h2Lt))
                (fun o ho => by simp at ho; subst ho; exact hNotInMean)
                (fun o ho d hd => by simp at ho; subst ho; exact hRedPos d hd)
              exact ⟨by simpa using hP.1,
                hP.2.1 (nd.outputs[2].1, { shape := redShape, dtype := xTy.dtype }) (by simp),
                hP.2.2⟩
            · have hLt3 : nd.outputs.size < 3 := by omega
              dsimp only [nodes6, invStdNd, invStdId, nextAfterInvStd, usedAfterInvStd]
              simp only [hge3, hLt3, ↓reduceIte]
              have hP := stepInv_pushFresh hPVarEps.1 hMeanUsedLt s!"{nd.name}_invstd"
                (.lut (.lut (.rsqrt 0.0))) #[varEpsId] { shape := redShape, dtype := xTy.dtype }
                (fun v hv => by simp at hv; subst hv; exact hPVarEps.2.1) hRedPos
              exact ⟨by simpa [List.append_nil] using hP.1, hP.2.1, hP.2.2⟩
          -- All preamble values `diffId`, `invStdId`, and original `st.outNodes` values are now defined!
          have hDiffDef : diffId ∈ (decompGraph g nodes6).allDefinedIds :=
            hPInvStd.2.2 _ (hPVarEps.2.2 _ (hPVar.2.2 _ (hPSq.2.2 _ hPDiff.2.1)))
          have hInvDef : invStdId ∈ (decompGraph g nodes6).allDefinedIds := hPInvStd.2.1
          have hOldLift : ∀ v ∈ (decompGraph g st.outNodes).allDefinedIds, v ∈ (decompGraph g nodes6).allDefinedIds :=
            fun v hv => hPInvStd.2.2 _ (hPVarEps.2.2 _ (hPVar.2.2 _ (hPSq.2.2 _ (hPDiff.2.2 _ (hPMean.2.2 _ hv)))))
          have hOut0NotUsed : nd.outputs[0].1 ∉ usedAfterInvStd := by
            intro hmem
            rw [List.mem_append] at hmem
            rcases hmem with h1 | h2
            · dsimp only [usedAfterMean] at h1
              split at h1
              · rename_i hge2
                have h1Lt : 1 < nd.outputs.size := by omega
                rw [getElem!_pos nd.outputs 1 h1Lt, List.mem_singleton] at h1
                have hPair := List.pairwise_iff_getElem.mp hOutNd 0 1 (by simp [IRNode.outputIds]; omega)
                  (by simp [IRNode.outputIds]; omega) (by omega)
                exact hPair (by simpa [IRNode.outputIds] using h1)
              · cases h1
            · split at h2
              · rename_i hge3
                have h2Lt : 2 < nd.outputs.size := by omega
                rw [getElem!_pos nd.outputs 2 h2Lt, List.mem_singleton] at h2
                have hPair := List.pairwise_iff_getElem.mp hOutNd 0 2 (by simp [IRNode.outputIds]; omega)
                  (by simp [IRNode.outputIds]; omega) (by omega)
                exact hPair (by simpa [IRNode.outputIds] using h2)
              · cases h2
          -- Now discharge the entire scale/bias tail in one call to `decomposeLayerNormTail_preserves_inv`!
          cases hTail : decomposeLayerNormTail g.inputs allowViewOps nd.name _ xTy _ normAxis normDim
              hasScale hasBias (nd.inputs.map (lookupAlias st.aliases)) nd.outputs[0]! diffId invStdId
              nodes6 nextAfterInvStd with
          | error _ => rw [hTail] at hStep; cases hStep
          | ok tailRes =>
            rcases tailRes with ⟨outNodes', nextId'⟩
            rw [hTail] at hStep; simp only [Except.ok.injEq] at hStep; subst hStep
            rw [hGet0] at hTail
            have hT := decomposeLayerNormTail_preserves_inv hPInvStd.1 hInvUsedLt allowViewOps nd.name _
              xTy hxPos _ normAxis normDim hNormPos hasScale hasBias (nd.inputs.map (lookupAlias st.aliases))
              (fun idx hidx => hOldLift _ (hInpAt idx hidx)) nd.outputs[0]
              (hOutLt _ (Array.getElem_mem_toList h0)) (hOutNotPrior _ (Array.getElem_mem_toList h0))
              hOut0NotUsed (hwf.nodeOutputsPositive k hk _ (Array.getElem_mem_toList h0))
              diffId invStdId hDiffDef hInvDef outNodes' nextId' hTail
            apply stepInv_finish hk hT.1
            · intro v hv
              rw [List.mem_append] at hv
              rcases hv with hU | hY
              · rw [List.mem_append] at hU
                rcases hU with h1 | h2
                · dsimp only [usedAfterMean] at h1
                  split at h1
                  · rename_i hge2
                    have h1Lt : 1 < g.nodes[k].outputs.size := by dsimp only [nd] at hge2; omega
                    rw [getElem!_pos nd.outputs 1 h1Lt, List.mem_singleton] at h1
                    rw [h1]; exact List.mem_map.mpr ⟨g.nodes[k].outputs[1], Array.getElem_mem_toList h1Lt, rfl⟩
                  · cases h1
                · split at h2
                  · rename_i hge3
                    have h2Lt : 2 < g.nodes[k].outputs.size := by dsimp only [nd] at hge3; omega
                    rw [getElem!_pos nd.outputs 2 h2Lt, List.mem_singleton] at h2
                    rw [h2]; exact List.mem_map.mpr ⟨g.nodes[k].outputs[2], Array.getElem_mem_toList h2Lt, rfl⟩
                  · cases h2
              · simp at hY; rw [hY]
                exact List.mem_map.mpr ⟨nd.outputs[0], Array.getElem_mem_toList h0, rfl⟩
            · intro v hv
              unfold IRNode.outputIds at hv
              rw [List.mem_map] at hv
              obtain ⟨o, ho, rfl⟩ := hv
              rw [hNotKeyLookup o ho]
              rw [List.mem_iff_getElem] at ho
              obtain ⟨idx, hIdx, hEqO⟩ := ho
              have hIdxSz : idx < nd.outputs.size := by simpa using hIdx
              have hOEq : nd.outputs[idx] = o := by simpa using hEqO
              subst hOEq
              have hCases : idx = 0 ∨ idx = 1 ∨ idx = 2 := by omega
              rcases hCases with rfl | rfl | rfl
              · exact hT.2.1
              · have hge2 : nd.outputs.size ≥ 2 := by omega
                have hMeanEq : meanId = nd.outputs[1].1 := by
                  dsimp only [meanId]; simp [hge2, getElem!_pos nd.outputs 1 hIdxSz]
                rw [← hMeanEq]
                exact hT.2.2 _ (hPInvStd.2.2 _ (hPVarEps.2.2 _ (hPVar.2.2 _ (hPSq.2.2 _ (hPDiff.2.2 _ hPMean.2.1)))))
              · have hge3 : nd.outputs.size ≥ 3 := by omega
                have hInvEq : invStdId = nd.outputs[2].1 := by
                  dsimp only [invStdId]; simp [hge3, getElem!_pos nd.outputs 2 hIdxSz]
                rw [← hInvEq]
                exact hT.2.2 _ hInvDef
  | rmsNorm axis eps hasScale =>
    rw [hOp] at hStep; dsimp only at hStep
    split at hStep; · cases hStep
    rename_i hInOut
    have hIn1 : (nd.inputs.map (lookupAlias st.aliases)).size ≥ 1 := by
      simp only [Bool.or_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq, not_or] at hInOut; omega
    have hOut1 : nd.outputs.size = 1 := by
      simp only [Bool.or_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq, not_or, Classical.not_not] at hInOut
      exact hInOut.2
    have h0 : 0 < nd.outputs.size := by omega
    have hGet0 : nd.outputs[0]! = nd.outputs[0] := getElem!_pos _ 0 h0
    have hOutList : nd.outputs.toList = [nd.outputs[0]] := by
      cases hl : nd.outputs.toList with
      | nil => have : nd.outputs.size = 0 := by rw [← Array.length_toList, hl]; rfl
               omega
      | cons a tl =>
        cases tl with
        | nil =>
          have : nd.outputs[0] = a := by simp [← Array.getElem_toList, hl]
          rw [this]
        | cons b tl' =>
          have : nd.outputs.size ≥ 2 := by rw [← Array.length_toList, hl]; simp
          omega
    cases hTy : lookupTypeInCore g.inputs st.outNodes (nd.inputs.map (lookupAlias st.aliases))[0]! with
    | none => rw [hTy] at hStep; cases hStep
    | some xTy =>
      rw [hTy] at hStep; dsimp only at hStep
      cases hAx : resolveRmsNormAxis xTy.shape.size axis _ with
      | error _ => rw [hAx] at hStep; cases hStep
      | ok axNat =>
      rw [hAx] at hStep; dsimp only at hStep
      cases hEpsBad : (eps.isNaN || eps.isInf || decide (eps < 0.0)) with
      | true => rw [hEpsBad] at hStep; cases hStep
      | false =>
      rw [hEpsBad] at hStep; simp only [Bool.false_eq_true, ↓reduceIte] at hStep
      cases hEps : validateFiniteFp16ScalarIR eps _ with
      | error _ => rw [hEps] at hStep; cases hStep
      | ok epsBits =>
        rw [hEps] at hStep; dsimp only at hStep
        have hxDef := hInpAt 0 (by omega)
        have hxPos := lookupTypeInCore_pos g st.outNodes hInv.wf _ xTy hTy
        have hRedPos := array_set!_pos xTy.shape axNat 1 hxPos (by omega)
        have hP1 := stepInv_pushFresh hInv (fun _ h => by cases h) s!"{nd.name}_invrms"
          (.reduce .rsqrtMeanSquare #[axNat] true epsBits)
          #[(nd.inputs.map (lookupAlias st.aliases))[0]!]
          { shape := xTy.shape.set! axNat 1, dtype := xTy.dtype }
          (fun v hv => by simp at hv; subst hv; exact hxDef) hRedPos
        have hFinishRMS := fun {nodesF : Array (IRNode CoreOp)} {nextF : Nat}
            (hInvF : DecompStepInv g k [] nodesF st.aliases nextF)
            (opF : CoreOp) (inpsF : Array ValueId)
            (hInF : ∀ v ∈ inpsF.toList, v ∈ (decompGraph g nodesF).allDefinedIds) => by
          have hPF := hPushOrigAt hInvF nd.name opF inpsF 0 h0 hInF (by simp)
          exact stepInv_finish hk hPF.1
            (fun v hv => by
              simp only [List.nil_append, List.map_cons, List.map_nil, List.mem_singleton] at hv
              subst hv
              exact List.mem_map.mpr ⟨nd.outputs[0], Array.getElem_mem_toList h0, rfl⟩)
            (fun v hv => by
              unfold IRNode.outputIds at hv
              change v ∈ nd.outputs.toList.map Prod.fst at hv
              rw [hOutList] at hv
              simp only [List.map_cons, List.map_nil, List.mem_singleton] at hv
              subst hv; rw [hNotKeyLookup _ (Array.getElem_mem_toList h0)]; exact hPF.2.1 _ (by simp))
        split at hStep
        · simp only [Except.ok.injEq] at hStep; subst hStep
          rw [hGet0]
          exact hFinishRMS hP1.1 (.elemBinary .mul)
            #[(nd.inputs.map (lookupAlias st.aliases))[0]!, st.nextId]
            (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hP1.2.2 _ hxDef; exact hP1.2.1)
        · rename_i hHasSc
          have hIn2Ge : (nd.inputs.map (lookupAlias st.aliases)).size ≥ 2 := by
            simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_true_eq, not_or] at hHasSc; omega
          have hScDef : (nd.inputs.map (lookupAlias st.aliases))[1]! ∈ (decompGraph g _).allDefinedIds :=
            hP1.2.2 _ (hInpAt 1 (by omega))
          cases hScTy : lookupTypeInCore g.inputs _ (nd.inputs.map (lookupAlias st.aliases))[1]! with
          | none => rw [hScTy] at hStep; cases hStep
          | some scTy =>
            rw [hScTy] at hStep; dsimp only at hStep
            cases hScOk : isValidRmsNormScaleShape scTy.shape axNat xTy.shape[axNat]! xTy.shape.size with
            | false => rw [hScOk] at hStep; cases hStep
            | true =>
            rw [hScOk] at hStep; simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hStep
            cases hScC : lookupConstInCore _ (nd.inputs.map (lookupAlias st.aliases))[1]! with
            | some scPair =>
              rw [hScC] at hStep; dsimp only at hStep
              cases hScF64 : scPair.2.toFloat64? with
              | none => rw [hScF64] at hStep; cases hStep
              | some scVals =>
                rw [hScF64] at hStep; dsimp only at hStep
                cases hU : arrayUniformFloat scVals with
                | some sVal =>
                  rw [hU] at hStep; dsimp only at hStep
                  split at hStep
                  · simp only [Except.ok.injEq] at hStep; subst hStep
                    rw [hGet0]
                    exact hFinishRMS hP1.1 (.elemBinary .mul)
                      #[(nd.inputs.map (lookupAlias st.aliases))[0]!, st.nextId]
                      (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hP1.2.2 _ hxDef; exact hP1.2.1)
                  · cases hSB : validateFiniteFp16ScalarIR sVal _ with
                    | error _ => rw [hSB] at hStep; cases hStep
                    | ok sBits =>
                      rw [hSB] at hStep; dsimp only at hStep
                      simp only [Except.ok.injEq] at hStep; subst hStep
                      have hPNorm := stepInv_pushFresh hP1.1 (fun _ h => by cases h) s!"{nd.name}_norm"
                        (.elemBinary .mul) #[(nd.inputs.map (lookupAlias st.aliases))[0]!, st.nextId] xTy
                        (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hP1.2.2 _ hxDef; exact hP1.2.1) hxPos
                      rw [hGet0]
                      exact hFinishRMS hPNorm.1 (.elemUnary (.scalarMul sBits)) #[st.nextId + 1]
                        (fun v hv => by simp at hv; subst hv; exact hPNorm.2.1)
                | none =>
                  rw [hU] at hStep; dsimp only at hStep
                  simp only [Except.ok.injEq] at hStep; subst hStep
                  have hPNorm := stepInv_pushFresh hP1.1 (fun _ h => by cases h) s!"{nd.name}_norm"
                    (.elemBinary .mul) #[(nd.inputs.map (lookupAlias st.aliases))[0]!, st.nextId] xTy
                    (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hP1.2.2 _ hxDef; exact hP1.2.1) hxPos
                  rw [hGet0]
                  exact hFinishRMS hPNorm.1 (.elemBinary .mul)
                    #[st.nextId + 1, (nd.inputs.map (lookupAlias st.aliases))[1]!]
                    (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hPNorm.2.1; exact hPNorm.2.2 _ hScDef)
            | none =>
              rw [hScC] at hStep; dsimp only at hStep
              simp only [Except.ok.injEq] at hStep; subst hStep
              have hPNorm := stepInv_pushFresh hP1.1 (fun _ h => by cases h) s!"{nd.name}_norm"
                (.elemBinary .mul) #[(nd.inputs.map (lookupAlias st.aliases))[0]!, st.nextId] xTy
                (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hP1.2.2 _ hxDef; exact hP1.2.1) hxPos
              rw [hGet0]
              exact hFinishRMS hPNorm.1 (.elemBinary .mul)
                #[st.nextId + 1, (nd.inputs.map (lookupAlias st.aliases))[1]!]
                (fun v hv => by simp at hv; rcases hv with rfl | rfl; exact hPNorm.2.1; exact hPNorm.2.2 _ hScDef)
  | split axis sizes =>
    rw [hOp] at hStep; dsimp only at hStep
    cases hIn1Ne : ((nd.inputs.map (lookupAlias st.aliases)).size != 1) with
    | true => rw [hIn1Ne] at hStep; cases hStep
    | false =>
    rw [hIn1Ne] at hStep; simp only [Bool.false_eq_true, ↓reduceIte] at hStep
    have hIn1 : (nd.inputs.map (lookupAlias st.aliases)).size = 1 := by
      simp only [bne_eq_false_iff_eq] at hIn1Ne; exact hIn1Ne
    cases hSzNe : (sizes.size != nd.outputs.size) with
    | true => rw [hSzNe] at hStep; cases hStep
    | false =>
    rw [hSzNe] at hStep; simp only [Bool.false_eq_true, ↓reduceIte] at hStep
    have hSzEq : sizes.size = nd.outputs.size := by
      simp only [bne_eq_false_iff_eq] at hSzNe; exact hSzNe
    cases hTy : lookupTypeInCore g.inputs st.outNodes (nd.inputs.map (lookupAlias st.aliases))[0]! with
    | none => rw [hTy] at hStep; cases hStep
    | some xTy =>
      rw [hTy] at hStep; dsimp only at hStep
      by_cases hAxGe : axis ≥ xTy.shape.size
      · simp only [hAxGe, ↓reduceIte] at hStep; cases hStep
      · simp only [hAxGe, ↓reduceIte] at hStep
        cases hTotNe : (sizes.foldl (· + ·) 0 != xTy.shape[axis]!) with
      | true => rw [hTotNe] at hStep; cases hStep
      | false =>
      rw [hTotNe] at hStep; simp only [Bool.false_eq_true, ↓reduceIte] at hStep
      have hxDef := hInpAt 0 (by omega)
      cases hSz1 : (sizes.size == 1) with
      | true =>
        rw [hSz1] at hStep; simp only [↓reduceIte, Except.ok.injEq] at hStep; subst hStep
        have hS1 : sizes.size = 1 := by simpa using hSz1
        have h0 : 0 < nd.outputs.size := by omega
        have hGet0 : nd.outputs[0]! = nd.outputs[0] := getElem!_pos _ 0 h0
        rw [hGet0]
        have hO0Mem : nd.outputs[0] ∈ nd.outputs.toList := Array.getElem_mem_toList h0
        let outId := nd.outputs[0].1
        let xId := (nd.inputs.map (lookupAlias st.aliases))[0]!
        have hOutNotDef : outId ∉ (decompGraph g st.outNodes).allDefinedIds := by
          intro hmem
          rcases hInv.defBound outId hmem with h1 | h2 | ⟨h3, _⟩
          · exact hOutNotPrior _ hO0Mem h1
          · cases h2
          · exact Nat.not_lt_of_ge h3 (hOutLt _ hO0Mem)
        have hNe : outId ≠ xId := fun hEq => hOutNotDef (hEq ▸ hxDef)
        have hXNotKey : ∀ p ∈ st.aliases.toList, p.1 ≠ xId :=
          fun p hp hEq => (hInv.aliasKeyPrior p hp).2 (hEq ▸ hxDef)
        have hAliasRes := lookupAlias_push_fresh st.aliases outId xId
          (hOrigNotAliasKey _ hO0Mem) hXNotKey hNe
        have hBefSucc : g.allDefinedIdsBefore (k + 1) = g.allDefinedIdsBefore k ++ nd.outputIds := by
          unfold SSAGraph.allDefinedIdsBefore
          have hTake : g.nodes.toList.take (k + 1) = g.nodes.toList.take k ++ [nd] := by
            rw [List.take_succ_eq_append_getElem (by simpa using hk)]; rfl
          rw [hTake, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.append_assoc]
        have hOutList : nd.outputs.toList = [nd.outputs[0]] := by
          cases hl : nd.outputs.toList with
          | nil => have : nd.outputs.size = 0 := by rw [← Array.length_toList, hl]; rfl
                   omega
          | cons a tl =>
            cases tl with
            | nil =>
              have : nd.outputs[0] = a := by simp [← Array.getElem_toList, hl]
              rw [this]
            | cons b tl' =>
              have : nd.outputs.size ≥ 2 := by rw [← Array.length_toList, hl]; simp
              omega
        refine ⟨hInv.nextGe, hInv.wf, ?_, ?_, ?_, ?_⟩
        · intro v hv
          rcases hInv.defBound v hv with h1 | h2 | h3
          · exact Or.inl (by rw [hBefSucc]; exact List.mem_append_left _ h1)
          · cases h2
          · exact Or.inr (Or.inr h3)
        · intro p hp
          rw [Array.toList_push, List.mem_append, List.mem_singleton] at hp
          rcases hp with hOld | rfl
          · exact hInv.aliasTargetDef p hOld
          · exact hxDef
        · intro p hp
          rw [Array.toList_push, List.mem_append, List.mem_singleton] at hp
          rcases hp with hOld | rfl
          · have ⟨h1, h2⟩ := hInv.aliasKeyPrior p hOld
            exact ⟨by rw [hBefSucc]; exact List.mem_append_left _ h1, h2⟩
          · exact ⟨by rw [hBefSucc]; exact List.mem_append_right _ (by simp [IRNode.outputIds, hOutList]), hOutNotDef⟩
        · intro v hv
          rw [hBefSucc, List.mem_append] at hv
          rcases hv with hOld | hCur
          · have hVne : v ≠ outId := fun hEq => hOutNotPrior _ hO0Mem (by simpa [outId, hEq] using hOld)
            have hTgtNe : ∀ p ∈ st.aliases.toList, p.2 ≠ outId :=
              fun p hp hEq => hOutNotDef (hEq ▸ hInv.aliasTargetDef p hp)
            rw [hAliasRes.2 v hVne hTgtNe]
            exact hInv.priorCovered v hOld
          · simp only [IRNode.outputIds, hOutList, List.map_cons, List.map_nil, List.mem_singleton] at hCur
            rw [hCur, hAliasRes.1]
            exact hxDef
      | false =>
        rw [hSz1] at hStep; simp only [Bool.false_eq_true, ↓reduceIte] at hStep
        cases allowViewOps with
        | false => simp at hStep
        | true =>
        simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, Except.ok.injEq] at hStep; subst hStep
        have hSpl := decomposeSplitSlicesAux_preserves g k hk hwf nd.name
          (nd.inputs.map (lookupAlias st.aliases))[0]! xTy.shape axis sizes hSzEq
          sizes.size 0 0 st.outNodes st.aliases st.nextId (by omega) (Nat.zero_le _)
          (by simpa using hInv) hxDef
        exact stepInv_finish hk hSpl.1
          (fun v hv => by
            rw [List.take_of_length_le (by simp [nd, hSzEq])] at hv
            exact hv)
          (fun v hv => by
            unfold IRNode.outputIds at hv; rw [List.mem_map] at hv
            obtain ⟨o, ho, rfl⟩ := hv
            rw [hNotKeyLookup o ho]
            exact hSpl.2.1 o (by simpa using ho))

/-- End-to-end `HighOp → CoreOp` `decomposeGraph` SSA `WellFormed`ness preservation theorem!
    Directly proves that for ANY `WellFormed` `HighOp` graph `g` with standard BN/LN output arity,
    if `decomposeGraph g allowViewOps = .ok (g', elided)`, then `g'` is `WellFormed` —
    with ZERO `validateGraph` assumption! -/
theorem decomposeGraph_preserves_wellFormed
    (g : SSAGraph HighOp) (allowViewOps : Bool) (g' : SSAGraph CoreOp) (elided : Nat)
    (hwf : WellFormed g)
    (hArity : ValidHighOpOutputArity g)
    (hok : decomposeGraph g allowViewOps = .ok (g', elided)) :
    WellFormed g' := by
  unfold decomposeGraph at hok
  dsimp only [bind, Except.bind, pure, Except.pure] at hok
  let initSt : DecomposeState := {
    outNodes := Array.mkEmpty g.nodes.size, aliases := #[],
    nextId := g.nextValueId, elidedCount := 0
  }
  have hInitWf : WellFormed (decompGraph g initSt.outNodes) := by
    refine ⟨hwf.inputsDense, hwf.inputsPositive, ?_, ?_, ?_, ?_⟩
    · have hNd := hwf.uniqueIds
      unfold SSAGraph.allDefinedIds at hNd ⊢
      have hNil : (decompGraph g initSt.outNodes).nodes.toList = [] := rfl
      rw [hNil, List.flatMap_nil, List.append_nil]
      exact (List.nodup_append.mp hNd).1
    · intro i hi; simp [initSt] at hi
    · intro i hi; simp [initSt] at hi
    · intro v hv; simp at hv
  have hInitInv : DecompStepInv g 0 [] initSt.outNodes initSt.aliases initSt.nextId := by
    refine ⟨Nat.le_refl _, hInitWf, ?_, fun _ h => by simp [initSt] at h,
            fun _ h => by simp [initSt] at h, ?_⟩
    · intro v hv
      left
      unfold SSAGraph.allDefinedIdsBefore
      simpa [SSAGraph.allDefinedIds, SSAGraph.inputIds, decompGraph, initSt] using hv
    · intro v hv
      unfold lookupAlias lookupAliasAux
      have hNone : initSt.aliases.find? (fun p => p.1 == v) = none := rfl
      rw [hNone]
      unfold SSAGraph.allDefinedIdsBefore at hv
      simpa [SSAGraph.allDefinedIds, SSAGraph.inputIds, decompGraph, initSt] using hv
  have hFoldInv : ∀ (k : Nat) (hk : k ≤ g.nodes.size) (st : DecomposeState) (finalSt : DecomposeState),
      DecompStepInv g k [] st.outNodes st.aliases st.nextId →
      (g.nodes.toList.drop k).foldlM (decomposeNode g.inputs allowViewOps) st = .ok finalSt →
      DecompStepInv g g.nodes.size [] finalSt.outNodes finalSt.aliases finalSt.nextId := by
    intro k
    have hGen : ∀ rem k (hk : k ≤ g.nodes.size) (hrem : rem = g.nodes.size - k)
        (st finalSt : DecomposeState),
        DecompStepInv g k [] st.outNodes st.aliases st.nextId →
        (g.nodes.toList.drop k).foldlM (decomposeNode g.inputs allowViewOps) st = .ok finalSt →
        DecompStepInv g g.nodes.size [] finalSt.outNodes finalSt.aliases finalSt.nextId := by
      intro rem
      induction rem with
      | zero =>
        intro k hk hrem st finalSt hI hF
        have hkEq : k = g.nodes.size := by omega
        subst hkEq
        rw [List.drop_eq_nil_of_le (by simp)] at hF
        simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at hF
        subst hF; exact hI
      | succ r ih =>
        intro k hk hrem st finalSt hI hF
        have hkLt : k < g.nodes.size := by omega
        have hDrop : g.nodes.toList.drop k = g.nodes[k] :: g.nodes.toList.drop (k + 1) := by
          rw [List.drop_eq_getElem_cons (by simpa using hkLt)]; rfl
        rw [hDrop, List.foldlM_cons] at hF
        dsimp only [bind, Except.bind] at hF
        cases hStep : decomposeNode g.inputs allowViewOps st g.nodes[k] with
        | error _ => rw [hStep] at hF; cases hF
        | ok stMid =>
          rw [hStep] at hF
          have hIMid := decomposeNode_preserves_inv g allowViewOps hwf hArity k hkLt st stMid hI hStep
          exact ih (k + 1) (by omega) (by omega) stMid finalSt hIMid hF
    exact fun hk st finalSt => hGen (g.nodes.size - k) k hk rfl st finalSt
  cases hF : g.nodes.foldlM (decomposeNode g.inputs allowViewOps) initSt with
  | error _ => rw [hF] at hok; cases hok
  | ok finalSt =>
    rw [hF] at hok
    simp only [Except.ok.injEq, Prod.mk.injEq] at hok
    rcases hok with ⟨rfl, _⟩
    have hFList : (g.nodes.toList.drop 0).foldlM (decomposeNode g.inputs allowViewOps) initSt = .ok finalSt := by
      simpa [← Array.foldlM_toList] using hF
    have hEnd := hFoldInv 0 (Nat.zero_le _) initSt finalSt hInitInv hFList
    apply redirectOutputs_preservesWF (decompGraph g finalSt.outNodes)
      (g.outputs.map (lookupAlias finalSt.aliases)) hEnd.wf
    intro v hv
    rw [Array.toList_map, List.mem_map] at hv
    obtain ⟨o, ho, rfl⟩ := hv
    have hoAll : o ∈ g.allDefinedIdsBefore g.nodes.size := by
      unfold SSAGraph.allDefinedIdsBefore
      rw [List.take_of_length_le (by simp)]
      exact hwf.outputsValid o ho
    exact hEnd.priorCovered o hoAll

-- ============================================================================
-- 1B. Backend/ChannelSplit (SOK) Soundness & Semantic Equivalence
-- ============================================================================

private theorem sumSizes_balanced_aux (base rem : Nat) :
    ∀ n, sumSizes ((List.range n).map fun i => base + if i < rem then 1 else 0) =
      n * base + min n rem := by
  intro n
  induction n with
  | zero => simp [sumSizes]
  | succ k ih =>
    rw [List.range_succ, List.map_append, List.map_cons, List.map_nil]
    have hApp : ∀ xs ys, sumSizes (xs ++ ys) = sumSizes xs + sumSizes ys := by
      intro xs ys
      induction xs with
      | nil => simp [sumSizes]
      | cons a tl iht => simp [sumSizes, iht]; omega
    rw [hApp, ih]
    simp only [sumSizes, Nat.add_zero]
    rw [Nat.add_mul, Nat.one_mul]
    split <;> omega

/-- `balancedSlices cout numSlices` has exact length `numSlices`. -/
theorem balancedSlices_length (cout numSlices : Nat) :
    (balancedSlices cout numSlices).length = numSlices := by
  unfold balancedSlices
  simp

/-- Channel conservation: for any `numSlices > 0`, the widths produced by `balancedSlices`
    sum to `cout` identically. -/
theorem balancedSlices_sum (cout numSlices : Nat) (hPos : numSlices > 0) :
    sumSizes (balancedSlices cout numSlices) = cout := by
  unfold balancedSlices
  rw [sumSizes_balanced_aux]
  have hMod : cout % numSlices < numSlices := Nat.mod_lt cout hPos
  have hMin : min numSlices (cout % numSlices) = cout % numSlices := Nat.min_eq_right (Nat.le_of_lt hMod)
  rw [hMin]
  have hDiv := Nat.div_add_mod' cout numSlices
  rw [Nat.mul_comm] at hDiv
  exact hDiv

/-- Every slice width in `balancedSlices cout numSlices` is strictly positive whenever
    `0 < numSlices ≤ cout`. -/
theorem balancedSlices_pos (cout numSlices : Nat) (hPos : numSlices > 0) (hle : numSlices ≤ cout) :
    ∀ dc ∈ balancedSlices cout numSlices, dc > 0 := by
  intro dc hdc
  unfold balancedSlices at hdc
  rw [List.mem_map] at hdc
  obtain ⟨i, _, rfl⟩ := hdc
  have hDiv : 1 ≤ cout / numSlices := Nat.div_pos hle hPos
  omega

/-- Every slice width in `balancedSlices cout (ceilDiv cout bestDc)` is at most `bestDc`
    (guaranteeing every SOK channel slice satisfies the 64 KiB kernel chunk cap!). -/
theorem balancedSlices_le_cap (cout bestDc : Nat) (hCap : bestDc > 0) (hCout : cout > 0) :
    ∀ dc ∈ balancedSlices cout (ceilDiv cout bestDc), dc ≤ bestDc := by
  intro dc hdc
  unfold balancedSlices at hdc
  rw [List.mem_map] at hdc
  obtain ⟨i, _, rfl⟩ := hdc
  have hNum : ceilDiv cout bestDc = (cout + bestDc - 1) / bestDc := by
    unfold ceilDiv; simp [Nat.ne_of_gt hCap]
  have hkPos : ceilDiv cout bestDc > 0 := by
    rw [hNum]
    exact Nat.div_pos (by omega) hCap
  have hProd : cout ≤ ceilDiv cout bestDc * bestDc := by
    have h1 : (cout + bestDc - 1) / bestDc * bestDc + (cout + bestDc - 1) % bestDc = cout + bestDc - 1 :=
      Nat.div_add_mod' (cout + bestDc - 1) bestDc
    have h2 : (cout + bestDc - 1) % bestDc < bestDc := Nat.mod_lt _ hCap
    rw [hNum]
    omega
  by_cases hlt : cout < ceilDiv cout bestDc * bestDc
  · have hDivLt : cout / ceilDiv cout bestDc < bestDc :=
      Nat.div_lt_of_lt_mul hlt
    split <;> omega
  · have hEq : cout = ceilDiv cout bestDc * bestDc := by omega
    have hDivEq : cout / ceilDiv cout bestDc = bestDc := by
      have h1 := Nat.mul_div_cancel_left bestDc hkPos
      rw [← hEq] at h1
      exact h1
    have hModEq : cout % ceilDiv cout bestDc = 0 := by
      have h1 := Nat.mul_mod_right (ceilDiv cout bestDc) bestDc
      rw [← hEq] at h1
      exact h1
    rw [hDivEq, hModEq]
    simp

/-- `prefixStartsAux` gives the exact start offset of every slice `i < sizes.length`
    as `acc + sumSizes (sizes.take i)`. -/
theorem prefixStartsAux_get (acc : Nat) (sizes : List Nat) :
    ∀ i (hi : i < sizes.length),
      (prefixStartsAux acc sizes)[i]'(by
        have hl : ∀ a l, (prefixStartsAux a l).length = l.length := by
          intro a l; induction l generalizing a with | nil => rfl | cons _ _ ih => simp [prefixStartsAux, ih]
        rw [hl]; exact hi) = acc + sumSizes (sizes.take i) := by
  induction sizes generalizing acc with
  | nil => intro i hi; cases hi
  | cons sz rest ih =>
    intro i hi
    cases i with
    | zero => simp [prefixStartsAux, sumSizes]
    | succ j =>
      simp only [prefixStartsAux, List.getElem_cons_succ, List.take_succ_cons, sumSizes]
      rw [ih (acc + sz) j (by simpa using hi)]
      omega

/-- Exact unique channel owner theorem for SOK channel partitioning:
    for ANY channel `c < sumSizes sizes`, there exists a UNIQUE slice index `si < sizes.length`
    such that `c0 ≤ c < c0 + sizes[si]` where `c0 = sumSizes (sizes.take si)`. -/
theorem sok_channel_unique_owner (sizes : List Nat) (c : Nat) (hc : c < sumSizes sizes) :
    ∃ si : Nat, ∃ hsi : si < sizes.length,
      let c0 := sumSizes (sizes.take si)
      c0 ≤ c ∧ c < c0 + sizes[si] ∧
      ∀ sj (hsj : sj < sizes.length),
        sumSizes (sizes.take sj) ≤ c → c < sumSizes (sizes.take sj) + sizes[sj] → sj = si := by
  have hTakeMono : ∀ (l : List Nat) (a b : Nat) (hab : a < b) (hb : b ≤ l.length),
      sumSizes (l.take a) + l[a]'(Nat.lt_of_lt_of_le hab hb) ≤ sumSizes (l.take b) := by
    intro l
    induction l with
    | nil => intro a b hab hb; simp at hb; omega
    | cons x xs ihl =>
      intro a b hab hb
      cases b with
      | zero => omega
      | succ b' =>
        cases a with
        | zero =>
          simp only [List.take_zero, sumSizes, Nat.zero_add, List.getElem_cons_zero, List.take_succ_cons]
          omega
        | succ a' =>
          simp only [List.take_succ_cons, sumSizes, List.getElem_cons_succ]
          have hsub := ihl a' b' (by omega) (by simpa using hb)
          omega
  suffices hEx : ∃ si : Nat, ∃ hsi : si < sizes.length,
      sumSizes (sizes.take si) ≤ c ∧ c < sumSizes (sizes.take si) + sizes[si] by
    obtain ⟨si, hsi, hlo, hhi⟩ := hEx
    refine ⟨si, hsi, hlo, hhi, ?_⟩
    intro sj hsj hloJ hhiJ
    by_cases hlt : sj < si
    · have hm := hTakeMono sizes sj si hlt (Nat.le_of_lt hsi)
      omega
    · by_cases hgt : si < sj
      · have hm := hTakeMono sizes si sj hgt (Nat.le_of_lt hsj)
        omega
      · omega
  induction sizes generalizing c with
  | nil => simp [sumSizes] at hc
  | cons sz rest ih =>
    by_cases h0 : c < sz
    · exact ⟨0, Nat.zero_lt_succ _, by simp [sumSizes], by simpa [sumSizes] using h0⟩
    · have hc' : c - sz < sumSizes rest := by simp [sumSizes] at hc; omega
      obtain ⟨si', hsi', hlo', hhi'⟩ := ih (c - sz) hc'
      refine ⟨si' + 1, Nat.succ_lt_succ hsi', ?_, ?_⟩
      · simp only [List.take_succ_cons, sumSizes]; omega
      · simp only [List.take_succ_cons, sumSizes, List.getElem_cons_succ]; omega

/-- SOK fp16 weight slice bounds: extracting `[2 * c0 * cin * k2, 2 * (c0 + dc) * cin * k2)`
    from a weight tensor of size `2 * cout * cin * k2` stays strictly in-bounds and has
    exact byte size `2 * dc * cin * k2` whenever `c0 + dc ≤ cout`. -/
theorem sok_weight_slice_bounds (c0 dc cout cin k2 : Nat) (hBound : c0 + dc ≤ cout) :
    2 * c0 * cin * k2 ≤ 2 * (c0 + dc) * cin * k2 ∧
    2 * (c0 + dc) * cin * k2 ≤ 2 * cout * cin * k2 ∧
    2 * (c0 + dc) * cin * k2 - 2 * c0 * cin * k2 = 2 * dc * cin * k2 := by
  have h1 : 2 * (c0 + dc) * cin * k2 = 2 * c0 * cin * k2 + 2 * dc * cin * k2 := by
    rw [Nat.mul_add, Nat.add_mul, Nat.add_mul]
  have h2 : 2 * (c0 + dc) * cin * k2 ≤ 2 * cout * cin * k2 :=
    Nat.mul_le_mul_right k2 (Nat.mul_le_mul_right cin (Nat.mul_le_mul_left 2 hBound))
  omega

/-- SOK bias slice bounds: extracting `[2 * c0, 2 * (c0 + dc))` from a bias vector of
    size `2 * cout` stays strictly in-bounds and has exact byte size `2 * dc`. -/
theorem sok_bias_slice_bounds (c0 dc cout : Nat) (hBound : c0 + dc ≤ cout) :
    2 * c0 ≤ 2 * (c0 + dc) ∧ 2 * (c0 + dc) ≤ 2 * cout ∧ 2 * (c0 + dc) - 2 * c0 = 2 * dc := by
  omega

/-- Every existing node in `g` has `opWeightGroupId ≤ maxExistingWeightGroupId g`. -/
theorem opWeightGroupId_le_maxExisting (g : SSAGraph AneOp) :
    ∀ nd ∈ g.nodes.toList, opWeightGroupId nd.op ≤ maxExistingWeightGroupId g := by
  intro nd hnd
  unfold maxExistingWeightGroupId
  rw [← Array.foldl_toList]
  exact foldl_max_ge_mem (fun n => opWeightGroupId n.op) g.nodes.toList 0 nd hnd

/-- Fresh SOK `weightGroupId` allocation (`maxExistingWeightGroupId g + 1 + si`) is strictly
    greater than every pre-existing `weightGroupId` in `g`, strictly non-zero, and injective
    across all slices `si ≠ sj` (preventing `__KERN_0` cache collisions!). -/
theorem sok_fresh_weightGroupId_distinct (g : SSAGraph AneOp) (si sj : Nat) :
    let base := maxExistingWeightGroupId g + 1
    (base + si > 0) ∧
    (∀ nd ∈ g.nodes.toList, opWeightGroupId nd.op < base + si) ∧
    (base + si = base + sj → si = sj) := by
  intro base
  refine ⟨by omega, fun nd hnd => ?_, fun h => by omega⟩
  have hle := opWeightGroupId_le_maxExisting g nd hnd
  omega

/-- `concatLookup` along axis 1 on channel-sliced 4D tensors `[1, dc_i, H, W]` recovers the
    exact slice tensor and local channel index `c - sum(dc_0 .. dc_{si-1})`! -/
theorem concatLookup_channel_slices {α : Type} (h w : Nat) :
    ∀ (ts : List (TensorVal α)) (c : Nat),
      (∀ t ∈ ts, ∃ dc, t.shape = [1, dc, h, w]) →
      c < ts.foldl (fun acc t => acc + t.shape.getD 1 0) 0 →
      ∃ si : Nat, ∃ hsi : si < ts.length,
        let c0 := (ts.take si).foldl (fun acc t => acc + t.shape.getD 1 0) 0
        c0 ≤ c ∧ c < c0 + ts[si].shape.getD 1 0 ∧
        concatLookup 1 ts c = some (ts[si], c - c0) := by
  have hFoldAdd : ∀ (l : List (TensorVal α)) (init : Nat),
      l.foldl (fun acc t => acc + t.shape.getD 1 0) init =
      init + l.foldl (fun acc t => acc + t.shape.getD 1 0) 0 := by
    intro l
    induction l with
    | nil => intro init; simp
    | cons x xs ih =>
      intro init
      simp only [List.foldl_cons, Nat.zero_add]
      rw [ih (init + x.shape.getD 1 0), ih (x.shape.getD 1 0)]
      omega
  intro ts
  induction ts with
  | nil => intro c _ hc; simp at hc
  | cons t rest ih =>
    intro c hSh hc
    simp only [List.foldl_cons, Nat.zero_add] at hc
    rw [hFoldAdd rest (t.shape.getD 1 0)] at hc
    by_cases hlt : c < t.shape.getD 1 0
    · refine ⟨0, Nat.zero_lt_succ _, Nat.zero_le _, by simpa using hlt, ?_⟩
      unfold concatLookup
      simp only [hlt, ↓reduceIte]
      rfl
    · have hc' : c - t.shape.getD 1 0 < rest.foldl (fun acc x => acc + x.shape.getD 1 0) 0 := by omega
      have hSh' : ∀ x ∈ rest, ∃ dc, x.shape = [1, dc, h, w] := fun x hx => hSh x (List.mem_cons_of_mem _ hx)
      obtain ⟨si', hsi', hlo', hhi', hlook'⟩ := ih (c - t.shape.getD 1 0) hSh' hc'
      have hTakeC0 : ((t :: rest).take (si' + 1)).foldl (fun acc x => acc + x.shape.getD 1 0) 0 =
          t.shape.getD 1 0 + (rest.take si').foldl (fun acc x => acc + x.shape.getD 1 0) 0 := by
        simp only [List.take_succ_cons, List.foldl_cons, Nat.zero_add]
        exact hFoldAdd (rest.take si') (t.shape.getD 1 0)
      refine ⟨si' + 1, Nat.succ_lt_succ hsi', ?_, ?_, ?_⟩
      · rw [hTakeC0]; omega
      · rw [hTakeC0, List.getElem_cons_succ]; omega
      · rw [hTakeC0, List.getElem_cons_succ]
        unfold concatLookup
        simp only [hlt, ↓reduceIte]
        rw [hlook']
        congr 1
        exact Prod.ext rfl (by omega)

/-- Exact 4D row-major `unflatten` inverse on `[1, C, H, W]`:
    for any `c < C`, `y < H`, `x < W`, `unflatten [1, C, H, W] (c * (H * W) + y * W + x) = [0, c, y, x]`. -/
private theorem unflatten_4d_eq (C H W c y x : Nat) (hc : c < C) (hy : y < H) (hx : x < W) :
    unflatten [1, C, H, W] (c * (H * W) + y * W + x) = [0, c, y, x] := by
  have hW : W > 0 := by omega
  have hHW : H * W > 0 := Nat.mul_pos (by omega) hW
  have hYW : y * W + x < H * W := by
    have h1 : (y + 1) * W ≤ H * W := Nat.mul_le_mul_right W (Nat.succ_le_of_lt hy)
    rw [Nat.add_mul, Nat.one_mul] at h1; omega
  have hAll : c * (H * W) + (y * W + x) < C * (H * W) := by
    have h1 : (c + 1) * (H * W) ≤ C * (H * W) := Nat.mul_le_mul_right (H * W) (Nat.succ_le_of_lt hc)
    rw [Nat.add_mul, Nat.one_mul] at h1; omega
  have hAssoc : c * (H * W) + y * W + x = c * (H * W) + (y * W + x) := by omega
  have hDiv0 : (c * (H * W) + (y * W + x)) / (C * (H * W)) = 0 := Nat.div_eq_of_lt hAll
  have hMod0 : (c * (H * W) + (y * W + x)) % (C * (H * W)) = c * (H * W) + (y * W + x) := Nat.mod_eq_of_lt hAll
  have hDiv1 : (c * (H * W) + (y * W + x)) / (H * W) = c := by
    rw [Nat.mul_comm c (H * W), Nat.mul_add_div hHW, Nat.div_eq_of_lt hYW, Nat.add_zero]
  have hMod1 : (c * (H * W) + (y * W + x)) % (H * W) = y * W + x := by
    rw [Nat.mul_comm c (H * W), Nat.mul_add_mod, Nat.mod_eq_of_lt hYW]
  have hDiv2 : (y * W + x) / W = y := by
    rw [Nat.mul_comm y W, Nat.mul_add_div hW, Nat.div_eq_of_lt hx, Nat.add_zero]
  have hMod2 : (y * W + x) % W = x := by
    rw [Nat.mul_comm y W, Nat.mul_add_mod, Nat.mod_eq_of_lt hx]
  simp only [unflatten, shapeProd, Nat.mul_one, hAssoc, hDiv0, hMod0, hDiv1, hMod1, hDiv2, hMod2, Nat.div_one]

/-- End-to-end SOK semantic equivalence on polymorphic `TensorVal α`:
    concatenating channel slices `ts` of shapes `[1, dc_i, H, W]` with `dc_i > 0` and
    `sum(dc_i) = cout` via `evalConcat 1 ts` succeeds, yields exact output shape `[1, cout, H, W]`,
    AND for EVERY valid coordinate `(c, y, x)` (`c < cout, y < h, x < w`),
    `out.data` at `[0, c, y, x]` equals `ts[si].data` at `[0, c - c0, y, x]` for the unique
    channel-slice owner `si`! -/
theorem sok_evalConcat_shape_and_elements {α : Type} (h w cout : Nat)
    (hh : h > 0) (hw : w > 0)
    (ts : List (TensorVal α)) (hne : ts ≠ [])
    (hShapes : ∀ t ∈ ts, ∃ dc > 0, t.shape = [1, dc, h, w])
    (hSum : ts.foldl (fun acc t => acc + t.shape.getD 1 0) 0 = cout) :
    ∃ out, evalConcat 1 ts = some out ∧ out.shape = [1, cout, h, w] ∧
      ∀ (c y x : Nat) (hc : c < cout) (hy : y < h) (hx : x < w)
        (hOutLt : c * (h * w) + y * w + x < shapeProd out.shape),
        ∃ si : Nat, ∃ hsi : si < ts.length,
          let c0 := (ts.take si).foldl (fun acc t => acc + t.shape.getD 1 0) 0
          c0 ≤ c ∧ c < c0 + ts[si].shape.getD 1 0 ∧
          ∃ hInLt : (c - c0) * (h * w) + y * w + x < shapeProd ts[si].shape,
            out.data ⟨c * (h * w) + y * w + x, hOutLt⟩ =
            ts[si].data ⟨(c - c0) * (h * w) + y * w + x, hInLt⟩ := by
  cases ts with
  | nil => exact absurd rfl hne
  | cons t0 rest =>
    obtain ⟨dc0, hdc0, ht0⟩ := hShapes t0 List.mem_cons_self
    have hr : t0.shape.length = 4 := by rw [ht0]; rfl
    have ht0Pos : t0.shape.all (· > 0) = true := by
      rw [ht0]; simp [hh, hw, hdc0]
    have hRange4 : List.range 4 = [0, 1, 2, 3] := rfl
    have hCompat : (t0 :: rest).all (fun t =>
        t.shape.length == t0.shape.length && t.shape.all (· > 0) &&
        (List.range t0.shape.length).all (fun i => i == 1 || t.shape.getD i 0 == t0.shape.getD i 0)) = true := by
      rw [List.all_eq_true]
      intro t ht
      obtain ⟨dc, hdc, hts⟩ := hShapes t ht
      rw [hr, hRange4, hts, ht0]
      simp [hh, hw, hdc]
    have hN0 : 0 < shapeProd t0.shape := by
      rw [ht0]
      simp only [shapeProd, Nat.mul_one, Nat.one_mul]
      exact Nat.mul_pos hdc0 (Nat.mul_pos hh hw)
    have hGeF : (decide (1 ≥ t0.shape.length)) = false := by rw [hr]; rfl
    have hOutShape : (List.range t0.shape.length).map (fun i =>
        if i == 1 then (t0 :: rest).foldl (fun acc t => acc + t.shape.getD 1 0) 0
        else t0.shape.getD i 0) = [1, cout, h, w] := by
      rw [hSum, hr, hRange4, ht0]; simp
    unfold evalConcat
    dsimp only
    rw [hGeF, ht0Pos]
    simp only [Bool.not_true, Bool.or_self, Bool.false_eq_true, ↓reduceIte]
    rw [hCompat]
    simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte, hN0, ↓reduceDIte]
    refine ⟨_, rfl, hOutShape, ?_⟩
    intro c y x hc hy hx hOutLt
    have hShSimple : ∀ t ∈ (t0 :: rest), ∃ dc, t.shape = [1, dc, h, w] := by
      intro t ht; obtain ⟨dc, _, hts⟩ := hShapes t ht; exact ⟨dc, hts⟩
    obtain ⟨si, hsi, hlo, hhi, hLook⟩ :=
      concatLookup_channel_slices h w (t0 :: rest) c hShSimple (hSum.symm ▸ hc)
    obtain ⟨dcsi, hdcsiPos, hTsi⟩ := hShapes (t0 :: rest)[si] (List.getElem_mem hsi)
    have hDcsiEq : (t0 :: rest)[si].shape.getD 1 0 = dcsi := by rw [hTsi]; rfl
    have hInLt : (c - ((t0 :: rest).take si).foldl (fun acc t => acc + t.shape.getD 1 0) 0) * (h * w) + y * w + x <
        shapeProd (t0 :: rest)[si].shape := by
      rw [hTsi]
      simp only [shapeProd, Nat.mul_one, Nat.one_mul]
      have hYW : y * w + x < h * w := by
        have h1 : (y + 1) * w ≤ h * w := Nat.mul_le_mul_right w (Nat.succ_le_of_lt hy)
        rw [Nat.add_mul, Nat.one_mul] at h1; omega
      have hLoc : c - ((t0 :: rest).take si).foldl (fun acc t => acc + t.shape.getD 1 0) 0 < dcsi := by
        rw [← hDcsiEq]; omega
      have h1 : ((c - ((t0 :: rest).take si).foldl (fun acc t => acc + t.shape.getD 1 0) 0) + 1) * (h * w) ≤ dcsi * (h * w) :=
        Nat.mul_le_mul_right (h * w) (Nat.succ_le_of_lt hLoc)
      rw [Nat.add_mul, Nat.one_mul] at h1; omega
    refine ⟨si, hsi, hlo, hhi, hInLt, ?_⟩
    dsimp only
    have hUnf : unflatten ((List.range t0.shape.length).map fun i =>
        if i == 1 then (t0 :: rest).foldl (fun acc t => acc + t.shape.getD 1 0) 0
        else t0.shape.getD i 0) (c * (h * w) + y * w + x) = [0, c, y, x] := by
      rw [hOutShape]; exact unflatten_4d_eq cout h w c y x hc hy hx
    rw [hUnf]
    have hAxC : [0, c, y, x].getD 1 0 = c := rfl
    rw [hAxC, hLook]
    dsimp only
    have hInCoords : (List.range t0.shape.length).map
        (fun i => if i == 1 then c - ((t0 :: rest).take si).foldl (fun acc t => acc + t.shape.getD 1 0) 0
                  else [0, c, y, x].getD i 0) =
        [0, c - ((t0 :: rest).take si).foldl (fun acc t => acc + t.shape.getD 1 0) 0, y, x] := by
      rw [hr, hRange4]; rfl
    rw [hInCoords]
    have hFlat : flatIndex (t0 :: rest)[si].shape
        [0, c - ((t0 :: rest).take si).foldl (fun acc t => acc + t.shape.getD 1 0) 0, y, x] =
        (c - ((t0 :: rest).take si).foldl (fun acc t => acc + t.shape.getD 1 0) 0) * (h * w) + y * w + x := by
      rw [hTsi]
      simp [flatIndex, shapeProd, Nat.add_assoc]
    simp only [hFlat, hInLt, ↓reduceDIte]

-- ============================================================================
-- 1C. Backend/Cascade Rolling Strip-Unrolling Index Soundness
-- ============================================================================

/-- Every H-strip (or W-strip) window `stripOutBand full step s` for `s < ceilDiv full step`
    is non-empty and strictly within `[0, full]`. -/
theorem stripOutBand_valid (full step s : Nat) (hFull : full > 0) (hStep : step > 0)
    (hs : s < ceilDiv full step) :
    let (lo, hi) := stripOutBand full step s
    lo < hi ∧ hi ≤ full ∧ hi - lo ≤ step := by
  have hCeil : ceilDiv full step = (full + step - 1) / step := by
    unfold ceilDiv; simp [Nat.ne_of_gt hStep]
  rw [hCeil] at hs
  have hLoLt : s * step < full := by
    have h1 : (s + 1) * step ≤ (full + step - 1) / step * step :=
      Nat.mul_le_mul_right step (Nat.succ_le_of_lt hs)
    have h2 : (full + step - 1) / step * step ≤ full + step - 1 :=
      Nat.div_mul_le_self (full + step - 1) step
    rw [Nat.add_mul, Nat.one_mul] at h1
    omega
  simp only [stripOutBand]
  omega

/-- Adjacent strips are strictly contiguous: strip `s` ends at the exact row where strip `s + 1` begins. -/
theorem stripOutBand_contiguous (full step s : Nat) (hStep : step > 0)
    (hs : s + 1 < ceilDiv full step) :
    (stripOutBand full step s).2 = (stripOutBand full step (s + 1)).1 := by
  have hCeil : ceilDiv full step = (full + step - 1) / step := by
    unfold ceilDiv; simp [Nat.ne_of_gt hStep]
  rw [hCeil] at hs
  have hNextLt : (s + 1) * step < full := by
    have h1 : (s + 2) * step ≤ (full + step - 1) / step * step :=
      Nat.mul_le_mul_right step (Nat.succ_le_of_lt hs)
    have h2 : (full + step - 1) / step * step ≤ full + step - 1 :=
      Nat.div_mul_le_self (full + step - 1) step
    have h3 : (s + 2) * step = (s + 1) * step + step := by
      rw [show s + 2 = (s + 1) + 1 by omega, Nat.add_mul, Nat.one_mul]
    omega
  simp only [stripOutBand]
  rw [Nat.add_mul, Nat.one_mul]
  exact Nat.min_eq_right (Nat.le_of_lt (by rw [Nat.add_mul, Nat.one_mul] at hNextLt; exact hNextLt))

/-- Distinct strips `s1 < s2` are strictly pairwise disjoint (`hi(s1) ≤ lo(s2)`). -/
theorem stripOutBand_pairwise_disjoint (full step s1 s2 : Nat) (hlt : s1 < s2) :
    (stripOutBand full step s1).2 ≤ (stripOutBand full step s2).1 := by
  simp only [stripOutBand]
  have h1 : s1 * step + step ≤ s2 * step := by
    have hle : s1 + 1 ≤ s2 := Nat.succ_le_of_lt hlt
    calc s1 * step + step = (s1 + 1) * step := by rw [Nat.add_mul, Nat.one_mul]
      _ ≤ s2 * step := Nat.mul_le_mul_right step hle
  exact Nat.le_trans (Nat.min_le_right _ _) h1

/-- Exact unique row/column coverage: every output index `r ∈ [0, full)` belongs to
    a UNIQUE strip `s = r / step < ceilDiv full step` with `lo(s) ≤ r < hi(s)`. -/
theorem stripOutBand_unique_coverage (full step r : Nat) (hStep : step > 0) (hr : r < full) :
    let s := r / step
    s < ceilDiv full step ∧
    (stripOutBand full step s).1 ≤ r ∧ r < (stripOutBand full step s).2 ∧
    ∀ s', (stripOutBand full step s').1 ≤ r → r < (stripOutBand full step s').2 → s' = s := by
  intro s
  have hCeil : ceilDiv full step = (full + step - 1) / step := by
    unfold ceilDiv; simp [Nat.ne_of_gt hStep]
  have hDivMod : s * step + r % step = r := Nat.div_add_mod' r step
  have hModLt : r % step < step := Nat.mod_lt r hStep
  have hsLt : s < ceilDiv full step := by
    rw [hCeil]
    have hLe : (s + 1) * step ≤ full + step - 1 := by
      rw [Nat.add_mul, Nat.one_mul]; omega
    have hDiv : (s + 1) ≤ (full + step - 1) / step :=
      (Nat.le_div_iff_mul_le hStep).mpr hLe
    omega
  refine ⟨hsLt, ?_, ?_, ?_⟩
  · simp only [stripOutBand]; omega
  · simp only [stripOutBand]; omega
  · intro s' hlo hhi
    simp only [stripOutBand] at hlo hhi
    by_cases hlt : s' < s
    · have hle : s' * step + step ≤ s * step := by
        calc s' * step + step = (s' + 1) * step := by rw [Nat.add_mul, Nat.one_mul]
          _ ≤ s * step := Nat.mul_le_mul_right step (Nat.succ_le_of_lt hlt)
      omega
    · by_cases hgt : s < s'
      · have hle : s * step + step ≤ s' * step := by
          calc s * step + step = (s + 1) * step := by rw [Nat.add_mul, Nat.one_mul]
            _ ≤ s' * step := Nat.mul_le_mul_right step (Nat.succ_le_of_lt hgt)
        omega
      · omega

/-- The sum of strip heights `(hi - lo)` across all `s ∈ [0, ceilDiv full step)` equals `full`. -/
theorem stripOutBand_total_sum (full step : Nat) (hStep : step > 0) :
    ((List.range (ceilDiv full step)).map fun s =>
      (stripOutBand full step s).2 - (stripOutBand full step s).1).sum = full := by
  have hPrefix : ∀ k, ((List.range k).map fun s =>
      (stripOutBand full step s).2 - (stripOutBand full step s).1).sum = min full (k * step) := by
    intro k
    induction k with
    | zero => simp
    | succ m ih =>
      rw [List.range_succ, List.map_append, List.sum_append, ih]
      simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero, stripOutBand]
      rw [Nat.add_mul, Nat.one_mul]
      omega
  rw [hPrefix]
  have hCeil : ceilDiv full step = (full + step - 1) / step := by
    unfold ceilDiv; simp [Nat.ne_of_gt hStep]
  rw [hCeil]
  have hGe : full ≤ (full + step - 1) / step * step := by
    have h1 := Nat.div_add_mod' (full + step - 1) step
    have h2 := Nat.mod_lt (full + step - 1) hStep
    omega
  exact Nat.min_eq_left hGe

/-- Every halo input interval `haloInBand hFull rad lo hi` is non-empty, stays strictly within
    `[0, hFull]`, and encloses `[lo, hi)`. -/
theorem haloInBand_in_bounds (hFull rad lo hi : Nat) (hNonEmpty : lo < hi) (hHi : hi ≤ hFull) :
    let (inLo, inHi) := haloInBand hFull rad lo hi
    inLo ≤ lo ∧ hi ≤ inHi ∧ inLo < inHi ∧ inHi ≤ hFull := by
  simp only [haloInBand]
  split <;> omega

/-- Exact 3x3 strip output-height preservation: for `rad = 1`, `lo < hi ≤ hFull`,
    `inH = inHi - inLo` with top padding `pt = stripPadTop lo` and bottom padding
    `pb = stripPadBot hi hFull` satisfies `inH + pt + pb = (hi - lo) + 2` identically! -/
theorem conv3x3_strip_pad_exact_height (hFull lo hi : Nat) (hNonEmpty : lo < hi) (hHi : hi ≤ hFull) :
    let (inLo, inHi) := haloInBand hFull 1 lo hi
    let pt := stripPadTop lo
    let pb := stripPadBot hi hFull
    (inHi - inLo) + pt + pb = (hi - lo) + 2 := by
  unfold haloInBand stripPadTop stripPadBot
  dsimp only
  by_cases h1 : lo = 0 <;> by_cases h2 : hi = hFull
  · simp [h1, h2]
  · simp [h1, h2]; omega
  · have hge1 : lo ≥ 1 := by omega
    simp [h1, h2, hge1]; omega
  · have hge1 : lo ≥ 1 := by omega
    simp [h1, h2, hge1]; omega

/-- Exact 3x3 strip boundary tap soundness: for every output row `r ∈ [lo, hi)` and vertical
    kernel tap `dr ∈ {0, 1, 2}`:
    - top-boundary zero padding occurs iff `r = 0 ∧ dr = 0` (matching `stripPadTop lo = 1`),
    - bottom-boundary zero padding occurs iff `r = hFull - 1 ∧ dr = 2` (matching `stripPadBot hi hFull = 1`),
    - every non-padded tap row `r + dr - 1` lies strictly inside `[inLo, inHi) ⊆ [0, hFull)`! -/
theorem conv3x3_strip_tap_in_bounds (hFull lo hi r dr : Nat)
    (hHi : hi ≤ hFull) (hLoR : lo ≤ r) (hRHi : r < hi) (hDr : dr ≤ 2) :
    let (inLo, inHi) := haloInBand hFull 1 lo hi
    (r = 0 ∧ dr = 0 ∧ stripPadTop lo = 1) ∨
    (r + 1 = hFull ∧ dr = 2 ∧ stripPadBot hi hFull = 1) ∨
    (1 ≤ r + dr ∧ inLo ≤ r + dr - 1 ∧ r + dr - 1 < inHi ∧ r + dr - 1 < hFull) := by
  unfold haloInBand stripPadTop stripPadBot
  dsimp only
  by_cases hTop : r = 0 ∧ dr = 0
  · left
    refine ⟨hTop.1, hTop.2, ?_⟩
    have : lo = 0 := by omega
    simp [this]
  · by_cases hBot : r + 1 = hFull ∧ dr = 2
    · right; left
      refine ⟨hBot.1, hBot.2, ?_⟩
      have : hi = hFull := by omega
      simp [this]
    · right; right
      split <;> omega

/-- Rolling halo-stitch height exactness: in `unrollRollingHaloCascade` (`hOut ≥ 2`, `sCount > 1`),
    for every strip `s < sCount`, the stitched tensor height
    `(if s > 0 then 1 else 0) + (y1 - y0) + (if s + 1 < sCount then 1 else 0)` equals the exact
    3x3 halo input height `inHi - inLo` of `haloInBand hFull 1 y0 y1`, and both neighbor halo
    slices (bottom row of strip `s - 1`, top row of strip `s + 1`) are strictly in-bounds! -/
theorem rollingHalo_stitch_bounds (hFull hOut s : Nat) (hFullPos : hFull > 0)
    (hOutGe2 : hOut ≥ 2) (hs : s < ceilDiv hFull hOut) :
    let sCount := ceilDiv hFull hOut
    let (y0, y1) := stripOutBand hFull hOut s
    let (inLo, inHi) := haloInBand hFull 1 y0 y1
    let stitchH := (if s > 0 then 1 else 0) + (y1 - y0) + (if s + 1 < sCount then 1 else 0)
    stitchH = inHi - inLo ∧
    (s > 0 → (stripOutBand hFull hOut (s - 1)).2 - (stripOutBand hFull hOut (s - 1)).1 ≥ 1) ∧
    (s + 1 < sCount → (stripOutBand hFull hOut (s + 1)).2 - (stripOutBand hFull hOut (s + 1)).1 ≥ 1) := by
  have hStepPos : hOut > 0 := by omega
  have hCur := stripOutBand_valid hFull hOut s hFullPos hStepPos hs
  have hCeil : ceilDiv hFull hOut = (hFull + hOut - 1) / hOut := by
    unfold ceilDiv; simp [Nat.ne_of_gt hStepPos]
  refine ⟨?_, ?_, ?_⟩
  · have hTopIff : (s > 0) ↔ (stripOutBand hFull hOut s).1 ≥ 1 := by
      simp only [stripOutBand]
      constructor
      · intro hgt; have : 1 * hOut ≤ s * hOut := Nat.mul_le_mul_right hOut hgt; omega
      · intro hge; cases s with | zero => simp at hge | succ _ => omega
    have hBotIff : (s + 1 < ceilDiv hFull hOut) ↔ (stripOutBand hFull hOut s).2 < hFull := by
      constructor
      · intro hnext
        have hCont := stripOutBand_contiguous hFull hOut s hStepPos hnext
        have hN := stripOutBand_valid hFull hOut (s + 1) hFullPos hStepPos hnext
        omega
      · intro hlt
        simp only [stripOutBand] at hlt
        have h1 : (s + 1) * hOut < hFull := by rw [Nat.add_mul, Nat.one_mul]; omega
        rw [hCeil]
        have h2 : (s + 2) * hOut ≤ hFull + hOut - 1 := by
          rw [show s + 2 = (s + 1) + 1 by omega, Nat.add_mul, Nat.one_mul]; omega
        have h3 := (Nat.le_div_iff_mul_le hStepPos).mpr h2
        omega
    change (if s > 0 then 1 else 0) + ((stripOutBand hFull hOut s).2 - (stripOutBand hFull hOut s).1) +
        (if s + 1 < ceilDiv hFull hOut then 1 else 0) =
      min hFull ((stripOutBand hFull hOut s).2 + 1) -
        if (stripOutBand hFull hOut s).1 ≥ 1 then (stripOutBand hFull hOut s).1 - 1 else 0
    by_cases hp : s > 0 <;> by_cases hn : s + 1 < ceilDiv hFull hOut
    · have ht := hTopIff.mp hp
      have hb := hBotIff.mp hn
      simp only [hp, hn, ht, ↓reduceIte]
      omega
    · have ht := hTopIff.mp hp
      have hb : ¬ ((stripOutBand hFull hOut s).2 < hFull) := fun h => hn (hBotIff.mpr h)
      simp only [hp, hn, ht, ↓reduceIte]
      omega
    · have ht : ¬ ((stripOutBand hFull hOut s).1 ≥ 1) := fun h => hp (hTopIff.mpr h)
      have hb := hBotIff.mp hn
      simp only [hp, hn, ht, ↓reduceIte]
      omega
    · have ht : ¬ ((stripOutBand hFull hOut s).1 ≥ 1) := fun h => hp (hTopIff.mpr h)
      have hb : ¬ ((stripOutBand hFull hOut s).2 < hFull) := fun h => hn (hBotIff.mpr h)
      simp only [hp, hn, ht, ↓reduceIte]
      omega
  · intro hp
    have hPrev := stripOutBand_valid hFull hOut (s - 1) hFullPos hStepPos (by omega)
    omega
  · intro hn
    have hNext := stripOutBand_valid hFull hOut (s + 1) hFullPos hStepPos hn
    omega

private theorem rollingHaloLayerLagsAux_spec (radii : Array Nat) :
    ∀ rem l accLag (acc : Array Nat),
      rem = radii.size - l → l ≤ radii.size → acc.size = l →
      (l > 0 → acc[l - 1]! = accLag) →
      (∀ j, j < acc.size → j > 0 →
        acc[j]! = acc[j - 1]! + (if radii[j]! > 0 then 1 else 0)) →
      let res := rollingHaloLayerLagsAux radii l accLag acc
      res.size = radii.size ∧
      ∀ j, j < res.size → j > 0 →
        res[j]! = res[j - 1]! + (if radii[j]! > 0 then 1 else 0) := by
  intro rem
  induction rem with
  | zero =>
    intro l accLag acc hrem hl hsz _ hRec
    have hlEq : l = radii.size := by omega
    subst hlEq
    unfold rollingHaloLayerLagsAux
    simp only [Nat.lt_irrefl, ↓reduceDIte]
    exact ⟨hsz, hRec⟩
  | succ r ih =>
    intro l accLag acc hrem hl hsz hLast hRec
    have hlLt : l < radii.size := by omega
    unfold rollingHaloLayerLagsAux
    simp only [hlLt, ↓reduceDIte]
    let nextLag := if l > 0 && radii[l] > 0 then accLag + 1 else accLag
    apply ih (l + 1) nextLag (acc.push nextLag) (by omega) (by omega) (by simp [hsz])
    · intro _
      simp only [Nat.add_sub_cancel]
      rw [getElem!_pos _ l (by simp [hsz])]
      have hPushEq : (acc.push nextLag)[acc.size] = nextLag := Array.getElem_push_eq
      simpa [hsz] using hPushEq
    · intro j hj hjPos
      simp only [Array.size_push] at hj
      by_cases hjl : j < acc.size
      · have hjPrev : j - 1 < acc.size := by omega
        rw [getElem!_pos _ j (by simp; omega), getElem!_pos _ (j - 1) (by simp; omega),
            Array.getElem_push_lt hjl, Array.getElem_push_lt hjPrev]
        simpa [getElem!_pos acc j hjl, getElem!_pos acc (j - 1) hjPrev] using hRec j hjl hjPos
      · have hjEq : j = acc.size := by omega
        subst hjEq
        have hlPos : l > 0 := by omega
        have hPrev : acc.size - 1 < acc.size := by omega
        rw [getElem!_pos _ acc.size (by simp), getElem!_pos _ (acc.size - 1) (by simp; omega),
            Array.getElem_push_eq, Array.getElem_push_lt hPrev]
        have hLastEq : acc[acc.size - 1] = accLag := by
          rw [← getElem!_pos acc (acc.size - 1) hPrev, hsz]
          exact hLast hlPos
        rw [hLastEq, hsz, getElem!_pos radii l hlLt]
        dsimp only [nextLag]
        by_cases hRad : radii[l] > 0 <;> simp [hlPos, hRad]

/-- Exact size and recurrence of `Vane.Backend.Cascade.rollingHaloLayerLags` (called directly by
    `unrollRollingHaloCascade`). -/
theorem rollingHaloLayerLags_spec (radii : Array Nat) :
    (rollingHaloLayerLags radii).size = radii.size ∧
    ∀ l, l < radii.size → l > 0 →
      (rollingHaloLayerLags radii)[l]! =
        (rollingHaloLayerLags radii)[l - 1]! + (if radii[l]! > 0 then 1 else 0) := by
  unfold rollingHaloLayerLags
  have h := rollingHaloLayerLagsAux_spec radii radii.size 0 0 (Array.mkEmpty radii.size)
    (by omega) (Nat.zero_le _) (by simp) (fun h => by omega) (fun _ h => by simp at h)
  exact ⟨h.1, fun l hl hlPos => h.2 l (by rw [h.1]; exact hl) hlPos⟩

/-- Wavefront def-before-use soundness directly on `rollingHaloLayerLags` & `rollingHaloWave`
    (called by `unrollRollingHaloCascade`):
    for any layer `l > 0` (`l < radii.size`) executing strip `s` at `(wave = rollingHaloWave lags l s, layer = l)`:
    - when `radii[l] = 0`, strip `s` of layer `l - 1` ran at the same wave with strictly earlier layer `l - 1 < l`;
    - when `radii[l] > 0`, strips `s - 1` (if `s > 0`), `s`, and `s + 1` of layer `l - 1` ALL ran
      strictly earlier in lexicographic `(wave, layer)` order! -/
theorem rollingHalo_wavefront_def_before_use (radii : Array Nat) (l s : Nat)
    (hlPos : l > 0) (hlLt : l < radii.size) :
    let lags := rollingHaloLayerLags radii
    let curWave := rollingHaloWave lags l s
    (radii[l]! = 0 → rollingHaloWave lags (l - 1) s = curWave ∧ l - 1 < l) ∧
    (radii[l]! > 0 →
      (s > 0 → rollingHaloWave lags (l - 1) (s - 1) < curWave) ∧
      (rollingHaloWave lags (l - 1) s < curWave) ∧
      (rollingHaloWave lags (l - 1) (s + 1) = curWave ∧ l - 1 < l)) := by
  intro lags curWave
  have hRec := (rollingHaloLayerLags_spec radii).2 l hlLt hlPos
  dsimp only [lags, curWave, rollingHaloWave]
  constructor
  · intro h0
    rw [hRec]
    simp only [h0, Nat.lt_irrefl, ↓reduceIte, Nat.add_zero]
    exact ⟨trivial, by omega⟩
  · intro hpos
    rw [hRec]
    simp only [hpos, ↓reduceIte]
    exact ⟨fun _ => by omega, by omega, ⟨by omega, by omega⟩⟩

/-- W/sequence strip 64B DMA alignment soundness: whenever `stripW % 32 = 0`, every W-strip
    destination byte offset `2 * (stripOutBand wFull stripW s).1` is 64-byte aligned. -/
theorem wStrip_byteOffset_aligned_64 (wFull stripW s : Nat) (hAlign : stripW % 32 = 0) :
    (2 * (stripOutBand wFull stripW s).1) % 64 = 0 := by
  simp only [stripOutBand]
  have hMul : 2 * (s * stripW) = s * (2 * stripW) := by
    rw [Nat.mul_left_comm]
  rw [hMul]
  have h64 : (2 * stripW) % 64 = 0 := by omega
  rw [Nat.mul_mod, h64, Nat.mul_zero, Nat.zero_mod]

end Vane.Proofs.C3Backend
