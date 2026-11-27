import Vane.IR.Semantics

namespace Vane.IR

-- ============================================================================
-- Fuel-Bounded Declarative Peephole Rewrite Engine & Proof Combinators
-- ============================================================================

/-- A declarative peephole rewrite rule over `SSAGraph Op`.
    `apply g` returns `some g'` when the pattern matches and fires once, or `none` otherwise. -/
structure RewriteRule (Op : Type) where
  name : String
  apply : SSAGraph Op → Option (SSAGraph Op)

/-- Structural well-formedness preservation predicate for a graph transformation `f`. -/
def PreservesWF {Op : Type} (f : SSAGraph Op → SSAGraph Op) : Prop :=
  ∀ g, WellFormed g → WellFormed (f g)

/-- Forward semantic refinement predicate (`SemRefines`) for a graph transformation `f`
    under semantics `sem`: whenever `g` evaluates to `outs`, `f g` evaluates to the exact same `outs`. -/
def PreservesSem {Op α : Type} (sem : OpSemantics Op α) (f : SSAGraph Op → SSAGraph Op) : Prop :=
  ∀ g, WellFormed g → SemRefines sem g (f g)

/-- Structural well-formedness preservation for an individual `RewriteRule`. -/
def RewriteRule.PreservesWF {Op : Type} (r : RewriteRule Op) : Prop :=
  ∀ g g', WellFormed g → r.apply g = some g' → WellFormed g'

/-- Forward semantic refinement (`SemRefines`) for an individual `RewriteRule`:
    whenever `g` evaluates to `outs` and `r.apply g = some g'`, `g'` evaluates to the exact same `outs`. -/
def RewriteRule.PreservesSem {Op α : Type} (sem : OpSemantics Op α) (r : RewriteRule Op) : Prop :=
  ∀ g g', WellFormed g → r.apply g = some g' → SemRefines sem g g'

/-- Try each rule in `rules` in order; return the first successful rewrite. -/
def applyFirstRule {Op : Type} (rules : List (RewriteRule Op)) (g : SSAGraph Op) :
    Option (SSAGraph Op) :=
  match rules with
  | [] => none
  | r :: rest =>
    match r.apply g with
    | some g' => some g'
    | none => applyFirstRule rest g

/-- Pure, structurally total, fuel-bounded fixed-point peephole rewrite engine.
    Repeatedly applies the first matching rule in `rules` until either a fixed point
    is reached (`applyFirstRule` returns `none`) or `fuel` reaches `0`. -/
def applyRewrites {Op : Type} (fuel : Nat) (rules : List (RewriteRule Op))
    (g : SSAGraph Op) : SSAGraph Op :=
  match fuel with
  | 0 => g
  | fuel' + 1 =>
    match applyFirstRule rules g with
    | some g' => applyRewrites fuel' rules g'
    | none => g

/-- Sequential composition of total graph passes. -/
def runPasses {Op : Type} (passes : List (SSAGraph Op → SSAGraph Op))
    (g : SSAGraph Op) : SSAGraph Op :=
  match passes with
  | [] => g
  | p :: ps => runPasses ps (p g)

end Vane.IR
