namespace Vane.Reach

/-- Return `true` iff `p j = true` for some `j < m`. -/
def anyLt (p : Nat → Bool) : Nat → Bool
  | 0 => false
  | m + 1 => anyLt p m || p m

/-- Build rows `0 .. m - 1` of the forward-edge transitive-closure matrix for `n` nodes.
    Row `i` records for every `k < n` whether `k` can reach `i` via one or more forward edges
    (`step a b = true → a < b`). -/
def reachTableAux (step : Nat → Nat → Bool) (n : Nat) : Nat → Array (Array Bool)
  | 0 => #[]
  | m + 1 =>
    let prev := reachTableAux step n m
    let row := Array.ofFn (n := n) fun k =>
      step k.val m ||
        anyLt (fun j =>
          step j m &&
            match prev[j]? with
            | some r => r.getD k.val false
            | none => false) m
    prev.push row

/-- Compute the `n × n` boolean transitive-closure table for a forward-edge DAG relation `step`. -/
def reachTable (step : Nat → Nat → Bool) (n : Nat) : Array (Array Bool) :=
  reachTableAux step n n

/-- Look up whether node `k` reaches node `i` in a transitive-closure table built by `reachTable`. -/
def reachB (tbl : Array (Array Bool)) (k i : Nat) : Bool :=
  match tbl[i]? with
  | some row => row.getD k false
  | none => false

end Vane.Reach
