namespace Vane.Alloc

/-- A half-open byte range `[base, base + bytes)`. -/
structure Range where
  base : Nat
  bytes : Nat
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Exclusive upper bound of a byte range (`base + bytes`). -/
def Range.hi (r : Range) : Nat :=
  r.base + r.bytes

/-- Two ranges are disjoint (non-overlapping). -/
def Apart (x y : Range) : Prop :=
  x.hi ≤ y.base ∨ y.hi ≤ x.base

instance (x y : Range) : Decidable (Apart x y) := by
  unfold Apart
  infer_instance

/-- A list of ranges is packed inside `[0, cap)` iff they are pairwise `Apart` and end by `cap`. -/
def Packed (cap : Nat) (rs : List Range) : Prop :=
  rs.Pairwise Apart ∧ ∀ r ∈ rs, r.hi ≤ cap

/-- Total byte sum of a list of ranges. -/
def bytesOf : List Range → Nat
  | [] => 0
  | r :: rs => r.bytes + bytesOf rs

/-- A buffer request with an inclusive live task interval `[first, last]` and allocation priority. -/
structure LiveBuf where
  id : Nat
  bytes : Nat
  first : Nat
  last : Nat
  priority : Nat := 0
  deriving Repr, BEq, DecidableEq, Inhabited

/-- A placed buffer assigned to `[base, base + bytes)` for inclusive live interval `[first, last]`. -/
structure PlacedBuf where
  id : Nat
  base : Nat
  bytes : Nat
  first : Nat
  last : Nat
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Exclusive upper byte offset of a placed buffer. -/
def PlacedBuf.hi (p : PlacedBuf) : Nat :=
  p.base + p.bytes

/-- Project a placed buffer to its underlying byte `Range`. -/
def PlacedBuf.toRange (p : PlacedBuf) : Range :=
  { base := p.base, bytes := p.bytes }

/-- A placed buffer is live at task `t` iff `first ≤ t ≤ last`. -/
def PlacedBuf.live (p : PlacedBuf) (t : Nat) : Prop :=
  p.first ≤ t ∧ t ≤ p.last

instance (p : PlacedBuf) (t : Nat) : Decidable (p.live t) := by
  unfold PlacedBuf.live
  infer_instance

/-- Two closed task intervals `[f1, l1]` and `[f2, l2]` intersect. -/
def meets (f1 l1 f2 l2 : Nat) : Prop :=
  f1 ≤ l2 ∧ f2 ≤ l1

instance (f1 l1 f2 l2 : Nat) : Decidable (meets f1 l1 f2 l2) := by
  unfold meets
  infer_instance

/-- Two placed buffers have overlapping live intervals. -/
def PlacedBuf.conflicts (a b : PlacedBuf) : Prop :=
  meets a.first a.last b.first b.last

instance (a b : PlacedBuf) : Decidable (a.conflicts b) := by
  unfold PlacedBuf.conflicts
  infer_instance

/-- Every pair of live-conflicting placed buffers occupies disjoint byte ranges. -/
def Disjoint (ps : List PlacedBuf) : Prop :=
  ps.Pairwise (fun a b => a.conflicts b → Apart a.toRange b.toRange)

/-- Highest exclusive end offset across all placed buffers (`0` if empty). -/
def topOf : List PlacedBuf → Nat
  | [] => 0
  | p :: ps => max p.hi (topOf ps)

/-- All placed buffers whose live interval contains task `t`, projected to `Range`. -/
def liveAt (ps : List PlacedBuf) (t : Nat) : List Range :=
  (ps.filter (fun p => decide (p.live t))).map PlacedBuf.toRange

/-- Check whether placing a buffer of `bytes` bytes at `off` is disjoint from every buffer in `ps`. -/
def clearFor (ps : List PlacedBuf) (bytes off : Nat) : Bool :=
  ps.all (fun p => decide (Apart p.toRange { base := off, bytes := bytes }))

/-- Choose the lowest candidate offset (`0` or some `p.hi`) clear of all buffers in `conf`. -/
def pickFirstFit (conf : List PlacedBuf) (bytes : Nat) : Nat :=
  let cands := 0 :: conf.map PlacedBuf.hi
  let valid := cands.filter (fun off => clearFor conf bytes off)
  valid.foldl min (topOf conf)

/-- Place a single `LiveBuf` against already-placed buffers `ps`. -/
def placeOne (firstFit : Bool) (ps : List PlacedBuf) (b : LiveBuf) : PlacedBuf :=
  let conf := ps.filter (fun p => decide (meets b.first b.last p.first p.last))
  let off := if firstFit then pickFirstFit conf b.bytes else topOf conf
  { id := b.id, base := off, bytes := b.bytes, first := b.first, last := b.last }

/-- Place a list of `LiveBuf` requests sequentially onto an accumulator list. -/
def placeAll (firstFit : Bool) : List LiveBuf → List PlacedBuf → List PlacedBuf
  | [], acc => acc
  | b :: rest, acc => placeAll firstFit rest (placeOne firstFit acc b :: acc)

/-- Deterministic sorting comparator for `LiveBuf`: descending by `(bytes, priority)`,
    breaking ties by ascending `id`. -/
def liveBufBefore (a b : LiveBuf) : Bool :=
  if a.bytes != b.bytes then a.bytes > b.bytes
  else if a.priority != b.priority then a.priority > b.priority
  else a.id < b.id

/-- Liveness-based buffer allocator.
    When `firstFit = true`, reuses the lowest non-overlapping gap across disjoint live intervals;
    when `firstFit = false`, stacks above the highest conflicting live buffer.
    If all input `bytes` are multiples of 64, every returned `base` is automatically 64-byte aligned. -/
def allocate (firstFit : Bool) (bs : List LiveBuf) : List PlacedBuf :=
  let sorted := (bs.toArray.qsort liveBufBefore).toList
  (placeAll firstFit sorted []).reverse

/-- Find the lowest offset in `[0, cap)` (at `0` or an existing `r.hi`) where a temporary
    buffer of `bytes` bytes fits without overlapping any range in `occupied`. -/
def findFreeOffset (occupied : List Range) (bytes cap : Nat) : Option Nat :=
  let cands := 0 :: occupied.map Range.hi
  let valid := cands.filter (fun off =>
    off + bytes <= cap &&
    occupied.all (fun r => decide (Apart r { base := off, bytes := bytes })))
  match valid with
  | [] => none
  | first :: rest => some (rest.foldl min first)

end Vane.Alloc
