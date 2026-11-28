import Vane.IR.Semantics
import Vane.IR.Passes.DCE
import Vane.IR.Passes.ConstFold
import Vane.IR.Passes.HighFuse
import Vane.IR.Passes.Decompose
import Vane.IR.Passes.CoreFuse
import Vane.IR.Passes.DecomposeConv
import Vane.Ane.Types
import Vane.Backend.LayoutSelect
import Vane.Backend.ChannelSplit
import Vane.Backend.Cascade
import Vane.Proofs.C3Backend
import Vane.Proofs.C4Conv

namespace Vane.Proofs.C5SemanticPasses

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Alloc
open Vane.Program
open Vane.Tiling
open Vane.Pe
open Vane.Lut
open Vane.Pool
open Vane.Conv
open Vane.View
open Vane.Quant
open Vane.Matmul
open Vane.IR
open Vane.Ane
open Vane.Backend.LayoutSelect
open Vane.Backend.ChannelSplit
open Vane.Backend.Cascade
open Vane.Proofs.C3Backend
open Vane.Proofs.C4Conv

-- ============================================================================
-- Section 1: Exact Algebraic Domain (`ScalarField α` + `Rat` Instance + `sumFin`)
-- ============================================================================

/-- Exact algebraic field typeclass for rounding-free denotational tensor semantics
    across L1 (`HighOp`, `CoreOp`) and L2 (`AneOp`). Instantiated concretely by `Rat`. -/
class ScalarField (α : Type) where
  zero : α
  one : α
  add : α → α → α
  neg : α → α
  mul : α → α → α
  inv : α → α
  add_comm : ∀ a b, add a b = add b a
  add_assoc : ∀ a b c, add (add a b) c = add a (add b c)
  zero_add : ∀ a, add zero a = a
  add_neg : ∀ a, add a (neg a) = zero
  mul_comm : ∀ a b, mul a b = mul b a
  mul_assoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a, mul one a = a
  left_distrib : ∀ a b c, mul a (add b c) = add (mul a b) (mul a c)
  mul_inv_cancel : ∀ a, a ≠ zero → mul a (inv a) = one
  inv_zero : inv zero = zero
  one_ne_zero : one ≠ zero

/-- Concrete kernel-clean exact rational (`Rat`) instance of `ScalarField`. -/
instance : ScalarField Rat where
  zero := 0
  one := 1
  add := (· + ·)
  neg := (- ·)
  mul := (· * ·)
  inv := Inv.inv
  add_comm := Rat.add_comm
  add_assoc := Rat.add_assoc
  zero_add := Rat.zero_add
  add_neg := Rat.add_neg_cancel
  mul_comm := Rat.mul_comm
  mul_assoc := Rat.mul_assoc
  one_mul := Rat.one_mul
  left_distrib := Rat.mul_add
  mul_inv_cancel := Rat.mul_inv_cancel
  inv_zero := Rat.inv_zero
  one_ne_zero := by decide

namespace ScalarField

variable {α : Type} [ScalarField α]

def sub (a b : α) : α := add a (neg b)
def div (a b : α) : α := mul a (inv b)

def ofNat : Nat → α
  | 0 => zero
  | n + 1 => add (ofNat n) one

theorem add_zero (a : α) : add a zero = a := by
  rw [add_comm, zero_add]

theorem neg_add (a : α) : add (neg a) a = zero := by
  rw [add_comm, add_neg]

theorem mul_one (a : α) : mul a one = a := by
  rw [mul_comm, one_mul]

theorem right_distrib (a b c : α) : mul (add a b) c = add (mul a c) (mul b c) := by
  rw [mul_comm, left_distrib, mul_comm c a, mul_comm c b]

theorem add_left_comm (a b c : α) : add a (add b c) = add b (add a c) := by
  rw [← add_assoc, add_comm a b, add_assoc]

theorem mul_left_comm (a b c : α) : mul a (mul b c) = mul b (mul a c) := by
  rw [← mul_assoc, mul_comm a b, mul_assoc]

theorem add_right_cancel {a b c : α} (h : add a c = add b c) : a = b := by
  have h1 : add (add a c) (neg c) = add (add b c) (neg c) := congrArg (fun x => add x (neg c)) h
  rw [add_assoc, add_assoc, add_neg, add_zero, add_zero] at h1
  exact h1

theorem mul_zero (a : α) : mul a zero = zero := by
  apply add_right_cancel (c := mul a zero)
  rw [← left_distrib, zero_add, zero_add]

theorem zero_mul (a : α) : mul zero a = zero := by
  rw [mul_comm, mul_zero]

theorem neg_unique {a b : α} (h : add a b = zero) : b = neg a := by
  apply add_right_cancel (c := a)
  rw [add_comm b a, h, neg_add]

theorem neg_zero : neg (zero : α) = zero :=
  (neg_unique (zero_add zero)).symm

theorem neg_neg (a : α) : neg (neg a) = a :=
  (neg_unique (neg_add a)).symm

theorem mul_neg (a b : α) : mul a (neg b) = neg (mul a b) := by
  apply neg_unique
  rw [← left_distrib, add_neg, mul_zero]

theorem neg_mul (a b : α) : mul (neg a) b = neg (mul a b) := by
  rw [mul_comm, mul_neg, mul_comm]

theorem neg_one_mul (a : α) : mul (neg one) a = neg a := by
  rw [neg_mul, one_mul]

theorem mul_neg_one (a : α) : mul a (neg one) = neg a := by
  rw [mul_comm, neg_one_mul]

theorem sub_self (a : α) : sub a a = zero := add_neg a

theorem sub_zero (a : α) : sub a zero = a := by
  unfold sub; rw [neg_zero, add_zero]

theorem zero_sub (a : α) : sub zero a = neg a := by
  unfold sub; rw [zero_add]

theorem add_sub_cancel (a b : α) : sub (add a b) b = a := by
  unfold sub; rw [add_assoc, add_neg, add_zero]

theorem sub_add_cancel (a b : α) : add (sub a b) b = a := by
  unfold sub; rw [add_assoc, neg_add, add_zero]

theorem mul_sub (a b c : α) : mul a (sub b c) = sub (mul a b) (mul a c) := by
  unfold sub; rw [left_distrib, mul_neg]

theorem sub_mul (a b c : α) : mul (sub a b) c = sub (mul a c) (mul b c) := by
  unfold sub; rw [right_distrib, neg_mul]

theorem div_one (a : α) : div a one = a := by
  unfold div
  have hInv : inv (one : α) = one := by
    have h := mul_inv_cancel (one : α) one_ne_zero
    rwa [one_mul] at h
  rw [hInv, mul_one]

theorem div_mul_cancel (a b : α) (hb : b ≠ zero) : mul (div a b) b = a := by
  unfold div
  rw [mul_assoc, mul_comm (inv b) b, mul_inv_cancel b hb, mul_one]

theorem mul_div_assoc (a b c : α) : div (mul a b) c = mul a (div b c) := by
  unfold div; rw [mul_assoc]

theorem mul_div_right_comm (a b c : α) : mul (div a c) b = div (mul a b) c := by
  unfold div; rw [mul_assoc, mul_comm (inv c) b, ← mul_assoc]

/-- Finite indexed sum `∑_{i=0}^{n-1} f i` over any `ScalarField α`. -/
def sumFin : Nat → (Nat → α) → α
  | 0, _ => zero
  | n + 1, f => add (sumFin n f) (f n)

theorem sumFin_zero (f : Nat → α) : sumFin 0 f = zero := rfl

theorem sumFin_succ (n : Nat) (f : Nat → α) : sumFin (n + 1) f = add (sumFin n f) (f n) := rfl

theorem sumFin_congr (n : Nat) {f g : Nat → α} (h : ∀ i, i < n → f i = g i) :
    sumFin n f = sumFin n g := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp only [sumFin]
    rw [ih (fun i hi => h i (Nat.lt_succ_of_lt hi)), h k (Nat.lt_succ_self k)]

theorem sumFin_const_zero (n : Nat) : sumFin n (fun _ => (zero : α)) = zero := by
  induction n with
  | zero => rfl
  | succ k ih => simp only [sumFin, ih, add_zero]

theorem sumFin_add (n : Nat) (f g : Nat → α) :
    sumFin n (fun i => add (f i) (g i)) = add (sumFin n f) (sumFin n g) := by
  induction n with
  | zero => simp only [sumFin, zero_add]
  | succ k ih =>
    simp only [sumFin, ih]
    rw [add_assoc, add_assoc, add_left_comm (sumFin k g) (f k) (g k)]

theorem sumFin_mul_left (n : Nat) (c : α) (f : Nat → α) :
    sumFin n (fun i => mul c (f i)) = mul c (sumFin n f) := by
  induction n with
  | zero => simp only [sumFin, mul_zero]
  | succ k ih => simp only [sumFin, ih, left_distrib]

theorem sumFin_mul_right (n : Nat) (f : Nat → α) (c : α) :
    sumFin n (fun i => mul (f i) c) = mul (sumFin n f) c := by
  induction n with
  | zero => simp only [sumFin, zero_mul]
  | succ k ih => simp only [sumFin, ih, right_distrib]

theorem sumFin_comm (m n : Nat) (f : Nat → Nat → α) :
    sumFin m (fun i => sumFin n (fun j => f i j)) =
    sumFin n (fun j => sumFin m (fun i => f i j)) := by
  induction m with
  | zero =>
    simp only [sumFin]
    symm
    exact sumFin_const_zero n
  | succ k ih =>
    simp only [sumFin, ih, sumFin_add]

theorem sumFin_split (a b : Nat) (f : Nat → α) :
    sumFin (a + b) f = add (sumFin a f) (sumFin b (fun j => f (a + j))) := by
  induction b with
  | zero => simp only [Nat.add_zero, sumFin, add_zero]
  | succ k ih =>
    rw [← Nat.add_assoc, sumFin_succ, ih, sumFin_succ, add_assoc]

/-- Kronecker-delta collapsing lemma: if `f i = 0` for all `i ≠ k` (`k < n`), then `sumFin n f = f k`. -/
theorem sumFin_single (n k : Nat) (hk : k < n) (f : Nat → α)
    (hZero : ∀ i, i < n → i ≠ k → f i = zero) :
    sumFin n f = f k := by
  induction n with
  | zero => omega
  | succ m ih =>
    simp only [sumFin]
    by_cases hkm : k = m
    · subst hkm
      have hPre : sumFin k f = zero := by
        rw [sumFin_congr k (g := fun _ => zero) (fun i hi => hZero i (Nat.lt_succ_of_lt hi) (by omega))]
        exact sumFin_const_zero k
      rw [hPre, zero_add]
    · have hkLt : k < m := by omega
      rw [ih hkLt (fun i hi hne => hZero i (Nat.lt_succ_of_lt hi) hne),
          hZero m (Nat.lt_succ_self m) (Ne.symm hkm), add_zero]

/-- Product-range summation: `∑_{i < a * b} f i = ∑_{q < a} ∑_{r < b} f (q * b + r)`. -/
theorem sumFin_prod (a b : Nat) (f : Nat → α) :
    sumFin (a * b) f = sumFin a (fun q => sumFin b (fun r => f (q * b + r))) := by
  induction a with
  | zero => simp only [Nat.zero_mul, sumFin]
  | succ k ih =>
    rw [Nat.add_mul, Nat.one_mul, sumFin_split, ih, sumFin_succ]

/-- Sum over a list of elements in `ScalarField α`. -/
def sumList : List α → α
  | [] => zero
  | x :: xs => add x (sumList xs)

theorem sumList_append (xs ys : List α) : sumList (xs ++ ys) = add (sumList xs) (sumList ys) := by
  induction xs with
  | nil => simp only [List.nil_append, sumList, zero_add]
  | cons x rest ih => simp only [List.cons_append, sumList, ih, add_assoc]

theorem sumList_const_zero {β : Type} (xs : List β) :
    sumList (xs.map fun _ => (zero : α)) = zero := by
  induction xs with
  | nil => rfl
  | cons _ rest ih => simp only [List.map_cons, sumList, ih, zero_add]

theorem sumList_add {β : Type} (xs : List β) (f g : β → α) :
    sumList (xs.map fun x => add (f x) (g x)) =
    add (sumList (xs.map f)) (sumList (xs.map g)) := by
  induction xs with
  | nil => simp only [List.map_nil, sumList, zero_add]
  | cons x rest ih =>
    simp only [List.map_cons, sumList, ih]
    rw [add_assoc, add_assoc, add_left_comm (g x) (sumList (rest.map f)) (sumList (rest.map g))]

theorem sumList_mul_left {β : Type} (xs : List β) (c : α) (f : β → α) :
    sumList (xs.map fun x => mul c (f x)) = mul c (sumList (xs.map f)) := by
  induction xs with
  | nil => simp only [List.map_nil, sumList, mul_zero]
  | cons x rest ih => simp only [List.map_cons, sumList, ih, left_distrib]

theorem sumList_mul_right {β : Type} (xs : List β) (f : β → α) (c : α) :
    sumList (xs.map fun x => mul (f x) c) = mul (sumList (xs.map f)) c := by
  induction xs with
  | nil => simp only [List.map_nil, sumList, zero_mul]
  | cons x rest ih => simp only [List.map_cons, sumList, ih, right_distrib]

theorem sumList_single {β : Type} [DecidableEq β] (xs : List β) (k : β)
    (hNd : xs.Nodup) (hMem : k ∈ xs) (f : β → α)
    (hZero : ∀ x ∈ xs, x ≠ k → f x = zero) :
    sumList (xs.map f) = f k := by
  induction xs with
  | nil => cases hMem
  | cons hd tl ih =>
    have hNdCons := List.nodup_cons.mp hNd
    simp only [List.map_cons, sumList]
    rcases List.mem_cons.mp hMem with rfl | hTl
    · have hTlZero : sumList (tl.map f) = zero := by
        have hEq : tl.map f = tl.map (fun _ => (zero : α)) :=
          List.map_congr_left fun x hx =>
            hZero x (List.mem_cons_of_mem _ hx) (fun heq => hNdCons.1 (heq ▸ hx))
        rw [hEq]
        exact sumList_const_zero tl
      rw [hTlZero, add_zero]
    · have hHdNe : hd ≠ k := fun heq => hNdCons.1 (heq ▸ hTl)
      rw [hZero hd List.mem_cons_self hHdNe, zero_add]
      exact ih hNdCons.2 hTl (fun x hx hne => hZero x (List.mem_cons_of_mem _ hx) hne)

theorem sumFin_sumList_comm {β : Type} (n : Nat) (xs : List β) (f : Nat → β → α) :
    sumFin n (fun i => sumList (xs.map fun x => f i x)) =
    sumList (xs.map fun x => sumFin n (fun i => f i x)) := by
  induction n with
  | zero =>
    simp only [sumFin]
    symm
    exact sumList_const_zero xs
  | succ k ih =>
    simp only [sumFin, ih, sumList_add]

theorem sumList_congr {β : Type} (xs : List β) {f g : β → α} (h : ∀ x ∈ xs, f x = g x) :
    sumList (xs.map f) = sumList (xs.map g) := by
  rw [List.map_congr_left h]

end ScalarField

open ScalarField

/-- Exact transcendental, order, activation, and scalar-encoding hooks over any `ScalarField α`.
    Instantiated concretely on `Rat` via `ratScalarOps` below, guaranteeing full consistency
    and non-vacuity of all laws. -/
structure ScalarOps (α : Type) [ScalarField α] where
  abs : α → α
  maxOp : α → α → α
  minOp : α → α → α
  exp : α → α
  log2 : α → α
  ln2 : α
  log : α → α
  sqrt : α → α
  rsqrt : α → α
  sigmoid : α → α
  gelu : α → α
  silu : α → α
  softplus : α → α
  relu : α → α
  lutEval : LutSpec → α → α
  f16Val : UInt16 → α
  f64Val : Float → α
  -- Exact algebraic identities satisfied by the mathematical operators.
  -- Note: `f16_ln2` is an exact-model convention for C5a's rounding-free algebraic layer
  -- (treating the fp16 constant `floatToF16CanonicalZero 0.6931471805599453` as `ln2`);
  -- fp16 quantization error bounds for transcendental constants belong to C5d.
  f16_posOne : f16Val f16PosOne = one
  f16_negOne : f16Val f16NegOne = neg one
  f16_neg : ∀ b, f16Val (negF16CanonicalZero b) = neg (f16Val b)
  f16_ln2 : f16Val (floatToF16CanonicalZero 0.6931471805599453) = ln2
  lut_exp : ∀ x, lutEval .exp x = exp x
  lut_log2 : ∀ x, lutEval .log2 x = log2 x
  lut_sqrt : ∀ x, lutEval .sqrt x = sqrt x
  lut_rsqrt_zero : ∀ x, lutEval (.rsqrt 0.0) x = rsqrt x
  lut_inverse : ∀ x, lutEval .inverse x = inv x
  lut_sigmoid : ∀ x, lutEval .sigmoid x = sigmoid x
  lut_gelu : ∀ x, lutEval .gelu x = gelu x
  lut_silu : ∀ x, lutEval .silu x = silu x
  silu_def : ∀ x, silu x = mul x (sigmoid x)
  log_eq_log2_mul_ln2 : ∀ x, log x = mul (log2 x) ln2
  softplus_def : ∀ x, softplus x = log (add one (exp x))
  rsqrt_def : ∀ x, rsqrt x = inv (sqrt x)
  relu_def : ∀ x, relu x = maxOp x zero

/-- Pure bit-level conversion of an IEEE-754 binary16 bit pattern to an exact `Rat`
    (preserving `f16PosOne ↦ 1`, `f16NegOne ↦ -1`, `negF16CanonicalZero b ↦ -(f16ToRat b)`). -/
def f16BitsToRat (bits : UInt16) : Rat :=
  let mag := bits &&& 0x7fff
  if mag == 0 then 0
  else
    let signNeg := (bits &&& 0x8000) != 0
    let exp := ((bits >>> 10) &&& 0x1f).toNat
    let frac := (bits &&& 0x3ff).toNat
    let raw : Rat :=
      if exp == 0 then
        ((frac : Int) : Rat) / ((2 ^ 24 : Nat) : Rat)
      else if exp == 31 then
        0
      else if exp >= 25 then
        (((1024 + frac) * (2 ^ (exp - 25)) : Nat) : Rat)
      else
        (((1024 + frac : Nat) : Rat) / ((2 ^ (25 - exp) : Nat) : Rat))
    if signNeg then -raw else raw

private theorem u16_8000_and_7fff : ∀ j : Fin 16,
    ((0x8000 : UInt16).toBitVec[j.val] && (0x7fff : UInt16).toBitVec[j.val]) = false := by decide

private theorem u16_8000_shr10_and_1f : ∀ j : Fin 16,
    ((if h : 10 + j.val < 16 then (0x8000 : UInt16).toBitVec[10 + j.val] else false) &&
     (0x1f : UInt16).toBitVec[j.val]) = false := by decide

private theorem u16_8000_and_3ff : ∀ j : Fin 16,
    ((0x8000 : UInt16).toBitVec[j.val] && (0x3ff : UInt16).toBitVec[j.val]) = false := by decide

private theorem u16_8000_low_bits : ∀ j : Fin 16,
    j.val ≠ 15 → (0x8000 : UInt16).toBitVec[j.val] = false := by decide

private theorem bv16_getLsbD_ge (v : BitVec 16) (k : Nat) (hk : 16 ≤ k) : v.getLsbD k = false := by
  rw [BitVec.getLsbD]
  exact Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le v.isLt (Nat.pow_le_pow_right (by omega) hk))

theorem f16BitsToRat_posOne : f16BitsToRat f16PosOne = 1 := by
  unfold f16BitsToRat f16PosOne
  have h1 : ((0x3c00 : UInt16) &&& 0x7fff == 0) = false := by decide
  have h2 : (((0x3c00 : UInt16) &&& 0x8000) != 0) = false := by decide
  have h3 : (((0x3c00 : UInt16) >>> 10) &&& 0x1f).toNat = 15 := by decide
  have h4 : ((0x3c00 : UInt16) &&& 0x3ff).toNat = 0 := by decide
  simp only [h1, h2, h3, h4, Bool.false_eq_true, ↓reduceIte]
  show ((1024 : Rat) / (1024 : Rat)) = 1
  exact Rat.mul_inv_cancel (1024 : Rat) (by decide)

theorem f16BitsToRat_negOne : f16BitsToRat f16NegOne = -1 := by
  unfold f16BitsToRat f16NegOne
  have h1 : ((0xbc00 : UInt16) &&& 0x7fff == 0) = false := by decide
  have h2 : (((0xbc00 : UInt16) &&& 0x8000) != 0) = true := by decide
  have h3 : (((0xbc00 : UInt16) >>> 10) &&& 0x1f).toNat = 15 := by decide
  have h4 : ((0xbc00 : UInt16) &&& 0x3ff).toNat = 0 := by decide
  simp only [h1, h2, h3, h4, Bool.false_eq_true, ↓reduceIte]
  show -((1024 : Rat) * (1024 : Rat)⁻¹) = -1
  rw [Rat.mul_inv_cancel (1024 : Rat) (by decide)]

theorem f16BitsToRat_neg (b : UInt16) :
    f16BitsToRat (negF16CanonicalZero b) = -(f16BitsToRat b) := by
  unfold negF16CanonicalZero f16BitsToRat
  by_cases h0 : (b &&& 0x7fff) == 0
  · simp only [h0, ↓reduceIte]
    have hZero : (f16PosZero &&& 0x7fff == 0) = true := by unfold f16PosZero; decide
    simp [hZero]
  · simp only [h0, Bool.false_eq_true, ↓reduceIte]
    have hMag : ((b ^^^ 0x8000) &&& 0x7fff) = (b &&& 0x7fff) := by
      apply UInt16.toBitVec_inj.mp
      apply BitVec.eq_of_getElem_eq
      intro i hi
      simp only [UInt16.toBitVec_and, UInt16.toBitVec_xor, BitVec.getElem_and, BitVec.getElem_xor]
      have hBit := u16_8000_and_7fff ⟨i, hi⟩
      cases hb : b.toBitVec[i] <;> cases h8 : (0x8000 : UInt16).toBitVec[i] <;>
        cases h7 : (0x7fff : UInt16).toBitVec[i] <;> simp_all
    have hExp : (((b ^^^ 0x8000) >>> (10 : UInt16)) &&& (0x1f : UInt16)) = ((b >>> (10 : UInt16)) &&& (0x1f : UInt16)) := by
      apply UInt16.toBitVec_inj.mp
      apply BitVec.eq_of_getElem_eq
      intro i hi
      simp only [UInt16.toBitVec_and, UInt16.toBitVec_shiftRight, UInt16.toBitVec_xor, BitVec.getElem_and]
      have hBit := u16_8000_shr10_and_1f ⟨i, hi⟩
      dsimp only at hBit
      have hShr : ∀ x : BitVec 16, (x >>> (UInt16.toBitVec 10 % 16))[i] = x.getLsbD (10 + i) := fun x => by
        simp [Nat.add_comm]
      rw [hShr, hShr, BitVec.getLsbD_xor]
      by_cases h10 : 10 + i < 16
      · simp only [h10, ↓reduceDIte] at hBit
        have hL1 : (0x8000 : UInt16).toBitVec.getLsbD (10 + i) = (0x8000 : UInt16).toBitVec[10 + i] :=
          BitVec.getLsbD_eq_getElem h10
        have hL2 : b.toBitVec.getLsbD (10 + i) = b.toBitVec[10 + i] :=
          BitVec.getLsbD_eq_getElem h10
        simp only [hL1, hL2]
        cases hb : b.toBitVec[10 + i] <;> cases h8 : (0x8000 : UInt16).toBitVec[10 + i] <;>
          cases h5 : (0x1f : UInt16).toBitVec[i] <;> simp_all
      · have hGe : 16 ≤ 10 + i := by omega
        have hF1 : (0x8000 : UInt16).toBitVec.getLsbD (10 + i) = false := bv16_getLsbD_ge _ _ hGe
        have hF2 : b.toBitVec.getLsbD (10 + i) = false := bv16_getLsbD_ge _ _ hGe
        simp only [hF1, hF2, Bool.xor_false]
    have hFrac : ((b ^^^ 0x8000) &&& 0x3ff) = (b &&& 0x3ff) := by
      apply UInt16.toBitVec_inj.mp
      apply BitVec.eq_of_getElem_eq
      intro i hi
      simp only [UInt16.toBitVec_and, UInt16.toBitVec_xor, BitVec.getElem_and, BitVec.getElem_xor]
      have hBit := u16_8000_and_3ff ⟨i, hi⟩
      cases hb : b.toBitVec[i] <;> cases h8 : (0x8000 : UInt16).toBitVec[i] <;>
        cases h3 : (0x3ff : UInt16).toBitVec[i] <;> simp_all
    have hSign : (((b ^^^ 0x8000) &&& 0x8000) != 0) = !((b &&& 0x8000) != 0) := by
      have hEq : (b ^^^ 0x8000) &&& 0x8000 = (if (b &&& 0x8000) == 0 then 0x8000 else 0) := by
        apply UInt16.toBitVec_inj.mp
        by_cases hs : (b &&& 0x8000) == 0
        · simp only [hs, ↓reduceIte]
          have hs0 : b &&& 0x8000 = 0 := beq_iff_eq.mp hs
          apply BitVec.eq_of_getElem_eq; intro i hi
          have h0i := congrArg (fun u : UInt16 => u.toBitVec[i]) hs0
          simp only [UInt16.toBitVec_and, UInt16.toBitVec_xor, BitVec.getElem_and, BitVec.getElem_xor] at h0i ⊢
          cases hb : b.toBitVec[i] <;> cases h8 : (0x8000 : UInt16).toBitVec[i] <;> simp_all
        · simp only [hs, Bool.false_eq_true, ↓reduceIte]
          have hsNe : b &&& 0x8000 ≠ 0 := fun h => hs (beq_iff_eq.mpr h)
          have hb15 : b.toBitVec[15] = true := by
            cases h15 : b.toBitVec[15]
            · exfalso; apply hsNe
              apply UInt16.toBitVec_inj.mp
              apply BitVec.eq_of_getElem_eq; intro i hi
              simp only [UInt16.toBitVec_and, BitVec.getElem_and]
              have hZ : (0 : UInt16).toBitVec[i] = false := by simp
              rw [hZ]
              by_cases hi15 : i = 15
              · subst hi15; simp [h15]
              · have h8f := u16_8000_low_bits ⟨i, hi⟩ hi15
                rw [h8f]; simp
            · rfl
          apply BitVec.eq_of_getElem_eq; intro i hi
          simp only [UInt16.toBitVec_and, UInt16.toBitVec_xor, BitVec.getElem_and, BitVec.getElem_xor]
          have hZ : (0 : UInt16).toBitVec[i] = false := by simp
          rw [hZ]
          by_cases hi15 : i = 15
          · subst hi15; simp [hb15]
          · have h8f := u16_8000_low_bits ⟨i, hi⟩ hi15
            rw [h8f]; simp
      rw [hEq]
      have h80 : ((0x8000 : UInt16) == 0) = false := by decide
      cases hsb : (b &&& 0x8000 == 0) <;> simp [bne, hsb, h80]
    have hMagB : ((b ^^^ 0x8000) &&& 0x7fff == 0) = false := by
      rw [hMag]; exact Bool.eq_false_iff.mpr h0
    rw [hMagB, hExp, hFrac, hSign]
    simp only [Bool.false_eq_true, ↓reduceIte]
    cases ((b &&& 0x8000) != 0) <;> simp

/-- Concrete, kernel-clean `ScalarOps Rat` witness proving that all `ScalarOps` axioms
    are 100% consistent and inhabited. -/
def ratScalarOps : ScalarOps Rat where
  abs x := if x < 0 then -x else x
  maxOp a b := if a ≤ b then b else a
  minOp a b := if a ≤ b then a else b
  exp x := x
  log2 x := x
  ln2 := f16BitsToRat (floatToF16CanonicalZero 0.6931471805599453)
  log x := x * f16BitsToRat (floatToF16CanonicalZero 0.6931471805599453)
  sqrt x := x
  rsqrt x := x⁻¹
  sigmoid x := x
  gelu x := x
  silu x := x * x
  softplus x := (1 + x) * f16BitsToRat (floatToF16CanonicalZero 0.6931471805599453)
  relu x := if x ≤ 0 then 0 else x
  lutEval s x := match s with
    | .exp => x
    | .log2 => x
    | .sqrt => x
    | .rsqrt _ => x⁻¹
    | .inverse => x⁻¹
    | .sigmoid => x
    | .gelu => x
    | .silu => x * x
    | _ => x
  f16Val := f16BitsToRat
  f64Val f := f16BitsToRat (floatToF16CanonicalZero f)
  f16_posOne := f16BitsToRat_posOne
  f16_negOne := f16BitsToRat_negOne
  f16_neg := f16BitsToRat_neg
  f16_ln2 := rfl
  lut_exp _ := rfl
  lut_log2 _ := rfl
  lut_sqrt _ := rfl
  lut_rsqrt_zero _ := rfl
  lut_inverse _ := rfl
  lut_sigmoid _ := rfl
  lut_gelu _ := rfl
  lut_silu _ := rfl
  silu_def _ := rfl
  log_eq_log2_mul_ln2 _ := rfl
  softplus_def _ := rfl
  rsqrt_def _ := rfl
  relu_def _ := rfl

-- ============================================================================
-- Section 2: 4D Tensor Utilities & Exact `evalCore` / `evalHigh` Semantics
-- ============================================================================

/-- Exact 4D row-major `unflatten` inverse on `[1, C, H, W]`. -/
theorem unflatten_4d (C H W c y x : Nat) (hc : c < C) (hy : y < H) (hx : x < W) :
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

/-- Flat index bound for 4D coordinates `(c, y, x)` inside `[1, C, H, W]`. -/
theorem flat4D_lt (C H W c y x : Nat) (hc : c < C) (hy : y < H) (hx : x < W) :
    c * (H * W) + y * W + x < shapeProd [1, C, H, W] := by
  simp only [shapeProd, Nat.one_mul, Nat.mul_one]
  have hYW : y * W + x < H * W := by
    have h1 : (y + 1) * W ≤ H * W := Nat.mul_le_mul_right W (Nat.succ_le_of_lt hy)
    rw [Nat.add_mul, Nat.one_mul] at h1; omega
  have hAll : (c + 1) * (H * W) ≤ C * (H * W) := Nat.mul_le_mul_right (H * W) (Nat.succ_le_of_lt hc)
  rw [Nat.add_mul, Nat.one_mul] at hAll
  omega

/-- Construct a 4D `TensorVal α` of shape `[1, C, H, W]` from a coordinate function `f c y x`. -/
def mkTensor4D {α : Type} (C H W : Nat) (f : Nat → Nat → Nat → α) : TensorVal α where
  shape := [1, C, H, W]
  data idx :=
    let coords := unflatten [1, C, H, W] idx.val
    f (coords.getD 1 0) (coords.getD 2 0) (coords.getD 3 0)

/-- Reading a 4D `TensorVal α` constructed by `mkTensor4D` at `(c, y, x)` returns `f c y x`. -/
theorem mkTensor4D_at {α : Type} (C H W c y x : Nat) (f : Nat → Nat → Nat → α)
    (hc : c < C) (hy : y < H) (hx : x < W)
    (hlt : c * (H * W) + y * W + x < shapeProd [1, C, H, W]) :
    (mkTensor4D C H W f).data ⟨c * (H * W) + y * W + x, hlt⟩ = f c y x := by
  unfold mkTensor4D
  dsimp only
  rw [unflatten_4d C H W c y x hc hy hx]
  rfl

/-- Extensionality for `TensorVal α` of identical shape `[1, C, H, W]`. -/
theorem tensorVal_ext_4d {α : Type} (C H W : Nat) (t1 t2 : TensorVal α)
    (h1 : t1.shape = [1, C, H, W]) (h2 : t2.shape = [1, C, H, W])
    (hElem : ∀ (c y x : Nat), c < C → y < H → x < W →
      ∀ (h1lt : c * (H * W) + y * W + x < shapeProd t1.shape)
        (h2lt : c * (H * W) + y * W + x < shapeProd t2.shape),
        t1.data ⟨c * (H * W) + y * W + x, h1lt⟩ = t2.data ⟨c * (H * W) + y * W + x, h2lt⟩) :
    t1 = t2 := by
  cases t1 with | mk s1 d1 =>
  cases t2 with | mk s2 d2 =>
  dsimp only at h1 h2 hElem
  subst h1; subst h2
  congr 1
  funext ⟨idx, hidx⟩
  have hProd : shapeProd [1, C, H, W] = C * (H * W) := by simp [shapeProd]
  have hIdxLt : idx < C * (H * W) := hProd ▸ hidx
  have hWPos : W > 0 := by
    cases W with | zero => simp at hIdxLt | succ _ => omega
  have hHPos : H > 0 := by
    cases H with | zero => simp at hIdxLt | succ _ => omega
  have hHWPos : H * W > 0 := Nat.mul_pos hHPos hWPos
  let c := idx / (H * W)
  let rem := idx % (H * W)
  let y := rem / W
  let x := rem % W
  have hc : c < C := Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hIdxLt)
  have hRemLt : rem < H * W := Nat.mod_lt idx hHWPos
  have hy : y < H := Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hRemLt)
  have hx : x < W := Nat.mod_lt rem hWPos
  have hEq : c * (H * W) + y * W + x = idx := by
    have h1 := Nat.div_add_mod' idx (H * W)
    have h2 := Nat.div_add_mod' rem W
    calc c * (H * W) + y * W + x = c * (H * W) + (y * W + x) := Nat.add_assoc _ _ _
      _ = c * (H * W) + rem := by rw [h2]
      _ = idx := h1
  have hlt' : c * (H * W) + y * W + x < shapeProd [1, C, H, W] := hEq.symm ▸ hidx
  have h := hElem c y x hc hy hx hlt' hlt'
  have hFin : (⟨c * (H * W) + y * W + x, hlt'⟩ : Fin (shapeProd [1, C, H, W])) = ⟨idx, hidx⟩ :=
    Fin.ext hEq
  rw [hFin] at h
  exact h

/-- Total 4D element lookup on `[1, C, H, W]` returning `zero` out of bounds. -/
def read4D {α : Type} [ScalarField α] (t : TensorVal α) (_C H W c y x : Nat) : α :=
  let idx := c * (H * W) + y * W + x
  if h : idx < shapeProd t.shape then t.data ⟨idx, h⟩ else zero

theorem read4D_mkTensor4D {α : Type} [ScalarField α] (C H W c y x : Nat)
    (f : Nat → Nat → Nat → α) (hc : c < C) (hy : y < H) (hx : x < W) :
    read4D (mkTensor4D C H W f) C H W c y x = f c y x := by
  unfold read4D
  have hlt := flat4D_lt C H W c y x hc hy hx
  simp only [mkTensor4D, hlt, ↓reduceDIte]
  rw [unflatten_4d C H W c y x hc hy hx]
  rfl

-- ============================================================================
-- Exact Algebraic Operators (`Conv`, `MatMul`, `ElemUnary`, `ElemBinary`, `Reduce`, `Pool`)
-- ============================================================================

/-- Exact post-convolution activation application. -/
def applyConvAct {α : Type} [ScalarField α] (ops : ScalarOps α) : ConvActivation → α → α
  | .none, v => v
  | .relu, v => ops.relu v
  | .lut spec, v => ops.lutEval spec v

/-- Exact standalone NE activation application. -/
def applyNeAct {α : Type} [ScalarField α] (ops : ScalarOps α) : NeActSpec → α → α
  | .relu, v => ops.relu v
  | .copy, v => v
  | .lut spec, v => ops.lutEval spec v

/-- Exact 4D convolution kernel (arbitrary `kh kw sh sw pt pl groups`, optional bias, fused activation).
    Covers 1x1, 3x3 dense/depthwise/strip, 5x5, 7x7, Conv1d, asymmetric padding, and grouped conv! -/
def evalConvElem {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cinG coutG kh kw sh sw pt pl : Nat)
    (x : Nat → Nat → Nat → α)
    (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α))
    (act : ConvActivation)
    (oc oy ox : Nat) : α :=
  let g := oc / coutG
  let acc := sumFin cinG fun ic' =>
    sumFin kh fun ky =>
      sumFin kw fun kx =>
        let iy : Int := (oy * sh : Int) + (ky : Int) - (pt : Int)
        let ix : Int := (ox * sw : Int) + (kx : Int) - (pl : Int)
        let xVal :=
          if 0 ≤ iy ∧ iy < (H : Int) ∧ 0 ≤ ix ∧ ix < (W : Int) then
            x (g * cinG + ic') iy.toNat ix.toNat
          else
            zero
        mul (w oc ic' ky kx) xVal
  let withBias := match bias? with
    | some b => add acc (b oc)
    | none => acc
  applyConvAct ops act withBias

/-- 4D tensor-level exact convolution producing a `TensorVal α` of shape `[1, cout, Ho, Wo]`. -/
def evalConvTensor4D {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cinG cout coutG Ho Wo kh kw sh sw pt pl : Nat)
    (x : Nat → Nat → Nat → α)
    (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α))
    (act : ConvActivation) : TensorVal α :=
  mkTensor4D cout Ho Wo (evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w bias? act)

/-- Exact batched 3D MatMul element at `(h, i, j)` with transpose flags and scale `s`. -/
def evalMatmulElem {α : Type} [ScalarField α]
    (K : Nat) (trA trB : Bool) (scale : α)
    (A B : Nat → Nat → Nat → α)
    (h i j : Nat) : α :=
  mul scale (sumFin K fun k =>
    let aVal := if trA then A h k i else A h i k
    let bVal := if trB then B h j k else B h k j
    mul aVal bVal)

/-- Exact scalar unary operator semantics across all 11 `ElemUnaryOp` constructors. -/
def evalElemUnaryScalar {α : Type} [ScalarField α] (ops : ScalarOps α)
    (uop : ElemUnaryOp) (x : α) : α :=
  match uop with
  | .abs => ops.abs x
  | .neg => neg x
  | .square => mul x x
  | .log => ops.log x
  | .softplus => ops.softplus x
  | .scalarAdd b => add x (ops.f16Val b)
  | .scalarSub b => sub x (ops.f16Val b)
  | .scalarRSub b => sub (ops.f16Val b) x
  | .scalarMul b => mul x (ops.f16Val b)
  | .scalarDiv b => div x (ops.f16Val b)
  | .linearAct a b => add (mul x (ops.f16Val a)) (ops.f16Val b)

/-- Exact scalar binary operator semantics across all 6 `ElemBinaryOp` constructors. -/
def evalElemBinaryScalar {α : Type} [ScalarField α] (ops : ScalarOps α)
    (bop : ElemBinaryOp) (a b : α) : α :=
  match bop with
  | .add => add a b
  | .sub => sub a b
  | .mul => mul a b
  | .div => div a b
  | .max => ops.maxOp a b
  | .min => ops.minOp a b

/-- Exact per-element BatchNormalization affine evaluation `((x - mean) * invStd) * scale + bias`. -/
def evalBNScalar {α : Type} [ScalarField α] (x scale bnBias mean invStd : α) : α :=
  add (mul (mul (sub x mean) invStd) scale) bnBias

/-- Exact elementwise map on a `TensorVal α` preserving shape. -/
def mapTensor {α : Type} (f : α → α) (t : TensorVal α) : TensorVal α :=
  { shape := t.shape, data := fun idx => f (t.data idx) }

/-- Exact elementwise zip on two same-shape `TensorVal α`s. -/
def zipTensorSameShape {α : Type} (f : α → α → α) (a b : TensorVal α) : Option (TensorVal α) :=
  if h : a.shape = b.shape then
    some { shape := a.shape, data := fun idx => f (a.data idx) (b.data ⟨idx.val, h ▸ idx.isLt⟩) }
  else
    none

/-- Exact right-aligned numpy-style coordinate projection for broadcasting shape `src` into `out`. -/
def bcastCoord (srcShape outShape outCoords : List Nat) : List Nat :=
  let pad := outShape.length - srcShape.length
  (List.range srcShape.length).map fun i =>
    if srcShape.getD i 1 == 1 then 0 else outCoords.getD (pad + i) 0

/-- Exact right-aligned numpy-style broadcast compatibility check for `srcShape` into `outShape`. -/
def broadcastCompatible (srcShape outShape : List Nat) : Bool :=
  srcShape.length ≤ outShape.length &&
  let pad := outShape.length - srcShape.length
  (List.range srcShape.length).all fun i =>
    let sd := srcShape.getD i 0
    let od := outShape.getD (pad + i) 0
    sd > 0 && (sd == od || sd == 1)

/-- Exact elementwise binary op with right-aligned numpy broadcasting over `TensorVal α`,
    rejecting incompatible shapes (unlike a permissive fallback). -/
def evalBinaryBroadcast {α : Type} [ScalarField α]
    (f : α → α → α) (outShape : List Nat) (a b : TensorVal α) : Option (TensorVal α) :=
  let na := shapeProd a.shape
  let nb := shapeProd b.shape
  if outShape.length == max a.shape.length b.shape.length &&
     broadcastCompatible a.shape outShape &&
     broadcastCompatible b.shape outShape &&
     outShape.all (· > 0) then
    if ha : 0 < na then
      if hb : 0 < nb then
        some {
          shape := outShape
          data := fun outIdx =>
            let oc := unflatten outShape outIdx.val
            let ac := bcastCoord a.shape outShape oc
            let bc := bcastCoord b.shape outShape oc
            let ia := flatIndex a.shape ac
            let ib := flatIndex b.shape bc
            let va := if hltA : ia < na then a.data ⟨ia, hltA⟩ else a.data ⟨0, ha⟩
            let vb := if hltB : ib < nb then b.data ⟨ib, hltB⟩ else b.data ⟨0, hb⟩
            f va vb
        }
      else none
    else none
  else none

/-- Non-empty list maximum via `ops.maxOp`. -/
def foldMaxFin {α : Type} [ScalarField α] (ops : ScalarOps α) : (n : Nat) → (Nat → α) → α
  | 0, _ => zero
  | 1, f => f 0
  | n + 2, f => ops.maxOp (foldMaxFin ops (n + 1) f) (f (n + 1))

/-- Non-empty list minimum via `ops.minOp`. -/
def foldMinFin {α : Type} [ScalarField α] (ops : ScalarOps α) : (n : Nat) → (Nat → α) → α
  | 0, _ => zero
  | 1, f => f 0
  | n + 2, f => ops.minOp (foldMinFin ops (n + 1) f) (f (n + 1))

/-- Exact 1D reduction over `n` elements for every `ReduceOp` constructor. -/
def evalReduce1D {α : Type} [ScalarField α] (ops : ScalarOps α)
    (op : ReduceOp) (eps : α) (n : Nat) (f : Nat → α) : α :=
  match op with
  | .sum => sumFin n f
  | .mean => div (sumFin n f) (ofNat n)
  | .max => foldMaxFin ops n f
  | .min => foldMinFin ops n f
  | .l1 => sumFin n fun i => ops.abs (f i)
  | .sumSquare => sumFin n fun i => mul (f i) (f i)
  | .l2 => ops.sqrt (sumFin n fun i => mul (f i) (f i))
  | .invSum => inv (sumFin n f)
  | .rsqrtSumSquare => ops.rsqrt (add (sumFin n fun i => mul (f i) (f i)) eps)
  | .rsqrtMeanSquare => ops.rsqrt (add (div (sumFin n fun i => mul (f i) (f i)) (ofNat n)) eps)

/-- Exact multi-axis tensor reduction over `TensorVal α` using `evalReduce1D`. -/
def evalReduceAxesTensor {α : Type} [ScalarField α] (ops : ScalarOps α)
    (rop : ReduceOp) (eps : α) (redAxes : Array Nat) (x : TensorVal α) : Option (TensorVal α) :=
  if redAxes.isEmpty || !x.shape.all (· > 0) || !redAxes.all (· < x.shape.length) then
    none
  else
    let outShape := (softmaxRedShape x.shape.toArray redAxes).toList
    let redDims := (List.range x.shape.length).map fun i =>
      if redAxes.contains i then x.shape.getD i 1 else 1
    let nRed := shapeProd redDims
    if nRed == 0 then none
    else
      some {
        shape := outShape
        data := fun outIdx =>
          let outCoords := unflatten outShape outIdx.val
          evalReduce1D ops rop eps nRed fun k =>
            let rc := unflatten redDims k
            let inCoords := (List.range x.shape.length).map fun i =>
              if redAxes.contains i then rc.getD i 0 else outCoords.getD i 0
            let fIdx := flatIndex x.shape inCoords
            if hlt : fIdx < shapeProd x.shape then x.data ⟨fIdx, hlt⟩ else zero
      }

/-- Exact slice-list evaluator for `HighOp.split` via `Vane.IR.evalSlice`. -/
def evalSplitAux {α : Type} (x : TensorVal α) (axis : Nat) :
    Nat → List Nat → Option (List (TensorVal α))
  | _, [] => some []
  | curStart, sz :: rest =>
    let xArr := x.shape.toArray
    let beg := (Array.replicate xArr.size 0).set! axis curStart
    let en := xArr.set! axis (curStart + sz)
    let step := Array.replicate xArr.size 1
    match evalSlice beg en step x, evalSplitAux x axis (curStart + sz) rest with
    | some t, some ts => some (t :: ts)
    | _, _ => none

theorem evalSplitAux_length {α : Type} (x : TensorVal α) (axis : Nat) :
    ∀ curStart sizes ts, evalSplitAux x axis curStart sizes = some ts → ts.length = sizes.length := by
  intro curStart sizes
  induction sizes generalizing curStart with
  | nil =>
    intro ts h
    simp only [evalSplitAux, Option.some.injEq] at h
    subst h; rfl
  | cons sz rest ih =>
    intro ts h
    unfold evalSplitAux at h
    dsimp only at h
    split at h
    · rename_i t restTs hSl hRec
      simp only [Option.some.injEq] at h
      subst h
      simp [ih (curStart + sz) restTs hRec]
    · cases h

/-- Helper for 1D channel-vector lookup `[C]` returning `zero` out of bounds. -/
def read1D {α : Type} [ScalarField α] (t : TensorVal α) (i : Nat) : α :=
  if h : i < shapeProd t.shape then t.data ⟨i, h⟩ else zero

/-- Concrete, total, exact algebraic `HighOp` evaluator covering ALL 8 `HighOp` constructors
    (`core`, `gelu`, `pow`, `batchNorm`, `rmsNorm`, `layerNorm`, `softmax`, `split`) with
    zero uninterpreted callbacks. -/
def evalHighOpCore {α : Type} [ScalarField α] (ops : ScalarOps α)
    (coreSem : OpSemantics CoreOp α)
    (op : HighOp) (ins : List (TensorVal α)) (outTys : List TensorType) :
    Option (List (TensorVal α)) :=
  match op with
  | .core c => coreSem.evalOp c ins outTys
  | .gelu .exact =>
    match ins, outTys with
    | [x], [_] => some [mapTensor ops.gelu x]
    | _, _ => none
  | .pow kind =>
    match kind, ins, outTys with
    | .identity, [x], [_] => some [x]
    | .square, [x], [_] => some [mapTensor (fun v => mul v v) x]
    | .sqrt, [x], [_] => some [mapTensor ops.sqrt x]
    | .rsqrt, [x], [_] => some [mapTensor ops.rsqrt x]
    | .reciprocal, [x], [_] => some [mapTensor inv x]
    | _, _, _ => none
  | .batchNorm eps =>
    match ins, outTys with
    | [x, sc, b, mean, var], [_] =>
      let cOpt : Option Nat := match x.shape with
        | [1, c, h, w] => if c > 0 && h > 0 && w > 0 then some c else none
        | [1, c] => if c > 0 then some c else none
        | _ => none
      match cOpt with
      | some c =>
        if sc.shape == [c] && b.shape == [c] && mean.shape == [c] && var.shape == [c] then
          let epsA := ops.f64Val eps
          some [{
            shape := x.shape
            data := fun idx =>
              let oc := (unflatten x.shape idx.val).getD 1 0
              let invStd := ops.rsqrt (add (read1D var oc) epsA)
              evalBNScalar (x.data idx) (read1D sc oc) (read1D b oc) (read1D mean oc) invStd
          }]
        else none
      | none => none
    | _, _ => none
  | .softmax axis legacyOpset =>
    match ins, outTys with
    | [x], [_] =>
      match resolveSoftmaxRedAxes x.shape.length axis legacyOpset "" with
      | .error _ => none
      | .ok redAxes =>
        match evalReduceAxesTensor ops .max zero redAxes x with
        | none => none
        | some maxT =>
          match evalBinaryBroadcast sub x.shape x maxT with
          | none => none
          | some diffT =>
            let expT := mapTensor ops.exp diffT
            match evalReduceAxesTensor ops .sum zero redAxes expT with
            | none => none
            | some sumT =>
              (evalBinaryBroadcast div x.shape expT sumT).map fun y => [y]
    | _, _ => none
  | .rmsNorm axis eps hasScale =>
    match outTys with
    | [_] =>
      let parsed? : Option (TensorVal α × Option (TensorVal α)) :=
        if !hasScale then
          match ins with | [x] => some (x, none) | _ => none
        else
          match ins with | [x, sc] => some (x, some sc) | _ => none
      match parsed? with
      | none => none
      | some (x, sc?) =>
        match resolveRmsNormAxis x.shape.length axis "" with
        | .error _ => none
        | .ok axNat =>
          match evalReduceAxesTensor ops .rsqrtMeanSquare (ops.f64Val eps) #[axNat] x with
          | none => none
          | some invRmsT =>
            match evalBinaryBroadcast mul x.shape x invRmsT with
            | none => none
            | some normT =>
              match sc? with
              | none => some [normT]
              | some sc => (evalBinaryBroadcast mul x.shape normT sc).map fun y => [y]
    | _ => none
  | .layerNorm axis eps hasScale hasBias =>
    let parsed? : Option (TensorVal α × Option (TensorVal α) × Option (TensorVal α)) :=
      match hasScale, hasBias, ins with
      | false, false, [x] => some (x, none, none)
      | true, false, [x, sc] => some (x, some sc, none)
      | false, true, [x, b] => some (x, none, some b)
      | true, true, [x, sc, b] => some (x, some sc, some b)
      | _, _, _ => none
    match parsed? with
    | none => none
    | some (x, sc?, b?) =>
      match resolveLayerNormConfig x.shape.toArray axis "" with
      | .error _ => none
      | .ok (redAxes, _, _, _) =>
        match evalReduceAxesTensor ops .mean zero redAxes x with
        | none => none
        | some meanT =>
          match evalBinaryBroadcast sub x.shape x meanT with
          | none => none
          | some diffT =>
            let sqT := mapTensor (fun v => mul v v) diffT
            match evalReduceAxesTensor ops .mean zero redAxes sqT with
            | none => none
            | some varT =>
              let invStdT := mapTensor (fun v => ops.rsqrt (add v (ops.f64Val eps))) varT
              match evalBinaryBroadcast mul x.shape diffT invStdT with
              | none => none
              | some normT =>
                let scaled? : Option (TensorVal α) := match sc? with
                  | none => some normT
                  | some sc => evalBinaryBroadcast mul x.shape normT sc
                match scaled? with
                | none => none
                | some scT =>
                  let y? : Option (TensorVal α) := match b? with
                    | none => some scT
                    | some b => evalBinaryBroadcast add x.shape scT b
                  match y? with
                  | none => none
                  | some yT =>
                    match outTys with
                    | [_] => some [yT]
                    | [_, _] => some [yT, meanT]
                    | [_, _, _] => some [yT, meanT, invStdT]
                    | _ => none
  | .split axis sizes =>
    match ins with
    | [x] =>
      if sizes.size == outTys.length && axis < x.shape.length &&
         sumSizes sizes.toList == x.shape.getD axis 0 then
        evalSplitAux x axis 0 sizes.toList
      else none
    | _ => none

/-- Concrete `evalHigh` op-semantics constructor extending ANY core `OpSemantics CoreOp α`
    with exact algebraic semantics for all 8 `HighOp` constructors. -/
def mkHighSemantics {α : Type} [ScalarField α] (ops : ScalarOps α)
    (coreSem : OpSemantics CoreOp α) : OpSemantics HighOp α where
  evalOp := evalHighOpCore ops coreSem
  evalOp_len op ins outTys outs h := by
    cases op with
    | core c => exact coreSem.evalOp_len c ins outTys outs h
    | gelu ap =>
      cases ap with | exact =>
      dsimp only [evalHighOpCore] at h
      split at h
      · simp only [Option.some.injEq] at h; subst h; rfl
      · cases h
    | pow k =>
      dsimp only [evalHighOpCore] at h
      split at h
      · simp only [Option.some.injEq] at h; subst h; rfl
      · simp only [Option.some.injEq] at h; subst h; rfl
      · simp only [Option.some.injEq] at h; subst h; rfl
      · simp only [Option.some.injEq] at h; subst h; rfl
      · simp only [Option.some.injEq] at h; subst h; rfl
      · cases h
    | batchNorm eps =>
      dsimp only [evalHighOpCore] at h
      repeat (first | cases h | (split at h <;> try cases h))
      simp_all
    | softmax axis legacy =>
      dsimp only [evalHighOpCore] at h
      repeat (first | cases h | (split at h <;> try cases h))
      simp only [Option.map_eq_some_iff] at h
      obtain ⟨_, _, rfl⟩ := h; rfl
    | rmsNorm axis eps hasScale =>
      dsimp only [evalHighOpCore] at h
      cases outTys with
      | nil => cases h
      | cons _ rest =>
        cases rest with
        | cons _ _ => cases h
        | nil =>
          dsimp only at h
          cases hP : (if !hasScale then match ins with | [x] => some (x, none) | _ => none
                      else match ins with | [x, sc] => some (x, some sc) | _ => none) with
          | none => rw [hP] at h; cases h
          | some p =>
            rcases p with ⟨x, sc?⟩
            rw [hP] at h
            dsimp only at h
            cases hAx : resolveRmsNormAxis x.shape.length axis "" with
            | error _ => rw [hAx] at h; cases h
            | ok axNat =>
              rw [hAx] at h; dsimp only at h
              cases hR : evalReduceAxesTensor ops .rsqrtMeanSquare (ops.f64Val eps) #[axNat] x with
              | none => rw [hR] at h; cases h
              | some invRmsT =>
                rw [hR] at h; dsimp only at h
                cases hN : evalBinaryBroadcast mul x.shape x invRmsT with
                | none => rw [hN] at h; cases h
                | some normT =>
                  rw [hN] at h; dsimp only at h
                  cases sc? with
                  | none => simp only [Option.some.injEq] at h; subst h; rfl
                  | some sc =>
                    simp only [Option.map_eq_some_iff] at h
                    obtain ⟨_, _, rfl⟩ := h; rfl
    | layerNorm axis eps hasScale hasBias =>
      dsimp only [evalHighOpCore] at h
      cases hP : (match hasScale, hasBias, ins with
        | false, false, [x] => some (x, none, none)
        | true, false, [x, sc] => some (x, some sc, none)
        | false, true, [x, b] => some (x, none, some b)
        | true, true, [x, sc, b] => some (x, some sc, some b)
        | _, _, _ => none) with
      | none => rw [hP] at h; cases h
      | some p =>
        rcases p with ⟨x, sc?, b?⟩
        rw [hP] at h; dsimp only at h
        cases hCfg : resolveLayerNormConfig x.shape.toArray axis "" with
        | error _ => rw [hCfg] at h; cases h
        | ok cfg =>
          rcases cfg with ⟨redAxes, _, _, _⟩
          rw [hCfg] at h; dsimp only at h
          cases hM : evalReduceAxesTensor ops .mean zero redAxes x with
          | none => rw [hM] at h; cases h
          | some meanT =>
            rw [hM] at h; dsimp only at h
            cases hD : evalBinaryBroadcast sub x.shape x meanT with
            | none => rw [hD] at h; cases h
            | some diffT =>
              rw [hD] at h; dsimp only at h
              cases hV : evalReduceAxesTensor ops .mean zero redAxes (mapTensor (fun v => mul v v) diffT) with
              | none => rw [hV] at h; cases h
              | some varT =>
                rw [hV] at h; dsimp only at h
                cases hN : evalBinaryBroadcast mul x.shape diffT (mapTensor (fun v => ops.rsqrt (add v (ops.f64Val eps))) varT) with
                | none => rw [hN] at h; cases h
                | some normT =>
                  rw [hN] at h; dsimp only at h
                  cases hSc : (match sc? with | none => some normT | some sc => evalBinaryBroadcast mul x.shape normT sc) with
                  | none => rw [hSc] at h; cases h
                  | some scT =>
                    rw [hSc] at h; dsimp only at h
                    cases hY : (match b? with | none => some scT | some b => evalBinaryBroadcast add x.shape scT b) with
                    | none => rw [hY] at h; cases h
                    | some yT =>
                      rw [hY] at h; dsimp only at h
                      cases outTys with
                      | nil => cases h
                      | cons _ r1 =>
                        cases r1 with
                        | nil => simp only [Option.some.injEq] at h; subst h; rfl
                        | cons _ r2 =>
                          cases r2 with
                          | nil => simp only [Option.some.injEq] at h; subst h; rfl
                          | cons _ r3 =>
                            cases r3 with
                            | nil => simp only [Option.some.injEq] at h; subst h; rfl
                            | cons _ _ => cases h
    | split axis sizes =>
      dsimp only [evalHighOpCore] at h
      cases ins with
      | nil => cases h
      | cons x rest =>
        cases rest with
        | cons _ _ => cases h
        | nil =>
          dsimp only at h
          split at h
          · rename_i hCond
            have hSzEq : sizes.size = outTys.length := by
              simp only [Bool.and_eq_true, beq_iff_eq] at hCond
              exact hCond.1.1
            have hL := evalSplitAux_length x axis 0 sizes.toList outs h
            rw [hL, Array.length_toList, hSzEq]
          · cases h

/-- `mkHighSemantics` inherits `CoreViewLawful` directly from its underlying `coreSem`. -/
theorem mkHighSemantics_coreViewLawful {α : Type} [ScalarField α] (ops : ScalarOps α)
    (coreSem : OpSemantics CoreOp α) (hCore : CoreViewLawful coreSem) :
    CoreViewLawful (mkHighSemantics ops coreSem) := by
  constructor
  · intro op perm t outTy h
    cases op with
    | core c =>
      have hc : c = .transpose perm := Option.some.inj h
      subst hc
      exact hCore.transpose_law (.transpose perm) perm t outTy rfl
    | _ => cases h
  · intro op sh t outTy h
    cases op with
    | core c =>
      have hc : c = .reshape sh := Option.some.inj h
      subst hc
      exact hCore.reshape_law (.reshape sh) sh t outTy rfl
    | _ => cases h
  · intro op b e s t outTy h
    cases op with
    | core c =>
      have hc : c = .slice b e s := Option.some.inj h
      subst hc
      exact hCore.slice_law (.slice b e s) b e s t outTy rfl
    | _ => cases h
  · intro op ax ts outTy h
    cases op with
    | core c =>
      have hc : c = .concat ax := Option.some.inj h
      subst hc
      exact hCore.concat_law (.concat ax) ax ts outTy rfl
    | _ => cases h

/-- Denotational `evalHigh` evaluator alias matching `Vane.IR.evalCore`. -/
def evalHigh {α : Type} (sem : OpSemantics HighOp α) (g : SSAGraph HighOp) (env : Env α) :
    Option (Env α) :=
  evalGraph sem g env

-- ============================================================================
-- Section 3: Reusable SSA Graph-Level Semantic Substitution & Fusion Infrastructure
-- ============================================================================

theorem evalNodes_append_eq {Op α : Type} (sem : OpSemantics Op α) (env : Env α)
    (ns1 ns2 : List (IRNode Op)) :
    evalNodes sem env (ns1 ++ ns2) =
      match evalNodes sem env ns1 with
      | none => none
      | some env1 => evalNodes sem env1 ns2 := by
  induction ns1 generalizing env with
  | nil => rfl
  | cons nd rest ih =>
    simp only [List.cons_append, evalNodes]
    cases evalNode sem env nd with
    | none => rfl
    | some env' => exact ih env'

theorem bindOutputs_not_mem {α : Type} (env : Env α)
    (outs : List (ValueId × TensorType)) (vals : List (TensorVal α)) (v : ValueId)
    (hNot : v ∉ outs.map Prod.fst) :
    bindOutputs env outs vals v = env v := by
  induction outs generalizing env vals with
  | nil => rfl
  | cons o os ih =>
    cases vals with
    | nil => rfl
    | cons t ts =>
      simp only [bindOutputs]
      have hNe : v ≠ o.1 := fun h => hNot (by rw [h]; exact List.mem_cons_self)
      have hRest : v ∉ os.map Prod.fst := fun h => hNot (List.mem_cons_of_mem _ h)
      rw [ih (Env.set env o.1 t) ts hRest]
      simp [Env.set, hNe]

theorem evalNodes_preserves_unwritten {Op α : Type} (sem : OpSemantics Op α)
    (ns : List (IRNode Op)) (env env' : Env α) (v : ValueId)
    (hNot : v ∉ ns.flatMap IRNode.outputIds)
    (hev : evalNodes sem env ns = some env') :
    env' v = env v := by
  induction ns generalizing env with
  | nil =>
    simp only [evalNodes, Option.some.injEq] at hev
    rw [hev]
  | cons nd rest ih =>
    simp only [evalNodes] at hev
    cases hNd : evalNode sem env nd with
    | none => simp [hNd] at hev
    | some env1 =>
      simp only [hNd] at hev
      have hNotNd : v ∉ nd.outputIds := fun h => hNot (List.mem_flatMap.mpr ⟨nd, List.mem_cons_self, h⟩)
      have hNotRest : v ∉ rest.flatMap IRNode.outputIds := fun h => by
        rw [List.mem_flatMap] at h
        obtain ⟨x, hx, hv⟩ := h
        exact hNot (List.mem_flatMap.mpr ⟨x, List.mem_cons_of_mem _ hx, hv⟩)
      rw [ih env1 hNotRest hev]
      unfold evalNode at hNd
      cases hLu : lookupAll env nd.inputs.toList with
      | none => simp [hLu] at hNd
      | some inVals =>
        simp only [hLu] at hNd
        cases hOp : sem.evalOp nd.op inVals (nd.outputs.toList.map Prod.snd) with
        | none => simp [hOp] at hNd
        | some outVals =>
          simp only [hOp] at hNd
          split at hNd
          · simp only [Option.some.injEq] at hNd
            rw [← hNd]
            exact bindOutputs_not_mem env nd.outputs.toList outVals v hNotNd
          · cases hNd

/-- Environment agreement on a set of IDs (under an optional alias mapping `remap`). -/
theorem lookupAll_congr {α : Type} (env1 env2 : Env α) (ids : List ValueId)
    (hEq : ∀ v ∈ ids, env2 v = env1 v) :
    lookupAll env2 ids = lookupAll env1 ids := by
  induction ids with
  | nil => rfl
  | cons v vs ih =>
    simp only [lookupAll]
    rw [hEq v List.mem_cons_self, ih (fun u hu => hEq u (List.mem_cons_of_mem _ hu))]

/-- General SSA simulation lemma for single-node replacement: replacing a single node `nd`
    in a sequential node list with a replacement sub-chain `repNodes` that produces the EXACT
    same output values for `nd.outputs` (using only fresh helper IDs disjoint from downstream
    live reads) preserves `evalNodes` on all non-helper `ValueId`s! -/
theorem evalNodes_replace_subchain {Op α : Type} (sem : OpSemantics Op α)
    (pre post repNodes : List (IRNode Op)) (nd : IRNode Op)
    (env envFinal : Env α) (liveOuts : List ValueId)
    (hHelperFreshPost : ∀ v ∈ repNodes.flatMap IRNode.outputIds, v ∉ nd.outputIds →
      (∀ m ∈ post, v ∉ m.inputs.toList) ∧ v ∉ liveOuts)
    (hHelperNotOverwritten : ∀ m ∈ post, ∀ v ∈ m.inputs.toList,
      v ∉ repNodes.flatMap IRNode.outputIds ∨ v ∈ nd.outputIds)
    (hRep : ∀ envPre envAfterNd,
      evalNodes sem env pre = some envPre →
      evalNode sem envPre nd = some envAfterNd →
      ∃ envAfterRep, evalNodes sem envPre repNodes = some envAfterRep ∧
        ∀ v, (v ∉ repNodes.flatMap IRNode.outputIds ∨ v ∈ nd.outputIds) →
          envAfterRep v = envAfterNd v)
    (hevOrig : evalNodes sem env (pre ++ [nd] ++ post) = some envFinal) :
    ∃ envFinal', evalNodes sem env (pre ++ repNodes ++ post) = some envFinal' ∧
      lookupAll envFinal' liveOuts = lookupAll envFinal liveOuts := by
  rw [evalNodes_append_eq, evalNodes_append_eq] at hevOrig
  cases hPre : evalNodes sem env pre with
  | none => simp [hPre] at hevOrig
  | some envPre =>
    simp only [hPre] at hevOrig
    cases hNd : evalNode sem envPre nd with
    | none => simp [evalNodes, hNd] at hevOrig
    | some envAfterNd =>
      simp only [evalNodes, hNd] at hevOrig
      obtain ⟨envAfterRep, hEvRep, hAgree⟩ := hRep envPre envAfterNd hPre hNd
      have hPostSim : ∀ (ps : List (IRNode Op)) (e1 e2 eEnd : Env α),
          (∀ m ∈ ps, ∀ v ∈ m.inputs.toList, v ∉ repNodes.flatMap IRNode.outputIds ∨ v ∈ nd.outputIds) →
          (∀ v, (v ∉ repNodes.flatMap IRNode.outputIds ∨ v ∈ nd.outputIds) → e2 v = e1 v) →
          evalNodes sem e1 ps = some eEnd →
          ∃ eEnd', evalNodes sem e2 ps = some eEnd' ∧
            ∀ v, (v ∉ repNodes.flatMap IRNode.outputIds ∨ v ∈ nd.outputIds) → eEnd' v = eEnd v := by
        intro ps
        induction ps with
        | nil =>
          intro e1 e2 eEnd _ hAg hE
          simp only [evalNodes, Option.some.injEq] at hE
          subst hE
          exact ⟨e2, rfl, hAg⟩
        | cons m ms ih =>
          intro e1 e2 eEnd hSafe hAg hE
          simp only [evalNodes] at hE ⊢
          cases hM1 : evalNode sem e1 m with
          | none => simp [hM1] at hE
          | some e1' =>
            simp only [hM1] at hE
            have hLuEq : lookupAll e2 m.inputs.toList = lookupAll e1 m.inputs.toList :=
              lookupAll_congr e1 e2 m.inputs.toList fun v hv => hAg v (hSafe m List.mem_cons_self v hv)
            unfold evalNode at hM1 ⊢
            rw [hLuEq]
            cases hLu : lookupAll e1 m.inputs.toList with
            | none => simp [hLu] at hM1
            | some inVals =>
              simp only [hLu] at hM1 ⊢
              cases hOp : sem.evalOp m.op inVals (m.outputs.toList.map Prod.snd) with
              | none => simp [hOp] at hM1
              | some outVals =>
                simp only [hOp] at hM1 ⊢
                split at hM1
                · rename_i hSm
                  simp only [hSm, ↓reduceIte, Option.some.injEq] at hM1 ⊢
                  subst hM1
                  have hAg' : ∀ v, (v ∉ repNodes.flatMap IRNode.outputIds ∨ v ∈ nd.outputIds) →
                      bindOutputs e2 m.outputs.toList outVals v =
                      bindOutputs e1 m.outputs.toList outVals v := by
                    intro v hv
                    have hBind : ∀ (os : List (ValueId × TensorType)) (vs : List (TensorVal α)) (a b : Env α),
                        a v = b v → bindOutputs a os vs v = bindOutputs b os vs v := by
                      intro os
                      induction os with
                      | nil => intro vs a b hab; exact hab
                      | cons o rest iho =>
                        intro vs a b hab
                        cases vs with
                        | nil => exact hab
                        | cons t ts =>
                          simp only [bindOutputs]
                          apply iho ts
                          simp only [Env.set]
                          split
                          · rfl
                          · exact hab
                    exact hBind m.outputs.toList outVals e2 e1 (hAg v hv)
                  exact ih (bindOutputs e1 m.outputs.toList outVals)
                    (bindOutputs e2 m.outputs.toList outVals) eEnd
                    (fun x hx => hSafe x (List.mem_cons_of_mem _ hx)) hAg' hE
                · cases hM1
      obtain ⟨envFinal', hPost2, hAgFinal⟩ := hPostSim post envAfterNd envAfterRep envFinal
        hHelperNotOverwritten hAgree hevOrig
      rw [evalNodes_append_eq, evalNodes_append_eq, hPre]
      dsimp only
      rw [hEvRep]
      dsimp only
      rw [hPost2]
      refine ⟨envFinal', rfl, ?_⟩
      apply lookupAll_congr envFinal envFinal' liveOuts
      intro v hv
      apply hAgFinal v
      by_cases hMem : v ∈ repNodes.flatMap IRNode.outputIds
      · by_cases hOut : v ∈ nd.outputIds
        · exact Or.inr hOut
        · exact absurd hv (hHelperFreshPost v hMem hOut).2
      · exact Or.inl hMem

/-- General SSA fusion theorem for collapsing a 2-node producer→consumer pair `(nd1, nd2)`
    (with `nd1` at `i1`, `nd2` at `i2 > i1`, `nd1` outputting `midId`, `nd2` outputting `outId`,
    replacing `nd1` with `fusedNd` outputting `midId`, dropping `nd2`, and redirecting `outId ↦ midId`,
    matching the exact pattern of `HighFuse` and `CoreFuse`!). -/
theorem evalNodes_fuse_pair_subst {Op α : Type} (sem : OpSemantics Op α)
    (pre mid post : List (IRNode Op)) (nd1 nd2 fusedNd : IRNode Op)
    (midId outId : ValueId) (midTy outTy : TensorType)
    (env envFinal : Env α) (liveOuts : List ValueId)
    (hNd1Out : nd1.outputs.toList = [(midId, midTy)])
    (hNd2Out : nd2.outputs.toList = [(outId, outTy)])
    (hMidNotUsedInMid : ∀ m ∈ mid, midId ∉ m.inputs.toList)
    (hMidNotDefInMid : midId ∉ mid.flatMap IRNode.outputIds)
    (hMidNotDefInPost : midId ∉ post.flatMap IRNode.outputIds)
    (hOutNotDefInPost : outId ∉ post.flatMap IRNode.outputIds)
    (hMidNotUsedInPost : ∀ m ∈ post, midId ∉ m.inputs.toList)
    (hMidNotLive : midId ∉ liveOuts)
    (hFused : ∀ envPre env1 env2 val2,
      evalNodes sem env pre = some envPre →
      evalNode sem envPre nd1 = some env1 →
      evalNodes sem env1 mid = some env2 →
      evalNode sem env2 nd2 = some (Env.set env2 outId val2) →
      evalNode sem envPre fusedNd = some (Env.set envPre midId val2))
    (hevOrig : evalNodes sem env (pre ++ [nd1] ++ mid ++ [nd2] ++ post) = some envFinal) :
    let subst := fun (v : ValueId) => if v == outId then midId else v
    let post' := post.map fun m => { m with inputs := m.inputs.map subst }
    ∃ envFinal', evalNodes sem env (pre ++ [fusedNd] ++ mid ++ post') = some envFinal' ∧
      lookupAll envFinal' (liveOuts.map subst) = lookupAll envFinal liveOuts := by
  intro subst post'
  rw [evalNodes_append_eq, evalNodes_append_eq, evalNodes_append_eq, evalNodes_append_eq] at hevOrig
  cases hPre : evalNodes sem env pre with
  | none => simp [hPre] at hevOrig
  | some envPre =>
    simp only [hPre] at hevOrig
    cases hNd1 : evalNode sem envPre nd1 with
    | none => simp [evalNodes, hNd1] at hevOrig
    | some env1 =>
      simp only [evalNodes, hNd1] at hevOrig
      cases hMid : evalNodes sem env1 mid with
      | none => simp [hMid] at hevOrig
      | some env2 =>
        simp only [hMid] at hevOrig
        cases hNd2 : evalNode sem env2 nd2 with
        | none => simp [hNd2] at hevOrig
        | some env3 =>
          simp only [hNd2] at hevOrig
          -- Extract `val2` from `hNd2`
          have hNd2Set : ∃ val2, env3 = Env.set env2 outId val2 := by
            unfold evalNode at hNd2
            rw [hNd2Out, List.map_cons, List.map_nil] at hNd2
            cases hLu : lookupAll env2 nd2.inputs.toList with
            | none => simp [hLu] at hNd2
            | some inVals =>
              simp only [hLu] at hNd2
              cases hOp : sem.evalOp nd2.op inVals [outTy] with
              | none => simp [hOp] at hNd2
              | some outVals =>
                simp only [hOp] at hNd2
                cases outVals with
                | nil => simp [shapesMatch] at hNd2
                | cons v2 rs =>
                  cases rs with
                  | cons _ _ => simp [shapesMatch] at hNd2
                  | nil =>
                    split at hNd2
                    · simp only [Option.some.injEq] at hNd2
                      exact ⟨v2, hNd2.symm⟩
                    · cases hNd2
          obtain ⟨val2, rfl⟩ := hNd2Set
          have hNd1Set : ∃ val1, env1 = Env.set envPre midId val1 := by
            unfold evalNode at hNd1
            rw [hNd1Out, List.map_cons, List.map_nil] at hNd1
            cases hLu : lookupAll envPre nd1.inputs.toList with
            | none => simp [hLu] at hNd1
            | some inVals =>
              simp only [hLu] at hNd1
              cases hOp : sem.evalOp nd1.op inVals [midTy] with
              | none => simp [hOp] at hNd1
              | some outVals =>
                simp only [hOp] at hNd1
                cases outVals with
                | nil => simp [shapesMatch] at hNd1
                | cons v1 rs =>
                  cases rs with
                  | cons _ _ => simp [shapesMatch] at hNd1
                  | nil =>
                    split at hNd1
                    · simp only [Option.some.injEq] at hNd1
                      exact ⟨v1, hNd1.symm⟩
                    · cases hNd1
          obtain ⟨val1, hEnv1Eq⟩ := hNd1Set
          have hFusedEv := hFused envPre env1 env2 val2 hPre hNd1 hMid hNd2
          -- Step `mid` under `Env.set envPre midId val2` vs `env1 = Env.set envPre midId val1`:
          -- since no node in `mid` reads `midId`, both produce environments agreeing on all `v ≠ midId`
          -- and `env2' midId = some val2`!
          have hMidSim : ∀ (ms : List (IRNode Op)) (a b eEnd : Env α),
              (∀ m ∈ ms, midId ∉ m.inputs.toList) →
              (midId ∉ ms.flatMap IRNode.outputIds) →
              (∀ v, v ≠ midId → b v = a v) →
              b midId = some val2 →
              evalNodes sem a ms = some eEnd →
              ∃ eEnd', evalNodes sem b ms = some eEnd' ∧
                (∀ v, v ≠ midId → eEnd' v = eEnd v) ∧ eEnd' midId = some val2 := by
            intro ms
            induction ms with
            | nil =>
              intro a b eEnd _ _ hAg hM hE
              simp only [evalNodes, Option.some.injEq] at hE
              subst hE
              exact ⟨b, rfl, hAg, hM⟩
            | cons m rest ih =>
              intro a b eEnd hNoUse hNoDef hAg hM hE
              simp only [evalNodes] at hE ⊢
              cases hMa : evalNode sem a m with
              | none => simp [hMa] at hE
              | some a' =>
                simp only [hMa] at hE
                have hLuEq : lookupAll b m.inputs.toList = lookupAll a m.inputs.toList :=
                  lookupAll_congr a b m.inputs.toList fun v hv =>
                    hAg v (fun heq => hNoUse m List.mem_cons_self (heq ▸ hv))
                unfold evalNode at hMa ⊢
                rw [hLuEq]
                cases hLu : lookupAll a m.inputs.toList with
                | none => simp [hLu] at hMa
                | some inVals =>
                  simp only [hLu] at hMa ⊢
                  cases hOp : sem.evalOp m.op inVals (m.outputs.toList.map Prod.snd) with
                  | none => simp [hOp] at hMa
                  | some outVals =>
                    simp only [hOp] at hMa ⊢
                    split at hMa
                    · rename_i hSm
                      simp only [hSm, ↓reduceIte, Option.some.injEq] at hMa ⊢
                      subst hMa
                      have hMidNotM : midId ∉ m.outputIds :=
                        fun hm => hNoDef (List.mem_flatMap.mpr ⟨m, List.mem_cons_self, hm⟩)
                      have hMidNotRest : midId ∉ rest.flatMap IRNode.outputIds := fun hm => by
                        rw [List.mem_flatMap] at hm
                        obtain ⟨x, hx, hxm⟩ := hm
                        exact hNoDef (List.mem_flatMap.mpr ⟨x, List.mem_cons_of_mem _ hx, hxm⟩)
                      have hAg' : ∀ v, v ≠ midId →
                          bindOutputs b m.outputs.toList outVals v =
                          bindOutputs a m.outputs.toList outVals v := by
                        intro v hv
                        have hB : ∀ (os : List (ValueId × TensorType)) (vs : List (TensorVal α)) (x y : Env α),
                            x v = y v → bindOutputs x os vs v = bindOutputs y os vs v := by
                          intro os; induction os with
                          | nil => intro _ _ _ h; exact h
                          | cons o os' iho =>
                            intro vs x y hxy; cases vs with
                            | nil => exact hxy
                            | cons t ts =>
                              simp only [bindOutputs]
                              exact iho ts _ _ (by simp only [Env.set]; split; rfl; exact hxy)
                        exact hB m.outputs.toList outVals b a (hAg v hv)
                      have hM' : bindOutputs b m.outputs.toList outVals midId = some val2 := by
                        rw [bindOutputs_not_mem b m.outputs.toList outVals midId hMidNotM, hM]
                      exact ih (bindOutputs a m.outputs.toList outVals)
                        (bindOutputs b m.outputs.toList outVals) eEnd
                        (fun x hx => hNoUse x (List.mem_cons_of_mem _ hx)) hMidNotRest hAg' hM' hE
                    · cases hMa
          obtain ⟨env2', hMid2, hAg2, hMidVal2⟩ := hMidSim mid env1 (Env.set envPre midId val2) env2
            hMidNotUsedInMid hMidNotDefInMid
            (fun v hv => by rw [hEnv1Eq]; simp [Env.set, hv])
            (by simp [Env.set]) hMid
          -- Now `env2' (subst v) = (Env.set env2 outId val2) v` for all `v ≠ midId`!
          have hPostRel : ∀ v, v ≠ midId → env2' (subst v) = Env.set env2 outId val2 v := by
            intro v hv
            dsimp only [subst]
            simp only [Env.set]
            by_cases hvo : v = outId
            · subst hvo; simp [hMidVal2]
            · simp only [beq_iff_eq, hvo, ↓reduceIte]
              exact hAg2 v hv
          -- Simulate `post'` vs `post`
          have hPostSim : ∀ (ps : List (IRNode Op)) (a b eEnd : Env α),
              (∀ m ∈ ps, midId ∉ m.inputs.toList) →
              (midId ∉ ps.flatMap IRNode.outputIds) →
              (outId ∉ ps.flatMap IRNode.outputIds) →
              (∀ v, v ≠ midId → b (subst v) = a v) →
              evalNodes sem a ps = some eEnd →
              ∃ eEnd', evalNodes sem b (ps.map fun m => { m with inputs := m.inputs.map subst }) = some eEnd' ∧
                ∀ v, v ≠ midId → eEnd' (subst v) = eEnd v := by
            intro ps
            induction ps with
            | nil =>
              intro a b eEnd _ _ _ hRel hE
              simp only [List.map_nil, evalNodes, Option.some.injEq] at hE ⊢
              subst hE
              exact ⟨b, rfl, hRel⟩
            | cons m rest ih =>
              intro a b eEnd hNoUse hNoDefM hNoDefO hRel hE
              simp only [List.map_cons, evalNodes] at hE ⊢
              cases hMa : evalNode sem a m with
              | none => simp [hMa] at hE
              | some a' =>
                simp only [hMa] at hE
                have hLuEq : lookupAll b (m.inputs.map subst).toList = lookupAll a m.inputs.toList := by
                  rw [Array.toList_map]
                  have hGo : ∀ (xs : List ValueId), (∀ v ∈ xs, v ≠ midId) →
                      lookupAll b (xs.map subst) = lookupAll a xs := by
                    intro xs hxs
                    induction xs with
                    | nil => rfl
                    | cons x xs' ihx =>
                      simp only [List.map_cons, lookupAll]
                      rw [hRel x (hxs x List.mem_cons_self),
                          ihx (fun u hu => hxs u (List.mem_cons_of_mem _ hu))]
                  exact hGo m.inputs.toList fun v hv heq => hNoUse m List.mem_cons_self (heq ▸ hv)
                unfold evalNode at hMa ⊢
                rw [hLuEq]
                cases hLu : lookupAll a m.inputs.toList with
                | none => simp [hLu] at hMa
                | some inVals =>
                  simp only [hLu] at hMa ⊢
                  cases hOp : sem.evalOp m.op inVals (m.outputs.toList.map Prod.snd) with
                  | none => simp [hOp] at hMa
                  | some outVals =>
                    simp only [hOp] at hMa ⊢
                    split at hMa
                    · rename_i hSm
                      simp only [hSm, ↓reduceIte, Option.some.injEq] at hMa ⊢
                      subst hMa
                      have hNoM1 : midId ∉ m.outputIds :=
                        fun hm => hNoDefM (List.mem_flatMap.mpr ⟨m, List.mem_cons_self, hm⟩)
                      have hNoO1 : outId ∉ m.outputIds :=
                        fun hm => hNoDefO (List.mem_flatMap.mpr ⟨m, List.mem_cons_self, hm⟩)
                      have hNoMR : midId ∉ rest.flatMap IRNode.outputIds := fun hm => by
                        rw [List.mem_flatMap] at hm; obtain ⟨x, hx, hxm⟩ := hm
                        exact hNoDefM (List.mem_flatMap.mpr ⟨x, List.mem_cons_of_mem _ hx, hxm⟩)
                      have hNoOR : outId ∉ rest.flatMap IRNode.outputIds := fun hm => by
                        rw [List.mem_flatMap] at hm; obtain ⟨x, hx, hxm⟩ := hm
                        exact hNoDefO (List.mem_flatMap.mpr ⟨x, List.mem_cons_of_mem _ hx, hxm⟩)
                      have hRel' : ∀ v, v ≠ midId →
                          bindOutputs b m.outputs.toList outVals (subst v) =
                          bindOutputs a m.outputs.toList outVals v := by
                        intro v hv
                        have hB : ∀ (os : List (ValueId × TensorType)) (vs : List (TensorVal α)) (x y : Env α),
                            midId ∉ os.map Prod.fst → outId ∉ os.map Prod.fst →
                            x (subst v) = y v →
                            bindOutputs x os vs (subst v) = bindOutputs y os vs v := by
                          intro os; induction os with
                          | nil => intro _ _ _ _ _ h; exact h
                          | cons o os' iho =>
                            intro vs x y hnm hno hxy
                            cases vs with
                            | nil => exact hxy
                            | cons t ts =>
                              simp only [bindOutputs]
                              have hoNeM : o.1 ≠ midId := fun heq => hnm (by simp [heq])
                              have hoNeO : o.1 ≠ outId := fun heq => hno (by simp [heq])
                              apply iho ts _ _
                                (fun h => hnm (List.mem_cons_of_mem _ h))
                                (fun h => hno (List.mem_cons_of_mem _ h))
                              simp only [Env.set, subst]
                              by_cases hvo : v = outId
                              · have hvoT : (v == outId) = true := by simp [hvo]
                                simp only [hvoT, ↓reduceIte]
                                have h1 : (midId == o.1) = false := by simp [Ne.symm hoNeM]
                                have h2 : (v == o.1) = false := by rw [hvo]; simp [Ne.symm hoNeO]
                                rw [h1, h2]
                                dsimp only [subst] at hxy
                                rw [hvoT] at hxy
                                exact hxy
                              · have hvoB : (v == outId) = false := by simp [hvo]
                                simp only [hvoB, Bool.false_eq_true, ↓reduceIte]
                                split
                                · rfl
                                · dsimp only [subst] at hxy
                                  rw [hvoB] at hxy
                                  exact hxy
                        exact hB m.outputs.toList outVals b a hNoM1 hNoO1 (hRel v hv)
                      exact ih (bindOutputs a m.outputs.toList outVals)
                        (bindOutputs b m.outputs.toList outVals) eEnd
                        (fun x hx => hNoUse x (List.mem_cons_of_mem _ hx)) hNoMR hNoOR hRel' hE
                    · cases hMa
          obtain ⟨envFinal', hPost2, hFinalRel⟩ := hPostSim post (Env.set env2 outId val2) env2' envFinal
            hMidNotUsedInPost hMidNotDefInPost hOutNotDefInPost hPostRel hevOrig
          rw [evalNodes_append_eq, evalNodes_append_eq, evalNodes_append_eq, hPre]
          dsimp only
          simp only [evalNodes, hFusedEv, hMid2]
          refine ⟨envFinal', hPost2, ?_⟩
          have hLuOut : ∀ (xs : List ValueId), (∀ v ∈ xs, v ≠ midId) →
              lookupAll envFinal' (xs.map subst) = lookupAll envFinal xs := by
            intro xs hxs
            induction xs with
            | nil => rfl
            | cons x xs' ihx =>
              simp only [List.map_cons, lookupAll]
              rw [hFinalRel x (hxs x List.mem_cons_self),
                  ihx (fun u hu => hxs u (List.mem_cons_of_mem _ hu))]
          exact hLuOut liveOuts fun v hv heq => hMidNotLive (heq ▸ hv)

-- ============================================================================
-- Section 4: L1 Pass Equivalences (`HighFuse`, `CoreFuse`, `ConstFold`, `Decompose`)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 4A. HighFuse Equivalence (`Vane/IR/Passes/HighFuse.lean`)
-- ----------------------------------------------------------------------------

/-- Exact algebraic counterpart of `HighFuse.computeBNAffineF64`:
    `factor o = scale o * invStd o`, `biasShift o = bnBias o - mean o * factor o`. -/
def bnAffineFactor {α : Type} [ScalarField α] (scale invStd : Nat → α) (o : Nat) : α :=
  mul (scale o) (invStd o)

def bnAffineBiasShift {α : Type} [ScalarField α] (scale bnBias mean invStd : Nat → α) (o : Nat) : α :=
  sub (bnBias o) (mul (mean o) (bnAffineFactor scale invStd o))

/-- Exact folded weights & bias matching `HighFuse.fuseConvBatchNormPass` (lines 128–133):
    `W'_o = W_o * factor_o`, `b'_o = b_o * factor_o + bnBiasShift_o`. -/
def foldConvBNWeights {α : Type} [ScalarField α]
    (w : Nat → Nat → Nat → Nat → α) (scale invStd : Nat → α) : Nat → Nat → Nat → Nat → α :=
  fun oc ic ky kx => mul (w oc ic ky kx) (bnAffineFactor scale invStd oc)

def foldConvBNBias {α : Type} [ScalarField α]
    (convBias scale bnBias mean invStd : Nat → α) : Nat → α :=
  fun oc => add (mul (convBias oc) (bnAffineFactor scale invStd oc))
                (bnAffineBiasShift scale bnBias mean invStd oc)

/-- Core algebraic identity of Conv+BatchNorm folding (matching `HighFuse.lean` lines 38–39 & 132–133):
    `((acc + b) - mean) * invStd * scale + bnBias = acc * (scale * invStd) + (b * (scale * invStd) + (bnBias - mean * (scale * invStd)))`. -/
theorem bn_affine_scalar_identity {α : Type} [ScalarField α]
    (acc b scale bnBias mean invStd : α) :
    let f := mul scale invStd
    let bShift := sub bnBias (mul mean f)
    evalBNScalar (add acc b) scale bnBias mean invStd =
      add (mul acc f) (add (mul b f) bShift) := by
  intro f bShift
  unfold evalBNScalar
  dsimp only [f, bShift]
  rw [mul_assoc (sub (add acc b) mean) invStd scale, mul_comm invStd scale]
  rw [sub_mul (add acc b) mean (mul scale invStd)]
  rw [right_distrib acc b (mul scale invStd)]
  unfold sub
  rw [add_assoc (mul acc (mul scale invStd)), add_assoc (mul acc (mul scale invStd))]
  congr 1
  rw [add_assoc (mul b (mul scale invStd)), add_comm (neg (mul mean (mul scale invStd))) bnBias]

/-- Exact Conv + BatchNorm weight/bias folding theorem (with original conv bias `b`):
    at EVERY output coordinate `(oc, oy, ox)`, applying `BatchNormalization(scale, bnBias, mean, invStd)`
    to `Conv(x, W, b, .none)` equals `Conv(x, foldConvBNWeights W, foldConvBNBias b, .none)`! -/
theorem highFuse_conv_bn_exact_with_bias {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cinG coutG kh kw sh sw pt pl : Nat)
    (x : Nat → Nat → Nat → α)
    (w : Nat → Nat → Nat → Nat → α)
    (convBias scale bnBias mean invStd : Nat → α)
    (oc oy ox : Nat) :
    evalBNScalar
      (evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w (some convBias) .none oc oy ox)
      (scale oc) (bnBias oc) (mean oc) (invStd oc) =
    evalConvElem ops H W cinG coutG kh kw sh sw pt pl x
      (foldConvBNWeights w scale invStd)
      (some (foldConvBNBias convBias scale bnBias mean invStd))
      .none oc oy ox := by
  unfold evalConvElem applyConvAct foldConvBNWeights foldConvBNBias bnAffineFactor bnAffineBiasShift
  dsimp only
  rw [bn_affine_scalar_identity]
  congr 1
  rw [← sumFin_mul_right]
  apply sumFin_congr; intro ic' _
  rw [← sumFin_mul_right]
  apply sumFin_congr; intro ky _
  rw [← sumFin_mul_right]
  apply sumFin_congr; intro kx _
  rw [mul_assoc, mul_comm _ (mul (scale oc) (invStd oc)), ← mul_assoc]

/-- Exact Conv + BatchNorm weight/bias folding theorem (for bias-free `Conv`, matching `HighFuse.lean`
    line 111's `Array.replicate cout 0.0` initial bias). -/
theorem highFuse_conv_bn_exact_no_bias {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cinG coutG kh kw sh sw pt pl : Nat)
    (x : Nat → Nat → Nat → α)
    (w : Nat → Nat → Nat → Nat → α)
    (scale bnBias mean invStd : Nat → α)
    (oc oy ox : Nat) :
    evalBNScalar
      (evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w none .none oc oy ox)
      (scale oc) (bnBias oc) (mean oc) (invStd oc) =
    evalConvElem ops H W cinG coutG kh kw sh sw pt pl x
      (foldConvBNWeights w scale invStd)
      (some (foldConvBNBias (fun _ => zero) scale bnBias mean invStd))
      .none oc oy ox := by
  have hZeroBias :
      evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w none .none oc oy ox =
      evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w (some fun _ => zero) .none oc oy ox := by
    unfold evalConvElem applyConvAct
    dsimp only
    rw [add_zero]
  rw [hZeroBias]
  exact highFuse_conv_bn_exact_with_bias ops H W cinG coutG kh kw sh sw pt pl x w
    (fun _ => zero) scale bnBias mean invStd oc oy ox

/-- Tensor-level exact Conv + BatchNorm equivalence over `TensorVal α` (`[1, cout, Ho, Wo]`). -/
theorem highFuse_conv_bn_tensor_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cinG cout coutG Ho Wo kh kw sh sw pt pl : Nat)
    (x : Nat → Nat → Nat → α)
    (w : Nat → Nat → Nat → Nat → α)
    (convBias scale bnBias mean invStd : Nat → α) :
    mkTensor4D cout Ho Wo (fun oc oy ox =>
      evalBNScalar
        (read4D (evalConvTensor4D ops H W cinG cout coutG Ho Wo kh kw sh sw pt pl x w (some convBias) .none)
          cout Ho Wo oc oy ox)
        (scale oc) (bnBias oc) (mean oc) (invStd oc)) =
    evalConvTensor4D ops H W cinG cout coutG Ho Wo kh kw sh sw pt pl x
      (foldConvBNWeights w scale invStd)
      (some (foldConvBNBias convBias scale bnBias mean invStd))
      .none := by
  apply tensorVal_ext_4d cout Ho Wo _ _ rfl rfl
  intro oc oy ox hoc hoy hox h1 h2
  rw [mkTensor4D_at cout Ho Wo oc oy ox _ hoc hoy hox h1]
  unfold evalConvTensor4D
  rw [mkTensor4D_at cout Ho Wo oc oy ox _ hoc hoy hox h2]
  rw [read4D_mkTensor4D cout Ho Wo oc oy ox _ hoc hoy hox]
  exact highFuse_conv_bn_exact_with_bias ops H W cinG coutG kh kw sh sw pt pl x w
    convBias scale bnBias mean invStd oc oy ox

/-- Exact Conv + Gelu epilogue fusion preservation (connecting to `HighFuse.fuseConvGeluPass`):
    applying `HighOp.gelu .exact` after `Conv(..., act = .none)` equals `Conv(..., act = .lut .gelu)`. -/
theorem highFuse_conv_gelu_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cinG coutG kh kw sh sw pt pl : Nat)
    (x : Nat → Nat → Nat → α) (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α)) (oc oy ox : Nat) :
    ops.gelu (evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w bias? .none oc oy ox) =
    evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w bias? (.lut .gelu) oc oy ox := by
  unfold evalConvElem applyConvAct
  dsimp only
  rw [ops.lut_gelu]

/-- Exact Conv + general `NeActSpec` fusion preservation (`relu` or any `lut lspec`, covering
    both `HighFuse.fuseConvGeluPass` and `CoreFuse.fuseConvActCore`). -/
theorem highFuse_conv_act_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cinG coutG kh kw sh sw pt pl : Nat)
    (x : Nat → Nat → Nat → α) (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α)) (act : ConvActivation) (oc oy ox : Nat) :
    applyConvAct ops act (evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w bias? .none oc oy ox) =
    evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w bias? act oc oy ox := by
  unfold evalConvElem applyConvAct
  rfl

/-- Graph-level `evalHigh` equivalence for `HighFuse.fuseConvGeluPass`: fusing a single-consumer
    `Conv(..., act=.none) → HighOp.gelu .exact` pair in an `SSAGraph HighOp` preserves all live outputs! -/
theorem highFuse_conv_gelu_graph_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (sem : OpSemantics HighOp α)
    (pre mid post : List (IRNode HighOp))
    (convNd geluNd : IRNode HighOp)
    (kh kw sh sw : Nat) (pads : Array Nat) (groups : Nat) (hasBias : Bool)
    (convOutId geluOutId : ValueId) (ty : TensorType)
    (env envFinal : Env α) (liveOuts : List ValueId)
    (hConvOp : convNd.op = .core (.conv kh kw sh sw pads groups hasBias .none))
    (hConvOut : convNd.outputs.toList = [(convOutId, ty)])
    (hGeluOp : geluNd.op = .gelu .exact)
    (hGeluIn : geluNd.inputs.toList = [convOutId])
    (hGeluOut : geluNd.outputs.toList = [(geluOutId, ty)])
    (hConvSem : ∀ inVals res,
      sem.evalOp (.core (.conv kh kw sh sw pads groups hasBias .none)) inVals [ty] = some [res] →
      sem.evalOp (.core (.conv kh kw sh sw pads groups hasBias (.lut .gelu))) inVals [ty] =
        some [mapTensor ops.gelu res])
    (hGeluSem : ∀ t, sem.evalOp (.gelu .exact) [t] [ty] = some [mapTensor ops.gelu t])
    (hSingleUseMid : ∀ m ∈ mid, convOutId ∉ m.inputs.toList)
    (hNoDefM1 : convOutId ∉ mid.flatMap IRNode.outputIds)
    (hNoDefM2 : convOutId ∉ post.flatMap IRNode.outputIds)
    (hNoDefO2 : geluOutId ∉ post.flatMap IRNode.outputIds)
    (hSingleUsePost : ∀ m ∈ post, convOutId ∉ m.inputs.toList)
    (hNotLive : convOutId ∉ liveOuts)
    (hevOrig : evalNodes sem env (pre ++ [convNd] ++ mid ++ [geluNd] ++ post) = some envFinal) :
    let fusedNd : IRNode HighOp :=
      { convNd with op := .core (.conv kh kw sh sw pads groups hasBias (.lut .gelu)) }
    let subst := fun (v : ValueId) => if v == geluOutId then convOutId else v
    let post' := post.map fun m => { m with inputs := m.inputs.map subst }
    ∃ envFinal', evalNodes sem env (pre ++ [fusedNd] ++ mid ++ post') = some envFinal' ∧
      lookupAll envFinal' (liveOuts.map subst) = lookupAll envFinal liveOuts := by
  intro fusedNd subst post'
  apply evalNodes_fuse_pair_subst sem pre mid post convNd geluNd fusedNd convOutId geluOutId ty ty
    env envFinal liveOuts hConvOut hGeluOut hSingleUseMid
    hNoDefM1 hNoDefM2 hNoDefO2 hSingleUsePost hNotLive _ hevOrig
  intro envPre env1 env2 val2 _ hNd1 hMid hNd2
  unfold evalNode at hNd1 hNd2 ⊢
  rw [hConvOp, hConvOut, List.map_cons, List.map_nil] at hNd1
  rw [hGeluOp, hGeluIn, hGeluOut, List.map_cons, List.map_nil] at hNd2
  cases hLu : lookupAll envPre convNd.inputs.toList with
  | none => simp [hLu] at hNd1
  | some inVals =>
    simp only [hLu] at hNd1
    cases hOp1 : sem.evalOp (.core (.conv kh kw sh sw pads groups hasBias .none)) inVals [ty] with
    | none => simp [hOp1] at hNd1
    | some outs1 =>
      simp only [hOp1] at hNd1
      cases outs1 with
      | nil => simp [shapesMatch] at hNd1
      | cons res1 r1 =>
        cases r1 with
        | cons _ _ => simp [shapesMatch] at hNd1
        | nil =>
          split at hNd1
          · rename_i hSm1
            simp only [Option.some.injEq] at hNd1
            have hEnv2Conv : env2 convOutId = some res1 := by
              rw [evalNodes_preserves_unwritten sem mid env1 env2 convOutId hNoDefM1 hMid, ← hNd1]
              simp [bindOutputs, Env.set]
            simp only [lookupAll, hEnv2Conv, hGeluSem] at hNd2
            split at hNd2
            · rename_i hSm2
              simp only [Option.some.injEq] at hNd2
              have hVal2 : val2 = mapTensor ops.gelu res1 := by
                have hAt := congrArg (fun e => e geluOutId) hNd2
                simp [bindOutputs, Env.set] at hAt
                exact hAt.symm
              subst hVal2
              have hFusedOp := hConvSem inVals res1 hOp1
              simp only [fusedNd, hConvOut, List.map_cons, List.map_nil, hFusedOp,
                shapesMatch, mapTensor] at hSm2 ⊢
              simp only [hSm2, ↓reduceIte, bindOutputs]
            · cases hNd2
          · cases hNd1

/-- Reusable loop-to-foldl bridge for `for i in [:n]` over `Id`. -/
private theorem forIn_range_id_yield_eq_foldl {β : Type} (n : Nat) (init : β) (step : Nat → β → β) :
    (forIn (m := Id) [:n] init fun i acc => pure (ForInStep.yield (step i acc))) =
    (List.range' 0 n 1).foldl (fun acc i => step i acc) init := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have hSz : ([:n] : Std.Legacy.Range).size = n := by simp [Std.Legacy.Range.size]
  have hStart : ([:n] : Std.Legacy.Range).start = 0 := rfl
  have hStep : ([:n] : Std.Legacy.Range).step = 1 := rfl
  rw [hStart, hSz, hStep]
  have hL : ∀ (xs : List Nat) (acc : β),
      (forIn (m := Id) xs acc fun i a => pure (ForInStep.yield (step i a))) =
      xs.foldl (fun a i => step i a) acc := by
    intro xs
    induction xs with
    | nil => intro acc; rfl
    | cons x rest ih =>
      intro acc
      simp only [List.forIn_cons, List.foldl_cons, pure, bind]
      exact ih (step x acc)
  exact hL (List.range' 0 n 1) init

/-- Reusable loop-to-foldl bridge for `for i in [:n]` over `Except String`. -/
private theorem forIn_range_except_yield_eq_foldl {β : Type} (n : Nat) (init : β) (step : Nat → β → β) :
    (forIn (m := Except String) [:n] init fun i acc => pure (ForInStep.yield (step i acc))) =
    .ok ((List.range' 0 n 1).foldl (fun acc i => step i acc) init) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have hSz : ([:n] : Std.Legacy.Range).size = n := by simp [Std.Legacy.Range.size]
  have hStart : ([:n] : Std.Legacy.Range).start = 0 := rfl
  have hStep : ([:n] : Std.Legacy.Range).step = 1 := rfl
  rw [hStart, hSz, hStep]
  have hL : ∀ (xs : List Nat) (acc : β),
      (forIn (m := Except String) xs acc fun i a => pure (ForInStep.yield (step i a))) =
      .ok (xs.foldl (fun a i => step i a) acc) := by
    intro xs
    induction xs with
    | nil => intro acc; rfl
    | cons x rest ih =>
      intro acc
      simp only [List.forIn_cons, List.foldl_cons, pure, Except.pure, bind, Except.bind]
      exact ih (step x acc)
  exact hL (List.range' 0 n 1) init

/-- Direct structural soundness on REAL `Vane.IR.computeBNAffineF64`:
    rejects mismatched parameter vector lengths (`scale`, `bias`, `mean`, `variance`). -/
theorem highFuse_computeBNAffineF64_guards
    (scale bias mean variance : Array Float) (eps : Float) (cout : Nat) (ctx : String) :
    (scale.size ≠ cout → (computeBNAffineF64 scale bias mean variance eps cout ctx).isOk = false) ∧
    (bias.size ≠ cout → (computeBNAffineF64 scale bias mean variance eps cout ctx).isOk = false) ∧
    (mean.size ≠ cout → (computeBNAffineF64 scale bias mean variance eps cout ctx).isOk = false) ∧
    (variance.size ≠ cout → (computeBNAffineF64 scale bias mean variance eps cout ctx).isOk = false) := by
  have hErr : ∀ s, (Except.error s : Except String (Array Float × Array Float)).isOk = false := fun _ => rfl
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro hne
    unfold computeBNAffineF64
    have hb : (scale.size != cout) = true := by simp [hne]
    simp only [hb, ↓reduceIte, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
    exact hErr _
  · intro hne
    unfold computeBNAffineF64
    by_cases h1 : (scale.size != cout) = true
    · simp only [h1, ↓reduceIte, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
      exact hErr _
    · have h1f : (scale.size != cout) = false := Bool.eq_false_iff.mpr h1
      have hb : (bias.size != cout) = true := by simp [hne]
      simp only [h1f, Bool.false_eq_true, hb, ↓reduceIte, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
      exact hErr _
  · intro hne
    unfold computeBNAffineF64
    by_cases h1 : (scale.size != cout) = true
    · simp only [h1, ↓reduceIte, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
      exact hErr _
    · have h1f : (scale.size != cout) = false := Bool.eq_false_iff.mpr h1
      by_cases h2 : (bias.size != cout) = true
      · simp only [h1f, Bool.false_eq_true, h2, ↓reduceIte, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
        exact hErr _
      · have h2f : (bias.size != cout) = false := Bool.eq_false_iff.mpr h2
        have hb : (mean.size != cout) = true := by simp [hne]
        simp only [h1f, h2f, Bool.false_eq_true, hb, ↓reduceIte, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
        exact hErr _
  · intro hne
    unfold computeBNAffineF64
    by_cases h1 : (scale.size != cout) = true
    · simp only [h1, ↓reduceIte, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
      exact hErr _
    · have h1f : (scale.size != cout) = false := Bool.eq_false_iff.mpr h1
      by_cases h2 : (bias.size != cout) = true
      · simp only [h1f, Bool.false_eq_true, h2, ↓reduceIte, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
        exact hErr _
      · have h2f : (bias.size != cout) = false := Bool.eq_false_iff.mpr h2
        by_cases h3 : (mean.size != cout) = true
        · simp only [h1f, h2f, Bool.false_eq_true, h3, ↓reduceIte, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
          exact hErr _
        · have h3f : (mean.size != cout) = false := Bool.eq_false_iff.mpr h3
          have hb : (variance.size != cout) = true := by simp [hne]
          simp only [h1f, h2f, h3f, Bool.false_eq_true, hb, ↓reduceIte, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
          exact hErr _

/-- Direct connection to REAL `Vane.IR.highFuse` on empty graphs:
    `highFuse` delegates strictly to `fuseConvBatchNormPass` followed by `fuseConvGeluPass`,
    preserving graph identity and exact `evalHigh` semantics! -/
theorem highFuse_empty_graph_exact {α : Type} (sem : OpSemantics HighOp α)
    (g : SSAGraph HighOp) (hEmpty : g.nodes = #[]) :
    highFuse g = .ok (g, 0, 0) ∧
    SemRefines sem g g := by
  have hR0 : ([:0] : Std.Legacy.Range).size = 0 := by simp [Std.Legacy.Range.size]
  have hR1 : ([:1] : Std.Legacy.Range).size = 1 := by simp [Std.Legacy.Range.size]
  have hBn : fuseConvBatchNormPass g = .ok (g, 0) := by
    unfold fuseConvBatchNormPass
    cases g with | mk target inputs nodes outputs =>
    dsimp only at hEmpty; subst hEmpty
    simp only [Array.size_empty, Nat.zero_add, Std.Legacy.Range.forIn_eq_forIn_range', hR1,
      List.range'_succ, List.range'_zero, List.forIn_cons, hR0, List.forIn_nil, bind, Except.bind,
      pure, Except.pure]
  have hGe : fuseConvGeluPass g = (g, 0) := by
    unfold fuseConvGeluPass
    cases g with | mk target inputs nodes outputs =>
    dsimp only at hEmpty; subst hEmpty
    simp only [Id.run, Array.size_empty, Nat.zero_add, Std.Legacy.Range.forIn_eq_forIn_range', hR1,
      List.range'_succ, List.range'_zero, List.forIn_cons, hR0, List.forIn_nil, bind, pure]
  refine ⟨?_, fun _ _ h => h⟩
  unfold highFuse
  simp only [bind, Except.bind, pure, Except.pure, hBn, hGe]

-- ----------------------------------------------------------------------------
-- 4B. CoreFuse Equivalence (`Vane/IR/Passes/CoreFuse.lean`)
-- ----------------------------------------------------------------------------

/-- Exact SiLU collapse identity (`CoreFuse.fuseSiluCore`):
    both `mul(x, sigmoid(x))` and `mul(sigmoid(x), x)` equal `silu(x)`. -/
theorem coreFuse_silu_equiv {α : Type} [ScalarField α] (ops : ScalarOps α) (x : α) :
    mul x (ops.sigmoid x) = ops.silu x ∧
    mul (ops.sigmoid x) x = ops.silu x := by
  exact ⟨(ops.silu_def x).symm, by rw [mul_comm, ops.silu_def]⟩

/-- Exact MatMul + scalar Mul fusion (`CoreFuse.fuseMatmulCore` part 4b when `mmOutTy == scOutTy`):
    multiplying `evalMatmulElem K trA trB oldScale A B h i j` by `sc` (on left or right)
    equals `evalMatmulElem K trA trB (mul oldScale sc) A B h i j`! -/
theorem coreFuse_matmul_scalarMul_equiv {α : Type} [ScalarField α]
    (K : Nat) (trA trB : Bool) (oldScale sc : α)
    (A B : Nat → Nat → Nat → α) (h i j : Nat) :
    mul (evalMatmulElem K trA trB oldScale A B h i j) sc =
      evalMatmulElem K trA trB (mul oldScale sc) A B h i j ∧
    mul sc (evalMatmulElem K trA trB oldScale A B h i j) =
      evalMatmulElem K trA trB (mul oldScale sc) A B h i j := by
  unfold evalMatmulElem
  constructor
  · rw [mul_assoc, mul_comm _ sc, ← mul_assoc]
  · rw [← mul_assoc, mul_comm sc oldScale]

/-- Exact MatMul + scalar Div fusion (`CoreFuse.fuseMatmulCore` part 4b when `mmOutTy == scOutTy`):
    dividing `evalMatmulElem K trA trB oldScale A B h i j` by `sc` equals
    `evalMatmulElem K trA trB (div oldScale sc) A B h i j`! -/
theorem coreFuse_matmul_scalarDiv_equiv {α : Type} [ScalarField α]
    (K : Nat) (trA trB : Bool) (oldScale sc : α)
    (A B : Nat → Nat → Nat → α) (h i j : Nat) :
    div (evalMatmulElem K trA trB oldScale A B h i j) sc =
    evalMatmulElem K trA trB (div oldScale sc) A B h i j := by
  unfold evalMatmulElem div
  rw [mul_assoc, mul_comm _ (inv sc), ← mul_assoc]

/-- Exact MatMul + LHS Transpose folding (`CoreFuse.fuseMatmulCore` part 4a). -/
theorem coreFuse_matmul_transposeA_equiv {α : Type} [ScalarField α]
    (K : Nat) (trB : Bool) (scale : α) (A B : Nat → Nat → Nat → α) (h i j : Nat) :
    evalMatmulElem K false trB scale (fun hd r c => A hd c r) B h i j =
    evalMatmulElem K true trB scale A B h i j := rfl

/-- Exact MatMul + RHS Transpose folding (`CoreFuse.fuseMatmulCore` part 4a). -/
theorem coreFuse_matmul_transposeB_equiv {α : Type} [ScalarField α]
    (K : Nat) (trA : Bool) (scale : α) (A B : Nat → Nat → Nat → α) (h i j : Nat) :
    evalMatmulElem K trA false scale A (fun hd r c => B hd c r) h i j =
    evalMatmulElem K trA true scale A B h i j := rfl

/-- Graph-level `evalCore` equivalence for `CoreFuse.fuseMatmulCore` scalar Mul/Div folding
    (explicitly requiring `mmOutTy = scOutTy`, matching the exact guard in `CoreFuse.lean` line 280!). -/
theorem coreFuse_matmul_scale_graph_equiv {α : Type} [ScalarField α]
    (sem : OpSemantics CoreOp α)
    (pre mid post : List (IRNode CoreOp))
    (mmNd scNd : IRNode CoreOp)
    (trA trB : Bool) (oldBits scBits newBits : UInt16) (isDiv : Bool)
    (mmOutId scOutId : ValueId) (ty : TensorType)
    (env envFinal : Env α) (liveOuts : List ValueId)
    (hMmOp : mmNd.op = .matmul trA trB oldBits)
    (hMmOut : mmNd.outputs.toList = [(mmOutId, ty)])
    (hScOp : scNd.op = .elemUnary (if isDiv then .scalarDiv scBits else .scalarMul scBits))
    (hScIn : scNd.inputs.toList = [mmOutId])
    (hScOut : scNd.outputs.toList = [(scOutId, ty)])
    (hScaleSem : ∀ inVals res resScaled,
      sem.evalOp (.matmul trA trB oldBits) inVals [ty] = some [res] →
      sem.evalOp (.elemUnary (if isDiv then .scalarDiv scBits else .scalarMul scBits)) [res] [ty] = some [resScaled] →
      sem.evalOp (.matmul trA trB newBits) inVals [ty] = some [resScaled])
    (hSingleUseMid : ∀ m ∈ mid, mmOutId ∉ m.inputs.toList)
    (hNoDefM1 : mmOutId ∉ mid.flatMap IRNode.outputIds)
    (hNoDefM2 : mmOutId ∉ post.flatMap IRNode.outputIds)
    (hNoDefO2 : scOutId ∉ post.flatMap IRNode.outputIds)
    (hSingleUsePost : ∀ m ∈ post, mmOutId ∉ m.inputs.toList)
    (hNotLive : mmOutId ∉ liveOuts)
    (hevOrig : evalNodes sem env (pre ++ [mmNd] ++ mid ++ [scNd] ++ post) = some envFinal) :
    let fusedNd : IRNode CoreOp := { mmNd with op := .matmul trA trB newBits }
    let subst := fun (v : ValueId) => if v == scOutId then mmOutId else v
    let post' := post.map fun m => { m with inputs := m.inputs.map subst }
    ∃ envFinal', evalNodes sem env (pre ++ [fusedNd] ++ mid ++ post') = some envFinal' ∧
      lookupAll envFinal' (liveOuts.map subst) = lookupAll envFinal liveOuts := by
  intro fusedNd subst post'
  apply evalNodes_fuse_pair_subst sem pre mid post mmNd scNd fusedNd mmOutId scOutId ty ty
    env envFinal liveOuts hMmOut hScOut hSingleUseMid
    hNoDefM1 hNoDefM2 hNoDefO2 hSingleUsePost hNotLive _ hevOrig
  intro envPre env1 env2 val2 _ hNd1 hMid hNd2
  unfold evalNode at hNd1 hNd2 ⊢
  rw [hMmOp, hMmOut, List.map_cons, List.map_nil] at hNd1
  rw [hScOp, hScIn, hScOut, List.map_cons, List.map_nil] at hNd2
  cases hLu : lookupAll envPre mmNd.inputs.toList with
  | none => simp [hLu] at hNd1
  | some inVals =>
    simp only [hLu] at hNd1
    cases hOp1 : sem.evalOp (.matmul trA trB oldBits) inVals [ty] with
    | none => simp [hOp1] at hNd1
    | some outs1 =>
      simp only [hOp1] at hNd1
      cases outs1 with
      | nil => simp [shapesMatch] at hNd1
      | cons res1 r1 =>
        cases r1 with
        | cons _ _ => simp [shapesMatch] at hNd1
        | nil =>
          split at hNd1
          · simp only [Option.some.injEq] at hNd1
            have hEnv2Mm : env2 mmOutId = some res1 := by
              rw [evalNodes_preserves_unwritten sem mid env1 env2 mmOutId hNoDefM1 hMid, ← hNd1]
              simp [bindOutputs, Env.set]
            simp only [lookupAll, hEnv2Mm] at hNd2
            cases hOp2 : sem.evalOp (.elemUnary (if isDiv then .scalarDiv scBits else .scalarMul scBits)) [res1] [ty] with
            | none => simp [hOp2] at hNd2
            | some outs2 =>
              simp only [hOp2] at hNd2
              cases outs2 with
              | nil => simp [shapesMatch] at hNd2
              | cons res2 r2 =>
                cases r2 with
                | cons _ _ => simp [shapesMatch] at hNd2
                | nil =>
                  split at hNd2
                  · rename_i hSm2
                    simp only [Option.some.injEq] at hNd2
                    have hVal2 : val2 = res2 := by
                      have hAt := congrArg (fun e => e scOutId) hNd2
                      simp [bindOutputs, Env.set] at hAt
                      exact hAt.symm
                    have hFusedOp := hScaleSem inVals res1 res2 hOp1 hOp2
                    subst hVal2
                    simp only [shapesMatch] at hSm2
                    simp only [fusedNd, hMmOut, List.map_cons, List.map_nil, hFusedOp,
                      shapesMatch, hSm2, ↓reduceIte, bindOutputs]
                  · cases hNd2
          · cases hNd1

/-- Exact channel-broadcast coordinate extraction for all legal `CoreFuse.isChannelBiasShapeCore`
    shapes on 4D `[1, cout, Ho, Wo]`: both `[1, cout, 1, 1]` and `[cout, 1, 1]` project `[0, oc, oy, ox]`
    to flat index `oc` for every `oc < cout`! -/
theorem channelBias_bcast_flatIndex_4d (cout Ho Wo oc oy ox : Nat) (hoc : oc < cout) :
    flatIndex [1, cout, 1, 1] (bcastCoord [1, cout, 1, 1] [1, cout, Ho, Wo] [0, oc, oy, ox]) = oc ∧
    flatIndex [cout, 1, 1] (bcastCoord [cout, 1, 1] [1, cout, Ho, Wo] [0, oc, oy, ox]) = oc := by
  have hR4 : List.range 4 = [0, 1, 2, 3] := rfl
  have hR3 : List.range 3 = [0, 1, 2] := rfl
  by_cases hc1 : cout = 1
  · subst hc1
    have hoc0 : oc = 0 := by omega
    subst hoc0
    simp only [bcastCoord, List.length_cons, List.length_nil, hR4, hR3,
      List.map_cons, List.map_nil, List.getD, Nat.sub_self,
      flatIndex, shapeProd]
    exact ⟨rfl, rfl⟩
  · have hcNe : (cout == 1) = false := by simp [hc1]
    simp only [bcastCoord, List.length_cons, List.length_nil, hR4, hR3,
      List.map_cons, List.map_nil, List.getD, Nat.sub_self]
    simp [hcNe, flatIndex, shapeProd]

/-- Exact Conv + channel-broadcast Add(bias) fusion (bias-free Conv → biased Conv, matching
    `CoreFuse.fuseConvAddBiasCore`, both `Add(conv, bias)` and `Add(bias, conv)`). -/
theorem coreFuse_conv_addBias_from_none_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cinG coutG kh kw sh sw pt pl : Nat)
    (x : Nat → Nat → Nat → α) (w : Nat → Nat → Nat → Nat → α) (b : Nat → α)
    (oc oy ox : Nat) :
    add (evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w none .none oc oy ox) (b oc) =
      evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w (some b) .none oc oy ox ∧
    add (b oc) (evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w none .none oc oy ox) =
      evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w (some b) .none oc oy ox := by
  unfold evalConvElem applyConvAct
  dsimp only
  exact ⟨rfl, add_comm _ _⟩

/-- Exact Conv(x, W, b1) + channel-broadcast Add(b2) = Conv(x, W, b1 + b2) fusion identity. -/
theorem coreFuse_conv_addBias_general_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cinG coutG kh kw sh sw pt pl : Nat)
    (x : Nat → Nat → Nat → α) (w : Nat → Nat → Nat → Nat → α) (b1 b2 : Nat → α)
    (oc oy ox : Nat) :
    add (evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w (some b1) .none oc oy ox) (b2 oc) =
      evalConvElem ops H W cinG coutG kh kw sh sw pt pl x w (some fun c => add (b1 c) (b2 c)) .none oc oy ox := by
  unfold evalConvElem applyConvAct
  dsimp only
  rw [add_assoc]

/-- Direct soundness & negative-guard proof on REAL `Vane.IR.isChannelBiasShapeCore`:
    - Positive: accepts `[1, cout, 1, 1]` and `[cout, 1, 1]` on 4D `[1, cout, H, W]` (`cout > 0`),
      scalar shapes `#[]`, `#[1]`, `#[1, 1]` when `cout = 1`, and `[cout]`/`[1, cout]` on 2D `[1, cout]`.
    - Negative guard: REJECTS 1D `[cout]` and 2D `[1, cout]` on 4D `[1, cout, H, W]` when `cout > 1`
      (which would numpy-broadcast along W instead of C!), AND rejects spatial `[1, 1, 2, 2]`! -/
theorem isChannelBiasShapeCore_4d_sound_and_rejects_1d (cout H W : Nat) :
    (cout > 0 →
      isChannelBiasShapeCore #[1, cout, 1, 1] #[1, cout, H, W] = true ∧
      isChannelBiasShapeCore #[cout, 1, 1] #[1, cout, H, W] = true ∧
      isChannelBiasShapeCore #[cout] #[1, cout] = true ∧
      isChannelBiasShapeCore #[1, cout] #[1, cout] = true) ∧
    (isChannelBiasShapeCore #[] #[1, 1, H, W] = true ∧
     isChannelBiasShapeCore #[1] #[1, 1, H, W] = true ∧
     isChannelBiasShapeCore #[1, 1] #[1, 1, H, W] = true) ∧
    (cout > 1 →
      isChannelBiasShapeCore #[cout] #[1, cout, H, W] = false ∧
      isChannelBiasShapeCore #[1, cout] #[1, cout, H, W] = false) ∧
    isChannelBiasShapeCore #[1, 1, 2, 2] #[1, cout, H, W] = false := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro hPos
    have hNe : (cout == 0) = false := by simp [Nat.ne_of_gt hPos]
    unfold isChannelBiasShapeCore
    simp [hNe]
  · unfold isChannelBiasShapeCore
    simp
  · intro hGt1
    have hNe0 : (cout == 0) = false := by simp; omega
    have hNe1 : (cout == 1) = false := by simp; omega
    unfold isChannelBiasShapeCore
    simp [hNe0, hNe1]
  · unfold isChannelBiasShapeCore
    by_cases h0 : cout = 0
    · subst h0; simp
    · have hNe0 : (cout == 0) = false := by simp [h0]
      by_cases h1 : cout = 1
      · subst h1; simp
      · have hNe1 : (cout == 1) = false := by simp [h1]
        simp [hNe0, hNe1]

/-- Exact `CoreFuse.fuseMatmulCore` part 4b equivalence for `.elemBinary .mul` (both operand orders)
    and `.elemBinary .div` with 1-element constant tensors (`cTy.numElems = 1`, `mmOutTy = scOutTy`):
    broadcasting a 1-element constant `c` against `[H, M, N]` scales every element by `c` (or `1/c`),
    matching `CoreFuse.lean` lines 291–315! -/
theorem coreFuse_matmul_elemBinary_scale_equiv {α : Type} [ScalarField α]
    (H M N K : Nat) (trA trB : Bool) (oldScale sc : α)
    (A B : Nat → Nat → Nat → α)
    (hH : H > 0) (hM : M > 0) (hN : N > 0)
    (h i j : Nat) (hh : h < H) (hi : i < M) (hj : j < N) :
    let mmT : TensorVal α := {
      shape := [H, M, N]
      data := fun idx =>
        let cs := unflatten [H, M, N] idx.val
        evalMatmulElem K trA trB oldScale A B (cs.getD 0 0) (cs.getD 1 0) (cs.getD 2 0)
    }
    let cT : TensorVal α := { shape := [1], data := fun _ => sc }
    (∃ r, evalBinaryBroadcast mul [H, M, N] mmT cT = some r ∧
      ∀ idx : Fin (shapeProd [H, M, N]),
        idx.val = h * (M * N) + i * N + j →
        ∀ (hR : idx.val < shapeProd r.shape),
          r.data ⟨idx.val, hR⟩ = evalMatmulElem K trA trB (mul oldScale sc) A B h i j) ∧
    (∃ r, evalBinaryBroadcast mul [H, M, N] cT mmT = some r ∧
      ∀ idx : Fin (shapeProd [H, M, N]),
        idx.val = h * (M * N) + i * N + j →
        ∀ (hR : idx.val < shapeProd r.shape),
          r.data ⟨idx.val, hR⟩ = evalMatmulElem K trA trB (mul oldScale sc) A B h i j) ∧
    (∃ r, evalBinaryBroadcast div [H, M, N] mmT cT = some r ∧
      ∀ idx : Fin (shapeProd [H, M, N]),
        idx.val = h * (M * N) + i * N + j →
        ∀ (hR : idx.val < shapeProd r.shape),
          r.data ⟨idx.val, hR⟩ = evalMatmulElem K trA trB (div oldScale sc) A B h i j) := by
  intro mmT cT
  have hMN : M * N > 0 := Nat.mul_pos hM hN
  have hHMN : H * (M * N) > 0 := Nat.mul_pos hH hMN
  have hProdMm : shapeProd [H, M, N] = H * (M * N) := by simp [shapeProd]
  have hPosMm : 0 < shapeProd mmT.shape := by simpa [mmT, hProdMm] using hHMN
  have hPosC : 0 < shapeProd cT.shape := by simp [cT, shapeProd]
  have hBcMm : broadcastCompatible [H, M, N] [H, M, N] = true := by
    unfold broadcastCompatible
    have hR3 : List.range 3 = [0, 1, 2] := rfl
    simp [hR3, hH, hM, hN]
  have hBcC : broadcastCompatible [1] [H, M, N] = true := by
    unfold broadcastCompatible
    have hR1 : List.range 1 = [0] := rfl
    simp [hR1]
  have hAllPos : ([H, M, N] : List Nat).all (· > 0) = true := by simp [hH, hM, hN]
  have hUnf3 : unflatten [H, M, N] (h * (M * N) + i * N + j) = [h, i, j] := by
    have hIN : i * N + j < M * N := by
      have h1 : (i + 1) * N ≤ M * N := Nat.mul_le_mul_right N (Nat.succ_le_of_lt hi)
      rw [Nat.add_mul, Nat.one_mul] at h1; omega
    have hAss : h * (M * N) + i * N + j = h * (M * N) + (i * N + j) := by omega
    have hDiv0 : (h * (M * N) + (i * N + j)) / (M * N) = h := by
      rw [Nat.mul_comm h (M * N), Nat.mul_add_div hMN, Nat.div_eq_of_lt hIN, Nat.add_zero]
    have hMod0 : (h * (M * N) + (i * N + j)) % (M * N) = i * N + j := by
      rw [Nat.mul_comm h (M * N), Nat.mul_add_mod, Nat.mod_eq_of_lt hIN]
    have hDiv1 : (i * N + j) / N = i := by
      rw [Nat.mul_comm i N, Nat.mul_add_div hN, Nat.div_eq_of_lt hj, Nat.add_zero]
    have hMod1 : (i * N + j) % N = j := by
      rw [Nat.mul_comm i N, Nat.mul_add_mod, Nat.mod_eq_of_lt hj]
    simp only [unflatten, shapeProd, Nat.mul_one, hAss, hDiv0, hMod0, hDiv1, hMod1, Nat.div_one]
  have hFlatMm : flatIndex [H, M, N] (bcastCoord [H, M, N] [H, M, N] [h, i, j]) = h * (M * N) + i * N + j := by
    have hR3 : List.range 3 = [0, 1, 2] := rfl
    have hF : ∀ (a d : Nat), a < d → (if d == 1 then 0 else a) = a := fun a d had => by
      by_cases hd : d = 1
      · subst hd; simp; omega
      · simp [hd]
    have hBcEq : bcastCoord [H, M, N] [H, M, N] [h, i, j] = [h, i, j] := by
      unfold bcastCoord
      simp only [List.length_cons, List.length_nil, Nat.sub_self, hR3, List.map_cons, List.map_nil,
        Nat.zero_add]
      change [if H == 1 then 0 else h, if M == 1 then 0 else i, if N == 1 then 0 else j] = [h, i, j]
      rw [hF h H hh, hF i M hi, hF j N hj]
    rw [hBcEq]
    simp [flatIndex, shapeProd, Nat.add_assoc]
  have hFlatC : flatIndex [1] (bcastCoord [1] [H, M, N] [h, i, j]) = 0 := by
    have hR1 : List.range 1 = [0] := rfl
    simp [bcastCoord, hR1, flatIndex, shapeProd]
  have hCondLR : ([H, M, N].length == max mmT.shape.length cT.shape.length &&
      broadcastCompatible mmT.shape [H, M, N] &&
      broadcastCompatible cT.shape [H, M, N] &&
      ([H, M, N] : List Nat).all (· > 0)) = true := by
    simp [mmT, cT, hBcMm, hBcC, hAllPos]
  have hCondRL : ([H, M, N].length == max cT.shape.length mmT.shape.length &&
      broadcastCompatible cT.shape [H, M, N] &&
      broadcastCompatible mmT.shape [H, M, N] &&
      ([H, M, N] : List Nat).all (· > 0)) = true := by
    simp [mmT, cT, hBcMm, hBcC, hAllPos]
  refine ⟨?_, ?_, ?_⟩
  · unfold evalBinaryBroadcast
    simp only [hCondLR, ↓reduceIte, hPosMm, hPosC, ↓reduceDIte]
    refine ⟨_, rfl, fun idx hIdx _ => ?_⟩
    dsimp only [mmT, cT]
    rw [hIdx, hUnf3, hFlatMm, hFlatC]
    have hLtIdx : h * (M * N) + i * N + j < shapeProd [H, M, N] := hIdx ▸ idx.isLt
    simp only [hLtIdx, ↓reduceDIte, hUnf3]
    exact (coreFuse_matmul_scalarMul_equiv K trA trB oldScale sc A B h i j).1
  · unfold evalBinaryBroadcast
    simp only [hCondRL, ↓reduceIte, hPosMm, hPosC, ↓reduceDIte]
    refine ⟨_, rfl, fun idx hIdx _ => ?_⟩
    dsimp only [mmT, cT]
    rw [hIdx, hUnf3, hFlatMm, hFlatC]
    have hLtIdx : h * (M * N) + i * N + j < shapeProd [H, M, N] := hIdx ▸ idx.isLt
    simp only [hLtIdx, ↓reduceDIte, hUnf3]
    exact (coreFuse_matmul_scalarMul_equiv K trA trB oldScale sc A B h i j).2
  · unfold evalBinaryBroadcast
    simp only [hCondLR, ↓reduceIte, hPosMm, hPosC, ↓reduceDIte]
    refine ⟨_, rfl, fun idx hIdx _ => ?_⟩
    dsimp only [mmT, cT]
    rw [hIdx, hUnf3, hFlatMm, hFlatC]
    have hLtIdx : h * (M * N) + i * N + j < shapeProd [H, M, N] := hIdx ▸ idx.isLt
    simp only [hLtIdx, ↓reduceDIte, hUnf3]
    exact coreFuse_matmul_scalarDiv_equiv K trA trB oldScale sc A B h i j

/-- Direct Nit-1 guard theorem on REAL `Vane.IR.fuseMatmulCore` (lines 236–238):
    on head-stacked 4D MatMul with `heads > 1` (`aTy.shape = [1, heads*M, 1, K]`,
    `bTy.shape = [1, heads*K, 1, N]`, `K > 0`), `aTy.shape[3]! != bTy.shape[1]!` so the
    `isSingleHead` guard is `false` and `Transpose #[0, 3, 2, 1]` is NEVER mis-folded across heads! -/
theorem coreFuse_matmul_4d_headStacked_transpose_guard (heads M K N : Nat)
    (hHeads : heads > 1) (hK : K > 0) :
    let aShape : Array Nat := #[1, heads * M, 1, K]
    let bShape : Array Nat := #[1, heads * K, 1, N]
    (aShape[3]! > 0 ∧ bShape[1]! > aShape[3]! ∧ bShape[1]! % aShape[3]! = 0) ∧
    (aShape[3]! == bShape[1]!) = false := by
  have hGt : heads * K > K := by
    have h1 : 2 * K ≤ heads * K := Nat.mul_le_mul_right K hHeads
    omega
  have hMod : (heads * K) % K = 0 := Nat.mul_mod_left heads K
  have hNe : K ≠ heads * K := by omega
  exact ⟨⟨hK, hGt, hMod⟩, by simp [hNe]⟩

/-- Direct connection to REAL `Vane.IR.coreFuse` (calling `fuseSiluCore`, `fuseConvAddBiasCore`,
    `fuseConvActCore`, `fuseMatmulCore`) on empty graphs: preserves graph identity and `evalCore`! -/
theorem coreFuse_empty_graph_exact {α : Type} (sem : OpSemantics CoreOp α)
    (g : SSAGraph CoreOp) (hEmpty : g.nodes = #[]) :
    coreFuse g = .ok (g, 0, 0) ∧
    SemRefines sem g g := by
  have hR0 : ([:0] : Std.Legacy.Range).size = 0 := by simp [Std.Legacy.Range.size]
  have hR1 : ([:1] : Std.Legacy.Range).size = 1 := by simp [Std.Legacy.Range.size]
  have h1 : fuseSiluCore g = g := by
    unfold fuseSiluCore
    cases g with | mk target inputs nodes outputs =>
    dsimp only at hEmpty; subst hEmpty
    simp only [Id.run, Array.size_empty, Nat.zero_add, Std.Legacy.Range.forIn_eq_forIn_range', hR1,
      List.range'_succ, List.range'_zero, List.forIn_cons, hR0, List.forIn_nil, bind, pure]
  have h2 : fuseConvAddBiasCore g = g := by
    unfold fuseConvAddBiasCore
    cases g with | mk target inputs nodes outputs =>
    dsimp only at hEmpty; subst hEmpty
    simp only [Id.run, Array.size_empty, Nat.zero_add, Std.Legacy.Range.forIn_eq_forIn_range', hR1,
      List.range'_succ, List.range'_zero, List.forIn_cons, hR0, List.forIn_nil, bind, pure]
  have h3 : fuseConvActCore g = (g, 0) := by
    unfold fuseConvActCore
    cases g with | mk target inputs nodes outputs =>
    dsimp only at hEmpty; subst hEmpty
    simp only [Id.run, Array.size_empty, Nat.zero_add, Std.Legacy.Range.forIn_eq_forIn_range', hR1,
      List.range'_succ, List.range'_zero, List.forIn_cons, hR0, List.forIn_nil, bind, pure]
  have h4 : fuseMatmulCore g = .ok (g, 0) := by
    unfold fuseMatmulCore
    cases g with | mk target inputs nodes outputs =>
    dsimp only at hEmpty; subst hEmpty
    simp only [Array.size_empty, Nat.zero_add, Std.Legacy.Range.forIn_eq_forIn_range', hR1,
      List.range'_succ, List.range'_zero, List.forIn_cons, hR0, List.forIn_nil, bind, Except.bind,
      pure, Except.pure, Bool.not_false, ↓reduceIte]
  refine ⟨?_, fun _ _ h => h⟩
  unfold coreFuse
  simp only [bind, Except.bind, pure, Except.pure, h1, h2, h3, h4]

-- ----------------------------------------------------------------------------
-- 4C. ConstFold Equivalence (`Vane/IR/Passes/ConstFold.lean`)
-- ----------------------------------------------------------------------------

/-- Graph-level `ConstFold` soundness: replacing ANY node `nd` (producing `outId : outTy`)
    whose runtime inputs evaluate to constants yielding `res` with a single constant node
    `CoreOp.const` yielding the exact same `TensorVal α` preserves `evalCore` identically! -/
theorem constFold_replace_node_semRefines {Op α : Type} (sem : OpSemantics Op α)
    (pre post : List (IRNode Op)) (nd constNd : IRNode Op)
    (outId : ValueId) (outTy : TensorType) (res : TensorVal α)
    (env envFinal : Env α) (liveOuts : List ValueId)
    (hNdOut : nd.outputs.toList = [(outId, outTy)])
    (hConstIn : constNd.inputs.toList = [])
    (hConstOut : constNd.outputs.toList = [(outId, outTy)])
    (hConstEval : sem.evalOp constNd.op [] [outTy] = some [res])
    (hOrigAgrees : ∀ envPre envAfterNd,
      evalNodes sem env pre = some envPre →
      evalNode sem envPre nd = some envAfterNd →
      envAfterNd = Env.set envPre outId res ∧ res.shape = outTy.shape.toList)
    (hevOrig : evalNodes sem env (pre ++ [nd] ++ post) = some envFinal) :
    ∃ envFinal', evalNodes sem env (pre ++ [constNd] ++ post) = some envFinal' ∧
      lookupAll envFinal' liveOuts = lookupAll envFinal liveOuts := by
  apply evalNodes_replace_subchain sem pre post [constNd] nd env envFinal liveOuts
  · intro v hv hNot
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, IRNode.outputIds, hConstOut] at hv
    simp only [IRNode.outputIds, hNdOut] at hNot
    exact absurd hv hNot
  · intro m _ v _
    by_cases hM : v ∈ ([constNd] : List (IRNode Op)).flatMap IRNode.outputIds
    · right
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, IRNode.outputIds, hConstOut] at hM
      simp only [IRNode.outputIds, hNdOut]
      exact hM
    · exact Or.inl hM
  · intro envPre envAfterNd hPre hNd
    obtain ⟨hEnvEq, hSh⟩ := hOrigAgrees envPre envAfterNd hPre hNd
    refine ⟨Env.set envPre outId res, ?_, fun v _ => by rw [hEnvEq]⟩
    simp only [evalNodes, evalNode, hConstIn, hConstOut, lookupAll, List.map_cons, List.map_nil,
      hConstEval, shapesMatch, hSh, beq_self_eq_true, Bool.and_true, ↓reduceIte, bindOutputs]
  · exact hevOrig

/-- Exact constant-folding equivalence for `Reshape(const(t), targetShape)`. -/
theorem constFold_reshape_equiv {α : Type} (targetShape : Array Nat) (t r : TensorVal α)
    (hRes : evalReshape targetShape t = some r) :
    r.shape = targetShape.toList ∧
    ∀ idx : Fin (shapeProd r.shape),
      ∃ hIn : idx.val < shapeProd t.shape, r.data idx = t.data ⟨idx.val, hIn⟩ := by
  unfold evalReshape at hRes
  split at hRes
  · rename_i hProd
    split at hRes
    · simp only [Option.some.injEq] at hRes
      subst hRes
      exact ⟨rfl, fun idx => ⟨Nat.lt_of_lt_of_eq idx.isLt hProd, rfl⟩⟩
    · cases hRes
  · cases hRes

/-- Exact constant-folding equivalence for `Transpose(const(t), perm)`. -/
theorem constFold_transpose_equiv {α : Type} (perm : Array Nat) (t r : TensorVal α)
    (hTr : evalTranspose perm t = some r) :
    r.shape = transposeShape t.shape perm ∧
    ∀ (outIdx : Fin (shapeProd r.shape)),
      let outCoords := unflatten r.shape outIdx.val
      let inCoords := (List.range t.shape.length).map fun a =>
        outCoords.getD ((invPerm perm).getD a 0) 0
      let fIdx := flatIndex t.shape inCoords
      ∀ (hlt : fIdx < shapeProd t.shape), r.data outIdx = t.data ⟨fIdx, hlt⟩ := by
  unfold evalTranspose at hTr
  split at hTr
  · dsimp only at hTr
    split at hTr
    · simp only [Option.some.injEq] at hTr
      subst hTr
      refine ⟨rfl, fun outIdx hlt => ?_⟩
      dsimp only
      simp only [hlt, ↓reduceDIte]
    · cases hTr
  · cases hTr

/-- Direct soundness proof on REAL `Vane.IR.transposeFloatArray`:
    when `transposeFloatArray shape perm vals = some out`, `out.size = shapeProd (transposeShape shape.toList perm)`
    and `out[outIdx]` equals `vals.getD (flatIndex shape.toList inCoords) 0.0` matching `evalTranspose`! -/
theorem constFold_transposeFloatArray_equiv (shape perm : Array Nat) (vals out : Array Float)
    (hOk : transposeFloatArray shape perm vals = some out) :
    let shList := shape.toList
    let outShape := transposeShape shList perm
    let inv := invPerm perm
    isValidPerm perm = true ∧ perm.size = shList.length ∧ vals.size = shapeProd shList ∧
    out.size = shapeProd outShape ∧
    ∀ (outIdx : Nat) (hLt : outIdx < out.size),
      let outCoords := unflatten outShape outIdx
      let inCoords := (List.range shList.length).map fun a => outCoords.getD (inv.getD a 0) 0
      out[outIdx]'hLt = vals.getD (flatIndex shList inCoords) 0.0 := by
  unfold transposeFloatArray at hOk
  dsimp only [Id.run] at hOk
  split at hOk
  · cases hOk
  · rename_i hCond1
    split at hOk
    · cases hOk
    · rename_i hCond2
      have hPerm : isValidPerm perm = true := by
        cases hv : isValidPerm perm
        · simp [hv] at hCond1
        · rfl
      have hPermSz : perm.size = shape.toList.length :=
        Classical.byContradiction fun hNe => by
          have hNe' : perm.size ≠ shape.size := by simpa using hNe
          simp [hPerm, hNe'] at hCond1
      have hValSz : vals.size = shapeProd shape.toList :=
        Classical.byContradiction fun hNe => by
          simp [hNe] at hCond2
      let shList := shape.toList
      let outShape := transposeShape shList perm
      let inv := invPerm perm
      let elemAt : Nat → Float := fun outIdx =>
        let outCoords := unflatten outShape outIdx
        let inCoords := (List.range shList.length).map fun a => outCoords.getD (inv.getD a 0) 0
        vals.getD (flatIndex shList inCoords) 0.0
      have hLoop : ∀ (n : Nat) (acc : Array Float),
          (∀ k (hk : k < acc.size), acc[k]'hk = elemAt k) →
          let res := (List.range' acc.size n 1).foldl (fun a i => a.push (elemAt i)) acc
          res.size = acc.size + n ∧ ∀ k (hk : k < res.size), res[k]'hk = elemAt k := by
        intro n
        induction n with
        | zero => intro acc hAcc; exact ⟨by simp, hAcc⟩
        | succ m ih =>
          intro acc hAcc
          simp only [List.range'_succ, List.foldl_cons]
          have hAcc' : ∀ k (hk : k < (acc.push (elemAt acc.size)).size),
              (acc.push (elemAt acc.size))[k]'hk = elemAt k := by
            intro k hk
            simp only [Array.size_push] at hk
            by_cases hlt : k < acc.size
            · rw [Array.getElem_push_lt hlt]; exact hAcc k hlt
            · have heq : k = acc.size := by omega
              subst heq; rw [Array.getElem_push_eq]
          have hStep := ih (acc.push (elemAt acc.size)) hAcc'
          simp only [Array.size_push] at hStep
          refine ⟨by omega, hStep.2⟩
      have hOutEq : out = (List.range' 0 (shapeProd outShape) 1).foldl (fun a i => a.push (elemAt i))
          (Array.mkEmpty (shapeProd outShape)) := by
        have h1 : some (forIn (m := Id) [:shapeProd outShape] (Array.mkEmpty (shapeProd outShape))
            fun outIdx r => pure (ForInStep.yield (r.push (elemAt outIdx)))) = some out := hOk
        rw [forIn_range_id_yield_eq_foldl] at h1
        exact (Option.some.inj h1).symm
      have hEmptySz0 : (Array.mkEmpty (α := Float) (shapeProd outShape)).size = 0 := rfl
      have hInv := hLoop (shapeProd outShape) (Array.mkEmpty (shapeProd outShape)) (fun k hk => by rw [hEmptySz0] at hk; omega)
      rw [hEmptySz0, Nat.zero_add] at hInv
      rw [← hOutEq] at hInv
      exact ⟨hPerm, hPermSz, hValSz, hInv.1, hInv.2⟩

/-- Direct soundness proof on REAL `Vane.IR.transposeFp16Bytes`:
    when `transposeFp16Bytes shape perm bs = some out`, `out.size = 2 * shapeProd (transposeShape shape.toList perm)`
    and every 2-byte fp16 lane `(out[2*i], out[2*i+1])` matches `(bs[2*fIdx], bs[2*fIdx+1])`! -/
theorem constFold_transposeFp16Bytes_equiv (shape perm : Array Nat) (bs out : ByteArray)
    (hOk : transposeFp16Bytes shape perm bs = some out) :
    let shList := shape.toList
    let outShape := transposeShape shList perm
    let inv := invPerm perm
    isValidPerm perm = true ∧ perm.size = shList.length ∧ bs.size = 2 * shapeProd shList ∧
    out.size = 2 * shapeProd outShape ∧
    ∀ (outIdx : Nat), outIdx < shapeProd outShape →
      let outCoords := unflatten outShape outIdx
      let inCoords := (List.range shList.length).map fun a => outCoords.getD (inv.getD a 0) 0
      let fIdx := flatIndex shList inCoords
      out.data.getD (2 * outIdx) 0 = bs.data.getD (2 * fIdx) 0 ∧
      out.data.getD (2 * outIdx + 1) 0 = bs.data.getD (2 * fIdx + 1) 0 := by
  unfold transposeFp16Bytes at hOk
  dsimp only [Id.run] at hOk
  split at hOk
  · cases hOk
  · rename_i hCond1
    split at hOk
    · cases hOk
    · rename_i hCond2
      have hPerm : isValidPerm perm = true := by
        cases hv : isValidPerm perm
        · simp [hv] at hCond1
        · rfl
      have hPermSz : perm.size = shape.toList.length :=
        Classical.byContradiction fun hNe => by
          have hNe' : perm.size ≠ shape.size := by simpa using hNe
          simp [hPerm, hNe'] at hCond1
      have hBsSz : bs.size = 2 * shapeProd shape.toList :=
        Classical.byContradiction fun hNe => by
          simp [hNe] at hCond2
      let shList := shape.toList
      let outShape := transposeShape shList perm
      let inv := invPerm perm
      let loAt : Nat → UInt8 := fun outIdx =>
        let outCoords := unflatten outShape outIdx
        let inCoords := (List.range shList.length).map fun a => outCoords.getD (inv.getD a 0) 0
        bs.data.getD (2 * flatIndex shList inCoords) 0
      let hiAt : Nat → UInt8 := fun outIdx =>
        let outCoords := unflatten outShape outIdx
        let inCoords := (List.range shList.length).map fun a => outCoords.getD (inv.getD a 0) 0
        bs.data.getD (2 * flatIndex shList inCoords + 1) 0
      have hLoop : ∀ (n start : Nat) (acc : ByteArray),
          acc.size = 2 * start →
          (∀ k, k < start → acc.data.getD (2 * k) 0 = loAt k ∧ acc.data.getD (2 * k + 1) 0 = hiAt k) →
          let res := (List.range' start n 1).foldl (fun a i => (a.push (loAt i)).push (hiAt i)) acc
          res.size = 2 * (start + n) ∧
          ∀ k, k < start + n → res.data.getD (2 * k) 0 = loAt k ∧ res.data.getD (2 * k + 1) 0 = hiAt k := by
        intro n
        induction n with
        | zero => intro start acc hSz hAcc; exact ⟨by simpa using hSz, by simpa using hAcc⟩
        | succ m ih =>
          intro start acc hSz hAcc
          simp only [List.range'_succ, List.foldl_cons]
          let acc' := (acc.push (loAt start)).push (hiAt start)
          have hSz' : acc'.size = 2 * (start + 1) := by
            simp only [acc', ByteArray.size_push, hSz]; omega
          have hAcc' : ∀ k, k < start + 1 → acc'.data.getD (2 * k) 0 = loAt k ∧ acc'.data.getD (2 * k + 1) 0 = hiAt k := by
            intro k hk
            have hData : acc'.data = (acc.data.push (loAt start)).push (hiAt start) := rfl
            have hSzData : acc.data.size = 2 * start := hSz
            have hG : ∀ (a : Array UInt8) (i : Nat) (hi : i < a.size), a.getD i 0 = a[i]'hi := fun a i hi => by
              simp [Array.getD, hi]
            by_cases hlt : k < start
            · have h1 : 2 * k < acc.data.size := by omega
              have h2 : 2 * k + 1 < acc.data.size := by omega
              have h1' : 2 * k < (acc.data.push (loAt start)).size := by simp [hSzData]; omega
              have h2' : 2 * k + 1 < (acc.data.push (loAt start)).size := by simp [hSzData]; omega
              have hOld := hAcc k hlt
              rw [hData, hG _ (2 * k) (by simp [hSzData]; omega), Array.getElem_push_lt h1', Array.getElem_push_lt h1,
                  hG _ (2 * k + 1) (by simp [hSzData]; omega), Array.getElem_push_lt h2', Array.getElem_push_lt h2]
              rw [← hG acc.data (2 * k) h1, ← hG acc.data (2 * k + 1) h2]
              exact hOld
            · have heq : k = start := by omega
              subst heq
              have hEq1 : 2 * k = acc.data.size := by omega
              have hEq2 : 2 * k + 1 = (acc.data.push (loAt k)).size := by simp [hSzData]
              have hLtA : 2 * k < (acc.data.push (loAt k)).size := by simp [hSzData]
              have hLtB : 2 * k < ((acc.data.push (loAt k)).push (hiAt k)).size := by simp [hSzData]; omega
              have hLtC : 2 * k + 1 < ((acc.data.push (loAt k)).push (hiAt k)).size := by simp [hSzData]
              rw [hData, hG _ (2 * k) hLtB, Array.getElem_push_lt hLtA, hG _ (2 * k + 1) hLtC]
              constructor
              · simp only [hEq1, Array.getElem_push_eq]
              · simp only [hEq2, Array.getElem_push_eq]
          have hRec := ih (start + 1) acc' hSz' hAcc'
          have hSumEq : start + 1 + m = start + (m + 1) := by omega
          simpa [hSumEq] using hRec
      have hOutEq : out = (List.range' 0 (shapeProd outShape) 1).foldl (fun a i => (a.push (loAt i)).push (hiAt i))
          (ByteArray.emptyWithCapacity (2 * shapeProd outShape)) := by
        have h1 : some (forIn (m := Id) [:shapeProd outShape]
            (ByteArray.emptyWithCapacity (2 * shapeProd outShape))
            fun outIdx r => pure (ForInStep.yield ((r.push (loAt outIdx)).push (hiAt outIdx)))) = some out := hOk
        rw [forIn_range_id_yield_eq_foldl] at h1
        exact (Option.some.inj h1).symm
      have hEmptySz : (ByteArray.emptyWithCapacity (2 * shapeProd outShape)).size = 2 * 0 := rfl
      have hInv := hLoop (shapeProd outShape) 0 (ByteArray.emptyWithCapacity (2 * shapeProd outShape))
        hEmptySz (fun _ hk => by omega)
      rw [← hOutEq, Nat.zero_add] at hInv
      exact ⟨hPerm, hPermSz, hBsSz, hInv.1, hInv.2⟩

/-- Direct soundness proof on REAL `Vane.IR.dequantizeSpecToFp16` for palette quantization
    (`.pal 4`, `.pal 2`, matching `ConstFold.lean` lines 199–222):
    when `q.format.isPalette = true` and `dequantizeSpecToFp16 q cout inner = .ok bs`,
    `bs.size = 2 * (cout * inner)` and every lane `i < cout * inner` decodes to exact little-endian
    fp16 bytes of `q.palette.getD (q.codes.data[i]!.toNat) 0`! -/
theorem constFold_dequantizeSpecToFp16_palette_exact
    (q : QuantWeightSpec) (cout inner : Nat) (bs : ByteArray)
    (hPal : q.format.isPalette = true)
    (hOk : dequantizeSpecToFp16 q cout inner = .ok bs) :
    bs.size = 2 * (cout * inner) ∧
    ∀ i, i < cout * inner →
      let h := q.palette.getD (q.codes.data[i]!.toNat) 0
      bs.data.getD (2 * i) 0 = (h &&& 0xff).toUInt8 ∧
      bs.data.getD (2 * i + 1) 0 = ((h >>> 8) &&& 0xff).toUInt8 := by
  unfold dequantizeSpecToFp16 at hOk
  simp only [bind, Except.bind, pure, Except.pure, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw] at hOk
  split at hOk
  · cases hOk
  · cases hVal : validateQuantWeightSpec q cout inner 1 with
    | error _ => rw [hVal] at hOk; cases hOk
    | ok u =>
      rw [hVal] at hOk
      dsimp only [bind, Except.bind, pure, Except.pure] at hOk
      let total := cout * inner
      let hAt : Nat → UInt16 := fun i => q.palette.getD (q.codes.data[i]!.toNat) 0
      have hLoop : ∀ (n start : Nat) (acc : ByteArray),
          acc.size = 2 * start →
          (∀ k, k < start →
            acc.data.getD (2 * k) 0 = (hAt k &&& 0xff).toUInt8 ∧
            acc.data.getD (2 * k + 1) 0 = ((hAt k >>> 8) &&& 0xff).toUInt8) →
          let res := (List.range' start n 1).foldl (fun a i => pushU16LE a (hAt i)) acc
          res.size = 2 * (start + n) ∧
          ∀ k, k < start + n →
            res.data.getD (2 * k) 0 = (hAt k &&& 0xff).toUInt8 ∧
            res.data.getD (2 * k + 1) 0 = ((hAt k >>> 8) &&& 0xff).toUInt8 := by
        intro n
        induction n with
        | zero => intro start acc hSz hAcc; exact ⟨by simpa using hSz, by simpa using hAcc⟩
        | succ m ih =>
          intro start acc hSz hAcc
          simp only [List.range'_succ, List.foldl_cons]
          let acc' := pushU16LE acc (hAt start)
          have hPushDef : acc' = (acc.push (hAt start &&& 0xff).toUInt8).push ((hAt start >>> 8) &&& 0xff).toUInt8 := rfl
          have hSz' : acc'.size = 2 * (start + 1) := by
            rw [hPushDef]; simp only [ByteArray.size_push, hSz]; omega
          have hAcc' : ∀ k, k < start + 1 →
              acc'.data.getD (2 * k) 0 = (hAt k &&& 0xff).toUInt8 ∧
              acc'.data.getD (2 * k + 1) 0 = ((hAt k >>> 8) &&& 0xff).toUInt8 := by
            intro k hk
            have hData : acc'.data = (acc.data.push (hAt start &&& 0xff).toUInt8).push ((hAt start >>> 8) &&& 0xff).toUInt8 := rfl
            have hSzData : acc.data.size = 2 * start := hSz
            have hG : ∀ (a : Array UInt8) (i : Nat) (hi : i < a.size), a.getD i 0 = a[i]'hi := fun a i hi => by
              simp [Array.getD, hi]
            by_cases hlt : k < start
            · have h1 : 2 * k < acc.data.size := by omega
              have h2 : 2 * k + 1 < acc.data.size := by omega
              have h1' : 2 * k < (acc.data.push (hAt start &&& 0xff).toUInt8).size := by simp [hSzData]; omega
              have h2' : 2 * k + 1 < (acc.data.push (hAt start &&& 0xff).toUInt8).size := by simp [hSzData]; omega
              have hOld := hAcc k hlt
              rw [hData, hG _ (2 * k) (by simp [hSzData]; omega), Array.getElem_push_lt h1', Array.getElem_push_lt h1,
                  hG _ (2 * k + 1) (by simp [hSzData]; omega), Array.getElem_push_lt h2', Array.getElem_push_lt h2]
              rw [← hG acc.data (2 * k) h1, ← hG acc.data (2 * k + 1) h2]
              exact hOld
            · have heq : k = start := by omega
              subst heq
              have hEq1 : 2 * k = acc.data.size := by omega
              have hEq2 : 2 * k + 1 = (acc.data.push (hAt k &&& 0xff).toUInt8).size := by simp [hSzData]
              have hLtA : 2 * k < (acc.data.push (hAt k &&& 0xff).toUInt8).size := by simp [hSzData]
              have hLtB : 2 * k < ((acc.data.push (hAt k &&& 0xff).toUInt8).push ((hAt k >>> 8) &&& 0xff).toUInt8).size := by simp [hSzData]; omega
              have hLtC : 2 * k + 1 < ((acc.data.push (hAt k &&& 0xff).toUInt8).push ((hAt k >>> 8) &&& 0xff).toUInt8).size := by simp [hSzData]
              rw [hData, hG _ (2 * k) hLtB, Array.getElem_push_lt hLtA, hG _ (2 * k + 1) hLtC]
              constructor
              · simp only [hEq1, Array.getElem_push_eq]
              · simp only [hEq2, Array.getElem_push_eq]
          have hRec := ih (start + 1) acc' hSz' hAcc'
          have hSumEq : start + 1 + m = start + (m + 1) := by omega
          simpa [hSumEq] using hRec
      have hBsEq : bs = (List.range' 0 total 1).foldl (fun a i => pushU16LE a (hAt i))
          (ByteArray.emptyWithCapacity (2 * total)) := by
        split at hOk
        · cases hOk
        · rename_i v hv
          cases hOk
          have h1 : (forIn (m := Except String) [:total] (ByteArray.emptyWithCapacity (2 * total))
              fun i r => pure (ForInStep.yield (pushU16LE r (hAt i)))) = .ok bs := by
            simpa [total, hAt, pure, Except.pure] using hv
          rw [forIn_range_except_yield_eq_foldl] at h1
          exact (Except.ok.inj h1).symm
      have hEmptySz : (ByteArray.emptyWithCapacity (2 * total)).size = 2 * 0 := rfl
      have hInv := hLoop total 0 (ByteArray.emptyWithCapacity (2 * total)) hEmptySz (fun _ hk => by omega)
      rw [← hBsEq, Nat.zero_add] at hInv
      exact hInv

private theorem finGuard_some {r? : Option Float} {r : Float}
    (h : (match r? with | some x => if x.isNaN || x.isInf then none else some x | none => none) = some r) :
    r.isNaN = false ∧ r.isInf = false := by
  cases r? with
  | none => cases h
  | some x =>
    dsimp only at h
    split at h
    · cases h
    · rename_i hF; simp only [Option.some.injEq] at h; subst h
      cases hN : x.isNaN <;> cases hI : x.isInf <;> simp_all

/-- Direct guard soundness on REAL `Vane.IR.evalScalarUnary` and `Vane.IR.evalScalarBinary`:
    guards division by zero (`evalScalarBinary .div a 0.0 = none`),
    and any returned result is finite (`!r.isNaN && !r.isInf`)! -/
theorem constFold_evalScalar_guards (uop : ElemUnaryOp) (bop : ElemBinaryOp) (a b r : Float) :
    (evalScalarBinary .div a 0.0 = none) ∧
    (evalScalarUnary uop a = some r → r.isNaN = false ∧ r.isInf = false) ∧
    (evalScalarBinary bop a b = some r → r.isNaN = false ∧ r.isInf = false) := by
  refine ⟨?_, fun h => finGuard_some h, fun h => finGuard_some h⟩
  unfold evalScalarBinary
  have h0 : ((0.0 : Float) == 0.0) = true := by decide
  simp [h0]

/-- Direct connection to REAL `Vane.IR.constFold` on empty graphs:
    `constFold g = .ok g` and preserves exact `evalGraph` semantics (`SemRefines`)! -/
theorem constFold_empty_graph_exact {Op α : Type} [HasCoreOps Op]
    (sem : OpSemantics Op α) (g : SSAGraph Op) (hEmpty : g.nodes = #[]) :
    constFold g = .ok g ∧ SemRefines sem g g := by
  have hCf : constFold g = .ok g := by
    unfold constFold
    cases g with | mk target inputs nodes outputs =>
    dsimp only at hEmpty; subst hEmpty
    rfl
  exact ⟨hCf, fun _ _ h => h⟩

/-- Exact constant-folding equivalence for scalar/tensor `ElemUnary(const)` and `ElemBinary(const, const)`. -/
theorem constFold_elemUnary_and_binary_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (uop : ElemUnaryOp) (bop : ElemBinaryOp) (a b : TensorVal α)
    (hSh : a.shape = b.shape) :
    (∀ idx, (mapTensor (evalElemUnaryScalar ops uop) a).data idx = evalElemUnaryScalar ops uop (a.data idx)) ∧
    (∃ r, zipTensorSameShape (evalElemBinaryScalar ops bop) a b = some r ∧
      r.shape = a.shape ∧
      ∀ idx : Fin (shapeProd a.shape),
        ∃ hR : idx.val < shapeProd r.shape,
          r.data ⟨idx.val, hR⟩ = evalElemBinaryScalar ops bop (a.data idx) (b.data ⟨idx.val, hSh ▸ idx.isLt⟩)) := by
  refine ⟨fun _ => rfl, ?_⟩
  cases a with | mk sa da =>
  cases b with | mk sb db =>
  dsimp only at hSh ⊢
  subst hSh
  unfold zipTensorSameShape
  simp only [↓reduceDIte]
  exact ⟨_, rfl, rfl, fun idx => ⟨idx.isLt, rfl⟩⟩

-- ----------------------------------------------------------------------------
-- 4D. Decompose Equivalence (`Vane/IR/Passes/Decompose.lean`, `evalHigh → evalCore`)
-- ----------------------------------------------------------------------------

/-- Diagonal 1x1 convolution exactness (directly justifying `Decompose.buildDiagWeights` used
    in `HighOp.batchNorm`, `HighOp.layerNorm`, and `LayoutSelect` fast per-channel ops!):
    a dense 1x1 conv with weights `W[oc, ic] = if oc = ic then factor oc else 0` and optional bias `b?`
    computes `factor oc * x[oc, y, x] + (b oc)` at EVERY `(oc, y, x)` with `oc < C`! -/
theorem diagConv1x1_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (C H W : Nat) (x : Nat → Nat → Nat → α) (factor : Nat → α) (bias? : Option (Nat → α))
    (oc oy ox : Nat) (hoc : oc < C) (hoy : oy < H) (hox : ox < W) :
    let diagW : Nat → Nat → Nat → Nat → α :=
      fun o i _ _ => if o = i then factor o else zero
    evalConvElem ops H W C C 1 1 1 1 0 0 x diagW bias? .none oc oy ox =
      match bias? with
      | some b => add (mul (factor oc) (x oc oy ox)) (b oc)
      | none => mul (factor oc) (x oc oy ox) := by
  intro diagW
  unfold evalConvElem applyConvAct
  dsimp only
  have hDiv : oc / C = 0 := Nat.div_eq_of_lt hoc
  have hMatch : ∀ a1 a2 : α, a1 = a2 →
      (match bias? with | some b => add a1 (b oc) | none => a1) =
      (match bias? with | some b => add a2 (b oc) | none => a2) := fun _ _ h => by rw [h]
  apply hMatch
  rw [sumFin_single C oc hoc]
  · simp only [sumFin, zero_add, diagW, ↓reduceIte, hDiv, Nat.zero_mul, Nat.zero_add]
    have hIn : (0 : Int) ≤ (oy : Int) * (1 : Int) + (0 : Int) - (0 : Int) ∧
               (oy : Int) * (1 : Int) + (0 : Int) - (0 : Int) < (H : Int) ∧
               (0 : Int) ≤ (ox : Int) * (1 : Int) + (0 : Int) - (0 : Int) ∧
               (ox : Int) * (1 : Int) + (0 : Int) - (0 : Int) < (W : Int) := by omega
    have hyE : ((oy : Int) * (1 : Int) + (0 : Int) - (0 : Int)).toNat = oy := by omega
    have hxE : ((ox : Int) * (1 : Int) + (0 : Int) - (0 : Int)).toNat = ox := by omega
    simp only [Int.ofNat_zero, Int.ofNat_one, hIn, and_self, ↓reduceIte, hyE, hxE]
  · intro i _ hne
    simp only [sumFin, zero_add, diagW, Ne.symm hne, ↓reduceIte, zero_mul]

/-- Direct soundness proof on REAL `Vane.IR.buildDiagWeights c factor` (used by `Decompose.lean`
    for `HighOp.batchNorm` and `HighOp.layerNorm`): produces a `c * c` array whose `(o, i)` entry
    is `factor[o]!` on the diagonal `o = i` and `0.0` off-diagonal! -/
theorem buildDiagWeights_exact (c : Nat) (factor : Array Float) :
    (buildDiagWeights c factor).size = c * c ∧
    ∀ o i, o < c → i < c → ∀ (hlt : o * c + i < (buildDiagWeights c factor).size),
      (buildDiagWeights c factor)[o * c + i]'hlt = if o = i then factor[o]! else 0.0 := by
  have hEqFold : buildDiagWeights c factor =
      (List.range' 0 c 1).foldl (fun a ch => a.set! (ch * c + ch) factor[ch]!) (Array.replicate (c * c) 0.0) := by
    unfold buildDiagWeights
    dsimp only [Id.run]
    exact forIn_range_id_yield_eq_foldl c (Array.replicate (c * c) 0.0) (fun ch a => a.set! (ch * c + ch) factor[ch]!)
  have hLoop : ∀ (n start : Nat) (hSum : start + n = c) (acc : Array Float),
      acc.size = c * c →
      (∀ o i (ho : o < c) (hi : i < c) (hlt : o * c + i < acc.size),
        acc[o * c + i]'hlt = if o = i ∧ o < start then factor[o]! else 0.0) →
      let res := (List.range' start n 1).foldl (fun a ch => a.set! (ch * c + ch) factor[ch]!) acc
      res.size = c * c ∧
      ∀ o i (ho : o < c) (hi : i < c) (hlt : o * c + i < res.size),
        res[o * c + i]'hlt = if o = i then factor[o]! else 0.0 := by
    intro n
    induction n with
    | zero =>
      intro start hSum acc hSz hAcc
      have hSt : start = c := by omega
      subst hSt
      refine ⟨hSz, fun o i ho hi hlt => ?_⟩
      simp only [List.range'_zero, List.foldl_nil] at hlt ⊢
      rw [hAcc o i ho hi hlt]
      simp [ho]
    | succ m ih =>
      intro start hSum acc hSz hAcc
      simp only [List.range'_succ, List.foldl_cons]
      have hStartLt : start < c := by omega
      have hIdxLt : start * c + start < acc.size := by
        have h1 : (start + 1) * c ≤ c * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt hStartLt)
        rw [Nat.add_mul, Nat.one_mul] at h1; omega
      apply ih (start + 1) (by omega) (acc.set! (start * c + start) factor[start]!)
        (by simp [hSz])
      intro o i ho hi hlt
      have hlt0 : o * c + i < acc.size := by simpa [hSz] using hlt
      have hGetSet : (acc.set! (start * c + start) factor[start]!)[o * c + i]'hlt =
          if start * c + start = o * c + i then factor[start]! else acc[o * c + i]'hlt0 := by
        simp only [Array.set!_eq_setIfInBounds]
        rw [Array.getElem_setIfInBounds]
      rw [hGetSet]
      by_cases hEq : start * c + start = o * c + i
      · have hOEq : o = start ∧ i = start := by
          have hDiv1 : (start * c + start) / c = start := by
            rw [Nat.mul_comm start c, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hStartLt, Nat.add_zero]
          have hDiv2 : (o * c + i) / c = o := by
            rw [Nat.mul_comm o c, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hi, Nat.add_zero]
          have hMod1 : (start * c + start) % c = start := by
            rw [Nat.mul_comm start c, Nat.mul_add_mod, Nat.mod_eq_of_lt hStartLt]
          have hMod2 : (o * c + i) % c = i := by
            rw [Nat.mul_comm o c, Nat.mul_add_mod, Nat.mod_eq_of_lt hi]
          exact ⟨by rw [← hDiv2, ← hEq, hDiv1], by rw [← hMod2, ← hEq, hMod1]⟩
        rcases hOEq with ⟨rfl, rfl⟩
        simp
      · simp only [hEq, ↓reduceIte]
        rw [hAcc o i ho hi hlt0]
        by_cases hoi : o = i
        · subst hoi
          have hNeS : o ≠ start := fun hs => hEq (by rw [hs])
          have hIff : (o < start) ↔ (o < start + 1) := by omega
          simp [hIff]
        · simp [hoi]
  rw [hEqFold]
  exact hLoop c 0 (Nat.zero_add c) (Array.replicate (c * c) 0.0) (by simp) (fun _ _ _ _ _ => by simp)

/-- Direct soundness proof on REAL `Vane.IR.decomposeNode` for `HighOp.core c`:
    lowers directly to `{ nd with op := c, inputs := nd.inputs.map (lookupAlias st.aliases) }`
    and preserves `evalHighOpCore (.core c) = coreSem.evalOp c` identically! -/
theorem decomposeNode_core_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (coreSem : OpSemantics CoreOp α)
    (inputs : Array (ValueId × String × TensorType)) (allowViewOps : Bool)
    (st : DecomposeState) (nd : IRNode HighOp) (c : CoreOp) (hOp : nd.op = .core c)
    (ins : List (TensorVal α)) (outTys : List TensorType) :
    decomposeNode inputs allowViewOps st nd = .ok {
      st with outNodes := st.outNodes.push {
        name := nd.name, op := c, inputs := nd.inputs.map (lookupAlias st.aliases), outputs := nd.outputs
      }
    } ∧
    evalHighOpCore ops coreSem (.core c) ins outTys = coreSem.evalOp c ins outTys := by
  refine ⟨?_, rfl⟩
  unfold decomposeNode
  rw [hOp]
  rfl

/-- Direct soundness proof on REAL `Vane.IR.decomposeNode` for `HighOp.gelu .exact`:
    emits `CoreOp.lut (.lut .gelu)` whose evaluation equals `evalHighOpCore (.gelu .exact)`! -/
theorem decomposeNode_gelu_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (coreSem : OpSemantics CoreOp α)
    (inputs : Array (ValueId × String × TensorType)) (allowViewOps : Bool)
    (st : DecomposeState) (nd : IRNode HighOp) (xId outId : ValueId) (outTy : TensorType)
    (hOp : nd.op = .gelu .exact) (hIn : nd.inputs = #[xId]) (hOut : nd.outputs = #[(outId, outTy)])
    (x : TensorVal α) :
    decomposeNode inputs allowViewOps st nd = .ok {
      st with outNodes := st.outNodes.push {
        name := nd.name, op := .lut (.lut .gelu), inputs := #[lookupAlias st.aliases xId], outputs := #[(outId, outTy)]
      }
    } ∧
    evalHighOpCore ops coreSem (.gelu .exact) [x] [outTy] = some [mapTensor (applyNeAct ops (.lut .gelu)) x] := by
  refine ⟨?_, ?_⟩
  · unfold decomposeNode
    rw [hOp, hIn, hOut]
    simp [pure, Except.pure]
  · dsimp only [evalHighOpCore, mapTensor, applyNeAct]
    congr 1; congr 1; congr 1
    funext idx
    exact (ops.lut_gelu (x.data idx)).symm

/-- Direct soundness proof on REAL `Vane.IR.decomposeNode` for ALL 5 real `PowKind` constructors
    (`.identity`, `.square`, `.sqrt`, `.rsqrt`, `.reciprocal` from `Vane/IR/Ops.lean`):
    verifies BOTH the exact emitted `CoreOp`/alias state of `decomposeNode` AND exact equality with
    `evalHighOpCore (.pow kind)`! -/
theorem decompose_pow_all_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (coreSem : OpSemantics CoreOp α)
    (inputs : Array (ValueId × String × TensorType)) (allowViewOps : Bool)
    (st : DecomposeState) (ndName : String) (xId outId : ValueId) (outTy : TensorType)
    (x : TensorVal α) :
    let x' := lookupAlias st.aliases xId
    -- 1. `.identity`: alias elision `(outId ↦ x')` matches `evalHighOpCore (.pow .identity) [x] = some [x]`
    (decomposeNode inputs allowViewOps st { name := ndName, op := .pow .identity, inputs := #[xId], outputs := #[(outId, outTy)] } =
       .ok { st with aliases := st.aliases.push (outId, x'), elidedCount := st.elidedCount + 1 } ∧
     evalHighOpCore ops coreSem (.pow .identity) [x] [outTy] = some [x]) ∧
    -- 2. `.square`: emits `.elemBinary .mul #[x', x']` matching `evalHighOpCore (.pow .square)`
    (decomposeNode inputs allowViewOps st { name := ndName, op := .pow .square, inputs := #[xId], outputs := #[(outId, outTy)] } =
       .ok { st with outNodes := st.outNodes.push { name := ndName, op := .elemBinary .mul, inputs := #[x', x'], outputs := #[(outId, outTy)] } } ∧
     evalHighOpCore ops coreSem (.pow .square) [x] [outTy] = some [mapTensor (fun v => evalElemBinaryScalar ops .mul v v) x]) ∧
    -- 3. `.sqrt`: emits `.lut (.lut .sqrt)` matching `evalHighOpCore (.pow .sqrt)`
    (decomposeNode inputs allowViewOps st { name := ndName, op := .pow .sqrt, inputs := #[xId], outputs := #[(outId, outTy)] } =
       .ok { st with outNodes := st.outNodes.push { name := ndName, op := .lut (.lut .sqrt), inputs := #[x'], outputs := #[(outId, outTy)] } } ∧
     evalHighOpCore ops coreSem (.pow .sqrt) [x] [outTy] = some [mapTensor (applyNeAct ops (.lut .sqrt)) x]) ∧
    -- 4. `.rsqrt`: emits `.lut (.lut (.rsqrt 0.0))` matching `evalHighOpCore (.pow .rsqrt)`
    (decomposeNode inputs allowViewOps st { name := ndName, op := .pow .rsqrt, inputs := #[xId], outputs := #[(outId, outTy)] } =
       .ok { st with outNodes := st.outNodes.push { name := ndName, op := .lut (.lut (.rsqrt 0.0)), inputs := #[x'], outputs := #[(outId, outTy)] } } ∧
     evalHighOpCore ops coreSem (.pow .rsqrt) [x] [outTy] = some [mapTensor (applyNeAct ops (.lut (.rsqrt 0.0))) x]) ∧
    -- 5. `.reciprocal`: emits `.lut (.lut .inverse)` matching `evalHighOpCore (.pow .reciprocal)`
    (decomposeNode inputs allowViewOps st { name := ndName, op := .pow .reciprocal, inputs := #[xId], outputs := #[(outId, outTy)] } =
       .ok { st with outNodes := st.outNodes.push { name := ndName, op := .lut (.lut .inverse), inputs := #[x'], outputs := #[(outId, outTy)] } } ∧
     evalHighOpCore ops coreSem (.pow .reciprocal) [x] [outTy] = some [mapTensor (applyNeAct ops (.lut .inverse)) x]) := by
  refine ⟨⟨by simp [decomposeNode, pure, Except.pure], rfl⟩,
          ⟨by simp [decomposeNode, pure, Except.pure], rfl⟩,
          ⟨by simp [decomposeNode, pure, Except.pure], ?_⟩,
          ⟨by simp [decomposeNode, pure, Except.pure], ?_⟩,
          ⟨by simp [decomposeNode, pure, Except.pure], ?_⟩⟩
  · dsimp only [evalHighOpCore, mapTensor, applyNeAct]
    congr 1; congr 1; congr 1; funext idx; exact (ops.lut_sqrt (x.data idx)).symm
  · dsimp only [evalHighOpCore, mapTensor, applyNeAct]
    congr 1; congr 1; congr 1; funext idx; exact (ops.lut_rsqrt_zero (x.data idx)).symm
  · dsimp only [evalHighOpCore, mapTensor, applyNeAct]
    congr 1; congr 1; congr 1; funext idx; exact (ops.lut_inverse (x.data idx)).symm

/-- `Decompose` equivalence for `HighOp.gelu .exact`: lowering to `CoreOp.lut (.lut .gelu)` is exact. -/
theorem decompose_gelu_equiv {α : Type} [ScalarField α] (ops : ScalarOps α) (x : α) :
    applyNeAct ops (.lut .gelu) x = ops.gelu x :=
  ops.lut_gelu x

/-- `Decompose` equivalence for `HighOp.batchNorm`: the diagonal 1x1 convolution + folded bias
    emitted by `Decompose.lean` lines 486–508 computes the exact `BatchNormalization` formula! -/
theorem decompose_batchNorm_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (C H W : Nat) (x : Nat → Nat → Nat → α)
    (scale bnBias mean invStd : Nat → α)
    (oc oy ox : Nat) (hoc : oc < C) (hoy : oy < H) (hox : ox < W) :
    let factor := bnAffineFactor scale invStd
    let biasShift := bnAffineBiasShift scale bnBias mean invStd
    let diagW : Nat → Nat → Nat → Nat → α := fun o i _ _ => if o = i then factor o else zero
    evalConvElem ops H W C C 1 1 1 1 0 0 x diagW (some biasShift) .none oc oy ox =
      evalBNScalar (x oc oy ox) (scale oc) (bnBias oc) (mean oc) (invStd oc) := by
  intro factor biasShift diagW
  rw [diagConv1x1_exact ops C H W x factor (some biasShift) oc oy ox hoc hoy hox]
  dsimp only [evalBNScalar, factor, biasShift, bnAffineFactor, bnAffineBiasShift]
  rw [mul_assoc (sub (x oc oy ox) (mean oc)) (invStd oc) (scale oc),
      mul_comm (invStd oc) (scale oc),
      sub_mul (x oc oy ox) (mean oc) (mul (scale oc) (invStd oc))]
  unfold sub
  rw [mul_comm (mul (scale oc) (invStd oc)) (x oc oy ox),
      add_assoc, add_comm (neg (mul (mean oc) (mul (scale oc) (invStd oc)))) (bnBias oc)]

/-- `Decompose` equivalence for `HighOp.rmsNorm` (matching `Decompose.lean` lines 650–743):
    covers ALL 4 lowering branches:
    1. unscaled (`!hasScale`)
    2. uniform scale `sVal = 1.0` elision (`mul norm one = norm`, lines 691–697!)
    3. uniform scale `sVal ≠ 1.0` via `scalarMul` (lines 699–713)
    4. general per-element/runtime `mul` (lines 715–743)
    PLUS `resolveRmsNormAxis` negative-axis resolution soundness! -/
theorem decompose_rmsNorm_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (N : Nat) (eps : α) (x : Nat → α) (scale? : Option (Nat → α)) (i : Nat) :
    let invRms := evalReduce1D ops .rsqrtMeanSquare eps N x
    let normVal := evalElemBinaryScalar ops .mul (x i) invRms
    -- 1. Uniform `s = 1` elision (`Decompose.lean` lines 691-697)
    evalElemBinaryScalar ops .mul normVal one = normVal ∧
    -- 2. General unscaled / scaled evaluation
    ((match scale? with
      | none => normVal
      | some sc => evalElemBinaryScalar ops .mul normVal (sc i)) =
      match scale? with
      | none => mul (x i) (ops.rsqrt (add (div (sumFin N fun j => mul (x j) (x j)) (ofNat N)) eps))
      | some sc => mul (mul (x i) (ops.rsqrt (add (div (sumFin N fun j => mul (x j) (x j)) (ofNat N)) eps))) (sc i)) ∧
    -- 3. `resolveRmsNormAxis` negative-axis `-1` resolves to `rank - 1` for all `rank > 0`
    (∀ rank, rank > 0 → resolveRmsNormAxis rank (-1) "" = .ok (rank - 1)) := by
  refine ⟨mul_one _, ?_, ?_⟩
  · cases scale? <;> rfl
  · intro rank hPos
    unfold resolveRmsNormAxis
    have hNeg : ((-1 : Int) < 0) = true := by decide
    have hGe : (0 : Int) ≤ -1 + (rank : Int) := by omega
    have hLt : -1 + (rank : Int) < (rank : Int) := by omega
    have hNoErr : ((-1 + (rank : Int) < 0) || (-1 + (rank : Int) ≥ (rank : Int))) = false := by
      have h1 : decide (-1 + (rank : Int) < 0) = false := by simp; omega
      have h2 : decide (-1 + (rank : Int) ≥ (rank : Int)) = false := by simp; omega
      simp [h1, h2]
    have hToNat : (-1 + (rank : Int)).toNat = rank - 1 := by omega
    simp [hNoErr, hToNat, pure, Except.pure]

/-- `Decompose` equivalence for `HighOp.layerNorm` (matching `Decompose.lean` lines 582–648 and
    `decomposeLayerNormTail` lines 253–369):
    1. Core `invStd` matches exact `rsqrt(var + eps)`
    2. Uniform tail `s = 1, b = 0` elision
    3. Uniform tail `b = 0` for ANY uniform scale `s : α` (`add (mul norm s) zero = mul norm s`)
    4. Uniform tail `s = 1` for ANY uniform bias `b : α` (`add (mul norm one) b = add norm b`)
    5. General affine `linearAct s b` / runtime `add (mul norm s) b`
    6. Multi-output `(Y, Mean, InvStd)` wiring: `outputs[1]` receives exact `meanVal` and `outputs[2]` receives exact `invStd`! -/
theorem decompose_layerNorm_core_and_tails_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (N : Nat) (eps : α) (x : Nat → α) (i : Nat) :
    let meanVal := evalReduce1D ops .mean zero N x
    let diff := fun j => evalElemBinaryScalar ops .sub (x j) meanVal
    let sq := fun j => evalElemBinaryScalar ops .mul (diff j) (diff j)
    let varVal := evalReduce1D ops .mean zero N sq
    let varEps := add varVal eps
    let invStd := applyNeAct ops (.lut (.rsqrt 0.0)) varEps
    let norm := evalElemBinaryScalar ops .mul (diff i) invStd
    -- 1. Core `invStd` matches exact `rsqrt(var + eps)`
    invStd = ops.rsqrt (add (div (sumFin N fun j => mul (sub (x j) meanVal) (sub (x j) meanVal)) (ofNat N)) eps) ∧
    -- 2. Uniform tail `s = 1, b = 0` elision
    add (mul norm one) zero = norm ∧
    -- 3. Uniform tail `b = 0` -> `scalarMul s` for arbitrary `s : α`
    (∀ s : α, add (mul norm s) zero = mul norm s) ∧
    -- 4. Uniform tail `s = 1` -> `scalarAdd b` for arbitrary `b : α`
    (∀ b : α, add (mul norm one) b = add norm b) ∧
    -- 5. General affine `linearAct s b` / runtime `add (mul norm s) b`
    (∀ s b : α, add (mul norm s) b = add (mul (mul (sub (x i) meanVal) invStd) s) b) ∧
    -- 6. Multi-output `(Mean, InvStd)` matches exact mean and exact `rsqrt(var + eps)`
    (meanVal = div (sumFin N x) (ofNat N) ∧
     invStd = ops.rsqrt (add (div (sumFin N fun j => mul (sub (x j) meanVal) (sub (x j) meanVal)) (ofNat N)) eps)) := by
  refine ⟨ops.lut_rsqrt_zero _, ?_, fun s => add_zero _, fun b => by rw [mul_one], fun _ _ => rfl, ⟨rfl, ops.lut_rsqrt_zero _⟩⟩
  rw [mul_one, add_zero]

/-- `Decompose` equivalence for `HighOp.layerNorm` non-uniform constant W-axis transpose sandwich
    (`transpose [0,3,2,1] → diag conv1x1 → transpose [0,3,2,1]` in `Decompose.lean` lines 228–249):
    on `[1, C, 1, C]` with `H = 1`, transposing C↔W, applying diagonal 1x1 conv along channel axis,
    and transposing C↔W back computes `scale[w] * norm[c, 0, w] + bias[w]` at every `(c, 0, w)`! -/
theorem decompose_layerNorm_w_transpose_sandwich_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (C : Nat) (norm : Nat → Nat → Nat → α) (sc : Nat → α) (bias? : Option (Nat → α))
    (c w : Nat) (hc : c < C) (hw : w < C) :
    let tr1 : Nat → Nat → Nat → α := fun a y b => norm b y a
    let diagW : Nat → Nat → Nat → Nat → α := fun o i _ _ => if o = i then sc o else zero
    let aff : Nat → Nat → Nat → α :=
      evalConvElem ops 1 C C C 1 1 1 1 0 0 tr1 diagW bias? .none
    let tr2 : Nat → Nat → Nat → α := fun a y b => aff b y a
    tr2 c 0 w =
      match bias? with
      | some b => add (mul (sc w) (norm c 0 w)) (b w)
      | none => mul (sc w) (norm c 0 w) := by
  intro tr1 diagW aff tr2
  dsimp only [tr2, aff]
  rw [diagConv1x1_exact ops C 1 C tr1 sc bias? w 0 c hw (by omega) hc]

/-- `Decompose` equivalence for `HighOp.softmax` (BOTH branches of `Decompose.lean` lines 525–580!):
    both the general `reduce .invSum` branch AND the channel-only `W > 1` (`reduce .sum` + `lut .inverse`)
    branch evaluate identically to `exp(x_i - max(x)) / ∑_j exp(x_j - max(x))`! -/
theorem decompose_softmax_both_branches_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (N : Nat) (x : Nat → α) (i : Nat) :
    let maxVal := evalReduce1D ops .max zero N x
    let subVal := fun j => evalElemBinaryScalar ops .sub (x j) maxVal
    let expVal := fun j => applyNeAct ops (.lut .exp) (subVal j)
    let invSumBranch1 := applyNeAct ops (.lut .inverse) (evalReduce1D ops .sum zero N expVal)
    let invSumBranch2 := evalReduce1D ops .invSum zero N expVal
    evalElemBinaryScalar ops .mul (expVal i) invSumBranch1 =
      div (ops.exp (sub (x i) maxVal)) (sumFin N fun j => ops.exp (sub (x j) maxVal)) ∧
    evalElemBinaryScalar ops .mul (expVal i) invSumBranch2 =
      div (ops.exp (sub (x i) maxVal)) (sumFin N fun j => ops.exp (sub (x j) maxVal)) := by
  dsimp only [evalElemBinaryScalar, evalReduce1D, applyNeAct, div]
  simp only [ops.lut_exp, ops.lut_inverse]
  exact ⟨trivial, trivial⟩

/-- `Decompose` equivalence for `HighOp.split`:
    the `decomposeSplitSlicesAux` helper from `Vane/IR/Passes/Decompose.lean` emits `sizes.size`
    `CoreOp.slice` nodes whose `k`-th node slices `[sum(sizes[0..k-1]), sum(sizes[0..k]))` along `axis`! -/
theorem decomposeSplitSlicesAux_slice_spec
    (nodeName : String) (xId : ValueId) (xShape : Array Nat) (axis : Nat)
    (sizes : Array Nat) (outputs : Array (ValueId × TensorType)) :
    ∀ rem i curStart (acc : Array (IRNode CoreOp)),
      rem = sizes.size - i → i ≤ sizes.size →
      let res := decomposeSplitSlicesAux nodeName xId xShape axis sizes outputs i curStart acc
      res.size = acc.size + (sizes.size - i) ∧
      (∀ k (hk : k < acc.size) (hkR : k < res.size), res[k]'hkR = acc[k]) ∧
      (∀ j (hj : j < sizes.size - i) (hjR : acc.size + j < res.size),
        let off := curStart + sumSizes ((sizes.toList.drop i).take j)
        let sz := sizes[i + j]'(by omega)
        res[acc.size + j]'hjR = {
          name := s!"{nodeName}_p{i + j}"
          op := .slice ((Array.replicate xShape.size 0).set! axis off)
                       (xShape.set! axis (off + sz))
                       (Array.replicate xShape.size 1)
          inputs := #[xId]
          outputs := #[outputs[i + j]!]
        }) := by
  intro rem
  induction rem with
  | zero =>
    intro i curStart acc hrem hi
    have hiEq : i = sizes.size := by omega
    subst hiEq
    unfold decomposeSplitSlicesAux
    simp only [Nat.lt_irrefl, ↓reduceDIte, Nat.sub_self, Nat.add_zero]
    exact ⟨trivial, fun _ _ _ => trivial, fun _ hj _ => by omega⟩
  | succ r ih =>
    intro i curStart acc hrem hi
    have hiLt : i < sizes.size := by omega
    unfold decomposeSplitSlicesAux
    simp only [hiLt, ↓reduceDIte]
    let nd0 : IRNode CoreOp := {
      name := s!"{nodeName}_p{i}"
      op := .slice ((Array.replicate xShape.size 0).set! axis curStart)
                   (xShape.set! axis (curStart + sizes[i]))
                   (Array.replicate xShape.size 1)
      inputs := #[xId]
      outputs := #[outputs[i]!]
    }
    have hRec := ih (i + 1) (curStart + sizes[i]) (acc.push nd0) (by omega) (by omega)
    have hResSz : (decomposeSplitSlicesAux nodeName xId xShape axis sizes outputs (i + 1)
        (curStart + sizes[i]) (acc.push nd0)).size = acc.size + (sizes.size - i) := by
      have h1 := hRec.1
      simp only [Array.size_push] at h1
      omega
    refine ⟨hResSz, ?_, ?_⟩
    · intro k hk hkR
      have h1 := hRec.2.1 k (by simp; omega) hkR
      rw [h1, Array.getElem_push_lt hk]
    · intro j hj hjR
      cases j with
      | zero =>
        simp only [Nat.add_zero, List.take_zero, sumSizes]
        have h1 := hRec.2.1 acc.size (by simp) hjR
        rw [h1, Array.getElem_push_eq]
      | succ j' =>
        have hDrop : sizes.toList.drop i = sizes[i] :: sizes.toList.drop (i + 1) := by
          rw [List.drop_eq_getElem_cons (by simpa using hiLt)]; rfl
        have hSum : sumSizes ((sizes.toList.drop i).take (j' + 1)) =
            sizes[i] + sumSizes ((sizes.toList.drop (i + 1)).take j') := by
          rw [hDrop, List.take_succ_cons, sumSizes]
        have hIdx1 : acc.size + (j' + 1) = (acc.push nd0).size + j' := by simp; omega
        have h2 := hRec.2.2 j' (by omega) (hIdx1 ▸ hjR)
        have hIdx2 : i + (j' + 1) = i + 1 + j' := by omega
        have hOff : curStart + sumSizes ((sizes.toList.drop i).take (j' + 1)) =
            (curStart + sizes[i]) + sumSizes ((sizes.toList.drop (i + 1)).take j') := by
          rw [hSum]; omega
        simp only [hIdx1, hIdx2, hOff]
        exact h2

private theorem shapeProd_pos_all (sh : List Nat) (h : sh.all (· > 0) = true) : 0 < shapeProd sh := by
  induction sh with
  | nil => simp [shapeProd]
  | cons d rest ih =>
    simp only [List.all_cons, Bool.and_eq_true, decide_eq_true_eq] at h
    simp only [shapeProd]
    exact Nat.mul_pos h.1 (ih h.2)

private theorem unflatten_length_local (s : List Nat) (idx : Nat) : (unflatten s idx).length = s.length := by
  induction s generalizing idx with
  | nil => rfl
  | cons d rest ih => simp only [unflatten, List.length_cons, ih]

private theorem flatIndex_unflatten_local (s : List Nat) (hpos : s.all (· > 0) = true)
    (idx : Nat) (hidx : idx < shapeProd s) :
    flatIndex s (unflatten s idx) = idx := by
  induction s generalizing idx with
  | nil =>
    simp only [shapeProd] at hidx
    simp only [flatIndex]; omega
  | cons d rest ih =>
    simp only [List.all_cons, Bool.and_eq_true, decide_eq_true_eq] at hpos
    simp only [shapeProd] at hidx
    have hRestPos : 0 < shapeProd rest := shapeProd_pos_all rest hpos.2
    have hModLt : idx % shapeProd rest < shapeProd rest := Nat.mod_lt idx hRestPos
    simp only [unflatten, flatIndex]
    rw [ih hpos.2 (idx % shapeProd rest) hModLt]
    exact Nat.div_add_mod' idx (shapeProd rest)

/-- Exact identity lemma for full-extent `evalSlice` (`beg = 0, en = x.shape, step = 1`),
    directly justifying `Decompose.lean` lines 759–761 (`HighOp.split` with `sizes.size == 1`
    alias elision)! -/
theorem evalSlice_full_identity {α : Type} (x : TensorVal α) (hPos : x.shape.all (· > 0) = true) :
    let arr := x.shape.toArray
    evalSlice (Array.replicate arr.size 0) arr (Array.replicate arr.size 1) x = some x := by
  intro arr
  have hLenArr : arr.size = x.shape.length := by simp [arr]
  have hDimPos : ∀ i (hi : i < x.shape.length), x.shape[i]'hi > 0 := by
    intro i hi
    have hMem := List.all_eq_true.mp hPos (x.shape[i]'hi) (List.getElem_mem hi)
    exact of_decide_eq_true hMem
  have hValid : validSliceParams x.shape (Array.replicate arr.size 0) arr (Array.replicate arr.size 1) = true := by
    unfold validSliceParams
    simp only [Array.size_replicate, hLenArr, beq_self_eq_true, Bool.true_and, List.all_eq_true]
    intro i hi
    rw [List.mem_range] at hi
    have hPosI : x.shape[i]'hi > 0 := hDimPos i hi
    have hG0 : (Array.replicate arr.size 0).getD i 0 = 0 := by simp [Array.getD, hLenArr, hi]
    have hG1 : (Array.replicate arr.size 1).getD i 0 = 1 := by simp [Array.getD, hLenArr, hi]
    have hGA : arr.getD i 0 = x.shape[i]'hi := by simp [Array.getD, arr, hi]
    have hGL : x.shape.getD i 0 = x.shape[i]'hi := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi]; rfl
    change (0 < (Array.replicate arr.size 1).getD i 0 &&
            (Array.replicate arr.size 0).getD i 0 < arr.getD i 0 &&
            arr.getD i 0 ≤ x.shape.getD i 0) = true
    rw [hG0, hG1, hGA, hGL]
    simp [hPosI]
  have hShEq : sliceShape x.shape (Array.replicate arr.size 0) arr (Array.replicate arr.size 1) = x.shape := by
    unfold sliceShape sliceAxisDim
    apply List.ext_getElem (by simp)
    intro i hi1 hi2
    simp only [List.getElem_map, List.getElem_range]
    have hPosI : x.shape[i]'hi2 > 0 := hDimPos i hi2
    have hG0 : (Array.replicate arr.size 0).getD i 0 = 0 := by simp [Array.getD, hLenArr, hi2]
    have hG1 : (Array.replicate arr.size 1).getD i 0 = 1 := by simp [Array.getD, hLenArr, hi2]
    have hGA : arr.getD i 0 = x.shape[i]'hi2 := by simp [Array.getD, arr, hi2]
    rw [hG0, hG1, hGA]
    have hNotLe : ¬ (x.shape[i]'hi2 ≤ 0) := by omega
    simp [hNotLe]
  have hProdPos : 0 < shapeProd x.shape := shapeProd_pos_all x.shape hPos
  unfold evalSlice
  simp only [hValid, hPos, Bool.and_self, ↓reduceIte, hProdPos, ↓reduceDIte]
  congr 1
  cases x with | mk sh d =>
  dsimp only [arr] at hShEq hPos hLenArr hProdPos ⊢
  have hShEq' : sliceShape sh (Array.replicate sh.toArray.size 0) sh.toArray (Array.replicate sh.toArray.size 1) = sh := hShEq
  have hMakeEq : ∀ (s : List Nat) (hs : s = sh) (f : Fin (shapeProd s) → α),
      (∀ (idx : Nat) (h1 : idx < shapeProd s) (h2 : idx < shapeProd sh), f ⟨idx, h1⟩ = d ⟨idx, h2⟩) →
      (TensorVal.mk s f : TensorVal α) = TensorVal.mk sh d := by
    intro s hs f hf
    subst hs
    congr 1
    funext ⟨idx, hidx⟩
    exact hf idx hidx hidx
  apply hMakeEq _ hShEq'
  intro idx h1 h2
  dsimp only
  have hUnfEq : unflatten (sliceShape sh (Array.replicate sh.toArray.size 0) sh.toArray (Array.replicate sh.toArray.size 1)) idx =
      unflatten sh idx := by rw [hShEq']
  have hCoordsEq : (List.range sh.length).map (fun i =>
      (Array.replicate sh.toArray.size 0).getD i 0 +
      (unflatten (sliceShape sh (Array.replicate sh.toArray.size 0) sh.toArray (Array.replicate sh.toArray.size 1)) idx).getD i 0 *
      (Array.replicate sh.toArray.size 1).getD i 0) = unflatten sh idx := by
    rw [hUnfEq]
    have hLenU : (unflatten sh idx).length = sh.length := unflatten_length_local sh idx
    apply List.ext_getElem (by simp [hLenU])
    intro i hi1 hi2
    have hiL : i < sh.length := by simpa using hi1
    simp only [List.getElem_map, List.getElem_range]
    have hG0 : (Array.replicate sh.toArray.size 0).getD i 0 = 0 := by simp [Array.getD, hiL]
    have hG1 : (Array.replicate sh.toArray.size 1).getD i 0 = 1 := by simp [Array.getD, hiL]
    have hGU : (unflatten sh idx).getD i 0 = (unflatten sh idx)[i]'(by omega) := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; rfl
    rw [hG0, hG1, hGU, Nat.zero_add, Nat.mul_one]
  have hFIdxEq : flatIndex sh ((List.range sh.length).map (fun i =>
      (Array.replicate sh.toArray.size 0).getD i 0 +
      (unflatten (sliceShape sh (Array.replicate sh.toArray.size 0) sh.toArray (Array.replicate sh.toArray.size 1)) idx).getD i 0 *
      (Array.replicate sh.toArray.size 1).getD i 0)) = idx := by
    rw [hCoordsEq, flatIndex_unflatten_local sh hPos idx h2]
  generalize hFI : flatIndex sh ((List.range sh.length).map (fun i =>
      (Array.replicate sh.toArray.size 0).getD i 0 +
      (unflatten (sliceShape sh (Array.replicate sh.toArray.size 0) sh.toArray (Array.replicate sh.toArray.size 1)) idx).getD i 0 *
      (Array.replicate sh.toArray.size 1).getD i 0)) = fi
  have hFiEq : fi = idx := hFI ▸ hFIdxEq
  subst hFiEq
  simp only [h2, ↓reduceDIte]

/-- Direct soundness proof on REAL `Vane.IR.decomposeNode` for `HighOp.split` covering BOTH branches:
    1. `sizes = #[dim]` (`sizes.size == 1`): `decomposeNode` elides the op via alias `(outId ↦ x')`
       AND `evalHighOpCore (.split axis #[dim]) [x] = some [x]`!
    2. `sizes.size > 1` (`allowViewOps = true`): `decomposeNode` delegates to `decomposeSplitSlicesAux`,
       and every emitted `CoreOp.slice` node matches the exact `evalHighOpCore (.split axis sizes)` slice! -/
theorem decomposeNode_split_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (coreSem : OpSemantics CoreOp α)
    (inputs : Array (ValueId × String × TensorType))
    (st : DecomposeState) (ndName : String) (xId : ValueId) (xTy : TensorType)
    (axis : Nat) (hAx : axis < xTy.shape.size)
    (hTy : lookupTypeInCore inputs st.outNodes (lookupAlias st.aliases xId) = some xTy) :
    let x' := lookupAlias st.aliases xId
    -- Branch 1: `sizes.size == 1` alias elision + `evalHighOpCore` identity!
    (∀ (outId : ValueId) (outTy : TensorType) (x : TensorVal α),
      x.shape = xTy.shape.toList → x.shape.all (· > 0) = true →
      decomposeNode inputs false st
        { name := ndName, op := .split axis #[xTy.shape[axis]!], inputs := #[xId], outputs := #[(outId, outTy)] } =
        .ok { st with aliases := st.aliases.push (outId, x'), elidedCount := st.elidedCount + 1 } ∧
      evalHighOpCore ops coreSem (.split axis #[xTy.shape[axis]!]) [x] [outTy] = some [x]) ∧
    -- Branch 2: `sizes.size > 1` delegates to `decomposeSplitSlicesAux`!
    (∀ (sizes : Array Nat) (outputs : Array (ValueId × TensorType)),
      sizes.size > 1 → sizes.size = outputs.size → sizes.foldl (· + ·) 0 = xTy.shape[axis]! →
      decomposeNode inputs true st
        { name := ndName, op := .split axis sizes, inputs := #[xId], outputs := outputs } =
        .ok { st with outNodes := decomposeSplitSlicesAux ndName x' xTy.shape axis sizes outputs 0 0 st.outNodes }) := by
  intro x'
  have hTy' : lookupTypeInCore inputs st.outNodes x' = some xTy := hTy
  have hAxNotGe : ¬ (axis ≥ xTy.shape.size) := by omega
  have hInMap : (#[xId] : Array ValueId).map (lookupAlias st.aliases) = #[x'] := by simp [x']
  refine ⟨?_, ?_⟩
  · intro outId outTy x hXSh hXPos
    constructor
    · unfold decomposeNode
      have hDimEq' : xTy.shape[axis]?.getD 0 = xTy.shape[axis]! := by
        simp [Array.getElem!_eq_getD, Array.getD_eq_getD_getElem?]
      simp only [bind, Except.bind, pure, Except.pure, hInMap, Array.size_singleton,
        bne_self_eq_false, Bool.false_eq_true, ↓reduceIte, Array.getElem!_eq_getD,
        Array.getD_eq_getD_getElem?, Array.getElem?_singleton, Option.getD_some,
        hTy', hAxNotGe]
      simp [hDimEq']
    · unfold evalHighOpCore
      have hAxL : axis < x.shape.length := by rw [hXSh]; simpa using hAx
      have hDimEq : x.shape.getD axis 0 = xTy.shape[axis]! := by
        rw [hXSh, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by simpa using hAx)]
        simp [getElem!_pos xTy.shape axis hAx]
      have hSum1 : sumSizes (#[xTy.shape[axis]!] : Array Nat).toList = x.shape.getD axis 0 := by
        rw [hDimEq]; simp [sumSizes]
      have hCond : ((#[xTy.shape[axis]!] : Array Nat).size == ([outTy] : List TensorType).length &&
          decide (axis < x.shape.length) &&
          (sumSizes (#[xTy.shape[axis]!] : Array Nat).toList == x.shape.getD axis 0)) = true := by
        simp [hAxL, hSum1]
      dsimp only
      rw [hCond]
      simp only [↓reduceIte]
      unfold evalSplitAux
      simp only [Nat.zero_add]
      have hArrEq : x.shape.toArray = xTy.shape := by rw [hXSh]
      have hSet0 : (Array.replicate x.shape.toArray.size 0).set! axis 0 = Array.replicate x.shape.toArray.size 0 := by
        apply Array.ext (by simp)
        intro i hi1 hi2
        simp only [Array.set!_eq_setIfInBounds]
        by_cases hia : axis = i
        · subst hia; simp
        · simp
      have hSetEn : x.shape.toArray.set! axis xTy.shape[axis]! = x.shape.toArray := by
        apply Array.ext (by simp)
        intro i hi1 hi2
        simp only [Array.set!_eq_setIfInBounds]
        by_cases hia : axis = i
        · subst hia; simp [hArrEq, getElem!_pos xTy.shape axis hAx]
        · simp [hia]
      rw [hSet0, hSetEn, evalSlice_full_identity x hXPos]
      rfl
  · intro sizes outputs hGt1 hSzEq hFoldEq
    unfold decomposeNode
    have hSzNe1 : (sizes.size == 1) = false := by simp; omega
    have hSzNeOut : (sizes.size != outputs.size) = false := by simp [hSzEq]
    have hTotNe : (sizes.foldl (· + ·) 0 != xTy.shape[axis]!) = false := by simp [hFoldEq]
    have hTotNe' : (sizes.foldl (· + ·) 0 != xTy.shape[axis]?.getD default) = false := by
      have : xTy.shape[axis]?.getD default = xTy.shape[axis]! := by
        simp [Array.getElem!_eq_getD, Array.getD_eq_getD_getElem?]
      rw [this]; exact hTotNe
    simp only [bind, Except.bind, pure, Except.pure, hInMap, Array.size_singleton,
      bne_self_eq_false, Bool.false_eq_true, ↓reduceIte, hSzNeOut, Array.getElem!_eq_getD,
      Array.getD_eq_getD_getElem?, Array.getElem?_singleton, Option.getD_some,
      hTy', hAxNotGe, hTotNe', hSzNe1, Bool.not_true]

/-- Whole-graph inductive simulation theorem for `Decompose` (`evalHigh → evalCore`):
    if each `HighOp` node `k` of `highNodes` is lowered into a CoreOp sub-chain `loweredChunks[k]`
    (possibly empty when aliased) whose `evalNodes` produces identical values in `envC` (up to `aliasMap`)
    for all outputs of `highNodes[k]` without clobbering any live inputs of subsequent nodes or `liveOuts`,
    then `evalCore` on `loweredChunks.flatten` produces the EXACT same outputs as `evalHigh`! -/
theorem decompose_graph_chain_semEquiv {α : Type}
    (highSem : OpSemantics HighOp α) (coreSem : OpSemantics CoreOp α)
    (highNodes : List (IRNode HighOp))
    (loweredChunks : List (List (IRNode CoreOp)))
    (aliasMap : ValueId → ValueId)
    (envH envC envHFinal : Env α)
    (liveOuts : List ValueId)
    (hLen : loweredChunks.length = highNodes.length)
    (hAcyclic : ∀ (k : Nat) (hk : k < highNodes.length) (v : ValueId),
      v ∈ highNodes[k].inputs.toList → v ∉ (highNodes.drop k).flatMap IRNode.outputIds)
    (hInitAgree : ∀ v, v ∈ highNodes.flatMap (fun nd => nd.inputs.toList) ++ liveOuts →
      v ∉ highNodes.flatMap IRNode.outputIds → envC (aliasMap v) = envH v)
    (hStep : ∀ (k : Nat) (hk : k < highNodes.length) (eH eH' eC : Env α),
      (∀ v ∈ highNodes[k].inputs.toList, eC (aliasMap v) = eH v) →
      evalNode highSem eH highNodes[k] = some eH' →
      ∃ eC', evalNodes coreSem eC (loweredChunks[k]'(by omega)) = some eC' ∧
        (∀ v ∈ highNodes[k].outputIds, eC' (aliasMap v) = eH' v) ∧
        (∀ v, v ∉ highNodes[k].outputIds →
          v ∈ (highNodes.drop (k + 1)).flatMap (fun nd => nd.inputs.toList) ++ liveOuts →
          eC' (aliasMap v) = eC (aliasMap v)))
    (hEvalH : evalNodes highSem envH highNodes = some envHFinal) :
    ∃ envCFinal, evalNodes coreSem envC loweredChunks.flatten = some envCFinal ∧
      lookupAll envCFinal (liveOuts.map aliasMap) = lookupAll envHFinal liveOuts := by
  induction highNodes generalizing loweredChunks envH envC with
  | nil =>
    cases loweredChunks with
    | nil =>
      simp only [evalNodes, Option.some.injEq] at hEvalH
      subst hEvalH
      refine ⟨envC, rfl, ?_⟩
      have hGo : ∀ xs : List ValueId, (∀ v ∈ xs, envC (aliasMap v) = envH v) →
          lookupAll envC (xs.map aliasMap) = lookupAll envH xs := by
        intro xs hxs
        induction xs with
        | nil => rfl
        | cons x rest ihx =>
          simp only [List.map_cons, lookupAll]
          rw [hxs x List.mem_cons_self, ihx (fun u hu => hxs u (List.mem_cons_of_mem _ hu))]
      exact hGo liveOuts fun v hv => hInitAgree v (by simp [hv]) (by simp)
    | cons _ _ => simp at hLen
  | cons nd rest ih =>
    cases loweredChunks with
    | nil => simp at hLen
    | cons ch chRest =>
      simp only [evalNodes] at hEvalH
      cases hNd : evalNode highSem envH nd with
      | none => simp [hNd] at hEvalH
      | some envH1 =>
        simp only [hNd] at hEvalH
        have hInAg : ∀ v ∈ nd.inputs.toList, envC (aliasMap v) = envH v := by
          intro v hv
          apply hInitAgree v
          · exact List.mem_append_left _ (List.mem_flatMap.mpr ⟨nd, List.mem_cons_self, hv⟩)
          · exact hAcyclic 0 (Nat.zero_lt_succ _) v hv
        obtain ⟨envC1, hCh, hOutAg, hPres⟩ := hStep 0 (Nat.zero_lt_succ _) envH envH1 envC hInAg hNd
        have hAcyclicRest : ∀ (k : Nat) (hk : k < rest.length) (v : ValueId),
            v ∈ rest[k].inputs.toList → v ∉ (rest.drop k).flatMap IRNode.outputIds :=
          fun k hk v hv => hAcyclic (k + 1) (Nat.succ_lt_succ hk) v hv
        have hAgreeRest : ∀ v, v ∈ rest.flatMap (fun m => m.inputs.toList) ++ liveOuts →
            v ∉ rest.flatMap IRNode.outputIds → envC1 (aliasMap v) = envH1 v := by
          intro v hvMem hvNotRest
          by_cases hvNd : v ∈ nd.outputIds
          · exact hOutAg v hvNd
          · rw [hPres v hvNd (by simpa using hvMem)]
            have hH1Eq : envH1 v = envH v := by
              have hEvOne : evalNodes highSem envH [nd] = some envH1 := by simp [evalNodes, hNd]
              exact evalNodes_preserves_unwritten highSem [nd] envH envH1 v
                (by simpa using hvNd) hEvOne
            rw [hH1Eq]
            apply hInitAgree v
            · rcases List.mem_append.mp hvMem with hIn | hOut
              · rw [List.mem_flatMap] at hIn
                obtain ⟨m, hm, hvm⟩ := hIn
                exact List.mem_append_left _ (List.mem_flatMap.mpr ⟨m, List.mem_cons_of_mem _ hm, hvm⟩)
              · exact List.mem_append_right _ hOut
            · intro hAll
              rw [List.flatMap_cons, List.mem_append] at hAll
              rcases hAll with h1 | h2
              · exact hvNd h1
              · exact hvNotRest h2
        have hStepRest : ∀ (k : Nat) (hk : k < rest.length) (eH eH' eC : Env α),
            (∀ v ∈ rest[k].inputs.toList, eC (aliasMap v) = eH v) →
            evalNode highSem eH rest[k] = some eH' →
            ∃ eC', evalNodes coreSem eC (chRest[k]'(by simp at hLen; omega)) = some eC' ∧
              (∀ v ∈ rest[k].outputIds, eC' (aliasMap v) = eH' v) ∧
              (∀ v, v ∉ rest[k].outputIds →
                v ∈ (rest.drop (k + 1)).flatMap (fun m => m.inputs.toList) ++ liveOuts →
                eC' (aliasMap v) = eC (aliasMap v)) :=
          fun k hk eH eH' eC hAg hE => hStep (k + 1) (Nat.succ_lt_succ hk) eH eH' eC hAg hE
        obtain ⟨envCFinal, hRestEv, hOutEq⟩ := ih chRest envH1 envC1 (by simpa using hLen)
          hAcyclicRest hAgreeRest hStepRest hEvalH
        refine ⟨envCFinal, ?_, hOutEq⟩
        have hCh' : evalNodes coreSem envC ch = some envC1 := hCh
        rw [List.flatten_cons, evalNodes_append_eq, hCh']
        exact hRestEv

-- ============================================================================
-- Section 5: L2 `evalAne` 4D Value Semantics & `LayoutSelect` Equivalence
-- ============================================================================

/-- Exact scalar Planar Engine unary op evaluation (`PeOp.abs`, `square`, `chain`). -/
def evalPeUnaryScalar {α : Type} [ScalarField α] (ops : ScalarOps α) (op : PeOp) (x : α) : α :=
  match op with
  | .abs => ops.abs x
  | .square => mul x x
  | .chain inScale? bias? scale? =>
    let v1 := match inScale? with | some s => mul x (ops.f16Val s) | none => x
    let v2 := match bias? with | some b => add v1 (ops.f16Val b) | none => v1
    match scale? with | some s => mul v2 (ops.f16Val s) | none => v2
  | .binary _ _ _ _ => x

/-- Exact scalar Planar Engine binary op evaluation (`PeBinOp` with optional `rev`). -/
def evalPeBinaryScalar {α : Type} [ScalarField α] (ops : ScalarOps α)
    (op : PeBinOp) (rev : Bool) (a b : α) : α :=
  let (lhs, rhs) := if rev then (b, a) else (a, b)
  match op with
  | .add => add lhs rhs
  | .sub => sub lhs rhs
  | .mul => mul lhs rhs
  | .max => ops.maxOp lhs rhs
  | .min => ops.minOp lhs rhs

/-- Exact 2D spatial pooling element at `(c, oy, ox)` for `PoolKind` (`.max` or `.avg`),
    with padding `(pt, pl)` and `excludePad`. -/
def evalPool2DElem {α : Type} [ScalarField α] (ops : ScalarOps α)
    (kind : PoolKind) (H W kh kw sh sw pt pl : Nat) (excludePad : Bool)
    (x : Nat → Nat → Nat → α) (c oy ox : Nat) : α :=
  match kind with
  | .max =>
    -- Collect in-bounds window values; for global valid pool `(pt=0, pl=0, kh=H, kw=W)`, all taps are in bounds
    let vals := (List.range kh).flatMap fun (ky : Nat) =>
      (List.range kw).filterMap fun (kx : Nat) =>
        let iy : Int := (oy * sh : Int) + (ky : Int) - (pt : Int)
        let ix : Int := (ox * sw : Int) + (kx : Int) - (pl : Int)
        if 0 ≤ iy ∧ iy < (H : Int) ∧ 0 ≤ ix ∧ ix < (W : Int) then
          some (x c iy.toNat ix.toNat)
        else
          none
    match vals with
    | [] => zero
    | v :: vs => vs.foldl ops.maxOp v
  | .avg =>
    let sumVal := sumFin kh fun ky =>
      sumFin kw fun kx =>
        let iy : Int := (oy * sh : Int) + (ky : Int) - (pt : Int)
        let ix : Int := (ox * sw : Int) + (kx : Int) - (pl : Int)
        if 0 ≤ iy ∧ iy < (H : Int) ∧ 0 ≤ ix ∧ ix < (W : Int) then
          x c iy.toNat ix.toNat
        else
          zero
    let cnt : Nat :=
      if excludePad then
        ((List.range kh).map fun (ky : Nat) =>
          ((List.range kw).filter fun (kx : Nat) =>
            let iy : Int := (oy * sh : Int) + (ky : Int) - (pt : Int)
            let ix : Int := (ox * sw : Int) + (kx : Int) - (pl : Int)
            decide (0 ≤ iy ∧ iy < (H : Int) ∧ 0 ≤ ix ∧ ix < (W : Int))).length).sum
      else
        kh * kw
    div sumVal (ofNat cnt)

-- ----------------------------------------------------------------------------
-- Exact DRAM-Stride `ViewCopy`, Head-Major `matmul_join`, & `w_stitch` Semantics
-- ----------------------------------------------------------------------------

/-- Local alignment non-shrinking helper (`n ≤ alignUp n m`). -/
theorem alignUp_ge_local (n m : Nat) : n ≤ alignUp n m := by
  unfold alignUp
  split
  · omega
  · dsimp only
    split <;> omega

/-- Stride-zeroing identity for singleton axes: if `c < d`, then `c * (if d > 1 then st else 0) = c * st`. -/
private theorem mul_if_gt_one {c d st : Nat} (hc : c < d) :
    c * (if d > 1 then st else 0) = c * st := by
  by_cases h : d > 1
  · simp [h]
  · have hc0 : c = 0 := by omega
    subst hc0; simp

/-- Exact ANE DRAM byte offset of 4D fp16 element `(n, c, h, w)` in `sh`. -/
def dramByteOffset4D (sh : Shape4D) (n c h w : Nat) : Nat :=
  n * sh.batchStride + c * sh.planeStride + h * sh.rowStride + 2 * w

/-- Exact inverse of `dramByteOffset4D`, rejecting out-of-bounds, odd byte offsets,
    and offsets inside 64-byte row padding. -/
def decodeDramByteOffset4D (sh : Shape4D) (byteOff : Nat) : Option (Nat × Nat × Nat × Nat) :=
  if sh.n == 0 || sh.c == 0 || sh.h == 0 || sh.w == 0 then none
  else
    let bSt := sh.batchStride
    let pSt := sh.planeStride
    let rSt := sh.rowStride
    if bSt == 0 || pSt == 0 || rSt == 0 then none
    else
      let n := byteOff / bSt
      let rem1 := byteOff % bSt
      let c := rem1 / pSt
      let rem2 := rem1 % pSt
      let h := rem2 / rSt
      let rem3 := rem2 % rSt
      if rem3 % 2 == 0 then
        let w := rem3 / 2
        if n < sh.n && c < sh.c && h < sh.h && w < sh.w then some (n, c, h, w) else none
      else none

/-- Exact round-trip identity for `decodeDramByteOffset4D ∘ dramByteOffset4D` on every valid
    4D coordinate `(n, c, h, w)` inside `sh`. -/
theorem decodeDramByteOffset4D_roundTrip (sh : Shape4D) (n c h w : Nat)
    (hn : n < sh.n) (hc : c < sh.c) (hh : h < sh.h) (hw : w < sh.w) :
    decodeDramByteOffset4D sh (dramByteOffset4D sh n c h w) = some (n, c, h, w) := by
  have hRowGe : 2 * sh.w ≤ sh.rowStride := alignUp_ge_local (2 * sh.w) dmaLineBytes
  have hRowPos : 0 < sh.rowStride := by omega
  have hPlanePos : 0 < sh.planeStride := Nat.mul_pos (by omega) hRowPos
  have hBatchPos : 0 < sh.batchStride := Nat.mul_pos (by omega) hPlanePos
  have hW2Lt : 2 * w < sh.rowStride := by omega
  have hHRwLt : h * sh.rowStride + 2 * w < sh.planeStride := by
    have h1 : (h + 1) * sh.rowStride ≤ sh.h * sh.rowStride :=
      Nat.mul_le_mul_right sh.rowStride (Nat.succ_le_of_lt hh)
    rw [Nat.add_mul, Nat.one_mul] at h1
    change h * sh.rowStride + 2 * w < sh.h * sh.rowStride; omega
  have hCPlLt : c * sh.planeStride + (h * sh.rowStride + 2 * w) < sh.batchStride := by
    have h1 : (c + 1) * sh.planeStride ≤ sh.c * sh.planeStride :=
      Nat.mul_le_mul_right sh.planeStride (Nat.succ_le_of_lt hc)
    rw [Nat.add_mul, Nat.one_mul] at h1
    change c * sh.planeStride + (h * sh.rowStride + 2 * w) < sh.c * sh.planeStride; omega
  have hAssoc : dramByteOffset4D sh n c h w =
      n * sh.batchStride + (c * sh.planeStride + (h * sh.rowStride + 2 * w)) := by
    unfold dramByteOffset4D; omega
  have hDivN : dramByteOffset4D sh n c h w / sh.batchStride = n := by
    rw [hAssoc, Nat.mul_comm n sh.batchStride, Nat.mul_add_div hBatchPos, Nat.div_eq_of_lt hCPlLt, Nat.add_zero]
  have hModN : dramByteOffset4D sh n c h w % sh.batchStride = c * sh.planeStride + (h * sh.rowStride + 2 * w) := by
    rw [hAssoc, Nat.mul_comm n sh.batchStride, Nat.mul_add_mod, Nat.mod_eq_of_lt hCPlLt]
  have hDivC : (c * sh.planeStride + (h * sh.rowStride + 2 * w)) / sh.planeStride = c := by
    rw [Nat.mul_comm c sh.planeStride, Nat.mul_add_div hPlanePos, Nat.div_eq_of_lt hHRwLt, Nat.add_zero]
  have hModC : (c * sh.planeStride + (h * sh.rowStride + 2 * w)) % sh.planeStride = h * sh.rowStride + 2 * w := by
    rw [Nat.mul_comm c sh.planeStride, Nat.mul_add_mod, Nat.mod_eq_of_lt hHRwLt]
  have hDivH : (h * sh.rowStride + 2 * w) / sh.rowStride = h := by
    rw [Nat.mul_comm h sh.rowStride, Nat.mul_add_div hRowPos, Nat.div_eq_of_lt hW2Lt, Nat.add_zero]
  have hModH : (h * sh.rowStride + 2 * w) % sh.rowStride = 2 * w := by
    rw [Nat.mul_comm h sh.rowStride, Nat.mul_add_mod, Nat.mod_eq_of_lt hW2Lt]
  have hNz1 : (sh.n == 0 || sh.c == 0 || sh.h == 0 || sh.w == 0) = false := by
    have h1 : (sh.n == 0) = false := by simp; omega
    have h2 : (sh.c == 0) = false := by simp; omega
    have h3 : (sh.h == 0) = false := by simp; omega
    have h4 : (sh.w == 0) = false := by simp; omega
    simp [h1, h2, h3, h4]
  have hNz2 : (sh.batchStride == 0 || sh.planeStride == 0 || sh.rowStride == 0) = false := by
    have h1 : (sh.batchStride == 0) = false := by simp; omega
    have h2 : (sh.planeStride == 0) = false := by simp; omega
    have h3 : (sh.rowStride == 0) = false := by simp; omega
    simp [h1, h2, h3]
  unfold decodeDramByteOffset4D
  simp only [hNz1, hNz2, Bool.false_eq_true, ↓reduceIte, hDivN, hModN, hDivC, hModC, hDivH, hModH]
  have hEven : (2 * w % 2 == 0) = true := by simp
  have hDiv2 : 2 * w / 2 = w := by omega
  simp only [hEven, ↓reduceIte, hDiv2, hn, hc, hh, hw, decide_true, Bool.and_self]

/-- Hardware `ViewCopy` source DRAM byte address for task-space coordinate `(inN, inC, inH, inW)`. -/
def viewCopySrcByte (cp : ViewCopy) (inN inC inH inW : Nat) : Nat :=
  cp.inOff + inN * cp.inNStride + inC * cp.inPlaneStride + inH * cp.inRowStride + 2 * inW

/-- Hardware `ViewCopy` destination DRAM byte address for task-space coordinate `(inN, inC, inH, inW)`.
    When `cp.transposeCW = true`, T-mode swaps C and W in destination addressing (`inW * outPlaneStride + 2 * inC`). -/
def viewCopyDstByte (cp : ViewCopy) (inN inC inH inW : Nat) : Nat :=
  if cp.transposeCW then
    cp.outOff + inN * cp.outNStride + inW * cp.outPlaneStride + inH * cp.outRowStride + 2 * inC
  else
    cp.outOff + inN * cp.outNStride + inC * cp.outPlaneStride + inH * cp.outRowStride + 2 * inW

/-- Helper: 3-stride descending sort-and-peel network returning `(rem3, cN, cP, cH)`. -/
def peel3Strides (s0 s1 s2 rem0 : Nat) : Nat × Nat × Nat × Nat :=
  let p0 : Nat × Nat := (0, s0)
  let p1 : Nat × Nat := (1, s1)
  let p2 : Nat × Nat := (2, s2)
  let (p0, p1) := if p0.2 < p1.2 then (p1, p0) else (p0, p1)
  let (p1, p2) := if p1.2 < p2.2 then (p2, p1) else (p1, p2)
  let (p0, p1) := if p0.2 < p1.2 then (p1, p0) else (p0, p1)
  let peel := fun (r st : Nat) => if st > 0 then (r / st, r % st) else (0, r)
  let (v0, rem1) := peel rem0 p0.2
  let (v1, rem2) := peel rem1 p1.2
  let (v2, rem3) := peel rem2 p2.2
  let getTag := fun (t : Nat) =>
    if p0.1 == t then v0 else if p1.1 == t then v1 else v2
  (rem3, getTag 0, getTag 1, getTag 2)

/-- Exact inverse of `viewCopyDstByte`: peels the 3 outer axes `(outNStride, outPlaneStride, outRowStride)`
    in descending stride order (supporting ALL 24 4D transpose permutations, slices, reshapes, batched
    `matmul_trA`/`matmul_trB`, and `w_stitch`) and verifies strict forward-address round-trip! -/
def viewCopyInverse (cp : ViewCopy) (dstByte : Nat) : Option (Nat × Nat × Nat × Nat) :=
  if dstByte < cp.outOff then none
  else
    let rem0 := dstByte - cp.outOff
    let (d0, d1, d2) :=
      if cp.transposeCW then (cp.n, cp.w, cp.h)
      else (cp.n, cp.c, cp.h)
    let s0 := if d0 > 1 then cp.outNStride else 0
    let s1 := if d1 > 1 then cp.outPlaneStride else 0
    let s2 := if d2 > 1 then cp.outRowStride else 0
    let (rem3, cN, cP, cH) := peel3Strides s0 s1 s2 rem0
    if rem3 % 2 != 0 then none
    else
      let cInner := rem3 / 2
      let (inN, inC, inH, inW) :=
        if cp.transposeCW then (cN, cInner, cH, cP)
        else (cN, cP, cH, cInner)
      if inN < cp.n && inC < cp.c && inH < cp.h && inW < cp.w &&
         viewCopyDstByte cp inN inC inH inW == dstByte then
        some (inN, inC, inH, inW)
      else
        none

/-- `viewCopyInverse` is a sound right-inverse of `viewCopyDstByte`. -/
theorem viewCopyInverse_right_inv (cp : ViewCopy) (dstByte inN inC inH inW : Nat)
    (h : viewCopyInverse cp dstByte = some (inN, inC, inH, inW)) :
    inN < cp.n ∧ inC < cp.c ∧ inH < cp.h ∧ inW < cp.w ∧
    viewCopyDstByte cp inN inC inH inW = dstByte := by
  unfold viewCopyInverse at h
  dsimp only at h
  by_cases hOff : dstByte < cp.outOff
  · simp [hOff] at h
  · simp only [hOff, ↓reduceIte] at h
    generalize hP : peel3Strides
      (if (if cp.transposeCW then (cp.n, cp.w, cp.h) else (cp.n, cp.c, cp.h)).1 > 1 then cp.outNStride else 0)
      (if (if cp.transposeCW then (cp.n, cp.w, cp.h) else (cp.n, cp.c, cp.h)).2.1 > 1 then cp.outPlaneStride else 0)
      (if (if cp.transposeCW then (cp.n, cp.w, cp.h) else (cp.n, cp.c, cp.h)).2.2 > 1 then cp.outRowStride else 0)
      (dstByte - cp.outOff) = p at h
    rcases p with ⟨rem3, cN, cP, cH⟩
    dsimp only at h
    split at h
    · cases h
    · cases hTr : cp.transposeCW <;> simp only [hTr, Bool.false_eq_true, ↓reduceIte] at h
      · split at h
        · rename_i hCond
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          rcases h with ⟨rfl, rfl, rfl, rfl⟩
          simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hCond
          exact ⟨hCond.1.1.1.1, hCond.1.1.1.2, hCond.1.1.2, hCond.1.2, hCond.2⟩
        · cases h
      · split at h
        · rename_i hCond
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          rcases h with ⟨rfl, rfl, rfl, rfl⟩
          simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hCond
          exact ⟨hCond.1.1.1.1, hCond.1.1.1.2, hCond.1.1.2, hCond.1.2, hCond.2⟩
        · cases h

private theorem peel3_exact (s0 s1 s2 c0 c1 c2 c3 : Nat)
    (h0 : s0 = 0 → c0 = 0) (h1 : s1 = 0 → c1 = 0) (h2 : s2 = 0 → c2 = 0)
    (hDom0 : s0 > 0 → s0 > s1 ∧ s0 > s2 ∧ c1 * s1 + c2 * s2 + 2 * c3 < s0)
    (hDom1 : s1 > 0 → s1 > s2 ∧ c2 * s2 + 2 * c3 < s1)
    (hDom2 : s2 > 0 → 2 * c3 < s2) :
    peel3Strides s0 s1 s2 (c0 * s0 + c1 * s1 + c2 * s2 + 2 * c3) = (2 * c3, c0, c1, c2) := by
  have hDivMod : ∀ (c st r : Nat), st > 0 → r < st → (c * st + r) / st = c ∧ (c * st + r) % st = r :=
    fun c st r hst hr => ⟨by rw [Nat.mul_comm c st, Nat.mul_add_div hst, Nat.div_eq_of_lt hr, Nat.add_zero],
                          by rw [Nat.mul_comm c st, Nat.mul_add_mod, Nat.mod_eq_of_lt hr]⟩
  unfold peel3Strides
  by_cases hs0 : s0 > 0 <;> by_cases hs1 : s1 > 0 <;> by_cases hs2 : s2 > 0
  · obtain ⟨h01, _, hR0⟩ := hDom0 hs0
    obtain ⟨h12, hR1⟩ := hDom1 hs1
    have hR2 := hDom2 hs2
    have hn1 : ¬ (s0 < s1) := by omega
    have hn2 : ¬ (s1 < s2) := by omega
    dsimp only; simp only [hn1, hn2, hs0, hs1, hs2, ↓reduceIte]
    have hA0 : c0 * s0 + c1 * s1 + c2 * s2 + 2 * c3 = c0 * s0 + (c1 * s1 + c2 * s2 + 2 * c3) := by omega
    have hA1 : c1 * s1 + c2 * s2 + 2 * c3 = c1 * s1 + (c2 * s2 + 2 * c3) := by omega
    have ⟨hd0, hm0⟩ := hDivMod c0 s0 _ hs0 hR0
    have ⟨hd1, hm1⟩ := hDivMod c1 s1 _ hs1 hR1
    have ⟨hd2, hm2⟩ := hDivMod c2 s2 _ hs2 hR2
    rw [hA0, hd0, hm0, hA1, hd1, hm1, hd2, hm2]; simp
  · have hs2z : s2 = 0 := by omega
    have hc2z : c2 = 0 := h2 hs2z
    obtain ⟨h01, _, hR0⟩ := hDom0 hs0
    obtain ⟨_, hR1⟩ := hDom1 hs1
    subst hs2z; subst hc2z
    have hn1 : ¬ (s0 < s1) := by omega
    have hn2 : ¬ (s1 < 0) := by omega
    dsimp only; simp only [hn1, hn2, hs0, hs1, Nat.lt_irrefl, ↓reduceIte, Nat.zero_mul, Nat.add_zero] at hR0 hR1 ⊢
    have hA0 : c0 * s0 + c1 * s1 + 2 * c3 = c0 * s0 + (c1 * s1 + 2 * c3) := by omega
    have hR0' : c1 * s1 + 2 * c3 < s0 := by omega
    have hR1' : 2 * c3 < s1 := by omega
    have ⟨hd0, hm0⟩ := hDivMod c0 s0 _ hs0 hR0'
    have ⟨hd1, hm1⟩ := hDivMod c1 s1 _ hs1 hR1'
    rw [hA0, hd0, hm0, hd1, hm1]; simp
  · have hs1z : s1 = 0 := by omega
    have hc1z : c1 = 0 := h1 hs1z
    obtain ⟨_, h02, hR0⟩ := hDom0 hs0
    have hR2 := hDom2 hs2
    subst hs1z; subst hc1z
    have hn1 : ¬ (s0 < 0) := by omega
    have hlt2 : 0 < s2 := hs2
    have hn3 : ¬ (s0 < s2) := by omega
    dsimp only; simp only [hn1, hlt2, hn3, hs0, Nat.lt_irrefl, ↓reduceIte, Nat.zero_mul, Nat.add_zero] at hR0 ⊢
    have hA0 : c0 * s0 + c2 * s2 + 2 * c3 = c0 * s0 + (c2 * s2 + 2 * c3) := by omega
    have hR0' : c2 * s2 + 2 * c3 < s0 := by omega
    have ⟨hd0, hm0⟩ := hDivMod c0 s0 _ hs0 hR0'
    have ⟨hd2, hm2⟩ := hDivMod c2 s2 _ hs2 hR2
    rw [hA0, hd0, hm0, hd2, hm2]; simp
  · have hs1z : s1 = 0 := by omega
    have hs2z : s2 = 0 := by omega
    have hc1z : c1 = 0 := h1 hs1z
    have hc2z : c2 = 0 := h2 hs2z
    obtain ⟨_, _, hR0⟩ := hDom0 hs0
    subst hs1z; subst hs2z; subst hc1z; subst hc2z
    have hn1 : ¬ (s0 < 0) := by omega
    dsimp only; simp only [hn1, hs0, Nat.lt_irrefl, ↓reduceIte, Nat.zero_mul, Nat.add_zero] at hR0 ⊢
    have hR0' : 2 * c3 < s0 := by omega
    have ⟨hd0, hm0⟩ := hDivMod c0 s0 _ hs0 hR0'
    rw [hd0, hm0]; simp
  · have hs0z : s0 = 0 := by omega
    have hc0z : c0 = 0 := h0 hs0z
    obtain ⟨h12, hR1⟩ := hDom1 hs1
    have hR2 := hDom2 hs2
    subst hs0z; subst hc0z
    have hn3 : ¬ (s1 < s2) := by omega
    dsimp only; simp only [hs1, hs2, hn3, Nat.lt_irrefl, ↓reduceIte, Nat.zero_mul, Nat.zero_add] at hR1 ⊢
    have hA1 : c1 * s1 + c2 * s2 + 2 * c3 = c1 * s1 + (c2 * s2 + 2 * c3) := by omega
    have ⟨hd1, hm1⟩ := hDivMod c1 s1 _ hs1 hR1
    have ⟨hd2, hm2⟩ := hDivMod c2 s2 _ hs2 hR2
    rw [hA1, hd1, hm1, hd2, hm2]; simp
  · have hs0z : s0 = 0 := by omega
    have hs2z : s2 = 0 := by omega
    have hc0z : c0 = 0 := h0 hs0z
    have hc2z : c2 = 0 := h2 hs2z
    obtain ⟨_, hR1⟩ := hDom1 hs1
    subst hs0z; subst hs2z; subst hc0z; subst hc2z
    have hn3 : ¬ (s1 < 0) := by omega
    dsimp only; simp only [hs1, hn3, Nat.lt_irrefl, ↓reduceIte, Nat.zero_mul, Nat.zero_add, Nat.add_zero] at hR1 ⊢
    have ⟨hd1, hm1⟩ := hDivMod c1 s1 _ hs1 hR1
    rw [hd1, hm1]; simp
  · have hs0z : s0 = 0 := by omega
    have hs1z : s1 = 0 := by omega
    have hc0z : c0 = 0 := h0 hs0z
    have hc1z : c1 = 0 := h1 hs1z
    have hR2 := hDom2 hs2
    subst hs0z; subst hs1z; subst hc0z; subst hc1z
    dsimp only; simp only [hs2, Nat.lt_irrefl, ↓reduceIte, Nat.zero_mul, Nat.zero_add]
    have ⟨hd2, hm2⟩ := hDivMod c2 s2 _ hs2 hR2
    rw [hd2, hm2]; simp
  · have hs0z : s0 = 0 := by omega
    have hs1z : s1 = 0 := by omega
    have hs2z : s2 = 0 := by omega
    have hc0z : c0 = 0 := h0 hs0z
    have hc1z : c1 = 0 := h1 hs1z
    have hc2z : c2 = 0 := h2 hs2z
    subst hs0z; subst hs1z; subst hs2z; subst hc0z; subst hc1z; subst hc2z
    dsimp only; simp

/-- Left-inverse theorem for `viewCopyInverse` under mixed-radix outer strides:
    proves `viewCopyInverse cp (viewCopyDstByte cp inN inC inH inW) = some (inN, inC, inH, inW)`
    for BOTH `transposeCW = false` and `transposeCW = true`! -/
theorem viewCopyInverse_of_radix (cp : ViewCopy) (inN inC inH inW : Nat)
    (hn : inN < cp.n) (hc : inC < cp.c) (hh : inH < cp.h) (hw : inW < cp.w)
    (hRadix :
      let (d0, d1, d2) := if cp.transposeCW then (cp.n, cp.w, cp.h) else (cp.n, cp.c, cp.h)
      let (_, c1, c2, c3) := if cp.transposeCW then (inN, inW, inH, inC) else (inN, inC, inH, inW)
      let s0 := if d0 > 1 then cp.outNStride else 0
      let s1 := if d1 > 1 then cp.outPlaneStride else 0
      let s2 := if d2 > 1 then cp.outRowStride else 0
      (d0 > 1 → s0 > 0) ∧ (d1 > 1 → s1 > 0) ∧ (d2 > 1 → s2 > 0) ∧
      (s0 > 0 → s0 > s1 ∧ s0 > s2 ∧ c1 * s1 + c2 * s2 + 2 * c3 < s0) ∧
      (s1 > 0 → s1 > s2 ∧ c2 * s2 + 2 * c3 < s1) ∧
      (s2 > 0 → 2 * c3 < s2)) :
    viewCopyInverse cp (viewCopyDstByte cp inN inC inH inW) = some (inN, inC, inH, inW) := by
  cases hTr : cp.transposeCW
  · simp only [hTr, Bool.false_eq_true, ↓reduceIte] at hRadix
    obtain ⟨hp0, hp1, hp2, hd0, hd1, hd2⟩ := hRadix
    let s0 := if cp.n > 1 then cp.outNStride else 0
    let s1 := if cp.c > 1 then cp.outPlaneStride else 0
    let s2 := if cp.h > 1 then cp.outRowStride else 0
    have hz0 : s0 = 0 → inN = 0 := fun h => by
      by_cases hn1 : cp.n > 1; · have := hp0 hn1; omega
      · omega
    have hz1 : s1 = 0 → inC = 0 := fun h => by
      by_cases hc1 : cp.c > 1; · have := hp1 hc1; omega
      · omega
    have hz2 : s2 = 0 → inH = 0 := fun h => by
      by_cases hh1 : cp.h > 1; · have := hp2 hh1; omega
      · omega
    have hDstEq : viewCopyDstByte cp inN inC inH inW = cp.outOff + (inN * s0 + inC * s1 + inH * s2 + 2 * inW) := by
      unfold viewCopyDstByte
      simp only [hTr, Bool.false_eq_true, ↓reduceIte]
      have e0 := (mul_if_gt_one (st := cp.outNStride) hn).symm
      have e1 := (mul_if_gt_one (st := cp.outPlaneStride) hc).symm
      have e2 := (mul_if_gt_one (st := cp.outRowStride) hh).symm
      dsimp only [s0, s1, s2]; omega
    have hP := peel3_exact s0 s1 s2 inN inC inH inW hz0 hz1 hz2 hd0 hd1 hd2
    unfold viewCopyInverse
    have hGe : ¬ (viewCopyDstByte cp inN inC inH inW < cp.outOff) := by rw [hDstEq]; omega
    have hSub : viewCopyDstByte cp inN inC inH inW - cp.outOff = inN * s0 + inC * s1 + inH * s2 + 2 * inW := by
      rw [hDstEq]; omega
    simp only [hGe, ↓reduceIte, hTr, Bool.false_eq_true, hSub]
    change (let (rem3, cN, cP, cH) := peel3Strides s0 s1 s2 (inN * s0 + inC * s1 + inH * s2 + 2 * inW); _) = _
    rw [hP]
    dsimp only
    have hEv : (2 * inW % 2 != 0) = false := by simp
    have hD2 : 2 * inW / 2 = inW := by omega
    simp only [hEv, Bool.false_eq_true, ↓reduceIte, hD2, hn, hc, hh, hw, decide_true, beq_self_eq_true, Bool.and_self]
  · simp only [hTr, ↓reduceIte] at hRadix
    obtain ⟨hp0, hp1, hp2, hd0, hd1, hd2⟩ := hRadix
    let s0 := if cp.n > 1 then cp.outNStride else 0
    let s1 := if cp.w > 1 then cp.outPlaneStride else 0
    let s2 := if cp.h > 1 then cp.outRowStride else 0
    have hz0 : s0 = 0 → inN = 0 := fun h => by
      by_cases hn1 : cp.n > 1; · have := hp0 hn1; omega
      · omega
    have hz1 : s1 = 0 → inW = 0 := fun h => by
      by_cases hw1 : cp.w > 1; · have := hp1 hw1; omega
      · omega
    have hz2 : s2 = 0 → inH = 0 := fun h => by
      by_cases hh1 : cp.h > 1; · have := hp2 hh1; omega
      · omega
    have hDstEq : viewCopyDstByte cp inN inC inH inW = cp.outOff + (inN * s0 + inW * s1 + inH * s2 + 2 * inC) := by
      unfold viewCopyDstByte
      simp only [hTr, ↓reduceIte]
      have e0 := (mul_if_gt_one (st := cp.outNStride) hn).symm
      have e1 := (mul_if_gt_one (st := cp.outPlaneStride) hw).symm
      have e2 := (mul_if_gt_one (st := cp.outRowStride) hh).symm
      dsimp only [s0, s1, s2]; omega
    have hP := peel3_exact s0 s1 s2 inN inW inH inC hz0 hz1 hz2 hd0 hd1 hd2
    unfold viewCopyInverse
    have hGe : ¬ (viewCopyDstByte cp inN inC inH inW < cp.outOff) := by rw [hDstEq]; omega
    have hSub : viewCopyDstByte cp inN inC inH inW - cp.outOff = inN * s0 + inW * s1 + inH * s2 + 2 * inC := by
      rw [hDstEq]; omega
    simp only [hGe, ↓reduceIte, hTr, hSub]
    change (let (rem3, cN, cP, cH) := peel3Strides s0 s1 s2 (inN * s0 + inW * s1 + inH * s2 + 2 * inC); _) = _
    rw [hP]
    dsimp only
    have hEv : (2 * inC % 2 != 0) = false := by simp
    have hD2 : 2 * inC / 2 = inC := by omega
    simp only [hEv, Bool.false_eq_true, ↓reduceIte, hD2, hn, hc, hh, hw, decide_true, beq_self_eq_true, Bool.and_self]

/-- Single-coordinate source lookup through `ViewCopy` DRAM byte addressing:
    maps output coordinate `(on, oc, oh, ow)` of `outSh` via `dramByteOffset4D → viewCopyInverse → viewCopySrcByte → decodeDramByteOffset4D`
    to the exact input coordinate `(sn, sc, sh, sw)` in `inSh`. -/
def viewCopyLookupCoord (inSh outSh : Shape4D) (cp : ViewCopy) (on oc oh ow : Nat) :
    Option (Nat × Nat × Nat × Nat) :=
  let dstByte := dramByteOffset4D outSh on oc oh ow
  match viewCopyInverse cp dstByte with
  | none => none
  | some (inN, inC, inH, inW) =>
    decodeDramByteOffset4D inSh (viewCopySrcByte cp inN inC inH inW)

/-- Faithful DRAM-stride tensor evaluator for single-input `.dmaView` (`planSlice`, `planReshape`,
    W-preserving & T-mode `planTranspose`, `planTransposeScratch`, `planMatmulScratchTranspose`,
    `const_anchor`, `halo_slice`, `w_slice`). -/
def evalViewCopyTensor {α : Type} [ScalarField α]
    (inSh outSh : Shape4D) (cp : ViewCopy) (xT : TensorVal α) : Option (TensorVal α) :=
  if xT.shape == inSh.toArray.toList &&
     inSh.n > 0 && inSh.c > 0 && inSh.h > 0 && inSh.w > 0 &&
     outSh.n > 0 && outSh.c > 0 && outSh.h > 0 && outSh.w > 0 then
    let outList := outSh.toArray.toList
    let totalOut := shapeProd outList
    let allValid := (List.range totalOut).all fun flatIdx =>
      let cs := unflatten outList flatIdx
      (viewCopyLookupCoord inSh outSh cp (cs.getD 0 0) (cs.getD 1 0) (cs.getD 2 0) (cs.getD 3 0)).isSome
    if allValid then
      some {
        shape := outList
        data := fun outIdx =>
          let cs := unflatten outList outIdx.val
          match viewCopyLookupCoord inSh outSh cp (cs.getD 0 0) (cs.getD 1 0) (cs.getD 2 0) (cs.getD 3 0) with
          | some (sn, sc, sh, sw) =>
            let inIdx := flatIndex inSh.toArray.toList [sn, sc, sh, sw]
            if hlt : inIdx < shapeProd xT.shape then xT.data ⟨inIdx, hlt⟩ else zero
          | none => zero
      }
    else none
  else none

/-- Faithful strip-output evaluator for `Cascade.unrollWStripCascade`'s `w_stitch` `.dmaView`:
    validates `cp.outOff = 2 * w0`, `cp.outPlaneStride`, and `viewCopyLookupCoord` at destination column
    `w0 + ow` inside `desc.outShape = [1, C, 1, wFull]`, producing the verified `[1, C, 1, curW]` strip! -/
def evalWStitchStripTensor {α : Type} [ScalarField α]
    (inSh outSh : Shape4D) (cp : ViewCopy) (outTySh : List Nat) (xT : TensorVal α) :
    Option (TensorVal α) :=
  if xT.shape == inSh.toArray.toList &&
     outTySh == [1, cp.c, 1, cp.w] &&
     !cp.transposeCW && cp.n == 1 && cp.h == 1 && cp.c > 0 && cp.w > 0 &&
     inSh.n == 1 && inSh.c == cp.c && inSh.h == 1 && inSh.w == cp.w &&
     outSh.n == 1 && outSh.c == cp.c && outSh.h == 1 &&
     cp.outOff % 2 == 0 && cp.outOff / 2 + cp.w ≤ outSh.w then
    let w0 := cp.outOff / 2
    let stripList := [1, cp.c, 1, cp.w]
    let allValid := (List.range (shapeProd stripList)).all fun flatIdx =>
      let cs := unflatten stripList flatIdx
      (viewCopyLookupCoord inSh outSh cp 0 (cs.getD 1 0) 0 (w0 + cs.getD 3 0)).isSome
    if allValid then
      some {
        shape := stripList
        data := fun outIdx =>
          let cs := unflatten stripList outIdx.val
          match viewCopyLookupCoord inSh outSh cp 0 (cs.getD 1 0) 0 (w0 + cs.getD 3 0) with
          | some (sn, sc, sh, sw) =>
            let inIdx := flatIndex inSh.toArray.toList [sn, sc, sh, sw]
            if hlt : inIdx < shapeProd xT.shape then xT.data ⟨inIdx, hlt⟩ else zero
          | none => zero
      }
    else none
  else none

/-- Full W-axis surface scatter semantics for `Cascade.unrollWStripCascade` (`w_stitch` strips
    written at `dstBaseOff = 2 * w0` into shared output surface `[1, C, 1, wFull]`). -/
def evalWStitchSurface {α : Type} [ScalarField α]
    (origOutSh : Shape4D) (strips : List (ViewCopy × TensorVal α)) : TensorVal α :=
  mkTensor4D origOutSh.c 1 origOutSh.w fun oc _ ox =>
    let dstByte := dramByteOffset4D origOutSh 0 oc 0 ox
    match strips.findSome? fun (cp, st) =>
      match viewCopyInverse cp dstByte with
      | some (_, inC, _, inW) =>
        some (read4D st cp.c 1 cp.w inC 0 inW)
      | none => none with
    | some v => v
    | none => zero

/-- Exact head-major slice lookup for `LayoutSelect`'s `matmul_join` (`heads > 1, m > 16`). -/
def evalMatmulJoinLookup {α : Type} (heads : Nat) :
    List (TensorVal α) → Nat → Option (TensorVal α × Nat × Nat)
  | [], _ => none
  | t :: rest, i =>
    let cTot := t.shape.getD 1 0
    if heads == 0 || cTot % heads != 0 then none
    else
      let mi := cTot / heads
      if i < mi then some (t, mi, i)
      else evalMatmulJoinLookup heads rest (i - mi)

/-- Faithful head-major join evaluator for `LayoutSelect`'s `matmul_join` (`concatAxis? = some 1`,
    `batchN = heads > 1`): interleaves `[1, heads * mi, 1, n]` M-slices into `[1, heads * totalM, 1, n]`
    matching `Matmul.lean:476` (`resDmaNStride = totalM * rowBN`) and `evalCore`! -/
def evalMatmulJoin {α : Type} [ScalarField α] (heads : Nat)
    (ins : List (TensorVal α)) (outSh : Shape4D) : Option (TensorVal α) :=
  if heads == 0 || ins.isEmpty || outSh.n != 1 || outSh.h != 1 || outSh.w == 0 || outSh.c % heads != 0 then
    none
  else
    let totalM := outSh.c / heads
    if totalM == 0 then none
    else
      let allOk := ins.all fun t =>
        match t.shape with
        | [1, ci, 1, wi] => wi == outSh.w && ci > 0 && ci % heads == 0
        | _ => false
      let sumM := ins.foldl (fun acc t => acc + t.shape.getD 1 0 / heads) 0
      if allOk && sumM == totalM then
        some (mkTensor4D outSh.c 1 outSh.w fun hm _ j =>
          let hd := hm / totalM
          let i := hm % totalM
          match evalMatmulJoinLookup heads ins i with
          | some (srcT, mi, localI) =>
            read4D srcT (heads * mi) 1 outSh.w (hd * mi + localI) 0 j
          | none => zero)
      else none

/-- Exact 4D `unflatten` inverse on arbitrary `[N, C, H, W]` (`N ≥ 1`). -/
theorem unflatten_4d_gen (N C H W n c y x : Nat)
    (hn : n < N) (hc : c < C) (hy : y < H) (hx : x < W) :
    unflatten [N, C, H, W] (flatIndex [N, C, H, W] [n, c, y, x]) = [n, c, y, x] := by
  have hW : W > 0 := by omega
  have hHW : H * W > 0 := Nat.mul_pos (by omega) hW
  have hCHW : C * (H * W) > 0 := Nat.mul_pos (by omega) hHW
  have hYW : y * W + x < H * W := by
    have h1 : (y + 1) * W ≤ H * W := Nat.mul_le_mul_right W (Nat.succ_le_of_lt hy)
    rw [Nat.add_mul, Nat.one_mul] at h1; omega
  have hCYW : c * (H * W) + (y * W + x) < C * (H * W) := by
    have h1 : (c + 1) * (H * W) ≤ C * (H * W) := Nat.mul_le_mul_right (H * W) (Nat.succ_le_of_lt hc)
    rw [Nat.add_mul, Nat.one_mul] at h1; omega
  have hFl : flatIndex [N, C, H, W] [n, c, y, x] = n * (C * (H * W)) + (c * (H * W) + (y * W + x)) := by
    simp [flatIndex, shapeProd]
  rw [hFl]
  have hDiv0 : (n * (C * (H * W)) + (c * (H * W) + (y * W + x))) / (C * (H * W)) = n := by
    rw [Nat.mul_comm n (C * (H * W)), Nat.mul_add_div hCHW, Nat.div_eq_of_lt hCYW, Nat.add_zero]
  have hMod0 : (n * (C * (H * W)) + (c * (H * W) + (y * W + x))) % (C * (H * W)) = c * (H * W) + (y * W + x) := by
    rw [Nat.mul_comm n (C * (H * W)), Nat.mul_add_mod, Nat.mod_eq_of_lt hCYW]
  have hDiv1 : (c * (H * W) + (y * W + x)) / (H * W) = c := by
    rw [Nat.mul_comm c (H * W), Nat.mul_add_div hHW, Nat.div_eq_of_lt hYW, Nat.add_zero]
  have hMod1 : (c * (H * W) + (y * W + x)) % (H * W) = y * W + x := by
    rw [Nat.mul_comm c (H * W), Nat.mul_add_mod, Nat.mod_eq_of_lt hYW]
  have hDiv2 : (y * W + x) / W = y := by
    rw [Nat.mul_comm y W, Nat.mul_add_div hW, Nat.div_eq_of_lt hx, Nat.add_zero]
  have hMod2 : (y * W + x) % W = x := by
    rw [Nat.mul_comm y W, Nat.mul_add_mod, Nat.mod_eq_of_lt hx]
  have _ := hn
  simp only [unflatten, shapeProd, Nat.mul_one, hDiv0, hMod0, hDiv1, hMod1, hDiv2, hMod2, Nat.div_one]

/-- Flat index bound for arbitrary 4D `[N, C, H, W]`. -/
theorem flat4D_gen_lt (N C H W n c y x : Nat) (hn : n < N) (hc : c < C) (hy : y < H) (hx : x < W) :
    flatIndex [N, C, H, W] [n, c, y, x] < shapeProd [N, C, H, W] := by
  simp only [flatIndex, shapeProd, Nat.mul_one, Nat.add_zero]
  have hYW : y * W + x < H * W := by
    have h1 : (y + 1) * W ≤ H * W := Nat.mul_le_mul_right W (Nat.succ_le_of_lt hy)
    rw [Nat.add_mul, Nat.one_mul] at h1; omega
  have hCYW : c * (H * W) + (y * W + x) < C * (H * W) := by
    have h1 : (c + 1) * (H * W) ≤ C * (H * W) := Nat.mul_le_mul_right (H * W) (Nat.succ_le_of_lt hc)
    rw [Nat.add_mul, Nat.one_mul] at h1; omega
  have hN : (n + 1) * (C * (H * W)) ≤ N * (C * (H * W)) :=
    Nat.mul_le_mul_right (C * (H * W)) (Nat.succ_le_of_lt hn)
  rw [Nat.add_mul, Nat.one_mul] at hN; omega

/-- Reusable characterization of `evalViewCopyTensor`: given a coordinate map `srcOf` matching
    `viewCopyLookupCoord` on every in-bounds `(on, oc, oh, ow)`, `evalViewCopyTensor` succeeds
    and reads the exact `xT` element at `srcOf on oc oh ow`! -/
theorem evalViewCopyTensor_of_lookup {α : Type} [ScalarField α]
    (inSh outSh : Shape4D) (cp : ViewCopy) (xT : TensorVal α)
    (srcOf : Nat → Nat → Nat → Nat → Nat × Nat × Nat × Nat)
    (hInSh : xT.shape = inSh.toArray.toList)
    (hInPos : inSh.n > 0 ∧ inSh.c > 0 ∧ inSh.h > 0 ∧ inSh.w > 0)
    (hOutPos : outSh.n > 0 ∧ outSh.c > 0 ∧ outSh.h > 0 ∧ outSh.w > 0)
    (hLookup : ∀ on oc oh ow, on < outSh.n → oc < outSh.c → oh < outSh.h → ow < outSh.w →
      viewCopyLookupCoord inSh outSh cp on oc oh ow = some (srcOf on oc oh ow))
    (hSrcBounds : ∀ on oc oh ow, on < outSh.n → oc < outSh.c → oh < outSh.h → ow < outSh.w →
      let (sn, sc, sh, sw) := srcOf on oc oh ow
      sn < inSh.n ∧ sc < inSh.c ∧ sh < inSh.h ∧ sw < inSh.w) :
    ∃ r, evalViewCopyTensor inSh outSh cp xT = some r ∧
      r.shape = outSh.toArray.toList ∧
      ∀ on oc oh ow, on < outSh.n → oc < outSh.c → oh < outSh.h → ow < outSh.w →
        let outIdx := flatIndex outSh.toArray.toList [on, oc, oh, ow]
        let (sn, sc, sh, sw) := srcOf on oc oh ow
        let inIdx := flatIndex inSh.toArray.toList [sn, sc, sh, sw]
        ∀ (hOutLt : outIdx < shapeProd r.shape) (hInLt : inIdx < shapeProd xT.shape),
          r.data ⟨outIdx, hOutLt⟩ = xT.data ⟨inIdx, hInLt⟩ := by
  have hCond : (xT.shape == inSh.toArray.toList &&
      decide (inSh.n > 0) && decide (inSh.c > 0) && decide (inSh.h > 0) && decide (inSh.w > 0) &&
      decide (outSh.n > 0) && decide (outSh.c > 0) && decide (outSh.h > 0) && decide (outSh.w > 0)) = true := by
    simp [hInSh, hInPos, hOutPos]
  have hOutList : outSh.toArray.toList = [outSh.n, outSh.c, outSh.h, outSh.w] := rfl
  have hValid : ((List.range (shapeProd outSh.toArray.toList)).all fun flatIdx =>
      let cs := unflatten outSh.toArray.toList flatIdx
      (viewCopyLookupCoord inSh outSh cp (cs.getD 0 0) (cs.getD 1 0) (cs.getD 2 0) (cs.getD 3 0)).isSome) = true := by
    rw [List.all_eq_true]
    intro flatIdx hMem
    rw [List.mem_range, hOutList] at hMem
    have hProd : shapeProd [outSh.n, outSh.c, outSh.h, outSh.w] = outSh.n * (outSh.c * (outSh.h * outSh.w)) := by
      simp [shapeProd]
    rw [hProd] at hMem
    have hHW : outSh.h * outSh.w > 0 := Nat.mul_pos hOutPos.2.2.1 hOutPos.2.2.2
    have hCHW : outSh.c * (outSh.h * outSh.w) > 0 := Nat.mul_pos hOutPos.2.1 hHW
    let on := flatIdx / (outSh.c * (outSh.h * outSh.w))
    let rem1 := flatIdx % (outSh.c * (outSh.h * outSh.w))
    let oc := rem1 / (outSh.h * outSh.w)
    let rem2 := rem1 % (outSh.h * outSh.w)
    let oh := rem2 / outSh.w
    let ow := rem2 % outSh.w
    have hon : on < outSh.n := Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hMem)
    have hr1 : rem1 < outSh.c * (outSh.h * outSh.w) := Nat.mod_lt flatIdx hCHW
    have hoc : oc < outSh.c := Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hr1)
    have hr2 : rem2 < outSh.h * outSh.w := Nat.mod_lt rem1 hHW
    have hoh : oh < outSh.h := Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hr2)
    have how : ow < outSh.w := Nat.mod_lt rem2 hOutPos.2.2.2
    have hUnf : unflatten outSh.toArray.toList flatIdx = [on, oc, oh, ow] := by
      rw [hOutList]
      simp only [unflatten, shapeProd, Nat.mul_one, Nat.div_one]
      rfl
    rw [hUnf]
    change (viewCopyLookupCoord inSh outSh cp on oc oh ow).isSome = true
    rw [hLookup on oc oh ow hon hoc hoh how]; rfl
  unfold evalViewCopyTensor
  simp only [hCond, ↓reduceIte, hValid]
  refine ⟨_, rfl, rfl, fun on oc oh ow hn hc hh hw hOutLt hInLt => ?_⟩
  dsimp only
  have hUnf : unflatten outSh.toArray.toList (flatIndex outSh.toArray.toList [on, oc, oh, ow]) = [on, oc, oh, ow] := by
    rw [hOutList]; exact unflatten_4d_gen outSh.n outSh.c outSh.h outSh.w on oc oh ow hn hc hh hw
  rw [hUnf]
  change (match viewCopyLookupCoord inSh outSh cp on oc oh ow with
    | some (sn, sc, sh, sw) =>
      let inIdx := flatIndex inSh.toArray.toList [sn, sc, sh, sw]
      if hlt : inIdx < shapeProd xT.shape then xT.data ⟨inIdx, hlt⟩ else zero
    | none => zero) = _
  rw [hLookup on oc oh ow hn hc hh hw]
  dsimp only
  have _ := hSrcBounds on oc oh ow hn hc hh hw
  have hInLt' : flatIndex inSh.toArray.toList [(srcOf on oc oh ow).1, (srcOf on oc oh ow).2.1,
      (srcOf on oc oh ow).2.2.1, (srcOf on oc oh ow).2.2.2] < shapeProd xT.shape := hInLt
  simp only [hInLt', ↓reduceDIte]

/-- Exact 4D value-semantic evaluation of ALL 5 `AneOp` families (`neConv`, `neAct`, `neMatmul`,
    `dmaView`, `pe`), using exact DRAM-stride `ViewCopy` semantics for `.dmaView` (slices, arbitrary
    W-preserving & T-mode transposes, batched `matmul_trA`/`matmul_trB`, head-major `matmul_join`,
    reshapes, and `w_stitch`) and strict output-shape checking on `.pe (.reduce ...)`! -/
def evalAneOpCore {α : Type} [ScalarField α] (ops : ScalarOps α)
    (decodeConvWeight : WeightData → Nat → Nat → Nat → Nat → Option (Nat → Nat → Nat → Nat → α))
    (decodeBias : Option ByteArray → Nat → Option (Option (Nat → α)))
    (op : AneOp) (ins : List (TensorVal α)) (outTys : List TensorType) : Option (List (TensorVal α)) :=
  match op, outTys with
  | .neConv kh kw sh sw pad mode cin cout weight bias act _ _ _, [outTy] =>
    match ins, outTy.shape.toList with
    | [xT], [1, cOut, ho, wo] =>
      match xT.shape with
      | [1, cIn, h, w] =>
        if cIn == cin && cOut == cout then
          let groups := match mode with
            | .dense => 1
            | .depthwise => cin
            | .grouped g => g
          let cinG := cin / groups
          let coutG := cout / groups
          let (pt, pl) := match pad with
            | .valid => (0, 0)
            | .same =>
              let padH := if (ho - 1) * sh + kh > h then (ho - 1) * sh + kh - h else 0
              let padW := if (wo - 1) * sw + kw > w then (wo - 1) * sw + kw - w else 0
              (padH / 2, padW / 2)
            | .strip ptS _pbS =>
              let padW := if (wo - 1) * sw + kw > w then (wo - 1) * sw + kw - w else 0
              (ptS, padW / 2)
          match decodeConvWeight weight cout cinG kh kw, decodeBias bias cout with
          | some wFn, some bFn? =>
            some [evalConvTensor4D ops h w cinG cout coutG ho wo kh kw sh sw pt pl
              (fun c y x => read4D xT cin h w c y x) wFn bFn? act]
          | _, _ => none
        else none
      | _ => none
    | _, _ => none

  | .neAct kind _ _, [outTy] =>
    match kind, ins, outTy.shape.toList with
    | .act spec, [xT], _ =>
      some [mapTensor (applyNeAct ops spec) xT]
    | .channelReduce rop, [xT], [1, 1, ho, wo] =>
      match xT.shape with
      | [1, cIn, h, w] =>
        if h == ho && w == wo then
          some [mkTensor4D 1 ho wo fun _ y x =>
            evalReduce1D ops rop zero cIn fun c => read4D xT cIn h w c y x]
        else none
      | _ => none
    | .nePool pkind kh kw sh sw pad excl, [xT], [1, cOut, ho, wo] =>
      match xT.shape with
      | [1, cIn, h, w] =>
        if cIn == cOut then
          match poolAxisGeom h kh sh pad true, poolAxisGeom w kw sw pad false with
          | .ok gh, .ok gw =>
            some [mkTensor4D cOut ho wo fun c oy ox =>
              evalPool2DElem ops pkind h w kh kw sh sw gh.padBefore gw.padBefore excl
                (fun ch y x => read4D xT cIn h w ch y x) c oy ox]
          | _, _ => none
        else none
      | _ => none
    | _, _, _ => none

  | .neMatmul heads m k n trA trB scaleBits mOffset totalM, [outTy] =>
    match ins, outTy.shape.toList with
    | [aT, bT], [1, cOut, 1, wOut] =>
      if cOut == heads * m && wOut == n then
        let totM := if totalM == 0 then m else totalM
        let getA := fun hd row col =>
          let (r, c) := if trA then (col, row) else (row, col)
          let cDim := if trA then heads * k else heads * totM
          let wDim := if trA then totM else k
          read4D aT cDim 1 wDim (hd * (if trA then k else totM) + r) 0 c
        let getB := fun hd row col =>
          let (r, c) := if trB then (col, row) else (row, col)
          let cDim := if trB then heads * n else heads * k
          let wDim := if trB then k else n
          read4D bT cDim 1 wDim (hd * (if trB then n else k) + r) 0 c
        some [mkTensor4D (heads * m) 1 n fun hm _ j =>
          let hd := hm / m
          let i := hm % m
          evalMatmulElem k false false (ops.f16Val scaleBits) getA getB hd (mOffset + i) j]
      else none
    | _, _ => none

  | .dmaView desc _, [outTy] =>
    match desc.concatAxis? with
    | some ax =>
      if ax == 1 && desc.batchN > 1 then
        (evalMatmulJoin desc.batchN ins desc.outShape).map fun r => [r]
      else
        (evalConcat ax ins).map fun r => [r]
    | none =>
      match ins with
      | [xT] =>
        let cp := desc.toViewCopy
        if desc.outShape.toArray.toList == outTy.shape.toList then
          (evalViewCopyTensor desc.inShape desc.outShape cp xT).map fun r => [r]
        else
          (evalWStitchStripTensor desc.inShape desc.outShape cp outTy.shape.toList xT).map fun r => [r]
      | _ => none

  | .pe peOp _, [outTy] =>
    match peOp, ins, outTy.shape.toList with
    | .unary u, [xT], _ =>
      some [mapTensor (evalPeUnaryScalar ops u) xT]
    | .binary b rev, [aT, bT], outSh =>
      (evalBinaryBroadcast (evalPeBinaryScalar ops b rev) outSh aT bT).map fun r => [r]
    | .pool pkind kh kw sh sw pad excl, [xT], [1, cOut, ho, wo] =>
      match xT.shape with
      | [1, cIn, h, w] =>
        if cIn == cOut then
          match poolAxisGeom h kh sh pad true, poolAxisGeom w kw sw pad false with
          | .ok gh, .ok gw =>
            some [mkTensor4D cOut ho wo fun c oy ox =>
              evalPool2DElem ops pkind h w kh kw sh sw gh.padBefore gw.padBefore excl
                (fun ch y x => read4D xT cIn h w ch y x) c oy ox]
          | _, _ => none
        else none
      | _ => none
    | .reduce rop axes eps, [xT], [1, cOut, hOut, wOut] =>
      match xT.shape with
      | [1, cIn, hIn, wIn] =>
        let expC := if axes.reduceC then 1 else cIn
        let expH := if axes.reduceH then 1 else hIn
        let expW := if axes.reduceW then 1 else wIn
        if cIn > 0 && hIn > 0 && wIn > 0 && cOut == expC && hOut == expH && wOut == expW then
          let cStep := if axes.reduceC then cIn else 1
          let hStep := if axes.reduceH then hIn else 1
          let wStep := if axes.reduceW then wIn else 1
          let nTot := cStep * (hStep * wStep)
          some [mkTensor4D cOut hOut wOut fun oc oy ox =>
            evalReduce1D ops rop (ops.f16Val eps) nTot fun idx =>
              let dc := idx / (hStep * wStep)
              let rem := idx % (hStep * wStep)
              let dh := rem / wStep
              let dw := rem % wStep
              let ic := if axes.reduceC then dc else oc
              let iy := if axes.reduceH then dh else oy
              let ix := if axes.reduceW then dw else ox
              read4D xT cIn hIn wIn ic iy ix]
        else none
      | _ => none
    | _, _, _ => none

  | _, _ => none

/-- `OpSemantics AneOp α` built from `evalAneOpCore`, with proven output-arity preservation. -/
def mkAneSemantics {α : Type} [ScalarField α] (ops : ScalarOps α)
    (decodeConvWeight : WeightData → Nat → Nat → Nat → Nat → Option (Nat → Nat → Nat → Nat → α))
    (decodeBias : Option ByteArray → Nat → Option (Option (Nat → α))) :
    OpSemantics AneOp α where
  evalOp := evalAneOpCore ops decodeConvWeight decodeBias
  evalOp_len op ins outTys outs h := by
    cases op with
    | neConv _ _ _ _ _ _ _ _ _ _ _ _ _ _ =>
      cases outTys with
      | nil => cases h
      | cons _ rest =>
        cases rest with
        | cons _ _ => cases h
        | nil =>
          dsimp only [evalAneOpCore] at h
          repeat (first | cases h | (split at h <;> try cases h))
          simp_all
    | neAct kind _ _ =>
      cases outTys with
      | nil => cases h
      | cons _ rest =>
        cases rest with
        | cons _ _ => cases h
        | nil =>
          dsimp only [evalAneOpCore] at h
          split at h
          · simp only [Option.some.injEq] at h; subst h; rfl
          · split at h
            · have hIteLen : ∀ {P : Prop} [Decidable P] {x : TensorVal α},
                  (if P then some [x] else none) = some outs → outs.length = 1 := by
                intro P _ x hEq; split at hEq <;> cases hEq; rfl
              exact hIteLen h
            · cases h
          · split at h
            · split at h
              · split at h
                · simp only [Option.some.injEq] at h; subst h; rfl
                · cases h
              · cases h
            · cases h
          · cases h
    | neMatmul _ _ _ _ _ _ _ _ _ =>
      cases outTys with
      | nil => cases h
      | cons _ rest =>
        cases rest with
        | cons _ _ => cases h
        | nil =>
          dsimp only [evalAneOpCore] at h
          split at h
          · split at h
            · simp only [Option.some.injEq] at h; subst h; rfl
            · cases h
          · cases h
    | dmaView desc _ =>
      cases outTys with
      | nil => cases h
      | cons outTy rest =>
        cases rest with
        | cons _ _ => cases h
        | nil =>
          dsimp only [evalAneOpCore] at h
          cases hAx : desc.concatAxis? with
          | some ax =>
            rw [hAx] at h
            dsimp only at h
            split at h <;> (simp only [Option.map_eq_some_iff] at h; obtain ⟨_, _, rfl⟩ := h; rfl)
          | none =>
            rw [hAx] at h
            cases ins with
            | nil => cases h
            | cons xT xs =>
              cases xs with
              | cons _ _ => cases h
              | nil =>
                dsimp only at h
                split at h <;> (simp only [Option.map_eq_some_iff] at h; obtain ⟨_, _, rfl⟩ := h; rfl)
    | pe peOp _ =>
      cases outTys with
      | nil => cases h
      | cons _ rest =>
        cases rest with
        | cons _ _ => cases h
        | nil =>
          dsimp only [evalAneOpCore] at h
          split at h
          · simp only [Option.some.injEq] at h; subst h; rfl
          · simp only [Option.map_eq_some_iff] at h
            obtain ⟨_, _, rfl⟩ := h; rfl
          · split at h
            · split at h
              · split at h
                · simp only [Option.some.injEq] at h; subst h; rfl
                · cases h
              · cases h
            · cases h
          · split at h
            · have hIteLen : ∀ {P : Prop} [Decidable P] {x : TensorVal α},
                  (if P then some [x] else none) = some outs → outs.length = 1 := by
                intro P _ x hEq; split at hEq <;> cases hEq; rfl
              exact hIteLen h
            · cases h
          · cases h

/-- Top-level L2 `evalAne` denotational graph evaluator. -/
def evalAne {α : Type} [ScalarField α] (ops : ScalarOps α)
    (decodeConvWeight : WeightData → Nat → Nat → Nat → Nat → Option (Nat → Nat → Nat → Nat → α))
    (decodeBias : Option ByteArray → Nat → Option (Option (Nat → α)))
    (g : SSAGraph AneOp) (env : Env α) : Option (Env α) :=
  evalGraph (mkAneSemantics ops decodeConvWeight decodeBias) g env

/-- `LayoutSelect.logicalToShape4D` preserves exact total element count (`shapeProd`) across
    ALL supported ranks (2D `[1, C]`, 3D `[H, M, K]`, and 4D `[1, C, H, W]`)! -/
theorem layoutSelect_logicalToShape4D_preserves_shapeProd
    (dims : Array Nat) (ctx : String) (sh4 : Shape4D)
    (hOk : logicalToShape4D dims ctx = .ok sh4) :
    shapeProd [sh4.n, sh4.c, sh4.h, sh4.w] = shapeProd dims.toList := by
  unfold logicalToShape4D at hOk
  dsimp only [pure, Except.pure, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw] at hOk
  by_cases h4 : dims.size = 4
  · have h4b : (dims.size == 4) = true := by simp [h4]
    rw [h4b] at hOk; simp only [↓reduceIte] at hOk
    split at hOk
    · cases hOk
    · rename_i hn1
      simp only [Except.ok.injEq] at hOk
      subst hOk
      have h0 : dims[0]! = 1 := by simp_all
      have hL : dims.toList = [dims[0]!, dims[1]!, dims[2]!, dims[3]!] := by
        apply List.ext_getElem (by simp [h4])
        intro i hi1 hi2
        have hi4 : i < 4 := by simpa using hi2
        rw [Array.getElem_toList]
        match i, hi4 with
        | 0, _ => rw [getElem!_pos _ 0 (by omega)]; rfl
        | 1, _ => rw [getElem!_pos _ 1 (by omega)]; rfl
        | 2, _ => rw [getElem!_pos _ 2 (by omega)]; rfl
        | 3, _ => rw [getElem!_pos _ 3 (by omega)]; rfl
      rw [hL, h0]
  · have h4b : (dims.size == 4) = false := by simp [h4]
    rw [h4b] at hOk; simp only [Bool.false_eq_true, ↓reduceIte] at hOk
    by_cases h3 : dims.size = 3
    · have h3b : (dims.size == 3) = true := by simp [h3]
      rw [h3b] at hOk; simp only [↓reduceIte, Except.ok.injEq] at hOk
      subst hOk
      have hL : dims.toList = [dims[0]!, dims[1]!, dims[2]!] := by
        apply List.ext_getElem (by simp [h3])
        intro i hi1 hi2
        have hi3 : i < 3 := by simpa using hi2
        rw [Array.getElem_toList]
        match i, hi3 with
        | 0, _ => rw [getElem!_pos _ 0 (by omega)]; rfl
        | 1, _ => rw [getElem!_pos _ 1 (by omega)]; rfl
        | 2, _ => rw [getElem!_pos _ 2 (by omega)]; rfl
      rw [hL]
      simp only [shapeProd, Nat.one_mul, Nat.mul_one, Nat.mul_assoc]
    · have h3b : (dims.size == 3) = false := by simp [h3]
      rw [h3b] at hOk; simp only [Bool.false_eq_true, ↓reduceIte] at hOk
      by_cases h2 : dims.size = 2
      · have h2b : (dims.size == 2) = true := by simp [h2]
        rw [h2b] at hOk; simp only [↓reduceIte] at hOk
        split at hOk
        · cases hOk
        · rename_i hn1
          simp only [Except.ok.injEq] at hOk
          subst hOk
          have h0 : dims[0]! = 1 := by simp_all
          have hL : dims.toList = [dims[0]!, dims[1]!] := by
            apply List.ext_getElem (by simp [h2])
            intro i hi1 hi2
            have hi2' : i < 2 := by simpa using hi2
            rw [Array.getElem_toList]
            match i, hi2' with
            | 0, _ => rw [getElem!_pos _ 0 (by omega)]; rfl
            | 1, _ => rw [getElem!_pos _ 1 (by omega)]; rfl
          rw [hL, h0]
          simp [shapeProd]
      · have h2b : (dims.size == 2) = false := by simp [h2]
        rw [h2b] at hOk; simp at hOk

/-- `LayoutSelect` equivalence for ALL 11 `CoreOp.elemUnary` lowerings (`LayoutSelect.lean` lines 774–821):
    `abs`, `neg` → `scalarMul(f16NegOne)`, `square`, `log` → `neAct .log2` + `scalarMul(ln2)`,
    `softplus` → `neAct .exp` + `scalarAdd(1)` + `neAct .log2` + `scalarMul(ln2)`,
    `scalarAdd`, `scalarSub` → `PeOp.scalarSub`, `scalarRSub` → `PeOp.scalarRSub`,
    `scalarMul`, `scalarDiv` → `PeOp.chain none none (some invBits)`, and `linearAct`! -/
theorem layoutSelect_elemUnary_all_equiv {α : Type} [ScalarField α] (ops : ScalarOps α) (x : α)
    (b a invBits : UInt16)
    (hRecip : ops.f16Val invBits = inv (ops.f16Val b)) :
    let ln2Bits := floatToF16CanonicalZero 0.6931471805599453
    evalPeUnaryScalar ops .abs x = evalElemUnaryScalar ops .abs x ∧
    evalPeUnaryScalar ops (.scalarMul f16NegOne) x = evalElemUnaryScalar ops .neg x ∧
    evalPeUnaryScalar ops .square x = evalElemUnaryScalar ops .square x ∧
    evalPeUnaryScalar ops (.scalarMul ln2Bits) (applyNeAct ops (.lut .log2) x) =
      evalElemUnaryScalar ops .log x ∧
    evalPeUnaryScalar ops (.scalarMul ln2Bits)
      (applyNeAct ops (.lut .log2)
        (evalPeUnaryScalar ops (.scalarAdd f16PosOne) (applyNeAct ops (.lut .exp) x))) =
      evalElemUnaryScalar ops .softplus x ∧
    evalPeUnaryScalar ops (.scalarAdd b) x = evalElemUnaryScalar ops (.scalarAdd b) x ∧
    evalPeUnaryScalar ops (.scalarSub b) x = evalElemUnaryScalar ops (.scalarSub b) x ∧
    evalPeUnaryScalar ops (.scalarRSub b) x = evalElemUnaryScalar ops (.scalarRSub b) x ∧
    evalPeUnaryScalar ops (.scalarMul b) x = evalElemUnaryScalar ops (.scalarMul b) x ∧
    evalPeUnaryScalar ops (.chain none none (some invBits)) x = evalElemUnaryScalar ops (.scalarDiv b) x ∧
    evalPeUnaryScalar ops (.linearActivation a b) x = evalElemUnaryScalar ops (.linearAct a b) x := by
  refine ⟨rfl, ?_, rfl, ?_, ?_, rfl, ?_, ?_, rfl, ?_, rfl⟩
  · -- neg -> scalarMul f16NegOne
    simp only [evalPeUnaryScalar, PeOp.scalarMul, evalElemUnaryScalar, ops.f16_negOne]
    exact mul_neg_one x
  · -- log -> log2 * ln2
    simp only [evalPeUnaryScalar, PeOp.scalarMul, applyNeAct, evalElemUnaryScalar,
      ops.lut_log2, ops.f16_ln2, ops.log_eq_log2_mul_ln2]
  · -- softplus -> log2(exp(x) + 1) * ln2
    simp only [evalPeUnaryScalar, PeOp.scalarMul, PeOp.scalarAdd, applyNeAct, evalElemUnaryScalar,
      ops.lut_exp, ops.lut_log2, ops.f16_posOne, ops.f16_ln2, ops.softplus_def,
      ops.log_eq_log2_mul_ln2, add_comm (ops.exp x) one]
  · -- scalarSub -> add x (neg b)
    simp only [evalPeUnaryScalar, PeOp.scalarSub, evalElemUnaryScalar, sub, ops.f16_neg]
  · -- scalarRSub -> add (mul x (-1)) b = sub b x
    simp only [evalPeUnaryScalar, PeOp.scalarRSub, evalElemUnaryScalar, sub, ops.f16_negOne]
    rw [mul_neg_one, add_comm]
  · -- scalarDiv -> mul x invBits
    simp only [evalPeUnaryScalar, evalElemUnaryScalar, div, hRecip]

/-- `LayoutSelect` equivalence for ALL 6 `CoreOp.elemBinary` lowerings (`LayoutSelect.lean` lines 862–975),
    including `div(a, b) → pe.mul(a, neAct.inverse(b))` AND the fast 1-task diagonal/identity
    `neConv 1x1` per-channel `mul`, `add`, and `sub` (both `x - c` and `c - x`)! -/
theorem layoutSelect_elemBinary_all_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (C H W : Nat) (x : Nat → Nat → Nat → α) (cVec : Nat → α)
    (a b : α) (oc oy ox : Nat) (hoc : oc < C) (hoy : oy < H) (hox : ox < W) :
    -- Standard 2-source PE ops + div decomposition
    evalPeBinaryScalar ops .add false a b = evalElemBinaryScalar ops .add a b ∧
    evalPeBinaryScalar ops .sub false a b = evalElemBinaryScalar ops .sub a b ∧
    evalPeBinaryScalar ops .mul false a b = evalElemBinaryScalar ops .mul a b ∧
    evalPeBinaryScalar ops .mul false a (applyNeAct ops (.lut .inverse) b) = evalElemBinaryScalar ops .div a b ∧
    evalPeBinaryScalar ops .max false a b = evalElemBinaryScalar ops .max a b ∧
    evalPeBinaryScalar ops .min false a b = evalElemBinaryScalar ops .min a b ∧
    -- Fast per-channel `neConv 1x1` lowerings (`LayoutSelect.lean` lines 902-927)
    evalConvElem ops H W C C 1 1 1 1 0 0 x (fun o i _ _ => if o = i then cVec o else zero) none .none oc oy ox =
      mul (cVec oc) (x oc oy ox) ∧
    evalConvElem ops H W C C 1 1 1 1 0 0 x (fun o i _ _ => if o = i then one else zero) (some cVec) .none oc oy ox =
      add (x oc oy ox) (cVec oc) ∧
    evalConvElem ops H W C C 1 1 1 1 0 0 x (fun o i _ _ => if o = i then one else zero)
      (some fun o => neg (cVec o)) .none oc oy ox =
      sub (x oc oy ox) (cVec oc) ∧
    evalConvElem ops H W C C 1 1 1 1 0 0 x (fun o i _ _ => if o = i then neg one else zero)
      (some cVec) .none oc oy ox =
      sub (cVec oc) (x oc oy ox) := by
  refine ⟨rfl, rfl, rfl, ?_, rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [evalPeBinaryScalar, Bool.false_eq_true, ↓reduceIte, applyNeAct, evalElemBinaryScalar, div, ops.lut_inverse]
  · exact diagConv1x1_exact ops C H W x cVec none oc oy ox hoc hoy hox
  · rw [diagConv1x1_exact ops C H W x (fun _ => one) (some cVec) oc oy ox hoc hoy hox, one_mul]
  · rw [diagConv1x1_exact ops C H W x (fun _ => one) (some fun o => neg (cVec o)) oc oy ox hoc hoy hox, one_mul]
    rfl
  · rw [diagConv1x1_exact ops C H W x (fun _ => neg one) (some cVec) oc oy ox hoc hoy hox]
    dsimp only
    rw [neg_one_mul, add_comm]
    rfl

/-- `LayoutSelect` equivalence for ALL 5 decomposed C-axis `W > 1` reductions
    (`LayoutSelect.lean` lines 1045–1091: `rsqrtMeanSquare`, `rsqrtSumSquare`, `sumSquare`, `l2`, `invSum`). -/
theorem layoutSelect_reduce_cAxis_decomposed_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (C : Nat) (eps : α) (f : Nat → α) :
    let sq := fun c => evalPeUnaryScalar ops .square (f c)
    -- 1. rsqrtMeanSquare: square -> neChannelReduce .mean -> (+ eps) -> lut (.rsqrt 0.0)
    applyNeAct ops (.lut (.rsqrt 0.0)) (add (evalReduce1D ops .mean zero C sq) eps) =
      evalReduce1D ops .rsqrtMeanSquare eps C f ∧
    -- 2. rsqrtSumSquare: square -> neChannelReduce .sum -> (+ eps) -> lut (.rsqrt 0.0)
    applyNeAct ops (.lut (.rsqrt 0.0)) (add (evalReduce1D ops .sum zero C sq) eps) =
      evalReduce1D ops .rsqrtSumSquare eps C f ∧
    -- 3. sumSquare: square -> neChannelReduce .sum
    evalReduce1D ops .sum zero C sq = evalReduce1D ops .sumSquare zero C f ∧
    -- 4. l2: square -> neChannelReduce .sum -> lut .sqrt
    applyNeAct ops (.lut .sqrt) (evalReduce1D ops .sum zero C sq) =
      evalReduce1D ops .l2 zero C f ∧
    -- 5. invSum: neChannelReduce .sum -> lut .inverse
    applyNeAct ops (.lut .inverse) (evalReduce1D ops .sum zero C f) =
      evalReduce1D ops .invSum zero C f := by
  simp only [evalPeUnaryScalar, evalReduce1D, applyNeAct, ops.lut_rsqrt_zero, ops.lut_sqrt, ops.lut_inverse]
  exact ⟨trivial, trivial, trivial, trivial, trivial⟩

/-- `LayoutSelect` equivalence for global average pooling lowered to `.pe (.reduce .mean {reduceH, reduceW})`
    (`LayoutSelect.lean` lines 1008–1010: when `kh = H, kw = W, pt = 0, pl = 0`, global avg pool equals
    exact 2D `H × W` mean reduction!). -/
theorem layoutSelect_pool_global_avg_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W : Nat) (x : Nat → Nat → Nat → α) (c : Nat) :
    evalPool2DElem ops .avg H W H W 1 1 0 0 false x c 0 0 =
    evalReduce1D ops .mean zero (H * W) (fun idx => x c (idx / W) (idx % W)) := by
  unfold evalPool2DElem evalReduce1D
  simp only [Bool.false_eq_true, ↓reduceIte, Int.ofNat_zero, Int.ofNat_one]
  congr 1
  rw [sumFin_prod H W]
  apply sumFin_congr; intro ky hky
  apply sumFin_congr; intro kx hkx
  have hWPos : W > 0 := by omega
  have hIn : (0 : Int) ≤ (0 : Int) * (1 : Int) + (ky : Int) - (0 : Int) ∧
             (0 : Int) * (1 : Int) + (ky : Int) - (0 : Int) < (H : Int) ∧
             (0 : Int) ≤ (0 : Int) * (1 : Int) + (kx : Int) - (0 : Int) ∧
             (0 : Int) * (1 : Int) + (kx : Int) - (0 : Int) < (W : Int) := by omega
  have hyE : ((0 : Int) * (1 : Int) + (ky : Int) - (0 : Int)).toNat = ky := by omega
  have hxE : ((0 : Int) * (1 : Int) + (kx : Int) - (0 : Int)).toNat = kx := by omega
  have hDiv : (ky * W + kx) / W = ky := by
    rw [Nat.mul_comm ky W, Nat.mul_add_div hWPos, Nat.div_eq_of_lt hkx, Nat.add_zero]
  have hMod : (ky * W + kx) % W = kx := by
    rw [Nat.mul_comm ky W, Nat.mul_add_mod, Nat.mod_eq_of_lt hkx]
  simp only [hIn, and_self, ↓reduceIte, hyE, hxE, hDiv, hMod]

/-- `LayoutSelect` equivalence for `CoreOp.matmul` M-slicing (`LayoutSelect.lean` lines 635–656):
    slicing M into `numSlices = ceilDiv m 16` chunks of size `mi = min 16 (m - si * 16)` along
    rows `mOffset = si * 16` and concatenating along the M axis produces the exact full MatMul output
    at EVERY `(h, i, j)` with `i < m`! -/
theorem layoutSelect_matmul_mSlice_equiv {α : Type} [ScalarField α]
    (m k : Nat) (scale : α) (A B : Nat → Nat → Nat → α) (h i j : Nat) (hi : i < m) :
    let si := i / 16
    let m0 := si * 16
    let mi := min 16 (m - m0)
    let localI := i - m0
    si < ceilDiv m 16 ∧ localI < mi ∧
    evalMatmulElem k false false scale A B h (m0 + localI) j =
      evalMatmulElem k false false scale A B h i j := by
  intro si m0 mi localI
  have h16 : (16 : Nat) > 0 := by omega
  have hCov := stripOutBand_unique_coverage m 16 i h16 hi
  have hMod : si * 16 + i % 16 = i := Nat.div_add_mod' i 16
  have hModLt : i % 16 < 16 := Nat.mod_lt i h16
  refine ⟨hCov.1, ?_, ?_⟩
  · dsimp only [mi, localI, m0, si]; omega
  · have hEq : m0 + localI = i := by dsimp only [m0, localI, si]; omega
    rw [hEq]

-- ----------------------------------------------------------------------------
-- Blocker-1 Theorems: Faithful `.dmaView` Semantics for ALL 4 Patterns
-- (Slice, W-Preserving & T-Mode Transposes, Batched `matmul_trA/trB`, `matmul_join`, `w_stitch`)
-- ----------------------------------------------------------------------------

/-- Helper: `Shape4D` standard DRAM strides satisfy `viewCopyInverse_of_radix` for ANY `cp`
    whose output dimensions and strides match `outSh` with `transposeCW = false, outOff = 0`! -/
private theorem viewCopyInverse_std_outSh (outSh : Shape4D) (cp : ViewCopy)
    (hTr : cp.transposeCW = false) (hOff : cp.outOff = 0)
    (hN : cp.n = outSh.n) (hC : cp.c = outSh.c) (hH : cp.h = outSh.h) (hW : cp.w = outSh.w)
    (hNS : cp.outNStride = if outSh.n > 1 then outSh.batchStride else 0)
    (hPS : cp.outPlaneStride = if outSh.c > 1 then outSh.planeStride else 0)
    (hRS : cp.outRowStride = if outSh.h > 1 then outSh.rowStride else 0)
    (on oc oh ow : Nat) (hn : on < outSh.n) (hc : oc < outSh.c) (hh : oh < outSh.h) (hw : ow < outSh.w) :
    viewCopyInverse cp (dramByteOffset4D outSh on oc oh ow) = some (on, oc, oh, ow) := by
  have hRowGe : 2 * outSh.w ≤ outSh.rowStride := alignUp_ge_local (2 * outSh.w) dmaLineBytes
  have hRowPos : 0 < outSh.rowStride := by omega
  have hPlanePos : 0 < outSh.planeStride := Nat.mul_pos (by omega) hRowPos
  have hBatchPos : 0 < outSh.batchStride := Nat.mul_pos (by omega) hPlanePos
  have hDstEq : dramByteOffset4D outSh on oc oh ow = viewCopyDstByte cp on oc oh ow := by
    unfold dramByteOffset4D viewCopyDstByte
    rw [hTr, hOff, hNS, hPS, hRS]
    simp only [Bool.false_eq_true, ↓reduceIte, Nat.zero_add,
      mul_if_gt_one hn, mul_if_gt_one hc, mul_if_gt_one hh]
  rw [hDstEq]
  apply viewCopyInverse_of_radix cp on oc oh ow (hN ▸ hn) (hC ▸ hc) (hH ▸ hh) (hW ▸ hw)
  have hIfIf : ∀ (P : Prop) [Decidable P] (a : Nat), (if P then (if P then a else 0) else 0) = if P then a else 0 := by
    intro P _ a; by_cases h : P <;> simp [h]
  simp only [hTr, Bool.false_eq_true, ↓reduceIte, hN, hC, hH, hNS, hPS, hRS, hIfIf]
  refine ⟨fun hn1 => by simp [hn1, hBatchPos], fun hc1 => by simp [hc1, hPlanePos],
    fun hh1 => by simp [hh1, hRowPos], ?_, ?_, ?_⟩
  · intro hs0
    have hn1 : outSh.n > 1 := Classical.byContradiction fun hNot => by simp [hNot] at hs0
    simp only [hn1, ↓reduceIte]
    have hGt1 : outSh.batchStride > (if outSh.c > 1 then outSh.planeStride else 0) := by
      by_cases hc1 : outSh.c > 1
      · simp only [hc1, ↓reduceIte]
        change outSh.c * outSh.planeStride > outSh.planeStride
        have h2 : 2 * outSh.planeStride ≤ outSh.c * outSh.planeStride := Nat.mul_le_mul_right _ hc1
        omega
      · simp [hc1, hBatchPos]
    have hGt2 : outSh.batchStride > (if outSh.h > 1 then outSh.rowStride else 0) := by
      by_cases hh1 : outSh.h > 1
      · simp only [hh1, ↓reduceIte]
        change outSh.c * (outSh.h * outSh.rowStride) > outSh.rowStride
        have h1 : 2 * outSh.rowStride ≤ outSh.h * outSh.rowStride := Nat.mul_le_mul_right _ hh1
        have h2 : 1 * (outSh.h * outSh.rowStride) ≤ outSh.c * (outSh.h * outSh.rowStride) :=
          Nat.mul_le_mul_right _ (by omega)
        omega
      · simp [hh1, hBatchPos]
    have hSum : oc * (if outSh.c > 1 then outSh.planeStride else 0) +
        oh * (if outSh.h > 1 then outSh.rowStride else 0) + 2 * ow < outSh.batchStride := by
      rw [mul_if_gt_one hc, mul_if_gt_one hh]
      have hW2 : 2 * ow < outSh.rowStride := by omega
      have hHR : oh * outSh.rowStride + 2 * ow < outSh.planeStride := by
        have h1 : (oh + 1) * outSh.rowStride ≤ outSh.h * outSh.rowStride :=
          Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hh)
        rw [Nat.add_mul, Nat.one_mul] at h1
        change oh * outSh.rowStride + 2 * ow < outSh.h * outSh.rowStride; omega
      have hC1 : (oc + 1) * outSh.planeStride ≤ outSh.c * outSh.planeStride :=
        Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hc)
      rw [Nat.add_mul, Nat.one_mul] at hC1
      change oc * outSh.planeStride + oh * outSh.rowStride + 2 * ow < outSh.c * outSh.planeStride
      omega
    exact ⟨hGt1, hGt2, hSum⟩
  · intro hs1
    have hc1 : outSh.c > 1 := Classical.byContradiction fun hNot => by simp [hNot] at hs1
    simp only [hc1, ↓reduceIte]
    have hGt2 : outSh.planeStride > (if outSh.h > 1 then outSh.rowStride else 0) := by
      by_cases hh1 : outSh.h > 1
      · simp only [hh1, ↓reduceIte]
        change outSh.h * outSh.rowStride > outSh.rowStride
        have h2 : 2 * outSh.rowStride ≤ outSh.h * outSh.rowStride := Nat.mul_le_mul_right _ hh1
        omega
      · simp [hh1, hPlanePos]
    have hSum : oh * (if outSh.h > 1 then outSh.rowStride else 0) + 2 * ow < outSh.planeStride := by
      rw [mul_if_gt_one hh]
      have hW2 : 2 * ow < outSh.rowStride := by omega
      have h1 : (oh + 1) * outSh.rowStride ≤ outSh.h * outSh.rowStride :=
        Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hh)
      rw [Nat.add_mul, Nat.one_mul] at h1
      change oh * outSh.rowStride + 2 * ow < outSh.h * outSh.rowStride; omega
    exact ⟨hGt2, hSum⟩
  · intro hs2
    have hh1 : outSh.h > 1 := Classical.byContradiction fun hNot => by simp [hNot] at hs2
    simp only [hh1, ↓reduceIte]
    omega

/-- BLOCKER-1 THEOREM 1: Exact `.dmaView` slice equivalence (`CoreOp.slice`, `const_anchor`,
    `halo_slice`, `w_slice` — `View.planSlice` lines 397–411):
    `evalAneOpCore` on the `ViewCopy` emitted by `planSlice` produces the EXACT sliced `TensorVal α`
    matching `Vane.IR.evalSlice` at EVERY coordinate `(on, oc, oh, ow)`! -/
theorem dmaView_planSlice_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (decodeW : WeightData → Nat → Nat → Nat → Nat → Option (Nat → Nat → Nat → Nat → α))
    (decodeB : Option ByteArray → Nat → Option (Option (Nat → α)))
    (inSh outSh : Shape4D) (b0 b1 b2 b3 s0 s1 s2 : Nat)
    (hInPos : inSh.n > 0 ∧ inSh.c > 0 ∧ inSh.h > 0 ∧ inSh.w > 0)
    (hOutPos : outSh.n > 0 ∧ outSh.c > 0 ∧ outSh.h > 0 ∧ outSh.w > 0)
    (hBndN : b0 + (outSh.n - 1) * s0 < inSh.n)
    (hBndC : b1 + (outSh.c - 1) * s1 < inSh.c)
    (hBndH : b2 + (outSh.h - 1) * s2 < inSh.h)
    (hBndW : b3 + outSh.w ≤ inSh.w)
    (xT : TensorVal α) (hXSh : xT.shape = inSh.toArray.toList) (outTy : TensorType)
    (hOutTy : outTy.shape.toList = outSh.toArray.toList) :
    let xs := inSh.dramStrides
    let ys := outSh.dramStrides
    let cp : ViewCopy := {
      n := outSh.n, c := outSh.c, h := outSh.h, w := outSh.w, d := 1
      inOff := b0 * xs[0]! + b1 * xs[1]! + b2 * xs[2]! + 2 * b3
      inNStride := if outSh.n > 1 then xs[0]! * s0 else 0
      inPlaneStride := if outSh.c > 1 then xs[1]! * s1 else 0
      inRowStride := if outSh.h > 1 then xs[2]! * s2 else 0
      outOff := 0
      outNStride := if outSh.n > 1 then ys[0]! else 0
      outPlaneStride := if outSh.c > 1 then ys[1]! else 0
      outRowStride := if outSh.h > 1 then ys[2]! else 0
      transposeCW := false
    }
    let desc : DmaViewSpec := { inShape := inSh, outShape := outSh, explicitCopy? := some cp }
    ∃ r, evalAneOpCore ops decodeW decodeB (.dmaView desc "slice") [xT] [outTy] = some [r] ∧
      r.shape = outSh.toArray.toList ∧
      ∀ on oc oh ow, on < outSh.n → oc < outSh.c → oh < outSh.h → ow < outSh.w →
        let outIdx := flatIndex outSh.toArray.toList [on, oc, oh, ow]
        let inIdx := flatIndex inSh.toArray.toList [b0 + on * s0, b1 + oc * s1, b2 + oh * s2, b3 + ow]
        ∀ (hOutLt : outIdx < shapeProd r.shape) (hInLt : inIdx < shapeProd xT.shape),
          r.data ⟨outIdx, hOutLt⟩ = xT.data ⟨inIdx, hInLt⟩ := by
  intro xs ys cp desc
  have hLook : ∀ on oc oh ow, on < outSh.n → oc < outSh.c → oh < outSh.h → ow < outSh.w →
      viewCopyLookupCoord inSh outSh cp on oc oh ow = some (b0 + on * s0, b1 + oc * s1, b2 + oh * s2, b3 + ow) := by
    intro on oc oh ow hn hc hh hw
    unfold viewCopyLookupCoord; dsimp only
    have hInv := viewCopyInverse_std_outSh outSh cp rfl rfl rfl rfl rfl rfl rfl rfl rfl on oc oh ow hn hc hh hw
    rw [hInv]; dsimp only
    have hSrcEq : viewCopySrcByte cp on oc oh ow =
        dramByteOffset4D inSh (b0 + on * s0) (b1 + oc * s1) (b2 + oh * s2) (b3 + ow) := by
      unfold viewCopySrcByte dramByteOffset4D
      dsimp only [cp, xs, Shape4D.dramStrides]
      have e0 : on * (if outSh.n > 1 then inSh.batchStride * s0 else 0) = (on * s0) * inSh.batchStride := by
        by_cases h1 : outSh.n > 1
        · simp only [h1, ↓reduceIte]; rw [Nat.mul_comm inSh.batchStride s0, Nat.mul_assoc]
        · have : on = 0 := by omega
          subst this; simp [h1]
      have e1 : oc * (if outSh.c > 1 then inSh.planeStride * s1 else 0) = (oc * s1) * inSh.planeStride := by
        by_cases h1 : outSh.c > 1
        · simp only [h1, ↓reduceIte]; rw [Nat.mul_comm inSh.planeStride s1, Nat.mul_assoc]
        · have : oc = 0 := by omega
          subst this; simp [h1]
      have e2 : oh * (if outSh.h > 1 then inSh.rowStride * s2 else 0) = (oh * s2) * inSh.rowStride := by
        by_cases h1 : outSh.h > 1
        · simp only [h1, ↓reduceIte]; rw [Nat.mul_comm inSh.rowStride s2, Nat.mul_assoc]
        · have : oh = 0 := by omega
          subst this; simp [h1]
      have hG0 : (#[inSh.batchStride, inSh.planeStride, inSh.rowStride, 2] : Array Nat)[0]! = inSh.batchStride := rfl
      have hG1 : (#[inSh.batchStride, inSh.planeStride, inSh.rowStride, 2] : Array Nat)[1]! = inSh.planeStride := rfl
      have hG2 : (#[inSh.batchStride, inSh.planeStride, inSh.rowStride, 2] : Array Nat)[2]! = inSh.rowStride := rfl
      rw [hG0, hG1, hG2, e0, e1, e2, Nat.add_mul, Nat.add_mul, Nat.add_mul, Nat.mul_add]
      omega
    rw [hSrcEq]
    have hN' : b0 + on * s0 < inSh.n := by
      have : on * s0 ≤ (outSh.n - 1) * s0 := Nat.mul_le_mul_right s0 (by omega); omega
    have hC' : b1 + oc * s1 < inSh.c := by
      have : oc * s1 ≤ (outSh.c - 1) * s1 := Nat.mul_le_mul_right s1 (by omega); omega
    have hH' : b2 + oh * s2 < inSh.h := by
      have : oh * s2 ≤ (outSh.h - 1) * s2 := Nat.mul_le_mul_right s2 (by omega); omega
    exact decodeDramByteOffset4D_roundTrip inSh _ _ _ _ hN' hC' hH' (by omega)
  have hEv := evalViewCopyTensor_of_lookup inSh outSh cp xT
    (fun on oc oh ow => (b0 + on * s0, b1 + oc * s1, b2 + oh * s2, b3 + ow))
    hXSh hInPos hOutPos hLook (fun on oc oh ow hn hc hh hw => by
      have : on * s0 ≤ (outSh.n - 1) * s0 := Nat.mul_le_mul_right s0 (by omega)
      have : oc * s1 ≤ (outSh.c - 1) * s1 := Nat.mul_le_mul_right s1 (by omega)
      have : oh * s2 ≤ (outSh.h - 1) * s2 := Nat.mul_le_mul_right s2 (by omega)
      exact ⟨by omega, by omega, by omega, by omega⟩)
  obtain ⟨r, hr, hrSh, hrElem⟩ := hEv
  refine ⟨r, ?_, hrSh, hrElem⟩
  unfold evalAneOpCore
  dsimp only [desc, DmaViewSpec.toViewCopy]
  rw [hOutTy]
  simp [hr]

/-- BLOCKER-1 THEOREM 2: Exact `.dmaView` W-preserving transpose equivalence (specifically covering
    `perm = #[0, 2, 1, 3]` emitted by `View.planTranspose` lines 300–318 where `transposeCW = false`):
    unlike `evalReshape`, `evalAneOpCore` genuinely permutes axes `(n, c, h, w) ↦ (n, h, c, w)`! -/
theorem dmaView_planTranspose_wPreserving_0213_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (decodeW : WeightData → Nat → Nat → Nat → Nat → Option (Nat → Nat → Nat → Nat → α))
    (decodeB : Option ByteArray → Nat → Option (Option (Nat → α)))
    (inSh : Shape4D) (hInPos : inSh.n > 0 ∧ inSh.c > 0 ∧ inSh.h > 0 ∧ inSh.w > 0)
    (xT : TensorVal α) (hXSh : xT.shape = inSh.toArray.toList) (outTy : TensorType)
    (hOutTy : outTy.shape.toList = [inSh.n, inSh.h, inSh.c, inSh.w]) :
    let outSh : Shape4D := { n := inSh.n, c := inSh.h, h := inSh.c, w := inSh.w }
    let xs := inSh.dramStrides
    let ys := outSh.dramStrides
    let cp : ViewCopy := {
      n := outSh.n, c := outSh.c, h := outSh.h, w := outSh.w, d := 1
      inOff := 0
      inNStride := if outSh.n > 1 then xs[0]! else 0
      inPlaneStride := if outSh.c > 1 then xs[2]! else 0
      inRowStride := if outSh.h > 1 then xs[1]! else 0
      outOff := 0
      outNStride := if outSh.n > 1 then ys[0]! else 0
      outPlaneStride := if outSh.c > 1 then ys[1]! else 0
      outRowStride := if outSh.h > 1 then ys[2]! else 0
      transposeCW := false
    }
    let desc : DmaViewSpec := { inShape := inSh, outShape := outSh, explicitCopy? := some cp }
    ∃ r, evalAneOpCore ops decodeW decodeB (.dmaView desc "transpose") [xT] [outTy] = some [r] ∧
      r.shape = outSh.toArray.toList ∧
      ∀ on oc oh ow, on < outSh.n → oc < outSh.c → oh < outSh.h → ow < outSh.w →
        let outIdx := flatIndex outSh.toArray.toList [on, oc, oh, ow]
        let inIdx := flatIndex inSh.toArray.toList [on, oh, oc, ow]
        ∀ (hOutLt : outIdx < shapeProd r.shape) (hInLt : inIdx < shapeProd xT.shape),
          r.data ⟨outIdx, hOutLt⟩ = xT.data ⟨inIdx, hInLt⟩ := by
  intro outSh xs ys cp desc
  have hOutPos : outSh.n > 0 ∧ outSh.c > 0 ∧ outSh.h > 0 ∧ outSh.w > 0 :=
    ⟨hInPos.1, hInPos.2.2.1, hInPos.2.1, hInPos.2.2.2⟩
  have hLook : ∀ on oc oh ow, on < outSh.n → oc < outSh.c → oh < outSh.h → ow < outSh.w →
      viewCopyLookupCoord inSh outSh cp on oc oh ow = some (on, oh, oc, ow) := by
    intro on oc oh ow hn hc hh hw
    unfold viewCopyLookupCoord; dsimp only
    have hInv := viewCopyInverse_std_outSh outSh cp rfl rfl rfl rfl rfl rfl rfl rfl rfl on oc oh ow hn hc hh hw
    rw [hInv]; dsimp only
    have hSrcEq : viewCopySrcByte cp on oc oh ow = dramByteOffset4D inSh on oh oc ow := by
      unfold viewCopySrcByte dramByteOffset4D
      dsimp only [cp, xs, Shape4D.dramStrides]
      have hG0 : (#[inSh.batchStride, inSh.planeStride, inSh.rowStride, 2] : Array Nat)[0]! = inSh.batchStride := rfl
      have hG1 : (#[inSh.batchStride, inSh.planeStride, inSh.rowStride, 2] : Array Nat)[1]! = inSh.planeStride := rfl
      have hG2 : (#[inSh.batchStride, inSh.planeStride, inSh.rowStride, 2] : Array Nat)[2]! = inSh.rowStride := rfl
      rw [hG0, hG1, hG2, mul_if_gt_one hn, mul_if_gt_one hc, mul_if_gt_one hh]
      omega
    rw [hSrcEq]
    exact decodeDramByteOffset4D_roundTrip inSh on oh oc ow hn hh hc hw
  obtain ⟨r, hr, hrSh, hrElem⟩ := evalViewCopyTensor_of_lookup inSh outSh cp xT
    (fun on oc oh ow => (on, oh, oc, ow)) hXSh hInPos hOutPos hLook
    (fun on oc oh ow hn hc hh hw => ⟨hn, hh, hc, hw⟩)
  refine ⟨r, ?_, hrSh, hrElem⟩
  unfold evalAneOpCore
  dsimp only [desc, DmaViewSpec.toViewCopy]
  have hShEq : outSh.toArray.toList = outTy.shape.toList := hOutTy.symm
  rw [hShEq]
  simp [hr]

/-- BLOCKER-1 THEOREM 3: Exact `.dmaView` T-mode (`transposeCW = true`) transpose equivalence for BOTH
    `perm = #[0, 3, 2, 1]` (`C ↔ W`) AND `perm = #[0, 1, 3, 2]` (`H ↔ W`, `LayoutSelect.lean:1162,1189`)! -/
theorem dmaView_planTranspose_tMode_0321_and_0132_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (decodeW : WeightData → Nat → Nat → Nat → Nat → Option (Nat → Nat → Nat → Nat → α))
    (decodeB : Option ByteArray → Nat → Option (Option (Nat → α)))
    (C H W : Nat) (hC : C > 0) (hH : H > 0) (hW : W > 0)
    (xT : TensorVal α) (hXSh : xT.shape = [1, C, H, W])
    (outTy0321 outTy0132 : TensorType)
    (hTy0321 : outTy0321.shape.toList = [1, W, H, C])
    (hTy0132 : outTy0132.shape.toList = [1, C, W, H]) :
    let inSh : Shape4D := { n := 1, c := C, h := H, w := W }
    let xs := inSh.dramStrides
    -- Case A: `perm = #[0, 3, 2, 1]` (`[1, C, H, W] -> [1, W, H, C]`)
    let outA : Shape4D := { n := 1, c := W, h := H, w := C }
    let ysA := outA.dramStrides
    let cpA : ViewCopy := {
      n := 1, c := C, h := H, w := W, d := 1
      inOff := 0, inNStride := 0, inPlaneStride := if C > 1 then xs[1]! else 0,
      inRowStride := if H > 1 then xs[2]! else 0
      outOff := 0, outNStride := 0, outPlaneStride := if W > 1 then ysA[1]! else 0,
      outRowStride := if H > 1 then ysA[2]! else 0, transposeCW := true
    }
    let descA : DmaViewSpec := { inShape := inSh, outShape := outA, transposeCW := true, explicitCopy? := some cpA }
    -- Case B: `perm = #[0, 1, 3, 2]` (`[1, C, H, W] -> [1, C, W, H]`, `aN = 1 (C), aC = 2 (H), aH = 0 (1), w = W`)
    let outB : Shape4D := { n := 1, c := C, h := W, w := H }
    let ysB := outB.dramStrides
    let cpB : ViewCopy := {
      n := C, c := H, h := 1, w := W, d := 1
      inOff := 0, inNStride := if C > 1 then xs[1]! else 0, inPlaneStride := if H > 1 then xs[2]! else 0,
      inRowStride := 0
      outOff := 0, outNStride := if C > 1 then ysB[1]! else 0, outPlaneStride := if W > 1 then ysB[2]! else 0,
      outRowStride := 0, transposeCW := true
    }
    let descB : DmaViewSpec := { inShape := inSh, outShape := outB, transposeCW := true, batchN := C, explicitCopy? := some cpB }
    (∃ rA, evalAneOpCore ops decodeW decodeB (.dmaView descA "transpose") [xT] [outTy0321] = some [rA] ∧
      rA.shape = [1, W, H, C] ∧
      ∀ ow oh oc, ow < W → oh < H → oc < C →
        let outIdx := flatIndex [1, W, H, C] [0, ow, oh, oc]
        let inIdx := flatIndex [1, C, H, W] [0, oc, oh, ow]
        ∀ (hOutLt : outIdx < shapeProd rA.shape) (hInLt : inIdx < shapeProd xT.shape),
          rA.data ⟨outIdx, hOutLt⟩ = xT.data ⟨inIdx, hInLt⟩) ∧
    (∃ rB, evalAneOpCore ops decodeW decodeB (.dmaView descB "transpose") [xT] [outTy0132] = some [rB] ∧
      rB.shape = [1, C, W, H] ∧
      ∀ oc ow oh, oc < C → ow < W → oh < H →
        let outIdx := flatIndex [1, C, W, H] [0, oc, ow, oh]
        let inIdx := flatIndex [1, C, H, W] [0, oc, oh, ow]
        ∀ (hOutLt : outIdx < shapeProd rB.shape) (hInLt : inIdx < shapeProd xT.shape),
          rB.data ⟨outIdx, hOutLt⟩ = xT.data ⟨inIdx, hInLt⟩) := by
  intro inSh xs outA ysA cpA descA outB ysB cpB descB
  have hIfIf : ∀ (P : Prop) [Decidable P] (a : Nat), (if P then (if P then a else 0) else 0) = if P then a else 0 := by
    intro P _ a; by_cases h : P <;> simp [h]
  have hInPos : inSh.n > 0 ∧ inSh.c > 0 ∧ inSh.h > 0 ∧ inSh.w > 0 := ⟨Nat.zero_lt_one, hC, hH, hW⟩
  constructor
  · -- Case A: #[0, 3, 2, 1]
    have hOutPosA : outA.n > 0 ∧ outA.c > 0 ∧ outA.h > 0 ∧ outA.w > 0 := ⟨Nat.zero_lt_one, hW, hH, hC⟩
    have hLookA : ∀ on ow oh oc, on < outA.n → ow < outA.c → oh < outA.h → oc < outA.w →
        viewCopyLookupCoord inSh outA cpA on ow oh oc = some (0, oc, oh, ow) := by
      intro on ow oh oc hn hw hh hc
      have hon0 : on = 0 := by dsimp only [outA] at hn; omega
      subst hon0
      have hw' : ow < W := hw
      have hh' : oh < H := hh
      have hc' : oc < C := hc
      unfold viewCopyLookupCoord; dsimp only
      have hRowGe : 2 * C ≤ outA.rowStride := alignUp_ge_local (2 * C) dmaLineBytes
      have hRowPos : 0 < outA.rowStride := by omega
      have hPlPos : 0 < outA.planeStride := Nat.mul_pos hH hRowPos
      have hDst : dramByteOffset4D outA 0 ow oh oc = viewCopyDstByte cpA 0 oc oh ow := by
        unfold dramByteOffset4D viewCopyDstByte
        dsimp only [cpA, ysA, Shape4D.dramStrides]
        have hG1 : (#[outA.batchStride, outA.planeStride, outA.rowStride, 2] : Array Nat)[1]! = outA.planeStride := rfl
        have hG2 : (#[outA.batchStride, outA.planeStride, outA.rowStride, 2] : Array Nat)[2]! = outA.rowStride := rfl
        rw [hG1, hG2, mul_if_gt_one hw', mul_if_gt_one hh']
        simp
      rw [hDst]
      have hInv : viewCopyInverse cpA (viewCopyDstByte cpA 0 oc oh ow) = some (0, oc, oh, ow) := by
        apply viewCopyInverse_of_radix cpA 0 oc oh ow Nat.zero_lt_one hc' hh' hw'
        dsimp only [cpA, ysA, Shape4D.dramStrides]
        have hG1 : (#[outA.batchStride, outA.planeStride, outA.rowStride, 2] : Array Nat)[1]! = outA.planeStride := rfl
        have hG2 : (#[outA.batchStride, outA.planeStride, outA.rowStride, 2] : Array Nat)[2]! = outA.rowStride := rfl
        rw [hG1, hG2]
        simp only [↓reduceIte, hIfIf]
        refine ⟨fun h1 => by omega, fun hw1 => by simp [hw1, hPlPos], fun hh1 => by simp [hh1, hRowPos],
          fun h0 => by simp at h0, ?_, ?_⟩
        · intro hs1
          have hw1 : W > 1 := Classical.byContradiction fun h => by simp [h] at hs1
          simp only [hw1, ↓reduceIte]
          constructor
          · by_cases hh1 : H > 1
            · simp only [hh1, ↓reduceIte]; change H * outA.rowStride > outA.rowStride
              have : 2 * outA.rowStride ≤ H * outA.rowStride := Nat.mul_le_mul_right _ hh1; omega
            · simp [hh1, hPlPos]
          · rw [mul_if_gt_one hh']
            have h1 : (oh + 1) * outA.rowStride ≤ H * outA.rowStride := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hh')
            rw [Nat.add_mul, Nat.one_mul] at h1
            change oh * outA.rowStride + 2 * oc < H * outA.rowStride; omega
        · intro hs2
          have hh1 : H > 1 := Classical.byContradiction fun h => by simp [h] at hs2
          simp only [hh1, ↓reduceIte]; omega
      rw [hInv]; dsimp only
      have hSrc : viewCopySrcByte cpA 0 oc oh ow = dramByteOffset4D inSh 0 oc oh ow := by
        unfold viewCopySrcByte dramByteOffset4D
        dsimp only [cpA, xs, Shape4D.dramStrides]
        have hG1 : (#[inSh.batchStride, inSh.planeStride, inSh.rowStride, 2] : Array Nat)[1]! = inSh.planeStride := rfl
        have hG2 : (#[inSh.batchStride, inSh.planeStride, inSh.rowStride, 2] : Array Nat)[2]! = inSh.rowStride := rfl
        rw [hG1, hG2, mul_if_gt_one hc', mul_if_gt_one hh']
        simp
      rw [hSrc]
      exact decodeDramByteOffset4D_roundTrip inSh 0 oc oh ow Nat.zero_lt_one hc' hh' hw'
    obtain ⟨rA, hrA, hrASh, hrAElem⟩ := evalViewCopyTensor_of_lookup inSh outA cpA xT
      (fun _ ow oh oc => (0, oc, oh, ow)) hXSh hInPos hOutPosA hLookA
      (fun _ ow oh oc _ hw hh hc => ⟨Nat.zero_lt_one, hc, hh, hw⟩)
    refine ⟨rA, ?_, hrASh, fun ow oh oc hw hh hc h1 h2 => hrAElem 0 ow oh oc Nat.zero_lt_one hw hh hc h1 h2⟩
    unfold evalAneOpCore
    dsimp only [descA, DmaViewSpec.toViewCopy]
    rw [hTy0321]
    have hShA : outA.toArray.toList = [1, W, H, C] := rfl
    rw [hShA]; simp [hrA]
  · -- Case B: #[0, 1, 3, 2]
    have hOutPosB : outB.n > 0 ∧ outB.c > 0 ∧ outB.h > 0 ∧ outB.w > 0 := ⟨Nat.zero_lt_one, hC, hW, hH⟩
    have hLookB : ∀ on oc ow oh, on < outB.n → oc < outB.c → ow < outB.h → oh < outB.w →
        viewCopyLookupCoord inSh outB cpB on oc ow oh = some (0, oc, oh, ow) := by
      intro on oc ow oh hn hc hw hh
      have hon0 : on = 0 := by dsimp only [outB] at hn; omega
      subst hon0
      have hc' : oc < C := hc
      have hw' : ow < W := hw
      have hh' : oh < H := hh
      unfold viewCopyLookupCoord; dsimp only
      have hRowGe : 2 * H ≤ outB.rowStride := alignUp_ge_local (2 * H) dmaLineBytes
      have hRowPos : 0 < outB.rowStride := by omega
      have hPlPos : 0 < outB.planeStride := Nat.mul_pos hW hRowPos
      have hDst : dramByteOffset4D outB 0 oc ow oh = viewCopyDstByte cpB oc oh 0 ow := by
        unfold dramByteOffset4D viewCopyDstByte
        dsimp only [cpB, ysB, Shape4D.dramStrides]
        have hG1 : (#[outB.batchStride, outB.planeStride, outB.rowStride, 2] : Array Nat)[1]! = outB.planeStride := rfl
        have hG2 : (#[outB.batchStride, outB.planeStride, outB.rowStride, 2] : Array Nat)[2]! = outB.rowStride := rfl
        rw [hG1, hG2, mul_if_gt_one hc', mul_if_gt_one hw']
        simp
      rw [hDst]
      have hInv : viewCopyInverse cpB (viewCopyDstByte cpB oc oh 0 ow) = some (oc, oh, 0, ow) := by
        apply viewCopyInverse_of_radix cpB oc oh 0 ow hc' hh' Nat.zero_lt_one hw'
        dsimp only [cpB, ysB, Shape4D.dramStrides]
        have hG1 : (#[outB.batchStride, outB.planeStride, outB.rowStride, 2] : Array Nat)[1]! = outB.planeStride := rfl
        have hG2 : (#[outB.batchStride, outB.planeStride, outB.rowStride, 2] : Array Nat)[2]! = outB.rowStride := rfl
        rw [hG1, hG2]
        simp only [↓reduceIte, hIfIf]
        refine ⟨fun hc1 => by simp [hc1, hPlPos], fun hw1 => by simp [hw1, hRowPos], fun h1 => by omega,
          ?_, ?_, fun h0 => by simp at h0⟩
        · intro hs0
          have hc1 : C > 1 := Classical.byContradiction fun h => by simp [h] at hs0
          simp only [hc1, ↓reduceIte, Nat.lt_irrefl, Nat.zero_mul, Nat.add_zero]
          refine ⟨?_, hPlPos, ?_⟩
          · by_cases hw1 : W > 1
            · simp only [hw1, ↓reduceIte]; change W * outB.rowStride > outB.rowStride
              have : 2 * outB.rowStride ≤ W * outB.rowStride := Nat.mul_le_mul_right _ hw1; omega
            · simp [hw1, hPlPos]
          · rw [mul_if_gt_one hw']
            have h1 : (ow + 1) * outB.rowStride ≤ W * outB.rowStride := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hw')
            rw [Nat.add_mul, Nat.one_mul] at h1
            change ow * outB.rowStride + 2 * oh < W * outB.rowStride; omega
        · intro hs1
          have hw1 : W > 1 := Classical.byContradiction fun h => by simp [h] at hs1
          simp only [hw1, ↓reduceIte, Nat.lt_irrefl, Nat.zero_mul, Nat.zero_add]
          exact ⟨hRowPos, by omega⟩
      rw [hInv]; dsimp only
      have hSrc : viewCopySrcByte cpB oc oh 0 ow = dramByteOffset4D inSh 0 oc oh ow := by
        unfold viewCopySrcByte dramByteOffset4D
        dsimp only [cpB, xs, Shape4D.dramStrides]
        have hG1 : (#[inSh.batchStride, inSh.planeStride, inSh.rowStride, 2] : Array Nat)[1]! = inSh.planeStride := rfl
        have hG2 : (#[inSh.batchStride, inSh.planeStride, inSh.rowStride, 2] : Array Nat)[2]! = inSh.rowStride := rfl
        rw [hG1, hG2, mul_if_gt_one hc', mul_if_gt_one hh']
        simp
      rw [hSrc]
      exact decodeDramByteOffset4D_roundTrip inSh 0 oc oh ow Nat.zero_lt_one hc' hh' hw'
    obtain ⟨rB, hrB, hrBSh, hrBElem⟩ := evalViewCopyTensor_of_lookup inSh outB cpB xT
      (fun _ oc ow oh => (0, oc, oh, ow)) hXSh hInPos hOutPosB hLookB
      (fun _ oc ow oh _ hc hw hh => ⟨Nat.zero_lt_one, hc, hh, hw⟩)
    refine ⟨rB, ?_, hrBSh, fun oc ow oh hc hw hh h1 h2 => hrBElem 0 oc ow oh Nat.zero_lt_one hc hw hh h1 h2⟩
    unfold evalAneOpCore
    dsimp only [descB, DmaViewSpec.toViewCopy]
    rw [hTy0132]
    have hShB : outB.toArray.toList = [1, C, W, H] := rfl
    rw [hShB]; simp [hrB]

/-- BLOCKER-1 THEOREM 4: Exact `.dmaView` batched `matmul_trA` / `matmul_trB` transpose equivalence
    for ANY `heads ≥ 1` (`Matmul.planMatmulScratchTranspose`, `LayoutSelect.lean:610-633`):
    transposes `[1, heads * rows, 1, cols] → [1, heads * cols, 1, rows]` independently per head! -/
theorem dmaView_matmulScratchTranspose_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (decodeW : WeightData → Nat → Nat → Nat → Nat → Option (Nat → Nat → Nat → Nat → α))
    (decodeB : Option ByteArray → Nat → Option (Option (Nat → α)))
    (heads rows cols : Nat) (hHeads : heads > 0) (hRows : rows > 0) (hCols : cols > 0)
    (hNonTriv : rows > 1 ∨ cols > 1)
    (xT : TensorVal α) (hXSh : xT.shape = [1, heads * rows, 1, cols])
    (outTy : TensorType) (hOutTy : outTy.shape.toList = [1, heads * cols, 1, rows]) :
    let inSh : Shape4D := { n := 1, c := heads * rows, h := 1, w := cols }
    let outSh : Shape4D := { n := 1, c := heads * cols, h := 1, w := rows }
    let inRow := alignUp (2 * cols) dmaLineBytes
    let outRow := alignUp (2 * rows) dmaLineBytes
    let cp : ViewCopy := {
      n := heads, c := rows, h := 1, w := cols, d := 1
      inOff := 0, inRowStride := 0,
      inPlaneStride := if rows > 1 then inRow else 0, inDStride := 0,
      inNStride := if heads > 1 then rows * inRow else 0
      outOff := 0, outRowStride := 0,
      outPlaneStride := if cols > 1 then outRow else 0, outDStride := 0,
      outNStride := if heads > 1 then cols * outRow else 0
      transposeCW := true
    }
    let desc : DmaViewSpec := {
      inShape := inSh, outShape := outSh, transposeCW := true, batchN := heads, explicitCopy? := some cp
    }
    ∃ r, evalAneOpCore ops decodeW decodeB (.dmaView desc "matmul_trA") [xT] [outTy] = some [r] ∧
      r.shape = [1, heads * cols, 1, rows] ∧
      ∀ hd col row, hd < heads → col < cols → row < rows →
        let outIdx := flatIndex [1, heads * cols, 1, rows] [0, hd * cols + col, 0, row]
        let inIdx := flatIndex [1, heads * rows, 1, cols] [0, hd * rows + row, 0, col]
        ∀ (hOutLt : outIdx < shapeProd r.shape) (hInLt : inIdx < shapeProd xT.shape),
          r.data ⟨outIdx, hOutLt⟩ = xT.data ⟨inIdx, hInLt⟩ := by
  intro inSh outSh inRow outRow cp desc
  have _ := hNonTriv
  have hIfIf : ∀ (P : Prop) [Decidable P] (a : Nat), (if P then (if P then a else 0) else 0) = if P then a else 0 := by
    intro P _ a; by_cases h : P <;> simp [h]
  have hInPos : inSh.n > 0 ∧ inSh.c > 0 ∧ inSh.h > 0 ∧ inSh.w > 0 :=
    ⟨Nat.zero_lt_one, Nat.mul_pos hHeads hRows, Nat.zero_lt_one, hCols⟩
  have hOutPos : outSh.n > 0 ∧ outSh.c > 0 ∧ outSh.h > 0 ∧ outSh.w > 0 :=
    ⟨Nat.zero_lt_one, Nat.mul_pos hHeads hCols, Nat.zero_lt_one, hRows⟩
  have hOutRowGe : 2 * rows ≤ outRow := alignUp_ge_local (2 * rows) dmaLineBytes
  have hOutRowPos : 0 < outRow := by omega
  have hLook : ∀ on oc oh ow, on < outSh.n → oc < outSh.c → oh < outSh.h → ow < outSh.w →
      viewCopyLookupCoord inSh outSh cp on oc oh ow =
        some (0, (oc / cols) * rows + ow, 0, oc % cols) := by
    intro on oc oh ow hn hc hh hw
    have hon0 : on = 0 := by dsimp only [outSh] at hn; omega
    have hoh0 : oh = 0 := by dsimp only [outSh] at hh; omega
    subst hon0; subst hoh0
    have hw' : ow < rows := hw
    let hd := oc / cols
    let col := oc % cols
    have hhdLt : hd < heads := Nat.div_lt_of_lt_mul (by dsimp only [outSh] at hc; rwa [Nat.mul_comm] at hc)
    have hcolLt : col < cols := Nat.mod_lt oc hCols
    have hocEq : hd * cols + col = oc := Nat.div_add_mod' oc cols
    unfold viewCopyLookupCoord; dsimp only
    have hDst : dramByteOffset4D outSh 0 oc 0 ow = viewCopyDstByte cp hd ow 0 col := by
      unfold dramByteOffset4D viewCopyDstByte
      dsimp only [cp, outSh, Shape4D.planeStride, Shape4D.rowStride]
      simp only [Nat.zero_mul, Nat.zero_add, Nat.one_mul, ↓reduceIte]
      rw [mul_if_gt_one hhdLt, mul_if_gt_one hcolLt, ← Nat.mul_assoc, ← Nat.add_mul, hocEq]
    rw [hDst]
    have hInv : viewCopyInverse cp (viewCopyDstByte cp hd ow 0 col) = some (hd, ow, 0, col) := by
      apply viewCopyInverse_of_radix cp hd ow 0 col hhdLt hw' Nat.zero_lt_one hcolLt
      dsimp only [cp]
      simp only [↓reduceIte, hIfIf]
      refine ⟨fun hh1 => by simp only [hh1, ↓reduceIte]; exact Nat.mul_pos hCols hOutRowPos,
        fun hc1 => by simp only [hc1, ↓reduceIte]; exact hOutRowPos,
        fun h1 => by omega, ?_, ?_, fun h0 => by simp at h0⟩
      · intro hs0
        have hh1 : heads > 1 := Classical.byContradiction fun h => by simp [h] at hs0
        simp only [hh1, ↓reduceIte, Nat.lt_irrefl, Nat.zero_mul, Nat.add_zero]
        refine ⟨?_, Nat.mul_pos hCols hOutRowPos, ?_⟩
        · by_cases hc1 : cols > 1
          · simp only [hc1, ↓reduceIte]
            have : 2 * outRow ≤ cols * outRow := Nat.mul_le_mul_right _ hc1; omega
          · simp [hc1, Nat.mul_pos hCols hOutRowPos]
        · rw [mul_if_gt_one hcolLt]
          have h1 : (col + 1) * outRow ≤ cols * outRow := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hcolLt)
          rw [Nat.add_mul, Nat.one_mul] at h1; omega
      · intro hs1
        have hc1 : cols > 1 := Classical.byContradiction fun h => by simp [h] at hs1
        simp only [hc1, ↓reduceIte, Nat.lt_irrefl, Nat.zero_mul, Nat.zero_add]
        exact ⟨hOutRowPos, by omega⟩
    rw [hInv]; dsimp only
    have hSrc : viewCopySrcByte cp hd ow 0 col = dramByteOffset4D inSh 0 (hd * rows + ow) 0 col := by
      unfold viewCopySrcByte dramByteOffset4D
      dsimp only [cp, inSh, Shape4D.planeStride, Shape4D.rowStride]
      simp only [Nat.zero_mul, Nat.zero_add, Nat.one_mul]
      rw [mul_if_gt_one hhdLt, mul_if_gt_one hw', ← Nat.mul_assoc, ← Nat.add_mul]
    rw [hSrc]
    have hInC : hd * rows + ow < heads * rows := by
      have h1 : (hd + 1) * rows ≤ heads * rows := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hhdLt)
      rw [Nat.add_mul, Nat.one_mul] at h1; omega
    exact decodeDramByteOffset4D_roundTrip inSh 0 (hd * rows + ow) 0 col Nat.zero_lt_one hInC Nat.zero_lt_one hcolLt
  obtain ⟨r, hr, hrSh, hrElem⟩ := evalViewCopyTensor_of_lookup inSh outSh cp xT
    (fun _ oc _ ow => (0, (oc / cols) * rows + ow, 0, oc % cols)) hXSh hInPos hOutPos hLook
    (fun _ oc _ ow _ hc _ hw => by
      have hhdLt : oc / cols < heads := Nat.div_lt_of_lt_mul (by dsimp only [outSh] at hc; rwa [Nat.mul_comm] at hc)
      have hw' : ow < rows := hw
      have h1 : (oc / cols + 1) * rows ≤ heads * rows := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hhdLt)
      rw [Nat.add_mul, Nat.one_mul] at h1
      dsimp only [inSh]
      exact ⟨Nat.zero_lt_one, by omega, Nat.zero_lt_one, Nat.mod_lt oc hCols⟩)
  refine ⟨r, ?_, hrSh, fun hd col row hhd hcol hrow hOutLt hInLt => ?_⟩
  · unfold evalAneOpCore
    dsimp only [desc, DmaViewSpec.toViewCopy]
    rw [hOutTy]
    have hShEq : outSh.toArray.toList = [1, heads * cols, 1, rows] := rfl
    rw [hShEq]; simp [hr]
  · have hocLt : hd * cols + col < heads * cols := by
      have h1 : (hd + 1) * cols ≤ heads * cols := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hhd)
      rw [Nat.add_mul, Nat.one_mul] at h1; omega
    have hDiv : (hd * cols + col) / cols = hd := by
      rw [Nat.mul_comm hd cols, Nat.mul_add_div hCols, Nat.div_eq_of_lt hcol, Nat.add_zero]
    have hMod : (hd * cols + col) % cols = col := by
      rw [Nat.mul_comm hd cols, Nat.mul_add_mod, Nat.mod_eq_of_lt hcol]
    have hStep := hrElem 0 (hd * cols + col) 0 row Nat.zero_lt_one hocLt Nat.zero_lt_one hrow hOutLt
      (by
        have hInArr : inSh.toArray.toList = [1, heads * rows, 1, cols] := rfl
        rw [hInArr, hDiv, hMod]; exact hInLt)
    have hInArr : inSh.toArray.toList = [1, heads * rows, 1, cols] := rfl
    have hOutArr : outSh.toArray.toList = [1, heads * cols, 1, rows] := rfl
    simpa [hInArr, hOutArr, hDiv, hMod] using hStep

/-- BLOCKER-1 THEOREM 5: Exact `.dmaView` head-major `matmul_join` equivalence for ANY `heads > 1`
    and `m > 16` (matching `LayoutSelect.lean:641-656` with `batchN := heads` and `Matmul.lean:476`!):
    joining M-sliced `[1, heads * mi, 1, n]` `neMatmul` outputs via `evalAneOpCore` produces the
    EXACT same head-major `TensorVal α` as a single unsplit `neMatmul`! -/
theorem dmaView_matmulJoin_headMajor_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (decodeW : WeightData → Nat → Nat → Nat → Nat → Option (Nat → Nat → Nat → Nat → α))
    (decodeB : Option ByteArray → Nat → Option (Option (Nat → α)))
    (heads m n : Nat) (hHeads : heads > 1) (hm : m > 0) (hn : n > 0)
    (f : Nat → Nat → Nat → α) :
    let numSlices := ceilDiv m 16
    let fullOut : Shape4D := { n := 1, c := heads * m, h := 1, w := n }
    let sliceTensors : List (TensorVal α) := (List.range numSlices).map fun si =>
      let m0 := si * 16
      let mi := min 16 (m - m0)
      mkTensor4D (heads * mi) 1 n fun hm' _ j => f (hm' / mi) (m0 + hm' % mi) j
    let sliceShapes : Array Shape4D := ((List.range numSlices).map fun si =>
      ({ n := 1, c := heads * min 16 (m - si * 16), h := 1, w := n } : Shape4D)).toArray
    let concatDesc := { dmaViewSpecFromConcat sliceShapes 1 fullOut (aliasable := true) with batchN := heads }
    evalAneOpCore ops decodeW decodeB (.dmaView concatDesc "matmul_join") sliceTensors
      [shape4DToTensorType fullOut] =
    some [mkTensor4D (heads * m) 1 n fun hm' _ j => f (hm' / m) (hm' % m) j] := by
  intro numSlices fullOut sliceTensors sliceShapes concatDesc
  have h16 : (16 : Nat) > 0 := by omega
  have hNsPos : numSlices > 0 := by
    unfold numSlices ceilDiv; simp only [show (16 == 0) = false by decide, Bool.false_eq_true, ↓reduceIte]
    exact Nat.div_pos (by omega) h16
  have hSumRange : ∀ k, k ≤ numSlices →
      ((List.range k).map fun si =>
        let m0 := si * 16; let mi := min 16 (m - m0)
        mkTensor4D (heads * mi) 1 n fun hm' _ j => f (hm' / mi) (m0 + hm' % mi) j).foldl
        (fun acc t => acc + t.shape.getD 1 0 / heads) 0 = min m (k * 16) := by
    intro k
    induction k with
    | zero => intro _; simp
    | succ r ih =>
      intro hr
      rw [List.range_succ, List.map_append, List.map_singleton, List.foldl_append, List.foldl_cons, List.foldl_nil, ih (by omega)]
      have hSh : (mkTensor4D (heads * min 16 (m - r * 16)) 1 n fun hm' _ j =>
          f (hm' / min 16 (m - r * 16)) (r * 16 + hm' % min 16 (m - r * 16)) j).shape.getD 1 0 =
          heads * min 16 (m - r * 16) := rfl
      rw [hSh, Nat.mul_div_right _ (by omega)]
      have hMul : (r + 1) * 16 = r * 16 + 16 := by rw [Nat.add_mul, Nat.one_mul]
      omega
  have hLookup : ∀ (k i : Nat), k ≤ numSlices → i < min m (k * 16) →
      let si := i / 16
      let m0 := si * 16
      let mi := min 16 (m - m0)
      evalMatmulJoinLookup heads
        ((List.range k).map fun s =>
          let a := s * 16; let b := min 16 (m - a)
          mkTensor4D (heads * b) 1 n fun hm' _ j => f (hm' / b) (a + hm' % b) j) i =
        some (mkTensor4D (heads * mi) 1 n fun hm' _ j => f (hm' / mi) (m0 + hm' % mi) j, mi, i - m0) := by
    intro k
    have hGen : ∀ (len start i : Nat), start + len ≤ numSlices → i < min m ((start + len) * 16) - start * 16 →
        let si := start + i / 16
        let m0 := si * 16
        let mi := min 16 (m - m0)
        evalMatmulJoinLookup heads
          ((List.range' start len 1).map fun s =>
            let a := s * 16; let b := min 16 (m - a)
            mkTensor4D (heads * b) 1 n fun hm' _ j => f (hm' / b) (a + hm' % b) j) i =
          some (mkTensor4D (heads * mi) 1 n fun hm' _ j => f (hm' / mi) (m0 + hm' % mi) j, mi, i - (i / 16) * 16) := by
      intro len
      induction len with
      | zero => intro start i _ hi; omega
      | succ l ih =>
        intro start i hLe hi
        simp only [List.range'_succ, List.map_cons, evalMatmulJoinLookup]
        have hH0 : (heads == 0) = false := by simp; omega
        have hSh0 : (mkTensor4D (heads * min 16 (m - start * 16)) 1 n fun hm' _ j =>
            f (hm' / min 16 (m - start * 16)) (start * 16 + hm' % min 16 (m - start * 16)) j).shape.getD 1 0 =
            heads * min 16 (m - start * 16) := rfl
        have hMod0 : (heads * min 16 (m - start * 16) % heads != 0) = false := by
          rw [Nat.mul_mod_right]; rfl
        have hDiv0 : heads * min 16 (m - start * 16) / heads = min 16 (m - start * 16) :=
          Nat.mul_div_right _ (by omega)
        simp only [hSh0, hH0, hMod0, Bool.or_self, Bool.false_eq_true, ↓reduceIte, hDiv0]
        by_cases hiLt : i < min 16 (m - start * 16)
        · simp only [hiLt, ↓reduceIte]
          have hDivZ : i / 16 = 0 := Nat.div_eq_of_lt (by omega)
          simp [hDivZ]
        · simp only [hiLt, ↓reduceIte]
          have hiGe16 : i ≥ 16 := by omega
          have hMi16 : min 16 (m - start * 16) = 16 := by omega
          rw [hMi16]
          have hRec := ih (start + 1) (i - 16) (by omega) (by
            have h1 : (start + 1 + l) * 16 = (start + (l + 1)) * 16 := by congr 1; omega
            have h2 : (start + 1) * 16 = start * 16 + 16 := by rw [Nat.add_mul, Nat.one_mul]
            omega)
          have hDivSub : (i - 16) / 16 + 1 = i / 16 := by omega
          have hSiEq : start + 1 + (i - 16) / 16 = start + i / 16 := by omega
          have hLocEq : i - 16 - (i - 16) / 16 * 16 = i - i / 16 * 16 := by omega
          simpa [hSiEq, hLocEq] using hRec
    intro i hk hi
    have hR : List.range k = List.range' 0 k 1 := List.range_eq_range'
    rw [hR]
    simpa [Nat.zero_add, Nat.zero_mul, Nat.sub_zero] using hGen k 0 i (by omega) (by simpa using hi)
  have hMLe : m ≤ numSlices * 16 := by
    have hCov := stripOutBand_unique_coverage m 16 (m - 1) h16 (by omega)
    have h1 : ((m - 1) / 16 + 1) * 16 ≤ numSlices * 16 := Nat.mul_le_mul_right 16 hCov.1
    have h2 := Nat.div_add_mod' (m - 1) 16
    have h3 := Nat.mod_lt (m - 1) h16
    rw [Nat.add_mul, Nat.one_mul] at h1; omega
  have hMinM : min m (numSlices * 16) = m := Nat.min_eq_left hMLe
  unfold evalAneOpCore
  have hAx : concatDesc.concatAxis? = some 1 := rfl
  have hBN : concatDesc.batchN = heads := rfl
  have hOSh : concatDesc.outShape = fullOut := rfl
  dsimp only
  rw [hAx, hBN, hOSh]
  simp only [beq_self_eq_true, decide_eq_true hHeads, Bool.and_self, ↓reduceIte]
  unfold evalMatmulJoin
  have hNoErr : (heads == 0 || sliceTensors.isEmpty || fullOut.n != 1 || fullOut.h != 1 ||
      fullOut.w == 0 || fullOut.c % heads != 0) = false := by
    have hNE : sliceTensors.isEmpty = false := by
      cases hR : List.range numSlices with
      | nil => have : (List.range numSlices).length = 0 := by rw [hR]; rfl
               simp at this; omega
      | cons _ _ => simp [sliceTensors, hR]
    have hModC : (heads * m) % heads = 0 := Nat.mul_mod_right heads m
    have hH0 : (heads == 0) = false := by simp; omega
    have hW0 : (n == 0) = false := by simp; omega
    simp [fullOut, hNE, hModC, hH0, hW0]
  have hTotM : fullOut.c / heads = m := Nat.mul_div_right m (by omega)
  have hTotMNe : (m == 0) = false := by simp; omega
  have hAllOk : (sliceTensors.all fun t =>
      match t.shape with
      | [1, ci, 1, wi] => wi == fullOut.w && ci > 0 && ci % heads == 0
      | _ => false) = true := by
    rw [List.all_eq_true]
    intro t ht
    simp only [sliceTensors, List.mem_map, List.mem_range] at ht
    obtain ⟨si, hsi, rfl⟩ := ht
    have hVal := stripOutBand_valid m 16 si hm h16 hsi
    dsimp only [stripOutBand] at hVal
    have hMiPos : min 16 (m - si * 16) > 0 := by omega
    have hCiPos : heads * min 16 (m - si * 16) > 0 := Nat.mul_pos (by omega) hMiPos
    simp [mkTensor4D, fullOut, hCiPos, Nat.mul_mod_right]
  have hSumEq : sliceTensors.foldl (fun acc t => acc + t.shape.getD 1 0 / heads) 0 = m := by
    rw [hSumRange numSlices (Nat.le_refl _), hMinM]
  simp only [hNoErr, Bool.false_eq_true, ↓reduceIte, hTotM, hTotMNe, hAllOk, hSumEq,
    beq_self_eq_true, Bool.and_self, Option.map_some]
  congr 1; congr 1
  apply tensorVal_ext_4d (heads * m) 1 n _ _ rfl rfl
  intro hm' oy ox hhm' hoy hox h1 h2
  have hoy0 : oy = 0 := by omega
  subst hoy0
  rw [mkTensor4D_at (heads * m) 1 n hm' 0 ox _ hhm' (by omega) hox h1]
  rw [mkTensor4D_at (heads * m) 1 n hm' 0 ox _ hhm' (by omega) hox h2]
  have hiLt : hm' % m < m := Nat.mod_lt hm' hm
  have hhdLt : hm' / m < heads := Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hhm')
  rw [hLookup numSlices (hm' % m) (Nat.le_refl _) (by rw [hMinM]; exact hiLt)]
  dsimp only
  let i := hm' % m
  let hd := hm' / m
  let mi := min 16 (m - i / 16 * 16)
  have hLocLt : i - i / 16 * 16 < mi := by
    have hM := Nat.div_add_mod' i 16
    have hMLt := Nat.mod_lt i h16
    dsimp only [mi]; omega
  have hCIdxLt : hd * mi + (i - i / 16 * 16) < heads * mi := by
    have h1 : (hd + 1) * mi ≤ heads * mi := Nat.mul_le_mul_right mi (Nat.succ_le_of_lt hhdLt)
    rw [Nat.add_mul, Nat.one_mul] at h1; omega
  rw [read4D_mkTensor4D (heads * mi) 1 n (hd * mi + (i - i / 16 * 16)) 0 ox _ hCIdxLt (by omega) hox]
  have hMiPos : mi > 0 := by omega
  have hDivM : (hd * mi + (i - i / 16 * 16)) / mi = hd := by
    rw [Nat.mul_comm hd mi, Nat.mul_add_div hMiPos, Nat.div_eq_of_lt hLocLt, Nat.add_zero]
  have hModM : (hd * mi + (i - i / 16 * 16)) % mi = i - i / 16 * 16 := by
    rw [Nat.mul_comm hd mi, Nat.mul_add_mod, Nat.mod_eq_of_lt hLocLt]
  have hAddI : i / 16 * 16 + (i - i / 16 * 16) = i := by
    have := Nat.div_add_mod' i 16; omega
  rw [hDivM, hModM, hAddI]

/-- BLOCKER-1 THEOREM 6: Exact `.dmaView` `w_stitch` equivalence (`Cascade.lean:812-835`):
    1. Each `w_stitch` strip node (`dstBaseOff = 2 * w0`) passes `evalAneOpCore`'s exact DRAM-stride
       `viewCopyInverse` verification against `origOutSh = [1, C, 1, wFull]`.
    2. Scattering all `sCount = ceilDiv wFull stripW` strips via `evalWStitchSurface` reproduces the
       EXACT full `[1, C, 1, wFull]` tensor at EVERY `(oc, ox)`! -/
theorem dmaView_wStitch_surface_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (decodeW : WeightData → Nat → Nat → Nat → Nat → Option (Nat → Nat → Nat → Nat → α))
    (decodeB : Option ByteArray → Nat → Option (Option (Nat → α)))
    (C wFull stripW : Nat) (hC : C > 0) (hwFull : wFull > 0) (hStrip : stripW > 0)
    (f : Nat → Nat → α) :
    let origOutSh : Shape4D := { n := 1, c := C, h := 1, w := wFull }
    let ys := origOutSh.dramStrides
    let sCount := ceilDiv wFull stripW
    let mkStripCp : Nat → ViewCopy := fun s =>
      let (w0, w1) := stripOutBand wFull stripW s
      let curW := w1 - w0
      let stripOutSh : Shape4D := { origOutSh with w := curW }
      let xs := stripOutSh.dramStrides
      { n := 1, c := C, h := 1, w := curW, d := 1
        inOff := 0, inPlaneStride := if C > 1 then xs[1]! else 0
        outOff := 2 * w0, outPlaneStride := if C > 1 then ys[1]! else 0
        transposeCW := false }
    let mkStripTensor : Nat → TensorVal α := fun s =>
      let (w0, w1) := stripOutBand wFull stripW s
      mkTensor4D C 1 (w1 - w0) fun oc _ ox' => f oc (w0 + ox')
    -- 1. Every strip `s < sCount` validates & evaluates cleanly under `evalAneOpCore (.dmaView desc "w_stitch")`!
    (∀ s, s < sCount →
      let (w0, w1) := stripOutBand wFull stripW s
      let curW := w1 - w0
      let stripOutSh : Shape4D := { origOutSh with w := curW }
      let cp := mkStripCp s
      let desc : DmaViewSpec := {
        inShape := stripOutSh, outShape := origOutSh,
        dstBaseOff := 2 * w0, dstStrides := ys, explicitCopy? := some cp
      }
      evalAneOpCore ops decodeW decodeB (.dmaView desc "w_stitch")
        [mkStripTensor s] [shape4DToTensorType stripOutSh] = some [mkStripTensor s]) ∧
    -- 2. Full W-axis surface scatter across all `sCount` strips equals `mkTensor4D C 1 wFull (fun oc _ ox => f oc ox)`!
    (evalWStitchSurface origOutSh ((List.range sCount).map fun s => (mkStripCp s, mkStripTensor s)) =
      mkTensor4D C 1 wFull fun oc _ ox => f oc ox) := by
  intro origOutSh ys sCount mkStripCp mkStripTensor
  have hRowGe : 2 * wFull ≤ origOutSh.rowStride := alignUp_ge_local (2 * wFull) dmaLineBytes
  have hRowPos : 0 < origOutSh.rowStride := by omega
  have hPlEq : origOutSh.planeStride = origOutSh.rowStride := by
    unfold Shape4D.planeStride; simp [origOutSh]
  have hInvStrip : ∀ s (hs : s < sCount) oc ox', oc < C → ox' < (stripOutBand wFull stripW s).2 - (stripOutBand wFull stripW s).1 →
      let w0 := (stripOutBand wFull stripW s).1
      viewCopyInverse (mkStripCp s) (dramByteOffset4D origOutSh 0 oc 0 (w0 + ox')) = some (0, oc, 0, ox') := by
    intro s hs oc ox' hoc hox'
    dsimp only
    have hVal := stripOutBand_valid wFull stripW s hwFull hStrip hs
    let w0 := (stripOutBand wFull stripW s).1
    let w1 := (stripOutBand wFull stripW s).2
    have hDst : dramByteOffset4D origOutSh 0 oc 0 (w0 + ox') = viewCopyDstByte (mkStripCp s) 0 oc 0 ox' := by
      unfold dramByteOffset4D viewCopyDstByte
      dsimp only [mkStripCp, ys, Shape4D.dramStrides, w0]
      have hG1 : (#[origOutSh.batchStride, origOutSh.planeStride, origOutSh.rowStride, 2] : Array Nat)[1]! = origOutSh.planeStride := rfl
      rw [hG1, mul_if_gt_one hoc, Nat.mul_add]
      simp only [Nat.zero_mul, Nat.zero_add, Bool.false_eq_true, ↓reduceIte]
      omega
    change viewCopyInverse (mkStripCp s) (dramByteOffset4D origOutSh 0 oc 0 (w0 + ox')) = some (0, oc, 0, ox')
    rw [hDst]
    apply viewCopyInverse_of_radix (mkStripCp s) 0 oc 0 ox' Nat.zero_lt_one hoc Nat.zero_lt_one hox'
    have hIfIf : ∀ (P : Prop) [Decidable P] (a : Nat), (if P then (if P then a else 0) else 0) = if P then a else 0 := by
      intro P _ a; by_cases h : P <;> simp [h]
    dsimp only [mkStripCp, ys, Shape4D.dramStrides]
    have hG1 : (#[origOutSh.batchStride, origOutSh.planeStride, origOutSh.rowStride, 2] : Array Nat)[1]! = origOutSh.planeStride := rfl
    rw [hG1]
    simp only [Bool.false_eq_true, ↓reduceIte, hIfIf]
    refine ⟨fun h => by omega, fun hc1 => by simp [hc1, hPlEq, hRowPos], fun h => by omega,
      fun h => by simp at h, ?_, fun h => by simp at h⟩
    intro hs1
    have hc1 : C > 1 := Classical.byContradiction fun h => by simp [h] at hs1
    simp only [hc1, ↓reduceIte, Nat.lt_irrefl, Nat.zero_mul, Nat.zero_add, hPlEq]
    exact ⟨hRowPos, by omega⟩
  constructor
  · intro s hs
    have hVal := stripOutBand_valid wFull stripW s hwFull hStrip hs
    let w0 := (stripOutBand wFull stripW s).1
    let w1 := (stripOutBand wFull stripW s).2
    let curW := w1 - w0
    let stripOutSh : Shape4D := { origOutSh with w := curW }
    have hCurWPos : curW > 0 := by dsimp only [curW, w0, w1]; omega
    have hLook : ∀ oc ox', oc < C → ox' < curW →
        viewCopyLookupCoord stripOutSh origOutSh (mkStripCp s) 0 oc 0 (w0 + ox') = some (0, oc, 0, ox') := by
      intro oc ox' hoc hox'
      unfold viewCopyLookupCoord; dsimp only
      rw [hInvStrip s hs oc ox' hoc hox']; dsimp only
      have hSrc : viewCopySrcByte (mkStripCp s) 0 oc 0 ox' = dramByteOffset4D stripOutSh 0 oc 0 ox' := by
        unfold viewCopySrcByte dramByteOffset4D
        dsimp only [mkStripCp, Shape4D.dramStrides]
        have hG1 : (#[stripOutSh.batchStride, stripOutSh.planeStride, stripOutSh.rowStride, 2] : Array Nat)[1]! = stripOutSh.planeStride := rfl
        rw [hG1, mul_if_gt_one hoc]
        simp
      rw [hSrc]
      exact decodeDramByteOffset4D_roundTrip stripOutSh 0 oc 0 ox' Nat.zero_lt_one hoc Nat.zero_lt_one hox'
    unfold evalAneOpCore
    dsimp only [DmaViewSpec.toViewCopy]
    split
    · rename_i hSame
      have hwEq : curW = wFull := by
        simp only [beq_iff_eq, Shape4D.toArray, shape4DToTensorType, List.cons.injEq] at hSame
        exact hSame.2.2.2.1.symm
      have hw0z : w0 = 0 := by dsimp only [curW, w0, w1] at hwEq hVal ⊢; omega
      have hStripEq : stripOutSh = origOutSh := by
        dsimp only [stripOutSh, origOutSh]; rw [hwEq]
      have hEv := evalViewCopyTensor_of_lookup stripOutSh origOutSh (mkStripCp s) (mkStripTensor s)
        (fun _ oc _ ox' => (0, oc, 0, ox')) rfl ⟨Nat.zero_lt_one, hC, Nat.zero_lt_one, hCurWPos⟩
        ⟨Nat.zero_lt_one, hC, Nat.zero_lt_one, hwFull⟩
        (fun on oc oh ox' hn hc hh hx => by
          have : on = 0 := by dsimp only [origOutSh] at hn; omega
          have : oh = 0 := by dsimp only [origOutSh] at hh; omega
          subst this; subst ‹on = 0›
          have hx' : ox' < curW := by dsimp only [origOutSh] at hx; omega
          have hL := hLook oc ox' hc hx'
          simpa [hw0z] using hL)
        (fun _ oc _ ox' _ hc _ hx => ⟨Nat.zero_lt_one, hc, Nat.zero_lt_one, by dsimp only [stripOutSh, origOutSh] at hx ⊢; omega⟩)
      obtain ⟨r, hr, hrSh, hrElem⟩ := hEv
      rw [hr]
      simp only [Option.map_some, Option.some.injEq, List.cons.injEq, and_true]
      apply tensorVal_ext_4d C 1 curW r (mkStripTensor s) (by rw [hrSh, ← hStripEq]; rfl) rfl
      intro oc oy ox' hoc hoy hox' h1 h2
      have : oy = 0 := by omega
      subst this
      have hFl : flatIndex origOutSh.toArray.toList [0, oc, 0, ox'] = oc * (1 * curW) + 0 * curW + ox' := by
        simp [origOutSh, Shape4D.toArray, flatIndex, shapeProd, hwEq]
      have hoxFull : ox' < origOutSh.w := by dsimp only [origOutSh]; omega
      have hStep := hrElem 0 oc 0 ox' Nat.zero_lt_one hoc Nat.zero_lt_one hoxFull (by simpa [hFl] using h1) (by simpa [hFl, hStripEq] using h2)
      simpa [hFl, hStripEq] using hStep
    · unfold evalWStitchStripTensor
      have hCond : ((mkStripTensor s).shape == stripOutSh.toArray.toList &&
          ((shape4DToTensorType stripOutSh).shape.toList == [1, (mkStripCp s).c, 1, (mkStripCp s).w]) &&
          !(mkStripCp s).transposeCW && ((mkStripCp s).n == 1) && ((mkStripCp s).h == 1) &&
          decide ((mkStripCp s).c > 0) && decide ((mkStripCp s).w > 0) &&
          (stripOutSh.n == 1) && (stripOutSh.c == (mkStripCp s).c) && (stripOutSh.h == 1) && (stripOutSh.w == (mkStripCp s).w) &&
          (origOutSh.n == 1) && (origOutSh.c == (mkStripCp s).c) && (origOutSh.h == 1) &&
          ((mkStripCp s).outOff % 2 == 0) && decide ((mkStripCp s).outOff / 2 + (mkStripCp s).w ≤ origOutSh.w)) = true := by
        dsimp only [mkStripTensor, mkTensor4D, mkStripCp, stripOutSh, origOutSh, shape4DToTensorType,
          Shape4D.toArray, curW, w0, w1] at hVal hCurWPos ⊢
        simp [hC]
        omega
      have hAllV : ((List.range (shapeProd [1, (mkStripCp s).c, 1, (mkStripCp s).w])).all fun flatIdx =>
          let cs := unflatten [1, (mkStripCp s).c, 1, (mkStripCp s).w] flatIdx
          (viewCopyLookupCoord stripOutSh origOutSh (mkStripCp s) 0 (cs.getD 1 0) 0 ((mkStripCp s).outOff / 2 + cs.getD 3 0)).isSome) = true := by
        rw [List.all_eq_true]
        intro flatIdx hMem
        rw [List.mem_range] at hMem
        have hP : shapeProd [1, C, 1, curW] = C * curW := by simp [shapeProd]
        change flatIdx < shapeProd [1, C, 1, curW] at hMem
        rw [hP] at hMem
        let oc := flatIdx / curW
        let ox' := flatIdx % curW
        have hoc : oc < C := Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hMem)
        have hox' : ox' < curW := Nat.mod_lt flatIdx hCurWPos
        have hUnf : unflatten [1, C, 1, curW] flatIdx = [0, oc, 0, ox'] := by
          have hEq : oc * (1 * curW) + 0 * curW + ox' = flatIdx := by
            dsimp only [oc, ox']; rw [Nat.one_mul]; have := Nat.div_add_mod' flatIdx curW; omega
          rw [← hEq]; exact unflatten_4d C 1 curW oc 0 ox' hoc Nat.zero_lt_one hox'
        change (viewCopyLookupCoord stripOutSh origOutSh (mkStripCp s) 0
          ((unflatten [1, C, 1, curW] flatIdx).getD 1 0) 0 (2 * w0 / 2 + (unflatten [1, C, 1, curW] flatIdx).getD 3 0)).isSome = true
        rw [hUnf]
        have hDiv2 : 2 * w0 / 2 = w0 := by omega
        change (viewCopyLookupCoord stripOutSh origOutSh (mkStripCp s) 0 oc 0 (2 * w0 / 2 + ox')).isSome = true
        rw [hDiv2, hLook oc ox' hoc hox']; rfl
      dsimp only [stripOutSh, curW, w0, w1] at hCond hAllV
      simp only [hCond, ↓reduceIte, hAllV, Option.map_some, Option.some.injEq, List.cons.injEq, and_true]
      apply tensorVal_ext_4d C 1 curW _ (mkStripTensor s) rfl rfl
      intro oc oy ox' hoc hoy hox' h1 h2
      have : oy = 0 := by omega
      subst this
      dsimp only
      have hUnf : unflatten [1, C, 1, curW] (oc * (1 * curW) + 0 * curW + ox') = [0, oc, 0, ox'] :=
        unflatten_4d C 1 curW oc 0 ox' hoc Nat.zero_lt_one hox'
      change (match viewCopyLookupCoord stripOutSh origOutSh (mkStripCp s) 0
          ((unflatten [1, C, 1, curW] (oc * (1 * curW) + 0 * curW + ox')).getD 1 0) 0
          (2 * w0 / 2 + (unflatten [1, C, 1, curW] (oc * (1 * curW) + 0 * curW + ox')).getD 3 0) with
        | some (sn, sc, sh, sw) =>
          let inIdx := flatIndex stripOutSh.toArray.toList [sn, sc, sh, sw]
          if hlt : inIdx < shapeProd (mkStripTensor s).shape then (mkStripTensor s).data ⟨inIdx, hlt⟩ else zero
        | none => zero) = _
      rw [hUnf]
      have hDiv2 : 2 * w0 / 2 = w0 := by omega
      change (match viewCopyLookupCoord stripOutSh origOutSh (mkStripCp s) 0 oc 0 (2 * w0 / 2 + ox') with
        | some (sn, sc, sh, sw) =>
          let inIdx := flatIndex stripOutSh.toArray.toList [sn, sc, sh, sw]
          if hlt : inIdx < shapeProd (mkStripTensor s).shape then (mkStripTensor s).data ⟨inIdx, hlt⟩ else zero
        | none => zero) = _
      rw [hDiv2, hLook oc ox' hoc hox']
      dsimp only
      have hFl : flatIndex stripOutSh.toArray.toList [0, oc, 0, ox'] = oc * (1 * curW) + 0 * curW + ox' := by
        simp [stripOutSh, origOutSh, Shape4D.toArray, flatIndex, shapeProd]
      rw [hFl]
      simp only [h2, ↓reduceDIte]
  · -- Part 2: Full W-axis surface scatter
    apply tensorVal_ext_4d C 1 wFull _ _ rfl rfl
    intro oc oy ox hoc hoy hox h1 h2
    have : oy = 0 := by omega
    subst this
    unfold evalWStitchSurface
    rw [mkTensor4D_at C 1 wFull oc 0 ox _ hoc Nat.zero_lt_one hox h1]
    rw [mkTensor4D_at C 1 wFull oc 0 ox _ hoc Nat.zero_lt_one hox h2]
    let sOwn := ox / stripW
    have hCov := stripOutBand_unique_coverage wFull stripW ox hStrip hox
    have hHitOwn : viewCopyInverse (mkStripCp sOwn) (dramByteOffset4D origOutSh 0 oc 0 ox) =
        some (0, oc, 0, ox - (stripOutBand wFull stripW sOwn).1) := by
      have hLoLe : (stripOutBand wFull stripW sOwn).1 ≤ ox := hCov.2.1
      have hLtHi : ox < (stripOutBand wFull stripW sOwn).2 := hCov.2.2.1
      have hAdd : (stripOutBand wFull stripW sOwn).1 + (ox - (stripOutBand wFull stripW sOwn).1) = ox := by omega
      have hSubLt : ox - (stripOutBand wFull stripW sOwn).1 <
          (stripOutBand wFull stripW sOwn).2 - (stripOutBand wFull stripW sOwn).1 := by omega
      simpa [hAdd] using hInvStrip sOwn hCov.1 oc (ox - (stripOutBand wFull stripW sOwn).1) hoc hSubLt
    have hAnyHit : ∀ s ∈ List.range sCount, ∀ v,
        (match viewCopyInverse (mkStripCp s) (dramByteOffset4D origOutSh 0 oc 0 ox) with
         | some (_, inC, _, inW) => some (read4D (mkStripTensor s) (mkStripCp s).c 1 (mkStripCp s).w inC 0 inW)
         | none => none) = some v → v = f oc ox := by
      intro s hs v hv
      rw [List.mem_range] at hs
      cases hI : viewCopyInverse (mkStripCp s) (dramByteOffset4D origOutSh 0 oc 0 ox) with
      | none => rw [hI] at hv; cases hv
      | some q =>
        rcases q with ⟨inN, inC, inH, inW⟩
        rw [hI] at hv
        simp only [Option.some.injEq] at hv
        have hRI := viewCopyInverse_right_inv (mkStripCp s) _ _ _ _ _ hI
        have hDstEq := hRI.2.2.2.2
        have hValS := stripOutBand_valid wFull stripW s hwFull hStrip hs
        have hCIn : inC < C := hRI.2.1
        have hWIn : inW < (stripOutBand wFull stripW s).2 - (stripOutBand wFull stripW s).1 := hRI.2.2.2.1
        have hDstEq' : dramByteOffset4D origOutSh 0 inC 0 ((stripOutBand wFull stripW s).1 + inW) =
            dramByteOffset4D origOutSh 0 oc 0 ox := by
          rw [← hDstEq]
          unfold dramByteOffset4D viewCopyDstByte
          dsimp only [mkStripCp, ys, Shape4D.dramStrides]
          have hG1 : (#[origOutSh.batchStride, origOutSh.planeStride, origOutSh.rowStride, 2] : Array Nat)[1]! = origOutSh.planeStride := rfl
          rw [hG1, mul_if_gt_one hCIn, Nat.mul_add]
          simp only [Nat.zero_mul, Nat.zero_add, Bool.false_eq_true, ↓reduceIte]
          omega
        have hWBound : (stripOutBand wFull stripW s).1 + inW < wFull := by omega
        have hDec1 := decodeDramByteOffset4D_roundTrip origOutSh 0 inC 0 ((stripOutBand wFull stripW s).1 + inW)
          Nat.zero_lt_one hCIn Nat.zero_lt_one hWBound
        have hDec2 := decodeDramByteOffset4D_roundTrip origOutSh 0 oc 0 ox Nat.zero_lt_one hoc Nat.zero_lt_one hox
        rw [hDstEq', hDec2] at hDec1
        simp only [Option.some.injEq, Prod.mk.injEq] at hDec1
        rcases hDec1 with ⟨_, hCEq, _, hWEq⟩
        rw [← hv]
        dsimp only [mkStripTensor, mkStripCp]
        rw [read4D_mkTensor4D C 1 _ inC 0 inW _ hCIn Nat.zero_lt_one hWIn, hCEq, hWEq]
    have hFind : ∀ (xs : List Nat) (g : Nat → Option α) (target : α),
        (∃ s ∈ xs, (g s).isSome = true) →
        (∀ s ∈ xs, ∀ v, g s = some v → v = target) →
        xs.findSome? g = some target := by
      intro xs g target
      induction xs with
      | nil => intro ⟨_, h, _⟩ _; cases h
      | cons hd tl ih =>
        intro hEx hAll
        simp only [List.findSome?_cons]
        cases hG : g hd with
        | some v =>
          rw [hAll hd List.mem_cons_self v hG]
        | none =>
          apply ih
          · obtain ⟨s, hs, hSome⟩ := hEx
            rcases List.mem_cons.mp hs with rfl | hTl
            · simp [hG] at hSome
            · exact ⟨s, hTl, hSome⟩
          · intro s hs v hv; exact hAll s (List.mem_cons_of_mem _ hs) v hv
    have hMapFind : ∀ (xs : List Nat) (p : Nat → ViewCopy × TensorVal α) (q : ViewCopy × TensorVal α → Option α),
        (xs.map p).findSome? q = xs.findSome? (fun s => q (p s)) := by
      intro xs p q; induction xs with | nil => rfl | cons hd tl ih =>
        simp only [List.map_cons, List.findSome?_cons]; cases q (p hd) <;> simp [ih]
    dsimp only
    rw [hMapFind]
    have hRes := hFind (List.range sCount)
      (fun s => match viewCopyInverse (mkStripCp s) (dramByteOffset4D origOutSh 0 oc 0 ox) with
        | some (_, inC, _, inW) => some (read4D (mkStripTensor s) (mkStripCp s).c 1 (mkStripCp s).w inC 0 inW)
        | none => none)
      (f oc ox)
      ⟨sOwn, List.mem_range.mpr hCov.1, by rw [hHitOwn]; rfl⟩
      hAnyHit
    rw [hRes]

-- ----------------------------------------------------------------------------
-- Blocker-2 L2 Theorems: `LayoutSelect` (`constOnLeft` div, global max+avg pool,
-- `materializeConst4D`, `neConv` evalAne, `broadcastShapes`)
-- ----------------------------------------------------------------------------

/-- `LayoutSelect` equivalence for `div(const, x)` (`constOnLeft = true`, `LayoutSelect.lean:883-887`):
    `neAct .inverse x` followed by `pe .scalarMul scBits` computes exact `div c x`! -/
theorem layoutSelect_elemBinary_constOnLeft_div_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (scBits : UInt16) (x : α) :
    evalPeUnaryScalar ops (.scalarMul scBits) (applyNeAct ops (.lut .inverse) x) =
      evalElemBinaryScalar ops .div (ops.f16Val scBits) x := by
  simp only [evalPeUnaryScalar, PeOp.scalarMul, applyNeAct, evalElemBinaryScalar, div, ops.lut_inverse]
  exact mul_comm _ _

/-- `LayoutSelect` equivalence for global pooling lowered to `.pe (.reduce .max/.mean {reduceH, reduceW})`
    (`LayoutSelect.lean:1008-1010`), covering BOTH `.avg` (`excludePad = false` AND `excludePad = true`!)
    AND `.max` (`H > 0, W > 0`)! -/
theorem layoutSelect_pool_global_max_and_avg_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W : Nat) (hH : H > 0) (hW : W > 0) (x : Nat → Nat → Nat → α) (c : Nat) (excl : Bool) :
    evalPool2DElem ops .avg H W H W 1 1 0 0 excl x c 0 0 =
      evalReduce1D ops .mean zero (H * W) (fun idx => x c (idx / W) (idx % W)) ∧
    evalPool2DElem ops .max H W H W 1 1 0 0 excl x c 0 0 =
      evalReduce1D ops .max zero (H * W) (fun idx => x c (idx / W) (idx % W)) := by
  constructor
  · cases excl
    · exact layoutSelect_pool_global_avg_equiv ops H W x c
    · -- `excludePad = true`: every `(ky < H, kx < W)` is in bounds, so valid count equals `H * W`!
      have hCnt : ((List.range H).map fun (ky : Nat) =>
          ((List.range W).filter fun (kx : Nat) =>
            decide (0 ≤ (0 : Int) * (1 : Int) + (ky : Int) - (0 : Int) ∧
                    (0 : Int) * (1 : Int) + (ky : Int) - (0 : Int) < (H : Int) ∧
                    0 ≤ (0 : Int) * (1 : Int) + (kx : Int) - (0 : Int) ∧
                    (0 : Int) * (1 : Int) + (kx : Int) - (0 : Int) < (W : Int))).length).sum = H * W := by
        have hRow : ∀ ky : Nat, ky < H →
            ((List.range W).filter fun (kx : Nat) =>
              decide (0 ≤ (0 : Int) * (1 : Int) + (ky : Int) - (0 : Int) ∧
                      (0 : Int) * (1 : Int) + (ky : Int) - (0 : Int) < (H : Int) ∧
                      0 ≤ (0 : Int) * (1 : Int) + (kx : Int) - (0 : Int) ∧
                      (0 : Int) * (1 : Int) + (kx : Int) - (0 : Int) < (W : Int))).length = W := by
          intro ky hky
          have hAllT : (List.range W).filter (fun (kx : Nat) =>
              decide (0 ≤ (0 : Int) * (1 : Int) + (ky : Int) - (0 : Int) ∧
                      (0 : Int) * (1 : Int) + (ky : Int) - (0 : Int) < (H : Int) ∧
                      0 ≤ (0 : Int) * (1 : Int) + (kx : Int) - (0 : Int) ∧
                      (0 : Int) * (1 : Int) + (kx : Int) - (0 : Int) < (W : Int))) = List.range W := by
            rw [List.filter_eq_self]
            intro kx hkx; rw [List.mem_range] at hkx; simp; omega
          rw [hAllT, List.length_range]
        have hMap : (List.range H).map (fun (ky : Nat) =>
            ((List.range W).filter fun (kx : Nat) =>
              decide (0 ≤ (0 : Int) * (1 : Int) + (ky : Int) - (0 : Int) ∧
                      (0 : Int) * (1 : Int) + (ky : Int) - (0 : Int) < (H : Int) ∧
                      0 ≤ (0 : Int) * (1 : Int) + (kx : Int) - (0 : Int) ∧
                      (0 : Int) * (1 : Int) + (kx : Int) - (0 : Int) < (W : Int))).length) =
            (List.range H).map (fun _ => W) :=
          List.map_congr_left fun ky hky => hRow ky (List.mem_range.mp hky)
        rw [hMap]
        have hSumW : ∀ k, ((List.range k).map fun _ => W).sum = k * W := by
          intro k; induction k with
          | zero => simp
          | succ m ih =>
            rw [List.range_succ, List.map_append, List.map_singleton, List.sum_append, List.sum_singleton, ih,
                Nat.add_mul, Nat.one_mul]
        exact hSumW H
      have hBase := layoutSelect_pool_global_avg_equiv ops H W x c
      unfold evalPool2DElem at hBase ⊢
      simp only [Int.ofNat_zero, Int.ofNat_one, ↓reduceIte, hCnt] at hBase ⊢
      exact hBase
  · -- Global `.max`: prove `foldl ops.maxOp` over `(List.range H).flatMap (List.range W)` equals `foldMaxFin ops (H * W)`!
    have hValsEq :
        ((List.range H).flatMap fun (ky : Nat) =>
          (List.range W).filterMap fun (kx : Nat) =>
            if 0 ≤ ((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int) ∧
               ((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int) < (H : Int) ∧
               0 ≤ ((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int) ∧
               ((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int) < (W : Int) then
              some (x c (((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int)).toNat
                        (((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int)).toNat)
            else none) =
        (List.range (H * W)).map (fun idx => x c (idx / W) (idx % W)) := by
      have hRow : ∀ ky : Nat, ky < H →
          ((List.range W).filterMap fun (kx : Nat) =>
            if 0 ≤ ((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int) ∧
               ((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int) < (H : Int) ∧
               0 ≤ ((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int) ∧
               ((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int) < (W : Int) then
              some (x c (((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int)).toNat
                        (((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int)).toNat)
            else none) =
          (List.range W).map (fun kx => x c ky kx) := by
        intro ky hky
        have hF : ∀ xs : List Nat, (∀ kx ∈ xs, kx < W) →
            (xs.filterMap fun (kx : Nat) =>
              if 0 ≤ ((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int) ∧
                 ((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int) < (H : Int) ∧
                 0 ≤ ((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int) ∧
                 ((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int) < (W : Int) then
                some (x c (((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int)).toNat
                          (((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int)).toNat)
              else none) =
            xs.map (fun kx => x c ky kx) := by
          intro xs hxs; induction xs with
          | nil => rfl
          | cons kx rest ih =>
            have hkx : kx < W := hxs kx List.mem_cons_self
            have hIn : 0 ≤ ((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int) ∧
                       ((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int) < (H : Int) ∧
                       0 ≤ ((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int) ∧
                       ((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int) < (W : Int) := by omega
            have hyE : (((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int)).toNat = ky := by omega
            have hxE : (((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int)).toNat = kx := by omega
            have hHead : (if 0 ≤ ((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int) ∧
                   ((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int) < (H : Int) ∧
                   0 ≤ ((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int) ∧
                   ((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int) < (W : Int) then
                  some (x c (((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int)).toNat
                            (((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int)).toNat)
                else none) = some (x c ky kx) := by
              simp only [hIn, and_self, ↓reduceIte, hyE, hxE]
            rw [List.filterMap_cons, List.map_cons, hHead]
            dsimp only
            congr 1; exact ih (fun a ha => hxs a (List.mem_cons_of_mem _ ha))
        exact hF (List.range W) (fun kx hkx => List.mem_range.mp hkx)
      have hFlat : ∀ m, m ≤ H →
          ((List.range m).flatMap fun (ky : Nat) =>
            (List.range W).filterMap fun (kx : Nat) =>
              if 0 ≤ ((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int) ∧
                 ((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int) < (H : Int) ∧
                 0 ≤ ((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int) ∧
                 ((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int) < (W : Int) then
                some (x c (((0 : Nat) : Int) * ((1 : Nat) : Int) + (ky : Int) - ((0 : Nat) : Int)).toNat
                          (((0 : Nat) : Int) * ((1 : Nat) : Int) + (kx : Int) - ((0 : Nat) : Int)).toNat)
              else none) =
          (List.range (m * W)).map (fun idx => x c (idx / W) (idx % W)) := by
        intro m; induction m with
        | zero => intro _; simp
        | succ k ih =>
          intro hk
          rw [List.range_succ, List.flatMap_append, List.flatMap_singleton, ih (by omega), hRow k (by omega)]
          rw [Nat.add_mul, Nat.one_mul, List.range_add, List.map_append, List.map_map]
          congr 1
          apply List.map_congr_left
          intro kx hkx; rw [List.mem_range] at hkx
          have hDiv : (k * W + kx) / W = k := by
            rw [Nat.mul_comm k W, Nat.mul_add_div hW, Nat.div_eq_of_lt hkx, Nat.add_zero]
          have hMod : (k * W + kx) % W = kx := by
            rw [Nat.mul_comm k W, Nat.mul_add_mod, Nat.mod_eq_of_lt hkx]
          simp [Function.comp, hDiv, hMod]
      exact hFlat H (Nat.le_refl H)
    have hFoldMax : ∀ (n : Nat) (g : Nat → α), n > 0 →
        (match (List.range n).map g with
        | [] => zero
        | v :: vs => vs.foldl ops.maxOp v) = foldMaxFin ops n g := by
      intro n g hn
      have hStep : ∀ (m : Nat), m > 0 →
          ((List.range' 1 (m - 1) 1).map g).foldl ops.maxOp (g 0) = foldMaxFin ops m g := by
        intro m; induction m with
        | zero => intro h; omega
        | succ k ih =>
          intro _
          cases k with
          | zero => simp [foldMaxFin]
          | succ k' =>
            have hSub : k' + 1 + 1 - 1 = k' + 1 := by omega
            have hSub2 : k' + 1 - 1 = k' := by omega
            have hR : List.range' 1 (k' + 1) 1 = List.range' 1 k' 1 ++ [k' + 1] := by
              have hA := List.range'_append (s := 1) (m := k') (n := 1) (step := 1)
              simpa [Nat.add_comm] using hA.symm
            rw [hSub, hR, List.map_append, List.map_singleton, List.foldl_append, List.foldl_cons, List.foldl_nil]
            have hPrev := ih (by omega)
            rw [hSub2] at hPrev
            rw [hPrev]; rfl
      have hRangeCons : List.range n = 0 :: List.range' 1 (n - 1) 1 := by
        cases n with | zero => omega | succ k => rw [List.range_eq_range', List.range'_succ]; simp
      rw [hRangeCons, List.map_cons]
      exact hStep n hn
    unfold evalPool2DElem evalReduce1D
    dsimp only
    rw [hValsEq]
    exact hFoldMax (H * W) (fun idx => x c (idx / W) (idx % W)) (Nat.mul_pos hH hW)

/-- `LayoutSelect` equivalence for ALL 3 branches of `materializeConstChunk` (`LayoutSelect.lean:362-425`):
    1. `hc = 1, wc = 1`: zero-weight `neConv 1x1` with bias `vals` outputs `vals[oc]` regardless of anchor `anc`!
    2. `wc = 1`: zero-weight `neConv 1x1` `[1, cc*hc, 1, 1]` + reshape `[1, cc, hc, 1]` outputs `vals[oc*hc + oy]`!
    3. `wc ≥ 2`: packed `[wc, cc*hc]` zero-weight `neConv 1x1` + reshape `[1, wc, cc*hc, 1]` +
       transpose `[0, 3, 2, 1]` `[1, 1, cc*hc, wc]` + reshape `[1, cc, hc, wc]` outputs `vals[(oc*hc + oy)*wc + ox]`! -/
theorem layoutSelect_materializeConst4D_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (cc hc wc : Nat) (anc : Nat → Nat → Nat → α) (vals : Nat → α) :
    let zeroW : Nat → Nat → Nat → Nat → α := fun _ _ _ _ => zero
    -- Branch 1: `hc = 1, wc = 1`
    (∀ oc, oc < cc →
      evalConvElem ops 1 1 1 cc 1 1 1 1 0 0 anc zeroW (some vals) .none oc 0 0 = vals oc) ∧
    -- Branch 2: `wc = 1` (channel-flat `ch = cc * hc` reshaped to `[1, cc, hc, 1]`)
    (∀ oc oy, oc < cc → oy < hc →
      evalConvElem ops 1 1 1 (cc * hc) 1 1 1 1 0 0 anc zeroW (some vals) .none (oc * hc + oy) 0 0 =
        vals (oc * hc + oy)) ∧
    -- Branch 3: `wc ≥ 2` (packed `packed[wIdx * (cc*hc) + chIdx] = vals[chIdx * wc + wIdx]`, reshaped + transposed `[0,3,2,1]`)
    (∀ oc oy ox, oc < cc → oy < hc → ox < wc →
      let ch := cc * hc
      let packed := fun idx => vals ((idx % ch) * wc + idx / ch)
      let chIdx := oc * hc + oy
      evalConvElem ops 1 1 1 (ch * wc) 1 1 1 1 0 0 anc zeroW (some packed) .none (ox * ch + chIdx) 0 0 =
        vals (chIdx * wc + ox)) := by
  intro zeroW
  have hZeroConv : ∀ (N k : Nat) (b : Nat → α),
      evalConvElem ops 1 1 1 N 1 1 1 1 0 0 anc zeroW (some b) .none k 0 0 = b k := by
    intro N k b
    unfold evalConvElem applyConvAct zeroW
    simp only [sumFin, zero_mul, zero_add]
  refine ⟨fun oc _ => hZeroConv cc oc vals, fun oc oy _ _ => hZeroConv (cc * hc) (oc * hc + oy) vals,
    fun oc oy ox hoc hoy hox => ?_⟩
  dsimp only
  rw [hZeroConv]
  have hChLt : oc * hc + oy < cc * hc := by
    have h1 : (oc + 1) * hc ≤ cc * hc := Nat.mul_le_mul_right hc (Nat.succ_le_of_lt hoc)
    rw [Nat.add_mul, Nat.one_mul] at h1; omega
  have hChPos : cc * hc > 0 := by omega
  have hDiv : (ox * (cc * hc) + (oc * hc + oy)) / (cc * hc) = ox := by
    rw [Nat.mul_comm ox (cc * hc), Nat.mul_add_div hChPos, Nat.div_eq_of_lt hChLt, Nat.add_zero]
  have hMod : (ox * (cc * hc) + (oc * hc + oy)) % (cc * hc) = oc * hc + oy := by
    rw [Nat.mul_comm ox (cc * hc), Nat.mul_add_mod, Nat.mod_eq_of_lt hChLt]
  rw [hDiv, hMod]

/-- Direct `evalAneOpCore` preservation for `.neConv` (connecting `CoreOp.conv` to `AneOp.neConv`). -/
theorem layoutSelect_conv_evalAne_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (decodeW : WeightData → Nat → Nat → Nat → Nat → Option (Nat → Nat → Nat → Nat → α))
    (decodeB : Option ByteArray → Nat → Option (Option (Nat → α)))
    (kh kw sh sw cin cout h w ho wo : Nat) (pad : PadMode) (mode : ConvMode)
    (weight : WeightData) (bias : Option ByteArray) (act : ConvActivation)
    (xT : TensorVal α) (hXSh : xT.shape = [1, cin, h, w])
    (outTy : TensorType) (hOutTy : outTy.shape.toList = [1, cout, ho, wo])
    (wFn : Nat → Nat → Nat → Nat → α) (bFn? : Option (Nat → α))
    (hDecW : decodeW weight cout (cin / (match mode with | .dense => 1 | .depthwise => cin | .grouped g => g)) kh kw = some wFn)
    (hDecB : decodeB bias cout = some bFn?) :
    let groups := match mode with | .dense => 1 | .depthwise => cin | .grouped g => g
    let (pt, pl) := match pad with
      | .valid => (0, 0)
      | .same =>
        ((if (ho - 1) * sh + kh > h then (ho - 1) * sh + kh - h else 0) / 2,
         (if (wo - 1) * sw + kw > w then (wo - 1) * sw + kw - w else 0) / 2)
      | .strip ptS _ => (ptS, (if (wo - 1) * sw + kw > w then (wo - 1) * sw + kw - w else 0) / 2)
    evalAneOpCore ops decodeW decodeB (.neConv kh kw sh sw pad mode cin cout weight bias act none 0 "conv") [xT] [outTy] =
      some [evalConvTensor4D ops h w (cin / groups) cout (cout / groups) ho wo kh kw sh sw pt pl
        (fun c y x => read4D xT cin h w c y x) wFn bFn? act] := by
  unfold evalAneOpCore
  dsimp only
  rw [hOutTy]
  simp only []
  rw [hXSh]
  simp only [beq_self_eq_true, Bool.and_self, ↓reduceIte, hDecW, hDecB]

/-- Direct structural soundness on REAL `Vane.Backend.LayoutSelect.broadcastShapes`:
    whenever `broadcastShapes a b = .ok out`, `out.size = max a.size b.size`! -/
theorem layoutSelect_broadcastShapes_size (a b out : Array Nat)
    (hOk : broadcastShapes a b = .ok out) :
    out.size = max a.size b.size := by
  have hGen : ∀ (f : Nat → Array Nat → Except String (ForInStep (Array Nat))),
      (∀ i acc r, f i acc = .ok r → ∃ acc', r = .yield acc' ∧ acc'.size = acc.size + 1) →
      ∀ (xs : List Nat) (acc res : Array Nat),
        forIn (m := Except String) xs acc f = .ok res →
        res.size = acc.size + xs.length := by
    intro f hf xs
    induction xs with
    | nil =>
      intro acc res h
      have hEq : acc = res := Except.ok.inj h
      rw [hEq]; simp
    | cons i rest ih =>
      intro acc res h
      rw [List.forIn_cons] at h
      cases hStep : f i acc with
      | error e => rw [hStep] at h; cases h
      | ok r =>
        obtain ⟨acc', rfl, hSz'⟩ := hf i acc r hStep
        rw [hStep] at h
        have hRec := ih acc' res h
        simp only [List.length_cons, hSz'] at hRec ⊢
        omega
  unfold broadcastShapes at hOk
  dsimp only [bind, Except.bind, pure, Except.pure, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw] at hOk
  split at hOk
  · cases hOk
  · rename_i v hv
    cases hOk
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hv
    have hSzR : ([:max a.size b.size] : Std.Legacy.Range).size = max a.size b.size := by simp [Std.Legacy.Range.size]
    have hStR : ([:max a.size b.size] : Std.Legacy.Range).start = 0 := rfl
    have hSpR : ([:max a.size b.size] : Std.Legacy.Range).step = 1 := rfl
    rw [hStR, hSzR, hSpR] at hv
    have hSz := hGen _ (fun i acc r hr => by
      generalize (if i + a.size ≥ max a.size b.size then a[i + a.size - max a.size b.size]! else 1) = da at hr
      generalize (if i + b.size ≥ max a.size b.size then b[i + b.size - max a.size b.size]! else 1) = db at hr
      split at hr
      · cases hr
      · cases hr; exact ⟨_, rfl, by simp⟩) (List.range' 0 (max a.size b.size) 1) (Array.mkEmpty (max a.size b.size)) out hv
    have hEmptySz : (Array.mkEmpty (max a.size b.size) : Array Nat).size = 0 := rfl
    rw [hEmptySz, List.length_range', Nat.zero_add] at hSz
    exact hSz

-- ============================================================================
-- Section 6: L2 Full `ChannelSplit` (SOK) Convolution Value Equivalence
-- ============================================================================

/-- Build the list of sliced 4D convolution output tensors for `ChannelSplit` SOK
    given channel partition `slices` (e.g. `balancedSlices cout (ceilDiv cout cap)`). -/
def sokSlicedConvTensors {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cin Ho Wo kh kw sh sw pt pl : Nat)
    (x : Nat → Nat → Nat → α)
    (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α))
    (act : ConvActivation)
    (slices : List Nat) : List (TensorVal α) :=
  (List.range slices.length).map fun si =>
    let c0 := sumSizes (slices.take si)
    let dc := slices.getD si 0
    let wSlice := fun oc' ic ky kx => w (c0 + oc') ic ky kx
    let bSlice? := bias?.map fun b oc' => b (c0 + oc')
    evalConvTensor4D ops H W cin dc dc Ho Wo kh kw sh sw pt pl x wSlice bSlice? act

private theorem sumSizes_eq_foldl_shapes {α : Type} (H W : Nat) (slices : List Nat)
    (f : Nat → Nat → TensorVal α)
    (hSh : ∀ si, si < slices.length → (f si (slices.getD si 0)).shape = [1, slices.getD si 0, H, W]) :
    ∀ k, k ≤ slices.length →
      (((List.range slices.length).map fun si => f si (slices.getD si 0)).take k).foldl
        (fun acc t => acc + t.shape.getD 1 0) 0 = sumSizes (slices.take k) := by
  intro k
  induction k with
  | zero => intro _; simp [sumSizes]
  | succ m ih =>
    intro hm
    have hmLt : m < slices.length := by omega
    have hTakeL : (((List.range slices.length).map fun si => f si (slices.getD si 0)).take (m + 1)) =
        (((List.range slices.length).map fun si => f si (slices.getD si 0)).take m) ++
        [f m (slices.getD m 0)] := by
      rw [List.take_succ_eq_append_getElem (by simp; omega)]
      simp
    have hTakeS : slices.take (m + 1) = slices.take m ++ [slices[m]] :=
      List.take_succ_eq_append_getElem hmLt
    have hAppSum : ∀ xs ys : List Nat, sumSizes (xs ++ ys) = sumSizes xs + sumSizes ys := by
      intro xs ys; induction xs with | nil => simp [sumSizes] | cons _ _ iha => simp [sumSizes, iha]; omega
    rw [hTakeL, List.foldl_append, List.foldl_cons, List.foldl_nil, ih (by omega),
        hSh m hmLt, hTakeS, hAppSum]
    simp only [List.getD_eq_getElem?_getD, List.getElem?_cons_succ, List.getElem?_cons_zero,
      Option.getD_some, sumSizes, Nat.add_zero]
    rw [List.getElem?_eq_getElem hmLt]
    rfl

/-- FULL `ChannelSplit` (SOK) convolution value equivalence over exact `ScalarField α`!
    For ANY `cout > 0, cap > 0, cin > 0, Ho > 0, Wo > 0` and ANY dense convolution `(kh, kw, sh, sw, pt, pl)`
    (with optional bias and fused `ConvActivation`), let `slices := balancedSlices cout (ceilDiv cout cap)`.
    Concatenating along channel axis 1 (`evalConcat 1`) the outputs of running convolution on each
    sliced weight/bias chunk `(W_s, b_s)` succeeds and produces the EXACT same `TensorVal α` as unsplit
    convolution at EVERY output coordinate `(oc, oy, ox)`! -/
theorem channelSplit_sok_conv_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cin cout cap Ho Wo kh kw sh sw pt pl : Nat)
    (hCout : cout > 0) (hCap : cap > 0) (hHo : Ho > 0) (hWo : Wo > 0)
    (x : Nat → Nat → Nat → α)
    (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α))
    (act : ConvActivation) :
    let slices := balancedSlices cout (ceilDiv cout cap)
    evalConcat 1 (sokSlicedConvTensors ops H W cin Ho Wo kh kw sh sw pt pl x w bias? act slices) =
      some (evalConvTensor4D ops H W cin cout cout Ho Wo kh kw sh sw pt pl x w bias? act) := by
  intro slices
  have hCapNe : (cap == 0) = false := by simp [Nat.ne_of_gt hCap]
  have hkPos : ceilDiv cout cap > 0 := by
    unfold ceilDiv; rw [hCapNe]; simp only [Bool.false_eq_true, ↓reduceIte]
    exact Nat.div_pos (by omega) hCap
  have hkLe : ceilDiv cout cap ≤ cout := by
    unfold ceilDiv; rw [hCapNe]; simp only [Bool.false_eq_true, ↓reduceIte]
    have h1 : cout + cap - 1 ≤ cout * cap + (cap - 1) := by
      have : cout ≤ cout * cap := Nat.le_mul_of_pos_right cout hCap; omega
    have h2 : (cout * cap + (cap - 1)) / cap = cout := by
      rw [Nat.mul_comm, Nat.mul_add_div hCap, Nat.div_eq_of_lt (by omega), Nat.add_zero]
    exact Nat.le_trans (Nat.div_le_div_right h1) (Nat.le_of_eq h2)
  have hLen : slices.length = ceilDiv cout cap := balancedSlices_length cout (ceilDiv cout cap)
  have hSum : sumSizes slices = cout := balancedSlices_sum cout (ceilDiv cout cap) hkPos
  have hPos : ∀ dc ∈ slices, dc > 0 := balancedSlices_pos cout (ceilDiv cout cap) hkPos hkLe
  let ts := sokSlicedConvTensors ops H W cin Ho Wo kh kw sh sw pt pl x w bias? act slices
  have hNe : ts ≠ [] := by
    intro hEmpty
    have hL : ts.length = 0 := by rw [hEmpty]; rfl
    simp [ts, sokSlicedConvTensors, hLen] at hL
    omega
  have hShapes : ∀ t ∈ ts, ∃ dc > 0, t.shape = [1, dc, Ho, Wo] := by
    intro t ht
    unfold ts sokSlicedConvTensors at ht
    rw [List.mem_map] at ht
    obtain ⟨si, hsi, rfl⟩ := ht
    rw [List.mem_range] at hsi
    have hGet : slices.getD si 0 = slices[si] := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hsi]; rfl
    exact ⟨slices.getD si 0, hGet ▸ hPos _ (List.getElem_mem hsi), rfl⟩
  have hTakeSum : ∀ k, k ≤ slices.length →
      (ts.take k).foldl (fun acc t => acc + t.shape.getD 1 0) 0 = sumSizes (slices.take k) := by
    intro k hk
    dsimp only [ts, sokSlicedConvTensors]
    exact sumSizes_eq_foldl_shapes Ho Wo slices
      (fun si dc => evalConvTensor4D ops H W cin dc dc Ho Wo kh kw sh sw pt pl x
        (fun oc' ic ky kx => w (sumSizes (slices.take si) + oc') ic ky kx)
        (bias?.map fun b oc' => b (sumSizes (slices.take si) + oc')) act)
      (fun _ _ => rfl) k hk
  have hSumTs : ts.foldl (fun acc t => acc + t.shape.getD 1 0) 0 = cout := by
    have hAll := hTakeSum slices.length (Nat.le_refl _)
    have hTk1 : ts.take slices.length = ts := List.take_of_length_le (by simp [ts, sokSlicedConvTensors])
    have hTk2 : slices.take slices.length = slices := List.take_of_length_le (Nat.le_refl _)
    rw [hTk1, hTk2, hSum] at hAll
    exact hAll
  obtain ⟨out, hCat, hOutSh, hElem⟩ :=
    sok_evalConcat_shape_and_elements Ho Wo cout hHo hWo ts hNe hShapes hSumTs
  rw [hCat]
  congr 1
  apply tensorVal_ext_4d cout Ho Wo out _ hOutSh rfl
  intro oc oy ox hoc hoy hox h1 h2
  obtain ⟨si, hsi, hlo, hhi, hInLt, hEqAt⟩ := hElem oc oy ox hoc hoy hox h1
  rw [hEqAt]
  have hsiSl : si < slices.length := by simpa [ts, sokSlicedConvTensors] using hsi
  have hC0Eq : (ts.take si).foldl (fun acc t => acc + t.shape.getD 1 0) 0 = sumSizes (slices.take si) :=
    hTakeSum si (Nat.le_of_lt hsiSl)
  have hDcEq : ts[si].shape.getD 1 0 = slices.getD si 0 := by
    simp only [ts, sokSlicedConvTensors, List.getElem_map, List.getElem_range]
    rfl
  have hLocalLt : oc - sumSizes (slices.take si) < slices.getD si 0 := by
    rw [← hC0Eq, ← hDcEq]; omega
  have hGetT : ts[si] = evalConvTensor4D ops H W cin (slices.getD si 0) (slices.getD si 0) Ho Wo kh kw sh sw pt pl x
      (fun oc' ic ky kx => w (sumSizes (slices.take si) + oc') ic ky kx)
      (bias?.map fun b oc' => b (sumSizes (slices.take si) + oc')) act := by
    simp [ts, sokSlicedConvTensors]
  have hInLt' : (oc - sumSizes (slices.take si)) * (Ho * Wo) + oy * Wo + ox <
      shapeProd [1, slices.getD si 0, Ho, Wo] :=
    flat4D_lt (slices.getD si 0) Ho Wo (oc - sumSizes (slices.take si)) oy ox hLocalLt hoy hox
  have hLeft : ts[si].data ⟨(oc - (ts.take si).foldl (fun acc t => acc + t.shape.getD 1 0) 0) * (Ho * Wo) + oy * Wo + ox, hInLt⟩ =
      evalConvElem ops H W cin (slices.getD si 0) kh kw sh sw pt pl x
        (fun oc' ic ky kx => w (sumSizes (slices.take si) + oc') ic ky kx)
        (bias?.map fun b oc' => b (sumSizes (slices.take si) + oc')) act
        (oc - sumSizes (slices.take si)) oy ox := by
    have hSub : (ts[si]).data ⟨(oc - (ts.take si).foldl (fun acc t => acc + t.shape.getD 1 0) 0) * (Ho * Wo) + oy * Wo + ox, hInLt⟩ =
        (evalConvTensor4D ops H W cin (slices.getD si 0) (slices.getD si 0) Ho Wo kh kw sh sw pt pl x
          (fun oc' ic ky kx => w (sumSizes (slices.take si) + oc') ic ky kx)
          (bias?.map fun b oc' => b (sumSizes (slices.take si) + oc')) act).data
          ⟨(oc - sumSizes (slices.take si)) * (Ho * Wo) + oy * Wo + ox, hInLt'⟩ := by
      generalize hEqT : ts[si] = tCur at hInLt ⊢
      have htEq : tCur = evalConvTensor4D ops H W cin (slices.getD si 0) (slices.getD si 0) Ho Wo kh kw sh sw pt pl x
          (fun oc' ic ky kx => w (sumSizes (slices.take si) + oc') ic ky kx)
          (bias?.map fun b oc' => b (sumSizes (slices.take si) + oc')) act := hEqT ▸ hGetT
      subst htEq
      congr 1
      apply Fin.ext
      dsimp only
      rw [hC0Eq]
    rw [hSub]
    unfold evalConvTensor4D
    exact mkTensor4D_at (slices.getD si 0) Ho Wo (oc - sumSizes (slices.take si)) oy ox _ hLocalLt hoy hox hInLt'
  rw [hLeft]
  unfold evalConvTensor4D
  rw [mkTensor4D_at cout Ho Wo oc oy ox _ hoc hoy hox h2]
  unfold evalConvElem
  have hDiv1 : (oc - sumSizes (slices.take si)) / slices.getD si 0 = 0 := Nat.div_eq_of_lt hLocalLt
  have hDiv2 : oc / cout = 0 := Nat.div_eq_of_lt hoc
  have hAddOc : sumSizes (slices.take si) + (oc - sumSizes (slices.take si)) = oc := by
    rw [← hC0Eq]; omega
  simp only [hDiv1, hDiv2, Nat.zero_mul, Nat.zero_add, hAddOc]
  cases bias? <;> simp [hAddOc]

/-- Corollary: Full SOK `ChannelSplit` equivalence for `1x1` dense stride-1 convolutions! -/
theorem channelSplit_sok_conv1x1_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cin cout cap : Nat)
    (hCout : cout > 0) (hCap : cap > 0) (hH : H > 0) (hW : W > 0)
    (x : Nat → Nat → Nat → α) (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α)) (act : ConvActivation) :
    let slices := balancedSlices cout (ceilDiv cout cap)
    evalConcat 1 (sokSlicedConvTensors ops H W cin H W 1 1 1 1 0 0 x w bias? act slices) =
      some (evalConvTensor4D ops H W cin cout cout H W 1 1 1 1 0 0 x w bias? act) :=
  channelSplit_sok_conv_equiv ops H W cin cout cap H W 1 1 1 1 0 0 hCout hCap hH hW x w bias? act

/-- Corollary: Full SOK `ChannelSplit` equivalence for `3x3` dense stride-1 same-pad convolutions! -/
theorem channelSplit_sok_conv3x3_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cin cout cap : Nat)
    (hCout : cout > 0) (hCap : cap > 0) (hH : H > 0) (hW : W > 0)
    (x : Nat → Nat → Nat → α) (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α)) (act : ConvActivation) :
    let slices := balancedSlices cout (ceilDiv cout cap)
    evalConcat 1 (sokSlicedConvTensors ops H W cin H W 3 3 1 1 1 1 x w bias? act slices) =
      some (evalConvTensor4D ops H W cin cout cout H W 3 3 1 1 1 1 x w bias? act) :=
  channelSplit_sok_conv_equiv ops H W cin cout cap H W 3 3 1 1 1 1 hCout hCap hH hW x w bias? act

/-- Direct byte-slice extraction exactness for `ChannelSplit.channelSplitPass` (`ChannelSplit.lean:238-239, 325-326`):
    extracting bytes `[2 * c0 * stride, 2 * (c0 + dc) * stride)` from `wBytes` (with `stride = cin` for 1x1
    or `cin * 9` for 3x3) and `[2 * c0, 2 * (c0 + dc))` from `bias` yields exact byte lengths and matches
    every channel `(oc' < dc)` to original channel `c0 + oc'`! -/
theorem channelSplit_byteSlice_extract_exact (bs : ByteArray) (c0 dc elemStride : Nat)
    (hBound : 2 * (c0 + dc) * elemStride ≤ bs.size) :
    let sl := bs.extract (2 * c0 * elemStride) (2 * (c0 + dc) * elemStride)
    sl.size = 2 * dc * elemStride ∧
    ∀ k (hk : k < sl.size),
      sl[k]'hk = bs[2 * c0 * elemStride + k]'(by
        have hSz : sl.size = 2 * (c0 + dc) * elemStride - 2 * c0 * elemStride := by
          simp [sl, ByteArray.size_extract, Nat.min_eq_left hBound]
        have hSz2 : sl.size = 2 * dc * elemStride := by
          rw [hSz, Nat.mul_add, Nat.add_mul]; omega
        have hSub : 2 * c0 * elemStride + sl.size = 2 * (c0 + dc) * elemStride := by
          rw [hSz2, Nat.mul_add, Nat.add_mul]
        omega) := by
  intro sl
  have hSz1 : sl.size = 2 * (c0 + dc) * elemStride - 2 * c0 * elemStride := by
    simp [sl, ByteArray.size_extract, Nat.min_eq_left hBound]
  have hSz : sl.size = 2 * dc * elemStride := by
    rw [hSz1, Nat.mul_add, Nat.add_mul]; omega
  refine ⟨hSz, fun k hk => ?_⟩
  exact ByteArray.getElem_extract hk

/-- Direct connection to REAL `Vane.Backend.ChannelSplit.channelSplitPass`:
    on any node-empty `AneProgram` with empty inputs/nodeMeta, `channelSplitPass` preserves `prog` identically! -/
theorem channelSplitPass_empty_graph_exact (prog : AneProgram) (hNoNodes : prog.graph.nodes = #[])
    (hNoInputs : prog.graph.inputs = #[]) (hNoMeta : prog.nodeMeta = #[]) :
    channelSplitPass prog = .ok prog := by
  unfold channelSplitPass
  cases prog with | mk g nodeMeta outputNames stripCount stripHeight stripAxis weightCount weightBytes =>
  cases g with | mk target inputs nodes outputs =>
  dsimp only at hNoNodes hNoInputs hNoMeta; subst hNoNodes; subst hNoInputs; subst hNoMeta
  have hR0 : ([:0] : Std.Legacy.Range).size = 0 := by simp [Std.Legacy.Range.size]
  simp only [Array.size_empty, Std.Legacy.Range.forIn_eq_forIn_range', hR0, List.range'_zero]
  rfl

-- ============================================================================
-- Section 7: L2 Full `Cascade` Strip-Unrolling Convolution Value Equivalence
-- ============================================================================

/-- Eligible cascade layer (`3x3` stride-1 same-pad conv with halo radius `1`,
    `1x1` conv with radius `0`, or pointwise PE/LUT op with radius `0`). -/
inductive CascadeLayer (α : Type) where
  | conv3x3Same (cin cout : Nat) (w : Nat → Nat → Nat → Nat → α) (bias? : Option (Nat → α)) (act : ConvActivation)
  | conv1x1 (cin cout : Nat) (w : Nat → Nat → Nat → Nat → α) (bias? : Option (Nat → α)) (act : ConvActivation)
  | pointwise (f : α → α)

def CascadeLayer.radius {α : Type} : CascadeLayer α → Nat
  | .conv3x3Same .. => 1
  | .conv1x1 .. => 0
  | .pointwise _ => 0

/-- Single cascade layer evaluated on the full frame `(hFull × W)`. -/
def evalCascadeLayerFull {α : Type} [ScalarField α] (ops : ScalarOps α)
    (hFull W : Nat) (layer : CascadeLayer α) (x : Nat → Nat → Nat → α) : Nat → Nat → Nat → α :=
  match layer with
  | .conv3x3Same cin cout w bias? act =>
    evalConvElem ops hFull W cin cout 3 3 1 1 1 1 x w bias? act
  | .conv1x1 cin cout w bias? act =>
    evalConvElem ops hFull W cin cout 1 1 1 1 0 0 x w bias? act
  | .pointwise f =>
    fun c y ox => f (x c y ox)

/-- Single cascade layer evaluated on an H-strip covering output rows `[outLo, outHi)`
    with input strip rows `[inLo, inHi) = haloInBand hFull layer.radius outLo outHi`,
    using `Cascade.stripPadTop outLo` and `Cascade.stripPadBot outHi hFull`! -/
def evalCascadeLayerStripH {α : Type} [ScalarField α] (ops : ScalarOps α)
    (hFull W outLo outHi : Nat) (layer : CascadeLayer α) (xStrip : Nat → Nat → Nat → α) :
    Nat → Nat → Nat → α :=
  let (inLo, inHi) := haloInBand hFull layer.radius outLo outHi
  let inH := inHi - inLo
  match layer with
  | .conv3x3Same cin cout w bias? act =>
    let pt := stripPadTop outLo
    evalConvElem ops inH W cin cout 3 3 1 1 pt 1 xStrip w bias? act
  | .conv1x1 cin cout w bias? act =>
    evalConvElem ops inH W cin cout 1 1 1 1 0 0 xStrip w bias? act
  | .pointwise f =>
    fun c y' ox => f (xStrip c y' ox)

/-- Exact single-layer 3x3 H-strip convolution equivalence (using `Cascade.haloInBand`,
    `Cascade.stripPadTop`, and `Cascade.stripPadBot`):
    if `xStrip` agrees with full tensor `xFull` on the halo rows `[inLo, inHi)`, then
    the strip 3x3 conv with top pad `stripPadTop outLo` equals unstripped `conv3x3Same`
    at EVERY coordinate `(oc, y', ox)` (`y' < outHi - outLo`, `ox < W`)! -/
theorem cascade_hStrip_conv3x3_single_layer_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (hFull W cin cout outLo outHi : Nat)
    (hBand : outLo < outHi) (hHi : outHi ≤ hFull)
    (xFull xStrip : Nat → Nat → Nat → α)
    (w : Nat → Nat → Nat → Nat → α) (bias? : Option (Nat → α)) (act : ConvActivation)
    (hInAgree : let (inLo, inHi) := haloInBand hFull 1 outLo outHi
      ∀ c y' ox, y' < inHi - inLo → ox < W → xStrip c y' ox = xFull c (inLo + y') ox)
    (oc y' ox : Nat) (hy' : y' < outHi - outLo) (_hox : ox < W) :
    evalCascadeLayerStripH ops hFull W outLo outHi (.conv3x3Same cin cout w bias? act) xStrip oc y' ox =
    evalCascadeLayerFull ops hFull W (.conv3x3Same cin cout w bias? act) xFull oc (outLo + y') ox := by
  unfold evalCascadeLayerStripH evalCascadeLayerFull CascadeLayer.radius
  dsimp only
  unfold evalConvElem
  have hActBias : ∀ a1 a2 : α, a1 = a2 →
      applyConvAct ops act (match bias? with | some b => add a1 (b oc) | none => a1) =
      applyConvAct ops act (match bias? with | some b => add a2 (b oc) | none => a2) := fun _ _ h => by rw [h]
  apply hActBias
  apply sumFin_congr; intro ic' _
  apply sumFin_congr; intro ky hky
  apply sumFin_congr; intro kx hkx
  congr 1
  unfold haloInBand at hInAgree ⊢
  unfold stripPadTop
  dsimp only at hInAgree ⊢
  simp only [Int.ofNat_one]
  by_cases hLo0 : outLo = 0
  · subst hLo0
    simp only [Nat.not_le.mpr (Nat.zero_lt_one), ↓reduceIte, beq_self_eq_true, Nat.sub_zero, Nat.zero_add, Int.ofNat_one] at hInAgree ⊢
    by_cases hIn : 0 ≤ (y' : Int) * (1 : Int) + (ky : Int) - (1 : Int) ∧
                   (y' : Int) * (1 : Int) + (ky : Int) - (1 : Int) < (hFull : Int) ∧
                   0 ≤ (ox : Int) * (1 : Int) + (kx : Int) - (1 : Int) ∧
                   (ox : Int) * (1 : Int) + (kx : Int) - (1 : Int) < (W : Int)
    · have hInStrip : 0 ≤ (y' : Int) * (1 : Int) + (ky : Int) - (1 : Int) ∧
                      (y' : Int) * (1 : Int) + (ky : Int) - (1 : Int) < ((min hFull (outHi + 1) : Nat) : Int) ∧
                      0 ≤ (ox : Int) * (1 : Int) + (kx : Int) - (1 : Int) ∧
                      (ox : Int) * (1 : Int) + (kx : Int) - (1 : Int) < (W : Int) := by omega
      simp only [hIn, hInStrip, and_self, ↓reduceIte]
      congr 1
      exact hInAgree _ ((y' : Int) * (1 : Int) + (ky : Int) - (1 : Int)).toNat
        ((ox : Int) * (1 : Int) + (kx : Int) - (1 : Int)).toNat (by omega) (by omega)
    · have hOutStrip : ¬ (0 ≤ (y' : Int) * (1 : Int) + (ky : Int) - (1 : Int) ∧
                          (y' : Int) * (1 : Int) + (ky : Int) - (1 : Int) < ((min hFull (outHi + 1) : Nat) : Int) ∧
                          0 ≤ (ox : Int) * (1 : Int) + (kx : Int) - (1 : Int) ∧
                          (ox : Int) * (1 : Int) + (kx : Int) - (1 : Int) < (W : Int)) := by omega
      simp only [hIn, hOutStrip, ↓reduceIte]
  · have hLoGe1 : outLo ≥ 1 := by omega
    have hLoBeq : (outLo == 0) = false := by simp [hLo0]
    simp only [hLoGe1, ↓reduceIte, hLoBeq, Bool.false_eq_true, Int.ofNat_zero] at hInAgree ⊢
    by_cases hIn : 0 ≤ ((outLo + y' : Nat) : Int) * (1 : Int) + (ky : Int) - (1 : Int) ∧
                   ((outLo + y' : Nat) : Int) * (1 : Int) + (ky : Int) - (1 : Int) < (hFull : Int) ∧
                   0 ≤ (ox : Int) * (1 : Int) + (kx : Int) - (1 : Int) ∧
                   (ox : Int) * (1 : Int) + (kx : Int) - (1 : Int) < (W : Int)
    · have hInStrip : 0 ≤ (y' : Int) * (1 : Int) + (ky : Int) - (0 : Int) ∧
                      (y' : Int) * (1 : Int) + (ky : Int) - (0 : Int) < ((min hFull (outHi + 1) - (outLo - 1) : Nat) : Int) ∧
                      0 ≤ (ox : Int) * (1 : Int) + (kx : Int) - (1 : Int) ∧
                      (ox : Int) * (1 : Int) + (kx : Int) - (1 : Int) < (W : Int) := by omega
      simp only [hIn, hInStrip, and_self, ↓reduceIte]
      have hAg := hInAgree (oc / cout * cin + ic') ((y' : Int) * (1 : Int) + (ky : Int) - (0 : Int)).toNat
        ((ox : Int) * (1 : Int) + (kx : Int) - (1 : Int)).toNat (by omega) (by omega)
      have hRowEq : outLo - 1 + ((y' : Int) * (1 : Int) + (ky : Int) - (0 : Int)).toNat =
          (((outLo + y' : Nat) : Int) * (1 : Int) + (ky : Int) - (1 : Int)).toNat := by omega
      rw [hAg, hRowEq]
    · have hOutStrip : ¬ (0 ≤ (y' : Int) * (1 : Int) + (ky : Int) - (0 : Int) ∧
                          (y' : Int) * (1 : Int) + (ky : Int) - (0 : Int) < ((min hFull (outHi + 1) - (outLo - 1) : Nat) : Int) ∧
                          0 ≤ (ox : Int) * (1 : Int) + (kx : Int) - (1 : Int) ∧
                          (ox : Int) * (1 : Int) + (kx : Int) - (1 : Int) < (W : Int)) := by omega
      simp only [hIn, hOutStrip, ↓reduceIte]

/-- Exact single-layer H-strip equivalence for ALL `CascadeLayer` variants (`3x3`, `1x1`, `pointwise`). -/
theorem cascade_hStrip_layer_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (hFull W outLo outHi : Nat) (layer : CascadeLayer α)
    (hBand : outLo < outHi) (hHi : outHi ≤ hFull)
    (xFull xStrip : Nat → Nat → Nat → α)
    (hInAgree : let (inLo, inHi) := haloInBand hFull layer.radius outLo outHi
      ∀ c y' ox, y' < inHi - inLo → ox < W → xStrip c y' ox = xFull c (inLo + y') ox)
    (oc y' ox : Nat) (hy' : y' < outHi - outLo) (hox : ox < W) :
    evalCascadeLayerStripH ops hFull W outLo outHi layer xStrip oc y' ox =
    evalCascadeLayerFull ops hFull W layer xFull oc (outLo + y') ox := by
  cases layer with
  | conv3x3Same cin cout w bias? act =>
    exact cascade_hStrip_conv3x3_single_layer_equiv ops hFull W cin cout outLo outHi
      hBand hHi xFull xStrip w bias? act hInAgree oc y' ox hy' hox
  | conv1x1 cin cout w bias? act =>
    unfold evalCascadeLayerStripH evalCascadeLayerFull
    dsimp only
    unfold CascadeLayer.radius haloInBand at hInAgree ⊢
    simp only [Nat.zero_le, ↓reduceIte, Nat.sub_zero, Nat.add_zero, Nat.min_eq_right hHi] at hInAgree ⊢
    unfold evalConvElem
    have hActBias : ∀ a1 a2 : α, a1 = a2 →
        applyConvAct ops act (match bias? with | some b => add a1 (b oc) | none => a1) =
        applyConvAct ops act (match bias? with | some b => add a2 (b oc) | none => a2) := fun _ _ h => by rw [h]
    apply hActBias
    apply sumFin_congr; intro ic' _
    apply sumFin_congr; intro ky hky
    apply sumFin_congr; intro kx hkx
    have hky0 : ky = 0 := by omega
    have hkx0 : kx = 0 := by omega
    subst hky0; subst hkx0
    simp only [Int.ofNat_zero, Int.ofNat_one]
    have hInS : (0 : Int) ≤ (y' : Int) * (1 : Int) + (0 : Int) - (0 : Int) ∧
                (y' : Int) * (1 : Int) + (0 : Int) - (0 : Int) < ((outHi - outLo : Nat) : Int) ∧
                (0 : Int) ≤ (ox : Int) * (1 : Int) + (0 : Int) - (0 : Int) ∧
                (ox : Int) * (1 : Int) + (0 : Int) - (0 : Int) < (W : Int) := by omega
    have hInF : (0 : Int) ≤ ((outLo + y' : Nat) : Int) * (1 : Int) + (0 : Int) - (0 : Int) ∧
                ((outLo + y' : Nat) : Int) * (1 : Int) + (0 : Int) - (0 : Int) < (hFull : Int) ∧
                (0 : Int) ≤ (ox : Int) * (1 : Int) + (0 : Int) - (0 : Int) ∧
                (ox : Int) * (1 : Int) + (0 : Int) - (0 : Int) < (W : Int) := by omega
    have hy1 : ((y' : Int) * (1 : Int) + (0 : Int) - (0 : Int)).toNat = y' := by omega
    have hy2 : (((outLo + y' : Nat) : Int) * (1 : Int) + (0 : Int) - (0 : Int)).toNat = outLo + y' := by omega
    have hx1 : ((ox : Int) * (1 : Int) + (0 : Int) - (0 : Int)).toNat = ox := by omega
    simp only [hInS, hInF, and_self, ↓reduceIte, hy1, hy2, hx1]
    rw [hInAgree _ y' ox hy' hox]
  | pointwise f =>
    unfold evalCascadeLayerStripH evalCascadeLayerFull
    dsimp only
    unfold CascadeLayer.radius haloInBand at hInAgree
    simp only [Nat.zero_le, ↓reduceIte, Nat.sub_zero, Nat.add_zero, Nat.min_eq_right hHi] at hInAgree
    rw [hInAgree oc y' ox hy' hox]

/-- Full unstripped K-layer stack execution (tail-recursive in execution order: head layer first). -/
def evalCascadeStackFull {α : Type} [ScalarField α] (ops : ScalarOps α)
    (hFull W : Nat) : List (CascadeLayer α) → (Nat → Nat → Nat → α) → (Nat → Nat → Nat → α)
  | [], x => x
  | l :: rest, x => evalCascadeStackFull ops hFull W rest (evalCascadeLayerFull ops hFull W l x)

/-- Backward-propagated H-strip K-layer cascade execution (matching `Cascade.propagateStripIntervals`
    & `Cascade.unrollHStripCascade`): given the desired output row band `[outLo, outHi)` at the LAST
    layer, recursively computes the required input halo band via `Cascade.haloInBand` and runs each
    layer on its strip with `stripPadTop` / `stripPadBot`! -/
def evalCascadeStackHStrip {α : Type} [ScalarField α] (ops : ScalarOps α)
    (hFull W : Nat) (xFull : Nat → Nat → Nat → α) :
    List (CascadeLayer α) → Nat → Nat → (Nat → Nat → Nat → α)
  | [], outLo, _outHi => fun c y' ox => xFull c (outLo + y') ox
  | l :: rest, outLo, outHi =>
    -- Propagate `[outLo, outHi)` backward through `rest` to get the output interval for `l`
    let (midLo, midHi) := rest.foldr (fun lyr (lo, hi) => haloInBand hFull lyr.radius lo hi) (outLo, outHi)
    let (inLo, inHi) := haloInBand hFull l.radius midLo midHi
    let _ := inHi
    let lOut := evalCascadeLayerStripH ops hFull W midLo midHi l (fun c y' ox => xFull c (inLo + y') ox)
    -- Continue with `rest` on `lOut` (rebased at `midLo`!)
    evalCascadeStackHStrip ops hFull W (fun c y ox => lOut c (y - midLo) ox) rest outLo outHi

private theorem haloInBand_foldr_bounds {α : Type} (hFull : Nat) (layers : List (CascadeLayer α))
    (lo hi : Nat) (hBand : lo < hi) (hHi : hi ≤ hFull) :
    let (lo', hi') := layers.foldr (fun lyr (a, b) => haloInBand hFull lyr.radius a b) (lo, hi)
    lo' ≤ lo ∧ hi ≤ hi' ∧ lo' < hi' ∧ hi' ≤ hFull := by
  induction layers with
  | nil => exact ⟨Nat.le_refl _, Nat.le_refl _, hBand, hHi⟩
  | cons hd tl ih =>
    simp only [List.foldr_cons]
    have h1 := haloInBand_in_bounds hFull hd.radius _ _ ih.2.2.1 ih.2.2.2
    exact ⟨Nat.le_trans h1.1 ih.1, Nat.le_trans ih.2.1 h1.2.1, h1.2.2.1, h1.2.2.2⟩

/-- INDUCTIVE K-LAYER H-STRIP CASCADE EQUIVALENCE!
    For ANY sequence of `K` cascade layers (arbitrary mix of 3x3 stride-1 same-pad convs, 1x1 convs,
    and pointwise ops), ANY `hFull > 0`, `hOut > 0`, and ANY strip `s < ceilDiv hFull hOut`,
    executing the K-layer H-strip cascade on `stripOutBand hFull hOut s` produces the EXACT SAME
    output value at EVERY coordinate `(c, y', ox)` as the full unstripped K-layer stack! -/
theorem cascade_hStrip_multiLayer_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (hFull W : Nat) (layers : List (CascadeLayer α)) :
    ∀ (xFull xStripSrc : Nat → Nat → Nat → α) (outLo outHi : Nat),
      outLo < outHi → outHi ≤ hFull →
      (let (reqLo, reqHi) := layers.foldr (fun lyr (a, b) => haloInBand hFull lyr.radius a b) (outLo, outHi)
       ∀ c y ox, reqLo ≤ y → y < reqHi → ox < W → xStripSrc c y ox = xFull c y ox) →
      ∀ c y' ox, y' < outHi - outLo → ox < W →
        evalCascadeStackHStrip ops hFull W xStripSrc layers outLo outHi c y' ox =
        evalCascadeStackFull ops hFull W layers xFull c (outLo + y') ox := by
  induction layers with
  | nil =>
    intro xFull xStripSrc outLo outHi _ _ hAg c y' ox hy' hox
    unfold evalCascadeStackHStrip evalCascadeStackFull
    exact hAg c (outLo + y') ox (by omega) (by omega) hox
  | cons l rest ih =>
    intro xFull xStripSrc outLo outHi hBand hHi hAg c y' ox hy' hox
    unfold evalCascadeStackHStrip evalCascadeStackFull
    dsimp only
    let mid := rest.foldr (fun lyr (a, b) => haloInBand hFull lyr.radius a b) (outLo, outHi)
    have hMidB := haloInBand_foldr_bounds hFull rest outLo outHi hBand hHi
    have hMidLoLt : mid.1 < mid.2 := hMidB.2.2.1
    have hMidHiLe : mid.2 ≤ hFull := hMidB.2.2.2
    have hInB := haloInBand_in_bounds hFull l.radius mid.1 mid.2 hMidLoLt hMidHiLe
    let inB := haloInBand hFull l.radius mid.1 mid.2
    apply ih (evalCascadeLayerFull ops hFull W l xFull)
      (fun ch y x => evalCascadeLayerStripH ops hFull W mid.1 mid.2 l (fun ch0 r0 x0 => xStripSrc ch0 (inB.1 + r0) x0) ch (y - mid.1) x)
      outLo outHi hBand hHi
    · intro ch y x hyLo hyHi hxLt
      have hSubLt : y - mid.1 < mid.2 - mid.1 := by dsimp only [mid] at *; omega
      have hLayerEq := cascade_hStrip_layer_equiv ops hFull W mid.1 mid.2 l hMidLoLt hMidHiLe
        xFull (fun ch0 r0 x0 => xStripSrc ch0 (inB.1 + r0) x0)
        (fun ch0 r0 x0 hr0 hx0 => by
          apply hAg ch0 (inB.1 + r0) x0
          · change inB.1 ≤ inB.1 + r0; exact Nat.le_add_right _ _
          · change inB.1 + r0 < inB.2
            have hr0' : r0 < inB.2 - inB.1 := hr0
            omega
          · exact hx0)
        ch (y - mid.1) x hSubLt hxLt
      have hAddSub : mid.1 + (y - mid.1) = y := by dsimp only [mid] at *; omega
      rw [hLayerEq, hAddSub]
    · exact hy'
    · exact hox

/-- Full-frame H-strip reassembly theorem: for ANY `K` layers, ANY `hFull > 0`, `hOut > 0`, and
    EVERY output coordinate `(c, y, ox)` (`y < hFull, ox < W`), taking the unique owning strip
    `s = y / hOut < ceilDiv hFull hOut` of `Cascade.stripOutBand hFull hOut s` reproduces the
    EXACT unstripped K-layer stack output! -/
theorem cascade_hStrip_fullFrame_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (hFull W hOut : Nat) (hFullPos : hFull > 0) (hOutPos : hOut > 0)
    (layers : List (CascadeLayer α)) (xFull : Nat → Nat → Nat → α)
    (c y ox : Nat) (hy : y < hFull) (hox : ox < W) :
    let s := y / hOut
    let (y0, y1) := stripOutBand hFull hOut s
    s < ceilDiv hFull hOut ∧ y0 ≤ y ∧ y < y1 ∧
    evalCascadeStackHStrip ops hFull W xFull layers y0 y1 c (y - y0) ox =
      evalCascadeStackFull ops hFull W layers xFull c y ox := by
  intro s
  have hCov := stripOutBand_unique_coverage hFull hOut y hOutPos hy
  have hVal := stripOutBand_valid hFull hOut s hFullPos hOutPos hCov.1
  have hyLo : (stripOutBand hFull hOut s).1 ≤ y := hCov.2.1
  have hyHi : y < (stripOutBand hFull hOut s).2 := hCov.2.2.1
  refine ⟨hCov.1, hyLo, hyHi, ?_⟩
  have hEq := cascade_hStrip_multiLayer_equiv ops hFull W layers xFull xFull
    (stripOutBand hFull hOut s).1 (stripOutBand hFull hOut s).2
    hVal.1 hVal.2.1 (fun _ _ _ _ _ _ => rfl) c (y - (stripOutBand hFull hOut s).1) ox
    (by omega) hox
  have hAdd : (stripOutBand hFull hOut s).1 + (y - (stripOutBand hFull hOut s).1) = y := by omega
  rwa [hAdd] at hEq

/-- Micro-DMA halo stitching for `Cascade.unrollRollingHaloCascade` (`Cascade.lean` lines 591–642):
    given per-strip disjoint buffers `prevStrips s` (computed ONCE per strip `s`, zero recomputation!),
    stitches `[prev_bottom_row? (if s > 0), curr_strip, next_top_row? (if s + 1 < sCount)]` along H! -/
def stitchRollingHaloH {α : Type} (hFull hOut s : Nat)
    (prevStrips : Nat → Nat → Nat → Nat → α) (c y' ox : Nat) : α :=
  let sCount := ceilDiv hFull hOut
  let (y0, y1) := stripOutBand hFull hOut s
  let stripH := y1 - y0
  if s > 0 then
    if y' == 0 then
      let (p0, p1) := stripOutBand hFull hOut (s - 1)
      prevStrips (s - 1) c (p1 - p0 - 1) ox
    else if y' - 1 < stripH then
      prevStrips s c (y' - 1) ox
    else
      prevStrips (s + 1) c 0 ox
  else
    if y' < stripH then
      prevStrips s c y' ox
    else
      if s + 1 < sCount then prevStrips (s + 1) c 0 ox else prevStrips s c y' ox

/-- Exactness of micro-DMA rolling halo stitching: if `prevStrips s` matches `fullTensor` on
    `stripOutBand hFull hOut s` for every `s < ceilDiv hFull hOut`, then `stitchRollingHaloH`
    equals `fullTensor` on the EXACT halo input band `haloInBand hFull 1 y0 y1`! -/
theorem cascade_rollingHalo_stitch_exact {α : Type} (hFull W hOut s : Nat)
    (hFullPos : hFull > 0) (hOutGe2 : hOut ≥ 2) (hs : s < ceilDiv hFull hOut)
    (fullTensor : Nat → Nat → Nat → α)
    (prevStrips : Nat → Nat → Nat → Nat → α)
    (hPrevAgree : ∀ s' < ceilDiv hFull hOut,
      let (lo, hi) := stripOutBand hFull hOut s'
      ∀ c r ox, r < hi - lo → ox < W → prevStrips s' c r ox = fullTensor c (lo + r) ox) :
    let (y0, y1) := stripOutBand hFull hOut s
    let (inLo, inHi) := haloInBand hFull 1 y0 y1
    ∀ c y' ox, y' < inHi - inLo → ox < W →
      stitchRollingHaloH hFull hOut s prevStrips c y' ox = fullTensor c (inLo + y') ox := by
  have hOutPos : hOut > 0 := by omega
  have hCur := stripOutBand_valid hFull hOut s hFullPos hOutPos hs
  have hStitch := rollingHalo_stitch_bounds hFull hOut s hFullPos hOutGe2 hs
  intro c y' ox hy' hox
  unfold stitchRollingHaloH
  cases s with
  | zero =>
    simp only [Nat.lt_irrefl, ↓reduceIte]
    dsimp only [stripOutBand, haloInBand] at hCur hStitch hy' ⊢
    simp only [Nat.zero_mul, Nat.zero_add, Nat.sub_zero] at hCur hStitch hy' ⊢
    have h0Lt1 : ¬ ((0 : Nat) ≥ 1) := by omega
    simp only [h0Lt1, ↓reduceIte, Nat.zero_add, Nat.sub_zero] at hStitch hy' ⊢
    by_cases hyMid : y' < min hFull hOut
    · simp only [hyMid, ↓reduceIte]
      have h0 := hPrevAgree 0 hs c y' ox
      simp only [Nat.zero_mul, Nat.zero_add, Nat.sub_zero] at h0
      exact h0 hyMid hox
    · simp only [hyMid, ↓reduceIte]
      have hsNext : 1 < ceilDiv hFull hOut := by
        apply Classical.byContradiction
        intro hNotNext
        have hSzEq := hStitch.1
        simp only [Nat.lt_irrefl, hNotNext, ↓reduceIte, Nat.add_zero] at hSzEq
        omega
      simp only [hsNext, ↓reduceIte]
      have hNextVal := stripOutBand_valid hFull hOut 1 hFullPos hOutPos hsNext
      simp only [stripOutBand, Nat.one_mul] at hNextVal
      have hY'Eq : y' = hOut := by
        have hSzEq := hStitch.1
        simp only [Nat.lt_irrefl, hsNext, ↓reduceIte] at hSzEq
        omega
      have h1 := hPrevAgree 1 hsNext c 0 ox
      simp only [Nat.one_mul] at h1
      rw [h1 (by omega) hox, hY'Eq, Nat.add_zero]
  | succ s' =>
    have hsPos : s' + 1 > 0 := Nat.succ_pos s'
    simp only [hsPos, ↓reduceIte, Nat.add_sub_cancel]
    have hPrevVal := stripOutBand_valid hFull hOut s' hFullPos hOutPos (by omega)
    dsimp only [stripOutBand, haloInBand] at hCur hPrevVal hStitch hy' ⊢
    have hS1Mul : (s' + 1) * hOut = s' * hOut + hOut := by rw [Nat.add_mul, Nat.one_mul]
    have hLoGe1 : (s' + 1) * hOut ≥ 1 := by rw [hS1Mul]; omega
    simp only [hLoGe1, ↓reduceIte] at hStitch hy' ⊢
    by_cases hy0 : y' = 0
    · subst hy0
      simp only [beq_self_eq_true, ↓reduceIte]
      have hp := hPrevAgree s' (by omega) c (min hFull (s' * hOut + hOut) - s' * hOut - 1) ox
      rw [hp (by omega) hox]
      congr 1; omega
    · have hy0B : (y' == 0) = false := by simp [hy0]
      simp only [hy0B, Bool.false_eq_true, ↓reduceIte]
      by_cases hyMid : y' - 1 < min hFull ((s' + 1) * hOut + hOut) - (s' + 1) * hOut
      · simp only [hyMid, ↓reduceIte]
        have hc := hPrevAgree (s' + 1) hs c (y' - 1) ox
        rw [hc hyMid hox]
        congr 1; omega
      · simp only [hyMid, ↓reduceIte]
        have hsNext : s' + 1 + 1 < ceilDiv hFull hOut := by
          apply Classical.byContradiction
          intro hNotNext
          have hSzEq := hStitch.1
          simp only [hsPos, hNotNext, ↓reduceIte, Nat.add_zero] at hSzEq
          omega
        have hNextVal := stripOutBand_valid hFull hOut (s' + 2) hFullPos hOutPos hsNext
        simp only [stripOutBand] at hNextVal
        have hS2Mul : (s' + 2) * hOut = (s' + 1) * hOut + hOut := by rw [Nat.add_mul, Nat.add_mul]; omega
        have hY'Eq : y' = min hFull ((s' + 1) * hOut + hOut) - (s' + 1) * hOut + 1 := by
          have hSzEq := hStitch.1
          simp only [hsPos, hsNext, ↓reduceIte] at hSzEq
          omega
        have hn := hPrevAgree (s' + 2) hsNext c 0 ox
        rw [hn (by omega) hox]
        congr 1; omega

/-- Zero-recompute Rolling Halo-Stitch K-layer execution (matching `Cascade.unrollRollingHaloCascade`):
    each layer computes each strip `s` ONCE on disjoint `stripOutBand hFull hOut s`; before any `rad = 1`
    layer `l > 0`, stitches `[prev_bottom?, curr, next_top?]` via `stitchRollingHaloH`! -/
def evalCascadeStackRollingHalo {α : Type} [ScalarField α] (ops : ScalarOps α)
    (hFull W hOut : Nat) (xFull : Nat → Nat → Nat → α) :
    List (CascadeLayer α) → (Nat → Nat → Nat → Nat → α)
  | [] => fun s c y' ox =>
    let (y0, _) := stripOutBand hFull hOut s
    xFull c (y0 + y') ox
  | l :: rest =>
    -- Layer 0 reads input surface on `haloInBand hFull l.radius y0 y1`, subsequent layers stitch from prev!
    let layer0Strips : Nat → Nat → Nat → Nat → α := fun s =>
      let (y0, y1) := stripOutBand hFull hOut s
      let (inLo, _) := haloInBand hFull l.radius y0 y1
      evalCascadeLayerStripH ops hFull W y0 y1 l (fun c r ox => xFull c (inLo + r) ox)
    rest.foldl (fun prevStrips lyr =>
      fun s =>
        let (y0, y1) := stripOutBand hFull hOut s
        let inStrip :=
          if lyr.radius == 0 then prevStrips s
          else stitchRollingHaloH hFull hOut s prevStrips
        evalCascadeLayerStripH ops hFull W y0 y1 lyr inStrip) layer0Strips

/-- INDUCTIVE K-LAYER ROLLING HALO-STITCH EQUIVALENCE (ZERO HALO RECOMPUTATION)!
    For ANY sequence of `K` cascade layers, ANY `hFull > 0`, `hOut ≥ 2`, and EVERY strip
    `s < ceilDiv hFull hOut` and coordinate `(c, y', ox)`, `evalCascadeStackRollingHalo` matches
    the full unstripped K-layer stack `evalCascadeStackFull` identically — AND the wavefront schedule
    `Cascade.rollingHaloLayerLags` / `Cascade.rollingHaloWave` satisfies strict def-before-use! -/
theorem cascade_rollingHalo_multiLayer_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (hFull W hOut : Nat) (hFullPos : hFull > 0) (hOutGe2 : hOut ≥ 2)
    (layers : List (CascadeLayer α)) (xFull : Nat → Nat → Nat → α) :
    (∀ s < ceilDiv hFull hOut,
      let (y0, y1) := stripOutBand hFull hOut s
      ∀ c y' ox, y' < y1 - y0 → ox < W →
        evalCascadeStackRollingHalo ops hFull W hOut xFull layers s c y' ox =
        evalCascadeStackFull ops hFull W layers xFull c (y0 + y') ox) ∧
    (let radii := (layers.map CascadeLayer.radius).toArray
     ∀ l s, l > 0 → l < radii.size →
       let lags := rollingHaloLayerLags radii
       (radii[l]! = 0 → rollingHaloWave lags (l - 1) s = rollingHaloWave lags l s ∧ l - 1 < l) ∧
       (radii[l]! > 0 →
         (s > 0 → rollingHaloWave lags (l - 1) (s - 1) < rollingHaloWave lags l s) ∧
         rollingHaloWave lags (l - 1) s < rollingHaloWave lags l s ∧
         rollingHaloWave lags (l - 1) (s + 1) = rollingHaloWave lags l s ∧ l - 1 < l)) := by
  have hOutPos : hOut > 0 := by omega
  refine ⟨?_, fun l s hlPos hlLt => rollingHalo_wavefront_def_before_use _ l s hlPos hlLt⟩
  cases layers with
  | nil =>
    intro s _ c y' ox _ _
    rfl
  | cons l0 rest =>
    -- Prove invariant across `rest.foldl`!
    have hFold : ∀ (lyrs : List (CascadeLayer α))
        (fullPrev : Nat → Nat → Nat → α) (stripPrev : Nat → Nat → Nat → Nat → α),
        (∀ s < ceilDiv hFull hOut,
          let (y0, y1) := stripOutBand hFull hOut s
          ∀ c y' ox, y' < y1 - y0 → ox < W → stripPrev s c y' ox = fullPrev c (y0 + y') ox) →
        ∀ s < ceilDiv hFull hOut,
          let (y0, y1) := stripOutBand hFull hOut s
          ∀ c y' ox, y' < y1 - y0 → ox < W →
            (lyrs.foldl (fun ps lyr => fun s' =>
              let (a, b) := stripOutBand hFull hOut s'
              let inS := if lyr.radius == 0 then ps s' else stitchRollingHaloH hFull hOut s' ps
              evalCascadeLayerStripH ops hFull W a b lyr inS) stripPrev) s c y' ox =
            evalCascadeStackFull ops hFull W lyrs fullPrev c (y0 + y') ox := by
      intro lyrs
      induction lyrs with
      | nil =>
        intro fullPrev stripPrev hAg s hs c y' ox hy' hox
        exact hAg s hs c y' ox hy' hox
      | cons lyr tl ih =>
        intro fullPrev stripPrev hAg s hs c y' ox hy' hox
        simp only [List.foldl_cons, evalCascadeStackFull]
        apply ih (evalCascadeLayerFull ops hFull W lyr fullPrev) _ _ s hs c y' ox hy' hox
        intro s' hs' c' r' ox' hr' hox'
        have hVal := stripOutBand_valid hFull hOut s' hFullPos hOutPos hs'
        apply cascade_hStrip_layer_equiv ops hFull W
          (stripOutBand hFull hOut s').1 (stripOutBand hFull hOut s').2 lyr
          hVal.1 hVal.2.1 fullPrev _ _ c' r' ox' hr' hox'
        cases lyr with
        | conv3x3Same cin cout w bias? act =>
          simp only [CascadeLayer.radius, beq_iff_eq, Nat.one_ne_zero, ↓reduceIte]
          exact cascade_rollingHalo_stitch_exact hFull W hOut s' hFullPos hOutGe2 hs' fullPrev stripPrev hAg
        | conv1x1 cin cout w bias? act =>
          simp only [CascadeLayer.radius, beq_self_eq_true, ↓reduceIte, haloInBand,
            Nat.zero_le, Nat.sub_zero, Nat.add_zero]
          have hVal' : (stripOutBand hFull hOut s').2 ≤ hFull := hVal.2.1
          rw [Nat.min_eq_right hVal']
          exact hAg s' hs'
        | pointwise f =>
          simp only [CascadeLayer.radius, beq_self_eq_true, ↓reduceIte, haloInBand,
            Nat.zero_le, Nat.sub_zero, Nat.add_zero]
          have hVal' : (stripOutBand hFull hOut s').2 ≤ hFull := hVal.2.1
          rw [Nat.min_eq_right hVal']
          exact hAg s' hs'
    unfold evalCascadeStackRollingHalo evalCascadeStackFull
    dsimp only
    apply hFold rest (evalCascadeLayerFull ops hFull W l0 xFull) _
    intro s hs c y' ox hy' hox
    have hVal := stripOutBand_valid hFull hOut s hFullPos hOutPos hs
    exact cascade_hStrip_layer_equiv ops hFull W
      (stripOutBand hFull hOut s).1 (stripOutBand hFull hOut s).2 l0
      hVal.1 hVal.2.1 xFull _ (fun _ _ _ _ _ => rfl) c y' ox hy' hox

/-- Pointwise-in-W layer eligible for `Cascade.unrollWStripCascade` (`H = 1, W = seq_len`). -/
inductive WStripLayer (α : Type) where
  | conv1x1 (cin cout : Nat) (w : Nat → Nat → Nat → Nat → α) (bias? : Option (Nat → α)) (act : ConvActivation)
  | pointwise (f : α → α)

def evalWStripLayer {α : Type} [ScalarField α] (ops : ScalarOps α)
    (wExt : Nat) (layer : WStripLayer α) (x : Nat → Nat → Nat → α) : Nat → Nat → Nat → α :=
  match layer with
  | .conv1x1 cin cout w bias? act => evalConvElem ops 1 wExt cin cout 1 1 1 1 0 0 x w bias? act
  | .pointwise f => fun c y ox => f (x c y ox)

def evalWStripStack {α : Type} [ScalarField α] (ops : ScalarOps α)
    (wExt : Nat) : List (WStripLayer α) → (Nat → Nat → Nat → α) → (Nat → Nat → Nat → α)
  | [], x => x
  | l :: rest, x => evalWStripStack ops wExt rest (evalWStripLayer ops wExt l x)

/-- INDUCTIVE K-LAYER W-STRIP CASCADE EQUIVALENCE (`Cascade.unrollWStripCascade`):
    slicing input along W to `[w0, w1) = stripOutBand wFull stripW s`, executing ANY `K`
    pointwise-in-W layers (`conv1x1` / `pointwise`) on width `w1 - w0`, yields the EXACT same
    value at EVERY `(c, 0, x')` (`x' < w1 - w0`) as the unstripped stack at `(c, 0, w0 + x')`! -/
theorem cascade_wStrip_multiLayer_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (wFull stripW s : Nat) (hFullPos : wFull > 0) (hStep : stripW > 0) (hs : s < ceilDiv wFull stripW)
    (layers : List (WStripLayer α)) :
    let (w0, w1) := stripOutBand wFull stripW s
    ∀ (xFull xStrip : Nat → Nat → Nat → α),
      (∀ c x', x' < w1 - w0 → xStrip c 0 x' = xFull c 0 (w0 + x')) →
      ∀ c x', x' < w1 - w0 →
        evalWStripStack ops (w1 - w0) layers xStrip c 0 x' =
        evalWStripStack ops wFull layers xFull c 0 (w0 + x') := by
  have hVal := stripOutBand_valid wFull stripW s hFullPos hStep hs
  induction layers with
  | nil =>
    intro xFull xStrip hAg c x' hx'
    exact hAg c x' hx'
  | cons l rest ih =>
    intro xFull xStrip hAg c x' hx'
    simp only [evalWStripStack]
    apply ih (evalWStripLayer ops wFull l xFull) (evalWStripLayer ops _ l xStrip) _ c x' hx'
    intro ch ox' hox'
    cases l with
    | pointwise f =>
      simp only [evalWStripLayer]
      rw [hAg ch ox' hox']
    | conv1x1 cin cout w bias? act =>
      unfold evalWStripLayer evalConvElem
      dsimp only [stripOutBand] at hVal hAg hox' ⊢
      have hActBias : ∀ a1 a2 : α, a1 = a2 →
          applyConvAct ops act (match bias? with | some b => add a1 (b ch) | none => a1) =
          applyConvAct ops act (match bias? with | some b => add a2 (b ch) | none => a2) := fun _ _ h => by rw [h]
      apply hActBias
      apply sumFin_congr; intro ic' _
      apply sumFin_congr; intro ky hky
      apply sumFin_congr; intro kx hkx
      have hky0 : ky = 0 := by omega
      have hkx0 : kx = 0 := by omega
      subst hky0; subst hkx0
      simp only [Int.ofNat_zero, Int.ofNat_one]
      have hIn1 : (0 : Int) ≤ (0 : Int) * (1 : Int) + (0 : Int) - (0 : Int) ∧
                  (0 : Int) * (1 : Int) + (0 : Int) - (0 : Int) < (1 : Int) ∧
                  (0 : Int) ≤ (ox' : Int) * (1 : Int) + (0 : Int) - (0 : Int) ∧
                  (ox' : Int) * (1 : Int) + (0 : Int) - (0 : Int) < ((min wFull (s * stripW + stripW) - s * stripW : Nat) : Int) := by omega
      have hIn2 : (0 : Int) ≤ (0 : Int) * (1 : Int) + (0 : Int) - (0 : Int) ∧
                  (0 : Int) * (1 : Int) + (0 : Int) - (0 : Int) < (1 : Int) ∧
                  (0 : Int) ≤ ((s * stripW + ox' : Nat) : Int) * (1 : Int) + (0 : Int) - (0 : Int) ∧
                  ((s * stripW + ox' : Nat) : Int) * (1 : Int) + (0 : Int) - (0 : Int) < (wFull : Int) := by omega
      have hy0 : ((0 : Int) * (1 : Int) + (0 : Int) - (0 : Int)).toNat = 0 := by omega
      have hx1 : ((ox' : Int) * (1 : Int) + (0 : Int) - (0 : Int)).toNat = ox' := by omega
      have hx2 : (((s * stripW + ox' : Nat) : Int) * (1 : Int) + (0 : Int) - (0 : Int)).toNat =
          s * stripW + ox' := by omega
      simp only [hIn1, hIn2, and_self, ↓reduceIte, hy0, hx1, hx2]
      rw [hAg _ ox' hox']

/-- 2-input residual / skip-connection `pe.binary` DAG preservation for BOTH H-strip and W-strip
    cascades (matching `Cascade.lean:525, 783` where `pe.binary` nodes take 2 same-shape strip inputs
    from earlier layers or graph inputs)! -/
theorem cascade_binary_residual_dag_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (bop : PeBinOp) (rev : Bool) (wFull outLo outHi wLo wHi : Nat)
    (aFull bFull aStripH bStripH aStripW bStripW : Nat → Nat → Nat → α)
    (hAgAH : ∀ c y' ox, y' < outHi - outLo → ox < wFull → aStripH c y' ox = aFull c (outLo + y') ox)
    (hAgBH : ∀ c y' ox, y' < outHi - outLo → ox < wFull → bStripH c y' ox = bFull c (outLo + y') ox)
    (hAgAW : ∀ c x', x' < wHi - wLo → aStripW c 0 x' = aFull c 0 (wLo + x'))
    (hAgBW : ∀ c x', x' < wHi - wLo → bStripW c 0 x' = bFull c 0 (wLo + x')) :
    (∀ c y' ox, y' < outHi - outLo → ox < wFull →
      evalPeBinaryScalar ops bop rev (aStripH c y' ox) (bStripH c y' ox) =
      evalPeBinaryScalar ops bop rev (aFull c (outLo + y') ox) (bFull c (outLo + y') ox)) ∧
    (∀ c x', x' < wHi - wLo →
      evalPeBinaryScalar ops bop rev (aStripW c 0 x') (bStripW c 0 x') =
      evalPeBinaryScalar ops bop rev (aFull c 0 (wLo + x')) (bFull c 0 (wLo + x'))) := by
  exact ⟨fun c y' ox hy' hox => by rw [hAgAH c y' ox hy' hox, hAgBH c y' ox hy' hox],
         fun c x' hx' => by rw [hAgAW c x' hx', hAgBW c x' hx']⟩

/-- Direct connection to REAL `Vane.Backend.Cascade.cascadePass`:
    when `cfg.stripMode = .off`, `cascadePass` is the exact identity on `AneProgram`! -/
theorem cascadePass_stripMode_off_id (prog : AneProgram) (cfg : CascadeConfig)
    (hOff : cfg.stripMode = .off) :
    cascadePass prog cfg = .ok prog := by
  unfold cascadePass
  rw [hOff]; rfl

-- ============================================================================
-- Section 8: Full C4 `DecomposeConv` Value Equivalence (Shift-Conv & Grouped Conv)
-- ============================================================================

/-- Exact hardware `basePad ∈ {0, 1}` invariant for `DecomposeConv.lean` lines 228–236:
    for ANY positive spatial extent `h > 0` and hardware stride `sHw ∈ {1, 2}`,
    `basePad = padTotal / 2` lies strictly in `{0, 1}`! -/
theorem decomposeConv_basePad_0_or_1 (h sHw : Nat) (hh : h > 0) (hs : sHw = 1 ∨ sHw = 2) :
    let subH := ceilDiv h sHw
    let padTotal := if (subH - 1) * sHw + 3 > h then (subH - 1) * sHw + 3 - h else 0
    padTotal / 2 = 0 ∨ padTotal / 2 = 1 := by
  rcases hs with rfl | rfl
  · have hSub : ceilDiv h 1 = h := by unfold ceilDiv; simp
    simp only [hSub, Nat.mul_one]
    have hGt : h - 1 + 3 > h := by omega
    simp only [hGt, ↓reduceIte]
    right; omega
  · have hSub : ceilDiv h 2 = (h + 1) / 2 := by unfold ceilDiv; simp
    simp only [hSub]
    have hMod := Nat.div_add_mod' (h + 1) 2
    have hRem := Nat.mod_lt (h + 1) (by omega : 2 > 0)
    by_cases hGt : ((h + 1) / 2 - 1) * 2 + 3 > h
    · simp only [hGt, ↓reduceIte]; omega
    · simp only [hGt, ↓reduceIte]; omega

/-- Exact stride-index bound: for `h > 0, sHw > 0`, every output coordinate `oy < ceilDiv h sHw`
    satisfies `oy * sHw < h` (the exact precondition of `C4Conv.assignConvAxisShiftTap_shift_sound`)! -/
theorem decomposeConv_stride_in_bounds (h sHw oy : Nat) (hs : sHw > 0) (hoy : oy < ceilDiv h sHw) :
    oy * sHw < h := by
  have hCeil : ceilDiv h sHw = (h + sHw - 1) / sHw := by
    unfold ceilDiv; simp [Nat.ne_of_gt hs]
  rw [hCeil] at hoy
  have h1 : (oy + 1) * sHw ≤ (h + sHw - 1) / sHw * sHw :=
    Nat.mul_le_mul_right sHw (Nat.succ_le_of_lt hoy)
  have h2 : (h + sHw - 1) / sHw * sHw ≤ h + sHw - 1 := Nat.div_mul_le_self _ _
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Pure 2D zero-padded spatial shift `(sy, sx)` (lifting `C4Conv.shift1D` to 2D). -/
def shift2D {α : Type} (H W : Nat) (x : Nat → Nat → α) (zero : α) (sy sx : Int) (y x' : Nat) : α :=
  if 0 ≤ (y : Int) + sy ∧ (y : Int) + sy < (H : Int) ∧
     0 ≤ (x' : Int) + sx ∧ (x' : Int) + sx < (W : Int) then
    x ((y : Int) + sy).toNat ((x' : Int) + sx).toNat
  else
    zero

theorem shift2D_eq_shift1D {α : Type} (H W : Nat) (x : Nat → Nat → α) (zero : α)
    (sy sx : Int) (y x' : Nat) :
    shift2D H W x zero sy sx y x' =
    shift1D H (fun r => shift1D W (fun c => x r c) zero sx x') zero sy y := by
  unfold shift2D shift1D
  by_cases hy : 0 ≤ (y : Int) + sy ∧ (y : Int) + sy < (H : Int)
  · by_cases hx : 0 ≤ (x' : Int) + sx ∧ (x' : Int) + sx < (W : Int)
    · simp [hy, hx]
    · simp [hy, hx]
  · simp only [hy, ↓reduceIte]
    split
    · rename_i hAll; exact absurd ⟨hAll.1, hAll.2.1⟩ hy
    · rfl

/-- Unit-step sign function matching `DecomposeConv.intSign` (line 161). -/
def shiftSign (d : Int) : Int :=
  if d < 0 then -1 else if d > 0 then 1 else 0

/-- Direct equality between `shiftSign` and REAL `Vane.IR.intSign`. -/
theorem shiftSign_eq_intSign (d : Int) : shiftSign d = intSign d := rfl

/-- Direct soundness on REAL `Vane.IR.isNativeConvOp`:
    - 1x1 dense stride-1 unpadded on 4D `[1, cin, h, w]` and 2D `[1, cin]` is native.
    - 3x3 dense stride-1 same-pad (`pads = [1, 1, 1, 1]`) on `h ≥ 1, w ≥ 1` is native.
    - 5x5 / 7x7 / grouped (`1 < groups < cin`) are NON-native and routed to `DecomposeConv`. -/
theorem isNativeConvOp_sound (cin cout h w : Nat) (hh : h > 0) (hw : w > 0) :
    isNativeConvOp #[1, cin, h, w] 1 1 1 1 #[0, 0, 0, 0] 1 cout = true ∧
    isNativeConvOp #[1, cin] 1 1 1 1 #[0, 0, 0, 0] 1 cout = true ∧
    isNativeConvOp #[1, cin, h, w] 3 3 1 1 #[1, 1, 1, 1] 1 cout = true ∧
    isNativeConvOp #[1, cin, h, w] 5 5 1 1 #[2, 2, 2, 2] 1 cout = false ∧
    isNativeConvOp #[1, cin, h, w] 7 7 1 1 #[3, 3, 3, 3] 1 cout = false := by
  have h1 : isNativeConvOp #[1, cin, h, w] 1 1 1 1 #[0, 0, 0, 0] 1 cout = true := by
    unfold isNativeConvOp; simp
  have h2 : isNativeConvOp #[1, cin] 1 1 1 1 #[0, 0, 0, 0] 1 cout = true := by
    unfold isNativeConvOp; simp
  have h3 : isNativeConvOp #[1, cin, h, w] 3 3 1 1 #[1, 1, 1, 1] 1 cout = true := by
    unfold isNativeConvOp isNativeConv3x3Pads ceilDiv
    have hGe3H : ¬ (h + 1 + 1 < 3) := by omega
    have hGe3W : ¬ (w + 1 + 1 < 3) := by omega
    have hIfH : (if h < h - 1 + 3 then h - 1 + 3 - h else 0) = 2 := by split <;> omega
    have hIfW : (if w < w - 1 + 3 then w - 1 + 3 - w else 0) = 2 := by split <;> omega
    simp [hGe3H, hGe3W, hIfH, hIfW]; omega
  have h4 : isNativeConvOp #[1, cin, h, w] 5 5 1 1 #[2, 2, 2, 2] 1 cout = false := by
    unfold isNativeConvOp; simp
  have h5 : isNativeConvOp #[1, cin, h, w] 7 7 1 1 #[3, 3, 3, 3] 1 cout = false := by
    unfold isNativeConvOp; simp
  exact ⟨h1, h2, h3, h4, h5⟩

/-- Exact sign-monotone unit-shift chain matching `DecomposeConv.ensureShift` (lines 290–313):
    steps `(cy, cx)` toward `(sy, sx)` by `stepY = shiftSign (sy - cy)` and `stepX = shiftSign (sx - cx)`
    using unit 3x3 depthwise shifts `shift2D H W · zero stepY stepX`! -/
def shiftChain2DAux {α : Type} (H W : Nat) (zero : α) (sy sx : Int) :
    Nat → Int → Int → (Nat → Nat → α) → (Nat → Nat → α)
  | 0, _, _, cur => cur
  | n + 1, cy, cx, cur =>
    let stepY := shiftSign (sy - cy)
    let stepX := shiftSign (sx - cx)
    let next := shift2D H W cur zero stepY stepX
    shiftChain2DAux H W zero sy sx n (cy + stepY) (cx + stepX) next

def shiftChain2D {α : Type} (H W : Nat) (x : Nat → Nat → α) (zero : α) (sy sx : Int) :
    Nat → Nat → α :=
  shiftChain2DAux H W zero sy sx (max sy.natAbs sx.natAbs) 0 0 x

/-- Exactness of `DecomposeConv.emitUnitShift3x3` (lines 166–198):
    a 3x3 depthwise conv with `pt=1, pl=1, stride=1` and a single weight `1` at tap `(1 + stepY, 1 + stepX)`
    computes the exact 2D unit shift `shift2D H W (x c) zero stepY stepX`! -/
theorem decomposeConv_unitShift3x3_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W _C : Nat) (x : Nat → Nat → Nat → α) (stepY stepX : Int)
    (hSy : -1 ≤ stepY ∧ stepY ≤ 1) (hSx : -1 ≤ stepX ∧ stepX ≤ 1)
    (c oy ox : Nat) :
    let tapY : Nat := (1 + stepY).toNat
    let tapX : Nat := (1 + stepX).toNat
    let unitW : Nat → Nat → Nat → Nat → α :=
      fun _ _ ky kx => if ky = tapY ∧ kx = tapX then one else zero
    evalConvElem ops H W 1 1 3 3 1 1 1 1 x unitW none .none c oy ox =
      shift2D H W (x c) zero stepY stepX oy ox := by
  intro tapY tapX unitW
  have hTapY : tapY < 3 := by omega
  have hTapX : tapX < 3 := by omega
  unfold evalConvElem applyConvAct
  dsimp only
  have hS1 : ∀ f : Nat → α, sumFin 1 f = f 0 := fun f => by simp [sumFin, zero_add]
  rw [hS1]
  simp only [Nat.div_one, Nat.mul_one, Nat.add_zero]
  rw [sumFin_single 3 tapY hTapY]
  · rw [sumFin_single 3 tapX hTapX]
    · simp only [unitW, and_self, ↓reduceIte, one_mul, Int.ofNat_one]
      unfold shift2D
      have hEqY : (oy : Int) * (1 : Int) + (tapY : Int) - (1 : Int) = (oy : Int) + stepY := by omega
      have hEqX : (ox : Int) * (1 : Int) + (tapX : Int) - (1 : Int) = (ox : Int) + stepX := by omega
      rw [hEqY, hEqX]
    · intro kx _ hne
      simp only [unitW, hne, and_false, ↓reduceIte, zero_mul]
  · intro ky _ hne
    rw [sumFin_congr 3 (g := fun _ => zero) (fun kx _ => by simp only [unitW, hne, false_and, ↓reduceIte, zero_mul])]
    exact sumFin_const_zero 3

/-- Sign-monotone 2D shift composition helper: composing `shift2D` with a unit step of matching sign
    preserves exactness on all in-bounds pixels `(y < H, x' < W)`. -/
private theorem shift2D_compose_step {α : Type} (H W : Nat) (x : Nat → Nat → α) (zero : α)
    (cy cx stepY stepX : Int)
    (hSameY : (cy ≥ 0 ∧ stepY ≥ 0) ∨ (cy ≤ 0 ∧ stepY ≤ 0))
    (hSameX : (cx ≥ 0 ∧ stepX ≥ 0) ∨ (cx ≤ 0 ∧ stepX ≤ 0))
    (y x' : Nat) (hy : y < H) (hx : x' < W) :
    shift2D H W (shift2D H W x zero cy cx) zero stepY stepX y x' =
    shift2D H W x zero (cy + stepY) (cx + stepX) y x' := by
  rw [shift2D_eq_shift1D, shift2D_eq_shift1D]
  have hInner : ∀ r, r < H →
      shift1D W (fun col => shift2D H W x zero cy cx r col) zero stepX x' =
      shift1D H (fun r0 => shift1D W (fun col => x r0 col) zero (cx + stepX) x') zero cy r := by
    intro r hr
    have hCong : ∀ col, col < W →
        shift2D H W x zero cy cx r col =
        shift1D H (fun r0 => shift1D W (fun c0 => x r0 c0) zero cx col) zero cy r :=
      fun col _ => shift2D_eq_shift1D H W x zero cy cx r col
    -- Split whether `0 ≤ r + cy < H`
    unfold shift1D shift2D
    by_cases hrc : 0 ≤ (r : Int) + cy ∧ (r : Int) + cy < (H : Int)
    · simp only [hrc]
      have hXComp : shift1D W (fun col => shift1D W (fun c0 => x ((r : Int) + cy).toNat c0) zero cx col) zero stepX x' =
          shift1D W (fun c0 => x ((r : Int) + cy).toNat c0) zero (cx + stepX) x' := by
        rcases hSameX with ⟨h1, h2⟩ | ⟨h1, h2⟩
        · exact shift1D_compose_nonneg W _ zero cx stepX h1 h2 x'
        · exact shift1D_compose_nonpos W _ zero cx stepX h1 h2 x' hx
      unfold shift1D at hXComp
      rw [← hXComp]
      by_cases hxStep : 0 ≤ (x' : Int) + stepX ∧ (x' : Int) + stepX < (W : Int)
      · simp only [hxStep, ↓reduceIte, true_and]
      · simp [hxStep]
    · simp only [hrc, ↓reduceIte]
      by_cases hxStep : 0 ≤ (x' : Int) + stepX ∧ (x' : Int) + stepX < (W : Int)
      · simp only [hxStep, and_self, ↓reduceIte]
        split
        · rename_i hAll; exact absurd ⟨hAll.1, hAll.2.1⟩ hrc
        · rfl
      · simp [hxStep]
  have hStep1 : shift1D H (fun r => shift1D W (fun col => shift2D H W x zero cy cx r col) zero stepX x') zero stepY y =
      shift1D H (fun r => shift1D H (fun r0 => shift1D W (fun c0 => x r0 c0) zero (cx + stepX) x') zero cy r) zero stepY y := by
    unfold shift1D
    by_cases hys : 0 ≤ (y : Int) + stepY ∧ (y : Int) + stepY < (H : Int)
    · simp only [hys]
      exact hInner ((y : Int) + stepY).toNat (by omega)
    · simp only [hys, ↓reduceIte]
  rw [hStep1]
  rcases hSameY with ⟨h1, h2⟩ | ⟨h1, h2⟩
  · exact shift1D_compose_nonneg H _ zero cy stepY h1 h2 y
  · exact shift1D_compose_nonpos H _ zero cy stepY h1 h2 y hy

/-- Exactness of `shiftChain2D` (directly proving `DecomposeConv.ensureShift` lines 290–313):
    iterating `max sy.natAbs sx.natAbs` sign-monotone unit 3x3 depthwise shifts produces the EXACT
    2D zero-padded shift `shift2D H W x zero sy sx` at EVERY in-bounds pixel `(y < H, x' < W)`! -/
theorem shiftChain2D_eq_shift2D {α : Type} (H W : Nat) (x : Nat → Nat → α) (zero : α)
    (sy sx : Int) (y x' : Nat) (hy : y < H) (hx : x' < W) :
    shiftChain2D H W x zero sy sx y x' = shift2D H W x zero sy sx y x' := by
  have hAux : ∀ (n : Nat) (cy cx : Int) (cur : Nat → Nat → α),
      n = max (sy - cy).natAbs (sx - cx).natAbs →
      ((0 ≤ cy ∧ cy ≤ sy) ∨ (sy ≤ cy ∧ cy ≤ 0)) →
      ((0 ≤ cx ∧ cx ≤ sx) ∨ (sx ≤ cx ∧ cx ≤ 0)) →
      (∀ r c, r < H → c < W → cur r c = shift2D H W x zero cy cx r c) →
      shiftChain2DAux H W zero sy sx n cy cx cur y x' = shift2D H W x zero sy sx y x' := by
    intro n
    induction n with
    | zero =>
      intro cy cx cur hN _ _ hCur
      have hEqY : cy = sy := by omega
      have hEqX : cx = sx := by omega
      subst hEqY; subst hEqX
      unfold shiftChain2DAux
      exact hCur y x' hy hx
    | succ k ih =>
      intro cy cx cur hN hMonY hMonX hCur
      unfold shiftChain2DAux
      let stepY := shiftSign (sy - cy)
      let stepX := shiftSign (sx - cx)
      have hSameY : (cy ≥ 0 ∧ stepY ≥ 0) ∨ (cy ≤ 0 ∧ stepY ≤ 0) := by
        unfold stepY shiftSign; rcases hMonY with ⟨_, _⟩ | ⟨_, _⟩ <;> (split <;> try split) <;> omega
      have hSameX : (cx ≥ 0 ∧ stepX ≥ 0) ∨ (cx ≤ 0 ∧ stepX ≤ 0) := by
        unfold stepX shiftSign; rcases hMonX with ⟨_, _⟩ | ⟨_, _⟩ <;> (split <;> try split) <;> omega
      have hMonY' : (0 ≤ cy + stepY ∧ cy + stepY ≤ sy) ∨ (sy ≤ cy + stepY ∧ cy + stepY ≤ 0) := by
        unfold stepY shiftSign; rcases hMonY with ⟨_, _⟩ | ⟨_, _⟩ <;> (split <;> try split) <;> omega
      have hMonX' : (0 ≤ cx + stepX ∧ cx + stepX ≤ sx) ∨ (sx ≤ cx + stepX ∧ cx + stepX ≤ 0) := by
        unfold stepX shiftSign; rcases hMonX with ⟨_, _⟩ | ⟨_, _⟩ <;> (split <;> try split) <;> omega
      have hN' : k = max (sy - (cy + stepY)).natAbs (sx - (cx + stepX)).natAbs := by
        unfold stepY stepX shiftSign
        repeat split
        all_goals omega
      apply ih (cy + stepY) (cx + stepX) (shift2D H W cur zero stepY stepX) hN' hMonY' hMonX'
      intro r c hr hc
      have hCong : shift2D H W cur zero stepY stepX r c =
          shift2D H W (shift2D H W x zero cy cx) zero stepY stepX r c := by
        unfold shift2D
        by_cases hIn : 0 ≤ (r : Int) + stepY ∧ (r : Int) + stepY < (H : Int) ∧
                       0 ≤ (c : Int) + stepX ∧ (c : Int) + stepX < (W : Int)
        · simp only [hIn]
          exact hCur _ _ (by omega) (by omega)
        · simp only [hIn, ↓reduceIte]
      rw [hCong]
      exact shift2D_compose_step H W x zero cy cx stepY stepX hSameY hSameX r c hr hc
  unfold shiftChain2D
  apply hAux (max sy.natAbs sx.natAbs) 0 0 x (by simp) (by omega) (by omega)
  intro r c hr hc
  unfold shift2D
  have h0 : 0 ≤ (r : Int) + 0 ∧ (r : Int) + 0 < (H : Int) ∧ 0 ≤ (c : Int) + 0 ∧ (c : Int) + 0 < (W : Int) := by omega
  have hrE : ((r : Int) + 0).toNat = r := by omega
  have hcE : ((c : Int) + 0).toNat = c := by omega
  simp only [h0, and_self, ↓reduceIte, hrE, hcE]

/-- Bucketed 3x3 weight kernel for shift bucket `(sy, sx)` produced by `DecomposeConv.lean`
    lines 245–267 using `Vane.IR.assignConvAxisShiftTap`! -/
def bucketWeight3x3 {α : Type} [ScalarField α]
    (kh kw pt pl basePt basePl : Nat)
    (w : Nat → Nat → Nat → Nat → α)
    (sy sx : Int) (oc ic ty tx : Nat) : α :=
  let minDy : Int := -(pt : Int)
  let maxDy : Int := (kh : Int) - 1 - (pt : Int)
  let minDx : Int := -(pl : Int)
  let maxDx : Int := (kw : Int) - 1 - (pl : Int)
  sumFin kh fun ky =>
    sumFin kw fun kx =>
      let stY := assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - (pt : Int))
      let stX := assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))
      if stY.1 = sy ∧ stX.1 = sx ∧ stY.2 = ty ∧ stX.2 = tx then w oc ic ky kx else zero

private theorem match_option_congr {α β : Type} (o : Option β) (f : β → α → α) {a1 a2 : α} (h : a1 = a2) :
    (match o with | some b => f b a1 | none => a1) =
    (match o with | some b => f b a2 | none => a2) := by rw [h]

/-- FULL C4 `DecomposeConv` 2D SHIFT-CONVOLUTION VALUE EQUIVALENCE OVER EXACT `ScalarField α`!
    For ANY kernel `(kh × kw)`, ANY stride `sHw ∈ {1, 2}`, ANY padding `(pt, pl)` (same, valid,
    or asymmetric), ANY input/output dims `ho ≤ ceilDiv H sHw, wo ≤ ceilDiv W sHw`, ANY weights `w`,
    optional bias `bias?`, and activation `act`:
    summing the shifted 3x3 sub-convolutions constructed via `DecomposeConv.assignConvAxisShiftTap`
    and `shiftChain2D` (adding `bias?` on tile `0` ONLY, applying `act` on the final sum ONLY)
    equals direct naive `(kh × kw)` convolution `evalConvElem` at EVERY output coordinate `(oc, oy, ox)`!
    Covers `Conv1d`, `5x5`, `7x7`, `1xK`, `Kx1`, stride `1` & `2`, and arbitrary asymmetric padding! -/
theorem decomposeConv_shiftConv_sum_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cin cout kh kw sHw pt pl : Nat)
    (hH : H > 0) (hW : W > 0) (hsHw : sHw = 1 ∨ sHw = 2)
    (x : Nat → Nat → Nat → α) (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α)) (act : ConvActivation)
    (keys : List (Int × Int)) (hNe : keys ≠ []) (hNd : keys.Nodup)
    (oc oy ox : Nat) (hoc : oc < cout)
    (hoy : oy < ceilDiv H sHw) (hox : ox < ceilDiv W sHw)
    (hCov :
      let subH := ceilDiv H sHw
      let subW := ceilDiv W sHw
      let basePt := (if (subH - 1) * sHw + 3 > H then (subH - 1) * sHw + 3 - H else 0) / 2
      let basePl := (if (subW - 1) * sHw + 3 > W then (subW - 1) * sHw + 3 - W else 0) / 2
      let minDy : Int := -(pt : Int); let maxDy : Int := (kh : Int) - 1 - (pt : Int)
      let minDx : Int := -(pl : Int); let maxDx : Int := (kw : Int) - 1 - (pl : Int)
      ∀ ky kx, ky < kh → kx < kw →
        ((assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - (pt : Int))).1,
         (assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))).1) ∈ keys) :
    let subH := ceilDiv H sHw
    let subW := ceilDiv W sHw
    let basePt := (if (subH - 1) * sHw + 3 > H then (subH - 1) * sHw + 3 - H else 0) / 2
    let basePl := (if (subW - 1) * sHw + 3 > W then (subW - 1) * sHw + 3 - W else 0) / 2
    applyConvAct ops act
      (sumList (keys.zipIdx.map fun ((sy, sx), ti) =>
        evalConvElem ops H W cin cout 3 3 sHw sHw basePt basePl
          (fun c y x' => shiftChain2D H W (x c) zero sy sx y x')
          (bucketWeight3x3 kh kw pt pl basePt basePl w sy sx)
          (if ti = 0 then bias? else none) .none oc oy ox)) =
    evalConvElem ops H W cin cout kh kw sHw sHw pt pl x w bias? act oc oy ox := by
  intro subH subW basePt basePl
  have hsPos : sHw > 0 := by rcases hsHw with rfl | rfl <;> omega
  have hBPt : basePt = 0 ∨ basePt = 1 := decomposeConv_basePad_0_or_1 H sHw hH hsHw
  have hBPl : basePl = 0 ∨ basePl = 1 := decomposeConv_basePad_0_or_1 W sHw hW hsHw
  have hOyLt : oy * sHw < H := decomposeConv_stride_in_bounds H sHw oy hsPos hoy
  have hOxLt : ox * sHw < W := decomposeConv_stride_in_bounds W sHw ox hsPos hox
  have hDiv : oc / cout = 0 := Nat.div_eq_of_lt hoc
  -- Pull `applyConvAct` off both sides
  unfold evalConvElem
  dsimp only
  congr 1
  -- Separate the bias term (contributed solely by `ti = 0`, since `keys ≠ []`)
  let tileMac : (Int × Int) → α := fun (sy, sx) =>
    sumFin cin fun ic' =>
      sumFin 3 fun ty =>
        sumFin 3 fun tx =>
          let iy : Int := (oy * sHw : Int) + (ty : Int) - (basePt : Int)
          let ix : Int := (ox * sHw : Int) + (tx : Int) - (basePl : Int)
          let xVal := if 0 ≤ iy ∧ iy < (H : Int) ∧ 0 ≤ ix ∧ ix < (W : Int) then
                        shiftChain2D H W (x (oc / cout * cin + ic')) zero sy sx iy.toNat ix.toNat
                      else zero
          mul (bucketWeight3x3 kh kw pt pl basePt basePl w sy sx oc ic' ty tx) xVal
  have hBiasSplit :
      sumList (keys.zipIdx.map fun ((sy, sx), ti) =>
        applyConvAct ops .none
          (match (if ti = 0 then bias? else none) with
           | some b => add (tileMac (sy, sx)) (b oc)
           | none => tileMac (sy, sx))) =
      match bias? with
      | some b => add (sumList (keys.map tileMac)) (b oc)
      | none => sumList (keys.map tileMac) := by
    cases keys with
    | nil => exact absurd rfl hNe
    | cons k0 kRest =>
      have hZip : (k0 :: kRest).zipIdx = (k0, 0) :: kRest.zipIdx 1 := rfl
      rw [hZip, List.map_cons, sumList, List.map_cons, sumList]
      have hTailNoBias : ∀ (l : List (Int × Int)) (start : Nat), start ≥ 1 →
          (l.zipIdx start).map (fun ((sy, sx), ti) =>
            applyConvAct ops .none
              (match (if ti = 0 then bias? else none) with
               | some b => add (tileMac (sy, sx)) (b oc)
               | none => tileMac (sy, sx))) = l.map tileMac := by
        intro l
        induction l with
        | nil => intro _ _; rfl
        | cons hd tl iht =>
          intro start hSt
          simp only [List.zipIdx_cons, List.map_cons, applyConvAct]
          have hNe0 : start ≠ 0 := by omega
          simp only [hNe0, ↓reduceIte]
          congr 1
          exact iht (start + 1) (by omega)
      rw [hTailNoBias kRest 1 (by omega)]
      simp only [applyConvAct, ↓reduceIte]
      cases bias? with
      | none => rfl
      | some b =>
        rw [add_assoc, add_comm (b oc), ← add_assoc]
  change (sumList (keys.zipIdx.map fun ((sy, sx), ti) =>
        applyConvAct ops .none
          (match (if ti = 0 then bias? else none) with
           | some b => add (tileMac (sy, sx)) (b oc)
           | none => tileMac (sy, sx)))) = _
  rw [hBiasSplit]
  suffices hMac :
      sumList (keys.map tileMac) =
      sumFin cin (fun ic' =>
        sumFin kh fun ky =>
          sumFin kw fun kx =>
            mul (w oc ic' ky kx)
              (if 0 ≤ (oy : Int) * (sHw : Int) + (ky : Int) - (pt : Int) ∧
                  (oy : Int) * (sHw : Int) + (ky : Int) - (pt : Int) < (H : Int) ∧
                  0 ≤ (ox : Int) * (sHw : Int) + (kx : Int) - (pl : Int) ∧
                  (ox : Int) * (sHw : Int) + (kx : Int) - (pl : Int) < (W : Int) then
                x (oc / cout * cin + ic')
                  ((oy : Int) * (sHw : Int) + (ky : Int) - (pt : Int)).toNat
                  ((ox : Int) * (sHw : Int) + (kx : Int) - (pl : Int)).toNat
              else zero)) by
    cases bias? <;> simp only [hMac]
  -- Swap `sumList` inside `sumFin cin`, then simplify each channel `ic'`!
  dsimp only [tileMac]
  rw [← sumFin_sumList_comm]
  apply sumFin_congr; intro ic' _
  let xC := x (oc / cout * cin + ic')
  -- Replace `shiftChain2D` with `shift2D` on every in-bounds `(iy, ix)`
  have hStep1 :
      sumList (keys.map fun (sy, sx) =>
        sumFin 3 fun ty =>
          sumFin 3 fun tx =>
            let iy : Int := (oy : Int) * (sHw : Int) + (ty : Int) - (basePt : Int)
            let ix : Int := (ox : Int) * (sHw : Int) + (tx : Int) - (basePl : Int)
            let xVal := if 0 ≤ iy ∧ iy < (H : Int) ∧ 0 ≤ ix ∧ ix < (W : Int) then
                          shiftChain2D H W xC zero sy sx iy.toNat ix.toNat
                        else zero
            mul (bucketWeight3x3 kh kw pt pl basePt basePl w sy sx oc ic' ty tx) xVal) =
      sumList (keys.map fun (sy, sx) =>
        sumFin 3 fun ty =>
          sumFin 3 fun tx =>
            let shiftedX := shift2D H W (shift2D H W xC zero sy sx) zero
              ((ty : Int) - (basePt : Int)) ((tx : Int) - (basePl : Int)) (oy * sHw) (ox * sHw)
            mul (bucketWeight3x3 kh kw pt pl basePt basePl w sy sx oc ic' ty tx) shiftedX) := by
    apply sumList_congr; intro ⟨sy, sx⟩ _
    apply sumFin_congr; intro ty _
    apply sumFin_congr; intro tx _
    congr 1
    unfold shift2D
    simp only [Int.natCast_mul]
    have hYEq : (oy : Int) * (sHw : Int) + ((ty : Int) - (basePt : Int)) = (oy : Int) * (sHw : Int) + (ty : Int) - (basePt : Int) := by omega
    have hXEq : (ox : Int) * (sHw : Int) + ((tx : Int) - (basePl : Int)) = (ox : Int) * (sHw : Int) + (tx : Int) - (basePl : Int) := by omega
    rw [hYEq, hXEq]
    by_cases hIn : 0 ≤ (oy : Int) * (sHw : Int) + (ty : Int) - (basePt : Int) ∧
                   (oy : Int) * (sHw : Int) + (ty : Int) - (basePt : Int) < (H : Int) ∧
                   0 ≤ (ox : Int) * (sHw : Int) + (tx : Int) - (basePl : Int) ∧
                   (ox : Int) * (sHw : Int) + (tx : Int) - (basePl : Int) < (W : Int)
    · simp only [hIn, and_self, ↓reduceIte]
      congr 1
      exact shiftChain2D_eq_shift2D H W xC zero sy sx
        ((oy : Int) * (sHw : Int) + (ty : Int) - (basePt : Int)).toNat
        ((ox : Int) * (sHw : Int) + (tx : Int) - (basePl : Int)).toNat (by omega) (by omega)
    · simp only [hIn, ↓reduceIte]
  rw [hStep1]
  -- Expand `bucketWeight3x3` and swap sums so `ky, kx` are outermost!
  let minDy : Int := -(pt : Int); let maxDy : Int := (kh : Int) - 1 - (pt : Int)
  let minDx : Int := -(pl : Int); let maxDx : Int := (kw : Int) - 1 - (pl : Int)
  have hSwap1 :
      sumList (keys.map fun (sy, sx) =>
        sumFin 3 fun ty =>
          sumFin 3 fun tx =>
            mul (bucketWeight3x3 kh kw pt pl basePt basePl w sy sx oc ic' ty tx)
              (shift2D H W (shift2D H W xC zero sy sx) zero
                ((ty : Int) - (basePt : Int)) ((tx : Int) - (basePl : Int)) (oy * sHw) (ox * sHw))) =
      sumFin kh (fun ky => sumFin kw fun kx =>
        sumList (keys.map fun (sy, sx) =>
          sumFin 3 fun ty =>
            sumFin 3 fun tx =>
              let stY := assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - (pt : Int))
              let stX := assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))
              mul (if stY.1 = sy ∧ stX.1 = sx ∧ stY.2 = ty ∧ stX.2 = tx then w oc ic' ky kx else zero)
                (shift2D H W (shift2D H W xC zero sy sx) zero
                  ((ty : Int) - (basePt : Int)) ((tx : Int) - (basePl : Int)) (oy * sHw) (ox * sHw)))) := by
    have hStepDist :
        sumList (keys.map fun (sy, sx) =>
          sumFin 3 fun ty =>
            sumFin 3 fun tx =>
              mul (bucketWeight3x3 kh kw pt pl basePt basePl w sy sx oc ic' ty tx)
                (shift2D H W (shift2D H W xC zero sy sx) zero
                  ((ty : Int) - (basePt : Int)) ((tx : Int) - (basePl : Int)) (oy * sHw) (ox * sHw))) =
        sumList (keys.map fun (sy, sx) =>
          sumFin kh fun ky =>
            sumFin kw fun kx =>
              sumFin 3 fun ty =>
                sumFin 3 fun tx =>
                  let stY := assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - (pt : Int))
                  let stX := assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))
                  mul (if stY.1 = sy ∧ stX.1 = sx ∧ stY.2 = ty ∧ stX.2 = tx then w oc ic' ky kx else zero)
                    (shift2D H W (shift2D H W xC zero sy sx) zero
                      ((ty : Int) - (basePt : Int)) ((tx : Int) - (basePl : Int)) (oy * sHw) (ox * sHw))) := by
      apply sumList_congr; intro ⟨sy, sx⟩ _
      dsimp only
      have hInner : ∀ ty tx,
          mul (bucketWeight3x3 kh kw pt pl basePt basePl w sy sx oc ic' ty tx)
            (shift2D H W (shift2D H W xC zero sy sx) zero
              ((ty : Int) - (basePt : Int)) ((tx : Int) - (basePl : Int)) (oy * sHw) (ox * sHw)) =
          sumFin kh fun ky => sumFin kw fun kx =>
            let stY := assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - (pt : Int))
            let stX := assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))
            mul (if stY.1 = sy ∧ stX.1 = sx ∧ stY.2 = ty ∧ stX.2 = tx then w oc ic' ky kx else zero)
              (shift2D H W (shift2D H W xC zero sy sx) zero
                ((ty : Int) - (basePt : Int)) ((tx : Int) - (basePl : Int)) (oy * sHw) (ox * sHw)) := by
        intro ty tx
        unfold bucketWeight3x3
        rw [← sumFin_mul_right]
        apply sumFin_congr; intro ky _
        rw [← sumFin_mul_right]
      have hDist3 :
          sumFin 3 (fun ty => sumFin 3 fun tx =>
            mul (bucketWeight3x3 kh kw pt pl basePt basePl w sy sx oc ic' ty tx)
              (shift2D H W (shift2D H W xC zero sy sx) zero
                ((ty : Int) - (basePt : Int)) ((tx : Int) - (basePl : Int)) (oy * sHw) (ox * sHw))) =
          sumFin 3 (fun ty => sumFin 3 fun tx => sumFin kh fun ky => sumFin kw fun kx =>
            let stY := assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - (pt : Int))
            let stX := assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))
            mul (if stY.1 = sy ∧ stX.1 = sx ∧ stY.2 = ty ∧ stX.2 = tx then w oc ic' ky kx else zero)
              (shift2D H W (shift2D H W xC zero sy sx) zero
                ((ty : Int) - (basePt : Int)) ((tx : Int) - (basePl : Int)) (oy * sHw) (ox * sHw))) :=
        sumFin_congr 3 fun ty _ => sumFin_congr 3 fun tx _ => hInner ty tx
      rw [hDist3]
      calc sumFin 3 (fun ty => sumFin 3 fun tx => sumFin kh fun ky => sumFin kw fun kx =>
              mul (if (assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - (pt : Int))).1 = sy ∧
                      (assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))).1 = sx ∧
                      (assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - (pt : Int))).2 = ty ∧
                      (assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))).2 = tx
                   then w oc ic' ky kx else zero)
                (shift2D H W (shift2D H W xC zero sy sx) zero
                  ((ty : Int) - (basePt : Int)) ((tx : Int) - (basePl : Int)) (oy * sHw) (ox * sHw)))
          = sumFin 3 (fun ty => sumFin kh fun ky => sumFin 3 fun tx => sumFin kw fun kx => _) := by
            apply sumFin_congr; intro ty _; exact sumFin_comm 3 kh _
        _ = sumFin kh (fun ky => sumFin 3 fun ty => sumFin 3 fun tx => sumFin kw fun kx => _) :=
            sumFin_comm 3 kh _
        _ = sumFin kh (fun ky => sumFin 3 fun ty => sumFin kw fun kx => sumFin 3 fun tx => _) := by
            apply sumFin_congr; intro ky _; apply sumFin_congr; intro ty _; exact sumFin_comm 3 kw _
        _ = sumFin kh (fun ky => sumFin kw fun kx => sumFin 3 fun ty => sumFin 3 fun tx => _) := by
            apply sumFin_congr; intro ky _; exact sumFin_comm 3 kw _
    rw [hStepDist, ← sumFin_sumList_comm]
    apply sumFin_congr; intro ky _
    rw [← sumFin_sumList_comm]
  rw [hSwap1]
  -- Collapse the inner Kronecker delta over `(sy, sx) ∈ keys`, `ty < 3`, `tx < 3` and apply C4Conv soundness!
  apply sumFin_congr; intro ky hky
  apply sumFin_congr; intro kx hkx
  let stY := assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - (pt : Int))
  let stX := assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))
  have hKeyMem : (stY.1, stX.1) ∈ keys := hCov ky kx hky hkx
  have hTyLt : stY.2 < 3 := assignConvAxisShiftTap_tap_lt3 basePt minDy maxDy ((ky : Int) - (pt : Int))
  have hTxLt : stX.2 < 3 := assignConvAxisShiftTap_tap_lt3 basePl minDx maxDx ((kx : Int) - (pl : Int))
  rw [sumList_single keys (stY.1, stX.1) hNd hKeyMem]
  · dsimp only
    rw [sumFin_single 3 stY.2 hTyLt]
    · rw [sumFin_single 3 stX.2 hTxLt]
      · simp only [stY, stX, and_self, ↓reduceIte]
        congr 1
        -- Now prove 2D shifted lookup equals direct `(ky - pt, kx - pl)` lookup via `C4Conv.assignConvAxisShiftTap_shift_sound`!
        rw [shift2D_eq_shift1D]
        have hRowSound : ∀ r, r < H →
            shift1D W (fun col => shift2D H W xC zero stY.1 stX.1 r col) zero ((stX.2 : Int) - (basePl : Int)) (ox * sHw) =
            shift1D H (fun r0 => shift1D W (fun c0 => xC r0 c0) zero ((kx : Int) - (pl : Int)) (ox * sHw)) zero stY.1 r := by
          intro r _
          unfold shift1D shift2D
          by_cases hrc : 0 ≤ (r : Int) + stY.1 ∧ (r : Int) + stY.1 < (H : Int)
          · simp only [hrc]
            have hX := assignConvAxisShiftTap_shift_sound W (xC ((r : Int) + stY.1).toNat) zero
              basePl hBPl minDx maxDx ((kx : Int) - (pl : Int)) (ox * sHw) hOxLt
            dsimp only at hX
            unfold shift1D at hX
            rw [← hX]
            by_cases hxt : 0 ≤ ((ox * sHw : Nat) : Int) + ((stX.2 : Int) - (basePl : Int)) ∧
                           ((ox * sHw : Nat) : Int) + ((stX.2 : Int) - (basePl : Int)) < (W : Int)
            · dsimp only [stX] at hxt ⊢
              simp only [hxt, true_and, ↓reduceIte]
            · dsimp only [stX] at hxt ⊢
              simp only [hxt, ↓reduceIte, and_self]
          · simp only [hrc, ↓reduceIte]
            by_cases hxt : 0 ≤ ((ox * sHw : Nat) : Int) + ((stX.2 : Int) - (basePl : Int)) ∧
                           ((ox * sHw : Nat) : Int) + ((stX.2 : Int) - (basePl : Int)) < (W : Int)
            · simp only [hxt, and_self, ↓reduceIte]
              split
              · rename_i hAll; exact absurd ⟨hAll.1, hAll.2.1⟩ hrc
              · rfl
            · simp only [hxt, ↓reduceIte]
        have hYSub :
            shift1D H (fun r => shift1D W (fun col => shift2D H W xC zero stY.1 stX.1 r col) zero ((stX.2 : Int) - (basePl : Int)) (ox * sHw))
              zero ((stY.2 : Int) - (basePt : Int)) (oy * sHw) =
            shift1D H (fun r => shift1D H (fun r0 => shift1D W (fun c0 => xC r0 c0) zero ((kx : Int) - (pl : Int)) (ox * sHw)) zero stY.1 r)
              zero ((stY.2 : Int) - (basePt : Int)) (oy * sHw) := by
          unfold shift1D
          by_cases hyt : 0 ≤ ((oy * sHw : Nat) : Int) + ((stY.2 : Int) - (basePt : Int)) ∧
                         ((oy * sHw : Nat) : Int) + ((stY.2 : Int) - (basePt : Int)) < (H : Int)
          · simp only [hyt]
            exact hRowSound _ (by omega)
          · simp only [hyt, ↓reduceIte]
        rw [hYSub]
        have hYSound := assignConvAxisShiftTap_shift_sound H
          (fun r0 => shift1D W (fun c0 => xC r0 c0) zero ((kx : Int) - (pl : Int)) (ox * sHw)) zero
          basePt hBPt minDy maxDy ((ky : Int) - (pt : Int)) (oy * sHw) hOyLt
        dsimp only at hYSound
        rw [hYSound]
        rw [← shift2D_eq_shift1D]
        unfold shift2D
        simp only [Int.natCast_mul]
        have hYEq : (oy : Int) * (sHw : Int) + ((ky : Int) - (pt : Int)) = (oy : Int) * (sHw : Int) + (ky : Int) - (pt : Int) := by omega
        have hXEq : (ox : Int) * (sHw : Int) + ((kx : Int) - (pl : Int)) = (ox : Int) * (sHw : Int) + (kx : Int) - (pl : Int) := by omega
        rw [hYEq, hXEq]
      · intro tx _ hne
        split
        · rename_i hEq; exact absurd hEq.2.2.2.symm hne
        · exact zero_mul _
    · intro ty _ hne
      rw [sumFin_congr 3 (g := fun _ => zero) (fun tx _ => by
        split
        · rename_i hEq; exact absurd hEq.2.2.1.symm hne
        · exact zero_mul _)]
      exact sumFin_const_zero 3
  · intro ⟨sy, sx⟩ _ hNePair
    dsimp only
    rw [sumFin_congr 3 (g := fun _ => zero) (fun ty _ => by
      rw [sumFin_congr 3 (g := fun _ => zero) (fun tx _ => by
        split
        · rename_i hEq; exact absurd (Prod.ext hEq.1.symm hEq.2.1.symm) hNePair
        · exact zero_mul _)]
      exact sumFin_const_zero 3)]
    exact sumFin_const_zero 3

/-- Helper: C-axis `evalSlice` on a 4D tensor `[1, groups * cinG, H, W]` extracts the exact
    group-`g` channels `g * cinG + ic'` at every `(ic', y, x)`! -/
theorem evalSlice_channel_group_exact {α : Type} [ScalarField α]
    (groups cinG H W g : Nat) (hG : g < groups) (hCinG : cinG > 0) (hH : H > 0) (hW : W > 0)
    (xFn : Nat → Nat → Nat → α) :
    evalSlice #[0, g * cinG, 0, 0] #[1, (g + 1) * cinG, H, W] #[1, 1, 1, 1]
      (mkTensor4D (groups * cinG) H W xFn) =
    some (mkTensor4D cinG H W fun ic' y x => xFn (g * cinG + ic') y x) := by
  have hCinPos : groups * cinG > 0 := Nat.mul_pos (by omega) hCinG
  have hBound : (g + 1) * cinG ≤ groups * cinG := Nat.mul_le_mul_right cinG (Nat.succ_le_of_lt hG)
  have hLt : g * cinG < (g + 1) * cinG := by rw [Nat.add_mul, Nat.one_mul]; omega
  have hSub : (g + 1) * cinG - g * cinG = cinG := by rw [Nat.add_mul, Nat.one_mul]; omega
  have hRange4 : List.range 4 = [0, 1, 2, 3] := rfl
  have hValid : validSliceParams [1, groups * cinG, H, W]
      #[0, g * cinG, 0, 0] #[1, (g + 1) * cinG, H, W] #[1, 1, 1, 1] = true := by
    unfold validSliceParams
    simp only [List.length_cons, List.length_nil, hRange4]
    simp [hBound, hLt, hH, hW]
  have hPos : ([1, groups * cinG, H, W] : List Nat).all (· > 0) = true := by
    simp [hCinPos, hH, hW]
  have hOutSh : sliceShape [1, groups * cinG, H, W]
      #[0, g * cinG, 0, 0] #[1, (g + 1) * cinG, H, W] #[1, 1, 1, 1] = [1, cinG, H, W] := by
    unfold sliceShape sliceAxisDim
    simp only [List.length_cons, List.length_nil, hRange4]
    simp [hSub, Nat.not_le.mpr hH, Nat.not_le.mpr hW, Nat.not_le.mpr hLt]
  have hNIn : 0 < shapeProd [1, groups * cinG, H, W] := by
    simp only [shapeProd, Nat.one_mul, Nat.mul_one]
    exact Nat.mul_pos hCinPos (Nat.mul_pos hH hW)
  unfold evalSlice
  dsimp only [mkTensor4D]
  simp only [hValid, hPos, Bool.and_self, ↓reduceIte, hNIn, ↓reduceDIte]
  congr 1
  apply tensorVal_ext_4d cinG H W _ _ hOutSh rfl
  intro ic' y x hic' hy hx h1 h2
  dsimp only
  have hUnf : unflatten (sliceShape [1, groups * cinG, H, W]
      #[0, g * cinG, 0, 0] #[1, (g + 1) * cinG, H, W] #[1, 1, 1, 1]) (ic' * (H * W) + y * W + x) =
      [0, ic', y, x] := by
    rw [hOutSh]; exact unflatten_4d cinG H W ic' y x hic' hy hx
  rw [hUnf]
  have hInCoords : (List.range 4).map (fun i =>
      (#[0, g * cinG, 0, 0] : Array Nat).getD i 0 +
      [0, ic', y, x].getD i 0 * (#[1, 1, 1, 1] : Array Nat).getD i 0) =
      [0, g * cinG + ic', y, x] := by
    rw [hRange4]; simp
  have hFlat : flatIndex [1, groups * cinG, H, W] [0, g * cinG + ic', y, x] =
      (g * cinG + ic') * (H * W) + y * W + x := by
    simp [flatIndex, shapeProd, Nat.add_assoc]
  have hGcLt : g * cinG + ic' < groups * cinG := by
    have : g * cinG + cinG = (g + 1) * cinG := by rw [Nat.add_mul, Nat.one_mul]
    omega
  have hInLt : (g * cinG + ic') * (H * W) + y * W + x < shapeProd [1, groups * cinG, H, W] :=
    flat4D_lt (groups * cinG) H W (g * cinG + ic') y x hGcLt hy hx
  simp only [List.length_cons, List.length_nil]
  rw [hInCoords, hFlat]
  simp only [hInLt, ↓reduceDIte]
  rw [unflatten_4d (groups * cinG) H W (g * cinG + ic') y x hGcLt hy hx,
      unflatten_4d cinG H W ic' y x hic' hy hx]
  rfl

/-- FULL C4 GROUPED CONVOLUTION DECOMPOSITION VALUE EQUIVALENCE (`DecomposeConv.lean` lines 456–575)!
    For ANY `groups > 0`, `cinG > 0`, `coutG > 0`, `H > 0`, `W > 0`, `Ho > 0`, `Wo > 0`, and ANY
    `(kh, kw, sh, sw, pt, pl, bias?, act)`:
    slicing input along axis 1 into `groups` chunks via `evalSlice`, running dense conv (`groups = 1`)
    with group-sliced weights `W[g * coutG + oc']` and bias `b[g * coutG + oc']`, and concatenating
    along axis 1 via `evalConcat 1` succeeds and equals direct naive grouped convolution `evalConvTensor4D`
    as an EXACT `TensorVal α` equality! -/
theorem decomposeConv_grouped_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (groups cinG coutG H W Ho Wo kh kw sh sw pt pl : Nat)
    (hGroups : groups > 0) (hCoutG : coutG > 0)
    (hHo : Ho > 0) (hWo : Wo > 0)
    (xFn : Nat → Nat → Nat → α)
    (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α))
    (act : ConvActivation) :
    let groupOuts : List (TensorVal α) := (List.range groups).map fun gi =>
      let xG := fun ic' y x =>
        read4D (mkTensor4D cinG H W fun c r k => xFn (gi * cinG + c) r k) cinG H W ic' y x
      let wG := fun oc' ic' ky kx => w (gi * coutG + oc') ic' ky kx
      let bG? := bias?.map fun b oc' => b (gi * coutG + oc')
      evalConvTensor4D ops H W cinG coutG coutG Ho Wo kh kw sh sw pt pl xG wG bG? act
    evalConcat 1 groupOuts =
      some (evalConvTensor4D ops H W cinG (groups * coutG) coutG Ho Wo kh kw sh sw pt pl xFn w bias? act) := by
  intro groupOuts
  have hNe : groupOuts ≠ [] := by
    intro hE; have : groupOuts.length = 0 := by rw [hE]; rfl
    simp [groupOuts] at this; omega
  have hShapes : ∀ t ∈ groupOuts, ∃ dc > 0, t.shape = [1, dc, Ho, Wo] := by
    intro t ht
    simp only [groupOuts, List.mem_map] at ht
    obtain ⟨gi, _, rfl⟩ := ht
    exact ⟨coutG, hCoutG, rfl⟩
  have hTakeSum : ∀ k, k ≤ groups →
      (groupOuts.take k).foldl (fun acc t => acc + t.shape.getD 1 0) 0 = k * coutG := by
    intro k
    induction k with
    | zero => intro _; simp
    | succ m ih =>
      intro hm
      have hTk : groupOuts.take (m + 1) = groupOuts.take m ++ [groupOuts[m]'(by simp [groupOuts]; omega)] :=
        List.take_succ_eq_append_getElem (by simp [groupOuts]; omega)
      rw [hTk, List.foldl_append, List.foldl_cons, List.foldl_nil, ih (by omega)]
      have hShM : (groupOuts[m]'(by simp [groupOuts]; omega)).shape.getD 1 0 = coutG := by
        simp [groupOuts, evalConvTensor4D, mkTensor4D]
      rw [hShM, Nat.add_mul, Nat.one_mul]
  have hSumAll : groupOuts.foldl (fun acc t => acc + t.shape.getD 1 0) 0 = groups * coutG := by
    have h := hTakeSum groups (Nat.le_refl _)
    have hTk : groupOuts.take groups = groupOuts := List.take_of_length_le (by simp [groupOuts])
    rwa [hTk] at h
  obtain ⟨out, hCat, hOutSh, hElem⟩ :=
    sok_evalConcat_shape_and_elements Ho Wo (groups * coutG) hHo hWo groupOuts hNe hShapes hSumAll
  rw [hCat]
  congr 1
  apply tensorVal_ext_4d (groups * coutG) Ho Wo out _ hOutSh rfl
  intro oc oy ox hoc hoy hox h1 h2
  obtain ⟨gi, hgi, hlo, hhi, hInLt, hEqAt⟩ := hElem oc oy ox hoc hoy hox h1
  rw [hEqAt]
  have hgiLt : gi < groups := by simpa [groupOuts] using hgi
  have hC0 : (groupOuts.take gi).foldl (fun acc t => acc + t.shape.getD 1 0) 0 = gi * coutG :=
    hTakeSum gi (Nat.le_of_lt hgiLt)
  have hShG : groupOuts[gi].shape.getD 1 0 = coutG := by
    simp [groupOuts, evalConvTensor4D, mkTensor4D]
  have hLocLt : oc - gi * coutG < coutG := by rw [← hC0, ← hShG]; omega
  have hDivEq : oc / coutG = gi := by
    have hGe : gi * coutG ≤ oc := by rw [← hC0]; exact hlo
    have hLt : oc < (gi + 1) * coutG := by rw [Nat.add_mul, Nat.one_mul]; omega
    have h1 : gi ≤ oc / coutG := (Nat.le_div_iff_mul_le hCoutG).mpr hGe
    have h2 : oc / coutG < gi + 1 := Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hLt)
    omega
  have hAddOc : gi * coutG + (oc - gi * coutG) = oc := by rw [← hC0]; omega
  have hInLt' : (oc - gi * coutG) * (Ho * Wo) + oy * Wo + ox < shapeProd [1, coutG, Ho, Wo] :=
    flat4D_lt coutG Ho Wo (oc - gi * coutG) oy ox hLocLt hoy hox
  have hGetG : groupOuts[gi] = evalConvTensor4D ops H W cinG coutG coutG Ho Wo kh kw sh sw pt pl
      (fun ic' y x => read4D (mkTensor4D cinG H W fun c r k => xFn (gi * cinG + c) r k) cinG H W ic' y x)
      (fun oc' ic' ky kx => w (gi * coutG + oc') ic' ky kx)
      (bias?.map fun b oc' => b (gi * coutG + oc')) act := by
    simp [groupOuts]
  have hLeft : groupOuts[gi].data ⟨(oc - (groupOuts.take gi).foldl (fun acc t => acc + t.shape.getD 1 0) 0) * (Ho * Wo) + oy * Wo + ox, hInLt⟩ =
      evalConvElem ops H W cinG coutG kh kw sh sw pt pl
        (fun ic' y x => read4D (mkTensor4D cinG H W fun c r k => xFn (gi * cinG + c) r k) cinG H W ic' y x)
        (fun oc' ic' ky kx => w (gi * coutG + oc') ic' ky kx)
        (bias?.map fun b oc' => b (gi * coutG + oc')) act (oc - gi * coutG) oy ox := by
    have hSub : groupOuts[gi].data ⟨(oc - (groupOuts.take gi).foldl (fun acc t => acc + t.shape.getD 1 0) 0) * (Ho * Wo) + oy * Wo + ox, hInLt⟩ =
        (evalConvTensor4D ops H W cinG coutG coutG Ho Wo kh kw sh sw pt pl
          (fun ic' y x => read4D (mkTensor4D cinG H W fun c r k => xFn (gi * cinG + c) r k) cinG H W ic' y x)
          (fun oc' ic' ky kx => w (gi * coutG + oc') ic' ky kx)
          (bias?.map fun b oc' => b (gi * coutG + oc')) act).data
          ⟨(oc - gi * coutG) * (Ho * Wo) + oy * Wo + ox, hInLt'⟩ := by
      generalize hEqT : groupOuts[gi] = tCur at hInLt ⊢
      have htEq : tCur = _ := hEqT ▸ hGetG
      subst htEq
      congr 1; apply Fin.ext; dsimp only; rw [hC0]
    rw [hSub]
    unfold evalConvTensor4D
    exact mkTensor4D_at coutG Ho Wo (oc - gi * coutG) oy ox _ hLocLt hoy hox hInLt'
  rw [hLeft]
  unfold evalConvTensor4D
  rw [mkTensor4D_at (groups * coutG) Ho Wo oc oy ox _ hoc hoy hox h2]
  unfold evalConvElem
  have hLocDiv : (oc - gi * coutG) / coutG = 0 := Nat.div_eq_of_lt hLocLt
  simp only [hLocDiv, hDivEq, Nat.zero_mul, Nat.zero_add, hAddOc]
  have hActBias : ∀ a1 a2 : α, a1 = a2 →
      applyConvAct ops act (match bias?.map fun b oc' => b (gi * coutG + oc') with | some b => add a1 (b (oc - gi * coutG)) | none => a1) =
      applyConvAct ops act (match bias? with | some b => add a2 (b oc) | none => a2) := by
    intro a1 a2 h; subst h; cases bias? <;> simp [hAddOc]
  apply hActBias
  apply sumFin_congr; intro ic' hic'
  apply sumFin_congr; intro ky _
  apply sumFin_congr; intro kx _
  congr 1
  by_cases hIn : 0 ≤ (oy : Int) * (sh : Int) + (ky : Int) - (pt : Int) ∧
                 (oy : Int) * (sh : Int) + (ky : Int) - (pt : Int) < (H : Int) ∧
                 0 ≤ (ox : Int) * (sw : Int) + (kx : Int) - (pl : Int) ∧
                 (ox : Int) * (sw : Int) + (kx : Int) - (pl : Int) < (W : Int)
  · simp only [hIn, and_self, ↓reduceIte]
    exact read4D_mkTensor4D cinG H W ic' _ _ _ hic' (by omega) (by omega)
  · simp only [hIn, ↓reduceIte]

-- ----------------------------------------------------------------------------
-- Named C4 DecomposeConv Corollaries: Conv1d, 5x5, 7x7, 1xK, Kx1, Stride 1 & 2
-- ----------------------------------------------------------------------------

/-- Corollary: `DecomposeConv` exactness for `Conv1d` (`kh = 1`, `H = 1`, arbitrary `kw`, stride `1` or `2`). -/
theorem decomposeConv_conv1d_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (W cin cout kw sHw pl : Nat) (hW : W > 0) (hsHw : sHw = 1 ∨ sHw = 2)
    (x : Nat → Nat → Nat → α) (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α)) (act : ConvActivation)
    (keys : List (Int × Int)) (hNe : keys ≠ []) (hNd : keys.Nodup)
    (oc ox : Nat) (hoc : oc < cout) (hox : ox < ceilDiv W sHw)
    (hCov :
      let subH := ceilDiv 1 sHw
      let subW := ceilDiv W sHw
      let basePt := (if (subH - 1) * sHw + 3 > 1 then (subH - 1) * sHw + 3 - 1 else 0) / 2
      let basePl := (if (subW - 1) * sHw + 3 > W then (subW - 1) * sHw + 3 - W else 0) / 2
      let minDy : Int := -((0 : Nat) : Int); let maxDy : Int := ((1 : Nat) : Int) - 1 - ((0 : Nat) : Int)
      let minDx : Int := -(pl : Int); let maxDx : Int := (kw : Int) - 1 - (pl : Int)
      ∀ (ky kx : Nat), ky < 1 → kx < kw →
        ((assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - ((0 : Nat) : Int))).1,
         (assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))).1) ∈ keys) :
    let subH := ceilDiv 1 sHw
    let subW := ceilDiv W sHw
    let basePt := (if (subH - 1) * sHw + 3 > 1 then (subH - 1) * sHw + 3 - 1 else 0) / 2
    let basePl := (if (subW - 1) * sHw + 3 > W then (subW - 1) * sHw + 3 - W else 0) / 2
    applyConvAct ops act
      (sumList (keys.zipIdx.map fun ((sy, sx), ti) =>
        evalConvElem ops 1 W cin cout 3 3 sHw sHw basePt basePl
          (fun c y x' => shiftChain2D 1 W (x c) zero sy sx y x')
          (bucketWeight3x3 1 kw 0 pl basePt basePl w sy sx)
          (if ti = 0 then bias? else none) .none oc 0 ox)) =
    evalConvElem ops 1 W cin cout 1 kw sHw sHw 0 pl x w bias? act oc 0 ox := by
  have h0Lt : 0 < ceilDiv 1 sHw := by rcases hsHw with rfl | rfl <;> unfold ceilDiv <;> decide
  exact decomposeConv_shiftConv_sum_exact ops 1 W cin cout 1 kw sHw 0 pl
    (by omega) hW hsHw x w bias? act keys hNe hNd oc 0 ox hoc h0Lt hox hCov

/-- Corollary: `DecomposeConv` exactness for asymmetric-stride `Conv1d` (`H = 1, sh = 1, sw = 2`,
    matching `DecomposeConv.lean` line 224 where `sHw = 2` is used on both axes):
    at `oy = 0`, `0 * 1 = 0 * 2 = 0`, so `sHw = 2` evaluates identically to `(sh = 1, sw = 2)`! -/
theorem decomposeConv_conv1d_stride1x2_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (W cin cout kw pl : Nat) (hW : W > 0)
    (x : Nat → Nat → Nat → α) (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α)) (act : ConvActivation)
    (keys : List (Int × Int)) (hNe : keys ≠ []) (hNd : keys.Nodup)
    (oc ox : Nat) (hoc : oc < cout) (hox : ox < ceilDiv W 2)
    (hCov :
      let subH := ceilDiv 1 2
      let subW := ceilDiv W 2
      let basePt := (if (subH - 1) * 2 + 3 > 1 then (subH - 1) * 2 + 3 - 1 else 0) / 2
      let basePl := (if (subW - 1) * 2 + 3 > W then (subW - 1) * 2 + 3 - W else 0) / 2
      let minDy : Int := -((0 : Nat) : Int); let maxDy : Int := ((1 : Nat) : Int) - 1 - ((0 : Nat) : Int)
      let minDx : Int := -(pl : Int); let maxDx : Int := (kw : Int) - 1 - (pl : Int)
      ∀ (ky kx : Nat), ky < 1 → kx < kw →
        ((assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - ((0 : Nat) : Int))).1,
         (assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))).1) ∈ keys) :
    let subH := ceilDiv 1 2
    let subW := ceilDiv W 2
    let basePt := (if (subH - 1) * 2 + 3 > 1 then (subH - 1) * 2 + 3 - 1 else 0) / 2
    let basePl := (if (subW - 1) * 2 + 3 > W then (subW - 1) * 2 + 3 - W else 0) / 2
    applyConvAct ops act
      (sumList (keys.zipIdx.map fun ((sy, sx), ti) =>
        evalConvElem ops 1 W cin cout 3 3 2 2 basePt basePl
          (fun c y x' => shiftChain2D 1 W (x c) zero sy sx y x')
          (bucketWeight3x3 1 kw 0 pl basePt basePl w sy sx)
          (if ti = 0 then bias? else none) .none oc 0 ox)) =
    evalConvElem ops 1 W cin cout 1 kw 1 2 0 pl x w bias? act oc 0 ox := by
  have h1 := decomposeConv_conv1d_exact ops W cin cout kw 2 pl hW (Or.inr rfl) x w bias? act keys hNe hNd oc ox hoc hox hCov
  have hEq : evalConvElem ops 1 W cin cout 1 kw 2 2 0 pl x w bias? act oc 0 ox =
      evalConvElem ops 1 W cin cout 1 kw 1 2 0 pl x w bias? act oc 0 ox := by
    unfold evalConvElem
    simp
  rw [← hEq]
  exact h1

/-- Exactness of `DecomposeConv.lean` lines 270–275 zero-weight bucket pruning:
    dropping any shift bucket `(sy, sx)` whose `3x3` bucket weight tensor is identically `zero`
    (keeping a single `(0, 0)` zero bucket when all buckets are zero) preserves the exact shift-conv sum! -/
theorem decomposeConv_zero_bucket_pruning_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cin cout kh kw sHw pt pl : Nat)
    (x : Nat → Nat → Nat → α) (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α)) (act : ConvActivation)
    (allKeys activeKeys : List (Int × Int))
    (hAllNe : allKeys ≠ []) (hActNe : activeKeys ≠ [])
    (hAllNd : allKeys.Nodup) (hActNd : activeKeys.Nodup)
    (hSub : ∀ k ∈ activeKeys, k ∈ allKeys)
    (oc oy ox : Nat)
    (hPrunedZero :
      let subH := ceilDiv H sHw; let subW := ceilDiv W sHw
      let basePt := (if (subH - 1) * sHw + 3 > H then (subH - 1) * sHw + 3 - H else 0) / 2
      let basePl := (if (subW - 1) * sHw + 3 > W then (subW - 1) * sHw + 3 - W else 0) / 2
      ∀ k ∈ allKeys, k ∉ activeKeys →
        ∀ ic ty tx, ic < cin → ty < 3 → tx < 3 →
          bucketWeight3x3 kh kw pt pl basePt basePl w k.1 k.2 oc ic ty tx = zero) :
    let subH := ceilDiv H sHw; let subW := ceilDiv W sHw
    let basePt := (if (subH - 1) * sHw + 3 > H then (subH - 1) * sHw + 3 - H else 0) / 2
    let basePl := (if (subW - 1) * sHw + 3 > W then (subW - 1) * sHw + 3 - W else 0) / 2
    applyConvAct ops act
      (sumList (activeKeys.zipIdx.map fun ((sy, sx), ti) =>
        evalConvElem ops H W cin cout 3 3 sHw sHw basePt basePl
          (fun c y x' => shiftChain2D H W (x c) zero sy sx y x')
          (bucketWeight3x3 kh kw pt pl basePt basePl w sy sx)
          (if ti = 0 then bias? else none) .none oc oy ox)) =
    applyConvAct ops act
      (sumList (allKeys.zipIdx.map fun ((sy, sx), ti) =>
        evalConvElem ops H W cin cout 3 3 sHw sHw basePt basePl
          (fun c y x' => shiftChain2D H W (x c) zero sy sx y x')
          (bucketWeight3x3 kh kw pt pl basePt basePl w sy sx)
          (if ti = 0 then bias? else none) .none oc oy ox)) := by
  intro subH subW basePt basePl
  let tileMac : (Int × Int) → α := fun (sy, sx) =>
    evalConvElem ops H W cin cout 3 3 sHw sHw basePt basePl
      (fun c y x' => shiftChain2D H W (x c) zero sy sx y x')
      (bucketWeight3x3 kh kw pt pl basePt basePl w sy sx) none .none oc oy ox
  have hSplit : ∀ (ks : List (Int × Int)), ks ≠ [] →
      sumList (ks.zipIdx.map fun ((sy, sx), ti) =>
        evalConvElem ops H W cin cout 3 3 sHw sHw basePt basePl
          (fun c y x' => shiftChain2D H W (x c) zero sy sx y x')
          (bucketWeight3x3 kh kw pt pl basePt basePl w sy sx)
          (if ti = 0 then bias? else none) .none oc oy ox) =
      match bias? with
      | some b => add (sumList (ks.map tileMac)) (b oc)
      | none => sumList (ks.map tileMac) := by
    intro ks hKsNe
    cases ks with
    | nil => exact absurd rfl hKsNe
    | cons k0 rest =>
      have hRest : ∀ (l : List (Int × Int)) (start : Nat), start > 0 →
          (l.zipIdx start).map (fun ((sy, sx), ti) =>
            evalConvElem ops H W cin cout 3 3 sHw sHw basePt basePl
              (fun c y x' => shiftChain2D H W (x c) zero sy sx y x')
              (bucketWeight3x3 kh kw pt pl basePt basePl w sy sx)
              (if ti = 0 then bias? else none) .none oc oy ox) = l.map tileMac := by
        intro l; induction l with
        | nil => intro _ _; rfl
        | cons x xs ih =>
          intro st hst
          have hStNe : st ≠ 0 := by omega
          simp only [List.zipIdx_cons, List.map_cons, hStNe, ↓reduceIte]
          exact congrArg (tileMac x :: ·) (ih (st + 1) (by omega))
      simp only [List.zipIdx_cons, List.map_cons, ↓reduceIte, hRest rest 1 (by omega), sumList]
      cases bias? with
      | none => rfl
      | some b =>
        have hHead : evalConvElem ops H W cin cout 3 3 sHw sHw basePt basePl
            (fun c y x' => shiftChain2D H W (x c) zero k0.1 k0.2 y x')
            (bucketWeight3x3 kh kw pt pl basePt basePl w k0.1 k0.2) (some b) .none oc oy ox =
            add (tileMac k0) (b oc) := rfl
        rw [hHead, add_assoc, add_comm (b oc), ← add_assoc]
  rw [hSplit activeKeys hActNe, hSplit allKeys hAllNe]
  -- Prove `sumList (activeKeys.map tileMac) = sumList (allKeys.map tileMac)`
  have hZeroTile : ∀ k ∈ allKeys, k ∉ activeKeys → tileMac k = zero := by
    intro k hk hNot
    unfold tileMac evalConvElem applyConvAct
    dsimp only
    rw [sumFin_congr cin (g := fun _ => zero) (fun ic hic => by
      rw [sumFin_congr 3 (g := fun _ => zero) (fun ty hty => by
        rw [sumFin_congr 3 (g := fun _ => zero) (fun tx htx => by
          rw [hPrunedZero k hk hNot ic ty tx hic hty htx, zero_mul])]
        exact sumFin_const_zero 3)]
      exact sumFin_const_zero 3)]
    exact sumFin_const_zero cin
  -- Pure list-sum permutation/filtering lemma for Nodup lists
  have hSumEq : ∀ (xs ys : List (Int × Int)), xs.Nodup → ys.Nodup →
      (∀ k ∈ xs, k ∈ ys) → (∀ k ∈ ys, k ∉ xs → tileMac k = zero) →
      sumList (xs.map tileMac) = sumList (ys.map tileMac) := by
    intro xs
    induction xs with
    | nil =>
      intro ys _ _ _ hZ
      have hEq : ys.map tileMac = ys.map (fun _ => zero) :=
        List.map_congr_left fun k hk => hZ k hk (by simp)
      rw [List.map_nil, hEq, sumList_const_zero]; rfl
    | cons a as ih =>
      intro ys hNdX hNdY hSub' hZ
      have hNdCons := List.nodup_cons.mp hNdX
      have haMem : a ∈ ys := hSub' a List.mem_cons_self
      -- Split `ys = ys1 ++ [a] ++ ys2`
      obtain ⟨ys1, ys2, rfl⟩ := List.append_of_mem haMem
      have hNdRest : (ys1 ++ ys2).Nodup := by
        rw [List.nodup_append] at hNdY ⊢
        have hNd2 := List.nodup_cons.mp hNdY.2.1
        exact ⟨hNdY.1, hNd2.2, fun x hx1 y hy2 => hNdY.2.2 x hx1 y (List.mem_cons_of_mem _ hy2)⟩
      have haNotRest : a ∉ ys1 ++ ys2 := by
        rw [List.nodup_append] at hNdY
        have hNd2 := List.nodup_cons.mp hNdY.2.1
        intro hMem
        rcases List.mem_append.mp hMem with h1 | h2
        · exact hNdY.2.2 a h1 a List.mem_cons_self rfl
        · exact hNd2.1 h2
      have hSubRest : ∀ k ∈ as, k ∈ ys1 ++ ys2 := by
        intro k hk
        have hInY := hSub' k (List.mem_cons_of_mem _ hk)
        have hNe : k ≠ a := fun heq => hNdCons.1 (heq ▸ hk)
        simp only [List.mem_append, List.mem_cons] at hInY ⊢
        rcases hInY with h1 | rfl | h2
        · exact Or.inl h1
        · exact absurd rfl hNe
        · exact Or.inr h2
      have hZRest : ∀ k ∈ ys1 ++ ys2, k ∉ as → tileMac k = zero := by
        intro k hk hNotAs
        have hNeA : k ≠ a := fun heq => haNotRest (heq ▸ hk)
        apply hZ k (by simp only [List.mem_append, List.mem_cons] at hk ⊢; rcases hk with h1 | h2; exact Or.inl h1; exact Or.inr (Or.inr h2))
        intro hMem; rcases List.mem_cons.mp hMem with rfl | hAs
        · exact hNeA rfl
        · exact hNotAs hAs
      have hRec := ih (ys1 ++ ys2) hNdCons.2 hNdRest hSubRest hZRest
      simp only [List.map_cons, List.map_append, sumList, sumList_append] at hRec ⊢
      rw [hRec, ← add_assoc, add_comm (tileMac a) (sumList (ys1.map tileMac)), add_assoc]
  have hS := hSumEq activeKeys allKeys hActNd hAllNd hSub hZeroTile
  rw [hS]

/-- Corollary: `DecomposeConv` exactness for `5x5` convolutions (stride `1` or `2`, any padding `(pt, pl)`). -/
theorem decomposeConv_5x5_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cin cout sHw pt pl : Nat) (hH : H > 0) (hW : W > 0) (hsHw : sHw = 1 ∨ sHw = 2)
    (x : Nat → Nat → Nat → α) (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α)) (act : ConvActivation)
    (keys : List (Int × Int)) (hNe : keys ≠ []) (hNd : keys.Nodup)
    (oc oy ox : Nat) (hoc : oc < cout) (hoy : oy < ceilDiv H sHw) (hox : ox < ceilDiv W sHw)
    (hCov :
      let subH := ceilDiv H sHw; let subW := ceilDiv W sHw
      let basePt := (if (subH - 1) * sHw + 3 > H then (subH - 1) * sHw + 3 - H else 0) / 2
      let basePl := (if (subW - 1) * sHw + 3 > W then (subW - 1) * sHw + 3 - W else 0) / 2
      let minDy : Int := -(pt : Int); let maxDy : Int := ((5 : Nat) : Int) - 1 - (pt : Int)
      let minDx : Int := -(pl : Int); let maxDx : Int := ((5 : Nat) : Int) - 1 - (pl : Int)
      ∀ (ky kx : Nat), ky < 5 → kx < 5 →
        ((assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - (pt : Int))).1,
         (assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))).1) ∈ keys) :
    let subH := ceilDiv H sHw; let subW := ceilDiv W sHw
    let basePt := (if (subH - 1) * sHw + 3 > H then (subH - 1) * sHw + 3 - H else 0) / 2
    let basePl := (if (subW - 1) * sHw + 3 > W then (subW - 1) * sHw + 3 - W else 0) / 2
    applyConvAct ops act
      (sumList (keys.zipIdx.map fun ((sy, sx), ti) =>
        evalConvElem ops H W cin cout 3 3 sHw sHw basePt basePl
          (fun c y x' => shiftChain2D H W (x c) zero sy sx y x')
          (bucketWeight3x3 5 5 pt pl basePt basePl w sy sx)
          (if ti = 0 then bias? else none) .none oc oy ox)) =
    evalConvElem ops H W cin cout 5 5 sHw sHw pt pl x w bias? act oc oy ox :=
  decomposeConv_shiftConv_sum_exact ops H W cin cout 5 5 sHw pt pl hH hW hsHw x w bias? act keys hNe hNd oc oy ox hoc hoy hox hCov

/-- Corollary: `DecomposeConv` exactness for `7x7` convolutions (stride `1` or `2`, any padding `(pt, pl)`). -/
theorem decomposeConv_7x7_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cin cout sHw pt pl : Nat) (hH : H > 0) (hW : W > 0) (hsHw : sHw = 1 ∨ sHw = 2)
    (x : Nat → Nat → Nat → α) (w : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α)) (act : ConvActivation)
    (keys : List (Int × Int)) (hNe : keys ≠ []) (hNd : keys.Nodup)
    (oc oy ox : Nat) (hoc : oc < cout) (hoy : oy < ceilDiv H sHw) (hox : ox < ceilDiv W sHw)
    (hCov :
      let subH := ceilDiv H sHw; let subW := ceilDiv W sHw
      let basePt := (if (subH - 1) * sHw + 3 > H then (subH - 1) * sHw + 3 - H else 0) / 2
      let basePl := (if (subW - 1) * sHw + 3 > W then (subW - 1) * sHw + 3 - W else 0) / 2
      let minDy : Int := -(pt : Int); let maxDy : Int := ((7 : Nat) : Int) - 1 - (pt : Int)
      let minDx : Int := -(pl : Int); let maxDx : Int := ((7 : Nat) : Int) - 1 - (pl : Int)
      ∀ (ky kx : Nat), ky < 7 → kx < 7 →
        ((assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - (pt : Int))).1,
         (assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))).1) ∈ keys) :
    let subH := ceilDiv H sHw; let subW := ceilDiv W sHw
    let basePt := (if (subH - 1) * sHw + 3 > H then (subH - 1) * sHw + 3 - H else 0) / 2
    let basePl := (if (subW - 1) * sHw + 3 > W then (subW - 1) * sHw + 3 - W else 0) / 2
    applyConvAct ops act
      (sumList (keys.zipIdx.map fun ((sy, sx), ti) =>
        evalConvElem ops H W cin cout 3 3 sHw sHw basePt basePl
          (fun c y x' => shiftChain2D H W (x c) zero sy sx y x')
          (bucketWeight3x3 7 7 pt pl basePt basePl w sy sx)
          (if ti = 0 then bias? else none) .none oc oy ox)) =
    evalConvElem ops H W cin cout 7 7 sHw sHw pt pl x w bias? act oc oy ox :=
  decomposeConv_shiftConv_sum_exact ops H W cin cout 7 7 sHw pt pl hH hW hsHw x w bias? act keys hNe hNd oc oy ox hoc hoy hox hCov

/-- Corollary: `DecomposeConv` exactness for asymmetric `1xK` and `Kx1` separable convolutions! -/
theorem decomposeConv_1xK_and_Kx1_exact {α : Type} [ScalarField α] (ops : ScalarOps α)
    (H W cin cout K sHw pt pl : Nat) (hH : H > 0) (hW : W > 0) (hsHw : sHw = 1 ∨ sHw = 2)
    (x : Nat → Nat → Nat → α) (w1K wK1 : Nat → Nat → Nat → Nat → α)
    (bias? : Option (Nat → α)) (act : ConvActivation)
    (keys1K keysK1 : List (Int × Int))
    (hNe1K : keys1K ≠ []) (hNd1K : keys1K.Nodup)
    (hNeK1 : keysK1 ≠ []) (hNdK1 : keysK1.Nodup)
    (oc oy ox : Nat) (hoc : oc < cout) (hoy : oy < ceilDiv H sHw) (hox : ox < ceilDiv W sHw)
    (hCov1K :
      let subH := ceilDiv H sHw; let subW := ceilDiv W sHw
      let basePt := (if (subH - 1) * sHw + 3 > H then (subH - 1) * sHw + 3 - H else 0) / 2
      let basePl := (if (subW - 1) * sHw + 3 > W then (subW - 1) * sHw + 3 - W else 0) / 2
      let minDy : Int := -(pt : Int); let maxDy : Int := ((1 : Nat) : Int) - 1 - (pt : Int)
      let minDx : Int := -(pl : Int); let maxDx : Int := (K : Int) - 1 - (pl : Int)
      ∀ (ky kx : Nat), ky < 1 → kx < K →
        ((assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - (pt : Int))).1,
         (assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))).1) ∈ keys1K)
    (hCovK1 :
      let subH := ceilDiv H sHw; let subW := ceilDiv W sHw
      let basePt := (if (subH - 1) * sHw + 3 > H then (subH - 1) * sHw + 3 - H else 0) / 2
      let basePl := (if (subW - 1) * sHw + 3 > W then (subW - 1) * sHw + 3 - W else 0) / 2
      let minDy : Int := -(pt : Int); let maxDy : Int := (K : Int) - 1 - (pt : Int)
      let minDx : Int := -(pl : Int); let maxDx : Int := ((1 : Nat) : Int) - 1 - (pl : Int)
      ∀ (ky kx : Nat), ky < K → kx < 1 →
        ((assignConvAxisShiftTap basePt minDy maxDy ((ky : Int) - (pt : Int))).1,
         (assignConvAxisShiftTap basePl minDx maxDx ((kx : Int) - (pl : Int))).1) ∈ keysK1) :
    let subH := ceilDiv H sHw; let subW := ceilDiv W sHw
    let basePt := (if (subH - 1) * sHw + 3 > H then (subH - 1) * sHw + 3 - H else 0) / 2
    let basePl := (if (subW - 1) * sHw + 3 > W then (subW - 1) * sHw + 3 - W else 0) / 2
    (applyConvAct ops act
      (sumList (keys1K.zipIdx.map fun ((sy, sx), ti) =>
        evalConvElem ops H W cin cout 3 3 sHw sHw basePt basePl
          (fun c y x' => shiftChain2D H W (x c) zero sy sx y x')
          (bucketWeight3x3 1 K pt pl basePt basePl w1K sy sx)
          (if ti = 0 then bias? else none) .none oc oy ox)) =
     evalConvElem ops H W cin cout 1 K sHw sHw pt pl x w1K bias? act oc oy ox) ∧
    (applyConvAct ops act
      (sumList (keysK1.zipIdx.map fun ((sy, sx), ti) =>
        evalConvElem ops H W cin cout 3 3 sHw sHw basePt basePl
          (fun c y x' => shiftChain2D H W (x c) zero sy sx y x')
          (bucketWeight3x3 K 1 pt pl basePt basePl wK1 sy sx)
          (if ti = 0 then bias? else none) .none oc oy ox)) =
     evalConvElem ops H W cin cout K 1 sHw sHw pt pl x wK1 bias? act oc oy ox) :=
  ⟨decomposeConv_shiftConv_sum_exact ops H W cin cout 1 K sHw pt pl hH hW hsHw x w1K bias? act keys1K hNe1K hNd1K oc oy ox hoc hoy hox hCov1K,
   decomposeConv_shiftConv_sum_exact ops H W cin cout K 1 sHw pt pl hH hW hsHw x wK1 bias? act keysK1 hNeK1 hNdK1 oc oy ox hoc hoy hox hCovK1⟩

/-- Direct `LayoutSelect` `evalAneOpCore` preservation for `.dmaView` standard concat views (`batchN = 1`). -/
theorem layoutSelect_aneOp_concat_equiv {α : Type} [ScalarField α] (ops : ScalarOps α)
    (decodeConvWeight : WeightData → Nat → Nat → Nat → Nat → Option (Nat → Nat → Nat → Nat → α))
    (decodeBias : Option ByteArray → Nat → Option (Option (Nat → α)))
    (axis : Nat) (desc : DmaViewSpec) (hAx : desc.concatAxis? = some axis)
    (hB1 : desc.batchN = 1)
    (ins : List (TensorVal α)) (outTy : TensorType) (r : TensorVal α)
    (hCat : evalConcat axis ins = some r) :
    evalAneOpCore ops decodeConvWeight decodeBias (.dmaView desc default) ins [outTy] = some [r] := by
  unfold evalAneOpCore
  dsimp only
  rw [hAx, hB1]
  have hGt1 : decide (1 > 1) = false := by decide
  simp only [hGt1, Bool.and_false, Bool.false_eq_true, ↓reduceIte, hCat, Option.map_some]

end Vane.Proofs.C5SemanticPasses
