import Vane.IR.Passes.DecomposeConv

namespace Vane.Proofs.C4Conv

open Vane.IR

-- ============================================================================
-- C4 Formal Soundness Proofs: Exact Sign-Monotone Shift-Conv Decomposition
-- ============================================================================

/-- Pure 1D zero-padded spatial shift: `shift1D n x zero s i` evaluates `x[i + s]`
    when `0 ≤ i + s < n`, and returns `zero` outside `[0, n)`. -/
def shift1D {α : Type} (n : Nat) (x : Nat → α) (zero : α) (s : Int) (i : Nat) : α :=
  if 0 ≤ (i : Int) + s ∧ (i : Int) + s < (n : Int) then
    x ((i : Int) + s).toNat
  else
    zero

/-- Zero shift is the identity on all in-bounds positions `i < n`. -/
theorem shift1D_zero {α : Type} (n : Nat) (x : Nat → α) (zero : α) (i : Nat) (hi : i < n) :
    shift1D n x zero 0 i = x i := by
  unfold shift1D
  have h1 : 0 ≤ (i : Int) + 0 ∧ (i : Int) + 0 < (n : Int) := by omega
  have hidx : ((i : Int) + 0).toNat = i := by omega
  simp only [h1, and_self, hidx, ↓reduceIte]

/-- Sign-preserving composition (non-negative shifts `a ≥ 0, b ≥ 0`):
    composing two non-negative zero-padded shifts never drops an in-bounds pixel. -/
theorem shift1D_compose_nonneg {α : Type} (n : Nat) (x : Nat → α) (zero : α)
    (a b : Int) (ha : 0 ≤ a) (hb : 0 ≤ b) (i : Nat) :
    shift1D n (shift1D n x zero a) zero b i = shift1D n x zero (a + b) i := by
  unfold shift1D
  by_cases hsum : 0 ≤ (i : Int) + (a + b) ∧ (i : Int) + (a + b) < (n : Int)
  · have hb_ok : 0 ≤ (i : Int) + b ∧ (i : Int) + b < (n : Int) := by omega
    have hab_ok : 0 ≤ (((i : Int) + b).toNat : Int) + a ∧ (((i : Int) + b).toNat : Int) + a < (n : Int) := by omega
    have hidx : ((((i : Int) + b).toNat : Int) + a).toNat = ((i : Int) + (a + b)).toNat := by omega
    simp only [hsum, hb_ok, hab_ok, and_self, hidx, ↓reduceIte]
  · by_cases hb_ok : 0 ≤ (i : Int) + b ∧ (i : Int) + b < (n : Int)
    · have hab_no : ¬ (0 ≤ (((i : Int) + b).toNat : Int) + a ∧ (((i : Int) + b).toNat : Int) + a < (n : Int)) := by omega
      simp only [hsum, hb_ok, hab_no, and_self, ↓reduceIte]
    · simp only [hsum, hb_ok, ↓reduceIte]

/-- Sign-preserving composition (non-positive shifts `a ≤ 0, b ≤ 0` on in-bounds `i < n`):
    composing two non-positive zero-padded shifts never drops an in-bounds pixel. -/
theorem shift1D_compose_nonpos {α : Type} (n : Nat) (x : Nat → α) (zero : α)
    (a b : Int) (ha : a ≤ 0) (hb : b ≤ 0) (i : Nat) (hi : i < n) :
    shift1D n (shift1D n x zero a) zero b i = shift1D n x zero (a + b) i := by
  unfold shift1D
  by_cases hsum : 0 ≤ (i : Int) + (a + b) ∧ (i : Int) + (a + b) < (n : Int)
  · have hb_ok : 0 ≤ (i : Int) + b ∧ (i : Int) + b < (n : Int) := by omega
    have hab_ok : 0 ≤ (((i : Int) + b).toNat : Int) + a ∧ (((i : Int) + b).toNat : Int) + a < (n : Int) := by omega
    have hidx : ((((i : Int) + b).toNat : Int) + a).toNat = ((i : Int) + (a + b)).toNat := by omega
    simp only [hsum, hb_ok, hab_ok, and_self, hidx, ↓reduceIte]
  · by_cases hb_ok : 0 ≤ (i : Int) + b ∧ (i : Int) + b < (n : Int)
    · have hab_no : ¬ (0 ≤ (((i : Int) + b).toNat : Int) + a ∧ (((i : Int) + b).toNat : Int) + a < (n : Int)) := by omega
      simp only [hsum, hb_ok, hab_no, and_self, ↓reduceIte]
    · simp only [hsum, hb_ok, ↓reduceIte]

/-- Every tap index produced by `assignConvAxisShiftTap` lies strictly in `{0, 1, 2}`. -/
theorem assignConvAxisShiftTap_tap_lt3 (basePad : Nat) (minD maxD d : Int) :
    (assignConvAxisShiftTap basePad minD maxD d).2 < 3 := by
  unfold assignConvAxisShiftTap
  split
  · split
    · rename_i h1
      simp only [Bool.and_eq_true, decide_eq_true_eq] at h1
      dsimp only; omega
    · split
      · split
        · decide
        · dsimp only; split <;> decide
      · split
        · decide
        · dsimp only; split <;> decide
  · split
    · rename_i h1
      simp only [Bool.and_eq_true, decide_eq_true_eq] at h1
      dsimp only; omega
    · split
      · dsimp only; omega
      · dsimp only; omega

/-- Exact relative-offset conservation: for every hardware `basePad ∈ {0, 1}` and tap offset `d`,
    `shift + (tap - basePad) = d`. -/
theorem assignConvAxisShiftTap_offset_exact (basePad : Nat) (hPad : basePad = 0 ∨ basePad = 1)
    (minD maxD d : Int) :
    (assignConvAxisShiftTap basePad minD maxD d).1 +
      (((assignConvAxisShiftTap basePad minD maxD d).2 : Int) - (basePad : Int)) = d := by
  rcases hPad with rfl | rfl
  · unfold assignConvAxisShiftTap
    have h01 : ((0 : Nat) == 1) = false := rfl
    simp only [h01, Bool.false_eq_true, ↓reduceIte]
    split
    · rename_i h1
      simp only [Bool.and_eq_true, decide_eq_true_eq] at h1
      dsimp only; omega
    · rename_i h1
      simp only [Bool.and_eq_true, decide_eq_true_eq, not_and] at h1
      split
      · dsimp only; omega
      · dsimp only; omega
  · unfold assignConvAxisShiftTap
    simp only [beq_self_eq_true, ↓reduceIte]
    split
    · rename_i h1
      simp only [Bool.and_eq_true, decide_eq_true_eq] at h1
      dsimp only; omega
    · rename_i h1
      simp only [Bool.and_eq_true, decide_eq_true_eq, not_and] at h1
      split
      · rename_i hlt
        split
        · rename_i hm
          simp only [Bool.and_eq_true, beq_iff_eq] at hm
          dsimp only; omega
        · dsimp only
          split
          · rename_i hev; simp only [beq_iff_eq] at hev; omega
          · rename_i hodd; simp only [beq_iff_eq] at hodd; omega
      · rename_i hge
        split
        · rename_i hm
          simp only [Bool.and_eq_true, beq_iff_eq] at hm
          dsimp only; omega
        · dsimp only
          split
          · rename_i hev; simp only [beq_iff_eq] at hev; omega
          · rename_i hodd; simp only [beq_iff_eq] at hodd; omega

/-- Sign-monotone negative invariant: whenever `assignConvAxisShiftTap` assigns a negative shift
    `shift < 0`, the relative 3x3 tap offset `tap - basePad` is non-positive (`≤ 0`). -/
theorem assignConvAxisShiftTap_monotone_neg (basePad : Nat) (hPad : basePad = 0 ∨ basePad = 1)
    (minD maxD d : Int)
    (hNeg : (assignConvAxisShiftTap basePad minD maxD d).1 < 0) :
    ((assignConvAxisShiftTap basePad minD maxD d).2 : Int) - (basePad : Int) ≤ 0 := by
  rcases hPad with rfl | rfl
  · unfold assignConvAxisShiftTap at hNeg ⊢
    have h01 : ((0 : Nat) == 1) = false := rfl
    simp only [h01, Bool.false_eq_true, ↓reduceIte] at hNeg ⊢
    by_cases h1 : (0 ≤ d && d ≤ 2) = true
    · simp only [h1, ↓reduceIte] at hNeg; omega
    · simp only [h1, Bool.false_eq_true, ↓reduceIte] at hNeg ⊢
      by_cases h2 : d < 0
      · simp only [h2, ↓reduceIte]; omega
      · simp only [h2, ↓reduceIte] at hNeg; omega
  · unfold assignConvAxisShiftTap at hNeg ⊢
    simp only [beq_self_eq_true, ↓reduceIte] at hNeg ⊢
    by_cases h1 : (-1 ≤ d && d ≤ 1) = true
    · simp only [h1, ↓reduceIte] at hNeg; omega
    · simp only [h1, Bool.false_eq_true, ↓reduceIte] at hNeg ⊢
      by_cases h2 : d < -1
      · simp only [h2, ↓reduceIte]
        by_cases h3 : (minD == -2 && d == -2) = true
        · simp only [h3, ↓reduceIte]; omega
        · simp only [h3, Bool.false_eq_true, ↓reduceIte]
          split <;> omega
      · simp only [h2, ↓reduceIte] at hNeg
        by_cases h3 : (maxD == 2 && d == 2) = true
        · simp only [h3, ↓reduceIte] at hNeg; omega
        · simp only [h3, Bool.false_eq_true, ↓reduceIte] at hNeg; omega

/-- Sign-monotone positive invariant: whenever `assignConvAxisShiftTap` assigns a positive shift
    `shift > 0`, the relative 3x3 tap offset `tap - basePad` is non-negative (`≥ 0`). -/
theorem assignConvAxisShiftTap_monotone_pos (basePad : Nat) (hPad : basePad = 0 ∨ basePad = 1)
    (minD maxD d : Int)
    (hPos : (assignConvAxisShiftTap basePad minD maxD d).1 > 0) :
    0 ≤ ((assignConvAxisShiftTap basePad minD maxD d).2 : Int) - (basePad : Int) := by
  rcases hPad with rfl | rfl
  · unfold assignConvAxisShiftTap at hPos ⊢
    have h01 : ((0 : Nat) == 1) = false := rfl
    simp only [h01, Bool.false_eq_true, ↓reduceIte] at hPos ⊢
    by_cases h1 : (0 ≤ d && d ≤ 2) = true
    · simp only [h1, ↓reduceIte] at hPos; omega
    · simp only [h1, Bool.false_eq_true, ↓reduceIte] at hPos ⊢
      by_cases h2 : d < 0
      · simp only [h2, ↓reduceIte] at hPos; omega
      · simp only [h2, ↓reduceIte]; omega
  · unfold assignConvAxisShiftTap at hPos ⊢
    simp only [beq_self_eq_true, ↓reduceIte] at hPos ⊢
    by_cases h1 : (-1 ≤ d && d ≤ 1) = true
    · simp only [h1, ↓reduceIte] at hPos; omega
    · simp only [h1, Bool.false_eq_true, ↓reduceIte] at hPos ⊢
      by_cases h2 : d < -1
      · simp only [h2, ↓reduceIte] at hPos
        by_cases h3 : (minD == -2 && d == -2) = true
        · simp only [h3, ↓reduceIte] at hPos; omega
        · simp only [h3, Bool.false_eq_true, ↓reduceIte] at hPos; omega
      · simp only [h2, ↓reduceIte]
        by_cases h3 : (maxD == 2 && d == 2) = true
        · simp only [h3, ↓reduceIte]; omega
        · simp only [h3, Bool.false_eq_true, ↓reduceIte]
          split <;> omega

/-- End-to-end exact boundary soundness of `assignConvAxisShiftTap`:
    for every `basePad ∈ {0, 1}`, tap offset `d`, and in-bounds index `i < n`,
    applying the 3x3 hardware tap `tap - basePad` to the shifted signal `shift1D n x zero shift`
    produces the exact zero-padded signal `shift1D n x zero d i`. -/
theorem assignConvAxisShiftTap_shift_sound {α : Type} (n : Nat) (x : Nat → α) (zero : α)
    (basePad : Nat) (hPad : basePad = 0 ∨ basePad = 1) (minD maxD d : Int)
    (i : Nat) (hi : i < n) :
    let st := assignConvAxisShiftTap basePad minD maxD d
    shift1D n (shift1D n x zero st.1) zero ((st.2 : Int) - (basePad : Int)) i =
      shift1D n x zero d i := by
  intro st
  have hExact : st.1 + ((st.2 : Int) - (basePad : Int)) = d :=
    assignConvAxisShiftTap_offset_exact basePad hPad minD maxD d
  by_cases hZero : st.1 = 0
  · have hEq : (st.2 : Int) - (basePad : Int) = d := by omega
    rw [hZero, hEq]
    unfold shift1D
    by_cases hd : 0 ≤ (i : Int) + d ∧ (i : Int) + d < (n : Int)
    · have h0 : 0 ≤ ((((i : Int) + d).toNat : Int) + 0) ∧ ((((i : Int) + d).toNat : Int) + 0) < (n : Int) := by omega
      have hidx : ((((i : Int) + d).toNat : Int) + 0).toNat = ((i : Int) + d).toNat := by omega
      simp only [hd, h0, and_self, hidx, ↓reduceIte]
    · simp only [hd, ↓reduceIte]
  · by_cases hNeg : st.1 < 0
    · have hTapLe := assignConvAxisShiftTap_monotone_neg basePad hPad minD maxD d hNeg
      rw [shift1D_compose_nonpos n x zero st.1 ((st.2 : Int) - (basePad : Int)) (by omega) hTapLe i hi, hExact]
    · have hPos : st.1 > 0 := by omega
      have hTapGe := assignConvAxisShiftTap_monotone_pos basePad hPad minD maxD d hPos
      rw [shift1D_compose_nonneg n x zero st.1 ((st.2 : Int) - (basePad : Int)) (by omega) hTapGe i, hExact]

end Vane.Proofs.C4Conv
