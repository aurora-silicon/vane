import Vane.Binary
import Vane.TaskStream
import Vane.Register
import Vane.Reach
import Vane.Alloc
import Vane.Tiling
import Vane.Pe
import Vane.Lut
import Vane.Pool
import Vane.Verify
import Vane.View

namespace Vane.Proofs

open Vane.Binary
open Vane.TaskStream
open Vane.Register
open Vane.Reach
open Vane.Alloc
open Vane.Tiling
open Vane.Pe
open Vane.Lut
open Vane.Pool
open Vane.Verify
open Vane.View

-- ============================================================================
-- 1. Alignment soundness (alignUp_ge, alignUp_dvd)
-- ============================================================================

/-- `alignUp n m` never shrinks `n`. -/
theorem alignUp_ge (n m : Nat) : n ≤ alignUp n m := by
  unfold alignUp
  split
  · omega
  · dsimp only
    split <;> omega

/-- For any positive alignment `m > 0`, `alignUp n m` is an exact multiple of `m`. -/
theorem alignUp_dvd (n m : Nat) (hm : m > 0) : alignUp n m % m = 0 := by
  unfold alignUp
  have hm0 : (m == 0) = false := by simp; omega
  rw [hm0]
  simp only [Bool.false_eq_true, ↓reduceIte]
  split
  · rename_i h
    simpa using h
  · rename_i h
    have hlt : n % m < m := Nat.mod_lt n hm
    have hdiv : m * (n / m) + n % m = n := Nat.div_add_mod n m
    have hmul : (n / m + 1) * m = m * (n / m) + m := by
      rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm]
    have h1 : n + (m - n % m) = (n / m + 1) * m := by omega
    rw [h1, Nat.mul_mod_left]

-- ============================================================================
-- 2. WordLayout well-formedness & kernel-clean encodeWord/decodeFieldChecked proofs
-- ============================================================================

theorem taskHeader0Layout_valid : (validateWordLayout taskHeader0Layout).isOk = true := by decide
theorem taskHeader8Layout_valid : (validateWordLayout taskHeader8Layout).isOk = true := by decide
theorem convCfgLayout_valid : (validateWordLayout convCfgLayout).isOk = true := by decide
theorem dwGroupsLayout_valid : (validateWordLayout dwGroupsLayout).isOk = true := by decide
theorem neCommonLayout_valid : (validateWordLayout neCommonLayout).isOk = true := by decide
theorem neTileLayout_valid : (validateWordLayout neTileLayout).isOk = true := by decide
theorem src1CfgLayout_valid : (validateWordLayout src1CfgLayout).isOk = true := by decide
theorem resCfgLayout_valid : (validateWordLayout resCfgLayout).isOk = true := by decide
theorem neCfgLayout_h16_valid : (validateWordLayout (neCfgLayout .h16)).isOk = true := by decide
theorem neCfgLayout_h17_valid : (validateWordLayout (neCfgLayout .h17)).isOk = true := by decide
theorem neOpLayout_valid : (validateWordLayout neOpLayout).isOk = true := by decide
theorem neActOpLayout_valid : (validateWordLayout neActOpLayout).isOk = true := by decide
theorem neActShiftEnLayout_valid : (validateWordLayout neActShiftEnLayout).isOk = true := by decide
theorem peCommonLayout_valid : (validateWordLayout peCommonLayout).isOk = true := by decide
theorem peTileLayout_valid : (validateWordLayout peTileLayout).isOk = true := by decide
theorem peBcastLayout_valid : (validateWordLayout peBcastLayout).isOk = true := by decide
theorem kernelStageLayout_valid : (validateWordLayout kernelStageLayout).isOk = true := by decide
theorem dim16Layout_valid (name : String) : (validateWordLayout (dim16Layout name)).isOk = true := by
  simp [validateWordLayout, dim16Layout, validateWordLayoutFields, BitField.mask, BitField.maskNat]; rfl
theorem u32Layout_valid (name : String) : (validateWordLayout (u32Layout name)).isOk = true := by
  simp [validateWordLayout, u32Layout, validateWordLayoutFields, BitField.mask, BitField.maskNat]; rfl
theorem scalarBiasLayout_valid : (validateWordLayout scalarBiasLayout).isOk = true := by decide
theorem neScaleLayout_valid : (validateWordLayout neScaleLayout).isOk = true := by decide
theorem src1DepLayout_valid : (validateWordLayout src1DepLayout).isOk = true := by decide
theorem dmaFmtLayout_valid (name : String) : (validateWordLayout (dmaFmtLayout name)).isOk = true := by
  simp [validateWordLayout, dmaFmtLayout, validateWordLayoutFields, BitField.mask, BitField.maskNat]
  have h : ((4278190080 : UInt32) &&& (16777215 : UInt32)) = 0 := by decide
  simp [h]; rfl
theorem resDepLayout_valid : (validateWordLayout resDepLayout).isOk = true := by decide
theorem chunkSlotLayout_valid : (validateWordLayout chunkSlotLayout).isOk = true := by decide
theorem out2Layout_valid : (validateWordLayout out2Layout).isOk = true := by decide
theorem taskHeader2Layout_valid : (validateWordLayout taskHeader2Layout).isOk = true := by decide
theorem taskHeader8MultiLayout_valid : (validateWordLayout taskHeader8MultiLayout).isOk = true := by decide
theorem resCfgMultiLayout_valid : (validateWordLayout resCfgMultiLayout).isOk = true := by decide
theorem kernelStageMultiLayout_valid : (validateWordLayout kernelStageMultiLayout).isOk = true := by decide
theorem portCfgMultiLayout_valid (name : String) : (validateWordLayout (portCfgMultiLayout name)).isOk = true := by
  simp [validateWordLayout, portCfgMultiLayout, validateWordLayoutFields, BitField.mask, BitField.maskNat]
  rfl
theorem srcDepMultiLayout_valid (name : String) : (validateWordLayout (srcDepMultiLayout name)).isOk = true := by
  simp [validateWordLayout, srcDepMultiLayout, validateWordLayoutFields, BitField.mask, BitField.maskNat]
  rfl

-- Kernel-clean BitVec / UInt32 bitwise algebra helpers (zero bv_decide, zero custom axioms)

private theorem u32_eq_of_bit_eq {a b : UInt32}
    (h : ∀ (i : Nat) (hi : i < 32), a.toBitVec[i] = b.toBitVec[i]) : a = b := by
  apply UInt32.toBitVec_inj.mp
  exact BitVec.eq_of_getElem_eq h

private theorem u32_bit_and (a b : UInt32) (i : Nat) (hi : i < 32) :
    (a &&& b).toBitVec[i] = (a.toBitVec[i] && b.toBitVec[i]) := by
  simp [UInt32.toBitVec_and, BitVec.getElem_and]

private theorem u32_bit_or (a b : UInt32) (i : Nat) (hi : i < 32) :
    (a ||| b).toBitVec[i] = (a.toBitVec[i] || b.toBitVec[i]) := by
  simp [UInt32.toBitVec_or, BitVec.getElem_or]

private theorem u32_bit_not (a : UInt32) (i : Nat) (hi : i < 32) :
    (~~~a).toBitVec[i] = !a.toBitVec[i] := by
  simp [UInt32.toBitVec_not, BitVec.getElem_not]

private theorem u32_bit_zero (i : Nat) (hi : i < 32) :
    (0 : UInt32).toBitVec[i] = false := by
  simp

private theorem u32_and_eq_zero_of_subset {x m k : UInt32}
    (hsub : x &&& ~~~m = 0) (hdisj : m &&& k = 0) : x &&& k = 0 := by
  apply u32_eq_of_bit_eq; intro i hi
  have h1 := congrArg (fun u : UInt32 => u.toBitVec[i]) hsub
  have h2 := congrArg (fun u : UInt32 => u.toBitVec[i]) hdisj
  simp only [u32_bit_and _ _ i hi, u32_bit_not _ i hi, u32_bit_zero i hi] at h1 h2 ⊢
  cases hx : x.toBitVec[i] <;> cases hm : m.toBitVec[i] <;> cases hk : k.toBitVec[i] <;>
    simp_all

private theorem u32_or_subset {x y m1 m2 : UInt32}
    (hx : x &&& ~~~m1 = 0) (hy : y &&& ~~~m2 = 0) :
    (x ||| y) &&& ~~~(m1 ||| m2) = 0 := by
  apply u32_eq_of_bit_eq; intro i hi
  have h1 := congrArg (fun u : UInt32 => u.toBitVec[i]) hx
  have h2 := congrArg (fun u : UInt32 => u.toBitVec[i]) hy
  simp only [u32_bit_and _ _ i hi, u32_bit_or _ _ i hi, u32_bit_not _ i hi, u32_bit_zero i hi] at h1 h2 ⊢
  cases bx : x.toBitVec[i] <;> cases by' : y.toBitVec[i] <;>
    cases bm1 : m1.toBitVec[i] <;> cases bm2 : m2.toBitVec[i] <;> simp_all

private theorem u32_subset_trans {a b c : UInt32}
    (hab : a &&& ~~~b = 0) (hbc : b &&& ~~~c = 0) : a &&& ~~~c = 0 := by
  apply u32_eq_of_bit_eq; intro i hi
  have h1 := congrArg (fun u : UInt32 => u.toBitVec[i]) hab
  have h2 := congrArg (fun u : UInt32 => u.toBitVec[i]) hbc
  simp only [u32_bit_and _ _ i hi, u32_bit_not _ i hi, u32_bit_zero i hi] at h1 h2 ⊢
  cases ba : a.toBitVec[i] <;> cases bb : b.toBitVec[i] <;> cases bc : c.toBitVec[i] <;> simp_all

private theorem u32_subset_or_left (a b : UInt32) : a &&& ~~~(a ||| b) = 0 := by
  apply u32_eq_of_bit_eq; intro i hi
  simp only [u32_bit_and _ _ i hi, u32_bit_or _ _ i hi, u32_bit_not _ i hi, u32_bit_zero i hi]
  cases a.toBitVec[i] <;> cases b.toBitVec[i] <;> rfl

private theorem u32_subset_or_right (a b : UInt32) : b &&& ~~~(a ||| b) = 0 := by
  apply u32_eq_of_bit_eq; intro i hi
  simp only [u32_bit_and _ _ i hi, u32_bit_or _ _ i hi, u32_bit_not _ i hi, u32_bit_zero i hi]
  cases a.toBitVec[i] <;> cases b.toBitVec[i] <;> rfl

private theorem u32_and_comm_zero {a b : UInt32} (h : a &&& b = 0) : b &&& a = 0 := by
  apply u32_eq_of_bit_eq; intro i hi
  have h1 := congrArg (fun u : UInt32 => u.toBitVec[i]) h
  simp only [u32_bit_and _ _ i hi, u32_bit_zero i hi] at h1 ⊢
  cases ba : a.toBitVec[i] <;> cases bb : b.toBitVec[i] <;> simp_all

private theorem u32_and_or_eq_of_disj {w x k v : UInt32}
    (hw : w &&& k = v) (hx : x &&& k = 0) : (w ||| x) &&& k = v := by
  apply u32_eq_of_bit_eq; intro i hi
  have h1 := congrArg (fun u : UInt32 => u.toBitVec[i]) hw
  have h2 := congrArg (fun u : UInt32 => u.toBitVec[i]) hx
  simp only [u32_bit_and _ _ i hi, u32_bit_or _ _ i hi, u32_bit_zero i hi] at h1 h2 ⊢
  cases bw : w.toBitVec[i] <;> cases bx : x.toBitVec[i] <;> cases bk : k.toBitVec[i] <;> simp_all

private theorem u32_and_self_of_subset {a m : UInt32} (h : a &&& ~~~m = 0) : a &&& m = a := by
  apply u32_eq_of_bit_eq; intro i hi
  have h1 := congrArg (fun u : UInt32 => u.toBitVec[i]) h
  simp only [u32_bit_and _ _ i hi, u32_bit_not _ i hi, u32_bit_zero i hi] at h1 ⊢
  cases ba : a.toBitVec[i] <;> cases bm : m.toBitVec[i] <;> simp_all

/-- Helper: `testBit` of `(1 <<< w) - 1` is `decide (i < w)`. -/
private theorem testBit_one_shift_sub_one (w i : Nat) :
    (((1 : Nat) <<< w) - 1).testBit i = decide (i < w) := by
  rw [Nat.shiftLeft_eq, Nat.one_mul, Nat.testBit_two_pow_sub_one]

/-- Helper: any `val < 2 ^ w` has zero bits at positions `i ≥ w`. -/
private theorem testBit_eq_false_of_lt_two_pow {x w i : Nat} (hx : x < 2 ^ w) (hi : w ≤ i) :
    x.testBit i = false := by
  apply Nat.testBit_lt_two_pow
  calc
    x < 2 ^ w := hx
    _ ≤ 2 ^ i := Nat.pow_le_pow_right (by omega) hi

private theorem bitfield_maskNat_lt (f : BitField) (hhi : f.lo + f.width ≤ 32) :
    f.maskNat < 2 ^ 32 := by
  unfold BitField.maskNat
  rw [Nat.shiftLeft_eq, Nat.shiftLeft_eq, Nat.one_mul]
  have h1 : (2 ^ f.width - 1) * 2 ^ f.lo < 2 ^ f.width * 2 ^ f.lo := by
    have hp : 0 < 2 ^ f.lo := Nat.two_pow_pos _
    have hp2 : 0 < 2 ^ f.width := Nat.two_pow_pos _
    exact Nat.mul_lt_mul_of_pos_right (by omega) hp
  rw [← Nat.pow_add] at h1
  have h2 : 2 ^ (f.width + f.lo) ≤ 2 ^ 32 := Nat.pow_le_pow_right (by omega) (by omega)
  omega

private theorem bitfield_mask_toNat (f : BitField) (hhi : f.lo + f.width ≤ 32) :
    f.mask.toNat = f.maskNat := by
  unfold BitField.mask
  change f.maskNat % 2 ^ 32 = f.maskNat
  exact Nat.mod_eq_of_lt (bitfield_maskNat_lt f hhi)

/-- Every bit set in `val <<< f.lo` (for `val < 2^f.width`, `f.lo + f.width ≤ 32`) lies inside `f.mask`. -/
private theorem val_shiftLeft_subset_mask (val : UInt32) (f : BitField)
    (hhi : f.lo + f.width ≤ 32) (hval : val.toNat < 2 ^ f.width) :
    (val <<< f.lo.toUInt32) &&& ~~~f.mask = 0 := by
  apply u32_eq_of_bit_eq; intro i hi
  simp only [u32_bit_and _ _ i hi, u32_bit_not _ i hi, u32_bit_zero i hi]
  have htoNatLo : f.lo.toUInt32.toNat = f.lo := by
    change f.lo % 2 ^ 32 = f.lo
    exact Nat.mod_eq_of_lt (by omega)
  have hbitL : (val <<< f.lo.toUInt32).toBitVec[i] = (val <<< f.lo.toUInt32).toNat.testBit i := by
    rw [BitVec.getElem_eq_testBit_toNat]; rfl
  have hbitM : f.mask.toBitVec[i] = f.mask.toNat.testBit i := by
    rw [BitVec.getElem_eq_testBit_toNat]; rfl
  rw [hbitL, hbitM, UInt32.toNat_shiftLeft, htoNatLo, bitfield_mask_toNat f hhi]
  by_cases h32 : f.lo = 32
  · have hw0 : f.width = 0 := by omega
    rw [hw0] at hval
    have hv0 : val.toNat = 0 := by omega
    simp [hv0]
  · have hloLt : f.lo < 32 := by omega
    rw [Nat.mod_eq_of_lt hloLt]
    have hmod2 : (val.toNat <<< f.lo) % 4294967296 = (val.toNat <<< f.lo) % 2 ^ 32 := rfl
    rw [hmod2, Nat.testBit_mod_two_pow]
    simp only [hi, decide_true, Bool.true_and, Nat.testBit_shiftLeft]
    unfold BitField.maskNat
    rw [Nat.testBit_shiftLeft]
    by_cases hge : f.lo ≤ i
    · simp only [hge, decide_true, Bool.true_and, testBit_one_shift_sub_one]
      by_cases hlt : i - f.lo < f.width
      · simp [hlt]
      · have hvFalse : val.toNat.testBit (i - f.lo) = false :=
          testBit_eq_false_of_lt_two_pow hval (by omega)
        simp [hvFalse]
    · simp [hge]

/-- General bitfield round-trip theorem: OR-ing a width-bounded value `val < 2^f.width`
    shifted by `f.lo` into any disjoint `rest` (`rest &&& f.mask = 0`) decodes back to `val`. -/
theorem decodeField_or_shift (rest val : UInt32) (f : BitField)
    (hhi : f.lo + f.width ≤ 32)
    (hval : val.toNat < 2 ^ f.width)
    (hdisj : rest &&& f.mask = 0) :
    decodeField (rest ||| (val <<< f.lo.toUInt32)) f = val := by
  apply UInt32.toNat_inj.mp
  apply Nat.eq_of_testBit_eq
  intro i
  have hmaskLt : ((1 : Nat) <<< f.width) - 1 < 2 ^ 32 := by
    rw [Nat.shiftLeft_eq, Nat.one_mul]
    have hpow : 2 ^ f.width ≤ 2 ^ 32 := Nat.pow_le_pow_right (by omega) (by omega)
    omega
  have htoNatMask : ((((1 : Nat) <<< f.width) - 1).toUInt32).toNat = ((1 : Nat) <<< f.width) - 1 := by
    change (((1 : Nat) <<< f.width) - 1) % 2 ^ 32 = _
    exact Nat.mod_eq_of_lt hmaskLt
  have htoNatLo : f.lo.toUInt32.toNat = f.lo := by
    change f.lo % 2 ^ 32 = f.lo
    exact Nat.mod_eq_of_lt (by omega)
  have htoNatFullMask : f.mask.toNat = f.maskNat := bitfield_mask_toNat f hhi
  unfold decodeField
  rw [UInt32.toNat_and, UInt32.toNat_shiftRight, UInt32.toNat_or, UInt32.toNat_shiftLeft,
      htoNatMask, htoNatLo, Nat.testBit_and, Nat.testBit_shiftRight, Nat.testBit_or,
      testBit_one_shift_sub_one]
  by_cases hi : i < f.width
  · simp only [hi, decide_true, Bool.and_true]
    have hloLt : f.lo < 32 := by omega
    rw [Nat.mod_eq_of_lt hloLt]
    have hloAdd : f.lo + i < 32 := by omega
    have hrestZero : rest.toNat.testBit (f.lo + i) = false := by
      have h0 : (rest &&& f.mask).toNat.testBit (f.lo + i) = false := by
        rw [hdisj]; simp
      rw [UInt32.toNat_and, htoNatFullMask, Nat.testBit_and] at h0
      have hmaskBit : f.maskNat.testBit (f.lo + i) = true := by
        unfold BitField.maskNat
        rw [Nat.testBit_shiftLeft]
        have hge : f.lo ≤ f.lo + i := by omega
        have hsub : f.lo + i - f.lo = i := by omega
        simp [hge, hsub, testBit_one_shift_sub_one, hi]
      rw [hmaskBit, Bool.and_true] at h0
      exact h0
    rw [hrestZero, Bool.false_or]
    have hmod2 : (val.toNat <<< f.lo) % 4294967296 = (val.toNat <<< f.lo) % 2 ^ 32 := rfl
    rw [hmod2, Nat.testBit_mod_two_pow]
    simp only [hloAdd, decide_true, Bool.true_and, Nat.testBit_shiftLeft]
    have hge : f.lo ≤ f.lo + i := by omega
    have hsub : f.lo + i - f.lo = i := by omega
    simp [hge, hsub]
  · simp only [hi, decide_false, Bool.and_false]
    symm
    exact testBit_eq_false_of_lt_two_pow hval (by omega)

/-- OR-ing bits disjoint from `f.mask` leaves `decodeField · f` unchanged. -/
theorem decodeField_or_disjoint (w x : UInt32) (f : BitField)
    (hhi : f.lo + f.width ≤ 32) (hdisj : x &&& f.mask = 0) :
    decodeField (w ||| x) f = decodeField w f := by
  apply UInt32.toNat_inj.mp
  apply Nat.eq_of_testBit_eq
  intro i
  have hmaskLt : ((1 : Nat) <<< f.width) - 1 < 2 ^ 32 := by
    rw [Nat.shiftLeft_eq, Nat.one_mul]
    have hpow : 2 ^ f.width ≤ 2 ^ 32 := Nat.pow_le_pow_right (by omega) (by omega)
    omega
  have htoNatMask : ((((1 : Nat) <<< f.width) - 1).toUInt32).toNat = ((1 : Nat) <<< f.width) - 1 := by
    change (((1 : Nat) <<< f.width) - 1) % 2 ^ 32 = _
    exact Nat.mod_eq_of_lt hmaskLt
  have htoNatLo : f.lo.toUInt32.toNat = f.lo := by
    change f.lo % 2 ^ 32 = f.lo
    exact Nat.mod_eq_of_lt (by omega)
  have htoNatFullMask : f.mask.toNat = f.maskNat := bitfield_mask_toNat f hhi
  unfold decodeField
  rw [UInt32.toNat_and, UInt32.toNat_and, UInt32.toNat_shiftRight, UInt32.toNat_shiftRight,
      UInt32.toNat_or, htoNatMask, htoNatLo, Nat.testBit_and, Nat.testBit_and,
      Nat.testBit_shiftRight, Nat.testBit_shiftRight, Nat.testBit_or, testBit_one_shift_sub_one]
  by_cases hi : i < f.width
  · simp only [hi, decide_true, Bool.and_true]
    have hloLt : f.lo < 32 := by omega
    rw [Nat.mod_eq_of_lt hloLt]
    have hxZero : x.toNat.testBit (f.lo + i) = false := by
      have h0 : (x &&& f.mask).toNat.testBit (f.lo + i) = false := by
        rw [hdisj]; simp
      rw [UInt32.toNat_and, htoNatFullMask, Nat.testBit_and] at h0
      have hmaskBit : f.maskNat.testBit (f.lo + i) = true := by
        unfold BitField.maskNat
        rw [Nat.testBit_shiftLeft]
        have hge : f.lo ≤ f.lo + i := by omega
        have hsub : f.lo + i - f.lo = i := by omega
        simp [hge, hsub, testBit_one_shift_sub_one, hi]
      rw [hmaskBit, Bool.and_true] at h0
      exact h0
    rw [hxZero, Bool.or_false]
  · simp only [hi, decide_false, Bool.and_false]

private theorem except_bind_ok {α β : Type} {x : Except String α} {f : α → Except String β} {v : β}
    (h : (x >>= f) = .ok v) : ∃ a, x = .ok a ∧ f a = .ok v := by
  cases x with
  | error e => simp [bind, Except.bind] at h
  | ok a => exact ⟨a, rfl, h⟩

private theorem except_if_err_ok {α : Type} {c : Prop} [Decidable c] {e : String}
    {k : Except String α} {v : α}
    (h : (if c then Except.error e else k) = .ok v) : ¬ c ∧ k = .ok v := by
  by_cases hc : c
  · simp [hc] at h
  · simp [hc] at h
    exact ⟨hc, h⟩

/-- Soundness of `validateWordLayoutFields`: extracts mask monotonicity, field bounds,
    disjointness from `usedMask` and `fixedMask`, and pairwise disjointness & name uniqueness. -/
private theorem validateWordLayoutFields_sound
    (layoutName : String) (fixedMask : UInt32)
    (fs : List BitField) (usedMask finalMask : UInt32) (seen : List String)
    (h : validateWordLayoutFields layoutName fixedMask fs usedMask seen = .ok finalMask) :
    usedMask &&& ~~~finalMask = 0 ∧
    (∀ f ∈ fs,
      f.width ≥ 1 ∧ f.lo + f.width ≤ 32 ∧
      f.mask &&& fixedMask = 0 ∧
      f.mask &&& usedMask = 0 ∧
      f.mask &&& ~~~finalMask = 0 ∧
      f.name ∉ seen) ∧
    fs.Pairwise (fun a b => a.name ≠ b.name ∧ a.mask &&& b.mask = 0) := by
  induction fs generalizing usedMask seen with
  | nil =>
    simp only [validateWordLayoutFields, Except.ok.injEq] at h
    subst h
    refine ⟨?_, ?_, List.Pairwise.nil⟩
    · apply u32_eq_of_bit_eq; intro i hi
      simp only [u32_bit_and _ _ i hi, u32_bit_not _ i hi, u32_bit_zero i hi]
      cases usedMask.toBitVec[i] <;> rfl
    · intro f hf; cases hf
  | cons f rest ih =>
    unfold validateWordLayoutFields at h
    obtain ⟨hw0, h⟩ := except_if_err_ok h
    obtain ⟨hhi, h⟩ := except_if_err_ok h
    obtain ⟨hseen, h⟩ := except_if_err_ok h
    dsimp only at h
    obtain ⟨hfix, h⟩ := except_if_err_ok h
    obtain ⟨hused, hrest⟩ := except_if_err_ok h
    have hfix0 : f.mask &&& fixedMask = 0 := by simpa using hfix
    have hused0 : f.mask &&& usedMask = 0 := by simpa using hused
    have hnotSeen : f.name ∉ seen := by
      intro hmem
      apply hseen
      rw [List.any_eq_true]
      exact ⟨f.name, hmem, by simp⟩
    obtain ⟨hmono, hall, hpw⟩ := ih (usedMask ||| f.mask) (f.name :: seen) hrest
    have husedSub : usedMask &&& ~~~finalMask = 0 :=
      u32_subset_trans (u32_subset_or_left usedMask f.mask) hmono
    have hfSub : f.mask &&& ~~~finalMask = 0 :=
      u32_subset_trans (u32_subset_or_right usedMask f.mask) hmono
    refine ⟨husedSub, ?_, List.Pairwise.cons ?_ hpw⟩
    · intro g hg
      cases hg with
      | head =>
        have hwNe : f.width ≠ 0 := by simpa using hw0
        exact ⟨by omega, by omega, hfix0, hused0, hfSub, hnotSeen⟩
      | tail _ hmem =>
        obtain ⟨gw, ghi, gfix, gused, gsub, gseen⟩ := hall g hmem
        have gused' : g.mask &&& usedMask = 0 :=
          u32_and_comm_zero (u32_and_eq_zero_of_subset (u32_subset_or_left usedMask f.mask) (u32_and_comm_zero gused))
        have gnotSeen : g.name ∉ seen := fun hm => gseen (List.mem_cons_of_mem _ hm)
        exact ⟨gw, ghi, gfix, gused', gsub, gnotSeen⟩
    · intro g hmem
      obtain ⟨_, _, _, gused, _, gseen⟩ := hall g hmem
      have hne : f.name ≠ g.name := fun heq => gseen (by rw [heq]; exact List.mem_cons_self)
      have hdisj : f.mask &&& g.mask = 0 :=
        u32_and_eq_zero_of_subset (u32_subset_or_right usedMask f.mask) (u32_and_comm_zero gused)
      exact ⟨hne, hdisj⟩

/-- Soundness of `encodeWordFields`: preserves fixed bits, stays inside `finalMask`, leaves
    disjoint fields untouched, and decodes every encoded field back to its supplied value. -/
private theorem encodeWordFields_sound
    (layoutName : String) (fixedMask fixedValue : UInt32)
    (vals : List (String × UInt32)) (fs : List BitField)
    (usedMask finalMask word out : UInt32) (seen : List String)
    (hvalid : validateWordLayoutFields layoutName fixedMask fs usedMask seen = .ok finalMask)
    (hsub : word &&& ~~~usedMask = 0)
    (hfix : word &&& fixedMask = fixedValue)
    (henc : encodeWordFields layoutName vals fs word = .ok out) :
    out &&& ~~~finalMask = 0 ∧
    out &&& fixedMask = fixedValue ∧
    (∀ q : BitField, q.lo + q.width ≤ 32 → (∀ f ∈ fs, f.mask &&& q.mask = 0) →
      decodeField out q = decodeField word q) ∧
    (∀ f ∈ fs, ∃ val : UInt32,
      vals.find? (fun p => decide (p.1 = f.name)) = some (f.name, val) ∧
      decodeField out f = val) := by
  induction fs generalizing usedMask seen word with
  | nil =>
    simp only [validateWordLayoutFields, Except.ok.injEq] at hvalid
    simp only [encodeWordFields, Except.ok.injEq] at henc
    subst hvalid; subst henc
    refine ⟨hsub, hfix, ?_, ?_⟩
    · intro _ _ _; rfl
    · intro f hf; cases hf
  | cons f rest ih =>
    have hvalidCopy := hvalid
    obtain ⟨_, hallAll, hpwAll⟩ := validateWordLayoutFields_sound layoutName fixedMask (f :: rest) usedMask finalMask seen hvalidCopy
    unfold validateWordLayoutFields at hvalid
    obtain ⟨hw0, hvalid⟩ := except_if_err_ok hvalid
    obtain ⟨hhi, hvalid⟩ := except_if_err_ok hvalid
    obtain ⟨_, hvalid⟩ := except_if_err_ok hvalid
    dsimp only at hvalid
    obtain ⟨hfixDisj, hvalid⟩ := except_if_err_ok hvalid
    obtain ⟨husedDisj, hvalidRest⟩ := except_if_err_ok hvalid
    have hfixDisj0 : f.mask &&& fixedMask = 0 := by simpa using hfixDisj
    have husedDisj0 : f.mask &&& usedMask = 0 := by simpa using husedDisj
    have hhiLe : f.lo + f.width ≤ 32 := by omega

    unfold encodeWordFields at henc
    cases hfind : vals.find? (fun p => decide (p.1 = f.name)) with
    | none => simp [hfind] at henc
    | some p =>
      rcases p with ⟨nm, val⟩
      have hnmEq : nm = f.name := by
        have hpred := List.find?_some hfind
        simpa using hpred
      subst hnmEq
      simp only [hfind] at henc
      obtain ⟨hvalLt, hencRest⟩ := except_if_err_ok henc
      have hvalPow : val.toNat < 2 ^ f.width := by
        rw [Nat.shiftLeft_eq, Nat.one_mul] at hvalLt
        omega
      have hshiftSub : (val <<< f.lo.toUInt32) &&& ~~~f.mask = 0 :=
        val_shiftLeft_subset_mask val f hhiLe hvalPow
      have hwordSub' : (word ||| (val <<< f.lo.toUInt32)) &&& ~~~(usedMask ||| f.mask) = 0 :=
        u32_or_subset hsub hshiftSub
      have hshiftFix0 : (val <<< f.lo.toUInt32) &&& fixedMask = 0 :=
        u32_and_eq_zero_of_subset hshiftSub hfixDisj0
      have hwordFix' : (word ||| (val <<< f.lo.toUInt32)) &&& fixedMask = fixedValue :=
        u32_and_or_eq_of_disj hfix hshiftFix0
      obtain ⟨houtSub, houtFix, houtDisj, houtFields⟩ :=
        ih (usedMask ||| f.mask) (word ||| (val <<< f.lo.toUInt32)) (f.name :: seen)
          hvalidRest hwordSub' hwordFix' hencRest
      refine ⟨houtSub, houtFix, ?_, ?_⟩
      · intro q hqHi hqDisj
        have hqRest : ∀ g ∈ rest, g.mask &&& q.mask = 0 :=
          fun g hg => hqDisj g (List.mem_cons_of_mem _ hg)
        have hqF : f.mask &&& q.mask = 0 := hqDisj f List.mem_cons_self
        rw [houtDisj q hqHi hqRest]
        have hshiftQ0 : (val <<< f.lo.toUInt32) &&& q.mask = 0 :=
          u32_and_eq_zero_of_subset hshiftSub hqF
        exact decodeField_or_disjoint word (val <<< f.lo.toUInt32) q hqHi hshiftQ0
      · intro g hg
        cases hg with
        | head =>
          refine ⟨val, hfind, ?_⟩
          have hrestDisjF : ∀ r ∈ rest, r.mask &&& f.mask = 0 := by
            intro r hr
            have hfr := (List.pairwise_cons.mp hpwAll).1 r hr
            exact u32_and_comm_zero hfr.2
          rw [houtDisj f hhiLe hrestDisjF]
          have hwordDisjF : word &&& f.mask = 0 :=
            u32_and_eq_zero_of_subset hsub (u32_and_comm_zero husedDisj0)
          exact decodeField_or_shift word val f hhiLe hvalPow hwordDisjF
        | tail _ hr =>
          exact houtFields g hr

private theorem validateEncodeValues_known
    (layoutName : String) (fields : List BitField)
    (vals : List (String × UInt32)) (seen : List String)
    (hv : validateEncodeValues layoutName fields vals seen = .ok ()) :
    ∀ p ∈ vals, ∃ f ∈ fields, f.name = p.1 := by
  induction vals generalizing seen with
  | nil => intro p hp; cases hp
  | cons p rest ih =>
    rcases p with ⟨fname, val⟩
    unfold validateEncodeValues at hv
    obtain ⟨_, hv⟩ := except_if_err_ok hv
    obtain ⟨hknown, hvRest⟩ := except_if_err_ok hv
    intro q hq
    cases hq with
    | head =>
      have hany : fields.any (fun f => decide (f.name = fname)) = true := by
        cases h : fields.any (fun f => decide (f.name = fname))
        · simp [h] at hknown
        · rfl
      rw [List.any_eq_true] at hany
      obtain ⟨f, hf, hEq⟩ := hany
      exact ⟨f, hf, by simpa using hEq⟩
    | tail _ hmem =>
      exact ih (fname :: seen) hvRest q hmem

private theorem findBitField_of_mem_pairwise
    (fs : List BitField) (f : BitField)
    (hpw : fs.Pairwise (fun a b => a.name ≠ b.name ∧ a.mask &&& b.mask = 0))
    (hmem : f ∈ fs) :
    findBitField? fs f.name = some f := by
  induction fs with
  | nil => cases hmem
  | cons hd tl ih =>
    unfold findBitField?
    cases hmem with
    | head =>
      simp [List.find?]
    | tail _ htl =>
      have hne : hd.name ≠ f.name := ((List.pairwise_cons.mp hpw).1 f htl).1
      have hdec : decide (hd.name = f.name) = false := by simp [hne]
      simp only [List.find?, hdec]
      exact ih (List.pairwise_cons.mp hpw).2 htl

/-- General kernel-clean round-trip theorem on production `Vane.Register.encodeWord` and
    `Vane.Register.decodeFieldChecked`: for ANY `WordLayout`, any word successfully encoded by
    `encodeWord` passes `validateWordBits` and decodes every supplied field back identically. -/
theorem encodeWord_decodeFieldChecked
    (layout : WordLayout) (values : Array (String × UInt32)) (w : UInt32)
    (fname : String) (val : UInt32)
    (henc : encodeWord layout values = .ok w)
    (hfind : values.toList.find? (fun p => decide (p.1 = fname)) = some (fname, val)) :
    decodeFieldChecked layout w fname = .ok val := by
  unfold encodeWord at henc
  obtain ⟨finalMask, hvalLayout, henc⟩ := except_bind_ok henc
  obtain ⟨⟨⟩, hvalVals, hencFields⟩ := except_bind_ok henc
  have hvalLayoutCopy := hvalLayout
  unfold validateWordLayout at hvalLayout
  obtain ⟨hfixSub, hvalFields⟩ := except_if_err_ok hvalLayout
  have hfixSub0 : layout.fixedValue &&& ~~~layout.fixedMask = 0 := by simpa using hfixSub
  have hfixSelf : layout.fixedValue &&& layout.fixedMask = layout.fixedValue :=
    u32_and_self_of_subset hfixSub0
  obtain ⟨_, _, hpw⟩ :=
    validateWordLayoutFields_sound layout.name layout.fixedMask layout.fields.toList
      layout.fixedMask finalMask [] hvalFields
  obtain ⟨hwSub, hwFix, _, hwFields⟩ :=
    encodeWordFields_sound layout.name layout.fixedMask layout.fixedValue
      values.toList layout.fields.toList layout.fixedMask finalMask
      layout.fixedValue w [] hvalFields hfixSub0 hfixSelf hencFields
  have hpMem : (fname, val) ∈ values.toList := List.mem_of_find?_eq_some hfind
  obtain ⟨f, hfMem, hfName⟩ :=
    validateEncodeValues_known layout.name layout.fields.toList values.toList [] hvalVals
      (fname, val) hpMem
  dsimp only at hfName
  have hfindField : findBitField? layout.fields.toList fname = some f := by
    rw [← hfName]
    exact findBitField_of_mem_pairwise layout.fields.toList f hpw hfMem
  obtain ⟨val', hfind', hdec⟩ := hwFields f hfMem
  rw [hfName, hfind] at hfind'
  cases hfind'
  unfold decodeFieldChecked validateWordBits
  rw [hvalLayoutCopy]
  simp only [bind, Except.bind, hwFix, hwSub, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte,
    hfindField, hdec]

-- ============================================================================
-- Concrete round-trip theorems on the REAL production WordLayout definitions
-- (zero shadow layouts, zero bv_decide, zero custom axioms)
-- ============================================================================

private theorem u32_not_ge_of_lt {x k : UInt32} (h : x < k) : ¬ x.toNat ≥ k.toNat := by
  rw [UInt32.lt_iff_toNat_lt] at h
  omega

theorem neTile_roundTrip (ocg fat wu : UInt32)
    (hocg : ocg < 8) (hfat : fat < 2) (hwu : wu < 4) :
    ∃ w, encodeWord neTileLayout #[("ocg", ocg), ("fat", fat), ("WU", wu)] = .ok w ∧
         decodeFieldChecked neTileLayout w "ocg" = .ok ocg ∧
         decodeFieldChecked neTileLayout w "fat" = .ok fat ∧
         decodeFieldChecked neTileLayout w "WU" = .ok wu := by
  have h1 : ¬ ocg.toNat ≥ 8 := u32_not_ge_of_lt hocg
  have h2 : ¬ fat.toNat ≥ 2 := u32_not_ge_of_lt hfat
  have h3 : ¬ wu.toNat ≥ 4 := u32_not_ge_of_lt hwu
  have henc : encodeWord neTileLayout #[("ocg", ocg), ("fat", fat), ("WU", wu)] =
      .ok ((((0 : UInt32) ||| (ocg <<< 0)) ||| (fat <<< 3)) ||| (wu <<< 4)) := by
    change (if ocg.toNat ≥ 8 then _ else if fat.toNat ≥ 2 then _ else if wu.toNat ≥ 4 then _ else _) = _
    simp only [h1, h2, h3, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "ocg" ocg henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "fat" fat henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "WU" wu henc rfl⟩

theorem neOp_roundTrip (bias relu lut : UInt32)
    (hbias : bias < 2) (hrelu : relu < 2) (hlut : lut < 2) :
    ∃ w, encodeWord neOpLayout #[("biasInKernel", bias), ("relu", relu), ("lut", lut)] = .ok w ∧
         decodeFieldChecked neOpLayout w "biasInKernel" = .ok bias ∧
         decodeFieldChecked neOpLayout w "relu" = .ok relu ∧
         decodeFieldChecked neOpLayout w "lut" = .ok lut := by
  have h1 : ¬ bias.toNat ≥ 2 := u32_not_ge_of_lt hbias
  have h2 : ¬ relu.toNat ≥ 2 := u32_not_ge_of_lt hrelu
  have h3 : ¬ lut.toNat ≥ 2 := u32_not_ge_of_lt hlut
  have henc : encodeWord neOpLayout #[("biasInKernel", bias), ("relu", relu), ("lut", lut)] =
      .ok ((((0x00103c00 : UInt32) ||| (bias <<< 4)) ||| (relu <<< 16)) ||| (lut <<< 17)) := by
    change (if bias.toNat ≥ 2 then _ else if relu.toNat ≥ 2 then _ else if lut.toNat ≥ 2 then _ else _) = _
    simp only [h1, h2, h3, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "biasInKernel" bias henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "relu" relu henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "lut" lut henc rfl⟩

theorem neActOp_roundTrip (noShift relu lut : UInt32)
    (hns : noShift < 2) (hrelu : relu < 2) (hlut : lut < 2) :
    ∃ w, encodeWord neActOpLayout #[("noShift", noShift), ("relu", relu), ("lut", lut)] = .ok w ∧
         decodeFieldChecked neActOpLayout w "noShift" = .ok noShift ∧
         decodeFieldChecked neActOpLayout w "relu" = .ok relu ∧
         decodeFieldChecked neActOpLayout w "lut" = .ok lut := by
  have h1 : ¬ noShift.toNat ≥ 2 := u32_not_ge_of_lt hns
  have h2 : ¬ relu.toNat ≥ 2 := u32_not_ge_of_lt hrelu
  have h3 : ¬ lut.toNat ≥ 2 := u32_not_ge_of_lt hlut
  have henc : encodeWord neActOpLayout #[("noShift", noShift), ("relu", relu), ("lut", lut)] =
      .ok ((((0x00103c08 : UInt32) ||| (noShift <<< 2)) ||| (relu <<< 16)) ||| (lut <<< 17)) := by
    change (if noShift.toNat ≥ 2 then _ else if relu.toNat ≥ 2 then _ else if lut.toNat ≥ 2 then _ else _) = _
    simp only [h1, h2, h3, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "noShift" noShift henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "relu" relu henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "lut" lut henc rfl⟩

theorem peTile_roundTrip (k m : UInt32) (hk : k < 16) (hm : m < 16) :
    ∃ w, encodeWord peTileLayout #[("k", k), ("m", m)] = .ok w ∧
         decodeFieldChecked peTileLayout w "k" = .ok k ∧
         decodeFieldChecked peTileLayout w "m" = .ok m := by
  have h1 : ¬ k.toNat ≥ 16 := u32_not_ge_of_lt hk
  have h2 : ¬ m.toNat ≥ 16 := u32_not_ge_of_lt hm
  have henc : encodeWord peTileLayout #[("k", k), ("m", m)] =
      .ok (((0 : UInt32) ||| (k <<< 0)) ||| (m <<< 4)) := by
    change (if k.toNat ≥ 16 then _ else if m.toNat ≥ 16 then _ else _) = _
    simp only [h1, h2, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "k" k henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "m" m henc rfl⟩

theorem peBcast_roundTrip (s1w s1h s1c s2w s2h s2c : UInt32)
    (h1w : s1w < 2) (h1h : s1h < 2) (h1c : s1c < 2)
    (h2w : s2w < 2) (h2h : s2h < 2) (h2c : s2c < 2) :
    ∃ w, encodeWord peBcastLayout #[
            ("src1W", s1w), ("src1H", s1h), ("src1C", s1c),
            ("src2W", s2w), ("src2H", s2h), ("src2C", s2c)
          ] = .ok w ∧
         decodeFieldChecked peBcastLayout w "src1W" = .ok s1w ∧
         decodeFieldChecked peBcastLayout w "src1H" = .ok s1h ∧
         decodeFieldChecked peBcastLayout w "src1C" = .ok s1c ∧
         decodeFieldChecked peBcastLayout w "src2W" = .ok s2w ∧
         decodeFieldChecked peBcastLayout w "src2H" = .ok s2h ∧
         decodeFieldChecked peBcastLayout w "src2C" = .ok s2c := by
  have c1 : ¬ s1w.toNat ≥ 2 := u32_not_ge_of_lt h1w
  have c2 : ¬ s1h.toNat ≥ 2 := u32_not_ge_of_lt h1h
  have c3 : ¬ s1c.toNat ≥ 2 := u32_not_ge_of_lt h1c
  have c4 : ¬ s2w.toNat ≥ 2 := u32_not_ge_of_lt h2w
  have c5 : ¬ s2h.toNat ≥ 2 := u32_not_ge_of_lt h2h
  have c6 : ¬ s2c.toNat ≥ 2 := u32_not_ge_of_lt h2c
  have henc : encodeWord peBcastLayout #[
      ("src1W", s1w), ("src1H", s1h), ("src1C", s1c),
      ("src2W", s2w), ("src2H", s2h), ("src2C", s2c)
    ] = .ok (((((((0 : UInt32) ||| (s1w <<< 0)) ||| (s1h <<< 1)) ||| (s1c <<< 3)) |||
      (s2w <<< 4)) ||| (s2h <<< 5)) ||| (s2c <<< 7)) := by
    change (if s1w.toNat ≥ 2 then _ else if s1h.toNat ≥ 2 then _ else
            if s1c.toNat ≥ 2 then _ else if s2w.toNat ≥ 2 then _ else
            if s2h.toNat ≥ 2 then _ else if s2c.toNat ≥ 2 then _ else _) = _
    simp only [c1, c2, c3, c4, c5, c6, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "src1W" s1w henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "src1H" s1h henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "src1C" s1c henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "src2W" s2w henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "src2H" s2h henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "src2C" s2c henc rfl⟩

theorem resCfg_roundTrip (mode direct p : UInt32)
    (hmode : mode < 4) (hdir : direct < 2) (hp : p < 16) :
    ∃ w, encodeWord resCfgLayout #[("mode", mode), ("direct", direct), ("P", p)] = .ok w ∧
         decodeFieldChecked resCfgLayout w "mode" = .ok mode ∧
         decodeFieldChecked resCfgLayout w "direct" = .ok direct ∧
         decodeFieldChecked resCfgLayout w "P" = .ok p := by
  have h1 : ¬ mode.toNat ≥ 4 := u32_not_ge_of_lt hmode
  have h2 : ¬ direct.toNat ≥ 2 := u32_not_ge_of_lt hdir
  have h3 : ¬ p.toNat ≥ 16 := u32_not_ge_of_lt hp
  have henc : encodeWord resCfgLayout #[("mode", mode), ("direct", direct), ("P", p)] =
      .ok ((((0x00500070 : UInt32) ||| (mode <<< 0)) ||| (direct <<< 3)) ||| (p <<< 8)) := by
    change (if mode.toNat ≥ 4 then _ else if direct.toNat ≥ 2 then _ else if p.toNat ≥ 16 then _ else _) = _
    simp only [h1, h2, h3, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "mode" mode henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "direct" direct henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "P" p henc rfl⟩

theorem kernelStage_roundTrip (sparse stage : UInt32)
    (hsp : sparse < 2) (hst : stage < 256) :
    ∃ w, encodeWord kernelStageLayout #[("sparse", sparse), ("stage", stage)] = .ok w ∧
         decodeFieldChecked kernelStageLayout w "sparse" = .ok sparse ∧
         decodeFieldChecked kernelStageLayout w "stage" = .ok stage := by
  have h1 : ¬ sparse.toNat ≥ 2 := u32_not_ge_of_lt hsp
  have h2 : ¬ stage.toNat ≥ 256 := u32_not_ge_of_lt hst
  have henc : encodeWord kernelStageLayout #[("sparse", sparse), ("stage", stage)] =
      .ok (((0x000002c0 : UInt32) ||| (sparse <<< 5)) ||| (stage <<< 16)) := by
    change (if sparse.toNat ≥ 2 then _ else if stage.toNat ≥ 256 then _ else _) = _
    simp only [h1, h2, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "sparse" sparse henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "stage" stage henc rfl⟩

theorem taskHeader0_roundTrip (taskId taskWords : UInt32)
    (hid : taskId < 65536) (hw : taskWords < 2048) :
    ∃ w, encodeWord taskHeader0Layout #[("taskId", taskId), ("taskWords", taskWords)] = .ok w ∧
         decodeFieldChecked taskHeader0Layout w "taskId" = .ok taskId ∧
         decodeFieldChecked taskHeader0Layout w "taskWords" = .ok taskWords := by
  have h1 : ¬ taskId.toNat ≥ 65536 := u32_not_ge_of_lt hid
  have h2 : ¬ taskWords.toNat ≥ 2048 := u32_not_ge_of_lt hw
  have henc : encodeWord taskHeader0Layout #[("taskId", taskId), ("taskWords", taskWords)] =
      .ok (((0 : UInt32) ||| (taskId <<< 0)) ||| (taskWords <<< 16)) := by
    change (if taskId.toNat ≥ 65536 then _ else if taskWords.toNat ≥ 2048 then _ else _) = _
    simp only [h1, h2, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "taskId" taskId henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "taskWords" taskWords henc rfl⟩

theorem taskHeader8_roundTrip (ene : UInt32) (hene : ene < 8) :
    ∃ w, encodeWord taskHeader8Layout #[("ene", ene)] = .ok w ∧
         decodeFieldChecked taskHeader8Layout w "ene" = .ok ene := by
  have h1 : ¬ ene.toNat ≥ 8 := u32_not_ge_of_lt hene
  have henc : encodeWord taskHeader8Layout #[("ene", ene)] =
      .ok ((0x00000009 : UInt32) ||| (ene <<< 16)) := by
    change (if ene.toNat ≥ 8 then _ else _) = _
    simp only [h1, ↓reduceIte]
    rfl
  exact ⟨_, henc, encodeWord_decodeFieldChecked _ _ _ "ene" ene henc rfl⟩

theorem neCommon_roundTrip (kind shPref shMin shMax f wide3x3 : UInt32)
    (hk : kind < 256) (hsp : shPref < 8) (hsmn : shMin < 8)
    (hsmx : shMax < 8) (hf : f < 8) (hw : wide3x3 < 2) :
    ∃ w, encodeWord neCommonLayout #[
            ("kind", kind), ("shPref", shPref), ("shMin", shMin),
            ("shMax", shMax), ("F", f), ("wide3x3", wide3x3)
          ] = .ok w ∧
         decodeFieldChecked neCommonLayout w "kind" = .ok kind ∧
         decodeFieldChecked neCommonLayout w "shPref" = .ok shPref ∧
         decodeFieldChecked neCommonLayout w "shMin" = .ok shMin ∧
         decodeFieldChecked neCommonLayout w "shMax" = .ok shMax ∧
         decodeFieldChecked neCommonLayout w "F" = .ok f ∧
         decodeFieldChecked neCommonLayout w "wide3x3" = .ok wide3x3 := by
  have h1 : ¬ kind.toNat ≥ 256 := u32_not_ge_of_lt hk
  have h2 : ¬ shPref.toNat ≥ 8 := u32_not_ge_of_lt hsp
  have h3 : ¬ shMin.toNat ≥ 8 := u32_not_ge_of_lt hsmn
  have h4 : ¬ shMax.toNat ≥ 8 := u32_not_ge_of_lt hsmx
  have h5 : ¬ f.toNat ≥ 8 := u32_not_ge_of_lt hf
  have h6 : ¬ wide3x3.toNat ≥ 2 := u32_not_ge_of_lt hw
  have henc : encodeWord neCommonLayout #[
      ("kind", kind), ("shPref", shPref), ("shMin", shMin),
      ("shMax", shMax), ("F", f), ("wide3x3", wide3x3)
    ] = .ok (((((((0 : UInt32) ||| (kind <<< 0)) ||| (shPref <<< 8)) ||| (shMin <<< 12)) |||
      (shMax <<< 16)) ||| (f <<< 19)) ||| (wide3x3 <<< 27)) := by
    change (if kind.toNat ≥ 256 then _ else if shPref.toNat ≥ 8 then _ else
            if shMin.toNat ≥ 8 then _ else if shMax.toNat ≥ 8 then _ else
            if f.toNat ≥ 8 then _ else if wide3x3.toNat ≥ 2 then _ else _) = _
    simp only [h1, h2, h3, h4, h5, h6, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "kind" kind henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "shPref" shPref henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "shMin" shMin henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "shMax" shMax henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "F" f henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "wide3x3" wide3x3 henc rfl⟩

theorem convCfg_roundTrip (kw kh sx sy pl pt hx vy : UInt32)
    (hkw : kw < 64) (hkh : kh < 128) (hsx : sx < 4) (hsy : sy < 4)
    (hpl : pl < 32) (hpt : pt < 64) (hhx : hx < 4) (hvy : vy < 4) :
    ∃ w, encodeWord convCfgLayout #[
            ("kw", kw), ("kh", kh), ("sx", sx), ("sy", sy),
            ("pl", pl), ("pt", pt), ("hx", hx), ("vy", vy)
          ] = .ok w ∧
         decodeFieldChecked convCfgLayout w "kw" = .ok kw ∧
         decodeFieldChecked convCfgLayout w "kh" = .ok kh ∧
         decodeFieldChecked convCfgLayout w "sx" = .ok sx ∧
         decodeFieldChecked convCfgLayout w "sy" = .ok sy ∧
         decodeFieldChecked convCfgLayout w "pl" = .ok pl ∧
         decodeFieldChecked convCfgLayout w "pt" = .ok pt ∧
         decodeFieldChecked convCfgLayout w "hx" = .ok hx ∧
         decodeFieldChecked convCfgLayout w "vy" = .ok vy := by
  have h1 : ¬ kw.toNat ≥ 64 := u32_not_ge_of_lt hkw
  have h2 : ¬ kh.toNat ≥ 128 := u32_not_ge_of_lt hkh
  have h3 : ¬ sx.toNat ≥ 4 := u32_not_ge_of_lt hsx
  have h4 : ¬ sy.toNat ≥ 4 := u32_not_ge_of_lt hsy
  have h5 : ¬ pl.toNat ≥ 32 := u32_not_ge_of_lt hpl
  have h6 : ¬ pt.toNat ≥ 64 := u32_not_ge_of_lt hpt
  have h7 : ¬ hx.toNat ≥ 4 := u32_not_ge_of_lt hhx
  have h8 : ¬ vy.toNat ≥ 4 := u32_not_ge_of_lt hvy
  have henc : encodeWord convCfgLayout #[
      ("kw", kw), ("kh", kh), ("sx", sx), ("sy", sy),
      ("pl", pl), ("pt", pt), ("hx", hx), ("vy", vy)
    ] = .ok (((((((((0 : UInt32) ||| (kw <<< 0)) ||| (kh <<< 6)) ||| (sx <<< 13)) ||| (sy <<< 15)) |||
      (pl <<< 17)) ||| (pt <<< 22)) ||| (hx <<< 28)) ||| (vy <<< 30)) := by
    change (if kw.toNat ≥ 64 then _ else if kh.toNat ≥ 128 then _ else
            if sx.toNat ≥ 4 then _ else if sy.toNat ≥ 4 then _ else
            if pl.toNat ≥ 32 then _ else if pt.toNat ≥ 64 then _ else
            if hx.toNat ≥ 4 then _ else if vy.toNat ≥ 4 then _ else _) = _
    simp only [h1, h2, h3, h4, h5, h6, h7, h8, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "kw" kw henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "kh" kh henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "sx" sx henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "sy" sy henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "pl" pl henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "pt" pt henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "hx" hx henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "vy" vy henc rfl⟩

-- ============================================================================
-- 3. validateConvSafety soundness bounds
-- ============================================================================

/-- Any tiling accepted by `validateConvSafety` satisfies the core hardware safety bounds. -/
theorem validateConvSafety_bounds
    (target : Target) (sh : ConvSpec) (t : Tiling)
    (h : validateConvSafety target sh t = .ok ()) :
    t.logNE ≤ 4 ∧
    t.fat ≤ 1 ∧
    t.f + t.fat ≤ t.logNE ∧
    t.wu ≤ 3 ∧
    t.shPref ≤ t.shMax ∧
    t.shMax ≤ 4 ∧
    t.tileHeight ≥ 1 ∧
    l2UseBytesConv target sh t ≤ l2CapacityBytes := by
  unfold validateConvSafety at h
  obtain ⟨⟨⟩, _, h⟩ := except_bind_ok h
  obtain ⟨hLogNE, h⟩ := except_if_err_ok h
  obtain ⟨hFat, h⟩ := except_if_err_ok h
  obtain ⟨hFfat, h⟩ := except_if_err_ok h
  obtain ⟨hWu, h⟩ := except_if_err_ok h
  obtain ⟨⟨⟩, _, h⟩ := except_bind_ok h
  obtain ⟨⟨⟩, _, h⟩ := except_bind_ok h
  dsimp only at h
  obtain ⟨_, h⟩ := except_if_err_ok h
  obtain ⟨hSh, h⟩ := except_if_err_ok h
  obtain ⟨hShMax, h⟩ := except_if_err_ok h
  obtain ⟨hTh, h⟩ := except_if_err_ok h
  obtain ⟨hL2, _⟩ := except_if_err_ok h
  have hThNe : t.tileHeight ≠ 0 := by simpa using hTh
  exact ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩

-- ============================================================================
-- 4. groupOutputBlocks output channel conservation
-- ============================================================================

/-- Sum of `fn i` for `i = 0 .. n - 1`. -/
def sumRange : Nat → (Nat → Nat) → Nat
  | 0, _ => 0
  | n + 1, fn => sumRange n fn + fn n

private theorem sumRange_add (n : Nat) (a b : Nat → Nat) :
    sumRange n (fun i => a i + b i) = sumRange n a + sumRange n b := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp only [sumRange, ih]
    omega

private theorem sumRange_const (n c : Nat) :
    sumRange n (fun _ => c) = n * c := by
  induction n with
  | zero => simp [sumRange]
  | succ k ih =>
    simp only [sumRange, ih, Nat.succ_mul]

private theorem sumRange_lt_indicator (n m : Nat) :
    sumRange n (fun i => if i < m then 1 else 0) = min n m := by
  induction n with
  | zero => simp [sumRange]
  | succ k ih =>
    simp only [sumRange, ih]
    split <;> omega

private theorem foldl_ofFn_snd (k : Nat) (f1 : Nat → Nat) (blk init : Nat) :
    (Array.ofFn (n := k) (fun r => (f1 r.val, blk))).foldl (fun acc x => acc + x.2) init =
    init + k * blk := by
  induction k generalizing init with
  | zero => simp
  | succ m ih =>
    rw [Array.ofFn_succ, Array.foldl_push]
    simp only [Fin.val_castSucc]
    rw [ih]
    simp only [Nat.succ_mul]
    omega

private theorem groupOutputCount_formula (cout f ocg g : Nat) :
    groupOutputCount cout f ocg g =
      (cout / (((1 : Nat) <<< f) * ((1 : Nat) <<< ocg))) * ((1 : Nat) <<< ocg) +
      ((cout % (((1 : Nat) <<< f) * ((1 : Nat) <<< ocg))) / ((1 : Nat) <<< f)) +
      (if g < (cout % (((1 : Nat) <<< f) * ((1 : Nat) <<< ocg))) % ((1 : Nat) <<< f) then 1 else 0) := by
  unfold groupOutputCount groupOutputBlocks
  dsimp only
  generalize hg : ((1 : Nat) <<< f) = groups
  generalize hb : ((1 : Nat) <<< ocg) = blk
  generalize hr : (groups * blk) = round
  generalize hq : (cout / round) = full
  generalize hm : (cout % round) = rem
  have hfold : (Array.ofFn (n := full) fun r => (r.val * round + g * blk, blk)).foldl
      (fun acc x => acc + x.2) 0 = 0 + full * blk :=
    foldl_ofFn_snd full (fun r => r * round + g * blk) blk 0
  by_cases hrem : rem > 0
  · simp only [hrem, ↓reduceIte]
    by_cases hls : rem / groups + (if g < rem % groups then 1 else 0) > 0
    · simp only [hls, ↓reduceIte, Array.foldl_push]
      rw [hfold]
      omega
    · simp only [hls, ↓reduceIte]
      rw [hfold]
      have hdiv0 : 0 ≤ rem / groups := Nat.zero_le _
      omega
  · simp only [hrem, ↓reduceIte]
    rw [hfold]
    have hrem0 : rem = 0 := by omega
    simp [hrem0]

/-- Conservation of output channels: for any `cout`, `f`, `ocg`, summing the block sizes
    assigned by `groupOutputBlocks` across all groups `0 .. 2^f - 1` equals `cout`. -/
theorem groupOutputBlocks_sum (cout f ocg : Nat) :
    sumRange ((1 : Nat) <<< f) (fun g => groupOutputCount cout f ocg g) = cout := by
  have hgroupsPos : 0 < (1 : Nat) <<< f := by
    rw [Nat.shiftLeft_eq, Nat.one_mul]
    exact Nat.two_pow_pos f
  generalize hg : ((1 : Nat) <<< f) = groups at hgroupsPos ⊢
  generalize hb : ((1 : Nat) <<< ocg) = blk
  have hfn : (fun g => groupOutputCount cout f ocg g) =
      (fun g => ((cout / (groups * blk)) * blk + (cout % (groups * blk)) / groups) +
                (if g < (cout % (groups * blk)) % groups then 1 else 0)) := by
    funext g
    have hcnt := groupOutputCount_formula cout f ocg g
    rw [hg, hb] at hcnt
    omega
  rw [hfn, sumRange_add, sumRange_const, sumRange_lt_indicator]
  have hremModLt : (cout % (groups * blk)) % groups < groups := Nat.mod_lt _ hgroupsPos
  have hmin : min groups ((cout % (groups * blk)) % groups) = (cout % (groups * blk)) % groups :=
    Nat.min_eq_right (Nat.le_of_lt hremModLt)
  rw [hmin]
  have hdivRem : groups * ((cout % (groups * blk)) / groups) + (cout % (groups * blk)) % groups =
      cout % (groups * blk) := Nat.div_add_mod _ groups
  have hdivCout : (groups * blk) * (cout / (groups * blk)) + cout % (groups * blk) = cout :=
    Nat.div_add_mod cout (groups * blk)
  have hmul : groups * ((cout / (groups * blk)) * blk + (cout % (groups * blk)) / groups) =
      (groups * blk) * (cout / (groups * blk)) + groups * ((cout % (groups * blk)) / groups) := by
    rw [Nat.mul_add]
    have hassoc : groups * ((cout / (groups * blk)) * blk) = (groups * blk) * (cout / (groups * blk)) := by
      rw [← Nat.mul_assoc, Nat.mul_comm groups (cout / (groups * blk)), Nat.mul_assoc,
          Nat.mul_comm (cout / (groups * blk)) (groups * blk)]
    omega
  omega

-- ============================================================================
-- 5. O(1) inRows3x3 equivalence to linear tile scan
-- ============================================================================

/-- Reference linear scan computing `max_{0 <= i < n} tileInRows3x3 h s pt ho th i`. -/
def maxTileInRowsLinear (h s pt ho th : Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => max (maxTileInRowsLinear h s pt ho th n) (tileInRows3x3 h s pt ho th n)

private theorem tileInRows3x3_le_tile1 (h s pt ho th i : Nat)
    (hs : s = 1 ∨ s = 2) (hpt : pt ≤ 1) (hth : th ≥ 1)
    (hn3 : 3 ≤ ceilDiv ho th) (hi : 1 ≤ i) :
    tileInRows3x3 h s pt ho th i ≤ tileInRows3x3 h s pt ho th 1 := by
  have hhoGt : 2 * th < ho := by
    unfold ceilDiv at hn3
    have hth0 : (th == 0) = false := by simp; omega
    rw [hth0] at hn3
    simp only [Bool.false_eq_true, ↓reduceIte] at hn3
    have hmul : 3 * th ≤ (ho + th - 1) / th * th := Nat.mul_le_mul_right th hn3
    have hdiv : th * ((ho + th - 1) / th) + (ho + th - 1) % th = ho + th - 1 :=
      Nat.div_add_mod (ho + th - 1) th
    rw [Nat.mul_comm] at hdiv
    have hmodLt : (ho + th - 1) % th < th := Nat.mod_lt (ho + th - 1) (by omega)
    omega
  unfold tileInRows3x3
  dsimp only
  have h1y0 : 1 * th = th := Nat.one_mul th
  have h1y1 : min ho (th + th) = 2 * th := by omega
  have h1cond : ¬ (2 * th ≤ th) := by omega
  rw [h1y0, h1y1]
  simp only [h1cond, ↓reduceIte]
  split
  · omega
  · have hiMul : th ≤ i * th := by
      calc th = 1 * th := (Nat.one_mul th).symm
      _ ≤ i * th := Nat.mul_le_mul_right th hi
    rcases hs with rfl | rfl
    · have hrk : alignUp 3 1 = 3 := by decide
      rw [hrk]
      omega
    · have hrk : alignUp 3 2 = 4 := by decide
      rw [hrk]
      omega

private theorem maxTileInRowsLinear_le (h s pt ho th : Nat)
    (hs : s = 1 ∨ s = 2) (hpt : pt ≤ 1) (hth : th ≥ 1)
    (hn3 : 3 ≤ ceilDiv ho th) (k : Nat) :
    maxTileInRowsLinear h s pt ho th k ≤
      max (tileInRows3x3 h s pt ho th 0) (tileInRows3x3 h s pt ho th 1) := by
  induction k with
  | zero => simp [maxTileInRowsLinear]
  | succ m ih =>
    simp only [maxTileInRowsLinear]
    by_cases hm0 : m = 0
    · subst hm0
      omega
    · have hle := tileInRows3x3_le_tile1 h s pt ho th m hs hpt hth hn3 (by omega)
      omega

private theorem tile_le_maxTileInRowsLinear (h s pt ho th k i : Nat) (hi : i < k) :
    tileInRows3x3 h s pt ho th i ≤ maxTileInRowsLinear h s pt ho th k := by
  induction k with
  | zero => omega
  | succ m ih =>
    simp only [maxTileInRowsLinear]
    by_cases hEq : i = m
    · subst hEq; omega
    · have := ih (by omega); omega

/-- O(1) `inRows3x3Raw` equals the exact maximum of `tileInRows3x3` across all `ceilDiv ho th` tiles. -/
theorem inRows3x3_eq_max (h s pt ho th : Nat)
    (hs : s = 1 ∨ s = 2) (hpt : pt ≤ 1) (hth : th ≥ 1) (hho : ho ≥ 1) :
    inRows3x3Raw h s pt ho th = maxTileInRowsLinear h s pt ho th (ceilDiv ho th) := by
  unfold inRows3x3Raw
  have hcond : (th == 0 || ho == 0) = false := by simp; omega
  rw [hcond]
  simp only [Bool.false_eq_true, ↓reduceIte]
  have hnPos : 1 ≤ ceilDiv ho th := by
    unfold ceilDiv
    have hth0 : (th == 0) = false := by simp; omega
    rw [hth0]; simp only [Bool.false_eq_true, ↓reduceIte]
    exact Nat.div_pos (by omega) (by omega)
  generalize hn : ceilDiv ho th = n at hnPos ⊢
  by_cases hn1 : n ≤ 1
  · have hnEq : n = 1 := by omega
    simp only [hn1, ↓reduceIte]
    rw [hnEq]
    simp [maxTileInRowsLinear]
  · simp only [hn1, ↓reduceIte]
    by_cases hn2 : n = 2
    · simp only [hn2, beq_self_eq_true, ↓reduceIte, maxTileInRowsLinear]
      omega
    · have hn2b : (n == 2) = false := by simp [hn2]
      simp only [hn2b, Bool.false_eq_true, ↓reduceIte]
      have hn3 : 3 ≤ ceilDiv ho th := by omega
      have hLastLe := tileInRows3x3_le_tile1 h s pt ho th (n - 1) hs hpt hth hn3 (by omega)
      have hLinLe := maxTileInRowsLinear_le h s pt ho th hs hpt hth hn3 n
      have h0Le := tile_le_maxTileInRowsLinear h s pt ho th n 0 (by omega)
      have h1Le := tile_le_maxTileInRowsLinear h s pt ho th n 1 (by omega)
      omega

-- ============================================================================
-- 6. Milestone 3 safety invariants (PE anti-hang, 128B LUT alignment, peTile packing)
-- ============================================================================

/-- PE anti-hang channel stride invariant: rounding width up to 8 never violates the
    hardware `l2ChanStride >= 2 * min(w, T)` minimum for any `w, T : Nat`. -/
theorem pe_antihang_chanStride_ge (w T : Nat) :
    2 * min w T ≤ 2 * min (alignUp w 8) T := by
  have h : w ≤ alignUp w 8 := alignUp_ge w 8
  omega

/-- PE 64-byte DMA line floor invariant: for `P = 1` DMA-fed ports, using `max T 32`
    both dominates the base `2 * min(w, T)` anti-hang bound and guarantees `>= 64` bytes
    whenever `alignUp w 8 >= 32`. -/
theorem pe_dma_line_floor_ge (w T : Nat) :
    2 * min w T ≤ 2 * min (alignUp w 8) (max T 32) ∧
    (32 ≤ alignUp w 8 → 64 ≤ 2 * min (alignUp w 8) (max T 32)) := by
  have h : w ≤ alignUp w 8 := alignUp_ge w 8
  constructor <;> omega

/-- Prepending a 128-byte LUT table preserves 64-byte DMA line alignment for every chunk offset. -/
theorem fused_lut_align64 (off : Nat) (h : off % 64 = 0) :
    (off + lutTableBytes) % 64 = 0 := by
  unfold lutTableBytes
  omega

/-- `peTile` packing bound: any legal `(k <= 9, m <= 5)` packs cleanly into 8 bits (`< 256`). -/
theorem peTile_pack_lt_256 (k m : Nat) (hk : k ≤ 9) (hm : m ≤ 5) :
    (m <<< 4) ||| k < 256 := by
  have h1 : m <<< 4 < 2 ^ 8 := by
    rw [Nat.shiftLeft_eq]
    omega
  have h2 : k < 2 ^ 8 := by omega
  have hor := Nat.or_lt_two_pow h1 h2
  omega

/-- Production `peTileM` is always `<= 5` whenever `k >= 4`. -/
theorem peTileM_le_5 (twoSource : Bool) (k th : Nat) (hk : 4 ≤ k) :
    peTileM twoSource k th ≤ 5 := by
  unfold peTileM peMaxK
  cases twoSource <;> dsimp <;> omega

/-- Production `peDirectResBytes` always reserves at least 2048 bytes of L2 headroom. -/
theorem peDirectResBytes_ge_2048 (outC drow th : Nat) :
    2048 ≤ peDirectResBytes outC drow th := by
  unfold peDirectResBytes
  have hge := alignUp_ge (max 2048 (min (outC * drow * th) (4 * drow * th))) Vane.Program.dmaLineBytes
  omega

/-- Production `peDirectResBytes` is always 64-byte DMA-line aligned. -/
theorem peDirectResBytes_dvd64 (outC drow th : Nat) :
    peDirectResBytes outC drow th % 64 = 0 := by
  unfold peDirectResBytes
  exact alignUp_dvd _ 64 (by omega)

/-- Production `neActDirectResBytes` is always strictly positive (`>= 512` bytes). -/
theorem neActDirectResBytes_pos (f fat : Nat) :
    0 < neActDirectResBytes f fat := by
  unfold neActDirectResBytes
  rw [Nat.shiftLeft_eq, Nat.one_mul]
  have hpow : 0 < 2 ^ (f + fat) := Nat.two_pow_pos (f + fat)
  omega

/-- Any PE tiling accepted by production `validatePeTiling` satisfies the core hardware safety bounds. -/
theorem validatePeTiling_bounds
    (target : Target) (outDims hwSrc1 : TensorDims) (hwSrc2? : Option TensorDims) (t : PeTiling)
    (hv : validatePeTiling target outDims hwSrc1 hwSrc2? t = .ok ()) :
    4 ≤ t.k ∧
    t.k ≤ peMaxK hwSrc2?.isSome ∧
    1 ≤ t.tileHeight ∧
    t.tileHeight ≤ outDims.h ∧
    (computePeL2Layout target outDims hwSrc1 hwSrc2? t).l2UseBytes ≤ l2CapacityBytes := by
  unfold validatePeTiling at hv
  obtain ⟨⟨⟩, _, hv⟩ := except_bind_ok hv
  obtain ⟨⟨⟩, _, hv⟩ := except_bind_ok hv
  obtain ⟨⟨⟩, _, hv⟩ := except_bind_ok hv
  dsimp only at hv
  obtain ⟨hkLo, hv⟩ := except_if_err_ok hv
  obtain ⟨hkHi, hv⟩ := except_if_err_ok hv
  obtain ⟨hth0, hv⟩ := except_if_err_ok hv
  obtain ⟨hthHi, hv⟩ := except_if_err_ok hv
  obtain ⟨_, hv⟩ := except_if_err_ok hv
  obtain ⟨_, hv⟩ := except_if_err_ok hv
  obtain ⟨_, hv⟩ := except_if_err_ok hv
  obtain ⟨hL2, _⟩ := except_if_err_ok hv
  have hthNe : t.tileHeight ≠ 0 := by simpa using hth0
  exact ⟨by omega, by omega, by omega, by omega, by omega⟩

/-- Any standalone NE activation tiling accepted by production `validateNeActTiling`
    satisfies all core hardware safety bounds including L2 direct-ring capacity. -/
theorem validateNeActTiling_bounds
    (target : Target) (c h w : Nat) (t : NeActTiling)
    (hv : validateNeActTiling target c h w t = .ok ()) :
    t.logNE ≤ 4 ∧
    t.fat ≤ 1 ∧
    t.f + t.fat ≤ t.logNE ∧
    t.wu ≤ 3 ∧
    t.shPref ≤ t.shMax ∧
    t.shMax ≤ 4 ∧
    1 ≤ t.tileHeight ∧
    t.tileHeight ≤ h ∧
    (computeNeActL2 target c h w t).l2UseBytes ≤ l2CapacityBytes := by
  unfold validateNeActTiling at hv
  obtain ⟨_, hv⟩ := except_if_err_ok hv
  obtain ⟨_, hv⟩ := except_if_err_ok hv
  obtain ⟨_, hv⟩ := except_if_err_ok hv
  obtain ⟨hLogNE, hv⟩ := except_if_err_ok hv
  obtain ⟨hFat, hv⟩ := except_if_err_ok hv
  obtain ⟨hFfat, hv⟩ := except_if_err_ok hv
  obtain ⟨hWu, hv⟩ := except_if_err_ok hv
  obtain ⟨hShPref, hv⟩ := except_if_err_ok hv
  obtain ⟨hShMax, hv⟩ := except_if_err_ok hv
  dsimp only at hv
  obtain ⟨_, hv⟩ := except_if_err_ok hv
  obtain ⟨hth0, hv⟩ := except_if_err_ok hv
  obtain ⟨hthHi, hv⟩ := except_if_err_ok hv
  obtain ⟨hL2, _⟩ := except_if_err_ok hv
  have hthNe : t.tileHeight ≠ 0 := by simpa using hth0
  exact ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩

-- ============================================================================
-- 7. Milestone 4: Multi-task WordLayout round-trips & [DL] / header invariants
-- ============================================================================

theorem taskHeader2_roundTrip (firstFlags lastFlag : UInt32)
    (hff : firstFlags < 256) (hlf : lastFlag < 2) :
    ∃ w, encodeWord taskHeader2Layout #[("firstFlags", firstFlags), ("lastFlag", lastFlag)] = .ok w ∧
         decodeFieldChecked taskHeader2Layout w "firstFlags" = .ok firstFlags ∧
         decodeFieldChecked taskHeader2Layout w "lastFlag" = .ok lastFlag := by
  have h1 : ¬ firstFlags.toNat ≥ 256 := u32_not_ge_of_lt hff
  have h2 : ¬ lastFlag.toNat ≥ 2 := u32_not_ge_of_lt hlf
  have henc : encodeWord taskHeader2Layout #[("firstFlags", firstFlags), ("lastFlag", lastFlag)] =
      .ok (((0 : UInt32) ||| (firstFlags <<< 0)) ||| (lastFlag <<< 26)) := by
    change (if firstFlags.toNat ≥ 256 then _ else if lastFlag.toNat ≥ 2 then _ else _) = _
    simp only [h1, h2, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "firstFlags" firstFlags henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "lastFlag" lastFlag henc rfl⟩

theorem taskHeader8Multi_roundTrip (active isLast ene : UInt32)
    (ha : active < 2) (hl : isLast < 2) (he : ene < 8) :
    ∃ w, encodeWord taskHeader8MultiLayout #[("active", active), ("isLast", isLast), ("ene", ene)] = .ok w ∧
         decodeFieldChecked taskHeader8MultiLayout w "active" = .ok active ∧
         decodeFieldChecked taskHeader8MultiLayout w "isLast" = .ok isLast ∧
         decodeFieldChecked taskHeader8MultiLayout w "ene" = .ok ene := by
  have h1 : ¬ active.toNat ≥ 2 := u32_not_ge_of_lt ha
  have h2 : ¬ isLast.toNat ≥ 2 := u32_not_ge_of_lt hl
  have h3 : ¬ ene.toNat ≥ 8 := u32_not_ge_of_lt he
  have henc : encodeWord taskHeader8MultiLayout #[("active", active), ("isLast", isLast), ("ene", ene)] =
      .ok ((((0 : UInt32) ||| (active <<< 0)) ||| (isLast <<< 3)) ||| (ene <<< 16)) := by
    change (if active.toNat ≥ 2 then _ else if isLast.toNat ≥ 2 then _ else if ene.toNat ≥ 8 then _ else _) = _
    simp only [h1, h2, h3, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "active" active henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "isLast" isLast henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "ene" ene henc rfl⟩

/-- Any 9-word multi-task `header[8]` encoded with `active = 1` never sets bit 1,
    so `(w &&& 3) ≠ 3` and task-stream decoders always classify it as a 9-word header. -/
theorem taskHeader8Multi_never_10word (isLast ene : UInt32)
    (hl : isLast < 2) (he : ene < 8) :
    ∃ w, encodeWord taskHeader8MultiLayout #[("active", 1), ("isLast", isLast), ("ene", ene)] = .ok w ∧
         (w &&& 3) ≠ 3 := by
  have h2 : ¬ isLast.toNat ≥ 2 := u32_not_ge_of_lt hl
  have h3 : ¬ ene.toNat ≥ 8 := u32_not_ge_of_lt he
  have henc : encodeWord taskHeader8MultiLayout #[("active", 1), ("isLast", isLast), ("ene", ene)] =
      .ok ((((0 : UInt32) ||| ((1 : UInt32) <<< 0)) ||| (isLast <<< 3)) ||| (ene <<< 16)) := by
    change (if isLast.toNat ≥ 2 then _ else if ene.toNat ≥ 8 then _ else _) = _
    simp only [h2, h3, ↓reduceIte]
    rfl
  refine ⟨_, henc, ?_⟩
  intro heq
  have hbit := congrArg (fun u : UInt32 => u.toBitVec[1]) heq
  have h3t : (3 : UInt32).toBitVec[1] = true := by simp [BitVec.getElem_eq_testBit_toNat]; decide
  have hbit1 : (((((0 : UInt32) ||| ((1 : UInt32) <<< 0)) ||| (isLast <<< 3)) ||| (ene <<< 16)) &&& 3).toBitVec[1] = false := by
    have h0 : (0 : UInt32).toBitVec[1] = false := by simp [BitVec.getElem_eq_testBit_toNat]
    have hone : ((1 : UInt32) <<< 0).toBitVec[1] = false := by simp [BitVec.getElem_eq_testBit_toNat]; decide
    have hl1 : (isLast <<< 3).toBitVec[1] = false := by
      rw [BitVec.getElem_eq_testBit_toNat]
      change (isLast <<< 3).toNat.testBit 1 = false
      rw [UInt32.toNat_shiftLeft]
      change ((isLast.toNat <<< 3) % 2 ^ 32).testBit 1 = false
      rw [Nat.testBit_mod_two_pow]
      simp [Nat.testBit_shiftLeft]
    have he1 : (ene <<< 16).toBitVec[1] = false := by
      rw [BitVec.getElem_eq_testBit_toNat]
      change (ene <<< 16).toNat.testBit 1 = false
      rw [UInt32.toNat_shiftLeft]
      change ((ene.toNat <<< 16) % 2 ^ 32).testBit 1 = false
      rw [Nat.testBit_mod_two_pow]
      simp [Nat.testBit_shiftLeft]
    rw [u32_bit_and _ _ 1 (by omega), u32_bit_or _ _ 1 (by omega),
        u32_bit_or _ _ 1 (by omega), u32_bit_or _ _ 1 (by omega),
        h0, hone, hl1, he1]
    rfl
  rw [hbit1, h3t] at hbit
  cases hbit

theorem srcDepMulti_roundTrip (name : String) (code task depDist depKind : UInt32)
    (hc : code < 256) (ht : task < 256) (hd : depDist < 16) (hk : depKind < 4) :
    ∃ w, encodeWord (srcDepMultiLayout name) #[
            ("code", code), ("task", task), ("depDist", depDist), ("depKind", depKind)
          ] = .ok w ∧
         decodeFieldChecked (srcDepMultiLayout name) w "code" = .ok code ∧
         decodeFieldChecked (srcDepMultiLayout name) w "task" = .ok task ∧
         decodeFieldChecked (srcDepMultiLayout name) w "depDist" = .ok depDist ∧
         decodeFieldChecked (srcDepMultiLayout name) w "depKind" = .ok depKind := by
  have h1 : ¬ code.toNat ≥ 256 := u32_not_ge_of_lt hc
  have h2 : ¬ task.toNat ≥ 256 := u32_not_ge_of_lt ht
  have h3 : ¬ depDist.toNat ≥ 16 := u32_not_ge_of_lt hd
  have h4 : ¬ depKind.toNat ≥ 4 := u32_not_ge_of_lt hk
  have hlay : validateWordLayout (srcDepMultiLayout name) = .ok 0x3fff00ff := by
    simp [validateWordLayout, srcDepMultiLayout, validateWordLayoutFields, BitField.mask, BitField.maskNat]
    rfl
  have hval : validateEncodeValues name (srcDepMultiLayout name).fields.toList
      [("code", code), ("task", task), ("depDist", depDist), ("depKind", depKind)] [] = .ok () := by
    simp [srcDepMultiLayout, validateEncodeValues]
  have henc : encodeWord (srcDepMultiLayout name) #[
      ("code", code), ("task", task), ("depDist", depDist), ("depKind", depKind)
    ] = .ok (((((0 : UInt32) ||| (code <<< 0)) ||| (task <<< 16)) ||| (depDist <<< 24)) ||| (depKind <<< 28)) := by
    unfold encodeWord
    rw [hlay]
    simp only [bind, Except.bind]
    have hval' : validateEncodeValues (srcDepMultiLayout name).name (srcDepMultiLayout name).fields.toList
        #[("code", code), ("task", task), ("depDist", depDist), ("depKind", depKind)].toList [] = .ok () := hval
    rw [hval']
    simp [srcDepMultiLayout, encodeWordFields, List.find?]
    have c1 : ¬ 256 ≤ code.toNat := by omega
    have c2 : ¬ 256 ≤ task.toNat := by omega
    have c3 : ¬ 16 ≤ depDist.toNat := by omega
    have c4 : ¬ 4 ≤ depKind.toNat := by omega
    simp [c1, c2, c3, c4]
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "code" code henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "task" task henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "depDist" depDist henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "depKind" depKind henc rfl⟩

theorem kernelStageMulti_roundTrip (sparse kernEn isLast kernCfg stage : UInt32)
    (hsp : sparse < 2) (hke : kernEn < 2) (hil : isLast < 2) (hkc : kernCfg < 2) (hst : stage < 256) :
    ∃ w, encodeWord kernelStageMultiLayout #[
            ("sparse", sparse), ("kernEn", kernEn), ("isLast", isLast), ("kernCfg", kernCfg), ("stage", stage)
          ] = .ok w ∧
         decodeFieldChecked kernelStageMultiLayout w "sparse" = .ok sparse ∧
         decodeFieldChecked kernelStageMultiLayout w "kernEn" = .ok kernEn ∧
         decodeFieldChecked kernelStageMultiLayout w "isLast" = .ok isLast ∧
         decodeFieldChecked kernelStageMultiLayout w "kernCfg" = .ok kernCfg ∧
         decodeFieldChecked kernelStageMultiLayout w "stage" = .ok stage := by
  have h1 : ¬ sparse.toNat ≥ 2 := u32_not_ge_of_lt hsp
  have h2 : ¬ kernEn.toNat ≥ 2 := u32_not_ge_of_lt hke
  have h3 : ¬ isLast.toNat ≥ 2 := u32_not_ge_of_lt hil
  have h4 : ¬ kernCfg.toNat ≥ 2 := u32_not_ge_of_lt hkc
  have h5 : ¬ stage.toNat ≥ 256 := u32_not_ge_of_lt hst
  have henc : encodeWord kernelStageMultiLayout #[
      ("sparse", sparse), ("kernEn", kernEn), ("isLast", isLast), ("kernCfg", kernCfg), ("stage", stage)
    ] = .ok ((((((0 : UInt32) ||| (sparse <<< 5)) ||| (kernEn <<< 6)) ||| (isLast <<< 7)) |||
      (kernCfg <<< 9)) ||| (stage <<< 16)) := by
    change (if sparse.toNat ≥ 2 then _ else if kernEn.toNat ≥ 2 then _ else
            if isLast.toNat ≥ 2 then _ else if kernCfg.toNat ≥ 2 then _ else
            if stage.toNat ≥ 256 then _ else _) = _
    simp only [h1, h2, h3, h4, h5, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "sparse" sparse henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "kernEn" kernEn henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "isLast" isLast henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "kernCfg" kernCfg henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "stage" stage henc rfl⟩

/-- `[DL]` Direct PE DRAM write line rule: whenever `peIsDirect` authorizes direct mode,
    either the output fits in a single strip (`outW ≤ 2^k`) or each strip spans an exact
    multiple of 64-byte DMA lines (`2^k % 32 = 0` at `P = 1`). -/
theorem peIsDirect_line_rule (target : Target) (outW k : Nat)
    (h : peIsDirect target outW k = true) :
    outW ≤ peStripWidth k ∨ peStripWidth k % 32 = 0 := by
  unfold peIsDirect at h
  dsimp only at h
  by_cases hc : outW > peStripWidth k ∧ peStripWidth k % 32 ≠ 0
  · have hcond : (decide (outW > peStripWidth k) && (peStripWidth k % 32 != 0)) = true := by
      simp [hc.1, hc.2]
    rw [hcond] at h
    simp at h
  · omega

-- ============================================================================
-- 8. Milestone 4: Vane.Alloc formal disjointness, packing & capacity proofs
-- ============================================================================

private theorem apart_symm {x y : Range} (h : Apart x y) : Apart y x := by
  unfold Apart at *
  exact h.symm

private theorem meets_symm {f1 l1 f2 l2 : Nat} (h : meets f1 l1 f2 l2) : meets f2 l2 f1 l1 :=
  ⟨h.2, h.1⟩

private theorem le_topOf {ps : List PlacedBuf} {p : PlacedBuf} (hp : p ∈ ps) :
    p.hi ≤ topOf ps := by
  induction ps with
  | nil => cases hp
  | cons hd tl ih =>
    simp only [topOf]
    cases hp with
    | head => exact Nat.le_max_left _ _
    | tail _ htl => exact Nat.le_trans (ih htl) (Nat.le_max_right _ _)

/-- Every buffer in any placed list ends by `topOf`. -/
theorem allocate_le_top (firstFit : Bool) (bs : List LiveBuf) :
    ∀ p ∈ allocate firstFit bs, p.hi ≤ topOf (allocate firstFit bs) :=
  fun _ hp => le_topOf hp

private theorem foldl_min_preserves {P : Nat → Prop} (xs : List Nat) (init : Nat)
    (hinit : P init) (hxs : ∀ x ∈ xs, P x) :
    P (xs.foldl min init) := by
  induction xs generalizing init with
  | nil => exact hinit
  | cons hd tl ih =>
    simp only [List.foldl_cons]
    apply ih
    · by_cases hle : init ≤ hd
      · rwa [Nat.min_eq_left hle]
      · rw [Nat.min_eq_right (Nat.le_of_not_le hle)]
        exact hxs hd List.mem_cons_self
    · intro x hx
      exact hxs x (List.mem_cons_of_mem _ hx)

private theorem clearFor_topOf (conf : List PlacedBuf) (bytes : Nat) :
    clearFor conf bytes (topOf conf) = true := by
  unfold clearFor
  rw [List.all_eq_true]
  intro p hp
  simp only [decide_eq_true_eq]
  left
  exact le_topOf hp

private theorem pickFirstFit_clear (conf : List PlacedBuf) (bytes : Nat) :
    clearFor conf bytes (pickFirstFit conf bytes) = true := by
  unfold pickFirstFit
  apply foldl_min_preserves (P := fun off => clearFor conf bytes off = true)
  · exact clearFor_topOf conf bytes
  · intro x hx
    rw [List.mem_filter] at hx
    exact hx.2

private theorem placeOne_apart (firstFit : Bool) (acc : List PlacedBuf) (b : LiveBuf) :
    ∀ p ∈ acc, (placeOne firstFit acc b).conflicts p →
      Apart (placeOne firstFit acc b).toRange p.toRange := by
  intro p hp hconf
  unfold placeOne at hconf ⊢
  dsimp only [PlacedBuf.conflicts] at hconf
  have hpConf : p ∈ acc.filter (fun q => decide (meets b.first b.last q.first q.last)) := by
    rw [List.mem_filter]
    exact ⟨hp, by simpa using hconf⟩
  cases firstFit
  · simp only [Bool.false_eq_true, ↓reduceIte]
    apply apart_symm
    left
    exact le_topOf hpConf
  · simp only [↓reduceIte]
    have hcl := pickFirstFit_clear (acc.filter (fun q => decide (meets b.first b.last q.first q.last))) b.bytes
    unfold clearFor at hcl
    rw [List.all_eq_true] at hcl
    have hpAp := hcl p hpConf
    simp only [decide_eq_true_eq] at hpAp
    exact apart_symm hpAp

private theorem placeAll_disjoint (firstFit : Bool) (bs : List LiveBuf) (acc : List PlacedBuf)
    (hacc : Disjoint acc) : Disjoint (placeAll firstFit bs acc) := by
  induction bs generalizing acc with
  | nil => exact hacc
  | cons b rest ih =>
    unfold placeAll
    apply ih
    exact List.Pairwise.cons (placeOne_apart firstFit acc b) hacc

private theorem pairwise_reverse_symm {α : Type} {R : α → α → Prop}
    (hsym : ∀ a b, R a b → R b a) (xs : List α) (h : xs.Pairwise R) :
    xs.reverse.Pairwise R := by
  rw [List.pairwise_reverse]
  exact h.imp (fun {a b} hab => hsym a b hab)

/-- Any two live-conflicting buffers placed by `allocate` occupy disjoint byte ranges. -/
theorem allocate_disjoint (firstFit : Bool) (bs : List LiveBuf) :
    Disjoint (allocate firstFit bs) := by
  unfold allocate Disjoint
  apply pairwise_reverse_symm
  · intro a b hab hba
    exact apart_symm (hab (meets_symm hba))
  · exact placeAll_disjoint firstFit _ [] List.Pairwise.nil

/-- Buffers live at any single task `t` are pairwise `Apart`. -/
theorem liveAt_pairwise (ps : List PlacedBuf) (t : Nat) (hd : Disjoint ps) :
    (liveAt ps t).Pairwise Apart := by
  unfold liveAt
  rw [List.pairwise_map]
  have hfilt : (ps.filter (fun p => decide (p.live t))).Pairwise
      (fun a b => a.live t ∧ b.live t ∧ (a.conflicts b → Apart a.toRange b.toRange)) := by
    induction ps with
    | nil => exact List.Pairwise.nil
    | cons p rest ih =>
      have ⟨hp, hrest⟩ := List.pairwise_cons.mp hd
      simp only [List.filter_cons]
      split
      · rename_i hlive
        have hpl : p.live t := by simpa using hlive
        apply List.Pairwise.cons
        · intro q hq
          rw [List.mem_filter] at hq
          have hql : q.live t := by simpa using hq.2
          exact ⟨hpl, hql, hp q hq.1⟩
        · exact ih hrest
      · exact ih hrest
  exact hfilt.imp fun {a b} h =>
    h.2.2 ⟨Nat.le_trans h.1.1 h.2.1.2, Nat.le_trans h.2.1.1 h.1.2⟩

private theorem bytesOf_partition (p : Range → Bool) (rs : List Range) :
    bytesOf rs = bytesOf (rs.filter p) + bytesOf (rs.filter (fun x => !p x)) := by
  induction rs with
  | nil => simp [bytesOf]
  | cons r rest ih =>
    simp only [bytesOf, List.filter_cons]
    cases h : p r <;> simp only [Bool.not_false, Bool.not_true, bytesOf, ↓reduceIte, Bool.false_eq_true] <;> omega

private theorem filter_length_le {α : Type} (p : α → Bool) (xs : List α) :
    (xs.filter p).length ≤ xs.length := by
  induction xs with
  | nil => simp
  | cons x rest ih =>
    simp only [List.filter_cons, List.length_cons]
    cases p x <;> simp <;> omega

private theorem sum_le_of_packed_in (n lo hi : Nat) (rs : List Range)
    (hlen : rs.length ≤ n)
    (hpw : rs.Pairwise Apart)
    (hlo : ∀ r ∈ rs, lo ≤ r.base)
    (hhi : ∀ r ∈ rs, r.hi ≤ hi) :
    bytesOf rs ≤ hi - lo := by
  induction n generalizing lo hi rs with
  | zero =>
    cases rs with
    | nil => simp [bytesOf]
    | cons _ _ => simp at hlen
  | succ m ih =>
    cases rs with
    | nil => simp [bytesOf]
    | cons r rest =>
      have ⟨hrAll, hrestPw⟩ := List.pairwise_cons.mp hpw
      have hrLo : lo ≤ r.base := hlo r List.mem_cons_self
      have hrHi : r.hi ≤ hi := hhi r List.mem_cons_self
      have hrestLen : rest.length ≤ m := by simp at hlen; omega
      let p : Range → Bool := fun x => decide (x.hi ≤ r.base)
      let left := rest.filter p
      let right := rest.filter (fun x => !p x)
      have hpart := bytesOf_partition p rest
      have hleftLen : left.length ≤ m := Nat.le_trans (filter_length_le p rest) hrestLen
      have hrightLen : right.length ≤ m := Nat.le_trans (filter_length_le _ rest) hrestLen
      have hleftPw : left.Pairwise Apart := hrestPw.filter p
      have hrightPw : right.Pairwise Apart := hrestPw.filter (fun x => !p x)
      have hleftLo : ∀ x ∈ left, lo ≤ x.base := fun x hx =>
        hlo x (List.mem_cons_of_mem _ (List.mem_filter.mp hx).1)
      have hleftHi : ∀ x ∈ left, x.hi ≤ r.base := by
        intro x hx
        have hpx := (List.mem_filter.mp hx).2
        simpa [p] using hpx
      have hrightLo : ∀ x ∈ right, r.hi ≤ x.base := by
        intro x hx
        have ⟨hmem, hnpx⟩ := List.mem_filter.mp hx
        have hnot : ¬ (x.hi ≤ r.base) := by
          intro hle
          have : p x = true := by simp [p, hle]
          simp [this] at hnpx
        cases hrAll x hmem with
        | inl hle => exact hle
        | inr hle => exact absurd hle hnot
      have hrightHi : ∀ x ∈ right, x.hi ≤ hi := fun x hx =>
        hhi x (List.mem_cons_of_mem _ (List.mem_filter.mp hx).1)
      have hSumLeft : bytesOf left ≤ r.base - lo := ih lo r.base left hleftLen hleftPw hleftLo hleftHi
      have hSumRight : bytesOf right ≤ hi - r.hi := ih r.hi hi right hrightLen hrightPw hrightLo hrightHi
      dsimp only [left, right] at hSumLeft hSumRight
      simp only [bytesOf, Range.hi] at hrHi hSumRight ⊢
      rw [hpart]
      omega

/-- Any list of pairwise `Apart` ranges packed within `[0, cap)` has total bytes `≤ cap`. -/
theorem sum_le_of_packed (cap : Nat) (rs : List Range) (hp : Packed cap rs) :
    bytesOf rs ≤ cap := by
  have h := sum_le_of_packed_in rs.length 0 cap rs (Nat.le_refl _) hp.1
    (fun _ _ => Nat.zero_le _) hp.2
  omega

/-- Total live footprint at any task `t` is bounded by `cap` whenever placed buffers are
    live-disjoint and end within `cap`. -/
theorem liveAt_sum (ps : List PlacedBuf) (t cap : Nat)
    (hd : Disjoint ps) (hcap : ∀ p ∈ ps, p.hi ≤ cap) :
    bytesOf (liveAt ps t) ≤ cap := by
  apply sum_le_of_packed cap (liveAt ps t)
  refine ⟨liveAt_pairwise ps t hd, ?_⟩
  intro r hr
  unfold liveAt at hr
  rw [List.mem_map] at hr
  obtain ⟨p, hp, rfl⟩ := hr
  rw [List.mem_filter] at hp
  exact hcap p hp.1

/-- Any offset returned by `findFreeOffset` fits inside `cap` and is `Apart` from every occupied range. -/
theorem findFreeOffset_sound (occupied : List Range) (bytes cap off : Nat)
    (h : findFreeOffset occupied bytes cap = some off) :
    off + bytes ≤ cap ∧ ∀ r ∈ occupied, Apart r { base := off, bytes := bytes } := by
  unfold findFreeOffset at h
  dsimp only at h
  generalize hv : (0 :: occupied.map Range.hi).filter (fun o =>
    decide (o + bytes ≤ cap) &&
    occupied.all (fun r => decide (Apart r { base := o, bytes := bytes }))) = valid at h
  cases valid with
  | nil => simp at h
  | cons first rest =>
    simp only [Option.some.injEq] at h
    subst h
    have hpred : ∀ x ∈ first :: rest,
        x + bytes ≤ cap ∧ ∀ r ∈ occupied, Apart r { base := x, bytes := bytes } := by
      intro x hx
      rw [← hv, List.mem_filter] at hx
      have hcond := hx.2
      simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at hcond
      exact hcond
    exact foldl_min_preserves rest first
      (hpred first List.mem_cons_self)
      (fun x hx => hpred x (List.mem_cons_of_mem _ hx))

-- ============================================================================
-- 9. Milestone 4: Vane.Reach forward-edge transitive closure soundness & completeness
-- ============================================================================

private theorem anyLt_iff (p : Nat → Bool) (m : Nat) :
    anyLt p m = true ↔ ∃ j, j < m ∧ p j = true := by
  induction m with
  | zero =>
    simp [anyLt]
  | succ k ih =>
    simp only [anyLt, Bool.or_eq_true, ih]
    constructor
    · rintro (⟨j, hj, hp⟩ | hp)
      · exact ⟨j, by omega, hp⟩
      · exact ⟨k, by omega, hp⟩
    · rintro ⟨j, hj, hp⟩
      by_cases hjk : j < k
      · left; exact ⟨j, hjk, hp⟩
      · right
        have hEq : j = k := by omega
        subst hEq
        exact hp

private theorem reachTableAux_size (step : Nat → Nat → Bool) (n m : Nat) :
    (reachTableAux step n m).size = m := by
  induction m with
  | zero => rfl
  | succ k ih =>
    simp [reachTableAux, ih]

private theorem reachTableAux_getElem?_lt (step : Nat → Nat → Bool) (n m i : Nat) (hi : i < m) :
    (reachTableAux step n m)[i]? = (reachTableAux step n (i + 1))[i]? := by
  induction m with
  | zero => omega
  | succ k ih =>
    by_cases hik : i = k
    · subst hik; rfl
    · have hlt : i < k := by omega
      conv => lhs; unfold reachTableAux
      dsimp only
      have hsz : (reachTableAux step n k).size = k := reachTableAux_size step n k
      have hne : i ≠ (reachTableAux step n k).size := by omega
      simp only [Array.getElem?_push, hne, ↓reduceIte]
      exact ih hlt

private theorem transGen_lt_of_fwd {step : Nat → Nat → Bool}
    (hFwd : ∀ a b, step a b = true → a < b) {k i : Nat}
    (h : Relation.TransGen (fun a b => step a b = true) k i) : k < i := by
  induction h with
  | single hki => exact hFwd _ _ hki
  | tail _ hji ih => exact Nat.lt_trans ih (hFwd _ _ hji)

private theorem ofFn_getD {n : Nat} (f : Fin n → Bool) (k : Nat) :
    (Array.ofFn f).getD k false = if h : k < n then f ⟨k, h⟩ else false := by
  unfold Array.getD
  split
  · rename_i hlt
    have hk : k < n := by simpa [Array.size_ofFn] using hlt
    simp only [hk, ↓reduceDIte]
    exact Array.getElem_ofFn hlt
  · rename_i hnot
    have hk : ¬ k < n := by simpa [Array.size_ofFn] using hnot
    simp only [hk, ↓reduceDIte]

private theorem reachB_aux_iff (step : Nat → Nat → Bool)
    (hFwd : ∀ a b, step a b = true → a < b) (n m : Nat) (hm : m ≤ n) :
    ∀ i < m, ∀ k,
      reachB (reachTableAux step n m) k i = true ↔
      Relation.TransGen (fun a b => step a b = true) k i := by
  induction m with
  | zero =>
    intro i hi
    omega
  | succ p ih =>
    intro i hi k
    have ihp := ih (by omega)
    by_cases hip : i < p
    · have hrow : (reachTableAux step n (p + 1))[i]? = (reachTableAux step n p)[i]? := by
        rw [reachTableAux_getElem?_lt step n (p + 1) i hi,
            reachTableAux_getElem?_lt step n p i hip]
      unfold reachB
      rw [hrow]
      exact ihp i hip k
    · have hiEq : i = p := by omega
      subst hiEq
      unfold reachB reachTableAux
      dsimp only
      have hsz : (reachTableAux step n i).size = i := reachTableAux_size step n i
      simp only [Array.getElem?_push, hsz, ↓reduceIte]
      rw [ofFn_getD]
      constructor
      · intro hget
        split at hget
        · simp only [Bool.or_eq_true] at hget
          rcases hget with hstep | hany
          · exact Relation.TransGen.single hstep
          · rw [anyLt_iff] at hany
            obtain ⟨j, hj, hcond⟩ := hany
            simp only [Bool.and_eq_true] at hcond
            have hreachJ : reachB (reachTableAux step n i) k j = true := hcond.2
            have htgJ : Relation.TransGen (fun a b => step a b = true) k j :=
              (ihp j hj k).mp hreachJ
            exact Relation.TransGen.tail htgJ hcond.1
        · cases hget
      · intro htg
        have hki : k < i := transGen_lt_of_fwd hFwd htg
        have hkn : k < n := by omega
        simp only [hkn, ↓reduceDIte, Bool.or_eq_true]
        cases htg with
        | single hstep =>
          left; exact hstep
        | tail hkj hji =>
          right
          rw [anyLt_iff]
          have hjiLt : _ < i := hFwd _ _ hji
          refine ⟨_, hjiLt, ?_⟩
          simp only [Bool.and_eq_true]
          exact ⟨hji, (ihp _ hjiLt k).mpr hkj⟩

/-- Soundness and completeness of the forward-edge transitive-closure table `Vane.Reach.reachTable`:
    for any forward relation (`step a b = true → a < b`) and any `i < n`,
    `reachB (reachTable step n) k i = true` iff `Relation.TransGen (fun a b => step a b = true) k i`. -/
theorem reachB_iff (step : Nat → Nat → Bool)
    (hFwd : ∀ a b, step a b = true → a < b) (n k i : Nat) (hi : i < n) :
    reachB (reachTable step n) k i = true ↔
    Relation.TransGen (fun a b => step a b = true) k i :=
  reachB_aux_iff step hFwd n n (Nat.le_refl n) i hi k

-- ============================================================================
-- 10. Milestone 5A spatial H-strip padding & surface row-window safety theorems
-- ============================================================================

/-- For stride-1 3x3 asymmetric vertical strip padding `(pt, pb)` with `pt, pb <= 1`
    and `h + pt + pb >= 3`, `ConvSpec.outH` equals `h + pt + pb - 2 >= 1` and
    `inRows3x3Raw` equals the exact linear tile maximum. -/
theorem conv3x3_strip_outH_and_inRows (mode : Conv3x3Mode) (h w cin cout pt pb th : Nat)
    (hpt : pt ≤ 1) (hpb : pb ≤ 1) (hh : h + pt + pb ≥ 3) (hth : th ≥ 1) :
    let sh : ConvSpec := {
      kernelSize := 3, mode := mode, h := h, w := w, cin := cin, cout := cout,
      stride := 1, pad := .strip pt pb
    }
    sh.outH = h + pt + pb - 2 ∧
    sh.outH ≥ 1 ∧
    inRowsConv sh th = maxTileInRowsLinear h 1 pt sh.outH th (ceilDiv sh.outH th) := by
  intro sh
  have hout : sh.outH = h + pt + pb - 2 := by
    simp only [sh, ConvSpec.outH]
    have hp1 : decide (pt > 1) = false := decide_eq_false (by omega)
    have hp2 : decide (pb > 1) = false := decide_eq_false (by omega)
    have hp3 : decide (h + pt + pb < 3) = false := decide_eq_false (by omega)
    simp only [hp1, hp2, hp3]
    rfl
  have houtPos : sh.outH ≥ 1 := by omega
  have hptEq : sh.padTop = pt := by
    simp [sh, ConvSpec.padTop]
  have hin : inRowsConv sh th = maxTileInRowsLinear h 1 pt sh.outH th (ceilDiv sh.outH th) := by
    simp only [inRowsConv, sh, hptEq]
    exact inRows3x3_eq_max h 1 pt sh.outH th (Or.inl rfl) hpt hth houtPos
  exact ⟨hout, houtPos, hin⟩

/-- Sub-height network surface DMA row-window no-bleed and span-bounds invariant:
    for any channel count `c >= 1`, full surface height `hFull`, row stride `rowStride`,
    and row band `[y0, y0 + h)` with `h >= 1` and `y0 + h <= hFull`,
    setting `baseOff = y0 * rowStride` and `planeStride = hFull * rowStride` satisfies
    both the per-plane no-bleed bound `baseOff + h * rowStride <= planeStride` and the
    exact strided extent bound `baseOff + ((c - 1) * planeStride + h * rowStride) <= c * planeStride`. -/
theorem surface_row_window_no_bleed_and_bounds
    (c hFull y0 h rowStride : Nat)
    (hc : c ≥ 1) (hy : y0 + h ≤ hFull) :
    let planeStride := hFull * rowStride
    let baseOff := y0 * rowStride
    let extent := (c - 1) * planeStride + h * rowStride
    baseOff + h * rowStride ≤ planeStride ∧
    baseOff + extent ≤ c * planeStride := by
  intro planeStride baseOff extent
  have hPlane : baseOff + h * rowStride ≤ planeStride := by
    show y0 * rowStride + h * rowStride ≤ hFull * rowStride
    rw [← Nat.add_mul]
    exact Nat.mul_le_mul_right rowStride hy
  have hSpan : baseOff + extent ≤ c * planeStride := by
    have hcEq : c = (c - 1) + 1 := by omega
    have hmul : c * planeStride = (c - 1) * planeStride + planeStride := by
      conv => lhs; rw [hcEq, Nat.add_mul, Nat.one_mul]
    show baseOff + ((c - 1) * planeStride + h * rowStride) ≤ c * planeStride
    rw [hmul]
    omega
  exact ⟨hPlane, hSpan⟩

-- ============================================================================
-- 11. Milestone 5B pooling & reduction layouts, row bounds, and axis conservation
-- ============================================================================

theorem peReduceOpLayout_valid : (validateWordLayout peReduceOpLayout).isOk = true := by decide

theorem neChanReduceGroupsLayout_valid : (validateWordLayout neChanReduceGroupsLayout).isOk = true := by decide

theorem peReduceOp_roundTrip (reducer preOp postOp : UInt32)
    (hr : reducer < 4) (hpre : preOp < 32) (hpost : postOp < 4) :
    ∃ w, encodeWord peReduceOpLayout #[("reducer", reducer), ("preOp", preOp), ("postOp", postOp)] = .ok w ∧
         decodeFieldChecked peReduceOpLayout w "reducer" = .ok reducer ∧
         decodeFieldChecked peReduceOpLayout w "preOp" = .ok preOp ∧
         decodeFieldChecked peReduceOpLayout w "postOp" = .ok postOp := by
  have h1 : ¬ reducer.toNat ≥ 4 := u32_not_ge_of_lt hr
  have h2 : ¬ preOp.toNat ≥ 32 := u32_not_ge_of_lt hpre
  have h3 : ¬ postOp.toNat ≥ 4 := u32_not_ge_of_lt hpost
  have henc : encodeWord peReduceOpLayout #[("reducer", reducer), ("preOp", preOp), ("postOp", postOp)] =
      .ok ((((0 : UInt32) ||| (reducer <<< 0)) ||| (preOp <<< 4)) ||| (postOp <<< 12)) := by
    change (if reducer.toNat ≥ 4 then _ else if preOp.toNat ≥ 32 then _ else if postOp.toNat ≥ 4 then _ else _) = _
    simp only [h1, h2, h3, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "reducer" reducer henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "preOp" preOp henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "postOp" postOp henc rfl⟩

theorem neChanReduceGroups_roundTrip (chanCount : UInt32) (hc : chanCount < 65536) :
    ∃ w, encodeWord neChanReduceGroupsLayout #[("chanCount", chanCount)] = .ok w ∧
         decodeFieldChecked neChanReduceGroupsLayout w "chanCount" = .ok chanCount := by
  have h1 : ¬ chanCount.toNat ≥ 65536 := u32_not_ge_of_lt hc
  have henc : encodeWord neChanReduceGroupsLayout #[("chanCount", chanCount)] =
      .ok ((0 : UInt32) ||| (chanCount <<< 16)) := by
    change (if chanCount.toNat ≥ 65536 then _ else _) = _
    simp only [h1, ↓reduceIte]
    rfl
  exact ⟨_, henc, encodeWord_decodeFieldChecked _ _ _ "chanCount" chanCount henc rfl⟩

private theorem poolTileRows_le_used (usedH kh sh pt oh th : Nat) (isNe : Bool) (i : Nat) :
    poolTileRows usedH kh sh pt oh th isNe i ≤ usedH := by
  unfold poolTileRows
  dsimp only
  split
  · exact Nat.zero_le usedH
  · exact Nat.le_trans (Nat.sub_le _ _) (Nat.min_le_left _ _)

private theorem poolInRowsAux_le_used (usedH kh sh pt oh th : Nat) (isNe : Bool) (n : Nat) :
    poolInRowsAux usedH kh sh pt oh th isNe n ≤ usedH := by
  induction n with
  | zero => exact Nat.zero_le usedH
  | succ k ih =>
    unfold poolInRowsAux
    have ht := poolTileRows_le_used usedH kh sh pt oh th isNe k
    omega

/-- For any valid row tile height `th >= 1` (indeed any `th`), the maximum input rows buffered
    per tile `poolInRows` never exceeds the active input extent `usedH`. -/
theorem poolInRows_le_used (usedH kh sh pt oh th : Nat) (isNe : Bool) :
    poolInRows usedH kh sh pt oh th isNe ≤ usedH := by
  unfold poolInRows
  split
  · exact Nat.zero_le usedH
  · exact poolInRowsAux_le_used usedH kh sh pt oh th isNe (ceilDiv oh th)

/-- Conservation of `(C, H)` elements across all 4D PE reduction axis mappings:
    `sN * sC * sH * sD = C * H` for every `(rc, rh)` subset. -/
theorem peReduceAxes_product (c h : Nat) (rc rh : Bool) :
    let m := peReduceAxes c h rc rh
    m.sN * m.sC * m.sH * m.sD = c * h := by
  cases rc <;> cases rh <;> unfold peReduceAxes <;> dsimp only
  · split
    · rename_i hc
      have hc1 : c = 1 := by simpa using hc
      subst hc1
      simp
    · simp only [Nat.mul_one]
      exact Nat.mul_comm h c
  · simp
  · simp only [Nat.one_mul, Nat.mul_one]
    exact Nat.mul_comm h c
  · split
    · simp
    · simp only [Nat.one_mul]
      exact Nat.mul_comm h c

-- ============================================================================
-- 12. Milestone 6A declarative TargetProfile universal invariants
-- ============================================================================

theorem target_profile_cpuSubtype_roundtrip (t : Target) :
    Target.ofCpuSubtype? t.profile.cpuSubtype = some t := by
  cases t <;> rfl

theorem target_profile_name_roundtrip (t : Target) :
    Target.parse? t.profile.name = some t := by
  cases t <;> decide

theorem target_profile_defaultPrologue_valid (t : Target) :
    t.profile.isValidPrologue t.profile.defaultPrologue = true := by
  cases t <;> decide

theorem target_profile_neCfgLayout_valid (t : Target) :
    (validateWordLayout (neCfgLayout t)).isOk = true := by
  cases t <;> decide

theorem target_profile_neCfg_sparse_bit_clear (t : Target) :
    (t.profile.neCfgValue &&& ((1 : UInt32) <<< 8)) = 0 := by
  cases t <;> decide

theorem target_profile_dense3x3_ocg_le_four (t : Target) :
    t.profile.dense3x3Stride1MaxOcg ≤ 4 := by
  cases t <;> decide

theorem target_profile_poolMultiStripMinW_pos (t : Target) :
    t.profile.poolMultiStripMinW > 0 := by
  cases t <;> decide

theorem target_profile_header4_disjoint_firstBit (t : Target) :
    (t.profile.taskHeader4Base &&& taskHeader4FirstBit) = 0 ∧
    (t.profile.taskHeader4EdgeBits &&& taskHeader4FirstBit) = 0 := by
  cases t <;> decide

theorem target_profile_isValidPrologue_binary (t : Target) (p : UInt32)
    (h : t.profile.isValidPrologue p = true) : p = 0 ∨ p = 1 := by
  cases t <;> simp [Target.profile, TargetProfile.isValidPrologue] at h
  · right; exact h
  · rcases h with h1 | h0
    · right; exact h1
    · left; exact h0

theorem target_profile_conv1x1MaxCout_pos (t : Target) (f ocg : Nat) :
    t.profile.conv1x1MaxCout f ocg > 0 := by
  unfold TargetProfile.conv1x1MaxCout
  rw [Nat.shiftLeft_eq, Nat.one_mul]
  exact Nat.two_pow_pos _

-- ============================================================================
-- 13. Milestone 5C view layouts, groupCode bounds, prefix partitions & S5 coverage
-- ============================================================================

theorem neViewCommonLayout_valid : (validateWordLayout neViewCommonLayout).isOk = true := by decide

theorem neViewCommon_roundTrip
    (kind shPref shMin shMax f transposeOut transposeWideGroup : UInt32)
    (hk : kind < 256) (hsp : shPref < 8) (hsm : shMin < 8) (hsx : shMax < 8)
    (hf : f < 8) (ht : transposeOut < 2) (hwg : transposeWideGroup < 2) :
    ∃ w, encodeWord neViewCommonLayout #[
        ("kind", kind), ("shPref", shPref), ("shMin", shMin), ("shMax", shMax),
        ("F", f), ("transposeOut", transposeOut), ("transposeWideGroup", transposeWideGroup)
      ] = .ok w ∧
      decodeFieldChecked neViewCommonLayout w "kind" = .ok kind ∧
      decodeFieldChecked neViewCommonLayout w "shPref" = .ok shPref ∧
      decodeFieldChecked neViewCommonLayout w "shMin" = .ok shMin ∧
      decodeFieldChecked neViewCommonLayout w "shMax" = .ok shMax ∧
      decodeFieldChecked neViewCommonLayout w "F" = .ok f ∧
      decodeFieldChecked neViewCommonLayout w "transposeOut" = .ok transposeOut ∧
      decodeFieldChecked neViewCommonLayout w "transposeWideGroup" = .ok transposeWideGroup := by
  have h1 : ¬ kind.toNat ≥ 256 := u32_not_ge_of_lt hk
  have h2 : ¬ shPref.toNat ≥ 8 := u32_not_ge_of_lt hsp
  have h3 : ¬ shMin.toNat ≥ 8 := u32_not_ge_of_lt hsm
  have h4 : ¬ shMax.toNat ≥ 8 := u32_not_ge_of_lt hsx
  have h5 : ¬ f.toNat ≥ 8 := u32_not_ge_of_lt hf
  have h6 : ¬ transposeOut.toNat ≥ 2 := u32_not_ge_of_lt ht
  have h7 : ¬ transposeWideGroup.toNat ≥ 2 := u32_not_ge_of_lt hwg
  have henc : encodeWord neViewCommonLayout #[
      ("kind", kind), ("shPref", shPref), ("shMin", shMin), ("shMax", shMax),
      ("F", f), ("transposeOut", transposeOut), ("transposeWideGroup", transposeWideGroup)
    ] = .ok ((((((((0 : UInt32) ||| (kind <<< 0)) ||| (shPref <<< 8)) ||| (shMin <<< 12)) |||
        (shMax <<< 16)) ||| (f <<< 19)) ||| (transposeOut <<< 28)) ||| (transposeWideGroup <<< 29)) := by
    change (if kind.toNat ≥ 256 then _ else if shPref.toNat ≥ 8 then _ else
            if shMin.toNat ≥ 8 then _ else if shMax.toNat ≥ 8 then _ else
            if f.toNat ≥ 8 then _ else if transposeOut.toNat ≥ 2 then _ else
            if transposeWideGroup.toNat ≥ 2 then _ else _) = _
    simp only [h1, h2, h3, h4, h5, h6, h7, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "kind" kind henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "shPref" shPref henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "shMin" shMin henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "shMax" shMax henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "F" f henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "transposeOut" transposeOut henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "transposeWideGroup" transposeWideGroup henc rfl⟩

/-- `groupCode c` is always at least `1`. -/
theorem groupCode_pos (c : Nat) : 1 ≤ groupCode c := by
  unfold groupCode
  split
  · omega
  · split
    · omega
    · split <;> omega

/-- `groupCode c` is always at most `8`. -/
theorem groupCode_le_eight (c : Nat) : groupCode c ≤ 8 := by
  unfold groupCode
  split
  · omega
  · split
    · omega
    · split <;> omega

/-- `groupCodeLog2 c` always fits in the 2-bit range `0..3`. -/
theorem groupCodeLog2_le_three (c : Nat) : groupCodeLog2 c ≤ 3 := by
  unfold groupCodeLog2
  split
  · omega
  · split
    · omega
    · split <;> omega

/-- `groupCode c` is the exact power of two `1 <<< groupCodeLog2 c`. -/
theorem groupCode_eq_pow (c : Nat) : groupCode c = 1 <<< groupCodeLog2 c := by
  unfold groupCode groupCodeLog2
  split
  · rfl
  · split
    · rfl
    · split <;> rfl

/-- `prefixStartsAux` preserves list length. -/
theorem prefixStartsAux_length (acc : Nat) (sizes : List Nat) :
    (prefixStartsAux acc sizes).length = sizes.length := by
  induction sizes generalizing acc with
  | nil => rfl
  | cons sz rest ih =>
    simp [prefixStartsAux, ih]

/-- `prefixStarts` preserves list length. -/
theorem prefixStarts_length (sizes : List Nat) :
    (prefixStarts sizes).length = sizes.length :=
  prefixStartsAux_length 0 sizes

/-- Every interval `(a, b)` in `sliceIntervalsAux acc sizes` lies within `[acc, acc + sumSizes sizes]`. -/
theorem sliceIntervalsAux_bounds (acc : Nat) (sizes : List Nat) :
    ∀ iv ∈ sliceIntervalsAux acc sizes, acc ≤ iv.1 ∧ iv.1 ≤ iv.2 ∧ iv.2 ≤ acc + sumSizes sizes := by
  induction sizes generalizing acc with
  | nil =>
    intro iv hiv
    simp [sliceIntervalsAux] at hiv
  | cons sz rest ih =>
    intro iv hiv
    simp only [sliceIntervalsAux, List.mem_cons] at hiv
    rcases hiv with rfl | hmem
    · simp only [sumSizes]
      omega
    · have h := ih (acc + sz) iv hmem
      simp only [sumSizes]
      omega

/-- Every interval `(a, b)` in `sliceIntervals sizes` lies within `[0, sumSizes sizes]`. -/
theorem sliceIntervals_bounds (sizes : List Nat) :
    ∀ iv ∈ sliceIntervals sizes, iv.1 ≤ iv.2 ∧ iv.2 ≤ sumSizes sizes := by
  intro iv hiv
  have h := sliceIntervalsAux_bounds 0 sizes iv hiv
  omega

/-- Intervals generated by `sliceIntervalsAux` are pairwise strictly non-overlapping (`a.2 ≤ b.1`). -/
theorem sliceIntervalsAux_pairwise_disjoint (acc : Nat) (sizes : List Nat) :
    (sliceIntervalsAux acc sizes).Pairwise (fun a b => a.2 ≤ b.1) := by
  induction sizes generalizing acc with
  | nil =>
    exact List.Pairwise.nil
  | cons sz rest ih =>
    simp only [sliceIntervalsAux]
    apply List.Pairwise.cons
    · intro iv hiv
      exact (sliceIntervalsAux_bounds (acc + sz) rest iv hiv).1
    · exact ih (acc + sz)

/-- Intervals generated by `sliceIntervals` are pairwise non-overlapping (`a.2 ≤ b.1`). -/
theorem sliceIntervals_pairwise_disjoint (sizes : List Nat) :
    (sliceIntervals sizes).Pairwise (fun a b => a.2 ≤ b.1) :=
  sliceIntervalsAux_pairwise_disjoint 0 sizes

private theorem insertInterval_head_of_le (a : Nat × Nat) (xs : List (Nat × Nat))
    (hle : ∀ b ∈ xs, a.1 ≤ b.1) :
    insertInterval a xs = a :: xs := by
  cases xs with
  | nil => rfl
  | cons hd tl =>
    unfold insertInterval
    have h1 : a.1 ≤ hd.1 := hle hd List.mem_cons_self
    simp [h1]

private theorem sortIntervals_sliceIntervalsAux (acc : Nat) (sizes : List Nat) :
    sortIntervals (sliceIntervalsAux acc sizes) = sliceIntervalsAux acc sizes := by
  induction sizes generalizing acc with
  | nil => rfl
  | cons sz rest ih =>
    simp only [sliceIntervalsAux, sortIntervals]
    rw [ih (acc + sz)]
    apply insertInterval_head_of_le
    intro iv hiv
    have h := (sliceIntervalsAux_bounds (acc + sz) rest iv hiv).1
    omega

private theorem sweepCover_sliceIntervalsAux (acc : Nat) (sizes : List Nat) :
    sweepCover (sliceIntervalsAux acc sizes) acc = acc + sumSizes sizes := by
  induction sizes generalizing acc with
  | nil =>
    simp [sliceIntervalsAux, sweepCover, sumSizes]
  | cons sz rest ih =>
    simp only [sliceIntervalsAux, sweepCover, sumSizes, Nat.le_refl, ↓reduceIte]
    have hmax : max acc (acc + sz) = acc + sz := by omega
    rw [hmax, ih (acc + sz)]
    omega

/-- Contiguous `sliceIntervals sizes` completely cover `[0, sumSizes sizes)` under `Vane.Verify.intervalsCover`. -/
theorem sliceIntervals_cover (sizes : List Nat) :
    intervalsCover (sliceIntervals sizes) 0 (sumSizes sizes) = true := by
  unfold intervalsCover sliceIntervals
  split
  · rfl
  · rw [sortIntervals_sliceIntervalsAux 0 sizes, sweepCover_sliceIntervalsAux 0 sizes]
    simp

/-- Membership characterization for `Vane.Verify.insertInterval`. -/
theorem mem_insertInterval (iv x : Nat × Nat) (xs : List (Nat × Nat)) :
    x ∈ insertInterval iv xs ↔ x = iv ∨ x ∈ xs := by
  induction xs with
  | nil =>
    simp [insertInterval]
  | cons hd tl ih =>
    unfold insertInterval
    split
    · simp [List.mem_cons]
    · simp only [List.mem_cons, ih]
      exact ⟨fun h => by rcases h with rfl | rfl | hm <;> simp [*],
             fun h => by rcases h with rfl | rfl | hm <;> simp [*]⟩

/-- Membership preservation for `Vane.Verify.sortIntervals`. -/
theorem mem_sortIntervals (x : Nat × Nat) (xs : List (Nat × Nat)) :
    x ∈ sortIntervals xs ↔ x ∈ xs := by
  induction xs with
  | nil => simp [sortIntervals]
  | cons hd tl ih =>
    simp only [sortIntervals, mem_insertInterval, ih, List.mem_cons]

/-- Soundness of `Vane.Verify.sweepCover`: every byte offset `x ∈ [cur, sweepCover ivs cur)` is
    covered by at least one interval in `ivs`. -/
theorem sweepCover_sound (ivs : List (Nat × Nat)) (cur x : Nat)
    (hcur : cur ≤ x) (hlt : x < sweepCover ivs cur) :
    ∃ iv ∈ ivs, iv.1 ≤ x ∧ x < iv.2 := by
  induction ivs generalizing cur with
  | nil =>
    simp [sweepCover] at hlt
    omega
  | cons iv rest ih =>
    rcases iv with ⟨a, b⟩
    simp only [sweepCover] at hlt
    split at hlt
    · rename_i hac
      by_cases hxb : x < b
      · exact ⟨(a, b), List.mem_cons_self, by omega, hxb⟩
      · have hmax : max cur b ≤ x := by omega
        obtain ⟨w, hwMem, hwLo, hwHi⟩ := ih (max cur b) hmax hlt
        exact ⟨w, List.mem_cons_of_mem _ hwMem, hwLo, hwHi⟩
    · omega

/-- Soundness of `Vane.Verify.intervalsCover`: if `intervalsCover ivs lo hi = true`, then every
    byte offset `x ∈ [lo, hi)` is written by at least one interval `(a, b) ∈ ivs`. -/
theorem intervalsCover_sound (ivs : List (Nat × Nat)) (lo hi x : Nat)
    (hcov : intervalsCover ivs lo hi = true) (hlo : lo ≤ x) (hhi : x < hi) :
    ∃ iv ∈ ivs, iv.1 ≤ x ∧ x < iv.2 := by
  unfold intervalsCover at hcov
  have hnot : ¬ (lo ≥ hi) := by omega
  simp only [hnot, ↓reduceIte, decide_eq_true_eq] at hcov
  have hlt : x < sweepCover (sortIntervals ivs) lo := by omega
  obtain ⟨iv, hivMem, hivLo, hivHi⟩ := sweepCover_sound (sortIntervals ivs) lo x hlo hlt
  exact ⟨iv, (mem_sortIntervals iv ivs).mp hivMem, hivLo, hivHi⟩

/-- Every element index `x < sumSizes sizes` belongs to a slice interval in `sliceIntervals sizes`. -/
theorem sliceIntervals_cover_point (sizes : List Nat) (x : Nat) (hx : x < sumSizes sizes) :
    ∃ iv ∈ sliceIntervals sizes, iv.1 ≤ x ∧ x < iv.2 :=
  intervalsCover_sound (sliceIntervals sizes) 0 (sumSizes sizes) x
    (sliceIntervals_cover sizes) (Nat.zero_le x) hx

end Vane.Proofs
