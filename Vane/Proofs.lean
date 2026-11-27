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
import Vane.Matmul
import Vane.Conv
import Vane.Conv3x3
import Vane.Frontend.SafeTensors
import Vane.Frontend.Lower
import Vane.Quant
import Vane.IR.Ops
import Vane.IR.Graph
import Vane.IR.Validate
import Vane.IR.Semantics
import Vane.IR.Rewrite
import Vane.IR.Passes.DCE
import Vane.IR.Passes.ViewSimplify
import Vane.IR.Pipeline

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
open Vane.Matmul
open Vane.Conv
open Vane.Conv3x3
open Vane.Frontend.SafeTensors
open Vane.Frontend.Lower
open Vane.Quant
open Vane.IR

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

/-- Single-task T-mode channel-group coverage bound: for all `c ≤ 128`,
    `F = 4` with `ocg = groupCodeLog2 c` covers all `c` channels (`c ≤ 2 ^ (4 + groupCodeLog2 c)`). -/
theorem tmode_channel_coverage_le_128 (c : Nat) (hc : c ≤ 128) :
    c ≤ 2 ^ (4 + groupCodeLog2 c) := by
  unfold groupCodeLog2
  split
  · omega
  · split
    · omega
    · split <;> omega

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

/-- Soundness of periodic multi-writer coverage (`Vane.Verify.periodicGroupCovered`):
    if `periodicGroupCovered outerBase outerCount outerStride innerIvs = some (lo, hi)`,
    then every byte offset `x ∈ [lo, hi)` falls inside `[outerBase + j * outerStride + iv.1,
    outerBase + j * outerStride + iv.2)` for some outer repetition `j < outerCount` and
    some inner slice `iv ∈ innerIvs`. -/
theorem periodicGroupCovered_sound
    (outerBase outerCount outerStride : Nat) (innerIvs : List (Nat × Nat))
    (lo hi x : Nat)
    (h : periodicGroupCovered outerBase outerCount outerStride innerIvs = some (lo, hi))
    (hlo : lo ≤ x) (hhi : x < hi) :
    ∃ j < outerCount, ∃ iv ∈ innerIvs,
      outerBase + j * outerStride + iv.1 ≤ x ∧
      x < outerBase + j * outerStride + iv.2 := by
  unfold periodicGroupCovered at h
  split at h
  · rename_i hcond
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    rcases h with ⟨rfl, rfl⟩
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hcond
    rcases hcond with ⟨⟨_, hStridePos⟩, hcov⟩
    let d := x - outerBase
    have hdLt : d < outerCount * outerStride := by omega
    have hjLt : d / outerStride < outerCount := by
      have hcomm : d < outerStride * outerCount := by rw [Nat.mul_comm]; exact hdLt
      exact Nat.div_lt_of_lt_mul hcomm
    have hrLt : d % outerStride < outerStride := Nat.mod_lt d hStridePos
    have hdivMod : outerStride * (d / outerStride) + d % outerStride = d := Nat.div_add_mod d outerStride
    have hmulComm : (d / outerStride) * outerStride = outerStride * (d / outerStride) := Nat.mul_comm _ _
    obtain ⟨iv, hivMem, hivLo, hivHi⟩ :=
      intervalsCover_sound innerIvs 0 outerStride (d % outerStride) hcov (Nat.zero_le _) hrLt
    refine ⟨d / outerStride, hjLt, iv, hivMem, ?_, ?_⟩ <;> omega
  · cases h

-- ============================================================================
-- 15. Milestone 5D: Dynamic MatMul Bitfields, Multi-Head Stride Identity,
--     and Kernel Slot Bounds
-- ============================================================================

theorem taskHeader8ExtLayout_valid : (validateWordLayout taskHeader8ExtLayout).isOk = true := by decide
theorem neMatmulCommonLayout_valid : (validateWordLayout neMatmulCommonLayout).isOk = true := by decide
theorem neMatmulCfgLayout_h16_valid : (validateWordLayout (neMatmulCfgLayout .h16)).isOk = true := by decide
theorem neMatmulCfgLayout_h17_valid : (validateWordLayout (neMatmulCfgLayout .h17)).isOk = true := by decide
theorem kernelStageDynLayout_valid : (validateWordLayout kernelStageDynLayout).isOk = true := by decide
theorem chunkSlotDynLayout_valid : (validateWordLayout chunkSlotDynLayout).isOk = true := by decide

theorem taskHeader8Ext_roundTrip (active hasExtHdr isLast ene : UInt32)
    (ha : active < 2) (hext : hasExtHdr < 2) (hl : isLast < 2) (he : ene < 8) :
    ∃ w, encodeWord taskHeader8ExtLayout #[
            ("active", active), ("hasExtHdr", hasExtHdr), ("isLast", isLast), ("ene", ene)
          ] = .ok w ∧
         decodeFieldChecked taskHeader8ExtLayout w "active" = .ok active ∧
         decodeFieldChecked taskHeader8ExtLayout w "hasExtHdr" = .ok hasExtHdr ∧
         decodeFieldChecked taskHeader8ExtLayout w "isLast" = .ok isLast ∧
         decodeFieldChecked taskHeader8ExtLayout w "ene" = .ok ene := by
  have h1 : ¬ active.toNat ≥ 2 := u32_not_ge_of_lt ha
  have h2 : ¬ hasExtHdr.toNat ≥ 2 := u32_not_ge_of_lt hext
  have h3 : ¬ isLast.toNat ≥ 2 := u32_not_ge_of_lt hl
  have h4 : ¬ ene.toNat ≥ 8 := u32_not_ge_of_lt he
  have henc : encodeWord taskHeader8ExtLayout #[
      ("active", active), ("hasExtHdr", hasExtHdr), ("isLast", isLast), ("ene", ene)
    ] = .ok (((((0 : UInt32) ||| (active <<< 0)) ||| (hasExtHdr <<< 1)) ||| (isLast <<< 3)) ||| (ene <<< 16)) := by
    change (if active.toNat ≥ 2 then _ else if hasExtHdr.toNat ≥ 2 then _ else
            if isLast.toNat ≥ 2 then _ else if ene.toNat ≥ 8 then _ else _) = _
    simp only [h1, h2, h3, h4, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "active" active henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "hasExtHdr" hasExtHdr henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "isLast" isLast henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "ene" ene henc rfl⟩

/-- Any extended 10-word task header (`active = 1`, `hasExtHdr = 1`) satisfies `(w &&& 3) = 3`,
    so `encodeTaskStream`, `validateTaskStream`, and `decodeTaskStream` always classify it as 10 words. -/
theorem taskHeader8Ext_is_10word (isLast ene : UInt32)
    (hl : isLast < 2) (he : ene < 8) :
    ∃ w, encodeWord taskHeader8ExtLayout #[
            ("active", 1), ("hasExtHdr", 1), ("isLast", isLast), ("ene", ene)
          ] = .ok w ∧
         (w &&& 3) = 3 := by
  have h3 : ¬ isLast.toNat ≥ 2 := u32_not_ge_of_lt hl
  have h4 : ¬ ene.toNat ≥ 8 := u32_not_ge_of_lt he
  have henc : encodeWord taskHeader8ExtLayout #[
      ("active", 1), ("hasExtHdr", 1), ("isLast", isLast), ("ene", ene)
    ] = .ok (((((0 : UInt32) ||| ((1 : UInt32) <<< 0)) ||| ((1 : UInt32) <<< 1)) |||
              (isLast <<< 3)) ||| (ene <<< 16)) := by
    change (if isLast.toNat ≥ 2 then _ else if ene.toNat ≥ 8 then _ else _) = _
    simp only [h3, h4, ↓reduceIte]
    rfl
  refine ⟨_, henc, ?_⟩
  apply u32_eq_of_bit_eq
  intro i hi
  rw [u32_bit_and _ _ i hi, u32_bit_or _ _ i hi, u32_bit_or _ _ i hi,
      u32_bit_or _ _ i hi, u32_bit_or _ _ i hi]
  by_cases hi2 : i < 2
  · have hcase : i = 0 ∨ i = 1 := by omega
    rcases hcase with rfl | rfl
    · simp
    · simp
  · have h3_i : (3#32 : BitVec 32)[i] = false := by
      rw [BitVec.getElem_eq_testBit_toNat]
      change (3 : Nat).testBit i = false
      exact Nat.testBit_lt_two_pow (by
        have hpow : 4 ≤ 2 ^ i := by
          have : 2 ^ 2 ≤ 2 ^ i := Nat.pow_le_pow_right (by omega) (by omega)
          simpa using this
        omega)
    simp [h3_i]

theorem neMatmulCommon_roundTrip (lo2 shPref shMin shMax f dynKern : UInt32)
    (hlo : lo2 < 2) (hsp : shPref < 8) (hsm : shMin < 8) (hsx : shMax < 8) (hf : f < 8) (hd : dynKern < 2) :
    ∃ w, encodeWord neMatmulCommonLayout #[
            ("lo2", lo2), ("shPref", shPref), ("shMin", shMin),
            ("shMax", shMax), ("F", f), ("dynKern", dynKern)
          ] = .ok w ∧
         decodeFieldChecked neMatmulCommonLayout w "lo2" = .ok lo2 ∧
         decodeFieldChecked neMatmulCommonLayout w "shPref" = .ok shPref ∧
         decodeFieldChecked neMatmulCommonLayout w "shMin" = .ok shMin ∧
         decodeFieldChecked neMatmulCommonLayout w "shMax" = .ok shMax ∧
         decodeFieldChecked neMatmulCommonLayout w "F" = .ok f ∧
         decodeFieldChecked neMatmulCommonLayout w "dynKern" = .ok dynKern := by
  have h1 : ¬ lo2.toNat ≥ 2 := u32_not_ge_of_lt hlo
  have h2 : ¬ shPref.toNat ≥ 8 := u32_not_ge_of_lt hsp
  have h3 : ¬ shMin.toNat ≥ 8 := u32_not_ge_of_lt hsm
  have h4 : ¬ shMax.toNat ≥ 8 := u32_not_ge_of_lt hsx
  have h5 : ¬ f.toNat ≥ 8 := u32_not_ge_of_lt hf
  have h6 : ¬ dynKern.toNat ≥ 2 := u32_not_ge_of_lt hd
  have henc : encodeWord neMatmulCommonLayout #[
      ("lo2", lo2), ("shPref", shPref), ("shMin", shMin),
      ("shMax", shMax), ("F", f), ("dynKern", dynKern)
    ] = .ok (((((((0 : UInt32) ||| (lo2 <<< 2)) ||| (shPref <<< 8)) ||| (shMin <<< 12)) |||
               (shMax <<< 16)) ||| (f <<< 19)) ||| (dynKern <<< 29)) := by
    change (if lo2.toNat ≥ 2 then _ else if shPref.toNat ≥ 8 then _ else
            if shMin.toNat ≥ 8 then _ else if shMax.toNat ≥ 8 then _ else
            if f.toNat ≥ 8 then _ else if dynKern.toNat ≥ 2 then _ else _) = _
    simp only [h1, h2, h3, h4, h5, h6, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "lo2" lo2 henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "shPref" shPref henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "shMin" shMin henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "shMax" shMax henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "F" f henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "dynKern" dynKern henc rfl⟩

theorem neMatmulCfg_roundTrip (target : Target) (dynKern : UInt32) (hd : dynKern < 2) :
    ∃ w, encodeWord (neMatmulCfgLayout target) #[("dynKern", dynKern)] = .ok w ∧
         decodeFieldChecked (neMatmulCfgLayout target) w "dynKern" = .ok dynKern := by
  have h1 : ¬ dynKern.toNat ≥ 2 := u32_not_ge_of_lt hd
  cases target with
  | h16 =>
    have henc : encodeWord (neMatmulCfgLayout .h16) #[("dynKern", dynKern)] =
        .ok ((neCfgValue .h16) ||| (dynKern <<< 16)) := by
      change (if dynKern.toNat ≥ 2 then _ else _) = _
      simp only [h1, ↓reduceIte]
      rfl
    exact ⟨_, henc, encodeWord_decodeFieldChecked _ _ _ "dynKern" dynKern henc rfl⟩
  | h17 =>
    have henc : encodeWord (neMatmulCfgLayout .h17) #[("dynKern", dynKern)] =
        .ok ((neCfgValue .h17) ||| (dynKern <<< 16)) := by
      change (if dynKern.toNat ≥ 2 then _ else _) = _
      simp only [h1, ↓reduceIte]
      rfl
    exact ⟨_, henc, encodeWord_decodeFieldChecked _ _ _ "dynKern" dynKern henc rfl⟩

theorem kernelStageDyn_roundTrip (sparse kernEn isLast kernCfg h17Dyn stage : UInt32)
    (hsp : sparse < 2) (hke : kernEn < 2) (hil : isLast < 2)
    (hkc : kernCfg < 2) (hd : h17Dyn < 2) (hst : stage < 256) :
    ∃ w, encodeWord kernelStageDynLayout #[
            ("sparse", sparse), ("kernEn", kernEn), ("isLast", isLast),
            ("kernCfg", kernCfg), ("h17Dyn", h17Dyn), ("stage", stage)
          ] = .ok w ∧
         decodeFieldChecked kernelStageDynLayout w "sparse" = .ok sparse ∧
         decodeFieldChecked kernelStageDynLayout w "kernEn" = .ok kernEn ∧
         decodeFieldChecked kernelStageDynLayout w "isLast" = .ok isLast ∧
         decodeFieldChecked kernelStageDynLayout w "kernCfg" = .ok kernCfg ∧
         decodeFieldChecked kernelStageDynLayout w "h17Dyn" = .ok h17Dyn ∧
         decodeFieldChecked kernelStageDynLayout w "stage" = .ok stage := by
  have h1 : ¬ sparse.toNat ≥ 2 := u32_not_ge_of_lt hsp
  have h2 : ¬ kernEn.toNat ≥ 2 := u32_not_ge_of_lt hke
  have h3 : ¬ isLast.toNat ≥ 2 := u32_not_ge_of_lt hil
  have h4 : ¬ kernCfg.toNat ≥ 2 := u32_not_ge_of_lt hkc
  have h5 : ¬ h17Dyn.toNat ≥ 2 := u32_not_ge_of_lt hd
  have h6 : ¬ stage.toNat ≥ 256 := u32_not_ge_of_lt hst
  have henc : encodeWord kernelStageDynLayout #[
      ("sparse", sparse), ("kernEn", kernEn), ("isLast", isLast),
      ("kernCfg", kernCfg), ("h17Dyn", h17Dyn), ("stage", stage)
    ] = .ok (((((((0 : UInt32) ||| (sparse <<< 5)) ||| (kernEn <<< 6)) ||| (isLast <<< 7)) |||
               (kernCfg <<< 9)) ||| (h17Dyn <<< 10)) ||| (stage <<< 16)) := by
    change (if sparse.toNat ≥ 2 then _ else if kernEn.toNat ≥ 2 then _ else
            if isLast.toNat ≥ 2 then _ else if kernCfg.toNat ≥ 2 then _ else
            if h17Dyn.toNat ≥ 2 then _ else if stage.toNat ≥ 256 then _ else _) = _
    simp only [h1, h2, h3, h4, h5, h6, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "sparse" sparse henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "kernEn" kernEn henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "isLast" isLast henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "kernCfg" kernCfg henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "h17Dyn" h17Dyn henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "stage" stage henc rfl⟩

theorem chunkSlotDyn_roundTrip (en on20 prod keepLive task : UInt32)
    (he : en < 2) (ho : on20 < 2) (hp : prod < 2) (hk : keepLive < 2) (ht : task < 256) :
    ∃ w, encodeWord chunkSlotDynLayout #[
            ("enable", en), ("on20", on20), ("produced", prod), ("keepLive", keepLive), ("task", task)
          ] = .ok w ∧
         decodeFieldChecked chunkSlotDynLayout w "enable" = .ok en ∧
         decodeFieldChecked chunkSlotDynLayout w "on20" = .ok on20 ∧
         decodeFieldChecked chunkSlotDynLayout w "produced" = .ok prod ∧
         decodeFieldChecked chunkSlotDynLayout w "keepLive" = .ok keepLive ∧
         decodeFieldChecked chunkSlotDynLayout w "task" = .ok task := by
  have h1 : ¬ en.toNat ≥ 2 := u32_not_ge_of_lt he
  have h2 : ¬ on20.toNat ≥ 2 := u32_not_ge_of_lt ho
  have h3 : ¬ prod.toNat ≥ 2 := u32_not_ge_of_lt hp
  have h4 : ¬ keepLive.toNat ≥ 2 := u32_not_ge_of_lt hk
  have h5 : ¬ task.toNat ≥ 256 := u32_not_ge_of_lt ht
  have henc : encodeWord chunkSlotDynLayout #[
      ("enable", en), ("on20", on20), ("produced", prod), ("keepLive", keepLive), ("task", task)
    ] = .ok ((((((0 : UInt32) ||| (en <<< 0)) ||| (on20 <<< 5)) ||| (prod <<< 6)) |||
              (keepLive <<< 7)) ||| (task <<< 16)) := by
    change (if en.toNat ≥ 2 then _ else if on20.toNat ≥ 2 then _ else
            if prod.toNat ≥ 2 then _ else if keepLive.toNat ≥ 2 then _ else
            if task.toNat ≥ 256 then _ else _) = _
    simp only [h1, h2, h3, h4, h5, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "enable" en henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "on20" on20 henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "produced" prod henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "keepLive" keepLive henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "task" task henc rfl⟩

/-- Exact byte-address identity between graph-IR stacked multi-head layout `[1, heads * M, 1, K]`
    and hardware batched layout `[heads, M, 1, K]`: element `(h, m, k)` lies at the exact same
    DRAM byte offset in both layouts under 64-byte row alignment. -/
theorem multiHead_stacked_offset_eq (m k h i j : Nat) :
    let rowStride := alignUp (2 * k) 64
    (h * m + i) * rowStride + 2 * j = h * (m * rowStride) + i * rowStride + 2 * j := by
  intro rowStride
  rw [Nat.add_mul, Nat.mul_assoc]

/-- Exact total DRAM span identity between `[1, heads * M, 1, K]` and `[heads, M, 1, K]`. -/
theorem multiHead_stacked_span_eq (heads m k : Nat) :
    Shape4D.span { n := 1, c := heads * m, h := 1, w := k } =
    Shape4D.span { n := heads, c := m, h := 1, w := k } := by
  simp [Shape4D.span, Shape4D.batchStride, Shape4D.planeStride, Shape4D.rowStride, Nat.mul_assoc]

/-- Every active dynamic-kernel chunk slot `s < min 16 m` fits inside one per-batch kernel slice
    `[0, m * chunkSize)`. -/
theorem dynKernelSlot_end_le_round (s m chunkSize : Nat) (hs : s < min 16 m) :
    s * chunkSize + chunkSize ≤ m * chunkSize := by
  have hsm : s + 1 ≤ m := by omega
  have hmul : (s + 1) * chunkSize ≤ m * chunkSize := Nat.mul_le_mul_right chunkSize hsm
  simpa [Nat.add_mul, Nat.one_mul] using hmul

/-- Every dynamic-kernel chunk slot offset `s * matmulChunkSize k` is 64-byte aligned. -/
theorem dynKernelSlot_aligned (s k : Nat) :
    (s * matmulChunkSize k) % 64 = 0 := by
  have hcs : matmulChunkSize k % 64 = 0 := alignUp_dvd (2 * k) 64 (by omega)
  rw [Nat.mul_mod, hcs, Nat.mul_zero, Nat.zero_mod]

/-- The dynamic kernel round stride `(if m > 1 then m else b) * chunkSize` never exceeds
    the total runtime kernel span `b * m * chunkSize`. -/
theorem dynKernelRound_le_total (b m chunkSize : Nat) (hb : 1 ≤ b) (hm : 1 ≤ m) :
    (if m > 1 then m else b) * chunkSize ≤ b * m * chunkSize := by
  apply Nat.mul_le_mul_right
  split
  · have h1 : 1 * m ≤ b * m := Nat.mul_le_mul_right m hb
    simpa using h1
  · have h1 : b * 1 ≤ b * m := Nat.mul_le_mul_left b hm
    simpa using h1

/-- Every batch slice `h < b` of size `m * chunkSize` fits inside the total kernel span `b * m * chunkSize`. -/
theorem dynKernelBatch_end_le_total (h b m chunkSize : Nat) (hh : h < b) :
    h * (m * chunkSize) + m * chunkSize ≤ b * (m * chunkSize) := by
  have hhb : h + 1 ≤ b := by omega
  have hmul : (h + 1) * (m * chunkSize) ≤ b * (m * chunkSize) := Nat.mul_le_mul_right _ hhb
  simpa [Nat.add_mul, Nat.one_mul] using hmul

/-- Row-slicing `M` into chunks of at most 16 rows produces valid slice bounds
    `1 ≤ Mi ≤ 16` and `m0 + Mi ≤ M` for every slice index `si < ceilDiv M 16`. -/
theorem matmulRowSlice_partition_bounds (m si : Nat) (hsi : si < ceilDiv m 16) :
    let m0 := si * 16
    let mi := min 16 (m - m0)
    1 ≤ mi ∧ mi ≤ 16 ∧ m0 + mi ≤ m := by
  dsimp only
  simp [ceilDiv] at hsi
  omega

/-- Every per-batch interval `[bIdx * (totalM * row) + m0 * row, bIdx * (totalM * row) + (m0 + mi) * row)`
    of a row slice `(m0, mi)` with `m0 + mi ≤ totalM` lies within `[0, b * (totalM * row))`. -/
theorem matmulRowSlice_end_le_total (bIdx b m0 mi totalM row : Nat)
    (hb : bIdx < b) (hslice : m0 + mi ≤ totalM) :
    bIdx * (totalM * row) + m0 * row + mi * row ≤ b * (totalM * row) := by
  have h1 : m0 * row + mi * row ≤ totalM * row := by
    rw [← Nat.add_mul]
    exact Nat.mul_le_mul_right row hslice
  have h2 : bIdx * (totalM * row) + totalM * row ≤ b * (totalM * row) :=
    dynKernelBatch_end_le_total bIdx b totalM row hb
  omega

/-- Any two disjoint row slices `(m0a, ma)` and `(m0b, mb)` (`m0a + ma ≤ m0b`, `m0b + mb ≤ totalM`)
    have pairwise disjoint per-batch byte intervals across all batch pairs `(ba, bb)`. -/
theorem matmulRowSlice_disjoint (ba bb m0a ma m0b mb totalM row : Nat)
    (hab : m0a + ma ≤ m0b) (hb : m0b + mb ≤ totalM) :
    let loA := ba * (totalM * row) + m0a * row
    let hiA := loA + ma * row
    let loB := bb * (totalM * row) + m0b * row
    let hiB := loB + mb * row
    hiA ≤ loB ∨ hiB ≤ loA := by
  dsimp only
  by_cases hlt : ba < bb
  · left
    have h1 : ba * (totalM * row) + m0a * row + ma * row ≤ bb * (totalM * row) :=
      matmulRowSlice_end_le_total ba bb m0a ma totalM row hlt (by omega)
    omega
  · by_cases heq : ba = bb
    · left
      subst heq
      have h1 : m0a * row + ma * row ≤ m0b * row := by
        rw [← Nat.add_mul]
        exact Nat.mul_le_mul_right row hab
      omega
    · right
      have hgt : bb < ba := by omega
      have h1 : bb * (totalM * row) + m0b * row + mb * row ≤ ba * (totalM * row) :=
        matmulRowSlice_end_le_total bb ba m0b mb totalM row hgt hb
      omega

-- ============================================================================
-- 16. Binary Codec & Little-Endian Round-Trip Soundness (Vane.Binary)
-- ============================================================================

private theorem byteArray_get!_eq_getElem {bs : ByteArray} {i : Nat} (hi : i < bs.size) :
    bs.get! i = bs.data[i] := by
  cases bs with
  | mk data =>
    change data[i]! = data[i]
    exact getElem!_pos data i hi

private theorem byteArray_get!_push_lt (bs : ByteArray) (b : UInt8) (i : Nat) (hi : i < bs.size) :
    (bs.push b).get! i = bs.get! i := by
  have hi' : i < (bs.push b).size := by rw [ByteArray.size_push]; omega
  rw [byteArray_get!_eq_getElem hi', byteArray_get!_eq_getElem hi]
  cases bs with
  | mk data =>
    exact Array.getElem_push_lt hi

private theorem byteArray_get!_push_eq (bs : ByteArray) (b : UInt8) :
    (bs.push b).get! bs.size = b := by
  have hi' : bs.size < (bs.push b).size := by rw [ByteArray.size_push]; omega
  rw [byteArray_get!_eq_getElem hi']
  cases bs with
  | mk data =>
    exact Array.getElem_push_eq

private theorem byteArray_get!_append_left (a b : ByteArray) (i : Nat) (hi : i < a.size) :
    (a ++ b).get! i = a.get! i := by
  have hi' : i < (a ++ b).size := by rw [ByteArray.size_append]; omega
  rw [byteArray_get!_eq_getElem hi', byteArray_get!_eq_getElem hi]
  exact ByteArray.getElem_append_left hi

private theorem byteArray_get!_append_right (a b : ByteArray) (i : Nat)
    (hlo : a.size ≤ i) (hhi : i < a.size + b.size) :
    (a ++ b).get! i = b.get! (i - a.size) := by
  have hi' : i < (a ++ b).size := by rw [ByteArray.size_append]; exact hhi
  have hsub : i - a.size < b.size := by omega
  rw [byteArray_get!_eq_getElem hi', byteArray_get!_eq_getElem hsub]
  exact ByteArray.getElem_append_right hlo

private theorem zeroBytes_size (n : Nat) : (zeroBytes n).size = n := by
  simp [zeroBytes, ByteArray.size]

private theorem zeroBytes_get! (n i : Nat) (hi : i < n) : (zeroBytes n).get! i = 0 := by
  have hi' : i < (zeroBytes n).size := by rw [zeroBytes_size]; exact hi
  rw [byteArray_get!_eq_getElem hi']
  simp [zeroBytes]

/-- `pushU16LE` appends exactly 2 bytes. -/
theorem pushU16LE_size (bs : ByteArray) (v : UInt16) :
    (pushU16LE bs v).size = bs.size + 2 := by
  simp [pushU16LE, ByteArray.size_push]

/-- `pushU32LE` appends exactly 4 bytes. -/
theorem pushU32LE_size (bs : ByteArray) (v : UInt32) :
    (pushU32LE bs v).size = bs.size + 4 := by
  simp [pushU32LE, ByteArray.size_push]

/-- `pushU64LE` appends exactly 8 bytes. -/
theorem pushU64LE_size (bs : ByteArray) (v : UInt64) :
    (pushU64LE bs v).size = bs.size + 8 := by
  simp [pushU64LE, pushU32LE_size]

/-- `pushU16LE` preserves all prefix bytes before `bs.size`. -/
theorem pushU16LE_get_prefix (bs : ByteArray) (v : UInt16) (i : Nat) (hi : i < bs.size) :
    (pushU16LE bs v).get! i = bs.get! i := by
  unfold pushU16LE
  rw [byteArray_get!_push_lt _ _ i (by rw [ByteArray.size_push]; omega),
      byteArray_get!_push_lt _ _ i hi]

/-- `pushU32LE` preserves all prefix bytes before `bs.size`. -/
theorem pushU32LE_get_prefix (bs : ByteArray) (v : UInt32) (i : Nat) (hi : i < bs.size) :
    (pushU32LE bs v).get! i = bs.get! i := by
  unfold pushU32LE
  dsimp only
  rw [byteArray_get!_push_lt _ _ i (by simp [ByteArray.size_push]; omega),
      byteArray_get!_push_lt _ _ i (by simp [ByteArray.size_push]; omega),
      byteArray_get!_push_lt _ _ i (by simp [ByteArray.size_push]; omega),
      byteArray_get!_push_lt _ _ i hi]

/-- `pushU64LE` preserves all prefix bytes before `bs.size`. -/
theorem pushU64LE_get_prefix (bs : ByteArray) (v : UInt64) (i : Nat) (hi : i < bs.size) :
    (pushU64LE bs v).get! i = bs.get! i := by
  unfold pushU64LE
  dsimp only
  rw [pushU32LE_get_prefix _ _ i (by rw [pushU32LE_size]; omega),
      pushU32LE_get_prefix _ _ i hi]

private theorem testBit_ff (i : Nat) : (255 : Nat).testBit i = decide (i < 8) := by
  have h : (255 : Nat) = 2 ^ 8 - 1 := rfl
  rw [h, Nat.testBit_two_pow_sub_one]

private theorem testBit_ffffffff (i : Nat) : (4294967295 : Nat).testBit i = decide (i < 32) := by
  have h : (4294967295 : Nat) = 2 ^ 32 - 1 := rfl
  rw [h, Nat.testBit_two_pow_sub_one]

/-- Exact little-endian round-trip for `pushU16LE` followed by `readU16LE` at arbitrary prefix `bs`. -/
theorem readU16LE_pushU16LE (bs : ByteArray) (v : UInt16) :
    readU16LE (pushU16LE bs v) bs.size = .ok v := by
  unfold readU16LE
  have hsz : (pushU16LE bs v).size = bs.size + 2 := pushU16LE_size bs v
  have hnot : ¬ (bs.size + 2 > (pushU16LE bs v).size) := by omega
  simp only [hnot, ↓reduceIte, pure, Except.pure, Except.ok.injEq]
  have hb0 : (pushU16LE bs v).get! bs.size = (v &&& 0xff).toUInt8 := by
    unfold pushU16LE
    rw [byteArray_get!_push_lt _ _ bs.size (by rw [ByteArray.size_push]; omega),
        byteArray_get!_push_eq]
  have hb1 : (pushU16LE bs v).get! (bs.size + 1) = ((v >>> 8) &&& 0xff).toUInt8 := by
    unfold pushU16LE
    have heq : bs.size + 1 = (bs.push (v &&& 0xff).toUInt8).size := by rw [ByteArray.size_push]
    rw [heq, byteArray_get!_push_eq]
  rw [hb0, hb1]
  apply UInt16.toNat_inj.mp
  apply Nat.eq_of_testBit_eq
  intro i
  rw [UInt16.toNat_or, UInt16.toNat_shiftLeft, Nat.testBit_or]
  have hmod16 : ∀ x : Nat, (x % 65536).testBit i = (decide (i < 16) && x.testBit i) := by
    intro x
    change (x % 2 ^ 16).testBit i = _
    exact Nat.testBit_mod_two_pow x 16 i
  have h0toNat : (v &&& 0xff).toUInt8.toUInt16.toNat = v.toNat % 256 := by
    simp [UInt16.toNat_and]
    rw [show (255 : Nat) = 2 ^ 8 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_mod]
  have h1toNat : ((v >>> 8) &&& 0xff).toUInt8.toUInt16.toNat = (v.toNat >>> 8) % 256 := by
    simp [UInt16.toNat_and, UInt16.toNat_shiftRight]
    rw [show (255 : Nat) = 2 ^ 8 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_mod]
  have h8 : (8 : UInt16).toNat % 16 = 8 := rfl
  rw [h0toNat, h1toNat, h8, hmod16, Nat.testBit_shiftLeft]
  have h256 : ∀ x k : Nat, (x % 256).testBit k = (decide (k < 8) && x.testBit k) := by
    intro x k
    change (x % 2 ^ 8).testBit k = _
    exact Nat.testBit_mod_two_pow x 8 k
  by_cases hi8 : i < 8
  · have hi16 : i < 16 := by omega
    have hge8 : ¬ (8 ≤ i) := by omega
    simp [h256, hi8, hi16, hge8]
  · by_cases hi16 : i < 16
    · have hge8 : 8 ≤ i := by omega
      have hsub8 : i - 8 < 8 := by omega
      have hadd : 8 + (i - 8) = i := by omega
      simp [h256, hi8, hi16, hge8, hsub8, Nat.testBit_shiftRight, hadd]
    · simp [h256, hi8, hi16]
      exact testBit_eq_false_of_lt_two_pow v.toNat_lt (Nat.le_of_not_lt hi16)

/-- Exact little-endian round-trip for `pushU32LE` followed by `readU32LE` at arbitrary prefix `bs`. -/
theorem readU32LE_pushU32LE (bs : ByteArray) (v : UInt32) :
    readU32LE (pushU32LE bs v) bs.size = .ok v := by
  unfold readU32LE
  have hsz : (pushU32LE bs v).size = bs.size + 4 := pushU32LE_size bs v
  have hnot : ¬ (bs.size + 4 > (pushU32LE bs v).size) := by omega
  simp only [hnot, ↓reduceIte, pure, Except.pure, Except.ok.injEq]
  have hb0 : (pushU32LE bs v).get! bs.size = (v &&& 0xff).toUInt8 := by
    unfold pushU32LE; dsimp only
    rw [byteArray_get!_push_lt _ _ bs.size (by rw [ByteArray.size_push, ByteArray.size_push, ByteArray.size_push]; omega),
        byteArray_get!_push_lt _ _ bs.size (by rw [ByteArray.size_push, ByteArray.size_push]; omega),
        byteArray_get!_push_lt _ _ bs.size (by rw [ByteArray.size_push]; omega),
        byteArray_get!_push_eq]
  have hb1 : (pushU32LE bs v).get! (bs.size + 1) = ((v >>> 8) &&& 0xff).toUInt8 := by
    unfold pushU32LE; dsimp only
    rw [byteArray_get!_push_lt _ _ (bs.size + 1) (by rw [ByteArray.size_push, ByteArray.size_push, ByteArray.size_push]; omega),
        byteArray_get!_push_lt _ _ (bs.size + 1) (by rw [ByteArray.size_push, ByteArray.size_push]; omega)]
    have heq : bs.size + 1 = (bs.push (v &&& 0xff).toUInt8).size := by rw [ByteArray.size_push]
    rw [heq, byteArray_get!_push_eq]
  have hb2 : (pushU32LE bs v).get! (bs.size + 2) = ((v >>> 16) &&& 0xff).toUInt8 := by
    unfold pushU32LE; dsimp only
    rw [byteArray_get!_push_lt _ _ (bs.size + 2) (by rw [ByteArray.size_push, ByteArray.size_push, ByteArray.size_push]; omega)]
    have heq : bs.size + 2 = ((bs.push (v &&& 0xff).toUInt8).push ((v >>> 8) &&& 0xff).toUInt8).size := by
      rw [ByteArray.size_push, ByteArray.size_push]
    rw [heq, byteArray_get!_push_eq]
  have hb3 : (pushU32LE bs v).get! (bs.size + 3) = ((v >>> 24) &&& 0xff).toUInt8 := by
    unfold pushU32LE; dsimp only
    have heq : bs.size + 3 =
        (((bs.push (v &&& 0xff).toUInt8).push ((v >>> 8) &&& 0xff).toUInt8).push
          ((v >>> 16) &&& 0xff).toUInt8).size := by
      rw [ByteArray.size_push, ByteArray.size_push, ByteArray.size_push]
    rw [heq, byteArray_get!_push_eq]
  rw [hb0, hb1, hb2, hb3]
  apply UInt32.toNat_inj.mp
  apply Nat.eq_of_testBit_eq
  intro i
  rw [UInt32.toNat_or, UInt32.toNat_or, UInt32.toNat_or,
      UInt32.toNat_shiftLeft, UInt32.toNat_shiftLeft, UInt32.toNat_shiftLeft,
      Nat.testBit_or, Nat.testBit_or, Nat.testBit_or]
  have hmod32 : ∀ x : Nat, (x % 4294967296).testBit i = (decide (i < 32) && x.testBit i) := by
    intro x
    change (x % 2 ^ 32).testBit i = _
    exact Nat.testBit_mod_two_pow x 32 i
  have h256 : ∀ x k : Nat, (x % 256).testBit k = (decide (k < 8) && x.testBit k) := by
    intro x k
    change (x % 2 ^ 8).testBit k = _
    exact Nat.testBit_mod_two_pow x 8 k
  have h0n : (v &&& 0xff).toUInt8.toUInt32.toNat = v.toNat % 256 := by
    simp [UInt32.toNat_and]
    rw [show (255 : Nat) = 2 ^ 8 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_mod]
  have h1n : ((v >>> 8) &&& 0xff).toUInt8.toUInt32.toNat = (v.toNat >>> 8) % 256 := by
    simp [UInt32.toNat_and, UInt32.toNat_shiftRight]
    rw [show (255 : Nat) = 2 ^ 8 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_mod]
  have h2n : ((v >>> 16) &&& 0xff).toUInt8.toUInt32.toNat = (v.toNat >>> 16) % 256 := by
    simp [UInt32.toNat_and, UInt32.toNat_shiftRight]
    rw [show (255 : Nat) = 2 ^ 8 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_mod]
  have h3n : ((v >>> 24) &&& 0xff).toUInt8.toUInt32.toNat = (v.toNat >>> 24) % 256 := by
    simp [UInt32.toNat_and, UInt32.toNat_shiftRight]
    rw [show (255 : Nat) = 2 ^ 8 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_mod]
  have s8 : (8 : UInt32).toNat % 32 = 8 := rfl
  have s16 : (16 : UInt32).toNat % 32 = 16 := rfl
  have s24 : (24 : UInt32).toNat % 32 = 24 := rfl
  rw [h0n, h1n, h2n, h3n, s8, s16, s24, hmod32, hmod32, hmod32,
      Nat.testBit_shiftLeft, Nat.testBit_shiftLeft, Nat.testBit_shiftLeft]
  by_cases hi8 : i < 8
  · have hi32 : i < 32 := by omega
    have hge8 : ¬ (8 ≤ i) := by omega
    have hge16 : ¬ (16 ≤ i) := by omega
    have hge24 : ¬ (24 ≤ i) := by omega
    simp [h256, hi8, hi32, hge8, hge16, hge24]
  · by_cases hi16 : i < 16
    · have hi32 : i < 32 := by omega
      have hge8 : 8 ≤ i := by omega
      have hge16 : ¬ (16 ≤ i) := by omega
      have hge24 : ¬ (24 ≤ i) := by omega
      have hsub8 : i - 8 < 8 := by omega
      have hadd : 8 + (i - 8) = i := by omega
      simp [h256, hi8, hi32, hge8, hge16, hge24, hsub8, Nat.testBit_shiftRight, hadd]
    · by_cases hi24 : i < 24
      · have hi32 : i < 32 := by omega
        have hge8 : 8 ≤ i := by omega
        have hge16 : 16 ≤ i := by omega
        have hge24 : ¬ (24 ≤ i) := by omega
        have hsub8 : ¬ (i - 8 < 8) := by omega
        have hsub16 : i - 16 < 8 := by omega
        have hadd : 16 + (i - 16) = i := by omega
        simp [h256, hi8, hi32, hge8, hge16, hge24, hsub8, hsub16, Nat.testBit_shiftRight, hadd]
      · by_cases hi32 : i < 32
        · have hge8 : 8 ≤ i := by omega
          have hge16 : 16 ≤ i := by omega
          have hge24 : 24 ≤ i := by omega
          have hsub8 : ¬ (i - 8 < 8) := by omega
          have hsub16 : ¬ (i - 16 < 8) := by omega
          have hsub24 : i - 24 < 8 := by omega
          have hadd : 24 + (i - 24) = i := by omega
          simp [h256, hi8, hi32, hge8, hge16, hge24, hsub8, hsub16, hsub24,
                Nat.testBit_shiftRight, hadd]
        · simp [h256, hi8, hi32]
          exact testBit_eq_false_of_lt_two_pow v.toNat_lt (Nat.le_of_not_lt hi32)

/-- `readU16LE` is preserved when any suffix `tail` is appended after `[off, off + 2)`. -/
theorem readU16LE_append_prefix (bs tail : ByteArray) (off : Nat) (h : off + 2 ≤ bs.size) :
    readU16LE (bs ++ tail) off = readU16LE bs off := by
  unfold readU16LE
  have h1 : ¬ (off + 2 > (bs ++ tail).size) := by rw [ByteArray.size_append]; omega
  have h2 : ¬ (off + 2 > bs.size) := by omega
  simp only [h1, h2, ↓reduceIte]
  rw [byteArray_get!_append_left bs tail off (by omega),
      byteArray_get!_append_left bs tail (off + 1) (by omega)]

/-- `readU32LE` is preserved when any suffix `tail` is appended after `[off, off + 4)`. -/
theorem readU32LE_append_prefix (bs tail : ByteArray) (off : Nat) (h : off + 4 ≤ bs.size) :
    readU32LE (bs ++ tail) off = readU32LE bs off := by
  unfold readU32LE
  have h1 : ¬ (off + 4 > (bs ++ tail).size) := by rw [ByteArray.size_append]; omega
  have h2 : ¬ (off + 4 > bs.size) := by omega
  simp only [h1, h2, ↓reduceIte]
  rw [byteArray_get!_append_left bs tail off (by omega),
      byteArray_get!_append_left bs tail (off + 1) (by omega),
      byteArray_get!_append_left bs tail (off + 2) (by omega),
      byteArray_get!_append_left bs tail (off + 3) (by omega)]

/-- `readU64LE` is preserved when any suffix `tail` is appended after `[off, off + 8)`. -/
theorem readU64LE_append_prefix (bs tail : ByteArray) (off : Nat) (h : off + 8 ≤ bs.size) :
    readU64LE (bs ++ tail) off = readU64LE bs off := by
  unfold readU64LE
  have h1 : ¬ (off + 8 > (bs ++ tail).size) := by rw [ByteArray.size_append]; omega
  have h2 : ¬ (off + 8 > bs.size) := by omega
  simp only [h1, h2, ↓reduceIte]
  rw [readU32LE_append_prefix bs tail off (by omega),
      readU32LE_append_prefix bs tail (off + 4) (by omega)]

private theorem readU32LE_pushU32LE_prefix (bs : ByteArray) (v : UInt32) (off : Nat)
    (h : off + 4 ≤ bs.size) :
    readU32LE (pushU32LE bs v) off = readU32LE bs off := by
  unfold readU32LE
  have h1 : ¬ (off + 4 > (pushU32LE bs v).size) := by rw [pushU32LE_size]; omega
  have h2 : ¬ (off + 4 > bs.size) := by omega
  simp only [h1, h2, ↓reduceIte]
  rw [pushU32LE_get_prefix bs v off (by omega),
      pushU32LE_get_prefix bs v (off + 1) (by omega),
      pushU32LE_get_prefix bs v (off + 2) (by omega),
      pushU32LE_get_prefix bs v (off + 3) (by omega)]

/-- Exact little-endian round-trip for `pushU64LE` followed by `readU64LE` at arbitrary prefix `bs`. -/
theorem readU64LE_pushU64LE (bs : ByteArray) (v : UInt64) :
    readU64LE (pushU64LE bs v) bs.size = .ok v := by
  unfold readU64LE pushU64LE
  dsimp only
  have hsz : (pushU32LE (pushU32LE bs (v &&& 0xffffffff).toUInt32)
      ((v >>> 32) &&& 0xffffffff).toUInt32).size = bs.size + 8 := by
    simp [pushU32LE_size]
  have hnot : ¬ (bs.size + 8 > (pushU32LE (pushU32LE bs (v &&& 0xffffffff).toUInt32)
      ((v >>> 32) &&& 0xffffffff).toUInt32).size) := by omega
  simp only [hnot, ↓reduceIte]
  have hlo : readU32LE (pushU32LE (pushU32LE bs (v &&& 0xffffffff).toUInt32)
      ((v >>> 32) &&& 0xffffffff).toUInt32) bs.size = .ok (v &&& 0xffffffff).toUInt32 := by
    rw [readU32LE_pushU32LE_prefix _ _ bs.size (by rw [pushU32LE_size]; omega)]
    exact readU32LE_pushU32LE bs _
  have hhi : readU32LE (pushU32LE (pushU32LE bs (v &&& 0xffffffff).toUInt32)
      ((v >>> 32) &&& 0xffffffff).toUInt32) (bs.size + 4) =
      .ok ((v >>> 32) &&& 0xffffffff).toUInt32 := by
    have heq : bs.size + 4 = (pushU32LE bs (v &&& 0xffffffff).toUInt32).size := by
      rw [pushU32LE_size]
    rw [heq]
    exact readU32LE_pushU32LE _ _
  rw [hlo, hhi]
  simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq]
  apply UInt64.toNat_inj.mp
  apply Nat.eq_of_testBit_eq
  intro i
  rw [UInt64.toNat_or, UInt64.toNat_shiftLeft, Nat.testBit_or]
  have hmod64 : ∀ x : Nat, (x % 18446744073709551616).testBit i = (decide (i < 64) && x.testBit i) := by
    intro x
    change (x % 2 ^ 64).testBit i = _
    exact Nat.testBit_mod_two_pow x 64 i
  have hmod32 : ∀ x k : Nat, (x % 4294967296).testBit k = (decide (k < 32) && x.testBit k) := by
    intro x k
    change (x % 2 ^ 32).testBit k = _
    exact Nat.testBit_mod_two_pow x 32 k
  have hlon : (v &&& 0xffffffff).toUInt32.toUInt64.toNat = v.toNat % 4294967296 := by
    simp [UInt64.toNat_and]
    rw [show (4294967295 : Nat) = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_mod]
  have hhin : ((v >>> 32) &&& 0xffffffff).toUInt32.toUInt64.toNat =
      (v.toNat >>> 32) % 4294967296 := by
    simp [UInt64.toNat_and, UInt64.toNat_shiftRight]
    rw [show (4294967295 : Nat) = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_mod]
  have s32 : (32 : UInt64).toNat % 64 = 32 := rfl
  rw [hlon, hhin, s32, hmod64, Nat.testBit_shiftLeft]
  by_cases hi32 : i < 32
  · have hi64 : i < 64 := by omega
    have hge32 : ¬ (32 ≤ i) := by omega
    simp [hmod32, hi32, hi64, hge32]
  · by_cases hi64 : i < 64
    · have hge32 : 32 ≤ i := by omega
      have hsub32 : i - 32 < 32 := by omega
      have hadd : 32 + (i - 32) = i := by omega
      simp [hmod32, hi32, hi64, hge32, hsub32, Nat.testBit_shiftRight, hadd]
    · simp [hmod32, hi32, hi64]
      exact testBit_eq_false_of_lt_two_pow v.toNat_lt (Nat.le_of_not_lt hi64)

/-- `pushZeros` increases length by exactly `count`. -/
theorem pushZeros_size (bs : ByteArray) (count : Nat) :
    (pushZeros bs count).size = bs.size + count := by
  unfold pushZeros
  split
  · rename_i h
    have hc : count = 0 := by simpa using h
    subst hc; omega
  · rw [ByteArray.size_append, zeroBytes_size]

/-- `pushZeros` preserves all prefix bytes before `bs.size`. -/
theorem pushZeros_get_prefix (bs : ByteArray) (count i : Nat) (hi : i < bs.size) :
    (pushZeros bs count).get! i = bs.get! i := by
  unfold pushZeros
  split
  · rfl
  · exact byteArray_get!_append_left bs (zeroBytes count) i hi

/-- Every byte appended by `pushZeros` is `0`. -/
theorem pushZeros_get_zero (bs : ByteArray) (count i : Nat)
    (hlo : bs.size ≤ i) (hhi : i < bs.size + count) :
    (pushZeros bs count).get! i = 0 := by
  unfold pushZeros
  split
  · rename_i h
    have hc : count = 0 := by simpa using h
    omega
  · rw [byteArray_get!_append_right bs (zeroBytes count) i hlo (by rw [zeroBytes_size]; exact hhi)]
    exact zeroBytes_get! count (i - bs.size) (by omega)

/-- `padToAlign` produces a `ByteArray` of exact length `alignUp bs.size align`. -/
theorem padToAlign_size (bs : ByteArray) (align : Nat) :
    (padToAlign bs align).size = alignUp bs.size align := by
  unfold padToAlign
  dsimp only
  rw [pushZeros_size]
  have hge := alignUp_ge bs.size align
  omega

/-- For any positive `align > 0`, `(padToAlign bs align).size` is divisible by `align`. -/
theorem padToAlign_dvd (bs : ByteArray) (align : Nat) (halign : align > 0) :
    (padToAlign bs align).size % align = 0 := by
  rw [padToAlign_size]
  exact alignUp_dvd bs.size align halign

/-- `padToAlign` preserves all original prefix bytes before `bs.size`. -/
theorem padToAlign_get_prefix (bs : ByteArray) (align i : Nat) (hi : i < bs.size) :
    (padToAlign bs align).get! i = bs.get! i := by
  unfold padToAlign
  exact pushZeros_get_prefix bs _ i hi

/-- Every padding byte added by `padToAlign` in `[bs.size, (padToAlign bs align).size)` is `0`. -/
theorem padToAlign_get_pad (bs : ByteArray) (align i : Nat)
    (hlo : bs.size ≤ i) (hhi : i < (padToAlign bs align).size) :
    (padToAlign bs align).get! i = 0 := by
  rw [padToAlign_size] at hhi
  unfold padToAlign
  have hge := alignUp_ge bs.size align
  exact pushZeros_get_zero bs (alignUp bs.size align - bs.size) i hlo (by omega)

-- ============================================================================
-- 17. Task-Stream Record Codec Round-Trip Soundness (Vane.TaskStream / Vane.Register)
-- ============================================================================

private theorem pushU32LEs_size (bs : ByteArray) (vals : List UInt32) :
    (pushU32LEs bs vals).size = bs.size + 4 * vals.length := by
  induction vals generalizing bs with
  | nil => simp [pushU32LEs]
  | cons v rest ih =>
    simp only [pushU32LEs, List.length_cons]
    rw [ih (pushU32LE bs v), pushU32LE_size]
    omega

private theorem readU32LE_pushU32LEs_prefix (bs : ByteArray) (vals : List UInt32) (off : Nat)
    (h : off + 4 ≤ bs.size) :
    readU32LE (pushU32LEs bs vals) off = readU32LE bs off := by
  induction vals generalizing bs with
  | nil => rfl
  | cons v rest ih =>
    simp only [pushU32LEs]
    rw [ih (pushU32LE bs v) (by rw [pushU32LE_size]; omega)]
    exact readU32LE_pushU32LE_prefix bs v off h

private theorem readU32LE_pushU32LEs_at (bs : ByteArray) (vals : List UInt32) (i : Nat)
    (hi : i < vals.length) :
    readU32LE (pushU32LEs bs vals) (bs.size + 4 * i) = .ok vals[i] := by
  induction vals generalizing bs i with
  | nil => simp at hi
  | cons v rest ih =>
    cases i with
    | zero =>
      simp only [pushU32LEs, Nat.mul_zero, Nat.add_zero, List.getElem_cons_zero]
      rw [readU32LE_pushU32LEs_prefix (pushU32LE bs v) rest bs.size (by rw [pushU32LE_size]; omega)]
      exact readU32LE_pushU32LE bs v
    | succ k =>
      simp only [pushU32LEs, List.getElem_cons_succ]
      have hk : k < rest.length := by simp at hi; omega
      have hoff : bs.size + 4 * (k + 1) = (pushU32LE bs v).size + 4 * k := by
        rw [pushU32LE_size]; omega
      rw [hoff]
      exact ih (pushU32LE bs v) k hk

/-- Encoding a valid sequential register record recovers exact `isBitmask = false`, `base`,
    `count = values.size`, `barSlot`, and `values` from the encoded 32-bit record header. -/
theorem encodeRegRecord_sequential_header
    (base : UInt16) (barSlot : Option UInt8) (values : Array UInt32)
    (hdr : UInt32) (vals : Array UInt32)
    (henc : encodeRegRecord (.sequential base barSlot values) = .ok (hdr, vals)) :
    vals = values ∧
    1 ≤ values.size ∧ values.size ≤ 64 ∧
    ((hdr >>> 31) &&& 1) = 0 ∧
    (hdr &&& 0x7fff).toNat = base.toNat ∧
    (((hdr >>> 15) &&& 0x3f).toNat) + 1 = values.size ∧
    (if ((hdr >>> 29) &&& 1) == 1 then some ((hdr >>> 23) &&& 0x3f).toUInt8 else none) = barSlot := by
  unfold encodeRegRecord at henc
  dsimp only at henc
  obtain ⟨hbase, henc⟩ := except_if_err_ok henc
  obtain ⟨hsz, henc⟩ := except_if_err_ok henc
  obtain ⟨⟨⟩, _, henc⟩ := except_bind_ok henc
  have hbaseLe : base.toNat ≤ 0x7fff := by omega
  have hszBounds : 1 ≤ values.size ∧ values.size ≤ 64 := by
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hsz
    have hnot1 : values.isEmpty = false := by cases h : values.isEmpty <;> simp_all
    have hne0 : values.size ≠ 0 := by
      unfold Array.isEmpty at hnot1
      simpa using hnot1
    have hle64 : values.size ≤ 64 := by omega
    omega
  have hbarExtract : ∃ barBits : UInt32,
      ((barSlot = none ∧ barBits = 0) ∨
       (∃ slot : UInt8, barSlot = some slot ∧ slot.toNat ≤ 63 ∧
         barBits = (((1 : UInt32) <<< 29) ||| (slot.toUInt32 <<< 23)))) ∧
      hdr = (barBits ||| (((values.size - 1).toUInt32 &&& 0x3f) <<< 15) |||
             (base.toUInt32 &&& 0x7fff)) ∧
      vals = values := by
    cases barSlot with
    | none =>
      simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at henc
      exact ⟨0, Or.inl ⟨rfl, rfl⟩, henc.1.symm, henc.2.symm⟩
    | some slot =>
      dsimp only at henc
      obtain ⟨hslot, henc⟩ := except_if_err_ok henc
      simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at henc
      exact ⟨(((1 : UInt32) <<< 29) ||| (slot.toUInt32 <<< 23)),
             Or.inr ⟨slot, rfl, by omega, rfl⟩, henc.1.symm, henc.2.symm⟩
  obtain ⟨barBits, hbarSpec, hhdrEq, hvalsEq⟩ := hbarExtract
  subst hdr
  subst vals
  have hbit31 : (((barBits ||| (((values.size - 1).toUInt32 &&& 0x3f) <<< 15) |||
      (base.toUInt32 &&& 0x7fff)) >>> 31) &&& 1) = 0 := by
    apply UInt32.toNat_inj.mp
    apply Nat.eq_of_testBit_eq; intro i
    rw [UInt32.toNat_and, UInt32.toNat_shiftRight, UInt32.toNat_or, UInt32.toNat_or,
        UInt32.toNat_shiftLeft, UInt32.toNat_and, UInt32.toNat_and]
    simp only [UInt32.toNat_zero, Nat.zero_testBit, Nat.testBit_and, Nat.testBit_shiftRight,
               Nat.testBit_or]
    have s31 : (31 : UInt32).toNat % 32 = 31 := rfl
    have s15 : (15 : UInt32).toNat % 32 = 15 := rfl
    rw [s31, s15]
    by_cases hi0 : i = 0
    · subst hi0
      simp only [Nat.add_zero]
      have h1 : (1 : UInt32).toNat.testBit 0 = true := by decide
      have hcnt : (((values.size - 1).toUInt32.toNat &&& (0x3f : UInt32).toNat) <<< 15 % 2 ^ 32).testBit 31 = false := by
        rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft, Nat.testBit_and]
        have : (0x3f : UInt32).toNat.testBit (31 - 15) = false := by decide
        rw [this]; simp
      have hbase0 : (0x7fff : UInt32).toNat.testBit 31 = false := by decide
      have hbar31 : barBits.toNat.testBit 31 = false := by
        rcases hbarSpec with ⟨_, rfl⟩ | ⟨slot, _, hslot, rfl⟩
        · simp
        · rw [UInt32.toNat_or, UInt32.toNat_shiftLeft, UInt32.toNat_shiftLeft, Nat.testBit_or]
          have hs23 : (23 : UInt32).toNat % 32 = 23 := rfl
          have hs29 : (29 : UInt32).toNat % 32 = 29 := rfl
          rw [hs23, hs29]
          have hslotLt : slot.toUInt32.toNat < 2 ^ 6 := by
            have : slot.toUInt32.toNat = slot.toNat := by simp
            omega
          have hslot31 : (slot.toUInt32.toNat <<< 23 % 2 ^ 32).testBit 31 = false := by
            rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft]
            have hlt : slot.toUInt32.toNat.testBit (31 - 23) = false :=
              testBit_eq_false_of_lt_two_pow hslotLt (by omega)
            rw [hlt]; simp
          have h29' : ((1 : UInt32).toNat <<< 29 % 2 ^ 32).testBit 31 = false := by decide
          rw [h29', hslot31]; rfl
      rw [h1, hcnt, hbase0, hbar31]; simp
    · have h1 : (1 : UInt32).toNat.testBit i = false :=
        testBit_eq_false_of_lt_two_pow (w := 1) (by decide) (by omega)
      rw [h1, Bool.and_false]
  have hbaseRec : ((barBits ||| (((values.size - 1).toUInt32 &&& 0x3f) <<< 15) |||
      (base.toUInt32 &&& 0x7fff)) &&& 0x7fff).toNat = base.toNat := by
    apply Nat.eq_of_testBit_eq; intro i
    rw [UInt32.toNat_and, UInt32.toNat_or, UInt32.toNat_or, UInt32.toNat_shiftLeft,
        UInt32.toNat_and, UInt32.toNat_and,
        Nat.testBit_and, Nat.testBit_or, Nat.testBit_or, Nat.testBit_and]
    have hmask : (0x7fff : UInt32).toNat.testBit i = decide (i < 15) := by
      change (2 ^ 15 - 1 : Nat).testBit i = _
      exact Nat.testBit_two_pow_sub_one 15 i
    have s15 : (15 : UInt32).toNat % 32 = 15 := rfl
    rw [hmask, s15]
    by_cases hi15 : i < 15
    · have hcnt0 : (((values.size - 1).toUInt32.toNat &&& (0x3f : UInt32).toNat) <<< 15 % 2 ^ 32).testBit i = false := by
        rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft]
        have : ¬ (15 ≤ i) := by omega
        simp [this]
      have hbar0 : barBits.toNat.testBit i = false := by
        rcases hbarSpec with ⟨_, rfl⟩ | ⟨slot, _, _, rfl⟩
        · simp
        · rw [UInt32.toNat_or, UInt32.toNat_shiftLeft, UInt32.toNat_shiftLeft, Nat.testBit_or]
          have hs29 : (29 : UInt32).toNat % 32 = 29 := rfl
          have hs23 : (23 : UInt32).toNat % 32 = 23 := rfl
          rw [hs29, hs23]
          have h29 : ((1 : UInt32).toNat <<< 29 % 2 ^ 32).testBit i = false := by
            rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft]
            have : ¬ (29 ≤ i) := by omega
            simp [this]
          have h23 : (slot.toUInt32.toNat <<< 23 % 2 ^ 32).testBit i = false := by
            rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft]
            have : ¬ (23 ≤ i) := by omega
            simp [this]
          rw [h29, h23]; rfl
      have hbEq : base.toUInt32.toNat = base.toNat := by simp
      rw [hcnt0, hbar0, hbEq]
      simp [hi15]
    · simp only [hi15, decide_false, Bool.and_false]
      exact (testBit_eq_false_of_lt_two_pow (w := 15) (by omega) (Nat.le_of_not_lt hi15)).symm
  have hcntRec : ((((barBits ||| (((values.size - 1).toUInt32 &&& 0x3f) <<< 15) |||
      (base.toUInt32 &&& 0x7fff)) >>> 15) &&& 0x3f).toNat) + 1 = values.size := by
    have hsub : ((((barBits ||| (((values.size - 1).toUInt32 &&& 0x3f) <<< 15) |||
        (base.toUInt32 &&& 0x7fff)) >>> 15) &&& 0x3f).toNat) = values.size - 1 := by
      apply Nat.eq_of_testBit_eq; intro i
      rw [UInt32.toNat_and, UInt32.toNat_shiftRight, UInt32.toNat_or, UInt32.toNat_or,
          UInt32.toNat_shiftLeft, UInt32.toNat_and, UInt32.toNat_and,
          Nat.testBit_and, Nat.testBit_shiftRight, Nat.testBit_or, Nat.testBit_or, Nat.testBit_and]
      have hmask : (0x3f : UInt32).toNat.testBit i = decide (i < 6) := by
        change (2 ^ 6 - 1 : Nat).testBit i = _
        exact Nat.testBit_two_pow_sub_one 6 i
      have s15 : (15 : UInt32).toNat % 32 = 15 := rfl
      rw [hmask, s15]
      by_cases hi6 : i < 6
      · have hbar0 : barBits.toNat.testBit (15 + i) = false := by
          rcases hbarSpec with ⟨_, rfl⟩ | ⟨slot, _, _, rfl⟩
          · simp
          · rw [UInt32.toNat_or, UInt32.toNat_shiftLeft, UInt32.toNat_shiftLeft, Nat.testBit_or]
            have hs29 : (29 : UInt32).toNat % 32 = 29 := rfl
            have hs23 : (23 : UInt32).toNat % 32 = 23 := rfl
            rw [hs29, hs23]
            have h29 : ((1 : UInt32).toNat <<< 29 % 2 ^ 32).testBit (15 + i) = false := by
              rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft]
              have : ¬ (29 ≤ 15 + i) := by omega
              simp [this]
            have h23 : (slot.toUInt32.toNat <<< 23 % 2 ^ 32).testBit (15 + i) = false := by
              rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft]
              have : ¬ (23 ≤ 15 + i) := by omega
              simp [this]
            rw [h29, h23]; rfl
        have hbase0 : (0x7fff : UInt32).toNat.testBit (15 + i) = false := by
          change (2 ^ 15 - 1 : Nat).testBit (15 + i) = false
          rw [Nat.testBit_two_pow_sub_one]
          have : ¬ (15 + i < 15) := by omega
          simp [this]
        have hcntBit : (((values.size - 1).toUInt32.toNat &&& (0x3f : UInt32).toNat) <<< 15 % 2 ^ 32).testBit (15 + i) =
            (values.size - 1).testBit i := by
          rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft, Nat.testBit_and]
          have hlt32 : 15 + i < 32 := by omega
          have hge15 : 15 ≤ 15 + i := by omega
          have hsub15 : 15 + i - 15 = i := by omega
          have hm6 : (0x3f : UInt32).toNat.testBit i = true := by
            change (2 ^ 6 - 1 : Nat).testBit i = true
            rw [Nat.testBit_two_pow_sub_one]; simp [hi6]
          have htoNat : (values.size - 1).toUInt32.toNat = values.size - 1 := by
            change (values.size - 1) % 2 ^ 32 = values.size - 1
            exact Nat.mod_eq_of_lt (by omega)
          rw [hsub15, hm6, htoNat]
          simp [hlt32, hge15]
        rw [hbar0, hbase0, hcntBit]
        simp [hi6]
      · simp only [hi6, decide_false, Bool.and_false]
        exact (testBit_eq_false_of_lt_two_pow (w := 6) (by omega) (Nat.le_of_not_lt hi6)).symm
    omega
  have hbarRec : (if (((barBits ||| (((values.size - 1).toUInt32 &&& 0x3f) <<< 15) |||
      (base.toUInt32 &&& 0x7fff)) >>> 29) &&& 1) == 1 then
      some (((barBits ||| (((values.size - 1).toUInt32 &&& 0x3f) <<< 15) |||
        (base.toUInt32 &&& 0x7fff)) >>> 23) &&& 0x3f).toUInt8 else none) = barSlot := by
    have hbit29 : (((barBits ||| (((values.size - 1).toUInt32 &&& 0x3f) <<< 15) |||
        (base.toUInt32 &&& 0x7fff)) >>> 29) &&& 1) = (if barSlot.isSome then 1 else 0) := by
      apply UInt32.toNat_inj.mp
      apply Nat.eq_of_testBit_eq; intro i
      rw [UInt32.toNat_and, UInt32.toNat_shiftRight, UInt32.toNat_or, UInt32.toNat_or,
          UInt32.toNat_shiftLeft, UInt32.toNat_and, UInt32.toNat_and,
          Nat.testBit_and, Nat.testBit_shiftRight, Nat.testBit_or, Nat.testBit_or, Nat.testBit_and]
      have s29 : (29 : UInt32).toNat % 32 = 29 := rfl
      have s15 : (15 : UInt32).toNat % 32 = 15 := rfl
      rw [s29, s15]
      by_cases hi0 : i = 0
      · subst hi0
        simp only [Nat.add_zero]
        have hcnt0 : (((values.size - 1).toUInt32.toNat &&& (0x3f : UInt32).toNat) <<< 15 % 2 ^ 32).testBit 29 = false := by
          rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft, Nat.testBit_and]
          have : (0x3f : UInt32).toNat.testBit (29 - 15) = false := by decide
          rw [this]; simp
        have hbase0 : (0x7fff : UInt32).toNat.testBit 29 = false := by decide
        have h1 : (1 : UInt32).toNat.testBit 0 = true := by decide
        rcases hbarSpec with ⟨rfl, rfl⟩ | ⟨slot, rfl, hslot, rfl⟩
        · rw [hcnt0, hbase0]; simp
        · have hbar29 : (((1 : UInt32) <<< 29) ||| (slot.toUInt32 <<< 23)).toNat.testBit 29 = true := by
            rw [UInt32.toNat_or, Nat.testBit_or]
            have : ((1 : UInt32) <<< 29).toNat.testBit 29 = true := by decide
            rw [this]; rfl
          rw [hcnt0, hbase0, hbar29, h1]; simp
      · have h1 : (1 : UInt32).toNat.testBit i = false :=
          testBit_eq_false_of_lt_two_pow (w := 1) (by decide) (by omega)
        rcases hbarSpec with ⟨rfl, rfl⟩ | ⟨slot, rfl, _, rfl⟩
        · rw [h1, Bool.and_false]; simp
        · rw [h1, Bool.and_false]
          change false = (1 : UInt32).toNat.testBit i
          exact h1.symm
    rw [hbit29]
    rcases hbarSpec with ⟨rfl, rfl⟩ | ⟨slot, rfl, hslot, rfl⟩
    · simp
    · simp only [Option.isSome_some, ↓reduceIte, beq_self_eq_true, Option.some.injEq]
      apply UInt8.toNat_inj.mp
      apply Nat.eq_of_testBit_eq; intro i
      have htoU8 : ∀ x : UInt32, x.toUInt8.toNat = x.toNat % 256 := fun _ => rfl
      rw [htoU8]
      change (((((1 : UInt32) <<< 29 ||| slot.toUInt32 <<< 23 |||
          ((values.size - 1).toUInt32 &&& 0x3f) <<< 15 |||
          (base.toUInt32 &&& 0x7fff)) >>> 23) &&& 0x3f).toNat % 2 ^ 8).testBit i = _
      rw [Nat.testBit_mod_two_pow, UInt32.toNat_and, UInt32.toNat_shiftRight,
          UInt32.toNat_or, UInt32.toNat_or, UInt32.toNat_or,
          UInt32.toNat_shiftLeft, UInt32.toNat_shiftLeft, UInt32.toNat_shiftLeft,
          UInt32.toNat_and, UInt32.toNat_and,
          Nat.testBit_and, Nat.testBit_shiftRight, Nat.testBit_or, Nat.testBit_or, Nat.testBit_or,
          Nat.testBit_and]
      have hmask : (0x3f : UInt32).toNat.testBit i = decide (i < 6) := by
        change (2 ^ 6 - 1 : Nat).testBit i = _
        exact Nat.testBit_two_pow_sub_one 6 i
      have s29 : (29 : UInt32).toNat % 32 = 29 := rfl
      have s23 : (23 : UInt32).toNat % 32 = 23 := rfl
      have s15 : (15 : UInt32).toNat % 32 = 15 := rfl
      rw [hmask, s29, s23, s15]
      by_cases hi6 : i < 6
      · have hi8 : i < 8 := by omega
        have h29 : ((1 : UInt32).toNat <<< 29 % 2 ^ 32).testBit (23 + i) = false := by
          rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft]
          have : ¬ (29 ≤ 23 + i) := by omega
          simp [this]
        have hcnt0 : (((values.size - 1).toUInt32.toNat &&& (0x3f : UInt32).toNat) <<< 15 % 2 ^ 32).testBit (23 + i) = false := by
          rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft, Nat.testBit_and]
          have hm6 : (0x3f : UInt32).toNat.testBit (23 + i - 15) = false :=
            testBit_eq_false_of_lt_two_pow (w := 6) (by decide) (by omega)
          rw [hm6]; simp
        have hbase0 : (0x7fff : UInt32).toNat.testBit (23 + i) = false := by
          change (2 ^ 15 - 1 : Nat).testBit (23 + i) = false
          rw [Nat.testBit_two_pow_sub_one]
          have : ¬ (23 + i < 15) := by omega
          simp [this]
        have hslotBit : (slot.toUInt32.toNat <<< 23 % 2 ^ 32).testBit (23 + i) = slot.toNat.testBit i := by
          rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft]
          have hlt32 : 23 + i < 32 := by omega
          have hge23 : 23 ≤ 23 + i := by omega
          have hsub23 : 23 + i - 23 = i := by omega
          have hsn : slot.toUInt32.toNat = slot.toNat := by simp
          simp [hlt32, hge23, hsub23, hsn]
        rw [h29, hcnt0, hbase0, hslotBit]
        simp [hi6, hi8]
      · simp only [hi6, decide_false, Bool.and_false]
        exact (testBit_eq_false_of_lt_two_pow (w := 6) (by omega) (Nat.le_of_not_lt hi6)).symm
  exact ⟨rfl, hszBounds.1, hszBounds.2, hbit31, hbaseRec, hcntRec, hbarRec⟩

/-- Encoding a valid bitmask register record recovers exact `isBitmask = true`, `base`, `mask`,
    and `values.size = 1 + popcount16 mask` from the encoded 32-bit record header. -/
theorem encodeRegRecord_bitmask_header
    (base mask : UInt16) (values : Array UInt32)
    (hdr : UInt32) (vals : Array UInt32)
    (henc : encodeRegRecord (.bitmask base mask values) = .ok (hdr, vals)) :
    vals = values ∧
    values.size = 1 + popcount16 mask ∧
    ((hdr >>> 31) &&& 1) = 1 ∧
    (hdr &&& 0x7fff).toNat = base.toNat ∧
    ((hdr >>> 15) &&& 0xffff).toNat = mask.toNat := by
  unfold encodeRegRecord at henc
  dsimp only at henc
  obtain ⟨hbase, henc⟩ := except_if_err_ok henc
  obtain ⟨hsz, henc⟩ := except_if_err_ok henc
  obtain ⟨_, henc⟩ := except_if_err_ok henc
  obtain ⟨⟨⟩, _, henc⟩ := except_bind_ok henc
  simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at henc
  rcases henc with ⟨rfl, rfl⟩
  have hbaseLe : base.toNat ≤ 0x7fff := by omega
  have hszEq : values.size = 1 + popcount16 mask := by simpa using hsz
  have hbit31 : ((((1 : UInt32) <<< 31 ||| (mask.toUInt32 <<< 15) |||
      (base.toUInt32 &&& 0x7fff)) >>> 31) &&& 1) = 1 := by
    apply UInt32.toNat_inj.mp
    apply Nat.eq_of_testBit_eq; intro i
    rw [UInt32.toNat_and, UInt32.toNat_shiftRight, UInt32.toNat_or, UInt32.toNat_or,
        Nat.testBit_and, Nat.testBit_shiftRight, Nat.testBit_or, Nat.testBit_or]
    have s31 : (31 : UInt32).toNat % 32 = 31 := rfl
    rw [s31]
    by_cases hi0 : i = 0
    · subst hi0
      have h31 : ((1 : UInt32) <<< 31).toNat.testBit (31 + 0) = true := by decide
      have h1 : (1 : UInt32).toNat.testBit 0 = true := by decide
      rw [h31, h1]; rfl
    · have h1 : (1 : UInt32).toNat.testBit i = false :=
        testBit_eq_false_of_lt_two_pow (w := 1) (by decide) (by omega)
      rw [h1, Bool.and_false]
  have hbaseRec : (((1 : UInt32) <<< 31 ||| (mask.toUInt32 <<< 15) |||
      (base.toUInt32 &&& 0x7fff)) &&& 0x7fff).toNat = base.toNat := by
    apply Nat.eq_of_testBit_eq; intro i
    rw [UInt32.toNat_and, UInt32.toNat_or, UInt32.toNat_or, UInt32.toNat_shiftLeft,
        UInt32.toNat_shiftLeft, UInt32.toNat_and,
        Nat.testBit_and, Nat.testBit_or, Nat.testBit_or, Nat.testBit_and]
    have hmask : (0x7fff : UInt32).toNat.testBit i = decide (i < 15) := by
      change (2 ^ 15 - 1 : Nat).testBit i = _
      exact Nat.testBit_two_pow_sub_one 15 i
    have s15 : (15 : UInt32).toNat % 32 = 15 := rfl
    have s31 : (31 : UInt32).toNat % 32 = 31 := rfl
    rw [hmask, s15, s31]
    by_cases hi15 : i < 15
    · have h31 : ((1 : UInt32).toNat <<< 31 % 2 ^ 32).testBit i = false := by
        rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft]
        have : ¬ (31 ≤ i) := by omega
        simp [this]
      have hm0 : (mask.toUInt32.toNat <<< 15 % 2 ^ 32).testBit i = false := by
        rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft]
        have : ¬ (15 ≤ i) := by omega
        simp [this]
      have hbEq : base.toUInt32.toNat = base.toNat := by simp
      rw [h31, hm0, hbEq]
      simp [hi15]
    · simp only [hi15, decide_false, Bool.and_false]
      symm
      exact testBit_eq_false_of_lt_two_pow (w := 15) (by omega) (by omega)
  have hmaskRec : ((((1 : UInt32) <<< 31 ||| (mask.toUInt32 <<< 15) |||
      (base.toUInt32 &&& 0x7fff)) >>> 15) &&& 0xffff).toNat = mask.toNat := by
    apply Nat.eq_of_testBit_eq; intro i
    rw [UInt32.toNat_and, UInt32.toNat_shiftRight, UInt32.toNat_or, UInt32.toNat_or,
        UInt32.toNat_shiftLeft, UInt32.toNat_shiftLeft, UInt32.toNat_and,
        Nat.testBit_and, Nat.testBit_shiftRight, Nat.testBit_or, Nat.testBit_or, Nat.testBit_and]
    have hmask : (0xffff : UInt32).toNat.testBit i = decide (i < 16) := by
      change (2 ^ 16 - 1 : Nat).testBit i = _
      exact Nat.testBit_two_pow_sub_one 16 i
    have s15 : (15 : UInt32).toNat % 32 = 15 := rfl
    have s31 : (31 : UInt32).toNat % 32 = 31 := rfl
    rw [hmask, s15, s31]
    by_cases hi16 : i < 16
    · have h31 : ((1 : UInt32).toNat <<< 31 % 2 ^ 32).testBit (15 + i) = false := by
        rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft]
        have : ¬ (31 ≤ 15 + i) := by omega
        simp [this]
      have hbase0 : (0x7fff : UInt32).toNat.testBit (15 + i) = false := by
        change (2 ^ 15 - 1 : Nat).testBit (15 + i) = false
        rw [Nat.testBit_two_pow_sub_one]
        have : ¬ (15 + i < 15) := by omega
        simp [this]
      have hmBit : (mask.toUInt32.toNat <<< 15 % 2 ^ 32).testBit (15 + i) = mask.toNat.testBit i := by
        rw [Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft]
        have hlt32 : 15 + i < 32 := by omega
        have hge15 : 15 ≤ 15 + i := by omega
        have hsub15 : 15 + i - 15 = i := by omega
        have hmEq : mask.toUInt32.toNat = mask.toNat := by simp
        simp [hlt32, hge15, hsub15, hmEq]
      rw [h31, hmBit, hbase0]
      simp [hi16]
    · simp only [hi16, decide_false, Bool.and_false]
      exact (testBit_eq_false_of_lt_two_pow mask.toNat_lt (Nat.le_of_not_lt hi16)).symm
  exact ⟨rfl, hszEq, hbit31, hbaseRec, hmaskRec⟩

private theorem decodeBitmaskAddrsAux_length (base mask k : Nat) (acc : List Nat) :
    (decodeBitmaskAddrsAux base mask k acc).length = acc.length + popcountBits mask k := by
  induction k generalizing acc with
  | zero => simp [decodeBitmaskAddrsAux, popcountBits]
  | succ i ih =>
    simp only [decodeBitmaskAddrsAux, popcountBits]
    split
    · rw [ih]; simp; omega
    · rw [ih]; omega

/-- The number of register addresses decoded by `decodeBitmaskAddrs` equals `1 + popcount16 mask`. -/
theorem decodeBitmaskAddrs_length (base : Nat) (mask : UInt16) :
    (decodeBitmaskAddrs base mask.toNat).length = 1 + popcount16 mask := by
  unfold decodeBitmaskAddrs popcount16
  simp only [List.length_cons, decodeBitmaskAddrsAux_length, List.length_nil]
  omega

/-- The first address of `decodeBitmaskAddrs base mask` is always `base * 4`. -/
theorem decodeBitmaskAddrs_head (base mask : Nat) :
    (decodeBitmaskAddrs base mask)[0]? = some (base * 4) := rfl

private theorem mem_decodeBitmaskAddrsAux_iff (base mask k : Nat) (acc : List Nat) (addr : Nat) :
    addr ∈ decodeBitmaskAddrsAux base mask k acc ↔
      addr ∈ acc ∨ ∃ i < k, ((mask >>> i) &&& 1) = 1 ∧ addr = (base + i + 1) * 4 := by
  induction k generalizing acc with
  | zero =>
    simp [decodeBitmaskAddrsAux]
  | succ j ih =>
    simp only [decodeBitmaskAddrsAux]
    split
    · rename_i hbit
      have hbitEq : ((mask >>> j) &&& 1) = 1 := by simpa using hbit
      rw [ih]
      simp only [List.mem_cons]
      constructor
      · rintro ((rfl | hacc) | ⟨i, hi, hb, rfl⟩)
        · right; exact ⟨j, by omega, hbitEq, rfl⟩
        · left; exact hacc
        · right; exact ⟨i, by omega, hb, rfl⟩
      · rintro (hacc | ⟨i, hi, hb, rfl⟩)
        · left; right; exact hacc
        · by_cases hij : i = j
          · subst hij; left; left; rfl
          · right; exact ⟨i, by omega, hb, rfl⟩
    · rename_i hbit
      have hbitNe : ((mask >>> j) &&& 1) ≠ 1 := by simpa using hbit
      rw [ih]
      constructor
      · rintro (hacc | ⟨i, hi, hb, rfl⟩)
        · left; exact hacc
        · right; exact ⟨i, by omega, hb, rfl⟩
      · rintro (hacc | ⟨i, hi, hb, rfl⟩)
        · left; exact hacc
        · by_cases hij : i = j
          · subst hij; exact absurd hb hbitNe
          · right; exact ⟨i, by omega, hb, rfl⟩

/-- Exact membership characterization for `decodeBitmaskAddrs`: an address is targeted iff it is
    `base * 4` or `(base + i + 1) * 4` for a set bit `i < 16` of `mask`. -/
theorem mem_decodeBitmaskAddrs_iff (base mask addr : Nat) :
    addr ∈ decodeBitmaskAddrs base mask ↔
      addr = base * 4 ∨ ∃ i < 16, ((mask >>> i) &&& 1) = 1 ∧ addr = (base + i + 1) * 4 := by
  unfold decodeBitmaskAddrs
  simp only [List.mem_cons, mem_decodeBitmaskAddrsAux_iff, List.not_mem_nil, false_or]

private theorem decodeSequentialWrites_spec
    (bs : ByteArray) (recPos base cnt : Nat) (slot? : Option UInt8)
    (vals : List UInt32) (i : Nat) (acc : Array DecodedWrite)
    (hread : ∀ k (hk : k < vals.length), readU32LE bs (recPos + 4 * (i + k)) = .ok vals[k]) :
    decodeSequentialWrites bs recPos base cnt slot? i vals.length acc =
      .ok (acc ++ (List.ofFn (n := vals.length) fun j => {
        byteAddr := (base + (i + j.val)) * 4
        value := vals[j.val]
        barSlot := slot?
        payloadOffset := recPos + 4 * (i + j.val)
        recordLen := cnt
      }).toArray) := by
  induction vals generalizing i acc with
  | nil =>
    simp [decodeSequentialWrites]
  | cons v rest ih =>
    simp only [List.length_cons, decodeSequentialWrites]
    have h0 : readU32LE bs (recPos + 4 * i) = .ok v := by
      have h := hread 0 (by simp)
      simp only [Nat.add_zero] at h
      exact h
    rw [h0]
    simp only [bind, Except.bind]
    have hrest : ∀ k (hk : k < rest.length), readU32LE bs (recPos + 4 * (i + 1 + k)) = .ok rest[k] := by
      intro k hk
      have h' := hread (k + 1) (by simp; omega)
      have hidx : i + (k + 1) = i + 1 + k := by omega
      rw [hidx] at h'
      simpa using h'
    rw [ih (i + 1) _ hrest]
    congr 1
    apply Array.ext'
    simp only [Array.toList_append, Array.toList_push, List.ofFn_succ]
    simp only [List.append_assoc, List.singleton_append]
    congr 2
    apply List.ext_getElem (by simp)
    intro j h1 h2
    simp only [List.getElem_ofFn, Fin.val_succ, List.getElem_cons_succ]
    have hadd : i + 1 + j = i + (j + 1) := by omega
    simp [hadd]

private theorem decodeBitmaskWrites_spec
    (bs : ByteArray) (recPos recLen : Nat)
    (addrs : List Nat) (vals : List UInt32) (i : Nat) (acc : Array DecodedWrite)
    (hlen : addrs.length = vals.length)
    (hread : ∀ k (hk : k < vals.length), readU32LE bs (recPos + 4 * (i + k)) = .ok vals[k]) :
    decodeBitmaskWrites bs recPos recLen i addrs acc =
      .ok (acc ++ (List.ofFn (n := vals.length) fun j => {
        byteAddr := addrs[j.val]'(by omega)
        value := vals[j.val]
        barSlot := none
        payloadOffset := recPos + 4 * (i + j.val)
        recordLen := recLen
      }).toArray) := by
  induction addrs generalizing vals i acc with
  | nil =>
    cases vals with
    | nil => simp [decodeBitmaskWrites]
    | cons _ _ => simp at hlen
  | cons a restA ih =>
    cases vals with
    | nil => simp at hlen
    | cons v restV =>
      simp only [decodeBitmaskWrites]
      have h0 : readU32LE bs (recPos + 4 * i) = .ok v := by
        have h := hread 0 (by simp)
        simp only [Nat.add_zero] at h
        exact h
      rw [h0]
      simp only [bind, Except.bind]
      have hlen' : restA.length = restV.length := by simp at hlen; omega
      have hrest : ∀ k (hk : k < restV.length), readU32LE bs (recPos + 4 * (i + 1 + k)) = .ok restV[k] := by
        intro k hk
        have h' := hread (k + 1) (by simp; omega)
        have hidx : i + (k + 1) = i + 1 + k := by omega
        rw [hidx] at h'
        simpa using h'
      rw [ih restV (i + 1) _ hlen' hrest]
      congr 1
      apply Array.ext'
      simp only [Array.toList_append, Array.toList_push, List.length_cons,
                 List.ofFn_succ, List.append_assoc, List.singleton_append]
      congr 2
      apply List.ext_getElem (by simp)
      intro j h1 h2
      simp only [List.getElem_ofFn, Fin.val_succ, List.getElem_cons_succ]
      have hadd : i + 1 + j = i + (j + 1) := by omega
      simp [hadd]

/-- Full byte-level round-trip soundness for production `encodeRegRecord` + `decodeRegRecordAt`
    on any valid single-word or multi-word sequential register record (plain or BAR-tagged) at
    arbitrary prefix `bs` and trailing bytes `suffix`. -/
theorem decodeRegRecordAt_sequential_roundTrip
    (bs suffix : ByteArray) (acc : Array DecodedWrite)
    (base : UInt16) (barSlot : Option UInt8) (values : Array UInt32)
    (hdr : UInt32) (vals : Array UInt32)
    (henc : encodeRegRecord (.sequential base barSlot values) = .ok (hdr, vals)) :
    let encoded := pushU32LEs (pushU32LE bs hdr) vals.toList ++ suffix
    decodeRegRecordAt encoded bs.size acc =
      .ok (acc ++ (List.ofFn (n := values.size) fun j => {
        byteAddr := (base.toNat + j.val) * 4
        value := values[j.val]
        barSlot := barSlot
        payloadOffset := bs.size + 4 + 4 * j.val
        recordLen := values.size
      }).toArray, bs.size + 4 + 4 * values.size) := by
  obtain ⟨hvalsEq, hsz1, hsz64, hbit31, hbase, hcnt, hslot⟩ :=
    encodeRegRecord_sequential_header base barSlot values hdr vals henc
  subst vals
  dsimp only
  unfold decodeRegRecordAt
  let pre := pushU32LEs (pushU32LE bs hdr) values.toList
  have hpreSz : pre.size = bs.size + 4 + 4 * values.size := by
    simp [pre, pushU32LEs_size, pushU32LE_size]
  have hhdrRead : readU32LE (pre ++ suffix) bs.size = .ok hdr := by
    rw [readU32LE_append_prefix pre suffix bs.size (by omega)]
    rw [readU32LE_pushU32LEs_prefix (pushU32LE bs hdr) values.toList bs.size (by rw [pushU32LE_size]; omega)]
    exact readU32LE_pushU32LE bs hdr
  rw [show readU32LE (pushU32LEs (pushU32LE bs hdr) values.toList ++ suffix) bs.size = .ok hdr from hhdrRead]
  simp only [bind, Except.bind, hbit31, hbase, hcnt, hslot]
  have hnotBm : ((0 : UInt32) == 1) = false := rfl
  simp only [hnotBm, Bool.false_eq_true, ↓reduceIte]
  have hread : ∀ k (hk : k < values.toList.length),
      readU32LE (pre ++ suffix) (bs.size + 4 + 4 * (0 + k)) = .ok values.toList[k] := by
    intro k hk
    have hk' : k < values.size := by simpa using hk
    rw [readU32LE_append_prefix pre suffix (bs.size + 4 + 4 * (0 + k)) (by omega)]
    have hoff : bs.size + 4 + 4 * (0 + k) = (pushU32LE bs hdr).size + 4 * k := by
      rw [pushU32LE_size]; omega
    rw [hoff]
    exact readU32LE_pushU32LEs_at (pushU32LE bs hdr) values.toList k hk
  have hseq := decodeSequentialWrites_spec (pre ++ suffix) (bs.size + 4) base.toNat values.size barSlot
        values.toList 0 acc hread
  simp only [Array.length_toList] at hseq
  rw [hseq]
  simp only [Nat.zero_add, pure, Except.pure, Array.getElem_toList]

/-- Specialized single-word sequential register record round-trip: encoding `(base, barSlot, #[val])`
    and decoding via `decodeRegRecordAt` recovers exact `byteAddr = base * 4`, `value = val`,
    `barSlot`, `payloadOffset = bs.size + 4`, and `recordLen = 1`. -/
theorem decodeRegRecordAt_single_roundTrip
    (bs suffix : ByteArray) (acc : Array DecodedWrite)
    (base : UInt16) (barSlot : Option UInt8) (val : UInt32)
    (hdr : UInt32) (vals : Array UInt32)
    (henc : encodeRegRecord (.sequential base barSlot #[val]) = .ok (hdr, vals)) :
    let encoded := pushU32LEs (pushU32LE bs hdr) vals.toList ++ suffix
    decodeRegRecordAt encoded bs.size acc =
      .ok (acc.push {
        byteAddr := base.toNat * 4
        value := val
        barSlot := barSlot
        payloadOffset := bs.size + 4
        recordLen := 1
      }, bs.size + 8) := by
  have h := decodeRegRecordAt_sequential_roundTrip bs suffix acc base barSlot #[val] hdr vals henc
  simpa using h

/-- Full byte-level round-trip soundness for production `encodeRegRecord` + `decodeRegRecordAt`
    on any valid bitmask register record at arbitrary prefix `bs` and trailing bytes `suffix`. -/
theorem decodeRegRecordAt_bitmask_roundTrip
    (bs suffix : ByteArray) (acc : Array DecodedWrite)
    (base mask : UInt16) (values : Array UInt32)
    (hdr : UInt32) (vals : Array UInt32)
    (henc : encodeRegRecord (.bitmask base mask values) = .ok (hdr, vals)) :
    let encoded := pushU32LEs (pushU32LE bs hdr) vals.toList ++ suffix
    let addrs := decodeBitmaskAddrs base.toNat mask.toNat
    decodeRegRecordAt encoded bs.size acc =
      .ok (acc ++ (List.ofFn (n := values.size) fun j => {
        byteAddr := addrs[j.val]'(by
          have h1 := (encodeRegRecord_bitmask_header base mask values hdr vals henc).2.1
          have h2 := decodeBitmaskAddrs_length base.toNat mask
          show j.val < (decodeBitmaskAddrs base.toNat mask.toNat).length
          rw [h2, ← h1]; exact j.isLt)
        value := values[j.val]
        barSlot := none
        payloadOffset := bs.size + 4 + 4 * j.val
        recordLen := values.size
      }).toArray, bs.size + 4 + 4 * values.size) := by
  obtain ⟨hvalsEq, hsz, hbit31, hbase, hmask⟩ :=
    encodeRegRecord_bitmask_header base mask values hdr vals henc
  subst vals
  dsimp only
  unfold decodeRegRecordAt
  let pre := pushU32LEs (pushU32LE bs hdr) values.toList
  have hpreSz : pre.size = bs.size + 4 + 4 * values.size := by
    simp [pre, pushU32LEs_size, pushU32LE_size]
  have hhdrRead : readU32LE (pre ++ suffix) bs.size = .ok hdr := by
    rw [readU32LE_append_prefix pre suffix bs.size (by omega)]
    rw [readU32LE_pushU32LEs_prefix (pushU32LE bs hdr) values.toList bs.size (by rw [pushU32LE_size]; omega)]
    exact readU32LE_pushU32LE bs hdr
  rw [show readU32LE (pushU32LEs (pushU32LE bs hdr) values.toList ++ suffix) bs.size = .ok hdr from hhdrRead]
  simp only [bind, Except.bind, hbit31, hbase, hmask, beq_self_eq_true, ↓reduceIte]
  have haddrsLen : (decodeBitmaskAddrs base.toNat mask.toNat).length = values.toList.length := by
    rw [decodeBitmaskAddrs_length base.toNat mask]
    simp [hsz]
  have hread : ∀ k (hk : k < values.toList.length),
      readU32LE (pre ++ suffix) (bs.size + 4 + 4 * (0 + k)) = .ok values.toList[k] := by
    intro k hk
    have hk' : k < values.size := by simpa using hk
    rw [readU32LE_append_prefix pre suffix (bs.size + 4 + 4 * (0 + k)) (by omega)]
    have hoff : bs.size + 4 + 4 * (0 + k) = (pushU32LE bs hdr).size + 4 * k := by
      rw [pushU32LE_size]; omega
    rw [hoff]
    exact readU32LE_pushU32LEs_at (pushU32LE bs hdr) values.toList k hk
  rw [decodeBitmaskWrites_spec (pre ++ suffix) (bs.size + 4)
        (decodeBitmaskAddrs base.toNat mask.toNat).length
        (decodeBitmaskAddrs base.toNat mask.toNat) values.toList 0 acc haddrsLen hread]
  have hlenVal : (decodeBitmaskAddrs base.toNat mask.toNat).length = values.size := by
    simpa using haddrsLen
  simp only [Nat.zero_add, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq, hlenVal,
             Array.getElem_toList, and_true]
  congr 1

/-- Expected `DecodedWrite` list produced by encoding a single `RegRecord` starting at byte offset `off`. -/
def expectedRecordWrites (off : Nat) : RegRecord → List DecodedWrite
  | .sequential base barSlot values =>
    List.ofFn (n := values.size) fun j => {
      byteAddr := (base.toNat + j.val) * 4
      value := values[j.val]
      barSlot := barSlot
      payloadOffset := off + 4 + 4 * j.val
      recordLen := values.size
    }
  | .bitmask base mask values =>
    let addrs := decodeBitmaskAddrs base.toNat mask.toNat
    List.ofFn (n := values.size) fun j => {
      byteAddr := addrs.getD j.val 0
      value := values[j.val]
      barSlot := none
      payloadOffset := off + 4 + 4 * j.val
      recordLen := values.size
    }

/-- Encode a list of `RegRecord`s into bytes (appending onto `bs`) and compute the exact expected
    sequence of `DecodedWrite`s. -/
def encodeRecordList (bs : ByteArray) : List RegRecord → Except String (ByteArray × List DecodedWrite)
  | [] => .ok (bs, [])
  | r :: rest =>
    encodeRegRecord r >>= fun (hdr, vals) =>
    let ws := expectedRecordWrites bs.size r
    let bs' := pushU32LEs (pushU32LE bs hdr) vals.toList
    encodeRecordList bs' rest >>= fun (outBs, restWs) =>
    .ok (outBs, ws ++ restWs)

private theorem byteArray_append_empty (a : ByteArray) : a ++ ByteArray.empty = a := by
  cases a with
  | mk da =>
    cases da with
    | mk la =>
      change ByteArray.mk (Array.mk (la ++ [])) = ByteArray.mk (Array.mk la)
      simp

private theorem byteArray_append_assoc (a b c : ByteArray) : (a ++ b) ++ c = a ++ (b ++ c) := by
  cases a with
  | mk da =>
    cases da with
    | mk la =>
      cases b with
      | mk db =>
        cases db with
        | mk lb =>
          cases c with
          | mk dc =>
            cases dc with
            | mk lc =>
              change ByteArray.mk (Array.mk ((la ++ lb) ++ lc)) = ByteArray.mk (Array.mk (la ++ (lb ++ lc)))
              simp [List.append_assoc]

private theorem encodeRecordList_append_split (bs : ByteArray) (recs : List RegRecord)
    (outBs : ByteArray) (ws : List DecodedWrite)
    (h : encodeRecordList bs recs = .ok (outBs, ws)) :
    ∃ tail : ByteArray, outBs = bs ++ tail := by
  induction recs generalizing bs outBs ws with
  | nil =>
    simp only [encodeRecordList, Except.ok.injEq, Prod.mk.injEq] at h
    rcases h with ⟨rfl, _⟩
    exact ⟨ByteArray.empty, (byteArray_append_empty bs).symm⟩
  | cons r rest ih =>
    unfold encodeRecordList at h
    obtain ⟨⟨hdr, vals⟩, henc, h⟩ := except_bind_ok h
    obtain ⟨⟨outBs', restWs⟩, hrest, h⟩ := except_bind_ok h
    simp only [Except.ok.injEq, Prod.mk.injEq] at h
    rcases h with ⟨rfl, rfl⟩
    obtain ⟨tail2, hout⟩ := ih _ _ _ hrest
    have hpush : ∀ (a : ByteArray) (b : UInt8), ∃ t, a.push b = a ++ t := by
      intro a b
      refine ⟨ByteArray.mk #[b], ?_⟩
      cases a with
      | mk da =>
        cases da with
        | mk la =>
          change ByteArray.mk (Array.push (Array.mk la) b) = ByteArray.mk (Array.mk (la ++ [b]))
          congr 1
          apply Array.ext'
          simp
    have hpush32 : ∀ (a : ByteArray) (v : UInt32), ∃ t, pushU32LE a v = a ++ t := by
      intro a v
      unfold pushU32LE; dsimp only
      obtain ⟨t1, h1⟩ := hpush a _
      obtain ⟨t2, h2⟩ := hpush _ _
      obtain ⟨t3, h3⟩ := hpush _ _
      obtain ⟨t4, h4⟩ := hpush _ _
      refine ⟨t1 ++ (t2 ++ (t3 ++ t4)), ?_⟩
      rw [h4, h3, h2, h1, byteArray_append_assoc, byteArray_append_assoc, byteArray_append_assoc]
    have hpush32s : ∀ (a : ByteArray) (vs : List UInt32), ∃ t, pushU32LEs a vs = a ++ t := by
      intro a vs
      induction vs generalizing a with
      | nil =>
        exact ⟨ByteArray.empty, (byteArray_append_empty a).symm⟩
      | cons v vs' ihv =>
        obtain ⟨t1, h1⟩ := hpush32 a v
        obtain ⟨t2, h2⟩ := ihv (pushU32LE a v)
        refine ⟨t1 ++ t2, ?_⟩
        simp only [pushU32LEs]
        rw [h2, h1, byteArray_append_assoc]
    obtain ⟨tHdr, hHdr⟩ := hpush32 bs hdr
    obtain ⟨tVals, hVals⟩ := hpush32s (pushU32LE bs hdr) vals.toList
    refine ⟨tHdr ++ (tVals ++ tail2), ?_⟩
    rw [hout, hVals, hHdr, byteArray_append_assoc, byteArray_append_assoc]

/-- Whole-task record-sequence round-trip soundness: encoding ANY list of valid `RegRecord`s
    via `encodeRegRecord` + `pushU32LE` and decoding `[bs.size, outBs.size)` via production
    `decodeTaskRecords` (called by `decodeTaskStream`) succeeds and recovers EVERY encoded write
    in exact order, proving the decoder/verifier can never silently drop an emitted register write. -/
theorem decodeTaskRecords_roundTrip
    (bs suffix : ByteArray) (recs : List RegRecord)
    (outBs : ByteArray) (ws : List DecodedWrite) (acc : Array DecodedWrite)
    (henc : encodeRecordList bs recs = .ok (outBs, ws)) :
    decodeTaskRecords (outBs ++ suffix) bs.size outBs.size acc = .ok (acc ++ ws.toArray) := by
  induction recs generalizing bs outBs ws acc with
  | nil =>
    simp only [encodeRecordList, Except.ok.injEq, Prod.mk.injEq] at henc
    rcases henc with ⟨rfl, rfl⟩
    unfold decodeTaskRecords
    simp
  | cons r rest ih =>
    unfold encodeRecordList at henc
    obtain ⟨⟨hdr, vals⟩, hr, henc⟩ := except_bind_ok henc
    obtain ⟨⟨outBs', restWs⟩, hrest, henc⟩ := except_bind_ok henc
    simp only [Except.ok.injEq, Prod.mk.injEq] at henc
    rcases henc with ⟨rfl, rfl⟩
    let bs' := pushU32LEs (pushU32LE bs hdr) vals.toList
    obtain ⟨tailRest, houtEq⟩ := encodeRecordList_append_split bs' rest outBs' restWs hrest
    have hrecLenPos : 1 ≤ vals.size := by
      cases r with
      | sequential base barSlot values =>
        obtain ⟨hveq, h1, _⟩ := encodeRegRecord_sequential_header base barSlot values hdr vals hr
        rw [hveq]; exact h1
      | bitmask base mask values =>
        obtain ⟨hveq, hsz, _⟩ := encodeRegRecord_bitmask_header base mask values hdr vals hr
        rw [hveq, hsz]; omega
    have hbs'Sz : bs'.size = bs.size + 4 + 4 * vals.size := by
      simp [bs', pushU32LEs_size, pushU32LE_size]
    have houtSz : outBs'.size = bs'.size + tailRest.size := by
      rw [houtEq, ByteArray.size_append]
    have hlt : bs.size < outBs'.size := by omega
    have hstep : decodeRegRecordAt (outBs' ++ suffix) bs.size acc =
        .ok (acc ++ (expectedRecordWrites bs.size r).toArray, bs'.size) := by
      have happAssoc : outBs' ++ suffix = bs' ++ (tailRest ++ suffix) := by
        rw [houtEq, byteArray_append_assoc]
      rw [happAssoc]
      cases r with
      | sequential base barSlot values =>
        have hseq := decodeRegRecordAt_sequential_roundTrip bs (tailRest ++ suffix) acc
          base barSlot values hdr vals hr
        obtain ⟨hveq, _⟩ := encodeRegRecord_sequential_header base barSlot values hdr vals hr
        rw [hbs'Sz, hveq]
        simpa [bs', hveq, expectedRecordWrites] using hseq
      | bitmask base mask values =>
        have hbm := decodeRegRecordAt_bitmask_roundTrip bs (tailRest ++ suffix) acc
          base mask values hdr vals hr
        obtain ⟨hveq, hsz, _⟩ := encodeRegRecord_bitmask_header base mask values hdr vals hr
        have hlenAddrs : (decodeBitmaskAddrs base.toNat mask.toNat).length = values.size := by
          rw [decodeBitmaskAddrs_length, ← hsz]
        have hlistEq :
            (List.ofFn (n := values.size) fun j => {
              byteAddr := (decodeBitmaskAddrs base.toNat mask.toNat)[j.val]'(by omega)
              value := values[j.val]
              barSlot := none
              payloadOffset := bs.size + 4 + 4 * j.val
              recordLen := values.size
            }) = expectedRecordWrites bs.size (.bitmask base mask values) := by
          unfold expectedRecordWrites
          apply List.ext_getElem (by simp)
          intro j hj1 hj2
          simp only [List.getElem_ofFn]
          have hjLt : j < (decodeBitmaskAddrs base.toNat mask.toNat).length := by simp at hj1; omega
          have hgetD : (decodeBitmaskAddrs base.toNat mask.toNat).getD j 0 =
              (decodeBitmaskAddrs base.toNat mask.toNat)[j]'hjLt := by
            rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hjLt]; rfl
          rw [hgetD]
        rw [hbs'Sz, hveq]
        simpa [bs', hveq, hlistEq] using hbm
    unfold decodeTaskRecords
    simp only [hlt, ↓reduceDIte, hstep]
    have hgt : bs.size < bs'.size := by omega
    simp only [hgt, ↓reduceDIte]
    rw [ih bs' outBs' restWs (acc ++ (expectedRecordWrites bs.size r).toArray) hrest]
    congr 1
    apply Array.ext'
    simp

-- ============================================================================
-- 18. Production WordLayout Completeness (remaining 16/37 layouts -> 100% coverage)
-- ============================================================================

theorem dim16_roundTrip (name : String) (val : UInt32) (hv : val < 65536) :
    ∃ w, encodeWord (dim16Layout name) #[("val", val)] = .ok w ∧
         decodeFieldChecked (dim16Layout name) w "val" = .ok val := by
  have h1 : ¬ val.toNat ≥ 65536 := u32_not_ge_of_lt hv
  have hlay : validateWordLayout (dim16Layout name) = .ok 0x0000ffff := by
    simp [validateWordLayout, dim16Layout, validateWordLayoutFields, BitField.mask, BitField.maskNat]
  have hval : validateEncodeValues name (dim16Layout name).fields.toList [("val", val)] [] = .ok () := by
    simp [dim16Layout, validateEncodeValues]
  have henc : encodeWord (dim16Layout name) #[("val", val)] = .ok ((0 : UInt32) ||| (val <<< 0)) := by
    unfold encodeWord
    rw [hlay]; simp only [bind, Except.bind]
    have hval' : validateEncodeValues (dim16Layout name).name (dim16Layout name).fields.toList
        #[("val", val)].toList [] = .ok () := hval
    rw [hval']
    simp [dim16Layout, encodeWordFields, List.find?]
    omega
  exact ⟨_, henc, encodeWord_decodeFieldChecked _ _ _ "val" val henc rfl⟩

theorem u32_roundTrip (name : String) (val : UInt32) :
    ∃ w, encodeWord (u32Layout name) #[("val", val)] = .ok w ∧
         decodeFieldChecked (u32Layout name) w "val" = .ok val := by
  have hlay : validateWordLayout (u32Layout name) = .ok 0xffffffff := by
    simp [validateWordLayout, u32Layout, validateWordLayoutFields, BitField.mask, BitField.maskNat]
  have hval : validateEncodeValues name (u32Layout name).fields.toList [("val", val)] [] = .ok () := by
    simp [u32Layout, validateEncodeValues]
  have henc : encodeWord (u32Layout name) #[("val", val)] = .ok ((0 : UInt32) ||| (val <<< 0)) := by
    unfold encodeWord
    rw [hlay]; simp only [bind, Except.bind]
    have hval' : validateEncodeValues (u32Layout name).name (u32Layout name).fields.toList
        #[("val", val)].toList [] = .ok () := hval
    rw [hval']
    simp [u32Layout, encodeWordFields, List.find?]
    exact Nat.lt_of_lt_of_eq val.toNat_lt rfl
  exact ⟨_, henc, encodeWord_decodeFieldChecked _ _ _ "val" val henc rfl⟩

theorem dwGroups_roundTrip (perGroupIn : UInt32) (h : perGroupIn < 65536) :
    ∃ w, encodeWord dwGroupsLayout #[("perGroupIn", perGroupIn)] = .ok w ∧
         decodeFieldChecked dwGroupsLayout w "perGroupIn" = .ok perGroupIn := by
  have h1 : ¬ perGroupIn.toNat ≥ 65536 := u32_not_ge_of_lt h
  have henc : encodeWord dwGroupsLayout #[("perGroupIn", perGroupIn)] =
      .ok ((0x00004000 : UInt32) ||| (perGroupIn <<< 16)) := by
    change (if perGroupIn.toNat ≥ 65536 then _ else _) = _
    simp only [h1, ↓reduceIte]; rfl
  exact ⟨_, henc, encodeWord_decodeFieldChecked _ _ _ "perGroupIn" perGroupIn henc rfl⟩

theorem peCommon_roundTrip (mode : UInt32) (hm : mode < 256) :
    ∃ w, encodeWord peCommonLayout #[("mode", mode)] = .ok w ∧
         decodeFieldChecked peCommonLayout w "mode" = .ok mode := by
  have h1 : ¬ mode.toNat ≥ 256 := u32_not_ge_of_lt hm
  have henc : encodeWord peCommonLayout #[("mode", mode)] =
      .ok ((0 : UInt32) ||| (mode <<< 0)) := by
    change (if mode.toNat ≥ 256 then _ else _) = _
    simp only [h1, ↓reduceIte]; rfl
  exact ⟨_, henc, encodeWord_decodeFieldChecked _ _ _ "mode" mode henc rfl⟩

theorem neActShiftEn_roundTrip (enable : UInt32) (he : enable < 2) :
    ∃ w, encodeWord neActShiftEnLayout #[("enable", enable)] = .ok w ∧
         decodeFieldChecked neActShiftEnLayout w "enable" = .ok enable := by
  have h1 : ¬ enable.toNat ≥ 2 := u32_not_ge_of_lt he
  have henc : encodeWord neActShiftEnLayout #[("enable", enable)] =
      .ok ((0 : UInt32) ||| (enable <<< 0)) := by
    change (if enable.toNat ≥ 2 then _ else _) = _
    simp only [h1, ↓reduceIte]; rfl
  exact ⟨_, henc, encodeWord_decodeFieldChecked _ _ _ "enable" enable henc rfl⟩

theorem src1Cfg_roundTrip (mode p : UInt32) (hmode : mode < 4) (hp : p < 16) :
    ∃ w, encodeWord src1CfgLayout #[("mode", mode), ("P", p)] = .ok w ∧
         decodeFieldChecked src1CfgLayout w "mode" = .ok mode ∧
         decodeFieldChecked src1CfgLayout w "P" = .ok p := by
  have h1 : ¬ mode.toNat ≥ 4 := u32_not_ge_of_lt hmode
  have h2 : ¬ p.toNat ≥ 16 := u32_not_ge_of_lt hp
  have henc : encodeWord src1CfgLayout #[("mode", mode), ("P", p)] =
      .ok (((0x00500070 : UInt32) ||| (mode <<< 0)) ||| (p <<< 8)) := by
    change (if mode.toNat ≥ 4 then _ else if p.toNat ≥ 16 then _ else _) = _
    simp only [h1, h2, ↓reduceIte]; rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "mode" mode henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "P" p henc rfl⟩

theorem portCfgMulti_roundTrip (name : String)
    (mode handoffNe handoffPe waitNe4 waitNe5 dmaFlag p waitPe20 waitPe22 : UInt32)
    (hmode : mode < 4) (hhn : handoffNe < 2) (hhp : handoffPe < 2)
    (hwn4 : waitNe4 < 2) (hwn5 : waitNe5 < 2) (hdf : dmaFlag < 2)
    (hp : p < 16) (hwp20 : waitPe20 < 2) (hwp22 : waitPe22 < 2) :
    ∃ w, encodeWord (portCfgMultiLayout name) #[
            ("mode", mode), ("handoffNe", handoffNe), ("handoffPe", handoffPe),
            ("waitNe4", waitNe4), ("waitNe5", waitNe5), ("dmaFlag", dmaFlag),
            ("P", p), ("waitPe20", waitPe20), ("waitPe22", waitPe22)
          ] = .ok w ∧
         decodeFieldChecked (portCfgMultiLayout name) w "mode" = .ok mode ∧
         decodeFieldChecked (portCfgMultiLayout name) w "handoffNe" = .ok handoffNe ∧
         decodeFieldChecked (portCfgMultiLayout name) w "handoffPe" = .ok handoffPe ∧
         decodeFieldChecked (portCfgMultiLayout name) w "waitNe4" = .ok waitNe4 ∧
         decodeFieldChecked (portCfgMultiLayout name) w "waitNe5" = .ok waitNe5 ∧
         decodeFieldChecked (portCfgMultiLayout name) w "dmaFlag" = .ok dmaFlag ∧
         decodeFieldChecked (portCfgMultiLayout name) w "P" = .ok p ∧
         decodeFieldChecked (portCfgMultiLayout name) w "waitPe20" = .ok waitPe20 ∧
         decodeFieldChecked (portCfgMultiLayout name) w "waitPe22" = .ok waitPe22 := by
  have c1 : ¬ 4 ≤ mode.toNat := u32_not_ge_of_lt hmode
  have c2 : ¬ 2 ≤ handoffNe.toNat := u32_not_ge_of_lt hhn
  have c3 : ¬ 2 ≤ handoffPe.toNat := u32_not_ge_of_lt hhp
  have c4 : ¬ 2 ≤ waitNe4.toNat := u32_not_ge_of_lt hwn4
  have c5 : ¬ 2 ≤ waitNe5.toNat := u32_not_ge_of_lt hwn5
  have c6 : ¬ 2 ≤ dmaFlag.toNat := u32_not_ge_of_lt hdf
  have c7 : ¬ 16 ≤ p.toNat := u32_not_ge_of_lt hp
  have c8 : ¬ 2 ≤ waitPe20.toNat := u32_not_ge_of_lt hwp20
  have c9 : ¬ 2 ≤ waitPe22.toNat := u32_not_ge_of_lt hwp22
  have hlay : validateWordLayout (portCfgMultiLayout name) = .ok 0x00500f7f := by
    simp [validateWordLayout, portCfgMultiLayout, validateWordLayoutFields, BitField.mask, BitField.maskNat]
    rfl
  have hval : validateEncodeValues name (portCfgMultiLayout name).fields.toList
      [("mode", mode), ("handoffNe", handoffNe), ("handoffPe", handoffPe),
       ("waitNe4", waitNe4), ("waitNe5", waitNe5), ("dmaFlag", dmaFlag),
       ("P", p), ("waitPe20", waitPe20), ("waitPe22", waitPe22)] [] = .ok () := by
    simp [portCfgMultiLayout, validateEncodeValues]
  have henc : encodeWord (portCfgMultiLayout name) #[
      ("mode", mode), ("handoffNe", handoffNe), ("handoffPe", handoffPe),
      ("waitNe4", waitNe4), ("waitNe5", waitNe5), ("dmaFlag", dmaFlag),
      ("P", p), ("waitPe20", waitPe20), ("waitPe22", waitPe22)
    ] = .ok ((((((((((0 : UInt32) ||| (mode <<< 0)) ||| (handoffNe <<< 2)) ||| (handoffPe <<< 3)) |||
      (waitNe4 <<< 4)) ||| (waitNe5 <<< 5)) ||| (dmaFlag <<< 6)) ||| (p <<< 8)) |||
      (waitPe20 <<< 20)) ||| (waitPe22 <<< 22)) := by
    unfold encodeWord
    rw [hlay]; simp only [bind, Except.bind]
    have hval' : validateEncodeValues (portCfgMultiLayout name).name (portCfgMultiLayout name).fields.toList
        #[("mode", mode), ("handoffNe", handoffNe), ("handoffPe", handoffPe),
          ("waitNe4", waitNe4), ("waitNe5", waitNe5), ("dmaFlag", dmaFlag),
          ("P", p), ("waitPe20", waitPe20), ("waitPe22", waitPe22)].toList [] = .ok () := hval
    rw [hval']
    simp [portCfgMultiLayout, encodeWordFields, List.find?, c1, c2, c3, c4, c5, c6, c7, c8, c9]
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "mode" mode henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "handoffNe" handoffNe henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "handoffPe" handoffPe henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "waitNe4" waitNe4 henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "waitNe5" waitNe5 henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "dmaFlag" dmaFlag henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "P" p henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "waitPe20" waitPe20 henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "waitPe22" waitPe22 henc rfl⟩

theorem resCfgMulti_roundTrip
    (mode direct waitNe4 waitNe5 dmaFlag p waitPe20 waitPe22 : UInt32)
    (hmode : mode < 4) (hdir : direct < 2) (hwn4 : waitNe4 < 2) (hwn5 : waitNe5 < 2)
    (hdf : dmaFlag < 2) (hp : p < 16) (hwp20 : waitPe20 < 2) (hwp22 : waitPe22 < 2) :
    ∃ w, encodeWord resCfgMultiLayout #[
            ("mode", mode), ("direct", direct), ("waitNe4", waitNe4), ("waitNe5", waitNe5),
            ("dmaFlag", dmaFlag), ("P", p), ("waitPe20", waitPe20), ("waitPe22", waitPe22)
          ] = .ok w ∧
         decodeFieldChecked resCfgMultiLayout w "mode" = .ok mode ∧
         decodeFieldChecked resCfgMultiLayout w "direct" = .ok direct ∧
         decodeFieldChecked resCfgMultiLayout w "waitNe4" = .ok waitNe4 ∧
         decodeFieldChecked resCfgMultiLayout w "waitNe5" = .ok waitNe5 ∧
         decodeFieldChecked resCfgMultiLayout w "dmaFlag" = .ok dmaFlag ∧
         decodeFieldChecked resCfgMultiLayout w "P" = .ok p ∧
         decodeFieldChecked resCfgMultiLayout w "waitPe20" = .ok waitPe20 ∧
         decodeFieldChecked resCfgMultiLayout w "waitPe22" = .ok waitPe22 := by
  have c1 : ¬ mode.toNat ≥ 4 := u32_not_ge_of_lt hmode
  have c2 : ¬ direct.toNat ≥ 2 := u32_not_ge_of_lt hdir
  have c3 : ¬ waitNe4.toNat ≥ 2 := u32_not_ge_of_lt hwn4
  have c4 : ¬ waitNe5.toNat ≥ 2 := u32_not_ge_of_lt hwn5
  have c5 : ¬ dmaFlag.toNat ≥ 2 := u32_not_ge_of_lt hdf
  have c6 : ¬ p.toNat ≥ 16 := u32_not_ge_of_lt hp
  have c7 : ¬ waitPe20.toNat ≥ 2 := u32_not_ge_of_lt hwp20
  have c8 : ¬ waitPe22.toNat ≥ 2 := u32_not_ge_of_lt hwp22
  have henc : encodeWord resCfgMultiLayout #[
      ("mode", mode), ("direct", direct), ("waitNe4", waitNe4), ("waitNe5", waitNe5),
      ("dmaFlag", dmaFlag), ("P", p), ("waitPe20", waitPe20), ("waitPe22", waitPe22)
    ] = .ok (((((((((0 : UInt32) ||| (mode <<< 0)) ||| (direct <<< 3)) ||| (waitNe4 <<< 4)) |||
      (waitNe5 <<< 5)) ||| (dmaFlag <<< 6)) ||| (p <<< 8)) |||
      (waitPe20 <<< 20)) ||| (waitPe22 <<< 22)) := by
    change (if mode.toNat ≥ 4 then _ else if direct.toNat ≥ 2 then _ else
            if waitNe4.toNat ≥ 2 then _ else if waitNe5.toNat ≥ 2 then _ else
            if dmaFlag.toNat ≥ 2 then _ else if p.toNat ≥ 16 then _ else
            if waitPe20.toNat ≥ 2 then _ else if waitPe22.toNat ≥ 2 then _ else _) = _
    simp only [c1, c2, c3, c4, c5, c6, c7, c8, ↓reduceIte]; rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "mode" mode henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "direct" direct henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "waitNe4" waitNe4 henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "waitNe5" waitNe5 henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "dmaFlag" dmaFlag henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "P" p henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "waitPe20" waitPe20 henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "waitPe22" waitPe22 henc rfl⟩

theorem scalarBias_roundTrip (val : UInt32) (hv : val < 65536) :
    ∃ w, encodeWord scalarBiasLayout #[("val", val)] = .ok w ∧
         decodeFieldChecked scalarBiasLayout w "val" = .ok val :=
  dim16_roundTrip "scalarBias" val hv

theorem neScale_roundTrip (val : UInt32) (hv : val < 65536) :
    ∃ w, encodeWord neScaleLayout #[("val", val)] = .ok w ∧
         decodeFieldChecked neScaleLayout w "val" = .ok val :=
  dim16_roundTrip "neScale" val hv

theorem src1Dep_roundTrip (code task : UInt32) (hc : code < 256) (ht : task < 256) :
    ∃ w, encodeWord src1DepLayout #[("code", code), ("task", task)] = .ok w ∧
         decodeFieldChecked src1DepLayout w "code" = .ok code ∧
         decodeFieldChecked src1DepLayout w "task" = .ok task := by
  have h1 : ¬ code.toNat ≥ 256 := u32_not_ge_of_lt hc
  have h2 : ¬ task.toNat ≥ 256 := u32_not_ge_of_lt ht
  have henc : encodeWord src1DepLayout #[("code", code), ("task", task)] =
      .ok (((0 : UInt32) ||| (code <<< 0)) ||| (task <<< 16)) := by
    change (if code.toNat ≥ 256 then _ else if task.toNat ≥ 256 then _ else _) = _
    simp only [h1, h2, ↓reduceIte]; rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "code" code henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "task" task henc rfl⟩

theorem dmaFmt_roundTrip (name : String) (fmt p : UInt32)
    (hf : fmt < 16777216) (hp : p < 256) :
    ∃ w, encodeWord (dmaFmtLayout name) #[("format", fmt), ("P", p)] = .ok w ∧
         decodeFieldChecked (dmaFmtLayout name) w "format" = .ok fmt ∧
         decodeFieldChecked (dmaFmtLayout name) w "P" = .ok p := by
  have c1 : ¬ 16777216 ≤ fmt.toNat := u32_not_ge_of_lt hf
  have c2 : ¬ 256 ≤ p.toNat := u32_not_ge_of_lt hp
  have hlay : validateWordLayout (dmaFmtLayout name) = .ok 0xffffffff := by
    simp [validateWordLayout, dmaFmtLayout, validateWordLayoutFields, BitField.mask, BitField.maskNat]
    have h0 : ((4278190080 : UInt32) &&& (16777215 : UInt32)) = 0 := by decide
    simp [h0]; rfl
  have hval : validateEncodeValues name (dmaFmtLayout name).fields.toList
      [("format", fmt), ("P", p)] [] = .ok () := by
    simp [dmaFmtLayout, validateEncodeValues]
  have henc : encodeWord (dmaFmtLayout name) #[("format", fmt), ("P", p)] =
      .ok (((0 : UInt32) ||| (fmt <<< 0)) ||| (p <<< 24)) := by
    unfold encodeWord
    rw [hlay]; simp only [bind, Except.bind]
    have hval' : validateEncodeValues (dmaFmtLayout name).name (dmaFmtLayout name).fields.toList
        #[("format", fmt), ("P", p)].toList [] = .ok () := hval
    rw [hval']
    simp [dmaFmtLayout, encodeWordFields, List.find?, c1, c2]
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "format" fmt henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "P" p henc rfl⟩

theorem resDep_roundTrip (code task direct : UInt32)
    (hc : code < 256) (ht : task < 256) (hd : direct < 2) :
    ∃ w, encodeWord resDepLayout #[("code", code), ("task", task), ("direct", direct)] = .ok w ∧
         decodeFieldChecked resDepLayout w "code" = .ok code ∧
         decodeFieldChecked resDepLayout w "task" = .ok task ∧
         decodeFieldChecked resDepLayout w "direct" = .ok direct := by
  have h1 : ¬ code.toNat ≥ 256 := u32_not_ge_of_lt hc
  have h2 : ¬ task.toNat ≥ 256 := u32_not_ge_of_lt ht
  have h3 : ¬ direct.toNat ≥ 2 := u32_not_ge_of_lt hd
  have henc : encodeWord resDepLayout #[("code", code), ("task", task), ("direct", direct)] =
      .ok ((((0 : UInt32) ||| (code <<< 0)) ||| (task <<< 16)) ||| (direct <<< 24)) := by
    change (if code.toNat ≥ 256 then _ else if task.toNat ≥ 256 then _ else if direct.toNat ≥ 2 then _ else _) = _
    simp only [h1, h2, h3, ↓reduceIte]; rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "code" code henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "task" task henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "direct" direct henc rfl⟩

theorem chunkSlot_roundTrip (en on20 task : UInt32)
    (he : en < 2) (ho : on20 < 2) (ht : task < 256) :
    ∃ w, encodeWord chunkSlotLayout #[("enable", en), ("on20", on20), ("task", task)] = .ok w ∧
         decodeFieldChecked chunkSlotLayout w "enable" = .ok en ∧
         decodeFieldChecked chunkSlotLayout w "on20" = .ok on20 ∧
         decodeFieldChecked chunkSlotLayout w "task" = .ok task := by
  have h1 : ¬ en.toNat ≥ 2 := u32_not_ge_of_lt he
  have h2 : ¬ on20.toNat ≥ 2 := u32_not_ge_of_lt ho
  have h3 : ¬ task.toNat ≥ 256 := u32_not_ge_of_lt ht
  have henc : encodeWord chunkSlotLayout #[("enable", en), ("on20", on20), ("task", task)] =
      .ok ((((0 : UInt32) ||| (en <<< 0)) ||| (on20 <<< 5)) ||| (task <<< 16)) := by
    change (if en.toNat ≥ 2 then _ else if on20.toNat ≥ 2 then _ else if task.toNat ≥ 256 then _ else _) = _
    simp only [h1, h2, h3, ↓reduceIte]; rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "enable" en henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "on20" on20 henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "task" task henc rfl⟩

theorem out2_roundTrip (form task : UInt32) (hf : form < 65536) (ht : task < 256) :
    ∃ w, encodeWord out2Layout #[("form", form), ("task", task)] = .ok w ∧
         decodeFieldChecked out2Layout w "form" = .ok form ∧
         decodeFieldChecked out2Layout w "task" = .ok task := by
  have h1 : ¬ form.toNat ≥ 65536 := u32_not_ge_of_lt hf
  have h2 : ¬ task.toNat ≥ 256 := u32_not_ge_of_lt ht
  have henc : encodeWord out2Layout #[("form", form), ("task", task)] =
      .ok (((0 : UInt32) ||| (form <<< 0)) ||| (task <<< 16)) := by
    change (if form.toNat ≥ 65536 then _ else if task.toNat ≥ 256 then _ else _) = _
    simp only [h1, h2, ↓reduceIte]; rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "form" form henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "task" task henc rfl⟩

theorem neCfg_roundTrip (target : Target) (sparse : UInt32) (hs : sparse < 2) :
    ∃ w, encodeWord (neCfgLayout target) #[("sparse", sparse)] = .ok w ∧
         decodeFieldChecked (neCfgLayout target) w "sparse" = .ok sparse := by
  have h1 : ¬ sparse.toNat ≥ 2 := u32_not_ge_of_lt hs
  cases target with
  | h16 =>
    have henc : encodeWord (neCfgLayout .h16) #[("sparse", sparse)] =
        .ok ((neCfgValue .h16) ||| (sparse <<< 8)) := by
      change (if sparse.toNat ≥ 2 then _ else _) = _
      simp only [h1, ↓reduceIte]; rfl
    exact ⟨_, henc, encodeWord_decodeFieldChecked _ _ _ "sparse" sparse henc rfl⟩
  | h17 =>
    have henc : encodeWord (neCfgLayout .h17) #[("sparse", sparse)] =
        .ok ((neCfgValue .h17) ||| (sparse <<< 8)) := by
      change (if sparse.toNat ≥ 2 then _ else _) = _
      simp only [h1, ↓reduceIte]; rfl
    exact ⟨_, henc, encodeWord_decodeFieldChecked _ _ _ "sparse" sparse henc rfl⟩

-- ============================================================================
-- 19. Whole-Program S1–S5 Hazard Verifier Soundness (Vane.Verify + Vane.Reach)
-- ============================================================================

/-- `vIntervalsOverlap` is symmetric. -/
theorem vIntervalsOverlap_symm (lo1 hi1 lo2 hi2 : Nat) :
    vIntervalsOverlap lo1 hi1 lo2 hi2 = vIntervalsOverlap lo2 hi2 lo1 hi1 := by
  unfold vIntervalsOverlap
  exact Bool.and_comm _ _

/-- `vIntervalsOverlap` is irreflexive on empty intervals (`hi ≤ lo`). -/
theorem vIntervalsOverlap_irrefl_empty (lo hi : Nat) (hempty : hi ≤ lo) :
    vIntervalsOverlap lo hi lo hi = false := by
  unfold vIntervalsOverlap
  have h : ¬ (lo < hi) := by omega
  simp [h]

/-- Whenever two half-open intervals `[lo1, hi1)` and `[lo2, hi2)` share any byte address `a`,
    `vIntervalsOverlap lo1 hi1 lo2 hi2 = true` unconditionally. -/
theorem vIntervalsOverlap_of_witness (lo1 hi1 lo2 hi2 : Nat)
    (hw : ∃ a, lo1 ≤ a ∧ a < hi1 ∧ lo2 ≤ a ∧ a < hi2) :
    vIntervalsOverlap lo1 hi1 lo2 hi2 = true := by
  rcases hw with ⟨a, h1, h2, h3, h4⟩
  unfold vIntervalsOverlap
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  omega

/-- Exact witness characterization for non-empty intervals: `vIntervalsOverlap lo1 hi1 lo2 hi2 = true`
    iff there exists a byte address `a` belonging to both `[lo1, hi1)` and `[lo2, hi2)`. -/
theorem vIntervalsOverlap_iff_witness (lo1 hi1 lo2 hi2 : Nat)
    (hne1 : lo1 < hi1) (hne2 : lo2 < hi2) :
    vIntervalsOverlap lo1 hi1 lo2 hi2 = true ↔
      ∃ a, lo1 ≤ a ∧ a < hi1 ∧ lo2 ≤ a ∧ a < hi2 := by
  constructor
  · intro h
    unfold vIntervalsOverlap at h
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    exact ⟨max lo1 lo2, by omega, by omega, by omega, by omega⟩
  · exact vIntervalsOverlap_of_witness lo1 hi1 lo2 hi2

/-- Contrapositive soundness: if `vIntervalsOverlap` returns `false`, no byte address `a`
    lies inside both `[lo1, hi1)` and `[lo2, hi2)`. -/
theorem no_shared_byte_of_not_vIntervalsOverlap (lo1 hi1 lo2 hi2 : Nat)
    (h : vIntervalsOverlap lo1 hi1 lo2 hi2 = false) :
    ∀ a, ¬ (lo1 ≤ a ∧ a < hi1 ∧ lo2 ≤ a ∧ a < hi2) := by
  intro a ha
  have ht := vIntervalsOverlap_of_witness lo1 hi1 lo2 hi2 ⟨a, ha⟩
  rw [h] at ht
  cases ht

/-- Every direct whole-task happens-before edge `stepHB fps j i` is strictly forward (`j < i`). -/
theorem stepHB_fwd (fps : Array VerifiedTaskFootprint) (j i : Nat)
    (h : stepHB fps j i = true) : j < i := by
  unfold stepHB at h
  split at h
  · cases h
  · rename_i hc
    simp only [Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not, not_or,
               Classical.not_not] at hc
    exact hc.1

/-- Every direct L2 happens-before edge `stepHBL fps j i` is strictly forward (`j < i`). -/
theorem stepHBL_fwd (fps : Array VerifiedTaskFootprint) (j i : Nat)
    (h : stepHBL fps j i = true) : j < i := by
  unfold stepHBL at h
  split at h
  · cases h
  · rename_i hc
    simp only [Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not, not_or,
               Classical.not_not] at hc
    exact hc.1

/-- Every direct whole-task HB edge implies an L2 HBL edge. -/
theorem stepHB_imp_stepHBL (fps : Array VerifiedTaskFootprint) (j i : Nat)
    (h : stepHB fps j i = true) : stepHBL fps j i = true := by
  have hlt : j < i ∧ i < fps.size := by
    unfold stepHB at h
    split at h
    · cases h
    · rename_i hc
      simp only [Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not, not_or,
                 Classical.not_not] at hc
      exact hc
  unfold stepHBL
  have hnot : (!(decide (j < i) && decide (i < fps.size))) = false := by simp [hlt.1, hlt.2]
  rw [hnot]
  simp [h]

/-- Formal whole-task happens-before relation on task indices: transitive closure of `stepHB`. -/
def HappensBefore (fps : Array VerifiedTaskFootprint) (k i : Nat) : Prop :=
  Relation.TransGen (fun a b => stepHB fps a b = true) k i

/-- Formal L2-level happens-before relation on task indices: transitive closure of `stepHBL`. -/
def HappensBeforeL2 (fps : Array VerifiedTaskFootprint) (k i : Nat) : Prop :=
  Relation.TransGen (fun a b => stepHBL fps a b = true) k i

/-- Reflexive-transitive whole-task happens-before (`k = i ∨ HappensBefore fps k i`). -/
def HappensBeforeEq (fps : Array VerifiedTaskFootprint) (k i : Nat) : Prop :=
  k = i ∨ HappensBefore fps k i

/-- Reflexive-transitive L2-level happens-before (`k = i ∨ HappensBeforeL2 fps k i`). -/
def HappensBeforeL2Eq (fps : Array VerifiedTaskFootprint) (k i : Nat) : Prop :=
  k = i ∨ HappensBeforeL2 fps k i

/-- `HappensBefore` strictly respects program order (`k < i`). -/
theorem happensBefore_lt (fps : Array VerifiedTaskFootprint) {k i : Nat}
    (h : HappensBefore fps k i) : k < i :=
  transGen_lt_of_fwd (stepHB_fwd fps) h

/-- `HappensBefore` is irreflexive. -/
theorem happensBefore_irrefl (fps : Array VerifiedTaskFootprint) (i : Nat) :
    ¬ HappensBefore fps i i := fun h =>
  Nat.lt_irrefl i (happensBefore_lt fps h)

/-- `HappensBefore` is asymmetric. -/
theorem happensBefore_asymm (fps : Array VerifiedTaskFootprint) {a b : Nat}
    (hab : HappensBefore fps a b) : ¬ HappensBefore fps b a := fun hba =>
  Nat.lt_asymm (happensBefore_lt fps hab) (happensBefore_lt fps hba)

/-- `HappensBefore` is transitive. -/
theorem happensBefore_trans (fps : Array VerifiedTaskFootprint) {a b c : Nat}
    (hab : HappensBefore fps a b) (hbc : HappensBefore fps b c) :
    HappensBefore fps a c :=
  Relation.TransGen.trans hab hbc

/-- Whole-task `HappensBefore` implies L2-level `HappensBeforeL2`. -/
theorem happensBefore_imp_l2 (fps : Array VerifiedTaskFootprint) {k i : Nat}
    (h : HappensBefore fps k i) : HappensBeforeL2 fps k i := by
  induction h with
  | single hstep => exact Relation.TransGen.single (stepHB_imp_stepHBL fps _ _ hstep)
  | tail _ hstep ih => exact Relation.TransGen.tail ih (stepHB_imp_stepHBL fps _ _ hstep)

private theorem vLatestEngArraysAux_spec
    (fps : Array VerifiedTaskFootprint) (n : Nat) :
    ∀ rem i lastNe lastPe neArr peArr,
      rem = n - i →
      i ≤ n →
      neArr.size = i →
      peArr.size = i →
      lastNe = vLatestEngBefore fps .ne i →
      lastPe = vLatestEngBefore fps .pe i →
      (∀ j < i, neArr[j]! = vLatestEngBefore fps .ne j) →
      (∀ j < i, peArr[j]! = vLatestEngBefore fps .pe j) →
      let res := vLatestEngArraysAux fps n i lastNe lastPe neArr peArr
      res.1.size = n ∧ res.2.size = n ∧
      (∀ j < n, res.1[j]! = vLatestEngBefore fps .ne j) ∧
      (∀ j < n, res.2[j]! = vLatestEngBefore fps .pe j) := by
  intro rem
  induction rem with
  | zero =>
    intro i lastNe lastPe neArr peArr hrem hiLe hneSz hpeSz _ _ hnePre hpePre
    have hiEq : i = n := by omega
    subst hiEq
    unfold vLatestEngArraysAux
    simp only [Nat.lt_irrefl, ↓reduceDIte]
    exact ⟨hneSz, hpeSz, hnePre, hpePre⟩
  | succ r ih =>
    intro i lastNe lastPe neArr peArr hrem hiLe hneSz hpeSz hlastNe hlastPe hnePre hpePre
    have hlt : i < n := by omega
    unfold vLatestEngArraysAux
    simp only [hlt, ↓reduceDIte]
    have hnePre' : ∀ j < i + 1, (neArr.push lastNe)[j]! = vLatestEngBefore fps .ne j := by
      intro j hj
      by_cases hji : j < i
      · have h1 : j < (neArr.push lastNe).size := by rw [Array.size_push]; omega
        have h2 : j < neArr.size := by omega
        rw [getElem!_pos (neArr.push lastNe) j h1, Array.getElem_push_lt h2,
            ← getElem!_pos neArr j h2]
        exact hnePre j hji
      · have hjEq : j = i := by omega
        subst j
        have h1 : i < (neArr.push lastNe).size := by rw [Array.size_push]; omega
        rw [getElem!_pos (neArr.push lastNe) i h1]
        have hgetPush : (neArr.push lastNe)[i] = lastNe := by
          simp [Array.getElem_push, hneSz]
        rw [hgetPush, hlastNe]
    have hpePre' : ∀ j < i + 1, (peArr.push lastPe)[j]! = vLatestEngBefore fps .pe j := by
      intro j hj
      by_cases hji : j < i
      · have h1 : j < (peArr.push lastPe).size := by rw [Array.size_push]; omega
        have h2 : j < peArr.size := by omega
        rw [getElem!_pos (peArr.push lastPe) j h1, Array.getElem_push_lt h2,
            ← getElem!_pos peArr j h2]
        exact hpePre j hji
      · have hjEq : j = i := by omega
        subst j
        have h1 : i < (peArr.push lastPe).size := by rw [Array.size_push]; omega
        rw [getElem!_pos (peArr.push lastPe) i h1]
        have hgetPush : (peArr.push lastPe)[i] = lastPe := by
          simp [Array.getElem_push, hpeSz]
        rw [hgetPush, hlastPe]
    have hnextEq :
        (match fps[i]? with
         | some fp => match fp.engine with | .ne => (some i, lastPe) | .pe => (lastNe, some i)
         | none => (lastNe, lastPe)) =
        (vLatestEngBefore fps .ne (i + 1), vLatestEngBefore fps .pe (i + 1)) := by
      unfold vLatestEngBefore
      cases hget : fps[i]? with
      | none => simp [hlastNe, hlastPe]
      | some fp =>
        cases heng : fp.engine with
        | ne =>
          simp [heng, hlastPe, show (Engine.ne == Engine.ne) = true from rfl,
                show (Engine.ne == Engine.pe) = false from rfl]
        | pe =>
          simp [heng, hlastNe, show (Engine.pe == Engine.pe) = true from rfl,
                show (Engine.pe == Engine.ne) = false from rfl]
    have hneNext :
        (match fps[i]? with
         | some fp => match fp.engine with | .ne => (some i, lastPe) | .pe => (lastNe, some i)
         | none => (lastNe, lastPe)).1 = vLatestEngBefore fps .ne (i + 1) :=
      congrArg Prod.fst hnextEq
    have hpeNext :
        (match fps[i]? with
         | some fp => match fp.engine with | .ne => (some i, lastPe) | .pe => (lastNe, some i)
         | none => (lastNe, lastPe)).2 = vLatestEngBefore fps .pe (i + 1) :=
      congrArg Prod.snd hnextEq
    exact ih (i + 1) _ _ (neArr.push lastNe) (peArr.push lastPe)
      (by omega) (by omega) (by rw [Array.size_push]; omega) (by rw [Array.size_push]; omega)
      hneNext hpeNext hnePre' hpePre'

private theorem vLatestEngArrays_eq (fps : Array VerifiedTaskFootprint) :
    (vLatestEngArrays fps).1.size = fps.size ∧
    (vLatestEngArrays fps).2.size = fps.size ∧
    (∀ i < fps.size, (vLatestEngArrays fps).1[i]! = vLatestEngBefore fps .ne i) ∧
    (∀ i < fps.size, (vLatestEngArrays fps).2[i]! = vLatestEngBefore fps .pe i) := by
  unfold vLatestEngArrays
  exact vLatestEngArraysAux_spec fps fps.size fps.size 0 none none
    (Array.mkEmpty fps.size) (Array.mkEmpty fps.size)
    (by omega) (by omega) (by simp) (by simp) rfl rfl
    (fun j hj => by omega) (fun j hj => by omega)

/-- Soundness and completeness of the production HB reachability table in `buildHBReachTables`:
    for every `i < fps.size`, `reachB (buildHBReachTables fps).1 k i = true` iff `HappensBefore fps k i`. -/
theorem buildHBReachTables_hb_iff (fps : Array VerifiedTaskFootprint) (k i : Nat) (hi : i < fps.size) :
    reachB (buildHBReachTables fps).1 k i = true ↔ HappensBefore fps k i := by
  obtain ⟨_, _, hneArr, hpeArr⟩ := vLatestEngArrays_eq fps
  unfold buildHBReachTables
  dsimp only
  have hstepEq :
      (fun j i =>
        if j < i && i < fps.size then
          ((Array.ofFn (n := fps.size) fun iFin =>
            let i := iFin.val
            let ti := fps[i]!
            let latestPe := (vLatestEngArrays fps).2[i]!
            let latestNe := (vLatestEngArrays fps).1[i]!
            Array.ofFn (n := i) fun jFin =>
              let j := jFin.val
              let tj := fps[j]!
              let poPe := ti.engine == .pe && latestPe == some j
              let poNeComp := ti.engine == .ne && tj.engine == .ne && latestNe == some j && tj.isNeCompute && ti.isNeCompute
              let w5 := tj.engine == .ne && (ti.sync.src1WaitNe5 || (ti.engine == .pe && ti.sync.src2WaitNe5))
              let wPe := tj.engine == .pe && latestPe == some j && (ti.sync.src1WaitPe22 || (ti.engine == .pe && ti.sync.src2WaitPe22))
              let depNonRaw :=
                j + 1 == i && ti.sync.hasSrc1Dma &&
                (ti.sync.src1DepKind == 1 || ti.sync.src1DepKind == 2) &&
                ti.sync.src1DepDist == 0 &&
                !vSrc1DramReadsFromWrite tj ti
              poPe || poNeComp || w5 || wPe || depNonRaw)[i]!)[j]!
        else false) = stepHB fps := by
    funext a b
    unfold stepHB
    by_cases hab : a < b ∧ b < fps.size
    · have hcond : (decide (a < b) && decide (b < fps.size)) = true := by simp [hab.1, hab.2]
      simp only [hcond, ↓reduceIte, Bool.not_true, Bool.false_eq_true]
      have hbSz : b < (Array.ofFn (n := fps.size) fun iFin : Fin fps.size =>
          let i := iFin.val
          let ti := fps[i]!
          let latestPe := (vLatestEngArrays fps).2[i]!
          let latestNe := (vLatestEngArrays fps).1[i]!
          Array.ofFn (n := i) fun jFin : Fin i =>
            let j := jFin.val
            let tj := fps[j]!
            let poPe := ti.engine == .pe && latestPe == some j
            let poNeComp := ti.engine == .ne && tj.engine == .ne && latestNe == some j && tj.isNeCompute && ti.isNeCompute
            let w5 := tj.engine == .ne && (ti.sync.src1WaitNe5 || (ti.engine == .pe && ti.sync.src2WaitNe5))
            let wPe := tj.engine == .pe && latestPe == some j && (ti.sync.src1WaitPe22 || (ti.engine == .pe && ti.sync.src2WaitPe22))
            let depNonRaw :=
              j + 1 == i && ti.sync.hasSrc1Dma &&
              (ti.sync.src1DepKind == 1 || ti.sync.src1DepKind == 2) &&
              ti.sync.src1DepDist == 0 &&
              !vSrc1DramReadsFromWrite tj ti
            poPe || poNeComp || w5 || wPe || depNonRaw).size := by
        rw [Array.size_ofFn]; exact hab.2
      rw [getElem!_pos _ b hbSz, Array.getElem_ofFn]
      have haSz : a < (Array.ofFn (n := b) fun jFin : Fin b =>
          let j := jFin.val
          let tj := fps[j]!
          let poPe := fps[b]!.engine == .pe && (vLatestEngArrays fps).2[b]! == some j
          let poNeComp := fps[b]!.engine == .ne && tj.engine == .ne && (vLatestEngArrays fps).1[b]! == some j && tj.isNeCompute && fps[b]!.isNeCompute
          let w5 := tj.engine == .ne && (fps[b]!.sync.src1WaitNe5 || (fps[b]!.engine == .pe && fps[b]!.sync.src2WaitNe5))
          let wPe := tj.engine == .pe && (vLatestEngArrays fps).2[b]! == some j && (fps[b]!.sync.src1WaitPe22 || (fps[b]!.engine == .pe && fps[b]!.sync.src2WaitPe22))
          let depNonRaw :=
            j + 1 == b && fps[b]!.sync.hasSrc1Dma &&
            (fps[b]!.sync.src1DepKind == 1 || fps[b]!.sync.src1DepKind == 2) &&
            fps[b]!.sync.src1DepDist == 0 &&
            !vSrc1DramReadsFromWrite tj fps[b]!
          poPe || poNeComp || w5 || wPe || depNonRaw).size := by
        rw [Array.size_ofFn]; exact hab.1
      rw [getElem!_pos _ a haSz, Array.getElem_ofFn]
      simp only [hneArr b hab.2, hpeArr b hab.2]
    · have hcond : (decide (a < b) && decide (b < fps.size)) = false := by
        simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]
        by_cases h1 : a < b
        · right; intro h2; exact hab ⟨h1, h2⟩
        · left; exact h1
      simp [hcond]
  rw [hstepEq]
  exact reachB_iff (stepHB fps) (stepHB_fwd fps) fps.size k i hi

/-- Soundness and completeness of the production L2 HBL reachability table in `buildHBReachTables`:
    for every `i < fps.size`, `reachB (buildHBReachTables fps).2 k i = true` iff `HappensBeforeL2 fps k i`. -/
theorem buildHBReachTables_hbl_iff (fps : Array VerifiedTaskFootprint) (k i : Nat) (hi : i < fps.size) :
    reachB (buildHBReachTables fps).2 k i = true ↔ HappensBeforeL2 fps k i := by
  obtain ⟨_, _, hneArr, hpeArr⟩ := vLatestEngArrays_eq fps
  unfold buildHBReachTables
  dsimp only
  have hstepEq :
      (fun j i =>
        if j < i && i < fps.size then
          ((Array.ofFn (n := fps.size) fun iFin =>
            let i := iFin.val
            let ti := fps[i]!
            let latestNe := (vLatestEngArrays fps).1[i]!
            let hbRow := (Array.ofFn (n := fps.size) fun iFin =>
              let i := iFin.val
              let ti := fps[i]!
              let latestPe := (vLatestEngArrays fps).2[i]!
              let latestNe := (vLatestEngArrays fps).1[i]!
              Array.ofFn (n := i) fun jFin =>
                let j := jFin.val
                let tj := fps[j]!
                let poPe := ti.engine == .pe && latestPe == some j
                let poNeComp := ti.engine == .ne && tj.engine == .ne && latestNe == some j && tj.isNeCompute && ti.isNeCompute
                let w5 := tj.engine == .ne && (ti.sync.src1WaitNe5 || (ti.engine == .pe && ti.sync.src2WaitNe5))
                let wPe := tj.engine == .pe && latestPe == some j && (ti.sync.src1WaitPe22 || (ti.engine == .pe && ti.sync.src2WaitPe22))
                let depNonRaw :=
                  j + 1 == i && ti.sync.hasSrc1Dma &&
                  (ti.sync.src1DepKind == 1 || ti.sync.src1DepKind == 2) &&
                  ti.sync.src1DepDist == 0 &&
                  !vSrc1DramReadsFromWrite tj ti
                poPe || poNeComp || w5 || wPe || depNonRaw)[i]!
            Array.ofFn (n := i) fun jFin =>
              let j := jFin.val
              hbRow[j]! || (ti.engine == .ne && latestNe == some j))[i]!)[j]!
        else false) = stepHBL fps := by
    funext a b
    unfold stepHBL stepHB
    by_cases hab : a < b ∧ b < fps.size
    · have hcond : (decide (a < b) && decide (b < fps.size)) = true := by simp [hab.1, hab.2]
      simp only [hcond, ↓reduceIte, Bool.not_true, Bool.false_eq_true]
      rw [getElem!_pos _ b (by rw [Array.size_ofFn]; exact hab.2), Array.getElem_ofFn,
          getElem!_pos _ a (by rw [Array.size_ofFn]; exact hab.1), Array.getElem_ofFn,
          getElem!_pos _ b (by rw [Array.size_ofFn]; exact hab.2), Array.getElem_ofFn,
          getElem!_pos _ a (by rw [Array.size_ofFn]; exact hab.1), Array.getElem_ofFn]
      simp only [hneArr b hab.2, hpeArr b hab.2]
    · have hcond : (decide (a < b) && decide (b < fps.size)) = false := by
        simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]
        by_cases h1 : a < b
        · right; intro h2; exact hab ⟨h1, h2⟩
        · left; exact h1
      simp [hcond]
  rw [hstepEq]
  exact reachB_iff (stepHBL fps) (stepHBL_fwd fps) fps.size k i hi

private theorem checkRange_ok {lo hi : Nat} {f : Nat → Except String Unit}
    (h : checkRange lo hi f = .ok ()) :
    ∀ i, lo ≤ i → i < hi → f i = .ok () := by
  intro i hlo hhi
  have haux : ∀ d lo, d = hi - lo → checkRange lo hi f = .ok () → lo ≤ i → i < hi → f i = .ok () := by
    intro d
    induction d with
    | zero => intro lo hd _ _ _; omega
    | succ r ih =>
      intro lo hd hchk hlo' hhi'
      have hlt : lo < hi := by omega
      unfold checkRange at hchk
      simp only [hlt, ↓reduceDIte] at hchk
      cases hflo : f lo with
      | error e => simp [hflo] at hchk
      | ok u =>
        cases u
        simp only [hflo] at hchk
        by_cases hEq : i = lo
        · subst hEq; exact hflo
        · exact ih (lo + 1) (by omega) hchk (by omega) hhi'
  exact haux (hi - lo) lo rfl h hlo hhi

private theorem barResource_beq_eq {a b : BarResource} (h : (a == b) = true) : a = b := by
  cases a with
  | surface i1 s1 =>
    cases b with
    | surface i2 s2 =>
      change ((i1 == i2) && (s1 == s2)) = true at h
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      rw [h.1, h.2]
    | scratch => cases h
    | kernel => cases h
  | scratch => cases b <;> first | rfl | cases h
  | kernel => cases b <;> first | rfl | cases h

private theorem dramAccessPort_beq_eq {a b : DramAccessPort} (h : (a == b) = true) : a = b := by
  cases a <;> cases b <;> first | rfl | cases h

private theorem l2AccessRole_beq_eq {a b : L2AccessRole} (h : (a == b) = true) : a = b := by
  cases a <;> cases b <;> first | rfl | cases h

private theorem engine_beq_eq {a b : Engine} (h : (a == b) = true) : a = b := by
  cases a <;> cases b <;> first | rfl | cases h

private theorem barResource_beq_self (a : BarResource) : (a == a) = true := by
  cases a with
  | surface i s => change ((i == i) && (s == s)) = true; simp
  | scratch => rfl
  | kernel => rfl

private theorem dramAccessPort_beq_self (a : DramAccessPort) : (a == a) = true := by
  cases a <;> rfl

private theorem l2AccessRole_beq_self (a : L2AccessRole) : (a == a) = true := by
  cases a <;> rfl

private theorem engine_beq_self (a : Engine) : (a == a) = true := by
  cases a <;> rfl

private theorem reachB_lt_size (tbl : Array (Array Bool)) (k p : Nat)
    (h : reachB tbl k p = true) : p < tbl.size := by
  unfold reachB at h
  cases hget : tbl[p]? with
  | none => simp [hget] at h
  | some _ =>
    cases Nat.lt_or_ge p tbl.size with
    | inl hlt => exact hlt
    | inr hge =>
      have hnone : tbl[p]? = none := Array.getElem?_eq_none hge
      rw [hnone] at hget
      cases hget

private theorem except_if_throw_bind_ok {α : Type} {c : Prop} [Decidable c] {e : String}
    {k : Unit → Except String α} {v : α}
    (h : ((if c then throw e else pure ()) >>= k) = .ok v) : ¬ c ∧ k () = .ok v := by
  by_cases hc : c
  · simp [hc, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  · simp [hc, pure, Except.pure, bind, Except.bind] at h
    exact ⟨hc, h⟩

private theorem except_if_do_throw_ok {α β : Type} {c : Prop} [Decidable c] {e : String}
    {t : β → Except String α} {k : Except String α} {v : α}
    (h : (if c then (throw e >>= t) else k) = .ok v) : ¬ c ∧ k = .ok v := by
  by_cases hc : c
  · simp [hc, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  · simp [hc] at h
    exact ⟨hc, h⟩

/-- Intra-task BAR 3 scratch non-overlap soundness: whenever production `verifyProgramHazards`
    accepts a program `fps`, no two distinct BAR 3 scratch accesses in the same task (where at
    least one is a write) share any byte address `x`. -/
theorem verifyProgramHazards_intraTask_scratch_disjoint
    (fps : Array VerifiedTaskFootprint)
    (hok : verifyProgramHazards fps = .ok ())
    (i : Nat) (hi : i < fps.size)
    (idx1 idx2 : Nat) (h1 : idx1 < fps[i]!.dramAccesses.size)
    (h2 : idx2 < fps[i]!.dramAccesses.size) (hne : idx1 ≠ idx2) :
    let a := fps[i]!.dramAccesses[idx1]!
    let b := fps[i]!.dramAccesses[idx2]!
    a.resource = .scratch → b.resource = .scratch → (a.isWrite = true ∨ b.isWrite = true) →
    ∀ x, ¬ (a.lo ≤ x ∧ x < a.hi ∧ b.lo ≤ x ∧ x < b.hi) := by
  unfold verifyProgramHazards at hok
  dsimp only at hok
  obtain ⟨⟨⟩, _, hok⟩ := except_bind_ok hok
  obtain ⟨⟨⟩, _, hok⟩ := except_bind_ok hok
  obtain ⟨⟨⟩, _, h4⟩ := except_bind_ok hok
  unfold verifyIntraTaskScratch at h4
  intro a b haRes hbRes hwr x hx
  have checkPair : ∀ p q, p < q → q < fps[i]!.dramAccesses.size →
      let ap := fps[i]!.dramAccesses[p]!
      let bq := fps[i]!.dramAccesses[q]!
      ap.resource = .scratch → bq.resource = .scratch → (ap.isWrite = true ∨ bq.isWrite = true) →
      vIntervalsOverlap ap.lo ap.hi bq.lo bq.hi = false := by
    intro p q hpq hq ap bq hpRes hqRes hpWr
    have hiOk := checkRange_ok h4 i (Nat.zero_le i) hi
    have hpOk := checkRange_ok hiOk p (Nat.zero_le p) (by omega)
    have hqOk := checkRange_ok hpOk q (by omega) hq
    cases hov : vIntervalsOverlap ap.lo ap.hi bq.lo bq.hi with
    | false => rfl
    | true =>
      have hwrB : (ap.isWrite || bq.isWrite) = true := by
        rcases hpWr with h | h <;> simp [h]
      have hprB : (ap.resource == .scratch) = true := by rw [hpRes]; rfl
      have hqrB : (bq.resource == .scratch) = true := by rw [hqRes]; rfl
      dsimp only at hqOk
      change (if (ap.resource == .scratch && bq.resource == .scratch && (ap.isWrite || bq.isWrite) &&
          vIntervalsOverlap ap.lo ap.hi bq.lo bq.hi) = true then _ else _) = Except.ok () at hqOk
      rw [hprB, hqrB, hwrB, hov] at hqOk
      cases hqOk
  by_cases hlt : idx1 < idx2
  · have hno := checkPair idx1 idx2 hlt h2 haRes hbRes hwr
    exact no_shared_byte_of_not_vIntervalsOverlap _ _ _ _ hno x hx
  · have hgt : idx2 < idx1 := by omega
    have hno := checkPair idx2 idx1 hgt h1 hbRes haRes hwr.symm
    exact no_shared_byte_of_not_vIntervalsOverlap _ _ _ _ hno x ⟨hx.2.2.1, hx.2.2.2, hx.1, hx.2.1⟩

/-- Formal access-level `src1Dma` `depKind` RAW barrier predicate (in terms of `HappensBeforeEq`). -/
def DramRawBarrier (fps : Array VerifiedTaskFootprint) (k i : Nat) : Prop :=
  let ti := fps[i]!
  ti.sync.hasSrc1Dma = true ∧
  (ti.sync.src1DepKind = 1 ∨ ti.sync.src1DepKind = 2) ∧
  ti.sync.src1DepDist = 0 ∧
  HappensBeforeEq fps k (i - 1)

/-- Formal access-level PE `src2Dma` `depKind` barrier predicate (in terms of `HappensBeforeEq`). -/
def Src2KindBarrier (fps : Array VerifiedTaskFootprint) (k i : Nat) : Prop :=
  let ti := fps[i]!
  fps[i - 1]!.engine = .pe ∧
  ti.sync.hasSrc2Dma = true ∧
  (ti.sync.src2DepKind = 1 ∨ ti.sync.src2DepKind = 2) ∧
  ti.sync.src2DepDist = 0 ∧
  HappensBeforeEq fps k (i - 1)

/-- Formal PE wait-bit-22 barrier predicate (in terms of `HappensBeforeEq`). -/
def WaitPe22Barrier (fps : Array VerifiedTaskFootprint) (k i : Nat) : Prop :=
  let ti := fps[i]!
  (ti.sync.src1WaitPe22 = true ∨ ti.sync.src2WaitPe22 = true) ∧
  ∃ j, vLatestEngBefore fps .pe i = some j ∧ HappensBeforeEq fps k j

/-- Formal NE wait-bit-5 barrier predicate (in terms of `HappensBeforeEq`). -/
def WaitNe5Barrier (fps : Array VerifiedTaskFootprint) (k i : Nat) : Prop :=
  let ti := fps[i]!
  (ti.sync.src1WaitNe5 = true ∨ ti.sync.src2WaitNe5 = true) ∧
  (fps[k]!.engine = .ne ∨ ∃ j < i, fps[j]!.engine = .ne ∧ HappensBeforeEq fps k j)

/-- Formal DRAM ordering relation established by the S5 hazard verifier for conflicting accesses
    `(a, b)` on tasks `(k, i)` (`k < i`). Every branch strictly orders `k` before `i` via
    whole-task `HappensBefore` or a hardware DMA/sync/extended-header barrier anchored in
    `HappensBeforeEq`. -/
def DramOrdered (fps : Array VerifiedTaskFootprint) (k i : Nat)
    (a b : VerifiedDramAccess) : Prop :=
  let ti := fps[i]!
  if b.port = .kernel then
    a.isWrite = true ∧ ∃ p, ti.extProducerTask = some p ∧ HappensBeforeEq fps k p
  else if ti.engine = .ne ∧ b.port = .src1Dma then
    a.isWrite = true ∧ DramRawBarrier fps k i
  else if ti.engine = .pe ∧ b.port = .src2Dma then
    Src2KindBarrier fps k i ∨ WaitPe22Barrier fps k i ∨ WaitNe5Barrier fps k i ∨ DramRawBarrier fps k i
  else if a.isWrite = true ∧ b.isWrite = false then
    HappensBefore fps k i ∨ DramRawBarrier fps k i
  else
    HappensBefore fps k i

/-- Formal L2 ordering relation established by the S5 hazard verifier for conflicting L2 accesses
    `(a, b)` on tasks `(k, i)` (`k < i`). -/
def L2Ordered (fps : Array VerifiedTaskFootprint) (k i : Nat)
    (a b : VerifiedL2Access) : Prop :=
  let ti := fps[i]!
  let handoff2 : Prop :=
    a.isWrite = true ∧
    ((b.role = .src1L2 ∧ ti.sync.src1HandoffNe = true) ∨
     (b.role = .src2L2 ∧ ti.sync.src2HandoffNe = true)) ∧
    ∃ j, vLatestEngBefore fps .ne i = some j ∧ HappensBeforeL2Eq fps k j
  let handoff3 : Prop :=
    a.isWrite = true ∧
    b.role = .src1L2 ∧
    ti.sync.src1HandoffPe = true ∧
    ∃ j, vLatestEngBefore fps .pe i = some j ∧ HappensBeforeL2Eq fps k j
  if ti.engine = .pe ∧ (b.role = .src2Stage ∨ b.role = .src2L2) then
    WaitPe22Barrier fps k i ∨ WaitNe5Barrier fps k i ∨ Src2KindBarrier fps k i ∨
      (b.role = .src2L2 ∧ handoff2)
  else if b.role = .src1Stage ∧ a.role = .resStage then
    WaitNe5Barrier fps k i ∨ WaitPe22Barrier fps k i
  else
    HappensBeforeL2 fps k i ∨ handoff2 ∨ handoff3

private theorem vLatestEngBefore_lt (fps : Array VerifiedTaskFootprint) (eng : Engine) :
    ∀ n j, vLatestEngBefore fps eng n = some j → j < n := by
  intro n
  induction n with
  | zero => intro j h; simp [vLatestEngBefore] at h
  | succ m ih =>
    intro j h
    unfold vLatestEngBefore at h
    cases hget : fps[m]? with
    | none => simp [hget] at h; have := ih j h; omega
    | some fp =>
      simp only [hget] at h
      split at h
      · simp at h; omega
      · have := ih j h; omega

private theorem hbEq_of_bool (fps : Array VerifiedTaskFootprint) (k j : Nat) (hj : j < fps.size)
    (h : (k == j || reachB (buildHBReachTables fps).1 k j) = true) :
    HappensBeforeEq fps k j := by
  simp only [Bool.or_eq_true, beq_iff_eq] at h
  rcases h with rfl | hhb
  · exact Or.inl rfl
  · exact Or.inr ((buildHBReachTables_hb_iff fps k j hj).mp hhb)

private theorem hblEq_of_bool (fps : Array VerifiedTaskFootprint) (k j : Nat) (hj : j < fps.size)
    (h : (k == j || reachB (buildHBReachTables fps).2 k j) = true) :
    HappensBeforeL2Eq fps k j := by
  simp only [Bool.or_eq_true, beq_iff_eq] at h
  rcases h with rfl | hhbl
  · exact Or.inl rfl
  · exact Or.inr ((buildHBReachTables_hbl_iff fps k j hj).mp hhbl)

private theorem vDramRawKind_sound (fps : Array VerifiedTaskFootprint) (k i : Nat) (hi : i < fps.size)
    (h : vDramRawKind fps (fun x y => x == y || reachB (buildHBReachTables fps).1 x y) k i = true) :
    DramRawBarrier fps k i := by
  unfold vDramRawKind at h
  simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
  rcases h with ⟨⟨⟨hhas, hkind⟩, hdist⟩, hhb⟩
  refine ⟨hhas, hkind, hdist, ?_⟩
  rcases hhb with rfl | hhb
  · exact Or.inl rfl
  · exact Or.inr ((buildHBReachTables_hb_iff fps k (i - 1) (by omega)).mp hhb)

private theorem vSrc2Kind_sound (fps : Array VerifiedTaskFootprint) (k i : Nat) (hi : i < fps.size)
    (h : vSrc2Kind fps (fun x y => x == y || reachB (buildHBReachTables fps).1 x y) k i = true) :
    Src2KindBarrier fps k i := by
  unfold vSrc2Kind at h
  simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
  rcases h with ⟨⟨⟨⟨heng, hhas⟩, hkind⟩, hdist⟩, hhb⟩
  refine ⟨engine_beq_eq heng, hhas, hkind, hdist, ?_⟩
  rcases hhb with rfl | hhb
  · exact Or.inl rfl
  · exact Or.inr ((buildHBReachTables_hb_iff fps k (i - 1) (by omega)).mp hhb)

private theorem vSrc2Wait22_sound (fps : Array VerifiedTaskFootprint) (k i : Nat) (hi : i < fps.size)
    (h : vSrc2Wait22 fps[i]! (fun x y => x == y || reachB (buildHBReachTables fps).1 x y)
          (vLatestEngBefore fps .pe i) k = true) :
    WaitPe22Barrier fps k i := by
  unfold vSrc2Wait22 at h
  simp only [Bool.and_eq_true, Bool.or_eq_true] at h
  rcases h with ⟨hw, hmatch⟩
  cases hlat : vLatestEngBefore fps .pe i with
  | none => simp [hlat] at hmatch
  | some j =>
    simp only [hlat] at hmatch
    have hjLt : j < i := vLatestEngBefore_lt fps .pe i j hlat
    exact ⟨hw, j, hlat, hbEq_of_bool fps k j (by omega) hmatch⟩

private theorem vSrc2Wait5_sound (fps : Array VerifiedTaskFootprint) (k i : Nat) (hi : i < fps.size)
    (h : vSrc2Wait5 fps (fun x y => x == y || reachB (buildHBReachTables fps).1 x y) k i = true) :
    WaitNe5Barrier fps k i := by
  unfold vSrc2Wait5 at h
  simp only [Bool.and_eq_true, Bool.or_eq_true] at h
  rcases h with ⟨hw, heng | hany⟩
  · exact ⟨hw, Or.inl (engine_beq_eq heng)⟩
  · rw [anyLt_iff] at hany
    obtain ⟨j, hjLt, hcond⟩ := hany
    simp only [Bool.and_eq_true] at hcond
    exact ⟨hw, Or.inr ⟨j, hjLt, engine_beq_eq hcond.1, hbEq_of_bool fps k j (by omega) hcond.2⟩⟩

private theorem vDramPairOrdered_sound (fps : Array VerifiedTaskFootprint)
    (k i : Nat) (hi : i < fps.size) (a b : VerifiedDramAccess)
    (h : vDramPairOrdered fps (buildHBReachTables fps).1 (vLatestEngBefore fps .pe i) k i a b = true) :
    DramOrdered fps k i a b := by
  unfold vDramPairOrdered at h
  unfold DramOrdered
  dsimp only at h ⊢
  by_cases hkern : b.port = .kernel
  · have hkernB : (b.port == .kernel) = true := by rw [hkern]; rfl
    simp only [hkern, ↓reduceIte]
    simp only [hkernB, ↓reduceIte] at h
    simp only [Bool.and_eq_true] at h
    rcases h with ⟨haw, hmatch⟩
    cases hp : fps[i]!.extProducerTask with
    | none => simp [hp] at hmatch
    | some p =>
      simp only [hp, Bool.or_eq_true, beq_iff_eq] at hmatch
      rcases hmatch with hkp | hhb
      · exact ⟨haw, p, rfl, Or.inl hkp⟩
      · have hsz : (buildHBReachTables fps).1.size = fps.size := by
          unfold buildHBReachTables reachTable; exact reachTableAux_size _ _ _
        have hpLt : p < fps.size := hsz ▸ reachB_lt_size (buildHBReachTables fps).1 k p hhb
        exact ⟨haw, p, rfl, Or.inr ((buildHBReachTables_hb_iff fps k p hpLt).mp hhb)⟩
  · have hkernB : (b.port == .kernel) = false := by
      cases hc : (b.port == .kernel) with
      | false => rfl
      | true => exact absurd (dramAccessPort_beq_eq hc) hkern
    simp only [hkern, ↓reduceIte]
    simp only [hkernB, Bool.false_eq_true, ↓reduceIte] at h
    by_cases hne1 : fps[i]!.engine = .ne ∧ b.port = .src1Dma
    · have hne1B : (fps[i]!.engine == .ne && b.port == .src1Dma) = true := by
        rw [hne1.1, hne1.2]; rfl
      simp only [hne1]
      simp only [hne1B, ↓reduceIte] at h
      simp only [Bool.and_eq_true] at h
      exact ⟨h.1, vDramRawKind_sound fps k i hi h.2⟩
    · have hne1B : (fps[i]!.engine == .ne && b.port == .src1Dma) = false := by
        cases hc : (fps[i]!.engine == .ne && b.port == .src1Dma) with
        | false => rfl
        | true =>
          simp only [Bool.and_eq_true] at hc
          exact absurd ⟨engine_beq_eq hc.1, dramAccessPort_beq_eq hc.2⟩ hne1
      simp only [hne1]
      simp only [hne1B, Bool.false_eq_true, ↓reduceIte] at h
      by_cases hpe2 : fps[i]!.engine = .pe ∧ b.port = .src2Dma
      · have hpe2B : (fps[i]!.engine == .pe && b.port == .src2Dma) = true := by
          rw [hpe2.1, hpe2.2]; rfl
        simp only [hpe2, ↓reduceIte]
        simp only [hpe2B, ↓reduceIte] at h
        simp only [Bool.or_eq_true] at h
        rcases h with ((hs2k | hw22) | hw5) | hdr
        · exact Or.inl (vSrc2Kind_sound fps k i hi hs2k)
        · exact Or.inr (Or.inl (vSrc2Wait22_sound fps k i hi hw22))
        · exact Or.inr (Or.inr (Or.inl (vSrc2Wait5_sound fps k i hi hw5)))
        · exact Or.inr (Or.inr (Or.inr (vDramRawKind_sound fps k i hi hdr)))
      · have hpe2B : (fps[i]!.engine == .pe && b.port == .src2Dma) = false := by
          cases hc : (fps[i]!.engine == .pe && b.port == .src2Dma) with
          | false => rfl
          | true =>
            simp only [Bool.and_eq_true] at hc
            exact absurd ⟨engine_beq_eq hc.1, dramAccessPort_beq_eq hc.2⟩ hpe2
        simp only [hpe2]
        simp only [hpe2B, Bool.false_eq_true, ↓reduceIte] at h
        by_cases hraw : a.isWrite = true ∧ b.isWrite = false
        · have hrawB : (a.isWrite && !b.isWrite) = true := by rw [hraw.1, hraw.2]; rfl
          simp only [hraw, ↓reduceIte]
          simp only [hrawB, ↓reduceIte] at h
          simp only [Bool.or_eq_true] at h
          rcases h with hhb | hdr
          · exact Or.inl ((buildHBReachTables_hb_iff fps k i hi).mp hhb)
          · exact Or.inr (vDramRawKind_sound fps k i hi hdr)
        · have hrawB : (a.isWrite && !b.isWrite) = false := by
            cases ha : a.isWrite <;> cases hb : b.isWrite <;> simp_all
          simp only [hraw]
          simp only [hrawB, Bool.false_eq_true, ↓reduceIte] at h
          exact (buildHBReachTables_hb_iff fps k i hi).mp h

private theorem vL2PairOrdered_sound (fps : Array VerifiedTaskFootprint)
    (k i : Nat) (hi : i < fps.size) (a b : VerifiedL2Access)
    (h : vL2PairOrdered fps (buildHBReachTables fps).1 (buildHBReachTables fps).2
          (vLatestEngBefore fps .ne i) (vLatestEngBefore fps .pe i) k i a b = true) :
    L2Ordered fps k i a b := by
  unfold vL2PairOrdered at h
  unfold L2Ordered
  dsimp only at h ⊢
  have hHO2 :
      (a.isWrite &&
       ((b.role == .src1L2 && fps[i]!.sync.src1HandoffNe) ||
        (b.role == .src2L2 && fps[i]!.sync.src2HandoffNe)) &&
       match vLatestEngBefore fps .ne i with
       | some j => k == j || reachB (buildHBReachTables fps).2 k j
       | none => false) = true →
      a.isWrite = true ∧
      ((b.role = .src1L2 ∧ fps[i]!.sync.src1HandoffNe = true) ∨
       (b.role = .src2L2 ∧ fps[i]!.sync.src2HandoffNe = true)) ∧
      ∃ j, vLatestEngBefore fps .ne i = some j ∧ HappensBeforeL2Eq fps k j := by
    intro h2
    simp only [Bool.and_eq_true, Bool.or_eq_true] at h2
    rcases h2 with ⟨⟨haw, ⟨hr1, h1⟩ | ⟨hr2, h2'⟩⟩, hmatch⟩
    · cases hlat : vLatestEngBefore fps .ne i with
      | none => simp [hlat] at hmatch
      | some j =>
        simp only [hlat] at hmatch
        have hjLt := vLatestEngBefore_lt fps .ne i j hlat
        exact ⟨haw, Or.inl ⟨l2AccessRole_beq_eq hr1, h1⟩, j, rfl, hblEq_of_bool fps k j (by omega) hmatch⟩
    · cases hlat : vLatestEngBefore fps .ne i with
      | none => simp [hlat] at hmatch
      | some j =>
        simp only [hlat] at hmatch
        have hjLt := vLatestEngBefore_lt fps .ne i j hlat
        exact ⟨haw, Or.inr ⟨l2AccessRole_beq_eq hr2, h2'⟩, j, rfl, hblEq_of_bool fps k j (by omega) hmatch⟩
  have hHO3 :
      (a.isWrite && (b.role == .src1L2) && fps[i]!.sync.src1HandoffPe &&
       match vLatestEngBefore fps .pe i with
       | some j => k == j || reachB (buildHBReachTables fps).2 k j
       | none => false) = true →
      a.isWrite = true ∧ b.role = .src1L2 ∧ fps[i]!.sync.src1HandoffPe = true ∧
      ∃ j, vLatestEngBefore fps .pe i = some j ∧ HappensBeforeL2Eq fps k j := by
    intro h3
    simp only [Bool.and_eq_true] at h3
    rcases h3 with ⟨⟨⟨haw, hrole⟩, hpe⟩, hmatch⟩
    cases hlat : vLatestEngBefore fps .pe i with
    | none => simp [hlat] at hmatch
    | some j =>
      simp only [hlat] at hmatch
      have hjLt := vLatestEngBefore_lt fps .pe i j hlat
      exact ⟨haw, l2AccessRole_beq_eq hrole, hpe, j, rfl, hblEq_of_bool fps k j (by omega) hmatch⟩
  by_cases hs2 : fps[i]!.engine = .pe ∧ (b.role = .src2Stage ∨ b.role = .src2L2)
  · have hs2B : (fps[i]!.engine == .pe && (b.role == .src2Stage || b.role == .src2L2)) = true := by
      rcases hs2 with ⟨he, hr | hr⟩ <;> rw [he, hr] <;> rfl
    simp only [hs2]
    simp only [hs2B, ↓reduceIte] at h
    simp only [Bool.or_eq_true] at h
    rcases h with ((hw22 | hw5) | hs2k) | hbo2
    · exact Or.inl (vSrc2Wait22_sound fps k i hi hw22)
    · exact Or.inr (Or.inl (vSrc2Wait5_sound fps k i hi hw5))
    · exact Or.inr (Or.inr (Or.inl (vSrc2Kind_sound fps k i hi hs2k)))
    · rw [Bool.and_eq_true] at hbo2
      exact Or.inr (Or.inr (Or.inr ⟨l2AccessRole_beq_eq hbo2.1, hHO2 hbo2.2⟩))
  · have hs2B : (fps[i]!.engine == .pe && (b.role == .src2Stage || b.role == .src2L2)) = false := by
      cases hc : (fps[i]!.engine == .pe && (b.role == .src2Stage || b.role == .src2L2)) with
      | false => rfl
      | true =>
        simp only [Bool.and_eq_true, Bool.or_eq_true] at hc
        rcases hc with ⟨he, h1 | h2⟩
        · exact absurd ⟨engine_beq_eq he, Or.inl (l2AccessRole_beq_eq h1)⟩ hs2
        · exact absurd ⟨engine_beq_eq he, Or.inr (l2AccessRole_beq_eq h2)⟩ hs2
    simp only [hs2]
    simp only [hs2B, Bool.false_eq_true, ↓reduceIte] at h
    by_cases hst : b.role = .src1Stage ∧ a.role = .resStage
    · have hstB : (b.role == .src1Stage && a.role == .resStage) = true := by
        rw [hst.1, hst.2]; rfl
      simp only [hst, ↓reduceIte]
      simp only [hstB, ↓reduceIte] at h
      simp only [Bool.or_eq_true] at h
      rcases h with hw5 | hw22
      · exact Or.inl (vSrc2Wait5_sound fps k i hi hw5)
      · exact Or.inr (vSrc2Wait22_sound fps k i hi hw22)
    · have hstB : (b.role == .src1Stage && a.role == .resStage) = false := by
        cases hc : (b.role == .src1Stage && a.role == .resStage) with
        | false => rfl
        | true =>
          simp only [Bool.and_eq_true] at hc
          exact absurd ⟨l2AccessRole_beq_eq hc.1, l2AccessRole_beq_eq hc.2⟩ hst
      simp only [hst]
      simp only [hstB, Bool.false_eq_true, ↓reduceIte] at h
      simp only [Bool.or_eq_true] at h
      rcases h with (hhbl | ho2) | ho3
      · exact Or.inl ((buildHBReachTables_hbl_iff fps k i hi).mp hhbl)
      · exact Or.inr (Or.inl (hHO2 ho2))
      · exact Or.inr (Or.inr (hHO3 ho3))

/-- Inter-task DRAM race-freedom soundness: whenever production `verifyProgramHazards` accepts
    a program `fps`, every pair of tasks `k < i < fps.size` with conflicting DRAM accesses
    `(a, b)` on the same resource (at least one write) sharing any byte `x` is strictly ordered
    by `DramOrdered fps k i a b`. -/
theorem verifyProgramHazards_dram_ordered
    (fps : Array VerifiedTaskFootprint)
    (hok : verifyProgramHazards fps = .ok ())
    (k i : Nat) (hki : k < i) (hi : i < fps.size)
    (ia ib : Nat) (hia : ia < fps[k]!.dramAccesses.size) (hib : ib < fps[i]!.dramAccesses.size) :
    let a := fps[k]!.dramAccesses[ia]!
    let b := fps[i]!.dramAccesses[ib]!
    a.resource = b.resource → (a.isWrite = true ∨ b.isWrite = true) →
    (∃ x, a.lo ≤ x ∧ x < a.hi ∧ b.lo ≤ x ∧ x < b.hi) →
    DramOrdered fps k i a b := by
  unfold verifyProgramHazards at hok
  dsimp only at hok
  obtain ⟨⟨⟩, _, hok⟩ := except_bind_ok hok
  obtain ⟨⟨⟩, h2, _⟩ := except_bind_ok hok
  unfold verifyPairwiseHazards at h2
  intro a b hres hwr hwit
  have hiOk := checkRange_ok h2 i (by omega) hi
  have hkOk := checkRange_ok hiOk k (Nat.zero_le k) hki
  obtain ⟨⟨⟩, _, hdramOk⟩ := except_bind_ok hkOk
  have hiaOk := checkRange_ok hdramOk ia (Nat.zero_le ia) hia
  have hibOk := checkRange_ok hiaOk ib (Nat.zero_le ib) hib
  have hov : vIntervalsOverlap a.lo a.hi b.lo b.hi = true :=
    vIntervalsOverlap_of_witness a.lo a.hi b.lo b.hi hwit
  have hwrB : (a.isWrite || b.isWrite) = true := by
    rcases hwr with h | h <;> simp [h]
  have hresB : (a.resource == b.resource) = true := by rw [hres]; exact barResource_beq_self _
  obtain ⟨_, _, _, hpeArr⟩ := vLatestEngArrays_eq fps
  have hord : vDramPairOrdered fps (buildHBReachTables fps).1 (vLatestEngBefore fps .pe i) k i a b = true := by
    cases hbo : vDramPairOrdered fps (buildHBReachTables fps).1 (vLatestEngBefore fps .pe i) k i a b with
    | true => rfl
    | false =>
      dsimp only at hibOk
      rw [hpeArr i hi] at hibOk
      change (if (a.resource == b.resource && (a.isWrite || b.isWrite) &&
          vIntervalsOverlap a.lo a.hi b.lo b.hi) = true then
        if (!vDramPairOrdered fps (buildHBReachTables fps).1 (vLatestEngBefore fps .pe i) k i a b) = true then _
        else _ else _) = Except.ok () at hibOk
      rw [hresB, hwrB, hov, hbo] at hibOk
      cases hibOk
  exact vDramPairOrdered_sound fps k i hi a b hord

/-- Inter-task DRAM WAR & WAW strict `HappensBefore` soundness: whenever `verifyProgramHazards`
    accepts `fps` and task `k < i` has any DRAM access `a` conflicting with a `resDma` write `b`
    in task `i` on the same resource (sharing any byte `x`), strict whole-task `HappensBefore fps k i`
    holds. -/
theorem verifyProgramHazards_dram_war_waw_hb
    (fps : Array VerifiedTaskFootprint)
    (hok : verifyProgramHazards fps = .ok ())
    (k i : Nat) (hki : k < i) (hi : i < fps.size)
    (ia ib : Nat) (hia : ia < fps[k]!.dramAccesses.size) (hib : ib < fps[i]!.dramAccesses.size) :
    let a := fps[k]!.dramAccesses[ia]!
    let b := fps[i]!.dramAccesses[ib]!
    a.resource = b.resource → b.port = .resDma → b.isWrite = true →
    (∃ x, a.lo ≤ x ∧ x < a.hi ∧ b.lo ≤ x ∧ x < b.hi) →
    HappensBefore fps k i := by
  intro a b hres hbPort hbWr hwit
  have hord := verifyProgramHazards_dram_ordered fps hok k i hki hi ia ib hia hib hres (Or.inr hbWr) hwit
  unfold DramOrdered at hord
  dsimp only at hord
  have h1 : ¬ (fps[i]!.dramAccesses[ib]!.port = .kernel) := by rw [hbPort]; decide
  have h2 : ¬ (fps[i]!.engine = .ne ∧ fps[i]!.dramAccesses[ib]!.port = .src1Dma) := fun h => by rw [hbPort] at h; cases h.2
  have h3 : ¬ (fps[i]!.engine = .pe ∧ fps[i]!.dramAccesses[ib]!.port = .src2Dma) := fun h => by rw [hbPort] at h; cases h.2
  have h4 : ¬ (fps[k]!.dramAccesses[ia]!.isWrite = true ∧ fps[i]!.dramAccesses[ib]!.isWrite = false) := fun h => by rw [hbWr] at h; cases h.2
  simp only [h1, h2, h3, h4, ↓reduceIte] at hord
  exact hord

/-- Inter-task L2 race-freedom soundness: whenever production `verifyProgramHazards` accepts
    a program `fps`, every pair of tasks `k < i < fps.size` with conflicting L2 accesses `(a, b)`
    (at least one write) sharing any byte `x` is strictly ordered by `L2Ordered fps k i a b`. -/
theorem verifyProgramHazards_l2_ordered
    (fps : Array VerifiedTaskFootprint)
    (hok : verifyProgramHazards fps = .ok ())
    (k i : Nat) (hki : k < i) (hi : i < fps.size)
    (ia ib : Nat) (hia : ia < fps[k]!.l2Accesses.size) (hib : ib < fps[i]!.l2Accesses.size) :
    let a := fps[k]!.l2Accesses[ia]!
    let b := fps[i]!.l2Accesses[ib]!
    (a.isWrite = true ∨ b.isWrite = true) →
    (∃ x, a.lo ≤ x ∧ x < a.hi ∧ b.lo ≤ x ∧ x < b.hi) →
    L2Ordered fps k i a b := by
  unfold verifyProgramHazards at hok
  dsimp only at hok
  obtain ⟨⟨⟩, _, hok⟩ := except_bind_ok hok
  obtain ⟨⟨⟩, h2, _⟩ := except_bind_ok hok
  unfold verifyPairwiseHazards at h2
  intro a b hwr hwit
  have hiOk := checkRange_ok h2 i (by omega) hi
  have hkOk := checkRange_ok hiOk k (Nat.zero_le k) hki
  obtain ⟨⟨⟩, hl2Ok, _⟩ := except_bind_ok hkOk
  have hiaOk := checkRange_ok hl2Ok ia (Nat.zero_le ia) hia
  have hibOk := checkRange_ok hiaOk ib (Nat.zero_le ib) hib
  have hov : vIntervalsOverlap a.lo a.hi b.lo b.hi = true :=
    vIntervalsOverlap_of_witness a.lo a.hi b.lo b.hi hwit
  have hwrB : (a.isWrite || b.isWrite) = true := by
    rcases hwr with h | h <;> simp [h]
  obtain ⟨_, _, hneArr, hpeArr⟩ := vLatestEngArrays_eq fps
  have hord : vL2PairOrdered fps (buildHBReachTables fps).1 (buildHBReachTables fps).2
      (vLatestEngBefore fps .ne i) (vLatestEngBefore fps .pe i) k i a b = true := by
    cases hbo : vL2PairOrdered fps (buildHBReachTables fps).1 (buildHBReachTables fps).2
        (vLatestEngBefore fps .ne i) (vLatestEngBefore fps .pe i) k i a b with
    | true => rfl
    | false =>
      dsimp only at hibOk
      rw [hneArr i hi, hpeArr i hi] at hibOk
      change (if ((a.isWrite || b.isWrite) && vIntervalsOverlap a.lo a.hi b.lo b.hi) = true then
        if (!vL2PairOrdered fps (buildHBReachTables fps).1 (buildHBReachTables fps).2
          (vLatestEngBefore fps .ne i) (vLatestEngBefore fps .pe i) k i a b) = true then _
        else _ else _) = Except.ok () at hibOk
      rw [hwrB, hov, hbo] at hibOk
      cases hibOk
  exact vL2PairOrdered_sound fps k i hi a b hord

/-- Formal byte-level semantics of `WriteCoverage`: byte offset `x` is guaranteed written by `c`. -/
def coversByte : WriteCoverage → Nat → Prop
  | .dense lo hi, x => lo ≤ x ∧ x < hi
  | .periodic start innerLen outerCount outerStride, x =>
    ∃ j < outerCount, start + j * outerStride ≤ x ∧ x < start + j * outerStride + innerLen
  | .sparse, _ => False

private theorem collectDenseCoverages_sound (allCovs covs : List WriteCoverage) (acc : List (Nat × Nat))
    (hsub : ∀ c ∈ covs, c ∈ allCovs)
    (hacc : ∀ iv ∈ acc, ∀ x, iv.1 ≤ x → x < iv.2 → ∃ c ∈ allCovs, coversByte c x) :
    ∀ iv ∈ collectDenseCoverages covs acc, ∀ x, iv.1 ≤ x → x < iv.2 →
      ∃ c ∈ allCovs, coversByte c x := by
  induction covs generalizing acc with
  | nil => exact hacc
  | cons c rest ih =>
    cases c with
    | dense lo hi =>
      simp only [collectDenseCoverages]
      apply ih _ (fun d hd => hsub d (List.mem_cons_of_mem _ hd))
      intro iv hiv x hx1 hx2
      split at hiv
      · cases hiv with
        | head => exact ⟨.dense lo hi, hsub _ List.mem_cons_self, hx1, hx2⟩
        | tail _ hm => exact hacc iv hm x hx1 hx2
      · exact hacc iv hiv x hx1 hx2
    | periodic _ _ _ _ =>
      exact ih acc (fun d hd => hsub d (List.mem_cons_of_mem _ hd)) hacc
    | sparse =>
      exact ih acc (fun d hd => hsub d (List.mem_cons_of_mem _ hd)) hacc

private theorem insertInterval_pairwise (iv : Nat × Nat) (xs : List (Nat × Nat))
    (hpw : xs.Pairwise (fun a b => a.1 ≤ b.1)) :
    (insertInterval iv xs).Pairwise (fun a b => a.1 ≤ b.1) := by
  induction xs with
  | nil => exact List.Pairwise.cons (by simp) List.Pairwise.nil
  | cons hd tl ih =>
    unfold insertInterval
    split
    · rename_i hle
      have ⟨hhd, _⟩ := List.pairwise_cons.mp hpw
      apply List.Pairwise.cons _ hpw
      intro y hy
      cases hy with
      | head => exact hle
      | tail _ htl => exact Nat.le_trans hle (hhd y htl)
    · rename_i hgt
      have ⟨hhd, htlPw⟩ := List.pairwise_cons.mp hpw
      apply List.Pairwise.cons _ (ih htlPw)
      intro y hy
      rw [mem_insertInterval] at hy
      rcases hy with rfl | hmem
      · omega
      · exact hhd y hmem

private theorem filterSortPeriodicGroup_spec (outerCount outerStride : Nat)
    (allCovs covs : List WriteCoverage) (acc : List (Nat × Nat))
    (hsub : ∀ c ∈ covs, c ∈ allCovs)
    (haccPw : acc.Pairwise (fun a b => a.1 ≤ b.1))
    (haccMem : ∀ p ∈ acc, WriteCoverage.periodic p.1 p.2 outerCount outerStride ∈ allCovs) :
    let res := filterSortPeriodicGroup outerCount outerStride covs acc
    res.Pairwise (fun a b => a.1 ≤ b.1) ∧
    (∀ p ∈ res, WriteCoverage.periodic p.1 p.2 outerCount outerStride ∈ allCovs) := by
  induction covs generalizing acc with
  | nil => exact ⟨haccPw, haccMem⟩
  | cons c rest ih =>
    cases c with
    | dense _ _ =>
      exact ih acc (fun d hd => hsub d (List.mem_cons_of_mem _ hd)) haccPw haccMem
    | sparse =>
      exact ih acc (fun d hd => hsub d (List.mem_cons_of_mem _ hd)) haccPw haccMem
    | periodic st len oc os =>
      simp only [filterSortPeriodicGroup]
      split
      · rename_i hcond
        simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hcond
        rcases hcond with ⟨⟨⟨⟨_, _⟩, _⟩, rfl⟩, rfl⟩
        apply ih _ (fun d hd => hsub d (List.mem_cons_of_mem _ hd))
          (insertInterval_pairwise (st, len) acc haccPw)
        intro p hp
        rw [mem_insertInterval] at hp
        rcases hp with rfl | hm
        · exact hsub _ List.mem_cons_self
        · exact haccMem p hm
      · exact ih acc (fun d hd => hsub d (List.mem_cons_of_mem _ hd)) haccPw haccMem

private theorem clusterPeriodicSorted_sound
    (outerCount outerStride : Nat) (allCovs : List WriteCoverage)
    (group : List (Nat × Nat)) (curBase? : Option Nat)
    (curInner acc : List (Nat × Nat))
    (hpw : group.Pairwise (fun a b => a.1 ≤ b.1))
    (hgrp : ∀ p ∈ group, WriteCoverage.periodic p.1 p.2 outerCount outerStride ∈ allCovs)
    (hbase : ∀ base, curBase? = some base →
      (∀ p ∈ group, base ≤ p.1) ∧
      (∀ iv ∈ curInner, ∃ len, iv.2 = iv.1 + len ∧
        WriteCoverage.periodic (base + iv.1) len outerCount outerStride ∈ allCovs))
    (hacc : ∀ iv ∈ acc, ∀ x, iv.1 ≤ x → x < iv.2 → ∃ c ∈ allCovs, coversByte c x) :
    ∀ iv ∈ clusterPeriodicSorted outerCount outerStride group curBase? curInner acc,
      ∀ x, iv.1 ≤ x → x < iv.2 → ∃ c ∈ allCovs, coversByte c x := by
  have hflush : ∀ base curInner' iv,
      (∀ ivIn ∈ curInner', ∃ len, ivIn.2 = ivIn.1 + len ∧
        WriteCoverage.periodic (base + ivIn.1) len outerCount outerStride ∈ allCovs) →
      periodicGroupCovered base outerCount outerStride curInner' = some iv →
      ∀ x, iv.1 ≤ x → x < iv.2 → ∃ c ∈ allCovs, coversByte c x := by
    intro base curInner' iv hcur hpg x hx1 hx2
    obtain ⟨j, hjLt, ivIn, hivMem, hlo, hhi⟩ :=
      periodicGroupCovered_sound base outerCount outerStride curInner' iv.1 iv.2 x hpg hx1 hx2
    obtain ⟨len, hlenEq, hcMem⟩ := hcur ivIn hivMem
    refine ⟨.periodic (base + ivIn.1) len outerCount outerStride, hcMem, j, hjLt, ?_, ?_⟩ <;> omega
  induction group generalizing curBase? curInner acc with
  | nil =>
    cases curBase? with
    | none => exact hacc
    | some base =>
      simp only [clusterPeriodicSorted]
      cases hpg : periodicGroupCovered base outerCount outerStride curInner with
      | none => exact hacc
      | some iv =>
        intro iv' hiv' x hx1 hx2
        cases hiv' with
        | head => exact hflush base curInner iv (hbase base rfl).2 hpg x hx1 hx2
        | tail _ hm => exact hacc iv' hm x hx1 hx2
  | cons p rest ih =>
    rcases p with ⟨st, len⟩
    have ⟨hpRest, hrestPw⟩ := List.pairwise_cons.mp hpw
    have hpMem : WriteCoverage.periodic st len outerCount outerStride ∈ allCovs :=
      hgrp (st, len) List.mem_cons_self
    have hrestGrp : ∀ q ∈ rest, WriteCoverage.periodic q.1 q.2 outerCount outerStride ∈ allCovs :=
      fun q hq => hgrp q (List.mem_cons_of_mem _ hq)
    cases curBase? with
    | none =>
      simp only [clusterPeriodicSorted]
      split
      · apply ih (some st) [(0, len)] acc hrestPw hrestGrp _ hacc
        intro b hb; cases hb
        refine ⟨fun q hq => hpRest q hq, ?_⟩
        intro iv hiv
        simp only [List.mem_singleton] at hiv
        subst hiv
        exact ⟨len, by simp, by simpa using hpMem⟩
      · exact ih none [] acc hrestPw hrestGrp (fun _ h => by cases h) hacc
    | some base =>
      have ⟨hgeAll, hcurInv⟩ := hbase base rfl
      have hbaseLeSt : base ≤ st := hgeAll (st, len) List.mem_cons_self
      simp only [clusterPeriodicSorted]
      split
      · have hacc' : ∀ iv ∈ (match periodicGroupCovered base outerCount outerStride curInner with
            | some iv => iv :: acc | none => acc),
            ∀ x, iv.1 ≤ x → x < iv.2 → ∃ c ∈ allCovs, coversByte c x := by
          cases hpg : periodicGroupCovered base outerCount outerStride curInner with
          | none => exact hacc
          | some iv =>
            intro iv' hiv' x hx1 hx2
            cases hiv' with
            | head => exact hflush base curInner iv hcurInv hpg x hx1 hx2
            | tail _ hm => exact hacc iv' hm x hx1 hx2
        split
        · apply ih (some st) [(0, len)] _ hrestPw hrestGrp _ hacc'
          intro b hb; cases hb
          refine ⟨fun q hq => hpRest q hq, ?_⟩
          intro iv hiv
          simp only [List.mem_singleton] at hiv
          subst hiv
          exact ⟨len, by simp, by simpa using hpMem⟩
        · exact ih none [] _ hrestPw hrestGrp (fun _ h => by cases h) hacc'
      · split
        · apply ih (some base) ((st - base, st - base + len) :: curInner) acc hrestPw hrestGrp _ hacc
          intro b hb; cases hb
          refine ⟨fun q hq => hgeAll q (List.mem_cons_of_mem _ hq), ?_⟩
          intro iv hiv
          cases hiv with
          | head =>
            have hadd : base + (st - base) = st := by omega
            exact ⟨len, rfl, by rw [hadd]; exact hpMem⟩
          | tail _ hm => exact hcurInv iv hm
        · apply ih (some base) curInner acc hrestPw hrestGrp _ hacc
          intro b hb; cases hb
          exact ⟨fun q hq => hgeAll q (List.mem_cons_of_mem _ hq), hcurInv⟩

private theorem collectPeriodicCoverages_sound
    (allCovs : List WriteCoverage) (keys acc : List (Nat × Nat))
    (hacc : ∀ iv ∈ acc, ∀ x, iv.1 ≤ x → x < iv.2 → ∃ c ∈ allCovs, coversByte c x) :
    ∀ iv ∈ collectPeriodicCoverages allCovs keys acc,
      ∀ x, iv.1 ≤ x → x < iv.2 → ∃ c ∈ allCovs, coversByte c x := by
  induction keys generalizing acc with
  | nil => exact hacc
  | cons key rest ih =>
    rcases key with ⟨oc, os⟩
    simp only [collectPeriodicCoverages]
    have ⟨hpw, hgrp⟩ := filterSortPeriodicGroup_spec oc os allCovs allCovs []
      (fun _ h => h) List.Pairwise.nil (fun _ h => by cases h)
    have hacc' := clusterPeriodicSorted_sound oc os allCovs _ none [] acc hpw hgrp
      (fun _ h => by cases h) hacc
    exact ih _ hacc'

/-- Soundness of `Vane.Verify.effectiveCoveredIntervals`: every byte offset `x` inside any
    interval emitted by `effectiveCoveredIntervals covs` is guaranteed written (`coversByte`)
    by at least one descriptor `c ∈ covs` (dense or periodic). -/
theorem effectiveCoveredIntervals_sound (covs : List WriteCoverage) (iv : Nat × Nat) (x : Nat)
    (hiv : iv ∈ effectiveCoveredIntervals covs) (hx1 : iv.1 ≤ x) (hx2 : x < iv.2) :
    ∃ c ∈ covs, coversByte c x := by
  unfold effectiveCoveredIntervals at hiv
  have hdense := collectDenseCoverages_sound covs covs [] (fun _ h => h) (fun _ h => by cases h)
  exact collectPeriodicCoverages_sound covs (collectPeriodicKeys covs [])
    (collectDenseCoverages covs []) hdense iv hiv x hx1 hx2

private theorem collectTaskScratchWriters_mem
    (k : Nat) (b : VerifiedDramAccess) (accs : Array VerifiedDramAccess) :
    ∀ rem j acc,
      rem = accs.size - j →
      (∀ p ∈ acc.toList, p.1 = k → ∃ ia < accs.size, accs[ia]! = p.2 ∧
        p.2.resource = .scratch ∧ p.2.port = .resDma ∧ p.2.isWrite = true ∧
        vIntervalsOverlap p.2.lo p.2.hi b.lo b.hi = true) →
      (∀ p ∈ (collectTaskScratchWriters k b accs j acc).toList,
        p ∈ acc.toList ∨
        (p.1 = k ∧ ∃ ia < accs.size, accs[ia]! = p.2 ∧
          p.2.resource = .scratch ∧ p.2.port = .resDma ∧ p.2.isWrite = true ∧
          vIntervalsOverlap p.2.lo p.2.hi b.lo b.hi = true)) := by
  intro rem
  induction rem with
  | zero =>
    intro j acc hrem _
    have hnot : ¬ j < accs.size := by omega
    unfold collectTaskScratchWriters
    simp only [hnot, ↓reduceDIte]
    intro p hp; exact Or.inl hp
  | succ r ih =>
    intro j acc hrem hacc
    have hj : j < accs.size := by omega
    unfold collectTaskScratchWriters
    simp only [hj, ↓reduceDIte]
    by_cases hcond : (accs[j].resource == .scratch && accs[j].port == .resDma &&
                      accs[j].isWrite && vIntervalsOverlap accs[j].lo accs[j].hi b.lo b.hi) = true
    · simp only [hcond, ↓reduceIte]
      simp only [Bool.and_eq_true] at hcond
      rcases hcond with ⟨⟨⟨hresB, hportB⟩, hwr⟩, hov⟩
      have hres : accs[j].resource = .scratch := barResource_beq_eq hresB
      have hport : accs[j].port = .resDma := dramAccessPort_beq_eq hportB
      have hacc' : ∀ p ∈ (acc.push (k, accs[j])).toList, p.1 = k →
          ∃ ia < accs.size, accs[ia]! = p.2 ∧
          p.2.resource = .scratch ∧ p.2.port = .resDma ∧ p.2.isWrite = true ∧
          vIntervalsOverlap p.2.lo p.2.hi b.lo b.hi = true := by
        intro p hp hpk
        simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hp
        rcases hp with hm | rfl
        · exact hacc p hm hpk
        · exact ⟨j, hj, getElem!_pos accs j hj, hres, hport, hwr, hov⟩
      intro p hp
      rcases ih (j + 1) (acc.push (k, accs[j])) (by omega) hacc' p hp with hm | hnew
      · simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hm
        rcases hm with hold | rfl
        · exact Or.inl hold
        · exact Or.inr ⟨rfl, j, hj, getElem!_pos accs j hj, hres, hport, hwr, hov⟩
      · exact Or.inr hnew
    · simp only [hcond, Bool.false_eq_true, ↓reduceIte]
      exact ih (j + 1) acc (by omega) hacc

private theorem collectPriorScratchWriters_mem
    (fps : Array VerifiedTaskFootprint) (b : VerifiedDramAccess) (hi : Nat) :
    ∀ rem k acc,
      rem = hi - k →
      (∀ p ∈ acc.toList, p.1 < hi ∧
        ∃ ia < fps[p.1]!.dramAccesses.size, fps[p.1]!.dramAccesses[ia]! = p.2 ∧
        p.2.resource = .scratch ∧ p.2.port = .resDma ∧ p.2.isWrite = true ∧
        vIntervalsOverlap p.2.lo p.2.hi b.lo b.hi = true) →
      ∀ p ∈ (collectPriorScratchWriters fps b k hi acc).toList,
        p.1 < hi ∧
        ∃ ia < fps[p.1]!.dramAccesses.size, fps[p.1]!.dramAccesses[ia]! = p.2 ∧
        p.2.resource = .scratch ∧ p.2.port = .resDma ∧ p.2.isWrite = true ∧
        vIntervalsOverlap p.2.lo p.2.hi b.lo b.hi = true := by
  intro rem
  induction rem with
  | zero =>
    intro k acc hrem hacc
    have hnot : ¬ k < hi := by omega
    unfold collectPriorScratchWriters
    simp only [hnot, ↓reduceDIte]
    exact hacc
  | succ r ih =>
    intro k acc hrem hacc
    have hk : k < hi := by omega
    unfold collectPriorScratchWriters
    simp only [hk, ↓reduceDIte]
    have hacc' : ∀ p ∈ (collectTaskScratchWriters k b fps[k]!.dramAccesses 0 acc).toList,
        p.1 < hi ∧
        ∃ ia < fps[p.1]!.dramAccesses.size, fps[p.1]!.dramAccesses[ia]! = p.2 ∧
        p.2.resource = .scratch ∧ p.2.port = .resDma ∧ p.2.isWrite = true ∧
        vIntervalsOverlap p.2.lo p.2.hi b.lo b.hi = true := by
      intro p hp
      have hstep := collectTaskScratchWriters_mem k b fps[k]!.dramAccesses
        fps[k]!.dramAccesses.size 0 acc (by omega)
        (fun q hq hqk => by
          have ⟨_, hex⟩ := hacc q hq
          rw [hqk] at hex
          exact hex) p hp
      rcases hstep with hold | ⟨hpk, hex⟩
      · exact hacc p hold
      · subst hpk
        exact ⟨hk, hex⟩
    exact ih (k + 1) _ (by omega) hacc'

/-- End-to-end S5 BAR 3 scratch read-before-write coverage & ordering soundness:
    whenever production `verifyProgramHazards` accepts a program `fps`, for every task
    `i < fps.size`, every BAR 3 scratch read `b` of task `i`, and every byte offset
    `x ∈ [b.lo, b.hi)`, there exists an earlier producer task `p < i` and a BAR 3 scratch
    `resDma` write `a` in task `p` such that:
    1. `coversByte a.coverage x` (byte `x` is guaranteed written by `a`),
    2. `vIntervalsOverlap a.lo a.hi b.lo b.hi = true`, and
    3. `DramOrdered fps p i a b` (ordered before consumer `i` via `HappensBefore` / hardware barrier). -/
theorem verifyProgramHazards_scratch_read_covered
    (fps : Array VerifiedTaskFootprint)
    (hok : verifyProgramHazards fps = .ok ())
    (i : Nat) (hi : i < fps.size)
    (ib : Nat) (hib : ib < fps[i]!.dramAccesses.size)
    (x : Nat) :
    let b := fps[i]!.dramAccesses[ib]!
    b.resource = .scratch → b.isWrite = false → b.lo ≤ x → x < b.hi →
    ∃ p < i, ∃ ia < fps[p]!.dramAccesses.size,
      let a := fps[p]!.dramAccesses[ia]!
      a.resource = .scratch ∧ a.port = .resDma ∧ a.isWrite = true ∧
      coversByte a.coverage x ∧
      vIntervalsOverlap a.lo a.hi b.lo b.hi = true ∧
      DramOrdered fps p i a b := by
  unfold verifyProgramHazards at hok
  dsimp only at hok
  obtain ⟨⟨⟩, _, hok⟩ := except_bind_ok hok
  obtain ⟨⟨⟩, h2, hok⟩ := except_bind_ok hok
  obtain ⟨⟨⟩, h3, _⟩ := except_bind_ok hok
  unfold verifyScratchReadIntegrity at h3
  unfold verifyPairwiseHazards at h2
  intro b hbRes hbRd hxLo hxHi
  have hiOk := checkRange_ok h3 i (Nat.zero_le i) hi
  have hibOk := checkRange_ok hiOk ib (Nat.zero_le ib) hib
  have hbCond : (b.resource == .scratch && !b.isWrite) = true := by
    rw [hbRes, hbRd]; rfl
  dsimp only at hibOk
  change (if (b.resource == .scratch && !b.isWrite) = true then _ else _) = Except.ok () at hibOk
  simp only [hbCond, ↓reduceIte] at hibOk
  obtain ⟨_, hibOk⟩ := except_if_do_throw_ok hibOk
  obtain ⟨_, hibOk⟩ := except_if_do_throw_ok hibOk
  obtain ⟨_, hibOk⟩ := except_if_do_throw_ok hibOk
  obtain ⟨_, hibOk⟩ := except_if_do_throw_ok hibOk
  obtain ⟨hcovNotFalse, _⟩ := except_if_do_throw_ok hibOk
  have hcovTrue : intervalsCover
      (effectiveCoveredIntervals ((collectPriorScratchWriters fps b 0 i #[]).toList.map (fun (_, a) => a.coverage)))
      b.lo b.hi = true := by
    cases hc : intervalsCover
        (effectiveCoveredIntervals ((collectPriorScratchWriters fps b 0 i #[]).toList.map (fun (_, a) => a.coverage)))
        b.lo b.hi with
    | true => rfl
    | false =>
      exact False.elim (hcovNotFalse (by
        show (!intervalsCover
          (effectiveCoveredIntervals ((collectPriorScratchWriters fps b 0 i #[]).toList.map (fun (_, a) => a.coverage)))
          b.lo b.hi) = true
        rw [hc]; rfl))
  obtain ⟨iv, hivMem, hivLo, hivHi⟩ := intervalsCover_sound _ b.lo b.hi x hcovTrue hxLo hxHi
  obtain ⟨cov, hcovMem, hcovX⟩ := effectiveCoveredIntervals_sound _ iv x hivMem hivLo hivHi
  rw [List.mem_map] at hcovMem
  obtain ⟨⟨p, a⟩, hpaMem, rfl⟩ := hcovMem
  have hpaSpec := collectPriorScratchWriters_mem fps b i i 0 #[] (by omega)
    (fun _ h => by simp at h) (p, a) hpaMem
  rcases hpaSpec with ⟨hpLt, ia, hia, haEq, haRes, haPort, haWr, hov⟩
  have hord : DramOrdered fps p i fps[p]!.dramAccesses[ia]! b := by
    have hi2Ok := checkRange_ok h2 i (by omega) hi
    have hp2Ok := checkRange_ok hi2Ok p (Nat.zero_le p) hpLt
    obtain ⟨⟨⟩, _, hdramOk⟩ := except_bind_ok hp2Ok
    have hia2Ok := checkRange_ok hdramOk ia (Nat.zero_le ia) hia
    have hib2Ok := checkRange_ok hia2Ok ib (Nat.zero_le ib) hib
    obtain ⟨_, _, _, hpeArr⟩ := vLatestEngArrays_eq fps
    have hboTrue : vDramPairOrdered fps (buildHBReachTables fps).1
        (vLatestEngBefore fps .pe i) p i fps[p]!.dramAccesses[ia]! b = true := by
      cases hbo : vDramPairOrdered fps (buildHBReachTables fps).1
          (vLatestEngBefore fps .pe i) p i fps[p]!.dramAccesses[ia]! b with
      | true => rfl
      | false =>
        have hresB : (fps[p]!.dramAccesses[ia]!.resource == b.resource) = true := by
          rw [haEq, haRes, hbRes]; rfl
        have hwrB : (fps[p]!.dramAccesses[ia]!.isWrite || b.isWrite) = true := by
          rw [haEq, haWr]; rfl
        have hov' : vIntervalsOverlap fps[p]!.dramAccesses[ia]!.lo fps[p]!.dramAccesses[ia]!.hi b.lo b.hi = true := by
          rw [haEq]; exact hov
        dsimp only at hib2Ok
        rw [hpeArr i hi] at hib2Ok
        change (if (fps[p]!.dramAccesses[ia]!.resource == b.resource &&
            (fps[p]!.dramAccesses[ia]!.isWrite || b.isWrite) &&
            vIntervalsOverlap fps[p]!.dramAccesses[ia]!.lo fps[p]!.dramAccesses[ia]!.hi b.lo b.hi) = true then
          if (!vDramPairOrdered fps (buildHBReachTables fps).1
            (vLatestEngBefore fps .pe i) p i fps[p]!.dramAccesses[ia]! b) = true then _
          else _ else _) = Except.ok () at hib2Ok
        rw [hresB, hwrB, hov', hbo] at hib2Ok
        cases hib2Ok
    exact vDramPairOrdered_sound fps p i hi fps[p]!.dramAccesses[ia]! b hboTrue
  refine ⟨p, hpLt, ia, hia, ?_, ?_, ?_, ?_, ?_, hord⟩
  · rw [haEq]; exact haRes
  · rw [haEq]; exact haPort
  · rw [haEq]; exact haWr
  · rw [haEq]; exact hcovX
  · rw [haEq]; exact hov

-- ============================================================================
-- 20. Compiler Tiling / Plan Legality Bridge (Vane.Matmul, Vane.Tiling, Vane.Conv)
-- ============================================================================

private theorem findFirstSome_eq_some {α β : Type} {xs : List α} {f : α → Option β} {y : β}
    (h : findFirstSome xs f = some y) : ∃ x ∈ xs, f x = some y := by
  induction xs with
  | nil => simp [findFirstSome] at h
  | cons hd tl ih =>
    unfold findFirstSome at h
    cases hhd : f hd with
    | some z =>
      simp only [hhd, Option.some.injEq] at h
      subst h
      exact ⟨hd, List.mem_cons_self, hhd⟩
    | none =>
      simp only [hhd] at h
      obtain ⟨x, hx, hfx⟩ := ih h
      exact ⟨x, List.mem_cons_of_mem _ hx, hfx⟩

private theorem mem_natRange_iff (lo hi x : Nat) :
    x ∈ natRange lo hi ↔ lo ≤ x ∧ x < hi := by
  have haux : ∀ d lo, d = hi - lo → (x ∈ natRange lo hi ↔ lo ≤ x ∧ x < hi) := by
    intro d
    induction d with
    | zero =>
      intro lo hd
      have hnot : ¬ lo < hi := by omega
      unfold natRange
      simp only [hnot, ↓reduceDIte, List.not_mem_nil, false_iff]
      omega
    | succ r ih =>
      intro lo hd
      have hlt : lo < hi := by omega
      unfold natRange
      simp only [hlt, ↓reduceDIte, List.mem_cons, ih (lo + 1) (by omega)]
      omega
  exact haux (hi - lo) lo rfl

/-- `ceilLog2 m` covers `m` (`m ≤ 2 ^ ceilLog2 m`) for all `m ≥ 1`, and `ceilLog2 m ≤ 4` for `m ≤ 16`. -/
theorem ceilLog2_spec (m : Nat) :
    (1 ≤ m → m ≤ 2 ^ ceilLog2 m) ∧ (m ≤ 16 → ceilLog2 m ≤ 4) := by
  constructor
  · intro hm
    unfold ceilLog2
    split
    · omega
    · have hpos : m - 1 ≠ 0 := by omega
      have hlt : m - 1 < 2 ^ (Nat.log2 (m - 1) + 1) := by
        have h := Nat.lt_log2_self (n := m - 1)
        exact Nat.log2_lt hpos |>.mp (Nat.lt_succ_self _)
      omega
  · intro hm16
    have hcases : m = 0 ∨ m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6 ∨ m = 7 ∨ m = 8 ∨
        m = 9 ∨ m = 10 ∨ m = 11 ∨ m = 12 ∨ m = 13 ∨ m = 14 ∨ m = 15 ∨ m = 16 := by omega
    rcases hcases with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
        rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

/-- `matmulChunkSize k` is always 64-byte aligned and covers `2 * k` fp16 bytes. -/
theorem matmulChunkSize_aligned (k : Nat) :
    matmulChunkSize k % 64 = 0 ∧ 2 * k ≤ matmulChunkSize k := by
  unfold matmulChunkSize
  exact ⟨alignUp_dvd (2 * k) 64 (by omega), alignUp_ge (2 * k) 64⟩

/-- Any `MatmulSpec` accepted by `validateMatmulSpec` satisfies all dimension and 64 KiB chunk bounds. -/
theorem validateMatmulSpec_bounds (spec : MatmulSpec)
    (h : validateMatmulSpec spec = .ok ()) :
    1 ≤ spec.b ∧ spec.b ≤ 65535 ∧
    1 ≤ spec.m ∧ spec.m ≤ 2048 ∧
    1 ≤ spec.k ∧ spec.k ≤ 65535 ∧
    1 ≤ spec.n ∧ spec.n ≤ 65535 ∧
    matmulChunkSize spec.k ≤ maxKernelChunkBytes := by
  unfold validateMatmulSpec at h
  obtain ⟨hb, h⟩ := except_if_do_throw_ok h
  obtain ⟨hm, h⟩ := except_if_do_throw_ok h
  obtain ⟨hk, h⟩ := except_if_do_throw_ok h
  obtain ⟨hn, h⟩ := except_if_do_throw_ok h
  obtain ⟨_, h⟩ := except_if_do_throw_ok h
  dsimp only at h
  obtain ⟨hcs, _⟩ := except_if_do_throw_ok h
  simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, not_or] at hb hm hk hn
  exact ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩

/-- Any scratch transpose planned by `planMatmulScratchTranspose` has a 64-byte-aligned scratch
    allocation covering the full `[B, cols, 1, rows]` fp16 output tensor span. -/
theorem planMatmulScratchTranspose_aligned_and_covers
    (b rows cols : Nat) (cp : ViewCopy) (reqScratch : Nat)
    (h : planMatmulScratchTranspose b rows cols = .ok (cp, reqScratch)) :
    reqScratch % 64 = 0 ∧ b * cols * alignUp (2 * rows) 64 ≤ reqScratch := by
  unfold planMatmulScratchTranspose at h
  obtain ⟨_, h1⟩ := except_if_do_throw_ok h
  split at h1
  · rename_i h11
    simp only [Bool.and_eq_true, beq_iff_eq] at h11
    rcases h11 with ⟨rfl, rfl⟩
    injection h1 with hpair
    injection hpair with _ hreq
    rw [← hreq]
    have hal : alignUp (2 * 1) 64 = 64 := by decide
    have hdma : Vane.Program.dmaLineBytes = 64 := rfl
    rw [hdma, hal]
    exact ⟨alignUp_dvd _ 64 (by omega), by simpa using alignUp_ge (b * 64) 64⟩
  · obtain ⟨_, h2⟩ := except_if_do_throw_ok h1
    injection h2 with hpair
    injection hpair with _ hreq
    rw [← hreq]
    have hdma : Vane.Program.dmaLineBytes = 64 := rfl
    rw [hdma]
    exact ⟨alignUp_dvd _ 64 (by omega), Nat.le_trans (Nat.le_max_left _ _) (alignUp_ge _ 64)⟩

private theorem maxEndOf_le (rs : List Range) (cap : Nat) (h : ∀ r ∈ rs, r.hi ≤ cap) :
    maxEndOf rs ≤ cap := by
  unfold maxEndOf
  have haux : ∀ acc, acc ≤ cap → rs.foldl (fun a r => max a r.hi) acc ≤ cap := by
    induction rs with
    | nil => intro acc ha; exact ha
    | cons r rest ih =>
      intro acc ha
      simp only [List.foldl_cons]
      have hr : r.hi ≤ cap := h r List.mem_cons_self
      exact ih (fun q hq => h q (List.mem_cons_of_mem _ hq)) (max acc r.hi) (by omega)
  exact haux 0 (Nat.zero_le cap)

private theorem option_ite_none_eq_some {α : Type} {c : Prop} [Decidable c] {x : Option α} {v : α}
    (h : (if c then x else none) = some v) : c ∧ x = some v := by
  by_cases hc : c
  · simp [hc] at h; exact ⟨hc, h⟩
  · simp [hc] at h

/-- Direct legality bridge for production `chooseMatmulTaskPlan`: whenever it returns `.ok plan`,
    the plan satisfies every structural, bitfield, coverage, strip-width, and 2 MiB L2 capacity
    bound enforced by `Vane/Verify.lean`. -/
theorem chooseMatmulTaskPlan_legal
    (target : Target) (m k n : Nat) (srcIsL2 resIsL2 : Bool)
    (occupied : List Range) (plan : MatmulTaskPlan)
    (hok : chooseMatmulTaskPlan target m k n srcIsL2 resIsL2 occupied = .ok plan) :
    1 ≤ m ∧ m ≤ 16 ∧
    1 ≤ k ∧ 1 ≤ n ∧ n ≤ 65535 ∧
    plan.tiling.f = ceilLog2 m ∧
    plan.tiling.logNE = plan.tiling.f ∧
    plan.tiling.f ≤ 4 ∧
    m ≤ 2 ^ plan.tiling.f ∧
    plan.tiling.fat = 0 ∧
    plan.tiling.ocg = 0 ∧
    plan.tiling.tileHeight = 1 ∧
    plan.tiling.shPref ≤ plan.tiling.shMax ∧
    plan.tiling.shMax ≤ 4 ∧
    8 ≤ stripWidth plan.tiling ∧ stripWidth plan.tiling ≤ 256 ∧
    ((∀ r ∈ occupied, r.hi ≤ l2CapacityBytes) → plan.l2TaskPeak ≤ l2CapacityBytes) := by
  unfold chooseMatmulTaskPlan at hok
  obtain ⟨hdims, hok⟩ := except_if_err_ok hok
  simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, not_or] at hdims
  have hm1 : 1 ≤ m := by omega
  have hm16 : m ≤ 16 := by omega
  have hk1 : 1 ≤ k := by omega
  have hn1 : 1 ≤ n := by omega
  have hn65 : n ≤ 65535 := by omega
  dsimp only at hok
  split at hok
  · rename_i p hch
    simp only [Except.ok.injEq] at hok
    subst hok
    obtain ⟨wantDirect, _, h1⟩ := findFirstSome_eq_some hch
    obtain ⟨shPref, hshPrefMem, h2⟩ := findFirstSome_eq_some h1
    split at h2
    · rename_i hstrip
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hstrip
      split at h2
      · rename_i hshMaxStart
        obtain ⟨shMax, hshMaxMem, hcand⟩ := findFirstSome_eq_some h2
        rw [mem_natRange_iff] at hshPrefMem hshMaxMem
        unfold tryMatmulPlanCandidate at hcand
        dsimp only at hcand
        obtain ⟨_, hcand⟩ := option_ite_none_eq_some hcand
        obtain ⟨⟨sb, rb, pk⟩, hpl, hpEq⟩ := Option.map_eq_some_iff.mp hcand
        subst hpEq
        dsimp only
        have hcl := ceilLog2_spec m
        have hpeak : (∀ r ∈ occupied, r.hi ≤ l2CapacityBytes) → pk ≤ l2CapacityBytes := by
          intro hocc
          have hmaxOcc := maxEndOf_le occupied l2CapacityBytes hocc
          cases srcIsL2 <;> cases resIsL2 <;> simp only [Bool.and_true, Bool.and_false, Bool.not_true,
            Bool.not_false, ↓reduceIte, Bool.false_eq_true] at hpl
          · split at hpl
            · cases hpl
            · rename_i sb' hf1
              split at hpl
              · cases hpl
              · rename_i rb' hf2
                simp only [Option.some.injEq, Prod.mk.injEq] at hpl
                rcases hpl with ⟨rfl, rfl, rfl⟩
                have hs1 := (findFreeOffset_sound _ _ _ _ hf1).1
                have hs2 := (findFreeOffset_sound _ _ _ _ hf2).1
                have ha1 := alignUp_ge (k * (2 * min (alignUp n 8) (max ((128 : Nat) >>> shPref) 32))) Vane.Program.dmaLineBytes
                have ha2 := alignUp_ge (if decide (target.profile.directThreshold 0 shMax ≥ min 32 n) = true then ((1 : Nat) <<< ceilLog2 m) * 512 else m * (2 * min (alignUp n 8) (max ((128 : Nat) >>> shPref) 32))) Vane.Program.dmaLineBytes
                omega
          · obtain ⟨sb', hf1, hpl⟩ := Option.map_eq_some_iff.mp hpl
            simp only [Prod.mk.injEq] at hpl
            rcases hpl with ⟨rfl, rfl, rfl⟩
            have hs1 := (findFreeOffset_sound _ _ _ _ hf1).1
            have ha1 := alignUp_ge (k * (2 * min (alignUp n 8) (max ((128 : Nat) >>> shPref) 32))) Vane.Program.dmaLineBytes
            omega
          · obtain ⟨rb', hf1, hpl⟩ := Option.map_eq_some_iff.mp hpl
            simp only [Prod.mk.injEq] at hpl
            rcases hpl with ⟨rfl, rfl, rfl⟩
            have hs1 := (findFreeOffset_sound _ _ _ _ hf1).1
            have ha1 := alignUp_ge (if decide (target.profile.directThreshold 0 shMax ≥ min 32 n) = true then ((1 : Nat) <<< ceilLog2 m) * 512 else m * (2 * min (alignUp n 8) (max ((128 : Nat) >>> shPref) 32))) Vane.Program.dmaLineBytes
            omega
          · simp only [Option.some.injEq, Prod.mk.injEq] at hpl
            rcases hpl with ⟨_, _, rfl⟩
            exact hmaxOcc
        refine ⟨hm1, hm16, hk1, hn1, hn65, rfl, rfl, hcl.2 hm16, hcl.1 hm1, rfl, rfl, rfl,
          by omega, by omega, ?_, ?_, hpeak⟩
        · simpa [stripWidth] using hstrip.1
        · simpa [stripWidth] using hstrip.2
      · simp at h2
    · simp at h2
  · cases hok

/-- Full legality bounds extracted from `validateConvTiling`: covers `logNE`, `fat`, `f`, `wu`,
    `ocg`, `cout` coverage, `stripWidth`, `shPref`, `shMax`, `tileHeight`, 64-byte L2 alignment,
    and the 2 MiB L2 SRAM capacity cap. -/
theorem validateConvTiling_full_bounds
    (target : Target) (sh : ConvSpec) (t : Tiling)
    (extraChunkBytes : Nat := 0) (hasBiasInKernel : Bool := false)
    (h : validateConvTiling target sh t extraChunkBytes hasBiasInKernel = .ok ()) :
    t.logNE ≤ 4 ∧
    t.fat ≤ 1 ∧
    t.f + t.fat ≤ t.logNE ∧
    t.wu ≤ 3 ∧
    (sh.effMode = .depthwise → t.ocg = 0) ∧
    (sh.effMode = .dense →
      t.ocg ≤ (if sh.kernelSize == 3 && sh.stride == 1 then target.profile.dense3x3Stride1MaxOcg else 4)) ∧
    (sh.effMode = .dense → sh.kernelSize = 1 → sh.cout ≤ target.profile.conv1x1MaxCout t.f t.ocg) ∧
    8 ≤ stripWidth t ∧ stripWidth t ≤ 256 ∧
    t.shPref ≤ t.shMax ∧ t.shMax ≤ 4 ∧
    1 ≤ t.tileHeight ∧ t.tileHeight ≤ sh.outH ∧
    resBaseBytesConv sh t % 64 = 0 ∧
    l2UseBytesConv target sh t ≤ l2CapacityBytes := by
  unfold validateConvTiling at h
  obtain ⟨⟨⟩, hsaf, hpol⟩ := except_bind_ok h
  unfold validateConvSafety at hsaf
  obtain ⟨⟨⟩, _, hsaf⟩ := except_bind_ok hsaf
  obtain ⟨hLogNE, hsaf⟩ := except_if_err_ok hsaf
  obtain ⟨hFat, hsaf⟩ := except_if_err_ok hsaf
  obtain ⟨hFfat, hsaf⟩ := except_if_err_ok hsaf
  obtain ⟨hWu, hsaf⟩ := except_if_err_ok hsaf
  obtain ⟨⟨⟩, hocg, hsaf⟩ := except_bind_ok hsaf
  obtain ⟨⟨⟩, hcov, hsaf⟩ := except_bind_ok hsaf
  dsimp only at hsaf
  obtain ⟨hstrip, hsaf⟩ := except_if_err_ok hsaf
  obtain ⟨hSh, hsaf⟩ := except_if_err_ok hsaf
  obtain ⟨hShMax, hsaf⟩ := except_if_err_ok hsaf
  obtain ⟨hTh, hsaf⟩ := except_if_err_ok hsaf
  obtain ⟨hL2, _⟩ := except_if_err_ok hsaf
  simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hstrip
  have hThNe : t.tileHeight ≠ 0 := by simpa using hTh
  unfold validateConvChooserPolicy at hpol
  dsimp only at hpol
  obtain ⟨_, hpol⟩ := except_if_do_throw_ok hpol
  obtain ⟨_, hpol⟩ := except_if_do_throw_ok hpol
  obtain ⟨_, hpol⟩ := except_if_do_throw_ok hpol
  obtain ⟨hthHo, _⟩ := except_if_do_throw_ok hpol
  have hocgDw : sh.effMode = .depthwise → t.ocg = 0 := by
    intro hdw
    unfold validateConvOcgSafety at hocg
    rw [hdw] at hocg
    obtain ⟨h0, _⟩ := except_if_err_ok hocg
    simpa using h0
  have hocgDense : sh.effMode = .dense →
      t.ocg ≤ (if sh.kernelSize == 3 && sh.stride == 1 then target.profile.dense3x3Stride1MaxOcg else 4) := by
    intro hd
    unfold validateConvOcgSafety at hocg
    rw [hd] at hocg
    dsimp only at hocg
    obtain ⟨hle, _⟩ := except_if_err_ok hocg
    omega
  have hcov1x1 : sh.effMode = .dense → sh.kernelSize = 1 →
      sh.cout ≤ target.profile.conv1x1MaxCout t.f t.ocg := by
    intro hd hk1
    unfold validateConvCoverageSafety at hcov
    rw [hd] at hcov
    dsimp only at hcov
    simp only [hk1, beq_self_eq_true, ↓reduceIte] at hcov
    obtain ⟨hle, _⟩ := except_if_err_ok hcov
    omega
  exact ⟨by omega, by omega, by omega, by omega, hocgDw, hocgDense, hcov1x1,
    by omega, by omega, by omega, by omega, by omega, by omega,
    alignUp_dvd _ 64 (by omega), by omega⟩

/-- Any tiling returned by production `chooseConvTiling` passes `validateConvTiling`. -/
theorem chooseConvTiling_valid
    (target : Target) (sh : ConvSpec) (t : Tiling)
    (extraChunkBytes : Nat := 0) (hasBiasInKernel : Bool := false)
    (h : chooseConvTiling target sh extraChunkBytes hasBiasInKernel = .ok t) :
    validateConvTiling target sh t extraChunkBytes hasBiasInKernel = .ok () := by
  unfold chooseConvTiling at h
  obtain ⟨⟨⟩, _, h⟩ := except_bind_ok h
  dsimp only at h
  split at h
  · rename_i cand hch
    simp only [Except.ok.injEq] at h
    subst h
    obtain ⟨logNE, _, h1⟩ := findFirstSome_eq_some hch
    obtain ⟨fat, _, h2⟩ := findFirstSome_eq_some h1
    obtain ⟨f, _, h3⟩ := findFirstSome_eq_some h2
    split at h3
    · split at h3
      · cases h3
      · obtain ⟨shPref, _, h4⟩ := findFirstSome_eq_some h3
        split at h4
        · split at h4
          · obtain ⟨shMax, _, h5⟩ := findFirstSome_eq_some h4
            split at h5
            · split at h5
              · rename_i hv
                simp only [Option.some.injEq] at h5
                subst h5
                exact hv
              · cases h5
            · simp at h5
          · simp at h4
        · simp at h4
    · simp at h3
  · cases h

/-- Direct legality bridge for production `chooseConvTiling`: every `.ok t` satisfies all hardware
    safety bounds, `ocg` & channel-coverage bounds, strip width, alignment, and the 2 MiB L2 cap. -/
theorem chooseConvTiling_legal
    (target : Target) (sh : ConvSpec) (t : Tiling)
    (extraChunkBytes : Nat := 0) (hasBiasInKernel : Bool := false)
    (h : chooseConvTiling target sh extraChunkBytes hasBiasInKernel = .ok t) :
    t.logNE ≤ 4 ∧ t.fat ≤ 1 ∧ t.f + t.fat ≤ t.logNE ∧ t.wu ≤ 3 ∧
    (sh.effMode = .depthwise → t.ocg = 0) ∧
    (sh.effMode = .dense →
      t.ocg ≤ (if sh.kernelSize == 3 && sh.stride == 1 then target.profile.dense3x3Stride1MaxOcg else 4)) ∧
    (sh.effMode = .dense → sh.kernelSize = 1 → sh.cout ≤ target.profile.conv1x1MaxCout t.f t.ocg) ∧
    8 ≤ stripWidth t ∧ stripWidth t ≤ 256 ∧
    t.shPref ≤ t.shMax ∧ t.shMax ≤ 4 ∧
    1 ≤ t.tileHeight ∧ t.tileHeight ≤ sh.outH ∧
    resBaseBytesConv sh t % 64 = 0 ∧
    l2UseBytesConv target sh t ≤ l2CapacityBytes :=
  validateConvTiling_full_bounds target sh t extraChunkBytes hasBiasInKernel
    (chooseConvTiling_valid target sh t extraChunkBytes hasBiasInKernel h)

/-- Direct legality bridge for 1x1 conv tiling (`chooseTiling`): every `.ok t` satisfies
    `ocg ≤ 4`, `Cout ≤ 2^(f + ocg + bonus)`, alignment, and `l2UseBytes ≤ 2 MiB`. -/
theorem chooseTiling_conv1x1_legal
    (target : Target) (sh : Conv1x1Shape) (t : Tiling)
    (h : chooseTiling target sh = .ok t) :
    t.logNE ≤ 4 ∧ t.fat ≤ 1 ∧ t.f + t.fat ≤ t.logNE ∧ t.wu ≤ 3 ∧
    t.ocg ≤ 4 ∧
    sh.cout ≤ target.profile.conv1x1MaxCout t.f t.ocg ∧
    8 ≤ stripWidth t ∧ stripWidth t ≤ 256 ∧
    t.shPref ≤ t.shMax ∧ t.shMax ≤ 4 ∧
    1 ≤ t.tileHeight ∧ t.tileHeight ≤ sh.h ∧
    l2UseBytes target sh t ≤ l2CapacityBytes := by
  have hb := chooseConvTiling_legal target sh.toConvSpec t 0 false h
  have heff : sh.toConvSpec.effMode = .dense := rfl
  have hk1 : sh.toConvSpec.kernelSize = 1 := rfl
  have hocg4 := hb.2.2.2.2.2.1 heff
  have hcov := hb.2.2.2.2.2.2.1 heff hk1
  exact ⟨hb.1, hb.2.1, hb.2.2.1, hb.2.2.2.1, hocg4, hcov,
    hb.2.2.2.2.2.2.2.1, hb.2.2.2.2.2.2.2.2.1, hb.2.2.2.2.2.2.2.2.2.1, hb.2.2.2.2.2.2.2.2.2.2.1,
    hb.2.2.2.2.2.2.2.2.2.2.2.1, hb.2.2.2.2.2.2.2.2.2.2.2.2.1, hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

/-- Direct legality bridge for 3x3 conv tiling (`chooseConv3x3Tiling`): every `.ok t` satisfies
    `ocg` bounds, strip width, tile height, and `l2UseBytes3x3 ≤ 2 MiB`. -/
theorem chooseConv3x3Tiling_legal
    (target : Target) (sh : Conv3x3Shape) (t : Tiling)
    (h : chooseConv3x3Tiling target sh = .ok t) :
    t.logNE ≤ 4 ∧ t.fat ≤ 1 ∧ t.f + t.fat ≤ t.logNE ∧ t.wu ≤ 3 ∧
    (sh.effMode = .depthwise → t.ocg = 0) ∧
    (sh.effMode = .dense →
      t.ocg ≤ (if sh.stride == 1 then target.profile.dense3x3Stride1MaxOcg else 4)) ∧
    8 ≤ stripWidth t ∧ stripWidth t ≤ 256 ∧
    t.shPref ≤ t.shMax ∧ t.shMax ≤ 4 ∧
    1 ≤ t.tileHeight ∧ t.tileHeight ≤ sh.outH ∧
    l2UseBytes3x3 target sh t ≤ l2CapacityBytes := by
  have hb := chooseConvTiling_legal target sh.toConvSpec t 0 false h
  exact ⟨hb.1, hb.2.1, hb.2.2.1, hb.2.2.2.1, hb.2.2.2.2.1, hb.2.2.2.2.2.1,
    hb.2.2.2.2.2.2.2.1, hb.2.2.2.2.2.2.2.2.1, hb.2.2.2.2.2.2.2.2.2.1, hb.2.2.2.2.2.2.2.2.2.2.1,
    hb.2.2.2.2.2.2.2.2.2.2.2.1, hb.2.2.2.2.2.2.2.2.2.2.2.2.1, hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2⟩

private theorem foldl_add_replicate (n s init : Nat) :
    (Array.replicate n s).foldl (· + ·) init = init + n * s := by
  induction n generalizing init with
  | zero => simp
  | succ k ih =>
    have hpush : Array.replicate (k + 1) s = (Array.replicate k s).push s := by
      apply Array.ext'
      simp [List.replicate_succ']
    rw [hpush, Array.foldl_push, ih]
    rw [Nat.succ_mul]
    omega

/-- Soundness & output-channel conservation for production `chooseConv1x1OutputSlices`:
    whenever `.ok slices` is returned, `1 ≤ slices.size ≤ 255`, the slice widths sum to exact
    `cout`, and every slice width `s ∈ slices` admits a legal single-task `chooseConvTiling`. -/
theorem chooseConv1x1OutputSlices_sound
    (target : Target) (h w cin cout : Nat)
    (extraChunkBytes : Nat := 0) (hasBias : Bool := false)
    (slices : Array Nat)
    (hok : chooseConv1x1OutputSlices target h w cin cout extraChunkBytes hasBias = .ok slices) :
    1 ≤ slices.size ∧ slices.size ≤ 255 ∧
    slices.foldl (· + ·) 0 = cout ∧
    ∀ s ∈ slices.toList,
      1 ≤ s ∧ s ≤ cout ∧
      (chooseConvTiling target { kernelSize := 1, mode := .dense, h := h, w := w, cin := cin, cout := s }
        extraChunkBytes (hasBias && s > 1)).isOk = true := by
  unfold chooseConv1x1OutputSlices at hok
  obtain ⟨⟨⟩, hsh, hok⟩ := except_bind_ok hok
  have hcoutPos : 1 ≤ cout := by
    unfold validateConvShape at hsh
    obtain ⟨_, hsh⟩ := except_if_do_throw_ok hsh
    obtain ⟨_, hsh⟩ := except_if_do_throw_ok hsh
    obtain ⟨_, hsh⟩ := except_if_do_throw_ok hsh
    obtain ⟨_, hsh⟩ := except_if_do_throw_ok hsh
    obtain ⟨hc, _⟩ := except_if_do_throw_ok hsh
    simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, not_or] at hc
    omega
  dsimp only at hok
  split at hok
  · rename_i hfit
    simp only [pure, Except.pure, Except.ok.injEq] at hok
    subst hok
    refine ⟨by simp, by simp, by simp, ?_⟩
    intro s hs
    simp only [List.mem_singleton] at hs
    subst hs
    exact ⟨hcoutPos, Nat.le_refl _, hfit⟩
  · obtain ⟨_, hok⟩ := except_if_do_throw_ok hok
    split at hok
    · rename_i sl hch
      simp only [pure, Except.pure, Except.ok.injEq] at hok
      subst hok
      obtain ⟨alignStep, _, h1⟩ := findFirstSome_eq_some hch
      obtain ⟨idx, _, htry⟩ := findFirstSome_eq_some h1
      generalize (min (cout - 1) (target.profile.conv1x1MaxCout 4 4) / alignStep) * alignStep - idx * alignStep = s at htry
      split at htry
      · cases htry
      · rename_i hsBounds
        simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, not_or] at hsBounds
        have hsPos : 1 ≤ s := by omega
        have hsLt : s < cout := by omega
        split at htry
        · cases htry
        · rename_i hnum255
          split at htry
          · rename_i hcan
            simp only [Option.some.injEq] at htry
            subst htry
            simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at hcan
            rcases hcan with ⟨hfitS, hremCase⟩
            have hdivMod : s * (cout / s) + cout % s = cout := Nat.div_add_mod cout s
            have hmulComm : (cout / s) * s = s * (cout / s) := Nat.mul_comm _ _
            have hqPos : 1 ≤ cout / s := Nat.div_pos (by omega) hsPos
            have hceilEq : ceilDiv cout s = cout / s + (if cout % s = 0 then 0 else 1) := by
              unfold ceilDiv
              have hs0 : (s == 0) = false := by simp; omega
              rw [hs0]; simp only [Bool.false_eq_true, ↓reduceIte]
              by_cases hr0 : cout % s = 0
              · simp only [hr0, ↓reduceIte, Nat.add_zero]
                have h1 : cout + s - 1 = s * (cout / s) + (s - 1) := by omega
                rw [h1, Nat.mul_add_div hsPos, show (s - 1) / s = 0 from Nat.div_eq_of_lt (by omega)]
                omega
              · simp only [hr0, ↓reduceIte]
                have hrLt : cout % s < s := Nat.mod_lt cout hsPos
                have hrPos : 1 ≤ cout % s := by omega
                have h1 : cout + s - 1 = s * (cout / s + 1) + (cout % s - 1) := by
                  rw [Nat.mul_add, Nat.mul_one]; omega
                rw [h1, Nat.mul_add_div hsPos, show (cout % s - 1) / s = 0 from Nat.div_eq_of_lt (by omega)]
            by_cases hr0 : cout % s = 0
            · simp only [hr0, beq_self_eq_true, ↓reduceIte] at hceilEq ⊢
              refine ⟨by simp; omega, by simp; omega, ?_, ?_⟩
              · rw [foldl_add_replicate]; omega
              · intro x hx
                simp only [Array.toList_replicate, List.mem_replicate] at hx
                rcases hx with ⟨_, rfl⟩
                exact ⟨hsPos, by omega, hfitS⟩
            · have hr0B : (cout % s == 0) = false := by simp [hr0]
              simp only [hr0, hr0B, Bool.false_eq_true, ↓reduceIte] at hceilEq ⊢
              have hfitRem : (chooseConvTiling target
                  { kernelSize := 1, mode := .dense, h := h, w := w, cin := cin, cout := cout % s }
                  extraChunkBytes (hasBias && decide (cout % s > 1))).isOk = true := by
                rcases hremCase with heq | hokR
                · exact absurd heq hr0
                · exact hokR
              refine ⟨by simp [Array.size_push], by simp [Array.size_push]; omega, ?_, ?_⟩
              · rw [Array.foldl_push, foldl_add_replicate]; omega
              · intro x hx
                simp only [Array.toList_push, Array.toList_replicate, List.mem_append,
                           List.mem_replicate, List.mem_singleton] at hx
                rcases hx with ⟨_, rfl⟩ | rfl
                · exact ⟨hsPos, by omega, hfitS⟩
                · have hrLt : cout % s < s := Nat.mod_lt cout hsPos
                  exact ⟨by omega, by omega, hfitRem⟩
          · cases htry
    · cases hok

-- ============================================================================
-- 21. Lean-Native SafeTensors & ONNX Frontend Soundness Invariants
-- ============================================================================

/-- Every recognized SafeTensors dtype has positive element width in `[1, 8]` bytes. -/
theorem safeTensors_dtypeBytes_pos (dtype : String) (b : Nat) (h : dtypeBytes? dtype = some b) :
    1 ≤ b ∧ b ≤ 8 := by
  unfold dtypeBytes? at h
  split at h <;> simp at h <;> omega

/-- A validated SafeTensors span `[beginOff, endOff)` satisfies exact in-bounds slicing
    `dataBase + beginOff ≤ dataBase + endOff ≤ fileSize`. -/
theorem safeTensors_validSpan_bounds (dataBase fileSize : Nat) (s : TensorSpan)
    (h : validSpan dataBase fileSize s = true) :
    s.beginOff ≤ s.endOff ∧ dataBase + s.beginOff ≤ dataBase + s.endOff ∧ dataBase + s.endOff ≤ fileSize := by
  simp [validSpan] at h
  omega

/-- A validated SafeTensors span whose byte length equals `numElems * elemBytes` fits
    within `fileSize` without overflow. -/
theorem safeTensors_validSpan_byteLen (dataBase fileSize numElems elemBytes : Nat) (s : TensorSpan)
    (h : validSpan dataBase fileSize s = true)
    (hlen : s.endOff - s.beginOff = numElems * elemBytes) :
    dataBase + s.beginOff + numElems * elemBytes ≤ fileSize := by
  simp [validSpan] at h
  omega

/-- Every span in a list is individually ordered (`beginOff ≤ endOff`). -/
def allSpansOrdered : List TensorSpan → Prop
  | [] => True
  | s :: rest => s.beginOff ≤ s.endOff ∧ allSpansOrdered rest

/-- If a sorted span list passes adjacent non-overlap checking, the head span ends at or before
    every subsequent span begins. -/
theorem safeTensors_checkSorted_head_le_all (a : TensorSpan) (rest : List TensorSpan)
    (hCheck : checkSortedSpansNonOverlapping (a :: rest) = true)
    (hOrd : allSpansOrdered (a :: rest)) :
    ∀ b ∈ rest, a.endOff ≤ b.beginOff := by
  induction rest generalizing a with
  | nil =>
    intro b hb
    simp at hb
  | cons c cs ih =>
    simp [checkSortedSpansNonOverlapping] at hCheck
    simp [allSpansOrdered] at hOrd
    intro b hb
    rcases List.mem_cons.mp hb with rfl | hmem
    · exact hCheck.1
    · have hcb : c.endOff ≤ b.beginOff := ih c hCheck.2 ⟨hOrd.2.1, hOrd.2.2⟩ b hmem
      omega

/-- Adjacent non-overlap checking on an ordered span list guarantees pairwise disjointness
    for every pair of distinct indices `i < j`. -/
theorem safeTensors_checkSorted_pairwise_disjoint (spans : List TensorSpan)
    (hCheck : checkSortedSpansNonOverlapping spans = true)
    (hOrd : allSpansOrdered spans)
    (i j : Nat) (hi : i < spans.length) (hj : j < spans.length) (hij : i < j) :
    spansDisjoint spans[i] spans[j] = true := by
  induction spans generalizing i j with
  | nil => simp at hi
  | cons a rest ih =>
    cases i with
    | zero =>
      cases j with
      | zero => omega
      | succ j' =>
        have hmem : rest[j']'(by simp at hj; omega) ∈ rest := List.getElem_mem _
        have hle : a.endOff ≤ (rest[j']'(by simp at hj; omega)).beginOff :=
          safeTensors_checkSorted_head_le_all a rest hCheck hOrd _ hmem
        simp [spansDisjoint]
        left
        exact hle
    | succ i' =>
      cases j with
      | zero => omega
      | succ j' =>
        have hCheckRest : checkSortedSpansNonOverlapping rest = true := by
          cases rest with
          | nil => rfl
          | cons c cs =>
            simp [checkSortedSpansNonOverlapping] at hCheck
            exact hCheck.2
        simp [allSpansOrdered] at hOrd
        exact ih hCheckRest hOrd.2 i' j' (by simp at hi; omega) (by simp at hj; omega) (by omega)

/-- Sum of `n` equal split chunks of size `k` equals `n * k`. -/
theorem split_replicate_sum (n k : Nat) : sumSizes (List.replicate n k) = n * k := by
  induction n with
  | zero => simp [sumSizes]
  | succ m ih =>
    simp [List.replicate_succ, sumSizes, ih, Nat.succ_mul]
    omega

/-- `sumSizes` distributes over list append. -/
theorem split_append_sum (xs ys : List Nat) : sumSizes (xs ++ ys) = sumSizes xs + sumSizes ys := by
  induction xs with
  | nil => simp [sumSizes]
  | cons x rest ih =>
    simp [sumSizes, ih]
    omega

/-- `computeDefaultSplitSizes` produces exact partition sizes summing to `dim` with every chunk `≥ 1`. -/
theorem split_defaultSizes_sound (dim n : Nat) (sizes : List Nat)
    (h : computeDefaultSplitSizes dim n = .ok sizes) :
    sizes.length = n ∧ sumSizes sizes = dim ∧ ∀ s ∈ sizes, 1 ≤ s := by
  unfold computeDefaultSplitSizes at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · split at h
      · injection h with hEq
        subst hEq
        rename_i hn0 hlt hmod
        simp at hn0 hlt hmod
        have hnPos : n > 0 := by omega
        have hdiv : n * (dim / n) = dim := Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hmod)
        refine ⟨by simp, ?_, ?_⟩
        · rw [split_replicate_sum, hdiv]
        · intro s hs
          rw [List.mem_replicate] at hs
          rw [hs.2]
          exact Nat.div_pos (by omega) hnPos
      · dsimp only at h
        split at h
        · injection h with hEq
          subst hEq
          rename_i hn0 hlt hmod hpre
          simp at hn0 hlt hmod
          have hnPos : n > 0 := by omega
          have hchunkPos : 1 ≤ ceilDiv dim n := by
            unfold ceilDiv
            have hnEq : (n == 0) = false := by simp; omega
            rw [hnEq]
            simp only [Bool.false_eq_true, ↓reduceIte]
            exact Nat.div_pos (by omega) hnPos
          refine ⟨by simp; omega, ?_, ?_⟩
          · rw [split_append_sum, split_replicate_sum]
            simp [sumSizes]
            have hcomm : (n - 1) * ceilDiv dim n = ceilDiv dim n * (n - 1) := Nat.mul_comm _ _
            omega
          · intro s hs
            rw [List.mem_append] at hs
            rcases hs with h1 | h2
            · rw [List.mem_replicate] at h1
              rw [h1.2]
              exact hchunkPos
            · simp at h2
              omega
        · simp at h

/-- `sliceIntervalsAux` preserves list length. -/
theorem split_intervals_length (acc : Nat) (sizes : List Nat) :
    (sliceIntervalsAux acc sizes).length = sizes.length := by
  induction sizes generalizing acc with
  | nil => rfl
  | cons sz rest ih =>
    simp [sliceIntervalsAux, ih]

/-- Every interval `(lo, hi)` emitted by `sliceIntervalsAux acc sizes` for positive sizes
    lies strictly within `[acc, acc + sumSizes sizes]` with `lo < hi`. -/
theorem split_intervals_bounds (acc : Nat) (sizes : List Nat) (hPos : ∀ s ∈ sizes, 1 ≤ s) :
    ∀ p ∈ sliceIntervalsAux acc sizes, acc ≤ p.1 ∧ p.1 < p.2 ∧ p.2 ≤ acc + sumSizes sizes := by
  induction sizes generalizing acc with
  | nil =>
    intro p hp
    simp [sliceIntervalsAux] at hp
  | cons sz rest ih =>
    have hsz : 1 ≤ sz := hPos sz (List.mem_cons_self ..)
    have hrest : ∀ s ∈ rest, 1 ≤ s := fun s hs => hPos s (List.mem_cons_of_mem sz hs)
    intro p hp
    simp [sliceIntervalsAux] at hp
    rcases hp with rfl | hmem
    · simp [sumSizes]
      omega
    · have hb := ih (acc + sz) hrest p hmem
      simp [sumSizes]
      omega

/-- Adjacent slice intervals emitted for ONNX `Split` are exactly contiguous (`hi_i = lo_{i+1}`). -/
theorem split_intervals_contiguous (acc : Nat) (sizes : List Nat) (i : Nat) (hi : i + 1 < sizes.length) :
    ((sliceIntervalsAux acc sizes)[i]'(by rw [split_intervals_length]; omega)).2 =
    ((sliceIntervalsAux acc sizes)[i + 1]'(by rw [split_intervals_length]; omega)).1 := by
  induction sizes generalizing acc i with
  | nil => simp at hi
  | cons sz rest ih =>
    cases i with
    | zero =>
      cases rest with
      | nil => simp at hi
      | cons sz2 rest2 =>
        simp [sliceIntervalsAux]
    | succ i' =>
      exact ih (acc + sz) i' (by simp at hi; omega)

/-- Any two distinct intervals `i < j` emitted by `sliceIntervalsAux` are non-overlapping (`hi_i ≤ lo_j`). -/
theorem split_intervals_disjoint (acc : Nat) (sizes : List Nat)
    (i j : Nat) (hi : i < sizes.length) (hj : j < sizes.length) (hij : i < j) :
    ((sliceIntervalsAux acc sizes)[i]'(by rw [split_intervals_length]; omega)).2 ≤
    ((sliceIntervalsAux acc sizes)[j]'(by rw [split_intervals_length]; omega)).1 := by
  induction sizes generalizing acc i j with
  | nil => simp at hi
  | cons sz rest ih =>
    cases i with
    | zero =>
      cases j with
      | zero => omega
      | succ j' =>
        have hmem : (sliceIntervalsAux (acc + sz) rest)[j']'(by rw [split_intervals_length]; simp at hj; omega) ∈
            sliceIntervalsAux (acc + sz) rest := List.getElem_mem _
        have hge : ∀ (a : Nat) (ls : List Nat) (p : Nat × Nat), p ∈ sliceIntervalsAux a ls → a ≤ p.1 := by
          intro a ls
          induction ls generalizing a with
          | nil => intro p hp; simp [sliceIntervalsAux] at hp
          | cons s rs ih2 =>
            intro p hp
            simp [sliceIntervalsAux] at hp
            rcases hp with rfl | hm
            · omega
            · have := ih2 (a + s) p hm
              omega
        exact hge (acc + sz) rest _ hmem
    | succ i' =>
      cases j with
      | zero => omega
      | succ j' =>
        exact ih (acc + sz) i' j' (by simp at hi; omega) (by simp at hj; omega) (by omega)

/-- W-axis split offset alignment: `start % 32 = 0` implies fp16 byte offset `2 * start` is 64-byte aligned. -/
theorem split_w_aligned_offset (start : Nat) (h : start % 32 = 0) : (2 * start) % 64 = 0 := by
  omega

/-- Exact element-count conservation between ONNX 4D `[1, H, M, K]` and Vane head-stacked `[1, H*M, 1, K]`. -/
theorem matmul_4d_stacked_numElems (h m k : Nat) :
    (matmul4DShape h m k).numElems = (matmulStackedShape h m k).numElems := by
  simp [matmul4DShape, matmulStackedShape, Shape4D.numElems, Nat.mul_assoc]

/-- Exact flat element-index equality between 4D `[1, H, M, K]` and head-stacked `[1, H*M, 1, K]`. -/
theorem matmul_4d_stacked_flatIndex_eq (m k hIdx mIdx kIdx : Nat) :
    packHeadRow m hIdx mIdx * k + kIdx = hIdx * (m * k) + mIdx * k + kIdx := by
  simp [packHeadRow, Nat.add_mul, Nat.mul_assoc]

/-- Left inverse of `(hIdx, mIdx) ↦ hIdx * M + mIdx` bijection for `mIdx < M`. -/
theorem matmul_headRow_unpack_pack (m hIdx mIdx : Nat) (hm : mIdx < m) :
    unpackHeadRow m (packHeadRow m hIdx mIdx) = (hIdx, mIdx) := by
  have hmPos : 0 < m := by omega
  simp [unpackHeadRow, packHeadRow]
  constructor
  · rw [Nat.mul_comm hIdx m, Nat.mul_add_div hmPos, Nat.div_eq_of_lt hm, Nat.add_zero]
  · exact Nat.mod_eq_of_lt hm

/-- Right inverse of `(hIdx, mIdx) ↦ hIdx * M + mIdx` bijection. -/
theorem matmul_headRow_pack_unpack (m idx : Nat) :
    packHeadRow m (unpackHeadRow m idx).1 (unpackHeadRow m idx).2 = idx := by
  simp [packHeadRow, unpackHeadRow]
  have h := Nat.div_add_mod idx m
  rw [Nat.mul_comm] at h
  exact h

/-- `packHeadRow` maps valid `(hIdx < H, mIdx < M)` into `[0, H * M)`. -/
theorem matmul_headRow_pack_lt (h m hIdx mIdx : Nat) (hh : hIdx < h) (hm : mIdx < m) :
    packHeadRow m hIdx mIdx < h * m := by
  unfold packHeadRow
  have h1 : (hIdx + 1) * m ≤ h * m := Nat.mul_le_mul_right m (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Exact ANE 64B-padded DRAM span equality between 4D `[1, H, M, K]` and head-stacked `[1, H*M, 1, K]`. -/
theorem matmul_4d_stacked_span_eq (h m k : Nat) :
    (matmul4DShape h m k).span = (matmulStackedShape h m k).span := by
  simp [matmul4DShape, matmulStackedShape, Shape4D.span, Shape4D.batchStride,
        Shape4D.planeStride, Shape4D.rowStride, Nat.mul_assoc]

/-- Every supported `PowKind` round-trips through `powKindToF64Bits` and `classifyPowBits`. -/
theorem pow_classify_roundTrip (k : PowKind) :
    classifyPowBits (powKindToF64Bits k) = some k := by
  cases k <;> decide

/-- `powKindToF64Bits` is injective across all 5 supported `PowKind` variants. -/
theorem pow_bits_injective (k1 k2 : PowKind)
    (h : powKindToF64Bits k1 = powKindToF64Bits k2) : k1 = k2 := by
  cases k1 <;> cases k2 <;> simp_all [powKindToF64Bits]

/-- `classifyPowBits` accepts a bit pattern iff it equals `powKindToF64Bits k`. -/
theorem pow_classify_complete (b : UInt64) (k : PowKind)
    (h : classifyPowBits b = some k) : b = powKindToF64Bits k := by
  unfold classifyPowBits at h
  split at h
  · simp at h; subst h; simp_all [powKindToF64Bits]
  · split at h
    · simp at h; subst h; simp_all [powKindToF64Bits]
    · split at h
      · simp at h; subst h; simp_all [powKindToF64Bits]
      · split at h
        · simp at h; subst h; simp_all [powKindToF64Bits]
        · split at h
          · simp at h; subst h; simp_all [powKindToF64Bits]
          · simp at h

/-- `classifyGeluApprox` accepts `""` and `"none"` as exact erf GELU. -/
theorem geluApprox_exact_iff (s : String) :
    classifyGeluApprox s = .ok .exact ↔ s = "" ∨ s = "none" := by
  unfold classifyGeluApprox
  split <;> simp_all

/-- `classifyGeluApprox` explicitly rejects `"tanh"` instead of silently approximating. -/
theorem geluApprox_tanh_rejected :
    (classifyGeluApprox "tanh").isOk = false := by
  decide

/-- `classifyGeluApprox` rejects every value outside `{"", "none"}`. -/
theorem geluApprox_invalid_rejected (s : String) (h1 : s ≠ "") (h2 : s ≠ "none") :
    (classifyGeluApprox s).isOk = false := by
  unfold classifyGeluApprox
  split <;> simp_all [Except.isOk, Except.toBool]

/-- `isValidLayerNormParamShape` rejects any parameter tensor whose rank exceeds the input rank. -/
theorem layerNorm_paramShape_rejects_excess_rank (dims : Array Nat) (normAxis normDim inRank : Nat)
    (h : dims.size > inRank) :
    isValidLayerNormParamShape dims normAxis normDim inRank = false := by
  unfold isValidLayerNormParamShape
  simp [h]

/-- `isValidLayerNormParamShape` accepts `[C]`, `[C, 1, 1]`, and `[1, C, 1, 1]` for 4D `axis=1`. -/
theorem layerNorm_paramShape_axis1_exact (c : Nat) :
    isValidLayerNormParamShape #[c] 1 c 4 = true ∧
    isValidLayerNormParamShape #[c, 1, 1] 1 c 4 = true ∧
    isValidLayerNormParamShape #[1, c, 1, 1] 1 c 4 = true := by
  simp [isValidLayerNormParamShape]

/-- `isValidLayerNormParamShape` accepts `[W]` and `[1, 1, 1, W]` for 4D `axis=3`. -/
theorem layerNorm_paramShape_axis3_exact (w : Nat) :
    isValidLayerNormParamShape #[w] 3 w 4 = true ∧
    isValidLayerNormParamShape #[1, 1, 1, w] 3 w 4 = true := by
  simp [isValidLayerNormParamShape]

-- ============================================================================
-- 22. Native NE Weight Quantization Proofs (M8A)
-- ============================================================================

-- 22.1 Quantized register WordLayout validity & encodeWord/decodeFieldChecked round-trips

theorem neCfgQuantLayout_valid : (validateWordLayout neCfgQuantLayout).isOk = true := by decide
theorem neOpQuantLayout_valid : (validateWordLayout neOpQuantLayout).isOk = true := by decide
theorem neScaleExtLayout_valid : (validateWordLayout neScaleExtLayout).isOk = true := by decide
theorem scalarBiasExtLayout_valid : (validateWordLayout scalarBiasExtLayout).isOk = true := by decide

theorem neCfgQuant_roundTrip (wKind wWidth sparse dynKern hasZP h17Flag : UInt32)
    (hk : wKind < 16) (hw : wWidth < 16) (hs : sparse < 2)
    (hd : dynKern < 2) (hz : hasZP < 2) (hh : h17Flag < 2) :
    ∃ w, encodeWord neCfgQuantLayout #[
            ("wKind", wKind), ("wWidth", wWidth), ("sparse", sparse),
            ("dynKern", dynKern), ("hasZP", hasZP), ("h17Flag", h17Flag)
          ] = .ok w ∧
         decodeFieldChecked neCfgQuantLayout w "wKind" = .ok wKind ∧
         decodeFieldChecked neCfgQuantLayout w "wWidth" = .ok wWidth ∧
         decodeFieldChecked neCfgQuantLayout w "sparse" = .ok sparse ∧
         decodeFieldChecked neCfgQuantLayout w "dynKern" = .ok dynKern ∧
         decodeFieldChecked neCfgQuantLayout w "hasZP" = .ok hasZP ∧
         decodeFieldChecked neCfgQuantLayout w "h17Flag" = .ok h17Flag := by
  have h1 : ¬ wKind.toNat ≥ 16 := u32_not_ge_of_lt hk
  have h2 : ¬ wWidth.toNat ≥ 16 := u32_not_ge_of_lt hw
  have h3 : ¬ sparse.toNat ≥ 2 := u32_not_ge_of_lt hs
  have h4 : ¬ dynKern.toNat ≥ 2 := u32_not_ge_of_lt hd
  have h5 : ¬ hasZP.toNat ≥ 2 := u32_not_ge_of_lt hz
  have h6 : ¬ h17Flag.toNat ≥ 2 := u32_not_ge_of_lt hh
  have henc : encodeWord neCfgQuantLayout #[
      ("wKind", wKind), ("wWidth", wWidth), ("sparse", sparse),
      ("dynKern", dynKern), ("hasZP", hasZP), ("h17Flag", h17Flag)
    ] = .ok (((((((0 : UInt32) ||| (wKind <<< 0)) ||| (wWidth <<< 4)) ||| (sparse <<< 8)) |||
              (dynKern <<< 16)) ||| (hasZP <<< 24)) ||| (h17Flag <<< 28)) := by
    change (if wKind.toNat ≥ 16 then _ else if wWidth.toNat ≥ 16 then _ else
            if sparse.toNat ≥ 2 then _ else if dynKern.toNat ≥ 2 then _ else
            if hasZP.toNat ≥ 2 then _ else if h17Flag.toNat ≥ 2 then _ else _) = _
    simp only [h1, h2, h3, h4, h5, h6, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "wKind" wKind henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "wWidth" wWidth henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "sparse" sparse henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "dynKern" dynKern henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "hasZP" hasZP henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "h17Flag" h17Flag henc rfl⟩

theorem neOpQuant_roundTrip (biasInKernel unscaled perLaneScale relu lut base20 : UInt32)
    (hb : biasInKernel < 2) (hu : unscaled < 16) (hp : perLaneScale < 2)
    (hr : relu < 2) (hl : lut < 2) (h20 : base20 < 2) :
    ∃ w, encodeWord neOpQuantLayout #[
            ("biasInKernel", biasInKernel), ("unscaled", unscaled), ("perLaneScale", perLaneScale),
            ("relu", relu), ("lut", lut), ("base20", base20)
          ] = .ok w ∧
         decodeFieldChecked neOpQuantLayout w "biasInKernel" = .ok biasInKernel ∧
         decodeFieldChecked neOpQuantLayout w "unscaled" = .ok unscaled ∧
         decodeFieldChecked neOpQuantLayout w "perLaneScale" = .ok perLaneScale ∧
         decodeFieldChecked neOpQuantLayout w "relu" = .ok relu ∧
         decodeFieldChecked neOpQuantLayout w "lut" = .ok lut ∧
         decodeFieldChecked neOpQuantLayout w "base20" = .ok base20 := by
  have h1 : ¬ biasInKernel.toNat ≥ 2 := u32_not_ge_of_lt hb
  have h2 : ¬ unscaled.toNat ≥ 16 := u32_not_ge_of_lt hu
  have h3 : ¬ perLaneScale.toNat ≥ 2 := u32_not_ge_of_lt hp
  have h4 : ¬ relu.toNat ≥ 2 := u32_not_ge_of_lt hr
  have h5 : ¬ lut.toNat ≥ 2 := u32_not_ge_of_lt hl
  have h6 : ¬ base20.toNat ≥ 2 := u32_not_ge_of_lt h20
  have henc : encodeWord neOpQuantLayout #[
      ("biasInKernel", biasInKernel), ("unscaled", unscaled), ("perLaneScale", perLaneScale),
      ("relu", relu), ("lut", lut), ("base20", base20)
    ] = .ok (((((((0 : UInt32) ||| (biasInKernel <<< 4)) ||| (unscaled <<< 10)) |||
              (perLaneScale <<< 14)) ||| (relu <<< 16)) ||| (lut <<< 17)) ||| (base20 <<< 20)) := by
    change (if biasInKernel.toNat ≥ 2 then _ else if unscaled.toNat ≥ 16 then _ else
            if perLaneScale.toNat ≥ 2 then _ else if relu.toNat ≥ 2 then _ else
            if lut.toNat ≥ 2 then _ else if base20.toNat ≥ 2 then _ else _) = _
    simp only [h1, h2, h3, h4, h5, h6, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "biasInKernel" biasInKernel henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "unscaled" unscaled henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "perLaneScale" perLaneScale henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "relu" relu henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "lut" lut henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "base20" base20 henc rfl⟩

theorem neScaleExt_roundTrip (val pexp : UInt32)
    (hv : val < 65536) (hp : pexp < 65536) :
    ∃ w, encodeWord neScaleExtLayout #[("val", val), ("pexp", pexp)] = .ok w ∧
         decodeFieldChecked neScaleExtLayout w "val" = .ok val ∧
         decodeFieldChecked neScaleExtLayout w "pexp" = .ok pexp := by
  have h1 : ¬ val.toNat ≥ 65536 := u32_not_ge_of_lt hv
  have h2 : ¬ pexp.toNat ≥ 65536 := u32_not_ge_of_lt hp
  have henc : encodeWord neScaleExtLayout #[("val", val), ("pexp", pexp)] =
      .ok (((0 : UInt32) ||| (val <<< 0)) ||| (pexp <<< 16)) := by
    change (if val.toNat ≥ 65536 then _ else if pexp.toNat ≥ 65536 then _ else _) = _
    simp only [h1, h2, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "val" val henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "pexp" pexp henc rfl⟩

theorem scalarBiasExt_roundTrip (val biasShift : UInt32)
    (hv : val < 65536) (hs : biasShift < 65536) :
    ∃ w, encodeWord scalarBiasExtLayout #[("val", val), ("biasShift", biasShift)] = .ok w ∧
         decodeFieldChecked scalarBiasExtLayout w "val" = .ok val ∧
         decodeFieldChecked scalarBiasExtLayout w "biasShift" = .ok biasShift := by
  have h1 : ¬ val.toNat ≥ 65536 := u32_not_ge_of_lt hv
  have h2 : ¬ biasShift.toNat ≥ 65536 := u32_not_ge_of_lt hs
  have henc : encodeWord scalarBiasExtLayout #[("val", val), ("biasShift", biasShift)] =
      .ok (((0 : UInt32) ||| (val <<< 0)) ||| (biasShift <<< 16)) := by
    change (if val.toNat ≥ 65536 then _ else if biasShift.toNat ≥ 65536 then _ else _) = _
    simp only [h1, h2, ↓reduceIte]
    rfl
  exact ⟨_, henc,
    encodeWord_decodeFieldChecked _ _ _ "val" val henc rfl,
    encodeWord_decodeFieldChecked _ _ _ "biasShift" biasShift henc rfl⟩

/-- `neCommon` bit 27 (`wide3x3`) helper clears bit 27 whenever `wide3x3` is false
    (H16, 1x1, linear, stride-2, depthwise, or `maxBlockSlots > 32768*rep`),
    `hasZP` is true, or `wKind` is unsigned (`uint8`/`uint4`). -/
theorem neCommonQuantBit27_cleared_when_illegal (wide3x3 hasZP : Bool) (wKind : UInt32)
    (h : wide3x3 = false ∨ hasZP = true ∨ wKind = wKindUint8 ∨ wKind = wKindUint4) :
    neCommonQuantBit27 wide3x3 hasZP wKind = 0 := by
  unfold neCommonQuantBit27 isUnsignedWeightKind
  rcases h with rfl | rfl | rfl | rfl <;> simp [wKindUint8, wKindUint4]

/-- `neCommon` bit 27 (`wide3x3`) is set to `1` iff `wide3x3 = true`, `hasZP = false`,
    and `wKind` is not unsigned. -/
theorem neCommonQuantBit27_eq_one_iff (wide3x3 hasZP : Bool) (wKind : UInt32) :
    neCommonQuantBit27 wide3x3 hasZP wKind = 1 ↔
      wide3x3 = true ∧ hasZP = false ∧ isUnsignedWeightKind wKind = false := by
  unfold neCommonQuantBit27
  cases wide3x3 <;> cases hasZP <;> cases isUnsignedWeightKind wKind <;> simp

-- 22.2 Canonical int4 / uint4 decode table theorems

theorem int4Table_size : int4Table.size = 16 := by decide
theorem uint4Table_size : uint4Table.size = 16 := by decide

/-- `int4Table` maps every 4-bit two's-complement nibble `i ∈ 0..15` to its exact signed int8 byte. -/
theorem int4Table_canonical (i : Fin 16) :
    let b := (int4Table[i.val]'(by rw [int4Table_size]; exact i.isLt)).toNat
    (if b ≥ 128 then (b : Int) - 256 else (b : Int)) =
    (if i.val < 8 then (i.val : Int) else (i.val : Int) - 16) := by
  revert i; decide

/-- `int4Table` is injective (bijective onto the 16 signed two's-complement int4 values). -/
theorem int4Table_injective (i j : Fin 16)
    (h : int4Table[i.val]'(by rw [int4Table_size]; exact i.isLt) =
         int4Table[j.val]'(by rw [int4Table_size]; exact j.isLt)) :
    i = j := by
  revert i j h; decide

/-- `uint4Table` maps every 4-bit unsigned nibble `i ∈ 0..15` identically to `i`. -/
theorem uint4Table_canonical (i : Fin 16) :
    (uint4Table[i.val]'(by rw [uint4Table_size]; exact i.isLt)).toNat = i.val := by
  revert i; decide

-- 22.3 Pure LSB-first bit-packing/unpacking lossless round-trip theorems

theorem packBitsLSB_size (w : Nat) (codes : Array UInt8) :
    (packBitsLSB w codes).size = packedByteCount w codes.size := by
  simp [packBitsLSB, ByteArray.size]

theorem unpackBitsLSB_size (w : Nat) (bytes : ByteArray) (count : Nat) :
    (unpackBitsLSB w bytes count).size = count := by
  simp [unpackBitsLSB]

private theorem testBit_shifted_single_bit (x k j r : Nat) :
    ((((x >>> k) &&& 1) <<< j)).testBit r = (decide (r = j) && x.testBit k) := by
  rw [Nat.testBit_shiftLeft, Nat.testBit_and, Nat.testBit_shiftRight]
  by_cases h : r = j
  · subst h
    simp
  · by_cases hle : j ≤ r
    · have hne : r - j ≠ 0 := by omega
      have h1 : (1 : Nat).testBit (r - j) = false := by
        have hlt : 1 < 2 ^ (r - j) := by
          have : 2 ^ 1 ≤ 2 ^ (r - j) := Nat.pow_le_pow_right (by omega) (by omega)
          omega
        exact Nat.testBit_lt_two_pow hlt
      simp [h, hle, h1]
    · simp [h, hle]

private theorem testBit_packByteLSB_bitAt (w : Nat) (codes : Array UInt8) (b j r : Nat) :
    (if 8 * b + j < codes.size * w then
       if h : (8 * b + j) / w < codes.size then
         ((codes[(8 * b + j) / w].toNat >>> ((8 * b + j) % w)) &&& 1) <<< j
       else 0
     else 0).testBit r =
    (decide (r = j) &&
      if 8 * b + j < codes.size * w then
        if h : (8 * b + j) / w < codes.size then
          codes[(8 * b + j) / w].toNat.testBit ((8 * b + j) % w)
        else false
      else false) := by
  split
  · split
    · exact testBit_shifted_single_bit _ _ _ _
    · simp
  · simp

private theorem testBit_packByteLSB (w : Nat) (codes : Array UInt8) (b r : Nat) (hr : r < 8) :
    (packByteLSB w codes b).toNat.testBit r =
    if 8 * b + r < codes.size * w then
      if h : (8 * b + r) / w < codes.size then
        codes[(8 * b + r) / w].toNat.testBit ((8 * b + r) % w)
      else false
    else false := by
  unfold packByteLSB
  have hmod : ∀ n : Nat, n.toUInt8.toNat = n % 2 ^ 8 := fun _ => rfl
  rw [hmod, Nat.testBit_mod_two_pow]
  simp only [hr, decide_true, Bool.true_and]
  simp only [Nat.testBit_or, testBit_packByteLSB_bitAt]
  match r with
  | 0 => simp
  | 1 => simp
  | 2 => simp
  | 3 => simp
  | 4 => simp
  | 5 => simp
  | 6 => simp
  | 7 => simp
  | _ + 8 => omega

private theorem testBit_unpackCodeLSB_go (w : Nat) (bytes : ByteArray) (i : Nat) :
    ∀ (k acc m : Nat),
      (unpackCodeLSB.go w bytes i k acc).testBit m =
      (acc.testBit m ||
        (decide (m < k) &&
          if h : (i * w + m) / 8 < bytes.data.size then
            bytes.data[(i * w + m) / 8].toNat.testBit ((i * w + m) % 8)
          else false)) := by
  intro k
  induction k with
  | zero =>
    intro acc m
    simp [unpackCodeLSB.go]
  | succ k' ih =>
    intro acc m
    unfold unpackCodeLSB.go
    rw [ih]
    rw [Nat.testBit_or]
    have hbit :
        (if h : (i * w + k') / 8 < bytes.data.size then
           ((bytes.data[(i * w + k') / 8].toNat >>> ((i * w + k') % 8)) &&& 1) <<< k'
         else 0).testBit m =
        (decide (m = k') &&
          if h : (i * w + k') / 8 < bytes.data.size then
            bytes.data[(i * w + k') / 8].toNat.testBit ((i * w + k') % 8)
          else false) := by
      split
      · exact testBit_shifted_single_bit _ _ _ _
      · simp
    rw [hbit]
    by_cases hlt : m < k'
    · have hne : m ≠ k' := by omega
      have hlt2 : m < k' + 1 := by omega
      simp [hlt, hne, hlt2]
    · by_cases heq : m = k'
      · subst heq
        simp
      · have hnot : ¬ m < k' + 1 := by omega
        simp [hlt, heq, hnot]

/-- Lossless bit-packing round-trip theorem: for ANY bit width `1 ≤ w ≤ 8` and ANY code array `codes`,
    `unpackBitsLSB w (packBitsLSB w codes) codes.size` recovers every code masked to its low `w` bits. -/
theorem unpackBitsLSB_packBitsLSB (w : Nat) (hw1 : 1 ≤ w) (hw8 : w ≤ 8) (codes : Array UInt8) :
    unpackBitsLSB w (packBitsLSB w codes) codes.size =
    codes.map (fun c => c &&& (((1 : Nat) <<< w) - 1).toUInt8) := by
  apply Array.ext
  · simp [unpackBitsLSB]
  · intro i hi1 hi2
    simp only [unpackBitsLSB, Array.getElem_ofFn, Array.getElem_map]
    apply UInt8.toNat_inj.mp
    apply Nat.eq_of_testBit_eq
    intro m
    have hi : i < codes.size := by simpa [unpackBitsLSB] using hi1
    unfold unpackCodeLSB
    have hmod8 : ∀ n : Nat, n.toUInt8.toNat = n % 2 ^ 8 := fun _ => rfl
    rw [hmod8, Nat.testBit_mod_two_pow, testBit_unpackCodeLSB_go]
    rw [UInt8.toNat_and, hmod8, Nat.testBit_and, Nat.testBit_mod_two_pow, testBit_one_shift_sub_one]
    simp only [Nat.zero_testBit, Bool.false_or]
    by_cases hm : m < w
    · have hm8 : m < 8 := by omega
      simp only [hm8, hm, decide_true, Bool.true_and, Bool.and_true]
      have hgLt : i * w + m < codes.size * w := by
        have h1 : (i + 1) * w ≤ codes.size * w := Nat.mul_le_mul_right w (by omega)
        rw [Nat.add_mul, Nat.one_mul] at h1
        omega
      have hbLt : (i * w + m) / 8 < (packBitsLSB w codes).data.size := by
        simp only [packBitsLSB, Array.size_ofFn, packedByteCount]
        omega
      simp only [hbLt, ↓reduceDIte]
      have hget : (packBitsLSB w codes).data[(i * w + m) / 8]'hbLt =
                  packByteLSB w codes ((i * w + m) / 8) := by
        simp [packBitsLSB]
      rw [hget, testBit_packByteLSB w codes ((i * w + m) / 8) ((i * w + m) % 8) (Nat.mod_lt _ (by omega))]
      have hdivmod : 8 * ((i * w + m) / 8) + (i * w + m) % 8 = i * w + m := Nat.div_add_mod _ 8
      rw [hdivmod]
      have hdivW : (i * w + m) / w = i := by
        rw [Nat.mul_comm i w, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hm, Nat.add_zero]
      have hmodW : (i * w + m) % w = m := by
        rw [Nat.mul_comm i w, Nat.mul_add_mod, Nat.mod_eq_of_lt hm]
      simp [hgLt, hdivW, hmodW, hi]
    · simp [hm]

/-- Explicit round-trip corollary across all hardware-supported bit widths `w ∈ {1, 2, 4, 6, 8}`. -/
theorem unpackBitsLSB_packBitsLSB_supported (w : Nat)
    (hw : w = 1 ∨ w = 2 ∨ w = 4 ∨ w = 6 ∨ w = 8) (codes : Array UInt8) :
    unpackBitsLSB w (packBitsLSB w codes) codes.size =
    codes.map (fun c => c &&& (((1 : Nat) <<< w) - 1).toUInt8) := by
  have h1 : 1 ≤ w := by rcases hw with rfl | rfl | rfl | rfl | rfl <;> omega
  have h8 : w ≤ 8 := by rcases hw with rfl | rfl | rfl | rfl | rfl <;> omega
  exact unpackBitsLSB_packBitsLSB w h1 h8 codes

/-- Exact identity round-trip when every input code is already in `[0, 2^w)`. -/
theorem unpackBitsLSB_packBitsLSB_exact (w : Nat) (hw1 : 1 ≤ w) (hw8 : w ≤ 8) (codes : Array UInt8)
    (hbound : ∀ i (hi : i < codes.size), codes[i].toNat < 2 ^ w) :
    unpackBitsLSB w (packBitsLSB w codes) codes.size = codes := by
  rw [unpackBitsLSB_packBitsLSB w hw1 hw8 codes]
  apply Array.ext
  · simp
  · intro i hi1 hi2
    simp only [Array.getElem_map]
    apply UInt8.toNat_inj.mp
    apply Nat.eq_of_testBit_eq
    intro m
    have hmod8 : ∀ n : Nat, n.toUInt8.toNat = n % 2 ^ 8 := fun _ => rfl
    rw [UInt8.toNat_and, hmod8, Nat.testBit_and, Nat.testBit_mod_two_pow, testBit_one_shift_sub_one]
    by_cases hm : m < w
    · have hm8 : m < 8 := by omega
      simp [hm, hm8]
    · have hzero : codes[i].toNat.testBit m = false :=
        testBit_eq_false_of_lt_two_pow (hbound i hi2) (by omega)
      simp [hm, hzero]

-- 22.4 Quantized chunk size, 64-byte alignment, monotonicity & packQuantChunk exactness

/-- Every quantized decode/palette table is 64-byte aligned. -/
theorem quantTableBytes_dvd_64 (wKind : UInt32) (wWidth : Nat) :
    quantTableBytes wKind wWidth % 64 = 0 := by
  unfold quantTableBytes
  split
  · rfl
  · split
    · exact alignUp_dvd _ 64 (by omega)
    · rfl

/-- `expectedQuantChunkSize` is always a multiple of 64 bytes (DMA line alignment). -/
theorem expectedQuantChunkSize_dvd_64
    (wKind : UInt32) (wWidth cin taps : Nat)
    (perLaneScale biasInKernel hasZeroPoint : Bool)
    (blocks : Array (Nat × Nat)) :
    expectedQuantChunkSize wKind wWidth cin taps perLaneScale biasInKernel hasZeroPoint blocks % 64 = 0 := by
  unfold expectedQuantChunkSize
  exact alignUp_dvd _ 64 (by omega)

private theorem foldl_add_ge_init {α : Type} (f : α → Nat) :
    ∀ (xs : List α) (init : Nat), init ≤ xs.foldl (fun acc x => acc + f x) init
  | [], init => Nat.le_refl init
  | x :: rest, init => by
    simp only [List.foldl_cons]
    exact Nat.le_trans (Nat.le_add_right init (f x)) (foldl_add_ge_init f rest (init + f x))

/-- `expectedQuantChunkSize` is always at least `quantTableBytes`. -/
theorem expectedQuantChunkSize_ge_table
    (wKind : UInt32) (wWidth cin taps : Nat)
    (perLaneScale biasInKernel hasZeroPoint : Bool)
    (blocks : Array (Nat × Nat)) :
    quantTableBytes wKind wWidth ≤
    expectedQuantChunkSize wKind wWidth cin taps perLaneScale biasInKernel hasZeroPoint blocks := by
  unfold expectedQuantChunkSize
  have h1 : quantTableBytes wKind wWidth ≤
      blocks.toList.foldl
        (fun acc p => acc + quantBlockBytes wWidth cin taps perLaneScale biasInKernel hasZeroPoint p.2)
        (quantTableBytes wKind wWidth) :=
    foldl_add_ge_init (fun p : Nat × Nat => quantBlockBytes wWidth cin taps perLaneScale biasInKernel hasZeroPoint p.2)
      blocks.toList (quantTableBytes wKind wWidth)
  have hfold : blocks.foldl
        (fun acc (_, n) => acc + quantBlockBytes wWidth cin taps perLaneScale biasInKernel hasZeroPoint n)
        (quantTableBytes wKind wWidth) =
      blocks.toList.foldl
        (fun acc p => acc + quantBlockBytes wWidth cin taps perLaneScale biasInKernel hasZeroPoint p.2)
        (quantTableBytes wKind wWidth) := by
    rw [← Array.foldl_toList]
  rw [hfold]
  exact Nat.le_trans h1 (alignUp_ge _ 64)

/-- `quantBlockBytes` is monotonic in lane count `n`. -/
theorem quantBlockBytes_mono_lanes
    (wWidth cin taps : Nat) (perLaneScale biasInKernel hasZeroPoint : Bool)
    {n1 n2 : Nat} (h : n1 ≤ n2) :
    quantBlockBytes wWidth cin taps perLaneScale biasInKernel hasZeroPoint n1 ≤
    quantBlockBytes wWidth cin taps perLaneScale biasInKernel hasZeroPoint n2 := by
  unfold quantBlockBytes packedByteCount
  have hmul : cin * taps * n1 * wWidth ≤ cin * taps * n2 * wWidth :=
    Nat.mul_le_mul_right wWidth (Nat.mul_le_mul_left (cin * taps) h)
  have hdiv : (cin * taps * n1 * wWidth + 7) / 8 ≤ (cin * taps * n2 * wWidth + 7) / 8 :=
    Nat.div_le_div_right (by omega)
  cases perLaneScale <;> cases biasInKernel <;> cases hasZeroPoint <;> simp <;> omega

/-- `encodeQuantTable` produces the exact byte length predicted by `quantTableBytes`. -/
theorem encodeQuantTable_size (format : QuantFormat) (palette : Array UInt16) :
    (encodeQuantTable format palette).size = quantTableBytes format.wKind format.wWidth := by
  cases format with
  | int8 =>
    rfl
  | uint8 =>
    rfl
  | int4 =>
    unfold encodeQuantTable
    rw [padToAlign_size]
    have hs : (ByteArray.mk int4Table).size = 16 := int4Table_size
    rw [hs]
    decide
  | uint4 =>
    unfold encodeQuantTable
    rw [padToAlign_size]
    have hs : (ByteArray.mk uint4Table).size = 16 := uint4Table_size
    rw [hs]
    decide
  | pal bits =>
    unfold encodeQuantTable
    rw [padToAlign_size]
    simp only [ByteArray.size, Array.size_ofFn]
    simp [quantTableBytes, QuantFormat.wKind, QuantFormat.wWidth, wKindPal, wKindInt4, wKindUint4]

/-- `encodeQuantBlock` produces the exact byte length predicted by `quantBlockBytes`. -/
theorem encodeQuantBlock_size
    (wWidth : Nat) (codes : ByteArray) (cout cin taps : Nat)
    (storedScales foldedBiases : Array UInt16) (zeroPoints : Option (Array UInt8))
    (perLaneScale biasInKernel hasZeroPoint : Bool) (o0 n : Nat) :
    (encodeQuantBlock wWidth codes cout cin taps storedScales foldedBiases zeroPoints
      perLaneScale biasInKernel hasZeroPoint o0 n).size =
    quantBlockBytes wWidth cin taps perLaneScale biasInKernel hasZeroPoint n := by
  unfold encodeQuantBlock quantBlockBytes
  simp only [ByteArray.size_append, packBitsLSB_size, Array.size_ofFn]
  have hempty : ByteArray.empty.data.size = 0 := rfl
  cases perLaneScale <;> cases biasInKernel <;> cases hasZeroPoint <;>
    simp only [Bool.false_eq_true, ↓reduceIte, ByteArray.size, hempty, Array.size_ofFn,
      Nat.zero_add, Nat.add_zero]

private theorem foldl_append_size {α : Type} (f : α → ByteArray) (g : α → Nat)
    (hfg : ∀ x, (f x).size = g x) :
    ∀ (xs : List α) (init : ByteArray),
      (xs.foldl (fun acc x => acc ++ f x) init).size =
      xs.foldl (fun acc x => acc + g x) init.size
  | [], init => rfl
  | x :: rest, init => by
    simp only [List.foldl_cons]
    rw [foldl_append_size f g hfg rest (init ++ f x), ByteArray.size_append, hfg x]

/-- Exactness theorem: `packQuantChunk` always emits a chunk of byte length exactly equal to
    `expectedQuantChunkSize` (for any format, geometry, flags, and block partition). -/
theorem packQuantChunk_size_eq_expected
    (format : QuantFormat) (palette : Array UInt16) (codes : ByteArray) (cout cin taps : Nat)
    (storedScales foldedBiases : Array UInt16) (zeroPoints : Option (Array UInt8))
    (perLaneScale biasInKernel hasZeroPoint : Bool) (blocks : Array (Nat × Nat)) :
    (packQuantChunk format palette codes cout cin taps storedScales foldedBiases zeroPoints
      perLaneScale biasInKernel hasZeroPoint blocks).size =
    expectedQuantChunkSize format.wKind format.wWidth cin taps perLaneScale biasInKernel hasZeroPoint blocks := by
  unfold packQuantChunk expectedQuantChunkSize
  rw [padToAlign_size]
  congr 1
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  have hstep : ∀ (p : Nat × Nat),
      (encodeQuantBlock format.wWidth codes cout cin taps storedScales foldedBiases zeroPoints
        perLaneScale biasInKernel hasZeroPoint p.1 p.2).size =
      quantBlockBytes format.wWidth cin taps perLaneScale biasInKernel hasZeroPoint p.2 := fun p =>
    encodeQuantBlock_size format.wWidth codes cout cin taps storedScales foldedBiases zeroPoints
      perLaneScale biasInKernel hasZeroPoint p.1 p.2
  rw [foldl_append_size
    (fun p : Nat × Nat => encodeQuantBlock format.wWidth codes cout cin taps storedScales foldedBiases zeroPoints
      perLaneScale biasInKernel hasZeroPoint p.1 p.2)
    (fun p : Nat × Nat => quantBlockBytes format.wWidth cin taps perLaneScale biasInKernel hasZeroPoint p.2)
    hstep blocks.toList (encodeQuantTable format palette)]
  rw [encodeQuantTable_size format palette]

-- ============================================================================
-- 23. Parameterized SSA TensorIR, Rewriter & Polymorphic evalCore Proofs (C1)
-- ============================================================================

private theorem arr_getD_eq {α : Type} (a : Array α) (i : Nat) (d : α) (h : i < a.size) :
    a.getD i d = a[i] := by
  simp [Array.getD, h]

private theorem list_getD_eq {α : Type} (l : List α) (i : Nat) (d : α) (h : i < l.length) :
    l.getD i d = l[i] := by
  simp [List.getD, h]

private theorem list_nodup_getElem_inj {α : Type} {l : List α} (hnd : l.Nodup)
    {i j : Nat} (hi : i < l.length) (hj : j < l.length) (heq : l[i] = l[j]) : i = j := by
  induction l generalizing i j with
  | nil => simp at hi
  | cons x xs ih =>
    rw [List.nodup_cons] at hnd
    cases i with
    | zero =>
      cases j with
      | zero => rfl
      | succ j' =>
        simp only [List.getElem_cons_zero, List.getElem_cons_succ] at heq
        have hj' : j' < xs.length := by simpa using hj
        exact absurd (heq ▸ List.getElem_mem hj') hnd.1
    | succ i' =>
      cases j with
      | zero =>
        simp only [List.getElem_cons_zero, List.getElem_cons_succ] at heq
        have hi' : i' < xs.length := by simpa using hi
        exact absurd (heq.symm ▸ List.getElem_mem hi') hnd.1
      | succ j' =>
        simp only [List.getElem_cons_succ] at heq
        have hi' : i' < xs.length := by simpa using hi
        have hj' : j' < xs.length := by simpa using hj
        exact congrArg Nat.succ (ih hnd.2 hi' hj' heq)

private theorem list_nodup_of_getElem_inj {α : Type} {l : List α}
    (hinj : ∀ i j (hi : i < l.length) (hj : j < l.length), l[i] = l[j] → i = j) : l.Nodup := by
  induction l with
  | nil => exact List.nodup_nil
  | cons x xs ih =>
    rw [List.nodup_cons]
    refine ⟨?_, ?_⟩
    · intro hmem
      obtain ⟨j, hj, hjeq⟩ := List.mem_iff_getElem.mp hmem
      have h0 := hinj 0 (j + 1) (by simp) (by simp; omega) (by simpa using hjeq.symm)
      omega
    · apply ih
      intro i j hi hj heq
      have h := hinj (i + 1) (j + 1) (by simp; omega) (by simp; omega) (by simpa using heq)
      omega

private theorem list_nodup_map_on {α β : Type} {f : α → β} {l : List α}
    (hinj : ∀ x ∈ l, ∀ y ∈ l, f x = f y → x = y) (hnd : l.Nodup) : (l.map f).Nodup := by
  induction l with
  | nil => exact List.nodup_nil
  | cons x xs ih =>
    rw [List.nodup_cons] at hnd
    rw [List.map_cons, List.nodup_cons]
    refine ⟨?_, ih (fun a ha b hb => hinj a (List.mem_cons_of_mem _ ha) b (List.mem_cons_of_mem _ hb)) hnd.2⟩
    intro hmem
    obtain ⟨y, hy, hfy⟩ := List.mem_map.mp hmem
    have hxy := hinj y (List.mem_cons_of_mem _ hy) x (List.mem_cons_self ..) hfy
    exact hnd.1 (hxy ▸ hy)

-- 23.1 Validator soundness (`validateGraph_ok_implies_wellFormed`) & SSA acyclicity

private theorem checkList_ok {α : Type} {p : α → Bool} {err : α → String} {xs : List α}
    (h : checkList p err xs = .ok ()) : ∀ x ∈ xs, p x = true := by
  induction xs with
  | nil => intro x hx; cases hx
  | cons a rest ih =>
    unfold checkList at h
    split at h
    · rename_i hpa
      intro x hx
      rcases List.mem_cons.mp hx with rfl | hmem
      · exact hpa
      · exact ih h x hmem
    · cases h

private theorem checkIdxRange_ok {lo hi : Nat} {f : Nat → Except String Unit}
    (h : checkIdxRange lo hi f = .ok ()) :
    ∀ i, lo ≤ i → i < hi → f i = .ok () := by
  intro i hlo hhi
  have haux : ∀ d lo, d = hi - lo → checkIdxRange lo hi f = .ok () → lo ≤ i → i < hi → f i = .ok () := by
    intro d
    induction d with
    | zero => intro lo hd _ _ _; omega
    | succ r ih =>
      intro lo hd hchk hlo' hhi'
      have hlt : lo < hi := by omega
      unfold checkIdxRange at hchk
      simp only [hlt, ↓reduceDIte] at hchk
      cases hflo : f lo with
      | error e => simp [hflo] at hchk
      | ok u =>
        cases u
        simp only [hflo] at hchk
        by_cases heq : i = lo
        · subst heq; exact hflo
        · exact ih (lo + 1) (by omega) hchk (by omega) hhi'
  exact haux (hi - lo) lo rfl h hlo hhi

private theorem validateUniqueIds_nodup {ids : List ValueId}
    (h : validateUniqueIds ids = .ok ()) : ids.Nodup := by
  induction ids with
  | nil => exact List.nodup_nil
  | cons x xs ih =>
    unfold validateUniqueIds at h
    split at h
    · cases h
    · rename_i hne
      rw [List.nodup_cons]
      refine ⟨?_, ih h⟩
      intro hmem
      have helem : xs.elem x = true := List.elem_eq_true_of_mem hmem
      exact hne helem

private theorem hasPositiveDims_iff (ty : TensorType) :
    ty.hasPositiveDims = true ↔ ∀ d ∈ ty.shape.toList, d > 0 := by
  unfold TensorType.hasPositiveDims
  rw [Array.all_eq_true_iff_forall_mem]
  constructor
  · intro h d hd
    have hdArr : d ∈ ty.shape := Array.mem_def.mpr hd
    simpa using h d hdArr
  · intro h d hd
    have hdList : d ∈ ty.shape.toList := Array.mem_def.mp hd
    simpa using h d hdList

/-- Executable `validateGraph` soundness: any graph accepted by `validateGraph` satisfies
    every Prop-level invariant of `WellFormed`. -/
theorem validateGraph_ok_implies_wellFormed {Op : Type} (g : SSAGraph Op)
    (hok : validateGraph g = .ok ()) : WellFormed g := by
  unfold validateGraph at hok
  obtain ⟨⟨⟩, hIn, hok⟩ := except_bind_ok hok
  obtain ⟨⟨⟩, hUniq, hok⟩ := except_bind_ok hok
  obtain ⟨⟨⟩, hNodes, hOuts⟩ := except_bind_ok hok
  constructor
  · -- inputsDense
    intro i hi
    have hiOk := checkIdxRange_ok hIn i (Nat.zero_le i) hi
    simp only [hi, ↓reduceDIte] at hiOk
    split at hiOk
    · cases hiOk
    · rename_i hne
      simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hne
      exact hne
  · -- inputsPositive
    intro i hi d hd
    have hiOk := checkIdxRange_ok hIn i (Nat.zero_le i) hi
    simp only [hi, ↓reduceDIte] at hiOk
    split at hiOk
    · cases hiOk
    · split at hiOk
      · cases hiOk
      · rename_i hpos
        simp only [Bool.not_eq_true'] at hpos
        have htr : g.inputs[i].2.2.hasPositiveDims = true := by
          cases hb : g.inputs[i].2.2.hasPositiveDims <;> simp_all
        exact (hasPositiveDims_iff _).mp htr d hd
  · -- uniqueIds
    exact validateUniqueIds_nodup hUniq
  · -- defBeforeUse
    intro i hi v hv
    unfold validateGraphNodes at hNodes
    have hiOk := checkIdxRange_ok hNodes i (Nat.zero_le i) hi
    dsimp only at hiOk
    simp only [hi, ↓reduceDIte] at hiOk
    split at hiOk
    · cases hiOk
    · rename_i hchk
      have hel := checkList_ok hchk v hv
      exact List.mem_of_elem_eq_true hel
  · -- nodeOutputsPositive
    intro i hi o ho d hd
    unfold validateGraphNodes at hNodes
    have hiOk := checkIdxRange_ok hNodes i (Nat.zero_le i) hi
    dsimp only at hiOk
    simp only [hi, ↓reduceDIte] at hiOk
    split at hiOk
    · cases hiOk
    · have hpos := checkList_ok hiOk o ho
      exact (hasPositiveDims_iff _).mp hpos d hd
  · -- outputsValid
    intro v hv
    have hel := checkList_ok hOuts v hv
    exact List.mem_of_elem_eq_true hel

private theorem nodup_append_disjoint {α : Type} {l1 l2 : List α}
    (h : (l1 ++ l2).Nodup) : ∀ x, x ∈ l1 → x ∉ l2 := by
  intro x hx1 hx2
  rw [List.nodup_append] at h
  exact h.2.2 x hx1 x hx2 rfl

private theorem allDefinedIds_split {Op : Type} (g : SSAGraph Op) (k : Nat) :
    g.allDefinedIds = g.allDefinedIdsBefore k ++ (g.nodes.toList.drop k).flatMap IRNode.outputIds := by
  unfold SSAGraph.allDefinedIds SSAGraph.allDefinedIdsBefore
  rw [List.append_assoc, ← List.flatMap_append, List.take_append_drop]

/-- Strict SSA acyclicity: in any `WellFormed` graph, if node `i` consumes a value `v`
    produced by node `j`, then strictly `j < i`. -/
theorem wellFormed_acyclic {Op : Type} {g : SSAGraph Op} (hwf : WellFormed g)
    {i j : Nat} (hi : i < g.nodes.size) (hj : j < g.nodes.size)
    {v : ValueId} (huse : v ∈ g.nodes[i].inputs.toList) (hdef : v ∈ g.nodes[j].outputIds) :
    j < i := by
  have hvBefore : v ∈ g.allDefinedIdsBefore i := hwf.defBeforeUse i hi v huse
  by_cases hlt : j < i
  · exact hlt
  · have hge : i ≤ j := by omega
    have hmemDrop : g.nodes[j] ∈ g.nodes.toList.drop i := by
      have hget : (g.nodes.toList.drop i)[j - i]'(by simp; omega) = g.nodes[j] := by
        simp [List.getElem_drop, Nat.add_sub_cancel' hge]
      rw [← hget]
      exact List.getElem_mem _
    have hvAfter : v ∈ (g.nodes.toList.drop i).flatMap IRNode.outputIds :=
      List.mem_flatMap.mpr ⟨g.nodes[j], hmemDrop, hdef⟩
    have hnodup := hwf.uniqueIds
    rw [allDefinedIds_split g i] at hnodup
    exact absurd hvAfter (nodup_append_disjoint hnodup v hvBefore)

-- 23.2 Permutation algebra: identity, composition, associativity, bijectivity & inverses

/-- `isIdentityPerm p = true` iff `p = idPerm p.size`. -/
theorem isIdentityPerm_iff (p : Array Nat) :
    isIdentityPerm p = true ↔ p = idPerm p.size := by
  unfold isIdentityPerm
  exact beq_iff_eq

/-- Element-wise characterization of `isIdentityPerm`. -/
theorem isIdentityPerm_iff_get (p : Array Nat) :
    isIdentityPerm p = true ↔ ∀ i (hi : i < p.size), p[i] = i := by
  rw [isIdentityPerm_iff]
  constructor
  · intro h i hi
    have hget := congrArg (fun a : Array Nat => a[i]?) h
    rw [Array.getElem?_eq_getElem hi, idPerm, Array.getElem?_eq_getElem (by simp [hi])] at hget
    simpa using hget
  · intro h
    apply Array.ext
    · simp [idPerm]
    · intro i hi1 hi2
      simp only [idPerm, Array.getElem_ofFn]
      exact h i hi1

private theorem listNodupB_iff {α : Type} [BEq α] [LawfulBEq α] (l : List α) :
    listNodupB l = true ↔ l.Nodup := by
  induction l with
  | nil => simp [listNodupB]
  | cons x xs ih =>
    simp only [listNodupB, Bool.and_eq_true, Bool.not_eq_true', List.nodup_cons, ih]
    constructor
    · intro ⟨hne, hnd⟩
      refine ⟨?_, hnd⟩
      intro hmem
      have : xs.elem x = true := List.elem_eq_true_of_mem hmem
      rw [this] at hne; cases hne
    · intro ⟨hnot, hnd⟩
      refine ⟨?_, hnd⟩
      cases hc : xs.elem x with
      | false => rfl
      | true => exact absurd (List.mem_of_elem_eq_true hc) hnot

/-- Executable `isValidPerm` matches Prop-level `ValidPerm`. -/
theorem isValidPerm_iff (p : Array Nat) :
    isValidPerm p = true ↔ ValidPerm p := by
  unfold isValidPerm ValidPerm
  simp only [Bool.and_eq_true, listNodupB_iff]
  constructor
  · intro ⟨hall, hnd⟩
    refine ⟨?_, hnd⟩
    intro i hi
    have hmem : p[i] ∈ p := Array.getElem_mem hi
    have hlt := Array.all_eq_true_iff_forall_mem.mp hall p[i] hmem
    simpa using hlt
  · intro ⟨hlt, hnd⟩
    refine ⟨?_, hnd⟩
    rw [Array.all_eq_true_iff_forall_mem]
    intro x hx
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp hx
    simpa using hlt i hi

/-- `idPerm n` is always a valid permutation. -/
theorem validPerm_idPerm (n : Nat) : ValidPerm (idPerm n) := by
  constructor
  · intro i hi
    simp only [idPerm, Array.size_ofFn] at hi ⊢
    simp [hi]
  · have hlist : (idPerm n).toList = List.range n := by
      apply List.ext_getElem
      · simp [idPerm]
      · intro i h1 h2
        simp [idPerm]
    rw [hlist]
    exact List.nodup_range

/-- Left identity law for `composePerm`. -/
theorem composePerm_id_left (p : Array Nat) (hp : ValidPerm p) :
    composePerm (idPerm p.size) p = p := by
  apply Array.ext
  · simp [composePerm]
  · intro i hi1 hi2
    simp only [composePerm, Array.getElem_map]
    have hlt : p[i] < (idPerm p.size).size := by
      simp only [idPerm, Array.size_ofFn]
      exact hp.1 i hi2
    rw [arr_getD_eq _ _ _ hlt]
    simp [idPerm]

/-- Right identity law for `composePerm`. -/
theorem composePerm_id_right (p : Array Nat) :
    composePerm p (idPerm p.size) = p := by
  apply Array.ext
  · simp [composePerm, idPerm]
  · intro i hi1 hi2
    simp only [composePerm, idPerm, Array.getElem_map, Array.getElem_ofFn]
    rw [arr_getD_eq _ _ _ hi2]

/-- Associativity of `composePerm` on equal-length valid permutations. -/
theorem composePerm_assoc (p1 p2 p3 : Array Nat)
    (_hp2 : ValidPerm p2) (hp3 : ValidPerm p3)
    (_h12 : p1.size = p2.size) (h23 : p2.size = p3.size) :
    composePerm (composePerm p1 p2) p3 = composePerm p1 (composePerm p2 p3) := by
  apply Array.ext
  · simp [composePerm]
  · intro i hi1 hi2
    have hi3 : i < p3.size := by simpa [composePerm] using hi1
    simp only [composePerm, Array.getElem_map]
    have h3lt : p3[i] < p2.size := by rw [h23]; exact hp3.1 i hi3
    have h3lt' : p3[i] < (p2.map fun idx => p1.getD idx 0).size := by simp [h3lt]
    rw [arr_getD_eq _ _ _ h3lt', Array.getElem_map]
    rw [arr_getD_eq _ _ _ h3lt]

/-- Bijectivity preservation: composing two valid equal-length permutations yields a valid permutation. -/
theorem validPerm_composePerm (p1 p2 : Array Nat)
    (hp1 : ValidPerm p1) (hp2 : ValidPerm p2) (hsz : p1.size = p2.size) :
    ValidPerm (composePerm p1 p2) := by
  have hcompSz : (composePerm p1 p2).size = p2.size := by simp [composePerm]
  constructor
  · intro i hi
    have hi2 : i < p2.size := by omega
    simp only [composePerm, Array.getElem_map, Array.size_map]
    have hp2i : p2[i] < p1.size := by rw [hsz]; exact hp2.1 i hi2
    rw [arr_getD_eq _ _ _ hp2i]
    rw [← hsz]
    exact hp1.1 _ hp2i
  · apply list_nodup_of_getElem_inj
    intro i j hi hj heq
    have hi2 : i < p2.size := by simpa [composePerm] using hi
    have hj2 : j < p2.size := by simpa [composePerm] using hj
    have hp2i : p2[i] < p1.size := by rw [hsz]; exact hp2.1 i hi2
    have hp2j : p2[j] < p1.size := by rw [hsz]; exact hp2.1 j hj2
    have heq' : p1.toList[p2[i]]'(by simpa using hp2i) = p1.toList[p2[j]]'(by simpa using hp2j) := by
      have heq1 : (composePerm p1 p2)[i]'(by omega) = (composePerm p1 p2)[j]'(by omega) := heq
      simp only [composePerm, Array.getElem_map] at heq1
      rw [arr_getD_eq _ _ _ hp2i, arr_getD_eq _ _ _ hp2j] at heq1
      exact heq1
    have hp2eq : p2.toList[i]'(by simpa using hi2) = p2.toList[j]'(by simpa using hj2) :=
      list_nodup_getElem_inj hp1.2 (by simpa using hp2i) (by simpa using hp2j) heq'
    exact list_nodup_getElem_inj hp2.2 (by simpa using hi2) (by simpa using hj2) hp2eq

-- Pigeonhole surjectivity for valid permutations (needed for `invPerm` and `evalTranspose`)

private theorem list_filter_length_le {α : Type} (p : α → Bool) (l : List α) :
    (l.filter p).length ≤ l.length :=
  List.length_filter_le p l

private theorem nodup_subset_range_perm (l : List Nat) (hnd : l.Nodup)
    (hlt : ∀ x ∈ l, x < l.length) : ∀ a < l.length, a ∈ l := by
  -- Prove by induction on `n = l.length` using erase/subset cardinality
  have hsub : ∀ (n : Nat) (xs : List Nat), xs.Nodup → xs.length = n → (∀ x ∈ xs, x < n) →
      ∀ a < n, a ∈ xs := by
    intro n
    induction n with
    | zero => intro xs _ _ _ a ha; omega
    | succ m ih =>
      intro xs hnd' hlen hlt' a ha
      by_cases haMem : a ∈ xs
      · exact haMem
      · -- If `a ∉ xs`, every element of `xs` is in `{0,..,m} \ {a}`, which has size `m` — contradiction!
        -- Map each element `x` of `xs` to `f x = if x < a then x else x - 1 < m`, injective on `xs`!
        let f : Nat → Nat := fun x => if x < a then x else x - 1
        let ys := xs.map f
        have hysLen : ys.length = m + 1 := by simp [ys, hlen]
        have hysLt : ∀ y ∈ ys, y < m := by
          intro y hy
          obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hy
          have hxLt : x < m + 1 := hlt' x hx
          have hxNe : x ≠ a := fun heq => haMem (heq ▸ hx)
          dsimp only [f]
          split <;> omega
        have hysNd : ys.Nodup := by
          apply list_nodup_map_on _ hnd'
          intro x1 hx1 x2 hx2 hfeq
          have hne1 : x1 ≠ a := fun heq => haMem (heq ▸ hx1)
          have hne2 : x2 ≠ a := fun heq => haMem (heq ▸ hx2)
          dsimp only [f] at hfeq
          split at hfeq <;> split at hfeq <;> omega
        -- `ys` has length `m + 1`, `Nodup`, all elements `< m`: split off head `y0`, tail has length `m`!
        cases hysEq : ys with
        | nil => rw [hysEq] at hysLen; simp at hysLen
        | cons y0 yrest =>
          rw [hysEq] at hysLen hysNd hysLt
          rw [List.nodup_cons] at hysNd
          have hy0Lt : y0 < m := hysLt y0 (List.mem_cons_self ..)
          have hyrLen : yrest.length = m := by simp at hysLen; exact hysLen
          have hyrLt : ∀ y ∈ yrest, y < m := fun y hy => hysLt y (List.mem_cons_of_mem _ hy)
          have hy0InRest : y0 ∈ yrest := ih yrest hysNd.2 hyrLen hyrLt y0 hy0Lt
          exact absurd hy0InRest hysNd.1
  exact hsub l.length l hnd rfl hlt

private theorem findIdxNat_lt_and_get (l : List Nat) (target : Nat) (hmem : target ∈ l) :
    findIdxNat l target < l.length ∧
    l[findIdxNat l target]'(by
      induction l with
      | nil => cases hmem
      | cons x xs ih =>
        unfold findIdxNat
        split
        · simp
        · rename_i hne
          rcases List.mem_cons.mp hmem with rfl | hm
          · simp at hne
          · have := ih hm; simp; omega) = target := by
  induction l with
  | nil => cases hmem
  | cons x xs ih =>
    unfold findIdxNat
    split
    · rename_i heq
      have hx : x = target := beq_iff_eq.mp heq
      exact ⟨by simp, by simp [hx]⟩
    · rename_i hne
      rcases List.mem_cons.mp hmem with rfl | hm
      · simp at hne
      · obtain ⟨hlt, hget⟩ := ih hm
        refine ⟨by simp; omega, ?_⟩
        have hidx : 1 + findIdxNat xs target = findIdxNat xs target + 1 := Nat.add_comm 1 _
        simp only [hidx, List.getElem_cons_succ]
        exact hget

private theorem findIdxNat_getElem (l : List Nat) (hnd : l.Nodup) (k : Nat) (hk : k < l.length) :
    findIdxNat l l[k] = k := by
  induction l generalizing k with
  | nil => simp at hk
  | cons x xs ih =>
    rw [List.nodup_cons] at hnd
    cases k with
    | zero => simp [findIdxNat]
    | succ k' =>
      have hk' : k' < xs.length := by simpa using hk
      have hmem : xs[k'] ∈ xs := List.getElem_mem hk'
      have hne : (x == xs[k']) = false := by
        cases hc : (x == xs[k']) with
        | false => rfl
        | true =>
          have heq : x = xs[k'] := beq_iff_eq.mp hc
          exact absurd (heq ▸ hmem) hnd.1
      simp only [List.getElem_cons_succ, findIdxNat, hne, Bool.false_eq_true, ↓reduceIte]
      rw [ih hnd.2 k' hk']
      omega

/-- For any valid permutation `p` and `a < p.size`, `invPerm p` satisfies `(invPerm p)[a] < p.size`
    and `p[(invPerm p)[a]] = a`. -/
theorem invPerm_right (p : Array Nat) (hp : ValidPerm p) (a : Nat) (ha : a < p.size) :
    (invPerm p)[a]'(by simp [invPerm, ha]) < p.size ∧
    p[(invPerm p)[a]'(by simp [invPerm, ha])]'(by
      simp only [invPerm, Array.getElem_ofFn]
      have hmem : a ∈ p.toList := nodup_subset_range_perm p.toList hp.2
        (fun x hx => by
          obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp hx
          have hi' : i < p.size := by simpa using hi
          simpa using hp.1 i hi') a (by simpa using ha)
      have := (findIdxNat_lt_and_get p.toList a hmem).1
      simpa using this) = a := by
  have hmem : a ∈ p.toList := nodup_subset_range_perm p.toList hp.2
    (fun x hx => by
      obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp hx
      have hi' : i < p.size := by simpa using hi
      simpa using hp.1 i hi') a (by simpa using ha)
  obtain ⟨hlt, hget⟩ := findIdxNat_lt_and_get p.toList a hmem
  refine ⟨by simpa [invPerm] using hlt, ?_⟩
  simpa [invPerm] using hget

/-- For any valid permutation `p` and `k < p.size`, `(invPerm p)[p[k]] = k`. -/
theorem invPerm_left (p : Array Nat) (hp : ValidPerm p) (k : Nat) (hk : k < p.size) :
    (invPerm p)[p[k]]'(by simp [invPerm]; exact hp.1 k hk) = k := by
  simp only [invPerm, Array.getElem_ofFn]
  have hk' : k < p.toList.length := by simpa using hk
  have h := findIdxNat_getElem p.toList hp.2 k hk'
  simpa using h

/-- `invPerm (idPerm n) = idPerm n`. -/
theorem invPerm_idPerm (n : Nat) : invPerm (idPerm n) = idPerm n := by
  apply Array.ext
  · simp [invPerm, idPerm]
  · intro i hi1 hi2
    have hi : i < n := by simpa [idPerm] using hi2
    have h := invPerm_left (idPerm n) (validPerm_idPerm n) i (by simpa [idPerm] using hi)
    simpa [idPerm] using h

/-- For valid equal-length permutations `p1, p2`, `invPerm (composePerm p1 p2)` matches
    `composePerm (invPerm p2) (invPerm p1)`. -/
theorem invPerm_composePerm (p1 p2 : Array Nat) (hp1 : ValidPerm p1) (hp2 : ValidPerm p2)
    (hsz : p1.size = p2.size) :
    invPerm (composePerm p1 p2) = composePerm (invPerm p2) (invPerm p1) := by
  have hcomp := validPerm_composePerm p1 p2 hp1 hp2 hsz
  apply Array.ext
  · simp [invPerm, composePerm, hsz]
  · intro a ha1 ha2
    have ha : a < p1.size := by simpa [composePerm, invPerm] using ha2
    have haInv1 : a < (invPerm p1).size := by simp [invPerm, ha]
    let b := (invPerm p1)[a]'haInv1
    have hbLt1 : b < p1.size := (invPerm_right p1 hp1 a ha).1
    have hbLt2 : b < p2.size := by omega
    have hget1 : p1[b]'hbLt1 = a := (invPerm_right p1 hp1 a ha).2
    have hbInv2 : b < (invPerm p2).size := by simp [invPerm, hbLt2]
    let k := (invPerm p2)[b]'hbInv2
    have hkLt2 : k < p2.size := (invPerm_right p2 hp2 b hbLt2).1
    have hget2 : p2[k]'hkLt2 = b := (invPerm_right p2 hp2 b hbLt2).2
    have hkComp : (composePerm p1 p2)[k]'(by simp [composePerm, hkLt2]) = a := by
      simp only [composePerm, Array.getElem_map]
      have hp2k : p2[k]'hkLt2 < p1.size := by rw [hsz]; exact hp2.1 k hkLt2
      rw [arr_getD_eq _ _ _ hp2k]
      simpa [hget2] using hget1
    have hleft := invPerm_left (composePerm p1 p2) hcomp k (by simp [composePerm, hkLt2])
    have hEq : (invPerm (composePerm p1 p2))[a]'ha1 = k := by
      simpa [hkComp] using hleft
    rw [hEq]
    simp only [composePerm, Array.getElem_map]
    rw [arr_getD_eq _ _ _ hbInv2]

-- 23.3 Row-major index bijection & exact polymorphic tensor view equivalence

/-- `shapeProd s > 0` whenever every dimension in `s` is strictly positive. -/
theorem shapeProd_pos (s : List Nat) (h : ∀ d ∈ s, d > 0) : shapeProd s > 0 := by
  induction s with
  | nil => simp [shapeProd]
  | cons d ds ih =>
    simp only [shapeProd]
    have hd : d > 0 := h d (List.mem_cons_self ..)
    have hds : shapeProd ds > 0 := ih (fun x hx => h x (List.mem_cons_of_mem _ hx))
    exact Nat.mul_pos hd hds

/-- `unflatten s idx` always has length `s.length`. -/
theorem unflatten_length (s : List Nat) (idx : Nat) : (unflatten s idx).length = s.length := by
  induction s generalizing idx with
  | nil => rfl
  | cons _ ds ih => simp [unflatten, ih]

/-- Every coordinate produced by `unflatten s idx` (for `idx < shapeProd s`) is in-bounds `< s[i]`. -/
theorem unflatten_lt (s : List Nat) (hpos : ∀ d ∈ s, d > 0) (idx : Nat) (hidx : idx < shapeProd s) :
    ∀ i (hi : i < s.length), (unflatten s idx)[i]'(by rw [unflatten_length]; exact hi) < s[i] := by
  induction s generalizing idx with
  | nil => intro i hi; simp at hi
  | cons d ds ih =>
    intro i hi
    have hdsPos : shapeProd ds > 0 := shapeProd_pos ds (fun x hx => hpos x (List.mem_cons_of_mem _ hx))
    cases i with
    | zero =>
      simp only [unflatten, List.getElem_cons_zero]
      simp only [shapeProd] at hidx
      exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hidx)
    | succ i' =>
      simp only [unflatten, List.getElem_cons_succ]
      exact ih (fun x hx => hpos x (List.mem_cons_of_mem _ hx)) (idx % shapeProd ds)
        (Nat.mod_lt idx hdsPos) i' (by simpa using hi)

/-- `flatIndex s cs < shapeProd s` whenever `cs` has length `s.length` and every `cs[i] < s[i]`. -/
theorem flatIndex_lt (s cs : List Nat) (hlen : cs.length = s.length)
    (hlt : ∀ i (hi : i < s.length), cs[i]'(by rw [hlen]; exact hi) < s[i]) :
    flatIndex s cs < shapeProd s := by
  induction s generalizing cs with
  | nil => simp [flatIndex, shapeProd]
  | cons d ds ih =>
    cases cs with
    | nil => simp at hlen
    | cons c rest =>
      simp only [flatIndex, shapeProd]
      have hc : c < d := hlt 0 (by simp)
      have hrest : flatIndex ds rest < shapeProd ds :=
        ih rest (by simpa using hlen) (fun i hi => by
          have h := hlt (i + 1) (by simp; omega)
          simpa [List.getElem_cons_succ] using h)
      have hmul : (c + 1) * shapeProd ds ≤ d * shapeProd ds :=
        Nat.mul_le_mul_right (shapeProd ds) hc
      rw [Nat.add_mul, Nat.one_mul] at hmul
      omega

/-- Left inverse: `flatIndex s (unflatten s idx) = idx` for all `idx < shapeProd s`. -/
theorem flatIndex_unflatten (s : List Nat) (hpos : ∀ d ∈ s, d > 0) (idx : Nat) (hidx : idx < shapeProd s) :
    flatIndex s (unflatten s idx) = idx := by
  induction s generalizing idx with
  | nil =>
    simp [shapeProd] at hidx
    simp [flatIndex, hidx]
  | cons d ds ih =>
    have hdsPos : shapeProd ds > 0 := shapeProd_pos ds (fun x hx => hpos x (List.mem_cons_of_mem _ hx))
    simp only [unflatten, flatIndex]
    rw [ih (fun x hx => hpos x (List.mem_cons_of_mem _ hx)) (idx % shapeProd ds) (Nat.mod_lt idx hdsPos)]
    have hdm := Nat.div_add_mod idx (shapeProd ds)
    rw [Nat.mul_comm] at hdm
    exact hdm

/-- Right inverse: `unflatten s (flatIndex s cs) = cs` for all in-bounds coordinates `cs`. -/
theorem unflatten_flatIndex (s cs : List Nat) (hpos : ∀ d ∈ s, d > 0)
    (hlen : cs.length = s.length)
    (hlt : ∀ i (hi : i < s.length), cs[i]'(by rw [hlen]; exact hi) < s[i]) :
    unflatten s (flatIndex s cs) = cs := by
  induction s generalizing cs with
  | nil =>
    cases cs with
    | nil => rfl
    | cons _ _ => simp at hlen
  | cons d ds ih =>
    cases cs with
    | nil => simp at hlen
    | cons c rest =>
      have hdsPos : shapeProd ds > 0 := shapeProd_pos ds (fun x hx => hpos x (List.mem_cons_of_mem _ hx))
      have hrestLt : flatIndex ds rest < shapeProd ds :=
        flatIndex_lt ds rest (by simpa using hlen)
          (fun i hi => by
            have h := hlt (i + 1) (by simp; omega)
            simpa [List.getElem_cons_succ] using h)
      simp only [flatIndex, unflatten]
      have hdiv : (c * shapeProd ds + flatIndex ds rest) / shapeProd ds = c := by
        rw [Nat.mul_comm c (shapeProd ds), Nat.mul_add_div hdsPos,
            Nat.div_eq_of_lt hrestLt, Nat.add_zero]
      have hmod : (c * shapeProd ds + flatIndex ds rest) % shapeProd ds = flatIndex ds rest := by
        rw [Nat.mul_comm c (shapeProd ds), Nat.mul_add_mod, Nat.mod_eq_of_lt hrestLt]
      rw [hdiv, hmod]
      congr 1
      exact ih rest (fun x hx => hpos x (List.mem_cons_of_mem _ hx)) (by simpa using hlen)
        (fun i hi => by
          have h := hlt (i + 1) (by simp; omega)
          simpa [List.getElem_cons_succ] using h)

private theorem list_all_pos_iff (s : List Nat) :
    s.all (· > 0) = true ↔ ∀ d ∈ s, d > 0 := by
  rw [List.all_eq_true]
  constructor
  · intro h d hd; simpa using h d hd
  · intro h d hd; simpa using h d hd

private theorem tensorVal_ext {α : Type} {t1 t2 : TensorVal α}
    (hsh : t1.shape = t2.shape)
    (hdata : ∀ (i : Nat) (h1 : i < shapeProd t1.shape) (h2 : i < shapeProd t2.shape),
      t1.data ⟨i, h1⟩ = t2.data ⟨i, h2⟩) : t1 = t2 := by
  cases t1 with
  | mk s1 d1 =>
    cases t2 with
    | mk s2 d2 =>
      dsimp only at hsh hdata
      subst hsh
      congr 1
      funext idx
      exact hdata idx.val idx.isLt idx.isLt

/-- Exact element-for-element identity transpose equivalence over arbitrary scalar type `α`. -/
theorem evalTranspose_id {α : Type} (t : TensorVal α) (hpos : ∀ d ∈ t.shape, d > 0) :
    evalTranspose (idPerm t.shape.length) t = some t := by
  unfold evalTranspose
  have hv : isValidPerm (idPerm t.shape.length) = true :=
    (isValidPerm_iff _).mpr (validPerm_idPerm t.shape.length)
  have hsz : (idPerm t.shape.length).size = t.shape.length := by simp [idPerm]
  have hall : t.shape.all (· > 0) = true := (list_all_pos_iff t.shape).mpr hpos
  have hn : 0 < shapeProd t.shape := shapeProd_pos t.shape hpos
  simp only [hv, hsz, beq_self_eq_true, hall, Bool.and_self, ↓reduceIte, hn, ↓reduceDIte]
  congr 1
  have houtSh : transposeShape t.shape (idPerm t.shape.length) = t.shape := by
    unfold transposeShape
    apply List.ext_getElem
    · simp [idPerm]
    · intro i h1 h2
      simp only [List.getElem_map, idPerm, Array.toList_ofFn, List.getElem_ofFn]
      exact list_getD_eq t.shape i 0 h2
  apply tensorVal_ext houtSh
  intro i h1 h2
  have hcoords :
      ((List.range t.shape.length).map fun a =>
        (unflatten (transposeShape t.shape (idPerm t.shape.length)) i).getD
          ((invPerm (idPerm t.shape.length)).getD a 0) 0) =
      unflatten t.shape i := by
    rw [houtSh, invPerm_idPerm]
    apply List.ext_getElem
    · simp [unflatten_length]
    · intro k hk1 hk2
      have hk : k < t.shape.length := by simpa using hk1
      simp only [List.getElem_map, List.getElem_range]
      have hid : (idPerm t.shape.length).getD k 0 = k := by
        rw [arr_getD_eq _ _ _ (by simp [idPerm, hk])]
        simp [idPerm]
      rw [hid, list_getD_eq _ _ _ (by rw [unflatten_length]; exact hk)]
  dsimp only
  rw [hcoords, flatIndex_unflatten t.shape hpos i h2]
  simp [h2]

private theorem transposeShape_compose (s : List Nat) (p1 p2 : Array Nat)
    (hp2 : ValidPerm p2) (hsz12 : p1.size = p2.size) :
    transposeShape (transposeShape s p1) p2 = transposeShape s (composePerm p1 p2) := by
  unfold transposeShape composePerm
  apply List.ext_getElem
  · simp
  · intro k hk1 hk2
    have hk : k < p2.size := by simpa using hk1
    have hp2k : p2[k] < p1.size := by rw [hsz12]; exact hp2.1 k hk
    simp only [List.getElem_map, Array.getElem_toList, Array.toList_map]
    rw [list_getD_eq _ _ _ (by simp; omega)]
    rw [arr_getD_eq _ _ _ hp2k]
    simp

private theorem transposeShape_pos (s : List Nat) (p : Array Nat)
    (hp : ValidPerm p) (hsz : p.size = s.length) (hpos : ∀ d ∈ s, d > 0) :
    ∀ d ∈ transposeShape s p, d > 0 := by
  intro d hd
  unfold transposeShape at hd
  obtain ⟨a, ha, rfl⟩ := List.mem_map.mp hd
  obtain ⟨k, hk, rfl⟩ := List.mem_iff_getElem.mp ha
  have hk' : k < p.size := by simpa using hk
  have hlt : p[k] < s.length := by rw [← hsz]; exact hp.1 k hk'
  have hEq : s.getD p.toList[k] 0 = s[p[k]] := by
    simpa using list_getD_eq s p[k] 0 hlt
  rw [hEq]
  exact hpos _ (List.getElem_mem hlt)

/-- Exact element-for-element composed transpose equivalence:
    `evalTranspose p2 (evalTranspose p1 t) = evalTranspose (composePerm p1 p2) t`. -/
theorem evalTranspose_compose {α : Type} (p1 p2 : Array Nat) (t t1 : TensorVal α)
    (hp1 : ValidPerm p1) (hp2 : ValidPerm p2) (hsz : p1.size = p2.size)
    (ht1 : evalTranspose p1 t = some t1) :
    evalTranspose p2 t1 = evalTranspose (composePerm p1 p2) t := by
  unfold evalTranspose at ht1
  have hv1 : isValidPerm p1 = true := (isValidPerm_iff p1).mpr hp1
  simp only [hv1, Bool.true_and] at ht1
  split at ht1
  · rename_i hcond1
    simp only [Bool.and_eq_true, beq_iff_eq] at hcond1
    obtain ⟨hp1Len, hallPos⟩ := hcond1
    have hpos : ∀ d ∈ t.shape, d > 0 := (list_all_pos_iff t.shape).mp hallPos
    have hn : 0 < shapeProd t.shape := shapeProd_pos t.shape hpos
    simp only [hn, ↓reduceDIte, Option.some.injEq] at ht1
    subst ht1
    have hpos1 : ∀ d ∈ transposeShape t.shape p1, d > 0 := transposeShape_pos t.shape p1 hp1 hp1Len hpos
    have hall1 : (transposeShape t.shape p1).all (· > 0) = true := (list_all_pos_iff _).mpr hpos1
    have hlen1 : (transposeShape t.shape p1).length = p2.size := by simp [transposeShape, hsz]
    have hn1 : 0 < shapeProd (transposeShape t.shape p1) := shapeProd_pos _ hpos1
    have hv2 : isValidPerm p2 = true := (isValidPerm_iff p2).mpr hp2
    have hcompValid : ValidPerm (composePerm p1 p2) := validPerm_composePerm p1 p2 hp1 hp2 hsz
    have hvC : isValidPerm (composePerm p1 p2) = true := (isValidPerm_iff _).mpr hcompValid
    have hcompLen : (composePerm p1 p2).size = t.shape.length := by simp [composePerm, ← hsz, hp1Len]
    unfold evalTranspose
    simp only [hv2, hlen1, beq_self_eq_true, hall1, Bool.and_self, ↓reduceIte, hn1, ↓reduceDIte,
               hvC, hcompLen, hallPos, hn]
    congr 1
    have hshEq : transposeShape (transposeShape t.shape p1) p2 =
        transposeShape t.shape (composePerm p1 p2) :=
      transposeShape_compose t.shape p1 p2 hp2 hsz
    apply tensorVal_ext hshEq
    intro i hi1 hi2
    dsimp only
    -- Show `inCoords1` is in-bounds for `transposeShape t.shape p1`, so `unflatten (flatIndex inCoords1) = inCoords1`!
    let outCoords1 := unflatten (transposeShape (transposeShape t.shape p1) p2) i
    let inCoords1 := (List.range (transposeShape t.shape p1).length).map fun b =>
      outCoords1.getD ((invPerm p2).getD b 0) 0
    have hin1Len : inCoords1.length = (transposeShape t.shape p1).length := by simp [inCoords1]
    have hin1Lt : ∀ b (hb : b < (transposeShape t.shape p1).length),
        inCoords1[b]'(by rw [hin1Len]; exact hb) < (transposeShape t.shape p1)[b] := by
      intro b hb
      have hb2 : b < p2.size := by omega
      simp only [inCoords1, List.getElem_map, List.getElem_range]
      have hinvLt : (invPerm p2)[b]'(by simp [invPerm, hb2]) < p2.size := (invPerm_right p2 hp2 b hb2).1
      have hinvGet : p2[(invPerm p2)[b]'(by simp [invPerm, hb2])]'hinvLt = b := (invPerm_right p2 hp2 b hb2).2
      rw [arr_getD_eq _ _ _ (by simp [invPerm, hb2])]
      have houtLen : outCoords1.length = p2.size := by simp [outCoords1, unflatten_length, transposeShape]
      rw [list_getD_eq _ _ _ (by omega)]
      have hltOut := unflatten_lt (transposeShape (transposeShape t.shape p1) p2)
        (transposeShape_pos _ p2 hp2 (by omega) hpos1) i hi1
        ((invPerm p2)[b]'(by simp [invPerm, hb2])) (by simp [transposeShape, hinvLt])
      have hdimEq : (transposeShape (transposeShape t.shape p1) p2)[(invPerm p2)[b]'(by simp [invPerm, hb2])]'(by simp [transposeShape, hinvLt]) =
          (transposeShape t.shape p1)[b] := by
        simp only [transposeShape, List.getElem_map, Array.getElem_toList]
        rw [hinvGet, list_getD_eq (List.map (fun a => t.shape.getD a 0) p1.toList) b 0 (by simpa [transposeShape] using hb)]
        simp
      simpa [outCoords1, hdimEq] using hltOut
    have hf1Lt : flatIndex (transposeShape t.shape p1) inCoords1 < shapeProd (transposeShape t.shape p1) :=
      flatIndex_lt _ inCoords1 hin1Len hin1Lt
    have hf1Lt' :
        flatIndex (transposeShape t.shape p1)
          ((List.range p2.size).map fun a =>
            (unflatten (transposeShape (transposeShape t.shape p1) p2) i).getD ((invPerm p2).getD a 0) 0) <
        shapeProd (transposeShape t.shape p1) := by
      simpa [inCoords1, outCoords1, hlen1] using hf1Lt
    rw [dite_eq_left_of_eq_true (eq_true hf1Lt')]
    have hunf1 :
        unflatten (transposeShape t.shape p1)
          (flatIndex (transposeShape t.shape p1)
            ((List.range p2.size).map fun a =>
              (unflatten (transposeShape (transposeShape t.shape p1) p2) i).getD ((invPerm p2).getD a 0) 0)) =
        inCoords1 := by
      have h0 := unflatten_flatIndex _ inCoords1 hpos1 hin1Len hin1Lt
      simpa [inCoords1, outCoords1, hlen1] using h0
    -- Now show the final coordinates into `t.shape` are identical on LHS and RHS!
    have hfinalCoords :
        ((List.range t.shape.length).map fun a =>
          (unflatten (transposeShape t.shape p1)
            (flatIndex (transposeShape t.shape p1)
              ((List.range p2.size).map fun a =>
                (unflatten (transposeShape (transposeShape t.shape p1) p2) i).getD ((invPerm p2).getD a 0) 0))).getD
            ((invPerm p1).getD a 0) 0) =
        ((List.range t.shape.length).map fun a =>
          (unflatten (transposeShape t.shape (composePerm p1 p2)) i).getD
            ((invPerm (composePerm p1 p2)).getD a 0) 0) := by
      rw [hunf1]
      apply List.ext_getElem
      · simp
      · intro a ha1 ha2
        have ha : a < t.shape.length := by simpa using ha1
        have ha1' : a < p1.size := by omega
        simp only [List.getElem_map, List.getElem_range]
        have hinv1Lt : (invPerm p1)[a]'(by simp [invPerm, ha1']) < p1.size := (invPerm_right p1 hp1 a ha1').1
        rw [arr_getD_eq _ _ _ (by simp [invPerm, ha1'])]
        have hinv1Lt' : (invPerm p1)[a]'(by simp [invPerm, ha1']) < inCoords1.length := by
          rw [hin1Len, hlen1, ← hsz]; exact hinv1Lt
        rw [list_getD_eq _ _ _ hinv1Lt']
        simp only [inCoords1, List.getElem_map, List.getElem_range]
        rw [invPerm_composePerm p1 p2 hp1 hp2 hsz]
        have hcompIdx : (composePerm (invPerm p2) (invPerm p1)).getD a 0 =
            (invPerm p2).getD ((invPerm p1)[a]'(by simp [invPerm, ha1'])) 0 := by
          rw [arr_getD_eq _ _ _ (by simp [composePerm, invPerm, ha1'])]
          simp [composePerm]
        rw [hcompIdx]
        simp only [outCoords1, hshEq]
    have hditeCongr : ∀ (c1 c2 : List Nat), c1 = c2 →
        (if hlt : flatIndex t.shape c1 < shapeProd t.shape then t.data ⟨flatIndex t.shape c1, hlt⟩ else t.data ⟨0, hn⟩) =
        (if hlt : flatIndex t.shape c2 < shapeProd t.shape then t.data ⟨flatIndex t.shape c2, hlt⟩ else t.data ⟨0, hn⟩) := by
      intro c1 c2 hc; subst hc; rfl
    exact hditeCongr _ _ hfinalCoords
  · cases ht1

/-- Exact identity `Reshape` equivalence over arbitrary scalar type `α`. -/
theorem evalReshape_id {α : Type} (t : TensorVal α) (hpos : ∀ d ∈ t.shape, d > 0) :
    evalReshape t.shape.toArray t = some t := by
  unfold evalReshape
  have hall : t.shape.toArray.all (· > 0) = true := by
    rw [Array.all_eq_true_iff_forall_mem]
    intro d hd
    have hdL : d ∈ t.shape := by simpa [Array.mem_def] using hd
    simpa using hpos d hdL
  simp only [List.toList_toArray, ↓reduceDIte, hall, ↓reduceIte]

/-- Exact composed `Reshape` equivalence: `evalReshape s2 (evalReshape s1 t) = evalReshape s2 t`. -/
theorem evalReshape_compose {α : Type} (s1 s2 : Array Nat) (t t1 : TensorVal α)
    (ht1 : evalReshape s1 t = some t1) :
    evalReshape s2 t1 = evalReshape s2 t := by
  unfold evalReshape at ht1 ⊢
  split at ht1
  · rename_i h1
    split at ht1
    · simp only [Option.some.injEq] at ht1
      subst ht1
      dsimp only
      by_cases h2 : shapeProd s2.toList = shapeProd s1.toList
      · have h2t : shapeProd s2.toList = shapeProd t.shape := h2.trans h1
        rw [dite_eq_left_of_eq_true (eq_true h2), dite_eq_left_of_eq_true (eq_true h2t)]
      · have h2t : shapeProd s2.toList ≠ shapeProd t.shape := fun hc => h2 (hc.trans h1.symm)
        rw [dite_eq_right_of_eq_false (eq_false h2), dite_eq_right_of_eq_false (eq_false h2t)]
    · cases ht1
  · cases ht1

/-- Exact full-span identity `Slice` equivalence over arbitrary scalar type `α`. -/
theorem evalSlice_id {α : Type} (t : TensorVal α) (hpos : ∀ d ∈ t.shape, d > 0) :
    evalSlice (Array.replicate t.shape.length 0) t.shape.toArray
      (Array.replicate t.shape.length 1) t = some t := by
  unfold evalSlice
  have hv : validSliceParams t.shape (Array.replicate t.shape.length 0) t.shape.toArray
      (Array.replicate t.shape.length 1) = true := by
    unfold validSliceParams
    simp only [Array.size_replicate, List.size_toArray, beq_self_eq_true, Bool.true_and,
               List.all_eq_true, List.mem_range, Bool.and_eq_true, decide_eq_true_eq]
    intro i hi
    have hdi : t.shape.getD i 0 = t.shape[i] := list_getD_eq t.shape i 0 hi
    have hgt : t.shape[i] > 0 := hpos _ (List.getElem_mem hi)
    have hb : (Array.replicate t.shape.length 0).getD i 0 = 0 := by
      rw [arr_getD_eq _ _ _ (by simp [hi])]; simp
    have he : t.shape.toArray.getD i 0 = t.shape[i] := by
      rw [arr_getD_eq _ _ _ (by simp [hi])]; simp
    have hs : (Array.replicate t.shape.length 1).getD i 0 = 1 := by
      rw [arr_getD_eq _ _ _ (by simp [hi])]; simp
    rw [hb, he, hs, hdi]
    exact ⟨⟨by omega, hgt⟩, Nat.le_refl _⟩
  have hall : t.shape.all (· > 0) = true := (list_all_pos_iff t.shape).mpr hpos
  have hn : 0 < shapeProd t.shape := shapeProd_pos t.shape hpos
  simp only [hv, hall, Bool.and_self, ↓reduceIte, hn, ↓reduceDIte]
  congr 1
  have houtSh : sliceShape t.shape (Array.replicate t.shape.length 0) t.shape.toArray
      (Array.replicate t.shape.length 1) = t.shape := by
    unfold sliceShape sliceAxisDim
    apply List.ext_getElem
    · simp
    · intro i h1 h2
      have hb : (Array.replicate t.shape.length 0).getD i 0 = 0 := by
        rw [arr_getD_eq _ _ _ (by simp [h2])]; simp
      have he : t.shape.toArray.getD i 0 = t.shape[i] := by
        rw [arr_getD_eq _ _ _ (by simp [h2])]; simp
      have hs : (Array.replicate t.shape.length 1).getD i 0 = 1 := by
        rw [arr_getD_eq _ _ _ (by simp [h2])]; simp
      have hgt : t.shape[i] > 0 := hpos _ (List.getElem_mem h2)
      simp only [List.getElem_map, List.getElem_range, hb, he, hs]
      have hcond : (1 == 0 || decide (t.shape[i] ≤ 0)) = false := by
        simp; omega
      simp only [hcond, Bool.false_eq_true, ↓reduceIte]
      omega
  apply tensorVal_ext houtSh
  intro i h1 h2
  dsimp only
  have hcoords :
      ((List.range t.shape.length).map fun k =>
        (Array.replicate t.shape.length 0).getD k 0 +
        (unflatten (sliceShape t.shape (Array.replicate t.shape.length 0) t.shape.toArray
          (Array.replicate t.shape.length 1)) i).getD k 0 *
        (Array.replicate t.shape.length 1).getD k 0) =
      unflatten t.shape i := by
    rw [houtSh]
    apply List.ext_getElem
    · simp [unflatten_length]
    · intro k hk1 hk2
      have hk : k < t.shape.length := by simpa using hk1
      have hb : (Array.replicate t.shape.length 0).getD k 0 = 0 := by
        rw [arr_getD_eq _ _ _ (by simp [hk])]; simp
      have hs : (Array.replicate t.shape.length 1).getD k 0 = 1 := by
        rw [arr_getD_eq _ _ _ (by simp [hk])]; simp
      simp only [List.getElem_map, List.getElem_range, hb, hs, Nat.zero_add, Nat.mul_one]
      exact list_getD_eq _ _ _ (by rw [unflatten_length]; exact hk)
  rw [hcoords, flatIndex_unflatten t.shape hpos i h2]
  simp only [h2, ↓reduceDIte]

-- 23.4 Generic `applyRewrites` and `runPasses` compositionality theorems

private theorem applyFirstRule_mem {Op : Type} {rules : List (RewriteRule Op)} {g g' : SSAGraph Op}
    (h : applyFirstRule rules g = some g') :
    ∃ r ∈ rules, r.apply g = some g' := by
  induction rules with
  | nil => cases h
  | cons r rest ih =>
    unfold applyFirstRule at h
    cases hr : r.apply g with
    | some g1 =>
      simp only [hr, Option.some.injEq] at h
      subst h
      exact ⟨r, List.mem_cons_self .., hr⟩
    | none =>
      simp only [hr] at h
      obtain ⟨r', hmem, happ⟩ := ih h
      exact ⟨r', List.mem_cons_of_mem _ hmem, happ⟩

/-- Generic `applyRewrites` soundness combinator: if every rule in `rules` preserves
    `WellFormed` and `SemRefines sem`, then `applyRewrites fuel rules` preserves both
    `WellFormed` and `SemRefines sem` for any fuel bound `fuel`. -/
theorem evalGraph_applyRewrites {Op α : Type} (sem : OpSemantics Op α)
    (fuel : Nat) (rules : List (RewriteRule Op))
    (hrules : ∀ r ∈ rules, r.PreservesWF ∧ r.PreservesSem sem) :
    PreservesWF (applyRewrites fuel rules) ∧ PreservesSem sem (applyRewrites fuel rules) := by
  induction fuel with
  | zero =>
    constructor
    · intro g hwf; exact hwf
    · intro g _ env outs h; exact h
  | succ f ih =>
    constructor
    · intro g hwf
      unfold applyRewrites
      cases hstep : applyFirstRule rules g with
      | none => exact hwf
      | some g' =>
        obtain ⟨r, hmem, happ⟩ := applyFirstRule_mem hstep
        have hwf' := (hrules r hmem).1 g g' hwf happ
        exact ih.1 g' hwf'
    · intro g hwf env outs hout
      unfold applyRewrites
      cases hstep : applyFirstRule rules g with
      | none => exact hout
      | some g' =>
        obtain ⟨r, hmem, happ⟩ := applyFirstRule_mem hstep
        have hwf' := (hrules r hmem).1 g g' hwf happ
        have hout' := (hrules r hmem).2 g g' hwf happ env outs hout
        exact ih.2 g' hwf' env outs hout'

/-- `runPasses` compositionality for `WellFormed` preservation. -/
theorem runPasses_preservesWF {Op : Type} (passes : List (SSAGraph Op → SSAGraph Op))
    (hpasses : ∀ p ∈ passes, PreservesWF p) :
    PreservesWF (runPasses passes) := by
  induction passes with
  | nil => intro g hwf; exact hwf
  | cons p ps ih =>
    intro g hwf
    unfold runPasses
    have hwf1 := hpasses p (List.mem_cons_self ..) g hwf
    exact ih (fun q hq => hpasses q (List.mem_cons_of_mem _ hq)) (p g) hwf1

/-- `runPasses` compositionality for semantic preservation (`PreservesSem`). -/
theorem runPasses_preservesSem {Op α : Type} (sem : OpSemantics Op α)
    (passes : List (SSAGraph Op → SSAGraph Op))
    (hpasses : ∀ p ∈ passes, PreservesWF p ∧ PreservesSem sem p) :
    PreservesSem sem (runPasses passes) := by
  induction passes with
  | nil => intro g _ env outs h; exact h
  | cons p ps ih =>
    intro g hwf env outs hout
    unfold runPasses
    have hp := hpasses p (List.mem_cons_self ..)
    have hwf1 := hp.1 g hwf
    have hout1 := hp.2 g hwf env outs hout
    exact ih (fun q hq => hpasses q (List.mem_cons_of_mem _ hq)) (p g) hwf1 env outs hout1

-- 23.5 Functional Equivalence of `DCE` (`wellFormed_dce`, `evalGraph_dce`)

private theorem lookupAll_congr {α : Type} {env1 env2 : Env α} {ids : List ValueId}
    (h : ∀ v ∈ ids, env2 v = env1 v) :
    lookupAll env2 ids = lookupAll env1 ids := by
  induction ids with
  | nil => rfl
  | cons v vs ih =>
    simp only [lookupAll]
    rw [h v (List.mem_cons_self ..), ih (fun x hx => h x (List.mem_cons_of_mem _ hx))]

private theorem bindOutputs_not_mem {α : Type} (env : Env α)
    (os : List (ValueId × TensorType)) (ts : List (TensorVal α))
    (v : ValueId) (hnot : ∀ o ∈ os, o.1 ≠ v) :
    bindOutputs env os ts v = env v := by
  induction os generalizing env ts with
  | nil => rfl
  | cons o rest ih =>
    cases ts with
    | nil => rfl
    | cons t ts' =>
      simp only [bindOutputs]
      rw [ih (Env.set env o.1 t) ts' (fun x hx => hnot x (List.mem_cons_of_mem _ hx))]
      have hne : (v == o.1) = false := by
        cases hc : (v == o.1) with
        | false => rfl
        | true => exact absurd (beq_iff_eq.mp hc).symm (hnot o (List.mem_cons_self ..))
      simp [Env.set, hne]

private theorem bindOutputs_agree {α : Type} (env1 env2 : Env α)
    (os : List (ValueId × TensorType)) (ts : List (TensorVal α))
    (v : ValueId) (hagree : env2 v = env1 v) :
    bindOutputs env2 os ts v = bindOutputs env1 os ts v := by
  induction os generalizing env1 env2 ts with
  | nil => exact hagree
  | cons o rest ih =>
    cases ts with
    | nil => exact hagree
    | cons t ts' =>
      simp only [bindOutputs]
      apply ih
      simp only [Env.set]
      split
      · rfl
      · exact hagree

private theorem evalNodes_dceNodes {Op α : Type} (sem : OpSemantics Op α)
    (nodes : List (IRNode Op)) (needed : List ValueId)
    (env1 env2 final1 : Env α)
    (hagree : ∀ v ∈ (dceNodes nodes needed).2, env2 v = env1 v)
    (heval : evalNodes sem env1 nodes = some final1) :
    ∃ final2, evalNodes sem env2 (dceNodes nodes needed).1 = some final2 ∧
      ∀ v ∈ needed, final2 v = final1 v := by
  induction nodes generalizing env1 env2 with
  | nil =>
    simp only [dceNodes, evalNodes, Option.some.injEq] at heval ⊢
    subst heval
    exact ⟨env2, rfl, hagree⟩
  | cons nd rest ih =>
    simp only [evalNodes] at heval
    cases hnd : evalNode sem env1 nd with
    | none => simp [hnd] at heval
    | some env1' =>
      simp only [hnd] at heval
      unfold dceNodes
      dsimp only
      split
      · -- Node `nd` is KEPT
        rename_i hlive
        have hagreeIn : ∀ v ∈ nd.inputs.toList, env2 v = env1 v := by
          intro v hv
          apply hagree
          simp only [dceNodes, hlive, ↓reduceIte, List.mem_append]
          exact Or.inl hv
        have hagreeAfter : ∀ v ∈ (dceNodes rest needed).2, env2 v = env1 v := by
          intro v hv
          apply hagree
          simp only [dceNodes, hlive, ↓reduceIte, List.mem_append]
          exact Or.inr hv
        -- Evaluate `nd` under `env2`
        unfold evalNode at hnd
        cases hlu : lookupAll env1 nd.inputs.toList with
        | none => simp [hlu] at hnd
        | some inVals =>
          simp only [hlu] at hnd
          cases hop : sem.evalOp nd.op inVals (nd.outputs.toList.map Prod.snd) with
          | none => simp [hop] at hnd
          | some outVals =>
            simp only [hop] at hnd
            split at hnd
            · rename_i hsm
              simp only [Option.some.injEq] at hnd
              subst hnd
              have hnd2 : evalNode sem env2 nd = some (bindOutputs env2 nd.outputs.toList outVals) := by
                unfold evalNode
                rw [lookupAll_congr hagreeIn, hlu]
                dsimp only
                rw [hop]
                dsimp only
                rw [hsm]
                rfl
              simp only [evalNodes, hnd2]
              exact ih (bindOutputs env1 nd.outputs.toList outVals)
                (bindOutputs env2 nd.outputs.toList outVals)
                (fun v hv => bindOutputs_agree env1 env2 nd.outputs.toList outVals v (hagreeAfter v hv))
                heval
            · cases hnd
      · -- Node `nd` is DROPPED (none of its outputs is in `(dceNodes rest needed).2`)
        rename_i hdead
        have hagreeAfter : ∀ v ∈ (dceNodes rest needed).2, env2 v = env1 v := by
          intro v hv
          apply hagree
          simp only [dceNodes, hdead, Bool.false_eq_true, ↓reduceIte]
          exact hv
        unfold evalNode at hnd
        cases hlu : lookupAll env1 nd.inputs.toList with
        | none => simp [hlu] at hnd
        | some inVals =>
          simp only [hlu] at hnd
          cases hop : sem.evalOp nd.op inVals (nd.outputs.toList.map Prod.snd) with
          | none => simp [hop] at hnd
          | some outVals =>
            simp only [hop] at hnd
            split at hnd
            · simp only [Option.some.injEq] at hnd
              subst hnd
              apply ih (bindOutputs env1 nd.outputs.toList outVals) env2 _ heval
              intro v hv
              have hnot : ∀ o ∈ nd.outputs.toList, o.1 ≠ v := by
                intro o ho heq
                subst heq
                have hany : nd.outputs.toList.any (fun o => (dceNodes rest needed).2.elem o.1) = true :=
                  List.any_eq_true.mpr ⟨o, ho, List.elem_eq_true_of_mem hv⟩
                exact hdead hany
              rw [bindOutputs_not_mem env1 nd.outputs.toList outVals v hnot]
              exact hagreeAfter v hv
            · cases hnd

private theorem dceNodes_sublist {Op : Type} (nodes : List (IRNode Op)) (needed : List ValueId) :
    (dceNodes nodes needed).1.Sublist nodes := by
  induction nodes with
  | nil => simp [dceNodes]
  | cons nd rest ih =>
    unfold dceNodes
    dsimp only
    split
    · exact List.Sublist.cons_cons nd ih
    · exact List.Sublist.cons nd ih

private theorem dceNodes_needed_sub {Op : Type} (nodes : List (IRNode Op)) (needed : List ValueId) :
    ∀ v ∈ needed, v ∈ (dceNodes nodes needed).2 := by
  induction nodes with
  | nil => intro v hv; exact hv
  | cons nd rest ih =>
    intro v hv
    unfold dceNodes
    dsimp only
    split
    · exact List.mem_append_right _ (ih v hv)
    · exact ih v hv

private theorem dceNodes_kept_inputs_in_needed {Op : Type} (nodes : List (IRNode Op)) (needed : List ValueId) :
    ∀ nd ∈ (dceNodes nodes needed).1, ∀ v ∈ nd.inputs.toList, v ∈ (dceNodes nodes needed).2 := by
  induction nodes with
  | nil => intro nd hnd; simp [dceNodes] at hnd
  | cons a rest ih =>
    intro nd hnd v hv
    unfold dceNodes at hnd ⊢
    dsimp only at hnd ⊢
    split at hnd
    · rename_i hlive
      simp only [hlive, ↓reduceIte, List.mem_append]
      rcases List.mem_cons.mp hnd with rfl | hmem
      · exact Or.inl hv
      · exact Or.inr (ih nd hmem v hv)
    · rename_i hdead
      simp only [hdead, Bool.false_eq_true, ↓reduceIte]
      exact ih nd hnd v hv

private theorem dceNodes_covers_needed {Op : Type} (nodes : List (IRNode Op)) (needed : List ValueId)
    (avail : List ValueId)
    (hdef : ∀ i (hi : i < nodes.length), ∀ v ∈ nodes[i].inputs.toList,
      v ∈ avail ++ (nodes.take i).flatMap IRNode.outputIds)
    (hroot : ∀ v ∈ needed, v ∈ avail ++ nodes.flatMap IRNode.outputIds) :
    (∀ v ∈ (dceNodes nodes needed).2, v ∈ avail ++ (dceNodes nodes needed).1.flatMap IRNode.outputIds) ∧
    (∀ i (hi : i < (dceNodes nodes needed).1.length), ∀ v ∈ (dceNodes nodes needed).1[i].inputs.toList,
      v ∈ avail ++ ((dceNodes nodes needed).1.take i).flatMap IRNode.outputIds) := by
  induction nodes generalizing avail with
  | nil =>
    simp only [dceNodes, List.flatMap_nil, List.append_nil] at hroot ⊢
    exact ⟨hroot, fun i hi => by simp at hi⟩
  | cons nd rest ih =>
    have hdefRest : ∀ i (hi : i < rest.length), ∀ v ∈ rest[i].inputs.toList,
        v ∈ (avail ++ nd.outputIds) ++ (rest.take i).flatMap IRNode.outputIds := by
      intro i hi v hv
      have h := hdef (i + 1) (by simp; omega) v (by simpa using hv)
      simpa [List.take_succ_cons, List.flatMap_cons, List.append_assoc] using h
    have hrootRest : ∀ v ∈ needed, v ∈ (avail ++ nd.outputIds) ++ rest.flatMap IRNode.outputIds := by
      intro v hv
      simpa [List.flatMap_cons, List.append_assoc] using hroot v hv
    obtain ⟨hNeedR, hDefR⟩ := ih (avail ++ nd.outputIds) hdefRest hrootRest
    have hndIn : ∀ v ∈ nd.inputs.toList, v ∈ avail := by
      intro v hv
      have h := hdef 0 (by simp) v (by simpa using hv)
      simpa using h
    unfold dceNodes
    dsimp only
    split
    · -- Node `nd` is kept
      refine ⟨?_, ?_⟩
      · intro v hv
        simp only [List.flatMap_cons, ← List.append_assoc]
        rcases List.mem_append.mp hv with hvIn | hvAfter
        · exact List.mem_append_left _ (List.mem_append_left _ (hndIn v hvIn))
        · exact hNeedR v hvAfter
      · intro i hi v hv
        cases i with
        | zero =>
          simp only [List.getElem_cons_zero] at hv
          simpa using hndIn v hv
        | succ i' =>
          simp only [List.getElem_cons_succ] at hv
          have hi' : i' < (dceNodes rest needed).1.length := by simpa using hi
          have h := hDefR i' hi' v hv
          simpa [List.take_succ_cons, List.flatMap_cons, List.append_assoc] using h
    · -- Node `nd` is dropped: none of `nd.outputIds` is in `(dceNodes rest needed).2`
      rename_i hdead
      have hnotOut : ∀ v ∈ (dceNodes rest needed).2, v ∉ nd.outputIds := by
        intro v hv hvOut
        unfold IRNode.outputIds at hvOut
        obtain ⟨o, ho, rfl⟩ := List.mem_map.mp hvOut
        have hany : nd.outputs.toList.any (fun o => (dceNodes rest needed).2.elem o.1) = true :=
          List.any_eq_true.mpr ⟨o, ho, List.elem_eq_true_of_mem hv⟩
        exact hdead hany
      refine ⟨?_, ?_⟩
      · intro v hv
        have h := hNeedR v hv
        simp only [List.mem_append] at h ⊢
        rcases h with (hAv | hNd) | hRest
        · exact Or.inl hAv
        · exact absurd hNd (hnotOut v hv)
        · exact Or.inr hRest
      · intro i hi v hv
        have h := hDefR i hi v hv
        have hvNeed : v ∈ (dceNodes rest needed).2 :=
          dceNodes_kept_inputs_in_needed rest needed _ (List.getElem_mem hi) v hv
        simp only [List.mem_append] at h ⊢
        rcases h with (hAv | hNd) | hRest
        · exact Or.inl hAv
        · exact absurd hNd (hnotOut v hvNeed)
        · exact Or.inr hRest

private theorem sublist_flatMap {α β : Type} (f : α → List β) {l1 l2 : List α}
    (h : l1.Sublist l2) : (l1.flatMap f).Sublist (l2.flatMap f) := by
  induction h with
  | slnil => exact List.Sublist.slnil
  | cons a _ ih =>
    simp only [List.flatMap_cons]
    exact List.Sublist.trans ih (List.sublist_append_right _ _)
  | cons_cons a _ ih =>
    simp only [List.flatMap_cons]
    exact List.Sublist.append (List.Sublist.refl _) ih

/-- `DCE` preserves `WellFormed` for every `SSAGraph Op`. -/
theorem wellFormed_dce {Op : Type} : PreservesWF (dce (Op := Op)) := by
  intro g hwf
  have hcov := dceNodes_covers_needed g.nodes.toList g.outputs.toList g.inputIds
    (fun i hi v hv => hwf.defBeforeUse i (by simpa using hi) v (by simpa using hv))
    (fun v hv => hwf.outputsValid v hv)
  have hsubNodes : (dceNodes g.nodes.toList g.outputs.toList).1.Sublist g.nodes.toList :=
    dceNodes_sublist g.nodes.toList g.outputs.toList
  constructor
  · exact hwf.inputsDense
  · exact hwf.inputsPositive
  · -- uniqueIds: sublist of `g.allDefinedIds`
    unfold SSAGraph.allDefinedIds SSAGraph.inputIds dce
    have hsubAll :
        (g.inputIds ++ (dceNodes g.nodes.toList g.outputs.toList).1.flatMap IRNode.outputIds).Sublist
        g.allDefinedIds :=
      List.Sublist.append (List.Sublist.refl _) (sublist_flatMap IRNode.outputIds hsubNodes)
    simpa [SSAGraph.inputIds] using List.Nodup.sublist hsubAll hwf.uniqueIds
  · -- defBeforeUse
    intro i hi v hv
    unfold SSAGraph.allDefinedIdsBefore dce at hi hv ⊢
    unfold SSAGraph.inputIds
    simp only [List.size_toArray] at hi
    simpa [SSAGraph.inputIds] using hcov.2 i hi v (by simpa using hv)
  · -- nodeOutputsPositive
    intro i hi o ho d hd
    have hmem : (dce g).nodes[i] ∈ g.nodes.toList :=
      hsubNodes.subset (by simp [dce])
    obtain ⟨j, hj, hjeq⟩ := List.mem_iff_getElem.mp hmem
    have hj' : j < g.nodes.size := by simpa using hj
    have ho' : o ∈ g.nodes[j].outputs.toList := by
      have : g.nodes[j] = (dce g).nodes[i] := by simpa using hjeq
      rw [this]; exact ho
    exact hwf.nodeOutputsPositive j hj' o ho' d hd
  · -- outputsValid
    intro v hv
    unfold SSAGraph.allDefinedIds SSAGraph.inputIds dce
    simpa [SSAGraph.inputIds] using hcov.1 v (dceNodes_needed_sub g.nodes.toList g.outputs.toList v hv)

/-- Forward semantic refinement (`SemRefines`) of `DCE`: for every `WellFormed` graph `g` and every
    input environment `env`, if `evalGraphOutputs sem g env = some outs`, then
    `evalGraphOutputs sem (dce g) env = some outs`. -/
theorem evalGraph_dce {Op α : Type} (sem : OpSemantics Op α) :
    PreservesSem sem (dce (Op := Op)) := by
  intro g _ env outs hout
  unfold evalGraphOutputs evalGraph at hout ⊢
  dsimp only [dce]
  by_cases hInMatch : inputShapesMatch env g.inputs.toList = true
  · simp only [hInMatch, ↓reduceIte] at hout ⊢
    cases hev : evalNodes sem env g.nodes.toList with
    | none => simp [hev] at hout
    | some final1 =>
      simp only [hev] at hout
      obtain ⟨final2, hev2, hagree⟩ := evalNodes_dceNodes sem g.nodes.toList g.outputs.toList env env final1
        (fun _ _ => rfl) hev
      simp only [hev2]
      rw [lookupAll_congr hagree]
      exact hout
  · simp [hInMatch] at hout

-- 23.6 Functional Equivalence of `ViewSimplify` Rules & Full `viewSimplify` Pass

private theorem evalNodes_append {Op α : Type} (sem : OpSemantics Op α)
    (env : Env α) (l1 l2 : List (IRNode Op)) :
    evalNodes sem env (l1 ++ l2) =
    match evalNodes sem env l1 with
    | none => none
    | some env1 => evalNodes sem env1 l2 := by
  induction l1 generalizing env with
  | nil => rfl
  | cons nd rest ih =>
    simp only [List.cons_append, evalNodes]
    cases evalNode sem env nd with
    | none => rfl
    | some env' => exact ih env'

private theorem evalNodes_not_mem_preserved {Op α : Type} (sem : OpSemantics Op α)
    (nodes : List (IRNode Op)) (env final : Env α) (v : ValueId)
    (hnot : v ∉ nodes.flatMap IRNode.outputIds)
    (hev : evalNodes sem env nodes = some final) :
    final v = env v := by
  induction nodes generalizing env with
  | nil =>
    simp only [evalNodes, Option.some.injEq] at hev
    rw [hev]
  | cons nd rest ih =>
    simp only [List.flatMap_cons, List.mem_append, not_or] at hnot
    simp only [evalNodes] at hev
    cases hnd : evalNode sem env nd with
    | none => simp [hnd] at hev
    | some env' =>
      simp only [hnd] at hev
      rw [ih env' hnot.2 hev]
      unfold evalNode at hnd
      cases hlu : lookupAll env nd.inputs.toList with
      | none => simp [hlu] at hnd
      | some inVals =>
        simp only [hlu] at hnd
        cases hop : sem.evalOp nd.op inVals (nd.outputs.toList.map Prod.snd) with
        | none => simp [hop] at hnd
        | some outVals =>
          simp only [hop] at hnd
          split at hnd
          · simp only [Option.some.injEq] at hnd
            subst hnd
            apply bindOutputs_not_mem
            intro o ho heq
            exact hnot.1 (List.mem_map.mpr ⟨o, ho, heq⟩)
          · cases hnd

private theorem findFirstNodeMatch_some {Op β : Type} {f : Nat → IRNode Op → Option β}
    {base : Nat} {xs : List (IRNode Op)} {y : β}
    (h : findFirstNodeMatch f base xs = some y) :
    ∃ k, ∃ (hk : k < xs.length), f (base + k) xs[k] = some y := by
  induction xs generalizing base with
  | nil => cases h
  | cons nd rest ih =>
    unfold findFirstNodeMatch at h
    cases hf : f base nd with
    | some z =>
      simp only [hf, Option.some.injEq] at h
      subst h
      exact ⟨0, by simp, by simpa using hf⟩
    | none =>
      simp only [hf] at h
      obtain ⟨k, hk, hfk⟩ := ih h
      exact ⟨k + 1, by simp; omega, by simpa [Nat.add_assoc, Nat.add_comm 1 k] using hfk⟩

private theorem findSoleProducerBefore_some {Op : Type} {nodes : List (IRNode Op)}
    {limit : Nat} {targetId : ValueId} {pnd : IRNode Op}
    (h : findSoleProducerBefore nodes limit targetId = some pnd) :
    ∃ j, ∃ (_ : j < limit), ∃ (hjLen : j < nodes.length), ∃ ty : TensorType,
      nodes[j] = pnd ∧ pnd.outputs.toList = [(targetId, ty)] := by
  unfold findSoleProducerBefore at h
  have hmem : pnd ∈ nodes.take limit := List.mem_of_find?_eq_some h
  have hpred := List.find?_some h
  obtain ⟨j, hjTake, hget⟩ := List.mem_iff_getElem.mp hmem
  have hjLim : j < limit := by simp at hjTake; omega
  have hjLen : j < nodes.length := by simp at hjTake; omega
  have hgetN : nodes[j] = pnd := by simpa [List.getElem_take] using hget
  cases hout : pnd.outputs.toList with
  | nil => simp [hout] at hpred
  | cons o rest =>
    cases rest with
    | nil =>
      simp only [hout, beq_iff_eq] at hpred
      cases o with
      | mk vid ty =>
        dsimp only at hpred
        subst hpred
        exact ⟨j, hjLim, hjLen, ty, hgetN, rfl⟩
    | cons _ _ => simp [hout] at hpred

private theorem inputShapesMatch_get {α : Type} {env : Env α}
    {inps : List (ValueId × String × TensorType)}
    (hm : inputShapesMatch env inps = true) :
    ∀ e ∈ inps, ∃ t, env e.1 = some t ∧ t.shape = e.2.2.shape.toList := by
  induction inps with
  | nil => intro e he; cases he
  | cons hd tl ih =>
    unfold inputShapesMatch at hm
    cases henv : env hd.1 with
    | none => simp [henv] at hm
    | some t =>
      simp only [henv, Bool.and_eq_true, beq_iff_eq] at hm
      intro e he
      rcases List.mem_cons.mp he with rfl | htl
      · exact ⟨t, henv, hm.1⟩
      · exact ih hm.2 e htl

private theorem shapesMatch_bind_spec {α : Type} (env : Env α)
    (os : List (ValueId × TensorType)) (ts : List (TensorVal α))
    (hsm : shapesMatch os ts = true) (hnd : (os.map Prod.fst).Nodup) :
    ∀ o ∈ os, ∃ t, bindOutputs env os ts o.1 = some t ∧ t.shape = o.2.shape.toList := by
  induction os generalizing env ts with
  | nil => intro o ho; cases ho
  | cons hd tl ih =>
    cases ts with
    | nil => simp [shapesMatch] at hsm
    | cons t rest =>
      simp only [shapesMatch, Bool.and_eq_true, beq_iff_eq] at hsm
      simp only [List.map_cons, List.nodup_cons] at hnd
      intro o ho
      rcases List.mem_cons.mp ho with rfl | htl
      · refine ⟨t, ?_, hsm.1⟩
        simp only [bindOutputs]
        rw [bindOutputs_not_mem (Env.set env o.1 t) tl rest o.1 (fun x hx heq => hnd.1 (List.mem_map.mpr ⟨x, hx, heq⟩))]
        simp [Env.set]
      · exact ih (Env.set env hd.1 t) rest hsm.2 hnd.2 o htl

private theorem valueType_unique_inputs {Op : Type} {g : SSAGraph Op}
    {v : ValueId} {ty : TensorType}
    (h : g.valueType? v = some ty) :
    (∃ e ∈ g.inputs.toList, e.1 = v ∧ e.2.2 = ty) ∨
    (∃ nd ∈ g.nodes.toList, ∃ o ∈ nd.outputs.toList, o.1 = v ∧ o.2 = ty) := by
  unfold SSAGraph.valueType? at h
  cases hfind : g.inputs.find? (fun e => e.1 == v) with
  | some e =>
    simp only [hfind, Option.some.injEq] at h
    have hmem : e ∈ g.inputs.toList := by
      have := Array.mem_of_find?_eq_some hfind
      exact Array.mem_def.mp this
    have hp := Array.find?_some hfind
    exact Or.inl ⟨e, hmem, beq_iff_eq.mp hp, h⟩
  | none =>
    simp only [hfind] at h
    have hfs : ∃ nd ∈ g.nodes.toList, ((nd.outputs.find? fun o => o.1 == v).map Prod.snd) = some ty := by
      have : ∃ nd ∈ g.nodes, ((nd.outputs.find? fun o => o.1 == v).map Prod.snd) = some ty := by
        rw [← Array.findSome?_toList] at h
        obtain ⟨nd, hnd, hval⟩ := List.exists_of_findSome?_eq_some h
        exact ⟨nd, Array.mem_def.mpr hnd, hval⟩
      obtain ⟨nd, hnd, hval⟩ := this
      exact ⟨nd, Array.mem_def.mp hnd, hval⟩
    obtain ⟨nd, hnd, hmap⟩ := hfs
    simp only [Option.map_eq_some_iff] at hmap
    obtain ⟨o, hofind, rfl⟩ := hmap
    have hom := Array.mem_def.mp (Array.mem_of_find?_eq_some hofind)
    have hop0 : (o.1 == v) = true := Array.find?_some (p := fun x : ValueId × TensorType => x.1 == v) hofind
    have hop := beq_iff_eq.mp hop0
    exact Or.inr ⟨nd, hnd, o, hom, hop, rfl⟩

private theorem evalNodes_value_shape {Op α : Type} (sem : OpSemantics Op α)
    {g : SSAGraph Op} (hwf : WellFormed g) {env envK : Env α} {k : Nat} (hk : k ≤ g.nodes.size)
    (hInMatch : inputShapesMatch env g.inputs.toList = true)
    (hev : evalNodes sem env (g.nodes.toList.take k) = some envK)
    {v : ValueId} {ty : TensorType} {t : TensorVal α}
    (hvDef : v ∈ g.allDefinedIdsBefore k)
    (hvTy : g.valueType? v = some ty)
    (hvVal : envK v = some t) :
    t.shape = ty.shape.toList ∧ ∀ d ∈ t.shape, d > 0 := by
  -- Extract the unique defining site of `v` inside `g.inputs` or `g.nodes.take k`
  have hnodup := hwf.uniqueIds
  rw [allDefinedIds_split g k] at hnodup
  have hnodupBefore : (g.allDefinedIdsBefore k).Nodup := (List.nodup_append.mp hnodup).1
  unfold SSAGraph.allDefinedIdsBefore at hvDef hnodupBefore
  rcases List.mem_append.mp hvDef with hvIn | hvNode
  · -- `v` is a graph input
    obtain ⟨e, heMem, rfl⟩ := List.mem_map.mp hvIn
    have hnotNodes : e.1 ∉ (g.nodes.toList.take k).flatMap IRNode.outputIds :=
      nodup_append_disjoint hnodupBefore e.1 hvIn
    have hpres := evalNodes_not_mem_preserved sem (g.nodes.toList.take k) env envK e.1 hnotNodes hev
    rw [hpres] at hvVal
    obtain ⟨t0, ht0, ht0Sh⟩ := inputShapesMatch_get hInMatch e heMem
    rw [ht0, Option.some.injEq] at hvVal
    subst hvVal
    -- Show `ty = e.2.2` using `valueType_unique_inputs` and `uniqueIds`
    have htyEq : ty = e.2.2 := by
      rcases valueType_unique_inputs hvTy with ⟨e', he'Mem, he'Id, rfl⟩ | ⟨nd, hndMem, o, hoMem, hoId, _⟩
      · obtain ⟨i1, hi1, hi1Eq⟩ := List.mem_iff_getElem.mp heMem
        obtain ⟨i2, hi2, hi2Eq⟩ := List.mem_iff_getElem.mp he'Mem
        have h1 : i1 < g.inputs.size := by simpa using hi1
        have h2 : i2 < g.inputs.size := by simpa using hi2
        have hd1 := hwf.inputsDense i1 h1
        have hd2 := hwf.inputsDense i2 h2
        have hEqIdx : i1 = i2 := by
          have he1 : g.inputs[i1].1 = e.1 := by simpa using congrArg Prod.fst hi1Eq
          have he2 : g.inputs[i2].1 = e'.1 := by simpa using congrArg Prod.fst hi2Eq
          have : i1 = e.1 := hd1.symm.trans he1
          have : i2 = e.1 := (hd2.symm.trans he2).trans he'Id
          omega
        subst hEqIdx
        have : e = e' := by rw [← hi1Eq, ← hi2Eq]
        rw [this]
      · have hvNodes : e.1 ∈ g.nodes.toList.flatMap IRNode.outputIds :=
          List.mem_flatMap.mpr ⟨nd, hndMem, List.mem_map.mpr ⟨o, hoMem, hoId⟩⟩
        exact absurd hvNodes (nodup_append_disjoint hwf.uniqueIds e.1 hvIn)
    subst htyEq
    refine ⟨ht0Sh, ?_⟩
    rw [ht0Sh]
    obtain ⟨i, hi, hiEq⟩ := List.mem_iff_getElem.mp heMem
    have hi' : i < g.inputs.size := by simpa using hi
    have hposI := hwf.inputsPositive i hi'
    have heqI : g.inputs[i] = e := by simpa using hiEq
    rw [← heqI]
    exact hposI
  · -- `v` is produced by some node `j < k`
    obtain ⟨nd, hndMem, hvOut⟩ := List.mem_flatMap.mp hvNode
    obtain ⟨j, hjTake, hjEq⟩ := List.mem_iff_getElem.mp hndMem
    have hjK : j < k := by simp at hjTake; omega
    have hjSz : j < g.nodes.size := by omega
    have hjNode : g.nodes.toList[j]'(by simpa using hjSz) = nd := by
      simpa [List.getElem_take] using hjEq
    obtain ⟨o, hoMem, hoId⟩ := List.mem_map.mp hvOut
    -- Split `g.nodes.toList.take k` at `j`
    have hsplitTake : g.nodes.toList.take k =
        g.nodes.toList.take j ++ [nd] ++ (g.nodes.toList.take k).drop (j + 1) := by
      have h2 : (g.nodes.toList.take k).drop j = nd :: (g.nodes.toList.take k).drop (j + 1) := by
        rw [List.drop_eq_getElem_cons hjTake, hjEq]
      calc g.nodes.toList.take k
          = (g.nodes.toList.take k).take j ++ (g.nodes.toList.take k).drop j := (List.take_append_drop j _).symm
        _ = g.nodes.toList.take j ++ (nd :: (g.nodes.toList.take k).drop (j + 1)) := by
            rw [List.take_take, Nat.min_eq_left (by omega), h2]
        _ = g.nodes.toList.take j ++ [nd] ++ (g.nodes.toList.take k).drop (j + 1) := by
            simp [List.append_assoc]
    rw [hsplitTake, evalNodes_append, evalNodes_append] at hev
    cases hevPre : evalNodes sem env (g.nodes.toList.take j) with
    | none => simp [hevPre] at hev
    | some envJ =>
      simp only [hevPre] at hev
      cases hevNd : evalNode sem envJ nd with
      | none => simp [evalNodes, hevNd] at hev
      | some envAfterJ =>
        simp only [evalNodes, hevNd] at hev
        -- `v` is not overwritten by `(g.nodes.toList.take k).drop (j + 1)`
        have hnotPost : v ∉ ((g.nodes.toList.take k).drop (j + 1)).flatMap IRNode.outputIds := by
          have hndFlat : ((g.nodes.toList.take k).flatMap IRNode.outputIds).Nodup :=
            (List.nodup_append.mp hnodupBefore).2.1
          rw [hsplitTake, List.flatMap_append, List.flatMap_append] at hndFlat
          have hmemMid : v ∈ (g.nodes.toList.take j).flatMap IRNode.outputIds ++ [nd].flatMap IRNode.outputIds :=
            List.mem_append_right _ (by simpa using hvOut)
          exact nodup_append_disjoint hndFlat v hmemMid
        have hpres := evalNodes_not_mem_preserved sem ((g.nodes.toList.take k).drop (j + 1)) envAfterJ envK v hnotPost hev
        rw [hpres] at hvVal
        -- Extract output shape from `evalNode sem envJ nd = some envAfterJ`
        unfold evalNode at hevNd
        cases hlu : lookupAll envJ nd.inputs.toList with
        | none => simp [hlu] at hevNd
        | some inVals =>
          simp only [hlu] at hevNd
          cases hop : sem.evalOp nd.op inVals (nd.outputs.toList.map Prod.snd) with
          | none => simp [hop] at hevNd
          | some outVals =>
            simp only [hop] at hevNd
            split at hevNd
            · rename_i hsm
              simp only [Option.some.injEq] at hevNd
              subst hevNd
              have hndOut : (nd.outputs.toList.map Prod.fst).Nodup := by
                have hndFlat : ((g.nodes.toList.take k).flatMap IRNode.outputIds).Nodup :=
                  (List.nodup_append.mp hnodupBefore).2.1
                rw [hsplitTake, List.flatMap_append, List.flatMap_append] at hndFlat
                have h1 := (List.nodup_append.mp hndFlat).1
                have h2 := (List.nodup_append.mp h1).2.1
                simpa [IRNode.outputIds] using h2
              obtain ⟨t0, ht0, ht0Sh⟩ := shapesMatch_bind_spec envJ nd.outputs.toList outVals hsm hndOut o hoMem
              rw [hoId] at ht0
              rw [ht0, Option.some.injEq] at hvVal
              subst hvVal
              -- Show `ty = o.2` by uniqueness of `v` in `g.allDefinedIds`
              have htyEq : ty = o.2 := by
                rcases valueType_unique_inputs hvTy with ⟨e', he'Mem, he'Id, _⟩ | ⟨nd', hnd'Mem, o', ho'Mem, ho'Id, rfl⟩
                · have hvIn' : v ∈ g.inputIds := List.mem_map.mpr ⟨e', he'Mem, he'Id⟩
                  exact absurd hvNode (nodup_append_disjoint hnodupBefore v hvIn')
                · -- Both `(nd, o)` and `(nd', o')` define `v`, so by `uniqueIds` they are the same node & output!
                  obtain ⟨j', hj', hj'Eq⟩ := List.mem_iff_getElem.mp hnd'Mem
                  have hjEqJ' : j = j' := by
                    have hndAllNodes : (g.nodes.toList.flatMap IRNode.outputIds).Nodup :=
                      (List.nodup_append.mp hwf.uniqueIds).2.1
                    rcases Nat.lt_trichotomy j j' with hlt | heq | hgt
                    · exfalso
                      have hsplit := (List.take_append_drop j' g.nodes.toList).symm
                      rw [hsplit, List.flatMap_append] at hndAllNodes
                      have hInL : v ∈ (g.nodes.toList.take j').flatMap IRNode.outputIds :=
                        List.mem_flatMap.mpr ⟨nd, by
                          have : (g.nodes.toList.take j')[j]'(by simp [Array.length_toList]; omega) = nd := by
                            simpa [List.getElem_take] using hjNode
                          rw [← this]; exact List.getElem_mem _, hvOut⟩
                      have hInR : v ∈ (g.nodes.toList.drop j').flatMap IRNode.outputIds :=
                        List.mem_flatMap.mpr ⟨nd', by
                          have : (g.nodes.toList.drop j')[0]'(by simp only [List.length_drop]; omega) = nd' := by
                            simpa [List.getElem_drop] using hj'Eq
                          rw [← this]; exact List.getElem_mem _, List.mem_map.mpr ⟨o', ho'Mem, ho'Id⟩⟩
                      exact nodup_append_disjoint hndAllNodes v hInL hInR
                    · exact heq
                    · exfalso
                      have hsplit := (List.take_append_drop j g.nodes.toList).symm
                      rw [hsplit, List.flatMap_append] at hndAllNodes
                      have hInL : v ∈ (g.nodes.toList.take j).flatMap IRNode.outputIds :=
                        List.mem_flatMap.mpr ⟨nd', by
                          have : (g.nodes.toList.take j)[j']'(by simp only [List.length_take]; omega) = nd' := by
                            simpa [List.getElem_take] using hj'Eq
                          rw [← this]; exact List.getElem_mem _, List.mem_map.mpr ⟨o', ho'Mem, ho'Id⟩⟩
                      have hInR : v ∈ (g.nodes.toList.drop j).flatMap IRNode.outputIds :=
                        List.mem_flatMap.mpr ⟨nd, by
                          have : (g.nodes.toList.drop j)[0]'(by simp only [List.length_drop, Array.length_toList]; omega) = nd := by
                            simpa [List.getElem_drop] using hjNode
                          rw [← this]; exact List.getElem_mem _, hvOut⟩
                      exact nodup_append_disjoint hndAllNodes v hInL hInR
                  subst hjEqJ'
                  have hndEq : nd' = nd := by rw [← hj'Eq, hjNode]
                  subst hndEq
                  obtain ⟨m1, hm1, hm1Eq⟩ := List.mem_iff_getElem.mp hoMem
                  obtain ⟨m2, hm2, hm2Eq⟩ := List.mem_iff_getElem.mp ho'Mem
                  have hmEq : m1 = m2 := by
                    apply list_nodup_getElem_inj hndOut (by simpa using hm1) (by simpa using hm2)
                    simp only [List.getElem_map]
                    rw [hm1Eq, hm2Eq, hoId, ho'Id]
                  subst hmEq
                  have : o' = o := by rw [← hm2Eq, ← hm1Eq]
                  rw [this]
              subst htyEq
              refine ⟨ht0Sh, ?_⟩
              rw [ht0Sh]
              have hoNode : o ∈ g.nodes[j].outputs.toList := by
                have : g.nodes[j] = nd := by simpa using hjNode
                rw [this]; exact hoMem
              exact hwf.nodeOutputsPositive j hjSz o hoNode
            · cases hevNd

-- General well-formedness & semantic preservation for `eliminateUnaryIdentityNode`

private theorem eliminateUnaryIdentityNode_preservesWF {Op : Type}
    {g : SSAGraph Op} (hwf : WellFormed g) {idx : Nat} (hidx : idx < g.nodes.size)
    {inId outId : ValueId} {outTy : TensorType}
    (hIn : g.nodes[idx].inputs.toList = [inId])
    (hOut : g.nodes[idx].outputs.toList = [(outId, outTy)]) :
    WellFormed (eliminateUnaryIdentityNode g idx inId outId) := by
  let g' := eliminateUnaryIdentityNode g idx inId outId
  let subst : ValueId → ValueId := fun v => if v == outId then inId else v
  have hinDef : inId ∈ g.allDefinedIdsBefore idx :=
    hwf.defBeforeUse idx hidx inId (by rw [hIn]; exact List.mem_cons_self ..)
  have houtDefIdx : g.nodes[idx].outputIds = [outId] := by
    unfold IRNode.outputIds; rw [hOut]; rfl
  have hsubErase : (g.nodes.toList.eraseIdx idx).Sublist g.nodes.toList :=
    List.eraseIdx_sublist g.nodes.toList idx
  have houtIdsMap : ∀ (f : ValueId → ValueId) (l : List (IRNode Op)),
      (l.map fun nd => { nd with inputs := nd.inputs.map f }).flatMap IRNode.outputIds =
      l.flatMap IRNode.outputIds := by
    intro f l
    induction l with
    | nil => rfl
    | cons a tl ih => simp [IRNode.outputIds, ih]
  have htakeErase : ∀ k : Nat, k ≤ idx →
      (g.nodes.toList.eraseIdx idx).take k = g.nodes.toList.take k := by
    intro k hk
    rw [List.eraseIdx_eq_take_drop_succ, List.take_append_of_le_length (by simp; omega),
        List.take_take, Nat.min_eq_left hk]
  have htakeEraseGe : ∀ k : Nat, idx ≤ k →
      ((g.nodes.toList.eraseIdx idx).take k).flatMap IRNode.outputIds =
      (g.nodes.toList.take idx).flatMap IRNode.outputIds ++
      ((g.nodes.toList.drop (idx + 1)).take (k - idx)).flatMap IRNode.outputIds := by
    intro k hk
    rw [List.eraseIdx_eq_take_drop_succ, List.take_append, List.length_take]
    have hmin : min idx g.nodes.toList.length = idx := Nat.min_eq_left (by simp; omega)
    rw [hmin, List.take_take, Nat.min_eq_right hk, List.flatMap_append]
  have hAllIdsBeforeGe : ∀ k : Nat, idx ≤ k → ∀ v,
      v ∈ g.allDefinedIdsBefore (k + 1) → v ≠ outId → v ∈ g'.allDefinedIdsBefore k := by
    intro k hk v hv hne
    unfold SSAGraph.allDefinedIdsBefore SSAGraph.inputIds at hv ⊢
    dsimp only [g', eliminateUnaryIdentityNode]
    rw [List.toList_toArray, ← List.map_take, houtIdsMap]
    rw [htakeEraseGe k hk]
    have hsplitK : g.nodes.toList.take (k + 1) =
        g.nodes.toList.take idx ++ [g.nodes[idx]] ++ (g.nodes.toList.drop (idx + 1)).take (k - idx) := by
      have h1 : g.nodes.toList.take (k + 1) =
          g.nodes.toList.take idx ++ (g.nodes.toList.drop idx).take (k + 1 - idx) := by
        rw [← List.take_append_drop idx (g.nodes.toList.take (k + 1)), List.take_take,
            Nat.min_eq_left (by omega), List.drop_take]
      rw [h1]
      have hdrop : g.nodes.toList.drop idx = g.nodes[idx] :: g.nodes.toList.drop (idx + 1) := by
        have hidxL : idx < g.nodes.toList.length := by simpa using hidx
        rw [List.drop_eq_getElem_cons hidxL]; rfl
      rw [hdrop]
      have hsub : k + 1 - idx = (k - idx) + 1 := by omega
      rw [hsub, List.take_succ_cons]
      simp [List.append_assoc]
    rw [hsplitK, List.flatMap_append, List.flatMap_append] at hv
    simp only [List.mem_append, List.flatMap_singleton, houtDefIdx, List.mem_singleton] at hv ⊢
    rcases hv with hvIn | ((hvPre | hvMid) | hvPost)
    · exact Or.inl hvIn
    · exact Or.inr (Or.inl hvPre)
    · exact absurd hvMid hne
    · exact Or.inr (Or.inr hvPost)

  constructor
  · exact hwf.inputsDense
  · exact hwf.inputsPositive
  · -- uniqueIds
    unfold SSAGraph.allDefinedIds SSAGraph.inputIds
    dsimp only [g', eliminateUnaryIdentityNode]
    rw [List.toList_toArray, houtIdsMap]
    have hsubAll : (g.inputIds ++ (g.nodes.toList.eraseIdx idx).flatMap IRNode.outputIds).Sublist g.allDefinedIds :=
      List.Sublist.append (List.Sublist.refl _) (sublist_flatMap IRNode.outputIds hsubErase)
    exact List.Nodup.sublist hsubAll hwf.uniqueIds
  · -- defBeforeUse
    intro k hk v hv
    have hkLen : k < (g.nodes.toList.eraseIdx idx).length := by
      simpa [g', eliminateUnaryIdentityNode] using hk
    have hndEq : g'.nodes[k] = { (g.nodes.toList.eraseIdx idx)[k] with
        inputs := (g.nodes.toList.eraseIdx idx)[k].inputs.map subst } := by
      simp [g', eliminateUnaryIdentityNode, subst]
    rw [hndEq] at hv
    simp only [Array.toList_map, List.mem_map] at hv
    obtain ⟨w, hwIn, rfl⟩ := hv
    by_cases hlt : k < idx
    · have heraseGet : (g.nodes.toList.eraseIdx idx)[k] = g.nodes[k]'(by omega) := by
        rw [List.getElem_eraseIdx_of_lt _ hlt]; rfl
      rw [heraseGet] at hwIn
      have hwBefore : w ∈ g.allDefinedIdsBefore k := hwf.defBeforeUse k (by omega) w hwIn
      -- `w ≠ outId` because `outId` is defined at `idx ≥ k` and `uniqueIds` holds!
      have hwNe : w ≠ outId := by
        intro heq
        have hnodup := hwf.uniqueIds
        rw [allDefinedIds_split g idx] at hnodup
        have houtBefore : outId ∈ g.allDefinedIdsBefore idx := by
          rw [← heq]
          unfold SSAGraph.allDefinedIdsBefore at hwBefore ⊢
          rcases List.mem_append.mp hwBefore with h1 | h2
          · exact List.mem_append_left _ h1
          · refine List.mem_append_right _ ?_
            have hsubT : (g.nodes.toList.take k).Sublist (g.nodes.toList.take idx) := by
              have : g.nodes.toList.take k = (g.nodes.toList.take idx).take k := by
                rw [List.take_take, Nat.min_eq_left (by omega)]
              rw [this]; exact List.take_sublist _ _
            exact (sublist_flatMap IRNode.outputIds hsubT).subset h2
        have houtDrop : outId ∈ (g.nodes.toList.drop idx).flatMap IRNode.outputIds :=
          List.mem_flatMap.mpr ⟨g.nodes[idx], by
            have : (g.nodes.toList.drop idx)[0]'(by simp; omega) = g.nodes[idx] := by
              simp [List.getElem_drop]
            rw [← this]; exact List.getElem_mem _, by rw [houtDefIdx]; exact List.mem_cons_self ..⟩
        exact nodup_append_disjoint hnodup outId houtBefore houtDrop
      have hsubId : subst w = w := by simp [subst, hwNe]
      rw [hsubId]
      unfold SSAGraph.allDefinedIdsBefore SSAGraph.inputIds
      dsimp only [g', eliminateUnaryIdentityNode]
      rw [List.toList_toArray, ← List.map_take, houtIdsMap, htakeErase k (by omega)]
      exact hwBefore
    · have hge : idx ≤ k := by omega
      have hk1 : k + 1 < g.nodes.size := by
        have hlenE : (g.nodes.toList.eraseIdx idx).length = g.nodes.size - 1 := by
          rw [List.length_eraseIdx_of_lt (by simpa using hidx)]; simp
        omega
      have heraseGet : (g.nodes.toList.eraseIdx idx)[k] = g.nodes[k + 1]'hk1 := by
        rw [List.getElem_eraseIdx_of_ge _ hge]; rfl
      rw [heraseGet] at hwIn
      have hwBefore : w ∈ g.allDefinedIdsBefore (k + 1) := hwf.defBeforeUse (k + 1) hk1 w hwIn
      by_cases hwEq : w = outId
      · have hsubOut : subst w = inId := by simp [subst, hwEq]
        rw [hsubOut]
        unfold SSAGraph.allDefinedIdsBefore SSAGraph.inputIds at hinDef ⊢
        dsimp only [g', eliminateUnaryIdentityNode]
        rw [List.toList_toArray, ← List.map_take, houtIdsMap, htakeEraseGe k hge]
        rcases List.mem_append.mp hinDef with h1 | h2
        · exact List.mem_append_left _ h1
        · exact List.mem_append_right _ (List.mem_append_left _ h2)
      · have hsubW : subst w = w := by simp [subst, hwEq]
        rw [hsubW]
        exact hAllIdsBeforeGe k hge w hwBefore hwEq
  · -- nodeOutputsPositive
    intro k hk o ho d hd
    have hkLen : k < (g.nodes.toList.eraseIdx idx).length := by
      simpa [g', eliminateUnaryIdentityNode] using hk
    have hndEq : g'.nodes[k] = { (g.nodes.toList.eraseIdx idx)[k] with
        inputs := (g.nodes.toList.eraseIdx idx)[k].inputs.map subst } := by
      simp [g', eliminateUnaryIdentityNode, subst]
    rw [hndEq] at ho
    have hmem : (g.nodes.toList.eraseIdx idx)[k] ∈ g.nodes.toList :=
      hsubErase.subset (List.getElem_mem hkLen)
    obtain ⟨j, hj, hjEq⟩ := List.mem_iff_getElem.mp hmem
    have hj' : j < g.nodes.size := by simpa using hj
    have ho' : o ∈ g.nodes[j].outputs.toList := by
      have : g.nodes[j] = (g.nodes.toList.eraseIdx idx)[k] := by simpa using hjEq
      rw [this]; exact ho
    exact hwf.nodeOutputsPositive j hj' o ho' d hd
  · -- outputsValid
    intro v hv
    dsimp only [g', eliminateUnaryIdentityNode] at hv
    simp only [Array.toList_map, List.mem_map] at hv
    obtain ⟨w, hwOut, rfl⟩ := hv
    have hwAll := hwf.outputsValid w hwOut
    have hAllBeforeSize : g.allDefinedIds = g.allDefinedIdsBefore g.nodes.size := by
      unfold SSAGraph.allDefinedIds SSAGraph.allDefinedIdsBefore
      rw [List.take_of_length_le (by simp)]
    have hAllBeforeG' : g'.allDefinedIds = g'.allDefinedIdsBefore g'.nodes.size := by
      unfold SSAGraph.allDefinedIds SSAGraph.allDefinedIdsBefore
      rw [List.take_of_length_le (by simp)]
    have hszG' : g'.nodes.size = g.nodes.size - 1 := by
      dsimp only [g', eliminateUnaryIdentityNode]
      simp only [List.size_toArray, List.length_map]
      rw [List.length_eraseIdx_of_lt (by rw [Array.length_toList]; exact hidx), Array.length_toList]
    by_cases hwEq : w = outId
    · have hsubOut : (if (w == outId) = true then inId else w) = inId := by simp [hwEq]
      rw [hsubOut]
      change inId ∈ g'.allDefinedIds
      rw [hAllBeforeG', hszG']
      unfold SSAGraph.allDefinedIdsBefore SSAGraph.inputIds at hinDef ⊢
      dsimp only [g', eliminateUnaryIdentityNode]
      rw [List.toList_toArray, ← List.map_take, houtIdsMap, htakeEraseGe (g.nodes.size - 1) (by omega)]
      rcases List.mem_append.mp hinDef with h1 | h2
      · exact List.mem_append_left _ h1
      · exact List.mem_append_right _ (List.mem_append_left _ h2)
    · have hsubW : (if (w == outId) = true then inId else w) = w := by simp [hwEq]
      rw [hsubW]
      change w ∈ g'.allDefinedIds
      rw [hAllBeforeG']
      have hwB : w ∈ g.allDefinedIdsBefore (g'.nodes.size + 1) := by
        rw [show g'.nodes.size + 1 = g.nodes.size by omega, ← hAllBeforeSize]
        exact hwAll
      exact hAllIdsBeforeGe g'.nodes.size (by omega) w hwB hwEq

private theorem evalNodes_subst_equiv {Op α : Type} (sem : OpSemantics Op α)
    (nodes : List (IRNode Op)) (inId outId : ValueId)
    (env1 env2 final1 : Env α)
    (hnotIn : inId ∉ nodes.flatMap IRNode.outputIds)
    (hnotOut : outId ∉ nodes.flatMap IRNode.outputIds)
    (hrel : ∀ v, env2 (if v == outId then inId else v) = env1 v)
    (hev : evalNodes sem env1 nodes = some final1) :
    ∃ final2,
      evalNodes sem env2 (nodes.map fun nd =>
        { nd with inputs := nd.inputs.map (fun v => if v == outId then inId else v) }) = some final2 ∧
      ∀ v, final2 (if v == outId then inId else v) = final1 v := by
  let subst : ValueId → ValueId := fun v => if v == outId then inId else v
  induction nodes generalizing env1 env2 with
  | nil =>
    simp only [evalNodes, List.map_nil, Option.some.injEq] at hev ⊢
    subst hev
    exact ⟨env2, rfl, hrel⟩
  | cons nd rest ih =>
    simp only [List.flatMap_cons, List.mem_append, not_or] at hnotIn hnotOut
    simp only [evalNodes, List.map_cons] at hev ⊢
    cases hnd : evalNode sem env1 nd with
    | none => simp [hnd] at hev
    | some env1' =>
      simp only [hnd] at hev
      unfold evalNode at hnd
      cases hlu : lookupAll env1 nd.inputs.toList with
      | none => simp [hlu] at hnd
      | some inVals =>
        simp only [hlu] at hnd
        cases hop : sem.evalOp nd.op inVals (nd.outputs.toList.map Prod.snd) with
        | none => simp [hop] at hnd
        | some outVals =>
          simp only [hop] at hnd
          split at hnd
          · rename_i hsm
            simp only [Option.some.injEq] at hnd
            subst hnd
            -- Show `lookupAll env2 (nd.inputs.map subst).toList = lookupAll env1 nd.inputs.toList`
            have hlu2 : lookupAll env2 (nd.inputs.map subst).toList = lookupAll env1 nd.inputs.toList := by
              rw [Array.toList_map]
              induction nd.inputs.toList with
              | nil => rfl
              | cons x xs ihx =>
                simp only [List.map_cons, lookupAll, ihx]
                have hx := hrel x
                dsimp only [subst]
                rw [hx]
            have hnd2 : evalNode sem env2 { nd with inputs := nd.inputs.map subst } =
                some (bindOutputs env2 nd.outputs.toList outVals) := by
              unfold evalNode
              dsimp only
              rw [hlu2, hlu]
              dsimp only
              rw [hop]
              dsimp only
              rw [hsm]
              rfl
            dsimp only [subst] at hnd2
            rw [hnd2]
            dsimp only
            apply ih (bindOutputs env1 nd.outputs.toList outVals)
              (bindOutputs env2 nd.outputs.toList outVals) hnotIn.2 hnotOut.2 _ hev
            intro v
            by_cases hvOut : v = outId
            · have hs : subst v = inId := by dsimp only [subst]; simp [hvOut]
              change bindOutputs env2 nd.outputs.toList outVals (subst v) =
                     bindOutputs env1 nd.outputs.toList outVals v
              rw [hs, hvOut]
              rw [bindOutputs_not_mem env2 nd.outputs.toList outVals inId
                    (fun o ho heq => hnotIn.1 (List.mem_map.mpr ⟨o, ho, heq⟩)),
                  bindOutputs_not_mem env1 nd.outputs.toList outVals outId
                    (fun o ho heq => hnotOut.1 (List.mem_map.mpr ⟨o, ho, heq⟩))]
              simpa using hrel outId
            · have hs : subst v = v := by dsimp only [subst]; simp [hvOut]
              change bindOutputs env2 nd.outputs.toList outVals (subst v) =
                     bindOutputs env1 nd.outputs.toList outVals v
              rw [hs]
              exact bindOutputs_agree env1 env2 nd.outputs.toList outVals v (by simpa [hvOut] using hrel v)
          · cases hnd

private theorem eliminateUnaryIdentityNode_preservesSem {Op α : Type} (sem : OpSemantics Op α)
    {g : SSAGraph Op} (hwf : WellFormed g) {idx : Nat} (hidx : idx < g.nodes.size)
    {inId outId : ValueId} {outTy : TensorType}
    (hIn : g.nodes[idx].inputs.toList = [inId])
    (hOut : g.nodes[idx].outputs.toList = [(outId, outTy)])
    (hid : ∀ (envIdx : Env α) (t res : TensorVal α),
      (∀ (ty : TensorType), g.valueType? inId = some ty → t.shape = ty.shape.toList ∧ ∀ d ∈ t.shape, d > 0) →
      envIdx inId = some t →
      sem.evalOp g.nodes[idx].op [t] [outTy] = some [res] →
      res = t) :
    SemRefines sem g (eliminateUnaryIdentityNode g idx inId outId) := by
  intro env outs hout
  unfold evalGraphOutputs evalGraph at hout ⊢
  cases hInMatch : inputShapesMatch env g.inputs.toList with
  | false => simp [hInMatch] at hout
  | true =>
    simp only [hInMatch, ↓reduceIte] at hout
    simp only [eliminateUnaryIdentityNode, hInMatch, ↓reduceIte]
    cases hev : evalNodes sem env g.nodes.toList with
    | none => simp [hev] at hout
    | some final1 =>
      simp only [hev] at hout
      -- Split `g.nodes.toList` into `pre ++ [nd] ++ post`
      have hsplit : g.nodes.toList =
          g.nodes.toList.take idx ++ [g.nodes[idx]] ++ g.nodes.toList.drop (idx + 1) := by
        have hdrop : g.nodes.toList.drop idx = g.nodes[idx] :: g.nodes.toList.drop (idx + 1) := by
          rw [List.drop_eq_getElem_cons (by simpa using hidx)]; rfl
        calc g.nodes.toList
            = g.nodes.toList.take idx ++ g.nodes.toList.drop idx := (List.take_append_drop idx _).symm
          _ = g.nodes.toList.take idx ++ (g.nodes[idx] :: g.nodes.toList.drop (idx + 1)) := by rw [hdrop]
          _ = g.nodes.toList.take idx ++ [g.nodes[idx]] ++ g.nodes.toList.drop (idx + 1) := by
              simp [List.append_assoc]
      rw [hsplit, evalNodes_append, evalNodes_append] at hev
      cases hevPre : evalNodes sem env (g.nodes.toList.take idx) with
      | none => simp [hevPre] at hev
      | some envIdx =>
        simp only [hevPre] at hev
        cases hevNd : evalNode sem envIdx g.nodes[idx] with
        | none => simp [evalNodes, hevNd] at hev
        | some envAfter =>
          simp only [evalNodes, hevNd] at hev
          -- Show `envAfter = Env.set envIdx outId t` where `envIdx inId = some t`
          unfold evalNode at hevNd
          rw [hIn, hOut] at hevNd
          simp only [List.map_cons, List.map_nil] at hevNd
          cases hlu : envIdx inId with
          | none => simp [lookupAll, hlu] at hevNd
          | some t =>
            simp only [lookupAll, hlu] at hevNd
            cases hop : sem.evalOp g.nodes[idx].op [t] [outTy] with
            | none => simp [hop] at hevNd
            | some outVals =>
              simp only [hop] at hevNd
              cases outVals with
              | nil => simp [shapesMatch] at hevNd
              | cons res restVals =>
                cases restVals with
                | cons _ _ => simp [shapesMatch] at hevNd
                | nil =>
                  split at hevNd
                  · simp only [Option.some.injEq] at hevNd
                    subst hevNd
                    have hinDef : inId ∈ g.allDefinedIdsBefore idx :=
                      hwf.defBeforeUse idx hidx inId (by rw [hIn]; exact List.mem_cons_self ..)
                    have hresEq : res = t := hid envIdx t res
                      (fun ty hty => evalNodes_value_shape sem hwf (by omega) hInMatch hevPre hinDef hty hlu)
                      hlu hop
                    subst hresEq
                    -- Check relation `∀ v, envIdx (subst v) = (Env.set envIdx outId res) v`!
                    have hrel : ∀ v, envIdx (if v == outId then inId else v) =
                        bindOutputs envIdx [(outId, outTy)] [res] v := by
                      intro v
                      simp only [bindOutputs, Env.set]
                      split
                      · rename_i heq
                        exact hlu
                      · rfl
                    -- In `g'`, nodes = `(pre.map subst) ++ (post.map subst)`.
                    -- Notice `pre.map subst = pre` because no node `< idx` uses `outId`!
                    have hpreEq : (g.nodes.toList.take idx).map (fun nd =>
                        { nd with inputs := nd.inputs.map (fun v => if v == outId then inId else v) }) =
                        g.nodes.toList.take idx := by
                      apply List.ext_getElem
                      · simp
                      · intro k hk1 hk2
                        have hkIdx : k < idx := by simp at hk2; omega
                        have hkSz : k < g.nodes.size := by omega
                        simp only [List.getElem_map, List.getElem_take]
                        have hgetN : g.nodes.toList[k]'(by simpa using hkSz) = g.nodes[k] := rfl
                        rw [hgetN]
                        cases hndK : g.nodes[k] with
                        | mk nm op inps outs =>
                          simp only [IRNode.mk.injEq, true_and, and_true]
                          apply Array.ext
                          · simp
                          · intro m hm1 hm2
                            simp only [Array.getElem_map]
                            have hmemIn : inps[m] ∈ g.nodes[k].inputs.toList := by
                              rw [hndK]; exact Array.getElem_mem_toList hm2
                            have hbef := hwf.defBeforeUse k hkSz inps[m] hmemIn
                            have hneOut : inps[m] ≠ outId := by
                              intro heq
                              have hnodup := hwf.uniqueIds
                              rw [allDefinedIds_split g idx] at hnodup
                              have hInBefore : outId ∈ g.allDefinedIdsBefore idx := by
                                rw [← heq]
                                unfold SSAGraph.allDefinedIdsBefore at hbef ⊢
                                rcases List.mem_append.mp hbef with h1 | h2
                                · exact List.mem_append_left _ h1
                                · refine List.mem_append_right _ ?_
                                  have hsubT : (g.nodes.toList.take k).Sublist (g.nodes.toList.take idx) := by
                                    have : g.nodes.toList.take k = (g.nodes.toList.take idx).take k := by
                                      rw [List.take_take, Nat.min_eq_left (by omega)]
                                    rw [this]; exact List.take_sublist _ _
                                  exact (sublist_flatMap IRNode.outputIds hsubT).subset h2
                              have hInDrop : outId ∈ (g.nodes.toList.drop idx).flatMap IRNode.outputIds :=
                                List.mem_flatMap.mpr ⟨g.nodes[idx], by
                                  have : (g.nodes.toList.drop idx)[0]'(by simp; omega) = g.nodes[idx] := by
                                    simp [List.getElem_drop]
                                  rw [← this]; exact List.getElem_mem _, by
                                    unfold IRNode.outputIds; rw [hOut]; exact List.mem_cons_self ..⟩
                              exact nodup_append_disjoint hnodup outId hInBefore hInDrop
                            simp [hneOut]
                    -- Show `inId` and `outId` are not redefined in `post`
                    have hnodupPost : (g.allDefinedIdsBefore (idx + 1) ++
                        (g.nodes.toList.drop (idx + 1)).flatMap IRNode.outputIds).Nodup := by
                      rw [← allDefinedIds_split g (idx + 1)]
                      exact hwf.uniqueIds
                    have hnotInPost : inId ∉ (g.nodes.toList.drop (idx + 1)).flatMap IRNode.outputIds := by
                      apply nodup_append_disjoint hnodupPost inId
                      unfold SSAGraph.allDefinedIdsBefore at hinDef ⊢
                      rcases List.mem_append.mp hinDef with h1 | h2
                      · exact List.mem_append_left _ h1
                      · refine List.mem_append_right _ ?_
                        have hsubT : (g.nodes.toList.take idx).Sublist (g.nodes.toList.take (idx + 1)) := by
                          have : g.nodes.toList.take idx = (g.nodes.toList.take (idx + 1)).take idx := by
                            rw [List.take_take, Nat.min_eq_left (by omega)]
                          rw [this]; exact List.take_sublist _ _
                        exact (sublist_flatMap IRNode.outputIds hsubT).subset h2
                    have hnotOutPost : outId ∉ (g.nodes.toList.drop (idx + 1)).flatMap IRNode.outputIds := by
                      apply nodup_append_disjoint hnodupPost outId
                      unfold SSAGraph.allDefinedIdsBefore
                      refine List.mem_append_right _ ?_
                      exact List.mem_flatMap.mpr ⟨g.nodes[idx], by
                        have : (g.nodes.toList.take (idx + 1))[idx]'(by simp; omega) = g.nodes[idx] := by
                          simp [List.getElem_take]
                        rw [← this]; exact List.getElem_mem _, by
                          unfold IRNode.outputIds; rw [hOut]; exact List.mem_cons_self ..⟩
                    -- Evaluate `g'`!
                    obtain ⟨final2, hevPost2, hfinalRel⟩ :=
                      evalNodes_subst_equiv sem (g.nodes.toList.drop (idx + 1)) inId outId
                        (bindOutputs envIdx [(outId, outTy)] [res]) envIdx final1 hnotInPost hnotOutPost hrel hev
                    rw [List.toList_toArray, List.eraseIdx_eq_take_drop_succ,
                        List.map_append, hpreEq, evalNodes_append, hevPre]
                    dsimp only
                    rw [hevPost2]
                    dsimp only
                    rw [Array.toList_map]
                    have hluOut : lookupAll final2 (g.outputs.toList.map fun v => if v == outId then inId else v) =
                        lookupAll final1 g.outputs.toList := by
                      induction g.outputs.toList with
                      | nil => rfl
                      | cons x xs ihx =>
                        simp only [List.map_cons, lookupAll, ihx]
                        rw [hfinalRel x]
                    rw [hluOut]
                    exact hout
                  · cases hevNd

-- General well-formedness & semantic preservation for `rewriteUnaryNodeInPlace`

private theorem rewriteUnaryNodeInPlace_preservesWF {Op : Type}
    {g : SSAGraph Op} (hwf : WellFormed g) {idx : Nat} (_hidx : idx < g.nodes.size)
    (newOp : Op) (newInId : ValueId)
    (hnewDef : newInId ∈ g.allDefinedIdsBefore idx) :
    WellFormed (rewriteUnaryNodeInPlace g idx newOp newInId) := by
  let g' := rewriteUnaryNodeInPlace g idx newOp newInId
  have hsz : g'.nodes.size = g.nodes.size := by simp [g', rewriteUnaryNodeInPlace]
  have houtSame : ∀ k (hk : k < g.nodes.size), (g'.nodes[k]'(by omega)).outputs = g.nodes[k].outputs := by
    intro k hk
    have hkMod : k < (g.nodes.modify idx fun nd => { nd with op := newOp, inputs := #[newInId] }).size := by
      simp [hk]
    change ((g.nodes.modify idx fun nd => { nd with op := newOp, inputs := #[newInId] })[k]'hkMod).outputs = g.nodes[k].outputs
    rw [Array.getElem_modify]
    split <;> rfl
  have hallBeforeEq : ∀ k, g'.allDefinedIdsBefore k = g.allDefinedIdsBefore k := by
    intro k
    unfold SSAGraph.allDefinedIdsBefore SSAGraph.inputIds
    congr 1
    have hlen : (g'.nodes.toList.take k).length = (g.nodes.toList.take k).length := by simp [hsz]
    have hmap : ∀ (l1 l2 : List (IRNode Op)), l1.length = l2.length →
        (∀ i (h1 : i < l1.length) (h2 : i < l2.length), l1[i].outputs = l2[i].outputs) →
        l1.flatMap IRNode.outputIds = l2.flatMap IRNode.outputIds := by
      intro l1 l2 hl hall
      induction l1 generalizing l2 with
      | nil => cases l2 with | nil => rfl | cons _ _ => simp at hl
      | cons a1 r1 ih =>
        cases l2 with
        | nil => simp at hl
        | cons a2 r2 =>
          simp only [List.flatMap_cons]
          have hhd : a1.outputs = a2.outputs := hall 0 (by simp) (by simp)
          rw [show a1.outputIds = a2.outputIds by unfold IRNode.outputIds; rw [hhd]]
          congr 1
          exact ih r2 (by simpa using hl) (fun i h1 h2 => hall (i + 1) (by simp; omega) (by simp; omega))
    apply hmap _ _ hlen
    intro i h1 h2
    simp only [List.getElem_take]
    have hiSz : i < g.nodes.size := by simp at h2; omega
    exact houtSame i hiSz
  have hallEq : g'.allDefinedIds = g.allDefinedIds := by
    have h1 : g'.allDefinedIds = g'.allDefinedIdsBefore g.nodes.size := by
      unfold SSAGraph.allDefinedIds SSAGraph.allDefinedIdsBefore
      rw [List.take_of_length_le (by simp [hsz])]
    have h2 : g.allDefinedIds = g.allDefinedIdsBefore g.nodes.size := by
      unfold SSAGraph.allDefinedIds SSAGraph.allDefinedIdsBefore
      rw [List.take_of_length_le (by simp)]
    rw [h1, h2, hallBeforeEq]
  constructor
  · exact hwf.inputsDense
  · exact hwf.inputsPositive
  · rw [hallEq]; exact hwf.uniqueIds
  · intro k hk v hv
    have hkSz : k < g.nodes.size := by simpa [rewriteUnaryNodeInPlace] using hk
    change v ∈ g'.allDefinedIdsBefore k
    rw [hallBeforeEq]
    have hkMod : k < (g.nodes.modify idx fun nd => { nd with op := newOp, inputs := #[newInId] }).size := by
      simp [hkSz]
    have hget : (rewriteUnaryNodeInPlace g idx newOp newInId).nodes[k]'hk =
        if idx = k then { g.nodes[k]'hkSz with op := newOp, inputs := #[newInId] } else g.nodes[k]'hkSz := by
      change (g.nodes.modify idx fun nd => { nd with op := newOp, inputs := #[newInId] })[k]'hkMod = _
      rw [Array.getElem_modify]
    rw [hget] at hv
    by_cases hke : idx = k
    · subst hke
      simp only [↓reduceIte] at hv
      have hvEq : v = newInId := by simpa using hv
      subst hvEq
      exact hnewDef
    · simp only [hke, ↓reduceIte] at hv
      exact hwf.defBeforeUse k hkSz v hv
  · intro k hk o ho d hd
    have hkSz : k < g.nodes.size := by simpa [rewriteUnaryNodeInPlace] using hk
    have ho' : o ∈ (g.nodes[k]'hkSz).outputs.toList := by
      rw [← houtSame k hkSz]; exact ho
    exact hwf.nodeOutputsPositive k hkSz o ho' d hd
  · intro v hv
    change v ∈ g'.allDefinedIds
    rw [hallEq]
    exact hwf.outputsValid v hv

private theorem rewriteUnaryNodeInPlace_preservesSem {Op α : Type} (sem : OpSemantics Op α)
    {g : SSAGraph Op} (hwf : WellFormed g)
    {j idx : Nat} (hj : j < idx) (hidx : idx < g.nodes.size)
    {xId midId outId : ValueId} {midTy outTy : TensorType}
    (hJIn : g.nodes[j].inputs.toList = [xId])
    (hJOut : g.nodes[j].outputs.toList = [(midId, midTy)])
    (hIIn : g.nodes[idx].inputs.toList = [midId])
    (hIOut : g.nodes[idx].outputs.toList = [(outId, outTy)])
    (newOp : Op)
    (hcomp : ∀ (tx tmid tout : TensorVal α),
      sem.evalOp g.nodes[j].op [tx] [midTy] = some [tmid] →
      sem.evalOp g.nodes[idx].op [tmid] [outTy] = some [tout] →
      sem.evalOp newOp [tx] [outTy] = some [tout]) :
    SemRefines sem g (rewriteUnaryNodeInPlace g idx newOp xId) := by
  intro env outs hout
  unfold evalGraphOutputs evalGraph at hout ⊢
  cases hInMatch : inputShapesMatch env g.inputs.toList with
  | false => simp [hInMatch] at hout
  | true =>
    simp only [hInMatch, ↓reduceIte] at hout
    simp only [rewriteUnaryNodeInPlace, hInMatch, ↓reduceIte]
    cases hev : evalNodes sem env g.nodes.toList with
    | none => simp [hev] at hout
    | some final1 =>
      simp only [hev] at hout
      -- Split `g.nodes.toList` into `pre ++ [nd] ++ post` around `idx`
      have hsplit : g.nodes.toList =
          g.nodes.toList.take idx ++ [g.nodes[idx]] ++ g.nodes.toList.drop (idx + 1) := by
        have hdrop : g.nodes.toList.drop idx = g.nodes[idx] :: g.nodes.toList.drop (idx + 1) := by
          rw [List.drop_eq_getElem_cons (by simpa using hidx)]; rfl
        calc g.nodes.toList
            = g.nodes.toList.take idx ++ g.nodes.toList.drop idx := (List.take_append_drop idx _).symm
          _ = g.nodes.toList.take idx ++ (g.nodes[idx] :: g.nodes.toList.drop (idx + 1)) := by rw [hdrop]
          _ = g.nodes.toList.take idx ++ [g.nodes[idx]] ++ g.nodes.toList.drop (idx + 1) := by
              simp [List.append_assoc]
      have hsplit' : (g.nodes.modify idx fun nd => { nd with op := newOp, inputs := #[xId] }).toList =
          g.nodes.toList.take idx ++ [{ g.nodes[idx] with op := newOp, inputs := #[xId] }] ++ g.nodes.toList.drop (idx + 1) := by
        apply List.ext_getElem
        · simp [Nat.min_eq_left (Nat.le_of_lt hidx)]; omega
        · intro k hk1 hk2
          have hkSz : k < g.nodes.size := by simpa using hk1
          simp only [Array.getElem_toList, Array.getElem_modify]
          by_cases hlt : k < idx
          · have hne : idx ≠ k := by omega
            simp only [hne, ↓reduceIte]
            rw [List.getElem_append_left (by simp [Nat.min_eq_left (Nat.le_of_lt hidx)]; omega),
                List.getElem_append_left (by simp; omega), List.getElem_take]
            rfl
          · by_cases heq : idx = k
            · subst heq
              simp only [↓reduceIte]
              rw [List.getElem_append_left (by simp [Nat.min_eq_left (Nat.le_of_lt hidx)]),
                  List.getElem_append_right (by simp; omega)]
              simp [Nat.min_eq_left (Nat.le_of_lt hidx)]
            · have hgt : idx < k := by omega
              simp only [heq, ↓reduceIte]
              rw [List.getElem_append_right (by simp [Nat.min_eq_left (Nat.le_of_lt hidx)]; omega)]
              have hlenPre : (g.nodes.toList.take idx ++ [{ g.nodes[idx] with op := newOp, inputs := #[xId] }]).length = idx + 1 := by
                simp [Nat.min_eq_left (Nat.le_of_lt hidx)]
              simp only [hlenPre, List.getElem_drop, Nat.add_sub_cancel' (by omega : idx + 1 ≤ k)]
              rfl
      rw [hsplit, evalNodes_append, evalNodes_append] at hev
      rw [hsplit', evalNodes_append, evalNodes_append]
      cases hevPre : evalNodes sem env (g.nodes.toList.take idx) with
      | none => simp [hevPre] at hev
      | some envIdx =>
        simp only [hevPre] at hev
        -- Extract `tx` and `tmid` by splitting `g.nodes.toList.take idx` at `j < idx`!
        have hsplitPre : g.nodes.toList.take idx =
            g.nodes.toList.take j ++ [g.nodes[j]] ++ (g.nodes.toList.take idx).drop (j + 1) := by
          have hjTake : j < (g.nodes.toList.take idx).length := by simp; omega
          have hdrop : (g.nodes.toList.take idx).drop j = g.nodes[j] :: (g.nodes.toList.take idx).drop (j + 1) := by
            rw [List.drop_eq_getElem_cons hjTake]; simp [List.getElem_take]
          calc g.nodes.toList.take idx
              = (g.nodes.toList.take idx).take j ++ (g.nodes.toList.take idx).drop j := (List.take_append_drop j _).symm
            _ = g.nodes.toList.take j ++ (g.nodes[j] :: (g.nodes.toList.take idx).drop (j + 1)) := by
                rw [List.take_take, Nat.min_eq_left (Nat.le_of_lt hj), hdrop]
            _ = g.nodes.toList.take j ++ [g.nodes[j]] ++ (g.nodes.toList.take idx).drop (j + 1) := by
                simp [List.append_assoc]
        have hevPre2 := hevPre
        rw [hsplitPre, evalNodes_append, evalNodes_append] at hevPre2
        cases hevJPre : evalNodes sem env (g.nodes.toList.take j) with
        | none => simp [hevJPre] at hevPre2
        | some envJ =>
          simp only [hevJPre] at hevPre2
          cases hevJNd : evalNode sem envJ g.nodes[j] with
          | none => simp [evalNodes, hevJNd] at hevPre2
          | some envAfterJ =>
            simp only [evalNodes, hevJNd] at hevPre2
            unfold evalNode at hevJNd
            rw [hJIn, hJOut] at hevJNd
            simp only [List.map_cons, List.map_nil] at hevJNd
            cases hluX : envJ xId with
            | none => simp [lookupAll, hluX] at hevJNd
            | some tx =>
              simp only [lookupAll, hluX] at hevJNd
              cases hopJ : sem.evalOp g.nodes[j].op [tx] [midTy] with
              | none => simp [hopJ] at hevJNd
              | some outJ =>
                simp only [hopJ] at hevJNd
                cases outJ with
                | nil => simp [shapesMatch] at hevJNd
                | cons tmid restJ =>
                  cases restJ with
                  | cons _ _ => simp [shapesMatch] at hevJNd
                  | nil =>
                    split at hevJNd
                    · simp only [Option.some.injEq] at hevJNd
                      subst hevJNd
                      -- Neither `xId` nor `midId` is overwritten between `j` and `idx`!
                      have hnodupJ : (g.allDefinedIdsBefore (j + 1) ++
                          (g.nodes.toList.drop (j + 1)).flatMap IRNode.outputIds).Nodup := by
                        rw [← allDefinedIds_split g (j + 1)]; exact hwf.uniqueIds
                      have hsubMid : ((g.nodes.toList.take idx).drop (j + 1)).Sublist (g.nodes.toList.drop (j + 1)) := by
                        rw [List.drop_take]; exact List.take_sublist _ _
                      have hxBeforeJ : xId ∈ g.allDefinedIdsBefore j :=
                        hwf.defBeforeUse j (by omega) xId (by rw [hJIn]; exact List.mem_cons_self ..)
                      have hnotXMid : xId ∉ ((g.nodes.toList.take idx).drop (j + 1)).flatMap IRNode.outputIds := by
                        intro hmem
                        have hmemD := (sublist_flatMap IRNode.outputIds hsubMid).subset hmem
                        apply nodup_append_disjoint hnodupJ xId _ hmemD
                        unfold SSAGraph.allDefinedIdsBefore at hxBeforeJ ⊢
                        rcases List.mem_append.mp hxBeforeJ with h1 | h2
                        · exact List.mem_append_left _ h1
                        · refine List.mem_append_right _ ?_
                          have hsubT : (g.nodes.toList.take j).Sublist (g.nodes.toList.take (j + 1)) := by
                            have : g.nodes.toList.take j = (g.nodes.toList.take (j + 1)).take j := by
                              rw [List.take_take, Nat.min_eq_left (by omega)]
                            rw [this]; exact List.take_sublist _ _
                          exact (sublist_flatMap IRNode.outputIds hsubT).subset h2
                      have hnotMidMid : midId ∉ ((g.nodes.toList.take idx).drop (j + 1)).flatMap IRNode.outputIds := by
                        intro hmem
                        have hmemD := (sublist_flatMap IRNode.outputIds hsubMid).subset hmem
                        apply nodup_append_disjoint hnodupJ midId _ hmemD
                        unfold SSAGraph.allDefinedIdsBefore
                        refine List.mem_append_right _ ?_
                        exact List.mem_flatMap.mpr ⟨g.nodes[j], by
                          have : (g.nodes.toList.take (j + 1))[j]'(by simp; omega) = g.nodes[j] := by
                            simp [List.getElem_take]
                          rw [← this]; exact List.getElem_mem _, by
                            unfold IRNode.outputIds; rw [hJOut]; exact List.mem_cons_self ..⟩
                      have hxNeMid : xId ≠ midId := by
                        intro heq
                        have hnodupJ0 := hwf.uniqueIds
                        rw [allDefinedIds_split g j] at hnodupJ0
                        have hInDrop : midId ∈ (g.nodes.toList.drop j).flatMap IRNode.outputIds :=
                          List.mem_flatMap.mpr ⟨g.nodes[j], by
                            have : (g.nodes.toList.drop j)[0]'(by simp; omega) = g.nodes[j] := by
                              simp [List.getElem_drop]
                            rw [← this]; exact List.getElem_mem _, by
                              unfold IRNode.outputIds; rw [hJOut]; exact List.mem_cons_self ..⟩
                        exact nodup_append_disjoint hnodupJ0 xId hxBeforeJ (heq ▸ hInDrop)
                      have hEnvX : envIdx xId = some tx := by
                        rw [evalNodes_not_mem_preserved sem _ _ _ xId hnotXMid hevPre2]
                        simp [bindOutputs, Env.set, hxNeMid, hluX]
                      have hEnvMid : envIdx midId = some tmid := by
                        rw [evalNodes_not_mem_preserved sem _ _ _ midId hnotMidMid hevPre2]
                        simp [bindOutputs, Env.set]
                      -- Now show node `idx` in `g'` produces the EXACT SAME `envAfter` as node `idx` in `g`!
                      cases hevI : evalNode sem envIdx g.nodes[idx] with
                      | none => simp [evalNodes, hevI] at hev
                      | some envAfter =>
                        simp only [evalNodes, hevI] at hev
                        unfold evalNode at hevI
                        rw [hIIn, hIOut] at hevI
                        simp only [lookupAll, hEnvMid, List.map_cons, List.map_nil] at hevI
                        cases hopI : sem.evalOp g.nodes[idx].op [tmid] [outTy] with
                        | none => simp [hopI] at hevI
                        | some outI =>
                          simp only [hopI] at hevI
                          cases outI with
                          | nil => simp [shapesMatch] at hevI
                          | cons tout restI =>
                            cases restI with
                            | cons _ _ => simp [shapesMatch] at hevI
                            | nil =>
                              split at hevI
                              · rename_i hsmI
                                simp only [Option.some.injEq] at hevI
                                subst hevI
                                have hopNew := hcomp tx tmid tout hopJ hopI
                                have hevNew : evalNode sem envIdx { g.nodes[idx] with op := newOp, inputs := #[xId] } =
                                    some (bindOutputs envIdx [(outId, outTy)] [tout]) := by
                                  unfold evalNode
                                  rw [hIOut]
                                  simp only [lookupAll, hEnvX, List.map_cons, List.map_nil, hopNew, hsmI, ↓reduceIte]
                                simp only [evalNodes, hevNew, hev]
                                exact hout
                              · cases hevI
                    · cases hevJNd

/-- Forward semantic refinement (`SemRefines`) & well-formedness of `ruleTransposeId`
    (`Transpose(x, idPerm)` with rank matching `inTy` → `x`). -/
theorem evalGraph_viewSimplify_transpose_id {Op α : Type} [HasCoreOps Op]
    (sem : OpSemantics Op α) (hlaw : CoreViewLawful sem) :
    (ruleTransposeId (Op := Op)).PreservesWF ∧ (ruleTransposeId (Op := Op)).PreservesSem sem := by
  constructor
  · intro g g' hwf happ
    unfold ruleTransposeId at happ
    obtain ⟨idx, hidx, hmatch⟩ := findFirstNodeMatch_some happ
    have hidx' : idx < g.nodes.size := by simpa using hidx
    simp only [Nat.zero_add] at hmatch
    have hnd : g.nodes.toList[idx] = g.nodes[idx] := rfl
    rw [hnd] at hmatch
    split at hmatch
    · rename_i perm inId outId outTy hCore hIn hOut
      cases hty : g.valueType? inId with
      | none => simp [hty] at hmatch
      | some inTy =>
        simp only [hty] at hmatch
        split at hmatch
        · simp only [Option.some.injEq] at hmatch
          subst hmatch
          exact eliminateUnaryIdentityNode_preservesWF hwf hidx' hIn hOut
        · cases hmatch
    · cases hmatch
  · intro g g' hwf happ
    unfold ruleTransposeId at happ
    obtain ⟨idx, hidx, hmatch⟩ := findFirstNodeMatch_some happ
    have hidx' : idx < g.nodes.size := by simpa using hidx
    simp only [Nat.zero_add] at hmatch
    have hnd : g.nodes.toList[idx] = g.nodes[idx] := rfl
    rw [hnd] at hmatch
    split at hmatch
    · rename_i perm inId outId outTy hCore hIn hOut
      cases hty : g.valueType? inId with
      | none => simp [hty] at hmatch
      | some inTy =>
        simp only [hty] at hmatch
        split at hmatch
        · rename_i hCond
          simp only [Bool.and_eq_true, beq_iff_eq] at hCond
          obtain ⟨hIdPerm, _⟩ := hCond
          simp only [Option.some.injEq] at hmatch
          subst hmatch
          apply eliminateUnaryIdentityNode_preservesSem sem hwf hidx' hIn hOut
          intro _ t res _ _ hop
          rw [hlaw.transpose_law g.nodes[idx].op perm t outTy hCore] at hop
          simp only [Option.map_eq_some_iff, List.cons.injEq, and_true] at hop
          obtain ⟨r, hr, rfl⟩ := hop
          have hlen : perm.size = t.shape.length ∧ ∀ d ∈ t.shape, d > 0 := by
            unfold evalTranspose at hr
            split at hr
            · rename_i hc
              simp only [Bool.and_eq_true, beq_iff_eq] at hc
              exact ⟨hc.1.2, (list_all_pos_iff t.shape).mp hc.2⟩
            · cases hr
          have hpEq : perm = idPerm t.shape.length := by
            rw [← hlen.1]
            exact (isIdentityPerm_iff perm).mp hIdPerm
          rw [hpEq, evalTranspose_id t hlen.2, Option.some.injEq] at hr
          exact hr.symm
        · cases hmatch
    · cases hmatch

/-- Forward semantic refinement (`SemRefines`) & well-formedness of `ruleTransposeCompose`
    (`Transpose(Transpose(x, p1), p2) → Transpose(x, composePerm p1 p2)`). -/
theorem evalGraph_viewSimplify_transpose_compose {Op α : Type} [HasCoreOps Op]
    (sem : OpSemantics Op α) (hlaw : CoreViewLawful sem) :
    (ruleTransposeCompose (Op := Op)).PreservesWF ∧ (ruleTransposeCompose (Op := Op)).PreservesSem sem := by
  constructor
  · intro g g' hwf happ
    unfold ruleTransposeCompose at happ
    obtain ⟨idx, hidx, hmatch⟩ := findFirstNodeMatch_some happ
    have hidx' : idx < g.nodes.size := by simpa using hidx
    simp only [Nat.zero_add] at hmatch
    have hnd : g.nodes.toList[idx] = g.nodes[idx] := rfl
    rw [hnd] at hmatch
    split at hmatch
    · rename_i p2 midId outId outTy _ _ _
      cases hfp : findSoleProducerBefore g.nodes.toList idx midId with
      | none => simp [hfp] at hmatch
      | some pnd =>
        simp only [hfp] at hmatch
        obtain ⟨j, hjLt, hjLen, _, hgetJ, _⟩ := findSoleProducerBefore_some hfp
        split at hmatch
        · rename_i p1 xId _ hJIn
          split at hmatch
          · simp only [Option.some.injEq] at hmatch
            subst hmatch
            have hxDefJ : xId ∈ g.allDefinedIdsBefore j := by
              have hJIn' : (g.nodes[j]'(by simpa using hjLen)).inputs.toList = [xId] := by
                have : g.nodes[j]'(by simpa using hjLen) = pnd := hgetJ
                rw [this]; exact hJIn
              exact hwf.defBeforeUse j (by simpa using hjLen) xId (by rw [hJIn']; exact List.mem_cons_self ..)
            have hxDefIdx : xId ∈ g.allDefinedIdsBefore idx := by
              unfold SSAGraph.allDefinedIdsBefore at hxDefJ ⊢
              rcases List.mem_append.mp hxDefJ with h1 | h2
              · exact List.mem_append_left _ h1
              · refine List.mem_append_right _ ?_
                have hsubT : (g.nodes.toList.take j).Sublist (g.nodes.toList.take idx) := by
                  have : g.nodes.toList.take j = (g.nodes.toList.take idx).take j := by
                    rw [List.take_take, Nat.min_eq_left (Nat.le_of_lt hjLt)]
                  rw [this]; exact List.take_sublist _ _
                exact (sublist_flatMap IRNode.outputIds hsubT).subset h2
            exact rewriteUnaryNodeInPlace_preservesWF hwf hidx' _ xId hxDefIdx
          · cases hmatch
        · cases hmatch
    · cases hmatch
  · intro g g' hwf happ
    unfold ruleTransposeCompose at happ
    obtain ⟨idx, hidx, hmatch⟩ := findFirstNodeMatch_some happ
    have hidx' : idx < g.nodes.size := by simpa using hidx
    simp only [Nat.zero_add] at hmatch
    have hnd : g.nodes.toList[idx] = g.nodes[idx] := rfl
    rw [hnd] at hmatch
    split at hmatch
    · rename_i p2 midId outId outTy hCore2 hIIn hIOut
      cases hfp : findSoleProducerBefore g.nodes.toList idx midId with
      | none => simp [hfp] at hmatch
      | some pnd =>
        simp only [hfp] at hmatch
        obtain ⟨j, hjLt, hjLen, midTy, hgetJ, hJOut⟩ := findSoleProducerBefore_some hfp
        have hgetJ' : g.nodes[j]'(by simpa using hjLen) = pnd := hgetJ
        split at hmatch
        · rename_i p1 xId hCore1 hJIn
          split at hmatch
          · rename_i hcond
            simp only [Bool.and_eq_true, beq_iff_eq] at hcond
            obtain ⟨⟨hv1, hv2⟩, hsz⟩ := hcond
            have hp1 : ValidPerm p1 := (isValidPerm_iff p1).mp hv1
            have hp2 : ValidPerm p2 := (isValidPerm_iff p2).mp hv2
            simp only [Option.some.injEq] at hmatch
            subst hmatch
            apply rewriteUnaryNodeInPlace_preservesSem sem hwf hjLt hidx'
              (hgetJ'.symm ▸ hJIn) (hgetJ'.symm ▸ hJOut) hIIn hIOut
            intro tx tmid tout hopJ hopI
            rw [hgetJ'] at hopJ
            rw [hlaw.transpose_law pnd.op p1 tx midTy hCore1] at hopJ
            rw [hlaw.transpose_law g.nodes[idx].op p2 tmid outTy hCore2] at hopI
            rw [hlaw.transpose_law _ (composePerm p1 p2) tx outTy (HasCoreOps.asCore_ofCore _)]
            simp only [Option.map_eq_some_iff, List.cons.injEq, and_true] at hopJ hopI ⊢
            obtain ⟨r1, hr1, rfl⟩ := hopJ
            obtain ⟨r2, hr2, rfl⟩ := hopI
            refine ⟨r2, ?_, rfl⟩
            rw [← evalTranspose_compose p1 p2 tx r1 hp1 hp2 hsz hr1]
            exact hr2
          · cases hmatch
        · cases hmatch
    · cases hmatch

/-- Forward semantic refinement (`SemRefines`) & well-formedness of `ruleReshapeId` (`Reshape(x, inShape) → x`). -/
theorem evalGraph_viewSimplify_reshape_id {Op α : Type} [HasCoreOps Op]
    (sem : OpSemantics Op α) (hlaw : CoreViewLawful sem) :
    (ruleReshapeId (Op := Op)).PreservesWF ∧ (ruleReshapeId (Op := Op)).PreservesSem sem := by
  constructor
  · intro g g' hwf happ
    unfold ruleReshapeId at happ
    obtain ⟨idx, hidx, hmatch⟩ := findFirstNodeMatch_some happ
    have hidx' : idx < g.nodes.size := by simpa using hidx
    simp only [Nat.zero_add] at hmatch
    have hnd : g.nodes.toList[idx] = g.nodes[idx] := rfl
    rw [hnd] at hmatch
    split at hmatch
    · rename_i targetShape inId outId outTy _ hIn hOut
      cases hty : g.valueType? inId with
      | none => simp [hty] at hmatch
      | some inTy =>
        simp only [hty] at hmatch
        split at hmatch
        · simp only [Option.some.injEq] at hmatch
          subst hmatch
          exact eliminateUnaryIdentityNode_preservesWF hwf hidx' hIn hOut
        · cases hmatch
    · cases hmatch
  · intro g g' hwf happ
    unfold ruleReshapeId at happ
    obtain ⟨idx, hidx, hmatch⟩ := findFirstNodeMatch_some happ
    have hidx' : idx < g.nodes.size := by simpa using hidx
    simp only [Nat.zero_add] at hmatch
    have hnd : g.nodes.toList[idx] = g.nodes[idx] := rfl
    rw [hnd] at hmatch
    split at hmatch
    · rename_i targetShape inId outId outTy hCore hIn hOut
      cases hty : g.valueType? inId with
      | none => simp [hty] at hmatch
      | some inTy =>
        simp only [hty] at hmatch
        split at hmatch
        · rename_i hshEq
          have htsEq : targetShape = inTy.shape := beq_iff_eq.mp hshEq
          simp only [Option.some.injEq] at hmatch
          subst hmatch
          apply eliminateUnaryIdentityNode_preservesSem sem hwf hidx' hIn hOut
          intro _ t res hshape _ hop
          obtain ⟨htSh, htPos⟩ := hshape inTy hty
          rw [hlaw.reshape_law g.nodes[idx].op targetShape t outTy hCore] at hop
          simp only [Option.map_eq_some_iff, List.cons.injEq, and_true] at hop
          obtain ⟨r, hr, rfl⟩ := hop
          have htarArr : targetShape = t.shape.toArray := by
            rw [htsEq, htSh, Array.toArray_toList]
          rw [htarArr, evalReshape_id t htPos, Option.some.injEq] at hr
          exact hr.symm
        · cases hmatch
    · cases hmatch

/-- Forward semantic refinement (`SemRefines`) & well-formedness of `ruleReshapeCompose`
    (`Reshape(Reshape(x, s1), s2) → Reshape(x, s2)`). -/
theorem evalGraph_viewSimplify_reshape_compose {Op α : Type} [HasCoreOps Op]
    (sem : OpSemantics Op α) (hlaw : CoreViewLawful sem) :
    (ruleReshapeCompose (Op := Op)).PreservesWF ∧ (ruleReshapeCompose (Op := Op)).PreservesSem sem := by
  constructor
  · intro g g' hwf happ
    unfold ruleReshapeCompose at happ
    obtain ⟨idx, hidx, hmatch⟩ := findFirstNodeMatch_some happ
    have hidx' : idx < g.nodes.size := by simpa using hidx
    simp only [Nat.zero_add] at hmatch
    have hnd : g.nodes.toList[idx] = g.nodes[idx] := rfl
    rw [hnd] at hmatch
    split at hmatch
    · rename_i s2 midId outId outTy _ _ _
      cases hfp : findSoleProducerBefore g.nodes.toList idx midId with
      | none => simp [hfp] at hmatch
      | some pnd =>
        simp only [hfp] at hmatch
        obtain ⟨j, hjLt, hjLen, _, hgetJ, _⟩ := findSoleProducerBefore_some hfp
        split at hmatch
        · rename_i s1 xId _ hJIn
          simp only [Option.some.injEq] at hmatch
          subst hmatch
          have hxDefJ : xId ∈ g.allDefinedIdsBefore j := by
            have hJIn' : (g.nodes[j]'(by simpa using hjLen)).inputs.toList = [xId] := by
              have : g.nodes[j]'(by simpa using hjLen) = pnd := hgetJ
              rw [this]; exact hJIn
            exact hwf.defBeforeUse j (by simpa using hjLen) xId (by rw [hJIn']; exact List.mem_cons_self ..)
          have hxDefIdx : xId ∈ g.allDefinedIdsBefore idx := by
            unfold SSAGraph.allDefinedIdsBefore at hxDefJ ⊢
            rcases List.mem_append.mp hxDefJ with h1 | h2
            · exact List.mem_append_left _ h1
            · refine List.mem_append_right _ ?_
              have hsubT : (g.nodes.toList.take j).Sublist (g.nodes.toList.take idx) := by
                have : g.nodes.toList.take j = (g.nodes.toList.take idx).take j := by
                  rw [List.take_take, Nat.min_eq_left (Nat.le_of_lt hjLt)]
                rw [this]; exact List.take_sublist _ _
              exact (sublist_flatMap IRNode.outputIds hsubT).subset h2
          exact rewriteUnaryNodeInPlace_preservesWF hwf hidx' _ xId hxDefIdx
        · cases hmatch
    · cases hmatch
  · intro g g' hwf happ
    unfold ruleReshapeCompose at happ
    obtain ⟨idx, hidx, hmatch⟩ := findFirstNodeMatch_some happ
    have hidx' : idx < g.nodes.size := by simpa using hidx
    simp only [Nat.zero_add] at hmatch
    have hnd : g.nodes.toList[idx] = g.nodes[idx] := rfl
    rw [hnd] at hmatch
    split at hmatch
    · rename_i s2 midId outId outTy hCore2 hIIn hIOut
      cases hfp : findSoleProducerBefore g.nodes.toList idx midId with
      | none => simp [hfp] at hmatch
      | some pnd =>
        simp only [hfp] at hmatch
        obtain ⟨j, hjLt, hjLen, midTy, hgetJ, hJOut⟩ := findSoleProducerBefore_some hfp
        have hgetJ' : g.nodes[j]'(by simpa using hjLen) = pnd := hgetJ
        split at hmatch
        · rename_i s1 xId hCore1 hJIn
          simp only [Option.some.injEq] at hmatch
          subst hmatch
          apply rewriteUnaryNodeInPlace_preservesSem sem hwf hjLt hidx'
            (hgetJ'.symm ▸ hJIn) (hgetJ'.symm ▸ hJOut) hIIn hIOut
          intro tx tmid tout hopJ hopI
          rw [hgetJ'] at hopJ
          rw [hlaw.reshape_law pnd.op s1 tx midTy hCore1] at hopJ
          rw [hlaw.reshape_law g.nodes[idx].op s2 tmid outTy hCore2] at hopI
          rw [hlaw.reshape_law _ s2 tx outTy (HasCoreOps.asCore_ofCore _)]
          simp only [Option.map_eq_some_iff, List.cons.injEq, and_true] at hopJ hopI ⊢
          obtain ⟨r1, hr1, rfl⟩ := hopJ
          obtain ⟨r2, hr2, rfl⟩ := hopI
          refine ⟨r2, ?_, rfl⟩
          rw [← evalReshape_compose s1 s2 tx r1 hr1]
          exact hr2
        · cases hmatch
    · cases hmatch

/-- Forward semantic refinement (`SemRefines`) & well-formedness of `ruleSliceId`
    (`Slice(x, [0..0], inShape, [1..1]) → x`). -/
theorem evalGraph_viewSimplify_slice_id {Op α : Type} [HasCoreOps Op]
    (sem : OpSemantics Op α) (hlaw : CoreViewLawful sem) :
    (ruleSliceId (Op := Op)).PreservesWF ∧ (ruleSliceId (Op := Op)).PreservesSem sem := by
  constructor
  · intro g g' hwf happ
    unfold ruleSliceId at happ
    obtain ⟨idx, hidx, hmatch⟩ := findFirstNodeMatch_some happ
    have hidx' : idx < g.nodes.size := by simpa using hidx
    simp only [Nat.zero_add] at hmatch
    have hnd : g.nodes.toList[idx] = g.nodes[idx] := rfl
    rw [hnd] at hmatch
    split at hmatch
    · rename_i beg en st inId outId outTy _ hIn hOut
      cases hty : g.valueType? inId with
      | none => simp [hty] at hmatch
      | some inTy =>
        simp only [hty] at hmatch
        split at hmatch
        · simp only [Option.some.injEq] at hmatch
          subst hmatch
          exact eliminateUnaryIdentityNode_preservesWF hwf hidx' hIn hOut
        · cases hmatch
    · cases hmatch
  · intro g g' hwf happ
    unfold ruleSliceId at happ
    obtain ⟨idx, hidx, hmatch⟩ := findFirstNodeMatch_some happ
    have hidx' : idx < g.nodes.size := by simpa using hidx
    simp only [Nat.zero_add] at hmatch
    have hnd : g.nodes.toList[idx] = g.nodes[idx] := rfl
    rw [hnd] at hmatch
    split at hmatch
    · rename_i beg en st inId outId outTy hCore hIn hOut
      cases hty : g.valueType? inId with
      | none => simp [hty] at hmatch
      | some inTy =>
        simp only [hty] at hmatch
        split at hmatch
        · rename_i hcond
          simp only [Bool.and_eq_true, beq_iff_eq] at hcond
          obtain ⟨⟨hbeg, hen⟩, hst⟩ := hcond
          simp only [Option.some.injEq] at hmatch
          subst hmatch
          apply eliminateUnaryIdentityNode_preservesSem sem hwf hidx' hIn hOut
          intro _ t res hshape _ hop
          obtain ⟨htSh, htPos⟩ := hshape inTy hty
          rw [hlaw.slice_law g.nodes[idx].op beg en st t outTy hCore] at hop
          simp only [Option.map_eq_some_iff, List.cons.injEq, and_true] at hop
          obtain ⟨r, hr, rfl⟩ := hop
          have hrLen : inTy.shape.size = t.shape.length := by rw [htSh, Array.length_toList]
          have henArr : en = t.shape.toArray := by rw [hen, htSh, Array.toArray_toList]
          rw [hbeg, hst, hrLen, henArr, evalSlice_id t htPos, Option.some.injEq] at hr
          exact hr.symm
        · cases hmatch
    · cases hmatch

/-- Full end-to-end soundness & functional equivalence of the polymorphic `viewSimplify` pass
    (composing `evalGraph_applyRewrites` over all 5 `viewSimplifyRules` + `wellFormed_dce` + `evalGraph_dce`). -/
theorem evalGraph_viewSimplify {Op α : Type} [HasCoreOps Op]
    (sem : OpSemantics Op α) (hlaw : CoreViewLawful sem) (fuel : Nat) :
    PreservesWF (fun g => viewSimplify (Op := Op) g fuel) ∧
    PreservesSem sem (fun g => viewSimplify (Op := Op) g fuel) := by
  have hrules : ∀ r ∈ viewSimplifyRules (Op := Op), r.PreservesWF ∧ r.PreservesSem sem := by
    intro r hr
    unfold viewSimplifyRules at hr
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl | rfl | rfl
    · exact evalGraph_viewSimplify_transpose_id sem hlaw
    · exact evalGraph_viewSimplify_transpose_compose sem hlaw
    · exact evalGraph_viewSimplify_reshape_id sem hlaw
    · exact evalGraph_viewSimplify_reshape_compose sem hlaw
    · exact evalGraph_viewSimplify_slice_id sem hlaw
  constructor
  · intro g hwf
    unfold viewSimplify
    have hwf1 := (evalGraph_applyRewrites sem (if fuel == 0 then 4 * g.nodes.size + 16 else fuel)
      viewSimplifyRules hrules).1 g hwf
    exact wellFormed_dce _ hwf1
  · intro g hwf env outs hout
    unfold viewSimplify
    have hrew := evalGraph_applyRewrites sem (if fuel == 0 then 4 * g.nodes.size + 16 else fuel)
      viewSimplifyRules hrules
    have hwf1 := hrew.1 g hwf
    have hout1 := hrew.2 g hwf env outs hout
    exact evalGraph_dce sem _ hwf1 env outs hout1

/-- `mkCoreSemantics` satisfies `CoreViewLawful` for any compute oracle. -/
theorem mkCoreSemantics_lawful {α : Type}
    (evalCompute : CoreOp → List (TensorVal α) → List TensorType → Option (List (TensorVal α)))
    (hComputeLen : ∀ op ins outTys outs, evalCompute op ins outTys = some outs → outs.length = outTys.length) :
    CoreViewLawful (mkCoreSemantics evalCompute hComputeLen) := by
  constructor
  · intro op perm t outTy h
    have heq : op = .transpose perm := Option.some.inj h
    subst heq; rfl
  · intro op sh t outTy h
    have heq : op = .reshape sh := Option.some.inj h
    subst heq; rfl
  · intro op beg en st t outTy h
    have heq : op = .slice beg en st := Option.some.inj h
    subst heq; rfl
  · intro op ax ts outTy h
    have heq : op = .concat ax := Option.some.inj h
    subst heq; rfl

end Vane.Proofs
