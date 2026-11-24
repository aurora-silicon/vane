import Std.Tactic.BVDecide
import Vane.Binary
import Vane.TaskStream
import Vane.Register
import Vane.Tiling

namespace Vane.Proofs

open Vane.Binary
open Vane.TaskStream
open Vane.Register
open Vane.Tiling

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
-- 2. WordLayout well-formedness & bitfield encode/decode round-trip theorems
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
theorem kernelStageLayout_valid : (validateWordLayout kernelStageLayout).isOk = true := by decide

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
  have hmaskNatLt : f.maskNat < 2 ^ 32 := by
    unfold BitField.maskNat
    rw [Nat.shiftLeft_eq, Nat.shiftLeft_eq, Nat.one_mul]
    have h1 : (2 ^ f.width - 1) * 2 ^ f.lo < 2 ^ f.width * 2 ^ f.lo := by
      have hp : 0 < 2 ^ f.lo := Nat.two_pow_pos _
      have hp2 : 0 < 2 ^ f.width := Nat.two_pow_pos _
      exact Nat.mul_lt_mul_of_pos_right (by omega) hp
    rw [← Nat.pow_add] at h1
    have h2 : 2 ^ (f.width + f.lo) ≤ 2 ^ 32 := Nat.pow_le_pow_right (by omega) (by omega)
    omega
  have htoNatMask : ((((1 : Nat) <<< f.width) - 1).toUInt32).toNat = ((1 : Nat) <<< f.width) - 1 := by
    change (((1 : Nat) <<< f.width) - 1) % 2 ^ 32 = _
    exact Nat.mod_eq_of_lt hmaskLt
  have htoNatLo : f.lo.toUInt32.toNat = f.lo := by
    change f.lo % 2 ^ 32 = f.lo
    exact Nat.mod_eq_of_lt (by omega)
  have htoNatFullMask : f.mask.toNat = f.maskNat := by
    unfold BitField.mask
    change f.maskNat % 2 ^ 32 = f.maskNat
    exact Nat.mod_eq_of_lt hmaskNatLt
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

/-- Concrete round-trip theorems on core machine register layouts. -/
theorem neTile_roundTrip (ocg fat wu : UInt32)
    (hocg : ocg < 8) (hfat : fat < 2) (hwu : wu < 4) :
    let w : UInt32 := ocg ||| (fat <<< 3) ||| (wu <<< 4)
    decodeField w { name := "ocg", lo := 0, width := 3 } = ocg ∧
    decodeField w { name := "fat", lo := 3, width := 1 } = fat ∧
    decodeField w { name := "WU", lo := 4, width := 2 } = wu := by
  intro w
  refine ⟨?_, ?_, ?_⟩
  · change (w >>> 0) &&& 7 = ocg; bv_decide
  · change (w >>> 3) &&& 1 = fat; bv_decide
  · change (w >>> 4) &&& 3 = wu; bv_decide

theorem neOp_roundTrip (bias relu : UInt32)
    (hbias : bias < 2) (hrelu : relu < 2) :
    let w : UInt32 := (0x00103c00 : UInt32) ||| (bias <<< 4) ||| (relu <<< 16)
    decodeField w { name := "biasInKernel", lo := 4, width := 1 } = bias ∧
    decodeField w { name := "relu", lo := 16, width := 1 } = relu := by
  intro w
  refine ⟨?_, ?_⟩
  · change (w >>> 4) &&& 1 = bias; bv_decide
  · change (w >>> 16) &&& 1 = relu; bv_decide

theorem resCfg_roundTrip (mode direct p : UInt32)
    (hmode : mode < 4) (hdir : direct < 2) (hp : p < 16) :
    let w : UInt32 := (0x00500070 : UInt32) ||| mode ||| (direct <<< 3) ||| (p <<< 8)
    decodeField w { name := "mode", lo := 0, width := 2 } = mode ∧
    decodeField w { name := "direct", lo := 3, width := 1 } = direct ∧
    decodeField w { name := "P", lo := 8, width := 4 } = p := by
  intro w
  refine ⟨?_, ?_, ?_⟩
  · change (w >>> 0) &&& 3 = mode; bv_decide
  · change (w >>> 3) &&& 1 = direct; bv_decide
  · change (w >>> 8) &&& 15 = p; bv_decide

theorem kernelStage_roundTrip (sparse stage : UInt32)
    (hsp : sparse < 2) (hst : stage < 256) :
    let w : UInt32 := (0x000002c0 : UInt32) ||| (sparse <<< 5) ||| (stage <<< 16)
    decodeField w { name := "sparse", lo := 5, width := 1 } = sparse ∧
    decodeField w { name := "stage", lo := 16, width := 8 } = stage := by
  intro w
  refine ⟨?_, ?_⟩
  · change (w >>> 5) &&& 1 = sparse; bv_decide
  · change (w >>> 16) &&& 255 = stage; bv_decide

theorem taskHeader0_roundTrip (taskId taskWords : UInt32)
    (hid : taskId < 65536) (hw : taskWords < 2048) :
    let w : UInt32 := taskId ||| (taskWords <<< 16)
    decodeField w { name := "taskId", lo := 0, width := 16 } = taskId ∧
    decodeField w { name := "taskWords", lo := 16, width := 11 } = taskWords := by
  intro w
  refine ⟨?_, ?_⟩
  · change (w >>> 0) &&& 65535 = taskId; bv_decide
  · change (w >>> 16) &&& 2047 = taskWords; bv_decide

theorem taskHeader8_roundTrip (ene : UInt32) (hene : ene < 8) :
    let w : UInt32 := (0x00000009 : UInt32) ||| (ene <<< 16)
    decodeField w { name := "ene", lo := 16, width := 3 } = ene := by
  intro w
  change (w >>> 16) &&& 7 = ene
  bv_decide

theorem neCommon_roundTrip (kind shPref shMin shMax f wide3x3 : UInt32)
    (hk : kind < 256) (hsp : shPref < 8) (hsmn : shMin < 8)
    (hsmx : shMax < 8) (hf : f < 8) (hw : wide3x3 < 2) :
    let w : UInt32 := kind ||| (shPref <<< 8) ||| (shMin <<< 12) ||| (shMax <<< 16) ||| (f <<< 19) ||| (wide3x3 <<< 27)
    decodeField w { name := "kind", lo := 0, width := 8 } = kind ∧
    decodeField w { name := "shPref", lo := 8, width := 3 } = shPref ∧
    decodeField w { name := "shMin", lo := 12, width := 3 } = shMin ∧
    decodeField w { name := "shMax", lo := 16, width := 3 } = shMax ∧
    decodeField w { name := "F", lo := 19, width := 3 } = f ∧
    decodeField w { name := "wide3x3", lo := 27, width := 1 } = wide3x3 := by
  intro w
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · change (w >>> 0) &&& 255 = kind; bv_decide
  · change (w >>> 8) &&& 7 = shPref; bv_decide
  · change (w >>> 12) &&& 7 = shMin; bv_decide
  · change (w >>> 16) &&& 7 = shMax; bv_decide
  · change (w >>> 19) &&& 7 = f; bv_decide
  · change (w >>> 27) &&& 1 = wide3x3; bv_decide

theorem convCfg_roundTrip (kw kh sx sy pl pt hx vy : UInt32)
    (hkw : kw < 64) (hkh : kh < 128) (hsx : sx < 4) (hsy : sy < 4)
    (hpl : pl < 32) (hpt : pt < 64) (hhx : hx < 4) (hvy : vy < 4) :
    let w : UInt32 := kw ||| (kh <<< 6) ||| (sx <<< 13) ||| (sy <<< 15) ||| (pl <<< 17) ||| (pt <<< 22) ||| (hx <<< 28) ||| (vy <<< 30)
    decodeField w { name := "kw", lo := 0, width := 6 } = kw ∧
    decodeField w { name := "kh", lo := 6, width := 7 } = kh ∧
    decodeField w { name := "sx", lo := 13, width := 2 } = sx ∧
    decodeField w { name := "sy", lo := 15, width := 2 } = sy ∧
    decodeField w { name := "pl", lo := 17, width := 5 } = pl ∧
    decodeField w { name := "pt", lo := 22, width := 6 } = pt ∧
    decodeField w { name := "hx", lo := 28, width := 2 } = hx ∧
    decodeField w { name := "vy", lo := 30, width := 2 } = vy := by
  intro w
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · change (w >>> 0) &&& 63 = kw; bv_decide
  · change (w >>> 6) &&& 127 = kh; bv_decide
  · change (w >>> 13) &&& 3 = sx; bv_decide
  · change (w >>> 15) &&& 3 = sy; bv_decide
  · change (w >>> 17) &&& 31 = pl; bv_decide
  · change (w >>> 22) &&& 63 = pt; bv_decide
  · change (w >>> 28) &&& 3 = hx; bv_decide
  · change (w >>> 30) &&& 3 = vy; bv_decide

-- ============================================================================
-- 3. validateConvSafety soundness bounds
-- ============================================================================

private theorem except_bind_ok {α β : Type} {x : Except String α} {f : α → Except String β} {v : β}
    (h : (x >>= f) = .ok v) : ∃ a, x = .ok a ∧ f a = .ok v := by
  cases x with
  | error e => simp [bind, Except.bind] at h
  | ok a => exact ⟨a, rfl, h⟩

private theorem except_if_err_ok {c : Prop} [Decidable c] {e : String} {k : Except String Unit}
    (h : (if c then Except.error e else k) = .ok ()) : ¬ c ∧ k = .ok () := by
  by_cases hc : c
  · simp [hc] at h
  · simp [hc] at h
    exact ⟨hc, h⟩

/-- Any tiling accepted by `validateConvSafety` satisfies the core hardware safety bounds. -/
theorem validateConvSafety_bounds
    (target : Target) (sh : ConvSpec) (t : Tiling)
    (h : validateConvSafety target sh t = .ok ()) :
    t.logNE ≤ 4 ∧
    t.fat ≤ 1 ∧
    t.f + t.fat ≤ t.logNE ∧
    t.wu ≤ 3 ∧
    t.shPref ≤ t.shMax ∧
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
  obtain ⟨_, h⟩ := except_if_err_ok h
  obtain ⟨hL2, _⟩ := except_if_err_ok h
  exact ⟨by omega, by omega, by omega, by omega, by omega, by omega⟩

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

end Vane.Proofs
