import Vane.Binary
import Vane.TaskStream
import Vane.Register
import Vane.Reach
import Vane.Alloc
import Vane.Verify
import Vane.IR.Graph
import Vane.IR.Validate
import Vane.Machine.Types
import Vane.Backend.SyncAndEmit

namespace Vane.Proofs.C3Sync

open Vane.Binary
open Vane.TaskStream
open Vane.Register
open Vane.Reach
open Vane.Alloc
open Vane.Verify
open Vane.IR
open Vane.Machine
open Vane.Backend.SyncAndEmit

-- ============================================================================
-- Pillar 3: L4 Synchronization Soundness (S1–S5 & Buffer Donation Barriers)
-- ============================================================================

/-- Transitive closure of `stepHB` (definitionally identical to `Vane.Proofs.HappensBefore`). -/
abbrev HB (fps : Array VerifiedTaskFootprint) (k i : Nat) : Prop :=
  Relation.TransGen (fun a b => stepHB fps a b = true) k i

/-- Transitive closure of `stepHBL` (definitionally identical to `Vane.Proofs.HappensBeforeL2`). -/
abbrev HBL (fps : Array VerifiedTaskFootprint) (k i : Nat) : Prop :=
  Relation.TransGen (fun a b => stepHBL fps a b = true) k i

private theorem engine_beq_iff (a b : Engine) : (a == b) = true ↔ a = b := by
  cases a <;> cases b <;> decide

private theorem engine_beq_self (a : Engine) : (a == a) = true := by
  cases a <;> rfl

private theorem dramPort_beq_iff (a b : DramAccessPort) : (a == b) = true ↔ a = b := by
  cases a <;> cases b <;> decide

private theorem dramPort_beq_false_of_ne {a b : DramAccessPort} (h : a ≠ b) : (a == b) = false := by
  cases a <;> cases b <;> first | exact absurd rfl h | rfl

private theorem l2Role_beq_iff (a b : L2AccessRole) : (a == b) = true ↔ a = b := by
  cases a <;> cases b <;> decide

-- ============================================================================
-- 3A. Program-Order Chains & Wait-Bit Happens-Before Theorems
-- ============================================================================

/-- `vLatestEngBefore` always returns an index strictly less than `i` whose engine matches `eng`. -/
theorem vLatestEngBefore_spec (fps : Array VerifiedTaskFootprint) (eng : Engine) :
    ∀ i j, vLatestEngBefore fps eng i = some j →
      j < i ∧ j < fps.size ∧ fps[j]!.engine = eng := by
  intro i
  induction i with
  | zero => intro j h; cases h
  | succ m ih =>
    intro j h
    unfold vLatestEngBefore at h
    cases hGet : fps[m]? with
    | none =>
      simp only [hGet] at h
      obtain ⟨h1, h2, h3⟩ := ih j h
      exact ⟨by omega, h2, h3⟩
    | some fp =>
      simp only [hGet] at h
      split at h
      · rename_i hEng
        simp only [Option.some.injEq] at h
        subst h
        rw [Array.getElem?_eq_some_iff] at hGet
        obtain ⟨hm, hEq⟩ := hGet
        refine ⟨Nat.lt_succ_self _, hm, ?_⟩
        rw [getElem!_pos fps m hm, hEq]
        exact (engine_beq_iff _ _).mp hEng
      · obtain ⟨h1, h2, h3⟩ := ih j h
        exact ⟨by omega, h2, h3⟩

/-- If there is any task `k < i` with engine `eng`, then `vLatestEngBefore fps eng i = some j`
    for some `j` with `k ≤ j < i`, `j < fps.size`, and `fps[j]!.engine = eng`. -/
theorem vLatestEngBefore_ge_of_mem (fps : Array VerifiedTaskFootprint) (eng : Engine) :
    ∀ i k, k < i → k < fps.size → fps[k]!.engine = eng →
      ∃ j, vLatestEngBefore fps eng i = some j ∧ k ≤ j ∧ j < i ∧ j < fps.size ∧ fps[j]!.engine = eng := by
  intro i
  induction i with
  | zero => intro k hk; omega
  | succ m ih =>
    intro k hk hkSz hkEng
    unfold vLatestEngBefore
    by_cases hkm : k = m
    · subst hkm
      have hGet : fps[k]? = some fps[k] := Array.getElem?_eq_getElem hkSz
      rw [hGet]
      have hEngEq : fps[k].engine = eng := by
        simpa [getElem!_pos fps k hkSz] using hkEng
      have hEngB : (fps[k].engine == eng) = true := (engine_beq_iff _ _).mpr hEngEq
      simp only [hEngB, ↓reduceIte]
      exact ⟨k, rfl, Nat.le_refl _, Nat.lt_succ_self _, hkSz, hkEng⟩
    · have hkLt : k < m := by omega
      cases hGet : fps[m]? with
      | none =>
        obtain ⟨j, hj, hkj, hjm, hjSz, hjEng⟩ := ih k hkLt hkSz hkEng
        exact ⟨j, hj, hkj, by omega, hjSz, hjEng⟩
      | some fp =>
        by_cases hEng : (fp.engine == eng) = true
        · simp only [hEng, ↓reduceIte]
          rw [Array.getElem?_eq_some_iff] at hGet
          obtain ⟨hmSz, hEq⟩ := hGet
          refine ⟨m, rfl, Nat.le_of_lt hkLt, Nat.lt_succ_self _, hmSz, ?_⟩
          rw [getElem!_pos fps m hmSz, hEq]
          exact (engine_beq_iff _ _).mp hEng
        · have hEngF : (fp.engine == eng) = false := by cases hb : (fp.engine == eng) <;> simp_all
          simp only [hEngF, Bool.false_eq_true, ↓reduceIte]
          obtain ⟨j, hj, hkj, hjm, hjSz, hjEng⟩ := ih k hkLt hkSz hkEng
          exact ⟨j, hj, hkj, by omega, hjSz, hjEng⟩

private theorem stepHB_imp_stepHBL_local (fps : Array VerifiedTaskFootprint) (j i : Nat)
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
  rw [hnot]; simp [h]

/-- Whole-task `HB` implies L2-level `HBL`. -/
theorem hb_imp_hbl (fps : Array VerifiedTaskFootprint) {k i : Nat} (h : HB fps k i) : HBL fps k i := by
  induction h with
  | single hs => exact Relation.TransGen.single (stepHB_imp_stepHBL_local fps _ _ hs)
  | tail _ hs ih => exact Relation.TransGen.tail ih (stepHB_imp_stepHBL_local fps _ _ hs)

/-- PE program-order chain: for ANY two PE tasks `k < j < fps.size`, `HB fps k j` holds unconditionally! -/
theorem pe_programOrder_chain (fps : Array VerifiedTaskFootprint) :
    ∀ j k, k < j → j < fps.size →
      fps[k]!.engine = .pe → fps[j]!.engine = .pe → HB fps k j := by
  intro j
  induction j using Nat.strongRecOn with
  | ind j ih =>
    intro k hkj hjSz hkPe hjPe
    obtain ⟨p, hpEq, hkp, hpj, hpSz, hpPe⟩ :=
      vLatestEngBefore_ge_of_mem fps .pe j k hkj (Nat.lt_trans hkj hjSz) hkPe
    have hStep : stepHB fps p j = true := by
      unfold stepHB
      have hnot : (!(decide (p < j) && decide (j < fps.size))) = false := by simp [hpj, hjSz]
      rw [hnot]; simp only [Bool.false_eq_true, ↓reduceIte]
      have hpo : (fps[j]!.engine == Engine.pe && vLatestEngBefore fps Engine.pe j == some p) = true := by
        rw [hjPe, hpEq, engine_beq_self]; simp
      simp [hpo]
    by_cases hEq : k = p
    · subst hEq
      exact Relation.TransGen.single hStep
    · have hkpLt : k < p := by omega
      exact Relation.TransGen.tail (ih p hpj k hkpLt hpSz hkPe hpPe) hStep

/-- NE L2 program-order chain: for ANY two NE tasks `k < j < fps.size`, `HBL fps k j` holds unconditionally! -/
theorem ne_l2_programOrder_chain (fps : Array VerifiedTaskFootprint) :
    ∀ j k, k < j → j < fps.size →
      fps[k]!.engine = .ne → fps[j]!.engine = .ne → HBL fps k j := by
  intro j
  induction j using Nat.strongRecOn with
  | ind j ih =>
    intro k hkj hjSz hkNe hjNe
    obtain ⟨p, hpEq, hkp, hpj, hpSz, hpNe⟩ :=
      vLatestEngBefore_ge_of_mem fps .ne j k hkj (Nat.lt_trans hkj hjSz) hkNe
    have hStep : stepHBL fps p j = true := by
      unfold stepHBL
      have hnot : (!(decide (p < j) && decide (j < fps.size))) = false := by simp [hpj, hjSz]
      rw [hnot]; simp only [Bool.false_eq_true, ↓reduceIte]
      have hpo : (fps[j]!.engine == Engine.ne && vLatestEngBefore fps Engine.ne j == some p) = true := by
        rw [hjNe, hpEq, engine_beq_self]; simp
      simp [hpo]
    by_cases hEq : k = p
    · subst hEq
      exact Relation.TransGen.single hStep
    · have hkpLt : k < p := by omega
      exact Relation.TransGen.tail (ih p hpj k hkpLt hpSz hkNe hpNe) hStep

/-- Setting `waitNe5` (on `src1`, or `src2` for PE) establishes a direct whole-task `HB` edge
    from EVERY earlier NE task `k < i`. -/
theorem waitNe5_implies_hb_all_prior_ne (fps : Array VerifiedTaskFootprint) (k i : Nat)
    (hki : k < i) (hiSz : i < fps.size) (hkNe : fps[k]!.engine = .ne)
    (hw5 : fps[i]!.sync.src1WaitNe5 = true ∨ (fps[i]!.engine = .pe ∧ fps[i]!.sync.src2WaitNe5 = true)) :
    HB fps k i := by
  apply Relation.TransGen.single
  unfold stepHB
  have hnot : (!(decide (k < i) && decide (i < fps.size))) = false := by simp [hki, hiSz]
  rw [hnot]; simp only [Bool.false_eq_true, ↓reduceIte]
  have hw : (fps[k]!.engine == Engine.ne && (fps[i]!.sync.src1WaitNe5 || (fps[i]!.engine == Engine.pe && fps[i]!.sync.src2WaitNe5))) = true := by
    rw [hkNe, engine_beq_self]
    rcases hw5 with h1 | ⟨hPe, h2⟩
    · simp [h1]
    · rw [hPe, engine_beq_self, h2]; simp
  simp [hw]

/-- Setting `waitPe22` (on `src1`, or `src2` for PE) establishes whole-task `HB` from
    EVERY earlier PE task `k < i` (via PE program-order chain to `latestPe` + `wPe`). -/
theorem waitPe22_implies_hb_all_prior_pe (fps : Array VerifiedTaskFootprint) (k i : Nat)
    (hki : k < i) (hiSz : i < fps.size) (hkPe : fps[k]!.engine = .pe)
    (hw22 : fps[i]!.sync.src1WaitPe22 = true ∨ (fps[i]!.engine = .pe ∧ fps[i]!.sync.src2WaitPe22 = true)) :
    HB fps k i := by
  obtain ⟨p, hpEq, hkp, hpi, hpSz, hpPe⟩ :=
    vLatestEngBefore_ge_of_mem fps .pe i k hki (Nat.lt_trans hki hiSz) hkPe
  have hStep : stepHB fps p i = true := by
    unfold stepHB
    have hnot : (!(decide (p < i) && decide (i < fps.size))) = false := by simp [hpi, hiSz]
    rw [hnot]; simp only [Bool.false_eq_true, ↓reduceIte]
    have hw : (fps[p]!.engine == Engine.pe && vLatestEngBefore fps Engine.pe i == some p &&
        (fps[i]!.sync.src1WaitPe22 || (fps[i]!.engine == Engine.pe && fps[i]!.sync.src2WaitPe22))) = true := by
      rw [hpPe, hpEq, engine_beq_self]
      rcases hw22 with h1 | ⟨hPe, h2⟩
      · simp [h1]
      · rw [hPe, engine_beq_self, h2]; simp
    simp [hw]
  by_cases hEq : k = p
  · subst hEq
    exact Relation.TransGen.single hStep
  · have hkpLt : k < p := by omega
    exact Relation.TransGen.tail (pe_programOrder_chain fps p k hkpLt hpSz hkPe hpPe) hStep

/-- Setting `waitNe5` also establishes L2-level `HBL` from EVERY earlier NE task `k < i`. -/
theorem waitNe5_implies_hbl_all_prior_ne (fps : Array VerifiedTaskFootprint) (k i : Nat)
    (hki : k < i) (hiSz : i < fps.size) (hkNe : fps[k]!.engine = .ne)
    (hw5 : fps[i]!.sync.src1WaitNe5 = true ∨ (fps[i]!.engine = .pe ∧ fps[i]!.sync.src2WaitNe5 = true)) :
    HBL fps k i :=
  hb_imp_hbl fps (waitNe5_implies_hb_all_prior_ne fps k i hki hiSz hkNe hw5)

/-- Setting `waitPe22` also establishes L2-level `HBL` from EVERY earlier PE task `k < i`. -/
theorem waitPe22_implies_hbl_all_prior_pe (fps : Array VerifiedTaskFootprint) (k i : Nat)
    (hki : k < i) (hiSz : i < fps.size) (hkPe : fps[k]!.engine = .pe)
    (hw22 : fps[i]!.sync.src1WaitPe22 = true ∨ (fps[i]!.engine = .pe ∧ fps[i]!.sync.src2WaitPe22 = true)) :
    HBL fps k i :=
  hb_imp_hbl fps (waitPe22_implies_hb_all_prior_pe fps k i hki hiSz hkPe hw22)

/-- Non-RAW `depKind ∈ {1, 2}` at `depDist = 0` establishes a direct `HB` edge `j → j + 1`. -/
theorem depNonRaw_implies_hb (fps : Array VerifiedTaskFootprint) (j : Nat)
    (hiSz : j + 1 < fps.size)
    (hDma : fps[j + 1]!.sync.hasSrc1Dma = true)
    (hKind : fps[j + 1]!.sync.src1DepKind = 1 ∨ fps[j + 1]!.sync.src1DepKind = 2)
    (hDist : fps[j + 1]!.sync.src1DepDist = 0)
    (hNotRaw : vSrc1DramReadsFromWrite fps[j]! fps[j + 1]! = false) :
    HB fps j (j + 1) := by
  apply Relation.TransGen.single
  unfold stepHB
  have hnot : (!(decide (j < j + 1) && decide (j + 1 < fps.size))) = false := by simp [hiSz]
  rw [hnot]; simp only [Bool.false_eq_true, ↓reduceIte]
  rcases hKind with hk1 | hk2
  · simp [hDma, hk1, hDist, hNotRaw]
  · simp [hDma, hk2, hDist, hNotRaw]

-- ============================================================================
-- 3B. Monotonicity & Redundant-Handoff Cleanup Soundness
-- ============================================================================

/-- Partial order on `TaskSyncInfo` adding whole-task wait bits while preserving DMA ports and dep fields. -/
structure SyncWaitLe (s1 s2 : TaskSyncInfo) : Prop where
  waitNe5 : s1.src1WaitNe5 = true → s2.src1WaitNe5 = true
  waitPe22 : s1.src1WaitPe22 = true → s2.src1WaitPe22 = true
  src2WaitNe5 : s1.src2WaitNe5 = true → s2.src2WaitNe5 = true
  src2WaitPe22 : s1.src2WaitPe22 = true → s2.src2WaitPe22 = true
  sameDma1 : s1.hasSrc1Dma = s2.hasSrc1Dma
  sameKind1 : s1.src1DepKind = s2.src1DepKind
  sameDist1 : s1.src1DepDist = s2.src1DepDist

/-- Adding whole-task wait bits is monotone with respect to `stepHB`! -/
theorem stepHB_mono_waitBits (fps1 fps2 : Array VerifiedTaskFootprint)
    (hSz : fps1.size = fps2.size)
    (hSame : ∀ m (hm : m < fps1.size),
      fps1[m].engine = (fps2[m]'(hSz ▸ hm)).engine ∧
      fps1[m].isNeCompute = (fps2[m]'(hSz ▸ hm)).isNeCompute ∧
      fps1[m].dramAccesses = (fps2[m]'(hSz ▸ hm)).dramAccesses ∧
      SyncWaitLe fps1[m].sync (fps2[m]'(hSz ▸ hm)).sync) :
    ∀ j i, stepHB fps1 j i = true → stepHB fps2 j i = true := by
  have hLat : ∀ eng m, m ≤ fps1.size → vLatestEngBefore fps1 eng m = vLatestEngBefore fps2 eng m := by
    intro eng m
    induction m with
    | zero => intro _; rfl
    | succ r ihr =>
      intro hr
      have hrLt : r < fps1.size := by omega
      unfold vLatestEngBefore
      rw [Array.getElem?_eq_getElem hrLt, Array.getElem?_eq_getElem (hSz ▸ hrLt)]
      dsimp only
      have he := (hSame r hrLt).1
      rw [he, ihr (Nat.le_of_lt hrLt)]
  intro j i h
  unfold stepHB at h ⊢
  by_cases hlt : j < i ∧ i < fps1.size
  · have hlt2 : j < i ∧ i < fps2.size := ⟨hlt.1, hSz ▸ hlt.2⟩
    have hn1 : (!(decide (j < i) && decide (i < fps1.size))) = false := by simp [hlt.1, hlt.2]
    have hn2 : (!(decide (j < i) && decide (i < fps2.size))) = false := by simp [hlt2.1, hlt2.2]
    rw [hn1] at h; rw [hn2]
    simp only [Bool.false_eq_true, ↓reduceIte] at h ⊢
    have hjLt : j < fps1.size := Nat.lt_trans hlt.1 hlt.2
    have hSj := hSame j hjLt
    have hSi := hSame i hlt.2
    have hj1 : fps1[j]! = fps1[j] := getElem!_pos fps1 j hjLt
    have hj2 : fps2[j]! = fps2[j]'(hSz ▸ hjLt) := getElem!_pos fps2 j (hSz ▸ hjLt)
    have hi1 : fps1[i]! = fps1[i] := getElem!_pos fps1 i hlt.2
    have hi2 : fps2[i]! = fps2[i]'(hSz ▸ hlt.2) := getElem!_pos fps2 i (hSz ▸ hlt.2)
    rw [hj1, hi1] at h
    rw [hj2, hi2, ← hLat .pe i (Nat.le_of_lt hlt.2), ← hLat .ne i (Nat.le_of_lt hlt.2)]
    rw [← hSj.1, ← hSi.1, ← hSj.2.1, ← hSi.2.1]
    have hRaw : vSrc1DramReadsFromWrite (fps2[j]'(hSz ▸ hjLt)) (fps2[i]'(hSz ▸ hlt.2)) =
                vSrc1DramReadsFromWrite fps1[j] fps1[i] := by
      unfold vSrc1DramReadsFromWrite
      rw [← hSj.2.2.1, ← hSi.2.2.1]
    have hD1 : (fps2[i]'(hSz ▸ hlt.2)).sync.hasSrc1Dma = fps1[i].sync.hasSrc1Dma := hSi.2.2.2.sameDma1.symm
    have hK1 : (fps2[i]'(hSz ▸ hlt.2)).sync.src1DepKind = fps1[i].sync.src1DepKind := hSi.2.2.2.sameKind1.symm
    have hDi1 : (fps2[i]'(hSz ▸ hlt.2)).sync.src1DepDist = fps1[i].sync.src1DepDist := hSi.2.2.2.sameDist1.symm
    rw [hRaw, hD1, hK1, hDi1]
    simp only [Bool.or_eq_true, Bool.and_eq_true] at h ⊢
    rcases h with (((h1 | h2) | h3) | h4) | h5
    · left; left; left; left; exact h1
    · left; left; left; right; exact h2
    · left; left; right
      refine ⟨h3.1, ?_⟩
      rcases h3.2 with hw1 | ⟨hpe, hw2⟩
      · left; exact hSi.2.2.2.waitNe5 hw1
      · right; exact ⟨hpe, hSi.2.2.2.src2WaitNe5 hw2⟩
    · left; right
      refine ⟨h4.1, ?_⟩
      rcases h4.2 with hw1 | ⟨hpe, hw2⟩
      · left; exact hSi.2.2.2.waitPe22 hw1
      · right; exact ⟨hpe, hSi.2.2.2.src2WaitPe22 hw2⟩
    · right; exact h5
  · have hn1 : (!(decide (j < i) && decide (i < fps1.size))) = true := by
      simp only [Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not]
      by_cases h1 : j < i
      · right; intro h2; exact hlt ⟨h1, h2⟩
      · left; exact h1
    rw [hn1] at h; cases h

/-- Apply `cleanupRedundantHandoffs` to every footprint in `fps`. -/
def cleanupFootprints (fps : Array VerifiedTaskFootprint) : Array VerifiedTaskFootprint :=
  fps.map fun fp => { fp with sync := cleanupRedundantHandoffs fp.sync }

/-- `cleanupRedundantHandoffs` preserves `stepHB` identically (since `stepHB` does not depend on handoff bits). -/
theorem cleanupFootprints_preserves_stepHB (fps : Array VerifiedTaskFootprint) (j i : Nat) :
    stepHB (cleanupFootprints fps) j i = stepHB fps j i := by
  have hSz : (cleanupFootprints fps).size = fps.size := Array.size_map
  have hLat : ∀ eng m, vLatestEngBefore (cleanupFootprints fps) eng m = vLatestEngBefore fps eng m := by
    intro eng m
    induction m with
    | zero => rfl
    | succ r ihr =>
      unfold vLatestEngBefore
      by_cases hr : r < fps.size
      · rw [Array.getElem?_eq_getElem (hSz ▸ hr), Array.getElem?_eq_getElem hr]
        dsimp only
        rw [ihr]
        simp [cleanupFootprints]
      · have h1 : (cleanupFootprints fps)[r]? = none := Array.getElem?_eq_none (by omega)
        have h2 : fps[r]? = none := Array.getElem?_eq_none (by omega)
        rw [h1, h2, ihr]
  unfold stepHB
  rw [hSz]
  by_cases hlt : j < i ∧ i < fps.size
  · have hn : (!(decide (j < i) && decide (i < fps.size))) = false := by simp [hlt.1, hlt.2]
    rw [hn]; simp only [Bool.false_eq_true, ↓reduceIte]
    have hj : j < fps.size := Nat.lt_trans hlt.1 hlt.2
    rw [getElem!_pos (cleanupFootprints fps) j (hSz ▸ hj),
        getElem!_pos (cleanupFootprints fps) i (hSz ▸ hlt.2),
        getElem!_pos fps j hj, getElem!_pos fps i hlt.2]
    rw [hLat .pe i, hLat .ne i]
    simp [cleanupFootprints, cleanupRedundantHandoffs, vSrc1DramReadsFromWrite]
  · have hn : (!(decide (j < i) && decide (i < fps.size))) = true := by
      simp only [Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not]
      by_cases h1 : j < i
      · right; intro h2; exact hlt ⟨h1, h2⟩
      · left; exact h1
    simp [hn]

/-- `cleanupRedundantHandoffs` preserves `stepHBL` identically. -/
theorem cleanupFootprints_preserves_stepHBL (fps : Array VerifiedTaskFootprint) (j i : Nat) :
    stepHBL (cleanupFootprints fps) j i = stepHBL fps j i := by
  have hSz : (cleanupFootprints fps).size = fps.size := Array.size_map
  have hLat : ∀ m, vLatestEngBefore (cleanupFootprints fps) .ne m = vLatestEngBefore fps .ne m := by
    intro m
    induction m with
    | zero => rfl
    | succ r ihr =>
      unfold vLatestEngBefore
      by_cases hr : r < fps.size
      · rw [Array.getElem?_eq_getElem (hSz ▸ hr), Array.getElem?_eq_getElem hr]
        dsimp only
        rw [ihr]
        simp [cleanupFootprints]
      · rw [Array.getElem?_eq_none (by omega), Array.getElem?_eq_none (by omega), ihr]
  unfold stepHBL
  rw [hSz, cleanupFootprints_preserves_stepHB]
  by_cases hlt : j < i ∧ i < fps.size
  · have hn : (!(decide (j < i) && decide (i < fps.size))) = false := by simp [hlt.1, hlt.2]
    rw [hn]; simp only [Bool.false_eq_true, ↓reduceIte]
    rw [getElem!_pos (cleanupFootprints fps) i (hSz ▸ hlt.2), getElem!_pos fps i hlt.2, hLat i]
    simp [cleanupFootprints]
  · have hn : (!(decide (j < i) && decide (i < fps.size))) = true := by
      simp only [Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not]
      by_cases h1 : j < i
      · right; intro h2; exact hlt ⟨h1, h2⟩
      · left; exact h1
    simp [hn]

/-- `cleanupRedundantHandoffs` preserves `HB` and `HBL` relations identically! -/
theorem cleanupFootprints_preserves_hb_hbl (fps : Array VerifiedTaskFootprint) (k i : Nat) :
    (HB (cleanupFootprints fps) k i ↔ HB fps k i) ∧
    (HBL (cleanupFootprints fps) k i ↔ HBL fps k i) := by
  have hHB : (fun a b => stepHB (cleanupFootprints fps) a b = true) = (fun a b => stepHB fps a b = true) := by
    funext a b; rw [cleanupFootprints_preserves_stepHB]
  have hHBL : (fun a b => stepHBL (cleanupFootprints fps) a b = true) = (fun a b => stepHBL fps a b = true) := by
    funext a b; rw [cleanupFootprints_preserves_stepHBL]
  exact ⟨by unfold HB; rw [hHB], by unfold HBL; rw [hHBL]⟩

/-- `cleanupRedundantHandoffs` preserves `buildHBReachTables` identically! -/
theorem cleanupFootprints_preserves_buildHBReachTables (fps : Array VerifiedTaskFootprint) :
    buildHBReachTables (cleanupFootprints fps) = buildHBReachTables fps := by
  have hSz : (cleanupFootprints fps).size = fps.size := Array.size_map
  have hEng : ∀ i (hi : i < fps.size), ((cleanupFootprints fps)[i]'(hSz ▸ hi)).engine = fps[i].engine := by
    intro i hi; simp [cleanupFootprints]
  have hLatAux : ∀ rem i lastNe lastPe neArr peArr,
      rem = fps.size - i → i ≤ fps.size →
      vLatestEngArraysAux (cleanupFootprints fps) (cleanupFootprints fps).size i lastNe lastPe neArr peArr =
      vLatestEngArraysAux fps fps.size i lastNe lastPe neArr peArr := by
    intro rem
    induction rem with
    | zero =>
      intro i lastNe lastPe neArr peArr hrem hi
      have hiEq : i = fps.size := by omega
      subst hiEq
      unfold vLatestEngArraysAux
      rw [hSz]
      simp only [Nat.lt_irrefl, ↓reduceDIte]
    | succ r ihr =>
      intro i lastNe lastPe neArr peArr hrem hi
      have hlt : i < fps.size := by omega
      unfold vLatestEngArraysAux
      rw [hSz]
      simp only [hlt, ↓reduceDIte]
      rw [Array.getElem?_eq_getElem (hSz ▸ hlt), Array.getElem?_eq_getElem hlt]
      dsimp only
      have he : ((cleanupFootprints fps)[i]'(hSz ▸ hlt)).engine = fps[i].engine := hEng i hlt
      rw [he]
      cases fps[i].engine with
      | ne =>
        simpa [hSz] using ihr (i + 1) (some i) lastPe (neArr.push lastNe) (peArr.push lastPe) (by omega) (by omega)
      | pe =>
        simpa [hSz] using ihr (i + 1) lastNe (some i) (neArr.push lastNe) (peArr.push lastPe) (by omega) (by omega)
  have hLatEq : vLatestEngArrays (cleanupFootprints fps) = vLatestEngArrays fps := by
    unfold vLatestEngArrays
    have hE : (Array.mkEmpty (cleanupFootprints fps).size : Array (Option Nat)) = Array.mkEmpty fps.size := by rw [hSz]
    rw [hE]
    exact hLatAux fps.size 0 none none _ _ (by omega) (Nat.zero_le _)
  unfold buildHBReachTables
  rw [hLatEq, hSz]
  have hStepEq : ∀ (a b : Nat → Nat → Bool), (∀ j i, a j i = b j i) → reachTable a fps.size = reachTable b fps.size := by
    intro a b hab; congr 1; funext j i; exact hab j i
  have hMatEq :
      (Array.ofFn (n := fps.size) fun iFin =>
        let i := iFin.val
        let ti := (cleanupFootprints fps)[i]!
        let latestPe := (vLatestEngArrays fps).2[i]!
        let latestNe := (vLatestEngArrays fps).1[i]!
        Array.ofFn (n := i) fun jFin =>
          let j := jFin.val
          let tj := (cleanupFootprints fps)[j]!
          let poPe := ti.engine == .pe && latestPe == some j
          let poNeComp := ti.engine == .ne && tj.engine == .ne && latestNe == some j && tj.isNeCompute && ti.isNeCompute
          let w5 := tj.engine == .ne && (ti.sync.src1WaitNe5 || (ti.engine == .pe && ti.sync.src2WaitNe5))
          let wPe := tj.engine == .pe && latestPe == some j && (ti.sync.src1WaitPe22 || (ti.engine == .pe && ti.sync.src2WaitPe22))
          let depNonRaw :=
            j + 1 == i && ti.sync.hasSrc1Dma &&
            (ti.sync.src1DepKind == 1 || ti.sync.src1DepKind == 2) &&
            ti.sync.src1DepDist == 0 &&
            !vSrc1DramReadsFromWrite tj ti
          poPe || poNeComp || w5 || wPe || depNonRaw) =
      (Array.ofFn (n := fps.size) fun iFin =>
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
          poPe || poNeComp || w5 || wPe || depNonRaw) := by
    apply Array.ext_getElem?
    intro i
    by_cases hi : i < fps.size
    · simp only [Array.getElem?_ofFn, hi, ↓reduceDIte]
      congr 1
      apply Array.ext_getElem?
      intro j
      by_cases hj : j < i
      · simp only [Array.getElem?_ofFn, hj, ↓reduceDIte]
        have hjSz : j < fps.size := Nat.lt_trans hj hi
        rw [getElem!_pos (cleanupFootprints fps) j (hSz ▸ hjSz),
            getElem!_pos (cleanupFootprints fps) i (hSz ▸ hi),
            getElem!_pos fps j hjSz, getElem!_pos fps i hi]
        simp [cleanupFootprints, cleanupRedundantHandoffs, vSrc1DramReadsFromWrite]
      · simp [hj]
    · simp [hi]
  dsimp only
  rw [hMatEq]
  have hHblMatEq :
      (Array.ofFn (n := fps.size) fun iFin =>
        let i := iFin.val
        let ti := (cleanupFootprints fps)[i]!
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
          hbRow[j]! || (ti.engine == .ne && latestNe == some j)) =
      (Array.ofFn (n := fps.size) fun iFin =>
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
          hbRow[j]! || (ti.engine == .ne && latestNe == some j)) := by
    apply Array.ext_getElem?
    intro i
    by_cases hi : i < fps.size
    · simp only [Array.getElem?_ofFn, hi, ↓reduceDIte]
      congr 1
      apply Array.ext_getElem?
      intro j
      by_cases hj : j < i
      · simp only [Array.getElem?_ofFn, hj, ↓reduceDIte]
        rw [getElem!_pos (cleanupFootprints fps) i (hSz ▸ hi), getElem!_pos fps i hi]
        simp [cleanupFootprints]
      · simp [hj]
    · simp [hi]
  rw [hHblMatEq]

-- ============================================================================
-- 3C. Real `syncStateConverged` → S1–S5 & Donation Soundness (Zero Shadow Structs)
-- ============================================================================

/-- `syncAllLt p n = true` iff `p i = true` for every `i < n`. -/
theorem syncAllLt_iff (p : Nat → Bool) (n : Nat) :
    syncAllLt p n = true ↔ ∀ i < n, p i = true := by
  induction n with
  | zero => simp [syncAllLt]
  | succ m ih =>
    simp only [syncAllLt, Bool.and_eq_true, ih]
    constructor
    · intro ⟨hPrev, hM⟩ i hi
      by_cases hEq : i = m
      · subst hEq; exact hM
      · exact hPrev i (by omega)
    · intro hAll
      exact ⟨fun i hi => hAll i (by omega), hAll m (Nat.lt_succ_self _)⟩

/-- `syncDramWantDepKind` ALWAYS chooses legal hardware `depKind ∈ {1, 2}`. -/
theorem syncDramWantDepKind_one_or_two
    (pre : Array PreparedEmitTask) (ti : VerifiedTaskFootprint) (i : Nat) (b : VerifiedDramAccess) :
    let dk := syncDramWantDepKind pre ti i b
    dk = 1 ∨ dk = 2 := by
  unfold syncDramWantDepKind
  split
  · split
    · exact Or.inl rfl
    · exact Or.inr rfl
  · exact Or.inr rfl

/-- Real `syncDramPairSatisfied` (checked inside `computeSynchronization`) implies
    `Vane.Verify.vDramPairOrdered = true` under the exact footprint port invariants:
    - `.kernel` reads are non-writes (`a.isWrite = true` on conflict) with a recorded `extProducerTask`,
    - NE `.src1Dma` reads are non-writes (`a.isWrite = true` on conflict) with `hasSrc1Dma = true`. -/
theorem syncDramPairSatisfied_implies_vDramPairOrdered
    (pre : Array PreparedEmitTask) (fps : Array VerifiedTaskFootprint)
    (hbTbl : Array (Array Bool)) (latestPe : Option Nat)
    (k i : Nat) (a b : VerifiedDramAccess)
    (hSat : syncDramPairSatisfied pre fps hbTbl latestPe k i a b = true)
    (hKern : b.port = .kernel → a.isWrite = true ∧ fps[i]!.extProducerTask.isSome = true)
    (hNeSrc1 : fps[i]!.engine = .ne → b.port = .src1Dma →
      a.isWrite = true ∧ fps[i]!.sync.hasSrc1Dma = true) :
    vDramPairOrdered fps hbTbl latestPe k i a b = true := by
  unfold syncDramPairSatisfied at hSat
  unfold vDramPairOrdered
  by_cases hK : b.port = .kernel
  · have hKb : (b.port == .kernel) = true := (dramPort_beq_iff _ _).mpr hK
    simp only [hKb, ↓reduceIte] at hSat ⊢
    obtain ⟨hAW, hExtSome⟩ := hKern hK
    rw [hAW]; simp only [Bool.true_and]
    cases hExt : fps[i]!.extProducerTask with
    | none => rw [hExt] at hExtSome; cases hExtSome
    | some p => rw [hExt] at hSat; exact hSat
  · have hKf : (b.port == .kernel) = false := dramPort_beq_false_of_ne hK
    simp only [hKf, Bool.false_eq_true, ↓reduceIte] at hSat ⊢
    by_cases hNeS1 : fps[i]!.engine = .ne ∧ b.port = .src1Dma
    · have hCond : (fps[i]!.engine == .ne && b.port == .src1Dma) = true := by
        rw [(engine_beq_iff _ _).mpr hNeS1.1, (dramPort_beq_iff _ _).mpr hNeS1.2]; rfl
      simp only [hCond, ↓reduceIte] at hSat ⊢
      obtain ⟨hAW, hDma⟩ := hNeSrc1 hNeS1.1 hNeS1.2
      simp only [Bool.and_eq_true, beq_iff_eq] at hSat
      have hDk := syncDramWantDepKind_one_or_two pre fps[i]! i b
      have hDk12 : (fps[i]!.sync.src1DepKind == 1 || fps[i]!.sync.src1DepKind == 2) = true := by
        rw [hSat.1.1]
        rcases hDk with h1 | h1 <;> simp [h1]
      rw [hAW]; simp only [Bool.true_and]
      unfold vDramRawKind
      dsimp only
      rw [hDma, hDk12, hSat.1.2]
      simpa using hSat.2
    · have hCondF : (fps[i]!.engine == .ne && b.port == .src1Dma) = false := by
        cases h1 : (fps[i]!.engine == .ne) with
        | false => rfl
        | true =>
          cases h2 : (b.port == .src1Dma) with
          | false => rfl
          | true => exact absurd ⟨(engine_beq_iff _ _).mp h1, (dramPort_beq_iff _ _).mp h2⟩ hNeS1
      simp only [hCondF, Bool.false_eq_true, ↓reduceIte] at hSat ⊢
      exact hSat

/-- Real `syncL2PairSatisfied` (checked inside `computeSynchronization`) is definitionally
    identical to `Vane.Verify.vL2PairOrdered`! -/
theorem syncL2PairSatisfied_eq_vL2PairOrdered
    (fps : Array VerifiedTaskFootprint) (hbTbl hblTbl : Array (Array Bool))
    (latestNe latestPe : Option Nat) (k i : Nat) (a b : VerifiedL2Access) :
    syncL2PairSatisfied fps hbTbl hblTbl latestNe latestPe k i a b =
    vL2PairOrdered fps hbTbl hblTbl latestNe latestPe k i a b :=
  rfl

/-- S1/S2 DRAM & BAR 3 race-freedom directly from `syncStateConverged` (the real fixpoint
    termination predicate of `computeSynchronization`): every conflicting DRAM/scratch pair
    `(a, b)` on `k < i` satisfies `Vane.Verify.vDramPairOrdered = true`! -/
theorem syncStateConverged_s1_s2_dram_ordered
    (pre : Array PreparedEmitTask) (donations : Array BufferDonation) (syncs : Array TaskSyncInfo)
    (hConv : syncStateConverged pre donations syncs = true)
    (i : Nat) (hi : i < pre.size) (k : Nat) (hk : k < i)
    (ia : Nat) (hia : ia < (buildSyncFootprints pre syncs)[k]!.dramAccesses.size)
    (ib : Nat) (hib : ib < (buildSyncFootprints pre syncs)[i]!.dramAccesses.size)
    (hRes : (buildSyncFootprints pre syncs)[k]!.dramAccesses[ia]!.resource =
            (buildSyncFootprints pre syncs)[i]!.dramAccesses[ib]!.resource)
    (hW : (buildSyncFootprints pre syncs)[k]!.dramAccesses[ia]!.isWrite = true ∨
          (buildSyncFootprints pre syncs)[i]!.dramAccesses[ib]!.isWrite = true)
    (hOv : vIntervalsOverlap
      (buildSyncFootprints pre syncs)[k]!.dramAccesses[ia]!.lo
      (buildSyncFootprints pre syncs)[k]!.dramAccesses[ia]!.hi
      (buildSyncFootprints pre syncs)[i]!.dramAccesses[ib]!.lo
      (buildSyncFootprints pre syncs)[i]!.dramAccesses[ib]!.hi = true)
    (hKern : (buildSyncFootprints pre syncs)[i]!.dramAccesses[ib]!.port = .kernel →
      (buildSyncFootprints pre syncs)[k]!.dramAccesses[ia]!.isWrite = true ∧
      (buildSyncFootprints pre syncs)[i]!.extProducerTask.isSome = true)
    (hNeSrc1 : (buildSyncFootprints pre syncs)[i]!.engine = .ne →
      (buildSyncFootprints pre syncs)[i]!.dramAccesses[ib]!.port = .src1Dma →
      (buildSyncFootprints pre syncs)[k]!.dramAccesses[ia]!.isWrite = true ∧
      (buildSyncFootprints pre syncs)[i]!.sync.hasSrc1Dma = true) :
    let fps := buildSyncFootprints pre syncs
    vDramPairOrdered fps (buildHBReachTables fps).1 (vLatestEngArrays fps).2[i]! k i
      fps[k]!.dramAccesses[ia]! fps[i]!.dramAccesses[ib]! = true := by
  unfold syncStateConverged at hConv
  dsimp only at hConv
  simp only [Bool.and_eq_true] at hConv
  have hTasks := (syncAllLt_iff _ _).mp hConv.1
  have hSz : (buildSyncFootprints pre syncs).size = pre.size := by simp [buildSyncFootprints]
  have hiOk := hTasks i (hSz ▸ hi)
  have hiNe0 : (i == 0) = false := by simp; omega
  rw [hiNe0] at hiOk
  simp only [Bool.false_eq_true, ↓reduceIte] at hiOk
  have hkOk := ((syncAllLt_iff _ _).mp hiOk) k hk
  simp only [Bool.and_eq_true] at hkOk
  have hiaOk := ((syncAllLt_iff _ _).mp hkOk.1) ia hia
  have hibOk := ((syncAllLt_iff _ _).mp hiaOk) ib hib
  have hResB : ((buildSyncFootprints pre syncs)[k]!.dramAccesses[ia]!.resource ==
                (buildSyncFootprints pre syncs)[i]!.dramAccesses[ib]!.resource) = true := by
    rw [hRes]
    cases (buildSyncFootprints pre syncs)[i]!.dramAccesses[ib]!.resource with
    | surface idx nm => show (idx == idx && nm == nm) = true; simp
    | scratch => rfl
    | kernel => rfl
  have hWb : ((buildSyncFootprints pre syncs)[k]!.dramAccesses[ia]!.isWrite ||
              (buildSyncFootprints pre syncs)[i]!.dramAccesses[ib]!.isWrite) = true := by
    rcases hW with h1 | h1 <;> simp [h1]
  rw [hResB, hWb, hOv] at hibOk
  simp only [and_self, ↓reduceIte] at hibOk
  exact syncDramPairSatisfied_implies_vDramPairOrdered pre (buildSyncFootprints pre syncs)
    (buildHBReachTables (buildSyncFootprints pre syncs)).1
    (vLatestEngArrays (buildSyncFootprints pre syncs)).2[i]! k i _ _ hibOk hKern hNeSrc1

/-- S1/S2 DRAM WAR/WAW happens-before directly from `syncStateConverged` (NO port preconditions):
    for any DRAM WAR or WAW conflict (`b.isWrite = true`, `b.port = .resDma`),
    `reachB (buildHBReachTables fps).1 k i = true` holds directly! -/
theorem syncStateConverged_s1_s2_dram_war_waw_hb
    (pre : Array PreparedEmitTask) (donations : Array BufferDonation) (syncs : Array TaskSyncInfo)
    (hConv : syncStateConverged pre donations syncs = true)
    (i : Nat) (hi : i < pre.size) (k : Nat) (hk : k < i)
    (ia : Nat) (hia : ia < (buildSyncFootprints pre syncs)[k]!.dramAccesses.size)
    (ib : Nat) (hib : ib < (buildSyncFootprints pre syncs)[i]!.dramAccesses.size)
    (hRes : (buildSyncFootprints pre syncs)[k]!.dramAccesses[ia]!.resource =
            (buildSyncFootprints pre syncs)[i]!.dramAccesses[ib]!.resource)
    (hBW : (buildSyncFootprints pre syncs)[i]!.dramAccesses[ib]!.isWrite = true)
    (hPort : (buildSyncFootprints pre syncs)[i]!.dramAccesses[ib]!.port = .resDma)
    (hOv : vIntervalsOverlap
      (buildSyncFootprints pre syncs)[k]!.dramAccesses[ia]!.lo
      (buildSyncFootprints pre syncs)[k]!.dramAccesses[ia]!.hi
      (buildSyncFootprints pre syncs)[i]!.dramAccesses[ib]!.lo
      (buildSyncFootprints pre syncs)[i]!.dramAccesses[ib]!.hi = true) :
    let fps := buildSyncFootprints pre syncs
    reachB (buildHBReachTables fps).1 k i = true := by
  have hOrd := syncStateConverged_s1_s2_dram_ordered pre donations syncs hConv i hi k hk ia hia ib hib
    hRes (Or.inr hBW) hOv
    (fun h => by rw [hPort] at h; cases h)
    (fun _ h => by rw [hPort] at h; cases h)
  dsimp only at hOrd
  unfold vDramPairOrdered at hOrd
  rw [hPort, hBW] at hOrd
  have h1 : (DramAccessPort.resDma == DramAccessPort.kernel) = false := rfl
  have h2 : (DramAccessPort.resDma == DramAccessPort.src1Dma) = false := rfl
  have h3 : (DramAccessPort.resDma == DramAccessPort.src2Dma) = false := rfl
  simp only [h1, h2, h3, Bool.and_false, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hOrd
  exact hOrd

/-- S3/S4 L2 race-freedom directly from `syncStateConverged`: EVERY conflicting L2 access pair
    `(a, b)` on `k < i` satisfies `Vane.Verify.vL2PairOrdered = true`! -/
theorem syncStateConverged_s3_s4_l2_ordered
    (pre : Array PreparedEmitTask) (donations : Array BufferDonation) (syncs : Array TaskSyncInfo)
    (hConv : syncStateConverged pre donations syncs = true)
    (i : Nat) (hi : i < pre.size) (k : Nat) (hk : k < i)
    (ia : Nat) (hia : ia < (buildSyncFootprints pre syncs)[k]!.l2Accesses.size)
    (ib : Nat) (hib : ib < (buildSyncFootprints pre syncs)[i]!.l2Accesses.size)
    (hW : (buildSyncFootprints pre syncs)[k]!.l2Accesses[ia]!.isWrite = true ∨
          (buildSyncFootprints pre syncs)[i]!.l2Accesses[ib]!.isWrite = true)
    (hOv : vIntervalsOverlap
      (buildSyncFootprints pre syncs)[k]!.l2Accesses[ia]!.lo
      (buildSyncFootprints pre syncs)[k]!.l2Accesses[ia]!.hi
      (buildSyncFootprints pre syncs)[i]!.l2Accesses[ib]!.lo
      (buildSyncFootprints pre syncs)[i]!.l2Accesses[ib]!.hi = true) :
    let fps := buildSyncFootprints pre syncs
    let tbls := buildHBReachTables fps
    let lats := vLatestEngArrays fps
    vL2PairOrdered fps tbls.1 tbls.2 lats.1[i]! lats.2[i]! k i
      fps[k]!.l2Accesses[ia]! fps[i]!.l2Accesses[ib]! = true := by
  unfold syncStateConverged at hConv
  dsimp only at hConv
  simp only [Bool.and_eq_true] at hConv
  have hTasks := (syncAllLt_iff _ _).mp hConv.1
  have hSz : (buildSyncFootprints pre syncs).size = pre.size := by simp [buildSyncFootprints]
  have hiOk := hTasks i (hSz ▸ hi)
  have hiNe0 : (i == 0) = false := by simp; omega
  rw [hiNe0] at hiOk
  simp only [Bool.false_eq_true, ↓reduceIte] at hiOk
  have hkOk := ((syncAllLt_iff _ _).mp hiOk) k hk
  simp only [Bool.and_eq_true] at hkOk
  have hiaOk := ((syncAllLt_iff _ _).mp hkOk.2) ia hia
  have hibOk := ((syncAllLt_iff _ _).mp hiaOk) ib hib
  have hWb : ((buildSyncFootprints pre syncs)[k]!.l2Accesses[ia]!.isWrite ||
              (buildSyncFootprints pre syncs)[i]!.l2Accesses[ib]!.isWrite) = true := by
    rcases hW with h1 | h1 <;> simp [h1]
  rw [hWb, hOv] at hibOk
  simp only [and_self, ↓reduceIte] at hibOk
  exact hibOk

/-- S3/S4 rolling cascade L2 slot-reuse WAR/WAW happens-before directly from `syncStateConverged`:
    whenever task `i` writes an L2-resident slot (`b.role = .resL2`, `b.isWrite = true`)
    overlapping an earlier task `k`'s L2 access `a`, `reachB (buildHBReachTables fps).2 k i = true`! -/
theorem syncStateConverged_s3_s4_l2_slot_reuse_hbl
    (pre : Array PreparedEmitTask) (donations : Array BufferDonation) (syncs : Array TaskSyncInfo)
    (hConv : syncStateConverged pre donations syncs = true)
    (i : Nat) (hi : i < pre.size) (k : Nat) (hk : k < i)
    (ia : Nat) (hia : ia < (buildSyncFootprints pre syncs)[k]!.l2Accesses.size)
    (ib : Nat) (hib : ib < (buildSyncFootprints pre syncs)[i]!.l2Accesses.size)
    (hResL2 : (buildSyncFootprints pre syncs)[i]!.l2Accesses[ib]!.role = .resL2)
    (hBW : (buildSyncFootprints pre syncs)[i]!.l2Accesses[ib]!.isWrite = true)
    (hOv : vIntervalsOverlap
      (buildSyncFootprints pre syncs)[k]!.l2Accesses[ia]!.lo
      (buildSyncFootprints pre syncs)[k]!.l2Accesses[ia]!.hi
      (buildSyncFootprints pre syncs)[i]!.l2Accesses[ib]!.lo
      (buildSyncFootprints pre syncs)[i]!.l2Accesses[ib]!.hi = true) :
    let fps := buildSyncFootprints pre syncs
    reachB (buildHBReachTables fps).2 k i = true := by
  have hOrd := syncStateConverged_s3_s4_l2_ordered pre donations syncs hConv i hi k hk ia hia ib hib
    (Or.inr hBW) hOv
  dsimp only at hOrd
  unfold vL2PairOrdered at hOrd
  rw [hResL2] at hOrd
  have hF1 : (L2AccessRole.resL2 == L2AccessRole.src2Stage) = false := rfl
  have hF2 : (L2AccessRole.resL2 == L2AccessRole.src2L2) = false := rfl
  have hF3 : (L2AccessRole.resL2 == L2AccessRole.src1Stage) = false := rfl
  have hF4 : (L2AccessRole.resL2 == L2AccessRole.src1L2) = false := rfl
  simp only [hF1, hF2, hF3, hF4, Bool.or_self, Bool.and_false, Bool.false_and,
             Bool.false_eq_true, ↓reduceIte, Bool.or_false] at hOrd
  exact hOrd

/-- Donation barrier soundness directly from `syncStateConverged`: EVERY active `BufferDonation`
    in `donations` satisfies `reachB (buildHBReachTables fps).1 d.lastReadTask d.firstWriteTask = true`! -/
theorem syncStateConverged_donation_hb
    (pre : Array PreparedEmitTask) (donations : Array BufferDonation) (syncs : Array TaskSyncInfo)
    (hConv : syncStateConverged pre donations syncs = true)
    (di : Nat) (hdi : di < donations.size)
    (hOrd : donations[di].lastReadTask < donations[di].firstWriteTask)
    (hBound : donations[di].firstWriteTask < pre.size) :
    let fps := buildSyncFootprints pre syncs
    reachB (buildHBReachTables fps).1 donations[di].lastReadTask donations[di].firstWriteTask = true := by
  unfold syncStateConverged at hConv
  dsimp only at hConv
  simp only [Bool.and_eq_true] at hConv
  have hDons := (syncAllLt_iff _ _).mp hConv.2 di hdi
  have hSz : (buildSyncFootprints pre syncs).size = pre.size := by simp [buildSyncFootprints]
  rw [getElem!_pos donations di hdi, hSz] at hDons
  simp only [hBound, hOrd, decide_true, and_self, ↓reduceIte, syncDonationSatisfied] at hDons
  exact hDons

-- ============================================================================
-- 3D. S5 BAR 3 Coverage, No Uninitialized Reads, liveL2Mask & Monotone Progress
-- ============================================================================

private theorem sweepCover_ge_init (ivs : List (Nat × Nat)) (cur : Nat) :
    cur ≤ sweepCover ivs cur := by
  induction ivs generalizing cur with
  | nil => exact Nat.le_refl _
  | cons iv rest ih =>
    simp only [sweepCover]
    split
    · exact Nat.le_trans (Nat.le_max_left _ _) (ih _)
    · exact Nat.le_refl _

private theorem mem_insertInterval (x y : Nat × Nat) (xs : List (Nat × Nat)) :
    x ∈ insertInterval y xs ↔ x = y ∨ x ∈ xs := by
  induction xs with
  | nil => simp [insertInterval]
  | cons hd tl ih =>
    unfold insertInterval
    split
    · simp
    · simp only [List.mem_cons, ih]
      constructor
      · intro h; rcases h with rfl | rfl | hmem <;> simp [*]
      · intro h; rcases h with rfl | rfl | hmem <;> simp [*]

private theorem mem_sortIntervals (x : Nat × Nat) (xs : List (Nat × Nat)) :
    x ∈ sortIntervals xs ↔ x ∈ xs := by
  induction xs with
  | nil => simp [sortIntervals]
  | cons hd tl ih =>
    simp only [sortIntervals, mem_insertInterval, ih, List.mem_cons]

private theorem insertInterval_sorted (y : Nat × Nat) (xs : List (Nat × Nat))
    (hs : xs.Pairwise (fun a b => a.1 ≤ b.1)) :
    (insertInterval y xs).Pairwise (fun a b => a.1 ≤ b.1) := by
  induction xs with
  | nil => simp [insertInterval]
  | cons hd tl ih =>
    have ⟨hHd, hTl⟩ := List.pairwise_cons.mp hs
    unfold insertInterval
    split
    · rename_i hle
      apply List.Pairwise.cons
      · intro z hz
        rcases List.mem_cons.mp hz with rfl | hmem
        · exact hle
        · exact Nat.le_trans hle (hHd z hmem)
      · exact hs
    · rename_i hgt
      apply List.Pairwise.cons
      · intro z hz
        rw [mem_insertInterval] at hz
        rcases hz with rfl | hmem
        · omega
        · exact hHd z hmem
      · exact ih hTl

private theorem sortIntervals_sorted (xs : List (Nat × Nat)) :
    (sortIntervals xs).Pairwise (fun a b => a.1 ≤ b.1) := by
  induction xs with
  | nil => exact List.Pairwise.nil
  | cons hd tl ih => exact insertInterval_sorted hd (sortIntervals tl) ih

private theorem sweepCover_ge_of_mem_sorted (ivs : List (Nat × Nat))
    (hs : ivs.Pairwise (fun a b => a.1 ≤ b.1)) (wLo wHi cur : Nat)
    (hMem : (wLo, wHi) ∈ ivs) (hLo : wLo ≤ cur) : wHi ≤ sweepCover ivs cur := by
  induction ivs generalizing cur with
  | nil => cases hMem
  | cons iv rest ih =>
    have ⟨hHd, hRest⟩ := List.pairwise_cons.mp hs
    simp only [sweepCover]
    rcases List.mem_cons.mp hMem with rfl | htl
    · simp only [hLo, ↓reduceIte]
      exact Nat.le_trans (Nat.le_max_right _ _) (sweepCover_ge_init rest _)
    · have hivLe : iv.1 ≤ cur := Nat.le_trans (hHd (wLo, wHi) htl) hLo
      simp only [hivLe, ↓reduceIte]
      exact ih hRest (max cur iv.2) htl (Nat.le_trans hLo (Nat.le_max_left _ _))

/-- S5 BAR 3 dense coverage soundness: if ANY prior dense scratch write `[wLo, wHi)` covers
    the read interval `[lo, hi)` (`wLo ≤ lo ∧ hi ≤ wHi`), then `Vane.Verify.intervalsCover`
    returns `true`! -/
theorem dense_writer_covers_scratch_read (ivs : List (Nat × Nat)) (wLo wHi lo hi : Nat)
    (hMem : (wLo, wHi) ∈ ivs) (hLo : wLo ≤ lo) (hHi : hi ≤ wHi) :
    intervalsCover ivs lo hi = true := by
  unfold intervalsCover
  split
  · rfl
  · simp only [decide_eq_true_eq]
    have hInSorted : (wLo, wHi) ∈ sortIntervals ivs := (mem_sortIntervals _ _).mpr hMem
    exact Nat.le_trans hHi (sweepCover_ge_of_mem_sorted (sortIntervals ivs) (sortIntervals_sorted ivs) wLo wHi lo hInSorted hLo)

/-- No uninitialized reads in any `WellFormed` SSA graph: every input `v` of node `j` is EITHER
    a graph input surface (`v ∈ g.inputIds`) OR an output of a strictly earlier node `i < j`! -/
theorem no_uninitialized_ssa_reads {Op : Type} {g : SSAGraph Op} (hwf : WellFormed g)
    (j : Nat) (hj : j < g.nodes.size) (v : ValueId) (hv : v ∈ g.nodes[j].inputs.toList) :
    v ∈ g.inputIds ∨ ∃ i : Nat, ∃ hi : i < j, v ∈ (g.nodes[i]'(Nat.lt_trans hi hj)).outputIds := by
  have hDef := hwf.defBeforeUse j hj v hv
  unfold SSAGraph.allDefinedIdsBefore at hDef
  rcases List.mem_append.mp hDef with hIn | hNd
  · exact Or.inl hIn
  · right
    rw [List.mem_flatMap] at hNd
    obtain ⟨nd, hnd, hvNd⟩ := hNd
    rw [List.mem_iff_getElem] at hnd
    obtain ⟨i, hiLen, hndEq⟩ := hnd
    have hiTake : i < j := by
      simp only [List.length_take, Array.length_toList] at hiLen
      omega
    have hEqNd : g.nodes[i]'(Nat.lt_trans hiTake hj) = nd := by
      rw [← hndEq, List.getElem_take]
      rfl
    exact ⟨i, hiTake, hEqNd ▸ hvNd⟩

/-- `header[7]` (`liveL2Mask`) bitmask computation is order-independent (associative, commutative,
    idempotent under bitwise OR) and depends solely on the set of live L2 ranges at task `t`. -/
theorem l2BlockMask_foldl_or_set_invariant (rs1 rs2 : List Range)
    (hSame : ∀ r, r ∈ rs1 ↔ r ∈ rs2) :
    rs1.foldl (fun acc r => acc ||| l2BlockMask r.base r.hi) (0 : UInt32) =
    rs2.foldl (fun acc r => acc ||| l2BlockMask r.base r.hi) (0 : UInt32) := by
  have hBitFold : ∀ (rs : List Range) (init : UInt32) (b : Nat) (hb : b < 32),
      (rs.foldl (fun acc r => acc ||| l2BlockMask r.base r.hi) init).toBitVec[b] =
      (init.toBitVec[b] || rs.any (fun r => (l2BlockMask r.base r.hi).toBitVec[b])) := by
    intro rs
    induction rs with
    | nil => intro init b hb; simp
    | cons r rest ih =>
      intro init b hb
      simp only [List.foldl_cons, List.any_cons]
      rw [ih (init ||| l2BlockMask r.base r.hi) b hb]
      simp [UInt32.toBitVec_or, BitVec.getElem_or, Bool.or_assoc]
  apply UInt32.eq_of_toBitVec_eq
  apply BitVec.eq_of_getElem_eq
  intro b hb
  rw [hBitFold rs1 0 b hb, hBitFold rs2 0 b hb]
  simp only [UInt32.toBitVec_ofNat, BitVec.getElem_zero, Bool.false_or]
  cases h1 : rs1.any (fun r => (l2BlockMask r.base r.hi).toBitVec[b]) with
  | true =>
    rw [List.any_eq_true] at h1
    obtain ⟨r, hr1, hbit⟩ := h1
    symm
    rw [List.any_eq_true]
    exact ⟨r, (hSame r).mp hr1, hbit⟩
  | false =>
    symm
    rw [List.any_eq_false]
    intro r hr2
    have hr1 := (hSame r).mpr hr2
    rw [List.any_eq_false] at h1
    exact h1 r hr1

/-- Number of set whole-task wait bits on a `TaskSyncInfo` (monotone progress measure in `[0, 4]`). -/
def syncWaitMeasure (s : TaskSyncInfo) : Nat :=
  (if s.src1WaitNe5 then 1 else 0) +
  (if s.src1WaitPe22 then 1 else 0) +
  (if s.src2WaitNe5 then 1 else 0) +
  (if s.src2WaitPe22 then 1 else 0)

/-- Every `TaskSyncInfo` has at most 4 whole-task wait bits. -/
theorem syncWaitMeasure_le_4 (s : TaskSyncInfo) : syncWaitMeasure s ≤ 4 := by
  unfold syncWaitMeasure
  cases s.src1WaitNe5 <;> cases s.src1WaitPe22 <;> cases s.src2WaitNe5 <;> cases s.src2WaitPe22 <;> decide

/-- `syncSetSrc1Wait` (called directly by `syncStep`) is monotone in `SyncWaitLe` (preserving
    all prior `stepHB` edges), strictly increases `syncWaitMeasure` by `+1` whenever it reports
    `changed = true`, and leaves `s` unchanged when `changed = false`. -/
theorem syncSetSrc1Wait_monotone_progress (s : TaskSyncInfo) (eng : Engine) :
    let (s', ch) := syncSetSrc1Wait s eng
    SyncWaitLe s s' ∧
    (ch = true → syncWaitMeasure s' = syncWaitMeasure s + 1) ∧
    (ch = false → s' = s) := by
  cases eng with
  | ne =>
    unfold syncSetSrc1Wait
    cases h : s.src1WaitNe5 with
    | false =>
      dsimp only
      refine ⟨⟨(fun h1 => by rw [h] at h1; cases h1), id, id, id, rfl, rfl, rfl⟩, ?_, (fun hc => by cases hc)⟩
      intro _
      unfold syncWaitMeasure
      simp [h]; omega
    | true =>
      dsimp only
      exact ⟨⟨id, id, id, id, rfl, rfl, rfl⟩, (fun hc => by cases hc), (fun _ => rfl)⟩
  | pe =>
    unfold syncSetSrc1Wait
    cases h : s.src1WaitPe22 with
    | false =>
      dsimp only
      refine ⟨⟨id, (fun h1 => by rw [h] at h1; cases h1), id, id, rfl, rfl, rfl⟩, ?_, (fun hc => by cases hc)⟩
      intro _
      unfold syncWaitMeasure
      simp [h]; omega
    | true =>
      dsimp only
      exact ⟨⟨id, id, id, id, rfl, rfl, rfl⟩, (fun hc => by cases hc), (fun _ => rfl)⟩

/-- `syncSetSrc2Wait` (called directly by `syncStep`) is monotone in `SyncWaitLe`, strictly
    increases `syncWaitMeasure` by `+1` whenever `changed = true`, and is identity otherwise. -/
theorem syncSetSrc2Wait_monotone_progress (s : TaskSyncInfo) (eng : Engine) :
    let (s', ch) := syncSetSrc2Wait s eng
    SyncWaitLe s s' ∧
    (ch = true → syncWaitMeasure s' = syncWaitMeasure s + 1) ∧
    (ch = false → s' = s) := by
  cases eng with
  | ne =>
    unfold syncSetSrc2Wait
    cases h : s.src2WaitNe5 with
    | false =>
      dsimp only
      refine ⟨⟨id, id, (fun h1 => by rw [h] at h1; cases h1), id, rfl, rfl, rfl⟩, ?_, (fun hc => by cases hc)⟩
      intro _
      unfold syncWaitMeasure
      simp [h]; omega
    | true =>
      dsimp only
      exact ⟨⟨id, id, id, id, rfl, rfl, rfl⟩, (fun hc => by cases hc), (fun _ => rfl)⟩
  | pe =>
    unfold syncSetSrc2Wait
    cases h : s.src2WaitPe22 with
    | false =>
      dsimp only
      refine ⟨⟨id, id, id, (fun h1 => by rw [h] at h1; cases h1), rfl, rfl, rfl⟩, ?_, (fun hc => by cases hc)⟩
      intro _
      unfold syncWaitMeasure
      simp [h]
    | true =>
      dsimp only
      exact ⟨⟨id, id, id, id, rfl, rfl, rfl⟩, (fun hc => by cases hc), (fun _ => rfl)⟩

/-- Total whole-task wait-bit measure across an `n`-task `syncs` array is bounded by `4 * n`. -/
theorem syncs_totalWaitMeasure_le_4n (syncs : Array TaskSyncInfo) :
    syncs.toList.foldl (fun acc s => acc + syncWaitMeasure s) 0 ≤ 4 * syncs.size := by
  have hList : ∀ (l : List TaskSyncInfo) (init : Nat),
      l.foldl (fun acc s => acc + syncWaitMeasure s) init ≤ init + 4 * l.length := by
    intro l
    induction l with
    | nil => intro init; simp
    | cons hd tl ih =>
      intro init
      simp only [List.foldl_cons, List.length_cons]
      have h1 := ih (init + syncWaitMeasure hd)
      have h2 := syncWaitMeasure_le_4 hd
      omega
  simpa using hList syncs.toList 0

end Vane.Proofs.C3Sync
