import Vane.Binary
import Vane.TaskStream
import Vane.Hwx
import Vane.Register
import Vane.Program
import Vane.Alloc
import Vane.Tiling
import Vane.View
import Vane.Ane.Types
import Vane.Machine.Types
import Vane.Backend.MemoryPlan

namespace Vane.Proofs.C3MemoryPlan

open Vane.Binary
open Vane.TaskStream
open Vane.Hwx
open Vane.Register
open Vane.Program
open Vane.Alloc
open Vane.Tiling
open Vane.View
open Vane.Ane
open Vane.Machine
open Vane.Backend.MemoryPlan

-- ============================================================================
-- Pillar 2: L3 Memory Planner Soundness
-- ============================================================================

private theorem alignUp_dvd_local (n m : Nat) (hm : m > 0) : alignUp n m % m = 0 := by
  unfold alignUp
  have hm0 : (m == 0) = false := by simp; omega
  rw [hm0]
  simp only [Bool.false_eq_true, ↓reduceIte]
  split
  · rename_i h; simpa using h
  · have hlt : n % m < m := Nat.mod_lt n hm
    have hdiv : m * (n / m) + n % m = n := Nat.div_add_mod n m
    have hmul : (n / m + 1) * m = m * (n / m) + m := by
      rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm]
    have h1 : n + (m - n % m) = (n / m + 1) * m := by omega
    rw [h1, Nat.mul_mod_left]

private theorem alignUp_ge_local (n m : Nat) : n ≤ alignUp n m := by
  unfold alignUp
  split
  · omega
  · dsimp only; split <;> omega

private theorem foldl_min_preserves_local {P : Nat → Prop} (xs : List Nat) (init : Nat)
    (hinit : P init) (hxs : ∀ x ∈ xs, P x) : P (xs.foldl min init) := by
  induction xs generalizing init with
  | nil => exact hinit
  | cons hd tl ih =>
    simp only [List.foldl_cons]
    apply ih
    · by_cases hle : init ≤ hd
      · rwa [Nat.min_eq_left hle]
      · rw [Nat.min_eq_right (Nat.le_of_not_le hle)]
        exact hxs hd List.mem_cons_self
    · intro x hx; exact hxs x (List.mem_cons_of_mem _ hx)

-- ============================================================================
-- 2A. L2 Disjointness, 64-Byte Alignment & Capacity (incl. Rolling Slot Reuse)
-- ============================================================================

/-- Every L2-resident 4D buffer byte size is 64-byte aligned. -/
theorem l2ResidentBytes4D_dvd_64 (s : Shape4D) : l2ResidentBytes4D s % 64 = 0 :=
  alignUp_dvd_local _ 64 (by omega)

/-- Every BAR 3 DRAM scratch 4D buffer byte size is 64-byte aligned. -/
theorem dramScratchBytes4D_dvd_64 (s : Shape4D) : dramScratchBytes4D s % 64 = 0 :=
  alignUp_dvd_local _ 64 (by omega)

/-- `topOf ps` is 64-byte aligned whenever every placed buffer in `ps` has 64B-aligned end offset. -/
theorem topOf_dvd_64 (ps : List PlacedBuf) (hAll : ∀ p ∈ ps, p.hi % 64 = 0) :
    topOf ps % 64 = 0 := by
  induction ps with
  | nil => simp [topOf]
  | cons p rest ih =>
    simp only [topOf]
    have hp := hAll p List.mem_cons_self
    have hr := ih (fun q hq => hAll q (List.mem_cons_of_mem _ hq))
    by_cases hle : p.hi ≤ topOf rest
    · rw [Nat.max_eq_right hle]; exact hr
    · rw [Nat.max_eq_left (Nat.le_of_not_le hle)]; exact hp

/-- `pickFirstFit` returns a 64-byte aligned offset whenever all conflicting buffers have 64B-aligned ends. -/
theorem pickFirstFit_dvd_64 (conf : List PlacedBuf) (bytes : Nat)
    (hConf : ∀ p ∈ conf, p.hi % 64 = 0) :
    pickFirstFit conf bytes % 64 = 0 := by
  unfold pickFirstFit
  apply foldl_min_preserves_local (P := fun off => off % 64 = 0)
  · exact topOf_dvd_64 conf hConf
  · intro x hx
    rw [List.mem_filter] at hx
    rcases List.mem_cons.mp hx.1 with rfl | hmap
    · rfl
    · rw [List.mem_map] at hmap
      obtain ⟨p, hp, rfl⟩ := hmap
      exact hConf p hp

/-- `placeAll` preserves 64-byte alignment of both `base` and `hi` for all placed buffers
    whenever every input `LiveBuf` has 64B-aligned `bytes`. -/
theorem placeAll_dvd_64 (firstFit : Bool) (bs : List LiveBuf) (acc : List PlacedBuf)
    (hBs : ∀ b ∈ bs, b.bytes % 64 = 0)
    (hAcc : ∀ p ∈ acc, p.base % 64 = 0 ∧ p.hi % 64 = 0) :
    ∀ p ∈ placeAll firstFit bs acc, p.base % 64 = 0 ∧ p.hi % 64 = 0 := by
  induction bs generalizing acc with
  | nil => exact hAcc
  | cons b rest ih =>
    unfold placeAll
    apply ih (placeOne firstFit acc b :: acc) (fun x hx => hBs x (List.mem_cons_of_mem _ hx))
    intro p hp
    rcases List.mem_cons.mp hp with rfl | hOld
    · unfold placeOne
      have hb64 : b.bytes % 64 = 0 := hBs b List.mem_cons_self
      have hConf64 : ∀ q ∈ acc.filter (fun q => decide (meets b.first b.last q.first q.last)), q.hi % 64 = 0 := by
        intro q hq
        exact (hAcc q (List.mem_filter.mp hq).1).2
      have hOff64 : (if firstFit then pickFirstFit (acc.filter fun q => decide (meets b.first b.last q.first q.last)) b.bytes
                     else topOf (acc.filter fun q => decide (meets b.first b.last q.first q.last))) % 64 = 0 := by
        cases firstFit
        · exact topOf_dvd_64 _ hConf64
        · exact pickFirstFit_dvd_64 _ _ hConf64
      exact ⟨hOff64, by unfold PlacedBuf.hi; dsimp only; omega⟩
    · exact hAcc p hOld

/-- `Vane.Alloc.allocate` preserves 64-byte alignment of `base` and `hi` whenever every
    input `LiveBuf` has 64B-aligned `bytes` (e.g. `l2ResidentBytes4D` or `dramScratchBytes4D`). -/
theorem allocate_dvd_64 (firstFit : Bool) (bs : List LiveBuf)
    (hBs : ∀ b ∈ bs, b.bytes % 64 = 0) :
    ∀ p ∈ allocate firstFit bs, p.base % 64 = 0 ∧ p.hi % 64 = 0 := by
  unfold allocate
  have hPerm : (bs.mergeSort liveBufBefore).Perm bs := List.mergeSort_perm bs liveBufBefore
  have hSorted : ∀ b ∈ bs.mergeSort liveBufBefore, b.bytes % 64 = 0 :=
    fun b hb => hBs b (hPerm.subset hb)
  intro p hp
  rw [List.mem_reverse] at hp
  exact placeAll_dvd_64 firstFit _ [] hSorted (fun _ h => by cases h) p hp

/-- `Vane.Alloc.allocate` preserves exact buffer count (`(allocate firstFit bs).length = bs.length`). -/
theorem allocate_length (firstFit : Bool) (bs : List LiveBuf) :
    (allocate firstFit bs).length = bs.length := by
  have hPlaceLen : ∀ l acc, (placeAll firstFit l acc).length = l.length + acc.length := by
    intro l
    induction l with
    | nil => intro acc; simp [placeAll]
    | cons x rest ih =>
      intro acc
      simp only [placeAll, ih, List.length_cons]
      omega
  unfold allocate
  rw [List.length_reverse, hPlaceLen, List.length_nil, Nat.add_zero]
  exact (List.mergeSort_perm bs liveBufBefore).length_eq

/-- Whenever `MemoryPlanWitness.l2WithinCapacity` holds, `l2Peak ≤ l2Capacity`,
    `topOf l2Placed ≤ l2Capacity`, and every live L2 buffer at every task `t` satisfies
    `r.hi ≤ w.l2Capacity`. -/
theorem witness_l2WithinCapacity_sound (w : MemoryPlanWitness)
    (hOk : w.l2WithinCapacity = true) :
    w.l2Peak ≤ w.l2Capacity ∧
    topOf w.l2Placed ≤ w.l2Capacity ∧
    ∀ t, ∀ r ∈ liveAt w.l2Placed t, r.hi ≤ w.l2Capacity := by
  unfold MemoryPlanWitness.l2WithinCapacity at hOk
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hOk
  have hLeTop : ∀ ps : List PlacedBuf, ∀ p ∈ ps, p.hi ≤ topOf ps := by
    intro ps
    induction ps with
    | nil => intro _ h; cases h
    | cons hd tl ih =>
      intro p hp
      simp only [topOf]
      rcases List.mem_cons.mp hp with rfl | htl
      · exact Nat.le_max_left _ _
      · exact Nat.le_trans (ih p htl) (Nat.le_max_right _ _)
  refine ⟨hOk.1, hOk.2, ?_⟩
  intro t r hr
  unfold liveAt at hr
  rw [List.mem_map] at hr
  obtain ⟨p, hp, rfl⟩ := hr
  rw [List.mem_filter] at hp
  have h1 : p.hi ≤ topOf w.l2Placed := hLeTop w.l2Placed p hp.1
  exact Nat.le_trans h1 hOk.2

/-- Rolling cascade slot reuse soundness: two buffers with non-overlapping live intervals
    (`p1.last < p2.first`) never conflict and are never simultaneously present in `liveAt ps t`
    for any task `t` (enabling zero-overhead ping-pong L2 slot reuse!). -/
theorem rolling_cascade_nonoverlapping_no_conflict (p1 p2 : PlacedBuf)
    (hDisj : p1.last < p2.first) :
    ¬ p1.conflicts p2 ∧ ∀ t, ¬ (p1.live t ∧ p2.live t) := by
  constructor
  · unfold PlacedBuf.conflicts meets; omega
  · intro t ⟨h1, h2⟩
    unfold PlacedBuf.live at h1 h2
    omega

-- ============================================================================
-- 2B. Zero-Copy Subview Bounds & Packed Multi-Output Disjointness
-- ============================================================================

/-- Every canonical 4D shape stride (`rowStride`, `planeStride`, `batchStride`, `span`) is 64B-aligned. -/
theorem shape4D_strides_dvd_64 (s : Shape4D) :
    s.rowStride % 64 = 0 ∧ s.planeStride % 64 = 0 ∧ s.batchStride % 64 = 0 ∧ s.span % 64 = 0 := by
  have hr : s.rowStride % 64 = 0 := alignUp_dvd_local (2 * s.w) 64 (by omega)
  have hp : s.planeStride % 64 = 0 := by
    unfold Shape4D.planeStride; rw [Nat.mul_mod, hr, Nat.mul_zero, Nat.zero_mod]
  have hb : s.batchStride % 64 = 0 := by
    unfold Shape4D.batchStride; rw [Nat.mul_mod, hp, Nat.mul_zero, Nat.zero_mod]
  have hs : s.span % 64 = 0 := by
    unfold Shape4D.span; rw [Nat.mul_mod, hb, Nat.mul_zero, Nat.zero_mod]
  exact ⟨hr, hp, hb, hs⟩

/-- A zero-copy C-axis subview `[c0, c0 + dc)` (`c0 + dc ≤ parent.c`) of a single-batch 4D buffer
    at `byteOffset = c0 * parent.planeStride` is 64B-aligned and lies strictly within
    `[0, parent.span)`. -/
theorem subview_c_axis_within_parent (parent : Shape4D) (c0 dc : Nat)
    (hn : parent.n = 1) (hBound : c0 + dc ≤ parent.c) :
    let byteOff := c0 * parent.planeStride
    let sliceBytes := dc * parent.planeStride
    byteOff % 64 = 0 ∧ sliceBytes % 64 = 0 ∧ byteOff + sliceBytes ≤ parent.span := by
  have hp64 := (shape4D_strides_dvd_64 parent).2.1
  have hOff64 : (c0 * parent.planeStride) % 64 = 0 := by
    rw [Nat.mul_mod, hp64, Nat.mul_zero, Nat.zero_mod]
  have hSl64 : (dc * parent.planeStride) % 64 = 0 := by
    rw [Nat.mul_mod, hp64, Nat.mul_zero, Nat.zero_mod]
  have hFit : c0 * parent.planeStride + dc * parent.planeStride ≤ parent.span := by
    unfold Shape4D.span Shape4D.batchStride
    rw [hn, Nat.one_mul, ← Nat.add_mul]
    exact Nat.mul_le_mul_right parent.planeStride hBound
  exact ⟨hOff64, hSl64, hFit⟩

/-- `surfaceDmaWindow` row-band subview bounds: for any valid window `win.rowOffset + h ≤ effFullH`
    with `c > 0`, `baseOff`, `rowStride`, `planeStride`, and `extent` are 64B-aligned, and
    `baseOff + extent ≤ c * effFullH * rowStride` (strictly inside the parent surface span!). -/
theorem surfaceDmaWindow_within_parent (win : SurfaceWindow) (c h w : Nat)
    (hc : c > 0)
    (hBound : win.rowOffset + h ≤ (if win.fullH == 0 then h else win.fullH)) :
    let effFullH := if win.fullH == 0 then h else win.fullH
    let (baseOff, rowSt, planeSt, ext) := surfaceDmaWindow win c h w
    baseOff % 64 = 0 ∧ rowSt % 64 = 0 ∧ planeSt % 64 = 0 ∧ ext % 64 = 0 ∧
    baseOff + ext ≤ c * effFullH * rowSt := by
  intro effFullH
  simp only [surfaceDmaWindow]
  have hRow64 : alignUp (2 * w) 64 % 64 = 0 := alignUp_dvd_local (2 * w) 64 (by omega)
  have hPlane64 : (effFullH * alignUp (2 * w) 64) % 64 = 0 := by
    rw [Nat.mul_mod, hRow64, Nat.mul_zero, Nat.zero_mod]
  have hBase64 : (win.rowOffset * alignUp (2 * w) 64) % 64 = 0 := by
    rw [Nat.mul_mod, hRow64, Nat.mul_zero, Nat.zero_mod]
  have hExt64 : (if (c == 0 || h == 0) = true then 0
      else (c - 1) * (effFullH * alignUp (2 * w) 64) + h * alignUp (2 * w) 64) % 64 = 0 := by
    split
    · rfl
    · have h1 : ((c - 1) * (effFullH * alignUp (2 * w) 64)) % 64 = 0 := by
        rw [Nat.mul_mod, hPlane64, Nat.mul_zero, Nat.zero_mod]
      have h2 : (h * alignUp (2 * w) 64) % 64 = 0 := by
        rw [Nat.mul_mod, hRow64, Nat.mul_zero, Nat.zero_mod]
      omega
  refine ⟨hBase64, hRow64, hPlane64, hExt64, ?_⟩
  let rSt := alignUp (2 * w) 64
  change win.rowOffset * rSt + (if (c == 0 || h == 0) = true then 0
    else (c - 1) * (effFullH * rSt) + h * rSt) ≤ c * effFullH * rSt
  split
  · rw [Nat.add_zero]
    have h1 : win.rowOffset ≤ effFullH := by omega
    have h2 : effFullH ≤ c * effFullH := Nat.le_mul_of_pos_left effFullH hc
    exact Nat.mul_le_mul_right rSt (Nat.le_trans h1 h2)
  · have hEq : win.rowOffset * rSt + ((c - 1) * (effFullH * rSt) + h * rSt) =
        ((c - 1) * effFullH + (win.rowOffset + h)) * rSt := by
      rw [Nat.add_mul, Nat.add_mul, Nat.mul_assoc]
      omega
    rw [hEq]
    have hRows : (c - 1) * effFullH + (win.rowOffset + h) ≤ c * effFullH := by
      have hSplit : c * effFullH = (c - 1) * effFullH + effFullH := by
        cases c with
        | zero => omega
        | succ c' => simp [Nat.succ_mul]
      omega
    exact Nat.mul_le_mul_right rSt hRows

/-- `resolveBufferStorageAux` preserves 64B offset alignment through any finite `subview` chain
    whenever every `.subview`, `.l2`, and `.dramScratch` buffer in `buffers` is 64B-aligned. -/
theorem resolveBufferStorageAux_aligned_64 (buffers : Array BufferStorage)
    (hBufs : ∀ b ∈ buffers.toList, match b with
      | .subview _ off _ _ => off % 64 = 0
      | .dramScratch off bytes _ => off % 64 = 0 ∧ bytes % 64 = 0
      | .l2 base bytes _ => base % 64 = 0 ∧ bytes % 64 = 0
      | .surface _ _ _ _ _ => True) :
    ∀ fuel bid accOff win res,
      accOff % 64 = 0 →
      resolveBufferStorageAux buffers fuel bid accOff win = .ok res →
      match res with
      | .surface _ _ _ bOff _ _ => bOff % 64 = 0
      | .l2 base bytes => base % 64 = 0 ∧ bytes % 64 = 0
      | .dramScratch off bytes => off % 64 = 0 ∧ bytes % 64 = 0 := by
  intro fuel
  induction fuel with
  | zero =>
    intro bid accOff win res _ h
    simp [resolveBufferStorageAux] at h
  | succ f ih =>
    intro bid accOff win res hAcc hRes
    unfold resolveBufferStorageAux at hRes
    cases hGet : buffers[bid]? with
    | none => simp [hGet] at hRes
    | some b =>
      have hbMem : b ∈ buffers.toList := by
        rw [Array.getElem?_eq_some_iff] at hGet
        obtain ⟨hlt, heq⟩ := hGet
        rw [← heq]
        exact Array.getElem_mem_toList hlt
      have hbProp := hBufs b hbMem
      rw [hGet] at hRes
      cases b with
      | surface idx dir slot sh sw =>
        dsimp only at hRes
        have hEq := Except.ok.inj hRes
        rw [← hEq]
        exact hAcc
      | l2 base bytes sh =>
        dsimp only at hRes hbProp
        split at hRes
        · cases hRes
        · have hEq := Except.ok.inj hRes
          rw [← hEq]
          exact hbProp
      | dramScratch off bytes sh =>
        dsimp only at hRes hbProp
        have hEq := Except.ok.inj hRes
        rw [← hEq]
        exact ⟨by omega, hbProp.2⟩
      | subview parent subOff sh sw =>
        dsimp only at hRes hbProp
        exact ih parent (accOff + subOff) _ res (by omega) hRes

/-- Byte span of a `BufferStorage` entry (physical size for root buffers, logical span for subviews). -/
def storageByteSpan : BufferStorage → Nat
  | .surface _ _ _ sh _ => sh.span
  | .l2 _ bytes _ => bytes
  | .dramScratch _ bytes _ => bytes
  | .subview _ _ sh _ => sh.span

/-- `resolveBufferStorageAux` byte-range containment: whenever every `.subview parent subOff sh _`
    in `buffers` lies within its immediate parent's byte span (`subOff + sh.span ≤ storageByteSpan p`),
    ANY resolved subview chain lies strictly inside its physical root buffer's `[rootBase, rootBase + rootBytes)`! -/
theorem resolveBufferStorageAux_within_parent (buffers : Array BufferStorage)
    (hSubBounds : ∀ b ∈ buffers.toList, match b with
      | .subview parent subOff sh _ =>
        ∃ p, buffers[parent]? = some p ∧ subOff + sh.span ≤ storageByteSpan p
      | _ => True) :
    ∀ fuel bid accOff win res childBytes,
      (∃ b, buffers[bid]? = some b ∧ accOff + childBytes ≤ storageByteSpan b) →
      resolveBufferStorageAux buffers fuel bid accOff win = .ok res →
      match res with
      | .surface _ _ _ bOff _ rootSh => accOff ≤ bOff ∧ bOff + childBytes ≤ rootSh.span
      | .l2 _ bytes => accOff = 0 ∧ childBytes ≤ bytes
      | .dramScratch off bytes => ∃ rootOff, rootOff + accOff ≤ off ∧ off + childBytes ≤ rootOff + bytes := by
  intro fuel
  induction fuel with
  | zero =>
    intro _ _ _ _ _ _ h
    simp [resolveBufferStorageAux] at h
  | succ f ih =>
    intro bid accOff win res childBytes ⟨b, hGet, hChildLe⟩ hRes
    unfold resolveBufferStorageAux at hRes
    rw [hGet] at hRes
    have hbMem : b ∈ buffers.toList := by
      rw [Array.getElem?_eq_some_iff] at hGet
      obtain ⟨hlt, heq⟩ := hGet
      rw [← heq]
      exact Array.getElem_mem_toList hlt
    cases b with
    | surface idx dir slot sh sw =>
      dsimp only [storageByteSpan] at hChildLe hRes
      have hEq := Except.ok.inj hRes
      rw [← hEq]
      exact ⟨Nat.le_refl _, by omega⟩
    | l2 base bytes sh =>
      dsimp only [storageByteSpan] at hChildLe hRes
      split at hRes
      · cases hRes
      · rename_i hne
        have hZero : accOff = 0 := by simpa [bne_iff_ne] using hne
        have hEq := Except.ok.inj hRes
        rw [← hEq]
        exact ⟨hZero, by omega⟩
    | dramScratch rootOff bytes sh =>
      dsimp only [storageByteSpan] at hChildLe hRes
      have hEq := Except.ok.inj hRes
      rw [← hEq]
      exact ⟨rootOff, Nat.le_refl _, by omega⟩
    | subview parent subOff sh sw =>
      dsimp only [storageByteSpan] at hChildLe hRes
      have hStep := hSubBounds (.subview parent subOff sh sw) hbMem
      dsimp only at hStep
      obtain ⟨p, hpGet, hSubFit⟩ := hStep
      have hRec := ih parent (accOff + subOff) _ res childBytes
        ⟨p, hpGet, by omega⟩ hRes
      cases res with
      | surface idx dir slot bOff effWin rootSh =>
        exact ⟨by omega, hRec.2⟩
      | l2 base bytes =>
        have hSum0 : accOff + subOff = 0 := hRec.1
        exact ⟨by omega, hRec.2⟩
      | dramScratch off bytes =>
        obtain ⟨rootOff, hLe, hHi⟩ := hRec
        exact ⟨rootOff, by omega, hHi⟩

/-- End-to-end subview containment for `resolveBufferStorage`: for any `BufferId` in `buffers`
    whose local subviews fit within their immediate parents, `resolveBufferStorage buffers bid = .ok res`
    guarantees `[bOff, bOff + storageByteSpan buffers[bid])` lies strictly within its physical root buffer! -/
theorem resolveBufferStorage_within_parent (buffers : Array BufferStorage)
    (hSubBounds : ∀ b ∈ buffers.toList, match b with
      | .subview parent subOff sh _ =>
        ∃ p, buffers[parent]? = some p ∧ subOff + sh.span ≤ storageByteSpan p
      | _ => True)
    (bid : BufferId) (hBid : bid < buffers.size) (res : ResolvedStorage)
    (hRes : resolveBufferStorage buffers bid = .ok res) :
    match res with
    | .surface _ _ _ bOff _ rootSh => bOff + storageByteSpan buffers[bid] ≤ rootSh.span
    | .l2 _ bytes => storageByteSpan buffers[bid] ≤ bytes
    | .dramScratch off bytes => ∃ rootOff, rootOff ≤ off ∧ off + storageByteSpan buffers[bid] ≤ rootOff + bytes := by
  have hAux := resolveBufferStorageAux_within_parent buffers hSubBounds
    (buffers.size + 1) bid 0 {} res (storageByteSpan buffers[bid])
    ⟨buffers[bid], Array.getElem?_eq_getElem hBid, by omega⟩ hRes
  cases res with
  | surface _ _ _ _ _ _ => exact hAux.2
  | l2 _ _ => exact hAux.2
  | dramScratch _ _ =>
    obtain ⟨rootOff, hLe, hHi⟩ := hAux
    exact ⟨rootOff, by omega, hHi⟩

/-- Any `.l2` resolution from `resolveBufferStorageAux` forces accumulated subview offset `= 0`. -/
theorem resolveBufferStorageAux_l2_zero_offset (buffers : Array BufferStorage) :
    ∀ fuel bid accOff win base bytes,
      resolveBufferStorageAux buffers fuel bid accOff win = .ok (.l2 base bytes) →
      accOff = 0 := by
  intro fuel
  induction fuel with
  | zero => intro _ _ _ _ _ h; simp [resolveBufferStorageAux] at h
  | succ f ih =>
    intro bid accOff win base bytes h
    unfold resolveBufferStorageAux at h
    cases hGet : buffers[bid]? with
    | none => simp [hGet] at h
    | some b =>
      rw [hGet] at h
      cases b with
      | surface _ _ _ _ _ => dsimp only at h; cases h
      | l2 b0 sz0 _ =>
        dsimp only at h
        split at h
        · cases h
        · rename_i hne
          simp only [bne_iff_ne, Classical.not_not] at hne
          exact hne
      | dramScratch _ _ _ => dsimp only at h; cases h
      | subview parent subOff _ _ =>
        dsimp only at h
        have hSum := ih parent (accOff + subOff) _ base bytes h
        omega

/-- Core invariant of `packOutputSlots.go`:
    every packed `OutputSlot` is 64B-aligned (`byteOffset`, `byteSize`, `rowStride`, `planeStride`,
    `batchStride`), lies strictly inside `[0, totalBytes)`, and all packed slots are pairwise
    disjoint (`slots[i].byteOffset + slots[i].byteSize ≤ slots[j].byteOffset` for `i < j`). -/
private theorem packOutputSlots_go_spec (shapes : Array Shape4D) (names : Array String) :
    ∀ rem i packOffset (acc : Array OutputSlot),
      rem = shapes.size - i →
      i ≤ shapes.size →
      acc.size = i →
      packOffset % 64 = 0 →
      (∀ k (hk : k < acc.size),
        acc[k].byteOffset % 64 = 0 ∧ acc[k].byteSize % 64 = 0 ∧
        acc[k].rowStride % 64 = 0 ∧ acc[k].planeStride % 64 = 0 ∧
        acc[k].batchStride % 64 = 0 ∧
        acc[k].byteOffset + acc[k].byteSize ≤ packOffset) →
      (∀ a b (ha : a < acc.size) (hb : b < acc.size), a < b →
        acc[a].byteOffset + acc[a].byteSize ≤ acc[b].byteOffset) →
      let res := packOutputSlots.go shapes names i packOffset acc
      res.1.size = shapes.size ∧
      res.2 % 64 = 0 ∧
      (∀ k (hk : k < res.1.size),
        res.1[k].byteOffset % 64 = 0 ∧ res.1[k].byteSize % 64 = 0 ∧
        res.1[k].rowStride % 64 = 0 ∧ res.1[k].planeStride % 64 = 0 ∧
        res.1[k].batchStride % 64 = 0 ∧
        res.1[k].byteOffset + res.1[k].byteSize ≤ res.2) ∧
      (∀ a b (ha : a < res.1.size) (hb : b < res.1.size), a < b →
        res.1[a].byteOffset + res.1[a].byteSize ≤ res.1[b].byteOffset) := by
  intro rem
  induction rem with
  | zero =>
    intro i packOffset acc hrem hi hsz hOff64 hPrev hPair
    have hiEq : i = shapes.size := by omega
    subst hiEq
    unfold packOutputSlots.go
    simp only [Nat.lt_irrefl, ↓reduceDIte]
    exact ⟨hsz, hOff64, hPrev, hPair⟩
  | succ r ih =>
    intro i packOffset acc hrem hi hsz hOff64 hPrev hPair
    have hlt : i < shapes.size := by omega
    unfold packOutputSlots.go
    simp only [hlt, ↓reduceDIte]
    let sh := shapes[i]
    let rSt := alignUp (2 * sh.w) dmaLineBytes
    let pSt := sh.h * rSt
    let bSt := sh.c * pSt
    let sz := alignUp (sh.n * bSt) dmaLineBytes
    let nm := names.getD i (if shapes.size == 1 then "y" else s!"y{i}")
    let slot : OutputSlot := {
      name := nm, shape := sh, byteOffset := packOffset, byteSize := sz,
      rowStride := rSt, planeStride := pSt, batchStride := bSt
    }
    have hr64 : rSt % 64 = 0 := alignUp_dvd_local (2 * sh.w) 64 (by omega)
    have hp64 : pSt % 64 = 0 := by dsimp [pSt]; rw [Nat.mul_mod, hr64, Nat.mul_zero, Nat.zero_mod]
    have hb64 : bSt % 64 = 0 := by dsimp [bSt]; rw [Nat.mul_mod, hp64, Nat.mul_zero, Nat.zero_mod]
    have hsz64 : sz % 64 = 0 := alignUp_dvd_local (sh.n * bSt) 64 (by omega)
    apply ih (i + 1) (packOffset + sz) (acc.push slot) (by omega) (by omega)
      (by simp [hsz]) (by omega)
    · intro k hk
      by_cases hkl : k < acc.size
      · rw [Array.getElem_push_lt hkl]
        have hp := hPrev k hkl
        exact ⟨hp.1, hp.2.1, hp.2.2.1, hp.2.2.2.1, hp.2.2.2.2.1, by omega⟩
      · have hkEq : k = acc.size := by simp at hk; omega
        subst hkEq
        rw [Array.getElem_push_eq]
        exact ⟨hOff64, hsz64, hr64, hp64, hb64, Nat.le_refl _⟩
    · intro a b ha hb hab
      by_cases hbl : b < acc.size
      · have hal : a < acc.size := Nat.lt_trans hab hbl
        rw [Array.getElem_push_lt hal, Array.getElem_push_lt hbl]
        exact hPair a b hal hbl hab
      · have hbEq : b = acc.size := by simp at hb; omega
        subst hbEq
        have hal : a < acc.size := hab
        rw [Array.getElem_push_lt hal, Array.getElem_push_eq]
        exact (hPrev a hal).2.2.2.2.2

/-- Packed multi-output soundness: every `OutputSlot` returned by `packOutputSlots` is 64B-aligned,
    strictly within `[0, totalBytes)`, and pairwise disjoint across all logical outputs! -/
theorem packOutputSlots_sound (shapes : Array Shape4D) (names : Array String) :
    let (slots, totalBytes) := packOutputSlots shapes names
    slots.size = shapes.size ∧
    totalBytes % 64 = 0 ∧
    (∀ k (hk : k < slots.size),
      slots[k].byteOffset % 64 = 0 ∧ slots[k].byteSize % 64 = 0 ∧
      slots[k].rowStride % 64 = 0 ∧ slots[k].planeStride % 64 = 0 ∧
      slots[k].batchStride % 64 = 0 ∧
      slots[k].byteOffset + slots[k].byteSize ≤ totalBytes) ∧
    (∀ a b (ha : a < slots.size) (hb : b < slots.size), a < b →
      slots[a].byteOffset + slots[a].byteSize ≤ slots[b].byteOffset) := by
  unfold packOutputSlots
  exact packOutputSlots_go_spec shapes names shapes.size 0 0 (Array.mkEmpty shapes.size)
    (by omega) (Nat.zero_le _) (by simp) (by rfl)
    (fun k hk => by simp at hk)
    (fun a b ha => by simp at ha)

/-- `packOutputSlots` total packed bytes equals the exact sum of `slot.byteSize` across all slots. -/
theorem packOutputSlots_total_eq_sum (shapes : Array Shape4D) (names : Array String) :
    let (slots, totalBytes) := packOutputSlots shapes names
    totalBytes = slots.foldl (fun acc s => acc + s.byteSize) 0 := by
  have hGo : ∀ rem i packOffset (acc : Array OutputSlot),
      rem = shapes.size - i → i ≤ shapes.size →
      packOffset = acc.foldl (fun a s => a + s.byteSize) 0 →
      (packOutputSlots.go shapes names i packOffset acc).2 =
      (packOutputSlots.go shapes names i packOffset acc).1.foldl (fun a s => a + s.byteSize) 0 := by
    intro rem
    induction rem with
    | zero =>
      intro i packOffset acc hrem hi hEq
      have : i = shapes.size := by omega
      subst this
      unfold packOutputSlots.go
      simp only [Nat.lt_irrefl, ↓reduceDIte]
      exact hEq
    | succ r ih =>
      intro i packOffset acc hrem hi hEq
      have hlt : i < shapes.size := by omega
      unfold packOutputSlots.go
      simp only [hlt, ↓reduceDIte]
      apply ih (i + 1) _ _ (by omega) (by omega)
      rw [← Array.foldl_toList, Array.toList_push, List.foldl_append, List.foldl_cons, List.foldl_nil]
      rw [Array.foldl_toList, ← hEq]
  exact hGo shapes.size 0 0 (Array.mkEmpty shapes.size) (by omega) (Nat.zero_le _) (by simp)

/-- Every packed output slot fits strictly inside `hwxOutShape.span` for BOTH homogeneous and
    heterogeneous multi-output packing in `computeHwxOutShape`! -/
theorem computeHwxOutShape_span_covers_packed (outSlots : Array OutputSlot) (packOffset : Nat)
    (hwxShape : Shape4D) (isHomog : Bool)
    (hNonEmpty : outSlots.size > 0)
    (hOff64 : packOffset % 64 = 0)
    (hTotal : packOffset = outSlots.foldl (fun acc s => acc + s.byteSize) 0)
    (hSlots : ∀ k (hk : k < outSlots.size), outSlots[k].byteSize = outSlots[k].shape.span)
    (hOk : computeHwxOutShape outSlots packOffset = .ok (hwxShape, isHomog)) :
    packOffset ≤ hwxShape.span := by
  unfold computeHwxOutShape at hOk
  by_cases h1 : outSlots.size = 1
  · simp only [h1, beq_self_eq_true, ↓reduceIte, Except.ok.injEq, Prod.mk.injEq] at hOk
    rcases hOk with ⟨rfl, _⟩
    have h0 : 0 < outSlots.size := by omega
    have hList : outSlots.toList = [outSlots[0]] := by
      cases hl : outSlots.toList with
      | nil =>
        have hsz0 : outSlots.size = 0 := by rw [← Array.length_toList, hl]; rfl
        omega
      | cons x xs =>
        cases xs with
        | nil =>
          have : outSlots[0] = x := by simp [← Array.getElem_toList, hl]
          rw [this]
        | cons y ys =>
          have : outSlots.size ≥ 2 := by rw [← Array.length_toList, hl]; simp
          omega
    rw [hTotal, ← Array.foldl_toList, hList]
    simp only [List.foldl_cons, List.foldl_nil, Nat.zero_add]
    have hGet : outSlots[0]! = outSlots[0] := getElem!_pos outSlots 0 h0
    rw [hGet, hSlots 0 h0]
    exact Nat.le_refl _
  · have hNe : (outSlots.size == 1) = false := by simp [h1]
    simp only [hNe, Bool.false_eq_true, ↓reduceIte] at hOk
    split at hOk
    · rename_i hHom
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hHom
      simp only [Except.ok.injEq, Prod.mk.injEq] at hOk
      rcases hOk with ⟨rfl, _⟩
      have hAll := hHom.1
      rw [Array.all_eq_true] at hAll
      have hSumEq : ∀ (k : Nat) (hk : k ≤ outSlots.size),
          (outSlots.toList.take k).foldl (fun acc s => acc + s.byteSize) 0 =
          ((outSlots.toList.take k).foldl (fun acc s => acc + s.shape.c) 0) * outSlots[0]!.shape.planeStride := by
        intro k
        induction k with
        | zero => intro _; simp
        | succ m ihm =>
          intro hm
          have hmlt : m < outSlots.size := by omega
          have hTake : outSlots.toList.take (m + 1) = outSlots.toList.take m ++ [outSlots[m]] := by
            rw [List.take_succ_eq_append_getElem (by simpa using hmlt)]
            rfl
          rw [hTake, List.foldl_append, List.foldl_append]
          simp only [List.foldl_cons, List.foldl_nil]
          rw [ihm (Nat.le_of_lt hmlt), hSlots m hmlt]
          have hmProp := hAll m hmlt
          simp only [Bool.and_eq_true, beq_iff_eq] at hmProp
          have hSpanM : outSlots[m].shape.span = outSlots[m].shape.c * outSlots[0]!.shape.planeStride := by
            unfold Shape4D.span Shape4D.batchStride Shape4D.planeStride Shape4D.rowStride
            rw [hmProp.1.1, hmProp.1.2, hmProp.2, Nat.one_mul]
          rw [hSpanM, ← Nat.add_mul]
      have hFull := hSumEq outSlots.size (Nat.le_refl _)
      rw [List.take_of_length_le (by simp), Array.foldl_toList, Array.foldl_toList] at hFull
      rw [hTotal, hFull]
      unfold Shape4D.span Shape4D.batchStride Shape4D.planeStride Shape4D.rowStride
      simp only [Nat.one_mul]
      exact Nat.le_refl _
    · have hRow32 : alignUp (2 * 32) dmaLineBytes = 64 := by decide
      have hPackEq : packOffset = (packOffset / 64) * 64 := by omega
      by_cases hSmall : packOffset / dmaLineBytes ≤ 65535
      · simp only [hSmall, ↓reduceIte] at hOk
        split at hOk
        · cases hOk
        · have hEq := Prod.mk.inj (Except.ok.inj hOk)
          rw [← hEq.1]
          unfold Shape4D.span Shape4D.batchStride Shape4D.planeStride Shape4D.rowStride
          dsimp only
          rw [hRow32, Nat.one_mul, Nat.one_mul]
          change packOffset ≤ (packOffset / 64) * 64
          omega
      · simp only [hSmall, ↓reduceIte] at hOk
        split at hOk
        · cases hOk
        · have hEq := Prod.mk.inj (Except.ok.inj hOk)
          rw [← hEq.1]
          unfold Shape4D.span Shape4D.batchStride Shape4D.planeStride Shape4D.rowStride
          dsimp only
          rw [hRow32, Nat.one_mul]
          let totalLines := packOffset / 64
          let hp := (totalLines + 65534) / 65535
          have hSmall64 : ¬ packOffset / 64 ≤ 65535 := hSmall
          have hHpPos : hp > 0 := Nat.div_pos (by omega) (by omega)
          have hGe : totalLines ≤ ((totalLines + hp - 1) / hp) * hp := by
            have h1 := Nat.div_add_mod' (totalLines + hp - 1) hp
            have h2 := Nat.mod_lt (totalLines + hp - 1) hHpPos
            omega
          have hGoal : packOffset ≤ ((totalLines + hp - 1) / hp) * (hp * 64) := by
            rw [← Nat.mul_assoc, hPackEq]
            exact Nat.mul_le_mul_right 64 hGe
          exact hGoal

-- ============================================================================
-- 2C. Temp Staging Non-Overlap (`placeTempBuffers`)
-- ============================================================================

/-- Convert `(sizes, offs)` from `placeTempBuffers` into the list of placed temp `Range`s. -/
def tempRangesOf : List Nat → List Nat → List Range
  | sz :: szs, off :: offs => { base := off, bytes := alignUp sz dmaLineBytes } :: tempRangesOf szs offs
  | _, _ => []

/-- `findFreeOffset` always returns a 64B-aligned offset when all occupied ranges have 64B-aligned ends. -/
theorem findFreeOffset_dvd_64 (occupied : List Range) (bytes cap off : Nat)
    (hOcc : ∀ r ∈ occupied, r.hi % 64 = 0)
    (hFind : findFreeOffset occupied bytes cap = some off) :
    off % 64 = 0 := by
  unfold findFreeOffset at hFind
  dsimp only at hFind
  generalize hv : (0 :: occupied.map Range.hi).filter (fun o =>
    decide (o + bytes ≤ cap) &&
    occupied.all (fun r => decide (Apart r { base := o, bytes := bytes }))) = valid at hFind
  cases valid with
  | nil => simp at hFind
  | cons first rest =>
    simp only [Option.some.injEq] at hFind
    subst hFind
    have hPred : ∀ x ∈ first :: rest, x % 64 = 0 := by
      intro x hx
      rw [← hv, List.mem_filter] at hx
      rcases List.mem_cons.mp hx.1 with rfl | hmap
      · rfl
      · rw [List.mem_map] at hmap
        obtain ⟨r, hr, rfl⟩ := hmap
        exact hOcc r hr
    exact foldl_min_preserves_local (P := fun x => x % 64 = 0) rest first
      (hPred first List.mem_cons_self)
      (fun x hx => hPred x (List.mem_cons_of_mem _ hx))

private theorem findFreeOffset_sound_local (occupied : List Range) (bytes cap off : Nat)
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
    exact foldl_min_preserves_local rest first
      (hpred first List.mem_cons_self)
      (fun x hx => hpred x (List.mem_cons_of_mem _ hx))

/-- Complete temp staging soundness for `placeTempBuffers`:
    whenever `placeTempBuffers occupied sizes = some offs`:
    1. `offs.length = sizes.length`,
    2. every temp staging range lies strictly within `[0, l2CapacityBytes)`,
    3. every temp staging range is disjoint (`Apart`) from EVERY live L2 range in `occupied`,
    4. all temp staging ranges at the task are pairwise disjoint (`Pairwise Apart`)! -/
theorem placeTempBuffers_sound (occupied : List Range) (sizes offs : List Nat)
    (hOk : placeTempBuffers occupied sizes = some offs) :
    offs.length = sizes.length ∧
    (∀ tr ∈ tempRangesOf sizes offs, tr.hi ≤ l2CapacityBytes) ∧
    (∀ tr ∈ tempRangesOf sizes offs, ∀ r ∈ occupied, Apart r tr) ∧
    (tempRangesOf sizes offs).Pairwise Apart := by
  revert occupied offs hOk
  induction sizes with
  | nil =>
    intro occupied offs hOk
    unfold placeTempBuffers at hOk
    simp only [Option.some.injEq] at hOk
    subst hOk
    simp [tempRangesOf]
  | cons sz rest ih =>
    intro occupied offs hOk
    unfold placeTempBuffers at hOk
    cases hFind : findFreeOffset occupied (alignUp sz dmaLineBytes) l2CapacityBytes with
    | none => simp [hFind] at hOk
    | some off =>
      simp only [hFind] at hOk
      cases hRest : placeTempBuffers ({ base := off, bytes := alignUp sz dmaLineBytes } :: occupied) rest with
      | none => simp [hRest] at hOk
      | some restOffs =>
        simp only [hRest, Option.some.injEq] at hOk
        subst hOk
        have hFirst := findFreeOffset_sound_local occupied (alignUp sz dmaLineBytes) l2CapacityBytes off hFind
        have hRec := ih ({ base := off, bytes := alignUp sz dmaLineBytes } :: occupied) restOffs hRest
        let tr0 : Range := { base := off, bytes := alignUp sz dmaLineBytes }
        refine ⟨by simp [hRec.1], ?_, ?_, ?_⟩
        · intro tr htr
          simp only [tempRangesOf, List.mem_cons] at htr
          rcases htr with rfl | hmem
          · exact hFirst.1
          · exact hRec.2.1 tr hmem
        · intro tr htr r hr
          simp only [tempRangesOf, List.mem_cons] at htr
          rcases htr with rfl | hmem
          · exact hFirst.2 r hr
          · exact hRec.2.2.1 tr hmem r (List.mem_cons_of_mem _ hr)
        · simp only [tempRangesOf]
          apply List.Pairwise.cons
          · intro tr htr
            exact hRec.2.2.1 tr htr tr0 List.mem_cons_self
          · exact hRec.2.2.2

/-- 64-byte alignment of `placeTempBuffers`: if all live L2 ranges in `occupied` have 64B-aligned
    ends, then every temp buffer range returned by `placeTempBuffers` has 64B-aligned `base` and `hi`. -/
theorem placeTempBuffers_aligned_64 (occupied : List Range) (sizes offs : List Nat)
    (hOcc : ∀ r ∈ occupied, r.hi % 64 = 0)
    (hOk : placeTempBuffers occupied sizes = some offs) :
    ∀ tr ∈ tempRangesOf sizes offs, tr.base % 64 = 0 ∧ tr.hi % 64 = 0 := by
  revert occupied offs hOcc hOk
  induction sizes with
  | nil =>
    intro occupied offs _ hOk
    unfold placeTempBuffers at hOk
    simp only [Option.some.injEq] at hOk
    subst hOk
    intro _ h; cases h
  | cons sz rest ih =>
    intro occupied offs hOcc hOk
    unfold placeTempBuffers at hOk
    cases hFind : findFreeOffset occupied (alignUp sz dmaLineBytes) l2CapacityBytes with
    | none => simp [hFind] at hOk
    | some off =>
      simp only [hFind] at hOk
      cases hRest : placeTempBuffers ({ base := off, bytes := alignUp sz dmaLineBytes } :: occupied) rest with
      | none => simp [hRest] at hOk
      | some restOffs =>
        simp only [hRest, Option.some.injEq] at hOk
        subst hOk
        have hOff64 := findFreeOffset_dvd_64 occupied (alignUp sz dmaLineBytes) l2CapacityBytes off hOcc hFind
        have hSz64 : alignUp sz dmaLineBytes % 64 = 0 := alignUp_dvd_local sz 64 (by omega)
        have hHi64 : ({ base := off, bytes := alignUp sz dmaLineBytes } : Range).hi % 64 = 0 := by
          unfold Range.hi; dsimp only; omega
        have hOcc' : ∀ r ∈ ({ base := off, bytes := alignUp sz dmaLineBytes } :: occupied), r.hi % 64 = 0 := by
          intro r hr
          rcases List.mem_cons.mp hr with rfl | hmem
          · exact hHi64
          · exact hOcc r hmem
        have hRec := ih _ restOffs hOcc' hRest
        intro tr htr
        simp only [tempRangesOf, List.mem_cons] at htr
        rcases htr with rfl | hmem
        · exact ⟨hOff64, hHi64⟩
        · exact hRec tr hmem

/-- Combined live-L2 + temp-staging packing soundness: if `occupied` is `Packed l2CapacityBytes`,
    then `tempRangesOf sizes offs ++ occupied` is ALSO `Packed l2CapacityBytes`! -/
theorem placeTempBuffers_packed (occupied : List Range) (sizes offs : List Nat)
    (hPack : Packed l2CapacityBytes occupied)
    (hOk : placeTempBuffers occupied sizes = some offs) :
    Packed l2CapacityBytes (tempRangesOf sizes offs ++ occupied) := by
  have hS := placeTempBuffers_sound occupied sizes offs hOk
  refine ⟨?_, ?_⟩
  · rw [List.pairwise_append]
    exact ⟨hS.2.2.2, hPack.1, fun tr htr r hr => (hS.2.2.1 tr htr r hr).symm⟩
  · intro r hr
    rcases List.mem_append.mp hr with h1 | h2
    · exact hS.2.1 r h1
    · exact hPack.2 r h2

-- ============================================================================
-- 2D. L2 & BAR 3 Scratch Allocators (`allocateL2Buffers`, `allocateScratchBuffers`)
-- ============================================================================

/-- `allocateL2Buffers` (called directly by `planMemory`) preserves 64B alignment of every
    L2 placement whenever every `LiveBuf` has 64B-aligned size (e.g. `l2ResidentBytes4D`). -/
theorem allocateL2Buffers_aligned_64 (liveBufs : List LiveBuf)
    (h64 : ∀ b ∈ liveBufs, b.bytes % 64 = 0) :
    ∀ p ∈ allocateL2Buffers liveBufs, p.base % 64 = 0 ∧ p.hi % 64 = 0 :=
  allocate_dvd_64 true liveBufs h64

/-- `allocateScratchBuffers` (called directly by `planMemory`) returns 64B-aligned placements
    and a 64B-aligned total `scratchBytes` upper-bounding every placed scratch buffer. -/
theorem allocateScratchBuffers_aligned_and_bounded (dramLiveBufs : List LiveBuf)
    (h64 : ∀ b ∈ dramLiveBufs, b.bytes % 64 = 0) :
    let (placed, scratchBytes) := allocateScratchBuffers dramLiveBufs
    scratchBytes % 64 = 0 ∧
    ∀ p ∈ placed, p.base % 64 = 0 ∧ p.hi % 64 = 0 ∧ p.hi ≤ scratchBytes := by
  unfold allocateScratchBuffers
  dsimp only
  have hAlign := allocate_dvd_64 true dramLiveBufs h64
  have hLeTop : ∀ ps : List PlacedBuf, ∀ p ∈ ps, p.hi ≤ topOf ps := by
    intro ps
    induction ps with
    | nil => intro _ h; cases h
    | cons hd tl ih =>
      intro p hp
      simp only [topOf]
      rcases List.mem_cons.mp hp with rfl | htl
      · exact Nat.le_max_left _ _
      · exact Nat.le_trans (ih p htl) (Nat.le_max_right _ _)
  exact ⟨topOf_dvd_64 _ (fun p hp => (hAlign p hp).2),
         fun p hp => ⟨(hAlign p hp).1, (hAlign p hp).2, hLeTop _ p hp⟩⟩

-- ============================================================================
-- 2E. Real `validateDonations` Safety Invariants (Directly Connected to `planMemory`)
-- ============================================================================

private theorem nodup_getElem_ne {α : Type} : ∀ (l : List α), l.Nodup →
    ∀ (i j : Nat) (hi : i < l.length) (hj : j < l.length), i ≠ j → l[i] ≠ l[j] := by
  intro l
  induction l with
  | nil => intro _ _ _ hi; cases hi
  | cons x xs ih =>
    intro hnd i j hi hj hne heq
    have ⟨hxNot, hxsNd⟩ := List.nodup_cons.mp hnd
    cases i with
    | zero =>
      cases j with
      | zero => exact hne rfl
      | succ j' =>
        simp only [List.getElem_cons_zero, List.getElem_cons_succ] at heq
        exact hxNot (heq ▸ List.getElem_mem _)
    | succ i' =>
      cases j with
      | zero =>
        simp only [List.getElem_cons_zero, List.getElem_cons_succ] at heq
        exact hxNot (heq.symm ▸ List.getElem_mem _)
      | succ j' =>
        simp only [List.getElem_cons_succ] at heq
        exact ih hxsNd i' j' (by simpa using hi) (by simpa using hj) (by omega) heq

private theorem except_if_throw_ok {α : Type} {c : Prop} [Decidable c] {e : String}
    {k : Except String α} {v : α}
    (h : (if c then Except.error e else k) = .ok v) : ¬ c ∧ k = .ok v := by
  by_cases hc : c
  · simp [hc] at h
  · simp [hc] at h
    exact ⟨hc, h⟩

private theorem validateDonationsAux_sound
    (inputSurfs : Array SurfaceSpec) (outSlots : Array OutputSlot)
    (hwxOutSpanBytes : Nat) (inSurfBufIds : Array BufferId) (outSurfBufId : BufferId)
    (buffers : Array BufferStorage) (machineTasks : Array MachineTask)
    (taskOutSurfIdx : Array (Option Nat)) :
    ∀ (raw : List (Nat × Nat)) (seenIn seenOut : List Nat)
      (acc ds : Array BufferDonation),
      (∀ d ∈ acc.toList, d.inputSurfIdx ∈ seenIn ∧ d.outputSurfIdx ∈ seenOut) →
      (acc.toList.map BufferDonation.inputSurfIdx).Nodup →
      (acc.toList.map BufferDonation.outputSurfIdx).Nodup →
      (∀ d ∈ acc.toList,
        ∃ (hin : d.inputSurfIdx < inputSurfs.size) (hout : d.outputSurfIdx < outSlots.size),
          let inS := inputSurfs[d.inputSurfIdx]
          let outSl := outSlots[d.outputSurfIdx]
          inS.channels.toNat = outSl.shape.c ∧ inS.height.toNat = outSl.shape.h ∧
          inS.width.toNat = outSl.shape.w ∧ inS.n.toNat = outSl.shape.n ∧
          inS.batchStride.toNat * inS.n.toNat = hwxOutSpanBytes ∧
          d.lastReadTask < d.firstWriteTask) →
      validateDonationsAux inputSurfs outSlots hwxOutSpanBytes inSurfBufIds outSurfBufId
        buffers machineTasks taskOutSurfIdx seenIn seenOut acc raw = .ok ds →
      ds.size = acc.size + raw.length ∧
      (ds.toList.map BufferDonation.inputSurfIdx).Nodup ∧
      (ds.toList.map BufferDonation.outputSurfIdx).Nodup ∧
      (∀ d ∈ ds.toList,
        ∃ (hin : d.inputSurfIdx < inputSurfs.size) (hout : d.outputSurfIdx < outSlots.size),
          let inS := inputSurfs[d.inputSurfIdx]
          let outSl := outSlots[d.outputSurfIdx]
          inS.channels.toNat = outSl.shape.c ∧ inS.height.toNat = outSl.shape.h ∧
          inS.width.toNat = outSl.shape.w ∧ inS.n.toNat = outSl.shape.n ∧
          inS.batchStride.toNat * inS.n.toNat = hwxOutSpanBytes ∧
          d.lastReadTask < d.firstWriteTask) := by
  intro raw
  induction raw with
  | nil =>
    intro seenIn seenOut acc ds _ hNdIn hNdOut hAll hOk
    unfold validateDonationsAux at hOk
    simp only [Except.ok.injEq] at hOk
    subst hOk
    exact ⟨by simp, hNdIn, hNdOut, hAll⟩
  | cons pair rest ih =>
    rcases pair with ⟨inIdx, outIdx⟩
    intro seenIn seenOut acc ds hSeen hNdIn hNdOut hAll hOk
    unfold validateDonationsAux at hOk
    dsimp only [bind, Except.bind, pure, Except.pure, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw] at hOk
    split at hOk; · cases hOk
    rename_i hBnd
    have hInLt : inIdx < inputSurfs.size := by
      simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hBnd; omega
    have hOutLt : outIdx < outSlots.size := by
      simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hBnd; omega
    split at hOk; · cases hOk
    rename_i hNotSeenIn
    split at hOk; · cases hOk
    rename_i hNotSeenOut
    split at hOk; · cases hOk
    rename_i hShapeOk
    split at hOk; · cases hOk
    rename_i hSpanOk
    cases hBounds : computeDonationTaskBounds buffers machineTasks taskOutSurfIdx inIdx outIdx with
    | error e => rw [hBounds] at hOk; cases hOk
    | ok lrFw =>
      rcases lrFw with ⟨lr, fw⟩
      rw [hBounds] at hOk
      dsimp only at hOk
      split at hOk; · cases hOk
      rename_i hOrdOk
      have hLrLtFw : lr < fw := by omega
      have hInNotMem : inIdx ∉ seenIn := by
        intro hmem; apply hNotSeenIn
        simp [hmem]
      have hOutNotMem : outIdx ∉ seenOut := by
        intro hmem; apply hNotSeenOut
        simp [hmem]
      let d : BufferDonation := {
        inputSurfIdx := inIdx, outputSurfIdx := outIdx,
        inputBuf := inSurfBufIds[inIdx]!, outputBuf := outSurfBufId,
        lastReadTask := lr, firstWriteTask := fw
      }
      have hSeen' : ∀ x ∈ (acc.push d).toList, x.inputSurfIdx ∈ inIdx :: seenIn ∧ x.outputSurfIdx ∈ outIdx :: seenOut := by
        intro x hx
        rw [Array.toList_push, List.mem_append, List.mem_singleton] at hx
        rcases hx with hxOld | rfl
        · exact ⟨List.mem_cons_of_mem _ (hSeen x hxOld).1, List.mem_cons_of_mem _ (hSeen x hxOld).2⟩
        · exact ⟨List.mem_cons_self, List.mem_cons_self⟩
      have hNdIn' : ((acc.push d).toList.map BufferDonation.inputSurfIdx).Nodup := by
        rw [Array.toList_push, List.map_append, List.map_cons, List.map_nil, List.nodup_append]
        refine ⟨hNdIn, by simp, ?_⟩
        intro a ha b hb hab
        rw [List.mem_singleton] at hb
        subst hb; subst hab
        rw [List.mem_map] at ha
        obtain ⟨x, hx, hEq⟩ := ha
        exact hInNotMem (hEq ▸ (hSeen x hx).1)
      have hNdOut' : ((acc.push d).toList.map BufferDonation.outputSurfIdx).Nodup := by
        rw [Array.toList_push, List.map_append, List.map_cons, List.map_nil, List.nodup_append]
        refine ⟨hNdOut, by simp, ?_⟩
        intro a ha b hb hab
        rw [List.mem_singleton] at hb
        subst hb; subst hab
        rw [List.mem_map] at ha
        obtain ⟨x, hx, hEq⟩ := ha
        exact hOutNotMem (hEq ▸ (hSeen x hx).2)
      have hShapeEq :
          inputSurfs[inIdx].channels.toNat = outSlots[outIdx].shape.c ∧
          inputSurfs[inIdx].height.toNat = outSlots[outIdx].shape.h ∧
          inputSurfs[inIdx].width.toNat = outSlots[outIdx].shape.w ∧
          inputSurfs[inIdx].n.toNat = outSlots[outIdx].shape.n := by
        rw [getElem!_pos inputSurfs inIdx hInLt, getElem!_pos outSlots outIdx hOutLt] at hShapeOk
        simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Classical.not_not] at hShapeOk
        exact ⟨hShapeOk.1.1.1, hShapeOk.1.1.2, hShapeOk.1.2, hShapeOk.2⟩
      have hSpanEq : inputSurfs[inIdx].batchStride.toNat * inputSurfs[inIdx].n.toNat = hwxOutSpanBytes := by
        rw [getElem!_pos inputSurfs inIdx hInLt] at hSpanOk
        simp only [bne_iff_ne, ne_eq, Classical.not_not] at hSpanOk
        exact hSpanOk
      have hAll' : ∀ x ∈ (acc.push d).toList,
          ∃ (hin : x.inputSurfIdx < inputSurfs.size) (hout : x.outputSurfIdx < outSlots.size),
            let inS := inputSurfs[x.inputSurfIdx]
            let outSl := outSlots[x.outputSurfIdx]
            inS.channels.toNat = outSl.shape.c ∧ inS.height.toNat = outSl.shape.h ∧
            inS.width.toNat = outSl.shape.w ∧ inS.n.toNat = outSl.shape.n ∧
            inS.batchStride.toNat * inS.n.toNat = hwxOutSpanBytes ∧
            x.lastReadTask < x.firstWriteTask := by
        intro x hx
        rw [Array.toList_push, List.mem_append, List.mem_singleton] at hx
        rcases hx with hxOld | rfl
        · exact hAll x hxOld
        · exact ⟨hInLt, hOutLt, hShapeEq.1, hShapeEq.2.1, hShapeEq.2.2.1, hShapeEq.2.2.2, hSpanEq, hLrLtFw⟩
      have hRec := ih (inIdx :: seenIn) (outIdx :: seenOut) (acc.push d) ds hSeen' hNdIn' hNdOut' hAll' hOk
      refine ⟨?_, hRec.2.1, hRec.2.2.1, hRec.2.2.2⟩
      simp only [Array.size_push, List.length_cons] at hRec ⊢
      omega

/-- Complete soundness of `Vane.Backend.MemoryPlan.validateDonations` (called directly by `planMemory`):
    every accepted `BufferDonation` array satisfies:
    1. non-empty donations forbid packed multi-output (`outSlots.size ≤ 1`),
    2. strictly distinct input and output surface indices (injective, no aliasing),
    3. exact `(c, h, w, n)` shape and output-surface span equality,
    4. strict `lastReadTask < firstWriteTask` separation (forbidding same-task or post-write reads). -/
theorem validateDonations_sound
    (rawDonations : Array (Nat × Nat))
    (inputSurfs : Array SurfaceSpec) (outSlots : Array OutputSlot)
    (hwxOutSpanBytes : Nat) (inSurfBufIds : Array BufferId) (outSurfBufId : BufferId)
    (buffers : Array BufferStorage) (machineTasks : Array MachineTask)
    (taskOutSurfIdx : Array (Option Nat))
    (ds : Array BufferDonation)
    (hOk : validateDonations rawDonations inputSurfs outSlots hwxOutSpanBytes
             inSurfBufIds outSurfBufId buffers machineTasks taskOutSurfIdx = .ok ds) :
    ds.size = rawDonations.size ∧
    (ds.size > 0 → outSlots.size ≤ 1) ∧
    (∀ (i j : Nat) (hi : i < ds.size) (hj : j < ds.size), i ≠ j →
      ds[i].inputSurfIdx ≠ ds[j].inputSurfIdx ∧ ds[i].outputSurfIdx ≠ ds[j].outputSurfIdx) ∧
    (∀ d ∈ ds.toList,
      ∃ (hin : d.inputSurfIdx < inputSurfs.size) (hout : d.outputSurfIdx < outSlots.size),
        let inS := inputSurfs[d.inputSurfIdx]
        let outSl := outSlots[d.outputSurfIdx]
        inS.channels.toNat = outSl.shape.c ∧ inS.height.toNat = outSl.shape.h ∧
        inS.width.toNat = outSl.shape.w ∧ inS.n.toNat = outSl.shape.n ∧
        inS.batchStride.toNat * inS.n.toNat = hwxOutSpanBytes ∧
        d.lastReadTask < d.firstWriteTask) := by
  unfold validateDonations at hOk
  dsimp only [bind, Except.bind, pure, Except.pure, throw, MonadExcept.throw, throwThe, MonadExceptOf.throw] at hOk
  split at hOk
  · cases hOk
  · rename_i hNotMulti
    have hAux := validateDonationsAux_sound inputSurfs outSlots hwxOutSpanBytes inSurfBufIds outSurfBufId
      buffers machineTasks taskOutSurfIdx rawDonations.toList [] [] #[] ds
      (fun _ h => by simp at h) (by simp) (by simp) (fun _ h => by simp at h) hOk
    have hSz : ds.size = rawDonations.size := by simpa using hAux.1
    refine ⟨hSz, ?_, ?_, hAux.2.2.2⟩
    · intro hPos
      by_cases hGt : outSlots.size > 1
      · exfalso; apply hNotMulti
        have hSzPos : rawDonations.size > 0 := by omega
        have hNe : rawDonations.isEmpty = false := by
          cases rawDonations with
          | mk l => cases l with
            | nil => simp at hSzPos
            | cons _ _ => rfl
        simp [hNe, hGt]
      · omega
    · intro i j hi hj hne
      have hMapNe : ∀ (f : BufferDonation → Nat), (ds.toList.map f).Nodup → f ds[i] ≠ f ds[j] := by
        intro f hnd
        have hLen : (ds.toList.map f).length = ds.size := by simp
        have h := nodup_getElem_ne (ds.toList.map f) hnd i j (by rw [hLen]; exact hi) (by rw [hLen]; exact hj) hne
        simpa using h
      exact ⟨hMapNe BufferDonation.inputSurfIdx hAux.2.1,
             hMapNe BufferDonation.outputSurfIdx hAux.2.2.1⟩

end Vane.Proofs.C3MemoryPlan
