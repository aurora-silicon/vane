import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Hwx
import Vane.Register
import Vane.Program
import Vane.Tiling
import Vane.Quant
import Vane.Pe
import Vane.Lut
import Vane.Pool
import Vane.Conv
import Vane.View
import Vane.Matmul
import Vane.Verify
import Vane.Ane.Types
import Vane.Machine.Types
import Vane.Backend.SyncAndEmit

namespace Vane.Proofs.C3Emit

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Hwx
open Vane.Register
open Vane.Program
open Vane.Tiling
open Vane.Quant
open Vane.Pe
open Vane.Lut
open Vane.Pool
open Vane.Conv
open Vane.View
open Vane.Matmul
open Vane.Verify
open Vane.Ane
open Vane.Machine
open Vane.Backend.SyncAndEmit

-- ============================================================================
-- Pillar 4: Unified 5-Engine Emission → Verifier Soundness
-- ============================================================================

private theorem alignUp_dvd_64 (n : Nat) : alignUp n 64 % 64 = 0 := by
  unfold alignUp
  have h64 : ((64 : Nat) == 0) = false := rfl
  rw [h64]
  simp only [Bool.false_eq_true, ↓reduceIte]
  split
  · rename_i h; simpa using h
  · have hlt : n % 64 < 64 := Nat.mod_lt n (by omega)
    have hdiv : 64 * (n / 64) + n % 64 = n := Nat.div_add_mod n 64
    have hmul : (n / 64 + 1) * 64 = 64 * (n / 64) + 64 := by
      rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm]
    have h1 : n + (64 - n % 64) = (n / 64 + 1) * 64 := by omega
    rw [h1, Nat.mul_mod_left]

private theorem alignUp_ge_local (n m : Nat) : n ≤ alignUp n m := by
  unfold alignUp
  split
  · omega
  · dsimp only; split <;> omega

private theorem alignUp_lt_add_64 (n : Nat) : alignUp n 64 < n + 64 := by
  unfold alignUp
  have h64 : ((64 : Nat) == 0) = false := rfl
  rw [h64]
  simp only [Bool.false_eq_true, ↓reduceIte]
  split
  · omega
  · rename_i hne
    have hrem : n % 64 ≠ 0 := by simpa using hne
    have hlt : n % 64 < 64 := Nat.mod_lt n (by omega)
    omega

private theorem natToU32Checked_ok_of_lt (n : Nat) (label : String) (h : n ≤ maxU32Nat) :
    natToU32Checked n label = .ok n.toUInt32 := by
  unfold natToU32Checked
  have hnot : ¬ (n > maxU32Nat) := by omega
  simp [hnot]

private theorem validateWordLayoutFields_name_irrel (n1 n2 : String) (fixedMask : UInt32)
    (fs : List BitField) (used : UInt32) (seen : List String) (m : UInt32)
    (h : validateWordLayoutFields n1 fixedMask fs used seen = .ok m) :
    validateWordLayoutFields n2 fixedMask fs used seen = .ok m := by
  induction fs generalizing used seen with
  | nil => simpa [validateWordLayoutFields] using h
  | cons f rest ih =>
    unfold validateWordLayoutFields at h ⊢
    split at h; · cases h
    split at h; · cases h
    split at h; · cases h
    dsimp only at h ⊢
    split at h; · cases h
    split at h; · cases h
    rename_i h5 h4 h3 h2 h1
    simp only [h1, h2, h3, h4, h5, Bool.false_eq_true, ↓reduceIte]
    exact ih (used ||| f.mask) (f.name :: seen) h

private theorem validateEncodeValues_name_irrel (n1 n2 : String) (fs : List BitField)
    (vals : List (String × UInt32)) (seen : List String)
    (h : validateEncodeValues n1 fs vals seen = .ok ()) :
    validateEncodeValues n2 fs vals seen = .ok () := by
  induction vals generalizing seen with
  | nil => rfl
  | cons v rest ih =>
    rcases v with ⟨fname, val⟩
    unfold validateEncodeValues at h ⊢
    split at h; · cases h
    split at h; · cases h
    rename_i h2 h1
    simp only [h1, h2, Bool.false_eq_true, ↓reduceIte]
    exact ih (fname :: seen) h

private theorem encodeWordFields_name_irrel (n1 n2 : String) (vals : List (String × UInt32))
    (fs : List BitField) (acc w : UInt32)
    (h : encodeWordFields n1 vals fs acc = .ok w) :
    encodeWordFields n2 vals fs acc = .ok w := by
  induction fs generalizing acc with
  | nil => simpa [encodeWordFields] using h
  | cons f rest ih =>
    unfold encodeWordFields at h ⊢
    cases hv : vals.find? (fun p => decide (p.1 = f.name)) with
    | none => simp [hv] at h
    | some p =>
      rcases p with ⟨_, val⟩
      simp only [hv] at h ⊢
      split at h; · cases h
      rename_i h1
      simp only [h1, ↓reduceIte]
      exact ih (acc ||| (val <<< f.lo.toUInt32)) h

private theorem validateWordLayout_name_irrel {l1 l2 : WordLayout} {m : UInt32}
    (hF : l1.fields = l2.fields) (hFM : l1.fixedMask = l2.fixedMask) (hFV : l1.fixedValue = l2.fixedValue)
    (h : validateWordLayout l1 = .ok m) : validateWordLayout l2 = .ok m := by
  unfold validateWordLayout at h ⊢
  rw [hF, hFM, hFV] at h
  by_cases hfix : (l2.fixedValue &&& ~~~l2.fixedMask != 0) = true
  · simp [hfix] at h
  · simp only [hfix] at h ⊢
    exact validateWordLayoutFields_name_irrel l1.name l2.name l2.fixedMask l2.fields.toList l2.fixedMask [] m h

private theorem encodeWord_name_irrel {l1 l2 : WordLayout} {vals : Array (String × UInt32)} {w : UInt32}
    (hF : l1.fields = l2.fields) (hFM : l1.fixedMask = l2.fixedMask) (hFV : l1.fixedValue = l2.fixedValue)
    (h : encodeWord l1 vals = .ok w) : encodeWord l2 vals = .ok w := by
  unfold encodeWord at h ⊢
  cases hv : validateWordLayout l1 with
  | error e => simp [bind, Except.bind, hv] at h
  | ok m =>
    have hv2 := validateWordLayout_name_irrel hF hFM hFV hv
    simp only [bind, Except.bind, hv, hv2] at h ⊢
    rw [hF, hFV] at h
    cases he : validateEncodeValues l1.name l2.fields.toList vals.toList [] with
    | error e => simp [he] at h
    | ok _ =>
      have he2 := validateEncodeValues_name_irrel l1.name l2.name l2.fields.toList vals.toList [] he
      simp only [he, he2] at h ⊢
      exact encodeWordFields_name_irrel l1.name l2.name vals.toList l2.fields.toList l2.fixedValue w h

private theorem decodeFieldChecked_name_irrel {l1 l2 : WordLayout} {w : UInt32} {fn : String} {v : UInt32}
    (hF : l1.fields = l2.fields) (hFM : l1.fixedMask = l2.fixedMask) (hFV : l1.fixedValue = l2.fixedValue)
    (h : decodeFieldChecked l1 w fn = .ok v) : decodeFieldChecked l2 w fn = .ok v := by
  unfold decodeFieldChecked validateWordBits at h ⊢
  cases hv : validateWordLayout l1 with
  | error e => simp [bind, Except.bind, hv] at h
  | ok m =>
    have hv2 := validateWordLayout_name_irrel hF hFM hFV hv
    simp only [bind, Except.bind, hv, hv2] at h ⊢
    rw [hF, hFM, hFV] at h
    by_cases h1 : (w &&& l2.fixedMask != l2.fixedValue) = true
    · simp [h1] at h
    · by_cases h2 : (w &&& ~~~m != 0) = true
      · simp [h1, h2] at h
      · simp only [h1, h2, Bool.false_eq_true, ↓reduceIte] at h ⊢
        cases hf : findBitField? l2.fields.toList fn with
        | none => simp [hf] at h
        | some f => simpa [hf] using h

-- ============================================================================
-- 4A. Shared Multi-Task Port, Dependency & Control Register Soundness
-- ============================================================================

/-- `encodeSrcPortCfgMulti` ALWAYS succeeds at the register-layout level for every boolean
    combination and decodes back to the exact field values.
    Note: `Vane.Verify.vDecodeSrcPortCfg` additionally enforces `isFirst = true → isL2 = false`
    (first task cannot use L2-resident input mode=0) and non-dangling wait/handoff bits. -/
theorem encodeSrcPortCfgMulti_succeeds
    (name : String) (isFirst isL2 waitNe5 waitPe22 handoffNe handoffPe : Bool) :
    ∃ w, encodeSrcPortCfgMulti name isFirst isL2 waitNe5 waitPe22 handoffNe handoffPe = .ok w ∧
      decodeFieldChecked (portCfgMultiLayout name) w "P" = .ok 1 ∧
      decodeFieldChecked (portCfgMultiLayout name) w "mode" = .ok (if isL2 then portModeL2 else portModeDma) ∧
      decodeFieldChecked (portCfgMultiLayout name) w "dmaFlag" = .ok (if isL2 then 0 else 1) ∧
      decodeFieldChecked (portCfgMultiLayout name) w "waitNe4" = .ok (if isFirst then 1 else 0) ∧
      decodeFieldChecked (portCfgMultiLayout name) w "waitPe20" = .ok (if isFirst then 1 else 0) ∧
      decodeFieldChecked (portCfgMultiLayout name) w "waitNe5" = .ok (if isFirst || waitNe5 then 1 else 0) ∧
      decodeFieldChecked (portCfgMultiLayout name) w "waitPe22" = .ok (if isFirst || waitPe22 then 1 else 0) ∧
      decodeFieldChecked (portCfgMultiLayout name) w "handoffNe" = .ok (if !isFirst && isL2 && handoffNe then 1 else 0) ∧
      decodeFieldChecked (portCfgMultiLayout name) w "handoffPe" = .ok (if !isFirst && isL2 && handoffPe then 1 else 0) := by
  have hEnc := fun vals w (h : encodeWord (portCfgMultiLayout "src1Cfg") vals = .ok w) =>
    encodeWord_name_irrel (l1 := portCfgMultiLayout "src1Cfg") (l2 := portCfgMultiLayout name) rfl rfl rfl h
  have hDec := fun w f v (h : decodeFieldChecked (portCfgMultiLayout "src1Cfg") w f = .ok v) =>
    decodeFieldChecked_name_irrel (l1 := portCfgMultiLayout "src1Cfg") (l2 := portCfgMultiLayout name) rfl rfl rfl h
  cases isFirst <;> cases isL2 <;> cases waitNe5 <;> cases waitPe22 <;> cases handoffNe <;> cases handoffPe <;>
    exact ⟨_, hEnc _ _ rfl, hDec _ _ _ rfl, hDec _ _ _ rfl, hDec _ _ _ rfl, hDec _ _ _ rfl,
              hDec _ _ _ rfl, hDec _ _ _ rfl, hDec _ _ _ rfl, hDec _ _ _ rfl, hDec _ _ _ rfl⟩

/-- `encodeResPortCfgMulti` ALWAYS succeeds for every boolean combination and decodes back through
    `resCfgMultiLayout` to exact verifier-compliant bits (`P=1`, `dmaFlag=1`, `isL2 → direct=0`,
    and first-task wait mask bits!). -/
theorem encodeResPortCfgMulti_succeeds (isFirst isL2 direct : Bool) :
    ∃ w, encodeResPortCfgMulti isFirst isL2 direct = .ok w ∧
      decodeFieldChecked resCfgMultiLayout w "P" = .ok 1 ∧
      decodeFieldChecked resCfgMultiLayout w "dmaFlag" = .ok 1 ∧
      decodeFieldChecked resCfgMultiLayout w "mode" = .ok (if isL2 then portModeL2 else portModeDma) ∧
      decodeFieldChecked resCfgMultiLayout w "direct" = .ok (if !isL2 && direct then 1 else 0) ∧
      decodeFieldChecked resCfgMultiLayout w "waitNe4" = .ok (if isFirst then 1 else 0) ∧
      decodeFieldChecked resCfgMultiLayout w "waitNe5" = .ok (if isFirst then 1 else 0) ∧
      decodeFieldChecked resCfgMultiLayout w "waitPe20" = .ok (if isFirst then 1 else 0) ∧
      decodeFieldChecked resCfgMultiLayout w "waitPe22" = .ok (if isFirst then 1 else 0) := by
  cases isFirst <;> cases isL2 <;> cases direct <;>
    exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- `encodeSrcDepMulti` succeeds for any hardware-legal `(taskId < 256, depKind ≤ 2, depDist < 16)`. -/
theorem encodeSrcDepMulti_succeeds (name : String) (taskId depKind depDist : Nat)
    (ht : taskId < 256) (hk : depKind ≤ 2) (hd : depDist < 16) :
    encodeSrcDepMulti name taskId depKind depDist =
      .ok (((((0 : UInt32) ||| (srcDepCodeDma <<< 0)) ||| (taskId.toUInt32 <<< 16)) |||
        (depDist.toUInt32 <<< 24)) ||| (depKind.toUInt32 <<< 28)) := by
  unfold encodeSrcDepMulti
  have h1 : natToU32Checked taskId s!"{name} task" = .ok taskId.toUInt32 :=
    natToU32Checked_ok_of_lt _ _ (by unfold maxU32Nat; omega)
  have h2 : natToU32Checked depKind s!"{name} depKind" = .ok depKind.toUInt32 :=
    natToU32Checked_ok_of_lt _ _ (by unfold maxU32Nat; omega)
  have h3 : natToU32Checked depDist s!"{name} depDist" = .ok depDist.toUInt32 :=
    natToU32Checked_ok_of_lt _ _ (by unfold maxU32Nat; omega)
  simp only [bind, Except.bind, h1, h2, h3]
  apply encodeWord_name_irrel (l1 := srcDepMultiLayout "src1Dep") (l2 := srcDepMultiLayout name) rfl rfl rfl
  have hT : ¬ (256 ≤ taskId % 4294967296) := by omega
  have hD : ¬ (16 ≤ depDist % 4294967296) := by omega
  have hK : ¬ (4 ≤ depKind % 4294967296) := by omega
  unfold encodeWord
  have hVL : validateWordLayout (srcDepMultiLayout "src1Dep") = .ok 0x3fff00ff := rfl
  have hVE : validateEncodeValues "src1Dep" (srcDepMultiLayout "src1Dep").fields.toList
      [("code", srcDepCodeDma), ("task", taskId.toUInt32), ("depDist", depDist.toUInt32), ("depKind", depKind.toUInt32)] [] = .ok () := by
    simp [srcDepMultiLayout, validateEncodeValues]
  rw [show (srcDepMultiLayout "src1Dep").name = "src1Dep" from rfl]
  simp only [bind, Except.bind, hVL, hVE]
  simp [srcDepMultiLayout, encodeWordFields, srcDepCodeDma, hT, hD, hK]

/-- `encodeKernelStageMulti` succeeds for any `taskId < 255` (the exact `__KERN_0` hardware limit!). -/
theorem encodeKernelStageMulti_succeeds (taskId : Nat) (isLast isSparse : Bool)
    (ht : taskId < 255) :
    (encodeKernelStageMulti taskId isLast isSparse).isOk = true := by
  unfold encodeKernelStageMulti
  have h1 : natToU32Checked (taskId + 1) "kernelStage.stage" = .ok (taskId + 1).toUInt32 :=
    natToU32Checked_ok_of_lt _ _ (by unfold maxU32Nat; omega)
  simp only [bind, Except.bind, h1]
  have hS : ¬ (256 ≤ (taskId + 1) % 4294967296) := by omega
  have hVL : ∃ m, validateWordLayout kernelStageMultiLayout = .ok m := ⟨_, rfl⟩
  obtain ⟨m, hm⟩ := hVL
  have hEnc : encodeWord kernelStageMultiLayout #[
      ("sparse", if isSparse then 1 else 0), ("kernEn", 1),
      ("isLast", if isLast then 1 else 0), ("kernCfg", 1), ("stage", (taskId + 1).toUInt32)
    ] = .ok ((((((0 : UInt32) ||| ((if isSparse then (1 : UInt32) else 0) <<< 5)) |||
      ((1 : UInt32) <<< 6)) ||| ((if isLast then (1 : UInt32) else 0) <<< 7)) |||
      ((1 : UInt32) <<< 9)) ||| ((taskId + 1).toUInt32 <<< 16)) := by
    unfold encodeWord
    have hVE : validateEncodeValues kernelStageMultiLayout.name kernelStageMultiLayout.fields.toList
        [("sparse", if isSparse then 1 else 0), ("kernEn", 1), ("isLast", if isLast then 1 else 0),
         ("kernCfg", 1), ("stage", (taskId + 1).toUInt32)] [] = .ok () := by
      simp [kernelStageMultiLayout, validateEncodeValues]
    simp only [bind, Except.bind, hm, hVE]
    cases isSparse <;> cases isLast <;>
      simp [kernelStageMultiLayout, encodeWordFields, hS]
  rw [hEnc]; rfl

/-- `outputStageWrites` succeeds for every `taskId < 256` and emits all 6 exact output stage registers
    `regOut0 .. regOut5`. -/
theorem outputStageWrites_succeeds (taskId : Nat) (ht : taskId < 256) :
    ∃ ws, outputStageWrites taskId = .ok ws ∧ ws.size = 6 ∧
      ws[0]!.byteAddr = regOut0 ∧ ws[1]!.byteAddr = regOut1 ∧ ws[2]!.byteAddr = regOut2 ∧
      ws[3]!.byteAddr = regOut3 ∧ ws[4]!.byteAddr = regOut4 ∧ ws[5]!.byteAddr = regOut5 := by
  unfold outputStageWrites
  have h1 : natToU32Checked taskId "out2 taskId" = .ok taskId.toUInt32 :=
    natToU32Checked_ok_of_lt _ _ (by unfold maxU32Nat; omega)
  simp only [bind, Except.bind, pure, Except.pure, h1]
  have hT : ¬ (256 ≤ taskId % 4294967296) := by omega
  have hVL : ∃ m, validateWordLayout out2Layout = .ok m := ⟨_, rfl⟩
  obtain ⟨m, hm⟩ := hVL
  have hEnc : encodeWord out2Layout #[("form", out2FormValue), ("task", taskId.toUInt32)] =
      .ok (((0 : UInt32) ||| (out2FormValue <<< 0)) ||| (taskId.toUInt32 <<< 16)) := by
    unfold encodeWord
    have hVE : validateEncodeValues out2Layout.name out2Layout.fields.toList
        [("form", out2FormValue), ("task", taskId.toUInt32)] [] = .ok () := by
      simp [out2Layout, validateEncodeValues]
    simp only [bind, Except.bind, hm, hVE]
    simp [out2Layout, encodeWordFields, out2FormValue, hT]
  rw [hEnc]
  exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

-- ============================================================================
-- 4B. Engine Family 1: `neConv` (fp16 & quantized) → Verifier Soundness
-- ============================================================================

/-- Every valid `QuantFormat` (`fmt.isValid = true`) satisfies `Vane.Verify.isLegalNeWeightFormat`! -/
theorem quantFormat_isLegalNeWeightFormat (fmt : QuantFormat) (hv : fmt.isValid = true) :
    isLegalNeWeightFormat fmt.wKind fmt.wWidth = true := by
  cases fmt with
  | int8 => decide
  | uint8 => decide
  | int4 => decide
  | uint4 => decide
  | pal bits =>
    unfold QuantFormat.isValid at hv
    simp only [Bool.or_eq_true, beq_iff_eq] at hv
    rcases hv with (((rfl | rfl) | rfl) | rfl) | rfl <;> decide

/-- Every valid `QuantFormat` has `wKind < 8` and `wWidth < 16` (strictly within `neCfgQuantLayout` bitfields). -/
theorem quantFormat_field_bounds (fmt : QuantFormat) (hv : fmt.isValid = true) :
    fmt.wKind < 8 ∧ fmt.wWidth < 16 := by
  cases fmt with
  | int8 => decide
  | uint8 => decide
  | int4 => decide
  | uint4 => decide
  | pal bits =>
    unfold QuantFormat.isValid at hv
    simp only [Bool.or_eq_true, beq_iff_eq] at hv
    rcases hv with (((rfl | rfl) | rfl) | rfl) | rfl <;> decide

/-- fp16 `neCfgLayout target` word (`0x4900`) is legal under `Vane.Verify`'s unified `neCfgQuantLayout`:
    its `wKind` is `wKindFp16`, `wWidth = 8`, `dynKern = 0`, `hasZP = 0`, and
    `isLegalNeWeightFormat wKindFp16 8 = true`! -/
theorem neConv_fp16_neCfg_legal (target : Target) (sparse : Bool) :
    ∃ w, encodeWord (neCfgLayout target) #[("sparse", if sparse then 1 else 0)] = .ok w ∧
      decodeFieldChecked neCfgQuantLayout w "wKind" = .ok wKindFp16 ∧
      decodeFieldChecked neCfgQuantLayout w "wWidth" = .ok 8 ∧
      decodeFieldChecked neCfgQuantLayout w "dynKern" = .ok 0 ∧
      decodeFieldChecked neCfgQuantLayout w "hasZP" = .ok 0 ∧
      decodeFieldChecked neCfgQuantLayout w "sparse" = .ok (if sparse then 1 else 0) ∧
      isLegalNeWeightFormat wKindFp16 8 = true := by
  cases target <;> cases sparse <;>
    exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, by decide⟩

/-- Quantized `neCommon` bit 27 (`neCommonQuantBit27`) is always in `{0, 1}` (`< 2`) and
    strictly cleared whenever `hasZeroPoint = true` or `wKind` is unsigned (`uint8` / `uint4`)
    (matching `Vane.Verify`!). -/
theorem neConv_quant_bit27_legal (wide3x3 hasZP : Bool) (wKind : UInt32) :
    neCommonQuantBit27 wide3x3 hasZP wKind < 2 ∧
    ((hasZP = true ∨ isUnsignedWeightKind wKind = true) →
      neCommonQuantBit27 wide3x3 hasZP wKind = 0) := by
  unfold neCommonQuantBit27
  constructor
  · split <;> decide
  · intro h
    split
    · rename_i hc
      simp only [Bool.and_eq_true, Bool.not_eq_true'] at hc
      rcases h with hzp | hu
      · rw [hzp] at hc; cases hc.1.2
      · rw [hu] at hc; cases hc.2
    · rfl

/-- `neTile` work-unit invariant: whenever `f + fat ≤ logNE` and `logNE - f - fat ≤ 3`,
    the emitted `WU = t.wu` satisfies `Vane.Verify`'s exact check `WU + F + fat = logNE ∧ WU ≤ 3`. -/
theorem neConv_neTile_wu_invariant (t : Tiling) (hSum : t.f + t.fat ≤ t.logNE)
    (hWu : t.logNE - t.f - t.fat ≤ 3) :
    t.wu ≤ 3 ∧ t.wu + t.f + t.fat = t.logNE := by
  unfold Tiling.wu
  omega

/-- `ConvSpec` 1×1 `convCfg` field bounds: for any 1×1 `ConvSpec` with positive dims,
    `(kw=1, kh=1, sx=1, sy=1, pl=0, pt=0, hx=1, vy=1)` and `outH = h > 0`, `outW = w > 0`. -/
theorem neConv_1x1_convCfg_legal (sh : ConvSpec)
    (hk : sh.kernelSize = 1) (hs : sh.stride = 1)
    (hh : sh.h > 0) (hw : sh.w > 0) :
    sh.padLeft = 0 ∧ sh.padTop = 0 ∧ sh.outH = sh.h ∧ sh.outW = sh.w ∧
    sh.outH > 0 ∧ sh.outW > 0 ∧
    ∃ wCfg, encodeWord convCfgLayout #[
      ("kw", sh.kernelSize.toUInt32), ("kh", sh.kernelSize.toUInt32),
      ("sx", sh.stride.toUInt32), ("sy", sh.stride.toUInt32),
      ("pl", sh.padLeft.toUInt32), ("pt", sh.padTop.toUInt32),
      ("hx", 1), ("vy", 1)
    ] = .ok wCfg ∧
      decodeFieldChecked convCfgLayout wCfg "kw" = .ok 1 ∧
      decodeFieldChecked convCfgLayout wCfg "kh" = .ok 1 ∧
      decodeFieldChecked convCfgLayout wCfg "sx" = .ok 1 ∧
      decodeFieldChecked convCfgLayout wCfg "sy" = .ok 1 ∧
      decodeFieldChecked convCfgLayout wCfg "pl" = .ok 0 ∧
      decodeFieldChecked convCfgLayout wCfg "pt" = .ok 0 := by
  have hpl : sh.padLeft = 0 := by unfold ConvSpec.padLeft; simp [hk]
  have hpt : sh.padTop = 0 := by unfold ConvSpec.padTop; simp [hk]
  have hoH : sh.outH = sh.h := by unfold ConvSpec.outH; simp [hk]
  have hoW : sh.outW = sh.w := by unfold ConvSpec.outW; simp [hk]
  refine ⟨hpl, hpt, hoH, hoW, by omega, by omega, ?_⟩
  rw [hk, hs, hpl, hpt]
  exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- `ConvSpec` 3×3 `convCfg` round-trip for all legal padding modes (`.same`, `.valid`, `.strip pt pb`
    with `pt, pb ≤ 1`) and `stride ∈ {1, 2}`. -/
theorem neConv_3x3_convCfg_legal (stride pl pt : Nat)
    (hs : stride = 1 ∨ stride = 2) (hpl : pl ≤ 1) (hpt : pt ≤ 1) :
    ∃ wCfg, encodeWord convCfgLayout #[
      ("kw", 3), ("kh", 3), ("sx", stride.toUInt32), ("sy", stride.toUInt32),
      ("pl", pl.toUInt32), ("pt", pt.toUInt32), ("hx", 1), ("vy", 1)
    ] = .ok wCfg ∧
      decodeFieldChecked convCfgLayout wCfg "kw" = .ok 3 ∧
      decodeFieldChecked convCfgLayout wCfg "kh" = .ok 3 ∧
      decodeFieldChecked convCfgLayout wCfg "sx" = .ok stride.toUInt32 ∧
      decodeFieldChecked convCfgLayout wCfg "sy" = .ok stride.toUInt32 ∧
      decodeFieldChecked convCfgLayout wCfg "pl" = .ok pl.toUInt32 ∧
      decodeFieldChecked convCfgLayout wCfg "pt" = .ok pt.toUInt32 := by
  rcases hs with rfl | rfl <;> (
    have hplC : pl = 0 ∨ pl = 1 := by omega
    have hptC : pt = 0 ∨ pt = 1 := by omega
    rcases hplC with rfl | rfl <;> rcases hptC with rfl | rfl <;>
      exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩)

/-- `ConvKernCacheKey` equality soundness: if two `neConv` tasks match on `ConvKernCacheKey`
    and `(weight, bias, lut)` payloads, then all kernel-packing parameters (`kernelSize`, `mode`,
    `cin`, `cout`, `stride`, `hasBias`, `logNE`, `f`, `fat`, `ocg`) are identical,
    guaranteeing bit-identical `__KERN_0` chunk payloads! -/
theorem convKernCacheKey_sound (k1 k2 : ConvKernCacheKey) (hEq : (k1 == k2) = true) :
    k1.weightGroupId = k2.weightGroupId ∧
    k1.kernelSize = k2.kernelSize ∧
    k1.mode = k2.mode ∧
    k1.cin = k2.cin ∧
    k1.cout = k2.cout ∧
    k1.stride = k2.stride ∧
    k1.hasBias = k2.hasBias ∧
    k1.logNE = k2.logNE ∧
    k1.f = k2.f ∧
    k1.fat = k2.fat ∧
    k1.ocg = k2.ocg := by
  cases k1 with
  | mk a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 =>
    cases k2 with
    | mk b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 =>
      simp only [BEq.beq, instBEqConvKernCacheKey.beq, Bool.and_eq_true] at hEq
      rcases hEq with ⟨h1, h2, h3, h4, h5, h6, _, h8, _, h10, h11, h12, h13⟩
      have hMode : a3 = b3 := by cases a3 <;> cases b3 <;> first | rfl | (exfalso; exact Bool.noConfusion h3)
      have hBias : a8 = b8 := by cases a8 <;> cases b8 <;> first | rfl | (exfalso; exact Bool.noConfusion h8)
      exact ⟨Nat.eq_of_beq_eq_true h1, Nat.eq_of_beq_eq_true h2, hMode,
             Nat.eq_of_beq_eq_true h4, Nat.eq_of_beq_eq_true h5, Nat.eq_of_beq_eq_true h6, hBias,
             Nat.eq_of_beq_eq_true h10, Nat.eq_of_beq_eq_true h11, Nat.eq_of_beq_eq_true h12,
             Nat.eq_of_beq_eq_true h13⟩

/-- Direct soundness of `encodeNeConvQuantCfgWord` (called inside `emitNeConvTask`):
    for ANY target, valid `QuantFormat` (`fmt.isValid = true`), and `hasZeroPoint` obeying the
    verifier rule `fmt.isPalette = true → hasZeroPoint = false`, `encodeNeConvQuantCfgWord`
    succeeds and satisfies EVERY `neCfg` (`0x4900`) check in `Vane.Verify` (`wKind`, `wWidth`,
    `isLegalNeWeightFormat`, `sparse = 0`, `dynKern = 0`, `hasZP`, and `h17Flag`)! -/
theorem encodeNeConvQuantCfgWord_verifier_sound
    (target : Target) (fmt : QuantFormat) (hFmt : fmt.isValid = true)
    (hasZeroPoint : Bool) (hNoPalZP : fmt.isPalette = true → hasZeroPoint = false) :
    ∃ w, encodeNeConvQuantCfgWord target fmt.wKind fmt.wWidth hasZeroPoint = .ok w ∧
      decodeFieldChecked neCfgQuantLayout w "wKind" = .ok fmt.wKind ∧
      decodeFieldChecked neCfgQuantLayout w "wWidth" = .ok fmt.wWidth.toUInt32 ∧
      decodeFieldChecked neCfgQuantLayout w "sparse" = .ok 0 ∧
      decodeFieldChecked neCfgQuantLayout w "dynKern" = .ok 0 ∧
      decodeFieldChecked neCfgQuantLayout w "hasZP" = .ok (if hasZeroPoint then 1 else 0) ∧
      decodeFieldChecked neCfgQuantLayout w "h17Flag" =
        .ok (if target == .h17 && !hasZeroPoint then 1 else 0) ∧
      isLegalNeWeightFormat fmt.wKind fmt.wWidth = true ∧
      (fmt.wKind = wKindPal → hasZeroPoint = false) := by
  cases fmt with
  | int8 =>
    cases target <;> cases hasZeroPoint <;>
      exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl, by decide, fun h => by cases h⟩
  | uint8 =>
    cases target <;> cases hasZeroPoint <;>
      exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl, by decide, fun h => by cases h⟩
  | int4 =>
    cases target <;> cases hasZeroPoint <;>
      exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl, by decide, fun h => by cases h⟩
  | uint4 =>
    cases target <;> cases hasZeroPoint <;>
      exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl, by decide, fun h => by cases h⟩
  | pal bits =>
    have hZPF : hasZeroPoint = false := hNoPalZP rfl
    subst hZPF
    unfold QuantFormat.isValid at hFmt
    simp only [Bool.or_eq_true, beq_iff_eq] at hFmt
    rcases hFmt with (((rfl | rfl) | rfl) | rfl) | rfl <;>
      (cases target <;> exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl, by decide, fun _ => rfl⟩)

/-- Direct soundness of `encodeNeConvQuantOpWord` (called inside `emitNeConvTask`):
    succeeds for all boolean flags and decodes cleanly to `Vane.Verify`'s `neOpQuantLayout`. -/
theorem encodeNeConvQuantOpWord_verifier_sound (biasInKernel isPalette perLaneScale isRelu : Bool) :
    ∃ w, encodeNeConvQuantOpWord biasInKernel isPalette perLaneScale isRelu = .ok w ∧
      decodeFieldChecked neOpQuantLayout w "biasInKernel" = .ok (if biasInKernel then 1 else 0) ∧
      decodeFieldChecked neOpQuantLayout w "unscaled" = .ok (if isPalette then neOpUnscaledBits else 0) ∧
      decodeFieldChecked neOpQuantLayout w "perLaneScale" = .ok (if perLaneScale then 1 else 0) ∧
      decodeFieldChecked neOpQuantLayout w "relu" = .ok (if isRelu then 1 else 0) ∧
      decodeFieldChecked neOpQuantLayout w "lut" = .ok 0 ∧
      decodeFieldChecked neOpQuantLayout w "base20" = .ok 1 := by
  cases biasInKernel <;> cases isPalette <;> cases perLaneScale <;> cases isRelu <;>
    exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Direct soundness of `encodeNeConvCommonWord` (called inside `emitNeConvTask` & `emitNeActTask`):
    for any verifier-legal shifts `shPref ≤ shMax ≤ 4`, `f ≤ 4`, and `bit27Val < 2`,
    succeeds and decodes `kind = 0`, `shMin = shPref ≤ shMax ≤ 4`, `F = f`, `wide3x3 = bit27Val`. -/
theorem encodeNeConvCommonWord_verifier_sound
    (shPref shMax f : Nat) (bit27Val : UInt32)
    (hSh : shPref ≤ shMax) (hMax : shMax ≤ 4) (hf : f ≤ 4) (hb : bit27Val < 2) :
    ∃ w, encodeNeConvCommonWord shPref shMax f bit27Val = .ok w ∧
      decodeFieldChecked neCommonLayout w "kind" = .ok 0 ∧
      decodeFieldChecked neCommonLayout w "shPref" = .ok shPref.toUInt32 ∧
      decodeFieldChecked neCommonLayout w "shMin" = .ok shPref.toUInt32 ∧
      decodeFieldChecked neCommonLayout w "shMax" = .ok shMax.toUInt32 ∧
      decodeFieldChecked neCommonLayout w "F" = .ok f.toUInt32 ∧
      decodeFieldChecked neCommonLayout w "wide3x3" = .ok bit27Val := by
  have hb01 : bit27Val = 0 ∨ bit27Val = 1 := by
    have hlt : bit27Val.toNat < 2 := hb
    rcases (show bit27Val.toNat = 0 ∨ bit27Val.toNat = 1 by omega) with h0 | h1
    · left; exact UInt32.toNat_inj.mp h0
    · right; exact UInt32.toNat_inj.mp h1
  have hsp : shPref = 0 ∨ shPref = 1 ∨ shPref = 2 ∨ shPref = 3 ∨ shPref = 4 := by omega
  have hsm : shMax = 0 ∨ shMax = 1 ∨ shMax = 2 ∨ shMax = 3 ∨ shMax = 4 := by omega
  have hfc : f = 0 ∨ f = 1 ∨ f = 2 ∨ f = 3 ∨ f = 4 := by omega
  rcases hsp with rfl | rfl | rfl | rfl | rfl <;>
  rcases hsm with rfl | rfl | rfl | rfl | rfl <;>
  rcases hfc with rfl | rfl | rfl | rfl | rfl <;>
  rcases hb01 with rfl | rfl <;>
    exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Direct soundness of `encodeNeConvTileWord` (called inside `emitNeConvTask`, `emitNeActTask`,
    and `emitDmaViewTask`): for `ocg ≤ 4`, `fat ≤ 1`, `wu ≤ 3`, succeeds and decodes identically. -/
theorem encodeNeConvTileWord_verifier_sound
    (ocg fat wu : Nat) (hocg : ocg ≤ 4) (hfat : fat ≤ 1) (hwu : wu ≤ 3) :
    ∃ w, encodeNeConvTileWord ocg fat wu = .ok w ∧
      decodeFieldChecked neTileLayout w "ocg" = .ok ocg.toUInt32 ∧
      decodeFieldChecked neTileLayout w "fat" = .ok fat.toUInt32 ∧
      decodeFieldChecked neTileLayout w "WU" = .ok wu.toUInt32 := by
  have ho : ocg = 0 ∨ ocg = 1 ∨ ocg = 2 ∨ ocg = 3 ∨ ocg = 4 := by omega
  have hf : fat = 0 ∨ fat = 1 := by omega
  have hw : wu = 0 ∨ wu = 1 ∨ wu = 2 ∨ wu = 3 := by omega
  rcases ho with rfl | rfl | rfl | rfl | rfl <;>
  rcases hf with rfl | rfl <;>
  rcases hw with rfl | rfl | rfl | rfl <;>
    exact ⟨_, rfl, rfl, rfl, rfl⟩

private theorem checkConvDenseChunks64KLoop_sound
    (sh : ConvSpec) (t : Tiling) (extraChunkBytes : Nat) (hasBiasInKernel : Bool)
    (groups : Nat) :
    ∀ rem g0, rem = groups - g0 →
      checkConvDenseChunks64KLoop sh t extraChunkBytes hasBiasInKernel groups g0 = .ok () →
      ∀ g, g0 ≤ g → g < groups →
        (groupOutputBlocksConv sh t g).foldl (fun (acc : Nat) (_, n) => acc + n) 0 > 0 →
        convDenseGroupChunkBytes sh t extraChunkBytes hasBiasInKernel g % 64 = 0 ∧
        convDenseGroupChunkBytes sh t extraChunkBytes hasBiasInKernel g ≤ maxKernelChunkBytes := by
  intro rem
  induction rem with
  | zero =>
    intro g0 hrem _ g hge hlt
    omega
  | succ r ih =>
    intro g0 hrem hOk g hge hlt hPos
    have hg0Lt : g0 < groups := by omega
    unfold checkConvDenseChunks64KLoop at hOk
    simp only [hg0Lt, ↓reduceIte] at hOk
    by_cases hEq : g = g0
    · subst hEq
      simp only [hPos, ↓reduceIte] at hOk
      split at hOk
      · cases hOk
      · rename_i hLe
        exact ⟨alignUp_dvd_64 _, by omega⟩
    · have hgt : g0 + 1 ≤ g := by omega
      split at hOk
      · split at hOk
        · cases hOk
        · exact ih (g0 + 1) (by omega) hOk g hgt hlt hPos
      · exact ih (g0 + 1) (by omega) hOk g hgt hlt hPos

/-- Direct soundness of `Vane.Tiling.checkConvDenseChunks64K` (called by `chooseConvTaskPlan`
    and `ChannelSplit`): whenever `sh.stride = 1` and `checkConvDenseChunks64K = .ok ()`,
    every non-empty output group's `__KERN_0` chunk size is 64B-aligned AND `≤ maxKernelChunkBytes` (64 KiB)! -/
theorem checkConvDenseChunks64K_sound
    (sh : ConvSpec) (t : Tiling) (extraChunkBytes : Nat) (hasBiasInKernel : Bool)
    (hStride : sh.stride = 1)
    (hOk : checkConvDenseChunks64K sh t extraChunkBytes hasBiasInKernel = .ok ()) :
    ∀ g < (1 : Nat) <<< t.f,
      (groupOutputBlocksConv sh t g).foldl (fun (acc : Nat) (_, n) => acc + n) 0 > 0 →
      convDenseGroupChunkBytes sh t extraChunkBytes hasBiasInKernel g % 64 = 0 ∧
      convDenseGroupChunkBytes sh t extraChunkBytes hasBiasInKernel g ≤ maxKernelChunkBytes := by
  unfold checkConvDenseChunks64K at hOk
  simp only [hStride, beq_self_eq_true, ↓reduceIte] at hOk
  intro g hg hPos
  exact checkConvDenseChunks64KLoop_sound sh t extraChunkBytes hasBiasInKernel
    ((1 : Nat) <<< t.f) ((1 : Nat) <<< t.f) 0 (by omega) hOk g (Nat.zero_le _) hg hPos

/-- `validateConvOcgSafety` guarantees `ocg = 0` for depthwise and `ocg ≤ 4` (strictly within
    target profile limits) for ALL targets and conv modes! -/
theorem convOcgSafety_bounds (target : Target) (sh : ConvSpec) (t : Tiling)
    (hOk : validateConvOcgSafety target sh t = .ok ()) :
    t.ocg ≤ 4 ∧ (sh.effMode = .depthwise → t.ocg = 0) := by
  unfold validateConvOcgSafety at hOk
  cases hMode : sh.effMode with
  | depthwise =>
    rw [hMode] at hOk
    dsimp only at hOk
    split at hOk
    · cases hOk
    · rename_i hne
      simp only [bne_iff_ne, Classical.not_not] at hne
      exact ⟨by omega, fun _ => hne⟩
  | dense =>
    rw [hMode] at hOk
    dsimp only at hOk
    by_cases hgt : t.ocg > (if sh.kernelSize == 3 && sh.stride == 1 then
        target.profile.dense3x3Stride1MaxOcg else 4)
    · simp only [hgt, ↓reduceIte] at hOk
      cases hOk
    · have hMaxLe4 : (if sh.kernelSize == 3 && sh.stride == 1 then
          target.profile.dense3x3Stride1MaxOcg else 4) ≤ 4 := by
        split
        · cases target <;> decide
        · omega
      exact ⟨by omega, fun h => by cases h⟩

-- ============================================================================
-- 4C. Engine Family 2 & 3: `pe` & `neAct` (LUT / Pool / Reduce) → Verifier Soundness
-- ============================================================================

/-- `peTileM` matches `Vane.Verify`'s exact check `expM = min (maxK - k) (ceilLog2 th)` and fits in 4 bits. -/
theorem pe_peTileM_matches_verifier (twoSource : Bool) (k th : Nat)
    (hkLo : 4 ≤ k) (hkHi : k ≤ peMaxK twoSource) :
    peTileM twoSource k th = min ((if twoSource then 8 else 9) - k) (ceilLog2 th) ∧
    peTileM twoSource k th < 16 ∧ k < 16 := by
  unfold peTileM peMaxK at *
  refine ⟨rfl, ?_, ?_⟩
  · have hMin : min ((if twoSource then 8 else 9) - k) (ceilLog2 th) ≤ (if twoSource then 8 else 9) - k :=
      Nat.min_le_left _ _
    have hBound : (if twoSource then 8 else 9) - k < 16 := by
      cases twoSource <;> simp only [↓reduceIte, Bool.false_eq_true] at hkHi ⊢ <;> omega
    exact Nat.lt_of_le_of_lt hMin hBound
  · cases twoSource <;> simp only [↓reduceIte, Bool.false_eq_true] at hkHi ⊢ <;> omega

/-- `broadcastDims` legality: whenever `broadcastDims s1 s2 = .ok out`, the emitted `peBcast` flags
    satisfy BOTH `Vane.Verify` broadcast invariants:
    1. never sets a broadcast flag on an axis of size `1`,
    2. never broadcasts both sources on the same axis! -/
theorem pe_broadcastDims_verifier_legal (s1 s2 out : TensorDims)
    (hOk : broadcastDims s1 s2 = .ok out) :
    let b1W := if s1.w != out.w then 1 else 0
    let b1H := if s1.h != out.h then 1 else 0
    let b1C := if s1.c != out.c then 1 else 0
    let b2W := if s2.w != out.w then 1 else 0
    let b2H := if s2.h != out.h then 1 else 0
    let b2C := if s2.c != out.c then 1 else 0
    (b1W = 1 → out.w > 1) ∧ (b1H = 1 → out.h > 1) ∧ (b1C = 1 → out.c > 1) ∧
    (b2W = 1 → out.w > 1) ∧ (b2H = 1 → out.h > 1) ∧ (b2C = 1 → out.c > 1) ∧
    ¬ (b1W = 1 ∧ b2W = 1) ∧ ¬ (b1H = 1 ∧ b2H = 1) ∧ ¬ (b1C = 1 ∧ b2C = 1) := by
  have hValAxis : ∀ a b lbl ax, validateBroadcastAxis a b lbl ax = .ok () → a = 1 ∨ a = b := by
    intro a b lbl ax h
    unfold validateBroadcastAxis at h
    split at h
    · cases h
    · rename_i hc
      by_cases h1 : a = 1
      · exact Or.inl h1
      · by_cases h2 : a = b
        · exact Or.inr h2
        · exfalso; apply hc
          simp [h1, h2]
  have hValDims : ∀ d lbl, validateTensorDims d lbl = .ok () → d.c > 0 ∧ d.h > 0 ∧ d.w > 0 := by
    intro d lbl h
    unfold validateTensorDims at h
    simp only [bind, Except.bind, pure, Except.pure] at h
    split at h; · cases h
    split at h; · cases h
    split at h; · cases h
    rename_i hw hh hc
    simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, not_or] at hw hh hc
    omega
  have hAxisRule : ∀ a b : Nat, a > 0 → b > 0 → (a = 1 ∨ a = max a b) → (b = 1 ∨ b = max a b) →
      ((if a != max a b then 1 else 0) = 1 → max a b > 1) ∧
      ((if b != max a b then 1 else 0) = 1 → max a b > 1) ∧
      ¬ ((if a != max a b then 1 else 0) = 1 ∧ (if b != max a b then 1 else 0) = 1) := by
    intro a b ha hb h1 h2
    refine ⟨?_, ?_, ?_⟩
    · intro h
      by_cases he : a = max a b
      · have hne : (a != max a b) = false := by rw [← he]; simp
        rw [hne] at h; simp at h
      · omega
    · intro h
      by_cases he : b = max a b
      · have hne : (b != max a b) = false := by rw [← he]; simp
        rw [hne] at h; simp at h
      · omega
    · intro ⟨hA, hB⟩
      by_cases heA : a = max a b
      · have hne : (a != max a b) = false := by rw [← heA]; simp
        rw [hne] at hA; simp at hA
      · by_cases heB : b = max a b
        · have hne : (b != max a b) = false := by rw [← heB]; simp
          rw [hne] at hB; simp at hB
        · omega
  unfold broadcastDims at hOk
  simp only [bind, Except.bind, pure, Except.pure] at hOk
  cases hv1 : validateTensorDims s1 "src1" with
  | error _ => rw [hv1] at hOk; cases hOk
  | ok _ =>
    rw [hv1] at hOk
    cases hv2 : validateTensorDims s2 "src2" with
    | error _ => rw [hv2] at hOk; cases hOk
    | ok _ =>
      rw [hv2] at hOk
      cases ha1 : validateBroadcastAxis s1.c (max s1.c s2.c) "src1" "C" with
      | error _ => rw [ha1] at hOk; cases hOk
      | ok _ =>
        rw [ha1] at hOk
        cases ha2 : validateBroadcastAxis s1.h (max s1.h s2.h) "src1" "H" with
        | error _ => rw [ha2] at hOk; cases hOk
        | ok _ =>
          rw [ha2] at hOk
          cases ha3 : validateBroadcastAxis s1.w (max s1.w s2.w) "src1" "W" with
          | error _ => rw [ha3] at hOk; cases hOk
          | ok _ =>
            rw [ha3] at hOk
            cases ha4 : validateBroadcastAxis s2.c (max s1.c s2.c) "src2" "C" with
            | error _ => rw [ha4] at hOk; cases hOk
            | ok _ =>
              rw [ha4] at hOk
              cases ha5 : validateBroadcastAxis s2.h (max s1.h s2.h) "src2" "H" with
              | error _ => rw [ha5] at hOk; cases hOk
              | ok _ =>
                rw [ha5] at hOk
                cases ha6 : validateBroadcastAxis s2.w (max s1.w s2.w) "src2" "W" with
                | error _ => rw [ha6] at hOk; cases hOk
                | ok _ =>
                  rw [ha6] at hOk
                  simp only [Except.ok.injEq] at hOk
                  subst hOk
                  have ⟨hc1, hh1, hw1⟩ := hValDims s1 "src1" hv1
                  have ⟨hc2, hh2, hw2⟩ := hValDims s2 "src2" hv2
                  have rc := hAxisRule s1.c s2.c hc1 hc2 (hValAxis _ _ _ _ ha1) (hValAxis _ _ _ _ ha4)
                  have rh := hAxisRule s1.h s2.h hh1 hh2 (hValAxis _ _ _ _ ha2) (hValAxis _ _ _ _ ha5)
                  have rw' := hAxisRule s1.w s2.w hw1 hw2 (hValAxis _ _ _ _ ha3) (hValAxis _ _ _ _ ha6)
                  exact ⟨rw'.1, rh.1, rc.1, rw'.2.1, rh.2.1, rc.2.1, rw'.2.2, rh.2.2, rc.2.2⟩

/-- Every `PeBinOp.regValue` matches one of the 5 verifier-accepted `regPeOp` encodings. -/
theorem peBinOp_regValue_valid (op : PeBinOp) :
    op.regValue = peOpAddValue ∨ op.regValue = peOpSubValue ∨ op.regValue = peOpMulValue ∨
    op.regValue = peOpMaxValue ∨ op.regValue = peOpMinValue := by
  cases op <;> simp [PeBinOp.regValue]

/-- PE anti-hang L2 channel stride invariant: `2 * min (alignUp w 8) (max (peStripWidth k) 32)`
    is always at least `2 * min (alignUp w 8) 32` (never below 64B when `w ≥ 32`) and even. -/
theorem pe_stage_chanStride_antihang (w k : Nat) :
    let chs := 2 * min (alignUp w 8) (max (peStripWidth k) 32)
    chs % 2 = 0 ∧ (w ≥ 32 → chs ≥ 64) := by
  have hGe : w ≤ alignUp w 8 := alignUp_ge_local w 8
  dsimp only
  constructor
  · omega
  · intro hw; omega

/-- Direct soundness of `encodePeTileWord` (called inside `emitPeTask`): for any legal
    `4 ≤ k ≤ peMaxK twoSource`, succeeds and decodes `(k, peTileM twoSource k th)` identically. -/
theorem encodePeTileWord_verifier_sound (twoSource : Bool) (k th : Nat)
    (hkLo : 4 ≤ k) (hkHi : k ≤ peMaxK twoSource) :
    ∃ w, encodePeTileWord twoSource k th = .ok w ∧
      decodeFieldChecked peTileLayout w "k" = .ok k.toUInt32 ∧
      decodeFieldChecked peTileLayout w "m" = .ok (peTileM twoSource k th).toUInt32 := by
  have hmBound : peTileM twoSource k th ≤ 5 := by
    unfold peTileM peMaxK at *
    have hMin := Nat.min_le_left ((if twoSource then 8 else 9) - k) (ceilLog2 th)
    cases twoSource <;> simp only [↓reduceIte, Bool.false_eq_true] at hkHi hMin ⊢ <;> omega
  have hkC : k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ k = 9 := by
    unfold peMaxK at hkHi
    cases twoSource <;> simp only [↓reduceIte, Bool.false_eq_true] at hkHi <;> omega
  have hmC : peTileM twoSource k th = 0 ∨ peTileM twoSource k th = 1 ∨ peTileM twoSource k th = 2 ∨
             peTileM twoSource k th = 3 ∨ peTileM twoSource k th = 4 ∨ peTileM twoSource k th = 5 := by omega
  unfold encodePeTileWord
  dsimp only
  generalize hmEq : peTileM twoSource k th = m at hmC
  rcases hkC with rfl | rfl | rfl | rfl | rfl | rfl <;>
  rcases hmC with rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact ⟨_, rfl, rfl, rfl⟩

/-- Direct soundness of `encodePeBcastWord` (called inside `emitPeTask`): succeeds for any
    `broadcastDims`-validated `(s1, s2, out)`, decodes every broadcast bit back identically,
    and guarantees `Vane.Verify`'s two broadcast invariants (never on axis=1, never both sources). -/
theorem encodePeBcastWord_verifier_sound (s1 s2 out : TensorDims)
    (hOk : broadcastDims s1 s2 = .ok out) :
    ∃ w, encodePeBcastWord s1 s2 out = .ok w ∧
      decodeFieldChecked peBcastLayout w "src1W" = .ok (if s1.w != out.w then 1 else 0) ∧
      decodeFieldChecked peBcastLayout w "src1H" = .ok (if s1.h != out.h then 1 else 0) ∧
      decodeFieldChecked peBcastLayout w "src1C" = .ok (if s1.c != out.c then 1 else 0) ∧
      decodeFieldChecked peBcastLayout w "src2W" = .ok (if s2.w != out.w then 1 else 0) ∧
      decodeFieldChecked peBcastLayout w "src2H" = .ok (if s2.h != out.h then 1 else 0) ∧
      decodeFieldChecked peBcastLayout w "src2C" = .ok (if s2.c != out.c then 1 else 0) ∧
      ¬ (s1.w != out.w ∧ s2.w != out.w) ∧
      ¬ (s1.h != out.h ∧ s2.h != out.h) ∧
      ¬ (s1.c != out.c ∧ s2.c != out.c) := by
  have hBr := pe_broadcastDims_verifier_legal s1 s2 out hOk
  have hEnc : ∃ w, encodePeBcastWord s1 s2 out = .ok w ∧
      decodeFieldChecked peBcastLayout w "src1W" = .ok (if s1.w != out.w then 1 else 0) ∧
      decodeFieldChecked peBcastLayout w "src1H" = .ok (if s1.h != out.h then 1 else 0) ∧
      decodeFieldChecked peBcastLayout w "src1C" = .ok (if s1.c != out.c then 1 else 0) ∧
      decodeFieldChecked peBcastLayout w "src2W" = .ok (if s2.w != out.w then 1 else 0) ∧
      decodeFieldChecked peBcastLayout w "src2H" = .ok (if s2.h != out.h then 1 else 0) ∧
      decodeFieldChecked peBcastLayout w "src2C" = .ok (if s2.c != out.c then 1 else 0) := by
    unfold encodePeBcastWord
    cases (s1.w != out.w) <;> cases (s1.h != out.h) <;> cases (s1.c != out.c) <;>
    cases (s2.w != out.w) <;> cases (s2.h != out.h) <;> cases (s2.c != out.c) <;>
      exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
  obtain ⟨w, he, h1, h2, h3, h4, h5, h6⟩ := hEnc
  refine ⟨w, he, h1, h2, h3, h4, h5, h6, ?_, ?_, ?_⟩
  · intro ⟨ha, hb⟩; exact hBr.2.2.2.2.2.2.1 ⟨by simp [ha], by simp [hb]⟩
  · intro ⟨ha, hb⟩; exact hBr.2.2.2.2.2.2.2.1 ⟨by simp [ha], by simp [hb]⟩
  · intro ⟨ha, hb⟩; exact hBr.2.2.2.2.2.2.2.2 ⟨by simp [ha], by simp [hb]⟩

/-- Standalone `neAct` register consistency directly on `encodeNeActOpWord` (called inside `emitNeActTask`). -/
theorem neAct_control_regs_sound (hasShift isRelu isLut : Bool) :
    (∃ wDw, encodeWord dwGroupsLayout #[("perGroupIn", 1)] = .ok wDw ∧
      decodeFieldChecked dwGroupsLayout wDw "perGroupIn" = .ok 1) ∧
    (∃ wOp, encodeNeActOpWord hasShift isRelu isLut = .ok wOp ∧
      decodeFieldChecked neActOpLayout wOp "noShift" = .ok (if hasShift then 0 else 1) ∧
      decodeFieldChecked neActOpLayout wOp "relu" = .ok (if isRelu then 1 else 0) ∧
      decodeFieldChecked neActOpLayout wOp "lut" = .ok (if isLut then 1 else 0)) := by
  constructor
  · exact ⟨_, rfl, rfl⟩
  · unfold encodeNeActOpWord
    cases hasShift <;> cases isRelu <;> cases isLut <;> exact ⟨_, rfl, rfl, rfl, rfl⟩

/-- `neAct` LUT table size is 128B (64B-aligned) and `encodeNeActLutCfgWord` (called inside
    `emitNeActTask`) encodes cleanly for any `taskId < 255`. -/
theorem neAct_lut_table_and_cfg_sound (taskId : Nat) (ht : taskId < 255) :
    lutTableBytes = 128 ∧ lutTableBytes % 64 = 0 ∧
    (encodeNeActLutCfgWord taskId).isOk = true := by
  refine ⟨rfl, by decide, ?_⟩
  unfold encodeNeActLutCfgWord
  have h1 : natToU32Checked taskId "taskId" = .ok taskId.toUInt32 :=
    natToU32Checked_ok_of_lt _ _ (by unfold maxU32Nat; omega)
  simp only [bind, Except.bind, h1]
  have hT : ¬ (256 ≤ taskId % 4294967296) := by omega
  have hVL : ∃ m, validateWordLayout chunkSlotLayout = .ok m := ⟨_, rfl⟩
  obtain ⟨m, hm⟩ := hVL
  have hEnc : encodeWord chunkSlotLayout #[("enable", 1), ("on20", 1), ("task", taskId.toUInt32)] =
      .ok ((((0 : UInt32) ||| ((1 : UInt32) <<< 0)) ||| ((1 : UInt32) <<< 5)) ||| (taskId.toUInt32 <<< 16)) := by
    unfold encodeWord
    have hVE : validateEncodeValues chunkSlotLayout.name chunkSlotLayout.fields.toList
        [("enable", 1), ("on20", 1), ("task", taskId.toUInt32)] [] = .ok () := by
      simp [chunkSlotLayout, validateEncodeValues]
    simp only [bind, Except.bind, hm, hVE]
    simp [chunkSlotLayout, encodeWordFields, hT]
  rw [hEnc]; rfl

/-- `peReduceAxes` preserves exact total elements (`sN * sC * sD * sH = c * h`) for all 4
    `(reduceC, reduceH)` axis combinations (matching `verifyPeReduceTaskMulti`!). -/
theorem peReduceAxes_element_conservation (c h : Nat) (reduceC reduceH : Bool) :
    let ax := peReduceAxes c h reduceC reduceH
    ax.sN * ax.sC * ax.sD * ax.sH = c * h := by
  cases reduceC <;> cases reduceH
  · by_cases hc : c = 1 <;> simp [peReduceAxes, hc, Nat.mul_comm]
  · simp [peReduceAxes, Nat.mul_comm]
  · simp [peReduceAxes, Nat.mul_comm]
  · by_cases hlt : h < c <;> simp [peReduceAxes, hlt, Nat.mul_comm]

-- ============================================================================
-- 4D. Engine Family 4: `dmaView` → Verifier Soundness
-- ============================================================================

/-- Direct soundness of `encodeDmaViewCommonWord` (called inside `emitDmaViewTask`):
    for verifier-legal shifts `shPref ≤ shMax ≤ 4` and `f ≤ 4`, succeeds and decodes
    `kind = 0`, `shMin = shPref ≤ shMax ≤ 4`, `F = f`, `transposeOut`, `transposeWideGroup`. -/
theorem encodeDmaViewCommonWord_verifier_sound
    (shPref shMax f : Nat) (transposeOut transposeWideGroup : Bool)
    (hSh : shPref ≤ shMax) (hMax : shMax ≤ 4) (hf : f ≤ 4) :
    ∃ w, encodeDmaViewCommonWord shPref shMax f transposeOut transposeWideGroup = .ok w ∧
      decodeFieldChecked neViewCommonLayout w "kind" = .ok 0 ∧
      decodeFieldChecked neViewCommonLayout w "shPref" = .ok shPref.toUInt32 ∧
      decodeFieldChecked neViewCommonLayout w "shMin" = .ok shPref.toUInt32 ∧
      decodeFieldChecked neViewCommonLayout w "shMax" = .ok shMax.toUInt32 ∧
      decodeFieldChecked neViewCommonLayout w "F" = .ok f.toUInt32 ∧
      decodeFieldChecked neViewCommonLayout w "transposeOut" = .ok (if transposeOut then 1 else 0) ∧
      decodeFieldChecked neViewCommonLayout w "transposeWideGroup" = .ok (if transposeWideGroup then 1 else 0) := by
  have hsp : shPref = 0 ∨ shPref = 1 ∨ shPref = 2 ∨ shPref = 3 ∨ shPref = 4 := by omega
  have hsm : shMax = 0 ∨ shMax = 1 ∨ shMax = 2 ∨ shMax = 3 ∨ shMax = 4 := by omega
  have hfc : f = 0 ∨ f = 1 ∨ f = 2 ∨ f = 3 ∨ f = 4 := by omega
  rcases hsp with rfl | rfl | rfl | rfl | rfl <;>
  rcases hsm with rfl | rfl | rfl | rfl | rfl <;>
  rcases hfc with rfl | rfl | rfl | rfl | rfl <;>
  cases transposeOut <;> cases transposeWideGroup <;>
    exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- `dmaView` T-mode (`transposeCW = true`) groupCode / ocg / channel-group coverage consistency
    matches `verifyNeViewTaskMulti`'s checks for all `c ≤ 128`! -/
theorem dmaView_tmode_matches_verifier (c : Nat) :
    groupCode c = (1 : Nat) <<< groupCodeLog2 c ∧
    groupCodeLog2 c ≤ 3 ∧
    (c ≤ 128 → c ≤ (1 : Nat) <<< (4 + groupCodeLog2 c)) := by
  unfold groupCode groupCodeLog2
  split
  · exact ⟨rfl, by omega, fun _ => by omega⟩
  · split
    · exact ⟨rfl, by omega, fun _ => by omega⟩
    · split
      · exact ⟨rfl, by omega, fun _ => by omega⟩
      · exact ⟨rfl, by omega, fun hc => by omega⟩

/-- `dmaView` `pack_out` DMA tile extent soundness (supporting `n ≥ 1` batch strides!):
    for any `OutputSlot` packed by `packOutputSlots` with `n, c, h, w ≥ 1`, the `ane_hx_dma_tile`
    raw extent `2 * w + (h - 1) * rSt + (c - 1) * pSt + (n - 1) * bSt` aligned up to 64B equals
    `slot.byteSize` EXACTLY, and therefore `slot.byteOffset + alignedExt ≤ totalBytes`! -/
theorem dmaView_packOut_dma_tile_bounds (n c h w : Nat)
    (hn : n > 0) (hc : c > 0) (hh : h > 0) :
    let rSt := alignUp (2 * w) 64
    let pSt := h * rSt
    let bSt := c * pSt
    let rawExt := 2 * w + (h - 1) * rSt + (c - 1) * pSt + (n - 1) * bSt
    let alignedExt := alignUp rawExt 64
    alignedExt % 64 = 0 ∧ alignedExt = n * bSt := by
  intro rSt pSt bSt rawExt alignedExt
  have hRow64 : rSt % 64 = 0 := alignUp_dvd_64 (2 * w)
  have hRowGe : 2 * w ≤ rSt := alignUp_ge_local (2 * w) 64
  have hRowLt : rSt < 2 * w + 64 := alignUp_lt_add_64 (2 * w)
  have hSplitH : h * rSt = (h - 1) * rSt + rSt := by
    cases h with | zero => omega | succ h' => simp [Nat.succ_mul]
  have hSplitC : c * pSt = (c - 1) * pSt + pSt := by
    cases c with | zero => omega | succ c' => simp [Nat.succ_mul]
  have hSplitN : n * bSt = (n - 1) * bSt + bSt := by
    cases n with | zero => omega | succ n' => simp [Nat.succ_mul]
  have hPrefix64 : ((h - 1) * rSt + (c - 1) * pSt + (n - 1) * bSt) % 64 = 0 := by
    have hp64 : pSt % 64 = 0 := by dsimp [pSt]; rw [Nat.mul_mod, hRow64, Nat.mul_zero, Nat.zero_mod]
    have hb64 : bSt % 64 = 0 := by dsimp [bSt]; rw [Nat.mul_mod, hp64, Nat.mul_zero, Nat.zero_mod]
    have h1 : ((h - 1) * rSt) % 64 = 0 := by rw [Nat.mul_mod, hRow64, Nat.mul_zero, Nat.zero_mod]
    have h2 : ((c - 1) * pSt) % 64 = 0 := by rw [Nat.mul_mod, hp64, Nat.mul_zero, Nat.zero_mod]
    have h3 : ((n - 1) * bSt) % 64 = 0 := by rw [Nat.mul_mod, hb64, Nat.mul_zero, Nat.zero_mod]
    omega
  have hTotalEq : n * bSt = rSt + ((h - 1) * rSt + (c - 1) * pSt + (n - 1) * bSt) := by
    dsimp [bSt, pSt] at hSplitH hSplitC hSplitN ⊢
    omega
  have hRawLt : rawExt ≤ n * bSt := by dsimp [rawExt]; omega
  have hRawGt : n * bSt < rawExt + 64 := by dsimp [rawExt]; omega
  have hTotal64 : (n * bSt) % 64 = 0 := by omega
  have hAlEq : alignedExt = n * bSt := by
    have hA64 : alignedExt % 64 = 0 := alignUp_dvd_64 rawExt
    have hAGe : rawExt ≤ alignedExt := alignUp_ge_local rawExt 64
    have hALt : alignedExt < rawExt + 64 := alignUp_lt_add_64 rawExt
    omega
  exact ⟨alignUp_dvd_64 rawExt, hAlEq⟩

/-- H-stripped multi-output `pack_out_{oi}_s{si}` DMA tile bounds:
    for strip `rowOff` with `rowOff + stripH ≤ fullH` and `stripH > 0, c > 0`,
    `subviewByteOff = slotOff + rowOff * rSt` is 64B-aligned and the aligned DMA extent
    stays strictly within `[slotOff, slotOff + c * fullH * rSt)`! -/
theorem dmaView_packOut_strip_dma_bounds (slotOff rowOff stripH fullH c w : Nat)
    (hSlot64 : slotOff % 64 = 0) (hc : c > 0) (hStrip : stripH > 0)
    (hBound : rowOff + stripH ≤ fullH) :
    let rSt := alignUp (2 * w) 64
    let pSt := fullH * rSt
    let subOff := slotOff + rowOff * rSt
    let rawExt := 2 * w + (stripH - 1) * rSt + (c - 1) * pSt
    subOff % 64 = 0 ∧ alignUp rawExt 64 % 64 = 0 ∧
    subOff + alignUp rawExt 64 ≤ slotOff + c * pSt := by
  intro rSt pSt subOff rawExt
  have hRow64 : rSt % 64 = 0 := alignUp_dvd_64 (2 * w)
  have hRowGe : 2 * w ≤ rSt := alignUp_ge_local (2 * w) 64
  have hRowLt : rSt < 2 * w + 64 := alignUp_lt_add_64 (2 * w)
  have hp64 : pSt % 64 = 0 := by dsimp [pSt]; rw [Nat.mul_mod, hRow64, Nat.mul_zero, Nat.zero_mod]
  have hSub64 : subOff % 64 = 0 := by
    have h1 : (rowOff * rSt) % 64 = 0 := by rw [Nat.mul_mod, hRow64, Nat.mul_zero, Nat.zero_mod]
    dsimp [subOff]; omega
  have hExtEq : alignUp rawExt 64 = stripH * rSt + (c - 1) * pSt := by
    have hSplitS : stripH * rSt = (stripH - 1) * rSt + rSt := by
      cases stripH with | zero => omega | succ s' => simp [Nat.succ_mul]
    have hTarget64 : (stripH * rSt + (c - 1) * pSt) % 64 = 0 := by
      have h1 : (stripH * rSt) % 64 = 0 := by rw [Nat.mul_mod, hRow64, Nat.mul_zero, Nat.zero_mod]
      have h2 : ((c - 1) * pSt) % 64 = 0 := by rw [Nat.mul_mod, hp64, Nat.mul_zero, Nat.zero_mod]
      omega
    have hA64 := alignUp_dvd_64 rawExt
    have hAGe := alignUp_ge_local rawExt 64
    have hALt : alignUp rawExt 64 < rawExt + 64 := alignUp_lt_add_64 rawExt
    dsimp [rawExt] at *
    omega
  refine ⟨hSub64, alignUp_dvd_64 rawExt, ?_⟩
  rw [hExtEq]
  have hSplitC : c * pSt = (c - 1) * pSt + pSt := by
    cases c with | zero => omega | succ c' => simp [Nat.succ_mul]
  have hRowFit : rowOff * rSt + stripH * rSt ≤ pSt := by
    dsimp [pSt]
    rw [← Nat.add_mul]
    exact Nat.mul_le_mul_right rSt hBound
  dsimp [subOff]
  omega

-- ============================================================================
-- 4E. Engine Family 5: `neMatmul` → Verifier Soundness
-- ============================================================================

/-- Direct soundness of `encodeMatmulCommonWord` (called inside `Vane.Matmul.buildMatmulTaskMulti`):
    for `shPref ≤ shMax ≤ 4`, `f ≤ 4`, and `isDyn := m * b > 1`, succeeds and matches EVERY check
    of `Vane.Verify` (`lo2 = if alignUp n 8 ≤ 64 then 1 else 0`, `shMin = shPref ≤ shMax ≤ 4`,
    `F = f`, and `dynKern = if isDyn then 1 else 0` — strictly `0` when `m * b = 1`)! -/
theorem encodeMatmulCommonWord_verifier_sound
    (n shPref shMax f : Nat) (isDyn : Bool)
    (hSh : shPref ≤ shMax) (hMax : shMax ≤ 4) (hf : f ≤ 4) :
    ∃ w, encodeMatmulCommonWord n shPref shMax f isDyn = .ok w ∧
      decodeFieldChecked neMatmulCommonLayout w "lo2" = .ok (if alignUp n 8 ≤ 64 then 1 else 0) ∧
      decodeFieldChecked neMatmulCommonLayout w "shPref" = .ok shPref.toUInt32 ∧
      decodeFieldChecked neMatmulCommonLayout w "shMin" = .ok shPref.toUInt32 ∧
      decodeFieldChecked neMatmulCommonLayout w "shMax" = .ok shMax.toUInt32 ∧
      decodeFieldChecked neMatmulCommonLayout w "F" = .ok f.toUInt32 ∧
      decodeFieldChecked neMatmulCommonLayout w "dynKern" = .ok (if isDyn then 1 else 0) := by
  have hsp : shPref = 0 ∨ shPref = 1 ∨ shPref = 2 ∨ shPref = 3 ∨ shPref = 4 := by omega
  have hsm : shMax = 0 ∨ shMax = 1 ∨ shMax = 2 ∨ shMax = 3 ∨ shMax = 4 := by omega
  have hfc : f = 0 ∨ f = 1 ∨ f = 2 ∨ f = 3 ∨ f = 4 := by omega
  unfold encodeMatmulCommonWord
  by_cases hLo : alignUp n 8 ≤ 64
  · simp only [↓reduceIte, hLo]
    rcases hsp with rfl | rfl | rfl | rfl | rfl <;>
    rcases hsm with rfl | rfl | rfl | rfl | rfl <;>
    rcases hfc with rfl | rfl | rfl | rfl | rfl <;>
    cases isDyn <;>
      exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
  · simp only [↓reduceIte, hLo]
    rcases hsp with rfl | rfl | rfl | rfl | rfl <;>
    rcases hsm with rfl | rfl | rfl | rfl | rfl <;>
    rcases hfc with rfl | rfl | rfl | rfl | rfl <;>
    cases isDyn <;>
      exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Direct soundness of `encodeMatmulKernelStageWord` (called inside `Vane.Matmul.buildMatmulTaskMulti`):
    for `taskId < 255`, succeeds and satisfies EVERY `kernelStage` (`0x5500`) check in `Vane.Verify`
    (`sparse = 0`, `kernEn = 1`, `kernCfg = 1`, `isLast`, `h17Dyn = target.profile.dynKernelStageBit && isDyn`,
    and `stage = taskId + 1`)! -/
theorem encodeMatmulKernelStageWord_verifier_sound
    (target : Target) (isLast isDyn : Bool) (taskId : Nat) (ht : taskId < 255) :
    encodeMatmulKernelStageWord target isLast isDyn taskId =
      .ok (((((((0 : UInt32) ||| ((0 : UInt32) <<< 5)) ||| ((1 : UInt32) <<< 6)) |||
        ((if isLast then (1 : UInt32) else 0) <<< 7)) ||| ((1 : UInt32) <<< 9)) |||
        ((if target.profile.dynKernelStageBit && isDyn then (1 : UInt32) else 0) <<< 10)) |||
        ((taskId + 1).toUInt32 <<< 16)) := by
  unfold encodeMatmulKernelStageWord
  have h1 : natToU32Checked (taskId + 1) "kernelStage.stage" = .ok (taskId + 1).toUInt32 :=
    natToU32Checked_ok_of_lt _ _ (by unfold maxU32Nat; omega)
  simp only [bind, Except.bind, h1]
  have hS : ¬ (256 ≤ (taskId + 1) % 4294967296) := by omega
  have hVL : ∃ m, validateWordLayout kernelStageDynLayout = .ok m := ⟨_, rfl⟩
  obtain ⟨m, hm⟩ := hVL
  unfold encodeWord
  have hVE : validateEncodeValues kernelStageDynLayout.name kernelStageDynLayout.fields.toList
      [("sparse", 0), ("kernEn", 1), ("isLast", if isLast then 1 else 0), ("kernCfg", 1),
       ("h17Dyn", if target.profile.dynKernelStageBit && isDyn then 1 else 0),
       ("stage", (taskId + 1).toUInt32)] [] = .ok () := by
    simp [kernelStageDynLayout, validateEncodeValues]
  simp only [bind, Except.bind, hm, hVE]
  cases target <;> cases isLast <;> cases isDyn <;>
    simp [kernelStageDynLayout, encodeWordFields, Target.profile, hS]

/-- `neMatmul` 1×1 stride-1 `convCfg`, `neMatmulCfg` (`dynKern=1`), and `neTile` (`ocg=0, fat=0, WU=0`)
    encode cleanly and satisfy `verifyNeMatmulTaskMulti`'s register checks for all targets! -/
theorem neMatmul_static_regs_sound (target : Target) :
    (∃ wConv, encodeWord convCfgLayout #[
        ("kw", 1), ("kh", 1), ("sx", 1), ("sy", 1), ("pl", 0), ("pt", 0), ("hx", 1), ("vy", 1)
      ] = .ok wConv ∧
      decodeFieldChecked convCfgLayout wConv "kw" = .ok 1 ∧
      decodeFieldChecked convCfgLayout wConv "kh" = .ok 1 ∧
      decodeFieldChecked convCfgLayout wConv "sx" = .ok 1 ∧
      decodeFieldChecked convCfgLayout wConv "sy" = .ok 1 ∧
      decodeFieldChecked convCfgLayout wConv "pl" = .ok 0 ∧
      decodeFieldChecked convCfgLayout wConv "pt" = .ok 0 ∧
      decodeFieldChecked convCfgLayout wConv "hx" = .ok 1 ∧
      decodeFieldChecked convCfgLayout wConv "vy" = .ok 1) ∧
    (decodeFieldChecked (neMatmulCfgLayout target) (neCfgDynValue target) "dynKern" = .ok 1) ∧
    (∃ wTile, encodeWord neTileLayout #[("ocg", 0), ("fat", 0), ("WU", 0)] = .ok wTile ∧
      decodeFieldChecked neTileLayout wTile "ocg" = .ok 0 ∧
      decodeFieldChecked neTileLayout wTile "fat" = .ok 0 ∧
      decodeFieldChecked neTileLayout wTile "WU" = .ok 0) := by
  refine ⟨⟨_, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩, ?_, ⟨_, rfl, rfl, rfl, rfl⟩⟩
  cases target <;> rfl

/-- `neMatmul` M-slice bound soundness: for any `1 ≤ m ≤ 16`, `f = ceilLog2 m` satisfies
    `f ≤ 4` and `m ≤ (1 : Nat) <<< f` (the exact `verifyNeMatmulTaskMulti` checks!). -/
theorem neMatmul_m_slice_f_sound (m : Nat) (h1 : 1 ≤ m) (h16 : m ≤ 16) :
    ceilLog2 m ≤ 4 ∧ m ≤ (1 : Nat) <<< ceilLog2 m := by
  match m with
  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 => decide
  | 0 => omega
  | _ + 17 => omega

/-- `neMatmul` kernel chunk size and M-slice offset `m0 * matmulChunkSize k` are always 64B-aligned. -/
theorem neMatmul_kernel_offset_aligned_64 (m0 k : Nat) :
    matmulChunkSize k % 64 = 0 ∧ (m0 * matmulChunkSize k) % 64 = 0 := by
  have h1 : matmulChunkSize k % 64 = 0 := alignUp_dvd_64 (2 * k)
  exact ⟨h1, by rw [Nat.mul_mod, h1, Nat.mul_zero, Nat.zero_mod]⟩

/-- `neMatmul` `planMatmulScratchTranspose` workspace soundness:
    whenever `planMatmulScratchTranspose b rows cols = .ok (cp, reqScratch)`:
    1. `reqScratch` is 64B-aligned,
    2. `reqScratch` covers the actual transposed tensor span `b * cols * alignUp (2 * rows) 64`,
    3. all active DMA strides (`inPlaneStride`, `inNStride`, `outPlaneStride`, `outNStride`) are 64B-aligned,
    4. if `cp.transposeCW = true`, then `rows ≤ 128` (satisfying the T-mode channel coverage bound)! -/
theorem neMatmul_scratch_transpose_workspace_sound (b rows cols : Nat) (cp : ViewCopy) (reqScratch : Nat)
    (hOk : planMatmulScratchTranspose b rows cols = .ok (cp, reqScratch)) :
    reqScratch % 64 = 0 ∧
    b * cols * alignUp (2 * rows) 64 ≤ reqScratch ∧
    cp.inPlaneStride % 64 = 0 ∧ cp.inNStride % 64 = 0 ∧
    cp.outPlaneStride % 64 = 0 ∧ cp.outNStride % 64 = 0 ∧
    (cp.transposeCW = true → rows ≤ 128) := by
  unfold planMatmulScratchTranspose at hOk
  simp only [bind, Except.bind, pure, Except.pure] at hOk
  split at hOk
  · cases hOk
  · split at hOk
    · rename_i h11
      simp only [Bool.and_eq_true, beq_iff_eq] at h11
      simp only [Except.ok.injEq, Prod.mk.injEq] at hOk
      rcases hOk with ⟨rfl, rfl⟩
      have hAl2 : alignUp (2 * 1) 64 = 64 := by decide
      refine ⟨alignUp_dvd_64 _, ?_, rfl, ?_, rfl, ?_, fun h => by cases h⟩
      · rw [h11.1, h11.2, Nat.mul_one, hAl2]
        exact alignUp_ge_local (b * dmaLineBytes) 64
      · by_cases hb : b > 1 <;> simp [hb, dmaLineBytes]
      · by_cases hb : b > 1 <;> simp [hb, dmaLineBytes]
    · split at hOk
      · cases hOk
      · rename_i hLe128
        simp only [Except.ok.injEq, Prod.mk.injEq] at hOk
        rcases hOk with ⟨rfl, rfl⟩
        have hInRow64 : alignUp (2 * cols) 64 % 64 = 0 := alignUp_dvd_64 (2 * cols)
        have hOutRow64 : alignUp (2 * rows) 64 % 64 = 0 := alignUp_dvd_64 (2 * rows)
        have hInN64 : (rows * alignUp (2 * cols) 64) % 64 = 0 := by
          rw [Nat.mul_mod, hInRow64, Nat.mul_zero, Nat.zero_mod]
        have hOutN64 : (cols * alignUp (2 * rows) 64) % 64 = 0 := by
          rw [Nat.mul_mod, hOutRow64, Nat.mul_zero, Nat.zero_mod]
        refine ⟨alignUp_dvd_64 _, ?_, ?_, ?_, ?_, ?_, fun _ => by omega⟩
        · exact Nat.le_trans (Nat.le_max_left _ _) (alignUp_ge_local _ 64)
        · by_cases hr : rows > 1 <;> simp [hr, dmaLineBytes, hInRow64]
        · by_cases hb : b > 1 <;> simp [hb, dmaLineBytes, hInN64]
        · by_cases hc : cols > 1 <;> simp [hc, dmaLineBytes, hOutRow64]
        · by_cases hb : b > 1 <;> simp [hb, dmaLineBytes, hOutN64]

end Vane.Proofs.C3Emit
