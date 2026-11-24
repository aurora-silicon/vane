import Vane.Binary
import Vane.TaskStream

namespace Vane.Tiling

open Vane.Binary
open Vane.TaskStream

/-- Hardware L2 SRAM capacity in bytes (2 MiB). -/
def l2CapacityBytes : Nat := 0x200000

/-- Spatial and channel shape for a single-task fp16 1x1 convolution. -/
structure Conv1x1Shape where
  h : Nat
  w : Nat
  cin : Nat
  cout : Nat
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Tiling configuration for a single-task fp16 1x1 convolution. -/
structure Tiling where
  logNE : Nat
  f : Nat
  fat : Nat
  ocg : Nat
  shPref : Nat
  shMax : Nat
  tileHeight : Nat
  srcSkew : Nat := 0
  resSkew : Nat := 0
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Ceiling base-2 logarithm: `ceilLog2 1 = 0`, `ceilLog2 n = floor(log2(n - 1)) + 1` for `n > 1`. -/
def ceilLog2 (n : Nat) : Nat :=
  if n <= 1 then 0 else Nat.log2 (n - 1) + 1

/-- Ceiling integer division for positive denominator `b > 0`. -/
def ceilDiv (a b : Nat) : Nat :=
  if b == 0 then 0 else (a + b - 1) / b

/-- Work-unit split exponent `WU = logNE - F - fat` (valid when `F + fat <= logNE`). -/
def Tiling.wu (t : Tiling) : Nat :=
  t.logNE - t.f - t.fat

/-- Preferred work-unit strip width in pixels: `T = (128 << fat) >> shPref`. -/
def stripWidth (t : Tiling) : Nat :=
  (128 <<< t.fat) >>> t.shPref

/-- Source strip pixels buffered in L2: `srcPx = min(align_up(W, 8), max(T, 32))`. -/
def srcPx (shape : Conv1x1Shape) (t : Tiling) : Nat :=
  min (alignUp shape.w 8) (max (stripWidth t) 32)

/-- Bytes per channel row in L2 source staging (`chs = 2 * srcPx`). -/
def chsBytes (shape : Conv1x1Shape) (t : Tiling) : Nat :=
  2 * srcPx shape t

/-- Bytes per channel row in L2 result staging (`dchs`, equal to `chs` for stride-1 1x1 conv). -/
def dchsBytes (shape : Conv1x1Shape) (t : Tiling) : Nat :=
  chsBytes shape t

/-- Source row stride in L2 SRAM: `Cin * chs + 16 * srcSkew`. -/
def srcRowStride (shape : Conv1x1Shape) (t : Tiling) : Nat :=
  shape.cin * chsBytes shape t + 16 * t.srcSkew

/-- Result row stride in L2 SRAM (staged mode): `Cout * dchs + 16 * resSkew`. -/
def resRowStride (shape : Conv1x1Shape) (t : Tiling) : Nat :=
  shape.cout * dchsBytes shape t + 16 * t.resSkew

/-- Whether the result staging operates in direct mode (`!staged`). -/
def isDirect (target : Target) (shape : Conv1x1Shape) (t : Tiling) : Bool :=
  let lhs := match target with
    | .h17 => (128 <<< t.fat) >>> t.shMax
    | .h16 => 128 >>> t.shMax
  lhs >= min 32 shape.w

/-- Bytes of L2 SRAM occupied by source staging: `min(TH, H) * srcRowStride`. -/
def srcL2Bytes (shape : Conv1x1Shape) (t : Tiling) : Nat :=
  min t.tileHeight shape.h * srcRowStride shape t

/-- Bytes of L2 SRAM occupied by result staging. -/
def resL2Bytes (target : Target) (shape : Conv1x1Shape) (t : Tiling) : Nat :=
  if isDirect target shape t then
    (1 <<< (t.f + t.ocg + t.fat)) * 512
  else
    min t.tileHeight shape.h * resRowStride shape t

/-- Total L2 SRAM footprint (`srcL2Bytes + resL2Bytes`). -/
def l2UseBytes (target : Target) (shape : Conv1x1Shape) (t : Tiling) : Nat :=
  srcL2Bytes shape t + resL2Bytes target shape t

/-- Smallest `i in 0..5` such that `wd % (base >> i) == 0`, or `6` if none. -/
def firstDiv (wd base : Nat) : Nat := Id.run do
  for i in [:6] do
    let d := base >>> i
    if d > 0 && wd % d == 0 then
      return i
  return 6

/-- Width-cover exponent for aligned width `wAligned`, strip width `stripW`, and `base = 128 << fat`.
    Equivalent to `min(4, max_{x = 0, T, 2T, ... < W'} firstDiv(min(T, W' - x), base))`. -/
def cover (wAligned stripW base : Nat) : Nat :=
  if stripW == 0 || wAligned == 0 then
    6
  else if wAligned <= stripW then
    min 4 (firstDiv wAligned base)
  else
    let fullDiv := firstDiv stripW base
    let rem := wAligned % stripW
    let mx := if rem == 0 then fullDiv else max fullDiv (firstDiv rem base)
    min 4 mx

/-- Minimum cover exponent across alignment multiples `{8, 16, 32, 64}`. -/
def coverMin (w stripW fat : Nat) : Nat :=
  let wuBase : Nat := (128 : Nat) <<< fat
  let c8 := cover (alignUp w 8) stripW wuBase
  let c16 := cover (alignUp w 16) stripW wuBase
  let c32 := cover (alignUp w 32) stripW wuBase
  let c64 := cover (alignUp w 64) stripW wuBase
  min (min c8 c16) (min c32 c64)

/-- Validate basic 1x1 conv shape bounds (`H, W in 1..65535`, `Cin, Cout >= 1`). -/
def validateShape (shape : Conv1x1Shape) : Except String Unit := do
  if shape.h == 0 || shape.h > 65535 then
    throw s!"invalid H={shape.h}: must be in 1..65535"
  if shape.w == 0 || shape.w > 65535 then
    throw s!"invalid W={shape.w}: must be in 1..65535"
  if shape.cin == 0 || shape.cin > maxU32Nat then
    throw s!"invalid Cin={shape.cin}: must be in 1..{maxU32Nat}"
  if shape.cout == 0 || shape.cout > maxU32Nat then
    throw s!"invalid Cout={shape.cout}: must be in 1..{maxU32Nat}"
  pure ()

/-- Decidable legality validator enforcing all 11 hardware tiling rules for single-task fp16 1x1 conv. -/
def validateTiling (target : Target) (shape : Conv1x1Shape) (t : Tiling) : Except String Unit := do
  validateShape shape

  -- Rule 1: NE count & chunk slots
  if t.logNE > 4 then
    throw s!"illegal tiling: logNE={t.logNE} exceeds 4"

  -- Rule 2: FatTile field width
  if t.fat > 1 then
    throw s!"illegal tiling: fat={t.fat} must be 0 or 1"

  -- Rule 3: Work-unit split (WU = logNE - F - fat <= 3)
  if t.f + t.fat > t.logNE then
    throw s!"illegal tiling: F({t.f}) + fat({t.fat}) > logNE({t.logNE})"
  let wu := t.logNE - t.f - t.fat
  if wu > 3 then
    throw s!"illegal tiling: WU={wu} exceeds 2-bit limit 3 (logNE={t.logNE}, F={t.f}, fat={t.fat})"

  -- Rule 4: Output channels per NE (0 <= ocg <= 4)
  if t.ocg > 4 then
    throw s!"illegal tiling: ocg={t.ocg} exceeds maximum 4"

  -- Rule 5: Output channel coverage
  let maxCout := match target with
    | .h16 => (1 : Nat) <<< (t.f + t.ocg)
    | .h17 => (1 : Nat) <<< (t.f + t.ocg + 1)
  if shape.cout > maxCout then
    throw s!"illegal tiling: Cout={shape.cout} exceeds {target.name} coverage {maxCout} (F={t.f}, ocg={t.ocg})"

  -- Rule 6: Group bound (F <= ceil_log2(Cout))
  let maxF := ceilLog2 shape.cout
  if t.f > maxF then
    throw s!"illegal tiling: F={t.f} exceeds ceil_log2(Cout={shape.cout})={maxF}"

  -- Rule 7: Strip width (8 <= T <= 256)
  let stripW := stripWidth t
  if stripW < 8 || stripW > 256 then
    throw s!"illegal tiling: strip width T={stripW} outside [8, 256]"

  -- Rule 8: Work-unit shifts (0 <= shPref <= shMax <= 4)
  if t.shPref > t.shMax || t.shMax > 4 then
    throw s!"illegal tiling: shifts shPref={t.shPref}, shMax={t.shMax} violate 0 <= shPref <= shMax <= 4"

  -- Rule 9: Width cover rule
  let cMin := coverMin shape.w stripW t.fat
  if cMin > t.shMax then
    throw s!"illegal tiling: coverMin(W={shape.w}, T={stripW}, fat={t.fat})={cMin} > shMax={t.shMax}"

  -- Rule 10: Tile height
  if t.tileHeight == 0 || t.tileHeight > shape.h then
    throw s!"illegal tiling: tileHeight={t.tileHeight} outside 1..{shape.h}"

  -- Rule 11: L2 footprint
  let l2Use := l2UseBytes target shape t
  if l2Use > l2CapacityBytes then
    throw s!"illegal tiling: L2 footprint {l2Use} bytes exceeds {l2CapacityBytes} bytes (2 MiB)"

  pure ()

/-- Compute the smallest legal `ocg` (`0 <= ocg <= 4`) for `(target, cout, f)` under rules 4 and 5. -/
def chooseOcg? (target : Target) (cout f : Nat) : Option Nat :=
  let need := ceilDiv cout ((1 : Nat) <<< f)
  let ocg1 := ceilLog2 need
  if ocg1 <= 4 then
    some ocg1
  else
    match target with
    | .h16 => none
    | .h17 =>
      if cout <= ((1 : Nat) <<< (f + 4 + 1)) then
        some 4
      else
        none

/-- Deterministic tile chooser: searches finite candidate configurations with `srcSkew = 0, resSkew = 0`,
    computes the maximal legal `tileHeight` in O(1) per configuration, and validates its result
    through `validateTiling`. -/
def chooseTiling (target : Target) (shape : Conv1x1Shape) : Except String Tiling := do
  validateShape shape

  let maxF := min 4 (ceilLog2 shape.cout)
  let logNEs : Array Nat := #[4, 3, 2, 1, 0]
  let fats : Array Nat := #[0, 1]

  for logNE in logNEs do
    for fat in fats do
      for fStep in [:maxF + 1] do
        let f := maxF - fStep
        if f + fat <= logNE && (logNE - f - fat) <= 3 then
          if let some ocg := chooseOcg? target shape.cout f then
            for shPref in [:5] do
              let stripW := (128 <<< fat) >>> shPref
              if stripW >= 8 && stripW <= 256 then
                let cMin := coverMin shape.w stripW fat
                let shMaxStart := max shPref cMin
                if shMaxStart <= 4 then
                  for shMax in [shMaxStart : 5] do
                    let probe : Tiling := {
                      logNE := logNE
                      f := f
                      fat := fat
                      ocg := ocg
                      shPref := shPref
                      shMax := shMax
                      tileHeight := 1
                      srcSkew := 0
                      resSkew := 0
                    }
                    let srcRow := srcRowStride shape probe
                    let direct := isDirect target shape probe
                    let maxTh :=
                      if direct then
                        let resBytes := (1 <<< (f + ocg + fat)) * 512
                        if resBytes >= l2CapacityBytes || srcRow == 0 then 0
                        else min shape.h ((l2CapacityBytes - resBytes) / srcRow)
                      else
                        let perRow := srcRow + resRowStride shape probe
                        if perRow == 0 then 0
                        else min shape.h (l2CapacityBytes / perRow)
                    if maxTh >= 1 then
                      let cand : Tiling := { probe with tileHeight := maxTh }
                      match validateTiling target shape cand with
                      | .ok () => return cand
                      | .error _ => pure ()

  throw s!"no legal {target.name} tiling found for 1x1 conv shape (H={shape.h}, W={shape.w}, Cin={shape.cin}, Cout={shape.cout})"

end Vane.Tiling
