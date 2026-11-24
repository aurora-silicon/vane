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

-- ============================================================================
-- 3x3 Convolution (Dense and Depthwise) Geometry, Legality, and O(1) Chooser
-- ============================================================================

/-- Mode for single-task fp16 3x3 convolution (`dense` or `depthwise`). -/
inductive Conv3x3Mode where
  | dense
  | depthwise
  deriving Repr, BEq, DecidableEq, Inhabited

namespace Conv3x3Mode

def name : Conv3x3Mode → String
  | .dense => "dense"
  | .depthwise => "depthwise"

def parse? : String → Option Conv3x3Mode
  | "dense" => some .dense
  | "depthwise" => some .depthwise
  | _ => none

end Conv3x3Mode

/-- Padding mode for single-task fp16 3x3 convolution (`same` or `valid`). -/
inductive PadMode where
  | same
  | valid
  deriving Repr, BEq, DecidableEq, Inhabited

namespace PadMode

def name : PadMode → String
  | .same => "same"
  | .valid => "valid"

def parse? : String → Option PadMode
  | "same" => some .same
  | "valid" => some .valid
  | _ => none

end PadMode

/-- Shape and parameters for a single-task fp16 3x3 convolution. -/
structure Conv3x3Shape where
  mode : Conv3x3Mode
  h : Nat
  w : Nat
  cin : Nat
  cout : Nat
  stride : Nat
  pad : PadMode
  deriving Repr, BEq, DecidableEq, Inhabited

namespace Conv3x3Shape

/-- Effective hardware mode: single-channel depthwise (`C == 1`) lowers to dense 3x3. -/
def effMode (sh : Conv3x3Shape) : Conv3x3Mode :=
  if sh.mode == .dense || sh.cin == 1 then .dense else .depthwise

/-- Whether stride-2 zero-skipping sparse kernel encoding is used. -/
def isSparse (sh : Conv3x3Shape) : Bool :=
  sh.stride == 2

/-- Output spatial height `Ho`. -/
def outH (sh : Conv3x3Shape) : Nat :=
  if sh.stride == 0 then 0
  else
    match sh.pad with
    | .same => ceilDiv sh.h sh.stride
    | .valid => if sh.h < 3 then 0 else (sh.h - 3) / sh.stride + 1

/-- Output spatial width `Wo`. -/
def outW (sh : Conv3x3Shape) : Nat :=
  if sh.stride == 0 then 0
  else
    match sh.pad with
    | .same => ceilDiv sh.w sh.stride
    | .valid => if sh.w < 3 then 0 else (sh.w - 3) / sh.stride + 1

/-- Top padding `pt` (floor of half total padding for `same`, `0` for `valid`). -/
def padTop (sh : Conv3x3Shape) : Nat :=
  match sh.pad with
  | .valid => 0
  | .same =>
    let ho := sh.outH
    if ho == 0 then 0 else ((ho - 1) * sh.stride + 3 - sh.h) / 2

/-- Left padding `pl` (floor of half total padding for `same`, `0` for `valid`). -/
def padLeft (sh : Conv3x3Shape) : Nat :=
  match sh.pad with
  | .valid => 0
  | .same =>
    let wo := sh.outW
    if wo == 0 then 0 else ((wo - 1) * sh.stride + 3 - sh.w) / 2

/-- Phase-rounded kernel extent `rk = align_up(3, stride)` (`3` for `s = 1`, `4` for `s = 2`). -/
def rk (sh : Conv3x3Shape) : Nat :=
  alignUp 3 sh.stride

/-- Cropped input width `cropW = min(W, (Wo - 1) * s + rk - pl)`. -/
def cropW (sh : Conv3x3Shape) : Nat :=
  let wo := sh.outW
  if wo == 0 then 0 else min sh.w ((wo - 1) * sh.stride + sh.rk - sh.padLeft)

/-- Cropped input height `cropH = min(H, (Ho - 1) * s + rk - pt)`. -/
def cropH (sh : Conv3x3Shape) : Nat :=
  let ho := sh.outH
  if ho == 0 then 0 else min sh.h ((ho - 1) * sh.stride + sh.rk - sh.padTop)

/-- Register `0x0004` (`Win`) value: `W` for `s = 1`, `cropW` for `s = 2`. -/
def winReg (sh : Conv3x3Shape) : Nat :=
  if sh.stride == 1 then sh.w else sh.cropW

/-- Register `0x0008` (`Hin`) value: `H` for `s = 1`, `cropH` for `s = 2`. -/
def hinReg (sh : Conv3x3Shape) : Nat :=
  if sh.stride == 1 then sh.h else sh.cropH

end Conv3x3Shape

/-- Effective `ocg` for 3x3 conv (`0` for true depthwise, `t.ocg` for dense). -/
def effOcg (sh : Conv3x3Shape) (t : Tiling) : Nat :=
  if sh.effMode == .depthwise then 0 else t.ocg

/-- Source strip width `srcPx` for 3x3 conv (`kw = 3 > s` for `s in {1, 2}`).
    Never narrow below this exact formula. -/
def srcPx3x3 (sh : Conv3x3Shape) (t : Tiling) : Nat :=
  let stripW := stripWidth t
  let ti := sh.stride * stripW
  let r := alignUp sh.winReg 8
  let halo : Int := if sh.padLeft > 0 then 8 else 0
  let flr : Int := if ti > 8 then ((min ti 32 : Nat) : Int) else 24
  let lookahead : Int := halo + min ((r : Int) - 2 * ((max ti 16 : Nat) : Int)) 32
  let extra : Nat := (max flr lookahead).toNat
  min r (ti + extra)

/-- Bytes per channel row in L2 source staging for 3x3 conv (`chs = 2 * srcPx3x3`). -/
def chsBytes3x3 (sh : Conv3x3Shape) (t : Tiling) : Nat :=
  2 * srcPx3x3 sh t

/-- Source row stride in L2 SRAM for 3x3 conv (`Cin * chs + 16 * srcSkew`). -/
def srcRowStride3x3 (sh : Conv3x3Shape) (t : Tiling) : Nat :=
  sh.cin * chsBytes3x3 sh t + 16 * t.srcSkew

/-- Input rows read by output row tile `tileIdx` (`0 <= tileIdx < ceil(Ho / th)`). -/
def tileInRows3x3 (h s pt ho th tileIdx : Nat) : Nat :=
  let rk := alignUp 3 s
  let y0 := tileIdx * th
  let y1 := min ho (y0 + th)
  if y1 <= y0 then 0
  else
    let hi := min h ((y1 - 1) * s + rk - pt)
    let lo := y0 * s - pt
    hi - lo

/-- Maximum input rows buffered per row tile in O(1) time (exact max over tiles `0`, `1`, `nTiles - 1`). -/
def inRows3x3 (sh : Conv3x3Shape) (th : Nat) : Nat :=
  let ho := sh.outH
  let pt := sh.padTop
  if th == 0 || ho == 0 then 0
  else
    let nTiles := ceilDiv ho th
    let r0 := tileInRows3x3 sh.h sh.stride pt ho th 0
    if nTiles <= 1 then r0
    else
      let r1 := tileInRows3x3 sh.h sh.stride pt ho th 1
      if nTiles == 2 then max r0 r1
      else
        let rLast := tileInRows3x3 sh.h sh.stride pt ho th (nTiles - 1)
        max (max r0 r1) rLast

/-- Whether result staging operates in direct mode (`!staged`) for 3x3 conv (evaluated on `Wo`). -/
def isDirect3x3 (target : Target) (sh : Conv3x3Shape) (t : Tiling) : Bool :=
  let lhs := match target with
    | .h17 => (128 <<< t.fat) >>> t.shMax
    | .h16 => 128 >>> t.shMax
  lhs >= min 32 sh.outW

/-- Bytes per channel row in L2 result staging for 3x3 conv (`dchs = 2 * min(align_up(Wo, 8), max(T, 32))`). -/
def dchsBytes3x3 (sh : Conv3x3Shape) (t : Tiling) : Nat :=
  2 * min (alignUp sh.outW 8) (max (stripWidth t) 32)

/-- Result row stride in L2 SRAM (staged mode) for 3x3 conv (`Cout * dchs + 16 * resSkew`). -/
def resRowStride3x3 (sh : Conv3x3Shape) (t : Tiling) : Nat :=
  sh.cout * dchsBytes3x3 sh t + 16 * t.resSkew

/-- Bytes of L2 SRAM occupied by source staging for 3x3 conv (`inRows3x3 * srcRowStride3x3`). -/
def srcL2Bytes3x3 (sh : Conv3x3Shape) (t : Tiling) : Nat :=
  inRows3x3 sh t.tileHeight * srcRowStride3x3 sh t

/-- Bytes of L2 SRAM occupied by result staging for 3x3 conv. -/
def resL2Bytes3x3 (target : Target) (sh : Conv3x3Shape) (t : Tiling) : Nat :=
  if isDirect3x3 target sh t then
    ((1 : Nat) <<< (t.f + effOcg sh t + t.fat)) * 512
  else
    min t.tileHeight sh.outH * resRowStride3x3 sh t

/-- Total L2 SRAM footprint for 3x3 conv (`srcL2Bytes3x3 + resL2Bytes3x3`). -/
def l2UseBytes3x3 (target : Target) (sh : Conv3x3Shape) (t : Tiling) : Nat :=
  srcL2Bytes3x3 sh t + resL2Bytes3x3 target sh t

/-- Validate basic 3x3 conv shape bounds and mode/padding constraints (Rule 1). -/
def validateConv3x3Shape (sh : Conv3x3Shape) : Except String Unit := do
  if sh.h == 0 || sh.h > 65535 then
    throw s!"invalid H={sh.h}: must be in 1..65535"
  if sh.w == 0 || sh.w > 65535 then
    throw s!"invalid W={sh.w}: must be in 1..65535"
  if sh.cin == 0 || sh.cin > maxU32Nat then
    throw s!"invalid Cin={sh.cin}: must be in 1..{maxU32Nat}"
  if sh.cout == 0 || sh.cout > maxU32Nat then
    throw s!"invalid Cout={sh.cout}: must be in 1..{maxU32Nat}"
  if sh.stride != 1 && sh.stride != 2 then
    throw s!"invalid stride={sh.stride}: must be 1 or 2"
  if sh.mode == .depthwise && sh.cin != sh.cout then
    throw s!"invalid depthwise shape: Cin={sh.cin} != Cout={sh.cout}"
  if sh.pad == .valid && (sh.h < 3 || sh.w < 3) then
    throw s!"invalid valid-padded 3x3 shape ({sh.h}x{sh.w}): requires H >= 3 and W >= 3"
  if sh.outH == 0 || sh.outW == 0 then
    throw s!"invalid 3x3 output shape ({sh.outH}x{sh.outW}): Ho and Wo must be >= 1"
  pure ()

/-- Decidable legality validator enforcing all 12 hardware tiling rules for single-task fp16 3x3 conv. -/
def validateConv3x3Tiling (target : Target) (sh : Conv3x3Shape) (t : Tiling) : Except String Unit := do
  -- Rule 1: Valid shape
  validateConv3x3Shape sh

  let eff := sh.effMode
  let ho := sh.outH
  let wo := sh.outW

  -- Rule 2: NE count & chunk slots
  if t.logNE > 4 then
    throw s!"illegal 3x3 tiling: logNE={t.logNE} exceeds 4"

  -- Rule 3: FatTile field width
  if t.fat > 1 then
    throw s!"illegal 3x3 tiling: fat={t.fat} must be 0 or 1"

  -- Rule 4: Work-unit split (WU = logNE - F - fat <= 3)
  if t.f + t.fat > t.logNE then
    throw s!"illegal 3x3 tiling: F({t.f}) + fat({t.fat}) > logNE({t.logNE})"
  let wu := t.logNE - t.f - t.fat
  if wu > 3 then
    throw s!"illegal 3x3 tiling: WU={wu} exceeds 2-bit limit 3 (logNE={t.logNE}, F={t.f}, fat={t.fat})"

  -- Rule 5: Output channels per NE (ocg)
  match eff with
  | .depthwise =>
    if t.ocg != 0 then
      throw s!"illegal 3x3 tiling: depthwise mode requires ocg=0, got ocg={t.ocg}"
  | .dense =>
    if target == .h17 && sh.stride == 1 then
      if t.ocg > 3 then
        throw s!"illegal 3x3 tiling: h17 stride-1 dense ocg={t.ocg} exceeds maximum 3"
    else
      if t.ocg > 4 then
        throw s!"illegal 3x3 tiling: ocg={t.ocg} exceeds maximum 4"

  -- Rule 6 & Rule 7: Output channel coverage and group bound
  let maxF := min 4 (ceilLog2 sh.cout)
  if t.f > maxF then
    throw s!"illegal 3x3 tiling: F={t.f} exceeds min(4, ceil_log2(Cout={sh.cout}))={maxF}"

  match eff with
  | .depthwise =>
    if sh.cout >= 16 && t.f != 4 then
      throw s!"illegal 3x3 tiling: depthwise with C={sh.cout} >= 16 requires F=4, got F={t.f}"
  | .dense =>
    let singleCap := (1 : Nat) <<< (t.f + t.ocg)
    if sh.stride == 2 then
      if sh.cout > singleCap then
        throw s!"illegal 3x3 tiling: stride-2 dense Cout={sh.cout} exceeds single-round coverage {singleCap} (F={t.f}, ocg={t.ocg})"
    else
      match target with
      | .h16 =>
        if sh.cout > singleCap then
          throw s!"illegal 3x3 tiling: h16 dense Cout={sh.cout} exceeds single-round coverage {singleCap} (F={t.f}, ocg={t.ocg})"
      | .h17 =>
        let multiRoundOk := t.f == 4 && t.ocg == 3
        if sh.cout > singleCap && !multiRoundOk then
          throw s!"illegal 3x3 tiling: h17 stride-1 dense Cout={sh.cout} exceeds coverage {singleCap} (F={t.f}, ocg={t.ocg}); Cout > 128 requires F=4, ocg=3"

  -- Rule 8: Strip width (8 <= T <= 256)
  let stripW := stripWidth t
  if stripW < 8 || stripW > 256 then
    throw s!"illegal 3x3 tiling: strip width T={stripW} outside [8, 256]"

  -- Rule 9: Work-unit shifts (0 <= shPref <= shMax <= 4)
  if t.shPref > t.shMax || t.shMax > 4 then
    throw s!"illegal 3x3 tiling: shifts shPref={t.shPref}, shMax={t.shMax} violate 0 <= shPref <= shMax <= 4"

  -- Rule 10: Width cover rule (on output width Wo)
  let cMin := coverMin wo stripW t.fat
  if cMin > t.shMax then
    throw s!"illegal 3x3 tiling: coverMin(Wo={wo}, T={stripW}, fat={t.fat})={cMin} > shMax={t.shMax}"

  -- Rule 11: Tile height (on output height Ho)
  if t.tileHeight == 0 || t.tileHeight > ho then
    throw s!"illegal 3x3 tiling: tileHeight={t.tileHeight} outside 1..{ho}"

  -- Rule 12: L2 footprint
  let l2Use := l2UseBytes3x3 target sh t
  if l2Use > l2CapacityBytes then
    throw s!"illegal 3x3 tiling: L2 footprint {l2Use} bytes exceeds {l2CapacityBytes} bytes (2 MiB)"

  pure ()

/-- Choose legal `ocg` for 3x3 conv `(target, sh, f)` under rules 5 and 6. -/
def chooseConv3x3Ocg? (target : Target) (sh : Conv3x3Shape) (f : Nat) : Option Nat :=
  match sh.effMode with
  | .depthwise => some 0
  | .dense =>
    let need := ceilDiv sh.cout ((1 : Nat) <<< f)
    let ocg1 := ceilLog2 need
    if target == .h17 && sh.stride == 1 then
      if ocg1 <= 3 then
        some ocg1
      else if f == 4 then
        some 3
      else
        none
    else
      if ocg1 <= 4 then
        some ocg1
      else
        none

/-- Compute maximal legal `tileHeight` in O(1) for a candidate 3x3 tiling configuration. -/
def maxTileHeight3x3 (target : Target) (sh : Conv3x3Shape) (probe : Tiling) : Nat := Id.run do
  let ho := sh.outH
  if ho == 0 then
    return 0
  -- Step 1: test th = Ho directly
  let fullCand : Tiling := { probe with tileHeight := ho }
  if l2UseBytes3x3 target sh fullCand <= l2CapacityBytes then
    return ho

  -- Step 2: compute exact tight upper bound thBound using inRows(th) >= (th - 1) * s + rk - pt
  let s := sh.stride
  let rk := sh.rk
  let pt := sh.padTop
  let srcRow := srcRowStride3x3 sh probe
  let direct := isDirect3x3 target sh probe

  let thBound : Nat :=
    if direct then
      let resBytes := ((1 : Nat) <<< (probe.f + effOcg sh probe + probe.fat)) * 512
      if resBytes >= l2CapacityBytes || srcRow == 0 then 0
      else
        let maxInRows := (l2CapacityBytes - resBytes) / srcRow
        if maxInRows + pt < rk then 0
        else min ho ((maxInRows + pt - rk) / s + 1)
    else
      let resRow := resRowStride3x3 sh probe
      -- rk - pt - s >= 1 for all s in {1, 2}, pt in {0, 1}, rk in {3, 4}
      let fixedSrc := (rk - pt - s) * srcRow
      let perTh := s * srcRow + resRow
      if fixedSrc >= l2CapacityBytes || perTh == 0 then 0
      else min ho ((l2CapacityBytes - fixedSrc) / perTh)

  if thBound == 0 then
    return 0

  -- Step 3: refine over at most {thBound, thBound - 1, thBound - 2}
  for dec in [:3] do
    if thBound > dec then
      let th := thBound - dec
      if th >= 1 then
        let cand : Tiling := { probe with tileHeight := th }
        if l2UseBytes3x3 target sh cand <= l2CapacityBytes then
          return th
  return 0

/-- Deterministic independent tile chooser for single-task fp16 3x3 conv (dense and depthwise).
    Computes maximal legal `tileHeight` in O(1) per candidate configuration and validates the
    selected tiling through `validateConv3x3Tiling`. -/
def chooseConv3x3Tiling (target : Target) (sh : Conv3x3Shape) : Except String Tiling := do
  validateConv3x3Shape sh

  let eff := sh.effMode
  let wo := sh.outW
  let maxF := min 4 (ceilLog2 sh.cout)
  let fCandidates : Array Nat :=
    if eff == .depthwise && sh.cout >= 16 then
      #[4]
    else
      Id.run do
        let mut fs : Array Nat := #[]
        for step in [:maxF + 1] do
          fs := fs.push (maxF - step)
        return fs

  let logNEs : Array Nat := #[4, 3, 2, 1, 0]
  let fats : Array Nat := #[0, 1]

  for logNE in logNEs do
    for fat in fats do
      for f in fCandidates do
        if f + fat <= logNE && (logNE - f - fat) <= 3 then
          if let some ocg := chooseConv3x3Ocg? target sh f then
            for shPref in [:5] do
              let stripW := (128 <<< fat) >>> shPref
              if stripW >= 8 && stripW <= 256 then
                let cMin := coverMin wo stripW fat
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
                    let maxTh := maxTileHeight3x3 target sh probe
                    if maxTh >= 1 then
                      let cand : Tiling := { probe with tileHeight := maxTh }
                      match validateConv3x3Tiling target sh cand with
                      | .ok () => return cand
                      | .error _ => pure ()

  throw s!"no legal {target.name} tiling found for 3x3 {sh.mode.name} conv shape (H={sh.h}, W={sh.w}, Cin={sh.cin}, Cout={sh.cout}, s={sh.stride}, pad={sh.pad.name})"

end Vane.Tiling
