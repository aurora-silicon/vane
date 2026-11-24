import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Hwx
import Vane.Sections
import Vane.Register
import Vane.Tiling
import Vane.Program
import Vane.Verify

namespace Vane.Lut

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Hwx
open Vane.Sections
open Vane.Register
open Vane.Tiling
open Vane.Program
open Vane.Verify

/-- Specification for all mathematically generated 128-byte Neural Engine LUT activations. -/
inductive LutSpec where
  -- Piecewise-linear (k = 0x40, mode = 0x01 / 0x21)
  | relu6
  | clip (lo hi : Float)
  | leakyRelu (alpha : Float := 0.01)
  | abs
  | sign
  | threshold (a : Float := 0.0)
  | thresholdedRelu (a : Float := 1.0)
  | hardSigmoid (alpha : Float := 0.2) (beta : Float := 0.5)
  | clampMax (v : Float := 0.0)
  | clampMin (v : Float := 0.0)
  -- Smooth & transcendental
  | elu (alpha : Float := 1.0)
  | tanh
  | scaledTanh (alpha : Float := 1.0) (beta : Float := 1.0)
  | erf
  | sigmoid
  | gelu
  | silu
  | exp2
  | exp
  | sqrt
  | rsqrt (eps : Float := 0.0)
  | inverse
  | log2
  deriving Repr, BEq, Inhabited

namespace LutSpec

def name : LutSpec → String
  | .relu6 => "relu6"
  | .clip _ _ => "clip"
  | .leakyRelu _ => "leaky_relu"
  | .abs => "abs"
  | .sign => "sign"
  | .threshold _ => "threshold"
  | .thresholdedRelu _ => "thresholded_relu"
  | .hardSigmoid _ _ => "hard_sigmoid"
  | .clampMax _ => "clamp_max"
  | .clampMin _ => "clamp_min"
  | .elu _ => "elu"
  | .tanh => "tanh"
  | .scaledTanh _ _ => "scaled_tanh"
  | .erf => "erf"
  | .sigmoid => "sigmoid"
  | .gelu => "gelu"
  | .silu => "silu"
  | .exp2 => "exp2"
  | .exp => "exp"
  | .sqrt => "sqrt"
  | .rsqrt _ => "rsqrt"
  | .inverse => "inverse"
  | .log2 => "log2"

end LutSpec

/-- Standalone Neural Engine activation specification (`relu`, `copy`, or any 128-byte LUT op). -/
inductive NeActSpec where
  | relu
  | copy
  | lut (spec : LutSpec)
  deriving Repr, BEq, Inhabited

namespace NeActSpec

def name : NeActSpec → String
  | .relu => "relu"
  | .copy => "copy"
  | .lut s => s.name

end NeActSpec

/-- Generated 128-byte LUT table payload and associated NE pre-LUT scale / input shift registers. -/
structure LutPayload where
  words : Array UInt16
  bytes : ByteArray
  preScaleBits : UInt16
  inputShiftBits : Option UInt16
  deriving BEq, Inhabited

/-- Encode 64 little-endian `UInt16` words into a 128-byte `ByteArray`. -/
private def packLutWords (words : Array UInt16) : Except String ByteArray := do
  if words.size != 64 then
    throw s!"internal error: LUT word count {words.size} != 64"
  let mut bs := ByteArray.emptyWithCapacity lutTableBytes
  for w in words do
    bs := pushU16LE bs w
  pure bs

/-- Assemble the 64-word (128-byte) hardware LUT table with reserved words `43..63 = 0`. -/
private def assembleLut
    (lo hi outLo outHi : UInt16)
    (samples : Array UInt16)
    (ls li rs ri : UInt16)
    (k mode : UInt16)
    (preScaleBits : UInt16 := f16PosOne)
    (inputShiftBits : Option UInt16 := none) : Except String LutPayload := do
  if samples.size != 33 then
    throw s!"internal error: LUT sample count {samples.size} != 33"
  let mut words : Array UInt16 := Array.mkEmpty 64
  words := words.push lo
  words := words.push hi
  words := words.push outLo
  words := words.push outHi
  for y in samples do
    words := words.push y
  words := words.push ls
  words := words.push li
  words := words.push rs
  words := words.push ri
  words := words.push k
  words := words.push mode
  for _ in [:21] do
    words := words.push 0
  let bytes ← packLutWords words
  pure {
    words := words
    bytes := bytes
    preScaleBits := preScaleBits
    inputShiftBits := inputShiftBits
  }

/-- Helper for piecewise-linear tables (`k = 0x40`, all 33 samples `y[0..32] = f0`). -/
private def assemblePwlLut
    (lo hi outLo outHi f0 ls li rs ri : UInt16)
    (mode : UInt16 := 0x0001)
    (preScaleBits : UInt16 := f16PosOne)
    (inputShiftBits : Option UInt16 := none) : Except String LutPayload :=
  assembleLut lo hi outLo outHi (Array.replicate 33 f0) ls li rs ri 0x0040 mode preScaleBits inputShiftBits

/-- Mathematically generate the 128-byte hardware LUT table for any `LutSpec`. -/
def buildLutTable (spec : LutSpec) : Except String LutPayload := do
  let f16 := floatToF16CanonicalZero
  match spec with
  | .relu6 =>
    assemblePwlLut f16PosZero (f16 6.0) f16PosZero (f16 6.0) f16PosZero
      f16PosOne f16PosZero f16PosOne f16PosZero 0x0001
  | .clip lo hi =>
    if lo.isNaN || hi.isNaN then
      throw "clip: lo and hi must not be NaN"
    if lo > hi then
      throw s!"clip: lo ({lo}) must be <= hi ({hi})"
    let f0 := max lo (min hi 0.0)
    assemblePwlLut (f16 lo) (f16 hi) (f16 lo) (f16 hi) (f16 f0)
      f16PosOne f16PosZero f16PosOne f16PosZero 0x0001
  | .leakyRelu alpha =>
    if alpha.isNaN then
      throw "leaky_relu: alpha must not be NaN"
    assemblePwlLut f16NegInf f16PosInf f16NegInf f16PosInf f16PosZero
      (f16 alpha) f16PosZero f16PosOne f16PosZero 0x0001
  | .abs =>
    assemblePwlLut f16NegInf f16PosInf f16PosInf f16PosInf f16PosZero
      f16NegOne f16PosZero f16PosOne f16PosZero 0x0021
  | .sign =>
    assemblePwlLut f16NegMinSubnormal f16MinSubnormal f16NegOne f16PosOne f16PosZero
      f16PosZero f16PosZero f16PosZero f16PosZero 0x0001
  | .threshold a =>
    if a.isNaN then
      throw "threshold: threshold a must not be NaN"
    let f0 := max 0.0 a
    assemblePwlLut (f16 a) f16PosInf (f16 a) f16PosInf (f16 f0)
      f16PosOne f16PosZero f16PosOne f16PosZero 0x0001
  | .thresholdedRelu a =>
    if a.isNaN then
      throw "thresholded_relu: threshold a must not be NaN"
    let loBits := prevF16 (f16 a)
    assemblePwlLut loBits f16PosInf f16PosZero f16PosInf f16PosZero
      f16PosOne f16PosZero f16PosOne f16PosZero 0x0001
  | .hardSigmoid alpha beta =>
    if alpha.isNaN || alpha.isInf || alpha <= 0.0 then
      throw s!"hard_sigmoid: alpha ({alpha}) must be finite and > 0"
    if beta.isNaN || beta.isInf then
      throw s!"hard_sigmoid: beta ({beta}) must be finite"
    let scale := alpha / 0.2
    let preScale := f16 scale
    let shift? : Option UInt16 :=
      if beta == 0.5 then none
      else
        let s := f16 ((5.0 * beta - 2.5) / scale)
        if s == f16PosZero then none else some s
    assemblePwlLut (f16 (-2.5)) (f16 2.5) f16PosZero f16PosOne (f16 0.5)
      (f16 0.2) (f16 0.5) (f16 0.2) (f16 0.5) 0x0001 preScale shift?
  | .clampMax v =>
    if v.isNaN then
      throw "clamp_max: bound v must not be NaN"
    let f0 := max 0.0 v
    assemblePwlLut (f16 v) f16PosInf (f16 v) f16PosInf (f16 f0)
      f16PosOne f16PosZero f16PosOne f16PosZero 0x0001
  | .clampMin v =>
    if v.isNaN then
      throw "clamp_min: bound v must not be NaN"
    let f0 := min 0.0 v
    assemblePwlLut f16NegInf (f16 v) f16NegInf (f16 v) (f16 f0)
      f16PosOne f16PosZero f16PosOne f16PosZero 0x0001
  | .elu alpha =>
    if alpha.isNaN || alpha.isInf then
      throw s!"elu: alpha ({alpha}) must be finite"
    let samples := Array.ofFn (n := 33) fun i =>
      let u := (Float.ofNat i.val - 32.0) / 4.0
      f16 (alpha * expm1 u)
    assembleLut (f16 (-8.0)) f16PosInf (f16 (-alpha)) f16PosInf samples
      f16PosZero (f16 (-alpha)) f16PosOne f16PosZero 2 0x0002
  | .tanh =>
    let samples := Array.ofFn (n := 33) fun i =>
      f16 (Float.tanh (Float.ofNat i.val / 8.0))
    assembleLut (f16 (-4.0)) (f16 4.0) f16NegOne f16PosOne samples
      f16PosZero f16NegOne f16PosZero f16PosOne 3 0x0060
  | .scaledTanh alpha beta =>
    if alpha.isNaN || alpha.isInf || beta.isNaN || beta.isInf then
      throw "scaled_tanh: alpha and beta must be finite"
    let a := Float.abs alpha
    let mode : UInt16 := if alpha >= 0.0 then 0x0060 else 0x00a0
    let samples := Array.ofFn (n := 33) fun i =>
      f16 (a * Float.tanh (Float.ofNat i.val / 8.0))
    assembleLut (f16 (-4.0)) (f16 4.0) (f16 (-a)) (f16 a) samples
      f16PosZero (f16 (-a)) f16PosZero (f16 a) 3 mode (f16 beta)
  | .erf =>
    let samples := Array.ofFn (n := 33) fun i =>
      f16 (erfApprox (Float.ofNat i.val / 8.0))
    assembleLut (f16 (-4.0)) (f16 4.0) f16NegOne f16PosOne samples
      f16PosZero f16NegOne f16PosZero f16PosOne 3 0x0060
  | .sigmoid =>
    let samples := Array.ofFn (n := 33) fun i =>
      let u := (Float.ofNat i.val - 16.0) / 2.0
      f16 (1.0 / (1.0 + Float.exp (-u)))
    assembleLut (f16 (-8.0)) (f16 8.0) f16PosZero f16PosOne samples
      f16PosZero f16PosZero f16PosZero f16PosOne 1 0x0011
  | .gelu =>
    let invSqrt2 := 0.707106781186547524400844362104849039
    let samples := Array.ofFn (n := 33) fun i =>
      let u := (Float.ofNat i.val - 16.0) / 4.0
      f16 (0.5 * u * (1.0 + erfApprox (u * invSqrt2)))
    assembleLut (f16 (-4.0)) f16PosInf f16PosZero f16PosInf samples
      f16PosZero f16PosZero f16PosOne f16PosZero 2 0x0011
  | .silu =>
    let samples := Array.ofFn (n := 33) fun i =>
      let u := (Float.ofNat i.val - 16.0) / 2.0
      f16 (u / (1.0 + Float.exp (-u)))
    assembleLut (f16 (-8.0)) f16PosInf f16PosZero f16PosInf samples
      f16PosZero f16PosZero f16PosOne f16PosZero 1 0x0011
  | .exp2 =>
    let samples := Array.ofFn (n := 33) fun i =>
      f16 (exp2Float (Float.ofNat i.val / 32.0))
    assembleLut (f16 (-24.0)) (f16 16.0) f16PosZero f16PosInf samples
      f16PosZero f16PosZero f16PosZero f16PosZero 5 0x000a
  | .exp =>
    let samples := Array.ofFn (n := 33) fun i =>
      f16 (exp2Float (Float.ofNat i.val / 32.0))
    assembleLut (f16 (-24.0)) (f16 16.0) f16PosZero f16PosInf samples
      f16PosZero f16PosZero f16PosZero f16PosZero 5 0x000a fp16InvLn2Bits
  | .sqrt =>
    let samples := Array.ofFn (n := 33) fun i =>
      f16 (Float.sqrt (Float.ofNat i.val / 16.0))
    assembleLut f16PosZero f16PosInf f16PosZero f16PosInf samples
      f16PosZero f16PosZero f16PosZero f16PosZero 4 0x0005
  | .rsqrt eps =>
    if eps.isNaN || eps.isInf || eps < 0.0 then
      throw s!"rsqrt: eps ({eps}) must be finite and >= 0"
    let samples := Array.ofFn (n := 33) fun i =>
      if i.val == 0 then f16PosZero
      else f16 (1.0 / Float.sqrt (Float.ofNat i.val / 16.0))
    let shift? : Option UInt16 :=
      if eps == 0.0 then none
      else
        let s := f16 eps
        if s == f16PosZero then none else some s
    assembleLut f16PosZero f16PosInf f16PosInf f16PosZero samples
      f16PosZero f16PosZero f16PosZero f16PosZero 4 0x0006 f16PosOne shift?
  | .inverse =>
    let samples := Array.ofFn (n := 33) fun i =>
      if i.val == 0 then f16PosZero
      else f16 (32.0 / Float.ofNat i.val)
    assembleLut f16NegInf f16PosInf f16PosZero f16PosZero samples
      f16PosZero f16PosZero f16PosZero f16PosZero 5 0x0067
  | .log2 =>
    let samples := Array.ofFn (n := 33) fun i =>
      if i.val == 0 then f16NegInf
      else f16 (log2Float (Float.ofNat i.val / 32.0))
    assembleLut f16PosZero f16PosInf f16NegInf f16PosInf samples
      f16PosZero f16PosZero f16PosZero f16PosZero 5 0x0009

/-- Linear interpolation over 33 table entries `y[0..32]` clamped to `[0, 32]`. -/
private def lerp33 (y : Array Float) (u : Float) : Float :=
  let uc := if u < 0.0 then 0.0 else if u > 32.0 then 32.0 else u
  let fl := Float.floor uc
  let i0 : Nat := min fl.toUInt64.toNat 31
  let frac := uc - Float.ofNat i0
  let y0 := y[i0]!
  let y1 := y[i0 + 1]!
  y0 + frac * (y1 - y0)

/-- Software model of the 128-byte NE LUT evaluator across all hardware grid modes and symmetry flags. -/
def evalLutTable
    (words : Array UInt16)
    (preScaleBits : UInt16)
    (inputShiftBits : Option UInt16)
    (xBits : UInt16) : UInt16 := Id.run do
  if words.size < 43 then
    return f16PosZero
  let mut x := f16ToFloat xBits
  if x.isNaN then
    return f16PosZero
  if let some shiftBits := inputShiftBits then
    x := x + f16ToFloat shiftBits
  if preScaleBits != f16PosOne then
    x := x * f16ToFloat preScaleBits
  if x.isNaN then
    return f16PosZero

  let lo := f16ToFloat words[0]!
  let hi := f16ToFloat words[1]!
  let outLo := f16ToFloat words[2]!
  let outHi := f16ToFloat words[3]!
  let y : Array Float := Array.ofFn (n := 33) fun i => f16ToFloat words[4 + i.val]!
  let ls := f16ToFloat words[37]!
  let li := f16ToFloat words[38]!
  let rs := f16ToFloat words[39]!
  let ri := f16ToFloat words[40]!
  let k := words[41]!.toNat
  let mode := words[42]!.toNat
  let sym := mode &&& 0xe0
  let base := mode &&& 0x1f

  let (negOut, xp) ← match sym with
    | 0x00 => pure (false, x)
    | 0x20 => pure (false, Float.abs x)
    | 0x60 => if x < 0.0 then pure (true, -x) else pure (false, x)
    | 0x80 => pure (true, x)
    | 0xa0 => if x < 0.0 then pure (false, -x) else pure (true, x)
    | 0xc0 =>
      if x < 0.0 then
        return f16PosZero
      else
        pure (false, x)
    | _ => pure (false, x)

  let rawOut : Float :=
    if xp <= lo then
      outLo
    else if xp >= hi then
      outHi
    else
      match base with
      | 0x01 =>
        if xp == 0.0 then y[0]!
        else if xp < 0.0 then ls * xp + li
        else rs * xp + ri
      | 0x00 =>
        let scale := ldexp 1.0 (k : Int)
        let dom := 32.0 / scale
        if xp < 0.0 then ls * xp + li
        else if xp >= dom then rs * xp + ri
        else lerp33 y (xp * scale)
      | 0x02 =>
        let scale := ldexp 1.0 (k : Int)
        let dom := 32.0 / scale
        if xp < -dom then ls * xp + li
        else if xp >= 0.0 then rs * xp + ri
        else lerp33 y (32.0 + xp * scale)
      | 0x11 =>
        let scale := ldexp 1.0 (k : Int)
        let dom := 16.0 / scale
        if xp < -dom then ls * xp + li
        else if xp >= dom then rs * xp + ri
        else lerp33 y (16.0 + xp * scale)
      | 0x05 =>
        let (m, e) := frexpPos xp
        let (m2, q) := if e % 2 == 0 then (m, e / 2) else (2.0 * m, (e - 1) / 2)
        ldexp (lerp33 y (ldexp m2 (k : Int))) q
      | 0x06 =>
        let (m, e) := frexpPos xp
        let (m2, q) := if e % 2 == 0 then (m, e / 2) else (2.0 * m, (e - 1) / 2)
        ldexp (lerp33 y (ldexp m2 (k : Int))) (-q)
      | 0x07 =>
        if xp == 0.0 then
          Float.ofBits 0x7ff0000000000000
        else
          let (m, e) := frexpPos xp
          ldexp (lerp33 y (ldexp m (k : Int))) (-e)
      | 0x09 =>
        let (m, e) := frexpPos xp
        let eF : Float := if e >= 0 then Float.ofNat e.toNat else -(Float.ofNat e.natAbs)
        eF + lerp33 y (ldexp m (k : Int))
      | 0x0a =>
        let s := ldexp xp (k : Int)
        let qF := Float.floor (s / 32.0)
        let q : Int := if qF >= 0.0 then (qF.toUInt64.toNat : Int) else -((-qF).toUInt64.toNat : Int)
        let u := s - 32.0 * qF
        ldexp (lerp33 y u) q
      | _ => 0.0

  let finalVal := if negOut then -rawOut else rawOut
  return rna16 finalVal

-- ============================================================================
-- Standalone NE Activation Tiling, Lowering, and Compiler
-- ============================================================================

/-- Tiling parameters for a standalone NE activation task (`ocg = 0`). -/
structure NeActTiling where
  logNE : Nat
  f : Nat
  fat : Nat
  shPref : Nat
  shMax : Nat
  tileHeight : Nat
  deriving Repr, BEq, DecidableEq, Inhabited

def NeActTiling.wu (t : NeActTiling) : Nat :=
  t.logNE - t.f - t.fat

def NeActTiling.stripWidth (t : NeActTiling) : Nat :=
  (128 <<< t.fat) >>> t.shPref

/-- Whether a standalone NE activation runs in direct result mode. -/
def neActIsDirect (target : Target) (w : Nat) (t : NeActTiling) : Bool :=
  let lhs := match target with
    | .h17 => (128 <<< t.fat) >>> t.shMax
    | .h16 => 128 >>> t.shMax
  lhs >= min 32 w

/-- L2 SRAM layout for a standalone NE activation task. -/
structure NeActL2Layout where
  stripW : Nat
  row : Nat
  u : Nat
  size1 : Nat
  direct : Bool
  resBase : Nat
  drow : Nat
  du : Nat
  resL2Bytes : Nat
  l2UseBytes : Nat
  deriving Repr, BEq, Inhabited

def computeNeActL2 (target : Target) (c _h w : Nat) (t : NeActTiling) : NeActL2Layout :=
  let stripW := t.stripWidth
  let row := 2 * min (alignUp w 8) stripW
  let u := c * row
  let size1 := t.tileHeight * u
  let direct := neActIsDirect target w t
  let resBase := alignUp size1 dmaLineBytes
  let drow := 2 * min (alignUp w 8) (max stripW 32)
  let du := c * drow
  let resL2 := if direct then 0 else t.tileHeight * du
  {
    stripW := stripW
    row := row
    u := u
    size1 := size1
    direct := direct
    resBase := resBase
    drow := drow
    du := du
    resL2Bytes := resL2
    l2UseBytes := resBase + resL2
  }

/-- Complete compilation artifact for a standalone NE activation task. -/
structure CompiledNeAct where
  target : Target
  act : NeActSpec
  c : Nat
  h : Nat
  w : Nat
  tiling : NeActTiling
  direct : Bool
  l2UseBytes : Nat
  lutPayload : Option LutPayload
  hwx : ByteArray
  info : HwxInfo
  sections : LoadSections

/-- Validate standalone NE activation shape and tiling legality. -/
def validateNeActTiling (target : Target) (c h w : Nat) (t : NeActTiling) : Except String Unit := do
  if c == 0 || c > maxU32Nat then
    throw s!"invalid NE activation C={c}: must be in 1..{maxU32Nat}"
  if h == 0 || h > 65535 then
    throw s!"invalid NE activation H={h}: must be in 1..65535"
  if w == 0 || w > 65535 then
    throw s!"invalid NE activation W={w}: must be in 1..65535"
  if t.logNE > 4 then
    throw s!"illegal NE activation tiling: logNE={t.logNE} exceeds 4"
  if t.fat > 1 then
    throw s!"illegal NE activation tiling: fat={t.fat} must be 0 or 1"
  if t.f + t.fat > t.logNE then
    throw s!"illegal NE activation tiling: F({t.f}) + fat({t.fat}) > logNE({t.logNE})"
  if t.wu > 3 then
    throw s!"illegal NE activation tiling: WU={t.wu} exceeds 3"
  if t.shPref > t.shMax || t.shMax > 4 then
    throw s!"illegal NE activation tiling: invalid shifts shPref={t.shPref}, shMax={t.shMax}"
  let stripW := t.stripWidth
  if stripW < 8 || stripW > 256 then
    throw s!"illegal NE activation tiling: strip width T={stripW} outside [8, 256]"
  if t.tileHeight == 0 || t.tileHeight > h then
    throw s!"illegal NE activation tiling: tileHeight={t.tileHeight} out of range 1..{h}"
  let l2 := computeNeActL2 target c h w t
  if l2.l2UseBytes > l2CapacityBytes then
    throw s!"illegal NE activation tiling: L2 footprint {l2.l2UseBytes}B exceeds {l2CapacityBytes}B (2 MiB)"
  pure ()

/-- Deterministic O(1) tile chooser for standalone NE activations. -/
def chooseNeActTiling (target : Target) (c h w : Nat) : Except String NeActTiling := do
  if c == 0 || c > maxU32Nat || h == 0 || h > 65535 || w == 0 || w > 65535 then
    throw s!"invalid NE activation shape (C={c}, H={h}, W={w})"
  for shPref in [:5] do
    let probe : NeActTiling := {
      logNE := 0
      f := 0
      fat := 0
      shPref := shPref
      shMax := shPref
      tileHeight := 1
    }
    let stripW := probe.stripWidth
    let row := 2 * min (alignUp w 8) stripW
    let u := c * row
    let direct := neActIsDirect target w probe
    let drow := 2 * min (alignUp w 8) (max stripW 32)
    let du := c * drow
    let perRow := if direct then u else u + du
    if perRow > 0 then
      let thUpper := min h (l2CapacityBytes / perRow)
      for dec in [:4] do
        if thUpper > dec then
          let th := thUpper - dec
          if th >= 1 then
            let cand : NeActTiling := { probe with tileHeight := th }
            match validateNeActTiling target c h w cand with
            | .ok () => return cand
            | .error _ => pure ()
  throw s!"no legal {target.name} NE activation tiling found for (C={c}, H={h}, W={w})"

/-- Lower a standalone NE activation task into a `TaskDesc`. -/
def lowerNeActTask
    (target : Target)
    (act : NeActSpec)
    (lutPayload? : Option LutPayload)
    (c h w : Nat)
    (tiling : NeActTiling)
    (taskId : Nat := 0) : Except String TaskDesc := do
  validateNeActTiling target c h w tiling
  let l2 := computeNeActL2 target c h w tiling

  let (isLut, isRelu, preScaleBits, shiftBits?, header4) ← match act with
    | .relu => pure (false, true, f16PosOne, none, taskHeaderWord4Const)
    | .copy => pure (false, false, f16PosOne, none, taskHeaderWord4CopyConst)
    | .lut _ =>
      let some lp := lutPayload?
        | throw "internal error: lowerNeActTask missing LutPayload for LUT activation"
      pure (true, false, lp.preScaleBits, lp.inputShiftBits, taskHeaderWord4Const)

  let wU32 ← natToU32Checked w "W"
  let hU32 ← natToU32Checked h "H"
  let cU32 ← natToU32Checked c "C"
  let thU32 ← natToU32Checked tiling.tileHeight "TH"
  let shPrefU32 ← natToU32Checked tiling.shPref "shPref"
  let shMaxU32 ← natToU32Checked tiling.shMax "shMax"
  let fU32 ← natToU32Checked tiling.f "F"
  let fatU32 ← natToU32Checked tiling.fat "fat"
  let wuU32 ← natToU32Checked tiling.wu "WU"
  let taskIdU32 ← natToU32Checked taskId "taskId"

  let wWin ← encodeWord (dim16Layout "Win") #[("val", wU32)]
  let wHin ← encodeWord (dim16Layout "Hin") #[("val", hU32)]
  let wWout ← encodeWord (dim16Layout "Wout") #[("val", wU32)]
  let wHout ← encodeWord (dim16Layout "Hout") #[("val", hU32)]
  let wDwGroups ← encodeWord dwGroupsLayout #[("perGroupIn", 1)]
  let wTH ← encodeWord (dim16Layout "TH") #[("val", thU32)]
  let wNeCommon ← encodeWord neCommonLayout #[
    ("kind", 0),
    ("shPref", shPrefU32),
    ("shMin", shPrefU32),
    ("shMax", shMaxU32),
    ("F", fU32),
    ("wide3x3", 0)
  ]
  let wNeTile ← encodeWord neTileLayout #[
    ("ocg", 0),
    ("fat", fatU32),
    ("WU", wuU32)
  ]
  let wSrc1Cfg ← encodeWord src1CfgLayout #[("mode", 2), ("P", 1)]
  let wResCfg ← encodeWord resCfgLayout #[
    ("mode", 2),
    ("direct", if l2.direct then 1 else 0),
    ("P", 1)
  ]
  let wNeActOp ← encodeWord neActOpLayout #[
    ("noShift", if shiftBits?.isSome then 0 else 1),
    ("relu", if isRelu then 1 else 0),
    ("lut", if isLut then 1 else 0)
  ]
  let wNeScale ← encodeWord neScaleLayout #[("val", preScaleBits.toUInt32)]
  let wSrc1Fmt ← encodeWord (dmaFmtLayout "src1Fmt") #[("format", 0x002031), ("P", 1)]
  let wResFmt ← encodeWord (dmaFmtLayout "resFmt") #[("format", 0x302031), ("P", 1)]
  let wOut2 ← encodeWord out2Layout #[("form", 0xe0), ("task", taskIdU32)]

  let rowU32 ← natToU32Checked l2.row "src1ChanStride"
  let uU32 ← natToU32Checked l2.u "src1RowStride"
  let resBaseU32 ← natToU32Checked l2.resBase "resBase"

  let dmaRow := alignUp (2 * w) dmaLineBytes
  let dmaPlane := h * dmaRow
  let dmaRowU32 ← natToU32Checked dmaRow "dmaRowStride"
  let dmaPlaneU32 ← natToU32Checked dmaPlane "dmaPlaneStride"

  let mut plain : Array RegWrite := #[
    -- Common block (0x0028 convCfg intentionally omitted!)
    { byteAddr := regWin, value := wWin },
    { byteAddr := regHin, value := wHin },
    { byteAddr := regCin, value := cU32 },
    { byteAddr := regWout, value := wWout },
    { byteAddr := regHout, value := wHout },
    { byteAddr := regCout, value := cU32 },
    { byteAddr := regBatchN, value := 1 },
    { byteAddr := regDwGroups, value := wDwGroups },
    { byteAddr := regTH, value := wTH },
    { byteAddr := regNeCommon, value := wNeCommon },
    { byteAddr := regNeTile, value := wNeTile },
    { byteAddr := regTargetCfg, value := targetCfgValue target },
    -- L2 staging
    { byteAddr := regSrc1Cfg, value := wSrc1Cfg },
    { byteAddr := regSrc2Cfg, value := src2CfgValue },
    { byteAddr := regSrc1Base, value := 0 },
    { byteAddr := regSrc1ChanStride, value := rowU32 },
    { byteAddr := regSrc1RowStride, value := uU32 },
    { byteAddr := regSrc1RowBytesA, value := uU32 },
    { byteAddr := regSrc1RowBytesB, value := uU32 },
    { byteAddr := regResCfg, value := wResCfg },
    { byteAddr := regResBase, value := resBaseU32 },
    -- NE block
    { byteAddr := regNeOp, value := wNeActOp },
    { byteAddr := regNeScale, value := wNeScale },
    -- Tile DMA
    { byteAddr := regSrc1DmaCfg, value := src1DmaCfgValue },
    { byteAddr := regSrc1DmaRowStride, value := dmaRowU32 },
    { byteAddr := regSrc1DmaPlaneStride, value := dmaPlaneU32 },
    { byteAddr := regSrc1Fmt, value := wSrc1Fmt },
    -- Result DMA
    { byteAddr := regResDmaRowStride, value := dmaRowU32 },
    { byteAddr := regResDmaPlaneStride, value := dmaPlaneU32 },
    { byteAddr := regResFmt, value := wResFmt },
    -- Output stage
    { byteAddr := regOut0, value := out0Value },
    { byteAddr := regOut1, value := out1Value },
    { byteAddr := regOut2, value := wOut2 },
    { byteAddr := regOut3, value := out3Value },
    { byteAddr := regOut4, value := out4Value },
    { byteAddr := regOut5, value := out5Value }
  ]

  -- Optional input shift (rsqrt eps / hard_sigmoid beta != 0.5)
  if let some shiftBits := shiftBits? then
    let wShiftEn ← encodeWord neActShiftEnLayout #[("enable", 1)]
    let wShiftVal ← encodeWord scalarBiasLayout #[("val", shiftBits.toUInt32)]
    plain := plain ++ #[
      { byteAddr := regNeActShiftEn, value := wShiftEn },
      { byteAddr := regScalarBias, value := wShiftVal }
    ]

  -- Staged mode result L2 registers
  if !l2.direct then
    let drowU32 ← natToU32Checked l2.drow "resChanStride"
    let duU32 ← natToU32Checked l2.du "resRowStride"
    plain := plain ++ #[
      { byteAddr := regResChanStride, value := drowU32 },
      { byteAddr := regResRowStride, value := duU32 },
      { byteAddr := regResRowBytesA, value := duU32 },
      { byteAddr := regResRowBytesB, value := duU32 }
    ]

  -- Kernel stage & LUT descriptor registers
  let wSrc1Dep ← encodeWord src1DepLayout #[("code", 0x21), ("task", taskIdU32)]
  let wResDep ← encodeWord resDepLayout #[
    ("code", resDepCode target),
    ("task", taskIdU32),
    ("direct", if l2.direct then 1 else 0)
  ]

  let mut barRecs : Array BarRegWrite := #[]
  let mut kernelRelocs : Array TaskReloc := #[]

  if isLut then
    let wKernStage ← encodeWord kernelStageLayout #[("sparse", 0), ("stage", 1)]
    let wLutCfg ← encodeWord chunkSlotLayout #[("enable", 1), ("on20", 1), ("task", taskIdU32)]
    plain := plain ++ #[
      { byteAddr := regNeCfg, value := neActCfgValue },
      { byteAddr := regKernelStage, value := wKernStage },
      { byteAddr := regLutCfg, value := wLutCfg },
      { byteAddr := regLutSize, value := lutTableBytes.toUInt32 }
    ]
    barRecs := barRecs.push { baseByteAddr := regLutBaseLo, barSlot := 8, values := #[0, 0] }
    kernelRelocs := #[{ barRecIndex := 0, payloadWordIndex := 0 }]
  else
    plain := plain.push { byteAddr := regKernelStage, value := kernelStageNoKernValue }

  barRecs := barRecs ++ #[
    { baseByteAddr := regSrc1BaseLo, barSlot := 6, values := #[0, 0] },
    { baseByteAddr := regResBaseLo, barSlot := 7, values := #[0, 0] },
    { baseByteAddr := regSrc1Dep, barSlot := 4, values := #[wSrc1Dep] },
    { baseByteAddr := regResDep, barSlot := 4, values := #[wResDep] }
  ]

  pure {
    taskId := taskId
    logNE := tiling.logNE
    eneOverride := some (tiling.logNE + 1)
    header4 := header4
    plainWrites := plain
    barWrites := barRecs
    kernelRelocs := kernelRelocs
  }

/-- Compile a standalone NE activation (`relu`, `copy`, or any 128-byte LUT op) into a
    statically verified HWX binary and its 4 host-built Linux LOAD sections. -/
def compileNeAct
    (target : Target)
    (act : NeActSpec)
    (c h w : Nat)
    (tilingOverride : Option NeActTiling := none) : Except String CompiledNeAct := do
  let tiling ← match tilingOverride with
    | some t => validateNeActTiling target c h w t; pure t
    | none => chooseNeActTiling target c h w

  let lutPayload? ← match act with
    | .lut spec => some <$> buildLutTable spec
    | _ => pure none

  let taskDesc ← lowerNeActTask target act lutPayload? c h w tiling

  let inSurf ← makeFp16Surface target "x" "input_x" .input c h w
  let outSurf ← makeFp16Surface target "y" "output_y" .output c h w

  let emitted ← emitProgram {
    target := target
    surfaces := #[inSurf, outSurf]
    tasks := #[taskDesc]
    kernel := lutPayload?.map (fun lp => lp.bytes)
    scratch := none
    constants := zeroBytes textConstBytes
    ident := some "vane"
  }

  let _ ← verifyHwxTasks emitted.hwx emitted.info (requireKnownTask := true)
  let l2 := computeNeActL2 target c h w tiling

  pure {
    target := target
    act := act
    c := c
    h := h
    w := w
    tiling := tiling
    direct := l2.direct
    l2UseBytes := l2.l2UseBytes
    lutPayload := lutPayload?
    hwx := emitted.hwx
    info := emitted.info
    sections := emitted.sections
  }

end Vane.Lut
