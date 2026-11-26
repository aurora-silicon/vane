namespace Vane.Float16

/-- IEEE-754 binary16 special bit patterns. -/
def f16PosZero : UInt16 := 0x0000
def f16NegZero : UInt16 := 0x8000
def f16PosOne : UInt16 := 0x3c00
def f16NegOne : UInt16 := 0xbc00
def f16PosInf : UInt16 := 0x7c00
def f16NegInf : UInt16 := 0xfc00
def f16QNaN : UInt16 := 0x7e00
def f16MinSubnormal : UInt16 := 0x0001
def f16NegMinSubnormal : UInt16 := 0x8001

/-- Return `true` iff `h` is an IEEE-754 binary16 NaN. -/
def f16IsNaN (h : UInt16) : Bool :=
  (h &&& 0x7c00) == 0x7c00 && (h &&& 0x03ff) != 0

/-- Return `true` iff `h` is `+inf` or `-inf`. -/
def f16IsInf (h : UInt16) : Bool :=
  (h &&& 0x7fff) == 0x7c00

/-- Return `true` iff `h` is finite (neither NaN nor Inf). -/
def f16IsFinite (h : UInt16) : Bool :=
  (h &&& 0x7c00) != 0x7c00

/-- Exact conversion of an IEEE-754 binary16 bit pattern to IEEE-754 binary64 `Float`.
    Handles normals, subnormals, `+-0`, `+-inf`, and `NaN`. -/
def f16ToFloat (h : UInt16) : Float :=
  let sign : UInt64 := ((h.toUInt64 >>> 15) &&& 1) <<< 63
  let exp : Nat := ((h >>> 10) &&& 0x1f).toNat
  let frac : Nat := (h &&& 0x03ff).toNat
  if exp == 0 then
    if frac == 0 then
      Float.ofBits sign
    else
      Id.run do
        let mut f := frac
        let mut e : Int := -14
        while (f &&& 0x0400) == 0 do
          f := f <<< 1
          e := e - 1
        let fNorm := f &&& 0x03ff
        let e64 : UInt64 := (e + 1023).toNat.toUInt64
        return Float.ofBits (sign ||| (e64 <<< 52) ||| (fNorm.toUInt64 <<< 42))
  else if exp == 0x1f then
    if frac == 0 then
      Float.ofBits (sign ||| (0x7ff0000000000000 : UInt64))
    else
      Float.ofBits (sign ||| (0x7ff8000000000000 : UInt64) ||| (frac.toUInt64 <<< 42))
  else
    let e64 : UInt64 := (exp + (1023 - 15)).toUInt64
    Float.ofBits (sign ||| (e64 <<< 52) ||| (frac.toUInt64 <<< 42))

/-- Exact lossless widening of an IEEE-754 binary16 bit pattern to IEEE-754 binary32 bits.
    Handles normals, subnormals, `+-0`, `+-inf`, and `NaN`. -/
def f16ToF32Bits (h : UInt16) : UInt32 :=
  let sign : UInt32 := ((h.toUInt32 >>> 15) &&& 1) <<< 31
  let exp : Nat := ((h >>> 10) &&& 0x1f).toNat
  let frac : Nat := (h &&& 0x03ff).toNat
  if exp == 0 then
    if frac == 0 then
      sign
    else
      Id.run do
        let mut f := frac
        let mut e : Int := -14
        while (f &&& 0x0400) == 0 do
          f := f <<< 1
          e := e - 1
        let fNorm := f &&& 0x03ff
        let e32 : UInt32 := (e + 127).toNat.toUInt32
        return sign ||| (e32 <<< 23) ||| (fNorm.toUInt32 <<< 13)
  else if exp == 0x1f then
    if frac == 0 then
      sign ||| (0x7f800000 : UInt32)
    else
      sign ||| (0x7fc00000 : UInt32) ||| (frac.toUInt32 <<< 13)
  else
    let e32 : UInt32 := (exp + (127 - 15)).toUInt32
    sign ||| (e32 <<< 23) ||| (frac.toUInt32 <<< 13)

/-- Convert an IEEE-754 binary64 `Float` to IEEE-754 binary32 bits (round-to-nearest-even). -/
def floatToF32Bits (f : Float) : UInt32 :=
  let bits := f.toBits
  let sign : UInt32 := ((bits >>> 32) &&& 0x80000000).toUInt32
  let exp : Nat := ((bits >>> 52) &&& 0x7ff).toNat
  let frac : UInt64 := bits &&& (((1 : UInt64) <<< 52) - 1)
  if exp == 0x7ff then
    if frac == 0 then
      sign ||| 0x7f800000
    else
      let payload := (frac >>> 29).toUInt32 &&& 0x003fffff
      sign ||| 0x7fc00000 ||| (if payload == 0 then 0x00200000 else payload)
  else if exp == 0 then
    sign
  else
    let e : Int := (exp : Int) - 1023
    let sig : UInt64 := ((1 : UInt64) <<< 52) ||| frac
    if e > 128 then
      sign ||| 0x7f800000
    else if e >= -126 then
      let shift : UInt64 := 29
      let q0 := sig >>> shift
      let rem := sig &&& (((1 : UInt64) <<< shift) - 1)
      let half : UInt64 := (1 : UInt64) <<< 28
      let roundUp := rem > half || (rem == half && (q0 &&& 1) == 1)
      let (q, eFin) :=
        if roundUp then
          let q1 := q0 + 1
          if q1 == 0x1000000 then (0x800000, e + 1) else (q1, e)
        else
          (q0, e)
      if eFin > 127 then
        sign ||| 0x7f800000
      else
        sign ||| (((eFin + 127).toNat.toUInt32) <<< 23) ||| (q.toUInt32 &&& 0x007fffff)
    else
      -- Subnormal / underflow: target LSB is 2^-149, sig LSB is 2^(e - 52) -> shift = -97 - e
      let shift : Nat := (-97 - e).toNat
      if shift > 53 then
        sign
      else if shift == 53 then
        if frac > 0 then sign ||| 0x00000001 else sign
      else
        let s64 := shift.toUInt64
        let q0 := sig >>> s64
        let rem := sig &&& (((1 : UInt64) <<< s64) - 1)
        let half : UInt64 := (1 : UInt64) <<< (shift - 1).toUInt64
        let roundUp := rem > half || (rem == half && (q0 &&& 1) == 1)
        let q := if roundUp then q0 + 1 else q0
        if q == 0x800000 then
          sign ||| 0x00800000
        else
          sign ||| q.toUInt32

/-- Direct binary64 `Float` to IEEE-754 binary16 conversion without double rounding.
    When `tiesAway = false`, uses round-to-nearest-even (`NaN -> 0x7e00`).
    When `tiesAway = true`, rounds ties away from zero (`NaN -> 0x7e00`). -/
def floatToF16Rounded (f : Float) (tiesAway : Bool) : UInt16 :=
  let bits := f.toBits
  let sign : UInt16 := ((bits >>> 48) &&& 0x8000).toUInt16
  let exp : Nat := ((bits >>> 52) &&& 0x7ff).toNat
  let frac : UInt64 := bits &&& (((1 : UInt64) <<< 52) - 1)
  if exp == 0x7ff then
    if frac == 0 then
      sign ||| f16PosInf
    else
      f16QNaN
  else if exp == 0 then
    -- Float64 zero or subnormal (|f| < 2^-1022 << 2^-25) -> signed zero
    sign
  else
    let e : Int := (exp : Int) - 1023
    let sig : UInt64 := ((1 : UInt64) <<< 52) ||| frac
    if e > 16 then
      sign ||| f16PosInf
    else if e >= -14 then
      let shift : UInt64 := 42
      let q0 := sig >>> shift
      let rem := sig &&& (((1 : UInt64) <<< shift) - 1)
      let half : UInt64 := (1 : UInt64) <<< 41
      let roundUp := rem > half || (rem == half && (tiesAway || (q0 &&& 1) == 1))
      let (q, eFin) :=
        if roundUp then
          let q1 := q0 + 1
          if q1 == 0x800 then (0x400, e + 1) else (q1, e)
        else
          (q0, e)
      if eFin > 15 then
        sign ||| f16PosInf
      else
        sign ||| (((eFin + 15).toNat.toUInt16) <<< 10) ||| (q.toUInt16 &&& 0x3ff)
    else
      -- Subnormal / underflow: target LSB is 2^-24, sig LSB is 2^(e - 52) -> shift = 28 - e
      let shift : Nat := (28 - e).toNat
      if shift > 53 then
        sign
      else if shift == 53 then
        if tiesAway || frac > 0 then sign ||| 0x0001 else sign
      else
        let s64 := shift.toUInt64
        let q0 := sig >>> s64
        let rem := sig &&& (((1 : UInt64) <<< s64) - 1)
        let half : UInt64 := (1 : UInt64) <<< (shift - 1).toUInt64
        let roundUp := rem > half || (rem == half && (tiesAway || (q0 &&& 1) == 1))
        let q := if roundUp then q0 + 1 else q0
        if q == 0x400 then
          sign ||| 0x0400
        else
          sign ||| q.toUInt16

/-- Convert a `Float` to IEEE-754 binary16 using round-to-nearest-even (`NaN -> 0x7e00`). -/
def floatToF16 (f : Float) : UInt16 :=
  floatToF16Rounded f false

/-- Convert a `Float` to IEEE-754 binary16 (round-to-nearest-even), canonicalizing `-0.0` to `+0.0`. -/
def floatToF16CanonicalZero (f : Float) : UInt16 :=
  let h := floatToF16 f
  if h == f16NegZero then f16PosZero else h

/-- Round a `Float` to fp16 using hardware `rna16` semantics:
    round-to-nearest ties away from zero, overflow `>= 65520 -> +-inf`,
    `-0.0 -> +0.0`, and `NaN -> +0.0`. -/
def rna16 (f : Float) : UInt16 :=
  if f.isNaN then
    f16PosZero
  else
    let h := floatToF16Rounded f true
    if h == f16NegZero then f16PosZero else h

/-- Next smaller IEEE-754 binary16 bit pattern strictly below `b` (saturating at `-inf`). -/
def prevF16 (b : UInt16) : UInt16 :=
  if (b &&& 0x7fff) == 0 then
    f16NegMinSubnormal
  else if (b &&& 0x8000) == 0 then
    b - 1
  else if b >= f16NegInf then
    f16NegInf
  else
    b + 1

/-- Next larger IEEE-754 binary16 bit pattern strictly above `b` (saturating at `+inf`). -/
def nextF16 (b : UInt16) : UInt16 :=
  if (b &&& 0x7fff) == 0 then
    f16MinSubnormal
  else if (b &&& 0x8000) != 0 then
    b - 1
  else if b >= f16PosInf then
    f16PosInf
  else
    b + 1

/-- Negate an IEEE-754 binary16 value, canonicalizing zero to `+0.0`. -/
def negF16CanonicalZero (b : UInt16) : UInt16 :=
  if (b &&& 0x7fff) == 0 then f16PosZero else b ^^^ 0x8000

/-- Exact `frexp(x)` for positive finite `Float`: returns `(m, e)` with `m in [0.5, 1.0)` and `x = m * 2^e`. -/
def frexpPos (x : Float) : Float × Int :=
  let bits := x.toBits
  let exp : Nat := ((bits >>> 52) &&& 0x7ff).toNat
  let frac : UInt64 := bits &&& (((1 : UInt64) <<< 52) - 1)
  if exp == 0 then
    if frac == 0 then
      (0.0, 0)
    else
      -- Normalize binary64 subnormal
      Id.run do
        let mut f := frac
        let mut e : Int := -1021
        while (f &&& ((1 : UInt64) <<< 52)) == 0 do
          f := f <<< 1
          e := e - 1
        let fNorm := f &&& (((1 : UInt64) <<< 52) - 1)
        let mBits : UInt64 := ((1022 : UInt64) <<< 52) ||| fNorm
        return (Float.ofBits mBits, e)
  else
    let mBits : UInt64 := ((1022 : UInt64) <<< 52) ||| frac
    (Float.ofBits mBits, (exp : Int) - 1022)

/-- Exact scaling `x * 2^e` for `Float` using `2^e` bit construction (handles large positive/negative `e`). -/
def ldexp (x : Float) (e : Int) : Float :=
  if x == 0.0 || x.isNaN || x.isInf then
    x
  else
    -- Step in chunks of <= 1020 to avoid intermediate exponent overflow/underflow
    Id.run do
      let mut acc := x
      let mut rem := e
      while rem > 1020 do
        acc := acc * Float.ofBits ((2043 : UInt64) <<< 52) -- 2^1020
        rem := rem - 1020
      while rem < -1020 do
        acc := acc * Float.ofBits ((3 : UInt64) <<< 52) -- 2^-1020
        rem := rem + 1020
      let scaleBits : UInt64 := ((rem + 1023).toNat.toUInt64) <<< 52
      return acc * Float.ofBits scaleBits

/-- Pure-Lean `erf(x)` to full IEEE-754 float64 precision:
    positive-term series `2/sqrt(pi) * exp(-x^2) * sum 2^n x^(2n+1)/(2n+1)!!` for `|x| < 2`,
    and Laplace continued fraction for `erfc(|x|)` for `|x| >= 2`. -/
def erfApprox (x : Float) : Float :=
  if x.isNaN then
    x
  else if x == 0.0 then
    0.0
  else
    let sign : Float := if x < 0.0 then -1.0 else 1.0
    let ax := Float.abs x
    let y : Float :=
      if ax >= 6.0 then
        1.0
      else if ax < 2.0 then Id.run do
        let twoX2 := 2.0 * ax * ax
        let mut term := ax
        let mut sum := ax
        for n in [1:45] do
          term := term * twoX2 / Float.ofNat (2 * n + 1)
          sum := sum + term
        return 1.12837916709551257389615890312154517 * Float.exp (-(ax * ax)) * sum
      else Id.run do
        let mut cf : Float := 0.0
        for i in [0:60] do
          let k := 60 - i
          cf := (0.5 * Float.ofNat k) / (ax + cf)
        let erfcVal := 0.564189583547756286948079451560772586 * Float.exp (-(ax * ax)) / (ax + cf)
        return 1.0 - erfcVal
    sign * y

/-- Pure-Lean `exp(x) - 1` avoiding catastrophic cancellation near `0`. -/
def expm1 (x : Float) : Float :=
  if Float.abs x < 1e-4 then
    x + 0.5 * x * x + x * x * x / 6.0
  else
    Float.exp x - 1.0

/-- Pure-Lean base-2 exponential `2^x`. -/
def exp2Float (x : Float) : Float :=
  Float.exp (x * 0.693147180559945309417232121458176568)

/-- Pure-Lean base-2 logarithm `log2(x)`. -/
def log2Float (x : Float) : Float :=
  Float.log x / 0.693147180559945309417232121458176568

/-- Parse a decimal floating-point literal (optional sign, integer/fractional digits, optional `e`/`E` exponent). -/
def parseFloat? (raw : String) : Option Float := Id.run do
  let s := raw.trimAscii.toString
  if s.isEmpty then
    return none
  let low := s.toLower
  if low == "inf" || low == "+inf" || low == "infinity" || low == "+infinity" then
    return some (Float.ofBits 0x7ff0000000000000)
  if low == "-inf" || low == "-infinity" then
    return some (Float.ofBits 0xfff0000000000000)
  let chars := s.toList
  let (neg, rest) := match chars with
    | '-' :: tl => (true, tl)
    | '+' :: tl => (false, tl)
    | _ => (false, chars)
  if rest.isEmpty then
    return none

  let mut cur := rest
  let mut intVal : Float := 0.0
  let mut intDigits : Nat := 0
  while !cur.isEmpty do
    match cur with
    | c :: tl =>
      if c >= '0' && c <= '9' then
        intVal := intVal * 10.0 + Float.ofNat (c.toNat - '0'.toNat)
        intDigits := intDigits + 1
        cur := tl
      else
        break
    | [] => break

  let mut fracVal : Float := 0.0
  let mut fracScale : Float := 1.0
  let mut fracDigits : Nat := 0
  match cur with
  | '.' :: tl =>
    cur := tl
    while !cur.isEmpty do
      match cur with
      | c :: tl2 =>
        if c >= '0' && c <= '9' then
          fracVal := fracVal * 10.0 + Float.ofNat (c.toNat - '0'.toNat)
          fracScale := fracScale * 10.0
          fracDigits := fracDigits + 1
          cur := tl2
        else
          break
      | [] => break
  | _ => pure ()

  if intDigits == 0 && fracDigits == 0 then
    return none

  let mut expVal : Int := 0
  match cur with
  | 'e' :: tl | 'E' :: tl =>
    let (expNeg, tl1) := match tl with
      | '-' :: t => (true, t)
      | '+' :: t => (false, t)
      | _ => (false, tl)
    if tl1.isEmpty then
      return none
    let mut eMag : Nat := 0
    let mut eDigits : Nat := 0
    cur := tl1
    while !cur.isEmpty do
      match cur with
      | c :: tl2 =>
        if c >= '0' && c <= '9' then
          if eMag < 10000 then
            eMag := eMag * 10 + (c.toNat - '0'.toNat)
          eDigits := eDigits + 1
          cur := tl2
        else
          return none
      | [] => break
    if eDigits == 0 then
      return none
    expVal := if expNeg then -(eMag : Int) else (eMag : Int)
  | [] => pure ()
  | _ => return none

  if !cur.isEmpty then
    return none

  let mut val := intVal + fracVal / fracScale
  if val == 0.0 then
    return some (if neg then -0.0 else 0.0)
  if expVal > 310 then
    return some (Float.ofBits (if neg then (0xfff0000000000000 : UInt64) else (0x7ff0000000000000 : UInt64)))
  if expVal < -330 then
    return some (if neg then -0.0 else 0.0)
  if expVal != 0 then
    let eAbs := expVal.natAbs
    let mut pow10 : Float := 1.0
    for _ in [:eAbs] do
      pow10 := pow10 * 10.0
    if expVal > 0 then
      val := val * pow10
    else
      val := val / pow10
  if neg then
    val := -val
  return some val

end Vane.Float16
