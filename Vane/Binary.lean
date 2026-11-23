namespace Vane.Binary

/-- Maximum representable 64-bit unsigned integer as a `Nat`. -/
def maxU64Nat : Nat := 0xffffffffffffffff

/-- Maximum representable 32-bit unsigned integer as a `Nat`. -/
def maxU32Nat : Nat := 0xffffffff

/-- Round `n` up to a multiple of `align` (when `align > 0`). -/
def alignUp (n align : Nat) : Nat :=
  if align == 0 then n
  else
    let rem := n % align
    if rem == 0 then n else n + (align - rem)

/-- Round a `UInt64` up to a multiple of `align` (as `Nat`, checking overflow). -/
def alignUpU64 (n align : UInt64) : Except String UInt64 := do
  let res := alignUp n.toNat align.toNat
  if res > maxU64Nat then
    throw s!"alignment overflow: alignUp({n}, {align})"
  pure res.toUInt64

/-- Safe addition of two `UInt64` values with overflow check. -/
def addU64 (a b : UInt64) (ctx : String) : Except String UInt64 := do
  let s := a.toNat + b.toNat
  if s > maxU64Nat then
    throw s!"u64 addition overflow in {ctx}: {a} + {b}"
  pure s.toUInt64

/-- Safe multiplication of two `UInt64` values with overflow check. -/
def mulU64 (a b : UInt64) (ctx : String) : Except String UInt64 := do
  let p := a.toNat * b.toNat
  if p > maxU64Nat then
    throw s!"u64 multiplication overflow in {ctx}: {a} * {b}"
  pure p.toUInt64

/-- Create a zero-filled `ByteArray` of length `n`. -/
def zeroBytes (n : Nat) : ByteArray :=
  ByteArray.mk (Array.replicate n 0)

/-- Append a single byte. -/
def pushU8 (bs : ByteArray) (v : UInt8) : ByteArray :=
  bs.push v

/-- Append a little-endian 16-bit unsigned integer. -/
def pushU16LE (bs : ByteArray) (v : UInt16) : ByteArray :=
  (bs.push (v &&& 0xff).toUInt8).push ((v >>> 8) &&& 0xff).toUInt8

/-- Append a little-endian 32-bit unsigned integer. -/
def pushU32LE (bs : ByteArray) (v : UInt32) : ByteArray :=
  let b0 := (v &&& 0xff).toUInt8
  let b1 := ((v >>> 8) &&& 0xff).toUInt8
  let b2 := ((v >>> 16) &&& 0xff).toUInt8
  let b3 := ((v >>> 24) &&& 0xff).toUInt8
  (((bs.push b0).push b1).push b2).push b3

/-- Append a little-endian 64-bit unsigned integer. -/
def pushU64LE (bs : ByteArray) (v : UInt64) : ByteArray :=
  let lo := (v &&& 0xffffffff).toUInt32
  let hi := ((v >>> 32) &&& 0xffffffff).toUInt32
  pushU32LE (pushU32LE bs lo) hi

/-- Append `count` zero bytes. -/
def pushZeros (bs : ByteArray) (count : Nat) : ByteArray :=
  if count == 0 then bs else bs ++ zeroBytes count

/-- Pad `bs` with zero bytes to a multiple of `align`. -/
def padToAlign (bs : ByteArray) (align : Nat) : ByteArray :=
  let target := alignUp bs.size align
  pushZeros bs (target - bs.size)

/-- Format a `Nat` as lowercase hex with `0x` prefix. -/
def toHex (n : Nat) : String :=
  "0x" ++ String.ofList (Nat.toDigits 16 n)

/-- Validate that a string contains only printable/non-NUL ASCII bytes (0x01..0x7f). -/
def validateAsciiNonNul (s : String) (label : String) : Except String ByteArray := do
  let bytes := s.toUTF8
  for i in [:bytes.size] do
    let b := bytes.get! i
    if b == 0 || b >= 0x80 then
      throw s!"{label}: invalid byte {toHex b.toNat} at index {i}"
  pure bytes

/-- Encode a NUL-terminated, zero-padded fixed-width ASCII string into `width` bytes. -/
def encodeFixedAscii (s : String) (width : Nat) (label : String) : Except String ByteArray := do
  if width == 0 then
    throw s!"{label}: fixed width must be > 0"
  let bytes ← validateAsciiNonNul s label
  if bytes.size >= width then
    throw s!"{label}: string '{s}' ({bytes.size} bytes) exceeds fixed NUL-terminated slot width {width}"
  pure (pushZeros bytes (width - bytes.size))

/-- Write a single byte at `off` into `bs`. -/
def putU8 (bs : ByteArray) (off : Nat) (v : UInt8) : Except String ByteArray := do
  if off >= bs.size then
    throw s!"putU8 out of bounds: offset {off} >= size {bs.size}"
  pure (bs.set! off v)

/-- Write a little-endian `UInt16` at `off` into `bs`. -/
def putU16LE (bs : ByteArray) (off : Nat) (v : UInt16) : Except String ByteArray := do
  if off + 2 > bs.size then
    throw s!"putU16LE out of bounds: offset {off}, size {bs.size}"
  let bs := bs.set! off (v &&& 0xff).toUInt8
  let bs := bs.set! (off + 1) ((v >>> 8) &&& 0xff).toUInt8
  pure bs

/-- Write a little-endian `UInt32` at `off` into `bs`. -/
def putU32LE (bs : ByteArray) (off : Nat) (v : UInt32) : Except String ByteArray := do
  if off + 4 > bs.size then
    throw s!"putU32LE out of bounds: offset {off}, size {bs.size}"
  let bs := bs.set! off (v &&& 0xff).toUInt8
  let bs := bs.set! (off + 1) ((v >>> 8) &&& 0xff).toUInt8
  let bs := bs.set! (off + 2) ((v >>> 16) &&& 0xff).toUInt8
  let bs := bs.set! (off + 3) ((v >>> 24) &&& 0xff).toUInt8
  pure bs

/-- Write a little-endian `UInt64` at `off` into `bs`. -/
def putU64LE (bs : ByteArray) (off : Nat) (v : UInt64) : Except String ByteArray := do
  if off + 8 > bs.size then
    throw s!"putU64LE out of bounds: offset {off}, size {bs.size}"
  let lo := (v &&& 0xffffffff).toUInt32
  let hi := ((v >>> 32) &&& 0xffffffff).toUInt32
  let bs ← putU32LE bs off lo
  putU32LE bs (off + 4) hi

/-- Write a `ByteArray` slice `src` at `off` into `bs`. -/
def putBytes (bs : ByteArray) (off : Nat) (src : ByteArray) : Except String ByteArray := do
  if off + src.size > bs.size then
    throw s!"putBytes out of bounds: offset {off} + len {src.size} > size {bs.size}"
  let mut out := bs
  for i in [:src.size] do
    out := out.set! (off + i) (src.get! i)
  pure out

/-- Read a single byte at `off`. -/
def readU8 (bs : ByteArray) (off : Nat) : Except String UInt8 := do
  if off >= bs.size then
    throw s!"readU8 out of bounds: offset {off} >= size {bs.size}"
  pure (bs.get! off)

/-- Read a little-endian `UInt16` at `off`. -/
def readU16LE (bs : ByteArray) (off : Nat) : Except String UInt16 := do
  if off + 2 > bs.size then
    throw s!"readU16LE out of bounds: offset {off}, size {bs.size}"
  let b0 := (bs.get! off).toUInt16
  let b1 := (bs.get! (off + 1)).toUInt16
  pure (b0 ||| (b1 <<< 8))

/-- Read a little-endian `UInt32` at `off`. -/
def readU32LE (bs : ByteArray) (off : Nat) : Except String UInt32 := do
  if off + 4 > bs.size then
    throw s!"readU32LE out of bounds: offset {off}, size {bs.size}"
  let b0 := (bs.get! off).toUInt32
  let b1 := (bs.get! (off + 1)).toUInt32
  let b2 := (bs.get! (off + 2)).toUInt32
  let b3 := (bs.get! (off + 3)).toUInt32
  pure (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24))

/-- Read a little-endian `UInt64` at `off`. -/
def readU64LE (bs : ByteArray) (off : Nat) : Except String UInt64 := do
  if off + 8 > bs.size then
    throw s!"readU64LE out of bounds: offset {off}, size {bs.size}"
  let lo ← readU32LE bs off
  let hi ← readU32LE bs (off + 4)
  pure (lo.toUInt64 ||| (hi.toUInt64 <<< 32))

/-- Extract a contiguous slice `[off, off + len)` from `bs`. -/
def readSlice (bs : ByteArray) (off len : Nat) : Except String ByteArray := do
  if off + len > bs.size then
    throw s!"readSlice out of bounds: [{off}, {off + len}) exceeds size {bs.size}"
  pure (bs.extract off (off + len))

/-- Verify that every byte in `[off, off + len)` is zero. -/
def checkAllZero (bs : ByteArray) (off len : Nat) (label : String) : Except String Unit := do
  if off + len > bs.size then
    throw s!"{label}: range [{off}, {off + len}) exceeds size {bs.size}"
  for i in [:len] do
    let b := bs.get! (off + i)
    if b != 0 then
      throw s!"{label}: non-zero reserved byte {toHex b.toNat} at offset {off + i}"

/-- Compare two `ByteArray` values byte-for-byte with an offset diagnostic on mismatch. -/
def checkBytesEqual (actual expected : ByteArray) (label : String) : Except String Unit := do
  if actual.size != expected.size then
    throw s!"{label}: length mismatch (actual {actual.size}, expected {expected.size})"
  for i in [:actual.size] do
    let a := actual.get! i
    let e := expected.get! i
    if a != e then
      throw s!"{label}: byte mismatch at +{toHex i} (actual {toHex a.toNat}, expected {toHex e.toNat})"

/-- Decode an ASCII byte slice into a `String` (checking all bytes are in `0x01..0x7f`). -/
def decodeAsciiBytes (bs : ByteArray) (label : String) : Except String String := do
  for i in [:bs.size] do
    let b := bs.get! i
    if b == 0 || b >= 0x80 then
      throw s!"{label}: invalid ASCII byte {toHex b.toNat} at index {i}"
  match String.fromUTF8? bs with
  | some s => pure s
  | none => throw s!"{label}: invalid UTF-8"

/-- Read a fixed-width NUL-terminated, zero-padded ASCII string from `[off, off + width)`. -/
def readFixedAscii (bs : ByteArray) (off width : Nat) (label : String) : Except String String := do
  if off + width > bs.size then
    throw s!"{label}: fixed string [{off}, {off + width}) exceeds size {bs.size}"
  let mut nulPos : Option Nat := none
  for i in [:width] do
    let b := bs.get! (off + i)
    match nulPos with
    | none =>
      if b == 0 then
        nulPos := some i
      else if b >= 0x80 then
        throw s!"{label}: non-ASCII byte {toHex b.toNat} at offset {off + i}"
    | some _ =>
      if b != 0 then
        throw s!"{label}: non-zero padding byte {toHex b.toNat} after NUL at offset {off + i}"
  match nulPos with
  | none => throw s!"{label}: fixed string at offset {off} is missing NUL terminator"
  | some len =>
    let slice := bs.extract off (off + len)
    decodeAsciiBytes slice label

/-- Read a NUL-terminated ASCII C-string starting at `off` within `[off, limit)`.
    Returns `(string, byteLenWithoutNul)`. -/
def readCString (bs : ByteArray) (off limit : Nat) (label : String) : Except String (String × Nat) := do
  if limit > bs.size || off > limit then
    throw s!"{label}: invalid cstring bounds [{off}, {limit}) for size {bs.size}"
  let mut found : Option Nat := none
  let mut i := off
  while i < limit && found.isNone do
    let b := bs.get! i
    if b == 0 then
      found := some (i - off)
    else if b >= 0x80 then
      throw s!"{label}: non-ASCII byte {toHex b.toNat} at offset {i}"
    i := i + 1
  match found with
  | none => throw s!"{label}: unterminated string starting at offset {off}"
  | some len =>
    let slice := bs.extract off (off + len)
    let s ← decodeAsciiBytes slice label
    pure (s, len)

/-- A named half-open interval `[start, start + size)` in `Nat` coordinates. -/
structure ByteRange where
  label : String
  start : Nat
  size : Nat
  deriving Repr, BEq, Inhabited

/-- Check that all ranges are non-empty, fit within `maxEnd` (if given), and are pairwise disjoint. -/
def checkRangesDisjoint (ranges : Array ByteRange) (kind : String)
    (minStart : Nat := 0) (maxEnd : Option Nat := none) : Except String Unit := do
  for r in ranges do
    if r.size == 0 then
      throw s!"{kind}: range '{r.label}' is empty"
    if r.start < minStart then
      throw s!"{kind}: range '{r.label}' starts at {r.start} < minimum {minStart}"
    let endPos := r.start + r.size
    if endPos > maxU64Nat + 1 then
      throw s!"{kind}: range '{r.label}' overflows 64-bit address space"
    if let some cap := maxEnd then
      if endPos > cap then
        throw s!"{kind}: range '{r.label}' [{r.start}, {endPos}) exceeds limit {cap}"
  for i in [:ranges.size] do
    let a := ranges[i]!
    let aEnd := a.start + a.size
    for j in [i + 1 : ranges.size] do
      let b := ranges[j]!
      let bEnd := b.start + b.size
      if a.start < bEnd && b.start < aEnd then
        throw s!"{kind}: overlapping ranges '{a.label}' [{a.start}, {aEnd}) and '{b.label}' [{b.start}, {bEnd})"

end Vane.Binary
