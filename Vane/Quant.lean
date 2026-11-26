import Vane.Binary
import Vane.Float16
import Vane.TaskStream
import Vane.Register
import Vane.Tiling

namespace Vane.Quant

open Vane.Binary
open Vane.Float16
open Vane.TaskStream
open Vane.Register
open Vane.Tiling

/-- Packed chunk slot metadata in `__KERN_0`. -/
structure KernelChunkSlot where
  slot : Nat
  offset : Nat
  size : Nat
  deriving Repr, BEq, Inhabited

/-- Supported native Neural Engine quantized and palettized weight formats. -/
inductive QuantFormat where
  | int8
  | uint8
  | int4
  | uint4
  | pal (bits : Nat)
  deriving Repr, BEq, DecidableEq, Inhabited

namespace QuantFormat

/-- Return `true` iff `f` is a hardware-supported quantized weight format. -/
def isValid : QuantFormat → Bool
  | .int8 | .uint8 | .int4 | .uint4 => true
  | .pal b => b == 1 || b == 2 || b == 4 || b == 6 || b == 8

/-- Canonical CLI / JSON format name. -/
def name : QuantFormat → String
  | .int8 => "int8"
  | .uint8 => "uint8"
  | .int4 => "int4"
  | .uint4 => "uint4"
  | .pal b => s!"pal{b}"

/-- Parse a CLI / JSON quantized weight format name (`int8`, `uint8`, `int4`, `uint4`, `pal1|2|4|6|8`). -/
def parse? (s : String) : Option QuantFormat :=
  match s.toLower with
  | "int8" => some .int8
  | "uint8" => some .uint8
  | "int4" => some .int4
  | "uint4" => some .uint4
  | "pal1" => some (.pal 1)
  | "pal2" => some (.pal 2)
  | "pal4" => some (.pal 4)
  | "pal6" => some (.pal 6)
  | "pal8" => some (.pal 8)
  | _ => none

/-- Hardware `0x4900[3:0]` (`wKind`) encoding for this format. -/
def wKind : QuantFormat → UInt32
  | .uint8 => wKindUint8
  | .int8 => wKindInt8
  | .uint4 => wKindUint4
  | .int4 => wKindInt4
  | .pal _ => wKindPal

/-- Hardware `0x4900[7:4]` (`wWidth`) stored bit-width per weight code. -/
def wWidth : QuantFormat → Nat
  | .int8 | .uint8 => 8
  | .int4 | .uint4 => 4
  | .pal b => b

/-- Return `true` iff this format is a scaled affine integer format (`int8`/`uint8`/`int4`/`uint4`). -/
def isInteger : QuantFormat → Bool
  | .int8 | .uint8 | .int4 | .uint4 => true
  | .pal _ => false

/-- Return `true` iff this format is unsigned (`uint8`/`uint4`). -/
def isUnsigned : QuantFormat → Bool
  | .uint8 | .uint4 => true
  | _ => false

/-- Return `true` iff this format is a palette lookup format (`pal*`). -/
def isPalette : QuantFormat → Bool
  | .pal _ => true
  | _ => false

/-- 64-byte-padded decode/palette table byte size at offset 0 of each kernel chunk. -/
def tableBytes (f : QuantFormat) : Nat :=
  quantTableBytes f.wKind f.wWidth

end QuantFormat

/-- Static quantized or palettized weight specification for `conv1x1`, `linear`, or dense `conv3x3`. -/
structure QuantWeightSpec where
  format : QuantFormat
  /-- Row-major `[Cout, Cin * taps]` raw code bytes. -/
  codes : ByteArray
  /-- Strictly positive finite scales: empty for `pal`, length `1` (per-tensor) or `Cout` (per-channel) for integers. -/
  scales : Array Float := #[]
  /-- Optional per-output-channel zero points (length `Cout`) for integer formats; `none` for `pal`. -/
  zeroPoints : Option (Array UInt8) := none
  /-- Exact `2^bits` finite IEEE fp16 entries for `pal`, empty otherwise. -/
  palette : Array UInt16 := #[]
  deriving BEq, Inhabited

/-- Lightweight chunk-sizing parameters for quantized tiling and 64 KiB chunk validation. -/
structure QuantChunkParams where
  wKind : UInt32
  wWidth : Nat
  perLaneScale : Bool
  biasInKernel : Bool
  hasZeroPoint : Bool
  deriving Repr, BEq, Inhabited

/-- Result of packing quantized weights, optional scales/bias/zero-points, and decode/palette tables into `__KERN_0`. -/
structure PackedQuantKernel where
  bytes : ByteArray
  slots : Array KernelChunkSlot
  format : QuantFormat
  perLaneScale : Bool
  hasZeroPoint : Bool
  biasInKernel : Bool
  tensorScaleFp16 : UInt16
  scalePexp : Nat
  biasShift : Nat
  deriving BEq

/-- Canonical 16-byte signed 4-bit two's-complement decode table for `int4`. -/
def int4Table : Array UInt8 := int4DecodeTable

/-- Canonical 16-byte unsigned 4-bit decode table for `uint4`. -/
def uint4Table : Array UInt8 := uint4DecodeTable

-- Re-export exact quantized chunk size calculation from `Vane.Register`.
export Vane.Register (expectedQuantChunkSize)

-- ============================================================================
-- Pure LSB-first bit-packer & inverse decoder (kernel-clean for formal proofs)
-- ============================================================================

/-- Pack one output byte `b` of an LSB-first `width`-bit packed code stream.
    Trailing high bits past `codes.size * width` are zero-padded. -/
def packByteLSB (width : Nat) (codes : Array UInt8) (b : Nat) : UInt8 :=
  let bitAt (r : Nat) : Nat :=
    let g := 8 * b + r
    if g < codes.size * width then
      let idx := g / width
      let k := g % width
      if h : idx < codes.size then
        ((codes[idx].toNat >>> k) &&& 1) <<< r
      else 0
    else 0
  (bitAt 0 ||| bitAt 1 ||| bitAt 2 ||| bitAt 3 |||
   bitAt 4 ||| bitAt 5 ||| bitAt 6 ||| bitAt 7).toUInt8

/-- Pure LSB-first bit-packer: packs the low `width` bits of each byte in `codes`
    into `ceil((codes.size * width) / 8)` bytes, zero-padding any trailing high bits. -/
def packBitsLSB (width : Nat) (codes : Array UInt8) : ByteArray :=
  ByteArray.mk (Array.ofFn (n := packedByteCount width codes.size) fun b =>
    packByteLSB width codes b.val)

/-- Decode the `i`-th `width`-bit code from an LSB-first packed byte stream `bytes`. -/
def unpackCodeLSB (width : Nat) (bytes : ByteArray) (i : Nat) : UInt8 :=
  let rec go (k : Nat) (acc : Nat) : Nat :=
    match k with
    | 0 => acc
    | k' + 1 =>
      let g := i * width + k'
      let bIdx := g / 8
      let r := g % 8
      let bit :=
        if h : bIdx < bytes.data.size then
          ((bytes.data[bIdx].toNat >>> r) &&& 1) <<< k'
        else 0
      go k' (acc ||| bit)
  (go width 0).toUInt8

/-- Pure inverse decoder for `packBitsLSB`: recovers `count` codes of bit-width `width`. -/
def unpackBitsLSB (width : Nat) (bytes : ByteArray) (count : Nat) : Array UInt8 :=
  Array.ofFn (n := count) fun i => unpackCodeLSB width bytes i.val

-- ============================================================================
-- Scale exponent (`pexp`) and bias folding shift (`biasShift`)
-- ============================================================================

/-- Choose the smallest `e ∈ [0, 30]` such that `minScale * 2^e >= 2^-16`.
    Returns `0` when `scales` is empty or `minScale >= 2^-16`. -/
def computePexp (scales : Array Float) : Nat := Id.run do
  if scales.isEmpty then
    return 0
  let mut minScale := scales[0]!
  for i in [1 : scales.size] do
    let s := scales[i]!
    if s < minScale then
      minScale := s
  let threshold := ldexp 1.0 (-16)
  for e in [:maxQuantScalePexp + 1] do
    if ldexp minScale (e : Int) >= threshold then
      return e
  return maxQuantScalePexp

/-- Choose the smallest `s ∈ [0, 15]` such that `max_o(|bias[o] / scale[o]|) / 2^s < 32768`.
    Returns `0` when `biases` or `scales` is empty. -/
def computeBiasShift (scales : Array Float) (biases : Array Float) : Nat := Id.run do
  if biases.isEmpty || scales.isEmpty then
    return 0
  let mut maxRatio : Float := 0.0
  for o in [:biases.size] do
    let s := if scales.size == 1 then scales[0]! else scales[o]!
    if s > 0.0 then
      let ratio := Float.abs (biases[o]! / s)
      if ratio > maxRatio then
        maxRatio := ratio
  for shift in [:maxQuantBiasShift + 1] do
    if ldexp maxRatio (-(shift : Int)) < 32768.0 then
      return shift
  return maxQuantBiasShift

-- ============================================================================
-- Validation, slicing, and quantized tiling helpers
-- ============================================================================

/-- Validate a `QuantWeightSpec` against expected `(cout, cin, taps)` geometry. -/
def validateQuantWeightSpec (q : QuantWeightSpec) (cout cin taps : Nat) : Except String Unit := do
  if !q.format.isValid then
    throw s!"unsupported quantized weight format '{q.format.name}'"
  let expectedCodes := cout * cin * taps
  if q.codes.size != expectedCodes then
    throw s!"quantized weight code byte size {q.codes.size} != expected {expectedCodes} (Cout={cout} * Cin={cin} * taps={taps})"
  if q.format.isInteger then
    if !q.palette.isEmpty then
      throw s!"quantized integer format '{q.format.name}' must not specify a palette table"
    if q.scales.size != 1 && q.scales.size != cout then
      throw s!"quantized format '{q.format.name}' scales length {q.scales.size} must be 1 (per-tensor) or Cout={cout} (per-channel)"
    for i in [:q.scales.size] do
      let s := q.scales[i]!
      if s.isNaN || s.isInf || s <= 0.0 then
        throw s!"quantized scale[{i}] = {s} must be finite and strictly positive"
    if let some zps := q.zeroPoints then
      if zps.size != cout then
        throw s!"quantized zero-points length {zps.size} != Cout={cout}"
      match q.format with
      | .int4 =>
        for i in [:zps.size] do
          let z := zps[i]!.toNat
          if z > 7 && z < 248 then
            throw s!"int4 zero-point[{i}] byte {z} outside signed [-8, 7] range (0..7 or 248..255)"
      | .uint4 =>
        for i in [:zps.size] do
          let z := zps[i]!.toNat
          if z > 15 then
            throw s!"uint4 zero-point[{i}] = {z} exceeds range [0, 15]"
      | _ => pure ()
    match q.format with
    | .int4 =>
      for i in [:q.codes.size] do
        let c := q.codes.data[i]!.toNat
        if c > 15 && c < 248 then
          throw s!"int4 code[{i}] byte {c} outside nibble/signed [-8, 7] range"
    | .uint4 =>
      for i in [:q.codes.size] do
        let c := q.codes.data[i]!.toNat
        if c > 15 then
          throw s!"uint4 code[{i}] = {c} exceeds range [0, 15]"
    | _ => pure ()
  else
    -- Palette format
    if !q.scales.isEmpty then
      throw s!"palette format '{q.format.name}' must not specify scales"
    if q.zeroPoints.isSome then
      throw s!"palette format '{q.format.name}' must not specify zero-points"
    let expEntries := (1 : Nat) <<< q.format.wWidth
    if q.palette.size != expEntries then
      throw s!"palette format '{q.format.name}' requires {expEntries} fp16 entries, got {q.palette.size}"
    for i in [:q.palette.size] do
      let h := q.palette[i]!
      if !f16IsFinite h then
        throw s!"palette[{i}] ({toHex h.toNat}) is not a finite IEEE fp16 value"
    for i in [:q.codes.size] do
      let c := q.codes.data[i]!.toNat
      if c >= expEntries then
        throw s!"palette '{q.format.name}' code[{i}] = {c} >= table size {expEntries}"
  pure ()

/-- Compute `QuantChunkParams` for a quantized conv slice with output channel count `coutSlice`. -/
def quantChunkParamsFor (q : QuantWeightSpec) (coutSlice : Nat) (hasBias : Bool) : QuantChunkParams := {
  wKind := q.format.wKind
  wWidth := q.format.wWidth
  perLaneScale := q.format.isInteger && q.scales.size > 1 && coutSlice > 1
  biasInKernel := hasBias
  hasZeroPoint := q.zeroPoints.isSome
}

/-- Slice a `QuantWeightSpec` along output channels `[c0, c0 + dc)` for wide-`Cout` output splitting. -/
def sliceQuantWeightSpec (q : QuantWeightSpec) (c0 dc cin taps : Nat) : Except String QuantWeightSpec := do
  let rowLen := cin * taps
  let codeStart := c0 * rowLen
  let codeEnd := (c0 + dc) * rowLen
  if codeEnd > q.codes.size then
    throw s!"sliceQuantWeightSpec: code range [{codeStart}, {codeEnd}) exceeds codes size {q.codes.size}"
  let slicedCodes := q.codes.extract codeStart codeEnd
  let slicedScales ←
    if q.scales.size > 1 then
      if c0 + dc > q.scales.size then
        throw s!"sliceQuantWeightSpec: scale range [{c0}, {c0 + dc}) exceeds scales size {q.scales.size}"
      pure (q.scales.extract c0 (c0 + dc))
    else
      pure q.scales
  let slicedZps ← match q.zeroPoints with
    | none => pure none
    | some zps =>
      if c0 + dc > zps.size then
        throw s!"sliceQuantWeightSpec: zero-point range [{c0}, {c0 + dc}) exceeds zeroPoints size {zps.size}"
      pure (some (zps.extract c0 (c0 + dc)))
  pure {
    format := q.format
    codes := slicedCodes
    scales := slicedScales
    zeroPoints := slicedZps
    palette := q.palette
  }

/-- Check per-group 64 KiB chunk size cap for a dense quantized convolution. -/
def checkConvQuantChunks64K
    (sh : ConvSpec) (t : Tiling) (qp : QuantChunkParams) : Except String Unit := do
  let groups := (1 : Nat) <<< t.f
  let cinW := sh.cinW
  let taps := sh.taps
  for g in [:groups] do
    let blocks := groupOutputBlocksConv sh t g
    if !blocks.isEmpty then
      let chunkBytes := expectedQuantChunkSize qp.wKind qp.wWidth cinW taps
        qp.perLaneScale qp.biasInKernel qp.hasZeroPoint blocks
      if chunkBytes > maxKernelChunkBytes then
        throw s!"illegal quantized tiling: group {g} chunk {chunkBytes}B exceeds {maxKernelChunkBytes}B (64 KiB)"
  pure ()

/-- Silicon safety validator for a dense quantized NE convolution (`1x1` or stride-1 dense `3x3`). -/
def validateConvSafetyQuant
    (target : Target) (sh : ConvSpec) (t : Tiling) (qp : QuantChunkParams) : Except String Unit :=
  validateConvShape sh >>= fun () =>
  if sh.effMode != .dense || sh.stride != 1 then
    .error s!"quantized weights require dense stride-1 convolution (got mode={sh.mode.name}, stride={sh.stride})"
  else if t.logNE > 4 then
    .error s!"illegal tiling: logNE={t.logNE} exceeds 4"
  else if t.fat > 1 then
    .error s!"illegal tiling: fat={t.fat} must be 0 or 1"
  else if t.f + t.fat > t.logNE then
    .error s!"illegal tiling: F({t.f}) + fat({t.fat}) > logNE({t.logNE})"
  else if t.wu > 3 then
    .error s!"illegal tiling: WU={t.wu} exceeds 2-bit limit 3"
  else
    validateConvOcgSafety target sh t >>= fun () =>
    validateConvCoverageSafety target sh t >>= fun () =>
    let stripW := stripWidth t
    if stripW < 8 || stripW > 256 then
      .error s!"illegal tiling: strip width T={stripW} outside [8, 256]"
    else if t.shPref > t.shMax then
      .error s!"illegal tiling: shifts shPref={t.shPref} > shMax={t.shMax}"
    else if t.shMax > 4 then
      .error s!"illegal tiling: shMax={t.shMax} exceeds maximum 4"
    else if t.tileHeight == 0 then
      .error s!"illegal tiling: tileHeight=0 must be >= 1"
    else if l2UseBytesConv target sh t > l2CapacityBytes then
      .error s!"illegal tiling: L2 footprint {l2UseBytesConv target sh t} bytes exceeds {l2CapacityBytes} bytes (2 MiB)"
    else
      checkConvQuantChunks64K sh t qp

/-- Full tiling legality check for dense quantized NE convolutions. -/
def validateConvTilingQuant
    (target : Target) (sh : ConvSpec) (t : Tiling) (qp : QuantChunkParams) : Except String Unit := do
  validateConvSafetyQuant target sh t qp
  validateConvChooserPolicy sh t

/-- Deterministic tile chooser for dense quantized NE convolutions (`1x1` and stride-1 dense `3x3`),
    checking exact quantized 64 KiB chunk bounds via `expectedQuantChunkSize`. -/
def chooseConvTilingQuant
    (target : Target) (sh : ConvSpec) (qp : QuantChunkParams) : Except String Tiling := do
  validateConvShape sh
  if sh.effMode != .dense || sh.stride != 1 then
    throw s!"quantized weights require dense stride-1 convolution (got mode={sh.mode.name}, stride={sh.stride})"

  let wo := sh.outW
  let maxF := min 4 (ceilLog2 sh.cout)
  let fCandidates : Array Nat := Id.run do
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
          if let some ocg := chooseConvOcg? target sh f then
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
                    let maxTh := maxTileHeightConv target sh probe
                    if maxTh >= 1 then
                      let cand : Tiling := { probe with tileHeight := maxTh }
                      match validateConvTilingQuant target sh cand qp with
                      | .ok () => return cand
                      | .error _ => pure ()

  throw s!"no legal {target.name} quantized tiling found for {sh.kernelSize}x{sh.kernelSize} dense conv (H={sh.h}, W={sh.w}, Cin={sh.cin}, Cout={sh.cout})"

-- ============================================================================
-- Pure Quantized Chunk & Kernel Packer (`__KERN_0`)
-- ============================================================================

/-- Encode the optional 64-byte-padded decode or palette table emitted once at chunk offset 0. -/
def encodeQuantTable (format : QuantFormat) (palette : Array UInt16) : ByteArray :=
  match format with
  | .int8 | .uint8 => ByteArray.empty
  | .int4 => padToAlign (ByteArray.mk int4Table) 64
  | .uint4 => padToAlign (ByteArray.mk uint4Table) 64
  | .pal bits =>
    let rawBytes := 2 * ((1 : Nat) <<< bits)
    let raw := ByteArray.mk (Array.ofFn (n := rawBytes) fun idx =>
      let entry := palette.getD (idx.val / 2) 0
      if idx.val % 2 == 0 then
        (entry &&& 0xff).toUInt8
      else
        ((entry >>> 8) &&& 0xff).toUInt8)
    padToAlign raw 64

/-- Encode one output-channel block `(o0, n)` inside a quantized kernel chunk:
    `[scale row][folded bias row][zp row][LSB-packed codes (c, tap, lane) lane-fastest]`. -/
def encodeQuantBlock
    (wWidth : Nat)
    (codes : ByteArray)
    (cout cin taps : Nat)
    (storedScales : Array UInt16)
    (foldedBiases : Array UInt16)
    (zeroPoints : Option (Array UInt8))
    (perLaneScale biasInKernel hasZeroPoint : Bool)
    (o0 n : Nat) : ByteArray :=
  let scaleBytes : ByteArray :=
    if perLaneScale then
      ByteArray.mk (Array.ofFn (n := 2 * n) fun idx =>
        let k := idx.val / 2
        let o := o0 + k
        let h : UInt16 := if o < cout then storedScales.getD o f16PosOne else 0
        if idx.val % 2 == 0 then (h &&& 0xff).toUInt8 else ((h >>> 8) &&& 0xff).toUInt8)
    else
      ByteArray.empty
  let biasBytes : ByteArray :=
    if biasInKernel then
      ByteArray.mk (Array.ofFn (n := 2 * n) fun idx =>
        let k := idx.val / 2
        let o := o0 + k
        let h : UInt16 := if o < cout then foldedBiases.getD o 0 else 0
        if idx.val % 2 == 0 then (h &&& 0xff).toUInt8 else ((h >>> 8) &&& 0xff).toUInt8)
    else
      ByteArray.empty
  let zpBytes : ByteArray :=
    if hasZeroPoint then
      ByteArray.mk (Array.ofFn (n := n) fun idx =>
        let o := o0 + idx.val
        if o < cout then
          match zeroPoints with
          | some zps => zps.getD o 0
          | none => 0
        else 0)
    else
      ByteArray.empty
  let codeArr : Array UInt8 :=
    Array.ofFn (n := cin * taps * n) fun idx =>
      let k := idx.val % n
      let rem := idx.val / n
      let tp := rem % taps
      let ci := rem / taps
      let o := o0 + k
      if o < cout then
        codes.data.getD ((o * cin + ci) * taps + tp) 0
      else
        0
  let packedCodes := packBitsLSB wWidth codeArr
  scaleBytes ++ biasBytes ++ zpBytes ++ packedCodes

/-- Build a single 64-byte-aligned quantized kernel chunk for one active NE group's blocks. -/
def packQuantChunk
    (format : QuantFormat)
    (palette : Array UInt16)
    (codes : ByteArray)
    (cout cin taps : Nat)
    (storedScales : Array UInt16)
    (foldedBiases : Array UInt16)
    (zeroPoints : Option (Array UInt8))
    (perLaneScale biasInKernel hasZeroPoint : Bool)
    (blocks : Array (Nat × Nat)) : ByteArray :=
  let raw := blocks.foldl
    (fun acc (o0, n) =>
      acc ++ encodeQuantBlock format.wWidth codes cout cin taps
        storedScales foldedBiases zeroPoints perLaneScale biasInKernel hasZeroPoint o0 n)
    (encodeQuantTable format palette)
  padToAlign raw 64

/-- Pack quantized/palettized weights and optional fp16 bias into `__KERN_0` chunk slots
    for dense `conv1x1`, `linear`, or stride-1 dense `conv3x3`. -/
def packQuantKernel
    (target : Target)
    (sh : ConvSpec)
    (tiling : Tiling)
    (q : QuantWeightSpec)
    (bias : Option ByteArray) : Except String PackedQuantKernel := do
  validateConvShape sh
  if sh.effMode != .dense || sh.stride != 1 then
    throw s!"packQuantKernel: quantized weights require dense stride-1 convolution (got mode={sh.mode.name}, stride={sh.stride})"
  let taps := sh.taps
  validateQuantWeightSpec q sh.cout sh.cin taps

  if let some b := bias then
    let expectedBiasBytes := 2 * sh.cout
    if b.size != expectedBiasBytes then
      throw s!"bias byte size {b.size} != expected {expectedBiasBytes} (2 * Cout={sh.cout})"

  let perLaneScale := q.format.isInteger && q.scales.size > 1 && sh.cout > 1
  let hasZeroPoint := q.zeroPoints.isSome
  let biasInKernel := bias.isSome
  let qp : QuantChunkParams := {
    wKind := q.format.wKind
    wWidth := q.format.wWidth
    perLaneScale := perLaneScale
    biasInKernel := biasInKernel
    hasZeroPoint := hasZeroPoint
  }
  validateConvSafetyQuant target sh tiling qp

  -- Compute scale exponent `pexp`, stored fp16 scales, and effective float64 scales
  let (scalePexp, tensorScaleFp16, storedScales, effScales) ←
    if q.format.isInteger then
      let pexp := computePexp q.scales
      let mut stored : Array UInt16 := Array.mkEmpty sh.cout
      let mut eff : Array Float := Array.mkEmpty sh.cout
      for o in [:sh.cout] do
        let rawS := if q.scales.size == 1 then q.scales[0]! else q.scales[o]!
        let h := floatToF16CanonicalZero (ldexp rawS (pexp : Int))
        if !f16IsFinite h || h == 0 || (h &&& 0x8000) != 0 then
          throw s!"quantized scale[{o}]={rawS} with pexp={pexp} does not represent a finite positive fp16 ({toHex h.toNat})"
        stored := stored.push h
        eff := eff.push (ldexp (f16ToFloat h) (-(pexp : Int)))
      let tScale : UInt16 := if perLaneScale then 0 else stored[0]!
      pure (pexp, tScale, stored, eff)
    else
      pure (0, f16PosOne, #[], #[])

  -- Compute bias shift & folded kernel biases (unified `biasInKernel = 1` whenever bias is present)
  let (biasShift, foldedBiases) ← match bias with
    | none => pure (0, #[])
    | some bBytes =>
      let mut bFloats : Array Float := Array.mkEmpty sh.cout
      let mut rawH : Array UInt16 := Array.mkEmpty sh.cout
      for o in [:sh.cout] do
        let h ← readU16LE bBytes (2 * o)
        if !f16IsFinite h then
          throw s!"bias[{o}] ({toHex h.toNat}) is not a finite fp16 value"
        rawH := rawH.push h
        bFloats := bFloats.push (f16ToFloat h)
      if q.format.isInteger then
        let bShift := computeBiasShift effScales bFloats
        let mut folded : Array UInt16 := Array.mkEmpty sh.cout
        for o in [:sh.cout] do
          let sEff := effScales[o]!
          let ratio := bFloats[o]! / sEff
          let h := floatToF16CanonicalZero (ldexp ratio (-(bShift : Int)))
          if !f16IsFinite h then
            throw s!"folded bias[{o}] (bias={bFloats[o]!}, scale={sEff}, shift={bShift}) overflows finite fp16"
          folded := folded.push h
        pure (bShift, folded)
      else
        pure (0, rawH)

  let groups := (1 : Nat) <<< tiling.f
  let rep := ((1 : Nat) <<< tiling.logNE) >>> tiling.f
  if rep == 0 || groups * rep > 16 then
    throw s!"invalid chunk replication geometry: groups={groups}, rep={rep}"

  let mut slotChunks : Array (Nat × ByteArray) := Array.mkEmpty 16
  let mut totalKernelBytes : Nat := 0

  for g in [:groups] do
    let blocks := groupOutputBlocksConv sh tiling g
    if !blocks.isEmpty then
      let chunk := packQuantChunk q.format q.palette q.codes sh.cout sh.cin taps
        storedScales foldedBiases q.zeroPoints perLaneScale biasInKernel hasZeroPoint blocks
      if chunk.size > maxKernelChunkBytes then
        throw s!"quantized {sh.kernelSize}x{sh.kernelSize} group {g} chunk size {chunk.size}B exceeds maximum {maxKernelChunkBytes}B (64 KiB)"
      if target.profile.replicateKernelChunkSlots then
        for j in [:rep] do
          let s := g * rep + j
          if s >= 16 then
            throw s!"chunk slot index {s} exceeds 0..15"
          slotChunks := slotChunks.push (s, chunk)
          totalKernelBytes := totalKernelBytes + chunk.size
      else
        let s := g * rep
        if s >= 16 then
          throw s!"chunk slot index {s} exceeds 0..15"
        slotChunks := slotChunks.push (s, chunk)
        totalKernelBytes := totalKernelBytes + chunk.size

  if slotChunks.isEmpty || totalKernelBytes == 0 then
    throw "quantized kernel packer produced 0 active chunk slots"

  let mut kernelBuf := ByteArray.emptyWithCapacity totalKernelBytes
  let mut slots : Array KernelChunkSlot := Array.mkEmpty slotChunks.size
  for (s, chunk) in slotChunks do
    let off := kernelBuf.size
    kernelBuf := kernelBuf ++ chunk
    slots := slots.push { slot := s, offset := off, size := chunk.size }

  pure {
    bytes := kernelBuf
    slots := slots
    format := q.format
    perLaneScale := perLaneScale
    hasZeroPoint := hasZeroPoint
    biasInKernel := biasInKernel
    tensorScaleFp16 := tensorScaleFp16
    scalePexp := scalePexp
    biasShift := biasShift
  }

end Vane.Quant
