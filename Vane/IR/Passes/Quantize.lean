import Vane.IR.Rewrite
import Vane.IR.Passes.DCE

namespace Vane.IR

open Vane.Binary
open Vane.Float16
open Vane.Quant
open Vane.Conv

-- ============================================================================
-- Pass 7: Quantization & PTQ Pass (`--quantize <int8|int4|pal4|pal2>` + QDQ)
-- ============================================================================

/-- Hardware-accurate dequantizer: expands a `QuantWeightSpec` of shape `[cout, inner]`
    into raw little-endian fp16 bytes (matching M8A hardware semantics). -/
def dequantizeSpecToFp16 (q : QuantWeightSpec) (cout inner : Nat) : Except String ByteArray := do
  if cout == 0 || inner == 0 then
    throw s!"dequantize: invalid dimensions cout={cout}, inner={inner}"
  validateQuantWeightSpec q cout inner 1
  let total := cout * inner
  let mut out := ByteArray.emptyWithCapacity (2 * total)
  if q.format.isPalette then
    for i in [:total] do
      let idx := q.codes.data[i]!.toNat
      let h := q.palette.getD idx 0
      out := pushU16LE out h
  else
    let pexp := computePexp q.scales
    let mut effScales : Array Float := Array.mkEmpty cout
    for o in [:cout] do
      let rawS := if q.scales.size == 1 then q.scales[0]! else q.scales[o]!
      let h := floatToF16CanonicalZero (ldexp rawS (pexp : Int))
      if !f16IsFinite h || h == 0 || (h &&& 0x8000) != 0 then
        throw s!"dequantize: scale[{o}]={rawS} with pexp={pexp} is not finite positive fp16"
      effScales := effScales.push (ldexp (f16ToFloat h) (-(pexp : Int)))
    for o in [:cout] do
      let s := effScales[o]!
      let zpInt : Int := match q.zeroPoints with
        | none => 0
        | some zps =>
          let z := zps[o]!.toNat
          if q.format.isUnsigned then (z : Int)
          else if z >= 128 then (z : Int) - 256 else (z : Int)
      let rowBase := o * inner
      for j in [:inner] do
        let rawByte := q.codes.data[rowBase + j]!.toNat
        let codeInt : Int :=
          match q.format with
          | .int8 => if rawByte >= 128 then (rawByte : Int) - 256 else (rawByte : Int)
          | .uint8 => (rawByte : Int)
          | .int4 =>
            let nib := rawByte &&& 0x0f
            if nib >= 8 then (nib : Int) - 16 else (nib : Int)
          | .uint4 => ((rawByte &&& 0x0f : Nat) : Int)
          | .pal _ => 0
        let diff := codeInt - zpInt
        let fVal := (if diff < 0 then -(Float.ofNat (-diff).toNat) else Float.ofNat diff.toNat) * s
        let h := floatToF16CanonicalZero fVal
        if !f16IsFinite h then
          throw s!"dequantize: value {fVal} at ({o},{j}) overflows finite fp16"
        out := pushU16LE out h
  pure out

/-- Round a finite `Float` to the nearest `Int` (ties away from zero), clamped to `[lo, hi]`. -/
private def roundClampInt (x : Float) (lo hi : Int) : Int :=
  let rounded : Int :=
    if x >= 0.0 then
      let n := (x + 0.5).toUInt64.toNat
      (n : Int)
    else
      let n := ((-x) + 0.5).toUInt64.toNat
      (0 : Int) - (n : Int)
  if rounded < lo then lo else if rounded > hi then hi else rounded

/-- Per-channel symmetric signed `int8` post-training quantizer (`[-127, 127]`, zero-point free). -/
def quantizeWeightsInt8 (vals : Array Float) (cout inner : Nat) : Except String QuantWeightSpec := do
  if cout == 0 || inner == 0 then
    throw s!"quantize int8: invalid dims cout={cout}, inner={inner}"
  if vals.size != cout * inner then
    throw s!"quantize int8: weights size {vals.size} != {cout * inner}"
  let mut scales : Array Float := Array.mkEmpty cout
  let mut codesArr : Array UInt8 := Array.mkEmpty (cout * inner)
  for o in [:cout] do
    let rowBase := o * inner
    let mut maxAbs : Float := 0.0
    for j in [:inner] do
      let w := vals[rowBase + j]!
      if w.isNaN || w.isInf then
        throw s!"quantize int8: non-finite weight at ({o},{j})"
      let a := Float.abs w
      if a > maxAbs then maxAbs := a
    let scale :=
      if maxAbs == 0.0 then 1.0
      else
        let s := maxAbs / 127.0
        if s <= 0.0 || s.isNaN || s.isInf then 1.0 else s
    scales := scales.push scale
    for j in [:inner] do
      let q := roundClampInt (vals[rowBase + j]! / scale) (-127) 127
      let b : UInt8 := (if q < 0 then (256 + q).toNat else q.toNat).toUInt8
      codesArr := codesArr.push b
  let spec : QuantWeightSpec := {
    format := .int8
    codes := ByteArray.mk codesArr
    scales := scales
    zeroPoints := none
    palette := #[]
  }
  validateQuantWeightSpec spec cout inner 1
  pure spec

/-- Per-channel symmetric signed `int4` post-training quantizer (`[-7, 7]`, low nibble `0..15`). -/
def quantizeWeightsInt4 (vals : Array Float) (cout inner : Nat) : Except String QuantWeightSpec := do
  if cout == 0 || inner == 0 then
    throw s!"quantize int4: invalid dims cout={cout}, inner={inner}"
  if vals.size != cout * inner then
    throw s!"quantize int4: weights size {vals.size} != {cout * inner}"
  let mut scales : Array Float := Array.mkEmpty cout
  let mut codesArr : Array UInt8 := Array.mkEmpty (cout * inner)
  for o in [:cout] do
    let rowBase := o * inner
    let mut maxAbs : Float := 0.0
    for j in [:inner] do
      let w := vals[rowBase + j]!
      if w.isNaN || w.isInf then
        throw s!"quantize int4: non-finite weight at ({o},{j})"
      let a := Float.abs w
      if a > maxAbs then maxAbs := a
    let scale :=
      if maxAbs == 0.0 then 1.0
      else
        let s := maxAbs / 7.0
        if s <= 0.0 || s.isNaN || s.isInf then 1.0 else s
    scales := scales.push scale
    for j in [:inner] do
      let q := roundClampInt (vals[rowBase + j]! / scale) (-7) 7
      let nib : UInt8 := ((if q < 0 then (16 + q).toNat else q.toNat) &&& 0x0f).toUInt8
      codesArr := codesArr.push nib
  let spec : QuantWeightSpec := {
    format := .int4
    codes := ByteArray.mk codesArr
    scales := scales
    zeroPoints := none
    palette := #[]
  }
  validateQuantWeightSpec spec cout inner 1
  pure spec

/-- Deterministic 1D k-means (Lloyd-Max) palette post-training quantizer for `pal2`/`pal4` (any `bits ∈ {1,2,4,6,8}`). -/
def quantizeWeightsPalette (bits : Nat) (vals : Array Float) (cout inner : Nat) :
    Except String QuantWeightSpec := do
  let fmt := QuantFormat.pal bits
  if !fmt.isValid then
    throw s!"quantize pal{bits}: unsupported palette bit width {bits}"
  if cout == 0 || inner == 0 then
    throw s!"quantize pal{bits}: invalid dims cout={cout}, inner={inner}"
  if vals.size != cout * inner then
    throw s!"quantize pal{bits}: weights size {vals.size} != {cout * inner}"
  let mut wMin := vals[0]!
  let mut wMax := vals[0]!
  for i in [:vals.size] do
    let w := vals[i]!
    if w.isNaN || w.isInf then
      throw s!"quantize pal{bits}: non-finite weight at index {i}"
    if w < wMin then wMin := w
    if w > wMax then wMax := w

  let k := (1 : Nat) <<< bits
  let mut centroids : Array Float := Array.mkEmpty k
  if k == 1 || wMax == wMin then
    centroids := Array.replicate k wMin
  else
    let denom := Float.ofNat (k - 1)
    for c in [:k] do
      centroids := centroids.push (wMin + (wMax - wMin) * (Float.ofNat c / denom))

  -- Fixed 12-iteration Lloyd-Max refinement (structurally total)
  let mut assign : Array Nat := Array.replicate vals.size 0
  for _ in [:12] do
    let mut sums : Array Float := Array.replicate k 0.0
    let mut cnts : Array Nat := Array.replicate k 0
    for i in [:vals.size] do
      let w := vals[i]!
      let mut bestC : Nat := 0
      let mut bestDist : Float := Float.abs (w - centroids[0]!)
      for c in [1:k] do
        let d := Float.abs (w - centroids[c]!)
        if d < bestDist then
          bestDist := d
          bestC := c
      assign := assign.set! i bestC
      sums := sums.set! bestC (sums[bestC]! + w)
      cnts := cnts.set! bestC (cnts[bestC]! + 1)
    for c in [:k] do
      if cnts[c]! > 0 then
        centroids := centroids.set! c (sums[c]! / Float.ofNat cnts[c]!)

  -- Convert final centroids to finite fp16 entries and perform final nearest-fp16 assignment
  let mut palette : Array UInt16 := Array.mkEmpty k
  let mut palF : Array Float := Array.mkEmpty k
  for c in [:k] do
    let h := floatToF16CanonicalZero centroids[c]!
    if !f16IsFinite h then
      throw s!"quantize pal{bits}: centroid[{c}]={centroids[c]!} overflows finite fp16"
    palette := palette.push h
    palF := palF.push (f16ToFloat h)

  let mut codesArr : Array UInt8 := Array.mkEmpty vals.size
  for i in [:vals.size] do
    let w := vals[i]!
    let mut bestC : Nat := 0
    let mut bestDist : Float := Float.abs (w - palF[0]!)
    for c in [1:k] do
      let d := Float.abs (w - palF[c]!)
      if d < bestDist then
        bestDist := d
        bestC := c
    codesArr := codesArr.push bestC.toUInt8

  let spec : QuantWeightSpec := {
    format := fmt
    codes := ByteArray.mk codesArr
    scales := #[]
    zeroPoints := none
    palette := palette
  }
  validateQuantWeightSpec spec cout inner 1
  pure spec

/-- Parse the `--quantize` CLI flag (`int8`, `int4`, `pal4`, `pal2`). -/
def parsePTQFormat? (s : String) : Option QuantFormat :=
  match s.toLower with
  | "int8" => some .int8
  | "int4" => some .int4
  | "pal4" => some (.pal 4)
  | "pal2" => some (.pal 2)
  | _ => none

/-- Dispatch PTQ weight quantization for supported formats (`int8`, `int4`, `pal4`, `pal2`, etc.). -/
def quantizeWeights (fmt : QuantFormat) (vals : Array Float) (cout inner : Nat) :
    Except String QuantWeightSpec :=
  match fmt with
  | .int8 => quantizeWeightsInt8 vals cout inner
  | .int4 => quantizeWeightsInt4 vals cout inner
  | .pal bits => quantizeWeightsPalette bits vals cout inner
  | other => throw s!"PTQ quantization format '{other.name}' is not supported (expected int8, int4, pal4, or pal2)"

/-- Return `true` iff a `CoreOp.conv` supports native M8A quantized weights
    (dense `groups == 1`, stride 1, kernel 1x1 or 3x3). -/
def isQuantCapableConv : CoreOp → Bool
  | .conv kh kw sh sw _ groups _ _ =>
    groups == 1 && sh == 1 && sw == 1 && ((kh == 1 && kw == 1) || (kh == 3 && kw == 3))
  | _ => false

/-- Return `true` iff a `CoreOp.conv` can natively absorb quantized weights without splitting its activation. -/
def isQuantCapableConvWithAct : CoreOp → Bool
  | .conv kh kw sh sw _ groups _ act =>
    groups == 1 && sh == 1 && sw == 1 && ((kh == 1 && kw == 1) || (kh == 3 && kw == 3)) &&
    (act == .none || act == .relu)
  | _ => false

/-- Run the C1 Quantize / QDQ pass on `SSAGraph CoreOp`:
    1. For explicit `dequantize(const (.quant spec))` feeding a quant-capable `conv` that has a fused
       `.lut` activation, unfuses the `.lut` so native M8A quantization remains legal.
    2. When `ptqFormat? = some fmt` (`--quantize <int8|int4|pal4|pal2>`), quantizes fp16/f32 `const`
       weights of every dense stride-1 `conv` into `dequantize(const (.quant spec))` using `Vane.Quant`,
       un-fusing any `.lut` epilogue onto a separate `CoreOp.lut` node (keeping `.none` and `.relu` fused). -/
def quantizePass (g : SSAGraph CoreOp) (ptqFormat? : Option QuantFormat) :
    Except String (SSAGraph CoreOp × Nat) := do
  -- Index constant and dequantize producers by ValueId
  let mut constMap : Array (ValueId × TensorType × ConstPayload) := #[]
  let mut deqMap : Array (ValueId × QuantFormat × ValueId) := #[]
  for nd in g.nodes do
    match nd.op, nd.inputs.toList, nd.outputs.toList with
    | .const payload, [], [(vid, ty)] =>
      constMap := constMap.push (vid, ty, payload)
    | .dequantize fmt, [qId], [(vid, _)] =>
      deqMap := deqMap.push (vid, fmt, qId)
    | _, _, _ => pure ()

  let lookupConst := fun (v : ValueId) => constMap.find? fun e => e.1 == v
  let lookupDeq := fun (v : ValueId) => deqMap.find? fun e => e.1 == v

  let mut nextId := g.nextValueId
  let mut outNodes : Array (IRNode CoreOp) := Array.mkEmpty g.nodes.size
  let mut quantCount : Nat := 0

  for nd in g.nodes do
    match nd.op, nd.outputs.toList with
    | .conv kh kw sh sw pads groups hasBias act, [(outId, outTy)] =>
      if isQuantCapableConv nd.op && nd.inputs.size >= 2 then
        let wId := nd.inputs[1]!
        -- Case A: Already fed by explicit ONNX DequantizeLinear (`dequantize(const quant)`).
        if let some (_, _, qId) := lookupDeq wId then
          match lookupConst qId with
          | some (_, qTy, .quant qSpec) =>
            if qTy.shape.size == 4 then
              let cout := qTy.shape[0]!
              let cin := qTy.shape[1]!
              let taps := kh * kw
              validateQuantWeightSpec qSpec cout cin taps
              quantCount := quantCount + 1
              match act with
              | .lut lspec =>
                -- Unfuse LUT onto separate CoreOp.lut node so quantized conv stays M8A-legal.
                let convOutId := nextId
                nextId := nextId + 1
                outNodes := outNodes.push {
                  nd with
                  op := .conv kh kw sh sw pads groups hasBias .none
                  outputs := #[(convOutId, outTy)]
                }
                outNodes := outNodes.push {
                  name := s!"{nd.name}_lut"
                  op := .lut (.lut lspec)
                  inputs := #[convOutId]
                  outputs := #[(outId, outTy)]
                }
              | _ =>
                outNodes := outNodes.push nd
            else
              outNodes := outNodes.push nd
          | _ =>
            outNodes := outNodes.push nd
        -- Case B: Optional PTQ `--quantize <int8|int4|pal4|pal2>` on float `const` weights.
        else if let some fmt := ptqFormat? then
          match lookupConst wId with
          | some (_, wTy, payload) =>
            match payload.toFloat64? with
            | some wF64 =>
              if wTy.shape.size == 4 then
                let cout := wTy.shape[0]!
                let cin := wTy.shape[1]!
                let inner := cin * kh * kw
                let qSpec ← quantizeWeights fmt wF64 cout inner
                let qId := nextId
                let dqId := nextId + 1
                nextId := nextId + 2
                let qTy : TensorType := { shape := wTy.shape, dtype := DType.ofQuantFormat fmt }
                let dqTy : TensorType := { shape := wTy.shape, dtype := .fp16 }
                outNodes := outNodes.push {
                  name := s!"{nd.name}_qweight"
                  op := .const (.quant qSpec)
                  inputs := #[]
                  outputs := #[(qId, qTy)]
                }
                outNodes := outNodes.push {
                  name := s!"{nd.name}_dequant"
                  op := .dequantize fmt
                  inputs := #[qId]
                  outputs := #[(dqId, dqTy)]
                }
                let newInps := nd.inputs.set! 1 dqId
                quantCount := quantCount + 1
                match act with
                | .lut lspec =>
                  let convOutId := nextId
                  nextId := nextId + 1
                  outNodes := outNodes.push {
                    nd with
                    op := .conv kh kw sh sw pads groups hasBias .none
                    inputs := newInps
                    outputs := #[(convOutId, outTy)]
                  }
                  outNodes := outNodes.push {
                    name := s!"{nd.name}_lut"
                    op := .lut (.lut lspec)
                    inputs := #[convOutId]
                    outputs := #[(outId, outTy)]
                  }
                | _ =>
                  outNodes := outNodes.push { nd with inputs := newInps }
              else
                outNodes := outNodes.push nd
            | none =>
              outNodes := outNodes.push nd
          | none =>
            outNodes := outNodes.push nd
        else
          outNodes := outNodes.push nd
      else
        outNodes := outNodes.push nd
    | _, _ =>
      outNodes := outNodes.push nd

  pure (dce { g with nodes := outNodes }, quantCount)

end Vane.IR
