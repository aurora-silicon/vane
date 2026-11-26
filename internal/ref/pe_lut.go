package ref

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/aurora-silicon/vane/internal/fp16"
)

// ============================================================================
// Planar Engine (PE) Reference Models (H17 exact ValInH17 / RNA16 semantics)
// ============================================================================

// PeUnary evaluates 1-source PE unary ops ("abs", "square") with exact H17 PE arithmetic.
func PeUnary(op string, x Tensor) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	out, _ := NewTensor(x.N, x.C, x.H, x.W)
	switch op {
	case "abs":
		for i, h := range x.Data {
			out.Data[i] = fp16.RNA16(math.Abs(fp16.ValInH17(h)))
		}
	case "square":
		for i, h := range x.Data {
			v := fp16.ValInH17(h)
			out.Data[i] = fp16.RNA16(v * v)
		}
	default:
		return Tensor{}, fmt.Errorf("unsupported PE unary op %q (expected abs or square)", op)
	}
	return out, nil
}

// PeChain evaluates a 1-source PE affine/scalar chain:
// optional inScale -> optional bias -> optional scale, using RNA16Step intermediates
// and final RNA16 writeback.
func PeChain(x Tensor, inScale, bias, scale *uint16) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if inScale == nil && bias == nil && scale == nil {
		return Tensor{}, fmt.Errorf("PE chain requires at least one of inScale, bias, or scale")
	}
	out, _ := NewTensor(x.N, x.C, x.H, x.W)
	for i, h := range x.Data {
		t := fp16.ValInH17(h)
		if inScale != nil {
			t = fp16.RNA16Step(t * fp16.ValInH17(*inScale))
		}
		if bias != nil {
			t = fp16.RNA16Step(t + fp16.ValInH17(*bias))
		}
		if scale != nil {
			t = fp16.RNA16Step(t * fp16.ValInH17(*scale))
		}
		out.Data[i] = fp16.RNA16(t)
	}
	return out, nil
}

// PeChainF64 evaluates a 1-source PE affine/scalar chain in pure float64 -> fp16.
func PeChainF64(x Tensor, inScale, bias, scale *uint16) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if inScale == nil && bias == nil && scale == nil {
		return Tensor{}, fmt.Errorf("PE chain requires at least one of inScale, bias, or scale")
	}
	out, _ := NewTensor(x.N, x.C, x.H, x.W)
	for i, h := range x.Data {
		t := fp16.ToFloat64(h)
		if inScale != nil {
			t *= fp16.ToFloat64(*inScale)
		}
		if bias != nil {
			t += fp16.ToFloat64(*bias)
		}
		if scale != nil {
			t *= fp16.ToFloat64(*scale)
		}
		out.Data[i] = fp16.FromFloat64(t)
	}
	return out, nil
}

func broadcastAxis(a, b int, axis string) (int, error) {
	if a == b {
		return a, nil
	}
	if a == 1 {
		return b, nil
	}
	if b == 1 {
		return a, nil
	}
	return 0, fmt.Errorf("incompatible broadcast on %s: %d vs %d", axis, a, b)
}

// PeBinary evaluates a 2-source PE binary op ("add", "sub", "rsub", "mul", "max", "min")
// with full broadcasting across (C, H, W) and optional operand reversal (rev=true computes z op x),
// using exact H17 ValInH17 + RNA16 semantics.
func PeBinary(op string, x, z Tensor, rev bool) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if err := z.Validate(); err != nil {
		return Tensor{}, err
	}
	if x.N != z.N {
		return Tensor{}, fmt.Errorf("PE binary batch mismatch: %d vs %d", x.N, z.N)
	}
	outC, err := broadcastAxis(x.C, z.C, "C")
	if err != nil {
		return Tensor{}, err
	}
	outH, err := broadcastAxis(x.H, z.H, "H")
	if err != nil {
		return Tensor{}, err
	}
	outW, err := broadcastAxis(x.W, z.W, "W")
	if err != nil {
		return Tensor{}, err
	}

	effOp := op
	effRev := rev
	if effOp == "rsub" {
		effOp = "sub"
		effRev = !effRev
	}
	switch effOp {
	case "add", "sub", "mul", "max", "min":
	default:
		return Tensor{}, fmt.Errorf("unsupported PE binary op %q", op)
	}

	out, err := NewTensor(x.N, outC, outH, outW)
	if err != nil {
		return Tensor{}, err
	}

	for n := 0; n < x.N; n++ {
		for c := 0; c < outC; c++ {
			xc := c
			if x.C == 1 {
				xc = 0
			}
			zc := c
			if z.C == 1 {
				zc = 0
			}
			for h := 0; h < outH; h++ {
				xh := h
				if x.H == 1 {
					xh = 0
				}
				zh := h
				if z.H == 1 {
					zh = 0
				}
				for w := 0; w < outW; w++ {
					xw := w
					if x.W == 1 {
						xw = 0
					}
					zw := w
					if z.W == 1 {
						zw = 0
					}
					a := fp16.ValInH17(x.Data[x.Index(n, xc, xh, xw)])
					b := fp16.ValInH17(z.Data[z.Index(n, zc, zh, zw)])
					if effRev {
						a, b = b, a
					}
					var y float64
					switch effOp {
					case "add":
						y = a + b
					case "sub":
						y = a - b
					case "mul":
						y = a * b
					case "max":
						if a >= b {
							y = a
						} else {
							y = b
						}
					case "min":
						if a <= b {
							y = a
						} else {
							y = b
						}
					}
					out.Data[out.Index(n, c, h, w)] = fp16.RNA16(y)
				}
			}
		}
	}
	return out, nil
}

// PeBinaryF64 evaluates a 2-source binary op with broadcasting in standard float64 -> fp16.
func PeBinaryF64(op string, x, z Tensor, rev bool) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if err := z.Validate(); err != nil {
		return Tensor{}, err
	}
	if x.N != z.N {
		return Tensor{}, fmt.Errorf("PE binary batch mismatch: %d vs %d", x.N, z.N)
	}
	outC, err := broadcastAxis(x.C, z.C, "C")
	if err != nil {
		return Tensor{}, err
	}
	outH, err := broadcastAxis(x.H, z.H, "H")
	if err != nil {
		return Tensor{}, err
	}
	outW, err := broadcastAxis(x.W, z.W, "W")
	if err != nil {
		return Tensor{}, err
	}
	effOp := op
	effRev := rev
	if effOp == "rsub" {
		effOp = "sub"
		effRev = !effRev
	}
	out, err := NewTensor(x.N, outC, outH, outW)
	if err != nil {
		return Tensor{}, err
	}
	for n := 0; n < x.N; n++ {
		for c := 0; c < outC; c++ {
			xc, zc := c, c
			if x.C == 1 {
				xc = 0
			}
			if z.C == 1 {
				zc = 0
			}
			for h := 0; h < outH; h++ {
				xh, zh := h, h
				if x.H == 1 {
					xh = 0
				}
				if z.H == 1 {
					zh = 0
				}
				for w := 0; w < outW; w++ {
					xw, zw := w, w
					if x.W == 1 {
						xw = 0
					}
					if z.W == 1 {
						zw = 0
					}
					a := fp16.ToFloat64(x.Data[x.Index(n, xc, xh, xw)])
					b := fp16.ToFloat64(z.Data[z.Index(n, zc, zh, zw)])
					if effRev {
						a, b = b, a
					}
					var y float64
					switch effOp {
					case "add":
						y = a + b
					case "sub":
						y = a - b
					case "mul":
						y = a * b
					case "max":
						if math.IsNaN(a) || math.IsNaN(b) {
							y = math.NaN()
						} else if a >= b {
							y = a
						} else {
							y = b
						}
					case "min":
						if math.IsNaN(a) || math.IsNaN(b) {
							y = math.NaN()
						} else if a <= b {
							y = a
						} else {
							y = b
						}
					default:
						return Tensor{}, fmt.Errorf("unsupported PE binary op %q", op)
					}
					out.Data[out.Index(n, c, h, w)] = fp16.FromFloat64(y)
				}
			}
		}
	}
	return out, nil
}

// ============================================================================
// 128-Byte NE LUT Hardware Evaluator, Table Builder, and Fused Conv+LUT Models
// ============================================================================

// ActSpec describes an NE activation (standalone or fused onto a convolution).
type ActSpec struct {
	Name   string
	Alpha  float64
	Beta   float64
	Lo     float64
	Hi     float64
	Eps    float64
	Scalar float64
}

func f16CanonZero(x float64) uint16 {
	h := fp16.FromFloat64(x)
	if h == fp16.NegZero {
		return fp16.PosZero
	}
	return h
}

func finiteFP16(v float64, label string) (uint16, error) {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, fmt.Errorf("LUT %s: parameter must be finite (got %g)", label, v)
	}
	h := f16CanonZero(v)
	if !fp16.IsFinite(h) {
		return 0, fmt.Errorf("LUT %s: value %g overflows fp16 finite range [-65504, 65504]", label, v)
	}
	return h, nil
}

func packLutTable(lo, hi, outLo, outHi uint16, samples [33]uint16, ls, li, rs, ri, k, mode uint16) []byte {
	var words [64]uint16
	words[0] = lo
	words[1] = hi
	words[2] = outLo
	words[3] = outHi
	copy(words[4:37], samples[:])
	words[37] = ls
	words[38] = li
	words[39] = rs
	words[40] = ri
	words[41] = k
	words[42] = mode
	// words[43..63] remain 0
	out := make([]byte, 128)
	for i, w := range words {
		binary.LittleEndian.PutUint16(out[2*i:], w)
	}
	return out
}

func packPwlTable(lo, hi, outLo, outHi, f0, ls, li, rs, ri, mode uint16) []byte {
	var samples [33]uint16
	for i := range samples {
		samples[i] = f0
	}
	return packLutTable(lo, hi, outLo, outHi, samples, ls, li, rs, ri, 0x0040, mode)
}

// BuildLutTable mathematically generates the 128-byte hardware LUT table, pre-LUT scale
// register (0x4910), and optional input shift register (0x490c) for any LUT activation.
func BuildLutTable(spec ActSpec) (tableBytes []byte, preScaleBits uint16, inputShiftBits *uint16, err error) {
	const (
		posOne  uint16 = 0x3c00
		negOne  uint16 = 0xbc00
		invLn2  uint16 = 0x3dc5
		posZero uint16 = fp16.PosZero
		posInf  uint16 = fp16.PosInf
		negInf  uint16 = fp16.NegInf
	)
	preScaleBits = posOne
	name := strings.ToLower(spec.Name)
	switch name {
	case "relu6":
		return packPwlTable(posZero, f16CanonZero(6.0), posZero, f16CanonZero(6.0), posZero,
			posOne, posZero, posOne, posZero, 0x0001), posOne, nil, nil
	case "clip":
		loH, err := finiteFP16(spec.Lo, "clip.lo")
		if err != nil {
			return nil, 0, nil, err
		}
		hiH, err := finiteFP16(spec.Hi, "clip.hi")
		if err != nil {
			return nil, 0, nil, err
		}
		if spec.Lo > spec.Hi {
			return nil, 0, nil, fmt.Errorf("clip requires lo <= hi (got lo=%g, hi=%g)", spec.Lo, spec.Hi)
		}
		f0H, err := finiteFP16(math.Max(spec.Lo, math.Min(spec.Hi, 0.0)), "clip.f0")
		if err != nil {
			return nil, 0, nil, err
		}
		return packPwlTable(loH, hiH, loH, hiH, f0H,
			posOne, posZero, posOne, posZero, 0x0001), posOne, nil, nil
	case "leaky_relu":
		alphaH, err := finiteFP16(spec.Alpha, "leaky_relu.alpha")
		if err != nil {
			return nil, 0, nil, err
		}
		return packPwlTable(negInf, posInf, negInf, posInf, posZero,
			alphaH, posZero, posOne, posZero, 0x0001), posOne, nil, nil
	case "abs":
		return packPwlTable(negInf, posInf, posInf, posInf, posZero,
			negOne, posZero, posOne, posZero, 0x0021), posOne, nil, nil
	case "sign":
		return packPwlTable(0x8001, 0x0001, negOne, posOne, posZero,
			posZero, posZero, posZero, posZero, 0x0001), posOne, nil, nil
	case "threshold":
		aH, err := finiteFP16(spec.Scalar, "threshold.a")
		if err != nil {
			return nil, 0, nil, err
		}
		f0H, err := finiteFP16(math.Max(0.0, spec.Scalar), "threshold.f0")
		if err != nil {
			return nil, 0, nil, err
		}
		return packPwlTable(aH, posInf, aH, posInf, f0H,
			posOne, posZero, posOne, posZero, 0x0001), posOne, nil, nil
	case "thresholded_relu":
		lo, err := finiteFP16(spec.Alpha, "thresholded_relu.a")
		if err != nil {
			return nil, 0, nil, err
		}
		return packPwlTable(lo, posInf, posZero, posInf, posZero,
			posOne, posZero, posOne, posZero, 0x0001), posOne, nil, nil
	case "hard_sigmoid":
		if math.IsNaN(spec.Alpha) || math.IsInf(spec.Alpha, 0) || spec.Alpha <= 0 {
			return nil, 0, nil, fmt.Errorf("hard_sigmoid alpha must be finite and > 0 (got %g)", spec.Alpha)
		}
		if _, err := finiteFP16(spec.Beta, "hard_sigmoid.beta"); err != nil {
			return nil, 0, nil, err
		}
		scale := spec.Alpha / 0.2
		preScaleBits, err = finiteFP16(scale, "hard_sigmoid.preScale")
		if err != nil {
			return nil, 0, nil, err
		}
		if preScaleBits == posZero {
			return nil, 0, nil, fmt.Errorf("hard_sigmoid: alpha (%g) underflows pre-LUT scale to 0", spec.Alpha)
		}
		if spec.Beta != 0.5 {
			s, err := finiteFP16((5.0*spec.Beta-2.5)/scale, "hard_sigmoid.inputShift")
			if err != nil {
				return nil, 0, nil, err
			}
			if s != posZero {
				inputShiftBits = &s
			}
		}
		return packPwlTable(f16CanonZero(-2.5), f16CanonZero(2.5), posZero, posOne, f16CanonZero(0.5),
				f16CanonZero(0.2), f16CanonZero(0.5), f16CanonZero(0.2), f16CanonZero(0.5), 0x0001),
			preScaleBits, inputShiftBits, nil
	case "clamp_max":
		vH, err := finiteFP16(spec.Scalar, "clamp_max.v")
		if err != nil {
			return nil, 0, nil, err
		}
		f0H, err := finiteFP16(math.Max(0.0, spec.Scalar), "clamp_max.f0")
		if err != nil {
			return nil, 0, nil, err
		}
		return packPwlTable(vH, posInf, vH, posInf, f0H,
			posOne, posZero, posOne, posZero, 0x0001), posOne, nil, nil
	case "clamp_min":
		vH, err := finiteFP16(spec.Scalar, "clamp_min.v")
		if err != nil {
			return nil, 0, nil, err
		}
		f0H, err := finiteFP16(math.Min(0.0, spec.Scalar), "clamp_min.f0")
		if err != nil {
			return nil, 0, nil, err
		}
		return packPwlTable(negInf, vH, negInf, vH, f0H,
			posOne, posZero, posOne, posZero, 0x0001), posOne, nil, nil
	case "elu":
		if _, err := finiteFP16(spec.Alpha, "elu.alpha"); err != nil {
			return nil, 0, nil, err
		}
		negAlphaH, err := finiteFP16(-spec.Alpha, "elu.-alpha")
		if err != nil {
			return nil, 0, nil, err
		}
		var samples [33]uint16
		for i := 0; i < 33; i++ {
			u := float64(i-32) / 4.0
			samples[i] = f16CanonZero(spec.Alpha * math.Expm1(u))
		}
		return packLutTable(f16CanonZero(-8.0), posInf, negAlphaH, posInf, samples,
			posZero, negAlphaH, posOne, posZero, 2, 0x0002), posOne, nil, nil
	case "tanh":
		var samples [33]uint16
		for i := 0; i < 33; i++ {
			samples[i] = f16CanonZero(math.Tanh(float64(i) / 8.0))
		}
		return packLutTable(f16CanonZero(-4.0), f16CanonZero(4.0), negOne, posOne, samples,
			posZero, negOne, posZero, posOne, 3, 0x0060), posOne, nil, nil
	case "scaled_tanh":
		if _, err := finiteFP16(spec.Alpha, "scaled_tanh.alpha"); err != nil {
			return nil, 0, nil, err
		}
		betaH, err := finiteFP16(spec.Beta, "scaled_tanh.beta")
		if err != nil {
			return nil, 0, nil, err
		}
		a := math.Abs(spec.Alpha)
		aH, err := finiteFP16(a, "scaled_tanh.|alpha|")
		if err != nil {
			return nil, 0, nil, err
		}
		negAH, err := finiteFP16(-a, "scaled_tanh.-|alpha|")
		if err != nil {
			return nil, 0, nil, err
		}
		mode := uint16(0x0060)
		if spec.Alpha < 0 {
			mode = 0x00a0
		}
		var samples [33]uint16
		for i := 0; i < 33; i++ {
			samples[i] = f16CanonZero(a * math.Tanh(float64(i)/8.0))
		}
		return packLutTable(f16CanonZero(-4.0), f16CanonZero(4.0), negAH, aH, samples,
			posZero, negAH, posZero, aH, 3, mode), betaH, nil, nil
	case "erf":
		var samples [33]uint16
		for i := 0; i < 33; i++ {
			samples[i] = f16CanonZero(math.Erf(float64(i) / 8.0))
		}
		return packLutTable(f16CanonZero(-4.0), f16CanonZero(4.0), negOne, posOne, samples,
			posZero, negOne, posZero, posOne, 3, 0x0060), posOne, nil, nil
	case "sigmoid":
		var samples [33]uint16
		for i := 0; i < 33; i++ {
			u := float64(i-16) / 2.0
			samples[i] = f16CanonZero(1.0 / (1.0 + math.Exp(-u)))
		}
		return packLutTable(f16CanonZero(-8.0), f16CanonZero(8.0), posZero, posOne, samples,
			posZero, posZero, posZero, posOne, 1, 0x0011), posOne, nil, nil
	case "gelu":
		invSqrt2 := 1.0 / math.Sqrt2
		var samples [33]uint16
		for i := 0; i < 33; i++ {
			u := float64(i-16) / 4.0
			samples[i] = f16CanonZero(0.5 * u * (1.0 + math.Erf(u*invSqrt2)))
		}
		return packLutTable(f16CanonZero(-4.0), posInf, posZero, posInf, samples,
			posZero, posZero, posOne, posZero, 2, 0x0011), posOne, nil, nil
	case "silu":
		var samples [33]uint16
		for i := 0; i < 33; i++ {
			u := float64(i-16) / 2.0
			samples[i] = f16CanonZero(u / (1.0 + math.Exp(-u)))
		}
		return packLutTable(f16CanonZero(-8.0), posInf, posZero, posInf, samples,
			posZero, posZero, posOne, posZero, 1, 0x0011), posOne, nil, nil
	case "exp2":
		var samples [33]uint16
		for i := 0; i < 33; i++ {
			samples[i] = f16CanonZero(math.Exp2(float64(i) / 32.0))
		}
		return packLutTable(f16CanonZero(-24.0), f16CanonZero(16.0), posZero, posInf, samples,
			posZero, posZero, posZero, posZero, 5, 0x000a), posOne, nil, nil
	case "exp":
		var samples [33]uint16
		for i := 0; i < 33; i++ {
			samples[i] = f16CanonZero(math.Exp2(float64(i) / 32.0))
		}
		return packLutTable(f16CanonZero(-24.0), f16CanonZero(16.0), posZero, posInf, samples,
			posZero, posZero, posZero, posZero, 5, 0x000a), invLn2, nil, nil
	case "sqrt":
		var samples [33]uint16
		for i := 0; i < 33; i++ {
			samples[i] = f16CanonZero(math.Sqrt(float64(i) / 16.0))
		}
		return packLutTable(posZero, posInf, posZero, posInf, samples,
			posZero, posZero, posZero, posZero, 4, 0x0005), posOne, nil, nil
	case "rsqrt":
		if math.IsNaN(spec.Eps) || math.IsInf(spec.Eps, 0) || spec.Eps < 0 {
			return nil, 0, nil, fmt.Errorf("rsqrt eps must be finite and >= 0 (got %g)", spec.Eps)
		}
		epsH, err := finiteFP16(spec.Eps, "rsqrt.eps")
		if err != nil {
			return nil, 0, nil, err
		}
		var samples [33]uint16
		samples[0] = posZero
		for i := 1; i < 33; i++ {
			samples[i] = f16CanonZero(1.0 / math.Sqrt(float64(i)/16.0))
		}
		if spec.Eps != 0 && epsH != posZero {
			inputShiftBits = &epsH
		}
		return packLutTable(posZero, posInf, posInf, posZero, samples,
			posZero, posZero, posZero, posZero, 4, 0x0006), posOne, inputShiftBits, nil
	case "inverse":
		var samples [33]uint16
		samples[0] = posZero
		for i := 1; i < 33; i++ {
			samples[i] = f16CanonZero(32.0 / float64(i))
		}
		return packLutTable(negInf, posInf, posZero, posZero, samples,
			posZero, posZero, posZero, posZero, 5, 0x0067), posOne, nil, nil
	case "log2":
		var samples [33]uint16
		samples[0] = negInf
		for i := 1; i < 33; i++ {
			samples[i] = f16CanonZero(math.Log2(float64(i) / 32.0))
		}
		return packLutTable(posZero, posInf, negInf, posInf, samples,
			posZero, posZero, posZero, posZero, 5, 0x0009), posOne, nil, nil
	default:
		return nil, 0, nil, fmt.Errorf("unknown LUT activation %q", spec.Name)
	}
}

func lerp33(y *[33]float64, u float64) float64 {
	if u < 0 {
		u = 0
	} else if u > 32 {
		u = 32
	}
	i := int(math.Floor(u))
	if i > 31 {
		i = 31
	}
	frac := u - float64(i)
	return y[i] + frac*(y[i+1]-y[i])
}

func decodeLutTable(tableBytes []byte) (words [64]uint16, err error) {
	if len(tableBytes) != 128 {
		return words, fmt.Errorf("LUT table size %d != 128 bytes", len(tableBytes))
	}
	for i := 0; i < 64; i++ {
		words[i] = binary.LittleEndian.Uint16(tableBytes[2*i:])
	}
	for i := 43; i < 64; i++ {
		if words[i] != 0 {
			return words, fmt.Errorf("LUT reserved word %d is non-zero (0x%04x)", i, words[i])
		}
	}
	return words, nil
}

// EvalLutFloat64 evaluates a 128-byte NE LUT table on a float64 input value (e.g. a Q15.16
// convolution accumulator before fp16 rounding), applying optional input shift and pre-LUT scale.
func EvalLutFloat64(tableBytes []byte, x float64, preScaleBits uint16, inputShiftBits *uint16) (uint16, error) {
	words, err := decodeLutTable(tableBytes)
	if err != nil {
		return 0, err
	}
	if math.IsNaN(x) {
		return fp16.PosZero, nil
	}
	if inputShiftBits != nil {
		x += fp16.ToFloat64(*inputShiftBits)
	}
	if preScaleBits != 0x3c00 {
		x *= fp16.ToFloat64(preScaleBits)
	}
	if math.IsNaN(x) {
		return fp16.PosZero, nil
	}

	lo := fp16.ToFloat64(words[0])
	hi := fp16.ToFloat64(words[1])
	outLo := fp16.ToFloat64(words[2])
	outHi := fp16.ToFloat64(words[3])
	var y [33]float64
	for i := 0; i < 33; i++ {
		y[i] = fp16.ToFloat64(words[4+i])
	}
	ls := fp16.ToFloat64(words[37])
	li := fp16.ToFloat64(words[38])
	rs := fp16.ToFloat64(words[39])
	ri := fp16.ToFloat64(words[40])
	k := int(words[41])
	mode := words[42]
	sym := mode & 0xe0
	base := mode & 0x1f

	negOut := false
	xp := x
	switch sym {
	case 0x00:
	case 0x20:
		xp = math.Abs(x)
	case 0x60:
		if x < 0 {
			negOut = true
			xp = -x
		}
	case 0x80:
		negOut = true
	case 0xa0:
		if x < 0 {
			xp = -x
		} else {
			negOut = true
		}
	case 0xc0:
		if x < 0 {
			return fp16.PosZero, nil
		}
	default:
		return 0, fmt.Errorf("unsupported LUT symmetry flag 0x%02x", sym)
	}

	var out float64
	if xp <= lo {
		out = outLo
	} else if xp >= hi {
		out = outHi
	} else {
		switch base {
		case 0x01:
			if xp == 0 {
				out = y[0]
			} else if xp < 0 {
				out = ls*xp + li
			} else {
				out = rs*xp + ri
			}
		case 0x00:
			scale := math.Ldexp(1.0, k)
			dom := 32.0 / scale
			if xp < 0 {
				out = ls*xp + li
			} else if xp >= dom {
				out = rs*xp + ri
			} else {
				out = lerp33(&y, xp*scale)
			}
		case 0x02:
			scale := math.Ldexp(1.0, k)
			dom := 32.0 / scale
			if xp < -dom {
				out = ls*xp + li
			} else if xp >= 0 {
				out = rs*xp + ri
			} else {
				out = lerp33(&y, 32.0+xp*scale)
			}
		case 0x11:
			scale := math.Ldexp(1.0, k)
			dom := 16.0 / scale
			if xp < -dom {
				out = ls*xp + li
			} else if xp >= dom {
				out = rs*xp + ri
			} else {
				out = lerp33(&y, 16.0+xp*scale)
			}
		case 0x05:
			m, e := math.Frexp(xp)
			var m2 float64
			var q int
			if e%2 == 0 {
				m2 = m
				q = e / 2
			} else {
				m2 = 2.0 * m
				q = (e - 1) / 2
			}
			out = math.Ldexp(lerp33(&y, math.Ldexp(m2, k)), q)
		case 0x06:
			m, e := math.Frexp(xp)
			var m2 float64
			var q int
			if e%2 == 0 {
				m2 = m
				q = e / 2
			} else {
				m2 = 2.0 * m
				q = (e - 1) / 2
			}
			out = math.Ldexp(lerp33(&y, math.Ldexp(m2, k)), -q)
		case 0x07:
			if xp == 0 {
				out = math.Inf(1)
			} else {
				m, e := math.Frexp(xp)
				out = math.Ldexp(lerp33(&y, math.Ldexp(m, k)), -e)
			}
		case 0x09:
			m, e := math.Frexp(xp)
			out = float64(e) + lerp33(&y, math.Ldexp(m, k))
		case 0x0a:
			s := math.Ldexp(xp, k)
			qF := math.Floor(s / 32.0)
			u := s - 32.0*qF
			out = math.Ldexp(lerp33(&y, u), int(qF))
		default:
			return 0, fmt.Errorf("unsupported LUT base mode 0x%02x", base)
		}
	}

	if negOut {
		out = -out
	}
	return fp16.RNA16(out), nil
}

// EvalLut evaluates a 128-byte NE LUT table on an IEEE-754 binary16 input bit pattern.
func EvalLut(tableBytes []byte, xBits uint16, preScaleBits uint16, inputShiftBits *uint16) (uint16, error) {
	return EvalLutFloat64(tableBytes, fp16.ToFloat64(xBits), preScaleBits, inputShiftBits)
}

// ApplyLutTensor applies EvalLut elementwise to an fp16 Tensor.
func ApplyLutTensor(x Tensor, tableBytes []byte, preScaleBits uint16, inputShiftBits *uint16) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if _, err := decodeLutTable(tableBytes); err != nil {
		return Tensor{}, err
	}
	out, _ := NewTensor(x.N, x.C, x.H, x.W)
	for i, h := range x.Data {
		y, err := EvalLut(tableBytes, h, preScaleBits, inputShiftBits)
		if err != nil {
			return Tensor{}, err
		}
		out.Data[i] = y
	}
	return out, nil
}

// EvalActExactF64 evaluates the exact analytical float64 formula for any NE activation.
func EvalActExactF64(spec ActSpec, x float64) (float64, error) {
	switch strings.ToLower(spec.Name) {
	case "", "none", "copy":
		return x, nil
	case "relu":
		if x < 0 {
			return 0, nil
		}
		return x, nil
	case "relu6":
		return math.Max(0, math.Min(6, x)), nil
	case "clip":
		return math.Max(spec.Lo, math.Min(spec.Hi, x)), nil
	case "leaky_relu":
		if x < 0 {
			return spec.Alpha * x, nil
		}
		return x, nil
	case "abs":
		return math.Abs(x), nil
	case "sign":
		if x < 0 {
			return -1, nil
		}
		if x > 0 {
			return 1, nil
		}
		return 0, nil
	case "threshold":
		return math.Max(x, spec.Scalar), nil
	case "thresholded_relu":
		if x > spec.Alpha {
			return x, nil
		}
		return 0, nil
	case "hard_sigmoid":
		return math.Max(0, math.Min(1, spec.Alpha*x+spec.Beta)), nil
	case "clamp_max":
		return math.Max(x, spec.Scalar), nil
	case "clamp_min":
		return math.Min(x, spec.Scalar), nil
	case "elu":
		if x >= 0 {
			return x, nil
		}
		return spec.Alpha * math.Expm1(x), nil
	case "tanh":
		return math.Tanh(x), nil
	case "scaled_tanh":
		return spec.Alpha * math.Tanh(spec.Beta*x), nil
	case "erf":
		return math.Erf(x), nil
	case "sigmoid":
		return 1.0 / (1.0 + math.Exp(-x)), nil
	case "gelu":
		return 0.5 * x * (1.0 + math.Erf(x/math.Sqrt2)), nil
	case "silu":
		return x / (1.0 + math.Exp(-x)), nil
	case "exp2":
		return math.Exp2(x), nil
	case "exp":
		return math.Exp(x), nil
	case "sqrt":
		if x <= 0 {
			return 0, nil
		}
		return math.Sqrt(x), nil
	case "rsqrt":
		v := x + spec.Eps
		if v <= 0 {
			return math.Inf(1), nil
		}
		return 1.0 / math.Sqrt(v), nil
	case "inverse":
		if x == 0 {
			return math.Inf(1), nil
		}
		return 1.0 / x, nil
	case "log2":
		if x <= 0 {
			return math.Inf(-1), nil
		}
		return math.Log2(x), nil
	default:
		return 0, fmt.Errorf("unknown activation %q", spec.Name)
	}
}

// ApplyActF64 evaluates an analytical float64 activation elementwise and rounds to fp16.
func ApplyActF64(x Tensor, spec ActSpec) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	out, _ := NewTensor(x.N, x.C, x.H, x.W)
	for i, h := range x.Data {
		y, err := EvalActExactF64(spec, fp16.ToFloat64(h))
		if err != nil {
			return Tensor{}, err
		}
		out.Data[i] = fp16.FromFloat64(y)
	}
	return out, nil
}

func (a *q1516Acc) toFloat64() float64 {
	if a.stickyInf > 0 {
		return math.Inf(1)
	}
	if a.stickyInf < 0 {
		return math.Inf(-1)
	}
	return float64(a.acc) / 65536.0
}

// Conv2d1x1Act computes a 1x1 convolution with float64 accumulation followed by an exact float64 activation.
func Conv2d1x1Act(x Tensor, weight []uint16, cout int, bias []uint16, spec ActSpec) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if cout <= 0 {
		return Tensor{}, fmt.Errorf("cout must be > 0, got %d", cout)
	}
	if len(weight) != cout*x.C {
		return Tensor{}, fmt.Errorf("weight length %d != cout*cin (%d)", len(weight), cout*x.C)
	}
	if bias != nil && len(bias) != cout {
		return Tensor{}, fmt.Errorf("bias length %d != cout (%d)", len(bias), cout)
	}
	out, err := NewTensor(x.N, cout, x.H, x.W)
	if err != nil {
		return Tensor{}, err
	}
	hw := x.H * x.W
	for n := 0; n < x.N; n++ {
		xBaseN := n * x.C * hw
		yBaseN := n * cout * hw
		for o := 0; o < cout; o++ {
			bVal := 0.0
			if bias != nil {
				bVal = fp16.ToFloat64(bias[o])
			}
			for p := 0; p < hw; p++ {
				acc := bVal
				for c := 0; c < x.C; c++ {
					acc += fp16.ToFloat64(x.Data[xBaseN+c*hw+p]) * fp16.ToFloat64(weight[o*x.C+c])
				}
				y, err := EvalActExactF64(spec, acc)
				if err != nil {
					return Tensor{}, err
				}
				out.Data[yBaseN+o*hw+p] = fp16.FromFloat64(y)
			}
		}
	}
	return out, nil
}

// foldFusedBiasShift folds an optional LUT input shift into the fp16 convolution bias
// vector (or synthesizes a constant bias vector when bias == nil), matching fused
// hardware lowering and checking for fp16 overflow.
func foldFusedBiasShift(bias []uint16, cout int, inputShiftBits *uint16) ([]uint16, error) {
	if inputShiftBits == nil {
		return bias, nil
	}
	if !fp16.IsFinite(*inputShiftBits) {
		return nil, fmt.Errorf("fused LUT input shift 0x%04x is not finite fp16", *inputShiftBits)
	}
	shiftF := fp16.ToFloat64(*inputShiftBits)
	eff := make([]uint16, cout)
	if bias != nil {
		for o := 0; o < cout; o++ {
			u := f16CanonZero(fp16.ToFloat64(bias[o]) + shiftF)
			if !fp16.IsFinite(u) {
				return nil, fmt.Errorf("fused LUT input shift overflowed fp16 on bias[%d] (orig=0x%04x, shift=0x%04x)", o, bias[o], *inputShiftBits)
			}
			eff[o] = u
		}
	} else {
		for o := 0; o < cout; o++ {
			eff[o] = *inputShiftBits
		}
	}
	return eff, nil
}

// Conv2d1x1Q1516Lut computes a 1x1 convolution in Q15.16 followed by a fused 128-byte NE LUT.
// Any inputShiftBits is folded into the fp16 bias before Q15.16 accumulation, matching hardware lowering.
func Conv2d1x1Q1516Lut(x Tensor, weight []uint16, cout int, bias []uint16, tableBytes []byte, preScaleBits uint16, inputShiftBits *uint16) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if cout <= 0 {
		return Tensor{}, fmt.Errorf("cout must be > 0, got %d", cout)
	}
	if len(weight) != cout*x.C {
		return Tensor{}, fmt.Errorf("weight length %d != cout*cin (%d)", len(weight), cout*x.C)
	}
	if bias != nil && len(bias) != cout {
		return Tensor{}, fmt.Errorf("bias length %d != cout (%d)", len(bias), cout)
	}
	if _, err := decodeLutTable(tableBytes); err != nil {
		return Tensor{}, err
	}
	effBias, err := foldFusedBiasShift(bias, cout, inputShiftBits)
	if err != nil {
		return Tensor{}, err
	}
	out, err := NewTensor(x.N, cout, x.H, x.W)
	if err != nil {
		return Tensor{}, err
	}
	hw := x.H * x.W
	for n := 0; n < x.N; n++ {
		xBaseN := n * x.C * hw
		yBaseN := n * cout * hw
		for o := 0; o < cout; o++ {
			wRow := weight[o*x.C : (o+1)*x.C]
			yBaseO := yBaseN + o*hw
			for p := 0; p < hw; p++ {
				var acc q1516Acc
				for c := 0; c < x.C; c++ {
					acc.addProduct(x.Data[xBaseN+c*hw+p], wRow[c])
				}
				if effBias != nil {
					acc.addBias(effBias[o])
				}
				y, err := EvalLutFloat64(tableBytes, acc.toFloat64(), preScaleBits, nil)
				if err != nil {
					return Tensor{}, err
				}
				out.Data[yBaseO+p] = y
			}
		}
	}
	return out, nil
}

// Conv2d3x3PadAct computes a 3x3 convolution in float64 followed by an exact float64 activation.
func Conv2d3x3PadAct(x Tensor, weight []uint16, cout int, bias []uint16, stride int, pad string, groups int, spec ActSpec) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if cout <= 0 || groups <= 0 || x.C%groups != 0 || cout%groups != 0 {
		return Tensor{}, fmt.Errorf("invalid 3x3 channels/groups: Cin=%d, Cout=%d, groups=%d", x.C, cout, groups)
	}
	cinPerG := x.C / groups
	coutPerG := cout / groups
	if len(weight) != cout*cinPerG*9 {
		return Tensor{}, fmt.Errorf("weight length %d != %d", len(weight), cout*cinPerG*9)
	}
	if bias != nil && len(bias) != cout {
		return Tensor{}, fmt.Errorf("bias length %d != %d", len(bias), cout)
	}
	hout, wout, padTop, padLeft, err := Conv3x3Geometry(x.H, x.W, stride, pad)
	if err != nil {
		return Tensor{}, err
	}
	out, err := NewTensor(x.N, cout, hout, wout)
	if err != nil {
		return Tensor{}, err
	}
	for n := 0; n < x.N; n++ {
		for g := 0; g < groups; g++ {
			cinStart := g * cinPerG
			coutStart := g * coutPerG
			for og := 0; og < coutPerG; og++ {
				o := coutStart + og
				bVal := 0.0
				if bias != nil {
					bVal = fp16.ToFloat64(bias[o])
				}
				for oh := 0; oh < hout; oh++ {
					for ow := 0; ow < wout; ow++ {
						acc := bVal
						for cg := 0; cg < cinPerG; cg++ {
							c := cinStart + cg
							wBase := (o*cinPerG + cg) * 9
							for kh := 0; kh < 3; kh++ {
								ih := oh*stride + kh - padTop
								if ih < 0 || ih >= x.H {
									continue
								}
								for kw := 0; kw < 3; kw++ {
									iw := ow*stride + kw - padLeft
									if iw < 0 || iw >= x.W {
										continue
									}
									acc += fp16.ToFloat64(x.Data[x.Index(n, c, ih, iw)]) * fp16.ToFloat64(weight[wBase+kh*3+kw])
								}
							}
						}
						y, err := EvalActExactF64(spec, acc)
						if err != nil {
							return Tensor{}, err
						}
						out.Data[out.Index(n, o, oh, ow)] = fp16.FromFloat64(y)
					}
				}
			}
		}
	}
	return out, nil
}

// Conv2d3x3Q1516Lut computes a 3x3 convolution in Q15.16 followed by a fused 128-byte NE LUT.
// Any inputShiftBits is folded into the fp16 bias before Q15.16 accumulation, matching hardware lowering.
func Conv2d3x3Q1516Lut(x Tensor, weight []uint16, cout int, bias []uint16, stride int, pad string, groups int, tableBytes []byte, preScaleBits uint16, inputShiftBits *uint16) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if cout <= 0 || groups <= 0 || x.C%groups != 0 || cout%groups != 0 {
		return Tensor{}, fmt.Errorf("invalid 3x3 channels/groups: Cin=%d, Cout=%d, groups=%d", x.C, cout, groups)
	}
	cinPerG := x.C / groups
	coutPerG := cout / groups
	if len(weight) != cout*cinPerG*9 {
		return Tensor{}, fmt.Errorf("weight length %d != %d", len(weight), cout*cinPerG*9)
	}
	if bias != nil && len(bias) != cout {
		return Tensor{}, fmt.Errorf("bias length %d != %d", len(bias), cout)
	}
	if _, err := decodeLutTable(tableBytes); err != nil {
		return Tensor{}, err
	}
	effBias, err := foldFusedBiasShift(bias, cout, inputShiftBits)
	if err != nil {
		return Tensor{}, err
	}
	hout, wout, padTop, padLeft, err := Conv3x3Geometry(x.H, x.W, stride, pad)
	if err != nil {
		return Tensor{}, err
	}
	out, err := NewTensor(x.N, cout, hout, wout)
	if err != nil {
		return Tensor{}, err
	}
	for n := 0; n < x.N; n++ {
		for g := 0; g < groups; g++ {
			cinStart := g * cinPerG
			coutStart := g * coutPerG
			for og := 0; og < coutPerG; og++ {
				o := coutStart + og
				for oh := 0; oh < hout; oh++ {
					for ow := 0; ow < wout; ow++ {
						var acc q1516Acc
						for cg := 0; cg < cinPerG; cg++ {
							c := cinStart + cg
							wBase := (o*cinPerG + cg) * 9
							for kh := 0; kh < 3; kh++ {
								ih := oh*stride + kh - padTop
								if ih < 0 || ih >= x.H {
									continue
								}
								for kw := 0; kw < 3; kw++ {
									iw := ow*stride + kw - padLeft
									if iw < 0 || iw >= x.W {
										continue
									}
									acc.addProduct(x.Data[x.Index(n, c, ih, iw)], weight[wBase+kh*3+kw])
								}
							}
						}
						if effBias != nil {
							acc.addBias(effBias[o])
						}
						y, err := EvalLutFloat64(tableBytes, acc.toFloat64(), preScaleBits, nil)
						if err != nil {
							return Tensor{}, err
						}
						out.Data[out.Index(n, o, oh, ow)] = y
					}
				}
			}
		}
	}
	return out, nil
}
