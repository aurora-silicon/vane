package ref

import (
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/aurora-silicon/vane/internal/fp16"
)

// QuantWeights holds quantized or palettized static weights matching Vane's
// native Neural Engine quantization IR.
type QuantWeights struct {
	// Format is one of "int8", "uint8", "int4", "uint4", "pal1", "pal2", "pal4", "pal6", "pal8".
	Format string
	// Codes stores row-major [Cout, Cin * taps] raw code bytes.
	Codes []byte
	// Scales stores strictly positive finite float32 scales: empty for pal*,
	// length 1 for per-tensor, or length Cout for per-channel integer formats.
	Scales []float32
	// ZeroPoints optionally stores length Cout zero-point bytes for integer formats; nil for pal*.
	ZeroPoints []uint8
	// Palette stores exact 2^bits finite IEEE fp16 entries for pal*; nil otherwise.
	Palette []uint16
}

// ParseQuantFormat validates and classifies a Vane quantized weight format name.
func ParseQuantFormat(format string) (norm string, bits int, isInt bool, isUnsigned bool, err error) {
	switch strings.ToLower(format) {
	case "int8":
		return "int8", 8, true, false, nil
	case "uint8":
		return "uint8", 8, true, true, nil
	case "int4":
		return "int4", 4, true, false, nil
	case "uint4":
		return "uint4", 4, true, true, nil
	case "pal1":
		return "pal1", 1, false, false, nil
	case "pal2":
		return "pal2", 2, false, false, nil
	case "pal4":
		return "pal4", 4, false, false, nil
	case "pal6":
		return "pal6", 6, false, false, nil
	case "pal8":
		return "pal8", 8, false, false, nil
	default:
		return "", 0, false, false, fmt.Errorf("unsupported quantized weight format %q", format)
	}
}

// ComputePexp chooses the smallest integer pexp in [0, 30] such that
// minScale * 2^pexp >= 2^-16, matching Vane's hardware scale exponent selection.
func ComputePexp(scales []float32) int {
	if len(scales) == 0 {
		return 0
	}
	minS := float64(scales[0])
	for _, s := range scales[1:] {
		if float64(s) < minS {
			minS = float64(s)
		}
	}
	threshold := math.Ldexp(1.0, -16)
	for e := 0; e <= 30; e++ {
		if math.Ldexp(minS, e) >= threshold {
			return e
		}
	}
	return 30
}

// ComputeBiasShift chooses the smallest integer biasShift in [0, 15] such that
// max_o(|bias[o] / scale[o]|) / 2^biasShift < 32768.
func ComputeBiasShift(scales []float64, bias []uint16) int {
	if len(bias) == 0 || len(scales) == 0 {
		return 0
	}
	maxRatio := 0.0
	for o, bh := range bias {
		s := scales[0]
		if len(scales) > 1 {
			s = scales[o]
		}
		if s > 0 {
			r := math.Abs(fp16.ToFloat64(bh) / s)
			if r > maxRatio {
				maxRatio = r
			}
		}
	}
	for shift := 0; shift <= 15; shift++ {
		if math.Ldexp(maxRatio, -shift) < 32768.0 {
			return shift
		}
	}
	return 15
}

func validateQuantInputs(weights []float64, cout, inner int) error {
	if cout <= 0 || inner <= 0 {
		return fmt.Errorf("invalid quantized weight dims (cout=%d, inner=%d): must be > 0", cout, inner)
	}
	if len(weights) != cout*inner {
		return fmt.Errorf("weights length %d != cout*inner (%d)", len(weights), cout*inner)
	}
	for i, w := range weights {
		if math.IsNaN(w) || math.IsInf(w, 0) {
			return fmt.Errorf("weight[%d] = %v must be finite", i, w)
		}
	}
	return nil
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func quantizeAffineInteger(weights []float64, cout, inner int, format string, qMin, qMax int, perChannel, withZP, isUnsigned, mask4 bool) (QuantWeights, error) {
	if err := validateQuantInputs(weights, cout, inner); err != nil {
		return QuantWeights{}, err
	}
	numGroups := 1
	groupLen := cout * inner
	if perChannel {
		numGroups = cout
		groupLen = inner
	}

	scales := make([]float32, numGroups)
	var zps []uint8
	if withZP || isUnsigned {
		zps = make([]uint8, cout)
	}
	codes := make([]byte, cout*inner)

	for g := 0; g < numGroups; g++ {
		start := g * groupLen
		end := start + groupLen
		wMin := weights[start]
		wMax := weights[start]
		for _, w := range weights[start+1 : end] {
			if w < wMin {
				wMin = w
			}
			if w > wMax {
				wMax = w
			}
		}

		var scale float64
		var zp int
		if withZP || isUnsigned {
			// Ensure 0 is representable within [wMin, wMax] for clean affine quantization
			if wMin > 0 {
				wMin = 0
			}
			if wMax < 0 {
				wMax = 0
			}
			span := wMax - wMin
			if span == 0 {
				scale = 1.0
				zp = clampInt(0, qMin, qMax)
			} else {
				scale = span / float64(qMax-qMin)
				if scale == 0 || float32(scale) <= 0 {
					scale = 1.0
				}
				zp = clampInt(int(math.Round(float64(qMin)-wMin/scale)), qMin, qMax)
			}
		} else {
			maxAbs := math.Max(math.Abs(wMin), math.Abs(wMax))
			if maxAbs == 0 {
				scale = 1.0
			} else {
				scale = maxAbs / float64(qMax)
				if scale == 0 || float32(scale) <= 0 {
					scale = 1.0
				}
			}
			zp = 0
		}

		scale32 := float32(scale)
		if scale32 <= 0 || math.IsInf(float64(scale32), 0) || math.IsNaN(float64(scale32)) {
			scale32 = 1.0
		}
		scales[g] = scale32
		s := float64(scale32)

		oStart, oEnd := 0, cout
		if perChannel {
			oStart, oEnd = g, g+1
		}
		for o := oStart; o < oEnd; o++ {
			if zps != nil {
				if isUnsigned {
					zps[o] = uint8(zp)
				} else {
					zps[o] = uint8(int8(zp))
				}
			}
			rowOff := o * inner
			for j := 0; j < inner; j++ {
				q := clampInt(int(math.Round(weights[rowOff+j]/s))+zp, qMin, qMax)
				var b byte
				if isUnsigned {
					b = byte(q)
				} else {
					b = byte(int8(q))
				}
				if mask4 {
					b &= 0x0f
				}
				codes[rowOff+j] = b
			}
		}
	}

	return QuantWeights{
		Format:     format,
		Codes:      codes,
		Scales:     scales,
		ZeroPoints: zps,
	}, nil
}

// QuantizeInt8 quantizes row-major [cout, inner] weights into native NE int8 format.
func QuantizeInt8(weights []float64, cout, inner int, perChannel, withZP bool) (QuantWeights, error) {
	return quantizeAffineInteger(weights, cout, inner, "int8", -128, 127, perChannel, withZP, false, false)
}

// QuantizeUint8 quantizes row-major [cout, inner] weights into native NE uint8 format with zero-points.
func QuantizeUint8(weights []float64, cout, inner int, perChannel bool) (QuantWeights, error) {
	return quantizeAffineInteger(weights, cout, inner, "uint8", 0, 255, perChannel, true, true, false)
}

// QuantizeInt4 quantizes row-major [cout, inner] weights into native NE int4 format (4-bit signed nibbles).
func QuantizeInt4(weights []float64, cout, inner int, perChannel, withZP bool) (QuantWeights, error) {
	return quantizeAffineInteger(weights, cout, inner, "int4", -8, 7, perChannel, withZP, false, true)
}

// QuantizeUint4 quantizes row-major [cout, inner] weights into native NE uint4 format with zero-points.
func QuantizeUint4(weights []float64, cout, inner int, perChannel bool) (QuantWeights, error) {
	return quantizeAffineInteger(weights, cout, inner, "uint4", 0, 15, perChannel, true, true, true)
}

// QuantizePalette clusters weights into 2^bits IEEE fp16 palette centroids via deterministic
// 1D k-means (Lloyd-Max), supporting bits in {1, 2, 4, 6, 8}.
func QuantizePalette(weights []float64, bits int) (QuantWeights, error) {
	if bits != 1 && bits != 2 && bits != 4 && bits != 6 && bits != 8 {
		return QuantWeights{}, fmt.Errorf("unsupported palette bit width %d (expected 1, 2, 4, 6, or 8)", bits)
	}
	if len(weights) == 0 {
		return QuantWeights{}, fmt.Errorf("weights cannot be empty")
	}
	for i, w := range weights {
		if math.IsNaN(w) || math.IsInf(w, 0) {
			return QuantWeights{}, fmt.Errorf("weight[%d] = %v must be finite", i, w)
		}
	}

	k := 1 << bits
	sorted := make([]float64, len(weights))
	copy(sorted, weights)
	sort.Float64s(sorted)

	// Initialize centroids deterministically at empirical quantiles
	centroids := make([]float64, k)
	for c := 0; c < k; c++ {
		if k == 1 {
			centroids[c] = sorted[len(sorted)/2]
		} else {
			pos := float64(c) * float64(len(sorted)-1) / float64(k-1)
			lo := int(math.Floor(pos))
			hi := int(math.Ceil(pos))
			frac := pos - float64(lo)
			centroids[c] = sorted[lo]*(1-frac) + sorted[hi]*frac
		}
	}

	assign := make([]int, len(weights))
	sums := make([]float64, k)
	counts := make([]int, k)

	// Lloyd-Max / 1D k-means iterations
	for iter := 0; iter < 32; iter++ {
		changed := false
		for c := 0; c < k; c++ {
			sums[c] = 0
			counts[c] = 0
		}
		for i, w := range weights {
			bestC := 0
			bestD := math.Abs(w - centroids[0])
			for c := 1; c < k; c++ {
				d := math.Abs(w - centroids[c])
				if d < bestD {
					bestD = d
					bestC = c
				}
			}
			if assign[i] != bestC {
				assign[i] = bestC
				changed = true
			}
			sums[bestC] += w
			counts[bestC]++
		}
		for c := 0; c < k; c++ {
			if counts[c] > 0 {
				centroids[c] = sums[c] / float64(counts[c])
			}
		}
		if !changed && iter > 0 {
			break
		}
	}

	// Round centroids to finite IEEE fp16 and run final nearest-palette assignment
	palette := make([]uint16, k)
	palF64 := make([]float64, k)
	for c := 0; c < k; c++ {
		h := fp16.FromFloat64(centroids[c])
		if !fp16.IsFinite(h) {
			if centroids[c] < 0 {
				h = fp16.FromFloat64(-65504.0)
			} else {
				h = fp16.FromFloat64(65504.0)
			}
		}
		if h == fp16.NegZero {
			h = fp16.PosZero
		}
		palette[c] = h
		palF64[c] = fp16.ToFloat64(h)
	}

	codes := make([]byte, len(weights))
	for i, w := range weights {
		bestC := 0
		bestD := math.Abs(w - palF64[0])
		for c := 1; c < k; c++ {
			d := math.Abs(w - palF64[c])
			if d < bestD {
				bestD = d
				bestC = c
			}
		}
		codes[i] = byte(bestC)
	}

	return QuantWeights{
		Format:  fmt.Sprintf("pal%d", bits),
		Codes:   codes,
		Palette: palette,
	}, nil
}

// DequantizeWeights produces hardware-accurate float64 and fp16 dequantized weights,
// accounting for pexp + fp16 scale rounding and exact int4/uint4/palette decoding.
func DequantizeWeights(qw QuantWeights, cout, inner int) ([]float64, []uint16, error) {
	if cout <= 0 || inner <= 0 {
		return nil, nil, fmt.Errorf("invalid dims (cout=%d, inner=%d)", cout, inner)
	}
	norm, bits, isInt, isUnsigned, err := ParseQuantFormat(qw.Format)
	if err != nil {
		return nil, nil, err
	}
	if len(qw.Codes) != cout*inner {
		return nil, nil, fmt.Errorf("quantized codes length %d != cout*inner (%d)", len(qw.Codes), cout*inner)
	}

	outF64 := make([]float64, cout*inner)
	outF16 := make([]uint16, cout*inner)

	if !isInt {
		if len(qw.Scales) != 0 {
			return nil, nil, fmt.Errorf("%s must not specify scales", norm)
		}
		if len(qw.ZeroPoints) != 0 {
			return nil, nil, fmt.Errorf("%s must not specify zero points", norm)
		}
		wantPal := 1 << bits
		if len(qw.Palette) != wantPal {
			return nil, nil, fmt.Errorf("%s palette length %d != %d", norm, len(qw.Palette), wantPal)
		}
		for i, h := range qw.Palette {
			if !fp16.IsFinite(h) {
				return nil, nil, fmt.Errorf("%s palette[%d] (0x%04x) is non-finite", norm, i, h)
			}
		}
		for i, c := range qw.Codes {
			if int(c) >= wantPal {
				return nil, nil, fmt.Errorf("%s code[%d] = %d >= %d", norm, i, c, wantPal)
			}
			h := qw.Palette[c]
			outF16[i] = h
			outF64[i] = fp16.ToFloat64(h)
		}
		return outF64, outF16, nil
	}

	if len(qw.Palette) != 0 {
		return nil, nil, fmt.Errorf("%s must not specify palette", norm)
	}
	if len(qw.Scales) != 1 && len(qw.Scales) != cout {
		return nil, nil, fmt.Errorf("%s scales length %d must be 1 or cout=%d", norm, len(qw.Scales), cout)
	}
	for i, s := range qw.Scales {
		if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) || s <= 0 {
			return nil, nil, fmt.Errorf("%s scale[%d] = %v must be finite positive", norm, i, s)
		}
	}
	if len(qw.ZeroPoints) != 0 && len(qw.ZeroPoints) != cout {
		return nil, nil, fmt.Errorf("%s zero_points length %d != cout=%d", norm, len(qw.ZeroPoints), cout)
	}

	pexp := ComputePexp(qw.Scales)
	effScales := make([]float64, cout)
	for o := 0; o < cout; o++ {
		rawS := float64(qw.Scales[0])
		if len(qw.Scales) > 1 {
			rawS = float64(qw.Scales[o])
		}
		h := fp16.FromFloat64(math.Ldexp(rawS, pexp))
		if h == fp16.NegZero {
			h = fp16.PosZero
		}
		if !fp16.IsFinite(h) || h == 0 || (h&0x8000) != 0 {
			return nil, nil, fmt.Errorf("%s scale[%d]=%g with pexp=%d fails finite positive fp16 encoding (0x%04x)", norm, o, rawS, pexp, h)
		}
		effScales[o] = math.Ldexp(fp16.ToFloat64(h), -pexp)
	}

	for o := 0; o < cout; o++ {
		sEff := effScales[o]
		zp := 0.0
		if len(qw.ZeroPoints) > 0 {
			z := qw.ZeroPoints[o]
			switch norm {
			case "int4":
				if z > 7 && z < 248 {
					return nil, nil, fmt.Errorf("int4 zero_point[%d]=%d outside [-8, 7]", o, z)
				}
				zp = float64(int8(z))
			case "uint4":
				if z > 15 {
					return nil, nil, fmt.Errorf("uint4 zero_point[%d]=%d outside [0, 15]", o, z)
				}
				zp = float64(z)
			case "int8":
				zp = float64(int8(z))
			case "uint8":
				zp = float64(z)
			}
		}

		rowOff := o * inner
		for j := 0; j < inner; j++ {
			c := qw.Codes[rowOff+j]
			var q float64
			switch norm {
			case "int8":
				q = float64(int8(c))
			case "uint8":
				q = float64(c)
			case "int4":
				if c > 15 && c < 248 {
					return nil, nil, fmt.Errorf("int4 code[%d]=%d outside nibble/signed [-8, 7] range", rowOff+j, c)
				}
				nib := c & 0x0f
				if nib >= 8 {
					q = float64(int(nib) - 16)
				} else {
					q = float64(nib)
				}
			case "uint4":
				if c > 15 {
					return nil, nil, fmt.Errorf("uint4 code[%d]=%d outside [0, 15]", rowOff+j, c)
				}
				q = float64(c)
			}
			_ = isUnsigned
			wEff := sEff * (q - zp)
			outF64[rowOff+j] = wEff
			outF16[rowOff+j] = fp16.FromFloat64(wEff)
		}
	}

	return outF64, outF16, nil
}

// effectiveQuantBiases mirrors hardware bias folding + biasShift + post-MAC scale multiplication.
func effectiveQuantBiases(qw QuantWeights, cout int, bias []uint16) ([]float64, error) {
	if len(bias) == 0 {
		return nil, nil
	}
	if len(bias) != cout {
		return nil, fmt.Errorf("bias length %d != cout=%d", len(bias), cout)
	}
	_, _, isInt, _, err := ParseQuantFormat(qw.Format)
	if err != nil {
		return nil, err
	}
	out := make([]float64, cout)
	if !isInt {
		for o, h := range bias {
			out[o] = fp16.ToFloat64(h)
		}
		return out, nil
	}

	pexp := ComputePexp(qw.Scales)
	effScales := make([]float64, cout)
	for o := 0; o < cout; o++ {
		rawS := float64(qw.Scales[0])
		if len(qw.Scales) > 1 {
			rawS = float64(qw.Scales[o])
		}
		h := fp16.FromFloat64(math.Ldexp(rawS, pexp))
		effScales[o] = math.Ldexp(fp16.ToFloat64(h), -pexp)
	}
	bShift := ComputeBiasShift(effScales, bias)
	for o, bh := range bias {
		folded := fp16.FromFloat64(math.Ldexp(fp16.ToFloat64(bh)/effScales[o], -bShift))
		if !fp16.IsFinite(folded) {
			return nil, fmt.Errorf("folded bias[%d] overflows finite fp16", o)
		}
		out[o] = effScales[o] * math.Ldexp(fp16.ToFloat64(folded), bShift)
	}
	return out, nil
}

// EvalConv1x1Quant evaluates a 1x1 convolution with hardware-accurate dequantized weights.
func EvalConv1x1Quant(x Tensor, qw QuantWeights, cout int, bias []uint16, relu bool) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	wf, _, err := DequantizeWeights(qw, cout, x.C)
	if err != nil {
		return Tensor{}, err
	}
	bf, err := effectiveQuantBiases(qw, cout, bias)
	if err != nil {
		return Tensor{}, err
	}

	out, err := NewTensor(x.N, cout, x.H, x.W)
	if err != nil {
		return Tensor{}, err
	}
	hw := x.H * x.W
	xf := make([]float64, len(x.Data))
	for i, h := range x.Data {
		xf[i] = fp16.ToFloat64(h)
	}

	for n := 0; n < x.N; n++ {
		xBaseN := n * x.C * hw
		yBaseN := n * cout * hw
		for o := 0; o < cout; o++ {
			wRow := wf[o*x.C : (o+1)*x.C]
			bVal := 0.0
			if bf != nil {
				bVal = bf[o]
			}
			yBaseO := yBaseN + o*hw
			for p := 0; p < hw; p++ {
				acc := bVal
				for c := 0; c < x.C; c++ {
					acc += xf[xBaseN+c*hw+p] * wRow[c]
				}
				if relu && acc < 0 {
					acc = 0
				}
				out.Data[yBaseO+p] = fp16.FromFloat64(acc)
			}
		}
	}
	return out, nil
}

// EvalLinearQuant evaluates a quantized linear layer [1, K, B, M] -> [1, N, B, M].
func EvalLinearQuant(x Tensor, qw QuantWeights, cout int, bias []uint16, relu bool) (Tensor, error) {
	return EvalConv1x1Quant(x, qw, cout, bias, relu)
}

// EvalConv3x3Quant evaluates a dense 3x3 convolution with hardware-accurate dequantized weights.
func EvalConv3x3Quant(x Tensor, qw QuantWeights, cout int, bias []uint16, stride int, pad string, relu bool) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if stride != 1 {
		return Tensor{}, fmt.Errorf("quantized conv3x3 only supports stride=1, got %d", stride)
	}
	wf, _, err := DequantizeWeights(qw, cout, x.C*9)
	if err != nil {
		return Tensor{}, err
	}
	bf, err := effectiveQuantBiases(qw, cout, bias)
	if err != nil {
		return Tensor{}, err
	}

	var outH, outW, pt, pl int
	switch strings.ToLower(pad) {
	case "", "same":
		outH = (x.H + stride - 1) / stride
		outW = (x.W + stride - 1) / stride
		pt = ((outH-1)*stride + 3 - x.H) / 2
		pl = ((outW-1)*stride + 3 - x.W) / 2
	case "valid":
		if x.H < 3 || x.W < 3 {
			return Tensor{}, fmt.Errorf("valid 3x3 requires H >= 3 and W >= 3, got (%d, %d)", x.H, x.W)
		}
		outH = (x.H-3)/stride + 1
		outW = (x.W-3)/stride + 1
	default:
		return Tensor{}, fmt.Errorf("unsupported conv3x3 pad %q", pad)
	}

	out, err := NewTensor(x.N, cout, outH, outW)
	if err != nil {
		return Tensor{}, err
	}
	xf := make([]float64, len(x.Data))
	for i, h := range x.Data {
		xf[i] = fp16.ToFloat64(h)
	}

	for n := 0; n < x.N; n++ {
		for o := 0; o < cout; o++ {
			bVal := 0.0
			if bf != nil {
				bVal = bf[o]
			}
			for oy := 0; oy < outH; oy++ {
				for ox := 0; ox < outW; ox++ {
					acc := bVal
					for c := 0; c < x.C; c++ {
						wBase := (o*x.C + c) * 9
						for ky := 0; ky < 3; ky++ {
							iy := oy*stride + ky - pt
							if iy < 0 || iy >= x.H {
								continue
							}
							for kx := 0; kx < 3; kx++ {
								ix := ox*stride + kx - pl
								if ix < 0 || ix >= x.W {
									continue
								}
								acc += xf[x.Index(n, c, iy, ix)] * wf[wBase+ky*3+kx]
							}
						}
					}
					if relu && acc < 0 {
						acc = 0
					}
					out.Data[out.Index(n, o, oy, ox)] = fp16.FromFloat64(acc)
				}
			}
		}
	}
	return out, nil
}

// EncodeFloat32LE encodes float32 values into raw little-endian bytes for Vane CLI/graph `--scales`.
func EncodeFloat32LE(vals []float32) []byte {
	buf := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
	}
	return buf
}

// EncodeUint16LE encodes uint16 fp16 values into raw little-endian bytes for Vane CLI/graph `--palette` / `--bias`.
func EncodeUint16LE(vals []uint16) []byte {
	buf := make([]byte, 2*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint16(buf[2*i:], v)
	}
	return buf
}
