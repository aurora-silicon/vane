// Package ref implements independent float64 and Q15.16 reference operators,
// strided fp16 surface packing/unpacking, and numerical comparison metrics.
package ref

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"

	"github.com/niteria/vane/internal/fp16"
)

// Tensor is a contiguous 4D NCHW tensor of IEEE-754 binary16 values.
type Tensor struct {
	N, C, H, W int
	Data       []uint16
}

// NewTensor allocates a zero-initialized NCHW binary16 tensor.
func NewTensor(n, c, h, w int) (Tensor, error) {
	if n <= 0 || c <= 0 || h <= 0 || w <= 0 {
		return Tensor{}, fmt.Errorf("invalid tensor dimensions (%d, %d, %d, %d): all must be > 0", n, c, h, w)
	}
	return Tensor{N: n, C: c, H: h, W: w, Data: make([]uint16, n*c*h*w)}, nil
}

// Validate checks that Tensor dimensions are positive and Data length matches N*C*H*W.
func (t Tensor) Validate() error {
	if t.N <= 0 || t.C <= 0 || t.H <= 0 || t.W <= 0 {
		return fmt.Errorf("invalid tensor shape (%d, %d, %d, %d)", t.N, t.C, t.H, t.W)
	}
	want := t.N * t.C * t.H * t.W
	if len(t.Data) != want {
		return fmt.Errorf("tensor data length %d != N*C*H*W (%d)", len(t.Data), want)
	}
	return nil
}

// Index returns the flat index of (n, c, h, w) in NCHW order.
func (t Tensor) Index(n, c, h, w int) int {
	return ((n*t.C+c)*t.H+h)*t.W + w
}

// AlignUp rounds n up to a multiple of align (when align > 0).
func AlignUp(n, align int) int {
	if align <= 0 {
		return n
	}
	rem := n % align
	if rem == 0 {
		return n
	}
	return n + (align - rem)
}

// PackSurface packs an (N, C, H, W) fp16 tensor into a strided little-endian byte surface.
// Passing 0 for a stride or totalSize selects the minimum tight value.
func PackSurface(t Tensor, batchStride, planeStride, rowStride, totalSize int) ([]byte, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}
	minRow := t.W * 2
	rs := minRow
	if rowStride != 0 {
		rs = rowStride
	}
	if rs < minRow {
		return nil, fmt.Errorf("row_stride %d < width*2 (%d)", rs, minRow)
	}
	minPlane := t.H * rs
	ps := minPlane
	if planeStride != 0 {
		ps = planeStride
	}
	if ps < minPlane {
		return nil, fmt.Errorf("plane_stride %d < height*row_stride (%d)", ps, minPlane)
	}
	minBatch := t.C * ps
	bs := minBatch
	if batchStride != 0 {
		bs = batchStride
	}
	if bs < minBatch {
		return nil, fmt.Errorf("batch_stride %d < channels*plane_stride (%d)", bs, minBatch)
	}
	span := t.N * bs
	size := span
	if totalSize != 0 {
		size = totalSize
	}
	if size < span {
		return nil, fmt.Errorf("total_size %d < span (%d)", size, span)
	}

	buf := make([]byte, size)
	for ni := 0; ni < t.N; ni++ {
		bOff := ni * bs
		for ci := 0; ci < t.C; ci++ {
			pOff := bOff + ci*ps
			for hi := 0; hi < t.H; hi++ {
				rOff := pOff + hi*rs
				baseIdx := t.Index(ni, ci, hi, 0)
				for wi := 0; wi < t.W; wi++ {
					binary.LittleEndian.PutUint16(buf[rOff+2*wi:], t.Data[baseIdx+wi])
				}
			}
		}
	}
	return buf, nil
}

// UnpackSurface unpacks a strided little-endian fp16 byte surface into an (N, C, H, W) Tensor,
// ignoring any row/plane/batch padding bytes beyond the active elements.
func UnpackSurface(data []byte, n, c, h, w, batchStride, planeStride, rowStride int) (Tensor, error) {
	t, err := NewTensor(n, c, h, w)
	if err != nil {
		return Tensor{}, err
	}
	minRow := w * 2
	rs := minRow
	if rowStride != 0 {
		rs = rowStride
	}
	if rs < minRow {
		return Tensor{}, fmt.Errorf("row_stride %d < width*2 (%d)", rs, minRow)
	}
	minPlane := h * rs
	ps := minPlane
	if planeStride != 0 {
		ps = planeStride
	}
	if ps < minPlane {
		return Tensor{}, fmt.Errorf("plane_stride %d < height*row_stride (%d)", ps, minPlane)
	}
	minBatch := c * ps
	bs := minBatch
	if batchStride != 0 {
		bs = batchStride
	}
	if bs < minBatch {
		return Tensor{}, fmt.Errorf("batch_stride %d < channels*plane_stride (%d)", bs, minBatch)
	}
	span := n * bs
	if len(data) < span {
		return Tensor{}, fmt.Errorf("buffer size %d < required span %d", len(data), span)
	}

	for ni := 0; ni < n; ni++ {
		bOff := ni * bs
		for ci := 0; ci < c; ci++ {
			pOff := bOff + ci*ps
			for hi := 0; hi < h; hi++ {
				rOff := pOff + hi*rs
				baseIdx := t.Index(ni, ci, hi, 0)
				for wi := 0; wi < w; wi++ {
					t.Data[baseIdx+wi] = binary.LittleEndian.Uint16(data[rOff+2*wi:])
				}
			}
		}
	}
	return t, nil
}

// Relu evaluates fp16 ReLU in float64 and rounds back to binary16.
func Relu(x Tensor) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	out, _ := NewTensor(x.N, x.C, x.H, x.W)
	for i, v := range x.Data {
		f := fp16.ToFloat64(v)
		if f < 0 {
			f = 0
		}
		out.Data[i] = fp16.FromFloat64(f)
	}
	return out, nil
}

// Conv2d1x1 computes a 1x1 convolution with float64 accumulation and IEEE-754 fp16 rounding.
// weight has shape (cout, x.C) in row-major order; bias is optional with length cout.
func Conv2d1x1(x Tensor, weight []uint16, cout int, bias []uint16, relu bool) (Tensor, error) {
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
	wf := make([]float64, len(weight))
	for i, w := range weight {
		wf[i] = fp16.ToFloat64(w)
	}
	var bf []float64
	if bias != nil {
		bf = make([]float64, cout)
		for o, b := range bias {
			bf[o] = fp16.ToFloat64(b)
		}
	}

	hw := x.H * x.W
	xf := make([]float64, len(x.Data))
	for i, v := range x.Data {
		xf[i] = fp16.ToFloat64(v)
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

// roundToQ1516 rounds v onto the signed 32-bit Q15.16 grid (LSB = 2^-16, ties away from zero).
// Returns (fixed, overflowSign) where overflowSign is 0 if in [-2^31, 2^31-1], +1 for +inf, -1 for -inf.
func roundToQ1516(v float64) (int64, int) {
	if math.IsInf(v, 1) {
		return 0, 1
	}
	if math.IsInf(v, -1) {
		return 0, -1
	}
	if math.IsNaN(v) {
		return 0, 0
	}
	scaled := v * 65536.0
	rounded := math.Round(scaled) // Go math.Round rounds half away from zero
	if rounded >= 2147483648.0 {
		return 0, 1
	}
	if rounded < -2147483648.0 {
		return 0, -1
	}
	return int64(rounded), 0
}

// combineStickyInf combines a new signed infinity (+1 or -1) into the sticky overflow state.
// +inf + -inf cancels to +0 (state 0, acc 0).
func combineStickyInf(curState int, curAcc int64, addSign int) (int, int64) {
	if addSign == 0 {
		return curState, curAcc
	}
	if curState == 0 {
		return addSign, 0
	}
	if curState == addSign {
		return curState, 0
	}
	// Opposite infinities cancel to +0
	return 0, 0
}

const (
	q1516Min int64 = -(1 << 31)
	q1516Max int64 = (1 << 31) - 1
)

// q1516Acc models the NE's 32-bit signed Q15.16 fixed-point accumulator with sticky +-inf overflow.
type q1516Acc struct {
	acc       int64
	stickyInf int // 0=normal, +1=+inf, -1=-inf
}

func (a *q1516Acc) combineInf(addSign int) {
	a.stickyInf, a.acc = combineStickyInf(a.stickyInf, a.acc, addSign)
}

func (a *q1516Acc) addQ(q int64, ov int) {
	if ov != 0 {
		a.combineInf(ov)
	} else if a.stickyInf == 0 {
		a.acc += q
		if a.acc > q1516Max {
			a.stickyInf, a.acc = 1, 0
		} else if a.acc < q1516Min {
			a.stickyInf, a.acc = -1, 0
		}
	}
}

// addProduct accumulates xh * wh on the Q15.16 grid.
// Any non-finite operand (exp == 31, i.e., +-inf or +-NaN) triggers sticky +-inf
// with sign = sign(xh) ^ sign(wh), matching measured H17 hardware behavior.
func (a *q1516Acc) addProduct(xh, wh uint16) {
	if (xh&0x7fff) == 0 || (wh&0x7fff) == 0 {
		return
	}
	if !fp16.IsFinite(xh) || !fp16.IsFinite(wh) {
		sign := 1
		if ((xh ^ wh) & 0x8000) != 0 {
			sign = -1
		}
		a.combineInf(sign)
		return
	}
	if fp16.UnbiasedExp(xh)+fp16.UnbiasedExp(wh) < -16 {
		return
	}
	q, ov := roundToQ1516(fp16.ToFloat64(xh) * fp16.ToFloat64(wh))
	a.addQ(q, ov)
}

// addBias accumulates a single fp16 bias value on the Q15.16 grid.
func (a *q1516Acc) addBias(bh uint16) {
	if (bh & 0x7fff) == 0 {
		return
	}
	if !fp16.IsFinite(bh) {
		sign := 1
		if (bh & 0x8000) != 0 {
			sign = -1
		}
		a.combineInf(sign)
		return
	}
	q, ov := roundToQ1516(fp16.ToFloat64(bh))
	a.addQ(q, ov)
}

// finalize applies optional ReLU and rounds the Q15.16 accumulator to fp16 (ties away, no -0.0).
func (a *q1516Acc) finalize(relu bool) uint16 {
	if a.stickyInf > 0 {
		return fp16.PosInf
	}
	if a.stickyInf < 0 {
		if relu {
			return fp16.PosZero
		}
		return fp16.NegInf
	}
	v := a.acc
	if relu && v < 0 {
		v = 0
	}
	if v == 0 {
		return fp16.PosZero
	}
	res := fp16.FromFloat64TiesAway(float64(v) / 65536.0)
	if res == fp16.NegZero {
		res = fp16.PosZero
	}
	return res
}

// Conv2d1x1Q1516 evaluates the exact Q15.16 fixed-point NE 1x1 convolution arithmetic model:
//   - 32-bit signed Q15.16 accumulator (LSB = 2^-16)
//   - Each product x*w rounds to 2^-16 (ties away from zero), flushing to 0 when unbiased exp sum < -16
//   - Optional bias joins on the 2^-16 grid
//   - Non-finite (exp == 31) operands and overflows trigger sticky +-inf (+inf + -inf -> +0)
//   - Optional ReLU clamps negatives to +0; final output rounds to fp16 ties-away-from-zero with no -0.0.
func Conv2d1x1Q1516(x Tensor, weight []uint16, cout int, bias []uint16, relu bool) (Tensor, error) {
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
			wRow := weight[o*x.C : (o+1)*x.C]
			yBaseO := yBaseN + o*hw
			for p := 0; p < hw; p++ {
				var acc q1516Acc
				for c := 0; c < x.C; c++ {
					acc.addProduct(x.Data[xBaseN+c*hw+p], wRow[c])
				}
				if bias != nil {
					acc.addBias(bias[o])
				}
				out.Data[yBaseO+p] = acc.finalize(relu)
			}
		}
	}
	return out, nil
}

// Conv2d3x3 computes a dense, grouped, or depthwise 3x3 convolution in float64 -> fp16.
// weight has shape (cout, x.C/groups, 3, 3) in row-major order.
func Conv2d3x3(x Tensor, weight []uint16, cout int, bias []uint16, strideH, strideW, padH, padW, groups int, relu bool) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if cout <= 0 {
		return Tensor{}, fmt.Errorf("cout must be > 0, got %d", cout)
	}
	if strideH <= 0 || strideW <= 0 {
		return Tensor{}, fmt.Errorf("strides must be > 0, got (%d, %d)", strideH, strideW)
	}
	if padH < 0 || padW < 0 {
		return Tensor{}, fmt.Errorf("padding must be >= 0, got (%d, %d)", padH, padW)
	}
	if groups <= 0 {
		return Tensor{}, fmt.Errorf("groups must be > 0, got %d", groups)
	}
	if x.C%groups != 0 || cout%groups != 0 {
		return Tensor{}, fmt.Errorf("Cin=%d and Cout=%d must be divisible by groups=%d", x.C, cout, groups)
	}
	cinPerG := x.C / groups
	coutPerG := cout / groups
	if len(weight) != cout*cinPerG*9 {
		return Tensor{}, fmt.Errorf("weight length %d != cout*(cin/groups)*9 (%d)", len(weight), cout*cinPerG*9)
	}
	if bias != nil && len(bias) != cout {
		return Tensor{}, fmt.Errorf("bias length %d != cout (%d)", len(bias), cout)
	}
	hp := x.H + 2*padH
	wp := x.W + 2*padW
	if hp < 3 || wp < 3 {
		return Tensor{}, fmt.Errorf("padded spatial size (%d, %d) smaller than 3x3 kernel", hp, wp)
	}
	hout := (hp-3)/strideH + 1
	wout := (wp-3)/strideW + 1

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
								ih := oh*strideH + kh - padH
								if ih < 0 || ih >= x.H {
									continue
								}
								for kw := 0; kw < 3; kw++ {
									iw := ow*strideW + kw - padW
									if iw < 0 || iw >= x.W {
										continue
									}
									xv := fp16.ToFloat64(x.Data[x.Index(n, c, ih, iw)])
									wv := fp16.ToFloat64(weight[wBase+kh*3+kw])
									acc += xv * wv
								}
							}
						}
						if relu && acc < 0 {
							acc = 0
						}
						out.Data[out.Index(n, o, oh, ow)] = fp16.FromFloat64(acc)
					}
				}
			}
		}
	}
	return out, nil
}

// Conv3x3Geometry computes output spatial dimensions (ho, wo) and top/left padding
// (padTop, padLeft) for a 3x3 convolution with dilation 1, stride in {1, 2},
// and pad in {"same", "valid"}.
func Conv3x3Geometry(h, w, stride int, pad string) (ho, wo, padTop, padLeft int, err error) {
	if h <= 0 || w <= 0 {
		return 0, 0, 0, 0, fmt.Errorf("invalid spatial dims (%d, %d): must be > 0", h, w)
	}
	if stride != 1 && stride != 2 {
		return 0, 0, 0, 0, fmt.Errorf("unsupported stride %d (expected 1 or 2)", stride)
	}
	switch pad {
	case "same":
		ho = (h + stride - 1) / stride
		wo = (w + stride - 1) / stride
		padTop = ((ho-1)*stride + 3 - h) / 2
		padLeft = ((wo-1)*stride + 3 - w) / 2
		return ho, wo, padTop, padLeft, nil
	case "valid":
		if h < 3 || w < 3 {
			return 0, 0, 0, 0, fmt.Errorf("valid-padded 3x3 requires H >= 3 and W >= 3, got (%d, %d)", h, w)
		}
		ho = (h-3)/stride + 1
		wo = (w-3)/stride + 1
		return ho, wo, 0, 0, nil
	default:
		return 0, 0, 0, 0, fmt.Errorf("unsupported pad mode %q (expected same or valid)", pad)
	}
}

// Conv2d3x3Pad computes a dense, grouped, or depthwise 3x3 convolution in float64 -> fp16
// with asymmetric same/valid padding and stride in {1, 2}.
// weight has shape (cout, x.C/groups, 3, 3) in row-major order.
func Conv2d3x3Pad(x Tensor, weight []uint16, cout int, bias []uint16, stride int, pad string, groups int, relu bool) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if cout <= 0 {
		return Tensor{}, fmt.Errorf("cout must be > 0, got %d", cout)
	}
	if groups <= 0 {
		return Tensor{}, fmt.Errorf("groups must be > 0, got %d", groups)
	}
	if x.C%groups != 0 || cout%groups != 0 {
		return Tensor{}, fmt.Errorf("Cin=%d and Cout=%d must be divisible by groups=%d", x.C, cout, groups)
	}
	cinPerG := x.C / groups
	coutPerG := cout / groups
	if len(weight) != cout*cinPerG*9 {
		return Tensor{}, fmt.Errorf("weight length %d != cout*(cin/groups)*9 (%d)", len(weight), cout*cinPerG*9)
	}
	if bias != nil && len(bias) != cout {
		return Tensor{}, fmt.Errorf("bias length %d != cout (%d)", len(bias), cout)
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
									xv := fp16.ToFloat64(x.Data[x.Index(n, c, ih, iw)])
									wv := fp16.ToFloat64(weight[wBase+kh*3+kw])
									acc += xv * wv
								}
							}
						}
						if relu && acc < 0 {
							acc = 0
						}
						out.Data[out.Index(n, o, oh, ow)] = fp16.FromFloat64(acc)
					}
				}
			}
		}
	}
	return out, nil
}

// Conv2d3x3Q1516 evaluates the exact Q15.16 fixed-point NE 3x3 convolution arithmetic model
// for dense, grouped, or depthwise convolutions with same/valid padding and stride in {1, 2}.
func Conv2d3x3Q1516(x Tensor, weight []uint16, cout int, bias []uint16, stride int, pad string, groups int, relu bool) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if cout <= 0 {
		return Tensor{}, fmt.Errorf("cout must be > 0, got %d", cout)
	}
	if groups <= 0 {
		return Tensor{}, fmt.Errorf("groups must be > 0, got %d", groups)
	}
	if x.C%groups != 0 || cout%groups != 0 {
		return Tensor{}, fmt.Errorf("Cin=%d and Cout=%d must be divisible by groups=%d", x.C, cout, groups)
	}
	cinPerG := x.C / groups
	coutPerG := cout / groups
	if len(weight) != cout*cinPerG*9 {
		return Tensor{}, fmt.Errorf("weight length %d != cout*(cin/groups)*9 (%d)", len(weight), cout*cinPerG*9)
	}
	if bias != nil && len(bias) != cout {
		return Tensor{}, fmt.Errorf("bias length %d != cout (%d)", len(bias), cout)
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
						if bias != nil {
							acc.addBias(bias[o])
						}
						out.Data[out.Index(n, o, oh, ow)] = acc.finalize(relu)
					}
				}
			}
		}
	}
	return out, nil
}

// Elementwise evaluates a two-source elementwise op (add, sub, mul, max, min) in float64 -> fp16.
func Elementwise(op string, a, b Tensor) (Tensor, error) {
	if err := a.Validate(); err != nil {
		return Tensor{}, err
	}
	if err := b.Validate(); err != nil {
		return Tensor{}, err
	}
	if a.N != b.N || a.C != b.C || a.H != b.H || a.W != b.W {
		return Tensor{}, fmt.Errorf("elementwise shape mismatch: (%d,%d,%d,%d) vs (%d,%d,%d,%d)",
			a.N, a.C, a.H, a.W, b.N, b.C, b.H, b.W)
	}
	out, _ := NewTensor(a.N, a.C, a.H, a.W)
	for i := range a.Data {
		af := fp16.ToFloat64(a.Data[i])
		bf := fp16.ToFloat64(b.Data[i])
		var y float64
		switch op {
		case "add":
			y = af + bf
		case "sub":
			y = af - bf
		case "mul":
			y = af * bf
		case "max":
			if math.IsNaN(af) || math.IsNaN(bf) {
				y = math.NaN()
			} else if af >= bf {
				y = af
			} else {
				y = bf
			}
		case "min":
			if math.IsNaN(af) || math.IsNaN(bf) {
				y = math.NaN()
			} else if af <= bf {
				y = af
			} else {
				y = bf
			}
		default:
			return Tensor{}, fmt.Errorf("unsupported elementwise op %q (expected add, sub, mul, max, min)", op)
		}
		out.Data[i] = fp16.FromFloat64(y)
	}
	return out, nil
}

// Affine computes `x * scale + bias` (with optional ReLU) in float64 -> fp16.
// scale and bias may each have length 1 (scalar), length x.C (per-channel), or length len(x.Data).
func Affine(x Tensor, scale, bias []uint16, relu bool) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	validLen := func(arr []uint16, name string) error {
		if len(arr) != 1 && len(arr) != x.C && len(arr) != len(x.Data) {
			return fmt.Errorf("affine %s length %d must be 1, C=%d, or total=%d", name, len(arr), x.C, len(x.Data))
		}
		return nil
	}
	if err := validLen(scale, "scale"); err != nil {
		return Tensor{}, err
	}
	if err := validLen(bias, "bias"); err != nil {
		return Tensor{}, err
	}

	out, _ := NewTensor(x.N, x.C, x.H, x.W)
	hw := x.H * x.W
	for n := 0; n < x.N; n++ {
		for c := 0; c < x.C; c++ {
			for p := 0; p < hw; p++ {
				idx := (n*x.C+c)*hw + p
				xv := fp16.ToFloat64(x.Data[idx])
				var sv, bv float64
				switch len(scale) {
				case 1:
					sv = fp16.ToFloat64(scale[0])
				case x.C:
					sv = fp16.ToFloat64(scale[c])
				default:
					sv = fp16.ToFloat64(scale[idx])
				}
				switch len(bias) {
				case 1:
					bv = fp16.ToFloat64(bias[0])
				case x.C:
					bv = fp16.ToFloat64(bias[c])
				default:
					bv = fp16.ToFloat64(bias[idx])
				}
				y := xv*sv + bv
				if relu && y < 0 {
					y = 0
				}
				out.Data[idx] = fp16.FromFloat64(y)
			}
		}
	}
	return out, nil
}

// ComparisonResult summarizes elementwise comparison metrics between two fp16 tensors.
type ComparisonResult struct {
	MaxAbsErr           float64
	MaxRelErr           float64
	RMSE                float64
	ExactFP16Count      int
	TotalElements       int
	MaxULPErr           int
	NonfiniteMismatches int
	Passed              bool
}

func jsonFloat(v float64) any {
	if math.IsNaN(v) {
		return "NaN"
	}
	if math.IsInf(v, 1) {
		return "Infinity"
	}
	if math.IsInf(v, -1) {
		return "-Infinity"
	}
	return v
}

// MarshalJSON encodes ComparisonResult with safe handling of non-finite error ratios.
func (c ComparisonResult) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]any{
		"max_abs_err":          jsonFloat(c.MaxAbsErr),
		"max_rel_err":          jsonFloat(c.MaxRelErr),
		"rmse":                 jsonFloat(c.RMSE),
		"exact_fp16_count":     c.ExactFP16Count,
		"total_elements":       c.TotalElements,
		"max_ulp_err":          c.MaxULPErr,
		"nonfinite_mismatches": c.NonfiniteMismatches,
		"passed":               c.Passed,
	})
}

// CompareFP16 compares two slices of IEEE-754 binary16 values with abs/rel/RMSE/ULP/non-finite checks.
func CompareFP16(actual, expected []uint16, atol, rtol float64, maxULP *int) (ComparisonResult, error) {
	if len(actual) != len(expected) {
		return ComparisonResult{}, fmt.Errorf("element count mismatch: actual %d != expected %d", len(actual), len(expected))
	}
	total := len(actual)
	if total == 0 {
		return ComparisonResult{Passed: true}, nil
	}

	var maxAbs, maxRel, sqErrSum float64
	var finiteCount, exactCount, maxULPErr, nonfiniteMismatches int
	tolOK := true

	for i := 0; i < total; i++ {
		ah := actual[i]
		eh := expected[i]
		aNaN := fp16.IsNaN(ah)
		eNaN := fp16.IsNaN(eh)
		if aNaN || eNaN {
			if aNaN && eNaN {
				exactCount++
			} else {
				nonfiniteMismatches++
			}
			continue
		}
		aInf := fp16.IsInf(ah, 0)
		eInf := fp16.IsInf(eh, 0)
		if aInf || eInf {
			if aInf && eInf && (ah&0x8000) == (eh&0x8000) {
				exactCount++
			} else {
				nonfiniteMismatches++
			}
			continue
		}

		// Both finite
		finiteCount++
		af := fp16.ToFloat64(ah)
		ef := fp16.ToFloat64(eh)
		if af == ef {
			exactCount++
		}
		diff := math.Abs(af - ef)
		if diff > maxAbs {
			maxAbs = diff
		}
		sqErrSum += diff * diff

		denom := math.Abs(ef)
		var rel float64
		if diff == 0 {
			rel = 0
		} else if denom > 0 {
			rel = diff / denom
		} else {
			rel = math.Inf(1)
		}
		if rel > maxRel {
			maxRel = rel
		}

		ulp := int(fp16.ULPDistance(ah, eh))
		if ulp > maxULPErr {
			maxULPErr = ulp
		}

		if diff > atol+rtol*denom {
			tolOK = false
		}
	}

	rmse := 0.0
	if finiteCount > 0 {
		rmse = math.Sqrt(sqErrSum / float64(finiteCount))
	}
	ulpOK := true
	if maxULP != nil && maxULPErr > *maxULP {
		ulpOK = false
	}

	return ComparisonResult{
		MaxAbsErr:           maxAbs,
		MaxRelErr:           maxRel,
		RMSE:                rmse,
		ExactFP16Count:      exactCount,
		TotalElements:       total,
		MaxULPErr:           maxULPErr,
		NonfiniteMismatches: nonfiniteMismatches,
		Passed:              nonfiniteMismatches == 0 && tolOK && ulpOK,
	}, nil
}

// CompareFP16Bytes compares two raw little-endian fp16 byte slices.
func CompareFP16Bytes(actual, expected []byte, atol, rtol float64, maxULP *int) (ComparisonResult, error) {
	if len(actual) != len(expected) {
		return ComparisonResult{}, fmt.Errorf("byte length mismatch: actual %d != expected %d", len(actual), len(expected))
	}
	aVals, err := fp16.DecodeLE(actual)
	if err != nil {
		return ComparisonResult{}, err
	}
	eVals, err := fp16.DecodeLE(expected)
	if err != nil {
		return ComparisonResult{}, err
	}
	return CompareFP16(aVals, eVals, atol, rtol, maxULP)
}
