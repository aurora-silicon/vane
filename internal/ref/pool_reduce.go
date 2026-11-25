package ref

import (
	"fmt"
	"math"
	"strings"

	"github.com/aurora-silicon/vane/internal/fp16"
)

// PoolSpec describes a 2D spatial pooling operation on NCHW fp16 tensors.
type PoolSpec struct {
	Op         string // "max", "avg", "max_pool", "avg_pool"
	KH, KW     int
	SH, SW     int
	Pad        string // "valid", "same", "custom"
	PT, PB     int
	PL, PR     int
	ExcludePad bool
}

// PoolAxisGeometry validates and resolves one spatial axis of a 2D pooling operation.
func PoolAxisGeometry(inp, k, s int, pad string, pBefore, pAfter int, axis string) (out, padBefore, padAfter int, err error) {
	if inp <= 0 {
		return 0, 0, 0, fmt.Errorf("pool %s: input extent %d must be > 0", axis, inp)
	}
	if k <= 0 {
		return 0, 0, 0, fmt.Errorf("pool %s: kernel size %d must be > 0", axis, k)
	}
	if s <= 0 {
		s = 1
	}
	if s != 1 && s != 2 {
		return 0, 0, 0, fmt.Errorf("pool %s: stride %d must be 1 or 2", axis, s)
	}
	switch strings.ToLower(pad) {
	case "", "valid":
		if inp < k {
			return 0, 0, 0, fmt.Errorf("pool %s (valid): input extent %d < kernel size %d", axis, inp, k)
		}
		return (inp-k)/s + 1, 0, 0, nil
	case "same":
		o := (inp + s - 1) / s
		padTotal := (o-1)*s + k - inp
		if padTotal < 0 {
			padTotal = 0
		}
		pb := padTotal / 2
		pa := padTotal - pb
		return o, pb, pa, nil
	case "custom":
		if pBefore < 0 || pAfter < 0 {
			return 0, 0, 0, fmt.Errorf("pool %s (custom): negative padding (%d, %d)", axis, pBefore, pAfter)
		}
		if pBefore >= k || pAfter >= k {
			return 0, 0, 0, fmt.Errorf("pool %s (custom): padding (%d, %d) must be < kernel %d", axis, pBefore, pAfter, k)
		}
		if inp+pBefore+pAfter < k {
			return 0, 0, 0, fmt.Errorf("pool %s (custom): inp(%d)+pad(%d+%d) < k(%d)", axis, inp, pBefore, pAfter, k)
		}
		o := (inp+pBefore+pAfter-k)/s + 1
		if (o-1)*s >= inp+pBefore {
			return 0, 0, 0, fmt.Errorf("pool %s (custom): last output window does not overlap any real pixel", axis)
		}
		return o, pBefore, pAfter, nil
	default:
		return 0, 0, 0, fmt.Errorf("pool %s: unsupported pad mode %q", axis, pad)
	}
}

// PoolGeometry resolves output spatial dimensions and padding for a 2D pooling op.
func PoolGeometry(h, w int, spec PoolSpec) (oh, ow, pt, pb, pl, pr int, err error) {
	sh, sw := spec.SH, spec.SW
	if sh == 0 {
		sh = 1
	}
	if sw == 0 {
		sw = 1
	}
	oh, pt, pb, err = PoolAxisGeometry(h, spec.KH, sh, spec.Pad, spec.PT, spec.PB, "H")
	if err != nil {
		return 0, 0, 0, 0, 0, 0, err
	}
	ow, pl, pr, err = PoolAxisGeometry(w, spec.KW, sw, spec.Pad, spec.PL, spec.PR, "W")
	if err != nil {
		return 0, 0, 0, 0, 0, 0, err
	}
	return oh, ow, pt, pb, pl, pr, nil
}

// EvalPool evaluates 2D spatial max_pool or avg_pool (include/exclude pad) in float64
// and rounds each output element to IEEE-754 binary16.
func EvalPool(x Tensor, spec PoolSpec) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	op := strings.ToLower(spec.Op)
	var isMax bool
	switch op {
	case "max", "max_pool":
		isMax = true
	case "avg", "avg_pool":
		isMax = false
	default:
		return Tensor{}, fmt.Errorf("unsupported pool op %q (expected max or avg)", spec.Op)
	}
	sh, sw := spec.SH, spec.SW
	if sh == 0 {
		sh = 1
	}
	if sw == 0 {
		sw = 1
	}
	oh, ow, pt, _, pl, _, err := PoolGeometry(x.H, x.W, spec)
	if err != nil {
		return Tensor{}, err
	}
	out, err := NewTensor(x.N, x.C, oh, ow)
	if err != nil {
		return Tensor{}, err
	}
	area := float64(spec.KH * spec.KW)

	for n := 0; n < x.N; n++ {
		for c := 0; c < x.C; c++ {
			for oy := 0; oy < oh; oy++ {
				yStart := oy*sh - pt
				for ox := 0; ox < ow; ox++ {
					xStart := ox*sw - pl
					if isMax {
						best := math.Inf(-1)
						hasVal := false
						for ky := 0; ky < spec.KH; ky++ {
							iy := yStart + ky
							if iy < 0 || iy >= x.H {
								continue
							}
							for kx := 0; kx < spec.KW; kx++ {
								ix := xStart + kx
								if ix < 0 || ix >= x.W {
									continue
								}
								v := fp16.ToFloat64(x.Data[x.Index(n, c, iy, ix)])
								if !hasVal || v > best || math.IsNaN(v) {
									best = v
									hasVal = true
								}
							}
						}
						out.Data[out.Index(n, c, oy, ox)] = fp16.FromFloat64(best)
					} else {
						sum := 0.0
						cnt := 0
						for ky := 0; ky < spec.KH; ky++ {
							iy := yStart + ky
							if iy < 0 || iy >= x.H {
								continue
							}
							for kx := 0; kx < spec.KW; kx++ {
								ix := xStart + kx
								if ix < 0 || ix >= x.W {
									continue
								}
								sum += fp16.ToFloat64(x.Data[x.Index(n, c, iy, ix)])
								cnt++
							}
						}
						denom := area
						if spec.ExcludePad {
							denom = float64(cnt)
						}
						out.Data[out.Index(n, c, oy, ox)] = fp16.FromFloat64(sum / denom)
					}
				}
			}
		}
	}
	return out, nil
}

// ReduceSpec describes a keep_dims=true tensor reduction over any non-empty subset of {C, H, W}.
type ReduceSpec struct {
	Op      string // sum, mean, max, min, l1, sum_square, l2, inv_sum, rsqrt_sum_square, rsqrt_mean_square
	ReduceC bool
	ReduceH bool
	ReduceW bool
	Eps     float64
}

// ParseReduceAxes parses an axis specification string ("w", "h", "c", "hw", "cw", "ch", "chw").
func ParseReduceAxes(s string) (rc, rh, rw bool, err error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "w":
		return false, false, true, nil
	case "h":
		return false, true, false, nil
	case "c":
		return true, false, false, nil
	case "hw", "wh":
		return false, true, true, nil
	case "cw", "wc":
		return true, false, true, nil
	case "ch", "hc":
		return true, true, false, nil
	case "chw", "cwh", "hcw", "hwc", "wch", "whc":
		return true, true, true, nil
	default:
		return false, false, false, fmt.Errorf("invalid reduce axes %q (expected w, h, c, hw, cw, ch, or chw)", s)
	}
}

// EvalReduce evaluates any of the 10 supported tensor reductions (keep_dims=true) in float64
// with fp16-quantized eps (for rsqrt ops), rounding outputs to IEEE-754 binary16.
func EvalReduce(x Tensor, spec ReduceSpec) (Tensor, error) {
	if err := x.Validate(); err != nil {
		return Tensor{}, err
	}
	if !spec.ReduceC && !spec.ReduceH && !spec.ReduceW {
		return Tensor{}, fmt.Errorf("reduce requires at least one reduced axis")
	}
	op := strings.ToLower(spec.Op)
	switch op {
	case "sum", "mean", "max", "min", "l1", "sum_square", "l2", "inv_sum", "rsqrt_sum_square", "rsqrt_mean_square":
	default:
		return Tensor{}, fmt.Errorf("unsupported reduce op %q", spec.Op)
	}

	epsVal := 0.0
	if op == "rsqrt_sum_square" || op == "rsqrt_mean_square" {
		if math.IsNaN(spec.Eps) || math.IsInf(spec.Eps, 0) || spec.Eps < 0 {
			return Tensor{}, fmt.Errorf("reduce %s: eps must be finite and >= 0 (got %g)", op, spec.Eps)
		}
		epsH := fp16.FromFloat64(spec.Eps)
		if !fp16.IsFinite(epsH) {
			return Tensor{}, fmt.Errorf("reduce %s: eps %g overflows fp16 finite range", op, spec.Eps)
		}
		if epsH == fp16.NegZero {
			epsH = fp16.PosZero
		}
		epsVal = fp16.ToFloat64(epsH)
	}

	outC, outH, outW := x.C, x.H, x.W
	if spec.ReduceC {
		outC = 1
	}
	if spec.ReduceH {
		outH = 1
	}
	if spec.ReduceW {
		outW = 1
	}
	out, err := NewTensor(x.N, outC, outH, outW)
	if err != nil {
		return Tensor{}, err
	}

	for n := 0; n < x.N; n++ {
		for oc := 0; oc < outC; oc++ {
			cStart, cEnd := oc, oc+1
			if spec.ReduceC {
				cStart, cEnd = 0, x.C
			}
			for oh := 0; oh < outH; oh++ {
				hStart, hEnd := oh, oh+1
				if spec.ReduceH {
					hStart, hEnd = 0, x.H
				}
				for ow := 0; ow < outW; ow++ {
					wStart, wEnd := ow, ow+1
					if spec.ReduceW {
						wStart, wEnd = 0, x.W
					}
					count := (cEnd - cStart) * (hEnd - hStart) * (wEnd - wStart)
					var res float64
					switch op {
					case "sum", "mean", "inv_sum":
						sum := 0.0
						for c := cStart; c < cEnd; c++ {
							for h := hStart; h < hEnd; h++ {
								for w := wStart; w < wEnd; w++ {
									sum += fp16.ToFloat64(x.Data[x.Index(n, c, h, w)])
								}
							}
						}
						switch op {
						case "sum":
							res = sum
						case "mean":
							res = sum / float64(count)
						case "inv_sum":
							res = 1.0 / sum
						}
					case "max":
						best := math.Inf(-1)
						first := true
						for c := cStart; c < cEnd; c++ {
							for h := hStart; h < hEnd; h++ {
								for w := wStart; w < wEnd; w++ {
									v := fp16.ToFloat64(x.Data[x.Index(n, c, h, w)])
									if first || v > best || math.IsNaN(v) {
										best = v
										first = false
									}
								}
							}
						}
						res = best
					case "min":
						best := math.Inf(1)
						first := true
						for c := cStart; c < cEnd; c++ {
							for h := hStart; h < hEnd; h++ {
								for w := wStart; w < wEnd; w++ {
									v := fp16.ToFloat64(x.Data[x.Index(n, c, h, w)])
									if first || v < best || math.IsNaN(v) {
										best = v
										first = false
									}
								}
							}
						}
						res = best
					case "l1":
						sum := 0.0
						for c := cStart; c < cEnd; c++ {
							for h := hStart; h < hEnd; h++ {
								for w := wStart; w < wEnd; w++ {
									sum += math.Abs(fp16.ToFloat64(x.Data[x.Index(n, c, h, w)]))
								}
							}
						}
						res = sum
					case "sum_square", "l2", "rsqrt_sum_square", "rsqrt_mean_square":
						ss := 0.0
						for c := cStart; c < cEnd; c++ {
							for h := hStart; h < hEnd; h++ {
								for w := wStart; w < wEnd; w++ {
									v := fp16.ToFloat64(x.Data[x.Index(n, c, h, w)])
									ss += v * v
								}
							}
						}
						switch op {
						case "sum_square":
							res = ss
						case "l2":
							res = math.Sqrt(ss)
						case "rsqrt_sum_square":
							res = 1.0 / math.Sqrt(ss+epsVal)
						case "rsqrt_mean_square":
							res = 1.0 / math.Sqrt(ss/float64(count)+epsVal)
						}
					}
					out.Data[out.Index(n, oc, oh, ow)] = fp16.FromFloat64(res)
				}
			}
		}
	}
	return out, nil
}
