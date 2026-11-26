package ref

import (
	"fmt"
	"math"

	"github.com/aurora-silicon/vane/internal/fp16"
)

// MatmulSpec configures a batched or multi-head dynamic matrix multiplication.
// Heads == 0 defaults to 1. Scale == nil defaults to 1.0.
type MatmulSpec struct {
	Heads      int
	TransposeA bool
	TransposeB bool
	Scale      *float64
}

func resolveMatmulGeometry(a, b Tensor, heads int, transposeA, transposeB bool) (m, k, n, aC, bC int, err error) {
	if err := a.Validate(); err != nil {
		return 0, 0, 0, 0, 0, fmt.Errorf("invalid matmul input A: %w", err)
	}
	if err := b.Validate(); err != nil {
		return 0, 0, 0, 0, 0, fmt.Errorf("invalid matmul input B: %w", err)
	}
	if a.N != b.N {
		return 0, 0, 0, 0, 0, fmt.Errorf("matmul batch mismatch: A.N=%d vs B.N=%d", a.N, b.N)
	}
	if a.H != 1 || b.H != 1 {
		return 0, 0, 0, 0, 0, fmt.Errorf("matmul requires H=1 for both operands, got A.H=%d B.H=%d", a.H, b.H)
	}
	if heads < 1 {
		return 0, 0, 0, 0, 0, fmt.Errorf("matmul heads must be >= 1, got %d", heads)
	}
	if a.C%heads != 0 {
		return 0, 0, 0, 0, 0, fmt.Errorf("matmul A.C=%d is not divisible by heads=%d", a.C, heads)
	}
	if b.C%heads != 0 {
		return 0, 0, 0, 0, 0, fmt.Errorf("matmul B.C=%d is not divisible by heads=%d", b.C, heads)
	}
	aC = a.C / heads
	bC = b.C / heads
	var kb int
	if transposeA {
		k = aC
		m = a.W
	} else {
		m = aC
		k = a.W
	}
	if transposeB {
		n = bC
		kb = b.W
	} else {
		kb = bC
		n = b.W
	}
	if k != kb {
		return 0, 0, 0, 0, 0, fmt.Errorf("matmul inner dimension mismatch: A has K=%d, B has K=%d", k, kb)
	}
	return m, k, n, aC, bC, nil
}

// EvalMatmulF64 evaluates batched / multi-head matrix multiplication Y = scale * (op(A) @ op(B))
// in float64 precision and rounds outputs to IEEE-754 binary16.
//
// Supports both standalone batched tensors (A:[B,Ma,1,Ka], B:[B,Kb,1,Nb], heads=1 -> [B,M,1,N])
// and Graph IR stacked-head tensors (A:[1,heads*Ma,1,Ka], B:[1,heads*Kb,1,Nb] -> [1,heads*M,1,N]).
func EvalMatmulF64(a, b Tensor, heads int, transposeA, transposeB bool, scale float64) (Tensor, error) {
	if math.IsNaN(scale) || math.IsInf(scale, 0) {
		return Tensor{}, fmt.Errorf("matmul scale must be finite, got %v", scale)
	}
	m, k, n, aC, bC, err := resolveMatmulGeometry(a, b, heads, transposeA, transposeB)
	if err != nil {
		return Tensor{}, err
	}

	out, err := NewTensor(a.N, heads*m, 1, n)
	if err != nil {
		return Tensor{}, err
	}

	af := make([]float64, len(a.Data))
	for i, v := range a.Data {
		af[i] = fp16.ToFloat64(v)
	}
	bf := make([]float64, len(b.Data))
	for i, v := range b.Data {
		bf[i] = fp16.ToFloat64(v)
	}

	for batch := 0; batch < a.N; batch++ {
		for h := 0; h < heads; h++ {
			aHead := h * aC
			bHead := h * bC
			oHead := h * m
			for mi := 0; mi < m; mi++ {
				for ni := 0; ni < n; ni++ {
					acc := 0.0
					for ki := 0; ki < k; ki++ {
						var aIdx, bIdx int
						if transposeA {
							aIdx = a.Index(batch, aHead+ki, 0, mi)
						} else {
							aIdx = a.Index(batch, aHead+mi, 0, ki)
						}
						if transposeB {
							bIdx = b.Index(batch, bHead+ni, 0, ki)
						} else {
							bIdx = b.Index(batch, bHead+ki, 0, ni)
						}
						acc += af[aIdx] * bf[bIdx]
					}
					out.Data[out.Index(batch, oHead+mi, 0, ni)] = fp16.FromFloat64(acc * scale)
				}
			}
		}
	}
	return out, nil
}

// finalizeScaledQ1516 applies an exact finite fp16 post-accumulation scale to a Q15.16 accumulator
// and rounds to binary16 using ties-away-from-zero with canonical +0.0.
func finalizeScaledQ1516(acc *q1516Acc, scaleBits uint16) uint16 {
	if scaleBits == 0x3c00 {
		return acc.finalize(false)
	}
	sf := fp16.ToFloat64(scaleBits)
	if acc.stickyInf != 0 {
		if sf == 0.0 {
			return fp16.PosZero
		}
		if (acc.stickyInf > 0) == (sf > 0) {
			return fp16.PosInf
		}
		return fp16.NegInf
	}
	if acc.acc == 0 || sf == 0.0 {
		return fp16.PosZero
	}
	return fp16.RNA16((float64(acc.acc) / 65536.0) * sf)
}

// EvalMatmulQ1516 evaluates batched / multi-head dynamic matrix multiplication using the
// NE Q15.16 fixed-point accumulator model with finite fp16 output scaling (`neScale`).
func EvalMatmulQ1516(a, b Tensor, heads int, transposeA, transposeB bool, scaleBits uint16) (Tensor, error) {
	if !fp16.IsFinite(scaleBits) {
		return Tensor{}, fmt.Errorf("matmul scale fp16 bits 0x%04x must be finite", scaleBits)
	}
	m, k, n, aC, bC, err := resolveMatmulGeometry(a, b, heads, transposeA, transposeB)
	if err != nil {
		return Tensor{}, err
	}

	out, err := NewTensor(a.N, heads*m, 1, n)
	if err != nil {
		return Tensor{}, err
	}

	for batch := 0; batch < a.N; batch++ {
		for h := 0; h < heads; h++ {
			aHead := h * aC
			bHead := h * bC
			oHead := h * m
			for mi := 0; mi < m; mi++ {
				for ni := 0; ni < n; ni++ {
					var acc q1516Acc
					for ki := 0; ki < k; ki++ {
						var aIdx, bIdx int
						if transposeA {
							aIdx = a.Index(batch, aHead+ki, 0, mi)
						} else {
							aIdx = a.Index(batch, aHead+mi, 0, ki)
						}
						if transposeB {
							bIdx = b.Index(batch, bHead+ni, 0, ki)
						} else {
							bIdx = b.Index(batch, bHead+ki, 0, ni)
						}
						acc.addProduct(a.Data[aIdx], b.Data[bIdx])
					}
					out.Data[out.Index(batch, oHead+mi, 0, ni)] = finalizeScaledQ1516(&acc, scaleBits)
				}
			}
		}
	}
	return out, nil
}

// EvalMatmul evaluates a MatmulSpec using either float64 (exactHW=false) or Q15.16 (exactHW=true).
func EvalMatmul(a, b Tensor, spec MatmulSpec, exactHW bool) (Tensor, error) {
	heads := spec.Heads
	if heads == 0 {
		heads = 1
	}
	scale := 1.0
	if spec.Scale != nil {
		scale = *spec.Scale
	}
	if math.IsNaN(scale) || math.IsInf(scale, 0) {
		return Tensor{}, fmt.Errorf("matmul scale must be finite, got %v", scale)
	}
	if exactHW {
		scaleBits := fp16.FromFloat64(scale)
		if !fp16.IsFinite(scaleBits) {
			return Tensor{}, fmt.Errorf("matmul scale %v overflows fp16", scale)
		}
		return EvalMatmulQ1516(a, b, heads, spec.TransposeA, spec.TransposeB, scaleBits)
	}
	return EvalMatmulF64(a, b, heads, spec.TransposeA, spec.TransposeB, scale)
}
