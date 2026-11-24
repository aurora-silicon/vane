package ref

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/niteria/vane/internal/fp16"
)

func toF16Slice(vals ...float64) []uint16 {
	out := make([]uint16, len(vals))
	for i, v := range vals {
		out[i] = fp16.FromFloat64(v)
	}
	return out
}

func TestCompareFP16MetricsAndULP(t *testing.T) {
	zeroULP := 0
	a := []uint16{fp16.FromFloat64(1.0), fp16.NegZero, fp16.FromFloat64(2.0), fp16.PosInf, fp16.QuietNaN}
	b := []uint16{fp16.FromFloat64(1.0), fp16.PosZero, fp16.FromFloat64(2.0), fp16.PosInf, fp16.QuietNaN}
	res, err := CompareFP16(a, b, 0, 0, &zeroULP)
	if err != nil {
		t.Fatalf("CompareFP16 failed: %v", err)
	}
	if !res.Passed || res.MaxULPErr != 0 || res.NonfiniteMismatches != 0 || res.ExactFP16Count != 5 {
		t.Fatalf("unexpected comparison result: %+v", res)
	}

	// 1 ULP difference on 1.0 (0x3c00 vs 0x3c01)
	oneULP := 1
	resULP, err := CompareFP16([]uint16{0x3c01}, []uint16{0x3c00}, 1e-2, 1e-2, &oneULP)
	if err != nil {
		t.Fatalf("CompareFP16 failed: %v", err)
	}
	if !resULP.Passed || resULP.MaxULPErr != 1 || resULP.ExactFP16Count != 0 {
		t.Fatalf("unexpected 1-ULP result: %+v", resULP)
	}
	resFail, _ := CompareFP16([]uint16{0x3c01}, []uint16{0x3c00}, 1e-2, 1e-2, &zeroULP)
	if resFail.Passed {
		t.Fatalf("expected max_ulp=0 failure")
	}

	// Non-finite mismatch fails
	resNF, _ := CompareFP16([]uint16{fp16.QuietNaN}, []uint16{0x3c00}, 1e-2, 1e-2, nil)
	if resNF.Passed || resNF.NonfiniteMismatches != 1 {
		t.Fatalf("expected nonfinite mismatch, got %+v", resNF)
	}

	// JSON marshaling succeeds even when MaxRelErr == +Inf (expected=0, actual!=0)
	resInfRel, _ := CompareFP16([]uint16{0x3c00}, []uint16{0x0000}, 2.0, 0.0, nil)
	if !math.IsInf(resInfRel.MaxRelErr, 1) {
		t.Fatalf("expected +Inf MaxRelErr, got %v", resInfRel.MaxRelErr)
	}
	if _, err := json.Marshal(resInfRel); err != nil {
		t.Fatalf("json.Marshal failed on +Inf MaxRelErr: %v", err)
	}
}

func TestStridedSurfacePackUnpack(t *testing.T) {
	rng := rand.New(rand.NewPCG(123, 456))
	orig, _ := NewTensor(2, 3, 5, 7)
	for i := range orig.Data {
		orig.Data[i] = fp16.FromFloat64(rng.Float64()*4.0 - 2.0)
	}
	rowStride := AlignUp(7*2, 64) // 64
	planeStride := 5 * rowStride
	batchStride := 3 * planeStride
	totalSize := 2*batchStride + 128

	packed, err := PackSurface(orig, batchStride, planeStride, rowStride, totalSize)
	if err != nil {
		t.Fatalf("PackSurface failed: %v", err)
	}
	if len(packed) != totalSize {
		t.Fatalf("packed len %d != %d", len(packed), totalSize)
	}
	// Poison padding bytes after 2*W in each row to ensure UnpackSurface ignores them
	for n := 0; n < 2; n++ {
		for c := 0; c < 3; c++ {
			for h := 0; h < 5; h++ {
				off := n*batchStride + c*planeStride + h*rowStride
				for p := 7 * 2; p < rowStride; p++ {
					packed[off+p] = 0xff
				}
			}
		}
	}

	unpacked, err := UnpackSurface(packed, 2, 3, 5, 7, batchStride, planeStride, rowStride)
	if err != nil {
		t.Fatalf("UnpackSurface failed: %v", err)
	}
	for i := range orig.Data {
		if unpacked.Data[i] != orig.Data[i] {
			t.Fatalf("mismatch at %d: got 0x%04x, want 0x%04x", i, unpacked.Data[i], orig.Data[i])
		}
	}
}

func TestConv2d1x1AndQ1516(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 99))
	x, _ := NewTensor(1, 4, 3, 5)
	for i := range x.Data {
		x.Data[i] = fp16.FromFloat64(rng.Float64()*2.0 - 1.0)
	}
	w := make([]uint16, 6*4)
	for i := range w {
		w[i] = fp16.FromFloat64(rng.Float64() - 0.5)
	}
	b := make([]uint16, 6)
	for i := range b {
		b[i] = fp16.FromFloat64(rng.Float64() - 0.5)
	}

	y64, err := Conv2d1x1(x, w, 6, b, true)
	if err != nil {
		t.Fatalf("Conv2d1x1 failed: %v", err)
	}
	yQ, err := Conv2d1x1Q1516(x, w, 6, b, true)
	if err != nil {
		t.Fatalf("Conv2d1x1Q1516 failed: %v", err)
	}

	// Verify float64 reference directly
	for o := 0; o < 6; o++ {
		for h := 0; h < 3; h++ {
			for wi := 0; wi < 5; wi++ {
				acc := fp16.ToFloat64(b[o])
				for c := 0; c < 4; c++ {
					acc += fp16.ToFloat64(x.Data[x.Index(0, c, h, wi)]) * fp16.ToFloat64(w[o*4+c])
				}
				if acc < 0 {
					acc = 0
				}
				want := fp16.FromFloat64(acc)
				if got := y64.Data[y64.Index(0, o, h, wi)]; got != want {
					t.Fatalf("Conv2d1x1 mismatch at (%d,%d,%d): got 0x%04x want 0x%04x", o, h, wi, got, want)
				}
			}
		}
	}

	// Q15.16 and float64 should agree closely on well-scaled inputs
	maxUlp := 2
	comp, err := CompareFP16(yQ.Data, y64.Data, 1e-3, 1e-3, &maxUlp)
	if err != nil {
		t.Fatalf("CompareFP16 failed: %v", err)
	}
	if !comp.Passed {
		t.Fatalf("Q15.16 vs float64 diverged unexpectedly: %+v", comp)
	}

	// Verify Q15.16 edge semantics:
	// 1. Flush to zero when sum of unbiased exponents < -16 (e.g. 2^-9 * 2^-8 = 2^-17 -> exp sum -17 < -16)
	xSmall := Tensor{N: 1, C: 1, H: 1, W: 1, Data: toF16Slice(math.Ldexp(1.0, -9))}
	wSmall := toF16Slice(math.Ldexp(1.0, -8))
	ySmall, _ := Conv2d1x1Q1516(xSmall, wSmall, 1, nil, false)
	if ySmall.Data[0] != fp16.PosZero {
		t.Fatalf("expected Q15.16 flush to +0 for exp_sum=-17, got 0x%04x", ySmall.Data[0])
	}
	// 2. Product with exp_sum == -16 (2^-8 * 2^-8 = 2^-16) is NOT flushed
	xExact := Tensor{N: 1, C: 1, H: 1, W: 1, Data: toF16Slice(math.Ldexp(1.0, -8))}
	wExact := toF16Slice(math.Ldexp(1.0, -8))
	yExact, _ := Conv2d1x1Q1516(xExact, wExact, 1, nil, false)
	if yExact.Data[0] != fp16.FromFloat64(math.Ldexp(1.0, -16)) {
		t.Fatalf("expected 2^-16 preserved at exp_sum=-16, got 0x%04x", yExact.Data[0])
	}
	// 3. Overflow +inf + -inf cancels to +0
	xOv := Tensor{N: 1, C: 2, H: 1, W: 1, Data: toF16Slice(256.0, 256.0)}
	wOv := toF16Slice(256.0, -256.0) // +65536 (>= 2^15 -> +inf), then -65536 (-inf) -> cancels to +0
	yOv, _ := Conv2d1x1Q1516(xOv, wOv, 1, nil, false)
	if yOv.Data[0] != fp16.PosZero {
		t.Fatalf("expected +inf + -inf -> +0, got 0x%04x", yOv.Data[0])
	}
}

func TestConv2d3x3DenseAndDepthwise(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 22))
	x, _ := NewTensor(1, 3, 5, 5)
	for i := range x.Data {
		x.Data[i] = fp16.FromFloat64(rng.Float64()*2.0 - 1.0)
	}
	wDense := make([]uint16, 4*3*9)
	for i := range wDense {
		wDense[i] = fp16.FromFloat64(rng.Float64() - 0.5)
	}
	bDense := make([]uint16, 4)
	for i := range bDense {
		bDense[i] = fp16.FromFloat64(rng.Float64() - 0.5)
	}

	yDense, err := Conv2d3x3(x, wDense, 4, bDense, 2, 2, 1, 1, 1, false)
	if err != nil {
		t.Fatalf("Conv2d3x3 dense failed: %v", err)
	}
	if yDense.H != 3 || yDense.W != 3 || yDense.C != 4 {
		t.Fatalf("unexpected dense shape: %+v", yDense)
	}
	// Manual check at (0, 2, 1, 1) -> input window [1..3, 1..3] (since oh=1, stride=2, pad=1 => top-left (1,1))
	acc := fp16.ToFloat64(bDense[2])
	for c := 0; c < 3; c++ {
		for kh := 0; kh < 3; kh++ {
			for kw := 0; kw < 3; kw++ {
				acc += fp16.ToFloat64(x.Data[x.Index(0, c, 1+kh, 1+kw)]) * fp16.ToFloat64(wDense[(2*3+c)*9+kh*3+kw])
			}
		}
	}
	if got, want := yDense.Data[yDense.Index(0, 2, 1, 1)], fp16.FromFloat64(acc); got != want {
		t.Fatalf("dense 3x3 pixel mismatch: got 0x%04x want 0x%04x", got, want)
	}

	// Depthwise 3x3
	wDW := make([]uint16, 3*1*9)
	for i := range wDW {
		wDW[i] = fp16.FromFloat64(rng.Float64() - 0.5)
	}
	yDW, err := Conv2d3x3(x, wDW, 3, nil, 1, 1, 1, 1, 3, true)
	if err != nil {
		t.Fatalf("Conv2d3x3 depthwise failed: %v", err)
	}
	for ch := 0; ch < 3; ch++ {
		v := 0.0
		for kh := 0; kh < 3; kh++ {
			for kw := 0; kw < 3; kw++ {
				v += fp16.ToFloat64(x.Data[x.Index(0, ch, kh, kw)]) * fp16.ToFloat64(wDW[ch*9+kh*3+kw])
			}
		}
		if v < 0 {
			v = 0
		}
		if got, want := yDW.Data[yDW.Index(0, ch, 1, 1)], fp16.FromFloat64(v); got != want {
			t.Fatalf("depthwise 3x3 mismatch ch=%d: got 0x%04x want 0x%04x", ch, got, want)
		}
	}
}

func TestElementwiseAffineAndRelu(t *testing.T) {
	a := Tensor{N: 1, C: 1, H: 2, W: 2, Data: toF16Slice(1.0, -2.0, 3.5, 0.25)}
	b := Tensor{N: 1, C: 1, H: 2, W: 2, Data: toF16Slice(0.5, 4.0, -1.5, 0.25)}

	checks := []struct {
		op   string
		want []uint16
	}{
		{"add", toF16Slice(1.5, 2.0, 2.0, 0.5)},
		{"sub", toF16Slice(0.5, -6.0, 5.0, 0.0)},
		{"mul", toF16Slice(0.5, -8.0, -5.25, 0.0625)},
		{"max", toF16Slice(1.0, 4.0, 3.5, 0.25)},
		{"min", toF16Slice(0.5, -2.0, -1.5, 0.25)},
	}
	for _, tc := range checks {
		got, err := Elementwise(tc.op, a, b)
		if err != nil {
			t.Fatalf("Elementwise(%s) error: %v", tc.op, err)
		}
		for i := range tc.want {
			if got.Data[i] != tc.want[i] {
				t.Fatalf("Elementwise(%s)[%d] = 0x%04x, want 0x%04x", tc.op, i, got.Data[i], tc.want[i])
			}
		}
	}

	x := Tensor{N: 1, C: 2, H: 1, W: 2, Data: toF16Slice(1.0, -3.0, 2.0, 0.5)}
	aff, err := Affine(x, toF16Slice(2.0, -1.0), toF16Slice(0.5, 1.0), true)
	if err != nil {
		t.Fatalf("Affine error: %v", err)
	}
	wantAff := toF16Slice(2.5, 0.0, 0.0, 0.5)
	for i := range wantAff {
		if aff.Data[i] != wantAff[i] {
			t.Fatalf("Affine[%d] = 0x%04x, want 0x%04x", i, aff.Data[i], wantAff[i])
		}
	}

	rel, err := Relu(a)
	if err != nil {
		t.Fatalf("Relu error: %v", err)
	}
	wantRel := toF16Slice(1.0, 0.0, 3.5, 0.25)
	for i := range wantRel {
		if rel.Data[i] != wantRel[i] {
			t.Fatalf("Relu[%d] = 0x%04x, want 0x%04x", i, rel.Data[i], wantRel[i])
		}
	}
}
