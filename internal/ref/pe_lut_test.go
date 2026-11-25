package ref

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/aurora-silicon/vane/internal/fp16"
)

func TestPeUnaryChainAndBinaryBroadcast(t *testing.T) {
	// 1-source unary with special fp16 patterns (including exp-31 H17 extended binade)
	xEdge := Tensor{
		N: 1, C: 1, H: 1, W: 6,
		Data: []uint16{0x8000, 0xbc00, 0x4000, 0x7c00, 0xfc00, 0x7e00}, // -0, -1, 2, +inf, -inf, qNaN
	}
	yAbs, err := PeUnary("abs", xEdge)
	if err != nil {
		t.Fatalf("PeUnary(abs): %v", err)
	}
	wantAbs := []uint16{0x0000, 0x3c00, 0x4000, 0x7c00, 0x7c00, 0x7c00}
	for i, w := range wantAbs {
		if yAbs.Data[i] != w {
			t.Fatalf("PeUnary(abs)[%d] = 0x%04x, want 0x%04x", i, yAbs.Data[i], w)
		}
	}

	ySq, err := PeUnary("square", xEdge)
	if err != nil {
		t.Fatalf("PeUnary(square): %v", err)
	}
	wantSq := []uint16{0x0000, 0x3c00, 0x4400, 0x7c00, 0x7c00, 0x7c00}
	for i, w := range wantSq {
		if ySq.Data[i] != w {
			t.Fatalf("PeUnary(square)[%d] = 0x%04x, want 0x%04x", i, ySq.Data[i], w)
		}
	}

	// 1-source 3-step chain: in_scale=1.5, bias=-0.25, scale=0.5
	// Also test intermediate in extended [65520, 131072) binade:
	// x = 65504 (0x7bff) * 1.5 = 98256 -> RNA16Step = 98240 -> + (-0.25) = 98240 -> * 0.5 = 49120 (finite!)
	inS := fp16.FromFloat64(1.5)
	bias := fp16.FromFloat64(-0.25)
	sc := fp16.FromFloat64(0.5)
	xChain := Tensor{N: 1, C: 1, H: 1, W: 3, Data: []uint16{fp16.FromFloat64(2.0), fp16.FromFloat64(-4.0), 0x7bff}}
	yChain, err := PeChain(xChain, &inS, &bias, &sc)
	if err != nil {
		t.Fatalf("PeChain: %v", err)
	}
	if yChain.Data[0] != fp16.RNA16(1.375) {
		t.Fatalf("PeChain[0] = 0x%04x, want 1.375", yChain.Data[0])
	}
	if yChain.Data[1] != fp16.RNA16(-3.125) {
		t.Fatalf("PeChain[1] = 0x%04x, want -3.125", yChain.Data[1])
	}
	// Extended intermediate stays finite after scaling by 0.5
	if !fp16.IsFinite(yChain.Data[2]) {
		t.Fatalf("expected finite output via extended intermediate binade, got 0x%04x", yChain.Data[2])
	}

	// 2-source binary with broadcast & rev (rsub)
	a := Tensor{N: 1, C: 2, H: 2, W: 2, Data: toF16Slice(1, 2, 3, 4, 5, 6, 7, 8)}
	b := Tensor{N: 1, C: 2, H: 1, W: 1, Data: toF16Slice(10, 20)}
	yAdd, err := PeBinary("add", a, b, false)
	if err != nil {
		t.Fatalf("PeBinary(add bcast): %v", err)
	}
	wantAdd := toF16Slice(11, 12, 13, 14, 25, 26, 27, 28)
	for i, w := range wantAdd {
		if yAdd.Data[i] != w {
			t.Fatalf("PeBinary(add)[%d] = 0x%04x, want 0x%04x", i, yAdd.Data[i], w)
		}
	}

	yRSub, err := PeBinary("rsub", a, b, false)
	if err != nil {
		t.Fatalf("PeBinary(rsub bcast): %v", err)
	}
	wantRSub := toF16Slice(9, 8, 7, 6, 15, 14, 13, 12)
	for i, w := range wantRSub {
		if yRSub.Data[i] != w {
			t.Fatalf("PeBinary(rsub)[%d] = 0x%04x, want 0x%04x", i, yRSub.Data[i], w)
		}
	}

	// All 5 binary ops
	for _, op := range []string{"add", "sub", "mul", "max", "min"} {
		if _, err := PeBinary(op, a, b, false); err != nil {
			t.Fatalf("PeBinary(%s): %v", op, err)
		}
	}
}

func TestEvalLutModesSymmetryAndAccuracy(t *testing.T) {
	specs := []struct {
		spec ActSpec
		xs   []float64
		tol  float64
	}{
		{ActSpec{Name: "relu6"}, []float64{-2, 0, 3, 6, 9}, 0},
		{ActSpec{Name: "clip", Lo: -1.5, Hi: 2.5}, []float64{-3, -1.5, 0, 1.25, 2.5, 4}, 0},
		{ActSpec{Name: "leaky_relu", Alpha: 0.25}, []float64{-4, -1, 0, 2}, 1e-3},
		{ActSpec{Name: "abs"}, []float64{-5, -0.5, 0, 0.5, 5}, 0},
		{ActSpec{Name: "sign"}, []float64{-2, -0.01, 0, 0.01, 2}, 0},
		{ActSpec{Name: "threshold", Scalar: 0.5}, []float64{-2, 0, 0.5, 2}, 0},
		{ActSpec{Name: "thresholded_relu", Alpha: 1.0}, []float64{-2, 0, 0.5, 1.5, 3}, 0},
		{ActSpec{Name: "hard_sigmoid", Alpha: 0.2, Beta: 0.5}, []float64{-4, -2, 0, 2, 4}, 2e-3},
		{ActSpec{Name: "hard_sigmoid", Alpha: 0.5, Beta: 0.25}, []float64{-2, -0.5, 0, 1, 3}, 2e-3},
		{ActSpec{Name: "clamp_max", Scalar: -0.5}, []float64{-2, -0.5, 0, 2}, 0},
		{ActSpec{Name: "clamp_min", Scalar: 2.0}, []float64{-2, 0, 2, 4}, 0},
		{ActSpec{Name: "elu", Alpha: 1.0}, []float64{-6, -2, -0.5, 0, 1, 3}, 1e-2},
		{ActSpec{Name: "tanh"}, []float64{-5, -2, -0.5, 0, 0.5, 2, 5}, 3e-3},
		{ActSpec{Name: "scaled_tanh", Alpha: 2.0, Beta: 0.5}, []float64{-4, -1, 0, 1, 4}, 5e-3},
		{ActSpec{Name: "scaled_tanh", Alpha: -2.0, Beta: 0.5}, []float64{-4, -1, 0, 1, 4}, 5e-3},
		{ActSpec{Name: "erf"}, []float64{-4, -1, 0, 1, 4}, 4e-3},
		{ActSpec{Name: "sigmoid"}, []float64{-8, -2, 0, 2, 8}, 5e-3},
		{ActSpec{Name: "gelu"}, []float64{-4, -1, 0, 1, 4}, 1e-2},
		{ActSpec{Name: "silu"}, []float64{-6, -1, 0, 1, 6}, 2e-2},
		{ActSpec{Name: "exp2"}, []float64{-4, -1.5, 0, 0.5, 3}, 2e-2},
		{ActSpec{Name: "exp"}, []float64{-4, -1, 0, 0.5, 2}, 0.1},
		{ActSpec{Name: "sqrt"}, []float64{0, 0.25, 1, 2, 4, 9}, 1e-2},
		{ActSpec{Name: "rsqrt", Eps: 0.0}, []float64{0.25, 1, 4, 16}, 1e-2},
		{ActSpec{Name: "rsqrt", Eps: 0.25}, []float64{0, 0.75, 3.75}, 1e-2},
		{ActSpec{Name: "inverse"}, []float64{-4, -1, -0.5, 0.5, 1, 4}, 1e-2},
		{ActSpec{Name: "log2"}, []float64{0.25, 0.5, 1, 2, 4, 16}, 1e-2},
	}

	for _, tc := range specs {
		tbl, preScale, shift, err := BuildLutTable(tc.spec)
		if err != nil {
			t.Fatalf("BuildLutTable(%s): %v", tc.spec.Name, err)
		}
		if len(tbl) != 128 {
			t.Fatalf("%s table len %d != 128", tc.spec.Name, len(tbl))
		}
		for _, x := range tc.xs {
			xh := fp16.FromFloat64(x)
			xf := fp16.ToFloat64(xh)
			gotBits, err := EvalLut(tbl, xh, preScale, shift)
			if err != nil {
				t.Fatalf("EvalLut(%s, %g): %v", tc.spec.Name, xf, err)
			}
			want, err := EvalActExactF64(tc.spec, xf)
			if err != nil {
				t.Fatalf("EvalActExactF64(%s): %v", tc.spec.Name, err)
			}
			got := fp16.ToFloat64(gotBits)
			if math.Abs(got-want) > tc.tol {
				t.Fatalf("%s(%g): got=%g, want=%g (err=%g > tol=%g)", tc.spec.Name, xf, got, want, math.Abs(got-want), tc.tol)
			}
		}
	}

	// Test symmetry flags 0x80 (negated) and 0xc0 (zero-negative) explicitly on synthetic tables
	tblNeg := make([]byte, 128)
	binary.LittleEndian.PutUint16(tblNeg[0:], fp16.NegInf)
	binary.LittleEndian.PutUint16(tblNeg[2:], fp16.PosInf)
	binary.LittleEndian.PutUint16(tblNeg[4:], fp16.NegInf)
	binary.LittleEndian.PutUint16(tblNeg[6:], fp16.PosInf)
	binary.LittleEndian.PutUint16(tblNeg[37*2:], 0x3c00) // ls=1
	binary.LittleEndian.PutUint16(tblNeg[39*2:], 0x3c00) // rs=1
	binary.LittleEndian.PutUint16(tblNeg[41*2:], 0x0040)
	binary.LittleEndian.PutUint16(tblNeg[42*2:], 0x0081) // negated PWL
	yNeg, err := EvalLut(tblNeg, fp16.FromFloat64(2.0), 0x3c00, nil)
	if err != nil || fp16.ToFloat64(yNeg) != -2.0 {
		t.Fatalf("sym 0x80 failed: got 0x%04x err=%v", yNeg, err)
	}

	binary.LittleEndian.PutUint16(tblNeg[42*2:], 0x00c1) // zero-negative PWL
	yZN, err := EvalLut(tblNeg, fp16.FromFloat64(-3.0), 0x3c00, nil)
	if err != nil || yZN != fp16.PosZero {
		t.Fatalf("sym 0xc0 neg failed: got 0x%04x err=%v", yZN, err)
	}
	yZP, err := EvalLut(tblNeg, fp16.FromFloat64(3.0), 0x3c00, nil)
	if err != nil || fp16.ToFloat64(yZP) != 3.0 {
		t.Fatalf("sym 0xc0 pos failed: got 0x%04x err=%v", yZP, err)
	}
}

func TestFusedConv1x1And3x3Lut(t *testing.T) {
	rng := rand.New(rand.NewPCG(314, 159))
	x, _ := NewTensor(1, 4, 6, 8)
	for i := range x.Data {
		x.Data[i] = fp16.FromFloat64(rng.Float64()*2.0 - 1.0)
	}
	w1 := make([]uint16, 4*4)
	for i := range w1 {
		w1[i] = fp16.FromFloat64((rng.Float64()*2.0 - 1.0) * 0.4)
	}
	w3 := make([]uint16, 4*4*9)
	for i := range w3 {
		w3[i] = fp16.FromFloat64((rng.Float64()*2.0 - 1.0) * 0.2)
	}
	bias := make([]uint16, 4)
	for i := range bias {
		bias[i] = fp16.FromFloat64(rng.Float64() - 0.5)
	}

	for _, spec := range []ActSpec{
		{Name: "tanh"},
		{Name: "gelu"},
		{Name: "silu"},
		{Name: "hard_sigmoid", Alpha: 0.5, Beta: 0.25},
		{Name: "rsqrt", Eps: 0.0625},
	} {
		tbl, preScale, shift, err := BuildLutTable(spec)
		if err != nil {
			t.Fatalf("BuildLutTable(%s): %v", spec.Name, err)
		}
		// Keep inputs strictly positive for rsqrt test so exact analytical reference is finite
		xTest := x
		w1Test := w1
		w3Test := w3
		bTest := bias
		if spec.Name == "rsqrt" {
			xTest, _ = NewTensor(1, 4, 6, 8)
			for i := range xTest.Data {
				xTest.Data[i] = fp16.FromFloat64(0.5 + rng.Float64()*0.5)
			}
			w1Test = make([]uint16, 4*4)
			for i := range w1Test {
				w1Test[i] = fp16.FromFloat64(0.1 + rng.Float64()*0.1)
			}
			w3Test = make([]uint16, 4*4*9)
			for i := range w3Test {
				w3Test[i] = fp16.FromFloat64(0.02 + rng.Float64()*0.02)
			}
			bTest = make([]uint16, 4)
			for i := range bTest {
				bTest[i] = fp16.FromFloat64(0.5)
			}
		}
		y1F64, err := Conv2d1x1Act(xTest, w1Test, 4, bTest, spec)
		if err != nil {
			t.Fatalf("Conv2d1x1Act(%s): %v", spec.Name, err)
		}
		y1Lut, err := Conv2d1x1Q1516Lut(xTest, w1Test, 4, bTest, tbl, preScale, shift)
		if err != nil {
			t.Fatalf("Conv2d1x1Q1516Lut(%s): %v", spec.Name, err)
		}
		comp1, err := CompareFP16(y1Lut.Data, y1F64.Data, 2e-2, 2e-2, nil)
		if err != nil || !comp1.Passed {
			t.Fatalf("1x1+%s mismatch: %+v err=%v", spec.Name, comp1, err)
		}

		y3F64, err := Conv2d3x3PadAct(xTest, w3Test, 4, bTest, 2, "same", 1, spec)
		if err != nil {
			t.Fatalf("Conv2d3x3PadAct(%s): %v", spec.Name, err)
		}
		y3Lut, err := Conv2d3x3Q1516Lut(xTest, w3Test, 4, bTest, 2, "same", 1, tbl, preScale, shift)
		if err != nil {
			t.Fatalf("Conv2d3x3Q1516Lut(%s): %v", spec.Name, err)
		}
		comp3, err := CompareFP16(y3Lut.Data, y3F64.Data, 2e-2, 2e-2, nil)
		if err != nil || !comp3.Passed {
			t.Fatalf("3x3+%s mismatch: %+v err=%v", spec.Name, comp3, err)
		}
	}

	// M2: fused bias + LUT shift fp16 overflow must return an error
	tblRsqrt, psRsqrt, shRsqrt, err := BuildLutTable(ActSpec{Name: "rsqrt", Eps: 32.0})
	if err != nil {
		t.Fatalf("BuildLutTable(rsqrt 32): %v", err)
	}
	bigBias := []uint16{0x7bff, 0x7bff, 0x7bff, 0x7bff}
	if _, err := Conv2d1x1Q1516Lut(x, w1, 4, bigBias, tblRsqrt, psRsqrt, shRsqrt); err == nil {
		t.Fatalf("expected Conv2d1x1Q1516Lut to reject fp16 bias+shift overflow")
	}
	if _, err := Conv2d3x3Q1516Lut(x, w3, 4, bigBias, 2, "same", 1, tblRsqrt, psRsqrt, shRsqrt); err == nil {
		t.Fatalf("expected Conv2d3x3Q1516Lut to reject fp16 bias+shift overflow")
	}
}

func TestThresholdedReluBoundaryAndLutParamValidation(t *testing.T) {
	// Q3: thresholded_relu(a) exact boundary: prevF16(a) -> 0, a -> 0, nextF16(a) -> nextF16(a)
	for _, a := range []float64{0.0, 1.0, -0.5, 2.25} {
		aBits := f16CanonZero(a)
		tbl, ps, sh, err := BuildLutTable(ActSpec{Name: "thresholded_relu", Alpha: a})
		if err != nil {
			t.Fatalf("BuildLutTable(thresholded_relu %g): %v", a, err)
		}
		if gotLo := binary.LittleEndian.Uint16(tbl[0:]); gotLo != aBits {
			t.Fatalf("thresholded_relu(%g): lo=0x%04x, want 0x%04x", a, gotLo, aBits)
		}
		yPrev, _ := EvalLut(tbl, fp16.PrevF16(aBits), ps, sh)
		yAt, _ := EvalLut(tbl, aBits, ps, sh)
		nextBits := fp16.NextF16(aBits)
		yNext, _ := EvalLut(tbl, nextBits, ps, sh)
		if yPrev != fp16.PosZero || yAt != fp16.PosZero || yNext != nextBits {
			t.Fatalf("thresholded_relu(%g) boundary: prev=0x%04x at=0x%04x next=0x%04x (want 0, 0, 0x%04x)",
				a, yPrev, yAt, yNext, nextBits)
		}
	}

	// M3: reject NaN, Inf, fp16 overflows, and hard_sigmoid preScale underflow
	badSpecs := []ActSpec{
		{Name: "clip", Lo: math.NaN(), Hi: 1.0},
		{Name: "clip", Lo: -1.0, Hi: math.Inf(1)},
		{Name: "clip", Lo: -70000, Hi: 1.0},
		{Name: "leaky_relu", Alpha: math.NaN()},
		{Name: "leaky_relu", Alpha: 70000},
		{Name: "threshold", Scalar: math.Inf(-1)},
		{Name: "thresholded_relu", Alpha: 70000},
		{Name: "hard_sigmoid", Alpha: 20000, Beta: 0.5},
		{Name: "hard_sigmoid", Alpha: 1e-9, Beta: 0.5},
		{Name: "hard_sigmoid", Alpha: 0.2, Beta: 1e5},
		{Name: "clamp_max", Scalar: math.Inf(1)},
		{Name: "clamp_min", Scalar: -70000},
		{Name: "elu", Alpha: 70000},
		{Name: "scaled_tanh", Alpha: 70000, Beta: 1.0},
		{Name: "scaled_tanh", Alpha: 1.0, Beta: math.Inf(1)},
		{Name: "rsqrt", Eps: 70000},
	}
	for _, spec := range badSpecs {
		if _, _, _, err := BuildLutTable(spec); err == nil {
			t.Fatalf("expected BuildLutTable(%+v) to fail, but succeeded", spec)
		}
	}
}
