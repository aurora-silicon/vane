package ref

import (
	"math"
	"testing"

	"github.com/aurora-silicon/vane/internal/fp16"
)

func TestEvalMatmulBasicAndTransposeModes(t *testing.T) {
	// A: [1, 2, 1, 3] = [[1, 2, 3], [4, -1, 0.5]]
	// B: [1, 3, 1, 2] = [[2, -1], [0.5, 3], [-2, 1]]
	// A @ B = [[-3, 8], [6.5, -6.5]]
	aVals := []float64{1, 2, 3, 4, -1, 0.5}
	bVals := []float64{2, -1, 0.5, 3, -2, 1}
	wantVals := []float64{-3, 8, 6.5, -6.5}

	makeTensor := func(n, c, h, w int, vals []float64) Tensor {
		ts, err := NewTensor(n, c, h, w)
		if err != nil {
			t.Fatalf("NewTensor: %v", err)
		}
		for i, v := range vals {
			ts.Data[i] = fp16.FromFloat64(v)
		}
		return ts
	}

	a := makeTensor(1, 2, 1, 3, aVals)
	b := makeTensor(1, 3, 1, 2, bVals)
	aT, err := Transpose(a, [4]int{0, 3, 2, 1}) // [1, 3, 1, 2]
	if err != nil {
		t.Fatalf("Transpose A: %v", err)
	}
	bT, err := Transpose(b, [4]int{0, 3, 2, 1}) // [1, 2, 1, 3]
	if err != nil {
		t.Fatalf("Transpose B: %v", err)
	}

	scale := 0.25
	scaleBits := fp16.FromFloat64(scale)

	cases := []struct {
		name string
		ta   bool
		tb   bool
		inA  Tensor
		inB  Tensor
	}{
		{"plain", false, false, a, b},
		{"transposeB", false, true, a, bT},
		{"transposeA", true, false, aT, b},
		{"transposeAB", true, true, aT, bT},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotF, err := EvalMatmulF64(tc.inA, tc.inB, 1, tc.ta, tc.tb, scale)
			if err != nil {
				t.Fatalf("EvalMatmulF64: %v", err)
			}
			gotQ, err := EvalMatmulQ1516(tc.inA, tc.inB, 1, tc.ta, tc.tb, scaleBits)
			if err != nil {
				t.Fatalf("EvalMatmulQ1516: %v", err)
			}
			if gotF.N != 1 || gotF.C != 2 || gotF.H != 1 || gotF.W != 2 {
				t.Fatalf("unexpected shape (%d,%d,%d,%d)", gotF.N, gotF.C, gotF.H, gotF.W)
			}
			for i, wv := range wantVals {
				want := fp16.FromFloat64(wv * scale)
				if gotF.Data[i] != want {
					t.Fatalf("F64[%d] = 0x%04x (%g), want 0x%04x (%g)",
						i, gotF.Data[i], fp16.ToFloat64(gotF.Data[i]), want, wv*scale)
				}
				if gotQ.Data[i] != want {
					t.Fatalf("Q15.16[%d] = 0x%04x (%g), want 0x%04x (%g)",
						i, gotQ.Data[i], fp16.ToFloat64(gotQ.Data[i]), want, wv*scale)
				}
			}
		})
	}
}

func TestEvalMatmulBatchedEqualsStackedHeads(t *testing.T) {
	const heads, m, k, n = 4, 5, 7, 6

	aBatch, _ := NewTensor(heads, m, 1, k)
	bBatch, _ := NewTensor(heads, n, 1, k) // transposeB = true
	aStack, _ := NewTensor(1, heads*m, 1, k)
	bStack, _ := NewTensor(1, heads*n, 1, k)

	for i := range aBatch.Data {
		v := fp16.FromFloat64(float64((i*7)%19-9) * 0.125)
		aBatch.Data[i] = v
		aStack.Data[i] = v
	}
	for i := range bBatch.Data {
		v := fp16.FromFloat64(float64((i*5)%17-8) * 0.125)
		bBatch.Data[i] = v
		bStack.Data[i] = v
	}

	scale := 1.0 / math.Sqrt(float64(k))
	scaleBits := fp16.FromFloat64(scale)

	yBatchF, err := EvalMatmulF64(aBatch, bBatch, 1, false, true, scale)
	if err != nil {
		t.Fatalf("EvalMatmulF64 batch: %v", err)
	}
	yStackF, err := EvalMatmulF64(aStack, bStack, heads, false, true, scale)
	if err != nil {
		t.Fatalf("EvalMatmulF64 stacked: %v", err)
	}
	if len(yBatchF.Data) != len(yStackF.Data) {
		t.Fatalf("length mismatch %d vs %d", len(yBatchF.Data), len(yStackF.Data))
	}
	for i := range yBatchF.Data {
		if yBatchF.Data[i] != yStackF.Data[i] {
			t.Fatalf("F64 mismatch at %d: 0x%04x vs 0x%04x", i, yBatchF.Data[i], yStackF.Data[i])
		}
	}

	yBatchQ, err := EvalMatmulQ1516(aBatch, bBatch, 1, false, true, scaleBits)
	if err != nil {
		t.Fatalf("EvalMatmulQ1516 batch: %v", err)
	}
	yStackQ, err := EvalMatmulQ1516(aStack, bStack, heads, false, true, scaleBits)
	if err != nil {
		t.Fatalf("EvalMatmulQ1516 stacked: %v", err)
	}
	for i := range yBatchQ.Data {
		if yBatchQ.Data[i] != yStackQ.Data[i] {
			t.Fatalf("Q15.16 mismatch at %d: 0x%04x vs 0x%04x", i, yBatchQ.Data[i], yStackQ.Data[i])
		}
	}
}

func TestEvalMatmulQ1516OverflowAndScaleZero(t *testing.T) {
	a, _ := NewTensor(1, 1, 1, 2)
	b, _ := NewTensor(1, 2, 1, 1)
	a.Data[0] = fp16.FromFloat64(256.0)
	a.Data[1] = fp16.FromFloat64(256.0)
	b.Data[0] = fp16.FromFloat64(256.0)
	b.Data[1] = fp16.FromFloat64(256.0)

	// 256*256 = 65536 overflows signed Q15.16 (max ~32768) -> sticky +inf
	yPos, err := EvalMatmulQ1516(a, b, 1, false, false, fp16.FromFloat64(0.5))
	if err != nil || yPos.Data[0] != fp16.PosInf {
		t.Fatalf("expected +inf on Q15.16 overflow with positive scale, got 0x%04x (err=%v)", yPos.Data[0], err)
	}
	yNeg, err := EvalMatmulQ1516(a, b, 1, false, false, fp16.FromFloat64(-0.5))
	if err != nil || yNeg.Data[0] != fp16.NegInf {
		t.Fatalf("expected -inf on Q15.16 overflow with negative scale, got 0x%04x (err=%v)", yNeg.Data[0], err)
	}
	yZero, err := EvalMatmulQ1516(a, b, 1, false, false, fp16.PosZero)
	if err != nil || yZero.Data[0] != fp16.PosZero {
		t.Fatalf("expected +0.0 on sticky inf * 0, got 0x%04x (err=%v)", yZero.Data[0], err)
	}
}

func TestEvalMatmulErrors(t *testing.T) {
	a, _ := NewTensor(1, 4, 1, 8)
	b, _ := NewTensor(1, 8, 1, 6)
	badH, _ := NewTensor(1, 4, 2, 8)
	badK, _ := NewTensor(1, 7, 1, 6)

	if _, err := EvalMatmulF64(badH, b, 1, false, false, 1.0); err == nil {
		t.Fatal("expected error for H != 1")
	}
	if _, err := EvalMatmulF64(a, badK, 1, false, false, 1.0); err == nil {
		t.Fatal("expected error for K mismatch")
	}
	if _, err := EvalMatmulF64(a, b, 3, false, false, 1.0); err == nil {
		t.Fatal("expected error for C not divisible by heads")
	}
	if _, err := EvalMatmulF64(a, b, 1, false, false, math.NaN()); err == nil {
		t.Fatal("expected error for NaN scale")
	}
	if _, err := EvalMatmulQ1516(a, b, 1, false, false, fp16.PosInf); err == nil {
		t.Fatal("expected error for +inf scaleBits")
	}
}

func TestEvalGraphMultiHeadSDPA(t *testing.T) {
	// Multi-head SDPA: H=2, L=4, S=4, D=4
	scale := 0.5 // 1 / sqrt(4)
	spec := GraphSpec{
		Target: "h17",
		Inputs: []GraphInputSpec{
			{Name: "q", C: 8, H: 1, W: 4},
			{Name: "k", C: 8, H: 1, W: 4},
			{Name: "v", C: 8, H: 1, W: 4},
		},
		Nodes: []GraphNodeSpec{
			{ID: "scores", Op: "matmul", Inputs: []string{"q", "k"}, Heads: 2, TransposeB: true, Scale: &scale},
			{ID: "row_max", Op: "reduce", ReduceOp: "max", Axes: "w", Inputs: []string{"scores"}},
			{ID: "shifted", Op: "pe", PeOp: "sub", Inputs: []string{"scores", "row_max"}},
			{ID: "exp_s", Op: "lut", Act: &GraphActSpec{Name: "exp"}, Inputs: []string{"shifted"}},
			{ID: "inv_sum", Op: "reduce", ReduceOp: "inv_sum", Axes: "w", Inputs: []string{"exp_s"}},
			{ID: "probs", Op: "pe", PeOp: "mul", Inputs: []string{"exp_s", "inv_sum"}},
			{ID: "out", Op: "matmul", Inputs: []string{"probs", "v"}, Heads: 2},
		},
	}

	makeIn := func(seed int) Tensor {
		ts, _ := NewTensor(1, 8, 1, 4)
		for i := range ts.Data {
			ts.Data[i] = fp16.FromFloat64(float64(((i+seed)*7)%13-6) * 0.125)
		}
		return ts
	}
	inputs := map[string]Tensor{
		"q": makeIn(1),
		"k": makeIn(5),
		"v": makeIn(11),
	}

	outF, _, err := EvalGraphF64(spec, inputs, nil)
	if err != nil {
		t.Fatalf("EvalGraphF64 SDPA: %v", err)
	}
	outQ, _, err := EvalGraphQ1516(spec, inputs, nil)
	if err != nil {
		t.Fatalf("EvalGraphQ1516 SDPA: %v", err)
	}
	if outF.N != 1 || outF.C != 8 || outF.H != 1 || outF.W != 4 {
		t.Fatalf("unexpected SDPA out shape (%d,%d,%d,%d)", outF.N, outF.C, outF.H, outF.W)
	}
	cmp, err := CompareFP16(outQ.Data, outF.Data, 5e-2, 5e-2, nil)
	if err != nil || !cmp.Passed {
		t.Fatalf("SDPA Q15.16 vs F64 compare failed: %+v (err=%v)", cmp, err)
	}
}
