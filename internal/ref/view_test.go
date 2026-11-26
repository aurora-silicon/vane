package ref

import (
	"testing"

	"github.com/aurora-silicon/vane/internal/fp16"
)

func mkSeqTensor(t *testing.T, n, c, h, w int) Tensor {
	t.Helper()
	x, err := NewTensor(n, c, h, w)
	if err != nil {
		t.Fatal(err)
	}
	for i := range x.Data {
		// Distinct deterministic fp16 bit patterns
		x.Data[i] = uint16(0x3800 + (i % 1024))
	}
	return x
}

func TestViewTransposeSliceConcatSplitReshape(t *testing.T) {
	// 1. Transpose W-preserving (C<->H) & T-mode (C<->W) & 4D N>1
	x := mkSeqTensor(t, 2, 4, 3, 6)
	trCH, err := Transpose(x, [4]int{0, 2, 1, 3})
	if err != nil {
		t.Fatalf("Transpose C<->H failed: %v", err)
	}
	if trCH.N != 2 || trCH.C != 3 || trCH.H != 4 || trCH.W != 6 {
		t.Fatalf("unexpected trCH shape: (%d,%d,%d,%d)", trCH.N, trCH.C, trCH.H, trCH.W)
	}
	for n := 0; n < 2; n++ {
		for c := 0; c < 4; c++ {
			for h := 0; h < 3; h++ {
				for w := 0; w < 6; w++ {
					got := trCH.Data[trCH.Index(n, h, c, w)]
					want := x.Data[x.Index(n, c, h, w)]
					if got != want {
						t.Fatalf("trCH(%d,%d,%d,%d) = 0x%04x, want 0x%04x", n, h, c, w, got, want)
					}
				}
			}
		}
	}

	trCW, err := Transpose(x, [4]int{0, 3, 2, 1})
	if err != nil {
		t.Fatalf("Transpose C<->W failed: %v", err)
	}
	if trCW.N != 2 || trCW.C != 6 || trCW.H != 3 || trCW.W != 4 {
		t.Fatalf("unexpected trCW shape: (%d,%d,%d,%d)", trCW.N, trCW.C, trCW.H, trCW.W)
	}
	for n := 0; n < 2; n++ {
		for c := 0; c < 4; c++ {
			for h := 0; h < 3; h++ {
				for w := 0; w < 6; w++ {
					got := trCW.Data[trCW.Index(n, w, h, c)]
					want := x.Data[x.Index(n, c, h, w)]
					if got != want {
						t.Fatalf("trCW(%d,%d,%d,%d) = 0x%04x, want 0x%04x", n, w, h, c, got, want)
					}
				}
			}
		}
	}

	// 2. Slice with stride along H and offset along N, C, W
	xS := mkSeqTensor(t, 4, 8, 6, 64)
	sl, err := Slice(xS, [4]int{1, 2, 1, 32}, [4]int{3, 6, 6, 64}, [4]int{1, 1, 2, 1})
	if err != nil {
		t.Fatalf("Slice failed: %v", err)
	}
	if sl.N != 2 || sl.C != 4 || sl.H != 3 || sl.W != 32 {
		t.Fatalf("unexpected slice shape: (%d,%d,%d,%d)", sl.N, sl.C, sl.H, sl.W)
	}
	for on := 0; on < sl.N; on++ {
		for oc := 0; oc < sl.C; oc++ {
			for oh := 0; oh < sl.H; oh++ {
				for ow := 0; ow < sl.W; ow++ {
					got := sl.Data[sl.Index(on, oc, oh, ow)]
					want := xS.Data[xS.Index(1+on, 2+oc, 1+oh*2, 32+ow)]
					if got != want {
						t.Fatalf("slice mismatch at (%d,%d,%d,%d)", on, oc, oh, ow)
					}
				}
			}
		}
	}

	// 3. SplitPart + Concat round-trip across C, H, W
	for _, tc := range []struct {
		axis  int
		sizes []int
	}{
		{1, []int{2, 3, 3}},
		{2, []int{2, 4}},
		{3, []int{32, 32}},
	} {
		parts := make([]Tensor, len(tc.sizes))
		for p := range tc.sizes {
			pt, err := SplitPart(xS, tc.axis, tc.sizes, p)
			if err != nil {
				t.Fatalf("SplitPart axis=%d part=%d failed: %v", tc.axis, p, err)
			}
			parts[p] = pt
		}
		cat, err := Concat(parts, tc.axis)
		if err != nil {
			t.Fatalf("Concat axis=%d failed: %v", tc.axis, err)
		}
		if cat.N != xS.N || cat.C != xS.C || cat.H != xS.H || cat.W != xS.W {
			t.Fatalf("Concat axis=%d shape mismatch", tc.axis)
		}
		for i := range xS.Data {
			if cat.Data[i] != xS.Data[i] {
				t.Fatalf("SplitPart+Concat axis=%d round-trip mismatch at %d", tc.axis, i)
			}
		}
	}

	// 4. Reshape
	rs, err := Reshape(xS, [4]int{2, 16, 12, 32})
	if err != nil {
		t.Fatalf("Reshape failed: %v", err)
	}
	if rs.N != 2 || rs.C != 16 || rs.H != 12 || rs.W != 32 {
		t.Fatalf("unexpected reshape shape")
	}
	for i := range xS.Data {
		if rs.Data[i] != xS.Data[i] {
			t.Fatalf("Reshape mismatch at %d", i)
		}
	}

	// 5. Negative error checks
	if _, err := Transpose(x, [4]int{0, 1, 1, 3}); err == nil {
		t.Fatal("expected error for duplicate axis in Transpose")
	}
	if _, err := Slice(x, [4]int{0, 0, 0, 0}, [4]int{2, 4, 3, 6}, [4]int{1, 0, 1, 1}); err == nil {
		t.Fatal("expected error for zero step in Slice")
	}
	if _, err := Concat([]Tensor{x}, 1); err == nil {
		t.Fatal("expected error for single-input Concat")
	}
	if _, err := SplitPart(x, 1, []int{2, 3}, 0); err == nil {
		t.Fatal("expected error for mismatched split sizes sum")
	}
	if _, err := Reshape(x, [4]int{1, 1, 1, 5}); err == nil {
		t.Fatal("expected error for mismatched Reshape element count")
	}
}

func TestEvalGraphM5cUnetAndTransformerFFN(t *testing.T) {
	// 1. U-Net skip-concat graph: conv3x3 -> conv1x1 -> concat(C) -> conv1x1
	axisC := 1
	unetSpec := GraphSpec{
		Target: "h17",
		Inputs: []GraphInputSpec{{Name: "x", C: 4, H: 4, W: 8}},
		Nodes: []GraphNodeSpec{
			{ID: "enc", Op: "conv3x3", Mode: "dense", Cout: 4, Stride: 1, Pad: "same", Relu: true, Inputs: []string{"x"}},
			{ID: "mid", Op: "conv1x1", Cout: 4, Relu: true, Inputs: []string{"enc"}},
			{ID: "cat", Op: "concat", Axis: &axisC, Inputs: []string{"mid", "enc"}},
			{ID: "dec", Op: "conv1x1", Cout: 4, Inputs: []string{"cat"}},
		},
	}
	xU, _ := NewTensor(1, 4, 4, 8)
	for i := range xU.Data {
		xU.Data[i] = fp16.FromFloat64(float64(i%7-3) * 0.25)
	}
	mkW := func(n int) []uint16 {
		w := make([]uint16, n)
		for i := range w {
			w[i] = fp16.FromFloat64(float64(i%5-2) * 0.125)
		}
		return w
	}
	uWeights := map[string]GraphNodeWeights{
		"enc": {Weights: mkW(4 * 4 * 9)},
		"mid": {Weights: mkW(4 * 4)},
		"dec": {Weights: mkW(4 * 8)},
	}
	yUQ, envU, err := EvalGraphQ1516(unetSpec, map[string]Tensor{"x": xU}, uWeights)
	if err != nil {
		t.Fatalf("U-Net EvalGraphQ1516 failed: %v", err)
	}
	yUF, _, err := EvalGraphF64(unetSpec, map[string]Tensor{"x": xU}, uWeights)
	if err != nil {
		t.Fatalf("U-Net EvalGraphF64 failed: %v", err)
	}
	if envU["cat"].C != 8 || yUQ.C != 4 || yUF.C != 4 {
		t.Fatalf("unexpected U-Net shapes: cat.C=%d out.C=%d", envU["cat"].C, yUQ.C)
	}

	// 2. Large-hidden Transformer FFN graph with linear + GELU + RMSNorm + residual
	eps := 0.001
	ffnSpec := GraphSpec{
		Target: "h17",
		Inputs: []GraphInputSpec{{Name: "x", C: 16, H: 1, W: 8}},
		Nodes: []GraphNodeSpec{
			{ID: "sq", Op: "pe", PeOp: "square", Inputs: []string{"x"}},
			{ID: "ms", Op: "reduce", ReduceOp: "mean", Axes: "c", Inputs: []string{"sq"}},
			{ID: "ms_eps", Op: "pe", PeOp: "sadd", Scalar: &eps, Inputs: []string{"ms"}},
			{ID: "inv_rms", Op: "lut", Act: &GraphActSpec{Name: "rsqrt"}, Inputs: []string{"ms_eps"}},
			{ID: "norm", Op: "pe", PeOp: "mul", Inputs: []string{"x", "inv_rms"}},
			{ID: "up", Op: "linear", K: 16, N: 32, Act: &GraphActSpec{Name: "gelu"}, Inputs: []string{"norm"}},
			{ID: "down", Op: "linear", K: 32, N: 16, Inputs: []string{"up"}},
			{ID: "out", Op: "pe", PeOp: "add", Inputs: []string{"down", "x"}},
		},
	}
	xF, _ := NewTensor(1, 16, 1, 8)
	for i := range xF.Data {
		xF.Data[i] = fp16.FromFloat64(float64(i%9-4) * 0.25)
	}
	fWeights := map[string]GraphNodeWeights{
		"up":   {Weights: mkW(32 * 16)},
		"down": {Weights: mkW(16 * 32)},
	}
	yFQ, envF, err := EvalGraphQ1516(ffnSpec, map[string]Tensor{"x": xF}, fWeights)
	if err != nil {
		t.Fatalf("FFN EvalGraphQ1516 failed: %v", err)
	}
	yFF, _, err := EvalGraphF64(ffnSpec, map[string]Tensor{"x": xF}, fWeights)
	if err != nil {
		t.Fatalf("FFN EvalGraphF64 failed: %v", err)
	}
	if envF["up"].C != 32 || yFQ.C != 16 || yFF.C != 16 {
		t.Fatalf("unexpected FFN shapes: up.C=%d out.C=%d", envF["up"].C, yFQ.C)
	}
	cmp, err := CompareFP16(yFQ.Data, yFF.Data, 5e-2, 5e-2, nil)
	if err != nil || !cmp.Passed {
		t.Fatalf("FFN Q15.16 vs F64 compare failed: %+v err=%v", cmp, err)
	}

	// 3. View-chained graph: transpose -> slice -> reshape
	viewSpec := GraphSpec{
		Target: "h17",
		Inputs: []GraphInputSpec{{Name: "x", C: 4, H: 8, W: 32}},
		Nodes: []GraphNodeSpec{
			{ID: "tr", Op: "transpose", Perm: []int{0, 2, 1, 3}, Inputs: []string{"x"}},
			{ID: "sl", Op: "slice", Begin: []int{0, 0, 0, 0}, End: []int{1, 4, 4, 32}, Inputs: []string{"tr"}},
			{ID: "rs", Op: "reshape", OutShape: []int{1, 8, 2, 32}, Inputs: []string{"sl"}},
		},
	}
	xV := mkSeqTensor(t, 1, 4, 8, 32)
	yVQ, _, err := EvalGraphQ1516(viewSpec, map[string]Tensor{"x": xV}, nil)
	if err != nil {
		t.Fatalf("View graph EvalGraphQ1516 failed: %v", err)
	}
	yVF, _, err := EvalGraphF64(viewSpec, map[string]Tensor{"x": xV}, nil)
	if err != nil {
		t.Fatalf("View graph EvalGraphF64 failed: %v", err)
	}
	if yVQ.C != 8 || yVQ.H != 2 || yVQ.W != 32 {
		t.Fatalf("unexpected view graph output shape: (%d,%d,%d,%d)", yVQ.N, yVQ.C, yVQ.H, yVQ.W)
	}
	for i := range yVQ.Data {
		if yVQ.Data[i] != yVF.Data[i] {
			t.Fatalf("view graph exact mismatch at %d", i)
		}
	}
}
