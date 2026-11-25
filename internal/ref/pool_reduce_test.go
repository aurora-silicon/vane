package ref

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/aurora-silicon/vane/internal/fp16"
)

func TestEvalPoolMaxAndAvgModes(t *testing.T) {
	// 1x1x4x4 tensor with values 1..16
	x, err := NewTensor(1, 1, 4, 4)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 16; i++ {
		x.Data[i] = fp16.FromFloat64(float64(i + 1))
	}

	// 2x2 s2 valid max_pool -> [[6, 8], [14, 16]]
	yMax, err := EvalPool(x, PoolSpec{Op: "max", KH: 2, KW: 2, SH: 2, SW: 2, Pad: "valid"})
	if err != nil {
		t.Fatalf("EvalPool max valid: %v", err)
	}
	if yMax.H != 2 || yMax.W != 2 {
		t.Fatalf("unexpected shape (%d,%d)", yMax.H, yMax.W)
	}
	wantMax := []float64{6, 8, 14, 16}
	for i, w := range wantMax {
		got := fp16.ToFloat64(yMax.Data[i])
		if got != w {
			t.Fatalf("max[%d] = %v, want %v", i, got, w)
		}
	}

	// 3x3 s1 same avg_pool exclude_pad=true vs false on corner (0,0)
	// Window at (0,0) with pt=1, pl=1 covers real pixels (0,0)=1, (0,1)=2, (1,0)=5, (1,1)=6 (sum=14, count=4)
	yExcl, err := EvalPool(x, PoolSpec{Op: "avg", KH: 3, KW: 3, SH: 1, SW: 1, Pad: "same", ExcludePad: true})
	if err != nil {
		t.Fatalf("EvalPool avg same excl: %v", err)
	}
	yIncl, err := EvalPool(x, PoolSpec{Op: "avg", KH: 3, KW: 3, SH: 1, SW: 1, Pad: "same", ExcludePad: false})
	if err != nil {
		t.Fatalf("EvalPool avg same incl: %v", err)
	}
	if got := fp16.ToFloat64(yExcl.Data[0]); math.Abs(got-3.5) > 1e-3 {
		t.Fatalf("avg excl corner = %v, want 3.5", got)
	}
	if got := fp16.ToFloat64(yIncl.Data[0]); math.Abs(got-14.0/9.0) > 2e-3 {
		t.Fatalf("avg incl corner = %v, want %v", got, 14.0/9.0)
	}

	// Custom padding (1, 0, 1, 0) with 2x2 s1 -> out 4x4
	yCust, err := EvalPool(x, PoolSpec{Op: "max", KH: 2, KW: 2, SH: 1, SW: 1, Pad: "custom", PT: 1, PB: 0, PL: 1, PR: 0})
	if err != nil {
		t.Fatalf("EvalPool custom: %v", err)
	}
	if yCust.H != 4 || yCust.W != 4 {
		t.Fatalf("custom shape (%d,%d), want (4,4)", yCust.H, yCust.W)
	}
	if got := fp16.ToFloat64(yCust.Data[0]); got != 1.0 {
		t.Fatalf("custom corner = %v, want 1.0", got)
	}
}

func TestEvalReduceAllTenOpsAndAxes(t *testing.T) {
	// 1x2x2x4 tensor with known values
	// c=0, h=0: [1, -2, 3, -4] -> sum=-2, l1=10, ss=30, max=3, min=-4
	// c=0, h=1: [2, 2, 2, 2]   -> sum=8,  l1=8,  ss=16, max=2, min=2
	x, _ := NewTensor(1, 2, 2, 4)
	vals := []float64{
		1, -2, 3, -4,
		2, 2, 2, 2,
		0.5, 0.5, 0.5, 0.5,
		4, 0, -3, 0,
	}
	for i, v := range vals {
		x.Data[i] = fp16.FromFloat64(v)
	}

	eps := 0.25
	cases := []struct {
		op   string
		want float64 // expected at (0, 0, 1, 0) i.e. row [2, 2, 2, 2]
	}{
		{"sum", 8.0},
		{"mean", 2.0},
		{"max", 2.0},
		{"min", 2.0},
		{"l1", 8.0},
		{"sum_square", 16.0},
		{"l2", 4.0},
		{"inv_sum", 0.125},
		{"rsqrt_sum_square", 1.0 / math.Sqrt(16.0+eps)},
		{"rsqrt_mean_square", 1.0 / math.Sqrt(4.0+eps)},
	}
	for _, tc := range cases {
		y, err := EvalReduce(x, ReduceSpec{Op: tc.op, ReduceW: true, Eps: eps})
		if err != nil {
			t.Fatalf("EvalReduce %s: %v", tc.op, err)
		}
		if y.C != 2 || y.H != 2 || y.W != 1 {
			t.Fatalf("%s shape (%d,%d,%d) != (2,2,1)", tc.op, y.C, y.H, y.W)
		}
		got := fp16.ToFloat64(y.Data[y.Index(0, 0, 1, 0)])
		if math.Abs(got-tc.want) > 2e-3 {
			t.Fatalf("reduce %s row(0,1) = %v, want %v", tc.op, got, tc.want)
		}
	}

	// Verify all 7 axis subsets produce expected output shapes
	axisExpect := map[string][3]int{
		"w":   {2, 2, 1},
		"h":   {2, 1, 4},
		"c":   {1, 2, 4},
		"hw":  {2, 1, 1},
		"cw":  {1, 2, 1},
		"ch":  {1, 1, 4},
		"chw": {1, 1, 1},
	}
	for axStr, want := range axisExpect {
		rc, rh, rw, err := ParseReduceAxes(axStr)
		if err != nil {
			t.Fatalf("ParseReduceAxes(%q): %v", axStr, err)
		}
		y, err := EvalReduce(x, ReduceSpec{Op: "sum", ReduceC: rc, ReduceH: rh, ReduceW: rw})
		if err != nil {
			t.Fatalf("EvalReduce axes=%s: %v", axStr, err)
		}
		if y.C != want[0] || y.H != want[1] || y.W != want[2] {
			t.Fatalf("axes=%s shape (%d,%d,%d), want %v", axStr, y.C, y.H, y.W, want)
		}
	}
}

func TestEvalGraphSoftmaxLayerNormAndResNet(t *testing.T) {
	// Softmax along W: x -> max[W] -> sub -> smul(log2e) -> lut(exp2) -> inv_sum[W] -> mul
	softmaxJSON := `{
		"target": "h17",
		"placement": "auto",
		"inputs": [{"name": "x", "c": 2, "h": 4, "w": 8}],
		"nodes": [
			{"id": "mx", "op": "reduce", "reduce_op": "max", "axes": "w", "inputs": ["x"]},
			{"id": "shifted", "op": "pe", "pe_op": "sub", "inputs": ["x", "mx"]},
			{"id": "scaled", "op": "pe", "pe_op": "smul", "scalar": 1.44269504, "inputs": ["shifted"]},
			{"id": "ex", "op": "lut", "act": "exp2", "inputs": ["scaled"]},
			{"id": "inv_s", "op": "reduce", "reduce_op": "inv_sum", "axes": "w", "inputs": ["ex"]},
			{"id": "out", "op": "pe", "pe_op": "mul", "inputs": ["ex", "inv_s"]}
		]
	}`
	var smSpec GraphSpec
	if err := json.Unmarshal([]byte(softmaxJSON), &smSpec); err != nil {
		t.Fatalf("unmarshal softmax: %v", err)
	}
	x := makeConstTensor(2, 4, 8, []float64{-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, -0.25, 0.75})
	ySM, _, err := EvalGraphF64(smSpec, map[string]Tensor{"x": x}, nil)
	if err != nil {
		t.Fatalf("EvalGraphF64 softmax: %v", err)
	}
	// Every row should sum to ~1.0
	for c := 0; c < ySM.C; c++ {
		for h := 0; h < ySM.H; h++ {
			sum := 0.0
			for w := 0; w < ySM.W; w++ {
				sum += fp16.ToFloat64(ySM.Data[ySM.Index(0, c, h, w)])
			}
			if math.Abs(sum-1.0) > 2e-2 {
				t.Fatalf("softmax row (%d,%d) sum = %v, want ~1.0", c, h, sum)
			}
		}
	}

	// LayerNorm along W: mean -> sub -> rsqrt_mean_square -> mul
	lnJSON := `{
		"target": "h16",
		"inputs": [{"name": "x", "c": 2, "h": 4, "w": 8}],
		"nodes": [
			{"id": "mu", "op": "reduce", "reduce_op": "mean", "axes": "w", "inputs": ["x"]},
			{"id": "xc", "op": "pe", "pe_op": "sub", "inputs": ["x", "mu"]},
			{"id": "inv_std", "op": "reduce", "reduce_op": "rsqrt_mean_square", "axes": "w", "eps": 0.001, "inputs": ["xc"]},
			{"id": "out", "op": "pe", "pe_op": "mul", "inputs": ["xc", "inv_std"]}
		]
	}`
	var lnSpec GraphSpec
	if err := json.Unmarshal([]byte(lnJSON), &lnSpec); err != nil {
		t.Fatalf("unmarshal layer_norm: %v", err)
	}
	yLN, _, err := EvalGraphF64(lnSpec, map[string]Tensor{"x": x}, nil)
	if err != nil {
		t.Fatalf("EvalGraphF64 layer_norm: %v", err)
	}
	for c := 0; c < yLN.C; c++ {
		for h := 0; h < yLN.H; h++ {
			mean := 0.0
			for w := 0; w < yLN.W; w++ {
				mean += fp16.ToFloat64(yLN.Data[yLN.Index(0, c, h, w)])
			}
			mean /= float64(yLN.W)
			if math.Abs(mean) > 2e-2 {
				t.Fatalf("layer_norm row (%d,%d) mean = %v, want ~0", c, h, mean)
			}
		}
	}

	// Pool + global avg_pool in graph
	poolJSON := `{
		"target": "h17",
		"inputs": [{"name": "x", "c": 4, "h": 8, "w": 8}],
		"nodes": [
			{"id": "p1", "op": "max_pool", "kh": 2, "kw": 2, "sh": 2, "sw": 2, "pad": "valid", "inputs": ["x"]},
			{"id": "gap", "op": "avg_pool", "kh": 4, "kw": 4, "sh": 1, "sw": 1, "pad": "valid", "inputs": ["p1"]}
		]
	}`
	var poolSpec GraphSpec
	if err := json.Unmarshal([]byte(poolJSON), &poolSpec); err != nil {
		t.Fatalf("unmarshal pool graph: %v", err)
	}
	xp := makeConstTensor(4, 8, 8, []float64{-0.5, 0.25, 1.0, 0.75})
	yP, _, err := EvalGraphQ1516(poolSpec, map[string]Tensor{"x": xp}, nil)
	if err != nil {
		t.Fatalf("EvalGraphQ1516 pool: %v", err)
	}
	if yP.C != 4 || yP.H != 1 || yP.W != 1 {
		t.Fatalf("pool graph output shape (%d,%d,%d), want (4,1,1)", yP.C, yP.H, yP.W)
	}
}
