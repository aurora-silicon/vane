package ref

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/niteria/vane/internal/fp16"
)

func makeConstTensor(c, h, w int, vals []float64) Tensor {
	t, _ := NewTensor(1, c, h, w)
	for i := range t.Data {
		t.Data[i] = fp16.FromFloat64(vals[i%len(vals)])
	}
	return t
}

func makeConstWeights(n int, vals []float64) []uint16 {
	out := make([]uint16, n)
	for i := range out {
		out[i] = fp16.FromFloat64(vals[i%len(vals)])
	}
	return out
}

func TestEvalGraphConvChainAndResidual(t *testing.T) {
	// Residual block: x -> c1 (conv3x3 dense + relu) -> c2 (conv1x1) -> add(c2, x)
	spec := GraphSpec{
		Target:    "h17",
		Placement: "auto",
		Inputs:    []GraphInputSpec{{Name: "x", C: 4, H: 4, W: 4}},
		Nodes: []GraphNodeSpec{
			{ID: "c1", Op: "conv3x3", Mode: "dense", Cout: 4, Stride: 1, Pad: "same", Relu: true, Inputs: []string{"x"}},
			{ID: "c2", Op: "conv1x1", Cout: 4, Inputs: []string{"c1"}},
			{ID: "res", Op: "pe", PeOp: "add", Inputs: []string{"c2", "x"}},
		},
	}

	x := makeConstTensor(4, 4, 4, []float64{0.5, -0.25, 0.75, 0.125})
	w1 := makeConstWeights(4*4*9, []float64{0.125, -0.0625, 0.0625})
	b1 := makeConstWeights(4, []float64{0.25, -0.125, 0.0, 0.125})
	w2 := makeConstWeights(4*4, []float64{0.25, -0.125, 0.5, 0.0625})

	nw := map[string]GraphNodeWeights{
		"c1": {Weights: w1, Bias: b1},
		"c2": {Weights: w2},
	}
	inps := map[string]Tensor{"x": x}

	gotQ, envQ, err := EvalGraphQ1516(spec, inps, nw)
	if err != nil {
		t.Fatalf("EvalGraphQ1516: %v", err)
	}
	gotF, _, err := EvalGraphF64(spec, inps, nw)
	if err != nil {
		t.Fatalf("EvalGraphF64: %v", err)
	}

	// Manually step Q15.16 to verify composition identity
	m1, err := Conv2d3x3Q1516(x, w1, 4, b1, 1, "same", 1, true)
	if err != nil {
		t.Fatalf("manual c1: %v", err)
	}
	m2, err := Conv2d1x1Q1516(m1, w2, 4, nil, false)
	if err != nil {
		t.Fatalf("manual c2: %v", err)
	}
	m3, err := PeBinary("add", m2, x, false)
	if err != nil {
		t.Fatalf("manual add: %v", err)
	}

	for i := range gotQ.Data {
		if gotQ.Data[i] != m3.Data[i] {
			t.Fatalf("Q15.16 mismatch at %d: got 0x%04x, want 0x%04x", i, gotQ.Data[i], m3.Data[i])
		}
	}
	if _, ok := envQ["c1"]; !ok {
		t.Fatalf("expected intermediate 'c1' in envQ")
	}

	// Q15.16 and F64 should closely agree on well-scaled inputs
	res, err := CompareFP16(gotQ.Data, gotF.Data, 2e-2, 2e-2, nil)
	if err != nil || !res.Passed {
		t.Fatalf("CompareFP16 Q15.16 vs F64 failed: err=%v passed=%t maxAbs=%g", err, res.Passed, res.MaxAbsErr)
	}
}

func TestEvalGraphMixedNePeLutAndBroadcast(t *testing.T) {
	// 2-input graph: x [4, 4, 8], z [4, 1, 1]
	// c1 = conv1x1(x, act=gelu) -> p1 = pe chain(in_scale=0.5, bias=0.25) -> l1 = lut(silu) -> out = pe add(l1, z)
	rawJSON := `{
		"target": "h17",
		"placement": "auto",
		"inputs": [
			{"name": "x", "c": 4, "h": 4, "w": 8},
			{"name": "z", "c": 4, "h": 1, "w": 1}
		],
		"nodes": [
			{"id": "c1", "op": "conv1x1", "inputs": ["x"], "cout": 4, "act": "gelu"},
			{"id": "p1", "op": "pe", "pe_op": "chain", "inputs": ["c1"], "in_scale": 0.5, "bias": 0.25},
			{"id": "l1", "op": "lut", "inputs": ["p1"], "act": {"name": "hard_sigmoid", "alpha": 0.2, "beta": 0.5}},
			{"id": "out", "op": "pe", "pe_op": "add", "inputs": ["l1", "z"]}
		]
	}`
	var spec GraphSpec
	if err := json.Unmarshal([]byte(rawJSON), &spec); err != nil {
		t.Fatalf("unmarshal graph JSON: %v", err)
	}
	x := makeConstTensor(4, 4, 8, []float64{-1.0, -0.25, 0.5, 1.25, 0.0})
	z := makeConstTensor(4, 1, 1, []float64{0.125, -0.125, 0.25, 0.5})
	w1 := makeConstWeights(4*4, []float64{0.25, -0.125, 0.5, 0.125})

	gotQ, _, err := EvalGraphQ1516(spec, map[string]Tensor{"x": x, "z": z}, map[string]GraphNodeWeights{
		"c1": {Weights: w1},
	})
	if err != nil {
		t.Fatalf("EvalGraphQ1516 mixed: %v", err)
	}
	gotF, _, err := EvalGraphF64(spec, map[string]Tensor{"x": x, "z": z}, map[string]GraphNodeWeights{
		"c1": {Weights: w1},
	})
	if err != nil {
		t.Fatalf("EvalGraphF64 mixed: %v", err)
	}
	if gotQ.C != 4 || gotQ.H != 4 || gotQ.W != 8 {
		t.Fatalf("unexpected output shape (%d,%d,%d)", gotQ.C, gotQ.H, gotQ.W)
	}
	res, err := CompareFP16(gotQ.Data, gotF.Data, 3e-2, 3e-2, nil)
	if err != nil || !res.Passed {
		t.Fatalf("mixed Q15.16 vs F64 mismatch: err=%v passed=%t maxAbs=%g", err, res.Passed, res.MaxAbsErr)
	}
}

func TestResolvePe1ChainSmulSdivPlacementAndGuards(t *testing.T) {
	f2 := 2.0
	inSc, b, sc, err := resolvePe1ChainParams(GraphNodeSpec{ID: "m", Op: "pe", PeOp: "smul", Scalar: &f2})
	if err != nil {
		t.Fatalf("smul failed: %v", err)
	}
	if inSc != nil || b != nil || sc == nil || *sc != fp16.FromFloat64(2.0) {
		t.Fatalf("smul expected scale=2.0 (inScale=nil, bias=nil), got inScale=%v bias=%v scale=%v", inSc, b, sc)
	}

	inSc, b, sc, err = resolvePe1ChainParams(GraphNodeSpec{ID: "d", Op: "pe", PeOp: "sdiv", Scalar: &f2})
	if err != nil {
		t.Fatalf("sdiv failed: %v", err)
	}
	if inSc != nil || b != nil || sc == nil || *sc != fp16.FromFloat64(0.5) {
		t.Fatalf("sdiv expected scale=0.5 (inScale=nil, bias=nil), got inScale=%v bias=%v scale=%v", inSc, b, sc)
	}

	// Non-finite fp16 divisors (+Inf, -Inf, NaN, and finite float64 overflowing fp16) must be rejected
	for _, bad := range []float64{math.Inf(1), math.Inf(-1), math.NaN(), 1e5, -1e5} {
		v := bad
		if _, _, _, err := resolvePe1ChainParams(GraphNodeSpec{ID: "bad", Op: "pe", PeOp: "sdiv", Scalar: &v}); err == nil {
			t.Fatalf("expected sdiv with scalar=%v to fail", bad)
		}
	}
}
