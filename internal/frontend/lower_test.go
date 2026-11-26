package frontend

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/aurora-silicon/vane/internal/fp16"
	"github.com/aurora-silicon/vane/internal/ref"
)

func loadRefGraphFromDir(t *testing.T, dir string) (ref.GraphSpec, map[string]ref.GraphNodeWeights) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, "graph.json"))
	if err != nil {
		t.Fatalf("read graph.json: %v", err)
	}
	var spec ref.GraphSpec
	if err := json.Unmarshal(raw, &spec); err != nil {
		t.Fatalf("unmarshal ref.GraphSpec: %v", err)
	}
	nw := make(map[string]ref.GraphNodeWeights)
	for _, nd := range spec.Nodes {
		op := strings.ToLower(nd.Op)
		if op != "conv1x1" && op != "conv3x3" {
			continue
		}
		wb, err := os.ReadFile(filepath.Join(dir, nd.Weights))
		if err != nil {
			t.Fatalf("read weights %s: %v", nd.Weights, err)
		}
		w16, err := fp16.DecodeLE(wb)
		if err != nil {
			t.Fatalf("decode weights %s: %v", nd.Weights, err)
		}
		var b16 []uint16
		wantsBias, bFile, err := nd.ConvBiasSpec()
		if err != nil {
			t.Fatalf("ConvBiasSpec(%s): %v", nd.ID, err)
		}
		if wantsBias && bFile != "" {
			bb, err := os.ReadFile(filepath.Join(dir, bFile))
			if err != nil {
				t.Fatalf("read bias %s: %v", bFile, err)
			}
			b16, err = fp16.DecodeLE(bb)
			if err != nil {
				t.Fatalf("decode bias %s: %v", bFile, err)
			}
		}
		nw[nd.ID] = ref.GraphNodeWeights{Weights: w16, Bias: b16}
	}
	return spec, nw
}

// Independent pure float64 NCHW reference primitives for validating lowered graphs.

type f64Tensor struct {
	C, H, W int
	Data    []float64
}

func newF64Tensor(c, h, w int) f64Tensor {
	return f64Tensor{C: c, H: h, W: w, Data: make([]float64, c*h*w)}
}

func (t f64Tensor) at(c, h, w int) float64 {
	return t.Data[(c*t.H+h)*t.W+w]
}

func (t f64Tensor) set(c, h, w int, v float64) {
	t.Data[(c*t.H+h)*t.W+w] = v
}

func (t f64Tensor) toRefFP16() ref.Tensor {
	out, _ := ref.NewTensor(1, t.C, t.H, t.W)
	for i, v := range t.Data {
		out.Data[i] = fp16.FromFloat64(v)
	}
	return out
}

// refConv3x3F64 evaluates an exact ONNX 3x3 convolution with explicit pads [pt, pl, pb, pr].
func refConv3x3F64(x f64Tensor, w []float64, b []float64, cout, stride int, pads [4]int) f64Tensor {
	pt, pl, pb, pr := pads[0], pads[1], pads[2], pads[3]
	ho := (x.H+pt+pb-3)/stride + 1
	wo := (x.W+pl+pr-3)/stride + 1
	out := newF64Tensor(cout, ho, wo)
	for o := 0; o < cout; o++ {
		bv := 0.0
		if b != nil {
			bv = b[o]
		}
		for oy := 0; oy < ho; oy++ {
			for ox := 0; ox < wo; ox++ {
				acc := bv
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
							acc += x.at(c, iy, ix) * w[wBase+ky*3+kx]
						}
					}
				}
				out.set(o, oy, ox, acc)
			}
		}
	}
	return out
}

func refConv1x1F64(x f64Tensor, w []float64, b []float64, cout int) f64Tensor {
	out := newF64Tensor(cout, x.H, x.W)
	for o := 0; o < cout; o++ {
		bv := 0.0
		if b != nil {
			bv = b[o]
		}
		for y := 0; y < x.H; y++ {
			for xi := 0; xi < x.W; xi++ {
				acc := bv
				for c := 0; c < x.C; c++ {
					acc += x.at(c, y, xi) * w[o*x.C+c]
				}
				out.set(o, y, xi, acc)
			}
		}
	}
	return out
}

func refBNF64(x f64Tensor, scale, b, mean, variance []float64, eps float64) f64Tensor {
	out := newF64Tensor(x.C, x.H, x.W)
	for c := 0; c < x.C; c++ {
		f := scale[c] / math.Sqrt(variance[c]+eps)
		for y := 0; y < x.H; y++ {
			for xi := 0; xi < x.W; xi++ {
				out.set(c, y, xi, (x.at(c, y, xi)-mean[c])*f+b[c])
			}
		}
	}
	return out
}

func refReluF64(x f64Tensor) f64Tensor {
	out := newF64Tensor(x.C, x.H, x.W)
	for i, v := range x.Data {
		if v > 0 {
			out.Data[i] = v
		}
	}
	return out
}

func refAddF64(a, b f64Tensor) f64Tensor {
	out := newF64Tensor(a.C, a.H, a.W)
	for i := range a.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return out
}

func toF32Slice(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

func assertTensorClose(t *testing.T, label string, got ref.Tensor, want f64Tensor, atol float64) {
	t.Helper()
	if got.C != want.C || got.H != want.H || got.W != want.W {
		t.Fatalf("%s: shape mismatch got (1,%d,%d,%d) != want (1,%d,%d,%d)", label, got.C, got.H, got.W, want.C, want.H, want.W)
	}
	for i := range want.Data {
		g := fp16.ToFloat64(got.Data[i])
		diff := math.Abs(g - want.Data[i])
		if diff > atol {
			t.Fatalf("%s: elem %d mismatch: got %.6g, want %.6g (diff=%.6g > atol=%.6g)", label, i, g, want.Data[i], diff, atol)
		}
	}
}

func TestEndToEndConvBNReluResidualBlock(t *testing.T) {
	// Graph: x [batch,4,4,4] -> Identity -> Conv3x3(4->4, pads=[1,1,1,1]) -> BN1 -> ReLU
	//        -> Dropout -> Conv1x1(4->4) -> BN2 -> Add(_, x) -> ReLU -> y
	const C, H, W = 4, 4, 4
	w1 := make([]float64, C*C*9)
	for i := range w1 {
		w1[i] = float64((i%7)-3) * 0.05
	}
	b1 := []float64{0.1, -0.05, 0.02, 0.08}
	bn1Scale := []float64{1.2, 0.8, 1.5, 0.9}
	bn1B := []float64{0.05, -0.1, 0.2, -0.05}
	bn1Mean := []float64{0.1, -0.2, 0.05, 0.0}
	bn1Var := []float64{0.5, 0.75, 1.0, 0.25}

	w2 := make([]float64, C*C)
	for i := range w2 {
		w2[i] = float64((i%5)-2) * 0.1
	}
	bn2Scale := []float64{0.9, 1.1, 0.7, 1.3}
	bn2B := []float64{-0.05, 0.1, -0.02, 0.04}
	bn2Mean := []float64{0.0, 0.15, -0.1, 0.05}
	bn2Var := []float64{0.6, 0.4, 0.9, 0.8}
	const eps = 1e-5

	rawONNX := buildModel(17, "resnet_block",
		[]*ProtoBuilder{buildValueInfo("x", ONNXFloat, []ONNXDim{
			{Param: "batch"}, {HasValue: true, Value: C}, {HasValue: true, Value: H}, {HasValue: true, Value: W},
		})},
		[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, C, H, W})},
		[]*ProtoBuilder{
			buildFloatTensor("w1", []int64{C, C, 3, 3}, toF32Slice(w1), true),
			buildFloatTensor("b1", []int64{C}, toF32Slice(b1), true),
			buildFloatTensor("bn1_s", []int64{C}, toF32Slice(bn1Scale), true),
			buildFloatTensor("bn1_b", []int64{C}, toF32Slice(bn1B), true),
			buildFloatTensor("bn1_m", []int64{C}, toF32Slice(bn1Mean), true),
			buildFloatTensor("bn1_v", []int64{C}, toF32Slice(bn1Var), true),
			buildFloatTensor("w2", []int64{C, C, 1, 1}, toF32Slice(w2), true),
			buildFloatTensor("bn2_s", []int64{C}, toF32Slice(bn2Scale), true),
			buildFloatTensor("bn2_b", []int64{C}, toF32Slice(bn2B), true),
			buildFloatTensor("bn2_m", []int64{C}, toF32Slice(bn2Mean), true),
			buildFloatTensor("bn2_v", []int64{C}, toF32Slice(bn2Var), true),
		},
		[]*ProtoBuilder{
			buildNode("Identity", "id0", []string{"x"}, []string{"x_id"}),
			buildNode("Conv", "conv1", []string{"x_id", "w1", "b1"}, []string{"c1"},
				buildAttrInts("strides", []int64{1, 1}, true),
				buildAttrInts("pads", []int64{1, 1, 1, 1}, true),
			),
			buildNode("BatchNormalization", "bn1", []string{"c1", "bn1_s", "bn1_b", "bn1_m", "bn1_v"}, []string{"b1_out"},
				buildAttrFloat("epsilon", eps),
			),
			buildNode("Relu", "relu1", []string{"b1_out"}, []string{"r1"}),
			buildNode("Dropout", "drop1", []string{"r1"}, []string{"d1"}),
			buildNode("Conv", "conv2", []string{"d1", "w2"}, []string{"c2"},
				buildAttrInts("strides", []int64{1, 1}, true),
				buildAttrInts("pads", []int64{0, 0, 0, 0}, true),
			),
			buildNode("BatchNormalization", "bn2", []string{"c2", "bn2_s", "bn2_b", "bn2_m", "bn2_v"}, []string{"b2_out"},
				buildAttrFloat("epsilon", eps),
			),
			buildNode("Add", "res_add", []string{"b2_out", "x"}, []string{"sum"}),
			buildNode("Relu", "relu_out", []string{"sum"}, []string{"y"}),
		},
	)

	model, err := ParseONNX(rawONNX)
	if err != nil {
		t.Fatalf("ParseONNX: %v", err)
	}

	if _, err := LowerONNX(model, LowerOptions{}); err == nil {
		t.Fatal("expected error for dynamic batch dim without InputShapes override")
	}

	res, err := LowerONNX(model, LowerOptions{
		Target:      "h17",
		InputShapes: map[string][3]int{"x": {C, H, W}},
	})
	if err != nil {
		t.Fatalf("LowerONNX fused: %v", err)
	}
	if res.Summary.FusedBNCount != 2 || res.Summary.FusedActCount != 1 || res.Summary.ElidedCount != 2 {
		t.Fatalf("unexpected fusion stats: %+v", res.Summary)
	}
	if len(res.Spec.Nodes) != 4 || !res.Spec.Nodes[0].Relu {
		t.Fatalf("expected 4 fused nodes with conv1 relu=true, got %+v", res.Spec.Nodes)
	}

	dir := t.TempDir()
	if err := WriteLoweredGraph(dir, res); err != nil {
		t.Fatalf("WriteLoweredGraph: %v", err)
	}
	refSpec, nw := loadRefGraphFromDir(t, dir)

	xF64 := newF64Tensor(C, H, W)
	for i := range xF64.Data {
		xF64.Data[i] = fp16.ToFloat64(fp16.FromFloat64(float64((i%11)-5) * 0.25))
	}
	xFP16 := xF64.toRefFP16()

	want := refConv3x3F64(xF64, w1, b1, C, 1, [4]int{1, 1, 1, 1})
	want = refBNF64(want, bn1Scale, bn1B, bn1Mean, bn1Var, eps)
	want = refReluF64(want)
	want = refConv1x1F64(want, w2, nil, C)
	want = refBNF64(want, bn2Scale, bn2B, bn2Mean, bn2Var, eps)
	want = refAddF64(want, xF64)
	want = refReluF64(want)

	gotF64, _, err := ref.EvalGraphF64(refSpec, map[string]ref.Tensor{"x": xFP16}, nw)
	if err != nil {
		t.Fatalf("EvalGraphF64 fused: %v", err)
	}
	gotQ, _, err := ref.EvalGraphQ1516(refSpec, map[string]ref.Tensor{"x": xFP16}, nw)
	if err != nil {
		t.Fatalf("EvalGraphQ1516 fused: %v", err)
	}
	assertTensorClose(t, "resnet fused F64", gotF64, want, 2e-2)
	assertTensorClose(t, "resnet fused Q1516", gotQ, want, 3e-2)

	resNoFuse, err := LowerONNX(model, LowerOptions{
		Target:      "h17",
		InputShapes: map[string][3]int{"x": {C, H, W}},
		NoFuse:      true,
	})
	if err != nil {
		t.Fatalf("LowerONNX no-fuse: %v", err)
	}
	if resNoFuse.Summary.FusedBNCount != 0 || resNoFuse.Summary.FusedActCount != 0 || len(resNoFuse.Spec.Nodes) != 7 {
		t.Fatalf("unexpected no-fuse summary: %+v", resNoFuse.Summary)
	}
	dirNF := t.TempDir()
	if err := WriteLoweredGraph(dirNF, resNoFuse); err != nil {
		t.Fatalf("WriteLoweredGraph no-fuse: %v", err)
	}
	refSpecNF, nwNF := loadRefGraphFromDir(t, dirNF)
	gotNF, _, err := ref.EvalGraphF64(refSpecNF, map[string]ref.Tensor{"x": xFP16}, nwNF)
	if err != nil {
		t.Fatalf("EvalGraphF64 no-fuse: %v", err)
	}
	assertTensorClose(t, "resnet unfused F64", gotNF, want, 2e-2)
}

func TestConv3x3Stride2PaddingValidation(t *testing.T) {
	buildStride2Conv := func(h, w int, autoPad string, pads []int64, weights []float64) *ONNXModel {
		var attrs []*ProtoBuilder
		attrs = append(attrs, buildAttrInts("strides", []int64{2, 2}, true))
		if autoPad != "" {
			attrs = append(attrs, buildAttrString("auto_pad", autoPad))
		}
		if pads != nil {
			attrs = append(attrs, buildAttrInts("pads", pads, true))
		}
		raw := buildModel(17, "s2_conv",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, int64(h), int64(w)})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 1, 1})},
			[]*ProtoBuilder{buildFloatTensor("w", []int64{2, 2, 3, 3}, toF32Slice(weights), true)},
			[]*ProtoBuilder{buildNode("Conv", "c", []string{"x", "w"}, []string{"y"}, attrs...)},
		)
		m, err := ParseONNX(raw)
		if err != nil {
			t.Fatalf("ParseONNX: %v", err)
		}
		return m
	}

	weights := make([]float64, 2*2*9)
	for i := range weights {
		weights[i] = float64((i%7)-3) * 0.1
	}

	// Reject mismatched stride-2 paddings that would silently miscompile:
	t.Run("reject_even_4x4_pads_1111", func(t *testing.T) {
		m := buildStride2Conv(4, 4, "", []int64{1, 1, 1, 1}, weights)
		if _, err := LowerONNX(m, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "SAME_UPPER") {
			t.Fatalf("expected rejection for stride-2 pads=[1,1,1,1] on 4x4, got: %v", err)
		}
	})
	t.Run("reject_even_4x4_same_lower", func(t *testing.T) {
		m := buildStride2Conv(4, 4, "SAME_LOWER", nil, weights)
		if _, err := LowerONNX(m, LowerOptions{}); err == nil {
			t.Fatal("expected rejection for stride-2 SAME_LOWER on 4x4")
		}
	})
	t.Run("reject_odd_5x5_pads_0011", func(t *testing.T) {
		m := buildStride2Conv(5, 5, "", []int64{0, 0, 1, 1}, weights)
		if _, err := LowerONNX(m, LowerOptions{}); err == nil {
			t.Fatal("expected rejection for stride-2 pads=[0,0,1,1] on odd 5x5")
		}
	})

	// Accept exact SAME_UPPER matches and verify numerical accuracy against exact ONNX pads:
	t.Run("pass_even_4x4_pads_0011", func(t *testing.T) {
		m := buildStride2Conv(4, 4, "", []int64{0, 0, 1, 1}, weights)
		res, err := LowerONNX(m, LowerOptions{})
		if err != nil {
			t.Fatalf("LowerONNX even 4x4 pads=[0,0,1,1]: %v", err)
		}
		if res.Spec.Nodes[0].Pad != "same" || res.Summary.OutputShape != [4]int{1, 2, 2, 2} {
			t.Fatalf("unexpected pad/shape: %+v %v", res.Spec.Nodes[0], res.Summary.OutputShape)
		}
		dir := t.TempDir()
		if err := WriteLoweredGraph(dir, res); err != nil {
			t.Fatalf("WriteLoweredGraph: %v", err)
		}
		refSpec, nw := loadRefGraphFromDir(t, dir)
		xF64 := newF64Tensor(2, 4, 4)
		for i := range xF64.Data {
			xF64.Data[i] = fp16.ToFloat64(fp16.FromFloat64(float64((i%9)-4) * 0.25))
		}
		got, _, err := ref.EvalGraphF64(refSpec, map[string]ref.Tensor{"x": xF64.toRefFP16()}, nw)
		if err != nil {
			t.Fatalf("EvalGraphF64: %v", err)
		}
		want := refConv3x3F64(xF64, weights, nil, 2, 2, [4]int{0, 0, 1, 1})
		assertTensorClose(t, "stride2 even 4x4 [0,0,1,1]", got, want, 1e-2)
	})

	t.Run("pass_odd_5x5_pads_1111", func(t *testing.T) {
		m := buildStride2Conv(5, 5, "", []int64{1, 1, 1, 1}, weights)
		res, err := LowerONNX(m, LowerOptions{})
		if err != nil {
			t.Fatalf("LowerONNX odd 5x5 pads=[1,1,1,1]: %v", err)
		}
		if res.Spec.Nodes[0].Pad != "same" || res.Summary.OutputShape != [4]int{1, 2, 3, 3} {
			t.Fatalf("unexpected pad/shape: %+v %v", res.Spec.Nodes[0], res.Summary.OutputShape)
		}
		dir := t.TempDir()
		if err := WriteLoweredGraph(dir, res); err != nil {
			t.Fatalf("WriteLoweredGraph: %v", err)
		}
		refSpec, nw := loadRefGraphFromDir(t, dir)
		xF64 := newF64Tensor(2, 5, 5)
		for i := range xF64.Data {
			xF64.Data[i] = fp16.ToFloat64(fp16.FromFloat64(float64((i%11)-5) * 0.25))
		}
		got, _, err := ref.EvalGraphF64(refSpec, map[string]ref.Tensor{"x": xF64.toRefFP16()}, nw)
		if err != nil {
			t.Fatalf("EvalGraphF64: %v", err)
		}
		want := refConv3x3F64(xF64, weights, nil, 2, 2, [4]int{1, 1, 1, 1})
		assertTensorClose(t, "stride2 odd 5x5 [1,1,1,1]", got, want, 1e-2)
	})
}

func TestEndToEndPoolingReduceSoftmaxBlock(t *testing.T) {
	// x [1, 4, 4, 4] -> MaxPool(2x2, stride=2, valid) [1,4,2,2]
	//                -> AveragePool(2x2, stride=1, pads=[1,1,1,1], count_include_pad=0) [1,4,3,3]
	//                -> Softmax(axis=1) [1,4,3,3] (W=3 > 1 -> reduce(sum) + lut(inverse)!)
	//                -> GlobalAveragePool [1,4,1,1]
	const C, H, W = 4, 4, 4
	rawONNX := buildModel(17, "pool_reduce_softmax",
		[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, C, H, W})},
		[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, C, 1, 1})},
		nil,
		[]*ProtoBuilder{
			buildNode("MaxPool", "mp", []string{"x"}, []string{"p1"},
				buildAttrInts("kernel_shape", []int64{2, 2}, true),
				buildAttrInts("strides", []int64{2, 2}, true),
			),
			buildNode("AveragePool", "ap", []string{"p1"}, []string{"p2"},
				buildAttrInts("kernel_shape", []int64{2, 2}, true),
				buildAttrInts("strides", []int64{1, 1}, true),
				buildAttrInts("pads", []int64{1, 1, 1, 1}, true),
				buildAttrInt("count_include_pad", 0),
			),
			buildNode("Softmax", "sm", []string{"p2"}, []string{"p3"},
				buildAttrInt("axis", 1),
			),
			buildNode("GlobalAveragePool", "gap", []string{"p3"}, []string{"y"}),
		},
	)

	model, err := ParseONNX(rawONNX)
	if err != nil {
		t.Fatalf("ParseONNX: %v", err)
	}
	res, err := LowerONNX(model, LowerOptions{Target: "h17"})
	if err != nil {
		t.Fatalf("LowerONNX: %v", err)
	}
	if res.Summary.OutputShape != [4]int{1, C, 1, 1} {
		t.Fatalf("unexpected output shape: %v", res.Summary.OutputShape)
	}
	// Verify Softmax(axis=1) with W=3 > 1 used reduce(sum)+lut(inverse), NOT reduce(inv_sum).
	sawSum, sawInv := false, false
	for _, nd := range res.Spec.Nodes {
		if nd.Op == "reduce" && nd.Axes == "c" && nd.ReduceOp == "inv_sum" {
			t.Fatalf("Softmax(axis=1) with W>1 must not emit reduce(inv_sum, axes='c')")
		}
		if nd.Op == "reduce" && nd.Axes == "c" && nd.ReduceOp == "sum" {
			sawSum = true
		}
		if nd.Op == "lut" && nd.Act != nil && nd.Act.Name == "inverse" {
			sawInv = true
		}
	}
	if !sawSum || !sawInv {
		t.Fatalf("expected reduce(sum, axes='c') + lut(inverse) for Softmax(axis=1, W>1); got %+v", res.Spec.Nodes)
	}

	dir := t.TempDir()
	if err := WriteLoweredGraph(dir, res); err != nil {
		t.Fatalf("WriteLoweredGraph: %v", err)
	}
	refSpec, nw := loadRefGraphFromDir(t, dir)

	xF64 := newF64Tensor(C, H, W)
	for i := range xF64.Data {
		xF64.Data[i] = fp16.ToFloat64(fp16.FromFloat64(float64((i%9)-4) * 0.35))
	}
	gotF64, _, err := ref.EvalGraphF64(refSpec, map[string]ref.Tensor{"x": xF64.toRefFP16()}, nw)
	if err != nil {
		t.Fatalf("EvalGraphF64: %v", err)
	}

	// Independent float64 reference:
	p1 := newF64Tensor(C, 2, 2)
	for c := 0; c < C; c++ {
		for oy := 0; oy < 2; oy++ {
			for ox := 0; ox < 2; ox++ {
				best := math.Inf(-1)
				for ky := 0; ky < 2; ky++ {
					for kx := 0; kx < 2; kx++ {
						if v := xF64.at(c, oy*2+ky, ox*2+kx); v > best {
							best = v
						}
					}
				}
				p1.set(c, oy, ox, best)
			}
		}
	}
	p2 := newF64Tensor(C, 3, 3)
	for c := 0; c < C; c++ {
		for oy := 0; oy < 3; oy++ {
			for ox := 0; ox < 3; ox++ {
				sum, cnt := 0.0, 0
				for ky := 0; ky < 2; ky++ {
					iy := oy + ky - 1
					if iy < 0 || iy >= 2 {
						continue
					}
					for kx := 0; kx < 2; kx++ {
						ix := ox + kx - 1
						if ix < 0 || ix >= 2 {
							continue
						}
						sum += p1.at(c, iy, ix)
						cnt++
					}
				}
				p2.set(c, oy, ox, sum/float64(cnt))
			}
		}
	}
	p3 := newF64Tensor(C, 3, 3)
	for y := 0; y < 3; y++ {
		for xi := 0; xi < 3; xi++ {
			mx := math.Inf(-1)
			for c := 0; c < C; c++ {
				if v := p2.at(c, y, xi); v > mx {
					mx = v
				}
			}
			sum := 0.0
			for c := 0; c < C; c++ {
				sum += math.Exp(p2.at(c, y, xi) - mx)
			}
			for c := 0; c < C; c++ {
				p3.set(c, y, xi, math.Exp(p2.at(c, y, xi)-mx)/sum)
			}
		}
	}
	want := newF64Tensor(C, 1, 1)
	for c := 0; c < C; c++ {
		s := 0.0
		for y := 0; y < 3; y++ {
			for xi := 0; xi < 3; xi++ {
				s += p3.at(c, y, xi)
			}
		}
		want.set(c, 0, 0, s/9.0)
	}
	assertTensorClose(t, "pool+reduce+softmax", gotF64, want, 1e-2)
}

func TestSoftmaxOpset11AndBackendConstraints(t *testing.T) {
	// Softmax opset 11 axis=1 flattens [1..3] -> axes="chw"
	raw11 := buildModel(11, "sm11",
		[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 2, 2})},
		[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
		nil,
		[]*ProtoBuilder{buildNode("Softmax", "sm", []string{"x"}, []string{"y"}, buildAttrInt("axis", 1))},
	)
	m11, _ := ParseONNX(raw11)
	res11, err := LowerONNX(m11, LowerOptions{})
	if err != nil {
		t.Fatalf("LowerONNX opset 11 Softmax(axis=1): %v", err)
	}
	if res11.Spec.Nodes[0].Axes != "chw" {
		t.Fatalf("expected opset 11 Softmax(axis=1) to reduce 'chw', got %q", res11.Spec.Nodes[0].Axes)
	}
	dir := t.TempDir()
	_ = WriteLoweredGraph(dir, res11)
	refSpec, nw := loadRefGraphFromDir(t, dir)
	xF64 := newF64Tensor(2, 2, 2)
	for i := range xF64.Data {
		xF64.Data[i] = float64(i-3) * 0.25
	}
	got, _, err := ref.EvalGraphF64(refSpec, map[string]ref.Tensor{"x": xF64.toRefFP16()}, nw)
	if err != nil {
		t.Fatalf("EvalGraphF64 opset11 softmax: %v", err)
	}
	mx := math.Inf(-1)
	for _, v := range xF64.Data {
		if v > mx {
			mx = v
		}
	}
	sum := 0.0
	for _, v := range xF64.Data {
		sum += math.Exp(v - mx)
	}
	want := newF64Tensor(2, 2, 2)
	for i, v := range xF64.Data {
		want.Data[i] = math.Exp(v-mx) / sum
	}
	assertTensorClose(t, "opset11 softmax chw", got, want, 1e-2)

	// Softmax opset 17 axis=2 (axes='h') with W > 1 must be rejected cleanly.
	rawH := buildModel(17, "sm_h",
		[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 3, 3})},
		[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 3, 3})},
		nil,
		[]*ProtoBuilder{buildNode("Softmax", "sm_h", []string{"x"}, []string{"y"}, buildAttrInt("axis", 2))},
	)
	mH, _ := ParseONNX(rawH)
	if _, err := LowerONNX(mH, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "reduce backend") {
		t.Fatalf("expected reduce backend error for Softmax(axis=2) with W>1, got: %v", err)
	}
}

func TestViewOpsGatingAndNoOpElision(t *testing.T) {
	// Non-no-op Concat U-Net skip block requires AllowViewOps: true; default rejects cleanly.
	wA := make([]float64, 4*4*9)
	wB := make([]float64, 2*4*9)
	wOut := make([]float64, 4*6)
	rawUNet := buildModel(17, "unet_skip",
		[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 4, 4})},
		[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 4, 2, 2})},
		[]*ProtoBuilder{
			buildFloatTensor("wA", []int64{4, 4, 3, 3}, toF32Slice(wA), true),
			buildFloatTensor("wB", []int64{2, 4, 3, 3}, toF32Slice(wB), true),
			buildFloatTensor("wOut", []int64{4, 6, 1, 1}, toF32Slice(wOut), true),
		},
		[]*ProtoBuilder{
			buildNode("Conv", "enc_conv", []string{"x", "wA"}, []string{"a0"},
				buildAttrInts("pads", []int64{1, 1, 1, 1}, true)),
			buildNode("Sigmoid", "enc_sig", []string{"a0"}, []string{"a0_sig"}),
			buildNode("Mul", "enc_silu", []string{"a0", "a0_sig"}, []string{"a1"}),
			buildNode("MaxPool", "enc_pool", []string{"a1"}, []string{"a2"},
				buildAttrInts("kernel_shape", []int64{2, 2}, true),
				buildAttrInts("strides", []int64{2, 2}, true)),
			buildNode("Conv", "skip_conv", []string{"x", "wB"}, []string{"b1"},
				buildAttrInts("strides", []int64{2, 2}, true),
				buildAttrInts("pads", []int64{0, 0, 1, 1}, true)),
			buildNode("Concat", "skip_cat", []string{"a2", "b1"}, []string{"cat"},
				buildAttrInt("axis", 1)),
			buildNode("Conv", "out_conv", []string{"cat", "wOut"}, []string{"c_out"}),
			buildNode("Relu", "out_relu", []string{"c_out"}, []string{"y"}),
		},
	)
	mUNet, err := ParseONNX(rawUNet)
	if err != nil {
		t.Fatalf("ParseONNX: %v", err)
	}
	if _, err := LowerONNX(mUNet, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "--allow-view-ops") {
		t.Fatalf("expected --allow-view-ops error when AllowViewOps=false, got: %v", err)
	}
	resUNet, err := LowerONNX(mUNet, LowerOptions{AllowViewOps: true})
	if err != nil {
		t.Fatalf("LowerONNX with AllowViewOps=true: %v", err)
	}
	if resUNet.Summary.FusedActCount != 2 || len(resUNet.Spec.Nodes) != 5 {
		t.Fatalf("unexpected U-Net summary: %+v", resUNet.Summary)
	}
	if resUNet.Spec.Nodes[0].Act == nil || resUNet.Spec.Nodes[0].Act.Name != "silu" {
		t.Fatalf("expected enc_conv fused act=silu, got %+v", resUNet.Spec.Nodes[0].Act)
	}
	if resUNet.Spec.Nodes[3].Op != "concat" || resUNet.Spec.Nodes[3].Axis == nil || *resUNet.Spec.Nodes[3].Axis != 1 {
		t.Fatalf("expected concat axis=1 at node 3, got %+v", resUNet.Spec.Nodes[3])
	}

	// True no-op views (Transpose [0,1,2,3], Reshape same shape, Flatten on [1,C,1,1]) MUST
	// be elided even when AllowViewOps == false!
	rawNoOp := buildModel(17, "noop_views",
		[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 1, 1})},
		[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 4, 1, 1})},
		[]*ProtoBuilder{
			buildInt64Tensor("same_shape", []int64{4}, []int64{1, 4, 1, 1}, false),
		},
		[]*ProtoBuilder{
			buildNode("Transpose", "tr_id", []string{"x"}, []string{"h0"}, buildAttrInts("perm", []int64{0, 1, 2, 3}, true)),
			buildNode("Reshape", "rs_id", []string{"h0", "same_shape"}, []string{"h1"}),
			buildNode("Flatten", "fl_id", []string{"h1"}, []string{"h2"}, buildAttrInt("axis", 1)),
			buildNode("Relu", "r", []string{"h2"}, []string{"y"}),
		},
	)
	mNoOp, _ := ParseONNX(rawNoOp)
	resNoOp, err := LowerONNX(mNoOp, LowerOptions{AllowViewOps: false})
	if err != nil {
		t.Fatalf("true no-op views should elide when AllowViewOps=false: %v", err)
	}
	if len(resNoOp.Spec.Nodes) != 1 || resNoOp.Spec.Nodes[0].Op != "lut" || resNoOp.Summary.ElidedCount != 3 {
		t.Fatalf("expected 3 elided no-op views and 1 relu node, got nodes=%+v summary=%+v", resNoOp.Spec.Nodes, resNoOp.Summary)
	}
}

func TestEndToEndSafeTensorsBackedModelAndPrecedence(t *testing.T) {
	// 1. SafeTensors-backed Conv+BN+Gelu model (with EXTERNAL ONNX initializers).
	w1 := []float64{
		0.5, -0.25, 0.125, 0.25,
		-0.5, 0.75, 0.0, -0.125,
		0.25, 0.25, -0.5, 0.5,
		0.125, -0.5, 0.25, 0.25,
	}
	bnS := []float64{1.0, 2.0, 0.5, 1.5}
	bnB := []float64{0.1, -0.2, 0.0, 0.05}
	bnM := []float64{0.0, 0.1, -0.1, 0.05}
	bnV := []float64{0.25, 1.0, 0.5, 0.75}

	sfBlob, err := WriteSafeTensors([]SafeTensorWrite{
		{Name: "conv.weight", DType: "F32", Shape: []int64{4, 4, 1, 1}, Data: encodeF32LE(toF32Slice(w1))},
		{Name: "bn.weight", DType: "F32", Shape: []int64{4}, Data: encodeF32LE(toF32Slice(bnS))},
		{Name: "bn.bias", DType: "BF16", Shape: []int64{4}, Data: encodeBF16LE(toF32Slice(bnB))},
		{Name: "bn.mean", DType: "F32", Shape: []int64{4}, Data: encodeF32LE(toF32Slice(bnM))},
		{Name: "bn.var", DType: "F32", Shape: []int64{4}, Data: encodeF32LE(toF32Slice(bnV))},
	}, map[string]string{"source": "test"})
	if err != nil {
		t.Fatalf("WriteSafeTensors: %v", err)
	}
	sf, err := ParseSafeTensors(sfBlob)
	if err != nil {
		t.Fatalf("ParseSafeTensors: %v", err)
	}

	mkExtInit := func(name string, dims []int64) *ProtoBuilder {
		var tb ProtoBuilder
		tb.WritePackedInt64(1, dims)
		tb.WriteInt64(2, int64(ONNXFloat))
		tb.WriteString(8, name)
		tb.WriteInt64(14, int64(ONNXDataExternal))
		return &tb
	}

	rawONNX := buildModel(17, "sf_model",
		[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 2, 2})},
		[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 4, 2, 2})},
		[]*ProtoBuilder{
			mkExtInit("conv.weight", []int64{4, 4, 1, 1}),
			mkExtInit("bn.weight", []int64{4}),
			mkExtInit("bn.bias", []int64{4}),
			mkExtInit("bn.mean", []int64{4}),
			mkExtInit("bn.var", []int64{4}),
		},
		[]*ProtoBuilder{
			buildNode("Conv", "c1", []string{"x", "conv.weight"}, []string{"h1"}),
			buildNode("BatchNormalization", "bn1", []string{"h1", "bn.weight", "bn.bias", "bn.mean", "bn.var"}, []string{"h2"},
				buildAttrFloat("epsilon", 1e-5)),
			buildNode("Gelu", "g1", []string{"h2"}, []string{"y"}),
		},
	)

	model, err := ParseONNX(rawONNX)
	if err != nil {
		t.Fatalf("ParseONNX: %v", err)
	}
	if _, err := LowerONNX(model, LowerOptions{}); err == nil {
		t.Fatal("expected error for EXTERNAL initializers without SafeTensors override")
	}

	res, err := LowerONNX(model, LowerOptions{SafeTensors: sf})
	if err != nil {
		t.Fatalf("LowerONNX with SafeTensors: %v", err)
	}
	if res.Summary.SafeTensorsUsed != 5 || res.Summary.FusedBNCount != 1 || res.Summary.FusedActCount != 1 || len(res.Spec.Nodes) != 1 {
		t.Fatalf("unexpected SafeTensors lowering summary: %+v", res.Summary)
	}

	dir := t.TempDir()
	_ = WriteLoweredGraph(dir, res)
	refSpec, nw := loadRefGraphFromDir(t, dir)

	xF64 := newF64Tensor(4, 2, 2)
	for i := range xF64.Data {
		xF64.Data[i] = fp16.ToFloat64(fp16.FromFloat64(float64((i%7)-3) * 0.25))
	}
	got, _, err := ref.EvalGraphF64(refSpec, map[string]ref.Tensor{"x": xF64.toRefFP16()}, nw)
	if err != nil {
		t.Fatalf("EvalGraphF64: %v", err)
	}
	bnBExact, _ := sf.ToFloat64("bn.bias")
	want := refBNF64(refConv1x1F64(xF64, w1, nil, 4), bnS, bnBExact, bnM, bnV, 1e-5)
	for i, v := range want.Data {
		want.Data[i] = 0.5 * v * (1.0 + math.Erf(v/math.Sqrt2))
	}
	assertTensorClose(t, "safetensors conv+bn+gelu", got, want, 1e-2)

	// 2. Weight listed in graph.input (not graph.initializer) supplied by SafeTensors must be skipped as runtime input.
	rawGraphInpWeight := buildModel(17, "sf_in_graph_input",
		[]*ProtoBuilder{
			buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 2, 2}),
			buildStaticValueInfo("conv.weight", ONNXFloat, []int64{4, 4, 1, 1}),
		},
		[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 4, 2, 2})},
		nil,
		[]*ProtoBuilder{buildNode("Conv", "c1", []string{"x", "conv.weight"}, []string{"y"})},
	)
	mGI, _ := ParseONNX(rawGraphInpWeight)
	resGI, err := LowerONNX(mGI, LowerOptions{SafeTensors: sf})
	if err != nil {
		t.Fatalf("SafeTensors weight in graph.input should succeed: %v", err)
	}
	if len(resGI.Spec.Inputs) != 1 || resGI.Spec.Inputs[0].Name != "x" {
		t.Fatalf("expected only runtime input 'x', got %+v", resGI.Spec.Inputs)
	}

	// 3. Constant node must NOT overwrite SafeTensors override of the same name.
	sfScalarBlob, _ := WriteSafeTensors([]SafeTensorWrite{
		{Name: "scale_c", DType: "F32", Shape: []int64{}, Data: encodeF32LE([]float32{3.0})},
	}, nil)
	sfScalar, _ := ParseSafeTensors(sfScalarBlob)
	rawConstOverride := buildModel(17, "const_override",
		[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 2, 2})},
		[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
		nil,
		[]*ProtoBuilder{
			buildNode("Constant", "c_node", nil, []string{"scale_c"}, buildAttrFloat("value_float", 99.0)),
			buildNode("Mul", "m", []string{"x", "scale_c"}, []string{"y"}),
		},
	)
	mCO, _ := ParseONNX(rawConstOverride)
	resCO, err := LowerONNX(mCO, LowerOptions{SafeTensors: sfScalar})
	if err != nil {
		t.Fatalf("SafeTensors override on Constant node failed: %v", err)
	}
	if resCO.Spec.Nodes[0].Scalar == nil || *resCO.Spec.Nodes[0].Scalar != 3.0 {
		t.Fatalf("expected SafeTensors scalar 3.0 to win over Constant 99.0, got %+v", resCO.Spec.Nodes[0].Scalar)
	}

	// 4. Scalar ONNX initializer (dims=[]) overridden by non-scalar [1] SafeTensors tensor must fail shape check.
	sfVec1Blob, _ := WriteSafeTensors([]SafeTensorWrite{
		{Name: "s", DType: "F32", Shape: []int64{1}, Data: encodeF32LE([]float32{2.0})},
	}, nil)
	sfVec1, _ := ParseSafeTensors(sfVec1Blob)
	rawScalarInit := buildModel(17, "scalar_init_mismatch",
		[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 2, 2})},
		[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
		[]*ProtoBuilder{buildFloatTensor("s", []int64{}, []float32{2.0}, false)},
		[]*ProtoBuilder{buildNode("Mul", "m", []string{"x", "s"}, []string{"y"})},
	)
	mSI, _ := ParseONNX(rawScalarInit)
	if _, err := LowerONNX(mSI, LowerOptions{SafeTensors: sfVec1}); err == nil || !strings.Contains(err.Error(), "shape") {
		t.Fatalf("expected shape mismatch error for scalar [] vs [1] SafeTensors override, got: %v", err)
	}
}

func TestMultiConsumerNonFusionAndPoolSameLower(t *testing.T) {
	t.Run("conv_multi_consumer_relu_and_add_not_fused", func(t *testing.T) {
		raw := buildModel(17, "mc_relu",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildFloatTensor("w", []int64{2, 2, 1, 1}, []float32{1, 0, 0, 1}, true)},
			[]*ProtoBuilder{
				buildNode("Conv", "c1", []string{"x", "w"}, []string{"c_out"}),
				buildNode("Relu", "r1", []string{"c_out"}, []string{"r_out"}),
				buildNode("Add", "a1", []string{"r_out", "c_out"}, []string{"y"}),
			},
		)
		m, _ := ParseONNX(raw)
		res, err := LowerONNX(m, LowerOptions{})
		if err != nil {
			t.Fatalf("LowerONNX: %v", err)
		}
		if res.Summary.FusedActCount != 0 || len(res.Spec.Nodes) != 3 || res.Spec.Nodes[0].Relu {
			t.Fatalf("expected ReLU NOT fused on multi-consumer Conv, got summary=%+v nodes=%+v", res.Summary, res.Spec.Nodes)
		}
	})

	t.Run("conv_multi_consumer_bn_and_add_not_folded", func(t *testing.T) {
		raw := buildModel(17, "mc_bn",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{
				buildFloatTensor("w", []int64{2, 2, 1, 1}, []float32{1, 0, 0, 1}, true),
				buildFloatTensor("s", []int64{2}, []float32{1, 1}, true),
				buildFloatTensor("b", []int64{2}, []float32{0, 0}, true),
				buildFloatTensor("m", []int64{2}, []float32{0, 0}, true),
				buildFloatTensor("v", []int64{2}, []float32{1, 1}, true),
			},
			[]*ProtoBuilder{
				buildNode("Conv", "c1", []string{"x", "w"}, []string{"c_out"}),
				buildNode("BatchNormalization", "bn1", []string{"c_out", "s", "b", "m", "v"}, []string{"bn_out"}),
				buildNode("Add", "a1", []string{"bn_out", "c_out"}, []string{"y"}),
			},
		)
		m, _ := ParseONNX(raw)
		res, err := LowerONNX(m, LowerOptions{})
		if err != nil {
			t.Fatalf("LowerONNX: %v", err)
		}
		if res.Summary.FusedBNCount != 0 || len(res.Spec.Nodes) != 3 {
			t.Fatalf("expected BN NOT folded on multi-consumer Conv, got summary=%+v nodes=%+v", res.Summary, res.Spec.Nodes)
		}
	})

	t.Run("conv_is_graph_output_with_dead_relu_not_fused", func(t *testing.T) {
		raw := buildModel(17, "conv_out",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildStaticValueInfo("c_out", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildFloatTensor("w", []int64{2, 2, 1, 1}, []float32{1, 0, 0, 1}, true)},
			[]*ProtoBuilder{
				buildNode("Conv", "c1", []string{"x", "w"}, []string{"c_out"}),
				buildNode("Relu", "dead_r", []string{"c_out"}, []string{"dead_y"}),
			},
		)
		m, _ := ParseONNX(raw)
		res, err := LowerONNX(m, LowerOptions{})
		if err != nil {
			t.Fatalf("LowerONNX: %v", err)
		}
		if res.Summary.FusedActCount != 0 || len(res.Spec.Nodes) != 1 || res.Spec.Nodes[0].Relu {
			t.Fatalf("expected Conv as graph output not fused, got %+v", res.Spec.Nodes)
		}
	})

	t.Run("pool_same_lower_emits_custom_pads_and_matches_ref", func(t *testing.T) {
		// 4x4 input, kernel 3x3, stride 2, SAME_LOWER:
		// oh = 2, padTotal = (2-1)*2 + 3 - 4 = 1 -> SAME_LOWER gives pt=1, pb=0, pl=1, pr=0.
		raw := buildModel(17, "pool_sl",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 4, 4})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
			nil,
			[]*ProtoBuilder{
				buildNode("MaxPool", "mp", []string{"x"}, []string{"y"},
					buildAttrInts("kernel_shape", []int64{3, 3}, true),
					buildAttrInts("strides", []int64{2, 2}, true),
					buildAttrString("auto_pad", "SAME_LOWER")),
			},
		)
		m, _ := ParseONNX(raw)
		res, err := LowerONNX(m, LowerOptions{})
		if err != nil {
			t.Fatalf("LowerONNX SAME_LOWER pool: %v", err)
		}
		nd := res.Spec.Nodes[0]
		if nd.Pad != "custom" || nd.PT != 1 || nd.PB != 0 || nd.PL != 1 || nd.PR != 0 {
			t.Fatalf("expected custom pads pt=1 pb=0 pl=1 pr=0, got %+v", nd)
		}
		dir := t.TempDir()
		_ = WriteLoweredGraph(dir, res)
		refSpec, nw := loadRefGraphFromDir(t, dir)
		xF64 := newF64Tensor(2, 4, 4)
		for i := range xF64.Data {
			xF64.Data[i] = float64(i) * 0.25
		}
		got, _, err := ref.EvalGraphF64(refSpec, map[string]ref.Tensor{"x": xF64.toRefFP16()}, nw)
		if err != nil {
			t.Fatalf("EvalGraphF64 SAME_LOWER pool: %v", err)
		}
		want := newF64Tensor(2, 2, 2)
		for c := 0; c < 2; c++ {
			for oy := 0; oy < 2; oy++ {
				for ox := 0; ox < 2; ox++ {
					best := math.Inf(-1)
					for ky := 0; ky < 3; ky++ {
						iy := oy*2 + ky - 1
						if iy < 0 || iy >= 4 {
							continue
						}
						for kx := 0; kx < 3; kx++ {
							ix := ox*2 + kx - 1
							if ix < 0 || ix >= 4 {
								continue
							}
							if v := xF64.at(c, iy, ix); v > best {
								best = v
							}
						}
					}
					want.set(c, oy, ox, best)
				}
			}
		}
		assertTensorClose(t, "maxpool SAME_LOWER", got, want, 1e-3)
	})
}

func TestOperatorCoverageAndDiagnostics(t *testing.T) {
	t.Run("gemm_and_matmul_add_lower_to_conv1x1_and_eval", func(t *testing.T) {
		w1 := []float64{
			0.5, -0.25, 0.75,
			-0.5, 0.25, 0.5,
			0.25, 0.5, -0.25,
			0.0, -0.5, 0.25,
		} // [4, 3]
		b1 := []float64{0.1, 0.2, -0.1}
		w2 := []float64{
			0.5, -0.5, 0.25,
			-0.25, 0.75, 0.5,
		} // [2, 3] (transB=1)
		b2 := []float64{-0.1, 0.2}

		raw := buildModel(17, "mlp",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 1, 1})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 1, 1})},
			[]*ProtoBuilder{
				buildFloatTensor("w1", []int64{4, 3}, toF32Slice(w1), true),
				buildFloatTensor("b1", []int64{3}, toF32Slice(b1), true),
				buildFloatTensor("w2", []int64{2, 3}, toF32Slice(w2), true),
				buildFloatTensor("b2", []int64{2}, toF32Slice(b2), true),
			},
			[]*ProtoBuilder{
				buildNode("MatMul", "mm1", []string{"x", "w1"}, []string{"h0"}),
				buildNode("Add", "add1", []string{"h0", "b1"}, []string{"h1"}),
				buildNode("Relu", "r1", []string{"h1"}, []string{"h2"}),
				buildNode("Gemm", "g2", []string{"h2", "w2", "b2"}, []string{"y"},
					buildAttrInt("transB", 1), buildAttrFloat("alpha", 0.5), buildAttrFloat("beta", 1.0)),
			},
		)
		m, err := ParseONNX(raw)
		if err != nil {
			t.Fatalf("ParseONNX: %v", err)
		}
		res, err := LowerONNX(m, LowerOptions{})
		if err != nil {
			t.Fatalf("LowerONNX: %v", err)
		}
		if len(res.Spec.Nodes) != 2 || res.Spec.Nodes[0].Op != "conv1x1" || !res.Spec.Nodes[0].Relu || res.Spec.Nodes[0].Bias == "" {
			t.Fatalf("unexpected MLP nodes: %+v", res.Spec.Nodes)
		}
		if res.Spec.Nodes[1].Op != "conv1x1" || res.Spec.Nodes[1].Cout != 2 {
			t.Fatalf("unexpected Gemm node: %+v", res.Spec.Nodes[1])
		}

		// Evaluate directly with internal/ref.EvalGraphF64!
		dir := t.TempDir()
		_ = WriteLoweredGraph(dir, res)
		refSpec, nw := loadRefGraphFromDir(t, dir)
		xF64 := newF64Tensor(4, 1, 1)
		xF64.Data = []float64{1.0, -0.5, 0.25, 0.75}
		got, _, err := ref.EvalGraphF64(refSpec, map[string]ref.Tensor{"x": xF64.toRefFP16()}, nw)
		if err != nil {
			t.Fatalf("EvalGraphF64 MLP: %v", err)
		}
		// Independent float64 reference:
		h := make([]float64, 3)
		for o := 0; o < 3; o++ {
			acc := b1[o]
			for c := 0; c < 4; c++ {
				acc += xF64.Data[c] * w1[c*3+o]
			}
			if acc > 0 {
				h[o] = acc
			}
		}
		want := newF64Tensor(2, 1, 1)
		for o := 0; o < 2; o++ {
			acc := b2[o]
			for c := 0; c < 3; c++ {
				acc += 0.5 * h[c] * w2[o*3+c]
			}
			want.Data[o] = acc
		}
		assertTensorClose(t, "mlp matmul+gemm conv1x1", got, want, 1e-2)
	})

	t.Run("foldMatMulAddBias_does_not_misfold_spatial_constant", func(t *testing.T) {
		raw := buildModel(17, "mm_spatial_add",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 4, 2, 2})},
			[]*ProtoBuilder{
				buildFloatTensor("w", []int64{2, 4}, make([]float32, 8), true),
				// Shape [1,1,2,2] has 4 elements == Cout, but is spatial, NOT a channel bias!
				buildFloatTensor("spatial_c", []int64{1, 1, 2, 2}, []float32{1, 2, 3, 4}, true),
			},
			[]*ProtoBuilder{
				buildNode("MatMul", "mm", []string{"x", "w"}, []string{"h"}),
				buildNode("Add", "add_sp", []string{"h", "spatial_c"}, []string{"y"}),
			},
		)
		m, _ := ParseONNX(raw)
		if _, err := LowerONNX(m, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "non-scalar constant") {
			t.Fatalf("expected spatial [1,1,2,2] constant NOT to fold as channel bias, got: %v", err)
		}
	})

	t.Run("scalar_pe_softplus_and_activations", func(t *testing.T) {
		raw := buildModel(17, "acts",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{
				buildFloatTensor("c2", []int64{}, []float32{2.0}, false),
				buildFloatTensor("c1", []int64{1}, []float32{1.0}, false),
			},
			[]*ProtoBuilder{
				buildNode("Add", "a0", []string{"x", "c1"}, []string{"t0"}),
				buildNode("Sub", "s0", []string{"c2", "t0"}, []string{"t1"}),
				buildNode("Div", "d0", []string{"t1", "c2"}, []string{"t2"}),
				buildNode("Abs", "ab0", []string{"t2"}, []string{"t3"}),
				buildNode("LeakyRelu", "lr", []string{"t3"}, []string{"t4"}, buildAttrFloat("alpha", 0.1)),
				buildNode("Elu", "el", []string{"t4"}, []string{"t5"}, buildAttrFloat("alpha", 1.0)),
				buildNode("Softplus", "sp", []string{"t5"}, []string{"y"}),
			},
		)
		m, err := ParseONNX(raw)
		if err != nil {
			t.Fatalf("ParseONNX: %v", err)
		}
		res, err := LowerONNX(m, LowerOptions{})
		if err != nil {
			t.Fatalf("LowerONNX: %v", err)
		}
		dir := t.TempDir()
		_ = WriteLoweredGraph(dir, res)
		refSpec, nw := loadRefGraphFromDir(t, dir)
		xF64 := newF64Tensor(2, 2, 2)
		for i := range xF64.Data {
			xF64.Data[i] = float64(i-3) * 0.5
		}
		got, _, err := ref.EvalGraphF64(refSpec, map[string]ref.Tensor{"x": xF64.toRefFP16()}, nw)
		if err != nil {
			t.Fatalf("EvalGraphF64: %v", err)
		}
		want := newF64Tensor(2, 2, 2)
		for i, v := range xF64.Data {
			v = math.Abs((2.0 - (v + 1.0)) / 2.0)
			if v < 0 {
				v *= 0.1
			}
			if v < 0 {
				v = math.Expm1(v)
			}
			want.Data[i] = math.Log(1.0 + math.Exp(v))
		}
		assertTensorClose(t, "scalar+acts+softplus", got, want, 1e-2)
	})

	t.Run("views_depthwise_and_noop_reduce", func(t *testing.T) {
		raw := buildModel(18, "views",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 4, 4})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 1})},
			[]*ProtoBuilder{
				buildFloatTensor("dw", []int64{4, 1, 3, 3}, make([]float32, 36), true),
				buildInt64Tensor("starts", []int64{2}, []int64{0, 0}, false),
				buildInt64Tensor("ends", []int64{2}, []int64{2, 2}, false),
				buildInt64Tensor("axes", []int64{2}, []int64{1, 2}, false),
				buildInt64Tensor("red_axes", []int64{1}, []int64{-1}, false),
			},
			[]*ProtoBuilder{
				buildNode("Constant", "shape_c", nil, []string{"rshape"},
					buildAttrTensor("value", buildInt64Tensor("", []int64{4}, []int64{1, -1, 2, 8}, false))),
				buildNode("Conv", "dw_conv", []string{"x", "dw"}, []string{"h0"},
					buildAttrInt("group", 4),
					buildAttrInts("pads", []int64{1, 1, 1, 1}, true)),
				buildNode("Transpose", "tr", []string{"h0"}, []string{"h1"},
					buildAttrInts("perm", []int64{0, 1, 3, 2}, true)),
				buildNode("Reshape", "rs", []string{"h1", "rshape"}, []string{"h2"}),
				buildNode("Slice", "sl", []string{"h2", "starts", "ends", "axes"}, []string{"h3"}),
				buildNode("ReduceSum", "noop_red", []string{"h3"}, []string{"h4"},
					buildAttrInt("keepdims", 1), buildAttrInt("noop_with_empty_axes", 1)),
				buildNode("ReduceSum", "red", []string{"h4", "red_axes"}, []string{"y"},
					buildAttrInt("keepdims", 1)),
			},
		)
		m, err := ParseONNX(raw)
		if err != nil {
			t.Fatalf("ParseONNX: %v", err)
		}
		res, err := LowerONNX(m, LowerOptions{AllowViewOps: true})
		if err != nil {
			t.Fatalf("LowerONNX: %v", err)
		}
		if res.Spec.Nodes[0].Op != "conv3x3" || res.Spec.Nodes[0].Mode != "depthwise" {
			t.Fatalf("expected depthwise conv3x3, got %+v", res.Spec.Nodes[0])
		}
		if res.Spec.Nodes[1].Op != "transpose" || res.Spec.Nodes[2].Op != "reshape" || res.Spec.Nodes[3].Op != "slice" || res.Spec.Nodes[4].Op != "reduce" {
			t.Fatalf("unexpected lowered view/reduce nodes: %+v", res.Spec.Nodes)
		}
		if res.Spec.Nodes[4].Axes != "w" || res.Summary.OutputShape != [4]int{1, 2, 2, 1} {
			t.Fatalf("unexpected reduce axes/shape: %+v %v", res.Spec.Nodes[4], res.Summary.OutputShape)
		}
	})

	t.Run("diagnostics_and_minor_guards", func(t *testing.T) {
		// Unsupported op
		rawUnsup := buildModel(17, "bad_op",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 4, 4})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 4, 8, 8})},
			nil,
			[]*ProtoBuilder{buildNode("Resize", "up0", []string{"x"}, []string{"y"})},
		)
		m, _ := ParseONNX(rawUnsup)
		if _, err := LowerONNX(m, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "up0") || !strings.Contains(err.Error(), "Resize") {
			t.Fatalf("expected error naming node 'up0' and op 'Resize', got: %v", err)
		}

		// Negative BN variance
		rawBadBN := buildModel(17, "bad_bn",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{
				buildFloatTensor("w", []int64{2, 2, 1, 1}, []float32{1, 0, 0, 1}, true),
				buildFloatTensor("s", []int64{2}, []float32{1, 1}, true),
				buildFloatTensor("b", []int64{2}, []float32{0, 0}, true),
				buildFloatTensor("m", []int64{2}, []float32{0, 0}, true),
				buildFloatTensor("v", []int64{2}, []float32{1, -0.5}, true),
			},
			[]*ProtoBuilder{
				buildNode("Conv", "c", []string{"x", "w"}, []string{"h"}),
				buildNode("BatchNormalization", "bn", []string{"h", "s", "b", "m", "v"}, []string{"y"}),
			},
		)
		mBN, _ := ParseONNX(rawBadBN)
		if _, err := LowerONNX(mBN, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "variance") {
			t.Fatalf("expected variance error, got: %v", err)
		}

		// BN training_mode != 0 rejected
		rawTrainBN := buildModel(17, "train_bn",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{
				buildFloatTensor("s", []int64{2}, []float32{1, 1}, true),
				buildFloatTensor("b", []int64{2}, []float32{0, 0}, true),
				buildFloatTensor("m", []int64{2}, []float32{0, 0}, true),
				buildFloatTensor("v", []int64{2}, []float32{1, 1}, true),
			},
			[]*ProtoBuilder{
				buildNode("BatchNormalization", "bn", []string{"x", "s", "b", "m", "v"}, []string{"y"},
					buildAttrInt("training_mode", 1)),
			},
		)
		mTBN, _ := ParseONNX(rawTrainBN)
		if _, err := LowerONNX(mTBN, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "training_mode") {
			t.Fatalf("expected training_mode error, got: %v", err)
		}

		// foldConvBatchNorm divide-by-zero guard (cout=0)
		rawZeroCout := buildModel(17, "zero_cout_bn",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{
				buildFloatTensor("w", []int64{0, 2, 1, 1}, []float32{}, false),
				buildFloatTensor("s", []int64{0}, []float32{}, false),
				buildFloatTensor("b", []int64{0}, []float32{}, false),
				buildFloatTensor("m", []int64{0}, []float32{}, false),
				buildFloatTensor("v", []int64{0}, []float32{}, false),
			},
			[]*ProtoBuilder{
				buildNode("Conv", "c", []string{"x", "w"}, []string{"h"}),
				buildNode("BatchNormalization", "bn", []string{"h", "s", "b", "m", "v"}, []string{"y"}),
			},
		)
		mZC, _ := ParseONNX(rawZeroCout)
		if _, err := LowerONNX(mZC, LowerOptions{}); err == nil {
			t.Fatal("expected clean error for cout=0 in Conv+BN folding")
		}

		// Grouped conv3x3 rejected
		rawGrp := buildModel(17, "bad_grp",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 4, 4})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 4, 4, 4})},
			[]*ProtoBuilder{buildFloatTensor("w", []int64{4, 2, 3, 3}, make([]float32, 72), true)},
			[]*ProtoBuilder{buildNode("Conv", "gconv", []string{"x", "w"}, []string{"y"}, buildAttrInt("group", 2))},
		)
		mGrp, _ := ParseONNX(rawGrp)
		if _, err := LowerONNX(mGrp, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "grouped") {
			t.Fatalf("expected grouped conv3x3 error, got: %v", err)
		}

		// ReduceMean with keepdims=0 rejected
		rawKeep := buildModel(17, "bad_keep",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 4, 4})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 4})},
			nil,
			[]*ProtoBuilder{buildNode("ReduceMean", "rm", []string{"x"}, []string{"y"},
				buildAttrInts("axes", []int64{2, 3}, true), buildAttrInt("keepdims", 0))},
		)
		mKeep, _ := ParseONNX(rawKeep)
		if _, err := LowerONNX(mKeep, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "keepdims") {
			t.Fatalf("expected keepdims error, got: %v", err)
		}

		// ReduceSum over axis H with W > 1 rejected
		rawRedH := buildModel(17, "bad_red_h",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 4, 4})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 4, 1, 4})},
			nil,
			[]*ProtoBuilder{buildNode("ReduceSum", "rs", []string{"x"}, []string{"y"},
				buildAttrInts("axes", []int64{2}, true), buildAttrInt("keepdims", 1))},
		)
		mRedH, _ := ParseONNX(rawRedH)
		if _, err := LowerONNX(mRedH, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "reduce backend") {
			t.Fatalf("expected reduce backend error for axes='h' W>1, got: %v", err)
		}

		// Gemm with H > 1 rejected
		rawGemm4D := buildModel(17, "bad_gemm",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 2, 2})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildFloatTensor("w", []int64{4, 2}, make([]float32, 8), true)},
			[]*ProtoBuilder{buildNode("Gemm", "g", []string{"x", "w"}, []string{"y"})},
		)
		mG4, _ := ParseONNX(rawGemm4D)
		if _, err := LowerONNX(mG4, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "2D") {
			t.Fatalf("expected Gemm 2D input error, got: %v", err)
		}

		// Pool with stride 0 rejected
		rawPoolS0 := buildModel(17, "bad_pool_s0",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 4, 4})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
			nil,
			[]*ProtoBuilder{buildNode("MaxPool", "mp", []string{"x"}, []string{"y"},
				buildAttrInts("kernel_shape", []int64{2, 2}, true),
				buildAttrInts("strides", []int64{0, 2}, true))},
		)
		mPS0, _ := ParseONNX(rawPoolS0)
		if _, err := LowerONNX(mPS0, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "strides") {
			t.Fatalf("expected pool stride error, got: %v", err)
		}

		// Clip with min > max rejected
		rawClip := buildModel(17, "bad_clip",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
			nil,
			[]*ProtoBuilder{buildNode("Clip", "cl", []string{"x"}, []string{"y"},
				buildAttrFloat("min", 5.0), buildAttrFloat("max", 1.0))},
		)
		mCl, _ := ParseONNX(rawClip)
		if _, err := LowerONNX(mCl, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "min <= max") {
			t.Fatalf("expected Clip min <= max error, got: %v", err)
		}

		// HardSigmoid alpha <= 0 rejected
		rawHS := buildModel(17, "bad_hs",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
			nil,
			[]*ProtoBuilder{buildNode("HardSigmoid", "hs", []string{"x"}, []string{"y"}, buildAttrFloat("alpha", -0.2))},
		)
		mHS, _ := ParseONNX(rawHS)
		if _, err := LowerONNX(mHS, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "alpha") {
			t.Fatalf("expected HardSigmoid alpha error, got: %v", err)
		}

		// Conv1x1 unknown auto_pad rejected
		rawC1Pad := buildModel(17, "bad_c1_pad",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 2, 2})},
			[]*ProtoBuilder{buildFloatTensor("w", []int64{2, 2, 1, 1}, []float32{1, 0, 0, 1}, true)},
			[]*ProtoBuilder{buildNode("Conv", "c1", []string{"x", "w"}, []string{"y"}, buildAttrString("auto_pad", "BOGUS"))},
		)
		mC1P, _ := ParseONNX(rawC1Pad)
		if _, err := LowerONNX(mC1P, LowerOptions{}); err == nil || !strings.Contains(err.Error(), "auto_pad") {
			t.Fatalf("expected conv1x1 auto_pad error, got: %v", err)
		}

		// Duplicate Slice axes rejected
		rawDupSlice := buildModel(17, "dup_slice",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 4, 4})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 4, 4})},
			[]*ProtoBuilder{
				buildInt64Tensor("st", []int64{2}, []int64{0, 0}, false),
				buildInt64Tensor("en", []int64{2}, []int64{2, 2}, false),
				buildInt64Tensor("ax", []int64{2}, []int64{1, -3}, false), // both axis 1
			},
			[]*ProtoBuilder{buildNode("Slice", "sl", []string{"x", "st", "en", "ax"}, []string{"y"})},
		)
		mDS, _ := ParseONNX(rawDupSlice)
		if _, err := LowerONNX(mDS, LowerOptions{AllowViewOps: true}); err == nil || !strings.Contains(err.Error(), "duplicate axis") {
			t.Fatalf("expected duplicate Slice axis error, got: %v", err)
		}
	})
}
