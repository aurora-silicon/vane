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

func refConv3x3F64(x f64Tensor, w []float64, b []float64, cout, stride int, padSame bool) f64Tensor {
	ho, wo, pt, pl, err := ref.Conv3x3Geometry(x.H, x.W, stride, map[bool]string{true: "same", false: "valid"}[padSame])
	if err != nil {
		panic(err)
	}
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

func refSiluF64(x f64Tensor) f64Tensor {
	out := newF64Tensor(x.C, x.H, x.W)
	for i, v := range x.Data {
		out.Data[i] = v / (1.0 + math.Exp(-v))
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
	maxErr := 0.0
	for i := range want.Data {
		g := fp16.ToFloat64(got.Data[i])
		diff := math.Abs(g - want.Data[i])
		if diff > maxErr {
			maxErr = diff
		}
		if diff > atol {
			t.Fatalf("%s: elem %d mismatch: got %.6g, want %.6g (diff=%.6g > atol=%.6g)", label, i, g, want.Data[i], diff, atol)
		}
	}
}

func TestEndToEndConvBNReluResidualBlock(t *testing.T) {
	// Graph: x [batch,4,4,4] -> Identity -> Conv3x3(4->4, same) -> BN1 -> ReLU
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

	// Without --input-shape override, symbolic batch dim must be rejected cleanly.
	if _, err := LowerONNX(model, LowerOptions{}); err == nil {
		t.Fatal("expected error for dynamic batch dim without InputShapes override")
	}

	// Lower with fusion enabled.
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
	// Expected fused nodes: conv1(3x3+BN+ReLU), conv2(1x1+BN), res_add(pe add), relu_out(lut relu) = 4 nodes
	if len(res.Spec.Nodes) != 4 {
		t.Fatalf("expected 4 fused nodes, got %d: %+v", len(res.Spec.Nodes), res.Spec.Nodes)
	}
	if !res.Spec.Nodes[0].Relu {
		t.Fatalf("expected conv1 to have fused relu=true")
	}

	dir := t.TempDir()
	if err := WriteLoweredGraph(dir, res); err != nil {
		t.Fatalf("WriteLoweredGraph: %v", err)
	}
	refSpec, nw := loadRefGraphFromDir(t, dir)

	// Build deterministic input and independent float64 reference.
	xF64 := newF64Tensor(C, H, W)
	for i := range xF64.Data {
		xF64.Data[i] = fp16.ToFloat64(fp16.FromFloat64(float64((i%11)-5) * 0.25))
	}
	xFP16 := xF64.toRefFP16()

	want := refConv3x3F64(xF64, w1, b1, C, 1, true)
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

	// Also verify NoFuse produces unfused graph that still evaluates accurately via ref.EvalGraphF64.
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

func TestEndToEndPoolingReduceSoftmaxBlock(t *testing.T) {
	// x [1, 4, 4, 4] -> MaxPool(2x2, stride=2, valid) [1,4,2,2]
	//                -> AveragePool(2x2, stride=1, pads=[1,1,1,1], count_include_pad=0) [1,4,3,3]
	//                -> Softmax(axis=1) [1,4,3,3]
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

	dir := t.TempDir()
	if err := WriteLoweredGraph(dir, res); err != nil {
		t.Fatalf("WriteLoweredGraph: %v", err)
	}
	refSpec, nw := loadRefGraphFromDir(t, dir)

	xF64 := newF64Tensor(C, H, W)
	for i := range xF64.Data {
		xF64.Data[i] = fp16.ToFloat64(fp16.FromFloat64(float64((i%9)-4) * 0.35))
	}
	xFP16 := xF64.toRefFP16()

	gotF64, _, err := ref.EvalGraphF64(refSpec, map[string]ref.Tensor{"x": xFP16}, nw)
	if err != nil {
		t.Fatalf("EvalGraphF64: %v", err)
	}

	// Independent float64 reference:
	// 1. MaxPool 2x2 stride 2 valid -> 2x2
	p1 := newF64Tensor(C, 2, 2)
	for c := 0; c < C; c++ {
		for oy := 0; oy < 2; oy++ {
			for ox := 0; ox < 2; ox++ {
				best := math.Inf(-1)
				for ky := 0; ky < 2; ky++ {
					for kx := 0; kx < 2; kx++ {
						v := xF64.at(c, oy*2+ky, ox*2+kx)
						if v > best {
							best = v
						}
					}
				}
				p1.set(c, oy, ox, best)
			}
		}
	}
	// 2. AveragePool 2x2 stride 1 pads=[1,1,1,1] exclude_pad=true -> 3x3
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
	// 3. Softmax along C
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
	// 4. GlobalAveragePool
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

func TestEndToEndConvConcatUNetSkipBlock(t *testing.T) {
	// U-Net skip block:
	//   x [1, 4, 4, 4]
	//   branch A: Conv3x3(4->4, same) -> Sigmoid -> Mul(conv_out, sig) [Silu!] -> MaxPool(2x2, stride=2) [1,4,2,2]
	//   branch B (skip): Conv3x3(4->2, stride=2, same) [1,2,2,2]
	//   Concat(A, B, axis=1) [1, 6, 2, 2]
	//   Conv1x1(6->4) -> ReLU -> y [1, 4, 2, 2]
	wA := make([]float64, 4*4*9)
	for i := range wA {
		wA[i] = float64((i%5)-2) * 0.06
	}
	wB := make([]float64, 2*4*9)
	for i := range wB {
		wB[i] = float64((i%7)-3) * 0.05
	}
	wOut := make([]float64, 4*6)
	for i := range wOut {
		wOut[i] = float64((i%5)-2) * 0.1
	}
	bOut := []float64{0.05, -0.05, 0.1, 0.0}

	rawONNX := buildModel(17, "unet_skip",
		[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 4, 4})},
		[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 4, 2, 2})},
		[]*ProtoBuilder{
			buildFloatTensor("wA", []int64{4, 4, 3, 3}, toF32Slice(wA), true),
			buildFloatTensor("wB", []int64{2, 4, 3, 3}, toF32Slice(wB), true),
			buildFloatTensor("wOut", []int64{4, 6, 1, 1}, toF32Slice(wOut), true),
			buildFloatTensor("bOut", []int64{4}, toF32Slice(bOut), true),
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
				buildAttrInts("pads", []int64{1, 1, 1, 1}, true)),
			buildNode("Concat", "skip_cat", []string{"a2", "b1"}, []string{"cat"},
				buildAttrInt("axis", 1)),
			buildNode("Conv", "out_conv", []string{"cat", "wOut", "bOut"}, []string{"c_out"}),
			buildNode("Relu", "out_relu", []string{"c_out"}, []string{"y"}),
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

	// Verify Silu fusion on enc_conv, ReLU fusion on out_conv, and concat axis=1.
	if res.Summary.FusedActCount != 2 {
		t.Fatalf("expected 2 fused activations (silu + relu), got %+v", res.Summary)
	}
	if len(res.Spec.Nodes) != 5 {
		t.Fatalf("expected 5 nodes (enc_conv, enc_pool, skip_conv, skip_cat, out_conv), got %d", len(res.Spec.Nodes))
	}
	if res.Spec.Nodes[0].Act == nil || res.Spec.Nodes[0].Act.Name != "silu" {
		t.Fatalf("expected enc_conv fused act=silu, got %+v", res.Spec.Nodes[0].Act)
	}
	if res.Spec.Nodes[3].Op != "concat" || res.Spec.Nodes[3].Axis == nil || *res.Spec.Nodes[3].Axis != 1 {
		t.Fatalf("expected concat axis=1 at node 3, got %+v", res.Spec.Nodes[3])
	}
	if !res.Spec.Nodes[4].Relu || res.Summary.OutputShape != [4]int{1, 4, 2, 2} {
		t.Fatalf("unexpected final node/shape: %+v, %v", res.Spec.Nodes[4], res.Summary.OutputShape)
	}

	// Write graph.json + weights, reload from disk, and evaluate every emitted node using
	// internal/ref.EvalGraphF64 (plus channel concat for the M5C concat node) vs independent float64 ref.
	dir := t.TempDir()
	if err := WriteLoweredGraph(dir, res); err != nil {
		t.Fatalf("WriteLoweredGraph: %v", err)
	}
	rawJSON, err := os.ReadFile(filepath.Join(dir, "graph.json"))
	if err != nil {
		t.Fatalf("read graph.json: %v", err)
	}
	var loaded VaneGraphSpec
	if err := json.Unmarshal(rawJSON, &loaded); err != nil {
		t.Fatalf("unmarshal VaneGraphSpec: %v", err)
	}

	xF64 := newF64Tensor(4, 4, 4)
	for i := range xF64.Data {
		xF64.Data[i] = fp16.ToFloat64(fp16.FromFloat64(float64((i%9)-4) * 0.25))
	}
	env := map[string]ref.Tensor{"x": xF64.toRefFP16()}

	evalSingleNode := func(nd VaneGraphNode) ref.Tensor {
		if nd.Op == "concat" {
			t0, t1 := env[nd.Inputs[0]], env[nd.Inputs[1]]
			out, _ := ref.NewTensor(1, t0.C+t1.C, t0.H, t0.W)
			copy(out.Data, t0.Data)
			copy(out.Data[len(t0.Data):], t1.Data)
			return out
		}
		// Marshal node into a single-node ref.GraphSpec and evaluate via ref.EvalGraphF64!
		ndBytes, _ := json.Marshal(nd)
		var rNode ref.GraphNodeSpec
		if err := json.Unmarshal(ndBytes, &rNode); err != nil {
			t.Fatalf("unmarshal node %s: %v", nd.ID, err)
		}
		inSpecs := make([]ref.GraphInputSpec, len(nd.Inputs))
		inMap := make(map[string]ref.Tensor, len(nd.Inputs))
		for i, nm := range nd.Inputs {
			inT := env[nm]
			inSpecs[i] = ref.GraphInputSpec{Name: nm, C: inT.C, H: inT.H, W: inT.W}
			inMap[nm] = inT
		}
		nw := make(map[string]ref.GraphNodeWeights)
		if nd.Weights != "" {
			wb, err := os.ReadFile(filepath.Join(dir, nd.Weights))
			if err != nil {
				t.Fatalf("read %s: %v", nd.Weights, err)
			}
			w16, _ := fp16.DecodeLE(wb)
			var b16 []uint16
			if nd.Bias != "" {
				bb, err := os.ReadFile(filepath.Join(dir, nd.Bias))
				if err != nil {
					t.Fatalf("read %s: %v", nd.Bias, err)
				}
				b16, _ = fp16.DecodeLE(bb)
			}
			nw[nd.ID] = ref.GraphNodeWeights{Weights: w16, Bias: b16}
		}
		subSpec := ref.GraphSpec{Target: loaded.Target, Inputs: inSpecs, Nodes: []ref.GraphNodeSpec{rNode}}
		out, _, err := ref.EvalGraphF64(subSpec, inMap, nw)
		if err != nil {
			t.Fatalf("EvalGraphF64(%s): %v", nd.ID, err)
		}
		return out
	}

	var gotOut ref.Tensor
	for _, nd := range loaded.Nodes {
		gotOut = evalSingleNode(nd)
		env[nd.ID] = gotOut
	}

	// Independent float64 U-Net reference
	a0 := refConv3x3F64(xF64, wA, nil, 4, 1, true)
	a1 := refSiluF64(a0)
	a2 := newF64Tensor(4, 2, 2)
	for c := 0; c < 4; c++ {
		for oy := 0; oy < 2; oy++ {
			for ox := 0; ox < 2; ox++ {
				best := math.Inf(-1)
				for ky := 0; ky < 2; ky++ {
					for kx := 0; kx < 2; kx++ {
						if v := a1.at(c, oy*2+ky, ox*2+kx); v > best {
							best = v
						}
					}
				}
				a2.set(c, oy, ox, best)
			}
		}
	}
	b1 := refConv3x3F64(xF64, wB, nil, 2, 2, true)
	cat := newF64Tensor(6, 2, 2)
	copy(cat.Data, a2.Data)
	copy(cat.Data[len(a2.Data):], b1.Data)
	want := refReluF64(refConv1x1F64(cat, wOut, bOut, 4))

	assertTensorClose(t, "unet conv+concat", gotOut, want, 2e-2)
}

func TestEndToEndSafeTensorsBackedModel(t *testing.T) {
	// ONNX topology with EXTERNAL initializers; actual F32 & BF16 weights supplied via SafeTensors.
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
	// Without SafeTensors override, EXTERNAL initializers must fail cleanly.
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
	if err := WriteLoweredGraph(dir, res); err != nil {
		t.Fatalf("WriteLoweredGraph: %v", err)
	}
	refSpec, nw := loadRefGraphFromDir(t, dir)

	xF64 := newF64Tensor(4, 2, 2)
	for i := range xF64.Data {
		xF64.Data[i] = fp16.ToFloat64(fp16.FromFloat64(float64((i%7)-3) * 0.25))
	}
	got, _, err := ref.EvalGraphF64(refSpec, map[string]ref.Tensor{"x": xF64.toRefFP16()}, nw)
	if err != nil {
		t.Fatalf("EvalGraphF64: %v", err)
	}

	// Independent float64 reference (matching BF16-quantized bnB from SafeTensors)
	bnBExact, _ := sf.ToFloat64("bn.bias")
	want := refBNF64(refConv1x1F64(xF64, w1, nil, 4), bnS, bnBExact, bnM, bnV, 1e-5)
	for i, v := range want.Data {
		want.Data[i] = 0.5 * v * (1.0 + math.Erf(v/math.Sqrt2))
	}
	assertTensorClose(t, "safetensors conv+bn+gelu", got, want, 1e-2)
}

func TestOperatorCoverageAndDiagnostics(t *testing.T) {
	t.Run("gemm_and_matmul_add_linear", func(t *testing.T) {
		raw := buildModel(17, "mlp",
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 1, 1})},
			[]*ProtoBuilder{buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 1, 1})},
			[]*ProtoBuilder{
				buildFloatTensor("w1", []int64{4, 3}, make([]float32, 12), true),
				buildFloatTensor("b1", []int64{3}, []float32{0.1, 0.2, 0.3}, true),
				buildFloatTensor("w2", []int64{2, 3}, make([]float32, 6), true),
				buildFloatTensor("b2", []int64{2}, []float32{-0.1, 0.1}, true),
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
		if len(res.Spec.Nodes) != 2 || res.Spec.Nodes[0].Op != "linear" || !res.Spec.Nodes[0].Relu || res.Spec.Nodes[0].Bias == "" {
			t.Fatalf("unexpected MLP nodes: %+v", res.Spec.Nodes)
		}
		if res.Spec.Nodes[1].Op != "linear" || res.Spec.Nodes[1].Cout != 2 {
			t.Fatalf("unexpected Gemm node: %+v", res.Spec.Nodes[1])
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
		if res.Spec.Nodes[0].PeOp != "sadd" || res.Spec.Nodes[1].PeOp != "srsub" || res.Spec.Nodes[2].PeOp != "sdiv" || res.Spec.Nodes[3].PeOp != "abs" {
			t.Fatalf("unexpected PE scalar/unary lowering: %+v", res.Spec.Nodes[:4])
		}

		// Evaluate entire chain via ref.EvalGraphF64 and check against direct formula!
		dir := t.TempDir()
		if err := WriteLoweredGraph(dir, res); err != nil {
			t.Fatalf("WriteLoweredGraph: %v", err)
		}
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

	t.Run("views_depthwise_and_constant_node", func(t *testing.T) {
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
					buildAttrTensor("value", buildInt64Tensor("", []int64{4}, []int64{1, -1, 4, 4}, false))),
				buildNode("Conv", "dw_conv", []string{"x", "dw"}, []string{"h0"},
					buildAttrInt("group", 4),
					buildAttrInts("pads", []int64{1, 1, 1, 1}, true)),
				buildNode("Transpose", "tr", []string{"h0"}, []string{"h1"},
					buildAttrInts("perm", []int64{0, 1, 3, 2}, true)),
				buildNode("Reshape", "rs", []string{"h1", "rshape"}, []string{"h2"}),
				buildNode("Slice", "sl", []string{"h2", "starts", "ends", "axes"}, []string{"h3"}),
				buildNode("ReduceSum", "red", []string{"h3", "red_axes"}, []string{"y"},
					buildAttrInt("keepdims", 1)),
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

	t.Run("diagnostics_errors", func(t *testing.T) {
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

		// Grouped conv3x3 (unsupported group=2, cin=4, cout=4)
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

		// ReduceMean with keepdims=0
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
	})
}
