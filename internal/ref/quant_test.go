package ref

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/aurora-silicon/vane/internal/fp16"
)

func makeSyntheticWeightsF64(cout, inner int, seed int) []float64 {
	out := make([]float64, cout*inner)
	for o := 0; o < cout; o++ {
		scale := 0.25 + float64(o%5)*0.15
		bias := float64((o%3)-1) * 0.1
		for j := 0; j < inner; j++ {
			v := math.Sin(float64((o*31+j*17+seed)%97)) * scale
			out[o*inner+j] = v + bias
		}
	}
	return out
}

func TestComputePexpAndBiasShift(t *testing.T) {
	if got := ComputePexp([]float32{0.0625}); got != 0 {
		t.Fatalf("ComputePexp(0.0625) = %d, want 0", got)
	}
	if got := ComputePexp([]float32{float32(math.Ldexp(1.0, -16))}); got != 0 {
		t.Fatalf("ComputePexp(2^-16) = %d, want 0", got)
	}
	if got := ComputePexp([]float32{float32(math.Ldexp(1.0, -18))}); got != 2 {
		t.Fatalf("ComputePexp(2^-18) = %d, want 2", got)
	}
	if got := ComputePexp([]float32{0.5, float32(math.Ldexp(1.0, -20))}); got != 4 {
		t.Fatalf("ComputePexp([0.5, 2^-20]) = %d, want 4", got)
	}

	sSmall := math.Ldexp(1.0, -15)
	bLarge := []uint16{fp16.FromFloat64(1.5)} // 1.5 / 2^-15 = 49152 >= 32768 -> shift 1
	if got := ComputeBiasShift([]float64{sSmall}, bLarge); got != 1 {
		t.Fatalf("ComputeBiasShift(2^-15, 1.5) = %d, want 1", got)
	}
	if got := ComputeBiasShift([]float64{0.125}, bLarge); got != 0 {
		t.Fatalf("ComputeBiasShift(0.125, 1.5) = %d, want 0", got)
	}
}

func TestQuantizersAndDequantizeAccuracy(t *testing.T) {
	const cout, inner = 16, 32
	wOrig := makeSyntheticWeightsF64(cout, inner, 7)

	cases := []struct {
		name   string
		build  func() (QuantWeights, error)
		maxErr float64
	}{
		{"int8_pt_sym", func() (QuantWeights, error) { return QuantizeInt8(wOrig, cout, inner, false, false) }, 2e-2},
		{"int8_pc_sym", func() (QuantWeights, error) { return QuantizeInt8(wOrig, cout, inner, true, false) }, 1.5e-2},
		{"int8_pc_zp", func() (QuantWeights, error) { return QuantizeInt8(wOrig, cout, inner, true, true) }, 1.5e-2},
		{"uint8_pc_zp", func() (QuantWeights, error) { return QuantizeUint8(wOrig, cout, inner, true) }, 1.5e-2},
		{"int4_pc_sym", func() (QuantWeights, error) { return QuantizeInt4(wOrig, cout, inner, true, false) }, 2e-1},
		{"int4_pc_zp", func() (QuantWeights, error) { return QuantizeInt4(wOrig, cout, inner, true, true) }, 2e-1},
		{"uint4_pc_zp", func() (QuantWeights, error) { return QuantizeUint4(wOrig, cout, inner, true) }, 2e-1},
		{"pal1", func() (QuantWeights, error) { return QuantizePalette(wOrig, 1) }, 0.9},
		{"pal2", func() (QuantWeights, error) { return QuantizePalette(wOrig, 2) }, 0.5},
		{"pal4", func() (QuantWeights, error) { return QuantizePalette(wOrig, 4) }, 0.15},
		{"pal6", func() (QuantWeights, error) { return QuantizePalette(wOrig, 6) }, 0.06},
		{"pal8", func() (QuantWeights, error) { return QuantizePalette(wOrig, 8) }, 0.02},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			qw, err := tc.build()
			if err != nil {
				t.Fatalf("quantize %s: %v", tc.name, err)
			}
			dqF64, dqF16, err := DequantizeWeights(qw, cout, inner)
			if err != nil {
				t.Fatalf("dequantize %s: %v", tc.name, err)
			}
			if len(dqF64) != len(wOrig) || len(dqF16) != len(wOrig) {
				t.Fatalf("dequantized length mismatch")
			}
			maxAbs := 0.0
			for i, w := range wOrig {
				d := math.Abs(w - dqF64[i])
				if d > maxAbs {
					maxAbs = d
				}
			}
			if maxAbs > tc.maxErr {
				t.Fatalf("%s max_abs=%g exceeds bound %g", tc.name, maxAbs, tc.maxErr)
			}
		})
	}
}

func TestEvalConv1x1LinearConv3x3QuantMatchesDequantizedFP16(t *testing.T) {
	x := makeConstTensor(8, 6, 8, []float64{-0.5, 0.25, 0.75, -0.125, 0.5})
	bias := makeConstWeights(8, []float64{0.125, -0.25, 0.0, 0.5})

	// 1x1 int8 per-channel + ZP
	w1 := makeSyntheticWeightsF64(8, 8, 11)
	qw1, err := QuantizeInt8(w1, 8, 8, true, true)
	if err != nil {
		t.Fatalf("QuantizeInt8: %v", err)
	}
	_, w1F16, err := DequantizeWeights(qw1, 8, 8)
	if err != nil {
		t.Fatalf("DequantizeWeights 1x1: %v", err)
	}
	yQ1, err := EvalConv1x1Quant(x, qw1, 8, bias, true)
	if err != nil {
		t.Fatalf("EvalConv1x1Quant: %v", err)
	}
	yRef1, err := Conv2d1x1(x, w1F16, 8, bias, true)
	if err != nil {
		t.Fatalf("Conv2d1x1: %v", err)
	}
	cmp1, err := CompareFP16(yQ1.Data, yRef1.Data, 2e-2, 2e-2, nil)
	if err != nil || !cmp1.Passed {
		t.Fatalf("1x1 quant vs dequantized fp16 failed: err=%v maxAbs=%g", err, cmp1.MaxAbsErr)
	}

	// Linear pal4
	xLin := makeConstTensor(8, 1, 16, []float64{0.5, -0.25, 0.125, 0.75})
	wLin := makeSyntheticWeightsF64(12, 8, 19)
	qwLin, err := QuantizePalette(wLin, 4)
	if err != nil {
		t.Fatalf("QuantizePalette: %v", err)
	}
	_, wLinF16, err := DequantizeWeights(qwLin, 12, 8)
	if err != nil {
		t.Fatalf("DequantizeWeights pal4: %v", err)
	}
	bLin := makeConstWeights(12, []float64{0.125, -0.125})
	yLinQ, err := EvalLinearQuant(xLin, qwLin, 12, bLin, false)
	if err != nil {
		t.Fatalf("EvalLinearQuant: %v", err)
	}
	yLinRef, err := Conv2d1x1(xLin, wLinF16, 12, bLin, false)
	if err != nil {
		t.Fatalf("Conv2d1x1 linear: %v", err)
	}
	// Palette weights are exact fp16 entries -> bit-identical to Conv2d1x1!
	for i := range yLinQ.Data {
		if yLinQ.Data[i] != yLinRef.Data[i] {
			t.Fatalf("pal4 linear mismatch at %d: 0x%04x != 0x%04x", i, yLinQ.Data[i], yLinRef.Data[i])
		}
	}

	// Dense 3x3 int4
	w3 := makeSyntheticWeightsF64(8, 8*9, 23)
	qw3, err := QuantizeInt4(w3, 8, 8*9, true, false)
	if err != nil {
		t.Fatalf("QuantizeInt4 3x3: %v", err)
	}
	_, w3F16, err := DequantizeWeights(qw3, 8, 8*9)
	if err != nil {
		t.Fatalf("DequantizeWeights 3x3: %v", err)
	}
	y3Q, err := EvalConv3x3Quant(x, qw3, 8, bias, 1, "same", true)
	if err != nil {
		t.Fatalf("EvalConv3x3Quant: %v", err)
	}
	y3Ref, err := Conv2d3x3Pad(x, w3F16, 8, bias, 1, "same", 1, true)
	if err != nil {
		t.Fatalf("Conv2d3x3Pad: %v", err)
	}
	cmp3, err := CompareFP16(y3Q.Data, y3Ref.Data, 2e-2, 2e-2, nil)
	if err != nil || !cmp3.Passed {
		t.Fatalf("3x3 quant vs dequantized fp16 failed: err=%v maxAbs=%g", err, cmp3.MaxAbsErr)
	}
}

func TestEvalGraphQuantizedMultiLayerAndLiveVaneCompile(t *testing.T) {
	// Multi-layer mixed-precision quantized graph:
	// x [16, 8, 16] -> c1 (conv3x3 dense int4 + relu) -> c2 (conv1x1 int8 + zp) -> c3 (linear pal4)
	w1, err := QuantizeInt4(makeSyntheticWeightsF64(16, 16*9, 3), 16, 16*9, true, false)
	if err != nil {
		t.Fatalf("QuantizeInt4: %v", err)
	}
	b1 := makeConstWeights(16, []float64{0.125, -0.0625})

	w2, err := QuantizeInt8(makeSyntheticWeightsF64(16, 16, 5), 16, 16, true, true)
	if err != nil {
		t.Fatalf("QuantizeInt8: %v", err)
	}
	b2 := makeConstWeights(16, []float64{0.25, -0.125})

	w3, err := QuantizePalette(makeSyntheticWeightsF64(16, 16, 9), 4)
	if err != nil {
		t.Fatalf("QuantizePalette: %v", err)
	}
	b3 := makeConstWeights(16, []float64{0.0625, 0.0})

	spec := GraphSpec{
		Target:    "h17",
		Placement: "auto",
		Inputs:    []GraphInputSpec{{Name: "x", C: 16, H: 8, W: 16}},
		Nodes: []GraphNodeSpec{
			{
				ID: "c1", Op: "conv3x3", Mode: "dense", Cin: 16, Cout: 16, Stride: 1, Pad: "same",
				Relu: true, Inputs: []string{"x"}, Weights: "c1_codes.bin", Bias: json.RawMessage(`"c1_bias.fp16"`),
				Quant: &GraphQuantSpec{Format: "int4", Scales: "c1_scales.f32"},
			},
			{
				ID: "c2", Op: "conv1x1", Cin: 16, Cout: 16, Inputs: []string{"c1"},
				Weights: "c2_codes.bin", Bias: json.RawMessage(`"c2_bias.fp16"`),
				Quant: &GraphQuantSpec{Format: "int8", Scales: "c2_scales.f32", ZeroPoints: "c2_zp.u8"},
			},
			{
				ID: "c3", Op: "linear", Cin: 16, Cout: 16, Inputs: []string{"c2"},
				Weights: "c3_codes.bin", Bias: json.RawMessage(`"c3_bias.fp16"`),
				Quant: &GraphQuantSpec{Format: "pal4", Palette: "c3_pal.fp16"},
			},
		},
	}

	x := makeConstTensor(16, 8, 16, []float64{0.5, -0.25, 0.75, 0.125, -0.5})
	nw := map[string]GraphNodeWeights{
		"c1": {Quant: &w1, Bias: b1},
		"c2": {Quant: &w2, Bias: b2},
		"c3": {Quant: &w3, Bias: b3},
	}

	got, env, err := EvalGraph(spec, map[string]Tensor{"x": x}, nw)
	if err != nil {
		t.Fatalf("EvalGraph quantized: %v", err)
	}
	if got.C != 16 || got.H != 8 || got.W != 16 {
		t.Fatalf("unexpected quantized graph output shape (%d,%d,%d)", got.C, got.H, got.W)
	}
	if _, ok := env["c2"]; !ok {
		t.Fatalf("expected intermediate c2 in env")
	}

	// Cross-check live compilation with VANE_BIN when set
	vaneBin := os.Getenv("VANE_BIN")
	if vaneBin == "" {
		t.Log("VANE_BIN not set; skipping live vane compile-graph/CLI quantization checks")
		return
	}
	if !filepath.IsAbs(vaneBin) {
		if _, errStat := os.Stat(vaneBin); errStat != nil {
			repoCand := filepath.Join("..", "..", vaneBin)
			if _, errRepo := os.Stat(repoCand); errRepo == nil {
				vaneBin = repoCand
			}
		}
		if abs, errAbs := filepath.Abs(vaneBin); errAbs == nil {
			vaneBin = abs
		}
	}

	tmp := t.TempDir()
	mustWrite := func(name string, data []byte) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(tmp, name), data, 0600); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	mustWrite("c1_codes.bin", w1.Codes)
	mustWrite("c1_scales.f32", EncodeFloat32LE(w1.Scales))
	mustWrite("c1_bias.fp16", EncodeUint16LE(b1))
	mustWrite("c2_codes.bin", w2.Codes)
	mustWrite("c2_scales.f32", EncodeFloat32LE(w2.Scales))
	mustWrite("c2_zp.u8", w2.ZeroPoints)
	mustWrite("c2_bias.fp16", EncodeUint16LE(b2))
	mustWrite("c3_codes.bin", w3.Codes)
	mustWrite("c3_pal.fp16", EncodeUint16LE(w3.Palette))
	mustWrite("c3_bias.fp16", EncodeUint16LE(b3))

	for _, tgt := range []string{"h16", "h17"} {
		spec.Target = tgt
		specBytes, err := json.MarshalIndent(spec, "", "  ")
		if err != nil {
			t.Fatalf("marshal spec: %v", err)
		}
		specFile := filepath.Join(tmp, "graph_"+tgt+".json")
		if err := os.WriteFile(specFile, specBytes, 0600); err != nil {
			t.Fatalf("write spec: %v", err)
		}
		outDir := filepath.Join(tmp, "out_"+tgt)
		cmd := exec.Command(vaneBin, "compile-graph", "--spec", specFile, "--out-dir", outDir)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("vane compile-graph (%s) failed: %v\n%s", tgt, err, string(out))
		}
		valCmd := exec.Command(vaneBin, "validate", "--hwx", filepath.Join(outDir, "program.hwx"),
			"--sections", outDir, "--check-tasks")
		if out, err := valCmd.CombinedOutput(); err != nil {
			t.Fatalf("vane validate (%s) failed: %v\n%s", tgt, err, string(out))
		}
	}

	// Verify weight compression ratio on large linear (256->256): int4 & pal2 vs fp16
	const kDim, nDim = 256, 256
	wBig := makeSyntheticWeightsF64(nDim, kDim, 42)
	wBigFP16 := make([]uint16, len(wBig))
	for i, v := range wBig {
		wBigFP16[i] = fp16.FromFloat64(v)
	}
	wBigInt8, err := QuantizeInt8(wBig, nDim, kDim, true, false)
	if err != nil {
		t.Fatalf("QuantizeInt8 big: %v", err)
	}
	wBigInt4, err := QuantizeInt4(wBig, nDim, kDim, true, false)
	if err != nil {
		t.Fatalf("QuantizeInt4 big: %v", err)
	}
	wBigPal2, err := QuantizePalette(wBig, 2)
	if err != nil {
		t.Fatalf("QuantizePalette big: %v", err)
	}

	mustWrite("big_fp16.bin", EncodeUint16LE(wBigFP16))
	mustWrite("big_int8.bin", wBigInt8.Codes)
	mustWrite("big_int8.f32", EncodeFloat32LE(wBigInt8.Scales))
	mustWrite("big_int4.bin", wBigInt4.Codes)
	mustWrite("big_int4.f32", EncodeFloat32LE(wBigInt4.Scales))
	mustWrite("big_pal2.bin", wBigPal2.Codes)
	mustWrite("big_pal2.fp16", EncodeUint16LE(wBigPal2.Palette))

	compileLinearAndKernelBytes := func(tag string, extraArgs ...string) int {
		t.Helper()
		outDir := filepath.Join(tmp, "lin_"+tag)
		args := append([]string{
			"compile-linear", "--target", "h17", "--m", "16", "--k", "256", "--n", "256", "--out-dir", outDir,
		}, extraArgs...)
		cmd := exec.Command(vaneBin, args...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("compile-linear %s failed: %v\n%s", tag, err, string(out))
		}
		var man struct {
			KernelBytes int `json:"kernel_bytes"`
		}
		raw, err := os.ReadFile(filepath.Join(outDir, "manifest.json"))
		if err != nil {
			t.Fatalf("read manifest %s: %v", tag, err)
		}
		if err := json.Unmarshal(raw, &man); err != nil {
			t.Fatalf("unmarshal manifest %s: %v", tag, err)
		}
		return man.KernelBytes
	}

	kbFP16 := compileLinearAndKernelBytes("fp16", "--weights", filepath.Join(tmp, "big_fp16.bin"))
	kbInt8 := compileLinearAndKernelBytes("int8", "--weights", filepath.Join(tmp, "big_int8.bin"),
		"--weight-format", "int8", "--scales", filepath.Join(tmp, "big_int8.f32"))
	kbInt4 := compileLinearAndKernelBytes("int4", "--weights", filepath.Join(tmp, "big_int4.bin"),
		"--weight-format", "int4", "--scales", filepath.Join(tmp, "big_int4.f32"))
	kbPal2 := compileLinearAndKernelBytes("pal2", "--weights", filepath.Join(tmp, "big_pal2.bin"),
		"--weight-format", "pal2", "--palette", filepath.Join(tmp, "big_pal2.fp16"))

	if !(kbPal2 < kbInt4 && kbInt4 < kbInt8 && kbInt8 < kbFP16) {
		t.Fatalf("expected strict compression ordering pal2(%d) < int4(%d) < int8(%d) < fp16(%d)",
			kbPal2, kbInt4, kbInt8, kbFP16)
	}
	if float64(kbFP16)/float64(kbInt8) < 1.9 {
		t.Fatalf("expected ~2x INT8 compression, got fp16=%d int8=%d", kbFP16, kbInt8)
	}
	if float64(kbFP16)/float64(kbInt4) < 3.5 {
		t.Fatalf("expected ~4x INT4 compression, got fp16=%d int4=%d", kbFP16, kbInt4)
	}
	if float64(kbFP16)/float64(kbPal2) < 6.5 {
		t.Fatalf("expected ~8x PAL2 compression, got fp16=%d pal2=%d", kbFP16, kbPal2)
	}
}
