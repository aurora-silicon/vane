package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/aurora-silicon/vane/internal/fp16"
	"github.com/aurora-silicon/vane/internal/frontend"
	"github.com/aurora-silicon/vane/internal/ref"
)

func buildTestResidualONNX() []byte {
	encodeF32 := func(vals []float32) []byte {
		b := make([]byte, 4*len(vals))
		for i, v := range vals {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
		}
		return b
	}
	mkVI := func(name string, dims []int64, symBatch bool) *frontend.ProtoBuilder {
		var sh frontend.ProtoBuilder
		for i, d := range dims {
			var db frontend.ProtoBuilder
			if i == 0 && symBatch {
				db.WriteString(2, "batch")
			} else {
				db.WriteInt64(1, d)
			}
			sh.WriteMessage(1, &db)
		}
		var tt, tp, vi frontend.ProtoBuilder
		tt.WriteInt64(1, int64(frontend.ONNXFloat))
		tt.WriteMessage(2, &sh)
		tp.WriteMessage(1, &tt)
		vi.WriteString(1, name)
		vi.WriteMessage(2, &tp)
		return &vi
	}
	mkF32Tensor := func(name string, dims []int64, vals []float32) *frontend.ProtoBuilder {
		var tb frontend.ProtoBuilder
		tb.WritePackedInt64(1, dims)
		tb.WriteInt64(2, int64(frontend.ONNXFloat))
		tb.WriteString(8, name)
		tb.WriteBytes(9, encodeF32(vals))
		return &tb
	}
	mkAttrInts := func(name string, vals []int64) *frontend.ProtoBuilder {
		var ab frontend.ProtoBuilder
		ab.WriteString(1, name)
		ab.WritePackedInt64(8, vals)
		ab.WriteInt64(20, int64(frontend.AttrInts))
		return &ab
	}
	mkAttrFloat := func(name string, v float32) *frontend.ProtoBuilder {
		var ab frontend.ProtoBuilder
		ab.WriteString(1, name)
		ab.WriteFloat32(2, v)
		ab.WriteInt64(20, int64(frontend.AttrFloat))
		return &ab
	}
	mkNode := func(op, name string, ins, outs []string, attrs ...*frontend.ProtoBuilder) *frontend.ProtoBuilder {
		var nb frontend.ProtoBuilder
		for _, in := range ins {
			nb.WriteString(1, in)
		}
		for _, out := range outs {
			nb.WriteString(2, out)
		}
		nb.WriteString(3, name)
		nb.WriteString(4, op)
		for _, a := range attrs {
			nb.WriteMessage(5, a)
		}
		return &nb
	}

	// 4->4 conv3x3 + BN + ReLU + residual add
	w := make([]float32, 4*4*9)
	for i := range w {
		w[i] = float32((i%5)-2) * 0.05
	}
	scale := []float32{1.0, 1.2, 0.8, 1.1}
	bias := []float32{0.05, -0.05, 0.1, 0.0}
	mean := []float32{0.0, 0.1, -0.05, 0.02}
	vari := []float32{0.5, 0.8, 0.6, 0.4}

	var gb frontend.ProtoBuilder
	gb.WriteMessage(1, mkNode("Conv", "conv1", []string{"x", "w1"}, []string{"c1"},
		mkAttrInts("strides", []int64{1, 1}), mkAttrInts("pads", []int64{1, 1, 1, 1})))
	gb.WriteMessage(1, mkNode("BatchNormalization", "bn1", []string{"c1", "s", "b", "m", "v"}, []string{"b1"},
		mkAttrFloat("epsilon", 1e-5)))
	gb.WriteMessage(1, mkNode("Relu", "relu1", []string{"b1"}, []string{"r1"}))
	gb.WriteMessage(1, mkNode("Add", "res", []string{"r1", "x"}, []string{"y"}))
	gb.WriteString(2, "cli_res_graph")
	gb.WriteMessage(5, mkF32Tensor("w1", []int64{4, 4, 3, 3}, w))
	gb.WriteMessage(5, mkF32Tensor("s", []int64{4}, scale))
	gb.WriteMessage(5, mkF32Tensor("b", []int64{4}, bias))
	gb.WriteMessage(5, mkF32Tensor("m", []int64{4}, mean))
	gb.WriteMessage(5, mkF32Tensor("v", []int64{4}, vari))
	gb.WriteMessage(11, mkVI("x", []int64{1, 4, 4, 4}, true))
	gb.WriteMessage(12, mkVI("y", []int64{1, 4, 4, 4}, false))

	var op, mb frontend.ProtoBuilder
	op.WriteString(1, "")
	op.WriteInt64(2, 17)
	mb.WriteInt64(1, 9)
	mb.WriteMessage(7, &gb)
	mb.WriteMessage(8, &op)
	return mb.Bytes()
}

func TestCLIImportONNXEndToEnd(t *testing.T) {
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "res.onnx")
	if err := os.WriteFile(modelPath, buildTestResidualONNX(), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	outDir := filepath.Join(tmp, "imported")

	var stdout, stderr bytes.Buffer
	rc := runCLI([]string{
		"import-onnx",
		"--model", modelPath,
		"--out", outDir,
		"--target", "h17",
		"--input-shape", "x=4,4,4",
		"--json",
	}, nil, &stdout, &stderr)
	if rc != 0 {
		t.Fatalf("import-onnx failed (rc=%d): stderr=%s", rc, stderr.String())
	}

	var report struct {
		Target        string `json:"target"`
		NodeCount     int    `json:"node_count"`
		FusedBNCount  int    `json:"fused_bn_count"`
		FusedActCount int    `json:"fused_act_count"`
		GraphFile     string `json:"graph_file"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal import-onnx JSON: %v\nstdout=%s", err, stdout.String())
	}
	if report.Target != "h17" || report.NodeCount != 2 || report.FusedBNCount != 1 || report.FusedActCount != 1 {
		t.Fatalf("unexpected import-onnx report: %+v", report)
	}

	// Load emitted graph.json directly into internal/ref.EvalGraphF64!
	rawGraph, err := os.ReadFile(report.GraphFile)
	if err != nil {
		t.Fatalf("read graph.json: %v", err)
	}
	var spec ref.GraphSpec
	if err := json.Unmarshal(rawGraph, &spec); err != nil {
		t.Fatalf("unmarshal ref.GraphSpec: %v", err)
	}
	wb, err := os.ReadFile(filepath.Join(outDir, spec.Nodes[0].Weights))
	if err != nil {
		t.Fatalf("read weights: %v", err)
	}
	w16, _ := fp16.DecodeLE(wb)
	_, bFile, err := spec.Nodes[0].ConvBiasSpec()
	if err != nil || bFile == "" {
		t.Fatalf("expected folded bias file on conv1, got %q (err=%v)", bFile, err)
	}
	bb, err := os.ReadFile(filepath.Join(outDir, bFile))
	if err != nil {
		t.Fatalf("read bias: %v", err)
	}
	b16, _ := fp16.DecodeLE(bb)

	x, _ := ref.NewTensor(1, 4, 4, 4)
	for i := range x.Data {
		x.Data[i] = fp16.FromFloat64(float64((i%7)-3) * 0.25)
	}
	y, _, err := ref.EvalGraphF64(spec, map[string]ref.Tensor{"x": x}, map[string]ref.GraphNodeWeights{
		spec.Nodes[0].ID: {Weights: w16, Bias: b16},
	})
	if err != nil {
		t.Fatalf("EvalGraphF64: %v", err)
	}
	if y.C != 4 || y.H != 4 || y.W != 4 {
		t.Fatalf("unexpected output shape (%d,%d,%d)", y.C, y.H, y.W)
	}
}

func TestCLIInspectSafeTensorsAndExport(t *testing.T) {
	tmp := t.TempDir()
	sfPath := filepath.Join(tmp, "weights.safetensors")
	exportDir := filepath.Join(tmp, "exported")

	f32Bytes := make([]byte, 16)
	for i, v := range []float32{0.5, -1.25, 2.0, 0.125} {
		binary.LittleEndian.PutUint32(f32Bytes[4*i:], math.Float32bits(v))
	}
	blob, err := frontend.WriteSafeTensors([]frontend.SafeTensorWrite{
		{Name: "conv.weight", DType: "F32", Shape: []int64{2, 2}, Data: f32Bytes},
		{Name: "conv.bias", DType: "F16", Shape: []int64{2}, Data: fp16.EncodeLE([]uint16{fp16.FromFloat64(0.25), fp16.FromFloat64(-0.5)})},
	}, map[string]string{"format": "pt"})
	if err != nil {
		t.Fatalf("WriteSafeTensors: %v", err)
	}
	if err := os.WriteFile(sfPath, blob, 0o644); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}

	var stdout, stderr bytes.Buffer
	rc := runCLI([]string{
		"inspect-safetensors",
		"--file", sfPath,
		"--export-dir", exportDir,
		"--strict",
		"--json",
	}, nil, &stdout, &stderr)
	if rc != 0 {
		t.Fatalf("inspect-safetensors failed (rc=%d): stderr=%s", rc, stderr.String())
	}

	var rep struct {
		TensorCnt int                       `json:"tensor_count"`
		Tensors   []frontend.SafeTensorInfo `json:"tensors"`
		Exported  []frontend.ExportedTensor `json:"exported"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal inspect JSON: %v\nstdout=%s", err, stdout.String())
	}
	if rep.TensorCnt != 2 || len(rep.Exported) != 2 {
		t.Fatalf("unexpected inspect report: %+v", rep)
	}

	// Verify exported fp16 file contents
	expBytes, err := os.ReadFile(filepath.Join(exportDir, "conv.weight.bin"))
	if err != nil {
		t.Fatalf("read exported conv.weight.bin: %v", err)
	}
	vals, err := fp16.DecodeLE(expBytes)
	if err != nil || len(vals) != 4 || vals[0] != fp16.FromFloat64(0.5) || vals[1] != fp16.FromFloat64(-1.25) {
		t.Fatalf("unexpected exported fp16 values: %v (err=%v)", vals, err)
	}

	// Also verify text output mode and missing-flag error handling
	stdout.Reset()
	stderr.Reset()
	rc = runCLI([]string{"inspect-safetensors", "--file", sfPath}, nil, &stdout, &stderr)
	if rc != 0 || !strings.Contains(stdout.String(), "conv.weight") {
		t.Fatalf("text inspect-safetensors failed: rc=%d out=%q err=%q", rc, stdout.String(), stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	if rc := runCLI([]string{"import-onnx"}, nil, &stdout, &stderr); rc == 0 {
		t.Fatal("expected non-zero exit when --model is omitted")
	}
}

func TestCLIImportONNXCompileGraphWithVane(t *testing.T) {
	vaneBin := findVaneBin(t)
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "res.onnx")
	if err := os.WriteFile(modelPath, buildTestResidualONNX(), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}
	importedDir := filepath.Join(tmp, "imported")
	compiledDir := filepath.Join(tmp, "compiled")

	var stdout, stderr bytes.Buffer
	rc := runCLI([]string{
		"import-onnx",
		"--model", modelPath,
		"--out", importedDir,
		"--target", "h17",
		"--input-shape", "x=4,4,4",
	}, nil, &stdout, &stderr)
	if rc != 0 {
		t.Fatalf("import-onnx failed (rc=%d): stderr=%s", rc, stderr.String())
	}

	out := runVane(t, vaneBin, "compile-graph", "--spec", filepath.Join(importedDir, "graph.json"), "--out-dir", compiledDir)
	if !strings.Contains(out, "compiled h17 graph") {
		t.Fatalf("unexpected compile-graph output: %s", out)
	}
	if _, err := os.Stat(filepath.Join(compiledDir, "program.hwx")); err != nil {
		t.Fatalf("missing compiled program.hwx: %v", err)
	}
}
