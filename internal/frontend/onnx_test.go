package frontend

import (
	"math"
	"strings"
	"testing"

	"github.com/aurora-silicon/vane/internal/fp16"
)

// Test helper builders for constructing genuine binary ONNX protobuf models.

func buildValueInfo(name string, elemType int32, dims []ONNXDim) *ProtoBuilder {
	var shape ProtoBuilder
	for _, d := range dims {
		var db ProtoBuilder
		if d.HasValue {
			db.WriteInt64(1, d.Value)
		} else if d.Param != "" {
			db.WriteString(2, d.Param)
		}
		shape.WriteMessage(1, &db)
	}
	var tt ProtoBuilder
	tt.WriteInt64(1, int64(elemType))
	tt.WriteMessage(2, &shape)

	var tp ProtoBuilder
	tp.WriteMessage(1, &tt)

	var vi ProtoBuilder
	vi.WriteString(1, name)
	vi.WriteMessage(2, &tp)
	return &vi
}

func buildStaticValueInfo(name string, elemType int32, shape []int64) *ProtoBuilder {
	dims := make([]ONNXDim, len(shape))
	for i, v := range shape {
		dims[i] = ONNXDim{HasValue: true, Value: v}
	}
	return buildValueInfo(name, elemType, dims)
}

func buildFloatTensor(name string, dims []int64, vals []float32, useRaw bool) *ProtoBuilder {
	var tb ProtoBuilder
	tb.WritePackedInt64(1, dims)
	tb.WriteInt64(2, int64(ONNXFloat))
	tb.WriteString(8, name)
	if useRaw {
		tb.WriteBytes(9, encodeF32LE(vals))
	} else {
		tb.WritePackedFloat32(4, vals)
	}
	return &tb
}

func buildFP16Tensor(name string, dims []int64, vals []uint16, useInt32Data bool) *ProtoBuilder {
	var tb ProtoBuilder
	tb.WritePackedInt64(1, dims)
	tb.WriteInt64(2, int64(ONNXFloat16))
	tb.WriteString(8, name)
	if useInt32Data {
		i32 := make([]int32, len(vals))
		for i, u := range vals {
			i32[i] = int32(u)
		}
		tb.WritePackedInt32(5, i32)
	} else {
		tb.WriteBytes(9, fp16.EncodeLE(vals))
	}
	return &tb
}

func buildBF16Tensor(name string, dims []int64, vals []float32) *ProtoBuilder {
	var tb ProtoBuilder
	tb.WritePackedInt64(1, dims)
	tb.WriteInt64(2, int64(ONNXBFloat16))
	tb.WriteString(8, name)
	tb.WriteBytes(9, encodeBF16LE(vals))
	return &tb
}

func buildInt64Tensor(name string, dims []int64, vals []int64, useRaw bool) *ProtoBuilder {
	var tb ProtoBuilder
	tb.WritePackedInt64(1, dims)
	tb.WriteInt64(2, int64(ONNXInt64))
	tb.WriteString(8, name)
	if useRaw {
		tb.WriteBytes(9, encodeI64LE(vals))
	} else {
		tb.WritePackedInt64(7, vals)
	}
	return &tb
}

func buildAttrInt(name string, v int64) *ProtoBuilder {
	var ab ProtoBuilder
	ab.WriteString(1, name)
	ab.WriteInt64(3, v)
	ab.WriteInt64(20, int64(AttrInt))
	return &ab
}

func buildAttrFloat(name string, v float32) *ProtoBuilder {
	var ab ProtoBuilder
	ab.WriteString(1, name)
	ab.WriteFloat32(2, v)
	ab.WriteInt64(20, int64(AttrFloat))
	return &ab
}

func buildAttrString(name string, s string) *ProtoBuilder {
	var ab ProtoBuilder
	ab.WriteString(1, name)
	ab.WriteString(4, s)
	ab.WriteInt64(20, int64(AttrString))
	return &ab
}

func buildAttrInts(name string, vals []int64, packed bool) *ProtoBuilder {
	var ab ProtoBuilder
	ab.WriteString(1, name)
	if packed {
		ab.WritePackedInt64(8, vals)
	} else {
		for _, v := range vals {
			ab.WriteInt64(8, v)
		}
	}
	ab.WriteInt64(20, int64(AttrInts))
	return &ab
}

func buildAttrFloats(name string, vals []float32, packed bool) *ProtoBuilder {
	var ab ProtoBuilder
	ab.WriteString(1, name)
	if packed {
		ab.WritePackedFloat32(7, vals)
	} else {
		for _, v := range vals {
			ab.WriteFloat32(7, v)
		}
	}
	ab.WriteInt64(20, int64(AttrFloats))
	return &ab
}

func buildAttrTensor(name string, tb *ProtoBuilder) *ProtoBuilder {
	var ab ProtoBuilder
	ab.WriteString(1, name)
	ab.WriteMessage(5, tb)
	ab.WriteInt64(20, int64(AttrTensor))
	return &ab
}

func buildNode(opType, name string, inputs, outputs []string, attrs ...*ProtoBuilder) *ProtoBuilder {
	var nb ProtoBuilder
	for _, in := range inputs {
		nb.WriteString(1, in)
	}
	for _, out := range outputs {
		nb.WriteString(2, out)
	}
	if name != "" {
		nb.WriteString(3, name)
	}
	nb.WriteString(4, opType)
	for _, a := range attrs {
		nb.WriteMessage(5, a)
	}
	return &nb
}

func buildModel(opset int64, graphName string, inputs, outputs, initializers, nodes []*ProtoBuilder) []byte {
	var gb ProtoBuilder
	for _, n := range nodes {
		gb.WriteMessage(1, n)
	}
	gb.WriteString(2, graphName)
	for _, init := range initializers {
		gb.WriteMessage(5, init)
	}
	for _, in := range inputs {
		gb.WriteMessage(11, in)
	}
	for _, out := range outputs {
		gb.WriteMessage(12, out)
	}

	var op ProtoBuilder
	op.WriteString(1, "")
	op.WriteInt64(2, opset)

	var mb ProtoBuilder
	mb.WriteInt64(1, 9) // ir_version
	mb.WriteString(2, "vane-test")
	mb.WriteMessage(7, &gb)
	mb.WriteMessage(8, &op)
	return mb.Bytes()
}

func TestParseONNXFullModelAndTensorEncodings(t *testing.T) {
	// Include packed + unpacked attributes, raw + typed + int32_data fp16 + bf16 initializers,
	// symbolic input dimension, and unknown fields skipped cleanly.
	wF32Raw := buildFloatTensor("w_f32_raw", []int64{2, 2, 1, 1}, []float32{0.5, -0.25, 0.75, 1.0}, true)
	wF32Typed := buildFloatTensor("w_f32_typed", []int64{2}, []float32{0.125, -0.5}, false)
	wF16Raw := buildFP16Tensor("w_f16_raw", []int64{2}, []uint16{fp16.FromFloat64(1.5), fp16.FromFloat64(-2.0)}, false)
	wF16I32 := buildFP16Tensor("w_f16_i32", []int64{2}, []uint16{fp16.FromFloat64(0.25), fp16.FromFloat64(3.0)}, true)
	wBF16 := buildBF16Tensor("w_bf16", []int64{2}, []float32{2.0, -4.0})
	axesI64 := buildInt64Tensor("axes", []int64{2}, []int64{2, 3}, false)

	// Unpacked dims & negative ints test
	var unpackedTensor ProtoBuilder
	unpackedTensor.WriteInt64(1, 2)
	unpackedTensor.WriteInt64(1, 1)
	unpackedTensor.WriteInt64(2, int64(ONNXInt64))
	unpackedTensor.WriteString(8, "neg_shape")
	unpackedTensor.WriteInt64(7, 1)
	unpackedTensor.WriteInt64(7, -1)

	// Node with both packed and unpacked attributes + unknown field 99
	convNode := buildNode("Conv", "conv0", []string{"x", "w_f32_raw", "w_f32_typed"}, []string{"y"},
		buildAttrInts("strides", []int64{1, 1}, true),
		buildAttrInts("pads", []int64{0, 0, 0, 0}, false),
		buildAttrFloats("scales", []float32{1.0, 2.0}, false),
		buildAttrFloat("epsilon", 1e-5),
		buildAttrInt("group", 1),
		buildAttrString("auto_pad", "NOTSET"),
	)
	convNode.WriteString(99, "ignored_unknown_field")

	inVI := buildValueInfo("x", ONNXFloat, []ONNXDim{
		{Param: "batch"},
		{HasValue: true, Value: 2},
		{HasValue: true, Value: 4},
		{HasValue: true, Value: 4},
	})
	outVI := buildStaticValueInfo("y", ONNXFloat, []int64{1, 2, 4, 4})

	raw := buildModel(17, "test_graph",
		[]*ProtoBuilder{inVI},
		[]*ProtoBuilder{outVI},
		[]*ProtoBuilder{wF32Raw, wF32Typed, wF16Raw, wF16I32, wBF16, axesI64, &unpackedTensor},
		[]*ProtoBuilder{convNode},
	)

	model, err := ParseONNX(raw)
	if err != nil {
		t.Fatalf("ParseONNX: %v", err)
	}
	if model.IRVersion != 9 || model.DefaultOpsetVersion() != 17 {
		t.Fatalf("unexpected header: ir=%d opset=%d", model.IRVersion, model.DefaultOpsetVersion())
	}
	if model.Graph.Name != "test_graph" || len(model.Graph.Nodes) != 1 || len(model.Graph.Initializers) != 7 {
		t.Fatalf("unexpected graph summary: name=%q nodes=%d inits=%d",
			model.Graph.Name, len(model.Graph.Nodes), len(model.Graph.Initializers))
	}

	// Verify symbolic dim
	if len(model.Graph.Inputs) != 1 || model.Graph.Inputs[0].Dims[0].Param != "batch" || model.Graph.Inputs[0].Dims[1].Value != 2 {
		t.Fatalf("unexpected input ValueInfo: %+v", model.Graph.Inputs)
	}

	// Verify node attributes
	nd := model.Graph.Nodes[0]
	pads, ok := nd.AttrInts("pads")
	if !ok || len(pads) != 4 {
		t.Fatalf("expected 4 pads, got %v", pads)
	}
	if got := nd.AttrInt("group", 0); got != 1 {
		t.Fatalf("expected group=1, got %d", got)
	}
	if got := nd.AttrString("auto_pad", ""); got != "NOTSET" {
		t.Fatalf("expected auto_pad=NOTSET, got %q", got)
	}
	if math.Abs(nd.AttrFloat("epsilon", 0)-1e-5) > 1e-7 {
		t.Fatalf("unexpected epsilon: %g", nd.AttrFloat("epsilon", 0))
	}

	// Verify tensor decoding across all encodings
	inits := map[string]*ONNXTensor{}
	for i := range model.Graph.Initializers {
		inits[model.Graph.Initializers[i].Name] = &model.Graph.Initializers[i]
	}
	vRaw, err := inits["w_f32_raw"].ToFloat64()
	if err != nil || len(vRaw) != 4 || vRaw[0] != 0.5 || vRaw[1] != -0.25 {
		t.Fatalf("w_f32_raw decode failed: %v, %v", vRaw, err)
	}
	vTyped, err := inits["w_f32_typed"].ToFloat64()
	if err != nil || len(vTyped) != 2 || vTyped[1] != -0.5 {
		t.Fatalf("w_f32_typed decode failed: %v, %v", vTyped, err)
	}
	vF16Raw, err := inits["w_f16_raw"].ToFloat64()
	if err != nil || vF16Raw[0] != 1.5 || vF16Raw[1] != -2.0 {
		t.Fatalf("w_f16_raw decode failed: %v, %v", vF16Raw, err)
	}
	vF16I32, err := inits["w_f16_i32"].ToFloat64()
	if err != nil || vF16I32[0] != 0.25 || vF16I32[1] != 3.0 {
		t.Fatalf("w_f16_i32 decode failed: %v, %v", vF16I32, err)
	}
	vBF16, err := inits["w_bf16"].ToFloat64()
	if err != nil || vBF16[0] != 2.0 || vBF16[1] != -4.0 {
		t.Fatalf("w_bf16 decode failed: %v, %v", vBF16, err)
	}
	vNeg, err := inits["neg_shape"].ToInt64()
	if err != nil || len(vNeg) != 2 || vNeg[0] != 1 || vNeg[1] != -1 {
		t.Fatalf("neg_shape decode failed: %v, %v", vNeg, err)
	}
}

func TestProtoDecoderMalformedAndDepthGuards(t *testing.T) {
	t.Run("truncated_varint", func(t *testing.T) {
		r := NewProtoReader([]byte{0x08, 0x80, 0x80})
		if _, _, err := r.NextTag(); err != nil {
			t.Fatalf("tag should succeed: %v", err)
		}
		if _, err := r.ReadVarint(); err == nil {
			t.Fatal("expected error for truncated varint")
		}
	})

	t.Run("varint_overflow", func(t *testing.T) {
		// 10-byte varint with 10th byte = 0x02 (> 0x01)
		buf := []byte{0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02}
		r := NewProtoReader(buf)
		if _, err := r.ReadVarint(); err == nil {
			t.Fatal("expected varint overflow error")
		}
	})

	t.Run("unsupported_wire_types", func(t *testing.T) {
		for _, wt := range []uint8{3, 4, 6, 7} {
			tag := (1 << 3) | wt
			r := NewProtoReader([]byte{tag})
			if _, _, err := r.NextTag(); err == nil {
				t.Fatalf("expected error for unsupported wire type %d", wt)
			}
		}
	})

	t.Run("truncated_length_delimited", func(t *testing.T) {
		// field 1, wire type 2, length 10, only 2 bytes present
		r := NewProtoReader([]byte{0x0a, 0x0a, 0x01, 0x02})
		_, wt, err := r.NextTag()
		if err != nil || wt != WireBytes {
			t.Fatalf("NextTag: wt=%d err=%v", wt, err)
		}
		if _, err := r.ReadBytes(); err == nil {
			t.Fatal("expected error for truncated bytes field")
		}
	})

	t.Run("recursion_depth_cap", func(t *testing.T) {
		r := NewProtoReader([]byte{})
		var err error
		for i := 0; i < MaxProtoDepth; i++ {
			r, err = r.SubReader([]byte{})
			if err != nil {
				t.Fatalf("depth %d should succeed, got %v", i+1, err)
			}
		}
		if _, err := r.SubReader([]byte{}); err == nil || !strings.Contains(err.Error(), "recursion depth") {
			t.Fatalf("expected recursion depth error at depth %d, got %v", MaxProtoDepth+1, err)
		}
	})

	t.Run("external_data_rejected", func(t *testing.T) {
		var tb ProtoBuilder
		tb.WritePackedInt64(1, []int64{4})
		tb.WriteInt64(2, int64(ONNXFloat))
		tb.WriteString(8, "ext_w")
		tb.WriteInt64(14, int64(ONNXDataExternal))
		raw := buildModel(17, "ext", []*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 1, 1})},
			[]*ProtoBuilder{buildStaticValueInfo("x", ONNXFloat, []int64{1, 4, 1, 1})},
			[]*ProtoBuilder{&tb}, nil)
		m, err := ParseONNX(raw)
		if err != nil {
			t.Fatalf("ParseONNX: %v", err)
		}
		if _, err := m.Graph.Initializers[0].ToFloat64(); err == nil || !strings.Contains(err.Error(), "EXTERNAL") {
			t.Fatalf("expected EXTERNAL data error, got %v", err)
		}
	})
}
