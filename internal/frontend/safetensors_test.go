package frontend

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/aurora-silicon/vane/internal/fp16"
)

func encodeF32LE(vals []float32) []byte {
	b := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return b
}

func encodeBF16LE(vals []float32) []byte {
	b := make([]byte, 2*len(vals))
	for i, v := range vals {
		// Round-to-nearest-even float32 -> bf16
		u := math.Float32bits(v)
		if (u & 0x7fffffff) > 0x7f800000 {
			// NaN
			binary.LittleEndian.PutUint16(b[2*i:], uint16((u>>16)|0x0040))
			continue
		}
		lsb := (u >> 16) & 1
		u += 0x7fff + lsb
		binary.LittleEndian.PutUint16(b[2*i:], uint16(u>>16))
	}
	return b
}

func encodeI64LE(vals []int64) []byte {
	b := make([]byte, 8*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint64(b[8*i:], uint64(v))
	}
	return b
}

func TestSafeTensorsRoundTripAndConversions(t *testing.T) {
	f16Vals := []uint16{fp16.FromFloat64(0.5), fp16.FromFloat64(-1.25), fp16.PosZero, fp16.FromFloat64(3.5)}
	f32Vals := []float32{1.0, -2.5, 0.125, 65504.0, 1.0 + 1.0/2048.0, 1.0 + 3.0/2048.0}
	bf16Vals := []float32{0.5, -3.0, 1.5, 16.0}
	i64Vals := []int64{1, 16, 8, 8}

	blob, err := WriteSafeTensors([]SafeTensorWrite{
		{Name: "w_f16", DType: "F16", Shape: []int64{2, 2}, Data: fp16.EncodeLE(f16Vals)},
		{Name: "w_f32", DType: "F32", Shape: []int64{2, 3}, Data: encodeF32LE(f32Vals)},
		{Name: "w_bf16", DType: "BF16", Shape: []int64{4}, Data: encodeBF16LE(bf16Vals)},
		{Name: "shape_const", DType: "I64", Shape: []int64{4}, Data: encodeI64LE(i64Vals)},
		{Name: "flag", DType: "BOOL", Shape: []int64{1}, Data: []byte{1}},
	}, map[string]string{
		"format":  "pt",
		"version": "1",
	})
	if err != nil {
		t.Fatalf("WriteSafeTensors: %v", err)
	}

	sf, err := ParseSafeTensorsWithOptions(blob, SafeTensorsOptions{Strict: true})
	if err != nil {
		t.Fatalf("ParseSafeTensorsWithOptions: %v", err)
	}

	if sf.Metadata["format"] != "pt" || sf.Metadata["version"] != "1" {
		t.Fatalf("unexpected metadata: %+v", sf.Metadata)
	}
	if _, hasMeta := sf.Tensors["__metadata__"]; hasMeta {
		t.Fatalf("__metadata__ must not appear in Tensors map")
	}
	if len(sf.Names) != 5 {
		t.Fatalf("expected 5 tensors, got %d (%v)", len(sf.Names), sf.Names)
	}

	gotF16, err := sf.ToFP16("w_f16")
	if err != nil {
		t.Fatalf("ToFP16(w_f16): %v", err)
	}
	for i := range f16Vals {
		if gotF16[i] != f16Vals[i] {
			t.Fatalf("w_f16[%d]: got 0x%04x, want 0x%04x", i, gotF16[i], f16Vals[i])
		}
	}

	gotF32, err := sf.ToFP16("w_f32")
	if err != nil {
		t.Fatalf("ToFP16(w_f32): %v", err)
	}
	for i, f := range f32Vals {
		want := fp16.FromFloat32(f)
		if gotF32[i] != want {
			t.Fatalf("w_f32[%d] (%g): got 0x%04x, want 0x%04x", i, f, gotF32[i], want)
		}
	}
	// Verify round-to-nearest-even ties: 1 + 1/2048 -> 1.0 (even LSB 0), 1 + 3/2048 -> 1 + 2/1024 (even LSB 0)
	if gotF32[4] != 0x3c00 || gotF32[5] != 0x3c02 {
		t.Fatalf("F32->F16 tie-to-even check failed: got 0x%04x, 0x%04x", gotF32[4], gotF32[5])
	}

	gotBF16, err := sf.ToFP16("w_bf16")
	if err != nil {
		t.Fatalf("ToFP16(w_bf16): %v", err)
	}
	for i, f := range bf16Vals {
		want := fp16.FromFloat32(f)
		if gotBF16[i] != want {
			t.Fatalf("w_bf16[%d] (%g): got 0x%04x, want 0x%04x", i, f, gotBF16[i], want)
		}
	}

	gotI64, err := sf.ToInt64("shape_const")
	if err != nil {
		t.Fatalf("ToInt64: %v", err)
	}
	for i := range i64Vals {
		if gotI64[i] != i64Vals[i] {
			t.Fatalf("shape_const[%d]: got %d, want %d", i, gotI64[i], i64Vals[i])
		}
	}

	if _, err := sf.ToFP16("shape_const"); err == nil {
		t.Fatalf("expected error exporting I64 tensor as fp16")
	}
	if _, err := sf.ToFP16("flag"); err == nil {
		t.Fatalf("expected error exporting BOOL tensor as fp16")
	}

	// Test ExportFP16Dir
	outDir := t.TempDir()
	exported, err := sf.ExportFP16Dir(outDir)
	if err != nil {
		t.Fatalf("ExportFP16Dir: %v", err)
	}
	if len(exported) != 3 {
		t.Fatalf("expected 3 exported float tensors, got %d", len(exported))
	}
	for _, ex := range exported {
		b, err := os.ReadFile(filepath.Join(outDir, ex.File))
		if err != nil {
			t.Fatalf("read exported %s: %v", ex.File, err)
		}
		if len(b) != ex.OutBytes {
			t.Fatalf("exported %s length %d != %d", ex.File, len(b), ex.OutBytes)
		}
	}
}

func TestSafeTensorsRejectsNonFiniteWeights(t *testing.T) {
	cases := []struct {
		name  string
		dtype string
		data  []byte
	}{
		{"f16_nan", "F16", fp16.EncodeLE([]uint16{fp16.FromFloat64(1.0), fp16.QuietNaN})},
		{"f16_posinf", "F16", fp16.EncodeLE([]uint16{fp16.PosInf, fp16.PosZero})},
		{"f16_neginf", "F16", fp16.EncodeLE([]uint16{fp16.NegInf, fp16.PosZero})},
		{"f32_nan", "F32", encodeF32LE([]float32{0.5, float32(math.NaN())})},
		{"f32_inf", "F32", encodeF32LE([]float32{float32(math.Inf(1)), 0.0})},
		{"f32_overflow_f16", "F32", encodeF32LE([]float32{100000.0, 0.0})},
		{"bf16_nan", "BF16", []byte{0xc0, 0x7f, 0x00, 0x00}},
		{"bf16_inf", "BF16", []byte{0x80, 0x7f, 0x00, 0x00}},
		{"bf16_overflow_f16", "BF16", encodeBF16LE([]float32{1e20, 0.0})},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			blob, err := WriteSafeTensors([]SafeTensorWrite{
				{Name: "t", DType: tc.dtype, Shape: []int64{2}, Data: tc.data},
			}, nil)
			if err != nil {
				t.Fatalf("WriteSafeTensors: %v", err)
			}
			sf, err := ParseSafeTensors(blob)
			if err != nil {
				t.Fatalf("ParseSafeTensors: %v", err)
			}
			if _, err := sf.ToFP16("t"); err == nil {
				t.Fatalf("expected ToFP16 to reject %s", tc.name)
			}
		})
	}
}

func makeRawSafeTensors(header string, body []byte) []byte {
	out := make([]byte, 8+len(header)+len(body))
	binary.LittleEndian.PutUint64(out[0:8], uint64(len(header)))
	copy(out[8:], header)
	copy(out[8+len(header):], body)
	return out
}

func TestSafeTensorsBoundsAndCorruptionGuards(t *testing.T) {
	t.Run("truncated_prefix", func(t *testing.T) {
		if _, err := ParseSafeTensors([]byte{1, 2, 3}); err == nil {
			t.Fatal("expected error for <8 byte buffer")
		}
	})

	t.Run("zero_header_len", func(t *testing.T) {
		buf := make([]byte, 16)
		if _, err := ParseSafeTensors(buf); err == nil {
			t.Fatal("expected error for zero header_len")
		}
	})

	t.Run("header_len_overflow_max_uint64", func(t *testing.T) {
		buf := make([]byte, 16)
		binary.LittleEndian.PutUint64(buf[0:8], math.MaxUint64)
		if _, err := ParseSafeTensors(buf); err == nil {
			t.Fatal("expected error for MaxUint64 header_len")
		}
	})

	t.Run("header_len_exceeds_limit", func(t *testing.T) {
		buf := make([]byte, 16)
		binary.LittleEndian.PutUint64(buf[0:8], MaxSafeTensorsHeaderBytes+1)
		if _, err := ParseSafeTensors(buf); err == nil {
			t.Fatal("expected error for header_len > 100MiB")
		}
	})

	t.Run("header_len_past_eof", func(t *testing.T) {
		buf := make([]byte, 16)
		binary.LittleEndian.PutUint64(buf[0:8], 64)
		if _, err := ParseSafeTensors(buf); err == nil {
			t.Fatal("expected error for header_len past EOF")
		}
	})

	t.Run("offset_past_eof", func(t *testing.T) {
		raw := makeRawSafeTensors(`{"w":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}}`, make([]byte, 4))
		if _, err := ParseSafeTensors(raw); err == nil {
			t.Fatal("expected error for offset past EOF")
		}
	})

	t.Run("begin_gt_end", func(t *testing.T) {
		raw := makeRawSafeTensors(`{"w":{"dtype":"F16","shape":[1],"data_offsets":[4,2]}}`, make([]byte, 8))
		if _, err := ParseSafeTensors(raw); err == nil {
			t.Fatal("expected error for begin > end")
		}
	})

	t.Run("span_mismatch", func(t *testing.T) {
		raw := makeRawSafeTensors(`{"w":{"dtype":"F16","shape":[3],"data_offsets":[0,4]}}`, make([]byte, 8))
		if _, err := ParseSafeTensors(raw); err == nil {
			t.Fatal("expected error for span != shape*dtype")
		}
	})

	t.Run("shape_product_overflow", func(t *testing.T) {
		raw := makeRawSafeTensors(`{"w":{"dtype":"F32","shape":[3037000500,3037000500],"data_offsets":[0,0]}}`, nil)
		if _, err := ParseSafeTensors(raw); err == nil {
			t.Fatal("expected error for shape product overflow")
		}
	})

	t.Run("overlapping_spans_strict", func(t *testing.T) {
		hdr, _ := json.Marshal(map[string]any{
			"a": map[string]any{"dtype": "F16", "shape": []int{2}, "data_offsets": []int{0, 4}},
			"b": map[string]any{"dtype": "F16", "shape": []int{2}, "data_offsets": []int{2, 6}},
		})
		raw := makeRawSafeTensors(string(hdr), make([]byte, 6))
		if _, err := ParseSafeTensors(raw); err != nil {
			t.Fatalf("non-strict mode should permit overlapping spans, got: %v", err)
		}
		if _, err := ParseSafeTensorsWithOptions(raw, SafeTensorsOptions{Strict: true}); err == nil || !strings.Contains(err.Error(), "overlapping") {
			t.Fatalf("strict mode should reject overlapping spans, got: %v", err)
		}
	})

	t.Run("trailing_non_json_garbage", func(t *testing.T) {
		raw := makeRawSafeTensors(`{"w":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}} GARBAGE`, make([]byte, 2))
		if _, err := ParseSafeTensors(raw); err == nil {
			t.Fatal("expected error for trailing non-JSON garbage in header")
		}
	})

	t.Run("trailing_json_object", func(t *testing.T) {
		raw := makeRawSafeTensors(`{"w":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}{"extra":1}`, make([]byte, 2))
		if _, err := ParseSafeTensors(raw); err == nil {
			t.Fatal("expected error for trailing JSON object in header")
		}
	})

	t.Run("missing_shape_field", func(t *testing.T) {
		raw := makeRawSafeTensors(`{"w":{"dtype":"F16","data_offsets":[0,2]}}`, make([]byte, 2))
		if _, err := ParseSafeTensors(raw); err == nil || !strings.Contains(err.Error(), "shape") {
			t.Fatalf("expected error for omitted shape field, got: %v", err)
		}
	})

	t.Run("null_shape_field", func(t *testing.T) {
		raw := makeRawSafeTensors(`{"w":{"dtype":"F16","shape":null,"data_offsets":[0,2]}}`, make([]byte, 2))
		if _, err := ParseSafeTensors(raw); err == nil || !strings.Contains(err.Error(), "shape") {
			t.Fatalf("expected error for null shape field, got: %v", err)
		}
	})

	t.Run("explicit_scalar_empty_shape_allowed", func(t *testing.T) {
		raw := makeRawSafeTensors(`{"s":{"dtype":"F16","shape":[],"data_offsets":[0,2]}}`, fp16.EncodeLE([]uint16{fp16.FromFloat64(1.5)}))
		sf, err := ParseSafeTensors(raw)
		if err != nil {
			t.Fatalf("explicit empty scalar shape should succeed, got: %v", err)
		}
		if len(sf.Tensors["s"].Shape) != 0 {
			t.Fatalf("expected scalar shape [], got %v", sf.Tensors["s"].Shape)
		}
	})
}
