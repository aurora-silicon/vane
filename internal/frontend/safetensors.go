// Package frontend implements pure-stdlib SafeTensors and ONNX model decoders
// and lowering into Vane's compile-graph JSON specification and fp16 weight files.
package frontend

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/aurora-silicon/vane/internal/fp16"
)

// MaxSafeTensorsHeaderBytes caps the JSON header size to defend against
// oversized allocation attempts.
const MaxSafeTensorsHeaderBytes uint64 = 100 << 20 // 100 MiB

// SafeTensorsOptions controls optional validation checks when parsing a
// SafeTensors buffer.
type SafeTensorsOptions struct {
	// Strict rejects overlapping tensor byte spans in the data buffer.
	Strict bool
}

// SafeTensorInfo describes one tensor entry in a SafeTensors header.
type SafeTensorInfo struct {
	Name        string   `json:"name"`
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
	ByteLen     int64    `json:"byte_len"`
}

// SafeTensorsFile holds a validated SafeTensors archive in memory.
type SafeTensorsFile struct {
	Metadata map[string]string
	Tensors  map[string]SafeTensorInfo
	Names    []string

	dataBase int64
	raw      []byte
}

// SafeTensorWrite specifies one tensor to encode with WriteSafeTensors.
type SafeTensorWrite struct {
	Name  string
	DType string
	Shape []int64
	Data  []byte
}

// ExportedTensor records one tensor exported to an fp16 binary file.
type ExportedTensor struct {
	Name     string  `json:"name"`
	DType    string  `json:"dtype"`
	Shape    []int64 `json:"shape"`
	File     string  `json:"file"`
	OutBytes int     `json:"out_bytes"`
}

// SafeTensorDTypeBytes returns the element width in bytes for a recognized
// SafeTensors dtype string.
func SafeTensorDTypeBytes(dtype string) (int64, bool) {
	switch dtype {
	case "BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3":
		return 1, true
	case "F16", "BF16", "I16", "U16":
		return 2, true
	case "F32", "I32", "U32":
		return 4, true
	case "F64", "I64", "U64":
		return 8, true
	default:
		return 0, false
	}
}

func mulInt64Checked(a, b int64) (int64, error) {
	if a < 0 || b < 0 {
		return 0, fmt.Errorf("negative dimension or size (%d * %d)", a, b)
	}
	if a == 0 || b == 0 {
		return 0, nil
	}
	if a > math.MaxInt64/b {
		return 0, fmt.Errorf("integer overflow computing %d * %d", a, b)
	}
	return a * b, nil
}

func shapeProductChecked(shape []int64) (int64, error) {
	prod := int64(1)
	for i, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("negative shape dim[%d]=%d", i, d)
		}
		var err error
		prod, err = mulInt64Checked(prod, d)
		if err != nil {
			return 0, fmt.Errorf("shape %v element count overflow: %w", shape, err)
		}
	}
	return prod, nil
}

type rawSafeTensorEntry struct {
	DType       string   `json:"dtype"`
	Shape       *[]int64 `json:"shape"`
	DataOffsets []int64  `json:"data_offsets"`
}

// LoadSafeTensors reads and parses a .safetensors file from disk with default options.
func LoadSafeTensors(path string) (*SafeTensorsFile, error) {
	return LoadSafeTensorsWithOptions(path, SafeTensorsOptions{})
}

// LoadSafeTensorsWithOptions reads and parses a .safetensors file from disk.
func LoadSafeTensorsWithOptions(path string, opts SafeTensorsOptions) (*SafeTensorsFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read safetensors %q: %w", path, err)
	}
	return ParseSafeTensorsWithOptions(data, opts)
}

// ParseSafeTensors decodes and validates a SafeTensors binary buffer with default options.
func ParseSafeTensors(data []byte) (*SafeTensorsFile, error) {
	return ParseSafeTensorsWithOptions(data, SafeTensorsOptions{})
}

// ParseSafeTensorsWithOptions decodes and validates a SafeTensors binary buffer.
func ParseSafeTensorsWithOptions(data []byte, opts SafeTensorsOptions) (*SafeTensorsFile, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("safetensors truncated: need at least 8 bytes, got %d", len(data))
	}
	headerLen := binary.LittleEndian.Uint64(data[0:8])
	if headerLen == 0 {
		return nil, fmt.Errorf("safetensors header_len must be > 0")
	}
	if headerLen > MaxSafeTensorsHeaderBytes {
		return nil, fmt.Errorf("safetensors header_len %d exceeds limit %d", headerLen, MaxSafeTensorsHeaderBytes)
	}
	avail := uint64(len(data) - 8)
	if headerLen > avail {
		return nil, fmt.Errorf("safetensors header_len %d exceeds remaining file size %d", headerLen, avail)
	}

	headerBytes := data[8 : 8+headerLen]
	if !utf8.Valid(headerBytes) {
		return nil, fmt.Errorf("safetensors header is not valid UTF-8")
	}

	var top map[string]json.RawMessage
	dec := json.NewDecoder(bytes.NewReader(headerBytes))
	dec.UseNumber()
	if err := dec.Decode(&top); err != nil {
		return nil, fmt.Errorf("safetensors invalid JSON header: %w", err)
	}
	if top == nil {
		return nil, fmt.Errorf("safetensors JSON header must be an object")
	}
	// Reject any trailing tokens or non-JSON garbage after the top-level JSON object.
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("safetensors header contains trailing JSON data")
		}
		return nil, fmt.Errorf("safetensors header contains trailing non-JSON data: %w", err)
	}

	dataBase := int64(8 + headerLen)
	dataLen := int64(len(data)) - dataBase

	sf := &SafeTensorsFile{
		Metadata: make(map[string]string),
		Tensors:  make(map[string]SafeTensorInfo, len(top)),
		dataBase: dataBase,
		raw:      data,
	}

	type spanEntry struct {
		name  string
		begin int64
		end   int64
	}
	var spans []spanEntry

	keys := make([]string, 0, len(top))
	for k := range top {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, name := range keys {
		rawVal := top[name]
		if name == "__metadata__" {
			trimmed := bytes.TrimSpace(rawVal)
			if bytes.Equal(trimmed, []byte("null")) {
				continue
			}
			var meta map[string]string
			if err := json.Unmarshal(trimmed, &meta); err != nil {
				// Accept metadata maps with arbitrary JSON values by stringifying fallback.
				var generic map[string]any
				if err2 := json.Unmarshal(trimmed, &generic); err2 != nil {
					return nil, fmt.Errorf("safetensors __metadata__ must be a JSON object: %w", err)
				}
				for mk, mv := range generic {
					sf.Metadata[mk] = fmt.Sprint(mv)
				}
				continue
			}
			for mk, mv := range meta {
				sf.Metadata[mk] = mv
			}
			continue
		}

		var entry rawSafeTensorEntry
		if err := json.Unmarshal(rawVal, &entry); err != nil {
			return nil, fmt.Errorf("safetensors tensor %q: invalid descriptor: %w", name, err)
		}
		if entry.DType == "" {
			return nil, fmt.Errorf("safetensors tensor %q: missing dtype", name)
		}
		if entry.Shape == nil {
			return nil, fmt.Errorf("safetensors tensor %q: missing shape array", name)
		}
		elemBytes, ok := SafeTensorDTypeBytes(entry.DType)
		if !ok {
			return nil, fmt.Errorf("safetensors tensor %q: unsupported dtype %q", name, entry.DType)
		}
		if len(entry.DataOffsets) != 2 {
			return nil, fmt.Errorf("safetensors tensor %q: data_offsets must have 2 elements, got %d", name, len(entry.DataOffsets))
		}
		begin, end := entry.DataOffsets[0], entry.DataOffsets[1]
		if begin < 0 || end < 0 {
			return nil, fmt.Errorf("safetensors tensor %q: negative data_offsets [%d, %d]", name, begin, end)
		}
		if begin > end {
			return nil, fmt.Errorf("safetensors tensor %q: data_offsets begin %d > end %d", name, begin, end)
		}
		if end > dataLen {
			return nil, fmt.Errorf("safetensors tensor %q: data_offsets end %d exceeds data section size %d", name, end, dataLen)
		}

		shape := *entry.Shape
		numElems, err := shapeProductChecked(shape)
		if err != nil {
			return nil, fmt.Errorf("safetensors tensor %q: %w", name, err)
		}
		expectedBytes, err := mulInt64Checked(numElems, elemBytes)
		if err != nil {
			return nil, fmt.Errorf("safetensors tensor %q: byte size overflow: %w", name, err)
		}
		actualBytes := end - begin
		if actualBytes != expectedBytes {
			return nil, fmt.Errorf("safetensors tensor %q: byte span %d != shape %v * %s (%d)",
				name, actualBytes, shape, entry.DType, expectedBytes)
		}

		shapeCopy := make([]int64, len(shape))
		copy(shapeCopy, shape)
		info := SafeTensorInfo{
			Name:        name,
			DType:       entry.DType,
			Shape:       shapeCopy,
			DataOffsets: [2]int64{begin, end},
			ByteLen:     actualBytes,
		}
		sf.Tensors[name] = info
		sf.Names = append(sf.Names, name)
		if actualBytes > 0 {
			spans = append(spans, spanEntry{name: name, begin: begin, end: end})
		}
	}

	if opts.Strict && len(spans) > 1 {
		sort.Slice(spans, func(i, j int) bool {
			if spans[i].begin != spans[j].begin {
				return spans[i].begin < spans[j].begin
			}
			return spans[i].end < spans[j].end
		})
		for i := 1; i < len(spans); i++ {
			if spans[i-1].end > spans[i].begin {
				return nil, fmt.Errorf("safetensors strict mode: overlapping spans between %q [%d,%d) and %q [%d,%d)",
					spans[i-1].name, spans[i-1].begin, spans[i-1].end,
					spans[i].name, spans[i].begin, spans[i].end)
			}
		}
	}

	return sf, nil
}

// RawBytes returns the raw little-endian payload slice for tensor name.
func (sf *SafeTensorsFile) RawBytes(name string) ([]byte, error) {
	info, ok := sf.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	start := sf.dataBase + info.DataOffsets[0]
	end := sf.dataBase + info.DataOffsets[1]
	return sf.raw[start:end], nil
}

// bf16ToFloat32 expands a bfloat16 bit pattern to IEEE-754 float32 exactly.
func bf16ToFloat32(b uint16) float32 {
	return math.Float32frombits(uint32(b) << 16)
}

// ToFP16 converts a float tensor ("F16", "F32", or "BF16") to finite IEEE-754
// binary16 values using internal/fp16 round-to-nearest-even. Non-finite inputs
// (NaN/Inf) or values that overflow fp16 finite range are rejected.
func (sf *SafeTensorsFile) ToFP16(name string) ([]uint16, error) {
	info, ok := sf.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	raw, err := sf.RawBytes(name)
	if err != nil {
		return nil, err
	}

	switch info.DType {
	case "F16":
		n := len(raw) / 2
		out := make([]uint16, n)
		for i := 0; i < n; i++ {
			u := binary.LittleEndian.Uint16(raw[2*i:])
			if !fp16.IsFinite(u) {
				return nil, fmt.Errorf("safetensors tensor %q: non-finite F16 element at index %d (0x%04x)", name, i, u)
			}
			out[i] = u
		}
		return out, nil

	case "F32":
		n := len(raw) / 4
		out := make([]uint16, n)
		for i := 0; i < n; i++ {
			f := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
			if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
				return nil, fmt.Errorf("safetensors tensor %q: non-finite F32 element at index %d (%v)", name, i, f)
			}
			h := fp16.FromFloat32(f)
			if !fp16.IsFinite(h) {
				return nil, fmt.Errorf("safetensors tensor %q: F32 element at index %d (%g) overflows fp16", name, i, f)
			}
			out[i] = h
		}
		return out, nil

	case "BF16":
		n := len(raw) / 2
		out := make([]uint16, n)
		for i := 0; i < n; i++ {
			b := binary.LittleEndian.Uint16(raw[2*i:])
			f := bf16ToFloat32(b)
			if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
				return nil, fmt.Errorf("safetensors tensor %q: non-finite BF16 element at index %d (0x%04x)", name, i, b)
			}
			h := fp16.FromFloat32(f)
			if !fp16.IsFinite(h) {
				return nil, fmt.Errorf("safetensors tensor %q: BF16 element at index %d (%g) overflows fp16", name, i, f)
			}
			out[i] = h
		}
		return out, nil

	default:
		return nil, fmt.Errorf("safetensors tensor %q: dtype %q cannot be exported as fp16 (supported: F16, F32, BF16)", name, info.DType)
	}
}

// ToFP16Bytes converts a float tensor ("F16", "F32", or "BF16") to little-endian
// IEEE-754 binary16 bytes, rejecting NaN/Inf and fp16 overflow.
func (sf *SafeTensorsFile) ToFP16Bytes(name string) ([]byte, error) {
	vals, err := sf.ToFP16(name)
	if err != nil {
		return nil, err
	}
	return fp16.EncodeLE(vals), nil
}

// ToFloat64 converts a float tensor ("F16", "F32", "BF16", or "F64") to finite
// float64 values, rejecting NaN/Inf.
func (sf *SafeTensorsFile) ToFloat64(name string) ([]float64, error) {
	info, ok := sf.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	raw, err := sf.RawBytes(name)
	if err != nil {
		return nil, err
	}

	switch info.DType {
	case "F16":
		n := len(raw) / 2
		out := make([]float64, n)
		for i := 0; i < n; i++ {
			u := binary.LittleEndian.Uint16(raw[2*i:])
			if !fp16.IsFinite(u) {
				return nil, fmt.Errorf("safetensors tensor %q: non-finite F16 element at index %d (0x%04x)", name, i, u)
			}
			out[i] = fp16.ToFloat64(u)
		}
		return out, nil
	case "F32":
		n := len(raw) / 4
		out := make([]float64, n)
		for i := 0; i < n; i++ {
			f := float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])))
			if math.IsNaN(f) || math.IsInf(f, 0) {
				return nil, fmt.Errorf("safetensors tensor %q: non-finite F32 element at index %d (%v)", name, i, f)
			}
			out[i] = f
		}
		return out, nil
	case "BF16":
		n := len(raw) / 2
		out := make([]float64, n)
		for i := 0; i < n; i++ {
			b := binary.LittleEndian.Uint16(raw[2*i:])
			f := float64(bf16ToFloat32(b))
			if math.IsNaN(f) || math.IsInf(f, 0) {
				return nil, fmt.Errorf("safetensors tensor %q: non-finite BF16 element at index %d (0x%04x)", name, i, b)
			}
			out[i] = f
		}
		return out, nil
	case "F64":
		n := len(raw) / 8
		out := make([]float64, n)
		for i := 0; i < n; i++ {
			f := math.Float64frombits(binary.LittleEndian.Uint64(raw[8*i:]))
			if math.IsNaN(f) || math.IsInf(f, 0) {
				return nil, fmt.Errorf("safetensors tensor %q: non-finite F64 element at index %d (%v)", name, i, f)
			}
			out[i] = f
		}
		return out, nil
	default:
		return nil, fmt.Errorf("safetensors tensor %q: dtype %q cannot be decoded as float64", name, info.DType)
	}
}

// ToInt64 decodes an integer tensor ("I64" or "I32") into int64 values.
func (sf *SafeTensorsFile) ToInt64(name string) ([]int64, error) {
	info, ok := sf.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	raw, err := sf.RawBytes(name)
	if err != nil {
		return nil, err
	}
	switch info.DType {
	case "I64":
		n := len(raw) / 8
		out := make([]int64, n)
		for i := 0; i < n; i++ {
			out[i] = int64(binary.LittleEndian.Uint64(raw[8*i:]))
		}
		return out, nil
	case "I32":
		n := len(raw) / 4
		out := make([]int64, n)
		for i := 0; i < n; i++ {
			out[i] = int64(int32(binary.LittleEndian.Uint32(raw[4*i:])))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("safetensors tensor %q: dtype %q cannot be decoded as int64", name, info.DType)
	}
}

func sanitizeFilenameBase(name string) string {
	var b strings.Builder
	b.Grow(len(name))
	for i := 0; i < len(name); i++ {
		c := name[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.' {
			b.WriteByte(c)
		} else {
			b.WriteByte('_')
		}
	}
	s := strings.Trim(b.String(), "._")
	if s == "" {
		return "tensor"
	}
	return s
}

// ExportFP16Dir exports every F16, F32, and BF16 tensor in the SafeTensors archive
// to little-endian fp16 .bin files inside outDir. Non-float metadata tensors (such
// as I64/I32/BOOL) are skipped.
func (sf *SafeTensorsFile) ExportFP16Dir(outDir string) ([]ExportedTensor, error) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return nil, fmt.Errorf("create export dir %q: %w", outDir, err)
	}
	usedFiles := make(map[string]bool, len(sf.Names))
	exported := make([]ExportedTensor, 0, len(sf.Names))

	for _, name := range sf.Names {
		info := sf.Tensors[name]
		switch info.DType {
		case "F16", "F32", "BF16":
		default:
			continue
		}
		buf, err := sf.ToFP16Bytes(name)
		if err != nil {
			return nil, err
		}
		base := sanitizeFilenameBase(name)
		fname := base + ".bin"
		for suffix := 2; usedFiles[fname]; suffix++ {
			fname = fmt.Sprintf("%s_%d.bin", base, suffix)
		}
		usedFiles[fname] = true

		if err := os.WriteFile(filepath.Join(outDir, fname), buf, 0o644); err != nil {
			return nil, fmt.Errorf("write %s: %w", fname, err)
		}
		exported = append(exported, ExportedTensor{
			Name:     name,
			DType:    info.DType,
			Shape:    info.Shape,
			File:     fname,
			OutBytes: len(buf),
		})
	}
	return exported, nil
}

// WriteSafeTensors encodes a deterministic .safetensors archive from tensor
// descriptors and optional __metadata__.
func WriteSafeTensors(tensors []SafeTensorWrite, metadata map[string]string) ([]byte, error) {
	sorted := make([]SafeTensorWrite, len(tensors))
	copy(sorted, tensors)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Name < sorted[j].Name
	})

	header := make(map[string]any, len(sorted)+1)
	if len(metadata) > 0 {
		header["__metadata__"] = metadata
	}

	var dataBuf bytes.Buffer
	seen := make(map[string]bool, len(sorted))
	for _, t := range sorted {
		if t.Name == "" {
			return nil, fmt.Errorf("safetensors write: empty tensor name")
		}
		if t.Name == "__metadata__" {
			return nil, fmt.Errorf("safetensors write: tensor name cannot be __metadata__")
		}
		if seen[t.Name] {
			return nil, fmt.Errorf("safetensors write: duplicate tensor name %q", t.Name)
		}
		seen[t.Name] = true

		elemBytes, ok := SafeTensorDTypeBytes(t.DType)
		if !ok {
			return nil, fmt.Errorf("safetensors write: unsupported dtype %q for %q", t.DType, t.Name)
		}
		numElems, err := shapeProductChecked(t.Shape)
		if err != nil {
			return nil, fmt.Errorf("safetensors write %q: %w", t.Name, err)
		}
		wantBytes, err := mulInt64Checked(numElems, elemBytes)
		if err != nil {
			return nil, fmt.Errorf("safetensors write %q: %w", t.Name, err)
		}
		if int64(len(t.Data)) != wantBytes {
			return nil, fmt.Errorf("safetensors write %q: data length %d != expected %d", t.Name, len(t.Data), wantBytes)
		}

		begin := int64(dataBuf.Len())
		dataBuf.Write(t.Data)
		end := int64(dataBuf.Len())

		shape := t.Shape
		if shape == nil {
			shape = []int64{}
		}
		header[t.Name] = map[string]any{
			"dtype":        t.DType,
			"shape":        shape,
			"data_offsets": []int64{begin, end},
		}
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return nil, fmt.Errorf("safetensors write: marshal header: %w", err)
	}
	if uint64(len(headerJSON)) > MaxSafeTensorsHeaderBytes {
		return nil, fmt.Errorf("safetensors write: header size %d exceeds limit %d", len(headerJSON), MaxSafeTensorsHeaderBytes)
	}

	out := make([]byte, 8+len(headerJSON)+dataBuf.Len())
	binary.LittleEndian.PutUint64(out[0:8], uint64(len(headerJSON)))
	copy(out[8:], headerJSON)
	copy(out[8+len(headerJSON):], dataBuf.Bytes())
	return out, nil
}
