package frontend

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/aurora-silicon/vane/internal/fp16"
)

// ONNX TensorProto.DataType enum values per onnx.proto3.
const (
	ONNXUndefined int32 = 0
	ONNXFloat     int32 = 1
	ONNXUint8     int32 = 2
	ONNXInt8      int32 = 3
	ONNXUint16    int32 = 4
	ONNXInt16     int32 = 5
	ONNXInt32     int32 = 6
	ONNXInt64     int32 = 7
	ONNXString    int32 = 8
	ONNXBool      int32 = 9
	ONNXFloat16   int32 = 10
	ONNXDouble    int32 = 11
	ONNXUint32    int32 = 12
	ONNXUint64    int32 = 13
	ONNXBFloat16  int32 = 16
)

// ONNX AttributeProto.AttributeType enum values per onnx.proto3.
const (
	AttrUndefined int32 = 0
	AttrFloat     int32 = 1
	AttrInt       int32 = 2
	AttrString    int32 = 3
	AttrTensor    int32 = 4
	AttrGraph     int32 = 5
	AttrFloats    int32 = 6
	AttrInts      int32 = 7
	AttrStrings   int32 = 8
)

// ONNX TensorProto.DataLocation enum values per onnx.proto3.
const (
	ONNXDataDefault  int32 = 0
	ONNXDataExternal int32 = 1
)

// ONNXModel represents a decoded ONNX ModelProto.
type ONNXModel struct {
	IRVersion    int64
	ProducerName string
	OpsetImports []ONNXOpsetImport
	Graph        ONNXGraph
	HasGraph     bool
}

// DefaultOpsetVersion returns the opset version for the default ONNX domain ("" or "ai.onnx"),
// or 0 if not specified.
func (m *ONNXModel) DefaultOpsetVersion() int64 {
	for _, op := range m.OpsetImports {
		if op.Domain == "" || op.Domain == "ai.onnx" {
			return op.Version
		}
	}
	return 0
}

// ONNXOpsetImport represents an OperatorSetIdProto.
type ONNXOpsetImport struct {
	Domain  string
	Version int64
}

// ONNXGraph represents a decoded ONNX GraphProto.
type ONNXGraph struct {
	Name         string
	Nodes        []ONNXNode
	Initializers []ONNXTensor
	Inputs       []ONNXValueInfo
	Outputs      []ONNXValueInfo
}

// ONNXNode represents a decoded ONNX NodeProto.
type ONNXNode struct {
	Inputs     []string
	Outputs    []string
	Name       string
	OpType     string
	Domain     string
	Attributes []ONNXAttribute
}

// Attr returns the first attribute matching name.
func (n *ONNXNode) Attr(name string) (*ONNXAttribute, bool) {
	for i := range n.Attributes {
		if n.Attributes[i].Name == name {
			return &n.Attributes[i], true
		}
	}
	return nil, false
}

// AttrInt returns an integer attribute value or def if absent.
func (n *ONNXNode) AttrInt(name string, def int64) int64 {
	if a, ok := n.Attr(name); ok {
		return a.I
	}
	return def
}

// AttrFloat returns a float attribute value as float64, or def if absent.
func (n *ONNXNode) AttrFloat(name string, def float64) float64 {
	if a, ok := n.Attr(name); ok {
		return float64(a.F)
	}
	return def
}

// AttrInts returns a repeated int64 attribute value if present.
func (n *ONNXNode) AttrInts(name string) ([]int64, bool) {
	if a, ok := n.Attr(name); ok {
		return a.Ints, true
	}
	return nil, false
}

// AttrFloats returns a repeated float attribute value if present.
func (n *ONNXNode) AttrFloats(name string) ([]float32, bool) {
	if a, ok := n.Attr(name); ok {
		return a.Floats, true
	}
	return nil, false
}

// AttrString returns a string attribute value or def if absent.
func (n *ONNXNode) AttrString(name string, def string) string {
	if a, ok := n.Attr(name); ok {
		return string(a.S)
	}
	return def
}

// AttrTensor returns a TensorProto attribute if present.
func (n *ONNXNode) AttrTensor(name string) (*ONNXTensor, bool) {
	if a, ok := n.Attr(name); ok && a.T != nil {
		return a.T, true
	}
	return nil, false
}

// ONNXAttribute represents a decoded ONNX AttributeProto.
type ONNXAttribute struct {
	Name    string
	Type    int32
	F       float32
	I       int64
	S       []byte
	T       *ONNXTensor
	Floats  []float32
	Ints    []int64
	Strings [][]byte
}

// ONNXTensor represents a decoded ONNX TensorProto.
type ONNXTensor struct {
	Name         string
	Dims         []int64
	DataType     int32
	FloatData    []float32
	Int32Data    []int32
	Int64Data    []int64
	DoubleData   []float64
	RawData      []byte
	DataLocation int32
}

// NumElements returns the total element count of t.Dims with overflow checking.
// A scalar tensor (len(Dims) == 0) has 1 element.
func (t *ONNXTensor) NumElements() (int64, error) {
	return shapeProductChecked(t.Dims)
}

// ToFloat64 extracts finite float64 values from a FLOAT, DOUBLE, FLOAT16, or BFLOAT16 TensorProto.
func (t *ONNXTensor) ToFloat64() ([]float64, error) {
	if t.DataLocation == ONNXDataExternal {
		return nil, fmt.Errorf("tensor %q uses external data_location=EXTERNAL (provide --safetensors override or embedded weights)", t.Name)
	}
	nElems, err := t.NumElements()
	if err != nil {
		return nil, fmt.Errorf("tensor %q: %w", t.Name, err)
	}

	switch t.DataType {
	case ONNXFloat:
		if len(t.RawData) > 0 {
			want, err := mulInt64Checked(nElems, 4)
			if err != nil || int64(len(t.RawData)) != want {
				return nil, fmt.Errorf("tensor %q (FLOAT): raw_data length %d != expected %d", t.Name, len(t.RawData), want)
			}
			out := make([]float64, nElems)
			for i := int64(0); i < nElems; i++ {
				f := float64(math.Float32frombits(binary.LittleEndian.Uint32(t.RawData[4*i:])))
				if math.IsNaN(f) || math.IsInf(f, 0) {
					return nil, fmt.Errorf("tensor %q (FLOAT): non-finite value at index %d", t.Name, i)
				}
				out[i] = f
			}
			return out, nil
		}
		if int64(len(t.FloatData)) != nElems {
			return nil, fmt.Errorf("tensor %q (FLOAT): float_data length %d != expected %d", t.Name, len(t.FloatData), nElems)
		}
		out := make([]float64, nElems)
		for i, v := range t.FloatData {
			f := float64(v)
			if math.IsNaN(f) || math.IsInf(f, 0) {
				return nil, fmt.Errorf("tensor %q (FLOAT): non-finite value at index %d", t.Name, i)
			}
			out[i] = f
		}
		return out, nil

	case ONNXDouble:
		if len(t.RawData) > 0 {
			want, err := mulInt64Checked(nElems, 8)
			if err != nil || int64(len(t.RawData)) != want {
				return nil, fmt.Errorf("tensor %q (DOUBLE): raw_data length %d != expected %d", t.Name, len(t.RawData), want)
			}
			out := make([]float64, nElems)
			for i := int64(0); i < nElems; i++ {
				f := math.Float64frombits(binary.LittleEndian.Uint64(t.RawData[8*i:]))
				if math.IsNaN(f) || math.IsInf(f, 0) {
					return nil, fmt.Errorf("tensor %q (DOUBLE): non-finite value at index %d", t.Name, i)
				}
				out[i] = f
			}
			return out, nil
		}
		if int64(len(t.DoubleData)) != nElems {
			return nil, fmt.Errorf("tensor %q (DOUBLE): double_data length %d != expected %d", t.Name, len(t.DoubleData), nElems)
		}
		out := make([]float64, nElems)
		for i, f := range t.DoubleData {
			if math.IsNaN(f) || math.IsInf(f, 0) {
				return nil, fmt.Errorf("tensor %q (DOUBLE): non-finite value at index %d", t.Name, i)
			}
			out[i] = f
		}
		return out, nil

	case ONNXFloat16:
		if len(t.RawData) > 0 {
			want, err := mulInt64Checked(nElems, 2)
			if err != nil || int64(len(t.RawData)) != want {
				return nil, fmt.Errorf("tensor %q (FLOAT16): raw_data length %d != expected %d", t.Name, len(t.RawData), want)
			}
			out := make([]float64, nElems)
			for i := int64(0); i < nElems; i++ {
				u := binary.LittleEndian.Uint16(t.RawData[2*i:])
				if !fp16.IsFinite(u) {
					return nil, fmt.Errorf("tensor %q (FLOAT16): non-finite value at index %d (0x%04x)", t.Name, i, u)
				}
				out[i] = fp16.ToFloat64(u)
			}
			return out, nil
		}
		// Per ONNX spec, FLOAT16 may be stored in the low 16 bits of int32_data.
		if int64(len(t.Int32Data)) != nElems {
			return nil, fmt.Errorf("tensor %q (FLOAT16): int32_data length %d != expected %d", t.Name, len(t.Int32Data), nElems)
		}
		out := make([]float64, nElems)
		for i, v := range t.Int32Data {
			u := uint16(v & 0xffff)
			if !fp16.IsFinite(u) {
				return nil, fmt.Errorf("tensor %q (FLOAT16): non-finite value at index %d (0x%04x)", t.Name, i, u)
			}
			out[i] = fp16.ToFloat64(u)
		}
		return out, nil

	case ONNXBFloat16:
		if len(t.RawData) > 0 {
			want, err := mulInt64Checked(nElems, 2)
			if err != nil || int64(len(t.RawData)) != want {
				return nil, fmt.Errorf("tensor %q (BFLOAT16): raw_data length %d != expected %d", t.Name, len(t.RawData), want)
			}
			out := make([]float64, nElems)
			for i := int64(0); i < nElems; i++ {
				b := binary.LittleEndian.Uint16(t.RawData[2*i:])
				f := float64(bf16ToFloat32(b))
				if math.IsNaN(f) || math.IsInf(f, 0) {
					return nil, fmt.Errorf("tensor %q (BFLOAT16): non-finite value at index %d (0x%04x)", t.Name, i, b)
				}
				out[i] = f
			}
			return out, nil
		}
		if int64(len(t.Int32Data)) != nElems {
			return nil, fmt.Errorf("tensor %q (BFLOAT16): int32_data length %d != expected %d", t.Name, len(t.Int32Data), nElems)
		}
		out := make([]float64, nElems)
		for i, v := range t.Int32Data {
			b := uint16(v & 0xffff)
			f := float64(bf16ToFloat32(b))
			if math.IsNaN(f) || math.IsInf(f, 0) {
				return nil, fmt.Errorf("tensor %q (BFLOAT16): non-finite value at index %d (0x%04x)", t.Name, i, b)
			}
			out[i] = f
		}
		return out, nil

	default:
		return nil, fmt.Errorf("tensor %q: unsupported float data_type %d", t.Name, t.DataType)
	}
}

// ToInt64 extracts int64 values from an INT64 or INT32 TensorProto.
func (t *ONNXTensor) ToInt64() ([]int64, error) {
	if t.DataLocation == ONNXDataExternal {
		return nil, fmt.Errorf("tensor %q uses external data_location=EXTERNAL", t.Name)
	}
	nElems, err := t.NumElements()
	if err != nil {
		return nil, fmt.Errorf("tensor %q: %w", t.Name, err)
	}

	switch t.DataType {
	case ONNXInt64:
		if len(t.RawData) > 0 {
			want, err := mulInt64Checked(nElems, 8)
			if err != nil || int64(len(t.RawData)) != want {
				return nil, fmt.Errorf("tensor %q (INT64): raw_data length %d != expected %d", t.Name, len(t.RawData), want)
			}
			out := make([]int64, nElems)
			for i := int64(0); i < nElems; i++ {
				out[i] = int64(binary.LittleEndian.Uint64(t.RawData[8*i:]))
			}
			return out, nil
		}
		if int64(len(t.Int64Data)) != nElems {
			return nil, fmt.Errorf("tensor %q (INT64): int64_data length %d != expected %d", t.Name, len(t.Int64Data), nElems)
		}
		out := make([]int64, nElems)
		copy(out, t.Int64Data)
		return out, nil

	case ONNXInt32:
		if len(t.RawData) > 0 {
			want, err := mulInt64Checked(nElems, 4)
			if err != nil || int64(len(t.RawData)) != want {
				return nil, fmt.Errorf("tensor %q (INT32): raw_data length %d != expected %d", t.Name, len(t.RawData), want)
			}
			out := make([]int64, nElems)
			for i := int64(0); i < nElems; i++ {
				out[i] = int64(int32(binary.LittleEndian.Uint32(t.RawData[4*i:])))
			}
			return out, nil
		}
		if int64(len(t.Int32Data)) != nElems {
			return nil, fmt.Errorf("tensor %q (INT32): int32_data length %d != expected %d", t.Name, len(t.Int32Data), nElems)
		}
		out := make([]int64, nElems)
		for i, v := range t.Int32Data {
			out[i] = int64(v)
		}
		return out, nil

	default:
		return nil, fmt.Errorf("tensor %q: unsupported integer data_type %d", t.Name, t.DataType)
	}
}

// ToFP16 converts a float TensorProto to finite IEEE-754 binary16 values.
func (t *ONNXTensor) ToFP16() ([]uint16, error) {
	vals, err := t.ToFloat64()
	if err != nil {
		return nil, err
	}
	return float64SliceToFP16(vals, t.Name)
}

func float64SliceToFP16(vals []float64, name string) ([]uint16, error) {
	out := make([]uint16, len(vals))
	for i, f := range vals {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, fmt.Errorf("tensor %q: non-finite value at index %d (%v)", name, i, f)
		}
		h := fp16.FromFloat64(f)
		if !fp16.IsFinite(h) {
			return nil, fmt.Errorf("tensor %q: value %g at index %d overflows fp16", name, f, i)
		}
		out[i] = h
	}
	return out, nil
}

// ONNXValueInfo represents a decoded ONNX ValueInfoProto.
type ONNXValueInfo struct {
	Name     string
	ElemType int32
	HasShape bool
	Dims     []ONNXDim
}

// ONNXDim represents a single TensorShapeProto.Dimension.
type ONNXDim struct {
	HasValue bool
	Value    int64
	Param    string
}

// LoadONNX reads and decodes a binary .onnx file from disk.
func LoadONNX(path string) (*ONNXModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read onnx %q: %w", path, err)
	}
	return ParseONNX(data)
}

// ParseONNX decodes a binary ONNX ModelProto message.
func ParseONNX(data []byte) (*ONNXModel, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("onnx buffer is empty")
	}
	r := NewProtoReader(data)
	return decodeModelProto(r)
}

func decodeModelProto(r *ProtoReader) (*ONNXModel, error) {
	m := &ONNXModel{}
	for !r.Done() {
		fn, wt, err := r.NextTag()
		if err != nil {
			return nil, fmt.Errorf("ModelProto: %w", err)
		}
		switch fn {
		case 1: // ir_version (int64)
			if wt != WireVarint {
				return nil, fmt.Errorf("ModelProto.ir_version: unexpected wire type %d", wt)
			}
			v, err := r.ReadVarint()
			if err != nil {
				return nil, err
			}
			m.IRVersion = int64(v)
		case 2: // producer_name (string)
			if wt != WireBytes {
				return nil, fmt.Errorf("ModelProto.producer_name: unexpected wire type %d", wt)
			}
			s, err := r.ReadString()
			if err != nil {
				return nil, err
			}
			m.ProducerName = s
		case 7: // graph (GraphProto)
			if wt != WireBytes {
				return nil, fmt.Errorf("ModelProto.graph: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return nil, err
			}
			sub, err := r.SubReader(b)
			if err != nil {
				return nil, err
			}
			g, err := decodeGraphProto(sub)
			if err != nil {
				return nil, err
			}
			m.Graph = g
			m.HasGraph = true
		case 8: // opset_import (repeated OperatorSetIdProto)
			if wt != WireBytes {
				return nil, fmt.Errorf("ModelProto.opset_import: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return nil, err
			}
			sub, err := r.SubReader(b)
			if err != nil {
				return nil, err
			}
			op, err := decodeOpsetImport(sub)
			if err != nil {
				return nil, err
			}
			m.OpsetImports = append(m.OpsetImports, op)
		default:
			if err := r.Skip(wt); err != nil {
				return nil, fmt.Errorf("ModelProto skip field %d: %w", fn, err)
			}
		}
	}
	if !m.HasGraph {
		return nil, fmt.Errorf("ModelProto: missing graph field")
	}
	return m, nil
}

func decodeOpsetImport(r *ProtoReader) (ONNXOpsetImport, error) {
	var op ONNXOpsetImport
	for !r.Done() {
		fn, wt, err := r.NextTag()
		if err != nil {
			return op, fmt.Errorf("OperatorSetIdProto: %w", err)
		}
		switch fn {
		case 1: // domain
			if wt != WireBytes {
				return op, fmt.Errorf("OperatorSetIdProto.domain: unexpected wire type %d", wt)
			}
			s, err := r.ReadString()
			if err != nil {
				return op, err
			}
			op.Domain = s
		case 2: // version
			if wt != WireVarint {
				return op, fmt.Errorf("OperatorSetIdProto.version: unexpected wire type %d", wt)
			}
			v, err := r.ReadVarint()
			if err != nil {
				return op, err
			}
			op.Version = int64(v)
		default:
			if err := r.Skip(wt); err != nil {
				return op, err
			}
		}
	}
	return op, nil
}

func decodeGraphProto(r *ProtoReader) (ONNXGraph, error) {
	var g ONNXGraph
	for !r.Done() {
		fn, wt, err := r.NextTag()
		if err != nil {
			return g, fmt.Errorf("GraphProto: %w", err)
		}
		switch fn {
		case 1: // node (repeated NodeProto)
			if wt != WireBytes {
				return g, fmt.Errorf("GraphProto.node: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return g, err
			}
			sub, err := r.SubReader(b)
			if err != nil {
				return g, err
			}
			nd, err := decodeNodeProto(sub)
			if err != nil {
				return g, err
			}
			g.Nodes = append(g.Nodes, nd)
		case 2: // name
			if wt != WireBytes {
				return g, fmt.Errorf("GraphProto.name: unexpected wire type %d", wt)
			}
			s, err := r.ReadString()
			if err != nil {
				return g, err
			}
			g.Name = s
		case 5: // initializer (repeated TensorProto)
			if wt != WireBytes {
				return g, fmt.Errorf("GraphProto.initializer: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return g, err
			}
			sub, err := r.SubReader(b)
			if err != nil {
				return g, err
			}
			t, err := decodeTensorProto(sub)
			if err != nil {
				return g, err
			}
			g.Initializers = append(g.Initializers, t)
		case 11: // input (repeated ValueInfoProto)
			if wt != WireBytes {
				return g, fmt.Errorf("GraphProto.input: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return g, err
			}
			sub, err := r.SubReader(b)
			if err != nil {
				return g, err
			}
			vi, err := decodeValueInfoProto(sub)
			if err != nil {
				return g, err
			}
			g.Inputs = append(g.Inputs, vi)
		case 12: // output (repeated ValueInfoProto)
			if wt != WireBytes {
				return g, fmt.Errorf("GraphProto.output: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return g, err
			}
			sub, err := r.SubReader(b)
			if err != nil {
				return g, err
			}
			vi, err := decodeValueInfoProto(sub)
			if err != nil {
				return g, err
			}
			g.Outputs = append(g.Outputs, vi)
		default:
			if err := r.Skip(wt); err != nil {
				return g, fmt.Errorf("GraphProto skip field %d: %w", fn, err)
			}
		}
	}
	return g, nil
}

func decodeNodeProto(r *ProtoReader) (ONNXNode, error) {
	var nd ONNXNode
	for !r.Done() {
		fn, wt, err := r.NextTag()
		if err != nil {
			return nd, fmt.Errorf("NodeProto: %w", err)
		}
		switch fn {
		case 1: // input
			if wt != WireBytes {
				return nd, fmt.Errorf("NodeProto.input: unexpected wire type %d", wt)
			}
			s, err := r.ReadString()
			if err != nil {
				return nd, err
			}
			nd.Inputs = append(nd.Inputs, s)
		case 2: // output
			if wt != WireBytes {
				return nd, fmt.Errorf("NodeProto.output: unexpected wire type %d", wt)
			}
			s, err := r.ReadString()
			if err != nil {
				return nd, err
			}
			nd.Outputs = append(nd.Outputs, s)
		case 3: // name
			if wt != WireBytes {
				return nd, fmt.Errorf("NodeProto.name: unexpected wire type %d", wt)
			}
			s, err := r.ReadString()
			if err != nil {
				return nd, err
			}
			nd.Name = s
		case 4: // op_type
			if wt != WireBytes {
				return nd, fmt.Errorf("NodeProto.op_type: unexpected wire type %d", wt)
			}
			s, err := r.ReadString()
			if err != nil {
				return nd, err
			}
			nd.OpType = s
		case 5: // attribute
			if wt != WireBytes {
				return nd, fmt.Errorf("NodeProto.attribute: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return nd, err
			}
			sub, err := r.SubReader(b)
			if err != nil {
				return nd, err
			}
			attr, err := decodeAttributeProto(sub)
			if err != nil {
				return nd, err
			}
			nd.Attributes = append(nd.Attributes, attr)
		case 7: // domain
			if wt != WireBytes {
				return nd, fmt.Errorf("NodeProto.domain: unexpected wire type %d", wt)
			}
			s, err := r.ReadString()
			if err != nil {
				return nd, err
			}
			nd.Domain = s
		default:
			if err := r.Skip(wt); err != nil {
				return nd, fmt.Errorf("NodeProto skip field %d: %w", fn, err)
			}
		}
	}
	return nd, nil
}

func decodeAttributeProto(r *ProtoReader) (ONNXAttribute, error) {
	var a ONNXAttribute
	for !r.Done() {
		fn, wt, err := r.NextTag()
		if err != nil {
			return a, fmt.Errorf("AttributeProto: %w", err)
		}
		switch fn {
		case 1: // name
			if wt != WireBytes {
				return a, fmt.Errorf("AttributeProto.name: unexpected wire type %d", wt)
			}
			s, err := r.ReadString()
			if err != nil {
				return a, err
			}
			a.Name = s
		case 2: // f (float)
			if wt != WireFixed32 {
				return a, fmt.Errorf("AttributeProto.f: unexpected wire type %d", wt)
			}
			f, err := r.ReadFloat32()
			if err != nil {
				return a, err
			}
			a.F = f
		case 3: // i (int64)
			if wt != WireVarint {
				return a, fmt.Errorf("AttributeProto.i: unexpected wire type %d", wt)
			}
			v, err := r.ReadVarint()
			if err != nil {
				return a, err
			}
			a.I = int64(v)
		case 4: // s (bytes)
			if wt != WireBytes {
				return a, fmt.Errorf("AttributeProto.s: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return a, err
			}
			cp := make([]byte, len(b))
			copy(cp, b)
			a.S = cp
		case 5: // t (TensorProto)
			if wt != WireBytes {
				return a, fmt.Errorf("AttributeProto.t: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return a, err
			}
			sub, err := r.SubReader(b)
			if err != nil {
				return a, err
			}
			t, err := decodeTensorProto(sub)
			if err != nil {
				return a, err
			}
			a.T = &t
		case 7: // floats (packed or unpacked)
			switch wt {
			case WireFixed32:
				f, err := r.ReadFloat32()
				if err != nil {
					return a, err
				}
				a.Floats = append(a.Floats, f)
			case WireBytes:
				b, err := r.ReadBytes()
				if err != nil {
					return a, err
				}
				fs, err := ReadPackedFloat32(b)
				if err != nil {
					return a, err
				}
				a.Floats = append(a.Floats, fs...)
			default:
				return a, fmt.Errorf("AttributeProto.floats: unexpected wire type %d", wt)
			}
		case 8: // ints (packed or unpacked)
			switch wt {
			case WireVarint:
				v, err := r.ReadVarint()
				if err != nil {
					return a, err
				}
				a.Ints = append(a.Ints, int64(v))
			case WireBytes:
				b, err := r.ReadBytes()
				if err != nil {
					return a, err
				}
				vs, err := ReadPackedInt64(b)
				if err != nil {
					return a, err
				}
				a.Ints = append(a.Ints, vs...)
			default:
				return a, fmt.Errorf("AttributeProto.ints: unexpected wire type %d", wt)
			}
		case 9: // strings
			if wt != WireBytes {
				return a, fmt.Errorf("AttributeProto.strings: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return a, err
			}
			cp := make([]byte, len(b))
			copy(cp, b)
			a.Strings = append(a.Strings, cp)
		case 20: // type
			if wt != WireVarint {
				return a, fmt.Errorf("AttributeProto.type: unexpected wire type %d", wt)
			}
			v, err := r.ReadVarint()
			if err != nil {
				return a, err
			}
			a.Type = int32(v)
		default:
			if err := r.Skip(wt); err != nil {
				return a, fmt.Errorf("AttributeProto skip field %d: %w", fn, err)
			}
		}
	}
	return a, nil
}

func decodeTensorProto(r *ProtoReader) (ONNXTensor, error) {
	var t ONNXTensor
	for !r.Done() {
		fn, wt, err := r.NextTag()
		if err != nil {
			return t, fmt.Errorf("TensorProto: %w", err)
		}
		switch fn {
		case 1: // dims (repeated int64, packed or unpacked)
			switch wt {
			case WireVarint:
				v, err := r.ReadVarint()
				if err != nil {
					return t, err
				}
				t.Dims = append(t.Dims, int64(v))
			case WireBytes:
				b, err := r.ReadBytes()
				if err != nil {
					return t, err
				}
				vs, err := ReadPackedInt64(b)
				if err != nil {
					return t, err
				}
				t.Dims = append(t.Dims, vs...)
			default:
				return t, fmt.Errorf("TensorProto.dims: unexpected wire type %d", wt)
			}
		case 2: // data_type
			if wt != WireVarint {
				return t, fmt.Errorf("TensorProto.data_type: unexpected wire type %d", wt)
			}
			v, err := r.ReadVarint()
			if err != nil {
				return t, err
			}
			t.DataType = int32(v)
		case 4: // float_data (packed or unpacked)
			switch wt {
			case WireFixed32:
				f, err := r.ReadFloat32()
				if err != nil {
					return t, err
				}
				t.FloatData = append(t.FloatData, f)
			case WireBytes:
				b, err := r.ReadBytes()
				if err != nil {
					return t, err
				}
				fs, err := ReadPackedFloat32(b)
				if err != nil {
					return t, err
				}
				t.FloatData = append(t.FloatData, fs...)
			default:
				return t, fmt.Errorf("TensorProto.float_data: unexpected wire type %d", wt)
			}
		case 5: // int32_data (packed or unpacked)
			switch wt {
			case WireVarint:
				v, err := r.ReadVarint()
				if err != nil {
					return t, err
				}
				t.Int32Data = append(t.Int32Data, int32(v))
			case WireBytes:
				b, err := r.ReadBytes()
				if err != nil {
					return t, err
				}
				vs, err := ReadPackedInt32(b)
				if err != nil {
					return t, err
				}
				t.Int32Data = append(t.Int32Data, vs...)
			default:
				return t, fmt.Errorf("TensorProto.int32_data: unexpected wire type %d", wt)
			}
		case 7: // int64_data (packed or unpacked)
			switch wt {
			case WireVarint:
				v, err := r.ReadVarint()
				if err != nil {
					return t, err
				}
				t.Int64Data = append(t.Int64Data, int64(v))
			case WireBytes:
				b, err := r.ReadBytes()
				if err != nil {
					return t, err
				}
				vs, err := ReadPackedInt64(b)
				if err != nil {
					return t, err
				}
				t.Int64Data = append(t.Int64Data, vs...)
			default:
				return t, fmt.Errorf("TensorProto.int64_data: unexpected wire type %d", wt)
			}
		case 8: // name
			if wt != WireBytes {
				return t, fmt.Errorf("TensorProto.name: unexpected wire type %d", wt)
			}
			s, err := r.ReadString()
			if err != nil {
				return t, err
			}
			t.Name = s
		case 9: // raw_data
			if wt != WireBytes {
				return t, fmt.Errorf("TensorProto.raw_data: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return t, err
			}
			cp := make([]byte, len(b))
			copy(cp, b)
			t.RawData = cp
		case 10: // double_data (packed or unpacked)
			switch wt {
			case WireFixed64:
				f, err := r.ReadFloat64()
				if err != nil {
					return t, err
				}
				t.DoubleData = append(t.DoubleData, f)
			case WireBytes:
				b, err := r.ReadBytes()
				if err != nil {
					return t, err
				}
				fs, err := ReadPackedFloat64(b)
				if err != nil {
					return t, err
				}
				t.DoubleData = append(t.DoubleData, fs...)
			default:
				return t, fmt.Errorf("TensorProto.double_data: unexpected wire type %d", wt)
			}
		case 14: // data_location
			if wt != WireVarint {
				return t, fmt.Errorf("TensorProto.data_location: unexpected wire type %d", wt)
			}
			v, err := r.ReadVarint()
			if err != nil {
				return t, err
			}
			t.DataLocation = int32(v)
		default:
			if err := r.Skip(wt); err != nil {
				return t, fmt.Errorf("TensorProto skip field %d: %w", fn, err)
			}
		}
	}
	return t, nil
}

func decodeValueInfoProto(r *ProtoReader) (ONNXValueInfo, error) {
	var vi ONNXValueInfo
	for !r.Done() {
		fn, wt, err := r.NextTag()
		if err != nil {
			return vi, fmt.Errorf("ValueInfoProto: %w", err)
		}
		switch fn {
		case 1: // name
			if wt != WireBytes {
				return vi, fmt.Errorf("ValueInfoProto.name: unexpected wire type %d", wt)
			}
			s, err := r.ReadString()
			if err != nil {
				return vi, err
			}
			vi.Name = s
		case 2: // type (TypeProto)
			if wt != WireBytes {
				return vi, fmt.Errorf("ValueInfoProto.type: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return vi, err
			}
			sub, err := r.SubReader(b)
			if err != nil {
				return vi, err
			}
			if err := decodeTypeProto(sub, &vi); err != nil {
				return vi, err
			}
		default:
			if err := r.Skip(wt); err != nil {
				return vi, fmt.Errorf("ValueInfoProto skip field %d: %w", fn, err)
			}
		}
	}
	return vi, nil
}

func decodeTypeProto(r *ProtoReader, vi *ONNXValueInfo) error {
	for !r.Done() {
		fn, wt, err := r.NextTag()
		if err != nil {
			return fmt.Errorf("TypeProto: %w", err)
		}
		switch fn {
		case 1: // tensor_type (TypeProto.Tensor)
			if wt != WireBytes {
				return fmt.Errorf("TypeProto.tensor_type: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return err
			}
			sub, err := r.SubReader(b)
			if err != nil {
				return err
			}
			if err := decodeTypeTensorProto(sub, vi); err != nil {
				return err
			}
		default:
			if err := r.Skip(wt); err != nil {
				return err
			}
		}
	}
	return nil
}

func decodeTypeTensorProto(r *ProtoReader, vi *ONNXValueInfo) error {
	for !r.Done() {
		fn, wt, err := r.NextTag()
		if err != nil {
			return fmt.Errorf("TypeProto.Tensor: %w", err)
		}
		switch fn {
		case 1: // elem_type
			if wt != WireVarint {
				return fmt.Errorf("TypeProto.Tensor.elem_type: unexpected wire type %d", wt)
			}
			v, err := r.ReadVarint()
			if err != nil {
				return err
			}
			vi.ElemType = int32(v)
		case 2: // shape (TensorShapeProto)
			if wt != WireBytes {
				return fmt.Errorf("TypeProto.Tensor.shape: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return err
			}
			sub, err := r.SubReader(b)
			if err != nil {
				return err
			}
			dims, err := decodeTensorShapeProto(sub)
			if err != nil {
				return err
			}
			vi.HasShape = true
			vi.Dims = dims
		default:
			if err := r.Skip(wt); err != nil {
				return err
			}
		}
	}
	return nil
}

func decodeTensorShapeProto(r *ProtoReader) ([]ONNXDim, error) {
	var dims []ONNXDim
	for !r.Done() {
		fn, wt, err := r.NextTag()
		if err != nil {
			return nil, fmt.Errorf("TensorShapeProto: %w", err)
		}
		switch fn {
		case 1: // dim (repeated Dimension)
			if wt != WireBytes {
				return nil, fmt.Errorf("TensorShapeProto.dim: unexpected wire type %d", wt)
			}
			b, err := r.ReadBytes()
			if err != nil {
				return nil, err
			}
			sub, err := r.SubReader(b)
			if err != nil {
				return nil, err
			}
			d, err := decodeDimensionProto(sub)
			if err != nil {
				return nil, err
			}
			dims = append(dims, d)
		default:
			if err := r.Skip(wt); err != nil {
				return nil, err
			}
		}
	}
	return dims, nil
}

func decodeDimensionProto(r *ProtoReader) (ONNXDim, error) {
	var d ONNXDim
	for !r.Done() {
		fn, wt, err := r.NextTag()
		if err != nil {
			return d, fmt.Errorf("Dimension: %w", err)
		}
		switch fn {
		case 1: // dim_value (int64)
			if wt != WireVarint {
				return d, fmt.Errorf("Dimension.dim_value: unexpected wire type %d", wt)
			}
			v, err := r.ReadVarint()
			if err != nil {
				return d, err
			}
			d.HasValue = true
			d.Value = int64(v)
		case 2: // dim_param (string)
			if wt != WireBytes {
				return d, fmt.Errorf("Dimension.dim_param: unexpected wire type %d", wt)
			}
			s, err := r.ReadString()
			if err != nil {
				return d, err
			}
			d.Param = s
		default:
			if err := r.Skip(wt); err != nil {
				return d, err
			}
		}
	}
	return d, nil
}
