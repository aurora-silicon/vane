package frontend

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Protobuf wire types supported by proto3.
const (
	WireVarint  uint8 = 0
	WireFixed64 uint8 = 1
	WireBytes   uint8 = 2
	WireFixed32 uint8 = 5
)

// MaxProtoDepth caps nested message recursion depth to guard against
// maliciously nested protobuf payloads.
const MaxProtoDepth = 32

// ProtoReader is a minimal, zero-dependency protobuf wire-format decoder.
type ProtoReader struct {
	buf   []byte
	pos   int
	depth int
}

// NewProtoReader constructs a top-level ProtoReader over buf.
func NewProtoReader(buf []byte) *ProtoReader {
	return &ProtoReader{buf: buf, pos: 0, depth: 0}
}

// SubReader constructs a nested ProtoReader enforcing MaxProtoDepth.
func (r *ProtoReader) SubReader(sub []byte) (*ProtoReader, error) {
	if r.depth >= MaxProtoDepth {
		return nil, fmt.Errorf("protobuf recursion depth limit %d exceeded", MaxProtoDepth)
	}
	return &ProtoReader{buf: sub, pos: 0, depth: r.depth + 1}, nil
}

// Done reports whether all bytes in the current message have been consumed.
func (r *ProtoReader) Done() bool {
	return r.pos >= len(r.buf)
}

// Remaining returns the number of unread bytes.
func (r *ProtoReader) Remaining() int {
	return len(r.buf) - r.pos
}

// ReadVarint decodes a base-128 unsigned 64-bit varint.
func (r *ProtoReader) ReadVarint() (uint64, error) {
	var val uint64
	for shift := uint(0); shift < 64; shift += 7 {
		if r.pos >= len(r.buf) {
			return 0, fmt.Errorf("protobuf truncated varint at offset %d", r.pos)
		}
		b := r.buf[r.pos]
		r.pos++
		if shift == 63 && b > 0x01 {
			return 0, fmt.Errorf("protobuf varint overflow at offset %d", r.pos-1)
		}
		val |= uint64(b&0x7f) << shift
		if (b & 0x80) == 0 {
			return val, nil
		}
	}
	return 0, fmt.Errorf("protobuf varint exceeds 10 bytes at offset %d", r.pos)
}

// NextTag reads the next field tag and validates its field number and wire type.
func (r *ProtoReader) NextTag() (fieldNum uint32, wireType uint8, err error) {
	tag, err := r.ReadVarint()
	if err != nil {
		return 0, 0, err
	}
	wt := uint8(tag & 0x07)
	fn := tag >> 3
	if fn == 0 || fn > (1<<29)-1 {
		return 0, 0, fmt.Errorf("protobuf invalid field number %d", fn)
	}
	switch wt {
	case WireVarint, WireFixed64, WireBytes, WireFixed32:
		return uint32(fn), wt, nil
	default:
		return 0, 0, fmt.Errorf("protobuf unsupported wire type %d for field %d", wt, fn)
	}
}

// ReadFixed32 decodes a 4-byte little-endian unsigned integer.
func (r *ProtoReader) ReadFixed32() (uint32, error) {
	if r.Remaining() < 4 {
		return 0, fmt.Errorf("protobuf truncated fixed32 at offset %d", r.pos)
	}
	v := binary.LittleEndian.Uint32(r.buf[r.pos : r.pos+4])
	r.pos += 4
	return v, nil
}

// ReadFloat32 decodes a 4-byte IEEE-754 float32.
func (r *ProtoReader) ReadFloat32() (float32, error) {
	u, err := r.ReadFixed32()
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(u), nil
}

// ReadFixed64 decodes an 8-byte little-endian unsigned integer.
func (r *ProtoReader) ReadFixed64() (uint64, error) {
	if r.Remaining() < 8 {
		return 0, fmt.Errorf("protobuf truncated fixed64 at offset %d", r.pos)
	}
	v := binary.LittleEndian.Uint64(r.buf[r.pos : r.pos+8])
	r.pos += 8
	return v, nil
}

// ReadFloat64 decodes an 8-byte IEEE-754 float64.
func (r *ProtoReader) ReadFloat64() (float64, error) {
	u, err := r.ReadFixed64()
	if err != nil {
		return 0, err
	}
	return math.Float64frombits(u), nil
}

// ReadBytes decodes a length-delimited byte slice (wire type 2).
func (r *ProtoReader) ReadBytes() ([]byte, error) {
	n, err := r.ReadVarint()
	if err != nil {
		return nil, err
	}
	if n > uint64(r.Remaining()) {
		return nil, fmt.Errorf("protobuf length-delimited field size %d exceeds remaining %d bytes", n, r.Remaining())
	}
	out := r.buf[r.pos : r.pos+int(n)]
	r.pos += int(n)
	return out, nil
}

// ReadString decodes a length-delimited UTF-8 string (wire type 2).
func (r *ProtoReader) ReadString() (string, error) {
	b, err := r.ReadBytes()
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// Skip advances past an unknown field of the given wire type.
func (r *ProtoReader) Skip(wireType uint8) error {
	switch wireType {
	case WireVarint:
		_, err := r.ReadVarint()
		return err
	case WireFixed64:
		_, err := r.ReadFixed64()
		return err
	case WireBytes:
		_, err := r.ReadBytes()
		return err
	case WireFixed32:
		_, err := r.ReadFixed32()
		return err
	default:
		return fmt.Errorf("protobuf cannot skip unsupported wire type %d", wireType)
	}
}

// ReadPackedInt64 decodes a packed repeated int64 byte slice.
func ReadPackedInt64(b []byte) ([]int64, error) {
	pr := NewProtoReader(b)
	var out []int64
	for !pr.Done() {
		v, err := pr.ReadVarint()
		if err != nil {
			return nil, err
		}
		out = append(out, int64(v))
	}
	return out, nil
}

// ReadPackedInt32 decodes a packed repeated int32 byte slice.
func ReadPackedInt32(b []byte) ([]int32, error) {
	pr := NewProtoReader(b)
	var out []int32
	for !pr.Done() {
		v, err := pr.ReadVarint()
		if err != nil {
			return nil, err
		}
		out = append(out, int32(v))
	}
	return out, nil
}

// ReadPackedFloat32 decodes a packed repeated float32 byte slice.
func ReadPackedFloat32(b []byte) ([]float32, error) {
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("protobuf packed float32 length %d is not a multiple of 4", len(b))
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out, nil
}

// ReadPackedFloat64 decodes a packed repeated float64 byte slice.
func ReadPackedFloat64(b []byte) ([]float64, error) {
	if len(b)%8 != 0 {
		return nil, fmt.Errorf("protobuf packed float64 length %d is not a multiple of 8", len(b))
	}
	out := make([]float64, len(b)/8)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:]))
	}
	return out, nil
}

// ProtoBuilder is a minimal, zero-dependency protobuf wire encoder for
// constructing valid binary protobuf messages.
type ProtoBuilder struct {
	buf []byte
}

func (b *ProtoBuilder) appendVarint(v uint64) {
	for v >= 0x80 {
		b.buf = append(b.buf, byte(v)|0x80)
		v >>= 7
	}
	b.buf = append(b.buf, byte(v))
}

func (b *ProtoBuilder) appendTag(fieldNum uint32, wireType uint8) {
	b.appendVarint((uint64(fieldNum) << 3) | uint64(wireType))
}

// WriteVarint appends a varint-encoded field.
func (b *ProtoBuilder) WriteVarint(fieldNum uint32, v uint64) {
	b.appendTag(fieldNum, WireVarint)
	b.appendVarint(v)
}

// WriteInt64 appends a signed int64 field encoded as a 64-bit two's-complement varint.
func (b *ProtoBuilder) WriteInt64(fieldNum uint32, v int64) {
	b.WriteVarint(fieldNum, uint64(v))
}

// WriteFloat32 appends a 32-bit fixed IEEE-754 float field.
func (b *ProtoBuilder) WriteFloat32(fieldNum uint32, v float32) {
	b.appendTag(fieldNum, WireFixed32)
	var tmp [4]byte
	binary.LittleEndian.PutUint32(tmp[:], math.Float32bits(v))
	b.buf = append(b.buf, tmp[:]...)
}

// WriteFloat64 appends a 64-bit fixed IEEE-754 float field.
func (b *ProtoBuilder) WriteFloat64(fieldNum uint32, v float64) {
	b.appendTag(fieldNum, WireFixed64)
	var tmp [8]byte
	binary.LittleEndian.PutUint64(tmp[:], math.Float64bits(v))
	b.buf = append(b.buf, tmp[:]...)
}

// WriteBytes appends a length-delimited byte slice field.
func (b *ProtoBuilder) WriteBytes(fieldNum uint32, data []byte) {
	b.appendTag(fieldNum, WireBytes)
	b.appendVarint(uint64(len(data)))
	b.buf = append(b.buf, data...)
}

// WriteString appends a length-delimited string field.
func (b *ProtoBuilder) WriteString(fieldNum uint32, s string) {
	b.WriteBytes(fieldNum, []byte(s))
}

// WriteMessage appends an embedded message built by sub.
func (b *ProtoBuilder) WriteMessage(fieldNum uint32, sub *ProtoBuilder) {
	b.WriteBytes(fieldNum, sub.Bytes())
}

// WritePackedInt64 appends a packed repeated int64 field.
func (b *ProtoBuilder) WritePackedInt64(fieldNum uint32, vals []int64) {
	var sub ProtoBuilder
	for _, v := range vals {
		sub.appendVarint(uint64(v))
	}
	b.WriteBytes(fieldNum, sub.Bytes())
}

// WritePackedInt32 appends a packed repeated int32 field.
func (b *ProtoBuilder) WritePackedInt32(fieldNum uint32, vals []int32) {
	var sub ProtoBuilder
	for _, v := range vals {
		sub.appendVarint(uint64(int64(v)))
	}
	b.WriteBytes(fieldNum, sub.Bytes())
}

// WritePackedFloat32 appends a packed repeated float32 field.
func (b *ProtoBuilder) WritePackedFloat32(fieldNum uint32, vals []float32) {
	raw := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	b.WriteBytes(fieldNum, raw)
}

// Bytes returns the encoded protobuf wire bytes.
func (b *ProtoBuilder) Bytes() []byte {
	return b.buf
}
