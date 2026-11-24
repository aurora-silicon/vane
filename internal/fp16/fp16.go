// Package fp16 implements exact IEEE-754 binary16 conversions and ULP utilities.
package fp16

import (
	"encoding/binary"
	"fmt"
	"math"
)

const (
	PosZero uint16 = 0x0000
	NegZero uint16 = 0x8000
	PosInf  uint16 = 0x7c00
	NegInf  uint16 = 0xfc00
	QuietNaN uint16 = 0x7e00
)

// IsNaN reports whether h is an IEEE-754 binary16 NaN.
func IsNaN(h uint16) bool {
	return (h&0x7c00) == 0x7c00 && (h&0x03ff) != 0
}

// IsInf reports whether h is an infinity according to sign:
// sign > 0 checks +Inf, sign < 0 checks -Inf, sign == 0 checks either.
func IsInf(h uint16, sign int) bool {
	if (h & 0x7fff) != 0x7c00 {
		return false
	}
	if sign > 0 {
		return h == PosInf
	}
	if sign < 0 {
		return h == NegInf
	}
	return true
}

// IsFinite reports whether h is neither NaN nor Inf.
func IsFinite(h uint16) bool {
	return (h & 0x7c00) != 0x7c00
}

// UnbiasedExp returns the unbiased IEEE-754 binary16 exponent.
// Subnormal and zero values return -14 (the effective exponent of binary16 subnormals).
func UnbiasedExp(h uint16) int {
	exp := int((h >> 10) & 0x1f)
	if exp == 0 {
		return -14
	}
	return exp - 15
}

// ToFloat32 converts an IEEE-754 binary16 value to float32 exactly.
func ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32((h >> 10) & 0x1f)
	frac := uint32(h & 0x03ff)

	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		// Normalize subnormal
		e := int32(-14)
		for (frac & 0x0400) == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x03ff
		bits := sign | (uint32(e+127) << 23) | (frac << 13)
		return math.Float32frombits(bits)
	case 0x1f:
		if frac == 0 {
			return math.Float32frombits(sign | 0x7f800000)
		}
		// Quiet NaN preserving payload bits
		return math.Float32frombits(sign | 0x7fc00000 | (frac << 13))
	default:
		bits := sign | ((exp + (127 - 15)) << 23) | (frac << 13)
		return math.Float32frombits(bits)
	}
}

// ToFloat64 converts an IEEE-754 binary16 value to float64 exactly.
func ToFloat64(h uint16) float64 {
	return float64(ToFloat32(h))
}

// fromFloat64Rounded converts float64 directly to IEEE-754 binary16 without double rounding.
// If tiesAway is false, uses IEEE-754 round-to-nearest-even; if true, rounds ties away from zero.
func fromFloat64Rounded(f float64, tiesAway bool) uint16 {
	bits := math.Float64bits(f)
	sign := uint16((bits >> 48) & 0x8000)
	exp := int((bits >> 52) & 0x7ff)
	frac := bits & ((uint64(1) << 52) - 1)

	if exp == 0x7ff {
		if frac == 0 {
			return sign | PosInf
		}
		// Quiet NaN
		payload := uint16(frac >> 42)
		if payload == 0 {
			payload = 0x0200
		}
		return sign | 0x7e00 | (payload & 0x03ff)
	}
	if exp == 0 {
		// Float64 zero or subnormal (|f| < 2^-1022 << 2^-25) -> signed zero
		return sign
	}

	e := exp - 1023
	sig := (uint64(1) << 52) | frac // 53-bit significand with implicit 1 at bit 52

	// Overflow check (above max finite normal range; threshold handled after rounding)
	if e > 16 {
		return sign | PosInf
	}

	// Normal fp16 range: e >= -14
	if e >= -14 {
		const shift = 42
		q := sig >> shift
		rem := sig & ((uint64(1) << shift) - 1)
		const half = uint64(1) << (shift - 1)
		roundUp := rem > half || (rem == half && (tiesAway || (q&1) == 1))
		if roundUp {
			q++
			if q == 0x800 {
				q = 0x400
				e++
			}
		}
		if e > 15 {
			return sign | PosInf
		}
		return sign | uint16(e+15)<<10 | uint16(q&0x3ff)
	}

	// Subnormal / underflow range: e < -14. Target LSB is 2^-24, sig LSB is 2^(e-52).
	// shift = (-24) - (e - 52) = 28 - e.
	shift := 28 - e
	if shift > 53 {
		return sign
	}
	if shift == 53 {
		// e == -25: value is in [2^-25, 2^-24). Halfway is 2^-25 (frac == 0).
		if tiesAway || frac > 0 {
			return sign | 0x0001
		}
		return sign
	}

	// 43 <= shift <= 52
	q := sig >> uint(shift)
	rem := sig & ((uint64(1) << uint(shift)) - 1)
	half := uint64(1) << uint(shift-1)
	roundUp := rem > half || (rem == half && (tiesAway || (q&1) == 1))
	if roundUp {
		q++
	}
	if q == 0x400 {
		// Rounded up into smallest normal (2^-14)
		return sign | 0x0400
	}
	return sign | uint16(q)
}

// FromFloat64 converts a float64 to IEEE-754 binary16 using round-to-nearest-even.
func FromFloat64(f float64) uint16 {
	return fromFloat64Rounded(f, false)
}

// FromFloat32 converts a float32 to IEEE-754 binary16 using round-to-nearest-even.
func FromFloat32(f float32) uint16 {
	return fromFloat64Rounded(float64(f), false)
}

// FromFloat64TiesAway converts a float64 to IEEE-754 binary16 using ties-away-from-zero rounding.
func FromFloat64TiesAway(f float64) uint16 {
	return fromFloat64Rounded(f, true)
}

// OrderedInt maps a finite IEEE-754 binary16 value onto a monotonic int32 ULP lattice
// where -0.0 and +0.0 both map to 0.
func OrderedInt(h uint16) int32 {
	mag := int32(h & 0x7fff)
	if (h & 0x8000) != 0 {
		return -mag
	}
	return mag
}

// ULPDistance returns the ULP distance between two finite binary16 values.
func ULPDistance(a, b uint16) int32 {
	d := OrderedInt(a) - OrderedInt(b)
	if d < 0 {
		return -d
	}
	return d
}

// DecodeLE decodes a little-endian byte slice into IEEE-754 binary16 values.
func DecodeLE(b []byte) ([]uint16, error) {
	if len(b)%2 != 0 {
		return nil, fmt.Errorf("fp16 byte buffer length %d must be even", len(b))
	}
	out := make([]uint16, len(b)/2)
	for i := range out {
		out[i] = binary.LittleEndian.Uint16(b[2*i:])
	}
	return out, nil
}

// EncodeLE encodes IEEE-754 binary16 values into a little-endian byte slice.
func EncodeLE(vals []uint16) []byte {
	out := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(out[2*i:], v)
	}
	return out
}
