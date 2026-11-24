package fp16

import (
	"math"
	"testing"
)

func TestExhaustiveRoundTrip(t *testing.T) {
	for i := 0; i <= 0xffff; i++ {
		h := uint16(i)
		f32 := ToFloat32(h)
		f64 := ToFloat64(h)
		if IsNaN(h) {
			if !math.IsNaN(f64) || !math.IsNaN(float64(f32)) {
				t.Fatalf("0x%04x expected NaN, got f32=%v f64=%v", h, f32, f64)
			}
			rt := FromFloat64(f64)
			if !IsNaN(rt) {
				t.Fatalf("0x%04x NaN roundtrip produced non-NaN 0x%04x", h, rt)
			}
			continue
		}
		if float64(f32) != f64 {
			t.Fatalf("0x%04x f32=%v != f64=%v", h, f32, f64)
		}
		if rt := FromFloat64(f64); rt != h {
			t.Fatalf("FromFloat64(ToFloat64(0x%04x)) = 0x%04x (val=%g)", h, rt, f64)
		}
		if rt := FromFloat32(f32); rt != h {
			t.Fatalf("FromFloat32(ToFloat32(0x%04x)) = 0x%04x", h, rt)
		}
		if rt := FromFloat64TiesAway(f64); rt != h {
			t.Fatalf("FromFloat64TiesAway(ToFloat64(0x%04x)) = 0x%04x", h, rt)
		}
	}
}

func TestRoundingAndBoundaries(t *testing.T) {
	// Exact halfway between 1.0 (0x3c00, even) and 1.0 + 2^-10 (0x3c01, odd): 1.0 + 2^-11
	halfEven := 1.0 + math.Ldexp(1.0, -11)
	if got := FromFloat64(halfEven); got != 0x3c00 {
		t.Fatalf("RNE(1+2^-11) = 0x%04x, want 0x3c00", got)
	}
	if got := FromFloat64TiesAway(halfEven); got != 0x3c01 {
		t.Fatalf("TiesAway(1+2^-11) = 0x%04x, want 0x3c01", got)
	}

	// Exact halfway between 0x3c01 (odd) and 0x3c02 (even): both round to 0x3c02
	halfOdd := 1.0 + math.Ldexp(3.0, -11)
	if got := FromFloat64(halfOdd); got != 0x3c02 {
		t.Fatalf("RNE(1+3*2^-11) = 0x%04x, want 0x3c02", got)
	}
	if got := FromFloat64TiesAway(halfOdd); got != 0x3c02 {
		t.Fatalf("TiesAway(1+3*2^-11) = 0x%04x, want 0x3c02", got)
	}

	// Negative halfway between -1.0 and -(1+2^-10)
	if got := FromFloat64(-halfEven); got != 0xbc00 {
		t.Fatalf("RNE(-(1+2^-11)) = 0x%04x, want 0xbc00", got)
	}
	if got := FromFloat64TiesAway(-halfEven); got != 0xbc01 {
		t.Fatalf("TiesAway(-(1+2^-11)) = 0x%04x, want 0xbc01", got)
	}

	// Subnormal boundary: half of min subnormal = 2^-25
	halfSub := math.Ldexp(1.0, -25)
	if got := FromFloat64(halfSub); got != PosZero {
		t.Fatalf("RNE(2^-25) = 0x%04x, want +0", got)
	}
	if got := FromFloat64TiesAway(halfSub); got != 0x0001 {
		t.Fatalf("TiesAway(2^-25) = 0x%04x, want 0x0001", got)
	}
	if got := FromFloat64(math.Nextafter(halfSub, 1.0)); got != 0x0001 {
		t.Fatalf("RNE(2^-25 + eps) = 0x%04x, want 0x0001", got)
	}

	// Subnormal rounding up into smallest normal (2^-14)
	nearNormal := math.Ldexp(1.0, -14) - math.Ldexp(1.0, -26)
	if got := FromFloat64(nearNormal); got != 0x0400 {
		t.Fatalf("RNE(2^-14 - 2^-26) = 0x%04x, want 0x0400", got)
	}

	// Max finite fp16 = 65504. Halfway to 65536 is 65520.
	if got := FromFloat64(65504.0); got != 0x7bff {
		t.Fatalf("FromFloat64(65504) = 0x%04x, want 0x7bff", got)
	}
	if got := FromFloat64(65520.0); got != PosInf {
		t.Fatalf("FromFloat64(65520) = 0x%04x, want +Inf", got)
	}
	if got := FromFloat64(-65520.0); got != NegInf {
		t.Fatalf("FromFloat64(-65520) = 0x%04x, want -Inf", got)
	}
	if got := FromFloat64(65519.0); got != 0x7bff {
		t.Fatalf("FromFloat64(65519) = 0x%04x, want 0x7bff", got)
	}

	// Signed zeros
	if got := FromFloat64(math.Copysign(0, -1)); got != NegZero {
		t.Fatalf("FromFloat64(-0.0) = 0x%04x, want 0x8000", got)
	}
	if got := FromFloat64(0.0); got != PosZero {
		t.Fatalf("FromFloat64(+0.0) = 0x%04x, want 0x0000", got)
	}
}

func TestULPLattice(t *testing.T) {
	if ULPDistance(PosZero, NegZero) != 0 {
		t.Fatalf("ULPDistance(+0, -0) != 0")
	}
	if ULPDistance(0x0001, 0x8001) != 2 {
		t.Fatalf("ULPDistance(+minsub, -minsub) = %d, want 2", ULPDistance(0x0001, 0x8001))
	}
	if ULPDistance(0x3c00, 0x3c01) != 1 {
		t.Fatalf("ULPDistance(1.0, 1.0+ulp) != 1")
	}
	// Subnormal max (0x03ff) to normal min (0x0400) is 1 ULP
	if ULPDistance(0x03ff, 0x0400) != 1 {
		t.Fatalf("ULPDistance(0x03ff, 0x0400) != 1")
	}
}
