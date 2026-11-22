package mlx

import (
	"math"
	"testing"
)

// The inputs and expected values of these tests are in the fixture; Python's
// mlx.core.fast computed the expected ones.

const fastTolerance = 1e-5

func TestRMSNorm(t *testing.T) {
	f := fixture(t)
	inScope(t, func() {
		got := RMSNorm(f["rms.x"], f["rms.weight"], 1e-5)
		wantFloats(t, "rms_norm", got, []int{2, 4}, f["rms.expected"].Float32s(), fastTolerance)
	})
}

func TestRoPE(t *testing.T) {
	f := fixture(t)
	inScope(t, func() {
		shape := []int{1, 2, 4, 8}
		wantFloats(t, "rope(offset 0)", RoPE(f["rope.x"], 8, false, 10000, 1, 0),
			shape, f["rope.expected_offset0"].Float32s(), fastTolerance)
		wantFloats(t, "rope(offset 3)", RoPE(f["rope.x"], 8, false, 10000, 1, 3),
			shape, f["rope.expected_offset3"].Float32s(), fastTolerance)
		wantFloats(t, "rope(traditional, base 100, scale 0.5, offset 1)", RoPE(f["rope.x"], 8, true, 100, 0.5, 1),
			shape, f["rope.expected_traditional"].Float32s(), fastTolerance)
	})
}

func TestScaledDotProductAttention(t *testing.T) {
	f := fixture(t)
	inScope(t, func() {
		q, k, v := f["sdpa.q"], f["sdpa.k"], f["sdpa.v"]
		scale := float32(1 / math.Sqrt(8))
		shape := []int{1, 2, 4, 8}
		wantFloats(t, "attention without a mask", ScaledDotProductAttention(q, k, v, scale, "", nil),
			shape, f["sdpa.expected_unmasked"].Float32s(), fastTolerance)
		wantFloats(t, "attention with a causal mask", ScaledDotProductAttention(q, k, v, scale, "causal", nil),
			shape, f["sdpa.expected_causal"].Float32s(), fastTolerance)
		wantFloats(t, "attention with a mask array", ScaledDotProductAttention(q, k, v, scale, "array", f["sdpa.mask"]),
			shape, f["sdpa.expected_array"].Float32s(), fastTolerance)
	})
}
