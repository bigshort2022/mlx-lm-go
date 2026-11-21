package mlx

/*
#include <stdlib.h>
#include <mlx/c/mlx.h>
*/
import "C"

import "unsafe"

// This file holds MLX's fused implementations of transformer building blocks
// (mlx.core.fast in Python).

// RMSNorm returns x divided by the root mean square of its last axis, then
// multiplied by weight. eps is added under the root for stability.
func RMSNorm(x, weight *Array, eps float32) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_fast_rms_norm(&res.c, x.c, weight.c, C.float(eps), s)
	})
}

// RoPE applies rotary position embeddings to the last axis of x, whose first
// dims features are rotated. The second-to-last axis is the sequence, and its
// first position is numbered offset. base sets the rotation frequencies and
// scale multiplies the positions. traditional selects the original pairing of
// features, which is slower.
func RoPE(x *Array, dims int, traditional bool, base, scale float32, offset int) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		cbase := C.mlx_optional_float{value: C.float(base), has_value: true}
		var noFreqs C.mlx_array // frequencies come from base
		return C.mlx_fast_rope(&res.c, x.c, cInt(dims), C.bool(traditional), cbase, C.float(scale),
			cInt(offset), noFreqs, s)
	})
}

// ScaledDotProductAttention returns softmax(scale * q @ k.T + mask) @ v over
// [batch, heads, length, features] inputs. maskMode is "" for no mask,
// "causal" to let each position attend only to itself and earlier ones, or
// "array" to use mask, which is either additive or bool (true means attend).
// mask is nil unless maskMode is "array".
func ScaledDotProductAttention(q, k, v *Array, scale float32, maskMode string, mask *Array) *Array {
	cmode := C.CString(maskMode)
	defer C.free(unsafe.Pointer(cmode))
	return apply(func(res *Array, s C.mlx_stream) C.int {
		var noSinks C.mlx_array
		const forceFused = false
		return C.mlx_fast_scaled_dot_product_attention(&res.c, q.c, k.c, v.c, C.float(scale), cmode,
			optional(mask), noSinks, forceFused, s)
	})
}
