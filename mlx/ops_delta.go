package mlx

/*
#include <mlx/c/mlx.h>

static int mlx_go_gated_delta_update(
	mlx_array *y, mlx_array *state, mlx_array q, mlx_array k, mlx_array v,
	mlx_array g, mlx_array beta, mlx_array initial_state, mlx_stream s) {
	mlx_vector_array out = {0};
	mlx_array no_mask = {0};
	// MLX 0.32.4 casts the FP32 decay to the common Q/K dtype. Promote
	// Q so the decay stays FP32, then restore the model output dtype.
	// The other inputs are promoted by mlx_fast_gated_delta_update itself.
	mlx_dtype dtype = mlx_array_dtype(q);
	bool promote = (dtype == MLX_BFLOAT16 || dtype == MLX_FLOAT16)
		&& mlx_array_dtype(g) == MLX_FLOAT32;
	mlx_array q32 = {0};
	mlx_array y32 = {0};
	int rc = 0;
	if (promote) {
		rc = mlx_astype(&q32, q, MLX_FLOAT32, s);
		q = q32;
	}
	rc = rc
		|| mlx_fast_gated_delta_update(&out, q, k, v, g, beta, initial_state, no_mask, s)
		|| mlx_vector_array_get(promote ? &y32 : y, out, 0)
		|| mlx_vector_array_get(state, out, 1)
		|| (promote && mlx_astype(y, y32, dtype, s));
	if (q32.ctx) mlx_array_free(q32);
	if (y32.ctx) mlx_array_free(y32);
	if (out.ctx) mlx_vector_array_free(out);
	if (rc) {
		if (y->ctx) mlx_array_free(*y);
		if (state->ctx) mlx_array_free(*state);
	}
	return rc;
}
*/
import "C"

// DepthwiseConv1D convolves x [batch, length, channels] with weight
// [channels, kernel, 1], with unit stride and dilation and no padding.
func DepthwiseConv1D(x, weight *Array) *Array {
	groups := cInt(x.Dim(-1))
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_conv1d(&res.c, x.c, weight.c, 1, 0, 1, groups, s)
	})
}

// GatedDeltaUpdate returns the unmasked gated delta rule output and final
// state. q and k are [batch, length, key heads, key dim]; v and the output
// are [batch, length, value heads]. q, k, v and beta use the model dtype,
// and g (decay) stays FP32 with output cast back to model dtype. State is
// FP32 [batch, value heads, value dim, key dim]; nil starts at zero.
func GatedDeltaUpdate(q, k, v, g, beta, initialState *Array) (y, state *Array) {
	y, state = &Array{}, &Array{}
	check(C.mlx_go_gated_delta_update(&y.c, &state.c, q.live(), k.live(), v.live(), g.live(), beta.live(), optional(initialState), gpu()))
	return own(y), own(state)
}

// Contiguous returns a in row-major storage, copying only if needed.
func Contiguous(a *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_contiguous(&res.c, a.c, false, s)
	})
}

// RMSNormUnweighted normalizes along the last axis without a learned weight.
func RMSNormUnweighted(a *Array, eps float32) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_fast_rms_norm(&res.c, a.c, optional(nil), C.float(eps), s)
	})
}

// LogAddExp returns log(exp(a) + exp(b)) without overflowing intermediate
// exponentials. Its inputs broadcast like those of Add.
func LogAddExp(a, b *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_logaddexp(&res.c, a.c, b.c, s)
	})
}
