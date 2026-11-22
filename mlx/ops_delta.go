package mlx

/*
#include <mlx/c/mlx.h>
#include <limits.h>
#include <string.h>
#include "gated_delta_metal.h"

static int mlx_go_gated_delta_fallback(
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

// Like the compiled SwiGLU, the source-only kernel descriptor is reused for
// the life of the process. MLX caches the Metal pipelines by specialization.
// This package is single-threaded by contract.
static mlx_fast_metal_kernel mlx_go_gated_delta_kernel;

static int mlx_go_gated_delta_init(void) {
	if (mlx_go_gated_delta_kernel.ctx) return 0;
	const char *input_names[] = {"q", "k", "v", "g", "beta", "state_in", "T"};
	const char *output_names[] = {"y", "state_out"};
	mlx_vector_string in = mlx_vector_string_new_data(input_names, 7);
	mlx_vector_string out = mlx_vector_string_new_data(output_names, 2);
	if (in.ctx && out.ctx) {
		mlx_go_gated_delta_kernel = mlx_fast_metal_kernel_new(
			"go_gated_delta_step", in, out, mlx_go_gated_delta_source, "", true, false);
	}
	if (out.ctx) mlx_vector_string_free(out);
	if (in.ctx) mlx_vector_string_free(in);
	return !mlx_go_gated_delta_kernel.ctx;
}

static bool mlx_go_delta_shape(mlx_array a, const int *shape, size_t rank) {
	return mlx_array_ndim(a) == rank
		&& memcmp(mlx_array_shape(a), shape, rank * sizeof(*shape)) == 0;
}

static int mlx_go_gated_delta_update(
	mlx_array *y, mlx_array *state, mlx_array q, mlx_array k, mlx_array v,
	mlx_array g, mlx_array beta, mlx_array initial_state, mlx_stream s) {
	if (mlx_array_ndim(q) != 4 || mlx_array_ndim(v) != 4) {
		mlx_error("[gated_delta_update] queries and values must be rank 4");
		return 1;
	}
	const int *q_shape = mlx_array_shape(q);
	const int *v_shape = mlx_array_shape(v);
	int batch = q_shape[0], length = q_shape[1], hk = q_shape[2], dk = q_shape[3];
	int hv = v_shape[2], dv = v_shape[3];
	int gate_shape[] = {batch, length, hv};
	int state_shape[] = {batch, hv, dv, dk};
	if (batch <= 0 || length <= 0 || hk <= 0 || dk <= 0 || hv <= 0 || dv <= 0
		|| hv % hk != 0 || batch > INT_MAX / hv
		|| v_shape[0] != batch || v_shape[1] != length
		|| !mlx_go_delta_shape(k, q_shape, 4)
		|| !mlx_go_delta_shape(g, gate_shape, 3)
		|| !mlx_go_delta_shape(beta, gate_shape, 3)
		|| (initial_state.ctx && !mlx_go_delta_shape(initial_state, state_shape, 4))) {
		mlx_error("[gated_delta_update] incompatible input shapes");
		return 1;
	}
	mlx_dtype dtype = mlx_array_dtype(q);
	mlx_dtype beta_dtype = mlx_array_dtype(beta);
	if ((dtype != MLX_BFLOAT16 && dtype != MLX_FLOAT16 && dtype != MLX_FLOAT32)
		|| mlx_array_dtype(k) != dtype || mlx_array_dtype(v) != dtype
		|| mlx_array_dtype(g) != MLX_FLOAT32
		|| (beta_dtype != dtype && beta_dtype != MLX_FLOAT32)
		|| (initial_state.ctx && mlx_array_dtype(initial_state) != MLX_FLOAT32)) {
		mlx_error("[gated_delta_update] incompatible input dtypes");
		return 1;
	}
	// The reference Metal kernel assigns Dk / 32 elements to each lane.
	// Keep the core's ops fallback for key dimensions that do not tile.
	if (dk % 32 != 0) {
		return mlx_go_gated_delta_fallback(y, state, q, k, v, g, beta, initial_state, s);
	}
	if (mlx_go_gated_delta_init()) return 1;

	mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
	if (!config.ctx) return 1;
	mlx_array zero_state = {0};
	mlx_array tokens = {0};
	mlx_vector_array in = {0}, out = {0};
	int rc = mlx_fast_metal_kernel_config_add_output_arg(config, v_shape, 4, dtype)
		|| mlx_fast_metal_kernel_config_add_output_arg(config, state_shape, 4, MLX_FLOAT32)
		|| mlx_fast_metal_kernel_config_set_grid(config, 32, dv, batch * hv)
		|| mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1)
		|| mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", dtype)
		|| mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", dk)
		|| mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", dv)
		|| mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", hk)
		|| mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", hv);
	if (!rc && !initial_state.ctx) {
		rc = mlx_zeros(&zero_state, state_shape, 4, MLX_FLOAT32, s);
		initial_state = zero_state;
	}
	if (!rc) {
		tokens = mlx_array_new_int(length);
		if (tokens.ctx) {
			const mlx_array inputs[] = {q, k, v, g, beta, initial_state, tokens};
			in = mlx_vector_array_new_data(inputs, 7);
		}
		rc = !in.ctx
			|| mlx_fast_metal_kernel_apply(&out, mlx_go_gated_delta_kernel, in, config, s)
			|| mlx_vector_array_get(y, out, 0)
			|| mlx_vector_array_get(state, out, 1);
	}
	if (out.ctx) mlx_vector_array_free(out);
	if (in.ctx) mlx_vector_array_free(in);
	if (tokens.ctx) mlx_array_free(tokens);
	if (zero_state.ctx) mlx_array_free(zero_state);
	mlx_fast_metal_kernel_config_free(config);
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
// are [batch, length, value heads, value dim]. g is FP32 decay, and beta
// is in model precision or FP32, both [batch, length, value heads].
// State is FP32 [batch, value heads, value dim, key dim]; nil starts at zero.
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
