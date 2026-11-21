package mlx

/*
#include <mlx/c/mlx.h>

// swiglu(gate, x) = gate * sigmoid(gate) * x. MLX calls this once per thread
// and per combination of element types to record the operations, and from
// then on runs them as one fused kernel.
static int mlx_go_swiglu_trace(mlx_vector_array *res, const mlx_vector_array in) {
	mlx_stream s = mlx_default_gpu_stream_new();
	mlx_array gate = mlx_array_new();
	mlx_array x = mlx_array_new();
	mlx_array sigmoid = mlx_array_new();
	mlx_array silu = mlx_array_new();
	mlx_array out = mlx_array_new();
	int rc = mlx_vector_array_get(&gate, in, 0)
		|| mlx_vector_array_get(&x, in, 1)
		|| mlx_sigmoid(&sigmoid, gate, s)
		|| mlx_multiply(&silu, gate, sigmoid, s)
		|| mlx_multiply(&out, silu, x, s)
		|| mlx_vector_array_set_value(res, out);
	mlx_array_free(out);
	mlx_array_free(silu);
	mlx_array_free(sigmoid);
	mlx_array_free(x);
	mlx_array_free(gate);
	mlx_stream_free(s);
	return rc != 0;
}

// The compiled form of mlx_go_swiglu_trace, made on first use and kept for
// the life of the process. A plain global, because the package is
// single-threaded by contract.
static mlx_closure mlx_go_swiglu_compiled;

static int mlx_go_swiglu(mlx_array *res, const mlx_array gate, const mlx_array x) {
	if (mlx_go_swiglu_compiled.ctx == NULL) {
		mlx_closure trace = mlx_closure_new_func(mlx_go_swiglu_trace);
		mlx_closure compiled = mlx_closure_new();
		const bool shapeless = true; // one compilation serves every shape
		int rc = mlx_compile(&compiled, trace, shapeless);
		mlx_closure_free(trace);
		if (rc != 0) {
			mlx_closure_free(compiled);
			return 1;
		}
		mlx_go_swiglu_compiled = compiled;
	}
	const mlx_array inputs[2] = {gate, x};
	mlx_vector_array in = mlx_vector_array_new_data(inputs, 2);
	mlx_vector_array out = mlx_vector_array_new();
	int rc = mlx_closure_apply(&out, mlx_go_swiglu_compiled, in)
		|| mlx_vector_array_get(res, out, 0);
	mlx_vector_array_free(out);
	mlx_vector_array_free(in);
	return rc != 0;
}
*/
import "C"

// This file holds the functions that Python builds with mx.compile, which
// fuses a chain of element-wise operations into one kernel: one pass over the
// data and no intermediate arrays.

// SwiGLU returns silu(gate) * x, where silu(g) is g * sigmoid(g): the
// activation of gated feed-forward layers (mlx_lm.models.activations.swiglu).
func SwiGLU(gate, x *Array) *Array {
	res := &Array{}
	check(C.mlx_go_swiglu(&res.c, gate.live(), x.live()))
	return own(res)
}
