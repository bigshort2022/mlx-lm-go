package mlx

// #include <mlx/c/mlx.h>
import "C"

// Seed seeds MLX's random number generator, whose state belongs to the
// calling thread.
func Seed(seed uint64) {
	check(C.mlx_random_seed(C.uint64_t(seed)))
}

// Categorical samples one index along axis from unnormalised log
// probabilities. The result is a uint32 array without that axis.
func Categorical(logits *Array, axis int) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		var globalKey C.mlx_array // empty: draw from the generator Seed seeds
		return C.mlx_random_categorical(&res.c, logits.c, C.int(axis), globalKey, s)
	})
}
