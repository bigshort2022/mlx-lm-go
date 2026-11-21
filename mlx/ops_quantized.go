package mlx

/*
#include <stdlib.h>
#include <mlx/c/mlx.h>
*/
import "C"

import "unsafe"

// This file holds the operations on weights quantized by MLX. groupSize, bits
// and mode ("affine", "mxfp4", ...) must be the ones the weights were
// quantized with. biases is nil for modes that have none.

// QuantizedMatmul multiplies x by the quantized matrix w without
// dequantizing it first: x @ w.T with transpose, x @ w without.
func QuantizedMatmul(x, w, scales, biases *Array, transpose bool, groupSize, bits int, mode string) *Array {
	cmode := C.CString(mode)
	defer C.free(unsafe.Pointer(cmode))
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_quantized_matmul(&res.c, x.c, w.c, scales.c, optional(biases), C.bool(transpose),
			optionalInt(groupSize), optionalInt(bits), cmode, s)
	})
}

// Dequantize returns the floating point matrix that the quantized matrix w
// stands for.
func Dequantize(w, scales, biases *Array, groupSize, bits int, mode string) *Array {
	cmode := C.CString(mode)
	defer C.free(unsafe.Pointer(cmode))
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_dequantize(&res.c, w.c, scales.c, optional(biases), optionalInt(groupSize), optionalInt(bits),
			cmode, C.mlx_array{}, C.mlx_optional_dtype{}, s)
	})
}

// optionalInt returns v as a present mlx_optional_int.
func optionalInt(v int) C.mlx_optional_int {
	return C.mlx_optional_int{value: C.int(v), has_value: true}
}
