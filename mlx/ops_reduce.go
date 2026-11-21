package mlx

// #include <mlx/c/mlx.h>
import "C"

// This file holds the operations that work along one axis. Axes may be
// negative to count from the end. Operations that return indices return
// uint32 arrays, as MLX does.

// Argmax returns the index of the largest element along axis. With keepDims
// the result keeps that axis with size one.
func Argmax(a *Array, axis int, keepDims bool) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_argmax_axis(&res.c, a.c, C.int(axis), C.bool(keepDims), s)
	})
}

// Softmax returns exp(a) normalised to sum to one along axis.
func Softmax(a *Array, axis int) *Array {
	const precise = false // the default of Python's mx.softmax
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_softmax_axis(&res.c, a.c, C.int(axis), precise, s)
	})
}

// LogSumExp returns log(sum(exp(a))) along axis, computed stably. With
// keepDims the result keeps that axis with size one.
func LogSumExp(a *Array, axis int, keepDims bool) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_logsumexp_axis(&res.c, a.c, C.int(axis), C.bool(keepDims), s)
	})
}

// Cumsum returns the running sum of a along axis, each element included in
// its own sum.
func Cumsum(a *Array, axis int) *Array {
	const (
		reverse   = false
		inclusive = true
	)
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_cumsum_axis(&res.c, a.c, C.int(axis), reverse, inclusive, C.mlx_optional_dtype{}, s)
	})
}

// Sort returns a sorted in ascending order along axis.
func Sort(a *Array, axis int) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_sort_axis(&res.c, a.c, C.int(axis), s)
	})
}

// Argsort returns the indices that sort a in ascending order along axis.
func Argsort(a *Array, axis int) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_argsort_axis(&res.c, a.c, C.int(axis), s)
	})
}

// ArgPartition returns indices that partition a along axis: the index at
// position kth is the one a full sort would put there, the indices before it
// select elements no larger, and those after it elements no smaller.
func ArgPartition(a *Array, kth, axis int) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_argpartition_axis(&res.c, a.c, C.int(kth), C.int(axis), s)
	})
}

// Max returns the largest element along axis. With keepDims the result keeps
// that axis with size one.
func Max(a *Array, axis int, keepDims bool) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_max_axis(&res.c, a.c, C.int(axis), C.bool(keepDims), s)
	})
}

// CumsumReverse returns the running sum of a along axis taken from the end:
// each element becomes the sum of the elements after it, plus itself when
// inclusive.
func CumsumReverse(a *Array, axis int, inclusive bool) *Array {
	const reverse = true
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_cumsum_axis(&res.c, a.c, C.int(axis), reverse, C.bool(inclusive), C.mlx_optional_dtype{}, s)
	})
}

// Sum returns the sum of the elements along axis. With keepDims the result
// keeps that axis with size one. The sum of a bool array is an int32 count.
func Sum(a *Array, axis int, keepDims bool) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_sum_axis(&res.c, a.c, C.int(axis), C.bool(keepDims), s)
	})
}
