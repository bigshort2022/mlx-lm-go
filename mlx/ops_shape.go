package mlx

// #include <mlx/c/mlx.h>
import "C"

import "unsafe"

// This file holds the operations that rearrange, join or pick elements. Axes
// may be negative to count from the end.

// Reshape returns a with a new shape of the same size. One dimension may be
// -1, which MLX infers.
func Reshape(a *Array, shape ...int) *Array {
	dims := cInts(shape)
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_reshape(&res.c, a.c, unsafe.SliceData(dims), C.size_t(len(dims)), s)
	})
}

// Transpose returns a with its axes permuted: axis i of the result is axis
// axes[i] of a. Without axes it reverses their order.
func Transpose(a *Array, axes ...int) *Array {
	if len(axes) == 0 {
		return apply(func(res *Array, s C.mlx_stream) C.int {
			return C.mlx_transpose(&res.c, a.c, s)
		})
	}
	perm := cInts(axes)
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_transpose_axes(&res.c, a.c, unsafe.SliceData(perm), C.size_t(len(perm)), s)
	})
}

// ExpandDims returns a with a dimension of size one inserted at axis.
func ExpandDims(a *Array, axis int) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_expand_dims(&res.c, a.c, C.int(axis), s)
	})
}

// Squeeze returns a without the given axis, which must have size one.
func Squeeze(a *Array, axis int) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_squeeze_axis(&res.c, a.c, C.int(axis), s)
	})
}

// Concatenate joins arrays along an existing axis.
func Concatenate(axis int, arrs ...*Array) *Array {
	v := vector(arrs)
	defer C.mlx_vector_array_free(v)
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_concatenate_axis(&res.c, v, C.int(axis), s)
	})
}

// Slice returns the elements of a from start up to but not including stop,
// with stride one. start and stop have one entry per dimension of a.
func Slice(a *Array, start, stop []int) *Array {
	from, to := cInts(start), cInts(stop)
	strides := make([]C.int, len(from))
	for i := range strides {
		strides[i] = 1
	}
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_slice(&res.c, a.c,
			unsafe.SliceData(from), C.size_t(len(from)),
			unsafe.SliceData(to), C.size_t(len(to)),
			unsafe.SliceData(strides), C.size_t(len(strides)), s)
	})
}

// Take returns the slices of a along axis selected by indices, which may
// repeat.
func Take(a, indices *Array, axis int) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_take_axis(&res.c, a.c, indices.c, C.int(axis), s)
	})
}

// TakeAlongAxis returns the elements of a picked by indices along axis.
// indices has as many dimensions as a.
func TakeAlongAxis(a, indices *Array, axis int) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_take_along_axis(&res.c, a.c, indices.c, C.int(axis), s)
	})
}

// PutAlongAxis returns a copy of a with values written at indices along axis.
// values broadcasts against indices.
func PutAlongAxis(a, indices, values *Array, axis int) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_put_along_axis(&res.c, a.c, indices.c, values.c, C.int(axis), s)
	})
}

// SliceUpdate returns a copy of a with the elements from start up to but not
// including stop replaced by update, which broadcasts to that slice. start and
// stop have one entry per dimension of a.
func SliceUpdate(a, update *Array, start, stop []int) *Array {
	from, to := cInts(start), cInts(stop)
	strides := make([]C.int, len(from))
	for i := range strides {
		strides[i] = 1
	}
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_slice_update(&res.c, a.c, update.c,
			unsafe.SliceData(from), C.size_t(len(from)),
			unsafe.SliceData(to), C.size_t(len(to)),
			unsafe.SliceData(strides), C.size_t(len(strides)), s)
	})
}

// ScatterAddAxis returns a copy of a with values added at indices along axis.
// An index that repeats receives every one of its values. values broadcasts
// against indices.
func ScatterAddAxis(a, indices, values *Array, axis int) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_scatter_add_axis(&res.c, a.c, indices.c, values.c, C.int(axis), s)
	})
}
