package mlx

// #include <mlx/c/mlx.h>
import "C"

import (
	"fmt"
	"slices"
	"unsafe"
)

// This file reads arrays: their shape and type, and, after evaluating them,
// their elements.

// Shape returns the size of each dimension.
func (a *Array) Shape() []int {
	c := a.live()
	dims := unsafe.Slice(C.mlx_array_shape(c), C.mlx_array_ndim(c))
	shape := make([]int, len(dims))
	for i, d := range dims {
		shape[i] = int(d)
	}
	return shape
}

// NDim returns the number of dimensions.
func (a *Array) NDim() int {
	return int(C.mlx_array_ndim(a.live()))
}

// Dim returns the size of one dimension. Negative axes count from the end. It
// panics with *Error if the array has no such axis.
func (a *Array) Dim(axis int) int {
	c := a.live()
	ndim := int(C.mlx_array_ndim(c))
	index := axis
	if index < 0 {
		index += ndim
	}
	if index < 0 || index >= ndim {
		panic(&Error{Msg: fmt.Sprintf("axis %d is out of range for an array with %d dimensions", axis, ndim)})
	}
	return int(C.mlx_array_dim(c, C.int(index)))
}

// Size returns the number of elements.
func (a *Array) Size() int {
	return int(C.mlx_array_size(a.live()))
}

// DType returns the element type.
func (a *Array) DType() DType {
	return DType(C.mlx_array_dtype(a.live()))
}

// Float32s evaluates the array and returns a copy of its elements in row-major
// order. It panics with *Error unless the element type is Float32.
func (a *Array) Float32s() []float32 {
	return elements[float32](a, Float32, func(c C.mlx_array) unsafe.Pointer {
		return unsafe.Pointer(C.mlx_array_data_float32(c))
	})
}

// Int32s evaluates the array and returns a copy of its elements in row-major
// order. It panics with *Error unless the element type is Int32.
func (a *Array) Int32s() []int32 {
	return elements[int32](a, Int32, func(c C.mlx_array) unsafe.Pointer {
		return unsafe.Pointer(C.mlx_array_data_int32(c))
	})
}

// elements evaluates a and returns a copy of its elements, which must be of
// type want. data is the mlx_array_data_* function for T.
func elements[T float32 | int32](a *Array, want DType, data func(C.mlx_array) unsafe.Pointer) []T {
	if got := a.DType(); got != want {
		panic(&Error{Msg: fmt.Sprintf("array holds %v, not %v", got, want)})
	}
	Eval(a)
	src := a.c
	var rowMajor C.bool
	check(C._mlx_array_is_row_contiguous(&rowMajor, src))
	if !rowMajor {
		// a is a strided view of another array's buffer, as transposes and
		// slices are. Its buffer is not in element order, so read a copy that is.
		var packed C.mlx_array
		check(C.mlx_contiguous(&packed, a.c, false, gpu()))
		defer C.mlx_array_free(packed)
		check(C.mlx_array_eval(packed))
		src = packed
	}
	return slices.Clone(unsafe.Slice((*T)(data(src)), a.Size()))
}

// vector returns an mlx_vector_array holding the given arrays. The caller
// must free it.
func vector(arrs []*Array) C.mlx_vector_array {
	handles := make([]C.mlx_array, len(arrs))
	for i, a := range arrs {
		handles[i] = a.live()
	}
	return C.mlx_vector_array_new_data(unsafe.SliceData(handles), C.size_t(len(handles)))
}

// Eval computes the given arrays and waits for the results.
func Eval(arrs ...*Array) {
	if len(arrs) == 1 {
		check(C.mlx_array_eval(arrs[0].c))
		return
	}
	v := vector(arrs)
	defer C.mlx_vector_array_free(v)
	check(C.mlx_eval(v))
}

// AsyncEval starts computing the given arrays and returns without waiting.
func AsyncEval(arrs ...*Array) {
	v := vector(arrs)
	defer C.mlx_vector_array_free(v)
	check(C.mlx_async_eval(v))
}
