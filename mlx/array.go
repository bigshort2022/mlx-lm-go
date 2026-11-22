package mlx

// #include <mlx/c/mlx.h>
import "C"

import (
	"fmt"
	"unsafe"
)

// Array is a handle to an MLX array. An Array created inside Scoped belongs to
// that scope; one created outside any scope belongs to the caller, who must
// Free it.
//
// Arrays are lazy: an operation only records what to compute, and Eval,
// AsyncEval, Float32s and Int32s compute it. A lazy array is bound to the OS
// thread that built it, whose stream its operations run on. It must be
// evaluated on that thread, before any other thread uses it and before that
// thread ends. An evaluated array may be used from another thread.
type Array struct {
	c     C.mlx_array
	owner *scope // the scope that frees the array; nil if the caller does
	next  *Array // the next array in the owner's list
}

// own gives a new array to the innermost scope, or leaves it with the caller
// when no scope is open.
func own(a *Array) *Array {
	if current != nil {
		current.adopt(a)
	}
	return a
}

// apply runs one MLX operation on the calling thread's default GPU stream and
// returns its result. op stores the result in res.c and returns MLX's status.
func apply(op func(res *Array, s C.mlx_stream) C.int) *Array {
	res := &Array{}
	check(op(res, gpu()))
	return own(res)
}

// Share returns a separately owned handle to the same immutable array.
// It does not copy or evaluate the array.
func (a *Array) Share() *Array {
	src := a.live()
	ref := &Array{}
	check(C.mlx_array_set(&ref.c, src))
	return own(ref)
}

// Free releases the array. It is safe to call twice.
func (a *Array) Free() {
	if a.c.ctx == nil {
		return
	}
	C.mlx_array_free(a.c)
	a.c.ctx = nil
}

// Valid reports whether the array has not been freed.
func (a *Array) Valid() bool {
	return a.c.ctx != nil
}

// cInt converts a Go int to the C int MLX takes for sizes, axes and indices.
// It panics with *Error if the value does not fit, rather than truncate it.
func cInt(v int) C.int {
	c := C.int(v)
	if int(c) != v {
		panic(&Error{Msg: fmt.Sprintf("%d does not fit in a C int", v)})
	}
	return c
}

// cInts converts a shape or a list of axes with cInt.
func cInts(values []int) []C.int {
	out := make([]C.int, len(values))
	for i, v := range values {
		out[i] = cInt(v)
	}
	return out
}

// cDType converts a DType to the mlx_dtype MLX takes. It panics with *Error
// if MLX has no such type, because mlx-c uses the value as an index into a
// table without checking it.
func cDType(d DType) C.mlx_dtype {
	if d < Bool || d > Complex64 {
		panic(&Error{Msg: fmt.Sprintf("%v is not an MLX element type", d)})
	}
	return C.mlx_dtype(d)
}

// created adopts the result of an mlx-c constructor, which reports failure by
// returning an empty handle.
func created(c C.mlx_array) *Array {
	if c.ctx == nil {
		panic(takeError())
	}
	return own(&Array{c: c})
}

// fromData returns an array of the given shape and element type holding a
// copy of data. It panics with *Error unless the shape holds exactly
// len(data) elements, because MLX would read past the data.
func fromData[T float32 | int32](data []T, shape []int, dtype C.mlx_dtype) *Array {
	want := 1
	for _, d := range shape {
		if d < 0 {
			panic(&Error{Msg: fmt.Sprintf("shape %v has a negative dimension", shape)})
		}
		want *= d
	}
	if len(data) != want {
		panic(&Error{Msg: fmt.Sprintf("data has %d elements, shape %v holds %d", len(data), shape, want)})
	}
	dims := cInts(shape)
	if len(data) == 0 {
		return created(C.mlx_array_new_data(nil, unsafe.SliceData(dims), cInt(len(dims)), dtype))
	}
	// The argument is written as &data[0] so that cgo's pointer check looks at
	// the slice alone. Given a bare unsafe.Pointer it would inspect the whole
	// allocation the slice is part of and reject one that also holds pointers.
	return created(C.mlx_array_new_data(unsafe.Pointer(&data[0]), unsafe.SliceData(dims), cInt(len(dims)), dtype))
}

// FromFloat32 returns a float32 array of the given shape holding a copy of
// data. It panics with *Error unless data has exactly one element per element
// of the shape.
func FromFloat32(data []float32, shape ...int) *Array {
	return fromData(data, shape, C.MLX_FLOAT32)
}

// FromInt32 returns an int32 array of the given shape holding a copy of data.
// It panics with *Error unless data has exactly one element per element of the
// shape.
func FromInt32(data []int32, shape ...int) *Array {
	return fromData(data, shape, C.MLX_INT32)
}

// Float returns a zero-dimensional float32 array.
func Float(v float32) *Array {
	return created(C.mlx_array_new_float32(C.float(v)))
}

// Int returns a zero-dimensional int32 array.
func Int(v int32) *Array {
	return created(C.mlx_array_new_int(C.int(v)))
}

// Zeros returns an array of the given element type and shape filled with zeros.
func Zeros(dtype DType, shape ...int) *Array {
	dims := cInts(shape)
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_zeros(&res.c, unsafe.SliceData(dims), C.size_t(len(dims)), cDType(dtype), s)
	})
}

// live returns the MLX handle, or panics with *Error if the array was freed.
// The inspection functions of mlx-c have no status to return, so they cannot
// report a freed array themselves.
func (a *Array) live() C.mlx_array {
	if a.c.ctx == nil {
		panic(&Error{Msg: "use of a freed array"})
	}
	return a.c
}

// optional returns the handle of an argument that may be nil. mlx-c reads the
// empty handle as "no array".
func optional(a *Array) C.mlx_array {
	if a == nil {
		return C.mlx_array{}
	}
	return a.live()
}
