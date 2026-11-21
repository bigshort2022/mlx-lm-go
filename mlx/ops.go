package mlx

// #include <mlx/c/mlx.h>
import "C"

// This file holds the element-wise operations and Matmul. Binary operations
// broadcast their operands. Every operation panics with *Error if MLX rejects
// its arguments.

// Add returns a + b.
func Add(a, b *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_add(&res.c, a.c, b.c, s)
	})
}

// Subtract returns a - b.
func Subtract(a, b *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_subtract(&res.c, a.c, b.c, s)
	})
}

// Multiply returns a * b.
func Multiply(a, b *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_multiply(&res.c, a.c, b.c, s)
	})
}

// Divide returns a / b.
func Divide(a, b *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_divide(&res.c, a.c, b.c, s)
	})
}

// Negative returns -a.
func Negative(a *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_negative(&res.c, a.c, s)
	})
}

// Exp returns e raised to each element of a.
func Exp(a *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_exp(&res.c, a.c, s)
	})
}

// Log returns the natural logarithm of each element of a.
func Log(a *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_log(&res.c, a.c, s)
	})
}

// Sigmoid returns 1 / (1 + exp(-a)).
func Sigmoid(a *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_sigmoid(&res.c, a.c, s)
	})
}

// Maximum returns the larger of a and b, element by element.
func Maximum(a, b *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_maximum(&res.c, a.c, b.c, s)
	})
}

// Minimum returns the smaller of a and b, element by element.
func Minimum(a, b *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_minimum(&res.c, a.c, b.c, s)
	})
}

// Greater returns a bool array that is true where a > b.
func Greater(a, b *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_greater(&res.c, a.c, b.c, s)
	})
}

// Less returns a bool array that is true where a < b.
func Less(a, b *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_less(&res.c, a.c, b.c, s)
	})
}

// Where returns x where cond is true and y elsewhere.
func Where(cond, x, y *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_where(&res.c, cond.c, x.c, y.c, s)
	})
}

// AsType returns a converted to the given element type.
func AsType(a *Array, dtype DType) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_astype(&res.c, a.c, cDType(dtype), s)
	})
}

// Matmul returns the matrix product of a and b.
func Matmul(a, b *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_matmul(&res.c, a.c, b.c, s)
	})
}

// Addmm returns c + a @ b in one operation.
func Addmm(c, a, b *Array) *Array {
	const alpha, beta = 1, 1 // the scales of a @ b and of c
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_addmm(&res.c, c.c, a.c, b.c, alpha, beta, s)
	})
}

// GreaterEqual returns a bool array that is true where a >= b.
func GreaterEqual(a, b *Array) *Array {
	return apply(func(res *Array, s C.mlx_stream) C.int {
		return C.mlx_greater_equal(&res.c, a.c, b.c, s)
	})
}
