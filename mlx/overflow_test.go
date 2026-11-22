package mlx

import "testing"

// tooBig does not fit in a C int. Cut down to 32 bits it becomes 1, which is a
// legal axis, index and offset for the arrays below, so a conversion that is
// not checked would go unnoticed.
const tooBig = 1<<32 + 1

// Every function that hands a caller's int to MLX must refuse a value that C
// cannot hold rather than pass on its low 32 bits.
func TestIntArgumentThatOverflowsCIntPanics(t *testing.T) {
	matrix := func() *Array { return counting(2, 3) }
	indices := func() *Array { return FromInt32([]int32{0, 1}, 2, 1) }
	sequence := func() *Array { return counting(1, 2, 4, 8) }

	cases := []struct {
		name string
		call func()
	}{
		{"Argmax axis", func() { Argmax(matrix(), tooBig, false) }},
		{"Softmax axis", func() { Softmax(matrix(), tooBig) }},
		{"LogSumExp axis", func() { LogSumExp(matrix(), tooBig, false) }},
		{"Cumsum axis", func() { Cumsum(matrix(), tooBig) }},
		{"CumsumReverse axis", func() { CumsumReverse(matrix(), tooBig, true) }},
		{"Sort axis", func() { Sort(matrix(), tooBig) }},
		{"Argsort axis", func() { Argsort(matrix(), tooBig) }},
		{"ArgPartition kth", func() { ArgPartition(matrix(), tooBig, 1) }},
		{"ArgPartition axis", func() { ArgPartition(matrix(), 1, tooBig) }},
		{"Max axis", func() { Max(matrix(), tooBig, false) }},
		{"Sum axis", func() { Sum(matrix(), tooBig, false) }},

		{"ExpandDims axis", func() { ExpandDims(matrix(), tooBig) }},
		{"Squeeze axis", func() { Squeeze(counting(2, 1), tooBig) }},
		{"Concatenate axis", func() { Concatenate(tooBig, matrix(), matrix()) }},
		{"Take axis", func() { Take(matrix(), FromInt32([]int32{0}, 1), tooBig) }},
		{"TakeAlongAxis axis", func() { TakeAlongAxis(matrix(), indices(), tooBig) }},
		{"PutAlongAxis axis", func() { PutAlongAxis(matrix(), indices(), Float(0), tooBig) }},
		{"ScatterAddAxis axis", func() { ScatterAddAxis(matrix(), indices(), Float(1), tooBig) }},

		{"RoPE dims", func() { RoPE(sequence(), tooBig, false, 10000, 1, 0) }},
		{"RoPE offset", func() { RoPE(sequence(), 8, false, 10000, 1, tooBig) }},
		{"Categorical axis", func() { Categorical(matrix(), tooBig) }},

		{"QuantizedMatmul groupSize", func() { QuantizedMatmul(matrix(), matrix(), matrix(), nil, true, tooBig, 4, "affine") }},
		{"QuantizedMatmul bits", func() { QuantizedMatmul(matrix(), matrix(), matrix(), nil, true, 64, tooBig, "affine") }},
		{"Dequantize groupSize", func() { Dequantize(matrix(), matrix(), nil, tooBig, 4, "affine") }},
		{"Dequantize bits", func() { Dequantize(matrix(), matrix(), nil, 64, tooBig, "affine") }},

		// These take their ints as a list, which cInts has checked all along.
		{"Transpose axes", func() { Transpose(matrix(), 0, tooBig) }},
		{"SliceUpdate start", func() { SliceUpdate(matrix(), Float(0), []int{0, tooBig}, []int{2, 3}) }},
		{"SliceUpdate stop", func() { SliceUpdate(matrix(), Float(0), []int{0, 0}, []int{2, tooBig}) }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			inScope(t, func() {
				wantOverflowError(t, mustPanic(t, c.call), tooBig)
			})
		})
	}
}

// A DType outside MLX's list must not reach mlx-c, which uses it unchecked as
// an index into a table. The first value would otherwise be cut down to 10,
// which is Float32.
func TestDTypeThatMLXDoesNotHavePanics(t *testing.T) {
	for _, dtype := range []DType{1<<32 + 10, Complex64 + 1, -1} {
		inScope(t, func() {
			mustPanic(t, func() { AsType(vec123(), dtype) })
			mustPanic(t, func() { Zeros(dtype, 2) })
		})
	}
}
