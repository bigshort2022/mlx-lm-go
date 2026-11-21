package mlx

import (
	"slices"
	"testing"
)

func TestFromFloat32RoundTrips(t *testing.T) {
	pin(t)
	data := []float32{1, 2, 3, 4, 5, 6}
	a := FromFloat32(data, 2, 3)
	defer a.Free()

	if got := a.Shape(); !slices.Equal(got, []int{2, 3}) {
		t.Errorf("Shape() = %v, want [2 3]", got)
	}
	if got := a.NDim(); got != 2 {
		t.Errorf("NDim() = %d, want 2", got)
	}
	if got := a.Size(); got != 6 {
		t.Errorf("Size() = %d, want 6", got)
	}
	if got := a.DType(); got != Float32 {
		t.Errorf("DType() = %v, want Float32", got)
	}
	if got := a.Dim(-1); got != 3 {
		t.Errorf("Dim(-1) = %d, want the last dimension 3", got)
	}
	if got := a.Dim(0); got != 2 {
		t.Errorf("Dim(0) = %d, want 2", got)
	}
	if got := a.Float32s(); !slices.Equal(got, data) {
		t.Errorf("Float32s() = %v, want %v", got, data)
	}
}

func TestFromInt32RoundTrips(t *testing.T) {
	pin(t)
	data := []int32{-1, 2, -3, 4, -5, 6}
	a := FromInt32(data, 2, 3)
	defer a.Free()

	if got := a.Shape(); !slices.Equal(got, []int{2, 3}) {
		t.Errorf("Shape() = %v, want [2 3]", got)
	}
	if got := a.NDim(); got != 2 {
		t.Errorf("NDim() = %d, want 2", got)
	}
	if got := a.Size(); got != 6 {
		t.Errorf("Size() = %d, want 6", got)
	}
	if got := a.DType(); got != Int32 {
		t.Errorf("DType() = %v, want Int32", got)
	}
	if got := a.Dim(-1); got != 3 {
		t.Errorf("Dim(-1) = %d, want the last dimension 3", got)
	}
	if got := a.Int32s(); !slices.Equal(got, data) {
		t.Errorf("Int32s() = %v, want %v", got, data)
	}
}

func TestFloatIsZeroDimensionalFloat32(t *testing.T) {
	pin(t)
	a := Float(1.5)
	defer a.Free()
	if a.NDim() != 0 || len(a.Shape()) != 0 || a.Size() != 1 {
		t.Errorf("NDim()=%d Shape()=%v Size()=%d, want a zero-dimensional array of one element", a.NDim(), a.Shape(), a.Size())
	}
	if got := a.DType(); got != Float32 {
		t.Errorf("DType() = %v, want Float32", got)
	}
	if got := a.Float32s(); !slices.Equal(got, []float32{1.5}) {
		t.Errorf("Float32s() = %v, want [1.5]", got)
	}
}

func TestIntIsZeroDimensionalInt32(t *testing.T) {
	pin(t)
	a := Int(7)
	defer a.Free()
	if a.NDim() != 0 || len(a.Shape()) != 0 || a.Size() != 1 {
		t.Errorf("NDim()=%d Shape()=%v Size()=%d, want a zero-dimensional array of one element", a.NDim(), a.Shape(), a.Size())
	}
	if got := a.DType(); got != Int32 {
		t.Errorf("DType() = %v, want Int32", got)
	}
	if got := a.Int32s(); !slices.Equal(got, []int32{7}) {
		t.Errorf("Int32s() = %v, want [7]", got)
	}
}

func TestZerosIsFilledWithZeros(t *testing.T) {
	pin(t)
	a := Zeros(Float32, 2, 2)
	defer a.Free()
	if got := a.Shape(); !slices.Equal(got, []int{2, 2}) {
		t.Errorf("Shape() = %v, want [2 2]", got)
	}
	if got := a.Float32s(); !slices.Equal(got, []float32{0, 0, 0, 0}) {
		t.Errorf("Float32s() = %v, want four zeros", got)
	}
	b := Zeros(Int32, 3)
	defer b.Free()
	if got := b.Int32s(); !slices.Equal(got, []int32{0, 0, 0}) {
		t.Errorf("Zeros(Int32, 3).Int32s() = %v, want three zeros", got)
	}
}

func TestFromFloat32PanicsWhenDataDoesNotFillShape(t *testing.T) {
	pin(t)
	mustPanic(t, func() { FromFloat32(make([]float32, 7), 2, 3) })
	mustPanic(t, func() { FromFloat32(make([]float32, 5), 2, 3) })
	mustPanic(t, func() { FromFloat32(nil, 2, 3) })
	mustPanic(t, func() { FromFloat32(make([]float32, 6), -2, -3) })
}

func TestFromInt32PanicsWhenDataDoesNotFillShape(t *testing.T) {
	pin(t)
	mustPanic(t, func() { FromInt32(make([]int32, 7), 2, 3) })
	mustPanic(t, func() { FromInt32(make([]int32, 5), 2, 3) })
}

func TestFromFloat32AcceptsAnEmptyShape(t *testing.T) {
	pin(t)
	a := FromFloat32(nil, 0, 3)
	defer a.Free()
	if !a.Valid() {
		t.Fatal("FromFloat32(nil, 0, 3) returned an invalid array")
	}
	if got := a.Shape(); !slices.Equal(got, []int{0, 3}) {
		t.Errorf("Shape() = %v, want [0 3]", got)
	}
	if got := a.Float32s(); len(got) != 0 {
		t.Errorf("Float32s() = %v, want no elements", got)
	}
}
