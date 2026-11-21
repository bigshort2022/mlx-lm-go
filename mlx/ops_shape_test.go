package mlx

import "testing"

// Expected values in this file were computed with the Python reference
// (mlx.core) on the same inputs.

// counting returns a float32 array of the given shape holding 0, 1, 2, ... in
// row-major order, like mx.arange(n).reshape(shape).
func counting(shape ...int) *Array {
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i)
	}
	return FromFloat32(data, shape...)
}

func TestReshape(t *testing.T) {
	inScope(t, func() {
		want := counting(24).Float32s()
		wantFloats(t, "reshape([2,3,4] -> [4,6])", Reshape(counting(2, 3, 4), 4, 6), []int{4, 6}, want, exact)
		wantFloats(t, "reshape([2,3,4] -> [-1,8])", Reshape(counting(2, 3, 4), -1, 8), []int{3, 8}, want, exact)
	})
}

func TestTranspose(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "transpose([2,3,4], axes 2,0,1)", Transpose(counting(2, 3, 4), 2, 0, 1), []int{4, 2, 3}, []float32{
			0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23,
		}, exact)
		wantFloats(t, "transpose([2,3,4]) reverses the axes", Transpose(counting(2, 3, 4)), []int{4, 3, 2}, []float32{
			0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23,
		}, exact)
	})
}

func TestExpandDims(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "expand_dims([3], 0)", ExpandDims(vec123(), 0), []int{1, 3}, []float32{1, 2, 3}, exact)
		wantFloats(t, "expand_dims([3], -1)", ExpandDims(vec123(), -1), []int{3, 1}, []float32{1, 2, 3}, exact)
	})
}

func TestSqueeze(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "squeeze([1,3,1], 0)", Squeeze(Zeros(Float32, 1, 3, 1), 0), []int{3, 1}, []float32{0, 0, 0}, exact)
		wantFloats(t, "squeeze([1,3,1], -1)", Squeeze(Zeros(Float32, 1, 3, 1), -1), []int{1, 3}, []float32{0, 0, 0}, exact)
	})
}

func TestConcatenate(t *testing.T) {
	inScope(t, func() {
		a := FromFloat32([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		b := FromFloat32([]float32{10, 20, 30, 40, 50, 60}, 2, 3)
		wantFloats(t, "concatenate(axis 0)", Concatenate(0, a, b), []int{4, 3},
			[]float32{1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60}, exact)
		wantFloats(t, "concatenate(axis -1)", Concatenate(-1, a, b), []int{2, 6},
			[]float32{1, 2, 3, 10, 20, 30, 4, 5, 6, 40, 50, 60}, exact)
	})
}

func TestSlice(t *testing.T) {
	inScope(t, func() {
		got := Slice(counting(4, 5), []int{1, 2}, []int{3, 5})
		wantFloats(t, "[4,5][1:3, 2:5]", got, []int{2, 3}, []float32{7, 8, 9, 12, 13, 14}, exact)
	})
}

func TestSlicePanicsWithoutOneEntryPerDimension(t *testing.T) {
	inScope(t, func() {
		mustPanic(t, func() { Slice(counting(4, 5), []int{1}, []int{3}) })
		mustPanic(t, func() { Slice(counting(4, 5), []int{1, 2}, []int{3}) })
	})
}

// threeByTwo is [[1,2],[3,4],[5,6]].
func threeByTwo() *Array { return FromFloat32([]float32{1, 2, 3, 4, 5, 6}, 3, 2) }

func TestTake(t *testing.T) {
	inScope(t, func() {
		rows := Take(threeByTwo(), FromInt32([]int32{2, 0, 2}, 3), 0)
		wantFloats(t, "take(rows 2,0,2)", rows, []int{3, 2}, []float32{5, 6, 1, 2, 5, 6}, exact)
		columns := Take(threeByTwo(), FromInt32([]int32{1, 1, 0}, 3), 1)
		wantFloats(t, "take(columns 1,1,0)", columns, []int{3, 3}, []float32{2, 2, 1, 4, 4, 3, 6, 6, 5}, exact)
	})
}

func TestTakeAlongAxis(t *testing.T) {
	inScope(t, func() {
		got := TakeAlongAxis(threeByTwo(), FromInt32([]int32{1, 0, 1}, 3, 1), 1)
		wantFloats(t, "take_along_axis(axis 1)", got, []int{3, 1}, []float32{2, 3, 6}, exact)
	})
}

func TestPutAlongAxis(t *testing.T) {
	inScope(t, func() {
		indices := FromInt32([]int32{1, 0, 1}, 3, 1)
		values := FromFloat32([]float32{10, 20, 30}, 3, 1)
		wantFloats(t, "put_along_axis(values)", PutAlongAxis(threeByTwo(), indices, values, 1), []int{3, 2},
			[]float32{1, 10, 20, 4, 5, 30}, exact)
		wantFloats(t, "put_along_axis(scalar)", PutAlongAxis(threeByTwo(), indices, Float(-1), 1), []int{3, 2},
			[]float32{1, -1, -1, 4, 5, -1}, exact)
	})
}

func TestSliceUpdate(t *testing.T) {
	inScope(t, func() {
		update := FromFloat32([]float32{-1, -2, -3, -4, -5, -6}, 2, 3)
		got := SliceUpdate(counting(4, 5), update, []int{1, 2}, []int{3, 5})
		wantFloats(t, "[4,5][1:3, 2:5] = update", got, []int{4, 5}, []float32{
			0, 1, 2, 3, 4,
			5, 6, -1, -2, -3,
			10, 11, -4, -5, -6,
			15, 16, 17, 18, 19,
		}, exact)
		got = SliceUpdate(counting(4, 5), Float(7), []int{1, 2}, []int{3, 5})
		wantFloats(t, "[4,5][1:3, 2:5] = 7", got, []int{4, 5}, []float32{
			0, 1, 2, 3, 4,
			5, 6, 7, 7, 7,
			10, 11, 7, 7, 7,
			15, 16, 17, 18, 19,
		}, exact)
	})
}

func TestSliceUpdateLeavesTheSourceAlone(t *testing.T) {
	inScope(t, func() {
		src := counting(2, 2)
		SliceUpdate(src, Float(9), []int{0, 0}, []int{1, 2})
		wantFloats(t, "source after SliceUpdate", src, []int{2, 2}, []float32{0, 1, 2, 3}, exact)
	})
}

func TestSliceUpdatePanicsWithoutOneEntryPerDimension(t *testing.T) {
	inScope(t, func() {
		mustPanic(t, func() { SliceUpdate(counting(4, 5), Float(7), []int{1}, []int{3}) })
		mustPanic(t, func() { SliceUpdate(counting(4, 5), Float(7), []int{1, 2}, []int{3}) })
	})
}

func TestScatterAddAxis(t *testing.T) {
	inScope(t, func() {
		a := FromFloat32([]float32{1, 2, 3, 4}, 1, 4)
		indices := FromInt32([]int32{1, 1, 3}, 1, 3)
		// Python: a.at[:, mx.array([1, 1, 3])].add(mx.array([10., 20, 30]))
		got := ScatterAddAxis(a, indices, FromFloat32([]float32{10, 20, 30}, 1, 3), 1)
		wantFloats(t, "scatter_add_axis(values)", got, []int{1, 4}, []float32{1, 32, 3, 34}, exact)
		// Python: a.at[:, mx.array([1, 1, 3])].subtract(0.5)
		got = ScatterAddAxis(a, indices, Float(-0.5), -1)
		wantFloats(t, "scatter_add_axis(scalar)", got, []int{1, 4}, []float32{1, 1, 3, 3.5}, exact)
	})
}
