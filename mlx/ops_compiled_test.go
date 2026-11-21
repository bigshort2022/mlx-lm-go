package mlx

import (
	"runtime"
	"slices"
	"strings"
	"testing"
)

// Expected values: mlx_lm.models.activations.swiglu in Python.

func TestSwiGLU(t *testing.T) {
	inScope(t, func() {
		gate := FromFloat32([]float32{-2, -0.5, 0, 1, 3}, 5)
		x := FromFloat32([]float32{1, 2, 3, 4, -5}, 5)
		wantFloats(t, "swiglu(gate, x)", SwiGLU(gate, x), []int{5},
			[]float32{-0.238406, -0.377541, 0, 2.924234, -14.288611}, 1e-6)
	})
}

// The function is compiled without fixed shapes: it must serve other shapes
// and element types than the first it saw.
func TestSwiGLUWithOtherShapesAndTypes(t *testing.T) {
	inScope(t, func() {
		SwiGLU(FromFloat32([]float32{1}, 1), FromFloat32([]float32{1}, 1))
		gate := FromFloat32([]float32{-2, -0.5, 0, 1, 3, 1}, 1, 2, 3)
		x := FromFloat32([]float32{1, 2, 3, 4, -5, 0.5}, 1, 2, 3)
		wantFloats(t, "swiglu([1,2,3])", SwiGLU(gate, x), []int{1, 2, 3},
			[]float32{-0.238406, -0.377541, 0, 2.924234, -14.288611, 0.365529}, 1e-6)

		half := SwiGLU(AsType(gate, Float16), AsType(x, Float16))
		if half.DType() != Float16 {
			t.Fatalf("swiglu of float16 arrays is %v, want float16", half.DType())
		}
		wantFloats(t, "swiglu(float16)", AsType(half, Float32), []int{1, 2, 3},
			[]float32{-0.238406, -0.377541, 0, 2.924234, -14.288611, 0.365529}, 1e-2)
	})
}

// The result must be exactly what the three separate operations give, so
// that fusing them does not change what a model computes.
func TestSwiGLUEqualsItsUnfusedDefinition(t *testing.T) {
	inScope(t, func() {
		data := make([]float32, 4096)
		for i := range data {
			data[i] = float32(i%257)/16 - 8
		}
		gate := AsType(FromFloat32(data, 64, 64), Float16)
		x := AsType(FromFloat32(data, 64, 64), Float16)
		x = Transpose(x)
		want := Multiply(Multiply(gate, Sigmoid(gate)), x)
		got := SwiGLU(gate, x)
		if !slices.Equal(AsType(got, Float32).Float32s(), AsType(want, Float32).Float32s()) {
			t.Error("the fused swiglu differs from silu(gate) * x computed step by step")
		}
	})
}

// The compiled function does not check shapes itself: once it has seen a
// rank, it takes any two shapes of that rank and reads past the smaller one.
// SwiGLU must reject what Multiply rejects, with the same message.
func TestSwiGLUPanicsOnShapesThatDoNotBroadcast(t *testing.T) {
	for _, pair := range [][2][]int{
		{{1, 6, 64}, {1, 6, 128}},
		{{2, 3}, {2, 4}},
		{{2, 3}, {3, 3}},
		{{4, 5}, {2, 3}},
		{{2, 3}, {4}},
	} {
		inScope(t, func() {
			gate, x := Zeros(Float32, pair[0]...), Zeros(Float32, pair[1]...)
			// Let the compiled function see the rank with shapes that match.
			SwiGLU(gate, gate)
			SwiGLU(x, x)
			want := mustPanic(t, func() { Multiply(gate, x) })
			got := mustPanic(t, func() { SwiGLU(gate, x) })
			// mlx-c appends where in its source the error was raised.
			if got == nil || want == nil || !strings.HasPrefix(want.Msg, got.Msg) || !strings.HasPrefix(got.Msg, "[broadcast_shapes]") {
				t.Errorf("SwiGLU(%v, %v) panicked with %v, want the error of Multiply: %v", pair[0], pair[1], got, want)
			}
		})
	}
	inScope(t, func() {
		// and keeps working afterwards
		wantFloats(t, "swiglu after a failure", SwiGLU(Float(1), Float(2)), []int{}, []float32{1.4621172}, 1e-6)
	})
}

// Expected values: gate * sigmoid(gate) * x with the same broadcasting.
func TestSwiGLUBroadcasts(t *testing.T) {
	inScope(t, func() {
		column := FromFloat32([]float32{1, 3}, 2, 1)
		row := FromFloat32([]float32{1, 2, -5}, 1, 3)
		SwiGLU(Zeros(Float32, 2, 3), Zeros(Float32, 2, 3)) // the rank is traced without broadcasting
		for name, c := range map[string]struct {
			gate, x *Array
			shape   []int
		}{
			"[2,1] with [1,3]": {column, row, []int{2, 3}},
			"[1,3] with [2,1]": {row, column, []int{2, 3}},
			"[2,1] with [3]":   {column, FromFloat32([]float32{1, 2, -5}, 3), []int{2, 3}},
			"[] with [1,3]":    {Float(3), row, []int{1, 3}},
			"[2,1] with []":    {column, Float(2), []int{2, 1}},
			"[2,3] with [1,3]": {Add(column, Zeros(Float32, 2, 3)), row, []int{2, 3}},
		} {
			want := Multiply(Multiply(c.gate, Sigmoid(c.gate)), c.x)
			got := SwiGLU(c.gate, c.x)
			if !slices.Equal(got.Shape(), c.shape) || !slices.Equal(got.Float32s(), want.Float32s()) {
				t.Errorf("swiglu(%s) = %v %v, want %v %v", name, got.Shape(), got.Float32s(), c.shape, want.Float32s())
			}
		}
	})
}

// MLX compiles per thread. Every thread that uses the package must get its
// own, working compilation.
func TestSwiGLUOnASecondThread(t *testing.T) {
	inScope(t, func() {
		SwiGLU(Float(1), Float(2))
	})
	done := make(chan []float32)
	go func() {
		runtime.LockOSThread() // never unlocked: the thread ends with the goroutine
		var values []float32
		_ = Scoped(func() error {
			values = SwiGLU(Float(1), Float(2)).Float32s()
			return nil
		})
		done <- values
	}()
	if got := <-done; len(got) != 1 || got[0] < 1.462 || got[0] > 1.4622 {
		t.Errorf("swiglu(1, 2) on a second thread = %v, want 1.4621172", got)
	}
}
