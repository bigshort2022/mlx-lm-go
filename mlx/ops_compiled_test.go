package mlx

import (
	"runtime"
	"slices"
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

func TestSwiGLUPanicsOnShapesThatDoNotBroadcast(t *testing.T) {
	inScope(t, func() {
		mustPanic(t, func() { SwiGLU(Zeros(Float32, 2, 3), Zeros(Float32, 4)) })
		// and keeps working afterwards
		wantFloats(t, "swiglu after a failure", SwiGLU(Float(1), Float(2)), []int{}, []float32{1.4621172}, 1e-6)
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
