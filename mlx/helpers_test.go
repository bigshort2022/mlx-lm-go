package mlx

import (
	"math"
	"runtime"
	"slices"
	"testing"
)

// pin locks the calling test to its OS thread until the test ends. MLX streams
// and the random key are per-thread, and an unpinned goroutine may resume on
// another thread after any cgo call. Every test that touches arrays calls it
// first; subtests run on their own goroutine and must pin themselves.
func pin(t *testing.T) {
	t.Helper()
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
}

// mustPanic runs fn and returns the *Error it panics with. It fails the test
// if fn returns or panics with anything else.
func mustPanic(t *testing.T, fn func()) (err *Error) {
	t.Helper()
	defer func() {
		t.Helper()
		switch r := recover().(type) {
		case *Error:
			err = r
		case nil:
			t.Fatal("did not panic, want a panic with *Error")
		default:
			t.Fatalf("panicked with %v (%T), want *Error", r, r)
		}
	}()
	fn()
	return nil
}

// inScope pins the test to its thread and runs fn inside a Scoped call, which
// must not fail.
func inScope(t *testing.T, fn func()) {
	t.Helper()
	pin(t)
	err := Scoped(func() error {
		fn()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// fixture pins the test to its thread and loads testdata/small.safetensors,
// the tensors Python computed. The arrays are freed when the test ends.
func fixture(t *testing.T) map[string]*Array {
	t.Helper()
	pin(t)
	arrays, _, err := LoadSafetensors(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		for _, a := range arrays {
			a.Free()
		}
	})
	return arrays
}

// wantFloats reports an error unless got is a float32 array of the given
// shape whose elements are within tol of want.
func wantFloats(t *testing.T, name string, got *Array, shape []int, want []float32, tol float64) {
	t.Helper()
	if !slices.Equal(got.Shape(), shape) {
		t.Errorf("%s has shape %v, want %v", name, got.Shape(), shape)
		return
	}
	if got.DType() != Float32 {
		t.Errorf("%s holds %v, want float32", name, got.DType())
		return
	}
	values := got.Float32s()
	for i := range want {
		if diff := math.Abs(float64(values[i]) - float64(want[i])); !(diff <= tol) {
			t.Errorf("%s = %v, want %v (element %d differs by %g, tolerance %g)", name, values, want, i, diff, tol)
			return
		}
	}
}

// wantInts reports an error unless got has the given shape and, converted to
// int32, exactly the elements of want. The conversion lets it check the bool
// and uint32 results of comparisons and index operations.
func wantInts(t *testing.T, name string, got *Array, shape []int, want []int32) {
	t.Helper()
	if !slices.Equal(got.Shape(), shape) {
		t.Errorf("%s has shape %v, want %v", name, got.Shape(), shape)
		return
	}
	if values := AsType(got, Int32).Int32s(); !slices.Equal(values, want) {
		t.Errorf("%s = %v, want %v", name, values, want)
	}
}
