package mlx

import (
	"slices"
	"testing"
)

func TestFloat32sPanicsOnInt32Array(t *testing.T) {
	pin(t)
	a := FromInt32([]int32{1, 2}, 2)
	defer a.Free()
	mustPanic(t, func() { a.Float32s() })
}

func TestInt32sPanicsOnFloat32Array(t *testing.T) {
	pin(t)
	a := FromFloat32([]float32{1, 2}, 2)
	defer a.Free()
	mustPanic(t, func() { a.Int32s() })
}

func TestDimPanicsWhenAxisIsOutOfRange(t *testing.T) {
	pin(t)
	a := FromFloat32([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	defer a.Free()
	mustPanic(t, func() { a.Dim(2) })
	mustPanic(t, func() { a.Dim(-3) })
}

func TestInspectingFreedArrayPanics(t *testing.T) {
	pin(t)
	a := FromFloat32([]float32{1, 2}, 2)
	a.Free()
	for name, inspect := range map[string]func(){
		"Shape":    func() { a.Shape() },
		"NDim":     func() { a.NDim() },
		"Dim":      func() { a.Dim(0) },
		"Size":     func() { a.Size() },
		"DType":    func() { a.DType() },
		"Float32s": func() { a.Float32s() },
		"Int32s":   func() { a.Int32s() },
		"Eval":     func() { Eval(a) },
		"Add":      func() { Add(a, a) },
	} {
		func() {
			defer func() {
				if _, ok := recover().(*Error); !ok {
					t.Errorf("%s on a freed array did not panic with *Error", name)
				}
			}()
			inspect()
		}()
	}
}

func TestEvalComputesEveryArray(t *testing.T) {
	pin(t)
	ClearCache() // settles memory still being released by earlier work
	err := Scoped(func() error {
		before := ActiveMemory()
		a := Zeros(Float32, 1024, 1024)
		b := Zeros(Float32, 1024, 1024)
		if lazy := ActiveMemory(); lazy >= before+4*mib {
			t.Errorf("ActiveMemory() went from %d to %d before Eval, want the arrays to be lazy", before, lazy)
		}
		Eval(a, b)
		if held := ActiveMemory(); held < before+8*mib {
			t.Errorf("ActiveMemory() went from %d to %d after Eval(a, b), want both 4 MiB arrays computed", before, held)
		}
		Eval()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestAsyncEvalStartsComputingEveryArray(t *testing.T) {
	pin(t)
	err := Scoped(func() error {
		a := Add(FromFloat32([]float32{1, 2}, 2), Float(1))
		b := Add(FromInt32([]int32{1, 2}, 2), Int(1))
		AsyncEval(a, b)
		if got := a.Float32s(); !slices.Equal(got, []float32{2, 3}) {
			t.Errorf("a.Float32s() = %v, want [2 3]", got)
		}
		if got := b.Int32s(); !slices.Equal(got, []int32{2, 3}) {
			t.Errorf("b.Int32s() = %v, want [2 3]", got)
		}
		AsyncEval()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
