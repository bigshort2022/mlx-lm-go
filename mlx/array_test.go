package mlx

import (
	"errors"
	"math"
	"runtime"
	"slices"
	"strconv"
	"strings"
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

// floatNode and intNode have the shape of a pooled buffer: an array inside a
// heap struct that also holds a Go pointer. cgo's pointer check rejects a
// pointer into such a struct unless it can see that only the array is passed.
type floatNode struct {
	buf  [4]float32
	next *floatNode
}

type intNode struct {
	buf  [4]int32
	next *intNode
}

// heapSink keeps the nodes of the tests below on the heap.
var heapSink any

func TestFromFloat32AcceptsArrayFieldOfStructWithPointers(t *testing.T) {
	pin(t)
	node := &floatNode{buf: [4]float32{1, 2, 3, 4}, next: &floatNode{}}
	heapSink = node
	a := FromFloat32(node.buf[:], 4)
	defer a.Free()
	if got := a.Float32s(); !slices.Equal(got, node.buf[:]) {
		t.Errorf("Float32s() = %v, want %v", got, node.buf)
	}
}

func TestFromInt32AcceptsArrayFieldOfStructWithPointers(t *testing.T) {
	pin(t)
	node := &intNode{buf: [4]int32{1, 2, 3, 4}, next: &intNode{}}
	heapSink = node
	a := FromInt32(node.buf[:], 4)
	defer a.Free()
	if got := a.Int32s(); !slices.Equal(got, node.buf[:]) {
		t.Errorf("Int32s() = %v, want %v", got, node.buf)
	}
}

// onOtherThread runs fn in a scope on another OS thread and waits for it. The
// caller must be pinned: it keeps its own thread while it waits, so the new
// goroutine cannot be given that one.
func onOtherThread(fn func() error) error {
	done := make(chan error)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		done <- Scoped(fn)
	}()
	return <-done
}

// twoThree returns the caller-owned array [1,2] + 1, evaluated or still lazy.
func twoThree(t *testing.T, evaluate bool) *Array {
	t.Helper()
	var a *Array
	err := Scoped(func() error {
		a = Add(FromFloat32([]float32{1, 2}, 2), Float(1))
		if evaluate {
			Eval(a)
		}
		Keep(a)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(a.Free)
	return a
}

// The two tests below hold the Array documentation to its thread rules.

func TestEvaluatedArrayCanBeUsedOnAnotherThread(t *testing.T) {
	pin(t)
	a := twoThree(t, true)
	var got []float32
	err := onOtherThread(func() error {
		got = Add(a, a).Float32s()
		return nil
	})
	if err != nil {
		t.Fatalf("using an evaluated array on another thread: %v", err)
	}
	if !slices.Equal(got, []float32{4, 6}) {
		t.Errorf("a + a on another thread = %v, want [4 6]", got)
	}
}

func TestLazyArrayCannotBeEvaluatedOnAnotherThread(t *testing.T) {
	pin(t)
	a := twoThree(t, false)
	err := onOtherThread(func() error {
		Eval(a)
		return nil
	})
	var mlxErr *Error
	if !errors.As(err, &mlxErr) {
		t.Fatalf("evaluating a lazy array on another thread returned %v, want *Error", err)
	}
	if got := a.Float32s(); !slices.Equal(got, []float32{2, 3}) {
		t.Errorf("back on its own thread the array is %v, want [2 3]", got)
	}
}

// A Go int that does not fit a C int must not reach MLX truncated. Before the
// check these three calls succeeded with a stop of -1, a shape of [2] and a
// shape of [5].

// wantOverflowError reports an error unless err names the offending value.
func wantOverflowError(t *testing.T, err *Error, value int) {
	t.Helper()
	if !strings.Contains(err.Msg, strconv.Itoa(value)) {
		t.Errorf("error %q does not name the value %d", err.Msg, value)
	}
}

func TestSlicePanicsWhenStopOverflowsCInt(t *testing.T) {
	inScope(t, func() {
		a := FromFloat32([]float32{1, 2, 3}, 3)
		err := mustPanic(t, func() { Slice(a, []int{1}, []int{math.MaxInt}) })
		wantOverflowError(t, err, math.MaxInt)
	})
}

func TestZerosPanicsWhenDimensionOverflowsCInt(t *testing.T) {
	inScope(t, func() {
		err := mustPanic(t, func() { Zeros(Float32, 1<<32+2) })
		wantOverflowError(t, err, 1<<32+2)
	})
}

func TestReshapePanicsWhenDimensionOverflowsCInt(t *testing.T) {
	inScope(t, func() {
		a := FromFloat32([]float32{1, 2, 3, 4, 5}, 5)
		err := mustPanic(t, func() { Reshape(a, 1<<32+5) })
		wantOverflowError(t, err, 1<<32+5)
	})
}

func TestCIntPanicsUnlessValueFits(t *testing.T) {
	for _, v := range []int{0, 7, -7, math.MaxInt32, math.MinInt32} {
		if got := int(cInt(v)); got != v {
			t.Errorf("cInt(%d) = %d", v, got)
		}
	}
	for _, v := range []int{math.MaxInt32 + 1, math.MinInt32 - 1, math.MaxInt, math.MinInt} {
		func() {
			defer func() {
				err, ok := recover().(*Error)
				if !ok {
					t.Errorf("cInt(%d) did not panic with *Error", v)
					return
				}
				wantOverflowError(t, err, v)
			}()
			cInt(v)
		}()
	}
}
