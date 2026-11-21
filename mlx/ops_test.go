package mlx

import "testing"

// Expected values in this file were computed with the Python reference
// (mlx.core) on the same inputs.

// exact is the tolerance for results that float32 represents exactly; ulps
// allows for the last-bit differences between CPU and GPU math functions.
const (
	exact = 0
	ulps  = 1e-5
)

func vec123() *Array { return FromFloat32([]float32{1, 2, 3}, 3) }
func vec456() *Array { return FromFloat32([]float32{4, 5, 6}, 3) }
func vec315() *Array { return FromFloat32([]float32{3, 1, 5}, 3) }

func TestAdd(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "[1,2,3] + [4,5,6]", Add(vec123(), vec456()), []int{3}, []float32{5, 7, 9}, exact)
		wantFloats(t, "[1,2,3] + 10", Add(vec123(), Float(10)), []int{3}, []float32{11, 12, 13}, exact)
	})
}

func TestSubtract(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "[1,2,3] - [4,5,6]", Subtract(vec123(), vec456()), []int{3}, []float32{-3, -3, -3}, exact)
	})
}

func TestMultiply(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "[1,2,3] * [4,5,6]", Multiply(vec123(), vec456()), []int{3}, []float32{4, 10, 18}, exact)
		wantFloats(t, "[1,2,3] * 2", Multiply(vec123(), Float(2)), []int{3}, []float32{2, 4, 6}, exact)
	})
}

func TestDivide(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "[1,2,3] / [4,5,6]", Divide(vec123(), vec456()), []int{3}, []float32{0.25, 0.4, 0.5}, ulps)
	})
}

func TestNegative(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "-[1,2,3]", Negative(vec123()), []int{3}, []float32{-1, -2, -3}, exact)
	})
}

func TestExp(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "exp([1,2,3])", Exp(vec123()), []int{3}, []float32{2.7182820, 7.3890562, 20.085537}, ulps)
	})
}

func TestLog(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "log([1,2,3])", Log(vec123()), []int{3}, []float32{0, 0.69314718, 1.0986123}, ulps)
	})
}

func TestSigmoid(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "sigmoid([1,2,3])", Sigmoid(vec123()), []int{3}, []float32{0.73105860, 0.88079709, 0.95257413}, ulps)
	})
}

func TestMaximum(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "maximum([1,2,3], [3,1,5])", Maximum(vec123(), vec315()), []int{3}, []float32{3, 2, 5}, exact)
	})
}

func TestMinimum(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "minimum([1,2,3], [3,1,5])", Minimum(vec123(), vec315()), []int{3}, []float32{1, 1, 3}, exact)
	})
}

func TestGreater(t *testing.T) {
	inScope(t, func() {
		got := Greater(vec123(), vec315())
		if got.DType() != Bool {
			t.Errorf("Greater returned %v, want bool", got.DType())
		}
		wantInts(t, "[1,2,3] > [3,1,5]", got, []int{3}, []int32{0, 1, 0})
	})
}

func TestLess(t *testing.T) {
	inScope(t, func() {
		got := Less(vec123(), vec315())
		if got.DType() != Bool {
			t.Errorf("Less returned %v, want bool", got.DType())
		}
		wantInts(t, "[1,2,3] < [3,1,5]", got, []int{3}, []int32{1, 0, 1})
	})
}

func TestWhere(t *testing.T) {
	inScope(t, func() {
		cond := Greater(vec123(), vec315())
		wantFloats(t, "where(a > c, a, b)", Where(cond, vec123(), vec456()), []int{3}, []float32{4, 2, 6}, exact)
	})
}

func TestMatmul(t *testing.T) {
	inScope(t, func() {
		a := FromFloat32([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		b := FromFloat32([]float32{7, 8, 9, 10, 11, 12}, 3, 2)
		wantFloats(t, "[2,3] x [3,2]", Matmul(a, b), []int{2, 2}, []float32{58, 64, 139, 154}, exact)
	})
}

func TestAsType(t *testing.T) {
	inScope(t, func() {
		got := AsType(FromFloat32([]float32{1.7, -1.7, 2.5}, 3), Int32)
		if got.DType() != Int32 {
			t.Errorf("AsType(..., Int32) returned %v", got.DType())
		}
		wantInts(t, "int32([1.7,-1.7,2.5])", got, []int{3}, []int32{1, -1, 2})
	})
}
