package mlx

import (
	"math"
	"slices"
	"testing"
)

var negInf = float32(math.Inf(-1))

// sampleTwice seeds the generator and draws two samples from the same logits.
func sampleTwice(seed uint64, logits *Array) [2][]int32 {
	Seed(seed)
	first := AsType(Categorical(logits, -1), Int32).Int32s()
	second := AsType(Categorical(logits, -1), Int32).Int32s()
	return [2][]int32{first, second}
}

func TestCategoricalRepeatsAfterTheSameSeed(t *testing.T) {
	inScope(t, func() {
		logits := FromFloat32([]float32{0.1, 0.2, 0.3, 0.4, 1, 0, -1, 2}, 2, 4)
		before := sampleTwice(7, logits)
		after := sampleTwice(7, logits)
		if !slices.Equal(before[0], after[0]) || !slices.Equal(before[1], after[1]) {
			t.Errorf("samples after Seed(7) were %v, then %v after a second Seed(7); want the same pair", before, after)
		}
		// Python: mx.random.seed(7), then mx.random.categorical(logits) twice.
		want := [2][]int32{{3, 3}, {0, 1}}
		if !slices.Equal(before[0], want[0]) || !slices.Equal(before[1], want[1]) {
			t.Errorf("samples after Seed(7) were %v, want %v as in Python", before, want)
		}
	})
}

func TestCategoricalReturnsUint32IndicesWithoutTheAxis(t *testing.T) {
	inScope(t, func() {
		got := Categorical(Zeros(Float32, 2, 4), -1)
		if got.DType() != Uint32 || !slices.Equal(got.Shape(), []int{2}) {
			t.Errorf("Categorical([2,4] logits, axis -1) is %v %v, want uint32 [2]", got.DType(), got.Shape())
		}
	})
}

func TestCategoricalAlwaysPicksTheOnlyFiniteLogit(t *testing.T) {
	inScope(t, func() {
		rows := FromFloat32([]float32{
			negInf, negInf, 0, negInf,
			0, negInf, negInf, negInf,
		}, 2, 4)
		columns := FromFloat32([]float32{
			negInf, 0,
			0, negInf,
		}, 2, 2)
		for range 20 {
			wantInts(t, "categorical(axis -1)", Categorical(rows, -1), []int{2}, []int32{2, 0})
			wantInts(t, "categorical(axis 0)", Categorical(columns, 0), []int{2}, []int32{1, 0})
		}
	})
}
