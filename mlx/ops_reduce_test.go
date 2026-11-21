package mlx

import (
	"math"
	"testing"
)

// Expected values in this file were computed with the Python reference
// (mlx.core) on the same inputs.

// twoByThree is [[1,5,2],[7,3,4]].
func twoByThree() *Array { return FromFloat32([]float32{1, 5, 2, 7, 3, 4}, 2, 3) }

func TestArgmax(t *testing.T) {
	inScope(t, func() {
		got := Argmax(twoByThree(), 1, false)
		if got.DType() != Uint32 {
			t.Errorf("Argmax returned %v, want uint32 like Python", got.DType())
		}
		wantInts(t, "argmax(axis 1)", got, []int{2}, []int32{1, 0})
		wantInts(t, "argmax(axis 1, keepdims)", Argmax(twoByThree(), 1, true), []int{2, 1}, []int32{1, 0})
		wantInts(t, "argmax(axis 0)", Argmax(twoByThree(), 0, false), []int{3}, []int32{1, 0, 1})
	})
}

func TestSoftmax(t *testing.T) {
	inScope(t, func() {
		rows := Softmax(twoByThree(), -1)
		wantFloats(t, "softmax(axis -1)", rows, []int{2, 3}, []float32{
			0.017147828, 0.93623954, 0.046612628, 0.93623954, 0.017147828, 0.046612628,
		}, 1e-6)
		values := rows.Float32s()
		for row := range 2 {
			sum := float64(values[3*row]) + float64(values[3*row+1]) + float64(values[3*row+2])
			if math.Abs(sum-1) > 1e-6 {
				t.Errorf("softmax row %d sums to %v, want 1", row, sum)
			}
		}
		wantFloats(t, "softmax(axis 0)", Softmax(twoByThree(), 0), []int{2, 3}, []float32{
			0.0024726233, 0.88079703, 0.11920293, 0.99752742, 0.11920293, 0.88079703,
		}, 1e-6)
	})
}

func TestLogSumExp(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "logsumexp(axis -1)", LogSumExp(twoByThree(), -1, false), []int{2},
			[]float32{5.0658841, 7.0658841}, ulps)
		wantFloats(t, "logsumexp(axis 0, keepdims)", LogSumExp(twoByThree(), 0, true), []int{1, 3},
			[]float32{7.0024757, 5.1269279, 4.1269279}, ulps)
	})
}

func TestCumsum(t *testing.T) {
	inScope(t, func() {
		wantFloats(t, "cumsum(axis 1)", Cumsum(twoByThree(), 1), []int{2, 3}, []float32{1, 6, 8, 7, 10, 14}, exact)
		wantFloats(t, "cumsum(axis 0)", Cumsum(twoByThree(), 0), []int{2, 3}, []float32{1, 5, 2, 8, 8, 6}, exact)
	})
}

func TestSort(t *testing.T) {
	inScope(t, func() {
		rows := FromFloat32([]float32{3, 1, 2, 9, 7, 8}, 2, 3)
		wantFloats(t, "sort(axis 1)", Sort(rows, 1), []int{2, 3}, []float32{1, 2, 3, 7, 8, 9}, exact)
		columns := FromFloat32([]float32{3, 1, 8, 2, 7, 5}, 2, 3)
		wantFloats(t, "sort(axis 0)", Sort(columns, 0), []int{2, 3}, []float32{2, 1, 5, 3, 7, 8}, exact)
	})
}

func TestArgsort(t *testing.T) {
	inScope(t, func() {
		rows := FromFloat32([]float32{3, 1, 2, 9, 7, 8}, 2, 3)
		wantInts(t, "argsort(axis 1)", Argsort(rows, 1), []int{2, 3}, []int32{1, 2, 0, 1, 2, 0})
	})
}

func TestArgPartitionPutsKthElementInSortedPosition(t *testing.T) {
	inScope(t, func() {
		const kth = 2
		values := FromFloat32([]float32{5, 1, 4, 2, 3, 0}, 6)
		indices := ArgPartition(values, kth, 0)
		if got := indices.Shape(); len(got) != 1 || got[0] != 6 {
			t.Fatalf("ArgPartition returned shape %v, want [6]", got)
		}
		// Python: values[mx.argpartition(values, kth=2)][2] == 2.0.
		partitioned := Take(values, indices, 0).Float32s()
		if partitioned[kth] != 2 {
			t.Errorf("element %d of the partitioned values %v is %v, want 2, its value in sorted order", kth, partitioned, partitioned[kth])
		}
		for i, v := range partitioned {
			if (i < kth && v > partitioned[kth]) || (i > kth && v < partitioned[kth]) {
				t.Errorf("partitioned values %v are not split around element %d", partitioned, kth)
			}
		}
	})
}
