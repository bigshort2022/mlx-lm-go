package mlx

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

func TestDepthwiseConv1D(t *testing.T) {
	inScope(t, func() {
		// Each output channel correlates only with its own input channel.
		x := FromFloat32([]float32{1, 10, 2, 20, 3, 30, 4, 40}, 1, 4, 2)
		w := FromFloat32([]float32{2, -1, -1, 3}, 2, 2, 1)
		wantFloats(t, "depthwise", DepthwiseConv1D(x, w), []int{1, 3, 2}, []float32{0, 50, 1, 70, 2, 90}, 0)
	})
}

func TestGatedDeltaUpdate(t *testing.T) {
	// mlx-lm 0.32.2 gated_delta_kernel (not mlx.core.fast) on MLX 0.32.4.
	// K=128 uses the Qwen3.8 dimensions to exercise native Metal dispatch;
	// its non-BF16-exact gates catch rounding the FP32 decay to model dtype.
	// Sample flattened output and states at strides size/23+1 and size/11+1.
	for _, tc := range []struct {
		keyDim, keyHeads, valueHeads, valueDim int
		gateScale                              float32
		y, prefixState, state                  []float32
	}{
		{
			keyDim: 32, keyHeads: 2, valueHeads: 4, valueDim: 8, gateScale: 1.0 / 32,
			y: []float32{
				-0.001556396484375, 0.0014190673828125, 0.0021209716796875, -0.000659942626953125,
				-0.000240325927734375, -0.00010776519775390625, 4.57763671875e-05, -0.0009765625,
				-0.000354766845703125, -0.00025177001953125, -0.0050048828125, -0.006805419921875,
				-0.0020904541015625, -0.00439453125, 0.0047607421875, 0.0024261474609375,
				-3.910064697265625e-05, 0.001068115234375, 0.0015106201171875, 0.002777099609375,
				0.00396728515625, -0.0020294189453125, 0.00189208984375,
			},
			prefixState: []float32{0.0008440298261, -0.004166848026, 0.009058157913, 0.0002125796746, 0.01369716786, 0.01394133642, 0.004521390889, -0.006869582925, 0.009063829668, 0.003630221589, 0.009646054357},
			state:       []float32{0.006801056210, 0.004925609566, 0.01027257647, -0.005326837767, 0.01181148365, 0.002416144591, -0.001470173011, -0.01952013932, 0.01112813037, 0.01648258977, 0.003606887069},
		},
		{
			keyDim: 128, keyHeads: 16, valueHeads: 48, valueDim: 128, gateScale: 33.0 / 1024,
			y: []float32{
				-0.00011539459228515625, 0.00135040283203125, 0.000316619873046875, -0.00445556640625,
				-0.01019287109375, -0.002716064453125, -0.007232666015625, -0.00145721435546875,
				0.0005950927734375, -0.0004062652587890625, 0.0032196044921875, -0.005950927734375,
				-0.00146484375, 0.0017852783203125, 0.0054931640625, -0.0040283203125,
				-0.008056640625, 0.00165557861328125, 0.006256103515625, -0.005096435546875,
				-0.006439208984375, 0.00118255615234375, 0.0076904296875,
			},
			prefixState: []float32{0.003028962761, -0.01066168770, -0.005883619655, 0.007741998415, 0.008159184828, 0.007204114459, 0.01186764427, -0.002093607094, 3.599331831e-05, -0.02762869000, 0.008429761045},
			state:       []float32{0.01034322381, -0.006323551293, -0.02039811015, 0.007160791662, 0.005857247394, -0.004332656041, 0.01948007569, -0.01643536240, 0.009093482047, -0.01201879606, 0.02892957255},
		},
	} {
		t.Run(fmt.Sprintf("K%d", tc.keyDim), func(t *testing.T) {
			inScope(t, func() {
				from := func(dtype DType, mul, mod, shift int, scale float32, shape ...int) *Array {
					n := 1
					for _, dim := range shape {
						n *= dim
					}
					data := make([]float32, n)
					for i := range data {
						data[i] = float32(i*mul%mod-shift) * scale
					}
					return AsType(FromFloat32(data, shape...), dtype)
				}
				yShape := []int{1, 5, tc.valueHeads, tc.valueDim}
				q := from(BFloat16, 7, 29, 14, 1.0/128, 1, 5, tc.keyHeads, tc.keyDim)
				k := from(BFloat16, 11, 31, 15, 1.0/256, 1, 5, tc.keyHeads, tc.keyDim)
				v := from(BFloat16, 3, 23, 11, 1.0/16, yShape...)
				g := from(Float32, 5, 13, -17, tc.gateScale, 1, 5, tc.valueHeads)
				beta := from(BFloat16, 7, 17, -8, 1.0/32, 1, 5, tc.valueHeads)
				y, state := GatedDeltaUpdate(q, k, v, g, beta, nil)
				stateShape := []int{1, tc.valueHeads, tc.valueDim, tc.keyDim}
				wantDeltaValues(t, "output", y, BFloat16, yShape, tc.y, y.Size()/23+1, 0)
				wantDeltaValues(t, "state", state, Float32, stateShape, tc.state, state.Size()/11+1, 1e-8)

				step := func(start, stop int, state *Array) (*Array, *Array) {
					chunk := func(a *Array) *Array {
						lo, hi := make([]int, a.NDim()), a.Shape()
						lo[1], hi[1] = start, stop
						return Slice(a, lo, hi)
					}
					return GatedDeltaUpdate(chunk(q), chunk(k), chunk(v), chunk(g), chunk(beta), state)
				}
				var prefix, carry *Array
				if err := Scoped(func() error {
					prefix, carry = step(0, 3, nil)
					Keep(prefix, carry)
					return nil
				}); err != nil {
					t.Fatal(err)
				}
				wantDeltaValues(t, "prefix state", carry, Float32, stateShape, tc.prefixState, carry.Size()/11+1, 1e-8)
				suffix, final := step(3, 5, carry)
				carry.Free() // The successor graph now owns the input dependency.
				wantDeltaValues(t, "split output", Concatenate(1, prefix, suffix), BFloat16, yShape, AsType(y, Float32).Float32s(), 1, 0)
				wantDeltaValues(t, "split state", final, Float32, stateShape, state.Float32s(), 1, 1e-8)

				var unkeptOutput, unkeptState *Array
				if err := Scoped(func() error {
					unkeptOutput, unkeptState = step(0, 1, nil)
					return nil
				}); err != nil {
					t.Fatal(err)
				}
				if unkeptOutput.Valid() || unkeptState.Valid() {
					t.Fatal("scope did not release both recurrence results")
				}
			})
		})
	}
}

func TestGatedDeltaUpdateChunkMatchesPython(t *testing.T) {
	inScope(t, func() {
		// mlx-lm 0.32.2 gated_delta_kernel_unpacked, verified against its
		// packed kernel. These indices specifically detect the different
		// reduction order in MLX 0.32.4's chunked primitive.
		const length, hk, hv, dk, dv = 32, 16, 48, 128, 128
		from := func(dtype DType, mul, mod, shift int, scale float32, shape ...int) *Array {
			n := 1
			for _, dim := range shape {
				n *= dim
			}
			values := make([]float32, n)
			for i := range values {
				values[i] = float32(i*mul%mod-shift) * scale
			}
			return AsType(FromFloat32(values, shape...), dtype)
		}
		q := from(BFloat16, 7, 29, 14, 1.0/128, 1, length, hk, dk)
		k := from(BFloat16, 11, 31, 15, 1.0/256, 1, length, hk, dk)
		v := from(BFloat16, 3, 23, 11, 1.0/16, 1, length, hv, dv)
		g := from(Float32, 5, 13, -17, 33.0/1024, 1, length, hv)
		beta := from(BFloat16, 7, 17, -8, 1.0/32, 1, length, hv)
		initial := from(Float32, 13, 37, 18, 1.0/4096, 1, hv, dv, dk)
		y, state := GatedDeltaUpdate(q, k, v, g, beta, initial)
		wantIndices := func(name string, a *Array, dtype DType, shape []int, indices []int, want []float32) {
			t.Helper()
			if a.DType() != dtype || !slices.Equal(a.Shape(), shape) {
				t.Fatalf("%s: got %v %v, want %v %v", name, a.DType(), a.Shape(), dtype, shape)
			}
			got := AsType(a, Float32).Float32s()
			for i, idx := range indices {
				if got[idx] != want[i] {
					t.Errorf("%s[%d]: got %.10g, want %.10g", name, idx, got[idx], want[i])
				}
			}
		}
		wantIndices("output", y, BFloat16, []int{1, length, hv, dv},
			[]int{9855, 15179, 20020, 78257, 79806, 183397},
			[]float32{-0.000957489013671875, 5.125999450683594e-05, 1.043081283569336e-05, -0.001953125, 0.003631591796875, 0.00384521484375})
		wantIndices("state", state, Float32, []int{1, hv, dv, dk},
			[]int{199441, 199472, 200465, 209311},
			[]float32{-0.0265731792896986, -0.026573218405246735, 0.027947695925831795, -0.026537491008639336})
	})
}

func wantDeltaValues(t *testing.T, name string, a *Array, dtype DType, shape []int, want []float32, stride int, tol float64) {
	t.Helper()
	if a.DType() != dtype || !slices.Equal(a.Shape(), shape) {
		t.Fatalf("%s: got %v %v, want %v %v", name, a.DType(), a.Shape(), dtype, shape)
	}
	got := AsType(a, Float32).Float32s()
	var maxDiff float64
	for i, v := range want {
		delta := math.Abs(float64(got[i*stride]) - float64(v))
		if !(delta <= tol) {
			t.Fatalf("%s[%d]: got %g, want %g (tolerance %g)", name, i*stride, got[i*stride], v, tol)
		}
		maxDiff = max(maxDiff, delta)
	}
	t.Logf("%s: max abs diff %.3g", name, maxDiff)
}
