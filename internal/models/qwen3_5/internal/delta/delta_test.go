package delta

import (
	"math"
	"slices"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/models/modeltest"
	"github.com/bigshort2022/mlx-lm-go/internal/nn"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

func TestForwardMatchesPythonAndStreaming(t *testing.T) {
	// mlx-lm 0.32.2 GatedDeltaNet.eval() with MLX 0.32.4. Weights are the
	// modular sequence in tinyArray with the scale and dtype of each tensor
	// below; ArraysCache(2) captures the same tail and state as Go.
	for _, tc := range []struct {
		name             string
		keyDim           int
		dtype, aLogDType mlx.DType
		y, tail, state   []float32
		yTol             float64
	}{
		{
			name: "K32BF16", keyDim: 32, dtype: mlx.BFloat16, aLogDType: mlx.Float32,
			y: []float32{
				-0.0274658203125, -0.038330078125, 0.10400390625, -0.026123046875, 0.025146484375, -0.018798828125, 0.0341796875, -0.023681640625,
				0.03857421875, -0.11328125, 0.2060546875, -0.1513671875, 0.1611328125, -0.0966796875, -0.208984375, 0.09521484375,
				0.25390625, -0.3203125, 0.154296875, -0.046142578125, -0.0123291015625, 0.248046875, -0.3671875, 0.150390625,
				0.02685546875, -0.193359375, 0.177734375, -0.1318359375, 0.07666015625, 0.1240234375, -0.244140625, 0.1982421875,
				0.15234375, -0.11328125, -0.1923828125, 0.0654296875, -0.13671875, 0.328125, -0.236328125, -0.13671875,
			},
			tail:  []float32{0.181640625, 0.345703125, 0.20703125, -0.6796875, -0.12109375, 0.193359375, -0.5234375, 0.18359375},
			state: []float32{0.004175197333, 0.008934916928, -0.001301823184, -0.008238994516, 0.009295358323, -0.01339059323, 0.0001978653891, 0.002016084502, -0.01084308885, 0.0001174496647, 0.003474995028},
		},
		{
			name: "K128BF16", keyDim: 128, dtype: mlx.BFloat16, aLogDType: mlx.BFloat16,
			y: []float32{
				-0.06494140625, 0.072265625, 0.068359375, 0.04638671875, 0.03076171875, -0.08544921875, -0.0196533203125, -0.0869140625,
				-0.1181640625, 0.1328125, -0.09375, -0.0595703125, 0.138671875, -0.1513671875, 0.208984375, -0.1171875,
				-0.1640625, 0.2060546875, -0.02294921875, -0.049560546875, 0.06640625, -0.1630859375, 0.2451171875, -0.17578125,
				0.008544921875, 0.15234375, 0.01556396484375, 0.00628662109375, 0.01416015625, -0.1328125, 0.000255584716796875, 0.006988525390625,
				0.09814453125, -0.1083984375, 0.1103515625, -0.014404296875, 0.031494140625, -0.2294921875, -0.034423828125, 0.236328125,
			},
			tail:  []float32{0.181640625, 0.091796875, 0.18359375, -0.2109375, -0.009765625, 0.373046875, -0.34375, -0.578125},
			state: []float32{-0.0003271229507, -0.0008374929894, 0.001241478487, -0.004487332422, -0.002141284291, 0.001596398884, -0.004209243692, 0.002327728085, 0.0001134063568, -0.004700961523, 0.001034466666},
		},
		{
			name: "K32FP32", keyDim: 32, dtype: mlx.Float32, aLogDType: mlx.Float32, yTol: 2e-7,
			y: []float32{
				-0.02651685849, -0.03872449324, 0.1030483246, -0.02591361851, 0.02451523952, -0.01820730045, 0.03298820555, -0.02235620096,
				0.03891507909, -0.1140134111, 0.2059761584, -0.1517035514, 0.1614609063, -0.09638651460, -0.2098995149, 0.09551327676,
				0.2537283897, -0.3199416399, 0.1533350796, -0.04691406712, -0.01103722863, 0.2480896413, -0.3654384315, 0.1490377784,
				0.02636459470, -0.1934349239, 0.1777195781, -0.1320181191, 0.07741294056, 0.1233700737, -0.2441954315, 0.1975922734,
				0.1525503248, -0.1137377769, -0.1918479800, 0.06519775838, -0.1363962740, 0.3272128701, -0.2365749776, -0.1366971135,
			},
			tail:  []float32{0.181640625, 0.345703125, 0.20703125, -0.677734375, -0.12109375, 0.193359375, -0.5234375, 0.18359375},
			state: []float32{0.004156216048, 0.008941081353, -0.001321811927, -0.008263085969, 0.009213803336, -0.01345112361, 0.0001993554470, 0.002028887393, -0.01105870400, 0.0001158460873, 0.003476379206},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			modeltest.Pin(t)
			inScope(t, func() {
				cfg := Config{HiddenSize: 8, KeyHeads: 2, ValueHeads: 4, KeyDim: tc.keyDim, ValueDim: 8, KernelSize: 4, Eps: 1e-6}
				layer, raw := tinyLayer(t, cfg, tc.dtype, tc.aLogDType)
				x := tinyArray(tc.dtype, 10, 1.0/16, 0, 1, 5, cfg.HiddenSize)
				var whole *mlx.Array
				inScope(t, func() {
					whole = layer.Forward(x, nil)
					mlx.Keep(whole)
				})
				yShape := []int{1, 5, cfg.HiddenSize}
				wantValues(t, "whole output", whole, tc.dtype, yShape, tc.y, 1, tc.yTol)
				wantY := mlx.AsType(whole, mlx.Float32).Float32s()
				channels := 2*cfg.KeyHeads*cfg.KeyDim + cfg.ValueHeads*cfg.ValueDim
				tailShape := []int{1, cfg.KernelSize - 1, channels}
				stateShape := []int{1, cfg.ValueHeads, cfg.ValueDim, cfg.KeyDim}
				var wantTail, wantState []float32
				for _, chunks := range [][]int{{5}, {1, 2, 2}, {2, 1, 1, 1}} {
					var cache State
					var results []*mlx.Array
					start := 0
					for _, length := range chunks {
						stop := start + length
						inScope(t, func() {
							previous := cache.Arrays()
							chunk := mlx.Slice(x, []int{0, start, 0}, []int{1, stop, cfg.HiddenSize})
							out := layer.Forward(chunk, &cache)
							for _, a := range previous {
								if a.Valid() {
									t.Fatal("replaced cache handle still live")
								}
							}
							results = append(results, out)
							mlx.Keep(out)
							mlx.Keep(cache.Arrays()...)
						})
						start = stop
					}
					buffers := cache.Arrays()
					if len(buffers) != 2 {
						t.Fatalf("cache has %d arrays, want tail and recurrence", len(buffers))
					}
					tail, recurrent := buffers[0], buffers[1]
					wantValues(t, "streamed output", mlx.Concatenate(1, results...), tc.dtype, yShape, wantY, 1, tc.yTol)
					if wantState == nil {
						wantValues(t, "tail oracle", tail, tc.dtype, tailShape, tc.tail, tail.Size()/8+1, 0)
						wantValues(t, "state oracle", recurrent, mlx.Float32, stateShape, tc.state, recurrent.Size()/11+1, 1e-8)
						wantTail, wantState = mlx.AsType(tail, mlx.Float32).Float32s(), recurrent.Float32s()
					} else {
						wantValues(t, "streamed tail", tail, tc.dtype, tailShape, wantTail, 1, 0)
						wantValues(t, "streamed state", recurrent, mlx.Float32, stateShape, wantState, 1, 1e-8)
					}
					cache.Free()
					cache.Free()
					if len(cache.Arrays()) != 0 || tail.Valid() || recurrent.Valid() {
						t.Fatal("Free did not release cache handles and reset cache")
					}
				}
				for _, a := range raw {
					if !a.Valid() {
						t.Fatal("layer freed borrowed weight")
					}
				}
			})
		})
	}
}

func tinyLayer(t *testing.T, cfg Config, dtype, aLogDType mlx.DType) (*Layer, []*mlx.Array) {
	t.Helper()
	h := cfg.HiddenSize
	v := cfg.ValueHeads * cfg.ValueDim
	c := cfg.KeyHeads*cfg.KeyDim*2 + v
	const prefix = "block.linear_attn"
	l := nn.NewLoader(map[string]*mlx.Array{
		prefix + ".in_proj_qkv.weight": tinyArray(dtype, 1, 1.0/32, 0, c, h),
		prefix + ".in_proj_z.weight":   tinyArray(dtype, 2, 1.0/32, 0, v, h),
		prefix + ".in_proj_b.weight":   tinyArray(dtype, 3, 1.0/16, 0, cfg.ValueHeads, h),
		prefix + ".in_proj_a.weight":   tinyArray(dtype, 4, 1.0/16, 0, cfg.ValueHeads, h),
		prefix + ".norm.weight":        tinyArray(dtype, 5, 1.0/64, 1, cfg.ValueDim),
		prefix + ".out_proj.weight":    tinyArray(dtype, 6, 1.0/64, 0, h, v),
	})
	conv := tinyArray(dtype, 7, 1.0/32, 0, c, cfg.KernelSize, 1)
	aLog := tinyArray(aLogDType, 8, 1.0/16, 0, cfg.ValueHeads)
	dtBias := tinyArray(dtype, 9, 1.0/16, 0, cfg.ValueHeads)
	layer := Load(l, prefix, cfg, conv, aLog, dtBias)
	if err := l.Err(); err != nil {
		t.Fatal(err)
	}
	if unused := l.Unused(); len(unused) != 0 {
		t.Fatalf("unloaded parameters: %v", unused)
	}
	return layer, []*mlx.Array{conv, aLog, dtBias}
}

func tinyArray(dtype mlx.DType, salt int, scale, offset float32, shape ...int) *mlx.Array {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32((i*(salt*2+1)+salt)%31-15)*scale + offset
	}
	return mlx.AsType(mlx.FromFloat32(data, shape...), dtype)
}

func inScope(t *testing.T, f func()) {
	t.Helper()
	if err := mlx.Scoped(func() error {
		f()
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func wantValues(t *testing.T, name string, a *mlx.Array, dtype mlx.DType, shape []int, want []float32, stride int, tol float64) {
	t.Helper()
	if !a.Valid() || a.DType() != dtype || !slices.Equal(a.Shape(), shape) {
		t.Fatalf("%s: wanted a live %v array of shape %v, got %v", name, dtype, shape, a)
	}
	got := mlx.AsType(a, mlx.Float32).Float32s()
	var maxDiff float64
	for i, v := range want {
		diff := math.Abs(float64(got[i*stride]) - float64(v))
		if !(diff <= tol) {
			t.Fatalf("%s[%d]: got %g, want %g (tolerance %g)", name, i*stride, got[i*stride], v, tol)
		}
		maxDiff = max(maxDiff, diff)
	}
	t.Logf("%s: max abs diff %.3g", name, maxDiff)
}
