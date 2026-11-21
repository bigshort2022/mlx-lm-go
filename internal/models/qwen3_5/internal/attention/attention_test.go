package attention

import (
	"math"
	"slices"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/cache"
	"github.com/bigshort2022/mlx-lm-go/internal/models/modeltest"
	"github.com/bigshort2022/mlx-lm-go/internal/nn"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// The expected values were recorded with mlx-lm 0.32.2's Qwen3NextAttention
// and MLX 0.32.4, using these synthetic float32 weights and inputs. For
// multi-token calls the oracle used explicit create_causal_mask arrays.
const (
	testHidden = 6
	testHeads  = 4
	testKV     = 2
	testDim    = 8
)

func testData(n, salt int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = float32((i*(salt*2+3)+salt*5)%37-18) / 29
	}
	return data
}

func testLayer(t *testing.T, bias bool, gateShift float32) *Layer {
	t.Helper()
	const prefix = "self_attn"
	weights := make(map[string]*mlx.Array)
	for i, p := range []struct {
		name    string
		in, out int
	}{
		{"q_proj", testHidden, 2 * testHeads * testDim},
		{"k_proj", testHidden, testKV * testDim},
		{"v_proj", testHidden, testKV * testDim},
		{"o_proj", testHeads * testDim, testHidden},
	} {
		path := prefix + "." + p.name
		weights[path+".weight"] = mlx.FromFloat32(testData(p.out*p.in, i+1), p.out, p.in)
		if bias {
			values := testData(p.out, i+5)
			if p.name == "q_proj" {
				for j := range values {
					if j%(2*testDim) >= testDim {
						values[j] += gateShift
					}
				}
			}
			weights[path+".bias"] = mlx.FromFloat32(values, p.out)
		}
	}
	for i, name := range []string{"q_norm", "k_norm"} {
		values := testData(testDim, i+9)
		for j := range values {
			values[j] += 1
		}
		weights[prefix+"."+name+".weight"] = mlx.FromFloat32(values, testDim)
	}
	loader := nn.NewLoader(weights)
	layer := Load(loader, prefix, Config{
		HiddenSize: testHidden, Heads: testHeads, KVHeads: testKV, HeadDim: testDim,
		RotaryDims: 4, Theta: 64, Eps: 1e-5, Bias: bias,
	})
	if err := loader.Err(); err != nil {
		t.Fatal(err)
	}
	if unused := loader.Unused(); len(unused) != 0 {
		t.Fatalf("unused parameters: %v", unused)
	}
	return layer
}

func inScope(t *testing.T, f func()) {
	t.Helper()
	modeltest.Pin(t)
	if err := mlx.Scoped(func() error {
		f()
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func wantOutput(t *testing.T, got *mlx.Array, want []float32) {
	t.Helper()
	shape := []int{1, len(want) / testHidden, testHidden}
	if !slices.Equal(got.Shape(), shape) {
		t.Fatalf("shape = %v, want %v", got.Shape(), shape)
	}
	for i, v := range got.Float32s() {
		if diff := math.Abs(float64(v) - float64(want[i])); !(diff <= 1e-5) {
			t.Errorf("output[%d,%d] = %.9g, want %.9g (absolute error %.3g)", i/testHidden, i%testHidden, v, want[i], diff)
			return
		}
	}
}

func TestGroupedPrefill(t *testing.T) {
	inScope(t, func() {
		layer := testLayer(t, true, 0)
		x := mlx.FromFloat32(testData(5*testHidden, 11), 1, 5, testHidden)
		wantOutput(t, layer.Forward(x, nil), []float32{
			-0.244446218, -0.426206589, 0.289737433, -0.147369087, 0.346821368, 0.2773754,
			-0.263778925, -0.424638152, 0.274145156, -0.135263473, 0.329395711, 0.292606771,
			-0.281291425, -0.426552534, 0.260244697, -0.126292437, 0.313498795, 0.303982317,
			-0.431518734, -0.169676647, 0.211062938, 0.116608351, 0.251180172, 0.277715474,
			-0.44235149, -0.117118731, 0.188114971, 0.128598109, 0.230282784, 0.216852352,
		})
	})
}

func TestGateSensitivity(t *testing.T) {
	inScope(t, func() {
		base := testLayer(t, true, 0)
		shifted := testLayer(t, true, 2)
		x := mlx.FromFloat32(testData(testHidden, 11), 1, 1, testHidden)
		// Only gate logits change. With one token, query/key scores cannot
		// affect the result, even if gate rows were confused with query rows.
		wantOutput(t, base.Forward(x, nil), []float32{
			-0.244446218, -0.426206589, 0.289737433, -0.147369087, 0.346821368, 0.2773754,
		})
		wantOutput(t, shifted.Forward(x, nil), []float32{
			-0.2280536, -0.553271532, 0.717356563, -0.0242115855, 0.784160793, 0.638420045,
		})
	})
}

// testContinuation keeps the cache across scopes just as separate prefill
// and decode calls do. The oracle's mask covers both the prefix and suffix.
func testContinuation(t *testing.T, prefixLen int, want []float32) {
	t.Helper()
	inScope(t, func() {
		const total = 5
		layer := testLayer(t, false, 0)
		x := mlx.FromFloat32(testData(total*testHidden, 11), 1, total, testHidden)
		var state cache.KVCache
		defer state.Free()
		if err := mlx.Scoped(func() error {
			prefix := mlx.Slice(x, []int{0, 0, 0}, []int{1, prefixLen, testHidden})
			mlx.Eval(layer.Forward(prefix, &state))
			mlx.Keep(state.Arrays()...)
			return nil
		}); err != nil {
			t.Fatal(err)
		}

		suffix := mlx.Slice(x, []int{0, prefixLen, 0}, []int{1, total, testHidden})
		wantOutput(t, layer.Forward(suffix, &state), want)
		if offset := state.Offset(); offset != total {
			t.Errorf("cache offset = %d, want %d", offset, total)
		}
	})
}

func TestDecodeUsesRotaryOffset(t *testing.T) {
	testContinuation(t, 4, []float32{
		-0.0365872979, 0.148378864, 0.0911934376, 0.174622998, 0.0876459926, 0.12371023,
	})
}

func TestCachedSuffixIsCausal(t *testing.T) {
	// The first suffix token sees the two prefix tokens and itself, not
	// the two tokens following it.
	testContinuation(t, 2, []float32{
		-0.0426426977, -0.264465868, 0.102950394, -0.0894436836, 0.101886258, 0.139788002,
		-0.0975966379, -0.0769909471, 0.0720091388, 0.0797446594, 0.0658065379, 0.185751647,
		-0.0365872793, 0.148378834, 0.0911934525, 0.174622998, 0.0876460224, 0.123710223,
	})
}
