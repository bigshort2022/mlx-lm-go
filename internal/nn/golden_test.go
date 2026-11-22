package nn

import (
	"errors"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/models/modeltest"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// layerTolerance bounds the difference from the Python activations: Go links
// a newer MLX than the one that made the fixtures.
const layerTolerance = 2e-2

// The dimensions of the reference model.
const (
	hiddenSize = 1536
	vocabSize  = 151936
)

// referenceLoader returns a Loader over the weights of the reference model.
func referenceLoader(t *testing.T) *Loader {
	t.Helper()
	return NewLoader(modeltest.Weights(t))
}

// half converts a golden activation back to the float16 the model computes in.
func half(a *mlx.Array) *mlx.Array { return mlx.AsType(a, mlx.Float16) }

// inScope pins the test and runs fn inside a scope that must not fail.
func inScope(t *testing.T, fn func()) {
	t.Helper()
	modeltest.Pin(t)
	err := mlx.Scoped(func() error {
		fn()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// wantClose reports the largest difference between got and want and fails
// the test unless it is below layerTolerance.
func wantClose(t *testing.T, name string, got, want *mlx.Array) {
	t.Helper()
	diff := modeltest.MaxAbsDiff(t, got, want)
	t.Logf("%s: max abs diff %.3g", name, diff)
	if !(diff < layerTolerance) {
		t.Errorf("%s differs from the golden by %g, want below %g", name, diff, layerTolerance)
	}
}

func TestQuantizedEmbeddingMatchesGolden(t *testing.T) {
	inScope(t, func() {
		loader := referenceLoader(t)
		golden := modeltest.Layers(t)
		embedding := loader.Embedding("model.embed_tokens", vocabSize, hiddenSize)
		if err := loader.Err(); err != nil {
			t.Fatal(err)
		}
		got := embedding.Forward(golden["input_ids"])
		if got.DType() != mlx.Float16 {
			t.Errorf("embedding is %v, want float16 like the weights", got.DType())
		}
		wantClose(t, "embed", got, golden["embed"])
	})
}

func TestRMSNormMatchesGolden(t *testing.T) {
	inScope(t, func() {
		loader := referenceLoader(t)
		golden := modeltest.Layers(t)
		norm := loader.RMSNorm("model.layers.0.input_layernorm", hiddenSize, 1e-6)
		if err := loader.Err(); err != nil {
			t.Fatal(err)
		}
		wantClose(t, "l0_norm1", norm.Forward(half(golden["embed"])), golden["l0_norm1"])
	})
}

// dequantized returns a dense copy of the quantized layer at path of the
// reference model, as tensors for a Loader.
func dequantized(weights map[string]*mlx.Array, path string) map[string]*mlx.Array {
	dense := map[string]*mlx.Array{
		path + ".weight": mlx.Dequantize(weights[path+".weight"], weights[path+".scales"], weights[path+".biases"], 64, 4, "affine"),
	}
	if bias, ok := weights[path+".bias"]; ok {
		dense[path+".bias"] = bias
	}
	return dense
}

// wantSame fails the test unless got equals want up to the rounding of
// float16, which has 11 bits of precision: the two may differ by a few units
// in the last place of the largest value.
func wantSame(t *testing.T, name string, got, want *mlx.Array) {
	t.Helper()
	diff := modeltest.MaxAbsDiff(t, got, want)
	largest := modeltest.MaxAbsDiff(t, want, mlx.Zeros(mlx.Float32, want.Shape()...))
	t.Logf("%s: max abs diff %.3g, largest value %.3g", name, diff, largest)
	if !(diff <= 2e-3*largest) {
		t.Errorf("%s differs by %g, want at most %g", name, diff, 2e-3*largest)
	}
}

// The quantized layers must compute what the dense layers compute on the
// dequantized weights. The golden activations of the attention and the MLP
// check them against Python (package qwen2).
func TestQuantizedLinearEqualsDenseOnDequantizedWeights(t *testing.T) {
	for _, tt := range []struct {
		path string
		bias bool
	}{
		{"model.layers.0.self_attn.q_proj", true},
		{"model.layers.0.self_attn.o_proj", false},
	} {
		t.Run(tt.path, func(t *testing.T) {
			inScope(t, func() {
				weights := modeltest.Weights(t)
				golden := modeltest.Layers(t)
				loader := NewLoader(weights)
				quantized := loader.Linear(tt.path, hiddenSize, hiddenSize, tt.bias)
				denseLoader := NewLoader(dequantized(weights, tt.path))
				dense := denseLoader.Linear(tt.path, hiddenSize, hiddenSize, tt.bias)
				if err := errors.Join(loader.Err(), denseLoader.Err()); err != nil {
					t.Fatal(err)
				}
				x := half(golden["l0_norm1"])
				wantSame(t, tt.path, quantized.Forward(x), dense.Forward(x))
			})
		})
	}
}

func TestQuantizedEmbeddingAsLinearEqualsDenseOnDequantizedWeights(t *testing.T) {
	inScope(t, func() {
		weights := modeltest.Weights(t)
		golden := modeltest.Layers(t)
		const path = "model.embed_tokens"
		quantized := NewLoader(weights).Embedding(path, vocabSize, hiddenSize)
		dense := NewLoader(dequantized(weights, path)).Embedding(path, vocabSize, hiddenSize)
		x := half(golden["final_norm"])
		wantSame(t, "as_linear", quantized.AsLinear(x), dense.AsLinear(x))
	})
}
