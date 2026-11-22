package qwen2

import (
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/models"
	"github.com/bigshort2022/mlx-lm-go/internal/models/modeltest"
	"github.com/bigshort2022/mlx-lm-go/internal/nn"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// layerTolerance bounds the difference from the Python activations: Go links
// a newer MLX than the one that made the fixtures.
const layerTolerance = 2e-2

// reference returns the arguments of the reference model and a loader over
// its weights. The test must be pinned.
func reference(t *testing.T) (*modelArgs, *nn.Loader) {
	t.Helper()
	cfg, err := models.LoadConfig(modeltest.Dir(t))
	if err != nil {
		t.Fatal(err)
	}
	args, err := parseArgs(cfg.Raw)
	if err != nil {
		t.Fatal(err)
	}
	return args, nn.NewLoader(modeltest.Weights(t))
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

func TestAttentionMatchesGolden(t *testing.T) {
	inScope(t, func() {
		args, loader := reference(t)
		golden := modeltest.Layers(t)
		attn := newAttention(loader, "model.layers.0.self_attn", args)
		if err := loader.Err(); err != nil {
			t.Fatal(err)
		}
		wantClose(t, "l0_attn", attn.forward(half(golden["l0_norm1"]), "causal", nil), golden["l0_attn"])
	})
}

func TestMLPMatchesGolden(t *testing.T) {
	inScope(t, func() {
		args, loader := reference(t)
		golden := modeltest.Layers(t)
		mlp := newMLP(loader, "model.layers.0.mlp", args)
		if err := loader.Err(); err != nil {
			t.Fatal(err)
		}
		wantClose(t, "l0_mlp", mlp.forward(half(golden["l0_norm2"])), golden["l0_mlp"])
	})
}

func TestBlockMatchesGolden(t *testing.T) {
	inScope(t, func() {
		args, loader := reference(t)
		golden := modeltest.Layers(t)
		block := newTransformerBlock(loader, "model.layers.0", args)
		if err := loader.Err(); err != nil {
			t.Fatal(err)
		}
		wantClose(t, "l0_out", block.forward(half(golden["embed"]), "causal", nil), golden["l0_out"])
	})
}
