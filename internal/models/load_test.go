package models_test

import (
	"context"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/models"
	"github.com/bigshort2022/mlx-lm-go/internal/models/modeltest"
	_ "github.com/bigshort2022/mlx-lm-go/internal/models/qwen2"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

func TestLoadRejectsUnknownModelType(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"nope"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	_, _, err := models.Load(dir)
	if err == nil {
		t.Fatal("Load succeeded for model_type nope")
	}
	for _, want := range []string{"nope", "qwen2"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// onThread runs fn on the inference thread of the package's tests.
func onThread(t *testing.T, fn func() error) {
	t.Helper()
	if err := thread.Do(context.Background(), fn); err != nil {
		t.Fatal(err)
	}
}

// openFiles returns the number of files the process has open. A tensor that
// was loaded but never evaluated keeps its file open until it is freed.
func openFiles(t *testing.T) int {
	t.Helper()
	entries, err := os.ReadDir("/dev/fd")
	if err != nil {
		t.Fatal(err)
	}
	return len(entries)
}

// loadTiny loads a model directory on the inference thread and reports the
// error of Load. It fails the test if a failed Load leaves arrays behind,
// evaluated or not.
func loadTiny(t *testing.T, config map[string]any, tensors map[string]modeltest.Tensor) error {
	t.Helper()
	dir := modeltest.WriteModel(t, config, tensors)
	var loadErr error
	onThread(t, func() error {
		mlx.ClearCache()
		memory, files := mlx.ActiveMemory(), openFiles(t)
		_, _, loadErr = models.Load(dir)
		if loadErr == nil {
			return nil
		}
		mlx.ClearCache()
		if after := mlx.ActiveMemory(); after != memory {
			t.Errorf("the failed Load left %d bytes of arrays behind", int64(after)-int64(memory))
		}
		if after := openFiles(t); after != files {
			t.Errorf("the failed Load left %d files open", after-files)
		}
		return nil
	})
	return loadErr
}

func TestLoadTinyModels(t *testing.T) {
	for name, model := range map[string]struct {
		config  map[string]any
		tensors map[string]modeltest.Tensor
	}{
		"dense":     {modeltest.TinyConfig(), modeltest.TinyDense()},
		"quantized": {modeltest.TinyQuantizedConfig(), modeltest.TinyQuantized()},
	} {
		t.Run(name, func(t *testing.T) {
			dir := modeltest.WriteModel(t, model.config, model.tensors)
			onThread(t, func() error {
				model, cfg, err := models.Load(dir)
				if err != nil {
					return err
				}
				defer model.(interface{ Free() }).Free()
				if !slices.Equal(cfg.EOSTokenIDs, []int32{modeltest.TinyEOS}) {
					t.Errorf("EOSTokenIDs = %v, want [%d]", cfg.EOSTokenIDs, modeltest.TinyEOS)
				}
				return mlx.Scoped(func() error {
					logits := model.Forward(mlx.FromInt32([]int32{1, 2, 3}, 1, 3), nil)
					if got, want := logits.Shape(), []int{1, 3, modeltest.TinyVocabSize}; !slices.Equal(got, want) {
						t.Errorf("logits have shape %v, want %v", got, want)
					}
					for _, v := range mlx.AsType(logits, mlx.Float32).Float32s() {
						if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
							t.Errorf("the logits hold %v", v)
							break
						}
					}
					return nil
				})
			})
		})
	}
}

// A tensor whose shape does not fit config.json must fail the load, as the
// reference does with "Expected shape ... for parameter ...": naming the
// tensor, the shape it has and the shape it should have. Loaded as it is,
// such a model answers with wrong tokens or with an MLX error that names no
// tensor.
func TestLoadRejectsTensorsOfTheWrongShape(t *testing.T) {
	const layer = "model.layers.0."
	type change struct {
		config  map[string]any              // entries to set in config.json
		tensors map[string]modeltest.Tensor // tensors to replace
		remove  []string                    // tensors to leave out
	}
	tests := []struct {
		name      string
		quantized bool
		change    change
		want      []string // what the error must mention
	}{
		{"bias of one element", false,
			change{tensors: map[string]modeltest.Tensor{layer + "self_attn.q_proj.bias": modeltest.Zeros("F32", 1)}},
			[]string{layer + "self_attn.q_proj.bias", "[1]", "[64]"}},
		{"gate_proj with too few rows", false,
			change{tensors: map[string]modeltest.Tensor{layer + "mlp.gate_proj.weight": modeltest.Zeros("F32", 64, 64)}},
			[]string{layer + "mlp.gate_proj.weight", "[64 64]", "[128 64]"}},
		{"hidden_size that is not the one of the weights", false,
			change{config: map[string]any{"hidden_size": 32}},
			[]string{"model.embed_tokens.weight", "[32 64]", "[32 32]"}},
		{"embedding with fewer rows than vocab_size", false,
			change{tensors: map[string]modeltest.Tensor{"model.embed_tokens.weight": modeltest.Zeros("F32", 16, 64)}},
			[]string{"model.embed_tokens.weight", "[16 64]", "[32 64]"}},
		{"vocab_size larger than the embedding", false,
			change{config: map[string]any{"vocab_size": 100}},
			[]string{"model.embed_tokens.weight", "[32 64]", "[100 64]"}},
		{"embedding of the wrong width", false,
			change{tensors: map[string]modeltest.Tensor{"model.embed_tokens.weight": modeltest.Zeros("F32", 32, 48)}},
			[]string{"model.embed_tokens.weight", "[32 48]", "[32 64]"}},
		{"q_proj with too few columns", false,
			change{tensors: map[string]modeltest.Tensor{layer + "self_attn.q_proj.weight": modeltest.Zeros("F32", 64, 32)}},
			[]string{layer + "self_attn.q_proj.weight", "[64 32]", "[64 64]"}},
		{"down_proj with too few columns", false,
			change{tensors: map[string]modeltest.Tensor{layer + "mlp.down_proj.weight": modeltest.Zeros("F32", 64, 64)}},
			[]string{layer + "mlp.down_proj.weight", "[64 64]", "[64 128]"}},
		{"weight with three dimensions", false,
			change{tensors: map[string]modeltest.Tensor{layer + "self_attn.o_proj.weight": modeltest.Zeros("F32", 1, 64, 64)}},
			[]string{layer + "self_attn.o_proj.weight", "[1 64 64]", "[64 64]"}},
		{"norm of the wrong width", false,
			change{tensors: map[string]modeltest.Tensor{layer + "input_layernorm.weight": modeltest.Zeros("F32", 32)}},
			[]string{layer + "input_layernorm.weight", "[32]", "[64]"}},
		{"final norm without dimensions", false,
			change{tensors: map[string]modeltest.Tensor{"model.norm.weight": modeltest.Zeros("F32")}},
			[]string{"model.norm.weight", "[]", "[64]"}},
		{"more attention heads than the weights have", false,
			change{config: map[string]any{"num_attention_heads": 8}},
			[]string{layer + "self_attn.k_proj.weight", "[32 64]", "[16 64]"}},
		{"attention heads that do not divide the width", false,
			change{config: map[string]any{"num_attention_heads": 3}},
			[]string{layer + "self_attn.q_proj.weight", "[64 64]", "[63 64]"}},
		{"more key-value heads than the weights have", false,
			change{config: map[string]any{"num_key_value_heads": 4}},
			[]string{layer + "self_attn.k_proj.weight", "[32 64]", "[64 64]"}},
		{"second layer checked like the first", false,
			change{tensors: map[string]modeltest.Tensor{"model.layers.1.mlp.up_proj.weight": modeltest.Zeros("F32", 128, 63)}},
			[]string{"model.layers.1.mlp.up_proj.weight", "[128 63]", "[128 64]"}},

		{"scales without dimensions", true,
			change{tensors: map[string]modeltest.Tensor{layer + "self_attn.q_proj.scales": modeltest.Zeros("F32")}},
			[]string{layer + "self_attn.q_proj.scales", "[]", "[64 "}},
		{"scales with columns that do not divide the inputs", true,
			change{tensors: map[string]modeltest.Tensor{layer + "self_attn.q_proj.scales": modeltest.Zeros("F32", 64, 3)}},
			[]string{layer + "self_attn.q_proj.scales", "[64 3]"}},
		{"scales with too few rows", true,
			change{tensors: map[string]modeltest.Tensor{layer + "mlp.gate_proj.scales": modeltest.Zeros("F32", 64, 1)}},
			[]string{layer + "mlp.gate_proj.scales", "[64 1]", "[128 "}},
		{"scales for a group size MLX does not have", true,
			change{tensors: map[string]modeltest.Tensor{
				layer + "self_attn.q_proj.scales": modeltest.Zeros("F32", 64, 4),
				layer + "self_attn.q_proj.biases": modeltest.Zeros("F32", 64, 4),
			}},
			[]string{layer + "self_attn.q_proj.scales", "[64 4]", "group size of 16"}},
		{"packed weight with columns for no whole number of bits", true,
			change{tensors: map[string]modeltest.Tensor{layer + "self_attn.q_proj.weight": modeltest.Zeros("U32", 64, 5)}},
			[]string{layer + "self_attn.q_proj.weight", "[64 5]"}},
		{"packed weight with a number of bits MLX does not have", true,
			change{tensors: map[string]modeltest.Tensor{layer + "self_attn.q_proj.weight": modeltest.Zeros("U32", 64, 14)}},
			[]string{layer + "self_attn.q_proj.weight", "[64 14]", "7 bits"}},
		{"packed weight with too few rows", true,
			change{tensors: map[string]modeltest.Tensor{layer + "mlp.gate_proj.weight": modeltest.Zeros("U32", 64, 8)}},
			[]string{layer + "mlp.gate_proj.weight", "[64 8]", "[128 "}},
		{"packed embedding with fewer rows than vocab_size", true,
			change{tensors: map[string]modeltest.Tensor{"model.embed_tokens.weight": modeltest.Zeros("U32", 16, 8)}},
			[]string{"model.embed_tokens.weight", "[16 8]", "[32 "}},
		{"missing biases", true,
			change{remove: []string{layer + "mlp.down_proj.biases"}},
			[]string{layer + "mlp.down_proj.biases"}},
		{"biases of another shape than the scales", true,
			change{tensors: map[string]modeltest.Tensor{layer + "mlp.down_proj.biases": modeltest.Zeros("F32", 64, 1)}},
			[]string{layer + "mlp.down_proj.biases", "[64 1]", "[64 2]"}},
		{"bias of a quantized layer with the wrong length", true,
			change{tensors: map[string]modeltest.Tensor{layer + "self_attn.v_proj.bias": modeltest.Zeros("F32", 64)}},
			[]string{layer + "self_attn.v_proj.bias", "[64]", "[32]"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config, tensors := modeltest.TinyConfig(), modeltest.TinyDense()
			if tt.quantized {
				config, tensors = modeltest.TinyQuantizedConfig(), modeltest.TinyQuantized()
			}
			maps.Copy(config, tt.change.config)
			maps.Copy(tensors, tt.change.tensors)
			for _, name := range tt.change.remove {
				delete(tensors, name)
			}
			err := loadTiny(t, config, tensors)
			if err == nil {
				t.Fatal("Load accepted the model")
			}
			for _, want := range tt.want {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not mention %q", err, want)
				}
			}
			if strings.Contains(err.Error(), "panic") {
				t.Errorf("Load failed with a panic: %v", err)
			}
		})
	}
}

// A file that ends before the data its header promises must fail the load
// without leaving the tensors of the file behind.
func TestLoadRejectsATruncatedFile(t *testing.T) {
	dir := modeltest.WriteModel(t, modeltest.TinyConfig(), modeltest.TinyDense())
	path := filepath.Join(dir, "model.safetensors")
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, content[:len(content)-4096], 0o644); err != nil {
		t.Fatal(err)
	}
	onThread(t, func() error {
		mlx.ClearCache()
		memory, files := mlx.ActiveMemory(), openFiles(t)
		_, _, err := models.Load(dir)
		if err == nil {
			t.Error("Load accepted a truncated model.safetensors")
			return nil
		}
		mlx.ClearCache()
		if after := mlx.ActiveMemory(); after != memory {
			t.Errorf("the failed Load left %d bytes of arrays behind", int64(after)-int64(memory))
		}
		if after := openFiles(t); after != files {
			t.Errorf("the failed Load left %d files open", after-files)
		}
		return nil
	})
}
