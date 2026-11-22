package generate

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
	"github.com/bigshort2022/mlx-lm-go/internal/models/modeltest"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

func TestDenseQwenLoad(t *testing.T) {
	e := tinyEngine(t, denseQwenConfig(), denseQwenWeights())
	if e.vocab != 32 {
		t.Fatalf("vocabulary = %d, want 32", e.vocab)
	}
}

func TestDenseQwenNested(t *testing.T) {
	text := denseQwenConfig()
	text["model_type"] = "qwen3_5_text"
	text["eos_token_id"] = 19
	text["layer_types"] = []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"}
	cfg := map[string]any{"model_type": "qwen3_5", "text_config": text}
	weights := map[string]modeltest.Tensor{
		"vision_tower.conv1d.weight": modeltest.Zeros("F32", 2, 1, 4),
		"model.visual.extra.weight":  modeltest.Zeros("F32", 1),
		"mtp.conv1d.weight":          modeltest.Zeros("F32", 2, 1, 4),
	}
	for name, a := range denseQwenWeights() {
		if name == "lm_head.weight" {
			weights["language_model."+name] = a
		} else {
			weights["model.language_model."+strings.TrimPrefix(name, "model.")] = a
		}
	}
	closeReference()
	e, err := Load(modeltest.WriteModel(t, cfg, weights), Options{EOSTokenIDs: []int32{7}})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if !slices.Equal(e.eos, []int32{7}) {
		t.Errorf("EOS = %v, want [7] from options, not the nested EOS", e.eos)
	}
}

func TestDenseQwenRejectsConfig(t *testing.T) {
	closeReference()
	for _, tc := range []struct {
		name, field string
		value       any
	}{
		{"hidden size", "hidden_size", 0},
		{"grouped attention", "num_key_value_heads", 3},
		{"grouped DeltaNet", "linear_num_key_heads", 3},
		{"kernel key dimension", "linear_key_head_dim", 16},
		{"rotary dimension", "head_dim", 36},
		{"non-stock interval", "full_attention_interval", 0},
		{"no full-attention layer", "num_hidden_layers", 3},
		{"schedule", "layer_types", []string{"full_attention", "linear_attention", "linear_attention", "linear_attention"}},
		{"MoE", "num_experts", 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := denseQwenConfig()
			cfg[tc.field] = tc.value
			checkDenseQwenLoadError(t, cfg, denseQwenWeights(), tc.field)
		})
	}
	t.Run("MoE architecture", func(t *testing.T) {
		cfg := denseQwenConfig()
		cfg["model_type"] = "qwen3_5_moe"
		checkDenseQwenLoadError(t, cfg, denseQwenWeights(), `model type "qwen3_5_moe" is not supported`)
	})
}

func TestDenseQwenRejectsWeights(t *testing.T) {
	closeReference()
	const prefix = "model.layers.0.linear_attn."
	for _, tc := range []struct {
		name, key string
		shape     []int
	}{
		{"missing", prefix + "A_log", nil},
		{"A_log shape", prefix + "A_log", []int{1, 2}},
		{"dt_bias shape", prefix + "dt_bias", []int{1}},
		{"conv shape", prefix + "conv1d.weight", []int{128, 4}},
		{"unexpected text", "model.layers.0.linear_attn.extra", []int{1}},
		{"ambiguous prefix", "language_model.model.norm.weight", []int{64}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			weights := denseQwenWeights()
			if tc.shape == nil {
				delete(weights, tc.key)
			} else {
				weights[tc.key] = modeltest.Zeros("F32", tc.shape...)
			}
			checkDenseQwenLoadError(t, denseQwenConfig(), weights, tc.key)
		})
	}
}

func TestDenseQwenAffine4(t *testing.T) {
	closeReference()
	const head = "lm_head"
	weights := denseQwenWeights()
	weights[head+".weight"] = modeltest.RandomBits(head, 32, 8)
	weights[head+".scales"] = modeltest.Random(head+".scales", 0.02, 32, 1)
	weights[head+".biases"] = modeltest.Random(head+".biases", 0.02, 32, 1)
	t.Run("valid", func(t *testing.T) {
		e := tinyEngine(t, denseQwenConfig(), weights)
		_, result := generate(t, e, greedy([]int32{1, 5, 9, 2}, 2))
		if result.CompletionTokens != 2 {
			t.Fatalf("completion tokens = %d, want 2", result.CompletionTokens)
		}
	})
	t.Run("invalid scales", func(t *testing.T) {
		weights[head+".scales"] = modeltest.Zeros("F32", 31, 1)
		checkDenseQwenLoadError(t, denseQwenConfig(), weights, head+".scales")
	})
	t.Run("invalid packed weight", func(t *testing.T) {
		weights[head+".scales"] = modeltest.Zeros("F32", 32, 1)
		weights[head+".weight"] = modeltest.RandomBits(head, 32, 7)
		checkDenseQwenLoadError(t, denseQwenConfig(), weights, head+".weight")
	})
}

func TestDenseQwenChunkedGeneration(t *testing.T) {
	closeReference()
	dir := modeltest.WriteModel(t, denseQwenConfig(), denseQwenWeights())
	before := activeMemory(t)
	prompt := make([]int32, 270) // cross the KV cache's 256-position growth boundary
	for i := range prompt {
		prompt[i] = int32((i*13 + i/7) % 32)
	}
	req := greedy(prompt, 8)
	run := func(step int, repeat bool) ([]int32, engine.Result) {
		t.Helper()
		e, err := Load(dir, Options{PrefillStepSize: step})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Close()
		// A distinct request must not affect the next one's hybrid cache.
		if repeat {
			generate(t, e, greedy([]int32{2, 3, 7, 4}, 3))
		}
		return generate(t, e, req)
	}
	want, wantResult := run(2048, false)
	if wantResult.PromptTokens != len(prompt) || wantResult.CompletionTokens != 8 || wantResult.FinishReason != "length" {
		t.Fatalf("whole prompt result = %+v", wantResult)
	}
	for _, repeat := range []bool{false, true} {
		got, result := run(41, repeat)
		if !slices.Equal(got, want) || result != wantResult {
			t.Errorf("chunked (after another request = %t): got %v %+v, want %v %+v", repeat, got, result, want, wantResult)
		}
	}
	if after := activeMemory(t); after != before {
		t.Errorf("after closing the tiny engines MLX holds %d bytes, want %d", after, before)
	}
}

func TestDenseQwenState(t *testing.T) {
	e := tinyEngine(t, denseQwenConfig(), denseQwenWeights())
	prompt := make([]int32, 270)
	for i := range prompt {
		prompt[i] = int32((i*13 + i/7) % 32)
	}
	err := e.thread.Do(context.Background(), func() error {
		state := e.model.NewState()
		defer state.Free()
		if state.Offset() != 0 || len(state.Arrays()) != 0 {
			return fmt.Errorf("new state is not empty")
		}
		last := func(z *mlx.Array, n int) []float32 {
			return mlx.Slice(z, []int{0, n - 1, 0}, []int{1, n, e.vocab}).Float32s()
		}
		var want, got []float32
		if err := mlx.Scoped(func() error {
			z := e.model.Forward(mlx.FromInt32(prompt, 1, len(prompt)), nil)
			if shape := z.Shape(); !slices.Equal(shape, []int{1, len(prompt), e.vocab}) {
				return fmt.Errorf("nil-state logits shape = %v", shape)
			}
			want = last(z, len(prompt))
			return nil
		}); err != nil {
			return err
		}
		for i := 0; i < len(prompt); i += 64 {
			j := min(i+64, len(prompt))
			if err := mlx.Scoped(func() error {
				z := e.model.Forward(mlx.FromInt32(prompt[i:j], 1, j-i), state)
				mlx.Keep(state.Arrays()...)
				if j == len(prompt) {
					got = last(z, j-i)
				}
				return nil
			}); err != nil {
				return err
			}
			if state.Offset() != j {
				return fmt.Errorf("state offset = %d, want %d", state.Offset(), j)
			}
			mlx.Eval(state.Arrays()...)
		}
		buffers := state.Arrays()
		wantDims := [][]int{
			{1, 3, 128}, {1, 2, 32, 32}, {1, 3, 128}, {1, 2, 32, 32},
			{1, 3, 128}, {1, 2, 32, 32}, {1, 1, 512, 32}, {1, 1, 512, 32},
		}
		if len(buffers) != len(wantDims) {
			return fmt.Errorf("hybrid state holds %d buffers, want %d", len(buffers), len(wantDims))
		}
		for i, a := range buffers {
			if shape := a.Shape(); !slices.Equal(shape, wantDims[i]) {
				return fmt.Errorf("cache buffer %d shape = %v, want %v", i, shape, wantDims[i])
			}
		}
		state.Free()
		if state.Offset() != 0 || len(state.Arrays()) != 0 || slices.ContainsFunc(buffers, (*mlx.Array).Valid) {
			return fmt.Errorf("reset did not empty and release state")
		}
		var worst float64
		for i, v := range got {
			worst = max(worst, math.Abs(float64(v-want[i])))
		}
		t.Logf("whole vs chunked tiny logits max |delta| = %g", worst)
		if !(worst <= 1e-4) {
			return fmt.Errorf("chunked logits differ from whole-sequence logits by %g", worst)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func checkDenseQwenLoadError(t *testing.T, cfg map[string]any, weights map[string]modeltest.Tensor, contains string) {
	t.Helper()
	e, err := Load(modeltest.WriteModel(t, cfg, weights), Options{})
	if err == nil {
		e.Close()
		t.Fatalf("Load succeeded; want error containing %q", contains)
	}
	if !strings.Contains(err.Error(), contains) {
		t.Fatalf("Load error = %v; want it to contain %q", err, contains)
	}
}

func denseQwenConfig() map[string]any {
	return map[string]any{
		"model_type": "qwen3_5", "hidden_size": 64, "intermediate_size": 128,
		"num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
		"head_dim": 32, "vocab_size": 32, "rms_norm_eps": 1e-6,
		"full_attention_interval": 4, "linear_num_key_heads": 1, "linear_num_value_heads": 2,
		"linear_key_head_dim": 32, "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
		"rope_parameters": map[string]any{"rope_type": "default", "rope_theta": 100000, "partial_rotary_factor": 0.25},
	}
}

func denseQwenWeights() map[string]modeltest.Tensor {
	weights := make(map[string]modeltest.Tensor)
	param := func(name string, scale float64, dims ...int) {
		weights[name] = modeltest.Random(name, scale, dims...)
	}
	linear := func(name string, rows, cols int) { param(name+".weight", 0.08, rows, cols) }
	linear("model.embed_tokens", 32, 64)
	linear("lm_head", 32, 64)
	param("model.norm.weight", 0.7, 64)
	for i := range 4 {
		p := fmt.Sprintf("model.layers.%d.", i)
		param(p+"input_layernorm.weight", 0.7, 64)
		param(p+"post_attention_layernorm.weight", 0.7, 64)
		linear(p+"mlp.gate_proj", 128, 64)
		linear(p+"mlp.up_proj", 128, 64)
		linear(p+"mlp.down_proj", 64, 128)
		if i == 3 {
			p += "self_attn."
			linear(p+"q_proj", 128, 64)
			linear(p+"k_proj", 32, 64)
			linear(p+"v_proj", 32, 64)
			linear(p+"o_proj", 64, 64)
			param(p+"q_norm.weight", 0.7, 32)
			param(p+"k_norm.weight", 0.7, 32)
		} else {
			p += "linear_attn."
			linear(p+"in_proj_qkv", 128, 64)
			linear(p+"in_proj_z", 64, 64)
			linear(p+"in_proj_b", 2, 64)
			linear(p+"in_proj_a", 2, 64)
			linear(p+"out_proj", 64, 64)
			param(p+"norm.weight", 0.7, 32)
			param(p+"conv1d.weight", 0.3, 128, 4, 1)
			param(p+"A_log", 0.5, 2)
			param(p+"dt_bias", 0.2, 2)
		}
	}
	return weights
}
