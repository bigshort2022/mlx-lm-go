package qwen2

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/models"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// tinyConfig describes a dense model small enough to check by hand: one
// layer, 8 features, 10 tokens.
const tinyConfig = `{
	"model_type": "qwen2",
	"hidden_size": 8,
	"num_hidden_layers": 1,
	"intermediate_size": 16,
	"num_attention_heads": 2,
	"num_key_value_heads": 1,
	"rms_norm_eps": 1e-6,
	"vocab_size": 10
}`

// filled returns a float32 array of the given shape holding v everywhere.
func filled(v float32, shape ...int) *mlx.Array {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return mlx.FromFloat32(slices.Repeat([]float32{v}, n), shape...)
}

// tinyWeights returns the tensors of the tiny model. Its layer adds nothing,
// its norms and its embedding hold ones: every token embeds to ones, which
// the final norm leaves as they are.
func tinyWeights() map[string]*mlx.Array {
	const layer = "model.layers.0."
	return map[string]*mlx.Array{
		"model.embed_tokens.weight":               filled(1, 10, 8),
		layer + "self_attn.q_proj.weight":         filled(0, 8, 8),
		layer + "self_attn.q_proj.bias":           filled(0, 8),
		layer + "self_attn.k_proj.weight":         filled(0, 4, 8),
		layer + "self_attn.k_proj.bias":           filled(0, 4),
		layer + "self_attn.v_proj.weight":         filled(0, 4, 8),
		layer + "self_attn.v_proj.bias":           filled(0, 4),
		layer + "self_attn.o_proj.weight":         filled(0, 8, 8),
		layer + "mlp.gate_proj.weight":            filled(0, 16, 8),
		layer + "mlp.up_proj.weight":              filled(0, 16, 8),
		layer + "mlp.down_proj.weight":            filled(0, 8, 16),
		layer + "input_layernorm.weight":          filled(1, 8),
		layer + "post_attention_layernorm.weight": filled(1, 8),
		"model.norm.weight":                       filled(1, 8),
	}
}

// config returns a Config for the given config.json content.
func config(raw string) models.Config {
	return models.Config{ModelType: "qwen2", Raw: json.RawMessage(raw)}
}

// wantLogits fails the test unless every logit of the model for three tokens
// is want.
func wantLogits(t *testing.T, model models.Model, want float32) {
	t.Helper()
	logits := model.Forward(mlx.FromInt32([]int32{3, 1, 4}, 1, 3), nil)
	if got := logits.Shape(); !slices.Equal(got, []int{1, 3, 10}) {
		t.Fatalf("logits have shape %v, want [1 3 10]", got)
	}
	for i, v := range logits.Float32s() {
		if diff := v - want; diff < -1e-4 || diff > 1e-4 {
			t.Fatalf("logit %d is %v, want %v", i, v, want)
		}
	}
}

func TestDenseModelWithTiedEmbeddings(t *testing.T) {
	inScope(t, func() {
		model, err := New(config(tinyConfig), tinyWeights())
		if err != nil {
			t.Fatal(err)
		}
		// Each logit is the dot product of two vectors of eight ones.
		wantLogits(t, model, 8)
	})
}

func TestDenseModelWithLMHead(t *testing.T) {
	inScope(t, func() {
		weights := tinyWeights()
		weights["lm_head.weight"] = filled(0.5, 10, 8)
		untied := strings.Replace(tinyConfig, "{", `{"tie_word_embeddings": false,`, 1)
		model, err := New(config(untied), weights)
		if err != nil {
			t.Fatal(err)
		}
		wantLogits(t, model, 4)
	})
}

func TestNewDropsTheTensorsTheReferenceSanitizesAway(t *testing.T) {
	inScope(t, func() {
		weights := tinyWeights()
		weights["lm_head.weight"] = filled(0.5, 10, 8) // unused: the embeddings are tied
		weights["model.layers.0.self_attn.rotary_emb.inv_freq"] = filled(0, 2)
		model, err := New(config(tinyConfig), weights)
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range []string{"lm_head.weight", "model.layers.0.self_attn.rotary_emb.inv_freq"} {
			if _, ok := weights[name]; ok {
				t.Errorf("New left %s in the weights", name)
			}
		}
		wantLogits(t, model, 8)
	})
}

func TestNewReportsAMissingTensor(t *testing.T) {
	inScope(t, func() {
		weights := tinyWeights()
		const name = "model.layers.0.mlp.up_proj.weight"
		delete(weights, name)
		_, err := New(config(tinyConfig), weights)
		if err == nil || !strings.Contains(err.Error(), name) {
			t.Errorf("New() error = %v, want one naming %s", err, name)
		}
	})
}

func TestNewRejectsTensorsThatAreNotInTheModel(t *testing.T) {
	inScope(t, func() {
		weights := tinyWeights()
		weights["model.layers.1.mlp.up_proj.weight"] = filled(0, 16, 8)
		_, err := New(config(tinyConfig), weights)
		if err == nil || !strings.Contains(err.Error(), "model.layers.1.mlp.up_proj.weight") {
			t.Errorf("New() error = %v, want one naming the extra tensor", err)
		}
	})
}

func TestNewFreesTheWeightsWhenItFails(t *testing.T) {
	inScope(t, func() {
		weights := tinyWeights()
		delete(weights, "model.norm.weight")
		if _, err := New(config(tinyConfig), weights); err == nil {
			t.Fatal("New succeeded without model.norm.weight")
		}
		for name, a := range weights {
			if a.Valid() {
				t.Fatalf("New failed but left %s alive", name)
			}
		}
	})
}

func TestParseArgsUsesTheDefaultsOfTheReference(t *testing.T) {
	args, err := parseArgs(json.RawMessage(tinyConfig))
	if err != nil {
		t.Fatal(err)
	}
	if args.MaxPositionEmbeddings != 32768 || args.RopeTheta != 1000000 || args.RopeTraditional || !args.TieWordEmbeddings {
		t.Errorf("defaults are %+v, want max_position_embeddings 32768, rope_theta 1e6, rope_traditional false, tie_word_embeddings true", *args)
	}
}

func TestParseArgsRequiresTheDimensions(t *testing.T) {
	for _, field := range []string{
		"hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads",
		"num_key_value_heads", "rms_norm_eps", "vocab_size",
	} {
		t.Run(field, func(t *testing.T) {
			without := strings.Replace(tinyConfig, `"`+field+`"`, `"x_`+field+`"`, 1)
			_, err := parseArgs(json.RawMessage(without))
			if err == nil || !strings.Contains(err.Error(), field) {
				t.Errorf("parseArgs() error = %v, want one naming %s", err, field)
			}
		})
	}
}

func TestRopeScaling(t *testing.T) {
	tests := []struct {
		name, scaling string
		wantScale     float32
	}{
		{"absent", ``, 1},
		{"null", `"rope_scaling": null,`, 1},
		{"default", `"rope_scaling": {"type": "default"},`, 1},
		{"linear", `"rope_scaling": {"type": "linear", "factor": 4},`, 0.25},
		{"rope_type key", `"rope_scaling": {"rope_type": "linear", "factor": 2},`, 0.5},
		{"mrope", `"rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},`, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			args, err := parseArgs(json.RawMessage(strings.Replace(tinyConfig, "{", "{"+tt.scaling, 1)))
			if err != nil {
				t.Fatal(err)
			}
			rope := args.rope(4)
			if rope.Dims != 4 || rope.Base != 1000000 || rope.Traditional || rope.Scale != tt.wantScale {
				t.Errorf("rope is %+v, want 4 dims, base 1e6, not traditional, scale %v", *rope, tt.wantScale)
			}
		})
	}
}

func TestRopeScalingRejectsTheTypesNotPorted(t *testing.T) {
	inScope(t, func() {
		yarn := strings.Replace(tinyConfig, "{", `{"rope_scaling": {"type": "yarn", "factor": 2},`, 1)
		_, err := New(config(yarn), tinyWeights())
		if err == nil || !strings.Contains(err.Error(), "yarn") {
			t.Errorf("New() error = %v, want one naming the yarn RoPE type", err)
		}
	})
}

func TestVocabSizeIsTheNumberOfEmbeddings(t *testing.T) {
	inScope(t, func() {
		model, err := New(config(tinyConfig), tinyWeights())
		if err != nil {
			t.Fatal(err)
		}
		if got := model.(*Model).VocabSize(); got != 10 {
			t.Errorf("VocabSize() = %d, want 10", got)
		}
	})
}

func TestFreeReleasesEveryWeight(t *testing.T) {
	inScope(t, func() {
		weights := tinyWeights()
		model, err := New(config(tinyConfig), weights)
		if err != nil {
			t.Fatal(err)
		}
		model.(*Model).Free()
		for name, a := range weights {
			if a.Valid() {
				t.Errorf("Free left %s alive", name)
			}
		}
	})
}
