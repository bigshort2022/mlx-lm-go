package qwen3_5

import (
	"slices"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/models"
	"github.com/bigshort2022/mlx-lm-go/internal/models/modeltest"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

func TestDenseQwenWeightsSanitized(t *testing.T) {
	const layer = "model.layers.0."
	const conv = layer + "linear_attn.conv1d.weight"
	for _, raw := range []bool{false, true} {
		name := "packaged"
		if raw {
			name = "raw"
		}
		t.Run(name, func(t *testing.T) {
			modeltest.Pin(t)
			if err := mlx.Scoped(func() error {
				put := func(dst map[string]*mlx.Array, key string, vals []float32) *mlx.Array {
					a := mlx.FromFloat32(vals, len(vals))
					dst[key] = a
					return a
				}
				weights := make(map[string]*mlx.Array)
				ignored := map[string]*mlx.Array{}
				for _, name := range []string{"vision_tower.conv1d.weight", "model.visual.conv1d.weight", "mtp.linear.weight", "lm_head.weight"} {
					ignored[name] = put(weights, name, []float32{2, 3})
				}
				if raw {
					weights[conv] = mlx.FromFloat32([]float32{1, 2, 3, 4}, 2, 1, 2)
				} else {
					weights[conv] = mlx.FromFloat32([]float32{1, 2, 3, 4}, 2, 2, 1)
				}
				origConv := weights[conv]
				adjusted := map[string]*mlx.Array{}
				for _, k := range []string{"model.norm.weight", layer + "input_layernorm.weight", layer + "post_attention_layernorm.weight", layer + "self_attn.q_norm.weight", layer + "self_attn.k_norm.weight"} {
					adjusted[k] = put(weights, k, []float32{0, 1})
				}
				unadjustedKey := layer + "linear_attn.norm.weight"
				origGatedNorm := put(weights, unadjustedKey, []float32{2, 3})
				defer freeWeights(weights)
				if err := sanitize(weights, true); err != nil {
					return err
				}
				for k, a := range ignored {
					if weights[k] != nil || weights[textPrefix+k] != nil || a.Valid() {
						t.Errorf("ignored weight %q still belongs to the evaluable map or is live", k)
					}
				}
				if got := weights[textPrefix+conv]; !slices.Equal(got.Shape(), []int{2, 2, 1}) || !slices.Equal(got.Float32s(), []float32{1, 2, 3, 4}) || origConv.Valid() == raw {
					t.Errorf("convolution layout or ownership is wrong (raw = %t)", raw)
				}
				want := []float32{0, 1}
				if raw {
					want = []float32{1, 2}
				}
				for k, a := range adjusted {
					if got := weights[textPrefix+k]; !slices.Equal(got.Float32s(), want) || a.Valid() == raw {
						t.Errorf("%q norm offset or ownership is wrong (raw = %t)", k, raw)
					}
				}
				if got := weights[textPrefix+unadjustedKey]; got != origGatedNorm || !slices.Equal(got.Float32s(), []float32{2, 3}) {
					t.Error("DeltaNet gated norm must never be offset")
				}
				return nil
			}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestDenseQwenRejectedWeightsAreFreed(t *testing.T) {
	modeltest.Pin(t)
	const raw = `{"model_type":"qwen3_5","hidden_size":64,"intermediate_size":128,"num_hidden_layers":4,"vocab_size":32,
		"num_attention_heads":2,"num_key_value_heads":1,"linear_num_key_heads":1,"linear_num_value_heads":2,
		"linear_key_head_dim":32,"linear_value_head_dim":32,"linear_conv_kernel_dim":4}`
	if err := mlx.Scoped(func() error {
		embed := mlx.Zeros(mlx.Float32, 32, 64)
		conv := mlx.Zeros(mlx.Float32, 128, 1, 4)
		ignored := mlx.Zeros(mlx.Float32, 1)
		weights := map[string]*mlx.Array{
			"model.embed_tokens.weight":                embed,
			"model.layers.0.linear_attn.conv1d.weight": conv,
			"vision_tower.blocks.0.extra.weight":       ignored,
		}
		m, err := New(models.Config{Raw: []byte(raw)}, weights)
		if err == nil || !strings.Contains(err.Error(), "model.norm.weight") || m != nil {
			t.Fatalf("New = %v, %v; want missing model.norm.weight", m, err)
		}
		if embed.Valid() || conv.Valid() || ignored.Valid() || len(weights) != 0 {
			t.Error("failed load left weight handles valid or weights in the map")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
