package qwen3_5

import (
	"encoding/json"
	"fmt"
	"math"
)

const fullAttentionInterval = 4

type modelArgs struct {
	ModelType        string      `json:"model_type"`
	HiddenSize       int         `json:"hidden_size"`
	IntermediateSize int         `json:"intermediate_size"`
	NumLayers        int         `json:"num_hidden_layers"`
	VocabSize        int         `json:"vocab_size"`
	Eps              float32     `json:"rms_norm_eps"`
	TieEmbeddings    bool        `json:"tie_word_embeddings"`
	AttentionBias    bool        `json:"attention_bias"`
	Heads            int         `json:"num_attention_heads"`
	KVHeads          int         `json:"num_key_value_heads"`
	HeadDim          *int        `json:"head_dim"`
	KeyHeads         int         `json:"linear_num_key_heads"`
	ValueHeads       int         `json:"linear_num_value_heads"`
	KeyDim           int         `json:"linear_key_head_dim"`
	ValueDim         int         `json:"linear_value_head_dim"`
	KernelSize       int         `json:"linear_conv_kernel_dim"`
	Interval         int         `json:"full_attention_interval"`
	LayerTypes       []string    `json:"layer_types"`
	NumExperts       int         `json:"num_experts"`
	Rope             *rotaryArgs `json:"rope_parameters"`
	RopeScaling      *rotaryArgs `json:"rope_scaling"`
	Theta            float32     `json:"rope_theta"`
	RotaryFactor     float64     `json:"partial_rotary_factor"`

	headDim, rotaryDims, convWidth int
}

type rotaryArgs struct {
	Type         string   `json:"type"`
	RopeType     string   `json:"rope_type"`
	Theta        *float32 `json:"rope_theta"`
	RotaryFactor *float64 `json:"partial_rotary_factor"`
}

func fullAttention(index int) bool { return (index+1)%fullAttentionInterval == 0 }

func parseArgs(raw json.RawMessage) (modelArgs, error) {
	var envelope struct {
		Text json.RawMessage `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return modelArgs{}, fmt.Errorf("config.json: %w", err)
	}
	if envelope.Text != nil {
		if string(envelope.Text) == "null" {
			return modelArgs{}, fmt.Errorf("config.json: text_config must be an object")
		}
		raw = envelope.Text
	}
	a := modelArgs{Eps: 1e-6, Interval: fullAttentionInterval, Theta: 100000, RotaryFactor: 0.25}
	if err := json.Unmarshal(raw, &a); err != nil {
		return modelArgs{}, fmt.Errorf("config.json: %w", err)
	}
	if err := a.validate(); err != nil {
		return modelArgs{}, fmt.Errorf("config.json: %w", err)
	}
	return a, nil
}

func (a *modelArgs) validate() error {
	if a.ModelType != "" && a.ModelType != "qwen3_5" && a.ModelType != "qwen3_5_text" {
		return fmt.Errorf("unsupported text model_type %q", a.ModelType)
	}
	if a.NumExperts != 0 {
		return fmt.Errorf("num_experts = %d; only dense Qwen is supported", a.NumExperts)
	}
	for _, dim := range []struct {
		name string
		size int
	}{
		{"hidden_size", a.HiddenSize}, {"intermediate_size", a.IntermediateSize},
		{"vocab_size", a.VocabSize}, {"num_hidden_layers", a.NumLayers},
		{"num_attention_heads", a.Heads}, {"num_key_value_heads", a.KVHeads},
		{"linear_num_key_heads", a.KeyHeads}, {"linear_num_value_heads", a.ValueHeads},
		{"linear_key_head_dim", a.KeyDim}, {"linear_value_head_dim", a.ValueDim},
		{"linear_conv_kernel_dim", a.KernelSize},
	} {
		if dim.size <= 0 || dim.size > math.MaxInt32 {
			return fmt.Errorf("%s must be a positive MLX dimension, got %d", dim.name, dim.size)
		}
	}
	if a.Eps <= 0 {
		return fmt.Errorf("rms_norm_eps must be positive")
	}
	if a.Interval != fullAttentionInterval {
		return fmt.Errorf("full_attention_interval = %d, want %d", a.Interval, fullAttentionInterval)
	}
	if a.NumLayers < fullAttentionInterval {
		return fmt.Errorf("num_hidden_layers = %d cannot contain a full-attention layer", a.NumLayers)
	}
	if a.LayerTypes != nil {
		if len(a.LayerTypes) != a.NumLayers {
			return fmt.Errorf("layer_types has length %d, want %d", len(a.LayerTypes), a.NumLayers)
		}
		for i, kind := range a.LayerTypes {
			want := "linear_attention"
			if fullAttention(i) {
				want = "full_attention"
			}
			if kind != want {
				return fmt.Errorf("layer_types[%d] = %q, want %q", i, kind, want)
			}
		}
	}
	if a.Heads%a.KVHeads != 0 {
		return fmt.Errorf("num_attention_heads must be divisible by num_key_value_heads")
	}
	if a.ValueHeads%a.KeyHeads != 0 {
		return fmt.Errorf("linear_num_value_heads must be divisible by linear_num_key_heads")
	}
	if a.KeyDim%32 != 0 {
		return fmt.Errorf("linear_key_head_dim must be divisible by 32 for fused DeltaNet")
	}
	keyWidth, valueWidth := a.KeyHeads*a.KeyDim, a.ValueHeads*a.ValueDim
	if keyWidth > math.MaxInt32 || valueWidth > math.MaxInt32 || 2*keyWidth+valueWidth > math.MaxInt32 {
		return fmt.Errorf("linear attention projection widths exceed MLX dimensions")
	}
	a.convWidth = 2*keyWidth + valueWidth
	if a.HeadDim == nil {
		if a.HiddenSize%a.Heads != 0 {
			return fmt.Errorf("hidden_size must be divisible by num_attention_heads when head_dim is absent")
		}
		a.headDim = a.HiddenSize / a.Heads
	} else {
		a.headDim = *a.HeadDim
	}
	if a.headDim <= 0 || a.headDim > math.MaxInt32/2/a.Heads {
		return fmt.Errorf("head_dim = %d is invalid for %d attention heads", a.headDim, a.Heads)
	}
	return a.validateRotary()
}

func (a *modelArgs) validateRotary() error {
	rope := a.Rope
	if rope == nil {
		rope = a.RopeScaling
	}
	if rope != nil {
		kind := rope.Type
		if kind == "" {
			kind = rope.RopeType
		}
		if kind != "" && kind != "default" {
			return fmt.Errorf("unsupported rotary type %q, want default", kind)
		}
		if rope.Theta != nil {
			a.Theta = *rope.Theta
		}
		if rope.RotaryFactor != nil {
			a.RotaryFactor = *rope.RotaryFactor
		}
	}
	if a.Theta <= 0 || a.RotaryFactor <= 0 || a.RotaryFactor > 1 {
		return fmt.Errorf("rope_theta must be positive and partial_rotary_factor must be in (0, 1]")
	}
	a.rotaryDims = int(float64(a.headDim) * a.RotaryFactor)
	if a.rotaryDims == 0 || a.rotaryDims%2 != 0 {
		return fmt.Errorf("head_dim and partial_rotary_factor give %d rotary dimensions; want a positive even number", a.rotaryDims)
	}
	return nil
}
