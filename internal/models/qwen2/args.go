package qwen2

import (
	"encoding/json"
	"fmt"

	"github.com/bigshort2022/mlx-lm-go/internal/nn"
)

// modelArgs holds the fields of config.json the architecture reads:
// ModelArgs in the reference.
type modelArgs struct {
	HiddenSize            int          `json:"hidden_size"`
	NumHiddenLayers       int          `json:"num_hidden_layers"`
	IntermediateSize      int          `json:"intermediate_size"`
	NumAttentionHeads     int          `json:"num_attention_heads"`
	RMSNormEps            float32      `json:"rms_norm_eps"`
	VocabSize             int          `json:"vocab_size"`
	NumKeyValueHeads      int          `json:"num_key_value_heads"`
	MaxPositionEmbeddings int          `json:"max_position_embeddings"`
	RopeTheta             float32      `json:"rope_theta"`
	RopeTraditional       bool         `json:"rope_traditional"`
	RopeScaling           *ropeScaling `json:"rope_scaling"`
	TieWordEmbeddings     bool         `json:"tie_word_embeddings"`
}

// ropeScaling is the "rope_scaling" object of config.json.
type ropeScaling struct {
	Type     string  `json:"type"`
	RopeType string  `json:"rope_type"` // read when Type is empty
	Factor   float32 `json:"factor"`
}

// parseArgs reads the arguments from the content of config.json. Fields the
// file leaves out keep the defaults of the reference; those without a default
// are required.
func parseArgs(raw json.RawMessage) (*modelArgs, error) {
	args := &modelArgs{
		MaxPositionEmbeddings: 32768,
		RopeTheta:             1000000,
		TieWordEmbeddings:     true,
	}
	if err := json.Unmarshal(raw, args); err != nil {
		return nil, fmt.Errorf("config.json: %w", err)
	}
	for _, required := range []struct {
		name string
		set  bool
	}{
		{"hidden_size", args.HiddenSize > 0},
		{"num_hidden_layers", args.NumHiddenLayers > 0},
		{"intermediate_size", args.IntermediateSize > 0},
		{"num_attention_heads", args.NumAttentionHeads > 0},
		{"rms_norm_eps", args.RMSNormEps > 0},
		{"vocab_size", args.VocabSize > 0},
		{"num_key_value_heads", args.NumKeyValueHeads > 0},
	} {
		if !required.set {
			return nil, fmt.Errorf("config.json needs a positive %s", required.name)
		}
	}
	switch t := args.ropeType(); t {
	case "default", "mrope":
	case "linear":
		if args.RopeScaling.Factor <= 0 {
			return nil, fmt.Errorf("config.json needs a positive rope_scaling.factor for linear RoPE")
		}
	default:
		return nil, fmt.Errorf("unsupported RoPE type %q; supported: default, linear, mrope", t)
	}
	return args, nil
}

// ropeType returns the kind of position scaling the configuration asks for.
func (a *modelArgs) ropeType() string {
	switch s := a.RopeScaling; {
	case s == nil:
		return "default"
	case s.Type != "":
		return s.Type
	case s.RopeType != "":
		return s.RopeType
	default:
		return "default"
	}
}

// rope returns the rotary embedding for heads of the given size:
// initialize_rope in the reference, for the types parseArgs lets through.
func (a *modelArgs) rope(dims int) *nn.RoPE {
	scale := float32(1)
	if a.ropeType() == "linear" {
		scale = 1 / a.RopeScaling.Factor
	}
	return &nn.RoPE{Dims: dims, Traditional: a.RopeTraditional, Base: a.RopeTheta, Scale: scale}
}
