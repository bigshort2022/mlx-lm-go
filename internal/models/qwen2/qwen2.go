// Package qwen2 is a port of mlx_lm/models/qwen2.py. Importing it makes the
// "qwen2" model type available to models.Load.
package qwen2

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/bigshort2022/mlx-lm-go/internal/cache"
	"github.com/bigshort2022/mlx-lm-go/internal/models"
	"github.com/bigshort2022/mlx-lm-go/internal/nn"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

func init() {
	models.Register("qwen2", New)
}

// Model is a Qwen2 language model.
type Model struct {
	weights map[string]*mlx.Array // every tensor of the model, by name

	embedTokens *nn.Embedding
	layers      []*transformerBlock
	norm        *nn.RMSNorm
	lmHead      *nn.Linear // nil when the output projection is the embedding
}

// New builds the model from its configuration and tensors, which it takes
// over. Tensors the architecture ignores are removed from weights and freed.
// Like a strict load in the reference, it fails when a tensor is missing or
// left over.
func New(cfg models.Config, weights map[string]*mlx.Array) (models.Model, error) {
	m, err := build(cfg, weights)
	if err != nil {
		for _, a := range weights {
			a.Free()
		}
		return nil, fmt.Errorf("qwen2: %w", err)
	}
	return m, nil
}

func build(cfg models.Config, weights map[string]*mlx.Array) (*Model, error) {
	args, err := parseArgs(cfg.Raw)
	if err != nil {
		return nil, err
	}
	sanitize(args, weights)

	l := nn.NewLoader(weights)
	m := &Model{
		weights:     weights,
		embedTokens: l.Embedding("model.embed_tokens", args.VocabSize, args.HiddenSize),
		layers:      make([]*transformerBlock, args.NumHiddenLayers),
		norm:        l.RMSNorm("model.norm", args.HiddenSize, args.RMSNormEps),
	}
	for i := range m.layers {
		m.layers[i] = newTransformerBlock(l, "model.layers."+strconv.Itoa(i), args)
	}
	if !args.TieWordEmbeddings {
		m.lmHead = l.Linear("lm_head", args.HiddenSize, args.VocabSize, false)
	}
	if err := l.Err(); err != nil {
		return nil, err
	}
	if unused := l.Unused(); len(unused) > 0 {
		return nil, fmt.Errorf("the weights have %d tensors that are not in the model: %s",
			len(unused), strings.Join(unused, ", "))
	}
	return m, nil
}

// sanitize removes and frees the tensors the model does not load.
func sanitize(args *modelArgs, weights map[string]*mlx.Array) {
	for name, a := range weights {
		tiedHead := args.TieWordEmbeddings && name == "lm_head.weight"
		// Unused precomputed rotary freqs
		if tiedHead || strings.Contains(name, "self_attn.rotary_emb.inv_freq") {
			a.Free()
			delete(weights, name)
		}
	}
}

// VocabSize returns the number of tokens the model knows: token ids run from
// 0 up to but not including it.
func (m *Model) VocabSize() int {
	return m.embedTokens.NumEmbeddings()
}

// Free releases the weights. The model must not be used afterwards.
func (m *Model) Free() {
	for _, a := range m.weights {
		a.Free()
	}
}

// NewState returns an empty key-value cache for every layer.
func (m *Model) NewState() models.State {
	return cache.NewLayers(len(m.layers))
}

// CloneState snapshots the attention cache for an exact prompt prefix.
func (m *Model) CloneState(state models.State) models.State {
	src := state.(cache.Layers)
	buf := make([]cache.KVCache, len(src))
	dst := make(cache.Layers, len(src))
	for i, c := range src {
		buf[i] = c.Clone()
		dst[i] = &buf[i]
	}
	return dst
}

// Forward implements models.Model. state is nil or comes from NewState.
func (m *Model) Forward(tokens *mlx.Array, state models.State) *mlx.Array {
	caches := make(cache.Layers, len(m.layers)) // nil caches: no state
	if state != nil {
		caches = state.(cache.Layers)
	}

	h := m.embedTokens.Forward(tokens)

	// create_attention_mask in the reference: one position needs no mask.
	mask := ""
	if h.Dim(1) > 1 {
		mask = "causal"
	}

	for i, layer := range m.layers {
		h = layer.forward(h, mask, caches[i])
	}
	h = m.norm.Forward(h)

	if m.lmHead != nil {
		return m.lmHead.Forward(h)
	}
	return m.embedTokens.AsLinear(h)
}
