// Package qwen3_5 implements the dense text model shared by Qwen3.5 and Qwen3.8.
package qwen3_5

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/bigshort2022/mlx-lm-go/internal/cache"
	"github.com/bigshort2022/mlx-lm-go/internal/models"
	"github.com/bigshort2022/mlx-lm-go/internal/models/qwen3_5/internal/attention"
	"github.com/bigshort2022/mlx-lm-go/internal/models/qwen3_5/internal/delta"
	"github.com/bigshort2022/mlx-lm-go/internal/nn"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

func init() { models.Register("qwen3_5", New) }

const textPrefix = "language_model."

// Model is the dense hybrid language model.
type Model struct {
	weights map[string]*mlx.Array
	embed   *nn.Embedding
	layers  []decoderLayer
	norm    *nn.RMSNorm
	head    *nn.Linear // nil when tied to the embedding
}

// New consumes weights and accepts only the tensors of a dense text model,
// apart from the recognized non-text modules discarded during sanitization.
func New(cfg models.Config, weights map[string]*mlx.Array) (models.Model, error) {
	m, err := build(cfg, weights)
	if err != nil {
		freeWeights(weights)
		return nil, fmt.Errorf("qwen3_5: %w", err)
	}
	return m, nil
}

func build(cfg models.Config, weights map[string]*mlx.Array) (*Model, error) {
	a, err := parseArgs(cfg.Raw)
	if err != nil {
		return nil, err
	}
	if err := sanitize(weights, a.TieEmbeddings); err != nil {
		return nil, err
	}
	l := nn.NewLoader(weights)
	m := &Model{
		weights: weights,
		embed:   l.Embedding(textPrefix+"model.embed_tokens", a.VocabSize, a.HiddenSize),
		layers:  make([]decoderLayer, 0, min(a.NumLayers, len(weights))),
		norm:    l.RMSNorm(textPrefix+"model.norm", a.HiddenSize, a.Eps),
	}
	if !a.TieEmbeddings {
		m.head = l.Linear(textPrefix+"lm_head", a.HiddenSize, a.VocabSize, false)
	}
	if err := l.Err(); err != nil {
		return nil, err
	}
	for i := range a.NumLayers {
		layer := loadLayer(l, textPrefix+"model.layers."+strconv.Itoa(i), &a, fullAttention(i))
		if err := l.Err(); err != nil {
			return nil, err
		}
		m.layers = append(m.layers, layer)
	}
	if unused := l.Unused(); len(unused) > 0 {
		return nil, fmt.Errorf("the weights have %d tensors that are not in the model: %s", len(unused), strings.Join(unused, ", "))
	}
	return m, nil
}

// VocabSize returns the number of token IDs the model can embed.
func (m *Model) VocabSize() int { return m.embed.NumEmbeddings() }

// Free releases the model parameters. The model must not be used afterwards.
func (m *Model) Free() { freeWeights(m.weights) }

func freeWeights(weights map[string]*mlx.Array) {
	for _, a := range weights {
		a.Free()
	}
	clear(weights)
}

type decoderLayer struct {
	delta          *delta.Layer
	attention      *attention.Layer
	inputNorm      *nn.RMSNorm
	postNorm       *nn.RMSNorm
	gate, up, down *nn.Linear
}

func loadLayer(l *nn.Loader, p string, a *modelArgs, full bool) decoderLayer {
	d := decoderLayer{
		inputNorm: l.RMSNorm(p+".input_layernorm", a.HiddenSize, a.Eps),
		postNorm:  l.RMSNorm(p+".post_attention_layernorm", a.HiddenSize, a.Eps),
		gate:      l.Linear(p+".mlp.gate_proj", a.HiddenSize, a.IntermediateSize, false),
		up:        l.Linear(p+".mlp.up_proj", a.HiddenSize, a.IntermediateSize, false),
		down:      l.Linear(p+".mlp.down_proj", a.IntermediateSize, a.HiddenSize, false),
	}
	if full {
		d.attention = attention.Load(l, p+".self_attn", attention.Config{
			HiddenSize: a.HiddenSize, Heads: a.Heads, KVHeads: a.KVHeads, HeadDim: a.headDim,
			RotaryDims: a.rotaryDims, Theta: a.Theta, Eps: a.Eps, Bias: a.AttentionBias,
		})
	} else {
		p += ".linear_attn"
		d.delta = delta.Load(l, p, delta.Config{
			HiddenSize: a.HiddenSize, KeyHeads: a.KeyHeads, ValueHeads: a.ValueHeads,
			KeyDim: a.KeyDim, ValueDim: a.ValueDim, KernelSize: a.KernelSize, Eps: a.Eps,
		}, l.Tensor(p+".conv1d.weight", a.convWidth, a.KernelSize, 1),
			l.Tensor(p+".A_log", a.ValueHeads), l.Tensor(p+".dt_bias", a.ValueHeads))
	}
	return d
}

func (l *decoderLayer) attend(x *mlx.Array, s *layerState) *mlx.Array {
	x = l.inputNorm.Forward(x)
	if l.delta != nil {
		var d *delta.State
		if s != nil {
			d = &s.delta
		}
		return l.delta.Forward(x, d)
	}
	var kv *cache.KVCache
	if s != nil {
		kv = &s.kv
	}
	return l.attention.Forward(x, kv)
}

func (l *decoderLayer) finish(x, attended *mlx.Array) *mlx.Array {
	h := mlx.Add(x, attended)
	z := l.postNorm.Forward(h)
	return mlx.Add(h, l.down.Forward(nn.SwiGLU(l.gate.Forward(z), l.up.Forward(z))))
}

// Forward implements models.Model. State is nil or returned by NewState.
func (m *Model) Forward(tokens *mlx.Array, state models.State) *mlx.Array {
	var s *State
	if state != nil {
		s = state.(*State)
	}
	x := m.embed.Forward(tokens)
	for i := range m.layers {
		l := &m.layers[i]
		var cs *layerState
		if s != nil {
			cs = &s.layers[i]
		}
		x = l.finish(x, l.attend(x, cs))
	}
	if s != nil {
		s.offset += tokens.Dim(1)
	}
	x = m.norm.Forward(x)
	if m.head == nil {
		return m.embed.AsLinear(x)
	}
	return m.head.Forward(x)
}

type layerState struct {
	// Exactly one of these owns buffers, depending on the decoder layer.
	delta delta.State
	kv    cache.KVCache
}

// State owns the caches of one request and tracks the number of tokens fed.
type State struct {
	layers []layerState
	offset int
}

// NewState returns empty DeltaNet and full-attention caches for this model.
func (m *Model) NewState() models.State {
	return &State{layers: make([]layerState, len(m.layers))}
}

// Offset returns the number of tokens fed since the state was created or reset.
func (s *State) Offset() int { return s.offset }

// Arrays returns both cache arrays for each populated layer, in layer order.
func (s *State) Arrays() []*mlx.Array {
	if s.offset == 0 {
		return nil
	}
	arrays := make([]*mlx.Array, 0, 2*len(s.layers))
	for i := range s.layers {
		l := &s.layers[i]
		arrays = append(arrays, l.delta.Arrays()...)
		arrays = append(arrays, l.kv.Arrays()...)
	}
	return arrays
}

// Free releases the cache buffers and resets the state for another prompt.
func (s *State) Free() {
	for i := range s.layers {
		l := &s.layers[i]
		l.delta.Free()
		l.kv.Free()
	}
	s.offset = 0
}
