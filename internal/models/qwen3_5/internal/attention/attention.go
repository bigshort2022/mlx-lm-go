// Package attention implements Qwen's output-gated grouped-query attention.
package attention

import (
	"math"

	"github.com/bigshort2022/mlx-lm-go/internal/cache"
	"github.com/bigshort2022/mlx-lm-go/internal/nn"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// Config describes the attention projections, head layout, and rotary positions.
type Config struct {
	HiddenSize, Heads, KVHeads, HeadDim, RotaryDims int
	Theta, Eps                                      float32
	Bias                                            bool
}

// Layer is one output-gated full-attention layer.
type Layer struct {
	heads, kvHeads, headDim    int
	scale                      float32
	qProj, kProj, vProj, oProj *nn.Linear
	qNorm, kNorm               *nn.RMSNorm
	rope                       nn.RoPE
}

// Load reads the layer's projections and norms at prefix through l.
func Load(l *nn.Loader, prefix string, cfg Config) *Layer {
	qSize := cfg.Heads * cfg.HeadDim
	kvSize := cfg.KVHeads * cfg.HeadDim
	return &Layer{
		heads:   cfg.Heads,
		kvHeads: cfg.KVHeads,
		headDim: cfg.HeadDim,
		scale:   float32(1 / math.Sqrt(float64(cfg.HeadDim))),
		qProj:   l.Linear(prefix+".q_proj", cfg.HiddenSize, 2*qSize, cfg.Bias),
		kProj:   l.Linear(prefix+".k_proj", cfg.HiddenSize, kvSize, cfg.Bias),
		vProj:   l.Linear(prefix+".v_proj", cfg.HiddenSize, kvSize, cfg.Bias),
		oProj:   l.Linear(prefix+".o_proj", qSize, cfg.HiddenSize, cfg.Bias),
		qNorm:   l.RMSNorm(prefix+".q_norm", cfg.HeadDim, cfg.Eps),
		kNorm:   l.RMSNorm(prefix+".k_norm", cfg.HeadDim, cfg.Eps),
		rope:    nn.RoPE{Dims: cfg.RotaryDims, Base: cfg.Theta, Scale: 1},
	}
}

// Forward applies causal self-attention to x of shape [batch, length, hidden].
// If state is non-nil, it is extended with this sequence's keys and values.
func (l *Layer) Forward(x *mlx.Array, state *cache.KVCache) *mlx.Array {
	b, n := x.Dim(0), x.Dim(1)
	d := l.headDim

	// Each head's query features are followed by its gate features.
	qg := mlx.Reshape(l.qProj.Forward(x), b, n, l.heads, 2*d)
	q := mlx.Slice(qg, []int{0, 0, 0, 0}, []int{b, n, l.heads, d})
	gate := mlx.Slice(qg, []int{0, 0, 0, d}, []int{b, n, l.heads, 2 * d})
	k := mlx.Reshape(l.kProj.Forward(x), b, n, l.kvHeads, d)
	v := mlx.Reshape(l.vProj.Forward(x), b, n, l.kvHeads, d)

	q = mlx.Transpose(l.qNorm.Forward(q), 0, 2, 1, 3)
	k = mlx.Transpose(l.kNorm.Forward(k), 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	offset := 0
	if state != nil {
		offset = state.Offset()
	}
	q = l.rope.Forward(q, offset)
	k = l.rope.Forward(k, offset)
	if state != nil {
		k, v = state.UpdateAndFetch(k, v)
	}

	mask := ""
	if n > 1 {
		// MLX aligns the causal mask to the end of the cached keys.
		mask = "causal"
	}
	output := mlx.ScaledDotProductAttention(q, k, v, l.scale, mask, nil)
	output = mlx.Multiply(mlx.Transpose(output, 0, 2, 1, 3), mlx.Sigmoid(gate))
	return l.oProj.Forward(mlx.Reshape(output, b, n, -1))
}
