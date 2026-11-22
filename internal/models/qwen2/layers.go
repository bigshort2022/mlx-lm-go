package qwen2

import (
	"math"

	"github.com/bigshort2022/mlx-lm-go/internal/cache"
	"github.com/bigshort2022/mlx-lm-go/internal/nn"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// attention is grouped-query self-attention with rotary positions.
type attention struct {
	nHeads, nKVHeads int
	scale            float32

	qProj, kProj, vProj, oProj *nn.Linear
	rope                       *nn.RoPE
}

func newAttention(l *nn.Loader, path string, args *modelArgs) *attention {
	dim := args.HiddenSize
	headDim := dim / args.NumAttentionHeads
	return &attention{
		nHeads:   args.NumAttentionHeads,
		nKVHeads: args.NumKeyValueHeads,
		scale:    float32(1 / math.Sqrt(float64(headDim))),
		qProj:    l.Linear(path+".q_proj", dim, args.NumAttentionHeads*headDim, true),
		kProj:    l.Linear(path+".k_proj", dim, args.NumKeyValueHeads*headDim, true),
		vProj:    l.Linear(path+".v_proj", dim, args.NumKeyValueHeads*headDim, true),
		oProj:    l.Linear(path+".o_proj", args.NumAttentionHeads*headDim, dim, false),
		rope:     args.rope(headDim),
	}
}

// forward attends over x of shape [B, L, D]. mask is "causal" or "" for none.
// With a cache, x continues the positions in it and is appended to it.
func (a *attention) forward(x *mlx.Array, mask string, c *cache.KVCache) *mlx.Array {
	b, l := x.Dim(0), x.Dim(1)

	queries, keys, values := a.qProj.Forward(x), a.kProj.Forward(x), a.vProj.Forward(x)

	// Prepare the queries, keys and values for the attention computation
	queries = mlx.Transpose(mlx.Reshape(queries, b, l, a.nHeads, -1), 0, 2, 1, 3)
	keys = mlx.Transpose(mlx.Reshape(keys, b, l, a.nKVHeads, -1), 0, 2, 1, 3)
	values = mlx.Transpose(mlx.Reshape(values, b, l, a.nKVHeads, -1), 0, 2, 1, 3)

	if c != nil {
		queries = a.rope.Forward(queries, c.Offset())
		keys = a.rope.Forward(keys, c.Offset())
		keys, values = c.UpdateAndFetch(keys, values)
	} else {
		queries = a.rope.Forward(queries, 0)
		keys = a.rope.Forward(keys, 0)
	}

	output := mlx.ScaledDotProductAttention(queries, keys, values, a.scale, mask, nil)
	output = mlx.Reshape(mlx.Transpose(output, 0, 2, 1, 3), b, l, -1)
	return a.oProj.Forward(output)
}

// mlp is the gated feed-forward network of a block.
type mlp struct {
	gateProj, downProj, upProj *nn.Linear
}

func newMLP(l *nn.Loader, path string, args *modelArgs) *mlp {
	dim, hiddenDim := args.HiddenSize, args.IntermediateSize
	return &mlp{
		gateProj: l.Linear(path+".gate_proj", dim, hiddenDim, false),
		downProj: l.Linear(path+".down_proj", hiddenDim, dim, false),
		upProj:   l.Linear(path+".up_proj", dim, hiddenDim, false),
	}
}

func (m *mlp) forward(x *mlx.Array) *mlx.Array {
	return m.downProj.Forward(nn.SwiGLU(m.gateProj.Forward(x), m.upProj.Forward(x)))
}

// transformerBlock is one layer: attention and MLP, each on the normalised
// input and added back to it.
type transformerBlock struct {
	selfAttn               *attention
	mlp                    *mlp
	inputLayernorm         *nn.RMSNorm
	postAttentionLayernorm *nn.RMSNorm
}

func newTransformerBlock(l *nn.Loader, path string, args *modelArgs) *transformerBlock {
	return &transformerBlock{
		selfAttn:               newAttention(l, path+".self_attn", args),
		mlp:                    newMLP(l, path+".mlp", args),
		inputLayernorm:         l.RMSNorm(path+".input_layernorm", args.HiddenSize, args.RMSNormEps),
		postAttentionLayernorm: l.RMSNorm(path+".post_attention_layernorm", args.HiddenSize, args.RMSNormEps),
	}
}

func (b *transformerBlock) forward(x *mlx.Array, mask string, c *cache.KVCache) *mlx.Array {
	r := b.selfAttn.forward(b.inputLayernorm.Forward(x), mask, c)
	h := mlx.Add(x, r)
	r = b.mlp.forward(b.postAttentionLayernorm.Forward(h))
	return mlx.Add(h, r)
}
