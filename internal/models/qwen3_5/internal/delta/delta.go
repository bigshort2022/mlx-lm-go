// Package delta implements the Gated DeltaNet layer of dense Qwen3.5.
package delta

import (
	"math"

	"github.com/bigshort2022/mlx-lm-go/internal/nn"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// Config describes a Gated DeltaNet layer. KeyDim and ValueDim are per-head.
type Config struct {
	HiddenSize, KeyHeads, ValueHeads, KeyDim, ValueDim, KernelSize int
	Eps                                                            float32
}

// Layer borrows its parameters from the model.
type Layer struct {
	cfg                           Config
	qkv, z, b, a, out             *nn.Linear
	norm                          *nn.RMSNorm
	conv, aLog, dtBias            *mlx.Array
	keyWidth, valueWidth, convDim int
	qScale, kScale, qkEps         float32
}

// Load loads the projections and norm at prefix. The caller owns and
// shape-checks conv, aLog, and dtBias.
func Load(l *nn.Loader, prefix string, cfg Config, conv, aLog, dtBias *mlx.Array) *Layer {
	keyWidth := cfg.KeyHeads * cfg.KeyDim
	valueWidth := cfg.ValueHeads * cfg.ValueDim
	convDim := 2*keyWidth + valueWidth
	invScale := 1 / math.Sqrt(float64(cfg.KeyDim))
	return &Layer{
		cfg:  cfg,
		qkv:  l.Linear(prefix+".in_proj_qkv", cfg.HiddenSize, convDim, false),
		z:    l.Linear(prefix+".in_proj_z", cfg.HiddenSize, valueWidth, false),
		b:    l.Linear(prefix+".in_proj_b", cfg.HiddenSize, cfg.ValueHeads, false),
		a:    l.Linear(prefix+".in_proj_a", cfg.HiddenSize, cfg.ValueHeads, false),
		out:  l.Linear(prefix+".out_proj", valueWidth, cfg.HiddenSize, false),
		norm: l.RMSNorm(prefix+".norm", cfg.ValueDim, cfg.Eps),
		conv: conv, aLog: aLog, dtBias: dtBias,
		keyWidth: keyWidth, valueWidth: valueWidth, convDim: convDim,
		qScale: float32(invScale * invScale), kScale: float32(invScale),
		qkEps: float32(1e-6 * invScale * invScale),
	}
}

// State keeps the convolution tail and FP32 recurrence between calls. The
// zero value is empty. The caller must Keep(Arrays()...) before closing the
// scope that owns them. Forward frees the previous arrays when replacing them.
type State struct {
	tail, recurrent *mlx.Array
}

// Arrays returns the two cache buffers, or none if empty.
func (s *State) Arrays() []*mlx.Array {
	if s == nil || s.recurrent == nil {
		return nil
	}
	return []*mlx.Array{s.tail, s.recurrent}
}

// Free releases the cache buffers and resets the state.
func (s *State) Free() {
	if s == nil || s.recurrent == nil {
		return
	}
	s.tail.Free()
	s.recurrent.Free()
	*s = State{}
}

// Clone keeps independent handles to the convolution and recurrent buffers.
func (s *State) Clone() State {
	if s.recurrent == nil {
		return State{}
	}
	return State{tail: s.tail.Share(), recurrent: s.recurrent.Share()}
}

// Forward evaluates [batch, length, hidden size] unpadded activations. With
// nil state it starts from zero and does not retain the final buffers.
func (l *Layer) Forward(x *mlx.Array, state *State) *mlx.Array {
	batch, length := x.Dim(0), x.Dim(1)
	cfg := l.cfg
	qkv := l.qkv.Forward(x)
	z := mlx.Reshape(l.z.Forward(x), batch, length, cfg.ValueHeads, cfg.ValueDim)
	beta := mlx.Sigmoid(l.b.Forward(x))
	a := mlx.Add(l.a.Forward(x), l.dtBias)
	// The bias add and softplus round in the promoted a/dt_bias dtype.
	// A_log (including BF16 checkpoints) is converted before exponentiation.
	sp := mlx.LogAddExp(a, mlx.AsType(mlx.Float(0), a.DType()))
	g := mlx.Exp(mlx.Multiply(mlx.Negative(mlx.Exp(mlx.AsType(l.aLog, mlx.Float32))), sp))

	nKeep := cfg.KernelSize - 1
	var prevTail, prevRecurrent, tail *mlx.Array
	if state != nil {
		prevTail, prevRecurrent = state.tail, state.recurrent
	}
	convInput := qkv
	if nKeep > 0 {
		if prevTail == nil {
			prevTail = mlx.Zeros(x.DType(), batch, nKeep, l.convDim)
		}
		convInput = mlx.Concatenate(1, prevTail, qkv)
	}
	if state != nil {
		tail = mlx.Contiguous(mlx.Slice(convInput, []int{0, length, 0}, []int{batch, length + nKeep, l.convDim}))
	}
	conv := mlx.DepthwiseConv1D(convInput, l.conv)
	conv = mlx.Multiply(conv, mlx.Sigmoid(conv))
	chunk := func(start, stop, heads, dim int) *mlx.Array {
		return mlx.Reshape(mlx.Slice(conv, []int{0, 0, start}, []int{batch, length, stop}), batch, length, heads, dim)
	}
	q := chunk(0, l.keyWidth, cfg.KeyHeads, cfg.KeyDim)
	k := chunk(l.keyWidth, 2*l.keyWidth, cfg.KeyHeads, cfg.KeyDim)
	v := chunk(2*l.keyWidth, l.convDim, cfg.ValueHeads, cfg.ValueDim)
	// RMS adds epsilon to mean(x²) rather than sum(x²); Python folds
	// the readout scale into Q, not into the recurrence kernel.
	dtype := q.DType()
	q = mlx.Multiply(mlx.AsType(mlx.Float(l.qScale), dtype), mlx.RMSNormUnweighted(q, l.qkEps))
	k = mlx.Multiply(mlx.AsType(mlx.Float(l.kScale), dtype), mlx.RMSNormUnweighted(k, l.qkEps))
	y, recurrent := mlx.GatedDeltaUpdate(q, k, v, g, beta, prevRecurrent)

	// Match precise_swiglu: norm in model precision, then SiLU and multiply
	// in FP32, then one cast back to the recurrence output dtype.
	normed := l.norm.Forward(y)
	gated := nn.SwiGLU(mlx.AsType(z, mlx.Float32), mlx.AsType(normed, mlx.Float32))
	out := l.out.Forward(mlx.Reshape(mlx.AsType(gated, dtype), batch, length, l.valueWidth))
	if state != nil {
		state.Free() // The successor graph owns any dependency on the previous buffers.
		*state = State{tail: tail, recurrent: recurrent}
	} else {
		recurrent.Free()
	}
	return out
}
