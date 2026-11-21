// Package nn holds the neural network layers the models are built from. Each
// works on dense weights or on weights quantized by MLX, like the layers of
// mlx.nn and their Quantized counterparts.
package nn

import (
	"fmt"
	"slices"

	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// Loader builds layers from the tensors of a model. A layer is quantized when
// the tensors hold scales for it, as mlx_lm.utils.load_model decides.
//
// A tensor that is missing does not stop the building: the layer comes back
// incomplete and Err reports the first such tensor, so that a model can build
// all its layers and check once.
type Loader struct {
	weights map[string]*mlx.Array
	used    map[string]bool
	err     error
}

// NewLoader returns a Loader over weights, which it does not own.
func NewLoader(weights map[string]*mlx.Array) *Loader {
	return &Loader{weights: weights, used: map[string]bool{}}
}

// Err returns the first error met while building layers.
func (l *Loader) Err() error { return l.err }

// Unused returns the names of the tensors no layer has taken, sorted.
func (l *Loader) Unused() []string {
	var names []string
	for name := range l.weights {
		if !l.used[name] {
			names = append(names, name)
		}
	}
	slices.Sort(names)
	return names
}

// fail records err unless an earlier one is on record.
func (l *Loader) fail(err error) {
	if l.err == nil {
		l.err = err
	}
}

// tensor returns the named tensor. When there is none it records the error
// and returns nil.
func (l *Loader) tensor(name string) *mlx.Array {
	a, ok := l.optional(name)
	if !ok {
		l.fail(fmt.Errorf("the weights have no tensor %q", name))
	}
	return a
}

// optional returns the named tensor if the weights have it.
func (l *Loader) optional(name string) (*mlx.Array, bool) {
	a, ok := l.weights[name]
	if ok {
		l.used[name] = true
	}
	return a, ok
}

// matrix is a weight matrix, dense or quantized by MLX.
type matrix struct {
	weight          *mlx.Array
	scales          *mlx.Array // nil for a dense matrix
	biases          *mlx.Array // nil for a dense matrix and for the modes without
	groupSize, bits int
	mode            string
}

// matrix returns the weight matrix of the layer at path, which has inDims
// columns.
func (l *Loader) matrix(path string, inDims int) matrix {
	m := matrix{weight: l.tensor(path + ".weight")}
	scales, quantized := l.optional(path + ".scales")
	if !quantized || m.weight == nil {
		return m
	}
	m.scales = scales
	m.biases, _ = l.optional(path + ".biases")
	if err := m.inferPacking(inDims); err != nil {
		l.fail(fmt.Errorf("%s: %w", path, err))
	}
	return m
}

// inferPacking recovers the group size, bits and mode the matrix was
// quantized with from the shapes of its tensors, like
// mlx_lm.utils.infer_quant_config.
func (m *matrix) inferPacking(inDims int) error {
	groups := m.scales.Dim(-1)
	if inDims <= 0 || groups <= 0 {
		return fmt.Errorf("cannot infer the quantization of %d input features with %d scales each", inDims, groups)
	}
	m.groupSize = inDims / groups
	m.bits = m.weight.Dim(-1) * 32 / inDims
	// Only affine keeps the scales in the weight dtype. Each of the other
	// modes allows exactly one (bits, group size) pair.
	switch {
	case m.scales.DType() != mlx.Uint8:
		m.mode = "affine"
	case m.bits == 4 && m.groupSize == 16:
		m.mode = "nvfp4"
	case m.bits == 4 && m.groupSize == 32:
		m.mode = "mxfp4"
	case m.bits == 8 && m.groupSize == 32:
		m.mode = "mxfp8"
	default:
		return fmt.Errorf("cannot infer the quantization mode: %d bits with group size %d", m.bits, m.groupSize)
	}
	return nil
}

// mulT returns x @ m.T.
func (m matrix) mulT(x *mlx.Array) *mlx.Array {
	if m.scales == nil {
		return mlx.Matmul(x, mlx.Transpose(m.weight))
	}
	return mlx.QuantizedMatmul(x, m.weight, m.scales, m.biases, true, m.groupSize, m.bits, m.mode)
}
