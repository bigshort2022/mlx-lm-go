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

// wantShape records an error unless the named tensor has the given shape. A
// nil tensor is one that is missing, which tensor has reported.
func (l *Loader) wantShape(name string, a *mlx.Array, want ...int) {
	if a != nil && !slices.Equal(a.Shape(), want) {
		l.fail(fmt.Errorf("tensor %q has shape %v, want %v", name, a.Shape(), want))
	}
}

// matrix returns the weight matrix of the layer at path, which has the given
// numbers of rows and columns. Like a strict load in the reference, it
// records an error when a tensor of the layer has another shape.
func (l *Loader) matrix(path string, rows, cols int) matrix {
	m := matrix{weight: l.tensor(path + ".weight")}
	scales, quantized := l.optional(path + ".scales")
	switch {
	case m.weight == nil:
	case !quantized:
		l.wantShape(path+".weight", m.weight, rows, cols)
	default:
		m.scales = scales
		if err := l.packing(path, &m, rows, cols); err != nil {
			l.fail(err)
		}
	}
	return m
}

// The group sizes and bits MLX quantizes with in affine mode.
var (
	affineGroupSizes = []int{32, 64, 128}
	affineBits       = []int{1, 2, 3, 4, 5, 6, 8}
)

// packing recovers the group size, bits and mode the matrix at path was
// quantized with from the shapes of its tensors, like
// mlx_lm.utils.infer_quant_config, and checks those shapes: scales and
// biases have one column per group of columns, the packed weight has 32 bits
// per column for bits per element.
func (l *Loader) packing(path string, m *matrix, rows, cols int) error {
	scales, weight := m.scales.Shape(), m.weight.Shape()
	if cols <= 0 || len(scales) != 2 || scales[0] != rows || scales[1] <= 0 || cols%scales[1] != 0 {
		return fmt.Errorf("tensor %q has shape %v, want [%d %d/group size]", path+".scales", scales, rows, cols)
	}
	if len(weight) != 2 || weight[0] != rows || weight[1] <= 0 || weight[1]*32%cols != 0 {
		return fmt.Errorf("tensor %q has shape %v, want [%d %d*bits/32]", path+".weight", weight, rows, cols)
	}
	m.groupSize, m.bits = cols/scales[1], weight[1]*32/cols

	// Only affine keeps the scales in the weight dtype. Each of the other
	// modes allows exactly one (bits, group size) pair and has no biases.
	switch {
	case m.scales.DType() != mlx.Uint8:
		if !slices.Contains(affineGroupSizes, m.groupSize) {
			return fmt.Errorf("tensor %q has shape %v, which means a group size of %d; MLX quantizes in groups of %v",
				path+".scales", scales, m.groupSize, affineGroupSizes)
		}
		if !slices.Contains(affineBits, m.bits) {
			return fmt.Errorf("tensor %q has shape %v, which means %d bits; MLX quantizes to %v bits",
				path+".weight", weight, m.bits, affineBits)
		}
		m.mode = "affine"
		m.biases = l.tensor(path + ".biases")
		l.wantShape(path+".biases", m.biases, scales...)
	case m.bits == 4 && m.groupSize == 16:
		m.mode = "nvfp4"
	case m.bits == 4 && m.groupSize == 32:
		m.mode = "mxfp4"
	case m.bits == 8 && m.groupSize == 32:
		m.mode = "mxfp8"
	default:
		return fmt.Errorf("cannot infer the quantization mode of %s: %d bits with group size %d", path, m.bits, m.groupSize)
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
