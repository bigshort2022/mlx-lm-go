package nn

import "github.com/bigshort2022/mlx-lm-go/mlx"

// Linear applies x @ W.T + b: mlx.nn.Linear and mlx.nn.QuantizedLinear.
type Linear struct {
	weight matrix
	bias   *mlx.Array // nil for a layer without bias
}

// Linear returns the linear layer at path, which maps inDims features to
// outDims and has a bias term or not.
func (l *Loader) Linear(path string, inDims, outDims int, bias bool) *Linear {
	linear := &Linear{weight: l.matrix(path, outDims, inDims)}
	if bias {
		linear.bias = l.tensor(path + ".bias")
		l.wantShape(path+".bias", linear.bias, outDims)
	}
	return linear
}

// Forward applies the layer to the last axis of x.
func (l *Linear) Forward(x *mlx.Array) *mlx.Array {
	switch {
	case l.bias == nil:
		return l.weight.mulT(x)
	case l.weight.scales == nil:
		return mlx.Addmm(l.bias, x, mlx.Transpose(l.weight.weight))
	default:
		return mlx.Add(l.weight.mulT(x), l.bias)
	}
}
