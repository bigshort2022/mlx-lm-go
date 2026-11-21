package nn

import "github.com/bigshort2022/mlx-lm-go/mlx"

// RMSNorm scales the last axis to a root mean square of one and multiplies it
// by a learned weight: mlx.nn.RMSNorm.
type RMSNorm struct {
	weight *mlx.Array
	eps    float32
}

// RMSNorm returns the normalisation layer at path.
func (l *Loader) RMSNorm(path string, eps float32) *RMSNorm {
	return &RMSNorm{weight: l.tensor(path + ".weight"), eps: eps}
}

// Forward returns x normalised.
func (n *RMSNorm) Forward(x *mlx.Array) *mlx.Array {
	return mlx.RMSNorm(x, n.weight, n.eps)
}
