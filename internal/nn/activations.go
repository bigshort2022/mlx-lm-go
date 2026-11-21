package nn

import "github.com/bigshort2022/mlx-lm-go/mlx"

// SwiGLU returns silu(gate) * x, where silu(g) is g * sigmoid(g):
// mlx_lm.models.activations.swiglu. Like the reference it runs as one fused
// kernel, without the two intermediate arrays the separate operations need.
func SwiGLU(gate, x *mlx.Array) *mlx.Array {
	return mlx.SwiGLU(gate, x)
}
