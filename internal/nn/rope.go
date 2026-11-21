package nn

import "github.com/bigshort2022/mlx-lm-go/mlx"

// RoPE applies rotary position embeddings: mlx.nn.RoPE.
type RoPE struct {
	Dims        int     // the features of the last axis that are rotated
	Traditional bool    // the original, slower pairing of features
	Base        float32 // sets the rotation frequencies
	Scale       float32 // multiplies the positions
}

// Forward rotates x, whose second-to-last axis is the sequence. offset is the
// position of its first element.
func (r *RoPE) Forward(x *mlx.Array, offset int) *mlx.Array {
	return mlx.RoPE(x, r.Dims, r.Traditional, r.Base, r.Scale, offset)
}
