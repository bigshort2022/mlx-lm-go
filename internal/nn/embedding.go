package nn

import "github.com/bigshort2022/mlx-lm-go/mlx"

// Embedding maps token ids to vectors: mlx.nn.Embedding and
// mlx.nn.QuantizedEmbedding.
type Embedding struct {
	table matrix // one row per token
}

// Embedding returns the embedding layer at path, which has a vector of dims
// features for each of numEmbeddings tokens.
func (l *Loader) Embedding(path string, numEmbeddings, dims int) *Embedding {
	return &Embedding{table: l.matrix(path, numEmbeddings, dims)}
}

// NumEmbeddings returns the number of tokens the layer has vectors for.
func (e *Embedding) NumEmbeddings() int {
	return e.table.weight.Dim(0)
}

// Forward returns the vectors of the token ids x, in a new last dimension.
func (e *Embedding) Forward(x *mlx.Array) *mlx.Array {
	t := e.table
	rows := mlx.Take(t.weight, x, 0)
	if t.scales == nil {
		return rows
	}
	var biases *mlx.Array
	if t.biases != nil {
		biases = mlx.Take(t.biases, x, 0)
	}
	return mlx.Dequantize(rows, mlx.Take(t.scales, x, 0), biases, t.groupSize, t.bits, t.mode)
}

// AsLinear applies the embedding table as a linear layer without bias, for
// models whose output projection is tied to the input embedding.
func (e *Embedding) AsLinear(x *mlx.Array) *mlx.Array {
	return e.table.mulT(x)
}
