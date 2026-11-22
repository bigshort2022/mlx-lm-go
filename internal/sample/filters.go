// Package sample turns logits into tokens. It is a port of
// mlx_lm/sample_utils.py: the logits processors (logit bias and the
// repetition, presence and frequency penalties), the filters on log
// probabilities (top-p, min-p, top-k) and the sampler that chains them.
//
// Like the reference, everything computes in the element type of the logits
// and expects them as [1, vocab].
package sample

import (
	"fmt"
	"math"

	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// scalar returns v as a zero-dimensional array of the element type of like,
// the way Python's weakly typed floats take the type of the other operand.
func scalar(v float64, like *mlx.Array) *mlx.Array {
	return mlx.AsType(mlx.Float(float32(v)), like.DType())
}

// negInf returns -inf in the element type of like.
func negInf(like *mlx.Array) *mlx.Array {
	return scalar(math.Inf(-1), like)
}

// lastFrom returns a[..., from:].
func lastFrom(a *mlx.Array, from int) *mlx.Array {
	stop := a.Shape()
	start := make([]int, len(stop))
	start[len(start)-1] = from
	return mlx.Slice(a, start, stop)
}

// lastTo returns a[..., :to].
func lastTo(a *mlx.Array, to int) *mlx.Array {
	stop := a.Shape()
	stop[len(stop)-1] = to
	return mlx.Slice(a, make([]int, len(stop)), stop)
}

// TopK keeps the k likeliest tokens and sets the log probability of the rest
// to -inf: apply_top_k. Like the reference it fails unless 0 < k < vocab.
func TopK(logprobs *mlx.Array, k int) (*mlx.Array, error) {
	vocab := logprobs.Dim(-1)
	if k <= 0 || k >= vocab {
		return nil, fmt.Errorf("top_k has to be an integer in the (0, %d) interval, but is %d", vocab, k)
	}
	maskIdx := lastFrom(mlx.ArgPartition(mlx.Negative(logprobs), k-1, -1), k)
	return mlx.PutAlongAxis(logprobs, maskIdx, negInf(logprobs), -1), nil
}

// MinP sets to -inf the log probability of the tokens that are less than p
// times as likely as the likeliest one: apply_min_p with min_tokens_to_keep
// 1, which is what the server passes.
func MinP(logprobs *mlx.Array, p float32) *mlx.Array {
	// Mask tokens that have a probability less than the max(p) * min_p
	topLogprobs := mlx.Max(logprobs, -1, true)
	scaledMinP := mlx.Add(topLogprobs, scalar(math.Log(float64(p)), logprobs))
	tokensToRemove := mlx.Less(logprobs, scaledMinP)
	return mlx.Where(tokensToRemove, negInf(logprobs), logprobs)
}

// TopP keeps the likeliest tokens that together hold the fraction p of the
// probability mass, with every token as likely as the last of them, and sets
// the log probability of the rest to -inf: apply_top_p (nucleus sampling).
func TopP(logprobs *mlx.Array, p float32) *mlx.Array {
	sortedLogprobs := mlx.Sort(logprobs, -1)
	sortedProbs := mlx.Exp(mlx.AsType(sortedLogprobs, mlx.Float32))
	massAbove := mlx.CumsumReverse(sortedProbs, -1, false)
	totalMass := mlx.Add(lastTo(massAbove, 1), lastTo(sortedProbs, 1))
	dropped := mlx.GreaterEqual(massAbove, mlx.Multiply(mlx.Float(p), totalMass))
	numDropped := mlx.Sum(dropped, -1, true)
	threshold := mlx.TakeAlongAxis(sortedLogprobs, numDropped, -1)
	return mlx.Where(mlx.Less(logprobs, threshold), negInf(logprobs), logprobs)
}
