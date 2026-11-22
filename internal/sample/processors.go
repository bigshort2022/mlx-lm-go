package sample

import (
	"fmt"
	"maps"
	"slices"

	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// The processors below take the tokens generated so far, prompt included, as
// a one-dimensional integer array, and the logits of the next token. A
// context size n selects tokens[-n:] as in Python: the last n tokens, or all
// of them when n is 0.

// window returns tokens[-n:], or nil when that selects no token.
func window(tokens *mlx.Array, n int) *mlx.Array {
	length := tokens.Dim(0)
	start := 0
	switch {
	case n > 0:
		start = max(length-n, 0)
	case n < 0:
		start = min(-n, length)
	}
	switch start {
	case length:
		return nil
	case 0:
		return tokens
	}
	return mlx.Slice(tokens, []int{start}, []int{length})
}

// row returns tokens as [1, n] indices along the vocabulary axis.
func row(tokens *mlx.Array) *mlx.Array {
	return mlx.Reshape(tokens, 1, -1)
}

// LogitBias adds bias[id] to the logit of every token id in bias: the
// logit_bias processor of make_logits_processors. Unlike the reference, which
// does not check, it fails for a token outside the vocabulary: MLX would
// write outside the logits.
func LogitBias(logits *mlx.Array, bias map[int32]float32) (*mlx.Array, error) {
	if len(bias) == 0 {
		return logits, nil
	}
	vocab := logits.Dim(-1)
	ids := slices.Sorted(maps.Keys(bias))
	values := make([]float32, len(ids))
	for i, id := range ids {
		if id < 0 || int(id) >= vocab {
			return nil, fmt.Errorf("logit_bias names token %d, which is outside the vocabulary of %d tokens", id, vocab)
		}
		values[i] = bias[id]
	}
	indices := mlx.FromInt32(ids, 1, len(ids))
	addends := mlx.AsType(mlx.FromFloat32(values, 1, len(values)), logits.DType())
	return mlx.ScatterAddAxis(logits, indices, addends, -1), nil
}

// RepetitionPenalty makes the tokens of the context less likely by dividing
// their positive logits by penalty and multiplying their negative ones:
// make_repetition_penalty.
func RepetitionPenalty(tokens, logits *mlx.Array, penalty float32, contextSize int) *mlx.Array {
	tokens = window(tokens, contextSize)
	if tokens == nil {
		return logits
	}
	selected := mlx.Take(logits, tokens, 1)
	p := scalar(float64(penalty), logits)
	selected = mlx.Where(mlx.Less(selected, scalar(0, logits)), mlx.Multiply(selected, p), mlx.Divide(selected, p))
	return mlx.PutAlongAxis(logits, row(tokens), selected, -1)
}

// PresencePenalty subtracts penalty from the logit of every token that occurs
// in the context: make_presence_penalty.
func PresencePenalty(tokens, logits *mlx.Array, penalty float32, contextSize int) *mlx.Array {
	tokens = window(tokens, contextSize)
	if tokens == nil {
		return logits
	}
	selected := mlx.Subtract(mlx.Take(logits, tokens, 1), scalar(float64(penalty), logits))
	return mlx.PutAlongAxis(logits, row(tokens), selected, -1)
}

// FrequencyPenalty subtracts penalty from the logit of a token once for every
// time it occurs in the context: make_frequency_penalty.
func FrequencyPenalty(tokens, logits *mlx.Array, penalty float32, contextSize int) *mlx.Array {
	tokens = window(tokens, contextSize)
	if tokens == nil {
		return logits
	}
	return mlx.ScatterAddAxis(logits, row(tokens), scalar(-float64(penalty), logits), -1)
}
