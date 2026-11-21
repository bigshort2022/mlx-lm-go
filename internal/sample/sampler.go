package sample

import (
	"fmt"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// Sampler picks the next token from the logits of the model the way
// generate_step does with make_logits_processors and make_sampler: logits
// processors, log-softmax, then either the argmax or the filters and a
// categorical draw.
type Sampler struct {
	params engine.Sampling
}

// New returns the sampler for the given parameters, or the error the
// reference raises for them. When params.Seed is set it seeds MLX's random
// generator, which belongs to the calling thread: call New on the thread that
// will sample.
func New(params engine.Sampling) (*Sampler, error) {
	if params.RepetitionPenalty < 0 {
		return nil, fmt.Errorf("repetition penalty must be a non-negative float, got %v", params.RepetitionPenalty)
	}
	if params.Temperature != 0 && (params.MinP < 0 || params.MinP > 1) {
		return nil, fmt.Errorf("min_p has to be a float in the [0, 1] interval, but is %v", params.MinP)
	}
	if params.Seed != nil {
		mlx.Seed(*params.Seed)
	}
	return &Sampler{params: params}, nil
}

// UsesTokens reports whether Sample needs the tokens so far, which only the
// penalties look at.
func (s *Sampler) UsesTokens() bool {
	p := s.params
	return p.RepetitionPenalty != 0 || p.PresencePenalty != 0 || p.FrequencyPenalty != 0
}

// Sample returns the next token as a uint32 array of shape [1], given the
// logits of the last position as [1, vocab]. tokens holds the tokens so far,
// prompt included, as a one-dimensional integer array; it may be nil unless
// UsesTokens.
func (s *Sampler) Sample(tokens, logits *mlx.Array) (*mlx.Array, error) {
	logits, err := s.process(tokens, logits)
	if err != nil {
		return nil, err
	}
	logprobs := mlx.Subtract(logits, mlx.LogSumExp(logits, -1, true))
	if s.params.Temperature == 0 {
		return mlx.Argmax(logprobs, -1, false), nil
	}
	if logprobs, err = s.filter(logprobs); err != nil {
		return nil, err
	}
	return mlx.Categorical(mlx.Multiply(logprobs, scalar(1/float64(s.params.Temperature), logprobs)), -1), nil
}

// process applies the logits processors in the order of
// make_logits_processors: logit bias, then the repetition, presence and
// frequency penalties that are not zero.
func (s *Sampler) process(tokens, logits *mlx.Array) (*mlx.Array, error) {
	p := s.params
	logits, err := LogitBias(logits, p.LogitBias)
	if err != nil {
		return nil, err
	}
	if p.RepetitionPenalty != 0 {
		logits = RepetitionPenalty(tokens, logits, p.RepetitionPenalty, p.RepetitionContextSize)
	}
	if p.PresencePenalty != 0 {
		logits = PresencePenalty(tokens, logits, p.PresencePenalty, p.PresenceContextSize)
	}
	if p.FrequencyPenalty != 0 {
		logits = FrequencyPenalty(tokens, logits, p.FrequencyPenalty, p.FrequencyContextSize)
	}
	return logits, nil
}

// filter applies the sampling methods in the order of make_sampler: top-p,
// min-p, top-k, each only when its parameter turns it on.
func (s *Sampler) filter(logprobs *mlx.Array) (*mlx.Array, error) {
	p := s.params
	if 0 < p.TopP && p.TopP < 1 {
		logprobs = TopP(logprobs, p.TopP)
	}
	if p.MinP != 0 {
		logprobs = MinP(logprobs, p.MinP)
	}
	if p.TopK > 0 {
		return TopK(logprobs, p.TopK)
	}
	return logprobs, nil
}
