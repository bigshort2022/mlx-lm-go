package sample

import (
	"math"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/mlx"
)

var inf = float32(math.Inf(-1))

// logOf returns the log of the given probabilities as a [1, n] float32 array.
func logOf(probs ...float32) *mlx.Array {
	return mlx.Log(mlx.FromFloat32(probs, 1, len(probs)))
}

// Expected values come from mlx_lm.sample_utils.apply_top_p of mlx-lm 0.32.2.
// Each of these inputs gives another result in mlx-lm 0.31.3, which masked
// by cumulative probability instead of by a threshold on the log probability.
func TestTopPWhereItDiffersFromOlderReferences(t *testing.T) {
	log02, log03, log04 := float32(-1.6094378), float32(-1.2039727), float32(-0.9162907)
	tests := []struct {
		name     string
		logprobs func() *mlx.Array
		p        float32
		want     []float32
	}{
		// 0.31.3 dropped two of the three tied tokens.
		{"tied tokens stay together", func() *mlx.Array { return logOf(0.4, 0.2, 0.2, 0.2) }, 0.5,
			[]float32{log04, log02, log02, log02}},
		// 0.31.3 kept the 0.2 token.
		{"mass exactly at the threshold", func() *mlx.Array { return logOf(0.1, 0.2, 0.3, 0.4) }, 0.7,
			[]float32{inf, inf, log03, log04}},
		// 0.31.3 kept the 0.4 token: it did not divide by the total mass.
		{"mass that does not sum to one", func() *mlx.Array { return logOf(0.2, 0.4, 0.6, 0.8) }, 0.5,
			[]float32{inf, inf, -0.5108255, -0.22314352}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inScope(t, func() {
				wantFiltered(t, TopP(tt.logprobs(), tt.p), tt.want, 1e-6)
			})
		})
	}
}

func TestTopPJustAboveTheThresholdKeepsOneMoreToken(t *testing.T) {
	inScope(t, func() {
		got := TopP(logOf(0.1, 0.2, 0.3, 0.4), 0.71)
		wantFiltered(t, got, []float32{inf, -1.6094378, -1.2039727, -0.9162907}, 1e-6)
	})
}

func TestTopKRejectsKOutsideTheVocabulary(t *testing.T) {
	inScope(t, func() {
		for _, k := range []int{0, -1, 4, 5} {
			_, err := TopK(logOf(0.1, 0.2, 0.3, 0.4), k)
			if err == nil || !strings.Contains(err.Error(), "top_k") {
				t.Errorf("TopK(k=%d) error = %v, want one about top_k", k, err)
			}
		}
		got, err := TopK(logOf(0.1, 0.2, 0.3, 0.4), 3)
		if err != nil {
			t.Fatal(err)
		}
		wantFiltered(t, got, []float32{inf, -1.6094378, -1.2039727, -0.9162907}, 1e-6)
	})
}

// The model computes in float16 and so must everything after it: a float32
// scalar would silently promote the logits.
func TestFiltersAndProcessorsKeepTheElementType(t *testing.T) {
	inScope(t, func() {
		half := func() *mlx.Array { return mlx.AsType(logOf(0.1, 0.2, 0.3, 0.4), mlx.Float16) }
		tokens := mlx.FromInt32([]int32{1, 1, 3}, 3)
		topK, err := TopK(half(), 2)
		if err != nil {
			t.Fatal(err)
		}
		biased, err := LogitBias(half(), map[int32]float32{2: 1.5})
		if err != nil {
			t.Fatal(err)
		}
		for name, got := range map[string]*mlx.Array{
			"TopK":              topK,
			"TopP":              TopP(half(), 0.5),
			"MinP":              MinP(half(), 0.5),
			"LogitBias":         biased,
			"RepetitionPenalty": RepetitionPenalty(tokens, half(), 1.3, 20),
			"PresencePenalty":   PresencePenalty(tokens, half(), 0.5, 20),
			"FrequencyPenalty":  FrequencyPenalty(tokens, half(), 0.5, 20),
		} {
			if got.DType() != mlx.Float16 {
				t.Errorf("%s returned %v for float16 input", name, got.DType())
			}
		}
		// Python: apply_top_p(logprobs.astype(mx.float16), 0.5)
		wantFiltered(t, TopP(half(), 0.5), []float32{inf, inf, -1.2041016, -0.9165039}, 0)
	})
}

// A context size n selects tokens[-n:] as Python does.
func TestContextSizeSelectsTheLastTokens(t *testing.T) {
	tests := []struct {
		name        string
		contextSize int
		want        []float32 // the logits of four tokens, all zero before
	}{
		{"last two", 2, []float32{0, -1, 0, -1}},
		{"more than there are", 20, []float32{-1, -2, 0, -1}},
		{"zero means all", 0, []float32{-1, -2, 0, -1}},
		{"negative skips from the front", -3, []float32{0, 0, 0, -1}},
		{"negative past the end selects none", -9, []float32{0, 0, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inScope(t, func() {
				tokens := mlx.FromInt32([]int32{0, 1, 1, 3}, 4)
				got := FrequencyPenalty(tokens, mlx.Zeros(mlx.Float32, 1, 4), 1, tt.contextSize)
				wantFiltered(t, got, tt.want, 0)
			})
		})
	}
}

func TestLogitBiasWithoutEntriesReturnsTheLogits(t *testing.T) {
	inScope(t, func() {
		logits := logOf(0.5, 0.5)
		got, err := LogitBias(logits, nil)
		if err != nil || got != logits {
			t.Errorf("LogitBias(nil) = %v, %v; want its input", got, err)
		}
	})
}

// MLX does not check indices: a token outside the vocabulary would write
// outside the logits.
func TestLogitBiasRejectsTokensOutsideTheVocabulary(t *testing.T) {
	inScope(t, func() {
		for _, id := range []int32{-1, 4, 1 << 20} {
			_, err := LogitBias(logOf(0.1, 0.2, 0.3, 0.4), map[int32]float32{1: 1, id: 1})
			if err == nil || !strings.Contains(err.Error(), "logit_bias") {
				t.Errorf("LogitBias with token %d of 4: error = %v, want one about logit_bias", id, err)
			}
		}
	})
}
