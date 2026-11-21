package sample

import (
	"slices"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
	"github.com/bigshort2022/mlx-lm-go/internal/models/modeltest"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// fixture loads the inputs of testdata/sampler/cases.safetensors: logits and
// logprobs of 32 tokens, of which token 5 is the likeliest, and a history of
// 14 tokens. The arrays belong to the scope of the caller.
func fixture(t *testing.T) (logits, logprobs, history *mlx.Array) {
	t.Helper()
	arrays, _, err := mlx.LoadSafetensors(modeltest.Testdata("sampler", "cases.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		for _, a := range arrays {
			a.Free()
		}
	})
	return arrays["logits"], arrays["logprobs"], arrays["history"]
}

// mustNew returns the sampler for s.
func mustNew(t *testing.T, s engine.Sampling) *Sampler {
	t.Helper()
	sampler, err := New(s)
	if err != nil {
		t.Fatal(err)
	}
	return sampler
}

// draw samples n tokens from the same logits.
func draw(t *testing.T, s *Sampler, tokens, logits *mlx.Array, n int) []int32 {
	t.Helper()
	out := make([]int32, n)
	for i := range out {
		token, err := s.Sample(tokens, logits)
		if err != nil {
			t.Fatal(err)
		}
		if got := token.Shape(); !slices.Equal(got, []int{1}) {
			t.Fatalf("the sampled token has shape %v, want [1]", got)
		}
		out[i] = mlx.AsType(token, mlx.Int32).Int32s()[0]
	}
	return out
}

func TestGreedyIgnoresSamplingFilters(t *testing.T) {
	inScope(t, func() {
		logits, _, _ := fixture(t)
		// A top-k as large as the vocabulary is an error when it is applied.
		sampler := mustNew(t, engine.Sampling{Temperature: 0, TopP: 0.1, MinP: 0.9, TopK: 32})
		if got := draw(t, sampler, nil, logits, 3); !slices.Equal(got, []int32{5, 5, 5}) {
			t.Errorf("greedy sampling returned %v, want the argmax 5 every time", got)
		}
	})
}

func TestSampleReportsABiasOutsideTheVocabulary(t *testing.T) {
	inScope(t, func() {
		logits, _, _ := fixture(t)
		sampler := mustNew(t, engine.Sampling{LogitBias: map[int32]float32{32: 1}})
		if _, err := sampler.Sample(nil, logits); err == nil || !strings.Contains(err.Error(), "logit_bias") {
			t.Errorf("Sample with a bias on token 32 of 32: error = %v, want one about logit_bias", err)
		}
	})
}

func TestSamplingAppliesTopK(t *testing.T) {
	inScope(t, func() {
		logits, _, _ := fixture(t)
		sampler := mustNew(t, engine.Sampling{Temperature: 1, TopK: 32})
		if _, err := sampler.Sample(nil, logits); err == nil || !strings.Contains(err.Error(), "top_k") {
			t.Errorf("Sample with top-k 32 of 32 tokens: error = %v, want one about top_k", err)
		}
	})
}

func TestSeedMakesSamplingReproducible(t *testing.T) {
	inScope(t, func() {
		logits, _, _ := fixture(t)
		sequence := func(seed uint64) []int32 {
			return draw(t, mustNew(t, engine.Sampling{Temperature: 1, Seed: &seed}), nil, logits, 20)
		}
		first, again, other := sequence(1), sequence(1), sequence(2)
		if !slices.Equal(first, again) {
			t.Errorf("seed 1 gave %v, then %v", first, again)
		}
		if slices.Equal(first, other) {
			t.Errorf("seeds 1 and 2 both gave %v", first)
		}
		if len(slices.Compact(slices.Sorted(slices.Values(first)))) < 2 {
			t.Errorf("20 samples at temperature 1 were all the same: %v", first)
		}
	})
}

func TestSamplingWithoutSeedContinuesTheGenerator(t *testing.T) {
	inScope(t, func() {
		logits, _, _ := fixture(t)
		seed := uint64(1)
		want := draw(t, mustNew(t, engine.Sampling{Temperature: 1, Seed: &seed}), nil, logits, 20)
		got := draw(t, mustNew(t, engine.Sampling{Temperature: 1, Seed: &seed}), nil, logits, 10)
		got = append(got, draw(t, mustNew(t, engine.Sampling{Temperature: 1}), nil, logits, 10)...)
		if !slices.Equal(got, want) {
			t.Errorf("a sampler without seed restarted the generator: got %v, want %v", got, want)
		}
	})
}

// The logit bias is a logits processor and so runs before the filters: with
// top-k 1 only the token it made the likeliest can be sampled.
func TestSamplerOrderMatchesReference(t *testing.T) {
	inScope(t, func() {
		logits, _, _ := fixture(t)
		sampler := mustNew(t, engine.Sampling{Temperature: 1, TopK: 1, LogitBias: map[int32]float32{7: 100}})
		got := draw(t, sampler, nil, logits, 20)
		if slices.ContainsFunc(got, func(token int32) bool { return token != 7 }) {
			t.Errorf("sampled %v, want the biased token 7 every time", got)
		}
	})
}

// Expected values: make_logits_processors of mlx-lm 0.32.2 with the same
// arguments, its processors applied in order to the fixture's logits and
// history.
func TestProcessorChainMatchesReference(t *testing.T) {
	inScope(t, func() {
		logits, _, history := fixture(t)
		sampler := mustNew(t, engine.Sampling{
			LogitBias:             map[int32]float32{0: -100, 7: 2.5, 31: 10},
			RepetitionPenalty:     1.3,
			RepetitionContextSize: 20,
			PresencePenalty:       0.7,
			PresenceContextSize:   4,
			FrequencyPenalty:      0.4,
			FrequencyContextSize:  20,
		})
		processed, err := sampler.process(history, logits)
		if err != nil {
			t.Fatal(err)
		}
		wantFiltered(t, processed, []float32{
			-103.670563, 2.585841, -8.409832, 0.582571, -2.830733, 3.655489, 1.656707, -1.522985,
			-0.874423, 0.504364, -2.323042, -1.268909, 3.877246, -0.106012, 4.367881, -0.515016,
			-1.655578, 0.179097, 0.167893, -3.92059, -0.187372, 3.830633, -4.50173, -1.678157,
			1.612959, 0.516996, -2.954977, -5.876956, -2.317971, -4.627281, -1.117857, 15.363989,
		}, 1e-4)
	})
}

func TestPenaltiesOfZeroAreOff(t *testing.T) {
	inScope(t, func() {
		logits, _, history := fixture(t)
		sampler := mustNew(t, engine.Sampling{RepetitionContextSize: 20, PresenceContextSize: 20, FrequencyContextSize: 20})
		if sampler.UsesTokens() {
			t.Error("a sampler without penalties asks for the token history")
		}
		if got, err := sampler.process(history, logits); err != nil || got != logits {
			t.Errorf("a sampler without bias and penalties returned %v, %v; want the logits", got, err)
		}
	})
}

func TestUsesTokensWithAnyPenalty(t *testing.T) {
	for name, s := range map[string]engine.Sampling{
		"repetition": {RepetitionPenalty: 1.1},
		"presence":   {PresencePenalty: 0.5},
		"frequency":  {FrequencyPenalty: 0.5},
	} {
		if !mustNew(t, s).UsesTokens() {
			t.Errorf("a sampler with a %s penalty does not ask for the token history", name)
		}
	}
	if mustNew(t, engine.Sampling{LogitBias: map[int32]float32{1: 1}}).UsesTokens() {
		t.Error("a sampler with only a logit bias asks for the token history")
	}
}

// Expected values: apply_top_p(logprobs, 0.9), then apply_min_p(..., 0.3),
// then apply_top_k(..., 5) of mlx-lm 0.32.2 on the fixture's logprobs. Top-p
// leaves four tokens and min-p two of them.
func TestFilterChainMatchesReference(t *testing.T) {
	inScope(t, func() {
		_, logprobs, _ := fixture(t)
		sampler := mustNew(t, engine.Sampling{Temperature: 1, TopP: 0.9, MinP: 0.3, TopK: 5})
		got, err := sampler.filter(logprobs)
		if err != nil {
			t.Fatal(err)
		}
		want := slices.Repeat([]float32{inf}, 32)
		want[5], want[31] = -0.632463, -1.45061
		wantFiltered(t, got, want, 1e-4)
	})
}

func TestTopPOfOneOrZeroIsOff(t *testing.T) {
	inScope(t, func() {
		_, logprobs, _ := fixture(t)
		for _, p := range []float32{0, 1} {
			got, err := mustNew(t, engine.Sampling{Temperature: 1, TopP: p}).filter(logprobs)
			if err != nil {
				t.Fatal(err)
			}
			if got != logprobs {
				t.Errorf("top-p %v filtered the log probabilities", p)
			}
		}
	})
}

func TestNewRejectsWhatTheReferenceRejects(t *testing.T) {
	for name, s := range map[string]engine.Sampling{
		"min_p above one":             {Temperature: 1, MinP: 1.5},
		"negative min_p":              {Temperature: 1, MinP: -0.1},
		"negative repetition penalty": {RepetitionPenalty: -1},
	} {
		if _, err := New(s); err == nil {
			t.Errorf("New accepted %s", name)
		}
	}
}

// Expected tokens: mx.random.seed(42), then twenty calls of
// make_sampler(temp=0.7, top_p=0.95, min_p=0.01, top_k=10) of mlx-lm 0.32.2
// on the fixture's log probabilities.
func TestSeededSamplesMatchReference(t *testing.T) {
	inScope(t, func() {
		logits, _, _ := fixture(t)
		seed := uint64(42)
		sampler := mustNew(t, engine.Sampling{Temperature: 0.7, TopP: 0.95, MinP: 0.01, TopK: 10, Seed: &seed})
		want := []int32{31, 5, 5, 5, 5, 5, 5, 14, 5, 21, 5, 14, 5, 5, 5, 31, 5, 5, 5, 5}
		if got := draw(t, sampler, nil, logits, 20); !slices.Equal(got, want) {
			t.Errorf("sampled %v, want %v as in Python", got, want)
		}
	})
}
