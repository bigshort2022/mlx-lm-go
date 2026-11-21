package generate

import (
	"context"
	"errors"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
	"github.com/bigshort2022/mlx-lm-go/internal/exec"
	"github.com/bigshort2022/mlx-lm-go/internal/models"
	"github.com/bigshort2022/mlx-lm-go/internal/models/modeltest"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// The tests share one engine, loaded by the first test that needs it. Only
// one engine can be open in a process: a test that loads its own calls
// closeReference first, and the next test that needs the shared one loads it
// again.
var reference *Engine

func TestMain(m *testing.M) {
	code := m.Run()
	closeReference()
	os.Exit(code)
}

// referenceEngine returns the engine of the reference model. It skips the
// test when the model is not on this machine.
func referenceEngine(t *testing.T) *Engine {
	t.Helper()
	if reference == nil {
		e, err := Load(modeltest.Dir(t), Options{})
		if err != nil {
			t.Fatal(err)
		}
		reference = e
	}
	return reference
}

// closeReference closes the shared engine if it is open.
func closeReference() {
	if reference != nil {
		reference.Close()
		reference = nil
	}
}

// greedy is the request for the greedy continuation of prompt.
func greedy(prompt []int32, maxTokens int) engine.Request {
	return engine.Request{Prompt: prompt, MaxTokens: maxTokens}
}

// generate runs the request and returns the emitted tokens.
func generate(t *testing.T, e *Engine, req engine.Request) ([]int32, engine.Result) {
	t.Helper()
	var tokens []int32
	res, err := e.Generate(context.Background(), req, func(token int32) { tokens = append(tokens, token) })
	if err != nil {
		t.Fatal(err)
	}
	return tokens, res
}

func TestGenerateGreedyMatchesGolden(t *testing.T) {
	e := referenceEngine(t)
	cases, eos := modeltest.Generation(t)
	for i, c := range cases {
		want, wantReason := c.GreedyIDs, "length"
		if c.EndsWithEOS(eos) {
			want, wantReason = c.GreedyIDs[:len(c.GreedyIDs)-1], "stop"
		}

		var tokens []int32
		var first, last time.Time
		start := time.Now()
		res, err := e.Generate(context.Background(), greedy(c.PromptIDs, 64), func(token int32) {
			last = time.Now()
			if len(tokens) == 0 {
				first = last
			}
			tokens = append(tokens, token)
		})
		total := time.Since(start)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		if len(tokens) > 1 {
			t.Logf("case %d: %d tokens in %v; decode %.1f tokens/s after the first token", i, len(tokens),
				total.Round(time.Millisecond), float64(len(tokens)-1)/last.Sub(first).Seconds())
		}

		if !slices.Equal(tokens, want) {
			t.Errorf("case %d: emitted %v, want %v", i, tokens, want)
		}
		if res.FinishReason != wantReason {
			t.Errorf("case %d: finish reason %q, want %q", i, res.FinishReason, wantReason)
		}
		if res.PromptTokens != len(c.PromptIDs) || res.CompletionTokens != len(c.GreedyIDs) {
			t.Errorf("case %d: usage is %d prompt and %d completion tokens, want %d and %d",
				i, res.PromptTokens, res.CompletionTokens, len(c.PromptIDs), len(c.GreedyIDs))
		}
		if res.CachedTokens != 0 {
			t.Errorf("case %d: %d cached tokens, want 0", i, res.CachedTokens)
		}
	}
}

func TestLoadReportsAMissingModel(t *testing.T) {
	closeReference()
	if _, err := Load(t.TempDir(), Options{}); err == nil || !errors.Is(err, os.ErrNotExist) {
		t.Errorf("Load of an empty directory: error = %v, want one about the missing config.json", err)
	}
	// The failed Load leaves no engine open.
	e, err := Load(modeltest.WriteModel(t, modeltest.TinyConfig(), modeltest.TinyDense()), Options{})
	if err != nil {
		t.Fatalf("Load after a failed Load: %v", err)
	}
	e.Close()
}

// The scopes of package mlx are process-wide: two engines generating at the
// same time would free each other's arrays. Close, then Load, swaps models.
func TestLoadFailsWhileAnotherEngineIsOpen(t *testing.T) {
	e := referenceEngine(t)
	cases, eos := modeltest.Generation(t)
	c := cases[0]
	want := c.GreedyIDs[:len(c.GreedyIDs)-1]
	if !c.EndsWithEOS(eos) {
		t.Fatal("case 0 does not end with an end-of-sequence token")
	}

	tiny := modeltest.WriteModel(t, modeltest.TinyConfig(), modeltest.TinyDense())
	for range 2 {
		second, err := Load(tiny, Options{})
		if err == nil {
			second.Close()
			t.Fatal("Load succeeded while another engine is open")
		}
		if got, want := err.Error(), "generate: an engine is already open; close it first"; got != want {
			t.Errorf("Load while another engine is open: error = %q, want %q", got, want)
		}
	}
	if tokens, _ := generate(t, e, greedy(c.PromptIDs, 64)); !slices.Equal(tokens, want) {
		t.Errorf("after the refused Load the open engine emitted %v, want %v", tokens, want)
	}

	closeReference()
	other, err := Load(tiny, Options{})
	if err != nil {
		t.Fatalf("Load after Close: %v", err)
	}
	other.Close()
	if tokens, _ := generate(t, referenceEngine(t), greedy(c.PromptIDs, 64)); !slices.Equal(tokens, want) {
		t.Errorf("the engine loaded after Close emitted %v, want %v", tokens, want)
	}
}

func TestGenerateStopsAtMaxTokens(t *testing.T) {
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	c := cases[4]
	tokens, res := generate(t, e, greedy(c.PromptIDs, 3))
	if !slices.Equal(tokens, c.GreedyIDs[:3]) {
		t.Errorf("emitted %v, want the first three greedy tokens %v", tokens, c.GreedyIDs[:3])
	}
	if res.FinishReason != "length" || res.CompletionTokens != 3 {
		t.Errorf("finished %q after %d completion tokens, want \"length\" after 3", res.FinishReason, res.CompletionTokens)
	}
}

func TestGenerateWithoutTokenBudgetSamplesNothing(t *testing.T) {
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	for _, maxTokens := range []int{0, -1} {
		tokens, res := generate(t, e, greedy(cases[0].PromptIDs, maxTokens))
		want := engine.Result{PromptTokens: len(cases[0].PromptIDs), FinishReason: "length"}
		if len(tokens) != 0 || res != want {
			t.Errorf("MaxTokens %d: emitted %v with result %+v, want nothing and %+v", maxTokens, tokens, res, want)
		}
	}
}

// Case 1 answers "Paris." and ends: two tokens and the end-of-sequence token.
func TestMaxTokensCountsTheEndOfSequenceToken(t *testing.T) {
	e := referenceEngine(t)
	cases, eos := modeltest.Generation(t)
	c := cases[1]
	if len(c.GreedyIDs) != 3 || !c.EndsWithEOS(eos) {
		t.Fatalf("case 1 has greedy tokens %v, want two tokens and an end-of-sequence token", c.GreedyIDs)
	}
	for _, tt := range []struct {
		maxTokens      int
		wantReason     string
		wantCompletion int
	}{
		{3, "stop", 3},
		{2, "length", 2},
	} {
		tokens, res := generate(t, e, greedy(c.PromptIDs, tt.maxTokens))
		if !slices.Equal(tokens, c.GreedyIDs[:2]) {
			t.Errorf("MaxTokens %d: emitted %v, want %v", tt.maxTokens, tokens, c.GreedyIDs[:2])
		}
		if res.FinishReason != tt.wantReason || res.CompletionTokens != tt.wantCompletion {
			t.Errorf("MaxTokens %d: finished %q after %d completion tokens, want %q after %d",
				tt.maxTokens, res.FinishReason, res.CompletionTokens, tt.wantReason, tt.wantCompletion)
		}
	}
}

// withPrefillStepSize returns an engine that shares the model and the thread
// of e but feeds the prompt in steps of n tokens.
func withPrefillStepSize(e *Engine, n int) *Engine {
	return &Engine{thread: e.thread, model: e.model, vocab: e.vocab, eos: e.eos, prefillStepSize: n}
}

// countingModel counts the forward passes of the model it wraps and calls
// onForward at the start of each with its number, from 1.
type countingModel struct {
	models.Model
	forwards  int
	onForward func(n int)
}

func (m *countingModel) Forward(tokens *mlx.Array, state models.State) *mlx.Array {
	m.forwards++
	m.onForward(m.forwards)
	return m.Model.Forward(tokens, state)
}

func TestGeneratePrefillStepSizeDoesNotChangeOutput(t *testing.T) {
	e := referenceEngine(t)
	if e.prefillStepSize != 2048 {
		t.Fatalf("the default prefill step size is %d, want 2048", e.prefillStepSize)
	}
	small := withPrefillStepSize(e, 4)
	cases, _ := modeltest.Generation(t)
	for i, c := range cases {
		whole, wholeRes := generate(t, e, greedy(c.PromptIDs, 64))
		stepped, steppedRes := generate(t, small, greedy(c.PromptIDs, 64))
		if !slices.Equal(stepped, whole) || steppedRes != wholeRes {
			t.Errorf("case %d: a prefill step of 4 gave %v (%+v), one of 2048 gave %v (%+v)",
				i, stepped, steppedRes, whole, wholeRes)
		}
	}
}

func TestGenerateCancelsWithinOneToken(t *testing.T) {
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	emitted := 0
	_, err := e.Generate(ctx, greedy(cases[4].PromptIDs, 64), func(int32) {
		emitted++
		if emitted == 2 {
			cancel()
		}
	})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Generate returned %v, want context.Canceled", err)
	}
	if emitted < 2 || emitted > 3 {
		t.Errorf("emit was called %d times, want 2 or at most 3", emitted)
	}
}

// The server cancels a generation when it sees a stop string and reports the
// usage of the returned Result.
func TestGenerateReportsUsageWhenCancelled(t *testing.T) {
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	c := cases[4]
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	emitted := 0
	res, err := e.Generate(ctx, greedy(c.PromptIDs, 64), func(int32) {
		emitted++
		if emitted == 5 {
			cancel()
		}
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Generate returned %v, want context.Canceled", err)
	}
	want := engine.Result{PromptTokens: len(c.PromptIDs), CompletionTokens: emitted}
	if res != want {
		t.Errorf("the result of a cancelled generation is %+v, want %+v", res, want)
	}
}

func TestGenerateReportsPromptTokensWhenCancelledBeforeItStarts(t *testing.T) {
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	res, err := e.Generate(ctx, greedy(cases[0].PromptIDs, 64), func(int32) { t.Error("emit was called") })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Generate returned %v, want context.Canceled", err)
	}
	if want := (engine.Result{PromptTokens: len(cases[0].PromptIDs)}); res != want {
		t.Errorf("the result is %+v, want %+v", res, want)
	}
}

func TestGenerateStopsPrefillWhenCancelled(t *testing.T) {
	e := withPrefillStepSize(referenceEngine(t), 1)
	cases, _ := modeltest.Generation(t)
	// 2000 prompt tokens in steps of one take many seconds to feed: the
	// deadline passes while that is under way.
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	prompt := slices.Repeat(cases[0].PromptIDs, 2000/len(cases[0].PromptIDs))
	start := time.Now()
	res, err := e.Generate(ctx, greedy(prompt, 4), func(int32) { t.Error("emit was called") })
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Generate returned %v, want context.DeadlineExceeded", err)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("Generate returned %v after the deadline of 100ms", elapsed)
	}
	if want := (engine.Result{PromptTokens: len(prompt)}); res != want {
		t.Errorf("the result is %+v, want %+v", res, want)
	}
}

func TestGenerateRunsOneAtATime(t *testing.T) {
	e := referenceEngine(t)
	cases, eos := modeltest.Generation(t)
	var wg sync.WaitGroup
	for _, i := range []int{0, 2} {
		wg.Go(func() {
			c := cases[i]
			if !c.EndsWithEOS(eos) {
				t.Errorf("case %d does not end with an end-of-sequence token", i)
				return
			}
			var tokens []int32
			res, err := e.Generate(context.Background(), greedy(c.PromptIDs, 64), func(token int32) { tokens = append(tokens, token) })
			if err != nil {
				t.Errorf("case %d: %v", i, err)
				return
			}
			if want := c.GreedyIDs[:len(c.GreedyIDs)-1]; !slices.Equal(tokens, want) || res.FinishReason != "stop" {
				t.Errorf("case %d: emitted %v and finished %q, want %v and \"stop\"", i, tokens, res.FinishReason, want)
			}
		})
	}
	wg.Wait()
}

func TestGenerateRejectsAnEmptyPrompt(t *testing.T) {
	e := referenceEngine(t)
	_, err := e.Generate(context.Background(), greedy(nil, 4), func(int32) { t.Error("emit was called") })
	if err == nil {
		t.Error("Generate accepted an empty prompt")
	}
}

func TestGenerateWithOneTokenPrompt(t *testing.T) {
	e := referenceEngine(t)
	tokens, res := generate(t, e, greedy([]int32{151644}, 2))
	if len(tokens) != 2 || res.PromptTokens != 1 || res.CompletionTokens != 2 || res.FinishReason != "length" {
		t.Errorf("emitted %v with result %+v, want two tokens for one prompt token", tokens, res)
	}
}

// onThread runs fn on the inference thread of e.
func onThread(t *testing.T, e *Engine, fn func()) {
	t.Helper()
	err := e.thread.Do(context.Background(), func() error {
		fn()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// settledMemory returns MLX's active memory once the work in flight on the
// inference thread is done and its buffers are returned.
func settledMemory(t *testing.T, e *Engine) uint64 {
	t.Helper()
	var active uint64
	onThread(t, e, func() {
		mlx.ClearCache()
		active = mlx.ActiveMemory()
	})
	return active
}

func TestGenerateMemoryIsBoundedAcrossRequests(t *testing.T) {
	const mib = 1 << 20
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	req := greedy(cases[4].PromptIDs, 64)
	req.Sampling = engine.Sampling{RepetitionPenalty: 1.05, RepetitionContextSize: 20} // keeps a token history too

	generate(t, e, req)
	afterFirst := settledMemory(t, e)
	for range 9 {
		generate(t, e, req)
	}
	afterTenth := settledMemory(t, e)

	t.Logf("active memory: %d bytes after the first generation, %d after the tenth", afterFirst, afterTenth)
	if diff := int64(afterTenth) - int64(afterFirst); diff > 16*mib || diff < -16*mib {
		t.Errorf("active memory moved by %d bytes over nine more generations, want within 16 MiB", diff)
	}
}

func TestGenerateFreesItsStateWhenItFailsOrIsCancelled(t *testing.T) {
	const mib = 1 << 20
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	generate(t, e, greedy(cases[0].PromptIDs, 4))
	before := settledMemory(t, e)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if _, err := e.Generate(ctx, greedy(cases[4].PromptIDs, 64), func(int32) { cancel() }); !errors.Is(err, context.Canceled) {
		t.Fatalf("Generate returned %v, want context.Canceled", err)
	}
	tooLarge := engine.Request{Prompt: cases[4].PromptIDs, MaxTokens: 8, Sampling: engine.Sampling{Temperature: 1, TopK: 1 << 20}}
	if _, err := e.Generate(context.Background(), tooLarge, func(int32) {}); err == nil {
		t.Fatal("Generate accepted a top-k larger than the vocabulary")
	}

	after := settledMemory(t, e)
	if diff := int64(after) - int64(before); diff > mib || diff < -mib {
		t.Errorf("active memory is %d bytes off the level before the two failed generations, want within 1 MiB", diff)
	}
}

// peakMemory returns the peak of MLX's active memory while e runs req, and
// the tokens it emits.
func peakMemory(t *testing.T, e *Engine, req engine.Request) (uint64, []int32) {
	t.Helper()
	onThread(t, e, func() {
		mlx.ClearCache()
		mlx.ResetPeakMemory()
	})
	tokens, _ := generate(t, e, req)
	var peak uint64
	onThread(t, e, func() { peak = mlx.PeakMemory() })
	return peak, tokens
}

// wantPeak fails the test unless peak is within 10% of the reference's.
func wantPeak(t *testing.T, peak, reference uint64) {
	t.Helper()
	ratio := float64(peak) / float64(reference)
	t.Logf("peak memory %d bytes, reference %d bytes: %.1f%% of the reference", peak, reference, 100*ratio)
	if ratio < 0.9 || ratio > 1.1 {
		t.Errorf("peak memory is %d bytes, want within 10%% of the reference's %d", peak, reference)
	}
}

func TestPeakMemoryCloseToReference(t *testing.T) {
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	peak, _ := peakMemory(t, e, greedy(cases[0].PromptIDs, 64))
	wantPeak(t, peak, cases[0].PeakMemory)
}

// A long prompt shows whether the activations of a forward pass are released
// while it is evaluated, layer after layer, or pile up until its end.
//
// The reference peak is mx.get_peak_memory() of mlx-lm 0.32.2 on MLX 0.32.4
// after mx.clear_cache(), mx.reset_peak_memory() and four tokens of
// generate_step on the prompt of case 0 repeated 27 times (999 tokens).
func TestPeakMemoryOfALongPromptCloseToReference(t *testing.T) {
	const referencePeak = 1631878192
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	peak, tokens := peakMemory(t, e, greedy(slices.Repeat(cases[0].PromptIDs, 27), 4))
	if want := []int32{16, 11, 220, 17}; !slices.Equal(tokens, want) {
		t.Errorf("emitted %v, want %v as the reference", tokens, want)
	}
	wantPeak(t, peak, referencePeak)
}

func TestGenerateReportsSamplingErrorsAndStaysUsable(t *testing.T) {
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	c := cases[1]
	tooLarge := engine.Request{Prompt: c.PromptIDs, MaxTokens: 8, Sampling: engine.Sampling{Temperature: 1, TopK: 1 << 20}}
	res, err := e.Generate(context.Background(), tooLarge, func(int32) { t.Error("emit was called") })
	if err == nil || !strings.Contains(err.Error(), "top_k") {
		t.Errorf("Generate with a top-k larger than the vocabulary: error = %v, want one about top_k", err)
	}
	if want := (engine.Result{PromptTokens: len(c.PromptIDs)}); res != want {
		t.Errorf("the result is %+v, want %+v", res, want)
	}
	negative := engine.Request{Prompt: c.PromptIDs, MaxTokens: 8, Sampling: engine.Sampling{RepetitionPenalty: -1}}
	if _, err := e.Generate(context.Background(), negative, func(int32) { t.Error("emit was called") }); err == nil {
		t.Error("Generate accepted a negative repetition penalty")
	}

	tokens, _ := generate(t, e, greedy(c.PromptIDs, 64))
	if want := c.GreedyIDs[:len(c.GreedyIDs)-1]; !slices.Equal(tokens, want) {
		t.Errorf("after the failed requests the engine emitted %v, want %v", tokens, want)
	}
}

// Expected tokens come from generate_step of mlx-lm 0.32.2 on the prompt of
// case 0, "Count from 1 to 5.", with
// make_logits_processors(repetition_penalty=1.3, repetition_context_size=20).
// The prompt holds the token "1" (16) that would start the answer; the
// penalty reaches it only because the prompt is part of the history. mlx-lm
// 0.31.3 kept the prompt out of the history and answers "1, 2, ..." here.
func TestPenaltiesSeeThePrompt(t *testing.T) {
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	c := cases[0]
	if c.GreedyIDs[0] != 16 || !slices.Contains(c.PromptIDs[len(c.PromptIDs)-20:], 16) {
		t.Fatalf("case 0 no longer starts its answer with a token of its prompt")
	}
	req := engine.Request{Prompt: c.PromptIDs, MaxTokens: 16,
		Sampling: engine.Sampling{RepetitionPenalty: 1.3, RepetitionContextSize: 20}}
	want := []int32{8420, 525, 279, 5109, 498, 11223, 1447, 12, 3070, 2507, 504, 334, 2303, 220, 481, 5542}
	for _, stepSize := range []int{2048, 8} {
		if tokens, _ := generate(t, withPrefillStepSize(e, stepSize), req); !slices.Equal(tokens, want) {
			t.Errorf("prefill step %d: emitted %v, want %v", stepSize, tokens, want)
		}
	}
}

// Expected tokens come from generate_step of mlx-lm 0.32.2 on the prompt of
// case 0 with make_logits_processors(logit_bias={16: -2.0, 8420: 1.0},
// repetition_penalty=1.1, repetition_context_size=20, presence_penalty=0.5,
// presence_context_size=30, frequency_penalty=0.3, frequency_context_size=40).
func TestAllProcessorsTogetherMatchReference(t *testing.T) {
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	req := engine.Request{Prompt: cases[0].PromptIDs, MaxTokens: 16, Sampling: engine.Sampling{
		LogitBias:             map[int32]float32{16: -2, 8420: 1},
		RepetitionPenalty:     1.1,
		RepetitionContextSize: 20,
		PresencePenalty:       0.5,
		PresenceContextSize:   30,
		FrequencyPenalty:      0.3,
		FrequencyContextSize:  40,
	}}
	want := []int32{8420, 374, 279, 1760, 504, 220, 16, 311, 220, 20, 1447, 12, 3070, 2507, 287, 66963}
	if tokens, _ := generate(t, e, req); !slices.Equal(tokens, want) {
		t.Errorf("emitted %v, want %v", tokens, want)
	}
}

// Expected tokens come from mx.random.seed(7) followed by generate_step of
// mlx-lm 0.32.2 on the prompt of case 0 with
// make_sampler(temp=0.8, top_p=0.95, top_k=40, min_p=0.02), without and with
// make_logits_processors(repetition_penalty=1.2, repetition_context_size=20).
func TestSeededSamplingMatchesReference(t *testing.T) {
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	seed := uint64(7)
	sampling := engine.Sampling{Temperature: 0.8, TopP: 0.95, TopK: 40, MinP: 0.02, Seed: &seed}
	req := engine.Request{Prompt: cases[0].PromptIDs, MaxTokens: 24, Sampling: sampling}

	tokens, res := generate(t, e, req)
	if want := []int32{16, 198, 17, 198, 18, 198, 19, 198, 20}; !slices.Equal(tokens, want) || res.FinishReason != "stop" {
		t.Errorf("emitted %v and finished %q, want %v and \"stop\"", tokens, res.FinishReason, want)
	}

	req.Sampling.RepetitionPenalty, req.Sampling.RepetitionContextSize = 1.2, 20
	tokens, _ = generate(t, e, req)
	want := []int32{40, 36879, 11, 714, 438, 264, 4128, 1614, 3465, 553, 54364, 14817, 1207, 16948, 311, 4226, 1196, 19556, 389, 5257, 13347, 1741, 438, 2562}
	if !slices.Equal(tokens, want) {
		t.Errorf("with a repetition penalty: emitted %v, want %v", tokens, want)
	}
}

func TestEndOfSequenceTokensComeFromTheConfigFirst(t *testing.T) {
	fromOptions := Options{EOSTokenIDs: []int32{1, 2}}
	if got := endOfSequence(models.Config{EOSTokenIDs: []int32{7}}, fromOptions); !slices.Equal(got, []int32{7}) {
		t.Errorf("with eos_token_id in config.json the end-of-sequence tokens are %v, want [7]", got)
	}
	if got := endOfSequence(models.Config{}, fromOptions); !slices.Equal(got, []int32{1, 2}) {
		t.Errorf("without eos_token_id in config.json the end-of-sequence tokens are %v, want those of the options", got)
	}
}

func TestGenerateAfterCloseFails(t *testing.T) {
	// A closed engine has no use for a model: an engine with only a thread
	// stands in for one.
	e := &Engine{thread: exec.Start()}
	e.Close()
	_, err := e.Generate(context.Background(), greedy([]int32{1}, 1), func(int32) { t.Error("emit was called") })
	if !errors.Is(err, exec.ErrClosed) {
		t.Errorf("Generate after Close returned %v, want exec.ErrClosed", err)
	}
}

// MLX does not check indices: a token outside the vocabulary would be read
// from outside the embedding table and, with a penalty, written outside the
// logits.
func TestGenerateRejectsPromptTokensOutsideTheVocabulary(t *testing.T) {
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	valid := cases[1].PromptIDs
	for _, id := range []int32{151936, 2147483647, -1, -151937, -2147483648} {
		for _, position := range []int{0, 5, len(valid) - 1} {
			for name, sampling := range map[string]engine.Sampling{
				"greedy":       {},
				"with penalty": {RepetitionPenalty: 1.2, RepetitionContextSize: 20},
			} {
				prompt := slices.Clone(valid)
				prompt[position] = id
				req := engine.Request{Prompt: prompt, MaxTokens: 4, Sampling: sampling}
				res, err := e.Generate(context.Background(), req, func(int32) { t.Error("emit was called") })
				if err == nil || !strings.Contains(err.Error(), strconv.Itoa(int(id))) ||
					!strings.Contains(err.Error(), "position "+strconv.Itoa(position)) || !strings.Contains(err.Error(), "151936") {
					t.Errorf("token %d at position %d, %s: error = %v, want one naming the token, its position and the vocabulary size",
						id, position, name, err)
				}
				if want := (engine.Result{PromptTokens: len(prompt)}); res != want {
					t.Errorf("token %d at position %d, %s: result is %+v, want %+v", id, position, name, res, want)
				}
			}
		}
	}
	// The last token of the vocabulary is a valid prompt, and the engine
	// still works.
	if _, err := e.Generate(context.Background(), greedy([]int32{0, 151935}, 1), func(int32) {}); err != nil {
		t.Errorf("a prompt of the first and the last token of the vocabulary: %v", err)
	}
	tokens, _ := generate(t, e, greedy(valid, 64))
	if want := cases[1].GreedyIDs[:len(cases[1].GreedyIDs)-1]; !slices.Equal(tokens, want) {
		t.Errorf("after the rejected prompts the engine emitted %v, want %v", tokens, want)
	}
}

// When emit panics, the token after the one it was given is already being
// computed. It must be freed with the rest of the generation.
func TestGenerateFreesTheTokenInFlightWhenEmitPanics(t *testing.T) {
	e := referenceEngine(t)
	cases, _ := modeltest.Generation(t)
	c := cases[4]
	generate(t, e, greedy(c.PromptIDs, 4))
	before := settledMemory(t, e)

	for range 50 {
		_, err := e.Generate(context.Background(), greedy(c.PromptIDs, 8), func(int32) { panic("boom") })
		var panicked *exec.PanicError
		if !errors.As(err, &panicked) || panicked.Value != "boom" {
			t.Fatalf("Generate returned %v, want the panic of emit", err)
		}
	}

	if after := settledMemory(t, e); after != before {
		t.Errorf("50 generations whose emit panicked left %d bytes behind", int64(after)-int64(before))
	}
	tokens, _ := generate(t, e, greedy(c.PromptIDs, 8))
	if !slices.Equal(tokens, c.GreedyIDs[:8]) {
		t.Errorf("after the panics the engine emitted %v, want %v", tokens, c.GreedyIDs[:8])
	}
}

// One evaluation cannot be interrupted: while the prompt is fed, a
// cancellation takes effect at the end of the prefill step under way.
func TestGenerateStopsAtTheNextPrefillStepWhenCancelled(t *testing.T) {
	e := referenceEngine(t)
	if e.prefillStepSize != 2048 {
		t.Fatalf("the prefill step size is %d, want the default 2048", e.prefillStepSize)
	}
	cases, _ := modeltest.Generation(t)
	// Three prefill steps and the token the first sample starts from.
	prompt := slices.Repeat(cases[0].PromptIDs, 3*2048/len(cases[0].PromptIDs)+1)[:3*2048+1]

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	counting := &countingModel{Model: e.model, onForward: func(n int) {
		if n == 1 {
			cancel() // during the first step
		}
	}}
	counted := &Engine{thread: e.thread, model: counting, vocab: e.vocab, eos: e.eos, prefillStepSize: e.prefillStepSize}
	res, err := counted.Generate(ctx, greedy(prompt, 4), func(int32) { t.Error("emit was called") })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Generate returned %v, want context.Canceled", err)
	}
	if counting.forwards != 1 {
		t.Errorf("%d prefill steps ran after the cancellation during the first, want only the first", counting.forwards)
	}
	if want := (engine.Result{PromptTokens: len(prompt)}); res != want {
		t.Errorf("the result is %+v, want %+v", res, want)
	}
}

// activeMemory returns MLX's active memory. It is for tests that have no
// engine open: nothing else may use MLX meanwhile.
func activeMemory(t *testing.T) uint64 {
	t.Helper()
	modeltest.Pin(t)
	return mlx.ActiveMemory()
}

func TestCloseFreesTheModel(t *testing.T) {
	const mib = 1 << 20
	dir := modeltest.Dir(t)
	closeReference()
	before := activeMemory(t)
	for cycle := range 3 {
		e, err := Load(dir, Options{})
		if err != nil {
			t.Fatal(err)
		}
		loaded := activeMemory(t)
		e.Close()
		after := activeMemory(t)
		t.Logf("cycle %d: active memory %d bytes before Load, %d loaded, %d after Close", cycle, before, loaded, after)
		if loaded < before+500*mib {
			t.Errorf("cycle %d: the loaded model holds %d bytes, want hundreds of MiB", cycle, int64(loaded)-int64(before))
		}
		if diff := int64(after) - int64(before); diff > mib || diff < -mib {
			t.Fatalf("cycle %d: active memory after Close is %d bytes off the level before Load, want within 1 MiB", cycle, diff)
		}
	}
}

func TestCloseTwiceIsHarmless(t *testing.T) {
	closeReference()
	e, err := Load(modeltest.WriteModel(t, modeltest.TinyConfig(), modeltest.TinyDense()), Options{})
	if err != nil {
		t.Fatal(err)
	}
	e.Close()
	e.Close()
	if _, err := e.Generate(context.Background(), greedy([]int32{1}, 1), func(int32) { t.Error("emit was called") }); !errors.Is(err, exec.ErrClosed) {
		t.Errorf("Generate after Close returned %v, want exec.ErrClosed", err)
	}
}

// Close lets the generation in progress finish and fails those that wait
// behind it, which must not run on the freed model.
func TestCloseFinishesTheRunningGenerationAndFailsTheQueued(t *testing.T) {
	e := referenceEngine(t)
	reference = nil // closed below
	cases, _ := modeltest.Generation(t)
	c := cases[4]

	started := make(chan struct{})
	var wg sync.WaitGroup
	var running []int32
	var runningErr, queuedErr error
	wg.Go(func() {
		_, runningErr = e.Generate(context.Background(), greedy(c.PromptIDs, 64), func(token int32) {
			if len(running) == 0 {
				close(started)
			}
			running = append(running, token)
		})
	})
	<-started
	wg.Go(func() {
		_, queuedErr = e.Generate(context.Background(), greedy(c.PromptIDs, 64), func(int32) { t.Error("the queued generation emitted a token") })
	})
	time.Sleep(20 * time.Millisecond) // lets the second call queue up; it must fail either way
	e.Close()
	wg.Wait()

	if runningErr != nil || !slices.Equal(running, c.GreedyIDs) {
		t.Errorf("the generation in progress returned %v after %v, want all of %v", runningErr, running, c.GreedyIDs)
	}
	if !errors.Is(queuedErr, exec.ErrClosed) {
		t.Errorf("the queued generation returned %v, want exec.ErrClosed", queuedErr)
	}
}

// tinyEngine closes the shared engine and loads a tiny model written for
// this test. The engine is closed when the test ends.
func tinyEngine(t *testing.T, config map[string]any, tensors map[string]modeltest.Tensor) *Engine {
	t.Helper()
	closeReference()
	e, err := Load(modeltest.WriteModel(t, config, tensors), Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(e.Close)
	return e
}

// Expected tokens come from generate_step of mlx-lm 0.32.2 on the same model
// directories, loaded with mlx_lm.utils.load_model: greedy with prefill steps
// of 2048 and of 3,
// make_logits_processors(repetition_penalty=1.5, repetition_context_size=20),
// and mx.random.seed(11) with make_sampler(temp=1.0, top_p=0.9, top_k=10).
// Token 31 ends a generation.
func TestTinyModelsMatchReference(t *testing.T) {
	prompt := []int32{1, 5, 9, 2, 30, 7, 7, 0}
	seed := uint64(11)
	tests := []struct {
		name                  string
		config                map[string]any
		tensors               map[string]modeltest.Tensor
		greedy, rep, sampled  []int32
		greedyEnds, sampleEnd bool // whether the reference sampled token 31 after them
	}{
		{"dense", modeltest.TinyConfig(), modeltest.TinyDense(),
			[]int32{30, 29, 13, 25, 4, 0, 30, 29, 23, 25},
			[]int32{30, 29, 13, 25, 4, 14, 27, 27, 27, 23, 20, 14, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27},
			[]int32{30, 23, 13, 8, 7, 20, 14, 29, 27, 27, 29, 23, 8},
			true, true},
		{"quantized", modeltest.TinyQuantizedConfig(), modeltest.TinyQuantized(),
			[]int32{29, 23, 1, 23, 1, 23, 1, 23, 1, 23, 1, 23, 1, 15, 29, 23, 1, 23, 1, 23, 1, 23, 1, 23},
			[]int32{29, 23, 1, 15, 12, 29, 23, 27, 23, 1, 23, 1, 23, 1, 23, 1, 23, 1, 23, 7, 14, 15, 29, 13},
			[]int32{22, 23, 1, 12, 15, 25, 12, 27, 25, 14, 29, 15, 4, 29, 22, 15, 3, 12, 7, 25, 12, 29, 25, 29},
			false, false},
	}
	reason := func(ends bool) string {
		if ends {
			return "stop"
		}
		return "length"
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := tinyEngine(t, tt.config, tt.tensors)
			if e.vocab != modeltest.TinyVocabSize {
				t.Errorf("the engine knows %d tokens, want the %d rows of the embedding", e.vocab, modeltest.TinyVocabSize)
			}
			for name, c := range map[string]struct {
				engine     *Engine
				sampling   engine.Sampling
				want       []int32
				wantReason string
			}{
				"greedy":              {e, engine.Sampling{}, tt.greedy, reason(tt.greedyEnds)},
				"greedy, prefill 3":   {withPrefillStepSize(e, 3), engine.Sampling{}, tt.greedy, reason(tt.greedyEnds)},
				"repetition penalty":  {e, engine.Sampling{RepetitionPenalty: 1.5, RepetitionContextSize: 20}, tt.rep, "length"},
				"seeded, top-p and k": {e, engine.Sampling{Temperature: 1, TopP: 0.9, TopK: 10, Seed: &seed}, tt.sampled, reason(tt.sampleEnd)},
			} {
				tokens, res := generate(t, c.engine, engine.Request{Prompt: prompt, MaxTokens: 24, Sampling: c.sampling})
				if !slices.Equal(tokens, c.want) || res.FinishReason != c.wantReason {
					t.Errorf("%s: emitted %v and finished %q, want %v and %q", name, tokens, res.FinishReason, c.want, c.wantReason)
				}
			}
			// The vocabulary is the embedding's: token 32 is one past its end.
			_, err := e.Generate(context.Background(), greedy([]int32{1, modeltest.TinyVocabSize}, 2), func(int32) {})
			if err == nil || !strings.Contains(err.Error(), "prompt token 32 at position 1 is outside the vocabulary of 32 tokens") {
				t.Errorf("a prompt with token 32: error = %v, want it rejected as outside the vocabulary", err)
			}
		})
	}
}

// A model whose tensors do not fit its config.json must not load, and the
// failed Load must leave the way free for the next one.
func TestLoadRejectsAModelOfTheWrongShape(t *testing.T) {
	closeReference()
	tensors := modeltest.TinyDense()
	tensors["model.layers.0.mlp.gate_proj.weight"] = modeltest.Zeros("F32", 64, 64)
	_, err := Load(modeltest.WriteModel(t, modeltest.TinyConfig(), tensors), Options{})
	want := `tensor "model.layers.0.mlp.gate_proj.weight" has shape [64 64], want [128 64]`
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("Load error = %v, want one saying: %s", err, want)
	}
	tinyEngine(t, modeltest.TinyConfig(), modeltest.TinyDense())
}
