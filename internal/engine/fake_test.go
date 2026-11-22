package engine

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"
)

// collect runs Generate and returns the tokens it emitted.
func collect(t *testing.T, e Engine, ctx context.Context, req Request) ([]int32, Result, error) {
	t.Helper()
	var got []int32
	res, err := e.Generate(ctx, req, func(token int32) { got = append(got, token) })
	return got, res, err
}

func TestFakeEmitsScriptInOrderAndStops(t *testing.T) {
	f := &Fake{Tokens: []int32{7, 8, 9}}
	got, res, err := collect(t, f, context.Background(), Request{Prompt: []int32{1, 2}, MaxTokens: 10})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if want := []int32{7, 8, 9}; !slices.Equal(got, want) {
		t.Fatalf("emitted %v, want %v", got, want)
	}
	want := Result{PromptTokens: 2, CompletionTokens: 3, FinishReason: "stop"}
	if res != want {
		t.Fatalf("result = %+v, want %+v", res, want)
	}
}

func TestFakeStopsAtMaxTokensWithLength(t *testing.T) {
	f := &Fake{Tokens: []int32{7, 8, 9}}
	got, res, err := collect(t, f, context.Background(), Request{Prompt: []int32{1}, MaxTokens: 2})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if want := []int32{7, 8}; !slices.Equal(got, want) {
		t.Fatalf("emitted %v, want %v", got, want)
	}
	want := Result{PromptTokens: 1, CompletionTokens: 2, FinishReason: "length"}
	if res != want {
		t.Fatalf("result = %+v, want %+v", res, want)
	}
}

func TestFakeBudgetEqualToScriptStillStops(t *testing.T) {
	f := &Fake{Tokens: []int32{7, 8}}
	_, res, err := collect(t, f, context.Background(), Request{MaxTokens: 2})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if res.FinishReason != "stop" {
		t.Fatalf("finish reason = %q, want stop", res.FinishReason)
	}
}

func TestFakeReturnsCanceledWhileBlockedInGate(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	entered := make(chan struct{})
	release := make(chan struct{})
	defer close(release)
	f := &Fake{Tokens: []int32{7, 8, 9}, Gate: func(i int, token int32) {
		if i == 1 {
			close(entered)
			<-release
		}
	}}

	var got []int32
	var res Result
	done := make(chan error, 1)
	go func() {
		var err error
		got, res, err = collect(t, f, ctx, Request{Prompt: []int32{1, 2}, MaxTokens: 10})
		done <- err
	}()
	<-entered
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Generate error = %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Generate did not return after the context was cancelled")
	}
	if want := []int32{7}; !slices.Equal(got, want) {
		t.Fatalf("emitted %v, want %v", got, want)
	}
	if want := (Result{PromptTokens: 2, CompletionTokens: 1}); res != want {
		t.Fatalf("result = %+v, want %+v", res, want)
	}
}

func TestFakeRecordsCallsAndLastRequest(t *testing.T) {
	f := &Fake{Tokens: []int32{7}}
	if f.Calls.Load() != 0 || f.Last.Load() != nil {
		t.Fatalf("fresh fake: Calls = %d, Last = %v", f.Calls.Load(), f.Last.Load())
	}
	first := Request{Prompt: []int32{1}, MaxTokens: 1}
	second := Request{Prompt: []int32{2, 3}, MaxTokens: 4, Sampling: Sampling{Temperature: 0.5, TopK: 40}}
	for _, req := range []Request{first, second} {
		if _, _, err := collect(t, f, context.Background(), req); err != nil {
			t.Fatalf("Generate: %v", err)
		}
	}
	if got := f.Calls.Load(); got != 2 {
		t.Fatalf("Calls = %d, want 2", got)
	}
	last := f.Last.Load()
	if last == nil || !slices.Equal(last.Prompt, second.Prompt) || last.MaxTokens != 4 || last.Sampling.Temperature != 0.5 || last.Sampling.TopK != 40 {
		t.Fatalf("Last = %+v, want %+v", last, second)
	}
}
