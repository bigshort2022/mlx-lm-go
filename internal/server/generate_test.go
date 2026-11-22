package server

import (
	"context"
	"errors"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
)

// failAfterFirstToken is an engine that emits one token and then fails.
func failAfterFirstToken(engine.Engine) engine.Engine {
	return engineFunc(func(_ context.Context, req engine.Request, emit func(int32)) (engine.Result, error) {
		emit(0)
		return engine.Result{PromptTokens: len(req.Prompt), CompletionTokens: 1}, errors.New("out of memory")
	})
}

func TestEngineFailureIsAServerError(t *testing.T) {
	h := newHarness(t, scripted(countPieces...), func(h *harness) { h.wrap = failAfterFirstToken })
	got := h.replay(loadExchange(t, "chat"))
	if msg := errorMessage(t, got, 500); msg != "out of memory" {
		t.Fatalf("error = %q, want the engine error", msg)
	}
}

func TestEngineFailureMidStreamEndsWithAnErrorEvent(t *testing.T) {
	h := newHarness(t, scripted(countPieces...), func(h *harness) { h.wrap = failAfterFirstToken })
	got := h.replay(loadExchange(t, "chat_stream"))
	events := sseData(t, got.Body.String())
	if len(events) != 2 {
		t.Fatalf("got %d events, want the first chunk and an error:\n%s", len(events), got.Body)
	}
	if content := decodeCompletion(t, events[0]).Choices[0].Delta.Content; content != "1" {
		t.Errorf("first delta = %q, want %q", content, "1")
	}
	if !jsonEqual(t, events[1], `{"error": "out of memory"}`) {
		t.Errorf("last event = %s, want the engine error", events[1])
	}
}
