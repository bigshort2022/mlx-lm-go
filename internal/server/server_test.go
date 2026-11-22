package server

import "testing"

func TestUnknownPathsAreNotFound(t *testing.T) {
	for _, name := range []string{"get_unknown_path", "post_unknown_path"} {
		t.Run(name, func(t *testing.T) {
			rec := loadExchange(t, name)
			h := newHarness(t)
			got := h.replay(rec)
			if got.Code != rec.Response.Status {
				t.Fatalf("status = %d, want %d", got.Code, rec.Response.Status)
			}
			if got.Body.String() != rec.Response.Body {
				t.Fatalf("body = %q, want %q", got.Body, rec.Response.Body)
			}
			if n := h.engine.Calls.Load(); n != 0 {
				t.Fatalf("engine was called %d times", n)
			}
		})
	}
}

func TestKnownPathWithWrongMethodIsNotFound(t *testing.T) {
	got := newHarness(t).do("POST", "/health", `{}`, nil)
	if got.Code != 404 || got.Body.String() != "Not Found" {
		t.Fatalf("POST /health = %d %q, want 404 \"Not Found\"", got.Code, got.Body)
	}
}

func TestChatCompletionsIsAlsoServedWithoutTheVersionPrefix(t *testing.T) {
	// The reference routes /chat/completions to the same handler.
	h := newHarness(t, scripted(countPieces...))
	rec := loadExchange(t, "chat")
	got := h.do("POST", "/chat/completions", rec.requestBody(), nil)
	if got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}
	if object := decodeCompletion(t, got.Body.String()).Object; object != "chat.completion" {
		t.Fatalf("object = %q, want chat.completion", object)
	}
}
