package server

import (
	"context"
	"errors"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
)

var uuidV4 = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)

func TestChatMatchesRecording(t *testing.T) {
	rec := loadExchange(t, "chat")
	h := newHarness(t, scripted(countPieces...))
	got := h.replay(rec)

	if got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}
	if ct := got.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", ct)
	}
	if g, w := shapeOf(t, got.Body.String()), shapeOf(t, rec.Response.Body); !reflect.DeepEqual(g, w) {
		t.Fatalf("shape =\n%v, want\n%v", g, w)
	}
	body := decodeCompletion(t, got.Body.String())
	if body.Object != "chat.completion" {
		t.Errorf("object = %q, want chat.completion", body.Object)
	}
	if id, ok := strings.CutPrefix(body.ID, "chatcmpl-"); !ok || !uuidV4.MatchString(id) {
		t.Errorf("id = %q, want chatcmpl-<uuid4>", body.ID)
	}
	if body.Model != "default_model" {
		t.Errorf("model = %q, want default_model", body.Model)
	}
	if fr := body.finish(t); fr != "stop" {
		t.Errorf("finish_reason = %q, want stop", fr)
	}
	if msg := body.Choices[0].Message; msg == nil || msg.Role != "assistant" || msg.Content != "1, 2, 3, 4, 5" {
		t.Errorf("message = %+v, want assistant %q", msg, "1, 2, 3, 4, 5")
	}
	promptTokens := len(h.tok.Encode(h.tmpl.out))
	if u := body.Usage; u.PromptTokens != promptTokens || u.CompletionTokens != len(countPieces) || u.TotalTokens != promptTokens+len(countPieces) {
		t.Errorf("usage = %+v, want %d prompt and %d completion tokens and their sum", *u, promptTokens, len(countPieces))
	}
}

func TestChatStopsAtMaxTokens(t *testing.T) {
	rec := loadExchange(t, "chat_length")
	h := newHarness(t, scripted(countPieces...))
	got := h.replay(rec)

	if got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}
	if g, w := shapeOf(t, got.Body.String()), shapeOf(t, rec.Response.Body); !reflect.DeepEqual(g, w) {
		t.Fatalf("shape =\n%v, want\n%v", g, w)
	}
	body := decodeCompletion(t, got.Body.String())
	if fr := body.finish(t); fr != "length" {
		t.Errorf("finish_reason = %q, want length", fr)
	}
	if content := body.Choices[0].Message.Content; content != "1" {
		t.Errorf("content = %q, want %q", content, "1")
	}
	if body.Usage.CompletionTokens != 1 {
		t.Errorf("completion_tokens = %d, want 1", body.Usage.CompletionTokens)
	}
}

func TestChatPromptIsTheEncodedTemplateOutput(t *testing.T) {
	rec := loadExchange(t, "chat")
	h := newHarness(t, scripted(countPieces...), func(h *harness) { h.tmpl.out = "<rendered chat>" })
	if got := h.replay(rec); got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}

	if want := `[{"role": "user", "content": "Count from 1 to 5."}]`; !jsonEqual(t, h.tmpl.messages, want) {
		t.Errorf("template messages = %s, want %s", h.tmpl.messages, want)
	}
	if h.tmpl.tools != "null" {
		t.Errorf("template tools = %q, want JSON null", h.tmpl.tools)
	}
	if !jsonEqual(t, h.tmpl.kwargs, `{}`) {
		t.Errorf("template kwargs = %s, want {}", h.tmpl.kwargs)
	}
	if !h.tmpl.addGenerationPrompt {
		t.Error("template was rendered without the generation prompt")
	}
	last := h.engine.Last.Load()
	if want := h.tok.Encode("<rendered chat>"); !slices.Equal(last.Prompt, want) {
		t.Errorf("engine prompt = %v, want %v", last.Prompt, want)
	}
	if last.MaxTokens != 24 {
		t.Errorf("engine MaxTokens = %d, want 24", last.MaxTokens)
	}
}

func TestChatTemplateKwargsMergeOverConfiguredArgs(t *testing.T) {
	configured := func(h *harness) {
		h.cfg.ChatTemplateArgs = []byte(`{"enable_thinking": true, "effort": "low"}`)
	}

	t.Run("request keys win", func(t *testing.T) {
		rec := loadExchange(t, "chat_template_kwargs")
		h := newHarness(t, scripted(countPieces...), configured)
		if got := h.replay(rec); got.Code != 200 {
			t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
		}
		if want := `{"enable_thinking": false, "effort": "low"}`; !jsonEqual(t, h.tmpl.kwargs, want) {
			t.Fatalf("template kwargs = %s, want %s", h.tmpl.kwargs, want)
		}
	})

	t.Run("configured args alone", func(t *testing.T) {
		h := newHarness(t, scripted(countPieces...), configured)
		if got := h.replay(loadExchange(t, "chat")); got.Code != 200 {
			t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
		}
		if want := `{"enable_thinking": true, "effort": "low"}`; !jsonEqual(t, h.tmpl.kwargs, want) {
			t.Fatalf("template kwargs = %s, want %s", h.tmpl.kwargs, want)
		}
	})

	t.Run("request kwargs alone", func(t *testing.T) {
		h := newHarness(t, scripted(countPieces...))
		if got := h.replay(loadExchange(t, "chat_template_kwargs")); got.Code != 200 {
			t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
		}
		if want := `{"enable_thinking": false}`; !jsonEqual(t, h.tmpl.kwargs, want) {
			t.Fatalf("template kwargs = %s, want %s", h.tmpl.kwargs, want)
		}
	})
}

func TestChatJoinsTextContentParts(t *testing.T) {
	rec := loadExchange(t, "chat_content_parts")
	h := newHarness(t, scripted("Hello", "!", " How", " can"))
	got := h.replay(rec)
	if got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}
	if g, w := shapeOf(t, got.Body.String()), shapeOf(t, rec.Response.Body); !reflect.DeepEqual(g, w) {
		t.Fatalf("shape =\n%v, want\n%v", g, w)
	}
	if want := `[{"role": "user", "content": "Say hi."}]`; !jsonEqual(t, h.tmpl.messages, want) {
		t.Fatalf("template messages = %s, want %s", h.tmpl.messages, want)
	}
}

func TestChatNormalisesMessageContent(t *testing.T) {
	const request = `{"messages": [
		{"role": "system", "content": "Be brief.", "name": "sys"},
		{"role": "assistant", "content": null},
		{"role": "assistant"},
		{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
	]}`
	const want = `[
		{"role": "system", "content": "Be brief.", "name": "sys"},
		{"role": "assistant", "content": ""},
		{"role": "assistant", "content": ""},
		{"role": "user", "content": "ab"}
	]`
	h := newHarness(t, scripted("ok"))
	if got := h.do("POST", "/v1/chat/completions", request, nil); got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}
	if !jsonEqual(t, h.tmpl.messages, want) {
		t.Fatalf("template messages = %s, want %s", h.tmpl.messages, want)
	}
}

func TestChatSamplingFieldsReachTheEngine(t *testing.T) {
	rec := loadExchange(t, "chat_sampling")
	h := newHarness(t, scripted(countPieces...))
	if got := h.replay(rec); got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}
	want := engine.Sampling{
		Temperature:           0.7,
		TopP:                  0.9,
		TopK:                  40,
		MinP:                  0.05,
		RepetitionPenalty:     1.1,
		RepetitionContextSize: 20,
		PresencePenalty:       0.1,
		PresenceContextSize:   20,
		FrequencyPenalty:      0.1,
		FrequencyContextSize:  20,
		LogitBias:             map[int32]float32{16: -100},
	}
	last := h.engine.Last.Load()
	if !reflect.DeepEqual(last.Sampling, want) {
		t.Errorf("sampling =\n%+v, want\n%+v", last.Sampling, want)
	}
	if last.MaxTokens != 8 {
		t.Errorf("MaxTokens = %d, want 8", last.MaxTokens)
	}
}

func TestChatSamplingDefaultsComeFromConfig(t *testing.T) {
	h := newHarness(t, scripted(countPieces...), func(h *harness) {
		h.cfg.Temperature, h.cfg.TopP, h.cfg.TopK, h.cfg.MinP, h.cfg.MaxTokens = 0.3, 0.8, 5, 0.02, 77
	})
	if got := h.do("POST", "/v1/chat/completions", `{"messages": []}`, nil); got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}
	want := engine.Sampling{
		Temperature:           0.3,
		TopP:                  0.8,
		TopK:                  5,
		MinP:                  0.02,
		RepetitionContextSize: 20,
		PresenceContextSize:   20,
		FrequencyContextSize:  20,
	}
	last := h.engine.Last.Load()
	if !reflect.DeepEqual(last.Sampling, want) {
		t.Errorf("sampling =\n%+v, want\n%+v", last.Sampling, want)
	}
	if last.MaxTokens != 77 {
		t.Errorf("MaxTokens = %d, want 77", last.MaxTokens)
	}
}

func TestChatContextSizesSeedAndMaxCompletionTokens(t *testing.T) {
	const request = `{"messages": [], "repetition_context_size": 7, "presence_context_size": 8,
		"frequency_context_size": 9, "seed": 42, "max_tokens": 5, "max_completion_tokens": 3}`
	h := newHarness(t, scripted(countPieces...))
	if got := h.do("POST", "/v1/chat/completions", request, nil); got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}
	last := h.engine.Last.Load()
	s := last.Sampling
	if s.RepetitionContextSize != 7 || s.PresenceContextSize != 8 || s.FrequencyContextSize != 9 {
		t.Errorf("context sizes = %d, %d, %d, want 7, 8, 9", s.RepetitionContextSize, s.PresenceContextSize, s.FrequencyContextSize)
	}
	if s.Seed == nil || *s.Seed != 42 {
		t.Errorf("seed = %v, want 42", s.Seed)
	}
	if last.MaxTokens != 3 {
		t.Errorf("MaxTokens = %d, want 3 from max_completion_tokens", last.MaxTokens)
	}
}

func TestChatEchoesAcceptedModelNames(t *testing.T) {
	h := newHarness(t, scripted(countPieces...))

	t.Run("recorded default_model", func(t *testing.T) {
		rec := loadExchange(t, "chat_model_field")
		got := h.replay(rec)
		if got.Code != 200 {
			t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
		}
		if g, w := shapeOf(t, got.Body.String()), shapeOf(t, rec.Response.Body); !reflect.DeepEqual(g, w) {
			t.Fatalf("shape =\n%v, want\n%v", g, w)
		}
		if model := decodeCompletion(t, got.Body.String()).Model; model != "default_model" {
			t.Fatalf("model = %q, want default_model", model)
		}
	})

	t.Run("served model", func(t *testing.T) {
		got := h.do("POST", "/v1/chat/completions", `{"model": "test/model", "messages": []}`, nil)
		if got.Code != 200 {
			t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
		}
		if model := decodeCompletion(t, got.Body.String()).Model; model != "test/model" {
			t.Fatalf("model = %q, want test/model", model)
		}
	})
}

func TestChatRejectsOtherModels(t *testing.T) {
	h := newHarness(t, scripted(countPieces...))
	got := h.do("POST", "/v1/chat/completions", `{"model": "gpt-4o", "messages": []}`, nil)
	msg := errorMessage(t, got, 400)
	if !strings.Contains(msg, "test/model") || !strings.Contains(msg, "gpt-4o") {
		t.Errorf("error = %q, want it to name the requested and the served model", msg)
	}
	if n := h.engine.Calls.Load(); n != 0 {
		t.Errorf("engine was called %d times", n)
	}
}

func TestChatTemplateErrorIsABadRequest(t *testing.T) {
	h := newHarness(t, scripted(countPieces...), func(h *harness) {
		h.tmpl.err = errors.New("unknown role 'robot'")
	})
	got := h.replay(loadExchange(t, "chat"))
	if msg := errorMessage(t, got, 400); msg != "unknown role 'robot'" {
		t.Errorf("error = %q, want the template error", msg)
	}
	if n := h.engine.Calls.Load(); n != 0 {
		t.Errorf("engine was called %d times", n)
	}
}

func TestChatUsageIsCopiedFromTheEngineResult(t *testing.T) {
	// A real engine also counts the end-of-sequence token it does not emit and
	// knows how much of the prompt came from its cache.
	countsEOS := func(inner engine.Engine) engine.Engine {
		return engineFunc(func(ctx context.Context, req engine.Request, emit func(int32)) (engine.Result, error) {
			res, err := inner.Generate(ctx, req, emit)
			res.PromptTokens = 37
			res.CompletionTokens++
			res.CachedTokens = 36
			return res, err
		})
	}
	h := newHarness(t, scripted(countPieces...), func(h *harness) { h.wrap = countsEOS })
	got := h.replay(loadExchange(t, "chat"))
	if got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}
	u := decodeCompletion(t, got.Body.String()).Usage
	emitted := len(countPieces)
	if u.PromptTokens != 37 || u.CompletionTokens != emitted+1 || u.TotalTokens != 37+emitted+1 || u.PromptTokensDetails.CachedTokens != 36 {
		t.Fatalf("usage = %+v, want the engine's counts: 37 prompt, %d completion, 36 cached", *u, emitted+1)
	}
}

func TestChatIncludesTextTheDetokenizerHeldBack(t *testing.T) {
	heldBack := func(h *harness) { h.tok.tail = " 6" }

	t.Run("not streamed", func(t *testing.T) {
		h := newHarness(t, scripted(countPieces...), heldBack)
		got := h.replay(loadExchange(t, "chat"))
		if content := decodeCompletion(t, got.Body.String()).Choices[0].Message.Content; content != "1, 2, 3, 4, 5 6" {
			t.Fatalf("content = %q, want %q", content, "1, 2, 3, 4, 5 6")
		}
	})

	t.Run("streamed", func(t *testing.T) {
		h := newHarness(t, scripted("a", "b"), heldBack)
		got := h.replay(loadExchange(t, "chat_stream"))
		var chunks []completionBody
		for _, data := range sseData(t, got.Body.String()) {
			if data != "[DONE]" {
				chunks = append(chunks, decodeCompletion(t, data))
			}
		}
		if g, w := deltas(chunks), []string{"a", "b", " 6", ""}; !slices.Equal(g, w) {
			t.Fatalf("delta contents = %q, want %q", g, w)
		}
	})
}
