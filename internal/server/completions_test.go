package server

import (
	"reflect"
	"slices"
	"strings"
	"testing"
)

// The eight tokens of the recorded text completions, then one the budget cuts
// off. The detokenizer holds each space back until the digit after it.
var completionPieces = []string{"", "4", ",", "", " 5", ",", "", " 6", ","}

func TestCompletionsMatchesRecording(t *testing.T) {
	rec := loadExchange(t, "completions")
	h := newHarness(t, scripted(completionPieces...))
	got := h.replay(rec)

	if got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}
	if g, w := shapeOf(t, got.Body.String()), shapeOf(t, rec.Response.Body); !reflect.DeepEqual(g, w) {
		t.Fatalf("shape =\n%v, want\n%v", g, w)
	}
	body := decodeCompletion(t, got.Body.String())
	if body.Object != "text_completion" {
		t.Errorf("object = %q, want text_completion", body.Object)
	}
	if id, ok := strings.CutPrefix(body.ID, "cmpl-"); !ok || !uuidV4.MatchString(id) {
		t.Errorf("id = %q, want cmpl-<uuid4>", body.ID)
	}
	if text := body.Choices[0].Text; text == nil || *text != "4, 5, 6" {
		t.Errorf("text = %v, want %q", text, "4, 5, 6")
	}
	if fr := body.finish(t); fr != "length" {
		t.Errorf("finish_reason = %q, want length", fr)
	}
	if u := body.Usage; u.PromptTokens != len("1, 2, 3,") || u.CompletionTokens != 8 || u.TotalTokens != u.PromptTokens+8 {
		t.Errorf("usage = %+v, want %d prompt and 8 completion tokens", *u, len("1, 2, 3,"))
	}

	if want := h.tok.Encode("1, 2, 3,"); !slices.Equal(h.engine.Last.Load().Prompt, want) {
		t.Errorf("engine prompt = %v, want the encoded prompt %v", h.engine.Last.Load().Prompt, want)
	}
	if h.tmpl.messages != "" {
		t.Errorf("the chat template was rendered with %s", h.tmpl.messages)
	}
}

func TestCompletionsStreamMatchesRecording(t *testing.T) {
	rec := loadExchange(t, "completions_stream")
	h := newHarness(t, scripted(completionPieces...))
	got := h.replay(rec)
	chunks := assertStreamShape(t, got, rec)

	var texts []string
	for i, c := range chunks {
		if c.Object != "text_completion" {
			t.Errorf("chunk %d object = %q, want text_completion", i, c.Object)
		}
		if !strings.HasPrefix(c.ID, "cmpl-") {
			t.Errorf("chunk %d id = %q, want a cmpl- id", i, c.ID)
		}
		texts = append(texts, *c.Choices[0].Text)
	}
	if want := []string{"4", ",", " 5", ",", " 6", ""}; !slices.Equal(texts, want) {
		t.Errorf("texts = %q, want %q", texts, want)
	}
	if fr := chunks[len(chunks)-1].finish(t); fr != "length" {
		t.Errorf("final finish_reason = %q, want length", fr)
	}
}

func TestCompletionsStreamUsageChunk(t *testing.T) {
	const request = `{"prompt": "1, 2, 3,", "max_tokens": 8, "stream": true, "stream_options": {"include_usage": true}}`
	h := newHarness(t, scripted(completionPieces...))
	got := h.do("POST", "/v1/completions", request, nil)
	events := sseData(t, got.Body.String())
	if len(events) < 2 || events[len(events)-1] != "[DONE]" {
		t.Fatalf("stream does not end with a usage event and [DONE]:\n%s", got.Body)
	}
	last := decodeCompletion(t, events[len(events)-2])
	if last.Object != "text_completion" || len(last.Choices) != 0 || last.Usage == nil || last.Usage.CompletionTokens != 8 {
		t.Fatalf("usage event = %s, want a text_completion without choices reporting 8 completion tokens", events[len(events)-2])
	}
}
