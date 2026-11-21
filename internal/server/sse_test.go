package server

import (
	"bufio"
	"context"
	"net/http"
	"slices"
	"strings"
	"testing"
)

func TestChatStreamMatchesRecording(t *testing.T) {
	rec := loadExchange(t, "chat_stream")
	h := newHarness(t, scripted(countPieces...))
	got := h.replay(rec)
	chunks := assertStreamShape(t, got, rec)

	// One chunk per piece, then a final chunk with no content.
	if g, w := deltas(chunks), append(slices.Clone(countPieces), ""); !slices.Equal(g, w) {
		t.Errorf("delta contents = %q, want %q", g, w)
	}
	if !strings.HasPrefix(chunks[0].ID, "chatcmpl-") {
		t.Errorf("id = %q, want a chatcmpl- id", chunks[0].ID)
	}
	final := len(chunks) - 1
	for i, c := range chunks {
		if c.Object != "chat.completion.chunk" {
			t.Errorf("chunk %d object = %q, want chat.completion.chunk", i, c.Object)
		}
		if fr := c.finish(t); i < final && fr != "" {
			t.Errorf("chunk %d finish_reason = %q, want null", i, fr)
		}
	}
	if fr := chunks[final].finish(t); fr != "stop" {
		t.Errorf("final finish_reason = %q, want stop", fr)
	}
	if d := *chunks[final].Choices[0].Delta; d != (chatMessage{Role: "assistant"}) {
		t.Errorf("final delta = %+v, want only the assistant role", d)
	}
}

func TestChatStreamFlushesEachChunk(t *testing.T) {
	g := newGate(t, 1) // blocks before the second token
	_, ts := newLiveHarness(t, scripted(countPieces...), func(h *harness) { h.engine.Gate = g.wait })
	resp := postJSON(t, ts.URL+"/v1/chat/completions", loadExchange(t, "chat_stream").requestBody())
	events := bufio.NewReader(resp.Body)

	// The second token is still held back, yet the first chunk has arrived.
	first := decodeCompletion(t, nextEvent(t, events))
	if got := first.Choices[0].Delta.Content; got != "1" {
		t.Fatalf("first delta = %q, want %q", got, "1")
	}

	g.open()
	var rest []string
	for {
		data := nextEvent(t, events)
		if data == "[DONE]" {
			break
		}
		rest = append(rest, decodeCompletion(t, data).Choices[0].Delta.Content)
	}
	if want := append(slices.Clone(countPieces[1:]), ""); !slices.Equal(rest, want) {
		t.Fatalf("remaining deltas = %q, want %q", rest, want)
	}
}

func TestChatStreamUsageChunk(t *testing.T) {
	rec := loadExchange(t, "chat_stream_usage")
	// The eight tokens of the recording: the detokenizer holds each space
	// back until the digit after it, so two of them produce no text.
	h := newHarness(t, scripted("1", ",", "", " 2", ",", "", " 3", ",", "", " 4"))
	got := h.replay(rec)
	chunks := assertStreamShape(t, got, rec)

	if g, w := deltas(chunks), []string{"1", ",", " 2", ",", " 3", ",", ""}; !slices.Equal(g, w) {
		t.Errorf("delta contents = %q, want %q", g, w)
	}
	if fr := chunks[len(chunks)-2].finish(t); fr != "length" {
		t.Errorf("final finish_reason = %q, want length", fr)
	}
	last := chunks[len(chunks)-1]
	if last.Object != "chat.completion" || len(last.Choices) != 0 {
		t.Errorf("usage chunk = %+v, want object chat.completion without choices", last)
	}
	prompt := len(h.tok.Encode(h.tmpl.out))
	if u := last.Usage; u == nil || u.PromptTokens != prompt || u.CompletionTokens != 8 || u.TotalTokens != prompt+8 {
		t.Errorf("usage = %+v, want %d prompt and 8 completion tokens", u, prompt)
	}
}

func TestChatStreamUsageCanBeDeclined(t *testing.T) {
	const request = `{"messages": [], "stream": true, "stream_options": {"include_usage": false}}`
	h := newHarness(t, scripted("hi"))
	got := h.do("POST", "/v1/chat/completions", request, nil)
	if events := sseData(t, got.Body.String()); len(events) != 3 {
		t.Fatalf("got %d events, want a text chunk, a final chunk and [DONE]:\n%s", len(events), got.Body)
	}
}

func TestClientDisconnectCancelsGeneration(t *testing.T) {
	for _, mode := range []string{"streamed", "not streamed"} {
		t.Run(mode, func(t *testing.T) {
			g := newGate(t, 1)
			errs := make(chan error, 1)
			_, ts := newLiveHarness(t, scripted(countPieces...), func(h *harness) {
				h.engine.Gate = g.wait
				h.wrap = reportErrors(errs)
			})

			name := "chat"
			if mode == "streamed" {
				name = "chat_stream"
			}
			ctx, hangUp := context.WithCancel(context.Background())
			defer hangUp()
			req, err := http.NewRequestWithContext(ctx, "POST", ts.URL+"/v1/chat/completions", strings.NewReader(loadExchange(t, name).requestBody()))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")
			go func() {
				if resp, err := http.DefaultClient.Do(req); err == nil {
					resp.Body.Close()
				}
			}()

			g.awaitEntered(t)
			hangUp() // closes the connection while the engine is still blocked
			awaitCanceled(t, errs)
		})
	}
}
