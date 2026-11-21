package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
)

// exchange is one request/response pair recorded from Python mlx_lm.server,
// read from testdata/server at the repository root.
type exchange struct {
	Request struct {
		Method  string            `json:"method"`
		Path    string            `json:"path"`
		Headers map[string]string `json:"headers"`
		Body    json.RawMessage   `json:"body"`
	} `json:"request"`
	Response struct {
		Status int    `json:"status"`
		Body   string `json:"body"`
	} `json:"response"`
}

func loadExchange(t *testing.T, name string) exchange {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "server", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var e exchange
	if err := json.Unmarshal(raw, &e); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return e
}

// requestBody returns the bytes the recorder sent: a JSON string in the
// fixture is a raw body, anything else was sent as JSON, and null is no body.
func (e exchange) requestBody() string {
	var raw string
	if json.Unmarshal(e.Request.Body, &raw) == nil {
		return raw
	}
	if string(e.Request.Body) == "null" {
		return ""
	}
	return string(e.Request.Body)
}

// harness is a server wired to fakes.
type harness struct {
	cfg     Config
	engine  *engine.Fake
	tok     *fakeTokenizer
	tmpl    *fakeTemplater
	wrap    func(engine.Engine) engine.Engine // if set, the server gets wrap(engine)
	ui      fs.FS                             // if set, the server serves it as the web UI
	tune    func(*server)                     // if set, adjusts the server after New
	handler http.Handler
}

// newHarness builds a server on loopback port 8080 serving "test/model".
// Each setup function may adjust the config and fakes before the handler is
// created.
func newHarness(t *testing.T, setup ...func(*harness)) *harness {
	t.Helper()
	h := &harness{
		cfg:    Config{Host: "127.0.0.1", Port: 8080, Model: "test/model", TopP: 1.0, MaxTokens: 512},
		engine: &engine.Fake{},
		tok:    &fakeTokenizer{},
		tmpl:   &fakeTemplater{out: "PROMPT"},
	}
	for _, fn := range setup {
		fn(h)
	}
	deps := Deps{Engine: h.engine, Tokenizer: h.tok, Template: h.tmpl, WebUI: h.ui}
	if h.wrap != nil {
		deps.Engine = h.wrap(h.engine)
	}
	h.handler = New(h.cfg, deps)
	if h.tune != nil {
		h.tune(h.handler.(*server))
	}
	return h
}

// do sends one request to the handler, addressed to the configured host and
// port unless headers name another Host. A non-empty body is sent as
// application/json unless headers say otherwise.
func (h *harness) do(method, path, body string, headers map[string]string) *httptest.ResponseRecorder {
	url := fmt.Sprintf("http://%s%s", net.JoinHostPort(h.cfg.Host, strconv.Itoa(h.cfg.Port)), path)
	req := httptest.NewRequest(method, url, strings.NewReader(body))
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	for k, v := range headers {
		if k == "Host" {
			req.Host = v
			continue
		}
		req.Header.Set(k, v)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	h.handler.ServeHTTP(rec, req)
	return rec
}

// replay sends the request of a recorded exchange.
func (h *harness) replay(e exchange) *httptest.ResponseRecorder {
	return h.do(e.Request.Method, e.Request.Path, e.requestBody(), e.Request.Headers)
}

// jsonEqual reports whether two JSON documents have the same value.
func jsonEqual(t *testing.T, a, b string) bool {
	t.Helper()
	var x, y any
	if err := json.Unmarshal([]byte(a), &x); err != nil {
		t.Fatalf("invalid JSON %q: %v", a, err)
	}
	if err := json.Unmarshal([]byte(b), &y); err != nil {
		t.Fatalf("invalid JSON %q: %v", b, err)
	}
	return reflect.DeepEqual(x, y)
}

// countPieces is the piece sequence of the recorded "Count from 1 to 5." chats.
var countPieces = []string{"1", ",", " 2", ",", " 3", ",", " 4", ",", " 5"}

// scripted makes the engine emit one token per piece and the detokenizer turn
// each token back into its piece.
func scripted(pieces ...string) func(*harness) {
	return func(h *harness) {
		h.tok.pieces = pieces
		h.engine.Tokens = make([]int32, len(pieces))
		for i := range pieces {
			h.engine.Tokens[i] = int32(i)
		}
	}
}

// completionBody decodes the fields tests assert on in a completion response
// or stream chunk, for both chat and text completions.
type completionBody struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Model   string `json:"model"`
	Choices []struct {
		FinishReason *string      `json:"finish_reason"`
		Message      *chatMessage `json:"message"`
		Delta        *chatMessage `json:"delta"`
		Text         *string      `json:"text"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		TotalTokens         int `json:"total_tokens"`
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
}

func decodeCompletion(t *testing.T, doc string) completionBody {
	t.Helper()
	var body completionBody
	if err := json.Unmarshal([]byte(doc), &body); err != nil {
		t.Fatalf("invalid completion body %q: %v", doc, err)
	}
	return body
}

// finish returns the finish_reason of the only choice, or "" when it is null.
func (b completionBody) finish(t *testing.T) string {
	t.Helper()
	if len(b.Choices) != 1 {
		t.Fatalf("got %d choices, want 1", len(b.Choices))
	}
	if b.Choices[0].FinishReason == nil {
		return ""
	}
	return *b.Choices[0].FinishReason
}

// errorMessage asserts that a response is a JSON error with the given status
// and returns its message.
func errorMessage(t *testing.T, got *httptest.ResponseRecorder, status int) string {
	t.Helper()
	if got.Code != status {
		t.Fatalf("status = %d, want %d: %s", got.Code, status, got.Body)
	}
	if ct := got.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", ct)
	}
	var body map[string]string
	if err := json.Unmarshal(got.Body.Bytes(), &body); err != nil || len(body) != 1 || body["error"] == "" {
		t.Fatalf("body = %s, want {\"error\": \"<message>\"}", got.Body)
	}
	return body["error"]
}

// engineFunc adapts a function to engine.Engine.
type engineFunc func(ctx context.Context, req engine.Request, emit func(token int32)) (engine.Result, error)

func (f engineFunc) Generate(ctx context.Context, req engine.Request, emit func(token int32)) (engine.Result, error) {
	return f(ctx, req, emit)
}

// sseData returns the payload of every data event in a server-sent event
// stream, in order. Comment lines such as the reference's keepalives are
// skipped.
func sseData(t *testing.T, stream string) []string {
	t.Helper()
	var events []string
	for block := range strings.SplitSeq(stream, "\n\n") {
		if block == "" || strings.HasPrefix(block, ":") {
			continue
		}
		data, ok := strings.CutPrefix(block, "data: ")
		if !ok || strings.Contains(data, "\n") {
			t.Fatalf("malformed event %q in stream %q", block, stream)
		}
		events = append(events, data)
	}
	return events
}

// assertStreamShape checks that a stream has the same events as a recorded
// one: the same number, each with the recorded shape, ending in [DONE]. It
// returns the decoded chunks without the [DONE] marker.
func assertStreamShape(t *testing.T, got *httptest.ResponseRecorder, rec exchange) []completionBody {
	t.Helper()
	if got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}
	if ct := got.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
	if cc := got.Header().Get("Cache-Control"); cc != "no-cache" {
		t.Errorf("Cache-Control = %q, want no-cache", cc)
	}
	if !strings.HasSuffix(got.Body.String(), "\n\n") {
		t.Errorf("stream does not end with a blank line: %q", got.Body)
	}
	events, want := sseData(t, got.Body.String()), sseData(t, rec.Response.Body)
	if len(events) != len(want) {
		t.Fatalf("got %d events, want %d:\n%s", len(events), len(want), got.Body)
	}
	last := len(events) - 1
	if events[last] != "[DONE]" {
		t.Fatalf("last event = %q, want [DONE]", events[last])
	}
	chunks := make([]completionBody, last)
	for i := range last {
		if g, w := shapeOf(t, events[i]), shapeOf(t, want[i]); !reflect.DeepEqual(g, w) {
			t.Fatalf("event %d shape =\n%v, want\n%v", i, g, w)
		}
		chunks[i] = decodeCompletion(t, events[i])
		if chunks[i].ID != chunks[0].ID {
			t.Errorf("event %d id = %q, want %q like the first", i, chunks[i].ID, chunks[0].ID)
		}
	}
	return chunks
}

// deltas returns the delta.content of every chat chunk that has a choice.
func deltas(chunks []completionBody) []string {
	var out []string
	for _, c := range chunks {
		if len(c.Choices) == 1 && c.Choices[0].Delta != nil {
			out = append(out, c.Choices[0].Delta.Content)
		}
	}
	return out
}
