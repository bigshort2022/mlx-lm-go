package server

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
)

// Use synthetic IDs outside the byte vocabulary to distinguish encoded markers
// from literal marker text in a prompt.
const (
	testThinkStart int32 = 1000
	testThinkEnd   int32 = 1001
)

type thinkingTokenizer struct {
	*fakeTokenizer
	prompt []int32
}

func (tok thinkingTokenizer) Encode(text string) []int32 {
	if text == "PROMPT" {
		return tok.prompt
	}
	return tok.fakeTokenizer.Encode(text)
}

func thinkingModel(prompt ...int32) func(*harness) {
	return func(h *harness) {
		h.tune = func(s *server) {
			s.deps.Tokenizer = thinkingTokenizer{h.tok, prompt}
			s.deps.ThinkingTokens = &ThinkingTokens{Start: testThinkStart, End: testThinkEnd}
		}
	}
}

func assertThinkingMessage(t *testing.T, doc, want string) {
	t.Helper()
	var res struct {
		Choices []struct {
			Message json.RawMessage `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(doc), &res); err != nil || len(res.Choices) != 1 {
		t.Fatalf("invalid response %s: %v", doc, err)
	}
	if got := string(res.Choices[0].Message); !jsonEqual(t, got, want) {
		t.Errorf("message = %s, want %s", got, want)
	}
}

func TestChatThinkingBuffered(t *testing.T) {
	cases := []struct {
		name, request, finish, want string
		prompt                      []int32
		pieces                      []string
	}{
		{"last start wins", `{"messages":[]}`, "stop", `{"role":"assistant","content":"answer","reasoning":"reason"}`,
			[]int32{testThinkStart, testThinkEnd, testThinkStart, ' '}, []string{"reason</thi", "nk>answer"}},
		{"disabled template", `{"messages":[],"chat_template_kwargs":{"enable_thinking":false}}`, "stop", `{"role":"assistant","content":"answer"}`,
			[]int32{testThinkStart, testThinkEnd, ' '}, []string{"answer"}},
		{"reasoning only at length", `{"messages":[],"max_tokens":1}`, "length", `{"role":"assistant","content":null,"reasoning":"reason</thi"}`,
			[]int32{testThinkStart}, []string{"reason</thi", "unreached"}},
		{"unclosed delimiter on EOS", `{"messages":[]}`, "stop", `{"role":"assistant","content":null,"reasoning":"reason"}`,
			[]int32{testThinkStart}, []string{"reason</thi"}},
		{"state-specific markers and near misses", `{"messages":[]}`, "stop", `{"role":"assistant","content":"a</think>bcd","reasoning":"x<think>y</thiz<<"}`,
			nil, []string{"a</think>b<th", "ink>x<think>y</thi", "z<<<", "/think>c<think></think>d"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, scripted(tc.pieces...), thinkingModel(tc.prompt...))
			got := h.do("POST", "/v1/chat/completions", tc.request, nil)
			if got.Code != 200 {
				t.Fatalf("status = %d: %s", got.Code, got.Body)
			}
			assertThinkingMessage(t, got.Body.String(), tc.want)
			if finish := decodeCompletion(t, got.Body.String()).finish(t); finish != tc.finish {
				t.Errorf("finish = %q, want %q", finish, tc.finish)
			}
			if tc.name == "disabled template" && h.tmpl.kwargs != `{"enable_thinking":false}` {
				t.Errorf("template kwargs = %s", h.tmpl.kwargs)
			}
		})
	}
}

func assertThinkingStream(t *testing.T, stream, finish string, want ...string) {
	t.Helper()
	events := sseData(t, stream)
	if len(events) != len(want)+2 || events[len(events)-1] != "[DONE]" {
		t.Fatalf("unexpected events: %v", events)
	}
	for i, event := range events[:len(events)-1] {
		var res struct {
			Choices []struct {
				Delta json.RawMessage `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(event), &res); err != nil || len(res.Choices) != 1 {
			t.Fatalf("invalid event %s: %v", event, err)
		}
		wantDelta, wantFinish := `{"role":"assistant"}`, finish
		if i < len(want) {
			wantDelta, wantFinish = want[i], ""
		}
		if got := string(res.Choices[0].Delta); !jsonEqual(t, got, wantDelta) {
			t.Errorf("delta %d = %s, want %s", i, got, wantDelta)
		}
		if got := decodeCompletion(t, event).finish(t); got != wantFinish {
			t.Errorf("finish %d = %q, want %q", i, got, wantFinish)
		}
	}
}

func TestChatThinkingStreamOrder(t *testing.T) {
	h := newHarness(t, scripted("pre<th", "ink>reason</th", "ink>answer<think>more</think>end"), thinkingModel())
	got := h.do("POST", "/v1/chat/completions", `{"messages":[],"stream":true}`, nil)
	assertThinkingStream(t, got.Body.String(), "stop",
		`{"role":"assistant","content":"pre"}`, `{"role":"assistant","reasoning":"reason"}`,
		`{"role":"assistant","content":"answer"}`, `{"role":"assistant","reasoning":"more"}`, `{"role":"assistant","content":"end"}`)
}

func TestChatThinkingUserStop(t *testing.T) {
	errs := make(chan error, 1)
	h := newHarness(t, scripted("reason", "</thi", "ST", "OP", "nk>unreached"), thinkingModel(testThinkStart), func(h *harness) {
		h.wrap = reportErrors(errs)
	})
	got := h.do("POST", "/v1/chat/completions", `{"messages":[],"stream":true,"stop":"STOP"}`, nil)
	assertThinkingStream(t, got.Body.String(), "stop", `{"role":"assistant","reasoning":"reason"}`)
	awaitCanceled(t, errs)
}

func TestChatThinkingClientCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h := newHarness(t, scripted("reason</thi"), thinkingModel(testThinkStart), func(h *harness) {
		h.wrap = func(engine.Engine) engine.Engine {
			return engineFunc(func(ctx context.Context, _ engine.Request, emit func(int32)) (engine.Result, error) {
				emit(0)
				cancel()
				return engine.Result{}, ctx.Err()
			})
		}
	})
	r := httptest.NewRequest("POST", "http://127.0.0.1:8080/v1/chat/completions", strings.NewReader(`{"messages":[],"stream":true}`)).WithContext(ctx)
	r.Header.Set("Content-Type", "application/json")
	got := httptest.NewRecorder()
	h.handler.ServeHTTP(got, r)
	events := sseData(t, got.Body.String())
	if len(events) != 2 || !strings.Contains(events[0], `"reasoning":"reason"`) || strings.Contains(events[0], "</thi") ||
		!jsonEqual(t, events[1], `{"error":"context canceled"}`) {
		t.Errorf("events = %v, want reasoning then cancellation without [DONE]", events)
	}
}

func TestChatThinkingLegacyAndTextCompletions(t *testing.T) {
	const literal = "<think>reason</think>answer"
	h := newHarness(t, scripted(literal))
	got := h.do("POST", "/v1/chat/completions", `{"messages":[]}`, nil)
	assertThinkingMessage(t, got.Body.String(), `{"role":"assistant","content":"`+literal+`"}`)

	h = newHarness(t, scripted(literal), thinkingModel(testThinkStart))
	got = h.do("POST", "/v1/completions", `{"prompt":"PROMPT"}`, nil)
	c := decodeCompletion(t, got.Body.String()).Choices[0]
	if c.Text == nil || *c.Text != literal || c.Message != nil {
		t.Errorf("text completion changed: %s", got.Body)
	}
}
