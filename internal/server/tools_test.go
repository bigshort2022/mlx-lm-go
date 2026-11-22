package server

import (
	"encoding/json"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/chattemplate"
)

type qwenTestTemplate struct{ *chattemplate.Template }

func (t qwenTestTemplate) Render(messages, tools, kwargs json.RawMessage, generate bool) (string, error) {
	return t.Template.Render(chattemplate.Input{Messages: messages, Tools: tools, Kwargs: kwargs, AddGenerationPrompt: generate})
}

func toolHarness(t *testing.T, pieces ...string) *harness {
	t.Helper()
	tmpl, err := chattemplate.Load(filepath.Join("..", "chattemplate", "testdata", "qwen38"))
	if err != nil {
		t.Fatal(err)
	}
	return newHarness(t, scripted(pieces...), func(h *harness) {
		h.cfg.ChatTemplateArgs = json.RawMessage(`{"enable_thinking":false}`)
		h.tune = func(s *server) { s.deps.Template = qwenTestTemplate{tmpl} }
	})
}

const toolDefinitions = `[{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"},"options":{"type":"object"}},"required":["path"]}}}]`
const toolRequest = `{"messages":[{"role":"user","content":"Read it."}],"tools":` + toolDefinitions

func TestQwenToolRoundTrip(t *testing.T) {
	const call = "<tool_call>\n<function=read>\n<parameter=path>\nnull\n</parameter>\n<parameter=limit>\n2\n</parameter>\n<parameter=options>\n{\"all\":true}\n</parameter>\n</function>\n</tool_call>"
	for _, stream := range []bool{false, true} {
		name, flags := "buffered", "}"
		if stream {
			name, flags = "streamed", `,"stream":true,"stream_options":{"include_usage":true}}`
		}
		t.Run(name, func(t *testing.T) {
			h := toolHarness(t, "Checking.\n", call[:5], call[5:len(call)-4], call[len(call)-4:], "\n", call)
			got := h.do("POST", "/v1/chat/completions", toolRequest+flags, nil)
			calls := assertToolReply(t, got, stream)
			if len(calls) != 2 || calls[0].ID == calls[1].ID {
				t.Fatalf("expected two distinct tool calls, got %+v", calls)
			}
			for i, c := range calls {
				if c.ID == "" || c.Type != "function" || c.Function.Name != "read" || !jsonEqual(t, c.Function.Arguments, `{"path":"null","limit":2,"options":{"all":true}}`) {
					t.Fatalf("unexpected tool call %+v", c)
				}
				if stream && (c.Index == nil || *c.Index != i) || !stream && c.Index != nil {
					t.Fatalf("unexpected index for %s call %d: %v", name, i, c.Index)
				}
			}

			replay, err := json.Marshal(calls[:1])
			if err != nil {
				t.Fatal(err)
			}
			followup := `{"messages":[{"role":"user","content":"Read it."},` +
				`{"role":"assistant","content":null,"tool_calls":` + string(replay) + `},` +
				`{"role":"tool","tool_call_id":"` + calls[0].ID + `","content":"file from OMP"}],"tools":` + toolDefinitions + `}`
			h = toolHarness(t, "I read it.")
			got = h.do("POST", "/v1/chat/completions", followup, nil)
			if got.Code != 200 || !strings.Contains(got.Body.String(), "I read it.") {
				t.Fatalf("tool result follow-up: %d %s", got.Code, got.Body)
			}
			var prompt strings.Builder
			for _, token := range h.engine.Last.Load().Prompt {
				prompt.WriteByte(byte(token))
			}
			// Parameter ordering in JSON objects is immaterial. Check tags individually.
			for _, part := range []string{"<function=read>", "<parameter=path>\nnull\n</parameter>", "<parameter=limit>\n2\n</parameter>", "<parameter=options>\n{\"all\": true}\n</parameter>", "<tool_response>\nfile from OMP\n</tool_response>"} {
				if !strings.Contains(prompt.String(), part) {
					t.Errorf("follow-up prompt missing %q: %s", part, prompt.String())
				}
			}
		})
	}
}

type wireTestCall struct {
	ID       string `json:"id"`
	Index    *int   `json:"index,omitempty"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func assertToolReply(t *testing.T, got *httptest.ResponseRecorder, stream bool) []wireTestCall {
	t.Helper()
	if got.Code != 200 {
		t.Fatalf("status = %d: %s", got.Code, got.Body)
	}
	events := []string{got.Body.String()}
	if stream {
		events = sseData(t, got.Body.String())
		if len(events) < 2 || events[len(events)-1] != "[DONE]" {
			t.Fatalf("missing stream terminator: %q", events)
		}
		events = events[:len(events)-1]
	}
	var calls []wireTestCall
	var text strings.Builder
	var finish string
	var usage bool
	for _, event := range events {
		if strings.Contains(event, `<tool_call>`) {
			t.Fatalf("raw tool markup leaked: %s", event)
		}
		type message struct {
			Content string         `json:"content"`
			Calls   []wireTestCall `json:"tool_calls"`
		}
		var body struct {
			Usage *struct {
				Total int `json:"total_tokens"`
			} `json:"usage"`
			Choices []struct {
				Finish string  `json:"finish_reason"`
				Delta  message `json:"delta"`
				Msg    message `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(event), &body); err != nil {
			t.Fatal(err)
		}
		if body.Usage != nil && body.Usage.Total > 0 {
			usage = true
		}
		for _, c := range body.Choices {
			if c.Finish != "" {
				finish = c.Finish
			}
			calls = append(calls, c.Delta.Calls...)
			calls = append(calls, c.Msg.Calls...)
			text.WriteString(c.Delta.Content)
			text.WriteString(c.Msg.Content)
		}
	}
	if strings.TrimSpace(text.String()) != "Checking." || finish != "tool_calls" || !usage {
		t.Fatalf("text=%q finish=%q usage=%v: %s", text.String(), finish, usage, got.Body)
	}
	return calls
}

func TestQwenToolNeverEmitsIncompleteOrUnknownCall(t *testing.T) {
	for _, test := range []struct{ name, output, end string }{
		{"length", "<tool_call>\n<function=read>", `,"max_tokens":1}`},
		{"unknown", "<tool_call>\n<function=unknown>\n</function>\n</tool_call>", "}"},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := toolHarness(t, test.output, "end")
			got := h.do("POST", "/v1/chat/completions", toolRequest+test.end, nil)
			if test.name == "unknown" {
				if got.Code != 500 || !strings.Contains(got.Body.String(), "unknown") {
					t.Fatalf("unknown function: %d %s", got.Code, got.Body)
				}
			} else if got.Code != 200 || decodeCompletion(t, got.Body.String()).finish(t) != "length" || strings.Contains(got.Body.String(), "tool_call") {
				t.Fatalf("truncated generation: %d %s", got.Code, got.Body)
			}
		})
	}
}

func TestQwenToolNullableParameter(t *testing.T) {
	const request = `{"messages":[{"role":"user","content":"Search"}],"tools":[{"type":"function","function":{"name":"grep","parameters":{"type":"object","properties":{"skip":{"type":["number","null"]}}}}}]}`
	for _, value := range []string{"12", "null"} {
		t.Run(value, func(t *testing.T) {
			h := toolHarness(t, "Checking.", "<tool_call>\n<function=grep>\n<parameter=skip>\n"+value+"\n</parameter>\n</function>\n</tool_call>")
			got := h.do("POST", "/v1/chat/completions", request, nil)
			calls := assertToolReply(t, got, false)
			if len(calls) != 1 || !jsonEqual(t, calls[0].Function.Arguments, `{"skip":`+value+`}`) {
				t.Fatalf("expected skip=%s, got %+v", value, calls)
			}
		})
	}
}
