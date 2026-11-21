package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// replayError replays a recorded bad request and returns the error message of
// the 400 response, after checking that the engine was left alone.
func replayError(t *testing.T, name string) (message string, rec exchange) {
	t.Helper()
	rec = loadExchange(t, name)
	h := newHarness(t, scripted(countPieces...))
	got := h.replay(rec)
	message = errorMessage(t, got, 400)
	if n := h.engine.Calls.Load(); n != 0 {
		t.Errorf("engine was called %d times", n)
	}
	return message, rec
}

func TestInvalidJSONBody(t *testing.T) {
	msg, rec := replayError(t, "err_invalid_json")
	if !strings.HasPrefix(msg, "Invalid JSON in request body: ") {
		t.Errorf("error = %q, want it to start like the reference", msg)
	}
	if g, w := shapeOf(t, `{"error": "`+msg+`"}`), shapeOf(t, rec.Response.Body); !reflect.DeepEqual(g, w) {
		t.Errorf("shape = %v, want %v", g, w)
	}
}

func TestBodyThatIsNotAnObject(t *testing.T) {
	msg, rec := replayError(t, "err_not_object")
	if want := `{"error": "` + msg + `"}`; !jsonEqual(t, want, rec.Response.Body) {
		t.Errorf("error = %q, want the recorded %s", msg, rec.Response.Body)
	}
}

// Every bad request recorded from the reference gets the reference's status
// and message, except where the message is not the reference's own wording
// but Python's: its JSON parser's, and an attribute error for messages that
// are not a list.
func TestRecordedBadRequestsMatchTheReference(t *testing.T) {
	pythonWording := map[string]bool{"err_invalid_json": true, "err_messages_wrong_type": true}
	paths, err := filepath.Glob(filepath.Join("..", "..", "testdata", "server", "err_*.json"))
	if err != nil || len(paths) < 9 {
		t.Fatalf("found %d err_*.json recordings (%v), want at least 9", len(paths), err)
	}
	names := []string{"chat_logprobs"}
	for _, path := range paths {
		names = append(names, strings.TrimSuffix(filepath.Base(path), ".json"))
	}
	for _, name := range names {
		if pythonWording[name] {
			continue
		}
		t.Run(name, func(t *testing.T) {
			rec := loadExchange(t, name)
			var want struct {
				Error string `json:"error"`
			}
			if err := json.Unmarshal([]byte(rec.Response.Body), &want); err != nil || want.Error == "" {
				t.Fatalf("recording has no error message: %s", rec.Response.Body)
			}
			h := newHarness(t, scripted(countPieces...))
			got := h.replay(rec)
			if got.Code != rec.Response.Status {
				t.Errorf("status = %d, want the recorded %d", got.Code, rec.Response.Status)
			}
			if msg := assertRejected(t, h, got, got.Code); msg != want.Error {
				t.Errorf("error = %q, want the recorded %q", msg, want.Error)
			}
		})
	}
}

func TestFieldValidation(t *testing.T) {
	cases := []struct {
		field string // JSON members added to a valid chat request
		want  string
	}{
		{`"top_p": -0.1`, "top_p must be at least 0"},
		{`"max_tokens": 0`, "max_tokens must be at least 1"},
		{`"max_completion_tokens": 0`, "max_completion_tokens must be at least 1"},
		// Numbers the engine's 32-bit floats cannot hold.
		{`"temperature": 1e39`, "temperature must be at most 3.4028234663852886e+38"},
		{`"temperature": 1e999`, "temperature must be at most 3.4028234663852886e+38"},
		{`"presence_penalty": -1e39`, "presence_penalty must be at least -3.4028234663852886e+38"},
		{`"frequency_penalty": 4e38`, "frequency_penalty must be at most 3.4028234663852886e+38"},
		{`"repetition_penalty": 1e39`, "repetition_penalty must be at most 3.4028234663852886e+38"},
		{`"top_p": 1e39`, "top_p must be at most 1"},
		{`"logit_bias": {"5": 1e39}`, "logit_bias values must be at most 3.4028234663852886e+38"},
		{`"logit_bias": {"5": -1e39}`, "logit_bias values must be at least -3.4028234663852886e+38"},
		{`"logit_bias": {"-1": 1}`, "logit_bias token ids must be at least 0"},
		{`"logit_bias": {"5": null}`, "logit_bias must be a dict of int to float"},
		{`"logit_bias": {"99999999999": 1}`, "logit_bias must be a dict of int to float"},
		// Fields this server does not act on are still validated as in the
		// reference.
		{`"num_draft_tokens": -1`, "num_draft_tokens must be at least 0"},
		{`"num_draft_tokens": "3"`, "num_draft_tokens must be of type int"},
		{`"logprobs": 0`, "logprobs must be of type bool"},
		{`"logprobs": "yes"`, "logprobs must be of type bool"},
		{`"top_logprobs": 12`, "top_logprobs must be at most 11"},
		{`"top_logprobs": -2`, "top_logprobs must be at least 0"},
		{`"top_logprobs": 1.5`, "top_logprobs must be of type int"},
		{`"xtc_probability": 1.5`, "xtc_probability must be at most 1"},
		{`"xtc_probability": -0.5`, "xtc_probability must be at least 0"},
		{`"xtc_threshold": 2`, "xtc_threshold must be at most 1"},
		{`"xtc_threshold": "low"`, "xtc_threshold must be of type float"},
		{`"adapters": 5`, "adapters must be of type str"},
		// The reference's order decides which of several mistakes is named.
		{`"logit_bias": [], "top_p": 2, "stream": "yes"`, "stream must be of type bool"},
		{`"logit_bias": [], "top_p": 2, "max_tokens": 0`, "max_tokens must be at least 1"},
		{`"logit_bias": [], "seed": "x", "top_k": -1`, "top_k must be at least 0"},
		{`"tools": [{}], "seed": "x"`, "seed must be of type int"},
		{`"top_k": -1`, "top_k must be at least 0"},
		{`"top_k": 1.5`, "top_k must be of type int"},
		{`"min_p": 2`, "min_p must be at most 1"},
		{`"min_p": -0.1`, "min_p must be at least 0"},
		{`"repetition_penalty": -1`, "repetition_penalty must be at least 0"},
		{`"repetition_context_size": -1`, "repetition_context_size must be at least 0"},
		{`"presence_context_size": -1`, "presence_context_size must be at least 0"},
		{`"frequency_context_size": -1`, "frequency_context_size must be at least 0"},
		{`"temperature": "hot"`, "temperature must be of type ('float', 'int')"},
		{`"max_tokens": "many"`, "max_tokens must be of type int"},
		{`"max_tokens": true`, "max_tokens must be of type int"},
		{`"max_tokens": 1.0`, "max_tokens must be of type int"},
		{`"max_completion_tokens": -1`, "max_completion_tokens must be at least 1"},
		{`"seed": -1`, "seed must be at least 0"},
		{`"seed": "x"`, "seed must be of type int"},
		{`"model": 5`, "model must be of type str"},
		{`"logit_bias": {"a": 1}`, "logit_bias must be a dict of int to float"},
		{`"logit_bias": {"1": "x"}`, "logit_bias must be a dict of int to float"},
		{`"stop": 5`, "stop must be a string or a list of strings"},
		{`"stop": [1]`, "stop must be a string or a list of strings"},
		{`"stream_options": []`, "stream_options must be of type dict"},
		{`"stream_options": {"include_usage": "yes"}`, "include_usage must be of type bool"},
		{`"chat_template_kwargs": []`, "chat_template_kwargs must be of type dict"},
	}
	for _, tc := range cases {
		t.Run(tc.field, func(t *testing.T) {
			h := newHarness(t, scripted(countPieces...))
			got := h.do("POST", "/v1/chat/completions", `{"messages": [], `+tc.field+`}`, nil)
			if msg := errorMessage(t, got, 400); msg != tc.want {
				t.Errorf("error = %q, want %q", msg, tc.want)
			}
			if n := h.engine.Calls.Load(); n != 0 {
				t.Errorf("engine was called %d times", n)
			}
		})
	}
}

func TestMessageAndPromptValidation(t *testing.T) {
	cases := []struct {
		path, body, want string
	}{
		{"/v1/chat/completions", `{"messages": ["hi"]}`, "messages must be a list of dicts"},
		{"/v1/chat/completions", `{"messages": "hi"}`, "messages must be of type list"},
		{"/v1/chat/completions", `{"messages": null}`, "Request did not contain messages"},
		{"/v1/completions", `{"prompt": ["a", "b"]}`, "prompt must be of type str"},
	}
	for _, tc := range cases {
		t.Run(tc.body, func(t *testing.T) {
			h := newHarness(t, scripted(countPieces...))
			if msg := errorMessage(t, h.do("POST", tc.path, tc.body, nil), 400); msg != tc.want {
				t.Errorf("error = %q, want %q", msg, tc.want)
			}
			if n := h.engine.Calls.Load(); n != 0 {
				t.Errorf("engine was called %d times", n)
			}
		})
	}
}

func TestLenientFieldValues(t *testing.T) {
	cases := []string{
		// null means "not given", as OpenAI clients send it.
		`"temperature": null, "top_p": null, "max_tokens": null, "stream": null, "stop": null, "seed": null, "logit_bias": null, "chat_template_kwargs": null, "stream_options": null, "model": null`,
		`"max_tokens": 1`,
		`"presence_penalty": -0.5, "frequency_penalty": -2`,
		`"temperature": 2, "top_p": 1, "min_p": 0`,
		`"temperature": 3e38, "presence_penalty": -1E2, "top_p": 1.0, "logit_bias": {"0": -3e38, "7": 2}`,
		`"top_logprobs": -1, "xtc_threshold": 0.1, "xtc_probability": 0`,
		`"unknown_field": {"anything": true}, "user": "someone", "n": 1`,
	}
	for _, fields := range cases {
		t.Run(fields, func(t *testing.T) {
			h := newHarness(t, scripted(countPieces...))
			got := h.do("POST", "/v1/chat/completions", `{"messages": [], `+fields+`}`, nil)
			if got.Code != 200 {
				t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
			}
		})
	}
}

func TestContentPartsMustBeText(t *testing.T) {
	// The reference answers 404 with this message; the status is not copied.
	msg, rec := replayError(t, "chat_image_part")
	if want := `{"error": "` + msg + `"}`; !jsonEqual(t, want, rec.Response.Body) {
		t.Errorf("error = %q, want the recorded %s", msg, rec.Response.Body)
	}

	for _, part := range []string{`{"type": "text"}`, `{"type": "text", "text": 5}`, `"text"`, `{"text": "untyped"}`} {
		h := newHarness(t, scripted(countPieces...))
		got := h.do("POST", "/v1/chat/completions", `{"messages": [{"role": "user", "content": [`+part+`]}]}`, nil)
		if msg := errorMessage(t, got, 400); msg != "Only 'text' content type is supported." {
			t.Errorf("part %s: error = %q, want the reference's message", part, msg)
		}
	}
}

func TestUnsupportedFieldsAreRejected(t *testing.T) {
	for name, field := range map[string]string{
		"chat_tools":        "tools",
		"chat_tools_stream": "tools",
	} {
		t.Run(name, func(t *testing.T) {
			if msg, _ := replayError(t, name); msg != field+" is not supported yet" {
				t.Errorf("error = %q, want it to say %s is not supported yet", msg, field)
			}
		})
	}

	fields := map[string]string{
		"adapters":         `"adapters/path"`,
		"draft_model":      `"org/draft"`,
		"num_draft_tokens": `3`,
		"role_mapping":     `{"user": "USER: "}`,
		"tools":            `[{"type": "function", "function": {"name": "f"}}]`,
		"logprobs":         `true`,
		"top_logprobs":     `5`,
		"xtc_probability":  `0.5`,
	}
	requests := map[string]string{
		"/v1/chat/completions": `{"messages": [], %q: %s}`,
		"/v1/completions":      `{"prompt": "hi", %q: %s}`,
	}
	for field, value := range fields {
		for path, format := range requests {
			t.Run(path+" "+field, func(t *testing.T) {
				h := newHarness(t, scripted(countPieces...))
				got := h.do("POST", path, fmt.Sprintf(format, field, value), nil)
				if msg := errorMessage(t, got, 400); msg != field+" is not supported yet" {
					t.Errorf("error = %q, want it to say %s is not supported yet", msg, field)
				}
				if n := h.engine.Calls.Load(); n != 0 {
					t.Errorf("engine was called %d times", n)
				}
			})
		}
	}
}

func TestUnsupportedFieldsMayBeSwitchedOff(t *testing.T) {
	const request = `{"messages": [], "adapters": null, "draft_model": "", "num_draft_tokens": 0,
		"role_mapping": { }, "tools": [ ], "logprobs": false, "top_logprobs": 0,
		"xtc_probability": 0.0, "xtc_threshold": 0}`
	h := newHarness(t, scripted(countPieces...))
	if got := h.do("POST", "/v1/chat/completions", request, nil); got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}
}

func TestBodyOverSixteenMebibytesIsTooLarge(t *testing.T) {
	huge := `{"prompt": "` + strings.Repeat("a", 16<<20) + `"}`

	t.Run("declared length", func(t *testing.T) {
		h := newHarness(t, scripted(countPieces...))
		got := h.do("POST", "/v1/completions", huge, nil)
		errorMessage(t, got, 413)
		if n := h.engine.Calls.Load(); n != 0 {
			t.Errorf("engine was called %d times", n)
		}
	})

	t.Run("undeclared length", func(t *testing.T) {
		h := newHarness(t, scripted(countPieces...))
		req := httptest.NewRequest("POST", "http://127.0.0.1:8080/v1/completions", io.MultiReader(strings.NewReader(huge)))
		req.Header.Set("Content-Type", "application/json")
		if req.ContentLength != -1 {
			t.Fatalf("test request declares its length %d", req.ContentLength)
		}
		got := httptest.NewRecorder()
		h.handler.ServeHTTP(got, req)
		errorMessage(t, got, 413)
		if n := h.engine.Calls.Load(); n != 0 {
			t.Errorf("engine was called %d times", n)
		}
	})
}
