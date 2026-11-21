package server

import (
	"fmt"
	"io"
	"net/http/httptest"
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

func TestRecordedBadRequestsAreRejectedByName(t *testing.T) {
	// The reference drops the connection or answers 404 for these; see the
	// recordings. This server answers 400 with the reference's wording.
	cases := map[string]string{
		"err_missing_messages":           "Request did not contain messages",
		"err_messages_wrong_type":        "messages must be of type list",
		"err_stream_wrong_type":          "stream must be of type bool",
		"err_max_tokens_negative":        "max_tokens must be at least 0",
		"err_temperature_negative":       "temperature must be at least 0",
		"err_top_p_range":                "top_p must be at most 1",
		"err_logit_bias_type":            "logit_bias must be of type dict",
		"err_completions_missing_prompt": "Request did not contain a prompt",
	}
	for name, want := range cases {
		t.Run(name, func(t *testing.T) {
			if msg, _ := replayError(t, name); msg != want {
				t.Errorf("error = %q, want %q", msg, want)
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
		{`"max_completion_tokens": -1`, "max_completion_tokens must be at least 0"},
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
		`"max_tokens": 0`,
		`"presence_penalty": -0.5, "frequency_penalty": -2`,
		`"temperature": 2, "top_p": 1, "min_p": 0`,
		`"temperature": 1e39, "presence_penalty": 1E2, "top_p": 1.0`,
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
		"chat_tools":         "tools",
		"chat_tools_stream":  "tools",
		"chat_logprobs":      "logprobs",
		"err_logprobs_range": "logprobs",
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
		"xtc_threshold":    `0.1`,
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
