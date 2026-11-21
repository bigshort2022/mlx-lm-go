package chattemplate

import (
	"strings"
	"testing"
)

// rejection is a request Render must refuse with an error that says why.
type rejection struct {
	name string
	in   Input
	want string // must appear in the error
}

func withKwargs(kwargs string) Input {
	return Input{Messages: oneUserMessage, Kwargs: []byte(kwargs), AddGenerationPrompt: true}
}

func checkRejections(t *testing.T, cases []rejection) {
	t.Helper()
	tmpl, err := Parse("static", nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		got, err := tmpl.Render(c.in)
		switch {
		case err == nil:
			t.Errorf("%s: Render returned %q, want an error with %q", c.name, got, c.want)
		case !strings.Contains(err.Error(), c.want):
			t.Errorf("%s: error %q, want one with %q", c.name, err, c.want)
		}
	}
}

// Python raises on each of these requests whatever the template: json.loads
// on the first group, apply_chat_template on the rest. The differential
// fixtures hold Python's verdict on the second group; this pins that the
// error names what was wrong.
func TestRenderRejectsRequestsPythonRejects(t *testing.T) {
	checkRejections(t, []rejection{
		{"messages missing", Input{}, "messages: "},
		{"messages malformed", Input{Messages: []byte(`[{"role": `)}, "messages: "},
		{"messages with trailing data", Input{Messages: []byte(`[] []`)}, "messages: "},
		{"invalid UTF-8 in messages", Input{Messages: []byte("[{\"role\": \"user\", \"content\": \"a\xffb\"}]")}, "messages: invalid UTF-8"},
		{"invalid UTF-8 in tools", Input{Messages: oneUserMessage, Tools: []byte("[{\"name\": \"\xc3(\"}]")}, "tools: invalid UTF-8"},
		{"invalid UTF-8 in kwargs", withKwargs("{\"a\": \"\xed\xa0\x80\"}"), "kwargs: invalid UTF-8"},
		{"kwargs not an object", withKwargs(`[1, 2]`), "kwargs: not a JSON object"},

		{"no messages", Input{Messages: []byte(`[]`)}, "messages: the conversation is empty"},
		{"tools not an array", Input{Messages: oneUserMessage, Tools: []byte(`{"a": 1}`)}, "tools: not a JSON array"},
		{"tool not an object", Input{Messages: oneUserMessage, Tools: []byte(`["get_weather"]`)}, "tools: tool 0 is not a JSON object"},
		{"kwargs replacing messages", withKwargs(`{"messages": []}`), `kwargs: "messages" is`},
		{"kwargs replacing tools", withKwargs(`{"tools": []}`), `kwargs: "tools" is`},
		{"kwargs replacing the generation prompt", withKwargs(`{"add_generation_prompt": true}`), `kwargs: "add_generation_prompt" is`},
		{"kwargs replacing tokenize", withKwargs(`{"tokenize": false}`), `kwargs: "tokenize" is`},
		{"kwargs replacing the conversation", withKwargs(`{"conversation": []}`), `kwargs: "conversation" is`},
		{"kwargs replacing self", withKwargs(`{"self": 1}`), `kwargs: "self" is`},
		{"assistant token mask without tokenizing", withKwargs(`{"return_assistant_tokens_mask": true}`), `kwargs: "return_assistant_tokens_mask" is`},
		{"continuing the final message after a generation prompt", withKwargs(`{"continue_final_message": true}`), `kwargs: "continue_final_message" is`},
		{"documents of strings", withKwargs(`{"documents": ["a"]}`), "kwargs: documents: document 0 is not a JSON object"},
		{"documents with one string", withKwargs(`{"documents": [{"title": "t"}, "a"]}`), "kwargs: documents: document 1 is not a JSON object"},
		{"documents as a string", withKwargs(`{"documents": "abc"}`), "kwargs: documents: not a JSON array"},
		{"documents as a number", withKwargs(`{"documents": 5}`), "kwargs: documents: not a JSON array"},
		{"documents as an object", withKwargs(`{"documents": {"a": 1}}`), "kwargs: documents: not a JSON array"},
	})
}

// Python accepts each of these requests. This package refuses them, each for
// the reason given; the package comment lists these limits.
func TestRenderRejectsWhatPythonAccepts(t *testing.T) {
	withoutGenerationPrompt := func(kwargs string) Input {
		in := withKwargs(kwargs)
		in.AddGenerationPrompt = false
		return in
	}
	checkRejections(t, []rejection{
		// Python renders a template that never looks at messages.
		{"messages as an object", Input{Messages: []byte(`{"role": "user"}`)}, "messages: not a JSON array"},
		// Python returns one prompt for each conversation.
		{"a batch of conversations", Input{Messages: []byte(`[[{"role": "user", "content": "hi"}]]`)}, "messages: a batch of conversations is not supported"},
		// A Python str can hold half a surrogate pair; a Go string cannot.
		{"lone high surrogate", Input{Messages: []byte(`[{"role": "user", "content": "a\ud800b"}]`)}, `messages: unpaired surrogate escape \ud800`},
		{"lone low surrogate", Input{Messages: oneUserMessage, Tools: []byte(`[{"name": "\uDC00"}]`)}, `tools: unpaired surrogate escape \uDC00`},
		{"high surrogate before another escape", withKwargs(`{"a": "\ud83d\u0041"}`), `kwargs: unpaired surrogate escape \ud83d`},
		{"high surrogate at the end of a string", withKwargs(`{"a": "x\\\ud83d"}`), `kwargs: unpaired surrogate escape \ud83d`},
		{"two high surrogates", withKwargs(`{"a": "\ud83d\ud83d\ude00"}`), `kwargs: unpaired surrogate escape \ud83d`},
		// json.loads reads these as floats; they are not JSON.
		{"NaN literal", withKwargs(`{"a": NaN}`), "kwargs: "},
		{"Infinity literal", withKwargs(`{"a": -Infinity}`), "kwargs: "},
		// Python accepts {} because it has no items to check.
		{"documents as an empty object", withKwargs(`{"documents": {}}`), "kwargs: documents: not a JSON array"},
		// Parameters of apply_chat_template, not template variables. Python
		// acts on them: chat_template replaces the model's template,
		// continue_final_message cuts the prompt, the rest configure the
		// tokenizer call that follows.
		{"chat_template", withKwargs(`{"chat_template": "{{ messages }}"}`), `kwargs: "chat_template" is`},
		{"continue_final_message", withoutGenerationPrompt(`{"continue_final_message": true}`), `kwargs: "continue_final_message" is`},
		{"continue_final_message off", withKwargs(`{"continue_final_message": false}`), `kwargs: "continue_final_message" is`},
		{"return_assistant_tokens_mask off", withKwargs(`{"return_assistant_tokens_mask": false}`), `kwargs: "return_assistant_tokens_mask" is`},
		{"padding", withKwargs(`{"padding": true}`), `kwargs: "padding" is`},
		{"truncation", withKwargs(`{"truncation": true}`), `kwargs: "truncation" is`},
		{"max_length", withKwargs(`{"max_length": 5}`), `kwargs: "max_length" is`},
		{"return_tensors", withKwargs(`{"return_tensors": "np"}`), `kwargs: "return_tensors" is`},
		{"return_dict", withKwargs(`{"return_dict": false}`), `kwargs: "return_dict" is`},
		{"tokenizer_kwargs", withKwargs(`{"tokenizer_kwargs": {}}`), `kwargs: "tokenizer_kwargs" is`},
	})
}

// A surrogate pair written as two escapes is one character.
func TestSurrogatePairEscapesAreOneCharacter(t *testing.T) {
	got := mustRenderSource(t, "{{ v }}|{{ v | length }}|{{ v | tojson }}", `{"v": "\ud83d\ude00 \\ud800 \\\\"}`)
	if want := "😀 \\ud800 \\\\|11|\"😀 \\\\ud800 \\\\\\\\\""; got != want {
		t.Fatalf("got %s, want %s", got, want)
	}
}
