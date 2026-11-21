package chattemplate

import (
	"strings"
	"testing"
)

// oneUserMessage is the conversation the tests of single Jinja rules render
// with. tools/oracle/make_template_differential.py uses the same one.
var oneUserMessage = []byte(`[{"role": "user", "content": "hi"}]`)

// renderSource parses source and renders it with oneUserMessage and the given
// kwargs JSON. The expected strings in the tests that call it were produced
// by Python with the environment Hugging Face builds in
// chat_template_utils.py.
func renderSource(t *testing.T, source, kwargs string) (string, error) {
	t.Helper()
	tmpl, err := Parse(source, nil)
	if err != nil {
		t.Fatalf("Parse(%q): %v", source, err)
	}
	in := Input{Messages: oneUserMessage}
	if kwargs != "" {
		in.Kwargs = []byte(kwargs)
	}
	return tmpl.Render(in)
}

func mustRenderSource(t *testing.T, source, kwargs string) string {
	t.Helper()
	got, err := renderSource(t, source, kwargs)
	if err != nil {
		t.Fatalf("Render(%q): %v", source, err)
	}
	return got
}

func TestTrimBlocksRemovesNewlineAfterBlockTag(t *testing.T) {
	got := mustRenderSource(t, "{% if true %}\nyes\n{% endif %}\nend", "")
	if want := "yes\nend"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestLstripBlocksRemovesIndentBeforeBlockTag(t *testing.T) {
	got := mustRenderSource(t, "  {% if true %}\n  yes\n  {% endif %}\nend", "")
	if want := "  yes\nend"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestUndefinedVariableIsFalsy(t *testing.T) {
	got := mustRenderSource(t, "{% if missing %}a{% else %}b{% endif %}", "")
	if want := "b"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestKwargsBecomeTopLevelVariables(t *testing.T) {
	got := mustRenderSource(t, "{{ a }}-{{ b is defined }}", `{"a": 1}`)
	if want := "1-False"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestKwargsOverrideSpecialTokens(t *testing.T) {
	tmpl, err := Parse("{{ bos_token }}{{ eos_token }}", map[string]string{"bos_token": "<s>", "eos_token": "</s>"})
	if err != nil {
		t.Fatal(err)
	}
	got, err := tmpl.Render(Input{Messages: oneUserMessage, Kwargs: []byte(`{"eos_token": "<end>"}`)})
	if err != nil {
		t.Fatal(err)
	}
	if want := "<s><end>"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// apply_chat_template passes documents=None unless the caller gives some.
func TestKwargsCanProvideDocuments(t *testing.T) {
	got := mustRenderSource(t, "{{ documents[0].title }}", `{"documents": [{"title": "T", "text": "x"}]}`)
	if want := "T"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestNullKwargsAreNoKwargs(t *testing.T) {
	got := mustRenderSource(t, "{{ a is defined }}", `null`)
	if want := "False"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestToolsDefaultToNone(t *testing.T) {
	for _, tools := range []string{"", "null"} {
		tmpl, err := Parse("{{ tools is none }}", nil)
		if err != nil {
			t.Fatal(err)
		}
		got, err := tmpl.Render(Input{Messages: oneUserMessage, Tools: []byte(tools)})
		if err != nil || got != "True" {
			t.Errorf("tools %q: Render returned (%q, %v), want True", tools, got, err)
		}
	}
}

// Each of these requests makes apply_chat_template raise before or instead of
// rendering, whatever the template.
func TestRenderRejectsRequestsPythonRejects(t *testing.T) {
	tmpl, err := Parse("static", nil)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name string
		in   Input
		want string
	}{
		{"no messages", Input{Messages: []byte(`[]`)}, "messages: the conversation is empty"},
		{"messages missing", Input{}, "messages: "},
		{"messages not an array", Input{Messages: []byte(`{"role": "user"}`)}, "messages: not a JSON array"},
		{"messages malformed", Input{Messages: []byte(`[{"role": `)}, "messages: "},
		{"messages with trailing data", Input{Messages: []byte(`[] []`)}, "messages: "},
		{"a batch of conversations", Input{Messages: []byte(`[[{"role": "user", "content": "hi"}]]`)}, "messages: a batch of conversations is not supported"},
		{"tools not an array", Input{Messages: oneUserMessage, Tools: []byte(`{"a": 1}`)}, "tools: not a JSON array"},
		{"tool not an object", Input{Messages: oneUserMessage, Tools: []byte(`["get_weather"]`)}, "tools: tool 0 is not a JSON object"},
		{"kwargs not an object", Input{Messages: oneUserMessage, Kwargs: []byte(`[1, 2]`)}, "kwargs: not a JSON object"},
		{"kwargs replacing messages", Input{Messages: oneUserMessage, Kwargs: []byte(`{"messages": []}`)}, `kwargs: "messages" is a request field`},
		{"kwargs replacing tools", Input{Messages: oneUserMessage, Kwargs: []byte(`{"tools": []}`)}, `kwargs: "tools" is a request field`},
		{"kwargs replacing the generation prompt", Input{Messages: oneUserMessage, Kwargs: []byte(`{"add_generation_prompt": true}`)}, `kwargs: "add_generation_prompt" is a request field`},
		{"integer beyond int64", Input{Messages: oneUserMessage, Kwargs: []byte(`{"n": 123456789012345678901234567890}`)}, "integer 123456789012345678901234567890 is out of range"},
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

// json.loads keeps the last value of a repeated key, at the first position.
func TestRepeatedObjectKeyKeepsItsLastValue(t *testing.T) {
	got := mustRenderSource(t, "{{ v | tojson }}", `{"v": {"a": 1, "b": 2, "a": 3}}`)
	if want := `{"a": 3, "b": 2}`; got != want {
		t.Fatalf("got %s, want %s", got, want)
	}
}

// Python integers do not overflow. Ours fail rather than wrap around.
func TestIntegerOverflowIsAnError(t *testing.T) {
	for _, source := range []string{
		"{{ big + 1 }}",
		"{{ 0 - big - 2 }}",
		"{{ -small }}",
	} {
		got, err := renderSource(t, source, `{"big": 9223372036854775807, "small": -9223372036854775808}`)
		if err == nil || !strings.Contains(err.Error(), "integer result is out of range") {
			t.Errorf("%s: Render returned (%q, %v), want an out-of-range error", source, got, err)
		}
	}
}

func TestIntegerArithmeticAtTheEdgeOfTheRange(t *testing.T) {
	got := mustRenderSource(t, "{{ big - 1 + 1 }}|{{ small + 1 - 1 }}|{{ -big }}",
		`{"big": 9223372036854775807, "small": -9223372036854775808}`)
	if want := "9223372036854775807|-9223372036854775808|-9223372036854775807"; got != want {
		t.Fatalf("got %s, want %s", got, want)
	}
}

// Python stops a macro that calls itself after a little under two hundred
// levels; a hundred must work here too.
func TestMacroCanRecurseAHundredLevels(t *testing.T) {
	got := mustRenderSource(t,
		"{% macro down(n) %}{% if n > 0 %}{{ down(n - 1) }}{% endif %}{% endmacro %}{{ down(100) }}ok", "")
	if got != "ok" {
		t.Fatalf("got %q, want %q", got, "ok")
	}
}
