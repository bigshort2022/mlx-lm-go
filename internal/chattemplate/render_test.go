package chattemplate

import "testing"

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

// json.loads keeps the last value of a repeated key, at the first position.
func TestRepeatedObjectKeyKeepsItsLastValue(t *testing.T) {
	got := mustRenderSource(t, "{{ v | tojson }}", `{"v": {"a": 1, "b": 2, "a": 3}}`)
	if want := `{"a": 3, "b": 2}`; got != want {
		t.Fatalf("got %s, want %s", got, want)
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
