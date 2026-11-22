package chattemplate

import (
	"encoding/json"
	"path/filepath"
	"testing"
)

func TestQwen38RenderMatchesPython(t *testing.T) {
	tmpl := loadVendoredTemplate(t, "qwen38")
	for _, row := range readRows[requestRow](t, filepath.Join(vendoredDir("qwen38"), "template.jsonl"), 12) {
		t.Run(row.Name, func(t *testing.T) {
			got, err := tmpl.Render(row.input())
			if mismatch := row.check(got, err); mismatch != nil {
				t.Fatal(mismatch)
			}
		})
	}
}

func TestQwen38DefaultFilter(t *testing.T) {
	// jinja2.Environment renders fallback only for Undefined unless boolean
	// mode is requested. An explicit None is a defined value.
	checkQwen38Expression(t, `{{ missing|default('x') }}|{{ none|default('x') }}|{{ false|default('x') }}|{{ ''|default('x') }}|{{ 0|default('x') }}`, "x|None|False||0")
	checkQwen38Expression(t, `{{ missing|default }}|{{ 0|default('x', true) }}|{{ ''|default(default_value='x', boolean=true) }}|{{ 0|default('x', boolean=false) }}|{{ 'ok'|default('x', true) }}`, "|x|x|0|ok")
}

func TestQwen38TupleLiteral(t *testing.T) {
	checkQwen38Expression(t, `{{ 'low' in ('xhigh', 'medium', 'low') }}|{{ 'bad' not in ('low',) }}|{{ (2 + 3) }}|{{ (2 + 3,) }}|{{ () }}`, "True|True|5|(5,)|()")
}

func checkQwen38Expression(t *testing.T, source, want string) {
	t.Helper()
	tmpl, err := Parse(source, nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := tmpl.Render(Input{Messages: json.RawMessage(`[{"role":"user","content":"hi"}]`)})
	if err != nil || got != want {
		t.Fatalf("Render = (%q, %v), want %q", got, err, want)
	}
}
