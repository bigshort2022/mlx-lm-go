package chattemplate

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

// semanticsRow is one small template exercising a single Jinja rule, with
// what Python did with it in Hugging Face's environment.
type semanticsRow struct {
	Name   string          `json:"name"`
	Source string          `json:"source"`
	Kwargs json.RawMessage `json:"kwargs"`
	// KwargsJSON replaces Kwargs in rows whose variables must reach the
	// renderer as exact JSON text, such as the number 1e309.
	KwargsJSON string `json:"kwargs_json"`
	pythonOutcome
}

// failsToCompile reports whether Python rejected the template itself, which
// Parse must then do too; every other error is raised while rendering.
func (r semanticsRow) failsToCompile() bool {
	for _, class := range []string{"TemplateSyntaxError: ", "TemplateAssertionError: ", "SyntaxError: "} {
		if strings.HasPrefix(r.Error, class) {
			return true
		}
	}
	return false
}

func TestSemanticsMatchPython(t *testing.T) {
	rows := readRows[semanticsRow](t, filepath.Join("testdata", "semantics.jsonl"), semanticsRowCount)
	for _, row := range rows {
		t.Run(row.Name, func(t *testing.T) {
			tmpl, err := Parse(row.Source, nil)
			switch {
			case row.failsToCompile() && err == nil:
				t.Fatalf("Parse(%q) succeeded, want an error (Python raised %s)", row.Source, row.Error)
			case row.failsToCompile():
				return
			case err != nil:
				t.Fatalf("Parse(%q): %v", row.Source, err)
			}
			kwargs := row.Kwargs
			if row.KwargsJSON != "" {
				kwargs = json.RawMessage(row.KwargsJSON)
			}
			got, err := tmpl.Render(Input{Messages: oneUserMessage, Kwargs: kwargs})
			if mismatch := row.check(got, err); mismatch != nil {
				t.Fatalf("%s\n%v", row.Source, mismatch)
			}
		})
	}
}
