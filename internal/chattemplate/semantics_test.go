package chattemplate

import (
	"encoding/json"
	"path/filepath"
	"testing"
)

// semanticsRow is one small template exercising a single Jinja rule, with
// what Python rendered for it in Hugging Face's environment.
type semanticsRow struct {
	Name   string          `json:"name"`
	Source string          `json:"source"`
	Kwargs json.RawMessage `json:"kwargs"`
	pythonOutcome
}

// Every row is valid Jinja, so Parse must accept it; the rows Python failed
// on fail while rendering.
func TestSemanticsMatchPython(t *testing.T) {
	for _, row := range readRows[semanticsRow](t, filepath.Join("testdata", "semantics.jsonl")) {
		t.Run(row.Name, func(t *testing.T) {
			tmpl, err := Parse(row.Source, nil)
			if err != nil {
				t.Fatalf("Parse(%q): %v", row.Source, err)
			}
			got, err := tmpl.Render(Input{Messages: oneUserMessage, Kwargs: row.Kwargs})
			if mismatch := row.check(got, err); mismatch != nil {
				t.Fatalf("%s\n%v", row.Source, mismatch)
			}
		})
	}
}
