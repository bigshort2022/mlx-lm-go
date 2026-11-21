package chattemplate

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/hub"
)

// goldenFamilies maps each fixture family to the model whose chat template
// produced it.
var goldenFamilies = []struct {
	name string
	repo string
}{
	{"qwen2", "mlx-community/Qwen2.5-1.5B-Instruct-4bit"},
	{"qwen3_5", "mlx-community/Qwen3.5-9B-MLX-4bit"},
}

// pythonOutcome is what Python did with a fixture row: the string it rendered
// or the exception it raised, as "ExceptionClass: message".
type pythonOutcome struct {
	Prompt *string `json:"prompt"`
	Error  string  `json:"error"`
}

// check compares one Render outcome with Python's and describes the mismatch,
// if any.
func (want pythonOutcome) check(got string, err error) error {
	const templateErrorPrefix = "TemplateError: "
	switch {
	case want.Prompt != nil:
		if err != nil {
			return fmt.Errorf("Render: %w", err)
		}
		if got != *want.Prompt {
			return errors.New(firstDifference(got, *want.Prompt))
		}
	case strings.HasPrefix(want.Error, templateErrorPrefix):
		message := strings.TrimPrefix(want.Error, templateErrorPrefix)
		var templateErr *TemplateError
		if !errors.As(err, &templateErr) {
			return fmt.Errorf("Render returned (%q, %v), want *TemplateError %q", got, err, message)
		}
		if templateErr.Msg != message {
			return fmt.Errorf("TemplateError.Msg = %q, want %q", templateErr.Msg, message)
		}
	case want.Error != "":
		if err == nil {
			return fmt.Errorf("Render returned %q, want an error (Python raised %s)", got, want.Error)
		}
	default:
		return errors.New("fixture row has neither prompt nor error")
	}
	return nil
}

// firstDifference describes where got and want diverge, with 40 bytes of
// context on each side of the first differing offset.
func firstDifference(got, want string) string {
	offset := 0
	for offset < len(got) && offset < len(want) && got[offset] == want[offset] {
		offset++
	}
	const context = 40
	window := func(s string) string {
		return s[max(0, offset-context):min(len(s), offset+context)]
	}
	return fmt.Sprintf("prompts differ at byte %d (got %d bytes, want %d)\n got: %q\nwant: %q",
		offset, len(got), len(want), window(got), window(want))
}

// requestRow is one request and its outcome, as written by
// tools/oracle/make_text_fixtures.py and make_template_differential.py. The
// request fields stay raw so that Render sees exactly the bytes Python saw.
type requestRow struct {
	Name                string          `json:"name"`
	Messages            json.RawMessage `json:"messages"`
	Tools               json.RawMessage `json:"tools"`
	Kwargs              json.RawMessage `json:"kwargs"`
	AddGenerationPrompt bool            `json:"add_generation_prompt"`
	pythonOutcome
}

func (r requestRow) input() Input {
	return Input{
		Messages:            r.Messages,
		Tools:               r.Tools,
		Kwargs:              r.Kwargs,
		AddGenerationPrompt: r.AddGenerationPrompt,
	}
}

// readRows decodes every line of a JSON Lines fixture.
func readRows[Row any](t *testing.T, path string) []Row {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var rows []Row
	for dec := json.NewDecoder(file); dec.More(); {
		var row Row
		if err := dec.Decode(&row); err != nil {
			t.Fatalf("%s row %d: %v", path, len(rows)+1, err)
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		t.Fatalf("%s has no rows", path)
	}
	return rows
}

// loadModelTemplate loads a model's template from the local Hugging Face
// cache. The test is skipped when the model has not been downloaded.
func loadModelTemplate(t *testing.T, repo string) *Template {
	t.Helper()
	dir, err := hub.Resolve(repo)
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	tmpl, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	return tmpl
}

// checkRequestRows renders every row of a fixture with the family's template
// and compares the outcome with Python's.
func checkRequestRows(t *testing.T, path func(family string) string) {
	for _, family := range goldenFamilies {
		t.Run(family.name, func(t *testing.T) {
			tmpl := loadModelTemplate(t, family.repo)
			for _, row := range readRows[requestRow](t, path(family.name)) {
				t.Run(row.Name, func(t *testing.T) {
					got, err := tmpl.Render(row.input())
					if mismatch := row.check(got, err); mismatch != nil {
						t.Fatal(mismatch)
					}
				})
			}
		})
	}
}

func goldenPath(family string) string {
	return filepath.Join("..", "..", "testdata", family, "template.jsonl")
}

func differentialPath(family string) string {
	return filepath.Join("testdata", "differential_"+family+".jsonl")
}

func TestRenderMatchesGolden(t *testing.T) {
	checkRequestRows(t, goldenPath)
}

// The differential rows are requests a server can receive that the golden
// rows do not cover, including the ones Python rejects.
func TestRenderMatchesDifferential(t *testing.T) {
	checkRequestRows(t, differentialPath)
}
