package chattemplate

import (
	"encoding/json"
	"maps"
	"strings"
)

// Template is a parsed chat template. It is immutable after Load or Parse.
type Template struct {
	body          []statement
	specialTokens map[string]string
	qwenXMLTools  bool
}

// Input is one chat completion request, as far as the template is concerned.
type Input struct {
	Messages            json.RawMessage // JSON array, as received
	Tools               json.RawMessage // JSON array; nil or "null" when the request has no tools
	Kwargs              json.RawMessage // JSON object of extra template variables; may be nil
	AddGenerationPrompt bool
}

// TemplateError is returned when the template calls raise_exception.
type TemplateError struct{ Msg string }

func (e *TemplateError) Error() string { return e.Msg }

// Parse builds a Template from source text.
func Parse(source string, specialTokens map[string]string) (*Template, error) {
	body, err := parse(source)
	if err != nil {
		return nil, err
	}
	return &Template{
		body: body, specialTokens: maps.Clone(specialTokens),
		qwenXMLTools: strings.Contains(source, "<tool_call>") &&
			strings.Contains(source, "<function=") && strings.Contains(source, "<parameter="),
	}, nil
}

// UsesQwenXMLTools reports whether the loaded template instructs the model
// to use Qwen's function/parameter tags rather than JSON tool-call bodies.
func (t *Template) UsesQwenXMLTools() bool { return t.qwenXMLTools }

// Render renders the template for one request.
func (t *Template) Render(in Input) (string, error) {
	variables, err := t.variables(in)
	if err != nil {
		return "", err
	}
	var out strings.Builder
	if err := execAll(t.body, &out, newRootScope(variables)); err != nil {
		return "", err
	}
	return out.String(), nil
}
