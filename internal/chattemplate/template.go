// Package chattemplate renders a model's Jinja chat template the way Python's
// tokenizer.apply_chat_template does.
//
// It is a small, strict interpreter for the Jinja constructs chat templates
// use, with Python's semantics. A construct it does not implement is a Parse
// error naming the construct; an operation Python would reject is a Render
// error. It never renders a prompt Python would not.
package chattemplate

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"strings"
)

// Template is a parsed chat template. It is immutable after Load or Parse.
type Template struct {
	body          []statement
	specialTokens map[string]string
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
	return &Template{body: body, specialTokens: maps.Clone(specialTokens)}, nil
}

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

// requestFields are the template variables Render sets from Input. Python
// refuses them among the kwargs, as a duplicate keyword argument.
var requestFields = []string{"messages", "tools", "add_generation_prompt"}

// variables builds the template's top-level variables the way
// apply_chat_template does: special tokens, overridden by kwargs, plus the
// request fields. It rejects what apply_chat_template rejects before it
// renders.
func (t *Template) variables(in Input) (map[string]any, error) {
	variables := map[string]any{"documents": nil}
	for name, token := range t.specialTokens {
		variables[name] = token
	}
	kwargs, err := decodeOptional(in.Kwargs)
	if err != nil {
		return nil, fmt.Errorf("kwargs: %w", err)
	}
	switch kwargs := kwargs.(type) {
	case nil:
	case *dict:
		for _, field := range requestFields {
			if _, clash := kwargs.values[field]; clash {
				return nil, fmt.Errorf("kwargs: %q is a request field, not a template variable to pass", field)
			}
		}
		maps.Copy(variables, kwargs.values)
	default:
		return nil, errors.New("kwargs: not a JSON object")
	}

	messages, err := decodeValue(in.Messages)
	if err != nil {
		return nil, fmt.Errorf("messages: %w", err)
	}
	conversation, ok := messages.([]any)
	switch {
	case !ok:
		return nil, errors.New("messages: not a JSON array")
	case len(conversation) == 0:
		return nil, errors.New("messages: the conversation is empty")
	}
	if _, nested := conversation[0].([]any); nested {
		// Python takes a list whose first item is a list for several
		// conversations and returns one prompt for each.
		return nil, errors.New("messages: a batch of conversations is not supported")
	}
	tools, err := decodeOptional(in.Tools)
	if err != nil {
		return nil, fmt.Errorf("tools: %w", err)
	}
	if tools != nil {
		list, ok := tools.([]any)
		if !ok {
			return nil, errors.New("tools: not a JSON array")
		}
		for i, tool := range list {
			if _, ok := tool.(*dict); !ok {
				return nil, fmt.Errorf("tools: tool %d is not a JSON object", i)
			}
		}
	}
	variables["messages"] = messages
	variables["tools"] = tools
	variables["add_generation_prompt"] = in.AddGenerationPrompt
	return variables, nil
}
