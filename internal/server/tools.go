package server

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

const toolOpen = "<tool_call>"

// toolSet is nil when this request is not allowed to produce tool calls.
type toolSet struct {
	functions map[string]parameterSchema
	parallel  bool
}

type parameterSchema struct {
	Properties map[string]struct {
		Type parameterKind `json:"type"`
	} `json:"properties"`
	Required []string `json:"required"`
}

// JSON Schema type can be one kind or a list of alternatives (e.g. number
// or null for OMP's grep skip parameter).
type parameterKind struct {
	single       string
	alternatives []string
}

func (k *parameterKind) UnmarshalJSON(raw []byte) error {
	if raw[0] == '[' {
		return json.Unmarshal(raw, &k.alternatives)
	}
	return json.Unmarshal(raw, &k.single)
}

func (r *reader) chatTools(supported bool) (json.RawMessage, *toolSet) {
	if !supported {
		r.unsupported("tools", r.given("tools"))
		return json.RawMessage("null"), nil
	}
	choice := r.text("tool_choice", "auto")
	if choice != "auto" && choice != "none" {
		r.fail("tool_choice must be auto or none")
	}
	parallel := r.boolean("parallel_tool_calls", true)
	raw, ok := r.raw("tools")
	if !ok {
		return json.RawMessage("null"), nil
	}
	var declarations []struct {
		Type     string `json:"type"`
		Function struct {
			Name       string          `json:"name"`
			Parameters parameterSchema `json:"parameters"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &declarations); err != nil {
		r.fail("tools must be an array of function definitions: %v", err)
		return nil, nil
	}
	if choice == "none" || len(declarations) == 0 {
		return json.RawMessage("null"), nil
	}
	set := &toolSet{functions: make(map[string]parameterSchema, len(declarations)), parallel: parallel}
	for _, decl := range declarations {
		name := decl.Function.Name
		if decl.Type != "function" || !functionName(name) {
			r.fail("tools must declare named functions")
			return nil, nil
		}
		if _, exists := set.functions[name]; exists {
			r.fail("duplicate tool %q", name)
			return nil, nil
		}
		set.functions[name] = decl.Function.Parameters
	}
	return raw, set
}

func functionName(name string) bool {
	if len(name) < 1 || len(name) > 64 {
		return false
	}
	for _, c := range name {
		if !(c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_' || c == '-') {
			return false
		}
	}
	return true
}

type toolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function toolFunction `json:"function"`
	Index    *int         `json:"index,omitempty"`
}

type toolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // OpenAI transports the JSON object as a string.
}

// toolOutput streams text up to the first call. It holds the whole call suffix
// so a malformed or truncated turn cannot authorize even one of its calls.
// Only a possible prefix of the opening marker is held while streaming text.
type toolOutput struct {
	pending int
	started bool
	body    strings.Builder
}

func (o *toolOutput) add(piece string, emit func(string, bool) error) error {
	if o.started {
		o.body.WriteString(piece)
		return nil
	}
	if o.pending > 0 {
		n := 0
		for n < len(piece) && o.pending < len(toolOpen) && piece[n] == toolOpen[o.pending] {
			n++
			o.pending++
		}
		piece = piece[n:]
		switch {
		case o.pending == len(toolOpen):
			o.started = true
			o.body.WriteString(piece)
		case piece == "":
			return nil
		default:
			if err := emit(toolOpen[:o.pending], false); err != nil {
				return err
			}
		}
		o.pending = 0
		if o.started {
			return nil
		}
	}
	text, rest, found := strings.Cut(piece, toolOpen)
	if found {
		o.started = true
		o.body.WriteString(rest)
	} else if at := strings.LastIndexByte(text, '<'); at >= 0 && strings.HasPrefix(toolOpen, text[at:]) {
		o.pending = len(text) - at
		text = text[:at]
	}
	if text != "" {
		return emit(text, false)
	}
	return nil
}

func (o *toolOutput) finish(set *toolSet, exhausted bool, emit func(string, bool) error) ([]toolCall, error) {
	if !o.started {
		if o.pending != 0 && !exhausted {
			return nil, emit(toolOpen[:o.pending], false)
		}
		return nil, nil
	}
	if exhausted {
		return nil, nil
	}
	var calls []toolCall
	body := o.body.String()
	for {
		call, rest, err := set.parseCall(body)
		if err != nil {
			return nil, fmt.Errorf("invalid generated tool call: %w", err)
		}
		calls = append(calls, call)
		rest = strings.TrimSpace(rest)
		if rest == "" {
			return calls, nil
		}
		if !set.parallel {
			return nil, fmt.Errorf("model generated more than one tool call with parallel_tool_calls disabled")
		}
		var ok bool
		body, ok = strings.CutPrefix(rest, toolOpen)
		if !ok {
			return nil, fmt.Errorf("unexpected content after generated tool call")
		}
	}
}

func (set *toolSet) parseCall(body string) (toolCall, string, error) {
	name, body, ok := tagged(body, "<function=")
	if !ok {
		return toolCall{}, "", fmt.Errorf("expected function tag")
	}
	schema, ok := set.functions[name]
	if !ok {
		return toolCall{}, "", fmt.Errorf("unknown function %q", name)
	}
	args := make(map[string]json.RawMessage)
	for {
		key, rest, ok := tagged(body, "<parameter=")
		if !ok {
			break
		}
		separator := "</parameter>"
		if s, multiline := strings.CutPrefix(rest, "\n"); multiline {
			rest, separator = s, "\n</parameter>"
		}
		value, rest, ok := strings.Cut(rest, separator)
		if !ok || key == "" || args[key] != nil {
			return toolCall{}, "", fmt.Errorf("invalid parameter %q", key)
		}
		v, err := parameterValue(value, schema.Properties[key].Type)
		if err != nil {
			return toolCall{}, "", fmt.Errorf("parameter %q: %w", key, err)
		}
		args[key], body = v, rest
	}
	for _, key := range schema.Required {
		if args[key] == nil {
			return toolCall{}, "", fmt.Errorf("missing required parameter %q", key)
		}
	}
	body, ok = strings.CutPrefix(strings.TrimSpace(body), "</function>")
	if !ok {
		return toolCall{}, "", fmt.Errorf("expected closing function tag")
	}
	body, ok = strings.CutPrefix(strings.TrimSpace(body), "</tool_call>")
	if !ok {
		return toolCall{}, "", fmt.Errorf("expected closing tool call tag")
	}
	raw, err := json.Marshal(args)
	if err != nil {
		return toolCall{}, "", err
	}
	return toolCall{ID: newID("call_"), Type: "function", Function: toolFunction{Name: name, Arguments: string(raw)}}, body, nil
}

func tagged(body, prefix string) (key, rest string, ok bool) {
	body, ok = strings.CutPrefix(strings.TrimSpace(body), prefix)
	if !ok {
		return "", "", false
	}
	return strings.Cut(body, ">")
}

// Qwen sends unquoted strings but JSON for typed values. Within a union
// prefer a matching non-string JSON kind, then fall back to the raw string
// if allowed. A string-only parameter keeps literal "null" and numbers.
func parameterValue(value string, kind parameterKind) (json.RawMessage, error) {
	if len(kind.alternatives) == 0 && (kind.single == "" || kind.single == "string") {
		return json.Marshal(value)
	}
	s := strings.TrimSpace(value)
	if json.Valid([]byte(s)) {
		if matchesJSONKind(s, kind.single) {
			return json.RawMessage(s), nil
		}
		for _, alternative := range kind.alternatives {
			if matchesJSONKind(s, alternative) {
				return json.RawMessage(s), nil
			}
		}
	}
	if slices.Contains(kind.alternatives, "string") {
		return json.Marshal(value)
	}
	expected := kind.single
	if len(kind.alternatives) != 0 {
		expected = strings.Join(kind.alternatives, " or ")
	}
	return nil, fmt.Errorf("expected JSON %s", expected)
}

// s is known to be valid JSON.
func matchesJSONKind(s, kind string) bool {
	switch kind {
	case "boolean":
		return s == "true" || s == "false"
	case "null":
		return s == "null"
	case "object":
		return s[0] == '{'
	case "array":
		return s[0] == '['
	case "number", "integer":
		return (s[0] == '-' || s[0] >= '0' && s[0] <= '9') && (kind == "number" || !strings.ContainsAny(s, ".eE"))
	}
	return false
}
