package server

import (
	"encoding/json"
	"reflect"
	"testing"
)

// shapeOf parses a JSON document and reduces it to its shape: objects keep
// their keys with the shape of each value, arrays keep the shape of their first
// element, and scalars become the name of their type. Two bodies with the same
// shape differ only in values such as ids, timestamps and generated text.
func shapeOf(t *testing.T, doc string) any {
	t.Helper()
	var v any
	if err := json.Unmarshal([]byte(doc), &v); err != nil {
		t.Fatalf("shapeOf: %v in %q", err, doc)
	}
	return shape(v)
}

func shape(v any) any {
	switch v := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(v))
		for k, e := range v {
			out[k] = shape(e)
		}
		return out
	case []any:
		if len(v) == 0 {
			return []any{}
		}
		return []any{shape(v[0])}
	case string:
		return "string"
	case float64:
		return "number"
	case bool:
		return "bool"
	default:
		return "null"
	}
}

func TestShapeOf(t *testing.T) {
	got := shapeOf(t, `{"id": "x", "n": 1.5, "ok": true, "none": null, "list": [{"a": 1}, {"b": "ignored"}], "empty": []}`)
	want := map[string]any{
		"id":    "string",
		"n":     "number",
		"ok":    "bool",
		"none":  "null",
		"list":  []any{map[string]any{"a": "number"}},
		"empty": []any{},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("shapeOf = %#v, want %#v", got, want)
	}
}
