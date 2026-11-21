package chattemplate

import (
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"
)

// filterFunc implements "value|filter(options)".
type filterFunc func(value any, options []namedValue) (any, error)

// filterSpec is a filter and the keyword arguments it accepts. No supported
// filter takes positional arguments.
type filterSpec struct {
	apply   filterFunc
	options []string
}

func (s filterSpec) takes(option string) bool {
	return slices.Contains(s.options, option)
}

// filters are the filters a template may use. tojson is Hugging Face's, not
// Jinja's; the others behave as in jinja2/filters.py.
var filters = map[string]filterSpec{
	"items":  {apply: filterItems},
	"length": {apply: filterLength},
	"safe":   {apply: filterSafe},
	"string": {apply: filterString},
	"tojson": {apply: filterToJSON, options: []string{"ensure_ascii", "indent"}},
	"trim":   {apply: filterTrim},
}

// filterLength is Python's len().
func filterLength(value any, _ []namedValue) (any, error) {
	switch value := value.(type) {
	case undefined:
		return 0, nil
	case string:
		return utf8.RuneCountInString(value), nil
	case safeString:
		return utf8.RuneCountInString(string(value)), nil
	case []any:
		return len(value), nil
	case tuple:
		return len(value), nil
	case *dict:
		return len(value.keys), nil
	}
	return nil, pyError("TypeError", "object of type '%s' has no len()", typeName(value))
}

// filterTrim strips whitespace from both ends of the value's text.
func filterTrim(value any, _ []namedValue) (any, error) {
	text, err := pyStr(value)
	if err != nil {
		return nil, err
	}
	text = strings.TrimFunc(text, isPythonSpace)
	if _, safe := value.(safeString); safe {
		return safeString(text), nil
	}
	return text, nil
}

// filterString converts the value to text; text stays as it is.
func filterString(value any, _ []namedValue) (any, error) {
	if _, safe := value.(safeString); safe {
		return value, nil
	}
	return pyStr(value)
}

// filterSafe marks the value's text as safe.
func filterSafe(value any, _ []namedValue) (any, error) {
	text, err := pyStr(value)
	return safeString(text), err
}

// generator is what the items filter returns: something a for loop can walk
// once, after which it is empty, and nothing else can use. Like Python's
// generator, it reports a wrong input only when it is iterated.
type generator struct {
	items []any
	err   error
}

// filterItems yields the (key, value) pairs of a mapping in key order.
func filterItems(value any, _ []namedValue) (any, error) {
	switch value := value.(type) {
	case undefined:
		return &generator{}, nil
	case *dict:
		pairs := make([]any, len(value.keys))
		for i, key := range value.keys {
			pairs[i] = tuple{key, value.values[key]}
		}
		return &generator{items: pairs}, nil
	}
	return &generator{err: pyError("TypeError", "Can only get item pairs from a mapping.")}, nil
}

// maxJSONIndent bounds tojson's indent. Python has no bound; without one, a
// template that takes the indent from the request would let a request ask
// for gigabytes of spaces.
const maxJSONIndent = 100

// filterToJSON is json.dumps(value, ensure_ascii=False) with the optional
// indent and ensure_ascii arguments.
func filterToJSON(value any, options []namedValue) (any, error) {
	var encoder jsonEncoder
	for _, option := range options {
		switch option.name {
		case "ensure_ascii":
			encoder.ensureASCII = truthy(option.value)
		case "indent":
			if option.value == nil {
				continue
			}
			indent, ok := asIndex(option.value)
			switch {
			case !ok:
				return nil, pyError("TypeError", "tojson: indent must be an integer or none, not %s", typeName(option.value))
			case indent > maxJSONIndent:
				return nil, fmt.Errorf("tojson: indent %d is above the limit of %d", indent, maxJSONIndent)
			}
			encoder.multiline = true
			encoder.indent = indent
		}
	}
	if err := encoder.encode(value, 0); err != nil {
		return nil, err
	}
	return encoder.out.String(), nil
}
