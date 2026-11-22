package chattemplate

import "strings"

// stringMethod implements one method of Python's str.
type stringMethod func(self string, args []any, kwargs []namedValue) (any, error)

// stringMethods are the str methods a template may call. Each takes the
// arguments chat templates pass; Python's further optional arguments are
// rejected rather than ignored.
var stringMethods = map[string]stringMethod{
	"startswith": func(self string, args []any, kwargs []namedValue) (any, error) {
		prefix, err := oneString("startswith", args, kwargs)
		return strings.HasPrefix(self, prefix), err
	},
	"endswith": func(self string, args []any, kwargs []namedValue) (any, error) {
		suffix, err := oneString("endswith", args, kwargs)
		return strings.HasSuffix(self, suffix), err
	},
	"split": func(self string, args []any, kwargs []namedValue) (any, error) {
		separator, err := oneString("split", args, kwargs)
		if err != nil {
			return nil, err
		}
		if separator == "" {
			return nil, pyError("ValueError", "empty separator")
		}
		parts := strings.Split(self, separator)
		list := make([]any, len(parts))
		for i, part := range parts {
			list[i] = part
		}
		return list, nil
	},
	"strip":  stripMethod("strip", strings.TrimFunc, strings.Trim),
	"lstrip": stripMethod("lstrip", strings.TrimLeftFunc, strings.TrimLeft),
	"rstrip": stripMethod("rstrip", strings.TrimRightFunc, strings.TrimRight),
}

// oneString returns the single string argument of a str method.
func oneString(method string, args []any, kwargs []namedValue) (string, error) {
	if len(args) != 1 || len(kwargs) > 0 {
		return "", pyError("TypeError", "str.%s() takes exactly one argument here, a string", method)
	}
	text, ok := args[0].(string)
	if !ok {
		return "", pyError("TypeError", "str.%s() argument must be str, not %s", method, typeName(args[0]))
	}
	return text, nil
}

// stripMethod builds strip, lstrip or rstrip: with no argument or none it
// removes whitespace, with a string it removes any of that string's
// characters.
func stripMethod(
	name string,
	trimSpace func(string, func(rune) bool) string,
	trimChars func(text, chars string) string,
) stringMethod {
	return func(self string, args []any, kwargs []namedValue) (any, error) {
		if len(args) > 1 || len(kwargs) > 0 {
			return nil, pyError("TypeError", "%s expected at most 1 argument, got %d", name, len(args)+len(kwargs))
		}
		if len(args) == 0 || args[0] == nil {
			return trimSpace(self, isPythonSpace), nil
		}
		chars, ok := args[0].(string)
		if !ok {
			return nil, pyError("TypeError", "%s arg must be None or str", name)
		}
		return trimChars(self, chars), nil
	}
}
