package chattemplate

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode"
)

// This file holds the Python rules template values follow: truthiness, type
// names and conversion to text.

// pyError builds the error for an operation Python or Jinja would raise on,
// named by the exception class they would use.
func pyError(class, format string, args ...any) error {
	return fmt.Errorf(class+": "+format, args...)
}

// truthy is Python's bool(value).
func truthy(value any) bool {
	switch value := value.(type) {
	case nil, undefined:
		return false
	case bool:
		return value
	case int:
		return value != 0
	case float64:
		return value != 0
	case string:
		return value != ""
	case safeString:
		return value != ""
	case []any:
		return len(value) > 0
	case tuple:
		return len(value) > 0
	case *dict:
		return len(value.keys) > 0
	}
	return true // generators, namespaces, loops and callables
}

// typeName is Python's type(value).__name__, for error messages.
func typeName(value any) string {
	switch value.(type) {
	case nil:
		return "NoneType"
	case undefined:
		return "Undefined"
	case bool:
		return "bool"
	case int:
		return "int"
	case float64:
		return "float"
	case string:
		return "str"
	case safeString:
		return "Markup"
	case []any:
		return "list"
	case tuple:
		return "tuple"
	case *dict:
		return "dict"
	case *generator:
		return "generator"
	case *namespace:
		return "Namespace"
	case *loopState:
		return "LoopContext"
	case *macro:
		return "Macro"
	}
	return "function"
}

// pyStr is Python's str(value), which is what {{ value }} writes.
func pyStr(value any) (string, error) {
	switch value := value.(type) {
	case undefined:
		return "", nil
	case string:
		return value, nil
	case safeString:
		return string(value), nil
	}
	var text strings.Builder
	if err := writeRepr(&text, value); err != nil {
		return "", err
	}
	return text.String(), nil
}

// writeRepr writes Python's repr(value).
func writeRepr(out *strings.Builder, value any) error {
	switch value := value.(type) {
	case nil:
		out.WriteString("None")
	case undefined:
		out.WriteString("Undefined")
	case bool:
		if value {
			out.WriteString("True")
		} else {
			out.WriteString("False")
		}
	case int:
		out.WriteString(strconv.Itoa(value))
	case float64:
		out.WriteString(pythonFloat(value))
	case string:
		writeStringRepr(out, value)
	case []any:
		return writeSequenceRepr(out, value, "[", "]")
	case tuple:
		if len(value) == 1 {
			return writeSequenceRepr(out, value, "(", ",)")
		}
		return writeSequenceRepr(out, value, "(", ")")
	case *dict:
		out.WriteByte('{')
		for i, key := range value.keys {
			if i > 0 {
				out.WriteString(", ")
			}
			writeStringRepr(out, key)
			out.WriteString(": ")
			if err := writeRepr(out, value.values[key]); err != nil {
				return err
			}
		}
		out.WriteByte('}')
	default:
		return fmt.Errorf("cannot render a %s as text", typeName(value))
	}
	return nil
}

func writeSequenceRepr(out *strings.Builder, items []any, open, close string) error {
	out.WriteString(open)
	for i, item := range items {
		if i > 0 {
			out.WriteString(", ")
		}
		if err := writeRepr(out, item); err != nil {
			return err
		}
	}
	out.WriteString(close)
	return nil
}

// writeStringRepr writes a string the way Python's repr does: in single
// quotes unless it holds a single quote and no double quote, with
// non-printable characters escaped.
func writeStringRepr(out *strings.Builder, text string) {
	quote := '\''
	if strings.Contains(text, "'") && !strings.Contains(text, `"`) {
		quote = '"'
	}
	out.WriteRune(quote)
	for _, char := range text {
		switch {
		case char == quote || char == '\\':
			out.WriteByte('\\')
			out.WriteRune(char)
		case char == '\n':
			out.WriteString(`\n`)
		case char == '\r':
			out.WriteString(`\r`)
		case char == '\t':
			out.WriteString(`\t`)
		case unicode.IsPrint(char):
			out.WriteRune(char)
		case char <= 0xff:
			fmt.Fprintf(out, `\x%02x`, char)
		case char <= 0xffff:
			fmt.Fprintf(out, `\u%04x`, char)
		default:
			fmt.Fprintf(out, `\U%08x`, char)
		}
	}
	out.WriteRune(quote)
}

// pythonFloat formats a float the way Python's repr does: the shortest digits
// that round-trip, always with a fraction or an exponent, and an exponent of
// at least two digits once the decimal exponent leaves [-4, 16).
func pythonFloat(value float64) string {
	switch {
	case math.IsNaN(value):
		return "nan"
	case math.IsInf(value, 1):
		return "inf"
	case math.IsInf(value, -1):
		return "-inf"
	}
	scientific := strconv.FormatFloat(value, 'e', -1, 64)
	exponent, _ := strconv.Atoi(scientific[strings.IndexByte(scientific, 'e')+1:])
	if exponent < -4 || exponent >= 16 {
		return scientific
	}
	positional := strconv.FormatFloat(value, 'f', -1, 64)
	if !strings.Contains(positional, ".") {
		positional += ".0"
	}
	return positional
}
