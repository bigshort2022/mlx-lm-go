package chattemplate

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf16"
)

// jsonEncoder writes values the way Python's json.dumps does, which differs
// from encoding/json in its separators, its float formatting and in leaving
// '<', '>' and '&' alone.
type jsonEncoder struct {
	out         strings.Builder
	multiline   bool // json.dumps(indent=...) was given
	indent      int  // spaces per nesting level when multiline
	ensureASCII bool // escape everything outside printable ASCII
}

func (e *jsonEncoder) encode(value any, depth int) error {
	switch value := value.(type) {
	case nil:
		e.out.WriteString("null")
	case bool:
		e.out.WriteString(strconv.FormatBool(value))
	case int:
		e.out.WriteString(strconv.Itoa(value))
	case float64:
		e.out.WriteString(jsonFloat(value))
	case string:
		e.encodeString(value)
	case safeString:
		e.encodeString(string(value))
	case []any:
		return e.encodeArray(value, depth)
	case tuple:
		return e.encodeArray(value, depth)
	case *dict:
		return e.encodeObject(value, depth)
	default:
		return pyError("TypeError", "Object of type %s is not JSON serializable", typeName(value))
	}
	return nil
}

func (e *jsonEncoder) encodeArray(items []any, depth int) error {
	if len(items) == 0 {
		e.out.WriteString("[]")
		return nil
	}
	e.out.WriteByte('[')
	for i, item := range items {
		e.separate(i, depth+1)
		if err := e.encode(item, depth+1); err != nil {
			return err
		}
	}
	e.closeContainer(']', depth)
	return nil
}

func (e *jsonEncoder) encodeObject(object *dict, depth int) error {
	if len(object.keys) == 0 {
		e.out.WriteString("{}")
		return nil
	}
	e.out.WriteByte('{')
	for i, key := range object.keys {
		e.separate(i, depth+1)
		e.encodeString(key)
		e.out.WriteString(": ")
		if err := e.encode(object.values[key], depth+1); err != nil {
			return err
		}
	}
	e.closeContainer('}', depth)
	return nil
}

// separate writes what goes before the i-th item of a container whose items
// sit at the given nesting depth.
func (e *jsonEncoder) separate(i, depth int) {
	switch {
	case e.multiline:
		if i > 0 {
			e.out.WriteByte(',')
		}
		e.newline(depth)
	case i > 0:
		e.out.WriteString(", ")
	}
}

func (e *jsonEncoder) closeContainer(bracket byte, depth int) {
	if e.multiline {
		e.newline(depth)
	}
	e.out.WriteByte(bracket)
}

func (e *jsonEncoder) newline(depth int) {
	e.out.WriteByte('\n')
	for range e.indent * depth {
		e.out.WriteByte(' ')
	}
}

func (e *jsonEncoder) encodeString(text string) {
	e.out.WriteByte('"')
	for _, char := range text {
		switch {
		case char == '"':
			e.out.WriteString(`\"`)
		case char == '\\':
			e.out.WriteString(`\\`)
		case char == '\n':
			e.out.WriteString(`\n`)
		case char == '\t':
			e.out.WriteString(`\t`)
		case char == '\r':
			e.out.WriteString(`\r`)
		case char == '\b':
			e.out.WriteString(`\b`)
		case char == '\f':
			e.out.WriteString(`\f`)
		case char < 0x20:
			fmt.Fprintf(&e.out, `\u%04x`, char)
		case e.ensureASCII && char > '~':
			e.encodeEscaped(char)
		default:
			e.out.WriteRune(char)
		}
	}
	e.out.WriteByte('"')
}

// encodeEscaped writes a character as \uXXXX, or as a surrogate pair when it
// lies beyond the basic multilingual plane.
func (e *jsonEncoder) encodeEscaped(char rune) {
	if high, low := utf16.EncodeRune(char); utf16.IsSurrogate(high) {
		fmt.Fprintf(&e.out, `\u%04x\u%04x`, high, low)
		return
	}
	fmt.Fprintf(&e.out, `\u%04x`, char)
}

// jsonFloat is json.dumps of a float: Python's repr, except for the values
// JSON has no literal for.
func jsonFloat(value float64) string {
	switch {
	case math.IsNaN(value):
		return "NaN"
	case math.IsInf(value, 1):
		return "Infinity"
	case math.IsInf(value, -1):
		return "-Infinity"
	}
	return pythonFloat(value)
}
