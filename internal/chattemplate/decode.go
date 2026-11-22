package chattemplate

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// decodeValue parses one JSON document into template values the way Python's
// json.loads does: objects keep the order of their keys, and a number is an
// int unless it has a fraction or an exponent.
func decodeValue(raw json.RawMessage) (any, error) {
	if err := checkText(raw); err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	value, err := readValue(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("unexpected data after the JSON value")
	}
	return value, nil
}

// decodeOptional is decodeValue for a request field that may be left out, in
// which case the template sees none.
func decodeOptional(raw json.RawMessage) (any, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	return decodeValue(raw)
}

func readValue(dec *json.Decoder) (any, error) {
	token, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch token := token.(type) {
	case json.Delim:
		if token == '[' {
			return readArray(dec)
		}
		return readObject(dec)
	case json.Number:
		return decodeNumber(token)
	default:
		return token, nil // string, bool or nil
	}
}

func readArray(dec *json.Decoder) ([]any, error) {
	items := []any{}
	for dec.More() {
		item, err := readValue(dec)
		if err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	_, err := dec.Token() // the closing bracket
	return items, err
}

func readObject(dec *json.Decoder) (*dict, error) {
	object := newDict(0)
	for dec.More() {
		key, err := dec.Token()
		if err != nil {
			return nil, err
		}
		value, err := readValue(dec)
		if err != nil {
			return nil, err
		}
		object.set(key.(string), value)
	}
	_, err := dec.Token() // the closing brace
	return object, err
}

// decodeNumber keeps Python's split between int and float. An integer that
// does not fit 64 bits is kept exact as a *big.Int; a float beyond the double
// range is an infinity, as in Python.
func decodeNumber(literal json.Number) (any, error) {
	if strings.ContainsAny(string(literal), ".eE") {
		value, err := strconv.ParseFloat(string(literal), 64)
		if err != nil && !errors.Is(err, strconv.ErrRange) {
			return nil, err
		}
		return value, nil
	}
	if integer, err := strconv.Atoi(string(literal)); err == nil {
		return integer, nil
	}
	integer, ok := new(big.Int).SetString(string(literal), 10)
	if !ok {
		return nil, fmt.Errorf("invalid number %s", literal)
	}
	return integer, nil
}

// checkText rejects JSON text a Go string cannot hold faithfully: bytes that
// are not UTF-8, which Python's json.loads rejects too, and \u escapes for
// one half of a surrogate pair, which Python keeps and encoding/json would
// quietly turn into U+FFFD.
func checkText(raw []byte) error {
	if !utf8.Valid(raw) {
		return errors.New("invalid UTF-8")
	}
	// In valid JSON a backslash only occurs in a string, where it starts an
	// escape. Text that is not valid JSON is the decoder's to report.
	for {
		backslash := bytes.IndexByte(raw, '\\')
		if backslash < 0 || backslash+1 == len(raw) {
			return nil
		}
		raw = raw[backslash+1:]
		if raw[0] != 'u' {
			raw = raw[1:] // a two-character escape such as \\ or \n
			continue
		}
		code, ok := hexEscape(raw)
		if !ok {
			return nil
		}
		if utf16.IsSurrogate(code) {
			low, paired := rune(0), false
			if len(raw) > 5 && raw[5] == '\\' {
				low, paired = hexEscape(raw[6:])
			}
			// A pair is a high surrogate followed by a low one.
			if !paired || code >= 0xDC00 || low < 0xDC00 || low > 0xDFFF {
				return fmt.Errorf("unpaired surrogate escape \\u%s", raw[1:5])
			}
			raw = raw[6:]
		}
		raw = raw[5:]
	}
}

// hexEscape reads the code unit of an escape whose text, after the
// backslash, starts at text: 'u' and four hexadecimal digits.
func hexEscape(text []byte) (rune, bool) {
	if len(text) < 5 || text[0] != 'u' {
		return 0, false
	}
	code, err := strconv.ParseUint(string(text[1:5]), 16, 16)
	return rune(code), err == nil
}
