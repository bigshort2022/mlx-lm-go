package chattemplate

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// decodeValue parses one JSON document into template values the way Python's
// json.loads does: objects keep the order of their keys, and a number is an
// int unless it has a fraction or an exponent.
func decodeValue(raw json.RawMessage) (any, error) {
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

// decodeNumber keeps Python's split between int and float. Python integers
// have no upper bound; ours do, and one beyond it is rejected rather than
// rounded.
func decodeNumber(literal json.Number) (any, error) {
	if strings.ContainsAny(string(literal), ".eE") {
		return literal.Float64()
	}
	integer, err := strconv.Atoi(string(literal))
	if err != nil {
		return nil, fmt.Errorf("integer %s is out of range", literal)
	}
	return integer, nil
}
