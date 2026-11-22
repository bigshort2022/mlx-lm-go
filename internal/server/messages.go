package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// normaliseMessages rewrites the messages of a chat request into the form chat
// templates expect, as the reference does: content given as a list of text
// parts becomes one string, and missing or null content becomes "".
//
// Qwen XML templates also need prior assistant arguments as objects, not
// OpenAI's JSON strings. Unchanged values retain their original bytes.
func normaliseMessages(raw json.RawMessage, xmlTools bool) (json.RawMessage, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	if open, _ := dec.Token(); open != json.Delim('[') {
		return nil, errors.New("messages must be of type list")
	}
	var out bytes.Buffer
	copied := 0 // raw[:copied] has been written to out, edits applied
	for dec.More() {
		var message json.RawMessage
		if err := dec.Decode(&message); err != nil {
			return nil, err
		}
		start := int(dec.InputOffset()) - len(message)
		var changes [2]edit
		edits, err := messageEdits(message, xmlTools, changes[:0])
		if err != nil {
			return nil, err
		}
		for _, change := range edits {
			out.Write(raw[copied : start+change.start])
			out.Write(change.text)
			copied = start + change.end
		}
	}
	if copied == 0 {
		return raw, nil
	}
	out.Write(raw[copied:])
	return out.Bytes(), nil
}

// edit replaces the bytes [start, end) of a JSON document with text. A nil
// text means there is nothing to change.
type edit struct {
	start, end int
	text       []byte
}

// messageEdits returns the normalizations of one message, sorted by offset.
func messageEdits(message json.RawMessage, xmlTools bool, edits []edit) ([]edit, error) {
	if message[0] != '{' {
		return nil, errors.New("messages must be a list of dicts")
	}
	dec := json.NewDecoder(bytes.NewReader(message))
	if _, err := dec.Token(); err != nil { // the opening brace
		return nil, err
	}
	var content, calls json.RawMessage
	contentEnd, callsEnd, keys := 0, 0, 0
	for ; dec.More(); keys++ {
		key, err := dec.Token()
		if err != nil {
			return nil, err
		}
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return nil, err
		}
		switch key {
		case "content": // a repeated key counts once, the last time, as in Python
			content, contentEnd = value, int(dec.InputOffset())
		case "tool_calls":
			if xmlTools {
				calls, callsEnd = value, int(dec.InputOffset())
			}
		}
	}
	if len(calls) > 0 && string(calls) != "null" {
		flat, err := normaliseCallArguments(calls)
		if err != nil {
			return nil, err
		}
		if flat != nil {
			edits = append(edits, edit{start: callsEnd - len(calls), end: callsEnd, text: flat})
		}
	}
	if content == nil {
		text, closingBrace := `"content":""`, len(message)-1
		if keys > 0 {
			text = "," + text
		}
		return append(edits, edit{start: closingBrace, end: closingBrace, text: []byte(text)}), nil
	}
	flat, err := flattenContent(content)
	if err != nil {
		return nil, err
	}
	if flat != nil {
		edits = append(edits, edit{start: contentEnd - len(content), end: contentEnd, text: flat})
		if len(edits) == 2 && edits[0].start > edits[1].start {
			edits[0], edits[1] = edits[1], edits[0]
		}
	}
	return edits, nil
}

func normaliseCallArguments(raw json.RawMessage) (json.RawMessage, error) {
	var calls []object
	if err := json.Unmarshal(raw, &calls); err != nil {
		return nil, fmt.Errorf("tool_calls must be an array of objects")
	}
	changed := false
	for _, call := range calls {
		function := call
		if wrapped, ok := call["function"]; ok {
			if err := json.Unmarshal(wrapped, &function); err != nil {
				return nil, fmt.Errorf("tool call function must be an object")
			}
		}
		args := function["arguments"]
		if len(args) == 0 || args[0] != '"' {
			continue
		}
		var text string
		_ = json.Unmarshal(args, &text)
		text = strings.TrimSpace(text)
		if text == "" || text[0] != '{' || !json.Valid([]byte(text)) {
			return nil, fmt.Errorf("tool call arguments must encode a JSON object")
		}
		function["arguments"] = json.RawMessage(text)
		if _, ok := call["function"]; ok {
			encoded, err := json.Marshal(function)
			if err != nil {
				return nil, err
			}
			call["function"] = encoded
		}
		changed = true
	}
	if !changed {
		return nil, nil
	}
	return json.Marshal(calls)
}

// errContentType is the reference's message for content that is not text.
var errContentType = errors.New("Only 'text' content type is supported.")

// contentPart is one element of a message whose content is a list.
type contentPart struct {
	Type string  `json:"type"`
	Text *string `json:"text"`
}

// flattenContent returns the JSON string that replaces a message's content,
// or nil when the content can stay as it is.
func flattenContent(content json.RawMessage) (json.RawMessage, error) {
	if string(content) == "null" {
		return json.RawMessage(`""`), nil
	}
	if content[0] != '[' {
		return nil, nil
	}
	var parts []contentPart
	if err := json.Unmarshal(content, &parts); err != nil {
		return nil, errContentType
	}
	var text strings.Builder
	for _, part := range parts {
		if part.Type != "text" || part.Text == nil {
			return nil, errContentType
		}
		text.WriteString(*part.Text)
	}
	return json.Marshal(text.String())
}
