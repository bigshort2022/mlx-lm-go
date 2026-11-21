package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
)

// normaliseMessages rewrites the messages of a chat request into the form chat
// templates expect, as the reference does: content given as a list of text
// parts becomes one string, and missing or null content becomes "".
//
// Templates serialise parts of a message, such as the arguments of a tool
// call, straight into the prompt. Everything but the rewritten content
// therefore keeps its original bytes: key order, spacing and all.
func normaliseMessages(raw json.RawMessage) (json.RawMessage, error) {
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
		edit, err := contentEdit(message)
		if err != nil {
			return nil, err
		}
		if edit.text == nil {
			continue
		}
		out.Write(raw[copied : start+edit.start])
		out.Write(edit.text)
		copied = start + edit.end
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

// contentEdit returns the change that normalises the content of one message.
func contentEdit(message json.RawMessage) (edit, error) {
	if message[0] != '{' {
		return edit{}, errors.New("messages must be a list of dicts")
	}
	dec := json.NewDecoder(bytes.NewReader(message))
	if _, err := dec.Token(); err != nil { // the opening brace
		return edit{}, err
	}
	var content json.RawMessage
	contentEnd, keys := 0, 0
	for ; dec.More(); keys++ {
		key, err := dec.Token()
		if err != nil {
			return edit{}, err
		}
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return edit{}, err
		}
		if key == "content" { // a repeated key counts once, the last time, as in Python
			content, contentEnd = value, int(dec.InputOffset())
		}
	}
	if content == nil {
		// Like the reference, add the missing content as the last key.
		text, closingBrace := `"content":""`, len(message)-1
		if keys > 0 {
			text = "," + text
		}
		return edit{start: closingBrace, end: closingBrace, text: []byte(text)}, nil
	}
	flat, err := flattenContent(content)
	return edit{start: contentEnd - len(content), end: contentEnd, text: flat}, err
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
