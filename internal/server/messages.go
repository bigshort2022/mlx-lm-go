package server

import (
	"encoding/json"
	"errors"
	"strings"
)

// normaliseMessages rewrites the messages of a chat request into the form chat
// templates expect, as the reference does: content given as a list of text
// parts becomes one string, and missing or null content becomes "". Messages
// that need no change keep their original bytes.
func normaliseMessages(raw json.RawMessage) (json.RawMessage, error) {
	var messages []json.RawMessage
	if err := json.Unmarshal(raw, &messages); err != nil {
		return nil, errors.New("messages must be of type list")
	}
	for i, rawMessage := range messages {
		var message map[string]json.RawMessage
		if err := json.Unmarshal(rawMessage, &message); err != nil || message == nil {
			return nil, errors.New("messages must be a list of dicts")
		}
		content, err := flattenContent(message["content"])
		if err != nil {
			return nil, err
		}
		if content == nil {
			continue
		}
		message["content"] = content
		if messages[i], err = json.Marshal(message); err != nil {
			return nil, err
		}
	}
	return json.Marshal(messages)
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
	if len(content) == 0 || string(content) == "null" {
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
