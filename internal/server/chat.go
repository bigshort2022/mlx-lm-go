package server

import (
	"encoding/json"
	"net/http"
)

// handleChat serves POST /v1/chat/completions.
func (s *server) handleChat(w http.ResponseWriter, r *http.Request) {
	body, ok := readBody(w, r)
	if !ok {
		return
	}
	fields := &reader{body: body}
	j := s.job(fields)
	j.chat = true
	xml, supported := s.deps.Template.(interface{ UsesQwenXMLTools() bool })
	xmlTools := supported && xml.UsesQwenXMLTools()
	tools, set := fields.chatTools(xmlTools)
	j.tools = set
	messages := chatMessages(fields, xmlTools)
	kwargs := s.templateArgs(fields)
	if fields.err != nil {
		writeError(w, http.StatusBadRequest, fields.err.Error())
		return
	}
	j.prompt = func() ([]int32, error) {
		text, err := s.deps.Template.Render(messages, tools, kwargs, true)
		if err != nil {
			return nil, err
		}
		return s.deps.Tokenizer.Encode(text), nil
	}
	s.run(w, r, j)
}

// templateArgs returns the extra template variables: the request's
// chat_template_kwargs merged over --chat-template-args, request keys winning.
func (s *server) templateArgs(fields *reader) json.RawMessage {
	merged, err := mergeObjects(s.cfg.ChatTemplateArgs, fields.dict("chat_template_kwargs"))
	if err != nil {
		fields.fail("chat template arguments: %v", err)
	}
	return merged
}

// mergeObjects returns one JSON object holding the keys of every given object,
// later objects winning. Empty inputs are skipped.
func mergeObjects(objects ...json.RawMessage) (json.RawMessage, error) {
	merged := map[string]json.RawMessage{}
	for _, raw := range objects {
		if len(raw) == 0 {
			continue
		}
		if err := json.Unmarshal(raw, &merged); err != nil {
			return nil, err
		}
	}
	return json.Marshal(merged)
}

// chatMessages returns the request's messages, normalised for the template.
func chatMessages(fields *reader, xmlTools bool) json.RawMessage {
	raw, ok := fields.raw("messages")
	if !ok {
		fields.fail("Request did not contain messages")
		return nil
	}
	messages, err := normaliseMessages(raw, xmlTools)
	if err != nil {
		fields.fail("%v", err)
	}
	return messages
}
