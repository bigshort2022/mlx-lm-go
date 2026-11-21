package server

import "net/http"

// handleCompletions serves POST /v1/completions. The prompt is tokenised as
// it is, without a chat template.
func (s *server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	body, ok := readBody(w, r)
	if !ok {
		return
	}
	fields := &reader{body: body}
	j := s.job(fields)
	if _, ok := fields.raw("prompt"); !ok {
		fields.fail("Request did not contain a prompt")
	}
	prompt := fields.text("prompt", "")
	if fields.err != nil {
		writeError(w, http.StatusBadRequest, fields.err.Error())
		return
	}
	j.prompt = func() ([]int32, error) {
		return s.deps.Tokenizer.Encode(prompt), nil
	}
	s.run(w, r, j)
}
