package server

import (
	"io"
	"net/http"
)

// eventStream writes a completion as server-sent events, pushing each one to
// the client as soon as it is written.
type eventStream struct {
	w   http.ResponseWriter
	out *http.ResponseController
}

// startEventStream sends the response headers of a stream.
func startEventStream(w http.ResponseWriter) *eventStream {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	e := &eventStream{w: w, out: http.NewResponseController(w)}
	_ = e.out.Flush()
	return e
}

// send writes one data event holding v as JSON. Write errors are ignored: they
// mean the client has gone, which also cancels the request.
func (e *eventStream) send(v any) {
	_, _ = io.WriteString(e.w, "data: ")
	_ = encodeJSON(e.w, v) // ends the data line
	_, _ = io.WriteString(e.w, "\n")
	_ = e.out.Flush()
}

// done writes the marker that ends an OpenAI-style stream.
func (e *eventStream) done() {
	_, _ = io.WriteString(e.w, "data: [DONE]\n\n")
	_ = e.out.Flush()
}
