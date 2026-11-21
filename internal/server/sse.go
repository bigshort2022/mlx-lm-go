package server

import (
	"bytes"
	"errors"
	"net/http"
	"time"
)

// streamWriteTimeout is how long a client may take to accept one stream
// event. A client that stops reading would otherwise keep the generation it
// asked for, and with it the only turn, for as long as it liked.
const streamWriteTimeout = 30 * time.Second

// eventStream writes a completion as server-sent events, pushing each one to
// the client as soon as it is written.
type eventStream struct {
	w       http.ResponseWriter
	out     *http.ResponseController
	timeout time.Duration // for the client to accept one event
	err     error         // the first failed write; nothing is written after it
}

// startEventStream sends the response headers of a stream.
func startEventStream(w http.ResponseWriter, timeout time.Duration) *eventStream {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	e := &eventStream{w: w, out: http.NewResponseController(w), timeout: timeout}
	_ = e.write(nil)
	return e
}

// send writes one data event holding v as JSON. It fails when the client has
// gone or does not accept the event in time; every later call then fails
// with the same error without writing.
func (e *eventStream) send(v any) error {
	var event bytes.Buffer
	event.WriteString("data: ")
	if err := encodeJSON(&event, v); err != nil { // ends the data line
		return err
	}
	event.WriteByte('\n')
	return e.write(event.Bytes())
}

// done writes the marker that ends an OpenAI-style stream.
func (e *eventStream) done() error {
	return e.write([]byte("data: [DONE]\n\n"))
}

// write sends p and everything buffered before it to the client, within the
// timeout.
func (e *eventStream) write(p []byte) error {
	if e.err != nil {
		return e.err
	}
	// A writer without deadlines, such as a test recorder, does not block.
	err := e.out.SetWriteDeadline(time.Now().Add(e.timeout))
	if err == nil || errors.Is(err, http.ErrNotSupported) {
		_, err = e.w.Write(p)
	}
	if err == nil {
		err = e.out.Flush()
	}
	e.err = err
	return err
}
