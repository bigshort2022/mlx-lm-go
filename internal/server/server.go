package server

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
)

// server is the http.Handler returned by New.
type server struct {
	cfg    Config
	deps   Deps
	hosts  map[string]bool             // accepted Host headers; nil accepts any
	routes map[string]http.HandlerFunc // keyed by "METHOD /path"
	queue  queue                       // admits one completion at a time
}

// New returns the HTTP handler.
func New(cfg Config, deps Deps) http.Handler {
	s := &server{cfg: cfg, deps: deps, hosts: allowedHosts(cfg)}
	s.routes = map[string]http.HandlerFunc{
		"GET /health":               s.handleHealth,
		"GET /v1/models":            s.handleModels,
		"POST /v1/chat/completions": s.handleChat,
		"POST /chat/completions":    s.handleChat, // also served by the reference
		"POST /v1/completions":      s.handleCompletions,
	}
	return s
}

// ServeHTTP applies the browser-safety rules and dispatches the request to
// its route. Anything else is answered like the reference: 404 with the plain
// text "Not Found".
func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !s.admit(w, r) {
		return
	}
	handle, ok := s.routes[r.Method+" "+r.URL.Path]
	if !ok {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, "Not Found")
		return
	}
	handle(w, r)
}

// writeJSON sends v as a JSON response. A failed write means the client has
// gone, so there is nobody left to report it to.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = encodeJSON(w, v)
}

// encodeJSON writes v as one line of JSON, leaving <, > and & unescaped.
func encodeJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	return enc.Encode(v)
}

// errorResponse is the body of every error the server reports as JSON.
type errorResponse struct {
	Error string `json:"error"`
}

// writeError sends a JSON error in the reference's shape.
func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, errorResponse{Error: message})
}

// maxBodyBytes is the largest request body the server reads.
const maxBodyBytes = 16 << 20

// readBody reads the JSON object of a completion request. When it returns
// false it has already answered the client.
func readBody(w http.ResponseWriter, r *http.Request) (object, bool) {
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		writeError(w, http.StatusRequestEntityTooLarge, "request body is larger than 16 MiB")
		return nil, false
	}
	if err != nil {
		writeError(w, http.StatusBadRequest, "could not read the request body: "+err.Error())
		return nil, false
	}
	body, err := decodeBody(raw)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return nil, false
	}
	return body, true
}
