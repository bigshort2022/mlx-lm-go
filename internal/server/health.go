package server

import "net/http"

// healthResponse is the body of GET /health.
type healthResponse struct {
	Status string `json:"status"`
}

func (s *server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, healthResponse{Status: "ok"})
}
