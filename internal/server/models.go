package server

import (
	"net/http"
	"time"
)

// modelList is the body of GET /v1/models.
type modelList struct {
	Object string      `json:"object"`
	Data   []modelInfo `json:"data"`
}

type modelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
}

// handleModels lists the one model this server serves. Python lists every
// model in the Hugging Face cache because it can load them on demand.
func (s *server) handleModels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, modelList{
		Object: "list",
		Data:   []modelInfo{{ID: s.cfg.Model, Object: "model", Created: time.Now().Unix()}},
	})
}
