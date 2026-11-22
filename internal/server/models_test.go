package server

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestModelsListsTheServedModel(t *testing.T) {
	rec := loadExchange(t, "models")
	h := newHarness(t)
	got := h.replay(rec)

	if got.Code != rec.Response.Status {
		t.Fatalf("status = %d, want %d: %s", got.Code, rec.Response.Status, got.Body)
	}
	if ct := got.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", ct)
	}
	if g, w := shapeOf(t, got.Body.String()), shapeOf(t, rec.Response.Body); !reflect.DeepEqual(g, w) {
		t.Fatalf("shape = %v, want %v", g, w)
	}
	var list struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int64  `json:"created"`
		} `json:"data"`
	}
	if err := json.Unmarshal(got.Body.Bytes(), &list); err != nil {
		t.Fatal(err)
	}
	if list.Object != "list" || len(list.Data) != 1 {
		t.Fatalf("list = %+v, want object \"list\" with one entry", list)
	}
	if m := list.Data[0]; m.ID != h.cfg.Model || m.Object != "model" || m.Created <= 0 {
		t.Fatalf("entry = %+v, want id %q, object \"model\" and a creation time", m, h.cfg.Model)
	}
}
