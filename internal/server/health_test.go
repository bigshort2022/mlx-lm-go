package server

import "testing"

func TestHealthMatchesRecording(t *testing.T) {
	rec := loadExchange(t, "health")
	got := newHarness(t).replay(rec)

	if got.Code != rec.Response.Status {
		t.Fatalf("status = %d, want %d", got.Code, rec.Response.Status)
	}
	if !jsonEqual(t, got.Body.String(), rec.Response.Body) {
		t.Fatalf("body = %s, want %s", got.Body, rec.Response.Body)
	}
	if ct := got.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", ct)
	}
}
