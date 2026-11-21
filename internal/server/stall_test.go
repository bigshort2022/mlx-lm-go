package server

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"strings"
	"testing"
	"time"
)

// flood makes the engine produce far more text than the socket buffers
// between server and client can hold: 4096 tokens of 64 KiB each.
func flood(h *harness) {
	h.tok.pieces = []string{strings.Repeat("x", 64<<10)}
	h.engine.Tokens = make([]int32, 4096)
}

// impatient shortens the time the server waits for a client to take an event.
func impatient(h *harness) {
	h.tune = func(s *server) { s.writeTimeout = 200 * time.Millisecond }
}

// sendWithoutReading sends a completion request over a plain TCP connection
// and never reads the answer. The connection stays open until the test ends.
func sendWithoutReading(t *testing.T, ts *httptest.Server, body string) {
	t.Helper()
	addr := ts.Listener.Addr().String()
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	_, err = fmt.Fprintf(conn, "POST /v1/completions HTTP/1.1\r\nHost: %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s", addr, len(body), body)
	if err != nil {
		t.Fatal(err)
	}
}

func TestStreamClientThatStopsReadingLosesItsTurn(t *testing.T) {
	errs := make(chan error, 2)
	h, ts := newLiveHarness(t, flood, impatient, func(h *harness) { h.wrap = reportErrors(errs) })

	sendWithoutReading(t, ts, `{"prompt": "go", "max_tokens": 4096, "stream": true}`)

	// The buffers fill, a write misses its deadline, and generation is
	// cancelled long before the 4096 tokens are out.
	awaitCanceled(t, errs)
	awaitWaiters(t, queueOf(h), 0)
	if n := h.engine.Calls.Load(); n != 1 {
		t.Fatalf("%d generations started, want 1", n)
	}

	// The turn is free again for a client that does read.
	if code := awaitResult(t, startCompletionWith(context.Background(), t, ts, `{"prompt": "next", "max_tokens": 2}`)); code != 200 {
		t.Fatalf("next request: status = %d, want 200", code)
	}
	if err := awaitResult(t, errs); err != nil {
		t.Fatalf("next generation ended with %v", err)
	}
}

func TestBufferedClientThatStopsReadingDoesNotHoldTheTurn(t *testing.T) {
	errs := make(chan error, 2)
	_, ts := newLiveHarness(t, flood, func(h *harness) { h.wrap = reportErrors(errs) })

	// 256 tokens make a 16 MiB body, which the client never takes.
	sendWithoutReading(t, ts, `{"prompt": "go", "max_tokens": 256}`)
	if err := awaitResult(t, errs); err != nil {
		t.Fatalf("first generation ended with %v", err)
	}

	// The first response is still being written, yet the next request runs.
	if code := awaitResult(t, startCompletionWith(context.Background(), t, ts, `{"prompt": "next", "max_tokens": 2}`)); code != 200 {
		t.Fatalf("next request: status = %d, want 200", code)
	}
}

// The deadline belongs to the stream. A later response on the same
// connection, however long its generation takes, must not run into it.
func TestStreamDeadlineEndsWithTheStream(t *testing.T) {
	const timeout = 100 * time.Millisecond
	_, ts := newLiveHarness(t, scripted(countPieces...), func(h *harness) {
		h.tune = func(s *server) { s.writeTimeout = timeout }
		h.engine.Gate = func(i int, _ int32) {
			if i == 0 && h.engine.Calls.Load() == 2 {
				time.Sleep(3 * timeout) // the second generation is slow to start
			}
		}
	})
	client := &http.Client{Transport: &http.Transport{MaxConnsPerHost: 1}}
	defer client.CloseIdleConnections()

	stream, err := client.Post(ts.URL+"/v1/completions", "application/json", strings.NewReader(`{"prompt": "a", "stream": true}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.Copy(io.Discard, stream.Body); err != nil {
		t.Fatal(err)
	}
	stream.Body.Close()

	reused := false
	trace := &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) { reused = info.Reused }}
	req, err := http.NewRequestWithContext(httptrace.WithClientTrace(context.Background(), trace),
		"POST", ts.URL+"/v1/completions", strings.NewReader(`{"prompt": "b"}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("second request on the connection: %v", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("second response = %d, read error %v: %s", resp.StatusCode, err, raw)
	}
	if text := decodeCompletion(t, string(raw)).Choices[0].Text; text == nil || *text != "1, 2, 3, 4, 5" {
		t.Errorf("second response text = %v, want the completion", text)
	}
	if !reused {
		t.Fatal("the second request did not reuse the stream's connection, so the test proved nothing")
	}
}
