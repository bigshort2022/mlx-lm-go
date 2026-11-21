package server

import (
	"bufio"
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
)

// eventTimeout bounds how long a test waits for something the server should
// do promptly.
const eventTimeout = 3 * time.Second

// newLiveHarness is newHarness with the handler listening on a real loopback
// port, for tests that need streaming or connection behaviour. The config's
// port is the one the server listens on.
func newLiveHarness(t *testing.T, setup ...func(*harness)) (*harness, *httptest.Server) {
	t.Helper()
	ts := httptest.NewUnstartedServer(nil)
	port := ts.Listener.Addr().(*net.TCPAddr).Port
	h := newHarness(t, append(setup, func(h *harness) { h.cfg.Port = port })...)
	ts.Config.Handler = h.handler
	ts.Start()
	t.Cleanup(func() {
		ts.CloseClientConnections() // unblocks handlers that still wait on a gate
		ts.Close()
	})
	return h, ts
}

// postJSON starts a POST and returns the response once its headers arrive.
func postJSON(t *testing.T, url, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

// nextEvent reads one server-sent data event and returns its payload. It
// fails the test if none arrives in time.
func nextEvent(t *testing.T, r *bufio.Reader) string {
	t.Helper()
	type result struct {
		data string
		err  error
	}
	done := make(chan result, 1)
	go func() {
		var data string
		for {
			line, err := r.ReadString('\n')
			if err != nil {
				done <- result{err: err}
				return
			}
			if line == "\n" {
				done <- result{data: data}
				return
			}
			data += strings.TrimSuffix(strings.TrimPrefix(line, "data: "), "\n")
		}
	}()
	select {
	case res := <-done:
		if res.err != nil {
			t.Fatalf("reading event: %v", res.err)
		}
		return res.data
	case <-time.After(eventTimeout):
		t.Fatal("no event arrived in time")
		return ""
	}
}

// gate is an engine gate that blocks generation before one token until a test
// opens it.
type gate struct {
	at      int
	entered chan struct{} // closed when generation reaches the gate
	enter   func()
	opened  chan struct{}
	open    func()
}

// newGate returns a gate that blocks before token index at. It is opened when
// the test ends at the latest.
func newGate(t *testing.T, at int) *gate {
	t.Helper()
	g := &gate{at: at, entered: make(chan struct{}), opened: make(chan struct{})}
	g.enter = sync.OnceFunc(func() { close(g.entered) })
	g.open = sync.OnceFunc(func() { close(g.opened) })
	t.Cleanup(g.open)
	return g
}

// wait is the function to install as engine.Fake.Gate. Every generation that
// reaches the gate blocks until it is opened.
func (g *gate) wait(i int, _ int32) {
	if i != g.at {
		return
	}
	g.enter()
	<-g.opened
}

// awaitEntered fails the test unless generation reaches the gate in time.
func (g *gate) awaitEntered(t *testing.T) {
	t.Helper()
	select {
	case <-g.entered:
	case <-time.After(eventTimeout):
		t.Fatal("generation did not reach the gate in time")
	}
}

// reportErrors wraps an engine so that the error of every Generate call is
// sent to the returned channel.
func reportErrors(errs chan<- error) func(engine.Engine) engine.Engine {
	return func(inner engine.Engine) engine.Engine {
		return engineFunc(func(ctx context.Context, req engine.Request, emit func(int32)) (engine.Result, error) {
			res, err := inner.Generate(ctx, req, emit)
			errs <- err
			return res, err
		})
	}
}

// awaitCanceled fails the test unless a Generate call ends with
// context.Canceled in time.
func awaitCanceled(t *testing.T, errs <-chan error) {
	t.Helper()
	select {
	case err := <-errs:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Generate returned %v, want context.Canceled", err)
		}
	case <-time.After(eventTimeout):
		t.Fatal("Generate did not return in time")
	}
}
