package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
)

// awaitWaiters fails the test unless exactly n callers are waiting in q in
// time.
func awaitWaiters(t *testing.T, q *queue, n int) {
	t.Helper()
	deadline := time.Now().Add(eventTimeout)
	for {
		q.mu.Lock()
		waiting := len(q.waiters)
		q.mu.Unlock()
		if waiting == n {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("%d callers waiting, want %d", waiting, n)
		}
		time.Sleep(time.Millisecond)
	}
}

// awaitResult returns the next value of ch, failing the test if none arrives
// in time.
func awaitResult[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(eventTimeout):
		t.Fatal("nothing arrived in time")
		panic("unreachable")
	}
}

func TestQueueAdmitsOneAtATime(t *testing.T) {
	var q queue
	if err := q.acquire(context.Background()); err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	second := make(chan error, 1)
	go func() { second <- q.acquire(context.Background()) }()
	awaitWaiters(t, &q, 1)
	select {
	case err := <-second:
		t.Fatalf("second acquire returned %v while the first still held the queue", err)
	default:
	}

	q.release()
	if err := awaitResult(t, second); err != nil {
		t.Fatalf("second acquire: %v", err)
	}
	q.release()

	// Free again: the next caller does not wait.
	if err := q.acquire(context.Background()); err != nil {
		t.Fatalf("acquire after release: %v", err)
	}
}

func TestQueueServesWaitersInArrivalOrder(t *testing.T) {
	var q queue
	if err := q.acquire(context.Background()); err != nil {
		t.Fatal(err)
	}
	served := make(chan int, 3)
	for i := range 3 {
		go func() {
			if err := q.acquire(context.Background()); err == nil {
				served <- i
			}
		}()
		awaitWaiters(t, &q, i+1) // i is in line before i+1 starts
	}
	for want := range 3 {
		q.release()
		if got := awaitResult(t, served); got != want {
			t.Fatalf("turn %d went to caller %d", want, got)
		}
	}
}

func TestQueueWaiterThatGivesUpIsSkipped(t *testing.T) {
	var q queue
	if err := q.acquire(context.Background()); err != nil {
		t.Fatal(err)
	}
	ctx, giveUp := context.WithCancel(context.Background())
	impatient := make(chan error, 1)
	go func() { impatient <- q.acquire(ctx) }()
	awaitWaiters(t, &q, 1)
	patient := make(chan error, 1)
	go func() { patient <- q.acquire(context.Background()) }()
	awaitWaiters(t, &q, 2)

	giveUp()
	if err := awaitResult(t, impatient); !errors.Is(err, context.Canceled) {
		t.Fatalf("acquire after giving up = %v, want context.Canceled", err)
	}
	awaitWaiters(t, &q, 1)

	q.release()
	if err := awaitResult(t, patient); err != nil {
		t.Fatalf("patient acquire: %v", err)
	}
	q.release()
	if err := q.acquire(context.Background()); err != nil {
		t.Fatalf("queue is not free after everyone left: %v", err)
	}
}

func TestQueueRefusesACallerWhoAlreadyGaveUp(t *testing.T) {
	var q queue
	ctx, giveUp := context.WithCancel(context.Background())
	giveUp()
	if err := q.acquire(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("acquire = %v, want context.Canceled", err)
	}
	if err := q.acquire(context.Background()); err != nil {
		t.Fatalf("queue was left taken: %v", err)
	}
}

// promptLog wraps an engine and records the prompt of every Generate call, in
// the order the calls start.
type promptLog struct {
	mu      sync.Mutex
	prompts []string
}

func (l *promptLog) wrap(inner engine.Engine) engine.Engine {
	return engineFunc(func(ctx context.Context, req engine.Request, emit func(int32)) (engine.Result, error) {
		text := make([]byte, len(req.Prompt)) // the fake tokenizer makes one token per byte
		for i, token := range req.Prompt {
			text[i] = byte(token)
		}
		l.mu.Lock()
		l.prompts = append(l.prompts, string(text))
		l.mu.Unlock()
		return inner.Generate(ctx, req, emit)
	})
}

func (l *promptLog) seen() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.Clone(l.prompts)
}

// startCompletion sends a text completion for prompt in the background. The
// returned channel gets the response status, or 0 if the request failed.
func startCompletion(ctx context.Context, t *testing.T, ts *httptest.Server, prompt string) <-chan int {
	t.Helper()
	return startCompletionWith(ctx, t, ts, fmt.Sprintf(`{"prompt": %q}`, prompt))
}

// startCompletionWith is startCompletion for a whole request body.
func startCompletionWith(ctx context.Context, t *testing.T, ts *httptest.Server, body string) <-chan int {
	t.Helper()
	req, err := http.NewRequestWithContext(ctx, "POST", ts.URL+"/v1/completions", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	status := make(chan int, 1)
	go func() {
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			status <- 0
			return
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		status <- resp.StatusCode
	}()
	return status
}

// queueOf returns the queue of the harness's server.
func queueOf(h *harness) *queue { return &h.handler.(*server).queue }

func TestSecondRequestWaitsForTheFirstGeneration(t *testing.T) {
	g := newGate(t, 0)
	h, ts := newLiveHarness(t, scripted(countPieces...), func(h *harness) { h.engine.Gate = g.wait })
	ctx := context.Background()

	first := startCompletion(ctx, t, ts, "first")
	g.awaitEntered(t)
	second := startCompletion(ctx, t, ts, "second")
	awaitWaiters(t, queueOf(h), 1)
	if n := h.engine.Calls.Load(); n != 1 {
		t.Fatalf("%d generations started while the first was still running, want 1", n)
	}

	g.open()
	if a, b := awaitResult(t, first), awaitResult(t, second); a != 200 || b != 200 {
		t.Fatalf("statuses = %d, %d, want 200, 200", a, b)
	}
	if n := h.engine.Calls.Load(); n != 2 {
		t.Fatalf("%d generations ran, want 2", n)
	}
}

func TestQueuedRequestsAreServedInArrivalOrder(t *testing.T) {
	g := newGate(t, 0)
	var log promptLog
	h, ts := newLiveHarness(t, scripted(countPieces...), func(h *harness) {
		h.engine.Gate = g.wait
		h.wrap = log.wrap
	})
	ctx := context.Background()

	running := startCompletion(ctx, t, ts, "running")
	g.awaitEntered(t)
	var queued []<-chan int
	for i, prompt := range []string{"a", "b", "c"} {
		queued = append(queued, startCompletion(ctx, t, ts, prompt))
		awaitWaiters(t, queueOf(h), i+1)
	}

	g.open()
	for _, status := range append(queued, running) {
		if code := awaitResult(t, status); code != 200 {
			t.Fatalf("status = %d, want 200", code)
		}
	}
	if got, want := log.seen(), []string{"running", "a", "b", "c"}; !slices.Equal(got, want) {
		t.Fatalf("generations ran in order %q, want %q", got, want)
	}
}

func TestQueuedRequestWhoseClientLeftNeverReachesTheEngine(t *testing.T) {
	g := newGate(t, 0)
	var log promptLog
	h, ts := newLiveHarness(t, scripted(countPieces...), func(h *harness) {
		h.engine.Gate = g.wait
		h.wrap = log.wrap
	})

	running := startCompletion(context.Background(), t, ts, "running")
	g.awaitEntered(t)
	ctx, hangUp := context.WithCancel(context.Background())
	defer hangUp()
	gone := startCompletion(ctx, t, ts, "gone")
	awaitWaiters(t, queueOf(h), 1)

	hangUp()
	awaitResult(t, gone)
	awaitWaiters(t, queueOf(h), 0)

	g.open()
	if code := awaitResult(t, running); code != 200 {
		t.Fatalf("status = %d, want 200", code)
	}
	if code := awaitResult(t, startCompletion(context.Background(), t, ts, "after")); code != 200 {
		t.Fatalf("status = %d, want 200", code)
	}
	if got, want := log.seen(), []string{"running", "after"}; !slices.Equal(got, want) {
		t.Fatalf("generations ran for %q, want %q", got, want)
	}
	if n := h.engine.Calls.Load(); n != 2 {
		t.Fatalf("%d generations ran, want 2", n)
	}
}

// lateQuitter is a context that cannot wake a waiter: Done never fires. Its
// Err starts to report cancellation when the test says so. That pins down the
// moment a real context would be caught at, just after the turn was handed
// over and just before the waiter looks.
type lateQuitter struct {
	context.Context
	quit atomic.Bool
}

func (c *lateQuitter) Done() <-chan struct{} { return nil }

func (c *lateQuitter) Err() error {
	if c.quit.Load() {
		return context.Canceled
	}
	return nil
}

func TestQueueTurnHandedToAWaiterWhoJustGaveUpPassesOn(t *testing.T) {
	var q queue
	if err := q.acquire(context.Background()); err != nil {
		t.Fatal(err)
	}
	quitter := &lateQuitter{Context: context.Background()}
	gaveUp := make(chan error, 1)
	go func() { gaveUp <- q.acquire(quitter) }()
	awaitWaiters(t, &q, 1)
	next := make(chan error, 1)
	go func() { next <- q.acquire(context.Background()) }()
	awaitWaiters(t, &q, 2)

	quitter.quit.Store(true) // gives up, but is still first in line
	q.release()              // so the turn goes to it

	if err := awaitResult(t, gaveUp); !errors.Is(err, context.Canceled) {
		t.Fatalf("acquire of the waiter who gave up = %v, want context.Canceled", err)
	}
	if err := awaitResult(t, next); err != nil {
		t.Fatalf("the turn was not passed on: %v", err)
	}
	q.release()
	if err := q.acquire(context.Background()); err != nil {
		t.Fatalf("queue is not free at the end: %v", err)
	}
}

func TestQueueRefusesMoreThanItsLimitOfWaiters(t *testing.T) {
	var q queue
	if err := q.acquire(context.Background()); err != nil {
		t.Fatal(err)
	}
	turns := make(chan error, maxQueued)
	for i := range maxQueued {
		go func() { turns <- q.acquire(context.Background()) }()
		awaitWaiters(t, &q, i+1)
	}

	if err := q.acquire(context.Background()); !errors.Is(err, errQueueFull) {
		t.Fatalf("acquire with %d waiting = %v, want errQueueFull", maxQueued, err)
	}
	awaitWaiters(t, &q, maxQueued) // the refused caller did not join the line

	for range maxQueued {
		q.release()
		if err := awaitResult(t, turns); err != nil {
			t.Fatalf("a waiter lost its turn: %v", err)
		}
	}
	q.release()
	if err := q.acquire(context.Background()); err != nil {
		t.Fatalf("queue is not free at the end: %v", err)
	}
}

func TestRequestBeyondTheQueueLimitIsTurnedAway(t *testing.T) {
	g := newGate(t, 0)
	h, ts := newLiveHarness(t, scripted(countPieces...), func(h *harness) { h.engine.Gate = g.wait })
	ctx := context.Background()

	running := startCompletion(ctx, t, ts, "running")
	g.awaitEntered(t)
	var queued []<-chan int
	for i := range maxQueued {
		queued = append(queued, startCompletion(ctx, t, ts, fmt.Sprint("queued ", i)))
		awaitWaiters(t, queueOf(h), i+1)
	}

	for _, body := range []string{`{"prompt": "one too many"}`, `{"prompt": "one too many", "stream": true}`} {
		resp := postJSON(t, ts.URL+"/v1/completions", body)
		raw, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != 503 || !jsonEqual(t, string(raw), `{"error": "server busy: too many queued requests"}`) {
			t.Errorf("request beyond the limit = %d %s, want 503 and the busy error", resp.StatusCode, raw)
		}
		if retry := resp.Header.Get("Retry-After"); retry != "1" {
			t.Errorf("Retry-After = %q, want 1", retry)
		}
		if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", ct)
		}
	}
	if n := h.engine.Calls.Load(); n != 1 {
		t.Errorf("%d generations started, want only the running one", n)
	}

	g.open()
	for _, status := range append(queued, running) {
		if code := awaitResult(t, status); code != 200 {
			t.Fatalf("status = %d, want 200", code)
		}
	}
}
