package exec

import (
	"context"
	"errors"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// patience bounds every wait in these tests, so a broken Thread fails a test
// instead of hanging it.
const patience = 5 * time.Second

// do calls thread.Do and fails the test if it does not return.
func do(t *testing.T, thread *Thread, ctx context.Context, fn func() error) error {
	t.Helper()
	result := make(chan error, 1)
	go func() { result <- thread.Do(ctx, fn) }()
	select {
	case err := <-result:
		return err
	case <-time.After(patience):
		t.Fatal("Do did not return")
		return nil
	}
}

// occupy makes the thread run a function that blocks until release is called.
// It returns once that function is running.
func occupy(t *testing.T, thread *Thread) (release func()) {
	t.Helper()
	started := make(chan struct{})
	gate := make(chan struct{})
	go thread.Do(context.Background(), func() error {
		close(started)
		<-gate
		return nil
	})
	select {
	case <-started:
	case <-time.After(patience):
		t.Fatal("the thread did not start the blocking function")
	}
	return sync.OnceFunc(func() { close(gate) })
}

// waitQueued waits until n calls are waiting behind the running function.
func waitQueued(t *testing.T, thread *Thread, n int) {
	t.Helper()
	deadline := time.Now().Add(patience)
	for {
		thread.mu.Lock()
		queued := len(thread.queue)
		thread.mu.Unlock()
		if queued == n {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("%d calls are queued, want %d", queued, n)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestDoRunsOneAtATime(t *testing.T) {
	thread := Start()
	defer thread.Close()

	var running, overlaps atomic.Int32
	errs := make([]error, 100)
	var wg sync.WaitGroup
	for i := range errs {
		wg.Go(func() {
			errs[i] = thread.Do(context.Background(), func() error {
				if running.Add(1) > 1 {
					overlaps.Add(1)
				}
				running.Add(-1)
				return nil
			})
		})
	}
	wg.Wait()

	if n := overlaps.Load(); n != 0 {
		t.Errorf("%d functions started while another was running, want none", n)
	}
	for i, err := range errs {
		if err != nil {
			t.Errorf("call %d returned %v, want nil", i, err)
		}
	}
}

func TestDoRunsInArrivalOrder(t *testing.T) {
	thread := Start()
	defer thread.Close()
	release := occupy(t, thread)
	defer release()

	var order []int // written only on the thread, read after wg.Wait
	var wg sync.WaitGroup
	for i := range 10 {
		wg.Go(func() {
			thread.Do(context.Background(), func() error {
				order = append(order, i)
				return nil
			})
		})
		waitQueued(t, thread, i+1)
	}
	release()
	wg.Wait()

	if want := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; !slices.Equal(order, want) {
		t.Errorf("functions ran in order %v, want arrival order %v", order, want)
	}
}

func TestDoSkipsFunctionWhenContextEndsWhileQueued(t *testing.T) {
	thread := Start()
	defer thread.Close()
	release := occupy(t, thread)
	defer release()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var ran atomic.Bool
	result := make(chan error, 1)
	go func() {
		result <- thread.Do(ctx, func() error {
			ran.Store(true)
			return nil
		})
	}()
	waitQueued(t, thread, 1)
	cancel()

	select {
	case err := <-result:
		if err != context.Canceled {
			t.Errorf("Do returned %v, want context.Canceled", err)
		}
	case <-time.After(patience):
		t.Fatal("Do did not return after its context was cancelled, want it to stop waiting")
	}
	release()
	// A later call finishing shows the thread has passed the cancelled one.
	if err := do(t, thread, context.Background(), func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	if ran.Load() {
		t.Error("the function ran although its context ended while it was queued")
	}
}

func TestDoSkipsFunctionWhenContextHasAlreadyEnded(t *testing.T) {
	thread := Start()
	defer thread.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var ran atomic.Bool
	err := do(t, thread, ctx, func() error {
		ran.Store(true)
		return nil
	})
	if err != context.Canceled {
		t.Errorf("Do returned %v, want context.Canceled", err)
	}
	if ran.Load() {
		t.Error("the function ran although its context had ended before Do")
	}
}

func TestDoWaitsForStartedFunctionWhenContextEnds(t *testing.T) {
	thread := Start()
	defer thread.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{})
	finish := make(chan struct{})
	result := make(chan error, 1)
	go func() {
		result <- thread.Do(ctx, func() error {
			close(started)
			<-finish
			return nil
		})
	}()
	<-started
	cancel()
	select {
	case err := <-result:
		t.Fatalf("Do returned %v while its function was still running, want it to wait", err)
	case <-time.After(50 * time.Millisecond):
	}
	close(finish)
	select {
	case err := <-result:
		if err != nil {
			t.Errorf("Do returned %v, want the nil the function returned", err)
		}
	case <-time.After(patience):
		t.Fatal("Do did not return after its function finished")
	}
}

// quietContext ends without closing Done, which stands for the moment between
// a context ending and the waiting Do noticing it.
type quietContext struct {
	context.Context
	ended atomic.Bool
}

func (c *quietContext) Done() <-chan struct{} { return nil }

func (c *quietContext) Err() error {
	if c.ended.Load() {
		return context.Canceled
	}
	return nil
}

func TestDoSkipsFunctionWhenContextEndedBeforeTheThreadReachedIt(t *testing.T) {
	thread := Start()
	defer thread.Close()
	release := occupy(t, thread)
	defer release()

	ctx := &quietContext{Context: context.Background()}
	var ran atomic.Bool
	result := make(chan error, 1)
	go func() {
		result <- thread.Do(ctx, func() error {
			ran.Store(true)
			return nil
		})
	}()
	waitQueued(t, thread, 1)
	ctx.ended.Store(true)
	release()

	select {
	case err := <-result:
		if err != context.Canceled {
			t.Errorf("Do returned %v, want context.Canceled", err)
		}
	case <-time.After(patience):
		t.Fatal("Do did not return")
	}
	if ran.Load() {
		t.Error("the function ran although its context ended before the thread reached it")
	}
}

func TestDoReturnsPanicAsError(t *testing.T) {
	thread := Start()
	defer thread.Close()

	err := do(t, thread, context.Background(), func() error { panic("boom") })
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("Do returned %v, want an error containing the panic value \"boom\"", err)
	}
	var panicErr *PanicError
	if !errors.As(err, &panicErr) {
		t.Fatalf("Do returned %T, want *PanicError", err)
	}
	if panicErr.Value != "boom" {
		t.Errorf("PanicError.Value = %v, want \"boom\"", panicErr.Value)
	}
	if !strings.Contains(string(panicErr.Stack), "TestDoReturnsPanicAsError") {
		t.Errorf("PanicError.Stack does not show where the panic happened:\n%s", panicErr.Stack)
	}

	ran := false
	if err := do(t, thread, context.Background(), func() error { ran = true; return nil }); err != nil || !ran {
		t.Errorf("after a panic Do returned %v and ran=%v, want the thread to keep serving", err, ran)
	}
}

func TestPanicErrorUnwrapsToPanickedError(t *testing.T) {
	thread := Start()
	defer thread.Close()

	cause := errors.New("cause")
	err := do(t, thread, context.Background(), func() error { panic(cause) })
	if !errors.Is(err, cause) {
		t.Errorf("errors.Is(%v, cause) is false, want a panic with an error to unwrap to it", err)
	}
}

func TestDoAfterCloseReturnsError(t *testing.T) {
	thread := Start()
	thread.Close()

	ran := false
	err := do(t, thread, context.Background(), func() error { ran = true; return nil })
	if !errors.Is(err, ErrClosed) {
		t.Errorf("Do after Close returned %v, want ErrClosed", err)
	}
	if ran {
		t.Error("Do after Close ran its function")
	}
}

func TestCloseWaitsForRunningFunctionAndFailsQueuedOnes(t *testing.T) {
	thread := Start()
	release := occupy(t, thread)
	defer release()

	var ran atomic.Bool
	queued := make(chan error, 1)
	go func() {
		queued <- thread.Do(context.Background(), func() error {
			ran.Store(true)
			return nil
		})
	}()
	waitQueued(t, thread, 1)

	closed := make(chan struct{})
	go func() {
		thread.Close()
		close(closed)
	}()
	select {
	case err := <-queued:
		if !errors.Is(err, ErrClosed) {
			t.Errorf("the queued Do returned %v, want ErrClosed", err)
		}
	case <-time.After(patience):
		t.Fatal("the queued Do did not return after Close")
	}
	select {
	case <-closed:
		t.Fatal("Close returned while a function was still running, want it to wait")
	case <-time.After(50 * time.Millisecond):
	}

	release()
	select {
	case <-closed:
	case <-time.After(patience):
		t.Fatal("Close did not return after the running function finished")
	}
	if ran.Load() {
		t.Error("a function that was queued at Close ran")
	}
	thread.Close() // a second Close is harmless
}

// goroutineHeader returns the first line of the calling goroutine's stack
// trace, such as "goroutine 7 [running, locked to thread]:".
func goroutineHeader() string {
	buf := make([]byte, 128)
	header, _, _ := strings.Cut(string(buf[:runtime.Stack(buf, false)]), "\n")
	return header
}

func TestEveryFunctionRunsOnTheSameLockedOSThread(t *testing.T) {
	thread := Start()
	defer thread.Close()

	// One goroutine that is locked to its OS thread always runs on that thread.
	var headers []string
	for range 3 {
		err := do(t, thread, context.Background(), func() error {
			headers = append(headers, goroutineHeader())
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, header := range headers {
		if !strings.Contains(header, "locked to thread") {
			t.Errorf("a function ran on %q, want a goroutine locked to its OS thread", header)
		}
		if header != headers[0] {
			t.Errorf("functions ran on %q and %q, want one goroutine for all of them", headers[0], header)
		}
	}
}

func TestDoReportsFunctionThatExitsItsGoroutine(t *testing.T) {
	thread := Start()
	defer thread.Close()

	// runtime.Goexit is what t.FailNow calls; it ends the thread's goroutine.
	err := do(t, thread, context.Background(), func() error {
		runtime.Goexit()
		return nil
	})
	if err == nil {
		t.Error("Do returned nil for a function that never returned, want an error")
	}
	err = do(t, thread, context.Background(), func() error { return nil })
	if !errors.Is(err, ErrClosed) {
		t.Errorf("Do on a thread whose goroutine has exited returned %v, want ErrClosed", err)
	}
}
