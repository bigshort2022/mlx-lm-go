// Package exec serialises work onto one OS thread.
//
// MLX keeps per-thread state (streams, the random key), so everything that
// touches package mlx must run on the same thread. A Thread is that thread.
package exec

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"runtime/debug"
	"slices"
	"sync"
)

// ErrClosed is returned by Do when the thread has been closed.
var ErrClosed = errors.New("exec: thread is closed")

// errExited is the result of a call whose function called runtime.Goexit.
var errExited = errors.New("exec: function exited the thread's goroutine instead of returning")

// PanicError is the error Do returns when its function panicked.
type PanicError struct {
	Value any    // the value passed to panic
	Stack []byte // the stack of the thread at the panic
}

func (e *PanicError) Error() string {
	return fmt.Sprintf("exec: panic: %v", e.Value)
}

// Unwrap returns the panic value if it is an error, so that errors.Is and
// errors.As see through the PanicError.
func (e *PanicError) Unwrap() error {
	err, _ := e.Value.(error)
	return err
}

// Thread runs functions one at a time on a single locked OS thread.
type Thread struct {
	mu      sync.Mutex
	wake    sync.Cond     // signalled when queue or closed changes; uses mu
	queue   []*call       // calls that have not started, oldest first
	closed  bool          // no call will start any more
	stopped chan struct{} // closed when the thread has exited
}

// Start starts a thread. The caller must Close it.
func Start() *Thread {
	t := &Thread{stopped: make(chan struct{})}
	t.wake.L = &t.mu
	go t.run()
	return t
}

// Do queues fn and waits for it. Functions run in arrival order. If ctx ends
// before fn starts, fn never runs and Do returns ctx.Err(). Once started, fn
// runs to completion. A panic in fn is returned as a *PanicError and the
// thread keeps serving. After Close, Do returns ErrClosed.
//
// fn must not call Do or Close on its own thread, which would wait for
// itself. If fn calls runtime.Goexit, as t.FailNow does, Do returns an error
// and the thread closes, because its goroutine is gone.
func (t *Thread) Do(ctx context.Context, fn func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	c := &call{ctx: ctx, fn: fn, done: make(chan struct{})}
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return ErrClosed
	}
	t.queue = append(t.queue, c)
	t.mu.Unlock()
	t.wake.Signal()

	select {
	case <-c.done:
	case <-ctx.Done():
		if t.withdraw(c) {
			return ctx.Err()
		}
		<-c.done // fn has started; it runs to completion
	}
	return c.err
}

// Close stops the thread after the running function and waits for it to
// exit. Calls still queued return ErrClosed without running, and so do later
// Do calls.
func (t *Thread) Close() {
	t.shut()
	t.wake.Signal()
	<-t.stopped
}

// run is the body of the thread: it runs queued calls until Close.
func (t *Thread) run() {
	// Never unlocked, so the OS thread ends with this goroutine instead of
	// going back to the scheduler with MLX's per-thread state on it.
	runtime.LockOSThread()
	defer close(t.stopped)
	// A function that calls runtime.Goexit ends this goroutine without a
	// Close; no call may wait for it after that.
	defer t.shut()
	for {
		c := t.next()
		if c == nil {
			return
		}
		c.run()
	}
}

// next waits for the oldest queued call and removes it from the queue. It
// returns nil once the thread is closed.
func (t *Thread) next() *call {
	t.mu.Lock()
	defer t.mu.Unlock()
	for len(t.queue) == 0 && !t.closed {
		t.wake.Wait()
	}
	if t.closed {
		return nil
	}
	c := t.queue[0]
	t.queue[0] = nil
	t.queue = t.queue[1:]
	return c
}

// withdraw removes c from the queue. It reports false if c has left the
// queue already, which means c.done is or will be closed.
func (t *Thread) withdraw(c *call) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	i := slices.Index(t.queue, c)
	if i < 0 {
		return false
	}
	t.queue = slices.Delete(t.queue, i, i+1)
	return true
}

// shut closes the thread to new calls and fails the queued ones.
func (t *Thread) shut() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.closed = true
	for _, c := range t.queue {
		c.err = ErrClosed
		close(c.done)
	}
	t.queue = nil
}

// call is one function passed to Do.
type call struct {
	ctx  context.Context
	fn   func() error
	err  error         // the result of Do; final once done is closed
	done chan struct{} // closed when the call has finished
}

// run executes the call on the thread and publishes its result.
func (c *call) run() {
	defer close(c.done)
	if c.err = c.ctx.Err(); c.err != nil {
		return // the context ended while the call was queued
	}
	defer func() {
		if r := recover(); r != nil {
			c.err = &PanicError{Value: r, Stack: debug.Stack()}
		}
	}()
	c.err = errExited // stands unless fn returns or panics
	c.err = c.fn()
}
