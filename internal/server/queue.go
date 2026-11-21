package server

import (
	"context"
	"slices"
	"sync"
)

// queue lets one caller at a time through and makes the others wait in
// arrival order. The zero value is an empty queue.
type queue struct {
	mu      sync.Mutex
	busy    bool
	waiters []chan struct{} // closed, oldest first, to hand the turn over
}

// acquire waits for the caller's turn and returns nil once it has it; the
// caller must then call release when it is done. If ctx ends first, acquire
// returns its error and the caller has no turn to release.
func (q *queue) acquire(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	q.mu.Lock()
	if !q.busy {
		q.busy = true
		q.mu.Unlock()
		return nil
	}
	turn := make(chan struct{})
	q.waiters = append(q.waiters, turn)
	q.mu.Unlock()

	select {
	case <-turn:
	case <-ctx.Done():
	}
	if err := ctx.Err(); err != nil {
		if !q.leave(turn) {
			q.release() // the turn had just been handed over; pass it on
		}
		return err
	}
	return nil
}

// leave takes a waiter out of the line and reports whether it was still in
// it.
func (q *queue) leave(turn chan struct{}) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	i := slices.Index(q.waiters, turn)
	if i < 0 {
		return false
	}
	q.waiters = slices.Delete(q.waiters, i, i+1)
	return true
}

// release ends the caller's turn and hands it to the longest-waiting caller.
func (q *queue) release() {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.waiters) == 0 {
		q.busy = false
		return
	}
	close(q.waiters[0])
	q.waiters = q.waiters[1:]
}
