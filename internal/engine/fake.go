package engine

import (
	"context"
	"sync/atomic"
)

// Fake is a scripted Engine for tests.
type Fake struct {
	Tokens []int32                  // emitted in order, then finish "stop"
	Gate   func(i int, token int32) // if set, called before emitting token i; tests block here
	Calls  atomic.Int64             // number of Generate calls started
	Last   atomic.Pointer[Request]  // most recent request
}

// Generate emits the scripted tokens until the script or the MaxTokens budget
// runs out. It finishes with "length" only when the budget cut the script
// short. It returns ctx.Err() as soon as ctx ends, even while Gate is blocking.
func (f *Fake) Generate(ctx context.Context, req Request, emit func(token int32)) (Result, error) {
	f.Calls.Add(1)
	f.Last.Store(&req)
	res := Result{PromptTokens: len(req.Prompt)}
	for i, token := range f.Tokens {
		if res.CompletionTokens == req.MaxTokens {
			res.FinishReason = "length"
			return res, nil
		}
		if err := f.pass(ctx, i, token); err != nil {
			return res, err
		}
		emit(token)
		res.CompletionTokens++
	}
	res.FinishReason = "stop"
	return res, nil
}

// pass waits for Gate to let token i through and reports ctx.Err() if ctx ends
// first. Gate runs on its own goroutine so that a blocked Gate cannot keep
// Generate from noticing cancellation.
func (f *Fake) pass(ctx context.Context, i int, token int32) error {
	if f.Gate != nil {
		opened := make(chan struct{})
		go func() {
			defer close(opened)
			f.Gate(i, token)
		}()
		select {
		case <-opened:
		case <-ctx.Done():
		}
	}
	return ctx.Err()
}
