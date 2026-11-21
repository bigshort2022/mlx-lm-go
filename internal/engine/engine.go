// Package engine defines the interface between the HTTP server and whatever
// generates tokens, together with a scripted fake for tests.
package engine

import "context"

// Engine generates tokens for one request at a time.
type Engine interface {
	// Generate runs until the model emits an end-of-sequence token, MaxTokens
	// tokens have been emitted, or ctx is cancelled. emit is called once per
	// generated token, in order, and must not block for long. The
	// end-of-sequence token is not emitted.
	//
	// emit is never called after Generate has returned.
	//
	// When ctx ends first, Generate returns ctx.Err() together with a Result
	// holding the token counts reached so far.
	Generate(ctx context.Context, req Request, emit func(token int32)) (Result, error)
}

// Request is one generation job.
type Request struct {
	Prompt    []int32 // token ids of the full prompt
	MaxTokens int     // upper bound on generated tokens
	Sampling  Sampling
}

// Sampling holds the per-request sampling parameters. Field meanings follow
// the request fields of the same name in mlx_lm.server.
type Sampling struct {
	Temperature           float32
	TopP                  float32
	TopK                  int
	MinP                  float32
	RepetitionPenalty     float32
	RepetitionContextSize int
	PresencePenalty       float32
	PresenceContextSize   int
	FrequencyPenalty      float32
	FrequencyContextSize  int
	LogitBias             map[int32]float32 // token id to additive bias
	Seed                  *uint64           // nil means unseeded
}

// Result summarises a finished generation.
type Result struct {
	PromptTokens     int
	CompletionTokens int
	CachedTokens     int    // prompt tokens served from a cache
	FinishReason     string // "stop" or "length"
}
