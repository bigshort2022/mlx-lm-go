package server

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
)

// job is a validated completion request that is ready to run.
type job struct {
	chat      bool // chat completion; otherwise a text completion
	model     string
	stream    bool
	usage     bool     // end the stream with a usage event
	stop      []string // strings that end the completion when generated
	maxTokens int
	sampling  engine.Sampling
	tools     *toolSet
	// prompt builds the prompt tokens. Its error is reported to the client as
	// a bad request.
	prompt func() ([]int32, error)
}

// run generates the completion for j and writes the response.
func (s *server) run(w http.ResponseWriter, r *http.Request, j job) {
	out := newReply(j.chat, j.model)
	if j.stream {
		s.runStreamed(w, r, j, out)
	} else {
		s.runBuffered(w, r, j, out)
	}
}

// runBuffered answers with the whole completion in one JSON body. The body is
// written after generate has given the turn back, so a client that is slow to
// take it delays nobody else.
func (s *server) runBuffered(w http.ResponseWriter, r *http.Request, j job, out reply) {
	var text, reasoning strings.Builder
	var calls []toolCall
	res, err := s.generate(r.Context(), j, func() {}, func(piece string, thinking bool) error {
		if thinking {
			reasoning.WriteString(piece)
		} else {
			text.WriteString(piece)
		}
		return nil
	}, func(parsed []toolCall) error { calls = parsed; return nil })
	if err != nil {
		writeFailure(w, err)
		return
	}
	writeJSON(w, http.StatusOK, out.full(text.String(), reasoning.String(), res, calls))
}

// runStreamed answers with one event per piece of text as it is generated,
// then a final event carrying the finish reason and, if asked for, a usage
// event. A failure after the headers have gone out is reported as an error
// event, and the stream ends without the [DONE] marker.
func (s *server) runStreamed(w http.ResponseWriter, r *http.Request, j job, out reply) {
	var events *eventStream
	res, err := s.generate(r.Context(), j,
		func() { events = startEventStream(w, s.writeTimeout) },
		func(piece string, thinking bool) error {
			if thinking {
				return events.send(out.chunk("", piece, ""))
			}
			return events.send(out.chunk(piece, "", ""))
		},
		func(calls []toolCall) error {
			for i := range calls {
				calls[i].Index = &i
			}
			return events.send(out.toolChunk(calls))
		},
	)
	if events == nil {
		writeFailure(w, err) // refused before the stream began
		return
	}
	if err != nil {
		// Reaches the client unless the failure was the client's own.
		_ = events.send(errorResponse{Error: err.Error()})
		return
	}
	_ = events.send(out.chunk("", "", res.FinishReason))
	if j.usage {
		_ = events.send(out.usageChunk(res))
	}
	_ = events.done()
}

// promptError is a failure to build the prompt, which is the client's doing.
type promptError struct{ error }

// writeFailure answers a request whose generation did not complete.
func writeFailure(w http.ResponseWriter, err error) {
	var badPrompt promptError
	switch {
	case errors.Is(err, errQueueFull):
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusServiceUnavailable, err.Error())
	case errors.As(err, &badPrompt):
		writeError(w, http.StatusBadRequest, err.Error())
	default:
		// Also reached when the client hung up; the write then goes nowhere.
		writeError(w, http.StatusInternalServerError, err.Error())
	}
}

// generate takes the request's turn, builds the prompt and runs the engine.
// Requests get their turn one at a time, in arrival order; everything that
// touches the tokenizer, the template or the engine happens inside the turn,
// and the turn ends when generate returns.
//
// begin is called once the prompt is ready and generation is about to start.
// Generated text and reasoning are streamed via onText; onTools receives a
// completed, validated tool turn, never a partial call. A sink error cancels
// generation and is returned to the caller.
// A user stop also cancels generation and sets the finish reason to "stop";
// the token counts are still the engine's.
func (s *server) generate(ctx context.Context, j job, begin func(), onText func(piece string, thinking bool) error, onTools func([]toolCall) error) (engine.Result, error) {
	if err := s.queue.acquire(ctx); err != nil {
		return engine.Result{}, err
	}
	defer s.queue.release()

	prompt, err := j.prompt()
	if err != nil {
		return engine.Result{}, promptError{err}
	}
	begin()

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	detok := s.deps.Tokenizer.NewDetokenizer()
	matcher := newStopMatcher(j.stop)
	var toolText toolOutput
	emitText := onText
	if j.tools != nil {
		emitText = func(piece string, reasoning bool) error {
			if reasoning {
				return onText(piece, true)
			}
			return toolText.add(piece, onText)
		}
	}
	emit := func(piece string) error { return emitText(piece, false) }
	var thinking thinkingSplitter
	splitThinking := j.chat && s.deps.ThinkingTokens != nil
	if splitThinking {
		thinking.reasoning = s.deps.ThinkingTokens.startsInReasoning(prompt)
		if j.tools != nil {
			thinking.literal = &toolText.started
		}
		emit = func(piece string) error { return thinking.add(piece, emitText) }
	}
	var (
		stopped bool  // a stop string was completed
		sinkErr error // onText failed
	)
	feed := func(piece string) {
		if stopped || sinkErr != nil {
			return
		}
		text, hit := matcher.add(piece)
		if text != "" {
			sinkErr = emit(text)
		}
		stopped = hit
		if stopped || sinkErr != nil {
			cancel()
		}
	}

	req := engine.Request{Prompt: prompt, MaxTokens: j.maxTokens, Sampling: j.sampling}
	res, err := s.deps.Engine.Generate(ctx, req, func(token int32) { feed(detok.Add(token)) })
	if err == nil {
		feed(detok.Flush())
		if rest := matcher.flush(); rest != "" && !stopped && sinkErr == nil {
			sinkErr = emit(rest)
		}
		if splitThinking && res.FinishReason == "length" && !stopped && sinkErr == nil {
			sinkErr = thinking.flush(emitText)
		}
	}
	switch {
	case sinkErr != nil:
		return res, sinkErr
	case stopped:
		res.FinishReason = "stop"
		err = nil
	}
	if err == nil && j.tools != nil {
		calls, parseErr := toolText.finish(j.tools, res.FinishReason == "length", onText)
		if parseErr != nil {
			return res, parseErr
		}
		if len(calls) != 0 {
			res.FinishReason = "tool_calls"
			return res, onTools(calls)
		}
	}
	return res, err
}
