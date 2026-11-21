package server

import (
	"context"
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
	// prompt builds the prompt tokens. Its error is reported to the client as
	// a bad request.
	prompt func() ([]int32, error)
}

// job reads the fields that chat and text completions share.
func (s *server) job(fields *reader) job {
	fields.rejectUnsupported()
	return job{
		model:     s.model(fields),
		stream:    fields.boolean("stream", false),
		usage:     fields.includeUsage(),
		stop:      fields.stop(),
		maxTokens: s.maxTokens(fields),
		sampling:  s.sampling(fields),
	}
}

// run waits for the request's turn, generates the completion for j and writes
// the response. Requests are served one at a time, in arrival order; one
// whose client leaves while it waits is dropped before it costs anything.
// Everything that touches the tokenizer, the template or the engine happens
// inside the turn.
func (s *server) run(w http.ResponseWriter, r *http.Request, j job) {
	if err := s.queue.acquire(r.Context()); err != nil {
		return // the client has gone, so there is nobody to answer
	}
	defer s.queue.release()

	prompt, err := j.prompt()
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	source := textSource{
		deps:    s.deps,
		request: engine.Request{Prompt: prompt, MaxTokens: j.maxTokens, Sampling: j.sampling},
		stop:    j.stop,
	}
	out := newReply(j.chat, j.model)
	if j.stream {
		runStreamed(r.Context(), w, source, out, j.usage)
	} else {
		runBuffered(r.Context(), w, source, out)
	}
}

// runBuffered answers with the whole completion in one JSON body.
func runBuffered(ctx context.Context, w http.ResponseWriter, source textSource, out reply) {
	var text strings.Builder
	res, err := source.generate(ctx, func(piece string) { text.WriteString(piece) })
	if err != nil {
		// Also reached when the client hung up; the write then goes nowhere.
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, out.full(text.String(), res))
}

// runStreamed answers with one event per piece of text as it is generated,
// then a final event carrying the finish reason and, if asked for, a usage
// event. A failure after the headers have gone out is reported as an error
// event, and the stream ends without the [DONE] marker.
func runStreamed(ctx context.Context, w http.ResponseWriter, source textSource, out reply, usage bool) {
	events := startEventStream(w)
	res, err := source.generate(ctx, func(piece string) { events.send(out.chunk(piece, "")) })
	if err != nil {
		events.send(errorResponse{Error: err.Error()})
		return
	}
	events.send(out.chunk("", res.FinishReason))
	if usage {
		events.send(out.usageChunk(res))
	}
	events.done()
}

// textSource generates the text of one completion.
type textSource struct {
	deps    Deps
	request engine.Request
	stop    []string
}

// generate runs the engine and hands the generated text to onText in pieces,
// as soon as each is known not to be part of a stop string. When a stop
// string appears it cancels the engine and reports the finish reason "stop";
// the token counts are still the engine's.
func (t textSource) generate(ctx context.Context, onText func(piece string)) (engine.Result, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	detok := t.deps.Tokenizer.NewDetokenizer()
	matcher := newStopMatcher(t.stop)
	stopped := false
	feed := func(piece string) {
		text, hit := matcher.add(piece)
		if text != "" {
			onText(text)
		}
		if hit {
			stopped = true
			cancel()
		}
	}

	res, err := t.deps.Engine.Generate(ctx, t.request, func(token int32) {
		if !stopped {
			feed(detok.Add(token))
		}
	})
	if !stopped {
		if err != nil {
			return res, err
		}
		feed(detok.Flush())
	}
	if stopped {
		res.FinishReason = "stop"
		return res, nil
	}
	if rest := matcher.flush(); rest != "" {
		onText(rest)
	}
	return res, nil
}
