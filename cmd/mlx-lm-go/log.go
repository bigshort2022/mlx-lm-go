package main

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
)

// logLevels maps the values of --log-level, which are those of
// mlx_lm.server, to slog levels.
var logLevels = map[string]slog.Level{
	"DEBUG":    slog.LevelDebug,
	"INFO":     slog.LevelInfo,
	"WARNING":  slog.LevelWarn,
	"ERROR":    slog.LevelError,
	"CRITICAL": slog.LevelError + 4,
}

// newLogger returns a logger that writes the records from level up to w.
func newLogger(w io.Writer, level string) *slog.Logger {
	return slog.New(slog.NewTextHandler(w, &slog.HandlerOptions{Level: logLevels[level]}))
}

// tokenCounts are the tokens the engine processed for one request.
type tokenCounts struct{ prompt, completion int }

// tokenCountsKey is the context key under which a request carries its
// *tokenCounts.
type tokenCountsKey struct{}

// logRequests wraps next so that every finished request is logged at INFO
// with its status, its token counts and how long it took. What the client
// sent in the body and what the model generated are never logged.
func logRequests(logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		var counts tokenCounts
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r.WithContext(context.WithValue(r.Context(), tokenCountsKey{}, &counts)))
		logger.LogAttrs(r.Context(), slog.LevelInfo, "request",
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
			slog.Int("status", rec.status),
			slog.Int("prompt_tokens", counts.prompt),
			slog.Int("completion_tokens", counts.completion),
			slog.Duration("duration", time.Since(start)))
	})
}

// statusRecorder remembers the status of a response.
type statusRecorder struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (r *statusRecorder) WriteHeader(status int) {
	if !r.wroteHeader {
		r.status, r.wroteHeader = status, true
	}
	r.ResponseWriter.WriteHeader(status)
}

func (r *statusRecorder) Write(p []byte) (int, error) {
	r.wroteHeader = true // an implicit 200
	return r.ResponseWriter.Write(p)
}

// Unwrap lets http.ResponseController reach the real writer, which streams
// need for flushing and write deadlines.
func (r *statusRecorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

// countingEngine reports the token counts of each generation to the request
// it runs for, which logRequests put into the context.
type countingEngine struct{ engine.Engine }

func (e countingEngine) Generate(ctx context.Context, req engine.Request, emit func(token int32)) (engine.Result, error) {
	res, err := e.Engine.Generate(ctx, req, emit)
	if counts, ok := ctx.Value(tokenCountsKey{}).(*tokenCounts); ok {
		counts.prompt, counts.completion = res.PromptTokens, res.CompletionTokens
	}
	return res, err
}
