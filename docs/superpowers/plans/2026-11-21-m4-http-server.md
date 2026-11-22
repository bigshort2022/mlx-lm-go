# M4: HTTP server on a fake engine

> **For agentic workers:** implement task by task with red/green TDD. For every test: write it, run it, see it fail for the stated reason, then write the smallest code that passes. Steps use checkbox (`- [ ]`) syntax.

**Goal:** the OpenAI-compatible HTTP layer of `mlx-lm-go`, shape-compatible with Python `mlx_lm.server` 0.31.3 and safe to reach from a browser, tested end to end against a deterministic fake engine.

**Architecture:** `internal/engine` defines the `Engine` interface and a fake. `internal/server` builds an `http.Handler` from a config and three dependencies (engine, tokenizer, chat template), each behind a small interface defined in `internal/server`. No MLX, no real tokenizer and no real template are imported here; they are wired in a later milestone.

**Tech stack:** Go 1.27 standard library only (`net/http`, `net/http/httptest`, `encoding/json`, `flag`).

---

## Starting point

- Recorded exchanges with the real Python server are in `testdata/server/*.json` (35 files plus `meta.json`). Each has `request` and `response` (`status`, `headers`, `body`). `tools/oracle/record_server.py` lists the cases.
- The reference implementation is `/Users/b33j_02/sandbox/mlx/.venv/lib/python3.13/site-packages/mlx_lm/server.py`. Read it for semantics the recordings do not settle: defaults, validation messages, stop-word handling, how `chat_template_kwargs` merge.
- Ten recordings show the Python server dropping the connection (`"connection_error"`) or answering 404 for a bad request. Those are reference bugs. This server answers such requests with HTTP 400 and `{"error": "<message>"}`.

## Files you own

Create: `internal/engine/{engine,fake}.go`, `internal/engine/fake_test.go`, `internal/server/{config,flags,server,security,chat,completions,models,health,sse,stop,queue,validate,shape_test_helpers_test}.go` and `_test.go` files, `tools/conformance/openai_client.py`.
Do not touch anything else, including `go.mod`, `go.sum` and `testdata/`.

## Contract

```go
package engine

// Engine generates tokens for one request at a time.
type Engine interface {
	// Generate runs until the model emits an end-of-sequence token, MaxTokens
	// tokens have been emitted, or ctx is cancelled. emit is called once per
	// generated token, in order, and must not block for long. The
	// end-of-sequence token is not emitted.
	Generate(ctx context.Context, req Request, emit func(token int32)) (Result, error)
}

type Request struct {
	Prompt    []int32
	MaxTokens int
	Sampling  Sampling
}

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
	LogitBias             map[int32]float32
	Seed                  *uint64
}

type Result struct {
	PromptTokens     int
	CompletionTokens int
	CachedTokens     int
	FinishReason     string // "stop" or "length"
}

// Fake is a scripted Engine for tests.
type Fake struct {
	Tokens []int32                    // emitted in order, then finish "stop"
	Gate   func(i int, token int32)   // if set, called before emitting token i; tests block here
	Calls  atomic.Int64               // number of Generate calls started
	Last   atomic.Pointer[Request]    // most recent request
}
```

`Fake.Generate` emits `Tokens` up to `MaxTokens` (finish `"length"` if it ran out of budget before the script ended, else `"stop"`), returns `ctx.Err()` as soon as the context ends, and reports `PromptTokens = len(req.Prompt)` and `CompletionTokens` = the number of tokens it emitted. The server copies `usage` from the `Result`; it does not count `emit` calls, because a real engine also counts the end-of-sequence token it does not emit.

```go
package server

type Tokenizer interface {
	Encode(text string) []int32
	NewDetokenizer() Detokenizer
}
type Detokenizer interface {
	Add(token int32) string
	Flush() string
}
type Templater interface {
	Render(messages, tools, kwargs json.RawMessage, addGenerationPrompt bool) (string, error)
}

type Config struct {
	Host             string   // default "127.0.0.1"
	Port             int      // default 8080
	AllowedOrigins   []string // default empty: same-origin only. "*" allows any origin.
	Model            string   // value of --model; the id listed by /v1/models
	LogLevel         string   // default "INFO"
	ChatTemplate     string   // --chat-template; "" means use the model's
	ChatTemplateArgs json.RawMessage // default {}
	Temperature      float32  // default 0.0
	TopP             float32  // default 1.0
	TopK             int      // default 0
	MinP             float32  // default 0.0
	MaxTokens        int      // default 512
	PrefillStepSize  int      // default 2048
}

type Deps struct {
	Engine    engine.Engine
	Tokenizer Tokenizer
	Template  Templater
}

// ParseFlags parses mlx_lm.server-compatible flags (without the program name).
func ParseFlags(args []string) (Config, error)

// New returns the HTTP handler.
func New(cfg Config, deps Deps) http.Handler
```

## Tasks

Run tests with `go test -race ./internal/server/ -run <Name> -v` (or `./internal/engine/`). Never run `go build ./...`, `go test ./...`, `go mod tidy` or `go get`.

Build two test helpers first, each test-driven:

- A fake tokenizer and detokenizer that map token ids to fixed text pieces, so a test can replay the exact piece sequence of a recording (for `chat_stream.json`: `"1"`, `","`, `" 2"`, `","`, `" 3"`, `","`, `" 4"`, `","`, `" 5"`).
- `shapeOf(json) `: maps a JSON value to its shape (object keys with the shape of each value, the shape of an array's first element, and the type name of scalars). Tests compare `shapeOf(our body)` with `shapeOf(recorded body)`.

### Task 1: the fake engine

- [ ] Emits the script in order and finishes `"stop"`.
- [ ] Stops at `MaxTokens` with `"length"`.
- [ ] Returns `context.Canceled` when the context ends while blocked in `Gate`.
- [ ] `Calls` and `Last` record the request.

### Task 2: flags

- [ ] Defaults equal the `Config` comments above.
- [ ] Table test: each of `--model --host --port --allowed-origins --log-level --chat-template --chat-template-args --temp --top-p --top-k --min-p --max-tokens --prefill-step-size` sets its field. `--allowed-origins a,b` gives two entries.
- [ ] Each of `--adapter-path --draft-model --num-draft-tokens --trust-remote-code --use-default-chat-template --decode-concurrency --prompt-concurrency --pipeline --prompt-cache-size --prompt-cache-bytes` returns an error that names the flag and says it is not supported yet.
- [ ] An unknown flag, a missing `--model`, and `--chat-template-args` that is not a JSON object each return an error.

### Task 3: health, models, unknown paths

- [ ] `GET /health` matches `health.json`: status 200, body JSON-equal to `{"status": "ok"}`, `Content-Type: application/json`.
- [ ] `GET /v1/models`: shape equals `models.json`; `data` has exactly one entry whose `id` is `Config.Model`. (Python lists every cached model; this server cannot switch models, so it lists the one it serves.)
- [ ] `GET /v1/nope` and `POST /v1/nope` return 404 like `get_unknown_path.json` and `post_unknown_path.json`.

### Task 4: chat completions, not streamed

- [ ] Replay `chat.json`: fake engine emits the recorded pieces. Status 200; `shapeOf` equal to the recording; `choices[0].message.content == "1, 2, 3, 4, 5"`; `finish_reason == "stop"`; `usage.completion_tokens` equals the number of emitted tokens and `total_tokens` is the sum; `object == "chat.completion"`; `id` starts with `chatcmpl-`; `model == "default_model"` when the request has no `model`.
- [ ] `chat_length.json`: `max_tokens: 1` gives `finish_reason == "length"` and one token of content.
- [ ] The prompt passed to the engine is `Tokenizer.Encode(Template.Render(messages, tools=null, kwargs, true))`. Assert it through `Fake.Last` with a fake templater that records its arguments.
- [ ] `chat_template_kwargs` in the request is merged over `Config.ChatTemplateArgs`, request keys winning. Assert the merged JSON the templater received.
- [ ] `chat_content_parts.json`: content given as text parts is joined into one string before templating.
- [ ] Sampling fields from `chat_sampling.json` arrive in `Fake.Last.Sampling`; fields absent from the request take the `Config` defaults.
- [ ] `model`: absent, `"default_model"`, or equal to `Config.Model` is accepted and echoed in the response. Any other value returns 400 naming the served model.
- [ ] A templater error returns 400 with the error text.

### Task 5: chat completions, streamed

- [ ] Replay `chat_stream.json`. Response headers `Content-Type: text/event-stream` and `Cache-Control: no-cache`. The sequence of `data:` events has the same shapes as the recording, the same `delta.content` values in order, a final chunk with `finish_reason` set and `delta == {"role": "assistant"}`, then `data: [DONE]`. Every chunk shares one `id`. The recording's leading `: keepalive` comment is not required.
- [ ] Tokens are flushed as they are produced: with `Fake.Gate` blocking before token 2, the client has already read the first chunk.
- [ ] `chat_stream_usage.json`: with `stream_options.include_usage` a usage chunk precedes `[DONE]`, same shape as recorded. Without it there is none.
- [ ] Client disconnect cancels generation: close the client connection while the engine is blocked in `Gate`; `Generate` returns `context.Canceled`.

### Task 6: stop strings

Read the stop-word handling in the reference before writing tests.

- [ ] Replay `chat_stop_string.json` (stop `["3"]`) and `chat_stop_single.json` (stop `", 4"`) with the recorded piece sequence: content equals the recording (`"1, 2,"` and `"1, 2, 3"`), `finish_reason == "stop"`, and the engine was cancelled rather than run to the end of its script.
- [ ] Replay `chat_stream_stop.json`: the streamed pieces equal the recording's; nothing from the stop match onward is sent.
- [ ] A stop string that arrives split across pieces (`"3"` then `"4"` with stop `"34"`) is never emitted, streamed or not.
- [ ] Text that only looked like the start of a stop string is released once the match fails (pieces `"3"`, `"5"` with stop `"34"` yield `"35"`).

### Task 7: text completions

- [ ] `completions.json` and `completions_stream.json`: same shape checks. The prompt is `Tokenizer.Encode(prompt)` with no template. `id` starts with `cmpl-`, `object == "text_completion"`.

### Task 8: validation

- [ ] `err_invalid_json.json` and `err_not_object.json`: status 400, body shape as recorded, same message for the not-an-object case.
- [ ] Each of these returns 400 with `{"error": ...}` naming the field: missing `messages`; `messages` not an array; `stream` not a boolean; `max_tokens` negative; `temperature` negative; `top_p` above 1; `logit_bias` not an object; missing `prompt` on `/v1/completions`. Use the reference's wording where it has one (for example `max_tokens must be at least 0`).
- [ ] A content part that is not text returns 400 with `Only 'text' content type is supported.` (`chat_image_part.json` shows the message; the reference's 404 is not copied).
- [ ] `adapters`, `draft_model`, `num_draft_tokens`, `role_mapping`, `tools`, `logprobs` and `top_logprobs` each return 400 saying the field is not supported yet. The engine is not called.
- [ ] A body over 16 MiB returns 413.

### Task 9: security

Each rule gets its own test, and each rejection asserts `Fake.Calls == 0`.

- [ ] Host: with a loopback `Config.Host`, a request whose `Host` header is not `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>` returns 403. With a non-loopback `Config.Host` the check is off.
- [ ] Origin: no `Origin` header is allowed. An `Origin` equal to `http://<Host header>` is allowed. An origin in `AllowedOrigins` is allowed. Any other origin returns 403, also on `GET` and `OPTIONS`. `AllowedOrigins: ["*"]` allows all.
- [ ] `POST` without `Content-Type: application/json` (parameters such as `; charset=utf-8` are fine) returns 415.
- [ ] CORS: for an allowed cross-origin request the response carries `Access-Control-Allow-Origin: <that origin>` and `Vary: Origin`; with `"*"` configured it carries `*`. Same-origin and origin-less requests get no CORS headers.
- [ ] Preflight: `OPTIONS` from an allowed origin returns 204 with `Access-Control-Allow-Methods: GET, POST, OPTIONS` and `Access-Control-Allow-Headers` echoing `Access-Control-Request-Headers`.

### Task 10: one request at a time

- [ ] Two concurrent chat requests: the second `Generate` starts only after the first returned (`Gate` on the first; assert `Calls == 1` while it blocks).
- [ ] Three queued requests are served in arrival order.
- [ ] A queued request whose client disconnects never reaches the engine.

### Task 11: client conformance script

- [ ] `tools/conformance/openai_client.py <base_url>`: uses the official `openai` Python package to call models list, a non-streamed chat, a streamed chat and a streamed chat with a stop string, and exits non-zero with a message on the first failed assertion.
- [ ] `TestOpenAIClientConformance`: starts the handler with `httptest` on a loopback port and the fake engine, runs the script with the interpreter named by `MLX_LM_GO_PYTHON`, and skips when that variable is unset. On this machine run it with `MLX_LM_GO_PYTHON=/Users/b33j_02/sandbox/mlx/.venv/bin/python`.

## Done when

`go test -race ./internal/engine/ ./internal/server/ && go vet ./internal/engine/ ./internal/server/` passes with clean output, the conformance test passes with `MLX_LM_GO_PYTHON` set, and `gofmt -l internal/engine internal/server` prints nothing.

## Report

For each task: the test names, the failure you saw before implementing (one line each), and the final pass. List every place you chose different behaviour from the reference and why. Do not commit.
