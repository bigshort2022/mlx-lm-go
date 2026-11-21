# M3: `qwen2` generation

> **For agentic workers:** implement task by task with red/green TDD. For every test: write it, run it, see it fail for the stated reason, then write the smallest code that passes. Steps use checkbox (`- [ ]`) syntax.

**Goal:** load a 4-bit `qwen2` model from disk and generate tokens that match Python mlx-lm, behind the `engine.Engine` interface the HTTP server already uses.

**Architecture:** `internal/nn` holds layers that work on dense or quantized weights. `internal/models/qwen2` is a line-for-line port of `mlx_lm/models/qwen2.py`. `internal/cache` holds the key-value cache. `internal/sample` ports `mlx_lm/sample_utils.py`. `internal/generate` owns one `exec.Thread`, runs stepped prefill and the decode loop on it, and implements `engine.Engine`.

**Tech stack:** Go 1.27, the `mlx` package from M1, `internal/exec`, `internal/engine`, `internal/hub`.

---

## Starting point

- Reference code, to port faithfully: `/Users/b33j_02/sandbox/mlx/.venv/lib/python3.13/site-packages/mlx_lm/` files `models/qwen2.py`, `models/base.py`, `models/cache.py` (`KVCache`), `models/rope_utils.py`, `models/activations.py`, `sample_utils.py`, `generate.py` (`generate_step`), `utils.py` (`load_model`, `load_config`). Quantized layers: `mlx/nn/layers/quantized.py` in the same site-packages.
- Model on disk: `hub.Resolve("mlx-community/Qwen2.5-1.5B-Instruct-4bit")`. Read its `config.json` and list the tensor names in `model.safetensors` before designing.
- Golden fixtures (made by `tools/oracle/make_model_fixtures.py` with mlx-lm 0.32.2 on MLX 0.32.4, the MLX version the Go binary links):
  - `testdata/qwen2/layers.safetensors`: `input_ids` `[1,8]` and float32 `embed`, `l0_norm1`, `l0_attn`, `l0_norm2`, `l0_mlp`, `l0_out`, `final_norm`, `logits_last`.
  - `testdata/qwen2/generation.json`: five cases with `prompt_ids`, `top50_ids`, `top50_logits`, `greedy_ids` (ends at and includes the end-of-sequence token when one was produced within 64 tokens), `peak_memory_bytes`; `meta.eos_token_ids`.
  - `testdata/sampler/cases.json` and `cases.safetensors`: inputs `logits` `[1,32]`, `logprobs`, `history`, and one expected float32 array per case name.
- The `mlx` package API is in `docs/superpowers/plans/2026-11-21-m1-mlx-bindings.md` and in `mlx/*.go`. All `mlx` calls must happen on one OS thread: in tests, call `runtime.LockOSThread()` at the top of each test or run the body through an `exec.Thread`.
- The goldens must come from the MLX version the binary links. GPU `quantized_matmul` on float16 changed between MLX 0.31.2 and 0.32.4, enough to move first-layer attention by 0.035 and final logits by 0.38. Python computes in the weights' dtype (float16); do the same and convert to float32 only to compare.
- Facts about the `mlx` package as built:
  - Streams and the random key are per OS thread. Every test that calls `mlx` must pin its thread (`runtime.LockOSThread()`, also inside each subtest) or run through an `exec.Thread`.
  - `Argmax`, `Argsort`, `ArgPartition` and `Categorical` return `uint32`; use `AsType(x, mlx.Int32)` before `Int32s()`.
  - `LoadSafetensors` loads lazily on the CPU stream. Evaluate the loaded arrays on the thread that loaded them before using them elsewhere; for the engine that means loading the weights on its own inference thread.
  - `ClearCache` synchronises the GPU stream first, so it is not free; call it every 256 tokens, not every token.
  - `exec.Thread.Do` returns a panic as `*exec.PanicError`, through which `errors.As` reaches a `*mlx.Error`.

## Files you own

Create: `internal/nn/*.go`, `internal/models/*.go`, `internal/models/qwen2/*.go`, `internal/cache/*.go`, `internal/sample/*.go`, `internal/generate/*.go`, with `_test.go` files.
You may add operations to `mlx/` when the port needs one that is missing (for example `SliceUpdate`): put each in the existing file for its group, test-first, with an expected value from the Python reference, following that package's conventions.
Do not touch `internal/engine`, `internal/server`, `internal/tokenizer`, `internal/chattemplate`, `go.mod` or root `testdata/`.

## Contract

```go
package models

// Model is a loaded language model. All methods must be called on the inference thread.
type Model interface {
	// Forward returns logits of shape [1, L, vocab] for tokens of shape [1, L],
	// reading and extending state.
	Forward(tokens *mlx.Array, state State) *mlx.Array
	NewState() State
}

// State is the per-generation cache of a model.
type State interface {
	Offset() int          // tokens processed so far
	Arrays() []*mlx.Array // everything that must be evaluated to materialise the state
	Free()
}

type Config struct {
	ModelType    string
	EOSTokenIDs  []int32 // from config.json "eos_token_id", a number or a list; empty when absent
	Quantization *Quantization
	Raw          json.RawMessage // the whole config.json, for the architecture package
}
type Quantization struct {
	GroupSize int
	Bits      int
	Mode      string // "affine" when absent
}

// Load reads config.json and every *.safetensors file in dir and builds the
// model registered for its model_type.
func Load(dir string) (Model, Config, error)
```

```go
package generate

type Options struct {
	PrefillStepSize int     // 0 means 2048
	EOSTokenIDs     []int32 // used only when config.json has none
}

// Engine implements engine.Engine. It owns one inference thread.
type Engine struct{ /* unexported */ }
func Load(dir string, opts Options) (*Engine, error)
func (e *Engine) Generate(ctx context.Context, req engine.Request, emit func(token int32)) (engine.Result, error)
func (e *Engine) Close()
```

Behaviour:

- `Generate` may be called from any goroutine; calls run one at a time.
- `MaxTokens` bounds the tokens the model samples. Generation stops with `"stop"` when an end-of-sequence token is sampled; that token is not emitted but is counted in `CompletionTokens`, as the Python server does (`testdata/server/chat.json`: 13 text tokens, `completion_tokens` 14). It stops with `"length"` when `MaxTokens` tokens were sampled without one, and with `ctx.Err()` within one token of cancellation.
- `Sampling.Temperature == 0` is greedy. Otherwise the order is the reference's: logits processors (logit bias, repetition, presence, frequency), log-softmax, top-p, min-p, top-k, categorical sampling with temperature. `Sampling.Seed`, when set, seeds MLX before the first sample.
- Memory does not grow with the number of generated tokens beyond the cache itself: free temporaries every step and clear the MLX buffer cache every 256 tokens, as the reference does.
- An unknown `model_type` is an error that names it and lists the supported ones.

## Tasks

Run tests per package, for example `go test ./internal/nn/ -run <Name> -v`. Never run `go build ./...`, `go test ./...`, `go mod tidy` or `go get`. Tests that need the model call `hub.Resolve` and `t.Skip` when it fails; on this machine it resolves, so none may be skipped here.

### Task 1: config and weights

- [ ] `TestLoadConfigReadsQuantizationAndEOS` against the real `config.json`: model type `qwen2`, 4 bits, group size 64, EOS `[151645]`.
- [ ] `TestConfigEOSAcceptsNumberOrList` and `TestConfigQuantizationAbsent` with small JSON written by the test.
- [ ] `TestLoadRejectsUnknownModelType`: a temp dir with `{"model_type":"nope"}` gives an error containing `nope` and `qwen2`.

### Task 2: layers

Each test loads the real weights and `layers.safetensors`, applies one layer to the golden input of that stage, and compares with the golden output. Print the maximum absolute difference; require it below `2e-2`.

- [ ] Quantized embedding: `input_ids` to `embed`.
- [ ] RMSNorm: `embed` to `l0_norm1`.
- [ ] Linear with and without bias, quantized: covered through the attention and MLP tests below; add a direct test of a dense `Linear` on hand-made weights so the unquantized path is exercised.

### Task 3: the transformer block

- [ ] Attention: `l0_norm1` to `l0_attn` with the `"causal"` mask and no cache.
- [ ] MLP: `l0_norm2` to `l0_mlp`.
- [ ] Block: `embed` to `l0_out`.

### Task 4: the whole model

- [ ] `TestForwardMatchesGoldenLogits`: `input_ids` to logits; compare the last position with `logits_last`: same argmax, maximum absolute difference below `1e-1`.
- [ ] `TestStep0Top50MatchesGolden`: for each case of `generation.json`, a forward pass over `prompt_ids`; the argmax equals `top50_ids[0]`, at least 48 of our top 50 ids are in the golden top 50, and the logits of the golden ids are within `1e-1`.

### Task 5: the cache

- [ ] `TestIncrementalForwardEqualsFullForward`: feeding the first seven golden tokens and then the eighth through a state gives last-position logits within `5e-2` of the full pass.
- [ ] `TestStateOffsetCountsTokens`.
- [ ] `TestCacheGrowsPastOneAllocationStep`: 300 single-token steps succeed and the offset is 300.
- [ ] `TestStateFreeReleasesMemory`: active memory after `Free` and `ClearCache` returns to within 1 MiB of the level before the state was created.

### Task 6: sampling

- [ ] One subtest per case in `testdata/sampler/cases.json`, comparing with the named array in `cases.safetensors`: finite values within `1e-4` and `-inf` in exactly the same positions.
- [ ] `TestGreedyIgnoresSamplingFilters`: temperature 0 returns the argmax whatever the other fields are.
- [ ] `TestSeedMakesSamplingReproducible`: the same seed gives the same 20 samples twice; a different seed gives a different sequence.
- [ ] `TestSamplerOrderMatchesReference`: with temperature 1, top-k 1 and a logit bias that moves the maximum, the biased token is returned.

### Task 7: the engine

- [ ] `TestGenerateGreedyMatchesGolden`: for each case, `Generate` with temperature 0 and `MaxTokens` 64 emits exactly `greedy_ids` without its final end-of-sequence token when it has one; the finish reason is `"stop"` in that case and `"length"` otherwise; `PromptTokens == len(prompt_ids)` and `CompletionTokens == len(greedy_ids)`. If a case diverges at a position where the golden top two logits are within `5e-2`, do not loosen the test: report the position and both values.
- [ ] `TestGenerateStopsAtMaxTokens`: `MaxTokens` 3 gives three tokens and `"length"`.
- [ ] `TestMaxTokensCountsTheEndOfSequenceToken`: for case 1 (`greedy_ids` has 3 entries, the last being end-of-sequence), `MaxTokens` 3 emits two tokens and finishes `"stop"`; `MaxTokens` 2 emits two tokens and finishes `"length"`.
- [ ] `TestGeneratePrefillStepSizeDoesNotChangeOutput`: step sizes 4 and 2048 give the same greedy tokens.
- [ ] `TestGenerateCancelsWithinOneToken`: cancel the context from inside `emit` after the second token; `Generate` returns `context.Canceled` and `emit` was called at most three times.
- [ ] `TestGenerateRunsOneAtATime`: two concurrent calls both finish with the golden tokens.
- [ ] `TestGenerateMemoryIsBoundedAcrossRequests`: after ten sequential generations, active memory is within 16 MiB of the level after the first.
- [ ] `TestPeakMemoryCloseToReference`: peak memory for case 0 is within 10% of its `peak_memory_bytes`. If it is not, report both numbers rather than changing the bound.

## Done when

`go test ./mlx/ ./internal/nn/ ./internal/models/... ./internal/cache/ ./internal/sample/ ./internal/generate/ && go vet ./internal/nn/ ./internal/models/... ./internal/cache/ ./internal/sample/ ./internal/generate/` passes with no skipped model test on this machine, and `gofmt -l internal/nn internal/models internal/cache internal/sample internal/generate mlx` prints nothing.

## Report

For each task: the test names, the failure you saw before implementing (one line each), and the final pass. Give the measured maximum differences for Tasks 2 to 5, the decode speed in tokens per second for case 4, and the peak-memory comparison. Do not commit.
