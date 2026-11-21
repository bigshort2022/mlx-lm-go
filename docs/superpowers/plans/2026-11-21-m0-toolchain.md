# M0: toolchain and spikes (completed 2026-11-21)

**Goal:** prove the toolchain before any milestone code is written.

## What was done

- [x] Installed Go 1.27.4 (`brew install go`).
- [x] Installed Homebrew's `mlx-c` 0.8.0 (`brew install mlx-c`), which links Homebrew's MLX 0.32.4. This replaces the submodule and CMake build in the master plan.
- [x] Created the module `github.com/bigshort2022/mlx-lm-go` with `LICENSE`, `Makefile` and `.gitignore`.
- [x] Pinned `dlclark/regexp2` v1.12.0, `nikolalohinski/gonja/v2` v2.9.1 and `golang.org/x/text` v0.44.0 in `go.mod`. `internal/deps/deps.go` holds them until real importers exist.
- [x] Spike 1, linking. Red: `go test ./mlx/` failed with `undefined: Version`. Green: `mlx/cgo.go` and `mlx/version.go`; `TestVersionReportsLinkedMLX` passes, also when the test binary runs from `/tmp`.
- [x] `internal/hub.Resolve`, written test-first: six tests in `internal/hub/resolve_test.go`.
- [x] Golden fixtures from the Python reference (mlx-lm 0.31.3, transformers 5.7.0, tokenizers 0.22.2, jinja2 3.1.6):
  - `testdata/{qwen2,qwen3_5}/{pretokenize,tokenize,decode,template}.jsonl` from `tools/oracle/make_text_fixtures.py`.
  - `testdata/server/*.json`, 35 recorded exchanges from `tools/oracle/record_server.py`.

## Findings

| Question | Answer |
|---|---|
| Where must `mlx.metallib` live? | Nowhere special. Linking `libmlxc.dylib` dynamically lets MLX find its own Metal library; a C example and the Go test both ran GPU code from `/tmp`. Static linking is not needed for 1.0. |
| Does the reference validate requests cleanly? | No. Ten recorded requests made Python `mlx_lm.server` drop the connection or answer 404. mlx-lm-go answers those with 400 and a JSON error. |
| What does `/v1/models` list? | Python lists every MLX model in the Hugging Face cache and loads on demand. mlx-lm-go serves one model and lists that one. |
| Does the template raise on bad input? | Qwen3.5 does (`No user query found in messages.`, `System message must be at the beginning.`). Qwen2.5 renders the same inputs. |

## Moved to later milestones

- Spike 2 (Jinja fidelity) is the first tasks of `2026-11-21-m2-chat-template.md`, now with 38 golden rows.
- Spike 3 (split regex) is Task 2 of `2026-11-21-m2-tokenizer.md`, now with 1,000 golden rows.
