# M2a: tokenizer

> **For agentic workers:** implement task by task with red/green TDD. For every test: write it, run it, see it fail for the stated reason, then write the smallest code that passes. Steps use checkbox (`- [ ]`) syntax.

**Goal:** a pure Go package `internal/tokenizer` that reads a Hugging Face `tokenizer.json` and encodes, decodes and stream-decodes exactly like the Python `tokenizers` library for the Qwen2.5 and Qwen3.5 tokenizers.

**Architecture:** load `tokenizer.json` once into immutable tables. Encoding is: split out added tokens, NFC-normalise the rest, split with the file's own regex, map bytes to the byte-level alphabet, apply BPE merges by rank. Decoding reverses the byte mapping and replaces invalid UTF-8 the way Python does.

**Tech stack:** Go 1.27, `github.com/dlclark/regexp2` (the split regex uses a lookahead that Go's `regexp` rejects), `golang.org/x/text/unicode/norm`. Both are already in `go.mod`.

---

## Starting point

- Golden fixtures made by the Python reference are in `testdata/qwen2/` and `testdata/qwen3_5/`: `meta.json`, `pretokenize.jsonl`, `tokenize.jsonl`, `decode.jsonl`. They define correct behaviour. `tools/oracle/make_text_fixtures.py` shows how each was produced.
- The real tokenizer files are in the Hugging Face cache. Find them with `hub.Resolve` from `internal/hub`:
  - `qwen2`: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
  - `qwen3_5`: `mlx-community/Qwen3.5-9B-MLX-4bit`
  A test that needs them calls `t.Skip` when `hub.Resolve` fails. On this machine both resolve, so no golden test may be skipped here.
- Each directory has `tokenizer.json` and `tokenizer_config.json`. Read them before designing: `model` (BPE vocab and merges), `added_tokens`, `normalizer`, `pre_tokenizer`, `decoder`.
- The Python library, for behaviour questions: `/Users/b33j_02/sandbox/mlx/.venv/bin/python` has `tokenizers` 0.22.2 and `transformers` 5.7.0.

## Files you own

Create: `internal/tokenizer/{tokenizer,load,added,pretokenize,bpe,decode,stream}.go`, `_test.go` files beside them, and `internal/tokenizer/testdata/tiny/tokenizer.json` (a hand-made vocabulary of about 20 tokens for tests that must run without the cache).
Do not touch anything else, including `go.mod`, `go.sum` and `testdata/` at the repository root.

## Contract

```go
package tokenizer

type Tokenizer struct{ /* unexported, immutable after Load */ }

// Load reads tokenizer.json and tokenizer_config.json from dir.
func Load(dir string) (*Tokenizer, error)

// Encode converts text to token ids. Added and special tokens that appear
// literally in the text, such as "<|im_start|>", become their single id.
func (t *Tokenizer) Encode(text string) []int32

// Decode converts ids to text, including special tokens. Invalid UTF-8 becomes U+FFFD.
func (t *Tokenizer) Decode(ids []int32) string

func (t *Tokenizer) VocabSize() int                      // base vocabulary plus added tokens
func (t *Tokenizer) TokenID(token string) (int32, bool)
func (t *Tokenizer) EOSToken() (token string, id int32, ok bool) // from tokenizer_config.json "eos_token"

// NewStream returns a streaming decoder for one generation.
func (t *Tokenizer) NewStream() *Stream

type Stream struct{ /* unexported */ }
// Add consumes one id and returns all text that is now complete. Only an
// unfinished trailing character is held back.
func (s *Stream) Add(id int32) string
// Flush returns whatever is still held back, with invalid bytes as U+FFFD.
func (s *Stream) Flush() string
```

`Encode`, `Decode`, `TokenID` and `NewStream` must be safe to call from many goroutines at once. Each `Stream` is used by one goroutine.

## Tasks

Run tests with `go test ./internal/tokenizer/ -run <Name> -v`. Never run `go build ./...`, `go test ./...`, `go mod tidy` or `go get`.

### Task 1: load

- [ ] `TestLoadTinyTokenizer`: `Load("testdata/tiny")` succeeds; `VocabSize` and two `TokenID` lookups match the file.
- [ ] `TestLoadMissingDirectoryFails`: error mentions `tokenizer.json`.
- [ ] `TestLoadMatchesMeta` (both families): `VocabSize()` equals `meta.json` `vocab_size`; `EOSToken()` equals `special_tokens.eos_token` and `eos_token_id`.

### Task 2: normalise and split

- [ ] `TestPreTokenizeMatchesGolden` (both families): for every row of `pretokenize.jsonl`, normalising `text` gives `normalized` and splitting gives `pieces`, exactly. Use the regex from `tokenizer.json`, not a copy in Go source. Expose the two steps through unexported functions the test can call.

If regexp2 cannot reproduce a row, find out why against the Python library before changing approach. A hand-written scanner is the fallback; say so in your report.

### Task 3: added tokens

- [ ] `TestEncodeSplitsAddedTokens` on the tiny tokenizer: text containing an added token yields that token's id at the right position; a partial added token (`"<|im_start"`) is encoded as ordinary text.
- [ ] Added tokens are matched before normalisation and are not themselves normalised or split.

### Task 4: BPE encode

- [ ] `TestEncodeTinyMergesByRank`: a word whose merges must be applied in rank order, not left to right, encodes correctly.
- [ ] `TestEncodeMatchesGolden` (both families): every row of `tokenize.jsonl` matches exactly. Report the mismatch count and the first differing row on failure.
- [ ] `TestEncodeIsSafeForConcurrentUse`: 16 goroutines encode the golden rows; run with `-race`.

### Task 5: decode

- [ ] `TestDecodeMatchesGolden` (both families): every row of `decode.jsonl` matches exactly. Some rows cut a multi-byte character; Python's output for them contains U+FFFD and yours must be identical.

### Task 6: streaming decode

- [ ] `TestStreamConcatenationEqualsDecode` (both families): for every `decode.jsonl` row, the concatenation of all `Add` results plus `Flush` equals `Decode(ids)`.
- [ ] `TestStreamNeverEmitsPartialCharacters`: for rows whose golden text has no U+FFFD, no single `Add` result contains U+FFFD and every `Add` result is valid UTF-8.
- [ ] `TestStreamHoldsBackSplitEmoji`: encode `"😀"`; if it is more than one token, every `Add` before the last returns `""`.

## Done when

`go test -race ./internal/tokenizer/ && go vet ./internal/tokenizer/` passes with zero skipped golden tests on this machine, and `gofmt -l internal/tokenizer` prints nothing.

## Report

For each task: the test names, the failure you saw before implementing (one line each), and the final pass. State whether regexp2 was kept. Do not commit.
