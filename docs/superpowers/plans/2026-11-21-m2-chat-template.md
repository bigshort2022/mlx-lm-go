# M2b: chat template

> **For agentic workers:** implement task by task with red/green TDD. For every test: write it, run it, see it fail for the stated reason, then write the smallest code that passes. Steps use checkbox (`- [ ]`) syntax.

**Goal:** a pure Go package `internal/chattemplate` that renders a model's Jinja chat template to exactly the same string as Python `tokenizer.apply_chat_template` for the Qwen2.5 and Qwen3.5 templates.

**Architecture:** load the template text and the tokenizer's special-token strings from the model directory. Decode request JSON into values that keep object key order. Render with a Jinja engine configured like Hugging Face's (`trim_blocks` and `lstrip_blocks` on) plus the helpers Hugging Face adds.

**Tech stack:** Go 1.27. `github.com/nikolalohinski/gonja/v2` v2.9.1 is in `go.mod` as the first choice of engine.

---

## Starting point

- Golden fixtures are in `testdata/qwen2/template.jsonl` and `testdata/qwen3_5/template.jsonl`, 19 rows each. A row has `name`, `messages`, `tools` (may be null), `kwargs`, `add_generation_prompt`, and either `prompt` (the expected string) or `error` (Python raised). `tools/oracle/make_text_fixtures.py` shows how they were made.
- Template sources, found with `hub.Resolve` from `internal/hub`:
  - `qwen2`: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, template in `tokenizer_config.json` under `chat_template`.
  - `qwen3_5`: `mlx-community/Qwen3.5-9B-MLX-4bit`, template in `chat_template.jinja`.
  Read both templates in full first. The Qwen3.5 one uses macros, `namespace`, `loop.previtem`, `loop.nextitem`, `raise_exception`, `is mapping`, string methods such as `.startswith` and `.split`, and reversed slices.
- How Hugging Face renders: read `apply_chat_template` and `render_jinja_template` in `/Users/b33j_02/sandbox/mlx/.venv/lib/python3.13/site-packages/transformers/utils/chat_template_utils.py`. Note the environment options, the `tojson` override, `raise_exception` and `strftime_now`.

## Files you own

Create: `internal/chattemplate/*.go` and `_test.go` files.
Do not touch anything else. In particular do not edit `go.mod`, `go.sum` or `internal/deps/deps.go`; if you stop using gonja, say so in your report and the integrator will remove it.

## Contract

```go
package chattemplate

type Template struct{ /* unexported, immutable after Load */ }

// Load reads the chat template from dir: chat_template.jinja if present,
// otherwise "chat_template" in tokenizer_config.json. Special-token strings
// (bos_token, eos_token, pad_token, unk_token) come from tokenizer_config.json
// and are available to the template under those names.
func Load(dir string) (*Template, error)

// Parse builds a Template from source text.
func Parse(source string, specialTokens map[string]string) (*Template, error)

type Input struct {
	Messages            json.RawMessage // JSON array, as received
	Tools               json.RawMessage // JSON array; nil or "null" when the request has no tools
	Kwargs              json.RawMessage // JSON object of extra template variables; may be nil
	AddGenerationPrompt bool
}

func (t *Template) Render(in Input) (string, error)

// TemplateError is returned when the template calls raise_exception.
type TemplateError struct{ Msg string }
func (e *TemplateError) Error() string
```

`Render` must be safe to call from many goroutines at once.

Behaviour the goldens enforce:

- Object key order from the request JSON is preserved everywhere, including through `tojson` and `|items`. Go maps lose order, so do not decode into `map[string]any`.
- `tojson` matches Python `json.dumps(value, ensure_ascii=False)`: separators `", "` and `": "`, no HTML escaping of `<`, `>` or `&`, non-ASCII written as is, `true`/`false`/`null`.
- Absent tools are the value none, so `{% if tools %}` is false.
- Each key of `Kwargs` becomes a top-level template variable. A variable that was not passed is undefined, so `enable_thinking is defined` is false.
- `raise_exception(msg)` makes `Render` return `*TemplateError` with that message.

## Tasks

Run tests with `go test ./internal/chattemplate/ -run <Name> -v`. Never run `go build ./...`, `go test ./...`, `go mod tidy` or `go get`.

### Task 1: the golden harness

- [ ] `TestRenderMatchesGolden` with one subtest per family and row. Decode each fixture line into a struct whose `messages`, `tools` and `kwargs` fields are `json.RawMessage`, so the bytes reach `Render` unchanged. For a row with `prompt`, compare byte for byte and print the first differing offset with 40 characters of context on failure. For a row with `error` starting `TemplateError: `, require `*TemplateError` with the rest as `Msg`. For any other `error` row, require a non-nil error.
- [ ] Watch it fail because `Load` does not exist.

### Task 2: plain conversations

Make these rows pass for both families, simplest first: `single_user`, `system_user`, `multi_turn`, `unicode_and_whitespace`, `empty_assistant_content`, `no_generation_prompt`.

- [ ] Add a focused unit test for each rule you had to implement that the goldens only cover indirectly, for example `TestTrimBlocksRemovesNewlineAfterBlockTag` and `TestUndefinedVariableIsFalsy`, using `Parse` with a three-line template.

### Task 3: template variables

- [ ] Rows `thinking_on`, `thinking_off`, `assistant_reasoning_content`, `assistant_inline_think`.
- [ ] `TestKwargsBecomeTopLevelVariables` with `Parse("{{ a }}-{{ b is defined }}", nil)`.

### Task 4: tools

- [ ] `TestToJSONMatchesPythonDumps`: a table of values, with the expected strings produced by Python beforehand: nested objects with keys out of alphabetical order, strings containing `<b>`, `&`, quotes, a newline, `é` and `天气`, numbers `1`, `1.5`, `1e-9`, `true`, `null`, an empty object and an empty array.
- [ ] Rows `tools_user`, `system_tools_user`, `tools_thinking_off`, `tool_call_and_result`, `parallel_tool_calls`.

### Task 5: errors

- [ ] Rows `no_messages`, `system_only`, `system_not_first`, `tool_result_first`. Qwen2.5 renders three of them; Qwen3.5 raises. Follow the golden for each family.
- [ ] `TestRaiseExceptionReturnsTemplateError` with `Parse`.
- [ ] `TestParseRejectsMalformedTemplate`: `Parse("{% if %}", nil)` returns an error.

### Task 6: concurrency

- [ ] `TestRenderIsSafeForConcurrentUse`: 16 goroutines render the golden rows; run with `-race`.

## Choosing the engine

Start with gonja and its extension points (custom filters, tests, globals and value types). If a golden row cannot pass because of an engine limit you cannot work around there, replace gonja with a small interpreter in this package that covers exactly the Jinja constructs the two templates use. Do not write per-model renderers in Go: a new model family must work by loading its template. Record which rows drove the decision.

## Done when

`go test -race ./internal/chattemplate/ && go vet ./internal/chattemplate/` passes with all 38 golden rows checked and none skipped on this machine, and `gofmt -l internal/chattemplate` prints nothing.

## Report

For each task: the test names, the failure you saw before implementing (one line each), and the final pass. State which engine you ended with and why. Do not commit.
