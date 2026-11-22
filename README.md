# mlx-lm-go

A native Go server for running language models with [MLX](https://github.com/ml-explore/mlx) on Apple silicon. It implements a subset of the OpenAI-compatible API exposed by Python's `mlx_lm.server`, loads supported `mlx-community` model directories, and ships as one binary with a chat page built in. There is no Python at run time.

## Status

This is early software. What works today:

- Models of the `qwen2` family in 4-bit or dense form, for example `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.
- Text inference for the dense `qwen3_5` family. Tested 4-bit checkpoints: [Qwen3.8-27B](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) (quality) and [Qwen3.5-9B](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit) (smaller and faster).
- `POST /v1/chat/completions` and `POST /v1/completions`, streamed or not, with stop strings, sampling controls, penalties and logit bias.
- Function calling for Qwen3.8 and Qwen3.5, with streamed and buffered replies and tool results in chat history.
- Opt-in prompt prefix caching for Qwen2.5, Qwen3.5 and Qwen3.8.
- `GET /v1/models`, `GET /health`, and a chat page at `/`.

Pinned test cases compare Qwen2.5, Qwen3.5 and Qwen3.8 with mlx-lm 0.32.2 on MLX 0.32.4. See the [Qwen3.8 plan](docs/superpowers/plans/2026-11-21-m6a-qwen38.md) for the tested revisions and results.

Not there yet:

- Sparse `qwen3_5_moe`, other architectures, and image or video input.
- `logprobs`, adapters, speculative decoding, batching, distributed inference, and model downloads.
- Linux and CUDA.

Known unsupported options such as `logprobs` and `--draft-model` return errors when requested. Unrecognized JSON fields are ignored.

## Install

You need a Mac with Apple silicon, Go 1.27 or later, and the MLX C library. From a checkout of this repository:

```sh
brew install go mlx-c
make build
```

The binary is `bin/mlx-lm-go`. It links Homebrew's `libmlxc` dynamically, so `mlx-c` must stay installed.

Models are read from a directory or from the local Hugging Face cache. The server does not download them:

```sh
hf download mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

## Use

Start a server:

```sh
bin/mlx-lm-go server --model mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

Then open <http://127.0.0.1:8080/> for the chat page, or call the API:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Count from 1 to 5."}]}'
```

For clients using the chat or text completions API, set `base_url` to `http://127.0.0.1:8080/v1`.

Run the flagship checkpoint at the tested revision. `hf download` prints its local snapshot directory:

```sh
snapshot=$(hf download mlx-community/Qwen3.8-27B-4bit --quiet \
  --revision 10c35caafbb80f7dc6a7a432cdd11af10a6d4818)
bin/mlx-lm-go server --model "$snapshot" \
  --chat-template-args '{"enable_thinking":false}'
```

Omit `--chat-template-args` to use the checkpoint's default. The chat API sends reasoning separately as `message.reasoning` or `delta.reasoning`, not in `content`. A chat request can override the server's template settings with `chat_template_kwargs`.

Generate once from the command line:

```sh
bin/mlx-lm-go generate --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --prompt "Count from 1 to 5."
```

`bin/mlx-lm-go server --help` lists supported flags and defaults, plus the Python server flags that this binary explicitly rejects.

## Use with OMP

The server converts Qwen's tool syntax to OpenAI `tool_calls`. It does not run functions or commands. OMP executes the tools enabled for its session and sends their results back to the model.

Start a local server for Qwen3.8:

```sh
bin/mlx-lm-go server --host 127.0.0.1 --port 8082 \
  --model mlx-community/Qwen3.8-27B-4bit --max-tokens 4096 \
  --prompt-cache-size 4 --prompt-cache-bytes 2GB \
  --chat-template-args '{"enable_thinking":false}'
```

The prompt cache keeps up to four recent exact token-prefix snapshots in memory, with a 2 GB budget for retained tensors and token IDs in this example. Model weights, active inference, and allocator overhead are separate. The cache reuses attention and recurrent state across requests, including tool-result turns, but never reuses sampled output. Cold or changed prefixes still need prefill. Usage reports reused tokens in `prompt_tokens_details.cached_tokens`. Leave `--prompt-cache-size` at its default of `0` to disable caching. Only enable it for trusted clients: cache hits and timing are visible to clients sharing the server.

Save this to `~/.omp/profiles/mlx-go-code/agent/models.yml`:

```yaml
providers:
  mlx-go:
    baseUrl: http://127.0.0.1:8082/v1
    api: openai-completions
    auth: none
    models:
      - id: mlx-community/Qwen3.8-27B-4bit
        input: [text]
        reasoning: false
        supportsTools: true
        tokenizer: qwen3
        contextWindow: 32768
        maxTokens: 4096
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        compat:
          supportsStore: false
          supportsDeveloperRole: false
          supportsReasoningEffort: false
          maxTokensField: max_tokens
```

From the project you want to work on:

```sh
omp --profile mlx-go-code --model mlx-go/mlx-community/Qwen3.8-27B-4bit \
  --offline --tools read,edit,write,grep,glob,bash --approval-mode always-ask
```

This approval mode lets the model read files but asks before writes or shell commands.

The chat API accepts `tool_choice: "auto"` (default) or `"none"`. It does not support required or named tool choice. With `parallel_tool_calls: false`, a response containing more than one call is rejected. Tools are not supported in the bundled chat page, on the text completions endpoint, or with Qwen2.5.

## Calls from web pages

The server is meant to be safe to leave running while you browse:

- It listens on `127.0.0.1` unless you pass `--host`.
- It refuses requests whose `Host` header is not its own address, which blocks DNS rebinding.
- It refuses requests from other web origins. `mlx_lm.server` allows every origin by default; this server allows none. To let a page call it, name the page's origin:

  ```sh
  bin/mlx-lm-go server --model <model> --allowed-origins https://unite4ai.com
  ```

- `POST` bodies must be `application/json`, so a cross-site form cannot start a generation.

The built-in chat page is served from the server's own origin and needs no flag. The same page is built to be hosted on a website, where the browser talks straight to the server on your machine; a page hosted that way needs the `--allowed-origins` flag above with its own origin.

There is no authentication. If you bind to another address with `--host`, anyone who can reach that address can use the model.

## Differences from `mlx_lm.server`

- `/v1/models` lists the one model being served. Python lists every cached model and loads on request; this server serves the model it started with, and a request naming another model is an error.
- Requests are served one at a time, in arrival order. At most 32 wait; further requests get `503`.
- Invalid requests get `400` with `{"error": "..."}`, using the wording of mlx-lm 0.32.2 where it has one.

## Development

```sh
make test       # every package; model tests need the Qwen2.5 model in the Hugging Face cache
make test-race
make vet
```

Tests compare against Python recordings under `testdata/`: tokenization, rendered prompts, layer activations, logits, sampled tokens and HTTP exchanges. Use `make fixtures` for the original fixtures and `tools/oracle/make_qwen38_model_fixtures.py` for the pinned Qwen3 fixtures. Run the Qwen3 tests below with both snapshots cached and the same MLX version used to record the goldens:

```sh
HF_HUB_OFFLINE=1 MLX_LM_GO_TEST_QWEN38=1 go test -p 1 \
  ./internal/models/qwen3_5 ./internal/generate ./cmd/mlx-lm-go \
  -run '^TestQwen(35|38)(DecoderParity|GenerationParity|Binary)$' -count=1
```

The design and the plans the code was built from are in `docs/superpowers/plans/`.

## Licence

MIT. See `LICENSE`.
