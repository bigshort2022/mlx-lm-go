# mlx-lm-go

A native Go server for running language models with [MLX](https://github.com/ml-explore/mlx) on Apple silicon. It implements a subset of the OpenAI-compatible API exposed by Python's `mlx_lm.server`, loads supported `mlx-community` model directories, and ships as one binary with a chat page built in. There is no Python at run time.

## Status

This is early software. What works today:

- Models of the `qwen2` family in 4-bit or dense form, for example `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.
- `POST /v1/chat/completions` and `POST /v1/completions`, streamed or not, with stop strings, sampling controls, penalties and logit bias.
- `GET /v1/models`, `GET /health`, and a chat page at `/`.

The Qwen2.5 test fixtures check greedy decoding against mlx-lm 0.32.2 on MLX 0.32.4. End-to-end performance parity has not been established.

Not there yet:

- Other architectures. `qwen3_5` is next.
- A prompt cache. Every request processes its whole prompt, so the first token of a follow-up turn arrives later than with Python.
- Tool calls, `logprobs`, adapters, speculative decoding, batching, distributed inference, and model downloads.
- Linux and CUDA.

Known unsupported options such as `tools`, `logprobs`, and `--draft-model` return errors when requested. Unrecognized JSON fields are ignored.

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

Generate once from the command line:

```sh
bin/mlx-lm-go generate --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --prompt "Count from 1 to 5."
```

`bin/mlx-lm-go server --help` lists supported flags and defaults, plus the Python server flags that this binary explicitly rejects.

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

Tests compare this code with recordings of the Python implementation under `testdata/`: tokenizer output, rendered chat templates, layer activations, logits, sampled tokens and HTTP exchanges. `make fixtures` regenerates them; see the comments in the `Makefile` for the Python environments it expects. Numeric fixtures must come from the same MLX version the binary links, because GPU kernels change between releases.

The design and the plans the code was built from are in `docs/superpowers/plans/`.

## Licence

MIT. See `LICENSE`.
