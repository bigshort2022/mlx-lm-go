# chattemplate test data

Everything here is written by `tools/oracle/make_template_differential.py`
from the Python reference. Do not edit it by hand; change the script and run
it again.

## Vendored chat templates

`<family>/chat_template.jinja` is the model's chat template, byte for byte,
and `<family>/tokenizer_config.json` holds the special tokens a template can
read. The tests render with these copies, so they need no model download.

| Directory | Repository | Snapshot |
|---|---|---|
| `qwen2/` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | `8b403126fc14f14cfc99bb4cfa72ecbc129ea677` |
| `qwen3_5/` | `mlx-community/Qwen3.5-9B-MLX-4bit` | `938d8919941c6e7efd3c7150eff7fe9d12afa631` |

## Fixtures

- `differential_<family>.jsonl`: requests beyond the golden rows in
  `testdata/<family>/template.jsonl` at the repository root, with what
  `tokenizer.apply_chat_template` returned or raised.
- `semantics.jsonl`: small templates, one Jinja rule each, with what the
  environment Hugging Face compiles chat templates in rendered or raised.

Reference versions: Python 3.13.13, transformers 5.7.0, jinja2 3.1.6.
