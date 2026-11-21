"""Record Qwen3.8 text fixtures without loading model weights.

From the repository root, with the pinned snapshot already cached:

    HF_HUB_OFFLINE=1 /opt/homebrew/opt/mlx-lm/libexec/bin/python tools/oracle/make_qwen38_text_fixtures.py

The output lives under internal/chattemplate/testdata/qwen38; the tokenizer
integration test also reads these JSON Lines so the prompts cannot drift.
"""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE
from jinja2 import TemplateError
from transformers import AutoTokenizer

from make_text_fixtures import ROOT, render, write_jsonl

REPO = "mlx-community/Qwen3.8-27B-4bit"
REVISION = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"


def cases():
    single = [{"role": "user", "content": "Say hi."}]
    history = [
        {"role": "system", "content": "Be succinct."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "  Add two pairs.  "},
        {"role": "user", "content": "And 4+4?"},
        {"role": "assistant", "content": "8", "reasoning_content": "Double four."},
    ]
    unicode = [{"role": "user", "content": "Cafe\u0301 中文 😀 👩🏽‍💻 <think>literal</think> <|im_end|> <|im_unknown|>"}]
    yield "thinking_default", single, {}, True
    yield "thinking_disabled", single, {"enable_thinking": False}, True
    yield "effort_xhigh", single, {"reasoning_effort": "xhigh"}, True
    yield "effort_medium", single, {"reasoning_effort": "medium"}, True
    yield "effort_low", single, {"reasoning_effort": "low"}, True
    yield "preserve_thinking_default", history, {}, True
    yield "preserve_thinking_false", history, {"preserve_thinking": False}, True
    yield "unicode_and_literal_markers", unicode, {}, True
    yield "invalid_effort", single, {"reasoning_effort": "high"}, False
    yield "null_effort_is_not_undefined", single, {"reasoning_effort": None}, False
    yield "empty_effort_is_not_undefined", single, {"reasoning_effort": ""}, False
    yield "disabled_ignores_invalid_effort", single, {"enable_thinking": False, "reasoning_effort": "high"}, True


def main():
    snapshot = Path(HF_HUB_CACHE) / f"models--{REPO.replace('/', '--')}" / "snapshots" / REVISION
    source = (snapshot / "chat_template.jinja").read_bytes()
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if tok.chat_template.encode("utf-8") != source:
        raise ValueError("the loaded template differs from the pinned chat_template.jinja")

    rows = []
    for name, messages, kwargs, should_render in cases():
        row = dict(name=name, messages=messages, tools=None, kwargs=kwargs, add_generation_prompt=True)
        try:
            prompt = render(tok, row, tokenize=False)
        except TemplateError as err:
            if should_render:
                raise
            row["error"] = f"{type(err).__name__}: {err}"
        else:
            if not should_render:
                raise ValueError(f"{name}: expected the pinned template to reject this input")
            row["prompt"] = prompt
            row["prompt_ids"] = render(tok, row, tokenize=True)
        rows.append(row)

    out = ROOT / "internal" / "chattemplate" / "testdata" / "qwen38"
    out.mkdir(parents=True, exist_ok=True)
    (out / "chat_template.jinja").write_bytes(source)
    config = {key: getattr(tok, key) for key in ("bos_token", "eos_token", "pad_token", "unk_token")}
    (out / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    meta = {
        "repo": REPO,
        "snapshot": REVISION,
        "template_sha256": hashlib.sha256(source).hexdigest(),
        **{name: version(name) for name in ("mlx-lm", "transformers", "tokenizers", "jinja2")},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    write_jsonl(out / "template.jsonl", rows)
    print(f"{REPO}@{REVISION}: {len(rows)} cases, {sum('error' in row for row in rows)} rejections")


if __name__ == "__main__":
    main()
