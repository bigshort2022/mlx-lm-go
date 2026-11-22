"""Generate model and sampler golden fixtures from Python mlx-lm.

Run from the repository root with a Python whose MLX version equals the one the
Go binary links (`mlx-lm-go version` prints it). GPU kernels change between MLX
releases, so numeric goldens from another version do not match. With Homebrew:

    /opt/homebrew/opt/mlx-lm/libexec/bin/python tools/oracle/make_model_fixtures.py

Writes:
  testdata/qwen2/generation.json      prompts, their token ids, step-0 top-50 logits,
                                      and the greedy continuation of each prompt
  testdata/qwen2/layers.safetensors   intermediate activations for one short input
  testdata/sampler/cases.json         sampler and logits-processor parameters
  testdata/sampler/cases.safetensors  their inputs and expected outputs

The Go tests treat these files as the definition of correct behaviour.
"""

import json
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import (
    apply_min_p,
    apply_top_k,
    apply_top_p,
    make_frequency_penalty,
    make_logits_processors,
    make_presence_penalty,
    make_repetition_penalty,
)

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
ROOT = Path(__file__).resolve().parents[2]
GREEDY_TOKENS = 64

CONVERSATIONS = [
    [{"role": "user", "content": "Count from 1 to 5."}],
    [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [{"role": "user", "content": "Write a haiku about the sea."}],
    [{"role": "user", "content": "用一句话介绍苹果公司。"}],
    [
        {"role": "user", "content": "Write a Go function that reverses a string."},
    ],
]


def f32(a):
    return a.astype(mx.float32)


def model_fixtures():
    model, tokenizer = load(MODEL)
    eos_ids = sorted(tokenizer.eos_token_ids)
    out = ROOT / "testdata" / "qwen2"

    cases = []
    for messages in CONVERSATIONS:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            ids = list(ids["input_ids"])

        logits = model(mx.array(ids)[None])[0, -1, :]
        logits = f32(logits)
        order = mx.argsort(-logits)[:50]
        top_ids = order.tolist()
        top_values = logits[order].tolist()

        mx.reset_peak_memory()
        start = time.perf_counter()
        greedy = []
        for token, _ in generate_step(mx.array(ids), model, max_tokens=GREEDY_TOKENS):
            greedy.append(int(token))
            if int(token) in eos_ids:
                break
        elapsed = time.perf_counter() - start

        cases.append(
            {
                "messages": messages,
                "prompt_ids": ids,
                "top50_ids": top_ids,
                "top50_logits": top_values,
                "greedy_ids": greedy,
                "greedy_text": tokenizer.decode(greedy),
                "peak_memory_bytes": mx.get_peak_memory(),
                "seconds": round(elapsed, 3),
            }
        )
        print(f"{len(ids)} prompt tokens -> {len(greedy)} greedy tokens: {tokenizer.decode(greedy)[:60]!r}")

    meta = {
        "model": MODEL,
        "mlx_lm": version("mlx-lm"),
        "mlx": mx.__version__,
        "eos_token_ids": eos_ids,
        "greedy_max_tokens": GREEDY_TOKENS,
        "note": "greedy_ids ends at the first end-of-sequence token, which is included",
    }
    (out / "generation.json").write_text(
        json.dumps({"meta": meta, "cases": cases}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Intermediate activations for one short input, to localise a wrong layer.
    ids = cases[0]["prompt_ids"][:8]
    x = mx.array(ids)[None]
    inner = model.model
    h0 = inner.embed_tokens(x)
    mask = create_attention_mask(h0, None)
    layer0 = inner.layers[0]
    n1 = layer0.input_layernorm(h0)
    attn = layer0.self_attn(n1, mask, None)
    h_mid = h0 + attn
    n2 = layer0.post_attention_layernorm(h_mid)
    mlp = layer0.mlp(n2)
    l0 = h_mid + mlp
    final = inner(x)
    logits = model(x)
    # The same eight tokens through a cache, seven then one. Float16 round-off
    # differs between the L=1 and L=8 kernel paths, so this is not logits_last.
    prompt_cache = make_prompt_cache(model)
    model(x[:, :7], cache=prompt_cache)
    incremental = model(x[:, 7:], cache=prompt_cache)
    mx.save_safetensors(
        str(out / "layers.safetensors"),
        {
            "input_ids": x.astype(mx.int32),
            "embed": f32(h0),
            "l0_norm1": f32(n1),
            "l0_attn": f32(attn),
            "l0_norm2": f32(n2),
            "l0_mlp": f32(mlp),
            "l0_out": f32(l0),
            "final_norm": f32(final),
            "logits_last": f32(logits[:, -1, :]),
            "incremental_7_1_logits_last": f32(incremental[:, -1, :]),
        },
        metadata={"model": MODEL, "mask": str(mask)},
    )
    print("layers.safetensors written; attention mask was", repr(mask))


def sampler_fixtures():
    out = ROOT / "testdata" / "sampler"
    out.mkdir(parents=True, exist_ok=True)

    mx.random.seed(20261121)
    logits = mx.random.normal((1, 32)) * 3.0
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    history = mx.array([3, 7, 7, 7, 11, 3, 19, 30, 30, 2, 2, 2, 2, 5])

    arrays = {"logits": logits, "logprobs": logprobs, "history": history.astype(mx.int32)}
    cases = []

    def add(name, kind, params, value):
        arrays[name] = f32(value)
        cases.append({"name": name, "kind": kind, **params})

    for k in (1, 5, 31):
        add(f"top_k_{k}", "top_k", {"top_k": k}, apply_top_k(logprobs, k))
    for p in (0.1, 0.5, 0.9, 0.999):
        add(f"top_p_{p}", "top_p", {"top_p": p}, apply_top_p(logprobs, p))
    # min_tokens_to_keep stays 1, as the server always passes it; larger values
    # raise a TypeError inside mlx-lm 0.31.3.
    for p in (0.05, 0.3, 0.9):
        add(f"min_p_{p}", "min_p", {"min_p": p}, apply_min_p(logprobs, p))
    for penalty, ctx in ((1.1, 20), (1.5, 4), (0.8, 20)):
        add(
            f"repetition_{penalty}_{ctx}",
            "repetition_penalty",
            {"penalty": penalty, "context_size": ctx},
            # The reference processors write into their argument, so each case gets a copy.
            make_repetition_penalty(penalty, ctx)(history, logits + 0),
        )
    for penalty, ctx in ((0.5, 20), (1.5, 4)):
        add(
            f"presence_{penalty}_{ctx}",
            "presence_penalty",
            {"penalty": penalty, "context_size": ctx},
            make_presence_penalty(penalty, ctx)(history, logits + 0),
        )
    for penalty, ctx in ((0.5, 20), (1.5, 4)):
        add(
            f"frequency_{penalty}_{ctx}",
            "frequency_penalty",
            {"penalty": penalty, "context_size": ctx},
            make_frequency_penalty(penalty, ctx)(history, logits + 0),
        )
    bias = {"0": -100.0, "7": 2.5, "31": 10.0}
    processors = make_logits_processors(logit_bias={int(k): v for k, v in bias.items()})
    biased = logits + 0
    for processor in processors:
        biased = processor(history, biased)
    add("logit_bias", "logit_bias", {"logit_bias": bias}, biased)

    mx.save_safetensors(str(out / "cases.safetensors"), arrays)
    meta = {"mlx_lm": version("mlx-lm"), "mlx": mx.__version__, "vocab": 32}
    (out / "cases.json").write_text(json.dumps({"meta": meta, "cases": cases}, indent=1) + "\n", encoding="utf-8")
    print(f"{len(cases)} sampler cases written")


if __name__ == "__main__":
    model_fixtures()
    sampler_fixtures()
