"""Record dense Qwen checkpoints with the pinned Python mlx-lm oracle.

From the repo root, use the pinned Hugging Face cache, or pass both paths:

    HF_HUB_OFFLINE=1 /opt/homebrew/opt/mlx-lm/libexec/bin/python \
        tools/oracle/make_qwen38_model_fixtures.py --qwen35 <snapshot> --qwen38 <snapshot>

The default output is testdata/qwen38. An alternate --out permits local
exploration with other revisions (or just one cached model) and is also used
to verify reproducibility without changing the goldens. Models load in
sequence. Output contains no weights, cache paths, or timing information.
"""

import argparse
import heapq
import json
import math
import os
import re
from contextlib import closing
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

VERSIONS = {"mlx": "0.32.4", "mlx_lm": "0.32.2"}
PREFILL_STEP_SIZES = (2048, 4)
PINNED_MODELS = {
    "qwen35": ("mlx-community/Qwen3.5-9B-MLX-4bit", "938d8919941c6e7efd3c7150eff7fe9d12afa631"),
    "qwen38": ("mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"),
}
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "testdata" / "qwen38"
CONVERSATIONS = (
    ("english", "Reply with exactly one word: yes."),
    ("chinese", "请只回答一个字：好。"),
)
LAYER_PROMPT = "The sky is blue. Clouds are white."
LAYER_WIDTH = 8
TOP_K = 10
MAX_GREEDY_TOKENS = 6
# Raw BF16 logits can be separated by only one representable step. Measure
# the actual sampler-space gap rather than inferring it from the raw logits.
NEAR_TIE_THRESHOLD = 0.125


def require(condition, message):
    if not condition:
        raise ValueError(message)


def repo_dir(model_id):
    return "models--" + model_id.replace("/", "--")


def validate_snapshot(path, model_id, revision, *, exploration):
    require(path.is_dir(), f"missing snapshot directory: {path}")
    in_cache = path.parent.name == "snapshots" and path.parent.parent.name == repo_dir(model_id)
    if not exploration:
        require(in_cache and path.name == revision, f"expected pinned snapshot {model_id}@{revision}, got {path}")

    for file in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        require((path / file).is_file(), f"{path}: missing {file}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    require(
        config.get("model_type") == "qwen3_5" and text.get("model_type") == "qwen3_5_text"
        and not text.get("num_experts"),
        f"{path}: expected a dense qwen3_5 text config",
    )
    quant = config.get("quantization", {})
    require(
        (quant.get("bits"), quant.get("group_size"), quant.get("mode", "affine")) == (4, 64, "affine"),
        f"{path}: expected 4-bit affine weights with group size 64",
    )
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        for shard in sorted(set(weight_map.values())):
            require((path / shard).is_file(), f"{path}: missing {shard}")
    require(any(file.is_file() for file in path.glob("model*.safetensors")), f"{path}: missing safetensors")
    return path.name if in_cache and re.fullmatch(r"[a-f0-9]{40}", path.name) else None


def validate_versions(versions):
    for package, expected in VERSIONS.items():
        actual = versions.get(package)
        require(actual == expected, f"{package}: expected {expected}, got {actual}")


def token_ids(ids):
    return bool(ids) and all(type(token) is int and token >= 0 for token in ids)


def validate_fixture(fixture, model_id, revision):
    require(
        (fixture["schema_version"], fixture["model_id"], fixture["revision"], fixture["versions"])
        == (1, model_id, revision, VERSIONS),
        "fixture metadata does not match the selected oracle",
    )
    eos = fixture["eos_token_ids"]
    require(token_ids(eos) and eos == sorted(set(eos)), "expected sorted distinct eos_token_ids")
    top_k, max_tokens = fixture["top_k"], fixture["max_greedy_tokens"]
    threshold = fixture["near_tie_threshold"]
    require(top_k >= 2 and max_tokens > 0 and math.isfinite(threshold) and threshold > 0, "invalid oracle settings")

    conversations = {}
    for case in fixture["cases"]:
        name = case["name"]
        require(token_ids(case["prompt_ids"]), f"{name}: expected nonempty prompt_ids")
        require(case["rendered_prompt"], f"{name}: expected a rendered_prompt")
        kwargs = case["template_kwargs"]
        thinking = kwargs["enable_thinking"]
        require(type(thinking) is bool and kwargs["add_generation_prompt"] is True, f"{name}: invalid template_kwargs")
        pair = conversations.setdefault(json.dumps(case["messages"], sort_keys=True), {})
        require(thinking not in pair, f"{name}: duplicate thinking mode")
        pair[thinking] = case["rendered_prompt"], case["prompt_ids"]
        require(
            tuple(run["prefill_step_size"] for run in case["runs"]) == PREFILL_STEP_SIZES,
            f"{name}: expected default and small prefill runs",
        )
        require(len(case["prompt_ids"]) - 1 > PREFILL_STEP_SIZES[1], f"{name}: prompt does not cross small prefill")
        for run in case["runs"]:
            ids, logits = run["top_ids"], run["top_logits"]
            require(token_ids(ids) and len(set(ids)) == len(ids) == top_k, f"{name}: invalid top_ids")
            require(len(logits) == top_k and all(map(math.isfinite, logits)), f"{name}: expected finite top_logits")
            scores = list(zip(ids, logits))
            require(scores == sorted(scores, key=lambda p: (-p[1], p[0])), f"{name}: unstable top logit ordering")

            greedy, margins = run["greedy_ids"], run["greedy_margins"]
            require(token_ids(greedy) and len(greedy) <= max_tokens, f"{name}: invalid greedy_ids")
            require(
                len(margins) == len(greedy) and all(math.isfinite(gap) and gap >= 0 for gap in margins),
                f"{name}: invalid greedy_margins",
            )
            require(not any(token in eos for token in greedy[:-1]), f"{name}: EOS must be the final token")
            stop = "eos" if greedy[-1] in eos else "length"
            require(
                run["stop_reason"] == stop and (stop == "eos" or len(greedy) == max_tokens),
                f"{name}: invalid stop_reason",
            )

    require(conversations, "no conversations recorded")
    for pair in conversations.values():
        require(
            len(pair) == 2 and all(a != b for a, b in zip(pair[True], pair[False])),
            "thinking on and off must have distinct renders and prompt_ids",
        )

    probe = fixture["layer_probe"]
    width = probe["width"]
    require(probe["text"] and token_ids(probe["prompt_ids"]) and width > 0, "invalid layer probe input")
    require(probe["delta"]["layer_index"] == 0 < probe["attention"]["layer_index"], "invalid probe layer indices")
    for unit in ("delta", "attention"):
        for field in ("output_prefix", "layer_output_prefix"):
            values = probe[unit][field]
            require(
                len(values) == len(probe["prompt_ids"])
                and all(len(row) == width and all(map(math.isfinite, row)) for row in values),
                f"layer probe {unit}: invalid {field}",
            )


class CapturePromptLogits:
    """Expose the raw last-prompt logits without changing generate_step."""

    def __init__(self, model, prompt_length):
        self.model = model
        self.remaining = prompt_length
        self.last = None

    def make_cache(self):
        return self.model.make_cache()

    def __call__(self, inputs, cache):
        logits = self.model(inputs, cache=cache)
        if self.remaining > 0:
            self.remaining -= inputs.shape[1]
            if self.remaining == 0:
                self.last = logits[0, -1, :]
        return logits


def record_run(model, prompt_ids, eos, prefill_step_size):
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    greedy_ids, margins = [], []
    capture = CapturePromptLogits(model, len(prompt_ids))
    with closing(generate_step(
        mx.array(prompt_ids), capture, max_tokens=MAX_GREEDY_TOKENS,
        prefill_step_size=prefill_step_size,
    )) as steps:
        for token, logprobs in steps:
            greedy_ids.append(token)
            a, b = mx.topk(logprobs.astype(mx.float32), 2).tolist()
            margins.append(abs(a - b))
            if token in eos:
                break

    raw = capture.last.astype(mx.float32).tolist()
    require(all(map(math.isfinite, raw)), "nonfinite raw prompt logits")
    # Resolve tied scores by vocabulary id, even when a tie crosses TOP_K.
    top = heapq.nlargest(TOP_K, enumerate(raw), key=lambda p: (p[1], -p[0]))
    return {
        "prefill_step_size": prefill_step_size,
        "top_ids": [i for i, _ in top],
        "top_logits": [score for _, score in top],
        "greedy_ids": greedy_ids,
        "greedy_margins": margins,
        "stop_reason": "eos" if greedy_ids[-1] in eos else "length",
    }


def record_layer_probe(model, tokenizer):
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    ids = tokenizer.encode(LAYER_PROMPT, add_special_tokens=False)
    hidden = model.model.embed_tokens(mx.array([ids]))
    mask = create_attention_mask(hidden)
    probe = {"text": LAYER_PROMPT, "prompt_ids": ids, "width": LAYER_WIDTH}

    def prefix(array):
        return array[0, :, :LAYER_WIDTH].astype(mx.float32).tolist()

    # Follow the reference decoder on one whole sequence (no caches). Capture
    # the real submodules' outputs before the residual and after the MLP.
    for i, layer in enumerate(model.layers):
        if layer.is_linear and i != 0:
            hidden = layer(hidden)
            continue
        normed = layer.input_layernorm(hidden)
        unit = "delta" if layer.is_linear else "attention"
        result = layer.linear_attn(normed) if layer.is_linear else layer.self_attn(normed, mask)
        residual = hidden + result
        hidden = residual + layer.mlp(layer.post_attention_layernorm(residual))
        probe[unit] = {
            "layer_index": i,
            "output_prefix": prefix(result),
            "layer_output_prefix": prefix(hidden),
        }
        if unit == "attention":
            break
    return probe


def record_checkpoint(path, model_id, revision, versions):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generation_stream

    model, tokenizer = load(
        str(path), tokenizer_config={"local_files_only": True, "trust_remote_code": False},
        trust_remote_code=False,
    )
    eos = sorted(tokenizer.eos_token_ids)
    cases = []
    for language, prompt in CONVERSATIONS:
        messages = [{"role": "user", "content": prompt}]
        for thinking in (True, False):
            kwargs = {"add_generation_prompt": True, "enable_thinking": thinking}
            name = f"{language}_thinking_{'on' if thinking else 'off'}"
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
            ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
            cases.append({
                "name": name,
                "messages": messages,
                "template_kwargs": kwargs,
                "rendered_prompt": rendered,
                "prompt_ids": ids,
                "runs": [record_run(model, ids, eos, size) for size in PREFILL_STEP_SIZES],
            })
    fixture = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision,
        "versions": versions,
        "eos_token_ids": eos,
        "top_k": TOP_K,
        "max_greedy_tokens": MAX_GREEDY_TOKENS,
        "near_tie_threshold": NEAR_TIE_THRESHOLD,
        "cases": cases,
        "layer_probe": record_layer_probe(model, tokenizer),
    }
    # generate_step evaluates one token ahead asynchronously. Finish that
    # work before this call releases the model and the next checkpoint loads.
    mx.synchronize(generation_stream)
    validate_fixture(fixture, model_id, revision)
    return fixture


def default_cache():
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    hf_home = Path(os.environ.get("HF_HOME", cache_home / "huggingface"))
    return Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub")).expanduser()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for alias in PINNED_MODELS:
        parser.add_argument(f"--{alias}", type=Path, help="local snapshot directory (defaults to the pinned cache path)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="fixture output directory")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    out = args.out.expanduser().resolve()
    exploration = out != DEFAULT_OUT.resolve()
    cache = default_cache().resolve()
    snapshots = []
    try:
        versions = {"mlx": version("mlx"), "mlx_lm": version("mlx-lm")}
        validate_versions(versions)
        require(not out.is_relative_to(cache), "--out must be outside the model cache")
        for alias, (model_id, pinned_revision) in PINNED_MODELS.items():
            supplied = getattr(args, alias)
            path = (supplied or cache / repo_dir(model_id) / "snapshots" / pinned_revision).expanduser().resolve()
            if exploration and supplied is None and not path.exists():
                continue
            require(not out.is_relative_to(path), "--out must be outside the input snapshots")
            revision = validate_snapshot(path, model_id, pinned_revision, exploration=exploration)
            snapshots.append((alias, path, model_id, revision))
        require(snapshots, "no local snapshots to record")
    except (ValueError, OSError, PackageNotFoundError) as error:
        parser.error(str(error))

    import mlx.core as mx

    out.mkdir(parents=True, exist_ok=True)
    for alias, path, model_id, revision in snapshots:
        fixture = record_checkpoint(path, model_id, revision, versions)
        mx.clear_cache()
        target = out / f"{alias}.json"
        target.write_text(json.dumps(fixture, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Recorded {model_id}@{revision} -> {target}", flush=True)


if __name__ == "__main__":
    main()
