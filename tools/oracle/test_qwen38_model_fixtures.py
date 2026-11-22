"""Weight-free behavioral tests for the numerical oracle recorder."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.oracle import make_qwen38_model_fixtures as recorder

QWEN35 = "mlx-community/Qwen3.5-9B-MLX-4bit"
REVISION = "938d8919941c6e7efd3c7150eff7fe9d12afa631"
VERSIONS = {"mlx": "0.32.4", "mlx_lm": "0.32.2"}


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.snapshots = Path(tmp.name) / "models--mlx-community--Qwen3.5-9B-MLX-4bit" / "snapshots"

    def snapshot(self, revision):
        path = self.snapshots / revision
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5_text"},
            "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        }))
        for file in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors"):
            (path / file).touch()
        return path

    def test_default_output_requires_pinned_revision(self):
        other = self.snapshot("a" * 40)
        with self.assertRaisesRegex(ValueError, "expected pinned snapshot.*" + REVISION):
            recorder.validate_snapshot(other, QWEN35, REVISION, exploration=False)

    def test_exploration_records_actual_revision(self):
        other = self.snapshot("a" * 40)
        self.assertEqual(recorder.validate_snapshot(other, QWEN35, REVISION, exploration=True), other.name)

    def test_missing_snapshot_is_not_a_hub_identifier(self):
        with self.assertRaisesRegex(ValueError, "missing snapshot directory"):
            recorder.validate_snapshot(self.snapshots / REVISION, QWEN35, REVISION, exploration=False)

    def test_partial_snapshot_reports_missing_weight_shard(self):
        path = self.snapshot(REVISION)
        (path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"a": "model.safetensors", "b": "model-00002.safetensors"},
        }))
        with self.assertRaisesRegex(ValueError, "missing model-00002.safetensors"):
            recorder.validate_snapshot(path, QWEN35, REVISION, exploration=False)

    def test_rejects_wrong_oracle_versions(self):
        for package in VERSIONS:
            with self.subTest(package=package):
                wrong = VERSIONS | {package: "0.0.1"}
                with self.assertRaisesRegex(ValueError, package + r".*0\.0\.1"):
                    recorder.validate_versions(wrong)


class FixtureValidationTests(unittest.TestCase):
    def setUp(self):
        run = {
            "top_ids": [42, 6], "top_logits": [10.0, 9.5], "greedy_ids": [42, 1],
            "greedy_margins": [0.5, 0.25], "stop_reason": "eos",
        }
        probe = {"output_prefix": [[0.5]], "layer_output_prefix": [[1.25]]}
        cases = []
        for name, text in (("english", "Reply yes."), ("chinese", "请回答。")):
            for thinking in (True, False):
                cases.append({
                    "name": f"{name}_{thinking}",
                    "messages": [{"role": "user", "content": text}],
                    "template_kwargs": {"add_generation_prompt": True, "enable_thinking": thinking},
                    "rendered_prompt": text + ("<think>" if thinking else "<think></think>"),
                    "prompt_ids": [20, 21, 22, 23] + ([24, 25] if thinking else [26, 27]),
                    "runs": [copy.deepcopy(run) | {"prefill_step_size": size} for size in (2048, 4)],
                })
        self.fixture = {
            "schema_version": 1, "model_id": QWEN35, "revision": REVISION,
            "versions": VERSIONS.copy(), "eos_token_ids": [1], "top_k": 2,
            "max_greedy_tokens": 3, "near_tie_threshold": 0.125, "cases": cases,
            "layer_probe": {
                "text": "The sky is blue.", "prompt_ids": [30], "width": 1,
                "delta": dict(probe, layer_index=0), "attention": dict(probe, layer_index=3),
            },
        }

    def validate(self):
        recorder.validate_fixture(self.fixture, QWEN35, REVISION)

    def test_accepts_consistent_fixture(self):
        self.validate()

    def test_rejects_empty_prompt_ids(self):
        self.fixture["cases"][0]["prompt_ids"] = []
        with self.assertRaisesRegex(ValueError, "prompt_ids"):
            self.validate()

    def test_rejects_nonfinite_logits(self):
        self.fixture["cases"][0]["runs"][0]["top_logits"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite top_logits"):
            self.validate()

    def test_rejects_collapsed_thinking_modes(self):
        first, second = self.fixture["cases"][:2]
        second.update(rendered_prompt=first["rendered_prompt"], prompt_ids=first["prompt_ids"])
        with self.assertRaisesRegex(ValueError, "thinking.*distinct"):
            self.validate()

    def test_rejects_wrong_fixture_metadata(self):
        self.fixture["revision"] = "a" * 40
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.validate()

    def test_rejects_tokens_after_eos(self):
        run = self.fixture["cases"][0]["runs"][0]
        run["greedy_ids"].append(9)
        run["greedy_margins"].append(0.125)
        with self.assertRaisesRegex(ValueError, "EOS must be the final token"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
