"""Record real HTTP exchanges from Python mlx_lm.server as golden fixtures.

Run from the repository root with a Python environment that has mlx-lm 0.32.2:

    /opt/homebrew/opt/mlx-lm/libexec/bin/python tools/oracle/record_server.py

It starts mlx_lm.server on a spare port with a small model from the Hugging
Face cache, replays a fixed list of requests, and writes one file per exchange
to testdata/server/<name>.json. The Go server tests compare status codes,
header names and JSON shapes against these files. Generated text, ids and
timestamps are not compared.
"""

import http.client
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
PORT = 18555
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "testdata" / "server"

USER = [{"role": "user", "content": "Count from 1 to 5."}]
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

JSON = {"Content-Type": "application/json"}

# (name, method, path, headers, body). A dict body is sent as JSON; a str body is sent raw.
CASES = [
    ("health", "GET", "/health", {}, None),
    ("models", "GET", "/v1/models", {}, None),
    ("get_unknown_path", "GET", "/v1/nope", {}, None),
    ("post_unknown_path", "POST", "/v1/nope", JSON, {"messages": USER}),
    ("options_chat", "OPTIONS", "/v1/chat/completions", {"Origin": "https://unite4ai.com", "Access-Control-Request-Method": "POST"}, None),
    ("chat", "POST", "/v1/chat/completions", JSON, {"messages": USER, "max_tokens": 24}),
    ("chat_with_origin", "POST", "/v1/chat/completions", {**JSON, "Origin": "https://unite4ai.com"}, {"messages": USER, "max_tokens": 4}),
    ("chat_model_field", "POST", "/v1/chat/completions", JSON, {"model": "default_model", "messages": USER, "max_tokens": 4}),
    ("chat_stream", "POST", "/v1/chat/completions", JSON, {"messages": USER, "max_tokens": 24, "stream": True}),
    ("chat_stream_usage", "POST", "/v1/chat/completions", JSON, {"messages": USER, "max_tokens": 8, "stream": True, "stream_options": {"include_usage": True}}),
    ("chat_length", "POST", "/v1/chat/completions", JSON, {"messages": USER, "max_tokens": 1}),
    ("chat_stop_string", "POST", "/v1/chat/completions", JSON, {"messages": USER, "max_tokens": 24, "stop": ["3"]}),
    ("chat_stop_single", "POST", "/v1/chat/completions", JSON, {"messages": USER, "max_tokens": 24, "stop": ", 4"}),
    ("chat_stream_stop", "POST", "/v1/chat/completions", JSON, {"messages": USER, "max_tokens": 24, "stop": ["3"], "stream": True}),
    ("chat_content_parts", "POST", "/v1/chat/completions", JSON, {"messages": [{"role": "user", "content": [{"type": "text", "text": "Say "}, {"type": "text", "text": "hi."}]}], "max_tokens": 4}),
    ("chat_image_part", "POST", "/v1/chat/completions", JSON, {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}]}], "max_tokens": 4}),
    ("chat_sampling", "POST", "/v1/chat/completions", JSON, {"messages": USER, "max_tokens": 8, "temperature": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "repetition_penalty": 1.1, "repetition_context_size": 20, "presence_penalty": 0.1, "frequency_penalty": 0.1, "logit_bias": {"16": -100}}),
    ("chat_logprobs", "POST", "/v1/chat/completions", JSON, {"messages": USER, "max_tokens": 4, "logprobs": 3}),
    ("chat_tools", "POST", "/v1/chat/completions", JSON, {"messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}], "tools": [TOOL], "max_tokens": 48}),
    ("chat_tools_stream", "POST", "/v1/chat/completions", JSON, {"messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}], "tools": [TOOL], "max_tokens": 48, "stream": True}),
    ("chat_template_kwargs", "POST", "/v1/chat/completions", JSON, {"messages": USER, "max_tokens": 4, "chat_template_kwargs": {"enable_thinking": False}}),
    ("completions", "POST", "/v1/completions", JSON, {"prompt": "1, 2, 3,", "max_tokens": 8}),
    ("completions_stream", "POST", "/v1/completions", JSON, {"prompt": "1, 2, 3,", "max_tokens": 8, "stream": True}),
    ("err_invalid_json", "POST", "/v1/chat/completions", JSON, "{not json"),
    ("err_not_object", "POST", "/v1/chat/completions", JSON, "[1, 2]"),
    ("err_missing_messages", "POST", "/v1/chat/completions", JSON, {"max_tokens": 4}),
    ("err_messages_wrong_type", "POST", "/v1/chat/completions", JSON, {"messages": "hi"}),
    ("err_stream_wrong_type", "POST", "/v1/chat/completions", JSON, {"messages": USER, "stream": "yes"}),
    ("err_max_tokens_negative", "POST", "/v1/chat/completions", JSON, {"messages": USER, "max_tokens": -1}),
    ("err_temperature_negative", "POST", "/v1/chat/completions", JSON, {"messages": USER, "temperature": -1}),
    ("err_top_p_range", "POST", "/v1/chat/completions", JSON, {"messages": USER, "top_p": 1.5}),
    ("err_logit_bias_type", "POST", "/v1/chat/completions", JSON, {"messages": USER, "logit_bias": [1, 2]}),
    ("err_logprobs_range", "POST", "/v1/chat/completions", JSON, {"messages": USER, "logprobs": 99}),
    ("err_completions_missing_prompt", "POST", "/v1/completions", JSON, {"max_tokens": 4}),
]


def exchange(method, path, headers, body):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    raw = None
    if body is not None:
        raw = body if isinstance(body, str) else json.dumps(body)
    try:
        conn.request(method, path, body=raw, headers=headers)
        resp = conn.getresponse()
        text = resp.read().decode("utf-8", errors="replace")
        return {
            "request": {"method": method, "path": path, "headers": headers, "body": body},
            "response": {"status": resp.status, "headers": resp.getheaders(), "body": text},
        }
    except Exception as e:  # the server dropped the connection
        return {
            "request": {"method": method, "path": path, "headers": headers, "body": body},
            "response": {"connection_error": f"{type(e).__name__}: {e}"},
        }
    finally:
        conn.close()


def wait_ready(proc):
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"mlx_lm.server exited with {proc.returncode}")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=2)
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                return
        except OSError:
            time.sleep(0.5)
    sys.exit("mlx_lm.server did not become ready")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlx_lm", "server", "--model", MODEL, "--port", str(PORT)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_ready(proc)
        for name, method, path, headers, body in CASES:
            record = exchange(method, path, headers, body)
            (OUT / f"{name}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            resp = record["response"]
            print(f"{name}: {resp.get('status', resp.get('connection_error'))}")
        meta = {"mlx_lm": version("mlx-lm"), "model": MODEL, "flags": ["--model", MODEL, "--port", str(PORT)]}
        (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main()
