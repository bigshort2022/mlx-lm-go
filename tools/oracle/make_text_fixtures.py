"""Generate tokenizer and chat-template golden fixtures from the Python reference.

Run from the repository root with a Python environment that has mlx-lm 0.31.3
(which brings transformers, tokenizers and jinja2):

    python tools/oracle/make_text_fixtures.py

The models must already be in the Hugging Face cache. Output goes to
testdata/<family>/{meta.json,pretokenize.jsonl,tokenize.jsonl,decode.jsonl,template.jsonl}.
The Go tests treat these files as the definition of correct behaviour.
"""

import json
import random
import unicodedata
from pathlib import Path

import jinja2
import tokenizers
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

FAMILIES = {
    "qwen2": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "qwen3_5": "mlx-community/Qwen3.5-9B-MLX-4bit",
}

ROOT = Path(__file__).resolve().parents[2]

HAND_WRITTEN = [
    "",
    " ",
    "  ",
    "\t",
    "\n",
    "\r\n",
    "\n\n\n",
    "Hello, world!",
    "hello world",
    " leading space",
    "trailing space ",
    "trailing spaces   ",
    "trailing newline\n",
    "trailing newlines\n\n",
    "two  spaces  between  words",
    "tab\tseparated\tvalues",
    "line one\nline two\nline three",
    "windows\r\nline\r\nendings\r\n",
    "mixed \n \t whitespace \n\n  runs",
    "don't can't won't I'll they're we've I'm he'd",
    "DON'T CAN'T I'LL THEY'RE WE'VE I'M HE'D",
    "it's 'quoted' and \"double quoted\"",
    "12345",
    "3.14159",
    "1,000,000",
    "v1.2.3-rc.4+build.5",
    "2026-11-21T02:38:00Z",
    "0x1F600 0b1010 1e-9",
    "!!!",
    "...",
    "?!?!",
    "a,b;c:d|e/f\\g",
    "(parentheses) [brackets] {braces} <angles>",
    "https://unite4ai.com/playground/local?server=http://127.0.0.1:8080#top",
    "someone@example.com",
    "def fizzbuzz(n):\n    for i in range(1, n + 1):\n        print('Fizz' * (i % 3 == 0) + 'Buzz' * (i % 5 == 0) or i)\n",
    'func main() {\n\tfmt.Println("hello")\n}\n',
    '{"name": "get_weather", "arguments": {"city": "São Paulo", "units": "metric"}}',
    "SELECT * FROM users WHERE id = 42;",
    "snake_case camelCase PascalCase kebab-case SCREAMING_SNAKE",
    "你好，世界！",
    "这是一个测试。今天天气怎么样？",
    "こんにちは世界。今日はいい天気ですね。",
    "カタカナとひらがなと漢字",
    "안녕하세요 세계",
    "مرحبا بالعالم",
    "שלום עולם",
    "नमस्ते दुनिया",
    "हिन्दी में कुछ शब्द और मात्राएँ",
    "สวัสดีชาวโลก",
    "Привет, мир!",
    "Γειά σου Κόσμε",
    "Tiếng Việt có dấu",
    "naïve café résumé coöperate",
    "e\u0301 a\u0300 o\u0302 n\u0303 (decomposed accents)",
    "A\u030a\u0301 stacked combining marks",
    "\u00a0non-breaking\u00a0space\u00a0",
    "zero\u200bwidth\u200bspace",
    "😀",
    "😀😃😄😁",
    "I ❤️ MLX",
    "👨‍👩‍👧‍👦 family",
    "👍🏽 skin tone",
    "🇺🇸🇯🇵🇧🇷 flags",
    "text with emoji 🚀 in the middle",
    "math: ∑ ∫ √ ∞ ≠ ≤ ≥ π",
    "currency: $ € £ ¥ ₹ ₿",
    "arrows → ← ↑ ↓ ⇒",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    "<|endoftext|>",
    "<|im_start",
    "im_start|>",
    "<|im_end|><|im_end|>",
    "before<|im_end|>after",
    "<think>\nreasoning\n</think>\n\nanswer",
    '<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>',
    "<tool_response>\n42\n</tool_response>",
    "a" * 100,
    " " * 50,
    "ab" * 64,
    "\n" * 20,
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog. " * 8,
    "ALL CAPS SENTENCE WITH NUMBERS 123 AND SYMBOLS #@!",
    "MiXeD cAsE wOrDs",
    "word1word2word3",
    "abc123def456",
    "a1b2c3",
    "１２３ fullwidth ＡＢＣ",
    "x² + y³ = z⁴",
    "\x00\x01\x02 control characters \x7f",
    "\ufeffbyte order mark",
    "ends with apostrophe'",
    "'s leading contraction",
    "rock'n'roll o'clock",
    "中文English混合text和123数字",
    "日本語とEnglishの混在",
    " 你好",
    "你好 ",
    "\n你好\n",
]

WORD_POOL = [
    "the", "model", "runs", "on", "Apple", "silicon", "MLX", "Go", "server", "token",
    "don't", "it's", "42", "3.14", "2026", "hello", "World", "naïve", "café", "über",
    "你好", "世界", "模型", "こんにちは", "テスト", "안녕", "привет", "मॉडल", "عالم", "שלום",
    "😀", "🚀", "❤️", "→", "≠", "π", "$", "€", "#", "@",
    "foo_bar", "fooBar", "foo-bar", "foo.bar", "foo/bar", "foo::bar", "x=1", "a+b", "(x)", "[0]",
    "<|im_end|>", "<think>", "</think>", "{", "}", '"', "'", ",", ".", "!",
]

JOINERS = [" ", " ", " ", "  ", "\n", "\t", "", ", ", ". ", "\n\n", " \n", "\r\n"]


def corpus():
    rng = random.Random(20261121)
    texts = list(HAND_WRITTEN)
    while len(texts) < 500:
        n = rng.randint(1, 24)
        parts = []
        for i in range(n):
            parts.append(rng.choice(WORD_POOL))
            if i < n - 1:
                parts.append(rng.choice(JOINERS))
        prefix = rng.choice(["", "", "", " ", "\n"])
        suffix = rng.choice(["", "", "", " ", "\n", "  "])
        texts.append(prefix + "".join(parts) + suffix)
    return texts


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city. Résumé: 天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["metric", "imperial"], "default": "metric"},
            },
            "required": ["city"],
        },
    },
}

# Keys are deliberately not in alphabetical order: tojson must keep request order.
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                "safe": {"type": "boolean"},
            },
        },
        "description": 'Search the web. Returns <b>HTML</b> & "quotes".',
    },
}


def template_cases(family):
    user = {"role": "user", "content": "What is the capital of France?"}
    system = {"role": "system", "content": "You are a terse assistant."}
    call_weather = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Paris", "units": "metric"}},
            }
        ],
    }
    cases = [
        {"name": "single_user", "messages": [user]},
        {"name": "system_user", "messages": [system, user]},
        {
            "name": "multi_turn",
            "messages": [
                system,
                user,
                {"role": "assistant", "content": "Paris."},
                {"role": "user", "content": "And of Japan?"},
            ],
        },
        {
            "name": "unicode_and_whitespace",
            "messages": [
                {"role": "user", "content": "  你好 😀\n\nnaïve café\t<|im_end|> literal\n"},
            ],
        },
        {
            "name": "empty_assistant_content",
            "messages": [user, {"role": "assistant", "content": ""}, {"role": "user", "content": "Again?"}],
        },
        {"name": "tools_user", "messages": [user], "tools": [WEATHER_TOOL]},
        {"name": "system_tools_user", "messages": [system, user], "tools": [WEATHER_TOOL, SEARCH_TOOL]},
        {
            "name": "tool_call_and_result",
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                call_weather,
                {"role": "tool", "content": '{"temp_c": 11, "sky": "overcast"}', "tool_call_id": "call_1"},
            ],
            "tools": [WEATHER_TOOL],
        },
        {
            "name": "parallel_tool_calls",
            "messages": [
                system,
                {"role": "user", "content": "Weather in Paris and Tokyo, then search for MLX."},
                {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [
                        {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
                        {"id": "b", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "東京"}}},
                        {"id": "c", "type": "function", "function": {"name": "web_search", "arguments": {"query": "MLX \"Apple\"", "limit": 3, "safe": True}}},
                    ],
                },
                {"role": "tool", "content": "11C", "tool_call_id": "a"},
                {"role": "tool", "content": "17C", "tool_call_id": "b"},
                {"role": "tool", "content": "[]", "tool_call_id": "c"},
                {"role": "assistant", "content": "Paris 11C, Tokyo 17C."},
                {"role": "user", "content": "Thanks."},
            ],
            "tools": [WEATHER_TOOL, SEARCH_TOOL],
        },
        {
            "name": "no_generation_prompt",
            "messages": [system, {"role": "user", "content": ""}],
            "add_generation_prompt": False,
        },
        {"name": "thinking_off", "messages": [system, user], "kwargs": {"enable_thinking": False}},
        {"name": "thinking_on", "messages": [system, user], "kwargs": {"enable_thinking": True}},
        {
            "name": "assistant_reasoning_content",
            "messages": [
                user,
                {"role": "assistant", "content": "Paris.", "reasoning_content": "France's capital is Paris."},
                {"role": "user", "content": "Sure?"},
            ],
        },
        {
            "name": "assistant_inline_think",
            "messages": [
                user,
                {"role": "assistant", "content": "<think>\nEasy.\n</think>\n\nParis."},
                {"role": "user", "content": "Sure?"},
            ],
            "kwargs": {"enable_thinking": False},
        },
        {
            "name": "tools_thinking_off",
            "messages": [system, user],
            "tools": [SEARCH_TOOL],
            "kwargs": {"enable_thinking": False},
        },
        # Inputs a template may reject. The expected outcome is whatever Python does.
        {"name": "no_messages", "messages": []},
        {"name": "system_only", "messages": [system]},
        {"name": "system_not_first", "messages": [user, system]},
        {"name": "tool_result_first", "messages": [{"role": "tool", "content": "x"}]},
    ]
    return cases


def render(tok, case, tokenize):
    out = tok.apply_chat_template(
        case["messages"],
        tools=case.get("tools"),
        add_generation_prompt=case.get("add_generation_prompt", True),
        tokenize=tokenize,
        **case.get("kwargs", {}),
    )
    if tokenize and not isinstance(out, list):
        out = list(out["input_ids"])
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    for family, repo in FAMILIES.items():
        path = snapshot_download(repo, local_files_only=True)
        tok = AutoTokenizer.from_pretrained(path)
        backend = tok.backend_tokenizer
        out = ROOT / "testdata" / family
        out.mkdir(parents=True, exist_ok=True)
        texts = corpus()

        pre_rows, tok_rows, dec_rows = [], [], []
        for text in texts:
            normalized = backend.normalizer.normalize_str(text) if backend.normalizer else text
            pieces = [normalized[s:e] for _, (s, e) in backend.pre_tokenizer.pre_tokenize_str(normalized)]
            pre_rows.append({"text": text, "normalized": normalized, "pieces": pieces})

            ids = tok.encode(text)
            assert ids == tok.encode(text, add_special_tokens=False), "tokenizer adds special tokens"
            tok_rows.append({"text": text, "ids": ids})
            dec_rows.append({"ids": ids, "text": tok.decode(ids)})

        # Prefixes that cut a multi-byte character in half: decode must match Python's
        # replacement-character behaviour, and streaming must hold the bytes back.
        for text in ["😀", "你好", "naïve", "👨‍👩‍👧‍👦", "नमस्ते"]:
            ids = tok.encode(text)
            for n in range(1, len(ids)):
                dec_rows.append({"ids": ids[:n], "text": tok.decode(ids[:n])})

        tpl_rows = []
        for case in template_cases(family):
            row = {
                "name": case["name"],
                "messages": case["messages"],
                "tools": case.get("tools"),
                "kwargs": case.get("kwargs", {}),
                "add_generation_prompt": case.get("add_generation_prompt", True),
            }
            try:
                row["prompt"] = render(tok, case, tokenize=False)
                row["prompt_ids"] = render(tok, case, tokenize=True)
            except Exception as e:  # the template called raise_exception, or failed to render
                row["error"] = f"{type(e).__name__}: {e}"
            tpl_rows.append(row)

        write_jsonl(out / "pretokenize.jsonl", pre_rows)
        write_jsonl(out / "tokenize.jsonl", tok_rows)
        write_jsonl(out / "decode.jsonl", dec_rows)
        write_jsonl(out / "template.jsonl", tpl_rows)

        special = {}
        for name in ("bos_token", "eos_token", "pad_token", "unk_token"):
            value = getattr(tok, name, None)
            special[name] = value
            special[name + "_id"] = tok.convert_tokens_to_ids(value) if value is not None else None
        meta = {
            "repo": repo,
            "snapshot": Path(path).name,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "jinja2": jinja2.__version__,
            "nfc_changed": sum(1 for r in pre_rows if r["text"] != unicodedata.normalize("NFC", r["text"])),
            "special_tokens": special,
            "vocab_size": len(tok),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        errors = [r["name"] for r in tpl_rows if "error" in r]
        print(f"{family}: {len(tok_rows)} texts, {len(dec_rows)} decodes, {len(tpl_rows)} templates, errors={errors}")


if __name__ == "__main__":
    main()
