"""Generate chat-template differential fixtures from the Python reference.

Run from the repository root with a Python environment that has mlx-lm 0.31.3
(which brings transformers and jinja2):

    python tools/oracle/make_template_differential.py

The models must already be in the Hugging Face cache. Output goes to
internal/chattemplate/testdata/:

    differential_<family>.jsonl  requests beyond testdata/<family>/template.jsonl,
                                 hand-written and seeded random ones, rendered
                                 with tokenizer.apply_chat_template
    semantics.jsonl              small templates, one Jinja rule each, rendered
                                 with the environment Hugging Face compiles
                                 chat templates in

Every row records the prompt Python produced or the error it raised. The Go
tests treat these files as the definition of correct behaviour.
"""

import json
import random
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template

FAMILIES = {
    "qwen2": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "qwen3_5": "mlx-community/Qwen3.5-9B-MLX-4bit",
}

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "internal" / "chattemplate" / "testdata"

USER = {"role": "user", "content": "Hi"}
SYSTEM = {"role": "system", "content": "Sys."}
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "d",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def call(arguments, content=""):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": arguments}}],
    }


def result(content="ok"):
    return {"role": "tool", "content": content}


def differential_cases():
    """Requests a server can receive that the golden rows do not cover."""
    return [
        # What a model answers when the prompt already opened the think block.
        {
            "name": "think_closed_but_never_opened",
            "messages": [USER, {"role": "assistant", "content": "reason</think>\n\nans"}, USER],
        },
        {
            "name": "think_closed_but_never_opened_before_tool",
            "messages": [USER, {"role": "assistant", "content": "reason</think>\n\nans"}, result("x")],
        },
        {
            "name": "two_think_blocks",
            "messages": [USER, {"role": "assistant", "content": "<think>a</think>mid<think>b</think>end"}, USER],
        },
        {
            "name": "think_with_tool_call",
            "messages": [
                USER,
                {
                    "role": "assistant",
                    "content": "<think>\nhm\n</think>\n\nans",
                    "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
                },
                result("r1"),
            ],
            "tools": [TOOL],
        },
        {
            "name": "reasoning_content_after_last_query",
            "messages": [USER, {"role": "assistant", "content": "ans", "reasoning_content": "  think  "}],
        },
        {
            "name": "reasoning_content_null",
            "messages": [USER, {"role": "assistant", "content": "ans", "reasoning_content": None}, USER],
        },
        # OpenAI multi-part content.
        {
            "name": "content_parts_text",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]}
            ],
        },
        {
            "name": "content_parts_image_and_video",
            "messages": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What?"}, {"type": "video"}]}
            ],
        },
        {
            "name": "content_parts_vision_ids",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image", "image": "u"},
                        {"image_url": {"url": "u"}},
                        {"type": "text", "text": "What?"},
                        {"type": "video", "video": "v"},
                    ],
                },
                {"role": "assistant", "content": "A."},
                {"role": "user", "content": [{"type": "image"}, {"type": "video"}]},
            ],
            "kwargs": {"add_vision_id": True},
        },
        {"name": "content_parts_empty", "messages": [{"role": "user", "content": []}]},
        {"name": "content_parts_in_system", "messages": [{"role": "system", "content": [{"text": "Sys."}]}, USER]},
        {"name": "system_image", "messages": [{"role": "system", "content": [{"type": "image"}]}, USER]},
        {"name": "system_video", "messages": [{"role": "system", "content": [{"video": "x"}]}, USER]},
        {"name": "content_part_of_unknown_type", "messages": [{"role": "user", "content": [{"type": "audio"}]}]},
        {"name": "content_part_not_an_object", "messages": [{"role": "user", "content": ["text"]}]},
        {"name": "content_is_a_number", "messages": [{"role": "user", "content": 5}]},
        {"name": "content_is_an_object", "messages": [{"role": "user", "content": {"text": "x"}}]},
        {"name": "content_is_null", "messages": [{"role": "user", "content": None}]},
        {"name": "content_is_missing", "messages": [{"role": "user"}]},
        {"name": "content_is_a_boolean", "messages": [{"role": "user", "content": True}]},
        # Tool calls as clients really send them.
        {
            "name": "tool_call_with_null_content",
            "messages": [USER, call({"a": 1}, content=None), result()],
            "tools": [TOOL],
        },
        {
            "name": "tool_call_without_content",
            "messages": [
                USER,
                {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]},
                result(),
            ],
            "tools": [TOOL],
        },
        {"name": "tool_call_arguments_as_string", "messages": [USER, call('{"a": 1}'), result()], "tools": [TOOL]},
        {
            "name": "tool_call_arguments_of_every_type",
            "messages": [
                USER,
                call(
                    {
                        "s": "x\ny",
                        "i": 3,
                        "neg": -7,
                        "f": 1.5,
                        "e": 1e-9,
                        "big": 1e20,
                        "whole": 2.0,
                        "b": True,
                        "b2": False,
                        "n": None,
                        "l": [1, "a", {"k": None}],
                        "el": [],
                        "d": {"z": 1, "a": [True]},
                        "ed": {},
                        "u": "東京 <b>&",
                        "empty": "",
                        "num_str": "1.0",
                        "q": '"quoted" and \'single\'',
                    }
                ),
                result(),
            ],
            "tools": [TOOL],
        },
        {"name": "tool_call_with_no_arguments", "messages": [USER, call({}), result()], "tools": [TOOL]},
        {
            "name": "tool_call_without_arguments_key",
            "messages": [
                USER,
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f"}}]},
                result(),
            ],
            "tools": [TOOL],
        },
        {
            "name": "tool_call_arguments_null",
            "messages": [USER, call(None), result()],
            "tools": [TOOL],
        },
        {
            "name": "tool_call_without_function_wrapper",
            "messages": [
                USER,
                {"role": "assistant", "content": "x", "tool_calls": [{"name": "f", "arguments": {"a": 1}}]},
                result(),
            ],
            "tools": [TOOL],
        },
        {
            "name": "tool_call_without_name",
            "messages": [
                USER,
                {"role": "assistant", "content": "x", "tool_calls": [{"function": {"arguments": {}}}]},
                USER,
            ],
        },
        {
            "name": "tool_call_with_unicode_keys",
            "messages": [USER, call({"ключ": "значение", "🙂": [1.0, 2.50, -0.0, 3]}), result()],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "description": "é\u2028\u007f\t",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        {"name": "tool_calls_empty", "messages": [USER, {"role": "assistant", "content": "x", "tool_calls": []}, USER]},
        {
            "name": "tool_calls_null",
            "messages": [USER, {"role": "assistant", "content": "x", "tool_calls": None}, USER],
        },
        {
            "name": "tool_calls_is_an_object",
            "messages": [USER, {"role": "assistant", "content": "x", "tool_calls": {"name": "f"}}, USER],
        },
        {"name": "tool_results_last", "messages": [USER, call({"a": 1}), result("r1"), result("r2")], "tools": [TOOL]},
        {
            "name": "tool_rounds",
            "messages": [USER, call({"a": 1}), result("r1"), call({"b": 2}, content="more"), result("r2")],
            "tools": [TOOL],
        },
        {"name": "tool_result_is_a_number", "messages": [USER, call({"a": 1}), result(42)], "tools": [TOOL]},
        {
            "name": "tool_result_is_content_parts",
            "messages": [USER, call({"a": 1}), result([{"type": "text", "text": "res"}])],
            "tools": [TOOL],
        },
        {
            "name": "user_message_that_looks_like_a_tool_response",
            "messages": [
                USER,
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "<tool_response>\nx\n</tool_response>"},
            ],
        },
        # Tools in unusual shapes.
        {"name": "tools_empty", "messages": [SYSTEM, USER], "tools": []},
        {"name": "tools_is_an_object", "messages": [USER], "tools": {"a": 1}},
        {"name": "tools_with_blank_system", "messages": [{"role": "system", "content": "   "}, USER], "tools": [TOOL]},
        {
            "name": "tool_schema_with_items_key",
            "messages": [USER],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {
                            "type": "object",
                            "properties": {"xs": {"type": "array", "items": {"type": "number"}, "keys": 1.0}},
                        },
                    },
                }
            ],
        },
        # Template variables of the wrong type.
        {"name": "thinking_null", "messages": [SYSTEM, USER], "kwargs": {"enable_thinking": None}},
        {"name": "thinking_string", "messages": [SYSTEM, USER], "kwargs": {"enable_thinking": "false"}},
        {"name": "thinking_zero", "messages": [SYSTEM, USER], "kwargs": {"enable_thinking": 0}},
        {"name": "unknown_kwarg", "messages": [SYSTEM, USER], "kwargs": {"unused": {"a": [1]}}},
        # Roles and ordering.
        {"name": "unknown_role", "messages": [USER, {"role": "developer", "content": "x"}]},
        {"name": "role_missing", "messages": [{"content": "x"}]},
        {"name": "role_null", "messages": [{"role": None, "content": "x"}, USER]},
        {"name": "message_is_a_string", "messages": ["hi"]},
        {"name": "message_is_null", "messages": [None]},
        {"name": "two_system_messages", "messages": [SYSTEM, SYSTEM, USER]},
        {"name": "assistant_first", "messages": [{"role": "assistant", "content": "a"}, USER]},
        {
            "name": "assistant_last_without_generation_prompt",
            "messages": [USER, {"role": "assistant", "content": "partial"}],
            "add_generation_prompt": False,
        },
        # Whitespace that trim must and must not remove.
        {
            "name": "whitespace_around_content",
            "messages": [
                {"role": "system", "content": "  \n sys \t\n"},
                {"role": "user", "content": "\u00a0 nbsp \u3000"},
                {"role": "assistant", "content": "\n\n<think>\n a \n</think>\n\n b \n"},
                {"role": "user", "content": "\x1f ctl \x1c"},
                {"role": "assistant", "content": "\u200b zero width \ufeff"},
                {"role": "user", "content": "\u2028 line sep \u0085"},
            ],
        },
        {"name": "blank_user_content", "messages": [{"role": "user", "content": " \n\t "}]},
    ]


# Strings random requests draw content from: think tags in every state of
# balance, tool-response look-alikes, template syntax, whitespace.
STRINGS = [
    "Hi",
    "",
    " ",
    "\n",
    "  padded  ",
    "é天😀",
    "<think>\nr\n</think>\n\nans",
    "r</think>ans",
    "<think>only",
    "</think>",
    "<think></think>",
    "a</think>b</think>c",
    "<think>a<think>b</think>c",
    "<tool_response>\nx\n</tool_response>",
    "<tool_response>x",
    "a\n\nb",
    "\x1f x \x1c",
    "\u00a0nb\u3000",
    "<|im_end|>",
    "{{ x }}",
    "{% if %}",
    "'q' \"dq\"",
]


def random_cases(count=150):
    """Seeded random requests, most of them malformed in some way.

    A run of tens of thousands of these found no difference between the Go
    renderer and Python; this sample keeps it that way.
    """
    rng = random.Random(20261121)

    def value(depth=0):
        r = rng.random()
        if r < 0.25:
            return rng.choice(STRINGS)
        if r < 0.4:
            return rng.choice([0, 1, -7, 3, 10**12, 2**53])
        if r < 0.5:
            return rng.choice([1.5, 0.0, -0.0, 1e-9, 1e20, 2.0, 0.1, 123456.789, 1e16, 1e15, 0.0001, 0.00001])
        if r < 0.6:
            return rng.choice([True, False, None])
        if depth > 2:
            return "leaf"
        if r < 0.8:
            return [value(depth + 1) for _ in range(rng.randint(0, 3))]
        keys = ["a", "b", "items", "keys", "é", "z k", "type", "text", "name"]
        return {rng.choice(keys): value(depth + 1) for _ in range(rng.randint(0, 3))}

    def part():
        r = rng.random()
        if r < 0.4:
            return {"type": "text", "text": rng.choice(STRINGS)}
        if r < 0.9:
            return rng.choice(
                [
                    {"type": "image"},
                    {"image": "u"},
                    {"image_url": {"url": "u"}},
                    {"type": "video"},
                    {"video": "v", "type": "video"},
                    {"text": rng.choice(STRINGS)},
                    rng.choice(STRINGS),
                    {"type": "audio"},
                ]
            )
        return value()

    def content():
        r = rng.random()
        if r < 0.6:
            return rng.choice(STRINGS)
        if r < 0.68:
            return None
        if r < 0.9:
            return [part() for _ in range(rng.randint(0, 3))]
        return value()

    def tool_call():
        function = {}
        if rng.random() < 0.9:
            function["name"] = rng.choice(["f", "get_weather", "", "é"])
        r = rng.random()
        if r < 0.7:
            keys = ["a", "b", "city", "items", "é", "q"]
            function["arguments"] = {rng.choice(keys): value() for _ in range(rng.randint(0, 3))}
        elif r < 0.8:
            function["arguments"] = '{"a": 1}'
        elif r < 0.85:
            function["arguments"] = None
        elif r < 0.9:
            function["arguments"] = value()
        if rng.random() < 0.8:
            return {"id": "c", "type": "function", "function": function}
        return function

    def message():
        role = rng.choices(["user", "assistant", "tool", "system", "developer", None], [35, 30, 20, 8, 3, 2])[0]
        msg = {}
        if role is not None or rng.random() < 0.5:
            msg["role"] = role
        if rng.random() < 0.93:
            msg["content"] = content()
        if role == "assistant":
            r = rng.random()
            if r < 0.45:
                msg["tool_calls"] = [tool_call() for _ in range(rng.randint(0, 3))]
            elif r < 0.5:
                msg["tool_calls"] = None
            elif r < 0.53:
                msg["tool_calls"] = value()
            r = rng.random()
            if r < 0.2:
                msg["reasoning_content"] = rng.choice(STRINGS)
            elif r < 0.25:
                msg["reasoning_content"] = None
            elif r < 0.28:
                msg["reasoning_content"] = value()
        return value() if rng.random() < 0.03 else msg

    def tool():
        properties = {rng.choice(["a", "items", "é"]): value() for _ in range(rng.randint(0, 2))}
        return {
            "type": "function",
            "function": {
                "name": rng.choice(["f", "g"]),
                "description": rng.choice(STRINGS),
                "parameters": {"type": "object", "properties": properties},
            },
        }

    cases = []
    for i in range(count):
        messages = []
        if rng.random() < 0.35:
            messages.append({"role": "system", "content": content()})
        messages += [message() for _ in range(rng.randint(1, 6))]
        if isinstance(messages[0], list):
            # apply_chat_template would take the request for a batch of
            # conversations, which the Go renderer has no equivalent of.
            messages[0] = {"role": "user", "content": content()}
        case = {"name": f"random_{i:03d}", "messages": messages}
        if rng.random() < 0.35:
            case["tools"] = [tool() for _ in range(rng.randint(0, 2))]
        kwargs = {}
        if rng.random() < 0.5:
            kwargs["enable_thinking"] = rng.choice([True, False, False, None, "false", 0, 1])
        if rng.random() < 0.15:
            kwargs["add_vision_id"] = rng.choice([True, False, None, 1, "yes", ""])
        case["kwargs"] = kwargs
        case["add_generation_prompt"] = rng.random() < 0.8
        cases.append(case)
    return cases


# The conversation every semantics row is rendered with; the Go test passes
# the same one.
USER_HI = {"role": "user", "content": "hi"}

# Variables every semantics row can read, besides the ones it adds itself.
VARIABLES = {
    "s": "a,b",
    "es": "",
    "i": 3,
    "z": 0,
    "f": 1.5,
    "zf": 0.0,
    "big": 1e20,
    "small": 1e-9,
    "whole": 2.0,
    "t": True,
    "fl": False,
    "n": None,
    "l": [1, 2],
    "el": [],
    "one": ["x"],
    "six": [0, 1, 2, 3, 4, 5],
    "d": {"z": 1, "a": None},
    "e": {},
    "m": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo", "extra": [1, 2.0]}],
    "ws": "\x1f x \x1c",
    "nb": "\u00a0x\u3000",
    "quotes": {"q": "it's", "dq": 'say "hi"', "both": 'it\'s "x"', "nl": "a\nb\t\r", "u": "é天😀", "ctl": "\x01\x7f\u00a0\u2028\\"},
    "schema": {"type": "array", "items": {"type": "string"}},
    "floats": [1.5, 1e20, 1e-9, 2.0, 0.0, -0.0, 1e16, 0.0001, 123456789.125],
    "think": "reason</think>\n\nans",
    "d2": {"a": None, "z": 1},
}


def falsy(expression):
    return "{% if " + expression + " %}T{% else %}F{% endif %}"


def semantics_cases():
    """(group, template source[, extra variables]) rows, one Jinja rule each."""
    rows = [
        # Template text outside tags.
        ("text", "hello"),
        ("text", "a\nb\n"),
        ("text", "a\n\n"),
        ("text", "\n"),
        ("text", ""),
        ("text", "a { b } c %} d }} e #}"),
        ("text", "{# comment #}a{# multi\nline #}b"),
        ("text", "é 天气 😀"),
        ("text", "a\r\nb\rc"),
        # Whitespace control: trim_blocks, lstrip_blocks and the minus signs.
        ("whitespace", "{% if true %}\nyes\n{% endif %}\nend"),
        ("whitespace", "  {% if true %}\n  yes\n  {% endif %}\nend"),
        ("whitespace", "x  {{- ' y ' -}}  z"),
        ("whitespace", "{%- if t -%}  a  {%- endif -%}  b"),
        ("whitespace", "a\n  {{ 'x' }}\nb"),
        ("whitespace", "a  {% if true %}b{% endif %}"),
        ("whitespace", "  {# c #}\nx"),
        ("whitespace", "a{#- c -#}  b"),
        ("whitespace", "a  {#- c #}  b"),
        ("whitespace", "{{ 'a' }}\n{{ 'b' }}\n"),
        ("whitespace", "{% if true %}  \nx{% endif %}"),
        ("whitespace", "{% if true %}\n\nx{% endif %}"),
        ("whitespace", "{% if true %}\r\nx{% endif %}"),
        ("whitespace", "a \n {%- if true %} b {%- endif %}"),
        ("whitespace", "a\n{% if false %}x{% endif %}\nb"),
        ("whitespace", "{% if true -%}\n\n  a{% endif %}"),
        ("whitespace", "   {%- if true %}a{% endif %}"),
        ("whitespace", "a\n\t {% if true %}\t b{% endif %}"),
        ("whitespace", "a\n  {% if true %}b{% endif %}  {% if true %}c{% endif %}"),
        ("whitespace", "{{ 'a' }}  {%- if true %}b{% endif %}"),
        ("whitespace", "{{ 'a' -}}\n\n  {% if true %}b{% endif %}"),
        ("whitespace", "{% if true %}a{% endif -%}\n\n  b\n"),
        ("whitespace", "{%-if true-%} a {%-endif-%}"),
        ("whitespace", "{{'a'}}{{-'b'-}}"),
        ("whitespace", "{{\u00a0'a'\u3000}}{%\u2003if true\u00a0%}b{%\tendif\n%}"),
        ("whitespace", "{{ 'a' }}{#- c #}\n b"),
        ("whitespace", "{{ 'a' if t else 'b' -}}  x"),
        ("whitespace", "a\u00a0{%- if true %}b{% endif -%}\u3000\u2028c"),
        ("whitespace", "a\n\u00a0{% if true %}b{% endif %}"),
        # Literals.
        ("literals", "{{ 'a' }}|{{ \"b\" }}|{{ 3 }}|{{ -3 }}|{{ 0 }}|{{ 12345678901 }}"),
        ("literals", "{{ true }}|{{ false }}|{{ none }}|{{ True }}|{{ False }}|{{ None }}"),
        ("literals", "{{ 'a\\nb' }}|{{ 'tab\\there' }}|{{ 'r\\rr' }}"),
        ("literals", "{{ \"q\\\"q\" }}|{{ 'q\\'q' }}|{{ 'a\\\\b' }}"),
        ("literals", "{{ \"it's\" }}|{{ 'say \"hi\"' }}"),
        ("literals", "{{ 'é天😀' }}|{{ '{{' }}|{{ '%}' }}|{{ '' }}|"),
        ("literals", "{{ 'two\nlines' }}"),
        # What a variable prints as.
        ("variables", "{{ s }}|{{ es }}|{{ i }}|{{ z }}|{{ t }}|{{ fl }}|{{ n }}|{{ q }}|"),
        ("variables", "{{ f }}|{{ zf }}|{{ big }}|{{ small }}|{{ whole }}"),
        ("variables", "{{ messages|length }}|{{ add_generation_prompt }}|{{ tools is none }}|{{ documents is none }}"),
        ("variables", "{{ documents is defined }}|{{ tools is defined }}|{{ enable_thinking is defined }}"),
        ("variables", "{{ l }}|{{ el }}|{{ one }}|{{ d }}|{{ e }}"),
        ("variables", "{{ m }}"),
        ("variables", "{{ quotes }}"),
        ("variables", "{{ floats }}"),
        ("variables", "{{ m[1].extra }}|{{ quotes.q }}|{{ quotes.ctl }}"),
        ("variables", "{% for pair in d|items %}{{ pair }}{% endfor %}"),
        # Attribute and item access; undefined.
        ("access", "{{ d.z }}|{{ d['z'] }}|{{ d.a }}|{{ d['a'] }}"),
        ("access", "{{ d.q }}|{{ d['q'] }}|{{ d.q is defined }}|{{ d['q'] is defined }}"),
        ("access", "{{ d.q.r }}"),
        ("access", "{{ d['q']['r'] }}"),
        ("access", "{{ q.a }}"),
        ("access", "{{ q['a'] }}"),
        ("access", "{{ q[0] }}"),
        ("access", "{{ m[0].role }}|{{ m[0]['role'] }}|{{ m[-1].content }}|{{ m[1]['extra'][1] }}"),
        ("access", "{{ m[2] }}|{{ m[2] is defined }}|{{ m[-3] is defined }}|{{ m[-2].role }}"),
        ("access", "{{ m[2].role }}"),
        ("access", "{{ l[0] }}|{{ l[1] }}|{{ l[-1] }}|{{ l[-2] }}|{{ one[-1] }}|{{ one[0] }}"),
        ("access", "{{ l[5] is defined }}|{{ l[-3] is defined }}|{{ l['x'] is defined }}|{{ l.x is defined }}"),
        ("access", "{{ s[0] }}|{{ s[-1] }}|{{ s[-3] }}|{{ s[5] is defined }}|{{ s.x is defined }}|{{ s['x'] is defined }}"),
        ("access", "{{ 'é天😀'[1] }}|{{ 'é天😀'[-1] }}|{{ 'é天😀'|length }}"),
        ("access", "{{ n.x is defined }}|{{ n[0] is defined }}|{{ i.x is defined }}|{{ i[0] is defined }}|{{ t.x is defined }}"),
        ("access", "{{ n.x.y }}"),
        ("access", "{{ d[0] is defined }}|{{ d[n] is defined }}|{{ l[n] is defined }}|{{ l[t] }}|{{ l[fl] }}"),
        ("access", "{{ l[f] is defined }}|{{ l['0'] is defined }}|{{ d[l] is defined }}"),
        ("access", "{{ d[q] }}"),
        ("access", "{{ l[q] }}"),
        ("access", "{{ schema.type }}|{{ schema['items'] | tojson }}|{{ schema['items'].type }}"),
        ("access", "{{ l[::-1] }}|{{ l[1:] }}|{{ l[:-1] }}|{{ l[:] }}|{{ l[5:] }}|{{ l[-1:] }}|{{ l[0:0] }}"),
        ("access", "{{ six[::2] }}|{{ six[1:5:2] }}|{{ six[::-2] }}|{{ six[4:1:-1] }}|{{ six[-2:] }}|{{ six[:-2] }}"),
        ("access", "{{ six[10:] }}|{{ six[-10:2] }}|{{ six[3:1] }}|{{ six[1:-10:-1] }}|{{ six[10::-3] }}"),
        ("access", "{{ s[::-1] }}|{{ s[1:] }}|{{ s[:1] }}|{{ 'é天😀'[::-1] }}|{{ 'é天😀'[1:2] }}"),
        ("access", "{{ m[::-1][0].role }}|{{ m[::-1]|length }}|{{ six[i:] }}|{{ six[:i] }}|{{ six[n:n] }}"),
        ("access", "{{ n[::-1] is defined }}"),
        ("access", "{{ d[1:2] is defined }}"),
        ("access", "{{ i[1:] is defined }}"),
        ("access", "{{ six[::0] is defined }}"),
        ("access", "{{ six['a':] is defined }}"),
        ("access", "{{ q[1:] }}"),
        # Operators.
        ("operators", "{{ s + s }}|{{ i + 1 }}|{{ i + f }}|{{ 7 - i }}|{{ i - f }}|{{ t + 1 }}|{{ l + l }}|{{ -i }}|{{ -f }}"),
        ("operators", "{{ s + l }}"),
        ("operators", "{{ s + n }}"),
        ("operators", "{{ s + i }}"),
        ("operators", "{{ i + s }}"),
        ("operators", "{{ l + s }}"),
        ("operators", "{{ d + d }}"),
        ("operators", "{{ q + 1 }}"),
        ("operators", "{{ 1 + q }}"),
        ("operators", "{{ n + 1 }}"),
        ("operators", "{{ i - s }}"),
        ("operators", "{{ s - s }}"),
        ("operators", "{{ -s }}"),
        ("operators", "{{ -q }}"),
        ("operators", "{{ s ~ i ~ n ~ f ~ t ~ q ~ '|' ~ l ~ d }}"),
        ("operators", "{{ 'a' ~ 'b' + 'c' }}"),
        ("operators", "{{ 1 + 2 ~ '' }}"),
        ("operators", "{{ 1 ~ 2 + 3 }}"),
        ("operators", "{{ 2 - 5 }}|{{ 0 - z }}|{{ i - 3 - 1 }}|{{ i - (3 - 1) }}"),
        ("operators", "{{ not t }}|{{ not fl }}|{{ not q }}|{{ not n }}|{{ not s }}|{{ not es }}|{{ not e }}|{{ not l }}"),
        ("operators", "{{ t and fl }}|{{ t and 'x' }}|{{ fl and 'x' }}|{{ es and 'x' }}|{{ n and 'x' }}|{{ q and 'x' }}|"),
        ("operators", "{{ t or fl }}|{{ fl or 'y' }}|{{ es or 'dflt' }}|{{ n or 'x' }}|{{ q or 'u' }}|{{ z or el }}|{{ l or 'x' }}"),
        ("operators", "{{ t or q.a.b }}|{{ fl and q.a.b }}|{{ 'x' if t else q.a.b }}"),
        ("operators", "{{ not t or t }}|{{ not (t or t) }}|{{ t or fl and fl }}|{{ (t or fl) and fl }}"),
        ("operators", "{{ i == 3 }}|{{ 2 == whole }}|{{ f == 1 }}|{{ t == 1 }}|{{ s == 'a,b' }}|{{ n == none }}|{{ fl == 0 }}|{{ s == 3 }}"),
        ("operators", "{{ l == l }}|{{ l == one }}|{{ d == d }}|{{ d == e }}|{{ l == d }}|{{ n == fl }}|{{ es == fl }}"),
        ("operators", "{{ q == none }}|{{ q == q2 }}|{{ q == '' }}|{{ q != q2 }}|{{ q == fl }}"),
        ("operators", "{{ i != 3 }}|{{ i != 4 }}|{{ s != 'x' }}|{{ n != none }}|{{ l != l }}"),
        ("operators", "{{ i > 2 }}|{{ i >= 3 }}|{{ i < 3 }}|{{ i <= 3 }}|{{ i < f }}|{{ f < i }}|{{ t < 2 }}"),
        ("operators", "{{ s < 'b' }}|{{ s > 'a' }}|{{ 'é' > 'z' }}|{{ 'a' < 'B' }}|{{ es < s }}|{{ 'ab' <= 'ab' }}"),
        ("operators", "{{ s > 5 }}"),
        ("operators", "{{ n < 1 }}"),
        ("operators", "{{ q < 1 }}"),
        ("operators", "{{ d < d }}"),
        ("operators", "{{ l > 1 }}"),
        ("operators", "{{ 'z' in d }}|{{ 'q' in d }}|{{ 1 in l }}|{{ 3 in l }}|{{ 'a' in s }}|{{ 'x' in s }}|{{ '' in s }}"),
        ("operators", "{{ 'q' not in d }}|{{ 'z' not in d }}|{{ 1 not in l }}|{{ 'x' in one }}|{{ n in d }}|{{ whole in l }}"),
        ("operators", "{{ 'x' in q }}|{{ l in six }}|{{ 'user' in m[0] }}|{{ 'role' in m[0] }}|{{ t in l }}"),
        ("operators", "{{ 1 in s }}"),
        ("operators", "{{ 'x' in n }}"),
        ("operators", "{{ 'x' in i }}"),
        ("operators", "{{ q in s }}"),
        ("operators", "{{ l in d }}"),
        ("operators", "{{ l|length - 1 }}|{{ not l|length == 2 }}|{{ -i|string|length }}|{{ (l|length) > 1 }}"),
        ("operators", "{{ 'a' if t else 'b' }}|{{ 'a' if fl else 'b' }}|{{ ('a' if fl else 'b') + 'c' }}|{{ 'x' if fl }}|"),
        ("operators", "{{ 'a' if fl else 'b' if fl else 'c' }}|{{ 'a' if t else 'b' if q.x.y else 'c' }}"),
        ("operators", "{{ ('x' if fl) is defined }}|{{ s if s is string else s|tojson }}|{{ l if l is string else l|tojson }}"),
        ("operators", "{{ (s) }}|{{ ((i + 1)) - (1) }}|{{ (not t) or t }}"),
        ("operators", "{{ i - whole }}|{{ i + zf }}|{{ fl + fl }}|{{ t - t }}|{{ -t }}|{{ -fl }}|{{ -zf }}|{{ f + f }}|{{ small + small }}"),
        ("operators", "{{ t == 'True' }}|{{ s == (s|safe) }}|{{ (s|safe) == (s|safe) }}|{{ 'x' in (s|safe) }}|{{ (s|safe) in s }}|{{ ',' in (s|safe) }}"),
        ("operators", "{{ t > fl }}|{{ s <= s }}|{{ (s|safe) < 'b' }}|{{ l == l[:] }}|{{ l[::-1][::-1] == l }}|{{ m[0] == m[0] }}|{{ m[0] == m[1] }}"),
        ("operators", "{{ d == d2 }}|{{ d != d2 }}|{{ d2 }}|{{ s[t] }}|{{ l|tojson|length }}|{{ ' a ' | trim | length }}"),
        ("operators", "{{ s.split(',')[0].strip().startswith('a') }}|{{ q|string|length }}|{{ (1 if fl) ~ 'x' }}"),
        # Tests.
        ("tests", "{{ fl is false }}|{{ z is false }}|{{ es is false }}|{{ n is false }}|{{ q is false }}|{{ zf is false }}"),
        ("tests", "{{ t is true }}|{{ i is true }}|{{ s is true }}|{{ 1 is true }}|{{ l is true }}"),
        ("tests", "{{ n is defined }}|{{ q is defined }}|{{ es is defined }}|{{ fl is defined }}|{{ d.a is defined }}"),
        ("tests", "{{ n is undefined }}|{{ q is undefined }}|{{ d.q is undefined }}|{{ z is undefined }}"),
        ("tests", "{{ n is none }}|{{ q is none }}|{{ fl is none }}|{{ es is none }}|{{ d.a is none }}|{{ d.q is none }}"),
        ("tests", "{{ s is string }}|{{ es is string }}|{{ l is string }}|{{ n is string }}|{{ q is string }}|{{ i is string }}"),
        ("tests", "{{ d is mapping }}|{{ e is mapping }}|{{ l is mapping }}|{{ s is mapping }}|{{ n is mapping }}|{{ q is mapping }}"),
        ("tests", "{{ l is iterable }}|{{ s is iterable }}|{{ d is iterable }}|{{ i is iterable }}|{{ n is iterable }}|{{ q is iterable }}|{{ t is iterable }}|{{ f is iterable }}"),
        ("tests", "{{ l is sequence }}|{{ s is sequence }}|{{ d is sequence }}|{{ i is sequence }}|{{ n is sequence }}|{{ q is sequence }}|{{ t is sequence }}"),
        ("tests", "{{ n is not none }}|{{ s is not string }}|{{ q is not defined }}|{{ l is not mapping }}|{{ fl is not false }}"),
        ("tests", "{{ i is defined and i is not none }}|{{ q is defined and q.x.y }}|{{ not q is defined }}"),
        ("tests", "{{ s|length is defined }}|{{ (d|items) is iterable }}|{{ namespace(a=1) is mapping }}|{{ namespace() is iterable }}"),
        ("tests", "{% macro mm() %}{% endmacro %}{{ mm is defined }}|{{ mm is string }}|{{ mm() is string }}|{{ mm is iterable }}"),
        ("tests", "{{ (s|safe) is string }}|{{ (s|safe) is sequence }}|{{ (l|tojson) is string }}"),
        ("tests", "{{ n is none and not q }}|{{ q is defined and q }}|{{ (not t)|string|length }}|{{ true and none }}"),
        # Filters.
        ("filters", "{{ l|length }}|{{ s|length }}|{{ d|length }}|{{ e|length }}|{{ es|length }}|{{ q|length }}|{{ m|length - 1 }}"),
        ("filters", "{{ n|length }}"),
        ("filters", "{{ i|length }}"),
        ("filters", "{{ t|length }}"),
        ("filters", "{{ (d|items)|length }}"),
        ("filters", "[{{ '  a b \n'|trim }}]|[{{ ws|trim }}]|[{{ nb|trim }}]|[{{ es|trim }}]|[{{ ' \t\n\r\x0b\x0c '|trim }}]"),
        ("filters", "[{{ i|trim }}]|[{{ n|trim }}]|[{{ q|trim }}]|[{{ t|trim }}]|[{{ f|trim }}]|[{{ l|trim }}]"),
        ("filters", "[{{ '\u200b a \ufeff'|trim }}]|[{{ '\u2028 b \u0085'|trim }}]|[{{ '\u1680\u2000\u200a\u202f\u205f c'|trim }}]"),
        ("filters", "{{ n|string }}|{{ f|string }}|{{ big|string }}|{{ small|string }}|{{ whole|string }}|{{ t|string }}|{{ fl|string }}|{{ i|string }}|{{ s|string }}|{{ q|string }}|"),
        ("filters", "{{ l|string }}|{{ d|string }}|{{ m|string }}|{{ n|string is string }}|{{ i|string + 'x' }}|{{ n|string|length }}"),
        ("filters", "{{ s|safe }}|{{ '<b>&'|safe }}|{{ l|tojson|safe }}|{{ (s|safe) ~ '<' }}|{{ i|safe }}|{{ n|safe }}|{{ (s|safe)|length }}"),
        ("filters", "{{ (s|safe)|string }}|{{ (' a '|safe)|trim }}|{{ (s|safe) == s }}|{{ s|safe|tojson }}|{{ (s|safe)|string|safe }}"),
        ("filters", "{% for k, v in d|items %}{{ k }}={{ v }};{% endfor %}|{% for k, v in e|items %}x{% endfor %}|{% for k, v in q|items %}x{% endfor %}"),
        ("filters", "{% for k, v in m[1]|items %}{{ k }}:{{ v is string }};{% endfor %}"),
        ("filters", "{% for k, v in n|items %}x{% endfor %}"),
        ("filters", "{% for k, v in l|items %}x{% endfor %}"),
        ("filters", "{% for k, v in s|items %}x{% endfor %}"),
        ("filters", "{% set unused = n|items %}{% if l|items %}lazy{% endif %}"),
        ("filters", "{{ l|tojson }}|{{ one|tojson }}|{{ d|tojson }}|{{ e|tojson }}|{{ el|tojson }}|{{ n|tojson }}|{{ t|tojson }}|{{ s|tojson }}|{{ i|tojson }}"),
        ("filters", "{{ f|tojson }}|{{ zf|tojson }}|{{ big|tojson }}|{{ small|tojson }}|{{ whole|tojson }}|{{ m|tojson }}|{{ quotes|tojson }}"),
        ("filters", "{{ six[::-1]|tojson }}|{{ s.split(',')|tojson }}|{{ (l + one)|tojson }}|{{ d.a|tojson }}|{{ m[0].content|tojson }}"),
        ("filters", "{{ quotes | tojson(indent=2) }}|{{ m | tojson(indent=1) }}|{{ e | tojson(indent=2) }}|{{ l | tojson(indent=0) }}"),
        ("filters", "{{ quotes | tojson(ensure_ascii=true) }}|{{ quotes | tojson(ensure_ascii=false) }}|{{ l | tojson(indent=none) }}"),
        ("filters", "{{ q|tojson }}"),
        ("filters", "{{ d.q|tojson }}"),
        ("filters", "{{ (d|items)|tojson }}"),
        ("filters", "{{ namespace(a=1)|tojson }}"),
        ("filters", "{% for pair in d|items %}{{ pair|tojson }}{{ pair[0] }}{{ pair|length }}{% endfor %}"),
        ("filters", "{{ s|trim()|length }}|{{ s | trim | length }}|{{ s|length|string|length }}|{{ (s|trim) + '!' }}"),
        # if / elif / else.
        ("if", "|".join(falsy(x) for x in ["e", "d", "el", "l", "z", "i", "es", "s", "n", "q", "f", "zf", "t", "fl"])),
        ("if", "|".join(falsy(x) for x in ["d.q", "d.a", "m[5]", "namespace()", "s|safe", "es|safe", "d|items", "e|items", "-0", "'0'"])),
        ("if", "{% if fl %}a{% elif t %}b{% else %}c{% endif %}|{% if fl %}a{% elif fl %}b{% else %}c{% endif %}|{% if t %}a{% elif t %}b{% endif %}|{% if fl %}a{% endif %}|{% if fl %}a{% elif fl %}b{% elif t %}c{% endif %}"),
        ("if", "{% if t %}{% if fl %}a{% else %}b{% endif %}{% else %}c{% endif %}"),
        ("if", "{% if q.x.y %}a{% endif %}"),
        ("if", "{% if fl %}{{ q.x.y }}{% endif %}ok"),
        ("if", "{% if not(s.startswith('x') and s.endswith('b')) %}a{% endif %}"),
        # for loops.
        ("for", "{% for x in l %}{{ x }};{% endfor %}|{% for x in el %}{{ x }};{% endfor %}|{% for x in q %}{{ x }};{% endfor %}|{% for x in s %}{{ x }};{% endfor %}"),
        ("for", "{% for x in d %}{{ x }};{% endfor %}|{% for x in 'é天' %}{{ x }};{% endfor %}|{% for x in six[::-2] %}{{ x }};{% endfor %}"),
        ("for", "{% for x in l %}{{ loop.index }}{{ loop.index0 }}{{ loop.first }}{{ loop.last }}{{ loop.length }}{{ loop.revindex }}{{ loop.revindex0 }};{% endfor %}"),
        ("for", "{% for x in six %}{{ loop.previtem }}-{{ loop.nextitem }};{% endfor %}"),
        ("for", "{% for x in l %}{{ loop.previtem is defined }}{{ loop.nextitem is defined }};{% endfor %}"),
        ("for", "{% for x in m %}{% if loop.previtem and loop.previtem.role != 'tool' %}p{% endif %}{% if not loop.last and loop.nextitem.role != 'tool' %}n{% endif %};{% endfor %}"),
        ("for", "{% for x in l %}{{ loop.nextitem.role }}{% endfor %}"),
        ("for", "{% for x in l %}{{ loop.nope is defined }}{% endfor %}"),
        ("for", "{% for a, b in d|items %}{{ a }}{{ b }};{% endfor %}"),
        ("for", "{% for a, b in pairs %}{{ a }}{{ b }};{% endfor %}", {"pairs": [[1, 2], ["x", None], "yz"]}),
        ("for", "{% for a, b in pairs %}{{ a }}{% endfor %}", {"pairs": [[1, 2, 3]]}),
        ("for", "{% for a, b in pairs %}{{ a }}{% endfor %}", {"pairs": [[1]]}),
        ("for", "{% for a, b in l %}{{ a }}{% endfor %}"),
        ("for", "{% for a, b in d %}{{ a }}{% endfor %}"),
        ("for", "{% for x in n %}{{ x }}{% endfor %}"),
        ("for", "{% for x in i %}{{ x }}{% endfor %}"),
        ("for", "{% for x in t %}{{ x }}{% endfor %}"),
        ("for", "{% for x in q.y %}{{ x }}{% endfor %}"),
        ("for", "{% for x in l %}{% for y in one %}{{ x }}{{ y }}{{ loop.index }}{{ loop.length }};{% endfor %}{{ loop.index }}|{% endfor %}"),
        ("for", "{% set v = 1 %}{% for x in l %}{% set v = v + x %}{{ v }};{% endfor %}{{ v }}"),
        ("for", "{% for x in l %}{% set inner = x %}{% endfor %}{{ inner is defined }}|{{ x is defined }}|{{ loop is defined }}"),
        ("for", "{% set x = 'outer' %}{% for x in l %}{{ x }}{% endfor %}{{ x }}"),
        ("for", "{% for x in l %}{% if x == 1 %}{% set seen = 'one' %}{% endif %}{{ seen }};{% endfor %}"),
        ("for", "{% for x in l %}{% set x = x + 10 %}{{ x }};{% endfor %}"),
        ("for", "{% for tc in calls %}{% if tc.function is defined %}{% set tc = tc.function %}{% endif %}{{ tc.name }};{% endfor %}", {"calls": [{"function": {"name": "a"}}, {"name": "b"}]}),
        ("for", "{% for x in l %}{{ s }}{{ loop.index0 > i }}{{ loop.index0 == 0 }};{% endfor %}"),
        ("for", "{{ loop is defined }}"),
        ("for", "{{ loop.index is defined }}"),
        ("for", "{% for x in l -%}\n  {{ x }}\n{%- endfor %}"),
        ("for", "{% for x in l %}\n  {{- x }}\n{% endfor %}"),
        ("for", "{% for pair in d|items %}{{ pair[0] }}{{ pair[-1] }}{{ pair[5] is defined }}{{ pair|length }}{{ pair == pair }}{{ 'z' in pair }}{{ pair[::-1] }}{{ pair[1:] }};{% endfor %}"),
        ("for", "{% for pair in d|items %}{{ pair is sequence }}{{ pair is iterable }}{{ pair is mapping }}{{ pair is string }}{% if pair %}T{% endif %};{% endfor %}"),
        ("for", "{% for a, b in d|items %}{{ loop.previtem }}{{ loop.nextitem }};{% endfor %}"),
        ("for", "{% for a, b, c in triples %}{{ a }}{{ b }}{{ c }};{% endfor %}", {"triples": [[1, 2, 3], "xyz"]}),
        ("for", "{% for x in l %}{% for y in l %}{% if y > x %}{{ x }}<{{ y }} {% endif %}{% endfor %}{% endfor %}"),
        # set and namespace.
        ("set", "{% set a = 1 %}{% set b = a + 1 %}{{ a }}{{ b }}"),
        ("set", "{% set a = 1 %}{% if t %}{% set a = 2 %}{% set c = 3 %}{% endif %}{{ a }}{{ c }}"),
        ("set", "{% if fl %}{% set c = 3 %}{% endif %}{{ c is defined }}"),
        ("set", "{% set s = s ~ '!' %}{{ s }}|{% set i = none %}{{ i }}|{% set u = q %}{{ u is defined }}"),
        ("set", "{% set ns = namespace(a=1, b='x') %}{% for x in l %}{% set ns.a = ns.a + x %}{% endfor %}{{ ns.a }}{{ ns.b }}"),
        ("set", "{% set ns = namespace() %}{{ ns.missing is defined }}|{% set ns.added = 1 %}{{ ns.added }}"),
        ("set", "{% set ns = namespace(found=false) %}{% for x in l %}{% if x == 2 %}{% set ns.found = true %}{% set ns.at = loop.index0 %}{% endif %}{% endfor %}{{ ns.found }}{{ ns.at }}"),
        ("set", "{% set ns = namespace(a=1) %}{% set ns.a = ns.a.q.r %}"),
        ("set", "{% set d.x = 1 %}{{ d.x }}"),
        ("set", "{% set q.x = 1 %}"),
        ("set", "{% set n.x = 1 %}"),
        ("set", "{% set v = q.x.y %}"),
        ("set", "{% set a = 'x' if fl else 'y' %}{{ a }}|{% set b = l|tojson|safe if l is mapping or (l is sequence and l is not string) else l|string %}{{ b }}"),
        ("set", "{% set c = messages|length - 1 %}{{ c }}|{% set t2 = ' a '|trim %}[{{ t2 }}]"),
        # Macros.
        ("macro", "{% macro mm(a, b=2, c=false) %}{{ a }}{{ b }}{{ c }}{% endmacro %}{{ mm(1) }}|{{ mm(1, 3) }}|{{ mm(1, c=true) }}|{{ mm(1, 3, 4) }}|{{ mm(b=5, a=6) }}"),
        ("macro", "{% macro mm(a) %}[{{ a is defined }}{{ a }}]{% endmacro %}{{ mm() }}|{{ mm(none) }}|{{ mm(q) }}"),
        ("macro", "{% macro mm(a) %}{{ a.x }}{% endmacro %}{{ mm() }}"),
        ("macro", "{% macro mm(a) %}{{ a }}{% endmacro %}{{ mm(1, 2) }}"),
        ("macro", "{% macro mm(a) %}{{ a }}{% endmacro %}{{ mm(1, z=2) }}"),
        ("macro", "{% macro mm(a) %}{{ a }}{% endmacro %}{{ mm(1, a=2) }}"),
        ("macro", "{% macro mm() %}  x {{ s }} {% endmacro %}[{{ mm() }}]|[{{ mm()|trim }}]|{{ mm() is string }}|{{ mm()|length }}"),
        ("macro", "{% set count = namespace(value=0) %}{% macro bump(by) %}{% set count.value = count.value + by %}{{ count.value }}{% endmacro %}{{ bump(1) }}{{ bump(2) }}|{{ count.value }}"),
        ("macro", "{% set top = 'T' %}{% macro mm() %}{{ top }}{{ s }}{{ later is defined }}{% endmacro %}{% set later = 1 %}{{ mm() }}"),
        ("macro", "{% macro mm(s) %}{% set s = s ~ '!' %}{% set leaked = 1 %}{{ s }}{% endmacro %}{{ mm('arg') }}|{{ s }}|{{ leaked is defined }}"),
        ("macro", "{% macro mm(x) %}{{ x }}{% endmacro %}{% for x in l %}{{ mm(x + 1) }}{% endfor %}"),
        ("macro", "{% macro mm() %}{{ x is defined }}{% endmacro %}{% for x in one %}{{ mm() }}{% endfor %}"),
        ("macro", "{% macro outer(a) %}<{{ inner(a) }}>{% endmacro %}{% macro inner(a) %}{{ a ~ a }}{% endmacro %}{{ outer('x') }}"),
        ("macro", "{{ later() }}{% macro later() %}x{% endmacro %}"),
        ("macro", "{% macro mm(a, flag=false) %}{% if flag %}{{ raise_exception('flagged ' ~ a) }}{% endif %}{{ a }}{% endmacro %}{{ mm('x') }}{% set y = mm('y', true)|trim %}{{ y }}"),
        ("macro", "{% macro mm(content, is_system=false) %}{% if content is string %}{{ content }}{% elif content is iterable and content is not mapping %}{% for item in content %}{% if 'text' in item %}{{ item.text }}{% endif %}{% endfor %}{% elif content is none or content is undefined %}{{ '' }}{% else %}{{ raise_exception('bad') }}{% endif %}{% endmacro %}[{{ mm('s') }}][{{ mm(parts) }}][{{ mm(none) }}][{{ mm(q) }}]", {"parts": [{"text": "a"}, {"type": "x"}, {"text": "b"}]}),
        ("macro", "{% macro mm(a=s, b=i + 1) %}{{ a }}{{ b }}{% endmacro %}{{ mm() }}|{{ mm(b=0) }}"),
        ("macro", "{% macro down(n) %}{{ n }}{% if n > 0 %}{{ down(n - 1) }}{% endif %}{% endmacro %}{{ down(3) }}"),
        ("macro", "{% macro forever(n) %}{{ forever(n) }}{% endmacro %}{{ forever(1) }}"),
        ("macro", "{% macro mm(a) %}{{ a }}{% endmacro %}{% set mm2 = mm %}{{ mm2('via alias') }}|{{ mm == mm2 }}|{{ mm == s }}"),
        # String methods.
        ("methods", "{{ s.startswith('a') }}|{{ s.startswith('b') }}|{{ s.startswith('') }}|{{ s.endswith('b') }}|{{ s.endswith(',b') }}|{{ s.endswith('a') }}|{{ es.startswith('a') }}"),
        ("methods", "{{ s.split(',') }}|{{ s.split(';') }}|{{ s.split(',')[-1] }}|{{ s.split(';')[-1] }}|{{ s.split(',')[0] }}|{{ es.split(',') }}|{{ 'a,,b,'.split(',') }}|{{ 'abab'.split('ab') }}"),
        ("methods", "{{ think.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') }}|{{ think.split('</think>')[-1].lstrip('\\n') }}"),
        ("methods", "[{{ ' x '.strip() }}]|[{{ ' x '.lstrip() }}]|[{{ ' x '.rstrip() }}]|[{{ ws.strip() }}]|[{{ nb.strip() }}]|[{{ ws.lstrip() }}]|[{{ ws.rstrip() }}]"),
        ("methods", "{{ 'xxaxx'.strip('x') }}|{{ 'xxaxx'.lstrip('x') }}|{{ 'xxaxx'.rstrip('x') }}|{{ 'xyaxy'.strip('yx') }}|{{ '\\n\\na\\n'.lstrip('\\n') }}|{{ 'a'.strip('') }}|{{ 'éaé'.strip('é') }}"),
        ("methods", "{{ ' a '.strip(none) }}|{{ s.strip().split(',')|length }}|{{ m[0].content.startswith('h') }}"),
        ("methods", "{{ s.split('') }}"),
        ("methods", "{{ s.split(1) }}"),
        ("methods", "{{ s.startswith(1) }}"),
        ("methods", "{{ s.startswith() }}"),
        ("methods", "{{ s.strip(1) }}"),
        ("methods", "{{ s.strip('a', 'b') }}"),
        ("methods", "{{ i.strip() }}"),
        ("methods", "{{ n.split('x') }}"),
        ("methods", "{{ q.strip() }}"),
        ("methods", "{{ l.strip() }}"),
        ("methods", "{{ d.strip() }}"),
        ("methods", "{{ s.nope() }}"),
        ("methods", "{{ s.strip is defined }}|{{ s.nope is defined }}|{{ d.strip is defined }}"),
        # Calls and raise_exception.
        ("calls", "before {{ raise_exception('boom') }} after"),
        ("calls", "{% for x in l %}{% if x == 2 %}{{ raise_exception('no ' ~ x) }}{% endif %}{% endfor %}"),
        ("calls", "{{ raise_exception('first') }}{{ raise_exception('second') }}"),
        ("calls", "{{ raise_exception('é天 ' + s) }}"),
        ("calls", "{{ raise_exception(i) }}"),
        ("calls", "{{ raise_exception() }}"),
        ("calls", "{{ raise_exception('a', 'b') }}"),
        ("calls", "{% if fl %}{{ raise_exception('never') }}{% endif %}fine"),
        ("calls", "{{ s() }}"),
        ("calls", "{{ q() }}"),
        ("calls", "{{ n() }}"),
        ("calls", "{{ d() }}"),
        ("calls", "{{ d.q() }}"),
        ("calls", "{{ namespace(a=1).a }}|{{ namespace(a=l).a[0] }}"),
        ("calls", "{{ raise_exception(message='by keyword') }}"),
        ("calls", "{{ raise_exception(none) }}"),
        ("calls", "{{ raise_exception(l) }}"),
        ("calls", "{{ raise_exception(msg='wrong keyword') }}"),
        ("calls", "{{ namespace(a=1)['a'] }}|{{ namespace(a=1)['b'] is defined }}|{{ namespace(a=1)[0] is defined }}"),
        ("calls", "{{ nope('x') }}"),
        ("calls", "{{ strftime_now is defined }}|{{ raise_exception is defined }}|{{ namespace is defined }}"),
    ]
    return rows


def apply(tok, case):
    return tok.apply_chat_template(
        case["messages"],
        tools=case.get("tools"),
        add_generation_prompt=case.get("add_generation_prompt", True),
        tokenize=False,
        **case.get("kwargs", {}),
    )


def outcome(render):
    try:
        return {"prompt": render()}
    except Exception as e:  # the template called raise_exception, or failed to compile or render
        return {"error": f"{type(e).__name__}: {e}"}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for family, repo in FAMILIES.items():
        tok = AutoTokenizer.from_pretrained(snapshot_download(repo, local_files_only=True))
        rows = []
        for case in differential_cases() + random_cases():
            row = {
                "name": case["name"],
                "messages": case["messages"],
                "tools": case.get("tools"),
                "kwargs": case.get("kwargs", {}),
                "add_generation_prompt": case.get("add_generation_prompt", True),
            }
            row.update(outcome(lambda: apply(tok, case)))
            rows.append(row)
        write_jsonl(OUT / f"differential_{family}.jsonl", rows)
        print(f"{family}: {len(rows)} rows, {sum('error' in r for r in rows)} errors")

    rows = []
    counts = {}
    for group, source, *extra in semantics_cases():
        counts[group] = counts.get(group, 0) + 1
        kwargs = {**VARIABLES, **(extra[0] if extra else {})}
        row = {"name": f"{group}/{counts[group]:02d}", "source": source, "kwargs": kwargs}
        row.update(
            outcome(
                lambda: _compile_jinja_template(source).render(
                    messages=[USER_HI], tools=None, documents=None, add_generation_prompt=False, **kwargs
                )
            )
        )
        rows.append(row)
    write_jsonl(OUT / "semantics.jsonl", rows)
    print(f"semantics: {len(rows)} rows, {sum('error' in r for r in rows)} errors")


if __name__ == "__main__":
    main()
