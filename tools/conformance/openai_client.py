"""Check an OpenAI-compatible server with the official openai Python package.

    python tools/conformance/openai_client.py http://127.0.0.1:8080/v1

The argument is the client's base URL, including /v1. The script lists the
models, then runs a chat completion, a streamed one and a streamed one with a
stop string, all asking the model to count from 1 to 5. It checks what an
OpenAI client relies on: ids, object names, roles, finish reasons, usage and
the framing of the stream. It exits non-zero with a message on the first check
that fails.
"""

import sys

from openai import OpenAI

MESSAGES = [{"role": "user", "content": "Count from 1 to 5."}]


def check(condition, message):
    if not condition:
        sys.exit(f"FAIL: {message}")


def check_models(client):
    models = client.models.list().data
    check(len(models) >= 1, "models.list() returned no models")
    for model in models:
        check(isinstance(model.id, str) and model.id, f"model without an id: {model!r}")
        check(model.object == "model", f"model object is {model.object!r}, want 'model'")
    print(f"ok models: {[m.id for m in models]}")
    return models[0].id


def check_chat(client, model):
    resp = client.chat.completions.create(model=model, messages=MESSAGES, max_tokens=24, temperature=0)
    check(resp.id.startswith("chatcmpl-"), f"chat id is {resp.id!r}, want a chatcmpl- id")
    check(resp.object == "chat.completion", f"chat object is {resp.object!r}")
    check(resp.model == model, f"chat model is {resp.model!r}, want {model!r}")
    check(len(resp.choices) == 1, f"chat has {len(resp.choices)} choices, want 1")
    choice = resp.choices[0]
    check(choice.message.role == "assistant", f"chat role is {choice.message.role!r}")
    check(isinstance(choice.message.content, str) and choice.message.content, "chat content is empty")
    check(choice.finish_reason in ("stop", "length"), f"chat finish_reason is {choice.finish_reason!r}")
    usage = resp.usage
    check(usage is not None, "chat has no usage")
    check(usage.prompt_tokens > 0 and usage.completion_tokens > 0, f"chat usage is {usage!r}")
    check(
        usage.total_tokens == usage.prompt_tokens + usage.completion_tokens,
        f"chat total_tokens is not the sum: {usage!r}",
    )
    print(f"ok chat: {choice.message.content!r} ({choice.finish_reason})")


def read_stream(stream, label):
    """Consume a chat stream. Return (text, finish_reason, usage)."""
    ids, text, finish, usage = set(), "", None, None
    for chunk in stream:
        ids.add(chunk.id)
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        check(finish is None, f"{label}: a chunk with a choice follows the finish_reason")
        check(chunk.object == "chat.completion.chunk", f"{label}: chunk object is {chunk.object!r}")
        choice = chunk.choices[0]
        text += choice.delta.content or ""
        finish = choice.finish_reason
    check(len(ids) == 1, f"{label}: chunks carry {len(ids)} different ids, want 1")
    check(finish in ("stop", "length"), f"{label}: final finish_reason is {finish!r}")
    return text, finish, usage


def check_stream(client, model):
    stream = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        max_tokens=24,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    text, finish, usage = read_stream(stream, "stream")
    check(text, "stream produced no text")
    check(usage is not None, "stream has no usage chunk although include_usage was set")
    check(
        usage.total_tokens == usage.prompt_tokens + usage.completion_tokens,
        f"stream total_tokens is not the sum: {usage!r}",
    )
    print(f"ok stream: {text!r} ({finish})")


def check_stream_stop(client, model):
    stream = client.chat.completions.create(
        model=model, messages=MESSAGES, max_tokens=24, temperature=0, stream=True, stop=["3"]
    )
    text, finish, usage = read_stream(stream, "stream with stop")
    check(text, "stream with stop produced no text")
    check("3" not in text, f"stream with stop sent the stop string: {text!r}")
    check(finish == "stop", f"stream with stop finished with {finish!r}, want 'stop'")
    check(usage is None, "stream with stop has a usage chunk although none was asked for")
    print(f"ok stream with stop: {text!r} ({finish})")


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <base_url>   e.g. http://127.0.0.1:8080/v1")
    client = OpenAI(base_url=sys.argv[1], api_key="not-needed", timeout=120, max_retries=0)
    model = check_models(client)
    check_chat(client, model)
    check_stream(client, model)
    check_stream_stop(client, model)
    print("all checks passed")


if __name__ == "__main__":
    main()
