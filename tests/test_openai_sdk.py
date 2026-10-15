"""Compatibility with the official ``openai`` Python SDK (not just our own client)."""

import pytest

from miniserve.backends.toy import ToyBackend

from conftest import live_server

openai = pytest.importorskip("openai")


def test_official_sdk_completions_chat_streaming_models_and_errors():
    with live_server(ToyBackend(), api_key="sk-test") as (url, _):
        client = openai.OpenAI(base_url=f"{url}/v1", api_key="sk-test", max_retries=0)

        r = client.completions.create(model="toy-echo", prompt="hello sdk", max_tokens=20)
        assert r.choices[0].text == "hello sdk"
        assert r.choices[0].finish_reason == "stop"
        assert r.usage.completion_tokens == 9

        stream = client.completions.create(
            model="toy-echo", prompt="stream me", max_tokens=4, stream=True
        )
        chunks = list(stream)
        assert "".join(c.choices[0].text for c in chunks) == "stre"
        assert chunks[-1].choices[0].finish_reason == "length"

        msgs = [{"role": "user", "content": "hi"}]
        chat = client.chat.completions.create(model="toy-echo", messages=msgs, max_tokens=50)
        assert chat.choices[0].message.role == "assistant"
        assert chat.choices[0].message.content == "user: hi\nassistant: "

        stream = client.chat.completions.create(
            model="toy-echo", messages=msgs, max_tokens=50, stream=True
        )
        text = "".join((c.choices[0].delta.content or "") for c in stream if c.choices)
        assert text == "user: hi\nassistant: "

        assert [m.id for m in client.models.list()] == ["toy-echo"]

        with pytest.raises(openai.AuthenticationError):
            openai.OpenAI(base_url=f"{url}/v1", api_key="wrong", max_retries=0).completions.create(
                model="toy-echo", prompt="x", max_tokens=1
            )
        with pytest.raises(openai.NotFoundError):
            client.completions.create(model="gpt-4", prompt="x", max_tokens=1)
        with pytest.raises(openai.BadRequestError):
            client.completions.create(model="toy-echo", prompt="x", max_tokens=10**6)
