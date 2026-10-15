"""Stream a chat completion from MiniServe with the official OpenAI SDK.

MINISERVE_API_KEY=demo-key uv run miniserve          # in another terminal
uv run python examples/openai_chat.py "Explain the KV cache in two sentences."
"""

import os
import sys

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("MINISERVE_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.environ.get("MINISERVE_API_KEY", "demo-key"),
)
model = client.models.list().data[0].id
question = " ".join(sys.argv[1:]) or "What is the capital of France?"
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": question}],
    max_tokens=128,
    temperature=0,
    stream=True,
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    if chunk.usage:  # MiniServe sends usage (incl. prefix-cached tokens) in the last chunk
        cached = getattr(chunk.usage.prompt_tokens_details, "cached_tokens", 0) or 0
        print(
            f"\n\n[{chunk.usage.prompt_tokens} prompt tokens ({cached} cached), "
            f"{chunk.usage.completion_tokens} generated]"
        )
