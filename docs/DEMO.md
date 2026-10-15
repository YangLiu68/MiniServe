# Demo walkthrough (Milestone 1)

Everything below was actually run on 2026-10-14 (Apple M4, macOS 15.3.2, SmolLM2-135M-Instruct
on MPS fp16). Outputs are copied from that session, lightly trimmed (`…`). Latency numbers in
them are **not** benchmarks — the host was saturated by other processes (see BENCHMARK.md).

## 1. Safety: refuse a public bind without a key

```console
$ MINISERVE_HOST=0.0.0.0 uv run miniserve
miniserve: Refusing to bind '0.0.0.0' without MINISERVE_API_KEY. Set an API key, bind
127.0.0.1, or set MINISERVE_ALLOW_UNAUTHENTICATED_PUBLIC=1 if a trusted gateway in front of
MiniServe handles auth.
$ echo $?
2
```

## 2. Start the server and watch readiness

```console
$ MINISERVE_PORT=8077 MINISERVE_API_KEY=demo-key uv run miniserve
$ curl -s localhost:8077/readyz          # while loading (HTTP 503)
{"status":"loading","error":null,"backend":"hf","model":null,…}
$ curl -s localhost:8077/readyz          # after load + warmup (HTTP 200)
{"status":"ready","error":null,"backend":"hf","model":"HuggingFaceTB/SmolLM2-135M-Instruct",
 "device":"mps","dtype":"float16","load_s":127.594,"active_requests":0,"version":"0.1.0"}
```

## 3. Auth and a JSON completion

```console
$ curl -s localhost:8077/v1/completions -H 'Content-Type: application/json' -d '{"prompt":"hi"}'
{"error":{"message":"invalid or missing API key","type":"authentication_error","code":null}}

$ curl -s localhost:8077/v1/completions -H 'Authorization: Bearer demo-key' \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"The three primary colors are","max_tokens":24,"temperature":0}'
{
  "id": "cmpl-6b3f588b751c46bd9cae1463",
  "object": "text_completion",
  "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "choices": [{"index": 0,
    "text": " the primary colors, which are the three primary colors: red, blue, and yellow. These colors are the foundation of",
    "finish_reason": "length", "logprobs": null}],
  "usage": {"prompt_tokens": 5, "completion_tokens": 24, "total_tokens": 29},
  "miniserve": {"timings": {"queue_ms": 0.45, "ttft_ms": 1635.191, "total_ms": 12818.749,
                            "decode_tokens_per_s": 2.06}, "error": null}
}
```

## 4. Streaming (SSE)

```console
$ curl -sN localhost:8077/v1/completions -H 'Authorization: Bearer demo-key' \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"List three fruits:","max_tokens":12,"temperature":0,"stream":true}'
data: {"id": "cmpl-a919d4e74d564bf99ee0a3e0", "object": "text_completion", …, "choices": [{"index": 0, "text": …, "finish_reason": null, …}]}
…                                   (12 content chunks in total, one per stable text delta)
data: {…, "choices": [{…, "finish_reason": "length"}], "usage": {"prompt_tokens": 4, "completion_tokens": 12, …}, "miniserve": {…}}
data: [DONE]
```

## 5. Client disconnect cancels generation

Ask for 400 tokens and hang up after 6 seconds:

```console
$ curl -sN --max-time 6 localhost:8077/v1/completions -H 'Authorization: Bearer demo-key' \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Once upon a time","max_tokens":400,"temperature":0,"stream":true}' | grep -c '^data:'
5
```

Server log — generation stopped at 5 of 400 tokens:

```
INFO miniserve.engine: request finished id=cmpl-945c4242… reason=cancelled prompt_tokens=4 completion_tokens=5 …
```

Hanging up *during* a long prefill (client gave up at 2 s, the step took ~14 s on the loaded
host) is reported as `cancelled` with 0 tokens — the step's result is discarded:

```
INFO miniserve.engine: request finished id=cmpl-43712b48… reason=cancelled prompt_tokens=6 completion_tokens=0 …
```

## 6. Graceful stop

```console
$ kill -TERM <pid>
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
```

## Try it without downloading anything

```bash
MINISERVE_BACKEND=toy MINISERVE_TOY_DECODE_MS_PER_TOKEN=50 uv run miniserve
curl -sN localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"héllo 🙂 streaming","stream":true}'
```

The toy backend echoes the prompt one byte-token at a time; `é` and `🙂` arrive as single
deltas even though they span 2 and 4 tokens. Deltas from that run, in order:

```
'h' 'é' 'l' 'l' 'o' ' ' '🙂' ' ' 's' 't' 'r' 'e' 'a' 'm' 'i' 'n' 'g' ''(finish_reason=stop) [DONE]
```
