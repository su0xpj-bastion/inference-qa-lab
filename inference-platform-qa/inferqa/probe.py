"""probe — one streamed chat completion against any OpenAI-compatible endpoint.

Works unchanged against mockserve, the local lb, an ingress-nginx host in
a kind cluster, or the Cerebras public API (https://api.cerebras.ai/v1);
only base_url and the key change.

Usage:
    python -m inferqa.probe --base-url http://127.0.0.1:8080/v1 "hello world"
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field

import openai

from inferqa.metrics import StreamTiming, stream_timing

MODEL = "llama3.1-8b"
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 0.1  # 2**attempt + jitter, scaled; production would use 1.0


@dataclass
class ProbeResult:
    ok: bool
    timing: StreamTiming | None = None
    content: str = ""
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    attempts: int = 0
    served_by: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["timing"] = asdict(self.timing) if self.timing else None
        return out


def stream_chat(
    client: openai.OpenAI,
    prompt: str,
    *,
    model: str = MODEL,
    max_tokens: int = 64,
    extra_headers: dict[str, str] | None = None,
    echo: bool = False,
) -> ProbeResult:
    """Stream one completion, retrying 429s with jittered backoff.

    Contract checks performed while streaming (each one is a real
    production failure mode seen behind proxies):
      * stream must end with a finish_reason   — else truncated stream
      * usage chunk must arrive when requested — else billing gap
      * usage.completion_tokens must match the chunks counted
    """
    usage = None
    finish_reason: str | None = None
    result = ProbeResult(ok=False)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        result.attempts = attempt
        content_parts: list[str] = []
        token_times: list[float] = []
        usage = None
        finish_reason = None
        start = time.monotonic()
        try:
            raw = client.chat.completions.with_raw_response.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_headers=extra_headers or {},
            )
            result.served_by = raw.headers.get("x-served-by")
            with raw.parse() as stream:
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token_times.append(time.monotonic())
                        content_parts.append(chunk.choices[0].delta.content)
                        if echo:
                            print(chunk.choices[0].delta.content, end="", flush=True, file=sys.stderr)
                    if chunk.choices and chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                    if chunk.usage:
                        usage = chunk.usage
            break
        except openai.RateLimitError:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(BACKOFF_BASE_S * (2 ** attempt + random.random()))
        except openai.APIStatusError:
            raise
        except openai.APIError:
            raise

    end = time.monotonic()
    if echo:
        print(file=sys.stderr)
    result.timing = stream_timing(start, token_times, end)
    result.content = "".join(content_parts)
    result.finish_reason = finish_reason

    if finish_reason is None:
        result.warnings.append("stream ended without finish_reason (truncated?)")
    elif finish_reason != "stop":
        result.warnings.append(f"[WARN] stream ended with finish_reason={finish_reason}")
    if usage is None:
        result.warnings.append("usage chunk missing despite include_usage=True")
    else:
        result.prompt_tokens = usage.prompt_tokens
        result.completion_tokens = usage.completion_tokens
        if usage.completion_tokens != len(token_times):
            result.warnings.append(
                f"usage.completion_tokens={usage.completion_tokens} but {len(token_times)} chunks streamed"
            )
    result.ok = finish_reason is not None and usage is not None
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stream one completion and print timing JSON")
    parser.add_argument("prompt")
    parser.add_argument("--base-url", default=os.environ.get("INFERQA_BASE_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args(argv)

    client = openai.OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get("CEREBRAS_API_KEY", "not-needed-for-mock"),
        max_retries=0,  # retries are ours, so attempts are countable
    )
    try:
        result = stream_chat(client, args.prompt, model=args.model, max_tokens=args.max_tokens, echo=True)
    except openai.RateLimitError:
        print("[ERROR] rate limited after retries", file=sys.stderr)
        return 2
    except openai.APIStatusError as exc:
        print(f"[ERROR] HTTP {exc.status_code}: {exc.message}", file=sys.stderr)
        return 2
    except openai.APIError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
