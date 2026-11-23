"""Vendor the two external sources this harness uses, pinned to a commit.

  python scripts/vendor_sources.py <openapi.json> <openapi_sha> <gsm8k_test.jsonl> <gsm_sha>

1. openai/openai-openapi -> ies/data/openai_chat_schemas.json
   Only CreateChatCompletionResponse, CreateChatCompletionStreamResponse and every schema
   they reference (the full spec is 5 MB).
2. openai/grade-school-math (MIT) -> ies/data/gsm8k_slice.jsonl
   A fixed, seeded 50-item slice of the test split: question + numeric answer.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOTS = ("CreateChatCompletionResponse", "CreateChatCompletionStreamResponse")
DATA = Path(__file__).resolve().parent.parent / "ies" / "data"
SLICE = 50
SEED = 2026


def _refs(node: object, out: set[str]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            out.add(ref.rsplit("/", 1)[1])
        for v in node.values():
            _refs(v, out)
    elif isinstance(node, list):
        for v in node:
            _refs(v, out)


def _strip_meta(node: object) -> object:
    if isinstance(node, dict):
        return {k: _strip_meta(v) for k, v in node.items() if not k.startswith("x-")}
    if isinstance(node, list):
        return [_strip_meta(v) for v in node]
    return node


def vendor_schemas(spec_path: Path, sha: str) -> None:
    with spec_path.open("r", encoding="utf-8") as fh:
        schemas = json.load(fh)["components"]["schemas"]
    keep: set[str] = set()
    todo = list(ROOTS)
    while todo:
        name = todo.pop()
        if name in keep:
            continue
        keep.add(name)
        found: set[str] = set()
        _refs(schemas[name], found)
        todo += sorted(found - keep)
    out = {"source": f"https://github.com/openai/openai-openapi/blob/{sha}/openapi.json",
           "roots": list(ROOTS),
           "components": {"schemas": {k: _strip_meta(schemas[k]) for k in sorted(keep)}}}
    (DATA / "openai_chat_schemas.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"schemas: {len(keep)} kept from {len(schemas)}")


def vendor_gsm8k(test_path: Path, sha: str) -> None:
    with test_path.open("r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    picked = sorted(random.Random(SEED).sample(range(len(rows)), SLICE))
    with (DATA / "gsm8k_slice.jsonl").open("w", encoding="utf-8") as fh:
        for i in picked:
            answer = rows[i]["answer"].split("####")[-1].strip().replace(",", "")
            fh.write(json.dumps({"id": f"gsm8k-test-{i}", "question": rows[i]["question"], "answer": answer}) + "\n")
    (DATA / "GSM8K_SOURCE.md").write_text(
        f"50-item slice (seed {SEED}) of the GSM8K test split from "
        f"https://github.com/openai/grade-school-math/tree/{sha} (MIT License, Copyright (c) 2021 OpenAI).\n",
        encoding="utf-8")
    print(f"gsm8k: {SLICE} of {len(rows)}")


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    vendor_schemas(Path(sys.argv[1]), sys.argv[2])
    vendor_gsm8k(Path(sys.argv[3]), sys.argv[4])
