"""Long context: recall across prompt length and depth, and a gist-memory chain.

  needle_recall       a fact hidden at a given depth of an N-token haystack is recalled,
                      over a grid of (length, depth)
  context_accounting  prompt_tokens grows with the haystack (>= tolerance x expected), so
                      a silently clipped context is caught even when the needle sits early
                      enough to survive the clip
  gist_chain          ReadAgent-style (cerebras/inference-examples/gist): summarize each page
                      in one call, then answer from the summaries only. Checks the answer and
                      times the whole sequential chain (its speed is bounded by per-call latency)
"""
from __future__ import annotations

import random

from ..core import Check, Ctx, guarded, pct

W = "longctx"
NEEDLE = "The secret launch code is ORCA-5521."
WORDS = ("orbit", "signal", "harbor", "lantern", "meadow", "copper", "glacier", "pillar", "canvas", "ember",
         "willow", "summit", "vessel", "quartz", "falcon", "prairie", "beacon", "thistle", "ledger", "cobalt",
         "the", "a", "of", "and", "near", "under", "beside", "toward", "with", "without")


def haystack(tokens: int, depth: float, seed: int) -> str:
    rng = random.Random(seed)
    sentences = []
    while sum(len(s.split()) + 1 for s in sentences) < tokens:
        sentences.append(" ".join(rng.choice(WORDS) for _ in range(11)).capitalize() + ".")
    sentences.insert(int(len(sentences) * depth), NEEDLE)
    return " ".join(sentences)


def _needles(ctx: Ctx) -> list[dict]:
    if "_needles" in ctx.__dict__:
        return ctx.__dict__["_needles"]
    rows = []
    for i, (tokens, depth) in enumerate(ctx.slo["needle_grid"]):
        text = haystack(tokens, depth, seed=100 + i)
        c = ctx.chat([{"role": "user", "content": text + "\n\nWhat is the secret launch code? "
                                                          "Reply with only the code."}],
                     temperature=0, max_completion_tokens=16)
        rows.append({"tokens": tokens, "depth": depth, "prompt_tokens": c.prompt_tokens or 0,
                     "found": "ORCA-5521" in c.content.upper(), "answer": c.content.strip()[:40],
                     "prompt_time": c.prompt_time, "ttft_wall": c.wall})
    ctx.__dict__["_needles"] = rows
    return rows


def needle_recall(ctx: Ctx) -> Check:
    rows = _needles(ctx)
    missed = [(r["tokens"], r["depth"], r["answer"]) for r in rows if not r["found"]]
    grid = ", ".join(f"{r['tokens'] // 1000}K@{r['depth']:.0%}{'✓' if r['found'] else '✗'}" for r in rows)
    return Check(W, "needle_recall", "fail" if missed else "pass",
                 f"{len(rows) - len(missed)}/{len(rows)} found: {grid}" + (f"; missed {missed}" if missed else ""),
                 metrics={"rows": rows})


def context_accounting(ctx: Ctx) -> Check:
    rows = _needles(ctx)
    tol = ctx.slo["longctx_tokens_tolerance"]
    short = [(r["tokens"], r["prompt_tokens"]) for r in rows if r["prompt_tokens"] < tol * r["tokens"]]
    pre = [f"{r['prompt_tokens']:,} tok {(r['prompt_time'] or 0) * 1e3:.0f} ms" for r in rows]
    return Check(W, "context_accounting", "fail" if short else "pass",
                 (f"prompt_tokens below {tol:.0%} of haystack (context clipped?): {short}; " if short else
                  f"prompt_tokens >= {tol:.0%} of haystack at every size; ") + "prefill " + ", ".join(pre))


def _pages(n: int, page_tokens: int) -> list[str]:
    rng = random.Random(7)
    pages = []
    for k in range(1, n + 1):
        filler = " ".join(" ".join(rng.choice(WORDS) for _ in range(11)).capitalize() + "."
                          for _ in range(max(1, page_tokens // 12)))
        pages.append(f"Page {k}. {filler} On page {k} the archive vault number is V-{k * 7 + 3}. {filler[:200]}")
    return pages


def gist_chain(ctx: Ctx) -> Check:
    n, target = ctx.slo["gist_pages"], ctx.slo["gist_target_page"]
    pages = _pages(n, ctx.slo["gist_page_tokens"])
    gists, walls = [], []
    for k, page in enumerate(pages, 1):
        c = ctx.chat([{"role": "user", "content": "Summarize this page in one sentence. Keep the page number and "
                                                  f"every number exactly.\n\n{page}"}],
                     temperature=0, max_completion_tokens=60)
        gists.append(f"Page {k} summary: {c.content.strip()}")
        walls.append(c.wall)
    q = ctx.chat([{"role": "user", "content": "Using only these page summaries:\n" + "\n".join(gists) +
                   f"\n\nWhat is the archive vault number on page {target}? Reply with only the vault number."}],
                 temperature=0, max_completion_tokens=16)
    walls.append(q.wall)
    want = f"V-{target * 7 + 3}"
    ok = want in q.content.upper()
    kept = sum(f"V-{k * 7 + 3}" in g.upper() for k, g in enumerate(gists, 1))
    return Check(W, "gist_chain", "pass" if ok else "fail",
                 f"answer {q.content.strip()!r} (want {want}); {kept}/{n} gists kept their number; "
                 f"chain {sum(walls) * 1e3:.0f} ms over {len(walls)} sequential calls "
                 f"(p50 {pct(walls, 50) * 1e3:.0f} ms/call)",
                 metrics={"chain_s": sum(walls), "gists_kept": kept})


CHECKS = (needle_recall, context_accounting, gist_chain)


def run(ctx: Ctx) -> list[Check]:
    return [guarded(W, fn.__name__, lambda fn=fn: fn(ctx)) for fn in CHECKS]
