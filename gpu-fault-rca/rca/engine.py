"""Cross-layer RCA: correlate candidates, effects and symptoms into root causes.

1. Detect user-visible symptoms (router) and collect candidates/effects (detectors).
2. Lagging-rank analysis: in a hung NCCL group the ranks that *report* the
   watchdog timeout are the healthy ones. The silent ranks are the suspects.
3. Walk each effect and symptom back through the causal graph to the candidate
   that can produce it, in the same replica, with onset no later than the effect.
   Ties go to the stronger and deeper (closer to hardware) candidate.
4. Score: base confidence per kind + independent sources + "explains user impact".
   Root cause = explains a symptom and has confidence >= threshold.
"""
from __future__ import annotations

from pathlib import Path

from rca import catalog
from rca.bundle import Bundle, load
from rca.detectors import Candidate, Effect, Evidence, collect
from rca.symptoms import Symptom, detect

THRESHOLD = 0.7
SKEW_S = 30.0                 # metric scrape + log shipping skew between layers
SOURCE_BONUS = 0.1
IMPACT_BONUS = 0.1
MAX_CONFIDENCE = 0.99
CAPACITY_LOSS = {"node_loss", "gpu_lost", "ecc_uncorrectable", "network_degraded", "driver_mismatch",
                 "gsp_error", "memory_leak", "bad_build", "nvlink_degraded", "collective_hang"}


def lagging_ranks(b: Bundle, effects: list[Effect]) -> dict[str, tuple[list[tuple[str, int]], Effect]]:
    """Per replica: (node, gpu) of ranks that did NOT report the first NCCL timeout wave."""
    out: dict[str, tuple[list[tuple[str, int]], Effect]] = {}
    per = b.topology["gpus_per_node"]
    for replica, nodes in b.topology["replicas"].items():
        timeouts = sorted((e for e in effects if e.kind == "nccl_timeout" and e.replica == replica),
                          key=lambda e: e.ts)
        if not timeouts:
            continue
        wave = [e for e in timeouts if e.ts - timeouts[0].ts <= 5.0]
        reporters = {e.attrs["rank"] for e in wave}
        everyone = set(range(len(nodes) * per))
        silent = sorted(everyone - reporters)
        if 0 < len(silent) < len(everyone):
            out[replica] = ([b.locate(replica, r) for r in silent], wave[0])
    return out


def _explains(c: Candidate, kind: str, replica: str | None, node: str | None, ts_end: float) -> bool:
    if kind not in catalog.CAN_CAUSE[c.kind]:
        return False
    if replica not in (None, "router") and c.replica != replica:
        return False
    if kind == "cuda_oom" and c.node != node:
        return False
    return c.onset - SKEW_S <= ts_end


def _rank(c: Candidate) -> tuple:
    return (-_pre_score(c), catalog.LAYER_DEPTH[c.layer], c.onset)


def _pre_score(c: Candidate) -> float:
    return catalog.BASE_CONFIDENCE[c.kind] + SOURCE_BONUS * (len(c.sources) - 1)


def correlate(b: Bundle, threshold: float = THRESHOLD) -> dict:
    cands, effects = collect(b)
    symptoms = detect(b.requests, b.start, b.slo["p99_latency_s"], b.slo["error_rate"])

    # 2. lagging ranks corroborate (or create) a candidate on the silent node
    for replica, (suspects, first) in lagging_ranks(b, effects).items():
        nodes = {n for n, _ in suspects}
        matched = False
        for c in cands:
            if c.replica == replica and c.node in nodes and (c.gpu is None or (c.node, c.gpu) in suspects):
                silent = ", ".join(f"{n}/GPU{g}" for n, g in suspects[:4]) + (" ..." if len(suspects) > 4 else "")
                c.add(Evidence(first.ts, "nccl", first.line,
                               f"{len(suspects)} rank(s) never reported the timeout ({silent}): the hang is on them"))
                matched = True
        if not matched and len(nodes) == 1:
            node = next(iter(nodes))
            gpu = suspects[0][1] if len(suspects) == 1 else None
            c = Candidate("collective_hang", node, gpu, replica, first.ts - b.manifest.get("nccl_timeout_s", 60))
            c.add(Evidence(first.ts, "nccl", first.line,
                           f"NCCL timeout wave; silent rank(s) on {node}, no lower-layer evidence found"))
            cands.append(c)

    # 3. walk effects and symptoms back to candidates
    for e in effects:
        options = [c for c in cands if _explains(c, e.kind, e.replica, e.node, e.ts)]
        if options:
            e.explained_by = min(options, key=_rank)
            e.explained_by.explains.append(e)
    for s in symptoms:
        options = [c for c in cands if _explains(c, s.kind, s.replica, None, s.end)]
        if options:
            min(options, key=_rank).explains.append(s)
            continue
        # failover overload: a replica that lost capacity pushes its load onto the survivors
        failover = [c for c in cands if c.kind in CAPACITY_LOSS and c.replica != s.replica
                    and c.onset - SKEW_S <= s.start and any(isinstance(x, Symptom) for x in c.explains)]
        if failover:
            s.failover_from = min(failover, key=_rank)
            s.failover_from.explains.append(s)

    # 4. score
    for c in cands:
        impact = any(isinstance(x, Symptom) for x in c.explains)
        c.score = min(MAX_CONFIDENCE, _pre_score(c) + (IMPACT_BONUS if impact else 0.0))
    roots = [c for c in cands if any(isinstance(x, Symptom) for x in c.explains) and c.score >= threshold]
    low = [c for c in cands if any(isinstance(x, Symptom) for x in c.explains) and c.score < threshold]
    precursors = [c for c in cands if c not in roots and c not in low]
    return {"bundle": b, "roots": sorted(roots, key=lambda c: c.onset), "low": low,
            "precursors": sorted(precursors, key=lambda c: c.onset), "symptoms": symptoms, "effects": effects,
            "threshold": threshold}


def blast_radius(b: Bundle, c: Candidate) -> dict:
    replicas = [c.replica] if c.replica else list(b.topology["replicas"])
    nodes = [n for r in replicas for n in b.topology["replicas"][r]]
    ranks = [f"{r}:{b.rank_of(n, g)}" for r in replicas for n in b.topology["replicas"][r]
             for g in range(b.topology["gpus_per_node"])]
    slo = b.slo["p99_latency_s"]
    after = [q for q in b.requests if q.end >= c.onset - SKEW_S and q.upstream in replicas + ["-"]]
    return {"replicas": replicas, "nodes": nodes, "ranks": len(ranks),
            "requests_failed": sum(1 for q in after if q.status >= 500),
            "requests_slow": sum(1 for q in after if q.status == 200 and q.rt > slo)}


def action_for(b: Bundle, c: Candidate) -> dict:
    code, text = catalog.ACTIONS[c.kind]
    return {"action": code, "target": {"node": c.node, "gpu": c.gpu, "replica": c.replica},
            "detail": text.format(node=c.node, gpu=c.gpu, replica=c.replica)}


def to_dict(result: dict) -> dict:
    b: Bundle = result["bundle"]
    t0 = b.start

    def cand(c: Candidate, full: bool = True) -> dict:
        d = {"cause": c.kind, "layer": c.layer, "node": c.node, "gpu": c.gpu, "replica": c.replica,
             "rank": b.rank_of(c.node, c.gpu) if c.node and c.gpu is not None else None,
             "confidence": round(c.score, 2), "onset_s": round(c.onset - t0, 1),
             "detected_at_s": round(c.detected_at - t0, 1), "sources": sorted(c.sources),
             "evidence": [e.to_dict(t0) for e in sorted(c.evidence, key=lambda e: e.ts)]}
        if full:
            explained = [x for x in c.explains if isinstance(x, Effect)]
            kinds: dict[str, set[str]] = {}
            for e in explained:
                kinds.setdefault(e.kind, set()).add(e.node or "?")
            d["explains"] = {
                "symptoms": [s.to_dict(t0) for s in c.explains if isinstance(s, Symptom)],
                "effects": {k: {"count": sum(1 for e in explained if e.kind == k), "nodes": sorted(v)}
                            for k, v in sorted(kinds.items())},
            }
            d["blast_radius"] = blast_radius(b, c)
            d["recommended_action"] = action_for(b, c)
        return d

    unexplained = [e for e in result["effects"] if e.explained_by is None]
    timeline = []
    for c in result["roots"] + result["low"] + result["precursors"]:
        for e in c.evidence:
            timeline.append((e.ts, e.source, f"{c.kind}: {e.note}", f"{e.line.source}:{e.line.lineno}"))
    for e in result["effects"]:
        timeline.append((e.ts, "effect", f"{e.kind} on {e.node}" +
                         (f" <- {e.explained_by.kind}@{e.explained_by.node or e.explained_by.replica}"
                          if e.explained_by else " (unexplained)"), f"{e.line.source}:{e.line.lineno}"))
    for s in result["symptoms"]:
        timeline.append((s.start, "symptom", f"{s.kind} on {s.replica} (p99 {s.worst_p99:.1f}s, {s.errors} errors)",
                         "router/access.log"))
    timeline.sort(key=lambda x: x[0])
    return {
        "bundle_id": b.manifest.get("bundle_id"),
        "window": {"start": b.manifest.get("start"), "duration_s": b.manifest.get("duration_s")},
        "threshold": result["threshold"],
        "verdict": "incident" if result["roots"] else ("degraded" if result["symptoms"] else "healthy"),
        "root_causes": [cand(c) for c in result["roots"]],
        "low_confidence": [cand(c) for c in result["low"]],
        "precursors": [cand(c, full=False) for c in result["precursors"]],
        "symptoms": [s.to_dict(t0) for s in result["symptoms"]],
        "unexplained_symptoms": [s.to_dict(t0) for s in result["symptoms"]
                                 if not any(s in c.explains for c in result["roots"] + result["low"]
                                            + result["precursors"])],
        "unexplained_effects": [{"kind": e.kind, "node": e.node, "t_s": round(e.ts - t0, 1),
                                 "where": f"{e.line.source}:{e.line.lineno}"} for e in unexplained[:20]],
        "timeline": [{"t_s": round(ts - t0, 1), "kind": k, "what": w, "where": where}
                     for ts, k, w, where in timeline],
    }


def analyze(root: Path, threshold: float = THRESHOLD) -> dict:
    return to_dict(correlate(load(Path(root)), threshold))
