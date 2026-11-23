"""Markdown incident report from the RCA JSON."""
from __future__ import annotations


def _fmt_t(t: float) -> str:
    return f"T+{t:.0f}s"


def markdown(r: dict) -> str:
    out = [f"# Incident report `{r['bundle_id']}`", "",
           f"Window: {r['window']['start']} + {r['window']['duration_s']}s · verdict **{r['verdict'].upper()}** · "
           f"confidence threshold {r['threshold']}", ""]
    if not r["root_causes"]:
        out += ["No root cause at or above the confidence threshold.", ""]
    for i, rc in enumerate(r["root_causes"], 1):
        where = rc["node"] or rc["replica"]
        if rc["gpu"] is not None:
            where += f" GPU {rc['gpu']} (rank {rc['rank']} of {rc['replica']})"
        act = rc["recommended_action"]
        br = rc["blast_radius"]
        out += [f"## Root cause {i}: `{rc['cause']}` on {where}", "",
                "| Field | Value |", "|---|---|",
                f"| Layer | {rc['layer']} |",
                f"| Confidence | {rc['confidence']:.2f} (sources: {', '.join(rc['sources'])}) |",
                f"| Onset / detected | {_fmt_t(rc['onset_s'])} / {_fmt_t(rc['detected_at_s'])} |",
                f"| Blast radius | {', '.join(br['replicas'])}: {br['ranks']} ranks on {', '.join(br['nodes'])}; "
                f"{br['requests_failed']} failed, {br['requests_slow']} slow requests |",
                f"| Action | **{act['action']}**: {act['detail']} |", "",
                "Evidence:", ""]
        for e in rc["evidence"]:
            out.append(f"- {_fmt_t(e['t_s'])} `{e['where']}` ({e['source']}): {e['note']}")
            out.append(f"  `{e['line'][:220]}`")
        eff = rc["explains"]["effects"]
        if eff:
            out += ["", "Explains downstream (not blamed):", ""]
            for k, v in eff.items():
                out.append(f"- {v['count']} × `{k}` on {', '.join(v['nodes'])}")
        syms = rc["explains"]["symptoms"]
        if syms:
            out += ["", "User-visible symptoms:", ""]
            for s in syms:
                via = f" (failover from {s['failover_from']})" if s.get("failover_from") else ""
                out.append(f"- `{s['kind']}` on {s['replica']} {_fmt_t(s['start_s'])}–{_fmt_t(s['end_s'])}: "
                           f"p99 {s['worst_p99_s']}s, {s['errors']} errors{via}")
        out.append("")
    if r["low_confidence"]:
        out += ["## Low-confidence hypotheses", ""]
        out += [f"- `{c['cause']}` on {c['node'] or c['replica']} ({c['confidence']:.2f})" for c in r["low_confidence"]]
        out.append("")
    if r["precursors"]:
        out += ["## Precursors (no user impact yet)", ""]
        out += [f"- `{c['cause']}` on {c['node'] or c['replica']}"
                + (f" GPU {c['gpu']}" if c["gpu"] is not None else "")
                + f" from {_fmt_t(c['onset_s'])}: {c['evidence'][0]['note']}" for c in r["precursors"]]
        out.append("")
    if r["unexplained_symptoms"] or r["unexplained_effects"]:
        out += ["## Unexplained", ""]
        out += [f"- symptom `{s['kind']}` on {s['replica']} {_fmt_t(s['start_s'])}" for s in r["unexplained_symptoms"]]
        out += [f"- effect `{e['kind']}` on {e['node']} {_fmt_t(e['t_s'])} `{e['where']}`"
                for e in r["unexplained_effects"]]
        out.append("")
    out += ["## Timeline (first 30 entries)", "", "| t | kind | what | where |", "|---|---|---|---|"]
    out += [f"| {_fmt_t(t['t_s'])} | {t['kind']} | {t['what']} | `{t['where']}` |" for t in r["timeline"][:30]]
    return "\n".join(out) + "\n"
