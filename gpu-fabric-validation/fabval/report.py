from __future__ import annotations

# Markdown + JSON rendering of an analyze.Report.

import json
from pathlib import Path

from fabval.analyze import Report

ENCODING = "utf-8"
ICON = {"PASS": "PASS", "FAIL": "FAIL", "INCONCLUSIVE": "INCONCLUSIVE"}


def _matrix_md(report: Report) -> list[str]:
    nodes = report.nodes
    if not report.pair_matrix or len(nodes) > 16:
        return []
    lines = ["| min busbw % of expected / worst small-msg us | " + " | ".join(nodes) + " |",
             "|---|" + "---|" * len(nodes)]
    for a in nodes:
        cells = []
        for b in nodes:
            cell = report.pair_matrix.get(f"{a}|{b}") or report.pair_matrix.get(f"{b}|{a}")
            if a == b or cell is None:
                cells.append("·")
                continue
            mark = " **x**" if cell["rails_bw_fail"] or cell["rails_lat_fail"] else ""
            cells.append(f"{cell['min_pct']:.0f}% / {cell['max_small_us']:.0f}{mark}")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    return lines


def to_markdown(report: Report) -> str:
    st = report.stages
    out = [f"## Fabric report: {ICON[report.verdict]}", ""]
    out.append(f"`{report.node_type}` x {len(report.nodes)} nodes, source: **{report.source}**. "
               f"Exit code {report.exit_code} (0 pass, 1 fail, 2 inconclusive).")
    out.append("")
    if report.root_causes:
        out += ["### Root causes", "", "| Location | Diagnosis | Evidence |", "|---|---|---|"]
        for c in report.root_causes:
            ev = "<br>".join(e.replace("|", "\\|") for e in c.evidence[:4]) or "·"
            out.append(f"| `{c.location}` | {c.message} | {ev} |")
        out.append("")
    out += ["### Stages", "", "| Stage | Result |", "|---|---|"]
    nh = st.get("node_health", {})
    out.append(f"| 0. node health (ibstat, perfquery/ethtool, show_gids, topo) | "
               f"{nh.get('nodes', 0) - len(nh.get('nodes_failed', []))}/{nh.get('nodes', 0)} nodes clean |")
    sn = st.get("single_node", {})
    if sn.get("nodes"):
        worst = min(sn["nodes"].items(), key=lambda kv: kv[1]["pct_expected"])
        out.append(f"| 1. single node all_reduce (NVLink, expected {sn['expected_busbw']} GB/s) | "
                   f"worst {worst[0]} at {worst[1]['pct_expected']}% |")
    pr = st.get("pairs", {})
    if pr:
        out.append(f"| 2. pairwise sendrecv per rail (expected {pr['expected_busbw']} GB/s, budget {pr['latency_budget_us']:.0f} us) | "
                   f"{pr['tests']} tests, {pr['bw_failed']} bandwidth fails, {pr['lat_failed']} latency fails |")
    co = st.get("collective", {})
    if "peak_busbw" in co:
        out.append(f"| 3. full all_reduce ({co.get('nranks')} ranks, expected {co['expected_busbw']} GB/s) | "
                   f"peak {co['peak_busbw']} GB/s = {co['pct_expected']}%, #wrong {co['wrong']} |")
    out.append("")
    if m := _matrix_md(report):
        out += ["### Pair matrix", ""] + m + [""]
    if report.findings:
        out += ["### All findings", "", "| Sev | Stage | Location | Finding |", "|---|---|---|---|"]
        for f in report.findings:
            out.append(f"| {f.severity} | {f.stage} | `{f.location}` | {f.message.replace('|', '/')} |")
        out.append("")
    if report.errors:
        out += ["### Coverage gaps (make the verdict INCONCLUSIVE when nothing failed)", ""]
        out += [f"- {e}" for e in report.errors] + [""]
    return "\n".join(out)


def write(report: Report, md_path: Path | None, json_path: Path | None) -> None:
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with md_path.open("w", encoding=ENCODING) as fh:
            fh.write(to_markdown(report))
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding=ENCODING) as fh:
            json.dump(report.to_dict(), fh, indent=2)
            fh.write("\n")
