from __future__ import annotations

# Bring-up report: JSON (machine-readable verdict) and Markdown (job summary).

import json
from pathlib import Path

from bringup.parsers.common import ENCODING

_ICON = {"ACCEPTED": "✅", "ACCEPTED_DEGRADED": "🟡", "REJECTED": "❌"}


def to_markdown(r: dict) -> str:
    s = r["summary"]
    out = [
        f"# Bring-up report: {r['cluster']}",
        "",
        f"**Decision: {_ICON[r['decision']]} {r['decision']}** (exit {r['exit_code']}). "
        f"{s['accepted']}/{s['nodes']} nodes accepted, {s['cordoned']} cordoned.",
        "",
        "| Stage | Cordoned at this stage |",
        "|---|---|",
    ]
    out += [f"| {st} | {n} |" for st, n in s["cordoned_by_stage"].items()]
    out += ["", "## Pools", "", "| Node type | Accepted | Required | Total | GPUs accepted |", "|---|---|---|---|---|"]
    out += [f"| {p} | {v['accepted']} | {v['required']} | {v['total']} | {v['gpus_accepted']} |"
            for p, v in r["pools"].items()]
    out += ["", "## Cluster gates", "", "| Check | Status | Detail |", "|---|---|---|"]
    out += [f"| {c['name']} | {c['status'].upper()} | {c['reason'] or _brief(c['evidence'])} |"
            for c in r["cluster_checks"]]
    if r["rack_findings"]:
        out += ["", "## Rack-level findings", ""]
        out += [f"- **{f['rack']} / {f['check']}**: {f['reason']} ({', '.join(f['nodes'])})"
                for f in r["rack_findings"]]
    if r["cordon"]:
        out += ["", "## Cordoned nodes", "", "| Node | Rack | Stage | Reason |", "|---|---|---|---|"]
        for c in r["cordon"]:
            reason = "<br>".join(x.replace("|", "\\|") for x in c["reasons"])
            out.append(f"| `{c['hostname']}` | {c['rack']} | {c['stage']} | {reason} |")
        out += ["", "<details><summary>Cordon commands (not executed)</summary>", "", "```bash"]
        for c in r["cordon"]:
            why = f"bringup:{c['stage']}:{','.join(c['checks'])}"
            out.append(f"scontrol update NodeName={c['hostname']} State=DRAIN Reason=\"{why}\"")
            out.append(f"kubectl taint node {c['hostname']} bringup.gpu/rejected={c['stage']}:NoSchedule --overwrite")
        out += ["```", "", "</details>"]
    out.append("")
    return "\n".join(out)


def _brief(ev: dict) -> str:
    keys = ("accepted", "required", "peak_busbw_gbps", "floor", "nodes")
    parts = [f"{k}={ev[k]:.1f}" if isinstance(ev[k], float) else f"{k}={ev[k]}" for k in keys if k in ev]
    return ", ".join(parts)


def write_reports(r: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath, mpath = out_dir / "bringup.json", out_dir / "bringup.md"
    with jpath.open("w", encoding=ENCODING) as fh:
        json.dump(r, fh, indent=2)
        fh.write("\n")
    with mpath.open("w", encoding=ENCODING) as fh:
        fh.write(to_markdown(r))
    return jpath, mpath
