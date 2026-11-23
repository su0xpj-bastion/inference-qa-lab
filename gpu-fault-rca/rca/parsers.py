"""Parsers for the real formats in an incident bundle.

Shares no code with `faultsim/`. Each parser is tested against hand-written
fixtures in `tests/fixtures/` that follow the upstream tools' formats,
including variants the simulator never emits (dmesg `[  123.456]` prefixes,
pre-525 Xid `pid=` formatting, Prometheus lines without timestamps).
"""
from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ENCODING = "utf-8"
MONTHS = {m: i + 1 for i, m in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))}


@dataclass(frozen=True)
class Line:
    source: str        # bundle-relative path
    lineno: int
    text: str


@dataclass
class LogEvent:
    ts: float
    node: str | None
    kind: str
    line: Line
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Request:
    end: float
    status: int
    rt: float
    upstream: str


# ------------------------------------------------------------------ time
def parse_iso(s: str) -> float:
    """2026-11-14T10:03:00.123456+00:00 | 2026-11-14T10:03:00Z | ...,123456+0000"""
    m = re.match(r"(\d{4})-(\d\d)-(\d\d)[T ](\d\d):(\d\d):(\d\d)(?:[.,](\d+))?(Z|[+-]\d\d:?\d\d)?$", s)
    if not m:
        raise ValueError(f"not an ISO timestamp: {s!r}")
    y, mo, d, hh, mm, ss, frac, tz = m.groups()
    ts = calendar.timegm((int(y), int(mo), int(d), int(hh), int(mm), int(ss), 0, 0, 0))
    if frac:
        ts += int(frac) / 10 ** len(frac)
    if tz and tz != "Z":
        sign = 1 if tz[0] == "+" else -1
        digits = tz[1:].replace(":", "")
        ts -= sign * (int(digits[:2]) * 3600 + int(digits[2:]) * 60)
    return ts


def _md_hms(year: int, mon: int, day: int, hms: str) -> float:
    hh, mm, ss = hms.split(":")
    sec = float(ss)
    return calendar.timegm((year, mon, day, int(hh), int(mm), 0, 0, 0, 0)) + sec


# ---------------------------------------------------------------- kernel
XID_RE = re.compile(r"NVRM: Xid \(PCI:([0-9a-fA-F:.]+)\): (\d+), (?:pid=('?[^,]*'?), name=([^,]*), )?(.*)$")
KERN_RE = re.compile(r"^(\S+) (\S+) kernel: (?:\[\s*\d+\.\d+\] )?(.*)$")
DMESG_RE = re.compile(r"^\[\s*(\d+\.\d+)\] (.*)$")


def parse_kern(path: Path, rel: str, default_node: str | None = None,
               boot_epoch: float | None = None) -> list[LogEvent]:
    """journalctl -k -o short-iso[-precise] lines, or raw `dmesg` lines when boot_epoch is given."""
    out: list[LogEvent] = []
    with path.open("r", encoding=ENCODING) as fh:
        for no, raw in enumerate(fh, 1):
            raw = raw.rstrip("\n")
            m = KERN_RE.match(raw)
            if m:
                ts, node, msg = parse_iso(m.group(1)), m.group(2), m.group(3)
            else:
                d = DMESG_RE.match(raw)
                if not d or boot_epoch is None:
                    continue
                ts, node, msg = boot_epoch + float(d.group(1)), default_node, d.group(2)
            line = Line(rel, no, raw)
            x = XID_RE.search(msg)
            if x:
                pci = x.group(1).lower()
                pid = (x.group(3) or "").strip("'")
                out.append(LogEvent(ts, node, "xid", line, {
                    "code": int(x.group(2)), "pci": pci,
                    "pid": int(pid) if pid.isdigit() else None,
                    "name": (x.group(4) or "").strip() or None, "text": x.group(5)}))
            elif "NVRM: API mismatch" in msg:
                v = re.search(r"client has the version ([\d.]+)", msg)
                out.append(LogEvent(ts, node, "driver_api_mismatch", line, {"client": v.group(1) if v else None}))
            elif "soft lockup" in msg or "rcu_sched self-detected stall" in msg:
                out.append(LogEvent(ts, node, "cpu_stall", line, {}))
            else:
                out.append(LogEvent(ts, node, "kern", line, {}))
    return out


# ------------------------------------------------------------ prometheus
PROM_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+(\d+))?$")
LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Sample:
    metric: str
    labels: tuple[tuple[str, str], ...]
    value: float
    ts: float | None
    line: Line

    def label(self, key: str, default: str = "") -> str:
        for k, v in self.labels:
            if k == key:
                return v
        return default


def parse_prom(path: Path, rel: str, scrape_ts: float | None = None) -> list[Sample]:
    """Prometheus text exposition. Sample timestamps are ms; absent ones take `scrape_ts`."""
    out: list[Sample] = []
    with path.open("r", encoding=ENCODING) as fh:
        for no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            m = PROM_RE.match(raw)
            if not m:
                continue
            labels = tuple(LABEL_RE.findall(m.group(2) or ""))
            ts = int(m.group(4)) / 1000.0 if m.group(4) else scrape_ts
            out.append(Sample(m.group(1), labels, float(m.group(3)), ts, Line(rel, no, raw)))
    return out


# ------------------------------------------------------- torch / NCCL logs
WATCHDOG_RE = re.compile(r"^\[rank(\d+)\]:\[E(\d\d)(\d\d) (\d\d:\d\d:\d\d\.\d+)[^\]]*\] .*Watchdog caught collective "
                         r"operation timeout: WorkNCCL\(SeqNum=(\d+), OpType=(\w+).*Timeout\(ms\)=(\d+)\)")
LAST_DONE_RE = re.compile(r"^\[rank(\d+)\]:\[E\d{4} [^\]]*\] .*last completed NCCL work: (\d+)")
NCCL_WARN_RE = re.compile(r"^(\S+?):(\d+):(\d+) \[(\d+)\] NCCL WARN (.*)$")
PEER_RE = re.compile(r"from peer (\d+\.\d+\.\d+\.\d+)<\d+>")


def parse_worker(path: Path, rel: str, node: str, year: int) -> list[LogEvent]:
    out: list[LogEvent] = []
    with path.open("r", encoding=ENCODING) as fh:
        for no, raw in enumerate(fh, 1):
            raw = raw.rstrip("\n")
            line = Line(rel, no, raw)
            m = WATCHDOG_RE.match(raw)
            if m:
                rank, mon, day, hms, seq, op, tmo = m.groups()
                out.append(LogEvent(_md_hms(year, int(mon), int(day), hms), node, "nccl_timeout", line,
                                    {"rank": int(rank), "seq": int(seq), "op": op, "timeout_ms": int(tmo)}))
                continue
            w = NCCL_WARN_RE.match(raw)
            if w:
                peer = PEER_RE.search(w.group(5))
                out.append(LogEvent(-1.0, node, "nccl_warn", line, {
                    "local_gpu": int(w.group(4)), "peer_ip": peer.group(1) if peer else None,
                    "net": "NET/IB" in w.group(5)}))
    return out


# --------------------------------------------------------- serving engine
VLLM_RE = re.compile(r"^(INFO|WARNING|ERROR|DEBUG|CRITICAL) (\d\d)-(\d\d) (\d\d:\d\d:\d\d)(?:\.\d+)? \[([^\]]+)\] (.*)$")
OOM_RE = re.compile(r"CUDA out of memory.*GPU (\d+) has a total capacity")


def parse_engine(path: Path, rel: str, node: str, year: int) -> list[LogEvent]:
    out: list[LogEvent] = []
    with path.open("r", encoding=ENCODING) as fh:
        for no, raw in enumerate(fh, 1):
            raw = raw.rstrip("\n")
            m = VLLM_RE.match(raw)
            if not m:
                continue
            level, mon, day, hms, _src, msg = m.groups()
            ts = _md_hms(year, int(mon), int(day), hms)
            line = Line(rel, no, raw)
            if level != "ERROR":
                continue
            oom = OOM_RE.search(msg)
            if oom:
                out.append(LogEvent(ts, node, "cuda_oom", line, {"gpu": int(oom.group(1))}))
            elif "illegal memory access" in msg:
                out.append(LogEvent(ts, node, "engine_crash", line, {"why": "illegal_memory_access"}))
            elif "uncorrectable ECC" in msg:
                out.append(LogEvent(ts, node, "engine_crash", line, {"why": "ecc"}))
            elif "Error 803" in msg or "cuda driver combination" in msg:
                out.append(LogEvent(ts, node, "engine_crash", line, {"why": "driver_incompatible"}))
            elif "visible CUDA devices" in msg:
                out.append(LogEvent(ts, node, "engine_crash", line, {"why": "gpu_missing"}))
            else:
                out.append(LogEvent(ts, node, "engine_crash", line, {"why": "other"}))
    return out


# --------------------------------------------------------------- router
NGINX_RE = re.compile(r'\[(\d\d)/(\w{3})/(\d{4}):(\d\d:\d\d:\d\d) ([+-]\d{4})\] "[^"]*" (\d{3}) \d+ .*? rt=([\d.]+)'
                      r'(?: urt=\S+)?(?: ups=(\S+))?')


def parse_access(path: Path) -> list[Request]:
    out: list[Request] = []
    with path.open("r", encoding=ENCODING) as fh:
        for raw in fh:
            m = NGINX_RE.search(raw)
            if not m:
                continue
            day, mon, year, hms, tz, status, rt, ups = m.groups()
            ts = _md_hms(int(year), MONTHS[mon], int(day), hms)
            sign = 1 if tz[0] == "+" else -1
            ts -= sign * (int(tz[1:3]) * 3600 + int(tz[3:]) * 60)
            out.append(Request(ts, int(status), float(rt), ups or "-"))
    return out


# ------------------------------------------------------------------ k8s
def parse_events(path: Path, rel: str) -> list[LogEvent]:
    with path.open("r", encoding=ENCODING) as fh:
        doc = json.load(fh)
    out: list[LogEvent] = []
    for i, ev in enumerate(doc.get("items", []), 1):
        stamp = ev.get("lastTimestamp") or ev.get("eventTime") or ev.get("firstTimestamp")
        if not stamp:
            continue
        obj = ev.get("involvedObject", {})
        src = ev.get("source", {})
        node = obj.get("name") if obj.get("kind") == "Node" else src.get("host")
        text = f'{ev.get("type", "")} {ev.get("reason", "")} {obj.get("kind", "")}/{obj.get("namespace", "")}/' \
               f'{obj.get("name", "")}: {ev.get("message", "")}'
        out.append(LogEvent(parse_iso(stamp), node, "k8s", Line(rel, i, text), {
            "reason": ev.get("reason", ""), "type": ev.get("type", ""), "message": ev.get("message", ""),
            "object_kind": obj.get("kind", ""), "object": obj.get("name", ""),
            "namespace": obj.get("namespace", ""), "host": src.get("host")}))
    return out
