"""Turn raw signals into root-cause candidates and downstream effects.

A *candidate* is a hypothesis at a layer that can originate a fault
(hardware, host, driver, runtime, serving). An *effect* is something that is
only ever downstream (NCCL watchdog timeout, CUDA OOM, engine crash,
NodeNotReady, CrashLoopBackOff). Effects are never blamed directly; the engine
walks them back to a candidate. That is what keeps a healthy rank that
*reported* a timeout from being named as the root cause.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rca import catalog
from rca.bundle import Bundle
from rca.parsers import Line, LogEvent, Sample

LEAK_WINDOW = 12             # samples (3 min at 15 s scrape)
LEAK_MIN_SLOPE_MIB_S = 2.0
LEAK_MIN_R2 = 0.9
LEAK_MIN_GROWTH_MIB = 1024.0
SEGMENT_JUMP_MIB = 2000.0    # restart / process death breaks a series
CRC_RATE_PER_S = 5.0
PSI_RATE = 0.5               # fraction of wall time with runnable tasks waiting
IB_SYMBOL_RATE_PER_S = 1.0
APP_FAULT_SPREAD_GPUS = 3
APP_FAULT_WINDOW_S = 120.0
DEPLOY_LOOKBACK_S = 900.0
NEIGHBOUR_LOOKBACK_S = 300.0
SERVING_NAMESPACES = ("inference", "gpu-operator", "kube-system")


@dataclass
class Evidence:
    ts: float
    source: str                      # kern | dcgm | node_exporter | nccl | engine | k8s
    line: Line
    note: str

    def to_dict(self, t0: float) -> dict:
        return {"t_s": round(self.ts - t0, 1), "source": self.source, "where": f"{self.line.source}:{self.line.lineno}",
                "note": self.note, "line": self.line.text}


@dataclass
class Candidate:
    kind: str
    node: str | None
    gpu: int | None
    replica: str | None
    onset: float
    evidence: list[Evidence] = field(default_factory=list)
    explains: list = field(default_factory=list)
    score: float = 0.0

    @property
    def layer(self) -> str:
        return catalog.CAUSE_LAYER[self.kind]

    @property
    def sources(self) -> set[str]:
        return {e.source for e in self.evidence}

    @property
    def detected_at(self) -> float:
        return min(e.ts for e in self.evidence)

    def add(self, ev: Evidence, onset: float | None = None) -> None:
        if len([e for e in self.evidence if e.source == ev.source]) < 4:
            self.evidence.append(ev)
        if onset is not None:
            self.onset = min(self.onset, onset)


@dataclass
class Effect:
    kind: str                         # nccl_timeout | cuda_oom | engine_crash | node_not_ready | crashloop | pod_evicted
    ts: float
    node: str | None
    replica: str | None
    line: Line
    attrs: dict = field(default_factory=dict)
    explained_by: Candidate | None = None


class Collector:
    def __init__(self, b: Bundle):
        self.b = b
        self.cands: dict[tuple[str, str | None, int | None], Candidate] = {}
        self.effects: list[Effect] = []
        self.deploys: list[tuple[float, str, LogEvent]] = []      # (ts, replica, event)
        self.app_faults: list[tuple[LogEvent, int]] = []

    def cand(self, kind: str, node: str | None, gpu: int | None, onset: float,
             replica: str | None = None) -> Candidate:
        key = (kind, node, gpu)
        if key not in self.cands:
            self.cands[key] = Candidate(kind, node, gpu, replica or self.b.replica_of(node), onset)
        return self.cands[key]

    def find(self, kind: str, node: str | None, gpu: int | None = None) -> Candidate | None:
        if gpu is not None and (kind, node, gpu) in self.cands:
            return self.cands[(kind, node, gpu)]
        for (k, n, _g), c in self.cands.items():
            if k == kind and n == node:
                return c
        return None

    # ----------------------------------------------------------- metrics
    def _series(self, samples: list[Sample]) -> dict[tuple[str, str], list[Sample]]:
        out: dict[tuple[str, str], list[Sample]] = {}
        for s in samples:
            out.setdefault((s.label("gpu"), s.metric), []).append(s)
        for v in out.values():
            v.sort(key=lambda s: s.ts)
        return out

    def dcgm(self) -> None:
        b = self.b
        last_global = max((s.ts for v in b.dcgm.values() for s in v), default=b.start)
        scrape = float(b.manifest.get("dcgm_scrape_s", 15))
        for node, samples in b.dcgm.items():
            if not samples:
                continue
            series = self._series(samples)
            node_last = max(s.ts for s in samples)
            for (g, metric), ss in series.items():
                gpu = int(g) if g.isdigit() else None
                if metric == "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS":
                    hit = next((s for s in ss if int(s.value) & catalog.THERMAL_MASK), None)
                    if hit:
                        temps = series.get((g, "DCGM_FI_DEV_GPU_TEMP"), [])
                        t_at = next((t for t in temps if t.ts >= hit.ts), None)
                        c = self.cand("thermal_throttle", node, gpu, hit.ts)
                        bits = ", ".join(v for k, v in catalog.THROTTLE_BITS.items() if int(hit.value) & k)
                        c.add(Evidence(hit.ts, "dcgm", hit.line, f"clock throttle reasons 0x{int(hit.value):x}: {bits}"))
                        if t_at:
                            c.add(Evidence(t_at.ts, "dcgm", t_at.line, f"GPU temperature {t_at.value:.0f} C"))
                elif metric == "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL":
                    for a, z in zip(ss, ss[1:]):
                        rate = (z.value - a.value) / max(z.ts - a.ts, 1e-9)
                        if rate > CRC_RATE_PER_S:
                            c = self.cand("nvlink_degraded", node, gpu, a.ts)
                            c.add(Evidence(z.ts, "dcgm", z.line, f"NVLink CRC flit errors {rate:.0f}/s"))
                            break
                elif metric == "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL":
                    for a, z in zip(ss, ss[1:]):
                        if z.value > a.value:
                            c = self.cand("ecc_uncorrectable", node, gpu, a.ts)
                            c.add(Evidence(z.ts, "dcgm", z.line, f"volatile DBE count {a.value:.0f} -> {z.value:.0f}"))
                            break
                elif metric == "DCGM_FI_DEV_ROW_REMAP_PENDING":
                    hit = next((s for s in ss if s.value > 0), None)
                    if hit:
                        self.cand("ecc_uncorrectable", node, gpu, hit.ts).add(
                            Evidence(hit.ts, "dcgm", hit.line, "row remap pending (needs GPU reset)"))
                elif metric == "DCGM_FI_DEV_FB_USED":
                    self._leak(node, gpu, ss)
                if metric == "DCGM_FI_DEV_GPU_TEMP" and ss[-1].ts < node_last - 2 * scrape:
                    self.cand("gpu_lost", node, gpu, ss[-1].ts).add(
                        Evidence(ss[-1].ts + scrape, "dcgm", ss[-1].line,
                                 f"GPU {gpu} disappears from DCGM while the node's other GPUs keep reporting"))
            if node_last < last_global - 3 * scrape and self._nodex_alive(node, node_last + 3 * scrape):
                self.markers_dcgm_down.append((node, node_last, samples[-1]))

    def _nodex_alive(self, node: str, after: float) -> bool:
        return any(s.ts > after for s in self.b.nodex.get(node, []))

    def _leak(self, node: str, gpu: int | None, ss: list[Sample]) -> None:
        segments, cur = [], [ss[0]]
        for a, z in zip(ss, ss[1:]):
            if abs(z.value - a.value) > SEGMENT_JUMP_MIB:
                segments.append(cur)
                cur = []
            cur.append(z)
        segments.append(cur)
        for seg in segments:
            if len(seg) < LEAK_WINDOW:
                continue
            base = sorted(s.value for s in seg[:4])[1]
            for i in range(LEAK_WINDOW - 1, len(seg)):
                win = seg[i - LEAK_WINDOW + 1: i + 1]
                slope, r2 = _fit([s.ts for s in win], [s.value for s in win])
                if slope > LEAK_MIN_SLOPE_MIB_S and r2 > LEAK_MIN_R2 and seg[i].value - base > LEAK_MIN_GROWTH_MIB:
                    onset = max(seg[0].ts, seg[i].ts - (seg[i].value - base) / slope)
                    c = self.cand("memory_leak", node, gpu, onset)
                    c.add(Evidence(seg[i].ts, "dcgm", seg[i].line,
                                   f"FB used +{seg[i].value - base:.0f} MiB at {slope:.1f} MiB/s (R2={r2:.2f}) "
                                   "while the KV pool is preallocated"))
                    return

    def node_exporter(self) -> None:
        b = self.b
        last_global = max((s.ts for v in b.nodex.values() for s in v), default=b.start)
        for node, samples in b.nodex.items():
            by_metric: dict[str, list[Sample]] = {}
            for s in samples:
                by_metric.setdefault(s.metric, []).append(s)
            psi = sorted(by_metric.get("node_pressure_cpu_waiting_seconds_total", []), key=lambda s: s.ts)
            self._rate_trigger(node, psi, PSI_RATE, "host_cpu_contention", "node_exporter",
                               "CPU pressure (PSI some): runnable tasks waiting {r:.0%} of wall time")
            ib = sorted(by_metric.get("node_infiniband_symbol_error_total", []), key=lambda s: s.ts)
            self._rate_trigger(node, ib, IB_SYMBOL_RATE_PER_S, "network_degraded", "node_exporter",
                               "InfiniBand symbol errors {r:.0f}/s on {dev}")
            if samples:
                last = max(samples, key=lambda s: s.ts)
                scrape = float(b.manifest.get("dcgm_scrape_s", 15))
                if last.ts < last_global - 4 * scrape:
                    c = self.cand("node_loss", node, None, last.ts)
                    c.add(Evidence(last.ts + scrape, "node_exporter", last.line,
                                   "last node_exporter sample; every later scrape of this node is missing"))
                    dl = b.dcgm.get(node)
                    if dl:
                        d = max(dl, key=lambda s: s.ts)
                        if d.ts <= last.ts + scrape:
                            c.add(Evidence(d.ts + scrape, "dcgm", d.line, "DCGM stops at the same moment"))

    def _rate_trigger(self, node: str, ss: list[Sample], limit: float, kind: str, source: str, note: str) -> None:
        streak = 0
        for a, z in zip(ss, ss[1:]):
            r = (z.value - a.value) / max(z.ts - a.ts, 1e-9)
            streak = streak + 1 if r > limit else 0
            if streak == 1:
                first = a
            if streak >= 2:
                c = self.cand(kind, node, None, first.ts)
                c.add(Evidence(z.ts, source, z.line, note.format(r=r, dev=z.label("device"))))
                return

    # -------------------------------------------------------------- logs
    def logs(self) -> None:
        b = self.b
        for ev in b.events:
            if ev.kind == "xid":
                self._xid(ev)
            elif ev.kind == "driver_api_mismatch":
                self.cand("driver_mismatch", ev.node, None, ev.ts).add(
                    Evidence(ev.ts, "kern", ev.line, f"driver API mismatch: userspace {ev.attrs['client']} "
                                                     "vs loaded kernel module"))
            elif ev.kind == "nccl_timeout":
                self.effects.append(Effect("nccl_timeout", ev.ts, ev.node, b.replica_of(ev.node), ev.line, ev.attrs))
            elif ev.kind == "cuda_oom":
                self.effects.append(Effect("cuda_oom", ev.ts, ev.node, b.replica_of(ev.node), ev.line, ev.attrs))
            elif ev.kind == "engine_crash":
                self.effects.append(Effect("engine_crash", ev.ts, ev.node, b.replica_of(ev.node), ev.line, ev.attrs))
            elif ev.kind == "k8s":
                self._k8s(ev)

    def _xid(self, ev: LogEvent) -> None:
        code = ev.attrs["code"]
        gpu = self.b.gpu_of_pci(ev.attrs["pci"])
        entry = catalog.XID.get(code)
        if entry is None:
            return
        name, _layer, role, kind, _action = entry
        note = f"Xid {code} ({name}) on GPU {gpu}"
        if role == "cause":
            self.cand(kind, ev.node, gpu, ev.ts).add(Evidence(ev.ts, "kern", ev.line, note), onset=ev.ts)
        elif role == "corroborate":
            c = self.find(kind, ev.node, gpu) or self.cand(kind, ev.node, gpu, ev.ts)
            c.add(Evidence(ev.ts, "kern", ev.line, note))
        elif role == "app":
            self.app_faults.append((ev, gpu))

    def _k8s(self, ev: LogEvent) -> None:
        a, b = ev.attrs, self.b
        reason, msg = a["reason"], a["message"]
        if reason == "NodeNotReady":
            self.effects.append(Effect("node_not_ready", ev.ts, ev.node, b.replica_of(ev.node), ev.line))
        elif reason == "BackOff":
            self.effects.append(Effect("crashloop", ev.ts, ev.node, b.replica_of(ev.node), ev.line))
        elif reason in ("TaintManagerEviction", "Evicted"):
            self.effects.append(Effect("pod_evicted", ev.ts, ev.node, b.replica_of(ev.node), ev.line))
        elif reason == "Pulled" and "nvidia/driver" in msg:
            self.driver_rollouts.append(ev)
        elif reason == "Pulled" and a["namespace"] == "inference" and "Successfully pulled" in msg:
            self.deploys.append((ev.ts, b.replica_of(ev.node), ev))
        elif reason == "Scheduled" and a["namespace"] not in SERVING_NAMESPACES:
            target = msg.rsplit(" to ", 1)[-1].strip()
            if target in b.nodes:
                self.neighbours.append((target, ev))

    # ------------------------------------------------------- correlation
    def corroborate(self) -> None:
        b = self.b
        for node, last, s in self.markers_dcgm_down:
            for c in self.cands.values():
                if c.node == node and c.kind in ("driver_mismatch", "gsp_error"):
                    c.add(Evidence(last + 15, "dcgm", s.line,
                                   "dcgm-exporter on this node stops (NVML cannot initialise) while the node is up"))
        for ev in self.driver_rollouts:
            c = self.find("driver_mismatch", ev.node)
            if c:
                c.add(Evidence(ev.ts, "k8s", ev.line, "driver image rolled to this node just before the mismatch"))
        for node, ev in self.neighbours:
            c = self.find("host_cpu_contention", node)
            if c and c.onset - NEIGHBOUR_LOOKBACK_S <= ev.ts <= c.onset + 60:
                c.add(Evidence(ev.ts, "k8s", ev.line, "non-serving pod scheduled onto the GPU node"), onset=ev.ts)
        for c in [c for c in self.cands.values() if c.kind == "host_cpu_contention"]:
            self._util_drop(c)
        for w in b.nccl_warns:
            peer = b.node_of_ip(w.attrs.get("peer_ip"))
            c = self.find("network_degraded", peer) if peer else None
            if c:
                c.add(Evidence(c.detected_at, "nccl", w.line,
                               f"NCCL NET/IB completion error on {w.node} names {peer} as the failing peer"))
        for eff in self.effects:
            why = eff.attrs.get("why")
            target = {"ecc": "ecc_uncorrectable", "driver_incompatible": "driver_mismatch",
                      "gpu_missing": "gpu_lost"}.get(why)
            if eff.kind == "cuda_oom":
                c = self.find("memory_leak", eff.node, eff.attrs.get("gpu"))
                if c and c.gpu == eff.attrs.get("gpu"):
                    c.add(Evidence(eff.ts, "engine", eff.line, "CUDA OOM on the GPU whose memory was trending up"))
            elif target:
                c = self.find(target, eff.node)
                if c:
                    c.add(Evidence(eff.ts, "engine", eff.line, f"engine error consistent with {target}"))
        self._app_faults()

    def _util_drop(self, c: Candidate) -> None:
        """GPUs idle while the replica is slow = host-bound, not GPU-bound."""
        b = self.b
        peers = [n for n in b.topology["replicas"][c.replica] if n != c.node]
        def mean_util(node: str) -> tuple[float, Sample | None]:
            ss = [s for s in b.dcgm.get(node, []) if s.metric == "DCGM_FI_DEV_GPU_UTIL"
                  and c.onset + 30 <= s.ts <= c.onset + 180]
            return (sum(s.value for s in ss) / len(ss), ss[0]) if ss else (0.0, None)
        mine, sample = mean_util(c.node)
        theirs = [mean_util(p)[0] for p in peers]
        if sample and theirs and theirs[0] > 0 and mine < 0.6 * theirs[0]:
            c.add(Evidence(sample.ts, "dcgm", sample.line,
                           f"GPU util {mine:.0f}% on {c.node} vs {theirs[0]:.0f}% on its pipeline peer"))

    def _app_faults(self) -> None:
        b = self.b
        by_replica: dict[str, list[tuple[LogEvent, int]]] = {}
        for ev, gpu in self.app_faults:
            by_replica.setdefault(b.replica_of(ev.node), []).append((ev, gpu))
        for replica, items in by_replica.items():
            items.sort(key=lambda x: x[0].ts)
            first = items[0][0].ts
            burst = [(e, g) for e, g in items if e.ts - first <= APP_FAULT_WINDOW_S]
            gpus = {(e.node, g) for e, g in burst}
            nodes = {e.node for e, _ in burst}
            if len(gpus) >= APP_FAULT_SPREAD_GPUS and len(nodes) >= 2:
                c = self.cand("bad_build", None, None, first, replica=replica)
                c.replica = replica
                c.add(Evidence(first, "kern", burst[0][0].line,
                               f"Xid 13/31 on {len(gpus)} GPUs across {len(nodes)} nodes within "
                               f"{APP_FAULT_WINDOW_S:.0f}s: one bad build, not {len(gpus)} bad GPUs"))
                for ts, rep, ev in self.deploys:
                    if rep == replica and first - DEPLOY_LOOKBACK_S <= ts <= first:
                        c.add(Evidence(ts, "k8s", ev.line, "new engine image rolled to this replica"), onset=ts)
                for eff in self.effects:
                    if eff.replica == replica and eff.attrs.get("why") == "illegal_memory_access":
                        c.add(Evidence(eff.ts, "engine", eff.line, "illegal memory access in the engine"))
            else:
                for ev, gpu in items:
                    c = self.cand("app_fault", ev.node, gpu, ev.ts)
                    c.add(Evidence(ev.ts, "kern", ev.line,
                                   f"isolated Xid {ev.attrs['code']} from pid {ev.attrs['pid']} ({ev.attrs['name']})"))

    def collect(self) -> tuple[list[Candidate], list[Effect]]:
        self.markers_dcgm_down: list = []
        self.driver_rollouts: list[LogEvent] = []
        self.neighbours: list[tuple[str, LogEvent]] = []
        self.logs()
        self.dcgm()
        self.node_exporter()
        self.corroborate()
        cands = [c for c in self.cands.values() if c.evidence]
        return cands, self.effects


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0, 0.0
    slope = sxy / sxx
    return slope, (sxy * sxy) / (sxx * syy)


def collect(b: Bundle) -> tuple[list[Candidate], list[Effect]]:
    return Collector(b).collect()
