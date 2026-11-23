"""Discrete-time (1 s) simulation of the serving deployment under load.

The simulator owns physics and side effects: request routing and latency,
replica hang/crash/restart, GPU counters, and every telemetry line. Faults
(`faultsim/faults.py`) only nudge state; this module turns state into the
log and metric formats an on-call engineer would actually pull.

Three independent RNG streams (requests, noise, faults) keep a replay with
remediation actions byte-identical to the incident run up to the first action.
"""
from __future__ import annotations

import bisect
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from faultsim import formats as fmt
from faultsim import topology as topo

ENCODING = "utf-8"
CAPACITY_RPS = 20.0          # per replica at full speed
BASE_LATENCY_S = 0.5
ROUTER_TIMEOUT_S = 30.0
EJECT_S = 10.0               # nginx max_fails=1 fail_timeout=10s
NCCL_TIMEOUT_S = 60
RESTART_S = 90               # engine cold start: weights load + CUDA graphs
SCRAPE_S = 15
FB_BASE_MIB = 72000.0        # engine preallocates KV cache at startup; flat when healthy
FB_DEAD_MIB = 512.0
NAMESPACE = "inference"
ACTION_DURATION_S = {
    "DRAIN_NODE": 5, "RESTART_WORKER": RESTART_S, "RESET_GPU": 60 + RESTART_S,
    "RESHARD_TP": 60 + RESTART_S, "ROLLBACK_DRIVER": 120 + RESTART_S,
    "EVICT_NEIGHBOUR": 15, "ROLLBACK_DEPLOY": 60,
}


def load_rps(profile: str, t: float, duration: float) -> float:
    if profile == "idle":
        return 2.0
    if profile == "ramp":                      # stress: 4 -> 28 rps
        return 4.0 + 24.0 * t / duration
    return 12.0                                # steady and soak


def queue_mult(rho: float) -> float:
    r = min(rho, 0.95)
    return 1.0 + r * r / (1.0 - r)


def pod_name(node: str) -> str:
    replica = topo.replica_of(node)
    return f"engine-{replica[-1]}-{topo.REPLICAS[replica].index(node)}"


@dataclass
class Gpu:
    node: str
    index: int
    fb_used: float = FB_BASE_MIB
    temp: float = 45.0
    sm_clock: int = 1980
    throttle: int = 0
    sbe: int = 0
    dbe: int = 0
    crc: float = 0.0
    last_xid: int = 0
    remap_pending: int = 0
    visible: bool = True
    util_scale: float = 1.0
    thermal_target: float | None = None


@dataclass
class Node:
    name: str
    alive: bool = True
    dcgm_ok: bool = True
    psi: float = 0.0
    psi_extra: float = 0.0
    ib_symbol: float = 0.0
    ib_remote_phys: float = 0.0


@dataclass
class Replica:
    name: str
    nodes: tuple[str, ...]
    state: str = "up"                  # up | hung | down
    hung_at: float = 0.0
    silent_ranks: frozenset[int] = frozenset()
    next_restart: float | None = None
    failures: int = 0
    drained: bool = False
    capacity_scale: float = 1.0
    rho: float = 0.0
    routed: int = 0
    seq: int = 48000
    fail_done: list[float] = field(default_factory=list)
    up_since: float = 0.0


@dataclass(frozen=True)
class Action:
    kind: str
    at: float
    node: str | None = None
    gpu: int | None = None
    replica: str | None = None


class Sim:
    def __init__(self, faults: list, load: str = "steady", duration: int = 600, seed: int = 0,
                 actions: tuple[Action, ...] = (), stray_xid31: bool = False):
        self.faults = faults
        self.load = load
        self.duration = duration
        self.seed = seed
        self.actions = sorted(actions, key=lambda a: a.at)
        self.stray_xid31 = stray_xid31
        self.req_rng = random.Random(seed * 7919 + 1)
        self.noise_rng = random.Random(seed * 7919 + 2)
        self.fault_rng = random.Random(seed * 7919 + 3)
        self.nodes = {n: Node(n) for n in topo.NODES}
        self.gpus = {(n, g): Gpu(n, g) for n in topo.NODES for g in range(topo.GPUS_PER_NODE)}
        self.replicas = {r: Replica(r, nodes) for r, nodes in topo.REPLICAS.items()}
        self.files: dict[str, list[tuple[float, int, str]]] = {}
        self.events: list[dict] = []
        self.requests: list[tuple[float, float, int, str]] = []   # end, start, status, upstream
        self._line_no = 0
        self._rr = 0
        self._pending: list[tuple[float, Action]] = []           # action completions
        self.blip_at = self.noise_rng.uniform(260, 420)
        self.actions_log: list[dict] = []

    # ------------------------------------------------------------------ emit
    def write(self, path: str, ts: float, line: str) -> None:
        self._line_no += 1
        self.files.setdefault(path, []).append((ts, self._line_no, line))

    def kern(self, node: str, ts: float, msg: str) -> None:
        if self.nodes[node].alive:
            self.write(f"nodes/{node}/kern.log", ts, fmt.kern(ts + fmt.START_EPOCH, node, msg))

    def engine(self, node: str, ts: float, level: str, src: str, msg: str) -> None:
        if self.nodes[node].alive:
            self.write(f"nodes/{node}/engine.log", ts, fmt.vllm(ts + fmt.START_EPOCH, level, src, msg))

    def worker(self, node: str, ts: float, line: str) -> None:
        if self.nodes[node].alive:
            self.write(f"nodes/{node}/worker.log", ts, line)

    def event(self, ts: float, kind: str, name: str, reason: str, message: str, etype: str = "Normal",
              namespace: str = NAMESPACE, component: str = "kubelet", host: str | None = None) -> None:
        ev = fmt.k8s_event(ts + fmt.START_EPOCH, kind, name, namespace, reason, message, etype, component, host)
        self.events.append(ev)

    # ------------------------------------------------------------- replicas
    def replica(self, node: str) -> Replica:
        return self.replicas[topo.replica_of(node)]

    def hang(self, rep: Replica, t: float, silent: set[int]) -> None:
        """A collective can't complete: every other rank blocks until the NCCL watchdog fires."""
        if rep.state == "up":
            rep.state, rep.hung_at, rep.silent_ranks = "hung", t, frozenset(silent)

    def crash(self, rep: Replica, t: float) -> None:
        if rep.state != "down":
            rep.state = "down"
            rep.next_restart = t + RESTART_S

    def _replica_step(self, rep: Replica, t: float) -> None:
        if rep.state == "hung" and t >= rep.hung_at + NCCL_TIMEOUT_S:
            rep.seq += 1
            for node in rep.nodes:
                for g in range(topo.GPUS_PER_NODE):
                    rank = topo.rank_of(node, g)
                    if rank in rep.silent_ranks or not self.nodes[node].alive:
                        continue
                    jitter = self.noise_rng.uniform(0.0, 0.4)
                    op = "ALLREDUCE" if g % 2 == 0 else ("RECV" if rep.nodes.index(node) else "SEND")
                    for line in fmt.nccl_watchdog(t + jitter + fmt.START_EPOCH, rank, rep.seq, op,
                                                  NCCL_TIMEOUT_S * 1000):
                        self.worker(node, t + jitter, line)
                self.engine(node, t + 0.6, "ERROR", "multiproc_executor.py:154",
                            "Worker proc VllmWorker-0 died unexpectedly, shutting down executor.")
            self.crash(rep, t)
        elif rep.state == "down" and rep.next_restart is not None and t >= rep.next_restart and not rep.drained:
            blockers = [f for f in self.faults if f.active and f.blocks_restart(self, rep)]
            if blockers:
                rep.failures += 1
                for f in blockers:
                    f.on_restart_fail(self, t, rep)
                for node in rep.nodes:
                    if self.nodes[node].alive:
                        self.event(t + 1, "Pod", pod_name(node), "BackOff",
                                   "Back-off restarting failed container engine in pod "
                                   f"{pod_name(node)}_{NAMESPACE}", "Warning", host=node)
                rep.next_restart = t + min(300, 30 * 2 ** rep.failures)
            else:
                self.restart(rep, t)

    def restart(self, rep: Replica, t: float) -> None:
        rep.state, rep.next_restart, rep.failures, rep.up_since = "up", None, 0, t
        for node in rep.nodes:
            for g in range(topo.GPUS_PER_NODE):
                self.gpus[(node, g)].fb_used = FB_BASE_MIB + self.noise_rng.uniform(-30, 30)
            self.event(t, "Pod", pod_name(node), "Started", "Started container engine", host=node)
            self.engine(node, t + 0.2, "INFO", "api_server.py:1024", "Starting vLLM API server on http://0.0.0.0:8000")
        for f in self.faults:
            if f.active:
                f.on_restart(self, t, rep)

    def slowdown(self, rep: Replica, t: float) -> float:
        s = 1.0
        for f in self.faults:
            if f.active:
                s *= f.slowdown(self, rep)
        if self.blip_at <= t < self.blip_at + 5:
            s *= 1.6
        return s

    # -------------------------------------------------------------- actions
    def _start_action(self, a: Action, t: float) -> None:
        self.actions_log.append({"action": a.kind, "at": t, "node": a.node, "gpu": a.gpu, "replica": a.replica})
        rep = self.replicas[a.replica] if a.replica else (self.replica(a.node) if a.node else None)
        if a.kind != "EVICT_NEIGHBOUR" and rep is not None:
            rep.drained = True
            if a.node:
                self.event(t, "Node", a.node, "NodeNotSchedulable",
                           f"Node {a.node} status is now: NodeNotSchedulable", namespace="default",
                           component="kubelet", host=a.node)
        self._pending.append((t + ACTION_DURATION_S[a.kind], a))

    def _finish_action(self, a: Action, t: float) -> None:
        for f in self.faults:
            if f.active and f.fixed_by(a):
                f.deactivate(self, t)
        rep = self.replicas[a.replica] if a.replica else (self.replica(a.node) if a.node else None)
        if rep is None or a.kind in ("DRAIN_NODE", "EVICT_NEIGHBOUR"):
            return
        if a.kind == "RESHARD_TP":
            rep.capacity_scale = 0.5
        rep.drained = False
        blocked = any(f.active and f.blocks_restart(self, rep) for f in self.faults)
        if blocked:
            rep.state, rep.next_restart = "down", t
        else:
            self.restart(rep, t)

    # ------------------------------------------------------------- requests
    def _route(self, t: float) -> Replica | None:
        """Weighted round-robin (nginx `weight=` = capacity) with passive ejection."""
        names = sorted(self.replicas)
        best: Replica | None = None
        for i in range(len(names)):
            rep = self.replicas[names[(self._rr + i) % len(names)]]
            if rep.drained:
                continue
            k = bisect.bisect_right(rep.fail_done, t)
            if k and rep.fail_done[k - 1] > t - EJECT_S:
                continue
            if best is None or rep.routed / rep.capacity_scale < best.routed / best.capacity_scale:
                best = rep
        if best is not None:
            self._rr = (names.index(best.name) + 1) % len(names)
        return best

    def _requests(self, t: float) -> None:
        lam = load_rps(self.load, t, self.duration)
        n, p, limit = 0, self.req_rng.random(), math.exp(-lam)
        while p > limit:
            n += 1
            p *= self.req_rng.random()
        for _ in range(n):
            start = t + self.req_rng.random()
            noise = math.exp(self.req_rng.gauss(0.0, 0.2))
            cancel = self.req_rng.random() < 0.001
            rep = self._route(start)
            if rep is None:
                self.requests.append((start + 0.001, start, 503, "-"))
                continue
            rep.routed += 1
            if rep.state == "up":
                lat = BASE_LATENCY_S * self.slowdown(rep, t) * queue_mult(rep.rho) * noise
                if cancel:
                    self.requests.append((start + lat * 0.4, start, 499, rep.name))
                elif lat >= ROUTER_TIMEOUT_S:
                    self.requests.append((start + ROUTER_TIMEOUT_S, start, 504, rep.name))
                    bisect.insort(rep.fail_done, start + ROUTER_TIMEOUT_S)
                else:
                    self.requests.append((start + lat, start, 200, rep.name))
            elif rep.state == "hung":
                self.requests.append((start + ROUTER_TIMEOUT_S, start, 504, rep.name))
                bisect.insort(rep.fail_done, start + ROUTER_TIMEOUT_S)
            else:
                self.requests.append((start + 0.002, start, 502, rep.name))
                bisect.insort(rep.fail_done, start + 0.002)

    def _update_rho(self, t: float) -> None:
        for rep in self.replicas.values():
            cap = CAPACITY_RPS * rep.capacity_scale / self.slowdown(rep, t)
            rep.rho = 0.7 * rep.rho + 0.3 * (rep.routed / cap if rep.state == "up" else 0.0)
            rep.routed = 0

    # ------------------------------------------------------------ telemetry
    def gpu_util(self, gpu: Gpu) -> float:
        rep = self.replica(gpu.node)
        if rep.state != "up" or rep.drained:
            return 0.0
        return min(99.0, 20.0 + 75.0 * min(rep.rho, 1.0)) * gpu.util_scale

    def _thermals(self) -> None:
        for gpu in self.gpus.values():
            util = self.gpu_util(gpu)
            target = gpu.thermal_target if gpu.thermal_target is not None else 38.0 + 0.3 * util
            gpu.temp += (target - gpu.temp) / 40.0

    def _scrape(self, t: float) -> None:
        ts = t + fmt.START_EPOCH
        for node in self.nodes.values():
            if not node.alive:
                continue
            node.psi += SCRAPE_S * (0.02 + node.psi_extra) + self.noise_rng.uniform(0, 0.05)
            nx = f"nodes/{node.name}/node_exporter.prom"
            self.write(nx, t, fmt.prom("node_pressure_cpu_waiting_seconds_total", {}, round(node.psi, 3), ts))
            self.write(nx, t, fmt.prom("node_load1", {}, round(6 + 40 * node.psi_extra
                                                               + self.noise_rng.uniform(0, 1), 2), ts))
            ib = {"device": topo.IB_DEVICE, "port": "1"}
            self.write(nx, t, fmt.prom("node_infiniband_symbol_error_total", ib, int(node.ib_symbol), ts))
            self.write(nx, t, fmt.prom("node_infiniband_port_receive_remote_physical_errors_total", ib,
                                       int(node.ib_remote_phys), ts))
            if not node.dcgm_ok:
                continue
            rep = self.replica(node.name)
            for g in range(topo.GPUS_PER_NODE):
                gpu = self.gpus[(node.name, g)]
                if not gpu.visible:
                    continue
                if self.noise_rng.random() < 0.004:
                    gpu.sbe += 1                       # correctable: normal background noise
                if self.noise_rng.random() < 0.01:
                    gpu.crc += 1
                util = self.gpu_util(gpu)
                alive = rep.state in ("up", "hung")
                fb = gpu.fb_used if alive else FB_DEAD_MIB
                fb += self.noise_rng.uniform(-25, 25)
                bus = topo.PCI_BUS[g]
                labels = {
                    "gpu": str(g), "UUID": topo.gpu_uuid(node.name, g),
                    "pci_bus_id": f"0000{bus.upper()}.0", "device": f"nvidia{g}",
                    "modelName": topo.GPU_MODEL, "Hostname": node.name,
                    "DCGM_FI_DRIVER_VERSION": topo.DRIVER_VERSION,
                }
                throttle = gpu.throttle or (0x1 if util < 5 else 0)
                fields = (
                    ("DCGM_FI_DEV_GPU_TEMP", round(gpu.temp)),
                    ("DCGM_FI_DEV_POWER_USAGE", round(90 + 5.5 * util + self.noise_rng.uniform(-4, 4), 3)),
                    ("DCGM_FI_DEV_SM_CLOCK", gpu.sm_clock if util > 0 else 345),
                    ("DCGM_FI_DEV_GPU_UTIL", round(util)),
                    ("DCGM_FI_DEV_FB_USED", round(fb)),
                    ("DCGM_FI_DEV_FB_FREE", round(topo.FB_TOTAL_MIB - fb)),
                    ("DCGM_FI_DEV_ECC_SBE_VOL_TOTAL", gpu.sbe),
                    ("DCGM_FI_DEV_ECC_DBE_VOL_TOTAL", gpu.dbe),
                    ("DCGM_FI_DEV_ROW_REMAP_PENDING", gpu.remap_pending),
                    ("DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL", int(gpu.crc)),
                    ("DCGM_FI_DEV_XID_ERRORS", gpu.last_xid),
                    ("DCGM_FI_DEV_CLOCK_THROTTLE_REASONS", throttle),
                )
                for metric, value in fields:
                    self.write(f"nodes/{node.name}/dcgm.prom", t, fmt.prom(metric, labels, value, ts))

    def _boot(self) -> None:
        for node in topo.NODES:
            self.kern(node, 0.01, f"NVRM: loading NVIDIA UNIX x86_64 Kernel Module  {topo.DRIVER_VERSION}  "
                                  "Release Build  (dvs-builder@U16-I3-B03-4-3)")
            for g in range(topo.GPUS_PER_NODE):
                rank = topo.rank_of(node, g)
                pid = topo.pid_of(node, g)
                self.worker(node, 0.5, f"{node}:{pid}:{pid + 190} [{g}] NCCL INFO NCCL version 2.21.5+cuda12.4")
                self.worker(node, 0.6, f"{node}:{pid}:{pid + 190} [{g}] NCCL INFO comm 0x55d1c0 rank {rank} "
                                       f"nranks 16 cudaDev {g} nvmlDev {g} busId {topo.PCI_BUS[g][5:]}0 "
                                       "commId 0x7d1e3b2a5c4f9e01 - Init COMPLETE")
            self.engine(node, 1.0, "INFO", "api_server.py:1024", "Starting vLLM API server on http://0.0.0.0:8000")
            self.event(0.0, "Pod", pod_name(node), "Started", "Started container engine", host=node)
        self.event(30.0, "Pod", "etl-ingest-5d8f7", "Pulled",
                   'Container image "registry.local/etl:2.3" already present on machine',
                   namespace="batch", host="cpu-node-0")

    def _engine_stats(self, t: float) -> None:
        for rep in self.replicas.values():
            if rep.state == "up" and not rep.drained:
                head = rep.nodes[0]
                kv = min(95.0, 20 + 60 * rep.rho)
                self.engine(head, t, "INFO", "metrics.py:455",
                            f"Avg prompt throughput: {900 * rep.rho:.1f} tokens/s, Avg generation throughput: "
                            f"{2400 * rep.rho:.1f} tokens/s, Running: {int(30 * rep.rho)} reqs, Swapped: 0 reqs, "
                            f"Pending: 0 reqs, GPU KV cache usage: {kv:.1f}%, CPU KV cache usage: 0.0%.")

    # ------------------------------------------------------------------ run
    def run(self) -> "Sim":
        self._boot()
        actions = list(self.actions)
        for t in range(self.duration):
            while actions and actions[0].at <= t:
                self._start_action(actions.pop(0), t)
            for done_at, a in list(self._pending):
                if done_at <= t:
                    self._pending.remove((done_at, a))
                    self._finish_action(a, t)
            for f in self.faults:
                if not f.started and t >= f.onset:
                    f.started = f.active = True
                    f.start(self, t)
                if f.active:
                    f.step(self, t)
            if self.stray_xid31 and t == 333:
                self.kern("gpu-node-2", t + 0.2, fmt.xid(topo.PCI_BUS[6], 31, 9917, "nsys",
                          "Ch 00000008, intr 00000000. MMU Fault: ENGINE GRAPHICS GPCCLIENT_T1_0 faulted @ "
                          "0x7f3a_1c000000. Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ"))
            for rep in self.replicas.values():
                self._replica_step(rep, t)
            self._requests(t)
            self._update_rho(t)
            self._thermals()
            if t % SCRAPE_S == 0:
                self._scrape(t)
            if t % 10 == 0:
                self._engine_stats(t)
        return self

    # ---------------------------------------------------------------- write
    def write_bundle(self, out: Path, truth: dict | None = None) -> Path:
        out.mkdir(parents=True, exist_ok=True)
        for rel, rows in self.files.items():
            path = out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            rows.sort(key=lambda r: (r[0], r[1]))
            header = ""
            if rel.endswith("dcgm.prom"):
                header = ("# HELP DCGM_FI_DEV_GPU_TEMP GPU temperature (in C).\n"
                          "# TYPE DCGM_FI_DEV_GPU_TEMP gauge\n")
            with path.open("w", encoding=ENCODING) as fh:
                fh.write(header + "".join(r[2] + "\n" for r in rows))
        router = out / "router" / "access.log"
        router.parent.mkdir(parents=True, exist_ok=True)
        reqs = sorted(self.requests)
        with router.open("w", encoding=ENCODING) as fh:
            for i, (end, start, status, ups) in enumerate(reqs):
                nbytes = 1800 + (i * 37) % 900 if status == 200 else 150
                fh.write(fmt.access(end + fmt.START_EPOCH, status, nbytes, end - start, ups,
                                    f"{(self.seed * 1_000_003 + i) & 0xFFFFFFFF:08x}") + "\n")
        k8s = out / "k8s" / "events.json"
        k8s.parent.mkdir(parents=True, exist_ok=True)
        events = sorted(self.events, key=lambda e: (e["lastTimestamp"], e["metadata"]["name"]))
        with k8s.open("w", encoding=ENCODING) as fh:
            json.dump({"apiVersion": "v1", "kind": "List", "items": events}, fh, indent=1)
        manifest = {
            "bundle_id": f"inc-{self.seed:04d}",
            "start": fmt.iso_z(fmt.START_EPOCH),
            "start_epoch": fmt.START_EPOCH,
            "duration_s": self.duration,
            "nccl_timeout_s": NCCL_TIMEOUT_S,
            "dcgm_scrape_s": SCRAPE_S,
            "slo": {"p99_latency_s": 3.0, "error_rate": 0.01},
            "topology": topo.manifest_topology(),
        }
        with (out / "manifest.json").open("w", encoding=ENCODING) as fh:
            json.dump(manifest, fh, indent=2)
        if truth is not None:
            with (out / "ground_truth.json").open("w", encoding=ENCODING) as fh:
                json.dump(truth, fh, indent=2)
        return out
