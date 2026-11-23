"""Fault-injection library. Each fault carries its own ground truth.

A fault only changes simulator state and writes the lines its *own* layer
would write (an Xid in kern.log, a CRC counter in DCGM). Everything
downstream (NCCL watchdog timeouts on healthy ranks, 5xx at the router,
CrashLoopBackOff) is produced by the simulator's generic machinery, exactly
as a real fault propagates through layers it doesn't own.
"""
from __future__ import annotations

from faultsim import formats as fmt
from faultsim import topology as topo
from faultsim.sim import FB_BASE_MIB, Action, Sim, pod_name

OOM_HEADROOM_MIB = 300


class Fault:
    kind = "fault"
    layer = "unknown"
    action = "INVESTIGATE"
    fix_actions: tuple[str, ...] = ()

    def __init__(self, node: str | None = None, gpu: int | None = None, onset: float = 200.0,
                 replica: str | None = None):
        self.node, self.gpu, self.onset = node, gpu, onset
        self.replica = replica or (topo.replica_of(node) if node else None)
        self.started = False
        self.active = False

    # hooks ---------------------------------------------------------------
    def start(self, sim: Sim, t: float) -> None: ...
    def step(self, sim: Sim, t: float) -> None: ...
    def slowdown(self, sim: Sim, rep) -> float: return 1.0
    def blocks_restart(self, sim: Sim, rep) -> bool: return False
    def on_restart_fail(self, sim: Sim, t: float, rep) -> None: ...
    def on_restart(self, sim: Sim, t: float, rep) -> None: ...
    def deactivate(self, sim: Sim, t: float) -> None: self.active = False

    def fixed_by(self, a: Action) -> bool:
        if a.kind not in self.fix_actions:
            return False
        if a.replica is not None:
            return a.replica == self.replica
        if a.node != self.node:
            return False
        return a.gpu is None or self.gpu is None or a.gpu == self.gpu

    def mine(self, rep) -> bool:
        return rep.name == self.replica

    def truth(self) -> dict:
        return {"cause": self.kind, "layer": self.layer, "node": self.node, "gpu": self.gpu,
                "replica": self.replica, "onset_s": self.onset, "action": self.action}

    def pci(self) -> str:
        return topo.PCI_BUS[self.gpu]

    def rank(self) -> int:
        return topo.rank_of(self.node, self.gpu)

    def node_ranks(self, node: str | None = None) -> set[int]:
        n = node or self.node
        return {topo.rank_of(n, g) for g in range(topo.GPUS_PER_NODE)}


class NodeLoss(Fault):
    """Hard power-off: no shutdown messages, every telemetry source on the node just stops."""
    kind, layer, action, fix_actions = "node_loss", "hardware", "DRAIN_NODE", ("DRAIN_NODE",)

    def start(self, sim, t):
        sim.nodes[self.node].alive = False
        sim.hang(sim.replica(self.node), t, self.node_ranks())
        self.not_ready_at = t + 40          # node-monitor-grace-period
        self.evict_at = t + 40 + 300        # default toleration for node.kubernetes.io/unreachable

    def step(self, sim, t):
        if t == int(self.not_ready_at):
            sim.event(t, "Node", self.node, "NodeNotReady", f"Node {self.node} status is now: NodeNotReady",
                      "Normal", namespace="default", component="node-controller")
        if t == int(self.evict_at):
            sim.event(t, "Pod", pod_name(self.node), "TaintManagerEviction",
                      f"Marking for deletion Pod {topo.replica_of(self.node)}/{pod_name(self.node)}",
                      "Normal", component="taint-controller")

    def blocks_restart(self, sim, rep):
        return self.mine(rep)

    def on_restart_fail(self, sim, t, rep):
        peer = [n for n in rep.nodes if n != self.node][0]
        sim.engine(peer, t, "ERROR", "parallel_state.py:1012",
                   f"torch.distributed.DistStoreError: Timed out after 300 seconds waiting for clients. "
                   f"1/2 clients joined. (rendezvous {topo.NODE_IP[self.node]}:29500)")


class GpuLost(Fault):
    """Xid 79: GPU has fallen off the bus. The GPU's rank goes silent; its peers time out."""
    kind, layer, action, fix_actions = "gpu_lost", "hardware", "DRAIN_NODE", ("DRAIN_NODE",)

    def start(self, sim, t):
        g = sim.gpus[(self.node, self.gpu)]
        g.visible = False
        sim.kern(self.node, t + 0.1, fmt.xid(self.pci(), 79, None, None, "GPU has fallen off the bus."))
        sim.kern(self.node, t + 0.1, f"NVRM: GPU {self.pci()}.0: GPU has fallen off the bus.")
        sim.kern(self.node, t + 0.2, "NVRM: A GPU crash dump has been created. If possible, please run")
        sim.hang(sim.replica(self.node), t, {self.rank()})

    def blocks_restart(self, sim, rep):
        return self.mine(rep)

    def on_restart_fail(self, sim, t, rep):
        sim.engine(self.node, t, "ERROR", "worker_base.py:612",
                   "RuntimeError: Expected 8 visible CUDA devices for tensor_parallel_size=8, found 7")


class EccUncorrectable(Fault):
    """Xid 94 contained ECC + Xid 63 row remap pending. Process dies; recurs until the GPU is reset."""
    kind, layer, action, fix_actions = "ecc_uncorrectable", "hardware", "RESET_GPU", ("RESET_GPU", "DRAIN_NODE")
    RECUR_S = 150

    def start(self, sim, t):
        self._hit(sim, t)

    def _hit(self, sim, t):
        g = sim.gpus[(self.node, self.gpu)]
        g.dbe += 1
        g.remap_pending = 1
        g.last_xid = 94
        pid = topo.pid_of(self.node, self.gpu)
        sim.kern(self.node, t + 0.1, fmt.xid(self.pci(), 94, pid, "python3",
                                             "Contained: SM (0x1). RST: No, D-RST: No"))
        sim.kern(self.node, t + 0.1, fmt.xid(self.pci(), 63, pid, "python3",
                                             "Row Remapper: New row (0x00000000000a3f20) Marked for Remapping, "
                                             "reset gpu to activate."))
        sim.engine(self.node, t + 0.3, "ERROR", "model_runner.py:1688",
                   "RuntimeError: CUDA error: uncorrectable ECC error encountered")
        sim.crash(sim.replica(self.node), t)
        self.next_hit = None

    def on_restart(self, sim, t, rep):
        if self.mine(rep):
            self.next_hit = t + self.RECUR_S

    def step(self, sim, t):
        rep = sim.replica(self.node)
        if getattr(self, "next_hit", None) is not None and t >= self.next_hit and rep.state == "up":
            self._hit(sim, t)

    def deactivate(self, sim, t):
        self.active = False
        sim.gpus[(self.node, self.gpu)].remap_pending = 0


class NvlinkDegraded(Fault):
    """CRC flit errors scale with link utilisation: latent at low load, TP slowdown under stress."""
    kind, layer, action, fix_actions = "nvlink_degraded", "hardware", "RESHARD_TP", ("RESHARD_TP", "DRAIN_NODE")
    LINK = 3
    KNEE = 0.35

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cur = 1.0
        self.last74 = -1e9

    def step(self, sim, t):
        rep = sim.replica(self.node)
        util = min(rep.rho, 1.0) if rep.state == "up" and not rep.drained else 0.0
        rate = 600.0 * max(0.0, util - self.KNEE) ** 2
        sim.gpus[(self.node, self.gpu)].crc += rate * sim.fault_rng.uniform(0.8, 1.2)
        self.cur = min(4.0, 1.0 + rate / 80.0)
        if rate > 60 and t - self.last74 >= 45:
            self.last74 = t
            sim.gpus[(self.node, self.gpu)].last_xid = 74
            sim.kern(self.node, t + 0.3, fmt.xid(self.pci(), 74, topo.pid_of(self.node, self.gpu), "python3",
                                                 f"NVLink: fatal error detected on link {self.LINK}"
                                                 "(0x0, 0x0, 0x10000, 0x0, 0x0, 0x0, 0x0)"))

    def slowdown(self, sim, rep):
        return self.cur if self.mine(rep) else 1.0


class NetworkDegraded(Fault):
    """Bad IB cable/transceiver on one node: symbol errors there, NET/IB warnings on its peers."""
    kind, layer, action, fix_actions = "network_degraded", "hardware", "DRAIN_NODE", ("DRAIN_NODE",)
    HANG_AFTER_S = 90

    def start(self, sim, t):
        self.hang_at = t + self.HANG_AFTER_S

    def step(self, sim, t):
        node = sim.nodes[self.node]
        node.ib_symbol += 25 * sim.fault_rng.uniform(0.6, 1.4)
        rep = sim.replica(self.node)
        peer = [n for n in rep.nodes if n != self.node][0]
        sim.nodes[peer].ib_remote_phys += 3 * sim.fault_rng.uniform(0.5, 1.5)
        if t % 8 == 0 and rep.state == "up":
            g = sim.fault_rng.randrange(topo.GPUS_PER_NODE)
            pid = topo.pid_of(peer, g)
            sim.worker(peer, t + 0.4,
                       f"{peer}:{pid}:{pid + 191} [{g}] NCCL WARN NET/IB : Got completion from peer "
                       f"{topo.NODE_IP[self.node]}<{41700 + g}> with status=12 opcode=129 len=0 vendor err 129 "
                       f"(Recv) localGid ::ffff:{topo.NODE_IP[peer]} remoteGids::ffff:{topo.NODE_IP[self.node]}")
        if rep.state == "up" and t >= self.hang_at:
            sim.hang(rep, t, self.node_ranks())

    def on_restart(self, sim, t, rep):
        if self.mine(rep):
            self.hang_at = t + self.HANG_AFTER_S

    def slowdown(self, sim, rep):
        return 2.5 if self.mine(rep) else 1.0


class DriverMismatch(Fault):
    """Partial rolling upgrade: userspace 560 lands on a node still running the 550 kernel module."""
    kind, layer, action, fix_actions = "driver_mismatch", "driver", "ROLLBACK_DRIVER", ("ROLLBACK_DRIVER",)

    def start(self, sim, t):
        pod = "nvidia-driver-daemonset-7xk2p"
        sim.event(t, "Pod", pod, "Pulled", f'Successfully pulled image "nvcr.io/nvidia/driver:'
                  f'{topo.UPGRADE_DRIVER_VERSION}-ubuntu22.04" in 41.2s', namespace="gpu-operator", host=self.node)
        sim.event(t + 1, "Pod", pod, "Started", "Started container nvidia-driver-ctr",
                  namespace="gpu-operator", host=self.node)
        sim.event(t + 2, "Pod", pod_name(self.node), "Killing", "Stopping container engine", host=self.node)
        self._mismatch(sim, t + 3)
        sim.nodes[self.node].dcgm_ok = False
        sim.hang(sim.replica(self.node), t + 2, self.node_ranks())

    def _mismatch(self, sim, t):
        for line in (f"NVRM: API mismatch: the client has the version {topo.UPGRADE_DRIVER_VERSION}, but",
                     f"NVRM: this kernel module has the version {topo.DRIVER_VERSION}.  Please",
                     "NVRM: make sure that this kernel module and all NVIDIA driver",
                     "NVRM: components have the same version."):
            sim.kern(self.node, t, line)

    def blocks_restart(self, sim, rep):
        return self.mine(rep)

    def on_restart_fail(self, sim, t, rep):
        self._mismatch(sim, t)
        sim.engine(self.node, t + 0.2, "ERROR", "cuda.py:118",
                   "RuntimeError: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions "
                   "before calling NumCudaDevices() that might have already set an error? Error 803: system has "
                   "unsupported display driver / cuda driver combination")

    def deactivate(self, sim, t):
        self.active = False
        sim.nodes[self.node].dcgm_ok = True


class ThermalThrottle(Fault):
    """Impaired cooling on one GPU: hot only when it works hard, so idle hides it."""
    kind, layer, action, fix_actions = "thermal_throttle", "hardware", "DRAIN_NODE", ("DRAIN_NODE",)
    SLOWDOWN_TEMP = 88
    RELEASE_TEMP = 85

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.throttled = False

    def step(self, sim, t):
        g = sim.gpus[(self.node, self.gpu)]
        rep = sim.replica(self.node)
        util = min(rep.rho, 1.0) if rep.state == "up" and not rep.drained else 0.0
        g.thermal_target = 86.0 + 20.0 * util
        if g.temp >= self.SLOWDOWN_TEMP:
            self.throttled = True
        elif g.temp <= self.RELEASE_TEMP:
            self.throttled = False
        g.throttle = 0x40 if self.throttled else 0
        g.sm_clock = 900 if self.throttled else 1980

    def slowdown(self, sim, rep):
        return 2.2 if self.mine(rep) and self.throttled else 1.0

    def deactivate(self, sim, t):
        self.active = False
        g = sim.gpus[(self.node, self.gpu)]
        g.thermal_target, g.throttle, g.sm_clock, self.throttled = None, 0, 1980, False


class MemoryLeak(Fault):
    """Worker leaks device memory per request outside the preallocated KV pool -> eventual CUDA OOM."""
    kind, layer, action, fix_actions = "memory_leak", "runtime", "RESTART_WORKER", ()

    def __init__(self, *a, mib_per_request: float = 5.0, **kw):
        super().__init__(*a, **kw)
        self.mib_per_request = mib_per_request

    def step(self, sim, t):
        rep = sim.replica(self.node)
        if rep.state != "up" or rep.drained:
            return
        g = sim.gpus[(self.node, self.gpu)]
        served = rep.rho * 20.0 / max(sim.slowdown(rep, t), 1.0)
        g.fb_used += self.mib_per_request * served
        if g.fb_used >= 81559 - OOM_HEADROOM_MIB:
            free = 81559 - g.fb_used
            sim.engine(self.node, t + 0.1, "ERROR", "engine.py:160",
                       f"torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU {self.gpu} "
                       f"has a total capacity of 79.10 GiB of which {free:.2f} MiB is free. Process "
                       f"{topo.pid_of(self.node, self.gpu)} has {g.fb_used / 1024:.2f} GiB memory in use.")
            sim.engine(self.node, t + 0.4, "ERROR", "multiproc_executor.py:154",
                       f"Worker proc VllmWorker-{self.gpu} died unexpectedly, shutting down executor.")
            sim.crash(rep, t)
            g.fb_used = FB_BASE_MIB


class HostCpuContention(Fault):
    """Noisy neighbour: a batch pod lands on the GPU node and starves the engine's CPU threads."""
    kind, layer, action, fix_actions = "host_cpu_contention", "host", "EVICT_NEIGHBOUR", ("EVICT_NEIGHBOUR",)
    POD = "feature-backfill-7c9f4"

    def start(self, sim, t):
        sim.event(t, "Pod", self.POD, "Scheduled", f"Successfully assigned batch/{self.POD} to {self.node}",
                  namespace="batch", component="default-scheduler")
        sim.event(t + 2, "Pod", self.POD, "Started", "Started container backfill", namespace="batch", host=self.node)
        sim.nodes[self.node].psi_extra = 0.85
        for g in range(topo.GPUS_PER_NODE):
            sim.gpus[(self.node, g)].util_scale = 0.4

    def slowdown(self, sim, rep):
        return 2.6 if self.mine(rep) else 1.0

    def deactivate(self, sim, t):
        self.active = False
        sim.event(t, "Pod", self.POD, "Killing", "Stopping container backfill", namespace="batch", host=self.node)
        sim.nodes[self.node].psi_extra = 0.0
        for g in range(topo.GPUS_PER_NODE):
            sim.gpus[(self.node, g)].util_scale = 1.0


class BadEngineBuild(Fault):
    """New engine image with a bad kernel: Xid 13 + 31 on many GPUs at once. Software, not hardware."""
    kind, layer, action, fix_actions = "bad_build", "serving", "ROLLBACK_DEPLOY", ("ROLLBACK_DEPLOY",)

    def start(self, sim, t):
        rep = sim.replicas[self.replica]
        for node in rep.nodes:
            sim.event(t, "Pod", pod_name(node), "Pulled",
                      f'Successfully pulled image "{topo.BAD_ENGINE_IMAGE}" in 12.9s', host=node)
            sim.event(t + 1, "Pod", pod_name(node), "Killing", "Stopping container engine", host=node)
        sim.crash(rep, t)
        rep.next_restart = t + 60
        self.crash_at = None

    def on_restart(self, sim, t, rep):
        if self.mine(rep):
            self.crash_at = t + sim.fault_rng.uniform(15, 30)

    def step(self, sim, t):
        rep = sim.replicas[self.replica]
        if self.crash_at is None or rep.state != "up" or t < self.crash_at:
            return
        self.crash_at = None
        for node in rep.nodes:
            for g in (1, 4, 6):
                pci, pid = topo.PCI_BUS[g], topo.pid_of(node, g)
                sim.kern(node, t + 0.05 * g, fmt.xid(pci, 13, pid, "python3",
                         "Graphics Exception: ChID 0010, Class 0000c7c0, Offset 00000000, Data 00000000"))
                sim.kern(node, t + 0.05 * g + 0.01, fmt.xid(pci, 31, pid, "python3",
                         "Ch 00000010, intr 00000000. MMU Fault: ENGINE GRAPHICS GPCCLIENT_T1_0 faulted @ "
                         "0x7f12_34560000. Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ"))
            sim.engine(node, t + 0.5, "ERROR", "model_runner.py:1688",
                       "RuntimeError: CUDA error: an illegal memory access was encountered")
        sim.crash(rep, t)

    def truth(self):
        d = super().truth()
        d["node"], d["gpu"] = None, None
        return d
