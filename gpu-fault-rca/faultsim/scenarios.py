"""Scenario library: named fault sets + load profile, each with ground truth."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from faultsim import faults as F
from faultsim.sim import Action, Sim


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    make: Callable[[], list]
    load: str = "steady"
    duration: int = 600
    stray_xid31: bool = False
    tags: tuple[str, ...] = field(default=())


SCENARIOS: dict[str, Scenario] = {s.name: s for s in (
    Scenario("no_fault", "Background noise only: correctable SBEs, a stray Xid 31 from a profiler, a latency blip",
             lambda: [], stray_xid31=True, tags=("noise",)),
    Scenario("node_power_loss", "gpu-node-3 loses power: telemetry stops mid-stream, peers' NCCL times out",
             lambda: [F.NodeLoss("gpu-node-3", onset=200)], tags=("red-herring",)),
    Scenario("gpu_fallen_off_bus", "Xid 79 on gpu-node-1 GPU 3; 15 healthy ranks report NCCL timeouts",
             lambda: [F.GpuLost("gpu-node-1", 3, onset=200)], tags=("red-herring",)),
    Scenario("ecc_uncorrectable", "Xid 94 + 63 on gpu-node-2 GPU 6; worker dies and recurs after restart",
             lambda: [F.EccUncorrectable("gpu-node-2", 6, onset=200)]),
    Scenario("nvlink_degraded", "NVLink CRC errors on gpu-node-1 GPU 2, visible only above 35% link load",
             lambda: [F.NvlinkDegraded("gpu-node-1", 2, onset=60)], load="ramp", tags=("stress-only",)),
    Scenario("ib_link_degraded", "Bad IB link on gpu-node-1; NET/IB warnings are logged by gpu-node-0",
             lambda: [F.NetworkDegraded("gpu-node-1", onset=200)], tags=("red-herring",)),
    Scenario("driver_partial_upgrade", "Rolling driver upgrade stalls on gpu-node-2: 560 userspace on 550 module",
             lambda: [F.DriverMismatch("gpu-node-2", onset=200)], tags=("red-herring",)),
    Scenario("thermal_throttle", "Cooling fault on gpu-node-3 GPU 5; straggler slows the whole TP group",
             lambda: [F.ThermalThrottle("gpu-node-3", 5, onset=150)], tags=("stress-only",)),
    Scenario("worker_memory_leak", "Worker on gpu-node-0 GPU 2 leaks 5 MiB/request until CUDA OOM",
             lambda: [F.MemoryLeak("gpu-node-0", 2, onset=60, mib_per_request=5.0)]),
    Scenario("noisy_neighbour_cpu", "Batch pod lands on gpu-node-2; engine CPU threads starve, GPUs idle",
             lambda: [F.HostCpuContention("gpu-node-2", onset=200)]),
    Scenario("bad_engine_build", "engine:v0.9.1 rolled to replica-a: Xid 13/31 on six GPUs across two nodes",
             lambda: [F.BadEngineBuild(replica="replica-a", onset=200)]),
    Scenario("combo_xid79_noisy_neighbour", "Xid 79 on gpu-node-1 and a noisy neighbour on gpu-node-2",
             lambda: [F.GpuLost("gpu-node-1", 3, onset=240), F.HostCpuContention("gpu-node-2", onset=200)],
             tags=("combined",)),
    Scenario("combo_leak_thermal", "Memory leak on gpu-node-0 GPU 2 and thermal throttle on gpu-node-3 GPU 5",
             lambda: [F.MemoryLeak("gpu-node-0", 2, onset=60, mib_per_request=5.0),
                      F.ThermalThrottle("gpu-node-3", 5, onset=150)], tags=("combined",)),
    Scenario("soak_slow_leak", "0.6 MiB/request leak on gpu-node-2 GPU 4: a 10-minute run only sees a trend",
             lambda: [F.MemoryLeak("gpu-node-2", 4, onset=120, mib_per_request=0.6)],
             load="soak", duration=3000, tags=("soak-only", "slow")),
)}


def truth_for(scenario: Scenario, faults: list, load: str, seed: int) -> dict:
    return {"scenario": scenario.name, "load": load, "seed": seed,
            "root_causes": [f.truth() for f in faults]}


def run(name: str, out: Path, seed: int = 0, load: str | None = None, duration: int | None = None,
        actions: tuple[Action, ...] = ()) -> tuple[Path, Sim, dict]:
    sc = SCENARIOS[name]
    faults = sc.make()
    use_load = load or sc.load
    sim = Sim(faults, load=use_load, duration=duration or sc.duration, seed=seed, actions=actions,
              stray_xid31=sc.stray_xid31).run()
    truth = truth_for(sc, faults, use_load, seed)
    sim.write_bundle(out, truth)
    return out, sim, truth
