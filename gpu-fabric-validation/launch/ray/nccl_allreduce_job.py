from __future__ import annotations

# NCCL all_reduce across Ray GPU actors, one actor per GPU, printed in nccl-tests layout so the
# same parser and gates (fabval.nccl_tests, fabval.model) apply to Ray-launched runs.
#
#   launch/ray/submit.sh                  # builds the runtime_env from launch/nccl-roce.env
#   python nccl_allreduce_job.py --nodes 8 --gpus-per-node 8 > all_reduce_perf.txt
#
# Placement: one STRICT_SPREAD bundle of 8 GPUs per node, actors pinned to bundles in rank
# order, so ranks 0-7 share a node exactly like an MPI ppr:8:node launch.
#
# ray.util.collective's allreduce is in-place only. Both nccl-tests column groups carry that
# in-place measurement, so a parser that expects 13 columns reads the file unchanged.
#
# The module imports neither ray nor torch at import time; the pure helpers below are unit
# tested without a GPU (tests/unit/test_launchers.py).

import argparse
import json
import socket
import sys
import time
from pathlib import Path

ENCODING = "utf-8"
GROUP = "fabric-allreduce"
WARMUP_ITERS = 5
ITERS = 20
BUSBW_FACTOR = {"all_reduce": lambda n: 2 * (n - 1) / n}


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    with path.open("r", encoding=ENCODING) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def runtime_env(env_file: Path, socket_ifname: str = "eth0") -> dict:
    env = load_env_file(env_file)
    env["NCCL_SOCKET_IFNAME"] = socket_ifname
    return {"env_vars": env}


def size_sweep(lo: int, hi: int, factor: int = 2) -> list[int]:
    out, s = [], lo
    while s <= hi:
        out.append(s)
        s *= factor
    return out


def fmt_time(us: float) -> str:
    if us >= 10000.0:
        return f"{us:7.0f}"
    if us >= 100.0:
        return f"{us:7.1f}"
    return f"{us:7.2f}"


def format_row(size: int, nranks: int, time_us: float, wrong: int) -> str:
    ts = fmt_time(time_us)
    alg = size / float(ts) / 1e3
    bus = alg * BUSBW_FACTOR["all_reduce"](nranks)
    cell = f"  {ts}  {alg:6.2f}  {bus:6.2f}  {wrong:5d}"
    return f"{size:12d}  {size // 4:12d}  {'float':>8s}  {'sum':>6s}  {-1:6d}" + cell + cell


def format_header(lo: int, hi: int, devices: list[tuple[str, int]]) -> str:
    lines = [f"# nThread 1 nGpus 1 minBytes {lo} maxBytes {hi} step: 2(factor) warmup iters: {WARMUP_ITERS} "
             f"iters: {ITERS} agg iters: 1 validation: 1 graph: 0", "#", "# Using devices"]
    for rank, (host, dev) in enumerate(devices):
        lines.append(f"#  Rank {rank:2d} Group  0 Pid {0:6d} on {host:>10s} device {dev:2d} [0x00] ray-actor")
    lines += ["#",
              "#                                                              out-of-place                       in-place          ",
              "#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong",
              "#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       "]
    return "\n".join(lines)


def make_rank_class(ray):  # pragma: no cover - needs GPUs
    @ray.remote(num_gpus=1)
    class CollectiveRank:
        def __init__(self, world: int, rank: int):
            import ray.util.collective as col
            import torch

            self.col, self.torch, self.world, self.rank = col, torch, world, rank
            col.init_collective_group(world, rank, backend="nccl", group_name=GROUP)

        def where(self) -> tuple[str, int]:
            return socket.gethostname(), int(self.torch.cuda.current_device())

        def run(self, size: int) -> tuple[float, int]:
            t = self.torch.ones(max(size // 4, 1), dtype=self.torch.float32, device="cuda")
            for _ in range(WARMUP_ITERS):
                self.col.allreduce(t, group_name=GROUP)
            self.torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(ITERS):
                self.col.allreduce(t, group_name=GROUP)
            self.torch.cuda.synchronize()
            elapsed_us = (time.perf_counter() - start) * 1e6 / ITERS
            # Validation pass: rank r contributes r+1, so every element must equal n(n+1)/2.
            t.fill_(float(self.rank + 1))
            self.col.allreduce(t, group_name=GROUP)
            want = self.world * (self.world + 1) / 2
            wrong = int((t != want).sum().item())
            return elapsed_us, wrong

    return CollectiveRank


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a Ray cluster with GPUs
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--min-bytes", type=int, default=8)
    p.add_argument("--max-bytes", type=int, default=8 << 30)
    p.add_argument("--print-runtime-env", metavar="ENV_FILE")
    args = p.parse_args(argv)

    if args.print_runtime_env:
        print(json.dumps(runtime_env(Path(args.print_runtime_env))))
        return 0

    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    ray.init(address="auto")
    g, world = args.gpus_per_node, args.nodes * args.gpus_per_node
    pg = placement_group([{"GPU": g, "CPU": g}] * args.nodes, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    rank_cls = make_rank_class(ray)
    actors = [
        rank_cls.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=r // g)).remote(world, r)
        for r in range(world)
    ]
    devices = ray.get([a.where.remote() for a in actors])
    print(format_header(args.min_bytes, args.max_bytes, devices))
    busbws, total_wrong = [], 0
    for size in size_sweep(args.min_bytes, args.max_bytes):
        results = ray.get([a.run.remote(size) for a in actors])
        t_us = sum(r[0] for r in results) / len(results)  # nccl-tests default: average over ranks
        wrong = sum(r[1] for r in results)
        total_wrong += wrong
        row = format_row(size, world, t_us, wrong)
        busbws.append(float(row.split()[7]))
        print(row, flush=True)
    print(f"# Out of bounds values : {total_wrong} {'OK' if total_wrong == 0 else 'FAILED'}")
    print(f"# Avg bus bandwidth    : {sum(busbws) / len(busbws):.4f} ")
    print("#")
    return 0 if total_wrong == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
