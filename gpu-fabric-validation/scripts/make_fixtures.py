from __future__ import annotations

# Regenerates the nccl-tests fixtures that are rendered by fabval.sim.format_nccl_tests.
# The hand-edited fixtures (legacy 12-column, -c 0 N/A, truncated) are not touched.
# Run once and commit; the parser tests pin values read from these files.

import random
from pathlib import Path

from fabval.sim import format_nccl_tests

ENCODING = "utf-8"
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "nccl_tests"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)
    two_nodes = [(f"gpu-a{n}", d) for n in (1, 2) for d in range(8)]
    files = {
        "all_reduce_perf.2node.healthy.txt": format_nccl_tests("all_reduce", two_nodes, 362.0, 31.0, (8, 8 << 30), rng),
        "all_reduce_perf.2node.degraded.txt": format_nccl_tests("all_reduce", two_nodes, 181.0, 31.0, (8, 8 << 30), rng),
        "all_reduce_perf.2node.wrong.txt": format_nccl_tests("all_reduce", two_nodes, 360.0, 31.0, (8, 8 << 30), rng,
                                                             wrong_sizes=frozenset({1 << 28, 1 << 32})),
        "all_gather_perf.2node.healthy.txt": format_nccl_tests("all_gather", two_nodes, 355.0, 28.0, (1024, 8 << 30), rng),
        "alltoall_perf.2node.healthy.txt": format_nccl_tests("alltoall", two_nodes, 41.0, 60.0, (1024, 4 << 30), rng),
        "sendrecv_perf.pair.healthy.txt": format_nccl_tests("sendrecv", [("gpu-a1", 3), ("gpu-a2", 3)], 46.4, 11.0, (8, 1 << 30), rng),
        "sendrecv_perf.pair.tail_latency.txt": format_nccl_tests("sendrecv", [("gpu-a1", 3), ("gpu-a2", 3)], 43.0, 11.0,
                                                                  (8, 1 << 30), rng, lat_jitter=(5.0, 9.0)),
    }
    for name, text in files.items():
        with (OUT / name).open("w", encoding=ENCODING) as fh:
            fh.write(text)
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
