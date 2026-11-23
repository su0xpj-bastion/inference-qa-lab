# Fabric validation runbook

Validate the fabric from the inside out, so each stage proves only one new thing. A cluster-wide all_reduce that comes in slow tells you *that* something is wrong. The earlier stages tell you *where*.

Everything lands in one run directory that `python -m fabval analyze <run_dir>` reads:

```
<run_dir>/cluster.json                    {"node_type": "...", "nodes": [...], "nccl_env": {...}, "source": "capture"}
<run_dir>/nodes/<node>/...                stage 0 + 1 (scripts/collect_node.sh)
<run_dir>/pairs/<a>__<b>__rail<r>.sendrecv_perf.txt   stage 2 (launch/slurm/pairwise.sh)
<run_dir>/collective/all_reduce_perf.txt  stage 3 (any launcher)
<run_dir>/collective/nccl_debug.log       stage 3, NCCL_DEBUG=INFO stderr
```

## Stage 0: node health (no traffic)

`srun --ntasks-per-node=1 scripts/collect_node.sh <run_dir>` clears the IB counters (`perfquery -R`), then captures the following:

| Check | Tool | Fails when |
|---|---|---|
| Port up at full speed | `ibstat` | State ≠ Active, phys ≠ LinkUp, Rate < nominal (e.g. 200 on an NDR400 port), wrong link layer |
| Physical errors (IB) | `perfquery` | SymbolErrorCounter > 10, any LinkDowned / LinkErrorRecovery / PortRcvErrors / LocalLinkIntegrity / BufferOverrun |
| Drops and lossless config (RoCE) | `ethtool -S` | `rx/tx_discards_phy` or `rx_out_of_buffer` > 1000, any CRC or symbol errors. Zero pause frames on the RoCE priority or zero ECN marks are reported as the cause |
| GID index (RoCE) | `show_gids` | `NCCL_IB_GID_INDEX` doesn't point to a RoCEv2 IPv4 GID on every HCA |
| GPU↔NIC affinity | `nvidia-smi topo -m` | GPU*n*↔NIC*n* is anything other than PIX/PXB |

Fix stage 0 before you run any traffic. A down-trained port shows up here in seconds, but in an all_reduce it takes an hour to find.

## Stage 1: single node (NVLink only)

`all_reduce_perf -b 8 -e 8G -f 2 -g 8 -c 1` on each node (also run by `collect_node.sh`). The run is gated at 85% of the NVLink-bound expectation, small-message latency is gated against the budget, and `#wrong` must be 0. This is the stage that can attribute data corruption to a node, because the full collective can't.

## Stage 2: pairs, per rail

`launch/slurm/pairwise.sh <run_dir>` runs `sendrecv_perf` for every node pair and every rail, with one rank per node, `CUDA_VISIBLE_DEVICES=r` and `NCCL_IB_HCA==mlx5_r`. Each result then involves exactly two ports.

- Bandwidth gate: the peak busbw at ≥ 256 MiB must be ≥ 85% of one NIC × p2p efficiency.
- Latency gate: the median time at ≤ 64 KiB must be within the node type's budget. This is the only gate that sees a lossless-config fault (drops plus go-back-N retransmits). The bandwidth gate passes it at ~92%.
- Localization: a greedy cover picks the (node, rail) that explains the most failing tests. It needs ≥ 2 failures and failures in ≥ 50% of that endpoint's tests. A healthy partner fails once per bad peer and is never blamed. When all rails of a node are blamed, the result is reported as the node. A single failing test is reported as a link (cable, optics, switch port).

Cost: N(N-1)/2 × 8 runs. Pairs on disjoint nodes can run concurrently, so 8 nodes means 224 runs in about 7 rounds.

## Stage 3: full collective

Run all ranks with `NCCL_DEBUG=INFO` through any launcher (`launch/slurm/nccl_tests.sbatch`, `launch/k8s/mpijob-nccl-tests-*.yaml`, `launch/ray/submit.sh`). The analyzer checks:

- busbw against `min(NVLink, 8 × NIC × ring efficiency)`. Collectives split evenly across channels, so one slow rail anywhere caps every rank.
- `#wrong` must be 0.
- The NCCL log: every host on `NET/IB`, never `NET/Socket`; `/GDRDMA` on every network connection; `NCCL_*` echoes identical across hosts.

## Triage table

| Symptom | First look |
|---|---|
| One rail slow from one node | `ibstat` rate, `perfquery` symbol errors → reseat or replace the cable or optic |
| One rail slow, ibstat clean | `nvidia-smi topo -m` (SYS/NODE), NCCL log GDR lines → BIOS/PCIe slot, `NCCL_NET_GDR_LEVEL` |
| Small messages slow, bandwidth fine (RoCE) | `ethtool -S` pause/ECN/discards → switch and NIC QoS (PFC on priority 3, DSCP 26 trust, ECN thresholds) |
| A whole node at socket speed | NCCL log `Using network Socket` → `show_gids`, `/dev/infiniband` in the container, the RDMA device plugin |
| `#wrong > 0` | Drain the node named by stage 1, then run DCGM diag level 3 |
