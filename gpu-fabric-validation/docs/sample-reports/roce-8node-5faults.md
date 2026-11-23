## Fabric report: FAIL

`h100-sxm-8x-400g-roce` x 8 nodes, source: **simulated**. Exit code 1 (0 pass, 1 fail, 2 inconclusive).

### Root causes

| Location | Diagnosis | Evidence |
|---|---|---|
| `node-02/mlx5_6` | NIC on the wrong NUMA node (GDR off) | rail 6 (mlx5_6) fails the pair bandwidth gate in 7/7 tests (limit 85% of 46.0 GB/s)<br>GPU6 <-> NIC6 (mlx5_6) is SYS (want PIX/PXB): NIC on the wrong PCIe root / NUMA node<br>GPUDirect RDMA off on mlx5_6: connections run via NET/IB/6 without /GDRDMA |
| `node-03/mlx5_2` | link down-trained / port not healthy | rail 2 (mlx5_2) fails the pair bandwidth gate in 7/7 tests (limit 85% of 46.0 GB/s)<br>mlx5_2 rate 200 Gb/s, expected 400 (link down-trained) |
| `node-04` | intra-node all_reduce returned wrong results (19920 elements across 5 sizes) | · |
| `node-05/mlx5_4` | RoCE lossless config (PFC/ECN) not in effect | rail 4 (mlx5_4) fails the pair latency gate in 7/7 tests (limit 30 us)<br>mlx5_4 rx_discards_phy=2148331; no PFC pause frames on priority 3, no ECN marks -> lossless config (PFC/ECN) not in effect |
| `node-07` | RoCE GID index mismatch | every rail fails the pair bandwidth gate in 56/56 tests (limit 85% of 46.0 GB/s)<br>NCCL fell back to TCP sockets (NET/Socket): the RDMA fabric is bypassed<br>mlx5_0 GID index 3 is v2 link-local/IPv6; RoCE v2 IPv4 GID is at index 5<br>mlx5_1 GID index 3 is v2 link-local/IPv6; RoCE v2 IPv4 GID is at index 5 |

### Stages

| Stage | Result |
|---|---|
| 0. node health (ibstat, perfquery/ethtool, show_gids, topo) | 4/8 nodes clean |
| 1. single node all_reduce (NVLink, expected 360.0 GB/s) | worst node-05 at 98.3% |
| 2. pairwise sendrecv per rail (expected 46.0 GB/s, budget 30 us) | 224 tests, 68 bandwidth fails, 62 latency fails |
| 3. full all_reduce (64 ranks, expected 360.0 GB/s) | peak 20.8 GB/s = 5.8%, #wrong 14156 |

### Pair matrix

| min busbw % of expected / worst small-msg us | node-01 | node-02 | node-03 | node-04 | node-05 | node-06 | node-07 | node-08 |
|---|---|---|---|---|---|---|---|---|
| node-01 | · | 63% / 11 **x** | 50% / 11 **x** | 97% / 11 | 95% / 66 **x** | 98% / 11 | 6% / 37 **x** | 99% / 11 |
| node-02 | 63% / 11 **x** | · | 52% / 11 **x** | 64% / 11 **x** | 61% / 69 **x** | 61% / 11 **x** | 6% / 37 **x** | 63% / 11 **x** |
| node-03 | 50% / 11 **x** | 52% / 11 **x** | · | 52% / 11 **x** | 52% / 68 **x** | 52% / 11 **x** | 6% / 36 **x** | 49% / 11 **x** |
| node-04 | 97% / 11 | 64% / 11 **x** | 52% / 11 **x** | · | 94% / 64 **x** | 98% / 11 | 6% / 38 **x** | 97% / 11 |
| node-05 | 95% / 66 **x** | 61% / 69 **x** | 52% / 68 **x** | 94% / 64 **x** | · | 90% / 78 **x** | 6% / 38 **x** | 94% / 77 **x** |
| node-06 | 98% / 11 | 61% / 11 **x** | 52% / 11 **x** | 98% / 11 | 90% / 78 **x** | · | 6% / 36 **x** | 98% / 11 |
| node-07 | 6% / 37 **x** | 6% / 37 **x** | 6% / 36 **x** | 6% / 38 **x** | 6% / 38 **x** | 6% / 36 **x** | · | 6% / 38 **x** |
| node-08 | 99% / 11 | 63% / 11 **x** | 49% / 11 **x** | 97% / 11 | 94% / 77 **x** | 98% / 11 | 6% / 38 **x** | · |

### All findings

| Sev | Stage | Location | Finding |
|---|---|---|---|
| FAIL | node health | `node-02/mlx5_6` | GPU6 <-> NIC6 (mlx5_6) is SYS (want PIX/PXB): NIC on the wrong PCIe root / NUMA node |
| FAIL | node health | `node-03/mlx5_2` | mlx5_2 rate 200 Gb/s, expected 400 (link down-trained) |
| FAIL | node health | `node-03/mlx5_2` | mlx5_2 physical-layer errors: rx_symbol_err_phy=312 |
| FAIL | node health | `node-05/mlx5_4` | mlx5_4 rx_discards_phy=2148331; no PFC pause frames on priority 3, no ECN marks -> lossless config (PFC/ECN) not in effect |
| FAIL | node health | `node-07/mlx5_0` | mlx5_0 GID index 3 is v2 link-local/IPv6; RoCE v2 IPv4 GID is at index 5 |
| FAIL | node health | `node-07/mlx5_1` | mlx5_1 GID index 3 is v2 link-local/IPv6; RoCE v2 IPv4 GID is at index 5 |
| FAIL | node health | `node-07/mlx5_2` | mlx5_2 GID index 3 is v2 link-local/IPv6; RoCE v2 IPv4 GID is at index 5 |
| FAIL | node health | `node-07/mlx5_3` | mlx5_3 GID index 3 is v2 link-local/IPv6; RoCE v2 IPv4 GID is at index 5 |
| FAIL | node health | `node-07/mlx5_4` | mlx5_4 GID index 3 is v2 link-local/IPv6; RoCE v2 IPv4 GID is at index 5 |
| FAIL | node health | `node-07/mlx5_5` | mlx5_5 GID index 3 is v2 link-local/IPv6; RoCE v2 IPv4 GID is at index 5 |
| FAIL | node health | `node-07/mlx5_6` | mlx5_6 GID index 3 is v2 link-local/IPv6; RoCE v2 IPv4 GID is at index 5 |
| FAIL | node health | `node-07/mlx5_7` | mlx5_7 GID index 3 is v2 link-local/IPv6; RoCE v2 IPv4 GID is at index 5 |
| FAIL | single node | `node-04` | intra-node all_reduce returned wrong results (19920 elements across 5 sizes) |
| FAIL | pairs | `node-02/mlx5_6` | rail 6 (mlx5_6) fails the pair bandwidth gate in 7/7 tests (limit 85% of 46.0 GB/s) |
| FAIL | pairs | `node-03/mlx5_2` | rail 2 (mlx5_2) fails the pair bandwidth gate in 7/7 tests (limit 85% of 46.0 GB/s) |
| FAIL | pairs | `node-07` | every rail fails the pair bandwidth gate in 56/56 tests (limit 85% of 46.0 GB/s) |
| FAIL | pairs | `node-05/mlx5_4` | rail 4 (mlx5_4) fails the pair latency gate in 7/7 tests (limit 30 us) |
| FAIL | pairs | `node-07` | every rail fails the pair latency gate in 56/56 tests (limit 30 us) |
| FAIL | full collective | `cluster` | full all_reduce busbw 20.8 GB/s = 6% of expected 360 GB/s (8 nodes x 8 GPUs, ring) |
| FAIL | full collective | `cluster` | full all_reduce returned wrong results (14156 elements) |
| WARN | full collective | `node-02/mlx5_6` | GPUDirect RDMA off on mlx5_6: connections run via NET/IB/6 without /GDRDMA |
| FAIL | full collective | `node-07` | NCCL fell back to TCP sockets (NET/Socket): the RDMA fabric is bypassed |
