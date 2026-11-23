# NCCL environment variables that matter for fabric validation

The shared values live in [`launch/nccl-roce.env`](../launch/nccl-roce.env) and [`launch/nccl-ib.env`](../launch/nccl-ib.env). Slurm sources the file, the Ray submit script turns it into a `runtime_env`, and the k8s ConfigMap mirrors it (enforced by `test_configmap_matches_env_file`).

| Variable | Value here | Why it matters |
|---|---|---|
| `NCCL_DEBUG` | `INFO` | Without it there's no evidence of transport (`NET/IB` vs `NET/Socket`), GDR, or channel layout. A silent socket fallback looks like "the cluster is slow". |
| `NCCL_DEBUG_SUBSYS` | `INIT,NET,GRAPH` | Enough to see the network choice, connections and topology without per-op spam. Add `TUNING` when you're chasing algorithm choice. |
| `NCCL_IB_HCA` | `^=mlx5_bond_0` | Excludes the frontend/storage bond so NCCL never puts collective traffic on it. An exclusion is used instead of an explicit `mlx5_0..7` list because SR-IOV VFs in a pod carry host-assigned names (`mlx5_12`, ...), so an explicit list would match nothing there. For per-rail pair tests, set `=mlx5_<r>` (exact match) to force one HCA. |
| `NCCL_SOCKET_IFNAME` | per launcher (`eth0` in pods, `bond0` on hosts) | The bootstrap/OOB interface, and the interface used by the socket transport if IB fails. Never `lo` or `docker0`. This is the one variable that legitimately differs between launchers. |
| `NCCL_IB_GID_INDEX` | `3` (RoCE) | Selects the RoCEv2 IPv4 GID. The value is per job, but GID tables are per host, so an extra IPv6 or VLAN address on one node shifts its table. Check every node with `show_gids` (stage 0). NCCL ≥ 2.21 can auto-select with `NCCL_IB_ADDR_FAMILY` / `NCCL_IB_ROCE_VERSION_NUM` instead. |
| `NCCL_IB_TC` | `106` (RoCE) | Traffic class = DSCP 26 << 2 \| ECN-capable. This has to match the switch's lossless class, or RoCE traffic lands in a lossy queue: drops, retransmits and tail latency. |
| `NCCL_IB_QPS_PER_CONNECTION` | `4` (RoCE) | More QPs mean more UDP source ports, which gives ECMP more entropy across spine links. With one QP, a single elephant flow can pin to one link. |
| `NCCL_CROSS_NIC` | `0` | Rail-optimized fabric: GPU *n* on every node talks over NIC *n*, so traffic stays on one rail switch. `1` lets rings cross rails through the spine. |
| `NCCL_NET_GDR_LEVEL` | `PXB` | GPUDirect RDMA only when the GPU and NIC share a PCIe switch. Raising it to `SYS` hides a mis-slotted NIC behind slow cross-socket DMA. Keeping the default makes that NIC visible as "GDR disabled". |
| `NCCL_TOPO_DUMP_FILE` | unset (set while debugging) | Dumps the topology NCCL detected. Compare it with `nvidia-smi topo -m` when affinity looks wrong. |
| `NCCL_IB_DISABLE` / `NCCL_P2P_DISABLE` / `NCCL_SHM_DISABLE` | unset | Use these only to isolate a layer during triage. Never ship them: each one silently forces a slower path. |
| `NCCL_ALGO` / `NCCL_PROTO` | unset | Pin them (`Ring`, `Tree`, `Simple`) only to compare against the ring/tree expectation in `fabval expected --algo`. Production should let NCCL tune. |
