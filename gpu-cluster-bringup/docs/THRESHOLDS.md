# Gate thresholds and why they are what they are

Every number lives in a manifest (`manifests/*.json`), not in code. The numbers in the
sample manifests are **placeholders of the right magnitude** for HGX H100 / MI300X
nodes. They are not vendor specifications. Re-baseline them on known-good nodes of
your own fleet before you trust a verdict.

## Node stage: binary facts, zero tolerance

| Check | Threshold | Why |
|---|---|---|
| `gpu_inventory` | exactly `gpu.count` GPUs visible to the driver; every GPU on the PCIe bus must be one of them | An 8-GPU job can't run on 7. The lspci cross-check separates "the GPU is dead" (`rev ff`, still on the bus) from "the slot is empty", and those go to different teams. |
| `gpu_memory_health` | volatile uncorrectable ECC `> 0` fails; row-remap pending or retirement pending fails; remap failure fails; correctable `> 1000` warns | An uncorrectable error since boot means the next job may already have wrong data. A pending remap is only applied at reset, so until then the bad row is still in use. Correctable errors are expected at low rates; a high count is an early warning, not a reason to cordon. |
| `pcie_link` | current **width** == expected; slot max gen (`max_host_link_gen`) ≥ expected; current **gen** is ignored | GPUs drop to gen1 at idle to save power, so gen measured at idle tells you nothing (`test_idle_gen1_is_not_a_downtrain`). Width doesn't change with power state, so x8 at idle is a real down-train. A slot capped below the device's gen is a BIOS or riser fault, and you can see it at idle. |
| `gpu_config` | persistence on (NVIDIA), power limit == golden ±1 W, max SM clock == golden, no HW slowdown / thermal / power-brake reason active at idle | A power limit lowered at the BMC or by a leftover `nvidia-smi -pl` makes the node pass functional tests and then lose every collective. A HW slowdown that's active at idle points to a PSU or sensor problem. |
| `software_versions` | driver, CUDA/ROCm, kernel are exact matches; VBIOS and HCA firmware come from an allow-list | A stack that differs on one node causes the hardest failures to debug: NCCL hangs and numerics that differ by rank. |
| `fabric_ports` | compute HCA count == golden; every port `Active`/`LinkUp` at the golden rate; Ethernet-layer NICs are excluded | A NDR port that trains at 200 Gb/s halves that rail's bandwidth, and nothing errors. |
| `gpu_nic_affinity` | every GPU gets a **distinct** compute NIC at `PIX`/`PXB` (NVIDIA) or on the same NUMA node (AMD, weaker evidence) | GPUDirect RDMA through the root complex or across UPI costs bandwidth and adds jitter on that rail. Matching is done by topology, not by index, so a renumbered GPU doesn't cause a false failure. |

## Burn-in stage: under load, with margin

| Check | Threshold | Why |
|---|---|---|
| `dcgm_diag` | every `dcgmi diag -r 3` test is Pass or Skip | Level 3 runs targeted power and stress, which is where weak power delivery shows up (the `dcgm_power_fail` defect). AMD is an explicit **SKIP** until RVS output is parsed. A skipped gate is visible in the report; a missing one would not be. |
| `thermal` | peak GPU temp ≤ 87 °C (H100 placeholder) / 95 °C (MI300X junction placeholder); HW thermal + power-brake throttle ≤ 30 s over the soak | Set a few degrees below the GPU's slowdown threshold (nvidia-smi reports it as `gpu_temp_slow_threshold`). A GPU that throttles during a 4 h soak will throttle during a 4-week training run, and it drags every rank with it. |
| `xid_events` | none of Xid 48, 62, 63, 64, 74, 79, 92, 94, 95, 119, 120, 123 | These indicate hardware or driver faults: DBE, row-remap events, NVLink errors, fallen off the bus, GSP timeouts. Xid 13/31/43 are usually application faults, so they are ignored here (`test_non_critical_xid_ignored`). Check the list against NVIDIA's current Xid catalogue for your driver branch. |
| `burnin_perf` | each GPU ≥ `min_gemm_tflops`, and within 5% of the median of the other GPUs in its node | The absolute floor catches a GPU that's broken. The sibling comparison catches one that's merely slow, which is the GPU that becomes the straggler. |

## Domain stage: the NVLink / XGMI domain

| Check | Threshold | Why |
|---|---|---|
| `nvlink_topology` | every GPU pair `NV18` (H100 SXM, 18 NVLink4 links) / `XGMI`; NVSwitch fabric state `Completed` | One bad GPU shows as degraded links to all of its peers. The gate reduces that to one line naming the GPU, which is what the vendor needs for the RMA. |
| `allreduce_single_node` | peak bus bandwidth ≥ 400 GB/s (H100 placeholder) / 290 GB/s (MI300X placeholder); `#wrong == 0` | Set at roughly 85–90% of what known-good nodes reach. `#wrong > 0` is data corruption, and it fails no matter how fast the run was. |

## Cluster stage: relative to the fleet

| Check | Threshold | Why |
|---|---|---|
| `performance_outlier` | node median GEMM no more than 6% below the pool median (pool of ≥ 3 accepted nodes) | Every collective runs at the speed of its slowest rank. The `straggler` defect clears every absolute floor and is still 8% slow, and only the fleet comparison can show that. |
| `capacity[pool]` | accepted nodes ≥ `pool.min_accepted_nodes` | This comes from the first workload's needs (20 × 8 = 160 GPUs in the sample), not from a percentage. Below it, the cluster is **REJECTED** even if each cordon was correct. |
| `allreduce_multi_node[pool]` | peak busbw ≥ `multi_node_min_busbw_gbps`, `#wrong == 0` | Node gates can't see fabric-wide problems such as adaptive routing being off, a mis-set PKey, or an oversubscribed spine. |
| `homogeneity[pool]` | one (driver, runtime, kernel) tuple among accepted nodes | A second check on the golden pins, in case a manifest is ever loosened to a range. |

## Rack correlation

If the same check fails on **≥ 2 nodes and ≥ 50%** of a rack, the report raises one
rack-level finding. The usual causes are a leaf switch, a cable bundle, a PDU or a CDU.
Each node is still cordoned, but the finding goes to network or facilities before
anyone opens per-node RMAs (`test_leaf_switch_failure_reported_as_one_rack_finding`).

## Exit codes

| Decision | Exit | Meaning for CI / Ansible |
|---|---|---|
| `ACCEPTED` | 0 | Every node accepted. |
| `ACCEPTED_DEGRADED` | 3 | Cluster gates pass and some nodes are cordoned with reasons. Usable once the capacity owner signs off. |
| `REJECTED` | 1 | A cluster gate failed: not enough capacity, or the fabric is below its floor. |
| usage error | 2 | argparse's default, so a typo can't be read as a verdict. |
