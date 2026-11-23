# Bring-up report: sim-a

**Decision: 🟡 ACCEPTED_DEGRADED** (exit 3). 24/36 nodes accepted, 12 cordoned.

| Stage | Cordoned at this stage |
|---|---|
| node | 7 |
| burnin | 3 |
| domain | 1 |
| cluster | 1 |

## Pools

| Node type | Accepted | Required | Total | GPUs accepted |
|---|---|---|---|---|
| hgx-h100-ndr | 21 | 20 | 32 | 168 |
| mi300x-ndr | 3 | 3 | 4 | 24 |

## Cluster gates

| Check | Status | Detail |
|---|---|---|
| homogeneity[hgx-h100-ndr] | PASS |  |
| capacity[hgx-h100-ndr] | PASS | accepted=21, required=20 |
| allreduce_multi_node[hgx-h100-ndr] | PASS | peak_busbw_gbps=372.5, floor=340, nodes=21 |
| homogeneity[mi300x-ndr] | PASS |  |
| capacity[mi300x-ndr] | PASS | accepted=3, required=3 |
| allreduce_multi_node[mi300x-ndr] | PASS | peak_busbw_gbps=339.3, floor=300, nodes=3 |

## Cordoned nodes

| Node | Rack | Stage | Reason |
|---|---|---|---|
| `amd-r05-n02` | r05 | node | gpu_memory_health: GPU3 (0000:65:00.0) 2 uncorrectable ECC errors since boot |
| `gpu-r01-n02` | r01 | node | gpu_inventory: 7/8 GPUs visible to the driver; 0000:9a:00.0 is on the PCIe bus (rev ff: config space unreadable) but not enumerated by the driver: fallen off the bus, reseat/RMA |
| `gpu-r01-n04` | r01 | node | gpu_memory_health: GPU3 (0000:5d:00.0) 2 uncorrectable ECC errors since boot; GPU3 (0000:5d:00.0) row remap pending: GPU reset required, then re-run acceptance |
| `gpu-r01-n06` | r01 | node | pcie_link: GPU5 (0000:ab:00.0) PCIe link x8, expected x16: down-trained, reseat/check riser |
| `gpu-r01-n08` | r01 | domain | nvlink_topology: GPU1 has degraded links to all 7 peers (e.g. GPU0<->GPU1 NV12, expected NV18): NVLink/NVSwitch fault on that GPU's baseboard position<br>allreduce_single_node: single-node all_reduce busbw 292 GB/s < 400 |
| `gpu-r02-n01` | r02 | node | software_versions: driver 550.90.07 != golden 550.127.08 |
| `gpu-r02-n03` | r02 | node | fabric_ports: mlx5_4 port 1 State=Down PhysState=Polling: check cable/transceiver and switch port |
| `gpu-r02-n05` | r02 | node | gpu_nic_affinity: GPU6 has no dedicated compute NIC within PIX/PXB (closest mlx5_4 is NODE: crosses host bridges inside the NUMA node); its GPUDirect RDMA traffic would leave the local PCIe switch; mlx5_6 is not within PIX/PXB of any GPU: wrong slot, or rail cabling/numbering swapped |
| `gpu-r02-n07` | r02 | burnin | dcgm_diag: dcgmi diag -r 3 'Targeted Power' failed on GPU 5 (GPU 5 Error: Target power 700 W could not be reached: max observed 552 W. Check power delivery and cooling.) |
| `gpu-r03-n03` | r03 | burnin | thermal: GPU2 peaked at 92 C (limit 87), hw_thermal_slowdown 340s: check cold plate, fans, airflow |
| `gpu-r04-n02` | r04 | cluster | performance_outlier: node median GEMM 656 TFLOPS is 8.3% below the hgx-h100-ndr pool median 715 (limit 6.0%): slowest rank gates every collective |
| `gpu-r04-n05` | r04 | burnin | xid_events: Xid 79 on PCI 0000:2a:00 during burn-in: GPU has fallen off the bus. |

<details><summary>Cordon commands (not executed)</summary>

```bash
scontrol update NodeName=amd-r05-n02 State=DRAIN Reason="bringup:node:gpu_memory_health"
kubectl taint node amd-r05-n02 bringup.gpu/rejected=node:NoSchedule --overwrite
scontrol update NodeName=gpu-r01-n02 State=DRAIN Reason="bringup:node:gpu_inventory"
kubectl taint node gpu-r01-n02 bringup.gpu/rejected=node:NoSchedule --overwrite
scontrol update NodeName=gpu-r01-n04 State=DRAIN Reason="bringup:node:gpu_memory_health"
kubectl taint node gpu-r01-n04 bringup.gpu/rejected=node:NoSchedule --overwrite
scontrol update NodeName=gpu-r01-n06 State=DRAIN Reason="bringup:node:pcie_link"
kubectl taint node gpu-r01-n06 bringup.gpu/rejected=node:NoSchedule --overwrite
scontrol update NodeName=gpu-r01-n08 State=DRAIN Reason="bringup:domain:nvlink_topology,allreduce_single_node"
kubectl taint node gpu-r01-n08 bringup.gpu/rejected=domain:NoSchedule --overwrite
scontrol update NodeName=gpu-r02-n01 State=DRAIN Reason="bringup:node:software_versions"
kubectl taint node gpu-r02-n01 bringup.gpu/rejected=node:NoSchedule --overwrite
scontrol update NodeName=gpu-r02-n03 State=DRAIN Reason="bringup:node:fabric_ports"
kubectl taint node gpu-r02-n03 bringup.gpu/rejected=node:NoSchedule --overwrite
scontrol update NodeName=gpu-r02-n05 State=DRAIN Reason="bringup:node:gpu_nic_affinity"
kubectl taint node gpu-r02-n05 bringup.gpu/rejected=node:NoSchedule --overwrite
scontrol update NodeName=gpu-r02-n07 State=DRAIN Reason="bringup:burnin:dcgm_diag"
kubectl taint node gpu-r02-n07 bringup.gpu/rejected=burnin:NoSchedule --overwrite
scontrol update NodeName=gpu-r03-n03 State=DRAIN Reason="bringup:burnin:thermal"
kubectl taint node gpu-r03-n03 bringup.gpu/rejected=burnin:NoSchedule --overwrite
scontrol update NodeName=gpu-r04-n02 State=DRAIN Reason="bringup:cluster:performance_outlier"
kubectl taint node gpu-r04-n02 bringup.gpu/rejected=cluster:NoSchedule --overwrite
scontrol update NodeName=gpu-r04-n05 State=DRAIN Reason="bringup:burnin:xid_events"
kubectl taint node gpu-r04-n05 bringup.gpu/rejected=burnin:NoSchedule --overwrite
```

</details>
