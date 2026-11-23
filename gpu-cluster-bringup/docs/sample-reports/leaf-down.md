# Bring-up report: sim-a

**Decision: 🟡 ACCEPTED_DEGRADED** (exit 3). 31/36 nodes accepted, 5 cordoned.

| Stage | Cordoned at this stage |
|---|---|
| node | 5 |
| burnin | 0 |
| domain | 0 |
| cluster | 0 |

## Pools

| Node type | Accepted | Required | Total | GPUs accepted |
|---|---|---|---|---|
| hgx-h100-ndr | 27 | 20 | 32 | 216 |
| mi300x-ndr | 4 | 3 | 4 | 32 |

## Cluster gates

| Check | Status | Detail |
|---|---|---|
| homogeneity[hgx-h100-ndr] | PASS |  |
| capacity[hgx-h100-ndr] | PASS | accepted=27, required=20 |
| allreduce_multi_node[hgx-h100-ndr] | PASS | peak_busbw_gbps=372.5, floor=340, nodes=27 |
| homogeneity[mi300x-ndr] | PASS |  |
| capacity[mi300x-ndr] | PASS | accepted=4, required=3 |
| allreduce_multi_node[mi300x-ndr] | PASS | peak_busbw_gbps=339.3, floor=300, nodes=4 |

## Rack-level findings

- **r03 / fabric_ports**: 5/8 nodes in rack r03 fail fabric_ports: suspect a shared cause (leaf switch, cabling bundle, PDU, CDU) before RMAing nodes one by one (gpu-r03-n01, gpu-r03-n02, gpu-r03-n03, gpu-r03-n04, gpu-r03-n05)

## Cordoned nodes

| Node | Rack | Stage | Reason |
|---|---|---|---|
| `gpu-r03-n01` | r03 | node | fabric_ports: mlx5_4 port 1 State=Down PhysState=Polling: check cable/transceiver and switch port |
| `gpu-r03-n02` | r03 | node | fabric_ports: mlx5_4 port 1 State=Down PhysState=Polling: check cable/transceiver and switch port |
| `gpu-r03-n03` | r03 | node | fabric_ports: mlx5_4 port 1 State=Down PhysState=Polling: check cable/transceiver and switch port |
| `gpu-r03-n04` | r03 | node | fabric_ports: mlx5_4 port 1 State=Down PhysState=Polling: check cable/transceiver and switch port |
| `gpu-r03-n05` | r03 | node | fabric_ports: mlx5_4 port 1 State=Down PhysState=Polling: check cable/transceiver and switch port |

<details><summary>Cordon commands (not executed)</summary>

```bash
scontrol update NodeName=gpu-r03-n01 State=DRAIN Reason="bringup:node:fabric_ports"
kubectl taint node gpu-r03-n01 bringup.gpu/rejected=node:NoSchedule --overwrite
scontrol update NodeName=gpu-r03-n02 State=DRAIN Reason="bringup:node:fabric_ports"
kubectl taint node gpu-r03-n02 bringup.gpu/rejected=node:NoSchedule --overwrite
scontrol update NodeName=gpu-r03-n03 State=DRAIN Reason="bringup:node:fabric_ports"
kubectl taint node gpu-r03-n03 bringup.gpu/rejected=node:NoSchedule --overwrite
scontrol update NodeName=gpu-r03-n04 State=DRAIN Reason="bringup:node:fabric_ports"
kubectl taint node gpu-r03-n04 bringup.gpu/rejected=node:NoSchedule --overwrite
scontrol update NodeName=gpu-r03-n05 State=DRAIN Reason="bringup:node:fabric_ports"
kubectl taint node gpu-r03-n05 bringup.gpu/rejected=node:NoSchedule --overwrite
```

</details>
