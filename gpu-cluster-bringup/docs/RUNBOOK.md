# GPU cluster bring-up runbook (day 0 → day 2)

The acceptance pipeline is the exit criterion for each phase. A node is "in" when
`python -m bringup run` accepts it; a cluster is "in" when the decision is
`ACCEPTED` or an `ACCEPTED_DEGRADED` that the capacity owner has signed off.

## Day 0: racked, cabled, powered

| # | Step | Done when | Owner |
|---|---|---|---|
| 0.1 | Asset sheet: serials, MACs, rack U, BMC MAC per node; cable map per rack (rail *k* → leaf *k*) | Sheet reviewed against the PO and the floor plan | DC ops |
| 0.2 | Power and cooling: PDU circuits per rack sized for **full** GPU TDP, CDU loops filled and leak-checked (liquid-cooled racks) | PDU readings at idle logged per rack; leak sensors green | Facilities |
| 0.3 | BMC on the OOB network, Redfish reachable, firmware bundle applied (BMC, BIOS, baseboard/VBIOS, NIC) | Redfish inventory serials == asset sheet | DC ops |
| 0.4 | BIOS profile applied (see `provision/onprem/PXE_BCM.md`) | Profile hash identical across the rack | Platform |
| 0.5 | Fabric up: UFM/OpenSM sees every port, no symbol errors on the leaf ports | Port count per leaf == cable map | Network |
| 0.6 | OS installed via PXE/BCM (on-prem) or the pinned AMI via Terraform (cloud) | Node answers SSH / SSM | Platform |

## Day 1: software stack and acceptance

| # | Step | Command | Done when |
|---|---|---|---|
| 1.1 | Golden stack | `ansible-playbook -i provision/onprem/inventory.ini provision/onprem/site.yml` (cloud: baked into the AMI) | Play green |
| 1.2 | **Node stage** (minutes) | `scripts/collect.sh` on each node, or the cloud-init hook | Inventory, ECC/row-remap, PCIe width, versions, IB ports, GPU↔NIC affinity all pass |
| 1.3 | **Burn-in stage** (hours) | `dcgmi diag -r 3` + 4 h GEMM soak (NVIDIA); RVS/AGFHC (AMD) | No critical Xid, no thermal throttle, every GPU within 5% of its siblings |
| 1.4 | **Domain stage** | single-node `all_reduce_perf` over NVLink/XGMI | Full NVLink mesh, busbw above floor, `#wrong == 0` |
| 1.5 | **Cluster stage** | `python -m bringup run --fleet <captures> --manifest manifests/golden.json --out reports/` | Pool capacity met, no performance outliers, multi-node all-reduce above floor |
| 1.6 | Cordon list actioned | the report's `scontrol`/`kubectl taint` block | Every cordoned node has a ticket with the reason text from the report |
| 1.7 | Rack findings actioned **first** | report's "Rack-level findings" | One network/facilities ticket per finding, before any node RMA |

Rules for day 1:
- **A cordon is a ticket, not a deletion.** The node stays powered for diagnosis; the ASG in `provision/aws-p5` uses scale-in protection and EC2-only health checks for exactly this reason.
- **Fix, then re-run the failed stage and every stage after it.** A reseated GPU has to repeat burn-in, not just the inventory check.
- **Missing evidence is a failure.** If `collect.sh` couldn't run a tool, fix the tool; don't hand-edit the capture.

## Day 2: operating the fleet

| Cadence | What | Gate |
|---|---|---|
| Every job start (prolog) | `bringup node --stage node` (seconds) on the allocated nodes | Drain on any failure; ECC/row-remap pending is the common one |
| Weekly / on drain | Burn-in stage on drained nodes before they return | Same thresholds as day 1 |
| On any driver / firmware / kernel change | Update `manifests/golden.json` **and** `provision/onprem/group_vars/all.yml` in one PR (`test_onprem_pins_match_the_golden_manifest` enforces it), roll by rack, re-run the full pipeline per rack | Rack accepted before the next rack starts |
| Monthly | Re-baseline performance floors from the current fleet median (`docs/THRESHOLDS.md`) | Floors move only with a written reason |
| On RMA return | Treat as a new node: full day-1 sequence | Accepted by all four stages |

## Escalation map

| Failing check | First responder | Typical fix |
|---|---|---|
| `gpu_inventory`, `pcie_link` | DC ops | Reseat GPU / riser / cable, then vendor RMA |
| `gpu_memory_health` (remap pending) | Platform | GPU reset (`nvidia-smi -r` or node reboot), re-run acceptance |
| `gpu_memory_health` (remap failure, uncorrectable ECC) | Vendor | RMA |
| `software_versions` | Platform | Re-run `site.yml` on the node; find out why it drifted |
| `fabric_ports`, rack finding on fabric | Network | Optic/cable swap, leaf port check in UFM |
| `gpu_nic_affinity` | DC ops | NIC in the wrong slot or rails cabled out of order |
| `thermal`, `dcgm_diag` power | Facilities / DC ops | Airflow, cold plate, PSU, power cap at the BMC |
| `xid_events` 79/48/94/95 | Vendor | RMA after one confirmation rerun |
| `nvlink_topology` | Vendor | Baseboard / NVSwitch tray |
| `performance_outlier` | Platform | Compare clocks, power and memory clocks with a pool-median node before suspecting silicon |
