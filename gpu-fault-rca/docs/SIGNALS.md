# GPU fault signal cheat-sheet

What each signal means, which layer it implicates, and the first action. `rca/catalog.py` is the machine version
of this page; `test_signal_cheatsheet_covers_catalog` fails if the two drift.

## NVIDIA Xid (kernel log, `NVRM: Xid (PCI:…): <code>, …`)

| Xid | Meaning | Layer | RCA role | First action |
|---|---|---|---|---|
| 13 | Graphics engine exception | serving (usually app) | app fault | Check for a recent deploy before touching hardware. Many GPUs at once after an image roll = bad build. |
| 31 | GPU memory page (MMU) fault | serving (usually app) | app fault | Illegal address in the app. RMA only if it follows one GPU across workloads. |
| 43 | GPU stopped processing | serving | app fault | Restart the worker; escalate if it repeats on the same GPU. |
| 45 | Preemptive cleanup | runtime | effect | Consequence of a killed process; find the preceding Xid. |
| 48 | Double-bit ECC error (DBE) | hardware | cause | Drain, reset the GPU; RMA on recurrence. |
| 61 | Internal micro-controller breakpoint | driver/firmware | cause | Drain, reset, nvidia-bug-report; check driver/firmware pairing. |
| 62 | Internal micro-controller halt | driver/firmware | cause | As 61. |
| 63 | Row remapping event | hardware | corroborates ECC | Reset the GPU at the next drain to activate the remap. |
| 64 | Row remapping failure | hardware | cause | No spare rows: drain and RMA. |
| 74 | NVLink error | hardware | cause | Read CRC/replay counters; exclude the GPU from its TP group; reseat/RMA. |
| 79 | GPU has fallen off the bus | hardware | cause | Cordon + drain, reboot; RMA if it recurs (PCIe, riser, power). |
| 92 | High single-bit ECC rate | hardware | corroborates ECC | Precursor: schedule a reset, watch for DBE / 94 / 95. |
| 94 | Contained ECC error | hardware | cause | Only the touching process died; reset the GPU before resuming. |
| 95 | Uncontained ECC error | hardware | cause | Everything on the GPU is suspect: drain the node, reset. |
| 119 | GSP RPC timeout | driver/firmware | cause | Drain, reset; check driver vs GSP firmware. |
| 120 | GSP error | driver/firmware | cause | As 119. |

A single Xid 13/31 from a non-serving pid (profilers, debug jobs) with no user impact is a **precursor**, not an incident.

## DCGM (dcgm-exporter)

| Field | Healthy | Fault signature | First action |
|---|---|---|---|
| `DCGM_FI_DEV_FB_USED` | Flat: the engine preallocates its KV pool | Linear growth (slope > 2 MiB/s, R² > 0.9, > 1 GiB) = leak outside the pool | Restart the worker before OOM; bisect the build |
| `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` | 0 | Any increment | Drain + reset GPU |
| `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL` | Occasional increments | Normal background; only a sustained rate matters (Xid 92) | None |
| `DCGM_FI_DEV_ROW_REMAP_PENDING` | 0 | 1 = needs a reset to activate | Reset at next drain |
| `DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL` | Near-zero rate | > 5/s; rises with link utilisation | Exclude GPU from TP, reseat/RMA |
| `DCGM_FI_DEV_GPU_TEMP` | < 85 C under load | ≥ 88 C with thermal throttle bits | Drain; facilities ticket |
| `DCGM_FI_DEV_GPU_UTIL` | Tracks load | Low while latency is high = host-bound (CPU starved) | Look at the host, not the GPU |
| `DCGM_FI_DEV_XID_ERRORS` | 0 | Last Xid code | Cross-check kern.log |
| Series disappears | — | One GPU missing = GPU lost; whole node missing but node_exporter alive = NVML can't init (driver) | See Xid 79 / driver mismatch |

`DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` bits:

| Bit | Meaning | Fault? |
|---|---|---|
| 0x1 | GPU idle | No |
| 0x2 | Application clocks setting | No |
| 0x4 | SW power cap | Capacity planning, not a fault |
| 0x8 | HW slowdown (thermal or power brake) | Yes |
| 0x10 | Sync boost | No |
| 0x20 | SW thermal slowdown | Yes |
| 0x40 | HW thermal slowdown | Yes |
| 0x80 | HW power brake slowdown | Yes: PSU / power delivery |

## Other layers

| Signal | Source | Meaning | First action |
|---|---|---|---|
| `NVRM: API mismatch: the client has the version X, but this kernel module has the version Y` | kern.log | Userspace driver upgraded, kernel module not reloaded | Pause the rollout; roll back or reboot the node |
| `Error 803: system has unsupported display driver / cuda driver combination` | engine log | Same, seen from CUDA | As above |
| `Watchdog caught collective operation timeout: WorkNCCL(SeqNum=…)` | worker (PyTorch) | This rank waited for a collective. **It is usually the healthy one.** | Find the ranks that did *not* report (silent rank = suspect) |
| `NCCL WARN NET/IB : Got completion from peer <ip> with status=12` | worker (NCCL) | Transport retry exceeded towards `<ip>` | Check IB counters on the named peer |
| `node_infiniband_symbol_error_total` rising | node_exporter | Physical-layer errors on that port | Drain; check cable/transceiver |
| `node_pressure_cpu_waiting_seconds_total` rate > 0.5 | node_exporter | CPU contention on the host | Find co-scheduled pods; pin CPUs |
| `torch.OutOfMemoryError: CUDA out of memory … GPU N` | engine log | Effect. Ask why memory grew | Look at the FB trend before the OOM |
| `NodeNotReady` | k8s events | Kubelet stopped heartbeating, ~40 s after the fact | Use as corroboration, not detection |
| `BackOff` | k8s events | Container crash-looping | Effect; find the first error |
