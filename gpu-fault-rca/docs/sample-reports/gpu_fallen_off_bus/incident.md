# Incident report `inc-0001`

Window: 2026-11-14T10:00:00Z + 600s · verdict **INCIDENT** · confidence threshold 0.7

## Root cause 1: `gpu_lost` on gpu-node-1 GPU 3 (rank 11 of replica-a)

| Field | Value |
|---|---|
| Layer | hardware |
| Confidence | 0.99 (sources: dcgm, engine, kern, nccl) |
| Onset / detected | T+200s / T+200s |
| Blast radius | replica-a: 16 ranks on gpu-node-0, gpu-node-1; 250 failed, 0 slow requests |
| Action | **DRAIN_NODE**: Cordon + drain gpu-node-1, reboot to recover GPU 3; RMA if Xid 79 recurs. |

Evidence:

- T+200s `nodes/gpu-node-1/kern.log:2` (kern): Xid 79 (GPU has fallen off the bus) on GPU 3
  `2026-11-14T10:03:20.100000+00:00 gpu-node-1 kernel: NVRM: Xid (PCI:0000:5d:00): 79, pid='<unknown>', name=<unknown>, GPU has fallen off the bus.`
- T+210s `nodes/gpu-node-1/dcgm.prom:1287` (dcgm): GPU 3 disappears from DCGM while the node's other GPUs keep reporting
  `DCGM_FI_DEV_GPU_TEMP{gpu="3",UUID="GPU-19d3534b-ff57-61ee-44f6-f90e79048095",pci_bus_id="00000000:5D:00.0",device="nvidia3",modelName="NVIDIA H100 80GB HBM3",Hostname="gpu-node-1",DCGM_FI_DRIVER_VERSION="550.90.07"} 51 1`
- T+260s `nodes/gpu-node-1/worker.log:17` (nccl): 1 rank(s) never reported the timeout (gpu-node-1/GPU3): the hang is on them
  `[rank14]:[E1114 10:04:20.012987 ProcessGroupNCCL.cpp:632] [Rank 14] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=48001, OpType=ALLREDUCE, NumelIn=8388608, NumelOut=8388608, Timeout(ms)=60000) ran for 600`
- T+350s `nodes/gpu-node-1/engine.log:3` (engine): engine error consistent with gpu_lost
  `ERROR 11-14 10:05:50 [worker_base.py:612] RuntimeError: Expected 8 visible CUDA devices for tensor_parallel_size=8, found 7`
- T+410s `nodes/gpu-node-1/engine.log:4` (engine): engine error consistent with gpu_lost
  `ERROR 11-14 10:06:50 [worker_base.py:612] RuntimeError: Expected 8 visible CUDA devices for tensor_parallel_size=8, found 7`
- T+530s `nodes/gpu-node-1/engine.log:5` (engine): engine error consistent with gpu_lost
  `ERROR 11-14 10:08:50 [worker_base.py:612] RuntimeError: Expected 8 visible CUDA devices for tensor_parallel_size=8, found 7`

Explains downstream (not blamed):

- 6 × `crashloop` on gpu-node-0, gpu-node-1
- 5 × `engine_crash` on gpu-node-0, gpu-node-1
- 15 × `nccl_timeout` on gpu-node-0, gpu-node-1

User-visible symptoms:

- `error_burst` on replica-a T+230s–T+260s: p99 30.0s, 176 errors
- `latency_slo` on replica-a T+230s–T+270s: p99 30.0s, 178 errors
- `error_burst` on replica-a T+300s–T+310s: p99 0.0s, 4 errors
- `latency_slo` on replica-b T+340s–T+370s: p99 21.92s, 0 errors (failover from gpu_lost@replica-a)
- `error_burst` on replica-a T+370s–T+400s: p99 0.0s, 6 errors
- `error_burst` on replica-a T+420s–T+460s: p99 0.0s, 10 errors
- `error_burst` on replica-a T+480s–T+500s: p99 0.0s, 8 errors
- `error_burst` on replica-a T+540s–T+550s: p99 0.0s, 3 errors
- `error_burst` on replica-a T+580s–T+600s: p99 0.0s, 6 errors

## Timeline (first 30 entries)

| t | kind | what | where |
|---|---|---|---|
| T+200s | kern | gpu_lost: Xid 79 (GPU has fallen off the bus) on GPU 3 | `nodes/gpu-node-1/kern.log:2` |
| T+210s | dcgm | gpu_lost: GPU 3 disappears from DCGM while the node's other GPUs keep reporting | `nodes/gpu-node-1/dcgm.prom:1287` |
| T+230s | symptom | error_burst on replica-a (p99 30.0s, 176 errors) | `router/access.log` |
| T+230s | symptom | latency_slo on replica-a (p99 30.0s, 178 errors) | `router/access.log` |
| T+260s | effect | engine_crash on gpu-node-0 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-0/engine.log:22` |
| T+260s | effect | engine_crash on gpu-node-1 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-1/engine.log:2` |
| T+260s | nccl | gpu_lost: 1 rank(s) never reported the timeout (gpu-node-1/GPU3): the hang is on them | `nodes/gpu-node-1/worker.log:17` |
| T+260s | effect | nccl_timeout on gpu-node-1 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-1/worker.log:17` |
| T+260s | effect | nccl_timeout on gpu-node-1 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-1/worker.log:19` |
| T+260s | effect | nccl_timeout on gpu-node-0 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-0/worker.log:17` |
| T+260s | effect | nccl_timeout on gpu-node-0 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-0/worker.log:19` |
| T+260s | effect | nccl_timeout on gpu-node-1 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-1/worker.log:21` |
| T+260s | effect | nccl_timeout on gpu-node-1 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-1/worker.log:23` |
| T+260s | effect | nccl_timeout on gpu-node-1 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-1/worker.log:25` |
| T+260s | effect | nccl_timeout on gpu-node-0 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-0/worker.log:21` |
| T+260s | effect | nccl_timeout on gpu-node-0 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-0/worker.log:23` |
| T+260s | effect | nccl_timeout on gpu-node-1 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-1/worker.log:27` |
| T+260s | effect | nccl_timeout on gpu-node-1 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-1/worker.log:29` |
| T+260s | effect | nccl_timeout on gpu-node-0 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-0/worker.log:25` |
| T+260s | effect | nccl_timeout on gpu-node-0 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-0/worker.log:27` |
| T+260s | effect | nccl_timeout on gpu-node-0 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-0/worker.log:29` |
| T+260s | effect | nccl_timeout on gpu-node-0 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-0/worker.log:31` |
| T+300s | symptom | error_burst on replica-a (p99 0.0s, 4 errors) | `router/access.log` |
| T+340s | symptom | latency_slo on replica-b (p99 21.9s, 0 errors) | `router/access.log` |
| T+350s | engine | gpu_lost: engine error consistent with gpu_lost | `nodes/gpu-node-1/engine.log:3` |
| T+350s | effect | engine_crash on gpu-node-1 <- gpu_lost@gpu-node-1 | `nodes/gpu-node-1/engine.log:3` |
| T+351s | effect | crashloop on gpu-node-0 <- gpu_lost@gpu-node-1 | `k8s/events.json:6` |
| T+351s | effect | crashloop on gpu-node-1 <- gpu_lost@gpu-node-1 | `k8s/events.json:7` |
| T+370s | symptom | error_burst on replica-a (p99 0.0s, 6 errors) | `router/access.log` |
| T+410s | engine | gpu_lost: engine error consistent with gpu_lost | `nodes/gpu-node-1/engine.log:4` |
