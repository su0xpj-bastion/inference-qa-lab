# RCA playbook: multi-node GPU inference

The engine in `rca/` follows these steps, and an on-call engineer can follow them by hand with the same bundle.

## 1. Start from user impact, not from the noisiest log

Open the router log first. Which replica, from when, and is it latency (p99 > SLO for 2+ windows) or errors
(≥ 3 5xx and > 1 % in a 10 s window)? Everything after this explains these windows. A signal that explains no
impact is a **precursor**: file it, but don't page on it.

## 2. Build one timeline across layers

Normalise every source to UTC: kern.log ISO stamps, DCGM sample ms, PyTorch glog `MMDD`, vLLM `MM-DD`, NGINX
`[dd/Mon/yyyy:… +zzzz]`, k8s `lastTimestamp`/`eventTime`. Expect skew: metrics lag by one scrape (15 s) and
k8s lags reality by design (NodeNotReady fires ~40 s after a node dies). The engine allows 30 s of skew.

## 3. Separate effects from causes

| Never a root cause on its own | Walk it back to |
|---|---|
| NCCL watchdog timeout | The rank(s) that did **not** report it, then the lowest-layer evidence on their node |
| CUDA OOM | The FB-used trend on that GPU before the OOM (leak), or a load step |
| Engine crash / CrashLoopBackOff | The first error line of the first crash |
| NodeNotReady / eviction | When telemetry from that node actually stopped |
| 5xx on the *healthy* replica | Failover overload from the replica that lost capacity |

## 4. Go to the deepest layer that has evidence

hardware/host → driver/firmware → runtime (NCCL, CUDA, allocator) → serving (engine, build) → orchestration.
Ties go to the stronger and deeper candidate. Counter-example: Xid 13/31 is the GPU *reporting* an application
error. If it shows up on many GPUs across nodes right after an image roll, the root cause is the build, at the
serving layer. Six RMAs would be wrong.

## 5. Corroborate from independent sources

Each extra independent source (kern, dcgm, node_exporter, nccl, engine, k8s) adds confidence. One source alone
(a thermal-only DCGM signal, for instance) can still clear the bar when it explains the impact. A candidate that
explains impact but stays below threshold is reported as **low confidence**, not dropped.

## 6. Act, then prove recovery

| Root cause | Action | Recovery check |
|---|---|---|
| gpu_lost, node_loss, thermal, IB link | DRAIN_NODE | No 5xx for 60 s after the drain |
| ecc_uncorrectable | RESET_GPU (activates remap) | Replica back, no recurrence |
| nvlink_degraded | RESHARD_TP, then drain to repair | Needs N-1 headroom at current load |
| driver_mismatch | ROLLBACK_DRIVER, pause the rollout | NVML/DCGM back on the node |
| memory_leak | RESTART_WORKER before OOM, then bisect | No OOM in the window |
| host_cpu_contention | EVICT_NEIGHBOUR, pin CPUs | p99 back under SLO |
| bad_build | ROLLBACK_DEPLOY | No Xid 13/31 after the rollback |

"Recovered" means **zero** 5xx for 60 s, not "no symptom window". Passive ejection in the load balancer hides a dead
replica down to about one failed probe request every 10 s. That is below any alerting threshold, but it is not mitigated.
`test_acting_on_the_red_herring_does_not_recover` covers exactly this case.

## 7. Write it up

`rca analyze <bundle> --out DIR` writes `incident.md` and `incident.json`: root cause, layer, confidence and
sources, the evidence lines with `file:line`, what it explains downstream, blast radius, and the action.
