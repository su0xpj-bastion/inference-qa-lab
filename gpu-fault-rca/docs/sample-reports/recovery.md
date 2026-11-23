## Recovery-path validation

| Scenario | Action(s) | TTD vs impact | TTM | Requests lost | Verdict |
|---|---|---|---|---|---|
| `node_power_loss` | DRAIN_NODE@T+220 | -20.0s | 20.0s | 110 | pass |
| `gpu_fallen_off_bus` | DRAIN_NODE@T+210 | -29.9s | 10.0s | 53 | pass |
| `ecc_uncorrectable` | RESET_GPU@T+210 | -10.0s | 160.0s | 2 | pass |
| `nvlink_degraded` | RESHARD_TP@T+352 | 2.3s | not mitigated | 72 | fail |
| `ib_link_degraded` | DRAIN_NODE@T+235 | 25.0s | 60.0s | 1 | pass |
| `driver_partial_upgrade` | ROLLBACK_DRIVER@T+210 | -30.0s | 190.0s | 54 | pass |
| `thermal_throttle` | DRAIN_NODE@T+265 | 15.0s | 30.0s | 0 | pass |
| `worker_memory_leak` | RESTART_WORKER@T+190 | -180.0s | 0.0s | 0 | pass |
| `noisy_neighbour_cpu` | EVICT_NEIGHBOUR@T+210 | 0.0s | 60.0s | 1 | pass |
| `bad_engine_build` | ROLLBACK_DEPLOY@T+210 | -10.0s | 60.0s | 2 | pass |
| `combo_xid79_noisy_neighbour` | EVICT_NEIGHBOUR@T+210, DRAIN_NODE@T+250 | 0.0s | 80.0s | 156 | pass |
| `combo_leak_thermal` | RESTART_WORKER@T+190, DRAIN_NODE@T+265 | -60.0s | 130.0s | 318 | pass |
