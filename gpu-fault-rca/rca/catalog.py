"""Signal catalog: Xid codes, DCGM throttle bits, cause kinds, causal edges, actions.

`docs/SIGNALS.md` is the human version of this file; a test keeps the two in sync.
"""
from __future__ import annotations

# Xid -> (short name, layer, role, cause kind, first action)
# role: "cause" opens a root-cause candidate, "corroborate" only adds evidence,
#       "app" is application-attributable (software unless it spreads to many GPUs after a deploy),
#       "effect" is downstream of something else.
XID = {
    13: ("Graphics engine exception", "serving", "app", "app_fault",
         "Usually the application (bad kernel / OOB access). Check for a recent deploy before touching hardware."),
    31: ("GPU memory page (MMU) fault", "serving", "app", "app_fault",
         "Usually an illegal address in the application. Same triage as 13; RMA only if it follows one GPU."),
    43: ("GPU stopped processing", "serving", "app", "app_fault",
         "Process-level; restart the worker. Escalate if it repeats on the same GPU with different workloads."),
    45: ("Preemptive cleanup", "runtime", "effect", None,
         "Consequence of a killed process; find the preceding Xid."),
    48: ("Double-bit ECC error (DBE)", "hardware", "cause", "ecc_uncorrectable",
         "Drain, reset the GPU (retires the page / remaps the row); RMA if it recurs."),
    61: ("Internal micro-controller breakpoint", "driver", "cause", "gsp_error",
         "Drain and reset the GPU; collect nvidia-bug-report; check driver/firmware pairing."),
    62: ("Internal micro-controller halt", "driver", "cause", "gsp_error",
         "As 61: drain, reset, bug report."),
    63: ("Row remapping event (ECC page retirement pending)", "hardware", "corroborate", "ecc_uncorrectable",
         "Reset the GPU at the next drain to activate the remap."),
    64: ("Row remapping failure", "hardware", "cause", "ecc_uncorrectable",
         "No spare rows left: drain and RMA the GPU."),
    74: ("NVLink error", "hardware", "cause", "nvlink_degraded",
         "Check NVLink CRC/replay counters; reseat or RMA; meanwhile exclude the GPU from its TP group."),
    79: ("GPU has fallen off the bus", "hardware", "cause", "gpu_lost",
         "Cordon + drain the node, reboot; RMA if it recurs (PCIe/riser/power)."),
    92: ("High single-bit ECC error rate", "hardware", "corroborate", "ecc_uncorrectable",
         "Precursor: schedule a GPU reset; watch for DBE / Xid 94/95."),
    94: ("Contained ECC error", "hardware", "cause", "ecc_uncorrectable",
         "Only the affected process died; reset the GPU to apply the remap before resuming work."),
    95: ("Uncontained ECC error", "hardware", "cause", "ecc_uncorrectable",
         "Every process on the GPU is suspect: drain the node, reset the GPU."),
    119: ("GSP RPC timeout", "driver", "cause", "gsp_error",
          "Drain, reset; check driver version vs GSP firmware; consider disabling GSP on affected drivers."),
    120: ("GSP error", "driver", "cause", "gsp_error",
          "As 119."),
}

# DCGM_FI_DEV_CLOCK_THROTTLE_REASONS (a.k.a. CLOCKS_EVENT_REASONS) bits
THROTTLE_BITS = {
    0x1: "GPU idle",
    0x2: "application clocks setting",
    0x4: "SW power cap",
    0x8: "HW slowdown (thermal or power brake)",
    0x10: "sync boost",
    0x20: "SW thermal slowdown",
    0x40: "HW thermal slowdown",
    0x80: "HW power brake slowdown",
}
THERMAL_MASK = 0x8 | 0x20 | 0x40

LAYER_DEPTH = {"hardware": 0, "host": 0, "driver": 1, "runtime": 2, "serving": 3, "orchestration": 4}

CAUSE_LAYER = {
    "node_loss": "hardware", "gpu_lost": "hardware", "ecc_uncorrectable": "hardware",
    "nvlink_degraded": "hardware", "thermal_throttle": "hardware", "network_degraded": "hardware",
    "host_cpu_contention": "host", "driver_mismatch": "driver", "gsp_error": "driver",
    "memory_leak": "runtime", "collective_hang": "runtime", "bad_build": "serving", "app_fault": "serving",
}

BASE_CONFIDENCE = {
    "gpu_lost": 0.80, "node_loss": 0.60, "ecc_uncorrectable": 0.75, "nvlink_degraded": 0.55,
    "thermal_throttle": 0.62, "network_degraded": 0.55, "host_cpu_contention": 0.50, "driver_mismatch": 0.70,
    "gsp_error": 0.60, "memory_leak": 0.55, "bad_build": 0.60, "app_fault": 0.40, "collective_hang": 0.40,
}

# Causal graph: cause kind -> effects/symptoms it can produce downstream.
_COMMON = {"error_burst", "latency_slo"}
CAN_CAUSE = {
    "node_loss": _COMMON | {"nccl_timeout", "engine_crash", "node_not_ready", "pod_evicted", "crashloop"},
    "gpu_lost": _COMMON | {"nccl_timeout", "engine_crash", "crashloop"},
    "ecc_uncorrectable": _COMMON | {"engine_crash", "crashloop", "nccl_timeout"},
    "nvlink_degraded": _COMMON | {"nccl_timeout"},
    "thermal_throttle": {"latency_slo", "error_burst"},
    "network_degraded": _COMMON | {"nccl_timeout", "engine_crash"},
    "host_cpu_contention": {"latency_slo", "error_burst"},
    "driver_mismatch": _COMMON | {"engine_crash", "crashloop", "nccl_timeout"},
    "gsp_error": _COMMON | {"engine_crash", "crashloop", "nccl_timeout"},
    "memory_leak": _COMMON | {"cuda_oom", "engine_crash", "crashloop"},
    "bad_build": _COMMON | {"engine_crash", "crashloop"},
    "app_fault": _COMMON | {"engine_crash"},
    "collective_hang": _COMMON | {"nccl_timeout", "engine_crash"},
}

ACTIONS = {
    "gpu_lost": ("DRAIN_NODE", "Cordon + drain {node}, reboot to recover GPU {gpu}; RMA if Xid 79 recurs."),
    "node_loss": ("DRAIN_NODE", "{node} is unreachable: cordon it, reschedule {replica}, check BMC/PSU logs."),
    "ecc_uncorrectable": ("RESET_GPU", "Drain {node}, reset GPU {gpu} to activate the pending row remap; "
                                       "RMA on Xid 64 or a repeat DBE."),
    "nvlink_degraded": ("RESHARD_TP", "Exclude GPU {gpu} from {replica}'s TP group (re-shard to TP=4), "
                                      "then drain {node} to reseat/RMA."),
    "thermal_throttle": ("DRAIN_NODE", "Drain {node}; facilities ticket for GPU {gpu} airflow/fans."),
    "network_degraded": ("DRAIN_NODE", "Drain {node}; check the IB cable/transceiver (port counters, ibdiagnet)."),
    "host_cpu_contention": ("EVICT_NEIGHBOUR", "Evict the co-scheduled pod from {node}; taint GPU nodes / "
                                               "use the static CPU manager so serving threads keep their cores."),
    "driver_mismatch": ("ROLLBACK_DRIVER", "Pause the driver rollout; roll {node} back (or reboot to load the "
                                           "matching kernel module) before uncordoning."),
    "gsp_error": ("RESET_GPU", "Drain {node}, reset GPU {gpu}, collect nvidia-bug-report."),
    "memory_leak": ("RESTART_WORKER", "Restart the worker on {node} GPU {gpu} before it OOMs (stopgap); "
                                      "bisect the leak in the engine build."),
    "bad_build": ("ROLLBACK_DEPLOY", "Roll {replica} back to the previous engine image. Do NOT RMA: "
                                     "Xid 13/31 on many GPUs right after a deploy is software."),
    "app_fault": ("INVESTIGATE", "Single-GPU Xid 13/31: restart the worker; RMA only if it follows the GPU."),
    "collective_hang": ("INVESTIGATE", "Unexplained collective hang: dump the NCCL flight recorder on the "
                                       "silent rank(s) on {node}."),
}
