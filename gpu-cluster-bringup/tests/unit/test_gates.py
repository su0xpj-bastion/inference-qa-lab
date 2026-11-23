from __future__ import annotations

import pytest

from bringup import fleetgen, gates
from bringup.allreduce import NcclTestsRunner
from bringup.collect import collect_node
from bringup.model import Hca, Port


def _facts(tmp_path, defect: str | None = None, idx: int = 0, mutate=None):
    node = fleetgen.build_fleet("healthy")[idx]
    if defect:
        fleetgen.inject(node, defect)
    if mutate:
        mutate(node)
    return collect_node(fleetgen.write_node(node, tmp_path))


def _spec(manifest, facts):
    return manifest["node_types"][facts.node_type], manifest["thresholds"]


def test_idle_gen1_is_not_a_downtrain(tmp_path, manifest):
    """Idle GPUs sit at PCIe gen1 to save power; gating on current gen at idle would cordon every node."""
    f = _facts(tmp_path)
    assert {g.pcie_gen_current for g in f.gpus} == {1}
    assert gates.pcie_link(f, *_spec(manifest, f)).status == "pass"


def test_slot_capped_at_gen4_fails(tmp_path, manifest):
    """A BIOS/riser that caps the slot at gen4 is visible at idle through max_host_link_gen."""
    def cap(n):
        n.gpus[2].host_gen_max = 4
    f = _facts(tmp_path, mutate=cap)
    chk = gates.pcie_link(f, *_spec(manifest, f))
    assert chk.status == "fail" and "slot capped at gen4" in chk.reason


def test_affinity_matching_survives_gpu_renumbering(tmp_path, manifest):
    """With one GPU gone nvidia-smi renumbers the rest; topology-based matching must not cry wolf."""
    f = _facts(tmp_path, "gpu_missing")
    assert len(f.gpus) == 7
    assert gates.gpu_nic_affinity(f, *_spec(manifest, f)).status == "pass"
    assert gates.gpu_inventory(f, *_spec(manifest, f)).status == "fail"


def test_affinity_names_the_misplaced_nic(tmp_path, manifest):
    f = _facts(tmp_path, "numa_affinity")
    chk = gates.gpu_nic_affinity(f, *_spec(manifest, f))
    assert chk.status == "fail"
    assert "GPU6 has no dedicated compute NIC" in chk.reason
    assert "mlx5_6 is not within PIX/PXB of any GPU" in chk.reason


def test_nvlink_failures_collapse_to_the_common_gpu(tmp_path, manifest):
    f = _facts(tmp_path, "nvlink_degraded")
    chk = gates.nvlink_topology(f, *_spec(manifest, f))
    assert chk.reason.startswith("GPU1 has degraded links to all 7 peers")


def test_nvswitch_fabric_not_completed_fails(tmp_path, manifest):
    f = _facts(tmp_path)
    f.gpus[0].fabric_state = "In Progress"
    chk = gates.nvlink_topology(f, *_spec(manifest, f))
    assert chk.status == "fail" and "nvidia-fabricmanager" in chk.reason


def test_correctable_ecc_warns_but_does_not_cordon(tmp_path, manifest):
    f = _facts(tmp_path)
    f.gpus[1].ecc_correctable_volatile = 5000
    chk = gates.gpu_memory_health(f, *_spec(manifest, f))
    assert chk.status == "warn" and "5000 correctable" in chk.reason


def test_remap_failure_and_ecc_disabled_fail(tmp_path, manifest):
    f = _facts(tmp_path)
    f.gpus[0].remap_failure = True
    f.gpus[4].ecc_enabled = False
    chk = gates.gpu_memory_health(f, *_spec(manifest, f))
    assert "out of spare rows, RMA" in chk.reason and "GPU4" in chk.reason and "ECC disabled" in chk.reason


def test_idle_throttle_power_and_persistence(tmp_path, manifest):
    f = _facts(tmp_path)
    f.gpus[0].active_throttle = ["hw_slowdown"]
    f.gpus[1].power_limit_w = 500.0
    f.gpus[2].persistence = False
    reason = gates.gpu_config(f, *_spec(manifest, f)).reason
    assert "throttling at idle: hw_slowdown" in reason
    assert "power limit 500 W, expected 700 W" in reason
    assert "persistence mode off" in reason


def test_firmware_and_kernel_drift_reported_together(tmp_path, manifest):
    def drift(n):
        n.kernel = "5.15.0-126-generic"
        n.nic_fw = "28.40.1000"
        n.vbios = "96.00.74.00.0B"
    f = _facts(tmp_path, mutate=drift)
    reason = gates.software_versions(f, *_spec(manifest, f)).reason
    assert "kernel 5.15.0-126-generic != golden" in reason
    assert "HCA firmware ['28.40.1000']" in reason
    assert "VBIOS ['96.00.74.00.0B']" in reason


def test_ethernet_frontend_nic_not_counted_as_compute(tmp_path, manifest):
    f = _facts(tmp_path)
    assert len(f.hcas) == 9
    chk = gates.fabric_ports(f, *_spec(manifest, f))
    assert chk.status == "pass" and "mlx5_8" not in chk.evidence["compute_hcas"]


def test_ib_rate_mismatch_fails(tmp_path, manifest):
    def slow(n):
        n.nics[0].rate = 200  # NDR port negotiated HDR: wrong cable or transceiver
    f = _facts(tmp_path, mutate=slow)
    chk = gates.fabric_ports(f, *_spec(manifest, f))
    assert "mlx5_0 port 1 rate 200 Gb/s, expected 400" in chk.reason


def test_efa_fabric_counts_devices(manifest, tmp_path):
    """Cloud path: an AWS p5 node type gates on EFA device count through the same gate."""
    f = _facts(tmp_path)
    f.hcas = [Hca(f"rdmap{i}s27", "efa", ports=[Port(1, "Active", "LinkUp", 100, "EFA")]) for i in range(31)]
    spec = dict(manifest["node_types"]["hgx-h100-ndr"])
    spec["fabric"] = {"kind": "efa", "count": 32}
    chk = gates.fabric_ports(f, spec, manifest["thresholds"])
    assert chk.status == "fail" and "31/32 compute EFA devices" in chk.reason


def test_non_critical_xid_ignored(tmp_path, manifest):
    """Xid 13 (graphics engine exception) is usually the application's fault, not the node's."""
    f = _facts(tmp_path)
    f.burnin["xid_lines"] = ["[ 77.1] NVRM: Xid (PCI:0000:18:00): 13, pid=1, Graphics Exception: ESR 0x5"]
    assert gates.xid_events(f, *_spec(manifest, f)).status == "pass"


def test_one_slow_gpu_fails_against_its_siblings(tmp_path, manifest):
    def slow(n):
        n.gpus[3].tflops = 660.0  # above the absolute floor of 650, 8% below siblings
    f = _facts(tmp_path, mutate=slow)
    chk = gates.burnin_perf(f, *_spec(manifest, f))
    assert chk.status == "fail" and chk.reason.startswith("GPU3 660 TFLOPS is >5% below its siblings")


def test_allreduce_wrong_results_fail_even_when_fast(tmp_path, manifest):
    f = _facts(tmp_path)
    f.allreduce[0]["wrong"] = 4
    chk = gates.allreduce_single_node(f, *_spec(manifest, f))
    assert chk.status == "fail" and "data corruption" in chk.reason


def test_amd_ecc_via_rocm_smi(tmp_path, manifest):
    f = _facts(tmp_path, "ecc_uncorrectable", idx=-1)
    assert f.vendor == "amd" and f.runtime_version == "6.2.4"
    chk = gates.gpu_memory_health(f, *_spec(manifest, f))
    assert chk.status == "fail" and "GPU3 (0000:65:00.0) 2 uncorrectable" in chk.reason


def test_amd_pcie_width_comes_from_lspci(tmp_path, manifest):
    f = _facts(tmp_path, "pcie_downtrain", idx=-1)
    assert f.gpus[5].pcie_width_current is None  # rocm-smi doesn't report it
    chk = gates.pcie_link(f, *_spec(manifest, f))
    assert chk.status == "fail" and "GPU5 (0000:a6:00.0) PCIe link x8" in chk.reason


def test_median():
    assert gates.median([3, 1, 2]) == 2 and gates.median([4, 1, 3, 2]) == 2.5
    with pytest.raises(ValueError):
        gates.median([])


def test_nccl_runner_command_and_missing_binary():
    r = NcclTestsRunner(binary="definitely-not-all_reduce_perf", extra_env={"NCCL_IB_HCA": "mlx5"})
    cmd = r.command(["a", "b"], 8)
    assert cmd[:5] == ["mpirun", "-np", "16", "-H", "a:8,b:8"]
    assert "NCCL_IB_HCA=mlx5" in cmd and cmd[cmd.index("-c") + 1] == "1"  # validation on
    with pytest.raises(RuntimeError, match="not on PATH"):
        r.run("pool", ["a", "b"], 8)
