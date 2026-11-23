from __future__ import annotations

import json
from pathlib import Path

import pytest

from bringup import fleetgen
from bringup.parsers import dcgm, fabric, lspci, nccl, nvidia_smi, rocm_smi, topo
from bringup.parsers.common import norm_bdf

FIX = Path(__file__).resolve().parents[1] / "fixtures"


# ---------------------------------------------------------------- nvidia-smi


def test_hopper_xml_fields_and_idle_gen():
    """Hopper tag names (gpu_power_readings, clocks_event_reasons, sram/dram ECC) parse into the neutral record."""
    node = fleetgen.build_fleet("healthy")[0]
    doc = nvidia_smi.parse_nvidia_smi_xml(fleetgen.render_nvidia_smi_xml(node))
    assert doc["driver_version"] == "550.127.08" and doc["cuda_version"] == "12.4"
    assert doc["attached_gpus"] == 8 and len(doc["gpus"]) == 8
    g = doc["gpus"][0]
    assert g.bus_id == "0000:18:00.0"
    assert g.persistence is True and g.ecc_enabled is True
    assert (g.pcie_width_current, g.pcie_width_max) == (16, 16)
    assert g.pcie_gen_current == 1 and g.pcie_gen_max == 5 and g.pcie_gen_host_max == 5
    assert g.power_limit_w == 700.0 and g.max_sm_clock_mhz == 1980
    assert g.active_throttle == []  # gpu_idle is not a throttle reason we act on
    assert g.fabric_state == "Completed"
    assert g.retired_pages_dbe is None  # N/A on Hopper, not zero
    assert g.remap_pending is False


def test_legacy_xml_tags_ecc_retirement_and_throttle():
    """Pre-Hopper tags (power_readings, clocks_throttle_reasons, single/double_bit, retired pages) still parse."""
    doc = nvidia_smi.load(FIX / "nvidia-smi-q-legacy.xml")
    a, b = doc["gpus"]
    assert a.bus_id == "0000:1a:00.0" and a.product == "Tesla V100-SXM2-32GB"
    assert a.persistence is False
    assert a.ecc_uncorrectable_volatile == 1 and a.ecc_correctable_volatile == 12
    assert a.ecc_uncorrectable_aggregate == 3
    assert a.retired_pages_sbe == 2 and a.retired_pages_dbe == 1 and a.retirement_pending is True
    assert a.active_throttle == ["hw_slowdown"]
    assert a.power_limit_w == 300.0
    assert b.ecc_pending_enabled is False
    assert b.temp_c is None  # N/A stays None
    assert b.retirement_pending is False


def test_not_nvidia_smi_xml_rejected():
    with pytest.raises(ValueError, match="not an nvidia-smi"):
        nvidia_smi.parse_nvidia_smi_xml("<foo/>")


# ---------------------------------------------------------------------- dcgm


def test_dcgm_failure_with_wrapped_warning_and_skips():
    """Continuation rows attach to the test above; wrapped warnings are rejoined; Skip is not Fail."""
    tests = {t.name: t for t in dcgm.load(FIX / "dcgmi-diag-r3-memfail.txt")}
    assert "DCGM Version" not in tests  # metadata is not a test
    mem = tests["GPU Memory"]
    assert mem.result == "Fail" and mem.failed_gpus == [2] and mem.category == "Hardware"
    assert mem.messages == ["GPU 2 Error using CUDA API cudaMemcpy Check DCGM and system logs for errors. "
                            "Reset GPU. Restart DCGM. Rerun diagnostics."]
    assert tests["Targeted Power"].result == "Skip"
    assert tests["Persistence Mode"].result == "Pass"


def test_dcgm_generated_matches_parser():
    node = fleetgen.build_fleet("healthy")[0]
    fleetgen.inject(node, "dcgm_power_fail")
    tests = {t.name: t for t in dcgm.parse_dcgmi_diag(fleetgen.render_dcgmi_diag(node))}
    tp = tests["Targeted Power"]
    assert tp.result == "Fail" and tp.failed_gpus == [5]
    assert "max observed 552 W" in tp.messages[0]
    assert sum(t.result == "Fail" for t in tests.values()) == 1


def test_dcgm_garbage_rejected():
    with pytest.raises(ValueError):
        dcgm.parse_dcgmi_diag("Error: unable to connect to host engine\n")


# ---------------------------------------------------------------------- topo


def test_topo_matrix_nic_legend_and_numa():
    t = topo.load(FIX / "topo-m-a100.txt")
    assert t.gpus == ["GPU0", "GPU1", "GPU2", "GPU3"]
    assert t.nics == ["mlx5_0", "mlx5_1", "mlx5_bond_0"]
    assert t.link("GPU0", "GPU1") == "NV12"
    assert t.link("GPU3", "GPU0") == "SYS"
    assert t.link("GPU2", "mlx5_1") == "PIX"
    assert t.link("GPU3", "mlx5_bond_0") == "NODE"
    assert t.gpu_numa == {"GPU0": 0, "GPU1": 0, "GPU2": 1, "GPU3": 1}


# -------------------------------------------------------------------- fabric


def test_ibstat_mixed_ib_and_ethernet():
    hcas = {h.name: h for h in fabric.load_ibstat(FIX / "ibstat-mixed.txt")}
    assert set(hcas) == {"mlx5_0", "mlx5_1", "mlx5_bond_0"}
    assert hcas["mlx5_0"].firmware == "20.38.1002"
    p0 = hcas["mlx5_0"].ports[0]
    assert (p0.state, p0.physical_state, p0.rate_gbps, p0.link_layer) == ("Active", "LinkUp", 200, "InfiniBand")
    p1 = hcas["mlx5_1"].ports[0]
    assert (p1.state, p1.physical_state) == ("Down", "Disabled")
    assert hcas["mlx5_bond_0"].ports[0].link_layer == "Ethernet"


def test_fi_info_counts_rdm_domains_once():
    hcas = fabric.load_fi_info(FIX / "fi_info-efa.txt")
    assert [h.name for h in hcas] == ["rdmap16s27", "rdmap32s27"]
    assert all(h.kind == "efa" for h in hcas)


# --------------------------------------------------------------------- lspci


def test_lspci_link_numa_and_fallen_off_bus():
    devs = {d.bdf: d for d in lspci.load(FIX / "lspci-vv-excerpt.txt")}
    a = devs["0000:07:00.0"]
    assert a.numa_node == 3 and (a.width_cap, a.width_sta) == (16, 16) and a.speed_sta_gts == 16.0
    dead = devs["0000:0f:00.0"]
    assert "rev ff" in dead.description and dead.width_sta is None
    nic = devs["0000:c4:00.0"]
    assert (nic.width_cap, nic.width_sta, nic.speed_sta_gts) == (16, 8, 8.0)
    assert lspci.is_gpu(a) and not lspci.is_gpu(nic)


@pytest.mark.parametrize("raw,want", [
    ("00000000:1A:00.0", "0000:1a:00.0"),
    ("1a:00.0", "0000:1a:00.0"),
    ("0000:1A:00.0", "0000:1a:00.0"),
])
def test_bdf_normalisation(raw, want):
    """nvidia-smi, lspci and rocm-smi print the same address three ways."""
    assert norm_bdf(raw) == want


# ------------------------------------------------------------------- rocm-smi


def test_rocm_smi_alternate_key_names():
    """Labels drift across ROCm releases; RAS counters are found by pattern."""
    doc = rocm_smi.load(FIX / "rocm-smi-alt-keys.json")
    a, b = doc["gpus"]
    assert doc["driver_version"] == "6.3.6"
    assert a.product == "AMD Instinct MI250X" and a.bus_id == "0000:c1:00.0"
    assert a.ecc_correctable_volatile == 7 and a.ecc_uncorrectable_volatile == 0
    assert b.ecc_uncorrectable_volatile == 1
    assert a.numa_node == 3 and b.temp_c is None


def test_rocm_topo_links_symmetric():
    node = fleetgen.build_fleet("healthy")[-1]
    links = rocm_smi.parse_rocm_topo_json(json.dumps(fleetgen.render_rocm_topo(node)))
    assert links[("GPU0", "GPU7")] == links[("GPU7", "GPU0")] == "XGMI"
    assert len(links) == 8 * 7


# ---------------------------------------------------------------------- nccl


def test_nccl_tests_rows_peak_and_wrong():
    r = nccl.load(FIX / "all_reduce_perf-2node.txt")
    assert len(r["rows"]) == 6
    assert r["peak_busbw_gbps"] == pytest.approx(411.90)
    assert r["avg_busbw_gbps"] == pytest.approx(300.252)
    assert r["wrong"] == 2  # an in-place #wrong counts even when out-of-place was clean


def test_nccl_no_rows_rejected():
    with pytest.raises(ValueError):
        nccl.parse_nccl_tests("# NCCL WARN NET/IB : Got completion with error 12\n")
