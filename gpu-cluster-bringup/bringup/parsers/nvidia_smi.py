from __future__ import annotations

# Parser for `nvidia-smi -q -x` (the XML query dump).
#
# Handles both the current tag names (gpu_power_readings, clocks_event_reasons,
# sram_/dram_ ECC counters on Hopper) and the older ones (power_readings,
# clocks_throttle_reasons, device_memory ECC counters). "N/A" becomes None so a
# gate can tell "not reported by this GPU generation" from "zero".

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from bringup.model import Gpu
from bringup.parsers.common import ENCODING, norm_bdf

_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")
_THROTTLE_REASONS = (
    "hw_slowdown",
    "hw_thermal_slowdown",
    "hw_power_brake_slowdown",
    "sw_thermal_slowdown",
)


def _text(node: ET.Element | None, path: str) -> str | None:
    if node is None:
        return None
    el = node.find(path)
    if el is None or el.text is None:
        return None
    value = el.text.strip()
    return None if value in ("", "N/A", "[N/A]", "Unknown Error") else value


def _num(node: ET.Element | None, path: str) -> float | None:
    raw = _text(node, path)
    if raw is None:
        return None
    m = _NUM.search(raw)
    return float(m.group()) if m else None


def _int(node: ET.Element | None, path: str) -> int | None:
    v = _num(node, path)
    return None if v is None else int(v)


def _flag(node: ET.Element | None, path: str, truthy: tuple[str, ...]) -> bool | None:
    raw = _text(node, path)
    return None if raw is None else raw in truthy


def _sum(node: ET.Element | None, paths: tuple[str, ...]) -> int:
    return sum(_int(node, p) or 0 for p in paths)


_UNC_VOLATILE = (
    "ecc_errors/volatile/dram_uncorrectable",
    "ecc_errors/volatile/sram_uncorrectable_parity",
    "ecc_errors/volatile/sram_uncorrectable_secded",
    "ecc_errors/volatile/sram_uncorrectable",
    "ecc_errors/volatile/double_bit/total",
)
_UNC_AGGREGATE = tuple(p.replace("/volatile/", "/aggregate/") for p in _UNC_VOLATILE)
_COR_VOLATILE = (
    "ecc_errors/volatile/dram_correctable",
    "ecc_errors/volatile/sram_correctable",
    "ecc_errors/volatile/single_bit/total",
)


def _gpu(index: int, g: ET.Element) -> Gpu:
    power = g.find("gpu_power_readings")
    if power is None:
        power = g.find("power_readings")
    reasons = g.find("clocks_event_reasons")
    prefix = "clocks_event_reason_"
    if reasons is None:
        reasons = g.find("clocks_throttle_reasons")
        prefix = "clocks_throttle_reason_"
    active = [r for r in _THROTTLE_REASONS if _text(reasons, prefix + r) == "Active"]
    link = g.find("pci/pci_gpu_link_info")
    return Gpu(
        index=index,
        bus_id=norm_bdf(g.get("id", "")),
        product=_text(g, "product_name") or "unknown",
        vbios=_text(g, "vbios_version"),
        serial=_text(g, "serial"),
        persistence=_flag(g, "persistence_mode", ("Enabled",)),
        pcie_gen_current=_int(link, "pcie_gen/current_link_gen"),
        pcie_gen_max=_int(link, "pcie_gen/max_device_link_gen") or _int(link, "pcie_gen/max_link_gen"),
        pcie_gen_host_max=_int(link, "pcie_gen/max_host_link_gen"),
        pcie_width_current=_int(link, "link_widths/current_link_width"),
        pcie_width_max=_int(link, "link_widths/max_link_width"),
        ecc_enabled=_flag(g, "ecc_mode/current_ecc", ("Enabled",)),
        ecc_pending_enabled=_flag(g, "ecc_mode/pending_ecc", ("Enabled",)),
        ecc_correctable_volatile=_sum(g, _COR_VOLATILE),
        ecc_uncorrectable_volatile=_sum(g, _UNC_VOLATILE),
        ecc_uncorrectable_aggregate=_sum(g, _UNC_AGGREGATE),
        retired_pages_sbe=_int(g, "retired_pages/multiple_single_bit_retirement/retired_count"),
        retired_pages_dbe=_int(g, "retired_pages/double_bit_retirement/retired_count"),
        retirement_pending=_flag(g, "retired_pages/pending_retirement", ("Yes",)),
        remapped_rows_uncorrectable=_int(g, "remapped_rows/remapped_row_unc"),
        remap_pending=_flag(g, "remapped_rows/remapped_row_pending", ("Yes",)),
        remap_failure=_flag(g, "remapped_rows/remapped_row_failure", ("Yes",)),
        power_limit_w=_num(power, "current_power_limit") or _num(power, "power_limit"),
        default_power_limit_w=_num(power, "default_power_limit"),
        max_sm_clock_mhz=_int(g, "max_clocks/sm_clock"),
        max_mem_clock_mhz=_int(g, "max_clocks/mem_clock"),
        temp_c=_num(g, "temperature/gpu_temp"),
        active_throttle=active,
        fabric_state=_text(g, "fabric/state"),
    )


def parse_nvidia_smi_xml(text: str) -> dict:
    """Return {"driver_version", "cuda_version", "attached_gpus", "gpus": [Gpu]}."""
    root = ET.fromstring(text)
    if root.tag != "nvidia_smi_log":
        raise ValueError(f"not an nvidia-smi -q -x document (root <{root.tag}>)")
    gpus = [_gpu(i, g) for i, g in enumerate(root.findall("gpu"))]
    return {
        "driver_version": _text(root, "driver_version"),
        "cuda_version": _text(root, "cuda_version"),
        "attached_gpus": _int(root, "attached_gpus"),
        "gpus": gpus,
    }


def load(path: Path) -> dict:
    with path.open("r", encoding=ENCODING) as fh:
        return parse_nvidia_smi_xml(fh.read())
