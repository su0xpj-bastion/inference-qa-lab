from __future__ import annotations

# AMD path: parser for `rocm-smi ... --json`.
#
# Captured with:
#   rocm-smi --showproductname --showbus --showvbios --showserial --showtemp \
#            --showpower --showmaxpower --showtoponuma --showrasinfo all \
#            --showdriverversion --json
#   rocm-smi --showtopotype --json
#
# rocm-smi --json is a flat {"card0": {"<human label>": "<string>"}, "system": {...}}
# map whose labels drift between ROCm releases, so lookups go through a list of
# known aliases and the RAS counters are found by pattern rather than exact key.
# The output is the same Gpu record the NVIDIA parser makes.

import json
import re
from pathlib import Path

from bringup.model import Gpu
from bringup.parsers.common import ENCODING, norm_bdf

_CARD = re.compile(r"^card(\d+)$")
_TOPO = re.compile(r"Link type between DRM devices (\d+) and (\d+)")
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _get(card: dict, *aliases: str) -> str | None:
    for key in aliases:
        value = card.get(key)
        if value is not None and str(value).strip() not in ("", "N/A"):
            return str(value).strip()
    return None


def _num(card: dict, *aliases: str) -> float | None:
    raw = _get(card, *aliases)
    m = _NUM.search(raw) if raw else None
    return float(m.group()) if m else None


def _ras(card: dict, uncorrectable: bool) -> int:
    total = 0
    for key, value in card.items():
        k = key.lower()
        if "ras" not in k and "ecc" not in k:
            continue
        is_unc = "uncorrect" in k
        if is_unc != uncorrectable or ("correct" not in k):
            continue
        m = _NUM.search(str(value))
        total += int(float(m.group())) if m else 0
    return total


def parse_rocm_smi_json(text: str) -> dict:
    doc = json.loads(text)
    if not isinstance(doc, dict):
        raise ValueError("rocm-smi --json output is not an object")
    cards = sorted(
        ((int(m.group(1)), v) for k, v in doc.items() if (m := _CARD.match(k))),
        key=lambda kv: kv[0],
    )
    if not cards:
        raise ValueError("no cardN entries in rocm-smi JSON")
    gpus: list[Gpu] = []
    for idx, card in cards:
        numa = _num(card, "(Topology) Numa Node", "NUMA Node")
        gpus.append(
            Gpu(
                index=idx,
                bus_id=norm_bdf(_get(card, "PCI Bus") or ""),
                product=_get(card, "Card Series", "Device Name", "Card series") or "unknown",
                vbios=_get(card, "VBIOS version"),
                serial=_get(card, "Serial Number"),
                ecc_enabled=True if any("ras" in k.lower() for k in card) else None,
                ecc_correctable_volatile=_ras(card, uncorrectable=False),
                ecc_uncorrectable_volatile=_ras(card, uncorrectable=True),
                power_limit_w=_num(card, "Max Graphics Package Power (W)"),
                temp_c=_num(card, "Temperature (Sensor junction) (C)", "Temperature (Sensor edge) (C)"),
                numa_node=None if numa is None else int(numa),
            )
        )
    system = doc.get("system", {})
    return {"driver_version": _get(system, "Driver version"), "gpus": gpus}


def parse_rocm_topo_json(text: str) -> dict[tuple[str, str], str]:
    """(GPUi, GPUj) -> link type ("XGMI" / "PCIE") from `rocm-smi --showtopotype --json`."""
    doc = json.loads(text)
    links: dict[tuple[str, str], str] = {}
    for key, value in doc.get("system", {}).items():
        if m := _TOPO.search(key):
            a, b = f"GPU{m.group(1)}", f"GPU{m.group(2)}"
            links[(a, b)] = str(value).strip()
            links[(b, a)] = str(value).strip()
    return links


def load(path: Path) -> dict:
    with path.open("r", encoding=ENCODING) as fh:
        return parse_rocm_smi_json(fh.read())


def load_topo(path: Path) -> dict[tuple[str, str], str]:
    with path.open("r", encoding=ENCODING) as fh:
        return parse_rocm_topo_json(fh.read())
