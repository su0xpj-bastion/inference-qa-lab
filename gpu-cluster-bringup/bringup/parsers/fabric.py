from __future__ import annotations

# Parsers for the scale-out fabric.
#
# - `ibstat` (infiniband-diags): on-prem InfiniBand / RoCE HCAs.
# - `fi_info -p efa` (libfabric): AWS EFA devices, which have no ibstat view.
# Both produce Hca records so the fabric gate is the same for either.

import re
from pathlib import Path

from bringup.model import Hca, Port
from bringup.parsers.common import ENCODING

_CA = re.compile(r"^CA '([^']+)'")
_PORT = re.compile(r"^\s+Port (\d+):")
_KV = re.compile(r"^\s+([A-Za-z ]+):\s*(.*)$")


def parse_ibstat(text: str) -> list[Hca]:
    hcas: list[Hca] = []
    hca: Hca | None = None
    port: dict | None = None

    def close_port() -> None:
        nonlocal port
        if hca is not None and port is not None:
            hca.ports.append(
                Port(
                    number=port["n"],
                    state=port.get("State", "Unknown"),
                    physical_state=port.get("Physical state", "Unknown"),
                    rate_gbps=int(port.get("Rate", "0") or 0),
                    link_layer=port.get("Link layer", "Unknown"),
                )
            )
        port = None

    for line in text.splitlines():
        if m := _CA.match(line):
            close_port()
            hca = Hca(name=m.group(1), kind="ib")
            hcas.append(hca)
        elif m := _PORT.match(line):
            close_port()
            port = {"n": int(m.group(1))}
        elif (m := _KV.match(line)) and hca is not None:
            key, value = m.group(1).strip(), m.group(2).strip()
            if port is not None:
                port[key] = value
            elif key == "Firmware version":
                hca.firmware = value
    close_port()
    if not hcas:
        raise ValueError("no CA blocks found in ibstat output")
    return hcas


_FI_DOMAIN = re.compile(r"^\s*domain:\s*(\S+)-rdm\s*$")


def parse_fi_info_efa(text: str) -> list[Hca]:
    """One Hca per EFA RDM domain. fi_info has no per-port state; presence is the signal."""
    seen: list[str] = []
    for line in text.splitlines():
        if (m := _FI_DOMAIN.match(line)) and m.group(1) not in seen:
            seen.append(m.group(1))
    return [
        Hca(name=d, kind="efa", ports=[Port(1, "Active", "LinkUp", 100, "EFA")]) for d in seen
    ]


def load_ibstat(path: Path) -> list[Hca]:
    with path.open("r", encoding=ENCODING) as fh:
        return parse_ibstat(fh.read())


def load_fi_info(path: Path) -> list[Hca]:
    with path.open("r", encoding=ENCODING) as fh:
        return parse_fi_info_efa(fh.read())
