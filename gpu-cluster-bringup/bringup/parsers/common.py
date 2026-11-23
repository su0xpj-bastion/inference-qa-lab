from __future__ import annotations

ENCODING = "utf-8"


def norm_bdf(raw: str) -> str:
    """Normalise a PCI address to dddd:bb:dd.f lower case.

    nvidia-smi prints an 8-digit domain (00000000:18:00.0); lspci prints either
    18:00.0 or 0000:18:00.0. They must compare equal.
    """
    raw = raw.strip().lower()
    if not raw:
        return raw
    parts = raw.split(":")
    if len(parts) == 2:
        parts = ["0000", *parts]
    domain = parts[0][-4:].rjust(4, "0")
    return f"{domain}:{parts[1]}:{parts[2]}"
