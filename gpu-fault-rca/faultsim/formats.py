"""Writers for the real on-disk formats the simulator emits.

The RCA side has its own parsers (`rca/parsers.py`) and imports nothing from
here, so a writer bug cannot be silently mirrored by a reader bug.

Formats:
  kern.log          journalctl -k -o short-iso-precise
  dcgm.prom         dcgm-exporter Prometheus text (with sample timestamps)
  node_exporter.prom node_exporter PSI + infiniband collectors
  worker.log        PyTorch ProcessGroupNCCL watchdog + NCCL INFO/WARN
  engine.log        vLLM-style serving engine log
  access.log        NGINX combined + rt/urt/upstream
  events.json       kubectl get events -o json
"""
from __future__ import annotations

import calendar
import time

START_EPOCH = calendar.timegm((2026, 11, 14, 10, 0, 0, 0, 0, 0))
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _split(ts: float) -> tuple[time.struct_time, int]:
    sec = int(ts)
    us = int(round((ts - sec) * 1_000_000))
    if us >= 1_000_000:
        sec, us = sec + 1, us - 1_000_000
    return time.gmtime(sec), us


def iso(ts: float) -> str:
    st, us = _split(ts)
    return time.strftime("%Y-%m-%dT%H:%M:%S", st) + f".{us:06d}+00:00"


def iso_z(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(ts)))


def glog(ts: float) -> str:
    st, us = _split(ts)
    return f"{st.tm_mon:02d}{st.tm_mday:02d} {time.strftime('%H:%M:%S', st)}.{us:06d}"


def vllm_ts(ts: float) -> str:
    st, _ = _split(ts)
    return f"{st.tm_mon:02d}-{st.tm_mday:02d} {time.strftime('%H:%M:%S', st)}"


def nginx_ts(ts: float) -> str:
    st, _ = _split(ts)
    return f"{st.tm_mday:02d}/{MONTHS[st.tm_mon - 1]}/{st.tm_year}:{time.strftime('%H:%M:%S', st)} +0000"


def kern(ts: float, node: str, msg: str) -> str:
    return f"{iso(ts)} {node} kernel: {msg}"


def xid(pci: str, code: int, pid: int | None, name: str | None, text: str) -> str:
    who = f"pid={pid}, name={name}" if pid is not None else "pid='<unknown>', name=<unknown>"
    return f"NVRM: Xid (PCI:{pci}): {code}, {who}, {text}"


def prom(metric: str, labels: dict[str, str], value: float, ts: float) -> str:
    lab = ",".join(f'{k}="{v}"' for k, v in labels.items())
    val = f"{value:.0f}" if float(value).is_integer() else f"{value:.3f}"
    return f"{metric}{{{lab}}} {val} {int(ts * 1000)}"


def nccl_watchdog(ts: float, rank: int, seq: int, op: str, timeout_ms: int) -> list[str]:
    head = f"[rank{rank}]:[E{glog(ts)} ProcessGroupNCCL.cpp:632]"
    ran = timeout_ms + 7
    return [
        f"{head} [Rank {rank}] Watchdog caught collective operation timeout: "
        f"WorkNCCL(SeqNum={seq}, OpType={op}, NumelIn=8388608, NumelOut=8388608, "
        f"Timeout(ms)={timeout_ms}) ran for {ran} milliseconds before timing out.",
        f"[rank{rank}]:[E{glog(ts)} ProcessGroupNCCL.cpp:2168] [PG ID 0 PG GUID 0(default_pg) Rank {rank}] "
        f"Timeout at NCCL work: {seq}, last enqueued NCCL work: {seq}, last completed NCCL work: {seq - 1}.",
    ]


def vllm(ts: float, level: str, src: str, msg: str) -> str:
    return f"{level} {vllm_ts(ts)} [{src}] {msg}"


def access(ts: float, status: int, nbytes: int, rt: float, upstream: str, req_id: str) -> str:
    urt = "-" if upstream == "-" else f"{max(rt - 0.002, 0.0):.3f}"
    return (f'10.0.9.17 - - [{nginx_ts(ts)}] "POST /v1/chat/completions HTTP/1.1" {status} {nbytes} '
            f'"-" "openai-python/1.109.1" rt={rt:.3f} urt={urt} ups={upstream} req={req_id}')


def k8s_event(ts: float, kind: str, name: str, namespace: str, reason: str, message: str,
              etype: str = "Normal", component: str = "kubelet", host: str | None = None) -> dict:
    ev = {
        "apiVersion": "v1",
        "kind": "Event",
        "metadata": {"name": f"{name}.{int(ts * 1000):x}", "namespace": namespace},
        "involvedObject": {"kind": kind, "name": name, "namespace": namespace},
        "reason": reason,
        "message": message,
        "type": etype,
        "count": 1,
        "firstTimestamp": iso_z(ts),
        "lastTimestamp": iso_z(ts),
        "source": {"component": component},
    }
    if host:
        ev["source"]["host"] = host
    return ev
