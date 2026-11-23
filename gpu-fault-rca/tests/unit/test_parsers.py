"""Parsers against hand-written fixtures in the upstream tools' formats (not simulator output)."""
from __future__ import annotations

from rca import parsers as P
from rca.bundle import norm_pci
from tests.conftest import FIXTURES


def test_kern_xid_new_and_mismatch_formats():
    """Xid lines (quoted '<unknown>' pid and numeric pid), driver API mismatch, soft lockup."""
    evs = P.parse_kern(FIXTURES / "kern.log", "kern.log")
    xids = [e for e in evs if e.kind == "xid"]
    assert [e.attrs["code"] for e in xids] == [79, 94, 63]
    assert xids[0].attrs["pid"] is None and xids[0].attrs["pci"] == "0000:3a:00"
    assert xids[1].attrs["pid"] == 2214 and xids[1].attrs["name"] == "python3"
    assert xids[0].node == "gpu-node-7"
    assert xids[0].ts == P.parse_iso("2026-11-14T10:03:12.481920Z")
    mm = [e for e in evs if e.kind == "driver_api_mismatch"]
    assert len(mm) == 1 and mm[0].attrs["client"] == "560.35.03"
    assert any(e.kind == "cpu_stall" for e in evs)


def test_raw_dmesg_and_legacy_xid_without_pid_name():
    """Raw `dmesg` uptime stamps are anchored to boot time; pre-525 Xid format still parses."""
    boot = P.parse_iso("2026-11-14T09:40:00Z")
    evs = P.parse_kern(FIXTURES / "dmesg.txt", "dmesg.txt", default_node="gpu-node-7", boot_epoch=boot)
    xids = [e for e in evs if e.kind == "xid"]
    assert [e.attrs["code"] for e in xids] == [74, 13]
    assert abs(xids[0].ts - (boot + 1234.567890)) < 1e-6
    assert xids[1].attrs["pid"] is None and xids[1].node == "gpu-node-7"


def test_iso_offsets():
    """UTC offsets are applied, not dropped."""
    assert P.parse_iso("2026-11-14T11:00:00+01:00") == P.parse_iso("2026-11-14T10:00:00Z")
    assert P.parse_iso("2026-11-14T10:00:00,5+0000") == P.parse_iso("2026-11-14T10:00:00Z") + 0.5


def test_prom_with_and_without_timestamps_and_extra_labels():
    """dcgm-exporter text: HELP/TYPE skipped, pod labels tolerated, missing ts falls back to scrape time."""
    ss = P.parse_prom(FIXTURES / "dcgm.prom", "dcgm.prom", scrape_ts=123.0)
    assert [s.metric for s in ss] == ["DCGM_FI_DEV_GPU_TEMP", "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS",
                                      "DCGM_FI_DEV_FB_USED"]
    assert ss[0].ts == 123.0 and ss[2].ts == 1794650400.0
    assert ss[1].value == 64 and ss[0].label("Hostname") == "gpu-node-7"
    assert norm_pci(ss[0].label("pci_bus_id")) == "0000:ba:00"


def test_nccl_watchdog_nanosecond_stamp_and_net_ib_warn():
    """PyTorch watchdog lines (ns-precision glog stamp) and NCCL WARN NET/IB peer extraction."""
    evs = P.parse_worker(FIXTURES / "worker.log", "worker.log", "gpu-node-7", 2026)
    to = [e for e in evs if e.kind == "nccl_timeout"]
    assert len(to) == 1 and to[0].attrs == {"rank": 11, "seq": 48213, "op": "ALLREDUCE", "timeout_ms": 600000}
    assert abs(to[0].ts - (P.parse_iso("2026-11-14T10:04:20Z") + 0.060118123)) < 1e-6
    warn = [e for e in evs if e.kind == "nccl_warn"]
    assert warn[0].attrs["peer_ip"] == "10.0.0.12" and warn[0].attrs["net"]


def test_engine_errors_classified():
    """CUDA OOM names the GPU; illegal access / Error 803 are classified, INFO is ignored."""
    evs = P.parse_engine(FIXTURES / "engine.log", "engine.log", "gpu-node-7", 2026)
    assert [e.kind for e in evs] == ["cuda_oom", "engine_crash", "engine_crash"]
    assert evs[0].attrs["gpu"] == 4
    assert [e.attrs.get("why") for e in evs[1:]] == ["illegal_memory_access", "driver_incompatible"]


def test_nginx_access_log_timezones_and_garbage():
    """Status, rt, upstream; +0100 normalised to UTC; junk lines skipped."""
    rs = P.parse_access(FIXTURES / "access.log")
    assert [r.status for r in rs] == [200, 504, 503]
    assert rs[1].end - rs[0].end == 1.0
    assert rs[2].upstream == "-" and rs[1].rt == 30.0


def test_k8s_events_node_host_and_event_time():
    """Node events map to the node; pod events to source.host; eventTime-only events parse."""
    evs = P.parse_events(FIXTURES / "events.json", "events.json")
    assert [(e.attrs["reason"], e.node) for e in evs] == [
        ("NodeNotReady", "gpu-node-7"), ("BackOff", "gpu-node-7"), ("Scheduled", None)]
    assert evs[2].ts == P.parse_iso("2026-11-14T10:01:00.123456Z")
