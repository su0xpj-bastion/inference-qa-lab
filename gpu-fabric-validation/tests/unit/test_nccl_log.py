from __future__ import annotations

from fabval import nccl_log


def test_healthy_ib_log(fixture_text):
    """IB transport with GDR on both HCAs, ring/tree channel counts, env and version."""
    hosts = nccl_log.parse(fixture_text("nccl_logs/ib_healthy_2node.log"))
    a1 = hosts["gpu-a1"]
    assert a1.version == "2.23.4+cuda12.6"
    assert a1.networks == {"IB"} and a1.transports == {"IB", "P2P"}
    assert not a1.socket_fallback and a1.gdr == {0: True, 1: True}
    assert len(a1.ring_channels) == 4 and a1.tree_channels == 4
    assert a1.env["NCCL_IB_HCA"] == "=mlx5_0,mlx5_1" and a1.ib_devices == ["mlx5_0:1/IB", "mlx5_1:1/IB"]
    assert a1.ranks == {0, 1} and hosts["gpu-a2"].ranks == {2, 3}


def test_gdr_off_and_socket_fallback(fixture_text):
    """GDR off shows as NET/IB/<n> without /GDRDMA; socket fallback shows as NET/Socket."""
    hosts = nccl_log.parse(fixture_text("nccl_logs/roce_gdr_off_and_socket.log"))
    r1, r2 = hosts["gpu-r1"], hosts["gpu-r2"]
    assert r1.gdr_off_hcas == [1] and not r1.socket_fallback
    assert any(d["state"] == "Disabled" and d["hca"] == 1 for d in r1.gdr_decisions)
    assert r2.socket_fallback and r2.ib_not_found and r2.networks == {"Socket"}
    assert r2.env["NCCL_IB_GID_INDEX"] == "0" and r1.env["NCCL_IB_GID_INDEX"] == "3"
    assert r2.warnings == ["Cuda failure 'out of memory'"]


def test_non_nccl_lines_ignored():
    assert nccl_log.parse("hello\n[rank0] stdout noise\n") == {}
