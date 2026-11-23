from __future__ import annotations

import pytest

from fabval.model import JobShape, NodeType, busbw_factor, expected_busbw


def test_busbw_factors_match_nccl_tests():
    """busbw normalization matches nccl-tests PERFORMANCE.md for every collective we gate."""
    assert busbw_factor("all_reduce", 16) == pytest.approx(2 * 15 / 16)
    assert busbw_factor("all_gather", 16) == pytest.approx(15 / 16)
    assert busbw_factor("alltoall", 8) == pytest.approx(7 / 8)
    assert busbw_factor("sendrecv", 2) == 1.0
    with pytest.raises(ValueError):
        busbw_factor("all_to_nowhere", 2)


def test_h100_ib_expected_values():
    """Intra-node is NVLink-bound; inter-node ring is min(NVLink, 8 NICs); pair test is one NIC."""
    nt = NodeType.load("h100-sxm-8x-ndr400-ib")
    assert expected_busbw(nt, JobShape("all_reduce", 1, 8)) == pytest.approx(360.0)
    assert expected_busbw(nt, JobShape("all_reduce", 2, 8)) == pytest.approx(360.0)
    assert expected_busbw(nt, JobShape("all_reduce", 2, 8, "tree")) == pytest.approx(8 * 50 * 0.84)
    assert expected_busbw(nt, JobShape("sendrecv", 2, 1)) == pytest.approx(50 * 0.94)
    assert expected_busbw(nt, JobShape("all_reduce", 2, 1)) == pytest.approx(50 * 0.92)


def test_a100_hdr_is_network_bound():
    """With 200G HCAs the inter-node ring falls below NVLink: the network is the bottleneck."""
    nt = NodeType.load("a100-sxm-8x-hdr200-ib")
    intra = expected_busbw(nt, JobShape("all_reduce", 1, 8))
    inter = expected_busbw(nt, JobShape("all_reduce", 4, 8))
    assert inter == pytest.approx(8 * 25 * 0.92) and inter < intra


def test_alltoall_counts_only_remote_share():
    """alltoall busbw scales by (n-1)/(n-g): local peers do not use the NIC."""
    nt = NodeType.load("h100-sxm-8x-ndr400-ib")
    two = expected_busbw(nt, JobShape("alltoall", 2, 8))
    many = expected_busbw(nt, JobShape("alltoall", 64, 8))
    assert two == pytest.approx(50 * 0.80 * 15 / 8)
    assert many < two


def test_bad_inputs_rejected():
    nt = NodeType.load("h100-sxm-8x-ndr400-ib")
    with pytest.raises(ValueError):
        expected_busbw(nt, JobShape("all_reduce", 2, 8, "butterfly"))
    with pytest.raises(ValueError):
        expected_busbw(nt, JobShape("all_reduce", 2, 16))
    with pytest.raises(KeyError):
        NodeType.load("b200-imaginary")
