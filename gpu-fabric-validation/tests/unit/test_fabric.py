from __future__ import annotations

from fabval import fabric
from fabval.model import load_thresholds

TH = load_thresholds()
HCAS4 = ("mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3")


def test_ibstat_flags_downtrained_down_and_missing(fixture_text):
    """Rate below NDR, a Polling port and a missing HCA each fail; the frontend bond is ignored."""
    ports = fabric.parse_ibstat(fixture_text("ibdiag/ibstat.ndr_one_downtrained.txt"))
    assert set(ports) == {"mlx5_0", "mlx5_1", "mlx5_2", "mlx5_bond_0"}
    assert ports["mlx5_1"].rate_gbps == 200 and ports["mlx5_bond_0"].link_layer == "Ethernet"
    probs = {(p.hca, p.check) for p in fabric.check_ib_ports(ports, HCAS4, 400, "InfiniBand")}
    assert probs == {("mlx5_1", "port_rate"), ("mlx5_2", "port_state"), ("mlx5_2", "port_phys"),
                     ("mlx5_2", "port_rate"), ("mlx5_3", "port_present")}


def test_ibstatus(fixture_text):
    ports = fabric.parse_ibstatus(fixture_text("ibdiag/ibstatus.txt"))
    assert ports["mlx5_0"].state == "Active" and ports["mlx5_0"].rate_gbps == 400
    assert ports["mlx5_1"].extra["rate_text"] == "200 Gb/sec (2X NDR)"
    probs = fabric.check_ib_ports(ports, ("mlx5_0", "mlx5_1"), 400, "InfiniBand")
    assert [(p.hca, p.check) for p in probs] == [("mlx5_1", "port_rate")]


def test_perfquery_thresholds(fixture_text):
    """Symbol errors over the limit, link downed and rcv errors fail; a clean port does not."""
    good = fabric.parse_perfquery(fixture_text("ibdiag/perfquery.healthy.txt"))
    bad = fabric.parse_perfquery(fixture_text("ibdiag/perfquery.bad_link.txt"))
    assert good["PortXmitData"] == 4294967295 and good["CounterSelect"] == 0
    assert fabric.check_perfquery("mlx5_0", good, TH["perfquery"]) == []
    checks = {p.check for p in fabric.check_perfquery("mlx5_0", bad, TH["perfquery"])}
    assert checks == {"perfquery:SymbolErrorCounter", "perfquery:LinkDownedCounter",
                      "perfquery:LinkErrorRecoveryCounter", "perfquery:PortRcvErrors"}


def test_ethtool_pfc_off_explained(fixture_text):
    """Drops with zero pause frames and zero ECN marks are attributed to lossless config."""
    good = fabric.parse_ethtool_stats(fixture_text("ethtool/ethtool_S.roce_healthy.txt"))
    bad = fabric.parse_ethtool_stats(fixture_text("ethtool/ethtool_S.roce_pfc_off.txt"))
    assert fabric.check_ethtool("mlx5_0", good, TH["ethtool"], 3) == []
    [p] = fabric.check_ethtool("mlx5_0", bad, TH["ethtool"], 3)
    assert "rx_discards_phy=4471209" in p.message and "no PFC pause frames on priority 3" in p.message
    assert "no ECN marks" in p.message


def test_ethtool_phy_errors_are_separate():
    stats = {"rx_symbol_err_phy": 40, "rx_prio3_pause": 10, "tx_prio3_pause": 10, "rx_ecn_mark": 5}
    [p] = fabric.check_ethtool("mlx5_0", stats, TH["ethtool"], 3)
    assert p.check == "ethtool:phy"


def test_show_gids_index(fixture_text):
    """Index 3 is RoCEv2 IPv4 on a clean table; a stray IPv6 address shifts it to 5 on mlx5_1."""
    good = fabric.parse_show_gids(fixture_text("ethtool/show_gids.good.txt"))
    shifted = fabric.parse_show_gids(fixture_text("ethtool/show_gids.shifted.txt"))
    assert len(good) == 8 and good[3].ipv4 == "10.100.0.21" and good[3].version == "v2"
    assert fabric.check_gid_index(good, ("mlx5_0", "mlx5_1"), 3) == []
    [p] = fabric.check_gid_index(shifted, ("mlx5_0", "mlx5_1"), 3)
    assert p.hca == "mlx5_1" and "index 5" in p.message


def test_topo_wrong_numa(fixture_text):
    """GPU2 <-> NIC2 is SYS: flagged. GPU3 <-> NIC3 PIX and GPU0/1 are fine."""
    topo = fabric.parse_topo(fixture_text("topo/nvidia_smi_topo_m.nic2_wrong_numa.txt"))
    assert topo["nic_names"]["NIC2"] == "mlx5_2" and topo["numa"]["GPU2"] == "1"
    probs = fabric.check_gpu_nic_affinity(topo, HCAS4, TH["topo"]["allowed_gpu_nic_paths"])
    assert [(p.hca, p.check) for p in probs] == [("mlx5_2", "gpu_nic_affinity")]


def test_ibdev2netdev(fixture_text):
    m = fabric.parse_ibdev2netdev(fixture_text("ibdiag/ibdev2netdev.txt"))
    assert m["mlx5_1"] == {"port": 1, "netdev": "ens2np0", "state": "Up"} and m["mlx5_2"]["state"] == "Down"
