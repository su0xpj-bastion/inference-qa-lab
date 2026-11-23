from __future__ import annotations

from itertools import combinations

from fabval.localize import PairTest, localize

NODES = [f"n{i}" for i in range(6)]


def matrix(bad_endpoints=(), bad_links=(), rails=4):
    tests = []
    for a, b in combinations(NODES, 2):
        for r in range(rails):
            failed = (a, r) in bad_endpoints or (b, r) in bad_endpoints or (a, b, r) in bad_links
            tests.append(PairTest(a, b, r, failed))
    return tests


def test_clean_matrix_has_no_suspects():
    loc = localize(matrix(), 4)
    assert loc.suspects == [] and loc.unexplained_links == []


def test_single_bad_port():
    """Partners fail once each (ratio 1/5) and are not blamed; the port fails 5/5."""
    loc = localize(matrix({("n2", 1)}), 4)
    [s] = loc.suspects
    assert (s.node, s.rail, s.failed, s.tested) == ("n2", 1, 5, 5)


def test_whole_node_collapses_to_node_level():
    loc = localize(matrix({("n4", r) for r in range(4)}), 4)
    [s] = loc.suspects
    assert s.node == "n4" and s.rail is None and s.failed == 20


def test_two_bad_ports_same_rail():
    """The test between two bad ports counts for both."""
    loc = localize(matrix({("n0", 2), ("n5", 2)}), 4)
    assert [(s.node, s.rail, s.failed) for s in loc.suspects] == [("n0", 2, 5), ("n5", 2, 5)]


def test_single_link_is_reported_as_link_not_node():
    """One failing test cannot tell its two ends apart: it is a cable/switch-port suspect."""
    loc = localize(matrix(bad_links={("n1", "n3", 0)}), 4)
    assert loc.suspects == [] and [(t.a, t.b, t.rail) for t in loc.unexplained_links] == [("n1", "n3", 0)]
