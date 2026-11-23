from __future__ import annotations

# Localize a bad node / HCA from a pairwise per-rail test matrix.
#
# Each test is one rail r between two nodes a and b: GPU r on a <-> GPU r on b, traffic on
# HCA r of each side (sendrecv_perf, 1 rank per node). A test result is a *symptom*; the
# question is which endpoint (node, rail) explains the most failing tests.
#
# Greedy cover: take the endpoint with the most unexplained failures; accept it as a suspect
# when it fails in at least MIN_FAILS tests AND in at least MIN_RATIO of the tests it was
# in. Remove the tests it explains and repeat. A healthy partner of a bad endpoint fails in
# exactly one test per bad peer, so its ratio stays low. Failures nobody explains are
# reported as a link between two specific ports (cable, optic or switch port), because with
# a single failing test you cannot tell the two ends apart.

from collections import defaultdict
from dataclasses import dataclass, field

MIN_FAILS = 2
MIN_RATIO = 0.5


@dataclass(frozen=True)
class PairTest:
    a: str
    b: str
    rail: int
    failed: bool
    detail: str = ""


@dataclass
class Suspect:
    node: str
    rail: int | None  # None: every rail on the node is implicated
    failed: int
    tested: int
    partners: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return self.failed / self.tested if self.tested else 0.0


@dataclass
class Localization:
    suspects: list[Suspect]
    unexplained_links: list[PairTest]


def localize(tests: list[PairTest], rails_per_node: int) -> Localization:
    tested: dict[tuple[str, int], int] = defaultdict(int)
    for t in tests:
        tested[(t.a, t.rail)] += 1
        tested[(t.b, t.rail)] += 1
    unexplained = [t for t in tests if t.failed]
    picked: list[Suspect] = []
    while unexplained:
        counts: dict[tuple[str, int], list[PairTest]] = defaultdict(list)
        for t in unexplained:
            counts[(t.a, t.rail)].append(t)
            counts[(t.b, t.rail)].append(t)
        ep, hits = max(counts.items(), key=lambda kv: (len(kv[1]), len(kv[1]) / tested[kv[0]], kv[0]))
        if len(hits) < MIN_FAILS or len(hits) / tested[ep] < MIN_RATIO:
            break
        node, rail = ep
        # Report every failing test the endpoint was in, including ones an earlier suspect
        # already explained (a test between two bad endpoints counts for both).
        all_fails = [t for t in tests if t.failed and t.rail == rail and node in (t.a, t.b)]
        partners = sorted(t.b if t.a == node else t.a for t in all_fails)
        picked.append(Suspect(node=node, rail=rail, failed=len(all_fails), tested=tested[ep], partners=partners))
        unexplained = [t for t in unexplained if t not in hits]

    # Collapse to a node-level suspect when every rail of a node is implicated.
    by_node: dict[str, list[Suspect]] = defaultdict(list)
    for s in picked:
        by_node[s.node].append(s)
    suspects: list[Suspect] = []
    for node, ss in by_node.items():
        if len(ss) == rails_per_node and rails_per_node > 1:
            suspects.append(Suspect(node=node, rail=None, failed=sum(s.failed for s in ss),
                                    tested=sum(s.tested for s in ss),
                                    partners=sorted({p for s in ss for p in s.partners})))
        else:
            suspects.extend(ss)
    suspects.sort(key=lambda s: (s.node, -1 if s.rail is None else s.rail))
    return Localization(suspects=suspects, unexplained_links=unexplained)
