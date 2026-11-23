"""The scenarios. Each takes a Ctx and returns a list of Checks.

`all` runs every scenario in DEFAULT order; `schema` runs last so it validates the
traffic the others produced. `soak` runs for minutes live, so it is opt-in
(`-w all,soak`).
"""
from __future__ import annotations

from . import (accelerator, agent, benchmark, cluster, compat, conversation, inference, longctx, reasoning,
               release, schema, soak, wire)

WORKFLOWS = {
    "inference": inference.run,
    "reasoning": reasoning.run,
    "accelerator": accelerator.run,
    "cluster": cluster.run,
    "release": release.run,
    "benchmark": benchmark.run,
    "wire": wire.run,
    "agent": agent.run,
    "compat": compat.run,
    "conversation": conversation.run,
    "longctx": longctx.run,
    "soak": soak.run,
    "schema": schema.run,
}
DEFAULT = [w for w in WORKFLOWS if w != "soak"]
