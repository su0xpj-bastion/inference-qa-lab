#!/bin/bash
# Stage 0 of the runbook: capture host-side fabric state on one node into
# <run_dir>/nodes/<hostname>/, in the layout `python -m fabval analyze` reads.
# Run on every node, e.g.  srun --ntasks-per-node=1 scripts/collect_node.sh runs/2026-11-14
# Counters are cleared first (perfquery -R) so thresholds apply to the test window only;
# rerun with COLLECT_ONLY=1 after the tests to read them back.

set -euo pipefail

RUN_DIR="${1:?usage: collect_node.sh <run_dir>}"
OUT="${RUN_DIR}/nodes/$(hostname -s)"
mkdir -p "${OUT}"

ibstat > "${OUT}/ibstat.txt"
nvidia-smi topo -m > "${OUT}/topo.txt"

for ca in $(ibstat -l); do
  [[ "${ca}" == mlx5_bond_* ]] && continue
  layer="$(cat "/sys/class/infiniband/${ca}/ports/1/link_layer")"
  if [[ "${layer}" == "InfiniBand" ]]; then
    lid="$(cat "/sys/class/infiniband/${ca}/ports/1/lid")"
    if [[ -z "${COLLECT_ONLY:-}" ]]; then perfquery -R "$((lid))" 1 >/dev/null; fi
    perfquery "$((lid))" 1 > "${OUT}/perfquery-${ca}.txt"
  fi
done

if command -v ibdev2netdev >/dev/null; then
  ibdev2netdev > "${OUT}/ibdev2netdev.txt"
  while read -r _ca _port _p _arrow netdev _state; do
    ethtool -S "${netdev}" > "${OUT}/ethtool-${netdev}.txt"
  done < "${OUT}/ibdev2netdev.txt"
fi
if command -v show_gids >/dev/null; then
  show_gids > "${OUT}/show_gids.txt"
fi

# Stage 1 for this node: NVLink all_reduce across its 8 GPUs, no network involved.
if [[ -z "${COLLECT_ONLY:-}" ]]; then
  /opt/nccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -c 1 -n 20 > "${OUT}/intra_all_reduce_perf.txt"
fi
