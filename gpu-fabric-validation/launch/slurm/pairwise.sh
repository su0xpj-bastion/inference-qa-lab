#!/bin/bash
# Stage 2 of the runbook: per-rail sendrecv between every node pair of an allocation.
# One rank per node, GPU r talking to GPU r on the peer through HCA r only, so a result
# implicates exactly two ports. Writes pairs/<a>__<b>__rail<r>.sendrecv_perf.txt, the layout
# `python -m fabval analyze` reads.
#
#   salloc --nodes=8 --gpus-per-node=8 --exclusive launch/slurm/pairwise.sh runs/$(date +%F)
#
# 8 nodes = 28 pairs x 8 rails = 224 runs of ~20 s; pairs on disjoint nodes run concurrently.

set -euo pipefail

RUN_DIR="${1:?usage: pairwise.sh <run_dir>}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
FABRIC="${FABRIC:-roce}"
IMAGE="${IMAGE:-registry.example.com#fabric/nccl-tests:2.13.11-cuda12.6}"
RAILS="${RAILS:-8}"

set -a
# shellcheck source=/dev/null
source "${REPO_DIR}/launch/nccl-${FABRIC}.env"
set +a
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME_HOST:-bond0}"
export NCCL_DEBUG=WARN    # per-pair logs are noise; the full collective keeps INFO

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?run inside an allocation}")
mkdir -p "${RUN_DIR}/pairs"

for ((i = 0; i < ${#NODES[@]}; i++)); do
  for ((j = i + 1; j < ${#NODES[@]}; j++)); do
    a="${NODES[$i]}"; b="${NODES[$j]}"
    for ((r = 0; r < RAILS; r++)); do
      out="${RUN_DIR}/pairs/${a}__${b}__rail${r}.sendrecv_perf.txt"
      CUDA_VISIBLE_DEVICES="${r}" NCCL_IB_HCA="=mlx5_${r}" \
        srun --mpi=pmix --nodes=2 --ntasks-per-node=1 --gpus-per-node=8 --nodelist="${a},${b}" \
             --container-image="${IMAGE}" --no-container-mount-home --cpu-bind=none \
             /opt/nccl-tests/build/sendrecv_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 20 > "${out}" \
        || echo "pair ${a} ${b} rail ${r} failed (rc=$?), kept partial output" >&2
    done
  done
done
