#!/bin/bash
# Submit the NCCL all_reduce job to a running RayCluster (raycluster.yaml).
#   RAY_ADDRESS=http://<head-svc>:8265 launch/ray/submit.sh 8
# The runtime_env is built from the same env file Slurm sources and the k8s ConfigMap mirrors.

set -euo pipefail

NODES="${1:-8}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FABRIC="${FABRIC:-roce}"
RUNTIME_ENV="$(python3 "${HERE}/nccl_allreduce_job.py" --nodes "${NODES}" --print-runtime-env "${HERE}/../nccl-${FABRIC}.env")"

ray job submit --address "${RAY_ADDRESS:?set RAY_ADDRESS}" \
  --working-dir "${HERE}" \
  --runtime-env-json "${RUNTIME_ENV}" \
  -- python nccl_allreduce_job.py --nodes "${NODES}" --gpus-per-node 8
