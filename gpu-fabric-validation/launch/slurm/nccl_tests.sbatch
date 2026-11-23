#!/bin/bash
# nccl-tests across N nodes x 8 GPUs under Slurm, one MPI rank per GPU, launched by srun with
# PMIx and run inside the nccl-tests container through pyxis/enroot.
#
#   sbatch --nodes=8 launch/slurm/nccl_tests.sbatch                  # all_reduce, all nodes
#   TEST=alltoall_perf sbatch --nodes=2 launch/slurm/nccl_tests.sbatch
#
#SBATCH --job-name=nccl-fabric
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# Topology: wait up to 15 min for an allocation under one leaf switch (topology/tree plugin)
# before accepting a placement that crosses the spine.
#SBATCH --switches=1@00:15:00

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
FABRIC="${FABRIC:-roce}"                       # roce | ib  -> launch/nccl-<fabric>.env
IMAGE="${IMAGE:-registry.example.com#fabric/nccl-tests:2.13.11-cuda12.6}"   # enroot: '#' separates registry
TEST="${TEST:-all_reduce_perf}"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/runs/${SLURM_JOB_ID}}"

# Shared NCCL settings: the same file the k8s ConfigMap and the Ray runtime_env are built from.
set -a
# shellcheck source=/dev/null
source "${REPO_DIR}/launch/nccl-${FABRIC}.env"
set +a
# Bootstrap/OOB interface: the host management bond, never a RoCE rail and never lo/docker0.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME_HOST:-bond0}"

mkdir -p "${RUN_DIR}/collective"

# pyxis passes the job environment into the container; enroot's 99-mellanox hook mounts
# /dev/infiniband and the verbs libraries when MELLANOX_VISIBLE_DEVICES is set by the site config.
srun --mpi=pmix \
     --container-image="${IMAGE}" \
     --container-mounts="${RUN_DIR}:${RUN_DIR}" \
     --no-container-mount-home \
     --cpu-bind=none \
     --kill-on-bad-exit=1 \
     "/opt/nccl-tests/build/${TEST}" -b 8 -e 8G -f 2 -g 1 -c 1 -n 20 \
     > "${RUN_DIR}/collective/${TEST}.txt" \
     2> "${RUN_DIR}/collective/nccl_debug.log"

echo "results in ${RUN_DIR}/collective; analyze with: python -m fabval analyze ${RUN_DIR}"
