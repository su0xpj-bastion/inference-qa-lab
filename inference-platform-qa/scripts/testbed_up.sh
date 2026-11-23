#!/usr/bin/env bash
# Start 3 mockserve replicas + the NGINX-semantics proxy on :8080. Ctrl-C stops all.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
pids=()
trap 'kill "${pids[@]}" 2>/dev/null' EXIT INT TERM
for i in 0 1 2; do
  REPLICA_ID=pod-$i MOCKSERVE_ALLOW_FAULTS=1 $PY -m mockserve --host 127.0.0.1 --port $((8100 + i)) & pids+=($!)
done
$PY -m testbed.lb --port "${LB_PORT:-8080}" 127.0.0.1:8100 127.0.0.1:8101 127.0.0.1:8102 & pids+=($!)
wait
