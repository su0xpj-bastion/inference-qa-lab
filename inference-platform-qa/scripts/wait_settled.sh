#!/usr/bin/env bash
# Wait until the inference Deployment is settled: rollout complete AND no pods still Terminating.
# `rollout status` returns once new pods are Ready, but old pods keep serving through
# preStop + drain grace (~15s, by design). Anything that measures spread or per-pod
# behaviour must wait for those to exit, or it measures the drain instead.
set -euo pipefail
NS=${NS:-inference}
kubectl -n "$NS" rollout status deploy/inference --timeout=180s
terminating=$(kubectl -n "$NS" get pods -l app=inference \
  -o jsonpath='{range .items[?(@.metadata.deletionTimestamp)]}{.metadata.name}{"\n"}{end}')
if [ -n "$terminating" ]; then
  echo "waiting for terminating pods: $(echo $terminating)"
  # shellcheck disable=SC2086
  kubectl -n "$NS" wait --for=delete pod $terminating --timeout=90s
fi
kubectl -n "$NS" get pods -l app=inference -o wide
