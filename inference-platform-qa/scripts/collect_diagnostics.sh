#!/usr/bin/env bash
# Snapshot cluster state for a failed run: the first thing anyone triaging asks for.
set -uo pipefail
mkdir -p reports
kubectl get all,endpointslices,ingress,pdb -n inference -o wide > reports/k8s-state.txt
kubectl -n inference describe pods > reports/pods.txt
kubectl -n inference logs -l app=inference --tail=500 --prefix --timestamps > reports/pod-logs.txt
# Drain timeline: absolute timestamps so pod SIGTERM can be lined up with ingress access-log errors.
kubectl get events -n inference --sort-by=.lastTimestamp \
  -o custom-columns=FIRST:.firstTimestamp,LAST:.lastTimestamp,REASON:.reason,OBJECT:.involvedObject.name,MSG:.message \
  > reports/events-inference.txt
kubectl get pods -n inference -o wide > reports/pods-wide.txt
kubectl -n ingress-nginx logs deploy/ingress-nginx-controller --tail=2000 --timestamps > reports/ingress-logs.txt
kubectl get events -A --sort-by=.lastTimestamp > reports/events.txt
echo "diagnostics in reports/"
