#!/usr/bin/env bash
# Install ingress-nginx, build + load the image, apply the kind overlay. Usage: scripts/kind_deploy.sh [cluster]
set -euo pipefail
cd "$(dirname "$0")/.."
CLUSTER=${1:-inference}
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.2/deploy/static/provider/kind/deploy.yaml
kubectl -n ingress-nginx rollout status deploy/ingress-nginx-controller --timeout=180s
docker build -t inference-mock:dev .
kind load docker-image inference-mock:dev --name "$CLUSTER"
kubectl create namespace inference --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -k deploy/overlays/kind
kubectl -n inference rollout status deploy/inference --timeout=180s
