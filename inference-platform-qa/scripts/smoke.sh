#!/usr/bin/env bash
# End-to-end smoke: testbed up → probe → Go spread check → load + SLO gate.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p reports
LB_PORT=${LB_PORT:-8080} scripts/testbed_up.sh & up=$!
trap 'kill $up 2>/dev/null; wait $up 2>/dev/null || true' EXIT
for _ in $(seq 50); do curl -sf -o /dev/null -X GET localhost:${LB_PORT:-8080}/v1/models && break; sleep 0.1; done
BASE=http://127.0.0.1:${LB_PORT:-8080}/v1
$PY -m inferqa.probe --base-url "$BASE" "smoke test one two three" > reports/probe.json
(cd tools/lbprobe && go run . -url "$BASE/chat/completions" -n 150 -c 12) > reports/spread.json
$PY -m inferqa.loadgen --base-url "$BASE" --requests 120 --concurrency 12 --out reports/load.json > /dev/null
echo "smoke OK — reports/probe.json reports/spread.json reports/load.json"
