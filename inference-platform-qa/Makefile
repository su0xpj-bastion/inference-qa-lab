PY ?= .venv/bin/python
.PHONY: venv test unit integration go smoke testbed kind cluster-test clean

venv:        ; python3 -m venv .venv && $(PY) -m pip install -r requirements-dev.txt
test:        ; $(PY) -m pytest
unit:        ; $(PY) -m pytest tests/unit
integration: ; $(PY) -m pytest -m integration
go:          ; cd tools/lbprobe && go vet ./... && go test ./...
smoke:       ; scripts/smoke.sh
testbed:     ; scripts/testbed_up.sh
kind:        ; kind create cluster --name inference --config deploy/kind/kind-config.yaml && scripts/kind_deploy.sh inference
cluster-test:; INGRESS_URL=http://localhost:8080 $(PY) -m pytest -m cluster tests/cluster -o addopts=""
clean:       ; rm -rf reports .pytest_cache tools/lbprobe/lbprobe; kind delete cluster --name inference 2>/dev/null || true
