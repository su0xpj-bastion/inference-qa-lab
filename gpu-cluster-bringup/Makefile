PY ?= .venv/bin/python

.PHONY: venv test fleet bringup lint
venv:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt ansible-core

test:
	$(PY) -m pytest

fleet:
	$(PY) -m bringup gen-fleet --scenario defects --out fleet/defects

bringup: fleet
	$(PY) -m bringup run --fleet fleet/defects --manifest manifests/golden.json --out reports/defects; \
	  rc=$$?; test $$rc -eq 0 -o $$rc -eq 3

lint:
	shellcheck scripts/collect.sh
	.venv/bin/ansible-playbook --syntax-check -i provision/onprem/inventory.ini provision/onprem/site.yml provision/onprem/acceptance.yml
