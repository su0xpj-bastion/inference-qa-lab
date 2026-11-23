.PHONY: venv test lint demo
venv:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
test:
	.venv/bin/python -m pytest
lint:
	.venv/bin/yamllint .
	shellcheck launch/slurm/*.sh launch/slurm/*.sbatch launch/ray/*.sh scripts/*.sh
demo:
	.venv/bin/python -m fabval simulate --nodes 8 --fault downtrain:node-03:2 --fault pfc:node-05:4 \
	  --fault numa:node-02:6 --fault gid:node-07 --fault corrupt:node-04 --out runs/demo
	.venv/bin/python -m fabval analyze runs/demo --md reports/fabric.md --json reports/fabric.json || test $$? -eq 1
