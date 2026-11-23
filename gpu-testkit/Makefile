PY := .venv/bin/python
FAKE ?= h100x8

.PHONY: venv test lint example report all

venv:
	python3 -m venv .venv && .venv/bin/pip install -q -e '.[dev]'

test:            ## the framework's own tests (unit + pytester end-to-end)
	$(PY) -m pytest

lint:
	.venv/bin/ruff check src tests examples && .venv/bin/ruff format --check src tests examples && .venv/bin/mypy

example:         ## the example GPU suite on a fake 8x H100 host, 4 xdist workers
	GPU_TESTKIT_FAKE_SPEC=$(FAKE) $(PY) -m pytest examples/llm_serving_suite -n 4 -rs \
	  --gpu-results=reports/example-results.jsonl --junitxml=reports/example-junit.xml

report: example  ## diff the example run against the committed baseline
	.venv/bin/gpu-testkit report reports/example-results.jsonl --base examples/llm_serving_suite/baseline.jsonl --fail-on-new

all: lint test report
