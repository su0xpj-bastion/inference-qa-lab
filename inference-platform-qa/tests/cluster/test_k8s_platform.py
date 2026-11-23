"""Cluster tests: the same assertions, against real Kubernetes + ingress-nginx.

Run in CI by the kind job, or by hand against EKS/staging:
    INGRESS_URL=http://localhost:8080 pytest -m cluster tests/cluster

Each test states the platform property it protects. Anything that
mutates the cluster (kill, restart) is scoped to the `inference`
namespace and waits for full recovery before returning.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections import Counter

import openai
import pytest

from inferqa.loadgen import build_report, run_load
from inferqa.probe import stream_chat

NS = os.environ.get("INFERENCE_NS", "inference")
INGRESS_URL = os.environ.get("INGRESS_URL", "http://localhost:8080")
INGRESS_SYNC_BUDGET_S = 10.0  # must be < preStop (5s) + drain grace (10s)
HOST_HEADER = {"Host": os.environ.get("INGRESS_HOST", "inference.local")}

pytestmark = [
    pytest.mark.cluster,
    pytest.mark.skipif(not shutil.which("kubectl"), reason="kubectl not installed"),
    pytest.mark.timeout(300),
]


def kubectl(*args: str) -> str:
    return subprocess.run(["kubectl", "-n", NS, *args], check=True, capture_output=True, text=True).stdout


def kjson(*args: str) -> dict:
    return json.loads(kubectl(*args, "-o", "json"))


def ready_pods() -> list[str]:
    pods = kjson("get", "pods", "-l", "app=inference")["items"]
    return [p["metadata"]["name"] for p in pods
            if p["metadata"].get("deletionTimestamp") is None
            and any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", []))]


def endpoint_addresses() -> int:
    slices = kjson("get", "endpointslices", "-l", "kubernetes.io/service-name=inference")["items"]
    return sum(1 for s in slices for e in s.get("endpoints") or [] if e["conditions"].get("ready"))


def wait_until(pred, timeout: float = 90, step: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(step)
    raise TimeoutError(pred.__doc__ or "condition not met")


@pytest.fixture
def client() -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{INGRESS_URL}/v1", api_key="test", max_retries=0,
                         timeout=30, default_headers=HOST_HEADER)


@pytest.fixture(autouse=True)
def healthy_baseline():
    kubectl("rollout", "status", "deployment/inference", "--timeout=120s")
    wait_until(lambda: endpoint_addresses() == 3)
    yield
    kubectl("rollout", "status", "deployment/inference", "--timeout=120s")


def test_endpoints_match_ready_pods():
    """Service discovery: every Ready pod is an endpoint, and nothing else is."""
    assert endpoint_addresses() == len(ready_pods()) == 3


def test_ingress_reaches_every_pod(client):
    """Load balancing: ingress-nginx spreads traffic across all replicas."""
    seen = Counter(stream_chat(client, "ping").served_by for _ in range(30))
    assert set(seen) == set(ready_pods()), seen


def test_ingress_does_not_buffer_sse(client):
    """proxy-buffering off is applied; TTFT is not inflated to total latency."""
    r = stream_chat(client, " ".join(["w"] * 30))
    assert r.ok and r.timing.ttft < r.timing.total * 0.5


def test_not_ready_pod_leaves_rotation(client):
    """Readiness gating: a pod failing /readyz leaves EndpointSlices, then ingress stops routing to it
    within INGRESS_SYNC_BUDGET_S.

    The two steps are separately asserted because they are separately eventual: CI run 43620786802
    showed ingress-nginx still routing to the pod after EndpointSlices had already dropped it. The
    budget must stay below preStop + drain grace (15s), or rolling restarts will drop requests.
    """
    victim = ready_pods()[0]
    kubectl("exec", victim, "--", "python", "-c",
            "import urllib.request as u;u.urlopen(u.Request('http://127.0.0.1:8000/admin/ready',"
            "data=b'{\"ready\":false}',method='POST'))")
    flipped = time.monotonic()
    try:
        wait_until(lambda: endpoint_addresses() == 2, timeout=30)
        k8s_lag = time.monotonic() - flipped

        def ingress_excludes_victim() -> bool:
            """ingress-nginx stopped routing to the not-ready pod"""
            return victim not in {stream_chat(client, "ping").served_by for _ in range(9)}

        wait_until(ingress_excludes_victim, timeout=INGRESS_SYNC_BUDGET_S, step=0.2)
        ingress_lag = time.monotonic() - flipped
        print(f"[LAG] readiness->EndpointSlice {k8s_lag:.1f}s, readiness->ingress {ingress_lag:.1f}s")
        # Converged, not flapping: it must stay out.
        assert victim not in Counter(stream_chat(client, "ping").served_by for _ in range(30))
    finally:
        kubectl("delete", "pod", victim, "--wait=false")


def test_pod_kill_under_load_stays_within_error_budget(client):
    """Chaos: SIGKILL one pod during load. Ingress retries connect errors; budget is 2%."""
    results = {}
    t = threading.Thread(target=lambda: results.setdefault("r", run_load(client, 200, 10)))
    t.start()
    time.sleep(2)
    kubectl("delete", "pod", ready_pods()[0], "--grace-period=0", "--force")
    t.join()
    report = build_report(results["r"])
    assert report["error_rate"] <= 0.02, report


def test_rolling_restart_is_zero_error(client):
    """Release safety: maxUnavailable 0 + preStop + SIGTERM drain ⇒ no failed requests during a rollout."""
    results = {}
    t = threading.Thread(target=lambda: results.setdefault("r", run_load(client, 300, 8)))
    t.start()
    time.sleep(1)
    kubectl("rollout", "restart", "deployment/inference")
    kubectl("rollout", "status", "deployment/inference", "--timeout=180s")
    t.join()
    report = build_report(results["r"])
    assert report["error_rate"] == 0, report
