"""Testbed fixtures: real processes for replicas, in-process proxy in front.

Replicas run as subprocesses (not threads) so tests can SIGKILL and
SIGTERM them exactly like the kubelet does to a pod.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import openai
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from testbed.lb import build_proxy  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 10.0, want: int = 200) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as r:
                if r.status == want:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.05)
    raise TimeoutError(f"{url} not {want} after {timeout}s")


def http_json(method: str, url: str, body: dict | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"content-type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, dict(r.headers), json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read() or b"{}")


@dataclass
class Replica:
    rid: str
    port: int
    proc: subprocess.Popen

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def kill(self) -> None:
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait(5)

    def term(self) -> None:
        self.proc.send_signal(signal.SIGTERM)


def spawn_replica(rid: str, **env: str) -> Replica:
    port = free_port()
    full_env = {**os.environ, "REPLICA_ID": rid, "MOCKSERVE_ALLOW_FAULTS": "1",
                "MOCKSERVE_TTFT_MS": "10", "MOCKSERVE_TBT_MS": "1", "PYTHONPATH": str(ROOT), **env}
    proc = subprocess.Popen([sys.executable, "-m", "mockserve", "--host", "127.0.0.1", "--port", str(port)],
                            env=full_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    replica = Replica(rid, port, proc)
    wait_http(f"{replica.url}/healthz")
    return replica


@dataclass
class Testbed:
    replicas: list[Replica]
    lb_port: int
    lb_server: object
    extra: list[Replica] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.lb_port}/v1"

    @property
    def lb_url(self) -> str:
        return f"http://127.0.0.1:{self.lb_port}"

    def client(self) -> openai.OpenAI:
        return openai.OpenAI(base_url=self.base_url, api_key="test", max_retries=0, timeout=10)


def _start_lb(peers: list[str], fail_timeout: float):
    server = build_proxy("127.0.0.1", 0, peers, fail_timeout=fail_timeout)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def replica():
    r = spawn_replica("solo")
    yield r
    if r.proc.poll() is None:
        r.kill()


@pytest.fixture
def testbed():
    """Three replicas behind the NGINX-semantics proxy."""
    replicas = [spawn_replica(f"pod-{i}") for i in range(3)]
    server = _start_lb([f"127.0.0.1:{r.port}" for r in replicas], fail_timeout=2.0)
    tb = Testbed(replicas, server.server_address[1], server)
    yield tb
    server.shutdown()
    for r in replicas + tb.extra:
        if r.proc.poll() is None:
            r.kill()
