"""The pytest plugin (entry point ``pytest11: gpu_testkit``).

Hook-level glue only: the decisions live in hardware/pool/classify/quarantine/
diagnostics/results, which are plain modules with their own unit tests.

Per test, the protocol is::

    collect   -> parse markers into a Requirement; skip with a reason if the host can't qualify
    setup     -> lease qualifying GPUs exclusively (flock), set CUDA_VISIBLE_DEVICES
    run       -> up to 1 + reruns attempts; each records exception types and Xids logged meanwhile
    classify  -> passed / flaky / product / infra
    quarantine-> an active entry turns a final failure into a non-gating xfail, still reported
    diagnose  -> on final failure, write a tarball and attach its path to the JUnit properties
    record    -> the controller folds reports into the JSONL result store and terminal summary
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
from _pytest.runner import call_and_report

from gpu_testkit import hookspecs
from gpu_testkit.backends import BACKEND_CHOICES, Backend, BackendUnavailable, select_backend
from gpu_testkit.classify import (
    DEFAULT_INFRA_EXCEPTIONS,
    DEFAULT_INFRA_PATTERNS,
    Attempt,
    Classification,
    Verdict,
    classify,
    exception_names,
)
from gpu_testkit.diagnostics import BUILTIN_COLLECTORS, Collector, DiagnosticContext, write_bundle
from gpu_testkit.hardware import ArchSpec, HostInfo, Requirement, parse_arch_specs
from gpu_testkit.numerics import CloseReport, assert_close, canonical_dtype
from gpu_testkit.pool import DEFAULT_TIMEOUT_S, DevicePool, Lease, default_lock_dir
from gpu_testkit.quarantine import QuarantineError, QuarantineList
from gpu_testkit.results import PROP_PREFIX, Outcome, Recorder, RunRecord, write_run
from gpu_testkit.xid import XidEvent, parse_xids

MARKERS = (
    "gpus(n=1, arch=None, mem_gb=0): needs n GPUs matching arch (e.g. '>=sm90') with at least "
    "mem_gb decimal GB each; leases them exclusively and sets CUDA_VISIBLE_DEVICES",
    "multinode(nodes=2): needs a job that spans at least this many nodes",
    "numerics(dtype='fp32', rtol=None, atol=None): needs a GPU with native support for dtype; "
    "sets the tolerances used by the `numerics` fixture",
)
DEFAULT_DIAGNOSTICS_DIR = ".gpu-testkit/diagnostics"
RUN_ID_ENV = "GPU_TESTKIT_RUN_ID"
MAX_MESSAGE_CHARS = 4000

REQUIREMENT_KEY = pytest.StashKey[Requirement]()
MARKER_ERROR_KEY = pytest.StashKey[str]()
EXCEPTION_KEY = pytest.StashKey[tuple[tuple[str, ...], str]]()
DEVICES_KEY = pytest.StashKey[tuple[int, ...]]()
PLUGIN_KEY = pytest.StashKey["GpuTestkitPlugin"]()


# --------------------------------------------------------------------------- options


def pytest_addhooks(pluginmanager: pytest.PytestPluginManager) -> None:
    pluginmanager.add_hookspecs(hookspecs)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("gpu-testkit", "GPU test suites")
    group.addoption("--gpu-backend", choices=BACKEND_CHOICES, default=None, help="hardware discovery backend")
    group.addoption("--gpu-fake-spec", default=None, help="fake backend spec: preset (h100x8), JSON, or path")
    group.addoption("--gpu-reruns", type=int, default=None, help="rerun failures up to N times to classify them")
    group.addoption("--gpu-quarantine", default=None, help="quarantine TOML file")
    group.addoption("--gpu-results", default=None, help="write a JSONL result store here")
    group.addoption("--gpu-run-id", default=None, help="run id recorded in the result store")
    group.addoption("--gpu-diagnostics-dir", default=None, help="where failure bundles go")
    group.addoption("--gpu-no-diagnostics", action="store_true", help="don't collect failure bundles")
    group.addoption("--gpu-lock-dir", default=None, help="directory for per-GPU lease lock files")
    group.addoption("--gpu-pool-timeout", type=float, default=None, help="seconds to wait for free GPUs")
    parser.addini("gpu_backend", "default for --gpu-backend", default="auto")
    parser.addini("gpu_reruns", "default for --gpu-reruns", default="0")
    parser.addini("gpu_quarantine", "default for --gpu-quarantine (relative to the ini file)", default="")
    parser.addini("gpu_diagnostics_dir", "default for --gpu-diagnostics-dir", default=DEFAULT_DIAGNOSTICS_DIR)
    parser.addini("gpu_pool_timeout", "default for --gpu-pool-timeout", default=str(DEFAULT_TIMEOUT_S))
    parser.addini(
        "gpu_infra_exceptions",
        "exception names that classify a failure as infra",
        type="linelist",
        default=list(DEFAULT_INFRA_EXCEPTIONS),
    )
    parser.addini(
        "gpu_infra_patterns",
        "failure-text substrings that classify a failure as infra",
        type="linelist",
        default=list(DEFAULT_INFRA_PATTERNS),
    )
    parser.addini(
        "gpu_xid_watch",
        "diff the kernel log around each test for Xid events",
        type="bool",
        default=True,
    )


def _ini_path(config: pytest.Config, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base = config.inipath.parent if config.inipath else config.rootpath
    return (base / path).resolve()


def _option_path(config: pytest.Config, option: str, ini: str | None = None) -> Path | None:
    value = config.getoption(option)
    if value:
        return Path(value).resolve()
    if ini:
        ini_value = str(config.getini(ini))
        return _ini_path(config, ini_value) if ini_value else None
    return None


def pytest_configure(config: pytest.Config) -> None:
    for marker in MARKERS:
        config.addinivalue_line("markers", marker)
    choice = config.getoption("gpu_backend") or str(config.getini("gpu_backend"))
    try:
        backend = select_backend(choice, config.getoption("gpu_fake_spec"))
        host = backend.discover()
        quarantine = QuarantineList.from_path(_option_path(config, "gpu_quarantine", "gpu_quarantine"))
    except (BackendUnavailable, QuarantineError, ValueError, OSError) as exc:
        raise pytest.UsageError(f"gpu-testkit: {exc}") from exc
    plugin = GpuTestkitPlugin(config, backend, host, quarantine)
    config.stash[PLUGIN_KEY] = plugin
    config.pluginmanager.register(plugin, "gpu_testkit_session")


def pytest_unconfigure(config: pytest.Config) -> None:
    plugin = config.stash.get(PLUGIN_KEY, None)
    if plugin is not None:
        config.pluginmanager.unregister(plugin)
        del config.stash[PLUGIN_KEY]


def pytest_gpu_testkit_diagnostic_collectors(config: pytest.Config) -> list[Collector]:
    return list(BUILTIN_COLLECTORS)


# --------------------------------------------------------------------------- markers


def requirement_for(item: pytest.Item) -> Requirement:
    gpus = item.get_closest_marker("gpus")
    multinode = item.get_closest_marker("multinode")
    numerics = item.get_closest_marker("numerics")
    count, mem_gb, nodes = 0, 0.0, 1
    arch: tuple[ArchSpec, ...] = ()
    dtypes: tuple[str, ...] = ()
    if gpus is not None:
        count = int(gpus.args[0] if gpus.args else gpus.kwargs.get("n", 1))
        if count < 1:
            raise ValueError(f"gpus(n={count}): n must be >= 1")
        arch = parse_arch_specs(gpus.kwargs.get("arch"))
        mem_gb = float(gpus.kwargs.get("mem_gb", 0))
    if multinode is not None:
        nodes = int(multinode.args[0] if multinode.args else multinode.kwargs.get("nodes", 2))
    if numerics is not None:
        dtypes = (canonical_dtype(str(numerics.args[0] if numerics.args else numerics.kwargs.get("dtype", "fp32"))),)
    return Requirement(gpus=count, arch=arch, mem_gb=mem_gb, nodes=nodes, dtypes=dtypes)


@dataclass(frozen=True)
class NumericsPolicy:
    """Tolerances from ``@pytest.mark.numerics``; ``None`` means the dtype default."""

    dtype: str
    rtol: float | None = None
    atol: float | None = None

    def assert_close(self, actual: object, expected: object, **overrides: float) -> CloseReport:
        kwargs: dict[str, float | None] = {"rtol": self.rtol, "atol": self.atol}
        kwargs.update(overrides)
        return assert_close(actual, expected, dtype=self.dtype, rtol=kwargs["rtol"], atol=kwargs["atol"])


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def gpu_host(pytestconfig: pytest.Config) -> HostInfo:
    """The hardware the active backend discovered."""
    return pytestconfig.stash[PLUGIN_KEY].host


@pytest.fixture(autouse=True)
def _gpu_testkit_lease(request: pytest.FixtureRequest) -> Iterator[Lease | None]:
    item = request.node
    error = item.stash.get(MARKER_ERROR_KEY, None)
    if error is not None:
        pytest.fail(f"invalid gpu-testkit marker: {error}", pytrace=False)
    requirement = item.stash.get(REQUIREMENT_KEY, None)
    if requirement is None or requirement.gpus == 0:
        yield None
        return
    plugin = request.config.stash[PLUGIN_KEY]
    candidates = [gpu.index for gpu in requirement.qualifying_devices(plugin.host)]
    lease = plugin.pool.acquire(candidates, requirement.gpus, owner=item.nodeid)
    item.stash[DEVICES_KEY] = lease.devices
    patch = pytest.MonkeyPatch()
    patch.setenv("CUDA_VISIBLE_DEVICES", lease.cuda_visible_devices)
    try:
        yield lease
    finally:
        patch.undo()
        lease.release()


@pytest.fixture
def gpu_devices(_gpu_testkit_lease: Lease | None) -> tuple[int, ...]:
    """Device indices leased exclusively to this test (also in CUDA_VISIBLE_DEVICES)."""
    if _gpu_testkit_lease is None:
        pytest.fail("the gpu_devices fixture needs @pytest.mark.gpus(n) on the test", pytrace=False)
    return _gpu_testkit_lease.devices


@pytest.fixture
def numerics(request: pytest.FixtureRequest) -> NumericsPolicy:
    """Tolerance policy from ``@pytest.mark.numerics(dtype, rtol=, atol=)``."""
    marker = request.node.get_closest_marker("numerics")
    if marker is None:
        return NumericsPolicy(dtype="fp32")
    dtype = canonical_dtype(str(marker.args[0] if marker.args else marker.kwargs.get("dtype", "fp32")))
    return NumericsPolicy(dtype=dtype, rtol=marker.kwargs.get("rtol"), atol=marker.kwargs.get("atol"))


# --------------------------------------------------------------------------- protocol


def _run_once(item: pytest.Item, nextitem: pytest.Item | None, may_retry: bool) -> list[pytest.TestReport]:
    """``runtestprotocol`` with one change: if setup/call failed and a retry is allowed,
    teardown stops at the parent, so module/session fixtures (a loaded model, a NCCL
    communicator) survive into the next attempt and only function fixtures are rebuilt."""
    if hasattr(item, "_request") and not item._request:
        item._initrequest()  # type: ignore[attr-defined]
    try:
        reports = [call_and_report(item, "setup", log=False)]
        if reports[0].passed:
            reports.append(call_and_report(item, "call", log=False))
        failed = any(report.failed for report in reports)
        teardown_next = item.parent if (failed and may_retry) else nextitem
        if item.session.shouldfail or item.session.shouldstop:
            teardown_next = None
        reports.append(call_and_report(item, "teardown", log=False, nextitem=teardown_next))
    finally:
        if hasattr(item, "_request"):
            item._request = False
            item.funcargs = None  # type: ignore[attr-defined]
    return reports


def _failure_text(reports: list[pytest.TestReport]) -> str:
    return "\n".join(report.longreprtext for report in reports if report.failed)


class GpuTestkitPlugin:
    def __init__(self, config: pytest.Config, backend: Backend, host: HostInfo, quarantine: QuarantineList) -> None:
        self.config = config
        self.backend = backend
        self.host = host
        self.quarantine = quarantine
        reruns = config.getoption("gpu_reruns")
        self.reruns = int(reruns if reruns is not None else config.getini("gpu_reruns"))
        timeout = config.getoption("gpu_pool_timeout")
        lock_dir = _option_path(config, "gpu_lock_dir") or default_lock_dir(host.hostname)
        self.pool = DevicePool(
            lock_dir,
            timeout_s=float(timeout if timeout is not None else config.getini("gpu_pool_timeout")),
        )
        self.run_id = config.getoption("gpu_run_id") or os.environ.get(RUN_ID_ENV) or time.strftime("%Y%m%d-%H%M%S")
        self.results_path = _option_path(config, "gpu_results")
        diag = (
            None
            if config.getoption("gpu_no_diagnostics")
            else _option_path(config, "gpu_diagnostics_dir", "gpu_diagnostics_dir")
        )
        self.diagnostics_dir = diag / self.run_id if diag else None
        self.infra_exceptions = tuple(config.getini("gpu_infra_exceptions"))
        self.infra_patterns = tuple(config.getini("gpu_infra_patterns"))
        self.xid_watch = bool(config.getini("gpu_xid_watch"))
        self.recorder = Recorder()
        self.is_worker = hasattr(config, "workerinput")

    # ---- collection

    def pytest_report_header(self, config: pytest.Config) -> list[str]:
        expired = len(self.quarantine.expired_entries())
        return [
            f"gpu-testkit: {self.host.summary()}",
            f"gpu-testkit: reruns={self.reruns}, quarantine={len(self.quarantine.entries)} entries "
            f"({expired} expired), lock dir={self.pool.lock_dir}",
        ]

    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        for item in items:
            try:
                requirement = requirement_for(item)
            except (ValueError, TypeError) as exc:
                item.stash[MARKER_ERROR_KEY] = str(exc)
                continue
            item.stash[REQUIREMENT_KEY] = requirement
            reason = requirement.unmet_reason(self.host)
            if reason is not None:
                item.add_marker(pytest.mark.skip(reason=f"gpu-testkit: {reason}"))

    # ---- running

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[None]:
        report = yield
        excinfo = call.excinfo
        if excinfo is not None and call.when in ("setup", "call") and EXCEPTION_KEY not in item.stash:
            item.stash[EXCEPTION_KEY] = (
                exception_names(excinfo.type),
                str(excinfo.value)[:MAX_MESSAGE_CHARS],
            )
        return report

    def _kernel_log(self) -> list[str]:
        if not self.xid_watch:
            return []
        return self.backend.kernel_log() or []

    def _xids_since(self, offset: int) -> tuple[XidEvent, ...]:
        lines = self._kernel_log()
        return tuple(parse_xids(lines[offset:] if len(lines) >= offset else lines))

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item: pytest.Item, nextitem: pytest.Item | None) -> bool | None:
        if item.config.getoption("usepdb", False):
            return None  # let pytest's own protocol drive --pdb
        item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
        attempts: list[Attempt] = []
        timings: list[dict[str, float | int | str]] = []
        earlier_failures: list[str] = []
        reports: list[pytest.TestReport] = []
        first_offset = len(self._kernel_log())
        for number in range(1, self.reruns + 2):
            may_retry = number <= self.reruns
            offset = first_offset if number == 1 else len(self._kernel_log())
            if EXCEPTION_KEY in item.stash:
                del item.stash[EXCEPTION_KEY]
            reports = _run_once(item, nextitem, may_retry)
            timings.extend({"attempt": number, "phase": r.when, "seconds": round(r.duration, 6)} for r in reports)
            failed = any(report.failed for report in reports)
            types, message = item.stash.get(EXCEPTION_KEY, ((), ""))
            text = _failure_text(reports)
            attempts.append(
                Attempt(
                    passed=not failed,
                    exception_types=types,
                    message=f"{message}\n{text}",
                    xids=self._xids_since(offset) if failed else (),
                )
            )
            setup_or_call_failed = any(r.failed for r in reports if r.when != "teardown")
            if not (failed and setup_or_call_failed and may_retry):
                break
            earlier_failures.append(text)
        self._finish(item, reports, attempts, timings, earlier_failures, first_offset)
        for report in reports:
            item.ihook.pytest_runtest_logreport(report=report)
        item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
        return True

    def _finish(
        self,
        item: pytest.Item,
        reports: list[pytest.TestReport],
        attempts: list[Attempt],
        timings: list[dict[str, float | int | str]],
        earlier_failures: list[str],
        kernel_log_offset: int,
    ) -> None:
        skipped = any(report.skipped for report in reports)
        verdict = (
            Classification(Verdict.PASSED)
            if skipped
            else classify(attempts, self.infra_exceptions, self.infra_patterns)
        )
        main = next((r for r in reports if r.when == "call"), reports[0])
        for number, text in enumerate(earlier_failures, start=1):
            main.sections.append((f"gpu-testkit: attempt {number} failed", text))
        props: list[tuple[str, object]] = [
            ("verdict", verdict.verdict.value),
            ("attempts", len(attempts)),
            ("backend", self.host.backend),
            ("host", self.host.hostname),
        ]
        if verdict.evidence:
            props.append(("evidence", verdict.evidence))
        requirement = item.stash.get(REQUIREMENT_KEY, None)
        if requirement is not None and requirement != Requirement():
            props.append(("requires", requirement.describe()))
        devices = item.stash.get(DEVICES_KEY, ())
        if devices:
            names = {gpu.index: f"{gpu.name} {gpu.arch_label}" for gpu in self.host.gpus}
            props.append(("devices", ",".join(map(str, devices))))
            props.append(("gpu", names.get(devices[0], "")))
        entry = self.quarantine.lookup(item.nodeid)
        if entry is not None:
            props.append(("quarantine", entry.label()))
        final_failed = any(report.failed for report in reports)
        if final_failed and self.diagnostics_dir is not None:
            props.append(
                (
                    "diagnostics",
                    str(self._bundle(item, reports, attempts, timings, verdict, devices, kernel_log_offset)),
                )
            )
        if entry is not None and final_failed:
            for report in reports:
                if report.failed and report.when != "teardown":  # a teardown leak still gates
                    report.outcome = "skipped"
                    report.wasxfail = entry.label()
        reports[-1].user_properties.extend((PROP_PREFIX + name, value) for name, value in props)

    def _bundle(
        self,
        item: pytest.Item,
        reports: list[pytest.TestReport],
        attempts: list[Attempt],
        timings: list[dict[str, float | int | str]],
        verdict: Classification,
        devices: tuple[int, ...],
        kernel_log_offset: int,
    ) -> Path:
        assert self.diagnostics_dir is not None
        ctx = DiagnosticContext(
            nodeid=item.nodeid,
            host=self.host,
            backend=self.backend,
            verdict=verdict.verdict.value,
            attempts=len(attempts),
            devices=devices,
            failure_text=_failure_text(reports),
            captured=[(title, body) for report in reports for title, body in report.sections],
            timings=timings,
            kernel_log_offset=kernel_log_offset,
        )
        collectors: list[Collector] = []
        for group in item.config.hook.pytest_gpu_testkit_diagnostic_collectors(config=item.config):
            collectors.extend(group)
        return write_bundle(self.diagnostics_dir, ctx, collectors)

    # ---- reporting (controller side; also runs in-process without xdist)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if not self.is_worker:
            self.recorder.observe(report)

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if self.is_worker or self.results_path is None:
            return
        meta = {
            "python": platform.python_version(),
            "pytest": pytest.__version__,
            "reruns": str(self.reruns),
            "exitstatus": str(int(exitstatus)),
        }
        host = asdict(self.host) | {"summary": self.host.summary()}
        write_run(
            self.results_path,
            RunRecord(run_id=self.run_id, host=host, meta=meta, tests=self.recorder.tests),
        )

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        if self.is_worker:
            return
        tests = self.recorder.tests.values()
        flaky = [r for r in tests if r.verdict == Verdict.FLAKY]
        infra = [r for r in tests if r.verdict == Verdict.INFRA and r.gating_failure]
        hidden = [r for r in tests if r.outcome == Outcome.QUARANTINED]
        passing = [r for r in tests if r.quarantine and r.outcome == Outcome.PASSED]
        expired = self.quarantine.expired_entries()
        stale = self.quarantine.stale_entries(set(self.recorder.tests)) if self.recorder.tests else []
        bundles = [r for r in tests if r.diagnostics]
        if not (flaky or infra or hidden or passing or expired or stale or bundles or self.results_path):
            return
        tr = terminalreporter
        tr.section("gpu-testkit", sep="=")
        for record in flaky:
            tr.line(f"FLAKY  {record.nodeid}: passed on attempt {record.attempts} ({record.tags.get('evidence', '')})")
        for record in infra:
            tr.line(f"INFRA  {record.nodeid}: {record.tags.get('evidence', '')}")
        if hidden:
            tr.line(f"quarantine is hiding {len(hidden)} failure(s):", yellow=True)
            for record in hidden:
                tr.line(f"  {record.nodeid}: {record.message} [{record.quarantine}]")
        if passing:
            tr.line(f"{len(passing)} quarantined test(s) passed; consider releasing:")
            for record in passing:
                tr.line(f"  {record.nodeid} [{record.quarantine}]")
        for entry in expired:
            tr.line(
                f"EXPIRED quarantine for {entry.test} ({entry.owner}, {entry.expires}): gating again",
                red=True,
            )
        for entry in stale:
            tr.line(f"STALE  quarantine entry matches no test that ran: {entry.test} ({entry.owner})")
        for record in bundles:
            tr.line(f"diagnostics: {record.diagnostics}")
        if self.results_path is not None:
            tr.line(f"results: {self.results_path} (run {self.run_id})")
