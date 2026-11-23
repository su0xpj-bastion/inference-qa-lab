# Design notes

## Why a custom rerun protocol instead of pytest-rerunfailures

Rerunning is a means to classify, not to get green. The protocol (`plugin._run_once`) re-implements
`runtestprotocol` with one change: when setup/call failed and a retry remains, teardown stops at the
item's parent. Function fixtures are rebuilt for the retry, but module and session fixtures (a loaded
model, an NCCL communicator) survive, which is what makes reruns affordable on GPU suites. A test
proves the module fixture is built once across three attempts.

Every attempt records the exception's MRO names, the failure text, and the Xid events the kernel
logged while it ran. `classify()` then decides:

| Verdict | Rule |
|---|---|
| `flaky` | a later attempt passed |
| `infra` | failed, with evidence: a hardware Xid (48, 74, 79, 94, 95, ...), an exception listed in `gpu_infra_exceptions`, or a `gpu_infra_patterns` substring |
| `product` | failed, no infra evidence (the default, on purpose) |

Application-side Xids (13, 31, 43, 45) are product evidence: an MMU fault usually means the kernel under
test read a bad address. Calling it infra would hide exactly the bug the suite exists to catch.

## Device leases

One `flock` file per device in a shared lock dir. Leases are all-or-nothing (no hold-and-wait, so no
deadlock between workers asking for different counts), the holder writes its node id and pid into the
file for `gpu-testkit pool` and for timeout messages, and the kernel releases locks when a process dies,
so a SIGKILLed worker can't leak a GPU. Pool timeouts are classified as infra.

Limits: POSIX only (`fcntl`); the lock dir must be on a local filesystem (flock over NFS is unreliable);
NVML ignores `CUDA_VISIBLE_DEVICES`, so a test that calls NVML itself sees every device.

## Memory units

`mem_gb` is decimal GB. An H100 80GB reports 81559 MiB = 79.6 GiB. A GiB comparison would skip
`mem_gb=80` tests on the card they were written for.

## Quarantine semantics

An active entry turns setup/call failures into xfail (non-gating) and tags the record `quarantined`.
Teardown failures still gate: a leaked resource affects the next test, which isn't covered by anyone's
quarantine. Passing quarantined tests are listed as release candidates; expired entries gate again and
are called out; entries that match nothing that ran are reported as stale.

## Result store

JSONL, one header line plus one line per test, written by the controller from `TestReport`s only, so
xdist and in-process runs produce identical files (tested both ways). `gpu-testkit report` diffs two
stores into: new failures, still failing, fixed, flaky, infra, quarantine-hidden, quarantined-but-passing,
lost coverage (passed before, skipped now: usually a runner that lost GPUs), slowest regressions
(ratio and absolute-delta thresholds, so a 10 ms test going to 40 ms isn't news), and removed tests.
