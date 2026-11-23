# Release validation plan: inference platform

The scope is the deployment path (CI/CD → Kubernetes → ingress → service discovery → pods). Model quality is out of scope.

## 1. Entry criteria
- The image builds, and unit and integration tests are green on the PR.
- The manifests render (`kubectl kustomize deploy/overlays/<env>`) with no diff surprises in ArgoCD.

## 2. Test layers and gates

| Gate | Where | Blocks | Budget |
|---|---|---|---|
| Unit | every PR | merge | 100% pass |
| Integration (local testbed) | every PR | merge | 100% pass, under 30s |
| Smoke + SLO (`slo/default.json`) | every PR | merge | 0 SLO violations |
| Cluster (kind + ingress-nginx) | every PR to main | merge | 100% pass; chaos error rate ≤ 2% |
| Staging (EKS, ArgoCD sync) | after merge | promotion | zero-error rolling restart; p99 TTFT within 10% of the previous release |
| Canary on hardware cluster | promotion | full rollout | `lbprobe` skew ≤ 0.5 and error rate ≤ 0.1% over 15 min |

## 3. Risk-based focus for a new platform feature
Each feature gets one question per layer before any code is written:

1. **Contract**: does the SSE / usage / error shape change? → contract test.
2. **Routing**: does it change which pod gets a request (sticky sessions, model-aware routing)? → spread and failover tests.
3. **Lifecycle**: does it add startup work or long-lived connections? → drain and rolling-restart tests with that workload.
4. **Capacity**: does it change per-request cost? → loadgen at current peak concurrency, with an SLO diff against the baseline.
5. **Observability**: can on-call tell it's broken from metrics alone? → assert on the new `/metrics` series.

## 4. Exit criteria
- All gates are green, with no unexplained flakes. A flake becomes a ticket with an owner, never a retry.
- The load report is attached to the release, and its p99 TTFT/TBT diff against the last release is explained.
- The rollback has been exercised: an ArgoCD sync to the previous revision completes, and the SLO gate passes after it.

## 5. Triage playbook (first 10 minutes)
1. `scripts/collect_diagnostics.sh`: pods, EndpointSlices, ingress logs, events.
2. Is the problem one replica or all of them? Compare `lbprobe` output (`replicas`, `status`) with `mockserve_requests_total{code}` per replica.
3. If it's one replica: readiness flapping? OOMKilled? Missing from EndpointSlices?
4. If it's all replicas: ingress config (buffering, timeouts, next-upstream), DNS, or the upstream SLO itself.
5. If TTFT regressed but nothing errors: run `python -m inferqa.latency --baseline <last-good.json>`. The hop that owns the delta names the team: `lb` means ingress or networking, `sched_wait` the serving scheduler, `prefill` the model or runtime.
6. If streams are truncated: check `finish_reason` warnings and the missing-usage warnings in the probe report.
