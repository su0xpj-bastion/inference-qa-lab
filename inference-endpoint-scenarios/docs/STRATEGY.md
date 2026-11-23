# Strategy: thirteen scenarios against one hosted inference endpoint

## One client, four questions
Every workflow goes through the same `client.chat()`: same retry policy (429 only; `retry-after` first, then `2**attempt + jitter`; SDK retries off so retries are counted, not hidden), same capture (usage, `time_info`, fingerprint, rate-limit headers, per-chunk timestamps). A difference between workflows is then a difference in the question asked, never in the plumbing.

## Use the server's clock, not just yours
Cerebras returns `time_info` (queue, prompt, completion, total). Client wall time mixes in network, TLS, and parsing; at 2,000 tok/s a 256-token decode takes ~130 ms, so network noise can hide a 2× kernel regression. So:
- **decode throughput** gates on `completion_tokens / time_info.completion_time`;
- **TTFT** is client-measured (that's what users feel) but decomposed: queue + prefill from the server, the rest is unaccounted wall time. A TTFT regression is triaged to the scheduler, prefill, or the network before anyone opens a profiler.

## What gates, what's reported
| Signal | Mock | Live | Why |
|---|---|---|---|
| Usage arithmetic, stream usage, max tokens, stop, schema, tool args, error codes | gate | gate | contract: these are bugs if wrong, at any rate |
| Decode tok/s, prefill tok/s, TTFT p50 | gate | gate (provisional floors) | performance SLOs; floors set well below advertised speed until calibrated |
| T=0 + seed identical output | gate | report | API documents seed as best-effort |
| Logprob drift on shared token prefix | gate (1e-6) | report (0.05) | numerics canary: moves before argmax does |
| Prompt cache hit | gate | report | caching is automatic and best-effort |
| Error rate after retries, retry ratio, latency inflation, isolation | gate | gate | capacity and correctness under concurrency |
| Rate-limit headers | report | report | headroom context for the other numbers |
| Release: regression vs baseline / flaky / pass rate | gate | gate | GO / NO-GO / INCONCLUSIVE |

## Wire: compare encodings, not just speed
Compression is a transport optimization; the risk is that the server decodes a different request than the client meant. The gate is cross-encoding equivalence (same `prompt_tokens`, same answer at T=0), because the dangerous failure is a 200 with a shorter prompt, which no single-encoding test can see. TTFT gain is measured on the same bytes every run (warm prompt cache), so only transfer and decode differ; it gates on the mock and is reported live, where it depends on the network.

## Agent: the loop is the unit
An agent user waits on plan → tools → answer, so latency is gated on the loop, with per-turn TTFT as the diagnosis. Tool-argument validity is gated separately from task success, because a model that recovers from its own malformed call still pays a wasted round trip, and that recovery hides the bug from an outcome-only check.

## Compat: two clients, one contract
Most third-party tooling reaches Cerebras through the OpenAI SDK, so compatibility is tested as parity: the same request through both SDKs must give the same usage and behaviour. The check worth having is `max_tokens`: OpenAI-SDK code still sends the old spelling, and a server that only honours `max_completion_tokens` would return 33x the tokens the caller budgeted, with no error.

## Silent context loss
Conversation and Long context both target 200-OK failures where the model never saw part of the prompt. Recall checks catch it when the lost part mattered; `prompt_tokens` accounting catches it even when it didn't (a clipped haystack whose needle happened to survive). The cache checks use the uncached share of each turn's prompt, not prefill time: with a large fixed system prompt, total prefill time barely moves even with no cache at all, so it can't distinguish a working cache from a dead one.

## Contract checks that cost nothing
Schema conformance validates the traffic the other scenarios already produced against the official OpenAPI spec, so every request doubles as a contract test. It found a useful property in the mock run: a single off-spec enum value (`finish_reason: "eos"`) makes a typed SDK fall back to raw dicts for the whole chunk, so one server-side typo breaks every client that reads `chunk.choices[0].delta`. That's why off-spec values gate, while extra fields (Cerebras's `time_info`, `reasoning`) are only listed: the spec allows them and clients ignore them.

## Accelerator effects you can see from outside
- **Load invariance**: GPU kernels can reduce in a different order depending on batch shape, so a T=0 request can answer differently when the server is busy. The check holds the request fixed and varies only the surrounding load.
- **Numerics baseline**: logprobs on fixed prompts are recorded with the `system_fingerprint`. On the same build, drift must stay at the noise floor (a fail means numerics changed without a release). On a new build, drift is reported, because the size of the move is what the release owner needs.

## Statistical release gate
Eight golden cases catch broken answers; they can't tell a small quality drop from noise. The GSM8K slice can: a one-sided two-proportion test against the qualified baseline gives NO-GO only for a significant drop (p < 0.01), GO when there's no evidence of one (p ≥ 0.10), and INCONCLUSIVE in between, which means "run a bigger slice", not "ship" or "block". With 50 items, 1 flip is noise (p=0.16), 3 is inconclusive (p=0.04), 15 is a regression.

## Soak
Short runs miss slow failures (leaks, a node degrading, connection exhaustion). The soak compares the last window to the first at a rate far below limits, so a creep stands out against its own baseline rather than a fixed threshold.

## Client SDK is part of the release
The SDK matrix runs the suite across pinned SDK releases plus a non-blocking "latest" canary. It found that SDK < 1.59 returns a nested usage field as a dict, which led to a declared minimum version and to newer API params going through `extra_body`.

## Release gate semantics
- **Regression** (a case that passed in the baseline fails every rep) → NO-GO, naming the case and an example bad answer.
- **Flaky** (mixed across reps at T=0) → INCONCLUSIVE. Re-run or investigate; don't let noise ship *or* block a release.
- **New `system_fingerprint`** → reported on its own. A new backend build behind the same model id is exactly when silent regressions ship, so the diff is shown even when everything passes.

## Detectors must be tested too
Each fault in the mock is a real failure class. `tests/test_faults.py` pins the exact set of checks that fail for each one. If a refactor makes `logprob_stability` blind (it once was: the first version stopped comparing at the first differing token, so text nondeterminism hid numeric drift; that's why `nondeterministic` and `logprob_noise` are separate faults), CI goes red.

## Being a good citizen on a shared API
- A request budget per run (`max_requests`), small concurrency (≤ 8), and short outputs. A full live run is ~80 requests and well under a cent of tokens on `qwen-3.8-27b`.
- 429s are never forced on the live API; retry behaviour is proven on the mock.

## Calibrating live thresholds
Run `python -m ies run` on 5+ separate occasions, then set each floor *between* the observed healthy distribution and the regression you want to catch, never at the edge of the healthy one. Record the release baseline once with `--update-baseline` and commit it; the fingerprint in it is the build you qualified.
