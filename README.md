# inference-qa-lab

Testing for AI model serving, from a laptop up to a GPU cluster: whether answers come back fast, correct and complete, whether the serving software and hardware underneath behave, and whether a release is safe to ship.

Ten projects, each in its own folder with its own README, dependencies, tests and GitHub Actions workflow. A workflow runs only when its folder changes. **local-inference-bench is measured live on real hardware**; the others run against simulated engines, clusters and devices, or parse logs and telemetry in their real formats, and each README says which.

| Project | What it is, in plain terms | Tests |
|---|---|---|
| [local-inference-bench](local-inference-bench/) | **Measured live** on a MacBook M1 (8 GB): speed, concurrency, caching and long-prompt checks for AI models running on the machine itself. Found that the engine silently drops the start of long prompts by default while still reporting success | 39 |
| [inference-platform-qa](inference-platform-qa/) | A test lab for an AI service on Kubernetes (load balancer, web server, service discovery, servers). Its first run found a real bug: 2 of 300 requests failed during a routine deploy, traced through the ingress logs to connection draining and fixed to zero | 62 + 6 on a real cluster in CI |
| [llm-serving-validation](llm-serving-validation/) | Checks the parts of a model server that break quietly (batching, memory paging, prompt caching) by comparing every answer with a reference computed the simple way | 78 |
| [inference-endpoint-scenarios](inference-endpoint-scenarios/) | 13 real-world usage scenarios for an AI API (heavy traffic, rate limits, long runs, long documents) against a fake server that breaks in 44 ways, each caught | 76 |
| [sampling-precision-validation](sampling-precision-validation/) | Statistical tests that catch a model picking words with the wrong odds, and an accuracy gate for compressed (low-precision) models | 34 |
| [inference-release-integration](inference-release-integration/) | Release testing for a device software stack (compiler, runtime, kernels): groups failing tests into tickets with owners and separates the change that caused a bug from the one that only exposed it | 41 |
| [gpu-cluster-bringup](gpu-cluster-bringup/) | The checks a new GPU server must pass before going live, in stages, with clear pass / fail limits; on a simulated 36-server fleet it caught 11 of 11 planted defects | 68 |
| [gpu-fabric-validation](gpu-fabric-validation/) | Checks the network between GPUs by running the same job three ways (Kubernetes, Slurm, Ray) and pins problems to the exact server and network card | 66 |
| [gpu-fault-rca](gpu-fault-rca/) | Breaks a simulated 32-GPU system in 13 ways and names the root cause from its logs and telemetry: 45 of 45 right, no false alarms | 56 |
| [gpu-testkit](gpu-testkit/) | A pytest plugin for hardware tests: tests declare the hardware they need, get exclusive devices, and failures are labelled flaky, product bug or hardware problem | 120 |

Every number above comes from the project's own tests or recorded runs.
