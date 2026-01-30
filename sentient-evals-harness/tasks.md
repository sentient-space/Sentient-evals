# Sentient Evals Harness (OSS) — Tasks

## Purpose

Build `sentient-evals`: a plug-and-play evaluation harness for AI agents (any framework) that can run locally/offline, and can also be embedded by the Sentient platform as the shared eval engine.

This repo is intentionally **separate** from the Sentient monorepo so:
- The Sentient CLI (`sen`) can remain private.
- The eval harness can be adopted by anyone without platform lock-in.
- Release cadence, CI, licensing, and community contributions are cleanly isolated.

## Repo + integration plan

- **Separate git repo**: `sentient-evals` (recommended)
  - In the Sentient monorepo, consume it as an **external dependency** (pinned version) in `packages/api` and the private CLI packaging.
  - Local dev: editable install (e.g. `pip install -e ../sentient-evals`).

## Execution model (local vs hosted)

Two modes, one shared harness:

- **Local/offline mode (default OSS path)**:
  - Runs fully on the developer machine / CI.
  - No Sentient API required.
  - Produces a Harbor-style job directory with per-trial artifacts (config/result/transcript/outcome) for debuggability.

- **Hosted/platform mode (Sentient backend + worker fleet)**:
  - **Sentient backend (API)** is the control plane: auth, RBAC, dataset ownership, quotas, run creation, status, results queries.
  - **Eval workers** are the data plane: execute trials in parallel, run graders, write artifacts.
  - Start as a separate **worker service** (ECS/Fargate or similar) pulling jobs from a queue; avoid a separate “eval backend” initially.
  - Keep the external contract stable so a future split can happen behind the API gateway if needed.

## Data + artifacts model (Harbor-inspired)

- **Local runs**: write to a `jobs/` folder by default:
  - `jobs/<run_id>/config.json`
  - `jobs/<run_id>/result.json`
  - `jobs/<run_id>/<trial_id>/...` (transcript, verifier outputs, etc.)
- **Hosted runs**:
  - Datasets + large artifacts in object storage (S3/GCS).
  - Structured results in DB (run/task/trial/grader rows).
  - Optional: store “debug bundle” as a Harbor-like tarball for download.

## Platform API surface (hosted mode)

These APIs live in the existing Sentient backend for now:
- Dataset CRUD with presigned uploads/downloads
- Run creation: “create run from suite + dataset + model config”
- Run status + streaming logs (optional)
- Results queries: per-run/per-suite summaries + drill-down to trial artifacts

The `sentient-evals` OSS CLI may include an optional plugin/adapter to submit runs to Sentient, but keep it off-by-default.

## Core abstractions (must-have)

- **Task**: inputs + environment spec + success criteria
- **Trial**: one attempt with deterministic seed/config snapshot
- **Transcript**: messages + tool calls + intermediate artifacts
- **Outcome**: final environment state (files, DB rows, HTTP responses, etc)
- **Grader**: pluggable scoring logic
- **Suite**: collection of tasks + shared config
- **Run**: orchestration metadata (timestamps, model, harness version, git SHA, etc)

### Additional must-have clarifications (production-grade)

These align with the evaluation definitions and practical guidance for agent evals in Anthropic’s guide:
https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents

- **Agentic system scope**: explicitly support evaluating both workflows and agents (e.g., prompt-chaining, routing, parallel voting, orchestrator-workers, evaluator-optimizer loops), not only “single prompt → single response”.
- **Reproducibility**: record model/provider identifiers, prompts/judge prompts, tool schemas, harness version, and configuration snapshots; support deterministic replays where possible.
- **Transcript vs outcome**: graders must be able to score either the full transcript/trajectory or the final outcome state (and often both).
- **Noise/variance**: multi-trial is default; report confidence/variance and avoid overfitting to single-run outcomes.

## Task list (stepwise)

- [x] 1. Define public schemas + contracts
  - Task/trial/transcript/outcome/result JSON schema
  - Stable IDs and reproducibility fields (seed, model, harness version)
  - Export formats: JSON + JSONL; optional JUnit for CI
  - Include explicit versions for: task schema, grader schema, and judge prompt schema
  - Include “transcript/trajectory” schema with tool calls + observations as first-class
  - Include “outcome pointers” schema (files/DB/etc) so outcomes can be validated without bloating artifacts
  - Implementation notes (current scaffold):
    - `sentient_evals.models` includes `schema_version="v1"` plus `PromptSpec`/`JudgeSpec` and `OutcomePointer`
    - Versioned JSON Schemas are in `sentient-evals/schemas/v1/*.schema.json` (export via `sentient-evals schema export`)
    - JUnit XML export for CI is available via `sentient-evals junit <run_dir> --out junit.xml`

- [x] 2. Job/trial filesystem format + result writer (Harbor-style)
  - Implemented canonical run-level `config.json` + `result.json` (replaces run/summary).
  - Trials now guarantee `config.json`, `result.json`, `trajectory.json`, `outcome.json` plus `judge/` and `verifier/` dirs; graders receive an artifact sink to write judge/verifier outputs.
  - Artifact writer uses atomic writes; schemas exported include `RunConfigFile`.
  - Standard on-disk layout (`jobs/<run_id>/...`) for local runs
  - Trial subfolders include:
    - `config.json` (trial config)
    - `result.json` (grader outputs + metrics)
    - `trajectory.json` (ATIF trajectory)
    - `outcome.json` (final state snapshot pointer + summaries)
  - Optional: `verifier/` folder outputs for deterministic graders (tests, stdout/stderr, rewards)
  - Add `judge/` folder for model-based graders (judge prompt, raw judge outputs, disagreement summaries)

- [x] 3. Harness runner (local mode)
  - Multi-trial execution with concurrency controls, resume/cancel, and adapter env wiring (tools via harness).
  - Deterministic replays: record/replay tool outputs per trial (`replay/tool_calls.jsonl`) with strict matching.
  - Capture transcripts + outcomes reliably; judge/verifier artifacts retained.
  - Run-level aggregation now includes pass@k, pass^k, per-task rates, and CI on pass rate.
  - Provenance recorded in run config (runtime/platform digest; ready for task-dir hashing extensions).

- [x] 4. Environment runners
  - Local runner (python function-based tasks)
  - Container runner (Docker) for sandboxed multi-turn agent evals + outcome grading via tests
  - Define sandboxing defaults (timeouts, cpu/mem limits, network policy) and document security boundaries
  - Implementation notes:
    - **Task bundle format**: Directory-based tasks (`task.toml`, `instruction.md`, `environment/`, `tests/`, `files/`) with deterministic SHA256 digest for reproducibility. Loader in `sentient_evals.task_bundles`.
    - **Environment abstraction**: `BaseEnvironment` interface with factory pattern (`sentient_evals.environments`). Supports pluggable backends: `local_python`, `docker_cli`, `docker_sdk`, `podman_cli`, `daytona`.
    - **Local Python runner**: Fast in-process execution via `LocalPythonEnvironment` for unit-style tasks.
    - **Docker CLI runner**: Production default using `docker` CLI with image caching (tagged by task digest), security hardening (`--cap-drop ALL`, `--security-opt no-new-privileges`), resource limits (CPU/memory), and network isolation.
    - **Docker SDK runner**: Optional backend using `docker` Python SDK for better control/streaming (requires `sentient-evals[docker]`).
    - **Podman support**: Rootless-friendly via `DockerCLIEnvironment` with `engine="podman"`.
    - **Daytona cloud runner**: Optional backend for high-concurrency (100s+ parallel trials) using async SDK with shared client manager, snapshot/image caching, and recursive file operations (requires `sentient-evals[daytona]`).
    - **ToolExecutor bridge**: `EnvironmentToolExecutor` adapts `BaseEnvironment` to existing `ToolExecutor` protocol, keeping adapters unchanged.
    - **Provenance**: Environment type, task bundle digest, container image refs, and resource policies recorded in `TrialConfig.provenance` and `RunConfigFile.provenance`.
    - **CLI integration**: `--tasks-dir` for task bundles, `--env` for backend selection, backend-specific flags (`--docker-image-tag-prefix`, `--daytona-snapshot-template`).
    - **Security defaults**: Untrusted task posture with capability dropping, network isolation, resource caps, and documented boundaries in README.

- [x] 5. Agent adapters
  - Minimal adapter interface: `AgentAdapter.run(task, seed, env: ToolExecutor) -> transcript, outcome`
  - **ATIF v1.5 trajectory output**: Trials now write `trajectory.json` (ATIF) as the canonical trace artifact (converted from `TranscriptEvent[]`), with custom metrics preserved under `metrics.extra`.
  - Built-in adapters for common patterns:
    - **Installed/CLI base**: `BaseInstalledAdapter` runs one-or-more `exec` commands (with optional `install`) and has a stdout-first fallback transcript.
    - **Workflow adapter interface**: `WorkflowAdapter` protocol plus a `WorkflowStubAdapter` demonstrating workflow metadata via `TranscriptEvent(kind="metric")`.
  - Optional integrations (behind extras): LangChain, CrewAI, AutoGen
    - Implemented as **thin wrappers** that delegate framework-specific wiring to a user-supplied factory/runner, while recording tool usage via a shared `ToolExecutorRecorder`.

- [x] 6. Graders (baseline set)
  - Code-based:
    - Outcome verification (unit tests, file diffs, DB assertions)
    - Static analysis hooks (lint/type checks)
    - Tool-usage verification (required/forbidden tools, argument constraints)
  - Model-based:
    - LLM-as-judge rubric grading (structured output, reference-free and reference-based)
    - Multi-LLM judge voting (majority/unanimous/average) + disagreement reporting
    - Calibration mode (sampled human review; store rubric + judge prompt version; track drift)
  - Budget graders:
    - Cost/tokens/latency ceilings + scoring
  - Add statistical reporting helpers:
    - paired comparisons for regressions
    - pass@k where applicable
    - confidence intervals / bootstrap for noisy tasks
  - **Implementation notes**:
    - Modular grader architecture with three tiers: deterministic (`graders/deterministic/`), model-based (`graders/judge/`), and human review (`graders/human/`).
    - Deterministic graders: `ExactMatchGrader`, `StateCheckGrader` (with operator support: `$eq`, `$regex`, etc.), `StaticAnalysisGrader`, `ToolUsageGrader`, `VerifierScriptGrader` (Harbor-style `tests/test.sh` execution).
    - Model-based graders: `LLMJudgeGrader` (using `JudgeClient` protocol), `MultiLLMJudgeGrader` (aggregation), `PairwiseJudgeGrader` (A/B comparison).
    - Human graders: `HumanReviewGrader` for creating review packets.
    - Budget graders: `BudgetGrader` for token/cost/latency limits.
    - CLI-driven configuration: graders can be specified via `--config` (TOML/JSON) or `--graders-file` (JSON array), enabling plug-and-play grader selection without writing Python code.
    - Environment-aware grading: `EnvironmentGrader` protocol allows graders to access the trial workspace (`env: ToolExecutor`) for outcome verification (e.g., `VerifierScriptGrader`, `StaticAnalysisGrader`).
    - Registry system (`registry.py`) supports built-in grader types and dynamic imports via `import_path`.
  - **Approach recap (what shipped / why)**:
    - Graders are configured via TOML/JSON and run per-trial, writing artifacts under `trials/<trial_id>/{verifier,judge}/` for debuggability.
    - Environment-aware graders (`VerifierScriptGrader`, `StaticAnalysisGrader`) execute inside the sandbox via `ToolExecutor`, so results reflect the final trial workspace state.
    - Harbor-style `tests/test.sh` verifiers are supported, with a safe fallback to run via `sh` when executable bits are not preserved by host↔container copying.

- [x] 6.5. Built-in adapters for common CLI agents 
  - Ship built-in adapters for commercial CLI agents so users can run evals with only a config file (no Python glue code).
  - Target agents (inspired by Harbor's installed agent pattern):
    - **Claude Code** (`claude-code`): Anthropic's CLI coding agent
    - **Codex CLI** (`codex`): OpenAI's CLI coding agent
    - **OpenCode** (`opencode`): Open-source CLI coding agent
    - Additional candidates: `cursor-cli`, `cline-cli`, `aider`, `gemini-cli`, `goose`, `qwen-coder`
  - Implementation approach (Harbor-inspired):
    - Extend `BaseInstalledAdapter` pattern (`adapters.py`) with agent-specific subclasses.
    - Each adapter handles:
      - Installation detection/verification (check if CLI tool is installed)
      - Command construction (map task input to CLI invocation with proper args/env vars)
      - Output parsing (extract transcript/outcome from CLI stdout/stderr)
      - Model parameter passing (via CLI flags/env vars)
    - Registry integration: register adapters in `registry.py` so they're available via `--adapter <name>` or config files.
    - CLI convenience: users can run `sentient-evals run --tasks-dir ./tasks --adapter claude-code --model anthropic/claude-opus-4-1` without writing adapter code.
    - Configuration via TOML/JSON:
      ```toml
      [adapter]
      type = "claude-code"
      config = { model = "anthropic/claude-opus-4-1", timeout_s = 300 }
      ```
    - Installation scripts: optional `install-{agent}.sh` helpers (similar to Harbor's `install-{agent-name}.sh.j2`) for CI/setup automation.
    - Fallback behavior: if agent CLI not found, provide clear error with installation instructions.
  - Success criteria:
    - Users can evaluate Claude Code, Codex CLI, and OpenCode with zero Python adapter code.
    - Adapters handle common CLI patterns (stdin/stdout, file-based I/O, environment variables for API keys).
    - Registry auto-discovers built-in adapters; external adapters via `import_path` still supported.
  - **Approach recap (what shipped / why)**:
    - Implemented a Harbor-style `BaseInstalledAdapter` + per-agent subclasses with `install-*.sh` scripts, registered in `registry.py` so users can select agents via `--adapter` or config files.
    - Docker-backed runs were hardened for real-world usage:
      - `--jobs-dir` is resolved to an absolute path (required for Docker bind-mounts).
      - Docker image existence checks correctly trigger builds.
      - Container keepalive command is properly quoted so the container stays running for `exec` operations.
    - Cursor CLI adapter was aligned with current Cursor CLI docs:
      - Uses `agent --print` with `--api-key`/`CURSOR_API_KEY` auth, and supports Cursor-native models (e.g. `composer-1`).
      - Supports `linux/amd64` platform override in task bundles (critical for Apple Silicon users running amd64-only agent CLIs).

- [ ] 6.6. Built-in benchmarks + dataset registry (Harbor-style plug-and-play)
  - Ship a small set of **built-in** benchmark/task packs (smoke + starter suites) and a registry mechanism for larger community datasets.
  - UX goal: users can run evals without authoring task folders:
    - `sentient-evals datasets list`
    - `sentient-evals run --dataset <name>@<version> --env docker_cli --adapter cursor-cli --config eval.toml`
    - `sentient-evals datasets pull <name>@<version>` (cache locally)
  - Also ship a scaffolder for custom tasks (Harbor parity):
    - `sentient-evals tasks create <task-name>` → generates `task.toml`, `instruction.md`, `environment/Dockerfile`, `tests/test.sh`
  - Implementation sketch:
    - `datasets/` module: registry index (JSON) + downloader (git/tarball) + local cache dir
    - CLI commands: `datasets list/pull/path`, `tasks create`
    - Keep the harness universal: registry is optional; local `--tasks-dir` remains supported.

- [ ] 7. Reporting + aggregation
  - Aggregate over trials (mean/variance, pass@k where relevant)
  - Per-suite dashboards in terminal output (summary + failure drill-down)
  - “Regression suite” mode (compare two runs; highlight deltas)
  - Add “judge reliability” reporting: agreement rates, tie rates, and examples of judge disagreement for calibration

- [ ] 8. OSS CLI (local mode)
  - `sentient-evals run --suite ...`
  - `sentient-evals report --run ...`
  - `sentient-evals diff --baseline ... --candidate ...`

- [ ] 9. Hosted mode: worker execution contract (Sentient backend + workers)
  - Define a minimal “run spec” and “trial spec” payload
  - Queue contract (job message schema; idempotency keys)
  - Worker writes:
    - artifact bundle to object storage (optional)
    - trial results to Sentient DB via internal API or direct DB session (choose one)
  - Concurrency controls + quotas at run level (avoid unbounded fan-out)

- [ ] 10. Hosted mode: Sentient backend endpoints (control plane)
  - Dataset upload/download (presigned URLs)
  - Create run, list runs, get run, cancel run
  - Results endpoints: summary + drill-down to trials + artifact links

- [ ] 11. Optional: cloud sandbox provider adapters (Harbor-like scaling)
  - Pluggable “sandbox backend” interface
  - First provider: pick one (e.g., Daytona/Modal/E2B), support single-container tasks initially
  - Document limitations (multi-container vs single-container)

- [ ] 12. CI templates + examples
  - Minimal example suites (toy + realistic)
  - GitHub Actions template running a small regression suite
  - Documentation: “How to add a task”, “How to write a grader”, “How to run multi-trial”

- [ ] 13. RL evaluation support (after baseline graders + harness are stable)
  - Standardize “reward” reporting + trajectory capture suitable for RL-style evaluation
  - Add RL-specific metrics hooks (episode reward, success rate, step budget, cost/latency per episode)
  - Keep RL training/optimization out of v1; focus on evaluation harness contracts first

## What the Sentient private CLI should do

Two modes (recommended):
- **Local mode**: `sen evals run` shells out to `sentient-evals` (no Sentient backend required).
- **Hosted mode**: `sen evals trigger` calls Sentient backend endpoints to run eval pipelines against a deployment and store results for the dashboard.

The private CLI should not re-implement harness logic; it should depend on `sentient-evals` for local mode and use Sentient API for hosted mode.

