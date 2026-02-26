# sentient-evals

`sentient-evals` is a plug-and-play evaluation harness for AI agents (any framework) that can run locally/offline and can also be embedded by the Sentient platform.

## Goals

- Run **tasks** concurrently with **multiple trials** to reduce variance.
- Capture **trajectories (ATIF)** and **outcomes** (final environment state).
- Support **code-based graders** (tests, static checks, tool-call verification), **model-based graders** (LLM-as-judge, multi-judge voting + calibration), and **human review** hooks for calibration.
- Produce a **Harbor-style jobs directory** for debuggability.
- Keep the core harness framework-agnostic; integrate frameworks via adapters.

This design is aligned with Anthropic’s definitions of task, trial, transcript, outcome, grader, harness, and suites ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).

## Install

```bash
pip install sentient-evals
```

For model-based graders:

```bash
pip install "sentient-evals[llm]"
```

## CLI (local mode)

```bash
sentient-evals --help
sentient-evals run --help
```

### Installed CLI agents (Harbor-style)

Built-in installed adapters (run via `--adapter <name>`) mirror Harbor's CLI agents:

- `claude-code`
- `codex`
- `opencode`
- `cursor-cli`
- `cline-cli`
- `gemini-cli`
- `goose`
- `qwen-coder`
- `openhands`
- `swe-agent`
- `mini-swe-agent`
- `aider`

These adapters install the CLI inside the trial environment at runtime.
For sandboxed runs, use Docker, Daytona, or E2B environments so the agent can be installed
in an isolated container.

### Plug-and-play graders via config

For real evaluations, pass an adapter and grader config (TOML or JSON). The CLI can also default to `verifier_script` for task bundles when `tests/test.sh` exists.

### Plug-and-play custom agents via --agent-file

If you have a local Python agent (LangChain, CrewAI, custom loop, etc.), you can point the CLI at a file without packaging your repo:

```bash
sentient-evals run --tasks-dir ./tasks --env local_python --agent-file ./path/to/agent.py:my_agent --config eval.toml
```

The `:my_agent` attribute can be:
- an adapter object with an async `run(task, instruction, seed, env, artifacts)` method
- a factory function that returns such an adapter
- a plain function (sync/async) that returns `dict`/`str` (it will be wrapped automatically)

Example (TOML):

```toml
[adapter]
type = "import"
import_path = "my_project.my_adapter:build_adapter"

[[graders]]
type = "verifier_script"

[[graders]]
type = "static_analysis"
config = { checks = [{ name = "ruff", cmd = "ruff check ." }] }
```

Run:

```bash
sentient-evals run --tasks-dir path/to/tasks --env docker_cli --config eval.toml --concurrency 4
```

## Task bundles (directory format)

For production agent evals, prefer **task bundles** (directories) over plain JSON tasks.

Bundle layout:

- `task.toml` (task id, inputs, env config)
- `instruction.md` (agent-facing instruction)
- `files/` (payload copied into the trial workspace)
- `environment/` (Dockerfile/build context for container backends)
- `tests/` (optional verifier scripts and fixtures)

Run a directory of task bundles:

```bash
sentient-evals run --tasks-dir path/to/tasks --env docker_cli
```

Supported `--env` values:
- `local_python`
- `docker_cli`
- `docker_sdk` (requires `sentient-evals[docker]`)
- `podman_cli` (requires `podman` installed)
- `daytona` (requires `sentient-evals[daytona]` and Daytona configured)
- `e2b` (requires `sentient-evals[e2b]` and `E2B_API_KEY`)
- `modal` (requires `sentient-evals[modal]`, `MODAL_TOKEN_ID`, and `MODAL_TOKEN_SECRET`)

Notes:
- Docker-based runs require Docker (or Podman) installed and running.
- For parallel sandboxed evals, prefer `docker_cli` locally or Daytona/E2B in the cloud.
- Modal is a strong cloud alternative when Daytona/E2B networking or runtime installation constraints block runs.
- If Daytona access is blocked (for example, client-side IP restrictions), use `--env e2b` as a cloud fallback.
- Cloud backends currently assume **single-container** tasks (multi-container orchestration is not yet supported).
- You can throttle cloud provider concurrency per task bundle via `environment.provider_concurrency` in `task.toml`.
- E2B does not build per-task Dockerfiles at runtime. For container tasks on E2B, set `[environment].image` to a valid E2B template id (for example `base`).
- Datasets that rely on Docker image parity (for example many SWE-bench style tasks with `FROM swebench/...`) should run on `docker_cli` or `daytona` unless you provide mapped E2B templates.

Install E2B support:

```bash
pip install "sentient-evals[e2b]"
export E2B_API_KEY=your_api_key
```

Example:

```bash
sentient-evals run --tasks-dir path/to/tasks --env e2b --adapter cursor-cli --config eval.toml
```

Install Modal support:

```bash
pip install "sentient-evals[modal]"
export MODAL_TOKEN_ID=your_token_id
export MODAL_TOKEN_SECRET=your_token_secret
```

Example:

```bash
sentient-evals run --tasks-dir path/to/tasks --env modal --adapter cursor-cli --config eval.toml
```

## Output layout (local runs)

By default, results are written under `jobs/<run_id>/`:

- `jobs/<run_id>/run_config.json`
- `jobs/<run_id>/run_result.json`
- `jobs/<run_id>/trials/<trial_id>/trial_config.json`
- `jobs/<run_id>/trials/<trial_id>/trajectory.json`
- `jobs/<run_id>/trials/<trial_id>/outcome.json`
- `jobs/<run_id>/trials/<trial_id>/trial_result.json`
- `jobs/<run_id>/trials/<trial_id>/judge/` (optional judge artifacts)
- `jobs/<run_id>/trials/<trial_id>/verifier/` (optional verifier artifacts)

## Artifact anatomy

Example directory structure for a run with one trial using an LLM-as-judge grader:

```
jobs/my-run-2025-01-20/
├── run_config.json                # Run-level configuration
├── run_result.json                # Run-level aggregated results
└── trials/
    └── task1__0/
        ├── trial_config.json      # Trial configuration (seed, adapter, etc.)
        ├── trajectory.json        # ATIF trajectory (full run)
        ├── outcome.json           # Final environment state snapshot
        ├── trial_result.json      # Trial-level grader results
        ├── judge/                 # LLM judge artifacts (when using LLM graders)
        │   ├── prompt.txt         # Judge prompt sent to LLM
        │   ├── response.json      # Raw LLM response (full API response)
        │   ├── response.txt       # Extracted text content
        │   └── verdict.json       # Parsed verdict (passed, score, model)
        └── verifier/              # Code-based grader outputs (optional)
            └── test_output.txt    # Example: test stdout/stderr
```

### Example: `jobs/my-run-2025-01-20/run_config.json`

```json
{
  "schema_version": "v1",
  "run_id": "my-run-2025-01-20",
  "suite": {
    "schema_version": "v1",
    "id": "default",
    "trials_per_task": 1,
    "concurrency": 1,
    "seeds": [123]
  },
  "adapter": "json_echo",
  "started_at": "2025-01-20T10:00:00Z",
  "harness_version": "0.0.1",
  "model": null
}
```

### Example: `jobs/my-run-2025-01-20/run_result.json`

```json
{
  "schema_version": "v1",
  "run_id": "my-run-2025-01-20",
  "suite_id": "default",
  "started_at": "2025-01-20T10:00:00Z",
  "finished_at": "2025-01-20T10:00:05Z",
  "task_count": 1,
  "trial_count": 1,
  "passed_trials": 1,
  "failed_trials": 0,
  "avg_score": 1.0
}
```

### Example: `jobs/my-run-2025-01-20/trials/task1__0/judge/verdict.json`

```json
{
  "passed": true,
  "score": 1.0,
  "judge_model": "gpt-4",
  "raw_head": "PASS"
}
```

## License

Apache-2.0. See `LICENSE`.
