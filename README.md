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

### Plug-and-play graders via config

For real evaluations, pass an adapter and grader config (TOML or JSON). The CLI can also default to `verifier_script` for task bundles when `tests/test.sh` exists.

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

## Output layout (local runs)

By default, results are written under `jobs/<run_id>/`:

- `jobs/<run_id>/config.json`
- `jobs/<run_id>/result.json`
- `jobs/<run_id>/trials/<trial_id>/config.json`
- `jobs/<run_id>/trials/<trial_id>/trajectory.json`
- `jobs/<run_id>/trials/<trial_id>/outcome.json`
- `jobs/<run_id>/trials/<trial_id>/result.json`
- `jobs/<run_id>/trials/<trial_id>/judge/` (optional judge artifacts)
- `jobs/<run_id>/trials/<trial_id>/verifier/` (optional verifier artifacts)

## Artifact anatomy

Example directory structure for a run with one trial using an LLM-as-judge grader:

```
jobs/my-run-2025-01-20/
├── config.json                    # Run-level configuration
├── result.json                    # Run-level aggregated results
└── trials/
    └── task1__0/
        ├── config.json            # Trial configuration (seed, adapter, etc.)
        ├── trajectory.json        # ATIF trajectory (full run)
        ├── outcome.json           # Final environment state snapshot
        ├── result.json            # Trial-level grader results
        ├── judge/                 # LLM judge artifacts (when using LLM graders)
        │   ├── prompt.txt         # Judge prompt sent to LLM
        │   ├── response.json      # Raw LLM response (full API response)
        │   ├── response.txt       # Extracted text content
        │   └── verdict.json       # Parsed verdict (passed, score, model)
        └── verifier/              # Code-based grader outputs (optional)
            └── test_output.txt    # Example: test stdout/stderr
```

### Example: `jobs/my-run-2025-01-20/config.json`

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

### Example: `jobs/my-run-2025-01-20/result.json`

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

