
**Inputs Captured**
- Daytona SDK supports create/start/stop/archive/delete, exec via process sessions, file upload/download, snapshots, and auto‑lifecycle settings. This maps cleanly to `BaseEnvironment` with async operations.
- Harbor’s `DaytonaEnvironment` uses a shared async client, snapshot preference, retries, network_block_all, file batching, and session‑based exec with polling. E2B environment follows similar async lifecycle and fs/exec primitives.

**Production‑Grade Plan for Task 11 (Cloud Sandbox Backends)**

**Phase 1 — Interface & Contracts (Design)**
1. Define a `CloudSandboxBackend` interface that extends existing `BaseEnvironment` but isolates cloud‑specific concerns.
2. Separate “capabilities” from “lifecycle.” Add a `SandboxCapabilities` structure (supports_network_toggle, supports_snapshots, supports_gpu, supports_attach, max_upload_batch, etc.).
3. Keep `BaseEnvironment` slim: `start`, `stop`, `exec`, `upload_file`, `upload_dir`, `download_file`, `download_dir`.
4. Add a provider‑agnostic `SandboxCreateParams` structure to unify snapshot/image/dockerfile references, and map to each provider’s API.
5. Apply SOLID:
   - Single‑responsibility: environment classes only orchestrate provider calls; no CLI/UI logic.
   - Open‑closed: new providers register via factory without modifying existing logic.
   - Interface segregation: avoid monolithic “cloud environment”; use small protocols for fs, exec, lifecycle.
   - Dependency inversion: provider SDK clients behind thin adapters for easier testing.

**Phase 2 — Cloud Provider Abstraction Layer**
1. Add a `providers/` module with `DaytonaProvider`, `ModalProvider`, `E2BProvider` adapters.
2. Each adapter implements:
   - `create_sandbox(params)`, `delete_sandbox()`
   - `exec(command, cwd, env, timeout)`
   - `fs.upload/download`
   - optional `snapshot` helpers if supported
3. Client lifecycle management:
   - Singleton async client manager (as Harbor does) with proper cleanup.
   - Per‑trial sandbox instances only, no shared mutable state.

**Phase 3 — Daytona Implementation (First Provider)**
1. Mirror Harbor’s implementation details:
   - Snapshot preference flow (snapshot → prebuilt image → dockerfile build).
   - `network_block_all` derived from `allow_internet`, with explicit override.
   - Retries around create, exec polling, and fs operations.
2. Adapt the existing `DaytonaEnvironment` in `sentient-evals` to:
   - Use provider adapter for cleaner separation.
   - Normalize exec results into `ExecResult` with stdout/stderr/exit_code/duration.
3. Add fast path for uploading test files and files dir using batch API if supported.

**Phase 4 — Concurrency & Parallelism**
1. Add a provider‑aware concurrency limiter:
   - Global `suite.concurrency`
   - Optional `provider_concurrency` in env config, to avoid provider rate limits.
2. Implement per‑provider task pools for fairness and rate safety.
3. Instrument per‑trial lifecycle events to support scheduling and cancellation.

**Phase 5 — Observability & Reliability**
1. Add structured logs for:
   - Create/start/stop, exec command metadata, file ops durations.
2. Artifact logging to include provider IDs, sandbox IDs, image/snapshot identifiers.
3. Retry policy with exponential backoff, plus standardized error mapping.

**Phase 6 — Modal/E2B Prep (Later)**
1. Use Context7 to fetch Modal/E2B docs when starting those providers.
2. Port Harbor’s E2B environment logic:
   - Async sandbox creation, fs operations, template cache if supported.
3. Modal: check if execution model is job‑based vs sandbox‑based; design adapter accordingly.

**Phase 7 — Tests & Validation**
1. Unit tests:
   - Mock provider client; test lifecycle, exec, upload/download, error mapping.
2. Integration tests (gated):
   - Single “smoke” task bundle run via Daytona.
3. Contract tests:
   - Validate required behavior of any provider adapter.

**Phase 8 — Documentation**
1. Update README:
   - `--env daytona` usage
   - `DAYTONA_API_KEY` and lifecycle expectations
   - limitations: single‑container tasks only
2. Document limitations explicitly:
   - No multi‑container orchestration
   - GPU support provider‑specific
   - Network toggle behavior differs per provider

**Proposed File Structure (Modular, SOLID)**
- `sentient_evals/environments/base.py`
- `sentient_evals/environments/providers/base.py`
- `sentient_evals/environments/providers/daytona.py`
- `sentient_evals/environments/providers/e2b.py`
- `sentient_evals/environments/providers/modal.py`
- `sentient_evals/environments/cloud.py` (glue env that uses a provider)
- `sentient_evals/environments/factory.py` (registration + selection)

**Deliverables**
1. Updated environment interface + provider adapter layer.
2. Daytona provider implementation with snapshot/image/dockerfile support.
3. Concurrency controls and provider rate limiting.
4. Tests + docs + limitations section.

**Execution Status**
- [x] Phase 1 — Interface & Contracts (Design)
- [x] Phase 2 — Cloud Provider Abstraction Layer
- [x] Phase 3 — Daytona Implementation (First Provider)
- [x] Phase 4 — Concurrency & Parallelism
- [x] Phase 5 — Observability & Reliability
- [x] Phase 6 — Modal/E2B Prep (Later)
- [x] Phase 7 — Tests & Validation
- [x] Phase 8 — Documentation (Daytona limitations + provider concurrency note)
