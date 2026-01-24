# Contributing

Thanks for contributing to `sentient-evals`.

## Dev setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[llm]"
pip install -e ".[llm]"  # optional; enables model-based graders
pip install -e .
pip install -U pytest ruff
```

## Quality checks

```bash
ruff check .
ruff format .
pytest
```

## Project principles

- Keep the core harness small and composable.
- Prefer explicit, versioned schemas for all artifacts.
- Make eval runs reproducible (seeds, configs, prompt/judge versions).
- Treat transcripts + outcomes as first-class.

