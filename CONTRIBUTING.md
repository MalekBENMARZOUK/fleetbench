# Contributing

FleetBench is designed to be small enough to understand locally and realistic enough to exercise production workflows. Contributions should preserve deterministic behavior, clear metrics, and fast feedback.

## Local Setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quality Gate

Before opening a pull request, run:

```bash
python -m pytest
python -m ruff check src tests scripts examples
python -m ruff format --check src tests scripts examples
python -m mypy
```

Use smaller smoke runs while iterating:

```bash
fleetbench benchmark --output-dir results/dev --family urban_depot_small --method naive_baseline --seeds 101 --no-plots --no-publication
```

## Contribution Guidelines

- Keep random behavior seeded and reproducible.
- Add tests for new scenario rules, methods, metrics, or report columns.
- Document user-facing CLI options and output schema changes.
- Avoid introducing real fleet data into this repository.
- Keep generated benchmark outputs under `results/`, which is ignored by Git.

## Pull Request Checklist

- Tests pass locally or the known failure is explained.
- New behavior has focused tests.
- Documentation reflects CLI, output, or workflow changes.
- Release notes are updated for user-facing changes.
