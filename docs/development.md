# Development Guide

## Repository Layout

```text
src/ev_fleet_benchmark/   package source
tests/                    pytest suite
scripts/                  profiling and study helpers
docs/                     project documentation
examples/                 small runnable examples
.github/workflows/        CI and release automation
```

## Environment

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Test Strategy

The suite covers:

- Scenario generation and sensitivity transformations.
- Method registry behavior and scheduling outputs.
- Simulation validation and edge cases.
- Reporting tables and Markdown summaries.
- CLI validation and progress telemetry.
- Profiling baseline comparison helpers.

Run all tests:

```bash
python -m pytest
```

Run a focused file:

```bash
python -m pytest tests/test_scenarios.py
```

## Adding a Method

1. Add a class under `src/ev_fleet_benchmark/methods/`.
2. Inherit from `ScheduleMethod`.
3. Define a unique `name`.
4. Return a `SchedulePlan` with shape `(vehicle_count, horizon_slots)`.
5. Register the class in `methods/__init__.py`.
6. Add tests for registry discovery and plan evaluation.
7. Update README and `docs/benchmarking.md` if the method is user-facing.

## Determinism

Use explicit seeds for random behavior and avoid global random state. Benchmark outputs should be reproducible for the same family, seed, time step, and method list.

## Generated Files

Benchmark and profiling outputs belong under `results/`, which is ignored by Git. Do not commit local `.venv`, cache, coverage, or generated build directories.
