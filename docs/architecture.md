# Architecture

FleetBench is organized as a small pipeline:

1. Scenario families define operational assumptions.
2. A seeded generator expands a family into a concrete scenario.
3. Scheduling methods produce a power dispatch matrix.
4. The simulator validates and evaluates the dispatch.
5. Reporting modules aggregate metrics and write artifacts.
6. CLI commands connect the pieces for repeatable runs.

## Core Modules

| Module | Responsibility |
|---|---|
| `model.py` | Dataclasses for vehicles, scenarios, plans, and evaluation results. |
| `scenarios.py` | Built-in scenario families, seeded scenario generation, and sensitivity profiles. |
| `methods/` | Scheduling method implementations and method registry. |
| `simulator.py` | Feasibility checks and metric computation for a submitted plan. |
| `economics.py` | Charging cost, demand charge, and battery wear helpers. |
| `benchmark.py` | Benchmark and sensitivity orchestration. |
| `reporting*.py` | Aggregation, statistics, plots, Markdown, and publication tables. |
| `telemetry.py` | Structured progress events for logs and JSONL streams. |
| `cli.py` | Typer-based command line interface. |

## Data Flow

```text
ScenarioFamily + seed
        |
        v
Scenario
        |
        v
ScheduleMethod.solve()
        |
        v
SchedulePlan
        |
        v
evaluate_plan()
        |
        v
CSV metrics + Markdown + plots + publication tables
```

## Reproducibility

The scenario generator uses NumPy random generators seeded from CLI input. Each generated scenario is serialized under the run output directory so results can be inspected and replayed. Benchmark metadata records families, seeds, methods, time step, and bootstrap sample counts.

## Method Registry

Methods inherit from `ScheduleMethod` and expose a unique `name`. The registry in `methods/__init__.py` builds method instances from CLI names and detects missing OR-Tools support with a clear error. This keeps CLI selection, tests, and benchmark orchestration aligned.

## Error Handling

Domain errors are wrapped in `FleetBenchError` subclasses where possible. CLI commands report these errors to stderr and exit with code `2`, which keeps batch and CI behavior predictable.
