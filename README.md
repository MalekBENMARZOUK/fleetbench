# FleetBench

FleetBench is a reproducible benchmark suite for electric vehicle depot charging. It generates realistic fleet scenarios, runs scheduling methods under identical seeded conditions, and writes artifacts that are useful for engineering reviews, operations research experiments, and demo environments.

The project is intentionally self-contained: scenario generation, scheduling methods, simulation, reporting, CLI workflows, tests, CI, Docker packaging, and documentation all live in this repository.

## Highlights

- Simulates EV depot charging with site capacity limits, arrival uncertainty, tariff windows, demand charges, and battery wear penalties.
- Benchmarks heuristic, stochastic, and OR-Tools optimization methods on the same seeded scenarios.
- Reports service level, energy shortfall, feasibility, cost, peak demand, and runtime.
- Produces CSV metrics, Markdown summaries, publication-style tables, plots, and structured progress logs.
- Supports larger benchmark studies, sensitivity studies, and profiling baselines.
- Ships with pytest coverage, Ruff, mypy strict mode, GitHub Actions, Docker, and release workflow scaffolding.

## Methods

| Method | Type | Purpose |
|---|---|---|
| `naive_baseline` | Heuristic | First-come, first-served reference point. |
| `greedy_urgency` | Heuristic | Prioritizes departures, unmet energy, and operational priority. |
| `optimization_ortools` | MIP | Solves a deterministic single-scenario charging model. |
| `rolling_horizon_ortools` | MIP | Re-optimizes repeatedly as the horizon advances. |
| `scenario_tree_ortools` | MIP | Uses uncertainty branches for anticipatory planning. |
| `stochastic_anticipatory` | Stochastic heuristic | Approximates uncertainty-aware planning without a full MIP tree. |

## Scenario Families

FleetBench includes four built-in scenario families:

- `urban_depot_small`: small depot, moderate constraints, commuter-style departures.
- `regional_mixed_medium`: medium heterogeneous fleet with tighter charging windows.
- `capacity_stressed_peak`: capacity-constrained depot under sharp tariff peaks.
- `uncertain_operations_large`: large fleet with higher arrival uncertainty and mixed urgency.

Each family varies fleet size, capacity tightness, delay probability, site derates, dwell times, target state of charge, priority mix, and tariff shape.

## Quick Start

FleetBench supports Python 3.11 through 3.14.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Run a small benchmark:

```bash
fleetbench benchmark \
  --output-dir results/latest \
  --family urban_depot_small \
  --method naive_baseline \
  --method greedy_urgency \
  --seeds 101 \
  --no-plots
```

Inspect available inputs:

```bash
fleetbench describe-families
fleetbench list-methods
fleetbench list-sensitivity
```

Run a sensitivity study:

```bash
fleetbench sensitivity \
  --output-dir results/sensitivity \
  --start-seed 101 \
  --seed-count 3 \
  --profile baseline \
  --profile tariff_stress \
  --method naive_baseline \
  --method optimization_ortools \
  --bootstrap-samples 200
```

## Outputs

A benchmark run writes:

- `scenario_metrics.csv`: one row per scenario and method.
- `vehicle_metrics.csv`: per-vehicle charging and service outcomes.
- `site_load_profiles.csv`: site load, capacity, and tariff by time slot.
- `aggregate_by_family.csv` and `aggregate_overall.csv`: summary comparisons.
- `benchmark_report.md`: human-readable benchmark summary.
- `publication/`: ranking, confidence interval, family winner, and pairwise comparison tables.
- `plots/`: generated visualizations when plots are enabled.
- `benchmark_metadata.json`: reproducibility metadata.

See [docs/output-schema.md](docs/output-schema.md) for column-level guidance.

## Development

Run the local quality gate:

```bash
python -m pytest
python -m ruff check src tests scripts examples
python -m ruff format --check src tests scripts examples
python -m mypy
```

Profile methods and compare against a retained baseline:

```bash
python scripts/profile_methods.py \
  --output-dir results/profiling \
  --family urban_depot_small \
  --seed 101 --seed 102 --seed 103 \
  --baseline-summary baselines/method_profile_summary.csv
```

Project documentation:

- [Architecture](docs/architecture.md)
- [Benchmarking Guide](docs/benchmarking.md)
- [Output Schema](docs/output-schema.md)
- [Development Guide](docs/development.md)
- [Release Process](docs/release.md)

## Docker

The Dockerfile builds a Python 3.12 runtime image.

```bash
docker build -t fleetbench .
docker run --rm -v "$(pwd)/results:/app/results" fleetbench \
  benchmark --output-dir /app/results --family urban_depot_small --seeds 101 --no-plots
```

PowerShell:

```powershell
docker build -t fleetbench .
docker run --rm -v "${PWD}/results:/app/results" fleetbench `
  benchmark --output-dir /app/results --family urban_depot_small --seeds 101 --no-plots
```

## Status

FleetBench is a realistic synthetic benchmark project intended for demos, testing, and reproducible experimentation. It is not calibrated against a specific fleet operator unless you add local calibration data.

## License

FleetBench is released under the MIT License. See [LICENSE](LICENSE).
