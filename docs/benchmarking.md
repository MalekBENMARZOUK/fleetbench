# Benchmarking Guide

This guide describes the common benchmark workflows used by FleetBench.

## Small Smoke Run

Use a smoke run while developing a method or validating a checkout:

```bash
fleetbench benchmark \
  --output-dir results/smoke \
  --family urban_depot_small \
  --method naive_baseline \
  --method greedy_urgency \
  --seeds 101 \
  --no-plots \
  --no-publication
```

## Method Comparison

Compare all registered methods on several seeds:

```bash
fleetbench benchmark \
  --output-dir results/method-comparison \
  --family urban_depot_small \
  --family capacity_stressed_peak \
  --seeds 101 \
  --seeds 102 \
  --seeds 103 \
  --plots
```

## Larger Study

The `study` command evaluates all built-in families over a consecutive seed range:

```bash
fleetbench study \
  --output-dir results/study \
  --start-seed 101 \
  --seed-count 24 \
  --bootstrap-samples 2000
```

## Sensitivity Study

Sensitivity profiles perturb tariff and battery wear assumptions:

```bash
fleetbench sensitivity \
  --output-dir results/sensitivity \
  --start-seed 101 \
  --seed-count 12 \
  --profile baseline \
  --profile tariff_stress \
  --profile wear_stress \
  --bootstrap-samples 1000
```

## Parallel Execution

Use `--workers` to run methods for each scenario in parallel:

```bash
fleetbench benchmark \
  --output-dir results/parallel \
  --family regional_mixed_medium \
  --seeds 101 \
  --seeds 102 \
  --workers 4
```

Parallel execution is most helpful when OR-Tools methods dominate runtime. Output row ordering remains deterministic by method selection order.

## Progress Telemetry

Structured progress can be emitted to stderr, JSONL, or both:

```bash
fleetbench benchmark \
  --output-dir results/observed \
  --family urban_depot_small \
  --method naive_baseline \
  --seeds 101 \
  --log-level info \
  --progress-file results/observed/progress.jsonl
```

Each JSONL row includes event type, run kind, elapsed seconds, total run count, completed run count, and optional scenario or method context.
