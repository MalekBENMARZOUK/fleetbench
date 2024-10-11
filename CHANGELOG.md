# Changelog

All notable changes to FleetBench are documented here.

The project follows semantic versioning while it is used as a demo and testing benchmark.

## 0.1.0 - 2024-12-16

### Added

- End-to-end EV fleet charging benchmark workflow.
- Four synthetic scenario families with tariff, site derate, delay, and priority variation.
- Six scheduling methods: naive, greedy, deterministic optimization, rolling horizon, scenario tree, and stochastic anticipatory.
- Simulation metrics for service level, energy shortfall, cost, peak demand, feasibility, and runtime.
- CLI commands for benchmark runs, larger studies, sensitivity studies, method discovery, and scenario discovery.
- CSV, Markdown, plot, and publication-style reporting outputs.
- Structured progress telemetry with optional JSONL output.
- Profiling scripts and optional regression baseline comparison.
- Strict pytest, Ruff, mypy, package build, Docker, CI, and release workflows.

### Fixed

- Source-layout pytest discovery for uninstalled local checkouts.
- Fallback package version lookup when running directly from source.

### Documentation

- Added architecture, benchmarking, output schema, development, and release guides.
- Added contribution, security, citation, and quickstart example materials.
