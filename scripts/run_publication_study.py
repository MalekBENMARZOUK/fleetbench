from __future__ import annotations

from ev_fleet_benchmark.benchmark import BenchmarkConfig, run_benchmark
from ev_fleet_benchmark.methods import build_methods
from ev_fleet_benchmark.scenarios import describe_families
from ev_fleet_benchmark.telemetry import ProgressReporter, configure_cli_logger


def main() -> None:
    families = [entry["name"] for entry in describe_families()]
    seeds = list(range(101, 125))
    progress_reporter = ProgressReporter(
        logger=configure_cli_logger("info"),
        jsonl_path=None,
    )
    run_benchmark(
        BenchmarkConfig(
            family_names=families,
            seeds=seeds,
            output_dir="results/publication_study",
            generate_plots=False,
            publication_outputs=True,
            bootstrap_samples=2000,
        ),
        methods=build_methods(),
        progress_reporter=progress_reporter,
    )


if __name__ == "__main__":
    main()
