from __future__ import annotations

from pathlib import Path

from ev_fleet_benchmark import BenchmarkConfig, run_benchmark
from ev_fleet_benchmark.methods import build_methods
from ev_fleet_benchmark.telemetry import ProgressReporter


def main() -> None:
    output_dir = Path("results/example-progress")
    progress_file = output_dir / "progress.jsonl"
    reporter = ProgressReporter(jsonl_path=progress_file, text_writer=print)
    run_benchmark(
        BenchmarkConfig(
            family_names=["urban_depot_small"],
            seeds=[101],
            output_dir=str(output_dir),
            generate_plots=False,
            publication_outputs=False,
        ),
        methods=build_methods(["naive_baseline"]),
        progress_reporter=reporter,
    )
    print(f"Wrote progress events to {progress_file}")


if __name__ == "__main__":
    main()
