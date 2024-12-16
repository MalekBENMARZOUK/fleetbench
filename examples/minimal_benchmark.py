from __future__ import annotations

from pathlib import Path

from ev_fleet_benchmark import BenchmarkConfig, run_benchmark
from ev_fleet_benchmark.methods import build_methods


def main() -> None:
    output_dir = Path("results/example-minimal")
    outputs = run_benchmark(
        BenchmarkConfig(
            family_names=["urban_depot_small"],
            seeds=[101],
            output_dir=str(output_dir),
            generate_plots=False,
            publication_outputs=False,
        ),
        methods=build_methods(["naive_baseline", "greedy_urgency"]),
    )
    scenario_metrics = outputs["scenario_metrics"]
    print(scenario_metrics[["scenario_name", "method", "service_level", "total_charging_cost"]])
    print(f"Wrote example outputs to {output_dir}")


if __name__ == "__main__":
    main()
