from __future__ import annotations

from ev_fleet_benchmark.benchmark import SensitivityStudyConfig, run_sensitivity_study
from ev_fleet_benchmark.methods import build_methods
from ev_fleet_benchmark.scenarios import describe_families, describe_sensitivity_profiles
from ev_fleet_benchmark.telemetry import ProgressReporter, configure_cli_logger


def main() -> None:
    families = [entry["name"] for entry in describe_families()]
    profiles = [entry["name"] for entry in describe_sensitivity_profiles()]
    seeds = list(range(101, 113))
    progress_reporter = ProgressReporter(
        logger=configure_cli_logger("info"),
        jsonl_path=None,
    )
    run_sensitivity_study(
        SensitivityStudyConfig(
            profile_names=profiles,
            family_names=families,
            seeds=seeds,
            output_dir="results/sensitivity_study",
            bootstrap_samples=1000,
        ),
        methods=build_methods(),
        progress_reporter=progress_reporter,
    )


if __name__ == "__main__":
    main()
