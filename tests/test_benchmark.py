import json
from pathlib import Path

import pandas as pd

from ev_fleet_benchmark.benchmark import BenchmarkConfig, SensitivityStudyConfig, run_benchmark, run_sensitivity_study
from ev_fleet_benchmark.telemetry import ProgressReporter


def test_benchmark_writes_expected_artifacts(tmp_path: Path) -> None:
    outputs = run_benchmark(
        BenchmarkConfig(
            family_names=["urban_depot_small"],
            seeds=[1],
            output_dir=str(tmp_path),
            generate_plots=False,
        )
    )

    assert not outputs["scenario_metrics"].empty
    assert (tmp_path / "scenario_metrics.csv").exists()
    assert (tmp_path / "aggregate_by_family.csv").exists()
    assert (tmp_path / "benchmark_report.md").exists()
    assert (tmp_path / "publication" / "publication_method_ranking.csv").exists()
    assert (tmp_path / "publication" / "publication_method_confidence.csv").exists()
    assert (tmp_path / "publication" / "publication_pairwise_comparison.csv").exists()
    assert (tmp_path / "publication" / "study_summary.md").exists()

    metadata = json.loads((tmp_path / "benchmark_metadata.json").read_text(encoding="utf-8"))
    ranking_df = pd.read_csv(tmp_path / "publication" / "publication_method_ranking.csv")
    report_text = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")

    assert metadata["family_names"] == ["urban_depot_small"]
    assert metadata["seeds"] == [1]
    assert metadata.get("methods")
    assert {"method", "rank", "reference_method", "pareto_efficient"}.issubset(ranking_df.columns)
    assert "# Benchmark Report" in report_text
    assert "## Overall Results" in report_text


def test_sensitivity_study_writes_expected_artifacts(tmp_path: Path) -> None:
    outputs = run_sensitivity_study(
        SensitivityStudyConfig(
            profile_names=["baseline", "tariff_stress"],
            family_names=["urban_depot_small"],
            seeds=[1],
            output_dir=str(tmp_path),
            bootstrap_samples=50,
        )
    )

    assert not outputs["sensitivity_scenario_metrics"].empty
    assert (tmp_path / "sensitivity_scenario_metrics.csv").exists()
    assert (tmp_path / "publication" / "sensitivity_profile_aggregate.csv").exists()
    assert (tmp_path / "publication" / "sensitivity_profile_ranking.csv").exists()
    assert (tmp_path / "publication" / "sensitivity_summary.md").exists()

    metadata = json.loads((tmp_path / "sensitivity_metadata.json").read_text(encoding="utf-8"))
    robustness_df = pd.read_csv(tmp_path / "publication" / "sensitivity_method_robustness.csv")
    summary_text = (tmp_path / "publication" / "sensitivity_summary.md").read_text(encoding="utf-8")

    assert metadata["profile_names"] == ["baseline", "tariff_stress"]
    assert metadata["family_names"] == ["urban_depot_small"]
    assert {"method", "mean_total_charging_cost", "mean_service_level"}.issubset(robustness_df.columns)
    assert "# Sensitivity Study Summary" in summary_text
    assert "## Method Robustness" in summary_text


def test_benchmark_progress_reporter_writes_jsonl_events(tmp_path: Path) -> None:
    progress_file = tmp_path / "progress" / "benchmark.jsonl"
    outputs = run_benchmark(
        BenchmarkConfig(
            family_names=["urban_depot_small"],
            seeds=[1],
            output_dir=str(tmp_path / "run"),
            generate_plots=False,
            publication_outputs=False,
        ),
        progress_reporter=ProgressReporter(jsonl_path=progress_file),
    )

    event_lines = [json.loads(line) for line in progress_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert not outputs["scenario_metrics"].empty
    assert event_lines[0]["event_type"] == "run_started"
    assert any(event["event_type"] == "method_completed" for event in event_lines)
    assert event_lines[-1]["event_type"] == "run_completed"
    assert event_lines[-1]["completed_runs"] == event_lines[-1]["total_runs"]
