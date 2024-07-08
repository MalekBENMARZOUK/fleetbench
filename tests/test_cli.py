from pathlib import Path

from typer.testing import CliRunner

from ev_fleet_benchmark.cli import app

runner = CliRunner()


def test_list_methods_command_lists_registry_names() -> None:
    result = runner.invoke(app, ["list-methods"])

    assert result.exit_code == 0
    assert "optimization_ortools" in result.stdout
    assert "scenario_tree_ortools" in result.stdout


def test_benchmark_command_rejects_invalid_time_step() -> None:
    result = runner.invoke(
        app,
        ["benchmark", "--output-dir", "unused", "--seeds", "1", "--time-step-minutes", "17", "--no-plots"],
    )

    assert result.exit_code != 0
    assert "divide evenly into 24 hours" in result.stderr


def test_benchmark_command_rejects_unknown_method(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--output-dir",
            str(tmp_path),
            "--seeds",
            "1",
            "--method",
            "not_a_method",
            "--no-plots",
        ],
    )

    assert result.exit_code == 2
    assert "Unknown method names requested" in result.stderr


def test_study_command_rejects_non_positive_seed_count() -> None:
    result = runner.invoke(app, ["study", "--seed-count", "0"])

    assert result.exit_code != 0
    assert "must be a positive integer" in result.stderr


def test_benchmark_command_writes_progress_file(tmp_path: Path) -> None:
    progress_file = tmp_path / "progress.jsonl"

    result = runner.invoke(
        app,
        [
            "benchmark",
            "--output-dir",
            str(tmp_path / "results"),
            "--seeds",
            "1",
            "--method",
            "naive_baseline",
            "--no-plots",
            "--no-publication",
            "--progress-file",
            str(progress_file),
        ],
    )

    assert result.exit_code == 0
    assert progress_file.exists()
    assert '"event_type": "run_started"' in progress_file.read_text(encoding="utf-8")


def test_benchmark_command_accepts_explicit_log_level(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--output-dir",
            str(tmp_path / "results"),
            "--seeds",
            "1",
            "--method",
            "naive_baseline",
            "--no-plots",
            "--no-publication",
            "--log-level",
            "info",
        ],
    )

    assert result.exit_code == 0


def test_benchmark_command_rejects_invalid_log_level() -> None:
    result = runner.invoke(
        app, ["benchmark", "--output-dir", "unused", "--seeds", "1", "--log-level", "trace", "--no-plots"]
    )

    assert result.exit_code != 0
    assert "must be one of" in result.stderr
