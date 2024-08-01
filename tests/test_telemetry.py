import io
import json
from pathlib import Path

from ev_fleet_benchmark.telemetry import ProgressEvent, ProgressReporter, format_progress_event


def test_format_progress_event_includes_human_readable_fields() -> None:
    event = ProgressEvent(
        event_type="method_completed",
        run_kind="benchmark",
        completed_runs=2,
        total_runs=5,
        output_dir="results/run",
        elapsed_s=1.25,
        family_name="urban_depot_small",
        scenario_name="urban_depot_small_seed_1",
        method_name="naive_baseline",
        seed=1,
        solve_time_s=0.01,
        status="completed",
    )

    rendered = format_progress_event(event)

    assert "[benchmark] 2/5 method_completed" in rendered
    assert "family=urban_depot_small" in rendered
    assert "method=naive_baseline" in rendered
    assert "solve=0.0100s" in rendered


def test_progress_reporter_writes_jsonl_and_text(tmp_path: Path) -> None:
    text_buffer = io.StringIO()
    progress_path = tmp_path / "progress.jsonl"

    def write_text(message: str) -> None:
        text_buffer.write(message + "\n")

    reporter = ProgressReporter(jsonl_path=progress_path, text_writer=write_text)
    event = ProgressEvent(
        event_type="run_started",
        run_kind="study",
        completed_runs=0,
        total_runs=4,
        output_dir="results/study",
        elapsed_s=0.0,
    )

    reporter.emit(event)

    stored_event = json.loads(progress_path.read_text(encoding="utf-8").strip())
    assert stored_event["event_type"] == "run_started"
    assert "[study] 0/4 run_started" in text_buffer.getvalue()
