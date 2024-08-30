import json
from pathlib import Path

from scripts.update_profile_baseline import update_profile_baseline


def test_update_profile_baseline_copies_summary_and_metadata(tmp_path: Path) -> None:
    summary_path = tmp_path / "results" / "method_profile_summary.csv"
    metadata_path = tmp_path / "results" / "method_profile_metadata.json"
    baseline_dir = tmp_path / "baselines"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("method,mean_planner_wall_time_s\nnaive_baseline,1.0\n", encoding="utf-8")
    metadata_path.write_text(json.dumps({"families": ["urban_depot_small"]}), encoding="utf-8")

    copied_files = update_profile_baseline(
        summary_path=summary_path,
        metadata_path=metadata_path,
        baseline_dir=baseline_dir,
        label="approved_baseline",
    )

    assert baseline_dir / "method_profile_summary.csv" in copied_files
    assert baseline_dir / "method_profile_metadata.json" in copied_files
    manifest = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["label"] == "approved_baseline"
