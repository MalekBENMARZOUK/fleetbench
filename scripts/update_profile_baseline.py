from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_BASELINE_DIR = Path("baselines")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the retained FleetBench profiling baseline from a generated summary."
    )
    parser.add_argument("--summary", required=True, help="Path to a generated method_profile_summary.csv file.")
    parser.add_argument("--metadata", help="Optional path to the generated method_profile_metadata.json file.")
    parser.add_argument(
        "--baseline-dir", default=str(DEFAULT_BASELINE_DIR), help="Directory where the retained baseline is stored."
    )
    parser.add_argument("--label", default="manual", help="Free-form label describing why the baseline was updated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copied_files = update_profile_baseline(
        summary_path=Path(args.summary),
        metadata_path=Path(args.metadata) if args.metadata else None,
        baseline_dir=Path(args.baseline_dir),
        label=args.label,
    )
    for copied_file in copied_files:
        print(copied_file)


def update_profile_baseline(
    *,
    summary_path: Path,
    metadata_path: Path | None,
    baseline_dir: Path,
    label: str,
) -> list[Path]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Profiling summary not found: {summary_path}")

    baseline_dir.mkdir(parents=True, exist_ok=True)
    copied_files: list[Path] = []

    destination_summary = baseline_dir / "method_profile_summary.csv"
    shutil.copy2(summary_path, destination_summary)
    copied_files.append(destination_summary)

    manifest = {
        "label": label,
        "summary_source": str(summary_path),
    }

    if metadata_path is not None:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Profiling metadata not found: {metadata_path}")
        destination_metadata = baseline_dir / "method_profile_metadata.json"
        shutil.copy2(metadata_path, destination_metadata)
        copied_files.append(destination_metadata)
        manifest["metadata_source"] = str(metadata_path)

    manifest_path = baseline_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    copied_files.append(manifest_path)
    return copied_files


if __name__ == "__main__":
    main()
