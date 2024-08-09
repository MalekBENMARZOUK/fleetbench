from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

from ev_fleet_benchmark.methods import build_methods, method_names
from ev_fleet_benchmark.scenarios import default_scenario_families, generate_scenario
from ev_fleet_benchmark.simulator import evaluate_plan


@dataclass(frozen=True)
class RegressionThresholds:
    runtime_ratio: float = 1.1
    cost_ratio: float = 1.05
    unmet_ratio: float = 1.05
    service_level_drop: float = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile FleetBench methods over selected families and seeds.")
    parser.add_argument(
        "--output-dir", default="results/profiling", help="Directory where profiling outputs are written."
    )
    parser.add_argument(
        "--family", action="append", dest="families", help="Scenario family to include. Repeat to select multiple."
    )
    parser.add_argument(
        "--method", action="append", dest="methods", help="Method to include. Repeat to select multiple."
    )
    parser.add_argument(
        "--seed", action="append", dest="seeds", type=int, help="Random seed to include. Repeat to select multiple."
    )
    parser.add_argument("--time-step-minutes", default=30, type=int, help="Simulation time-step in minutes.")
    parser.add_argument("--baseline-summary", help="Optional prior method_profile_summary.csv to compare against.")
    parser.add_argument(
        "--runtime-regression-ratio",
        default=1.1,
        type=float,
        help="Flag runtime regressions when current mean wall time exceeds this ratio versus baseline.",
    )
    parser.add_argument(
        "--cost-regression-ratio",
        default=1.05,
        type=float,
        help="Flag cost regressions when current mean total cost exceeds this ratio versus baseline.",
    )
    parser.add_argument(
        "--unmet-regression-ratio",
        default=1.05,
        type=float,
        help="Flag unmet-energy regressions when current mean unmet energy exceeds this ratio versus baseline.",
    )
    parser.add_argument(
        "--service-level-drop",
        default=0.01,
        type=float,
        help="Flag service-level regressions when current mean service level drops by more than this amount.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 2 when baseline comparison finds regressions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_families = args.families or sorted(default_scenario_families())
    selected_seeds = args.seeds or [101, 102, 103]
    selected_methods = build_methods(args.methods or method_names())

    raw_df, summary_df = profile_methods(
        family_names=selected_families,
        seeds=selected_seeds,
        method_names_to_build=[method.name for method in selected_methods],
        time_step_minutes=args.time_step_minutes,
    )

    raw_df.to_csv(output_dir / "method_profile_runs.csv", index=False)
    summary_df.to_csv(output_dir / "method_profile_summary.csv", index=False)
    metadata = {
        "families": selected_families,
        "methods": [method.name for method in selected_methods],
        "seeds": selected_seeds,
        "time_step_minutes": args.time_step_minutes,
        "baseline_summary": args.baseline_summary,
        "thresholds": {
            "runtime_ratio": args.runtime_regression_ratio,
            "cost_ratio": args.cost_regression_ratio,
            "unmet_ratio": args.unmet_regression_ratio,
            "service_level_drop": args.service_level_drop,
        },
    }
    (output_dir / "method_profile_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    comparison_df = pd.DataFrame()
    if args.baseline_summary:
        baseline_path = Path(args.baseline_summary).resolve()
        if not baseline_path.is_file():
            print(f"ERROR: Baseline summary not found: {baseline_path}", file=sys.stderr)
            raise SystemExit(1)
        thresholds = RegressionThresholds(
            runtime_ratio=args.runtime_regression_ratio,
            cost_ratio=args.cost_regression_ratio,
            unmet_ratio=args.unmet_regression_ratio,
            service_level_drop=args.service_level_drop,
        )
        baseline_df = pd.read_csv(baseline_path)
        comparison_df = compare_against_baseline(summary_df, baseline_df, thresholds)
        comparison_df.to_csv(output_dir / "method_profile_comparison.csv", index=False)

    print(summary_df.to_string(index=False))
    if not comparison_df.empty:
        print()
        print(comparison_df.to_string(index=False))
        if args.fail_on_regression:
            regression_messages = summarize_comparison_issues(comparison_df)
            if regression_messages:
                for message in regression_messages:
                    print(f"REGRESSION: {message}", file=sys.stderr)
                raise SystemExit(2)


def profile_methods(
    *,
    family_names: list[str],
    seeds: list[int],
    method_names_to_build: list[str],
    time_step_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_methods = build_methods(method_names_to_build)
    rows: list[dict[str, object]] = []
    for family_name in family_names:
        for seed in seeds:
            scenario = generate_scenario(family_name=family_name, seed=seed, time_step_minutes=time_step_minutes)
            for method in selected_methods:
                wall_start = perf_counter()
                plan = method.solve(scenario)
                wall_s = perf_counter() - wall_start
                evaluation = evaluate_plan(scenario, plan)
                rows.append(
                    {
                        "family": family_name,
                        "seed": seed,
                        "scenario_name": scenario.name,
                        "method": method.name,
                        "planner_wall_time_s": wall_s,
                        "reported_solve_time_s": float(evaluation.summary["solve_time_s"]),
                        "total_charging_cost": float(evaluation.summary["total_charging_cost"]),
                        "unmet_charge_demand_kwh": float(evaluation.summary["unmet_charge_demand_kwh"]),
                        "service_level": float(evaluation.summary["service_level"]),
                        "feasibility_rate": float(evaluation.summary["feasibility_rate"]),
                    }
                )

    raw_df = pd.DataFrame(rows).sort_values(["family", "seed", "method"])
    summary_df = (
        raw_df.groupby("method", as_index=False)
        .agg(
            runs=("scenario_name", "count"),
            mean_planner_wall_time_s=("planner_wall_time_s", "mean"),
            p95_planner_wall_time_s=("planner_wall_time_s", lambda values: float(pd.Series(values).quantile(0.95))),
            mean_reported_solve_time_s=("reported_solve_time_s", "mean"),
            mean_total_charging_cost=("total_charging_cost", "mean"),
            mean_unmet_charge_demand_kwh=("unmet_charge_demand_kwh", "mean"),
            mean_service_level=("service_level", "mean"),
            mean_feasibility_rate=("feasibility_rate", "mean"),
        )
        .sort_values(["mean_planner_wall_time_s", "mean_total_charging_cost"])
    )
    return raw_df, summary_df


def compare_against_baseline(
    current_summary_df: pd.DataFrame,
    baseline_summary_df: pd.DataFrame,
    thresholds: RegressionThresholds,
) -> pd.DataFrame:
    merged = current_summary_df.merge(
        baseline_summary_df,
        on="method",
        how="outer",
        suffixes=("_current", "_baseline"),
        indicator=True,
    )

    rows: list[dict[str, object]] = []
    for row in merged.to_dict(orient="records"):
        merge_state = str(row["_merge"])
        method_name = str(row["method"])
        if merge_state == "left_only":
            rows.append({"method": method_name, "comparison_status": "new_method"})
            continue
        if merge_state == "right_only":
            rows.append({"method": method_name, "comparison_status": "missing_from_current"})
            continue

        current_runtime = float(row["mean_planner_wall_time_s_current"])
        baseline_runtime = float(row["mean_planner_wall_time_s_baseline"])
        current_cost = float(row["mean_total_charging_cost_current"])
        baseline_cost = float(row["mean_total_charging_cost_baseline"])
        current_unmet = float(row["mean_unmet_charge_demand_kwh_current"])
        baseline_unmet = float(row["mean_unmet_charge_demand_kwh_baseline"])
        current_service = float(row["mean_service_level_current"])
        baseline_service = float(row["mean_service_level_baseline"])

        runtime_ratio = (
            current_runtime / baseline_runtime
            if baseline_runtime > 1e-9
            else (1.0 if current_runtime <= 1e-9 else float("inf"))
        )
        cost_ratio = (
            current_cost / baseline_cost if baseline_cost > 1e-9 else (1.0 if current_cost <= 1e-9 else float("inf"))
        )
        unmet_ratio = (
            current_unmet / baseline_unmet
            if baseline_unmet > 1e-9
            else (1.0 if current_unmet <= 1e-9 else float("inf"))
        )
        service_level_delta = current_service - baseline_service

        regression_reasons: list[str] = []
        if runtime_ratio > thresholds.runtime_ratio:
            regression_reasons.append("runtime")
        if cost_ratio > thresholds.cost_ratio:
            regression_reasons.append("cost")
        if unmet_ratio > thresholds.unmet_ratio:
            regression_reasons.append("unmet")
        if service_level_delta < -thresholds.service_level_drop:
            regression_reasons.append("service_level")

        rows.append(
            {
                "method": method_name,
                "comparison_status": "regression" if regression_reasons else "ok",
                "regression_reasons": ",".join(regression_reasons),
                "runtime_ratio": runtime_ratio,
                "cost_ratio": cost_ratio,
                "unmet_ratio": unmet_ratio,
                "service_level_delta": service_level_delta,
            }
        )

    return pd.DataFrame(rows).sort_values(["comparison_status", "method"])


def summarize_comparison_issues(comparison_df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    for row in comparison_df.to_dict(orient="records"):
        status = str(row.get("comparison_status", ""))
        method_name = str(row.get("method", "unknown"))
        if status == "regression":
            reasons = str(row.get("regression_reasons", ""))
            issues.append(f"{method_name} exceeded baseline thresholds: {reasons}")
    return issues


if __name__ == "__main__":
    main()
