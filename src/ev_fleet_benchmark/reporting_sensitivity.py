from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path
from warnings import warn

import pandas as pd

from ev_fleet_benchmark.reporting_common import (
    BASELINE_SENSITIVITY_PROFILE,
    require_columns,
    validate_bootstrap_samples,
)
from ev_fleet_benchmark.reporting_stats import pareto_flags_by_group


class SensitivityTables(TypedDict):
    sensitivity_profile_aggregate: pd.DataFrame
    sensitivity_profile_ranking: pd.DataFrame
    sensitivity_method_robustness: pd.DataFrame
    sensitivity_profile_deltas: pd.DataFrame


def create_sensitivity_tables(
    summary_df: pd.DataFrame, output_dir: Path, bootstrap_samples: int = 1000
) -> SensitivityTables:
    validate_bootstrap_samples(bootstrap_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        empty = pd.DataFrame()
        return {
            "sensitivity_profile_aggregate": empty,
            "sensitivity_profile_ranking": empty,
            "sensitivity_method_robustness": empty,
            "sensitivity_profile_deltas": empty,
        }

    require_columns(
        summary_df,
        [
            "sensitivity_profile",
            "method",
            "scenario_name",
            "feasibility_rate",
            "service_level",
            "unmet_charge_demand_kwh",
            "demand_charge_cost",
            "battery_degradation_cost",
            "total_charging_cost",
            "solve_time_s",
        ],
    )

    profile_aggregate = (
        summary_df.groupby(["sensitivity_profile", "method"], as_index=False)
        .agg(
            scenarios=("scenario_name", "count"),
            feasibility_rate=("feasibility_rate", "mean"),
            service_level=("service_level", "mean"),
            unmet_charge_demand_kwh=("unmet_charge_demand_kwh", "mean"),
            demand_charge_cost=("demand_charge_cost", "mean"),
            battery_degradation_cost=("battery_degradation_cost", "mean"),
            total_charging_cost=("total_charging_cost", "mean"),
            solve_time_s=("solve_time_s", "mean"),
        )
        .sort_values(["sensitivity_profile", "method"])
    )
    profile_ranking = profile_aggregate.sort_values(
        [
            "sensitivity_profile",
            "feasibility_rate",
            "service_level",
            "unmet_charge_demand_kwh",
            "total_charging_cost",
        ],
        ascending=[True, False, False, True, True],
    ).copy()
    profile_ranking["rank_within_profile"] = profile_ranking.groupby("sensitivity_profile").cumcount() + 1
    profile_ranking["pareto_efficient"] = pareto_flags_by_group(profile_ranking, ["sensitivity_profile"])

    method_robustness = (
        profile_aggregate.groupby("method", as_index=False)
        .agg(
            profiles=("sensitivity_profile", "count"),
            mean_total_charging_cost=("total_charging_cost", "mean"),
            std_total_charging_cost=("total_charging_cost", "std"),
            worst_total_charging_cost=("total_charging_cost", "max"),
            mean_unmet_charge_demand_kwh=("unmet_charge_demand_kwh", "mean"),
            worst_unmet_charge_demand_kwh=("unmet_charge_demand_kwh", "max"),
            mean_service_level=("service_level", "mean"),
            worst_service_level=("service_level", "min"),
            mean_feasibility_rate=("feasibility_rate", "mean"),
        )
        .sort_values(
            ["mean_feasibility_rate", "mean_service_level", "mean_total_charging_cost"],
            ascending=[False, False, True],
        )
    )

    baseline_df = summary_df[summary_df["sensitivity_profile"] == BASELINE_SENSITIVITY_PROFILE]
    if baseline_df.empty:
        warn(
            (
                f"Sensitivity profile '{BASELINE_SENSITIVITY_PROFILE}' not found in results; "
                "delta calculations will be skipped."
            ),
            stacklevel=2,
        )
        profile_deltas = pd.DataFrame()
    else:
        require_columns(summary_df, ["base_scenario_name"])
        baseline_keys = set(zip(baseline_df["base_scenario_name"], baseline_df["method"], strict=True))
        non_baseline = summary_df[summary_df["sensitivity_profile"] != BASELINE_SENSITIVITY_PROFILE]
        required_keys = set(zip(non_baseline["base_scenario_name"], non_baseline["method"], strict=True))
        missing_keys = required_keys - baseline_keys
        if missing_keys:
            warn(
                f"Missing baseline records for {len(missing_keys)} (scenario, method) pairs; "
                "delta rows for these pairs will be dropped.",
                stacklevel=2,
            )
        merged = summary_df.merge(
            baseline_df[
                [
                    "base_scenario_name",
                    "method",
                    "total_charging_cost",
                    "unmet_charge_demand_kwh",
                    "service_level",
                ]
            ].rename(
                columns={
                    "total_charging_cost": "baseline_total_charging_cost",
                    "unmet_charge_demand_kwh": "baseline_unmet_charge_demand_kwh",
                    "service_level": "baseline_service_level",
                }
            ),
            on=["base_scenario_name", "method"],
            how="left",
        )
        merged = merged[merged["sensitivity_profile"] != BASELINE_SENSITIVITY_PROFILE].copy()
        merged = merged.dropna(subset=["baseline_total_charging_cost"])
        merged["delta_total_charging_cost"] = merged["total_charging_cost"] - merged["baseline_total_charging_cost"]
        merged["delta_unmet_charge_demand_kwh"] = (
            merged["unmet_charge_demand_kwh"] - merged["baseline_unmet_charge_demand_kwh"]
        )
        merged["delta_service_level"] = merged["service_level"] - merged["baseline_service_level"]
        profile_deltas = (
            merged.groupby(["sensitivity_profile", "method"], as_index=False)
            .agg(
                delta_total_charging_cost=("delta_total_charging_cost", "mean"),
                delta_unmet_charge_demand_kwh=("delta_unmet_charge_demand_kwh", "mean"),
                delta_service_level=("delta_service_level", "mean"),
            )
            .sort_values(["sensitivity_profile", "method"])
        )

    profile_aggregate.to_csv(output_dir / "sensitivity_profile_aggregate.csv", index=False)
    profile_ranking.to_csv(output_dir / "sensitivity_profile_ranking.csv", index=False)
    method_robustness.to_csv(output_dir / "sensitivity_method_robustness.csv", index=False)
    profile_deltas.to_csv(output_dir / "sensitivity_profile_deltas.csv", index=False)
    return {
        "sensitivity_profile_aggregate": profile_aggregate,
        "sensitivity_profile_ranking": profile_ranking,
        "sensitivity_method_robustness": method_robustness,
        "sensitivity_profile_deltas": profile_deltas,
    }
