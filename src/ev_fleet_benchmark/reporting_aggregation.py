from __future__ import annotations

import pandas as pd

from ev_fleet_benchmark.reporting_common import require_columns

SUMMARY_METRICS = {
    "scenarios": ("scenario_name", "count"),
    "feasibility_rate": ("feasibility_rate", "mean"),
    "unmet_charge_demand_kwh": ("unmet_charge_demand_kwh", "mean"),
    "energy_cost": ("energy_cost", "mean"),
    "all_day_demand_charge_cost": ("all_day_demand_charge_cost", "mean"),
    "onpeak_demand_charge_cost": ("onpeak_demand_charge_cost", "mean"),
    "demand_charge_cost": ("demand_charge_cost", "mean"),
    "battery_degradation_cost": ("battery_degradation_cost", "mean"),
    "total_charging_cost": ("total_charging_cost", "mean"),
    "peak_site_power_kw": ("peak_site_power_kw", "mean"),
    "load_variance_kw2": ("load_variance_kw2", "mean"),
    "solve_time_s": ("solve_time_s", "mean"),
    "service_level": ("service_level", "mean"),
}


def aggregate_results(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df.empty:
        empty = pd.DataFrame()
        return empty, empty

    require_columns(summary_df, ["family", "method", *[value[0] for value in SUMMARY_METRICS.values()]])

    family_aggregate = (
        summary_df.groupby(["family", "method"], as_index=False)
        .agg(**SUMMARY_METRICS)
        .sort_values(["family", "method"])
    )
    overall_aggregate = summary_df.groupby(["method"], as_index=False).agg(**SUMMARY_METRICS).sort_values(["method"])
    return family_aggregate, overall_aggregate
