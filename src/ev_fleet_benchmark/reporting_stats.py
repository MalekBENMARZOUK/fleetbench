from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ev_fleet_benchmark.reporting_common import validate_bootstrap_samples

_logger = logging.getLogger(__name__)


def annotate_method_ranking(
    method_stats: pd.DataFrame, pairwise_df: pd.DataFrame, reference_method: str
) -> pd.DataFrame:
    annotated = method_stats.copy()
    annotated["reference_method"] = reference_method
    annotated["pareto_efficient"] = pareto_flags(annotated)
    annotated["cost_vs_reference"] = "reference"
    annotated["unmet_vs_reference"] = "reference"
    annotated["service_vs_reference"] = "reference"
    annotated["dominance_flag"] = "reference"

    if pairwise_df.empty:
        return annotated

    for row in pairwise_df.to_dict(orient="records"):
        method_mask = annotated["method"] == str(row["method"])
        cost_flag = direction_from_ci(
            float(row["cost_gap_ci_low"]), float(row["cost_gap_ci_high"]), higher_is_better=False
        )
        unmet_flag = direction_from_ci(
            float(row["unmet_gap_ci_low"]), float(row["unmet_gap_ci_high"]), higher_is_better=False
        )
        service_flag = direction_from_ci(
            float(row["service_gap_ci_low"]), float(row["service_gap_ci_high"]), higher_is_better=True
        )
        annotated.loc[method_mask, "cost_vs_reference"] = cost_flag
        annotated.loc[method_mask, "unmet_vs_reference"] = unmet_flag
        annotated.loc[method_mask, "service_vs_reference"] = service_flag
        if cost_flag == "better" and unmet_flag == "better" and service_flag == "better":
            dominance_flag = "dominates_reference"
        elif cost_flag == "worse" and unmet_flag == "worse" and service_flag == "worse":
            dominance_flag = "dominated_by_reference"
        else:
            dominance_flag = "mixed"
        annotated.loc[method_mask, "dominance_flag"] = dominance_flag
    return annotated


def direction_from_ci(lower: float, upper: float, higher_is_better: bool) -> str:
    if higher_is_better:
        if lower > 0.0:
            return "better"
        if upper < 0.0:
            return "worse"
        return "inconclusive"
    if upper < 0.0:
        return "better"
    if lower > 0.0:
        return "worse"
    return "inconclusive"


def pareto_flags(df: pd.DataFrame) -> list[bool]:
    methods = df["method"].astype(str).tolist()
    feasibility = df["feasibility_rate"].to_numpy(dtype=float)
    service = df["service_level"].to_numpy(dtype=float)
    unmet = df["unmet_charge_demand_kwh"].to_numpy(dtype=float)
    total_cost = df["total_charging_cost"].to_numpy(dtype=float)
    flags: list[bool] = []

    for row_index, _ in enumerate(methods):
        same_method = np.array([index == row_index for index in range(len(methods))], dtype=bool)
        no_worse = (
            (feasibility >= feasibility[row_index])
            & (service >= service[row_index])
            & (unmet <= unmet[row_index])
            & (total_cost <= total_cost[row_index])
        )
        strictly_better = (
            (feasibility > feasibility[row_index])
            | (service > service[row_index])
            | (unmet < unmet[row_index])
            | (total_cost < total_cost[row_index])
        )
        dominated = bool(np.any((~same_method) & no_worse & strictly_better))
        flags.append(not dominated)
    return flags


def pareto_flags_by_group(df: pd.DataFrame, group_cols: list[str]) -> list[bool]:
    positions = {index: position for position, index in enumerate(df.index.tolist())}
    flags = [False] * len(df)
    for _, group_df in df.groupby(group_cols):
        group_flags = pareto_flags(group_df)
        for group_index, group_flag in zip(group_df.index.tolist(), group_flags, strict=True):
            flags[positions[group_index]] = group_flag
    return flags


def bootstrap_group_confidence(
    summary_df: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_key, group_df in summary_df.groupby(group_cols):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)  # type: ignore[unreachable]
        row = dict(zip(group_cols, group_key, strict=True))
        for metric in metrics:
            mean_value, lower_ci, upper_ci = bootstrap_mean_ci(
                group_df[metric].to_numpy(dtype=float), bootstrap_samples
            )
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_ci_low"] = lower_ci
            row[f"{metric}_ci_high"] = upper_ci
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_comparison(
    summary_df: pd.DataFrame, reference_method: str, bootstrap_samples: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pivot_cost = summary_df.pivot_table(
        index="scenario_name", columns="method", values="total_charging_cost", aggfunc="mean"
    )
    pivot_unmet = summary_df.pivot_table(
        index="scenario_name", columns="method", values="unmet_charge_demand_kwh", aggfunc="mean"
    )
    pivot_service = summary_df.pivot_table(
        index="scenario_name", columns="method", values="service_level", aggfunc="mean"
    )

    if reference_method not in pivot_cost.columns:
        _logger.warning("Reference method '%s' not found in results; skipping pairwise comparison", reference_method)
        return pd.DataFrame(rows)

    for method in pivot_cost.columns:
        if method == reference_method:
            continue
        joint = pd.concat(
            [
                pivot_cost[[reference_method, method]].rename(
                    columns={reference_method: "ref_cost", method: "method_cost"}
                ),
                pivot_unmet[[reference_method, method]].rename(
                    columns={reference_method: "ref_unmet", method: "method_unmet"}
                ),
                pivot_service[[reference_method, method]].rename(
                    columns={reference_method: "ref_service", method: "method_service"}
                ),
            ],
            axis=1,
        ).dropna()
        if joint.empty:
            continue
        cost_diff = (joint["method_cost"] - joint["ref_cost"]).to_numpy(dtype=float)
        unmet_diff = (joint["method_unmet"] - joint["ref_unmet"]).to_numpy(dtype=float)
        service_diff = (joint["method_service"] - joint["ref_service"]).to_numpy(dtype=float)
        cost_mean, cost_low, cost_high = bootstrap_mean_ci(cost_diff, bootstrap_samples)
        unmet_mean, unmet_low, unmet_high = bootstrap_mean_ci(unmet_diff, bootstrap_samples)
        service_mean, service_low, service_high = bootstrap_mean_ci(service_diff, bootstrap_samples)
        rows.append(
            {
                "reference_method": reference_method,
                "method": method,
                "scenario_count": len(joint),
                "cost_gap_mean": cost_mean,
                "cost_gap_ci_low": cost_low,
                "cost_gap_ci_high": cost_high,
                "unmet_gap_mean": unmet_mean,
                "unmet_gap_ci_low": unmet_low,
                "unmet_gap_ci_high": unmet_high,
                "service_gap_mean": service_mean,
                "service_gap_ci_low": service_low,
                "service_gap_ci_high": service_high,
                "prob_cost_better_than_reference": float(np.mean(cost_diff < 0.0)),
                "prob_unmet_better_than_reference": float(np.mean(unmet_diff < 0.0)),
                "prob_service_better_than_reference": float(np.mean(service_diff > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, bootstrap_samples: int, seed: int = 42) -> tuple[float, float, float]:
    validate_bootstrap_samples(bootstrap_samples)
    cleaned = np.asarray(values, dtype=float)
    cleaned = cleaned[np.isfinite(cleaned)]
    if cleaned.size == 0:
        return 0.0, 0.0, 0.0
    mean_value = float(np.mean(cleaned))
    if cleaned.size == 1:
        return mean_value, mean_value, mean_value
    rng = np.random.default_rng(seed + cleaned.size)
    indices = rng.integers(0, cleaned.size, size=(bootstrap_samples, cleaned.size))
    bootstrap_means = cleaned[indices].mean(axis=1)
    lower_ci, upper_ci = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean_value, float(lower_ci), float(upper_ci)
