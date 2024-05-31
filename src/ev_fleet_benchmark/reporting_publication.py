from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path
from warnings import warn

import pandas as pd

from ev_fleet_benchmark.reporting_common import (
    DEFAULT_PUBLICATION_REFERENCE_METHOD,
    require_columns,
    validate_bootstrap_samples,
)
from ev_fleet_benchmark.reporting_stats import (
    annotate_method_ranking,
    bootstrap_group_confidence,
    paired_bootstrap_comparison,
)


class PublicationTables(TypedDict):
    method_ranking: pd.DataFrame
    family_winners: pd.DataFrame
    method_confidence: pd.DataFrame
    family_confidence: pd.DataFrame
    pairwise_comparison: pd.DataFrame


def create_publication_tables(
    summary_df: pd.DataFrame, output_dir: Path, bootstrap_samples: int = 1000
) -> PublicationTables:
    validate_bootstrap_samples(bootstrap_samples)
    if summary_df.empty:
        empty = pd.DataFrame()
        return {
            "method_ranking": empty,
            "family_winners": empty,
            "method_confidence": empty,
            "family_confidence": empty,
            "pairwise_comparison": empty,
        }

    require_columns(
        summary_df,
        [
            "family",
            "method",
            "scenario_name",
            "feasibility_rate",
            "service_level",
            "unmet_charge_demand_kwh",
            "total_charging_cost",
            "peak_site_power_kw",
            "solve_time_s",
            "demand_charge_cost",
        ],
    )

    ranking_columns = [
        "feasibility_rate",
        "service_level",
        "unmet_charge_demand_kwh",
        "total_charging_cost",
        "peak_site_power_kw",
        "solve_time_s",
    ]
    method_stats = (
        summary_df.groupby("method", as_index=False)[ranking_columns]
        .mean()
        .sort_values(
            ["feasibility_rate", "service_level", "unmet_charge_demand_kwh", "total_charging_cost"],
            ascending=[False, False, True, True],
        )
    )
    method_stats["rank"] = range(1, len(method_stats) + 1)

    family_winners = (
        summary_df.sort_values(
            ["family", "feasibility_rate", "service_level", "unmet_charge_demand_kwh", "total_charging_cost"],
            ascending=[True, False, False, True, True],
        )
        .groupby("family", as_index=False)
        .first()[
            ["family", "method", "feasibility_rate", "service_level", "unmet_charge_demand_kwh", "total_charging_cost"]
        ]
        .rename(columns={"method": "best_method"})
    )

    method_confidence = bootstrap_group_confidence(
        summary_df,
        group_cols=["method"],
        metrics=[
            "feasibility_rate",
            "service_level",
            "unmet_charge_demand_kwh",
            "total_charging_cost",
            "demand_charge_cost",
        ],
        bootstrap_samples=bootstrap_samples,
    )
    family_confidence = bootstrap_group_confidence(
        summary_df,
        group_cols=["family", "method"],
        metrics=["feasibility_rate", "service_level", "unmet_charge_demand_kwh", "total_charging_cost"],
        bootstrap_samples=bootstrap_samples,
    )
    reference_method = _select_reference_method(summary_df, method_stats)
    pairwise_comparison = paired_bootstrap_comparison(summary_df, reference_method, bootstrap_samples)
    method_stats = annotate_method_ranking(method_stats, pairwise_comparison, reference_method)

    output_dir.mkdir(parents=True, exist_ok=True)
    method_stats.to_csv(output_dir / "publication_method_ranking.csv", index=False)
    family_winners.to_csv(output_dir / "publication_family_winners.csv", index=False)
    method_confidence.to_csv(output_dir / "publication_method_confidence.csv", index=False)
    family_confidence.to_csv(output_dir / "publication_family_confidence.csv", index=False)
    pairwise_comparison.to_csv(output_dir / "publication_pairwise_comparison.csv", index=False)
    return {
        "method_ranking": method_stats,
        "family_winners": family_winners,
        "method_confidence": method_confidence,
        "family_confidence": family_confidence,
        "pairwise_comparison": pairwise_comparison,
    }


def _select_reference_method(summary_df: pd.DataFrame, method_stats: pd.DataFrame) -> str:
    available_methods = set(summary_df["method"].astype(str))
    if DEFAULT_PUBLICATION_REFERENCE_METHOD in available_methods:
        return DEFAULT_PUBLICATION_REFERENCE_METHOD
    reference_method = str(method_stats.iloc[0]["method"])
    warn(
        (
            f"Reference method '{DEFAULT_PUBLICATION_REFERENCE_METHOD}' not found in results. "
            f"Using '{reference_method}' instead."
        ),
        stacklevel=2,
    )
    return reference_method
