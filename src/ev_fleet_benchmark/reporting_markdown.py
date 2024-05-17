from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

from ev_fleet_benchmark.reporting_common import metadata_int_list, metadata_str_list


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    separator = ["---"] * len(headers)
    table_rows = [headers, separator, *rows]
    return "\n".join(f"| {' | '.join(row)} |" for row in table_rows)


def write_markdown_report(
    path: Path,
    summary_df: pd.DataFrame,
    family_aggregate_df: pd.DataFrame,
    overall_aggregate_df: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    family_names = metadata_str_list(metadata, "family_names", required=True)
    seeds = metadata_int_list(metadata, "seeds", required=True)
    methods = metadata_str_list(metadata, "methods", required=True)
    lines = [
        "# Benchmark Report",
        "",
        "## Configuration",
        "",
        f"- Scenario families: {', '.join(family_names)}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Time step: {metadata['time_step_minutes']} minutes",
        f"- Methods: {', '.join(methods)}",
        "",
        "## Overall Results",
        "",
        dataframe_to_markdown(overall_aggregate_df.round(4)) if not overall_aggregate_df.empty else "No results.",
        "",
        "## Results by Scenario Family",
        "",
        dataframe_to_markdown(family_aggregate_df.round(4)) if not family_aggregate_df.empty else "No results.",
        "",
        "## Run Count",
        "",
        f"- Total evaluated runs: {len(summary_df)}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_publication_report(
    path: Path, ranking_df: pd.DataFrame, winners_df: pd.DataFrame, metadata: dict[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    method_confidence_df = metadata.get("method_confidence_df")
    pairwise_df = metadata.get("pairwise_df")
    family_names = metadata_str_list(metadata, "family_names", required=True)
    seeds = metadata_int_list(metadata, "seeds", required=True)
    methods = metadata_str_list(metadata, "methods", required=True)
    method_confidence_markdown = (
        dataframe_to_markdown(method_confidence_df.round(4))
        if isinstance(method_confidence_df, pd.DataFrame) and not method_confidence_df.empty
        else "No confidence interval data."
    )
    pairwise_markdown = (
        dataframe_to_markdown(pairwise_df.round(4))
        if isinstance(pairwise_df, pd.DataFrame) and not pairwise_df.empty
        else "No pairwise comparison data."
    )
    lines = [
        "# Benchmark Study Summary",
        "",
        "## Study Configuration",
        "",
        f"- Scenario families: {', '.join(family_names)}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Methods: {', '.join(methods)}",
        f"- Bootstrap samples: {metadata.get('bootstrap_samples', 'n/a')}",
        "",
        "## Method Ranking",
        "",
        dataframe_to_markdown(ranking_df.round(4)) if not ranking_df.empty else "No ranking data.",
        "",
        "## Method Confidence Intervals",
        "",
        method_confidence_markdown,
        "",
        "## Family Winners",
        "",
        dataframe_to_markdown(winners_df.round(4)) if not winners_df.empty else "No family winner data.",
        "",
        "## Pairwise Comparison Versus Reference",
        "",
        pairwise_markdown,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sensitivity_report(path: Path, tables: dict[str, pd.DataFrame], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile_ranking = tables["sensitivity_profile_ranking"]
    robustness = tables["sensitivity_method_robustness"]
    deltas = tables["sensitivity_profile_deltas"]
    profiles = metadata_str_list(metadata, "profile_names", required=True)
    family_names = metadata_str_list(metadata, "family_names", required=True)
    seeds = metadata_int_list(metadata, "seeds", required=True)
    methods = metadata_str_list(metadata, "methods", required=True)
    lines = [
        "# Sensitivity Study Summary",
        "",
        "## Configuration",
        "",
        f"- Sensitivity profiles: {', '.join(profiles)}",
        f"- Scenario families: {', '.join(family_names)}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Methods: {', '.join(methods)}",
        f"- Bootstrap samples: {metadata.get('bootstrap_samples', 'n/a')}",
        "",
        "## Ranking by Sensitivity Profile",
        "",
        dataframe_to_markdown(profile_ranking.round(4))
        if not profile_ranking.empty
        else "No sensitivity ranking data.",
        "",
        "## Method Robustness",
        "",
        dataframe_to_markdown(robustness.round(4)) if not robustness.empty else "No robustness data.",
        "",
        "## Mean Delta Versus Baseline",
        "",
        dataframe_to_markdown(deltas.round(4)) if not deltas.empty else "No baseline delta data.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
