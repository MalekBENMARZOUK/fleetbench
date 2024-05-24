from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from matplotlib.figure import Figure

from ev_fleet_benchmark.reporting_common import require_columns


def create_plots(summary_df: pd.DataFrame, family_aggregate_df: pd.DataFrame, plots_dir: Path) -> None:
    if summary_df.empty or family_aggregate_df.empty:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    require_columns(summary_df, ["method", "solve_time_s"])

    plt.style.use("ggplot")
    figures = [
        ("total_charging_cost", "Mean Cost by Family", "Cost [currency units]", "cost_by_family.png"),
        (
            "demand_charge_cost",
            "Demand Charge by Family",
            "Demand Charge [currency units]",
            "demand_charge_by_family.png",
        ),
        (
            "battery_degradation_cost",
            "Battery Wear Cost by Family",
            "Degradation Cost [currency units]",
            "degradation_by_family.png",
        ),
        ("unmet_charge_demand_kwh", "Unmet Energy by Family", "Unmet Demand [kWh]", "unmet_by_family.png"),
        ("peak_site_power_kw", "Peak Site Power by Family", "Peak Power [kW]", "peak_power_by_family.png"),
    ]

    for metric, title, ylabel, filename in figures:
        if metric not in family_aggregate_df.columns:
            continue
        pivot = family_aggregate_df.pivot(index="family", columns="method", values=metric)
        if pivot.empty:
            continue
        ax = pivot.plot(kind="bar", figsize=(10, 5), rot=20, colormap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Scenario Family")
        ax.set_ylabel(ylabel)
        figure = cast("Figure", ax.get_figure())
        figure.tight_layout()
        figure.savefig(plots_dir / filename, dpi=180)
        plt.close(figure)

    solve_df = summary_df.groupby("method", as_index=False)["solve_time_s"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(solve_df)))
    ax.bar(solve_df["method"], solve_df["solve_time_s"], color=colors)
    ax.set_title("Mean Solve Time by Method")
    ax.set_ylabel("Solve Time [s]")
    fig.tight_layout()
    fig.savefig(plots_dir / "solve_time_overall.png", dpi=180)
    plt.close(fig)
