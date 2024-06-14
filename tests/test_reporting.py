from pathlib import Path

import pandas as pd
import pytest

from ev_fleet_benchmark.reporting import (
    aggregate_results,
    create_plots,
    create_publication_tables,
    create_sensitivity_tables,
    dataframe_to_markdown,
)


def _summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_name": "urban_depot_small_seed_1",
                "family": "urban_depot_small",
                "method": "optimization_ortools",
                "feasibility_rate": 1.0,
                "service_level": 1.0,
                "unmet_charge_demand_kwh": 0.0,
                "energy_cost": 12.0,
                "all_day_demand_charge_cost": 5.0,
                "onpeak_demand_charge_cost": 3.0,
                "demand_charge_cost": 8.0,
                "battery_degradation_cost": 1.0,
                "total_charging_cost": 21.0,
                "peak_site_power_kw": 40.0,
                "load_variance_kw2": 4.0,
                "solve_time_s": 0.3,
                "sensitivity_profile": "baseline",
                "base_scenario_name": "urban_depot_small_seed_1",
            },
            {
                "scenario_name": "urban_depot_small_seed_1",
                "family": "urban_depot_small",
                "method": "greedy_urgency",
                "feasibility_rate": 1.0,
                "service_level": 0.98,
                "unmet_charge_demand_kwh": 0.2,
                "energy_cost": 11.5,
                "all_day_demand_charge_cost": 5.4,
                "onpeak_demand_charge_cost": 3.1,
                "demand_charge_cost": 8.5,
                "battery_degradation_cost": 0.9,
                "total_charging_cost": 20.9,
                "peak_site_power_kw": 42.0,
                "load_variance_kw2": 4.5,
                "solve_time_s": 0.05,
                "sensitivity_profile": "baseline",
                "base_scenario_name": "urban_depot_small_seed_1",
            },
            {
                "scenario_name": "urban_depot_small_seed_1__tariff_stress",
                "family": "urban_depot_small",
                "method": "optimization_ortools",
                "feasibility_rate": 1.0,
                "service_level": 1.0,
                "unmet_charge_demand_kwh": 0.0,
                "energy_cost": 14.0,
                "all_day_demand_charge_cost": 6.0,
                "onpeak_demand_charge_cost": 4.0,
                "demand_charge_cost": 10.0,
                "battery_degradation_cost": 1.2,
                "total_charging_cost": 25.2,
                "peak_site_power_kw": 40.0,
                "load_variance_kw2": 4.1,
                "solve_time_s": 0.32,
                "sensitivity_profile": "tariff_stress",
                "base_scenario_name": "urban_depot_small_seed_1",
            },
        ]
    )


def test_aggregate_results_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_results(pd.DataFrame([{"method": "optimization_ortools"}]))


def test_publication_and_sensitivity_tables_write_expected_outputs(tmp_path: Path) -> None:
    summary_df = _summary_df()

    publication_tables = create_publication_tables(summary_df, tmp_path / "publication", bootstrap_samples=20)
    sensitivity_tables = create_sensitivity_tables(summary_df, tmp_path / "publication", bootstrap_samples=20)

    assert not publication_tables["method_ranking"].empty
    assert not sensitivity_tables["sensitivity_profile_aggregate"].empty
    assert (tmp_path / "publication" / "publication_method_ranking.csv").exists()
    assert (tmp_path / "publication" / "sensitivity_profile_aggregate.csv").exists()


def test_create_plots_and_markdown_helpers_work(tmp_path: Path) -> None:
    summary_df = _summary_df().drop(columns=["sensitivity_profile", "base_scenario_name"])
    family_df, _ = aggregate_results(summary_df)

    create_plots(summary_df, family_df, tmp_path / "plots")

    markdown = dataframe_to_markdown(summary_df.head(1))
    assert "| scenario_name |" in markdown
    assert (tmp_path / "plots" / "cost_by_family.png").exists()


def test_publication_tables_warn_when_default_reference_method_is_missing(tmp_path: Path) -> None:
    summary_df = _summary_df().copy()
    summary_df["method"] = ["greedy_urgency", "naive_baseline", "greedy_urgency"]

    with pytest.warns(UserWarning, match="Reference method 'optimization_ortools' not found"):
        tables = create_publication_tables(summary_df, tmp_path / "publication", bootstrap_samples=20)

    assert tables["method_ranking"]["reference_method"].nunique() == 1


def test_sensitivity_tables_warn_when_baseline_profile_is_missing(tmp_path: Path) -> None:
    summary_df = _summary_df().query("sensitivity_profile != 'baseline'").copy()

    with pytest.warns(UserWarning, match="delta calculations will be skipped"):
        tables = create_sensitivity_tables(summary_df, tmp_path / "publication", bootstrap_samples=20)

    assert tables["sensitivity_profile_deltas"].empty
