import pandas as pd

from scripts.profile_methods import RegressionThresholds, compare_against_baseline, summarize_comparison_issues


def test_compare_against_baseline_flags_regression_reasons() -> None:
    current_df = pd.DataFrame(
        [
            {
                "method": "naive_baseline",
                "mean_planner_wall_time_s": 1.3,
                "mean_total_charging_cost": 110.0,
                "mean_unmet_charge_demand_kwh": 11.0,
                "mean_service_level": 0.91,
            }
        ]
    )
    baseline_df = pd.DataFrame(
        [
            {
                "method": "naive_baseline",
                "mean_planner_wall_time_s": 1.0,
                "mean_total_charging_cost": 100.0,
                "mean_unmet_charge_demand_kwh": 10.0,
                "mean_service_level": 0.95,
            }
        ]
    )

    comparison_df = compare_against_baseline(current_df, baseline_df, RegressionThresholds())

    assert comparison_df.iloc[0]["comparison_status"] == "regression"
    assert "runtime" in comparison_df.iloc[0]["regression_reasons"]
    assert "cost" in comparison_df.iloc[0]["regression_reasons"]
    assert "service_level" in comparison_df.iloc[0]["regression_reasons"]


def test_compare_against_baseline_marks_new_methods() -> None:
    current_df = pd.DataFrame(
        [
            {
                "method": "stochastic_anticipatory",
                "mean_planner_wall_time_s": 0.5,
                "mean_total_charging_cost": 90.0,
                "mean_unmet_charge_demand_kwh": 8.0,
                "mean_service_level": 0.97,
            }
        ]
    )
    baseline_df = pd.DataFrame(
        [
            {
                "method": "naive_baseline",
                "mean_planner_wall_time_s": 1.0,
                "mean_total_charging_cost": 100.0,
                "mean_unmet_charge_demand_kwh": 10.0,
                "mean_service_level": 0.95,
            }
        ]
    )

    comparison_df = compare_against_baseline(current_df, baseline_df, RegressionThresholds())

    assert set(comparison_df["comparison_status"]) == {"missing_from_current", "new_method"}


def test_summarize_comparison_issues_returns_only_regressions() -> None:
    comparison_df = pd.DataFrame(
        [
            {"method": "naive_baseline", "comparison_status": "ok", "regression_reasons": ""},
            {"method": "optimization_ortools", "comparison_status": "regression", "regression_reasons": "runtime,cost"},
        ]
    )

    issues = summarize_comparison_issues(comparison_df)

    assert issues == ["optimization_ortools exceeded baseline thresholds: runtime,cost"]
