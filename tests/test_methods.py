from ev_fleet_benchmark.methods import (
    GreedyUrgencyMethod,
    NaiveBaselineMethod,
    OptimizationMethod,
    RollingHorizonOptimizationMethod,
    ScenarioTreeOptimizationMethod,
    StochasticAnticipatoryMethod,
)
from ev_fleet_benchmark.scenarios import generate_scenario
from ev_fleet_benchmark.simulator import evaluate_plan


def test_all_methods_produce_valid_schedule_shapes() -> None:
    scenario = generate_scenario("urban_depot_small", seed=21)
    methods = [
        NaiveBaselineMethod(),
        GreedyUrgencyMethod(),
        OptimizationMethod(),
        RollingHorizonOptimizationMethod(),
        ScenarioTreeOptimizationMethod(),
        StochasticAnticipatoryMethod(),
    ]

    for method in methods:
        plan = method.solve(scenario)
        evaluation = evaluate_plan(scenario, plan)

        assert plan.power_kw.shape == (len(scenario.vehicles), scenario.horizon_slots)
        assert evaluation.summary["total_charging_cost"] >= 0.0
        assert evaluation.summary["all_day_demand_charge_cost"] >= 0.0
        assert evaluation.summary["onpeak_demand_charge_cost"] >= 0.0
        assert evaluation.summary["unmet_charge_demand_kwh"] >= 0.0
        assert evaluation.summary["solve_time_s"] >= 0.0


def test_online_methods_return_expected_metadata() -> None:
    scenario = generate_scenario("uncertain_operations_large", seed=4)

    rolling_plan = RollingHorizonOptimizationMethod().solve(scenario)
    tree_plan = ScenarioTreeOptimizationMethod().solve(scenario)
    stochastic_plan = StochasticAnticipatoryMethod().solve(scenario)

    assert rolling_plan.metadata["lookahead_slots"] > 0
    assert tree_plan.metadata["node_count"] >= 1
    assert stochastic_plan.metadata["sample_count"] > 0


def test_optimization_is_never_worse_than_naive_on_unmet_demand_for_same_instance() -> None:
    scenario = generate_scenario("regional_mixed_medium", seed=12)

    naive_result = evaluate_plan(scenario, NaiveBaselineMethod().solve(scenario))
    optimization_result = evaluate_plan(scenario, OptimizationMethod().solve(scenario))

    assert (
        optimization_result.summary["unmet_charge_demand_kwh"] <= naive_result.summary["unmet_charge_demand_kwh"] + 1e-6
    )
