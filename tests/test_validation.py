import numpy as np
import pytest

from ev_fleet_benchmark.benchmark import BenchmarkConfig, SensitivityStudyConfig
from ev_fleet_benchmark.methods import MethodFactoryConfigSet, build_methods, method_names
from ev_fleet_benchmark.methods.config import (
    GreedyMethodConfig,
    OptimizationPenaltyConfig,
    RollingHorizonMethodConfig,
    ScenarioTreeMethodConfig,
    StochasticMethodConfig,
)
from ev_fleet_benchmark.methods.greedy import GreedyUrgencyMethod
from ev_fleet_benchmark.methods.optimization import OptimizationMethod
from ev_fleet_benchmark.methods.rolling_horizon import RollingHorizonOptimizationMethod
from ev_fleet_benchmark.methods.scenario_tree import ScenarioTreeOptimizationMethod
from ev_fleet_benchmark.methods.stochastic import StochasticAnticipatoryMethod
from ev_fleet_benchmark.model import SchedulePlan, VehicleRequest
from ev_fleet_benchmark.scenarios import apply_sensitivity_profile, generate_scenario
from ev_fleet_benchmark.simulator import evaluate_plan


def test_benchmark_config_rejects_invalid_time_step() -> None:
    with pytest.raises(ValueError, match="divide evenly into 24 hours"):
        BenchmarkConfig(family_names=["urban_depot_small"], seeds=[1], time_step_minutes=17)


def test_sensitivity_config_deduplicates_inputs() -> None:
    config = SensitivityStudyConfig(
        profile_names=["baseline", "baseline", "tariff_stress"],
        family_names=["urban_depot_small", "urban_depot_small"],
        seeds=[5, 5, 6],
    )

    assert config.profile_names == ["baseline", "tariff_stress"]
    assert config.family_names == ["urban_depot_small"]
    assert config.seeds == [5, 6]


def test_build_methods_rejects_empty_explicit_selection() -> None:
    with pytest.raises(ValueError, match="At least one method"):
        build_methods([])


def test_method_names_are_unique() -> None:
    names = method_names()

    assert len(names) == len(set(names))


def test_build_methods_applies_shared_config_set() -> None:
    methods = build_methods(
        [
            "optimization_ortools",
            "greedy_urgency",
            "rolling_horizon_ortools",
            "scenario_tree_ortools",
            "stochastic_anticipatory",
        ],
        config_set=MethodFactoryConfigSet(
            greedy=GreedyMethodConfig(
                capacity_reserve_fraction=0.1,
                all_day_demand_guard_divisor=9.0,
                onpeak_demand_guard_divisor=7.0,
                onpeak_score_discount=0.91,
            ),
            optimization=OptimizationPenaltyConfig(peak_penalty=0.03, unmet_penalty=400.0),
            rolling_horizon=RollingHorizonMethodConfig(
                lookahead_slots=5, penalties=OptimizationPenaltyConfig(peak_penalty=0.04, unmet_penalty=410.0)
            ),
            scenario_tree=ScenarioTreeMethodConfig(
                lookahead_slots=4, penalties=OptimizationPenaltyConfig(peak_penalty=0.02, unmet_penalty=390.0)
            ),
            stochastic=StochasticMethodConfig(lookahead_slots=7, sample_count=12, reserve_quantile=0.6),
        ),
    )

    optimization = next(method for method in methods if isinstance(method, OptimizationMethod))
    greedy = next(method for method in methods if isinstance(method, GreedyUrgencyMethod))
    rolling = next(method for method in methods if isinstance(method, RollingHorizonOptimizationMethod))
    tree = next(method for method in methods if isinstance(method, ScenarioTreeOptimizationMethod))
    stochastic = next(method for method in methods if isinstance(method, StochasticAnticipatoryMethod))

    assert optimization.config.peak_penalty == pytest.approx(0.03)
    assert optimization.config.unmet_penalty == pytest.approx(400.0)
    assert greedy.config.capacity_reserve_fraction == pytest.approx(0.1)
    assert rolling.config.lookahead_slots == 5
    assert rolling.config.penalties.unmet_penalty == pytest.approx(410.0)
    assert tree.config.lookahead_slots == 4
    assert stochastic.config.sample_count == 12


def test_schedule_plan_rejects_negative_dispatch() -> None:
    scenario = generate_scenario("urban_depot_small", seed=42)
    power = np.zeros((len(scenario.vehicles), scenario.horizon_slots), dtype=float)
    power[0, scenario.vehicles[0].arrival_slot] = -0.5

    with pytest.raises(ValueError, match="must not contain negative dispatch values"):
        SchedulePlan(method_name="broken", power_kw=power, solve_time_s=0.1, status="invalid")


def test_evaluate_plan_rejects_wrong_shape() -> None:
    scenario = generate_scenario("urban_depot_small", seed=43)
    bad_power = np.zeros((len(scenario.vehicles) + 1, scenario.horizon_slots), dtype=float)
    plan = SchedulePlan(method_name="bad_shape", power_kw=bad_power, solve_time_s=0.1, status="invalid")

    with pytest.raises(ValueError, match="expected"):
        evaluate_plan(scenario, plan)


def test_vehicle_request_rejects_invalid_soc_ordering() -> None:
    with pytest.raises(ValueError, match="target_soc must be greater than or equal to initial_soc"):
        VehicleRequest(
            vehicle_id="EV-001",
            battery_capacity_kwh=60.0,
            initial_soc=0.8,
            target_soc=0.5,
            max_charge_kw=11.0,
            arrival_slot=2,
            departure_slot=6,
            priority_class="medium",
        )


def test_generate_scenario_rejects_unknown_family_with_available_values() -> None:
    with pytest.raises(ValueError, match="Available families"):
        generate_scenario("not_a_family", seed=1)


def test_apply_sensitivity_profile_rejects_unknown_profile_with_available_values() -> None:
    scenario = generate_scenario("urban_depot_small", seed=7)

    with pytest.raises(ValueError, match="Available profiles"):
        apply_sensitivity_profile(scenario, "not_a_profile")
