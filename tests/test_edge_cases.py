from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
import pytest

from ev_fleet_benchmark.economics import (
    approximate_degradation_penalty_per_vehicle,
    battery_degradation_cost,
    demand_charge_cost,
    onpeak_mask_from_scenario,
)
from ev_fleet_benchmark.methods import build_methods
from ev_fleet_benchmark.methods.naive import NaiveBaselineMethod
from ev_fleet_benchmark.model import (
    Scenario,
    SchedulePlan,
    VehicleRequest,
)
from ev_fleet_benchmark.scenarios import generate_scenario
from ev_fleet_benchmark.simulator import criticality_score, evaluate_plan
from ev_fleet_benchmark.telemetry import (
    ProgressEvent,
    ProgressReporter,
    configure_cli_logger,
)


def _minimal_vehicle(
    vehicle_id: str = "v0",
    battery_kwh: float = 50.0,
    initial_soc: float = 0.2,
    target_soc: float = 0.8,
    max_charge_kw: float = 11.0,
    arrival: int = 0,
    departure: int = 2,
    priority: str = "medium",
) -> VehicleRequest:
    return VehicleRequest(
        vehicle_id=vehicle_id,
        battery_capacity_kwh=battery_kwh,
        initial_soc=initial_soc,
        target_soc=target_soc,
        max_charge_kw=max_charge_kw,
        arrival_slot=arrival,
        departure_slot=departure,
        priority_class=priority,
    )


def _minimal_scenario(
    vehicles: list[VehicleRequest] | None = None,
    horizon_slots: int = 4,
    site_capacity_kw: float = 100.0,
    tariff: float = 0.15,
    metadata: dict[str, object] | None = None,
) -> Scenario:
    veh = vehicles or [_minimal_vehicle(departure=horizon_slots)]
    return Scenario(
        name="test_minimal",
        family="test",
        seed=0,
        time_step_minutes=30,
        horizon_slots=horizon_slots,
        vehicles=veh,
        site_capacity_kw=np.full(horizon_slots, site_capacity_kw),
        tariff_per_kwh=np.full(horizon_slots, tariff),
        metadata=metadata or {},
    )


class TestVehicleRequestEdgeCases:
    def test_reject_empty_vehicle_id(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            _minimal_vehicle(vehicle_id="   ")

    def test_reject_zero_battery(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _minimal_vehicle(battery_kwh=0.0)

    def test_reject_soc_above_one(self) -> None:
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            _minimal_vehicle(initial_soc=1.01)

    def test_reject_target_below_initial(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal"):
            _minimal_vehicle(initial_soc=0.8, target_soc=0.5)

    def test_reject_departure_before_arrival(self) -> None:
        with pytest.raises(ValueError, match="greater than arrival"):
            _minimal_vehicle(arrival=5, departure=3)

    def test_reject_unknown_priority(self) -> None:
        with pytest.raises(ValueError, match="priority_class"):
            _minimal_vehicle(priority="urgent")

    def test_minimal_energy_when_soc_equal(self) -> None:
        v = _minimal_vehicle(initial_soc=0.5, target_soc=0.5)
        assert v.required_energy_kwh == 0.0

    def test_available_slots_property(self) -> None:
        v = _minimal_vehicle(arrival=2, departure=10)
        assert v.available_slots == 8


class TestScenarioEdgeCases:
    def test_reject_nan_in_site_capacity(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Scenario(
                name="bad",
                family="test",
                seed=0,
                time_step_minutes=30,
                horizon_slots=2,
                vehicles=[_minimal_vehicle()],
                site_capacity_kw=np.array([100.0, float("nan")]),
                tariff_per_kwh=np.array([0.1, 0.1]),
            )

    def test_reject_inf_in_tariff(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Scenario(
                name="bad",
                family="test",
                seed=0,
                time_step_minutes=30,
                horizon_slots=2,
                vehicles=[_minimal_vehicle()],
                site_capacity_kw=np.array([100.0, 100.0]),
                tariff_per_kwh=np.array([0.1, float("inf")]),
            )

    def test_reject_negative_site_capacity(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Scenario(
                name="bad",
                family="test",
                seed=0,
                time_step_minutes=30,
                horizon_slots=2,
                vehicles=[_minimal_vehicle()],
                site_capacity_kw=np.array([100.0, -1.0]),
                tariff_per_kwh=np.array([0.1, 0.1]),
            )

    def test_reject_mismatched_array_lengths(self) -> None:
        with pytest.raises(ValueError, match="length must match"):
            Scenario(
                name="bad",
                family="test",
                seed=0,
                time_step_minutes=30,
                horizon_slots=4,
                vehicles=[_minimal_vehicle()],
                site_capacity_kw=np.full(3, 100.0),
                tariff_per_kwh=np.full(4, 0.1),
            )

    def test_reject_vehicle_departure_exceeds_horizon(self) -> None:
        v = _minimal_vehicle(departure=10)
        with pytest.raises(ValueError, match="exceeds scenario horizon"):
            _minimal_scenario(vehicles=[v], horizon_slots=4)


class TestSchedulePlanEdgeCases:
    def test_reject_nan_in_power_matrix(self) -> None:
        power = np.array([[0.0, float("nan")]])
        with pytest.raises(ValueError, match="finite"):
            SchedulePlan(method_name="test", power_kw=power, solve_time_s=0.0, status="ok")

    def test_reject_negative_power(self) -> None:
        power = np.array([[0.0, -1.0]])
        with pytest.raises(ValueError, match="negative dispatch"):
            SchedulePlan(method_name="test", power_kw=power, solve_time_s=0.0, status="ok")

    def test_reject_1d_power(self) -> None:
        power = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="two-dimensional"):
            SchedulePlan(method_name="test", power_kw=power, solve_time_s=0.0, status="ok")


class TestSimulatorEdgeCases:
    def test_zero_power_plan(self) -> None:
        scenario = _minimal_scenario(horizon_slots=4)
        power = np.zeros((1, 4))
        plan = SchedulePlan(method_name="zero", power_kw=power, solve_time_s=0.0, status="ok")
        result = evaluate_plan(scenario, plan)
        assert result.summary["energy_cost"] == 0.0
        assert result.summary["peak_site_power_kw"] == 0.0

    def test_plan_shape_mismatch_raises(self) -> None:
        scenario = _minimal_scenario(horizon_slots=4)
        power = np.zeros((2, 4))
        plan = SchedulePlan(method_name="bad", power_kw=power, solve_time_s=0.0, status="ok")
        with pytest.raises(ValueError, match="shape"):
            evaluate_plan(scenario, plan)

    def test_power_clipped_to_max_charge_kw(self) -> None:
        v = _minimal_vehicle(max_charge_kw=5.0, departure=4)
        scenario = _minimal_scenario(vehicles=[v], horizon_slots=4)
        power = np.full((1, 4), 50.0)
        plan = SchedulePlan(method_name="excess", power_kw=power, solve_time_s=0.0, status="ok")
        result = evaluate_plan(scenario, plan)
        assert result.summary["peak_site_power_kw"] <= 5.0 + 1e-6

    def test_capacity_overload_is_scaled_down(self) -> None:
        v = _minimal_vehicle(max_charge_kw=200.0, departure=4)
        scenario = _minimal_scenario(vehicles=[v], horizon_slots=4, site_capacity_kw=10.0)
        power = np.full((1, 4), 200.0)
        plan = SchedulePlan(method_name="overload", power_kw=power, solve_time_s=0.0, status="ok")
        result = evaluate_plan(scenario, plan)
        assert result.summary["peak_site_power_kw"] <= 10.0 + 1e-6

    def test_negative_tolerance_raises(self) -> None:
        scenario = _minimal_scenario()
        power = np.zeros((1, 4))
        plan = SchedulePlan(method_name="t", power_kw=power, solve_time_s=0.0, status="ok")
        with pytest.raises(ValueError, match="non-negative"):
            evaluate_plan(scenario, plan, tolerance_kwh=-0.1)

    def test_criticality_score_single_slot(self) -> None:
        score = criticality_score(
            remaining_energy_kwh=10.0,
            remaining_slots=1,
            max_charge_kw=11.0,
            dt_hours=0.5,
            priority_weight=1.3,
        )
        assert score > 0.0


class TestEconomicsEdgeCases:
    def test_zero_power_degradation_cost(self) -> None:
        scenario = _minimal_scenario()
        power = np.zeros((1, 4))
        assert battery_degradation_cost(power, scenario) == 0.0

    def test_degradation_with_nonzero_parameters(self) -> None:
        scenario = Scenario(
            name="degrade",
            family="test",
            seed=0,
            time_step_minutes=30,
            horizon_slots=4,
            vehicles=[_minimal_vehicle(departure=4)],
            site_capacity_kw=np.full(4, 100.0),
            tariff_per_kwh=np.full(4, 0.1),
            degradation_cost_per_kwh=0.05,
            degradation_high_soc_multiplier=0.5,
            degradation_c_rate_coefficient=0.1,
        )
        power = np.full((1, 4), 10.0)
        cost = battery_degradation_cost(power, scenario)
        assert cost > 0.0

    def test_demand_charge_empty_array(self) -> None:
        site = np.array([], dtype=float)
        mask = np.array([], dtype=bool)
        all_day, onpeak, total = demand_charge_cost(site, mask, 10.0, 5.0)
        assert all_day == 0.0
        assert onpeak == 0.0
        assert total == 0.0

    def test_demand_charge_no_onpeak(self) -> None:
        site = np.array([10.0, 20.0, 15.0])
        mask = np.array([False, False, False])
        all_day, onpeak, _total = demand_charge_cost(site, mask, 10.0, 5.0)
        assert all_day == 20.0 * 10.0
        assert onpeak == 0.0

    def test_onpeak_mask_missing_metadata(self) -> None:
        scenario = _minimal_scenario(metadata={})
        mask = onpeak_mask_from_scenario(scenario)
        assert not np.any(mask)

    def test_onpeak_mask_out_of_range_slots_ignored(self) -> None:
        scenario = _minimal_scenario(
            horizon_slots=4,
            metadata={"economic_model": {"onpeak_slots": [0, 1, 99]}},
        )
        mask = onpeak_mask_from_scenario(scenario)
        assert mask[0]
        assert mask[1]
        assert mask.sum() == 2

    def test_approximate_degradation_penalty(self) -> None:
        scenario = Scenario(
            name="pen",
            family="test",
            seed=0,
            time_step_minutes=30,
            horizon_slots=4,
            vehicles=[_minimal_vehicle(departure=4), _minimal_vehicle(vehicle_id="v1", departure=4)],
            site_capacity_kw=np.full(4, 200.0),
            tariff_per_kwh=np.full(4, 0.1),
            degradation_cost_per_kwh=0.05,
            degradation_c_rate_coefficient=0.1,
            degradation_high_soc_multiplier=0.3,
        )
        penalties = approximate_degradation_penalty_per_vehicle(scenario)
        assert penalties.shape == (2,)
        assert np.all(penalties >= 0.0)


class TestMethodFactoryEdgeCases:
    def test_build_methods_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            build_methods(["nonexistent_method"])

    def test_build_methods_deduplicates(self) -> None:
        methods = build_methods(["naive_baseline", "naive_baseline"])
        names = [m.name for m in methods]
        assert names == ["naive_baseline"]


class TestTelemetryEdgeCases:
    def test_progress_event_to_payload_strips_none(self) -> None:
        event = ProgressEvent(
            event_type="test",
            run_kind="unit",
            completed_runs=0,
            total_runs=1,
            output_dir="results/test",
            elapsed_s=0.0,
        )
        payload = event.to_payload()
        assert "family_name" not in payload
        assert "method_name" not in payload

    def test_reporter_with_no_sinks_does_not_crash(self) -> None:
        reporter = ProgressReporter()
        event = ProgressEvent(
            event_type="test",
            run_kind="unit",
            completed_runs=0,
            total_runs=1,
            output_dir="results/test",
            elapsed_s=0.0,
        )
        reporter.emit(event)

    def test_configure_cli_logger_rejects_bad_level(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            configure_cli_logger("EXTREME")

    def test_configure_cli_logger_silent_returns_none(self) -> None:
        assert configure_cli_logger("silent") is None

    def test_configure_cli_logger_no_handler_duplication(self) -> None:
        logger = configure_cli_logger("info")
        assert logger is not None
        handler_count_1 = len(logger.handlers)
        configure_cli_logger("debug")
        handler_count_2 = len(logger.handlers)
        assert handler_count_2 == handler_count_1

    def test_reporter_handles_unwritable_jsonl(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("occupied")
        bad_path = blocker / "subdir" / "progress.jsonl"
        reporter = ProgressReporter(jsonl_path=bad_path)
        event = ProgressEvent(
            event_type="test",
            run_kind="unit",
            completed_runs=0,
            total_runs=1,
            output_dir="results/test",
            elapsed_s=0.0,
        )
        reporter.emit(event)


class TestScenarioGenerationEdgeCases:
    def test_all_families_generate_successfully(self) -> None:
        families = [
            "urban_depot_small",
            "regional_mixed_medium",
            "capacity_stressed_peak",
            "uncertain_operations_large",
        ]
        for family in families:
            scenario = generate_scenario(family, seed=42)
            assert len(scenario.vehicles) > 0
            assert scenario.horizon_slots > 0
            assert np.all(np.isfinite(scenario.site_capacity_kw))
            assert np.all(np.isfinite(scenario.tariff_per_kwh))

    def test_different_seeds_produce_different_scenarios(self) -> None:
        s1 = generate_scenario("urban_depot_small", seed=1)
        s2 = generate_scenario("urban_depot_small", seed=2)
        tariffs_differ = not np.array_equal(s1.tariff_per_kwh, s2.tariff_per_kwh)
        fleet_differs = len(s1.vehicles) != len(s2.vehicles)
        assert tariffs_differ or fleet_differs

    def test_same_seed_produces_identical_scenario(self) -> None:
        s1 = generate_scenario("urban_depot_small", seed=99)
        s2 = generate_scenario("urban_depot_small", seed=99)
        assert np.array_equal(s1.tariff_per_kwh, s2.tariff_per_kwh)
        assert np.array_equal(s1.site_capacity_kw, s2.site_capacity_kw)
        assert len(s1.vehicles) == len(s2.vehicles)


class TestNaiveMethodMinimal:
    def test_single_vehicle_single_slot(self) -> None:
        v = _minimal_vehicle(
            battery_kwh=50.0,
            initial_soc=0.5,
            target_soc=0.8,
            max_charge_kw=11.0,
            arrival=0,
            departure=2,
        )
        scenario = _minimal_scenario(vehicles=[v], horizon_slots=2, site_capacity_kw=50.0)
        method = NaiveBaselineMethod()
        plan = method.solve(scenario)
        result = evaluate_plan(scenario, plan)
        assert plan.power_kw.shape == (1, 2)
        assert result.summary["total_charging_cost"] >= 0.0


class TestRoundTripSerialization:
    def test_vehicle_request_round_trip(self) -> None:
        v = _minimal_vehicle(vehicle_id="rt-1", battery_kwh=75.0, initial_soc=0.3, target_soc=0.9)
        rebuilt = VehicleRequest.from_dict(v.to_dict())
        assert rebuilt.vehicle_id == v.vehicle_id
        assert rebuilt.battery_capacity_kwh == v.battery_capacity_kwh
        assert rebuilt.initial_soc == v.initial_soc
        assert rebuilt.target_soc == v.target_soc

    def test_scenario_round_trip(self) -> None:
        scenario = generate_scenario("urban_depot_small", seed=77)
        data = scenario.to_dict()
        rebuilt = Scenario.from_dict(data)
        assert rebuilt.name == scenario.name
        assert rebuilt.family == scenario.family
        assert rebuilt.seed == scenario.seed
        assert rebuilt.horizon_slots == scenario.horizon_slots
        assert len(rebuilt.vehicles) == len(scenario.vehicles)
        assert np.array_equal(rebuilt.site_capacity_kw, scenario.site_capacity_kw)
        assert np.array_equal(rebuilt.tariff_per_kwh, scenario.tariff_per_kwh)
        assert rebuilt.demand_charge_per_kw == scenario.demand_charge_per_kw
        assert rebuilt.degradation_cost_per_kwh == scenario.degradation_cost_per_kwh

    def test_scenario_round_trip_preserves_vehicle_fields(self) -> None:
        scenario = generate_scenario("regional_mixed_medium", seed=33)
        rebuilt = Scenario.from_dict(scenario.to_dict())
        for original, restored in zip(scenario.vehicles, rebuilt.vehicles, strict=True):
            assert original.vehicle_id == restored.vehicle_id
            assert original.battery_capacity_kwh == restored.battery_capacity_kwh
            assert original.arrival_slot == restored.arrival_slot
            assert original.departure_slot == restored.departure_slot
            assert original.priority_class == restored.priority_class


class TestSolverInfeasibility:
    def test_optimization_handles_zero_capacity(self) -> None:
        from ev_fleet_benchmark.methods import OptimizationMethod

        v = _minimal_vehicle(departure=4, max_charge_kw=11.0)
        scenario = _minimal_scenario(vehicles=[v], horizon_slots=4, site_capacity_kw=0.0)
        method = OptimizationMethod()
        plan = method.solve(scenario)
        result = evaluate_plan(scenario, plan)
        assert result.summary["delivered_energy_kwh"] == pytest.approx(0.0, abs=1e-3)

    def test_optimization_handles_tight_departure(self) -> None:
        from ev_fleet_benchmark.methods import OptimizationMethod

        v = _minimal_vehicle(
            battery_kwh=100.0,
            initial_soc=0.0,
            target_soc=1.0,
            max_charge_kw=7.2,
            arrival=0,
            departure=1,
        )
        scenario = _minimal_scenario(vehicles=[v], horizon_slots=4, site_capacity_kw=50.0)
        method = OptimizationMethod()
        plan = method.solve(scenario)
        result = evaluate_plan(scenario, plan)
        assert result.summary["unmet_charge_demand_kwh"] > 0.0
