from __future__ import annotations

import numpy as np
import pytest

from ev_fleet_benchmark.model import Scenario, VehicleRequest
from ev_fleet_benchmark.scenario_tree import ScenarioTreeNode, build_uncertainty_tree


def _make_scenario(
    *,
    delay_probability: float = 0.2,
    delay_slot_range: tuple[int, int] = (1, 3),
    derate_probability: float = 0.1,
    derate_severity_range: tuple[float, float] = (0.1, 0.2),
    horizon_slots: int = 48,
    planned_arrival_offset: int = 0,
) -> Scenario:
    vehicles = [
        VehicleRequest(
            vehicle_id="EV-001",
            battery_capacity_kwh=60.0,
            initial_soc=0.2,
            target_soc=0.9,
            max_charge_kw=11.0,
            arrival_slot=14 + planned_arrival_offset,
            departure_slot=30,
            priority_class="medium",
            planned_arrival_slot=14,
        ),
        VehicleRequest(
            vehicle_id="EV-002",
            battery_capacity_kwh=75.0,
            initial_soc=0.3,
            target_soc=0.85,
            max_charge_kw=22.0,
            arrival_slot=18,
            departure_slot=40,
            priority_class="high",
            planned_arrival_slot=16,
        ),
    ]
    return Scenario(
        name="tree_test",
        family="test_family",
        seed=42,
        time_step_minutes=30,
        horizon_slots=horizon_slots,
        vehicles=vehicles,
        site_capacity_kw=np.full(horizon_slots, 100.0),
        tariff_per_kwh=np.full(horizon_slots, 0.2),
        demand_charge_per_kw=10.0,
        metadata={
            "uncertainty_model": {
                "arrival_delay_probability": delay_probability,
                "arrival_delay_slot_range": delay_slot_range,
                "site_derate_probability": derate_probability,
                "site_derate_duration_range": (1, 3),
                "site_derate_severity_range": derate_severity_range,
            },
            "economic_model": {"onpeak_slots": [10, 11, 12, 13, 14]},
        },
    )


class TestBuildUncertaintyTree:
    def test_returns_four_nodes_with_nonzero_probabilities(self) -> None:
        scenario = _make_scenario()
        nodes = build_uncertainty_tree(scenario, current_slot=10, lookahead_slots=8)
        assert len(nodes) == 4
        node_ids = {n.node_id for n in nodes}
        assert node_ids == {"nominal", "delay_only", "derate_only", "delay_and_derate"}

    def test_probabilities_sum_to_one(self) -> None:
        scenario = _make_scenario()
        nodes = build_uncertainty_tree(scenario, current_slot=10, lookahead_slots=8)
        total = sum(n.probability for n in nodes)
        assert abs(total - 1.0) < 1e-9

    def test_probabilities_are_non_negative(self) -> None:
        scenario = _make_scenario()
        nodes = build_uncertainty_tree(scenario, current_slot=10, lookahead_slots=8)
        for node in nodes:
            assert node.probability >= 0.0

    def test_nominal_node_uses_planned_arrivals_for_future_vehicles(self) -> None:
        scenario = _make_scenario()
        nodes = build_uncertainty_tree(scenario, current_slot=10, lookahead_slots=8)
        nominal = next(n for n in nodes if n.node_id == "nominal")
        # EV-001 planned_arrival_slot=14 > current_slot=10, so nominal uses planned=14
        assert nominal.arrival_slots[0] == 14
        # EV-002 planned_arrival_slot=16 > current_slot=10, so nominal uses planned=16
        assert nominal.arrival_slots[1] == 16

    def test_past_arrivals_use_actual_slot(self) -> None:
        scenario = _make_scenario()
        # current_slot=20 means EV-001 (arrival=14) and EV-002 (arrival=18) are both in the past
        nodes = build_uncertainty_tree(scenario, current_slot=20, lookahead_slots=8)
        nominal = next(n for n in nodes if n.node_id == "nominal")
        assert nominal.arrival_slots[0] == 14  # actual arrival
        assert nominal.arrival_slots[1] == 18  # actual arrival

    def test_delay_node_adds_delay_to_future_planned_arrivals(self) -> None:
        scenario = _make_scenario(delay_slot_range=(2, 4))
        nodes = build_uncertainty_tree(scenario, current_slot=10, lookahead_slots=8)
        delay_node = next(n for n in nodes if n.node_id == "delay_only")
        delay_mid = max(2, round((2 + 4) / 2.0))  # 3
        # EV-001: planned=14, future -> 14 + 3 = 17
        assert delay_node.arrival_slots[0] == 14 + delay_mid
        # EV-002: planned=16, future -> 16 + 3 = 19
        assert delay_node.arrival_slots[1] == 16 + delay_mid

    def test_derate_node_reduces_capacity(self) -> None:
        scenario = _make_scenario(derate_severity_range=(0.1, 0.3))
        nodes = build_uncertainty_tree(scenario, current_slot=10, lookahead_slots=8)
        derate_node = next(n for n in nodes if n.node_id == "derate_only")
        derate_mid = (0.1 + 0.3) / 2.0  # 0.2
        expected_capacity = 100.0 * (1.0 - derate_mid)
        np.testing.assert_allclose(derate_node.site_capacity_kw, expected_capacity)

    def test_delay_and_derate_node_applies_both(self) -> None:
        scenario = _make_scenario(
            delay_slot_range=(1, 3),
            derate_severity_range=(0.1, 0.2),
        )
        nodes = build_uncertainty_tree(scenario, current_slot=10, lookahead_slots=8)
        both = next(n for n in nodes if n.node_id == "delay_and_derate")
        # delay applied
        delay_mid = max(1, round((1 + 3) / 2.0))  # 2
        assert both.arrival_slots[0] == 14 + delay_mid
        # derate applied (uses max(derate_mid, severity_low))
        derate_mid = (0.1 + 0.2) / 2.0
        severity_low = 0.1
        effective_derate = max(derate_mid, severity_low)
        expected_capacity = 100.0 * (1.0 - effective_derate)
        np.testing.assert_allclose(both.site_capacity_kw, expected_capacity)

    def test_capacity_window_matches_lookahead(self) -> None:
        scenario = _make_scenario(horizon_slots=48)
        nodes = build_uncertainty_tree(scenario, current_slot=10, lookahead_slots=6)
        for node in nodes:
            assert len(node.site_capacity_kw) == 6

    def test_lookahead_clamped_to_horizon(self) -> None:
        scenario = _make_scenario(horizon_slots=48)
        nodes = build_uncertainty_tree(scenario, current_slot=45, lookahead_slots=10)
        for node in nodes:
            assert len(node.site_capacity_kw) == 3  # 48 - 45

    def test_zero_delay_probability(self) -> None:
        scenario = _make_scenario(delay_probability=0.0, derate_probability=0.2)
        nodes = build_uncertainty_tree(scenario, current_slot=10, lookahead_slots=8)
        total = sum(n.probability for n in nodes)
        assert abs(total - 1.0) < 1e-9
        delay_only = next(n for n in nodes if n.node_id == "delay_only")
        delay_and_derate = next(n for n in nodes if n.node_id == "delay_and_derate")
        assert delay_only.probability == 0.0
        assert delay_and_derate.probability == 0.0

    def test_zero_derate_probability(self) -> None:
        scenario = _make_scenario(delay_probability=0.3, derate_probability=0.0)
        nodes = build_uncertainty_tree(scenario, current_slot=10, lookahead_slots=8)
        total = sum(n.probability for n in nodes)
        assert abs(total - 1.0) < 1e-9
        derate_only = next(n for n in nodes if n.node_id == "derate_only")
        delay_and_derate = next(n for n in nodes if n.node_id == "delay_and_derate")
        assert derate_only.probability == 0.0
        assert delay_and_derate.probability == 0.0

    def test_both_probabilities_zero_returns_nominal_with_full_weight(self) -> None:
        scenario = _make_scenario(delay_probability=0.0, derate_probability=0.0)
        nodes = build_uncertainty_tree(scenario, current_slot=10, lookahead_slots=8)
        # All 4 nodes returned; only nominal has non-zero probability
        assert len(nodes) == 4
        nominal = next(n for n in nodes if n.node_id == "nominal")
        assert nominal.probability == pytest.approx(1.0)
        for node in nodes:
            if node.node_id != "nominal":
                assert node.probability == 0.0

    def test_missing_uncertainty_model_metadata(self) -> None:
        scenario = _make_scenario()
        # Strip metadata
        import dataclasses

        bare = dataclasses.replace(scenario, metadata={})
        nodes = build_uncertainty_tree(bare, current_slot=10, lookahead_slots=8)
        # All 4 nodes returned; only nominal has weight
        assert len(nodes) == 4
        nominal = next(n for n in nodes if n.node_id == "nominal")
        assert nominal.probability == pytest.approx(1.0)
        total = sum(n.probability for n in nodes)
        assert abs(total - 1.0) < 1e-9

    def test_delay_does_not_exceed_horizon(self) -> None:
        scenario = _make_scenario(delay_slot_range=(5, 10), horizon_slots=48)
        nodes = build_uncertainty_tree(scenario, current_slot=10, lookahead_slots=8)
        for node in nodes:
            assert np.all(node.arrival_slots < scenario.horizon_slots)

    def test_node_arrival_slots_have_correct_length(self) -> None:
        scenario = _make_scenario()
        nodes = build_uncertainty_tree(scenario, current_slot=10, lookahead_slots=8)
        for node in nodes:
            assert len(node.arrival_slots) == len(scenario.vehicles)

    def test_vehicle_without_planned_arrival(self) -> None:
        vehicles = [
            VehicleRequest(
                vehicle_id="EV-001",
                battery_capacity_kwh=60.0,
                initial_soc=0.2,
                target_soc=0.9,
                max_charge_kw=11.0,
                arrival_slot=14,
                departure_slot=30,
                priority_class="medium",
                planned_arrival_slot=None,  # No planned arrival
            ),
        ]
        scenario = Scenario(
            name="no_planned",
            family="test",
            seed=1,
            time_step_minutes=30,
            horizon_slots=48,
            vehicles=vehicles,
            site_capacity_kw=np.full(48, 100.0),
            tariff_per_kwh=np.full(48, 0.2),
            metadata={
                "uncertainty_model": {
                    "arrival_delay_probability": 0.2,
                    "arrival_delay_slot_range": (1, 3),
                    "site_derate_probability": 0.1,
                    "site_derate_severity_range": (0.1, 0.2),
                },
            },
        )
        nodes = build_uncertainty_tree(scenario, current_slot=10, lookahead_slots=8)
        # planned_arrival_slot is None -> falls back to arrival_slot (14)
        nominal = next(n for n in nodes if n.node_id == "nominal")
        assert nominal.arrival_slots[0] == 14


class TestScenarioTreeNode:
    def test_frozen_dataclass(self) -> None:
        node = ScenarioTreeNode(
            node_id="test",
            probability=0.5,
            arrival_slots=np.array([10, 20]),
            site_capacity_kw=np.array([100.0, 100.0]),
        )
        with pytest.raises(AttributeError):
            node.node_id = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        arrivals = np.array([5, 10])
        capacity = np.array([50.0, 60.0])
        node = ScenarioTreeNode(
            node_id="n1",
            probability=0.75,
            arrival_slots=arrivals,
            site_capacity_kw=capacity,
        )
        assert node.node_id == "n1"
        assert node.probability == 0.75
        np.testing.assert_array_equal(node.arrival_slots, arrivals)
        np.testing.assert_array_equal(node.site_capacity_kw, capacity)
