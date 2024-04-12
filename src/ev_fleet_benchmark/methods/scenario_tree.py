from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace
from time import perf_counter

import numpy as np
from ortools.linear_solver import pywraplp

from ev_fleet_benchmark.economics import approximate_degradation_penalty_per_vehicle, onpeak_mask_from_scenario
from ev_fleet_benchmark.exceptions import SolverUnavailableError
from ev_fleet_benchmark.methods.base import ScheduleMethod
from ev_fleet_benchmark.methods.config import ScenarioTreeMethodConfig, with_optimization_overrides
from ev_fleet_benchmark.methods.optimization import solver_status_name
from ev_fleet_benchmark.model import PRIORITY_WEIGHTS, Scenario, SchedulePlan
from ev_fleet_benchmark.scenario_tree import ScenarioTreeNode, build_uncertainty_tree

_logger = logging.getLogger(__name__)


class ScenarioTreeOptimizationMethod(ScheduleMethod):
    name = "scenario_tree_ortools"

    def __init__(
        self,
        config: ScenarioTreeMethodConfig | None = None,
        *,
        lookahead_slots: int | None = None,
        unmet_penalty: float | None = None,
        peak_penalty: float | None = None,
    ) -> None:
        base = config or ScenarioTreeMethodConfig()
        penalties = with_optimization_overrides(base.penalties, peak_penalty=peak_penalty, unmet_penalty=unmet_penalty)
        overrides = {"lookahead_slots": lookahead_slots} if lookahead_slots is not None else {}
        self.config = replace(base, penalties=penalties, **overrides)

    def solve(self, scenario: Scenario) -> SchedulePlan:
        start = perf_counter()
        _logger.debug(
            "Solving %s on %s (%d vehicles, %d slots)",
            self.name,
            scenario.name,
            len(scenario.vehicles),
            scenario.horizon_slots,
        )
        vehicle_count = len(scenario.vehicles)
        power = np.zeros((vehicle_count, scenario.horizon_slots), dtype=float)
        remaining_energy = np.array([vehicle.required_energy_kwh for vehicle in scenario.vehicles], dtype=float)
        efficiencies = np.array([vehicle.charge_efficiency for vehicle in scenario.vehicles], dtype=float)
        statuses: list[str] = []
        node_count = 0

        for current_slot in range(scenario.horizon_slots):
            if np.all(remaining_energy <= 1e-9):
                break
            nodes = build_uncertainty_tree(
                scenario, current_slot=current_slot, lookahead_slots=self.config.lookahead_slots
            )
            node_count = len(nodes)
            first_stage_dispatch, status = _solve_tree_subproblem(
                scenario=scenario,
                nodes=nodes,
                current_slot=current_slot,
                remaining_energy=remaining_energy,
                unmet_penalty=self.config.penalties.unmet_penalty,
                peak_penalty=self.config.penalties.peak_penalty,
            )
            statuses.append(status)

            for vehicle_index, vehicle in enumerate(scenario.vehicles):
                if not (vehicle.arrival_slot <= current_slot < vehicle.departure_slot):
                    first_stage_dispatch[vehicle_index] = 0.0

            total_dispatch = float(first_stage_dispatch.sum())
            capacity = float(scenario.site_capacity_kw[current_slot])
            if total_dispatch > capacity + 1e-9 and total_dispatch > 0.0:
                first_stage_dispatch *= capacity / total_dispatch

            power[:, current_slot] = first_stage_dispatch
            remaining_energy -= first_stage_dispatch * scenario.dt_hours * efficiencies
            remaining_energy = np.clip(remaining_energy, 0.0, None)

        return SchedulePlan(
            method_name=self.name,
            power_kw=power,
            solve_time_s=perf_counter() - start,
            status="scenario_tree",
            metadata={
                "lookahead_slots": self.config.lookahead_slots,
                "node_count": node_count,
                "subproblem_status_counts": dict(Counter(statuses)),
            },
        )


def _solve_tree_subproblem(
    scenario: Scenario,
    nodes: list[ScenarioTreeNode],
    current_slot: int,
    remaining_energy: np.ndarray,
    unmet_penalty: float,
    peak_penalty: float,
) -> tuple[np.ndarray, str]:
    solver = pywraplp.Solver.CreateSolver("GLOP") or pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise SolverUnavailableError("OR-Tools linear solver is unavailable — ensure ortools is installed correctly")

    local_slots = list(
        range(current_slot, min(scenario.horizon_slots, current_slot + max(1, len(nodes[0].site_capacity_kw))))
    )
    local_width = len(local_slots)
    vehicle_count = len(scenario.vehicles)

    first_stage = {}
    scenario_power = {}
    unmet = {}
    scenario_peak = {}
    onpeak_peak = {}
    onpeak_mask = onpeak_mask_from_scenario(scenario)
    degradation_penalties = approximate_degradation_penalty_per_vehicle(scenario)

    for vehicle_index, vehicle in enumerate(scenario.vehicles):
        upper_bound = vehicle.max_charge_kw if vehicle.arrival_slot <= current_slot < vehicle.departure_slot else 0.0
        first_stage[vehicle_index] = solver.NumVar(0.0, upper_bound, f"first_{vehicle_index}")

    for node in nodes:
        scenario_peak[node.node_id] = solver.NumVar(0.0, solver.infinity(), f"peak_{node.node_id}")
        onpeak_peak[node.node_id] = solver.NumVar(0.0, solver.infinity(), f"onpeak_peak_{node.node_id}")
        for vehicle_index, vehicle in enumerate(scenario.vehicles):
            unmet[node.node_id, vehicle_index] = solver.NumVar(
                0.0, solver.infinity(), f"unmet_{node.node_id}_{vehicle_index}"
            )
            for local_position, slot in enumerate(local_slots):
                upper_bound = 0.0
                predicted_arrival = int(node.arrival_slots[vehicle_index])
                if predicted_arrival <= slot < vehicle.departure_slot:
                    upper_bound = vehicle.max_charge_kw
                scenario_power[node.node_id, vehicle_index, local_position] = solver.NumVar(
                    0.0,
                    upper_bound,
                    f"p_{node.node_id}_{vehicle_index}_{slot}",
                )

    for node in nodes:
        for local_position, slot in enumerate(local_slots):
            slot_sum = solver.Sum(
                scenario_power[node.node_id, vehicle_index, local_position] for vehicle_index in range(vehicle_count)
            )
            capacity = float(node.site_capacity_kw[local_position])
            solver.Add(slot_sum <= capacity)
            solver.Add(scenario_peak[node.node_id] >= slot_sum)
            if onpeak_mask[slot]:
                solver.Add(onpeak_peak[node.node_id] >= slot_sum)
            if local_position == 0:
                for vehicle_index in range(vehicle_count):
                    solver.Add(
                        scenario_power[node.node_id, vehicle_index, local_position] == first_stage[vehicle_index]
                    )

    for node in nodes:
        for vehicle_index, vehicle in enumerate(scenario.vehicles):
            delivered_energy = solver.Sum(
                scenario_power[node.node_id, vehicle_index, local_position]
                * scenario.dt_hours
                * vehicle.charge_efficiency
                for local_position in range(local_width)
            )
            solver.Add(delivered_energy + unmet[node.node_id, vehicle_index] >= float(remaining_energy[vehicle_index]))

    expected_cost = solver.Sum(
        node.probability
        * scenario_power[node.node_id, vehicle_index, local_position]
        * scenario.dt_hours
        * (float(scenario.tariff_per_kwh[slot]) + float(degradation_penalties[vehicle_index]))
        for node in nodes
        for vehicle_index in range(vehicle_count)
        for local_position, slot in enumerate(local_slots)
    )
    expected_demand = solver.Sum(
        node.probability
        * (
            scenario_peak[node.node_id] * float(scenario.demand_charge_per_kw)
            + onpeak_peak[node.node_id] * float(scenario.onpeak_demand_charge_per_kw)
        )
        for node in nodes
    )
    expected_peak_penalty = solver.Sum(node.probability * peak_penalty * scenario_peak[node.node_id] for node in nodes)
    unmet_cost = solver.Sum(
        node.probability
        * unmet[node.node_id, vehicle_index]
        * unmet_penalty
        * PRIORITY_WEIGHTS[scenario.vehicles[vehicle_index].priority_class]
        for node in nodes
        for vehicle_index in range(vehicle_count)
    )
    solver.Minimize(expected_cost + expected_demand + expected_peak_penalty + unmet_cost)
    status_code = solver.Solve()
    status = solver_status_name(status_code)

    if status_code in {pywraplp.Solver.INFEASIBLE, pywraplp.Solver.UNBOUNDED, pywraplp.Solver.ABNORMAL}:
        _logger.warning(
            "Tree subproblem solver returned %s at slot %d with %d nodes",
            status,
            current_slot,
            len(nodes),
        )
        return np.zeros(vehicle_count, dtype=float), status

    dispatch = np.array(
        [first_stage[vehicle_index].solution_value() for vehicle_index in range(vehicle_count)], dtype=float
    )
    return dispatch, status
