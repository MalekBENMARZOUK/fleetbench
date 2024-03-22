from __future__ import annotations

import logging
from time import perf_counter

import numpy as np
from ortools.linear_solver import pywraplp

from ev_fleet_benchmark.economics import approximate_degradation_penalty_per_vehicle, onpeak_mask_from_scenario
from ev_fleet_benchmark.exceptions import SolverUnavailableError
from ev_fleet_benchmark.methods.base import ScheduleMethod
from ev_fleet_benchmark.methods.config import OptimizationPenaltyConfig, with_optimization_overrides
from ev_fleet_benchmark.model import PRIORITY_WEIGHTS, Scenario, SchedulePlan

_DEFAULT_SOLVER_TIME_LIMIT_S = 120.0

_logger = logging.getLogger(__name__)


class OptimizationMethod(ScheduleMethod):
    name = "optimization_ortools"

    def __init__(
        self,
        config: OptimizationPenaltyConfig | None = None,
        *,
        peak_penalty: float | None = None,
        unmet_penalty: float | None = None,
    ) -> None:
        self.config = with_optimization_overrides(
            config or OptimizationPenaltyConfig(), peak_penalty=peak_penalty, unmet_penalty=unmet_penalty
        )

    def solve(self, scenario: Scenario) -> SchedulePlan:
        start = perf_counter()
        _logger.debug(
            "Solving %s on %s (%d vehicles, %d slots)",
            self.name,
            scenario.name,
            len(scenario.vehicles),
            scenario.horizon_slots,
        )
        remaining_energy = np.array([vehicle.required_energy_kwh for vehicle in scenario.vehicles], dtype=float)
        power_matrix, status, objective_value = solve_charging_subproblem(
            scenario=scenario,
            remaining_energy_kwh=remaining_energy,
            start_slot=0,
            end_slot=scenario.horizon_slots,
            peak_penalty=self.config.peak_penalty,
            unmet_penalty=self.config.unmet_penalty,
        )

        return SchedulePlan(
            method_name=self.name,
            power_kw=power_matrix,
            solve_time_s=perf_counter() - start,
            status=status,
            metadata={"objective_value": objective_value},
        )


def solve_charging_subproblem(
    scenario: Scenario,
    remaining_energy_kwh: np.ndarray,
    start_slot: int,
    end_slot: int,
    peak_penalty: float,
    unmet_penalty: float,
    forecast_arrival_slots: np.ndarray | None = None,
    candidate_vehicle_indices: list[int] | None = None,
    solver_time_limit_s: float = _DEFAULT_SOLVER_TIME_LIMIT_S,
) -> tuple[np.ndarray, str, float]:
    solver = pywraplp.Solver.CreateSolver("GLOP") or pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise SolverUnavailableError("OR-Tools linear solver is unavailable — ensure ortools is installed correctly")

    if solver_time_limit_s > 0:
        solver.SetTimeLimit(int(solver_time_limit_s * 1000))

    vehicle_count = len(scenario.vehicles)
    local_slots = list(range(start_slot, end_slot))
    local_width = len(local_slots)
    power_matrix = np.zeros((vehicle_count, local_width), dtype=float)
    if local_width == 0:
        return power_matrix, "empty_horizon", 0.0

    if candidate_vehicle_indices is None:
        candidate_vehicle_indices = list(range(vehicle_count))
    candidate_set = set(candidate_vehicle_indices)
    if forecast_arrival_slots is None:
        forecast_arrival_slots = np.array([vehicle.arrival_slot for vehicle in scenario.vehicles], dtype=int)

    power = {}
    unmet = {}
    peak = solver.NumVar(0.0, solver.infinity(), "peak")
    onpeak_peak = solver.NumVar(0.0, solver.infinity(), "onpeak_peak")
    degradation_penalties = approximate_degradation_penalty_per_vehicle(scenario)
    onpeak_mask = onpeak_mask_from_scenario(scenario)

    for vehicle_index, vehicle in enumerate(scenario.vehicles):
        if vehicle_index not in candidate_set or remaining_energy_kwh[vehicle_index] <= 1e-9:
            continue
        unmet[vehicle_index] = solver.NumVar(0.0, solver.infinity(), f"unmet_{vehicle_index}")
        predicted_arrival = int(forecast_arrival_slots[vehicle_index])
        for local_position, slot in enumerate(local_slots):
            upper_bound = vehicle.max_charge_kw if predicted_arrival <= slot < vehicle.departure_slot else 0.0
            power[vehicle_index, local_position] = solver.NumVar(0.0, upper_bound, f"p_{vehicle_index}_{slot}")

    if not power:
        return power_matrix, "no_candidate_load", 0.0

    for local_position, slot in enumerate(local_slots):
        slot_sum = solver.Sum(
            power[vehicle_index, local_position]
            for vehicle_index in candidate_set
            if (vehicle_index, local_position) in power
        )
        solver.Add(slot_sum <= float(scenario.site_capacity_kw[slot]))
        solver.Add(peak >= slot_sum)
        if onpeak_mask[slot]:
            solver.Add(onpeak_peak >= slot_sum)

    for vehicle_index in candidate_set:
        if vehicle_index not in unmet:
            continue
        vehicle = scenario.vehicles[vehicle_index]
        delivered_energy = solver.Sum(
            power[vehicle_index, local_position] * scenario.dt_hours * vehicle.charge_efficiency
            for local_position in range(local_width)
            if (vehicle_index, local_position) in power
        )
        solver.Add(delivered_energy + unmet[vehicle_index] >= float(remaining_energy_kwh[vehicle_index]))

    total_cost = solver.Sum(
        power[vehicle_index, local_position]
        * scenario.dt_hours
        * (float(scenario.tariff_per_kwh[slot]) + float(degradation_penalties[vehicle_index]))
        for local_position, slot in enumerate(local_slots)
        for vehicle_index in candidate_set
        if (vehicle_index, local_position) in power
    )
    demand_cost = peak * float(scenario.demand_charge_per_kw) + onpeak_peak * float(
        scenario.onpeak_demand_charge_per_kw
    )
    unmet_cost = solver.Sum(
        unmet[vehicle_index] * unmet_penalty * PRIORITY_WEIGHTS[scenario.vehicles[vehicle_index].priority_class]
        for vehicle_index in candidate_set
        if vehicle_index in unmet
    )
    solver.Minimize(total_cost + unmet_cost + demand_cost + peak_penalty * peak)
    status_code = solver.Solve()
    status = solver_status_name(status_code)

    if status_code in {pywraplp.Solver.INFEASIBLE, pywraplp.Solver.UNBOUNDED, pywraplp.Solver.ABNORMAL}:
        _logger.warning(
            "Solver returned %s for slots [%d, %d) with %d candidate vehicles",
            status,
            start_slot,
            end_slot,
            len(candidate_set),
        )
        return power_matrix, status, 0.0

    objective_value = (
        solver.Objective().Value() if status_code in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE} else 0.0
    )

    for vehicle_index in candidate_set:
        for local_position in range(local_width):
            if (vehicle_index, local_position) in power:
                power_matrix[vehicle_index, local_position] = power[vehicle_index, local_position].solution_value()

    return power_matrix, status, objective_value


def solver_status_name(status_code: int) -> str:
    status_lookup = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
        pywraplp.Solver.ABNORMAL: "abnormal",
        pywraplp.Solver.NOT_SOLVED: "not_solved",
    }
    return status_lookup.get(status_code, f"status_{status_code}")
