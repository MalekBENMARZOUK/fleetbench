from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace
from time import perf_counter

import numpy as np

from ev_fleet_benchmark.methods.base import ScheduleMethod
from ev_fleet_benchmark.methods.config import RollingHorizonMethodConfig, with_optimization_overrides
from ev_fleet_benchmark.methods.optimization import solve_charging_subproblem
from ev_fleet_benchmark.model import Scenario, SchedulePlan

_logger = logging.getLogger(__name__)


class RollingHorizonOptimizationMethod(ScheduleMethod):
    name = "rolling_horizon_ortools"

    def __init__(
        self,
        config: RollingHorizonMethodConfig | None = None,
        *,
        lookahead_slots: int | None = None,
        peak_penalty: float | None = None,
        unmet_penalty: float | None = None,
        peak_guard_fraction: float | None = None,
        demand_guard_divisor: float | None = None,
    ) -> None:
        base = config or RollingHorizonMethodConfig()
        penalties = with_optimization_overrides(base.penalties, peak_penalty=peak_penalty, unmet_penalty=unmet_penalty)
        overrides = {
            k: v
            for k, v in {
                "lookahead_slots": lookahead_slots,
                "peak_guard_fraction": peak_guard_fraction,
                "demand_guard_divisor": demand_guard_divisor,
            }.items()
            if v is not None
        }
        self.config = replace(base, penalties=penalties, **overrides)  # type: ignore[arg-type]

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
        objectives: list[float] = []

        for current_slot in range(scenario.horizon_slots):
            if np.all(remaining_energy <= 1e-9):
                break
            horizon_end = min(scenario.horizon_slots, current_slot + self.config.lookahead_slots)
            candidate_indices = [
                vehicle_index
                for vehicle_index, vehicle in enumerate(scenario.vehicles)
                if remaining_energy[vehicle_index] > 1e-9
                and vehicle.departure_slot > current_slot
                and _forecast_arrival_slot(vehicle.arrival_slot, vehicle.planned_arrival_slot, current_slot)
                < horizon_end
            ]
            if not candidate_indices:
                continue

            forecast_arrivals = np.array(
                [
                    _forecast_arrival_slot(vehicle.arrival_slot, vehicle.planned_arrival_slot, current_slot)
                    for vehicle in scenario.vehicles
                ],
                dtype=int,
            )
            local_power, status, objective_value = solve_charging_subproblem(
                scenario=scenario,
                remaining_energy_kwh=remaining_energy,
                start_slot=current_slot,
                end_slot=horizon_end,
                peak_penalty=self.config.penalties.peak_penalty,
                unmet_penalty=self.config.penalties.unmet_penalty,
                forecast_arrival_slots=forecast_arrivals,
                candidate_vehicle_indices=candidate_indices,
            )
            statuses.append(status)
            objectives.append(objective_value)

            slot_dispatch = local_power[:, 0].copy()
            for vehicle_index, vehicle in enumerate(scenario.vehicles):
                if not (vehicle.arrival_slot <= current_slot < vehicle.departure_slot):
                    slot_dispatch[vehicle_index] = 0.0

            total_dispatch = float(slot_dispatch.sum())
            capacity = float(scenario.site_capacity_kw[current_slot])
            if total_dispatch > capacity + 1e-9 and total_dispatch > 0.0:
                slot_dispatch *= capacity / total_dispatch
                total_dispatch = float(slot_dispatch.sum())
            peak_guard = min(
                capacity * self.config.peak_guard_fraction,
                float(scenario.demand_charge_per_kw) / self.config.demand_guard_divisor,
            )
            if total_dispatch > capacity - peak_guard and total_dispatch > 0.0:
                slot_dispatch *= max(0.0, capacity - peak_guard) / total_dispatch

            power[:, current_slot] = slot_dispatch
            remaining_energy -= slot_dispatch * scenario.dt_hours * efficiencies
            remaining_energy = np.clip(remaining_energy, 0.0, None)

        return SchedulePlan(
            method_name=self.name,
            power_kw=power,
            solve_time_s=perf_counter() - start,
            status="receding_horizon",
            metadata={
                "lookahead_slots": self.config.lookahead_slots,
                "subproblem_status_counts": dict(Counter(statuses)),
                "mean_subproblem_objective": float(np.mean(objectives)) if objectives else 0.0,
            },
        )


def _forecast_arrival_slot(actual_arrival: int, planned_arrival: int | None, current_slot: int) -> int:
    if actual_arrival <= current_slot:
        return actual_arrival
    if planned_arrival is None:
        return actual_arrival
    if planned_arrival <= current_slot:
        return current_slot + 1
    return planned_arrival
