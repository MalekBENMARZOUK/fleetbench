from __future__ import annotations

import logging
from dataclasses import replace
from time import perf_counter

import numpy as np

from ev_fleet_benchmark.economics import approximate_degradation_penalty_per_vehicle, onpeak_mask_from_scenario
from ev_fleet_benchmark.methods.base import ScheduleMethod
from ev_fleet_benchmark.methods.config import GreedyMethodConfig
from ev_fleet_benchmark.model import PRIORITY_WEIGHTS, Scenario, SchedulePlan
from ev_fleet_benchmark.simulator import criticality_score

_logger = logging.getLogger(__name__)


class GreedyUrgencyMethod(ScheduleMethod):
    name = "greedy_urgency"

    def __init__(
        self,
        config: GreedyMethodConfig | None = None,
        *,
        capacity_reserve_fraction: float | None = None,
        all_day_demand_guard_divisor: float | None = None,
        onpeak_demand_guard_divisor: float | None = None,
        onpeak_score_discount: float | None = None,
    ) -> None:
        base = config or GreedyMethodConfig()
        overrides = {
            k: v
            for k, v in {
                "capacity_reserve_fraction": capacity_reserve_fraction,
                "all_day_demand_guard_divisor": all_day_demand_guard_divisor,
                "onpeak_demand_guard_divisor": onpeak_demand_guard_divisor,
                "onpeak_score_discount": onpeak_score_discount,
            }.items()
            if v is not None
        }
        self.config = replace(base, **overrides) if overrides else base

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
        degradation_penalties = approximate_degradation_penalty_per_vehicle(scenario)
        onpeak_mask = onpeak_mask_from_scenario(scenario)

        for slot in range(scenario.horizon_slots):
            available_capacity = float(scenario.site_capacity_kw[slot])
            demand_guard = scenario.demand_charge_per_kw / self.config.all_day_demand_guard_divisor
            if onpeak_mask[slot]:
                demand_guard += scenario.onpeak_demand_charge_per_kw / self.config.onpeak_demand_guard_divisor
            effective_capacity = max(
                0.0,
                available_capacity - min(available_capacity * self.config.capacity_reserve_fraction, demand_guard),
            )
            candidates: list[tuple[float, int]] = []
            for vehicle_index, vehicle in enumerate(scenario.vehicles):
                if remaining_energy[vehicle_index] <= 1e-9:
                    continue
                if not (vehicle.arrival_slot <= slot < vehicle.departure_slot):
                    continue
                remaining_slots = vehicle.departure_slot - slot
                score = criticality_score(
                    remaining_energy_kwh=float(remaining_energy[vehicle_index]),
                    remaining_slots=remaining_slots,
                    max_charge_kw=vehicle.max_charge_kw,
                    dt_hours=scenario.dt_hours,
                    priority_weight=PRIORITY_WEIGHTS[vehicle.priority_class],
                )
                if onpeak_mask[slot]:
                    score *= self.config.onpeak_score_discount
                score /= 1.0 + float(scenario.tariff_per_kwh[slot]) + float(degradation_penalties[vehicle_index])
                candidates.append((score, vehicle_index))

            for _, vehicle_index in sorted(candidates, reverse=True):
                if effective_capacity <= 1e-9:
                    break
                vehicle = scenario.vehicles[vehicle_index]
                charge_kw = min(vehicle.max_charge_kw, effective_capacity)
                max_needed_kw = remaining_energy[vehicle_index] / (scenario.dt_hours * vehicle.charge_efficiency)
                charge_kw = min(charge_kw, max_needed_kw)
                if charge_kw <= 0:
                    continue
                power[vehicle_index, slot] = charge_kw
                effective_capacity -= charge_kw
                remaining_energy[vehicle_index] -= charge_kw * scenario.dt_hours * vehicle.charge_efficiency

        remaining_energy = np.clip(remaining_energy, 0.0, None)

        return SchedulePlan(
            method_name=self.name,
            power_kw=power,
            solve_time_s=perf_counter() - start,
            status="heuristic",
        )
