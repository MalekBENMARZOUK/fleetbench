from __future__ import annotations

import logging
from time import perf_counter

import numpy as np

from ev_fleet_benchmark.economics import approximate_degradation_penalty_per_vehicle, onpeak_mask_from_scenario
from ev_fleet_benchmark.methods.base import ScheduleMethod
from ev_fleet_benchmark.methods.config import NaiveMethodConfig
from ev_fleet_benchmark.model import PRIORITY_WEIGHTS, Scenario, SchedulePlan

_logger = logging.getLogger(__name__)


class NaiveBaselineMethod(ScheduleMethod):
    name = "naive_baseline"

    def __init__(
        self, config: NaiveMethodConfig | None = None, *, onpeak_demand_bias_divisor: float | None = None
    ) -> None:
        base_config = config or NaiveMethodConfig()
        if onpeak_demand_bias_divisor is not None:
            base_config = NaiveMethodConfig(onpeak_demand_bias_divisor=onpeak_demand_bias_divisor)
        self.config = base_config

    def solve(self, scenario: Scenario) -> SchedulePlan:
        start = perf_counter()
        _logger.debug(
            "Solving %s on %s (%d vehicles, %d slots)",
            self.name,
            scenario.name,
            len(scenario.vehicles),
            scenario.horizon_slots,
        )
        power = np.zeros((len(scenario.vehicles), scenario.horizon_slots), dtype=float)
        remaining_capacity = scenario.site_capacity_kw.astype(float).copy()
        degradation_penalties = approximate_degradation_penalty_per_vehicle(scenario)
        onpeak_mask = onpeak_mask_from_scenario(scenario)

        ordered_indices = sorted(
            range(len(scenario.vehicles)),
            key=lambda index: (
                scenario.vehicles[index].arrival_slot,
                -PRIORITY_WEIGHTS[scenario.vehicles[index].priority_class],
                scenario.vehicles[index].departure_slot,
            ),
        )

        for vehicle_index in ordered_indices:
            vehicle = scenario.vehicles[vehicle_index]
            remaining_energy = vehicle.required_energy_kwh
            candidate_slots = sorted(
                range(vehicle.arrival_slot, vehicle.departure_slot),
                key=lambda slot: (
                    scenario.tariff_per_kwh[slot]
                    + degradation_penalties[vehicle_index]
                    + (
                        scenario.onpeak_demand_charge_per_kw / self.config.onpeak_demand_bias_divisor
                        if onpeak_mask[slot]
                        else 0.0
                    ),
                    remaining_capacity[slot] <= 0.0,
                    slot,
                ),
            )
            for slot in candidate_slots:
                if remaining_energy <= 1e-9:
                    break
                deliverable_kw = min(vehicle.max_charge_kw, remaining_capacity[slot])
                if deliverable_kw <= 0:
                    continue
                energy_if_full = deliverable_kw * scenario.dt_hours * vehicle.charge_efficiency
                if energy_if_full > remaining_energy:
                    deliverable_kw = remaining_energy / (scenario.dt_hours * vehicle.charge_efficiency)
                power[vehicle_index, slot] = deliverable_kw
                remaining_capacity[slot] -= deliverable_kw
                remaining_energy -= deliverable_kw * scenario.dt_hours * vehicle.charge_efficiency

        return SchedulePlan(
            method_name=self.name,
            power_kw=power,
            solve_time_s=perf_counter() - start,
            status="heuristic",
        )
