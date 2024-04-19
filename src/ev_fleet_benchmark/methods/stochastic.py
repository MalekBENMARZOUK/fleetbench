from __future__ import annotations

import logging
from dataclasses import replace
from time import perf_counter

import numpy as np

from ev_fleet_benchmark.methods.base import ScheduleMethod
from ev_fleet_benchmark.methods.config import StochasticMethodConfig
from ev_fleet_benchmark.model import PRIORITY_WEIGHTS, Scenario, SchedulePlan
from ev_fleet_benchmark.simulator import criticality_score

_logger = logging.getLogger(__name__)


class StochasticAnticipatoryMethod(ScheduleMethod):
    name = "stochastic_anticipatory"

    def __init__(
        self,
        config: StochasticMethodConfig | None = None,
        *,
        lookahead_slots: int | None = None,
        sample_count: int | None = None,
        reserve_quantile: float | None = None,
        reserve_capacity_limit_fraction: float | None = None,
        demand_guard_fraction: float | None = None,
        demand_guard_divisor: float | None = None,
        tariff_headroom_weight: float | None = None,
        tiebreak_noise_scale: float | None = None,
    ) -> None:
        base = config or StochasticMethodConfig()
        overrides = {
            k: v
            for k, v in {
                "lookahead_slots": lookahead_slots,
                "sample_count": sample_count,
                "reserve_quantile": reserve_quantile,
                "reserve_capacity_limit_fraction": reserve_capacity_limit_fraction,
                "demand_guard_fraction": demand_guard_fraction,
                "demand_guard_divisor": demand_guard_divisor,
                "tariff_headroom_weight": tariff_headroom_weight,
                "tiebreak_noise_scale": tiebreak_noise_scale,
            }.items()
            if v is not None
        }
        self.config = replace(base, **overrides) if overrides else base  # type: ignore[arg-type]

    def solve(self, scenario: Scenario) -> SchedulePlan:
        start = perf_counter()
        _logger.debug(
            "Solving %s on %s (%d vehicles, %d slots)",
            self.name,
            scenario.name,
            len(scenario.vehicles),
            scenario.horizon_slots,
        )
        rng = np.random.default_rng(scenario.seed + 7919)
        vehicle_count = len(scenario.vehicles)
        power = np.zeros((vehicle_count, scenario.horizon_slots), dtype=float)
        remaining_energy = np.array([vehicle.required_energy_kwh for vehicle in scenario.vehicles], dtype=float)
        max_tariff = float(np.max(scenario.tariff_per_kwh)) if len(scenario.tariff_per_kwh) else 1.0

        for slot in range(scenario.horizon_slots):
            if np.all(remaining_energy <= 1e-9):
                break
            active_indices = [
                vehicle_index
                for vehicle_index, vehicle in enumerate(scenario.vehicles)
                if remaining_energy[vehicle_index] > 1e-9 and vehicle.arrival_slot <= slot < vehicle.departure_slot
            ]
            if not active_indices:
                continue

            reserve_kw = self._estimate_reserve(scenario, remaining_energy, slot, rng)
            demand_guard = min(
                float(scenario.site_capacity_kw[slot]) * self.config.demand_guard_fraction,
                float(scenario.demand_charge_per_kw) / self.config.demand_guard_divisor,
            )
            available_capacity = max(0.0, float(scenario.site_capacity_kw[slot]) - reserve_kw - demand_guard)
            tariff_multiplier = 1.0 + self.config.tariff_headroom_weight * (
                1.0 - float(scenario.tariff_per_kwh[slot]) / max(max_tariff, 1e-9)
            )
            economic_multiplier = 1.0 / (
                1.0 + float(scenario.tariff_per_kwh[slot]) + float(scenario.degradation_cost_per_kwh)
            )
            ranked_candidates: list[tuple[float, int]] = []
            for vehicle_index in active_indices:
                vehicle = scenario.vehicles[vehicle_index]
                score = (
                    criticality_score(
                        remaining_energy_kwh=float(remaining_energy[vehicle_index]),
                        remaining_slots=vehicle.departure_slot - slot,
                        max_charge_kw=vehicle.max_charge_kw,
                        dt_hours=scenario.dt_hours,
                        priority_weight=PRIORITY_WEIGHTS[vehicle.priority_class],
                    )
                    * tariff_multiplier
                    * economic_multiplier
                )
                ranked_candidates.append((score + rng.uniform(0.0, self.config.tiebreak_noise_scale), vehicle_index))

            for _, vehicle_index in sorted(ranked_candidates, reverse=True):
                if available_capacity <= 1e-9:
                    break
                vehicle = scenario.vehicles[vehicle_index]
                dispatch_kw = min(vehicle.max_charge_kw, available_capacity)
                max_needed_kw = remaining_energy[vehicle_index] / (scenario.dt_hours * vehicle.charge_efficiency)
                dispatch_kw = min(dispatch_kw, max_needed_kw)
                if dispatch_kw <= 0.0:
                    continue
                power[vehicle_index, slot] = dispatch_kw
                available_capacity -= dispatch_kw
                remaining_energy[vehicle_index] -= dispatch_kw * scenario.dt_hours * vehicle.charge_efficiency

        remaining_energy = np.clip(remaining_energy, 0.0, None)

        return SchedulePlan(
            method_name=self.name,
            power_kw=power,
            solve_time_s=perf_counter() - start,
            status="stochastic_heuristic",
            metadata={
                "lookahead_slots": self.config.lookahead_slots,
                "sample_count": self.config.sample_count,
                "reserve_quantile": self.config.reserve_quantile,
            },
        )

    def _estimate_reserve(
        self,
        scenario: Scenario,
        remaining_energy: np.ndarray,
        slot: int,
        rng: np.random.Generator,
    ) -> float:
        uncertainty_model = scenario.metadata.get("uncertainty_model", {})
        delay_probability = float(uncertainty_model.get("arrival_delay_probability", 0.0))
        delay_low, delay_high = uncertainty_model.get("arrival_delay_slot_range", (0, 0))
        derate_probability = float(uncertainty_model.get("site_derate_probability", 0.0))
        severity_low, severity_high = uncertainty_model.get("site_derate_severity_range", (0.0, 0.0))
        horizon_end = min(scenario.horizon_slots, slot + self.config.lookahead_slots)
        if horizon_end <= slot + 1:
            return 0.0

        future_requirements: list[float] = []
        for _ in range(self.config.sample_count):
            sampled_required_kw = 0.0
            for vehicle_index, vehicle in enumerate(scenario.vehicles):
                if (
                    remaining_energy[vehicle_index] <= 1e-9
                    or vehicle.arrival_slot <= slot
                    or vehicle.departure_slot <= slot
                ):
                    continue
                planned_arrival = (
                    vehicle.planned_arrival_slot if vehicle.planned_arrival_slot is not None else vehicle.arrival_slot
                )
                sampled_delay = 0
                if delay_high > 0 and rng.random() < delay_probability:
                    sampled_delay = int(rng.integers(delay_low, delay_high + 1))
                sampled_arrival = max(slot + 1, min(scenario.horizon_slots - 1, planned_arrival + sampled_delay))
                remaining_slots = max(0, min(vehicle.departure_slot, horizon_end) - sampled_arrival)
                if remaining_slots == 0:
                    continue
                sampled_required_kw += min(
                    vehicle.max_charge_kw,
                    remaining_energy[vehicle_index] / (remaining_slots * scenario.dt_hours * vehicle.charge_efficiency),
                )
            future_requirements.append(sampled_required_kw)

        if not future_requirements:
            return 0.0

        derate_multiplier = 1.0 + derate_probability * (severity_low + severity_high) / 2.0
        reserve_kw = float(np.quantile(future_requirements, self.config.reserve_quantile)) * derate_multiplier
        return float(
            min(reserve_kw, float(scenario.site_capacity_kw[slot]) * self.config.reserve_capacity_limit_fraction)
        )
