from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ev_fleet_benchmark.model import Scenario

_logger = logging.getLogger(__name__)


def battery_degradation_cost(power_kw: np.ndarray, scenario: Scenario) -> float:
    clipped_power = np.clip(power_kw, 0.0, None)
    if not np.any(clipped_power > 0.0):
        return 0.0

    dt = scenario.dt_hours
    threshold = scenario.degradation_high_soc_threshold
    denominator = max(1e-9, 1.0 - threshold)

    caps = np.array([v.battery_capacity_kwh for v in scenario.vehicles], dtype=float)
    effs = np.array([v.charge_efficiency for v in scenario.vehicles], dtype=float)
    init_socs = np.array([v.initial_soc for v in scenario.vehicles], dtype=float)

    valid = caps >= 1e-6
    if not np.any(valid):
        return 0.0

    safe_caps = np.where(valid, caps, 1.0)[:, np.newaxis]
    energies = clipped_power * dt * effs[:, np.newaxis]
    c_rates = clipped_power / safe_caps

    cumulative_energy = np.cumsum(energies, axis=1)
    cum_before = np.zeros_like(cumulative_energy)
    cum_before[:, 1:] = cumulative_energy[:, :-1]
    soc_before = np.clip(init_socs[:, np.newaxis] + cum_before / safe_caps, 0.0, 1.0)

    midpoint_soc = np.clip(soc_before + 0.5 * energies / safe_caps, 0.0, 1.0)
    high_soc_exposure = np.clip((midpoint_soc - threshold) / denominator, 0.0, None)

    marginal_arr = scenario.degradation_cost_per_kwh * (
        1.0 + scenario.degradation_high_soc_multiplier * high_soc_exposure
    )
    marginal_arr *= 1.0 + scenario.degradation_c_rate_coefficient * c_rates

    return float(np.sum(np.where(valid[:, np.newaxis], energies * marginal_arr, 0.0)))


def demand_charge_cost(
    site_power_kw: np.ndarray,
    onpeak_mask: np.ndarray,
    all_day_demand_charge_per_kw: float,
    onpeak_demand_charge_per_kw: float,
) -> tuple[float, float, float]:
    peak_power = float(np.max(site_power_kw)) if len(site_power_kw) else 0.0
    onpeak_peak_power = float(np.max(site_power_kw[onpeak_mask])) if np.any(onpeak_mask) else 0.0
    all_day_cost = peak_power * all_day_demand_charge_per_kw
    onpeak_cost = onpeak_peak_power * onpeak_demand_charge_per_kw
    return all_day_cost, onpeak_cost, all_day_cost + onpeak_cost


def approximate_degradation_penalty_per_vehicle(scenario: Scenario) -> np.ndarray:
    penalties = []
    for vehicle in scenario.vehicles:
        cap = vehicle.battery_capacity_kwh
        c_rate = vehicle.max_charge_kw / cap if cap >= 1e-6 else 0.0
        soc_range = max(1e-9, 1.0 - scenario.degradation_high_soc_threshold)
        high_soc_exposure = max(0.0, vehicle.target_soc - scenario.degradation_high_soc_threshold) / soc_range
        penalty = scenario.degradation_cost_per_kwh
        penalty *= 1.0 + scenario.degradation_c_rate_coefficient * c_rate
        penalty *= 1.0 + scenario.degradation_high_soc_multiplier * high_soc_exposure
        penalties.append(penalty)
    return np.array(penalties, dtype=float)


def onpeak_mask_from_scenario(scenario: Scenario) -> np.ndarray:
    mask = np.zeros(scenario.horizon_slots, dtype=bool)
    economic_model = scenario.metadata.get("economic_model")
    if economic_model is None:
        _logger.debug("Scenario %s has no 'economic_model' in metadata — on-peak mask will be empty", scenario.name)
        return mask
    onpeak_slots = economic_model.get("onpeak_slots")
    if onpeak_slots is None:
        _logger.debug("Scenario %s has no 'onpeak_slots' in economic_model — on-peak mask will be empty", scenario.name)
        return mask
    for slot in onpeak_slots:
        slot_index = int(slot)
        if 0 <= slot_index < scenario.horizon_slots:
            mask[slot_index] = True
    return mask
