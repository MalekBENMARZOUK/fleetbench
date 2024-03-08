from __future__ import annotations

import numpy as np

from ev_fleet_benchmark.economics import battery_degradation_cost, demand_charge_cost, onpeak_mask_from_scenario
from ev_fleet_benchmark.model import (
    ENERGY_TOLERANCE_KWH,
    SCHEDULE_TOLERANCE_KW,
    EvaluationResult,
    Scenario,
    SchedulePlan,
)


def evaluate_plan(
    scenario: Scenario, plan: SchedulePlan, tolerance_kwh: float = ENERGY_TOLERANCE_KWH
) -> EvaluationResult:
    if tolerance_kwh < 0.0:
        raise ValueError("tolerance_kwh must be non-negative")
    vehicle_count = len(scenario.vehicles)
    expected_shape = (vehicle_count, scenario.horizon_slots)
    if plan.power_kw.shape != expected_shape:
        raise ValueError(f"Schedule for {plan.method_name} has shape {plan.power_kw.shape}, expected {expected_shape}")
    if np.any(plan.power_kw < -SCHEDULE_TOLERANCE_KW):
        raise ValueError(f"Schedule for {plan.method_name} contains negative charging power")

    requested = np.clip(plan.power_kw.astype(float, copy=True), 0.0, None)
    active_mask = np.zeros_like(requested, dtype=bool)

    for vehicle_index, vehicle in enumerate(scenario.vehicles):
        active_mask[vehicle_index, vehicle.arrival_slot : vehicle.departure_slot] = True
        requested[vehicle_index, ~active_mask[vehicle_index]] = 0.0
        requested[vehicle_index] = np.minimum(requested[vehicle_index], vehicle.max_charge_kw)

    requested_site_kw = requested.sum(axis=0)
    capacity = scenario.site_capacity_kw.astype(float)
    scale = np.ones(scenario.horizon_slots)
    overloaded = requested_site_kw > capacity + SCHEDULE_TOLERANCE_KW
    scale[overloaded] = capacity[overloaded] / requested_site_kw[overloaded]
    actual = requested * scale
    actual_site_kw = actual.sum(axis=0)

    delivered_energy_values: list[float] = []
    required_energy_values: list[float] = []
    unmet_energy_values: list[float] = []
    served_fraction_values: list[float] = []

    for vehicle_index, vehicle in enumerate(scenario.vehicles):
        delivered = actual[vehicle_index].sum() * scenario.dt_hours * vehicle.charge_efficiency
        required = vehicle.required_energy_kwh
        unmet = max(0.0, required - delivered)
        delivered_energy_values.append(delivered)
        required_energy_values.append(required)
        unmet_energy_values.append(unmet)
        served_fraction_values.append(1.0 if required <= tolerance_kwh else min(1.0, delivered / required))

    delivered_energy_kwh = np.array(delivered_energy_values, dtype=float)
    required_energy_kwh = np.array(required_energy_values, dtype=float)
    unmet_energy_kwh = np.array(unmet_energy_values, dtype=float)
    served_fraction = np.array(served_fraction_values, dtype=float)

    energy_cost = float(np.sum(actual_site_kw * scenario.dt_hours * scenario.tariff_per_kwh))
    onpeak_mask = onpeak_mask_from_scenario(scenario)
    all_day_demand_cost, onpeak_demand_cost, demand_cost = demand_charge_cost(
        actual_site_kw,
        onpeak_mask=onpeak_mask,
        all_day_demand_charge_per_kw=scenario.demand_charge_per_kw,
        onpeak_demand_charge_per_kw=scenario.onpeak_demand_charge_per_kw,
    )
    degradation_cost = battery_degradation_cost(actual, scenario)
    total_economic_cost = energy_cost + demand_cost + degradation_cost
    peak_power = float(np.max(actual_site_kw)) if len(actual_site_kw) else 0.0
    load_variance = float(np.var(actual_site_kw)) if len(actual_site_kw) else 0.0
    max_capacity_violation = float(np.max(np.maximum(requested_site_kw - capacity, 0.0))) if len(capacity) else 0.0
    feasible = bool((max_capacity_violation <= SCHEDULE_TOLERANCE_KW) and np.all(unmet_energy_kwh <= tolerance_kwh))

    summary = {
        "scenario_name": scenario.name,
        "family": scenario.family,
        "seed": scenario.seed,
        "method": plan.method_name,
        "solve_time_s": float(plan.solve_time_s),
        "status": plan.status,
        "fleet_size": len(scenario.vehicles),
        "feasibility_rate": 1.0 if feasible else 0.0,
        "unmet_charge_demand_kwh": float(unmet_energy_kwh.sum()),
        "delivered_energy_kwh": float(delivered_energy_kwh.sum()),
        "required_energy_kwh": float(required_energy_kwh.sum()),
        "service_level": float(np.mean(served_fraction)),
        "energy_cost": energy_cost,
        "all_day_demand_charge_cost": all_day_demand_cost,
        "onpeak_demand_charge_cost": onpeak_demand_cost,
        "demand_charge_cost": demand_cost,
        "battery_degradation_cost": degradation_cost,
        "total_charging_cost": total_economic_cost,
        "peak_site_power_kw": peak_power,
        "load_variance_kw2": load_variance,
        "max_capacity_violation_kw": max_capacity_violation,
        "mean_tariff_per_kwh": float(np.mean(scenario.tariff_per_kwh)),
        "demand_charge_per_kw": float(scenario.demand_charge_per_kw),
        "onpeak_demand_charge_per_kw": float(scenario.onpeak_demand_charge_per_kw),
        "degradation_cost_per_kwh": float(scenario.degradation_cost_per_kwh),
    }

    per_vehicle = []
    for vehicle_index, vehicle in enumerate(scenario.vehicles):
        per_vehicle.append(
            {
                "scenario_name": scenario.name,
                "family": scenario.family,
                "seed": scenario.seed,
                "method": plan.method_name,
                "vehicle_id": vehicle.vehicle_id,
                "priority_class": vehicle.priority_class,
                "arrival_slot": vehicle.arrival_slot,
                "departure_slot": vehicle.departure_slot,
                "required_energy_kwh": float(required_energy_kwh[vehicle_index]),
                "delivered_energy_kwh": float(delivered_energy_kwh[vehicle_index]),
                "unmet_charge_demand_kwh": float(unmet_energy_kwh[vehicle_index]),
                "served_fraction": float(served_fraction[vehicle_index]),
            }
        )

    site_profile = []
    for slot in range(scenario.horizon_slots):
        site_profile.append(
            {
                "scenario_name": scenario.name,
                "family": scenario.family,
                "seed": scenario.seed,
                "method": plan.method_name,
                "slot": slot,
                "hour": slot * scenario.dt_hours,
                "tariff_per_kwh": float(scenario.tariff_per_kwh[slot]),
                "site_capacity_kw": float(capacity[slot]),
                "requested_site_kw": float(requested_site_kw[slot]),
                "actual_site_kw": float(actual_site_kw[slot]),
                "is_onpeak_demand_window": bool(onpeak_mask[slot]),
                "marginal_degradation_cost_per_kwh": float(scenario.degradation_cost_per_kwh),
            }
        )

    return EvaluationResult(summary=summary, per_vehicle=per_vehicle, site_profile=site_profile)


def criticality_score(
    remaining_energy_kwh: float, remaining_slots: int, max_charge_kw: float, dt_hours: float, priority_weight: float
) -> float:
    remaining_deliverable = max(max_charge_kw * dt_hours * remaining_slots, SCHEDULE_TOLERANCE_KW)
    urgency = remaining_energy_kwh / remaining_deliverable
    return priority_weight * 10.0 + urgency + 1.0 / max(1, remaining_slots)
