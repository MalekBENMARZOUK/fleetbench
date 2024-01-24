from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

from ev_fleet_benchmark.model import PRIORITY_WEIGHTS, Scenario, VehicleRequest


@dataclass(frozen=True)
class ScenarioFamily:
    name: str
    description: str
    fleet_size_range: tuple[int, int]
    capacity_tightness: tuple[float, float]
    delay_probability: float
    delay_slot_range: tuple[int, int]
    site_derate_probability: float
    site_derate_duration_range: tuple[int, int]
    site_derate_severity_range: tuple[float, float]
    target_soc_range: tuple[float, float]
    dwell_slots_range: tuple[int, int]
    tariff_profile: str
    priority_mix: dict[str, float]


@dataclass(frozen=True)
class SensitivityProfile:
    name: str
    description: str
    energy_tariff_multiplier: float = 1.0
    onpeak_tariff_multiplier: float = 1.0
    all_day_demand_multiplier: float = 1.0
    onpeak_demand_multiplier: float = 1.0
    degradation_base_multiplier: float = 1.0
    degradation_high_soc_multiplier: float = 1.0
    degradation_c_rate_multiplier: float = 1.0


class ScenarioFamilyDescription(TypedDict):
    name: str
    description: str
    fleet_size_range: tuple[int, int]
    capacity_tightness: tuple[float, float]
    delay_probability: float
    delay_slot_range: tuple[int, int]
    site_derate_probability: float
    site_derate_duration_range: tuple[int, int]
    site_derate_severity_range: tuple[float, float]
    target_soc_range: tuple[float, float]
    dwell_slots_range: tuple[int, int]
    tariff_profile: str
    priority_mix: dict[str, float]


class SensitivityProfileDescription(TypedDict):
    name: str
    description: str
    energy_tariff_multiplier: float
    onpeak_tariff_multiplier: float
    all_day_demand_multiplier: float
    onpeak_demand_multiplier: float
    degradation_base_multiplier: float
    degradation_high_soc_multiplier: float
    degradation_c_rate_multiplier: float


def default_scenario_families() -> dict[str, ScenarioFamily]:
    return {
        "urban_depot_small": ScenarioFamily(
            name="urban_depot_small",
            description="Small depot with moderate capacity headroom and commuter-like departures.",
            fleet_size_range=(8, 15),
            capacity_tightness=(0.42, 0.52),
            delay_probability=0.12,
            delay_slot_range=(1, 2),
            site_derate_probability=0.08,
            site_derate_duration_range=(1, 2),
            site_derate_severity_range=(0.08, 0.14),
            target_soc_range=(0.75, 0.95),
            dwell_slots_range=(8, 18),
            tariff_profile="time_of_use",
            priority_mix={"low": 0.2, "medium": 0.45, "high": 0.25, "critical": 0.1},
        ),
        "regional_mixed_medium": ScenarioFamily(
            name="regional_mixed_medium",
            description="Medium heterogeneous fleet with tighter windows and mixed charging rates.",
            fleet_size_range=(20, 35),
            capacity_tightness=(0.32, 0.44),
            delay_probability=0.18,
            delay_slot_range=(1, 3),
            site_derate_probability=0.12,
            site_derate_duration_range=(1, 3),
            site_derate_severity_range=(0.1, 0.18),
            target_soc_range=(0.8, 0.98),
            dwell_slots_range=(5, 16),
            tariff_profile="peaky_evening",
            priority_mix={"low": 0.15, "medium": 0.4, "high": 0.3, "critical": 0.15},
        ),
        "capacity_stressed_peak": ScenarioFamily(
            name="capacity_stressed_peak",
            description="Capacity-constrained depot under strong tariff peaks and compressed dwell times.",
            fleet_size_range=(18, 28),
            capacity_tightness=(0.24, 0.34),
            delay_probability=0.22,
            delay_slot_range=(1, 4),
            site_derate_probability=0.18,
            site_derate_duration_range=(2, 4),
            site_derate_severity_range=(0.12, 0.24),
            target_soc_range=(0.82, 1.0),
            dwell_slots_range=(4, 12),
            tariff_profile="sharp_peak",
            priority_mix={"low": 0.1, "medium": 0.35, "high": 0.35, "critical": 0.2},
        ),
        "uncertain_operations_large": ScenarioFamily(
            name="uncertain_operations_large",
            description="Large fleet with higher arrival uncertainty and mixed operational urgency.",
            fleet_size_range=(35, 55),
            capacity_tightness=(0.26, 0.38),
            delay_probability=0.3,
            delay_slot_range=(1, 5),
            site_derate_probability=0.22,
            site_derate_duration_range=(2, 5),
            site_derate_severity_range=(0.1, 0.22),
            target_soc_range=(0.75, 0.98),
            dwell_slots_range=(4, 14),
            tariff_profile="two_peak",
            priority_mix={"low": 0.18, "medium": 0.37, "high": 0.28, "critical": 0.17},
        ),
    }


def default_sensitivity_profiles() -> dict[str, SensitivityProfile]:
    return {
        "baseline": SensitivityProfile(
            name="baseline",
            description="Reference economics without additional sensitivity perturbation.",
        ),
        "tariff_relief": SensitivityProfile(
            name="tariff_relief",
            description="Lower volumetric and demand-charge pressure.",
            energy_tariff_multiplier=0.9,
            onpeak_tariff_multiplier=0.9,
            all_day_demand_multiplier=0.85,
            onpeak_demand_multiplier=0.85,
        ),
        "tariff_stress": SensitivityProfile(
            name="tariff_stress",
            description="Higher on-peak energy and demand charges.",
            energy_tariff_multiplier=1.1,
            onpeak_tariff_multiplier=1.25,
            all_day_demand_multiplier=1.15,
            onpeak_demand_multiplier=1.3,
        ),
        "wear_relief": SensitivityProfile(
            name="wear_relief",
            description="Lower battery wear penalties.",
            degradation_base_multiplier=0.8,
            degradation_high_soc_multiplier=0.75,
            degradation_c_rate_multiplier=0.8,
        ),
        "wear_stress": SensitivityProfile(
            name="wear_stress",
            description="Higher battery wear penalties for aggressive charging.",
            degradation_base_multiplier=1.35,
            degradation_high_soc_multiplier=1.4,
            degradation_c_rate_multiplier=1.35,
        ),
        "tariff_and_wear_stress": SensitivityProfile(
            name="tariff_and_wear_stress",
            description="Combined tariff and battery wear stress test.",
            energy_tariff_multiplier=1.12,
            onpeak_tariff_multiplier=1.3,
            all_day_demand_multiplier=1.15,
            onpeak_demand_multiplier=1.35,
            degradation_base_multiplier=1.3,
            degradation_high_soc_multiplier=1.4,
            degradation_c_rate_multiplier=1.35,
        ),
    }


def generate_scenario(family_name: str, seed: int, time_step_minutes: int = 30) -> Scenario:
    families = default_scenario_families()
    if family_name not in families:
        raise ValueError(f"Unknown scenario family: {family_name}. Available families: {', '.join(sorted(families))}")

    family = families[family_name]
    rng = np.random.default_rng(seed)
    horizon_slots = int(24 * 60 / time_step_minutes)
    fleet_size = int(rng.integers(family.fleet_size_range[0], family.fleet_size_range[1] + 1))

    battery_choices = np.array([50.0, 60.0, 75.0, 90.0])
    max_power_choices = np.array([7.2, 11.0, 22.0])
    arrival_centers = np.array([14, 18, 22]) * 60 / time_step_minutes
    arrival_center = int(rng.choice(arrival_centers, p=np.array([0.25, 0.4, 0.35])))

    vehicles: list[VehicleRequest] = []
    total_max_power = 0.0
    priority_labels = list(family.priority_mix.keys())
    priority_probabilities = np.array(list(family.priority_mix.values()))

    for index in range(fleet_size):
        battery_capacity = float(rng.choice(battery_choices))
        max_charge_kw = float(rng.choice(max_power_choices, p=np.array([0.35, 0.45, 0.2])))
        initial_soc = float(rng.uniform(0.12, 0.62))
        target_soc = float(rng.uniform(*family.target_soc_range))
        target_soc = max(target_soc, initial_soc + 0.1)
        target_soc = min(target_soc, 1.0)
        dwell_slots = int(rng.integers(family.dwell_slots_range[0], family.dwell_slots_range[1] + 1))
        planned_arrival = int(np.clip(rng.normal(arrival_center, 5), 0, horizon_slots - 2))

        delay = 0
        if rng.random() < family.delay_probability:
            delay = int(rng.integers(family.delay_slot_range[0], family.delay_slot_range[1] + 1))
        arrival_slot = min(horizon_slots - 2, planned_arrival + delay)
        departure_slot = min(horizon_slots, arrival_slot + dwell_slots)
        if departure_slot <= arrival_slot:
            departure_slot = min(horizon_slots, arrival_slot + 1)

        priority_class = str(rng.choice(priority_labels, p=priority_probabilities))

        vehicles.append(
            VehicleRequest(
                vehicle_id=f"EV-{index + 1:03d}",
                battery_capacity_kwh=battery_capacity,
                initial_soc=initial_soc,
                target_soc=target_soc,
                max_charge_kw=max_charge_kw,
                arrival_slot=arrival_slot,
                departure_slot=departure_slot,
                priority_class=priority_class,
                planned_arrival_slot=planned_arrival,
            )
        )
        total_max_power += max_charge_kw

    capacity_ratio = float(rng.uniform(*family.capacity_tightness))
    base_capacity = max(20.0, capacity_ratio * total_max_power)
    site_capacity = _site_capacity_profile(family.tariff_profile, horizon_slots, base_capacity)
    site_capacity, derate_events = _apply_site_derates(site_capacity, family, rng)
    tariff = _tariff_profile(family.tariff_profile, horizon_slots)
    onpeak_mask = _onpeak_mask(tariff)
    demand_charge_per_kw = float(rng.uniform(5.0, 10.0))
    onpeak_demand_charge_per_kw = float(rng.uniform(7.0, 16.0))
    degradation_cost_per_kwh = float(rng.uniform(0.008, 0.018))
    degradation_high_soc_threshold = float(rng.uniform(0.78, 0.86))
    degradation_high_soc_multiplier = float(rng.uniform(0.3, 0.9))
    degradation_c_rate_coefficient = float(rng.uniform(0.15, 0.45))

    name = f"{family_name}_seed_{seed}"
    metadata = {
        "family_description": family.description,
        "capacity_ratio": capacity_ratio,
        "fleet_size": fleet_size,
        "mean_priority_weight": float(np.mean([PRIORITY_WEIGHTS[v.priority_class] for v in vehicles])),
        "time_step_minutes": time_step_minutes,
        "uncertainty_model": {
            "arrival_delay_probability": family.delay_probability,
            "arrival_delay_slot_range": family.delay_slot_range,
            "site_derate_probability": family.site_derate_probability,
            "site_derate_duration_range": family.site_derate_duration_range,
            "site_derate_severity_range": family.site_derate_severity_range,
        },
        "site_derate_events": derate_events,
        "economic_model": {
            "demand_charge_per_kw": demand_charge_per_kw,
            "onpeak_demand_charge_per_kw": onpeak_demand_charge_per_kw,
            "degradation_cost_per_kwh": degradation_cost_per_kwh,
            "degradation_high_soc_threshold": degradation_high_soc_threshold,
            "degradation_high_soc_multiplier": degradation_high_soc_multiplier,
            "degradation_c_rate_coefficient": degradation_c_rate_coefficient,
            "onpeak_slots": np.flatnonzero(onpeak_mask).tolist(),
        },
    }
    return Scenario(
        name=name,
        family=family_name,
        seed=seed,
        time_step_minutes=time_step_minutes,
        horizon_slots=horizon_slots,
        vehicles=vehicles,
        site_capacity_kw=site_capacity,
        tariff_per_kwh=tariff,
        demand_charge_per_kw=demand_charge_per_kw,
        onpeak_demand_charge_per_kw=onpeak_demand_charge_per_kw,
        degradation_cost_per_kwh=degradation_cost_per_kwh,
        degradation_high_soc_threshold=degradation_high_soc_threshold,
        degradation_high_soc_multiplier=degradation_high_soc_multiplier,
        degradation_c_rate_coefficient=degradation_c_rate_coefficient,
        metadata=metadata,
    )


def save_scenario_json(scenario: Scenario, path: Path) -> None:
    path.write_text(json.dumps(scenario.to_dict(), indent=2), encoding="utf-8")


def describe_families() -> list[ScenarioFamilyDescription]:
    return [cast("ScenarioFamilyDescription", asdict(family)) for family in default_scenario_families().values()]


def describe_sensitivity_profiles() -> list[SensitivityProfileDescription]:
    return [
        cast("SensitivityProfileDescription", asdict(profile)) for profile in default_sensitivity_profiles().values()
    ]


def apply_sensitivity_profile(scenario: Scenario, profile_name: str) -> Scenario:
    profiles = default_sensitivity_profiles()
    if profile_name not in profiles:
        raise ValueError(
            f"Unknown sensitivity profile: {profile_name}. Available profiles: {', '.join(sorted(profiles))}"
        )

    profile = profiles[profile_name]
    onpeak_slots = scenario.metadata.get("economic_model", {}).get("onpeak_slots", [])
    onpeak_mask = np.zeros(scenario.horizon_slots, dtype=bool)
    for slot in onpeak_slots:
        slot_index = int(slot)
        if 0 <= slot_index < scenario.horizon_slots:
            onpeak_mask[slot_index] = True

    adjusted_tariff = scenario.tariff_per_kwh.astype(float).copy() * profile.energy_tariff_multiplier
    adjusted_tariff[onpeak_mask] *= profile.onpeak_tariff_multiplier

    metadata = dict(scenario.metadata)
    economic_model = dict(metadata.get("economic_model", {}))
    economic_model["base_demand_charge_per_kw"] = float(scenario.demand_charge_per_kw)
    economic_model["base_onpeak_demand_charge_per_kw"] = float(scenario.onpeak_demand_charge_per_kw)
    economic_model["base_degradation_cost_per_kwh"] = float(scenario.degradation_cost_per_kwh)
    economic_model["sensitivity_profile"] = profile.name
    economic_model["sensitivity_description"] = profile.description
    economic_model["sensitivity_multipliers"] = {
        "energy_tariff_multiplier": profile.energy_tariff_multiplier,
        "onpeak_tariff_multiplier": profile.onpeak_tariff_multiplier,
        "all_day_demand_multiplier": profile.all_day_demand_multiplier,
        "onpeak_demand_multiplier": profile.onpeak_demand_multiplier,
        "degradation_base_multiplier": profile.degradation_base_multiplier,
        "degradation_high_soc_multiplier": profile.degradation_high_soc_multiplier,
        "degradation_c_rate_multiplier": profile.degradation_c_rate_multiplier,
    }
    metadata["economic_model"] = economic_model
    metadata["sensitivity_profile"] = profile.name
    metadata["sensitivity_description"] = profile.description

    return replace(
        scenario,
        name=f"{scenario.name}__{profile.name}",
        tariff_per_kwh=adjusted_tariff,
        demand_charge_per_kw=scenario.demand_charge_per_kw * profile.all_day_demand_multiplier,
        onpeak_demand_charge_per_kw=scenario.onpeak_demand_charge_per_kw * profile.onpeak_demand_multiplier,
        degradation_cost_per_kwh=scenario.degradation_cost_per_kwh * profile.degradation_base_multiplier,
        degradation_high_soc_multiplier=scenario.degradation_high_soc_multiplier
        * profile.degradation_high_soc_multiplier,
        degradation_c_rate_coefficient=scenario.degradation_c_rate_coefficient * profile.degradation_c_rate_multiplier,
        metadata=metadata,
    )


def _tariff_profile(profile_name: str, horizon_slots: int) -> np.ndarray:
    hours = np.arange(horizon_slots) * 24.0 / horizon_slots
    base = np.full(horizon_slots, 0.16)

    if profile_name == "time_of_use":
        base[(hours >= 7) & (hours < 10)] = 0.23
        base[(hours >= 17) & (hours < 21)] = 0.31
    elif profile_name == "peaky_evening":
        base[(hours >= 6) & (hours < 9)] = 0.22
        base[(hours >= 16) & (hours < 22)] = 0.36
    elif profile_name == "sharp_peak":
        base[(hours >= 8) & (hours < 12)] = 0.24
        base[(hours >= 17) & (hours < 21)] = 0.42
    elif profile_name == "two_peak":
        base[(hours >= 7) & (hours < 10)] = 0.26
        base[(hours >= 17) & (hours < 20)] = 0.37
        base[(hours >= 20) & (hours < 23)] = 0.3
    else:
        raise KeyError(f"Unknown tariff profile: {profile_name}")

    base[(hours >= 0) & (hours < 5)] = 0.11
    return base


def _site_capacity_profile(profile_name: str, horizon_slots: int, base_capacity: float) -> np.ndarray:
    hours = np.arange(horizon_slots) * 24.0 / horizon_slots
    capacity = np.full(horizon_slots, base_capacity)
    if profile_name in {"peaky_evening", "sharp_peak", "two_peak"}:
        capacity[(hours >= 17) & (hours < 20)] *= 0.8
    if profile_name in {"sharp_peak", "two_peak"}:
        capacity[(hours >= 12) & (hours < 14)] *= 0.88
    capacity[(hours >= 0) & (hours < 5)] *= 1.08
    return capacity


def _onpeak_mask(tariff_per_kwh: np.ndarray) -> np.ndarray:
    threshold = float(np.quantile(tariff_per_kwh, 0.75))
    return tariff_per_kwh >= threshold - 1e-12


def _apply_site_derates(
    capacity: np.ndarray,
    family: ScenarioFamily,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    updated_capacity = capacity.copy()
    events: list[dict[str, float | int]] = []
    event_count = 0
    if rng.random() < family.site_derate_probability:
        event_count += 1
    if rng.random() < family.site_derate_probability / 2.0:
        event_count += 1

    for _ in range(event_count):
        duration = int(rng.integers(family.site_derate_duration_range[0], family.site_derate_duration_range[1] + 1))
        start_slot = int(rng.integers(0, max(1, len(updated_capacity) - duration)))
        severity = float(rng.uniform(*family.site_derate_severity_range))
        end_slot = min(len(updated_capacity), start_slot + duration)
        updated_capacity[start_slot:end_slot] *= 1.0 - severity
        events.append(
            {
                "start_slot": start_slot,
                "end_slot": end_slot,
                "duration_slots": end_slot - start_slot,
                "severity_fraction": severity,
            }
        )

    return updated_capacity, events
