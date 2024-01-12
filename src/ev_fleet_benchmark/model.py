from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

DEFAULT_CHARGE_EFFICIENCY = 0.95
SCHEDULE_TOLERANCE_KW = 1e-9
ENERGY_TOLERANCE_KWH = 1e-6


PRIORITY_WEIGHTS = {
    "low": 1.0,
    "medium": 1.3,
    "high": 1.8,
    "critical": 2.4,
}


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@dataclass(frozen=True)
class VehicleRequest:
    vehicle_id: str
    battery_capacity_kwh: float
    initial_soc: float
    target_soc: float
    max_charge_kw: float
    arrival_slot: int
    departure_slot: int
    priority_class: str
    charge_efficiency: float = DEFAULT_CHARGE_EFFICIENCY
    planned_arrival_slot: int | None = None

    def __post_init__(self) -> None:
        if not self.vehicle_id.strip():
            raise ValueError("vehicle_id must be a non-empty string")
        if self.battery_capacity_kwh <= 0.0:
            raise ValueError(f"Vehicle {self.vehicle_id} battery_capacity_kwh must be positive")
        if self.max_charge_kw <= 0.0:
            raise ValueError(f"Vehicle {self.vehicle_id} max_charge_kw must be positive")
        if not 0.0 <= self.initial_soc <= 1.0:
            raise ValueError(f"Vehicle {self.vehicle_id} initial_soc must be in [0, 1]")
        if not 0.0 <= self.target_soc <= 1.0:
            raise ValueError(f"Vehicle {self.vehicle_id} target_soc must be in [0, 1]")
        if self.target_soc < self.initial_soc:
            raise ValueError(f"Vehicle {self.vehicle_id} target_soc must be greater than or equal to initial_soc")
        if self.arrival_slot < 0:
            raise ValueError(f"Vehicle {self.vehicle_id} arrival_slot must be non-negative")
        if self.departure_slot <= self.arrival_slot:
            raise ValueError(f"Vehicle {self.vehicle_id} departure_slot must be greater than arrival_slot")
        if self.priority_class not in PRIORITY_WEIGHTS:
            raise ValueError(f"Vehicle {self.vehicle_id} priority_class must be one of {sorted(PRIORITY_WEIGHTS)}")
        if not 0.0 < self.charge_efficiency <= 1.0:
            raise ValueError(f"Vehicle {self.vehicle_id} charge_efficiency must be in (0, 1]")
        if self.planned_arrival_slot is not None and self.planned_arrival_slot < 0:
            raise ValueError(f"Vehicle {self.vehicle_id} planned_arrival_slot must be non-negative when provided")

    @property
    def required_energy_kwh(self) -> float:
        return max(0.0, (self.target_soc - self.initial_soc) * self.battery_capacity_kwh)

    @property
    def available_slots(self) -> int:
        return max(0, self.departure_slot - self.arrival_slot)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VehicleRequest:
        return cls(**data)


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    seed: int
    time_step_minutes: int
    horizon_slots: int
    vehicles: list[VehicleRequest]
    site_capacity_kw: np.ndarray
    tariff_per_kwh: np.ndarray
    demand_charge_per_kw: float = 0.0
    onpeak_demand_charge_per_kw: float = 0.0
    degradation_cost_per_kwh: float = 0.0
    degradation_high_soc_threshold: float = 0.8
    degradation_high_soc_multiplier: float = 0.0
    degradation_c_rate_coefficient: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Scenario name must be a non-empty string")
        if not self.family.strip():
            raise ValueError("Scenario family must be a non-empty string")
        if self.time_step_minutes <= 0:
            raise ValueError("Scenario time_step_minutes must be positive")
        if 1440 % self.time_step_minutes != 0:
            raise ValueError("Scenario time_step_minutes must divide evenly into 24 hours")
        if self.horizon_slots <= 0:
            raise ValueError("Scenario horizon_slots must be positive")
        if not self.vehicles:
            raise ValueError("Scenario must contain at least one vehicle")

        vehicle_ids = [v.vehicle_id for v in self.vehicles]
        if len(vehicle_ids) != len(set(vehicle_ids)):
            duplicates = [vid for vid in vehicle_ids if vehicle_ids.count(vid) > 1]
            raise ValueError(f"Scenario contains duplicate vehicle_ids: {sorted(set(duplicates))}")

        site_capacity_kw = np.asarray(self.site_capacity_kw, dtype=float)
        tariff_per_kwh = np.asarray(self.tariff_per_kwh, dtype=float)
        if site_capacity_kw.ndim != 1:
            raise ValueError("Scenario site_capacity_kw must be a one-dimensional array")
        if tariff_per_kwh.ndim != 1:
            raise ValueError("Scenario tariff_per_kwh must be a one-dimensional array")
        if len(site_capacity_kw) != self.horizon_slots:
            raise ValueError("Scenario site_capacity_kw length must match horizon_slots")
        if len(tariff_per_kwh) != self.horizon_slots:
            raise ValueError("Scenario tariff_per_kwh length must match horizon_slots")
        if not np.all(np.isfinite(site_capacity_kw)):
            raise ValueError("Scenario site_capacity_kw must contain only finite values")
        if not np.all(np.isfinite(tariff_per_kwh)):
            raise ValueError("Scenario tariff_per_kwh must contain only finite values")
        if np.any(site_capacity_kw < 0.0):
            raise ValueError("Scenario site_capacity_kw must be non-negative")
        if np.any(tariff_per_kwh < 0.0):
            raise ValueError("Scenario tariff_per_kwh must be non-negative")

        for vehicle in self.vehicles:
            if vehicle.departure_slot > self.horizon_slots:
                raise ValueError(f"Vehicle {vehicle.vehicle_id} departure_slot exceeds scenario horizon_slots")

        for field_name, value in {
            "demand_charge_per_kw": self.demand_charge_per_kw,
            "onpeak_demand_charge_per_kw": self.onpeak_demand_charge_per_kw,
            "degradation_cost_per_kwh": self.degradation_cost_per_kwh,
            "degradation_high_soc_multiplier": self.degradation_high_soc_multiplier,
            "degradation_c_rate_coefficient": self.degradation_c_rate_coefficient,
        }.items():
            if value < 0.0:
                raise ValueError(f"Scenario {field_name} must be non-negative")

        if not 0.0 <= self.degradation_high_soc_threshold <= 1.0:
            raise ValueError("Scenario degradation_high_soc_threshold must be in [0, 1]")

        object.__setattr__(self, "site_capacity_kw", site_capacity_kw)
        object.__setattr__(self, "tariff_per_kwh", tariff_per_kwh)

    @property
    def dt_hours(self) -> float:
        return self.time_step_minutes / 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "seed": self.seed,
            "time_step_minutes": self.time_step_minutes,
            "horizon_slots": self.horizon_slots,
            "vehicles": [vehicle.to_dict() for vehicle in self.vehicles],
            "site_capacity_kw": self.site_capacity_kw.tolist(),
            "tariff_per_kwh": self.tariff_per_kwh.tolist(),
            "demand_charge_per_kw": self.demand_charge_per_kw,
            "onpeak_demand_charge_per_kw": self.onpeak_demand_charge_per_kw,
            "degradation_cost_per_kwh": self.degradation_cost_per_kwh,
            "degradation_high_soc_threshold": self.degradation_high_soc_threshold,
            "degradation_high_soc_multiplier": self.degradation_high_soc_multiplier,
            "degradation_c_rate_coefficient": self.degradation_c_rate_coefficient,
            "metadata": _sanitize_for_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        vehicles = [VehicleRequest.from_dict(v) for v in data["vehicles"]]
        return cls(
            name=data["name"],
            family=data["family"],
            seed=data["seed"],
            time_step_minutes=data["time_step_minutes"],
            horizon_slots=data["horizon_slots"],
            vehicles=vehicles,
            site_capacity_kw=np.array(data["site_capacity_kw"], dtype=float),
            tariff_per_kwh=np.array(data["tariff_per_kwh"], dtype=float),
            demand_charge_per_kw=data.get("demand_charge_per_kw", 0.0),
            onpeak_demand_charge_per_kw=data.get("onpeak_demand_charge_per_kw", 0.0),
            degradation_cost_per_kwh=data.get("degradation_cost_per_kwh", 0.0),
            degradation_high_soc_threshold=data.get("degradation_high_soc_threshold", 0.8),
            degradation_high_soc_multiplier=data.get("degradation_high_soc_multiplier", 0.0),
            degradation_c_rate_coefficient=data.get("degradation_c_rate_coefficient", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SchedulePlan:
    method_name: str
    power_kw: np.ndarray
    solve_time_s: float
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.method_name.strip():
            raise ValueError("SchedulePlan method_name must be a non-empty string")
        power_kw = np.asarray(self.power_kw, dtype=float)
        if power_kw.ndim != 2:
            raise ValueError("SchedulePlan power_kw must be a two-dimensional array")
        if not np.all(np.isfinite(power_kw)):
            raise ValueError("SchedulePlan power_kw must contain only finite values")
        if np.any(power_kw < -SCHEDULE_TOLERANCE_KW):
            raise ValueError("SchedulePlan power_kw must not contain negative dispatch values")
        if not np.isfinite(self.solve_time_s) or self.solve_time_s < 0.0:
            raise ValueError("SchedulePlan solve_time_s must be finite and non-negative")
        if not self.status.strip():
            raise ValueError("SchedulePlan status must be a non-empty string")
        self.power_kw = power_kw


@dataclass
class EvaluationResult:
    summary: dict[str, Any]
    per_vehicle: list[dict[str, Any]]
    site_profile: list[dict[str, Any]]
