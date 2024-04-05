from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ev_fleet_benchmark.model import Scenario


@dataclass(frozen=True)
class ScenarioTreeNode:
    node_id: str
    probability: float
    arrival_slots: np.ndarray
    site_capacity_kw: np.ndarray


def build_uncertainty_tree(scenario: Scenario, current_slot: int, lookahead_slots: int) -> list[ScenarioTreeNode]:
    uncertainty_model = scenario.metadata.get("uncertainty_model", {})
    delay_low, delay_high = uncertainty_model.get("arrival_delay_slot_range", (0, 0))
    severity_low, severity_high = uncertainty_model.get("site_derate_severity_range", (0.0, 0.0))
    delay_probability = float(uncertainty_model.get("arrival_delay_probability", 0.0))
    derate_probability = float(uncertainty_model.get("site_derate_probability", 0.0))

    horizon_end = min(scenario.horizon_slots, current_slot + lookahead_slots)
    base_arrivals = np.array([vehicle.arrival_slot for vehicle in scenario.vehicles], dtype=int)
    planned_arrivals = np.array(
        [
            (vehicle.planned_arrival_slot if vehicle.planned_arrival_slot is not None else vehicle.arrival_slot)
            for vehicle in scenario.vehicles
        ],
        dtype=int,
    )
    base_capacity = scenario.site_capacity_kw[current_slot:horizon_end].copy()

    delay_mid = 0 if delay_high <= 0 else max(delay_low, round((delay_low + delay_high) / 2.0))
    derate_mid = max(0.0, (severity_low + severity_high) / 2.0)

    nodes = [
        ScenarioTreeNode(
            node_id="nominal",
            probability=max(0.0, (1.0 - delay_probability) * (1.0 - derate_probability)),
            arrival_slots=np.where(base_arrivals <= current_slot, base_arrivals, planned_arrivals),
            site_capacity_kw=base_capacity,
        ),
        ScenarioTreeNode(
            node_id="delay_only",
            probability=max(0.0, delay_probability * (1.0 - derate_probability)),
            arrival_slots=np.where(
                base_arrivals <= current_slot,
                base_arrivals,
                np.minimum(scenario.horizon_slots - 1, planned_arrivals + delay_mid),
            ),
            site_capacity_kw=base_capacity,
        ),
        ScenarioTreeNode(
            node_id="derate_only",
            probability=max(0.0, (1.0 - delay_probability) * derate_probability),
            arrival_slots=np.where(base_arrivals <= current_slot, base_arrivals, planned_arrivals),
            site_capacity_kw=base_capacity * (1.0 - derate_mid),
        ),
        ScenarioTreeNode(
            node_id="delay_and_derate",
            probability=max(0.0, delay_probability * derate_probability),
            arrival_slots=np.where(
                base_arrivals <= current_slot,
                base_arrivals,
                np.minimum(scenario.horizon_slots - 1, planned_arrivals + delay_mid),
            ),
            site_capacity_kw=base_capacity * (1.0 - max(derate_mid, severity_low)),
        ),
    ]

    total_probability = sum(node.probability for node in nodes)
    if total_probability <= 0.0:
        return [nodes[0]]

    return [
        ScenarioTreeNode(
            node_id=node.node_id,
            probability=node.probability / total_probability,
            arrival_slots=node.arrival_slots,
            site_capacity_kw=node.site_capacity_kw,
        )
        for node in nodes
    ]
