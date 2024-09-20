# Output Schema

FleetBench writes tabular outputs to the selected `--output-dir`. Column sets may grow over time, but the core columns below are stable for the 0.1 line.

## `scenario_metrics.csv`

One row per scenario and method.

| Column | Meaning |
|---|---|
| `scenario_name` | Concrete generated scenario name. |
| `family` | Scenario family used to generate the case. |
| `seed` | Random seed for reproducibility. |
| `method` | Scheduling method name. |
| `status` | Method execution status. |
| `service_level` | Fraction of requested energy delivered, clipped to 1.0. |
| `energy_shortfall_kwh` | Total unmet requested energy. |
| `total_energy_kwh` | Energy delivered by the plan. |
| `total_charging_cost` | Tariff, demand charge, and degradation cost total. |
| `peak_site_load_kw` | Maximum aggregate site load. |
| `solve_time_s` | Method solve time in seconds. |

## `vehicle_metrics.csv`

One row per scenario, method, and vehicle.

| Column | Meaning |
|---|---|
| `vehicle_id` | Vehicle identifier in the generated scenario. |
| `required_energy_kwh` | Energy required to hit the target state of charge. |
| `delivered_energy_kwh` | Energy delivered before departure. |
| `shortfall_kwh` | Remaining unmet energy. |
| `service_level` | Vehicle-level service fraction. |
| `priority_class` | Operational priority label. |

## `site_load_profiles.csv`

One row per scenario, method, and time slot.

| Column | Meaning |
|---|---|
| `slot` | Integer time slot in the scenario horizon. |
| `site_load_kw` | Aggregate charging power. |
| `site_capacity_kw` | Available site capacity for the slot. |
| `tariff_per_kwh` | Energy tariff for the slot. |

## Aggregate Outputs

- `aggregate_by_family.csv`: method metrics grouped by scenario family.
- `aggregate_overall.csv`: method metrics across all evaluated families.
- `publication/publication_method_ranking.csv`: multi-metric method ranking.
- `publication/publication_method_confidence.csv`: bootstrap confidence intervals.
- `publication/publication_family_winners.csv`: best method by family.
- `publication/publication_pairwise_comparison.csv`: method-level pairwise deltas.

## Sensitivity Outputs

Sensitivity studies write analogous files prefixed with `sensitivity_` plus:

- `publication/sensitivity_profile_aggregate.csv`
- `publication/sensitivity_profile_ranking.csv`
- `publication/sensitivity_method_robustness.csv`
- `publication/sensitivity_profile_deltas.csv`
