from ev_fleet_benchmark.scenarios import apply_sensitivity_profile, generate_scenario


def test_scenario_generation_is_reproducible() -> None:
    scenario_a = generate_scenario("urban_depot_small", seed=7)
    scenario_b = generate_scenario("urban_depot_small", seed=7)

    assert scenario_a.to_dict() == scenario_b.to_dict()
    assert scenario_a.horizon_slots == 48
    assert len(scenario_a.vehicles) >= 8


def test_scenario_has_consistent_vectors() -> None:
    scenario = generate_scenario("capacity_stressed_peak", seed=9)

    assert len(scenario.site_capacity_kw) == scenario.horizon_slots
    assert len(scenario.tariff_per_kwh) == scenario.horizon_slots
    assert all(vehicle.departure_slot > vehicle.arrival_slot for vehicle in scenario.vehicles)
    assert "uncertainty_model" in scenario.metadata
    assert "site_derate_events" in scenario.metadata
    assert scenario.demand_charge_per_kw > 0.0
    assert scenario.degradation_cost_per_kwh > 0.0


def test_sensitivity_profile_modifies_economic_parameters() -> None:
    scenario = generate_scenario("urban_depot_small", seed=13)
    stressed = apply_sensitivity_profile(scenario, "tariff_and_wear_stress")

    assert stressed.name.endswith("__tariff_and_wear_stress")
    assert stressed.demand_charge_per_kw > scenario.demand_charge_per_kw
    assert stressed.onpeak_demand_charge_per_kw > scenario.onpeak_demand_charge_per_kw
    assert stressed.degradation_cost_per_kwh > scenario.degradation_cost_per_kwh
    assert stressed.metadata["sensitivity_profile"] == "tariff_and_wear_stress"
