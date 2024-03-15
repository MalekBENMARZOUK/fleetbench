from __future__ import annotations

import pytest

from ev_fleet_benchmark.exceptions import (
    FleetBenchError,
    MethodExecutionError,
    PlanValidationError,
    SolverError,
    SolverUnavailableError,
)
from ev_fleet_benchmark.validation import (
    validate_fraction,
    validate_non_negative,
    validate_positive,
    validate_positive_int,
    validate_time_step_minutes,
)


class TestExceptionHierarchy:
    def test_all_exceptions_are_fleetbench_error(self) -> None:
        exceptions: list[FleetBenchError] = [
            SolverError("solver error"),
            SolverUnavailableError("no solver"),
            MethodExecutionError("method", "scenario"),
            PlanValidationError("bad plan"),
        ]
        for exc in exceptions:
            assert isinstance(exc, FleetBenchError)

    def test_solver_hierarchy(self) -> None:
        assert issubclass(SolverUnavailableError, SolverError)

    def test_method_execution_error_attributes(self) -> None:
        exc = MethodExecutionError("naive_baseline", "urban_seed_1", reason="timeout")
        assert exc.method_name == "naive_baseline"
        assert exc.scenario_name == "urban_seed_1"
        assert exc.reason == "timeout"
        assert "naive_baseline" in str(exc)
        assert "urban_seed_1" in str(exc)
        assert "timeout" in str(exc)

    def test_method_execution_error_without_reason(self) -> None:
        exc = MethodExecutionError("greedy", "scenario_x")
        assert exc.reason is None
        assert "greedy" in str(exc)


class TestValidationImprovements:
    def test_validate_positive_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="finite positive"):
            validate_positive(float("nan"), "test_field")

    def test_validate_positive_rejects_inf(self) -> None:
        with pytest.raises(ValueError, match="finite positive"):
            validate_positive(float("inf"), "test_field")

    def test_validate_non_negative_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="finite non-negative"):
            validate_non_negative(float("nan"), "test_field")

    def test_validate_non_negative_rejects_inf(self) -> None:
        with pytest.raises(ValueError, match="finite non-negative"):
            validate_non_negative(float("inf"), "test_field")

    def test_validate_fraction_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_fraction(float("nan"), "test_field")

    def test_validate_positive_int_includes_value_in_message(self) -> None:
        with pytest.raises(ValueError, match="-5"):
            validate_positive_int(-5, "count")

    def test_validate_time_step_includes_value_in_message(self) -> None:
        with pytest.raises(ValueError, match="17"):
            validate_time_step_minutes(17)

    def test_validate_positive_accepts_valid(self) -> None:
        validate_positive(0.001, "x")
        validate_positive(1000.0, "x")

    def test_validate_non_negative_accepts_zero(self) -> None:
        validate_non_negative(0.0, "x")

    def test_validate_fraction_accepts_boundaries(self) -> None:
        validate_fraction(0.0, "x")
        validate_fraction(1.0, "x")
        validate_fraction(0.5, "x")


class TestVersion:
    def test_version_is_accessible(self) -> None:
        from ev_fleet_benchmark import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_version_looks_like_semver(self) -> None:
        from ev_fleet_benchmark import __version__

        parts = __version__.split(".")
        assert len(parts) >= 2


class TestORToolsAvailability:
    def test_ortools_available_flag_is_bool(self) -> None:
        from ev_fleet_benchmark.methods import ORTOOLS_AVAILABLE

        assert isinstance(ORTOOLS_AVAILABLE, bool)

    def test_method_registry_includes_heuristics(self) -> None:
        from ev_fleet_benchmark.methods import METHOD_REGISTRY

        assert "naive_baseline" in METHOD_REGISTRY
        assert "greedy_urgency" in METHOD_REGISTRY
        assert "stochastic_anticipatory" in METHOD_REGISTRY


class TestBenchmarkConfigWorkers:
    def test_benchmark_config_accepts_workers(self) -> None:
        from ev_fleet_benchmark.benchmark import BenchmarkConfig

        config = BenchmarkConfig(
            family_names=["urban_depot_small"],
            seeds=[1],
            max_workers=4,
        )
        assert config.max_workers == 4

    def test_benchmark_config_rejects_zero_workers(self) -> None:
        from ev_fleet_benchmark.benchmark import BenchmarkConfig

        with pytest.raises(ValueError, match="max_workers"):
            BenchmarkConfig(
                family_names=["urban_depot_small"],
                seeds=[1],
                max_workers=0,
            )
