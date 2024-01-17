from __future__ import annotations


class FleetBenchError(Exception):
    """Base exception for all FleetBench errors."""


class SolverError(FleetBenchError):
    """An OR-Tools solver invocation failed or was unavailable."""


class SolverUnavailableError(SolverError):
    """The requested solver backend could not be created."""


class MethodExecutionError(FleetBenchError):
    def __init__(self, method_name: str, scenario_name: str, reason: str | None = None) -> None:
        self.method_name = method_name
        self.scenario_name = scenario_name
        self.reason = reason
        detail = f" — {reason}" if reason else ""
        super().__init__(f"Method '{method_name}' failed on scenario '{scenario_name}'{detail}")


class PlanValidationError(FleetBenchError, ValueError):
    """A schedule plan failed post-hoc validation."""
