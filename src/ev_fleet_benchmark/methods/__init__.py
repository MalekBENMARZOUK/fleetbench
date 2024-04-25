import logging as _logging
from typing import TYPE_CHECKING

from ev_fleet_benchmark.methods.base import ScheduleMethod

if TYPE_CHECKING:
    from collections.abc import Callable
from ev_fleet_benchmark.methods.config import MethodFactoryConfigSet
from ev_fleet_benchmark.methods.greedy import GreedyUrgencyMethod
from ev_fleet_benchmark.methods.naive import NaiveBaselineMethod
from ev_fleet_benchmark.methods.stochastic import StochasticAnticipatoryMethod

_logger = _logging.getLogger(__name__)

try:
    from ev_fleet_benchmark.methods.optimization import OptimizationMethod
    from ev_fleet_benchmark.methods.rolling_horizon import RollingHorizonOptimizationMethod
    from ev_fleet_benchmark.methods.scenario_tree import ScenarioTreeOptimizationMethod

    _ORTOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ORTOOLS_AVAILABLE = False
    _logger.info("ortools not installed - optimization-based methods are unavailable")

__all__ = [
    "ORTOOLS_AVAILABLE",
    "GreedyUrgencyMethod",
    "MethodFactoryConfigSet",
    "NaiveBaselineMethod",
    "StochasticAnticipatoryMethod",
    "build_methods",
    "method_names",
]

if _ORTOOLS_AVAILABLE:
    __all__ += [
        "OptimizationMethod",
        "RollingHorizonOptimizationMethod",
        "ScenarioTreeOptimizationMethod",
    ]

ORTOOLS_AVAILABLE: bool = _ORTOOLS_AVAILABLE

_HEURISTIC_TYPES: tuple[type[ScheduleMethod], ...] = (
    NaiveBaselineMethod,
    GreedyUrgencyMethod,
    StochasticAnticipatoryMethod,
)

_ORTOOLS_TYPES: tuple[type[ScheduleMethod], ...] = ()
if _ORTOOLS_AVAILABLE:
    _ORTOOLS_TYPES = (
        OptimizationMethod,
        RollingHorizonOptimizationMethod,
        ScenarioTreeOptimizationMethod,
    )

METHOD_TYPES: tuple[type[ScheduleMethod], ...] = _HEURISTIC_TYPES + _ORTOOLS_TYPES

METHOD_REGISTRY: dict[str, type[ScheduleMethod]] = {method_type.name: method_type for method_type in METHOD_TYPES}
if len(METHOD_REGISTRY) != len(METHOD_TYPES):
    raise ValueError("Duplicate schedule method names detected in method registry")

_KNOWN_ORTOOLS_METHODS = {"optimization_ortools", "rolling_horizon_ortools", "scenario_tree_ortools"}


def build_methods(
    selected_names: list[str] | None = None, config_set: MethodFactoryConfigSet | None = None
) -> list[ScheduleMethod]:
    resolved_config = config_set or MethodFactoryConfigSet()
    if selected_names is None:
        names = list(METHOD_REGISTRY)
    else:
        names = list(dict.fromkeys(selected_names))
        if not names:
            raise ValueError("At least one method must be selected")
    unknown_names = [name for name in names if name not in METHOD_REGISTRY]
    if unknown_names:
        missing_ortools = [n for n in unknown_names if n in _KNOWN_ORTOOLS_METHODS]
        if missing_ortools and not _ORTOOLS_AVAILABLE:
            raise ValueError(
                f"Methods {missing_ortools} require the 'ortools' package which is not installed. "
                f"Install it with: pip install ortools"
            )
        raise ValueError(
            f"Unknown method names requested: {unknown_names}. Available methods: {', '.join(METHOD_REGISTRY)}"
        )

    factories: dict[str, Callable[[], ScheduleMethod]] = {
        "naive_baseline": lambda: NaiveBaselineMethod(config=resolved_config.naive),
        "greedy_urgency": lambda: GreedyUrgencyMethod(config=resolved_config.greedy),
        "stochastic_anticipatory": lambda: StochasticAnticipatoryMethod(config=resolved_config.stochastic),
    }
    if _ORTOOLS_AVAILABLE:
        factories.update(
            {
                "optimization_ortools": lambda: OptimizationMethod(config=resolved_config.optimization),
                "rolling_horizon_ortools": lambda: RollingHorizonOptimizationMethod(
                    config=resolved_config.rolling_horizon
                ),
                "scenario_tree_ortools": lambda: ScenarioTreeOptimizationMethod(config=resolved_config.scenario_tree),
            }
        )
    return [factories[name]() for name in names]


def method_names() -> list[str]:
    return list(METHOD_REGISTRY)
