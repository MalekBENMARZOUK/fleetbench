from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from ev_fleet_benchmark.benchmark import (
    BenchmarkConfig,
    SensitivityStudyConfig,
    run_benchmark,
    run_sensitivity_study,
)

try:
    __version__: str = _pkg_version("ev-fleet-benchmark")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["BenchmarkConfig", "SensitivityStudyConfig", "__version__", "run_benchmark", "run_sensitivity_study"]
