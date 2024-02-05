from __future__ import annotations

from dataclasses import dataclass, field, replace

from ev_fleet_benchmark.validation import (
    validate_fraction,
    validate_non_negative,
    validate_positive,
    validate_positive_int,
)


@dataclass(frozen=True)
class OptimizationPenaltyConfig:
    peak_penalty: float = 0.015
    unmet_penalty: float = 250.0

    def __post_init__(self) -> None:
        validate_non_negative(self.peak_penalty, "peak_penalty")
        validate_positive(self.unmet_penalty, "unmet_penalty")


@dataclass(frozen=True)
class NaiveMethodConfig:
    onpeak_demand_bias_divisor: float = 100.0

    def __post_init__(self) -> None:
        validate_positive(self.onpeak_demand_bias_divisor, "onpeak_demand_bias_divisor")


@dataclass(frozen=True)
class GreedyMethodConfig:
    capacity_reserve_fraction: float = 0.18
    all_day_demand_guard_divisor: float = 12.0
    onpeak_demand_guard_divisor: float = 10.0
    onpeak_score_discount: float = 0.94

    def __post_init__(self) -> None:
        validate_fraction(self.capacity_reserve_fraction, "capacity_reserve_fraction")
        validate_positive(self.all_day_demand_guard_divisor, "all_day_demand_guard_divisor")
        validate_positive(self.onpeak_demand_guard_divisor, "onpeak_demand_guard_divisor")
        validate_fraction(self.onpeak_score_discount, "onpeak_score_discount")


@dataclass(frozen=True)
class RollingHorizonMethodConfig:
    lookahead_slots: int = 8
    penalties: OptimizationPenaltyConfig = field(default_factory=OptimizationPenaltyConfig)
    peak_guard_fraction: float = 0.12
    demand_guard_divisor: float = 8.0

    def __post_init__(self) -> None:
        validate_positive_int(self.lookahead_slots, "lookahead_slots")
        validate_fraction(self.peak_guard_fraction, "peak_guard_fraction")
        validate_positive(self.demand_guard_divisor, "demand_guard_divisor")


@dataclass(frozen=True)
class ScenarioTreeMethodConfig:
    lookahead_slots: int = 6
    penalties: OptimizationPenaltyConfig = field(default_factory=OptimizationPenaltyConfig)

    def __post_init__(self) -> None:
        validate_positive_int(self.lookahead_slots, "lookahead_slots")


@dataclass(frozen=True)
class StochasticMethodConfig:
    lookahead_slots: int = 8
    sample_count: int = 32
    reserve_quantile: float = 0.75
    reserve_capacity_limit_fraction: float = 0.75
    demand_guard_fraction: float = 0.15
    demand_guard_divisor: float = 8.0
    tariff_headroom_weight: float = 0.2
    tiebreak_noise_scale: float = 1e-5

    def __post_init__(self) -> None:
        validate_positive_int(self.lookahead_slots, "lookahead_slots")
        validate_positive_int(self.sample_count, "sample_count")
        validate_fraction(self.reserve_quantile, "reserve_quantile")
        validate_fraction(self.reserve_capacity_limit_fraction, "reserve_capacity_limit_fraction")
        validate_fraction(self.demand_guard_fraction, "demand_guard_fraction")
        validate_positive(self.demand_guard_divisor, "demand_guard_divisor")
        validate_non_negative(self.tariff_headroom_weight, "tariff_headroom_weight")
        validate_non_negative(self.tiebreak_noise_scale, "tiebreak_noise_scale")


@dataclass(frozen=True)
class MethodFactoryConfigSet:
    naive: NaiveMethodConfig = field(default_factory=NaiveMethodConfig)
    greedy: GreedyMethodConfig = field(default_factory=GreedyMethodConfig)
    optimization: OptimizationPenaltyConfig = field(default_factory=OptimizationPenaltyConfig)
    rolling_horizon: RollingHorizonMethodConfig = field(default_factory=RollingHorizonMethodConfig)
    scenario_tree: ScenarioTreeMethodConfig = field(default_factory=ScenarioTreeMethodConfig)
    stochastic: StochasticMethodConfig = field(default_factory=StochasticMethodConfig)


def with_optimization_overrides(
    config: OptimizationPenaltyConfig,
    *,
    peak_penalty: float | None = None,
    unmet_penalty: float | None = None,
) -> OptimizationPenaltyConfig:
    return replace(
        config,
        peak_penalty=config.peak_penalty if peak_penalty is None else peak_penalty,
        unmet_penalty=config.unmet_penalty if unmet_penalty is None else unmet_penalty,
    )
