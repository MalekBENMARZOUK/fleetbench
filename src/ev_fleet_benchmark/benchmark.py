from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, cast

import pandas as pd

from ev_fleet_benchmark.exceptions import MethodExecutionError, PlanValidationError
from ev_fleet_benchmark.methods import build_methods

if TYPE_CHECKING:
    from ev_fleet_benchmark.methods.base import ScheduleMethod
from ev_fleet_benchmark.model import EvaluationResult, Scenario
from ev_fleet_benchmark.reporting import (
    aggregate_results,
    create_plots,
    create_publication_tables,
    create_sensitivity_tables,
    write_markdown_report,
    write_publication_report,
    write_sensitivity_report,
)
from ev_fleet_benchmark.scenarios import (
    apply_sensitivity_profile,
    default_scenario_families,
    default_sensitivity_profiles,
    generate_scenario,
    save_scenario_json,
)
from ev_fleet_benchmark.simulator import evaluate_plan
from ev_fleet_benchmark.telemetry import ProgressEvent, ProgressReporter
from ev_fleet_benchmark.validation import (
    validate_positive_int,
    validate_time_step_minutes,
)

ScenarioModifier = Callable[[Scenario], Scenario]

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkConfig:
    family_names: list[str]
    seeds: list[int]
    time_step_minutes: int = 30
    output_dir: str = "results/latest"
    generate_plots: bool = True
    publication_outputs: bool = True
    bootstrap_samples: int = 1000
    max_workers: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family_names",
            _normalize_string_list(self.family_names, "family_names"),
        )
        object.__setattr__(self, "seeds", _normalize_int_list(self.seeds, "seeds"))
        validate_time_step_minutes(self.time_step_minutes)
        _validate_output_dir(self.output_dir)
        validate_positive_int(self.bootstrap_samples, "bootstrap_samples")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")


@dataclass(frozen=True)
class SensitivityStudyConfig:
    profile_names: list[str]
    family_names: list[str]
    seeds: list[int]
    time_step_minutes: int = 30
    output_dir: str = "results/sensitivity"
    bootstrap_samples: int = 1000
    max_workers: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_names",
            _normalize_string_list(self.profile_names, "profile_names"),
        )
        object.__setattr__(
            self,
            "family_names",
            _normalize_string_list(self.family_names, "family_names"),
        )
        object.__setattr__(self, "seeds", _normalize_int_list(self.seeds, "seeds"))
        validate_time_step_minutes(self.time_step_minutes)
        _validate_output_dir(self.output_dir)
        validate_positive_int(self.bootstrap_samples, "bootstrap_samples")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")


def default_methods() -> list[ScheduleMethod]:
    return build_methods()


def run_benchmark(
    config: BenchmarkConfig,
    methods: Iterable[ScheduleMethod] | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> dict[str, pd.DataFrame]:
    return _run_benchmark_core(
        config,
        methods=methods,
        scenario_modifier=None,
        progress_reporter=progress_reporter,
        run_kind="benchmark",
    )


def run_sensitivity_study(
    config: SensitivityStudyConfig,
    methods: Iterable[ScheduleMethod] | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> dict[str, pd.DataFrame]:
    families = default_scenario_families()
    profiles = default_sensitivity_profiles()
    _validate_requested_names(config.family_names, families, "scenario families")
    _validate_requested_names(config.profile_names, profiles, "sensitivity profiles")

    selected_methods = _resolve_methods(methods)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios_dir = output_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    total_runs = len(config.profile_names) * len(config.family_names) * len(config.seeds) * len(selected_methods)
    completed_runs = 0
    started_at = perf_counter()
    _emit_progress(
        progress_reporter,
        ProgressEvent(
            event_type="run_started",
            run_kind="sensitivity",
            completed_runs=0,
            total_runs=total_runs,
            output_dir=str(output_dir),
            elapsed_s=0.0,
        ),
    )

    summary_rows: list[dict[str, object]] = []
    vehicle_rows: list[dict[str, object]] = []
    site_rows: list[dict[str, object]] = []

    for profile_name in config.profile_names:
        profile_scenarios_dir = scenarios_dir / profile_name
        profile_scenarios_dir.mkdir(parents=True, exist_ok=True)
        for family_name in config.family_names:
            for seed in config.seeds:
                scenario = generate_scenario(
                    family_name=family_name,
                    seed=seed,
                    time_step_minutes=config.time_step_minutes,
                )
                scenario = apply_sensitivity_profile(scenario, profile_name)
                save_scenario_json(scenario, profile_scenarios_dir / f"{scenario.name}.json")
                _emit_progress(
                    progress_reporter,
                    ProgressEvent(
                        event_type="scenario_started",
                        run_kind="sensitivity",
                        completed_runs=completed_runs,
                        total_runs=total_runs,
                        output_dir=str(output_dir),
                        elapsed_s=perf_counter() - started_at,
                        family_name=family_name,
                        profile_name=profile_name,
                        scenario_name=scenario.name,
                        seed=seed,
                    ),
                )
                method_evaluations = _run_methods_on_scenario(selected_methods, scenario, config.max_workers)
                for method, evaluation in method_evaluations:
                    evaluation.summary["sensitivity_profile"] = profile_name
                    evaluation.summary["base_scenario_name"] = f"{family_name}_seed_{seed}"
                    for row in evaluation.per_vehicle:
                        row["sensitivity_profile"] = profile_name
                    for row in evaluation.site_profile:
                        row["sensitivity_profile"] = profile_name
                    summary_rows.append(evaluation.summary)
                    vehicle_rows.extend(evaluation.per_vehicle)
                    site_rows.extend(evaluation.site_profile)
                    completed_runs += 1
                    _emit_progress(
                        progress_reporter,
                        ProgressEvent(
                            event_type="method_completed",
                            run_kind="sensitivity",
                            completed_runs=completed_runs,
                            total_runs=total_runs,
                            output_dir=str(output_dir),
                            elapsed_s=perf_counter() - started_at,
                            family_name=family_name,
                            profile_name=profile_name,
                            scenario_name=scenario.name,
                            method_name=method.name,
                            seed=seed,
                            solve_time_s=float(evaluation.summary.get("solve_time_s", 0.0)),
                            status=str(evaluation.summary.get("status", "completed")),
                        ),
                    )

    summary_df = pd.DataFrame(summary_rows)
    vehicle_df = pd.DataFrame(vehicle_rows)
    site_df = pd.DataFrame(site_rows)
    sensitivity_tables = create_sensitivity_tables(
        summary_df,
        output_dir / "publication",
        bootstrap_samples=config.bootstrap_samples,
    )

    _write_csv(summary_df, output_dir / "sensitivity_scenario_metrics.csv")
    _write_csv(vehicle_df, output_dir / "sensitivity_vehicle_metrics.csv")
    _write_csv(site_df, output_dir / "sensitivity_site_load_profiles.csv")

    metadata = {
        "profile_names": config.profile_names,
        "family_names": config.family_names,
        "seeds": config.seeds,
        "time_step_minutes": config.time_step_minutes,
        "methods": [method.name for method in selected_methods],
        "bootstrap_samples": config.bootstrap_samples,
    }
    _write_json(output_dir / "sensitivity_metadata.json", metadata)
    write_sensitivity_report(
        output_dir / "publication" / "sensitivity_summary.md",
        cast("dict[str, pd.DataFrame]", dict(sensitivity_tables)),
        metadata,
    )
    _emit_progress(
        progress_reporter,
        ProgressEvent(
            event_type="run_completed",
            run_kind="sensitivity",
            completed_runs=completed_runs,
            total_runs=total_runs,
            output_dir=str(output_dir),
            elapsed_s=perf_counter() - started_at,
            status="completed",
        ),
    )

    return {
        "sensitivity_scenario_metrics": summary_df,
        "sensitivity_vehicle_metrics": vehicle_df,
        "sensitivity_site_load_profiles": site_df,
        "sensitivity_profile_aggregate": sensitivity_tables["sensitivity_profile_aggregate"],
        "sensitivity_profile_ranking": sensitivity_tables["sensitivity_profile_ranking"],
        "sensitivity_method_robustness": sensitivity_tables["sensitivity_method_robustness"],
        "sensitivity_profile_deltas": sensitivity_tables["sensitivity_profile_deltas"],
    }


def _run_benchmark_core(
    config: BenchmarkConfig,
    methods: Iterable[ScheduleMethod] | None,
    scenario_modifier: ScenarioModifier | None,
    progress_reporter: ProgressReporter | None,
    run_kind: str,
) -> dict[str, pd.DataFrame]:
    families = default_scenario_families()
    _validate_requested_names(config.family_names, families, "scenario families")

    selected_methods = _resolve_methods(methods)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios_dir = output_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    total_runs = len(config.family_names) * len(config.seeds) * len(selected_methods)
    completed_runs = 0
    started_at = perf_counter()
    _emit_progress(
        progress_reporter,
        ProgressEvent(
            event_type="run_started",
            run_kind=run_kind,
            completed_runs=0,
            total_runs=total_runs,
            output_dir=str(output_dir),
            elapsed_s=0.0,
        ),
    )

    summary_rows: list[dict[str, object]] = []
    vehicle_rows: list[dict[str, object]] = []
    site_rows: list[dict[str, object]] = []

    for family_name in config.family_names:
        for seed in config.seeds:
            scenario = generate_scenario(
                family_name=family_name,
                seed=seed,
                time_step_minutes=config.time_step_minutes,
            )
            if scenario_modifier is not None:
                scenario = scenario_modifier(scenario)
            save_scenario_json(scenario, scenarios_dir / f"{scenario.name}.json")
            _emit_progress(
                progress_reporter,
                ProgressEvent(
                    event_type="scenario_started",
                    run_kind=run_kind,
                    completed_runs=completed_runs,
                    total_runs=total_runs,
                    output_dir=str(output_dir),
                    elapsed_s=perf_counter() - started_at,
                    family_name=family_name,
                    scenario_name=scenario.name,
                    seed=seed,
                ),
            )
            method_evaluations = _run_methods_on_scenario(selected_methods, scenario, config.max_workers)
            for method, evaluation in method_evaluations:
                summary_rows.append(evaluation.summary)
                vehicle_rows.extend(evaluation.per_vehicle)
                site_rows.extend(evaluation.site_profile)
                completed_runs += 1
                _emit_progress(
                    progress_reporter,
                    ProgressEvent(
                        event_type="method_completed",
                        run_kind=run_kind,
                        completed_runs=completed_runs,
                        total_runs=total_runs,
                        output_dir=str(output_dir),
                        elapsed_s=perf_counter() - started_at,
                        family_name=family_name,
                        scenario_name=scenario.name,
                        method_name=method.name,
                        seed=seed,
                        solve_time_s=float(evaluation.summary.get("solve_time_s", 0.0)),
                        status=str(evaluation.summary.get("status", "completed")),
                    ),
                )

    summary_df = pd.DataFrame(summary_rows)
    vehicle_df = pd.DataFrame(vehicle_rows)
    site_df = pd.DataFrame(site_rows)
    family_aggregate_df, overall_aggregate_df = aggregate_results(summary_df)

    _write_csv(summary_df, output_dir / "scenario_metrics.csv")
    _write_csv(vehicle_df, output_dir / "vehicle_metrics.csv")
    _write_csv(site_df, output_dir / "site_load_profiles.csv")
    _write_csv(family_aggregate_df, output_dir / "aggregate_by_family.csv")
    _write_csv(overall_aggregate_df, output_dir / "aggregate_overall.csv")

    metadata = {
        "family_names": config.family_names,
        "seeds": config.seeds,
        "time_step_minutes": config.time_step_minutes,
        "methods": [method.name for method in selected_methods],
        "bootstrap_samples": config.bootstrap_samples,
    }
    _write_json(output_dir / "benchmark_metadata.json", metadata)
    write_markdown_report(
        output_dir / "benchmark_report.md",
        summary_df,
        family_aggregate_df,
        overall_aggregate_df,
        metadata,
    )
    if config.generate_plots:
        plots_dir = output_dir / "plots"
        create_plots(summary_df, family_aggregate_df, plots_dir)
    if config.publication_outputs:
        publication_dir = output_dir / "publication"
        publication_tables = create_publication_tables(
            summary_df, publication_dir, bootstrap_samples=config.bootstrap_samples
        )
        write_publication_report(
            publication_dir / "study_summary.md",
            publication_tables["method_ranking"],
            publication_tables["family_winners"],
            {
                **metadata,
                "method_confidence_df": publication_tables["method_confidence"],
                "pairwise_df": publication_tables["pairwise_comparison"],
            },
        )
    _emit_progress(
        progress_reporter,
        ProgressEvent(
            event_type="run_completed",
            run_kind=run_kind,
            completed_runs=completed_runs,
            total_runs=total_runs,
            output_dir=str(output_dir),
            elapsed_s=perf_counter() - started_at,
            status="completed",
        ),
    )

    return {
        "scenario_metrics": summary_df,
        "vehicle_metrics": vehicle_df,
        "site_load_profiles": site_df,
        "aggregate_by_family": family_aggregate_df,
        "aggregate_overall": overall_aggregate_df,
    }


def _resolve_methods(methods: Iterable[ScheduleMethod] | None) -> list[ScheduleMethod]:
    selected_methods = list(methods or default_methods())
    if not selected_methods:
        raise ValueError("At least one benchmark method must be provided")
    return selected_methods


def _normalize_string_list(values: list[str], field_name: str) -> list[str]:
    normalized = [value.strip() for value in values if value.strip()]
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one value")
    return list(dict.fromkeys(normalized))


def _normalize_int_list(values: list[int], field_name: str) -> list[int]:
    if not values:
        raise ValueError(f"{field_name} must contain at least one value")
    return list(dict.fromkeys(int(value) for value in values))


def _validate_requested_names(requested: list[str], available: Mapping[str, object], label: str) -> None:
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(f"Unknown {label} requested: {unknown}. Available values: {', '.join(sorted(available))}")


def _validate_output_dir(output_dir: str) -> None:
    if not output_dir.strip():
        raise ValueError("output_dir must be a non-empty path")


def _write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _emit_progress(progress_reporter: ProgressReporter | None, event: ProgressEvent) -> None:
    if progress_reporter is not None:
        progress_reporter.emit(event)


def _run_methods_on_scenario(
    methods: list[ScheduleMethod],
    scenario: Scenario,
    max_workers: int,
) -> list[tuple[ScheduleMethod, EvaluationResult]]:
    if max_workers <= 1 or len(methods) <= 1:
        return [(method, _execute_method(method, scenario)) for method in methods]

    results: list[tuple[ScheduleMethod, EvaluationResult] | None] = [None] * len(methods)
    with ProcessPoolExecutor(max_workers=min(max_workers, len(methods))) as pool:
        future_to_index = {pool.submit(_execute_method, method, scenario): idx for idx, method in enumerate(methods)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = (methods[idx], future.result())
            except (MethodExecutionError, PlanValidationError):
                raise
            except Exception as exc:
                raise MethodExecutionError(methods[idx].name, scenario.name, reason=str(exc)) from exc
    return [r for r in results if r is not None]


def _execute_method(method: ScheduleMethod, scenario: Scenario) -> EvaluationResult:
    try:
        plan = method.solve(scenario)
    except (MethodExecutionError, PlanValidationError):
        raise
    except (ValueError, RuntimeError, TypeError, ArithmeticError) as exc:
        raise MethodExecutionError(method.name, scenario.name, reason=str(exc)) from exc
    try:
        return evaluate_plan(scenario, plan)
    except (MethodExecutionError, PlanValidationError):
        raise
    except (ValueError, RuntimeError, TypeError, ArithmeticError) as exc:
        raise PlanValidationError(
            f"Method {method.name} produced an invalid plan for scenario {scenario.name}: {exc}"
        ) from exc
