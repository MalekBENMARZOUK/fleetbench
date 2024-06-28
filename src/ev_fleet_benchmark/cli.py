from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

import typer

from ev_fleet_benchmark.benchmark import BenchmarkConfig, SensitivityStudyConfig, run_benchmark, run_sensitivity_study
from ev_fleet_benchmark.exceptions import FleetBenchError
from ev_fleet_benchmark.methods import build_methods, method_names
from ev_fleet_benchmark.scenarios import describe_families, describe_sensitivity_profiles
from ev_fleet_benchmark.telemetry import LOG_LEVELS, ProgressReporter, configure_cli_logger


def _version_callback(value: bool) -> None:
    if value:
        from ev_fleet_benchmark import __version__

        typer.echo(f"fleetbench {__version__}")
        raise typer.Exit()


app = typer.Typer(
    add_completion=False,
    help="EV fleet charging benchmark CLI - reproducible simulation, optimization, and analysis workflows.",
)


@app.callback(invoke_without_command=True)
def main_callback(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    """EV fleet charging benchmark CLI."""


def _positive_int_callback(ctx: typer.Context, param: typer.CallbackParam, value: int) -> int:
    del ctx, param
    if value <= 0:
        raise typer.BadParameter("must be a positive integer")
    return value


def _time_step_callback(ctx: typer.Context, param: typer.CallbackParam, value: int) -> int:
    del ctx, param
    if value <= 0:
        raise typer.BadParameter("must be a positive integer")
    if 1440 % value != 0:
        raise typer.BadParameter("must divide evenly into 24 hours")
    return value


def _normalize_path(path: Path) -> Path:
    if not str(path).strip():
        raise typer.BadParameter("must be a non-empty path")
    return path


def _run_or_exit(command: Callable[[], Mapping[str, Any]]) -> Mapping[str, Any]:
    try:
        return command()
    except (FleetBenchError, KeyError, ValueError, RuntimeError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc


def _normalize_log_level(log_level: str) -> str:
    normalized_level = log_level.lower()
    if normalized_level not in LOG_LEVELS:
        raise typer.BadParameter(f"must be one of {', '.join(LOG_LEVELS)}")
    return normalized_level


def _build_progress_reporter_with_level(
    verbose: bool, log_level: str, progress_file: Path | None
) -> ProgressReporter | None:
    resolved_level = "info" if verbose and log_level == "silent" else log_level
    logger = configure_cli_logger(resolved_level)
    if logger is None and progress_file is None:
        return None
    return ProgressReporter(logger=logger, jsonl_path=progress_file)


@app.command("describe-families")
def describe_families_command() -> None:
    for family in describe_families():
        typer.echo(f"{family['name']}: {family['description']}")
        typer.echo(
            "  "
            f"fleet_size_range={tuple(family['fleet_size_range'])}, "
            f"capacity_tightness={tuple(family['capacity_tightness'])}, "
            f"delay_probability={family['delay_probability']}, "
            f"site_derate_probability={family['site_derate_probability']}"
        )


@app.command("list-methods")
def list_methods_command() -> None:
    for name in method_names():
        typer.echo(name)


@app.command("list-sensitivity")
def list_sensitivity_command() -> None:
    for profile in describe_sensitivity_profiles():
        typer.echo(f"{profile['name']}: {profile['description']}")


@app.command("benchmark")
def benchmark_command(
    output_dir: Path = typer.Option(
        Path("results/latest"),
        callback=lambda ctx, param, value: _normalize_path(value),
        help="Directory where benchmark artifacts are written.",
    ),
    family: list[str] | None = typer.Option(
        None, "--family", help="Scenario family to evaluate. Repeat to select multiple."
    ),
    method: list[str] | None = typer.Option(None, "--method", help="Method to evaluate. Repeat to select multiple."),
    seeds: list[int] = typer.Option(..., "--seeds", help="Random seeds used to instantiate scenario families."),
    time_step_minutes: int = typer.Option(30, callback=_time_step_callback, help="Simulation time-step in minutes."),
    generate_plots: bool = typer.Option(True, "--plots/--no-plots", help="Generate benchmark plots."),
    publication_outputs: bool = typer.Option(
        True, "--publication/--no-publication", help="Write publication-style summary tables and report."
    ),
    bootstrap_samples: int = typer.Option(
        1000, callback=_positive_int_callback, help="Bootstrap resamples for publication confidence intervals."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Emit structured progress logs to stderr during execution."),
    log_level: Annotated[
        str,
        typer.Option(
            callback=lambda ctx, param, value: _normalize_log_level(value),
            help="Progress log level: debug, info, warning, error, or silent.",
        ),
    ] = "silent",
    progress_file: Path | None = typer.Option(
        None, help="Optional JSONL file that receives structured progress events."
    ),
    workers: int = typer.Option(
        1, "--workers", min=1, help="Number of parallel workers for method execution within each scenario."
    ),
) -> None:
    requested_families = family or [entry["name"] for entry in describe_families()]
    progress_reporter = _build_progress_reporter_with_level(verbose, log_level, progress_file)
    outputs = _run_or_exit(
        lambda: run_benchmark(
            BenchmarkConfig(
                family_names=requested_families,
                seeds=seeds,
                time_step_minutes=time_step_minutes,
                output_dir=str(output_dir),
                generate_plots=generate_plots,
                publication_outputs=publication_outputs,
                bootstrap_samples=bootstrap_samples,
                max_workers=workers,
            ),
            methods=build_methods(method),
            progress_reporter=progress_reporter,
        )
    )
    typer.echo(f"Benchmark completed. Wrote outputs to {output_dir}")
    typer.echo(f"Scenario runs: {len(outputs['scenario_metrics'])}")


@app.command("study")
def study_command(
    output_dir: Path = typer.Option(
        Path("results/study"),
        callback=lambda ctx, param, value: _normalize_path(value),
        help="Directory where the larger benchmark study is written.",
    ),
    start_seed: int = typer.Option(101, help="First seed in the study range."),
    seed_count: int = typer.Option(
        24, callback=_positive_int_callback, help="Number of consecutive seeds to evaluate."
    ),
    time_step_minutes: int = typer.Option(30, callback=_time_step_callback, help="Simulation time-step in minutes."),
    generate_plots: bool = typer.Option(True, "--plots/--no-plots", help="Generate study plots."),
    method: list[str] | None = typer.Option(None, "--method", help="Optional subset of methods for the study."),
    bootstrap_samples: int = typer.Option(
        2000, callback=_positive_int_callback, help="Bootstrap resamples for study confidence intervals."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Emit structured progress logs to stderr during execution."),
    log_level: Annotated[
        str,
        typer.Option(
            callback=lambda ctx, param, value: _normalize_log_level(value),
            help="Progress log level: debug, info, warning, error, or silent.",
        ),
    ] = "silent",
    progress_file: Path | None = typer.Option(
        None, help="Optional JSONL file that receives structured progress events."
    ),
    workers: int = typer.Option(
        1, "--workers", min=1, help="Number of parallel workers for method execution within each scenario."
    ),
) -> None:
    families = [entry["name"] for entry in describe_families()]
    seeds = list(range(start_seed, start_seed + seed_count))
    progress_reporter = _build_progress_reporter_with_level(verbose, log_level, progress_file)
    outputs = _run_or_exit(
        lambda: run_benchmark(
            BenchmarkConfig(
                family_names=families,
                seeds=seeds,
                time_step_minutes=time_step_minutes,
                output_dir=str(output_dir),
                generate_plots=generate_plots,
                publication_outputs=True,
                bootstrap_samples=bootstrap_samples,
                max_workers=workers,
            ),
            methods=build_methods(method),
            progress_reporter=progress_reporter,
        ),
    )
    typer.echo(f"Study completed. Wrote outputs to {output_dir}")
    typer.echo(f"Scenario runs: {len(outputs['scenario_metrics'])}")


@app.command("sensitivity")
def sensitivity_command(
    output_dir: Path = typer.Option(
        Path("results/sensitivity"),
        callback=lambda ctx, param, value: _normalize_path(value),
        help="Directory where the sensitivity study is written.",
    ),
    start_seed: int = typer.Option(101, help="First seed in the sensitivity range."),
    seed_count: int = typer.Option(
        12, callback=_positive_int_callback, help="Number of consecutive seeds to evaluate."
    ),
    profile: list[str] | None = typer.Option(
        None, "--profile", help="Sensitivity profile to evaluate. Repeat to select multiple."
    ),
    method: list[str] | None = typer.Option(
        None, "--method", help="Optional subset of methods for the sensitivity study."
    ),
    bootstrap_samples: int = typer.Option(
        1000, callback=_positive_int_callback, help="Bootstrap resamples for sensitivity summaries."
    ),
    time_step_minutes: int = typer.Option(30, callback=_time_step_callback, help="Simulation time-step in minutes."),
    verbose: bool = typer.Option(False, "--verbose", help="Emit structured progress logs to stderr during execution."),
    log_level: Annotated[
        str,
        typer.Option(
            callback=lambda ctx, param, value: _normalize_log_level(value),
            help="Progress log level: debug, info, warning, error, or silent.",
        ),
    ] = "silent",
    progress_file: Path | None = typer.Option(
        None, help="Optional JSONL file that receives structured progress events."
    ),
    workers: int = typer.Option(
        1, "--workers", min=1, help="Number of parallel workers for method execution within each scenario."
    ),
) -> None:
    families = [entry["name"] for entry in describe_families()]
    profiles = profile or [entry["name"] for entry in describe_sensitivity_profiles()]
    seeds = list(range(start_seed, start_seed + seed_count))
    progress_reporter = _build_progress_reporter_with_level(verbose, log_level, progress_file)
    outputs = _run_or_exit(
        lambda: run_sensitivity_study(
            SensitivityStudyConfig(
                profile_names=profiles,
                family_names=families,
                seeds=seeds,
                time_step_minutes=time_step_minutes,
                output_dir=str(output_dir),
                bootstrap_samples=bootstrap_samples,
                max_workers=workers,
            ),
            methods=build_methods(method),
            progress_reporter=progress_reporter,
        ),
    )
    typer.echo(f"Sensitivity study completed. Wrote outputs to {output_dir}")
    typer.echo(f"Scenario runs: {len(outputs['sensitivity_scenario_metrics'])}")


if __name__ == "__main__":
    app()
