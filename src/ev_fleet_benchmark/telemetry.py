from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "silent": logging.CRITICAL + 1,
}

_telemetry_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProgressEvent:
    event_type: str
    run_kind: str
    completed_runs: int
    total_runs: int
    output_dir: str
    elapsed_s: float
    family_name: str | None = None
    profile_name: str | None = None
    scenario_name: str | None = None
    method_name: str | None = None
    seed: int | None = None
    solve_time_s: float | None = None
    status: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class ProgressReporter:
    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        jsonl_path: Path | None = None,
        text_writer: Callable[[str], None] | None = None,
    ) -> None:
        self.logger = logger
        self.jsonl_path = jsonl_path
        self.text_writer = text_writer
        self._write_lock = threading.Lock()

    def emit(self, event: ProgressEvent) -> None:
        payload = event.to_payload()
        if self.logger is not None:
            self.logger.info(format_progress_event(event))
        if self.text_writer is not None:
            self.text_writer(format_progress_event(event))
        if self.jsonl_path is not None:
            try:
                with self._write_lock:
                    self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.jsonl_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(payload, sort_keys=True))
                        handle.write("\n")
            except (OSError, ValueError) as exc:
                _telemetry_logger.error("Failed to write telemetry to %s: %s", self.jsonl_path, exc)


def configure_cli_logger(log_level: str) -> logging.Logger | None:
    normalized_level = log_level.lower()
    if normalized_level not in LOG_LEVELS:
        raise ValueError(f"Unsupported log level: {log_level}")
    if normalized_level == "silent":
        return None
    logger = logging.getLogger("ev_fleet_benchmark.progress")
    logger.setLevel(LOG_LEVELS[normalized_level])
    has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    for existing_handler in logger.handlers:
        existing_handler.setLevel(LOG_LEVELS[normalized_level])
    logger.propagate = False
    return logger


def format_progress_event(event: ProgressEvent) -> str:
    prefix = f"[{event.run_kind}] {event.completed_runs}/{event.total_runs} {event.event_type}"
    details = [f"elapsed={event.elapsed_s:.2f}s"]
    if event.family_name is not None:
        details.append(f"family={event.family_name}")
    if event.profile_name is not None:
        details.append(f"profile={event.profile_name}")
    if event.scenario_name is not None:
        details.append(f"scenario={event.scenario_name}")
    if event.method_name is not None:
        details.append(f"method={event.method_name}")
    if event.seed is not None:
        details.append(f"seed={event.seed}")
    if event.solve_time_s is not None:
        details.append(f"solve={event.solve_time_s:.4f}s")
    if event.status is not None:
        details.append(f"status={event.status}")
    return f"{prefix} | {' '.join(details)}"
