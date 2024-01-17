from __future__ import annotations

import math


def validate_positive(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field_name} must be a finite positive number, got {value!r}")


def validate_non_negative(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number, got {value!r}")


def validate_fraction(value: float, field_name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1, got {value!r}")


def validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")


def validate_time_step_minutes(time_step_minutes: int) -> None:
    validate_positive_int(time_step_minutes, "time_step_minutes")
    if 1440 % time_step_minutes != 0:
        raise ValueError(f"time_step_minutes must divide evenly into 24 hours (1440 minutes), got {time_step_minutes}")
