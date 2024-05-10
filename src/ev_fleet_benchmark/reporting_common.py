from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_PUBLICATION_REFERENCE_METHOD = "optimization_ortools"
BASELINE_SENSITIVITY_PROFILE = "baseline"


def require_columns(dataframe: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns: {missing}")


def validate_bootstrap_samples(bootstrap_samples: int) -> None:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")


def metadata_str_list(metadata: dict[str, object], key: str, *, required: bool = False) -> list[str]:
    values = _metadata_values(metadata, key, required=required)
    return [str(value) for value in values]


def metadata_int_list(metadata: dict[str, object], key: str, *, required: bool = False) -> list[int]:
    values = _metadata_values(metadata, key, required=required)
    return [int(str(value)) for value in values]


def _metadata_values(metadata: dict[str, object], key: str, *, required: bool) -> Iterable[object]:
    if key not in metadata:
        if required:
            raise ValueError(f"Required metadata key '{key}' not found")
        return ()
    values = metadata[key]
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ValueError(f"Metadata key '{key}' must contain a sequence of values")
    return cast("Iterable[object]", values)
