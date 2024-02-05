from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from ev_fleet_benchmark.model import Scenario, SchedulePlan


class ScheduleMethod(ABC):
    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        method_name = getattr(cls, "name", "")
        if not isinstance(method_name, str) or not method_name.strip():
            raise TypeError(f"ScheduleMethod subclass {cls.__name__} must define a non-empty string 'name'")

    @abstractmethod
    def solve(self, scenario: Scenario) -> SchedulePlan:
        raise NotImplementedError
