from abc import ABC, abstractmethod
from typing import Dict, List
import attr

from .task_generator import TaskGenerator

@attr.s(kw_only=True)
class TaskTargetSetter(ABC):
    _task_generators: List[TaskGenerator] = attr.ib()

    def set_targets(self, current_step: int) -> None:
        """Update a target id of the task generators"""
        for tg in self._task_generators:
            self._set_target(current_step, tg)

    @abstractmethod
    def _set_target(self, current_step: int, tg: TaskGenerator) -> None:
        """Set target id of a task generator"""