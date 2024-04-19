from abc import ABC, abstractmethod

from ..task_size import TaskSize


class TaskSizeGenerator(ABC):
    @abstractmethod
    def get_size(self) -> TaskSize:
        """Get task size"""


class FixedSizeGenerator(TaskSizeGenerator):
    def __init__(self, task_size: TaskSize) -> None:
        super().__init__()
        self._task_size = task_size

    def get_size(self) -> TaskSize:
        return self._task_size