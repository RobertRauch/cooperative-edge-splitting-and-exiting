from abc import ABC, abstractmethod
from typing import Generator, List

from ..task_size import TaskSize
from .task_size_generator import FixedSizeGenerator, TaskSizeGenerator
from ..task import Task


class TaskGenerator(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.task_owner_id = None
        self.target_id = None

    @abstractmethod
    def generate(self, dt: float) -> List[Task]:
        """ Generate tasks """

    @abstractmethod
    def itergenerate(self, dt: float) -> Generator[Task, None, None]:
        """ Generate tasks as generator. """


class PeriodTaskGenerator(TaskGenerator):
    """Class which generates task after a time period."""

    def __init__(
        self,
        period: float,
        task_ttl: float,
        size_generator: TaskSizeGenerator,
        start_delay: float = 0,
    ) -> None:
        super().__init__()
        self._period = period
        self._remained_time: float = start_delay
        self._task_ttl = task_ttl
        self._size_generator = size_generator
        self.was_generated = False

    def generate(self, dt: float) -> List[Task]:
        return [t for t in self.itergenerate(dt)]

    def itergenerate(self, dt: float) -> Generator[Task, None, None]:
        if self.was_generated:
            return
        time_spent = self._remained_time

        while time_spent < dt:
            if self.was_generated:
                break
            yield self._generate_task(used_timestamp_interval=time_spent)
            time_spent += self._period

        self._remained_time = time_spent - dt

    def _generate_task(self, used_timestamp_interval: float) -> Task:
        """Generate a task from attributes."""
        size = self._size_generator.get_size()
        t = Task(
            ttl=self._task_ttl,
            request_size=size.request_size,
            response_size=size.response_size,
            instruction_count=size.instruction_count,
            task_owner_id=self.task_owner_id,
            target_id=self.target_id,
            created_timestep=used_timestamp_interval
        )
        self.was_generated = True
        return t