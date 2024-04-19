from mec.tasks.task_generators.task_generator import TaskGenerator
from mec.tasks.task_generators.task_size_generator import TaskSizeGenerator
from typing import List, Generator
from mec.tasks.task import Task
from simulations.sc_ee_simulation.common.entities import SimulationVehicle


# TODO add logging of task accuracy
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
        self.owner = None

    def set_owner(self, owner: SimulationVehicle):
        self.owner = owner

    def generate(self, dt: float) -> List[Task]:
        return [t for t in self.itergenerate(dt)]

    def itergenerate(self, dt: float) -> Generator[Task, None, None]:
        time_spent = self._remained_time

        while time_spent < dt:
            yield self._generate_task(used_timestamp_interval=time_spent)
            time_spent += self._period

        self._remained_time = time_spent - dt

    def _generate_task(self, used_timestamp_interval: float) -> Task:
        """Generate a task from attributes."""
        size, data = self._size_generator.get_size()
        t = Task(
            ttl=self._task_ttl,
            request_size=size.request_size,
            response_size=size.response_size,
            instruction_count=size.instruction_count,
            instruction_count_ue=size.instruction_count_ue,
            task_owner_id=self.task_owner_id,
            target_id=self.target_id,
            created_timestep=used_timestamp_interval,
            owner=self.owner
        )
        t.data["prediction"] = data.prediction
        t.data["chosen_exit"] = data.chosen_exit
        t.data["target"] = data.target
        return t