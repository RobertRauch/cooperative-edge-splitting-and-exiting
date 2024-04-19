from mec.tasks.task_generators.task_size_generator import TaskSizeGenerator
from simulations.sc_ee_simulation.task.task_executor import TaskExecutor
from mec.tasks.task_size import TaskSize
from typing import NamedTuple


# @attr.s(kw_only=True)
class TaskData(NamedTuple):
    prediction: int
    target: int
    chosen_exit: int


class OursTaskSizeGenerator(TaskSizeGenerator):
    def __init__(self, task_executor: TaskExecutor, input_size: int = 12288):
        self.task_executor = task_executor
        self.input_size = input_size

    def get_size(self) -> (TaskSize, TaskData):
        ed, es, size, prediction, chosen_exit, target = self.task_executor.execute_model()
        output_size = 40 if not es == 0 else 0
        size = self.input_size if ed == 0 else size
        size = 0 if es == 0 else size
        # return TaskSize(1, 1, 3, 1), TaskData(prediction, target, chosen_exit)
        return TaskSize(size/1048576, output_size/1048576, es, ed), TaskData(prediction, target, chosen_exit)
