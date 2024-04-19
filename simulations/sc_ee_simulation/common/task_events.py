from typing import Protocol

import numpy as np

from mec.tasks.task import Task


class TaskEventsProtocol(Protocol):
    def accepted_task(self, task: Task) -> None:
        pass

    def generated_task(self, task: Task) -> None:
        pass

    def discarded_task(self, task: Task) -> None:
        pass


class DummyTaskEventsListener(TaskEventsProtocol):
    def accepted_task(self, task: Task) -> None:
        pass

    def generated_task(self, task: Task) -> None:
        pass

    def discarded_task(self, task: Task) -> None:
        pass


class DDPGTaskEventsListener(TaskEventsProtocol):
    def __init__(self):
        self.execution_time = 0
        self.executed_tasks = 0
        self.transmission_data_size = 0
        self.transmission_data_time = 0
        self.discarded_tasks = 0
        self.accepted_tasks = 0
        self.generated_tasks = 0
        self.sum_exits = 0
        self.device_utilization = 0
        self.server_utilization = 0
        self.correct_predictions = []

    def reset(self):
        self.execution_time = 0
        self.executed_tasks = 0
        self.transmission_data_size = 0
        self.transmission_data_time = 0
        self.discarded_tasks = 0
        self.accepted_tasks = 0
        self.generated_tasks = 0
        self.sum_exits = 0
        self.device_utilization = 0
        self.server_utilization = 0
        # self.correct_predictions.clear()

    def get_average_latency(self) -> float:
        return 0 if self.executed_tasks == 0 else self.execution_time / self.executed_tasks

    def get_average_transmission_rate(self) -> float:
        return 0 if self.transmission_data_time == 0 else self.transmission_data_size / self.transmission_data_time

    def get_accuracy(self) -> float:
        if len(self.correct_predictions) == 0:
            return 0
        return np.average(self.correct_predictions)

    def get_average_exit(self) -> float:
        return 0 if self.executed_tasks == 0 else self.sum_exits / self.executed_tasks

    def accepted_task(self, task: Task) -> None:
        self.correct_predictions.append(1 if "prediction" in task.data and task.data["prediction"] == task.data["target"] else 0)
        self.save(task)
        self.accepted_tasks += 1

    def generated_task(self, task: Task) -> None:
        self.generated_tasks += 1

    def discarded_task(self, task: Task) -> None:
        self.correct_predictions.append(0)
        self.save(task)
        self.discarded_tasks += 1

    def save(self, task: Task) -> None:
        if round(task.ttl, 7) < round(task.lived_timestep, 7):
            raise Exception("Task went over time to live {}".format(task.lived_timestep))
        self.execution_time += task.lived_timestep
        self.executed_tasks += 1
        self.transmission_data_time += 0 if "uplink" not in task.data.keys() else task.data["uplink"]
        self.transmission_data_size += task.request_size - task.request.remaining_computation
        self.transmission_data_time += 0 if "downlink" not in task.data.keys() else task.data["downlink"]
        self.transmission_data_size += task.response_size - task.response.remaining_computation
        self.sum_exits += task.data["chosen_exit"] if "chosen_exit" in task.data else -100
        self.device_utilization += 0 if "computation" not in task.data.keys() else 0 if "vehicle" not in task.data["computation"].keys() else task.data["computation"]["vehicle"]
        self.server_utilization += 0 if "computation" not in task.data.keys() else 0 if "server" not in task.data["computation"].keys() else task.data["computation"]["server"]
        if len(self.correct_predictions) > 200:
            self.correct_predictions.pop(0)
