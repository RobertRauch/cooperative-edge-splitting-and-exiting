from typing import Protocol

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
