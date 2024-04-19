import uuid
from abc import ABC, abstractmethod, abstractproperty
from functools import cached_property
from typing import Optional

import attr

# from mec.entities import VehicleIIsmotion
from mec.mec_types import EntityID


@attr.s(kw_only=True)
class TaskComputationEntity(ABC):
    task: "Task" = attr.ib()
    remaining_computation: float = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.reset_computation()

    @abstractmethod
    def _get_computation_size(self) -> float:
        pass

    def reset_computation(self) -> None:
        self.remaining_computation = self._get_computation_size()


@attr.s(kw_only=True)
class Message(TaskComputationEntity, ABC):
    link_origin_id: Optional[EntityID] = attr.ib(default=None)
    link_destination_id: Optional[EntityID] = attr.ib(default=None)

    # TODO remove uplink/ donwlink => use origin destination instead
    @property
    def is_uplink(self) -> bool:
        return self.origin_id == self.task.task_owner_id

    @property
    def is_downlink(self) -> bool:
        return not self.is_uplink

    @abstractproperty
    def origin_id(self) -> Optional[EntityID]:
        pass

    @abstractproperty
    def destination_id(self) -> Optional[EntityID]:
        pass


@attr.s(kw_only=True)
class Response(Message):
    def _get_computation_size(self) -> float:
        return self.task.response_size

    @property
    def origin_id(self) -> Optional[EntityID]:
        return self.task.target_id

    @property
    def destination_id(self) -> Optional[EntityID]:
        return self.task.task_owner_id


@attr.s(kw_only=True)
class Request(Message):
    def _get_computation_size(self) -> float:
        return self.task.request_size

    @property
    def origin_id(self) -> Optional[EntityID]:
        return self.task.task_owner_id

    @property
    def destination_id(self) -> Optional[EntityID]:
        return self.task.target_id


@attr.s(kw_only=True)
class Computing(TaskComputationEntity):
    def _get_computation_size(self) -> float:
        return self.task.instruction_count


@attr.s(kw_only=True)
class ComputingUE(TaskComputationEntity):
    def _get_computation_size(self) -> float:
        return self.task.instruction_count_ue


@attr.s(kw_only=True)
class Task:
    # Total time to live
    id: str = attr.ib(factory=lambda: str(uuid.uuid4()), on_setattr=attr.setters.frozen)
    ttl: float = attr.ib()
    request_size: float = attr.ib()
    response_size: float = attr.ib()
    instruction_count: float = attr.ib()
    instruction_count_ue: float = attr.ib(default=None)
    task_owner_id: EntityID = attr.ib()

    # id of entity where task should be computed
    target_id: Optional[EntityID] = attr.ib(default=None)

    # idicates how much time in seconds was allready used from dt
    used_timestep_interval: float = attr.ib(default=0.0, init=False)

    # this should be increased by dt after every simulation step
    used_timestep: float = attr.ib(default=0.0, init=False)

    # indicates at which time of dt was task created
    created_timestep: float = attr.ib(default=0.0, on_setattr=attr.setters.frozen)

    data: dict = attr.ib(factory=lambda: {})

    owner: any = attr.ib()

    def __attrs_post_init__(self) -> None:
        self.used_timestep_interval = self.created_timestep

    @property
    def lived_timestep(self) -> float:
        return self.total_used_timestep - self.created_timestep

    @property
    def total_used_timestep(self) -> float:
        return self.used_timestep + self.used_timestep_interval

    @cached_property
    def request(self) -> Request:
        return Request(task=self)

    @cached_property
    def response(self) -> Response:
        return Response(task=self)

    @cached_property
    def computing(self) -> Computing:
        return Computing(task=self)

    @cached_property
    def computing_ue(self) -> ComputingUE:
        return ComputingUE(task=self)
