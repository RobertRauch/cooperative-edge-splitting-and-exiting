from abc import ABC, abstractproperty
from typing import Any, Dict, List, Union
from mec.tasks.task import Computing, Message, Task, ComputingUE
from mec.mec_types import EntityID
from src.common.Location import Location
from src.placeable.Placeable import Placeable
from src.placeable.movable.Movable import Movable
import attr


UEsMap = Dict[EntityID, "UE"]

BSsMap = Dict[EntityID, "BaseStation"]

UEsList = List["UE"]

BSsList = List["BaseStation"]


class DataObjProtocol:
    data: dict


@attr.s()
class DataWrapper:
    obj: DataObjProtocol = attr.ib()

    @property
    def data(self) -> dict:
        return self.obj.data
    # def _get_data(self, key: str, default=None) -> Any:
    #     return self.obj.data.get(key, default)

    # def _set_data(self, key: str, value: Any) -> Any:
    #     self.obj.data[key] = value


@attr.s(kw_only=True)
class MecEntity(ABC):
    tasks_computing: List[Union[Computing, ComputingUE]] = attr.ib(factory=lambda: [], init=False)
    tasks_sending: List[Message] = attr.ib(factory=lambda: [], init=False)
    data: dict = attr.ib(factory=lambda: {}, init=False)
    ips = attr.ib(type=int, default=0)
    used_timestep_interval = 0

    @abstractproperty
    def id(self) -> EntityID:
        pass

    @abstractproperty
    def location(self) -> Location:
        pass


@attr.s(kw_only=True)
class BaseStation(MecEntity, ABC):
    ips: int = attr.ib()
    bandwidth: int = attr.ib()
    resource_blocks: int = attr.ib()
    tx_frequency: int = attr.ib()
    tx_power: float = attr.ib()
    coverage_radius: float = attr.ib()


@attr.s(kw_only=True)
class UE(MecEntity, ABC):
    step_data = {}

    @abstractproperty
    def tasks_to_compute(self) -> List[Task]:
        pass

    def accept_computed_task(self, task: Task) -> None:
        pass

    def generate_tasks(self, dt:float) -> None:
        pass


@attr.s(kw_only=True)
class Vehicle(UE, ABC):
    pass


@attr.s(kw_only=True)
class VehicleIIsmotion(Vehicle):
    _iismotion_movable: Movable = attr.ib()

    @property
    def location(self) -> Location:
        return self._iismotion_movable.getLocation()

    @property
    def id(self) -> EntityID:
        return self._iismotion_movable.id


@attr.s(kw_only=True)
class BaseStationIISMotion(BaseStation):
    _iismotion_placeable: Placeable = attr.ib()

    @property
    def location(self) -> Location:
        return self._iismotion_placeable.getLocation()

    @property
    def id(self) -> EntityID:
        return self._iismotion_placeable.id
