from typing import List

import attr
import structlog

import os
print(os.getcwd())

from mec.entities import BaseStationIISMotion, VehicleIIsmotion
from mec.iismotion_extension.locations_loader import LocationsLoader
from mec.tasks.task import Task
from mec.tasks.task_generators.task_generator import TaskGenerator
from src.city.Map import Map
from src.movement.ActorCollection import ActorCollection
from src.placeable.Placeable import Placeable, PlaceableFactory

from simulations.sc_ee_simulation.config import BaseStationData, Weights
from simulations.sc_ee_simulation.common.task_events import DummyTaskEventsListener, TaskEventsProtocol, DDPGTaskEventsListener
from simulations.sc_ee_simulation.common.strategy import Strategy, Requirements
from simulations.sc_ee_simulation.simulation.maddpg import Agent


@attr.s(kw_only=True)
class SimulationVehicle(VehicleIIsmotion):
    task_generator: TaskGenerator = attr.ib()
    _event_listener: TaskEventsProtocol = attr.ib(
        factory=lambda: DummyTaskEventsListener()
    )
    ddpg_event_listener: DDPGTaskEventsListener = attr.ib()
    _generated_tasks: List[Task] = attr.ib(factory=lambda: [], init=False)
    strategy: Strategy = attr.ib()
    requirements: Requirements = attr.ib()
    weights: Weights = attr.ib()
    ma_agent: Agent = attr.ib()

    _log: structlog.stdlib.BoundLogger = attr.ib(init=False)
    prev_state: List[float] = attr.ib()
    selected_action: List[float] = attr.ib()
    log_idx: int = attr.ib()
    model_save_location: str = attr.ib(default=None)

    def __attrs_post_init__(self) -> None:
        self._log = structlog.get_logger().bind(ue_id=self.id)

    @property
    def tasks_to_compute(self) -> List[Task]:
        return self._generated_tasks

    def generate_tasks(self, dt: float) -> None:
        for task in self.task_generator.itergenerate(dt):
            # self._log.debug(
            #     "simulation_vehicle.task_generated",
            #     created_of_dt=task.created_timestep,
            #     task_id=task.id,
            # )
            self._generated_tasks.append(task)
            self._event_listener.generated_task(task)
            self.ddpg_event_listener.generated_task(task)

    def accept_computed_task(self, task: Task) -> None:
        # self._log.debug(
        #     "simulation_vehicle.accept_computed_task",
        #     latency=task.lived_timestep,
        #     task_id=task.id,
        # )
        self._event_listener.accepted_task(task)
        self.ddpg_event_listener.accepted_task(task)


class BaseStationPlaceableFactory(PlaceableFactory):
    def createPlaceable(self, locationsTable, map: Map) -> Placeable:
        return Placeable(locationsTable, map)


def create_base_station(
    placeable: Placeable, config: BaseStationData
) -> BaseStationIISMotion:
    return BaseStationIISMotion(
        ips=config.ips,
        bandwidth=config.bandwidth,
        resource_blocks=config.resource_blocks,
        tx_frequency=config.tx_frequency,
        tx_power=config.tx_power,
        coverage_radius=config.coverage_radius,
        iismotion_placeable=placeable,
    )


def init_locations_for_base_stations(
    map: Map, bs_collection: ActorCollection, min_radius: int
) -> None:
    loc_loader = LocationsLoader()
    # locs = loc_loader.generate_random_locations(
    #     map=bs_collection.map, count=len(bs_collection.actorSet), min_radius=min_radius
    # )
    base_stations = list(bs_collection.actorSet.values())
    locs = loc_loader.load_locations_from_file(map, f"bs_{len(base_stations)}")
    for bs, loc in zip(base_stations, locs):
        bs.setLocation(loc)

    # loc_loader.store_placeables_locations_into_file(
    #     base_stations, f"bs_{len(base_stations)}"
    # )
