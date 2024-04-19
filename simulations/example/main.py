from asyncio import Task
from typing import List
import os

from mec.mec_network import MECNetwork

from functools import cached_property
from mec.iismotion_extension.locations_loader import LocationsLoader
from mec.mec import MEC, SimulationContext

from mec.simulation import IISMotionSimulation
from src.IISMotion import IISMotion
from src.city.Map import Map
from src.movement.ActorCollection import ActorCollection
from src.movement.movementStrategies.MovementStrategyType import MovementStrategyType
from mec.entities import BaseStationIISMotion, MecEntity, VehicleIIsmotion
from src.placeable.Placeable import Placeable, PlaceableFactory
import attr
import osmnx as ox
import matplotlib.pyplot as plt
import copy
import os


def plott_map_and_nodes(G_map, nodes: list[MecEntity], filename:str):
    node_id = max(G_map.nodes) + 1
    G = copy.deepcopy(G_map)
    for node in nodes:
        location = node.location
        G.add_node(node_id, y=location.latitude, x=location.longitude)
        node_id += 1

    fig, ax = ox.plot_graph(G, save=True, filepath=filename, show=False)
    plt.close(fig)


@attr.s(kw_only=True)
class ExampleSimulation(IISMotionSimulation):
    mec: MEC = attr.ib()

    def simulation_step(self, context: SimulationContext) -> None:
        if self.gui_enabled:
            plott_map_and_nodes(
                self.iismotion.map.driveGraph,
                list(self.mec.ue_dict.values()) + list(self.mec.bs_dict.values()),
                "mecmap.png",
            )

        # update mec and network params
        self.mec.update_network(context)

        self.mec.process_uplink(context)
        self.mec.process_computation(context)
        self.mec.process_downlink(context)

        # move with vehicles
        self.iismotion.stepAllCollections(context.sim_info.new_day)


@attr.s(kw_only=True)
class MECSimulation(MEC):
    backhaul_datarate_capacity_Mb: float = attr.ib()

    @cached_property
    def network(self) -> MECNetwork:
        return MECNetwork.as_fullmesh_backhaul(
            list(self.bs_dict.values()),
            list(self.ue_dict.values()),
            self.backhaul_datarate_capacity_Mb,
        )

    def update_network(self, context: SimulationContext) -> None:
        """
        good place to select connecting bs, select computing bs
        """
        print("mec.update_network", context.sim_info.simulation_step)

    def process_uplink(self, context: SimulationContext) -> None:
        print("mec.process_uplink", context.sim_info.simulation_step)

    def process_downlink(self, context: SimulationContext) -> None:
        print("mec.process_downlink", context.sim_info.simulation_step)

    def process_computation(self, context: SimulationContext) -> None:
        print("mec.process_computation", context.sim_info.simulation_step)



class BaseStationPlaceableFactory(PlaceableFactory):
    def createPlaceable(self, locationsTable, map: Map) -> Placeable:
        return Placeable(locationsTable, map)


def init_locations_for_base_stations(
    bs_collection: ActorCollection, min_radius: int
) -> None:
    loc_loader = LocationsLoader()
    locs = loc_loader.generate_random_locations(
        map=bs_collection.map, count=len(bs_collection.actorSet), min_radius=min_radius
    )
    for bs, loc in zip(bs_collection.actorSet.values(), locs):
        bs.setLocation(loc)


def create_base_station(placeable: Placeable, config) -> BaseStationIISMotion:
    return BaseStationIISMotion(
        ips=config.ips,
        bandwidth=config.bandwidth,
        resource_blocks=config.resource_blocks,
        tx_frequency=config.tx_frequency,
        tx_power=config.tx_power,
        coverage_radius=config.coverage_radius,
        iismotion_placeable=placeable,
    )


class SimulationVehicle(VehicleIIsmotion):
    def tasks_to_compute(self) -> List[Task]:
        return []


class Config:
    pass

def main():
    sim_steps = 200
    movement_dt = 0.5
    stepping = False # step simulation by pressing enter
    gui_enabled = True
    gui_timeout = 1

    bs_count = 1
    veh_count = 50
    veh_speed = 13
    bs_min_radius = 20

    task_config = Config()
    task_config.request_size = 0.5
    task_config.response_size = 0.5
    task_config.instruction_count = 10e6

    bs_config = Config()
    bs_config.ips = 3.6e9
    bs_config.bandwidth = 100e6
    bs_config.resource_blocks = 4000
    bs_config.tx_frequency = 2e9
    bs_config.tx_power = 0.1
    bs_config.coverage_radius = 200

    backhaul_config = Config()
    backhaul_config.datarate_capacity_Mb = 10

    print(os.getcwd())
    map = Map.fromFile(
        "../../data/maps/map_3x3_manhattan.pkl", radius=110, contains_deadends=False
    )
    iismotion = IISMotion(map, guiEnabled=gui_enabled, gridRows=22, secondsPerTick=movement_dt)

    vehicles_collection = iismotion.createActorCollection(
        "vehicles",
        ableOfMovement=True,
        movementStrategy=MovementStrategyType.RANDOM_INTERSECTION_WAYPOINT_CITY_CUDA,
    )

    vehicles_movables = vehicles_collection.generateMovables(veh_count)

    vehicles = {}
    for m in vehicles_movables:
        m.setSpeed(veh_speed * movement_dt) # speed is m/sim_dt, not m/s
        vehicles[m.id] = SimulationVehicle(
            iismotion_movable=m,
        )

    base_stations_collection = iismotion.createActorCollection(
        "base_stations",
        ableOfMovement=False,
        movementStrategy=MovementStrategyType.DRONE_MOVEMENT_CUDA,
    ).addPlaceableFactory(BaseStationPlaceableFactory(), bs_count)

    init_locations_for_base_stations(base_stations_collection, bs_min_radius)

    base_stations = {}
    for placeable in base_stations_collection.actorSet.values():
        base_stations[placeable.id] = create_base_station(
            placeable, bs_config
        )


    mec = MECSimulation(
        ue_dict=vehicles,
        bs_dict=base_stations,
        backhaul_datarate_capacity_Mb=backhaul_config.datarate_capacity_Mb,
    )

    simulation = ExampleSimulation(
        iismotion=iismotion,
        number_of_ticks=sim_steps,
        simulation_dt=movement_dt,
        mec=mec,
        gui_enabled=gui_enabled,
        gui_timeout=gui_timeout,
        stepping=stepping,
    )

    simulation.run_simulation()

# run simulation via command: python -m simulations.example.main
if __name__ == "__main__":
    main()
