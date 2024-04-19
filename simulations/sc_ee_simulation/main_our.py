import os
import time
from mec.mec_types import EntityID
from typing import Dict, cast
import random
import numpy as np
import json
import time
import multiprocessing
import copy
from pathlib import Path
from collections import OrderedDict

# from simulations.sc_ee_simulation.baseline.edge_ml.edge_ml_simulation import EdgeMLSimulation
# from simulations.sc_ee_simulation.baseline.edge_ml.task.task_size_generator import EdgeMLTaskSizeGenerator
from simulations.sc_ee_simulation.common.stats_collector import DDPGStatsCollector
from simulations.sc_ee_simulation.common.task_events import DDPGTaskEventsListener
from src.city.Map import Map
from src.IISMotion import IISMotion
from mec.entities import BaseStationIISMotion
from src.movement.movementStrategies.MovementStrategyType import MovementStrategyType
from mec.sinr.sinr_map import SINRMap
from src.placeable.Placeable import Placeable, PlaceableFactory
from mec.iismotion_extension.locations_loader import LocationsLoader
from src.movement.ActorCollection import ActorCollection

# from simulations.sc_ee_simulation.baseline.edge_ai.task.task_size_generator import EdgeAITaskSizeGenerator
from simulations.sc_ee_simulation.common.strategy import Strategy, Requirements
from simulations.sc_ee_simulation.task.image_id_generator import RandomIdGenerator
from simulations.sc_ee_simulation.task.task_executor import TaskExecutor
from simulations.sc_ee_simulation.config import SimulationConfig, VehicleData, BaseStationData, SimulationType, Weights
from simulations.sc_ee_simulation.common.entities import SimulationVehicle
from simulations.sc_ee_simulation.task.task_generator import PeriodTaskGenerator
from simulations.sc_ee_simulation.common.mec import MECSimulation
from simulations.sc_ee_simulation.simulation.maddpg import MADDPG

import matplotlib.pyplot as plt
from simulations.sc_ee_simulation.task.task_events import TaskEventsProtocol

from simulations.sc_ee_simulation.task.task_size_generator import OursTaskSizeGenerator
from simulations.sc_ee_simulation.simulation.main_simulation import MainSimulation

with open(os.getcwd() + "/data/computer_vision/layer_data.json") as f:
    layer_data = json.load(f, object_pairs_hook=OrderedDict)
with open(os.getcwd() + "/data/computer_vision/output_data_ae.json") as f:
    output_data = json.load(f, object_pairs_hook=OrderedDict)
with open(os.getcwd() + "/data/computer_vision/ae_data.json") as f:
    ae_data = json.load(f, object_pairs_hook=OrderedDict)

def _add_vehicles(
    iismotion: IISMotion,
    vehicle_configs: VehicleData,
    movement_dt: float,
    event_listener: TaskEventsProtocol,
    sim_name: str,
    maddpg: MADDPG,
    sim_idx: int = -1,
    ep_idx: int = -1,
) -> Dict[EntityID, SimulationVehicle]:
    task_generators = []
    vehicles = {}
    vehicles_collection = iismotion.createActorCollection(
        "vehicles",
        ableOfMovement=True,
        movementStrategy=MovementStrategyType.RANDOM_INTERSECTION_WAYPOINT_CITY_CUDA,
    )
    idx = 0

    for vehicle_config in vehicle_configs:
        task_config = vehicle_config.task
        vehicles_movables = vehicles_collection.generateMovables(vehicle_config.count)
        for m in vehicles_movables:
            img_generator = RandomIdGenerator()
            strategy = Strategy()
            strategy.update_strategy(3, 0, {"EE 1": 0.9999, "EE 2": 0.999, "EE 3": 0.999}, False, -1)
            requirement = Requirements(vehicle_config.requirements.latency, vehicle_config.requirements.accuracy)

            task_executor = TaskExecutor(
                layer_data,
                output_data,
                ae_data,
                img_generator,
                strategy
            )
            task_size_generator = OursTaskSizeGenerator(
                task_executor,
                12288
            )

            t_generator = PeriodTaskGenerator(
                period=task_config.generation_period,
                task_ttl=task_config.ttl,
                size_generator=task_size_generator,
                start_delay=random.random() * task_config.generation_delay_range,
            )
            t_generator.task_owner_id = m.id
            m.setSpeed((vehicle_config.speed_ms + (vehicle_config.speed_variation * random.random())) * movement_dt)
            task_generators.append(t_generator)
            vehicles[m.id] = SimulationVehicle(
                iismotion_movable=m,
                task_generator=t_generator,
                event_listener=event_listener,
                requirements=requirement,
                weights=vehicle_config.weights,
                strategy=strategy,
                ddpg_event_listener=DDPGTaskEventsListener(),
                ma_agent=maddpg.agents[idx],
                ips=vehicle_config.ips,
                prev_state=[0, 0, 0, 0, 0],
                selected_action=[0] * 34,
                log_idx=idx,
                model_save_location=(os.getcwd()+vehicle_config.model_location).format(sim_name, str(sim_idx))
            )

            cast(PeriodTaskGenerator, vehicles[m.id].task_generator).set_owner(vehicles[m.id])
            idx += 1

    return vehicles


class BaseStationPlaceableFactory(PlaceableFactory):
    def createPlaceable(self, locationsTable, map: Map) -> Placeable:
        return Placeable(locationsTable, map)


def create_base_station(
    placeable: Placeable, config: BaseStationData
) -> BaseStationIISMotion:
    return BaseStationIISMotion(
        ips=int(config.ips),
        bandwidth=int(config.bandwidth),
        resource_blocks=config.resource_blocks,
        tx_frequency=int(config.tx_frequency),
        tx_power=config.tx_power,
        coverage_radius=config.coverage_radius,
        iismotion_placeable=placeable,
    )


def init_locations_for_base_stations(
    map: Map, bs_collection: ActorCollection, min_radius: int
) -> None:
    loc_loader = LocationsLoader()
    base_stations = list(bs_collection.actorSet.values())
    locs = loc_loader.load_locations_from_file(map, f"bs_{len(base_stations)}")
    for bs, loc in zip(base_stations, locs):
        bs.setLocation(loc)


def _add_base_stations(
    iismotion: IISMotion, bs_config: BaseStationData
) -> Dict[EntityID, BaseStationIISMotion]:
    base_stations_collection = iismotion.createActorCollection(
        "base_stations",
        ableOfMovement=False,
        movementStrategy=MovementStrategyType.DRONE_MOVEMENT_CUDA,
    ).addPlaceableFactory(BaseStationPlaceableFactory(), bs_config.count)

    init_locations_for_base_stations(
        iismotion.map, base_stations_collection, bs_config.min_radius
    )

    base_stations = {}
    for placeable in base_stations_collection.actorSet.values():
        base_stations[placeable.id] = create_base_station(placeable, bs_config)

    return base_stations

def main_grid_search(config: SimulationConfig, config_data):
    processes = []
    for lat in config.requirements_search.latency_requirements:
        for acc in config.requirements_search.accuracy_requirements:
            if len(processes) < config.requirements_search.workers:
                process = multiprocessing.Process(target=update_requirements_and_start_main, args=(lat, acc, config, config_data, ))
                process.start()
                processes.append(process)


            while(len(processes) >= config.requirements_search.workers):
                tmp_processes = []
                for process in processes:
                    if process.is_alive():
                        tmp_processes.append(process)
                processes = tmp_processes


def update_requirements_and_start_main(lat, acc, config: SimulationConfig, config_data):
    cur_config = copy.deepcopy(config)
    cur_config_data = copy.deepcopy(config_data)
    cur_config.result_dir= cur_config.result_dir.replace("{lat}", str(lat))
    cur_config.result_dir = cur_config.result_dir.replace("{acc}", str(acc))
    for vehicle in cur_config.vehicles:
        vehicle.requirements.latency = lat
        vehicle.requirements.accuracy = acc
        vehicle.model_location = vehicle.model_location.replace("{lat}", str(lat))
        vehicle.model_location = vehicle.model_location.replace("{acc}", str(acc))

    cur_config_data['result_dir'] = cur_config_data['result_dir'].replace("{lat}", str(lat))
    cur_config_data['result_dir'] = cur_config_data['result_dir'].replace("{acc}", str(acc))
    for vehicle in cur_config_data['vehicles']:
        vehicle['requirements']['latency'] = lat
        vehicle['requirements']['accuracy'] = acc
        vehicle['model_location'] = vehicle['model_location'].replace("{lat}", str(lat))
        vehicle['model_location'] = vehicle['model_location'].replace("{acc}", str(acc))

    main_repeat(cur_config, cur_config_data)

def main_repeat(config: SimulationConfig, config_data):
    for i in range(config.repeat_start_idx, config.repeat):
        for ep in range(config.episodes):
            main(config, config_data, i, ep)


def main(config, config_data = None, idx = None, ep_idx=None):
    task_event_listener = DDPGTaskEventsListener()
    map = Map.fromFile(
        "data/maps/map_3x3_manhattan.pkl", radius=110, contains_deadends=False
    )
    n_vehicles = 0
    actor_dims = []
    for vehicle in config.vehicles:
        n_vehicles += vehicle.count
    for _ in range(n_vehicles):
        actor_dims.append(5)
    critic_dims = sum(actor_dims)
    maddpg = MADDPG(actor_dims, critic_dims, n_vehicles, 34)
    if (not ep_idx == 0 
                and config.vehicles[0].model_location is not None 
                and not config.vehicles[0].model_location == "" 
                and os.path.exists(config.vehicles[0].model_location + "checkpoint.pt".format(idx))):
                maddpg.load(config.vehicles[0].model_location + "checkpoint.pt".format(idx))

    iismotion = IISMotion(
        map,
        guiEnabled=config.simulation.gui_enabled,
        gridRows=22,
        secondsPerTick=config.simulation.dt,
    )

    vehicles = _add_vehicles(
        iismotion, config.vehicles, config.simulation.dt, 
        task_event_listener, SimulationType(config.simulation.simulation_type).name, maddpg,
        sim_idx=idx, ep_idx=ep_idx
    )

    base_stations = _add_base_stations(iismotion, config.base_stations)

    map_grid = iismotion.map.mapGrid

    sinr_map = SINRMap(
        x_size=40,
        y_size=40,
        latmin=map_grid.latmin,
        latmax=map_grid.latmax,
        lonmin=map_grid.lonmin,
        lonmax=map_grid.lonmax,
    )

    mec = MECSimulation(
        ue_dict=vehicles,
        bs_dict=base_stations,
        sinr_map=sinr_map,
        backhaul_datarate_capacity_Mb=10,  # not used
        task_event_listener=task_event_listener
    )

    ddpg_stats = DDPGStatsCollector()
    simulation = MainSimulation(
        stats_collector=ddpg_stats,
        iismotion=iismotion,
        number_of_ticks=config.simulation.steps,
        simulation_dt=config.simulation.dt,
        mec=mec,
        gui_enabled=config.simulation.gui_enabled,
        gui_timeout=config.simulation.gui_timeout,
        training_steps=config.simulation.training_steps,
        episode_index=ep_idx,
        simulation_index=idx,
        stepping=False,
        maddpg=maddpg,
        name=SimulationType(config.simulation.simulation_type).name,
        # plot_map=config.simulation.plot_map,
    )

    start = time.time()
    simulation.run_simulation()
    end = time.time()

    for key, stat_data in ddpg_stats.statistics.items():
        timestamps = [i.timestamp for i in stat_data]
        latency = [i.latency for i in stat_data]
        exits = [i.exit for i in stat_data]
        splits = [i.split for i in stat_data]
        autoencoders = [i.autoencoder for i in stat_data]
        a_loss = [i.actor_loss for i in stat_data]
        c_loss = [i.critic_loss for i in stat_data]

        accuracy = [i.accuracy for i in stat_data]
        dropped = [i.dropped for i in stat_data]
        accepted = [i.accepted for i in stat_data]
        bandwidth = [i.bandwidth for i in stat_data]
        bandwidth_2 = [i.bandwidth_2 for i in stat_data]
        server_utilization = [i.server_utilization for i in stat_data]
        device_utilization = [i.device_utilization for i in stat_data]
        orig_rewards = [i.reward for i in stat_data]
        simulation_data = {
            "start_time": start,
            "end_time": end,
            "execution_time": end - start
        }

        result_dir = config.result_dir if idx is None or "{}" not in config.result_dir else config.result_dir.format(SimulationType(config.simulation.simulation_type).name, str(idx), str(ep_idx))
        result_name = config.result_name if key is None or "{}" not in config.result_name else config.result_name.format(str(key))
        result_dir_path = os.getcwd() + result_dir + result_name
        Path(result_dir_path).mkdir(parents=True, exist_ok=True)
        # if not os.path.exists(result_dir_path + "plots"):
        #     os.makedirs(result_dir_path + "plots")
        if not os.path.exists(result_dir_path + "data"):
            os.makedirs(result_dir_path + "data")

        # plt.plot(timestamps, orig_rewards)
        # plt.savefig(result_dir_path + "plots/rewards.png")
        # plt.clf()
        # plt.plot(timestamps, exits)
        # plt.savefig(result_dir_path + "plots/exits.png")
        # plt.clf()
        # plt.plot(timestamps, splits)
        # plt.savefig(result_dir_path + "plots/splits.png")
        # plt.clf()
        # plt.plot(timestamps, latency)
        # plt.savefig(result_dir_path + "plots/latency.png")

        with open(result_dir_path + "data/timestamps.txt", 'w') as filehandle:
            np.savetxt(filehandle, timestamps)
        with open(result_dir_path + "data/exits.txt", 'w') as filehandle:
            np.savetxt(filehandle, exits)
        with open(result_dir_path + "data/splits.txt", 'w') as filehandle:
            np.savetxt(filehandle, splits)
        with open(result_dir_path + "data/autoencoders.txt", 'w') as filehandle:
            np.savetxt(filehandle, autoencoders)
        with open(result_dir_path + "data/latency.txt", 'w') as filehandle:
            np.savetxt(filehandle, latency)
        with open(result_dir_path + "data/rewards.txt", 'w') as filehandle:
            np.savetxt(filehandle, orig_rewards)
        with open(result_dir_path + "data/c_loss.txt", 'w') as filehandle:
            np.savetxt(filehandle, c_loss)
        with open(result_dir_path + "data/a_loss.txt", 'w') as filehandle:
            np.savetxt(filehandle, a_loss)
        with open(result_dir_path + "data/bandwidth.txt", 'w') as filehandle:
            np.savetxt(filehandle, bandwidth)
        with open(result_dir_path + "data/accuracy.txt", 'w') as filehandle:
            np.savetxt(filehandle, accuracy)
        with open(result_dir_path + "data/dropped.txt", 'w') as filehandle:
            np.savetxt(filehandle, dropped)
        with open(result_dir_path + "data/accepted.txt", 'w') as filehandle:
            np.savetxt(filehandle, accepted)
        with open(result_dir_path + "data/bandwidth_2.txt", 'w') as filehandle:
            np.savetxt(filehandle, bandwidth_2)
        with open(result_dir_path + "data/device_utilization.txt", 'w') as filehandle:
            np.savetxt(filehandle, device_utilization)
        with open(result_dir_path + "data/server_utilization.txt", 'w') as filehandle:
            np.savetxt(filehandle, server_utilization)

        with open(result_dir_path + "data/simulation_data.json", 'w') as filehandle:
            json.dump(simulation_data, filehandle, indent = 4)
    if config_data is not None:
        with open(os.getcwd() + result_dir + "config.json", 'w') as f:
            json.dump(config_data, f, indent=4)