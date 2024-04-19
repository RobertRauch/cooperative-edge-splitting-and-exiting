import unittest

from mec.iismotion_extension.locations_loader import LocationsLoader
from mec.entities import BaseStationIISMotion
from mec.sinr.sinr_map import SINRMap
from src.IISMotion import IISMotion
from src.city.Map import Map
from src.movement.ActorCollection import ActorCollection
from src.movement.movementStrategies.MovementStrategyType import MovementStrategyType
from src.placeable.Placeable import Placeable, PlaceableFactory
from mec.tasks.task import TaskComputationEntity

from mec.entities import UE
from typing import Callable

# from asyncio import Task

from mec.tasks.task import Task
from typing import List

from collections import OrderedDict
import json
import os
import random
import structlog
import numpy as np

from simulations.sc_ee_simulation.task.task_generator import PeriodTaskGenerator
from simulations.sc_ee_simulation.common.mec import MECSimulation
# from simulations.sc_ee_simulation.baseline.edge_ai.edge_ai_simulation import EdgeAISimulation
from simulations.sc_ee_simulation.baseline.edge_ai.task.task_size_generator import EdgeAITaskSizeGenerator
from simulations.sc_ee_simulation.task.task_executor import TaskExecutor
from simulations.sc_ee_simulation.task.image_id_generator import RandomIdGenerator
from simulations.sc_ee_simulation.common.strategy import Strategy, Requirements
from simulations.sc_ee_simulation.common.entities import SimulationVehicle
# from simulations.sc_ee_simulation.baseline.edge_ai.predicator import Predicator
# from simulations.sc_ee_simulation.baseline.edge_ai.edge_ai import EdgeAI
# from simulations.sc_ee_simulation.baseline.edge_ml.ddpg_edge_ml import DDPG, Hyperparameters
from simulations.sc_ee_simulation.common.task_events import DDPGTaskEventsListener

from simulations.sc_ee_simulation.tests.test_config import test_config

logger = structlog.get_logger()
from mec.processing.task_processing import (
    process_bs_tasks_computation_round_robin_per_task_owner,
    process_radio_downlink_round_robin,
    process_radio_uplink_round_robin,
    process_ue_tasks_computation_round_robin,
    process_tasks_computation_fifo,
)

class Generic:
    pass


def create_test_MBps_datarate_calculator() -> Callable[[UE], float]:
    def calculator(ue: UE) -> float:
        datarate = 100
        return datarate

    return calculator


class TestCommunication(unittest.TestCase):
    base_station: BaseStationIISMotion
    vehicle: SimulationVehicle
    datarate_calculator: Callable[[UE], float]
    logger: any
      
    def setUp(self):
        self.logger = structlog.get_logger()
        movable = Generic()
        movable.id = 1
        bs = Generic()
        bs.id = 2
        self.datarate_calculator = create_test_MBps_datarate_calculator()

        strategy = Strategy()
        strategy.update_strategy(3, 0, {"EE 1": 0.9999, "EE 2": 0.999, "EE 3": 0.999}, False, 2)
        requirement = Requirements(0.1)

        task_listener = DDPGTaskEventsListener()
        self.vehicle = SimulationVehicle(
            iismotion_movable=movable,
            task_generator=None,
            event_listener=task_listener,
            ips=10,
            strategy=strategy,
            requirements=requirement,
            ma_agent=None,
            ddpg_event_listener=DDPGTaskEventsListener(),
            prev_state=[0,0],
            selected_action=[0,0]
        )
        self.base_station = BaseStationIISMotion(
            ips=100,
            bandwidth=100,
            resource_blocks=10,
            tx_frequency=10,
            tx_power=10,
            coverage_radius=20,
            iismotion_placeable=bs,
        )
  
    def test_task_uplink_one(self):        
        t1 = Task(
            ttl=0.5,
            request_size=1,
            response_size=1,
            instruction_count=5,
            instruction_count_ue=2,
            task_owner_id=1,
            target_id=2,
            created_timestep=0,
            owner=self.vehicle
        )
        connected_ues = [self.vehicle]
        self.vehicle.tasks_sending.append(t1.request)
        self.assertEqual(len(self.vehicle.tasks_sending), 1)
        self.assertEqual(len(self.base_station.tasks_computing), 0)
        process_radio_uplink_round_robin(self.base_station, connected_ues, 1, self.datarate_calculator, self.logger)
        self.assertEqual(t1.lived_timestep, 0.01)
        self.assertEqual(len(self.vehicle.tasks_sending), 0)
        self.assertEqual(len(self.base_station.tasks_computing), 1)
    
  
    def test_task_uplink_many(self):
        connected_ues = [self.vehicle]
        for t in range(99):
            self.vehicle.tasks_sending.append(
                Task(
                    ttl=0.5,
                    request_size=1,
                    response_size=1,
                    instruction_count=5,
                    instruction_count_ue=2,
                    task_owner_id=1,
                    target_id=2,
                    created_timestep=t/99,
                    owner=self.vehicle
                ).request
            )
        self.assertEqual(len(self.vehicle.tasks_sending), 99)
        self.assertEqual(len(self.base_station.tasks_computing), 0)
        process_radio_uplink_round_robin(self.base_station, connected_ues, 1, self.datarate_calculator, self.logger)
        for t in self.base_station.tasks_computing:
            self.assertEqual(round(t.task.lived_timestep*10000)/10000, 0.01)
        self.assertEqual(len(self.vehicle.tasks_sending), 0)
        self.assertEqual(len(self.base_station.tasks_computing), 99)
  
    def test_task_uplink_many_2(self):
        connected_ues = [self.vehicle]
        for t in range(99):
            self.vehicle.tasks_sending.append(
                Task(
                    ttl=0.5,
                    request_size=1,
                    response_size=1,
                    instruction_count=5,
                    instruction_count_ue=2,
                    task_owner_id=1,
                    target_id=2,
                    created_timestep=t/99,
                    owner=self.vehicle
                ).request
            )
        for t in range(21):
            self.vehicle.tasks_sending.append(
                Task(
                    ttl=0.5,
                    request_size=1,
                    response_size=1,
                    instruction_count=5,
                    instruction_count_ue=2,
                    task_owner_id=1,
                    target_id=2,
                    created_timestep=0.99,
                    owner=self.vehicle
                ).request
            )
        for t in range(20):
            self.vehicle.tasks_sending.append(
                Task(
                    ttl=0.5,
                    request_size=1,
                    response_size=1,
                    instruction_count=5,
                    instruction_count_ue=2,
                    task_owner_id=1,
                    target_id=2,
                    created_timestep=0.4,
                    owner=self.vehicle
                ).request
            )
        self.assertEqual(len(self.vehicle.tasks_sending), 140)
        self.assertEqual(len(self.base_station.tasks_computing), 0)
        process_radio_uplink_round_robin(self.base_station, connected_ues, 1, self.datarate_calculator, self.logger)
        for t in self.base_station.tasks_computing:
            self.assertEqual(round(t.task.lived_timestep*10000)/10000, 0.01)
        self.assertEqual(len(self.vehicle.tasks_sending), 21)
        self.assertEqual(len(self.base_station.tasks_computing), 99)

    def test_task_uplink_large_one(self):
        t1 = Task(
            ttl=0.5,
            request_size=200,
            response_size=1,
            instruction_count=5,
            instruction_count_ue=2,
            task_owner_id=1,
            target_id=2,
            created_timestep=0,
            owner=self.vehicle
        )
        connected_ues = [self.vehicle]
        self.vehicle.tasks_sending.append(t1.request)
        self.assertEqual(len(self.vehicle.tasks_sending), 1)
        self.assertEqual(len(self.base_station.tasks_computing), 0)
        process_radio_uplink_round_robin(self.base_station, connected_ues, 1, self.datarate_calculator, self.logger)
        self.assertEqual(t1.lived_timestep, 0.5)
        self.assertEqual(len(self.vehicle.tasks_sending), 0)
        self.assertEqual(len(self.base_station.tasks_computing), 0)

    def test_task_uplink_large_many(self):        
        t1 = Task(
            ttl=0.5,
            request_size=200,
            response_size=1,
            instruction_count=5,
            instruction_count_ue=2,
            task_owner_id=1,
            target_id=2,
            created_timestep=0,
            owner=self.vehicle
        )
        t2 = Task(
            ttl=0.5,
            request_size=200,
            response_size=1,
            instruction_count=5,
            instruction_count_ue=2,
            task_owner_id=1,
            target_id=2,
            created_timestep=0,
            owner=self.vehicle
        )
        t3 = Task(
            ttl=0.5,
            request_size=200,
            response_size=1,
            instruction_count=5,
            instruction_count_ue=2,
            task_owner_id=1,
            target_id=2,
            created_timestep=0.9,
            owner=self.vehicle
        )
        t4 = Task(
            ttl=0.25,
            request_size=200,
            response_size=1,
            instruction_count=5,
            instruction_count_ue=2,
            task_owner_id=1,
            target_id=2,
            created_timestep=0.5,
            owner=self.vehicle
        )
        connected_ues = [self.vehicle]
        self.vehicle.tasks_sending.append(t1.request)
        self.vehicle.tasks_sending.append(t2.request)
        self.vehicle.tasks_sending.append(t3.request)
        self.vehicle.tasks_sending.append(t4.request)
        self.assertEqual(len(self.vehicle.tasks_sending), 4)
        self.assertEqual(len(self.base_station.tasks_computing), 0)
        process_radio_uplink_round_robin(self.base_station, connected_ues, 1, self.datarate_calculator, self.logger)
        self.assertEqual(round(t1.lived_timestep*10000)/10000, 0.5)
        self.assertEqual(round(t2.lived_timestep*10000)/10000, 0.5)
        self.assertEqual(round(t3.lived_timestep*10000)/10000, 0.1)
        self.assertEqual(round(t4.lived_timestep*10000)/10000, 0.25)
        self.assertEqual(len(self.vehicle.tasks_sending), 1)
        self.assertEqual(len(self.base_station.tasks_computing), 0)

    def test_task_uplink_large_many_2(self):        
        t1 = Task(
            ttl=0.5,
            request_size=70,
            response_size=1,
            instruction_count=5,
            instruction_count_ue=2,
            task_owner_id=1,
            target_id=2,
            created_timestep=0,
            owner=self.vehicle
        )
        t2 = Task(
            ttl=0.2,
            request_size=20,
            response_size=1,
            instruction_count=5,
            instruction_count_ue=2,
            task_owner_id=1,
            target_id=2,
            created_timestep=0.1,
            owner=self.vehicle
        )
        connected_ues = [self.vehicle]
        self.vehicle.tasks_sending.append(t1.request)
        self.vehicle.tasks_sending.append(t2.request)
        self.assertEqual(len(self.vehicle.tasks_sending), 2)
        self.assertEqual(len(self.base_station.tasks_computing), 0)
        process_radio_uplink_round_robin(self.base_station, connected_ues, 1, self.datarate_calculator, self.logger)
        self.assertEqual(round(t1.lived_timestep*10000)/10000, 0.5)
        self.assertEqual(round(t2.lived_timestep*10000)/10000, 0.2)        
        self.assertEqual(t1.request.remaining_computation, 20)     
        self.assertEqual(t2.request.remaining_computation, 20)
        self.assertEqual(len(self.vehicle.tasks_sending), 0)
        self.assertEqual(len(self.base_station.tasks_computing), 0)

    def test_task_uplink_large_many_3(self):        
        t1 = Task(
            ttl=0.5,
            request_size=70,
            response_size=1,
            instruction_count=5,
            instruction_count_ue=2,
            task_owner_id=1,
            target_id=2,
            created_timestep=0,
            owner=self.vehicle
        )
        t2 = Task(
            ttl=0.4,
            request_size=20,
            response_size=1,
            instruction_count=5,
            instruction_count_ue=2,
            task_owner_id=1,
            target_id=2,
            created_timestep=0.4,
            owner=self.vehicle
        )
        connected_ues = [self.vehicle]
        self.vehicle.tasks_sending.append(t1.request)
        self.vehicle.tasks_sending.append(t2.request)
        self.assertEqual(len(self.vehicle.tasks_sending), 2)
        self.assertEqual(len(self.base_station.tasks_computing), 0)
        process_radio_uplink_round_robin(self.base_station, connected_ues, 1, self.datarate_calculator, self.logger)
        self.assertEqual(round(t1.lived_timestep*10000)/10000, 0.5)
        self.assertEqual(round(t2.lived_timestep*10000)/10000, 0.3)
        self.assertEqual(len(self.vehicle.tasks_sending), 0)
        self.assertEqual(len(self.base_station.tasks_computing), 1)

    def test_task_uplink_large_many_4(self):        
        t1 = Task(
            ttl=0.5,
            request_size=70,
            response_size=1,
            instruction_count=5,
            instruction_count_ue=2,
            task_owner_id=1,
            target_id=2,
            created_timestep=0.3,
            owner=self.vehicle
        )
        t2 = Task(
            ttl=0.5,
            request_size=30,
            response_size=1,
            instruction_count=5,
            instruction_count_ue=2,
            task_owner_id=1,
            target_id=2,
            created_timestep=0.6,
            owner=self.vehicle
        )
        connected_ues = [self.vehicle]
        self.vehicle.tasks_sending.append(t1.request)
        self.vehicle.tasks_sending.append(t2.request)
        self.assertEqual(len(self.vehicle.tasks_sending), 2)
        self.assertEqual(len(self.base_station.tasks_computing), 0)
        process_radio_uplink_round_robin(self.base_station, connected_ues, 1, self.datarate_calculator, self.logger)
        self.assertEqual(round(t1.lived_timestep*10000)/10000, 0.5)
        self.assertEqual(round(t2.lived_timestep*10000)/10000, 0.4)
        self.assertEqual(len(self.vehicle.tasks_sending), 1)
        self.assertEqual(len(self.base_station.tasks_computing), 0)
  
if __name__ == '__main__':
    unittest.main()