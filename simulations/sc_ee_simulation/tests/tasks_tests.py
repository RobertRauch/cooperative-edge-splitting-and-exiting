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


class TestTaskCommunicationAndComputation(unittest.TestCase):
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
            ips=100,
            strategy=strategy,
            requirements=requirement,
            ma_agent=None,
            ddpg_event_listener=DDPGTaskEventsListener(),
            selected_action=None,
            prev_state=None
        )
        self.base_station = BaseStationIISMotion(
            ips=25000,
            bandwidth=100,
            resource_blocks=10,
            tx_frequency=10,
            tx_power=10,
            coverage_radius=20,
            iismotion_placeable=bs,
        )

    def test_task_computation_many_wait_drop(self):
        num_of_tasks = 0
        for i in np.arange(0, 1, 0.0002):
            num_of_tasks += 1
            t = Task(
                ttl=0.5,
                request_size=1,
                response_size=1,
                instruction_count=10,
                instruction_count_ue=1,
                task_owner_id=1,
                target_id=2,
                created_timestep=0,
                owner=self.vehicle
            )
            self.base_station.tasks_computing.append(t.computing)

        self.assertEqual(len(self.vehicle.tasks_computing), 0)
        self.assertEqual(len(self.vehicle.tasks_sending), 0)
        self.assertEqual(len(self.base_station.tasks_sending), 0)
        self.assertEqual(len(self.base_station.tasks_computing), num_of_tasks)
        process_tasks_computation_fifo(self.base_station, 1, logger, False)
        add = 0
        for i in self.base_station.tasks_sending:
            self.assertEqual(round(i.task.lived_timestep, 7), round(0.0004 + add, 7))
            if add < 0.5:
                add += 0.0004

        self.assertEqual(len(self.vehicle.tasks_computing), 0)
        self.assertEqual(len(self.vehicle.tasks_sending), 0)
        self.assertEqual(len(self.base_station.tasks_sending), 1249)
        self.assertEqual(len(self.base_station.tasks_computing), 0)

    def test_task_computation_many_wait_drop_2(self):
        t = Task(
            ttl=0.0002,
            request_size=1,
            response_size=1,
            instruction_count=10,
            instruction_count_ue=1,
            task_owner_id=1,
            target_id=2,
            created_timestep=0.999999999,
            owner=self.vehicle
        )
        # t.used_timestep_interval = 0
        # t.used_timestep = 1
        self.base_station.tasks_computing.append(t.computing)

        self.assertEqual(len(self.vehicle.tasks_computing), 0)
        self.assertEqual(len(self.vehicle.tasks_sending), 0)
        self.assertEqual(len(self.base_station.tasks_sending), 0)
        self.assertEqual(len(self.base_station.tasks_computing), 1)
        process_tasks_computation_fifo(self.base_station, 1, logger, False)

        self.assertEqual(round(t.lived_timestep, 7), 0)
        self.reset_used_timestamp()

        self.assertEqual(len(self.vehicle.tasks_computing), 0)
        self.assertEqual(len(self.vehicle.tasks_sending), 0)
        self.assertEqual(len(self.base_station.tasks_sending), 0)
        self.assertEqual(len(self.base_station.tasks_computing), 1)

        process_tasks_computation_fifo(self.base_station, 1, logger, False)

        self.assertEqual(round(t.lived_timestep, 7), 0.0002)
        self.reset_used_timestamp()

        self.assertEqual(len(self.vehicle.tasks_computing), 0)
        self.assertEqual(len(self.vehicle.tasks_sending), 0)
        self.assertEqual(len(self.base_station.tasks_sending), 0)
        self.assertEqual(len(self.base_station.tasks_computing), 0)

    def reset_used_timestamp(self):
        tasks: List[TaskComputationEntity] = []
        sim_dt = 1
        tasks += self.vehicle.tasks_computing + self.vehicle.tasks_sending + self.vehicle.tasks_to_compute
        self.vehicle.used_timestep_interval = 0
        tasks += self.base_station.tasks_computing + self.base_station.tasks_sending
        self.base_station.used_timestep_interval = 0
        for task in tasks:
            if task.task.used_timestep_interval > sim_dt:
                raise Exception(
                    "used more timestamp for task then is the simulation step"
                )
            task.task.used_timestep_interval = 0
            task.task.used_timestep += sim_dt
            if round(task.task.lived_timestep, 10) >= task.task.ttl:
                task.task.used_timestep = task.task.ttl + task.task.created_timestep
                self._pop_computation_entity(tasks, task)

    def _pop_computation_entity(
        self, l: List[TaskComputationEntity], ce: TaskComputationEntity
    ) -> None:
        l[:] = [c for c in l if c.task.id != ce.task.id]

        
if __name__ == '__main__':
    unittest.main()