import os
cwd = os.getcwd()
print(cwd)
import sys
sys.path.insert(0,cwd)

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
from simulations.sc_ee_simulation.task.task_size_generator import OursTaskSizeGenerator
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


class TestTaskGeneration(unittest.TestCase):
    base_station: BaseStationIISMotion
    vehicle: SimulationVehicle
    datarate_calculator: Callable[[UE], float]
    logger: any
    
    with open(os.getcwd() + "/data/computer_vision/layer_data.json") as f:
        layer_data = json.load(f, object_pairs_hook=OrderedDict)
    with open(os.getcwd() + "/data/computer_vision/output_data.json") as f:
        output_data = json.load(f, object_pairs_hook=OrderedDict)
    with open(os.getcwd() + "/data/computer_vision/output_data_ae.json") as f:
        output_data_ae = json.load(f, object_pairs_hook=OrderedDict)
    with open(os.getcwd() + "/data/computer_vision/ae_data.json") as f:
        ae_data = json.load(f, object_pairs_hook=OrderedDict)
      
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
            prev_state=None,
            weights=None,
            log_idx=0
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
  
    # def test_task_generation(self):
    #     img_generator = RandomIdGenerator()
    #     task_executor = TaskExecutor(
    #         os.getcwd() + "/data/computer_vision/layer_data.json",
    #         os.getcwd() + "/data/computer_vision/output_data_ae.json",
    #         os.getcwd() + "/data/computer_vision/ae_data.json",
    #         img_generator,
    #         self.vehicle.strategy
    #     )
    #     size_generator = EdgeAITaskSizeGenerator(
    #         task_executor
    #     )
    #     generator = PeriodTaskGenerator(
    #         period=0.25,
    #         task_ttl=0.5,
    #         size_generator=size_generator,
    #         start_delay=0.25,
    #     )
    #     self.vehicle.task_generator = generator
    #     generator.task_owner_id = self.vehicle.id
    #     generator.target_id = self.base_station.id
    #     self.vehicle.generate_tasks(1)
    #     self.assertEqual( len(self.vehicle.tasks_to_compute), 3)
    #     self.assertEqual( self.vehicle.tasks_to_compute[0].ttl, 0.5)
    #     self.assertEqual( self.vehicle.tasks_to_compute[0].task_owner_id, 1)
    #     self.assertEqual( self.vehicle.tasks_to_compute[0].target_id, 2)
    #     self.assertEqual( self.vehicle.tasks_to_compute[1].ttl, 0.5)
    #     self.assertEqual( self.vehicle.tasks_to_compute[1].task_owner_id, 1)
    #     self.assertEqual( self.vehicle.tasks_to_compute[1].target_id, 2)
    #     self.assertEqual( self.vehicle.tasks_to_compute[2].ttl, 0.5)
    #     self.assertEqual( self.vehicle.tasks_to_compute[2].task_owner_id, 1)
    #     self.assertEqual( self.vehicle.tasks_to_compute[2].target_id, 2)
  
    def test_task_generation_mips_edge_ai(self):
        img_generator = RandomIdGenerator()
        task_executor = TaskExecutor(
            self.layer_data,
            self.output_data,
            self.ae_data,
            img_generator,
            self.vehicle.strategy
        )
        size_generator = EdgeAITaskSizeGenerator(
            task_executor
        )
        generator = PeriodTaskGenerator(
            period=0.25,
            task_ttl=0.5,
            size_generator=size_generator,
            start_delay=0.25,
        )
        self.vehicle.task_generator = generator
        generator.task_owner_id = self.vehicle.id
        generator.target_id = self.base_station.id

        instructions = [0, 2031616.0, 40042496.0, 40108032.0, 40108032.0, 59113472.0, 96993280.0, 97026048.0, 97026048.0, 115965952.0, 153780224.0, 191594496.0, 191610880.0, 191610880.0, 210518016.0, 248299520.0, 286081024.0, 286089216.0, 295534592.0, 304979968.0, 314425344.0, 333350922.0]
        test_data = [
                [20, 22, 333350922.0, 4, 0], 
                [12, 14, 234668042.0, 3, 43057162.0],
                [7, 9, 156876810.0, 2, 59850762.0],
                [3, 5, 133545994.0, 1, 93437962.0]
                ]
        request_size = [0, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288]
        response_size = [0, 262144, 262144, 65536, 65536, 131072, 131072, 32768, 32768, 65536, 65536, 65536, 16384, 16384, 32768, 32768, 32768, 8192, 8192, 8192, 8192, 40]

        for ins in test_data:
            instruction = ins[2]
            for idx in range(ins[1]):
                # print("[{}] {} | {} | {} | {}".format(idx, instructions[idx], instruction-instructions[idx], request_size[idx], response_size[idx]))
                self.vehicle.tasks_to_compute.clear()
                self.vehicle.strategy.update_strategy(idx, -1, None, True, ins[3])
                self.vehicle.generate_tasks(1)
                print("{} | {}".format(ins, idx))
                for task in self.vehicle.tasks_to_compute:
                    # print(task)
                    self.assertEqual(task.instruction_count, instructions[idx] + (ins[4] if idx > ins[0] else 0))
                    self.assertEqual(task.instruction_count_ue, instruction-instructions[idx] - (ins[4] if idx > ins[0] else 0))
                    self.assertEqual(task.request_size, request_size[idx]/1048576)
                    self.assertEqual(task.response_size, response_size[idx]/1048576 if idx < (ins[1]-1) else 40/1048576)
  
    def test_task_generation_mips_edge_ml(self):
        img_generator = RandomIdGenerator()
        task_executor = TaskExecutor(
            self.layer_data,
            self.output_data,
            self.ae_data,
            img_generator,
            self.vehicle.strategy
        )
        size_generator = OursTaskSizeGenerator(
            task_executor
        )
        generator = PeriodTaskGenerator(
            period=0.25,
            task_ttl=0.5,
            size_generator=size_generator,
            start_delay=0.25,
        )
        self.vehicle.task_generator = generator
        generator.task_owner_id = self.vehicle.id
        generator.target_id = self.base_station.id

        instructions = [0, 2031616.0, 40042496.0, 40108032.0, 40108032.0, 59113472.0, 96993280.0, 97026048.0, 97026048.0, 115965952.0, 153780224.0, 191594496.0, 191610880.0, 191610880.0, 210518016.0, 248299520.0, 286081024.0, 286089216.0, 295534592.0, 304979968.0, 314425344.0, 333350922.0]
        # instruction = 333350922.0
        test_data = [
                        [20, 22, 333350922.0, 4, 0], 
                        [12, 14, 234668042.0, 3, 43057162.0],
                        [7, 9, 156876810.0, 2, 59850762.0],
                        [3, 5, 133545994.0, 1, 93437962.0]
                       ]
        request_size = [12288, 262144, 262144, 65536, 65536, 131072, 131072, 32768, 32768, 65536, 65536, 65536, 16384, 16384, 32768, 32768, 32768, 8192, 8192, 8192, 8192, 0]
        response_size = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 0]

        for ins in test_data:
            instruction = ins[2]
            for idx in range(ins[1]):
                # print("[{}] {} | {} | {} | {}".format(idx, instructions[idx], instruction-instructions[idx], request_size[idx], response_size[idx]))
                self.vehicle.tasks_to_compute.clear()
                self.vehicle.strategy.update_strategy(idx, -1, None, True, ins[3])
                self.vehicle.generate_tasks(1)
                pass
                for task in self.vehicle.tasks_to_compute:
                    self.assertEqual(task.instruction_count, instruction-instructions[idx] - (ins[4] if idx > ins[0] else 0))
                    self.assertEqual(task.instruction_count_ue, instructions[idx] + (ins[4] if idx > ins[0] else 0))
                    self.assertEqual(task.request_size, request_size[idx]/1048576 if not idx > ins[0] else 0)
                    self.assertEqual(task.response_size, response_size[idx]/1048576 if not idx > ins[0] else 0)

        instructions = [0, 2031616.0, 40042496.0, 40108032.0, 133545994.0, 152551434.0, 190431242.0, 190464010.0, 250314772.0, 269254676.0, 307068948.0, 344883220.0, 344899604.0, 387956766.0, 406863902.0, 444645406.0, 482426910.0, 482435102.0, 491880478.0, 501325854.0, 510771230.0, 529696808.0]
        test_data = [
                        [20, 22, 529696808.0, 1, 1, 1], 
                        [12, 14, 387956766.0, 1, 1, 0],
                        [7, 9, 250314772.0, 1, 0, 0],
                        [3, 5, 133545994.0, 0, 0, 0]
                       ]
        # instruction = 529696808.0
        request_size = [12288, 262144, 262144, 65536, 65536, 131072, 131072, 32768, 32768, 65536, 65536, 65536, 16384, 16384, 32768, 32768, 32768, 8192, 8192, 8192, 8192, 0]
        response_size = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 0]

        for ins in test_data:
            instruction = ins[2]
            for idx in range(ins[1]):
                # print("[{}] {} | {} | {} | {}".format(idx, instructions[idx], instruction-instructions[idx], request_size[idx], response_size[idx]))
                self.vehicle.tasks_to_compute.clear()
                self.vehicle.strategy.update_strategy(idx, -1, {"EE 1": ins[3], "EE 2": ins[4], "EE 3": ins[5]}, False, -1)
                self.vehicle.generate_tasks(1)
                pass
                for task in self.vehicle.tasks_to_compute:
                    self.assertEqual(task.instruction_count, instruction-instructions[idx])
                    self.assertEqual(task.instruction_count_ue, instructions[idx])
                    self.assertEqual(task.request_size, request_size[idx]/1048576 if not idx > ins[0] else 0)
                    self.assertEqual(task.response_size, response_size[idx]/1048576 if not idx > ins[0] else 0)

    def test_task_generation_mips_edge_ml_ae(self):
        img_generator = RandomIdGenerator()
        task_executor = TaskExecutor(
            self.layer_data,
            self.output_data_ae,
            self.ae_data,
            img_generator,
            self.vehicle.strategy
        )
        size_generator = OursTaskSizeGenerator(
            task_executor
        )
        generator = PeriodTaskGenerator(
            period=0.25,
            task_ttl=0.5,
            size_generator=size_generator,
            start_delay=0.25,
        )
        self.vehicle.task_generator = generator
        generator.task_owner_id = self.vehicle.id
        generator.target_id = self.base_station.id
        # with open("./data/computer_vision/ae_data.json") as f:
        #     ae_data = json.load(f)


        instructions = [0, 2031616.0, 40042496.0, 40108032.0, 40108032.0, 59113472.0, 96993280.0, 97026048.0, 97026048.0, 115965952.0, 153780224.0, 191594496.0, 191610880.0, 191610880.0, 210518016.0, 248299520.0, 286081024.0, 286089216.0, 295534592.0, 304979968.0, 314425344.0, 333350922.0]
        test_data = [
                        [20, 22, 333350922.0, 4, 0], 
                        [12, 14, 234668042.0, 3, 43057162.0],
                        [7, 9, 156876810.0, 2, 59850762.0],
                        [3, 5, 133545994.0, 1, 93437962.0]
                       ]
        request_size = [12288, 262144, 262144, 65536, 65536, 131072, 131072, 32768, 32768, 65536, 65536, 65536, 16384, 16384, 32768, 32768, 32768, 8192, 8192, 8192, 8192, 0]
        response_size = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 0]

        for ins in test_data:
            instruction = ins[2]
            for idx in range(ins[1]):
                for ae_idx in range(19):                
                    # print("[{} - {}] {} | {} | {} | {}".format(idx, ae_idx, instructions[idx], instruction-instructions[idx], request_size[idx], response_size[idx]))
                    self.vehicle.tasks_to_compute.clear()
                    self.vehicle.strategy.update_strategy(idx, ae_idx, None, True, ins[3])
                    self.vehicle.generate_tasks(1)
                    encoder = self.ae_data[str(idx)][str(ae_idx)]["encoder"] if not idx == 0 and not idx > ins[0] else 0
                    decoder = self.ae_data[str(idx)][str(ae_idx)]["decoder"] if not idx == 0 and not idx > ins[0] else 0
                    size = 12288 if idx == 0 else 0 if idx > ins[0] else self.ae_data[str(idx)][str(ae_idx)]["size"]
                    pass
                    for task in self.vehicle.tasks_to_compute:
                        self.assertEqual(task.instruction_count, instruction-instructions[idx] + decoder - (ins[4] if idx > ins[0] else 0))
                        self.assertEqual(task.instruction_count_ue, instructions[idx] + encoder + (ins[4] if idx > ins[0] else 0))
                        self.assertEqual(task.request_size,  size/1048576 if not idx > ins[0] else 0)
                        self.assertEqual(task.response_size, response_size[idx]/1048576 if not idx > ins[0] else 0)

        # instructions = [0, 2031616.0, 40042496.0, 40108032.0, 133545994.0, 152551434.0, 190431242.0, 190464010.0, 250314772.0, 269254676.0, 307068948.0, 344883220.0, 344899604.0, 387956766.0, 406863902.0, 444645406.0, 482426910.0, 482435102.0, 491880478.0, 501325854.0, 510771230.0, 529696808.0]
        # test_data = [
        #                 [20, 22, 529696808.0, 1, 1, 1], 
        #                 [12, 14, 387956766.0, 1, 1, 0],
        #                 [7, 9, 250314772.0, 1, 0, 0],
        #                 [3, 5, 133545994.0, 0, 0, 0]
        #                ]
        # request_size = [12288, 262144, 262144, 65536, 65536, 131072, 131072, 32768, 32768, 65536, 65536, 65536, 16384, 16384, 32768, 32768, 32768, 8192, 8192, 8192, 8192, 0]
        # response_size = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 0]

        # for ins in test_data:
        #     instruction = ins[2]
        #     for idx in range(ins[1]):
        #         # print("[{}] {} | {} | {} | {}".format(idx, instructions[idx], instruction-instructions[idx], request_size[idx], response_size[idx]))
        #         self.vehicle.tasks_to_compute.clear()
        #         self.vehicle.strategy.update_strategy(idx, -1, {"EE 1": ins[3], "EE 2": ins[4], "EE 3": ins[5]}, False, -1)
        #         self.vehicle.generate_tasks(1)
        #         pass
        #         for task in self.vehicle.tasks_to_compute:
        #             self.assertEqual(task.instruction_count, instruction-instructions[idx])
        #             self.assertEqual(task.instruction_count_ue, instructions[idx])
        #             self.assertEqual(task.request_size, request_size[idx]/1048576 if not idx > ins[0] else 0)
        #             self.assertEqual(task.response_size, response_size[idx]/1048576 if not idx > ins[0] else 0)


if __name__ == '__main__':
    unittest.main()