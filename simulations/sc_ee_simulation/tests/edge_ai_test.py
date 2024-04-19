import os
cwd = os.getcwd()
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
from simulations.sc_ee_simulation.baseline.edge_ai.task.task_size_generator import EdgeAITaskSizeGenerator
from simulations.sc_ee_simulation.task.task_executor import TaskExecutor
from simulations.sc_ee_simulation.task.image_id_generator import RandomIdGenerator
from simulations.sc_ee_simulation.common.strategy import Strategy, Requirements
from simulations.sc_ee_simulation.common.entities import SimulationVehicle
from simulations.sc_ee_simulation.common.task_events import DDPGTaskEventsListener

from simulations.sc_ee_simulation.baseline.edge_ai.edge_ai import EdgeAI
from simulations.sc_ee_simulation.baseline.edge_ai.predicator import Predicator

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
  
class TestComputation(unittest.TestCase):
    base_station: BaseStationIISMotion
    vehicle: SimulationVehicle
    datarate_calculator: Callable[[UE], float]
    logger: any
      
    def setUp(self):
        layer_data = json.load(open(os.getcwd() + "/data/computer_vision/layer_data.json"), object_pairs_hook=OrderedDict)
        predicator = Predicator(layer_data=layer_data)
        # FLOPS are for FP32 float
        predicator.train_predicator([
                                    82580000000000/2,   # RTX 4090
                                    #  10070000000000/2,   # RTX 2080
                                    #  7465000000000/2,    # RTX 2070
                                    #  8873000000000/2,    # GTX 1080
                                    #  12740000000000/2,   # RTX 3060
                                    #  19470000000000/2,   # RTX 4060
                                    #  29770000000000/2,   # RTX 3080
                                    #  4375000000000/2,    # GTX 1060
                                    #  29150000000000/2,   # RTX 4070
                                    #  48740000000000/2,   # RTX 4080
                                    #  40090000000000/2,   # RTX 4070 TI
                                    #  35580000000000/2,   # RTX 3090
                                    #  51480000000000/2,   # RX 7900 XT
                                    #  3481000000000/2, # Adreno 740 - Snapdragon 8 Gen 2
                                    #  17000000000000/2, # A16 bionic - iPhone 14
                                    #  20800000000/2, # PowerVR GE8300 - redmi a1
                                    #  54400000000/2, # PowerVR GE8320 - redmi 9c
                                    #  686000000000/2, # Mali-G68 MC4 - MediaTek Dimensity 1080
                                    #  753000000000/2, # Adreno 642L - Qualcomm Snapdragon 778G
                                    #  1330000000000/2, # NVIDIA Jetson TX2
                                    #  236000000000/2, # NVIDIA Jetson Nano
                                    23600000000, 
                                    ],
                                    [1, 2, 3, 4]
                                    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                                    )  
        self.edge_ai = EdgeAI(predicator=predicator, layer_data=layer_data)
  
    

    def test_input_being_best(self):
        best_e, best_s = self.edge_ai.static_optimizer(1.368, 4, 22, 12288, 0.0055, 82580000000000/2, 23600000000)
        print(best_e)
        print(best_s)

  
if __name__ == '__main__':
    unittest.main()