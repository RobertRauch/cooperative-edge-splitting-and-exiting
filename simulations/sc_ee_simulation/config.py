from enum import Enum
from typing import List, Optional

from pydantic import BaseModel

from utils.run_utils import BaseSimulationConfig


class SimulationType(str, Enum):
    ours = "ours"


class RewardType(str, Enum):
    reward_v1 = "reward_v1"
    reward_v2 = "reward_v2"
    reward_v3 = "reward_v3"
    reward_v4 = "reward_v4"
    reward_v5 = "reward_v5"
    reward_v6 = "reward_v6"
    reward_v7 = "reward_v7"
    reward_v8 = "reward_v8"
    reward_v9 = "reward_v9"


# class SimulationData(BaseModel):
#     dt: float
#     steps: int
#     gui_enabled: bool
#     gui_timeout: float
#     plot_map: bool
#     checkpoint_every: Optional[int]


# class TaskData(BaseModel):
#     ttl: float
#     request_size_MB: float
#     response_size_MB: float
#     instruction_count: int
#     generation_period: float
#     generation_delay_range: float = 0
#     latency_reserve: float = 0.9


# class VehicleData(BaseModel):
#     count: int
#     speed_ms: float
#     task: TaskData


# class BaseStationData(BaseModel):
#     count: int
#     min_radius: int
#     ips: float
#     bandwidth: float
#     resource_blocks: int
#     tx_frequency: float
#     tx_power: float
#     coverage_radius: float


class DDPGConfig(BaseModel):
    pass


class AgentData(BaseModel):
    # ddpg: DDPGConfig
    action_count: int
    buffer_size: int
    min_buffer_size: int
    batch_size: int
    lr: float
    target_update_tau: float
    gamma: float
    epsilon: float
    epsilon_decay_factor: float
    epsilon_min: float = 0.05


class GNNAlgorithmData(BaseModel):
    agent: AgentData
    reward: RewardType
    nn_linear_layers: List[int]
    nn_gcn_layers: List[int]
    is_training: bool = True
    gcn_aggregator: str = "sum"
    agent_nn_models_path: Optional[str] = None


# class SimulationConfig(BaseSimulationConfig):
#     simulation: SimulationData
#     vehicles: VehicleData
#     base_stations: BaseStationData
#     type: SimulationType
#     algorithm: Optional[GNNAlgorithmData]


class BaseStationData(BaseModel):
    count: int
    min_radius: int
    ips: float
    bandwidth: float
    resource_blocks: int
    tx_frequency: float
    tx_power: float
    coverage_radius: float


class TaskData(BaseModel):
    ttl: float
    generation_period: float
    generation_delay_range: float = 0
    # latency_reserve: float = 0.9


class Requirements(BaseModel):
    latency: float
    accuracy: float

class Weights(BaseModel):
    latency: float
    accuracy: float

class VehicleData(BaseModel):
    count: int
    speed_ms: float
    task: TaskData
    requirements: Requirements
    weights: Weights
    speed_variation: Optional[float]
    ips: float
    model_location: Optional[str]


class SimulationData(BaseModel):
    dt: float
    # name: str
    simulation_type: SimulationType
    steps: int
    training_steps: int
    gui_enabled: bool
    gui_timeout: float
    # plot_map: bool
    checkpoint_every: Optional[int]


class RequirementsSearch(BaseModel):
    latency_requirements: list[float]
    accuracy_requirements: list[float]
    workers: Optional[int]


class SimulationConfig(BaseSimulationConfig):
    result_dir: str
    result_name: str
    repeat: int
    repeat_start_idx: int
    episodes: int
    requirements_search: Optional[RequirementsSearch]
    simulation: SimulationData
    vehicles: list[VehicleData]
    base_stations: BaseStationData
