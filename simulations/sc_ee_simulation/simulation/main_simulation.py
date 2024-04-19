from typing import List
import attr
import torch
import sys

from mec.simulation import IISMotionSimulation, SimulationContext
from mec.tasks.task import TaskComputationEntity
from mec.mec import MEC
from mec.utils.plot_utils import plot_map_and_nodes
from mec.entities import MecEntity, UEsMap
from simulations.sc_ee_simulation.common.entities import SimulationVehicle
from simulations.sc_ee_simulation.common.stats_collector import DDPGStatsCollector
from simulations.sc_ee_simulation.common.task_events import DDPGTaskEventsListener
from mec.datarate import RadioDataRate
from typing import cast
from simulations.sc_ee_simulation.simulation.maddpg import MADDPG

from pathlib import Path
import osmnx as ox
import matplotlib.pyplot as plt
import copy
import numpy as np
import random
import math


@attr.s(kw_only=True)
class MainSimulation(IISMotionSimulation):
    mec: MEC = attr.ib()
    stats_collector: DDPGStatsCollector = attr.ib()
    maddpg: MADDPG = attr.ib()
    rewards = []
    actor_loss, critic_loss = 0, 0
    strategies = []
    for i in range(22):
        for j in range(8):
            for z in range(4):
                strategies.append({'id': id, 'split': i, 'autoencoder': j, 'exit': z})

    def map(self, value, left_min, left_max, right_min, right_max):
        leftSpan = left_max - left_min
        rightSpan = right_max - right_min

        valueScaled = float(value - left_min) / float(leftSpan)

        return right_min + (valueScaled * rightSpan)

    def get_split_idx(self, action: float):
        scaled_action = self.map(action, 0, 1, 0, 21)
        return int(round(scaled_action))
    
    def get_ae_idx(self, action: float):
        scaled_action = self.map(action, 0, 1, -1, 7)
        return int(round(scaled_action))
    
    def obs_list_to_state_vector(self, observation):
        state = np.array([])
        for obs in observation:
            state = np.concatenate([state, obs])
        return state

    def update_strategy(self, context: SimulationContext):
        for bs in self.mec.bs_dict.values():
            connected_ues_mapping = cast(UEsMap, self.mec.network.connected_ues(bs.id))
            connected_ues = list(connected_ues_mapping.values())
            is_training = context.sim_info.simulation_step < context.sim_info.training_steps or context.sim_info.training_steps == -1
            n_agents = len(connected_ues)

            obs = []
            obs_ = []
            rewards = []
            actions = []

            for ue in connected_ues:
                ue = cast(SimulationVehicle, ue)
                
                if is_training:
                    ue.ma_agent.train_mode()
                else:
                    ue.ma_agent.eval()

                avg_latency = ue.ddpg_event_listener.get_average_latency()
                avg_accuracy = ue.ddpg_event_listener.get_accuracy()
                avg_transmission = ue.ddpg_event_listener.get_average_transmission_rate()
                avg_exit = ue.ddpg_event_listener.get_average_exit()

                w_a, w_l = ue.weights.accuracy, ue.weights.latency
                

                cur_bandwidth = ((ue.step_data["uplink"] if "uplink" in ue.step_data else 0)
                                                     + (ue.step_data["downlink"] if "downlink" in ue.step_data else 0)) / ( 2 if 
                                                     ("downlink" in ue.step_data and "uplink" in ue.step_data) else 1)

                state = [
                    avg_latency,
                    cur_bandwidth,
                    avg_transmission,
                    ue.requirements.latency,
                    ue.requirements.accuracy,
                ]


                #ours 4
                acc_reward = ((avg_accuracy - ue.requirements.accuracy)/(1 - ue.requirements.accuracy))
                lat_reward = (1 - (avg_latency/ue.requirements.latency))

                reward = (acc_reward * w_a) + (lat_reward * w_l)


                if is_training:
                    action = ue.ma_agent.select_action(np.array(state))
                else:
                    action = ue.ma_agent.select_unrandomized_action(np.array(state))

                actions.append(action)
                rewards.append(reward)
                obs.append(state)
                obs_.append(ue.prev_state)


                ue.selected_action = action

                s = np.argmax(action[0:22])
                a = np.argmax(action[22:30])
                e = np.argmax(action[30:34])
                ue.strategy.update_strategy(s, a, None, True, e)
                ue.prev_state = state
                actor_loss, critic_loss = self.actor_loss, self.critic_loss

                self.stats_collector.timestamp = context.sim_info.simulation_step
                self.stats_collector.exit = avg_exit
                self.stats_collector.split = ue.strategy.split
                self.stats_collector.reward = reward
                self.stats_collector.latency = avg_latency
                self.stats_collector.actor_loss = actor_loss
                self.stats_collector.critic_loss = critic_loss
                self.stats_collector.bandwidth = avg_transmission
                self.stats_collector.accuracy = avg_accuracy
                self.stats_collector.accepted = ue.ddpg_event_listener.accepted_tasks
                self.stats_collector.dropped = ue.ddpg_event_listener.discarded_tasks
                self.stats_collector.server_utilization = ue.ddpg_event_listener.server_utilization
                self.stats_collector.device_utilization = ue.ddpg_event_listener.device_utilization
                self.stats_collector.autoencoder = ue.strategy.autoencoder
                self.stats_collector.bandwidth_2 = cur_bandwidth
                self.stats_collector.create_statistic(ue.log_idx)

                # print("[{}]: {} | {} | {}".format(context.sim_info.simulation_step, action, self.get_split_idx(action[0]), reward))
                if context.sim_info.simulation_step % 100 == 0 and is_training:
                    ue.ma_agent.noise_decay()
                self.rewards.append(reward)

            if n_agents == self.maddpg.n_agents:
                if is_training:
                    self.maddpg.memory.add(obs, self.obs_list_to_state_vector(obs), actions, rewards, obs_, self.obs_list_to_state_vector(obs_), [False]*self.maddpg.n_agents)
                if is_training and context.sim_info.simulation_step % 100 == 0:
                    self.actor_loss, self.critic_loss = self.maddpg.train()
                if (ue.model_save_location is not None 
                    and not ue.model_save_location == ""
                    and is_training):
                    Path(ue.model_save_location).mkdir(parents=True, exist_ok=True)
                    self.maddpg.save(ue.model_save_location + "checkpoint.pt")

            if context.sim_info.simulation_step % 50 == 0:
                print("Step {}".format(context.sim_info.simulation_step))

            for ue in connected_ues:
                ue.ddpg_event_listener.reset()

    def plott_map_and_nodes(self, G_map, nodes: list[MecEntity], filename: str):
        node_id = max(G_map.nodes) + 1
        G = copy.deepcopy(G_map)
        for node in nodes:
            location = node.location
            G.add_node(node_id, y=location.latitude, x=location.longitude)
            node_id += 1

        fig, ax = ox.plot_graph(G, save=True, filepath=filename, show=False)
        plt.close(fig)

    def reset_used_timestamp(self, context: SimulationContext):
        tasks: List[TaskComputationEntity] = []
        sim_dt = context.sim_info.simulation_dt
        for ue in self.mec.ue_dict.values():
            tasks += ue.tasks_computing + ue.tasks_sending + ue.tasks_to_compute
            ue.used_timestep_interval = 0
        for bs in self.mec.bs_dict.values():
            tasks += bs.tasks_computing + bs.tasks_sending
            bs.used_timestep_interval = 0
        for task in tasks:
            if round(task.task.used_timestep_interval, 8) > sim_dt:
                raise Exception(
                    "used more timestamp for task then is the simulation step {}".format(task.task.used_timestep_interval)
                )
            task.task.used_timestep_interval = 0
            task.task.used_timestep += sim_dt
            if task.task.lived_timestep >= task.task.ttl:
                task.task.used_timestep = task.task.ttl + task.task.created_timestep
                self.mec.on_deadline(task.task)

    def generate_tasks(self, context: SimulationContext) -> None:
        for ue in self.mec.ue_dict.values():
            ue.generate_tasks(context.sim_info.simulation_dt)

            ue.tasks_computing += [t.computing_ue for t in ue.tasks_to_compute if
                                   t.computing_ue.remaining_computation > 0]
            ue.tasks_sending += [t.request for t in ue.tasks_to_compute if
                                 t.computing_ue.remaining_computation == 0]
            ue.tasks_to_compute.clear()

    def simulation_step(self, context: SimulationContext) -> None:
        if self.gui_enabled:
            self.plott_map_and_nodes(
                self.iismotion.map.driveGraph,
                list(self.mec.ue_dict.values()) + list(self.mec.bs_dict.values()),
                "mecmap.png",
            )

        self.reset_used_timestamp(context)
        self.generate_tasks(context)
        self.update_strategy(context)

        self.mec.update_network(context)

        self.mec.process_ue_computation(context)
        self.mec.update_network(context)
        self.mec.process_uplink(context)
        self.mec.process_computation(context)
        self.mec.process_downlink(context)

        # move with vehicles
        self.iismotion.stepAllCollections(context.sim_info.new_day)
