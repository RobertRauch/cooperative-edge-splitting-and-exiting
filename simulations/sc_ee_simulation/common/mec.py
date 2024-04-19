import os
import sys

from simulations.sc_ee_simulation.common.entities import SimulationVehicle

sys.path.append(os.getcwd())

import attr
import structlog
from mec.mec import MEC
from mec.mec_network import MECNetwork, NetworkRadioEdgeAttrs
from mec.sinr.sinr_map import SINRMap
from mec.sinr import sinr
from mec.entities import UE, BaseStation, UEsMap
from mec.datarate import RadioDataRate
from mec.processing.task_processing import (
    process_bs_tasks_computation_round_robin_per_task_owner,
    process_radio_downlink_round_robin,
    process_radio_uplink_round_robin,
    process_ue_tasks_computation_round_robin,
    process_tasks_computation_fifo,
)
from typing import Callable, cast
from mec.simulation import SimulationContext
from functools import cached_property

from simulations.sc_ee_simulation.task.task_events import TaskEventsProtocol

logger = structlog.get_logger()


def create_even_rb_MBps_datarate_calculator(
    bs: BaseStation,
    connected_ues_count: int,
    network: MECNetwork,
) -> Callable[[UE], float]:
    def calculator(ue: UE) -> float:
        datarate = (
            RadioDataRate.calculate_avgrb(
                network.attrs_of_radio_connection(bs.id, ue.id).downlink_sinr,
                bs.resource_blocks,
                connected_ues_count,
            )
            / 8
        )
        # logger.debug(
        #     "create_even_rb_MBps_datarate_calculator.sinr",
        #     sinr_value=network.attrs_of_radio_connection(bs.id, ue.id).downlink_sinr,
        #     datarate=datarate,
        # )
        return datarate

    return calculator


@attr.s(kw_only=True)
class MECSimulation(MEC):
    backhaul_datarate_capacity_Mb: float = attr.ib()
    sinr_map: SINRMap = attr.ib()
    task_event_listener: TaskEventsProtocol = attr.ib()

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
        bs_list = list(self.bs_dict.values())
        for ue_id, ue in self.ue_dict.items():
            best_bs_id = self.sinr_map.get_best_bs_loc(ue.location)
            if best_bs_id is None:
                sinr_value, conn_bs = sinr.calculate_highest_sinr(ue.location, bs_list)
                if conn_bs:
                    self.sinr_map.update_loc(ue.location, sinr_value)
                    self.sinr_map.update_best_bs_loc(ue.location, conn_bs.id)

                if conn_bs is None:
                    self.network.nxgraph.remove_edges_from(
                        list(self.network.nxgraph.edges(ue_id))
                    )
                    for task in ue.tasks_sending:
                        task.target_id = None
                    continue
            else:
                conn_bs = self.bs_dict[best_bs_id]
                sinr_value = self.sinr_map.get_loc(ue.location)
            # logger.debug("sinr_value.ue", sinr_value=sinr_value, ue_id=ue_id)
            conn_bs_id = conn_bs.id

            # for now the conn bs is also computing bs
            # comp_bs_id = select_computing_bs(ue_id, conn_bs_id, self.ue_dict, self.bs_dict)
            comp_bs_id = conn_bs_id

            # select task targets
            for message in ue.tasks_sending:
                message.task.target_id = comp_bs_id

            if self.network.nxgraph.has_edge(ue_id, conn_bs_id):
                self.network.update_radio_connection(
                    ue_id,
                    conn_bs_id,
                    NetworkRadioEdgeAttrs(sinr_value, sinr_value),
                )
                continue

            # remove current connections TODO: what to do with not transfered tasks?
            self.network.nxgraph.remove_edges_from(
                list(self.network.nxgraph.edges(ue_id))
            )

            self.network.nxgraph.add_edge(
                ue_id,
                conn_bs_id,
                type="radio",
                data=NetworkRadioEdgeAttrs(sinr_value, sinr_value),
            )

    def process_uplink(self, context: SimulationContext) -> None:
        sim_dt = context.sim_info.simulation_dt
        for bs_id, bs in self.bs_dict.items():
            connected_ues_mapping = cast(UEsMap, self.network.connected_ues(bs_id))
            connected_ues = list(connected_ues_mapping.values())
            process_radio_uplink_round_robin(
                bs,
                connected_ues,
                sim_dt,
                create_even_rb_MBps_datarate_calculator(
                    bs, len(connected_ues), self.network
                ),
                logger,
                on_deadline=self.on_deadline,
            )

    def process_downlink(self, context: SimulationContext) -> None:
        sim_dt = context.sim_info.simulation_dt
        for bs_id, bs in self.bs_dict.items():
            connected_ues_mapping = cast(UEsMap, self.network.connected_ues(bs_id))
            connected_ues = list(connected_ues_mapping.values())

            process_radio_downlink_round_robin(
                bs,
                connected_ues,
                sim_dt,
                create_even_rb_MBps_datarate_calculator(
                    bs, len(connected_ues), self.network
                ),
                logger,
                on_deadline=self.on_deadline,
            )

    def process_computation(self, context: SimulationContext) -> None:
        sim_dt = context.sim_info.simulation_dt
        for bs in self.bs_dict.values():
            process_tasks_computation_fifo(
                bs, sim_dt, logger, False,
                on_deadline=self.on_deadline
            )

    def process_ue_computation(self, context: SimulationContext) -> None:
        sim_dt = context.sim_info.simulation_dt
        for ue in self.ue_dict.values():
            process_tasks_computation_fifo(
                ue, sim_dt, logger, True,
                on_deadline=self.on_deadline
            )

    def on_deadline(self, task):
        self.task_event_listener.discarded_task(task)
        if task.owner is not None:
            cast(SimulationVehicle, task.owner).ddpg_event_listener.discarded_task(task)
        # self.
