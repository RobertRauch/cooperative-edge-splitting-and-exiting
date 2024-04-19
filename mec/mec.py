from functools import cached_property
from typing import Optional

import attr
import networkx as nx

from mec.mec_network import MECNetwork
from mec.simulation import SimulationContext
from mec.entities import BSsMap, UEsMap


@attr.s(kw_only=True)
class MEC:
    __mec_context_key: str = "__mec"

    ue_dict: UEsMap = attr.ib()
    bs_dict: BSsMap = attr.ib()

    @cached_property
    def network(self) -> MECNetwork:
        return MECNetwork()

    def process_uplink(self, context: SimulationContext) -> None:
        pass

    def process_downlink(self, context: SimulationContext) -> None:
        pass

    def process_computation(self, context: SimulationContext) -> None:
        pass

    def process_ue_computation(self, context: SimulationContext) -> None:
        pass

    def update_network(self, context: SimulationContext) -> None:
        pass

    def on_deadline(self, task) -> None:
        pass

    def process(self, context: SimulationContext) -> None:
        self.process_uplink(context)
        self.process_computation(context)
        self.process_downlink(context)

    @classmethod
    def of(cls, context: SimulationContext) -> Optional['MEC']:
        return context.data[cls.__mec_context_key]

    def assign_to_context(self, context: SimulationContext) -> None:
        context.data[self.__mec_context_key] = self
