from typing import Any, Dict, Iterator, List, cast
from dataclasses import dataclass, field

import networkx as nx
import attr

from mec.entities import MecEntity

from .mec_types import EntityID


@dataclass(frozen=True)
class NetworkRadioEdgeAttrs:
    uplink_sinr: float
    downlink_sinr: float
    data: dict = field(default_factory=lambda: {})


@dataclass(frozen=True)
class NetworkBackhaulEdgeAttrs:
    backhaul_datarate_capacity_Mb: float
    data: dict = field(default_factory=lambda: {})


@attr.s
class MECNetwork:
    nxgraph: nx.Graph = attr.ib(default=nx.Graph())

    @classmethod
    def as_fullmesh_backhaul(
        cls,
        base_stations: List[MecEntity],
        ues: List[MecEntity],
        backhaul_datarate_capacity_Mb: float
    ) -> "MECNetwork":
        n_graph = nx.Graph()

        # add ues and base stations to graph
        for bs in base_stations:
            n_graph.add_node(bs.id, type="bs", data=bs)
        for ue in ues:
            n_graph.add_node(ue.id, type="ue", data=ue)

        # connect base stations as fullmesh
        for bs_id1 in (b.id for b in base_stations):
            for bs_id2 in (b.id for b in base_stations):
                if bs_id1 == bs_id2 or n_graph.has_edge(bs_id1, bs_id2):
                    continue

                n_graph.add_edge(
                    bs_id1,
                    bs_id2,
                    type="backhaul",
                    data=NetworkBackhaulEdgeAttrs(backhaul_datarate_capacity_Mb)
                )

        return cls(n_graph)

    def connected_ues(self, bs_id: EntityID) -> Dict[EntityID, MecEntity]:
        edges = list(self.nxgraph.edges(bs_id))
        connected_ues: Dict[MecEntity] = {}
        for edge in edges:
            edge_attrs: dict = self.nxgraph.get_edge_data(*edge)
            # TODO check node type instead of connection type
            if not edge_attrs.get("type") == "radio":
                continue

            if edge[0] != bs_id:
                ue_id = edge[0]
            else:
                ue_id = edge[1]
            ue = cast(MecEntity, self.nxgraph.nodes[ue_id].get("data"))
            connected_ues[ue_id] = ue
        return connected_ues

    def connections(self, entity_id: EntityID) -> Iterator[EntityID]:
        return self.nxgraph.neighbors(entity_id)

    def attrs_of_radio_connection(
        self, bs_id: EntityID, ue_id: EntityID
    ) -> NetworkRadioEdgeAttrs:
        edge_attrs = self.nxgraph.get_edge_data(bs_id, ue_id)
        if edge_attrs["type"] != "radio":
            raise Exception("Not an radio connection")
        return cast(NetworkRadioEdgeAttrs, edge_attrs["data"])

    def update_radio_connection(
        self,
        ue_id: EntityID,
        bs_id: EntityID,
        attrs: NetworkRadioEdgeAttrs,
    ) -> None:
        self.nxgraph[ue_id][bs_id]["data"] = attrs
