import copy
import osmnx as ox
import matplotlib.pyplot as plt

from mec.entities import MecEntity


def plot_map_and_nodes(G_map, nodes: list[MecEntity], filename: str):
    node_id = max(G_map.nodes) + 1
    G = copy.deepcopy(G_map)
    for node in nodes:
        location = node.location
        G.add_node(node_id, y=location.latitude, x=location.longitude)
        node_id += 1

    fig, ax = ox.plot_graph(G, save=True, filepath=filename, show=False)
    plt.close(fig)
