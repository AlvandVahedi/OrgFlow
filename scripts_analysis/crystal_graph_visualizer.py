import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx

"""
Crystal Graph Visualizer

This class loads crystal structure graphs from a PyTorch `.pt` file (a list of `Data` objects)
and visualizes them using NetworkX. Each atom is represented as a node labeled by its atomic number.

Supports both organic and inorganic samples (toggleable via commented block).

Output:
    - Saves visualized graphs as PNG images in the specified output directory.
"""

data_list = torch.load("/path/to/data.pt")
os.makedirs("graphs", exist_ok=True)

for i, item in enumerate(data_list):

    ###################### Uncomment this part for inorganic samples ########################

    # arrays = item["graph_arrays"]
    #
    # frac_coords = torch.tensor(arrays[0], dtype=torch.float)
    # atom_types = torch.tensor(arrays[1], dtype=torch.long)
    # lengths = torch.tensor(arrays[2], dtype=torch.float)
    # angles = torch.tensor(arrays[3], dtype=torch.float)
    # edge_index = torch.tensor(arrays[4].T, dtype=torch.long)
    # to_jimages = torch.tensor(arrays[5], dtype=torch.long)
    # num_atoms = torch.tensor([arrays[6]], dtype=torch.long)
    #
    # data = Data(
    #     atom_types=atom_types,
    #     frac_coords=frac_coords,
    #     lengths=lengths,
    #     angles=angles,
    #     edge_index=edge_index,
    #     to_jimages=to_jimages,
    #     num_atoms=num_atoms,
    # )
    ##########################################################################################
    G = to_networkx(item, to_undirected=True)
    print(f"Graph #{i}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    plt.figure(figsize=(5, 5))
    node_labels = {i: str(item.atom_types[i].item()) for i in range(item.num_atoms.item())}
    nx.draw(G, labels=node_labels, node_size=50, font_size=6)
    plt.savefig(f"graphs/graph_{i}.png", dpi=300)
    plt.close()