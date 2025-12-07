import numpy as np
import torch
from flowmm.model.bond_data import get_bond_data_from_key

order_to_type = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 5: "AROMATIC"}

def compute_bond_length_loss(structure, edge_index, bond_orders, device, verbose=False):
    """
    Variance-aware bond length loss with debug printing.
    """

    edge_index = edge_index.T.cpu().numpy()  # shape (num_edges, 2)
    atom_symbols = [site.specie.symbol for site in structure]

    order_to_type = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 5: "AROMATIC"}

    losses = []

    for idx, ((i, j), bond_order) in enumerate(zip(edge_index, bond_orders)):
        bond_type = order_to_type.get(int(bond_order), "SINGLE")
        a1, a2 = atom_symbols[i], atom_symbols[j]
        pos_i = structure[i].coords
        pos_j = structure[j].coords
        dist = np.linalg.norm(pos_i - pos_j)

        bond_key = (a1, a2, bond_type)
        bond_data = get_bond_data_from_key(bond_key)

        if bond_data is not None:
            mean = bond_data["mean"]
            var = bond_data["variance"]
            loss = ((dist - mean) ** 2) / (var + 1e-8)
            losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.tensor(losses, device=device).mean()