import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
import os
from torch_geometric.data import Data
import pickle
import numpy as np
from pathlib import Path

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, save_path: ValueNode, tolerance: ValueNode, use_space_group: ValueNode, use_pos_index: ValueNode,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance

        self.preprocess(save_path, preprocess_workers, prop)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

        self.__repr__()

    def preprocess(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop],
            use_space_group=self.use_space_group,
            tol=self.tolerance)
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        prop = self.scaler.transform(data_dict[self.prop])

        # # scaler is set in DataModule set stage
        # if 'properties' in data_dict and self.prop in data_dict['properties']:
        #     prop = self.scaler.transform(data_dict['properties'][self.prop])
        # else:
        #     raise KeyError(
        #         f"Property '{self.prop}' not found in data_dict['properties']. Available: {list(data_dict['properties'].keys())}")
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
            # cif=self.df.iloc[index]['CIF'], # To calculate the additional loss in model_pl
            # smiles=self.df.iloc[index]['SMILES']
        )

        if 'bond_mean' in data_dict:
            data.bond_mean = torch.tensor(data_dict['bond_mean'], dtype=torch.float32)  # [E]
        if 'bond_var' in data_dict:
            data.bond_var = torch.tensor(data_dict['bond_var'], dtype=torch.float32)  # [E]
        if 'bond_mask' in data_dict:
            # torch.bool is best for masking; batching will concat along E
            data.bond_mask = torch.tensor(data_dict['bond_mask'], dtype=torch.bool)  # [E]
        if 'bond_orders' in data_dict:
            data.bond_orders = torch.tensor(data_dict['bond_orders'], dtype=torch.long)  # [E]
        if 'bond_types' in data_dict:
            data.bond_types = data_dict['bond_types']

        # ==== SANITY CHECKS
        E = data.edge_index.size(1)
        assert data.to_jimages.size(0) == E and data.to_jimages.size(1) == 3, \
            f"to_jimages shape {tuple(data.to_jimages.shape)} != (E,3) with E={E}"
        assert int(data.num_atoms) == data.num_nodes, "num_atoms mismatch num_nodes"

        if hasattr(data, "bond_mean"):
            assert data.bond_mean.numel() == E, f"bond_mean len {data.bond_mean.numel()} != E={E}"
            assert data.bond_var.numel() == E, f"bond_var len {data.bond_var.numel()} != E={E}"
            assert data.bond_mask.numel() == E, f"bond_mask len {data.bond_mask.numel()} != E={E}"
            # no NaNs/inf in numeric fields
            assert torch.isfinite(data.bond_mean).all(), "non-finite bond_mean"
            assert torch.isfinite(data.bond_var).all(), "non-finite bond_var"
            assert (data.bond_var >= 0).all(), "negative bond_var"
            # quick peek (rare)
            # if torch.rand(()) < 0.01:
            #     cov = data.bond_mask.float().mean().item() * 100.0
            #     print(f"[dataset] idx={index} E={E} bond_cov={cov:.1f}%  "
            #           f"mean0-5={data.bond_mean[:5].tolist()} var0-5={data.bond_var[:5].tolist()} "
            #           f"edges0-5={data.edge_index[:, :5].t().tolist()}")


        if "graph_arrays_initial" in data_dict:
            (frac_coords_initial, atom_types_initial, lengths_initial, angles_initial, edge_indices_initial,
            to_jimages_initial, num_atoms_initial) = data_dict['graph_arrays_initial']
            data.frac_coords_initial = torch.Tensor(frac_coords_initial)
            data.atom_types_initial = torch.LongTensor(atom_types_initial)
            data.lengths_initial = torch.Tensor(lengths_initial).view(1, -1)
            data.angles_initial = torch.Tensor(angles_initial).view(1, -1)
            data.edge_index_initial = torch.LongTensor(
                edge_indices_initial.T).contiguous()
            data.to_jimages_initial = torch.LongTensor(to_jimages_initial)
            data.num_atoms_initial = num_atoms_initial
            data.num_bonds_initial = edge_indices_initial.shape[0]
            data.num_nodes_initial = num_atoms_initial

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict['spacegroup']])
            data.ops = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )

        if "graph_arrays_initial" in data_dict:
            (frac_coords_initial, atom_types_initial, lengths_initial, angles_initial, edge_indices_initial,
            to_jimages_initial, num_atoms_initial) = data_dict['graph_arrays_initial']
            data.frac_coords_initial = torch.Tensor(frac_coords_initial)
            data.atom_types_initial = torch.LongTensor(atom_types_initial)
            data.lengths_initial = torch.Tensor(lengths_initial).view(1, -1)
            data.angles_initial = torch.Tensor(angles_initial).view(1, -1)
            data.edge_index_initial = torch.LongTensor(
                edge_indices_initial.T).contiguous()
            data.to_jimages_initial = torch.LongTensor(to_jimages_initial)
            data.num_atoms_initial = num_atoms_initial
            data.num_bonds_initial = edge_indices_initial.shape[0]
            data.num_nodes_initial = num_atoms_initial

        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from diffcsp.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
