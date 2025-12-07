import warnings

import copy
import faulthandler
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import torch
from p_tqdm import p_umap
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from rdkit import Chem
from sklearn.metrics import accuracy_score, recall_score, precision_score
from torch_scatter import segment_coo, segment_csr

from scipy.spatial import cKDTree
from pymatgen.core.operations import SymmOp
from collections import defaultdict
import re

from flowmm.model.bond_data import get_bond_data_from_key, ATOMIC_SYMBOLS


# from multiprocessing import Pool
faulthandler.enable()

BOND_ORDER_TO_TYPE = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 5: "AROMATIC"}

# Tensor of unit cells. Assumes 27 cells in -1, 0, 1 offsets in the x and y dimensions
# Note that differing from OCP, we have 27 offsets here because we are in 3D
OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]

EPSILON = 1e-5

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']


CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)

ALLOWED_ATOMS = {'H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'}


def _extract_cif_labels(cif_str: str):
    """
    Try to read _atom_site_label from the CIF.
    Returns a Python list of labels or None if not available.
    """
    try:
        parser = CifParser(StringIO(cif_str), occupancy_tolerance=1.0)
        block = next(iter(parser._cif.data.values()))
        labels = block.get("_atom_site_label", None)
        if labels is None:
            return None
        # Make sure it's a list of strings
        return [str(x) for x in labels]
    except Exception:
        return None

def contains_only_allowed_atoms(smiles: str) -> bool:
    """
    Returns True if all atoms in the RDKit-parsed molecule belong to ALLOWED_ATOMS.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            warnings.warn(f"RDKit could not parse SMILES: {smiles}. Excluding molecule.")
            return False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in ALLOWED_ATOMS:
                warnings.warn(
                    f"Molecule '{smiles}' contains disallowed atom '{atom.GetSymbol()}'. Excluding."
                )
                return False
        return True
    except Exception as e:
        warnings.warn(f"Error checking atoms for SMILES '{smiles}': {e}. Excluding molecule.")
        return False


def build_crystal_manually(crystal_str, niggli=True, primitive=False):
      try:
        """Build crystal from cif string."""
        cif_stream = StringIO(crystal_str)
        parser = CifParser(cif_stream, occupancy_tolerance=1.0)

        block = next(iter(parser._cif.data.values()))

        a = float(block["_cell_length_a"])
        b = float(block["_cell_length_b"])
        c = float(block["_cell_length_c"])
        alpha = float(block["_cell_angle_alpha"])
        beta = float(block["_cell_angle_beta"])
        gamma = float(block["_cell_angle_gamma"])
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

        # Atom site fields
        symbols = block["_atom_site_type_symbol"]
        labels = block["_atom_site_label"] # block.data.get('_atom_site_label')
        x = block["_atom_site_fract_x"]
        y = block["_atom_site_fract_y"]
        z = block["_atom_site_fract_z"]

        species = []
        coords = []
        for i in range(len(symbols)):
            species.append(symbols[i])
            coords.append([float(x[i]), float(y[i]), float(z[i])])

        # Add labels as site property
        props = {"_atom_site_label": labels}

        # Create structure with original labels
        crystal = Structure(
            lattice=lattice,
            species=species,
            coords=coords,
            site_properties=props,
            validate_proximity=False
        )

        if primitive:
            crystal = crystal.get_primitive_structure()
        if niggli:
            crystal = crystal.get_reduced_structure()

        return crystal

      except Exception as e:
          print(e)
          return None

def check_disconnected_components(mol):
    """Check for disconnected components using Union-Find algorithm."""
    n_nodes = mol.GetNumAtoms()
    parent = list(range(n_nodes))

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # undirected
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        union(src, dst)

    components = {}
    for node in range(n_nodes):
        root = find(node)
        components.setdefault(root, []).append(node)

    return list(components.values())

def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
    try:
        crystal = Structure.from_str(crystal_str, fmt='cif')

        if primitive:
            crystal = crystal.get_primitive_structure()

        if niggli:
            crystal = crystal.get_reduced_structure()

        canonical_crystal = Structure(
            lattice=Lattice.from_parameters(*crystal.lattice.parameters),
            species=crystal.species,
            coords=crystal.frac_coords,
            coords_are_cartesian=False,
        )
        # match is gaurantteed because cif only uses lattice params & frac_coords
        # assert canonical_crystal.matches(crystal)
        return canonical_crystal
    except Exception as e:
        return None

def build_crystal_flowmm(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
    try:
        crystal = Structure.from_str(crystal_str, fmt='cif')

        if primitive:
            crystal = crystal.get_primitive_structure()

        if niggli:
            crystal = crystal.get_reduced_structure()

        canonical_crystal = Structure(
            lattice=Lattice.from_parameters(*crystal.lattice.parameters),
            species=crystal.species,
            coords=crystal.frac_coords,
            coords_are_cartesian=False,
        )
        return canonical_crystal
    except Exception as e:
        return None

def refine_spacegroup(crystal, tol=0.01):
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_conventional_standard_structure()
    space_group = spga.get_space_group_number()
    crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return crystal, space_group


def get_symmetry_info(crystal, tol=0.01):
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal_refined = spga.get_refined_structure()
    c = pyxtal()
    try:
        c.from_seed(crystal_refined, tol=0.01)
    except Exception:
        c.from_seed(crystal_refined, tol=0.0001)

    space_group = c.group.number
    species = []
    coords = []
    parent_idx = []
    op_idx = []
    matrices = []
    labels = []

    for i_site, site in enumerate(c.atom_sites):
        specie = site.specie
        base_pos = site.position
        for j_op, syms in enumerate(site.wp):
            matrices.append(syms.affine_matrix)
            frac = syms.operate(base_pos) % 1.0
            species.append(specie)
            coords.append(frac)
            parent_idx.append(i_site)
            op_idx.append(j_op)
            labels.append(f"{str(specie)}|u{i_site}|op{j_op}")

    species = np.array(species, dtype=object)
    coords = (np.array(coords, dtype=float) % 1.0)
    parent_idx = np.array(parent_idx, dtype=int)
    op_idx = np.array(op_idx, dtype=int)
    matrices = np.array(matrices, dtype=float)

    sym_info = {
        'anchors': parent_idx,
        'wyckoff_ops': matrices,
        'spacegroup': space_group,
        'op_index': op_idx,
        'labels': labels,
    }

    crystal_out = Structure(
        lattice=Lattice.from_parameters(*np.array(c.lattice.get_para(degree=True))),
        species=species.tolist(),
        coords=coords,
        coords_are_cartesian=False,
        site_properties={"_atom_site_label": labels},
    )
    return crystal_out, sym_info

def process_one(row, niggli, primitive, graph_method, prop_list, use_space_group=False, tol=0.01):
     if graph_method == 'crystalnn':
        crystal_str = row['cif']
        crystal = build_crystal_flowmm(
            crystal_str, niggli=niggli, primitive=primitive)
        if crystal is None:
            return None
        result_dict = {}
        if use_space_group:
            crystal, sym_info = get_symmetry_info(crystal, tol = tol)
            result_dict.update(sym_info)
        else:
            result_dict['spacegroup'] = 1
        graph_arrays = build_crystal_graph_flowmm(crystal, graph_method)
        properties = {k: row[k] for k in prop_list if k in row.keys()}
        result_dict.update({
            'mp_id': row['material_id'],
            'cif': crystal_str,
            'graph_arrays': graph_arrays
        })
        result_dict.update(properties)

        if "cif_initial" in row:
            crystal_str_initial = row['cif_initial']
            crystal_initial = build_crystal_flowmm(
                crystal_str_initial, niggli=niggli, primitive=primitive)
            graph_arrays_initial = build_crystal_graph_flowmm(crystal_initial, graph_method)
            result_dict['graph_arrays_initial'] = graph_arrays_initial
            result_dict['cif_initial'] = crystal_str_initial

        return result_dict
     else:
        try:
            crystal_str = row['cif']
            material_id = row.get('material_id', 'unknown')

            # === SMILES checks unchanged ===
            smiles = row.get('SMILES', None)
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"Invalid SMILES for material {material_id}: {smiles}")
                    return None
                if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
                    print(f"Sanitization failed for material {material_id} with SMILES: {smiles}")
                    return None
                components = check_disconnected_components(mol)
                if len(components) > 1:
                    print(f"Material {material_id} skipped due to disconnected components in SMILES: {smiles}")
                    return None

            try:
                crystal = build_crystal_manually(crystal_str, niggli=niggli, primitive=primitive)
            except Exception as e:
                print(f"Failed to build crystal for material {material_id}: {str(e)}")
                return None

            if crystal is None:
                print(f"build_crystal returned None for material {material_id}")
                return None

            if use_space_group:
                try:
                    crystal, sym_info = get_symmetry_info(crystal, tol=tol)
                except Exception as e:
                    print(f"Failed to get symmetry info for material {material_id}: {str(e)}")
                    return None
            else:
                sym_info = {'spacegroup': 1}

            custom_connectivity_str = row.get('atom_connectivity', None)
            symmetry_ops_str = row.get('symmetry', None)

            try:
                edge_indices, to_jimages, num_atoms = build_crystal_graph(
                    crystal,
                    custom_connectivity_str=custom_connectivity_str,
                    symmetry_ops_str=symmetry_ops_str
                )
            except Exception as e:
                print(f"Failed to build crystal graph for material {material_id}: {str(e)}")
                return None

            # Atom features
            frac_coords = crystal.frac_coords
            atom_types_Z = [Element(el).Z for el in crystal.species]
            atom_symbols = [el.symbol if hasattr(el,'symbol') else str(el) for el in crystal.species]
            lengths = crystal.lattice.abc
            angles = crystal.lattice.angles

            # Targets
            properties = {k: v for k, v in row.items() if k in prop_list}

            # Validation
            if len(edge_indices) == 0:
                print(f"Empty edge_index for material {material_id}")
                return None
            if np.max(edge_indices) >= num_atoms or np.min(edge_indices) < 0:
                print(f"edge_index out of bounds for material {material_id}: max={np.max(edge_indices)}, num_atoms={num_atoms}")
                return None
            if np.min(atom_types_Z) < 1 or np.max(atom_types_Z) > 100:
                print(f"Invalid atom types for material {material_id}: {atom_types_Z}")
                return None

            # === New: per-edge bond metadata for later loss ===
            bond_orders = []
            bond_types = []
            bond_mean = []
            bond_var = []
            bond_mask = []

            # Map label->index once for fast lookup of CSV orders
            label_list = crystal.site_properties.get("_atom_site_label", None)
            label_to_index = None
            if label_list is not None:
                label_to_index = {label: i for i, label in enumerate(label_list)}

            # Build a quick dict of CSV orders if available
            csv_order = {}
            if custom_connectivity_str is not None and label_to_index is not None:
                for line in custom_connectivity_str.strip().splitlines():
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    a1, a2, o = parts[0], parts[1], parts[2]
                    try:
                        i0, j0 = label_to_index[a1], label_to_index[a2]
                    except KeyError:
                        continue
                    try:
                        o_int = int(o)
                    except:
                        o_int = 1
                    # store both directions as same order
                    csv_order[(i0, j0)] = o_int
                    csv_order[(j0, i0)] = o_int

            # For every final directed edge, assign order -> type -> stats
            for (u, v), tj in zip(edge_indices, to_jimages):
                order_int = csv_order.get((u, v), 1)
                btype = BOND_ORDER_TO_TYPE.get(order_int, "SINGLE")
                a1 = atom_symbols[u]
                a2 = atom_symbols[v]
                data = get_bond_data_from_key((a1, a2, btype))
                if data is None:
                    bond_mask.append(False)
                    bond_mean.append(0.0)
                    bond_var.append(1.0)
                else:
                    bond_mask.append(True)
                    bond_mean.append(float(data["mean"]))
                    bond_var.append(float(max(data["variance"], 1e-6)))
                bond_orders.append(int(order_int))
                bond_types.append(btype)

            result_dict = {
                'mp_id': material_id,
                'graph_arrays': (frac_coords, atom_types_Z, lengths, angles, edge_indices, to_jimages, num_atoms),
                'properties': properties,
                'sym_info': sym_info,
            }
            for k, v in properties.items():
                result_dict[k] = v

            if 'spacegroup' in sym_info:
                result_dict['spacegroup'] = sym_info['spacegroup']
            if 'wyckoff_ops' in sym_info:
                result_dict['wyckoff_ops'] = sym_info['wyckoff_ops']  # np.array OK
            if 'anchors' in sym_info:
                result_dict['anchors'] = sym_info['anchors']

            # add bond fields we computed
            result_dict.update({
                'bond_orders': np.asarray(bond_orders, dtype=np.int64),
                'bond_types': np.asarray(bond_types, dtype=object),
                'bond_mean': np.asarray(bond_mean, dtype=np.float32),
                'bond_var': np.asarray(bond_var, dtype=np.float32),
                'bond_mask': np.asarray(bond_mask, dtype=bool),
            })
            return result_dict

        except Exception as e:
            print(f'Error processing material {row.get("material_id", "unknown")}: {str(e)}')
            return None


_sym_re = re.compile(r'([+-]?\d*/?\d*)?([xyz])')

def _parse_one_symop_frac(expr: str):
    # returns 3x4 affine (fractional) from an 'x,y,z' style string
    rows = []
    for token in expr.split(','):
        token = token.strip()
        # build row coefficients for x,y,z and constant shift
        a = {'x':0.0,'y':0.0,'z':0.0}
        shift = 0.0
        # find terms like 'x', '-y', '1/2', '+z', '1/2-x'
        idx = 0
        s = token.replace(' ', '')
        # accumulate as sum(coeff*var) + shift
        # read sign+number/var chunks
        j = 0
        cur = ''
        parts = []
        for ch in s:
            if ch in '+-' and cur:
                parts.append(cur)
                cur = ch
            else:
                cur += ch
        if cur:
            parts.append(cur)
        for p in parts:
            # variable term?
            m = _sym_re.fullmatch(p)
            if m and m.group(2) in 'xyz':
                coef = m.group(1)
                if coef in (None, '', '+'): val = 1.0
                elif coef == '-': val = -1.0
                else:
                    num = coef
                    if '/' in num:
                        n,d = num.split('/')
                        val = float(n)/float(d)
                    else:
                        val = float(num)
                a[m.group(2)] += val
            else:
                # pure shift like '+1/2' or '-1'
                if '/' in p:
                    sgn = 1.0
                    if p[0] == '+': p = p[1:]
                    elif p[0] == '-':
                        sgn = -1.0
                        p = p[1:]
                    n,d = p.split('/')
                    shift += sgn * float(n)/float(d)
                else:
                    shift += float(p)
        rows.append([a['x'], a['y'], a['z'], shift])
    # make SymmOp (fractional)
    M = np.eye(4)
    for r, row in enumerate(rows):
        M[r, :3] = row[:3]
        M[r, 3] = row[3]
    return SymmOp(M)

def _parse_symmetry_ops(symmetry_ops_str):
    if not symmetry_ops_str:
        return [SymmOp.from_rotation_and_translation(np.eye(3), [0,0,0])]
    ops = []
    for line in symmetry_ops_str.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        ops.append(_parse_one_symop_frac(line))
    return ops

def _nearest_image_shift(fi, fj):
    # fi, fj are 3-vectors (np.array), fractional space
    df = fj - fi
    return np.round(df).astype(int)

def build_crystal_graph(crystal, custom_connectivity_str=None, symmetry_ops_str=None):
    labels = crystal.site_properties.get("_atom_site_label", [f"{site.species_string}{i}" for i, site in enumerate(crystal)])
    if labels is None:
        labels = [f"{site.species_string}{i + 1}" for i, site in enumerate(crystal)]
    label_to_index = {label: i for i, label in enumerate(labels)}

    try:
        ops = _parse_symmetry_ops(symmetry_ops_str)
    except Exception:
        ops = [SymmOp.from_rotation_and_translation(np.eye(3), [0,0,0])]

    frac = np.mod(crystal.frac_coords, 1.0)  # wrap
    edge_pairs = set()  # (u,v,sx,sy,sz)
    edge_indices = []
    to_jimages = []

    seeds = []
    for line in custom_connectivity_str.strip().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        a1, a2 = parts[0], parts[1]
        try:
            i0, j0 = label_to_index[a1], label_to_index[a2]
        except KeyError:
            continue
        seeds.append((i0, j0))

    tree = cKDTree(frac)
    # expand by symmetry and compute image shifts
    for i0, j0 in seeds:
        fi0, fj0 = frac[i0], frac[j0]
        for op in ops:
            fi = np.mod(op.operate(fi0), 1.0)
            fj = np.mod(op.operate(fj0), 1.0)
            di = tree.query(fi, k=1)[1]
            dj = tree.query(fj, k=1)[1]
            shift = _nearest_image_shift(frac[di], frac[dj])  # int3

            #  Dropping true self-loops (same node, same image) ----
            if int(di) == int(dj) and (shift[0] == 0 and shift[1] == 0 and shift[2] == 0):
                continue
            #  ---------------------------------------------------

            key = (int(di), int(dj), int(shift[0]), int(shift[1]), int(shift[2]))
            if key not in edge_pairs:
                edge_pairs.add(key)
                edge_indices.append([di, dj])
                to_jimages.append(shift.tolist())

            # add reverse
            key_r = (int(dj), int(di), int(-shift[0]), int(-shift[1]), int(-shift[2]))
            if key_r not in edge_pairs:
                edge_pairs.add(key_r)
                edge_indices.append([dj, di])
                to_jimages.append((-shift).tolist())

    edge_indices = np.array(edge_indices, dtype=int)
    to_jimages = np.array(to_jimages, dtype=int)
    num_atoms = len(crystal)

    # Double-checking to not see any self-loops
    if edge_indices.size > 0:
        same = (edge_indices[:, 0] == edge_indices[:, 1]) & np.all(to_jimages == 0, axis=1)
        assert not np.any(same), f"Found {int(same.sum())} true self-loops"

    return edge_indices, to_jimages, num_atoms

def build_crystal_graph_flowmm(crystal, graph_method='crystalnn'):
    """
    """

    if graph_method == 'crystalnn':
        try:
            crystal_graph = StructureGraph.from_local_env_strategy(
                crystal, CrystalNN)
        except:
            crystalNN_tmp = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False, search_cutoff=10)
            crystal_graph = StructureGraph.from_local_env_strategy(
                crystal, crystalNN_tmp)
    elif graph_method == 'none':
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix,
                       lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != 'none':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]
    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms

def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)

def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])

def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)

def compute_volume(batch_lattice):
    """Compute volume from batched lattice matrix

    batch_lattice: (N, 3, 3)
    """
    vector_a, vector_b, vector_c = torch.unbind(batch_lattice, dim=1)
    return torch.abs(torch.einsum('bi,bi->b', vector_a,
                                  torch.cross(vector_b, vector_c, dim=1)))

def lengths_angles_to_volume(lengths, angles):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    return compute_volume(lattice)

def lattice_matrix_to_params(matrix):
    lengths = np.sqrt(np.sum(matrix ** 2, axis=1)).tolist()

    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(np.dot(matrix[j], matrix[k]) /
                            (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / np.pi
    a, b, c = lengths
    alpha, beta, gamma = angles
    return a, b, c, alpha, beta, gamma

def lattices_to_params_shape(lattices):

    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[...,i] = torch.clamp(torch.sum(lattices[...,j,:] * lattices[...,k,:], dim = -1) /
                            (lengths[...,j] * lengths[...,k]), -1., 1.)
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles

def frac_to_cart_coords(
    frac_coords,
    lengths,
    angles,
    num_atoms,
    regularized = True,
    lattices = None
):
    if regularized:
        frac_coords = frac_coords % 1.
    if lattices is None:
        lattices = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattices, num_atoms, dim=0)
    pos = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)  # cart coords

    return pos

def cart_to_frac_coords(
    cart_coords,
    lengths,
    angles,
    num_atoms,
    regularized = True
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.linalg.pinv(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum('bi,bij->bj', cart_coords, inv_lattice_nodes)
    if regularized:
        frac_coords = frac_coords % 1.
    return frac_coords


def get_pbc_distances(
    coords,
    edge_index,
    lengths,
    angles,
    to_jimages,
    num_atoms,
    num_bonds,
    coord_is_cart=False,
    return_offsets=False,
    return_distance_vec=False,
    lattices=None
):
    if lattices is None:
        lattices = lattice_params_to_matrix_torch(lengths, angles)

    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = torch.repeat_interleave(lattices, num_atoms, dim=0)
        pos = torch.einsum('bi,bij->bj', coords, lattice_nodes)  # cart coords

    j_index, i_index = edge_index

    distance_vectors = pos[j_index] - pos[i_index]

    # correct for pbc
    lattice_edges = torch.repeat_interleave(lattices, num_bonds, dim=0)
    offsets = torch.einsum('bi,bij->bj', to_jimages.float(), lattice_edges)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors

    if return_offsets:
        out["offsets"] = offsets

    return out


def radius_graph_pbc_wrapper(data, radius, max_num_neighbors_threshold, device):
    cart_coords = frac_to_cart_coords(
        data.frac_coords, data.lengths, data.angles, data.num_atoms)
    return radius_graph_pbc(
        cart_coords, data.lengths, data.angles, data.num_atoms, radius,
        max_num_neighbors_threshold, device)

def repeat_blocks(
    sizes,
    repeats,
    continuous_indexing=True,
    start_idx=0,
    block_inc=0,
    repeat_inc=0,
):
    """Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(
        torch.arange(len(sizes), device=sizes.device), repeats
    )

    # Get total size of output array, as needed to initialize output indexing array
    N = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(
                block_inc[: r1[-1]], indptr, reduce="sum"
            )
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    res = id_ar.cumsum(0)
    return res


def radius_graph_pbc(pos, lengths, angles, natoms, radius, max_num_neighbors_threshold, device, lattices=None):
    
    # device = pos.device
    batch_size = len(natoms)
    if lattices is None:
        cell = lattice_params_to_matrix_torch(lengths, angles)
    else:
        cell = lattices
    # position of the atoms
    atom_pos = pos


    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = (
        torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(
            atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor"
        )
    ) + index_offset_expand
    index2 = (
        atom_count_sqr % num_atoms_per_image_expand
    ) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).
    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)
    inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
    min_dist_a1 = (1 / inv_min_dist_a1).reshape(-1,1)

    cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
    inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
    min_dist_a2 = (1 / inv_min_dist_a2).reshape(-1,1)
    
    cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
    inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
    min_dist_a3 = (1 / inv_min_dist_a3).reshape(-1,1)
    
    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = torch.ones(3, dtype=torch.long, device=device)
    min_dist = torch.cat([min_dist_a1, min_dist_a2, min_dist_a3], dim=-1) # N_graphs * 3
#     reps = torch.cat([rep_a1.reshape(-1,1), rep_a2.reshape(-1,1), rep_a3.reshape(-1,1)], dim=1) # N_graphs * 3
    
    unit_cell_all = []
    num_cells_all = []

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float)
        for rep in max_rep
    ]
    
    unit_cell = torch.cat([_.reshape(-1,1) for _ in torch.meshgrid(cells_per_dim)], dim=-1)
    
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

#     # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    
    
    radius_real = (min_dist.min(dim=-1)[0] + 0.01)#.clamp(max=radius)
    
    radius_real = torch.repeat_interleave(radius_real, num_atoms_per_image_sqr * num_cells)

    # print(min_dist.min(dim=-1)[0])
    
    # radius_real = radius
    
    mask_within_radius = torch.le(atom_distance_sqr, radius_real * radius_real)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)
    
    if max_num_neighbors_threshold is not None:

        mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
            natoms=natoms,
            index=index1,
            atom_distance=atom_distance_sqr,
            max_num_neighbors_threshold=max_num_neighbors_threshold,
        )

        if not torch.all(mask_num_neighbors):
            # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
            index1 = torch.masked_select(index1, mask_num_neighbors)
            index2 = torch.masked_select(index2, mask_num_neighbors)
            unit_cell = torch.masked_select(
                unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
            )
            unit_cell = unit_cell.view(-1, 3)
            
    else:
        ones = index1.new_ones(1).expand_as(index1)
        num_neighbors = segment_coo(ones, index1, dim_size=natoms.sum())

        # Get number of (thresholded) neighbors per image
        image_indptr = torch.zeros(
            natoms.shape[0] + 1, device=device, dtype=torch.long
        )
        image_indptr[1:] = torch.cumsum(natoms, dim=0)
        num_neighbors_image = segment_csr(num_neighbors, image_indptr)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


def get_max_neighbors_mask(
    natoms, index, atom_distance, max_num_neighbors_threshold
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(
        max=max_num_neighbors_threshold
    )

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(
        natoms.shape[0] + 1, device=device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor(
            [True], dtype=bool, device=device
        ).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full(
        [num_atoms * max_num_neighbors], np.inf, device=device
    )

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_real_cutoff = distance_sort[:,max_num_neighbors_threshold].reshape(-1,1).expand(-1,max_num_neighbors) + 0.01
    
    mask_distance = distance_sort < distance_real_cutoff
    
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors
    )
    
    
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
#     index_sort = torch.masked_select(index_sort, mask_finite)
    index_sort = torch.masked_select(index_sort, mask_finite & mask_distance)
    
    num_neighbor_per_node = (mask_finite & mask_distance).sum(dim=-1)
    num_neighbors_image = segment_csr(num_neighbor_per_node, image_indptr)
    

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


def radius_graph_pbc_(cart_coords, lengths, angles, num_atoms,
                     radius, max_num_neighbors_threshold, device,
                     topk_per_pair=None):
    """Computes pbc graph edges under pbc.

    topk_per_pair: (num_atom_pairs,), select topk edges per atom pair

    Note: topk should take into account self-self edge for (i, i)
    """
    batch_size = len(num_atoms)

    # position of the atoms
    atom_pos = cart_coords

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = num_atoms
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()

    # index offset between images
    index_offset = (
        torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        (atom_count_sqr // num_atoms_per_image_expand)
    ).long() + index_offset_expand
    index2 = (
        atom_count_sqr % num_atoms_per_image_expand
    ).long() + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)

    if topk_per_pair is not None:
        assert topk_per_pair.size(0) == num_atom_pairs
        atom_distance_sqr_sort_index = torch.argsort(atom_distance_sqr, dim=1)
        assert atom_distance_sqr_sort_index.size() == (num_atom_pairs, num_cells)
        atom_distance_sqr_sort_index = (
            atom_distance_sqr_sort_index +
            torch.arange(num_atom_pairs, device=device)[:, None] * num_cells).view(-1)
        topk_mask = (torch.arange(num_cells, device=device)[None, :] <
                     topk_per_pair[:, None])
        topk_mask = topk_mask.view(-1)
        topk_indices = atom_distance_sqr_sort_index.masked_select(topk_mask)

        topk_mask = torch.zeros(num_atom_pairs * num_cells, device=device)
        topk_mask.scatter_(0, topk_indices, 1.)
        topk_mask = topk_mask.bool()

    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask)

    num_neighbors = torch.zeros(len(cart_coords), device=device)
    num_neighbors.index_add_(0, index1, torch.ones(len(index1), device=device))
    num_neighbors = num_neighbors.long()
    max_num_neighbors = torch.max(num_neighbors).long()

    # Compute neighbors per image
    _max_neighbors = copy.deepcopy(num_neighbors)
    _max_neighbors[
        _max_neighbors > max_num_neighbors_threshold
    ] = max_num_neighbors_threshold
    _num_neighbors = torch.zeros(len(cart_coords) + 1, device=device).long()
    _natoms = torch.zeros(num_atoms.shape[0] + 1, device=device).long()
    _num_neighbors[1:] = torch.cumsum(_max_neighbors, dim=0)
    _natoms[1:] = torch.cumsum(num_atoms, dim=0)
    num_neighbors_image = (
        _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]
    )

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        if topk_per_pair is None:
            return torch.stack((index2, index1)), unit_cell, num_neighbors_image
        else:
            return torch.stack((index2, index1)), unit_cell, num_neighbors_image, topk_mask

    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with values greater than radius*radius so we can easily remove unused distances later.
    distance_sort = torch.zeros(
        len(cart_coords) * max_num_neighbors, device=device
    ).fill_(radius * radius + 1.0)

    # Create an index map to map distances from atom_distance_sqr to distance_sort
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index1 * max_num_neighbors
        + torch.arange(len(index1), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance_sqr)
    distance_sort = distance_sort.view(len(cart_coords), max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index1
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with distances greater than the radius
    mask_within_radius = torch.le(distance_sort, radius * radius)
    index_sort = torch.masked_select(index_sort, mask_within_radius)

    # At this point index_sort contains the index into index1 of the closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index1), device=device).bool()
    mask_num_neighbors.index_fill_(0, index_sort, True)

    # Finally mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
    index1 = torch.masked_select(index1, mask_num_neighbors)
    index2 = torch.masked_select(index2, mask_num_neighbors)
    unit_cell = torch.masked_select(
        unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)

    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask_num_neighbors)

    edge_index = torch.stack((index2, index1))

    if topk_per_pair is None:
        return edge_index, unit_cell, num_neighbors_image
    else:
        return edge_index, unit_cell, num_neighbors_image, topk_mask


def min_distance_sqr_pbc(cart_coords1, cart_coords2, lengths, angles,
                         num_atoms, device, return_vector=False,
                         return_to_jimages=False):
    """Compute the pbc distance between atoms in cart_coords1 and cart_coords2.
    This function assumes that cart_coords1 and cart_coords2 have the same number of atoms
    in each data point.
    returns:
        basic return:
            min_atom_distance_sqr: (N_atoms, )
        return_vector == True:
            min_atom_distance_vector: vector pointing from cart_coords1 to cart_coords2, (N_atoms, 3)
        return_to_jimages == True:
            to_jimages: (N_atoms, 3), position of cart_coord2 relative to cart_coord1 in pbc
    """
    batch_size = len(num_atoms)

    # Get the positions for each atom
    pos1 = cart_coords1
    pos2 = cart_coords2

    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(cart_coords2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the vector between atoms
    # shape (num_atom_squared_sum, 3, 27)
    atom_distance_vector = pos1 - pos2
    atom_distance_sqr = torch.sum(atom_distance_vector ** 2, dim=1)

    min_atom_distance_sqr, min_indices = atom_distance_sqr.min(dim=-1)

    return_list = [min_atom_distance_sqr]

    if return_vector:
        min_indices = min_indices[:, None, None].repeat([1, 3, 1])

        min_atom_distance_vector = torch.gather(
            atom_distance_vector, 2, min_indices).squeeze(-1)

        return_list.append(min_atom_distance_vector)

    if return_to_jimages:
        to_jimages = unit_cell.T[min_indices].long()
        return_list.append(to_jimages)

    return return_list[0] if len(return_list) == 1 else return_list


class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        if isinstance(X, torch.Tensor):
            X = X.clone().detach()
        else:
            X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        # https://github.com/pytorch/pytorch/issues/29372
        self.stds = torch.std(X, dim=0, unbiased=False) + EPSILON

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(),
            stds=self.stds.clone().detach())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )


def get_scaler_from_data_list(data_list, key):
    dl = np.asarray([d[key] for d in data_list])
    targets = torch.from_numpy(dl)
    scaler = StandardScalerTorch()
    scaler.fit(targets)
    return scaler

def preprocess(input_file, num_workers, niggli, primitive, graph_method,
               prop_list, use_space_group=False, tol=0.01):
    df = pd.read_csv(input_file)
    df = df[df['SMILES'].apply(contains_only_allowed_atoms)].reset_index(drop=True)
    print(f"Processing {len(df)} rows from {input_file}")

    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],
        [niggli] * len(df),
        [primitive] * len(df),
        [graph_method] * len(df),
        [prop_list] * len(df),
        [use_space_group] * len(df),
        [tol] * len(df),
        num_cpus=num_workers)

    # Filter out None results and count failures
    valid_results = []
    failed_results = []
    for i, result in enumerate(unordered_results):
        if result is not None:
            valid_results.append(result)
        else:
            failed_results.append(i)

    print(f"\nProcessing Summary:")
    print(f"Total rows: {len(df)}")
    print(f"Valid results: {len(valid_results)}")
    print(f"Failed results: {len(failed_results)}")

    if not valid_results:
        raise ValueError(
            "No valid results after preprocessing. "
            "Check your data and preprocessing parameters. "
            f"Failed rows: {failed_results[:10]}..."
        )

    # Create mapping of material IDs to results
    mpid_to_results = {}
    for result in valid_results:
        mp_id = result.get('mp_id', None)
        if mp_id is not None:
            mpid_to_results[mp_id] = result
        else:
            print("Warning: One result missing 'mp_id' field, skipping.")

    # Only include results for which we have valid data
    ordered_results = []
    missing_ids = []
    for idx in range(len(df)):
        material_id = df.iloc[idx]['material_id']
        if material_id in mpid_to_results:
            ordered_results.append(mpid_to_results[material_id])
        else:
            missing_ids.append(material_id)

    if missing_ids:
        print(f"\nWarning: {len(missing_ids)} material IDs from input CSV not found in processed results")
        print("First few missing IDs:", missing_ids[:5])

    print(f"\nFinal dataset size: {len(ordered_results)}")
    return ordered_results


def preprocess_tensors(crystal_array_list, niggli, primitive, graph_method):
    def process_one(batch_idx, crystal_array, niggli, primitive, graph_method):
        frac_coords = crystal_array['frac_coords']
        atom_types = crystal_array['atom_types']
        lengths = crystal_array['lengths']
        angles = crystal_array['angles']
        crystal = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False)
        graph_arrays = build_crystal_graph(crystal, graph_method)
        result_dict = {
            'batch_idx': batch_idx,
            'graph_arrays': graph_arrays,
        }
        return result_dict

    unordered_results = p_umap(
        process_one,
        list(range(len(crystal_array_list))),
        crystal_array_list,
        [niggli] * len(crystal_array_list),
        [primitive] * len(crystal_array_list),
        [graph_method] * len(crystal_array_list),
        num_cpus=30,
    )
    ordered_results = list(
        sorted(unordered_results, key=lambda x: x['batch_idx']))
    return ordered_results

def add_scaled_lattice_prop(data_list, lattice_scale_method):
    for d in data_list:
        graph_arrays = d['graph_arrays']
        lengths = np.array(graph_arrays[2])
        angles = np.array(graph_arrays[3])
        num_atoms = graph_arrays[-1]

        assert len(lengths) == len(angles) == 3
        assert isinstance(num_atoms, int)

        if lattice_scale_method == 'scale_length':
            lengths = lengths / float(num_atoms)**(1/3)

        d['scaled_lattice'] = np.concatenate([lengths, angles])

        if "graph_arrays_initial" in d:
            graph_arrays_initial = d['graph_arrays_initial']
            lengths_initial = np.array(graph_arrays_initial[2])
            angles_initial = np.array(graph_arrays_initial[3])
            num_atoms_initial = graph_arrays_initial[-1]

            assert len(lengths_initial) == len(angles_initial) == 3
            assert isinstance(num_atoms_initial, int)

            if lattice_scale_method == 'scale_length':
                lengths_initial = lengths_initial / float(num_atoms_initial)**(1/3)

            d['scaled_lattice_initial'] = np.concatenate([lengths_initial, angles_initial])

# def add_scaled_lattice_prop(data_list, lattice_scale_method):
#     for dict in data_list:
#         graph_arrays = dict['graph_arrays']
#         # the indexes are brittle if more objects are returned
#         lengths = graph_arrays[2]
#         angles = graph_arrays[3]
#         num_atoms = graph_arrays[-1]
#
#         assert len(lengths) == len(angles) == 3
#         assert isinstance(num_atoms, int)
#
#         if lattice_scale_method == 'scale_length':
#             lengths = lengths / float(num_atoms)**(1/3)
#
#         dict['scaled_lattice'] = np.concatenate([lengths, angles])
#
#         if "graph_arrays_initial" in dict:
#             graph_arrays_initial = dict['graph_arrays_initial']
#             lengths_initial = graph_arrays_initial[2]
#             angles_initial = graph_arrays_initial[3]
#             num_atoms_initial = graph_arrays_initial[-1]
#             assert lengths_initial.shape[0] == angles_initial.shape[0] == 3
#             assert isinstance(num_atoms_initial, int)
#
#             if lattice_scale_method == 'scale_length':
#                 lengths_initial = lengths_initial / float(num_atoms_initial)**(1/3)
#
#             dict['scaled_lattice_initial'] = np.concatenate([lengths_initial, angles_initial])
#

def mard(targets, preds):
    """Mean absolute relative difference."""
    assert torch.all(targets > 0.)
    return torch.mean(torch.abs(targets - preds) / targets)


def batch_accuracy_precision_recall(
    pred_edge_probs,
    edge_overlap_mask,
    num_bonds
):
    if (pred_edge_probs is None and edge_overlap_mask is None and
            num_bonds is None):
        return 0., 0., 0.
    pred_edges = pred_edge_probs.max(dim=1)[1].float()
    target_edges = edge_overlap_mask.float()

    start_idx = 0
    accuracies, precisions, recalls = [], [], []
    for num_bond in num_bonds.tolist():
        pred_edge = pred_edges.narrow(
            0, start_idx, num_bond).detach().cpu().numpy()
        target_edge = target_edges.narrow(
            0, start_idx, num_bond).detach().cpu().numpy()

        accuracies.append(accuracy_score(target_edge, pred_edge))
        precisions.append(precision_score(
            target_edge, pred_edge, average='binary'))
        recalls.append(recall_score(target_edge, pred_edge, average='binary'))

        start_idx = start_idx + num_bond

    return np.mean(accuracies), np.mean(precisions), np.mean(recalls)


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means),
                              np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds),
                             np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(
            self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none
