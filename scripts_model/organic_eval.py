import torch
import numpy as np
import pandas as pd
import argparse
import warnings
from tqdm import tqdm
from rdkit import Chem
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from scipy.stats import wasserstein_distance
from pathlib import Path
from p_tqdm import p_map

# Suppress pymatgen warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

# Initialize featurizer globally to avoid re-creating it in each process
cn_fingerprint = CrystalNNFingerprint.from_preset("cn")


# --- Helper functions (no changes needed) ---
def create_pymatgen_structure(crys_dict):
    try:
        lattice = Lattice.from_parameters(
            a=crys_dict['lengths'][0], b=crys_dict['lengths'][1], c=crys_dict['lengths'][2],
            alpha=crys_dict['angles'][0], beta=crys_dict['angles'][1], gamma=crys_dict['angles'][2]
        )
        structure = Structure(lattice, crys_dict['atom_types'], crys_dict['frac_coords'])
        return structure
    except Exception:
        return None


def unpack_data(pred_data, gt_data_list):
    num_atoms_tensor = pred_data['num_atoms'][0]
    num_structures = num_atoms_tensor.size(0)
    ptr = torch.cat([torch.tensor([0]), num_atoms_tensor.cumsum(dim=0)])
    atom_types_tensor = pred_data['atom_types'][0]
    frac_coords_tensor = pred_data['frac_coords'][0]
    lengths_tensor = pred_data['lengths'][0]
    angles_tensor = pred_data['angles'][0]
    predicted_crystals, ground_truth_crystals = [], []
    for i in range(num_structures):
        start_idx, end_idx = ptr[i], ptr[i + 1]
        pred_atom_types_onehot = atom_types_tensor[start_idx:end_idx]
        pred_atom_types = torch.argmax(pred_atom_types_onehot, dim=1) + 1
        predicted_crystals.append({
            'atom_types': pred_atom_types.tolist(),
            'frac_coords': frac_coords_tensor[start_idx:end_idx],
            'lengths': lengths_tensor[i],
            'angles': angles_tensor[i],
        })
        gt_data = gt_data_list[i]
        ground_truth_crystals.append({
            'atom_types': gt_data.atom_types.tolist(),
            'frac_coords': gt_data.frac_coords,
            'lengths': gt_data.lengths[0],
            'angles': gt_data.angles[0],
            'smiles': gt_data.smiles,
        })
    return predicted_crystals, ground_truth_crystals


def calculate_lattice_similarity(gt_struct, pred_struct):
    gt_params = np.array(list(gt_struct.lattice.abc) + list(gt_struct.lattice.angles))
    pred_params = np.array(list(pred_struct.lattice.abc) + list(pred_struct.lattice.angles))
    mape = np.mean(np.abs((gt_params - pred_params) / gt_params))
    return max(0, 1 - mape)


def calculate_bond_similarity(gt_struct, pred_struct, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return 0.0
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() != len(gt_struct.species): return 0.0
    gt_bond_lengths = [gt_struct.get_distance(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    pred_bond_lengths = [pred_struct.get_distance(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    if not gt_bond_lengths: return 1.0
    distance = wasserstein_distance(gt_bond_lengths, pred_bond_lengths)
    return 1.0 / (1.0 + distance)


def calculate_coord_similarity(gt_struct, pred_struct):
    if len(gt_struct) != len(pred_struct): return 0.0
    match_count = 0
    for i in range(len(gt_struct)):
        try:
            gt_cn = cn_fingerprint.featurize(gt_struct, i)[0]
            pred_cn = cn_fingerprint.featurize(pred_struct, i)[0]
            if gt_cn == pred_cn: match_count += 1
        except Exception:
            continue
    return match_count / len(gt_struct) if len(gt_struct) > 0 else 1.0


# --- NEW Worker function for parallel processing ---
def process_single_structure(packed_args):
    """
    This function takes one pair of predicted/ground-truth crystal data,
    calculates all metrics, and returns a result dictionary.
    """
    pred_crys_dict, gt_crys_dict = packed_args

    pred_struct = create_pymatgen_structure(pred_crys_dict)
    gt_struct = create_pymatgen_structure(gt_crys_dict)

    if gt_struct is None or pred_struct is None:
        return None

    smiles = gt_crys_dict['smiles']
    lattice_sim = calculate_lattice_similarity(gt_struct, pred_struct)
    bond_sim = calculate_bond_similarity(gt_struct, pred_struct, smiles)
    coord_sim = calculate_coord_similarity(gt_struct, pred_struct)

    overall_sim = (lattice_sim + bond_sim + coord_sim) / 3.0

    return {
        'smiles': smiles,
        'lattice_similarity': lattice_sim,
        'bond_similarity': bond_sim,
        'coord_similarity': coord_sim,
        'structural_similarity_score': overall_sim,
    }


def main(args):
    print(f"Loading consolidated prediction file from: {args.input_file}")
    data = torch.load(args.input_file, map_location='cpu')

    pred_data_batch = data[0] if isinstance(data, list) else data

    print("Unpacking predicted and ground truth structures...")
    predicted_crystals, ground_truth_crystals = unpack_data(
        pred_data_batch,
        pred_data_batch['input_data_batch'][0]
    )

    print(f"Analyzing {len(predicted_crystals)} structure pairs in parallel...")

    # THE FIX: Convert the zip object to a list
    tasks = list(zip(predicted_crystals, ground_truth_crystals))

    # Use p_map to run the worker function in parallel
    results = p_map(process_single_structure, tasks)

    # Filter out any structures that failed processing
    results = [res for res in results if res is not None]

    if not results:
        print("No valid structures found to compare.")
        return

    df = pd.DataFrame(results)
    print("\n--- Fairer Evaluation Metrics Summary ---")
    print(df.describe())

    output_path = Path(args.input_file).parent / "fairer_metrics_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a fairer, multi-component evaluation on model predictions.")
    parser.add_argument("input_file", type=str, help="Path to the 'consolidated_reconstruct.pt' file.")
    args = parser.parse_args()
    main(args)