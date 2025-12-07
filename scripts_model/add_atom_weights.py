import pandas as pd
from pymatgen.core import Structure
from rdkit import Chem
from tqdm import tqdm
import warnings
import os

# --- Configuration ---
RIGID_ATOM_WEIGHT = 5.0
FLEXIBLE_ATOM_WEIGHT = 1.0
ALLOWED_ATOMS = {'H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'}
project_root = "/storage/alvand/flowmm"

# --- End Configuration ---

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


def calculate_rigidity_weights_from_smiles(smiles: str) -> list[float]:
    """
    Calculates rigidity weights for each atom from a SMILES string.
    Atoms in rings are considered rigid.
    """
    if not isinstance(smiles, str):
        warnings.warn("Encountered non-string SMILES. Returning empty weights.")
        return []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        warnings.warn(f"RDKit could not parse SMILES: {smiles}. Returning empty weights.")
        return []

    mol = Chem.AddHs(mol)
    ring_info = mol.GetRingInfo()
    is_in_ring = [False] * mol.GetNumAtoms()
    for ring in ring_info.AtomRings():
        for atom_idx in ring:
            is_in_ring[atom_idx] = True

    return [RIGID_ATOM_WEIGHT if in_ring else FLEXIBLE_ATOM_WEIGHT for in_ring in is_in_ring]


def process_csv(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    if 'SMILES' not in df.columns:
        raise ValueError(f"CSV file {input_path} must contain a 'SMILES' column.")

    # Filter to only allowed molecules
    original_count = len(df)
    df = df[df['SMILES'].apply(contains_only_allowed_atoms)].reset_index(drop=True)
    filtered_count = len(df)
    print(f"Filtered {original_count - filtered_count} molecules; {filtered_count} remain.")

    print(f"Processing {filtered_count} SMILES strings from {input_path}...")
    tqdm.pandas()

    weights_list = df['SMILES'].progress_apply(calculate_rigidity_weights_from_smiles)
    df['atom_rigidity_weights'] = weights_list.apply(
        lambda w: ' '.join(map(str, w)) if w else ''
    )

    df.to_csv(output_path, index=False)
    print(f"Successfully saved weighted data to {output_path}")


if __name__ == "__main__":
    train_csv = os.path.join(project_root, "data/organic/train.csv")
    val_csv = os.path.join(project_root, "data/organic/val.csv")
    test_csv = os.path.join(project_root, "data/organic/test.csv")

    train_csv_out = os.path.join(project_root, "data/organic/train_weighted.csv")
    val_csv_out = os.path.join(project_root, "data/organic/val_weighted.csv")
    test_csv_out = os.path.join(project_root, "data/organic/test_weighted.csv")

    process_csv(train_csv, train_csv_out)
    process_csv(val_csv, val_csv_out)
    process_csv(test_csv, test_csv_out)