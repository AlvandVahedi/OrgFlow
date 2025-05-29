import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm
from joblib import Parallel, delayed

from diffcsp.common.data_utils import process_one_cif

def process_wrapper(row, args):
    import warnings
    import logging

    # Suppress CIF parsing warnin gs
    warnings.filterwarnings("ignore", message="Possible issue in CIF file at line*")
    logging.getLogger("pymatgen.io.cif").setLevel(logging.ERROR)

    return process_one_cif(
        row=row,
        niggli=args.niggli,
        primitive=args.primitive,
        graph_method=args.graph_method,
        prop_list=[args.prop],
        use_space_group=args.use_space_group,
        tol=args.tolerance,
    )

def process_split(split_name, df_split, args, output_dir, num_workers=180):
    print(f"\nProcessing {split_name} set ({len(df_split)} samples)...")

    results = Parallel(n_jobs=num_workers, backend="loky")(
        delayed(process_wrapper)(row, args)
        for row in tqdm(df_split.to_dict(orient="records"), desc=f"Processing {split_name}")
    )

    data_list = [r for r in results if r is not None]
    save_path = os.path.join(output_dir, f"{split_name}.pt")
    torch.save(data_list, save_path)
    print(f"Saved {len(data_list)} structures to {save_path}")

def main(args):
    df = pd.read_csv(args.csv)

    if 'material_id' not in df.columns:
        raise ValueError("CSV must contain a 'material_id' column.")

    output_dir = os.path.dirname(args.csv) if args.output_dir is None else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    n = len(df)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train + n_val]
    df_test = df.iloc[n_train + n_val:]

    process_split("train_organic", df_train, args, output_dir)
    process_split("val_organic", df_val, args, output_dir)
    process_split("test_organic", df_test, args, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CIF dataset into train/val/test .pt files.")
    parser.add_argument("--csv", type=str, required=True, help="Path to input .csv file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save train/val/test .pt files.")
    parser.add_argument("--prop", type=str, default="density", help="Property name to extract as target.")
    parser.add_argument("--niggli", action="store_true", help="Apply Niggli reduction.")
    parser.add_argument("--primitive", action="store_true", help="Convert to primitive cell.")
    parser.add_argument("--graph_method", type=str, default="crystalnn", choices=["crystalnn", "custom", "none"], help="How to build graph.")
    parser.add_argument("--use_space_group", action="store_true", help="Use symmetry info.")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Symmetry tolerance.")
    args = parser.parse_args()

    main(args)