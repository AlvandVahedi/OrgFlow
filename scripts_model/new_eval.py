#!/usr/bin/env python3
# scripts_model/new_eval.py

import argparse
import pickle
import math
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

from orgflow.model.eval_utils import (
    register_omega_conf_resolvers,
    load_model,
    get_loaders,
)
from torch_geometric.data import DataLoader as GeomDataLoader


def evaluate(checkpoint_path: str, stage: str):
    # 1. Register OmegaConf resolvers before loading
    register_omega_conf_resolvers()

    # 2. Load model + config, get dataloaders
    cfg, model = load_model(Path(checkpoint_path))
    loaders = get_loaders(cfg)
    stage = stage.lower()
    idx = ["train", "val", "test"].index(stage)
    orig_loader = loaders[idx]

    # 3. Reinstantiate DataLoader with num_workers=0 to avoid worker IPC issues
    loader = GeomDataLoader(
        orig_loader.dataset,
        batch_size=orig_loader.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 4. Run inference on single GPU
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
    )
    predictions = trainer.predict(
        model,
        dataloaders=[loader],
        return_predictions=True,
        ckpt_path=checkpoint_path,
    )

    # 5. Set up StructureMatcher and accumulators
    matcher = StructureMatcher()
    total = 0
    matched = 0
    rmsd_list = []
    length_mse = []
    angle_mse = []

    # 6. Loop and compute metrics
    for batch_out in tqdm(predictions, desc="Evaluating"):
        at_pred    = batch_out["atom_types"]      # [B, N]
        fc_pred    = batch_out["frac_coords"]     # [B, N, 3]
        lat_pred   = batch_out["lattices"]        # [B, 3, 3]
        lengths_pr = batch_out["lengths"]         # [B, 3]
        angles_pr  = batch_out["angles"]          # [B, 3]
        data       = batch_out["input_data_batch"]

        B = lengths_pr.shape[0]
        for i in range(B):
            # ground truth
            mask = data.batch == i
            atoms_true = data.atom_types[mask].cpu().tolist()
            frac_true  = data.frac_coords[mask].cpu().tolist()
            lengths_tr = data.lengths[i].cpu().tolist()
            angles_tr  = data.angles[i].cpu().tolist()

            lat_tr = Lattice.from_lengths_and_angles(lengths_tr, angles_tr)
            struct_tr = Structure(lat_tr, atoms_true, frac_true,
                                  coords_are_cartesian=False)

            lat_pr = Lattice(lat_pred[i].cpu().numpy())
            struct_pr = Structure(lat_pr,
                                  at_pred[i].cpu().tolist(),
                                  fc_pred[i].cpu().tolist(),
                                  coords_are_cartesian=False)

            total += 1
            if matcher.fit(struct_tr, struct_pr):
                matched += 1
            try:
                rmsd = matcher.get_rms_dist(struct_tr, struct_pr)
            except Exception:
                rmsd = float('nan')
            rmsd_list.append(rmsd)

            # lattice params MSE
            lp_diff = np.array(lengths_pr[i].cpu()) - np.array(lengths_tr)
            ap_diff = np.array(angles_pr[i].cpu())  - np.array(angles_tr)
            length_mse.append((lp_diff**2).mean())
            angle_mse.append((ap_diff**2).mean())

    # 7. Aggregate
    match_rate = matched / total
    mean_rmsd  = float(np.nanmean(rmsd_list))
    rmse_len   = math.sqrt(np.mean(length_mse))
    rmse_ang   = math.sqrt(np.mean(angle_mse))

    metrics = {
        'split': stage,
        'total': total,
        'match_rate': match_rate,
        'mean_rmsd': mean_rmsd,
        'rmse_lattice_lengths': rmse_len,
        'rmse_lattice_angles': rmse_ang,
    }

    # 8. Print & save
    print("\n=== Evaluation ===")
    for k, v in metrics.items():
        print(f"{k:25s}: {v}")
    out_p = Path(checkpoint_path).with_suffix('.eval.pkl')
    with open(out_p, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"\nSaved to {out_p}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('checkpoint', type=str, help='path to .ckpt')
    parser.add_argument(
        '--stage',
        choices=['train','val','test'],
        default='val',
        help='which split to evaluate',
    )
    args = parser.parse_args()
    evaluate(args.checkpoint, args.stage)

if __name__ == '__main__':
    main()
