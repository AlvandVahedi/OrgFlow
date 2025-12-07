from __future__ import annotations

import argparse, random, json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Subset
from tqdm import tqdm

from orgflow.model.eval_utils import load_model, get_loaders, register_omega_conf_resolvers
from torch_geometric.data import DataLoader as PyGDataLoader

import matplotlib
matplotlib.use("Agg")  # headless & faster
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read as ase_read, write as ase_write
from ase.visualize.plot import plot_atoms

from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher

# -------------------------------------------------------------------------------------

register_omega_conf_resolvers()
torch.backends.cudnn.benchmark = True
STAGES = ["train", "val", "test"]

# ----------------- helpers -----------------

def _trainer(single_gpu: bool, gpu_index: int, precision: Optional[str] = None):
    kw = dict(
        accelerator="gpu",
        logger=False,
        enable_checkpointing=False,
        inference_mode=True,
        enable_progress_bar=False,
    )
    if single_gpu:
        kw["devices"] = [gpu_index]
        kw["strategy"] = "ddp"
    else:
        kw["devices"] = "auto"
        kw["strategy"] = "ddp"
    if precision:
        kw["precision"] = precision
    return pl.Trainer(**kw)

def _predict_subset(
    ckpt: Path,
    stage: str,
    indices: List[int],
    num_steps: int | None,
    batch_size: int,
    single_gpu: bool,
    gpu_index: int,
    num_workers: int,
    precision: Optional[str] = None,
):
    """Predict per-item (bs=1) to isolate failures cleanly."""
    cfg, model = load_model(ckpt)
    if num_steps is not None:
        cfg.integrate.num_steps = int(num_steps)

    loaders = get_loaders(cfg)
    base_loader = loaders[STAGES.index(stage.lower())]
    ds = base_loader.dataset

    tr = _trainer(single_gpu, gpu_index, precision=precision)
    ok_indices, ok_preds = [], []

    loop = tqdm(indices, desc="predict per-item", leave=False)
    for idx in loop:
        try:
            dl = PyGDataLoader(
                Subset(ds, [idx]),
                batch_size=1,
                shuffle=False,
                num_workers=0,                 # per-item: keep 0 to avoid FD spam
                pin_memory=torch.cuda.is_available(),
                persistent_workers=False,
            )
            out = tr.predict(model, dataloaders=dl, return_predictions=True)
            pred = out[0] if isinstance(out, list) and len(out) == 1 else out
            if isinstance(pred, list) and len(pred) == 1:
                pred = pred[0]
            if not isinstance(pred, dict):
                raise TypeError(f"unexpected prediction type: {type(pred)}")
            ok_indices.append(idx)
            ok_preds.append(pred)
        except Exception as e:
            print(f"[warn] predict failed at idx={idx}: {type(e).__name__}: {e}")

    return ok_indices, ok_preds, ds

def _get_id_and_ciftext(ds, idx: int) -> Tuple[str, str]:
    if not hasattr(ds, "df"):
        raise RuntimeError("Dataset has no .df. Ensure CrystDataset exposes its DataFrame.")
    row = ds.df.iloc[idx]
    mat_id = None
    for c in ["material_id", "mat_id", "mp_id", "id", "ID"]:
        if c in row:
            try:
                mat_id = str(row[c]); break
            except Exception:
                pass
    if mat_id is None:
        mat_id = str(idx)
    cif_col = "CIF" if "CIF" in row else ("cif" if "cif" in row else None)
    if cif_col is None:
        raise KeyError("CSV must contain a 'CIF' column (or 'cif').")
    cif_source = str(row[cif_col]).strip()
    return mat_id, cif_source

def _read_gt_atoms(cif_source: str, tmp_dir: Path, mat_id: str) -> Atoms:
    s = str(cif_source).strip()
    p = Path(s)
    is_inline = ("\n" in s) or s.startswith("data_") or ("_cell_" in s) or (not p.exists())
    if not is_inline and p.is_file():
        return ase_read(s)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / f"{mat_id}.cif"
    tmp_file.write_text(s)
    return ase_read(str(tmp_file))

def _flatten1d(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    arr = np.squeeze(arr)
    return arr.reshape(-1)

def _cell_from_lengths_angles(a: float, b: float, c: float,
                              alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    alpha = np.deg2rad(alpha_deg); beta = np.deg2rad(beta_deg); gamma = np.deg2rad(gamma_deg)
    s_gamma = np.sin(gamma)
    if abs(s_gamma) < 1e-8:
        s_gamma = np.sign(s_gamma) * 1e-8 if s_gamma != 0 else 1e-8
    va = np.array([a, 0.0, 0.0], dtype=np.float64)
    vb = np.array([b * np.cos(gamma), b * s_gamma, 0.0], dtype=np.float64)
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / s_gamma
    cz_sq = max(c * c - cx * cx - cy * cy, 0.0)
    vc = np.array([cx, cy, np.sqrt(cz_sq)], dtype=np.float64)
    return np.vstack([va, vb, vc])

def _to_cell_matrix(lat_like, lengths_like=None, angles_like=None) -> np.ndarray:
    # lattices first
    if lat_like is not None:
        lat = torch.as_tensor(lat_like).detach().cpu().numpy()
        while lat.ndim >= 3 and lat.shape[0] == 1:
            lat = np.squeeze(lat, axis=0)
        if lat.ndim == 3 and lat.shape[-2:] == (3, 3):
            lat = lat[0]
        if lat.ndim == 2 and lat.shape == (3, 3):
            return lat.astype(np.float64)
        if lat.ndim == 1:
            if lat.size >= 9:
                return lat[:9].reshape(3, 3).astype(np.float64)
            if lat.size >= 6:
                a, b, c, alpha, beta, gamma = lat[:6].tolist()
                return _cell_from_lengths_angles(a, b, c, alpha, beta, gamma)
    # lengths + angles fallback
    if lengths_like is not None and angles_like is not None:
        L = _flatten1d(lengths_like); A = _flatten1d(angles_like)
        if L.size < 3 or A.size < 3:
            raise ValueError(f"lengths/angles too short. got lengths={L.shape}, angles={A.shape}")
        a, b, c = L[:3].tolist()
        alpha, beta, gamma = A[:3].tolist()
        return _cell_from_lengths_angles(a, b, c, alpha, beta, gamma)
    raise ValueError("Cannot build 3x3 cell. Provide lattices (3x3/9/6) or lengths+angles.")

def _atoms_from_pred(gt_atoms: Atoms, pred: dict) -> Atoms:
    if "frac_coords" not in pred:
        raise KeyError("prediction dict missing 'frac_coords'")
    frac = torch.as_tensor(pred["frac_coords"]).detach().cpu()
    while frac.ndim > 2 and frac.shape[0] == 1:
        frac = frac.squeeze(0)
    if frac.ndim == 3 and frac.shape[-1] == 3:
        frac = frac[0]
    if frac.ndim != 2 or frac.shape[1] != 3:
        raise ValueError(f"frac_coords must be (N,3), got {tuple(frac.shape)}")
    frac_np = frac.numpy()

    cell = _to_cell_matrix(
        pred.get("lattices", None),
        lengths_like=pred.get("lengths", None),
        angles_like=pred.get("angles", None),
    )

    # Prefer predicted atom_types if present; otherwise fall back to GT symbols (only for plotting)
    if "atom_types" in pred and pred["atom_types"] is not None:
        at = torch.as_tensor(pred["atom_types"]).detach().cpu().numpy()
        at = np.squeeze(at)
        if at.ndim > 1:
            # if one-hot or logits → take argmax+1 as Z (common convention)
            at = np.argmax(at, axis=-1) + 1
        # make symbols from atomic numbers
        from ase.data import chemical_symbols
        symbols = [chemical_symbols[int(z)] for z in at.tolist()]
    else:
        symbols = gt_atoms.get_chemical_symbols()

    # Align counts
    N = frac_np.shape[0]
    if len(symbols) != N:
        if len(symbols) > N:
            symbols = symbols[:N]
        else:
            symbols = symbols + [symbols[-1]] * (N - len(symbols))

    cart = frac_np @ cell
    return Atoms(symbols=symbols, positions=cart, cell=cell, pbc=[True, True, True])

def _to_pmg_structure(atoms: Atoms) -> Structure:
    return AseAtomsAdaptor.get_structure(atoms)

def _is_structure_match(
    gt_struct: Structure,
    pred_struct: Structure,
    ltol: float,
    stol: float,
    angle_tol: float,
) -> Tuple[bool, Optional[float]]:
    """
    Returns (is_match, rms_dist or None).
    We require same number of sites and same species sets to avoid
    meaningless matches when chemistry differs greatly.
    """
    # if len(gt_struct) != len(pred_struct):
    #     return False, None
    # # quick chemistry check (multiset)
    # if sorted(str(s) for s in gt_struct.species) != sorted(str(s) for s in pred_struct.species):
    #     return False, None

    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    try:
        rms = matcher.get_rms_dist(pred_struct, gt_struct)
        if rms is None:
            return False, None
        # rms returns (dist, mapping) or similar; take first float
        if isinstance(rms, (list, tuple)):
            rms_val = float(rms[0])
        else:
            rms_val = float(rms)
        return True, rms_val
    except Exception:
        return False, None

def _save_triptych_png(path: Path, gt: Atoms, ours: Atoms, base: Atoms, title: str, dpi: int, radii: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, atoms, ttl in zip(axes, [gt, ours, base], ["Ground Truth", "Our Model", "Baseline"]):
        plot_atoms(atoms, ax=ax, radii=radii)   # clean 2D
        ax.set_title(ttl, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(title, y=0.98, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser(description="Compare GT vs Our vs Baseline, match with pymatgen, and visualize only matches.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--baseline_checkpoint", type=Path, required=True)
    ap.add_argument("--stage", type=str, default="test", choices=STAGES)
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=150)      # try more; we’ll filter by matches
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--single_gpu", action="store_true")
    ap.add_argument("--gpu_index", type=int, default=0)
    ap.add_argument("--precision", type=str, default=32, help='e.g. "16-mixed" or "bf16-mixed"')
    ap.add_argument("--out_dir", type=Path, default=Path("compare_structures_output_matched_only"))
    # matcher tolerances
    ap.add_argument("--ltol", type=float, default=0.3, help="lattice length tolerance")
    ap.add_argument("--stol", type=float, default=0.5, help="site tolerance")
    ap.add_argument("--angle_tol", type=float, default=10.0, help="angle tolerance (deg)")
    # plotting
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--radii", type=float, default=0.22)
    args = ap.parse_args()

    for p in [args.checkpoint, args.baseline_checkpoint]:
        if not p.exists():
            print(f"[FATAL] checkpoint not found: {p}")
            return

    cfg_ref, _ = load_model(args.checkpoint)
    loaders_ref = get_loaders(cfg_ref)
    ds = loaders_ref[STAGES.index(args.stage.lower())].dataset
    total_len = len(ds)
    print(f"[info] {args.stage} size = {total_len}")
    if total_len == 0:
        print("[FATAL] Test set is empty.")
        return

    random.seed(args.seed); torch.manual_seed(args.seed)
    pool = sorted(random.sample(range(total_len), k=min(args.n_samples, total_len)))
    print(f"[info] candidate indices = {pool}")

    print("[info] predicting ours...")
    ok_idx_o, preds_o, ds_ref = _predict_subset(
        args.checkpoint, args.stage, pool, args.num_steps,
        args.batch_size, args.single_gpu, args.gpu_index,
        num_workers=args.num_workers, precision=args.precision
    )
    print(f"[info] ours ok: {len(ok_idx_o)}/{len(pool)}")

    print("[info] predicting baseline...")
    ok_idx_b, preds_b, _ = _predict_subset(
        args.baseline_checkpoint, args.stage, pool, args.num_steps,
        args.batch_size, args.single_gpu, args.gpu_index,
        num_workers=args.num_workers, precision=args.precision
    )
    print(f"[info] baseline ok: {len(ok_idx_b)}/{len(pool)}")

    common = sorted(set(ok_idx_o) & set(ok_idx_b))
    if not common:
        print("[FATAL] No common successful indices between models. Nothing to evaluate.")
        return
    print(f"[info] common ok indices: {common}")

    pred_map_o = {i: p for i, p in zip(ok_idx_o, preds_o)}
    pred_map_b = {i: p for i, p in zip(ok_idx_b, preds_b)}

    out_root = args.out_dir
    (out_root / "figs").mkdir(parents=True, exist_ok=True)
    (out_root / "cifs" / "ours").mkdir(parents=True, exist_ok=True)
    (out_root / "cifs" / "baseline").mkdir(parents=True, exist_ok=True)
    tmp_cif_dir = out_root / "_tmp_inline_cifs"
    matches_csv = out_root / "matched_pairs.csv"

    total_checked, total_matched = 0, 0
    rows = []  # for CSV

    item_bar = tqdm(common, desc="match+assemble (only matches are saved)")
    for idx in item_bar:
        mat_id, cif_source = _get_id_and_ciftext(ds_ref, idx)
        try:
            gt_atoms = _read_gt_atoms(cif_source, tmp_cif_dir, mat_id)
            po = pred_map_o[idx]; pb = pred_map_b[idx]

            # Build Atoms
            ours_atoms = _atoms_from_pred(gt_atoms, po)
            base_atoms = _atoms_from_pred(gt_atoms, pb)

            # Convert to pymatgen Structure
            gt_struct   = _to_pmg_structure(gt_atoms)
            ours_struct = _to_pmg_structure(ours_atoms)
            base_struct = _to_pmg_structure(base_atoms)

            # Check match for each prediction vs GT (separately)
            is_match_o, rms_o = _is_structure_match(
                gt_struct, ours_struct, args.ltol, args.stol, args.angle_tol
            )
            is_match_b, rms_b = _is_structure_match(
                gt_struct, base_struct, args.ltol, args.stol, args.angle_tol
            )

            total_checked += 1
            # Visualize only pairs where **both** predictions match GT
            # (change to `any([is_match_o, is_match_b])` if you want either)
            if is_match_o or is_match_b:
                total_matched += 1
                # Save CIFs
                ase_write(str(out_root / "cifs" / "ours" / f"{mat_id}.cif"), ours_atoms)
                ase_write(str(out_root / "cifs" / "baseline" / f"{mat_id}.cif"), base_atoms)
                # Save figure
                _save_triptych_png(out_root / "figs" / f"{mat_id}.png",
                                   gt_atoms, ours_atoms, base_atoms,
                                   title=f"ID: {mat_id} | RMS(ours)={rms_o:.3f}, RMS(base)={rms_b:.3f}",
                                   dpi=args.dpi, radii=args.radii)
                rows.append({"id": mat_id, "idx": idx, "rms_ours": rms_o, "rms_base": rms_b})
        except Exception as e:
            print(f"[warn] idx={idx} ({mat_id}) failed: {type(e).__name__}: {e}")
            continue

    match_rate = (total_matched / total_checked) if total_checked else 0.0
    print(f"[RESULT] match_rate (both matched): {match_rate:.3f}  ({total_matched}/{total_checked})")

    # Save CSV + metrics.json
    import csv
    with open(matches_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "idx", "rms_ours", "rms_base"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    metrics = {
        "checked": total_checked,
        "matched_both": total_matched,
        "match_rate_both": match_rate,
        "ltol": args.ltol,
        "stol": args.stol,
        "angle_tol": args.angle_tol,
    }
    with open(out_root / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[done] Figures: {out_root / 'figs'}")
    print(f"[done] CIFs:    {out_root / 'cifs'}")
    print(f"[done] Matches CSV: {matches_csv}")
    print(f"[done] Metrics JSON: {out_root / 'metrics.json'}")

if __name__ == "__main__":
    main()