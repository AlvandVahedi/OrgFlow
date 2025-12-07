import os
import csv
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Set

from pymatgen.core import Structure, Lattice, Element
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph

# ------------------ utils ------------------

def as_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def split_starts(num_atoms) -> Tuple[List[int], List[int]]:
    if hasattr(num_atoms, "tolist"):
        num_atoms = num_atoms.tolist()
    na = [int(n) for n in num_atoms]
    idx = [0]
    for n in na:
        idx.append(idx[-1] + n)
    return idx, na

def load_top_graph_indices(ranking_csv: Path, topk: int) -> List[int]:
    rows = []
    with open(ranking_csv, "r") as f:
        r = csv.DictReader(f)
        # If CSV already has 'rank', use it; otherwise assume file is sorted ascending
        for idx, row in enumerate(r, start=1):
            gi = int(row["graph_index"])
            rk_str = (row.get("rank") or "").strip()
            rk = int(rk_str) if rk_str else idx
            rms = row.get("rms_kabsch") or row.get("rms") or ""
            try:
                rmsf = float(rms) if rms not in ("", "None") else float("inf")
            except Exception:
                rmsf = float("inf")
            rows.append((rk, gi, rmsf))
    rows.sort(key=lambda x: x[0])
    return [gi for (_, gi, _) in rows[:topk]]

def pick_lattice_for(i: int, t: int, L: torch.Tensor) -> np.ndarray:
    """
    Handle multiple lattice layouts:
      (N,T,3,3) → L[i,t]
      (T,N,3,3) → L[t,i]
      (T,3,3)   → L[t]
      (N,3,3)   → L[i]
      (3,3)     → L (shared)
    """
    if L.dim() == 4:
        if L.shape[-2:] != (3, 3):
            raise ValueError(f"Unexpected L shape {tuple(L.shape)}")
        # Prefer time-first if both indices fit
        if L.shape[0] > t and L.shape[1] > i:
            return as_np(L[t, i])
        return as_np(L[i, t])
    elif L.dim() == 3:
        if L.shape == torch.Size([3, 3]):
            return as_np(L)
        if L.shape[0] > t and L.shape[1:] == torch.Size([3, 3]):
            return as_np(L[t])
        if L.shape[0] > i and L.shape[1:] == torch.Size([3, 3]):
            return as_np(L[i])
        raise ValueError(f"Unexpected 3D lattice shape {tuple(L.shape)}")
    else:
        raise ValueError(f"Unsupported lattice tensor dim={L.dim()} {tuple(L.shape)}")

def slice_Z_for_sample(i: int, a0: int, a1: int, t: int, Z: torch.Tensor) -> np.ndarray:
    """
    Handle atom types in shapes:
      (N,T,A), (N,A), (T, A_total), (A_total,), (N,T,Amax) padded.
      Returns 1D array of length (a1-a0) with ints (atomic numbers).
    """
    if Z.dim() == 3:
        # Prefer Z[i, 0] (types constant over t)
        if Z.shape[0] > i and Z.shape[1] >= 1:
            zi = Z[i, 0]
            out = as_np(zi[: (a1 - a0)]).astype(int)
        else:
            zt = Z[t]
            out = as_np(zt[a0:a1]).astype(int)
    elif Z.dim() == 2:
        # Could be (N,A) or (T, A_total)
        if Z.shape[0] > i and Z.shape[1] >= (a1 - a0):
            out = as_np(Z[i, : (a1 - a0)]).astype(int)
        else:
            zt = Z[min(t, Z.shape[0] - 1)]
            out = as_np(zt[a0:a1]).astype(int)
    elif Z.dim() == 1:
        out = as_np(Z[a0:a1]).astype(int)
    else:
        raise ValueError(f"Unsupported Z shape {tuple(Z.shape)}")

    # Clamp to valid periodic table range to avoid crashes
    out = np.clip(out, 1, 118).astype(int)
    return out

# ------------------ bonding (same logic as viz) ------------------

def compute_bonds(struct: Structure, same_image_only: bool = True) -> List[Tuple[int, int]]:
    """
    Build bonds via CrystalNN. If same_image_only=True, drop edges with to_jimage != (0,0,0).
    Returns undirected edge list (i<j).
    """
    edges: Set[Tuple[int, int]] = set()
    try:
        cnn = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)
        sg = StructureGraph.with_local_env_strategy(struct, cnn)
        for u, v, data in sg.graph.edges(data=True):
            tj = data.get("to_jimage", (0, 0, 0))
            if same_image_only and tuple(tj) != (0, 0, 0):
                continue
            ui, vi = int(u), int(v)
            if ui == vi:
                continue
            a, b = (ui, vi) if ui < vi else (vi, ui)
            edges.add((a, b))
    except Exception:
        # If CrystalNN fails, return empty bonds rather than crashing export
        pass
    return sorted(edges)

def write_bonds_csv(path: Path, bonds: List[Tuple[int, int]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "j"])
        for (i, j) in bonds:
            w.writerow([i, j])

def write_structure_graph_json(path: Path, struct: Structure) -> None:
    """
    Full StructureGraph JSON (including to_jimage per edge). Useful for later re-use.
    """
    try:
        cnn = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)
        sg = StructureGraph.with_local_env_strategy(struct, cnn)
        with open(path, "w") as f:
            json.dump(sg.as_dict(), f)
    except Exception:
        # Don't block export on graph JSON errors
        pass

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_pt", required=True, help="Path to consolidated_recon_trajectory.pt")
    ap.add_argument("--ranking_csv", required=True, help="Path to ranking.csv with graph_index column")
    ap.add_argument("--out_dir", required=True, help="Directory to write per-frame CIFs (+ bonds)")
    ap.add_argument("--topk", type=int, default=10)

    # NEW: bond export options
    ap.add_argument("--save_bonds", action="store_true", help="Also write frame_XXXX_bonds.csv")
    ap.add_argument("--save_graph_json", action="store_true", help="Also write frame_XXXX_graph.json (full StructureGraph)")
    ap.add_argument("--allow_pbc_links", action="store_true", help="Allow bonds across periodic images (to_jimage ≠ 0)")

    args = ap.parse_args()

    traj_path = Path(args.traj_pt)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    top_graph_indices = load_top_graph_indices(Path(args.ranking_csv), args.topk)
    want = set(top_graph_indices)

    cons = torch.load(traj_path, map_location="cpu")
    cons0 = {k: v[0] for k, v in cons.items()}

    FC = cons0["frac_coords"]
    L = cons0["lattices"]
    Z = cons0["atom_types"]
    NA = cons0["num_atoms"]
    BI = cons0["batch_indices"]

    # optional num_steps
    num_steps_txt = traj_path.parent / "num_steps.txt"
    num_steps = None
    if num_steps_txt.exists():
        try:
            num_steps = int(eval(num_steps_txt.read_text().strip()))
        except Exception:
            pass

    NA = NA.view(-1)
    BI = BI.view(-1)
    N = int(NA.shape[0])

    # Case A: (N, T, A, 3)
    if FC.dim() == 4:
        N_fc, T, Amax, _ = FC.shape
        assert N_fc == N, f"Mismatch: N from num_atoms={N}, but frac_coords has N={N_fc}"
        print(f"[info] frac_coords layout: (N={N}, T={T}, A≈{Amax}, 3)")
        for i in range(N):
            graph_index = int(BI[i]) if BI.numel() == N else i
            if graph_index not in want:
                continue
            na = int(NA[i])
            sample_dir = out_root / f"graph_{graph_index:06d}"
            sample_dir.mkdir(exist_ok=True)
            for t in range(T):
                fc = as_np(FC[i, t, :na, :]) % 1.0
                latt = pick_lattice_for(i, t, L)
                # Z block
                try:
                    if Z.dim() == 3 and Z.shape[0] == N and Z.shape[1] >= 1:
                        zi = as_np(Z[i, 0, :na]).astype(int)
                    else:
                        zi = slice_Z_for_sample(i, 0, na, t, Z)
                except Exception:
                    zi = np.ones((na,), dtype=int)
                zi = np.clip(zi, 1, 118).astype(int)
                els = [Element.from_Z(int(z)) for z in zi]
                st = Structure(Lattice(latt), els, fc, coords_are_cartesian=False)

                # Write CIF
                cif_path = sample_dir / f"frame_{t:04d}.cif"
                st.to(fmt="cif", filename=cif_path)

                # Bonds (optional)
                if args.save_bonds or args.save_graph_json:
                    same_image_only = not args.allow_pbc_links
                    if args.save_bonds:
                        bonds = compute_bonds(st, same_image_only=same_image_only)
                        write_bonds_csv(sample_dir / f"frame_{t:04d}_bonds.csv", bonds)
                    if args.save_graph_json:
                        write_structure_graph_json(sample_dir / f"frame_{t:04d}_graph.json", st)

            print(f"[exported] graph_index={graph_index} → {sample_dir} ({T} frames)")

    # Case B: (T, A_total, 3) time-first, atoms flattened
    elif FC.dim() == 3 and FC.shape[-1] == 3:
        T = FC.shape[0]
        if num_steps is not None and T != num_steps:
            print(f"[warn] T from tensor = {T} differs from num_steps.txt = {num_steps}")
        starts, na_list = split_starts(NA)
        Atotal = starts[-1]
        assert FC.shape[1] == Atotal, f"Expected A_total={Atotal}, but frac_coords has {FC.shape[1]}"
        print(f"[info] frac_coords layout: (T={T}, A_total={Atotal}, 3) with N={N}")

        for i in range(N):
            graph_index = int(BI[i]) if BI.numel() == N else i
            if graph_index not in want:
                continue
            a0, a1 = starts[i], starts[i + 1]
            na = a1 - a0
            sample_dir = out_root / f"graph_{graph_index:06d}"
            sample_dir.mkdir(exist_ok=True)
            for t in range(T):
                fc = as_np(FC[t, a0:a1, :]) % 1.0
                latt = pick_lattice_for(i, t, L)
                zi = slice_Z_for_sample(i, a0, a1, t, Z)
                zi = np.clip(zi, 1, 118).astype(int)
                els = [Element.from_Z(int(z)) for z in zi]
                st = Structure(Lattice(latt), els, fc, coords_are_cartesian=False)

                # Write CIF
                cif_path = sample_dir / f"frame_{t:04d}.cif"
                st.to(fmt="cif", filename=cif_path)

                # Bonds (optional)
                if args.save_bonds or args.save_graph_json:
                    same_image_only = not args.allow_pbc_links
                    if args.save_bonds:
                        bonds = compute_bonds(st, same_image_only=same_image_only)
                        write_bonds_csv(sample_dir / f"frame_{t:04d}_bonds.csv", bonds)
                    if args.save_graph_json:
                        write_structure_graph_json(sample_dir / f"frame_{t:04d}_graph.json", st)

            print(f"[exported] graph_index={graph_index} → {sample_dir} ({T} frames)")

    else:
        raise ValueError(f"Unsupported frac_coords shape {tuple(FC.shape)}")

if __name__ == "__main__":
    main()


# Reconstruct trajectories:
# python scripts_model/evaluate.py recon_trajectory /storage/alvand/orgflow/runs/trash/2025-10-15/05-26-58/null_params-rfm_cspnet-2325wzeb/rfmcsp-conditional-organic/1z12s8cy/checkpoints/epoch=1147-step=94136.ckpt  --stage val --batch_size 256 --num_evals 1 --limit_predict_batches "1." --num_steps 10 --single_gpu --subdir "" --inference_anneal_coords --no-inference_anneal_lattice

# Consolidate trajectories:
# python scripts_model/evaluate.py consolidate /storage/alvand/orgflow/runs/trash/2025-10-15/05-26-58/null_params-rfm_cspnet-2325wzeb/rfmcsp-conditional-organic/1z12s8cy/checkpoints/epoch=1147-step=94136.ckpt --task_to_save recon_trajectory

# Export top k predictions to cif file.
# python scripts_model/export_top_traj.py --traj_pt "/storage/alvand/orgflow/runs/trash/2025-10-15/05-26-58/null_params-rfm_cspnet-2325wzeb/rfmcsp-conditional-organic/1z12s8cy/checkpoints/consolidated_recon_trajectory.pt" --ranking_csv "/storage/alvand/orgflow/ranked_kabsch/ranking.csv" --out_dir "/storage/alvand/orgflow/top_traj_cifs" --topk 10