import os, argparse, csv
from typing import List, Tuple, Optional
import math
import numpy as np
import torch
from tqdm import tqdm

from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph

# Optional: image deps (handled with try/except)
try:
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.io import write as ase_write
    _HAVE_ASE = True
except Exception:
    _HAVE_ASE = False


# ---------- Kabsch helpers ----------

def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Return rotation R and translation t aligning P → Q (rows = points)."""
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Q.mean(axis=0) - P.mean(axis=0) @ R
    return R, t

def _rmsd_aligned(Pa: np.ndarray, Q: np.ndarray) -> float:
    """RMSD between already-aligned Pa and Q."""
    if len(Pa) == 0:
        return math.inf
    d = Pa - Q
    return float(np.sqrt((d * d).sum(axis=1).mean()))


# ---------- helpers ----------

def as_t(x):
    if isinstance(x, torch.Tensor): return x
    if isinstance(x, np.ndarray):   return torch.from_numpy(x)
    return torch.as_tensor(x)

def split_starts(num_atoms) -> Tuple[List[int], List[int]]:
    if hasattr(num_atoms, "tolist"): num_atoms = num_atoms.tolist()
    na = [int(n.item()) if isinstance(n, torch.Tensor) else int(n) for n in num_atoms]
    idx = [0]
    for n in na:
        idx.append(idx[-1] + n)
    return idx, na

def load_vocab_symbols() -> List[str]:
    try:
        from orgflow.model.bond_data import ATOMIC_SYMBOLS as VOCAB
        if isinstance(VOCAB, (list, tuple)) and len(VOCAB) > 0:
            return list(VOCAB)
    except Exception:
        pass
    return ["X", "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]

def symbols_to_Z_map(symbols: List[str]) -> List[int]:
    out = []
    for s in symbols:
        if s in ("X", "PAD", "PADDED", "MASK", "UNK", "NONE", ""):
            out.append(0)
        else:
            out.append(Element(s).Z)
    return out

def decode_atom_types_block(Z_block, vocab_Z: List[int]) -> torch.LongTensor:
    Zb = as_t(Z_block)
    if Zb.dim() == 2:  # [A, C] scores/logits
        idx = Zb.argmax(dim=-1).long()
        Z = torch.tensor([vocab_Z[int(i)] if 0 <= int(i) < len(vocab_Z) else 0 for i in idx], dtype=torch.long)
        return Z
    zb = Zb.long().view(-1)
    uniq = torch.unique(zb)
    if (uniq.max() > 18) or any(int(u) in (6, 7, 8, 16, 17, 35, 53, 14, 15) for u in uniq.tolist()):
        return zb
    Z = torch.tensor([vocab_Z[int(i)] if 0 <= int(i) < len(vocab_Z) else 0 for i in zb], dtype=torch.long)
    return Z

def build_structure_from_pred(Z_i, FC_i, L_i) -> Optional[Structure]:
    Z_i = as_t(Z_i).view(-1)
    FC_i = as_t(FC_i).view(-1, 3)
    L_i = as_t(L_i).view(3, 3)

    keep = Z_i > 0
    if keep.sum().item() == 0:
        return None
    Z_i = Z_i[keep]
    FC_i = FC_i[keep]

    try:
        species = [Element.from_Z(int(z)) for z in Z_i.tolist()]
    except Exception:
        return None

    lat = Lattice(L_i.detach().cpu().numpy())
    fc = (FC_i % 1.0).detach().cpu().numpy()
    try:
        return Structure(lat, species, fc, coords_are_cartesian=False)
    except Exception:
        return None

def refine(st: Structure, symprec: float) -> Tuple[Structure, int]:
    try:
        spa = SpacegroupAnalyzer(st, symprec=symprec)
        st_r = spa.get_refined_structure()
        sg = spa.get_space_group_number()
        return st_r, int(sg)
    except Exception:
        return st, 1

def batch_to_gt_list(batch_obj) -> List[Optional[Structure]]:
    if hasattr(batch_obj, "to_data_list"):
        data_list = batch_obj.to_data_list()
    else:
        data_list = list(batch_obj)
    out = []
    for d in data_list:
        try:
            Z = as_t(d.atom_types).view(-1)
            FC = as_t(d.frac_coords).view(-1, 3)
            lengths = as_t(d.lengths).view(-1)
            angles = as_t(d.angles).view(-1)
            lat = Lattice.from_parameters(*(lengths.tolist() + angles.tolist()))
            species = [Element.from_Z(int(z)) for z in Z.tolist()]
            fc = (FC % 1.0).detach().cpu().numpy()
            out.append(Structure(lat, species, fc, coords_are_cartesian=False))
        except Exception:
            out.append(None)
    return out

def get_eval0(cons: dict) -> dict:
    return {k: v[0] for k, v in cons.items()}


# ---------- image helpers ----------

def _extract_largest_molecule(structure: Structure) -> Optional["Molecule"]:
    try:
        cnn = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)
        sg = StructureGraph.with_local_env_strategy(structure, cnn)
        mols = sg.get_subgraphs_as_molecules()
        if not mols:
            return None
        mols = sorted(mols, key=lambda m: len(m), reverse=True)
        return mols[0]
    except Exception:
        return None

def _write_png_atoms(atoms, out_png: str, show_cell=False):
    if not _HAVE_ASE:
        return
    try:
        ase_write(out_png, atoms, show_unit_cell=show_cell, rotation='-90x,-90z')
    except Exception:
        pass

def write_images_for_structure(st_unrefined: Structure, out_prefix: str) -> None:
    if not _HAVE_ASE:
        return
    try:
        mol = _extract_largest_molecule(st_unrefined)
        if mol is not None:
            atoms = AseAtomsAdaptor.get_atoms(mol)
            _write_png_atoms(atoms, out_prefix + "_mol.png", show_cell=False)
            return
    except Exception:
        pass
    try:
        atoms = AseAtomsAdaptor.get_atoms(st_unrefined)
        _write_png_atoms(atoms, out_prefix + "_cell.png", show_cell=False)
    except Exception:
        pass


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to consolidated_reconstruct.pt")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--first_k", type=int, default=None)
    ap.add_argument("--symprec", type=float, default=0.01)
    # kept for compatibility; not used in ranking anymore:
    ap.add_argument("--stol", type=float, default=0.5)
    ap.add_argument("--angle_tol", type=float, default=10.0)
    ap.add_argument("--ltol", type=float, default=0.3)
    ap.add_argument("--no_images", action="store_true", help="Disable PNG rendering")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cons = torch.load(args.path, map_location="cpu")
    cons0 = get_eval0(cons)

    Z_all = cons0["atom_types"]
    FC_all = cons0["frac_coords"]  # [Atotal, 3]
    L_all = cons0["lattices"]      # [N, 3, 3]
    NA_all = cons0["num_atoms"]    # [N]
    batch = cons0.get("input_data_batch", None)

    gt_structs = batch_to_gt_list(batch) if batch is not None else None
    starts, na_list = split_starts(NA_all)

    vocab_symbols = load_vocab_symbols()
    vocab_Z = symbols_to_Z_map(vocab_symbols)

    Z_all_t = as_t(Z_all)
    FC_all_t = as_t(FC_all)
    L_all_t = as_t(L_all)

    # Determine how to slice Z
    if Z_all_t.dim() == 3:
        mode = "pad_scores"  # [N, Amax, C]
    elif Z_all_t.dim() == 2:
        N = len(na_list)
        mode = "pad" if Z_all_t.shape[0] == N else "flat_scores"  # [N, Amax] vs [Atotal, C]
    else:
        mode = "flat"  # [Atotal]

    # Collect rows
    rows = []  # list of dicts so we can sort & write consistently
    N_graphs = len(na_list) if args.first_k is None else min(len(na_list), args.first_k)
    pbar = tqdm(range(N_graphs), desc="rank")
    for i in pbar:
        a0, a1 = starts[i], starts[i + 1]
        na_i = na_list[i]

        if mode == "pad":
            Z_block = Z_all_t[i, :na_i]
        elif mode == "flat":
            Z_block = Z_all_t[a0:a1]
        elif mode == "pad_scores":
            Z_block = Z_all_t[i, :na_i, :]
        else:  # flat_scores
            Z_block = Z_all_t[a0:a1, :]

        # decode to atomic numbers
        Z_i = decode_atom_types_block(Z_block, vocab_Z=vocab_Z)
        FC_i = FC_all_t[a0:a1, :]
        L_i = L_all_t[i, ...]

        st_pred_unref = build_structure_from_pred(Z_i, FC_i, L_i)
        if st_pred_unref is None:
            rows.append({"idx": i, "rms": math.inf, "pred_unref": None, "pred_ref": None, "valid": False})
            continue

        # refine pred for fair matching/export
        st_pred_ref, _ = refine(st_pred_unref, args.symprec) if (args.symprec and args.symprec > 0) else (st_pred_unref, 1)

        if gt_structs is None or gt_structs[i] is None:
            rms = math.inf
            valid = True
        else:
            try:
                st_gt_ref, _ = refine(gt_structs[i], args.symprec) if (args.symprec and args.symprec > 0) else (gt_structs[i], 1)
                P = st_pred_ref.cart_coords
                Q = st_gt_ref.cart_coords
                m = min(len(P), len(Q))
                if m <= 0:
                    rms = math.inf
                    valid = False
                else:
                    R, t = _kabsch(P[:m], Q[:m])
                    Pa = P[:m] @ R + t
                    rms = _rmsd_aligned(Pa, Q[:m])
                    valid = True
            except Exception:
                rms = math.inf
                valid = False

        rows.append({"idx": i, "rms": rms, "pred_unref": st_pred_unref, "pred_ref": st_pred_ref, "valid": valid})

    # ---- SORT by Kabsch RMSD and write CSV in that order ----
    ranked = [r for r in rows if (r["valid"] and r["pred_ref"] is not None)]
    ranked.sort(key=lambda r: r["rms"])

    csv_path = os.path.join(args.out, "ranking.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "graph_index", "rms_kabsch"])
        for rank, r in enumerate(ranked, 1):
            rms_str = None if math.isinf(r["rms"]) else r["rms"]
            w.writerow([rank, r["idx"], rms_str])
    print(f"[info] wrote ranking (sorted by Kabsch RMSD): {csv_path}")

    if not ranked:
        print("[warn] No valid predictions to write.")
        return

    # ---- TOP-K from the same sorted list ----
    topk = ranked[:args.topk]

    for rank, r in enumerate(topk, 1):
        i = r["idx"]
        rms = r["rms"]
        st_pred_unref = r["pred_unref"]
        st_pred_ref   = r["pred_ref"]

        # write CIFs (refined)
        st_pred_r, sg = refine(st_pred_ref, args.symprec or 0.01)
        base = f"rank{rank:02d}_rms{(9999 if math.isinf(rms) else rms):.4f}_sg{sg:03d}"
        st_pred_r.to(fmt="cif", filename=os.path.join(args.out, base + "_pred.cif"))

        # also write GT (refined) if available
        # (we can reconstruct it from batch again to avoid storing in rows)
        if gt_structs is not None and gt_structs[i] is not None:
            st_gt_r, sg_gt = refine(gt_structs[i], args.symprec or 0.01)
            st_gt_r.to(fmt="cif", filename=os.path.join(args.out, base + "_gt.cif"))

        # images from unrefined pred (optional)
        # if not args.no_images and _HAVE_ASE and st_pred_unref is not None:
        #     try:
        #         write_images_for_structure(st_pred_unref, os.path.join(args.out, base + "_pred"))
        #         if gt_structs is not None and gt_structs[i] is not None:
        #             write_images_for_structure(gt_structs[i], os.path.join(args.out, base + "_gt"))
        #     except Exception:
        #         pass

    print(f"[done] wrote {len(topk)} CIFs (+ PNGs if ASE available) → {args.out}")


if __name__ == "__main__":
    main()


# python scripts_model/write_pred_cifs_refined.py --path "/steps_10/consolidated_reconstruct.pt"  --out ranked_kabsch --topk 10 --symprec 0.01