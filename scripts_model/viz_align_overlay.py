#!/usr/bin/env python3
import os, glob, argparse
import numpy as np
from typing import List, Tuple, Optional, Dict, Set

import plotly.graph_objects as go
from pymatgen.core import Structure, Element
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# ----------------- shared helpers -----------------

def refine(st: Structure, symprec: float) -> Structure:
    """Refine with pymatgen’s space group analyzer (same step as ranking)."""
    try:
        spa = SpacegroupAnalyzer(st, symprec=symprec)
        return spa.get_refined_structure()
    except Exception:
        return st


def kabsch(P: np.ndarray, Q: np.ndarray):
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


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD between two same-length point sets."""
    d = P - Q
    return float(np.sqrt((d * d).sum(axis=1).mean()))


def covalent_size(z: int) -> float:
    try:
        r = Element.from_Z(int(z)).average_covalent_radius or 0.77
    except Exception:
        r = 0.77
    return 8.0 + 10.0 * (r / 0.77)


def unit_cell_edges(lattice_matrix: np.ndarray):
    a, b, c = lattice_matrix
    O = np.zeros(3)
    A = a; B = b; C = c
    AB = a + b; AC = a + c; BC = b + c; ABC = a + b + c
    edges = [
        (O, A), (O, B), (O, C),
        (A, AB), (A, AC),
        (B, AB), (B, BC),
        (C, AC), (C, BC),
        (AB, ABC), (AC, ABC), (BC, ABC),
    ]
    return edges


# ----------------- component-aware bonds (for nicer visuals only) -----------------

def get_components_by_graph(struct: Structure, same_image_only: bool = True) -> List[List[int]]:
    """
    Connected components via CrystalNN. Used only to decide which bonds to draw.
    """
    try:
        # NOTE: with_local_env_strategy is deprecated in 2025; switch when you update pymatgen
        cnn = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)
        sg = StructureGraph.with_local_env_strategy(struct, cnn)
        adj: Dict[int, Set[int]] = {i: set() for i in range(len(struct))}
        for u, v, data in sg.graph.edges(data=True):
            tj = data.get("to_jimage", (0, 0, 0))
            if same_image_only and tuple(tj) != (0, 0, 0):
                continue
            if u == v:
                continue
            ui, vi = int(u), int(v)
            adj[ui].add(vi)
            adj[vi].add(ui)
        seen = set()
        comps = []
        for i in range(len(struct)):
            if i in seen: continue
            q = [i]; seen.add(i); cur = [i]
            while q:
                x = q.pop()
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y); q.append(y); cur.append(y)
            comps.append(sorted(cur))
        comps.sort(key=len, reverse=True)
        return comps
    except Exception:
        return [[i for i in range(len(struct))]]


def _component_traces(
    coords: np.ndarray,
    Z: np.ndarray,
    comp: List[int],
    name_prefix: str,
    atom_color: str,
    bond_color: str,
    opacity: float,
) -> List[go.Scatter3d]:
    comp = list(comp)
    if not comp:
        return []
    sizes = [covalent_size(int(Z[i])) for i in comp]
    atoms = go.Scatter3d(
        x=coords[comp, 0], y=coords[comp, 1], z=coords[comp, 2],
        mode="markers",
        name=f"{name_prefix} atoms ({len(comp)})",
        marker=dict(size=sizes, color=atom_color, opacity=opacity, line=dict(width=1, color="black")),
        hovertext=[str(Element.from_Z(int(Z[i]))) for i in comp],
        hoverinfo="text",
        showlegend=True,
    )
    traces = [atoms]
    # naive proximity bonds within component (for visuals only)
    radii = []
    for i in comp:
        try:
            r = Element.from_Z(int(Z[i])).average_covalent_radius or 0.75
        except Exception:
            r = 0.75
        radii.append(r)
    for a in range(len(comp)):
        i = comp[a]
        for b in range(a + 1, len(comp)):
            j = comp[b]
            d = np.linalg.norm(coords[i] - coords[j])
            if d < 1.2 * (radii[a] + radii[b]):
                traces.append(go.Scatter3d(
                    x=[coords[i, 0], coords[j, 0]],
                    y=[coords[i, 1], coords[j, 1]],
                    z=[coords[i, 2], coords[j, 2]],
                    mode="lines",
                    line=dict(color=bond_color, width=5),
                    name=f"{name_prefix} bonds",
                    showlegend=False,
                    hoverinfo="skip",
                    opacity=opacity,
                ))
    return traces


# ----------------- main figure builder -----------------

def make_overlay_figure(
    pred_in: Structure,
    gt_in: Structure,
    title_prefix: str = "",
    components_mode: str = "largest",  # 'largest' | 'all' | 'none' (for bonds only)
    same_image_only: bool = True,
    show_unit_cell: bool = True,
    symprec: float = 0.01,            # NEW: refine like ranking
) -> go.Figure:
    """
    Align + RMSD exactly like ranking (Kabsch on refined coords, first m atoms).
    Draw bonds per chosen components for visuals, but *RMSD is independent of that*.
    """
    # 1) REFINE (match ranking)
    pred = refine(pred_in, symprec)
    gt   = refine(gt_in,   symprec)

    # 2) Kabsch using ALL refined atoms, RMSD on the SAME subset (first m)
    P_all = pred.cart_coords
    Q_all = gt.cart_coords
    nP, nQ = len(P_all), len(Q_all)
    m = min(nP, nQ)
    if m == 0:
        # nothing to align; show raw
        P_align_all = P_all
        Q_align_all = Q_all
        rt = float("inf")
    else:
        R, t = kabsch(P_all[:m], Q_all[:m])
        P_align_all = P_all @ R + t
        Q_align_all = Q_all
        rt = rmsd(P_align_all[:m], Q_align_all[:m])

    # species arrays for marker sizing / hover
    Z_gt = np.array([sp.Z for sp in gt.species], dtype=int)
    Z_pr = np.array([sp.Z for sp in pred.species], dtype=int)

    # 3) Build Plotly traces
    traces = []
    if components_mode == "none":
        # atoms only (no bonds)
        sizes_gt = [covalent_size(int(z)) for z in Z_gt]
        sizes_pr = [covalent_size(int(z)) for z in Z_pr]
        traces += [
            go.Scatter3d(
                x=Q_align_all[:, 0], y=Q_align_all[:, 1], z=Q_align_all[:, 2],
                mode="markers",
                name="GT atoms",
                marker=dict(size=sizes_gt, color="#aaaaaa", opacity=0.95, line=dict(width=1, color="black")),
                hovertext=[str(sp) for sp in gt.species],
                hoverinfo="text",
            ),
            go.Scatter3d(
                x=P_align_all[:, 0], y=P_align_all[:, 1], z=P_align_all[:, 2],
                mode="markers",
                name="Pred atoms",
                marker=dict(size=sizes_pr, color="#2ecc71", opacity=0.70, line=dict(width=1, color="black")),
                hovertext=[str(sp) for sp in pred.species],
                hoverinfo="text",
            ),
        ]
    else:
        # compute components JUST to decide which bonds to draw
        comps_gt = get_components_by_graph(gt, same_image_only=same_image_only)
        comps_pr = get_components_by_graph(pred, same_image_only=same_image_only)
        draw_gt = comps_gt if components_mode == "all" else comps_gt[:1]
        draw_pr = comps_pr if components_mode == "all" else comps_pr[:1]

        # GT (gray)
        for k, comp in enumerate(draw_gt, 1):
            traces += _component_traces(
                coords=Q_align_all, Z=Z_gt, comp=comp,
                name_prefix=f"GT[{k}]",
                atom_color="#aaaaaa", bond_color="#888888", opacity=0.95
            )
        # Pred (green)
        for k, comp in enumerate(draw_pr, 1):
            traces += _component_traces(
                coords=P_align_all, Z=Z_pr, comp=comp,
                name_prefix=f"Pred[{k}]",
                atom_color="#2ecc71", bond_color="#1f9e54", opacity=0.75
            )

    # 4) Optional unit cell (from refined GT)
    if show_unit_cell:
        for s, e in unit_cell_edges(gt.lattice.matrix):
            traces.append(go.Scatter3d(
                x=[s[0], e[0]], y=[s[1], e[1]], z=[s[2], e[2]],
                mode="lines",
                line=dict(color="#666666", width=3),
                name="Unit cell",
                showlegend=False,
                hoverinfo="skip",
                opacity=0.35,
            ))

    # 5) layout with perspective + consistent ranges
    all_pts = np.vstack([P_align_all, Q_align_all]) if len(P_align_all) and len(Q_align_all) else (
        P_align_all if len(P_align_all) else Q_align_all
    )
    if len(all_pts) == 0:
        # fallback to unit cube
        all_pts = np.array([[0,0,0],[1,1,1]], dtype=float)
    mins = all_pts.min(axis=0); maxs = all_pts.max(axis=0)
    span = (maxs - mins).max() * 0.7
    center = (maxs + mins) / 2
    xlim = (center[0]-span, center[0]+span)
    ylim = (center[1]-span, center[1]+span)
    zlim = (center[2]-span, center[2]+span)

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=f"{title_prefix}  |  Kabsch RMSD ≈ {rt:.4f} Å (refined, first m atoms)",
            scene=dict(
                xaxis=dict(visible=False, range=xlim),
                yaxis=dict(visible=False, range=ylim),
                zaxis=dict(visible=False, range=zlim),
                aspectmode="cube",
                camera=dict(
                    projection=dict(type="perspective"),
                    eye=dict(x=1.8, y=1.8, z=1.6),
                ),
            ),
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.6)"
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )
    )
    return fig


# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Dir with *_pred.cif & *_gt.cif pairs")
    ap.add_argument("--suffix", default="_overlay3d.html", help="Output HTML suffix")
    ap.add_argument("--first_k", type=int, default=None)
    ap.add_argument("--components", choices=["largest", "all", "none"], default="largest",
                    help="Which connected components to draw bonds for (RMSD unaffected)")
    ap.add_argument("--no_cell", action="store_true", help="Hide unit cell wireframe")
    ap.add_argument("--allow_pbc_links", action="store_true",
                    help="(Only affects bond drawing) allow bonds across to_jimage≠0")
    ap.add_argument("--symprec", type=float, default=0.01, help="Refinement tolerance; must match ranking")
    args = ap.parse_args()

    pairs = {}
    for p in glob.glob(os.path.join(args.in_dir, "*_pred.cif")):
        base = p[:-9]
        g = base + "_gt.cif"
        if os.path.exists(g):
            pairs[base] = (p, g)

    bases = sorted(pairs.keys())
    if args.first_k is not None:
        bases = bases[:args.first_k]

    for base in bases:
        p, g = pairs[base]
        try:
            sp = Structure.from_file(p)
            sg = Structure.from_file(g)
            title = os.path.basename(base)
            fig = make_overlay_figure(
                sp, sg, title_prefix=title,
                components_mode=args.components,
                same_image_only=not args.allow_pbc_links,
                show_unit_cell=not args.no_cell,
                symprec=args.symprec,
            )
            out = base + args.suffix
            fig.write_html(out, include_plotlyjs="cdn", full_html=True)
            print(f"[ok] {os.path.basename(out)}")
        except Exception as e:
            print(f"[skip] {os.path.basename(base)} ({e})")


if __name__ == "__main__":
    main()




# Each Element with its own color code

# import os, glob, argparse
# import numpy as np
# from typing import List, Tuple, Optional, Dict, Set
#
# import plotly.graph_objects as go
# from pymatgen.core import Structure, Element
# from pymatgen.analysis import local_env
# from pymatgen.analysis.graphs import StructureGraph
# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
#
#
# # ----------------- shared helpers -----------------
#
# def refine(st: Structure, symprec: float) -> Structure:
#     try:
#         spa = SpacegroupAnalyzer(st, symprec=symprec)
#         return spa.get_refined_structure()
#     except Exception:
#         return st
#
#
# def kabsch(P: np.ndarray, Q: np.ndarray):
#     Pc = P - P.mean(axis=0, keepdims=True)
#     Qc = Q - Q.mean(axis=0, keepdims=True)
#     H = Pc.T @ Qc
#     U, S, Vt = np.linalg.svd(H)
#     R = Vt.T @ U.T
#     if np.linalg.det(R) < 0:
#         Vt[-1, :] *= -1
#         R = Vt.T @ U.T
#     t = Q.mean(axis=0) - P.mean(axis=0) @ R
#     return R, t
#
#
# def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
#     d = P - Q
#     return float(np.sqrt((d * d).sum(axis=1).mean()))
#
#
# def covalent_size(z: int) -> float:
#     try:
#         r = Element.from_Z(int(z)).average_covalent_radius or 0.77
#     except Exception:
#         r = 0.77
#     return 8.0 + 10.0 * (r / 0.77)
#
#
# def unit_cell_edges(lattice_matrix: np.ndarray):
#     a, b, c = lattice_matrix
#     O = np.zeros(3)
#     A = a; B = b; C = c
#     AB = a + b; AC = a + c; BC = b + c; ABC = a + b + c
#     edges = [
#         (O, A), (O, B), (O, C),
#         (A, AB), (A, AC),
#         (B, AB), (B, BC),
#         (C, AC), (C, BC),
#         (AB, ABC), (AC, ABC), (BC, ABC),
#     ]
#     return edges
#
#
# # ----------------- element → color (atoms only) -----------------
#
# # You can extend this map anytime. Unknown elements fall back to neutral gray.
# ELEMENT_COLORS: Dict[str, str] = {
#     "H": "#FFFFFF",  # white (outline stays black so it is visible)
#     "C": "#000000",  # black
#     "N": "#0000FF",  # blue
#     "O": "#FF0000",  # red
#     "F": "#00FF00",  # green
#     "Cl": "#228B22", # dark green
#     "Br": "#8B0000", # dark red
#     "I": "#9400D3",  # violet
#     "S": "#FFD700",  # golden yellow
#     "P": "#FFA500",  # orange
#     "Na": "#4169E1", # royal blue
#     "K": "#9932CC",  # dark orchid
#     "Mg": "#32CD32", # lime green
#     "Ca": "#7FFFD4", # aquamarine
#     "Si": "#DAA520", # goldenrod
#     "B": "#FF1493",  # deep pink
# }
#
# def element_color_list(Z: np.ndarray) -> List[str]:
#     colors = []
#     for z in Z:
#         try:
#             sym = Element.from_Z(int(z)).symbol
#         except Exception:
#             sym = None
#         colors.append(ELEMENT_COLORS.get(sym, "#808080"))  # default neutral gray
#     return colors
#
#
# # ----------------- component-aware bonds (for visuals only) -----------------
#
# def get_components_by_graph(struct: Structure, same_image_only: bool = True) -> List[List[int]]:
#     try:
#         cnn = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)
#         sg = StructureGraph.with_local_env_strategy(struct, cnn)
#         adj: Dict[int, Set[int]] = {i: set() for i in range(len(struct))}
#         for u, v, data in sg.graph.edges(data=True):
#             tj = data.get("to_jimage", (0, 0, 0))
#             if same_image_only and tuple(tj) != (0, 0, 0):
#                 continue
#             if u == v:
#                 continue
#             ui, vi = int(u), int(v)
#             adj[ui].add(vi)
#             adj[vi].add(ui)
#         seen = set()
#         comps = []
#         for i in range(len(struct)):
#             if i in seen: continue
#             q = [i]; seen.add(i); cur = [i]
#             while q:
#                 x = q.pop()
#                 for y in adj[x]:
#                     if y not in seen:
#                         seen.add(y); q.append(y); cur.append(y)
#             comps.append(sorted(cur))
#         comps.sort(key=len, reverse=True)
#         return comps
#     except Exception:
#         return [[i for i in range(len(struct))]]
#
#
# def _component_traces(
#     coords: np.ndarray,
#     Z: np.ndarray,
#     comp: List[int],
#     name_prefix: str,
#     bond_color: str,          # bonds keep your requested colors
#     opacity: float,
# ) -> List[go.Scatter3d]:
#     comp = list(comp)
#     if not comp:
#         return []
#     sizes = [covalent_size(int(Z[i])) for i in comp]
#     atom_colors_full = element_color_list(Z)
#     atom_colors = [atom_colors_full[i] for i in comp]
#
#     atoms = go.Scatter3d(
#         x=coords[comp, 0], y=coords[comp, 1], z=coords[comp, 2],
#         mode="markers",
#         name=f"{name_prefix} atoms ({len(comp)})",
#         marker=dict(size=sizes, color=atom_colors, opacity=opacity, line=dict(width=1, color="black")),
#         hovertext=[str(Element.from_Z(int(Z[i]))) for i in comp],
#         hoverinfo="text",
#         showlegend=True,
#     )
#     traces = [atoms]
#
#     # naive proximity bonds within component (for visuals only)
#     radii = []
#     for i in comp:
#         try:
#             r = Element.from_Z(int(Z[i])).average_covalent_radius or 0.75
#         except Exception:
#             r = 0.75
#         radii.append(r)
#     for a in range(len(comp)):
#         i = comp[a]
#         for b in range(a + 1, len(comp)):
#             j = comp[b]
#             d = np.linalg.norm(coords[i] - coords[j])
#             if d < 1.2 * (radii[a] + radii[b]):
#                 traces.append(go.Scatter3d(
#                     x=[coords[i, 0], coords[j, 0]],
#                     y=[coords[i, 1], coords[j, 1]],
#                     z=[coords[i, 2], coords[j, 2]],
#                     mode="lines",
#                     line=dict(color=bond_color, width=5),
#                     name=f"{name_prefix} bonds",
#                     showlegend=False,
#                     hoverinfo="skip",
#                     opacity=opacity,
#                 ))
#     return traces
#
#
# # ----------------- main figure builder -----------------
#
# def make_overlay_figure(
#     pred_in: Structure,
#     gt_in: Structure,
#     title_prefix: str = "",
#     components_mode: str = "largest",  # 'largest' | 'all' | 'none'
#     same_image_only: bool = True,
#     show_unit_cell: bool = True,
#     symprec: float = 0.01,
# ) -> go.Figure:
#     # 1) Refine
#     pred = refine(pred_in, symprec)
#     gt   = refine(gt_in,   symprec)
#
#     # 2) Kabsch
#     P_all = pred.cart_coords
#     Q_all = gt.cart_coords
#     nP, nQ = len(P_all), len(Q_all)
#     m = min(nP, nQ)
#     if m == 0:
#         P_align_all = P_all
#         Q_align_all = Q_all
#         rt = float("inf")
#     else:
#         R, t = kabsch(P_all[:m], Q_all[:m])
#         P_align_all = P_all @ R + t
#         Q_align_all = Q_all
#         rt = rmsd(P_align_all[:m], Q_align_all[:m])
#
#     Z_gt = np.array([sp.Z for sp in gt.species], dtype=int)
#     Z_pr = np.array([sp.Z for sp in pred.species], dtype=int)
#
#     traces = []
#     if components_mode == "none":
#         # atoms only, per-element colors
#         sizes_gt = [covalent_size(int(z)) for z in Z_gt]
#         sizes_pr = [covalent_size(int(z)) for z in Z_pr]
#         colors_gt = element_color_list(Z_gt)
#         colors_pr = element_color_list(Z_pr)
#         traces += [
#             go.Scatter3d(
#                 x=Q_align_all[:, 0], y=Q_align_all[:, 1], z=Q_align_all[:, 2],
#                 mode="markers",
#                 name="GT atoms",
#                 marker=dict(size=sizes_gt, color=colors_gt, opacity=0.95, line=dict(width=1, color="black")),
#                 hovertext=[str(sp) for sp in gt.species],
#                 hoverinfo="text",
#             ),
#             go.Scatter3d(
#                 x=P_align_all[:, 0], y=P_align_all[:, 1], z=P_align_all[:, 2],
#                 mode="markers",
#                 name="Pred atoms",
#                 marker=dict(size=sizes_pr, color=colors_pr, opacity=0.70, line=dict(width=1, color="black")),
#                 hovertext=[str(sp) for sp in pred.species],
#                 hoverinfo="text",
#             ),
#         ]
#     else:
#         # components only influence where bonds get drawn; atom colors are per-element
#         comps_gt = get_components_by_graph(gt, same_image_only=same_image_only)
#         comps_pr = get_components_by_graph(pred, same_image_only=same_image_only)
#         draw_gt = comps_gt if components_mode == "all" else comps_gt[:1]
#         draw_pr = comps_pr if components_mode == "all" else comps_pr[:1]
#
#         # GT bonds = green, Pred bonds = gray  (kept as you requested)
#         GT_BOND = "#2ecc71"
#         PR_BOND = "#888888"
#
#         for k, comp in enumerate(draw_gt, 1):
#             traces += _component_traces(
#                 coords=Q_align_all, Z=Z_gt, comp=comp,
#                 name_prefix=f"GT[{k}]",
#                 bond_color=GT_BOND, opacity=0.95
#             )
#         for k, comp in enumerate(draw_pr, 1):
#             traces += _component_traces(
#                 coords=P_align_all, Z=Z_pr, comp=comp,
#                 name_prefix=f"Pred[{k}]",
#                 bond_color=PR_BOND, opacity=0.75
#             )
#
#     if show_unit_cell:
#         for s, e in unit_cell_edges(gt.lattice.matrix):
#             traces.append(go.Scatter3d(
#                 x=[s[0], e[0]], y=[s[1], e[1]], z=[s[2], e[2]],
#                 mode="lines",
#                 line=dict(color="#666666", width=3),
#                 name="Unit cell",
#                 showlegend=False,
#                 hoverinfo="skip",
#                 opacity=0.35,
#             ))
#
#     all_pts = np.vstack([P_align_all, Q_align_all]) if len(P_align_all) and len(Q_align_all) else (
#         P_align_all if len(P_align_all) else Q_align_all
#     )
#     if len(all_pts) == 0:
#         all_pts = np.array([[0,0,0],[1,1,1]], dtype=float)
#     mins = all_pts.min(axis=0); maxs = all_pts.max(axis=0)
#     span = (maxs - mins).max() * 0.7
#     center = (maxs + mins) / 2
#     xlim = (center[0]-span, center[0]+span)
#     ylim = (center[1]-span, center[1]+span)
#     zlim = (center[2]-span, center[2]+span)
#
#     fig = go.Figure(
#         data=traces,
#         layout=go.Layout(
#             title=f"{title_prefix}  |  Kabsch RMSD ≈ {rt:.4f} Å (refined, first m atoms)",
#             scene=dict(
#                 xaxis=dict(visible=False, range=xlim),
#                 yaxis=dict(visible=False, range=ylim),
#                 zaxis=dict(visible=False, range=zlim),
#                 aspectmode="cube",
#                 camera=dict(
#                     projection=dict(type="perspective"),
#                     eye=dict(x=1.8, y=1.8, z=1.6),
#                 ),
#             ),
#             legend=dict(
#                 yanchor="top", y=0.99, xanchor="left", x=0.01,
#                 bgcolor="rgba(255,255,255,0.6)"
#             ),
#             margin=dict(l=0, r=0, t=40, b=0),
#         )
#     )
#     return fig
#
#
# # ----------------- CLI -----------------
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--in_dir", required=True, help="Dir with *_pred.cif & *_gt.cif pairs")
#     ap.add_argument("--suffix", default="_overlay3d.html", help="Output HTML suffix")
#     ap.add_argument("--first_k", type=int, default=None)
#     ap.add_argument("--components", choices=["largest", "all", "none"], default="largest",
#                     help="Which connected components to draw bonds for (RMSD unaffected)")
#     ap.add_argument("--no_cell", action="store_true", help="Hide unit cell wireframe")
#     ap.add_argument("--allow_pbc_links", action="store_true",
#                     help="(Only affects bond drawing) allow bonds across to_jimage≠0")
#     ap.add_argument("--symprec", type=float, default=0.01, help="Refinement tolerance; must match ranking")
#     args = ap.parse_args()
#
#     pairs = {}
#     for p in glob.glob(os.path.join(args.in_dir, "*_pred.cif")):
#         base = p[:-9]
#         g = base + "_gt.cif"
#         if os.path.exists(g):
#             pairs[base] = (p, g)
#
#     bases = sorted(pairs.keys())
#     if args.first_k is not None:
#         bases = bases[:args.first_k]
#
#     for base in bases:
#         p, g = pairs[base]
#         try:
#             sp = Structure.from_file(p)
#             sg = Structure.from_file(g)
#             title = os.path.basename(base)
#             fig = make_overlay_figure(
#                 sp, sg, title_prefix=title,
#                 components_mode=args.components,
#                 same_image_only=not args.allow_pbc_links,
#                 show_unit_cell=not args.no_cell,
#                 symprec=args.symprec,
#             )
#             out = base + args.suffix
#             fig.write_html(out, include_plotlyjs="cdn", full_html=True)
#             print(f"[ok] {os.path.basename(out)}")
#         except Exception as e:
#             print(f"[skip] {os.path.basename(base)} ({e})")
#
#
# if __name__ == "__main__":
#     main()

# python scripts_model/viz_align_overlay.py --in_dir ranked_kabsch/ --components largest