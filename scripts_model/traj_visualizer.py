#!/usr/bin/env python3
import argparse, glob, os, shutil, tempfile
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

# Plotly (for the interactive HTML)
import plotly.graph_objects as go

# Matplotlib (for GIF export without Chrome/Orca)
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D
import imageio.v2 as imageio

from pymatgen.core import Structure, Element, Lattice

# ---------- element colors & sizes ----------
ELEMENT_COLOR: Dict[str, str] = {
    # common organics
    "H": "#FFFFFF",  # white
    "C": "#909090",  # gray
    "N": "#2B6CB0",  # blue-ish
    "O": "#E53E3E",  # red
    "F": "#38B2AC",  # teal
    "P": "#D69E2E",  # gold
    "S": "#ECC94B",  # yellow
    "Cl": "#48BB78", # green
    "Br": "#9F7AEA", # purple
    "I": "#805AD5",  # deep purple
    "Si": "#ED8936", # orange
    # fallback default
}
DEFAULT_COLOR = "#A0AEC0"  # slate gray

def element_color(sym: str) -> str:
    return ELEMENT_COLOR.get(sym, DEFAULT_COLOR)

def element_size(sym: str) -> float:
    """Simple, readable sizes for scatter markers."""
    base = {
        "H": 4, "C": 6, "N": 6.5, "O": 7, "F": 6.5, "P": 7.5, "S": 7.5,
        "Cl": 7, "Br": 7.5, "I": 8, "Si": 7.5
    }
    return base.get(sym, 6.0)

# ---------- unit cell helper ----------
def unit_cell_edges(lattice: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    a, b, c = lattice
    O = np.zeros(3)
    A = a; B = b; C = c
    AB = a + b; AC = a + c; BC = b + c; ABC = a + b + c
    return [
        (O, A), (O, B), (O, C),
        (A, AB), (A, AC),
        (B, AB), (B, BC),
        (C, AC), (C, BC),
        (AB, ABC), (AC, ABC), (BC, ABC),
    ]

# ---------- Plotly HTML (interactive) ----------
def make_plotly_html(frames_cif: List[str], out_html_path: Path) -> None:
    first = Structure.from_file(frames_cif[0])
    P0 = first.cart_coords
    syms0 = [str(sp) for sp in first.species]

    # one scatter per element for colored legend
    data0 = []
    for sym in sorted(set(syms0)):
        mask = [s == sym for s in syms0]
        pts = P0[mask]
        if len(pts) == 0: continue
        data0.append(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode="markers",
            name=sym,
            marker=dict(
                size=element_size(sym),
                color=element_color(sym),
                opacity=0.95,
                line=dict(width=0.5, color="black"),
            ),
            hovertext=[sym]*len(pts),
            hoverinfo="text",
            showlegend=True,
        ))

    # unit cell (from first frame)
    edges = unit_cell_edges(first.lattice.matrix)
    for s, e in edges:
        data0.append(go.Scatter3d(
            x=[s[0], e[0]], y=[s[1], e[1]], z=[s[2], e[2]],
            mode="lines",
            line=dict(color="#666666", width=3),
            name="Unit cell",
            showlegend=False,
            hoverinfo="skip",
            opacity=0.35,
        ))

    # frames
    frs = []
    for f in frames_cif:
        st = Structure.from_file(f)
        P = st.cart_coords
        syms = [str(sp) for sp in st.species]
        traces = []
        for sym in sorted(set(syms)):
            mask = [s == sym for s in syms]
            pts = P[mask]
            if len(pts) == 0: continue
            traces.append(go.Scatter3d(
                x=pts[:,0], y=pts[:,1], z=pts[:,2],
                mode="markers",
                marker=dict(
                    size=element_size(sym),
                    color=element_color(sym),
                    opacity=0.95,
                    line=dict(width=0.5, color="black"),
                ),
                name=sym,
                hovertext=[sym]*len(pts),
                hoverinfo="text",
                showlegend=False,  # avoid legend spam while animating
            ))
        # unit cell per frame (keeps camera/scale consistent if lattice changes)
        edges = unit_cell_edges(st.lattice.matrix)
        for s, e in edges:
            traces.append(go.Scatter3d(
                x=[s[0], e[0]], y=[s[1], e[1]], z=[s[2], e[2]],
                mode="lines",
                line=dict(color="#666666", width=3),
                name="Unit cell",
                showlegend=False,
                hoverinfo="skip",
                opacity=0.35,
            ))
        frs.append(go.Frame(data=traces, name=os.path.basename(f)))

    # bounds for nice aspect
    all_pts = np.vstack([Structure.from_file(f).cart_coords for f in frames_cif])
    mins = all_pts.min(axis=0); maxs = all_pts.max(axis=0)
    span = (maxs - mins).max() * 0.7
    center = (maxs + mins) / 2
    xlim = (center[0]-span, center[0]+span)
    ylim = (center[1]-span, center[1]+span)
    zlim = (center[2]-span, center[2]+span)

    fig = go.Figure(
        data=data0,
        frames=frs,
        layout=go.Layout(
            title=os.path.basename(out_html_path).replace(".html",""),
            scene=dict(
                xaxis=dict(
                    visible=False,
                    range=xlim,
                    showbackground=False,
                    showgrid=False,
                    zeroline=False,
                ),
                yaxis=dict(
                    visible=False,
                    range=ylim,
                    showbackground=False,
                    showgrid=False,
                    zeroline=False,
                ),
                zaxis=dict(
                    visible=False,
                    range=zlim,
                    showbackground=False,
                    showgrid=False,
                    zeroline=False,
                ),
                aspectmode="cube",
                bgcolor="white",
                camera=dict(
                    projection=dict(type="perspective"),
                    eye=dict(x=1.8, y=1.8, z=1.6),
                ),
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.6)",
                x=0.01, y=0.99, xanchor="left", yanchor="top"
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            updatemenus=[{
                "type":"buttons",
                "buttons":[
                    {"label":"Play","method":"animate","args":[None,{"frame":{"duration":120,"redraw":True},"fromcurrent":True}]},
                    {"label":"Pause","method":"animate","args":[[None],{"frame":{"duration":0},"mode":"immediate"}]}
                ]
            }],
        )
    )
    fig.write_html(str(out_html_path), include_plotlyjs="cdn")

# ---------- Matplotlib GIF (no Chrome required) ----------
def render_frame_png(st: Structure, png_path: Path, width=800, height=800):
    P = st.cart_coords
    syms = [str(sp) for sp in st.species]

    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # scatter per element
    for sym in sorted(set(syms)):
        mask = [s == sym for s in syms]
        pts = P[mask]
        if len(pts) == 0: continue
        ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                   s=element_size(sym)*6,  # tune size for visibility in GIF
                   c=element_color(sym),
                   depthshade=True, edgecolors="none", linewidths=0, alpha=0.95, label=sym)

    # unit cell
    A = st.lattice.matrix
    for s, e in unit_cell_edges(A):
        ax.plot([s[0], e[0]],[s[1], e[1]],[s[2], e[2]], color="#666666", linewidth=1.5, alpha=0.5)

    # bounds & aspect
    all_pts = P
    mins = all_pts.min(axis=0); maxs = all_pts.max(axis=0)
    span = (maxs - mins).max() * 0.7
    center = (maxs + mins) / 2
    ax.set_xlim(center[0]-span, center[0]+span)
    ax.set_ylim(center[1]-span, center[1]+span)
    ax.set_zlim(center[2]-span, center[2]+span)

    # clean look
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.grid(False)
    # consistent camera angle
    ax.view_init(elev=18, azim=45)

    # legend (small, on white)
    ax.legend(loc="upper left", framealpha=0.8, fontsize=8)

    plt.tight_layout(pad=0.1)
    fig.savefig(png_path, dpi=100, facecolor="white", edgecolor="white")
    plt.close(fig)

def make_gif_from_cifs(frames_cif: List[str], out_gif_path: Path, fps=20, width=600, height=600):
    tmp_dir = Path(tempfile.mkdtemp(prefix="traj_gif_"))
    pngs = []
    try:
        for t, f in enumerate(frames_cif):
            st = Structure.from_file(f)
            png_path = tmp_dir / f"frame_{t:04d}.png"
            render_frame_png(st, png_path, width=width, height=height)
            pngs.append(imageio.imread(png_path))
        imageio.mimsave(out_gif_path, pngs, fps=fps, palettesize=256)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_dir", required=True,
                    help="PARENT folder containing graph_* subfolders (repurposed).")
    ap.add_argument("--out_html", required=True,
                    help="Output directory to write one HTML and one GIF per graph (repurposed).")
    args = ap.parse_args()

    parent = Path(args.sample_dir).resolve()
    out_dir = Path(args.out_html).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_dirs = sorted([p for p in parent.glob("graph_*") if p.is_dir()])
    if not graph_dirs:
        raise SystemExit(f"No graph_* subfolders found in {parent}")

    for gdir in graph_dirs:
        frames = sorted(gdir.glob("frame_*.cif"))
        if not frames:
            print(f"[skip] {gdir.name}: no frame_*.cif")
            continue

        html_path = out_dir / f"{gdir.name}.html"
        gif_path  = out_dir / f"{gdir.name}.gif"

        try:
            print(f"[html] {gdir.name} → {html_path.name}")
            make_plotly_html([str(f) for f in frames], html_path)
        except Exception as e:
            print(f"[warn] HTML failed for {gdir.name}: {e}")

        try:
            print(f"[gif ] {gdir.name} → {gif_path.name}")
            make_gif_from_cifs([str(f) for f in frames], gif_path, fps=4, width=600, height=600)
        except Exception as e:
            print(f"[warn] GIF failed for {gdir.name}: {e}")

    print(f"[done] Wrote outputs to: {out_dir}")

if __name__ == "__main__":
    main()


# python scripts_model/traj_visualizer.py --sample_dir /storage/alvand/flowmm/top_traj_cifs/ --out_html trajs_htmls/