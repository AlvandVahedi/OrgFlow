# OrgFlow: Generative Modeling of Organic Crystal Structures from Molecular Graphs

OrgFlow is a conditional flow-matching framework purpose-built for **organic** crystal structure prediction. It learns a deterministic transport from a Gaussian prior to periodic crystal structures, explicitly conditioned on the asymmetric-unit molecular graph. Key ingredients:
- **Periodic molecular graphs** with integer image shifts and curated bond orders.
- **E(3)-equivariant message passing** over periodic edges for fractional coordinates and lattice velocities.
- **Bond-aware Student‑t regularization** from empirical bond-length statistics to enforce chemically realistic local geometry.
- **Fast sampling**: near-linear transport converges in ~20 ODE steps.
- **Organic-only benchmark** (177k CSD-derived structures) with molecule-disjoint splits and full preprocessing scripts.

> This repository contains the codebase for the OrgFlow paper “Generative Modeling of Organic Crystal Structures from Molecular Graphs” (Vahediahmar, McDonald, Liu, Drexel University).The model and data pipelines are tailored to organic crystals and bond-aware conditioning.

## What’s in this repo
- `scripts_model/`: training and evaluation entry points.
- `src/orgflow/`: core OrgFlow model, periodic graph builder, bond-aware loss, and data utils.
- `remote/`: third-party components kept as submodules/forks (cdvae, DiffCSP, riemannian-fm); the main OrgFlow code lives under `src/orgflow/`.
- `scripts_analysis/`: evaluation, visualization, and DFT/pre-relax utilities.

## Installation (recommended)
```bash
git submodule update --init --recursive
micromamba env create -f environment.yml
micromamba activate orgflow   # env name retained for compatibility
```

## Data (organic CSD-derived)
- Raw CIFs are not redistributed due to CSD licensing. Use a licensed CSD installation or WebCSD access.
- Run the provided preprocessing scripts to build periodic molecular graphs, bond statistics, and dataset splits (drug-like, non-drug-like, small, smallest). The pipeline:
  1) Parse/refine CIFs (Niggli/space-group consistency).
  2) Validate SMILES and atom types; ensure single connected molecule; drop unreliable H atoms (can be reconstructed later).
  3) Lift molecular bonds into PBC with integer image shifts; build periodic neighbor edges if explicit bonds are missing.
  4) Compute bond-length means/variances per atom-type pair and bond order for the bond-aware loss.
  5) Save tensors for fast loading during training/inference.

## Training
Example (adjust dataset/config as needed):
```bash
python scripts_model/run.py data=organic model=orgflow \
  train.pl_trainer.max_epochs=XXX \
  logging.wandb.mode=disabled
```
Key hyperparameters (see paper/appendix):
- λ_f (fractional coord loss), λ_l (lattice loss), λ_b (bond-aware loss)
- Hidden dim 128, layers 12, time dim 256, SiLU activation
- ODE steps ~20 for inference

## Evaluation
- Reconstruction and match rate using pymatgen’s `StructureMatcher` + spglib symmetry checks.
- Bond-aware metrics and lattice/coord RMSD.
- Scripts under `scripts_model/evaluate.py` and `scripts_analysis/` for consolidation, lattice metrics, pre-relaxation, and DFT setup.

## Citing
If you use OrgFlow, please cite:
```
OrgFlow: Generative Modeling of Organic Crystal Structures from Molecular Graphs
M. Vahediahmar, M. A. McDonald, F. Liu. Drexel University.
```

## License
See `LICENSE.md`. Third-party components in `remote/` follow their respective licenses.
