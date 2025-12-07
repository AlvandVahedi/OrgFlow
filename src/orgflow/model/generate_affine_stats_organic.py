from pathlib import Path
import torch
import yaml
from tqdm import tqdm

from orgflow.model.standardize import (
    compute_affine_stats,
    get_affine_stats_filename,
)

from orgflow.cfg_utils import dataset_options
from orgflow.rfm.manifold_getter import (
    atom_type_manifold_types,
    coord_manifold_types,
    lattice_manifold_types,
)

# Target dataset
target_dataset = "organic"  # <-- make sure this is a valid option

# Output stats dir
output_dir = Path(__file__).parent / "model"
output_dir.mkdir(exist_ok=True)

# Loop over stats to collect
for collect_stats_on in ["coord", "lattice"]:
    if collect_stats_on == "coord":
        coord_manifolds = [
            "flat_torus_01",
            "flat_torus_01_normal",
            "flat_torus_01_fixfirst",
            "flat_torus_01_fixfirst_normal",
        ]
        for cm in coord_manifolds:
            stats = compute_affine_stats(
                dataset=target_dataset,
                collect_stats_on=collect_stats_on,
                atom_type_manifold="null_manifold",
                coord_manifold=cm,
                lattice_manifold="non_symmetric",
            )
            stats["u_t_mean"] = torch.zeros_like(stats["u_t_mean"])
            stats["x_t_mean"] = torch.zeros_like(stats["x_t_mean"])
            stats["x_t_std"] = torch.ones_like(stats["x_t_std"])

            file = output_dir / get_affine_stats_filename(target_dataset, cm)
            with open(file, "w") as f:
                yaml.dump({k: v.tolist() for k, v in stats.items()}, f)

    elif collect_stats_on == "lattice":
        lattice_manifolds = [
            "non_symmetric",
            "spd_euclidean_geo",
            "spd_riemanian_geo",
            "lattice_params",
            "lattice_params_normal_base",
        ]

        for lm in lattice_manifolds:
            if lm == "non_symmetric":
                stats = {
                    "x_t_mean": torch.zeros(9),
                    "x_t_std": torch.ones(9),
                    "u_t_mean": torch.zeros(9),
                    "u_t_std": torch.ones(9),
                }
            else:
                stats = compute_affine_stats(
                    dataset=target_dataset,
                    collect_stats_on=collect_stats_on,
                    atom_type_manifold="null_manifold",
                    coord_manifold="flat_torus_01_fixfirst",
                    lattice_manifold=lm,
                )

            file = output_dir / get_affine_stats_filename(target_dataset, lm)
            with open(file, "w") as f:
                yaml.dump({k: v.tolist() for k, v in stats.items()}, f)