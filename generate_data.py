"""
Generate thin-film training data via TMM simulation.

Usage:
    python generate_data.py \\
        --n_samples 100000 --min_layers 1 --max_layers 10 \\
        --dev_split 0.1 --val_split 0.1 --out_dir ./data --nk_dir ./nk \\
        --workers 8 --seed 42

Outputs data/train.arrow, data/dev.arrow, and data/val.arrow (Apache Arrow / Feather format).
"""

import argparse
import os
import random

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from tqdm import tqdm

from optoformer.constants import MATERIALS, THK_MIN, THK_MAX, THK_STEP, MIN_LAYERS, MAX_LAYERS
from optoformer.data.sim import load_nk, simulate

# ── Worker-process globals ─────────────────────────────────────────────────────
_nk_dict: dict | None = None


def _worker_init(nk_dir: str) -> None:
    global _nk_dict
    _nk_dict = load_nk(nk_dir)


def _simulate_one(args: tuple[list[str], list[float]]) -> list[float]:
    materials, thicknesses = args
    return simulate(materials, thicknesses, _nk_dict)  # type: ignore[arg-type]


# ── Sampling ───────────────────────────────────────────────────────────────────

def sample_structure(
    rng: random.Random,
    min_layers: int,
    max_layers: int,
) -> tuple[list[str], list[float]]:
    n = rng.randint(min_layers, max_layers)
    materials   = rng.choices(MATERIALS, k=n)
    thicknesses = [float(rng.randrange(THK_MIN, THK_MAX + THK_STEP, THK_STEP)) for _ in range(n)]
    return materials, thicknesses


# ── Arrow writer ───────────────────────────────────────────────────────────────

def _write_arrow(
    path: str,
    materials_list: list[list[str]],
    thicknesses_list: list[list[float]],
    spectra: list[list[float]],
) -> None:
    table = pa.table({
        "materials":   pa.array(materials_list,   type=pa.list_(pa.string())),
        "thicknesses": pa.array(thicknesses_list, type=pa.list_(pa.float32())),
        "spectra":     pa.array(spectra,          type=pa.list_(pa.float32())),
    })
    feather.write_feather(table, path, compression="uncompressed")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thin-film TMM dataset")
    parser.add_argument("--n_samples",  type=int,   default=3_000_000)
    parser.add_argument("--min_layers", type=int,   default=MIN_LAYERS)
    parser.add_argument("--max_layers", type=int,   default=MAX_LAYERS)
    parser.add_argument("--dev_split",  type=float, default=0.05)
    parser.add_argument("--val_split",  type=float, default=0.01)
    parser.add_argument("--out_dir",    default="./data")
    parser.add_argument("--nk_dir",     default="./nk")
    parser.add_argument("--workers",    type=int,   default=32)
    parser.add_argument("--seed",       type=int,   default=17291)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Sampling {args.n_samples:,} structures…")
    structures = [
        sample_structure(rng, args.min_layers, args.max_layers)
        for _ in range(args.n_samples)
    ]
    materials_list, thicknesses_list = zip(*structures)

    print(f"Simulating with {args.workers} workers…")
    from multiprocessing import Pool

    with Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.nk_dir,),
    ) as pool:
        spectra = list(
            tqdm(
                pool.imap(_simulate_one, zip(materials_list, thicknesses_list)),
                total=args.n_samples,
                unit="sample",
            )
        )

    n_dev   = int(args.n_samples * args.dev_split)
    n_val   = int(args.n_samples * args.val_split)
    n_train = args.n_samples - n_dev - n_val

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.arrow")
    dev_path   = os.path.join(args.out_dir, "dev.arrow")
    val_path   = os.path.join(args.out_dir, "val.arrow")

    _write_arrow(
        train_path,
        list(materials_list[:n_train]),
        list(thicknesses_list[:n_train]),
        spectra[:n_train],
    )
    _write_arrow(
        dev_path,
        list(materials_list[n_train:n_train + n_dev]),
        list(thicknesses_list[n_train:n_train + n_dev]),
        spectra[n_train:n_train + n_dev],
    )
    _write_arrow(
        val_path,
        list(materials_list[n_train + n_dev:]),
        list(thicknesses_list[n_train + n_dev:]),
        spectra[n_train + n_dev:],
    )

    print(f"train: {n_train:,} samples → {train_path}")
    print(f"dev:   {n_dev:,} samples → {dev_path}")
    print(f"val:   {n_val:,} samples → {val_path}")


if __name__ == "__main__":
    main()
