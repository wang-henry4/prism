"""
Generate thin-film training data via TMM simulation.

Usage:
    # First generation — creates part_000.arrow in each split folder
    python generate_data.py --n_samples 3000000

    # Add more data — auto-detects next partition number
    python generate_data.py --n_samples 1000000 --seed 99999

Outputs partitioned Arrow files into data/train/, data/dev/, data/val/.
"""

import argparse
import glob
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

def _build_length_weights(min_layers: int, max_layers: int, alpha: float = 1.0) -> list[float]:
    """P(L) ∝ L^alpha weights for sequence lengths min_layers..max_layers."""
    lengths = range(min_layers, max_layers + 1)
    raw = [l ** alpha for l in lengths]
    total = sum(raw)
    return [w / total for w in raw]


def sample_structure(
    rng: random.Random,
    min_layers: int,
    max_layers: int,
    length_weights: list[float],
) -> tuple[list[str], list[float]]:
    lengths = list(range(min_layers, max_layers + 1))
    n = rng.choices(lengths, weights=length_weights, k=1)[0]
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


def _next_partition_index(split_dir: str) -> int:
    """Find the next available partition index in a split directory."""
    existing = glob.glob(os.path.join(split_dir, "part_*.arrow"))
    if not existing:
        return 0
    indices = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in existing]
    return max(indices) + 1


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thin-film TMM dataset")
    parser.add_argument("--n_samples",  type=int,   default=10_000_000)
    parser.add_argument("--min_layers", type=int,   default=MIN_LAYERS)
    parser.add_argument("--max_layers", type=int,   default=MAX_LAYERS)
    parser.add_argument("--dev_split",  type=float, default=0.01)
    parser.add_argument("--val_split",  type=float, default=0.001)
    parser.add_argument("--out_dir",    default="./data")
    parser.add_argument("--nk_dir",     default="./nk")
    parser.add_argument("--workers",    type=int,   default=32)
    parser.add_argument("--seed",       type=int,   default=None,
                        help="Random seed (default: random). Set for reproducibility.")
    args = parser.parse_args()

    if args.seed is not None:
        print(f"Seed: {args.seed}")
    rng = random.Random(args.seed)

    length_weights = _build_length_weights(args.min_layers, args.max_layers, alpha=1.0)
    print(f"Sampling {args.n_samples:,} structures (length weights α=1.0)…")
    structures = [
        sample_structure(rng, args.min_layers, args.max_layers, length_weights)
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

    splits = {
        "train": (0, n_train),
        "dev":   (n_train, n_train + n_dev),
        "val":   (n_train + n_dev, args.n_samples),
    }

    for split_name, (start, end) in splits.items():
        if start == end:
            print(f"{split_name}: 0 samples → skipping")
            continue
        split_dir = os.path.join(args.out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        part_idx = _next_partition_index(split_dir)
        part_path = os.path.join(split_dir, f"part_{part_idx:03d}.arrow")

        _write_arrow(
            part_path,
            list(materials_list[start:end]),
            list(thicknesses_list[start:end]),
            spectra[start:end],
        )
        n_split = end - start
        print(f"{split_name}: {n_split:,} samples → {part_path}")


if __name__ == "__main__":
    main()
