"""
Generate thin-film training data via TMM simulation.

Usage:
    # Generate 10M training samples in 1M chunks + 100K dev + 10K val
    python generate_data.py --n_train 10000000 --n_dev 100000 --n_val 10000

    # Add more training data — auto-detects next partition number
    python generate_data.py --n_train 5000000 --n_dev 0 --n_val 0 --seed 99999

Outputs partitioned Arrow files into data/train/, data/dev/, data/val/.
Training data is written in chunks (default 1M) to keep memory bounded.
"""

import argparse
import glob
import os
import random

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from tqdm import tqdm

from prism.constants import MATERIALS, THK_MIN, THK_MAX, THK_STEP, MIN_LAYERS, MAX_LAYERS
from prism.data.sim import load_nk, simulate

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
    thk_min: int = THK_MIN,
    thk_max: int = THK_MAX,
    thk_step: int = THK_STEP,
) -> tuple[list[str], list[float]]:
    lengths = list(range(min_layers, max_layers + 1))
    n = rng.choices(lengths, weights=length_weights, k=1)[0]
    materials   = rng.choices(MATERIALS, k=n)
    thicknesses = [float(rng.randrange(thk_min, thk_max + thk_step, thk_step)) for _ in range(n)]
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


# ── Chunk generation ──────────────────────────────────────────────────────────

def _generate_chunk(
    rng: random.Random,
    n_samples: int,
    min_layers: int,
    max_layers: int,
    length_weights: list[float],
    nk_dir: str,
    workers: int,
    out_path: str,
    desc: str,
    thk_min: int = THK_MIN,
    thk_max: int = THK_MAX,
    thk_step: int = THK_STEP,
) -> None:
    """Sample structures, simulate, and write a single Arrow partition."""
    structures = [
        sample_structure(rng, min_layers, max_layers, length_weights, thk_min, thk_max, thk_step)
        for _ in range(n_samples)
    ]
    materials_list, thicknesses_list = zip(*structures)

    from multiprocessing import Pool

    with Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(nk_dir,),
    ) as pool:
        spectra = list(
            tqdm(
                pool.imap(_simulate_one, zip(materials_list, thicknesses_list)),
                total=n_samples,
                unit="sample",
                desc=desc,
            )
        )

    _write_arrow(
        out_path,
        list(materials_list),
        list(thicknesses_list),
        spectra,
    )
    print(f"  → {out_path} ({n_samples:,} samples)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thin-film TMM dataset")
    parser.add_argument("--n_train",     type=int,   default=10_000_000)
    parser.add_argument("--n_dev",       type=int,   default=100_000)
    parser.add_argument("--n_val",       type=int,   default=10_000)
    parser.add_argument("--chunk_size",  type=int,   default=1_000_000,
                        help="Samples per Arrow partition for training data")
    parser.add_argument("--min_layers",  type=int,   default=MIN_LAYERS)
    parser.add_argument("--max_layers",  type=int,   default=MAX_LAYERS)
    parser.add_argument("--out_dir",     default="./data")
    parser.add_argument("--nk_dir",      default="./nk")
    parser.add_argument("--workers",     type=int,   default=32)
    parser.add_argument("--seed",        type=int,   default=None,
                        help="Random seed (default: random). Set for reproducibility.")
    parser.add_argument("--thk_min",     type=int,   default=THK_MIN)
    parser.add_argument("--thk_max",     type=int,   default=THK_MAX)
    parser.add_argument("--thk_step",    type=int,   default=THK_STEP)
    args = parser.parse_args()

    if args.seed is not None:
        print(f"Seed: {args.seed}")
    rng = random.Random(args.seed)

    length_weights = _build_length_weights(args.min_layers, args.max_layers, alpha=1.0)

    # ── Training data: chunked ────────────────────────────────────────────────
    if args.n_train > 0:
        train_dir = os.path.join(args.out_dir, "train")
        os.makedirs(train_dir, exist_ok=True)
        part_idx = _next_partition_index(train_dir)

        remaining = args.n_train
        chunk_num = 0
        n_chunks = (args.n_train + args.chunk_size - 1) // args.chunk_size
        print(f"Generating {args.n_train:,} training samples in {n_chunks} chunk(s) of ≤{args.chunk_size:,}…")

        while remaining > 0:
            chunk_n = min(args.chunk_size, remaining)
            part_path = os.path.join(train_dir, f"part_{part_idx:03d}.arrow")
            _generate_chunk(
                rng, chunk_n, args.min_layers, args.max_layers,
                length_weights, args.nk_dir, args.workers, part_path,
                desc=f"train chunk {chunk_num + 1}/{n_chunks}",
                thk_min=args.thk_min, thk_max=args.thk_max, thk_step=args.thk_step,
            )
            part_idx += 1
            chunk_num += 1
            remaining -= chunk_n

    # ── Dev set ───────────────────────────────────────────────────────────────
    if args.n_dev > 0:
        dev_dir = os.path.join(args.out_dir, "dev")
        os.makedirs(dev_dir, exist_ok=True)
        part_idx = _next_partition_index(dev_dir)
        part_path = os.path.join(dev_dir, f"part_{part_idx:03d}.arrow")
        print(f"\nGenerating {args.n_dev:,} dev samples…")
        _generate_chunk(
            rng, args.n_dev, args.min_layers, args.max_layers,
            length_weights, args.nk_dir, args.workers, part_path,
            desc="dev",
            thk_min=args.thk_min, thk_max=args.thk_max, thk_step=args.thk_step,
        )

    # ── Val set ───────────────────────────────────────────────────────────────
    if args.n_val > 0:
        val_dir = os.path.join(args.out_dir, "val")
        os.makedirs(val_dir, exist_ok=True)
        part_idx = _next_partition_index(val_dir)
        part_path = os.path.join(val_dir, f"part_{part_idx:03d}.arrow")
        print(f"\nGenerating {args.n_val:,} val samples…")
        _generate_chunk(
            rng, args.n_val, args.min_layers, args.max_layers,
            length_weights, args.nk_dir, args.workers, part_path,
            desc="val",
            thk_min=args.thk_min, thk_max=args.thk_max, thk_step=args.thk_step,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
