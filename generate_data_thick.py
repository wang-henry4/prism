"""
Generate test data biased toward long sequences with high cumulative thickness.

Usage:
    python generate_data_thick.py --n_samples 10000 --out_dir ./data/thick_test
    python generate_data_thick.py --n_samples 10000 --min_layers 15 --min_cum_thk 8000
"""

import argparse
import glob
import os
import random

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

def sample_thick_structure(
    rng: random.Random,
    min_layers: int,
    max_layers: int,
    min_cum_thk: float,
) -> tuple[list[str], list[float]]:
    """Rejection-sample structures with high layer count and cumulative thickness."""
    thk_choices = list(range(THK_MIN, THK_MAX + THK_STEP, THK_STEP))
    # Bias thickness sampling toward upper half
    thk_weights = [t ** 2 for t in thk_choices]

    while True:
        n = rng.randint(min_layers, max_layers)
        materials = rng.choices(MATERIALS, k=n)
        thicknesses = [float(rng.choices(thk_choices, weights=thk_weights, k=1)[0]) for _ in range(n)]
        if sum(thicknesses) >= min_cum_thk:
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


def _next_partition_index(out_dir: str) -> int:
    existing = glob.glob(os.path.join(out_dir, "part_*.arrow"))
    if not existing:
        return 0
    indices = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in existing]
    return max(indices) + 1


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate test data biased toward long, thick stacks"
    )
    parser.add_argument("--n_samples",    type=int,   default=10_000)
    parser.add_argument("--min_layers",   type=int,   default=15)
    parser.add_argument("--max_layers",   type=int,   default=MAX_LAYERS)
    parser.add_argument("--min_cum_thk",  type=float, default=8000.0,
                        help="Minimum cumulative thickness in nm")
    parser.add_argument("--out_dir",      default="./data/thick_test")
    parser.add_argument("--nk_dir",       default="./nk")
    parser.add_argument("--workers",      type=int,   default=32)
    parser.add_argument("--seed",         type=int,   default=None)
    args = parser.parse_args()

    if args.seed is not None:
        print(f"Seed: {args.seed}")
    rng = random.Random(args.seed)

    print(f"Sampling {args.n_samples:,} structures "
          f"(layers {args.min_layers}–{args.max_layers}, cum_thk ≥ {args.min_cum_thk:.0f} nm)…")
    structures = [
        sample_thick_structure(rng, args.min_layers, args.max_layers, args.min_cum_thk)
        for _ in tqdm(range(args.n_samples), desc="sampling", unit="sample")
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
                desc="simulating",
            )
        )

    os.makedirs(args.out_dir, exist_ok=True)
    part_idx = _next_partition_index(args.out_dir)
    part_path = os.path.join(args.out_dir, f"part_{part_idx:03d}.arrow")

    _write_arrow(
        part_path,
        list(materials_list),
        list(thicknesses_list),
        spectra,
    )
    print(f"→ {part_path} ({args.n_samples:,} samples)")


if __name__ == "__main__":
    main()
