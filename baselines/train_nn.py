"""
Train and evaluate Tandem Network and CVAE baselines.

Usage:
    python baselines/train_nn.py --model tandem
    python baselines/train_nn.py --model cvae
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.feather as feather
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optoformer.constants import MATERIALS, MAX_LAYERS, N_SPECTRUM, THK_MIN, THK_MAX
from optoformer.data.sim import load_nk, simulate
from optoformer.eval.metrics import SpectrumMetrics
from optoformer.eval.targets import HANDCRAFTED_TARGETS
from baselines.nn_models import TandemNetwork, CVAE, N_MATERIALS

_nk_dict = None


def _worker_init(nk_dir):
    global _nk_dict
    _nk_dict = load_nk(nk_dir)


def _simulate_one(args):
    materials, thicknesses = args
    if not materials:
        return [0.0] * N_SPECTRUM
    try:
        return simulate(materials, thicknesses, _nk_dict)
    except Exception:
        return [0.0] * N_SPECTRUM


# ── Dataset ───────────────────────────────────────────────────────────────────

class ThinFilmDataset(Dataset):
    def __init__(self, path: str):
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.arrow")))
        else:
            files = [path]

        all_mats, all_thks, all_specs = [], [], []
        for f in files:
            t = feather.read_table(f, memory_map=True)
            all_mats.extend(t["materials"].to_pylist())
            all_thks.extend(t["thicknesses"].to_pylist())
            all_specs.extend(t["spectra"].to_pylist())

        self.mat_to_idx = {m: i for i, m in enumerate(MATERIALS)}
        self.n = len(all_specs)
        self.spectra = np.array(all_specs, dtype=np.float32)
        self.mat_ids = np.full((self.n, MAX_LAYERS), -1, dtype=np.int64)
        self.thk_vals = np.zeros((self.n, MAX_LAYERS), dtype=np.float32)
        self.lengths = np.zeros(self.n, dtype=np.int64)

        for i, (mats, thks) in enumerate(zip(all_mats, all_thks)):
            n_layers = len(mats)
            self.lengths[i] = n_layers
            for j in range(n_layers):
                self.mat_ids[i, j] = self.mat_to_idx[mats[j]]
                self.thk_vals[i, j] = thks[j]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.spectra[idx], self.mat_ids[idx], self.thk_vals[idx], self.lengths[idx]


def encode_batch(mat_ids, thk_vals, lengths):
    B = mat_ids.shape[0]
    mat_onehot = torch.zeros(B, MAX_LAYERS, N_MATERIALS, device=mat_ids.device)
    valid = mat_ids >= 0
    mat_onehot[valid] = F.one_hot(mat_ids[valid].clamp(min=0), N_MATERIALS).float()
    thk_norm = thk_vals / THK_MAX
    length_norm = lengths.float().unsqueeze(-1) / MAX_LAYERS
    return mat_onehot, thk_norm, length_norm


# ── Training ──────────────────────────────────────────────────────────────────

def train_tandem(model, train_loader, dev_loader, epochs, lr, device, save_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_dev_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.perf_counter()

        for spectra, mat_ids, thk_vals, lengths in train_loader:
            spectra = spectra.to(device)
            mat_ids, thk_vals, lengths = mat_ids.to(device), thk_vals.to(device), lengths.to(device)
            mat_onehot, thk_norm, length_norm = encode_batch(mat_ids, thk_vals, lengths)

            mat_logits, thk_pred, len_logits, spec_recon = model(spectra)

            valid = mat_ids >= 0
            mat_loss = F.cross_entropy(mat_logits[valid], mat_ids[valid])
            thk_loss = F.mse_loss(thk_pred[valid], thk_norm[valid])
            len_loss = F.cross_entropy(len_logits, lengths - 1)
            spec_loss = F.mse_loss(spec_recon, spectra)

            loss = mat_loss + thk_loss + len_loss + 10.0 * spec_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.perf_counter() - t0
        dev_loss = eval_loss(model, dev_loader, device, "tandem")
        print(f"Epoch {epoch}/{epochs}  loss={total_loss/n_batches:.4f}  dev={dev_loss:.4f}  ({elapsed:.0f}s)")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(save_dir, "best.pt"), weights_only=True))
    return model


def train_cvae(model, train_loader, dev_loader, epochs, lr, device, save_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_dev_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.perf_counter()

        for spectra, mat_ids, thk_vals, lengths in train_loader:
            spectra = spectra.to(device)
            mat_ids, thk_vals, lengths = mat_ids.to(device), thk_vals.to(device), lengths.to(device)
            mat_onehot, thk_norm, length_norm = encode_batch(mat_ids, thk_vals, lengths)

            mat_logits, thk_pred, len_logits, mu, logvar = model(
                spectra, mat_onehot, thk_norm, length_norm
            )

            valid = mat_ids >= 0
            mat_loss = F.cross_entropy(mat_logits[valid], mat_ids[valid])
            thk_loss = F.mse_loss(thk_pred[valid], thk_norm[valid])
            len_loss = F.cross_entropy(len_logits, lengths - 1)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            kl_weight = min(1.0, epoch / (epochs * 0.3))
            loss = mat_loss + thk_loss + len_loss + kl_weight * 0.1 * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.perf_counter() - t0
        dev_loss = eval_loss(model, dev_loader, device, "cvae")
        print(f"Epoch {epoch}/{epochs}  loss={total_loss/n_batches:.4f}  dev={dev_loss:.4f}  ({elapsed:.0f}s)")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(save_dir, "best.pt"), weights_only=True))
    return model


@torch.no_grad()
def eval_loss(model, loader, device, model_type):
    model.eval()
    total, n = 0.0, 0
    for spectra, mat_ids, thk_vals, lengths in loader:
        spectra = spectra.to(device)
        mat_ids, thk_vals, lengths = mat_ids.to(device), thk_vals.to(device), lengths.to(device)
        mat_onehot, thk_norm, length_norm = encode_batch(mat_ids, thk_vals, lengths)

        if model_type == "tandem":
            mat_logits, thk_pred, len_logits, spec_recon = model(spectra)
            loss = F.mse_loss(spec_recon, spectra)
        else:
            mat_logits, thk_pred, len_logits, mu, logvar = model(
                spectra, mat_onehot, thk_norm, length_norm
            )
            valid = mat_ids >= 0
            loss = F.cross_entropy(mat_logits[valid], mat_ids[valid]) + \
                   F.mse_loss(thk_pred[valid], thk_norm[valid])
        total += loss.item()
        n += 1
    return total / max(n, 1)


# ── Decode + Evaluate ─────────────────────────────────────────────────────────

def decode_prediction(mat_logits, thk_pred, len_logits):
    n_layers = len_logits.argmax(dim=-1).item() + 1
    n_layers = min(n_layers, MAX_LAYERS)
    mat_ids = mat_logits[:n_layers].argmax(dim=-1).cpu().tolist()
    thk_vals = (thk_pred[:n_layers] * THK_MAX).cpu().tolist()
    thk_vals = [max(THK_MIN, t) for t in thk_vals]
    return [MATERIALS[i] for i in mat_ids], thk_vals


def evaluate_model(model, model_type, val_spectra, hc_targets, device, nk_dir, workers):
    model.eval()
    results = {"model": model_type}

    # Val set
    if val_spectra:
        all_mats, all_thks = [], []
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(0, len(val_spectra), 512):
                batch = torch.tensor(val_spectra[i:i+512], dtype=torch.float32, device=device)
                if model_type == "tandem":
                    mat_logits, thk_pred, len_logits, _ = model(batch)
                else:
                    mat_logits_all, thk_pred_all, len_logits_all = model.sample(batch, 10)
                    mat_logits = mat_logits_all[:, 0]
                    thk_pred = thk_pred_all[:, 0]
                    len_logits = len_logits_all[:, 0]
                for j in range(mat_logits.shape[0]):
                    mats, thks = decode_prediction(mat_logits[j], thk_pred[j], len_logits[j])
                    all_mats.append(mats)
                    all_thks.append(thks)
        decode_time = time.perf_counter() - t0

        with Pool(workers, initializer=_worker_init, initargs=(nk_dir,)) as pool:
            pred_spectra = pool.map(_simulate_one, zip(all_mats, all_thks))

        pred_arr, target_arr = np.array(pred_spectra), np.array(val_spectra)
        metrics = SpectrumMetrics.compute(pred_arr, target_arr)
        per_sample_mae = np.mean(np.abs(pred_arr - target_arr), axis=1)
        metrics["median_mae"] = float(np.median(per_sample_mae))
        metrics["p90_mae"] = float(np.percentile(per_sample_mae, 90))
        results.update(metrics)
        results["n_val_samples"] = len(val_spectra)
        results["decode_time_s"] = decode_time
        print(f"Val: MAE={metrics['mae']:.6f}  MSE={metrics['mse']:.6f}  R²={metrics['r2']:.4f}")

    # Handcrafted targets
    if hc_targets:
        hc_spectra = [t["spectrum"] for t in hc_targets]
        all_mats, all_thks = [], []
        with torch.no_grad():
            batch = torch.tensor(hc_spectra, dtype=torch.float32, device=device)
            if model_type == "tandem":
                mat_logits, thk_pred, len_logits, _ = model(batch)
            else:
                mat_logits_all, thk_pred_all, len_logits_all = model.sample(batch, 10)
                mat_logits = mat_logits_all[:, 0]
                thk_pred = thk_pred_all[:, 0]
                len_logits = len_logits_all[:, 0]
            for j in range(len(hc_targets)):
                mats, thks = decode_prediction(mat_logits[j], thk_pred[j], len_logits[j])
                all_mats.append(mats)
                all_thks.append(thks)

        with Pool(workers, initializer=_worker_init, initargs=(nk_dir,)) as pool:
            pred_spectra = pool.map(_simulate_one, zip(all_mats, all_thks))

        hc_metrics_all = []
        for i, (target, sim_spec) in enumerate(zip(hc_targets, pred_spectra)):
            target_spec = np.array(target["spectrum"])
            m = SpectrumMetrics.compute(np.array(sim_spec).reshape(1, -1), target_spec.reshape(1, -1))
            m["name"], m["label"] = target["name"], target["label"]
            m["materials"], m["thicknesses"] = all_mats[i], all_thks[i]
            hc_metrics_all.append(m)
            print(f"  {target['label']:45s}  MAE={m['mae']:.6f}  R²={m['r2']:.4f}")

        hc_maes = [m["mae"] for m in hc_metrics_all]
        hc_mses = [m["mse"] for m in hc_metrics_all]
        hc_r2s = [m["r2"] for m in hc_metrics_all]
        results["handcrafted_mean_mae"] = float(np.mean(hc_maes))
        results["handcrafted_mean_mse"] = float(np.mean(hc_mses))
        results["handcrafted_mean_r2"] = float(np.mean(hc_r2s))
        results["handcrafted"] = hc_metrics_all
        print(f"HC: MAE={np.mean(hc_maes):.6f}  MSE={np.mean(hc_mses):.6f}  R²={np.mean(hc_r2s):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["tandem", "cvae"])
    parser.add_argument("--train_path", default="./data/max_len_20_10nm/train")
    parser.add_argument("--dev_path", default="./data/max_len_20_10nm/dev/part_000.arrow")
    parser.add_argument("--val_path", default="./data/max_len_20_10nm/val/part_000.arrow")
    parser.add_argument("--nk_dir", default="./nk")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_hidden", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save_dir", default="./saved_models/baselines")
    parser.add_argument("--plot_dir", default="./plots/baselines")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = os.path.join(args.save_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)

    print("Loading training data...")
    train_ds = ThinFilmDataset(args.train_path)
    print(f"  Train: {len(train_ds):,}")
    dev_ds = ThinFilmDataset(args.dev_path)
    print(f"  Dev: {len(dev_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if args.model == "tandem":
        model = TandemNetwork(args.d_hidden).to(device)
    else:
        model = CVAE(args.d_hidden).to(device)
    print(f"Model: {args.model}, params: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\nTraining {args.epochs} epochs...")
    if args.model == "tandem":
        model = train_tandem(model, train_loader, dev_loader, args.epochs, args.lr, device, save_dir)
    else:
        model = train_cvae(model, train_loader, dev_loader, args.epochs, args.lr, device, save_dir)

    print("\nEvaluating...")
    table = feather.read_table(args.val_path, memory_map=True)
    val_spectra = table["spectra"].to_pylist()

    results = evaluate_model(model, args.model, val_spectra, HANDCRAFTED_TARGETS, device, args.nk_dir, args.workers)

    out_dir = os.path.join(args.plot_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
