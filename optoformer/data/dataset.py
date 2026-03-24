"""
Vocabulary, Dataset, Batch, and DataLoader helpers for thin-film training data.

Arrow data format (columns):
    materials:   list<string>   — per-sample layer material names
    thicknesses: list<float32>  — per-sample layer thicknesses (nm)
    spectra:     list<float32>  — 142 floats (71 R + 71 T)

Sequence encoding:
    mat_ids:  [BOS, mat_id_1, ..., mat_id_N, EOS]
    thk_vals: [0.0, thk_1,   ..., thk_N,   0.0]   (BOS/EOS get thickness 0)
"""

import glob
import os
from functools import partial

import pyarrow.feather as feather
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from optoformer.constants import MATERIALS


class Vocab:
    """
    Material vocabulary with special tokens PAD, BOS, EOS.

    IDs:
        0 → PAD
        1 → BOS
        2 → EOS
        3… → material names in order
    """

    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(self, materials: list[str] = MATERIALS):
        self.id2word: list[str] = ["PAD", "BOS", "EOS"] + list(materials)
        self.word2id: dict[str, int] = {w: i for i, w in enumerate(self.id2word)}

    def __len__(self) -> int:
        return len(self.id2word)

    def encode(self, name: str) -> int:
        return self.word2id[name]

    def decode(self, idx: int) -> str:
        return self.id2word[idx]

    def encode_sequence(self, names: list[str]) -> list[int]:
        """BOS + [mat_ids] + EOS."""
        return [self.BOS] + [self.encode(n) for n in names] + [self.EOS]

    def decode_sequence(self, ids: list[int]) -> list[str]:
        """Strip BOS/EOS/PAD and return material names."""
        return [
            self.decode(i)
            for i in ids
            if i not in (self.PAD, self.BOS, self.EOS)
        ]


class ThinFilmDataset(Dataset):
    """PyTorch Dataset wrapping a train.arrow or dev.arrow file (lazy-loaded)."""

    def __init__(self, path: str, vocab: Vocab):
        self.table = feather.read_table(path, memory_map=True)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.slice(idx, 1)
        materials   = row["materials"][0].as_py()
        thicknesses = row["thicknesses"][0].as_py()
        spectrum    = row["spectra"][0].as_py()

        mat_ids  = self.vocab.encode_sequence(materials)
        thk_vals = [0.0] + list(thicknesses) + [0.0]

        return (
            torch.tensor(mat_ids,  dtype=torch.long),
            torch.tensor(thk_vals, dtype=torch.float32),
            torch.tensor(spectrum, dtype=torch.float32),
        )


class Batch:
    """
    Holds one padded batch with masks for forward and inverse training.

    Attributes (forward model):
        src_mat:  [B, S]     long    — padded material IDs
        src_thk:  [B, S]     float   — padded thickness values (nm)
        spectrum: [B, 142]   float   — target spectra
        src_mask: [B, 1, S]  bool    — True at non-PAD positions

    Attributes (inverse model, teacher-forced decoder):
        tgt_mat:     [B, S-1]      long  — decoder input  (BOS … last-1)
        tgt_thk:     [B, S-1]      float — decoder input thicknesses
        tgt_y_mat:   [B, S-1]      long  — decoder target (1 … EOS)
        tgt_y_thk:   [B, S-1]      float — decoder target thicknesses
        tgt_mask:    [B, S-1, S-1] bool  — causal + padding mask
        ntokens_tgt: int            — number of non-PAD target tokens
    """

    def __init__(self, mat_seqs: Tensor, thk_seqs: Tensor, spectra: Tensor, pad_id: int = Vocab.PAD):
        self.src_mat  = mat_seqs
        self.src_thk  = thk_seqs
        self.spectrum = spectra
        self.src_mask = (mat_seqs != pad_id).unsqueeze(1)   # [B, 1, S]
        self.ntokens  = int((mat_seqs != pad_id).sum())

        self.tgt_mat   = mat_seqs[:, :-1]
        self.tgt_thk   = thk_seqs[:, :-1]
        self.tgt_y_mat = mat_seqs[:, 1:]
        self.tgt_y_thk = thk_seqs[:, 1:]

        tgt_len = self.tgt_mat.size(1)
        pad_mask = (self.tgt_mat != pad_id).unsqueeze(1)
        causal   = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool))
        self.tgt_mask = pad_mask & causal.unsqueeze(0)

        self.ntokens_tgt = int((self.tgt_mat != pad_id).sum())


def _collate(batch: list, pad_id: int) -> Batch:
    mat_seqs, thk_seqs, spectra = zip(*batch)
    mat_padded = pad_sequence(mat_seqs, batch_first=True, padding_value=pad_id)
    thk_padded = pad_sequence(thk_seqs, batch_first=True, padding_value=0.0)
    return Batch(mat_padded, thk_padded, torch.stack(spectra), pad_id)


def make_dataloader(
    path: str,
    vocab: Vocab,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader from an Arrow file or a directory of partitions."""
    if os.path.isdir(path):
        parts = sorted(glob.glob(os.path.join(path, "part_*.arrow")))
        if not parts:
            raise FileNotFoundError(f"No part_*.arrow files found in {path}")
        dataset = ConcatDataset([ThinFilmDataset(p, vocab) for p in parts])
    else:
        dataset = ThinFilmDataset(path, vocab)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(_collate, pad_id=vocab.PAD),
    )
