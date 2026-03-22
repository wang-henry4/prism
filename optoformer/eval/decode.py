"""Greedy autoregressive decoding for the InverseModel."""

import torch
from torch import Tensor


def greedy_decode(
    model,
    spectrum: Tensor,       # [B, 142]
    vocab,
    max_len: int = 12,      # max sequence length including BOS
    device: torch.device | None = None,
) -> tuple[list[list[int]], list[list[float]]]:
    """
    Greedy autoregressive decoding for InverseModel.

    Generates one token at a time, stopping at EOS or max_len.

    Returns:
        mat_ids:  list of B lists — material IDs (no BOS/EOS/PAD)
        thk_vals: list of B lists — predicted thickness values (nm)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    B = spectrum.size(0)
    spectrum = spectrum.to(device)

    mat_seqs = torch.full((B, 1), vocab.BOS, dtype=torch.long, device=device)
    thk_seqs = torch.zeros(B, 1, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    mat_results: list[list[int]]   = [[] for _ in range(B)]
    thk_results: list[list[float]] = [[] for _ in range(B)]

    with torch.no_grad():
        for _ in range(max_len - 1):
            T = mat_seqs.size(1)
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
            tgt_mask = causal.unsqueeze(0).expand(B, -1, -1)

            mat_logits, thk_pred = model(spectrum, mat_seqs, thk_seqs, tgt_mask)

            next_mat = mat_logits[:, -1, :].argmax(dim=-1)  # [B]
            next_thk = thk_pred[:, -1]                       # [B]

            for b in range(B):
                if not finished[b]:
                    token = next_mat[b].item()
                    if token == vocab.EOS:
                        finished[b] = True
                    elif token != vocab.PAD:
                        mat_results[b].append(token)
                        thk_results[b].append(float(next_thk[b].item()))

            mat_seqs = torch.cat([mat_seqs, next_mat.unsqueeze(1)], dim=1)
            thk_seqs = torch.cat([thk_seqs, next_thk.unsqueeze(1)], dim=1)

            if finished.all():
                break

    return mat_results, thk_results
