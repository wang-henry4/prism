"""Autoregressive decoding strategies for the InverseModel."""

import torch
import torch.nn.functional as F
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
            # thk_pred is [B, T] or [B, T, vocab_size]
            if thk_pred.dim() == 3:
                # Per-material thickness: index by chosen material
                next_thk = thk_pred[:, -1, :].gather(-1, next_mat.unsqueeze(-1)).squeeze(-1)  # [B]
            else:
                next_thk = thk_pred[:, -1]  # [B]

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


def beam_search_decode(
    model,
    spectrum: Tensor,       # [B, 142]
    vocab,
    beam_width: int = 5,
    max_len: int = 12,
    length_penalty: float = 0.3,
    device: torch.device | None = None,
) -> tuple[list[list[int]], list[list[float]]]:
    """
    Beam search decoding for InverseModel.

    All beams for a given sample are batched into a single forward pass.
    Final sequences are scored as: log_prob_sum / (length ^ length_penalty).
    length_penalty=0 means no normalization, 1.0 means full per-token normalization.

    Returns:
        mat_ids:  list of B lists — material IDs (no BOS/EOS/PAD)
        thk_vals: list of B lists — predicted thickness values (nm)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    B = spectrum.size(0)
    K = beam_width
    spectrum = spectrum.to(device)

    mat_results: list[list[int]]   = []
    thk_results: list[list[float]] = []

    with torch.no_grad():
        for b in range(B):
            spec_b = spectrum[b:b+1]  # [1, 142]

            # Initialise K beams: all start from BOS
            # beam_mat: [K, T], beam_thk: [K, T], beam_scores: [K]
            beam_mat    = torch.full((K, 1), vocab.BOS, dtype=torch.long, device=device)
            beam_thk    = torch.zeros(K, 1, device=device)
            beam_scores = torch.zeros(K, device=device)
            beam_active = torch.ones(K, dtype=torch.bool, device=device)

            # Only the first beam is real initially; suppress others
            beam_scores[1:] = -float("inf")

            completed_scores: list[float]   = []
            completed_lengths: list[int]    = []
            completed_mat:    list[Tensor]  = []
            completed_thk:    list[Tensor]  = []

            spec_k = spec_b.expand(K, -1)  # [K, 142]

            for step in range(max_len - 1):
                n_active = beam_active.sum().item()
                if n_active == 0:
                    break

                T = beam_mat.size(1)
                causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
                tgt_mask = causal.unsqueeze(0).expand(K, -1, -1)

                mat_logits, thk_pred = model(spec_k, beam_mat, beam_thk, tgt_mask)

                log_probs = F.log_softmax(mat_logits[:, -1, :], dim=-1)  # [K, V]
                # thk_pred is [K, T] or [K, T, vocab_size]
                per_material_thk = thk_pred.dim() == 3
                if per_material_thk:
                    next_thk_all = thk_pred[:, -1, :]  # [K, V] — thickness per material
                else:
                    next_thk = thk_pred[:, -1]  # [K]

                # Mask inactive beams
                log_probs[~beam_active] = -float("inf")
                log_probs[~beam_active, vocab.PAD] = 0.0  # dummy so topk doesn't fail

                # Expand scores: [K, V]
                candidate_scores = beam_scores.unsqueeze(1) + log_probs  # [K, V]

                # Flatten and pick top K
                flat_scores = candidate_scores.view(-1)  # [K*V]
                topk_scores, topk_flat_idx = flat_scores.topk(K, dim=0)

                V = log_probs.size(1)
                beam_idx  = topk_flat_idx // V  # which beam
                token_idx = topk_flat_idx % V   # which token

                # Build new beams
                new_mat    = torch.cat([beam_mat[beam_idx], token_idx.unsqueeze(1)], dim=1)
                if per_material_thk:
                    # Each beam gets the thickness predicted for its chosen material
                    chosen_thk = next_thk_all[beam_idx, token_idx].unsqueeze(1)  # [K, 1]
                else:
                    chosen_thk = next_thk[beam_idx].unsqueeze(1)  # [K, 1]
                new_thk    = torch.cat([beam_thk[beam_idx], chosen_thk], dim=1)
                new_scores = topk_scores
                new_active = torch.ones(K, dtype=torch.bool, device=device)

                # Check for EOS / PAD
                for k in range(K):
                    tok = token_idx[k].item()
                    if tok == vocab.EOS:
                        seq_len = new_mat[k].size(0)  # includes BOS + tokens + EOS
                        completed_scores.append(new_scores[k].item())
                        completed_lengths.append(seq_len)
                        completed_mat.append(new_mat[k])
                        completed_thk.append(new_thk[k])
                        new_active[k] = False
                        new_scores[k] = -float("inf")
                    elif tok == vocab.PAD:
                        new_active[k] = False
                        new_scores[k] = -float("inf")

                beam_mat    = new_mat
                beam_thk    = new_thk
                beam_scores = new_scores
                beam_active = new_active

            # Add remaining active beams to completed
            for k in range(K):
                if beam_active[k]:
                    seq_len = beam_mat[k].size(0)
                    completed_scores.append(beam_scores[k].item())
                    completed_lengths.append(seq_len)
                    completed_mat.append(beam_mat[k])
                    completed_thk.append(beam_thk[k])

            if not completed_scores:
                mat_results.append([])
                thk_results.append([])
                continue

            # Pick best with length-normalized scoring
            def _normed_score(i: int) -> float:
                return completed_scores[i] / (completed_lengths[i] ** length_penalty)

            best_idx = max(range(len(completed_scores)), key=_normed_score)
            mat_ids  = completed_mat[best_idx].tolist()
            thk_vals = completed_thk[best_idx].tolist()

            mat_out = []
            thk_out = []
            for m, t in zip(mat_ids, thk_vals):
                if m not in (vocab.PAD, vocab.BOS, vocab.EOS):
                    mat_out.append(m)
                    thk_out.append(float(t))

            mat_results.append(mat_out)
            thk_results.append(thk_out)

    return mat_results, thk_results


def beam_search_decode_topk(
    model,
    spectrum: Tensor,       # [B, 142]
    vocab,
    beam_width: int = 5,
    max_len: int = 12,
    length_penalty: float = 0.3,
    device: torch.device | None = None,
) -> list[list[dict]]:
    """
    Beam search returning top-K candidates per sample (sorted best-first).

    Returns:
        list of B lists, each containing up to K dicts:
            {"mat_ids": list[int], "thk_vals": list[float], "score": float}
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    B = spectrum.size(0)
    K = beam_width
    spectrum = spectrum.to(device)

    all_results: list[list[dict]] = []

    with torch.no_grad():
        for b in range(B):
            spec_b = spectrum[b:b+1]

            beam_mat    = torch.full((K, 1), vocab.BOS, dtype=torch.long, device=device)
            beam_thk    = torch.zeros(K, 1, device=device)
            beam_scores = torch.zeros(K, device=device)
            beam_active = torch.ones(K, dtype=torch.bool, device=device)
            beam_scores[1:] = -float("inf")

            completed_scores: list[float]   = []
            completed_lengths: list[int]    = []
            completed_mat:    list[Tensor]  = []
            completed_thk:    list[Tensor]  = []

            spec_k = spec_b.expand(K, -1)

            for step in range(max_len - 1):
                n_active = beam_active.sum().item()
                if n_active == 0:
                    break

                T = beam_mat.size(1)
                causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
                tgt_mask = causal.unsqueeze(0).expand(K, -1, -1)

                mat_logits, thk_pred = model(spec_k, beam_mat, beam_thk, tgt_mask)

                log_probs = F.log_softmax(mat_logits[:, -1, :], dim=-1)
                per_material_thk = thk_pred.dim() == 3
                if per_material_thk:
                    next_thk_all = thk_pred[:, -1, :]
                else:
                    next_thk = thk_pred[:, -1]

                log_probs[~beam_active] = -float("inf")
                log_probs[~beam_active, vocab.PAD] = 0.0

                candidate_scores = beam_scores.unsqueeze(1) + log_probs
                flat_scores = candidate_scores.view(-1)
                topk_scores, topk_flat_idx = flat_scores.topk(K, dim=0)

                V = log_probs.size(1)
                beam_idx  = topk_flat_idx // V
                token_idx = topk_flat_idx % V

                new_mat = torch.cat([beam_mat[beam_idx], token_idx.unsqueeze(1)], dim=1)
                if per_material_thk:
                    chosen_thk = next_thk_all[beam_idx, token_idx].unsqueeze(1)
                else:
                    chosen_thk = next_thk[beam_idx].unsqueeze(1)
                new_thk    = torch.cat([beam_thk[beam_idx], chosen_thk], dim=1)
                new_scores = topk_scores
                new_active = torch.ones(K, dtype=torch.bool, device=device)

                for k in range(K):
                    tok = token_idx[k].item()
                    if tok == vocab.EOS:
                        completed_scores.append(new_scores[k].item())
                        completed_lengths.append(new_mat[k].size(0))
                        completed_mat.append(new_mat[k])
                        completed_thk.append(new_thk[k])
                        new_active[k] = False
                        new_scores[k] = -float("inf")
                    elif tok == vocab.PAD:
                        new_active[k] = False
                        new_scores[k] = -float("inf")

                beam_mat    = new_mat
                beam_thk    = new_thk
                beam_scores = new_scores
                beam_active = new_active

            for k in range(K):
                if beam_active[k]:
                    completed_scores.append(beam_scores[k].item())
                    completed_lengths.append(beam_mat[k].size(0))
                    completed_mat.append(beam_mat[k])
                    completed_thk.append(beam_thk[k])

            if not completed_scores:
                all_results.append([])
                continue

            # Rank all candidates by length-normalized score
            normed = [
                completed_scores[i] / (completed_lengths[i] ** length_penalty)
                for i in range(len(completed_scores))
            ]
            ranked = sorted(range(len(normed)), key=lambda i: normed[i], reverse=True)

            candidates = []
            for idx in ranked[:K]:
                mat_ids = completed_mat[idx].tolist()
                thk_vals = completed_thk[idx].tolist()
                mat_out, thk_out = [], []
                for m, t in zip(mat_ids, thk_vals):
                    if m not in (vocab.PAD, vocab.BOS, vocab.EOS):
                        mat_out.append(m)
                        thk_out.append(float(t))
                candidates.append({
                    "mat_ids": mat_out,
                    "thk_vals": thk_out,
                    "score": normed[idx],
                })

            all_results.append(candidates)

    return all_results
