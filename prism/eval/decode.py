"""Autoregressive decoding strategies for the InverseModel."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from prism.constants import THK_MIN, THK_MAX


def greedy_decode(
    model,
    spectrum: Tensor,       # [B, 142]
    vocab,
    max_len: int = 101,      # max sequence length including BOS
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
            thk_pred = model.thk_to_nm(thk_pred)

            next_mat = mat_logits[:, -1, :].argmax(dim=-1)  # [B]
            # Per-material thickness: index by chosen material
            next_thk = thk_pred[:, -1, :].gather(-1, next_mat.unsqueeze(-1)).squeeze(-1)  # [B]

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
    max_len: int = 101,
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
                thk_pred = model.thk_to_nm(thk_pred)

                log_probs = F.log_softmax(mat_logits[:, -1, :], dim=-1)  # [K, V]
                next_thk_all = thk_pred[:, -1, :]  # [K, V] — thickness per material

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
                chosen_thk = next_thk_all[beam_idx, token_idx].unsqueeze(1)  # [K, 1]
                new_thk    = torch.cat([beam_thk[beam_idx], chosen_thk], dim=1)
                new_scores = topk_scores
                new_active = torch.ones(K, dtype=torch.bool, device=device)

                # Check for EOS / PAD
                for k in range(K):
                    tok = token_idx[k].item()
                    if tok == vocab.EOS:
                        n_scored = step + 1  # number of tokens that contributed log-probs
                        completed_scores.append(new_scores[k].item())
                        completed_lengths.append(n_scored)
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
                    n_scored = beam_mat[k].size(0) - 1  # exclude BOS
                    completed_scores.append(beam_scores[k].item())
                    completed_lengths.append(n_scored)
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
    max_len: int = 101,
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
                thk_pred = model.thk_to_nm(thk_pred)

                log_probs = F.log_softmax(mat_logits[:, -1, :], dim=-1)
                next_thk_all = thk_pred[:, -1, :]

                log_probs[~beam_active] = -float("inf")
                log_probs[~beam_active, vocab.PAD] = 0.0

                candidate_scores = beam_scores.unsqueeze(1) + log_probs
                flat_scores = candidate_scores.view(-1)
                topk_scores, topk_flat_idx = flat_scores.topk(K, dim=0)

                V = log_probs.size(1)
                beam_idx  = topk_flat_idx // V
                token_idx = topk_flat_idx % V

                new_mat = torch.cat([beam_mat[beam_idx], token_idx.unsqueeze(1)], dim=1)
                chosen_thk = next_thk_all[beam_idx, token_idx].unsqueeze(1)
                new_thk    = torch.cat([beam_thk[beam_idx], chosen_thk], dim=1)
                new_scores = topk_scores
                new_active = torch.ones(K, dtype=torch.bool, device=device)

                for k in range(K):
                    tok = token_idx[k].item()
                    if tok == vocab.EOS:
                        completed_scores.append(new_scores[k].item())
                        completed_lengths.append(step + 1)  # tokens that contributed log-probs
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
                    completed_lengths.append(beam_mat[k].size(0) - 1)  # exclude BOS
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


# ── Stochastic sampling for RLVR ─────────────────────────────────────────────

def sample_decode(
    model,
    spectrum: Tensor,               # [B, 142]
    vocab,
    n_samples: int = 8,             # G rollouts per spectrum
    temperature: float = 1.0,       # material sampling temperature
    thk_noise_std: float = 5.0,     # Gaussian noise σ for thickness (nm)
    max_len: int = 101,              # max sequence length including BOS
    device: torch.device | None = None,
) -> dict:
    """
    Stochastic autoregressive decoding for GRPO / RLVR.

    Generates n_samples rollouts per input spectrum. Materials are sampled from
    Categorical(softmax(logits / temperature)); thicknesses are perturbed with
    additive Gaussian noise and clamped to [THK_MIN, THK_MAX].

    Runs WITH gradients so log-probabilities can be back-propagated through the
    policy gradient loss.

    Returns dict with:
        mat_ids:         list[B*G] of list[int]   — material ID sequences (no BOS/EOS)
        thk_vals:        list[B*G] of list[float]  — thickness sequences (nm)
        mat_log_probs:   Tensor [B*G]             — sum of per-step material log-probs
        thk_log_probs:   Tensor [B*G]             — sum of per-step thickness log-probs
        total_log_probs: Tensor [B*G]             — mat + thk combined
        group_ids:       Tensor [B*G] long        — spectrum index each rollout belongs to
        lengths:         Tensor [B*G] long        — number of material tokens (excl BOS/EOS)
    """
    if device is None:
        device = next(model.parameters()).device

    B = spectrum.size(0)
    G = n_samples
    N = B * G  # total rollouts

    # Expand spectrum: each spectrum repeated G times  [B*G, 142]
    spec_expanded = spectrum.to(device).unsqueeze(1).expand(B, G, -1).reshape(N, -1)

    # Group IDs: [0,0,...,0, 1,1,...,1, ..., B-1,B-1,...,B-1]
    group_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, G).reshape(N)

    # Initialize sequences with BOS
    mat_seqs = torch.full((N, 1), vocab.BOS, dtype=torch.long, device=device)
    thk_seqs = torch.zeros(N, 1, device=device)

    finished = torch.zeros(N, dtype=torch.bool, device=device)

    # Accumulators for per-step log-probs (as lists, summed at end)
    step_mat_lps: list[Tensor] = []   # each [N] (zero for finished rollouts)
    step_thk_lps: list[Tensor] = []

    mat_results: list[list[int]]   = [[] for _ in range(N)]
    thk_results: list[list[float]] = [[] for _ in range(N)]

    log_norm_const = math.log(thk_noise_std * math.sqrt(2.0 * math.pi))

    for _ in range(max_len - 1):
        if finished.all():
            break

        T = mat_seqs.size(1)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        tgt_mask = causal.unsqueeze(0).expand(N, -1, -1)

        mat_logits, thk_pred = model(spec_expanded, mat_seqs, thk_seqs, tgt_mask)
        thk_pred = model.thk_to_nm(thk_pred)
        # mat_logits: [N, T, V],  thk_pred: [N, T, V] (per-material), in nm

        # ── Material sampling ──
        logits_last = mat_logits[:, -1, :]  # [N, V]
        # Mask PAD token (index 0) to prevent sampling it
        logits_last[:, vocab.PAD] = -float("inf")
        # For finished rollouts, force PAD
        logits_last[finished] = -float("inf")
        logits_last[finished, vocab.PAD] = 0.0

        scaled_logits = logits_last / temperature
        log_probs = F.log_softmax(scaled_logits, dim=-1)  # [N, V]
        dist = Categorical(logits=scaled_logits)
        next_mat = dist.sample()                           # [N]

        # Log-prob of sampled material
        mat_lp = log_probs.gather(-1, next_mat.unsqueeze(-1)).squeeze(-1)  # [N]
        mat_lp = mat_lp * (~finished).float()  # zero out finished
        step_mat_lps.append(mat_lp)

        # ── Thickness sampling ──
        thk_mean = thk_pred[:, -1, :].gather(-1, next_mat.unsqueeze(-1)).squeeze(-1)  # [N]

        # Reparameterized sampling: thk = mean + σ * ε  (always in nm)
        eps = torch.randn_like(thk_mean)
        thk_sampled = thk_mean + thk_noise_std * eps
        thk_sampled = thk_sampled.clamp(THK_MIN, THK_MAX)

        # Gaussian log-prob (using unclamped for proper gradient flow)
        thk_lp = -0.5 * ((thk_sampled.detach() - thk_mean) / thk_noise_std) ** 2 - log_norm_const
        thk_lp = thk_lp * (~finished).float()
        step_thk_lps.append(thk_lp)

        # For finished rollouts, override to PAD/0
        next_mat = torch.where(finished, torch.full_like(next_mat, vocab.PAD), next_mat)
        thk_sampled = torch.where(finished, torch.zeros_like(thk_sampled), thk_sampled)

        # ── Collect results ──
        for i in range(N):
            if not finished[i]:
                tok = next_mat[i].item()
                if tok == vocab.EOS:
                    finished[i] = True
                elif tok != vocab.PAD:
                    mat_results[i].append(tok)
                    thk_results[i].append(float(thk_sampled[i].item()))

        # Check if newly-generated EOS
        newly_eos = (next_mat == vocab.EOS) & (~finished)
        finished = finished | (next_mat == vocab.EOS)

        # Extend sequences
        mat_seqs = torch.cat([mat_seqs, next_mat.unsqueeze(1)], dim=1)
        thk_seqs = torch.cat([thk_seqs, thk_sampled.unsqueeze(1)], dim=1)

    # Sum log-probs across steps
    mat_log_probs = torch.stack(step_mat_lps, dim=1).sum(dim=1)    # [N]
    thk_log_probs = torch.stack(step_thk_lps, dim=1).sum(dim=1)    # [N]
    total_log_probs = mat_log_probs + thk_log_probs                 # [N]

    lengths = torch.tensor([len(m) for m in mat_results], dtype=torch.long, device=device)

    return {
        "mat_ids": mat_results,
        "thk_vals": thk_results,
        "mat_log_probs": mat_log_probs,
        "thk_log_probs": thk_log_probs,
        "total_log_probs": total_log_probs,
        "group_ids": group_ids,
        "lengths": lengths,
    }
