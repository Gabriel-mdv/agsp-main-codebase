"""
utils/pruning.py
────────────────
Pruning strategies for MobileVLM V2-1.7B attention heads.

  1. AGSP — Alignment-Guided Structural Pruning (prune lowest-AIS heads)
  2. MBP  — Magnitude-Based Pruning (prune lowest L2-norm heads)
  3. RAND — Random Pruning (new baseline for Stage 3 ablation)

Both AGSP and MBP support soft (zero weights) and structural (reshape
matrices) modes. RAND supports structural mode only.

MobileVLM V2 specifics:
  - Layer access via safe_get_llm_layers() — handles flat model.model.layers
  - Config via safe_get_llm_config() — flat config (no text_config wrapper)
  - 24 layers × 16 heads = 384 total heads
  - prune_heads() not used — manual weight slicing throughout

Changes from v3:
  - Fixed zero_pruned_head_gradients: head_dim now computed per-layer from
    current attn.num_heads (after structural pruning) not from original config
  - Added RAND (random pruning) strategy for Stage 3 lower-bound baseline
  - Added measure_flops_reduction() for Appendix D latency table
"""

import logging
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from utils.alignment_scoring import safe_get_llm_config, safe_get_llm_layers

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Magnitude-based scores
# ─────────────────────────────────────────────────────────────────────────────

def _compute_mbp_scores(model, n_layers: int, n_heads: int) -> np.ndarray:
    llm_cfg  = safe_get_llm_config(model)
    head_dim = llm_cfg.hidden_size // n_heads
    scores   = np.zeros((n_layers, n_heads), dtype=np.float32)

    for layer_idx, layer in enumerate(safe_get_llm_layers(model)):
        W_o = layer.self_attn.o_proj.weight.detach().cpu().float()
        for head_idx in range(n_heads):
            start = head_idx * head_dim
            scores[layer_idx, head_idx] = (
                W_o[:, start:start + head_dim].norm(p=2).item()
            )

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Head selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_heads_to_prune(
    scores: np.ndarray,
    pruning_ratio: float,
    per_layer_cap: float,
    seed: int = 42,
    strategy: str = "agsp",
) -> List[Tuple[int, int]]:
    n_layers, n_heads = scores.shape
    target   = int(n_layers * n_heads * pruning_ratio)
    max_each = int(n_heads * per_layer_cap)

    if strategy == "rand":
        # Random selection respecting per-layer cap
        random.seed(seed)
        all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
        random.shuffle(all_heads)
        layer_counts = np.zeros(n_layers, dtype=int)
        pruned: List[Tuple[int, int]] = []
        for layer, head in all_heads:
            if len(pruned) >= target:
                break
            if layer_counts[layer] < max_each:
                pruned.append((layer, head))
                layer_counts[layer] += 1
    else:
        # Score-based selection: ascending = least important first
        flat_order   = np.argsort(scores.flatten())
        layer_counts = np.zeros(n_layers, dtype=int)
        pruned = []
        for flat_idx in flat_order:
            if len(pruned) >= target:
                break
            layer = int(flat_idx // n_heads)
            head  = int(flat_idx %  n_heads)
            if layer_counts[layer] < max_each:
                pruned.append((layer, head))
                layer_counts[layer] += 1

    logger.info(
        "Selected %d / %d heads to prune (target=%d, cap=%d/layer, strategy=%s).",
        len(pruned), n_layers * n_heads, target, max_each, strategy.upper()
    )
    return pruned


# ─────────────────────────────────────────────────────────────────────────────
# Soft pruning
# ─────────────────────────────────────────────────────────────────────────────

def apply_soft_pruning(model, heads_to_prune: List[Tuple[int, int]]) -> None:
    llm_cfg  = safe_get_llm_config(model)
    head_dim = llm_cfg.hidden_size // llm_cfg.num_attention_heads
    layers   = safe_get_llm_layers(model)

    with torch.no_grad():
        for layer_idx, head_idx in heads_to_prune:
            W_o = layers[layer_idx].self_attn.o_proj.weight
            s   = head_idx * head_dim
            W_o[:, s:s + head_dim] = 0.0

    logger.info("Soft pruning applied to %d heads.", len(heads_to_prune))


# ─────────────────────────────────────────────────────────────────────────────
# Structural pruning — reshape Q/K/V/O matrices
# ─────────────────────────────────────────────────────────────────────────────

def apply_structural_pruning(model, heads_to_prune: List[Tuple[int, int]]) -> None:
    """
    Physically remove head parameters by reshaping Q/K/V/O projection matrices.

    For each layer with pruned heads:
      - W_Q, W_K, W_V: remove rows corresponding to pruned heads (out_features)
      - W_O: remove columns corresponding to pruned heads (in_features)
      - Update attn.num_heads and attn.num_key_value_heads
    """
    llm_cfg  = safe_get_llm_config(model)
    n_heads  = llm_cfg.num_attention_heads
    head_dim = llm_cfg.hidden_size // n_heads
    layers   = safe_get_llm_layers(model)

    # Group pruned heads by layer
    layer_prune: Dict[int, Set[int]] = {}
    for layer_idx, head_idx in heads_to_prune:
        layer_prune.setdefault(layer_idx, set()).add(head_idx)

    with torch.no_grad():
        for layer_idx, layer in enumerate(layers):
            if layer_idx not in layer_prune:
                continue

            pruned_set = sorted(layer_prune[layer_idx])
            kept       = [h for h in range(n_heads) if h not in pruned_set]
            if not kept:
                logger.warning("Layer %d: all heads pruned — skipping!", layer_idx)
                continue

            attn = layer.self_attn

            def _keep_rows(W: nn.Linear) -> nn.Linear:
                rows = torch.cat(
                    [W.weight[h * head_dim:(h + 1) * head_dim] for h in kept]
                )
                new = nn.Linear(
                    W.in_features, len(kept) * head_dim,
                    bias=W.bias is not None,
                    device=W.weight.device, dtype=W.weight.dtype,
                )
                new.weight = nn.Parameter(rows)
                if W.bias is not None:
                    new.bias = nn.Parameter(
                        torch.cat(
                            [W.bias[h * head_dim:(h + 1) * head_dim] for h in kept]
                        )
                    )
                return new

            def _keep_cols(W: nn.Linear) -> nn.Linear:
                cols = torch.cat(
                    [W.weight[:, h * head_dim:(h + 1) * head_dim] for h in kept],
                    dim=1,
                )
                new = nn.Linear(
                    len(kept) * head_dim, W.out_features,
                    bias=W.bias is not None,
                    device=W.weight.device, dtype=W.weight.dtype,
                )
                new.weight = nn.Parameter(cols)
                if W.bias is not None:
                    new.bias = nn.Parameter(W.bias.clone())
                return new

            attn.q_proj = _keep_rows(attn.q_proj)
            attn.k_proj = _keep_rows(attn.k_proj)
            attn.v_proj = _keep_rows(attn.v_proj)
            attn.o_proj = _keep_cols(attn.o_proj)

            attn.num_heads = len(kept)
            if hasattr(attn, "num_key_value_heads"):
                attn.num_key_value_heads = len(kept)

            logger.info(
                "Layer %d: removed %d heads, kept %d.",
                layer_idx, len(pruned_set), len(kept)
            )

    logger.info("Structural pruning complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

def prune_model(
    model,
    ais_matrix: Optional[np.ndarray],
    pruning_ratio: float,
    per_layer_cap: float = 0.50,
    strategy: str = "agsp",
    pruning_mode: str = "structural",
    seed: int = 42,
) -> Tuple[object, List[Tuple[int, int]]]:
    """
    Prune model attention heads using the specified strategy and mode.

    Parameters
    ----------
    model          : MobileVLM V2 (bfloat16, eager attention)
    ais_matrix     : (n_layers × n_heads) AIS scores — required for AGSP
    pruning_ratio  : fraction of total heads to remove
    per_layer_cap  : max fraction of heads to remove per layer (default 0.50)
    strategy       : 'agsp' | 'mbp' | 'rand'
    pruning_mode   : 'structural' | 'soft'
    seed           : random seed (used for RAND strategy)
    """
    llm_cfg  = safe_get_llm_config(model)
    n_layers = llm_cfg.num_hidden_layers
    n_heads  = llm_cfg.num_attention_heads

    if strategy == "agsp":
        if ais_matrix is None:
            raise ValueError("AIS matrix required for AGSP strategy.")
        scores = ais_matrix
    elif strategy == "mbp":
        scores = _compute_mbp_scores(model, n_layers, n_heads)
    elif strategy == "rand":
        # Scores unused for random — pass zeros as placeholder
        scores = np.zeros((n_layers, n_heads), dtype=np.float32)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'agsp', 'mbp', or 'rand'.")

    heads_to_prune = _select_heads_to_prune(
        scores, pruning_ratio, per_layer_cap, seed=seed, strategy=strategy
    )

    logger.info(
        "[%s / %s] Pruning %d heads (%.1f%% of %d total).",
        strategy.upper(), pruning_mode,
        len(heads_to_prune),
        100.0 * len(heads_to_prune) / (n_layers * n_heads),
        n_layers * n_heads,
    )

    if pruning_mode == "soft":
        apply_soft_pruning(model, heads_to_prune)
    elif pruning_mode == "structural":
        apply_structural_pruning(model, heads_to_prune)
    else:
        raise ValueError(f"Unknown pruning_mode: {pruning_mode!r}. Use 'structural' or 'soft'.")

    return model, heads_to_prune


# ─────────────────────────────────────────────────────────────────────────────
# Gradient zeroing for fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def zero_pruned_head_gradients(
    model,
    pruned_heads: List[Tuple[int, int]],
) -> None:
    """
    Zero gradients for pruned heads after backward pass during fine-tuning.

    FIX from v3: head_dim is now computed per-layer from the current
    attn.num_heads (which may have changed after structural pruning),
    NOT from the original config value. Using the original config value
    after structural pruning causes incorrect index calculations.
    """
    layers = safe_get_llm_layers(model)

    # Group pruned heads by layer for efficient per-layer processing
    layer_heads: Dict[int, List[int]] = {}
    for layer_idx, head_idx in pruned_heads:
        layer_heads.setdefault(layer_idx, []).append(head_idx)

    for layer_idx, head_list in layer_heads.items():
        layer   = layers[layer_idx]
        W_o     = layer.self_attn.o_proj.weight

        # Compute head_dim from current layer dimensions (post-pruning safe)
        current_n_heads = layer.self_attn.num_heads
        current_in_dim  = W_o.shape[1]
        head_dim        = current_in_dim // current_n_heads

        if W_o.grad is not None:
            for head_idx in head_list:
                s = head_idx * head_dim
                if s + head_dim <= W_o.grad.shape[1]:
                    W_o.grad[:, s:s + head_dim] = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Measurement utilities
# ─────────────────────────────────────────────────────────────────────────────

def measure_sparsity(
    model,
    pruned_heads: List[Tuple[int, int]],
) -> Dict:
    llm_cfg = safe_get_llm_config(model)
    total   = llm_cfg.num_hidden_layers * llm_cfg.num_attention_heads
    return {
        "heads_pruned":    len(pruned_heads),
        "total_heads":     total,
        "sparsity_pct":    100.0 * len(pruned_heads) / total,
        "layers_affected": len(set(l for l, _ in pruned_heads)),
    }


def measure_flops_reduction(
    model_original,
    model_pruned,
    seq_len: int = 256,
) -> Dict:
    """
    Estimate attention FLOPs reduction from structural head pruning.

    Attention FLOPs ≈ 4 * seq_len^2 * n_heads * head_dim per layer
    (QK^T + softmax + AV + projection)

    Returns absolute and percentage FLOPs reduction.
    """
    llm_cfg_orig   = safe_get_llm_config(model_original)
    n_layers       = llm_cfg_orig.num_hidden_layers
    orig_heads     = llm_cfg_orig.num_attention_heads
    orig_head_dim  = llm_cfg_orig.hidden_size // orig_heads

    # Count remaining heads per layer in pruned model
    pruned_layers  = safe_get_llm_layers(model_pruned)
    total_orig_attn_flops   = 0
    total_pruned_attn_flops = 0

    for layer_idx in range(n_layers):
        orig_flops   = 4 * seq_len ** 2 * orig_heads * orig_head_dim
        pruned_heads = pruned_layers[layer_idx].self_attn.num_heads
        pruned_flops = 4 * seq_len ** 2 * pruned_heads * orig_head_dim

        total_orig_attn_flops   += orig_flops
        total_pruned_attn_flops += pruned_flops

    reduction_abs = total_orig_attn_flops - total_pruned_attn_flops
    reduction_pct = 100.0 * reduction_abs / total_orig_attn_flops

    return {
        "orig_attn_flops":    total_orig_attn_flops,
        "pruned_attn_flops":  total_pruned_attn_flops,
        "flops_reduction_abs": reduction_abs,
        "flops_reduction_pct": reduction_pct,
    }
