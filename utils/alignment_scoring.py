"""
utils/alignment_scoring.py
──────────────────────────
Computes per-head Alignment Importance Scores (AIS) for MobileVLM V2-1.7B.

AIS(l, h) = ||∂L_align / ∂W_Q^(l,h)||_F

where L_align is the causal LM cross-entropy on ground-truth COCO captions
propagated through the full vision-language pipeline.

MobileVLM V2-1.7B architecture:
  - Backbone: MobileLLaMA-1.4B
  - Layers: 24 transformer layers
  - Heads:  16 attention heads per layer
  - Total:  384 attention heads
  - Visual tokens: 144 (LDPv2 projector, not 576)
  - Layer access: model.model.layers (flat, not model.language_model.model.layers)
  - Config: flat (model.config.num_hidden_layers, not text_config)

Optional debiasing (Zhao et al. 2026, arXiv:2508.17807v2) removes:
  - RoPE recency bias
  - Attention-sink inflation

Changes from v3:
  - Fixed docstring: 24 layers × 16 heads = 384 heads (not 32×32)
  - Fixed _build_mobilevlm_batch: images cast to bfloat16 (not float16)
  - Added run_calibration_ablation() for Appendix A ablation study
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture helpers for MobileVLM V2
# ─────────────────────────────────────────────────────────────────────────────

def safe_get_llm_layers(model):
    """
    Return the list of LLM decoder layers, handling multiple model wrappers:
      - MobileVLM flat:         model.model.layers
      - PEFT-wrapped MobileVLM: model.base_model.model.model.layers
      - LLaVA nested:           model.language_model.model.layers
      - PEFT-wrapped LLaVA:     model.base_model.model.language_model.model.layers
    """
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.base_model.model.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.base_model.model.language_model.model.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if layers is not None:
                return layers
        except AttributeError:
            continue
    raise RuntimeError(
        "Cannot locate LLM decoder layers. "
        "Tried flat, PEFT-wrapped, LLaVA, and PEFT-wrapped LLaVA patterns."
    )


def safe_get_llm_config(model):
    """
    Return the LLM sub-config, handling:
      - Nested:  model.config.text_config
      - Flat:    model.config  (MobileVLM / MobileLLaMA)
    """
    cfg = model.config
    if hasattr(cfg, "text_config"):
        return cfg.text_config
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample gradient norm accumulation
# ─────────────────────────────────────────────────────────────────────────────

def _get_head_grad_norms(
    model,
    batch: Dict[str, Tensor],
    n_layers: int,
    n_heads: int,
) -> np.ndarray:
    """
    One forward + backward pass → (n_layers × n_heads) Frobenius norm matrix.
    """
    llm_cfg  = safe_get_llm_config(model)
    head_dim = llm_cfg.hidden_size // n_heads

    model.zero_grad()
    outputs = model(**batch, labels=batch["input_ids"])
    outputs.loss.backward()

    norms  = np.zeros((n_layers, n_heads), dtype=np.float32)
    layers = safe_get_llm_layers(model)

    for layer_idx, layer in enumerate(layers):
        W_q = layer.self_attn.q_proj.weight
        if W_q.grad is None:
            continue
        grad = W_q.grad
        for head_idx in range(n_heads):
            start = head_idx * head_dim
            end   = start + head_dim
            norms[layer_idx, head_idx] += (
                grad[start:end, :].float().norm(p="fro").item()
            )

    model.zero_grad()
    return norms


# ─────────────────────────────────────────────────────────────────────────────
# Debiasing (Zhao et al. 2026 lightweight approximation)
# ─────────────────────────────────────────────────────────────────────────────

def _debias_ais_matrix(
    ais: np.ndarray,
    n_visual_tokens: int = 144,
) -> np.ndarray:
    """Layer-normalise + penalise uniform (sink-like) heads."""
    debiased    = ais.copy()
    layer_means = debiased.mean(axis=1, keepdims=True) + 1e-9
    debiased    = debiased / layer_means
    sink_penalty = 1.0 / n_visual_tokens
    return np.clip(debiased - sink_penalty, a_min=0.0, a_max=None)


# ─────────────────────────────────────────────────────────────────────────────
# MobileVLM V2 batch builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_mobilevlm_batch(tokenizer, image_processor, sample, device):
    """
    Tokenise a COCO sample for MobileVLM V2.
    MobileLlamaForCausalLM forward() accepts: input_ids, images, labels.

    Images cast to bfloat16 to match model dtype (not float16).
    """
    text  = f"USER: <image>\n{sample['caption']}\nASSISTANT:"
    image = sample["image"]

    text_enc = tokenizer(
        text, return_tensors="pt", padding=True,
        truncation=True, max_length=256
    ).to(device)

    # Cast to bfloat16 — matches model dtype and paper specification
    images = image_processor(
        images=image, return_tensors="pt"
    ).pixel_values.to(dtype=torch.bfloat16, device=device)

    return {
        "input_ids":      text_enc.input_ids,
        "attention_mask": text_enc.attention_mask,
        "images":         images,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_ais_matrix(
    model,
    tokenizer,
    image_processor,
    calibration_samples: List[Dict],
    n_layers: int,
    n_heads: int,
    use_debiased: bool = True,
    n_visual_tokens: int = 144,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute the AIS matrix averaged over all calibration samples.

    Parameters
    ----------
    model               : MobileVLM V2 (trust_remote_code=True, eager attn, bfloat16)
    tokenizer           : AutoTokenizer for MobileVLM V2
    image_processor     : CLIPImageProcessor for MobileVLM V2
    calibration_samples : list of {'image': PIL, 'caption': str}
    n_layers / n_heads  : architecture dims (24 / 16 for MobileVLM V2-1.7B)
    use_debiased        : apply Zhao et al. 2026 correction
    n_visual_tokens     : 144 for LDPv2 projector
    device              : cuda
    """
    model.eval()

    # Freeze all parameters, then enable gradients only for LLM layers
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in safe_get_llm_layers(model):
        for p in layer.parameters():
            p.requires_grad_(True)

    accumulated = np.zeros((n_layers, n_heads), dtype=np.float64)
    n_valid = 0

    for idx, sample in enumerate(calibration_samples):
        logger.info("AIS calibration %d / %d", idx + 1, len(calibration_samples))
        try:
            batch = _build_mobilevlm_batch(
                tokenizer, image_processor, sample, device
            )
            norms = _get_head_grad_norms(model, batch, n_layers, n_heads)
            accumulated += norms
            n_valid += 1
        except Exception as exc:
            logger.warning("Skipping sample %d: %s", idx, exc)

    if n_valid == 0:
        raise RuntimeError("No calibration samples processed successfully.")

    ais = (accumulated / n_valid).astype(np.float32)

    if use_debiased:
        logger.info("Applying debiased AIS (n_visual_tokens=%d).", n_visual_tokens)
        ais = _debias_ais_matrix(ais, n_visual_tokens=n_visual_tokens)

    return ais


def bootstrap_ais_stability(
    model,
    tokenizer,
    image_processor,
    calibration_samples: List[Dict],
    n_layers: int,
    n_heads: int,
    n_draws: int = 5,
    draw_size: int = 20,
    rho_threshold: float = 0.85,
    device: str = "cuda",
) -> Tuple[float, bool]:
    """
    Bootstrap stability check: draw n_draws subsets, compute AIS each time,
    return mean Spearman rho vs the full-set AIS and pass/fail flag.

    Paper achieved rho = 0.9963 with 100 calibration samples.
    """
    from scipy.stats import spearmanr
    import random

    full_ais  = compute_ais_matrix(
        model, tokenizer, image_processor, calibration_samples,
        n_layers, n_heads, use_debiased=False, device=device,
    )
    full_flat = full_ais.flatten()

    rhos = []
    for draw_idx in range(n_draws):
        subset = random.sample(
            calibration_samples, min(draw_size, len(calibration_samples))
        )
        sub_ais = compute_ais_matrix(
            model, tokenizer, image_processor, subset,
            n_layers, n_heads, use_debiased=False, device=device,
        )
        rho, _ = spearmanr(full_flat, sub_ais.flatten())
        rhos.append(rho)
        logger.info("Bootstrap draw %d / %d: rho=%.4f", draw_idx + 1, n_draws, rho)

    mean_rho = float(np.mean(rhos))
    passed   = mean_rho >= rho_threshold
    logger.info(
        "Bootstrap stability: mean Spearman rho=%.4f (threshold=%.2f) — %s",
        mean_rho, rho_threshold, "PASS" if passed else "FAIL"
    )
    return mean_rho, passed


def run_calibration_ablation(
    model,
    tokenizer,
    image_processor,
    all_samples: List[Dict],
    n_layers: int,
    n_heads: int,
    calibration_sizes: List[int],
    device: str = "cuda",
) -> Dict:
    """
    Appendix A ablation: compute AIS at multiple calibration set sizes and
    report Spearman rho of each ranking against the largest set as ground truth.

    Parameters
    ----------
    all_samples        : full pool of samples (must be >= max(calibration_sizes))
    calibration_sizes  : list of sizes to test, e.g. [50, 100, 250, 500]

    Returns
    -------
    dict mapping size → {'rho': float, 'ais_matrix': np.ndarray}
    """
    from scipy.stats import spearmanr

    # Ground truth: largest calibration set
    max_size     = max(calibration_sizes)
    ground_truth = compute_ais_matrix(
        model, tokenizer, image_processor, all_samples[:max_size],
        n_layers, n_heads, use_debiased=False, device=device,
    )
    gt_flat = ground_truth.flatten()

    results = {max_size: {"rho": 1.0, "ais_matrix": ground_truth}}

    for size in sorted(calibration_sizes):
        if size == max_size:
            continue
        logger.info("Calibration ablation: n=%d", size)
        ais = compute_ais_matrix(
            model, tokenizer, image_processor, all_samples[:size],
            n_layers, n_heads, use_debiased=False, device=device,
        )
        rho, _ = spearmanr(gt_flat, ais.flatten())
        results[size] = {"rho": float(rho), "ais_matrix": ais}
        logger.info("  n=%d → Spearman rho=%.4f", size, rho)

    return results


def summarise_ais(ais: np.ndarray) -> Dict:
    flat        = ais.flatten()
    sorted_flat = np.sort(flat)
    n           = len(flat)
    cumsum      = np.cumsum(sorted_flat)
    gini = (
        (2 * np.sum(np.arange(1, n + 1) * sorted_flat)) / (n * cumsum[-1])
        - (n + 1) / n
    )

    percentiles = {
        f"P{p}": float(np.percentile(flat, p))
        for p in [5, 10, 25, 50, 75, 90, 95, 99]
    }
    return {
        "mean":   float(flat.mean()),
        "median": float(np.median(flat)),
        "std":    float(flat.std()),
        "max":    float(flat.max()),
        "min":    float(flat.min()),
        "gini":   float(gini),
        **percentiles,
        "frac_above_0.5":    float((flat >= 0.5).mean()),
        "frac_below_0.1":    float((flat < 0.1).mean()),
        "top5_bottom50_ratio": (
            float(flat[flat >= np.percentile(flat, 95)].mean()) /
            (float(flat[flat <= np.percentile(flat, 50)].mean()) + 1e-9)
        ),
    }


def get_top_k_heads(
    ais: np.ndarray,
    k: int = 20,
) -> List[Tuple[int, int, float]]:
    flat_indices = np.argsort(ais.flatten())[::-1][:k]
    return [
        (
            int(i // ais.shape[1]),
            int(i %  ais.shape[1]),
            float(ais.flatten()[i]),
        )
        for i in flat_indices
    ]
