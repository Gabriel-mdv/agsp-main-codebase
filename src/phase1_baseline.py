#!/usr/bin/env python3
"""
phase1_baseline.py
──────────────────
Phase 1: Baseline Analysis and Alignment Importance Scoring — MobileVLM V2-1.7B.

PURPOSE
───────
Compute the Alignment Importance Score (AIS) for every attention head in
MobileVLM V2-1.7B. AIS measures how causally important each head is for
cross-modal grounding, as opposed to weight magnitude which measures how
large the weights are (used by MBP). This is the core methodological
contribution of the AGSP paper.

AIS formula (Equation 1 in paper):
  AIS(l, h) = || ∂L_align / ∂W_Q^(l,h) ||_F

where L_align is the causal LM cross-entropy on 100 COCO image-caption pairs
propagated through the FULL vision-language pipeline (not just the LLM).

ARCHITECTURE (MobileVLM V2-1.7B)
─────────────────────────────────
  Vision encoder : CLIP ViT-L/14 (frozen)
  Projector      : LDPv2 — 576 → 144 visual tokens (4× fewer than LLaVA)
  Language model : MobileLLaMA-1.4B
                   24 layers × 16 heads = 384 total attention heads
                   head_dim = 128

OUTPUTS
───────
  outputs/phase1/ais_matrix.npy       — (24 × 16) float32 AIS scores
  outputs/phase1/ais_summary.json     — statistics: gini, mean, top5/bottom50 ratio
  outputs/phase1/top20_heads.json     — ranked list of top-20 alignment-critical heads
  outputs/phase1/calibration_ablation.json  — Appendix A: rho vs calibration size
  outputs/phase1/fig_ais_distribution.png
  outputs/phase1/fig_layer_profile.png
  outputs/phase1/fig_heatmap_sample_{i}.png

USAGE
─────
  # Standard run (100 calibration samples, matches paper)
  python phase1_baseline.py

  # Custom sample count
  python phase1_baseline.py --n_samples 100

  # Run calibration ablation study (Appendix A)
  python phase1_baseline.py --run_ablation

  # Smoke test (10 samples, no bootstrap)
  python phase1_baseline.py --n_samples 10 --no_bootstrap

  # Disable Zhao et al. 2026 debiasing
  python phase1_baseline.py --no_debias

CHANGES FROM v3
───────────────
  - Removed load_in_8bit — model loaded in bfloat16 (matches paper)
  - Added --run_ablation flag for Appendix A calibration set size study
  - Fixed architecture comment: 24 layers × 16 heads (was wrong 32 × 32)
  - Dynamic snapshot detection replaces hardcoded hash path
  - Explicit bfloat16 cast when loading model
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

from mobilevlm.model.mobilellama import MobileLlamaForCausalLM


# Add project root to path so utils imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import get_config
from utils.alignment_scoring import (
    bootstrap_ais_stability,
    compute_ais_matrix,
    get_top_k_heads,
    run_calibration_ablation,
    summarise_ais,
)
from utils.data_utils import load_coco_calibration_samples
from utils.visualization import (
    plot_ais_distribution,
    plot_attention_heatmap,
    plot_layer_ais_profile,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_processors(cfg):
    """
    Load MobileVLM V2-1.7B in bfloat16 with eager attention.

    Why bfloat16 and not 8-bit?
    ───────────────────────────
    The paper states all models were loaded in bfloat16. 8-bit quantisation
    changes gradient magnitudes during backpropagation, which would alter the
    AIS scores and produce results inconsistent with the paper.

    Why eager attention?
    ────────────────────
    Flash attention does not expose intermediate attention weights for gradient
    computation. eager mode is required for Phase 1 gradient backpropagation.
    Phase 2 and 3 can use flash attention if needed, but Phase 1 cannot.

    Why trust_remote_code?
    ──────────────────────
    MobileVLM V2 uses a custom MobileLlamaForCausalLM architecture not
    registered in standard transformers. trust_remote_code allows the model
    to load its own class definition from the HuggingFace hub.

    Returns
    -------
    model            : MobileLlamaForCausalLM in bfloat16
    tokenizer        : AutoTokenizer
    image_processor  : CLIPImageProcessor
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor

    logger.info("Loading MobileVLM V2-1.7B from: %s", cfg.paths.model_id)
    logger.info("dtype=bfloat16, attn=eager, trust_remote_code=True")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.paths.model_id,
        cache_dir=cfg.paths.model_cache_dir,
        trust_remote_code=cfg.hardware.trust_remote_code,
        use_fast=False,
    )

    image_processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-large-patch14-336",
    cache_dir=cfg.paths.model_cache_dir,
)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.paths.model_id,
        torch_dtype=torch.bfloat16,        # bfloat16 — matches paper, not 8-bit
        attn_implementation=cfg.hardware.attn_implementation,  # "eager" for gradients
        cache_dir=cfg.paths.model_cache_dir,
        trust_remote_code=cfg.hardware.trust_remote_code,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    logger.info("MobileVLM V2-1.7B loaded successfully.")
    return model, tokenizer, image_processor


# ─────────────────────────────────────────────────────────────────────────────
# Attention heatmap extraction (visualisation only — not used for AIS)
# ─────────────────────────────────────────────────────────────────────────────

def extract_attention_heatmap(model, tokenizer, image_processor, sample, device):
    """
    Extract text→visual cross-attention heatmap for one sample.

    Returns (n_text_tokens × n_visual_tokens) numpy array.

    NOTE: This function is used ONLY for visualisation (fig_heatmap_sample_*.png).
    It is NOT used to compute AIS. AIS uses gradient Frobenius norms from
    compute_ais_matrix(), not raw attention weights.

    Why not use attention weights for AIS?
    ──────────────────────────────────────
    Raw attention weights are susceptible to RoPE recency bias and attention
    sink effects (Zhao et al. 2026). A head may show high attention to visual
    tokens due to positional artefacts rather than semantic importance.
    Gradient norms measure causal sensitivity which is robust to these biases.

    Architecture note:
    ──────────────────
    MobileVLM V2 uses 144 visual tokens (LDPv2 projector: 576 → 144).
    LLaVA uses 576. This 4× reduction is why MobileVLM V2 fits on edge devices.
    """
    text = f"USER: <image>\n{sample['caption']}\nASSISTANT:"
    enc  = tokenizer(text, return_tensors="pt").to(device)
    pv   = image_processor(
        images=sample["image"], return_tensors="pt"
    ).pixel_values.to(dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        out = model(**enc, images=pv, output_attentions=True)

    # Use last 4 layers for a cleaner high-level representation
    attentions = out.attentions[-4:]
    stacked = torch.stack([a[0].float() for a in attentions]) # (4, heads, seq, seq)
    avg_attn   = stacked.mean(dim=0).mean(dim=0)           # (seq, seq)

    n_visual = 144   # LDPv2 projector — 4× fewer than LLaVA's 576
    n_total  = avg_attn.shape[0]
    n_text   = max(n_total - n_visual, 1)

    # Slice text rows × visual columns for cross-modal heatmap
    heatmap = avg_attn[:n_text, n_total - n_visual:].cpu().numpy()
    return heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    cfg    = get_config()
    torch.manual_seed(cfg.hardware.seed)
    device = cfg.hardware.device

    # ── Step 1: Load model ───────────────────────────────────────────────────
    model, tokenizer, image_processor = load_model_and_processors(cfg)

    # Verify architecture matches paper expectations
    llm_cfg  = (model.config if not hasattr(model.config, "text_config")
                else model.config.text_config)
    n_layers = llm_cfg.num_hidden_layers
    n_heads  = llm_cfg.num_attention_heads
    logger.info(
        "Architecture confirmed: %d layers × %d heads = %d total heads.",
        n_layers, n_heads, n_layers * n_heads,
    )
    # Sanity check — paper uses MobileVLM V2-1.7B with 24 layers × 16 heads
    if n_layers != 24 or n_heads != 16:
        logger.warning(
            "Unexpected architecture: expected 24 × 16 = 384 heads. "
            "Got %d × %d = %d. Continuing but verify model ID.",
            n_layers, n_heads, n_layers * n_heads,
        )

    # ── Step 2: Load calibration samples ────────────────────────────────────
    # Paper: 100 MS-COCO val2017 image-caption pairs
    # These are passed through the full vision-language pipeline to compute
    # gradients that reflect cross-modal alignment sensitivity.
    n_samples = args.n_samples or cfg.phase1.n_calibration_samples

    # For ablation, load the maximum needed (500 pairs)
    n_load = (max(cfg.phase1.calibration_sizes_ablation)
              if args.run_ablation else n_samples)

    all_samples = load_coco_calibration_samples(
        cfg.paths.coco_val_annotations,
        cfg.paths.coco_val_images_dir,
        n=n_load,
        seed=cfg.hardware.seed,
    )
    calibration_samples = all_samples[:n_samples]

    # ── Step 3: Bootstrap stability check ───────────────────────────────────
    # Verifies that AIS rankings are consistent across different random subsets
    # of the calibration data. Paper achieved Spearman rho = 0.9963.
    # Threshold: rho >= 0.85 (if lower, increase n_samples).
    if not args.no_bootstrap:
        logger.info("Running bootstrap AIS stability check (5 draws × 20 samples)...")
        rho, passed = bootstrap_ais_stability(
            model, tokenizer, image_processor, calibration_samples,
            n_layers, n_heads, device=device,
        )
        if not passed:
            logger.warning(
                "Bootstrap Spearman rho=%.4f < 0.85 threshold. "
                "Consider increasing --n_samples to 150+.", rho
            )
        else:
            logger.info("Bootstrap stability PASSED: rho=%.4f", rho)

    # ── Step 4: Compute full AIS matrix ─────────────────────────────────────
    # Core Phase 1 computation. Backpropagates alignment loss through the
    # full vision-language pipeline and accumulates gradient Frobenius norms.
    # Debiasing (Zhao et al. 2026) removes RoPE recency and attention-sink bias.
    use_debias = not args.no_debias and cfg.phase1.use_debiased_scoring

    logger.info(
        "Computing AIS matrix (%d samples, debiased=%s)...",
        n_samples, use_debias,
    )
    ais = compute_ais_matrix(
        model, tokenizer, image_processor, calibration_samples,
        n_layers=n_layers, n_heads=n_heads,
        use_debiased=use_debias,
        n_visual_tokens=cfg.phase1.n_visual_tokens,
        device=device,
    )

    # ── Step 5: Save AIS matrix and summary ─────────────────────────────────
    np.save(cfg.paths.ais_matrix_path, ais)
    logger.info("AIS matrix saved → %s  (shape: %s)", cfg.paths.ais_matrix_path, ais.shape)

    # Compute summary statistics (Gini, percentiles, top5/bottom50 ratio)
    stats = summarise_ais(ais)
    top_k = get_top_k_heads(ais, k=cfg.phase1.top_k_heads)

    with open(os.path.join(cfg.paths.phase1_dir, "ais_summary.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(cfg.paths.phase1_dir, "top20_heads.json"), "w") as f:
        json.dump(
            [{"rank": i + 1, "layer": l, "head": h, "ais": v}
             for i, (l, h, v) in enumerate(top_k)],
            f, indent=2,
        )

    # Log key statistics for quick verification against paper values
    logger.info("=== AIS Summary (compare with paper Table 1) ===")
    logger.info("  Gini coefficient  : %.4f  (paper: 0.232)", stats["gini"])
    logger.info("  Mean AIS          : %.4f  (paper: 0.993)", stats["mean"])
    logger.info("  Median AIS        : %.4f  (paper: 0.943)", stats["median"])
    logger.info("  Top5/Bottom50 ratio: %.2f×  (paper: 3.3×)", stats["top5_bottom50_ratio"])
    logger.info("  Most critical head: L%dH%d (AIS=%.3f)  (paper: L18H9, AIS=3.477)",
                top_k[0][0], top_k[0][1], top_k[0][2])

    # ── Step 6: Calibration ablation (Appendix A) ───────────────────────────
    # Tests whether AIS rankings are stable across calibration set sizes.
    # Runs AIS at 50, 100, 250, 500 pairs and reports Spearman rho vs 500-pair
    # ground truth. Expected result: rho > 0.95 at all sizes.
    if args.run_ablation:
        logger.info("Running calibration set size ablation (Appendix A)...")
        ablation_results = run_calibration_ablation(
            model, tokenizer, image_processor, all_samples,
            n_layers, n_heads,
            calibration_sizes=cfg.phase1.calibration_sizes_ablation,
            device=device,
        )
        # Save rho values only (not full AIS matrices — too large)
        ablation_summary = {
            str(size): {"rho": res["rho"]}
            for size, res in ablation_results.items()
        }
        ablation_path = os.path.join(cfg.paths.phase1_dir, "calibration_ablation.json")
        with open(ablation_path, "w") as f:
            json.dump(ablation_summary, f, indent=2)
        logger.info("Calibration ablation saved → %s", ablation_path)

        # Print table for quick inspection
        logger.info("=== Calibration Ablation Results (Appendix A) ===")
        logger.info("  %-15s  %s", "Calib. size", "Spearman rho")
        for size in sorted(ablation_summary.keys(), key=int):
            rho = ablation_summary[size]["rho"]
            logger.info("  %-15s  %.4f", size, rho)

    # ── Step 7: Visualisations ───────────────────────────────────────────────
    # Figure 1: AIS distribution histogram + top-20 heads bar chart
    plot_ais_distribution(
        ais, top_k,
        save_path=os.path.join(cfg.paths.phase1_dir, "fig_ais_distribution.png"),
        gini=stats["gini"],
    )

    # Figure 2: Per-layer mean and max AIS profile
    plot_layer_ais_profile(
        ais,
        save_path=os.path.join(cfg.paths.phase1_dir, "fig_layer_profile.png"),
    )

    # Figure 3: Cross-modal attention heatmaps for first 5 samples
    # These are for visualisation only — not used for AIS computation
    logger.info("Generating attention heatmaps (first 5 samples)...")
    for i, sample in enumerate(calibration_samples[:5]):
        try:
            hm = extract_attention_heatmap(
                model, tokenizer, image_processor, sample, device
            )
            plot_attention_heatmap(
                hm, sample["caption"],
                save_path=os.path.join(
                    cfg.paths.phase1_dir, f"fig_heatmap_sample_{i}.png"
                ),
            )
        except Exception as exc:
            logger.warning("Heatmap %d failed: %s", i, exc)

    logger.info("Phase 1 complete. Outputs in: %s", cfg.paths.phase1_dir)
    logger.info("Next step: run phase2_pruning.py")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AGSP Phase 1 — AIS computation for MobileVLM V2-1.7B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phase1_baseline.py                          # standard run
  python phase1_baseline.py --n_samples 100          # explicit sample count
  python phase1_baseline.py --run_ablation           # + Appendix A ablation
  python phase1_baseline.py --n_samples 10 --no_bootstrap  # smoke test
        """,
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Number of COCO calibration samples (default: 100 from config)",
    )
    parser.add_argument(
        "--no_debias", action="store_true",
        help="Disable Zhao et al. 2026 debiasing (RoPE + attention-sink correction)",
    )
    parser.add_argument(
        "--no_bootstrap", action="store_true",
        help="Skip bootstrap stability check (faster, use for smoke tests)",
    )
    parser.add_argument(
        "--run_ablation", action="store_true",
        help="Run calibration set size ablation for Appendix A (loads 500 samples)",
    )
    args = parser.parse_args()
    main(args)
