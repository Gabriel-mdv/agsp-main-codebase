"""
utils/evaluation.py
───────────────────
Evaluation utilities for AGSP-VLM (MobileVLM V2-1.7B).

MobileVLM V2 specifics:
  - Uses tokenizer + image_processor separately (no combined LlavaProcessor)
  - Conversation template: USER: <image>\n{question}\nASSISTANT:
  - Forward pass uses 'images' key (NOT 'pixel_values' — MobileLLaMA convention)
  - max_new_tokens=20 fixed for valid latency comparison across all models

Metrics:
  - POPE adversarial (primary grounding metric — 500 samples)
  - VQAv2 (2,000-sample subset — optional)
  - Latency + peak VRAM (Appendix D)
  - AIS ablation (zero top-5 heads; measure POPE drop)

Changes from v3:
  - Fixed VQA inference: pixel_values → images (was silently failing)
  - Fixed load_in_8bit: removed — bfloat16 throughout
  - Fixed results_to_markdown: moved to top-level function (was dead code
    inside evaluate_model after a return statement)
  - Added evaluate_pope_no_recalibration() for Appendix C
    (POPE at 20% and 30% without recalibration stage)
  - Added measure_vram_reduction() for Appendix D table
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Fixed across all models for valid latency comparison
MAX_NEW_TOKENS = 20


# ─────────────────────────────────────────────────────────────────────────────
# MobileVLM V2 inference helper
# ─────────────────────────────────────────────────────────────────────────────

def _generate(model, tokenizer, image_processor, image, prompt, device):
    """
    Build inputs and run model.generate() for MobileVLM V2.

    Uses MobileVLM conversation template via mobilevlm utilities.
    Images passed as 'images' key (not 'pixel_values').
    """
    from mobilevlm.conversation import conv_templates
    from mobilevlm.utils import process_images, tokenizer_image_token
    from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    conv = conv_templates["v1"].copy()
    forced = prompt + " Please answer yes or no."
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + forced)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    stop_str    = conv.sep2

    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    # bfloat16 — matches model dtype
    images_tensor = process_images([image], image_processor, model.config).to(
        device, dtype=torch.bfloat16
    )

    with torch.no_grad():
        out = model.generate(
            input_ids,
            images=images_tensor,          # 'images' not 'pixel_values'
            do_sample=False,
            num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )

    answer = tokenizer.batch_decode(
        out[:, input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip().lower()
    if stop_str and answer.endswith(stop_str):
        answer = answer[:-len(stop_str)]
    return answer


def _generate_simple(model, tokenizer, image_processor, image, prompt, device):
    """
    Simplified inference for VQA and latency benchmarks.
    Uses direct tokenizer + image_processor without conv template.
    Images passed as 'images' key (not 'pixel_values').
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    pv  = image_processor(
        images=image, return_tensors="pt"
    ).pixel_values.to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        out = model.generate(
            **enc,
            images=pv,                     # 'images' not 'pixel_values'
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Latency and VRAM
# ─────────────────────────────────────────────────────────────────────────────

def measure_latency_and_memory(
    model,
    tokenizer,
    image_processor,
    sample_image,
    n_trials: int = 10,
    device: str = "cuda",
) -> Dict:
    """Measure inference latency and peak VRAM with fixed max_new_tokens=20."""
    torch.cuda.reset_peak_memory_stats(device)
    prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"
    enc    = tokenizer(prompt, return_tensors="pt").to(device)
    pv     = image_processor(
        images=sample_image, return_tensors="pt"
    ).pixel_values.to(device, dtype=torch.bfloat16)

    latencies = []
    with torch.no_grad():
        for _ in range(n_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(
                **enc,
                images=pv,                 # 'images' not 'pixel_values'
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms":  float(np.std(latencies)),
        "min_latency_ms":  float(np.min(latencies)),
        "peak_vram_gb":    torch.cuda.max_memory_allocated(device) / (1024 ** 3),
    }


def measure_vram_reduction(
    baseline_vram_gb: float,
    pruned_vram_gb: float,
) -> Dict:
    """Compute VRAM reduction for Appendix D table."""
    reduction_gb  = baseline_vram_gb - pruned_vram_gb
    reduction_pct = 100.0 * reduction_gb / baseline_vram_gb
    return {
        "baseline_vram_gb":  baseline_vram_gb,
        "pruned_vram_gb":    pruned_vram_gb,
        "reduction_gb":      reduction_gb,
        "reduction_pct":     reduction_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POPE adversarial evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pope(
    model,
    tokenizer,
    image_processor,
    pope_json_path: str,
    images_dir: str,
    n_samples: int = 500,
    device: str = "cuda",
) -> Dict:
    """
    POPE adversarial evaluation — primary grounding metric.

    Paper: 500 samples from COCO val2014, adversarial split.
    Returns accuracy, precision, recall, F1, yes-ratio.
    """
    with open(pope_json_path) as f:
        data = [json.loads(line) for line in f if line.strip()]
    data = data[:n_samples]

    from tqdm import tqdm
    tp = fp = tn = fn = 0
    debug_count = 0

    for item in tqdm(data, desc="POPE eval", unit="sample"):
        img_path = Path(images_dir) / item["image"]
        if not img_path.exists():
            logger.warning("POPE image missing: %s", img_path)
            continue

        image  = Image.open(img_path).convert("RGB")
        prompt = item["text"]
        answer = _generate(model, tokenizer, image_processor, image, prompt, device)

        pred = "yes" if "yes" in answer else "no"
        gt   = item["label"].lower().strip()

        # Debug: log first 5 raw answers to diagnose parsing issues
        if debug_count < 5:
            logger.info(
                "POPE debug [%d]: Q='%s' | raw='%s' | pred=%s | gt=%s",
                debug_count, prompt[:60], repr(answer[:100]), pred, gt,
            )
            debug_count += 1

        if   pred == "yes" and gt == "yes": tp += 1
        elif pred == "yes" and gt == "no":  fp += 1
        elif pred == "no"  and gt == "no":  tn += 1
        else:                               fn += 1

    total = tp + fp + tn + fn
    acc   = (tp + tn) / total if total else 0.0
    prec  = tp / (tp + fp) if (tp + fp) else 0.0
    rec   = tp / (tp + fn) if (tp + fn) else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    yes_r = (tp + fp) / total if total else 0.0

    return {
        "accuracy":    acc,
        "precision":   prec,
        "recall":      rec,
        "f1":          f1,
        "yes_ratio":   yes_r,
        "n_evaluated": total,
    }


def evaluate_pope_no_recalibration(
    model,
    tokenizer,
    image_processor,
    pope_json_path: str,
    images_dir: str,
    n_samples: int = 500,
    device: str = "cuda",
) -> Dict:
    """
    POPE evaluation without recalibration stage.

    Used for Appendix C (POPE at 20% and 30% compression).
    At low compression output collapse does not occur so recalibration
    is not needed — this provides cleaner H1 downstream evidence.

    Returns same metrics as evaluate_pope() plus output_collapsed flag.
    """
    results = evaluate_pope(
        model, tokenizer, image_processor,
        pope_json_path, images_dir, n_samples, device,
    )

    # Detect output collapse: yes-ratio outside [0.3, 0.7] signals collapse
    yes_ratio = results["yes_ratio"]
    results["output_collapsed"] = yes_ratio > 0.85 or yes_ratio < 0.15
    results["recalibration_applied"] = False

    if results["output_collapsed"]:
        logger.warning(
            "Output collapse detected (yes-ratio=%.3f). "
            "Consider recalibration for accurate POPE results.", yes_ratio
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# AIS ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_ais_ablation(
    model,
    tokenizer,
    image_processor,
    ais_matrix,
    pope_json_path: str,
    images_dir: str,
    n_samples: int = 200,
    device: str = "cuda",
    top_k: int = 5,
) -> Dict:
    """
    Zero the top-k highest-AIS heads and measure POPE accuracy drop.
    Quantifies the causal importance of top alignment heads.
    """
    from utils.alignment_scoring import (
        get_top_k_heads, safe_get_llm_config, safe_get_llm_layers
    )

    llm_cfg   = safe_get_llm_config(model)
    head_dim  = llm_cfg.hidden_size // llm_cfg.num_attention_heads
    layers    = safe_get_llm_layers(model)
    top_heads = get_top_k_heads(ais_matrix, k=top_k)

    # Save original weights
    saved = {}
    for layer_idx, head_idx, _ in top_heads:
        W_o = layers[layer_idx].self_attn.o_proj.weight
        s   = head_idx * head_dim
        saved[(layer_idx, head_idx)] = W_o[:, s:s + head_dim].clone()

    # Zero top heads
    with torch.no_grad():
        for layer_idx, head_idx, _ in top_heads:
            W_o = layers[layer_idx].self_attn.o_proj.weight
            s   = head_idx * head_dim
            W_o[:, s:s + head_dim] = 0.0

    result = evaluate_pope(
        model, tokenizer, image_processor,
        pope_json_path, images_dir, n_samples, device,
    )

    # Restore original weights
    with torch.no_grad():
        for (li, hi), weights in saved.items():
            layers[li].self_attn.o_proj.weight[
                :, hi * head_dim:(hi + 1) * head_dim
            ] = weights

    return {
        "pope_after_ablation": result["accuracy"],
        "yes_ratio_after":     result["yes_ratio"],
        "top_k_zeroed":        top_k,
        "heads_zeroed":        [(l, h) for l, h, _ in top_heads],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Safety evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_safety_detection(
    model,
    tokenizer,
    image_processor,
    images_dir: str,
    safety_categories: List[str],
    n_samples: int = 100,
    device: str = "cuda",
) -> Dict:
    """
    H3 safety-critical object detection on nuScenes images.
    Open-ended non-leaky prompt to avoid answer leakage.
    """
    images_path = Path(images_dir)
    image_files = list(images_path.rglob("*.jpg"))[:n_samples]

    cat_results: Dict[str, List[int]] = {c: [] for c in safety_categories}

    for img_file in image_files:
        try:
            image = Image.open(img_file).convert("RGB")
        except Exception:
            continue
        fname_lower = img_file.stem.lower()

        prompt = (
            "USER: <image>\nWhat objects are visible in this image? "
            "List all you can see.\nASSISTANT:"
        )
        answer = _generate_simple(
            model, tokenizer, image_processor, image, prompt, device
        )

        for category in safety_categories:
            gt   = any(kw in fname_lower for kw in category.split())
            pred = any(kw in answer      for kw in category.split())
            cat_results[category].append(int(pred == gt))

    per_cat = {
        c: float(np.mean(v)) if v else 0.0
        for c, v in cat_results.items()
    }
    per_cat["average"] = float(np.mean(list(per_cat.values())))
    return per_cat


# ─────────────────────────────────────────────────────────────────────────────
# Results formatting
# ─────────────────────────────────────────────────────────────────────────────

def results_to_markdown(all_results: Dict) -> str:
    """
    Format all_results dict as a markdown table.

    NOTE: This is a top-level function — NOT inside evaluate_model().
    In v3 it was incorrectly placed after a return statement (dead code).
    """
    lines = [
        "# Phase 3 Results — AGSP vs. MBP vs. Baseline (MobileVLM V2-1.7B)",
        "",
        "| Metric | Baseline | AGSP (40%) | MBP (40%) |",
        "|--------|----------|------------|-----------|",
    ]

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    metrics = [
        ("POPE Accuracy",       ("pope",     "accuracy")),
        ("POPE F1",             ("pope",     "f1")),
        ("POPE Yes-ratio",      ("pope",     "yes_ratio")),
        ("VQAv2 Soft Acc.",     ("vqa",      "soft_accuracy")),
        ("Mean Latency (ms)",   ("hardware", "mean_latency_ms")),
        ("Peak VRAM (GB)",      ("hardware", "peak_vram_gb")),
        ("Safety Avg.",         ("safety",   "average")),
    ]

    for name, (sec, key) in metrics:
        row = [name]
        for label in ["baseline", "agsp", "mbp"]:
            val = all_results.get(label, {}).get(sec, {}).get(key, "—")
            row.append(_fmt(val))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    model,
    tokenizer,
    image_processor,
    cfg,
    device: str = "cuda",
    label: str = "model",
) -> Dict:
    """
    Run all Phase 3 evaluations for one model and return unified results dict.

    Evaluation order:
      1. Latency + VRAM (fast, no data needed)
      2. POPE adversarial (primary grounding metric)
      3. Safety detection (H3, optional — skipped if nuScenes not found)
    """
    logger.info("=== Evaluation: %s ===", label)
    results: Dict = {"label": label}

    # 1. Latency and VRAM
    dummy = Image.new("RGB", (336, 336), (128, 128, 128))
    results["hardware"] = measure_latency_and_memory(
        model, tokenizer, image_processor, dummy,
        n_trials=cfg.phase3.n_latency_trials, device=device,
    )
    logger.info(
        "[%s] Latency: %.1f ms | VRAM: %.2f GB",
        label,
        results["hardware"]["mean_latency_ms"],
        results["hardware"]["peak_vram_gb"],
    )

    # 2. POPE
    if Path(cfg.paths.pope_val_path).exists():
        results["pope"] = evaluate_pope(
            model, tokenizer, image_processor,
            cfg.paths.pope_val_path, cfg.paths.pope_images_dir,
            n_samples=cfg.phase3.n_pope_samples, device=device,
        )
        logger.info(
            "[%s] POPE accuracy: %.4f | F1: %.4f | yes-ratio: %.3f",
            label,
            results["pope"]["accuracy"],
            results["pope"]["f1"],
            results["pope"]["yes_ratio"],
        )
    else:
        logger.warning("[%s] POPE data not found — skipping.", label)

    # 3. Safety (H3 — optional)
    if Path(cfg.paths.nuscenes_images_dir).exists():
        results["safety"] = evaluate_safety_detection(
            model, tokenizer, image_processor,
            cfg.paths.nuscenes_images_dir,
            cfg.phase3.safety_categories,
            n_samples=cfg.phase3.n_safety_samples, device=device,
        )
        logger.info(
            "[%s] Safety avg: %.4f", label, results["safety"]["average"]
        )
    else:
        logger.info("[%s] nuScenes not found — safety eval skipped.", label)

    return results
