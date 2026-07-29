#!/usr/bin/env python3
"""
phase3_eval.py
──────────────
Phase 3: Evaluation — MobileVLM V2-1.7B.

PURPOSE
───────
Evaluate baseline, AGSP-pruned, and MBP-pruned models across all metrics
needed for the paper and appendices. This phase produces:
  - Paper Table 4 (POPE adversarial results)
  - Paper Figure 8 (POPE accuracy, F1, yes-ratio)
  - Appendix C (POPE at 20% and 30% — no recalibration)
  - Appendix D (latency and FLOPs benchmarks)
  - Optional: VQAv2 (general capability preservation)
  - Optional: nuScenes safety (H3 hypothesis)
  - Optional: AIS ablation (causal importance of top-5 heads)

EVALUATION METRICS
──────────────────
Primary (paper):
  - POPE adversarial accuracy, F1, yes-ratio (500 samples, COCO val2014)
  - Epoch-1 loss and final loss (from Phase 2 pareto_data.json)

Secondary (appendices):
  - POPE at 20% and 30% without recalibration (Appendix C)
  - Latency (ms), VRAM (GB), FLOPs reduction (Appendix D)
  - VQAv2 soft accuracy (optional — 2,000 samples)
  - nuScenes safety detection accuracy (optional — H3)

POPE EVALUATION NOTES
──────────────────────
At 40% compression, output collapse occurs in opposite directions:
  - AGSP: yes-bias (yes-ratio ~0.990) — retained alignment-critical heads
  - MBP:  no-bias  (yes-ratio ~0.022) — retained text-generation heads
Both score ~0.506 by chance. Recalibration (Stage 2 in Phase 2) corrects this.

At 20–30% compression, output collapse does NOT occur so POPE can be
evaluated directly without recalibration. This provides cleaner H1 evidence.

OUTPUTS
───────
  outputs/phase3/results_baseline.json
  outputs/phase3/results_agsp.json
  outputs/phase3/results_mbp.json
  outputs/phase3/results_all.json
  outputs/phase3/results_table.md
  outputs/phase3/pope_low_compression.json   — Appendix C
  outputs/phase3/latency_flops_table.json    — Appendix D
  outputs/phase3/fig_comparison.png
  outputs/phase3/fig_pareto_pope.png
  outputs/phase3/fig_safety_radar.png        — if nuScenes available

USAGE
─────
  # Full evaluation (baseline + AGSP + MBP at 40%)
  python phase3_eval.py

  # Specific pruning ratio
  python phase3_eval.py --pruning_ratio 0.40

  # POPE at low compression (Appendix C — no recalibration)
  python phase3_eval.py --pope_low_compression

  # Run AIS ablation (zero top-5 heads, measure POPE drop)
  python phase3_eval.py --run_ablation

  # Include VQAv2 evaluation
  python phase3_eval.py --run_vqa

  # Skip baseline (already evaluated)
  python phase3_eval.py --skip_baseline

  # Multi-seed evaluation (pass seed used in phase2)
  python phase3_eval.py --pruning_ratio 0.20 --seed 123

CHANGES FROM v3
───────────────
  - Fixed results_to_markdown: moved to top-level in evaluation.py (was dead code)
  - Fixed load_in_8bit → bfloat16
  - Added --pruning_ratio argument for multi-ratio POPE evaluation
  - Added --seed argument for multi-seed experiments
  - Dynamic model path: model_agsp_{ratio}_seed{seed}/ (not hardcoded)
  - Added --pope_low_compression for Appendix C
  - Added latency/FLOPs table generation for Appendix D
  - VQAv2 moved to optional --run_vqa flag (was always running, slow)
  - nuScenes safety moved to optional --run_safety flag
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "MobileVLM_code"))

from config import get_config
from utils.evaluation import (
    evaluate_pope,
    evaluate_pope_no_recalibration,
    measure_latency_and_memory,
    measure_vram_reduction,
    results_to_markdown,
    run_ais_ablation,
    run_full_evaluation,
)
from utils.visualization import plot_agsp_vs_mbp_comparison, plot_pareto_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Ocean storage paths — same convention as Phase 2
OCEAN_OUT = "/ocean/projects/cis260140p/bgodwin/outputs"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_snapshot_dir(model_cache_dir: str, model_id: str) -> str:
    """Dynamically locate the HuggingFace snapshot directory (same as Phase 2)."""
    cache_name    = "models--" + model_id.replace("/", "--")
    snapshots_dir = os.path.join(model_cache_dir, cache_name, "snapshots")
    if not os.path.exists(snapshots_dir):
        return model_id
    snapshots = sorted(
        [os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir)],
        key=os.path.getmtime, reverse=True,
    )
    if not snapshots:
        return model_id
    logger.info("Using snapshot: %s", snapshots[0])
    return snapshots[0]


def load_mobilevlm(model_dir_or_id: str, cfg) -> tuple:
    """
    Load a MobileVLM V2 model from a saved directory or HuggingFace ID.

    Uses MobileLlamaForCausalLM directly (not AutoModelForCausalLM) because
    the custom mobilevlm architecture is not registered in transformers' auto
    model registry.
    """
    from transformers import AutoTokenizer, CLIPImageProcessor
    from mobilevlm.model.mobilellama import MobileLlamaForCausalLM

    logger.info("Loading model from: %s", model_dir_or_id)

    # For HuggingFace model IDs, resolve to local snapshot
    if not os.path.isdir(model_dir_or_id):
        model_dir_or_id = _find_snapshot_dir(cfg.paths.model_cache_dir, model_dir_or_id)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir_or_id,
        trust_remote_code=cfg.hardware.trust_remote_code,
        use_fast=False,
    )
    # MobileVLM doesn't ship a preprocessor_config.json — load from CLIP directly
    image_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14-336",
        cache_dir=cfg.paths.model_cache_dir,
    )
    model = MobileLlamaForCausalLM.from_pretrained(
        model_dir_or_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        ignore_mismatched_sizes=True,
    )
    model.eval()

    # Ensure vision tower is loaded
    model.get_model().vision_tower.load_model()
    model.get_model().vision_tower.vision_tower.to(
        dtype=torch.bfloat16, device="cuda"
    )

    # Log architecture for verification
    llm_cfg  = model.config if not hasattr(model.config, "text_config") else model.config.text_config
    n_layers = llm_cfg.num_hidden_layers
    n_heads  = getattr(llm_cfg, "num_attention_heads", "?")
    logger.info(
        "Loaded: %d layers, %s heads/layer, dtype=%s",
        n_layers, n_heads, next(model.parameters()).dtype,
    )

    return model, tokenizer, image_processor


def _build_model_path(strategy: str, ratio: float, seed: int) -> str:
    """
    Build the ocean storage path for a pruned model checkpoint.

    Convention: model_{strategy}_{ratio_pct}_seed{seed}/
    Example:    model_agsp_40_seed42/

    This convention matches Phase 2 save paths. Always use this function
    rather than hardcoding paths to keep Phase 2 and Phase 3 in sync.
    """
    return os.path.join(
        OCEAN_OUT, f"model_{strategy}_{int(ratio * 100)}_seed{seed}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-model evaluation wrapper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, tokenizer, image_processor, cfg, device, label, args) -> dict:
    """
    Run all requested evaluations for one model and return unified results dict.

    Evaluation order:
      1. run_full_evaluation (latency + VRAM + POPE + safety if available)
      2. VQAv2 (optional — --run_vqa flag)
      3. nuScenes safety (optional — --run_safety flag, separate from run_full)

    Why separate VQA and safety?
    ────────────────────────────
    These evaluations are slow (VQA: ~30 min, safety: ~20 min). Making them
    optional allows fast POPE-only runs during development and full runs
    before final submission.
    """
    # Core evaluations: latency + VRAM + POPE + safety (if nuScenes found)
    results = run_full_evaluation(
        model, tokenizer, image_processor, cfg,
        device=device, label=label,
    )

    # VQAv2 (optional)
    if args.run_vqa:
        logger.info("[%s] Running VQAv2 evaluation (n=%d)...", label, cfg.phase3.n_vqa_samples)
        try:
            from utils.vqa_utils import load_vqa_samples, run_vqa_evaluation
            vqa_samples = load_vqa_samples(cfg, n=cfg.phase3.n_vqa_samples,
                                            seed=cfg.hardware.seed)
            results["vqa"] = run_vqa_evaluation(
                model, tokenizer, image_processor, vqa_samples, device=device,
            )
            logger.info(
                "[%s] VQAv2 soft accuracy: %.4f",
                label, results["vqa"].get("soft_accuracy", 0),
            )
        except Exception as exc:
            logger.warning("[%s] VQAv2 eval failed: %s", label, exc)
            results["vqa"] = {"error": str(exc)}

    # nuScenes safety (optional — H3 hypothesis)
    if args.run_safety:
        logger.info("[%s] Running nuScenes safety evaluation...", label)
        try:
            from utils.nuscenes_utils import run_nuscenes_safety_eval
            results["safety"] = run_nuscenes_safety_eval(
                model, tokenizer, image_processor,
                nuscenes_root=cfg.paths.nuscenes_root,
                safety_categories=cfg.phase3.safety_categories,
                n_per_category=cfg.phase3.n_safety_samples,
                device=device,
            )
            logger.info("[%s] Safety avg: %.4f", label, results["safety"]["average"])
        except Exception as exc:
            logger.warning("[%s] nuScenes eval failed: %s", label, exc)
            results["safety"] = {"error": str(exc)}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Appendix C: POPE at low compression ratios
# ─────────────────────────────────────────────────────────────────────────────

def run_pope_low_compression(cfg, args) -> dict:
    """
    Appendix C: Evaluate POPE at 20% and 30% compression without recalibration.

    Why this matters:
    ─────────────────
    The paper evaluates POPE only at 40% compression — the ratio where AGSP's
    Pareto advantage is weakest (Δ = +0.003). This is the paper's acknowledged
    limitation. Evaluating at 20% and 30% provides cleaner H1 downstream
    evidence because:
      1. Output collapse does not occur at low compression
      2. No recalibration stage needed — fewer confounds
      3. AGSP's Pareto advantage is largest at 20% (Δ = +0.088)

    Results go in Appendix C.
    """
    low_compression_results = {}

    for ratio in cfg.phase3.pope_low_compression_ratios:
        ratio_results = {}

        for strategy in ["agsp", "mbp"]:
            model_path = _build_model_path(strategy, ratio, args.seed)

            if not os.path.isdir(model_path):
                logger.warning(
                    "Model not found for %s @ %.0f%% — skipping. "
                    "Run: python phase2_pruning.py --target_ratio %.2f --seed %d",
                    strategy.upper(), ratio * 100, ratio, args.seed,
                )
                continue

            logger.info(
                "Appendix C: POPE without recalibration [%s @ %.0f%%]",
                strategy.upper(), ratio * 100,
            )

            model, tokenizer, image_processor = load_mobilevlm(model_path, cfg)

            # evaluate_pope_no_recalibration includes output_collapsed flag
            # which detects if collapse occurred (unexpected at low compression)
            result = evaluate_pope_no_recalibration(
                model, tokenizer, image_processor,
                cfg.paths.pope_val_path,
                cfg.paths.pope_images_dir,
                n_samples=cfg.phase3.n_pope_samples,
                device=cfg.hardware.device,
            )
            ratio_results[strategy] = result

            logger.info(
                "  [%s @ %.0f%%] POPE acc=%.4f | F1=%.4f | yes-ratio=%.3f | collapsed=%s",
                strategy.upper(), ratio * 100,
                result["accuracy"], result["f1"],
                result["yes_ratio"], result["output_collapsed"],
            )

            del model
            torch.cuda.empty_cache()

        low_compression_results[ratio] = ratio_results

    return low_compression_results


# ─────────────────────────────────────────────────────────────────────────────
# Appendix D: Latency and FLOPs table
# ─────────────────────────────────────────────────────────────────────────────

def build_latency_flops_table(cfg, args, all_results: dict) -> dict:
    """
    Appendix D: Latency, VRAM, and FLOPs reduction table.

    Aggregates hardware measurements already collected in all_results and
    adds FLOPs reduction computed from pruning information in pareto_data.json.

    Table format (matches Appendix D in paper):
      Ratio | Strategy | VRAM (GB) | FLOPs reduction (%) | tokens/sec
    """
    table_rows = []

    # Extract hardware measurements from evaluation results
    for label, res in all_results.items():
        hw = res.get("hardware", {})
        if hw:
            table_rows.append({
                "label":           label,
                "vram_gb":         hw.get("peak_vram_gb", "—"),
                "mean_latency_ms": hw.get("mean_latency_ms", "—"),
                "std_latency_ms":  hw.get("std_latency_ms", "—"),
            })

    # Add FLOPs reduction from pareto_data if available
    pareto_path = os.path.join(cfg.paths.phase2_dir, "pareto_data.json")
    flops_by_ratio = {}

    if os.path.exists(pareto_path):
        with open(pareto_path) as f:
            pareto = json.load(f)
        for ratio_str, ratio_data in pareto.items():
            for strategy, res in ratio_data.items():
                flops_pct = res.get("flops_reduction_pct", None)
                if flops_pct is not None:
                    flops_by_ratio[f"{strategy}_{ratio_str}"] = flops_pct

    return {
        "hardware_measurements": table_rows,
        "flops_reduction_by_ratio": flops_by_ratio,
        "reference_baseline_tokspersec": 51.63,  # MobileVLM V2 published Jetson Orin value
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    cfg    = get_config()
    seed   = args.seed if args.seed is not None else cfg.hardware.seed
    torch.manual_seed(seed)
    device = cfg.hardware.device

    # Determine pruning ratio for this evaluation run
    # Used to build model paths and for labelling results
    ratio = args.pruning_ratio or cfg.phase2.target_pruning_ratio
    all_results = {}

    # ── Baseline evaluation ──────────────────────────────────────────────────
    # Establishes upper-bound performance (unpruned MobileVLM V2-1.7B).
    # Paper: POPE adversarial accuracy = 0.812, F1 = 0.832, yes-ratio = 0.620.
    if not args.skip_baseline:
        logger.info("Evaluating baseline (unpruned MobileVLM V2-1.7B)...")
        model, tokenizer, ip = load_mobilevlm(cfg.paths.model_id, cfg)
        res = evaluate_model(model, tokenizer, ip, cfg, device=device,
                             label="baseline", args=args)
        all_results["baseline"] = res

        with open(os.path.join(cfg.paths.phase3_dir, "results_baseline.json"), "w") as f:
            json.dump(res, f, indent=2)

        logger.info(
            "Baseline POPE: acc=%.4f | F1=%.4f | yes-ratio=%.3f",
            res.get("pope", {}).get("accuracy", 0),
            res.get("pope", {}).get("f1", 0),
            res.get("pope", {}).get("yes_ratio", 0),
        )
        del model
        torch.cuda.empty_cache()

    # ── AIS ablation (optional) ───────────────────────────────────────────────
    # Zeros the top-5 highest-AIS heads and measures POPE accuracy drop.
    # Quantifies the causal importance of alignment-critical heads.
    # Expected: large POPE drop after zeroing top-5 → confirms AIS validity.
    if args.run_ablation and os.path.exists(cfg.paths.ais_matrix_path):
        logger.info("Running AIS ablation (zeroing top-5 heads, measuring POPE drop)...")
        ais = np.load(cfg.paths.ais_matrix_path)
        model, tokenizer, ip = load_mobilevlm(cfg.paths.model_id, cfg)
        ablation = run_ais_ablation(
            model, tokenizer, ip, ais,
            cfg.paths.pope_val_path, cfg.paths.pope_images_dir,
            n_samples=200, device=device, top_k=5,
        )
        all_results["ablation"] = ablation
        logger.info(
            "AIS ablation: POPE after zeroing top-5 heads = %.4f",
            ablation.get("pope_after_ablation", 0),
        )
        del model
        torch.cuda.empty_cache()

    # ── AGSP evaluation ───────────────────────────────────────────────────────
    # Load and evaluate the AGSP-pruned model at the target ratio.
    # Dynamic path convention: model_agsp_{ratio_pct}_seed{seed}/
    agsp_path = _build_model_path("agsp", ratio, seed)

    if os.path.isdir(agsp_path):
        logger.info("Evaluating AGSP @ %.0f%% (seed=%d)...", ratio * 100, seed)
        model, tokenizer, ip = load_mobilevlm(agsp_path, cfg)
        res = evaluate_model(model, tokenizer, ip, cfg, device=device,
                             label=f"agsp_{int(ratio*100)}", args=args)
        all_results["agsp"] = res

        with open(os.path.join(cfg.paths.phase3_dir, "results_agsp.json"), "w") as f:
            json.dump(res, f, indent=2)

        logger.info(
            "AGSP POPE: acc=%.4f | F1=%.4f | yes-ratio=%.3f",
            res.get("pope", {}).get("accuracy", 0),
            res.get("pope", {}).get("f1", 0),
            res.get("pope", {}).get("yes_ratio", 0),
        )
        del model
        torch.cuda.empty_cache()
    else:
        logger.warning(
            "AGSP model not found at %s. "
            "Run: python phase2_pruning.py --target_ratio %.2f --seed %d",
            agsp_path, ratio, seed,
        )

    # ── MBP evaluation ────────────────────────────────────────────────────────
    # Load and evaluate the MBP baseline model at the target ratio.
    if not args.skip_mbp:
        mbp_path = _build_model_path("mbp", ratio, seed)

        if os.path.isdir(mbp_path):
            logger.info("Evaluating MBP @ %.0f%% (seed=%d)...", ratio * 100, seed)
            model, tokenizer, ip = load_mobilevlm(mbp_path, cfg)
            res = evaluate_model(model, tokenizer, ip, cfg, device=device,
                                 label=f"mbp_{int(ratio*100)}", args=args)
            all_results["mbp"] = res

            with open(os.path.join(cfg.paths.phase3_dir, "results_mbp.json"), "w") as f:
                json.dump(res, f, indent=2)

            logger.info(
                "MBP POPE: acc=%.4f | F1=%.4f | yes-ratio=%.3f",
                res.get("pope", {}).get("accuracy", 0),
                res.get("pope", {}).get("f1", 0),
                res.get("pope", {}).get("yes_ratio", 0),
            )
            del model
            torch.cuda.empty_cache()
        else:
            logger.warning(
                "MBP model not found at %s. "
                "Run: python phase2_pruning.py --target_ratio %.2f --seed %d",
                mbp_path, ratio, seed,
            )

    # ── Appendix C: POPE at low compression ──────────────────────────────────
    # Evaluates POPE at 20% and 30% without recalibration.
    # Provides cleaner H1 downstream evidence than the 40% result.
    if args.pope_low_compression:
        logger.info("Running Appendix C: POPE at low compression ratios...")
        low_comp_results = run_pope_low_compression(cfg, args)
        all_results["pope_low_compression"] = low_comp_results

        pope_low_path = os.path.join(cfg.paths.phase3_dir, "pope_low_compression.json")
        with open(pope_low_path, "w") as f:
            json.dump(low_comp_results, f, indent=2)
        logger.info("Appendix C results saved → %s", pope_low_path)

    # ── Save all results ──────────────────────────────────────────────────────
    all_results_path = os.path.join(cfg.paths.phase3_dir, "results_all.json")
    with open(all_results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate and save markdown table
    # results_to_markdown is now a top-level function — fixed from v3 dead code
    md = results_to_markdown(all_results)
    md_path = os.path.join(cfg.paths.phase3_dir, "results_table.md")
    with open(md_path, "w") as f:
        f.write(md)
    print("\n" + md + "\n")

    # ── Appendix D: Latency and FLOPs table ──────────────────────────────────
    latency_table = build_latency_flops_table(cfg, args, all_results)
    latency_path  = os.path.join(cfg.paths.phase3_dir, "latency_flops_table.json")
    with open(latency_path, "w") as f:
        json.dump(latency_table, f, indent=2)
    logger.info("Appendix D latency/FLOPs table saved → %s", latency_path)

    # ── Visualisations ────────────────────────────────────────────────────────
    # Figure: AGSP vs MBP vs Baseline comparison bar chart
    def _summary(res):
        return {
            "pope":    res.get("pope", {}).get("accuracy", 0),
            "vram":    res.get("hardware", {}).get("peak_vram_gb", 0),
            "latency": res.get("hardware", {}).get("mean_latency_ms", 0),
        }

    if len(all_results) >= 2:
        plot_agsp_vs_mbp_comparison(
            {k: _summary(v) for k, v in all_results.items()
             if k in ["baseline", "agsp", "mbp"]},
            save_path=os.path.join(cfg.paths.phase3_dir, "fig_comparison.png"),
        )

    # Pareto POPE curve (if pareto_data.json has POPE values from earlier runs)
    pareto_path = os.path.join(cfg.paths.phase2_dir, "pareto_data.json")
    if os.path.exists(pareto_path):
        with open(pareto_path) as f:
            pareto = json.load(f)
        ratios    = sorted([float(k) for k in pareto])
        agsp_pope = [pareto[str(r)].get("agsp", {}).get("pope_acc", 0) for r in ratios]
        mbp_pope  = [pareto[str(r)].get("mbp",  {}).get("pope_acc", 0) for r in ratios]
        if any(v > 0 for v in agsp_pope):
            plot_pareto_curve(
                ratios, agsp_pope, mbp_pope, "POPE Accuracy",
                os.path.join(cfg.paths.phase3_dir, "fig_pareto_pope.png"),
            )

    # Safety radar chart (H3 — only if safety results available)
    if all(k in all_results for k in ["baseline", "agsp", "mbp"]):
        b = all_results["baseline"].get("safety", {})
        a = all_results["agsp"].get("safety", {})
        m = all_results["mbp"].get("safety", {})
        if b and a and m and not any("error" in x for x in [b, a, m]):
            from utils.visualization import plot_safety_radar
            plot_safety_radar(
                b, a, m,
                os.path.join(cfg.paths.phase3_dir, "fig_safety_radar.png"),
            )

    logger.info("Phase 3 complete. Outputs in: %s", cfg.paths.phase3_dir)
    logger.info("Results table: %s", md_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AGSP Phase 3 — Evaluation (MobileVLM V2-1.7B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard evaluation at 40%
  python phase3_eval.py

  # Evaluation at specific ratio (must match phase2 run)
  python phase3_eval.py --pruning_ratio 0.40

  # Multi-seed evaluation (pass same seed used in phase2)
  python phase3_eval.py --pruning_ratio 0.20 --seed 42
  python phase3_eval.py --pruning_ratio 0.20 --seed 123
  python phase3_eval.py --pruning_ratio 0.20 --seed 7

  # Appendix C: POPE at 20% and 30% without recalibration
  python phase3_eval.py --pope_low_compression

  # Full evaluation including VQA, safety, and AIS ablation
  python phase3_eval.py --run_vqa --run_safety --run_ablation

  # Skip baseline (already evaluated in a previous run)
  python phase3_eval.py --skip_baseline
        """,
    )
    parser.add_argument(
        "--pruning_ratio", type=float, default=None,
        help="Pruning ratio to evaluate (default: 0.40 from config)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed used in phase2 (default: 42). Must match phase2 seed to find checkpoints.",
    )
    parser.add_argument(
        "--skip_baseline", action="store_true",
        help="Skip baseline evaluation (use if already evaluated)",
    )
    parser.add_argument(
        "--skip_mbp", action="store_true",
        help="Skip MBP evaluation",
    )
    parser.add_argument(
        "--run_ablation", action="store_true",
        help="Run AIS ablation: zero top-5 heads and measure POPE drop",
    )
    parser.add_argument(
        "--run_vqa", action="store_true",
        help="Run VQAv2 evaluation (2,000 samples, ~30 min)",
    )
    parser.add_argument(
        "--run_safety", action="store_true",
        help="Run nuScenes safety evaluation (H3 hypothesis, ~20 min)",
    )
    parser.add_argument(
        "--pope_low_compression", action="store_true",
        help="Evaluate POPE at 20% and 30% without recalibration (Appendix C)",
    )
    args = parser.parse_args()
    main(args)
