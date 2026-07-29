#!/usr/bin/env python3
"""
phase2_pruning.py
─────────────────
Phase 2: Structural Pruning and LoRA Fine-Tuning — MobileVLM V2-1.7B.

PURPOSE
───────
Apply attention head pruning to MobileVLM V2-1.7B using three strategies:
  1. AGSP — Alignment-Guided Structural Pruning (uses AIS from Phase 1)
  2. MBP  — Magnitude-Based Pruning (baseline: prune by weight L2 norm)
  3. RAND — Random Pruning (lower-bound baseline for Stage 3)

Then recover the pruned models using LoRA fine-tuning on COCO captions.

TWO-STAGE FINE-TUNING
──────────────────────
Stage 1 (grounding recovery):
  - Dataset:    6,000 COCO val2017 image-caption pairs  ← matches paper Table 3
  - LoRA:       rank=16, alpha=32, q_proj + v_proj
  - LR:         1e-5 (AdamW), batch=4, 3 epochs
  - KD loss:    λ=0.3 against unpruned teacher logits

Stage 2 (output recalibration, 40%+ only):
  - Dataset:    10,324 balanced yes/no QA pairs from LLaVA-Instruct-150K
  - LR:         5e-6, 3 epochs, no KD loss
  - Purpose:    Corrects yes/no output bias after heavy compression

EPOCH-1 LOSS DIAGNOSTIC
────────────────────────
The paper's primary criterion quality metric is Epoch-1 loss — training
loss before any LoRA recovery. Lower = less grounding damage from criterion.
  Paper results: AGSP 1.78–2.01, MBP 3.04–4.23 (2× gap across all ratios).

OUTPUTS
───────
  /ocean/.../model_agsp_{ratio}_seed{seed}/  — AGSP pruned + LoRA recovered
  /ocean/.../model_mbp_{ratio}_seed{seed}/   — MBP pruned + LoRA recovered
  /ocean/.../pareto_data_seed{seed}.json     — Epoch-1 and final loss
  outputs/phase2/pareto_data.json            — local copy for figures
  outputs/phase2/pruned_heads_{strategy}_seed{seed}.json

USAGE
─────
  # Smoke test (fast end-to-end check, ~5 minutes)
  python src/phase2_pruning.py --smoke_test

  # Full Pareto sweep (all 5 ratios, AGSP + MBP, ~4 hours)
  python src/phase2_pruning.py

  # Single ratio with specific seed
  python src/phase2_pruning.py --target_ratio 0.40 --seed 42

  # Multi-seed runs (Stage 3 statistical validation)
  python src/phase2_pruning.py --target_ratio 0.20 --seed 42
  python src/phase2_pruning.py --target_ratio 0.20 --seed 123
  python src/phase2_pruning.py --target_ratio 0.20 --seed 7

  # Include random pruning baseline
  python src/phase2_pruning.py --include_rand

  # Scope boundary experiment (Section 6.2, RefCOCO fine-tuning)
  python src/phase2_pruning.py --scope_boundary --target_ratio 0.40

  # Soft pruning (zero weights instead of structural reshape)
  python src/phase2_pruning.py --soft
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

# Add project root to path so config and utils are found
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "MobileVLM_code"))

from config import get_config
from utils.finetuning import finetune_coco_captions, finetune_recalibration
from utils.pruning import measure_flops_reduction, measure_sparsity, prune_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Ocean storage — avoids home directory quota issues on PSC Bridges-2
OCEAN_OUT = "/ocean/projects/cis260140p/bgodwin/outputs"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_snapshot_dir(model_cache_dir: str, model_id: str) -> str:
    """
    Dynamically locate the HuggingFace snapshot directory.

    Why dynamic detection?
    ──────────────────────
    Hardcoding the snapshot hash breaks if the cache is cleared, the model
    is updated, or the code runs on a different machine. Dynamic detection
    always finds the most recently downloaded version.
    """
    cache_name    = "models--" + model_id.replace("/", "--")
    snapshots_dir = os.path.join(model_cache_dir, cache_name, "snapshots")

    if not os.path.exists(snapshots_dir):
        logger.info("Snapshot dir not found — loading from HuggingFace: %s", model_id)
        return model_id

    snapshots = sorted(
        [os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir)],
        key=os.path.getmtime, reverse=True,
    )
    if not snapshots:
        return model_id

    logger.info("Using snapshot: %s", snapshots[0])
    return snapshots[0]


def load_model_fresh(cfg):
    """
    Load a fresh MobileVLM V2-1.7B copy in bfloat16.

    Why 'fresh'?
    ────────────
    Each Pareto experiment needs an independent unpruned model copy so
    one experiment's pruning does not affect another.

    Why MobileLlamaForCausalLM directly?
    ────────────────────────────────────
    Structural pruning requires manual Q/K/V/O weight slicing which works
    best with direct access to the custom architecture. AutoModelForCausalLM
    works for loading but MobileLlamaForCausalLM is safer for weight surgery.

    Why bfloat16?
    ─────────────
    Paper specifies bfloat16. float16 has narrower dynamic range and can
    produce NaN gradients during fine-tuning on some batches.
    """
    from transformers import AutoTokenizer, CLIPImageProcessor
    from mobilevlm.model.mobilellama import MobileLlamaForCausalLM

    snapshot_dir = _find_snapshot_dir(cfg.paths.model_cache_dir, cfg.paths.model_id)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.paths.model_id,
        cache_dir=cfg.paths.model_cache_dir,
        use_fast=False,
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14-336",
        cache_dir=cfg.paths.model_cache_dir,
    )
    model = MobileLlamaForCausalLM.from_pretrained(
    snapshot_dir,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map={"": 0},
    ignore_mismatched_sizes=True,
)
    model.eval()

    # Ensure vision tower is loaded in bfloat16
    model.get_model().vision_tower.load_model()
    model.get_model().vision_tower.vision_tower.to(
        dtype=torch.bfloat16, device="cuda"
    )

    return model, tokenizer, image_processor


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_single_experiment(
    cfg, ais_matrix, strategy, ratio, pruning_mode, seed, scope_boundary=False
):
    """
    Run one complete pruning + fine-tuning experiment.

    Steps:
      1. Load fresh student model
      2. Load separate teacher model (for KD loss)
      3. Prune student according to strategy
      4. Stage 1: LoRA fine-tuning on COCO captions
      5. Stage 2: Recalibration (only at 40%+ and not smoke_test)
      6. Return model + results

    Why a separate teacher?
    ────────────────────────
    KD loss requires soft labels from the unpruned model. The teacher must
    stay frozen at unpruned weights while the student is updated. In v3,
    teacher_model was never passed — KD loss silently never ran.
    """
    logger.info("=" * 65)
    logger.info(
        "Experiment: %s @ %.0f%% | mode=%s | seed=%d | scope=%s",
        strategy.upper(), ratio * 100, pruning_mode, seed, scope_boundary,
    )
    logger.info("=" * 65)

    # Load student model
    model, tokenizer, image_processor = load_model_fresh(cfg)

    # Load teacher model for KD loss (separate frozen copy)
    teacher_model = None
    if cfg.phase2.use_kd_loss:
        logger.info("Loading teacher model for KD loss (λ=%.2f)...", cfg.phase2.kd_weight)
        teacher_model, _, _ = load_model_fresh(cfg)
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        teacher_model.eval()

    # Apply structural pruning
    model, pruned_heads = prune_model(
        model,
        ais_matrix=ais_matrix if strategy == "agsp" else None,
        pruning_ratio=ratio,
        per_layer_cap=cfg.phase2.per_layer_cap,
        strategy=strategy,
        pruning_mode=pruning_mode,
        seed=seed,
    )
    sparsity = measure_sparsity(model, pruned_heads)
    logger.info("Sparsity: %s", sparsity)

    # Compute FLOPs reduction for Appendix D
    flops_info = {}
    if teacher_model is not None:
        try:
            flops_info = measure_flops_reduction(teacher_model, model)
            logger.info("FLOPs reduction: %.1f%%", flops_info.get("flops_reduction_pct", 0))
        except Exception as exc:
            logger.warning("FLOPs measurement failed: %s", exc)

    # Stage 1: Fine-tuning
    if scope_boundary:
        # Section 6.2 only — RefCOCO scope boundary experiment
        logger.info("Scope boundary: using RefCOCO dataset (Section 6.2).")
        from utils.data_utils import RefCOCODataset
        from utils.finetuning import finetune_pruned_model
        dataset = RefCOCODataset(
            json_path=cfg.paths.refcoco_val_path,
            images_dir=cfg.paths.coco_val_images_dir,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_samples=cfg.phase2.n_finetune_samples,
            split="train",
        )
        epoch_losses = finetune_pruned_model(
            model, tokenizer, image_processor, dataset,
            pruned_heads=pruned_heads,
            teacher_model=teacher_model,
            kd_weight=cfg.phase2.kd_weight,
            n_epochs=cfg.phase2.finetune_epochs,
            batch_size=cfg.phase2.finetune_batch_size,
            lr=cfg.phase2.finetune_lr,
            zero_pruned_grads=cfg.phase2.zero_pruned_grads,
            use_lora=cfg.phase2.use_lora,
            lora_r=cfg.phase2.lora_r,
            lora_alpha=cfg.phase2.lora_alpha,
            device=cfg.hardware.device,
        )
    else:
        # Primary: COCO captions — matches paper Table 3
        model, epoch_losses = finetune_coco_captions(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            teacher_model=teacher_model,
            pruned_heads=pruned_heads,
            cfg=cfg,
            device=cfg.hardware.device,
        )

    epoch1_loss = epoch_losses[0] if epoch_losses else None
    final_loss  = epoch_losses[-1] if epoch_losses else None

    logger.info(
        "[%s @ %.0f%%] Epoch-1 loss: %.4f | Final loss: %.4f",
        strategy.upper(), ratio * 100,
        epoch1_loss or float("nan"),
        final_loss  or float("nan"),
    )

    result = {
        "strategy":       strategy,
        "pruning_ratio":  ratio,
        "pruning_mode":   pruning_mode,
        "seed":           seed,
        "scope_boundary": scope_boundary,
        **sparsity,
        **flops_info,
        "epoch_losses":   epoch_losses,
        "epoch1_loss":    epoch1_loss,
        "final_loss":     final_loss,
    }

    # Free teacher memory
    if teacher_model is not None:
        del teacher_model
        torch.cuda.empty_cache()

    return model, tokenizer, image_processor, pruned_heads, result


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    cfg = get_config()

    seed = args.seed if args.seed is not None else cfg.hardware.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Smoke test override ───────────────────────────────────────────────────
    # Quick end-to-end check: single ratio, 1 epoch, 8 samples, no KD loss.
    # Runs in ~5 minutes to verify the full pipeline works before the full run.
    if args.smoke_test:
        logger.info("=" * 65)
        logger.info("SMOKE TEST MODE — overriding config for fast end-to-end check")
        logger.info("  Single ratio: 40%")
        logger.info("  Epochs:       1")
        logger.info("  Samples:      8")
        logger.info("  KD loss:      disabled")
        logger.info("=" * 65)
        cfg.phase2.pruning_ratios      = [0.40]
        cfg.phase2.finetune_epochs     = 1
        cfg.phase2.n_finetune_samples  = 8
        cfg.phase2.finetune_batch_size = 2
        cfg.phase2.use_kd_loss         = False
        args.target_ratio              = 0.40

    # ── Verify Phase 1 output exists ─────────────────────────────────────────
    if not os.path.exists(cfg.paths.ais_matrix_path):
        logger.error(
            "AIS matrix not found at %s. Run phase1_baseline.py first.",
            cfg.paths.ais_matrix_path,
        )
        sys.exit(1)

    ais_matrix   = np.load(cfg.paths.ais_matrix_path)
    pruning_mode = "soft" if args.soft else cfg.phase2.pruning_mode
    target_ratio = args.target_ratio or cfg.phase2.target_pruning_ratio

    # Override Pareto sweep ratios if --ratios is provided
    if args.ratios:
        cfg.phase2.pruning_ratios = args.ratios
        logger.info("Overriding pruning ratios → %s", cfg.phase2.pruning_ratios)

    if args.no_lora:
        cfg.phase2.use_lora = False

    strategies = ["agsp", "mbp"]
    if args.include_rand:
        strategies.append("rand")
        logger.info("Including RAND baseline.")

    os.makedirs(OCEAN_OUT, exist_ok=True)

    # ── Pareto sweep ──────────────────────────────────────────────────────────
    # Runs all ratios × strategies. Records Epoch-1 and final loss.
    # Each experiment loads a fresh model to avoid cross-contamination.
    pareto_data: dict = {}

    for ratio in cfg.phase2.pruning_ratios:
        pareto_data[ratio] = {}
        for strategy in strategies:
            _, _, _, _, result = run_single_experiment(
                cfg, ais_matrix, strategy, ratio, pruning_mode,
                seed=seed, scope_boundary=args.scope_boundary,
            )
            pareto_data[ratio][strategy] = result
            logger.info(
                "Pareto: [%s @ %.0f%%] sparsity=%.1f%% | "
                "epoch1=%.4f | final=%.4f",
                strategy.upper(), ratio * 100,
                result["sparsity_pct"],
                result["epoch1_loss"] or float("nan"),
                result["final_loss"]  or float("nan"),
            )

    # Save Pareto data
    pareto_path = os.path.join(OCEAN_OUT, f"pareto_data_seed{seed}.json")
    with open(pareto_path, "w") as f:
        json.dump({str(k): v for k, v in pareto_data.items()}, f, indent=2)
    logger.info("Pareto data saved → %s", pareto_path)

    local_pareto = os.path.join(cfg.paths.phase2_dir, "pareto_data.json")
    with open(local_pareto, "w") as f:
        json.dump({str(k): v for k, v in pareto_data.items()}, f, indent=2)

    # ── Save full models at target ratio ──────────────────────────────────────
    # Full model checkpoints saved at target ratio only (default 40%).
    # For other ratios needed by Appendix C, re-run with --target_ratio.
    logger.info("Saving full models at target ratio %.0f%%...", target_ratio * 100)

    for strategy in strategies:
        save_dir = os.path.join(
            OCEAN_OUT, f"model_{strategy}_{int(target_ratio * 100)}_seed{seed}"
        )
        model, tokenizer, image_processor, pruned_heads, result = run_single_experiment(
            cfg, ais_matrix, strategy, target_ratio, pruning_mode,
            seed=seed, scope_boundary=args.scope_boundary,
        )

        # Stage 2 recalibration — only at 40%+ and not in smoke test mode
        # At 20-30%, output collapse does not occur so recalibration is skipped
        if target_ratio >= 0.40 and not args.smoke_test and not args.scope_boundary:
            logger.info(
                "Applying Stage 2 recalibration for %s @ %.0f%%...",
                strategy.upper(), target_ratio * 100,
            )
            model, recal_losses = finetune_recalibration(
                model, tokenizer, image_processor, cfg,
                device=cfg.hardware.device,
            )
            result["recalibration_losses"] = recal_losses
            logger.info(
                "Recalibration complete. Final recal loss: %.4f",
                recal_losses[-1] if recal_losses else float("nan"),
            )
        else:
            reason = "smoke test" if args.smoke_test else f"{int(target_ratio*100)}% < 40%"
            logger.info("Skipping recalibration (%s).", reason)

        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        image_processor.save_pretrained(save_dir)
        logger.info("%s model saved → %s", strategy.upper(), save_dir)

        heads_path = os.path.join(
            cfg.paths.phase2_dir, f"pruned_heads_{strategy}_seed{seed}.json"
        )
        with open(heads_path, "w") as f:
            json.dump({"pruned_heads": pruned_heads, "summary": result}, f, indent=2)

        del model
        torch.cuda.empty_cache()

    if args.smoke_test:
        logger.info("Smoke test complete — pipeline works end to end.")
        logger.info("Run without --smoke_test for the full experiment.")
    else:
        logger.info("Phase 2 complete.")
        logger.info("Next step: python src/phase3_eval.py")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AGSP Phase 2 — Structural Pruning + LoRA Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test (~5 min) before committing to the full run
  python src/phase2_pruning.py --smoke_test

  # Full Pareto sweep (~4 hours on V100)
  python src/phase2_pruning.py

  # Single ratio for multi-seed Stage 3 validation
  python src/phase2_pruning.py --target_ratio 0.20 --seed 42
  python src/phase2_pruning.py --target_ratio 0.20 --seed 123
  python src/phase2_pruning.py --target_ratio 0.20 --seed 7

  # Include random pruning lower-bound baseline
  python src/phase2_pruning.py --include_rand

  # Section 6.2 scope boundary experiment
  python src/phase2_pruning.py --scope_boundary --target_ratio 0.40
        """,
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Quick sanity check: single ratio 40%%, 1 epoch, 8 samples, no KD loss",
    )
    parser.add_argument(
        "--target_ratio", type=float, default=None,
        help="Pruning ratio for full model saving (default: 0.40)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: 42). Use 42/123/7 for multi-seed runs.",
    )
    parser.add_argument(
        "--soft", action="store_true",
        help="Soft pruning (zero weights) instead of structural (matrix reshape)",
    )
    parser.add_argument(
        "--no_lora", action="store_true",
        help="Disable LoRA — use full fine-tuning",
    )
    parser.add_argument(
        "--include_rand", action="store_true",
        help="Include random pruning baseline in Pareto sweep",
    )
    parser.add_argument(
        "--scope_boundary", action="store_true",
        help="RefCOCO fine-tuning for Section 6.2 scope boundary experiment",
    )
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=None,
        help="Override Pareto sweep ratios (e.g. --ratios 0.40 or --ratios 0.20 0.40)",
    )
    args = parser.parse_args()
    main(args)