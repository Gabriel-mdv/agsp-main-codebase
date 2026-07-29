#!/usr/bin/env python3
"""
recalibrate_only.py
───────────────────
Re-apply Stage 2 recalibration to already-saved Phase 2 models.

Use this after downloading COCO train2014, if the original Phase 2 run
completed recalibration with missing images (loss = 0.0000).

Usage:
  python src/recalibrate_only.py
  python src/recalibrate_only.py --seed 42
"""

import argparse
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "MobileVLM_code"))

from config import get_config
from utils.finetuning import apply_lora
from utils.finetuning import finetune_recalibration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

OCEAN_OUT = "/ocean/projects/cis260140p/bgodwin/outputs"


def main():
    parser = argparse.ArgumentParser(description="Re-run Stage 2 recalibration only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategies", nargs="+", default=["agsp", "mbp"],
                        help="Which saved models to recalibrate")
    parser.add_argument("--ratio", type=int, default=40,
                        help="Pruning ratio percentage (default: 40)")
    args = parser.parse_args()

    cfg = get_config()

    from transformers import AutoTokenizer, CLIPImageProcessor
    from mobilevlm.model.mobilellama import MobileLlamaForCausalLM

    for strategy in args.strategies:
        model_dir = os.path.join(OCEAN_OUT, f"model_{strategy}_{args.ratio}_seed{args.seed}")
        if not os.path.exists(model_dir):
            logger.error("Model not found: %s — skipping.", model_dir)
            continue

        logger.info("=" * 65)
        logger.info("Loading saved %s model from %s", strategy.upper(), model_dir)
        logger.info("=" * 65)

        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14-336",
            cache_dir=cfg.paths.model_cache_dir,
        )
        model = MobileLlamaForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": 0},
            ignore_mismatched_sizes=True,
        )

        # Apply fresh LoRA adapters — gives the optimizer trainable parameters.
        # The saved model has Stage 1 LoRA baked into weights; recalibration
        # trains a new lightweight adapter on top.
        model = apply_lora(
            model,
            lora_r=cfg.phase2.lora_r,
            lora_alpha=cfg.phase2.lora_alpha,
            lora_dropout=cfg.phase2.lora_dropout,
            target_modules=cfg.phase2.lora_target_modules,
        )

        logger.info("Applying Stage 2 recalibration for %s...", strategy.upper())
        model, recal_losses = finetune_recalibration(
            model, tokenizer, image_processor, cfg,
            device=cfg.hardware.device,
        )
        logger.info(
            "Recalibration complete. Losses: %s",
            [f"{l:.4f}" for l in recal_losses],
        )

        # Re-save the recalibrated model
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        image_processor.save_pretrained(model_dir)
        logger.info("%s model re-saved → %s", strategy.upper(), model_dir)

        # Update the pruned_heads JSON with recalibration info
        heads_path = os.path.join(
            cfg.paths.phase2_dir, f"pruned_heads_{strategy}_seed{args.seed}.json"
        )
        if os.path.exists(heads_path):
            with open(heads_path) as f:
                heads_data = json.load(f)
            heads_data["summary"]["recalibration_losses"] = recal_losses
            with open(heads_path, "w") as f:
                json.dump(heads_data, f, indent=2)

        del model
        torch.cuda.empty_cache()

    logger.info("Recalibration-only pass complete.")


if __name__ == "__main__":
    main()
