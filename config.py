"""
config.py — Central configuration for AGSP-VLM (MobileVLM V2-1.7B).
All hyperparameters, paths, and experiment settings live here.

Changes from v3:
  - load_in_8bit removed; bfloat16 used throughout (matches paper)
  - seeds list added for multi-seed Pareto experiments
  - calibration_sizes_ablation added for Phase 1 ablation study
  - finetune_dataset changed to "coco_captions" (was RefCOCO — mismatch with paper)
  - teacher_model_id added for KD loss in Phase 2
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Paths:
    # MobileVLM V2 1.7B
    model_id: str        = "mtgv/MobileVLM_V2-1.7B"
    model_cache_dir: str = "./model_cache"

    output_dir: str            = "./outputs"
    phase1_dir: str            = "./outputs/phase1"
    phase2_dir: str            = "./outputs/phase2"
    phase3_dir: str            = "./outputs/phase3"
    pruned_model_agsp_dir: str = "./outputs/phase2/model_agsp"
    pruned_model_mbp_dir:  str = "./outputs/phase2/model_mbp"

    ais_matrix_path: str = "./outputs/phase1/ais_matrix.npy"

    # COCO val2017 — calibration (Phase 1) and fine-tuning (Phase 2)
    coco_val_annotations: str = "./data/coco/annotations/captions_val2017.json"
    coco_val_images_dir:  str = "./data/coco/val2017"

    # COCO val2014 — POPE evaluation (Phase 3)
    pope_val_path:   str = "./data/pope/coco_pope_adversarial.json"
    pope_images_dir: str = "./data/coco/val2014"

    # RefCOCO — scope boundary experiment only (not used for fine-tuning)
    refcoco_val_path: str = "./data/refcoco/val.json"

    # LLaVA-Instruct yes/no pairs — output recalibration (Phase 2 stage 2)
    llava_yesno_json:   str = "./data/llava_instruct/yesno_10324.json"
    llava_yesno_images: str = "./data/coco/train2014"

    # VQAv2 — optional Phase 3 evaluation
    vqa_questions_json:   str = "./data/vqa/v2_OpenEnded_mscoco_val2014_questions.json"
    vqa_annotations_json: str = "./data/vqa/v2_mscoco_val2014_annotations.json"
    vqa_images_dir:       str = "./data/coco/val2014"

    # nuScenes — optional H3 safety evaluation
    nuscenes_images_dir: str = "./data/nuscenes/samples"
    nuscenes_root:       str = "./data/nuscenes"


@dataclass
class Phase1Config:
    # Primary calibration set size (paper: 100)
    n_calibration_samples: int = 100

    # Calibration ablation sizes — for Appendix A
    calibration_sizes_ablation: List[int] = field(
        default_factory=lambda: [50, 100, 250, 500]
    )

    # Debiasing (Zhao et al. 2026) — RoPE recency + attention-sink correction
    use_debiased_scoring: bool = True

    # MobileVLM V2 LDPv2 projector: 144 visual tokens (not 576)
    n_visual_tokens: int = 144

    top_k_heads: int = 20


@dataclass
class Phase2Config:
    pruning_ratios: List[float] = field(
        default_factory=lambda: [0.20, 0.30, 0.40, 0.50, 0.60]
    )
    target_pruning_ratio: float = 0.40
    per_layer_cap: float        = 0.50
    pruning_mode: str           = "structural"

    # Fine-tuning dataset — MUST be coco_captions to match paper Table 3
    # RefCOCO is used ONLY in the scope boundary experiment (Section 6.2)
    finetune_dataset: str       = "coco_captions"
    n_finetune_samples: int     = 6000   # paper: 6,000 COCO val2017 captions

    finetune_epochs: int      = 3
    finetune_lr: float        = 1e-5
    finetune_batch_size: int  = 4

    # LoRA (paper: rank 16, alpha 32, q_proj + v_proj)
    use_lora: bool       = True
    lora_r: int          = 16
    lora_alpha: int      = 32
    lora_dropout: float  = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # Knowledge distillation — lambda=0.3 against unpruned teacher logits
    use_kd_loss: bool  = True
    kd_weight: float   = 0.3

    zero_pruned_grads: bool = True

    # Output recalibration (Phase 2 stage 2) — yes/no instruction fine-tuning
    recalibration_epochs: int = 3
    recalibration_lr: float   = 5e-6
    n_recalibration_samples: int = 10324


@dataclass
class Phase3Config:
    n_pope_samples:    int   = 500
    n_vqa_samples:     int   = 2000
    n_safety_samples:  int   = 100
    n_latency_trials:  int   = 10
    edge_latency_target_ms: float = 100.0
    safety_categories: List[str] = field(
        default_factory=lambda: [
            "pedestrian", "vehicle", "stop sign", "traffic light", "cyclist"
        ]
    )
    # POPE evaluation at lower compression ratios (no recalibration needed)
    pope_low_compression_ratios: List[float] = field(
        default_factory=lambda: [0.20, 0.30]
    )


@dataclass
class HardwareConfig:
    # bfloat16 throughout — matches paper and PSC A100/V100 capabilities
    torch_dtype: str         = "bfloat16"
    load_in_8bit: bool       = False   # disabled — use bfloat16 instead
    load_in_4bit: bool       = False   # fallback for <20GB VRAM only

    trust_remote_code: bool  = True
    attn_implementation: str = "eager"   # required for gradient access in Phase 1

    # Primary seed (paper experiments)
    seed: int = 42

    # Multi-seed list for statistical validation (Stage 3 experiments)
    seeds: List[int] = field(default_factory=lambda: [42, 123, 7])

    device: str = "cuda"


@dataclass
class Config:
    paths:    Paths          = field(default_factory=Paths)
    phase1:   Phase1Config   = field(default_factory=Phase1Config)
    phase2:   Phase2Config   = field(default_factory=Phase2Config)
    phase3:   Phase3Config   = field(default_factory=Phase3Config)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


def get_config() -> Config:
    cfg = Config()
    for attr in [
        "output_dir", "phase1_dir", "phase2_dir", "phase3_dir",
        "pruned_model_agsp_dir", "pruned_model_mbp_dir",
    ]:
        os.makedirs(getattr(cfg.paths, attr), exist_ok=True)
    return cfg


CFG = get_config()
