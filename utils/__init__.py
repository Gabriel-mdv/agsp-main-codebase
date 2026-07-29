"""
utils/__init__.py
─────────────────
Utility package for AGSP-VLM (MobileVLM V2-1.7B).

Submodules
──────────
  alignment_scoring   AIS computation, bootstrap stability, calibration ablation
  pruning             AGSP / MBP / RAND structural pruning
  finetuning          LoRA Stage 1 (COCO captions) + Stage 2 (recalibration)
  data_utils          Dataset classes: COCOCaptions, LLaVAYesNo, RefCOCO
  evaluation          POPE, latency, VRAM, AIS ablation
  visualization       AIS plots, Pareto curves, comparison figures
  vqa_utils           VQAv2 loader and soft accuracy scorer
  nuscenes_utils      nuScenes safety evaluation (H3)
"""
