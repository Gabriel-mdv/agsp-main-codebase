# AGSP-VLM
## Attention-Guided Structural Pruning for Cross-Modal Alignment Preservation in Vision-Language Models

**Carnegie Mellon University Africa · 2026**  
Boniface Chukwuemeka Godwin · Jean Gabriel Mpuhwezimana · Mohamed Awud

---

## What is this project?

Vision-Language Models (VLMs) connect images and text through attention heads
that learn to associate visual regions with words. When you compress these
models by removing attention heads — a process called structural pruning —
you risk destroying the cross-modal grounding that makes them useful.

The standard way to decide which heads to remove is **weight magnitude**: remove
the heads with the smallest weights. This is a proxy for how much a head
contributes to text generation. It says nothing about how much a head
contributes to *understanding images*.

This project introduces **AGSP (Attention-Guided Structural Pruning)**, which
instead asks: *how much does each head causally contribute to cross-modal
alignment?* We answer this by computing the gradient of a cross-modal alignment
loss with respect to each head's query projection — the Alignment Importance
Score (AIS). Heads with low AIS are removed; heads with high AIS are protected.

The central question is whether this criterion produces less grounding damage
than weight magnitude when the same number of heads are removed.

---

## The model

We experiment on **MobileVLM V2-1.7B** — a compact VLM designed for edge
deployment.

| Component | Details |
|---|---|
| Vision encoder | CLIP ViT-L/14 (frozen during pruning) |
| Projector | LDPv2 — 576 patch tokens → 144 visual tokens |
| Language backbone | MobileLLaMA-1.4B |
| Total attention heads | 24 layers × 16 heads = **384 heads** |
| VRAM (bfloat16) | ~3.4 GB |
| HuggingFace ID | `mtgv/MobileVLM_V2-1.7B` |

We chose MobileVLM V2-1.7B because it is small enough to run on edge hardware
(Jetson Orin Nano) while still having enough structure to make head pruning
meaningful. Its LDPv2 projector produces 144 visual tokens versus 576 in
LLaVA, reducing attention operations fourfold.

---

## Hypotheses

**H1 (Criterion Quality)**  
Gradient-based AIS is a better predictor of grounding damage than weight
magnitude. We test this by measuring training loss before any fine-tuning
recovery (Epoch-1 loss) — a clean diagnostic that isolates criterion quality
from recovery quality.

**H2 (Compression Robustness)**  
AGSP's advantage over magnitude-based pruning (MBP) grows as compression
increases. At 50% and 60% pruning, fewer heads remain, making the selection
criterion more consequential.

---

## Experiment design

The experiment runs in three sequential phases.

```
Phase 1                   Phase 2                    Phase 3
────────────────────      ─────────────────────      ──────────────────────
Compute AIS for all  →    Prune heads (AGSP vs   →   Evaluate on POPE
384 heads using           MBP vs Random) then         adversarial, VQAv2,
100 COCO captions         recover with LoRA            latency, VRAM
                          on 6,000 COCO captions
```

**Phase 1 — Alignment Importance Scoring**  
Backpropagate a cross-modal alignment loss through the full vision-language
pipeline on 100 MS-COCO val2017 image-caption pairs. Accumulate gradient
Frobenius norms per head. The result is a 24 × 16 matrix of AIS scores.

**Phase 2 — Structural Pruning + LoRA Fine-Tuning**  
Rank all 384 heads by AIS and remove the bottom N% (with a per-layer cap of
50%). The same experiment is repeated with heads ranked by L2 weight magnitude
(MBP baseline) and by random selection (lower-bound baseline). All three
strategies remove exactly the same number of heads — only the selection
criterion differs.

After pruning, each model is recovered using LoRA fine-tuning on 6,000 COCO
val2017 image-caption pairs for 3 epochs with knowledge distillation (λ=0.3)
against the unpruned teacher.

Models pruned at 40% or above undergo a second recalibration stage on 10,324
balanced yes/no QA pairs from LLaVA-Instruct-150K to correct output bias that
develops after heavy compression.

**Phase 3 — Evaluation**  
Each pruned model is evaluated on:
- POPE adversarial (500 samples, COCO val2014) — primary grounding metric
- Wall-clock inference latency and peak VRAM
- VQAv2 (2,000 samples) — general capability preservation (optional)
- nuScenes safety detection (optional, H3 hypothesis)

---

## Project structure

```
agsp_v4/
│
├── config.py                   All hyperparameters, paths, and settings
├── phase1_baseline.py          Phase 1: AIS computation
├── phase2_pruning.py           Phase 2: Pruning + LoRA fine-tuning
├── phase3_eval.py              Phase 3: POPE, VQA, latency evaluation
│
├── run_pipeline.sh             SLURM: submit full pipeline as one job
├── run_multiseed.sh            SLURM: submit 3-seed validation jobs
├── aggregate_multiseed.py      Compute mean ± std from multi-seed runs
│
├── environment.yml             Conda environment specification
├── setup_data.py               Download and organise all datasets
│
├── utils/
│   ├── __init__.py
│   ├── alignment_scoring.py    AIS formula, bootstrap stability, calibration ablation
│   ├── pruning.py              AGSP / MBP / RAND structural pruning logic
│   ├── finetuning.py           LoRA Stage 1 (COCO) + Stage 2 (recalibration)
│   ├── data_utils.py           Dataset classes: COCO captions, LLaVA yes/no, RefCOCO
│   ├── evaluation.py           POPE, latency, VRAM, AIS ablation
│   ├── visualization.py        AIS plots, Pareto curves, comparison figures
│   ├── vqa_utils.py            VQAv2 loader and soft accuracy scorer
│   └── nuscenes_utils.py       nuScenes safety evaluation (H3)
│
├── MobileVLM_code/             MobileVLM V2 source — cloned from GitHub (see Setup)
├── tests/
│   └── test_smoke.py           Quick smoke tests to verify environment
│
├── data/                       Datasets — populated by setup_data.py
│   ├── coco/
│   │   ├── annotations/        captions_val2017.json
│   │   ├── val2017/            COCO val2017 images (calibration + fine-tuning)
│   │   └── val2014/            COCO val2014 images (POPE evaluation)
│   ├── pope/                   coco_pope_adversarial.json
│   ├── refcoco/                RefCOCO val.json (scope boundary experiment)
│   ├── llava_instruct/         yesno_10324.json (output recalibration)
│   └── vqa/                    VQAv2 questions + annotations (optional)
│
├── outputs/                    Created automatically when experiments run
│   ├── phase1/                 ais_matrix.npy, ais_summary.json, figures
│   ├── phase2/                 pareto_data.json, pruned_heads_*.json
│   └── phase3/                 results_table.md, results_all.json, figures
│
└── logs/                       SLURM job logs (agsp_job_<JOBID>.log)
```

---

## Setup

Work through these steps in order. Do not skip any step.

### Step 1 — Create the project folder

```bash
mkdir ~/agsp_v4
cd ~/agsp_v4
```

Copy all project files into this folder (config.py, phase scripts, utils/,
run scripts, environment.yml, setup_data.py, tests/).

### Step 2 — Check available CUDA on Bridges-2

```bash
module avail cuda
```

Open `environment.yml` and set `pytorch-cuda` to the highest available version.
For example if `cuda/12.1` is available, set `pytorch-cuda=12.1`. If only
`cuda/11.8` is available, set `pytorch-cuda=11.8`.

### Step 3 — Create the conda environment

```bash
module load anaconda3
conda env create -f environment.yml
conda activate agsp
```

If the environment already exists from a previous attempt and you want a clean
start:

```bash
conda env remove -n agsp
conda env create -f environment.yml
conda activate agsp
```

Alternatively, if you prefer pip over conda:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 4 — Clone and install MobileVLM V2 source

The `MobileLlamaForCausalLM` class used in Phase 2 is not part of standard
transformers. You must clone the official MobileVLM V2 repository and install
it as a local package.

```bash
cd ~/agsp_v4
git clone https://github.com/Meituan-AutoML/MobileVLM.git MobileVLM_code
cd MobileVLM_code
pip install -e .
cd ..
```

This creates the `MobileVLM_code/` folder inside your project. Do not rename
it — the phase scripts import from this exact path.

Verify the install:

```bash
python -c "from mobilevlm.model.mobilellama import MobileLlamaForCausalLM; print('OK')"
```

### Step 5 — Download datasets

```bash
python setup_data.py
```

This downloads and organises all required datasets into `data/`.

| Dataset | Used for | Approx. size |
|---|---|---|
| MS-COCO val2017 images | AIS calibration + LoRA fine-tuning | ~1 GB |
| MS-COCO val2017 annotations | Caption loading | ~1 MB |
| MS-COCO val2014 images | POPE evaluation | ~6 GB |
| POPE adversarial JSON | Primary evaluation benchmark | ~2 MB |
| RefCOCO val JSON | Scope boundary experiment (Section 6.2) | ~500 MB |
| LLaVA-Instruct yes/no | Output recalibration (40%+ pruning) | ~50 MB |
| VQAv2 val2014 | Optional general VQA evaluation | ~7 GB |

VQAv2 is large and optional. If you want to skip it:

```bash
python setup_data.py --skip_vqa
```

### Step 6 — Verify setup

```bash
# Check GPU, CUDA, and bfloat16 support
python -c "
import torch
print('CUDA available:    ', torch.cuda.is_available())
print('GPU:               ', torch.cuda.get_device_name(0))
print('bfloat16 supported:', torch.cuda.is_bf16_supported())
print('PyTorch version:   ', torch.__version__)
"
```

All three of CUDA, GPU, and bfloat16 must be available. If bfloat16 is `False`
your GPU does not support it and results will not match the paper.

```bash
# Run smoke tests to confirm the full environment is working
python -m pytest tests/test_smoke.py -v
```

All tests should pass before you run any experiments.

---

## Running the experiment

### Option A — Full pipeline as a single SLURM job (recommended)

Submit Phase 1 → Phase 2 → Phase 3 as one job on a single GPU node.

```bash
sbatch run_pipeline.sh
```

Estimated time: **12 hours on V100-32GB**.

Monitor progress:

```bash
squeue -u $USER
tail -f logs/agsp_job_<JOBID>.log
```

When complete, your primary results are in:

```
outputs/phase3/results_table.md
```

### Option B — Run phases individually

Use this if you want to inspect outputs between phases, or if a phase fails
and you need to resume without restarting from the beginning.

```bash
# Phase 1: AIS computation (~45 minutes on V100)
python phase1_baseline.py --n_samples 100 --run_ablation

# Check Phase 1 output before continuing
cat outputs/phase1/ais_summary.json

# Phase 2: Pruning + fine-tuning (~4 hours on V100)
python phase2_pruning.py

# Phase 3: Evaluation (~2 hours on V100)
python phase3_eval.py --pope_low_compression --run_ablation
```

### Option C — Submit phases as separate SLURM jobs

Use this if you want each phase on a separate node or need to queue them
independently.

```bash
# Phase 1 only
sbatch --export=PHASE=phase1 run_pipeline.sh

# Phase 2 only — run after Phase 1 completes
sbatch --export=PHASE=phase2 run_pipeline.sh

# Phase 3 only — run after Phase 2 completes
sbatch --export=PHASE=phase3 run_pipeline.sh
```

---

## Multi-seed statistical validation

The primary experiments use a single seed (42). To provide statistical
confidence for the Pareto table at the two most important compression ratios:

```bash
# Submit 6 parallel jobs (3 seeds × 2 ratios: 20% and 60%)
bash run_multiseed.sh

# Preview what will be submitted without actually submitting
bash run_multiseed.sh --dry_run
```

After all 6 jobs complete (~8 hours each), aggregate the results:

```bash
python aggregate_multiseed.py
```

This prints an updated Table 3 with mean ± std and saves it to
`outputs/phase2/pareto_table_multiseed.json`.

---

## What to expect at each phase

### After Phase 1

Check `outputs/phase1/ais_summary.json`. These values confirm the AIS
computation ran correctly:

```
gini:                ~0.232
mean:                ~0.993
top5_bottom50_ratio: ~3.3×
most critical head:  Layer 18, Head 9  (AIS ≈ 3.477)
bootstrap rho:       ~0.9963
```

Also check `outputs/phase1/fig_ais_distribution.png` to visually confirm
the right-skewed distribution with a small number of high-AIS heads.

### After Phase 2

Check `outputs/phase2/pareto_data.json`. The Epoch-1 loss gap is the key
result — lower Epoch-1 loss means less grounding damage from the pruning
criterion:

| Ratio | AGSP Epoch-1 | MBP Epoch-1 |
|---|---|---|
| 20% | ~1.78 | ~3.04 |
| 40% | ~1.97 | ~3.94 |
| 60% | ~2.01 | ~4.23 |

AGSP should consistently cause ~2× less damage than MBP across all ratios.

### After Phase 3

Check `outputs/phase3/results_table.md`:

| Model | POPE Acc. | F1 | Yes-ratio |
|---|---|---|---|
| Baseline (unpruned) | ~0.812 | ~0.832 | ~0.620 |
| AGSP-40% (recalibrated) | ~0.566 | ~0.662 | ~0.782 |
| MBP-40% (recalibrated) | ~0.506 | ~0.656 | ~0.934 |

---

## Hyperparameters

All hyperparameters are in `config.py`. Nothing needs changing for
standard paper-matching experiments.

| Parameter | Value | Purpose |
|---|---|---|
| Calibration samples | 100 | AIS computation (stable at this size) |
| Per-layer pruning cap | 50% max | Prevents total layer collapse |
| LoRA rank | 16 | Efficient fine-tuning adapter |
| LoRA alpha | 32 | Standard 2× rank scaling |
| Fine-tuning LR | 1e-5 | Conservative for LoRA stability |
| Batch size | 4 | Fits V100-32GB in bfloat16 |
| Fine-tuning epochs | 3 | Sufficient for LoRA convergence |
| KD loss λ | 0.3 | Teacher distribution preservation |
| Recalibration LR | 5e-6 | Conservative for output bias correction |
| POPE samples | 500 | Standard adversarial evaluation |
| dtype | bfloat16 | Stable training, matches paper |

---

## Troubleshooting

**CUDA out of memory**  
Reduce `finetune_batch_size` in `config.py` from 4 to 2. Alternatively
switch to the A100 partition by uncommenting the A100 block in
`run_pipeline.sh`.

**Phase 2 fails with "AIS matrix not found"**  
Phase 1 must complete successfully before Phase 2. Verify that
`outputs/phase1/ais_matrix.npy` exists.

**Phase 3 fails with "model not found"**  
Phase 3 looks for checkpoints saved by Phase 2 at
`/ocean/.../model_agsp_40_seed42/`. Verify Phase 2 completed and that
the `--seed` flag matches between Phase 2 and Phase 3 (default: 42).

**POPE images missing**  
Run `python setup_data.py` to re-download COCO val2014. Check that
`config.paths.pope_images_dir` points to `data/coco/val2014`.

**MobileVLM import error**  
Run the clone and install step again:
```bash
cd ~/agsp_v4/MobileVLM_code
pip install -e .
```
Then verify: `python -c "from mobilevlm.model.mobilellama import MobileLlamaForCausalLM; print('OK')"`

**bfloat16 not supported**  
Results will not match the paper. As a fallback set `load_in_8bit=True`
in `config.py` — but expect degraded and non-comparable results.

**Smoke tests failing**  
Check which test fails and read the error. Common causes: MobileVLM not
installed, COCO data not downloaded, wrong CUDA version in environment.yml.

---

## Reproducibility

All random operations are seeded. Use the same seed in Phase 2 and Phase 3
to ensure checkpoints and evaluations stay in sync:

```bash
python phase2_pruning.py --seed 42
python phase3_eval.py    --seed 42
```

The COCO calibration sample order in Phase 1 is fixed by `seed=42` in
`config.py`. Bootstrap stability confirms AIS rankings are consistent across
different random subsets (Spearman ρ ≈ 0.9963).

---

## Citation

```bibtex
@inproceedings{godwin2026agsp,
  title     = {Attention-Guided Structural Pruning for Cross-Modal
               Alignment Preservation in Vision-Language Models},
  author    = {Godwin, Boniface Chukwuemeka and
               Mpuhwezimana, Jean Gabriel and
               Awud, Mohamed},
  booktitle = {DLI 2026},
  year      = {2026},
}
```

---

## Contact

Boniface Godwin — bgodwin@andrew.cmu.edu  
Jean Gabriel Mpuhwezimana — jmpuhwez@andrew.cmu.edu  
Mohamed Awud — mawud@andrew.cmu.edu  

Carnegie Mellon University Africa, Kigali, Rwanda
