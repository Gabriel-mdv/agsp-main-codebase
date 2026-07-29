#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh — PSC Bridges-2 SLURM submission script
# AGSP-VLM · MobileVLM V2-1.7B
#
# PURPOSE
# ───────
# Runs the full AGSP pipeline (Phase 1 → Phase 2 → Phase 3) as a single
# SLURM job on PSC Bridges-2. Uses bfloat16 throughout (matches paper).
#
# GPU PRIORITY
# ────────────
#   1. A100-40GB (GPU partition)         — preferred, fastest
#   2. V100-32GB (GPU-shared partition)  — fallback, tested in paper
#   3. H100-80GB (GPU-HM partition)      — last resort
#
# USAGE
# ─────
#   sbatch run_pipeline.sh                              # full pipeline
#   sbatch --export=PHASE=phase1 run_pipeline.sh        # Phase 1 only
#   sbatch --export=PHASE=phase2 run_pipeline.sh        # Phase 2 only
#   sbatch --export=PHASE=phase3 run_pipeline.sh        # Phase 3 only
#   sbatch --export=EXTRA_ARGS="--run_ablation" run_pipeline.sh
#
#   squeue -u $USER
#   tail -f logs/agsp_job_<JOBID>.log
# ─────────────────────────────────────────────────────────────────────────────

# ── SLURM configuration ───────────────────────────────────────────────────────
#SBATCH --job-name=agsp_mobilevlm
#SBATCH --output=logs/agsp_job_%j.log
#SBATCH --error=logs/agsp_job_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G

# ── GPU selection ─────────────────────────────────────────────────────────────
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1

## Uncomment for A100-40GB:
## #SBATCH --partition=GPU
## #SBATCH --gres=gpu:a100-40:1

## Uncomment for H100-80GB:
## #SBATCH --partition=GPU-HM
## #SBATCH --gres=gpu:h100-80:1

# ─────────────────────────────────────────────────────────────────────────────
# NOTE: set -u is temporarily disabled around conda activation because
# the MKL activate script references unbound variables (MKL_INTERFACE_LAYER).
# set -u is re-enabled immediately after activation completes.
# ─────────────────────────────────────────────────────────────────────────────
set -eo pipefail   # -e and -o pipefail only, NOT -u (see note above)

# ── Logging helpers ───────────────────────────────────────────────────────────
LOGDIR="logs"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
hr()  { echo "────────────────────────────────────────────────────────────────"; }

hr
log "Job ID:    $SLURM_JOB_ID"
log "Node:      $SLURMD_NODENAME"
log "Partition: $SLURM_JOB_PARTITION"
log "GPUs:      $SLURM_GPUS_ON_NODE"
log "CPUs:      $SLURM_CPUS_PER_TASK"
log "Memory:    ${SLURM_MEM_PER_NODE:-unknown} MB"
hr

# ── GPU detection and dtype configuration ─────────────────────────────────────
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
          2>/dev/null | head -1 || echo "0")
log "GPU VRAM detected: ${GPU_MEM} MiB"

if [ "$GPU_MEM" -lt 8000 ]; then
    log "WARNING: GPU has < 8GB VRAM. Results will NOT match paper."
    export AGSP_QUANT=4bit
elif [ "$GPU_MEM" -lt 20000 ]; then
    log "WARNING: GPU has < 20GB VRAM. Using 8-bit (not paper-matching)."
    export AGSP_QUANT=8bit
else
    log "INFO: Sufficient VRAM (>= 20GB). Using bfloat16 — matches paper."
    export AGSP_QUANT=bfloat16
fi

QUANT_OVERRIDE_FILE="${PROJECT_DIR:-$PWD}/agsp_quant_override.txt"
echo "$AGSP_QUANT" > "$QUANT_OVERRIDE_FILE"
log "AGSP_QUANT=$AGSP_QUANT written to $QUANT_OVERRIDE_FILE"

# ── Environment setup ─────────────────────────────────────────────────────────
log "Loading modules..."
module purge

CUDA_VERSION=""
for candidate in "cuda/12.4" "cuda/12.3" "cuda/12.2" "cuda/12.1" "cuda/12.0" \
                 "cuda/11.8" "cuda/11.7"; do
    if module avail "$candidate" 2>/dev/null | grep -q "$candidate"; then
        CUDA_VERSION="$candidate"
        break
    fi
done

if [ -n "$CUDA_VERSION" ]; then
    module load "$CUDA_VERSION"
    log "Loaded: $CUDA_VERSION"
else
    log "WARNING: No CUDA module found. Attempting to proceed."
fi

module load anaconda3

# ── Conda activation ──────────────────────────────────────────────────────────
# set +u: allow unbound variables during conda/MKL activation scripts.
# The MKL activate script references MKL_INTERFACE_LAYER which may be unset.
# set -u is restored immediately after activation.
set +u
if conda activate /ocean/projects/cis260140p/bgodwin/envs/agsp 2>/dev/null; then
    log "Conda environment activated."
else
    log "Activation failed — trying to create environment..."
    conda env create -f environment.yml \
        --prefix /ocean/projects/cis260140p/bgodwin/envs/agsp
    conda activate /ocean/projects/cis260140p/bgodwin/envs/agsp
fi
set -u
# ─────────────────────────────────────────────────────────────────────────────

export PYTHONPATH=$PYTHONPATH:~/agsp_neurips/MobileVLM_code

log "Python:         $(python --version)"
log "PyTorch:        $(python -c 'import torch; print(torch.__version__)')"
log "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
log "GPU name:       $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
log "bfloat16 ok:    $(python -c 'import torch; print(torch.cuda.is_bf16_supported())')"

# ── Project directory ─────────────────────────────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
log "Working directory: $PROJECT_DIR"

mkdir -p logs outputs/phase1 outputs/phase2 outputs/phase3

# ── Arguments ─────────────────────────────────────────────────────────────────
PHASE="${PHASE:-all}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SEED="${SEED:-42}"

log "Pipeline config: PHASE=$PHASE | SEED=$SEED | EXTRA_ARGS='$EXTRA_ARGS'"

# ── Phase runner ──────────────────────────────────────────────────────────────
run_phase() {
    local name="$1"
    local script="$2"
    shift 2
    local phase_args="$@"

    hr
    log ">>> Starting: $name"
    log "    Command:  python $script $phase_args"
    hr

    time python "$script" $phase_args 2>&1 \
        | tee -a "$LOGDIR/agsp_job_${SLURM_JOB_ID}.log"
    local rc=${PIPESTATUS[0]}

    if [ $rc -ne 0 ]; then
        log "ERROR: $name failed with exit code $rc"
        exit $rc
    fi

    log "<<< Completed: $name"
    hr
}

# ── Pipeline execution ────────────────────────────────────────────────────────
log "Starting AGSP pipeline (PHASE=$PHASE)..."

case "$PHASE" in
    phase1)
        run_phase "Phase 1 — AIS Scoring" src/phase1_baseline.py \
            --n_samples 100 \
            --run_ablation
        ;;
    phase2)
        run_phase "Phase 2 — Pruning + Fine-Tuning" src/phase2_pruning.py \
            --seed "$SEED"
        ;;
    phase3)
        run_phase "Phase 3 — Evaluation" src/phase3_eval.py \
            --pope_low_compression \
            --run_ablation \
            $EXTRA_ARGS
        ;;
    all|*)
        run_phase "Phase 1 — AIS Scoring" src/phase1_baseline.py \
            --n_samples 100 \
            --run_ablation

        run_phase "Phase 2 — Pruning + Fine-Tuning" src/phase2_pruning.py \
            --seed "$SEED"

        run_phase "Phase 3 — Evaluation" src/phase3_eval.py \
            --pope_low_compression \
            --run_ablation \
            $EXTRA_ARGS
        ;;
esac

# ── Summary ───────────────────────────────────────────────────────────────────
hr
log "Pipeline complete."
log ""
log "Key outputs:"
log "  Phase 1: outputs/phase1/ais_matrix.npy"
log "  Phase 1: outputs/phase1/ais_summary.json"
log "  Phase 1: outputs/phase1/calibration_ablation.json  (Appendix A)"
log "  Phase 2: /ocean/.../pareto_data_seed${SEED}.json"
log "  Phase 2: /ocean/.../model_agsp_40_seed${SEED}/"
log "  Phase 2: /ocean/.../model_mbp_40_seed${SEED}/"
log "  Phase 3: outputs/phase3/results_table.md"
log "  Phase 3: outputs/phase3/pope_low_compression.json  (Appendix C)"
log "  Phase 3: outputs/phase3/latency_flops_table.json   (Appendix D)"
log ""
log "Full log: $LOGDIR/agsp_job_${SLURM_JOB_ID}.log"
hr

# ── Archive results ───────────────────────────────────────────────────────────
ARCHIVE="agsp_results_job${SLURM_JOB_ID}_seed${SEED}.tar.gz"
tar -czf "$ARCHIVE" \
    outputs/ \
    logs/agsp_job_${SLURM_JOB_ID}* \
    2>/dev/null || true

log "Results archived → $ARCHIVE"
log "Copy to ocean: cp $ARCHIVE /ocean/projects/cis260140p/bgodwin/submissions/"