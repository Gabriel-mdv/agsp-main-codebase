#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_multiseed.sh — Multi-seed Pareto experiments for Stage 3 validation
# AGSP-VLM · MobileVLM V2-1.7B
#
# PURPOSE
# ───────
# Runs Phase 2 + Phase 3 with multiple seeds at the two extreme pruning
# ratios (20% and 60%) to provide statistical validation of the Pareto
# table claims. This addresses the paper's single-seed limitation.
#
# WHY 20% AND 60%?
# ─────────────────
# - 20%: largest claimed advantage (Δ = +0.088). Needs statistical anchoring.
# - 60%: peak claimed advantage (Δ = +0.119). Needs statistical anchoring.
# - 30% and 40%: near-tie (Δ = -0.001 and +0.003) — within noise, not worth
#   multi-seed validation since we already disclaim them in the paper.
# - 50%: moderate advantage (Δ = +0.085) — covered by 60% trend.
#
# SEEDS
# ─────
# Seeds: 42 (primary), 123, 7
# These correspond to cfg.hardware.seeds in config.py.
# Seed 42 was used in original paper experiments.
# Seeds 123 and 7 are the new validation seeds.
#
# WHAT THIS PRODUCES
# ──────────────────
# For each (ratio, seed) combination:
#   /ocean/.../pareto_data_seed{seed}.json
#   /ocean/.../model_agsp_{ratio_pct}_seed{seed}/
#   /ocean/.../model_mbp_{ratio_pct}_seed{seed}/
#   outputs/phase3/results_all_ratio{ratio_pct}_seed{seed}.json
#
# After all jobs complete, run aggregate_multiseed.py to compute
# mean ± std and update Table 3 in the paper.
#
# USAGE
# ─────
#   # Submit all 6 jobs (3 seeds × 2 ratios) at once
#   bash run_multiseed.sh
#
#   # Submit only 20% ratio jobs
#   bash run_multiseed.sh --ratio 0.20
#
#   # Dry run (print commands without submitting)
#   bash run_multiseed.sh --dry_run
#
# ESTIMATED COMPUTE
# ─────────────────
# Phase 2 per job:  ~3-4 hours on V100-32GB
# Phase 3 per job:  ~1-2 hours on V100-32GB
# Total (6 jobs):   ~24-36 hours (jobs run in parallel on separate nodes)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SEEDS=(42 123 7)                        # must match cfg.hardware.seeds
RATIOS=(0.20 0.60)                      # extreme ratios where anchoring matters
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
LOGDIR="$PROJECT_DIR/logs"
OCEAN_OUT="/ocean/projects/cis250019p/bgodwin/outputs"

mkdir -p "$LOGDIR" "$OCEAN_OUT"

# ── Argument parsing ──────────────────────────────────────────────────────────
DRY_RUN=false
FILTER_RATIO=""

for arg in "$@"; do
    case "$arg" in
        --dry_run)   DRY_RUN=true ;;
        --ratio=*)   FILTER_RATIO="${arg#*=}" ;;
    esac
done

# ── Submit function ───────────────────────────────────────────────────────────
submit_job() {
    local ratio="$1"
    local seed="$2"
    local ratio_pct
    ratio_pct=$(python3 -c "print(int($ratio * 100))")

    local job_name="agsp_seed${seed}_r${ratio_pct}"
    local log_file="$LOGDIR/${job_name}_%j.log"
    local err_file="$LOGDIR/${job_name}_%j.err"

    # SLURM submission command
    local sbatch_cmd=(
        sbatch
        --job-name="$job_name"
        --output="$log_file"
        --error="$err_file"
        --time=08:00:00
        --nodes=1
        --ntasks-per-node=1
        --cpus-per-task=8
        --mem=64G
        --partition=GPU-shared
        --gres=gpu:v100-32:1
        --export="ALL,SEED=$seed,RATIO=$ratio,RATIO_PCT=$ratio_pct,PROJECT_DIR=$PROJECT_DIR"
        --wrap="bash $PROJECT_DIR/run_singleseed.sh"
    )

    if $DRY_RUN; then
        echo "[DRY RUN] Would submit: ratio=${ratio} seed=${seed}"
        echo "  Job name: $job_name"
        echo "  Command: ${sbatch_cmd[*]}"
    else
        job_id=$("${sbatch_cmd[@]}")
        echo "Submitted $job_name → Job ID: $job_id"
        echo "  Monitor: tail -f $LOGDIR/${job_name}_${job_id#Submitted batch job }.log"
    fi
}

# ── Generate per-job runner script ────────────────────────────────────────────
# Creates run_singleseed.sh which is called by each SLURM job.
# This script runs Phase 2 + Phase 3 for one (ratio, seed) combination.
cat > "$PROJECT_DIR/run_singleseed.sh" << 'SINGLESEED'
#!/bin/bash
set -euo pipefail

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
hr()  { echo "────────────────────────────────────────────────────────────────"; }

hr
log "Single-seed job: ratio=${RATIO} seed=${SEED}"
log "Node: $SLURMD_NODENAME | Job: $SLURM_JOB_ID"
hr

# Load environment
module purge
for candidate in "cuda/12.4" "cuda/12.3" "cuda/12.2" "cuda/12.1" "cuda/12.0" "cuda/11.8"; do
    if module avail "$candidate" 2>/dev/null | grep -q "$candidate"; then
        module load "$candidate"
        log "Loaded: $candidate"
        break
    fi
done
module load anaconda3
conda activate agsp

cd "${PROJECT_DIR:-$PWD}"

# ── Phase 2: Pruning + Fine-tuning ────────────────────────────────────────────
# Run at the single target ratio with the specified seed.
# --include_rand adds random pruning for Stage 3 lower-bound baseline.
log "Phase 2: pruning + fine-tuning (ratio=${RATIO}, seed=${SEED})..."
python phase2_pruning.py \
    --target_ratio "$RATIO" \
    --seed "$SEED" \
    --include_rand

log "Phase 2 complete."

# ── Phase 3: Evaluation ───────────────────────────────────────────────────────
# Evaluate at the same ratio and seed.
# --pope_low_compression is only relevant for 20% and 30% ratios.
log "Phase 3: evaluation (ratio=${RATIO}, seed=${SEED})..."
if python3 -c "exit(0 if float('${RATIO}') <= 0.35 else 1)"; then
    # Low compression — evaluate without recalibration (Appendix C)
    python phase3_eval.py \
        --pruning_ratio "$RATIO" \
        --seed "$SEED" \
        --pope_low_compression
else
    # High compression — standard evaluation
    python phase3_eval.py \
        --pruning_ratio "$RATIO" \
        --seed "$SEED"
fi

# Save results with seed and ratio in filename for easy aggregation
cp outputs/phase3/results_all.json \
   "/ocean/projects/cis250019p/bgodwin/outputs/results_all_ratio${RATIO_PCT}_seed${SEED}.json"

log "Phase 3 complete. Results saved to ocean storage."
SINGLESEED

chmod +x "$PROJECT_DIR/run_singleseed.sh"

# ── Submit all jobs ───────────────────────────────────────────────────────────
echo ""
echo "AGSP Multi-Seed Pareto Validation"
echo "Seeds:  ${SEEDS[*]}"
echo "Ratios: ${RATIOS[*]}"
echo "Dry run: $DRY_RUN"
echo ""

total_jobs=0
for ratio in "${RATIOS[@]}"; do
    # Skip if ratio filter is set and doesn't match
    if [ -n "$FILTER_RATIO" ] && [ "$ratio" != "$FILTER_RATIO" ]; then
        continue
    fi

    for seed in "${SEEDS[@]}"; do
        submit_job "$ratio" "$seed"
        total_jobs=$((total_jobs + 1))
        sleep 1  # Brief pause to avoid SLURM rate limiting
    done
done

echo ""
echo "Submitted $total_jobs jobs."
echo ""
echo "Monitor all jobs:"
echo "  squeue -u \$USER"
echo ""
echo "After all jobs complete, aggregate results:"
echo "  python aggregate_multiseed.py"
