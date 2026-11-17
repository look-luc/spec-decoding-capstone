#!/usr/bin/env bash
#
# SLURM job-array sweep for distillation hyperparameter search.
#
# Each array task = one (lr, steps, grad_accum, warmup, weight_decay) combo.
# All tasks run in parallel (up to Alpine's 21-GPU per-user limit on aa100).
#
# Quick sweep  (8 combos):  each run ~30-45 min  → done in 1-2 hours total
# Full  sweep (324 combos): each run ~30-60 min  → done in ~3-5 hours total
#                           (21 jobs run simultaneously, so 324/21 ≈ 16 waves)
#
# ── Quick sweep (8 runs) ────
#   sbatch --array=0-7   scripts/run_params_sweep_alpine.sh
#
# ── Full sweep (324 runs) ────
#   sbatch --array=0-323 scripts/run_params_sweep_alpine.sh
#
# ── Override language / mode ─────
#   sbatch --array=0-323 --export=LANG_CODE=oci,MODE=general \
#          scripts/run_params_sweep_alpine.sh
#
# ── Multiple languages (each language = its own array batch) ───
#   for lang in ber npi haw smo; do
#     sbatch --array=0-323 --export=LANG_CODE=$lang,MODE=general \
#            --job-name=distill-${lang} scripts/run_params_sweep_alpine.sh
#   done
#

#!/usr/bin/env bash
#
# SLURM job-array sweep for Eagle training & evaluation hyperparameters.
#
# Usage:
#   sbatch --array=0-7 scripts/run_params_sweep_alpine.sh
#

#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --account=ucb-general
#SBATCH --job-name=eagle-sweep
#SBATCH --output=logs/eagle_sweep_%A_%a.out
#SBATCH --error=logs/eagle_sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ── Defaults ───
LANG_CODE="${LANG_CODE:-ber}"
CONFIG="experiments/eagle_greedy.cfg"

# ── Hyperparameter grid for Eagle ───
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

LR_GRID=(5e-5 1e-4)
STEPS_GRID=(3000 5000)
GA_GRID=(2 4)
GAMMA_GRID=(4 5)

N_GAMMA="${#GAMMA_GRID[@]}"
N_GA="${#GA_GRID[@]}"
N_STEPS="${#STEPS_GRID[@]}"

GAMMA_IDX=$(( TASK_ID % N_GAMMA ))
GA_IDX=$(( (TASK_ID / N_GAMMA) % N_GA ))
STEPS_IDX=$(( (TASK_ID / (N_GAMMA * N_GA)) % N_STEPS ))
LR_IDX=$(( (TASK_ID / (N_GAMMA * N_GA * N_STEPS)) ))

LR="${LR_GRID[$LR_IDX]}"
STEPS="${STEPS_GRID[$STEPS_IDX]}"
GA="${GA_GRID[$GA_IDX]}"
GAMMA="${GAMMA_GRID[$GAMMA_IDX]}"

LABEL="lr${LR}_s${STEPS}_ga${GA}_g${GAMMA}"

module purge
module load uv

source .venv/bin/activate

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export PYTHONPATH="$(pwd)"
export HF_HOME="/scratch/alpine/${USER}/hf_cache"
export WANDB_DIR="/scratch/alpine/${USER}/wandb"
mkdir -p "${HF_HOME}" "${WANDB_DIR}"

OUTPUT_DIR="checkpoints/eagle_${LANG_CODE}_${LABEL}"

echo "======================================================================"
echo "  Eagle Array Job: ${SLURM_ARRAY_JOB_ID}, task ${TASK_ID}"
echo "  Node:       $(hostname)"
echo "  Language:   ${LANG_CODE}"
echo "  Combo:      lr=${LR} steps=${STEPS} ga=${GA} gamma=${GAMMA}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Start:      $(date)"
echo "======================================================================"

# Run through the main experiment driver run.py, which triggers train_eagle.py
# if the Eagle checkpoint doesn't exist yet, then evaluates it.
python run.py "${CONFIG}" \
    -o language_code="${LANG_CODE}" \
       output_dir="${OUTPUT_DIR}" \
       learning_rate="${LR}" \
       max_steps="${STEPS}" \
       grad_accum_steps="${GA}" \
       gamma="${GAMMA}"

echo ""
echo "  Done: $(date)"
echo "=========================="
