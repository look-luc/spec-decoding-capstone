#!/usr/bin/env bash
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --account=ucb-general
#SBATCH --job-name=eagle-sweep
#SBATCH --output=logs/eagle_%A_%a.out
#SBATCH --error=logs/eagle_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lude4390@colorado.edu

set -euo pipefail

# Capture config passed from sbatch (defaulting to eagle_greedy.cfg)
CONFIG="${1:-experiments/eagle_greedy.cfg}"
LANG_CODE="${LANG_CODE:-chr}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# ── 16-Combination Eagle Grid (Matching --array=0-15) ───
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

# Run via run.py for online Eagle training and evaluation
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
