#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=7:30:00
#SBATCH --output=/projects/%u/spec-decoding-capstone/logs/%j.log
#SBATCH --job-name=specdec_framework
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --mail-type=END,FAIL

SLURM_TMPDIR="${SLURM_SCRATCH:-/tmp/$USER/scratch}"
export HF_HOME="/scratch/alpine/$USER/.cache/huggingface"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $HF_HOME

# 1. Define local scratch directories for active logging and checkpoints
export LOCAL_OUTPUT_DIR="$SLURM_TMPDIR/outputs"
export WANDB_DIR="$SLURM_TMPDIR/wandb"
mkdir -p "$LOCAL_OUTPUT_DIR" "$WANDB_DIR"

module load uv
uv sync

echo "=== CUDA + PyTorch diagnostics ==="
uv run python - <<'PY'
import torch, os
print("CUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch CUDA version:", torch.version.cuda)
print("Torch built with:", torch.__config__.show())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Detected GPUs:", torch.cuda.device_count())
    print("GPU 0:", torch.cuda.get_device_name(0))
PY

cd ..

uv run python run.py "$1" "${@:2}" --output_dir "$LOCAL_OUTPUT_DIR"

PROJECT_DIR="/projects/$USER/spec-decoding-capstone"
echo "Copying outputs back to $PROJECT_DIR..."
mkdir -p "$PROJECT_DIR/outputs"
cp -r "$LOCAL_OUTPUT_DIR"/* "$PROJECT_DIR/outputs/"

if [ -d "$WANDB_DIR" ]; then
    cp -r "$WANDB_DIR" "$PROJECT_DIR/"
fi

echo "Job execution and file synchronization complete."
