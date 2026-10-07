#!/bin/bash
#SBATCH --partition=al40
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:10:00
#SBATCH --output=/projects/%u/spec-decoding-capstone/logs/%j.log
#SBATCH --job-name=specdec_l40
#SBATCH --account=ucb-general
#SBATCH --mail-type=END,FAIL

export HF_HOME="/scratch/alpine/$USER/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $HF_HOME

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

uv run python run.py "$1" "${@:2}"
