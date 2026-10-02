#!/bin/bash
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/projects/%u/spec-decoding-capstone/logs/%j.log
#SBATCH --job-name=specdec_framework
#SBATCH --partition=aa100
#SBATCH --account=ucb-general
#SBATCH --qos=gpu-testing
#SBATCH --mail-type=END,FAIL

export HF_HOME="/scratch/alpine/$USER/.cache/huggingface"
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
