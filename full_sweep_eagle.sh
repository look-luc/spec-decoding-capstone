#!/usr/bin/env bash
set -euo pipefail

NUM_LAYERS="${NUM_LAYERS:-6}"
TREE_CHOICES="${TREE_CHOICES:-wide_shallow}"
BETA="${BETA:-0.3}"
CFG_FILE="${CFG_FILE:-experiments/eagle_greedy.cfg}"

case "$TREE_CHOICES" in
  wide_shallow)   GAMMA=4 ;;
  wide_shallow_b) GAMMA=3 ;;
  conservative)   GAMMA=4 ;;
  custom)         GAMMA=4 ;;
  *)              GAMMA="${GAMMA:-4}" ;;
esac

for lang in oci que zgh zh amh yor grn aed deu spa hin; do
    OUT_DIR="checkpoints/eagle_${lang}_l${NUM_LAYERS}_${TREE_CHOICES}_g${GAMMA}"
    echo "=================================================="
    echo "Submitting language: $lang | num_layers=$NUM_LAYERS | tree=$TREE_CHOICES | gamma=$GAMMA | beta=$BETA"
    echo "Config: $CFG_FILE | Output dir: $OUT_DIR"
    echo "=================================================="

    sbatch scripts/run_alpine.sh "$CFG_FILE" \
        -o language_code="${lang}" \
           num_layers="${NUM_LAYERS}" \
           tree_choices="${TREE_CHOICES}" \
           gamma="${GAMMA}" \
           distillation_beta="${BETA}" \
           output_dir="${OUT_DIR}"

    echo "Submitted $lang. Moving to next language..."
    sleep 15
done
