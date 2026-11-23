#!/usr/bin/env bash

NUM_LAYERS="${NUM_LAYERS:-6}"
TREE_CHOICES="${TREE_CHOICES:-wide_shallow}"

case "$TREE_CHOICES" in
  wide_shallow)   GAMMA=4 ;;
  wide_shallow_b) GAMMA=3 ;;
  conservative)   GAMMA=4 ;;
  custom)         GAMMA=4 ;;
  *)              GAMMA="${GAMMA:-4}" ;;
esac
for lang in ber chr haw ibo lkt mus npi oci oji que zgh zh amh yor grn aed deu spa hin; do
    OUT_DIR="checkpoints/eagle_${lang}_l${NUM_LAYERS}_${TREE_CHOICES}_g${GAMMA}"
    echo "=================================================="
    echo "Submitting language: $lang | num_layers=$NUM_LAYERS | tree=$TREE_CHOICES | gamma=$GAMMA"
    echo "Output dir: $OUT_DIR"
    echo "=================================================="

    sbatch scripts/run_alpine.sh experiments/eagle_greedy.cfg \
        -o language_code="${lang}" \
            num_layers="${NUM_LAYERS}" \
            tree_choices="${TREE_CHOICES}" \
            gamma="${GAMMA}" \
            output_dir="${OUT_DIR}"

        echo "Submitted $lang. Moving to next language..."
        sleep 15
    sleep 15
done
