#!/usr/bin/env bash
for lang in chr haw ibo lkt mus npi oci oji que zgh zh amh yor grn aed eng deu spa hin; do
    echo "=================================================="
    echo "Submitting and waiting for language: $lang"
    echo "=================================================="
    LABEL="lr${LR}_s${STEPS}_ga${GA}_g${GAMMA}"

    OUTPUT_DIR="checkpoints/eagle_${LANG_CODE}_${LABEL}"
    # --wait blocks the loop until all 16 tasks of this language finish and clear the queue
    sbatch --wait --export=LANG_CODE=$lang scripts/run_alpine.sh experiments/eagle_greedy.cfg -o output_dir="${OUTPUT_DIR}"
    echo "Finished $lang. Moving to next language..."
    wait 5
done
