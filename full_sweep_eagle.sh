#!/usr/bin/env bash
for lang in que zgh zh amh; do
    echo "=================================================="
    echo "Submitting and waiting for language: $lang"
    echo "=================================================="

    sbatch scripts/run_alpine.sh experiments/eagle_greedy.cfg \
        -o language_code="${lang}" \
            output_dir="checkpoints/eagle_${lang}"

    echo "Finished $lang. Moving to next language..."
    sleep 15
done
