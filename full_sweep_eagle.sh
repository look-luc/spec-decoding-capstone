#!/usr/bin/env bash

for lang in chr haw ibo lkt mus npi oci oji que zgh zh amh yor grn aed eng deu spa hin; do
  echo "=================================================="
  echo "Submitting and waiting for language: $lang"
  echo "=================================================="

  # --wait blocks the loop until all 16 tasks of this language finish and clear the queue
  sbatch --wait --array=0-15%16 --export=LANG_CODE=$lang scripts/run_params_sweep_alpine_eagle.sh

  echo "Finished $lang. Moving to next language..."
  sleep 5
done
