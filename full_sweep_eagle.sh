#!/usr/bin/env bash

for lang in lkt mus npi oci oji que zgh amh yor grn aed; do
  echo "=================================================="
  echo "Submitting and waiting for language: $lang"
  echo "=================================================="

  # --wait blocks the loop until all 16 tasks of this language finish and clear the queue
  sbatch --wait --array=0-15%16 --export=LANG_CODE=$lang scripts/run_params_sweep_alpine_eagle.sh experiments/eagle_greedy.cfg

  echo "Finished $lang. Moving to next language..."
  sleep 5
done
