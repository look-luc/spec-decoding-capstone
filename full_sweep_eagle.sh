#!/usr/bin/env bash

for lang in chr haw ibo lkt mus npi oci oji que zgh zh amh yor grn aed eng deu spa hin; do
  echo "Submitting sweep for language: $lang"
  sbatch --array=0-15%16 --export=LANG_CODE=$lang scripts/run_params_sweep_alpine_eagle.sh
  # Optional: sleep 2 seconds between language batches to be gentle on the scheduler
  sleep 2
done
