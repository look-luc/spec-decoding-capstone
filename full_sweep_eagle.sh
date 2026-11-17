#!/usr/bin/env bash

for lang in chr haw ibo lkt mus npi oci oji que zgh zh amh yor grn aed eng deu spa hin; do
  sbatch --array=0-323 --export=LANG_CODE=$lang scripts/run_params_sweep_alpine_eagle.sh
  sleep 2
done
