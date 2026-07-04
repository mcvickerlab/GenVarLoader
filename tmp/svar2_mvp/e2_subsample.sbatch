#!/usr/bin/env bash
#SBATCH -p carter-compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -J e2_subsample
#SBATCH -o /carter/users/dlaub/repos/for_loukik/svar2_mvp/e2_subsample.log
set -euo pipefail
W=/carter/users/dlaub/repos/for_loukik/svar2_mvp
SRC=$W/gdc.chr21.norm.filt.bcf
bcftools query -l "$SRC" > "$W/somatic.samples.txt"
TOTAL=$(wc -l < "$W/somatic.samples.txt")
echo "total somatic samples=$TOTAL"
: > "$W/e2_variant_counts.txt"
for S in 1000 2000 4000 8000; do
  head -n "$S" "$W/somatic.samples.txt" > "$W/somatic.s${S}.list"
  bcftools view -S "$W/somatic.s${S}.list" --force-samples --threads 8 -Ob \
    -o "$W/gdc.chr21.s${S}.bcf" "$SRC"
  bcftools index -f "$W/gdc.chr21.s${S}.bcf"
  N=$(bcftools view -H "$W/gdc.chr21.s${S}.bcf" | wc -l)
  echo "S=$S variants=$N" | tee -a "$W/e2_variant_counts.txt"
done
N=$(bcftools view -H "$SRC" | wc -l)
echo "S=16007 variants=$N" | tee -a "$W/e2_variant_counts.txt"
echo "E2_SUBSAMPLE_DONE"
