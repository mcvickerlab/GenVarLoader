#!/usr/bin/env bash
#SBATCH -p carter-compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH -J e2build
#SBATCH -o /carter/users/dlaub/repos/for_loukik/svar2_mvp/e2_build_%a.log
set -euo pipefail
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
W=/carter/users/dlaub/repos/for_loukik/svar2_mvp
SIZES=(1000 2000 4000 8000)
S=${SIZES[$SLURM_ARRAY_TASK_ID]}
echo "building S=$S host=$(hostname) cpus=$SLURM_CPUS_ON_NODE"
pixi run -e default python tmp/svar2_mvp/build_stores.py \
  "$W/gdc.chr21.s${S}.bcf" chr21 "$W/somatic_s${S}"
echo "E2_BUILD_DONE S=$S"
