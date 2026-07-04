#!/usr/bin/env bash
#SBATCH -p carter-compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -J genoray_dbg
#SBATCH -o /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/tmp/svar2_mvp/genoray_debug_build.log
set -euo pipefail
cd /carter/users/dlaub/projects/genoray
CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
RUSTFLAGS="-C force-frame-pointers=yes" \
pixi run -e py310 maturin build --release
echo "WHEEL_BUILD_DONE"
ls -t target/wheels/genoray-2.15.0-*.whl | head -1
