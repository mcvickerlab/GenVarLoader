#!/usr/bin/env bash
# Generate benchmark-scale VCF/BCF (via `vcfixture bulk`) and PGEN (via
# `plink2 --make-pgen`) fixtures for the streaming VCF/PGEN backend
# cohort-size sweep (benchmarking/streaming/bench_streaming.py, issue #276
# Task 13). Wrapped by the `gen-bench-vcf`/`gen-bench-pgen` pixi tasks; can
# also be run directly.
#
# PREREQUISITE: `vcfixture` CLI on PATH, built with its `cli` feature:
#     cargo install vcfixture --features cli
#   (the `bulk` subcommand lives behind that feature -- a plain
#   `cargo install vcfixture` will NOT provide it). If `vcfixture` isn't on
#   PATH, set VCFIXTURE_RS_DIR to a local `vcfixture-rs` source checkout and
#   this script falls back to `cargo run --release --features cli` against
#   it (works with no install step, but re-pays cargo's up-to-date check
#   every invocation).
#
# Generated fixtures (*.bcf, *.csi, *.summary.json, *.pgen, *.pvar, *.psam,
# *.log) are LARGE and are gitignored (benchmarking/streaming/.gitignore) --
# never commit them.
#
# Usage:
#   N=1000 SIZE=50MB benchmarking/streaming/gen_fixtures.sh vcf
#   N=1000 SIZE=50MB benchmarking/streaming/gen_fixtures.sh pgen
#
# Env vars (all optional; defaults produce a small smoke-test fixture):
#   N        number of samples (default 1000)
#   SIZE     vcfixture --target-size, e.g. "50MB" (default 50MB)
#   CONTIGS  comma-separated contig list (default chr1 -- vcfixture's
#            built-in profiles default to chr1,chr2,chr3; narrow to chr1
#            for faster small-fixture generation)
#   SEED     PRNG seed, for reproducible fixtures (default 42)
#   PROFILE  vcfixture --profile (default germline-1kgp)
#   OUT_DIR  output directory (default benchmarking/streaming)
set -euo pipefail

KIND="${1:?usage: gen_fixtures.sh <vcf|pgen>}"
N="${N:-1000}"
SIZE="${SIZE:-50MB}"
CONTIGS="${CONTIGS:-chr1}"
SEED="${SEED:-42}"
PROFILE="${PROFILE:-germline-1kgp}"
OUT_DIR="${OUT_DIR:-benchmarking/streaming}"
mkdir -p "$OUT_DIR"

BCF_PATH="$OUT_DIR/bench_${N}.bcf"
PGEN_PREFIX="$OUT_DIR/bench_${N}"

resolve_vcfixture() {
  if command -v vcfixture >/dev/null 2>&1; then
    echo "vcfixture"
    return
  fi
  if [ -n "${VCFIXTURE_RS_DIR:-}" ] && [ -d "${VCFIXTURE_RS_DIR:-}" ]; then
    echo "cargo run --manifest-path ${VCFIXTURE_RS_DIR}/Cargo.toml --features cli --release --bin vcfixture --"
    return
  fi
  echo "ERROR: 'vcfixture' not found on PATH and VCFIXTURE_RS_DIR is unset." >&2
  echo "  Install:            cargo install vcfixture --features cli" >&2
  echo "  Or run from source: VCFIXTURE_RS_DIR=/path/to/vcfixture-rs $0 $KIND" >&2
  exit 1
}

case "$KIND" in
  vcf)
    VCFIXTURE_CMD="$(resolve_vcfixture)"
    echo "generating $BCF_PATH (N=$N SIZE=$SIZE CONTIGS=$CONTIGS SEED=$SEED PROFILE=$PROFILE)" >&2
    # shellcheck disable=SC2086
    $VCFIXTURE_CMD bulk \
      --profile "$PROFILE" \
      --samples "$N" \
      --contigs "$CONTIGS" \
      --target-size "$SIZE" \
      --seed "$SEED" \
      -o "$BCF_PATH"
    ;;
  pgen)
    if [ ! -f "$BCF_PATH" ]; then
      echo "ERROR: $BCF_PATH does not exist; generate it first with the same N" >&2
      echo "  (e.g. 'N=$N SIZE=$SIZE $0 vcf', or the gen-bench-vcf pixi task)." >&2
      exit 1
    fi
    echo "converting $BCF_PATH -> ${PGEN_PREFIX}.{pgen,pvar,psam}" >&2
    # --allow-extra-chr --output-chr chrM: keeps "chr1"-style contig names
    # instead of plink2's default un-prefixed human-contig coding (same
    # convention tests/dataset/conftest.py's PGEN fixtures use) -- required
    # so the resulting .pvar's contig names match the reference FASTA /
    # StreamingDataset's `contigs`.
    plink2 --bcf "$BCF_PATH" --make-pgen --allow-extra-chr --output-chr chrM \
      --out "$PGEN_PREFIX"
    ;;
  *)
    echo "usage: gen_fixtures.sh <vcf|pgen>" >&2
    exit 1
    ;;
esac
