"""DataLoader throughput bench entrypoint.

Compares to_dataloader modes (None / buffered / double_buffered) across
threads × region_length × batch_size × buffer_bytes for three output modes.

Run (writes experiments/dataloader/results.csv); the loader needs PyTorch,
which lives in the `default` pixi env:
    pixi run -e default python experiments/dataloader/bench.py

Thread counts are pinned per child process via re-exec, because rayon's pool
size is fixed once initialized.
"""

from __future__ import annotations

import datetime as _dt
import os
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results.csv"
TMP_DIR = HERE / "tmp"

REPO = HERE.parents[1]
SVAR = REPO / "tests" / "data" / "1kg" / "filtered.svar"
REGIONS_BED = REPO / "tests" / "data" / "1kg" / "regions.bed"
# haplotypes/annotated reconstruct against a reference; variants ignores it.
REF = REPO / "tests" / "data" / "fasta" / "hg38.fa.bgz"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_child(n_threads: int) -> None:
    """Child process: run only the cells pinned to this thread count."""
    git_sha = os.environ.get("BENCH_GIT_SHA", "")
    started_at = os.environ.get("BENCH_STARTED_AT", "")
    host = socket.gethostname()
    tmp_dir = Path(os.environ["BENCH_TMP"])

    # dataset paths were written by the parent, keyed by region length
    ds_paths = {
        length: tmp_dir / f"dataset_rL{length}.gvl"
        for length in C.REGION_LENGTHS
    }

    cells = C.cells_for_threads(n_threads)
    for i, cell in enumerate(cells, 1):
        ds_path = ds_paths[cell.region_length]
        try:
            row = C.measure_cell(
                cell, ds_path, REF,
                git_sha=git_sha, host=host, started_at=started_at,
            )
        except Exception as e:  # noqa: BLE001 - one bad cell must not kill the run
            print(f"[threads={n_threads}] cell {i}/{len(cells)} FAILED: {cell} -> {e}")
            continue
        C.append_row(RESULTS_CSV, row)
        print(
            f"[threads={n_threads}] {i}/{len(cells)} "
            f"{row['mode'] or 'None'}/{row['with_seqs']} "
            f"r{cell.region_length} b{cell.batch_size} "
            f"-> {row['instances_per_s']:.0f} inst/s"
            + (" TIMEOUT" if row["timed_out"] else "")
        )


def run_parent() -> None:
    print(f"Preparing datasets in {TMP_DIR} ...")
    C.prepare_datasets(C.REGION_LENGTHS, SVAR, REGIONS_BED, TMP_DIR)
    C.init_csv(RESULTS_CSV)

    started_at = _dt.datetime.now().isoformat(timespec="seconds")
    git_sha = _git_sha()

    for n_threads in C.ALL_THREADS:
        env = {
            **os.environ,
            "BENCH_THREADS": str(n_threads),
            "BENCH_TMP": str(TMP_DIR),
            "BENCH_GIT_SHA": git_sha,
            "BENCH_STARTED_AT": started_at,
            "RAYON_NUM_THREADS": str(n_threads),
            "POLARS_MAX_THREADS": str(n_threads),
            "OMP_NUM_THREADS": str(n_threads),
            "MKL_NUM_THREADS": str(n_threads),
            "OPENBLAS_NUM_THREADS": str(n_threads),
        }
        print(f"\n=== child: threads={n_threads} ===")
        subprocess.run([sys.executable, __file__, "--child"], env=env, check=True)

    print(f"\nDone. Results: {RESULTS_CSV}")


def main() -> None:
    if "--child" in sys.argv:
        run_child(int(os.environ["BENCH_THREADS"]))
    else:
        try:
            run_parent()
        finally:
            import shutil

            shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
