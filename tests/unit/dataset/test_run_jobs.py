"""Unit tests for _run_jobs (joblib loky parallelism helper)."""

from pathlib import Path

from genvarloader._dataset._write import _run_jobs


# Module-level callable so loky can pickle it.
def _record(mm: int, path: Path) -> None:
    """Write the received per-job max_mem budget to *path*."""
    path.write_text(str(mm))


def test_run_jobs_two_jobs_loky_path(tmp_path: Path) -> None:
    """2 jobs → loky backend; each receives max_mem // 2."""
    file0 = tmp_path / "job0.txt"
    file1 = tmp_path / "job1.txt"

    from functools import partial

    job0 = partial(_record, path=file0)
    job1 = partial(_record, path=file1)

    _run_jobs([job0, job1], max_mem=1000)

    assert file0.exists(), "job0 did not write its file"
    assert file1.exists(), "job1 did not write its file"
    assert int(file0.read_text()) == 500, (
        f"job0 got {file0.read_text()!r}, expected 500"
    )
    assert int(file1.read_text()) == 500, (
        f"job1 got {file1.read_text()!r}, expected 500"
    )


def test_run_jobs_one_job_inline(tmp_path: Path) -> None:
    """1 job → runs inline (no loky spawn); receives the full max_mem."""
    file0 = tmp_path / "job0.txt"

    from functools import partial

    job0 = partial(_record, path=file0)

    _run_jobs([job0], max_mem=1000)

    assert file0.exists(), "job0 did not write its file"
    assert int(file0.read_text()) == 1000, f"expected 1000, got {file0.read_text()!r}"


def test_run_jobs_zero_jobs(tmp_path: Path) -> None:
    """0 jobs → no-op, no errors."""
    _run_jobs([], max_mem=1000)  # must not raise


def test_run_jobs_none_jobs_filtered(tmp_path: Path) -> None:
    """None entries are filtered out before dispatch."""
    file0 = tmp_path / "job0.txt"

    from functools import partial

    job0 = partial(_record, path=file0)

    # Pass a list with None — should be treated as 1 real job (inline path).
    _run_jobs([None, job0], max_mem=1000)  # type: ignore[list-item]

    assert file0.exists()
    assert int(file0.read_text()) == 1000
