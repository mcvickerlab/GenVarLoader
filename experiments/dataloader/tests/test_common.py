"""Pure-logic tests for the dataloader bench. Run explicitly:

    pixi run -e dev pytest experiments/dataloader/tests/test_common.py -v

These are NOT collected by the default `pixi run -e dev test` task.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _common as C


def test_axis_constants_match_spec():
    assert C.OUTPUTS == ["haplotypes", "annotated", "variants"]
    assert C.MODES_NEW == ["buffered", "double_buffered"]
    assert C.ALL_MODES == [None, "buffered", "double_buffered"]

    assert C.THREADS_FACT == [1, 8]
    assert C.REGION_FACT == [1_000, 10_000]
    assert C.BATCH_FACT == [16, 128]
    assert C.BUFFER_FACT == [256 * C.MiB, 2 * C.GiB]

    assert C.THREADS_FAN == [2, 4, 16]
    assert C.REGION_FAN == [2_500, 5_000, 25_000]
    assert C.BATCH_FAN == [32, 64, 256]
    assert C.BUFFER_FAN == [512 * C.MiB, 1 * C.GiB, 4 * C.GiB]

    assert C.THREADS_MID == 4
    assert C.REGION_MID == 5_000
    assert C.BATCH_MID == 64
    assert C.BUFFER_MID == 1 * C.GiB


def test_dispatch_unions():
    assert C.ALL_THREADS == [1, 2, 4, 8, 16]
    assert C.REGION_LENGTHS == [1_000, 2_500, 5_000, 10_000, 25_000]
    # midpoints are members of their own fans
    assert C.THREADS_MID in C.THREADS_FAN
    assert C.REGION_MID in C.REGION_FAN
    assert C.BATCH_MID in C.BATCH_FAN
    assert C.BUFFER_MID in C.BUFFER_FAN


def test_new_mode_cell_count_is_25_per_mode_output():
    for mode in C.MODES_NEW:
        for output in C.OUTPUTS:
            cells = [
                c for c in C.enumerate_cells()
                if c.mode == mode and c.with_seqs == output
            ]
            # 16 factorial + 9 fan (12 raw − 3 shared midpoints) = 25
            assert len(cells) == 25, (mode, output, len(cells))


def test_baseline_cell_count_is_15_per_output_and_has_no_buffer():
    for output in C.OUTPUTS:
        cells = [
            c for c in C.enumerate_cells()
            if c.mode is None and c.with_seqs == output
        ]
        # 8 factorial corners + 7 fan (9 raw − 2 shared midpoints) = 15
        assert len(cells) == 15, (output, len(cells))
        assert all(c.buffer_bytes is None for c in cells)


def test_total_cell_count_is_195_and_all_unique():
    cells = C.enumerate_cells()
    assert len(cells) == 195
    keys = {
        (c.mode, c.with_seqs, c.threads, c.region_length, c.batch_size, c.buffer_bytes)
        for c in cells
    }
    assert len(keys) == 195  # no duplicate configurations


def test_baseline_fan_cells_sit_at_midpoints():
    # the threads fan for baseline pins region=MID, batch=MID
    base = [c for c in C.enumerate_cells() if c.mode is None and c.with_seqs == "variants"]
    threads_fan = [
        c for c in base
        if c.region_length == C.REGION_MID and c.batch_size == C.BATCH_MID
    ]
    assert sorted(c.threads for c in threads_fan) == [2, 4, 16]


def test_new_mode_buffer_fan_pins_other_axes_at_midpoint():
    buf_fan = [
        c for c in C.enumerate_cells()
        if c.mode == "buffered" and c.with_seqs == "haplotypes"
        and c.threads == C.THREADS_MID
        and c.region_length == C.REGION_MID
        and c.batch_size == C.BATCH_MID
    ]
    assert sorted(c.buffer_bytes for c in buf_fan) == sorted(C.BUFFER_FAN)


def test_cells_for_threads_partitions_by_thread_count():
    all_cells = C.enumerate_cells()
    union = []
    for n in C.ALL_THREADS:
        sub = C.cells_for_threads(n)
        assert all(c.threads == n for c in sub)
        union.extend(sub)
    assert len(union) == len(all_cells)


import pytest


@pytest.mark.slow
def test_prepare_datasets_writes_one_gvl_per_region_length(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    svar = repo / "tests" / "data" / "1kg" / "filtered.svar"
    regions = repo / "tests" / "data" / "1kg" / "regions.bed"
    if not svar.is_dir():
        pytest.skip("missing tests/data/1kg/filtered.svar; run pixi run -e dev gen")

    paths = C.prepare_datasets([1_000, 2_500], svar, regions, tmp_path)

    assert set(paths) == {1_000, 2_500}
    for length, p in paths.items():
        assert p.is_dir(), p
        assert (p / "metadata.json").exists()


@pytest.mark.slow
def test_generate_bed_resizes_to_target_length():
    import seqpro as sp

    repo = Path(__file__).resolve().parents[3]
    regions = repo / "tests" / "data" / "1kg" / "regions.bed"
    if not regions.exists():
        pytest.skip("missing regions.bed")

    bed = C.generate_bed(regions, 2_500)
    lengths = (bed["chromEnd"] - bed["chromStart"]).unique().to_list()
    assert lengths == [2_500]
    assert bed.height == 100  # regions.bed has 100 regions


@pytest.mark.slow
def test_output_bytes_table_matches_actual_nbytes(tmp_path):
    import genvarloader as gvl

    repo = Path(__file__).resolve().parents[3]
    svar = repo / "tests" / "data" / "1kg" / "filtered.svar"
    regions = repo / "tests" / "data" / "1kg" / "regions.bed"
    ref = repo / "tests" / "data" / "fasta" / "hg38.fa.bgz"
    if not svar.is_dir() or not ref.exists():
        pytest.skip("missing filtered.svar or hg38 reference; run pixi run -e dev gen")

    paths = C.prepare_datasets([1_000], svar, regions, tmp_path)
    ds = gvl.Dataset.open(paths[1_000], reference=ref).with_seqs("variants")

    instances, total_bytes, table = C.output_bytes_table(ds)
    assert instances == table.size
    assert total_bytes == int(table.sum())
    assert instances == 100 * 5  # 100 regions × 5 samples


def test_csv_init_then_append_roundtrips(tmp_path):
    import csv

    csv_path = tmp_path / "results.csv"
    C.init_csv(csv_path)

    with csv_path.open() as f:
        header = next(csv.reader(f))
    assert header == C.CSV_COLUMNS

    row = {col: 0 for col in C.CSV_COLUMNS}
    row["mode"] = "buffered"
    row["with_seqs"] = "variants"
    C.append_row(csv_path, row)
    C.append_row(csv_path, row)

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["mode"] == "buffered"
    assert rows[0]["with_seqs"] == "variants"


def test_append_row_rejects_unknown_columns(tmp_path):
    csv_path = tmp_path / "results.csv"
    C.init_csv(csv_path)
    with pytest.raises((ValueError, KeyError)):
        C.append_row(csv_path, {"bogus": 1})


@pytest.mark.slow
def test_measure_cell_returns_a_complete_row(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    svar = repo / "tests" / "data" / "1kg" / "filtered.svar"
    regions = repo / "tests" / "data" / "1kg" / "regions.bed"
    ref = repo / "tests" / "data" / "fasta" / "hg38.fa.bgz"
    if not svar.is_dir() or not ref.exists():
        pytest.skip("missing filtered.svar or hg38 reference; run pixi run -e dev gen")

    paths = C.prepare_datasets([1_000], svar, regions, tmp_path)
    cell = C.Cell(
        mode=None, with_seqs="variants",
        threads=1, region_length=1_000, batch_size=16, buffer_bytes=None,
    )
    # tiny stop conditions so the test is fast
    row = C.measure_cell(
        cell, paths[1_000], ref, min_epochs=1, min_seconds=0.0, hard_cap_s=10.0,
    )

    for col in C.CSV_COLUMNS:
        assert col in row, col
    assert row["mode"] == ""          # None serialized as empty
    assert row["with_seqs"] == "variants"
    assert row["n_epochs"] >= 1
    assert row["instances"] == 100 * 5 * row["n_epochs"]
    assert row["instances_per_s"] > 0
    assert row["timed_out"] in (True, False)
