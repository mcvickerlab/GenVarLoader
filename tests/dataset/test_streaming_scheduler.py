import numpy as np
import polars as pl
from genvarloader._dataset._streaming import StreamingDataset


def _bed(rows):
    return pl.DataFrame(
        rows, schema={"chrom": pl.Utf8, "chromStart": pl.Int64, "chromEnd": pl.Int64}
    )


def test_plan_is_region_major_and_covers_grid():
    # 3 regions x 2 samples, batch_size 2 -> region-major flat order
    bed = _bed(
        [{"chrom": "chr1", "chromStart": s, "chromEnd": s + 10} for s in (30, 10, 20)]
    )
    seen = []

    def stub(r_idx, s_idx):
        # Cartesian window callback: r_idx and s_idx are independent index sets
        # (not parallel arrays of the same length), C-order (region, sample).
        seen.append((tuple(r_idx), tuple(s_idx)))
        rr, ss = np.meshgrid(r_idx, s_idx, indexing="ij")
        return np.stack([rr.ravel(), ss.ravel()], axis=1)  # fake "data"

    sds = StreamingDataset(
        bed, contigs=["chr1"], n_samples=2, ploidy=2, _reconstruct_window=stub
    )
    batches = list(sds.to_iter(batch_size=2))
    # region order is sorted by (contig,start): input starts 30,10,20 -> sorted r order 1,2,0
    # region-major flat index over (n_regions=3, n_samples=2): r sorted-inner sample
    flat_r = np.concatenate([b[1] for b in batches])
    flat_s = np.concatenate([b[2] for b in batches])
    # every (r,s) cell appears exactly once
    cells = set(zip(flat_r.tolist(), flat_s.tolist()))
    assert cells == {(r, s) for r in range(3) for s in range(2)}
    # region-major: sample index varies fastest within a region
    assert flat_r.tolist() == [1, 1, 2, 2, 0, 0]
    assert flat_s.tolist() == [0, 1, 0, 1, 0, 1]
    assert len(sds) == 6
    assert sds.shape == (3, 2)


def test_sort_order_is_duplicate_safe():
    # two identical rows (same chrom/chromStart/chromEnd) among distinct ones.
    # A value-based join on all BED columns fans out on the duplicate pair,
    # corrupting `_sort_order`'s length/contents; a positional row-index carry
    # through the sort must not.
    n_samples = 2
    rows = [
        {"chrom": "chr1", "chromStart": 30, "chromEnd": 40},
        {"chrom": "chr1", "chromStart": 10, "chromEnd": 20},
        {"chrom": "chr1", "chromStart": 10, "chromEnd": 20},  # duplicate of row 1
        {"chrom": "chr1", "chromStart": 20, "chromEnd": 30},
    ]
    bed = _bed(rows)
    n_regions = len(rows)

    def stub(r_idx, s_idx):
        rr, ss = np.meshgrid(r_idx, s_idx, indexing="ij")
        return np.stack([rr.ravel(), ss.ravel()], axis=1)

    sds = StreamingDataset(
        bed, contigs=["chr1"], n_samples=n_samples, ploidy=2, _reconstruct_window=stub
    )

    assert len(sds._sort_order) == n_regions
    assert set(sds._sort_order.tolist()) == set(range(n_regions))

    assert len(sds) == n_regions * n_samples

    batches = list(sds.to_iter())
    flat_r = np.concatenate([b[1] for b in batches])
    flat_s = np.concatenate([b[2] for b in batches])
    cells = set(zip(flat_r.tolist(), flat_s.tolist()))
    assert cells == {(r, s) for r in range(n_regions) for s in range(n_samples)}


def _injected_sds(n_regions, n_samples, ploidy=2, max_mem="512MB", contigs=("chr1",)):
    """Build a StreamingDataset via the injected-callback path (no store) with all
    regions on one contig, so we can inspect the window plan in isolation."""
    bed = pl.DataFrame(
        {
            "chrom": [contigs[0]] * n_regions,
            "chromStart": list(range(0, 100 * n_regions, 100)),
            "chromEnd": list(range(100, 100 * n_regions + 100, 100)),
        }
    )
    return StreamingDataset(
        bed,
        contigs=list(contigs),
        n_samples=n_samples,
        ploidy=ploidy,
        max_mem=max_mem,
        _reconstruct_window=lambda r, s: None,
    )


def test_plan_chunks_samples_under_a_tight_budget():
    # ploidy=2 -> cell_bytes=32. max_mem tiny (2 KB) -> max_cells small, so a
    # 1000-sample cohort must be split into multiple sample chunks per region window.
    sds = _injected_sds(n_regions=4, n_samples=1000, max_mem=2048)
    windows = list(sds._plan())
    sample_chunk_sizes = {len(s) for _, s in windows}
    assert max(sample_chunk_sizes) < 1000, (
        "samples must be chunked under a tight budget"
    )
    # Every (region, sample) cell appears exactly once across the plan.
    seen = set()
    for r_idx, s_idx in windows:
        for r in r_idx.tolist():
            for s in s_idx.tolist():
                assert (r, s) not in seen
                seen.add((r, s))
    assert len(seen) == 4 * 1000


def test_plan_keeps_all_samples_when_they_fit():
    # Generous budget -> a small cohort stays as one sample chunk per window.
    sds = _injected_sds(n_regions=4, n_samples=8, max_mem="512MB")
    for _, s_idx in sds._plan():
        assert len(s_idx) == 8, "all samples should fit in one chunk under a big budget"


def test_plan_is_single_contig_per_window():
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "chromStart": [0, 100, 0],
            "chromEnd": [100, 200, 100],
        }
    )
    sds = StreamingDataset(
        bed,
        contigs=["chr1", "chr2"],
        n_samples=4,
        ploidy=2,
        _reconstruct_window=lambda r, s: None,
    )
    for r_idx, _ in sds._plan():
        contigs = sds._regions[r_idx, 0]
        assert len(set(contigs.tolist())) == 1
