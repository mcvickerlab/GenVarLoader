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
        seen.append((tuple(r_idx), tuple(s_idx)))
        return np.stack([r_idx, s_idx], axis=1)  # fake "data"

    sds = StreamingDataset(
        bed, contigs=["chr1"], n_samples=2, ploidy=2, _reconstruct_window=stub
    )
    sds = sds._with_batch_size(2)
    batches = list(sds)
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
