from pathlib import Path

import numpy as np
import polars as pl
import seqpro as sp
from einops import repeat
from natsort import natsorted

from ._dataset._genotypes import SparseGenotypes
from ._dataset._impl import RaggedDataset
from ._dataset._indexing import DatasetIndexer
from ._dataset._intervals import tracks_to_intervals
from ._dataset._reconstruct import Haps, HapsTracks, Reference, Tracks, _Variants
from ._dataset._utils import bed_to_regions
from ._ragged import Ragged, RaggedIntervals
from ._utils import _lengths_to_offsets
from ._variants._records import VLenAlleles


def get_dummy_dataset():
    """Return a dummy :class:`Dataset <genvarloader.Dataset>`  with 4 regions, 4 samples, max jitter of 2, a reference genome of all :code:`"N"`, genotypes, and
    1 track "read-depth" where each track is :code:`[1, 2, 3, 4, 5, 6]` in the reference coordinate system, where :code:`3` is aligned
    with each region's start coordinate. Is initialized to return ragged haplotypes and tracks with no jitter and deterministic reconstruction algorithms.
    """
    max_jitter = 2

    dummy_samples = ["Aang", "Katara", "Sokka", "Toph"]

    dummy_contigs = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
    dummy_bed = pl.DataFrame(
        {
            "chrom": ["8", "3", "6", "2"],
            "chromStart": [5, 13, 8, 2],
            "chromEnd": [8, 16, 11, 5],
            "strand": ["+", "-", "+", "+"],
        }
    )

    with pl.StringCache():
        pl.Series(natsorted(dummy_contigs), dtype=pl.Categorical())
        sorted_bed = dummy_bed.with_row_index().sort(
            pl.col("chrom").cast(pl.Categorical())
        )

    r_idx_map = np.argsort(sorted_bed["index"])
    sorted_bed = sorted_bed.drop("index")
    dummy_idxer = DatasetIndexer.from_region_and_sample_idxs(
        r_idx_map, np.arange(len(dummy_samples)), dummy_samples
    )

    dummy_regions = bed_to_regions(sorted_bed, dummy_contigs)

    ref_len = 20
    ref_lens = np.full(len(dummy_contigs), ref_len, dtype=np.int32)
    ref = np.full(ref_len * len(dummy_contigs), b"N", dtype="S1").view(np.uint8)
    dummy_ref = Reference(
        reference=ref,
        contigs=dummy_contigs,
        offsets=_lengths_to_offsets(ref_lens, np.uint64),
        pad_char=ord(b"N"),
    )

    dummy_vars = _Variants(
        positions=repeat(dummy_regions[:, 1], "r -> (r s)", s=4),
        sizes=repeat(np.array([-2, -1, 0, 1], np.int32), "s -> (r s)", r=4),
        alts=VLenAlleles(
            alleles=repeat(sp.cast_seqs("ACGTT"), "a -> (r a)", r=4),
            offsets=_lengths_to_offsets(
                repeat(np.array([1, 1, 1, 2]), "s -> (r s)", r=4)
            ),
        ),
    )

    v_idxs = (
        np.array([[3, 2, 4, 1], [1, 3, 2, 4], [2, 1, 4, 3], [4, 2, 3, 1]])[
            [3, 1, 2, 0]
        ]  # target lengths
        - 1  # idx within region
        + 4 * np.arange(4)[:, None]  # adjust by region/contig offset
    )
    dummy_genos = SparseGenotypes(
        variant_idxs=v_idxs.ravel(),
        offsets=np.arange(0, 4 * 4 + 1, dtype=np.int64),  # every entry has 1 variant
        n_regions=4,
        n_samples=4,
        ploidy=1,
    )

    dummy_haps = Haps(dummy_ref, dummy_vars, dummy_genos, False)

    # (r s), want tracks of [1, 2, 3, 4, 5] for each region so that pad values of 0 are obvious
    track_regions = dummy_regions.copy()
    track_regions[:, 1] -= max_jitter
    track_regions[:, 2] = track_regions[:, 1] + 5 + max_jitter
    t_len = 5
    data, offsets = tracks_to_intervals(
        regions=track_regions,
        tracks=repeat(
            np.arange(1, 1 + t_len + 2 * max_jitter, dtype=np.float32),
            "l -> (r l)",
            r=4,
        ),
        track_offsets=_lengths_to_offsets(np.full(4, t_len + 2 * max_jitter)),
    )
    lengths = np.diff(offsets)
    data = repeat(data, "(r i) -> (r s i)", r=4, s=4)
    offsets = _lengths_to_offsets(repeat(lengths, "r -> (r s)", s=4))
    dummy_itvs = {
        "read-depth": RaggedIntervals.from_offsets(
            data=data, shape=(4, 4), offsets=offsets
        )
    }

    dummy_tracks = Tracks(dummy_itvs, ["read-depth"])

    dummy_recon = HapsTracks(dummy_haps, dummy_tracks)

    dummy_dataset: RaggedDataset[Ragged[np.bytes_], Ragged[np.float32], None, None] = (
        RaggedDataset(
            path=Path("dummy"),
            output_length="ragged",
            max_jitter=max_jitter,
            return_indices=False,
            contigs=dummy_contigs,
            jitter=0,
            deterministic=True,
            rc_neg=True,
            transform=None,
            _full_bed=dummy_bed,
            _full_regions=dummy_regions,
            _jittered_regions=dummy_regions.copy(),
            _idxer=dummy_idxer,
            _seqs=dummy_haps,
            _tracks=dummy_tracks,
            _recon=dummy_recon,
            _rng=np.random.default_rng(),
        )
    )

    return dummy_dataset
