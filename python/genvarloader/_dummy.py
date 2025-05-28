from pathlib import Path

import numpy as np
import polars as pl
import seqpro as sp
from einops import repeat
from genoray._svar import POS_TYPE, SparseGenotypes
from natsort import natsorted

from ._dataset._impl import RaggedDataset
from ._dataset._indexing import DatasetIndexer
from ._dataset._intervals import tracks_to_intervals
from ._dataset._reconstruct import Haps, HapsTracks, Tracks, TrackType, _Variants
from ._dataset._reference import Reference
from ._dataset._utils import bed_to_regions
from ._ragged import Ragged, RaggedIntervals, RaggedSeqs
from ._utils import lengths_to_offsets
from ._variants._records import RaggedAlleles


def get_dummy_dataset():
    """Return a dummy :class:`Dataset <genvarloader.Dataset>`  with 4 regions, 4 samples, max jitter of 2, a reference genome of all :code:`"N"`, genotypes, and
    1 track "read-depth" where each track is :code:`[1, 2, 3, 4, 5, 6]` in the reference coordinate system, where :code:`3` is aligned
    with each region's start coordinate. Is initialized to return ragged haplotypes and tracks with no jitter and deterministic reconstruction algorithms.
    """
    max_jitter = 2

    dummy_samples = ["Aang", "Katara", "Sokka", "Toph"]
    n_samples = len(dummy_samples)

    dummy_contigs = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
    dummy_bed = pl.DataFrame(
        {
            "chrom": ["8", "3", "6", "2"],
            "chromStart": [5, 13, 8, 2],
            "chromEnd": [8, 16, 11, 5],
            "strand": ["+", "-", "+", "+"],
        }
    )
    n_regions = len(dummy_bed)

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
        path=Path("dummy"),
        reference=ref,
        contigs=dummy_contigs,
        offsets=lengths_to_offsets(ref_lens, np.int64),
        pad_char=ord(b"N"),
    )

    dummy_vars = _Variants(
        v_starts=repeat(
            dummy_regions[:, 1].astype(POS_TYPE), "r -> (r s)", s=n_samples
        ),
        ilens=repeat(np.array([-2, -1, 0, 1], np.int32), "s -> (r s)", r=n_regions),
        alts=RaggedAlleles.from_offsets(
            data=repeat(sp.cast_seqs("ACGTT"), "a -> (r a)", r=n_regions),
            shape=n_regions * n_samples,
            offsets=lengths_to_offsets(
                repeat(np.array([1, 1, 1, 2]), "s -> (r s)", r=n_regions)
            ),
        ),
    )

    v_idxs = (
        np.array([[3, 2, 4, 1], [1, 3, 2, 4], [2, 1, 4, 3], [4, 2, 3, 1]])[
            [3, 1, 2, 0]
        ]  # target lengths
        - 1  # 0-based idx within region
        + 4 * np.arange(4)[:, None]  # adjust by region/contig offset
    ).astype(np.int32)
    shape = (4, 4, 1)
    dummy_genos = SparseGenotypes.from_offsets(
        data=v_idxs.ravel(),
        shape=shape,
        offsets=np.arange(0, 4 * 4 + 1, dtype=np.int64),  # every entry has 1 variant
    )

    dummy_haps = Haps(
        reference=dummy_ref,
        variants=dummy_vars,
        genotypes=dummy_genos,
        dosages=None,
        kind=RaggedSeqs,
    )

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
        track_offsets=lengths_to_offsets(np.full(4, t_len + 2 * max_jitter)),
    )
    lengths = np.diff(offsets)
    data = repeat(data, "(r i) -> (r s i)", r=4, s=4)
    offsets = lengths_to_offsets(repeat(lengths, "r -> (r s)", s=4))
    dummy_itvs = {
        "read-depth": RaggedIntervals.from_offsets(
            data=data, shape=(4, 4), offsets=offsets
        )
    }

    # (r), want tracks of [0, 0, 1, 0, 0] for each region so that pad values of 0 are obvious
    track_regions = dummy_regions.copy()
    track_regions[:, 1] -= max_jitter
    track_regions[:, 2] = track_regions[:, 1] + 5 + max_jitter
    t_len = 5
    one_track = np.zeros(t_len + 2 * max_jitter, dtype=np.float32)
    one_track[2] = 1
    data, offsets = tracks_to_intervals(
        regions=track_regions,
        tracks=repeat(one_track, "l -> (r l)", r=len(dummy_regions)),
        track_offsets=lengths_to_offsets(np.full(4, t_len + 2 * max_jitter)),
    )
    lengths = np.diff(offsets)
    dummy_itvs["annot"] = RaggedIntervals.from_offsets(
        data=data, shape=4, offsets=offsets
    )

    avail_tracks = {"read-depth": TrackType.SAMPLE, "annot": TrackType.ANNOT}

    dummy_tracks = Tracks(dummy_itvs, avail_tracks, avail_tracks)

    dummy_recon = HapsTracks(dummy_haps, dummy_tracks)

    dummy_dataset: RaggedDataset[RaggedSeqs, Ragged[np.float32]] = RaggedDataset(
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
        _idxer=dummy_idxer,
        _seqs=dummy_haps,
        _tracks=dummy_tracks,
        _recon=dummy_recon,
        _rng=np.random.default_rng(),
    )

    return dummy_dataset
