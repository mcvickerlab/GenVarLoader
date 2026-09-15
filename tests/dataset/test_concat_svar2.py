"""Characterization + merge tests for `gvl.concat` over a `.svar2` backend.

`_concat_svar2_ranges` had no coverage at all before #357. These tests pin its
OBSERVABLE contract -- a concatenated dataset reads identically to a single-shot
`gvl.write` over the union -- so the sparse-merge refactor has a real baseline
rather than a reference written in the same PR.

They deliberately assert on READS, not on file bytes: the on-disk layout is what
#357 changes, the read semantics are what must not change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl

_BED = pl.DataFrame(
    {
        "chrom": ["chr1", "chr1", "chr1"],
        "chromStart": [0, 5, 25],
        "chromEnd": [20, 15, 40],
    }
)


def _region_starts(path: Path) -> list[int]:
    """A dataset's public region row order, as `chromStart` values.

    `axis="regions"` concat's public row order is each shard's OWN input-bed
    row order concatenated in shard order (`_merged_bed` never re-sorts the
    *public* `input_regions.arrow`, only the on-disk `regions.npy`) -- it is
    NOT, in general, the same permutation as a single-shot write over the
    reassembled bed. `svar2_shards_by_regions` splits rows `[0, 2]` + `[1]`
    (not a contiguous prefix/suffix), which is exactly the non-block case
    where the two orders diverge: the merged dataset's public order comes out
    `[A, B, C]` while a single-shot write's is the original bed order
    `[A, C, B]`. `chromStart` is unique across this fixture's three regions,
    so it is a safe row-identity key -- the same idea as `_region_key_to_row`
    in `tests/dataset/test_concat.py`.
    """
    return pl.read_ipc(path / "input_regions.arrow")["chromStart"].to_list()


def _read_all(path: Path, reference: Path) -> list[np.ndarray]:
    """Every (region, sample) haplotype in the dataset, as padded byte arrays.

    Reading through the public API is the point: it exercises the range cache
    via `Svar2Haps._gather_inputs`, which is what the layout change touches.

    `reference` is the FASTA backing `svar2_store` (from `vcf_and_ref`):
    `Dataset.open` needs an explicit reference to reconstruct haplotypes --
    without one it can only return `RaggedVariants`, not `RaggedSeqs` (see
    `_open.py`'s `_build_seqs`, and every `reference=ref`-passing test in
    `test_svar2_dataset.py`).
    """
    ds = gvl.Dataset.open(path, reference=reference).with_seqs("haplotypes")
    out = []
    for r in range(ds.n_regions):
        for s in range(ds.n_samples):
            hap = ds[r, s]
            out.append(np.asarray(hap.to_padded(b"N")))
    return out


@pytest.fixture(scope="module")
def svar2_shards_by_samples(
    svar2_store: Path, tmp_path_factory
) -> tuple[list[Path], Path]:
    """Two single-sample shards over the same regions + the single-shot oracle."""
    from genoray import SparseVar2

    d = tmp_path_factory.mktemp("svar2_concat_samples")
    shards = []
    # Disjoint, sorted sample sets: exactly what _concat_validate requires on
    # axis="samples" (no overlap, and the merged order is sorted(union)).
    for names in (["S0", "S2"], ["S1"]):
        p = d / f"shard_{'_'.join(names)}.gvl"
        gvl.write(
            p, _BED, variants=SparseVar2(svar2_store), samples=names, overwrite=True
        )
        shards.append(p)
    full = d / "full.gvl"
    gvl.write(
        full,
        _BED,
        variants=SparseVar2(svar2_store),
        samples=["S0", "S1", "S2"],
        overwrite=True,
    )
    return shards, full


@pytest.fixture(scope="module")
def svar2_shards_by_regions(
    svar2_store: Path, tmp_path_factory
) -> tuple[list[Path], Path]:
    """Two region shards over the same samples + the single-shot oracle."""
    from genoray import SparseVar2

    d = tmp_path_factory.mktemp("svar2_concat_regions")
    shards = []
    for i, rows in enumerate(([0, 2], [1])):
        p = d / f"shard_{i}.gvl"
        gvl.write(
            p,
            _BED[rows],
            variants=SparseVar2(svar2_store),
            samples=None,
            overwrite=True,
        )
        shards.append(p)
    full = d / "full.gvl"
    gvl.write(
        full, _BED, variants=SparseVar2(svar2_store), samples=None, overwrite=True
    )
    return shards, full


def test_concat_svar2_samples_reads_like_single_write(
    svar2_shards_by_samples, vcf_and_ref: tuple[Path, Path], tmp_path: Path
):
    """Concatenating sample shards must read identically to writing once."""
    shards, full = svar2_shards_by_samples
    ref = vcf_and_ref[1]
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="samples", overwrite=True)

    ds_m = gvl.Dataset.open(out, reference=ref)
    ds_f = gvl.Dataset.open(full, reference=ref)
    assert ds_m.samples == ds_f.samples
    assert ds_m.n_regions == ds_f.n_regions

    for a, b in zip(_read_all(out, ref), _read_all(full, ref)):
        np.testing.assert_array_equal(a, b)


def test_concat_svar2_regions_reads_like_single_write(
    svar2_shards_by_regions, vcf_and_ref: tuple[Path, Path], tmp_path: Path
):
    """Concatenating region shards must read identically to writing once.

    Matches merged rows to single-shot rows by `chromStart` (see
    `_region_starts`), not raw row position: `svar2_shards_by_regions`'s
    `[0, 2]` + `[1]` split deliberately does not reconstruct the original
    bed's row order when concatenated, so the merged dataset's public region
    order genuinely differs from the single-shot write's. The two assertions
    below pin exactly what each one is -- see their failure messages -- rather
    than leaving the permutation unverified prose. This divergence is not a
    `_concat_svar2_ranges` bug: the underlying `svar2_ranges` cache arrays
    (`vk_snp_range`/`vk_indel_range`/`dense_snp_range`/`dense_indel_range`)
    were verified byte-identical between the two datasets when compared at
    matching on-disk (sorted) rows.
    """
    shards, full = svar2_shards_by_regions
    ref = vcf_and_ref[1]
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions", overwrite=True)

    ds_m = gvl.Dataset.open(out, reference=ref)
    ds_f = gvl.Dataset.open(full, reference=ref)
    assert ds_m.n_regions == ds_f.n_regions
    assert ds_m.samples == ds_f.samples

    n_samples = ds_m.n_samples
    all_m = _read_all(out, ref)
    all_f = _read_all(full, ref)
    starts_m = _region_starts(out)
    row_f_by_start = {start: i for i, start in enumerate(_region_starts(full))}
    assert set(starts_m) == set(row_f_by_start), (
        "merged and single-shot datasets disagree on which regions exist"
    )
    assert starts_m == [0, 25, 5], (
        "axis='regions' concat's public row order is each shard's own input-bed "
        f"row order concatenated in shard order, not a re-sort of the union; got "
        f"{starts_m}, expected [0, 25, 5] (shard 0's rows [0, 25] then shard 1's "
        "row [5])"
    )
    assert _region_starts(full) == [0, 5, 25], (
        "a single-shot write's public row order is the input bed's own row "
        f"order; got {_region_starts(full)}, expected [0, 5, 25] (this fixture's "
        "_BED is already sorted by chromStart)"
    )

    for rm, start in enumerate(starts_m):
        rf = row_f_by_start[start]
        for s in range(n_samples):
            a = all_m[rm * n_samples + s]
            b = all_f[rf * n_samples + s]
            np.testing.assert_array_equal(a, b)
