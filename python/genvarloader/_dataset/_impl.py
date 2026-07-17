from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, NoReturn, TypeVar, overload

import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray
from seqpro.rag import Ragged
from typing_extensions import Self, assert_never

from .._ragged import (
    RaggedAnnotatedHaps,
    RaggedIntervals,
    RaggedSeqs,
    RaggedTracks,
)
from .._torch import TORCH_AVAILABLE, TorchDataset, get_dataloader
from .._types import AnnotatedHaps, Idx, StrIdx
from ._flat_variants import DummyVariant
from ._indexing import DatasetIndexer, SpliceIndexer, is_str_arr
from ._insertion_fill import InsertionFill
from ._rag_variants import RaggedVariants
from ._haps import _svar_format_fields
from ._reconstruct import (
    Haps,
    HapsTracks,
    Ref,
    SeqsTracks,
    Tracks,
    TrackType,
    _build_reconstructor,
)
from ._flat_variants import VarWindowOpt
from ._reference import Reference
from ._splice import SpliceMap

if TYPE_CHECKING:
    import seqpro as sp

if TORCH_AVAILABLE:
    import torch
    import torch.utils.data as td


_py_open = open


@dataclass(slots=True, frozen=True)
class Dataset:
    """A dataset of genotypes, reference sequences, and intervals.

    .. note::

        This class is not meant to be instantiated directly. Use the :py:meth:`Dataset.open() <genvarloader.Dataset.open()>`
        method to open a dataset after writing the data with :py:func:`genvarloader.write()` or the GenVarLoader CLI.

    GVL Datasets act like a collection of lazy ragged arrays that can be lazily subset or eagerly indexed as a 2D NumPy array. They
    have an effective shape of :code:`(n_regions, n_samples, [tracks], [ploidy], output_length)`, but only the region and sample
    dimensions can be indexed directly since the return value is generally a tuple of arrays.

    **Eager indexing**

    .. code-block:: python

        dataset[0, 9]  # first region, 10th sample
        dataset[:10]  # first 10 regions and all samples
        dataset[:10, :5]  # first 10 regions and 5 samples
        dataset[[2, 2], [0, 1]]  # 3rd region, 1st and 2nd samples

    **Lazy indexing**

    See :meth:`Dataset.subset_to() <Dataset.subset_to()>`. This is useful, for example, to create
    splits for training, validation, and testing, or filter out regions or samples after writing a full dataset.
    This is also necessary if you intend to create a Pytorch :class:`DataLoader <torch.utils.data.DataLoader>`
    from the Dataset using :meth:`Dataset.to_dataloader() <Dataset.to_dataloader()>`.

    **Return values**

    The return value depends on the :code:`Dataset` state, namely :attr:`sequence_type <Dataset.sequence_type>`,
    :attr:`active_tracks <Dataset.active_tracks>`, and :attr:`output_length <Dataset.output_length>`.
    These can all be modified after opening a :code:`Dataset` using the following methods:
    - :meth:`Dataset.with_seqs() <Dataset.with_seqs()>`
    - :meth:`Dataset.with_tracks() <Dataset.with_tracks()>`
    - :meth:`Dataset.with_len() <Dataset.with_len()>`
    """

    @staticmethod
    @overload
    def open(
        path: str | Path,
        reference: None = ...,
        jitter: int = 0,
        rng: int | np.random.Generator | None = False,
        deterministic: bool = True,
        rc_neg: bool = True,
        min_af: float | None = None,
        max_af: float | None = None,
        var_fields: list[str] | None = None,
        region_names: str | None = None,
        splice_info: str | tuple[str, str] | None = None,
        var_filter: Literal["exonic"] | None = None,
        *,
        svar: str | Path | None = None,
        svar2: str | Path | None = None,
    ) -> RaggedDataset[MaybeRSEQ, MaybeRTRK]: ...
    @staticmethod
    @overload
    def open(
        path: str | Path,
        reference: str | Path | Reference,
        jitter: int = 0,
        rng: int | np.random.Generator | None = False,
        deterministic: bool = True,
        rc_neg: bool = True,
        min_af: float | None = None,
        max_af: float | None = None,
        var_fields: list[str] | None = None,
        region_names: str | None = None,
        splice_info: str | tuple[str, str] | None = None,
        var_filter: Literal["exonic"] | None = None,
        *,
        svar: str | Path | None = None,
        svar2: str | Path | None = None,
    ) -> RaggedDataset[RaggedSeqs, MaybeRTRK]: ...
    @staticmethod
    def open(
        path: str | Path,
        reference: str | Path | Reference | None = None,
        jitter: int = 0,
        rng: int | np.random.Generator | None = False,
        deterministic: bool = True,
        rc_neg: bool = True,
        min_af: float | None = None,
        max_af: float | None = None,
        var_fields: list[str] | None = None,
        region_names: str | None = None,
        splice_info: str | tuple[str, str] | None = None,
        var_filter: Literal["exonic"] | None = None,
        *,
        svar: str | Path | None = None,
        svar2: str | Path | None = None,
    ) -> RaggedDataset[MaybeRSEQ, MaybeRTRK]:
        """Open a dataset from a path.

        If no reference genome is provided, the dataset cannot yield sequences.
        Will initialize the dataset such that it will return tracks and haplotypes (reference sequences if no genotypes) if possible.
        If tracks are available, they will be set to be returned in alphabetical order.

        Args:
            path: Path to a dataset.
            reference: Path to a reference genome.
            jitter: Amount of jitter to use, cannot be more than the maximum jitter of the dataset.
            rng: Random seed or np.random.Generator for any stochastic operations.
            deterministic: Whether to use randomized or deterministic algorithms. If set to True, this will disable random
                shifting of longer-than-requested haplotypes.
            rc_neg: Whether to reverse-complement sequences and reverse tracks on negative strands.
            min_af: The minimum allele frequency to include in the dataset. If dataset is not backed by SVAR genotypes, this will raise an error.
            max_af: The maximum allele frequency to include in the dataset. If dataset is not backed by SVAR genotypes, this will raise an error.
            var_fields: The variant fields to include in the dataset. Defaults to the
                minimum useful set ``["alt", "ilen", "start"]``. Pass additional
                field names (e.g. ``"ref"``, ``"dosage"``, or any info column
                present in the source variants table) to load them eagerly at open
                time. Must be a subset of :attr:`available_var_fields`.
            region_names: The name of the column in the region-of-interest table (BED) to
                use as the region names.
            splice_info: A string or tuple of strings representing the splice information to use.
                If a string, it will be used as the transcript ID and the exons are expected to be in order.
                If a tuple of strings, the first string will be used as the transcript ID and the second string will be used as the exon number.
                If a dictionary, the keys will be used as the transcript ID and the values should be the row number for each exon, in order.
                If False, splicing will be disabled.
            var_filter: Whether to filter variants. If set to :code:`"exonic"`, only exonic variants will be applied.
            svar: Override the recorded SVAR location. Use when the original SVAR has
                moved and the dataset cannot find it via the stored relative/absolute
                path or by sibling discovery.
            svar2: Override the recorded ``.svar2`` location. Use when the original
                ``.svar2`` store has moved and the dataset cannot find it via the
                stored relative/absolute path or by sibling discovery.
        """
        from ._open import OpenRequest

        return OpenRequest(
            path=Path(path),
            reference=reference,
            jitter=jitter,
            rng=rng,
            deterministic=deterministic,
            rc_neg=rc_neg,
            min_af=min_af,
            max_af=max_af,
            var_fields=var_fields,
            region_names=region_names,
            splice_info=splice_info,
            var_filter=var_filter,
            svar=svar,
            svar2=svar2,
        ).resolve()

    def with_settings(
        self,
        jitter: int | None = None,
        rng: int | np.random.Generator | None = None,
        deterministic: bool | None = None,
        rc_neg: bool | None = None,
        min_af: float | Literal[False] | None = None,
        max_af: float | Literal[False] | None = None,
        var_fields: list[str] | None = None,
        splice_info: str | tuple[str, str] | Literal[False] | None = None,
        var_filter: Literal[False, "exonic"] | None = None,
        flank_length: int | None = None,
        token_alphabet: "str | bytes | sp.NucleotideAlphabet | None" = None,
        unknown_token: int | None = None,
        dummy_variant: "DummyVariant | Literal[False] | None" = None,
        unphased_union: bool | None = None,
        realign_tracks: bool | None = None,
    ) -> Self:
        """Modify settings of the dataset, returning a new dataset without modifying the old one.

        Args:
            jitter: How much jitter to use. Must be non-negative and <= the :attr:`max_jitter <genvarloader.Dataset.max_jitter>` of the dataset.
            rng: Random seed or np.random.Generator for non-deterministic operations e.g. jittering and shifting longer-than-requested haplotypes.
            deterministic: Whether to use randomized or deterministic algorithms. If set to True, this will disable random
                shifting of longer-than-requested haplotypes and, for unphased variants, will enable deterministic variant assignment
                and always apply the highest CCF group. Note that for unphased variants, this will mean not all possible haplotypes
                can be returned.
            rc_neg: Whether to reverse-complement sequences and reverse tracks on negative strands.
            min_af: The minimum allele frequency to include in the dataset. If set to :code:`False`, disables this filter.
                If dataset is not backed by SVAR genotypes, this will raise an error.
            max_af: The maximum allele frequency to include in the dataset. If set to :code:`False`, disables this filter.
                If dataset is not backed by SVAR genotypes, this will raise an error.
            var_fields: The variant fields to include in the dataset.
            splice_info: A string or tuple of strings representing the splice information to use.
                If a string, it will be used as the transcript ID and the exons are expected to be in order.
                If a tuple of strings, the first string will be used as the transcript ID and the second string will be used as the exon number.
                If a dictionary, the keys will be used as the transcript ID and the values should be the row number for each exon, in order.
                If False, splicing will be disabled.
            var_filter: Whether to filter variants. If set to :code:`"exonic"`, only exonic variants will be applied.
            flank_length: Number of reference-sequence bases to fetch as flanks around each variant. Stored on
                the :class:`Haps` reconstructor for use by the flat-window output mode.
            token_alphabet: Characters that define the token alphabet (e.g. ``b"ACGT"``, ``"ACGT"``, or
                ``seqpro.alphabets.DNA``). Accepts a :class:`str`, :class:`bytes`, or
                :class:`seqpro.NucleotideAlphabet` and is normalized to ``bytes``; position ``i``
                in the alphabet maps to integer token ``i``. Must be supplied together with *unknown_token*.
            unknown_token: Integer token to assign to any byte not present in *token_alphabet*. Must be supplied
                together with *token_alphabet*.
            dummy_variant: A :class:`DummyVariant` to insert into empty (region, sample, ploid) variant
                groups so every group has at least one variant. Valid for the ``"variants"``
                and ``"variant-windows"`` outputs (see
                :meth:`with_seqs <genvarloader.Dataset.with_seqs>`); indexing any other output
                kind with a dummy set raises. For token outputs (the ride-along ``flank_tokens``
                and the variant-window token buffers) the dummy entry is filled entirely with
                ``unknown_token``. Pass :code:`False` to disable.
            unphased_union: When :code:`True`, fold the stored ``ploidy`` haplotypes onto a single haploid
                sequence: the union of called ALTs per ``(region, sample)``. ``ds.ploidy`` and
                ``n_variants(...)`` then report ploidy ``1``, and ``"variants"`` /
                ``"variant-windows"`` output decode at ploidy ``1``. Phase is discarded (suited
                to unphased somatic calls); ALT occurrences are concatenated across haplotypes
                with no sort or dedup (a hom call appears once per haplotype). Requires a dataset
                with genotypes and is incompatible with ``"haplotypes"`` / ``"annotated"``
                output (raises). See issue #222.
            realign_tracks: Whether to re-align track values to haplotype coordinates when both
                haplotypes and float tracks are active. Default ``True``. Set ``False``
                for reference-coordinate (as-is) tracks; required ``False`` for
                ``variant-windows`` + tracks and for ``kind="intervals"`` with any
                variant-aware seq mode.
        """
        to_evolve = {}

        if jitter is not None:
            if jitter != self.jitter:
                if isinstance(self.output_length, int):
                    min_r_len: int = (
                        self._full_regions[:, 2] - self._full_regions[:, 1]
                    ).min()
                    max_output_length = min_r_len + 2 * self.max_jitter
                    eff_length = self.output_length + 2 * jitter

                    if eff_length > max_output_length:
                        raise ValueError(
                            f"Jitter-expanded output length (out_len={self.output_length}) + 2 * ({jitter=}) = {eff_length} must be less"
                            f" than or equal to the maximum output length of the dataset ({max_output_length})."
                            f" The maximum output length is the minimum region length ({min_r_len}) + 2 * (max_jitter={self.max_jitter})."
                        )

                to_evolve["jitter"] = jitter

        if rng is not None:
            to_evolve["_rng"] = np.random.default_rng(rng)

        if deterministic is not None:
            to_evolve["deterministic"] = deterministic

        if rc_neg is not None:
            to_evolve["rc_neg"] = rc_neg

        if min_af is not None or max_af is not None:
            if not isinstance(self._seqs, Haps):
                raise ValueError("Dataset has no genotypes to filter.")

            if min_af is None:
                min_af = self._seqs.min_af
            elif min_af is False:
                min_af = None

            if max_af is None:
                max_af = self._seqs.max_af
            elif max_af is False:
                max_af = None

            haps = to_evolve.get("_seqs", self._seqs)
            haps = replace(haps, min_af=min_af, max_af=max_af)
            to_evolve["_seqs"] = haps

        if var_fields is not None:
            missing = list(set(var_fields) - set(self.available_var_fields))
            if missing or not isinstance(self._seqs, Haps):
                raise ValueError(f"Missing variant fields: {missing}")

            from ._svar2_haps import Svar2Haps

            if isinstance(self._seqs, Svar2Haps):
                # SVAR2 field values are read on demand by the decode kernel
                # (decode_variants_from_svar2_readbound); there is no SVAR1 variants
                # table to lazily load INFO/dosage/custom-FORMAT columns from — this
                # reconstructor's `variants` is a dummy placeholder.
                haps = replace(
                    to_evolve.get("_seqs", self._seqs), var_fields=var_fields
                )
                to_evolve["_seqs"] = haps
            else:
                haps = to_evolve.get("_seqs", self._seqs)
                # Discover custom FORMAT fields so we don't try to load them as INFO.
                custom_fmt = _svar_format_fields(haps.variants.path.parent)
                # Lazily load any newly-requested info columns into the existing
                # _Variants struct (mutates haps.variants.info in place).
                builtin = {"alt", "ilen", "start", "ref", "dosage"}
                new_info_fields = [
                    f
                    for f in var_fields
                    if f not in builtin
                    and f not in haps.variants.info
                    and f not in custom_fmt
                ]
                if new_info_fields:
                    haps.variants.load_info(new_info_fields)
                # Lazily memmap dosages if newly requested.
                if "dosage" in var_fields and haps.dosages is None:
                    haps = _lazy_load_dosages(self, haps)
                # Lazily memmap custom FORMAT fields if newly requested.
                new_custom_fields = {
                    f: custom_fmt[f]
                    for f in var_fields
                    if f in custom_fmt and f not in haps.var_field_data
                }
                if new_custom_fields:
                    haps = _lazy_load_custom_fields(self, haps, new_custom_fields)
                haps = replace(haps, var_fields=var_fields)
                to_evolve["_seqs"] = haps

        if splice_info is not None:
            if splice_info is False:
                splice_idxer = None
                spliced_bed = None
            else:
                sm, spliced_bed = SpliceMap.from_bed(splice_info, self._full_bed)
                if (
                    sm.splice_map.to_packed().data.max() >= self._idxer.n_regions
                    or sm.splice_map.to_packed().data.min() < -self._idxer.n_regions
                ):
                    raise ValueError(
                        "Found indices in the splice map that are out of bounds for the dataset."
                    )
                splice_idxer = SpliceIndexer(map=sm, dsi=self._idxer)
            to_evolve["_sp_idxer"] = splice_idxer
            to_evolve["_spliced_bed"] = spliced_bed

        if var_filter is not None:
            if not isinstance(self._seqs, Haps):
                raise ValueError(
                    "Filtering variants can only be done when the dataset has variants."
                )

            if var_filter is False:
                var_filter = None

            if var_filter != self._seqs.filter:
                haps = to_evolve.get("_seqs", self._seqs)
                to_evolve["_seqs"] = replace(haps, filter=var_filter)

        if (
            flank_length is not None
            or token_alphabet is not None
            or unknown_token is not None
        ):
            if not isinstance(self._seqs, Haps):
                raise ValueError(
                    "Flank settings require a dataset with genotypes (variants)."
                )
            haps = to_evolve.get("_seqs", self._seqs)
            new_flank_len = haps.flank_length if flank_length is None else flank_length
            lut, lut_dtype = haps.token_lut, haps.token_dtype
            if token_alphabet is not None or unknown_token is not None:
                if token_alphabet is None or unknown_token is None:
                    raise ValueError(
                        "token_alphabet and unknown_token must be set together."
                    )
                from ._flat_flanks import build_token_lut

                lut, lut_dtype = build_token_lut(token_alphabet, unknown_token)
            if new_flank_len and lut is None:
                raise ValueError(
                    "flank_length requires a token LUT; pass token_alphabet and"
                    " unknown_token to with_settings(...) (in this or a prior call)."
                )
            to_evolve["_seqs"] = replace(
                haps,
                flank_length=new_flank_len,
                token_lut=lut,
                token_dtype=lut_dtype,
                unknown_token=(
                    unknown_token if unknown_token is not None else haps.unknown_token
                ),
            )

        if dummy_variant is not None:
            if dummy_variant is False:
                # disable is a no-op on datasets without variants/genotypes
                if isinstance(self._seqs, Haps):
                    haps = to_evolve.get("_seqs", self._seqs)
                    to_evolve["_seqs"] = replace(haps, dummy_variant=None)
            else:
                if not isinstance(self._seqs, Haps):
                    raise ValueError(
                        "dummy_variant requires a dataset with variants/genotypes."
                    )
                haps = to_evolve.get("_seqs", self._seqs)
                to_evolve["_seqs"] = replace(haps, dummy_variant=dummy_variant)

        if unphased_union is not None:
            if not isinstance(self._seqs, Haps):
                raise ValueError(
                    "unphased_union requires a dataset with genotypes (variants)."
                )
            haps = to_evolve.get("_seqs", self._seqs)
            to_evolve["_seqs"] = replace(haps, unphased_union=unphased_union)

        if realign_tracks is not None:
            to_evolve["realign_tracks"] = realign_tracks

        # If any source state changed, rebuild _recon via the factory.
        if (
            "_seqs" in to_evolve
            or "_tracks" in to_evolve
            or "realign_tracks" in to_evolve
        ):
            new_seqs = to_evolve.get("_seqs", self._seqs)
            new_tracks = to_evolve.get("_tracks", self._tracks)
            new_realign = to_evolve.get("realign_tracks", self.realign_tracks)
            to_evolve["_recon"] = _build_reconstructor(
                new_seqs, new_tracks, self._seqs_kind, new_realign
            )

        self = replace(self, **to_evolve)
        self._check_valid_state()

        return self

    def _check_valid_state(self):
        if self.is_spliced:
            if self.jitter > 0:
                raise RuntimeError(
                    "Jitter is not supported with splicing. Please set jitter to 0."
                )

            if not self.deterministic:
                raise RuntimeError(
                    "Non-deterministic algorithms are not supported with splicing. Please set deterministic to True."
                )

            if self.sequence_type == "variant-windows":
                raise ValueError("Splicing is not supported with variant-windows.")
            if self.sequence_type == "variants" and self.output_format == "flat":
                raise ValueError("Spliced variants require output_format='ragged'.")

        if self.jitter < 0:
            raise ValueError(f"Jitter ({self.jitter}) must be a non-negative integer.")
        elif self.jitter > self.max_jitter:
            raise ValueError(
                f"Jitter ({self.jitter}) must be less than or equal to the maximum jitter of the dataset ({self.max_jitter})."
            )

        if isinstance(self.output_length, int):
            if self.sequence_type == "variants":
                raise ValueError(
                    "Output length must be ragged when the sequence type is variants."
                )

            if self.output_length < 1:
                raise ValueError(
                    f"Output length ({self.output_length}) must be a positive integer."
                )

            min_r_len: int = (self._full_regions[:, 2] - self._full_regions[:, 1]).min()
            max_output_length = min_r_len + 2 * self.max_jitter
            eff_length = self.output_length + 2 * self.jitter
            if eff_length > max_output_length:
                raise ValueError(
                    f"Effective length (out_len={self.output_length}) + 2 * ({self.jitter=}) = {eff_length} must be less"
                    f" than or equal to the maximum output length of the dataset ({max_output_length})."
                    f" The maximum output length is the minimum region length ({min_r_len}) + 2 * (max_jitter={self.max_jitter})."
                )
        elif self.output_length == "variable" and self.sequence_type == "variants":
            raise ValueError(
                "Output length must be ragged when the sequence type is variants."
            )

        if (
            isinstance(self._seqs, Haps)
            and self._seqs.unphased_union
            and self.sequence_type in ("haplotypes", "annotated")
        ):
            raise ValueError(
                "unphased_union is incompatible with 'haplotypes'/'annotated' output"
                " (a union of phased sequences is ill-defined). Use 'variant-windows'"
                " or 'variants', or clear the flag with"
                " with_settings(unphased_union=False)."
            )

        if self.sequence_type == "variant-windows":
            haps = self._seqs
            if not isinstance(haps, Haps) or haps.window_opt is None:
                raise ValueError(
                    "with_seqs('variant-windows') requires a VarWindowOpt"
                    " (pass it to with_seqs)."
                )

    def with_len(
        self, output_length: Literal["ragged", "variable"] | int
    ) -> ArrayDataset | RaggedDataset:
        """Modify the output length of the dataset, returning a new dataset without modifying the old one.

        Args:
            output_length: The output length. Can be set to :code:`"ragged"` or :code:`"variable"` to allow for variable length sequences.
                If set to an integer, all sequences will be padded or truncated to this length. See the
                `online documentation <https://genvarloader.readthedocs.io/en/latest/dataset.html>`_ for more information.
        """
        if isinstance(output_length, int) or output_length == "variable":
            if isinstance(output_length, int):
                if output_length < 1:
                    raise ValueError(
                        f"Output length ({output_length}) must be a positive integer."
                    )
                min_r_len: int = (
                    self._full_regions[:, 2] - self._full_regions[:, 1]
                ).min()
                max_output_length = min_r_len + 2 * self.max_jitter
                eff_length = output_length + 2 * self.jitter

                if eff_length > max_output_length:
                    raise ValueError(
                        f"Jitter-expanded output length (out_len={self.output_length}) + 2 * ({self.jitter=}) = {eff_length} must be less"
                        f" than or equal to the maximum output length of the dataset ({max_output_length})."
                        f" The maximum output length is the minimum region length ({min_r_len}) + 2 * (max_jitter={self.max_jitter})."
                    )

            return ArrayDataset(
                path=self.path,
                output_length=output_length,
                max_jitter=self.max_jitter,
                jitter=self.jitter,
                contigs=self.contigs,
                return_indices=self.return_indices,
                rc_neg=self.rc_neg,
                deterministic=self.deterministic,
                _idxer=self._idxer,
                _sp_idxer=self._sp_idxer,
                _full_bed=self._full_bed,
                _spliced_bed=self._spliced_bed,
                _full_regions=self._full_regions,
                _seqs=self._seqs,
                _tracks=self._tracks,
                _seqs_kind=self._seqs_kind,
                _recon=self._recon,
                _rng=self._rng,
                output_format=self.output_format,
                realign_tracks=self.realign_tracks,
            )
        else:
            out = RaggedDataset(
                path=self.path,
                output_length=output_length,
                max_jitter=self.max_jitter,
                jitter=self.jitter,
                contigs=self.contigs,
                return_indices=self.return_indices,
                rc_neg=self.rc_neg,
                deterministic=self.deterministic,
                _idxer=self._idxer,
                _sp_idxer=self._sp_idxer,
                _full_bed=self._full_bed,
                _spliced_bed=self._spliced_bed,
                _full_regions=self._full_regions,
                _seqs=self._seqs,
                _tracks=self._tracks,
                _seqs_kind=self._seqs_kind,
                _recon=self._recon,
                _rng=self._rng,
                output_format=self.output_format,
                realign_tracks=self.realign_tracks,
            )

        out._check_valid_state()

        return out

    def with_seqs(
        self,
        kind: Literal[
            "reference", "haplotypes", "annotated", "variants", "variant-windows"
        ]
        | None,
        window_opt: "VarWindowOpt | None" = None,
    ):
        """Return a new dataset with the specified sequence type.

        The sequence type can be one of the following:

        - :code:`"reference"`: reference sequences.
        - :code:`"haplotypes"`: personalized haplotype sequences.
        - :code:`"annotated"`: annotated haplotype sequences, which includes personalized haplotypes along with annotations.
        - :code:`"variants"`: no sequences, just variants as :class:`RaggedVariants`

        Annotated haplotypes are returned as the :class:`~genvarloader._types.AnnotatedHaps` class which is roughly:

        .. code-block:: python

            class AnnotatedHaps:
                haps: NDArray[np.bytes_]
                var_idxs: NDArray[np.int32]
                ref_coords: NDArray[np.int32]

        where :code:`haps` are the haplotypes as bytes/S1, and :code:`var_idxs` and :code:`ref_coords` are
        arrays with the same shape as :code:`haps` that annotate every nucleotide with the variant index and
        reference coordinate it corresponds to. A variant index of -1 corresponds to a reference nucleotide, and a reference
        coordinate of -1 corresponds to padded nucleotides that were added for regions beyond the bounds of the reference genome.
        i.e. if the region's start position is negative or the end position is beyond the end of the reference genome.

        For example, a toy result for :code:`chr1:1-10` could be:

        .. code-block:: text

            haps:        A C G  T ...  T T  A ...
            var_idxs:   -1 3 3 -1 ... -1 4 -1 ...
            ref_coords:  1 2 2  3 ...  6 7  9 ...

        where variant 3 is a 1 bp :code:`CG` insertion and variant 4 is a 1 bp deletion :code:`T-`. Note that the first nucleotide
        of every indel maps to a reference position since :func:`gvl.write() <genvarloader.write()>` expects that variants
        are all left-aligned.

        .. important::

            The :code:`var_idxs` are numbered with respect to the full set of variants even if the variants were extracted from per-chromosome VCFs/PGENs.
            So a variant index of 0 corresponds to the first variant across all chromosomes. Thus, if you want to map the variant index to per-chromosome VCFs/PGENs, you will
            need to subtract the number of variants on all other chromosomes before the variant index to get the correct variant index in the VCF/PGEN. Relevant values
            can be obtained by instantiating a `gvl.Variants` class from the VCFs/PGENs and accessing the `Variants.records.contig_offsets` attribute.

        If the Dataset's output length is :code:`"ragged"`, then annotated haplotypes will be :class:`~genvarloader._ragged.RaggedAnnotatedHaps` where each
        field is a Ragged array instead of NumPy arrays.

        Args:
            kind: The type of sequences to return. Can be one of :code:`"reference"`, :code:`"haplotypes"`, :code:`"annotated"`, :code:`"variants"`, or :code:`None`
                to return no sequences.
            window_opt: Required when :code:`kind="variant-windows"`. A :class:`VarWindowOpt`
                configuring the flank length, token alphabet, and unknown token used to
                extract fixed-length windows around each variant.
        """
        # Validate the requested kind against storage state.
        if kind is None:
            tracks_active = self._tracks is not None and bool(
                self._tracks.active_tracks
            )
            if not tracks_active:
                raise RuntimeError(
                    "Dataset is set to only return sequences, so setting sequence_type to None would"
                    " result in a Dataset that cannot return anything."
                )
        elif kind == "reference":
            if not isinstance(self._seqs, (Haps, Ref)):
                raise ValueError("Dataset has no reference to yield sequences from.")
            if self._seqs.reference is None:
                raise ValueError(
                    "Dataset has no reference genome to reconstruct sequences from."
                )
        elif kind in ("haplotypes", "annotated", "variants"):
            if not isinstance(self._seqs, Haps):
                raise ValueError(
                    "Dataset has no genotypes to yield haplotypes/variants from."
                )
        elif kind == "variant-windows":
            if not isinstance(self._seqs, Haps):
                raise ValueError(
                    "Dataset has no genotypes to yield variant windows from."
                )
            if window_opt is None:
                raise ValueError(
                    "with_seqs('variant-windows') requires a VarWindowOpt, e.g."
                    " with_seqs('variant-windows', VarWindowOpt(flank_length=...,"
                    " token_alphabet=..., unknown_token=...))."
                )
            if window_opt.ref == "allele" and self._seqs.variants.ref is None:
                raise ValueError(
                    "VarWindowOpt(ref='allele') needs REF alleles, but this dataset"
                    " has none. Use ref='window', or write the dataset with REF."
                )
        else:
            assert_never(kind)

        new_seqs = self._seqs
        if kind == "variant-windows":
            from ._flat_flanks import build_token_lut

            # Both invariants were established in the validation branch above; the
            # assert narrows them for the type checker (Ref has no flank fields).
            assert isinstance(self._seqs, Haps) and window_opt is not None
            lut, lut_dtype = build_token_lut(
                window_opt.token_alphabet, window_opt.unknown_token
            )
            new_seqs = replace(
                self._seqs,
                token_lut=lut,
                token_dtype=lut_dtype,
                window_opt=window_opt,
                unknown_token=window_opt.unknown_token,
            )
        if (
            kind in ("haplotypes", "annotated")
            and isinstance(new_seqs, Haps)
            and new_seqs.unphased_union
        ):
            raise ValueError(
                "unphased_union is incompatible with 'haplotypes'/'annotated' output"
                " (a union of phased sequences is ill-defined). Use 'variant-windows'"
                " or 'variants', or clear the flag with"
                " with_settings(unphased_union=False)."
            )
        new_recon = _build_reconstructor(
            new_seqs, self._tracks, kind, self.realign_tracks
        )
        return replace(self, _seqs=new_seqs, _seqs_kind=kind, _recon=new_recon)

    def with_tracks(
        self,
        tracks: str | list[str] | Literal[False] | None = None,
        kind: Literal["tracks", "intervals"] | None = None,
    ):
        """Modify which tracks to return, returning a new dataset without modifying the old one.

        Args:
            tracks: The tracks to return. Can be a (list of) track names or :code:`False` to return no tracks.
            kind: The container type to return tracks as: :code:`"tracks"` for :class:`RaggedTracks`
                or :code:`"intervals"` for :class:`RaggedIntervals`. If :code:`None`, keeps the
                dataset's current track container kind.
        """
        if self._tracks is None:
            logger.warning("Dataset has no tracks, so this method has no effect.")
            return self

        if tracks is None:
            tracks = False if self.active_tracks is None else self.active_tracks

        if kind == "tracks":
            _kind = RaggedTracks
        elif kind == "intervals":
            _kind = RaggedIntervals
        elif kind is None:
            _kind = self._tracks.kind
        else:
            assert_never(kind)

        # Compute the new tracks state (active set + kind).
        if tracks is False:
            # User-deactivate all tracks.
            new_tracks = self._tracks.with_tracks(None)
        elif isinstance(tracks, str):
            new_tracks = self._tracks.with_tracks([tracks]).to_kind(
                _kind,  # type: ignore[bad-argument-type]  # _kind is broader union; runtime branch ensures correct subtype
            )
        else:
            new_tracks = self._tracks.with_tracks(tracks).to_kind(
                _kind,  # type: ignore[bad-argument-type]  # _kind is broader union; runtime branch ensures correct subtype
            )

        # Validate: at least one of (seqs, tracks) must remain active.
        seqs_active = self._seqs_kind is not None and self._seqs is not None
        tracks_active = bool(new_tracks.active_tracks)
        if not seqs_active and not tracks_active:
            raise RuntimeError(
                "Dataset is set to only return tracks, so setting tracks to None would"
                " result in a Dataset that cannot return anything."
            )

        new_recon = _build_reconstructor(
            self._seqs, new_tracks, self._seqs_kind, self.realign_tracks
        )
        return replace(self, _tracks=new_tracks, _recon=new_recon)

    def with_insertion_fill(
        self,
        fill: InsertionFill | Mapping[str, InsertionFill],
    ) -> Self:
        """Configure how track values are filled at insertion sites.

        Only meaningful when the dataset returns haplotypes *and* tracks (i.e.
        when the reconstructor is :class:`HapsTracks`). Pure-reference and
        pure-haplotype datasets have no insertion fill to configure.

        Args:
            fill: Either a single :class:`InsertionFill` strategy applied to every
                active track, or a dict mapping track name to strategy. Tracks not
                in the dict fall back to :class:`Repeat5p`.
        """
        if self._tracks is None:
            raise ValueError("Dataset has no tracks; cannot configure insertion fill.")
        if self._seqs_kind not in ("haplotypes", "annotated", "variants"):
            raise ValueError(
                "with_insertion_fill is only meaningful for datasets with both "
                "haplotypes and tracks (use with_seqs to activate haplotypes first)."
            )
        if not self._tracks.active_tracks:
            raise ValueError(
                "with_insertion_fill is only meaningful when tracks are active "
                "(use with_tracks to activate tracks first)."
            )
        if not self.realign_tracks:
            raise ValueError(
                "with_insertion_fill has no effect when realign_tracks=False"
                " (insertion fill only applies during track re-alignment). Set"
                " with_settings(realign_tracks=True) first, or drop the call."
            )
        new_tracks = self._tracks.with_insertion_fill(fill)
        new_recon = _build_reconstructor(
            self._seqs, new_tracks, self._seqs_kind, self.realign_tracks
        )
        return replace(self, _tracks=new_tracks, _recon=new_recon)

    def with_output_format(self, fmt: Literal["ragged", "flat"]) -> "Dataset":
        """Return a copy that yields ``fmt`` containers from eager indexing.

        Args:
            fmt: ``"ragged"`` for ``_core.Ragged``-backed ``Ragged``/``RaggedVariants`` (default),
                or ``"flat"`` for pure-numpy ``FlatRagged``/``FlatVariants``.
        """
        if fmt not in ("ragged", "flat"):
            raise ValueError(f"output_format must be 'ragged' or 'flat', got {fmt!r}.")
        return replace(self, output_format=fmt)

    path: Path
    """Path to the dataset."""
    output_length: Literal["ragged", "variable"] | int
    """The output length. Can be set to :code:`"ragged"` or :code:`"variable"` to allow for variable length sequences.
    If set to an integer, all sequences will be padded or truncated to this length. See the
    `online documentation <https://genvarloader.readthedocs.io/en/latest/dataset.html>`_ for more information."""
    max_jitter: int
    """Maximum jitter."""
    return_indices: bool
    """Whether to return row and sample indices corresponding to the full dataset (no subsetting)."""
    contigs: list[str]
    """List of unique contigs."""
    jitter: int
    """How much jitter to use."""
    deterministic: bool
    """Whether to use randomized or deterministic algorithms. If set to :code:`False`, this will enable random
    shifting of longer-than-requested haplotypes and, for unphased variants, enable choosing sets of compatible variants proportional to their CCF;
    otherwise the dataset will always apply compatible sets with the highest CCF.
    
    .. note::
        This setting is independent of :attr:`~Dataset.jitter`, if you want no :attr:`~Dataset.jitter` you should set it to 0.
    """
    rc_neg: bool
    """Whether to reverse-complement the sequences on negative strands."""
    _full_bed: pl.DataFrame
    _spliced_bed: pl.DataFrame | None
    _full_regions: NDArray[np.int32]
    """Unjittered, sorted regions matching order on-disk."""
    _idxer: DatasetIndexer
    _sp_idxer: SpliceIndexer | None
    _seqs: (
        Ref | Haps[RaggedSeqs] | Haps[RaggedAnnotatedHaps] | Haps[RaggedVariants] | None
    )
    _tracks: Tracks[RaggedTracks] | Tracks[RaggedIntervals] | None
    _seqs_kind: (
        Literal["haplotypes", "reference", "annotated", "variants", "variant-windows"]
        | None
    )
    _recon: (
        Ref
        | Haps[RaggedSeqs]
        | Haps[RaggedAnnotatedHaps]
        | Haps[RaggedVariants]
        | Tracks
        | SeqsTracks
        | HapsTracks[RaggedSeqs, RaggedTracks]
        | HapsTracks[RaggedAnnotatedHaps, RaggedTracks]
        | HapsTracks[RaggedVariants, RaggedTracks]
    )
    _rng: np.random.Generator
    output_format: Literal["ragged", "flat"] = "ragged"
    """Container format for eager indexing. ``"ragged"`` (default) returns
    seqpro ``_core.Ragged`` / ``RaggedVariants``; ``"flat"`` returns pure-numpy ``FlatRagged`` /
    ``FlatVariants`` with zero allocations on the hot path. See ``with_output_format``."""
    realign_tracks: bool = True
    """Whether to re-align track *values* to haplotype coordinates when both
    haplotypes and float tracks (``kind="tracks"``) are active. ``True`` (default)
    uses the indel-aware realignment kernel; ``False`` returns reference-coordinate
    (as-is) tracks. Only affects ``Haps`` + float tracks; a no-op otherwise.
    Required ``False`` for ``variant-windows`` + tracks and for ``kind="intervals"``
    with any variant-aware seq mode."""

    @property
    def is_subset(self) -> bool:
        """Whether the dataset is a subset."""
        return self._idxer.is_subset

    @property
    def is_spliced(self) -> bool:
        """Whether the dataset is spliced."""
        return self._sp_idxer is not None

    @property
    def has_reference(self) -> bool:
        """Whether the dataset was provided a reference genome."""
        return self._seqs is not None

    @property
    def reference(self) -> Reference | None:
        """The reference genome."""
        if self._seqs is None:
            return None
        return self._seqs.reference

    @property
    def has_genotypes(self):
        """Whether the dataset has genotypes."""
        return isinstance(self._seqs, Haps)

    @property
    def has_intervals(self) -> bool:
        """Whether the dataset has intervals."""
        return self._tracks is not None

    @property
    def samples(self) -> list[str]:
        """The samples in the dataset."""
        return self._idxer.samples

    @property
    def regions(self) -> pl.DataFrame:
        """The input regions in the dataset.

        As they were provided to :func:`gvl.write() <genvarloader.write()>` i.e. with all BED columns plus any
        extra columns that were present.
        """
        if self._idxer.region_subset_idxs is None:
            return self._full_bed
        return self._full_bed[self._idxer.region_subset_idxs]

    @property
    def n_regions(self) -> int:
        """The number of (spliced) regions in the dataset."""
        return self.shape[0]

    @property
    def spliced_regions(self) -> pl.DataFrame | None:
        """The spliced regions in the dataset."""
        if self._spliced_bed is None or self._sp_idxer is None:
            raise ValueError("Dataset does not have splice information.")
        if self._sp_idxer.row_subset_idxs is None:
            return self._spliced_bed
        else:
            return self._spliced_bed[self._sp_idxer.row_subset_idxs]

    @property
    def n_samples(self) -> int:
        """The number of samples in the dataset."""
        return self._idxer.n_samples

    @property
    def ploidy(self) -> int | None:
        """The ploidy of the dataset.

        Reports ``1`` when ``unphased_union`` is set (the two stored haplotypes are
        folded onto a single haploid sequence); otherwise the stored ploidy.
        """
        if isinstance(self._seqs, Haps):
            if self._seqs.unphased_union:
                return 1
            return self._seqs.genotypes.shape[-2]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the dataset: :code:`(n_rows, n_samples)`."""
        if self._sp_idxer is None:
            return self._idxer.shape
        else:
            return self._sp_idxer.shape

    @property
    def full_shape(self) -> tuple[int, int]:
        """Return the full shape of the dataset, ignoring any subsetting: :code:`(n_rows, n_samples)`."""
        if self._sp_idxer is None:
            return self._idxer.full_shape
        else:
            return self._sp_idxer.full_shape

    @property
    def available_var_fields(self) -> list[str]:
        """Available variant fields."""
        match self._seqs:
            case Haps():
                return self._seqs.available_var_fields
            case _:
                return []

    @property
    def active_var_fields(self) -> list[str]:
        """Active variant fields."""
        match self._recon:
            case (Haps() as haps) | HapsTracks(haps=haps):
                return haps.var_fields
            case _:
                return []

    @property
    def available_tracks(self) -> list[str] | None:
        """The available tracks in the dataset."""
        if self._tracks is None:
            return
        return list(self._tracks.intervals)

    @property
    def active_tracks(self) -> list[str] | None:
        """The active tracks in the dataset."""
        if self._tracks is None:
            return
        return list(self._tracks.active_tracks)

    @property
    def _available_sequences(self) -> list[str] | None:
        """The available sequences in the dataset."""
        match self._seqs:
            case None:
                return None
            case Ref():
                return ["reference"]
            case Haps():
                return [
                    "reference",
                    "haplotypes",
                    "annotated",
                    "variants",
                    "variant-windows",
                ]
            case s:
                assert_never(s)

    @property
    def sequence_type(
        self,
    ) -> (
        Literal["haplotypes", "reference", "annotated", "variants", "variant-windows"]
        | None
    ):
        """The type of sequences in the dataset."""
        return self._seqs_kind

    def __len__(self):
        return self.n_regions * self.n_samples

    def __str__(self) -> str:
        splice_status = "Spliced" if self.is_spliced else "Unspliced"

        if self._available_sequences is None:
            seq_type = None
        else:
            seqs = self._available_sequences
            if self.sequence_type is not None:
                seqs[seqs.index(self.sequence_type)] = f"[{self.sequence_type}]"
            seq_type = " ".join(seqs)

        if self.available_tracks is None:
            tracks = None
        else:
            tracks = f"{', '.join(self.available_tracks[:5])}"
            if len(self.available_tracks) > 5:
                tracks += f" + {len(self.available_tracks) - 5} more"

        if self.active_tracks is None:
            act_tracks = None
        else:
            act_tracks = f"{', '.join(self.active_tracks[:5])}"
            if len(self.active_tracks) > 5:
                act_tracks += f" + {len(self.active_tracks) - 5} more"
        return (
            splice_status + f" GVL dataset at {self.path}\n"
            f"Is subset: {self.is_subset}\n"
            f"# of regions: {self.n_regions}\n"
            f"# of samples: {self.n_samples}\n"
            f"Output length: {self.output_length}\n"
            f"Jitter: {self.jitter} (max: {self.max_jitter})\n"
            f"Deterministic: {self.deterministic}\n"
            f"Sequence type: {seq_type}\n"
            f"Active tracks: {act_tracks}\n"
            f"Tracks available: {tracks}\n"
        )

    def __repr__(self) -> str:
        return str(self)

    def subset_to(
        self,
        regions: StrIdx | None = None,
        samples: StrIdx | None = None,
    ) -> Self:
        """Subset the dataset to specific regions and/or samples by index or a boolean mask.

        If regions or samples are not provided, the corresponding dimension will not be subset.

        Args:
            regions: The regions to subset to.
            samples: The samples to subset to.

        Examples:
            Subsetting to the first 10 regions:

            .. code-block:: python

                ds.subset_to(slice(10))

            Subsetting to the 2nd and 4th samples:

            .. code-block:: python

                ds.subset_to(samples=[1, 3])


            Subsetting to chromosome 1, assuming it's labeled :code:`"chr1"`:

            .. code-block:: python

                r_idx = ds.regions["chrom"] == "chr1"
                ds.subset_to(regions=r_idx)


            Subsetting to regions labeled by a column "split", assuming "split" existed in the input regions:

            .. code-block:: python

                r_idx = ds.regions["split"] == "train"
                ds.subset_to(regions=r_idx)


            Subsetting to the intersection with another set of regions:

            .. code-block:: python

                import seqpro as sp

                regions = gvl.read_bedlike("regions.bed")
                regions_pr = sp.bed.to_pyr(regions)
                ds_regions_pr = sp.bed.to_pyr(ds.regions.with_row_index())
                r_idx = ds_regions_pr.overlap(regions_pr).df["index"].to_numpy()
                ds.subset_to(regions=r_idx)
        """
        if regions is None and samples is None:
            return self

        if is_str_arr(regions) and self._idxer.r2i_map is None:
            raise ValueError(
                "Cannot subset to regions by name because no region name was set."
            )

        if self._sp_idxer is None:
            idxer = self._idxer.subset_to(regions=regions, samples=samples)
            return replace(self, _idxer=idxer)
        else:
            sp_idxer, sub_dsi = self._sp_idxer.subset_to(rows=regions, samples=samples)
            return replace(self, _idxer=sub_dsi, _sp_idxer=sp_idxer)

    def to_full_dataset(self) -> Self:
        """Return a full sized dataset, undoing any subsetting."""
        if self._sp_idxer is None:
            return replace(self, _idxer=self._idxer.to_full_dataset())
        else:
            return replace(
                self,
                _idxer=self._idxer.to_full_dataset(),
                _sp_idxer=self._sp_idxer.to_full_dataset(),
            )

    def haplotype_lengths(
        self,
        regions: Idx | None = None,
        samples: Idx | str | Sequence[str] | None = None,
    ) -> NDArray[np.int32] | None:
        """The lengths of jitter-extended haplotypes for specified regions and samples.

        If the dataset is not phased or not deterministic, this will return :code:`None` because the haplotypes are not guaranteed to be
        a consistent length due to randomness in what variants are used.

        .. note::

            Currently not implemented for spliced datasets.

        Args:
            regions: Regions to compute haplotype lengths for.
            samples: Samples to compute haplotype lengths for.
        """
        if self._sp_idxer is not None:
            raise NotImplementedError(
                "Haplotype lengths are not yet implemented for spliced datasets."
            )

        if not isinstance(self._seqs, Haps) or not self.deterministic:
            return None

        if regions is None:
            regions = slice(None)
        if samples is None:
            samples = slice(None)
        idx = (regions, samples)

        ds_idx, squeeze, out_reshape = self._idxer.parse_idx(idx)

        r_idx, _ = np.unravel_index(ds_idx, self.full_shape)

        # (b)
        regions = self._full_regions[r_idx]
        regions[:, 1] -= self.jitter
        regions[:, 2] += self.jitter

        # (b p)
        hap_lens = (
            regions[:, 2, None]  # (b 1)
            - regions[:, 1, None]  # (b 1)
            + self._seqs._haplotype_ilens(ds_idx, regions, self.deterministic)  # (b p)
        )

        if squeeze:
            hap_lens = hap_lens.squeeze(0)

        if out_reshape is not None:
            hap_lens = hap_lens.reshape(
                *out_reshape,
                self._seqs.genotypes.shape[-2],
            )

        return hap_lens

    def n_variants(
        self,
        regions: Idx | None = None,
        samples: StrIdx | None = None,
    ) -> NDArray[np.int32]:
        """The number of variants in the dataset for specified regions and samples.

        Args:
            regions: Regions to compute the number of variants for.
            samples: Samples to compute the number of variants for.

        Returns:
            Array with shape (..., ploidy). The number of variants in the dataset for the specified regions and samples.
            If the dataset does not have genotypes, this will return :code:`None`.
        """
        if regions is None:
            regions = slice(None)
        if samples is None:
            samples = slice(None)
        idx = (regions, samples)

        ds_idx, squeeze, out_reshape = self._idxer.parse_idx(idx)

        r_idx, s_idx = np.unravel_index(ds_idx, self.full_shape)

        if not isinstance(self._seqs, Haps):
            n_vars = np.zeros((len(r_idx), len(s_idx), 1), dtype=np.int32)
        else:
            # ((...), P)
            n_vars = self._seqs.n_variants[r_idx, s_idx]
            if self._seqs.unphased_union:
                # Fold the ploidy axis: union count per (region, sample) is the
                # naive sum of per-haplotype counts (no dedup). ((...), 1)
                # Keep int32 to match the method's return contract (sum() upcasts).
                n_vars = n_vars.sum(-1, keepdims=True, dtype=np.int32)

        if squeeze:
            # (1, P) -> (P)
            n_vars = n_vars.squeeze(0)

        if out_reshape is not None:
            # ((...), P) -> (..., P)
            n_vars = n_vars.reshape(*out_reshape, n_vars.shape[-1])

        return n_vars

    def _output_bytes_per_instance(
        self,
        regions: Idx | None = None,
        samples: Idx | str | Sequence[str] | None = None,
        *,
        include_offsets: bool = False,
    ) -> NDArray[np.int64]:
        """Exact bytes one (region, sample) instance materializes to under the current schema.

        Shape: (n_instances,) of int64.

        Args:
            regions: Regions to compute output bytes for.
            samples: Samples to compute output bytes for.
            include_offsets: If ``False`` (default), return the *payload* bytes — the
                ``numpy.nbytes`` of the materialized output. If ``True``, add the
                per-instance share of the int64 offset/lengths arrays that the
                shared-memory chunk serialization writes alongside the payload
                (see ``_shm_layout.write_chunk``): ``8 * ploidy`` per ragged
                output array (outer offsets) and, for ``variants`` ``alt``/``ref``
                fields, ``8 * n_variants`` (inner allele offsets). This is the
                footprint that must fit in a ``double_buffered`` slot; payload
                alone undersizes the slot for ragged outputs. The per-chunk
                ``+1`` offset terminators and 8-byte alignment padding are not
                included here — they are absorbed by the slot's fixed slack.

        Raises:
            NotImplementedError: For spliced datasets.
            ValueError: For non-deterministic datasets when with_seqs is in
                {"haplotypes", "annotated"}.
        """
        if self._sp_idxer is not None:
            raise NotImplementedError(
                "_output_bytes_per_instance is not implemented for spliced datasets."
            )

        if regions is None:
            regions = slice(None)
        if samples is None:
            samples = slice(None)
        idx = (regions, samples)
        ds_idx, squeeze, out_reshape = self._idxer.parse_idx(idx)
        r_idx, _s_idx = np.unravel_index(ds_idx, self.full_shape)

        seq_kind = (
            self.sequence_type
        )  # "reference" | "haplotypes" | "annotated" | "variants" | None
        total = np.zeros(len(r_idx), dtype=np.int64)
        # Per-instance share of int64 offset/lengths arrays (filled when
        # include_offsets); added to `total` just before the final reshape.
        offset_total = np.zeros(len(r_idx), dtype=np.int64)
        OFF = 8  # int64 offset entry
        # Stored ploidy: this is the on-disk value (2 under unphased_union, not the
        # folded 1). Only consumed by the haplotypes/annotated/tracks branches, which
        # are unreachable under the flag; the "variants" branch re-derives the folded
        # ploidy from n_variants(...) below. Don't reuse this for a variants-branch path.
        ploidy = self._seqs.n_variants.shape[-1] if isinstance(self._seqs, Haps) else 1
        # These are computed conditionally below; declared here to satisfy the type checker.
        hap_len_sum: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        region_lens: NDArray[np.int64] = np.empty(0, dtype=np.int64)

        # --- seqs payload ---
        if seq_kind == "reference":
            # region length × 1 byte/nt (S1), no ploidy expansion.
            regions_arr = self._full_regions[r_idx].copy()
            regions_arr[:, 1] -= self.jitter
            regions_arr[:, 2] += self.jitter
            region_lens = (regions_arr[:, 2] - regions_arr[:, 1]).astype(np.int64)
            total += region_lens
            if include_offsets and self.output_length == "ragged":
                # ragged reference: 1 outer-offset entry per instance (no ploidy).
                offset_total += OFF
        elif seq_kind in ("haplotypes", "annotated"):
            if not self.deterministic:
                raise ValueError(
                    f"with_seqs={seq_kind!r} requires deterministic=True for "
                    "_output_bytes_per_instance. Use dataset.with_settings(deterministic=True)."
                )
            hap_lens = self.haplotype_lengths(regions, samples)
            if hap_lens is None:
                raise ValueError(
                    f"with_seqs={seq_kind!r} requires haplotype_lengths() to be available."
                )
            # hap_lens shape: (..., ploidy). Flatten to (n_inst, ploidy).
            hap_lens_flat = hap_lens.reshape(-1, hap_lens.shape[-1]).astype(np.int64)
            hap_len_sum = hap_lens_flat.sum(-1)  # sum over ploidy
            total += hap_len_sum  # haps S1: 1 byte/nt
            if seq_kind == "annotated":
                # annotated: var_idxs and ref_coords are per-position (same
                # length as haps), not per-variant. Both are int32 (4 bytes).
                total += hap_len_sum * 4  # var_idxs int32
                total += hap_len_sum * 4  # ref_coords int32
            if include_offsets:
                # Each ragged array's outer offsets carry `ploidy` entries per
                # instance: 1 array for haplotypes, 3 (haps/var_idxs/ref_coords)
                # for annotated.
                n_seq_arrays = 1 if seq_kind == "haplotypes" else 3
                offset_total += OFF * ploidy * n_seq_arrays
        elif seq_kind == "variants":
            if not isinstance(self._seqs, Haps):
                raise AssertionError("variants mode requires Haps")
            haps_obj = self._seqs
            var_fields = haps_obj.var_fields
            n_vars = self.n_variants(regions, samples)  # (n_inst, ploidy)
            n_vars_flat = n_vars.reshape(-1, n_vars.shape[-1]).astype(np.int64)
            n_vars_total = n_vars_flat.sum(-1)  # over ploidy → (n_inst,)
            ploidy = n_vars.shape[-1]

            for f in var_fields:
                if f == "start":
                    total += n_vars_total * haps_obj.variants.start.dtype.itemsize
                elif f == "ilen":
                    total += n_vars_total * haps_obj.variants.ilen.dtype.itemsize
                elif f == "dosage":
                    if haps_obj.dosages is None:
                        continue
                    dosage_dtype = haps_obj.dosages.data.dtype
                    total += n_vars_total * dosage_dtype.itemsize
                elif f in ("alt", "ref"):
                    # Allele scan: _allele_bytes_sum returns (len(ds_idx) * ploidy,).
                    per_ploid = haps_obj._allele_bytes_sum(ds_idx, f)
                    total += per_ploid.reshape(-1, ploidy).sum(-1)
                else:
                    # INFO column: numeric, known dtype from on-disk schema.
                    # Svar2Haps.variants is a dummy placeholder (info={}) --
                    # store fields' dtypes live in the store manifest instead.
                    from ._svar2_haps import Svar2Haps

                    if isinstance(haps_obj, Svar2Haps) and f in haps_obj.store_fields:
                        info_dtype = haps_obj.store_fields[f].dtype
                    else:
                        info_dtype = haps_obj.variants.info[f].dtype
                    total += n_vars_total * info_dtype.itemsize
            if include_offsets:
                # RaggedVariants (kind=2) writes, per field: outer offsets
                # (ploidy entries/instance) and, for alt/ref allele fields,
                # inner offsets (one entry per variant → n_vars_total/instance).
                n_allele_fields = sum(1 for f in var_fields if f in ("alt", "ref"))
                offset_total += OFF * ploidy * len(var_fields)
                offset_total += OFF * n_vars_total * n_allele_fields
        elif seq_kind is None:
            pass
        else:
            raise AssertionError(f"unknown sequence_type {seq_kind!r}")

        # --- tracks payload ---
        if self.active_tracks:
            n_tracks = len(self.active_tracks)
            track_itemsize = np.dtype(np.float32).itemsize  # tracks are always float32
            if seq_kind in ("haplotypes", "annotated"):
                # Tracks have shape (b, t, p, ~l): length = haplotype length per ploid.
                # hap_len_sum already sums over ploidy → total per instance = hap_len_sum * n_tracks.
                total += hap_len_sum * n_tracks * track_itemsize
            else:
                # reference, variants, or no-seq: tracks have shape (b, t, ~l), length = region length.
                # "reference" already computed region_lens above; others need to compute it now.
                if seq_kind != "reference":
                    regions_arr = self._full_regions[r_idx].copy()
                    regions_arr[:, 1] -= self.jitter
                    regions_arr[:, 2] += self.jitter
                    region_lens = (regions_arr[:, 2] - regions_arr[:, 1]).astype(
                        np.int64
                    )
                total += region_lens * n_tracks * track_itemsize
            if include_offsets:
                # Each track is its own ragged array. Grouped by (instance ×
                # ploidy) for haplotype-shaped outputs, else by instance.
                if seq_kind in ("haplotypes", "annotated"):
                    offset_total += OFF * ploidy * n_tracks
                else:
                    offset_total += OFF * n_tracks

        if include_offsets:
            total += offset_total

        if squeeze:
            return total
        if out_reshape is not None:
            return total.reshape(out_reshape)
        return total

    def n_intervals(
        self,
        regions: Idx | None = None,
        samples: StrIdx | None = None,
    ) -> NDArray[np.int32]:
        """The number of intervals in the dataset for specified regions and samples.

        Args:
            regions: Regions to compute the number of intervals for.
            samples: Samples to compute the number of intervals for.

        Returns:
            Array with shape (..., tracks). The number of intervals in the dataset for the specified regions and samples.
            If the dataset does not have intervals, this will return :code:`None`.
        """
        if regions is None:
            regions = slice(None)
        if samples is None:
            samples = slice(None)
        idx = (regions, samples)

        ds_idx, squeeze, out_reshape = self._idxer.parse_idx(idx)

        r_idx, s_idx = np.unravel_index(ds_idx, self.full_shape)

        if self._tracks is None:
            n_itvs = np.zeros((len(r_idx), len(s_idx)), dtype=np.int32)
        else:
            ls = []
            for name, kind in self._tracks.active_tracks.items():
                if kind is TrackType.SAMPLE:
                    ls.append(self._tracks.intervals[name].values.lengths[r_idx, s_idx])
                elif kind is TrackType.ANNOT:
                    ls.append(self._tracks.intervals[name].values.lengths[r_idx])
                else:
                    assert_never(kind)
            n_itvs = np.stack(ls, axis=-1)

        if squeeze:
            # (1, P) -> (P)
            n_itvs = n_itvs.squeeze(0)

        if out_reshape is not None:
            # ((...), P) -> (..., P)
            n_itvs = n_itvs.reshape(*out_reshape, n_itvs.shape[-1])

        return n_itvs

    def write_transformed_track(
        self,
        new_track: str,
        existing_track: str,
        transform: Callable[
            [NDArray[np.intp], NDArray[np.intp], Ragged[np.float32]],
            Ragged[np.float32],
        ],
        max_mem: int = 2**30,
        overwrite: bool = False,
    ) -> ArrayDataset | RaggedDataset:
        """Write transformed tracks to the dataset.

        Args:
            new_track: The name of the new track.
            existing_track: The name of the existing track to transform.
            transform: A function to apply to the existing track to get a new, transformed track.
                This will be done in chunks such that the tracks provided will not exceed :code:`max_mem`.
                The arguments given to the transform will be the region and sample indices as numpy arrays
                and the tracks themselves as a :class:`Ragged` array with
                shape (regions, samples). The tracks must be a :class:`Ragged` array since regions may be
                different lengths to accommodate indels. This function should then return the transformed
                tracks as a :class:`Ragged` array with the same shape and lengths.
            max_mem: The maximum memory to use in bytes, by default 1 GiB (2**30 bytes)
            overwrite: Whether to overwrite the existing track, by default False
        """
        if self._tracks is None:
            raise ValueError("Dataset has no tracks to transform.")

        new_tracks = self._tracks.write_transformed_track(
            new_track,
            existing_track,
            transform,
            self.path,
            self._full_regions,
            self.max_jitter,
            self._idxer,
            self._seqs if isinstance(self._seqs, Haps) else None,
            max_mem=max_mem,
            overwrite=overwrite,
        )

        return replace(self, _tracks=new_tracks)  # type: ignore[bad-return]  # dataclasses.replace returns Self but pyrefly widens to base Dataset union

    def to_torch_dataset(
        self, return_indices: bool, transform: Callable | None
    ) -> TorchDataset:
        """Convert the dataset to a PyTorch :class:`Dataset <torch.utils.data.Dataset>`.

        Requires PyTorch to be installed.

        Args:
            return_indices: Whether to append arrays of row and sample indices of the non-subset dataset to each batch.
            transform: The transform to apply to each batch of data. The transform should take input matching the output of the dataset and can
                return anything that can be converted to a PyTorch tensor. In combination with indices, this allows you to combine arbitrary
                row- and sample-specific data with dataset output on-the-fly.

                .. note::
                    Depending on how transforms are implemented, they can easily introduce a dataloading bottleneck. If you find
                    dataloading is slow, it's often a good idea to try disabling your transform to see if it's impacting throughput.
        """
        if self.output_length == "ragged":
            logger.warning(
                '`output_length` is currently set to "ragged" and ragged output cannot be converted to PyTorch Tensors.'
            )
        return TorchDataset(self, return_indices, transform)

    def to_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: td.Sampler | Iterable | None = None,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable | None = None,
        multiprocessing_context: Callable | None = None,
        generator: torch.Generator | None = None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        return_indices: bool = False,
        transform: Callable | None = None,
        mode: Literal["buffered", "double_buffered"] | None = None,
        buffer_bytes: int = 2 * 1024**3,
        copy: bool = True,
        heartbeat_seconds: float = 60.0,
    ) -> td.DataLoader:
        """Convert the dataset to a PyTorch :class:`DataLoader <torch.utils.data.DataLoader>`.

        The parameters are the same as a :class:`DataLoader <torch.utils.data.DataLoader>`
        with a few omissions e.g. :code:`batch_sampler`. Requires PyTorch to be installed.

        Args:
            batch_size: How many samples per batch to load.
            shuffle: Set to True to have the data reshuffled at every epoch.
            sampler: Defines the strategy to draw samples from the dataset. Can be any :py:class:`Iterable <typing.Iterable>` with :code:`__len__` implemented. If specified, shuffle must not be specified.

                .. important::
                    Do not provide a :class:`BatchSampler <torch.utils.data.BatchSampler>` here. GVL Datasets use multithreading when indexed with batches of indices to avoid the overhead of multi-processing.
                    To leverage this, GVL will automatically wrap the :code:`sampler` with a :class:`BatchSampler <torch.utils.data.BatchSampler>`
                    so that lists of indices are given to the GVL Dataset instead of one index at a time. See `this post <https://discuss.pytorch.org/t/dataloader-sample-by-slices-from-dataset/113005>`_
                    for more information.
            num_workers: How many subprocesses to use for dataloading. :code:`0` means that the data will be loaded in the main process.

                .. tip::
                    For GenVarLoader, it is generally best to set this to 0 or 1 since almost everything in
                    GVL is multithreaded. However, if you are using a transform that is compute intensive and single threaded, there may
                    be a benefit to setting this > 1.
            collate_fn: Merges a list of samples to form a mini-batch of Tensor(s).
            pin_memory: If :code:`True`, the data loader will copy Tensors into device/CUDA pinned memory before returning them. If your data elements are a custom type, or your :code:`collate_fn` returns a batch that is a custom type, see the example below.
            drop_last: Set to :code:`True` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If :code:`False` and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
            timeout: If positive, the timeout value for collecting a batch from workers. Should always be non-negative.
            worker_init_fn: If not :code:`None`, this will be called on each worker subprocess with the worker id (an int in :code:`[0, num_workers - 1]`) as input, after seeding and before data loading.
            multiprocessing_context: If :code:`None`, the default multiprocessing context of your operating system will be used.
            generator: If not :code:`None`, this RNG will be used by RandomSampler to generate random indexes and multiprocessing to generate :code:`base_seed` for workers.
            prefetch_factor: Number of batches loaded in advance by each worker. 2 means there will be a total of 2 * num_workers batches prefetched across all workers. (default value depends on the set value for num_workers. If value of num_workers=0 default is None. Otherwise, if value of num_workers > 0 default is 2).
            persistent_workers: If :code:`True`, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive.
            pin_memory_device: The device to :code:`pin_memory` to if :code:`pin_memory` is :code:`True`.
            return_indices: Whether to append arrays of row and sample indices of the non-subset dataset to each batch.
            transform: The transform to apply to each batch of data. The transform should take input matching the output of the dataset and can
                return anything that can be converted to a PyTorch tensor. In combination with indices, this allows you to combine arbitrary
                row- and sample-specific data with dataset output on-the-fly.

                .. note::
                    Depending on how transforms are implemented, they can easily introduce a dataloading bottleneck. If you find
                    dataloading is slow, it's often a good idea to try disabling your transform to see if it's impacting throughput.
            mode: Dataloading strategy. :code:`None` (default) uses the standard PyTorch :class:`DataLoader <torch.utils.data.DataLoader>`
                over a map-style dataset. :code:`"buffered"` and :code:`"double_buffered"` use a prefetching producer that fills an
                in-memory buffer ahead of consumption to hide read latency; both are incompatible with :code:`num_workers > 0` since
                the loader is itself the concurrency strategy. :code:`"double_buffered"` serializes chunks into two fixed-size
                shared-memory slots, allowing a producer thread to fill one slot while the consumer drains the other.
            buffer_bytes: Total byte budget for the prefetch buffer when :code:`mode` is :code:`"buffered"` or :code:`"double_buffered"`.
                For :code:`"double_buffered"` this is split across two shared-memory slots (:code:`buffer_bytes / 2` each). Defaults to 2 GiB.
            copy: Only used when :code:`mode="double_buffered"`. If :code:`True` (default), each batch owns its data. If :code:`False`,
                batches are zero-copy views into shared memory and are only valid until the next batch is yielded.
            heartbeat_seconds: Only used when :code:`mode="double_buffered"`. Seconds to wait per slot before checking that the producer is still alive.
        """
        if mode is not None:
            # Buffered modes operate directly on the Dataset, not on a TorchDataset wrapper,
            # because they need access to _output_bytes_per_instance and raw indexing.
            return get_dataloader(
                dataset=self,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=drop_last,
                generator=generator,
                pin_memory_device=pin_memory_device,
                mode=mode,
                buffer_bytes=buffer_bytes,
                copy=copy,
                heartbeat_seconds=heartbeat_seconds,
            )
        return get_dataloader(
            dataset=self.to_torch_dataset(return_indices, transform),
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    def __getitem__(
        self, idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx]
    ) -> (
        Ragged[np.bytes_ | np.float32]
        | RaggedAnnotatedHaps
        | RaggedVariants
        | RaggedIntervals
        | NDArray[np.bytes_ | np.float32]
        | AnnotatedHaps
        | tuple[
            Ragged[np.bytes_ | np.float32]
            | RaggedAnnotatedHaps
            | RaggedVariants
            | RaggedIntervals
            | NDArray[np.bytes_ | np.float32]
            | AnnotatedHaps,
            ...,
        ]
    ):
        # Thin facade: package state into a QueryView and hand off to the
        # query module's free functions, which carry the actual logic.
        if (
            isinstance(self._seqs, Haps)
            and self._seqs.dummy_variant is not None
            and self._seqs_kind not in ("variants", "variant-windows")
        ):
            raise ValueError(
                "dummy_variant is only valid for the 'variants' and "
                "'variant-windows' outputs; call with_seqs('variants') or "
                "with_seqs('variant-windows') (got output kind "
                f"{self._seqs_kind!r})."
            )

        from ._query import QueryView, getitem

        view = QueryView(
            idxer=self._idxer,
            sp_idxer=self._sp_idxer,
            full_regions=self._full_regions,
            rng=self._rng,
            recon=self._recon,
            output_length=self.output_length,
            jitter=self.jitter,
            deterministic=self.deterministic,
            rc_neg=self.rc_neg,
            flat_output=self.output_format == "flat",
        )
        return getitem(view, idx)


def _lazy_load_dosages(dataset: Dataset, haps: Haps) -> Haps:
    """Open the dosages memmap for a Haps that didn't request them at open time.

    Reuses the same path-resolution logic that ``Haps.from_path`` used. Returns
    a new ``Haps`` with ``dosages`` populated (does NOT mutate the input).
    """
    import json as _json

    from genoray._types import DOSAGE_TYPE

    from ._svar_link import _resolve_svar
    from ._write import Metadata

    path = haps.path
    svar_meta_path = path / "genotypes" / "svar_meta.json"
    if not svar_meta_path.exists():
        raise ValueError(
            "Dosage requested but this dataset is not SVAR-backed; no dosages.npy possible."
        )

    with open(svar_meta_path) as f:
        svar_meta = _json.load(f)
    shape = tuple(svar_meta["shape"])
    dtype = np.dtype(svar_meta["dtype"])

    offset_path = path / "genotypes" / "offsets.npy"

    # Resolve the SVAR directory the same way Haps.from_path did. Dataset does
    # not retain Metadata, so re-read metadata.json from disk.
    meta = Metadata.model_validate_json((path / "metadata.json").read_text())
    svar_link = meta.svar_link
    if svar_link is not None:
        svar_path = _resolve_svar(path, svar_link, None)
    else:
        legacy_link = path / "genotypes" / "link.svar"
        svar_path = legacy_link.resolve()

    dosage_path = svar_path / "dosages.npy"
    if not dosage_path.exists():
        raise ValueError(
            f"Dosage requested but {dosage_path} does not exist. "
            f"Check the SVAR was built with dosages."
        )

    offsets = np.memmap(offset_path, shape=shape, dtype=dtype, mode="r")
    dosages_mm = np.memmap(dosage_path, dtype=DOSAGE_TYPE, mode="r")
    rag_shape = (*shape[1:], None)
    dosages = Ragged.from_offsets(dosages_mm, rag_shape, offsets.reshape(2, -1))
    return replace(haps, dosages=dosages)


def _lazy_load_custom_fields(
    dataset: Dataset,
    haps: Haps,
    new_fields: dict[str, np.dtype],
) -> Haps:
    """Memmap custom FORMAT fields (Number=G, stored as <name>.npy) into ``haps.var_field_data`` for fields that were not loaded at open time.

    ``new_fields`` maps field name → numpy dtype (already confirmed present in
    the SVAR metadata). Returns a new ``Haps`` with updated ``var_field_data``.
    """
    import json as _json

    path = haps.path
    svar_meta_path = path / "genotypes" / "svar_meta.json"
    if not svar_meta_path.exists():
        raise ValueError(
            "Custom FORMAT fields requested but this dataset is not SVAR-backed."
        )

    with open(svar_meta_path) as f:
        svar_meta = _json.load(f)
    shape = tuple(svar_meta["shape"])
    dtype = np.dtype(svar_meta["dtype"])

    offset_path = path / "genotypes" / "offsets.npy"

    # The resolved SVAR directory is already embedded in haps.variants.path
    # (which was set to <svar_path>/index.arrow by Haps.from_path, respecting any
    # svar_override). Using .parent avoids re-resolving from metadata and correctly
    # handles the svar_override case that the legacy link.svar branch would miss.
    svar_path = haps.variants.path.parent

    offsets = np.memmap(offset_path, shape=shape, dtype=dtype, mode="r")
    rag_shape = (*shape[1:], None)

    updated_var_field_data = dict(haps.var_field_data)
    for name, ftype in new_fields.items():
        field_path = svar_path / f"{name}.npy"
        if not field_path.exists():
            raise ValueError(
                f"Custom FORMAT field '{name}' registered in SVAR metadata but "
                f"{field_path} does not exist."
            )
        field_mm = np.memmap(field_path, dtype=ftype, mode="r")
        updated_var_field_data[name] = Ragged.from_offsets(
            field_mm, rag_shape, offsets.reshape(2, -1)
        )
    return replace(haps, var_field_data=updated_var_field_data)


SEQ = TypeVar("SEQ", NDArray[np.bytes_], AnnotatedHaps, RaggedVariants)
MaybeSEQ = TypeVar("MaybeSEQ", None, NDArray[np.bytes_], AnnotatedHaps, RaggedVariants)
TRK = TypeVar("TRK", NDArray[np.float32], RaggedIntervals)
MaybeTRK = TypeVar("MaybeTRK", None, NDArray[np.float32], RaggedIntervals)

RSEQ = TypeVar("RSEQ", RaggedSeqs, RaggedAnnotatedHaps, RaggedVariants)
MaybeRSEQ = TypeVar("MaybeRSEQ", None, RaggedSeqs, RaggedAnnotatedHaps, RaggedVariants)
RTRK = TypeVar("RTRK", Ragged[np.float32], RaggedIntervals)
MaybeRTRK = TypeVar("MaybeRTRK", None, Ragged[np.float32], RaggedIntervals)


class ArrayDataset(Dataset, Generic[MaybeSEQ, MaybeTRK]):
    """Only for type checking purposes, you should never instantiate this class directly."""

    output_length: Literal["variable"] | int

    @overload
    def with_len(
        self: ArrayDataset[NDArray[np.bytes_], None],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[RaggedSeqs, None]: ...
    @overload
    def with_len(
        self: ArrayDataset[AnnotatedHaps, None],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[RaggedAnnotatedHaps, None]: ...
    @overload
    def with_len(
        self: ArrayDataset[None, NDArray[np.float32]],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[None, Ragged[np.float32]]: ...
    @overload
    def with_len(
        self: ArrayDataset[NDArray[np.bytes_], NDArray[np.float32]],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[RaggedSeqs, Ragged[np.float32]]: ...
    @overload
    def with_len(
        self: ArrayDataset[AnnotatedHaps, NDArray[np.float32]],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[RaggedAnnotatedHaps, Ragged[np.float32]]: ...
    @overload
    def with_len(
        self,
        output_length: Literal["variable"] | int,
    ) -> ArrayDataset[NDArray[np.bytes_], MaybeTRK]: ...
    def with_len(
        self, output_length: Literal["ragged", "variable"] | int
    ) -> RaggedDataset[MaybeRSEQ, MaybeRTRK] | ArrayDataset[SEQ, MaybeTRK]:
        return super().with_len(output_length)

    @overload
    def with_seqs(self, kind: None) -> ArrayDataset[None, MaybeTRK]: ...
    @overload
    def with_seqs(
        self, kind: Literal["reference", "haplotypes"]
    ) -> ArrayDataset[NDArray[np.bytes_], MaybeTRK]: ...
    @overload
    def with_seqs(
        self, kind: Literal["annotated"]
    ) -> ArrayDataset[AnnotatedHaps, MaybeTRK]: ...
    @overload
    def with_seqs(
        self, kind: Literal["variants"]
    ) -> ArrayDataset[RaggedVariants, MaybeTRK]: ...
    def with_seqs(
        self,
        kind: Literal[
            "reference", "haplotypes", "annotated", "variants", "variant-windows"
        ]
        | None,
        window_opt: "VarWindowOpt | None" = None,
    ) -> ArrayDataset:
        return super().with_seqs(kind, window_opt)

    @overload
    def with_tracks(self, tracks: None = None, kind: None = None) -> Self: ...
    @overload
    def with_tracks(
        self, *, tracks: None = None, kind: Literal["tracks"]
    ) -> ArrayDataset[MaybeSEQ, NDArray[np.float32]]: ...
    @overload
    def with_tracks(
        self, *, tracks: None = None, kind: Literal["intervals"]
    ) -> ArrayDataset[MaybeSEQ, RaggedIntervals]: ...
    @overload
    def with_tracks(
        self,
        tracks: Literal[False],
        kind: Literal["tracks", "intervals"] | None = None,
    ) -> ArrayDataset[MaybeSEQ, None]: ...
    @overload
    def with_tracks(
        self, tracks: str | list[str], kind: None = None
    ) -> ArrayDataset[MaybeSEQ, MaybeTRK]: ...
    @overload
    def with_tracks(
        self, tracks: str | list[str], kind: Literal["tracks"]
    ) -> ArrayDataset[MaybeSEQ, NDArray[np.float32]]: ...
    @overload
    def with_tracks(
        self, tracks: str | list[str], kind: Literal["intervals"]
    ) -> ArrayDataset[MaybeSEQ, RaggedIntervals]: ...
    def with_tracks(
        self,
        tracks: str | list[str] | Literal[False] | None = None,
        kind: Literal["tracks", "intervals"] | None = None,
    ) -> ArrayDataset:
        return super().with_tracks(tracks, kind)

    @overload
    def __getitem__(
        self: ArrayDataset[SEQ, None],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> SEQ: ...
    @overload
    def __getitem__(
        self: ArrayDataset[None, NDArray[np.float32]],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> NDArray[np.float32]: ...
    @overload
    def __getitem__(
        self: ArrayDataset[SEQ, NDArray[np.float32]],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> tuple[SEQ, NDArray[np.float32]]: ...
    @overload
    def __getitem__(
        self: ArrayDataset[None, RaggedIntervals],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> RaggedIntervals: ...
    @overload
    def __getitem__(
        self: ArrayDataset[SEQ, RaggedIntervals],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> tuple[SEQ, RaggedIntervals]: ...
    @overload
    def __getitem__(
        self: ArrayDataset[SEQ, MaybeTRK],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> SEQ | tuple[SEQ, NDArray[np.float32]]: ...
    @overload
    def __getitem__(
        self: ArrayDataset[MaybeSEQ, NDArray[np.float32]],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> NDArray[np.float32] | tuple[SEQ, NDArray[np.float32]]: ...
    @overload
    def __getitem__(
        self: ArrayDataset[MaybeSEQ, RaggedIntervals],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> RaggedIntervals | tuple[SEQ, RaggedIntervals]: ...
    @overload
    def __getitem__(
        self: ArrayDataset[MaybeSEQ, MaybeTRK],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> SEQ | NDArray[np.float32] | tuple[SEQ, NDArray[np.float32]]: ...
    def __getitem__(
        self, idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx]
    ) -> SEQ | TRK | tuple[SEQ, TRK]:
        return super().__getitem__(idx)  # type: ignore[bad-return]  # base Dataset returns broad union; SEQ/TRK typevars narrow at use sites


class RaggedDataset(Dataset, Generic[MaybeRSEQ, MaybeRTRK]):
    """Only for type checking purposes, you should never instantiate this class directly."""

    output_length: Literal["ragged"]

    @overload
    def with_len(
        self: RaggedDataset[RaggedSeqs, None],
        output_length: Literal["variable"] | int,
    ) -> ArrayDataset[NDArray[np.bytes_], None]: ...
    @overload
    def with_len(
        self: RaggedDataset[RaggedAnnotatedHaps, None],
        output_length: Literal["variable"] | int,
    ) -> ArrayDataset[AnnotatedHaps, None]: ...
    @overload
    def with_len(
        self: RaggedDataset[None, Ragged[np.float32]],
        output_length: Literal["variable"] | int,
    ) -> ArrayDataset[None, NDArray[np.float32]]: ...
    @overload
    def with_len(
        self: RaggedDataset[None, MaybeRTRK],
        output_length: Literal["variable"] | int,
    ) -> ArrayDataset[None, MaybeTRK]: ...
    @overload
    def with_len(
        self: RaggedDataset[RaggedSeqs, Ragged[np.float32]],
        output_length: Literal["variable"] | int,
    ) -> ArrayDataset[NDArray[np.bytes_], NDArray[np.float32]]: ...
    @overload
    def with_len(
        self: RaggedDataset[RaggedAnnotatedHaps, Ragged[np.float32]],
        output_length: Literal["variable"] | int,
    ) -> ArrayDataset[AnnotatedHaps, NDArray[np.float32]]: ...
    @overload
    def with_len(
        self,
        output_length: Literal["ragged"],
    ) -> RaggedDataset[MaybeRSEQ, MaybeRTRK]: ...
    def with_len(
        self, output_length: Literal["ragged", "variable"] | int
    ) -> RaggedDataset[MaybeRSEQ, MaybeRTRK] | ArrayDataset[MaybeSEQ, MaybeTRK]:
        return super().with_len(output_length)

    @overload
    def with_seqs(self, kind: None) -> RaggedDataset[None, MaybeRTRK]: ...
    @overload
    def with_seqs(
        self, kind: Literal["reference", "haplotypes"]
    ) -> RaggedDataset[RaggedSeqs, MaybeRTRK]: ...
    @overload
    def with_seqs(
        self, kind: Literal["annotated"]
    ) -> RaggedDataset[RaggedAnnotatedHaps, MaybeRTRK]: ...
    @overload
    def with_seqs(
        self, kind: Literal["variants"]
    ) -> RaggedDataset[RaggedVariants, MaybeRTRK]: ...
    def with_seqs(
        self,
        kind: Literal[
            "reference", "haplotypes", "annotated", "variants", "variant-windows"
        ]
        | None,
        window_opt: "VarWindowOpt | None" = None,
    ) -> RaggedDataset:
        return super().with_seqs(kind, window_opt)

    @overload
    def with_tracks(self, tracks: None = None, kind: None = None) -> Self: ...
    @overload
    def with_tracks(
        self, *, tracks: None = None, kind: Literal["tracks"]
    ) -> RaggedDataset[MaybeRSEQ, RaggedTracks]: ...
    @overload
    def with_tracks(
        self, *, tracks: None = None, kind: Literal["intervals"]
    ) -> RaggedDataset[MaybeRSEQ, RaggedIntervals]: ...
    @overload
    def with_tracks(
        self,
        tracks: Literal[False],
        kind: Literal["tracks", "intervals"] | None = None,
    ) -> RaggedDataset[MaybeRSEQ, None]: ...
    @overload
    def with_tracks(
        self, tracks: str | list[str], kind: None = None
    ) -> RaggedDataset[MaybeRSEQ, MaybeRTRK]: ...
    @overload
    def with_tracks(
        self, tracks: str | list[str], kind: Literal["tracks"]
    ) -> RaggedDataset[MaybeRSEQ, RaggedTracks]: ...
    @overload
    def with_tracks(
        self, tracks: str | list[str], kind: Literal["intervals"]
    ) -> RaggedDataset[MaybeRSEQ, RaggedIntervals]: ...
    def with_tracks(
        self,
        tracks: str | list[str] | Literal[False] | None = None,
        kind: Literal["tracks", "intervals"] | None = None,
    ) -> RaggedDataset:
        return super().with_tracks(tracks, kind)

    @overload
    def __getitem__(
        self: RaggedDataset[None, None],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> NoReturn: ...
    @overload
    def __getitem__(
        self: RaggedDataset[RSEQ, None],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> RSEQ: ...
    @overload
    def __getitem__(
        self: RaggedDataset[None, Ragged[np.float32]],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> Ragged[np.float32]: ...
    @overload
    def __getitem__(
        self: RaggedDataset[RSEQ, Ragged[np.float32]],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> tuple[RSEQ, Ragged[np.float32]]: ...
    @overload
    def __getitem__(
        self: RaggedDataset[None, RaggedIntervals],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> RaggedIntervals: ...
    @overload
    def __getitem__(
        self: RaggedDataset[RSEQ, RaggedIntervals],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> tuple[RSEQ, RaggedIntervals]: ...
    @overload
    def __getitem__(
        self: RaggedDataset[RSEQ, MaybeRTRK],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> RSEQ | tuple[RSEQ, Ragged[np.float32]]: ...
    @overload
    def __getitem__(
        self: RaggedDataset[MaybeRSEQ, Ragged[np.float32]],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> Ragged[np.float32] | tuple[RSEQ, Ragged[np.float32]]: ...
    @overload
    def __getitem__(
        self: RaggedDataset[MaybeRSEQ, RaggedIntervals],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> RaggedIntervals | tuple[RSEQ, RaggedIntervals]: ...
    @overload
    def __getitem__(
        self: RaggedDataset[MaybeRSEQ, MaybeRTRK],
        idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx],
    ) -> RSEQ | Ragged[np.float32] | tuple[RSEQ, Ragged[np.float32]]: ...
    def __getitem__(
        self, idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx]
    ) -> RSEQ | RTRK | tuple[RSEQ, RTRK]:
        return super().__getitem__(idx)  # type: ignore[bad-return]  # base Dataset returns broad union; RSEQ/RTRK typevars narrow at use sites
