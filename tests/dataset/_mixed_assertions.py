"""Shared mixed-output parity assertions (issue #279/#375).

`_assert_cell_equal` and `_assert_haps_cell_equal` are the byte-identical
parity checks used by both the SVAR1 mixed-tracks tests
(`test_streaming_tracks.py`) and the VCF/PGEN record-backend mixed-tracks
tests (`test_streaming_tracks_record.py`). Moved here verbatim (Ruling 41,
task-8-brief.md) so neither suite has to redefine -- or, worse, reinvent a
weaker version of -- the accessor-trap-aware assertions below.
"""

from __future__ import annotations

import numpy as np


def _assert_cell_equal(streamed, expected, ctx=""):
    """Assert one ``(region, sample)`` cell is byte-identical to the oracle.

    Compares a streamed cell against ``Dataset[r, s]``'s tracks half through
    the Ragged's own ``shape`` / ``lengths`` / packed values, NOT by indexing
    down to leaves. Two accessor traps make the obvious spelling wrong:

    1. Chained integer indexing (``cell[t][h]``) does NOT return a sub-Ragged.
       seqpro CONCATENATES the indexed group, so ``cell[0]`` on a
       ``(t, p, None)`` Ragged yields one flat array holding BOTH haplotypes
       and ``cell[0][0]`` collapses to a 0-d scalar -- comparing a scalar
       against an 18-element array and reporting a value diff for what is
       really a bad accessor.
    2. ``.data`` on a cell sliced out of a batch is the WHOLE batch's backing
       buffer, not the cell's slice of it, so its length is the batch total
       (80) rather than the cell's (38) even when ``.lengths`` already agrees.
       ``.to_packed()`` trims it to just this cell's values.

    ``with_len(L)`` returns a dense ndarray rather than a Ragged, so handle
    that shape-only case separately instead of demanding a ``.lengths``.

    The ragged assertions are ordered so the most diagnostic one fails first,
    matching the spec's three required parity axes:

    1. ``shape[:-1]`` -- rank plus the track and ploidy sizes. A spurious
       squeeze or a missing ploidy axis fails HERE, as a shape error, rather
       than silently broadcasting into a value diff.
    2. ``lengths`` -- per-(track, hap) output length. A mismatch is the
       signature of the extend_to_length span bug and must fail loudly and
       separately from a value diff.
    3. packed values -- the flat values in ``(track, ploidy)`` order. This is
       the byte oracle, and comparing the FLAT buffer is what makes the
       assertion sensitive to the ``(t, b, p)`` vs ``(b, t, p)`` assembly
       ordering: a mis-ordered buffer has identical shape and identical
       lengths, and differs only here.
    """
    if not hasattr(streamed, "lengths") or not hasattr(expected, "lengths"):
        # `with_len(L)` yields dense arrays on both sides; shape carries the
        # track and ploidy axes directly, so one comparison covers everything.
        s, e = np.asarray(streamed), np.asarray(expected)
        assert s.shape == e.shape, f"{ctx}shape {s.shape} != oracle {e.shape}"
        np.testing.assert_array_equal(s, e, err_msg=f"{ctx}track values differ")
        return

    s_shape, e_shape = streamed.shape[:-1], expected.shape[:-1]
    assert s_shape == e_shape, f"{ctx}shape {s_shape} != oracle {e_shape}"
    np.testing.assert_array_equal(
        np.asarray(streamed.lengths),
        np.asarray(expected.lengths),
        err_msg=f"{ctx}per-(track, hap) lengths differ",
    )
    np.testing.assert_array_equal(
        np.asarray(streamed.to_packed().data),
        np.asarray(expected.to_packed().data),
        err_msg=f"{ctx}track values differ",
    )


def _assert_haps_cell_equal(streamed, expected, ploidy: int, ctx="") -> None:
    """Assert the HAPLOTYPE half of one mixed cell matches ``Dataset[r, s][0]``.

    Issue #380: before the mixed drive returned both halves, streaming
    reconstructed these bytes and discarded them, so no mixed fixture could
    ever check them -- the haplotype half of the mixed path was un-oracled.

    Compared per haplotype (the spelling ``test_streaming_with_len.py`` and
    ``test_streaming_vcf_parity.py`` already use) rather than through
    ``to_packed()``: it is the one form that works unchanged for BOTH the
    ragged output and the dense ndarray ``with_len(L)`` yields, on either
    side, and it names the offending haplotype when it fails.
    """
    for h in range(ploidy):
        got = np.asarray(streamed[h])
        exp = np.asarray(expected[h])
        assert got.shape == exp.shape, (
            f"{ctx}hap {h}: shape {got.shape} != oracle {exp.shape}"
        )
        np.testing.assert_array_equal(
            got, exp, err_msg=f"{ctx}hap {h}: haplotype bytes differ"
        )
