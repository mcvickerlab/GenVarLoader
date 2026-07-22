"""slot_overhead_bytes must upper-bound the real per-chunk serialization overhead
(offset-array +1 terminators + _align padding) that _output_bytes_per_instance omits."""

import numpy as np
import seqpro as sp
import genvarloader as gvl
from genvarloader._slot_overhead import slot_overhead_bytes
from genvarloader._shm_layout import write_chunk, HEADER_RESERVED


def _vw(ds, L=8):
    opt = gvl.VarWindowOpt(
        flank_length=L,
        token_alphabet=sp.alphabets.DNA,
        unknown_token=len(sp.alphabets.DNA),
        ref="window",
        alt="allele",
    )
    return (
        ds.with_tracks(False)
        .with_output_format("flat")
        .with_seqs("variant-windows", opt)
        .with_settings(unphased_union=True, jitter=0)
    )


def test_overhead_covers_real_minus_estimate():
    ds = _vw(gvl.get_dummy_dataset())
    R, S = ds.shape[:2]
    rr, ss = np.meshgrid(np.arange(R), np.arange(S), indexing="ij")
    r, s = rr.reshape(-1), ss.reshape(-1)
    chunk = ds[r, s]
    arrays = list(chunk) if isinstance(chunk, tuple) else [chunk]
    buf = memoryview(bytearray(64 * 1024 * 1024))
    real = write_chunk(buf, arrays, n_instances=len(r)) - HEADER_RESERVED
    est = int(
        np.asarray(ds._output_bytes_per_instance(r, s, include_offsets=True)).sum()
    )
    overhead = slot_overhead_bytes(ds)
    assert real - est <= overhead, f"real-est={real - est} exceeds overhead={overhead}"
    assert overhead >= 4096  # floor
