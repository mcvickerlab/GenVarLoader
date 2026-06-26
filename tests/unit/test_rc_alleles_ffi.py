import numpy as np
import genvarloader.genvarloader as _gvl  # compiled rust extension module


def test_rc_alleles_ffi_inplace():
    # 2 rows. row0 (masked): alleles "AC","G". row1 (unmasked): "TT".
    data = np.frombuffer(b"ACGTT", np.uint8).copy()
    seq_offsets = np.array([0, 2, 3, 5], np.int64)
    var_offsets = np.array([0, 2, 3], np.int64)
    to_rc_row = np.array([True, False], np.bool_)
    _gvl.rc_alleles(data, seq_offsets, var_offsets, to_rc_row)
    assert data.tobytes() == b"GTCTT"
