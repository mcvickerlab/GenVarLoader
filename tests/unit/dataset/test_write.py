from pathlib import Path

import polars as pl
import pytest

from genvarloader._dataset._write import _write_track


def test_write_track_rejects_unsupported_type():
    """Custom IntervalTrack types are unsupported now that the legacy path is gone."""
    with pytest.raises(TypeError, match="BigWigs.*Table"):
        _write_track(Path("/tmp/unused"), pl.DataFrame(), object(), None, 1)
