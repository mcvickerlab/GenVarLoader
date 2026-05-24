from pathlib import Path

from genvarloader._bigwig import BigWigs
from genvarloader._types import IntervalTrack


def test_bigwigs_satisfies_interval_track_protocol():
    data_dir = Path(__file__).parent.parent / "data" / "bigwig"
    bw = BigWigs(
        "signal",
        {
            "sample_0": str(data_dir / "sample_0.bw"),
            "sample_1": str(data_dir / "sample_1.bw"),
        },
    )
    # runtime structural check: required attributes/methods are present
    assert hasattr(bw, "name")
    assert hasattr(bw, "samples")
    assert hasattr(bw, "contigs")
    assert callable(getattr(bw, "count_intervals", None))
    assert callable(getattr(bw, "_intervals_from_offsets", None))
    # static checker affirms it via type alias usage
    track: IntervalTrack = bw  # noqa: F841
