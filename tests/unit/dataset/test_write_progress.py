import asyncio
import threading
from concurrent.futures import CancelledError as FuturesCancelledError
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from genoray._svar import dense2sparse

import genvarloader as gvl
from genvarloader import _atomic
from genvarloader import ProgressEvent
from genvarloader._dataset import _write
from genvarloader._dataset._write import (
    Metadata,
    _ProgressReporter,
    _write_from_svar,
    _write_from_svar2,
    _write_phased_chunked,
)


def _bed() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "chromStart": [0, 10, 0],
            "chromEnd": [5, 15, 5],
        }
    )


def _region_iter():
    for i, end in enumerate((5, 15, 5)):
        genos = np.empty((1, 1, 0), dtype=np.int8)
        var_idxs = np.empty(0, dtype=np.int32)
        desc = f"batch {i}" if i in (0, 2) else None
        yield [dense2sparse(genos, var_idxs)], end, desc


def _track():
    return SimpleNamespace(
        name="signal",
        contigs=["chr1", "chr2"],
        samples=["sample"],
    )


def test_phased_progress_counts_are_monotonic_without_native_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events: list[ProgressEvent] = []
    reporter = _ProgressReporter(events.append, total=3)
    reporter.start()
    monkeypatch.setattr(
        _write,
        "tqdm",
        lambda **kwargs: pytest.fail("managed progress must not construct tqdm"),
    )

    _write_phased_chunked(tmp_path, _bed(), _region_iter(), reporter)

    assert [event.completed for event in events] == [0, 1, 2]
    assert [event.message for event in events] == [None, "batch 0", None]
    assert all(event.state == "running" for event in events)
    assert reporter.completed == 3


def test_unmanaged_phased_writer_preserves_native_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class RecordingBar:
        def __init__(self):
            self.descriptions: list[str] = []
            self.updates: list[int] = []
            self.closed = False

        def set_description(self, description: str) -> None:
            self.descriptions.append(description)

        def update(self, amount: int = 1) -> None:
            self.updates.append(amount)

        def close(self) -> None:
            self.closed = True

    bar = RecordingBar()
    monkeypatch.setattr(_write, "tqdm", lambda **kwargs: bar)

    _write_phased_chunked(tmp_path, _bed(), _region_iter())

    assert bar.descriptions == ["batch 0", "batch 2"]
    assert bar.updates == [1, 1, 1]
    assert bar.closed


def test_svar_progress_uses_contig_batch_counts_without_native_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.svar"
    source.mkdir()
    (source / "variant_idxs.npy").write_bytes(b"")

    class FakeSvar:
        index = pl.DataFrame({"POS": [1], "ILEN": [[0]]})
        ploidy = 1
        path = source
        genos = SimpleNamespace(data=np.empty(0, dtype=np.int32))

        def _find_starts_ends_with_length(self, contig, starts, ends, samples, out):
            out[:] = 0

    events: list[ProgressEvent] = []
    reporter = _ProgressReporter(events.append, total=3)
    reporter.start()
    monkeypatch.setattr(_write, "_reject_unsupported_variants", lambda *args: None)
    monkeypatch.setattr(
        _write,
        "tqdm",
        lambda **kwargs: pytest.fail("managed progress must not construct tqdm"),
    )

    _write_from_svar(
        tmp_path / "out.gvl",
        _bed(),
        FakeSvar(),
        ["sample"],
        True,
        reporter,
    )

    assert [event.completed for event in events] == [0, 2]
    assert reporter.completed == 3


def test_svar2_progress_uses_contig_batch_counts_without_native_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.svar2"
    source.mkdir()

    class FakeSvar2:
        n_samples = 1
        ploidy = 1
        available_samples = ["sample"]
        path = source

        def _find_ranges(self, contig, starts, ends, samples):
            n_regions = len(starts)
            return {
                "vk_snp_range": np.zeros((n_regions, 2), dtype=np.int64),
                "vk_indel_range": np.zeros((n_regions, 2), dtype=np.int64),
                "dense_snp_range": np.zeros((n_regions, 2), dtype=np.int64),
                "dense_indel_range": np.zeros((n_regions, 2), dtype=np.int64),
            }

    events: list[ProgressEvent] = []
    reporter = _ProgressReporter(events.append, total=3)
    reporter.start()
    monkeypatch.setattr(
        _write,
        "tqdm",
        lambda **kwargs: pytest.fail("managed progress must not construct tqdm"),
    )
    monkeypatch.setattr(
        _write,
        "_svar2_region_max_ends",
        lambda svar2, contig, starts, ends, samples: np.asarray(ends, np.int32),
    )
    from genvarloader._dataset import _svar2_link

    monkeypatch.setattr(_svar2_link, "make_svar2_link", lambda *args: object())

    _write_from_svar2(
        tmp_path / "out.gvl",
        _bed(),
        FakeSvar2(),
        ["sample"],
        True,
        reporter,
    )

    assert [event.completed for event in events] == [0, 2]
    assert reporter.completed == 3


def test_write_emits_exact_completion_after_published_metadata_is_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dest = tmp_path / "dataset.gvl"
    events: list[ProgressEvent] = []
    terminal_metadata: list[Metadata] = []
    monkeypatch.setattr(_write, "_write_track", lambda *args, **kwargs: None)

    def callback(event: ProgressEvent) -> None:
        events.append(event)
        if event.state == "complete":
            terminal_metadata.append(
                Metadata.model_validate_json((dest / "metadata.json").read_text())
            )

    gvl.write(dest, _bed(), tracks=_track(), progress_callback=callback)

    assert [event.state for event in events] == ["running", "complete"]
    assert [event.phase for event in events] == ["write", "finalize"]
    assert [event.completed for event in events] == [0, 3]
    assert [event.percent for event in events] == [0, 100]
    assert terminal_metadata[0].n_regions == 3


def test_write_isolates_callback_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dest = tmp_path / "dataset.gvl"
    calls = 0
    monkeypatch.setattr(_write, "_write_track", lambda *args, **kwargs: None)

    def callback(event: ProgressEvent) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("consumer unavailable")

    gvl.write(dest, _bed(), tracks=_track(), progress_callback=callback)

    assert calls == 1
    assert (
        Metadata.model_validate_json((dest / "metadata.json").read_text()).n_regions
        == 3
    )


@pytest.mark.parametrize(
    ("error", "expected_state"),
    [
        (RuntimeError("track failed"), "failed"),
        (KeyboardInterrupt("cancelled"), "cancelled"),
        (asyncio.CancelledError("async cancelled"), "cancelled"),
        (FuturesCancelledError("future cancelled"), "cancelled"),
    ],
)
def test_write_emits_terminal_event_on_failure_or_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    expected_state: str,
):
    dest = tmp_path / "dataset.gvl"
    events: list[ProgressEvent] = []

    def fail_track(*_args, **_kwargs) -> None:
        raise error

    monkeypatch.setattr(_write, "_write_track", fail_track)

    with pytest.raises(type(error), match=str(error)):
        gvl.write(dest, _bed(), tracks=_track(), progress_callback=events.append)

    assert [event.state for event in events] == ["running", expected_state]
    assert events[-1].completed == 0
    assert events[-1].total == 3
    assert events[-1].percent == 0
    assert events[-1].message == str(error)
    assert not dest.exists()


def test_write_emits_indeterminate_terminal_event_before_total_is_known(
    tmp_path: Path,
):
    dest = tmp_path / "dataset.gvl"
    events: list[ProgressEvent] = []

    with pytest.raises(ValueError, match="At least one"):
        gvl.write(dest, _bed(), progress_callback=events.append)

    assert len(events) == 1
    assert events[0].state == "failed"
    assert events[0].completed == 0
    assert events[0].total is None
    assert events[0].percent is None
    assert events[0].message is not None
    assert not dest.exists()


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_unsuccessful_terminal_keeps_total_without_reporting_completion(state: str):
    events: list[ProgressEvent] = []
    reporter = _ProgressReporter(events.append, total=2)
    reporter.start()
    reporter.advance(2)

    reporter.stop(state, "publication did not finish")  # type: ignore[arg-type]

    assert events[-1].state == state
    assert events[-1].completed == 1
    assert events[-1].total == 2
    assert events[-1].percent == 50


@pytest.mark.parametrize("via_environment", [False, True])
def test_write_rejects_progress_snapshot_inside_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    via_environment: bool,
):
    dest = tmp_path / "dataset.gvl"
    snapshot = dest / "progress.json"
    kwargs = {}
    if via_environment:
        monkeypatch.setenv("NF_SEQLAB_PROGRESS_PATH", str(snapshot))
    else:
        kwargs["progress_path"] = snapshot
    monkeypatch.setattr(_write, "_write_track", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="outside the dataset destination"):
        gvl.write(dest, _bed(), tracks=_track(), **kwargs)

    assert not dest.exists()
    assert not snapshot.exists()


@pytest.mark.parametrize("via_environment", [False, True])
def test_write_rejects_progress_snapshot_at_reserved_lock_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    via_environment: bool,
):
    dest = tmp_path / "dataset.gvl"
    snapshot = Path(f"{dest}.lock")
    kwargs = {}
    if via_environment:
        monkeypatch.setenv("NF_SEQLAB_PROGRESS_PATH", str(snapshot))
    else:
        kwargs["progress_path"] = snapshot
    monkeypatch.setattr(_write, "_write_track", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="reserved dataset lock path"):
        gvl.write(dest, _bed(), tracks=_track(), **kwargs)

    assert not dest.exists()
    assert not snapshot.exists()


@pytest.mark.parametrize("via_environment", [False, True])
def test_write_rejects_progress_snapshot_beneath_reserved_lock_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    via_environment: bool,
):
    dest = tmp_path / "dataset.gvl"
    reserved_lock = Path(f"{dest}.lock")
    snapshot = reserved_lock / "progress.json"
    kwargs = {}
    if via_environment:
        monkeypatch.setenv("NF_SEQLAB_PROGRESS_PATH", str(snapshot))
    else:
        kwargs["progress_path"] = snapshot
    monkeypatch.setattr(_write, "_write_track", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="reserved dataset lock path"):
        gvl.write(dest, _bed(), tracks=_track(), **kwargs)

    assert not dest.exists()
    assert not reserved_lock.exists()


@pytest.mark.parametrize("via_environment", [False, True])
@pytest.mark.parametrize("beneath_lock", [False, True], ids=["exact", "descendant"])
def test_write_rejects_reserved_lock_paths_for_symlink_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    via_environment: bool,
    beneath_lock: bool,
):
    target_parent = tmp_path / "target"
    target_parent.mkdir()
    target = target_parent / "dataset.gvl"
    dest = tmp_path / "dataset-link.gvl"
    dest.symlink_to(target, target_is_directory=True)
    reserved_lock = Path(f"{dest}.lock")
    snapshot = reserved_lock / "progress.json" if beneath_lock else reserved_lock
    kwargs = {}
    if via_environment:
        monkeypatch.setenv("NF_SEQLAB_PROGRESS_PATH", str(snapshot))
    else:
        kwargs["progress_path"] = snapshot
    monkeypatch.setattr(_write, "_write_track", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="reserved dataset lock path"):
        gvl.write(dest, _bed(), tracks=_track(), **kwargs)

    assert dest.is_symlink()
    assert not target.exists()
    assert not reserved_lock.exists()


def test_concurrent_write_only_publisher_reports_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dest = tmp_path / "dataset.gvl"
    first_inside_write = threading.Event()
    release_first = threading.Event()
    second_waiting_for_lock = threading.Event()
    shared_lock = threading.Lock()
    acquire_count = 0
    acquire_count_lock = threading.Lock()

    class CoordinatedFileLock:
        def __init__(self, _path: str) -> None:
            pass

        def acquire(self, timeout: float) -> None:
            nonlocal acquire_count
            with acquire_count_lock:
                acquire_count += 1
                if acquire_count == 2:
                    second_waiting_for_lock.set()
            if not shared_lock.acquire(timeout=timeout):
                raise _atomic.Timeout

        def release(self) -> None:
            shared_lock.release()

    def write_track(_path, bed, *_args) -> None:
        if bed.height == 1:
            first_inside_write.set()
            assert release_first.wait(timeout=5)

    monkeypatch.setattr(_atomic, "FileLock", CoordinatedFileLock)
    monkeypatch.setattr(_write, "_write_track", write_track)

    first_events: list[ProgressEvent] = []
    second_events: list[ProgressEvent] = []

    def run(bed: pl.DataFrame, events: list[ProgressEvent]) -> BaseException | None:
        try:
            gvl.write(dest, bed, tracks=_track(), progress_callback=events.append)
        except BaseException as exc:
            return exc
        return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, _bed().head(1), first_events)
        assert first_inside_write.wait(timeout=5)
        second = pool.submit(run, _bed().head(2), second_events)
        assert second_waiting_for_lock.wait(timeout=5)
        release_first.set()
        first_error = first.result(timeout=5)
        second_error = second.result(timeout=5)

    assert first_error is None
    assert isinstance(second_error, FileExistsError)
    assert [event.state for event in first_events] == ["running", "complete"]
    assert all(event.state != "complete" for event in second_events)
    assert (
        Metadata.model_validate_json((dest / "metadata.json").read_text()).n_regions
        == 1
    )
