import json
import os
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from genvarloader import JsonProgressSink, ProgressEvent
from genvarloader._progress import _ProgressFanout, _resolve_progress


def test_progress_event_matches_shared_schema_and_supports_indeterminate_work():
    event = ProgressEvent(
        phase="read",
        completed=3,
        total=None,
        unit="regions",
        state="running",
        message="Reading regions",
    )

    assert event.percent is None
    with pytest.raises(FrozenInstanceError):
        event.completed = 4  # type: ignore[misc]


@pytest.mark.parametrize(("completed", "total"), [(-1, None), (2, 1), (0, 0)])
def test_progress_event_rejects_invalid_counts(completed: int, total: int | None):
    with pytest.raises(ValueError):
        ProgressEvent(phase="write", completed=completed, total=total, unit="regions")


@pytest.mark.parametrize(("field", "value"), [("phase", "   "), ("unit", "\t")])
def test_progress_event_rejects_blank_labels(field: str, value: str):
    kwargs = {"phase": "write", "unit": "regions"}
    kwargs[field] = value

    with pytest.raises(ValueError, match=f"{field} must be a nonblank string"):
        ProgressEvent(completed=0, total=1, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("state", ["completed", "unknown", ""])
def test_progress_event_rejects_invalid_state(state: str):
    with pytest.raises(ValueError, match="state must be one of"):
        ProgressEvent(
            phase="write",
            completed=0,
            total=1,
            unit="regions",
            state=state,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_progress_event_rejects_non_integer_completed(value: object):
    with pytest.raises(ValueError, match="completed must be an integer"):
        ProgressEvent(
            phase="write",
            completed=value,  # type: ignore[arg-type]
            total=2,
            unit="regions",
        )


@pytest.mark.parametrize("value", [True, 2.0, "2"])
def test_progress_event_rejects_non_integer_total(value: object):
    with pytest.raises(ValueError, match="total must be an integer or None"):
        ProgressEvent(
            phase="write",
            completed=1,
            total=value,  # type: ignore[arg-type]
            unit="regions",
        )


@pytest.mark.parametrize(("completed", "total"), [(0, None), (0, 1)])
def test_progress_event_complete_requires_exact_known_total(
    completed: int, total: int | None
):
    with pytest.raises(ValueError, match="complete progress requires"):
        ProgressEvent(
            phase="finalize",
            completed=completed,
            total=total,
            unit="regions",
            state="complete",
        )


@pytest.mark.parametrize("state", ["running", "failed", "cancelled"])
def test_progress_event_noncomplete_state_must_remain_below_known_total(state: str):
    with pytest.raises(
        ValueError, match="non-complete progress must remain below total"
    ):
        ProgressEvent(
            phase="write",
            completed=2,
            total=2,
            unit="regions",
            state=state,  # type: ignore[arg-type]
        )


def test_json_progress_sink_atomically_replaces_parseable_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "nested" / "progress.json"
    sink = JsonProgressSink(
        path,
        environ={
            "NF_SEQLAB_PROGRESS_RUN_ID": "aou-v8",
            "NF_SEQLAB_PROGRESS_STAGE_ID": "build-gvl",
            "NF_SEQLAB_PROGRESS_PROCESS": "SEQLAB_BUILD_GVL",
            "NF_SEQLAB_PROGRESS_FILE_ID": "chr22",
            "NF_SEQLAB_PROGRESS_PARENT_FILE_ID": "cohort",
            "NF_SEQLAB_PROGRESS_TASK_ID": "99/b6e5e2",
            "NF_SEQLAB_PROGRESS_ATTEMPT": "2",
        },
    )
    sink(
        ProgressEvent(
            phase="write", completed=1, total=4, unit="regions", message="chr1"
        )
    )
    old_payload = path.read_text()
    real_replace = os.replace
    replaced: list[tuple[Path, Path]] = []

    def inspect_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]):
        src_path, dst_path = Path(src), Path(dst)
        assert json.loads(src_path.read_text())["completed"] == 4
        assert dst_path.read_text() == old_payload
        replaced.append((src_path, dst_path))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", inspect_replace)
    sink(
        ProgressEvent(
            phase="finalize",
            completed=4,
            total=4,
            unit="regions",
            state="complete",
        )
    )

    assert replaced
    payload = json.loads(path.read_text())
    assert payload == {
        "schema": "nf-seqlab.progress/v1",
        "run_id": "aou-v8",
        "stage_id": "build-gvl",
        "process": "SEQLAB_BUILD_GVL",
        "file_id": "chr22",
        "parent_file_id": "cohort",
        "task_id": "99/b6e5e2",
        "attempt": 2,
        "state": "completed",
        "phase": "finalize",
        "completed": 4,
        "total": 4,
        "unit": "regions",
        "percent": 100.0,
        "message": None,
        "updated_at": payload["updated_at"],
    }
    assert payload["updated_at"].endswith("Z")
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_json_progress_sink_serializes_publication_so_terminal_is_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "progress.json"
    sink = JsonProgressSink(path, environ={})
    sink._min_interval_seconds = 0
    first_replace_started = threading.Event()
    later_replace_finished = threading.Event()
    replace_lock = threading.Lock()
    replace_count = 0
    real_replace = os.replace

    def delay_first_replace(
        src: str | os.PathLike[str], dst: str | os.PathLike[str]
    ) -> None:
        nonlocal replace_count
        with replace_lock:
            replace_count += 1
            is_first = replace_count == 1
        if is_first:
            first_replace_started.set()
            later_replace_finished.wait(timeout=0.5)
        real_replace(src, dst)
        if not is_first:
            later_replace_finished.set()

    monkeypatch.setattr(os, "replace", delay_first_replace)
    running = ProgressEvent(
        phase="write", completed=1, total=2, unit="regions", state="running"
    )
    terminal = ProgressEvent(
        phase="finalize", completed=2, total=2, unit="regions", state="complete"
    )

    running_thread = threading.Thread(target=sink, args=(running,))
    terminal_thread = threading.Thread(target=sink, args=(terminal,))
    running_thread.start()
    assert first_replace_started.wait(timeout=1)
    terminal_thread.start()
    running_thread.join(timeout=1)
    terminal_thread.join(timeout=1)

    assert not running_thread.is_alive()
    assert not terminal_thread.is_alive()
    assert json.loads(path.read_text())["state"] == "complete"


def test_json_progress_sink_ignores_events_after_terminal(tmp_path: Path):
    path = tmp_path / "progress.json"
    sink = JsonProgressSink(path, environ={})
    sink._min_interval_seconds = 0

    sink(
        ProgressEvent(
            phase="finalize",
            completed=2,
            total=2,
            unit="regions",
            state="complete",
        )
    )
    sink(ProgressEvent(phase="write", completed=1, total=2, unit="regions"))

    payload = json.loads(path.read_text())
    assert payload["state"] == "complete"
    assert payload["completed"] == 2


@pytest.mark.parametrize("parent_file_id", ["", "   "])
def test_json_progress_sink_normalizes_blank_parent_identity(
    tmp_path: Path, parent_file_id: str
):
    path = tmp_path / "progress.json"
    sink = JsonProgressSink(
        path,
        environ={
            "NF_SEQLAB_PROGRESS_RUN_ID": "aou-v8",
            "NF_SEQLAB_PROGRESS_STAGE_ID": "build-gvl",
            "NF_SEQLAB_PROGRESS_PROCESS": "SEQLAB_BUILD_GVL",
            "NF_SEQLAB_PROGRESS_FILE_ID": "chr22",
            "NF_SEQLAB_PROGRESS_PARENT_FILE_ID": parent_file_id,
            "NF_SEQLAB_PROGRESS_TASK_ID": "99/b6e5e2",
            "NF_SEQLAB_PROGRESS_ATTEMPT": "2",
        },
    )

    sink(ProgressEvent(phase="write", completed=0, total=1, unit="regions"))

    payload = json.loads(path.read_text())
    assert payload["schema"] == "nf-seqlab.progress/v1"
    assert payload["parent_file_id"] == "chr22"


def test_json_progress_sink_uses_generic_schema_without_managed_identity(
    tmp_path: Path,
):
    path = tmp_path / "progress.json"
    sink = JsonProgressSink(
        path,
        environ={"NF_SEQLAB_PROGRESS_RUN_ID": "local-run"},
    )

    sink(
        ProgressEvent(
            phase="finalize",
            completed=1,
            total=1,
            unit="artifact",
            state="complete",
        )
    )

    payload = json.loads(path.read_text())
    assert payload == {
        "schema_version": 1,
        "source": "genvarloader",
        "identity": {"run_id": "local-run"},
        "state": "complete",
        "phase": "finalize",
        "completed": 1,
        "total": 1,
        "unit": "artifact",
        "percent": 100.0,
        "message": None,
        "timestamp": payload["timestamp"],
    }
    assert payload["timestamp"].endswith("Z")


@pytest.mark.parametrize("attempt", ["not-an-integer", "0", "-1"])
def test_json_progress_sink_uses_generic_schema_for_invalid_attempt(
    tmp_path: Path, attempt: str
):
    path = tmp_path / "progress.json"
    sink = JsonProgressSink(
        path,
        environ={
            "NF_SEQLAB_PROGRESS_RUN_ID": "aou-v8",
            "NF_SEQLAB_PROGRESS_STAGE_ID": "build-gvl",
            "NF_SEQLAB_PROGRESS_PROCESS": "SEQLAB_BUILD_GVL",
            "NF_SEQLAB_PROGRESS_FILE_ID": "chr22",
            "NF_SEQLAB_PROGRESS_PARENT_FILE_ID": "cohort",
            "NF_SEQLAB_PROGRESS_TASK_ID": "99/b6e5e2",
            "NF_SEQLAB_PROGRESS_ATTEMPT": attempt,
        },
    )

    sink(ProgressEvent(phase="write", completed=0, total=1, unit="regions"))

    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert payload["source"] == "genvarloader"
    assert "schema" not in payload
    assert "attempt" not in payload


def test_environment_selects_json_sink_and_explicit_progress_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    env_path = tmp_path / "env.json"
    explicit_path = tmp_path / "explicit.json"
    events: list[ProgressEvent] = []
    environ = {"NF_SEQLAB_PROGRESS_PATH": str(env_path)}
    callback = _resolve_progress(events.append, explicit_path, environ=environ)
    event = ProgressEvent(phase="write", completed=1, total=2, unit="regions")

    assert callback is not None
    callback(event)

    assert events == [event]
    assert explicit_path.exists()
    assert not env_path.exists()


def test_single_failing_progress_callback_is_isolated():
    calls = 0

    def fail(_event: ProgressEvent) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("dashboard unavailable")

    callback = _resolve_progress(fail)
    assert callback is not None
    event = ProgressEvent(phase="write", completed=1, total=2, unit="regions")

    callback(event)
    callback(event)

    assert calls == 1


def test_json_sink_is_throttled_while_callback_receives_every_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "progress.json"
    sink = JsonProgressSink(path, environ={})
    now = 0.0
    sink._clock = lambda: now
    sink._min_interval_seconds = 1.0
    callback_events: list[ProgressEvent] = []
    fanout = _ProgressFanout([callback_events.append, sink])
    replace_calls = 0
    real_replace = os.replace

    def count_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]):
        nonlocal replace_calls
        replace_calls += 1
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", count_replace)
    emitted: list[ProgressEvent] = []
    for completed, timestamp in [(0, 0.0), (1, 0.1), (2, 0.2), (3, 1.1)]:
        now = timestamp
        event = ProgressEvent(
            phase="write", completed=completed, total=4, unit="regions"
        )
        emitted.append(event)
        fanout(event)

    now = 1.2
    terminal = ProgressEvent(
        phase="finalize",
        completed=4,
        total=4,
        unit="regions",
        state="complete",
    )
    emitted.append(terminal)
    fanout(terminal)

    assert callback_events == emitted
    assert replace_calls == 3
    payload = json.loads(path.read_text())
    assert payload["state"] == "complete"
    assert payload["completed"] == 4


@pytest.mark.parametrize(
    ("environ", "relative_path"),
    [
        ({"NF_SEQLAB_PROGRESS_FILE": "file.json"}, "file.json"),
        (
            {"NF_SEQLAB_PROGRESS_SNAPSHOT_PATH": "snapshot.json"},
            "snapshot.json",
        ),
        ({"NF_SEQLAB_PROGRESS_DIR": "progress"}, "progress/.nf-seqlab-progress.json"),
        ({"NF_SEQLAB_PROGRESS": "shorthand.json"}, "shorthand.json"),
        ({"NF_SEQLAB_PROGRESS": "true"}, ".nf-seqlab-progress.json"),
    ],
)
def test_progress_environment_path_forms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environ: dict[str, str],
    relative_path: str,
):
    monkeypatch.chdir(tmp_path)
    callback = _resolve_progress(environ=environ)

    assert callback is not None
    callback(ProgressEvent(phase="write", completed=0, total=1, unit="regions"))

    assert (tmp_path / relative_path).exists()
