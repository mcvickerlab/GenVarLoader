import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from genvarloader import JsonProgressSink, ProgressEvent
from genvarloader._progress import _resolve_progress


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
        "state": "complete",
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


def test_json_progress_sink_ignores_invalid_attempt_metadata(tmp_path: Path):
    path = tmp_path / "progress.json"
    sink = JsonProgressSink(
        path,
        environ={"NF_SEQLAB_PROGRESS_ATTEMPT": "not-an-integer"},
    )

    sink(ProgressEvent(phase="write", completed=0, total=1, unit="regions"))

    assert json.loads(path.read_text())["attempt"] is None


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
