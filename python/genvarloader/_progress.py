"""Structured progress events and atomic JSON snapshots."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeAlias

from loguru import logger

ProgressState: TypeAlias = Literal["running", "complete", "failed", "cancelled"]
_PROGRESS_STATES = frozenset({"running", "complete", "failed", "cancelled"})


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One immutable, schema-validated artifact progress update."""

    phase: str
    completed: int
    total: int | None
    unit: str
    state: ProgressState = "running"
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise ValueError("phase must be a nonblank string")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("unit must be a nonblank string")
        if not isinstance(self.completed, int) or isinstance(self.completed, bool):
            raise ValueError("completed must be an integer")
        if self.total is not None and (
            not isinstance(self.total, int) or isinstance(self.total, bool)
        ):
            raise ValueError("total must be an integer or None")
        if not isinstance(self.state, str) or self.state not in _PROGRESS_STATES:
            raise ValueError(
                "state must be one of: running, complete, failed, cancelled"
            )
        if self.completed < 0:
            raise ValueError("completed must be >= 0")
        if self.total is not None:
            if self.total <= 0:
                raise ValueError("total must be > 0 when provided")
        if self.state == "complete":
            if self.total is None or self.completed != self.total:
                raise ValueError(
                    "complete progress requires a known total and completed == total"
                )
        elif self.total is not None and self.completed >= self.total:
            raise ValueError("non-complete progress must remain below total")

    @property
    def percent(self) -> float | None:
        """Return exact completion percent, or ``None`` when indeterminate."""
        if self.total is None:
            return None
        return 100.0 * self.completed / self.total


ProgressCallback: TypeAlias = Callable[[ProgressEvent], None]

_SNAPSHOT_NAME = ".nf-seqlab-progress.json"
_SNAPSHOT_MIN_INTERVAL_SECONDS = 0.25
_IDENTITY_ENV = {
    "run_id": "NF_SEQLAB_PROGRESS_RUN_ID",
    "stage_id": "NF_SEQLAB_PROGRESS_STAGE_ID",
    "process": "NF_SEQLAB_PROGRESS_PROCESS",
    "file_id": "NF_SEQLAB_PROGRESS_FILE_ID",
    "parent_file_id": "NF_SEQLAB_PROGRESS_PARENT_FILE_ID",
    "task_id": "NF_SEQLAB_PROGRESS_TASK_ID",
}


def _snapshot_path(
    progress_path: str | Path | None,
    environ: Mapping[str, str],
) -> Path | None:
    if progress_path is not None:
        return Path(progress_path)
    configured = (
        environ.get("NF_SEQLAB_PROGRESS_PATH")
        or environ.get("NF_SEQLAB_PROGRESS_FILE")
        or environ.get("NF_SEQLAB_PROGRESS_SNAPSHOT_PATH")
    )
    if configured:
        return Path(configured)
    if directory := environ.get("NF_SEQLAB_PROGRESS_DIR"):
        return Path(directory) / _SNAPSHOT_NAME
    shorthand = environ.get("NF_SEQLAB_PROGRESS")
    if shorthand and shorthand.lower() not in {"0", "false", "no", "off"}:
        if shorthand.lower() in {"1", "true", "yes", "on"}:
            return Path.cwd() / _SNAPSHOT_NAME
        return Path(shorthand)
    return None


class JsonProgressSink:
    """Atomically replace one throttled structured-progress JSON snapshot.

    A standalone sink writes the generic GenVarLoader schema. When every managed
    identity field and a positive attempt are configured, it writes the
    ``nf-seqlab.progress/v1`` schema instead. Publication is serialized per sink;
    terminal events are immediate and final, while running events are throttled.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.path = Path(path)
        env = os.environ if environ is None else environ
        self._identity = {field: env.get(name) for field, name in _IDENTITY_ENV.items()}
        parent_file_id = self._identity["parent_file_id"]
        if parent_file_id is None or not parent_file_id.strip():
            self._identity["parent_file_id"] = self._identity["file_id"]
        attempt = env.get("NF_SEQLAB_PROGRESS_ATTEMPT")
        try:
            self._attempt = int(attempt) if attempt is not None else None
        except ValueError:
            logger.warning(f"Ignoring invalid NF_SEQLAB_PROGRESS_ATTEMPT={attempt!r}")
            self._attempt = None
        if self._attempt is not None and self._attempt < 1:
            logger.warning(f"Ignoring invalid NF_SEQLAB_PROGRESS_ATTEMPT={attempt!r}")
            self._attempt = None
        self._managed = self._attempt is not None and all(
            value is not None and value.strip() for value in self._identity.values()
        )
        self._clock: Callable[[], float] = time.monotonic
        self._min_interval_seconds = _SNAPSHOT_MIN_INTERVAL_SECONDS
        self._last_write_at: float | None = None
        self._publication_lock = threading.Lock()
        self._terminal_published = False

    def __call__(self, event: ProgressEvent) -> None:
        with self._publication_lock:
            if self._terminal_published:
                return
            now = self._clock()
            if (
                event.state == "running"
                and self._last_write_at is not None
                and now - self._last_write_at < self._min_interval_seconds
            ):
                return

            self._write(event)
            self._last_write_at = now
            self._terminal_published = event.state != "running"

    def _write(self, event: ProgressEvent) -> None:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if self._managed:
            payload = {
                "schema": "nf-seqlab.progress/v1",
                **self._identity,
                "attempt": self._attempt,
                "state": "completed" if event.state == "complete" else event.state,
                "phase": event.phase,
                "completed": event.completed,
                "total": event.total,
                "unit": event.unit,
                "percent": event.percent,
                "message": event.message,
                "updated_at": timestamp,
            }
        else:
            identity: dict[str, str | int] = {
                field: value
                for field, value in self._identity.items()
                if value is not None and value.strip()
            }
            if self._attempt is not None:
                identity["attempt"] = self._attempt
            payload = {
                "schema_version": 1,
                "source": "genvarloader",
                "identity": identity,
                "state": event.state,
                "phase": event.phase,
                "completed": event.completed,
                "total": event.total,
                "unit": event.unit,
                "percent": event.percent,
                "message": event.message,
                "timestamp": timestamp,
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
            tmp_path = None
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)


class _ProgressFanout:
    """Deliver to independent observers, disabling each one after a failure."""

    def __init__(self, callbacks: list[ProgressCallback]) -> None:
        self._callbacks = callbacks

    def __call__(self, event: ProgressEvent) -> None:
        active: list[ProgressCallback] = []
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as exc:
                logger.warning(f"Progress callback failed; disabling it: {exc!r}")
            else:
                active.append(callback)
        self._callbacks = active


def _resolve_progress(
    progress_callback: ProgressCallback | None = None,
    progress_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    output_path: str | Path | None = None,
) -> ProgressCallback | None:
    """Combine an explicit callback with an explicit or environment JSON sink."""
    env = os.environ if environ is None else environ
    callbacks: list[ProgressCallback] = []
    if progress_callback is not None:
        callbacks.append(progress_callback)
    if path := _snapshot_path(progress_path, env):
        if output_path is not None:
            original_output = Path(output_path)
            output = original_output.resolve()
            snapshot = path.resolve()
            reserved_lock = Path(str(original_output) + ".lock").resolve()
            if snapshot == reserved_lock or reserved_lock in snapshot.parents:
                raise ValueError(
                    "progress snapshot path must be outside the reserved dataset "
                    f"lock path {reserved_lock}: {path}"
                )
            if snapshot == output or output in snapshot.parents:
                raise ValueError(
                    "progress snapshot path must be outside the dataset "
                    f"destination {output_path}: {path}"
                )
        callbacks.append(JsonProgressSink(path, environ=env))
    if not callbacks:
        return None
    return _ProgressFanout(callbacks)


__all__ = ["JsonProgressSink", "ProgressCallback", "ProgressEvent"]
