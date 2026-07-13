"""Local IsaacLab-Arena evaluation subprocesses and durable metadata."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .alex_models import EvaluationConfig, build_evaluation_command

EvaluationState = Literal["running", "done", "failed", "stopped", "interrupted"]


class EvaluationRecord(BaseModel):
    id: str
    state: EvaluationState
    config: EvaluationConfig
    command: list[str]
    started_at: float
    ended_at: float | None = None
    exit_code: int | None = None
    pid: int | None = None
    log_path: str
    artifacts: list[str] = Field(default_factory=list)
    error_message: str | None = None


class EvaluationManager:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, EvaluationRecord] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._log_handles: dict[str, object] = {}
        self._lock = threading.RLock()
        self._load()

    def _dir(self, evaluation_id: str) -> Path:
        return self.root / evaluation_id

    def _persist(self, record: EvaluationRecord) -> None:
        path = self._dir(record.id) / "evaluation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(record.model_dump_json(indent=2))
        os.replace(temp, path)

    def _load(self) -> None:
        for path in self.root.glob("*/evaluation.json"):
            try:
                record = EvaluationRecord.model_validate_json(path.read_text())
            except Exception:
                continue
            if record.state == "running" and (record.pid is None or not self._pid_alive(record.pid)):
                record.state = "interrupted"
                record.ended_at = time.time()
                self._persist(record)
            self._records[record.id] = record

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except (OSError, ValueError):
            return False
        return True

    def start(self, config: EvaluationConfig) -> EvaluationRecord:
        arena = Path(config.arena_root).expanduser().resolve()
        script = arena / "isaaclab_arena/evaluation/policy_runner.py"
        if not script.is_file():
            raise FileNotFoundError(f"IsaacLab-Arena policy runner not found: {script}")
        command = build_evaluation_command(config)
        evaluation_id = f"eval-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        log_path = self._dir(evaluation_id) / "evaluation.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a")
        record = EvaluationRecord(
            id=evaluation_id,
            state="running",
            config=config,
            command=command,
            started_at=time.time(),
            log_path=str(log_path),
        )
        try:
            process = subprocess.Popen(
                command,
                cwd=str(arena),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            handle.close()
            raise
        record.pid = process.pid
        with self._lock:
            self._records[record.id] = record
            self._processes[record.id] = process
            self._log_handles[record.id] = handle
            self._persist(record)
        return record

    def _artifacts(self, record: EvaluationRecord) -> list[str]:
        candidates = [self._dir(record.id)]
        if record.config.video_dir:
            candidates.append(Path(record.config.video_dir).expanduser())
        found: set[str] = set()
        for root in candidates:
            if not root.is_dir():
                continue
            for pattern in ("*.mp4", "*.json", "*.csv"):
                for path in root.rglob(pattern):
                    if path.is_file():
                        found.add(str(path.resolve()))
        return sorted(found)

    def get(self, evaluation_id: str) -> EvaluationRecord:
        with self._lock:
            record = self._records.get(evaluation_id)
            process = self._processes.get(evaluation_id)
        if record is None:
            raise KeyError(evaluation_id)
        if record.state == "running" and process is not None:
            code = process.poll()
            if code is not None:
                record.exit_code = code
                record.state = "done" if code == 0 else "failed"
                record.ended_at = time.time()
                record.artifacts = self._artifacts(record)
                with self._lock:
                    self._processes.pop(evaluation_id, None)
                    handle = self._log_handles.pop(evaluation_id, None)
                if handle is not None:
                    handle.close()
                self._persist(record)
        return record

    def list(self) -> list[EvaluationRecord]:
        with self._lock:
            ids = list(self._records)
        records = [self.get(item) for item in ids]
        return sorted(records, key=lambda item: item.started_at, reverse=True)

    def logs(self, evaluation_id: str) -> str:
        record = self.get(evaluation_id)
        path = Path(record.log_path)
        return path.read_text(errors="replace") if path.is_file() else ""

    def stop(self, evaluation_id: str) -> EvaluationRecord:
        record = self.get(evaluation_id)
        if record.state != "running" or record.pid is None:
            raise RuntimeError("evaluation is not running")
        with contextlib.suppress(ProcessLookupError):
            os.killpg(record.pid, signal.SIGTERM)
        record.state = "stopped"
        record.ended_at = time.time()
        with self._lock:
            self._processes.pop(evaluation_id, None)
            handle = self._log_handles.pop(evaluation_id, None)
        if handle is not None:
            handle.close()
        record.artifacts = self._artifacts(record)
        self._persist(record)
        return record


_DEFAULT_EVALUATION_ROOT = Path(
    os.environ.get(
        "ALEX_EVALUATION_ROOT",
        Path.home() / ".cache" / "alex-lab" / "evaluations",
    )
)
evaluation_manager = EvaluationManager(_DEFAULT_EVALUATION_ROOT)
