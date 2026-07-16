"""Local Isaac Lab teleoperation session orchestration.

Unlike rollouts, a teleoperation session has no policy to serve and needs no
remote GPU host: it's a single local ``isaaclab.sh`` process that opens the
interactive Isaac Sim viewer for a human to drive directly.
"""

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

from pydantic import BaseModel

from .alex_models import TeleopConfig, build_isaaclab_teleop_command

TeleopState = Literal["starting", "running", "done", "failed", "stopped", "interrupted"]


class TeleopRecord(BaseModel):
    id: str
    state: TeleopState
    config: TeleopConfig
    started_at: float
    ended_at: float | None = None
    exit_code: int | None = None
    pid: int | None = None
    log_path: str
    error_message: str | None = None


class TeleopManager:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, TeleopRecord] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._handles: dict[str, object] = {}
        self._lock = threading.RLock()
        self._load()

    def _dir(self, teleop_id: str) -> Path:
        return self.root / teleop_id

    def _persist(self, record: TeleopRecord) -> None:
        path = self._dir(record.id) / "session.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(record.model_dump_json(indent=2))
        os.replace(tmp, path)

    def _load(self) -> None:
        for path in self.root.glob("*/session.json"):
            try:
                record = TeleopRecord.model_validate_json(path.read_text())
            except Exception:
                continue
            if record.state in {"starting", "running"}:
                record.state = "interrupted"
                record.ended_at = time.time()
                record.error_message = "Alex Lab restarted while the teleop session was active"
                self._persist(record)
            self._records[record.id] = record

    def start(self, config: TeleopConfig) -> TeleopRecord:
        isaaclab_root = Path(config.isaaclab_root).expanduser().resolve()
        launcher = isaaclab_root / "isaaclab.sh"
        if not launcher.is_file():
            raise FileNotFoundError(f"Isaac Lab launcher not found: {launcher}")

        teleop_id = f"teleop-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        log_path = self._dir(teleop_id) / "teleop.log"
        record = TeleopRecord(
            id=teleop_id,
            state="starting",
            config=config,
            started_at=time.time(),
            log_path=str(log_path),
        )

        command = build_isaaclab_teleop_command(config)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a")
        isaaclab_env = os.environ.copy()
        isaaclab_env.pop("CONDA_PREFIX", None)
        try:
            process = subprocess.Popen(
                command,
                cwd=str(isaaclab_root),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=isaaclab_env,
            )
        except Exception as exc:
            handle.close()
            record.state = "failed"
            record.ended_at = time.time()
            record.error_message = str(exc)
            with self._lock:
                self._records[record.id] = record
            self._persist(record)
            raise RuntimeError(str(exc)) from exc

        record.pid = process.pid
        record.state = "running"
        with self._lock:
            self._records[record.id] = record
            self._processes[record.id] = process
            self._handles[record.id] = handle
        self._persist(record)
        return record

    def get(self, teleop_id: str) -> TeleopRecord:
        record = self._records.get(teleop_id)
        if record is None:
            raise KeyError(teleop_id)
        process = self._processes.get(teleop_id)
        if record.state == "running" and process is not None and process.poll() is not None:
            record.exit_code = process.returncode
            record.state = "done" if process.returncode == 0 else "failed"
            record.ended_at = time.time()
            self._processes.pop(teleop_id, None)
            self._cleanup(record)
            self._persist(record)
        return record

    def list(self) -> list[TeleopRecord]:
        return sorted(
            (self.get(item) for item in list(self._records)), key=lambda r: r.started_at, reverse=True
        )

    def logs(self, teleop_id: str) -> str:
        record = self.get(teleop_id)
        path = Path(record.log_path)
        return path.read_text(errors="replace") if path.is_file() else ""

    def stop(self, teleop_id: str) -> TeleopRecord:
        record = self.get(teleop_id)
        if record.state != "running":
            raise RuntimeError("teleop session is not running")
        process = self._processes.pop(teleop_id, None)
        if process is not None and process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
        record.state = "stopped"
        record.ended_at = time.time()
        self._cleanup(record)
        self._persist(record)
        return record

    def _cleanup(self, record: TeleopRecord) -> None:
        handle = self._handles.pop(record.id, None)
        if handle is not None:
            handle.close()


_DEFAULT_ROOT = Path(os.environ.get("ALEX_TELEOP_ROOT", Path.home() / ".cache/alex-lab/teleop"))
teleop_manager = TeleopManager(_DEFAULT_ROOT)
