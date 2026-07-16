"""Durable Docker jobs launched over the active SSH cluster session."""

from __future__ import annotations

import json
import os
import re
import shlex
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .alex_models import LeRobotTrainingConfig, RemoteTrainingRequest, build_lerobot_training_command
from .cluster import ClusterManager, cluster_manager
from .jobs import MetricsHistoryPoint, TrainingMetrics, parse_metrics_into

RemoteJobState = Literal["running", "done", "failed", "stopped", "unknown"]
_CONTAINER_SAFE = re.compile(r"[^a-z0-9_.-]+")


class RemoteJobRecord(BaseModel):
    id: str
    name: str
    state: RemoteJobState
    kind: Literal["lerobot", "gr00t", "ccil"]
    container_name: str
    container_id: str | None = None
    gpu_ids: list[str]
    gpu_uuids: list[str]
    config: dict[str, Any]
    started_at: float
    ended_at: float | None = None
    exit_code: int | None = None
    error_message: str | None = None
    log_path: str
    metrics: TrainingMetrics = Field(default_factory=TrainingMetrics)
    runner: Literal["ssh_docker"] = "ssh_docker"


def parse_remote_metrics_history(logs: str) -> list[MetricsHistoryPoint]:
    """Extract the durable LeRobot loss/lr series from Docker output."""
    by_step: dict[int, MetricsHistoryPoint] = {}
    for line in logs.splitlines():
        if "step:" not in line or "loss:" not in line:
            continue
        metrics = TrainingMetrics()
        parse_metrics_into(line, metrics)
        if metrics.current_step <= 0:
            continue
        by_step[metrics.current_step] = MetricsHistoryPoint(
            step=metrics.current_step,
            loss=metrics.current_loss,
            lr=metrics.current_lr,
            grad_norm=metrics.grad_norm,
        )
    return [by_step[step] for step in sorted(by_step)]


def build_remote_docker_command(
    job_id: str,
    config: LeRobotTrainingConfig,
    gpu_uuids: list[str],
    container_name: str | None = None,
    image: str | None = None,
) -> str:
    """Build a shell-safe detached Docker launch command containing no secrets."""
    if not gpu_uuids:
        raise ValueError("at least one GPU is required")
    container_name = container_name or f"alex-{job_id}"
    argv = [
        "docker",
        "run",
        "--detach",
        "--name",
        container_name,
        "--gpus",
        # Docker's --gpus parser treats a comma as a request separator unless
        # the device expression itself retains quotes after shell parsing.
        # This yields the documented argv value: "device=GPU-a,GPU-b".
        f'"device={",".join(gpu_uuids)}"',
        "--shm-size",
        "64g",
        "--label",
        f"alex.job_id={job_id}",
    ]
    output_dir = "/outputs/run"
    video_tolerance_s = os.environ.get("ALEX_VIDEO_TIMESTAMP_TOLERANCE_S", "0.012")
    argv += [
        "--env",
        "HF_TOKEN",
        "--env",
        "HF_HOME=/cache/huggingface",
        "--env",
        f"ALEX_VIDEO_TIMESTAMP_TOLERANCE_S={video_tolerance_s}",
    ]
    argv += ["--volume", "alex_hf_cache:/cache/huggingface"]
    argv += ["--volume", f"alex_{job_id}_checkpoints:/outputs"]
    argv.append(image or remote_training_image())
    manifest = {
        "version": 1,
        "dataset_repo_id": config.dataset_repo_id,
        "dataset_revision": config.dataset_revision,
        "model_repo_id": config.model_repo_id,
        "policy_type": config.policy_type,
        "target_profile": "alex_v2_ability_hands",
    }
    argv += [
        "python3",
        "/opt/alex/alex_train_with_manifest.py",
        "--manifest",
        json.dumps(manifest, separators=(",", ":")),
        "--",
    ]
    argv += build_lerobot_training_command(config, output_dir, len(gpu_uuids))
    launch = shlex.join(argv)
    return with_remote_hf_token(launch)


def remote_training_image() -> str:
    return os.environ.get("ALEX_LEROBOT_IMAGE", "alex-lerobot-train:0.6.0")


def with_remote_hf_token(command: str) -> str:
    """Export gpu2's active Hub token or fail without ever printing it."""
    return remote_hf_token_prelude() + command


def remote_hf_token_prelude() -> str:
    """POSIX shell prelude that also works in Paramiko's minimal PATH."""
    cli_missing = (
        "Hugging Face CLI is not visible to non-interactive SSH on gpu2; "
        "install it in ~/.local/bin or set a system-wide PATH"
    )
    token_missing = "Hugging Face authentication is missing on gpu2; run: hf auth login"
    candidates = (
        '"$HOME/.local/bin/hf" "$HOME/miniconda3/bin/hf" '
        '"$HOME/anaconda3/bin/hf" "$HOME/.conda/bin/hf"'
    )
    return (
        'HF_CLI="$(command -v hf 2>/dev/null || true)"; '
        f"for candidate in {candidates}; do "
        'if [ -z "$HF_CLI" ] && [ -x "$candidate" ]; then HF_CLI="$candidate"; fi; done; '
        'if [ -z "$HF_CLI" ]; then '
        f"echo {shlex.quote(cli_missing)} >&2; exit 1; fi; "
        'HF_TOKEN="$("$HF_CLI" auth token 2>/dev/null)" || '
        f"{{ echo {shlex.quote(token_missing)} >&2; exit 1; }}; "
        'if [ -z "$HF_TOKEN" ]; then '
        f"echo {shlex.quote(token_missing)} >&2; exit 1; fi; "
        "export HF_TOKEN; "
    )


class RemoteJobManager:
    def __init__(self, root: Path, cluster: ClusterManager = cluster_manager) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cluster = cluster
        self._lock = threading.RLock()
        self._records: dict[str, RemoteJobRecord] = {}
        self._load()

    def _record_path(self, job_id: str) -> Path:
        return self.root / job_id / "job.json"

    def _log_path(self, job_id: str) -> Path:
        return self.root / job_id / "docker.log"

    def _metrics_path(self, job_id: str) -> Path:
        return self.root / job_id / "metrics.json"

    def _persist(self, record: RemoteJobRecord) -> None:
        path = self._record_path(record.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(record.model_dump_json(indent=2))
        os.replace(temp, path)

    def _load(self) -> None:
        for path in self.root.glob("*/job.json"):
            try:
                record = RemoteJobRecord.model_validate_json(path.read_text())
            except Exception:
                continue
            self._records[record.id] = record

    def reservations(self) -> dict[str, str]:
        with self._lock:
            return {
                gpu_uuid: record.id
                for record in self._records.values()
                if record.state == "running"
                for gpu_uuid in record.gpu_uuids
            }

    def start(self, request: RemoteTrainingRequest) -> RemoteJobRecord:
        available = self.cluster.gpus()
        by_identifier = {
            identifier: gpu for gpu in available for identifier in (str(gpu["index"]), gpu["uuid"])
        }
        try:
            selected = [by_identifier[item] for item in request.gpus]
        except KeyError as exc:
            raise ValueError(f"unknown GPU identifier: {exc.args[0]}") from exc
        uuids = [gpu["uuid"] for gpu in selected]
        with self._lock:
            reserved = self.reservations()
            occupied = [gpu for gpu in uuids if gpu in reserved]
            if occupied:
                raise ValueError(f"GPU already reserved: {', '.join(occupied)}")
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            job_id = f"{request.config.policy_type}-{timestamp}-{uuid.uuid4().hex[:8]}"
            container = _CONTAINER_SAFE.sub("-", f"alex-{job_id}".lower()).strip("-")[:63]
            log_path = self._log_path(job_id)
            record = RemoteJobRecord(
                id=job_id,
                name=request.name
                or f"{request.config.policy_type.upper()} · {request.config.dataset_repo_id}",
                state="running",
                kind=request.config.kind,
                container_name=container,
                gpu_ids=request.gpus,
                gpu_uuids=uuids,
                config=request.config.model_dump(),
                started_at=time.time(),
                log_path=str(log_path),
            )
            self._records[job_id] = record
            self._persist(record)
            command = build_remote_docker_command(job_id, request.config, uuids, container)
            try:
                code, stdout, stderr = self.cluster.execute(command, timeout=60)
                if code:
                    raise RuntimeError(stderr.strip() or "docker run failed")
                record.container_id = stdout.strip().splitlines()[0] if stdout.strip() else None
                self._persist(record)
            except Exception as exc:
                record.state = "failed"
                record.ended_at = time.time()
                record.error_message = str(exc)
                self._persist(record)
                raise
        return record

    def get(self, job_id: str, refresh: bool = True) -> RemoteJobRecord:
        with self._lock:
            record = self._records.get(job_id)
        if record is None:
            raise KeyError(job_id)
        if refresh and record.state == "running" and self.cluster.status()["connected"]:
            self.refresh(job_id)
        return record

    def list(self) -> list[RemoteJobRecord]:
        with self._lock:
            records = list(self._records.values())
        return sorted(records, key=lambda item: item.started_at, reverse=True)

    def refresh(self, job_id: str) -> RemoteJobRecord:
        record = self.get(job_id, refresh=False)
        command = (
            "docker inspect --format "
            + shlex.quote("{{json .State}}")
            + " "
            + shlex.quote(record.container_name)
        )
        code, stdout, stderr = self.cluster.execute(command)
        if code:
            record.state = "unknown"
            record.error_message = stderr.strip() or "container not found"
        else:
            state = json.loads(stdout.strip())
            running = bool(state.get("Running"))
            exit_code = state.get("ExitCode")
            if running:
                record.state = "running"
            else:
                record.exit_code = int(exit_code) if exit_code is not None else None
                record.state = "done" if record.exit_code == 0 else "failed"
                record.ended_at = record.ended_at or time.time()
                record.error_message = state.get("Error") or record.error_message
        self._persist(record)
        return record

    def logs(self, job_id: str, tail: int = 2000) -> str:
        record = self.get(job_id, refresh=False)
        tail = min(max(tail, 1), 10000)
        command = f"docker logs --timestamps --tail {tail} {shlex.quote(record.container_name)}"
        _code, stdout, stderr = self.cluster.execute(command)
        logs = stdout + stderr
        path = Path(record.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(logs)
        for line in logs.splitlines():
            parse_metrics_into(line, record.metrics)
        self._persist(record)
        return logs

    def persisted_logs(self, job_id: str) -> str:
        record = self.get(job_id, refresh=False)
        path = Path(record.log_path)
        return path.read_text() if path.is_file() else ""

    def metrics_history(self, job_id: str) -> list[MetricsHistoryPoint]:
        """Return loss/lr history, refreshing the persisted Docker log when possible."""
        self.get(job_id, refresh=False)
        path = self._metrics_path(job_id)
        cached: list[MetricsHistoryPoint] = []
        if path.is_file():
            try:
                cached = [MetricsHistoryPoint.model_validate(item) for item in json.loads(path.read_text())]
            except (OSError, ValueError, TypeError):
                cached = []
        if self.cluster.status()["connected"]:
            logs = self.logs(job_id, tail=10000)
            fresh = parse_remote_metrics_history(logs)
            merged = {point.step: point for point in cached}
            merged.update({point.step: point for point in fresh})
            points = [merged[step] for step in sorted(merged)][-2000:]
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps([point.model_dump() for point in points]))
            os.replace(temp, path)
            return points
        if cached:
            return cached[-2000:]
        logs = self.persisted_logs(job_id)
        return parse_remote_metrics_history(logs)[-2000:]

    def stop(self, job_id: str) -> RemoteJobRecord:
        record = self.get(job_id, refresh=False)
        if record.state != "running":
            raise RuntimeError("job is not running")
        code, _stdout, stderr = self.cluster.execute(
            f"docker stop --time 10 {shlex.quote(record.container_name)}", timeout=20
        )
        if code:
            raise RuntimeError(stderr.strip() or "docker stop failed")
        record.state = "stopped"
        record.ended_at = time.time()
        self._persist(record)
        return record

    def reattach(self) -> list[RemoteJobRecord]:
        if not self.cluster.status()["connected"]:
            return self.list()
        for record in self.list():
            if record.state in {"running", "unknown"}:
                try:
                    self.refresh(record.id)
                except Exception:
                    continue
        return self.list()


_DEFAULT_REMOTE_ROOT = Path(
    os.environ.get(
        "ALEX_REMOTE_JOB_ROOT",
        Path.home() / ".cache" / "alex-lab" / "remote-jobs",
    )
)
remote_job_manager = RemoteJobManager(_DEFAULT_REMOTE_ROOT)
