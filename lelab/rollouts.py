"""Unified rollout orchestration for remote policy inference and Alex targets."""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .alex_models import RolloutConfig, build_arena_rollout_command, build_isaaclab_rollout_command
from .cluster import ClusterManager, SshTunnel, cluster_manager
from .remote_jobs import (
    RemoteJobManager,
    remote_hf_token_prelude,
    remote_job_manager,
    remote_training_image,
)

RolloutState = Literal["starting", "running", "done", "failed", "stopped", "blocked", "interrupted"]
_LOCAL_POLICY_MOUNT = "/policy"


class RolloutRecord(BaseModel):
    id: str
    state: RolloutState
    config: RolloutConfig
    policy_ref: str
    manifest: dict
    started_at: float
    ended_at: float | None = None
    exit_code: int | None = None
    pid: int | None = None
    inference_container: str | None = None
    inference_port: int | None = None
    log_path: str
    artifacts: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    error_message: str | None = None
    blockers: list[str] = Field(default_factory=list)


def resolve_rollout_source(config: RolloutConfig, jobs: RemoteJobManager) -> tuple[str, dict]:
    """Resolve a job/checkpoint or direct ref and create its compatibility manifest."""
    dataset_repo_id = config.dataset_repo_id
    policy_type = None
    portable: dict = {}
    if config.job_id:
        job = jobs.get(config.job_id, refresh=False)
        job_config = job.config
        repo = job_config.get("model_repo_id") or job_config.get("policy_repo_id")
        if not repo:
            raise ValueError(f"job {config.job_id!r} has no model repository")
        dataset_repo_id = dataset_repo_id or job_config.get("dataset_repo_id")
        policy_type = job_config.get("policy_type")
        if config.checkpoint == "latest":
            policy_ref = f"{repo}@latest"
        elif config.checkpoint.isdigit():
            policy_ref = f"{repo}@checkpoints/{int(config.checkpoint):06d}"
        else:
            raise ValueError("checkpoint must be 'latest' or a numeric training step")
    else:
        assert config.policy_ref is not None
        policy_ref = str(Path(config.policy_ref).expanduser()) if Path(config.policy_ref).expanduser().exists() else config.policy_ref
        if "@" not in policy_ref and not Path(policy_ref).is_dir():
            if config.checkpoint == "latest":
                policy_ref = f"{policy_ref}@latest"
            elif config.checkpoint.isdigit():
                policy_ref = f"{policy_ref}@checkpoints/{int(config.checkpoint):06d}"
            else:
                raise ValueError("checkpoint must be 'latest' or a numeric training step")

        portable = _load_portable_manifest(policy_ref) or {}
        if portable:
            dataset_repo_id = dataset_repo_id or portable.get("dataset_repo_id")
            policy_type = portable.get("policy_type")

    if not dataset_repo_id:
        raise ValueError(
            "dataset metadata is required for compatibility checks; select a LeLab job or provide dataset_repo_id"
        )
    manifest = {
        **portable,
        "version": 1,
        "dataset_repo_id": dataset_repo_id,
        "policy_type": policy_type,
        "fps": config.fps,
        "target_profile": config.embodiment,
        "required_inputs": ["observation.state"],
        "required_outputs": ["action"],
        "camera_prefix": "observation.images.",
    }
    return policy_ref, manifest


def _load_portable_manifest(policy_ref: str) -> dict | None:
    """Read LeLab rollout metadata from a local or Hub checkpoint when present."""
    local = Path(policy_ref).expanduser()
    if local.is_dir():
        path = local / "lelab_rollout.json"
        if path.is_file():
            return json.loads(path.read_text())
        return None
    repo = policy_ref
    filename = "lelab_rollout.json"
    if "@checkpoints/" in policy_ref:
        repo, step = policy_ref.split("@checkpoints/", 1)
        filename = f"checkpoints/{step}/pretrained_model/lelab_rollout.json"
    elif policy_ref.endswith("@latest"):
        repo = policy_ref.removesuffix("@latest")
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo, repo_type="model", filename=filename)
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def build_inference_container_command(
    rollout_id: str,
    policy_ref: str,
    gpu_uuid: str,
    port: int,
) -> str:
    name = f"alex-rollout-{rollout_id}"[:63]
    argv = [
        "docker", "run", "--detach", "--name", name,
        "--network", "host",
        "--gpus", f'"device={gpu_uuid}"',
        "--shm-size", "16g",
        "--label", f"alex.rollout_id={rollout_id}",
        "--env", "HF_TOKEN", "--env", "HF_HOME=/cache/huggingface",
        "--volume", "alex_hf_cache:/cache/huggingface",
        remote_training_image(),
        "python3", "/opt/alex/alex_policy_server.py",
        "--policy", policy_ref, "--host", "127.0.0.1", "--port", str(port), "--device", "cuda",
    ]
    return remote_hf_token_prelude() + shlex.join(argv)


def build_local_inference_container_command(
    rollout_id: str,
    policy_ref: str,
    gpu: str,
    port: int,
) -> list[str]:
    """Build a local Docker command that downloads and serves a LeRobot policy."""
    name = f"alex-rollout-{rollout_id}"[:63]
    gpu_request = "all" if gpu == "all" else f"device={gpu}"
    policy_path = Path(policy_ref).expanduser()
    mounted_policy_ref = policy_ref
    volume_args: list[str] = []
    if policy_path.is_dir():
        resolved = policy_path.resolve()
        volume_args = ["--volume", f"{resolved}:{_LOCAL_POLICY_MOUNT}:ro"]
        mounted_policy_ref = _LOCAL_POLICY_MOUNT
    return [
        "docker", "run", "--detach", "--name", name,
        "--network", "host",
        "--gpus", gpu_request,
        "--shm-size", "16g",
        "--label", f"alex.rollout_id={rollout_id}",
        "--env", "HF_TOKEN", "--env", "HF_HOME=/cache/huggingface",
        "--volume", "alex_hf_cache:/cache/huggingface",
        *volume_args,
        remote_training_image(),
        "python3", "/opt/alex/alex_policy_server.py",
        "--policy", mounted_policy_ref, "--host", "127.0.0.1", "--port", str(port), "--device", "cuda",
    ]


class RolloutManager:
    def __init__(
        self,
        root: Path,
        cluster: ClusterManager = cluster_manager,
        jobs: RemoteJobManager = remote_job_manager,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cluster = cluster
        self.jobs = jobs
        self._records: dict[str, RolloutRecord] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._handles: dict[str, object] = {}
        self._tunnels: dict[str, SshTunnel] = {}
        self._lock = threading.RLock()
        self._load()

    def _dir(self, rollout_id: str) -> Path:
        return self.root / rollout_id

    def _persist(self, record: RolloutRecord) -> None:
        path = self._dir(record.id) / "rollout.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(record.model_dump_json(indent=2))
        os.replace(tmp, path)

    def _load(self) -> None:
        for path in self.root.glob("*/rollout.json"):
            try:
                record = RolloutRecord.model_validate_json(path.read_text())
            except Exception:
                continue
            if record.state in {"starting", "running"}:
                record.state = "interrupted"
                record.ended_at = time.time()
                record.error_message = "Alex Lab restarted while the rollout was active"
                self._persist(record)
            self._records[record.id] = record

    def _select_gpu(self, identifier: str) -> str:
        for gpu in self.cluster.gpus():
            if identifier in {str(gpu["index"]), gpu["uuid"]}:
                return str(gpu["uuid"])
        raise ValueError(f"unknown GPU identifier: {identifier}")

    def _wait_ready(self, local_port: int, timeout: float = 600, record: RolloutRecord | None = None) -> dict:
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{local_port}/schema"
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - loopback tunnel
                    return json.loads(response.read())
            except Exception as exc:
                last_error = exc
                if record is not None and self._local_container_exited(record):
                    raise RuntimeError("policy-server container exited before becoming ready") from exc
                time.sleep(1)
        raise TimeoutError(f"policy server did not become ready: {last_error}")

    def _wait_policy_server_ready(self, record: RolloutRecord, local_port: int, timeout: float = 600) -> dict:
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{local_port}/schema"
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - loopback tunnel
                    return json.loads(response.read())
            except Exception as exc:
                last_error = exc
                if self._remote_container_exited(record):
                    raise RuntimeError("policy-server container exited before becoming ready") from exc
                time.sleep(1)
        raise TimeoutError(f"policy server did not become ready: {last_error}")

    def _remote_container_exited(self, record: RolloutRecord) -> bool:
        if not record.inference_container or not self.cluster.status()["connected"]:
            return False
        with contextlib.suppress(Exception):
            code, stdout, _stderr = self.cluster.execute(
                f"docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(record.inference_container)}",
                timeout=10,
            )
            return code == 0 and stdout.strip().lower() == "false"
        return False

    @staticmethod
    def _local_container_exited(record: RolloutRecord) -> bool:
        if not record.inference_container:
            return False
        with contextlib.suppress(Exception):
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", record.inference_container],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0 and result.stdout.strip().lower() == "false"
        return False

    @staticmethod
    def _hardware_blockers() -> list[str]:
        return [
            "authoritative Alex state readback provider is not configured",
            "pelvis/world frame transform provider is not configured",
            "complete forearm, neck, and Ability Hand action sinks are not configured",
            "hardware watchdog/deadman acknowledgement is unavailable",
        ]

    def start(self, config: RolloutConfig) -> RolloutRecord:
        policy_ref, manifest = resolve_rollout_source(config, self.jobs)
        if config.inference_location == "remote" and Path(policy_ref).expanduser().is_dir():
            raise ValueError(
                "local policy_ref paths require local inference so the checkpoint can be mounted into Docker"
            )
        rollout_id = f"rollout-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        log_path = self._dir(rollout_id) / "rollout.log"
        record = RolloutRecord(
            id=rollout_id,
            state="starting",
            config=config,
            policy_ref=policy_ref,
            manifest=manifest,
            started_at=time.time(),
            log_path=str(log_path),
        )
        if config.target == "robot":
            record.state = "blocked"
            record.ended_at = time.time()
            record.blockers = self._hardware_blockers()
            record.error_message = "Real Alex rollout is safety-gated until every hardware capability is available"
            self._records[record.id] = record
            self._persist(record)
            return record

        if config.target == "sim":
            isaaclab_root = Path(config.isaaclab_root).expanduser().resolve()
            if not (isaaclab_root / "isaaclab.sh").is_file():
                raise FileNotFoundError(f"Isaac Lab launcher not found: {isaaclab_root / 'isaaclab.sh'}")
        else:
            self._require_arena_container(config.container_name)

        gpu_uuid = self._select_gpu(config.gpu) if config.inference_location == "remote" else config.gpu
        remote_port = 24000 + (int(uuid.uuid4().hex[:4], 16) % 16000)
        container = f"alex-rollout-{rollout_id}"[:63]
        record.inference_container = container
        record.inference_port = remote_port
        self._records[record.id] = record
        self._persist(record)

        tunnel = None
        handle = None
        try:
            if config.inference_location == "local":
                command = build_local_inference_container_command(rollout_id, policy_ref, gpu_uuid, remote_port)
                result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
                if result.returncode:
                    raise RuntimeError((result.stderr or result.stdout).strip() or "policy-server container failed to start")
                schema = self._wait_ready(remote_port, record=record)
                inference_port = remote_port
            else:
                command = build_inference_container_command(rollout_id, policy_ref, gpu_uuid, remote_port)
                code, _stdout, stderr = self.cluster.execute(command, timeout=60)
                if code:
                    raise RuntimeError(stderr.strip() or "policy-server container failed to start")
                tunnel = self.cluster.forward_remote_port(remote_port)
                self._tunnels[record.id] = tunnel
                schema = self._wait_policy_server_ready(record, tunnel.local_port)
                inference_port = tunnel.local_port
            manifest["policy_schema"] = schema
            artifact_dir = self._dir(record.id) / "artifacts"
            metrics_path = artifact_dir / "metrics.json"
            if (config.video or config.camera_video) and not config.video_dir:
                config.video_dir = str(artifact_dir / "videos")
            inference_url = f"http://127.0.0.1:{inference_port}"
            if config.target == "arena":
                command = build_arena_rollout_command(config, inference_url, manifest, str(metrics_path))
                cwd: str | None = None
            else:
                isaaclab_root = Path(config.isaaclab_root).expanduser().resolve()
                command = build_isaaclab_rollout_command(config, inference_url, manifest, str(metrics_path))
                cwd = str(isaaclab_root)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a")
            child_env = os.environ.copy()
            child_env.pop("CONDA_PREFIX", None)
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=child_env,
            )
            record.pid = process.pid
            record.state = "running"
            with self._lock:
                self._processes[record.id] = process
                self._handles[record.id] = handle
            self._persist(record)
            return record
        except Exception as exc:
            if handle is not None:
                handle.close()
            diagnostics = (
                self._local_container_diagnostics(record)
                if config.inference_location == "local"
                else self._remote_container_diagnostics(record)
            )
            if tunnel is not None:
                tunnel.close()
            self._tunnels.pop(record.id, None)
            if config.inference_location == "local":
                self._stop_local(record)
            else:
                self._stop_remote(record)
            record.state = "failed"
            record.ended_at = time.time()
            record.error_message = str(exc) if not diagnostics else f"{exc}\n\n{diagnostics}"
            self._persist(record)
            raise RuntimeError(record.error_message) from exc

    @staticmethod
    def _require_arena_container(container_name: str) -> None:
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"unable to inspect Arena container {container_name!r}: {exc}") from exc
        if result.returncode != 0 or result.stdout.strip().lower() != "true":
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"Arena container {container_name!r} is not running"
                + (f": {detail}" if detail else "")
            )

    def _remote_container_diagnostics(self, record: RolloutRecord) -> str:
        if not record.inference_container or not self.cluster.status()["connected"]:
            return ""
        quoted = shlex.quote(record.inference_container)
        parts: list[str] = []
        with contextlib.suppress(Exception):
            _code, stdout, stderr = self.cluster.execute(
                f"docker ps -a --filter name=^{quoted}$ --format '{{{{.Names}}}} {{{{.Status}}}}'",
                timeout=20,
            )
            status = (stdout + stderr).strip()
            if status:
                parts.append(f"container status:\n{status}")
        with contextlib.suppress(Exception):
            _code, stdout, stderr = self.cluster.execute(f"docker logs --tail 200 {quoted}", timeout=30)
            logs = (stdout + stderr).strip()
            if logs:
                parts.append(f"container logs:\n{logs}")
        return "\n\n".join(parts)

    def _local_container_diagnostics(self, record: RolloutRecord) -> str:
        if not record.inference_container:
            return ""
        parts: list[str] = []
        with contextlib.suppress(Exception):
            status = subprocess.run(
                [
                    "docker", "ps", "-a",
                    "--filter", f"name=^{record.inference_container}$",
                    "--format", "{{.Names}} {{.Status}}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            text = (status.stdout + status.stderr).strip()
            if text:
                parts.append(f"container status:\n{text}")
        with contextlib.suppress(Exception):
            logs = subprocess.run(
                ["docker", "logs", "--tail", "200", record.inference_container],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            text = (logs.stdout + logs.stderr).strip()
            if text:
                parts.append(f"container logs:\n{text}")
        return "\n\n".join(parts)

    def _stop_remote(self, record: RolloutRecord) -> None:
        if record.inference_container and self.cluster.status()["connected"]:
            with contextlib.suppress(Exception):
                self.cluster.execute(
                    f"docker rm --force {shlex.quote(record.inference_container)}", timeout=20
                )

    @staticmethod
    def _stop_local(record: RolloutRecord) -> None:
        if record.inference_container:
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["docker", "rm", "--force", record.inference_container],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )

    def _cleanup(self, record: RolloutRecord) -> None:
        handle = self._handles.pop(record.id, None)
        if handle is not None:
            handle.close()
        tunnel = self._tunnels.pop(record.id, None)
        if tunnel is not None:
            tunnel.close()
        if record.config.inference_location == "local":
            self._stop_local(record)
        else:
            self._stop_remote(record)

    def get(self, rollout_id: str) -> RolloutRecord:
        record = self._records.get(rollout_id)
        if record is None:
            raise KeyError(rollout_id)
        process = self._processes.get(rollout_id)
        if record.state == "running" and process is not None and process.poll() is not None:
            record.exit_code = process.returncode
            record.ended_at = time.time()
            artifact_dir = self._dir(record.id) / "artifacts"
            if artifact_dir.is_dir():
                record.artifacts = sorted(str(path) for path in artifact_dir.rglob("*") if path.is_file())
            metrics_path = artifact_dir / "metrics.json"
            error_path = artifact_dir / "error.txt"
            if metrics_path.is_file():
                with contextlib.suppress(Exception):
                    record.metrics = json.loads(metrics_path.read_text())
            # SimulationApp.close() can hard-exit 0 even after a Python exception.
            # Prefer an explicit error artifact / missing metrics over a bare exit code for
            # the local Isaac Lab runner. Arena policy_runner does not write metrics.json.
            if error_path.is_file():
                record.state = "failed"
                record.error_message = error_path.read_text(errors="replace").strip() or "Isaac Lab rollout failed"
            elif process.returncode == 0 and (
                record.config.target == "arena" or metrics_path.is_file()
            ):
                record.state = "done"
            elif process.returncode == 0:
                record.state = "failed"
                record.error_message = (
                    "Isaac Lab exited before writing metrics (often gym.make without cfg=parse_env_cfg(...))"
                )
            else:
                record.state = "failed"
            self._processes.pop(rollout_id, None)
            self._cleanup(record)
            self._persist(record)
        return record

    def list(self) -> list[RolloutRecord]:
        return sorted((self.get(item) for item in list(self._records)), key=lambda r: r.started_at, reverse=True)

    def reattach(self) -> list[RolloutRecord]:
        """Reconcile persisted rollouts after reconnecting to the GPU host.

        Isaac Lab is a local process, so it cannot be safely resumed after the Lab
        server exits.  Persisted active rollouts are marked interrupted during
        startup; once SSH is available again, remove their orphaned inference
        containers so they cannot keep occupying a GPU.
        """
        for record in self._records.values():
            if record.state != "interrupted" or not record.inference_container:
                continue
            if record.config.inference_location == "local":
                self._stop_local(record)
            elif self.cluster.status()["connected"]:
                self._stop_remote(record)
        return self.list()

    def logs(self, rollout_id: str) -> str:
        record = self.get(rollout_id)
        local = Path(record.log_path).read_text(errors="replace") if Path(record.log_path).is_file() else ""
        container = ""
        if record.inference_container and record.config.inference_location == "local":
            with contextlib.suppress(Exception):
                result = subprocess.run(
                    ["docker", "logs", "--tail", "1000", record.inference_container],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                container = result.stdout + result.stderr
        elif record.inference_container and self.cluster.status()["connected"]:
            with contextlib.suppress(Exception):
                _code, stdout, stderr = self.cluster.execute(
                    f"docker logs --tail 1000 {shlex.quote(record.inference_container)}"
                )
                container = stdout + stderr
        return "\n".join(part for part in (container, local) if part)

    def stop(self, rollout_id: str) -> RolloutRecord:
        record = self.get(rollout_id)
        if record.state != "running":
            raise RuntimeError("rollout is not running")
        process = self._processes.pop(rollout_id, None)
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


_DEFAULT_ROOT = Path(os.environ.get("ALEX_ROLLOUT_ROOT", Path.home() / ".cache/alex-lab/rollouts"))
rollout_manager = RolloutManager(_DEFAULT_ROOT)
