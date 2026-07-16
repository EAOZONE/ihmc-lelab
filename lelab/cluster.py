"""Session-only SSH access and GPU telemetry for an Alex training host."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import logging
import select
import shlex
import socket
import threading
from typing import Any

import paramiko
from pydantic import BaseModel, Field

from .alex_models import DEFAULT_ARENA_ROOT

GPU_QUERY_FIELDS = (
    "index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
)
GPU_QUERY_COMMAND = f"nvidia-smi --query-gpu={GPU_QUERY_FIELDS} --format=csv,noheader,nounits"
PROCESS_QUERY_COMMAND = (
    "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits"
)
logger = logging.getLogger(__name__)


class ClusterConnectRequest(BaseModel):
    host: str = Field(min_length=1, max_length=255)
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, exclude=True)
    port: int = Field(default=22, ge=1, le=65535)
    expected_fingerprint: str | None = None
    timeout_seconds: float = Field(default=10, gt=0, le=60)


class HostKeyVerificationError(RuntimeError):
    pass


class SshTunnel:
    """Small loopback-only local forward over an existing Paramiko transport."""

    def __init__(self, transport: paramiko.Transport, remote_host: str, remote_port: int) -> None:
        self._transport = transport
        self._remote = (remote_host, remote_port)
        self._stop = threading.Event()
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self._listener.settimeout(0.5)
        self.local_port = int(self._listener.getsockname()[1])
        self._thread = threading.Thread(target=self._accept, name="alex-ssh-forward", daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                client, address = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=self._bridge, args=(client, address), daemon=True).start()

    def _bridge(self, client: socket.socket, address: tuple[str, int]) -> None:
        channel = None
        try:
            try:
                channel = self._transport.open_channel("direct-tcpip", self._remote, address)
            except paramiko.SSHException as exc:
                # This commonly happens during readiness polling while the
                # remote policy server container is still starting or has
                # already failed. The caller owns the timeout and diagnostics;
                # avoid one traceback per poll attempt.
                logger.debug("SSH tunnel could not connect to %s:%s: %s", *self._remote, exc)
                return
            if channel is None:
                return
            while not self._stop.is_set():
                readable, _, _ = select.select([client, channel], [], [], 0.5)
                for source in readable:
                    data = source.recv(1024 * 1024)
                    if not data:
                        return
                    (channel if source is client else client).sendall(data)
                if channel.closed:
                    break
        finally:
            client.close()
            if channel is not None:
                channel.close()

    def close(self) -> None:
        self._stop.set()
        self._listener.close()
        self._thread.join(timeout=2)


def sha256_fingerprint(key: paramiko.PKey) -> str:
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def normalize_fingerprint(value: str) -> str:
    value = value.strip()
    if not value.startswith("SHA256:") or len(value) <= len("SHA256:"):
        raise ValueError("expected_fingerprint must use SHA256:<base64> format")
    return value.rstrip("=")


class FingerprintPolicy(paramiko.MissingHostKeyPolicy):
    """Accept an otherwise unknown host only when its SHA256 key matches."""

    def __init__(self, expected: str) -> None:
        self.expected = normalize_fingerprint(expected)

    def missing_host_key(self, client: paramiko.SSHClient, hostname: str, key: paramiko.PKey) -> None:
        actual = sha256_fingerprint(key)
        if actual != self.expected:
            raise HostKeyVerificationError(
                f"host key mismatch for {hostname}: expected {self.expected}, received {actual}"
            )
        # Intentionally do not save the key: trust is session-only.


def _number(value: str, cast: type[int] | type[float]) -> int | float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[NOT SUPPORTED]", "NOT SUPPORTED"}:
        return None
    try:
        return cast(value)
    except ValueError:
        return None


def parse_nvidia_smi_gpus(gpu_output: str, process_output: str = "") -> list[dict[str, Any]]:
    """Parse output from the fixed CSV queries used by :class:`ClusterManager`."""
    process_map: dict[str, list[dict[str, Any]]] = {}
    for row in csv.reader(io.StringIO(process_output)):
        if len(row) < 4:
            continue
        uuid, pid, name, memory = (cell.strip() for cell in row[:4])
        if not uuid or "no running processes" in uuid.lower():
            continue
        process_map.setdefault(uuid, []).append(
            {
                "pid": _number(pid, int),
                "name": name,
                "memory_used_mb": _number(memory, int),
            }
        )

    result = []
    for row in csv.reader(io.StringIO(gpu_output)):
        if len(row) < 9:
            continue
        index, uuid, name, util, used, total, temp, power, power_limit = (cell.strip() for cell in row[:9])
        result.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "utilization_percent": _number(util, int),
                "memory_used_mb": _number(used, int),
                "memory_total_mb": _number(total, int),
                "temperature_c": _number(temp, int),
                "power_draw_w": _number(power, float),
                "power_limit_w": _number(power_limit, float),
                "processes": process_map.get(uuid, []),
            }
        )
    return result


class ClusterManager:
    """Own one Paramiko connection; credentials never leave process memory."""

    def __init__(self) -> None:
        self._client: paramiko.SSHClient | None = None
        self._host: str | None = None
        self._username: str | None = None
        self._port: int | None = None
        self._fingerprint: str | None = None
        self._lock = threading.RLock()

    def connect(self, request: ClusterConnectRequest) -> dict[str, Any]:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        if request.expected_fingerprint:
            client.set_missing_host_key_policy(FingerprintPolicy(request.expected_fingerprint))
        else:
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect(
                hostname=request.host,
                port=request.port,
                username=request.username,
                password=request.password,
                timeout=request.timeout_seconds,
                banner_timeout=request.timeout_seconds,
                auth_timeout=request.timeout_seconds,
                allow_agent=False,
                look_for_keys=False,
            )
            transport = client.get_transport()
            if transport is None or not transport.is_active():
                raise ConnectionError("SSH transport did not become active")
            key = transport.get_remote_server_key()
            actual = sha256_fingerprint(key)
            if request.expected_fingerprint and actual != normalize_fingerprint(request.expected_fingerprint):
                raise HostKeyVerificationError(
                    f"host key mismatch: expected {request.expected_fingerprint}, received {actual}"
                )
        except Exception:
            client.close()
            raise

        with self._lock:
            old = self._client
            self._client = client
            self._host = request.host
            self._username = request.username
            self._port = request.port
            self._fingerprint = actual
        if old is not None:
            old.close()
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        with self._lock:
            client, self._client = self._client, None
            self._host = self._username = self._fingerprint = None
            self._port = None
        if client is not None:
            client.close()
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            transport = self._client.get_transport() if self._client is not None else None
            connected = bool(transport and transport.is_active())
            return {
                "connected": connected,
                "host": self._host if connected else None,
                "username": self._username if connected else None,
                "port": self._port if connected else None,
                "fingerprint": self._fingerprint if connected else None,
            }

    def execute(self, command: str, timeout: float = 30) -> tuple[int, str, str]:
        with self._lock:
            client = self._client
        if client is None or not self.status()["connected"]:
            raise ConnectionError("cluster is not connected")
        _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        return exit_code, stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")

    def forward_remote_port(self, remote_port: int, remote_host: str = "127.0.0.1") -> SshTunnel:
        """Forward an ephemeral local loopback port to the connected host."""
        with self._lock:
            transport = self._client.get_transport() if self._client is not None else None
        if transport is None or not transport.is_active():
            raise ConnectionError("cluster is not connected")
        return SshTunnel(transport, remote_host, remote_port)

    def gpus(self) -> list[dict[str, Any]]:
        code, stdout, stderr = self.execute(GPU_QUERY_COMMAND)
        if code:
            raise RuntimeError(f"nvidia-smi GPU query failed: {stderr.strip()}")
        process_code, process_stdout, _ = self.execute(PROCESS_QUERY_COMMAND)
        if process_code:
            process_stdout = ""
        return parse_nvidia_smi_gpus(stdout, process_stdout)

    def setup_checks(self, arena_root: str = str(DEFAULT_ARENA_ROOT)) -> dict[str, Any]:
        from .remote_jobs import remote_hf_token_prelude, remote_training_image

        checks = {
            "docker": "command -v docker >/dev/null 2>&1",
            "docker_daemon": "docker info >/dev/null 2>&1",
            "nvidia_smi": "command -v nvidia-smi >/dev/null 2>&1",
            "nvidia_container_runtime": "docker info --format '{{json .Runtimes}}' | grep -q nvidia",
            "training_image": (
                f"docker image inspect {shlex.quote(remote_training_image())} >/dev/null 2>&1"
            ),
            "rollout_runtime": (
                f"docker run --rm {shlex.quote(remote_training_image())} "
                "test -f /opt/alex/alex_policy_server.py"
            ),
            "huggingface_login": remote_hf_token_prelude() + "true",
            "writable_home": 'test -w "$HOME"',
        }
        results = {}
        for name, command in checks.items():
            code, _stdout, stderr = self.execute(command)
            results[name] = {"ok": code == 0, "message": stderr.strip() or None}
        return {"ready": all(item["ok"] for item in results.values()), "checks": results}


cluster_manager = ClusterManager()
