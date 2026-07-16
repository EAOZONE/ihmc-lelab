"""Remote offline dataset evaluation for LeRobot policies."""

from __future__ import annotations

import json
import os
import re
import shlex
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .alex_models import DatasetEvalConfig
from .cluster import ClusterManager, cluster_manager
from .remote_jobs import RemoteJobManager, remote_hf_token_prelude, remote_job_manager, remote_training_image

DatasetEvalState = Literal["running", "done", "failed", "stopped", "unknown"]
_CONTAINER_SAFE = re.compile(r"[^a-z0-9_.-]+")


class DatasetEvalRecord(BaseModel):
    id: str
    state: DatasetEvalState
    container_name: str
    container_id: str | None = None
    gpu_id: str
    gpu_uuid: str
    config: DatasetEvalConfig
    policy_ref: str
    dataset_repo_id: str
    started_at: float
    ended_at: float | None = None
    exit_code: int | None = None
    log_path: str
    metrics: dict = Field(default_factory=dict)
    error_message: str | None = None


_DATASET_EVAL_SCRIPT = r'''
import argparse
import json
import re
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn


def resolve_policy_ref(ref: str) -> str:
    path = Path(ref).expanduser()
    if path.is_dir():
        return str(path)
    if "@checkpoints/" in ref:
        repo, step = ref.split("@checkpoints/", 1)
        root = snapshot_download(
            repo_id=repo,
            repo_type="model",
            allow_patterns=[f"checkpoints/{step}/pretrained_model/*"],
        )
        return str(Path(root) / "checkpoints" / step / "pretrained_model")
    if ref.endswith("@latest"):
        repo = ref.removesuffix("@latest")
        steps = sorted(
            {
                match.group(1)
                for item in HfApi().model_info(repo).siblings
                if (match := re.match(r"checkpoints/([^/]+)/pretrained_model/config.json$", item.rfilename))
            }
        )
        if steps:
            return resolve_policy_ref(f"{repo}@checkpoints/{steps[-1]}")
        ref = repo
    return snapshot_download(repo_id=ref, repo_type="model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-ref", required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--dataset-episodes", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    print(
        "ALEX_DATASET_EVAL_STATUS="
        + json.dumps(
            {
                "stage": "starting",
                "policy_ref": args.policy_ref,
                "dataset_repo_id": args.dataset_repo_id,
                "dataset_revision": args.dataset_revision,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    episodes = json.loads(args.dataset_episodes)
    print(
        "ALEX_DATASET_EVAL_STATUS="
        + json.dumps({"stage": "resolving_policy", "num_episodes": len(episodes)}, separators=(",", ":")),
        flush=True,
    )
    policy_path = resolve_policy_ref(args.policy_ref)
    print(
        "ALEX_DATASET_EVAL_STATUS="
        + json.dumps({"stage": "loading_policy", "resolved_policy_path": policy_path}, separators=(",", ":")),
        flush=True,
    )
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(policy_path)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device(args.device)
    policy.to(device, dtype=dtype).eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        "ALEX_DATASET_EVAL_STATUS="
        + json.dumps({"stage": "loading_dataset", "policy_type": policy_cfg.type}, separators=(",", ":")),
        flush=True,
    )
    ds_meta = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        revision=args.dataset_revision,
    )
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        revision=args.dataset_revision,
        video_backend="pyav",
        return_uint8=True,
    )
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "device_processor": {
                "device": str(device),
                "float_dtype": args.dtype,
            }
        },
    )
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )
    print(
        "ALEX_DATASET_EVAL_STATUS="
        + json.dumps(
            {
                "stage": "evaluating",
                "num_frames": dataset.num_frames,
                "num_episodes": dataset.num_episodes,
                "batch_size": args.batch_size,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    total_loss = 0.0
    total_batches = 0
    total_samples = 0
    with torch.inference_mode():
        for batch in loader:
            batch_size = None
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    batch_size = int(value.shape[0])
                    break
            for cam_key in dataset.meta.camera_keys:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32).div_(255.0)
            processed = preprocessor(batch)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=args.dtype == "bfloat16"):
                loss, _outputs = policy.forward(processed)
            current_batch_size = batch_size or 0
            total_loss += float(loss.detach().cpu()) * current_batch_size
            total_batches += 1
            total_samples += current_batch_size
            if total_batches == 1 or total_batches % 10 == 0:
                print(
                    "ALEX_DATASET_EVAL_STATUS="
                    + json.dumps(
                        {
                            "stage": "evaluating",
                            "batches": total_batches,
                            "samples": total_samples,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

    result = {
        "eval_loss": total_loss / max(total_samples, 1),
        "num_batches": total_batches,
        "num_samples": total_samples,
        "num_frames": dataset.num_frames,
        "num_episodes": dataset.num_episodes,
        "episodes": episodes,
        "policy_ref": args.policy_ref,
        "resolved_policy_path": policy_path,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_revision": args.dataset_revision,
    }
    print("ALEX_DATASET_EVAL_RESULT=" + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
'''


def resolve_dataset_eval_source(config: DatasetEvalConfig, jobs: RemoteJobManager) -> tuple[str, str, str]:
    dataset_repo_id = config.dataset_repo_id
    dataset_revision = config.dataset_revision
    if config.job_id:
        job = jobs.get(config.job_id, refresh=False)
        repo = job.config.get("model_repo_id") or job.config.get("policy_repo_id")
        if not repo:
            raise ValueError(f"job {config.job_id!r} has no model repository")
        dataset_repo_id = dataset_repo_id or job.config.get("dataset_repo_id")
        dataset_revision = dataset_revision or job.config.get("dataset_revision")
        if config.checkpoint == "latest":
            policy_ref = f"{repo}@latest"
        elif config.checkpoint.isdigit():
            policy_ref = f"{repo}@checkpoints/{int(config.checkpoint):06d}"
        else:
            raise ValueError("checkpoint must be 'latest' or a numeric training step")
    else:
        if config.policy_ref is None:
            raise ValueError("policy_ref is required")
        policy_ref = config.policy_ref
        if "@" not in policy_ref and not Path(policy_ref).expanduser().exists():
            if config.checkpoint == "latest":
                policy_ref = f"{policy_ref}@latest"
            elif config.checkpoint.isdigit():
                policy_ref = f"{policy_ref}@checkpoints/{int(config.checkpoint):06d}"
            else:
                raise ValueError("checkpoint must be 'latest' or a numeric training step")
    if not dataset_repo_id:
        raise ValueError("dataset_repo_id is required")
    return policy_ref, dataset_repo_id, dataset_revision or "main"


def build_dataset_eval_docker_command(
    eval_id: str,
    config: DatasetEvalConfig,
    policy_ref: str,
    dataset_repo_id: str,
    dataset_revision: str,
    gpu_uuid: str,
    container_name: str | None = None,
    image: str | None = None,
) -> str:
    if not gpu_uuid:
        raise ValueError("a GPU is required")
    container_name = container_name or f"alex-dataset-eval-{eval_id}"
    argv = [
        "docker",
        "run",
        "--detach",
        "--name",
        container_name,
        "--gpus",
        f'"device={gpu_uuid}"',
        "--shm-size",
        "32g",
        "--label",
        f"alex.dataset_eval_id={eval_id}",
        "--env",
        "HF_TOKEN",
        "--env",
        "HF_HOME=/cache/huggingface",
        "--env",
        "PYTHONUNBUFFERED=1",
        "--volume",
        "alex_hf_cache:/cache/huggingface",
        image or remote_training_image(),
        "python3",
        "-c",
        _DATASET_EVAL_SCRIPT,
        "--policy-ref",
        policy_ref,
        "--dataset-repo-id",
        dataset_repo_id,
        "--dataset-revision",
        dataset_revision,
        "--dataset-episodes",
        json.dumps(config.dataset_episodes),
        "--batch-size",
        str(config.batch_size),
        "--num-workers",
        str(config.num_workers),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16" if config.policy_use_bf16 else "float32",
    ]
    return remote_hf_token_prelude() + shlex.join(argv)


class DatasetEvalManager:
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
        self._records: dict[str, DatasetEvalRecord] = {}
        self._load()

    def _record_path(self, eval_id: str) -> Path:
        return self.root / eval_id / "dataset_eval.json"

    def _log_path(self, eval_id: str) -> Path:
        return self.root / eval_id / "docker.log"

    def _persist(self, record: DatasetEvalRecord) -> None:
        path = self._record_path(record.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(record.model_dump_json(indent=2))
        os.replace(tmp, path)

    def _load(self) -> None:
        for path in self.root.glob("*/dataset_eval.json"):
            try:
                record = DatasetEvalRecord.model_validate_json(path.read_text())
            except Exception:
                continue
            self._records[record.id] = record

    def _select_gpu(self, identifier: str) -> tuple[str, str]:
        for gpu in self.cluster.gpus():
            if identifier in {str(gpu["index"]), gpu["uuid"]}:
                return str(gpu["index"]), str(gpu["uuid"])
        raise ValueError(f"unknown GPU identifier: {identifier}")

    @staticmethod
    def _parse_metrics(logs: str) -> dict:
        for line in reversed(logs.splitlines()):
            if "ALEX_DATASET_EVAL_RESULT=" not in line:
                continue
            payload = line.split("ALEX_DATASET_EVAL_RESULT=", 1)[1].strip()
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return {}
        return {}

    def start(self, config: DatasetEvalConfig) -> DatasetEvalRecord:
        if not self.cluster.status()["connected"]:
            raise ConnectionError("Alex cluster is not connected")
        gpu_id, gpu_uuid = self._select_gpu(config.gpu)
        policy_ref, dataset_repo_id, dataset_revision = resolve_dataset_eval_source(config, self.jobs)
        eval_id = f"dataset-eval-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        safe = _CONTAINER_SAFE.sub("-", eval_id.lower()).strip("-")[:48]
        container_name = f"alex-{safe}"
        command = build_dataset_eval_docker_command(
            eval_id,
            config,
            policy_ref,
            dataset_repo_id,
            dataset_revision,
            gpu_uuid,
            container_name=container_name,
        )
        code, stdout, stderr = self.cluster.execute(command, timeout=60)
        if code:
            raise RuntimeError(stderr.strip() or stdout.strip() or "dataset eval container failed to start")
        record = DatasetEvalRecord(
            id=eval_id,
            state="running",
            container_name=container_name,
            container_id=stdout.strip() or None,
            gpu_id=gpu_id,
            gpu_uuid=gpu_uuid,
            config=config,
            policy_ref=policy_ref,
            dataset_repo_id=dataset_repo_id,
            started_at=time.time(),
            log_path=str(self._log_path(eval_id)),
        )
        self._records[record.id] = record
        self._persist(record)
        return record

    def get(self, eval_id: str, refresh: bool = True) -> DatasetEvalRecord:
        record = self._records.get(eval_id)
        if record is None:
            raise KeyError(eval_id)
        if refresh and self.cluster.status()["connected"] and record.state in {"running", "unknown"}:
            return self.refresh(eval_id)
        return record

    def refresh(self, eval_id: str) -> DatasetEvalRecord:
        record = self.get(eval_id, refresh=False)
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
                logs = self.logs(eval_id)
                record.metrics = self._parse_metrics(logs)
        self._persist(record)
        return record

    def logs(self, eval_id: str, tail: int = 2000) -> str:
        record = self.get(eval_id, refresh=False)
        tail = min(max(tail, 1), 10000)
        command = f"docker logs --timestamps --tail {tail} {shlex.quote(record.container_name)}"
        _code, stdout, stderr = self.cluster.execute(command)
        logs = stdout + stderr
        path = Path(record.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(logs)
        metrics = self._parse_metrics(logs)
        if metrics:
            record.metrics = metrics
            self._persist(record)
        return logs

    def persisted_logs(self, eval_id: str) -> str:
        record = self.get(eval_id, refresh=False)
        path = Path(record.log_path)
        return path.read_text() if path.is_file() else ""

    def list(self) -> list[DatasetEvalRecord]:
        return sorted((self.get(item) for item in list(self._records)), key=lambda item: item.started_at, reverse=True)

    def stop(self, eval_id: str) -> DatasetEvalRecord:
        record = self.get(eval_id, refresh=False)
        if record.state != "running":
            raise RuntimeError("dataset eval is not running")
        code, _stdout, stderr = self.cluster.execute(
            f"docker stop --time 10 {shlex.quote(record.container_name)}",
            timeout=20,
        )
        if code:
            raise RuntimeError(stderr.strip() or "docker stop failed")
        record.state = "stopped"
        record.ended_at = time.time()
        self._persist(record)
        return record

    def reattach(self) -> list[DatasetEvalRecord]:
        if not self.cluster.status()["connected"]:
            return self.list()
        for record in self.list():
            if record.state in {"running", "unknown"}:
                try:
                    self.refresh(record.id)
                except Exception:
                    continue
        return self.list()


_DEFAULT_DATASET_EVAL_ROOT = Path(
    os.environ.get(
        "ALEX_DATASET_EVAL_ROOT",
        Path.home() / ".cache" / "alex-lab" / "dataset-evals",
    )
)
dataset_eval_manager = DatasetEvalManager(_DEFAULT_DATASET_EVAL_ROOT)
