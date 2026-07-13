"""Standalone FastAPI application for Alex Lab.

This module deliberately avoids importing LeLab's SO-101 hardware stack so the
Alex UI can start without a local LeRobot installation.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .alex_models import (
    DatasetConversionConfig,
    DatasetInspectRequest,
    EvaluationConfig,
    RemoteTrainingRequest,
    build_dataset_conversion_command,
)
from .cluster import ClusterConnectRequest, HostKeyVerificationError, cluster_manager
from .datasets import inspect_dataset, list_all_datasets
from .evaluation import evaluation_manager
from .policies import get_training_capabilities
from .remote_jobs import remote_job_manager

logger = logging.getLogger(__name__)
app = FastAPI(title="Alex Lab", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": "alex-lab"}


@app.get("/alex/setup")
def setup(arena_root: str = "/home/bpratt/IsaacLab-Arena") -> dict:
    root = Path(arena_root).expanduser()
    local = {
        "arena_root": root.is_dir(),
        "evaluation_runner": (root / "isaaclab_arena/evaluation/policy_runner.py").is_file(),
        "gr00t_converter": (
            root / "isaaclab_arena_gr00t/lerobot/convert_lerobot_v3_to_gr00t.py"
        ).is_file(),
        "ccil_converter": (
            root / "isaaclab_arena_ccil/data/convert_lerobot_to_ccil.py"
        ).is_file(),
    }
    cluster = cluster_manager.status()
    remote = cluster_manager.setup_checks() if cluster["connected"] else None
    return {
        "ready": bool(remote and remote["ready"]),
        "arena_root": str(root),
        "local": local,
        "cluster": cluster,
        "remote": remote,
    }


@app.post("/alex/cluster/connect")
def connect(body: ClusterConnectRequest) -> dict:
    try:
        status = cluster_manager.connect(body)
        remote_job_manager.reattach()
        return {**status, "user": status.get("username")}
    except HostKeyVerificationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("SSH connection failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"SSH connection failed: {exc}") from exc


@app.post("/alex/cluster/disconnect")
def disconnect() -> dict:
    status = cluster_manager.disconnect()
    return {**status, "user": status.get("username")}


@app.get("/alex/cluster/status")
def cluster_status() -> dict:
    status = cluster_manager.status()
    return {**status, "user": status.get("username")}


@app.get("/alex/cluster/gpus")
def cluster_gpus() -> list[dict]:
    try:
        gpus = cluster_manager.gpus()
    except ConnectionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    reservations = remote_job_manager.reservations()
    for gpu in gpus:
        gpu["reserved_by"] = reservations.get(gpu["uuid"])
        gpu["utilization"] = gpu["utilization_percent"]
        gpu["power_w"] = gpu["power_draw_w"]
        gpu["occupied"] = bool(gpu["processes"]) or gpu["reserved_by"] is not None
        for process in gpu["processes"]:
            process["memory_mb"] = process["memory_used_mb"]
    return gpus


@app.post("/alex/datasets/inspect")
def dataset_inspect(body: DatasetInspectRequest) -> dict:
    try:
        return inspect_dataset(path=body.path, repo_id=body.repo_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Dataset inspection failed: {exc}") from exc


@app.get("/alex/datasets")
def dataset_list() -> list[dict]:
    """Hub-owned datasets available to the remote trainer."""
    return [item for item in list_all_datasets() if item.get("source") in {"hub", "both"}]


@app.get("/alex/training/capabilities")
def training_capabilities(dataset_repo_id: str | None = None) -> dict:
    try:
        return get_training_capabilities(dataset_repo_id=dataset_repo_id)
    except ConnectionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("Policy capability discovery failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Policy discovery failed: {exc}") from exc


@app.post("/alex/datasets/convert")
async def dataset_convert(request: Request) -> dict:
    raw = await request.json()
    try:
        if "source" in raw:
            format_name = "gr00t" if raw.get("format") in {"groot", "gr00t"} else raw.get("format")
            root = Path(raw.get("arena_root", "/home/bpratt/IsaacLab-Arena"))
            converted = {
                "format": format_name,
                "input_path": raw["source"],
                "output_path": raw["destination"],
                "arena_root": str(root),
            }
            if format_name == "gr00t":
                converted["modality_template"] = raw.get("modality_template") or str(
                    root
                    / "isaaclab_arena_gr00t/embodiments/alex/alex_test_obs_new_modality.json"
                )
            body = DatasetConversionConfig.model_validate(converted)
        else:
            body = DatasetConversionConfig.model_validate(raw)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    command = build_dataset_conversion_command(body)
    return {
        "source": body.input_path or body.repo_id,
        "destination": body.output_path,
        "status": "ready",
        "message": "Validated conversion command. Run it in the configured Arena environment.",
        "command": command,
        "shell_command": shlex.join(command),
    }


def _job_response(record) -> dict:
    state_map = {"done": "completed", "unknown": "queued"}
    return {
        **record.model_dump(),
        "method": "groot" if record.kind == "gr00t" else record.kind,
        "status": state_map.get(record.state, record.state),
        "gpus": [int(item) for item in record.gpu_ids if item.isdigit()],
        "error": record.error_message,
        "created_at": record.started_at,
        "finished_at": record.ended_at,
    }


@app.post("/alex/training", status_code=201)
def start_training(body: RemoteTrainingRequest) -> dict:
    try:
        capabilities = get_training_capabilities(dataset_repo_id=body.config.dataset_repo_id)
        policy = next(
            (item for item in capabilities["policies"] if item["type"] == body.config.policy_type),
            None,
        )
        if policy is None:
            raise ValueError(f"Policy '{body.config.policy_type}' is not provided by the training image")
        if not policy["available"]:
            raise ValueError(policy.get("unavailable_reason") or "Policy dependencies are unavailable")
        if not policy["compatible"]:
            raise ValueError(policy.get("compatibility_reason") or "Policy is incompatible with this dataset")
        if (
            body.config.policy_type == "groot"
            and body.config.policy_use_relative_actions
            and not capabilities.get("groot_relative_actions_ready")
        ):
            reason = capabilities.get("groot_relative_actions_reason") or (
                "the dataset does not have compatible chunked action metadata"
            )
            raise ValueError(
                f"GR00T relative actions are unavailable: {reason} Disable 'Use relative actions' "
                "and launch again."
            )
        return _job_response(remote_job_manager.start(body))
    except ConnectionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Remote training launch failed")
        raise HTTPException(status_code=502, detail=f"Remote launch failed: {exc}") from exc


@app.get("/alex/jobs")
def jobs() -> list[dict]:
    return [_job_response(record) for record in remote_job_manager.list()]


@app.post("/alex/jobs/reattach")
def reattach_jobs() -> dict:
    return {"jobs": [_job_response(record) for record in remote_job_manager.reattach()]}


@app.get("/alex/jobs/{job_id}")
def job(job_id: str) -> dict:
    try:
        return _job_response(remote_job_manager.get(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Training job not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not refresh job: {exc}") from exc


@app.get("/alex/jobs/{job_id}/logs")
def job_logs(job_id: str, tail: int = 2000) -> dict:
    try:
        logs = (
            remote_job_manager.logs(job_id, tail)
            if cluster_manager.status()["connected"]
            else remote_job_manager.persisted_logs(job_id)
        )
        return {"logs": logs}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Training job not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not read logs: {exc}") from exc


@app.post("/alex/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict:
    try:
        return _job_response(remote_job_manager.stop(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Training job not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _evaluation_response(record) -> dict:
    return {
        **record.model_dump(),
        "status": "completed" if record.state == "done" else record.state,
        "output_path": record.config.video_dir,
    }


@app.post("/alex/evaluations", status_code=201)
def start_evaluation(body: EvaluationConfig) -> dict:
    try:
        return _evaluation_response(evaluation_manager.start(body))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Evaluation launch failed")
        raise HTTPException(status_code=500, detail=f"Evaluation launch failed: {exc}") from exc


@app.get("/alex/evaluations")
def evaluations() -> dict:
    return {"evaluations": [_evaluation_response(item) for item in evaluation_manager.list()]}


@app.get("/alex/evaluations/{evaluation_id}")
def evaluation(evaluation_id: str) -> dict:
    try:
        return _evaluation_response(evaluation_manager.get(evaluation_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Evaluation not found") from exc


@app.get("/alex/evaluations/{evaluation_id}/logs")
def evaluation_logs(evaluation_id: str) -> dict:
    try:
        return {"logs": evaluation_manager.logs(evaluation_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Evaluation not found") from exc


@app.post("/alex/evaluations/{evaluation_id}/stop")
def stop_evaluation(evaluation_id: str) -> dict:
    try:
        return _evaluation_response(evaluation_manager.stop(evaluation_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Evaluation not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.is_dir():
    assets = _DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        requested = (_DIST / path).resolve()
        if requested.is_file() and _DIST.resolve() in requested.parents:
            return FileResponse(requested)
        return FileResponse(_DIST / "index.html")
