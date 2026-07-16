"""Remote LeRobot policy discovery and dataset compatibility checks."""

from __future__ import annotations

import json
import shlex
import time
from typing import Any

from .cluster import ClusterManager, cluster_manager
from .remote_jobs import remote_training_image, with_remote_hf_token

_CACHE_TTL_SECONDS = 300
_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}

_LABELS = {
    "act": "ACT",
    "diffusion": "Diffusion Policy",
    "eo1": "EO-1",
    "evo1": "EVO1",
    "fastwam": "FastWAM",
    "gaussian_actor": "Gaussian Actor",
    "groot": "GR00T N1.7",
    "lingbot_va": "LingBot-VA",
    "multi_task_dit": "Multi-task DiT",
    "pi0": "π₀",
    "pi0_fast": "π₀-FAST",
    "pi05": "π₀.₅",
    "smolvla": "SmolVLA",
    "tdmpc": "TD-MPC",
    "vla_jepa": "VLA-JEPA",
    "vqbet": "VQ-BeT",
    "wall_x": "Wall-X",
    "xvla": "X-VLA",
}

# These policies have a hard visual-input requirement in their config
# validation. ACT/TD-MPC/VQ-BeT can also train from non-visual state features.
_VISUAL_POLICIES = {
    "diffusion",
    "eo1",
    "evo1",
    "fastwam",
    "groot",
    "lingbot_va",
    "multi_task_dit",
    "pi0",
    "pi0_fast",
    "pi05",
    "smolvla",
    "vqbet",
    "vla_jepa",
    "wall_x",
    "xvla",
}
_STATE_POLICIES = {
    "tdmpc",
    "vqbet",
}
_LANGUAGE_POLICIES = {
    "eo1",
    "evo1",
    "groot",
    "lingbot_va",
    "multi_task_dit",
    "pi0",
    "pi0_fast",
    "pi05",
    "smolvla",
    "vla_jepa",
    "wall_x",
    "xvla",
}
_POLICY_FEATURE_FILTER_POLICIES = {"tdmpc", "vqbet"}

_PROBE_SCRIPT = r'''
import importlib.metadata
import importlib.util
import json
import pkgutil
import sys
import torch
from pathlib import Path
import lerobot.policies as policies
from lerobot.policies.factory import get_policy_class, make_policy_config

rows = []
for module in sorted(pkgutil.iter_modules(policies.__path__), key=lambda item: item.name):
    name = module.name
    if name.startswith("_"):
        continue
    spec = importlib.util.find_spec(f"lerobot.policies.{name}")
    locations = list(spec.submodule_search_locations or []) if spec else []
    if not any(Path(location).glob("configuration_*.py") for location in locations):
        continue
    try:
        make_policy_config(name)
    except Exception as exc:
        rows.append({"type": name, "available": False, "unavailable_reason": str(exc)})
        continue
    try:
        get_policy_class(name)
        rows.append({"type": name, "available": True, "unavailable_reason": None})
    except Exception as exc:
        rows.append({"type": name, "available": False, "unavailable_reason": str(exc)})
print("ALEX_CAPABILITIES=" + json.dumps({
    "lerobot_version": importlib.metadata.version("lerobot"),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "stats_sanitizer": bool(getattr(sys.modules.get("alex_lerobot_compat"), "ALEX_STATS_SANITIZER", False)),
    "policy_feature_filter": bool(getattr(sys.modules.get("alex_lerobot_compat"), "ALEX_POLICY_FEATURE_FILTER", False)),
    "policies": rows,
}, separators=(",", ":")))
'''

_DATASET_PROBE_SCRIPT = r'''
import json
import math
import sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

repo_id = sys.argv[1]
info_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="meta/info.json")
info = json.loads(Path(info_file).read_text())
dataset_info = HfApi().dataset_info(repo_id)
siblings = {item.rfilename for item in dataset_info.siblings}
features = info.get("features") or {}
action_feature = features.get("action") if isinstance(features, dict) else None
action_names = action_feature.get("names") if isinstance(action_feature, dict) else None
action_shape = action_feature.get("shape") if isinstance(action_feature, dict) else None
action_dim = action_shape[0] if isinstance(action_shape, list) and action_shape else None
state_feature = features.get("observation.state") if isinstance(features, dict) else None
state_shape = state_feature.get("shape") if isinstance(state_feature, dict) else None
state_dim = state_shape[0] if isinstance(state_shape, list) and state_shape else None
groot_relative_actions_ready = (
    isinstance(action_names, list)
    and isinstance(action_dim, int)
    and len(action_names) >= action_dim
)
if groot_relative_actions_ready:
    groot_relative_actions_reason = None
elif isinstance(action_names, dict):
    groot_relative_actions_reason = (
        "The action feature uses grouped name metadata, but LeRobot 0.6 GR00T relative-action "
        "grouping requires a flat action-name list."
    )
else:
    groot_relative_actions_reason = (
        "The action feature does not provide one flat name per action dimension."
    )
stats = {}
try:
    stats_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="meta/stats.json")
    stats = json.loads(Path(stats_file).read_text())
except Exception:
    pass

non_finite_stats = []
for feature_name in ("observation.state", "action"):
    for stat_name, values in (stats.get(feature_name) or {}).items():
        if stat_name == "count" or not isinstance(values, list):
            continue
        invalid = 0
        for value in values:
            try:
                invalid += int(not math.isfinite(float(value)))
            except (TypeError, ValueError):
                invalid += 1
        if invalid:
            non_finite_stats.append({"feature": feature_name, "stat": stat_name, "count": invalid})
cameras = [
    name for name, spec in features.items()
    if name.startswith("observation.images.")
    or (isinstance(spec, dict) and spec.get("dtype") in {"video", "image"})
]
has_episodes = "meta/episodes.jsonl" in siblings
has_tasks_parquet = "meta/tasks.parquet" in siblings
layout = "v3" if has_tasks_parquet else ("v2" if has_episodes else "unknown")
print("ALEX_DATASET=" + json.dumps({
    "repo_id": repo_id,
    "revision": getattr(dataset_info, "sha", None),
    "valid": bool(features) and layout != "unknown",
    "features": list(features),
    "state_dim": state_dim,
    "action_dim": action_dim,
    "cameras": cameras,
    "has_tasks": has_tasks_parquet or "meta/tasks.jsonl" in siblings,
    "non_finite_stats": non_finite_stats,
    "groot_relative_actions_ready": groot_relative_actions_ready,
    "groot_relative_actions_reason": groot_relative_actions_reason,
}, separators=(",", ":")))
'''


def _remote_probe(cluster: ClusterManager, image: str) -> dict[str, Any]:
    command = shlex.join(
        ["docker", "run", "--rm", "--gpus", "all", image, "python3", "-c", _PROBE_SCRIPT]
    )
    code, stdout, stderr = cluster.execute(command, timeout=120)
    if code:
        raise RuntimeError(stderr.strip() or "policy probe failed")
    marker = "ALEX_CAPABILITIES="
    payload = next((line[len(marker) :] for line in reversed(stdout.splitlines()) if line.startswith(marker)), None)
    if payload is None:
        raise RuntimeError("training image returned no policy capability payload")
    result = json.loads(payload)
    if not result.get("cuda_available"):
        raise RuntimeError(
            "The training image cannot initialize a gpu2 GPU "
            f"(PyTorch {result.get('torch_version')}, CUDA wheel {result.get('torch_cuda_version')}). "
            "Rebuild Dockerfile.alex-training with the CUDA 12.8 PyTorch wheel."
        )
    return result


def _remote_dataset_probe(cluster: ClusterManager, image: str, repo_id: str) -> dict[str, Any]:
    docker = shlex.join(
        [
            "docker",
            "run",
            "--rm",
            "--env",
            "HF_TOKEN",
            image,
            "python3",
            "-c",
            _DATASET_PROBE_SCRIPT,
            repo_id,
        ]
    )
    command = with_remote_hf_token(docker)
    code, stdout, stderr = cluster.execute(command, timeout=120)
    if code:
        raise RuntimeError(stderr.strip() or "dataset metadata probe failed")
    marker = "ALEX_DATASET="
    payload = next((line[len(marker) :] for line in reversed(stdout.splitlines()) if line.startswith(marker)), None)
    if payload is None:
        raise RuntimeError("training image returned no dataset metadata payload")
    return json.loads(payload)


def _compatibility(policy_type: str, dataset: dict[str, Any] | None) -> tuple[bool, str | None]:
    if dataset is None:
        return True, None
    if not dataset.get("valid"):
        return False, "Dataset metadata is not a valid LeRobot dataset."
    features = set(dataset.get("features") or [])
    if "action" not in features:
        return False, "Dataset is missing the action feature."
    if policy_type in _STATE_POLICIES and "observation.state" not in features:
        return False, "This policy requires the observation.state feature."
    cameras = dataset.get("cameras") or []
    if policy_type in _VISUAL_POLICIES and not cameras:
        return False, "This policy requires at least one image or video camera feature."
    if policy_type in _LANGUAGE_POLICIES and not dataset.get("has_tasks"):
        return False, "This language-conditioned policy requires dataset task metadata."
    return True, None


def _needs_alex_policy_feature_filter(policy_type: str, dataset: dict[str, Any] | None) -> bool:
    if policy_type not in _POLICY_FEATURE_FILTER_POLICIES or dataset is None:
        return False
    return (
        dataset.get("state_dim") == 48
        and dataset.get("action_dim") == 46
        and len(dataset.get("cameras") or []) >= 2
    )


def get_training_capabilities(
    dataset: dict[str, Any] | None = None,
    dataset_repo_id: str | None = None,
    cluster: ClusterManager = cluster_manager,
) -> dict[str, Any]:
    """Return policies importable in the exact image used by remote jobs."""
    if not cluster.status().get("connected"):
        raise ConnectionError("cluster is not connected")
    image = remote_training_image()
    dataset_key = dataset_repo_id or str((dataset or {}).get("repo_id") or "")
    key = (image, dataset_key)
    cached = _cache.get(key)
    if cached and time.monotonic() - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    if dataset is None and dataset_repo_id:
        dataset = _remote_dataset_probe(cluster, image, dataset_repo_id)
    probe = _remote_probe(cluster, image)
    dataset_warnings: list[str] = []
    invalid_stats = (dataset or {}).get("non_finite_stats") or []
    if invalid_stats:
        affected = ", ".join(
            f"{item['feature']}.{item['stat']} ({item['count']})" for item in invalid_stats
        )
        if probe.get("stats_sanitizer"):
            dataset_warnings.append(
                "Dataset contains non-finite normalization stats: "
                f"{affected}. The Alex image will repair them in memory from min/max; "
                "recompute and republish meta/stats.json for a permanent fix."
            )
        else:
            dataset_warnings.append(
                f"Dataset contains non-finite normalization stats: {affected}. "
                "Rebuild the Alex image or recompute the dataset stats before training."
            )
    relative_actions_ready = bool((dataset or {}).get("groot_relative_actions_ready"))
    relative_actions_reason = (dataset or {}).get("groot_relative_actions_reason")
    if dataset is not None and not relative_actions_ready:
        dataset_warnings.append(
            "GR00T relative actions are disabled for this dataset. "
            + str(relative_actions_reason or "Compatible chunked action metadata is unavailable.")
            + " Train with absolute actions, or republish the dataset with flat action names and "
            "horizon-preserving relative-action statistics."
        )
    policies = []
    for item in probe.get("policies", []):
        policy_type = str(item.get("type") or "")
        if policy_type not in _LABELS:
            continue
        compatible, reason = _compatibility(policy_type, dataset)
        if (
            compatible
            and _needs_alex_policy_feature_filter(policy_type, dataset)
            and not probe.get("policy_feature_filter")
        ):
            compatible = False
            reason = (
                "This policy needs the Alex LeRobot feature-filter compatibility shim for "
                "H2Ozone/test_obs_new. Rebuild the Alex training image."
            )
        fields: list[dict[str, Any]] = [
            {
                "name": "policy_pretrained_path",
                "label": "Pretrained policy (optional)",
                "type": "string",
                "default": None,
            }
        ]
        if policy_type == "groot":
            fields += [
                {
                    "name": "policy_base_model_path",
                    "label": "Base model",
                    "type": "string",
                    "default": "nvidia/GR00T-N1.7-3B",
                },
                {"name": "policy_embodiment_tag", "label": "Embodiment", "type": "string", "default": "new_embodiment"},
                {"name": "policy_chunk_size", "label": "Chunk size", "type": "integer", "default": 16},
                {"name": "policy_n_action_steps", "label": "Action steps", "type": "integer", "default": 16},
                {
                    "name": "policy_use_relative_actions",
                    "label": "Relative actions",
                    "type": "boolean",
                    "default": False,
                },
                {"name": "policy_use_bf16", "label": "Use bfloat16", "type": "boolean", "default": True},
            ]
        policies.append(
            {
                **item,
                "label": _LABELS.get(policy_type, policy_type.replace("_", " ").title()),
                "compatible": compatible,
                "compatibility_reason": reason,
                "fields": fields,
            }
        )
    result = {
        "image": image,
        "lerobot_version": probe.get("lerobot_version"),
        "torch_version": probe.get("torch_version"),
        "torch_cuda_version": probe.get("torch_cuda_version"),
        "cuda_device_count": probe.get("cuda_device_count"),
        "dataset_repo_id": (dataset or {}).get("repo_id"),
        "dataset_warnings": dataset_warnings,
        "groot_relative_actions_ready": relative_actions_ready,
        "groot_relative_actions_reason": relative_actions_reason,
        "policies": policies,
    }
    _cache[key] = (time.monotonic(), result)
    return result
