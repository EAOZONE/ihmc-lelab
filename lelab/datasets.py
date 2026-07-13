# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub.errors import HfHubHTTPError

from .utils.hf_auth import cached_whoami, shared_hf_api

logger = logging.getLogger(__name__)


def _lerobot_cache_root() -> Path:
    return Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def _is_dataset_dir(path: Path) -> bool:
    """A directory is a LeRobot dataset iff <dir>/meta/info.json exists."""
    try:
        return (path / "meta" / "info.json").is_file()
    except OSError:
        return False


def _dir_mtime_iso(path: Path) -> str | None:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    except OSError:
        return None


def list_local_datasets() -> list[dict[str, Any]]:
    """Scan the LeRobot cache for local datasets (dirs containing meta/info.json).

    Walks one level deep: a top-level dataset dir is recorded as "<name>"; if a
    top-level dir is not itself a dataset, each subdir that is a dataset is
    recorded as "<top>/<sub>". Does not descend further.
    """
    root = _lerobot_cache_root()
    if not root.is_dir():
        return []

    out: list[dict[str, Any]] = []
    try:
        top_entries = list(root.iterdir())
    except OSError as e:
        logger.warning(f"Could not read LeRobot cache root {root}: {e}")
        return []

    for top in top_entries:
        try:
            if not top.is_dir():
                continue
        except OSError:
            continue

        if _is_dataset_dir(top):
            out.append(
                {
                    "repo_id": top.name,
                    "last_modified": _dir_mtime_iso(top),
                    "private": False,
                }
            )
            continue

        # Not a dataset itself — descend one level.
        try:
            sub_entries = list(top.iterdir())
        except OSError:
            continue
        for sub in sub_entries:
            try:
                if not sub.is_dir():
                    continue
            except OSError:
                continue
            if _is_dataset_dir(sub):
                out.append(
                    {
                        "repo_id": f"{top.name}/{sub.name}",
                        "last_modified": _dir_mtime_iso(sub),
                        "private": False,
                    }
                )

    out.sort(key=lambda d: d["last_modified"] or "", reverse=True)
    return out


def list_user_datasets() -> list[dict[str, Any]]:
    info = cached_whoami()
    if info is None:
        return []

    authors = [info["name"]] + [o["name"] for o in info.get("orgs", [])]
    api = shared_hf_api()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for author in authors:
        try:
            for ds in api.list_datasets(author=author, filter="LeRobot", limit=200):
                if ds.id in seen:
                    continue
                seen.add(ds.id)
                out.append(
                    {
                        "repo_id": ds.id,
                        "last_modified": ds.last_modified.isoformat() if ds.last_modified else None,
                        "private": bool(getattr(ds, "private", False)),
                    }
                )
        except HfHubHTTPError as e:
            logger.warning(f"list_datasets({author}) failed: {e}")

    out.sort(key=lambda d: d["last_modified"] or "", reverse=True)
    return out


def list_all_datasets() -> list[dict[str, Any]]:
    """Merged listing: Hub datasets + local cache, with `source` field.

    A repo_id present in both lists is collapsed to one entry with
    source="both" and last_modified set to the more recent of the two.
    """
    hub = list_user_datasets()
    local = list_local_datasets()

    merged: dict[str, dict[str, Any]] = {}
    for item in hub:
        merged[item["repo_id"]] = {**item, "source": "hub"}
    for item in local:
        rid = item["repo_id"]
        if rid in merged:
            existing = merged[rid]
            existing["source"] = "both"
            # Keep the newer timestamp; ISO strings sort lexically.
            a = existing.get("last_modified") or ""
            b = item.get("last_modified") or ""
            existing["last_modified"] = max(a, b) or None
        else:
            merged[rid] = {**item, "source": "local"}

    out = list(merged.values())
    out.sort(key=lambda d: d["last_modified"] or "", reverse=True)
    return out


def _feature_dimension(shape: object) -> int | None:
    if not isinstance(shape, list) or not shape:
        return None
    result = 1
    for value in shape:
        if not isinstance(value, int):
            return None
        result *= value
    return result


def inspect_dataset(path: str | None = None, repo_id: str | None = None) -> dict[str, Any]:
    """Inspect a LeRobot dataset without loading its parquet/video payload.

    Local paths are read directly. Hub inspection downloads only small metadata
    files through the shared authenticated API cache.
    """
    if bool(path) == bool(repo_id):
        raise ValueError("provide exactly one of path or repo_id")

    source = "local"
    if path:
        root = Path(path).expanduser().resolve()
        info_path = root / "meta" / "info.json"
        modality_path = root / "meta" / "modality.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"missing LeRobot metadata: {info_path}")
        info = json.loads(info_path.read_text())
        modality = json.loads(modality_path.read_text()) if modality_path.is_file() else None
        has_episodes_jsonl = (root / "meta" / "episodes.jsonl").is_file()
        has_tasks_parquet = (root / "meta" / "tasks.parquet").is_file()
        has_tasks = has_tasks_parquet or (root / "meta" / "tasks.jsonl").is_file()
        revision = None
        location = str(root)
        try:
            size_bytes = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
        except OSError:
            size_bytes = None
    else:
        from huggingface_hub import hf_hub_download

        source = "hub"
        assert repo_id is not None
        info_path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="meta/info.json"))
        info = json.loads(info_path.read_text())
        modality = None
        try:
            modality_file = hf_hub_download(
                repo_id=repo_id, repo_type="dataset", filename="meta/modality.json"
            )
            modality = json.loads(Path(modality_file).read_text())
        except Exception:
            pass
        dataset_info = shared_hf_api().dataset_info(repo_id)
        siblings = {item.rfilename for item in dataset_info.siblings}
        has_episodes_jsonl = "meta/episodes.jsonl" in siblings
        has_tasks_parquet = "meta/tasks.parquet" in siblings
        has_tasks = has_tasks_parquet or "meta/tasks.jsonl" in siblings
        revision = getattr(dataset_info, "sha", None)
        location = repo_id
        size_bytes = None

    features = info.get("features") or {}
    feature_summary = []
    cameras = []
    for name, spec in features.items():
        spec = spec if isinstance(spec, dict) else {}
        item = {
            "name": name,
            "dtype": spec.get("dtype"),
            "shape": spec.get("shape"),
            "dimension": _feature_dimension(spec.get("shape")),
        }
        feature_summary.append(item)
        if name.startswith("observation.images.") or spec.get("dtype") in {"video", "image"}:
            cameras.append(name)

    version = str(info.get("codebase_version") or info.get("version") or "unknown")
    layout = "v3" if has_tasks_parquet else ("v2" if has_episodes_jsonl else "unknown")
    warnings = []
    if layout == "unknown":
        warnings.append("Could not identify the LeRobot episode metadata layout")
    return {
        "path": location,
        "valid": bool(features) and layout != "unknown",
        "format": f"LeRobot {version} ({layout})",
        "source": source,
        "location": location,
        "repo_id": info.get("repo_id") or repo_id,
        "revision": revision,
        "codebase_version": version,
        "layout": layout,
        "fps": info.get("fps"),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "size_bytes": size_bytes,
        "features": [item["name"] for item in feature_summary],
        "feature_schema": feature_summary,
        "cameras": cameras,
        "has_tasks": has_tasks,
        "warnings": warnings,
        "modality": modality,
        "conversion": {
            # LeRobot 0.6 trains GR00T N1.7 directly from a standard LeRobot
            # Hub dataset. Keep the legacy keys for older API consumers, but
            # do not direct new users through the Arena conversion pipeline.
            "gr00t_required": False,
            "gr00t_ready": has_episodes_jsonl and modality is not None,
            "ccil_supported": bool(features),
            "lerobot_ready": bool(features) and layout != "unknown",
        },
    }
