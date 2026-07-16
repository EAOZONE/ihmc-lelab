#!/usr/bin/env python3
"""Run LeRobot training, then attach portable rollout metadata to its Hub model."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def enrich_manifest(manifest: dict) -> dict:
    repo_id = manifest["dataset_repo_id"]
    info_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="meta/info.json")
    info = json.loads(Path(info_path).read_text())
    dataset_info = HfApi().dataset_info(repo_id)
    return {
        **manifest,
        "dataset_revision": getattr(dataset_info, "sha", None),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "features": info.get("features") or {},
    }


def upload_manifest(manifest: dict) -> None:
    api = HfApi()
    repo_id = manifest["model_repo_id"]
    payload = json.dumps(enrich_manifest(manifest), indent=2).encode()
    siblings = [item.rfilename for item in api.model_info(repo_id).siblings]
    destinations = ["lelab_rollout.json"]
    destinations.extend(
        path.removesuffix("config.json") + "lelab_rollout.json"
        for path in siblings
        if path.startswith("checkpoints/") and path.endswith("/pretrained_model/config.json")
    )
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "lelab_rollout.json"
        source.write_bytes(payload)
        for destination in sorted(set(destinations)):
            api.upload_file(
                path_or_fileobj=source,
                path_in_repo=destination,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add LeLab rollout manifest",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a training command is required after --")
    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        try:
            upload_manifest(json.loads(args.manifest))
        except Exception as exc:
            print(f"WARNING: training succeeded but rollout manifest upload failed: {exc}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
