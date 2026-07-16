#!/usr/bin/env python3
"""Loopback HTTP inference service for schema-compatible LeRobot policies.

The wire format is a zip archive containing ``meta.json`` and one NumPy ``.npy``
file per observation. It is intentionally data-only: unlike LeRobot's older
async inference service, no Python objects are unpickled from the network.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import traceback
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors


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


def feature_schema(features) -> dict:
    result = {}
    for name, feature in (features or {}).items():
        shape = getattr(feature, "shape", None)
        feature_type = getattr(feature, "type", None)
        result[name] = {
            "shape": list(shape) if shape is not None else None,
            "type": getattr(feature_type, "value", str(feature_type)),
        }
    return result


_TORCH_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class Runtime:
    def __init__(self, policy_ref: str, device: str, dtype: str = "float32") -> None:
        self.path = resolve_policy_ref(policy_ref)
        self.device = torch.device(device)
        config = PreTrainedConfig.from_pretrained(self.path)
        policy_cls = get_policy_class(config.type)
        self.policy = policy_cls.from_pretrained(self.path)
        self.policy.to(self.device, dtype=_TORCH_DTYPES[dtype]).eval()
        # nn.Module.to(dtype=...) allocates new casted storage before the old
        # full-precision storage is freed; the caching allocator won't hand that
        # freed CUDA memory back to the driver on its own, so nvidia-smi keeps
        # reporting the pre-cast footprint until empty_cache() is called.
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        # Inputs must match the policy's dtype, but outputs are cast back to float32
        # regardless: numpy has no bfloat16, and predict() calls .numpy() on the result.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.path,
            preprocessor_overrides={"device_processor": {"device": str(self.device), "float_dtype": dtype}},
            postprocessor_overrides={"device_processor": {"device": str(self.device), "float_dtype": "float32"}},
        )
        self.schema = {
            "protocol_version": 1,
            "policy_type": config.type,
            "input_features": feature_schema(config.input_features),
            "output_features": feature_schema(config.output_features),
        }

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def predict(self, payload: bytes) -> bytes:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            meta = json.loads(archive.read("meta.json"))
            observation = {}
            for key, filename in meta["features"].items():
                array = np.load(io.BytesIO(archive.read(filename)), allow_pickle=False)
                tensor = torch.from_numpy(array)
                if "image" in key:
                    if tensor.dtype == torch.uint8:
                        tensor = tensor.float().div_(255)
                    if tensor.ndim == 3:
                        tensor = tensor.unsqueeze(0)
                    if tensor.shape[-1] in (1, 3, 4):
                        tensor = tensor.permute(0, 3, 1, 2).contiguous()
                elif tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                observation[key] = tensor.to(self.device)
            batch = next(v.shape[0] for v in observation.values() if isinstance(v, torch.Tensor))
            observation["task"] = [meta.get("task", "")] * batch
            observation["robot_type"] = [meta.get("robot_type", "alex")] * batch
        with torch.inference_mode():
            processed = self.preprocessor(observation)
            actions = self.policy.predict_action_chunk(processed)
            actions = self.postprocessor(actions).detach().cpu().numpy()
        # Arena's lerobot_remote client expects rank-3 chunks (B, T, action_dim).
        if actions.ndim == 2:
            actions = actions[:, None, :]
        elif actions.ndim == 1:
            actions = actions[None, None, :]
        output = io.BytesIO()
        np.save(output, actions, allow_pickle=False)
        return output.getvalue()


def make_handler(runtime: Runtime):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, b'{"status":"ok"}', "application/json")
            elif self.path == "/schema":
                self._send(200, json.dumps(runtime.schema).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            try:
                if self.path == "/reset":
                    runtime.reset()
                    self._send(200, b'{"status":"ok"}', "application/json")
                    return
                if self.path != "/predict":
                    self._send(404, b"not found", "text/plain")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 256 * 1024 * 1024:
                    raise ValueError("invalid request size")
                self._send(200, runtime.predict(self.rfile.read(length)), "application/x-npy")
            except Exception as exc:
                traceback.print_exc()
                self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")

        def log_message(self, fmt, *args):
            print(f"[policy-server] {fmt % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=sorted(_TORCH_DTYPES))
    args = parser.parse_args()
    runtime = Runtime(args.policy, args.device, args.dtype)
    print("POLICY_SERVER_READY=" + json.dumps(runtime.schema, separators=(",", ":")), flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(runtime)).serve_forever()


if __name__ == "__main__":
    main()
