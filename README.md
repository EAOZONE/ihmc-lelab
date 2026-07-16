# Alex Lab

Alex Lab is a local web interface for Alex humanoid learning workflows. It
provides dataset inspection and conversion, multi-policy LeRobot training, live
multi-GPU cluster telemetry, durable remote jobs, and unified policy rollout to
Isaac Lab/Isaac Sim. Physical Alex deployment is represented by the same workflow
but remains motion-gated until the authoritative IHMC state, frame-transform,
complete actuation, and watchdog providers are configured.

The app is derived from LeLab's FastAPI/React architecture and is intended for
one operator on `localhost`.

## Requirements

Local machine:

- Linux, Python 3.12+, Node.js, and npm
- `/home/bpratt/IsaacLab` for local Isaac Lab/Isaac Sim rollout
- `/home/bpratt/IsaacLab-Arena` for legacy dataset conversion helpers

Remote training host:

- SSH password login and a stable host key
- Docker plus the NVIDIA Container Toolkit
- Seven H100 GPUs (indices 0–6) visible to `nvidia-smi`
- The pinned `alex-lerobot-train:0.6.0` image built from
  `Dockerfile.alex-training`
- Hugging Face authentication configured for the remote user
- A writable checkpoint/cache root

Revoke any access token that has appeared in a command transcript or chat.
Authenticate once on the remote host with the Hugging Face CLI instead of
putting a token in Alex Lab.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
cd frontend
npm install
npm run build
cd ..
alexlab
```

Open <http://localhost:8000>. For frontend development, run `npm run dev` in
`frontend` and keep `python -m uvicorn lelab.alex_server:app --host 127.0.0.1
--port 8000` running in another terminal.

## First connection to gpu2

1. Open **Setup** and enter the SSH host, port, and username.
2. Compare the displayed host-key fingerprint with one obtained through a
   trusted channel before approving it.
3. Enter the SSH password. The password remains only in backend memory and is
   discarded on disconnect, idle expiry, or server restart.
4. Run prerequisite checks for Docker, the NVIDIA runtime, image, Hugging Face
   login, storage roots, and the pinned LeRobot training image.

Connection profile fields and the approved fingerprint may be retained; SSH
passwords and Hugging Face tokens are never written to Alex Lab job records.
After an Alex Lab restart, reconnect to resume monitoring surviving named
containers.

## Captury sim demos → Hub → train

For **Captury teleop in Isaac Lab-Arena** (lever on `alex_empty`), HDF5 → LeRobot
conversion, Hugging Face upload, and training/rollout in this app, see
[`docs/CAPTURY_TO_LEROBOT_TRAINING.md`](docs/CAPTURY_TO_LEROBOT_TRAINING.md).

## Repo-owned boundary

Alex Lab is the single operator entry point, but it only owns the orchestration
code that needs to live here:

| Workflow | This repo owns | External implementation |
|---|---|---|
| Teleop | `TeleopConfig`, `build_isaaclab_teleop_command`, `TeleopManager`, `alexlab teleop` | Isaac Lab task/device scripts |
| Mimic annotation | `DemoAnnotationConfig`, `build_demo_annotation_command`, `alexlab annotate-demos` | IsaacLab-Arena `annotate_demos.py`, mimic logic, USD/task setup |
| Training | LeRobot request schemas, compatibility checks, remote Docker/HF launch, `alexlab train-command` | LeRobot trainer and policy code |
| Rollout | Policy source resolution, policy server launch, rollout manifests, `alexlab rollout` | Isaac Lab runner, Arena `policy_runner.py`, LeRobot policy server |

## Training

The Training page refreshes GPU utilization, memory, temperature, power, and
compute processes every few seconds. Select any available GPU subset and any
policy detected in the pinned remote LeRobot image. Multi-GPU jobs use
Hugging Face Accelerate. Datasets are downloaded from the Hub; checkpoints are
kept in a per-job Docker volume and pushed to the selected Hub model repo.

Build the training image on the remote host before connecting Alex Lab:

```bash
cd /home/bpratt/leLab_for_alex
docker build --pull -f Dockerfile.alex-training -t alex-lerobot-train:0.6.0 .
docker run --rm --gpus all alex-lerobot-train:0.6.0 \
  python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Set `ALEX_LEROBOT_IMAGE` before starting Alex Lab to use a published or
site-specific image tag instead.

The image pins PyTorch's CUDA 12.8 wheel to match gpu2. Policy discovery runs
inside the image with GPU access and refuses to launch training if CUDA cannot
initialize. Dataset stats containing JSON `"NaN"` values are repaired in memory
from their finite min/max range, with a warning; the source Hub dataset is not
modified. Training explicitly uses LeRobot's PyAV video backend so decoding does
not depend on TorchCodec's host FFmpeg/libpython ABI.

GR00T defaults to absolute actions. Relative actions are only enabled when the
dataset exposes one flat name per action dimension and compatible chunked
statistics; unsupported launches are rejected before the model is downloaded.

Each job receives a named container and its own durable remote checkpoint
directory. Stopping a job stops its container. Existing processes not launched
by Alex Lab are displayed as occupied and are never terminated.

GR00T uses LeRobot's standard GR00T N1.7 policy integration. Legacy CCIL and
custom GR00T jobs remain visible in job history but cannot be launched.

To inspect the exact in-container LeRobot command without starting a remote job:

```bash
alexlab train-command --dataset-repo <owner/dataset> --model-repo <owner/model> \
  --policy-type groot --policy-base-model-path nvidia/GR00T-N1.7-3B
```

## Rollout

Rollout loads a schema-compatible LeRobot checkpoint on one GPU of the connected
training host and forwards its typed observation/action protocol through the
existing authenticated SSH connection. Isaac Lab/Isaac Sim runs locally and
owns the Alex simulator state, episode metrics, and viewport/camera videos.

## Teleop

Alex Lab owns the thin launcher for local Isaac Lab teleoperation; Isaac Lab
still owns the simulator, task registration, viewer, and device implementation.

```bash
alexlab teleop --environment Isaac-Alex-Lever-Play-v0 --teleop-device keyboard
alexlab teleop-status <teleop-id>
alexlab teleop-logs <teleop-id>
alexlab teleop-stop <teleop-id>
```

From the UI, open **Rollout** and select a completed training job. The equivalent
CLI is:

```bash
alexlab rollout --target sim --job <job-id> --checkpoint latest \
  --environment Isaac-Alex-Lever-Play-v0 --episodes 20 --camera-video
alexlab rollout-status <rollout-id>
alexlab rollout-logs <rollout-id>
alexlab rollout-stop <rollout-id>
```

Direct local or Hub policies use `--policy`; pass `--dataset-repo` when the
checkpoint does not contain `lelab_rollout.json`. Training uploads this manifest
to the model root and each saved checkpoint. Compatibility is based on the saved
feature schema, not merely the policy class.

To have IsaacLab-Arena open locally while the model is downloaded and served by
the local Docker policy server, use a direct Hub ref with `--local-inference` and
`--show`:

```bash
alexlab rollout --target arena --policy <owner/model> --checkpoint latest \
  --dataset-repo <owner/dataset> --local-inference --show \
  --environment alex_empty --usd isaaclab_arena/assets/lever_sim/LEVER_AGAIN.usd \
  --task "Turn the lever"
```

For a LeLab-trained model, the policy server resolves `<owner/model>@latest` to
the newest Hub checkpoint and caches the downloaded weights in the
`alex_hf_cache` Docker volume. Omit `--dataset-repo` when the model already has
`lelab_rollout.json`.

`--target robot` currently performs the physical-Alex readiness audit and exits
without motion unless all required capability providers are installed. The old
send-only wrist UDP bridge is deliberately not accepted as a complete hardware
target.

## Storage and recovery

- Local Alex Lab job metadata contains no credentials.
- Remote training output stays under the configured checkpoint root.
- GR00T's image resumes from the latest checkpoint in the mounted output
  directory.
- Named containers allow status and logs to be reattached after reconnecting.
- If the remote host is shared with workloads outside Alex Lab, GPU telemetry
  is advisory; use the cluster's scheduler when one is introduced.

## Development checks

```bash
pytest
ruff check lelab tests
cd frontend
npm run build
```

The Apache-2.0 license from LeLab is retained.
