# Alex Lab

Alex Lab is a local web interface for Alex humanoid learning workflows. It
provides dataset inspection and conversion, multi-policy LeRobot training, live
multi-GPU cluster telemetry, durable remote jobs, and Isaac Lab-Arena
simulation evaluation. Teleoperation and real-robot deployment are
intentionally outside this version.

The app is derived from LeLab's FastAPI/React architecture and is intended for
one operator on `localhost`.

## Requirements

Local machine:

- Linux, Python 3.12+, Node.js, and npm
- `/home/bpratt/IsaacLab-Arena` for conversion and simulation evaluation
- Docker and the NVIDIA Container Toolkit for local Isaac Lab-Arena evaluation

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

## Training

The Training page refreshes GPU utilization, memory, temperature, power, and
compute processes every few seconds. Select any available GPU subset and any
policy detected in the pinned remote LeRobot image. Multi-GPU jobs use
Hugging Face Accelerate. Datasets are downloaded from the Hub; checkpoints are
kept in a per-job Docker volume and pushed to the selected Hub model repo.

Build the training image on the remote host before connecting Alex Lab:

```bash
cd /home/bpratt/leLab_for_alex
docker build --pull -f Dockerfile.alex-training -t bpratt/alex-lerobot-train:0.6.0 .
docker run --rm --gpus all bpratt/alex-lerobot-train:0.6.0 \
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

## Evaluation

Evaluation wraps `isaaclab_arena/evaluation/policy_runner.py` and supports Alex
task/embodiment selection, CCIL or GR00T checkpoints, episode counts,
instructions, recorded viewport/camera videos, and optional poke robustness
tests. Evaluation runs through the local Isaac Lab-Arena environment, so its
model and dataset mounts must be available locally.

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
