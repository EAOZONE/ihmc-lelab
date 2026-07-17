"""Typed request models and command builders for Alex workflows."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_ARENA_ROOT = Path("/home/bpratt/IsaacLab-Arena")
DEFAULT_ISAACLAB_ROOT = Path("/home/bpratt/IsaacLab")
_SAFE_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@-]*$")
_SAFE_POLICY = re.compile(r"^[a-z][a-z0-9_]*$")
_HUB_REPO = re.compile(r"^[^\s/]+/[^\s/]+$")

_ALEX_TEST_OBS_NEW_STATE_DIM = 48
_ALEX_TEST_OBS_NEW_ACTION_DIM = 46
_ALEX_TEST_OBS_NEW_REPOS = {
    "H2Ozone/test_obs_new",
    "H2Ozone/split_data",
    "H2Ozone/full_dataset",
    "H2Ozone/new_full_data",
}
_MAX_DIM_POLICY_TYPES = {
    "eo1",
    "evo1",
    "pi0",
    "pi05",
    "pi0_fast",
    "smolvla",
    "wall_x",
    "xvla",
}


class DatasetInspectRequest(BaseModel):
    """A local LeRobot dataset path or a Hugging Face dataset repo id."""

    path: str | None = None
    repo_id: str | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> DatasetInspectRequest:
        if bool(self.path) == bool(self.repo_id):
            raise ValueError("provide exactly one of path or repo_id")
        return self


class DatasetConversionConfig(BaseModel):
    format: Literal["gr00t", "ccil"]
    input_path: str | None = None
    repo_id: str | None = None
    output_path: str
    arena_root: str = str(DEFAULT_ARENA_ROOT)
    modality_template: str | None = None
    action_from_state_dims: str | None = None
    image_keys: list[str] = Field(default_factory=list)
    output_image_keys: list[str] = Field(default_factory=list)
    image_size: tuple[int, int] = (128, 128)

    @model_validator(mode="after")
    def validate_source_and_format(self) -> DatasetConversionConfig:
        if bool(self.input_path) == bool(self.repo_id):
            raise ValueError("provide exactly one of input_path or repo_id")
        if self.format == "gr00t" and not self.input_path:
            raise ValueError("GR00T conversion requires input_path")
        if self.format == "gr00t" and not self.modality_template:
            raise ValueError("GR00T conversion requires modality_template")
        if self.output_image_keys and len(self.output_image_keys) != len(self.image_keys):
            raise ValueError("output_image_keys must match image_keys")
        if self.action_from_state_dims and not re.fullmatch(r"\d+:\d+", self.action_from_state_dims):
            raise ValueError("action_from_state_dims must have start:end form")
        return self


class DemoAnnotationConfig(BaseModel):
    """Arena HDF5 demo annotation using the existing IsaacLab-Arena script."""

    input_file: str
    output_file: str
    environment: str = "alex_lever_turn"
    embodiment: str = "alex_v2_ability_hands"
    usd: str = "isaaclab_arena/assets/lever_sim/another_try_lever.usd"
    device: str = "cuda"
    viz: Literal["kit", "none"] = "kit"
    mimic: bool = True
    lever_dr: bool = True
    lever_pose_dr_xy_jitter: float | None = 0.08
    lever_pose_dr_yaw_jitter_deg: float | None = 15.0
    success_angle_threshold: float | None = 0.7853981633974483
    wrist_stiffness: int | None = 800
    wrist_damping: int | None = 50
    arena_workdir: str = "/workspaces/isaaclab_arena"
    python_executable: str = "/isaac-sim/python.sh"
    container_name: str | None = None

    @field_validator("device")
    @classmethod
    def non_empty_device(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("device cannot be empty")
        return value


class GR00TTrainingConfig(BaseModel):
    kind: Literal["gr00t"] = "gr00t"
    dataset_id: str
    model_repo: str | None = None
    dataset_path: str | None = None
    image: str = "ghcr.io/eaozone/alex-gr00t-train:latest"
    base_model_path: str = "nvidia/GR00T-N1.6-3B"
    modality_config: str = "alex_test_obs_new_data_config.py"
    modality_template: str = "alex_test_obs_new_modality.json"
    action_from_state_dims: str | None = None
    output_dir: str = "/checkpoints"
    max_steps: int = Field(default=30000, ge=1)
    save_steps: int = Field(default=5000, ge=1)
    global_batch_size: int = Field(default=32, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    dataloader_workers: int = Field(default=16, ge=0)
    use_lora: bool = False
    lora_rank: int = Field(default=64, ge=1)
    low_vram: bool = False
    skip_upload: bool = True
    upload_optimizer_state: bool = False
    color_jitter_brightness: float = Field(default=0.4, ge=0)
    color_jitter_contrast: float = Field(default=0.5, ge=0)
    color_jitter_saturation: float = Field(default=0.6, ge=0)
    color_jitter_hue: float = Field(default=0.1, ge=0, le=0.5)
    random_rotation_angle: float = Field(default=10, ge=0)

    @field_validator("image")
    @classmethod
    def safe_image_name(cls, value: str) -> str:
        if not _SAFE_IMAGE.fullmatch(value):
            raise ValueError("invalid Docker image name")
        return value


class CCILTrainingConfig(BaseModel):
    kind: Literal["ccil"] = "ccil"
    pickle_path: str
    config_path: str = "config/alex_microwave.yml"
    ccil_root: str = "/workspace/CCIL"
    image: str = "alex-ccil-train"
    naive: bool = False
    seed: int = 42
    train_epochs: int = Field(default=200, ge=1)
    batch_size: int = Field(default=256, ge=1)
    output_dir: str = "/checkpoints"

    @field_validator("image")
    @classmethod
    def safe_image_name(cls, value: str) -> str:
        if not _SAFE_IMAGE.fullmatch(value):
            raise ValueError("invalid Docker image name")
        return value


class LeRobotTrainingConfig(BaseModel):
    """A standard LeRobot training run executed in the pinned remote image."""

    kind: Literal["lerobot"] = "lerobot"
    dataset_repo_id: str
    dataset_revision: str | None = None
    dataset_episodes: list[int] | None = None
    model_repo_id: str
    policy_type: str = "act"
    policy_pretrained_path: str | None = None

    steps: int = Field(default=10000, ge=1)
    batch_size: int = Field(default=8, ge=1)
    seed: int | None = 1000
    num_workers: int = Field(default=4, ge=0)
    log_freq: int = Field(default=250, ge=1)
    save_freq: int = Field(default=1000, ge=1)
    save_checkpoint: bool = True
    policy_use_amp: bool = False
    wandb_enable: bool = False
    wandb_project: str | None = None

    # High-value GR00T N1.7 settings. They are rejected for other policies so
    # no unknown draccus flags can leak into their policy configs.
    policy_base_model_path: str | None = None
    policy_embodiment_tag: str | None = None
    policy_chunk_size: int | None = Field(default=None, ge=1)
    policy_n_action_steps: int | None = Field(default=None, ge=1)
    policy_use_relative_actions: bool | None = None
    policy_relative_exclude_joints: list[str] | None = None
    policy_use_bf16: bool | None = None
    dataset_image_transforms_enable: bool = False

    @field_validator("dataset_repo_id", "model_repo_id")
    @classmethod
    def hub_repo_id(cls, value: str) -> str:
        value = value.strip()
        if not _HUB_REPO.fullmatch(value):
            raise ValueError("must be a Hugging Face repository ID in owner/name form")
        return value

    @field_validator("policy_type")
    @classmethod
    def safe_policy_type(cls, value: str) -> str:
        value = value.strip()
        if not _SAFE_POLICY.fullmatch(value):
            raise ValueError("invalid policy type")
        return value

    @field_validator("dataset_episodes")
    @classmethod
    def valid_dataset_episodes(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return value
        if any(episode < 0 for episode in value):
            raise ValueError("dataset episodes must be non-negative")
        if len(value) != len(set(value)):
            raise ValueError("dataset episodes must be unique")
        return value

    @model_validator(mode="after")
    def policy_specific_fields(self) -> LeRobotTrainingConfig:
        groot_values = (
            self.policy_base_model_path,
            self.policy_embodiment_tag,
            self.policy_chunk_size,
            self.policy_n_action_steps,
            self.policy_use_relative_actions,
            self.policy_relative_exclude_joints,
            self.policy_use_bf16,
        )
        if self.policy_type != "groot" and any(value is not None for value in groot_values):
            raise ValueError("GR00T policy options require policy_type='groot'")
        if (
            self.policy_chunk_size is not None
            and self.policy_n_action_steps is not None
            and self.policy_n_action_steps > self.policy_chunk_size
        ):
            raise ValueError("policy_n_action_steps cannot exceed policy_chunk_size")
        return self


class RemoteTrainingRequest(BaseModel):
    config: LeRobotTrainingConfig
    gpus: list[str] = Field(min_length=1)
    name: str | None = Field(default=None, max_length=100)

    @field_validator("gpus")
    @classmethod
    def unique_gpus(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("GPU identifiers cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("GPU identifiers must be unique")
        return normalized


class DatasetEvalConfig(BaseModel):
    """Offline loss evaluation of a LeRobot policy over selected dataset episodes."""

    job_id: str | None = None
    policy_ref: str | None = None
    checkpoint: str = "latest"
    dataset_repo_id: str | None = None
    dataset_revision: str | None = None
    dataset_episodes: list[int]
    gpu: str = "0"
    batch_size: int = Field(default=8, ge=1)
    num_workers: int = Field(default=4, ge=0)
    policy_use_bf16: bool = False

    @model_validator(mode="after")
    def validate_sources(self) -> DatasetEvalConfig:
        if bool(self.job_id) == bool(self.policy_ref):
            raise ValueError("provide exactly one of job_id or policy_ref")
        if self.policy_ref and not self.dataset_repo_id:
            raise ValueError("dataset_repo_id is required when policy_ref is provided")
        if not self.dataset_episodes:
            raise ValueError("dataset_episodes is required")
        if any(episode < 0 for episode in self.dataset_episodes):
            raise ValueError("dataset episodes must be non-negative")
        if len(self.dataset_episodes) != len(set(self.dataset_episodes)):
            raise ValueError("dataset episodes must be unique")
        return self


def build_lerobot_training_command(
    config: LeRobotTrainingConfig,
    output_dir: str,
    gpu_count: int,
) -> list[str]:
    """Build the in-container LeRobot command for one or more selected GPUs."""
    if gpu_count < 1:
        raise ValueError("at least one GPU is required")

    trainer = ["python3", "-m", "lerobot.scripts.lerobot_train"]
    if gpu_count > 1:
        trainer = [
            "accelerate",
            "launch",
            "--multi_gpu",
            "--num_processes",
            str(gpu_count),
        ]
        if config.policy_type == "groot" and config.policy_use_bf16:
            trainer += ["--mixed_precision", "bf16"]
        trainer += ["--module", "lerobot.scripts.lerobot_train"]

    command = [
        *trainer,
        "--dataset.repo_id",
        config.dataset_repo_id,
        # TorchCodec's native extension is sensitive to the exact FFmpeg and
        # libpython shared libraries in the image. PyAV is part of the pinned
        # LeRobot environment and is the supported portable decoder fallback.
        "--dataset.video_backend",
        "pyav",
        "--steps",
        str(config.steps),
        "--batch_size",
        str(config.batch_size),
        "--num_workers",
        str(config.num_workers),
        "--log_freq",
        str(config.log_freq),
        "--save_freq",
        str(config.save_freq),
        "--save_checkpoint",
        str(config.save_checkpoint).lower(),
        "--save_checkpoint_to_hub",
        str(config.save_checkpoint).lower(),
        "--output_dir",
        output_dir,
        "--policy.device",
        "cuda",
        "--policy.use_amp",
        str(config.policy_use_amp).lower(),
        "--policy.push_to_hub",
        "true",
        "--policy.repo_id",
        config.model_repo_id,
        "--use_policy_training_preset",
        "true",
        "--env_eval_freq",
        "0",
        "--eval_steps",
        "0",
        "--wandb.enable",
        str(config.wandb_enable).lower(),
    ]
    if config.policy_pretrained_path:
        command += ["--policy.path", config.policy_pretrained_path]
    else:
        command += ["--policy.type", config.policy_type]
    command += ["--dataset.revision", config.dataset_revision or "main"]
    if config.dataset_episodes:
        command += ["--dataset.episodes", json.dumps(config.dataset_episodes)]
    if config.seed is not None:
        command += ["--seed", str(config.seed)]
    if config.wandb_enable and config.wandb_project:
        command += ["--wandb.project", config.wandb_project]
    if config.dataset_image_transforms_enable:
        command += ["--dataset.image_transforms.enable", "true"]

    if config.dataset_repo_id in _ALEX_TEST_OBS_NEW_REPOS:
        command += _alex_test_obs_new_policy_overrides(config.policy_type)

    if config.policy_type == "groot":
        optional = {
            "--policy.base_model_path": config.policy_base_model_path,
            "--policy.embodiment_tag": config.policy_embodiment_tag,
            "--policy.chunk_size": config.policy_chunk_size,
            "--policy.n_action_steps": config.policy_n_action_steps,
            "--policy.use_relative_actions": config.policy_use_relative_actions,
            "--policy.use_bf16": config.policy_use_bf16,
        }
        for flag, value in optional.items():
            if value is not None:
                rendered = str(value).lower() if isinstance(value, bool) else str(value)
                command += [flag, rendered]
        if config.policy_relative_exclude_joints is not None:
            command += [
                "--policy.relative_exclude_joints",
                json.dumps(config.policy_relative_exclude_joints),
            ]
    return command


def build_demo_annotation_env(config: DemoAnnotationConfig) -> dict[str, str]:
    """Return environment overrides needed by Arena's mimic annotation path."""
    env: dict[str, str] = {}
    if config.wrist_stiffness is not None:
        env["ALEX_TELEOP_WRIST_STIFFNESS"] = str(config.wrist_stiffness)
    if config.wrist_damping is not None:
        env["ALEX_TELEOP_WRIST_DAMPING"] = str(config.wrist_damping)
    return env


def build_demo_annotation_command(config: DemoAnnotationConfig) -> list[str]:
    """Build the minimal command for Arena's annotate_demos.py workflow.

    The heavy dependencies stay in IsaacLab-Arena/Isaac Sim. Alex Lab only owns
    this launcher contract so teleop, training, annotation, and rollout can be
    driven from one repo without copying Arena internals.
    """
    script = "isaaclab_arena/scripts/imitation_learning/annotate_demos.py"
    command = [
        config.python_executable,
        script,
        "--device",
        config.device,
        "--viz",
        config.viz,
    ]
    if config.mimic:
        command.append("--mimic")
    command += [
        "--input_file",
        config.input_file,
        "--output_file",
        config.output_file,
        config.environment,
        "--embodiment",
        config.embodiment,
        "--usd",
        config.usd,
    ]
    if config.lever_dr:
        command.append("--lever_dr")
    if config.lever_pose_dr_xy_jitter is not None:
        command += ["--lever_pose_dr_xy_jitter", str(config.lever_pose_dr_xy_jitter)]
    if config.lever_pose_dr_yaw_jitter_deg is not None:
        command += ["--lever_pose_dr_yaw_jitter_deg", str(config.lever_pose_dr_yaw_jitter_deg)]
    if config.success_angle_threshold is not None:
        command += ["--success_angle_threshold", str(config.success_angle_threshold)]
    if config.container_name:
        docker_prefix = ["docker", "exec", "-w", config.arena_workdir]
        for key, value in build_demo_annotation_env(config).items():
            docker_prefix += ["-e", f"{key}={value}"]
        command = [*docker_prefix, config.container_name, *command]
    return command


def _alex_test_obs_new_policy_overrides(policy_type: str) -> list[str]:
    """Return LeRobot policy flags required by H2Ozone/test_obs_new.

    The dataset has a 48-D robot state and a 46-D action. Several LeRobot 0.6
    policies default to smaller max/action dimensions and otherwise reject the
    dataset during config validation or first forward pass.
    """
    if policy_type in _MAX_DIM_POLICY_TYPES:
        return [
            "--policy.max_state_dim",
            str(_ALEX_TEST_OBS_NEW_STATE_DIM),
            "--policy.max_action_dim",
            str(_ALEX_TEST_OBS_NEW_ACTION_DIM),
        ]
    if policy_type == "fastwam":
        return [
            "--policy.action_dim",
            str(_ALEX_TEST_OBS_NEW_ACTION_DIM),
            "--policy.proprio_dim",
            str(_ALEX_TEST_OBS_NEW_STATE_DIM),
        ]
    if policy_type == "lingbot_va":
        import json

        return [
            "--policy.action_dim",
            str(_ALEX_TEST_OBS_NEW_ACTION_DIM),
            "--policy.used_action_channel_ids",
            json.dumps(list(range(_ALEX_TEST_OBS_NEW_ACTION_DIM))),
        ]
    return []


class EvaluationConfig(BaseModel):
    policy_type: str
    model_path: str
    environment: str = "alex_open_microwave"
    embodiment: str = "alex_v2_ability_hands"
    meta_path: str | None = None
    policy_device: str = "cuda"
    device: str = "cuda"
    num_episodes: int = Field(default=20, ge=1)
    headless: bool = True
    enable_cameras: bool = True
    video: bool = False
    camera_video: bool = False
    video_dir: str | None = None
    language_instruction: str | None = None
    arena_root: str = str(DEFAULT_ARENA_ROOT)
    container_name: str = "isaaclab_arena-latest"
    container_workdir: str = "/workspaces/isaaclab_arena"
    python_executable: str = "/isaac-sim/python.sh"


DEFAULT_ARENA_LEVER_USD = "isaaclab_arena/assets/lever_sim/LEVER_AGAIN.usd"
_LEROBOT_REMOTE_POLICY = "isaaclab_arena.policy.lerobot_remote_policy.LeRobotRemotePolicy"


class RolloutConfig(BaseModel):
    """A policy deployment against Arena, Isaac Lab/Sim, or the physical Alex robot."""

    target: Literal["sim", "arena", "robot"] = "arena"
    inference_location: Literal["remote", "local"] = "remote"
    job_id: str | None = None
    policy_ref: str | None = None
    checkpoint: str = "latest"
    dataset_repo_id: str | None = None
    gpu: str = "0"
    task: str = ""
    fps: int = Field(default=30, ge=1, le=240)
    actions_per_chunk: int | None = Field(default=None, ge=1)

    # Arena defaults match Captury/demo recording (alex_empty + LEVER_AGAIN).
    # For target=sim, clients should set environment to an Isaac Lab task id.
    environment: str = "alex_empty"
    embodiment: str = "alex_v2_ability_hands"
    usd: str = DEFAULT_ARENA_LEVER_USD
    num_episodes: int = Field(default=20, ge=1)
    headless: bool = True
    enable_cameras: bool = True
    video: bool = False
    camera_video: bool = False
    video_dir: str | None = None
    isaaclab_root: str = str(DEFAULT_ISAACLAB_ROOT)
    arena_root: str = str(DEFAULT_ARENA_ROOT)
    container_name: str = "isaaclab_arena-latest"
    container_workdir: str = "/workspaces/isaaclab_arena"
    python_executable: str = "/isaac-sim/python.sh"

    # Real Alex target. Motion remains capability-gated by the LeRobot adapter.
    ikstreamer_host: str = "127.0.0.1"
    ikstreamer_port: int = Field(default=2102, ge=1, le=65535)

    @model_validator(mode="after")
    def exactly_one_policy_source(self) -> RolloutConfig:
        if bool(self.job_id) == bool(self.policy_ref):
            raise ValueError("provide exactly one of job_id or policy_ref")
        if (
            self.policy_ref
            and "@" not in self.policy_ref
            and not Path(self.policy_ref).expanduser().exists()
            and not _HUB_REPO.fullmatch(self.policy_ref)
        ):
            # A plain owner/name Hub reference is valid. Other non-existent bare
            # paths are rejected early instead of failing in a remote container.
            raise ValueError("policy_ref must be a local path or Hub owner/name reference")
        return self


def build_isaaclab_rollout_command(
    config: RolloutConfig,
    inference_url: str,
    manifest: dict[str, Any],
    metrics_output: str | None = None,
) -> list[str]:
    """Build the direct Isaac Lab rollout command for the remote LeRobot adapter."""
    import json

    runner = str(Path(__file__).with_name("isaaclab_rollout_runner.py"))
    isaaclab_root = Path(config.isaaclab_root).expanduser()
    command = [
        str(isaaclab_root / "isaaclab.sh"),
        "-p",
        runner,
    ]
    if config.headless:
        command.append("--headless")
    if config.enable_cameras:
        command.append("--enable_cameras")
    command += [
        "--environment",
        config.environment,
        "--num_episodes",
        str(config.num_episodes),
        "--remote_url",
        inference_url,
        "--rollout_manifest",
        json.dumps(manifest, separators=(",", ":")),
        "--fps",
        str(config.fps),
    ]
    if config.task:
        command += ["--language_instruction", config.task]
    if config.video:
        command.append("--video")
    if config.camera_video:
        command.append("--camera_video")
    if config.video_dir:
        command += ["--video_dir", config.video_dir]
    if metrics_output:
        command += ["--metrics_output", metrics_output]
    command += ["--embodiment", config.embodiment]
    return command


class TeleopConfig(BaseModel):
    """A local Isaac Lab teleoperation session driving Alex directly."""

    environment: str = "Isaac-Alex-Lever-Play-v0"
    teleop_device: Literal["keyboard", "spacemouse", "gamepad", "handtracking"] = "keyboard"
    num_envs: int = Field(default=1, ge=1)
    sensitivity: float = Field(default=1.0, gt=0)
    isaaclab_root: str = str(DEFAULT_ISAACLAB_ROOT)


def build_isaaclab_teleop_command(config: TeleopConfig) -> list[str]:
    """Build the direct Isaac Lab teleoperation command for a local viewer session."""
    isaaclab_root = Path(config.isaaclab_root).expanduser()
    script = isaaclab_root / "scripts" / "environments" / "teleoperation" / "teleop_se3_agent.py"
    return [
        str(isaaclab_root / "isaaclab.sh"),
        "-p",
        str(script),
        "--task",
        config.environment,
        "--teleop_device",
        config.teleop_device,
        "--num_envs",
        str(config.num_envs),
        "--sensitivity",
        str(config.sensitivity),
    ]


def build_arena_rollout_command(
    config: RolloutConfig,
    inference_url: str,
    manifest: dict[str, Any],
    metrics_output: str | None = None,
) -> list[str]:
    """Build docker-exec Arena policy_runner command for a remote LeRobot policy."""
    import json

    del metrics_output  # Arena prints metrics; LeLab tracks exit code.
    runner = str(Path(config.container_workdir) / "isaaclab_arena/evaluation/policy_runner.py")
    command = [
        "docker",
        "exec",
        config.container_name,
        config.python_executable,
        runner,
        "--device",
        "cuda",
    ]
    if config.headless:
        command.append("--headless")
    if config.enable_cameras:
        command.append("--enable_cameras")
    command += [
        "--num_episodes",
        str(config.num_episodes),
        "--policy_type",
        _LEROBOT_REMOTE_POLICY,
        "--remote_url",
        inference_url,
        "--rollout_manifest",
        json.dumps(manifest, separators=(",", ":")),
    ]
    # No --policy_device: LeRobotRemotePolicy.add_args_to_parser doesn't register it, so an
    # unrecognized flag here shifts argparse's positional matching and "cuda" gets swallowed
    # as the environment subparser choice, raising "invalid choice: 'cuda'".
    if config.task:
        command += ["--language_instruction", config.task]
    if config.video:
        command.append("--video")
    if config.camera_video:
        command.append("--camera_video")
    if config.video_dir:
        command += ["--video_dir", config.video_dir]
    # Environment is an argparse subparser and must precede env-specific args.
    command += [config.environment, "--embodiment", config.embodiment]
    if config.usd:
        command += ["--usd", config.usd]
    return command


def build_dataset_conversion_command(config: DatasetConversionConfig) -> list[str]:
    root = Path(config.arena_root)
    if config.format == "gr00t":
        command = [
            "python",
            str(root / "isaaclab_arena_gr00t/lerobot/convert_lerobot_v3_to_gr00t.py"),
            "--input_dir",
            str(config.input_path),
            "--output_dir",
            config.output_path,
            "--modality_template",
            str(config.modality_template),
        ]
        if config.action_from_state_dims:
            command += ["--action_from_state_dims", config.action_from_state_dims]
        return command

    command = [
        "python",
        str(root / "isaaclab_arena_ccil/data/convert_lerobot_to_ccil.py"),
        "--out_file",
        config.output_path,
    ]
    command += ["--repo_id", config.repo_id] if config.repo_id else ["--dataset_path", str(config.input_path)]
    if config.image_keys:
        command += ["--image_keys", *config.image_keys, "--image_size", *(str(v) for v in config.image_size)]
    if config.output_image_keys:
        command += ["--output_image_keys", *config.output_image_keys]
    return command


def build_ccil_script(config: CCILTrainingConfig) -> str:
    overrides = [
        "data.pkl",
        config.pickle_path,
        "seed",
        str(config.seed),
        "policy.train_epochs",
        str(config.train_epochs),
        "policy.batch_size",
        str(config.batch_size),
    ]
    quoted = " ".join(shlex.quote(item) for item in [config.config_path, *overrides])
    if config.naive:
        return f"cd {shlex.quote(config.ccil_root)} && python correct_il/train_bc_policy.py {quoted} policy.naive true"
    return " && ".join(
        [
            f"cd {shlex.quote(config.ccil_root)}",
            f"python correct_il/train_dynamics_model.py {quoted}",
            f"python correct_il/gen_aug_label.py {quoted}",
            f"python correct_il/train_bc_policy.py {quoted} policy.naive false",
        ]
    )


def build_evaluation_command(config: EvaluationConfig) -> list[str]:
    runner = str(Path(config.container_workdir) / "isaaclab_arena/evaluation/policy_runner.py")
    command = [
        "docker",
        "exec",
        config.container_name,
        config.python_executable,
        runner,
        "--device",
        config.device,
    ]
    if config.headless:
        command.append("--headless")
    if config.enable_cameras:
        command.append("--enable_cameras")
    command += [
        "--num_episodes",
        str(config.num_episodes),
        "--policy_type",
        config.policy_type,
        "--model_path",
        config.model_path,
        "--policy_device",
        config.policy_device,
    ]
    if config.meta_path:
        command += ["--meta_path", config.meta_path]
    if config.language_instruction:
        command += ["--language_instruction", config.language_instruction]
    if config.video:
        command.append("--video")
    if config.camera_video:
        command.append("--camera_video")
    if config.video_dir:
        command += ["--video_dir", config.video_dir]
    # Environment is an argparse subparser and must precede env-specific args.
    command += [config.environment, "--embodiment", config.embodiment]
    return command
