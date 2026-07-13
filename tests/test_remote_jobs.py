import shlex

from lelab.alex_models import LeRobotTrainingConfig, RemoteTrainingRequest
from lelab.remote_jobs import RemoteJobManager, build_remote_docker_command


class FakeCluster:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def status(self):
        return {"connected": True}

    def gpus(self):
        return [
            {"index": 0, "uuid": "GPU-aaa"},
            {"index": 1, "uuid": "GPU-bbb"},
        ]

    def execute(self, command: str, timeout: float = 30):
        self.commands.append(command)
        return 0, "container-id\n", ""


def test_docker_command_is_named_detached_and_multi_gpu() -> None:
    config = LeRobotTrainingConfig(
        dataset_repo_id="user/alex",
        model_repo_id="user/alex-act",
        policy_type="act",
    )
    command = build_remote_docker_command("job-1", config, ["GPU-aaa", "GPU-bbb"])
    assert "docker run --detach --name alex-job-1" in command
    assert "device=GPU-aaa,GPU-bbb" in command
    argv = shlex.split(command[command.index("docker run") :])
    assert argv[argv.index("--gpus") + 1] == '"device=GPU-aaa,GPU-bbb"'
    assert "export HF_TOKEN" in command
    assert "hf auth login" in command
    assert "$HOME/.local/bin/hf" in command
    assert "$HOME/miniconda3/bin/hf" in command
    assert "accelerate launch --multi_gpu --num_processes 2 --module lerobot.scripts.lerobot_train" in command
    assert "--env HF_TOKEN" in command
    assert "alex_job-1_checkpoints:/outputs" in command
    assert "secret" not in command.lower()


def test_remote_job_persists_nonsecret_record_and_reservation(tmp_path) -> None:
    cluster = FakeCluster()
    manager = RemoteJobManager(tmp_path, cluster=cluster)
    request = RemoteTrainingRequest(
        config={
            "kind": "lerobot",
            "dataset_repo_id": "user/alex",
            "model_repo_id": "user/alex-act",
            "policy_type": "act",
        },
        gpus=["0", "GPU-bbb"],
    )
    record = manager.start(request)
    assert record.state == "running"
    assert record.gpu_uuids == ["GPU-aaa", "GPU-bbb"]
    assert manager.reservations() == {"GPU-aaa": record.id, "GPU-bbb": record.id}
    persisted = (tmp_path / record.id / "job.json").read_text()
    assert "password" not in persisted.lower()
    assert record.container_id == "container-id"


def test_remote_job_rejects_reserved_gpu(tmp_path) -> None:
    cluster = FakeCluster()
    manager = RemoteJobManager(tmp_path, cluster=cluster)
    request = RemoteTrainingRequest(
        config={
            "kind": "lerobot",
            "dataset_repo_id": "user/alex",
            "model_repo_id": "user/alex-act",
            "policy_type": "act",
        },
        gpus=["0"],
    )
    manager.start(request)
    try:
        manager.start(request)
    except ValueError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("expected reservation conflict")


def test_remote_job_manager_loads_legacy_ccil_history(tmp_path) -> None:
    import json

    job_dir = tmp_path / "ccil-old"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "id": "ccil-old",
                "name": "Old CCIL",
                "state": "done",
                "kind": "ccil",
                "container_name": "alex-ccil-old",
                "gpu_ids": ["0"],
                "gpu_uuids": ["GPU-aaa"],
                "config": {"kind": "ccil", "pickle_path": "/old/data.pkl"},
                "started_at": 1.0,
                "ended_at": 2.0,
                "log_path": str(job_dir / "docker.log"),
            }
        )
    )
    manager = RemoteJobManager(tmp_path, cluster=FakeCluster())
    assert manager.get("ccil-old", refresh=False).kind == "ccil"
