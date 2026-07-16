import shlex

from lelab.alex_models import DatasetEvalConfig
from lelab.dataset_eval import DatasetEvalManager, build_dataset_eval_docker_command


class FakeCluster:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def status(self):
        return {"connected": True}

    def gpus(self):
        return [{"index": 0, "uuid": "GPU-aaa"}]

    def execute(self, command: str, timeout: float = 30):
        self.commands.append(command)
        if "docker inspect" in command:
            return 0, '{"Running":false,"ExitCode":0,"Error":""}', ""
        if "docker logs" in command:
            return 0, 'ALEX_DATASET_EVAL_RESULT={"eval_loss":0.123,"num_episodes":2}\n', ""
        return 0, "container-id\n", ""


class FakeJobs:
    def get(self, job_id: str, refresh: bool = False):
        class Job:
            config = {
                "dataset_repo_id": "user/alex-data",
                "dataset_revision": "v3.0",
                "model_repo_id": "user/alex-groot",
            }

        return Job()


def test_dataset_eval_docker_command_passes_single_episode_json_token() -> None:
    command = build_dataset_eval_docker_command(
        "eval-1",
        DatasetEvalConfig(
            policy_ref="user/alex-groot",
            dataset_repo_id="user/alex-data",
            dataset_revision="v1.2",
            dataset_episodes=[0, 1, 5],
        ),
        "user/alex-groot@latest",
        "user/alex-data",
        "v1.2",
        "GPU-aaa",
    )
    argv = shlex.split(command[command.index("docker run") :])
    assert argv[argv.index("--dataset-revision") + 1] == "v1.2"
    assert argv[argv.index("--dataset-episodes") + 1] == "[0, 1, 5]"
    assert argv[argv.index("--policy-ref") + 1] == "user/alex-groot@latest"
    assert "ALEX_DATASET_EVAL_RESULT" in argv[argv.index("-c") + 1]


def test_dataset_eval_manager_resolves_job_and_parses_metrics(tmp_path) -> None:
    cluster = FakeCluster()
    manager = DatasetEvalManager(tmp_path, cluster=cluster, jobs=FakeJobs())
    record = manager.start(
        DatasetEvalConfig(
            job_id="job-1",
            checkpoint="1000",
            dataset_episodes=[0, 1],
            gpu="0",
        )
    )
    assert record.policy_ref == "user/alex-groot@checkpoints/001000"
    assert record.dataset_repo_id == "user/alex-data"
    assert "docker run" in cluster.commands[0]
    argv = shlex.split(cluster.commands[0][cluster.commands[0].index("docker run") :])
    assert argv[argv.index("--dataset-revision") + 1] == "v3.0"

    finished = manager.get(record.id)
    assert finished.state == "done"
    assert finished.metrics["eval_loss"] == 0.123
