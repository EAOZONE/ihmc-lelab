from unittest.mock import MagicMock

from lelab.alex_models import EvaluationConfig
from lelab.evaluation import EvaluationManager


def test_evaluation_manager_starts_and_finishes_mocked_process(tmp_path, monkeypatch) -> None:
    arena = tmp_path / "arena"
    script = arena / "isaaclab_arena" / "evaluation" / "policy_runner.py"
    script.parent.mkdir(parents=True)
    script.write_text("# test")

    process = MagicMock()
    process.pid = 4242
    process.poll.return_value = 0
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("lelab.evaluation.subprocess.Popen", popen)

    manager = EvaluationManager(tmp_path / "records")
    record = manager.start(
        EvaluationConfig(
            policy_type="zero_action",
            model_path="/tmp/policy.pt",
            arena_root=str(arena),
            python_executable="python",
        )
    )
    assert record.state == "running"
    assert popen.call_args.kwargs["cwd"] == str(arena)
    finished = manager.get(record.id)
    assert finished.state == "done"
    assert finished.exit_code == 0
