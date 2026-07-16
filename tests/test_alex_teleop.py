from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lelab.alex_models import TeleopConfig
from lelab.alex_teleop import TeleopManager


def test_teleop_requires_isaaclab_launcher(tmp_path: Path) -> None:
    manager = TeleopManager(tmp_path)
    with pytest.raises(FileNotFoundError, match="Isaac Lab launcher"):
        manager.start(TeleopConfig(isaaclab_root=str(tmp_path / "missing")))
    assert manager.list() == []


def test_teleop_launches_local_isaaclab_process(tmp_path: Path, monkeypatch) -> None:
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir()
    launcher = isaaclab / "isaaclab.sh"
    launcher.write_text("#!/usr/bin/env bash\n")

    manager = TeleopManager(tmp_path / "teleop")

    popen = MagicMock()
    popen.return_value.pid = 4321
    monkeypatch.setattr("lelab.alex_teleop.subprocess.Popen", popen)

    record = manager.start(TeleopConfig(isaaclab_root=str(isaaclab)))

    assert record.state == "running"
    assert record.pid == 4321
    command = popen.call_args.args[0]
    assert command[:2] == [str(launcher), "-p"]
    assert Path(command[2]).name == "teleop_se3_agent.py"
    assert popen.call_args.kwargs["cwd"] == str(isaaclab.resolve())
    assert "CONDA_PREFIX" not in popen.call_args.kwargs["env"]
    assert (tmp_path / "teleop" / record.id / "session.json").is_file()


def test_teleop_stop_signals_process_group(tmp_path: Path, monkeypatch) -> None:
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir()
    (isaaclab / "isaaclab.sh").write_text("#!/usr/bin/env bash\n")

    manager = TeleopManager(tmp_path / "teleop")

    process = MagicMock()
    process.pid = 4321
    process.poll.return_value = None
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("lelab.alex_teleop.subprocess.Popen", popen)

    killpg = MagicMock()
    monkeypatch.setattr("lelab.alex_teleop.os.killpg", killpg)

    record = manager.start(TeleopConfig(isaaclab_root=str(isaaclab)))
    stopped = manager.stop(record.id)

    assert stopped.state == "stopped"
    killpg.assert_any_call(4321, 15)  # SIGTERM


def test_teleop_stop_requires_running_session(tmp_path: Path) -> None:
    manager = TeleopManager(tmp_path)
    with pytest.raises(KeyError):
        manager.stop("unknown")
