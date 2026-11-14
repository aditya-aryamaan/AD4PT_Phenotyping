from unittest.mock import Mock

import pytest

from ad4pt import cli
from ad4pt.cli import main


def test_default_and_demo_command(monkeypatch):
    command = Mock(return_value=0)
    monkeypatch.setattr(cli, "demo_command", command)
    assert main([]) == 0
    assert main(["demo"]) == 0
    defaults, explicit = [call.args[0] for call in command.call_args_list]
    assert defaults.device == explicit.device == "mps"
    assert defaults.run_root == explicit.run_root


def test_removed_commands_are_not_exposed():
    for command in ("recover", "stream", "images", "video", "detect"):
        with pytest.raises(SystemExit):
            main([command])
