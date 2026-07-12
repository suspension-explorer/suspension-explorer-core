from importlib.metadata import entry_points

from typer.testing import CliRunner

from kinematics.cli.app import app


def test_console_script_targets_cli_adapter() -> None:
    script = next(iter(entry_points(group="console_scripts", name="kinematics")))

    assert script.value == "kinematics.cli.bootstrap:main"


def test_cli_help_lists_compatible_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "sweep" in result.stdout
    assert "visualize" in result.stdout
