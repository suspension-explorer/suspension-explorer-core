"""
Dependency-light console entry point for the optional CLI.
"""

import sys

CLI_DEPENDENCIES = frozenset({"pyarrow", "typer", "yaml"})


def main() -> None:
    """
    Run the CLI or report how to install its optional dependencies.
    """
    try:
        from kinematics.cli.app import app

        app()
    except ModuleNotFoundError as error:
        if error.name not in CLI_DEPENDENCIES:
            raise
        print(
            'The kinematics CLI is not installed. Install with "kinematics[cli]".',
            file=sys.stderr,
        )
        raise SystemExit(1) from error
