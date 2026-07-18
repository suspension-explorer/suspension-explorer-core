import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
CORE_PACKAGE = PROJECT_ROOT / "src" / "kinematics" / "core"
CLI_ONLY_DEPENDENCIES = ("matplotlib", "pyarrow", "typer", "yaml")
PUBLIC_CORE_MODULES = {
    "kinematics.core.assembly",
    "kinematics.core.elements",
    "kinematics.core.export",
    "kinematics.core.input",
    "kinematics.core.metrics.main",
    "kinematics.core.metrics.registry",
    "kinematics.core.enums",
    "kinematics.core.primitives.geometry",
    "kinematics.core.primitives.point_ref",
    "kinematics.core.presentation",
    "kinematics.core.schema.geometry",
    "kinematics.core.schema.sweep",
    "kinematics.core.solver",
    "kinematics.core.state",
    "kinematics.core.suspensions.base",
    "kinematics.core.suspensions.build",
    "kinematics.core.sweep",
    "kinematics.core.targeting",
}


def test_core_import_succeeds_without_cli_dependencies() -> None:
    blocked_modules = ", ".join(repr(name) for name in CLI_ONLY_DEPENDENCIES)
    script = (
        "import sys\n"
        f"for name in ({blocked_modules},): sys.modules[name] = None\n"
        "import kinematics\n"
        "import kinematics.core\n"
        "from kinematics.core.input import build_suspension, build_sweep\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_low_level_core_import_does_not_load_solver_stack() -> None:
    script = (
        "import sys\n"
        "from kinematics.core.enums import Axis\n"
        "assert Axis.X.value == 0\n"
        "assert 'scipy' not in sys.modules\n"
        "assert 'kinematics.core.analysis' not in sys.modules\n"
        "assert 'kinematics.core.solver' not in sys.modules\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("typer") is None,
    reason="Requires the CLI extra; the core-only CI environment omits typer.",
)
def test_cli_app_import_does_not_load_sweep_runtime() -> None:
    script = (
        "import sys\n"
        "from kinematics.cli.app import app\n"
        "assert app is not None\n"
        "assert 'pyarrow' not in sys.modules\n"
        "assert 'scipy' not in sys.modules\n"
        "assert 'kinematics.cli.commands.sweep' not in sys.modules\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_core_does_not_import_cli_package() -> None:
    forbidden_imports: list[str] = []
    for path in CORE_PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "kinematics.cli" or node.module.startswith(
                    "kinematics.cli."
                ):
                    forbidden_imports.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "kinematics.cli" or alias.name.startswith(
                        "kinematics.cli."
                    ):
                        forbidden_imports.append(
                            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
                        )

    assert forbidden_imports == []


def test_core_does_not_import_cli_only_dependencies() -> None:
    forbidden_imports: list[str] = []
    for path in CORE_PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported_modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
            elif isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            if any(
                module.split(".", maxsplit=1)[0] in CLI_ONLY_DEPENDENCIES
                for module in imported_modules
            ):
                forbidden_imports.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{getattr(node, 'lineno', 0)}"
                )

    assert forbidden_imports == []


def test_schema_and_build_modules_do_not_hide_dispatch_imports() -> None:
    forbidden_imports: list[str] = []
    for relative_path in (
        "schema/geometry.py",
        "suspensions/build.py",
    ):
        path = CORE_PACKAGE / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(
                isinstance(child, (ast.Import, ast.ImportFrom))
                for child in ast.walk(node)
            ):
                forbidden_imports.append(f"{relative_path}:{node.lineno}")

    assert forbidden_imports == []


def test_cli_uses_documented_public_core_modules() -> None:
    # This allowlist is the adapter API contract. Adding a module deliberately
    # expands the supported surface and should receive architecture review.
    forbidden_imports: list[str] = []
    cli_package = PROJECT_ROOT / "src" / "kinematics" / "cli"
    for path in cli_package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported_modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
            elif isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            if any(
                module.startswith("kinematics.core")
                and module not in PUBLIC_CORE_MODULES
                for module in imported_modules
            ):
                forbidden_imports.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{getattr(node, 'lineno', 0)}"
                )

    assert forbidden_imports == []


def test_console_bootstrap_reports_missing_cli_dependencies() -> None:
    blocked_modules = ", ".join(repr(name) for name in CLI_ONLY_DEPENDENCIES)
    script = (
        "import sys\n"
        f"for name in ({blocked_modules},): sys.modules[name] = None\n"
        "from kinematics.cli.bootstrap import main\n"
        "main()\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 1
    assert 'Install with "kinematics[cli]"' in result.stderr
