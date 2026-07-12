import ast
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CORE_PACKAGE = PROJECT_ROOT / "src" / "kinematics" / "core"
CLI_ONLY_DEPENDENCIES = ("matplotlib", "pyarrow", "typer", "yaml")
PUBLIC_CORE_MODULES = {
    "kinematics.core",
    "kinematics.core.export",
    "kinematics.core.topology",
    "kinematics.core.types",
}


def test_core_import_succeeds_without_cli_dependencies() -> None:
    blocked_modules = ", ".join(repr(name) for name in CLI_ONLY_DEPENDENCIES)
    script = (
        "import sys\n"
        f"for name in ({blocked_modules},): sys.modules[name] = None\n"
        "import kinematics\n"
        "import kinematics.core\n"
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


def test_cli_uses_documented_public_core_modules() -> None:
    forbidden_imports: list[str] = []
    cli_package = PROJECT_ROOT / "src" / "kinematics" / "cli"
    for path in cli_package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if node.module.startswith("kinematics.core") and (
                node.module not in PUBLIC_CORE_MODULES
            ):
                forbidden_imports.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
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
