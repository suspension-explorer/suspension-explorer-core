"""Smoke tests for repository-level helper scripts."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

matplotlib = pytest.importorskip(
    "matplotlib",
    reason="visualization smoke tests require the optional viz extra",
)
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_BUMP_SWEEP = REPO_ROOT / "scripts" / "plot_bump_sweep.py"
VISUALIZE_CAMBER_SHIM = REPO_ROOT / "visualize_camber_shim.py"


def _load_script(path: Path) -> Any:
    """Import a standalone script by file path."""
    name = f"_script_smoke_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_plot_bump_sweep_imports() -> None:
    module = _load_script(PLOT_BUMP_SWEEP)
    assert callable(module.main)


def test_visualize_camber_shim_imports() -> None:
    module = _load_script(VISUALIZE_CAMBER_SHIM)
    assert callable(module.main)
    assert callable(module.plot_front_view_comparison)


def test_camber_shim_setup_reconstruction_preserves_suspension_data(
    test_data_dir: Path,
) -> None:
    """Exercise the setup reconstruction used by the script's main path."""
    from kinematics.cli.io.loaders import load_geometry
    from kinematics.core.primitives.point_ref import Side

    module = _load_script(VISUALIZE_CAMBER_SHIM)
    suspension = load_geometry(test_data_dir / "geometry.yaml")
    assert suspension.config is not None
    assert suspension.config.camber_shim is not None
    suspension.initial_state()
    setup_shim = suspension.config.camber_shim.model_copy(
        update={"setup_thickness": 0.0}
    )

    setup_suspension = module.create_setup_suspension(suspension, setup_shim)

    assert type(setup_suspension) is type(suspension)
    assert setup_suspension.side is Side.LEFT
    assert setup_suspension.name == suspension.name
    assert setup_suspension.version == suspension.version
    assert setup_suspension.units is suspension.units
    assert setup_suspension.hardpoints is not suspension.hardpoints
    assert setup_suspension.config is not None
    assert setup_suspension.config.camber_shim == setup_shim
    setup_suspension.initial_state()


def test_camber_shim_front_view_renders(tmp_path: Path, test_data_dir: Path) -> None:
    """Exercise the camber-shim comparison plot with the headless backend."""
    from kinematics.cli.io.loaders import load_geometry

    module = _load_script(VISUALIZE_CAMBER_SHIM)
    suspension = load_geometry(test_data_dir / "geometry.yaml")
    output = tmp_path / "comparison.png"

    module.plot_front_view_comparison(
        suspension,
        suspension,
        output,
        shim_delta=0.0,
    )

    assert output.exists()
    assert output.stat().st_size > 0


@pytest.mark.manual
def test_plot_bump_sweep_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the full bump-sweep script, including its MP4 animation."""
    if shutil.which("ffmpeg") is None:
        # pytest wraps skip() so its stub drops the reason parameter from view.
        pytest.skip("ffmpeg not available")  # ty: ignore[too-many-positional-arguments]

    module = _load_script(PLOT_BUMP_SWEEP)
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    module.main()

    assert (tmp_path / "dashboard.png").exists()
    assert (tmp_path / "bump_sweep.mp4").exists()
