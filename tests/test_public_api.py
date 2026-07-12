import importlib
import sys


def test_no_matplotlib_import_on_core():
    # ensure matplotlib is not pulled in by importing kinematics
    sys.modules.pop("matplotlib", None)

    importlib.invalidate_caches()
    importlib.import_module("kinematics")
    assert "matplotlib" not in sys.modules
