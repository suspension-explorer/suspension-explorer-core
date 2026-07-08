"""
File IO.

YAML loading (files -> validated specs -> live objects) and results writing
(solved sweeps -> Parquet/CSV). This is the only package that touches the
filesystem; programmatic consumers use ``kinematics.schema`` directly and
never import from here.
"""

from kinematics.io.loaders import load_geometry, load_sweep
from kinematics.io.results_writer import (
    CsvWriter,
    ParquetWriter,
    SolutionFrame,
    create_writer_for_path,
)

__all__ = [
    "CsvWriter",
    "ParquetWriter",
    "SolutionFrame",
    "create_writer_for_path",
    "load_geometry",
    "load_sweep",
]
