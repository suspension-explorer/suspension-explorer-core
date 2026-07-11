"""Tests for physical-unit metadata at flat result-file boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from kinematics.io.results_writer import CsvWriter, ParquetWriter, SolutionFrame
from kinematics.metrics.registry import MetricSpec
from kinematics.metrics.units import MetricUnit
from kinematics.solver import SolverInfo


def _frame() -> SolutionFrame:
    spec = MetricSpec("camber", "Camber", MetricUnit.DEG, "state", "corner")
    return SolutionFrame(
        positions={"wheel_center": (1.0, 2.0, 3.0)},
        solver_info=SolverInfo(True, 3, 1e-8),
        metrics={"camber": -1.25},
        metric_specs={"camber": spec},
    )


def test_csv_writes_explicit_column_unit_metadata(tmp_path: Path) -> None:
    output = tmp_path / "result.csv"
    writer = CsvWriter(output)
    writer.add_frame(0, _frame())

    writer.write()

    units_line = next(
        line
        for line in output.read_text().splitlines()
        if line.startswith("# column_units:")
    )
    units = json.loads(units_line.partition(":")[2].strip())
    assert units["camber"] == "deg"
    assert units["wheel_center_z"] == "mm"


def test_parquet_writes_units_on_arrow_fields(tmp_path: Path) -> None:
    output = tmp_path / "result.parquet"
    writer = ParquetWriter(output)
    writer.add_frame(0, _frame())

    writer.write()

    schema = pq.read_schema(output)
    assert schema.field("camber").metadata == {b"unit": b"deg"}
    assert schema.field("wheel_center_x").metadata == {b"unit": b"mm"}
