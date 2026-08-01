"""CLI export coverage for measured element-length sweep coordinates."""

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from kinematics.cli.commands.sweep import run_sweep_files

DATA_DIR = Path(__file__).parent / "data"
TARGET_COLUMN = "target_damper_length_left"


def _write_sweep(path: Path) -> None:
    path.write_text(
        """\
version: 1
targets:
  - kind: element_length
    element: damper
    side: left
    mode: relative
    values: [0, 1, -1]
""",
        encoding="utf-8",
    )


def test_csv_exports_measured_element_coordinate_and_unit(tmp_path: Path) -> None:
    sweep_path = tmp_path / "damper-sweep.yaml"
    output_path = tmp_path / "result.csv"
    _write_sweep(sweep_path)

    run = run_sweep_files(
        DATA_DIR / "trailing_arm_coilover_geometry.yaml",
        sweep_path,
        output_path,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    unit_line = next(line for line in lines if line.startswith("# column_units:"))
    units = json.loads(unit_line.partition(":")[2].strip())
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    expected = [
        run.evaluated.states[index] for index in range(len(run.evaluated.states))
    ]
    assert TARGET_COLUMN in rows[0]
    assert units[TARGET_COLUMN] == "mm"
    coordinate = run.suspension.drive_coordinates()[0].target(0.0)
    assert [float(row[TARGET_COLUMN]) for row in rows] == pytest.approx(
        [coordinate.measure(state.positions) for state in expected]
    )


def test_parquet_exports_measured_element_coordinate_and_unit(tmp_path: Path) -> None:
    sweep_path = tmp_path / "damper-sweep.yaml"
    output_path = tmp_path / "result.parquet"
    _write_sweep(sweep_path)

    run = run_sweep_files(
        DATA_DIR / "trailing_arm_coilover_geometry.yaml",
        sweep_path,
        output_path,
    )
    table = pq.read_table(output_path)
    field = table.schema.field(TARGET_COLUMN)
    coordinate = run.suspension.drive_coordinates()[0].target(0.0)

    assert field.metadata == {b"unit": b"mm"}
    assert table[TARGET_COLUMN].to_pylist() == pytest.approx(
        [coordinate.measure(state.positions) for state in run.evaluated.states]
    )
