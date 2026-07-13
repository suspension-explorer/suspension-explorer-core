"""
End-to-end integration tests for the CLI interface using direct CLI function calls.

Tests all combinations of:
- Output formats: CSV and Parquet
- With and without visualization
- Using real test data files and 'golden' reference files

This version calls the CLI sweep() function directly instead of using subprocess,
making debugging easier while still testing the same code path.
"""

import csv
import io
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest

from kinematics.cli.app import sweep as cli_sweep
from kinematics.core.primitives.constants import TEST_TOLERANCE
from kinematics.core.sweep import solve_evaluated_sweep

# Check if matplotlib is available for animation tests without importing it.
HAS_MATPLOTLIB = find_spec("matplotlib") is not None

requires_viz = pytest.mark.skipif(
    not HAS_MATPLOTLIB,
    reason="matplotlib not installed (install with: uv pip install -e '.[cli,viz]')",
)

# Columns that contain solver internals which vary across platforms.
# These are excluded from numerical comparison.
SOLVER_METADATA_COLUMNS = {"solver_max_residual", "solver_nfev"}


def load_csv_data(file_path: Path) -> tuple[list[str], list[list[str]]]:
    """
    Load CSV data and return headers and rows, skipping comment lines.
    """
    with open(file_path, "r") as f:
        # Skip comment lines starting with #
        lines = []
        for line in f:
            if not line.strip().startswith("#"):
                lines.append(line)

        # Use csv reader on the non-comment lines
        reader = csv.reader(lines)
        headers = next(reader)
        rows = list(reader)
    return headers, rows


def load_parquet_data(file_path: Path) -> tuple[list[str], list[list[str]]]:
    """
    Load Parquet data and return headers and rows as strings for comparison.
    """
    table = pq.read_table(file_path)
    headers = table.column_names

    # Convert to list of rows (as strings for comparison).
    rows = []
    for i in range(table.num_rows):
        row = []
        for col_name in headers:
            value = table.column(col_name)[i].as_py()
            row.append(str(value))
        rows.append(row)

    return headers, rows


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test outputs.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_data_dir() -> Path:
    """
    Path to test data directory.
    """
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def geometry_file(test_data_dir: Path) -> Path:
    """
    Path to test geometry file.
    """
    return test_data_dir / "geometry.yaml"


@pytest.fixture
def sweep_file(test_data_dir: Path) -> Path:
    """
    Path to test sweep file.
    """
    return test_data_dir / "sweep.yaml"


def run_cli_sweep_direct(
    geometry_file: Path,
    sweep_file: Path,
    output_file: Path,
    animation_file: Path | None = None,
) -> tuple[bool, str]:
    """
    Run the CLI sweep function directly instead of via subprocess.

    Args:
        geometry_file: Path to geometry YAML file
        sweep_file: Path to sweep YAML file
        output_file: Output path for results (.csv or .parquet)
        animation_file: Optional animation output path

    Returns:
        Tuple of (success, captured_output) indicating result
    """
    # Capture typer.echo() output
    captured_output = io.StringIO()

    try:
        # Mock typer.echo to capture output instead of printing
        with patch("typer.echo") as mock_echo:
            # Store all echo calls
            echo_calls = []

            def capture_echo(message, **kwargs):
                echo_calls.append(str(message))
                captured_output.write(str(message) + "\n")

            mock_echo.side_effect = capture_echo

            # Call the CLI function directly
            cli_sweep(
                geometry=geometry_file,
                sweep=sweep_file,
                out=output_file,
                animation_out=animation_file,
            )

            # If we get here, it succeeded
            return True, captured_output.getvalue()

    except Exception as e:
        return False, str(e)


def compare_numerical_csv(
    actual_file: Path,
    expected_file: Path,
) -> None:
    """
    Compare two CSV files using numerical tolerances.

    Skips comment lines, timestamp columns, and solver metadata columns.
    Asserts that all position/geometry columns match within tolerance.

    Args:
        actual_file: Path to the generated CSV file
        expected_file: Path to the expected reference CSV file
    """
    actual_headers, actual_rows = load_csv_data(actual_file)
    expected_headers, expected_rows = load_csv_data(expected_file)

    # Headers must match exactly (column structure shouldn't change).
    assert actual_headers == expected_headers, (
        f"CSV headers differ:\n"
        f"  actual:   {actual_headers}\n"
        f"  expected: {expected_headers}"
    )

    assert len(actual_rows) == len(expected_rows), (
        f"Row count differs: {len(actual_rows)} vs {len(expected_rows)}"
    )

    for row_idx, (actual_row, expected_row) in enumerate(
        zip(actual_rows, expected_rows)
    ):
        for col_idx, col_name in enumerate(actual_headers):
            # Skip solver metadata -- these vary across platforms.
            if col_name in SOLVER_METADATA_COLUMNS:
                continue

            actual_val = actual_row[col_idx]
            expected_val = expected_row[col_idx]

            # Try numeric comparison first, fall back to exact string match
            # for non-numeric columns (e.g. booleans, step_index).
            try:
                a = float(actual_val)
                e = float(expected_val)
                np.testing.assert_allclose(
                    a,
                    e,
                    atol=TEST_TOLERANCE,
                    rtol=TEST_TOLERANCE,
                    err_msg=(
                        f"Row {row_idx}, column '{col_name}': "
                        f"{actual_val} != {expected_val}"
                    ),
                )
            except ValueError:
                assert actual_val == expected_val, (
                    f"Row {row_idx}, column '{col_name}': "
                    f"'{actual_val}' != '{expected_val}'"
                )


def compare_numerical_parquet(
    actual_file: Path,
    expected_file: Path,
) -> None:
    """
    Compare two Parquet files using numerical tolerances.

    Args:
        actual_file: Path to the generated Parquet file
        expected_file: Path to the expected reference Parquet file
    """
    actual_table = pq.read_table(actual_file)
    expected_table = pq.read_table(expected_file)

    actual_cols = actual_table.column_names
    expected_cols = expected_table.column_names

    assert actual_cols == expected_cols, (
        f"Parquet columns differ:\n"
        f"  actual:   {actual_cols}\n"
        f"  expected: {expected_cols}"
    )
    assert actual_table.num_rows == expected_table.num_rows, (
        f"Row count differs: {actual_table.num_rows} vs {expected_table.num_rows}"
    )

    for col_name in actual_cols:
        if col_name in SOLVER_METADATA_COLUMNS:
            continue

        actual_col = actual_table.column(col_name).to_pylist()
        expected_col = expected_table.column(col_name).to_pylist()

        # Check if numeric by inspecting first non-None value.
        first_val = next((v for v in actual_col if v is not None), None)
        if isinstance(first_val, (int, float)):
            np.testing.assert_allclose(
                actual_col,
                expected_col,
                atol=TEST_TOLERANCE,
                rtol=TEST_TOLERANCE,
                err_msg=f"Column '{col_name}' values differ",
            )
        else:
            assert actual_col == expected_col, (
                f"Column '{col_name}': {actual_col} != {expected_col}"
            )


def validate_output_against_reference(output_file: Path, file_format: str) -> None:
    """
    Validate output file against reference file using numerical tolerances.

    Args:
        output_file: Path to the generated output file
        file_format: Expected format ('csv' or 'parquet')
    """
    reference_file = (
        Path(__file__).parent.parent / "data" / "e2e" / f"output.{file_format}"
    )

    assert output_file.exists(), f"Output file {output_file} was not created"
    assert output_file.stat().st_size > 0, f"Output file {output_file} is empty"
    assert reference_file.exists(), f"Reference file {reference_file} does not exist"

    if file_format == "csv":
        compare_numerical_csv(output_file, reference_file)
    elif file_format == "parquet":
        compare_numerical_parquet(output_file, reference_file)
    else:
        raise ValueError(f"Unsupported format: {file_format}")


def validate_animation_file(animation_file: Path) -> None:
    """
    Validate the animation file exists and has reasonable size.

    Args:
        animation_file: Path to the animation file
    """
    assert animation_file.exists(), f"Animation file {animation_file} was not created"
    file_size = animation_file.stat().st_size
    assert file_size > 1000, (
        f"Animation file {animation_file} is too small: {file_size} bytes"
    )


class TestCliEndToEnd:
    """
    End-to-end tests for CLI sweep output.

    Non-viz tests run by default (including CI). Viz tests require matplotlib
    and are skipped when it is not installed.
    """

    def test_csv_output(
        self,
        temp_dir: Path,
        geometry_file: Path,
        sweep_file: Path,
    ) -> None:
        output_file = temp_dir / "test_output.csv"

        success, output = run_cli_sweep_direct(geometry_file, sweep_file, output_file)

        assert success, f"CLI sweep failed: {output}"
        assert "wrote" in output.lower(), f"Unexpected output: {output}"
        validate_output_against_reference(output_file, "csv")

    def test_parquet_output(
        self,
        temp_dir: Path,
        geometry_file: Path,
        sweep_file: Path,
    ) -> None:
        output_file = temp_dir / "test_output.parquet"

        success, output = run_cli_sweep_direct(geometry_file, sweep_file, output_file)

        assert success, f"CLI sweep failed: {output}"
        assert "wrote" in output.lower(), f"Unexpected output: {output}"
        validate_output_against_reference(output_file, "parquet")

    def test_output_formats_produce_same_data(
        self,
        temp_dir: Path,
        geometry_file: Path,
        sweep_file: Path,
    ) -> None:
        csv_output = temp_dir / "test_output.csv"
        parquet_output = temp_dir / "test_output.parquet"

        csv_success, csv_message = run_cli_sweep_direct(
            geometry_file, sweep_file, csv_output
        )
        parquet_success, parquet_message = run_cli_sweep_direct(
            geometry_file, sweep_file, parquet_output
        )

        assert csv_success, f"CSV command failed: {csv_message}"
        assert parquet_success, f"Parquet command failed: {parquet_message}"

        validate_output_against_reference(csv_output, "csv")
        validate_output_against_reference(parquet_output, "parquet")

    def test_invalid_geometry_file(
        self,
        temp_dir: Path,
        sweep_file: Path,
    ) -> None:
        invalid_geometry = temp_dir / "nonexistent.yaml"
        output_file = temp_dir / "test_output.csv"

        success, output = run_cli_sweep_direct(
            invalid_geometry, sweep_file, output_file
        )

        assert not success, "CLI should fail with invalid geometry file"
        assert output  # Should have an error message

    def test_invalid_sweep_file(
        self,
        temp_dir: Path,
        geometry_file: Path,
    ) -> None:
        invalid_sweep = temp_dir / "nonexistent.yaml"
        output_file = temp_dir / "test_output.csv"

        success, output = run_cli_sweep_direct(
            geometry_file, invalid_sweep, output_file
        )

        assert not success, "CLI should fail with invalid sweep file"
        assert output  # Should have an error message

    @requires_viz
    def test_animation_reuses_the_primary_sweep_solution(
        self,
        temp_dir: Path,
        geometry_file: Path,
        sweep_file: Path,
    ) -> None:
        output_file = temp_dir / "test_output.csv"
        animation_file = temp_dir / "test_animation.gif"

        with (
            patch(
                "kinematics.cli.commands.sweep.solve_evaluated_sweep",
                wraps=solve_evaluated_sweep,
            ) as solve_mock,
            patch("kinematics.cli.visualization.api.visualize_suspension_sweep"),
        ):
            success, output = run_cli_sweep_direct(
                geometry_file,
                sweep_file,
                output_file,
                animation_file,
            )

        assert success, output
        assert solve_mock.call_count == 1

    @requires_viz
    def test_axle_geometry_visualization_uses_both_contact_patches(
        self,
        temp_dir: Path,
        test_data_dir: Path,
    ) -> None:
        from kinematics.cli.io.loaders import load_geometry
        from kinematics.cli.visualization.api import visualize_geometry

        output_file = temp_dir / "axle_geometry.png"
        suspension = load_geometry(test_data_dir / "axle_geometry.yaml")

        result = visualize_geometry(suspension, output_file)

        assert output_file.exists()
        assert len(result.contact_patch_z) == 2
        assert result.contact_patch_on_ground

    @requires_viz
    def test_csv_output_with_animation(
        self,
        temp_dir: Path,
        geometry_file: Path,
        sweep_file: Path,
    ) -> None:
        output_file = temp_dir / "test_output.csv"
        animation_file = temp_dir / "test_animation.mp4"

        success, output = run_cli_sweep_direct(
            geometry_file, sweep_file, output_file, animation_file
        )

        assert success, f"CLI sweep failed: {output}"
        assert "wrote" in output.lower(), f"Unexpected output: {output}"
        validate_output_against_reference(output_file, "csv")
        validate_animation_file(animation_file)

    @requires_viz
    def test_parquet_output_with_animation(
        self,
        temp_dir: Path,
        geometry_file: Path,
        sweep_file: Path,
    ) -> None:
        output_file = temp_dir / "test_output.parquet"
        animation_file = temp_dir / "test_animation.mp4"

        success, output = run_cli_sweep_direct(
            geometry_file, sweep_file, output_file, animation_file
        )

        assert success, f"CLI sweep failed: {output}"
        assert "wrote" in output.lower(), f"Unexpected output: {output}"
        validate_output_against_reference(output_file, "parquet")
        validate_animation_file(animation_file)

    @requires_viz
    def test_gif_animation_output(
        self,
        temp_dir: Path,
        geometry_file: Path,
        sweep_file: Path,
    ) -> None:
        output_file = temp_dir / "test_output.csv"
        animation_file = temp_dir / "test_animation.gif"

        success, output = run_cli_sweep_direct(
            geometry_file, sweep_file, output_file, animation_file
        )

        assert success, f"CLI sweep failed: {output}"
        assert "wrote" in output.lower(), f"Unexpected output: {output}"
        validate_output_against_reference(output_file, "csv")
        validate_animation_file(animation_file)
