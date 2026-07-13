"""
Results output and serialization.

This module provides functionality for writing kinematic analysis results to file,
handling the serialization of simulation data, solver information, and configuration
parameters.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kinematics.core.metrics.registry import MetricSpec
from kinematics.core.solver import SolverInfo


class MetadataKey(Enum):
    """
    Standard metadata keys for result files.
    """

    FORMAT_VERSION = "format_version"
    TIMESTAMP = "timestamp"
    GEOMETRY_PATH = "geometry_path"
    SWEEP_PATH = "sweep_path"
    GEOMETRY_HASH = "geometry_hash"
    SWEEP_HASH = "sweep_hash"
    COLUMN_UNITS = "column_units"


class StandardColumn(Enum):
    """
    Standard column names in result files.
    """

    STEP_INDEX = "step_index"
    SOLVER_CONVERGED = "solver_converged"
    SOLVER_NFEV = "solver_nfev"
    SOLVER_MAX_RESIDUAL = "solver_max_residual"


class SupportedFormat(Enum):
    """
    Supported output file formats.
    """

    PARQUET = ".parquet"
    CSV = ".csv"


FORMAT_VERSION = "3"
PACKAGE_NAME = "kinematics"
METADATA_KEY = b"kinematics_meta"  # Need bytes for pyarrow.


def compute_file_hash(path: str | Path) -> str:
    """
    Compute SHA-256 hash of a file for provenance tracking.

    Args:
        path: Path to the file to hash.

    Returns:
        Hexadecimal hash string, or empty string if file cannot be read.
    """
    try:
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except Exception:
        return ""


@dataclass
class SolutionFrame:
    """
    A single frame of solution data.

    Attributes:
        positions: Dictionary mapping point IDs to (x, y, z) coordinates.
        solver_info: Solver diagnostic information.
        metrics: Ordered metric column names to values. Values may be None
            for undefined geometry (e.g. parallel links).
    """

    positions: dict[str, tuple[float, float, float]]
    solver_info: SolverInfo
    metrics: dict[str, float | None] = field(default_factory=dict)
    metric_specs: dict[str, MetricSpec] = field(default_factory=dict)


class BaseResultsWriter(ABC):
    """
    Abstract base class for kinematic solution results writers.

    This writer collects solution frames and provides a common interface for different
    output formats. Subclasses implement the specific write logic.
    """

    def __init__(
        self,
        output_path: str | Path,
        geometry_path: str | Path | None = None,
        sweep_path: str | Path | None = None,
        **extra_metadata: str,
    ):
        """
        Initialize the results writer.

        Args:
            output_path: Path where the file will be written.
            geometry_path: Optional path to geometry file for provenance.
            sweep_path: Optional path to sweep file for provenance.
            **extra_metadata: Additional metadata key-value pairs to store.
        """
        self.output_path = Path(output_path)
        self.frames: list[dict[str, Any]] = []
        self.column_units: dict[str, str] = {}

        # Build standard metadata.
        self.metadata: dict[str, str] = {
            MetadataKey.FORMAT_VERSION.value: FORMAT_VERSION,
            MetadataKey.TIMESTAMP.value: str(time.time()),
            **extra_metadata,
        }

        # Add file paths and hashes for provenance.
        if geometry_path is not None:
            geo_path = str(geometry_path)
            self.metadata[MetadataKey.GEOMETRY_PATH.value] = geo_path
            self.metadata[MetadataKey.GEOMETRY_HASH.value] = compute_file_hash(geo_path)

        if sweep_path is not None:
            swp_path = str(sweep_path)
            self.metadata[MetadataKey.SWEEP_PATH.value] = swp_path
            self.metadata[MetadataKey.SWEEP_HASH.value] = compute_file_hash(swp_path)

    def add_frame(self, frame_index: int, frame: SolutionFrame) -> None:
        """
        Add a solution frame to the accumulated results.

        Args:
            frame_index: Sequential index for this frame.
            frame: Solution data for this frame.
        """
        # Start with standard columns.
        row: dict[str, Any] = {
            StandardColumn.STEP_INDEX.value: int(frame_index),
        }
        row[StandardColumn.SOLVER_CONVERGED.value] = frame.solver_info.converged
        row[StandardColumn.SOLVER_MAX_RESIDUAL.value] = frame.solver_info.max_residual
        row[StandardColumn.SOLVER_NFEV.value] = frame.solver_info.nfev

        # Metric columns (between solver and position columns).
        for metric_name, metric_value in frame.metrics.items():
            row[metric_name] = metric_value
            spec = frame.metric_specs.get(metric_name)
            if spec is not None:
                self._record_column_unit(metric_name, spec.unit.symbol)

        # Flatten position data into separate x/y/z columns.
        for point_id, (x, y, z) in frame.positions.items():
            row[f"{point_id}_x"] = float(x)
            row[f"{point_id}_y"] = float(y)
            row[f"{point_id}_z"] = float(z)
            for axis in ("x", "y", "z"):
                self._record_column_unit(f"{point_id}_{axis}", "mm")

        self.frames.append(row)

    def _record_column_unit(self, column: str, unit: str) -> None:
        """Record one consistent physical unit for an output column."""
        existing = self.column_units.get(column)
        if existing is not None and existing != unit:
            raise ValueError(
                f"Conflicting units for column '{column}': {existing} and {unit}"
            )
        self.column_units[column] = unit

    def build_column_list(self) -> list[str]:
        """
        Validate that all frames have consistent columns and return the column list.

        Returns:
            List of column names from the first frame, preserving order.

        Raises:
            ValueError: If frames have inconsistent columns or no frames exist.
        """
        if not self.frames:
            raise ValueError("No frames to validate")

        all_columns = list(self.frames[0].keys())
        first_frame_columns = set(all_columns)

        for i, frame in enumerate(self.frames[1:], 1):
            frame_columns = set(frame.keys())
            if frame_columns != first_frame_columns:
                missing = first_frame_columns - frame_columns
                extra = frame_columns - first_frame_columns
                error_parts = []
                if missing:
                    error_parts.append(f"Missing columns: {sorted(missing)}")
                if extra:
                    error_parts.append(f"Extra columns: {sorted(extra)}")
                raise ValueError(
                    f"Frame {i} has inconsistent columns - {', '.join(error_parts)}"
                )

        return all_columns

    @abstractmethod
    def write(self) -> None:
        """
        Write all accumulated frames to the output file.

        Must be implemented by subclasses.
        """
        pass


class ParquetWriter(BaseResultsWriter):
    """
    Accumulates and writes kinematic solution results to Parquet format.

    This writer collects solution frames and writes them to a Parquet file with
    comprehensive metadata including provenance information and solver diagnostics.

    Example:
        writer = ParquetWriter("output.parquet", geometry_path="geo.yaml")

        for frame_idx, solution in enumerate(solutions):
            frame = SolutionFrame(
                positions={"wheel_center": (0.0, 100.0, 50.0), ...},
                solver_info=SolverInfo(converged=True, nfev=12, max_residual=0.001)
            )
            writer.add_frame(frame_idx, frame)

        writer.write()
    """

    def write(self) -> None:
        """
        Write all accumulated frames to the Parquet file.

        Frames are sorted by frame_index before writing to ensure consistent
        ordering. The schema is inferred from the data with appropriate type
        conversions.

        Raises:
            ValueError: If no frames have been added or data is malformed.
            OSError: If the file cannot be written.
        """
        if not self.frames:
            raise ValueError("No frames to write")

        # Sort by frame index.
        self.frames.sort(key=lambda r: r[StandardColumn.STEP_INDEX.value])

        # Validate consistent columns and get column list.
        all_columns = self.build_column_list()

        # Transpose: rows to columns.
        column_data: dict[str, list[Any]] = {col: [] for col in all_columns}
        for frame in self.frames:
            for col in all_columns:
                column_data[col].append(frame.get(col))

        # Validate data types and shapes before type inference.
        for col in all_columns:
            values = column_data[col]

            for frame_idx, val in enumerate(values):
                if val is None:
                    continue  # None is always acceptable.

                # Check for nested structures (lists, tuples, arrays).
                if isinstance(val, (list, tuple, np.ndarray)):
                    raise ValueError(
                        f"Column '{col}' at frame {frame_idx} contains nested "
                        f"data: {val!r}. Expected scalar value (bool, int, float, "
                        f"or str). This usually indicates corrupted position data "
                        f"- check that all point coordinates are being flattened "
                        f"correctly (e.g., positions should be stored as separate "
                        f"_x, _y, _z columns)."
                    )

                # Check for unexpected types.
                if not isinstance(val, (bool, int, float, str)):
                    raise ValueError(
                        f"Column '{col}' at frame {frame_idx} contains unexpected "
                        f"type {type(val).__name__}: {val!r}. Expected bool, int, "
                        f"float, str, or None. This indicates data corruption in "
                        f"the solver or state management."
                    )

        # Now proceed with type inference (existing code continues unchanged)
        # Infer appropriate Arrow types for each column.
        arrays: list[pa.Array] = []
        names: list[str] = []

        for col in all_columns:
            values = column_data[col]

            # Type inference heuristics.
            if all(isinstance(v, bool) or v is None for v in values):
                arr = pa.array(values, type=pa.bool_())
            elif all(isinstance(v, int) or v is None for v in values):
                # Use float64 for coordinate columns to avoid precision loss.
                if col.endswith(("_x", "_y", "_z")):
                    arr = pa.array(
                        [None if v is None else float(v) for v in values],
                        type=pa.float64(),
                    )
                else:
                    arr = pa.array(values, type=pa.int64())
            elif all(isinstance(v, (int, float)) or v is None for v in values):
                arr = pa.array(
                    [None if v is None else float(v) for v in values], type=pa.float64()
                )
            else:
                arr = pa.array(
                    [None if v is None else str(v) for v in values], type=pa.string()
                )

            arrays.append(arr)
            names.append(col)

        # Build table and attach metadata.
        fields = [
            pa.field(
                name,
                array.type,
                metadata=(
                    {b"unit": self.column_units[name].encode("utf-8")}
                    if name in self.column_units
                    else None
                ),
            )
            for name, array in zip(names, arrays)
        ]
        table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))

        existing_meta = table.schema.metadata or {}
        metadata_bytes = {
            **existing_meta,
            METADATA_KEY: json.dumps(self.metadata).encode("utf-8"),
        }

        table = table.replace_schema_metadata(metadata_bytes)

        # Ensure output directory exists.
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to disk.
        pq.write_table(table, self.output_path)


class CsvWriter(BaseResultsWriter):
    """
    Accumulates and writes kinematic solution results to CSV format.

    This writer collects solution frames and writes them to a CSV file with
    metadata as comments at the top of the file.

    Example:
        writer = CsvWriter("output.csv", geometry_path="geo.yaml")

        for frame_idx, solution in enumerate(solutions):
            frame = SolutionFrame(
                positions={"wheel_center": (0.0, 100.0, 50.0), ...},
                solver_info=SolverInfo(converged=True, nfev=12, max_residual=0.001)
            )
            writer.add_frame(frame_idx, frame)

        writer.write()
    """

    def write(self) -> None:
        """
        Write all accumulated frames to the CSV file.

        Frames are sorted by frame_index before writing to ensure consistent
        ordering. Metadata is written as comments at the top of the file.

        Raises:
            ValueError: If no frames have been added or data is malformed.
            OSError: If the file cannot be written.
        """
        if not self.frames:
            raise ValueError("No frames to write")

        # Sort by frame index.
        self.frames.sort(key=lambda r: r[StandardColumn.STEP_INDEX.value])

        # Validate consistent columns and get column list.
        all_columns = self.build_column_list()

        # Validate data types and shapes.
        for frame_idx, frame in enumerate(self.frames):
            for col in all_columns:
                val = frame.get(col)

                if val is None:
                    continue  # None is acceptable.

                # Check for nested structures.
                if isinstance(val, (list, tuple, np.ndarray)):
                    raise ValueError(
                        f"Frame {frame_idx}, column '{col}' contains nested data: "
                        f"{val!r}. Expected scalar value. Check position data "
                        f"flattening."
                    )

                # Check for unexpected types.
                if not isinstance(val, (bool, int, float, str)):
                    raise ValueError(
                        f"Frame {frame_idx}, column '{col}' contains unexpected "
                        f"type {type(val).__name__}: {val!r}. Expected bool, int, "
                        f"float, str, or None."
                    )

        # Ensure output directory exists.
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to CSV.
        with open(self.output_path, "w", newline="") as csvfile:
            # Write metadata to top.
            for key, value in self.metadata.items():
                csvfile.write(f"# {key}: {value}\n")
            csvfile.write(
                f"# {MetadataKey.COLUMN_UNITS.value}: "
                f"{json.dumps(self.column_units, sort_keys=True)}\n"
            )
            csvfile.write("#\n")

            # Write data using DictWriter.
            writer = csv.DictWriter(
                csvfile,
                fieldnames=all_columns,
                lineterminator="\n",
            )
            writer.writeheader()

            for frame in self.frames:
                # Fill missing columns with None for consistent output.
                row = {col: frame.get(col) for col in all_columns}
                writer.writerow(row)


def create_writer_for_path(
    output_path: Path,
    geometry_path: str | Path | None = None,
    sweep_path: str | Path | None = None,
    **extra_metadata: str,
) -> BaseResultsWriter:
    """
    Create an appropriate results writer based on the output file extension.

    Args:
        output_path: Path where the file will be written.
        geometry_path: Optional path to geometry file for provenance.
        sweep_path: Optional path to sweep file for provenance.
        **extra_metadata: Additional metadata key-value pairs to store.

    Returns:
        A results writer appropriate for the file extension.

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix = output_path.suffix.lower()

    if suffix == SupportedFormat.PARQUET.value:
        return ParquetWriter(output_path, geometry_path, sweep_path, **extra_metadata)
    elif suffix == SupportedFormat.CSV.value:
        return CsvWriter(output_path, geometry_path, sweep_path, **extra_metadata)
    else:
        supported_formats = ", ".join(f.value for f in SupportedFormat)
        raise ValueError(
            f"Unsupported file extension: {suffix}. "
            f"Supported formats: {supported_formats}"
        )
