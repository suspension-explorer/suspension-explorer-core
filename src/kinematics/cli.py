from pathlib import Path

import typer

from kinematics.diagnostics import diagnose_sweep
from kinematics.io import (
    SolutionFrame,
    create_writer_for_path,
    load_geometry,
    load_sweep,
)
from kinematics.main import compute_sweep_metrics, solve_sweep
from kinematics.metrics.main import AxleMetricRows

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def sweep(
    geometry: Path = typer.Option(..., exists=True, help="Path to geometry YAML"),
    sweep: Path = typer.Option(..., exists=True, help="Path to sweep YAML"),
    out: Path = typer.Option(..., help="Output path (.parquet or .csv)"),
    animation_out: Path | None = typer.Option(
        None, help="Optional animation output path (.mp4, .gif, etc.)"
    ),
):
    """
    Run a sweep from file and write results to Parquet or CSV format.

    Example:
        kinematics sweep --geometry=geo.yaml --sweep=sweep.yaml --out=out.parquet
        kinematics sweep --geometry=geo.yaml --sweep=sweep.yaml --out=out.csv
    """
    suspension = load_geometry(geometry)
    sweep_config = load_sweep(sweep, suspension)

    solution_states, solver_stats = solve_sweep(suspension, sweep_config)

    # Post-sweep diagnostics: the solver already raised for hard infeasibility,
    # so anything here is advisory (branch snaps, chirality inversion,
    # transmission margin) or a belt-and-braces re-check. Data is still written.
    diagnostics = diagnose_sweep(suspension, solution_states, solver_stats)
    for issue in diagnostics.issues:
        prefix = "ERROR" if issue.severity == "error" else "WARNING"
        typer.echo(f"{prefix}: {issue.message}", err=True)
    if diagnostics.issues:
        typer.echo(
            f"Diagnostics: {len(diagnostics.errors)} error(s), "
            f"{len(diagnostics.warnings)} warning(s).",
            err=True,
        )

    # Write out in wide format.
    writer = create_writer_for_path(
        out, geometry_path=str(geometry), sweep_path=str(sweep)
    )
    output_points = suspension.output_points()

    # Full metric rows per state (per-state metrics plus derivative metrics
    # such as motion ratios and camber gain), computed by the package's single
    # high-level metrics entry point.
    metric_rows = compute_sweep_metrics(suspension, sweep_config, solution_states)

    for idx, (st, solver_info) in enumerate(zip(solution_states, solver_stats)):
        # Filter to the suspension type's declared output points, in order.
        positions = {
            pid.name: (float(pos[0]), float(pos[1]), float(pos[2]))
            for pid in output_points
            if (pos := st.positions.get(pid)) is not None
        }

        # Structured axle rows render to canonical flat column names (corner
        # columns location-suffixed) at this export boundary only.
        rows = metric_rows[idx]
        flat_metrics = rows.flat_row() if isinstance(rows, AxleMetricRows) else rows

        frame = SolutionFrame(
            positions=positions,
            solver_info=solver_info,
            metrics=dict(flat_metrics),
        )

        writer.add_frame(idx, frame)
    writer.write()

    typer.echo(f"wrote {out}")

    # Generate animation if requested.
    if animation_out:
        try:
            from kinematics.visualization.api import visualize_suspension_sweep

            # Get wheel parameters from suspension configuration.
            if suspension.config is None:
                typer.echo("Error: No config in suspension", err=True)
                raise typer.Exit(1)

            wheel_cfg = suspension.config.wheel

            # Create animation.
            visualize_suspension_sweep(
                suspension=suspension,
                solution_states=solution_states,
                output_path=animation_out,
                wheel_diameter=wheel_cfg.tire.nominal_radius * 2,
                wheel_width=wheel_cfg.tire.section_width,
                fps=20,
                show_live=False,
            )

            typer.echo(f"Wrote animation: {animation_out}")

        except ImportError as e:
            typer.echo(
                f"Error: Visualization dependencies not installed.\n"
                f'Install with: pip install "kinematics[viz]"\n'
                f"Details: {e}",
                err=True,
            )
            typer.Exit(1)


@app.command()
def visualize(
    geometry: Path = typer.Option(..., exists=True, help="Path to geometry YAML."),
    output: Path = typer.Option(
        ..., help="Output path for the plot image (.png, .jpg)."
    ),
):
    """
    Visualize a suspension geometry at its design condition.

    This command loads a single geometry file, calculates its initial state, and
    generates a debug plot. It also reports whether the contact patch approximation
    (minimum Z position on wheel center plane) is tangent to the ground plane (Z=0).

    Example:
    uv run kinematics visualize --geometry=tests/data/geometry.yaml --output=plot.png
    """
    try:
        from kinematics.visualization.api import visualize_geometry
    except ImportError as e:
        typer.echo(
            f"Error: Visualization dependencies not installed.\n"
            f'Install with: pip install "kinematics[viz]"\n'
            f"Details: {e}",
            err=True,
        )
        raise typer.Exit(1)

    suspension = load_geometry(geometry)

    visualize_geometry(
        suspension=suspension,
        output_path=output,
    )


if __name__ == "__main__":
    app()
