from pathlib import Path

import typer

from kinematics import analyze_sweep, load_geometry, load_sweep
from kinematics.core.point_ref import point_key_name
from kinematics.io.results_writer import SolutionFrame, create_writer_for_path
from kinematics.metrics.main import flatten_metric_rows
from kinematics.metrics.registry import flat_specs_for_suspension

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
    analysis = analyze_sweep(suspension, sweep_config)
    if analysis.diagnostics:
        typer.echo("Diagnostics:", err=True)
        for issue in analysis.diagnostics:
            typer.echo(f"{issue.severity.upper()}: {issue.message}", err=True)

    # Write out in wide format.
    writer = create_writer_for_path(
        out, geometry_path=str(geometry), sweep_path=str(sweep)
    )
    output_points = suspension.output_points()
    metric_specs = flat_specs_for_suspension(suspension)
    for frame_data in analysis.frames:
        # Filter to the suspension type's declared output points, in order.
        positions = {
            point_key_name(pid): analysis_position
            for pid in output_points
            if (analysis_position := frame_data.positions.get(point_key_name(pid)))
            is not None
        }

        metrics = flatten_metric_rows(
            frame_data.metrics,
            frame_data.corner_metrics,
        )

        frame = SolutionFrame(
            positions=positions,
            solver_info=frame_data.solver,
            metrics=metrics,
            metric_specs=metric_specs,
        )

        writer.add_frame(frame_data.index, frame)
    writer.write()

    typer.echo(f"wrote {out}")

    # Generate animation if requested.
    if animation_out:
        try:
            from kinematics.main import solve_sweep
            from kinematics.visualization.api import visualize_suspension_sweep

            # Get wheel parameters from suspension configuration.
            if suspension.config is None:
                typer.echo("Error: No config in suspension", err=True)
                raise typer.Exit(1)

            wheel_cfg = suspension.config.wheel
            solution_states, _ = solve_sweep(suspension, sweep_config)

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
            raise typer.Exit(1)


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
