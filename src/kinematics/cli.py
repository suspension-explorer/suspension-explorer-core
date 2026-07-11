from pathlib import Path

import typer

from kinematics.io import load_geometry
from kinematics.io.results_writer import SolutionFrame, create_writer_for_path
from kinematics.io.sweep_loader import parse_sweep_file
from kinematics.main import compute_sweep_metrics, solve_sweep

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
    sweep_config = parse_sweep_file(sweep)

    solution_states, solver_stats = solve_sweep(suspension, sweep_config)
    metric_result = compute_sweep_metrics(suspension, sweep_config, solution_states)
    if metric_result.derivative_error is not None:
        typer.echo(
            f"Warning: derivative metrics unavailable: "
            f"{metric_result.derivative_error}",
            err=True,
        )

    # Write out in wide format.
    writer = create_writer_for_path(
        out, geometry_path=str(geometry), sweep_path=str(sweep)
    )
    output_points = suspension.OUTPUT_POINTS
    for idx, (st, solver_info) in enumerate(zip(solution_states, solver_stats)):
        # Filter to the suspension type's declared output points, in order.
        positions = {
            pid.name: (float(pos[0]), float(pos[1]), float(pos[2]))
            for pid in output_points
            if (pos := st.positions.get(pid)) is not None
        }

        # Compute post-solve metrics for this state.
        metrics = metric_result.rows[idx]

        frame = SolutionFrame(
            positions=positions,
            solver_info=solver_info,
            metrics=metrics,
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
