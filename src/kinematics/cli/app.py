from importlib import import_module
from pathlib import Path
from types import ModuleType

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


def require_visualization() -> ModuleType:
    """
    Load the optional visualization API or exit with installation guidance.
    """
    try:
        import_module("matplotlib")
        return import_module("kinematics.cli.visualization.api")
    except ImportError as error:
        typer.echo(
            "Error: Visualization dependencies not installed.\n"
            'Install with: pip install "kinematics[cli,viz]"\n'
            f"Details: {error}",
            err=True,
        )
        raise typer.Exit(1) from error


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
    from kinematics.cli.commands.sweep import run_sweep_files

    run = run_sweep_files(geometry, sweep, out)
    if run.evaluated.diagnostics:
        typer.echo("Diagnostics:", err=True)
        for issue in run.evaluated.diagnostics:
            typer.echo(f"{issue.severity.upper()}: {issue.message}", err=True)

    typer.echo(f"wrote {out}")

    # Generate animation if requested.
    if animation_out:
        visualization = require_visualization()

        # Create animation.
        visualization.visualize_suspension_sweep(
            suspension=run.suspension,
            solution_states=run.evaluated.states,
            output_path=animation_out,
            fps=20,
            show_live=False,
            instantaneous_steering_axes=(run.evaluated.instantaneous_steering_axes),
        )

        typer.echo(f"Wrote animation: {animation_out}")


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
    generates a debug plot. It also reports whether the wheel contact centres
    points lie on the reconstructed design road plane.

    Example:
    uv run kinematics visualize --geometry=tests/data/geometry.yaml --output=plot.png
    """
    from kinematics.cli.io.loaders import load_geometry

    visualization = require_visualization()
    suspension = load_geometry(geometry)

    typer.echo("Checking and visualizing suspension geometry...")
    result = visualization.visualize_geometry(
        suspension=suspension,
        output_path=output,
    )
    road_distances = ", ".join(
        f"{value:.3f}" for value in result.wheel_contact_centre_road_distance_mm
    )
    if result.wheel_contact_centres_on_road:
        typer.secho(
            "Geometry Check: OK. Wheel contact centres lie on the reconstructed "
            f"design road plane (distances = {road_distances} mm).",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(
            "Geometry Check: WARNING. Wheel contact centres do not lie on the "
            "reconstructed design road plane.",
            fg=typer.colors.RED,
        )
        typer.echo(f"Their signed distances from that plane are {road_distances} mm.")
    typer.secho(
        f"Visualization saved to: {result.output_path}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
