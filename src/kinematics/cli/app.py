from pathlib import Path

import typer

from kinematics.cli.commands.sweep import run_sweep_files
from kinematics.cli.io.yaml import load_geometry

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
    run = run_sweep_files(geometry, sweep, out)
    if run.analysis.diagnostics:
        typer.echo("Diagnostics:", err=True)
        for issue in run.analysis.diagnostics:
            typer.echo(f"{issue.severity.upper()}: {issue.message}", err=True)

    typer.echo(f"wrote {out}")

    # Generate animation if requested.
    if animation_out:
        try:
            from kinematics.cli.visualization.api import visualize_suspension_sweep

            # Get wheel parameters from suspension configuration.
            if run.suspension.config is None:
                typer.echo("Error: No config in suspension", err=True)
                raise typer.Exit(1)

            wheel_cfg = run.suspension.config.wheel

            # Create animation.
            visualize_suspension_sweep(
                suspension=run.suspension,
                solution_states=run.states,
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
                f'Install with: pip install "kinematics[cli,viz]"\n'
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
        from kinematics.cli.visualization.api import visualize_geometry
    except ImportError as e:
        typer.echo(
            f"Error: Visualization dependencies not installed.\n"
            f'Install with: pip install "kinematics[cli,viz]"\n'
            f"Details: {e}",
            err=True,
        )
        raise typer.Exit(1)

    suspension = load_geometry(geometry)

    typer.echo("Checking and visualizing suspension geometry...")
    result = visualize_geometry(
        suspension=suspension,
        output_path=output,
    )
    if result.contact_patch_on_ground:
        typer.secho(
            "Geometry Check: OK. Contact patch at ground "
            f"(Z = {result.contact_patch_z:.3f} mm).",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(
            "Geometry Check: WARNING. Contact patch center is not on the ground.",
            fg=typer.colors.RED,
        )
        typer.echo(
            "The contact patch center is currently located at "
            f"Z = {result.contact_patch_z:.3f} mm."
        )
    typer.secho(
        f"Visualization saved to: {result.output_path}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
