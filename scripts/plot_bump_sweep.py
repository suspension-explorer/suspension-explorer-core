"""
Run an unsteered bump sweep and produce validation plots.

Usage:
    uv run python scripts/plot_bump_sweep.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kinematics import analyze_sweep, load_geometry, load_sweep, solve_sweep
from kinematics.core.enums import PointID
from kinematics.visualization.api import visualize_suspension_sweep

GEOMETRY = Path("tests/data/geometry.yaml")
SWEEP = Path("scripts/bump_sweep.yaml")
OUTPUT_DIR = Path("scripts/plots")


def main() -> None:
    """
    Run an unsteered bump sweep and produce validation plots.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suspension = load_geometry(GEOMETRY)
    sweep_config = load_sweep(SWEEP, suspension)
    analysis = analyze_sweep(suspension, sweep_config)
    states, _ = solve_sweep(suspension, sweep_config)
    config = suspension.config
    assert config is not None

    # Extract data from each solved state.
    wheel_center_z: list[float] = []
    camber: list[float | None] = []
    caster: list[float | None] = []
    roadwheel_angle: list[float | None] = []
    svic_x: list[float | None] = []
    svic_z: list[float | None] = []
    fvic_y: list[float | None] = []
    fvic_z: list[float | None] = []
    svsa: list[float | None] = []
    fvsa: list[float | None] = []
    kpi: list[float | None] = []
    mechanical_trail: list[float | None] = []
    scrub_radius: list[float | None] = []

    # Design position wheel center Z for computing bump travel.
    design_wc_z = float(suspension.initial_state().get(PointID.WHEEL_CENTER)[2])

    for frame in analysis.frames:
        if not frame.solver.converged:
            continue

        wc_z = frame.positions[PointID.WHEEL_CENTER.name.lower()][2]
        wheel_center_z.append(wc_z - design_wc_z)

        metrics = frame.metrics
        camber.append(metrics["camber"])
        caster.append(metrics["caster"])
        roadwheel_angle.append(metrics["roadwheel_angle"])
        svic_x.append(metrics["svic_x"])
        svic_z.append(metrics["svic_z"])
        fvic_y.append(metrics["fvic_y"])
        fvic_z.append(metrics["fvic_z"])
        svsa.append(metrics["svsa_length"])
        fvsa.append(metrics["fvsa_length"])
        kpi.append(metrics["kpi"])
        mechanical_trail.append(metrics["mechanical_trail"])
        scrub_radius.append(metrics["scrub_radius"])

    bump = np.array(wheel_center_z)

    def to_masked(
        data: list[float | None], threshold: float = 50_000
    ) -> np.ma.MaskedArray:
        """
        Convert a list with Nones to a masked array.

        Mask out None values and any value whose absolute magnitude
        exceeds threshold (IC singularity suppression).
        """
        arr = np.array([v if v is not None else np.nan for v in data])
        return np.ma.masked_where(np.isnan(arr) | (np.abs(arr) > threshold), arr)

    camber_m = to_masked(camber)
    caster_m = to_masked(caster)
    rwa_m = to_masked(roadwheel_angle)
    kpi_m = to_masked(kpi)
    svic_x_m = to_masked(svic_x)
    svic_z_m = to_masked(svic_z)
    fvic_y_m = to_masked(fvic_y)
    fvic_z_m = to_masked(fvic_z)
    svsa_m = to_masked(svsa)
    fvsa_m = to_masked(fvsa)
    trail_m = to_masked(mechanical_trail)
    scrub_m = to_masked(scrub_radius)

    x_label = r"$\Delta$ Wheel Center Z [mm]"

    # -- Figure 2: Instant center positions vs bump travel --
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    fig2.suptitle("Instant Center Positions vs. Bump Travel", fontsize=14)

    axes2[0, 0].plot(bump, svic_x_m, "b-o", markersize=3)
    axes2[0, 0].set_ylabel("SVIC X [mm]")
    axes2[0, 0].set_title("Side-View IC: X")
    axes2[0, 0].grid(True, alpha=0.3)

    axes2[0, 1].plot(bump, svic_z_m, "b-o", markersize=3)
    axes2[0, 1].set_ylabel("SVIC Z [mm]")
    axes2[0, 1].set_title("Side-View IC: Z")
    axes2[0, 1].grid(True, alpha=0.3)

    axes2[1, 0].plot(bump, fvic_y_m, "r-o", markersize=3)
    axes2[1, 0].set_ylabel("FVIC Y [mm]")
    axes2[1, 0].set_xlabel(x_label)
    axes2[1, 0].set_title("Front-View IC: Y")
    axes2[1, 0].grid(True, alpha=0.3)

    axes2[1, 1].plot(bump, fvic_z_m, "r-o", markersize=3)
    axes2[1, 1].set_ylabel("FVIC Z [mm]")
    axes2[1, 1].set_xlabel(x_label)
    axes2[1, 1].set_title("Front-View IC: Z")
    axes2[1, 1].grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "instant_centers.png", dpi=150)
    print(f"Saved {OUTPUT_DIR / 'instant_centers.png'}")

    # Figure 3: Swing arm lengths vs bump travel
    fig3, axes3 = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig3.suptitle("Virtual Swing Arm Lengths vs. Bump Travel", fontsize=14)

    axes3[0].plot(bump, svsa_m, "b-o", markersize=3)
    axes3[0].set_ylabel("SVSA Length [mm]")
    axes3[0].grid(True, alpha=0.3)

    axes3[1].plot(bump, fvsa_m, "r-o", markersize=3)
    axes3[1].set_ylabel("FVSA Length [mm]")
    axes3[1].set_xlabel(x_label)
    axes3[1].grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / "swing_arm_lengths.png", dpi=150)
    print(f"Saved {OUTPUT_DIR / 'swing_arm_lengths.png'}")

    # Figure 4: Summary dashboard
    fig4, axes4 = plt.subplots(2, 3, figsize=(15, 8))
    fig4.suptitle("Pure Bump Sweep Metrics", fontsize=14)

    axes4[0, 0].plot(bump, camber_m, "b-", linewidth=1.5)
    axes4[0, 0].set_ylabel("Camber [deg]")
    axes4[0, 0].set_title("Camber Angle")
    axes4[0, 0].set_xlabel(x_label)
    axes4[0, 0].set_ylim(-2.5, -1.5)
    axes4[0, 0].grid(True, alpha=0.3)
    axes4[0, 0].axhline(0, color="k", linewidth=0.5)

    axes4[0, 1].plot(bump, caster_m, "r-", linewidth=1.5)
    axes4[0, 1].set_ylabel("Caster [deg]")
    axes4[0, 1].set_title("Caster Angle")
    axes4[0, 1].set_xlabel(x_label)
    axes4[0, 1].grid(True, alpha=0.3)

    axes4[0, 2].plot(bump, rwa_m, "g-", linewidth=1.5)
    axes4[0, 2].set_ylabel("RWA [deg]")
    axes4[0, 2].set_title("Roadwheel Angle")
    axes4[0, 2].set_xlabel(x_label)
    axes4[0, 2].grid(True, alpha=0.3)
    axes4[0, 2].axhline(0, color="k", linewidth=0.5)

    axes4[1, 0].plot(bump, kpi_m, "m-", linewidth=1.5)
    axes4[1, 0].set_ylabel("KPI [deg]")
    axes4[1, 0].set_title("Kingpin Inclination")
    axes4[1, 0].set_xlabel(x_label)
    axes4[1, 0].grid(True, alpha=0.3)

    axes4[1, 1].plot(bump, trail_m, "b-", linewidth=1.5)
    axes4[1, 1].set_ylabel("Trail [mm]")
    axes4[1, 1].set_title("Mechanical Trail")
    axes4[1, 1].set_xlabel(x_label)
    axes4[1, 1].grid(True, alpha=0.3)

    axes4[1, 2].plot(bump, scrub_m, "r-", linewidth=1.5)
    axes4[1, 2].set_ylabel("Scrub Radius [mm]")
    axes4[1, 2].set_title("Scrub Radius")
    axes4[1, 2].set_xlabel(x_label)
    axes4[1, 2].grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4.savefig(OUTPUT_DIR / "dashboard.png", dpi=150)
    print(f"Saved {OUTPUT_DIR / 'dashboard.png'}")

    plt.close("all")

    # Animation
    wheel_cfg = config.wheel
    animation_path = OUTPUT_DIR / "bump_sweep.mp4"
    visualize_suspension_sweep(
        suspension=suspension,
        solution_states=states,
        output_path=animation_path,
        wheel_diameter=wheel_cfg.tire.nominal_radius * 2,
        wheel_width=wheel_cfg.tire.section_width,
        fps=20,
        show_live=False,
    )
    print(f"Saved {animation_path}")


if __name__ == "__main__":
    main()
