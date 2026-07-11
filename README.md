# open-kinematics

> ⚠️   
>
> **Note that this system is both experimental and still under development. I do not recommend using it for anything important.**

`open-kinematics` is a Python-based geometric constraint solver for simulating the kinematic behaviour of vehicle suspension systems. It allows users to analyse suspension geometries by running parametric sweeps, then offering exports of solved system positions alongside visualisations of suspension state.

The tool is built around a numerical solver that determines the positions of all suspension components for a given set of boundary conditions, such as wheel height or trackrod inboard position.

<p align="center">
  <img src="/images/plot.png" alt="Design Condition Visualisation" width="80%">
  <br>
  <em>Visualisation of a double wishbone suspension at its design condition.</em>
</p>

## Features

- Geometric Constraint Solver: Uses a numerical approach (Levenberg-Marquardt) with analytical Jacobians to solve for the kinematic state of the system based on geometric constraints.
- Parametric Sweeps: Simulate suspension motion by sweeping through a range of inputs, such as vertical wheel travel and trackrod inboard displacement.
- Explicit Suspension Models: Define double-wishbone corners and coupled axles with validated YAML. Supported topologies include direct and inboard coilovers, pushrod-rocker actuation, torsion bars, and a shared anti-roll bar.
- Full-Axle Simulation: Solve left and right corners together with a fixed separation between their trackrod inboard points. Geometry can provide one explicitly sided corner to mirror or both sides independently.
- Camber Shim Simulation: Model outboard camber shim configurations to simulate shimmed ball joint offsets.
- Derived Points System: A dependency-aware system for calculating the position of non-kinematic points (like wheel centers) based on the solved positions of core hard points.
- Suspension Metrics: Compute roadwheel angle, camber, caster, kingpin inclination, scrub radius, mechanical trail, instant-centre geometry, wheel travel, half-track, damper length, and anti-pitch geometry. Axle models add track, roll centre, heave, roll, ride-height change, trackrod inboard displacement, and anti-roll-bar metrics.
- Exact Derivatives: Evaluate declarative `d(response) / d(driver)` metrics, including wheel-centre X with respect to wheel-centre Z, camber, roadwheel angle, damper, rocker, torsion-bar, and anti-roll-bar ratios, using analytical constraint Jacobians and forward-mode automatic differentiation.
- Sweep Diagnostics: Report convergence, residual acceptance, branch continuity, derivative availability, mechanism chirality, and transmission-margin issues without discarding otherwise available results.
- Structured Analysis API: Use `analyze_sweep()` and `initial_pose()` to obtain name-keyed positions, structural corner locations, metric metadata, display topology, diagnostics, and solved frames.
- Data Export: Save wide-format CSV or Apache Parquet results with lowercase `snake_case` columns. Units are metadata rather than part of metric names.
- Visualization: Generate static plots of the design condition and create MP4/GIF animations of sweep motions.

## How it works

The core of the tool is a numerical solver that treats the suspension as a collection of rigid bodies connected by ideal spherical joints. The geometric relationships, such as the fixed length of a wishbone or a track rod, are defined as a system of nonlinear equations.

For each step in a simulation sweep, the solver's objective is to find the 3D coordinates for all free-moving points that will drive the residuals of these constraint equations to zero. Though really a root-finding problem, it is approached as a nonlinear least squares problem using SciPy's `least_squares` implementation of the Levenberg-Marquardt algorithm.

This numerical approach is highly flexible, allowing the system to be "driven" by various targets (e.g., wheel center height or trackrod inboard position), hard or derived, without needing to derive new analytical equations for each case.

## Installation

Use of a virtual environment is recommended. [uv](https://github.com/astral-sh/uv) is used in the examples below.

The package is not published to PyPI; install it from this repository.

### Basic Installation

For core kinematics functionality without visualisation dependencies:

```bash
uv pip install "kinematics @ git+https://github.com/nickmccleery/open-kinematics.git"
```

### Full Installation (with Visualization)

To generate plots and animations, you need to install the `[viz]` extra, which includes `matplotlib`.

```bash
uv pip install "kinematics[viz] @ git+https://github.com/nickmccleery/open-kinematics.git"
```

## Usage

The primary way to use `open-kinematics` is through its command-line interface.

### 1. Visualising a geometry at 'design condition'

You can generate a multi-view plot of your suspension geometry to verify the initial 'design condition' defined in your YAML file. This is useful for debugging your geometry definition.

```bash
uv run kinematics visualize --geometry tests/data/geometry.yaml --output plot.png
```

This command will produce an image like the one at the top of this README.

### 2. Running a kinematic sweep

A sweep simulates the suspension's movement through a range of inputs. This requires a `geometry.yaml` file and a `sweep.yaml` file.

A typical sweep file defines the targets, range, and number of steps:

```yaml
# sweep.yaml
version: 1
steps: 41
targets:
  - point: TRACKROD_INBOARD # Drive trackrod inboard position.
    direction:
      axis: Y
    mode: relative
    start: -40
    stop: 40
  - point: WHEEL_CENTER # Drive vertical wheel travel.
    direction:
      axis: Z
    start: -40
    stop: 120
```

To run the sweep and save the results, use the `sweep` command.

**Basic sweep with CSV export:**

```bash
uv run kinematics sweep --geometry tests/data/geometry.yaml --sweep tests/data/sweep.yaml --out results.csv
```

**Full sweep with parquet export and animation:**
This command will generate both a Parquet data file and an MP4 animation of the motion.

```bash
uv run kinematics sweep --geometry tests/data/geometry.yaml --sweep tests/data/sweep.yaml --out results.parquet --animation-out animation.mp4
```

This will produce a video like the one below, showing the suspension articulating through a range of bump, droop, and steering travel.

<p align="center">
  <img src="/images/animation.gif" alt="Kinematic Sweep Animation" width="80%">
  <br>
  <em>Animation of a full kinematic sweep.</em>
</p>

**Note:** If you try to use visualisation features (`--animation-out` or the `visualize` command) without installing the `[viz]` extra, you will receive an error indicating that the required dependencies are not installed.
