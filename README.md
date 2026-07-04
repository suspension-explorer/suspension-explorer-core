# open-kinematics

> ⚠️   
>
> **Note that this system is both experimental and still under development. I do not recommend using it for anything important.**

`open-kinematics` is a Python-based geometric constraint solver for simulating the kinematic behaviour of vehicle suspension systems. It allows users to analyse suspension geometries by running parametric sweeps, then offering exports of solved system positions alongside visualisations of suspension state.

The tool is built around a numerical solver that determines the unique positions of all suspension components for a given set of boundary conditions (e.g., a specific wheel height or steering rack position).

<p align="center">
  <img src="/images/plot.png" alt="Design Condition Visualisation" width="80%">
  <br>
  <em>Visualisation of a double wishbone suspension at its design condition.</em>
</p>

## Features

- Geometric Constraint Solver: Uses a numerical approach (Levenberg-Marquardt) with analytical Jacobians to solve for the kinematic state of the system based on geometric constraints.
- Parametric Sweeps: Simulate suspension motion by sweeping through a range of inputs, such as vertical wheel travel and steering rack displacement.
- Template-Based Suspension Models: Define suspension geometries using templates (currently double wishbone, as a single corner or a full two-corner axle) with simple YAML configuration files.
- Full-Axle Simulation: Solve left and right corners together as one coupled system (`double_wishbone_axle`), linked by a rigid steering rack. Geometry can be given for one side and mirrored, or for both sides explicitly.
- Inboard Actuation: Optional F1-style pushrod/rocker packages per corner (rocker axes parallel to the XZ plane), with torsion bars on each rocker pivot and an inboard anti-roll bar actuated via droplinks.
- Camber Shim Simulation: Model outboard camber shim configurations to simulate shimmed ball joint offsets.
- Derived Points System: A dependency-aware system for calculating the position of non-kinematic points (like wheel centers) based on the solved positions of core hard points.
- Suspension Metrics: Computes camber, caster, toe, kingpin inclination (KPI), scrub radius, mechanical trail, and side-view/front-view instant centres from the solved geometry. Axle simulations add per-side metric columns plus roll centre position, total toe, track, rack displacement, rocker/torsion-bar angles, and anti-roll-bar twist.
- Data Export: Save simulation results in wide-format CSV or Apache Parquet files for further analysis.
- Visualization: Generate static plots of the design condition and create MP4/GIF animations of sweep motions.

## How it works

The core of the tool is a numerical solver that treats the suspension as a collection of rigid bodies connected by ideal spherical joints. The geometric relationships, such as the fixed length of a wishbone or a track rod, are defined as a system of nonlinear equations.

For each step in a simulation sweep, the solver's objective is to find the 3D coordinates for all free-moving points that will drive the residuals of these constraint equations to zero. Though really a root-finding problem, it is approached as nonlinear least squares problem using SciPy's `least_squares` implementation of the Levenberg-Marquardt algorithm.

This numerical approach is highly flexible, allowing the system to be "driven" by various targets (e.g., wheel center height, rack position), hard or derived, without needing to derive new analytical equations for each case.

## Installation

Use of a virtual environment is recommended. [uv](https://github.com/astral-sh/uv) is used in the examples below.

### Basic Installation

For core kinematics functionality without visualisation dependencies:

```bash
uv pip install kinematics
```

### Full Installation (with Visualization)

To generate plots and animations, you need to install the `[viz]` extra, which includes `matplotlib`.

```bash
uv pip install "kinematics[viz]"
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
  - point: TRACKROD_INBOARD # Drive steering rack position.
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

This will produce a video like the one below, showing the suspension articulating through a range of bump, droop, and rack travel.

<p align="center">
  <img src="/images/animation.gif" alt="Kinematic Sweep Animation" width="80%">
  <br>
  <em>Animation of a full kinematic sweep.</em>
</p>

**Note:** If you try to use visualisation features (`--animation-out` or the `visualize` command) without installing the `[viz]` extra, you will receive an error indicating that the required dependencies are not installed.
