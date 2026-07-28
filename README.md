<p align="center">
  <img
    src="docs/logo-wordmark-paper.svg#gh-dark-mode-only"
    alt="Suspension Explorer"
    width="420"
  >
  <img
    src="docs/logo-wordmark-ink.svg#gh-light-mode-only"
    alt="Suspension Explorer"
    width="420"
  >
</p>

# `suspension-explorer-core`

> [!WARNING]
> Suspension Explorer is experimental and under active development. If using it for any
> real-world project, please validate its results independently before using them for
> design decisions.

Suspension Explorer is a geometric constraint solver for vehicle suspension
kinematics. This repository contains the open-core Python solver and its CLI
adapter. It can validate suspension geometry, solve coordinated bump, roll, and
steering sweeps, calculate suspension metrics, export results, and render simple
plots or animations.

The solver models ideal rigid parts and joints. It calculates geometry and
motion; it is not a compliance, load, or structural analysis
tool.

<p align="center">
  <img src="images/plot.png" alt="Design condition visualization" width="80%">
  <br>
  <em>A double-wishbone suspension at its design condition.</em>
</p>

## What is supported

| Area                      | Supported                                                                                 | Important limits                                                                                                   |
| ------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Locating architectures    | Double wishbone and MacPherson strut                                                      | Each may be built as one corner or a composed two-corner axle.                                                     |
| Axle geometry             | Mirrored or explicitly authored left and right corners                                    | If `hardpoints.right` is omitted, the complete left geometry and side-local setup are mirrored through `Y = 0`.    |
| Wheel-heading control     | Translating steering rack or fixed toe link                                               | Select `steering.type: rack` or `steering.type: none`; front/rear position does not select steering automatically. |
| Double-wishbone actuation | Direct or pushrod-rocker, mounted to the lower wishbone or upright                        | Direct actuation cannot be combined with a torsion bar.                                                            |
| Double-wishbone springs   | None, coilover, or torsion bar                                                            | A torsion bar requires pushrod-rocker actuation.                                                                   |
| Axle mechanisms           | U-bar or T-bar anti-roll mechanism and rocker-to-rocker heave link                        | These mechanisms require a double-wishbone axle with pushrod-rocker actuation.                                     |
| Setup changes             | Outboard camber shims on double-wishbone corners                                          | Explicit asymmetric axle hardpoints require corresponding side-local setup when a shim is used.                    |
| Outputs                   | Solved point positions, solver statistics, diagnostics, metrics, in either CSV or Parquet | Plotting and animation require the optional visualization dependencies.                                            |

The calculated metrics include wheel travel, longitudinal wheel-center travel,
half-track and track, roadwheel angle, camber, caster, kingpin inclination,
scrub radius, mechanical trail, instant-center geometry, roll center, heave,
roll, ride-height change, anti-pitch geometry, damper and mechanism
travel, and applicable motion ratios. Metric availability depends on the
architecture and installed mechanisms.

Analytical constraint Jacobians are used by the nonlinear solver. Applicable
motion ratios and response derivatives are evaluated from the solved constraint
Jacobian rather than by finite differencing adjacent sweep steps.

### Explicitly outside the current model

- Multibody dynamics, inertia, damping, applied loads, and transient behavior.
- Bushing, chassis, tire, or component compliance.
- Stress, fatigue, strength, and packaging or interference checks.
- Suspension architectures other than double wishbone and MacPherson strut.
- Offset-axis MacPherson struts. The model requires the authored strut clamp to
  lie on the lower-ball-joint-to-top-mount steering axis within 1 mm.
- Arbitrary mechanism combinations. Geometry is rejected when the selected
  mechanisms do not have an implemented physical connection.

## Coordinate system and units

Suspension Explorer uses the ISO 8855 vehicle coordinate system:

- Positive X points forwards.
- Positive Y points left.
- Positive Z points upwards.
- Authored hardpoints and linear outputs use millimeters.
- Tire section width is in millimeters; rim diameter is in inches.
- Angles use radians internally and degrees in configuration and output.
- Wheel offset follows the ET convention: positive offset is inboard.

Hardpoints describe the design-condition assembly in chassis space. Fixed
chassis hardpoints remain fixed while suspension, wheel, and tire points move
relative to them. Left-side hardpoints therefore normally have positive Y
coordinates and right-side hardpoints normally have negative Y coordinates.

### Chassis and world axis systems

The system uses two right-handed axis systems: `Chassis` and `World`. Solver
state lives in chassis space; world space is constructed after solving for
metrics measured relative to ground and for presentation.

At design condition, chassis space and world space are aligned: the front axle
centreline is `X = 0`, the flat ground plane is `Z = 0`, and positive X/Y/Z
point forwards/left/upwards. The world ground plane is always level, gravity
always points along world `-Z`, and inclined roads and yaw are not supported.

Fixed chassis hardpoints do not move, and suspension points are solved relative
to them. Constructing world space never changes the accepted solver state.
The two spaces may consequently translate and rotate apart during a sweep.

A modelled axle supplies two derived wheel contact tangent points. Requiring
both to lie on world `Z = 0` establishes lateral attitude and vertical
placement, but cannot by itself determine vehicle pitch. The configured
`wheelbase`, `axle_position`, and `opposite_axle_axis_height` locate the point
where the vehicle centreline crosses the unmodelled axle's spin axis. That
point is fixed in chassis coordinates and held at the configured height in
world space. For the two current contacts `q_L` and `q_R`, opposite axle
reference `P`, configured height `h`, and world +Z unit direction `n`, placement
satisfies:

```text
n · (q_L - P) = -h
n · (q_R - P) = -h
||n|| = 1
```

This allows pitch calculation and produces the structured `WorldSpace`
transform carried by each axle analysis frame. A standalone corner cannot
determine full vehicle placement and therefore has no `WorldSpace`.
`wheel_ground_tangent` is a derived output and cannot be used as a sweep
target.

No chassis pitch scalar metric is exported. The inferred attitude is carried
by `WorldSpace` and used internally by metrics that require a ground plane. In
an axle analysis, the fixed opposite axle assumption therefore applies to:

- `ride_height_change`;
- `steering_axis_offset_ground`, `scrub_radius`, and `mechanical_trail`;
- `svsa_length` and `fvsa_length`;
- `anti_dive`, `anti_lift`, and `anti_squat`;
- `track`, which is resolved along the inferred ground lateral direction; and
- `roll`, only through that projected track value in its denominator.

These values are evaluated against the ground plane implied by holding the
opposite axle reference fixed; they do not incorporate a full vehicle pitch
solution. Chassis space metrics such as heave, wheel travel, wheel alignment,
instant centre coordinates, damper length, and rack displacement do not use
the inferred attitude.

This is a deliberately basic single axle contact model:

- The opposite axle's spin axis reference is fixed in world space; its own
  suspension movement and tyre contacts are not modelled.
- Each tyre is a nominal rigid disc. Tyre deflection, loaded radius,
  contact patch extent, forces, and compliance are not represented.
- The coupled contact solve uses a lateral/vertical tangent line extruded
  parallel to chassis X. A single axle cannot infer longitudinal road shape,
  and contact locations are not re-solved longitudinally after vehicle pitch
  is established.
- Invalid or ambiguous placement is rejected rather than replaced with a
  different ground assumption.

A future full vehicle model with both axles and all four tyre contacts could
derive vehicle placement directly and remove the opposite axle reference
assumption.

## Installation

Python 3.12 or newer is required. The package is not currently published to
PyPI.

### Core library only

Install the transport-independent solver API:

```bash
uv pip install "kinematics @ git+https://github.com/suspension-explorer/suspension-explorer-core.git"
```

This installs NumPy, SciPy, and Pydantic. It does not install YAML, CLI, export,
or plotting dependencies.

### CLI and file export

Install YAML loading and CSV/Parquet export support:

```bash
uv pip install "kinematics[cli] @ git+https://github.com/suspension-explorer/suspension-explorer-core.git"
```

### CLI with visualization

Install the CLI plus static plotting and animation support:

```bash
uv pip install "kinematics[cli,viz] @ git+https://github.com/suspension-explorer/suspension-explorer-core.git"
```

### Development checkout

```bash
git clone https://github.com/suspension-explorer/suspension-explorer-core.git
cd suspension-explorer-core
just setup
```

The development workflow uses [uv](https://docs.astral.sh/uv/) and
[`just`](https://github.com/casey/just).

## Quick start

A CLI run uses two YAML files:

1. A geometry file defines the design-condition hardpoints, architecture, and
   installed mechanisms.
2. A sweep file defines one or more coordinated target motions.

### 1. Define a corner geometry

The following is a complete rack-steered double-wishbone corner with no spring
mechanism. Save it as `geometry.yaml`.

```yaml
name: example corner
version: 1.0.0
units: millimeters
type: double_wishbone
scope: corner
side: left

actuation:
  type: direct
  mount: lower_wishbone
spring:
  type: none

config:
  steering:
    type: rack
  wheel:
    offset: 0
    tire:
      aspect_ratio: 0.55
      section_width: 270
      rim_diameter: 13
  cg_position: { x: 1250, y: 0, z: 450 }
  wheelbase: 2500
  opposite_axle_axis_height: 300

hardpoints:
  lower_wishbone_inboard_front: { x: 250, y: 400, z: 200 }
  lower_wishbone_inboard_rear: { x: -250, y: 450, z: 200 }
  lower_wishbone_outboard: { x: 0, y: 900, z: 200 }

  upper_wishbone_inboard_front: { x: 225, y: 350, z: 500 }
  upper_wishbone_inboard_rear: { x: -275, y: 350, z: 500 }
  upper_wishbone_outboard: { x: -25, y: 750, z: 500 }

  trackrod_inboard: { x: 50, y: 200, z: 250 }
  trackrod_outboard: { x: 150, y: 800, z: 275 }

  axle_inboard: { x: -20, y: 800, z: 308.426 }
  axle_outboard: { x: -20, y: 950, z: 313.426 }
```

For `steering.type: rack`, use `trackrod_inboard` and `trackrod_outboard`.
For `steering.type: none`, replace them with `toe_link_inboard` and
`toe_link_outboard`. A fixed toe link is part of the chassis geometry and is
not a steering actuator.

### 2. Define a bump sweep

Save the following as `sweep.yaml`. The wheel center moves from 40 mm of droop
to 40 mm of bump while the rack remains at its design position.

```yaml
version: 1
steps: 41
targets:
  - point: wheel_center
    direction: { axis: z }
    mode: relative
    start: -40
    stop: 40

  - point: trackrod_inboard
    direction: { axis: y }
    mode: relative
    start: 0
    stop: 0
```

Every physical actuator must be controlled exactly once. A rack-steered model
therefore needs one `trackrod_inboard` target along Y, even when the rack is
held at zero displacement. `relative` values are measured from the authored
design condition; `absolute` values are coordinates in chassis space.

All target sequences must have the same number of values. Multiple targets are
paired by index rather than expanded into a Cartesian product. Use `start`,
`stop`, and the file-level `steps`, or give every target an equal-length
`values` list.

### 3. Check the design condition

```bash
uv run kinematics visualize --geometry geometry.yaml --output geometry.png
```

This validates and builds the geometry, reports whether every derived
wheel-ground tangent point lies on `Z = 0`, and writes a static image. It
requires `[cli,viz]`.

### 4. Solve and export the sweep

Write CSV output:

```bash
uv run kinematics sweep \
  --geometry geometry.yaml \
  --sweep sweep.yaml \
  --out results.csv
```

Use a `.parquet` output suffix for Parquet. Add `--animation-out motion.gif` or
`--animation-out motion.mp4` to render the solved motion when visualization
dependencies and the corresponding animation writer are installed.

The output is wide-form: each row is one sweep step, point coordinates use
lowercase `snake_case` columns, and applicable metrics and solver information
are included alongside the positions. Diagnostics are printed to stderr and do
not discard otherwise usable solved frames.

<p align="center">
  <img src="images/animation.gif" alt="Kinematic sweep animation" width="80%">
  <br>
  <em>A coordinated bump, droop, and steering sweep.</em>
</p>

## Full-axle inputs

Set `scope: axle` to solve two corners together. Axle files separate
vehicle-wide configuration, axle configuration, side hardpoints, and shared
center hardpoints:

```yaml
type: double_wishbone
scope: axle
name: example axle
version: 1.0.0
units: millimeters

vehicle_config:
  cg_position: { x: 1250, y: 0, z: 450 }
  wheelbase: 2500
  opposite_axle_axis_height: 300

axle_config:
  axle_position: front
  steering: { type: rack }
  actuation: { type: direct, mount: lower_wishbone }
  spring: { type: none }
  anti_roll: { type: none }
  heave_link: { type: none }
  wheel:
    offset: 0
    tire:
      aspect_ratio: 0.55
      section_width: 270
      rim_diameter: 13

hardpoints:
  left:
    # The same left-corner hardpoints used above.
    # Omit `right` to mirror this complete map through Y = 0.
    # ...
```

The complete maintained examples are:

- [Mirrored double-wishbone axle](tests/data/axle_geometry.yaml)
- [Explicit asymmetric double-wishbone axle](tests/data/axle_geometry_explicit.yaml)
- [MacPherson axle](tests/data/macpherson_axle_geometry.yaml)
- [Pushrod-rocker axle with U-bar](tests/data/axle_geometry_rocker.yaml)
- [Pushrod-rocker axle with T-bar](tests/data/axle_geometry_t_bar.yaml)

Axle sweep targets must identify `side: left` or `side: right` for side-local
points. A rack has one shared lateral degree of freedom, so target either side's
`trackrod_inboard` along Y exactly once. For example, a three-step roll sweep is:

```yaml
version: 1
targets:
  - point: wheel_center
    side: left
    direction: { axis: z }
    mode: relative
    values: [-30, 0, 30]

  - point: wheel_center
    side: right
    direction: { axis: z }
    mode: relative
    values: [30, 0, -30]

  - point: trackrod_inboard
    side: left
    direction: { axis: y }
    mode: relative
    values: [0, 0, 0]
```

## Python API

`kinematics.core` accepts already-decoded mappings and has no YAML or filesystem
dependency. This is the preferred boundary for applications embedding the
solver:

```python
from kinematics.core.analysis import analyze_sweep
from kinematics.core.input import build_suspension, build_sweep

# `geometry_data` and `sweep_data` are decoded mappings supplied by the caller.
suspension = build_suspension(geometry_data)
sweep = build_sweep(sweep_data, suspension)
analysis = analyze_sweep(suspension, sweep)

for frame in analysis.frames:
    print(frame.index, frame.positions, frame.metrics)
```

`analyze_sweep()` returns structured suspension metadata, named point positions,
metric metadata, per-frame solver information, applicable corner and axle
metrics, renderer-neutral element paths, reference conditions, and diagnostics.
The CLI is a thin adapter around this core API for YAML input and file output.

## How the solver works

```text
decoded geometry mapping
        |
        v
validate schema and build suspension topology
        |
        v
derive initial points, constraints, and actuator degrees of freedom
        |
        v
validate and expand coordinated sweep targets
        |
        v
solve each step with scipy.optimize.least_squares
        |
        v
calculate derived points, metrics, derivatives, and diagnostics
        |
        v
structured analysis or CLI file export
```

Rigid links and bodies are represented by geometric constraints. For each sweep
step, the solver finds the coordinates of all free points that minimize the
constraint and target residuals. The problem is solved as nonlinear least
squares with SciPy's Levenberg-Marquardt implementation and analytical
Jacobians.

This lets the same suspension topology be driven by targets such as wheel-center
height and rack displacement without deriving a separate closed-form solution
for every motion. The previous solved state seeds the next step, and diagnostics
report convergence, residual acceptance, branch continuity, derivative
availability, mechanism chirality, and transmission-margin problems.

## Project structure

```text
src/kinematics/
  core/                    Transport-independent solver and analysis API
    schema/                Strict geometry, configuration, and sweep models
    suspensions/
      corner/              Double-wishbone and MacPherson corner models
      axle/                Generic two-corner composer and shared mechanisms
    points/derived/        Dependency-aware derived point calculations
    metrics/               Corner, axle, and derivative metrics
    primitives/            Geometry, rigid bodies, vectors, and point keys
    constraints.py         Constraint residuals and analytical Jacobians
    solver.py              Nonlinear solve and Jacobian assembly
    sweep.py               Sweep solving, metrics, and diagnostics
    analysis.py            Structured application-facing result model
  cli/                     YAML, export, terminal, and visualization adapters
tests/
  data/                    Valid example geometries, sweeps, and e2e references
tools/
  generate_jacobians.py    Symbolic Jacobian generator
```

`Suspension` defines the common model interface. Concrete corner classes own
their locating geometry and point-role hooks. `AxleSuspension` composes two
already-built corners and the optional shared anti-roll and heave mechanisms;
new locating architectures belong in `suspensions/corner/`, not in a new axle
class.

## Development

Common commands are:

```bash
just test
just check
just format
just spellcheck
```

Run manual visualization tests with:

```bash
uv run pytest tests/ -m ""
```

Generated analytical Jacobians live in `src/kinematics/core/jacobians.py`. Edit
their symbolic definitions in `tools/generate_jacobians.py` and regenerate them
with `just generate-jacobians` rather than manually changing generated
expressions.
