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
| Locating architectures    | Double wishbone, MacPherson strut, multi-link (five-rod), and semi-trailing arm           | Each may be built as one corner or a composed two-corner axle.                                                     |
| Axle geometry             | Mirrored or explicitly authored left and right corners                                    | If `hardpoints.right` is omitted, the complete left geometry and side-local setup are mirrored through `Y = 0`.    |
| Wheel-heading control     | Translating steering rack or fixed toe link                                               | Select `steering.type: rack` or `steering.type: none`; front/rear position does not select steering automatically. |
| Double-wishbone actuation | Direct or pushrod-rocker, mounted to the lower wishbone or upright                        | Direct actuation cannot be combined with a torsion bar.                                                            |
| Double-wishbone springs   | None, coilover, or torsion bar                                                            | A torsion bar requires pushrod-rocker actuation.                                                                   |
| Multi-link corners        | Four independent locating rods plus track rod or toe link; actuation on the upright or a direct spring on a lower link's centreline | No physical kingpin exists; steering geometry reports through the virtual (screw-axis) metric family only.        |
| Axle mechanisms           | U-bar or T-bar anti-roll mechanism and rocker-to-rocker heave link                        | These mechanisms require a double-wishbone axle with pushrod-rocker actuation.                                     |
| Setup changes             | Outboard camber shims on double-wishbone corners                                          | Explicit asymmetric axle hardpoints require corresponding side-local setup when a shim is used.                    |
| Outputs                   | Solved point positions, solver statistics, diagnostics, metrics, in either CSV or Parquet | Plotting and animation require the optional visualization dependencies.                                            |

The calculated metrics include wheel travel, longitudinal wheel-center travel,
half-track, ISO track, track change, toe angle, ISO steer angle, camber, caster,
kingpin inclination, scrub radius, mechanical trail, instant-center geometry,
roll center, heave, suspension roll, ride-height change, anti-pitch geometry,
damper and mechanism travel, and applicable motion ratios. Metric availability
depends on the architecture and installed mechanisms.

Analytical constraint Jacobians are used by the nonlinear solver. Applicable
motion ratios and response derivatives are evaluated from the solved constraint
Jacobian rather than by finite differencing adjacent sweep steps.

### Explicitly outside the current model

- Multibody dynamics, inertia, damping, applied loads, and transient behavior.
- Bushing, chassis, tire, or component compliance.
- Stress, fatigue, strength, and packaging or interference checks.
- Suspension architectures other than double wishbone, MacPherson strut,
  multi-link, and semi-trailing arm.
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

The system introduces two right-handed coordinate systems as `Chassis` and
`World`. Chassis space follows the ISO 8855:2011 vehicle axis system: X points
forwards, Y left, and Z upwards, with the basis fixed to the sprung mass.
Solver variables, constraints, hardpoints, and solved positions exist only in
this system.

World space follows the ISO earth-fixed axis system. World X and Y lie in the
ISO ground plane, world Z points upwards, and gravity is always world `-Z`.
This project considers only a straight, level road, so the ISO local road
plane, ground plane, and world `Z = 0` plane coincide. Road grade, road bank,
yaw, and non-planar surfaces are outside the model.

At design condition the chassis and world axes are aligned, the front axle
centreline is `X = 0`, and the wheel contact-centre line is `Z = 0`. During a sweep,
fixed hardpoints remain fixed in chassis space while the road plane may move
relative to them as the modelled axle heaves or rolls. This represents
suspension motion associated with vehicle-generated vertical, lateral, or
longitudinal forces; the solver is kinematic and does not calculate those
forces or a dynamic body attitude.

The axle contact closure models each tyre as a rigid disc and returns two
wheel contact centres. It constructs the single plane tangent to both wheels
and extrudes the contact line parallel to chassis X. Consequently:

- local axle heave and roll relative to the road are observable;
- longitudinal road gradient is zero by construction;
- one axle cannot determine whole-vehicle pitch, yaw, or longitudinal
  translation, so these are assigned zero rather than inferred;
- an opposite-axle pivot is neither required nor modelled; and
- `wheel_contact_centre` is an output and cannot be a sweep target.

The resulting `WorldSpace` value is a presentation transform only. It maps the
same axle-local road plane to world `Z = 0`, preserves chassis +X as world +X,
and rotates/translates for local roll and heave. Metric calculation does not
consume this transform.

Metric reference systems are deliberate:

- `camber` is the ISO vehicle-relative camber angle, while road-relative wheel
  inclination is not currently exported;
- `caster`, `kpi`, and ISO `steer_angle` use the chassis/vehicle axes;
- `toe_angle` is the project convention: a side-folded roadwheel heading where
  positive means toe-in; it is reported alongside, rather than substituted for,
  the ISO vehicle-fixed `steer_angle`;
- `steering_axis_offset_ground`, `scrub_radius`, and `mechanical_trail` use
  the ISO tyre axes on the local road plane;
- `track` is the ISO rest dimension on horizontal ground; `track_change`,
  `ride_height_change`, swing-arm lengths, and geometric anti percentages use
  the axle-local road plane represented in chassis coordinates;
- wheel travel, heave, instant-centre coordinates, rack displacement, and
  roll-centre coordinates use chassis axes; and
- damper length and other Euclidean link lengths are invariant under the
  chassis-to-world rigid transform.

`ride_height_change` is therefore the change in perpendicular clearance from
the chassis origin to the axle-local road plane. It is not a full-vehicle ride
height or pitch result. Likewise, the exported `roll` is a kinematic axle
state calculated as the ISO suspension roll angle of the current line joining
the wheel centres, not a solved sprung-mass attitude. The anti percentages are
geometric construction metrics; they do not predict pitch under load.

The contact model omits tyre deflection, loaded radius, contact-patch extent,
forces, compliance, and interaction with another axle. A future full-vehicle
model could observe pitch from both axles, but that degree of freedom is
intentionally absent from the present single-axle model.

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
  - type: point
    point: wheel_center
    side: left
    direction: { axis: z }
    mode: relative
    start: -40
    stop: 40

  - type: actuator_position
    actuator: rack
    direction: { axis: y }
    hold: true
```

Every physical actuator must be controlled exactly once. A rack-steered model
therefore needs one `type: actuator_position`, `actuator: rack` control along Y.
`hold: true` captures that coordinate at the sweep reference state and enforces
the captured value at every step; it can likewise hold a point coordinate or an
element length. Held controls have no `mode`, `start`, `stop`, or `values`.
Swept `relative` values are measured from the authored design condition;
`absolute` values are coordinates in chassis space. Every corner-owned target
must identify `side: left` or `side: right`.
A standalone corner exposes only `left`; an axle exposes both sides. Shared
coordinates such as `rack` remain unsided. A physical `trackrod_inboard` point
target does not substitute for the named rack actuator coordinate.

All swept target sequences must have the same number of values. Multiple targets are
paired by index rather than expanded into a Cartesian product. Use `start`,
`stop`, and the file-level `steps`, or give every target an equal-length
`values` list. A profile must contain at least one swept target in addition to
any held controls.

### 3. Check the design condition

```bash
uv run kinematics visualize --geometry geometry.yaml --output geometry.png
```

This validates and builds the geometry, reports whether every derived wheel
contact centre lies on the reconstructed road plane, and writes a static
image. The diagnostic also prints each centre's raw chassis Z coordinate and
signed road-plane distance. It requires `[cli,viz]`.

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
points. A rack has one shared lateral degree of freedom, exposed as the
side-independent `rack` actuator position. For example, a three-step roll sweep
is:

```yaml
version: 1
targets:
  - type: point
    point: wheel_center
    side: left
    direction: { axis: z }
    mode: relative
    values: [-30, 0, 30]

  - type: point
    point: wheel_center
    side: right
    direction: { axis: z }
    mode: relative
    values: [30, 0, -30]

  - type: actuator_position
    actuator: rack
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

### Steering-response axes

> A steering-response axis is obtained from unit positive rack motion at the
> current solved configuration while topology-declared suspension-travel
> coordinates are held at their current values.

The response is a separate analytical derivative at each already-solved state;
it does not inherit the targets that authored the surrounding sweep. A double
wishbone defaults to its lower-wishbone angle, while the supported
MacPherson model defaults to strut length. An axle composes independent left and
right holds with its shared rack. If a topology cannot declare a sufficient
suspension hold, the response is unavailable rather than selected from an
underconstrained motion family.

Each topology also publishes its available hold choices. The sweep may leave the
default choice implicit or select one by stable ID:

```yaml
analysis:
  virtual_steering:
    suspension_hold: upper_wishbone_angle
```

This block changes analysis only; it never adds a target to the state solve.
Each option names the local counterfactual it represents. Availability warnings,
such as fixed damper length with an upright-mounted pushrod, remain in structured
analysis output.

One result is calculated for every upright at every rack-steered sweep step and
exposed as `frame.steering_response_axes`. It carries the axis point and
direction, angular rate, screw pitch, rigid-body fit residuals, point count, and
an explicit validity status. The same data is drawn as a clipped dash-dot line
in every view of CLI animations without contributing to automatic plot bounds.

This is an instantaneous kinematic axis, not necessarily a physical kingpin or
ball-joint line. A spatial linkage or a real steering-to-spring coupling may
produce nonzero screw pitch. An incomplete or inconsistent suspension hold,
near-pure translation, degenerate upright geometry, or a poor rigid-body fit
leaves the axis unavailable for that frame and reports a diagnostic. The response
uses an analytical tangent at that state; it does not finite-difference adjacent
sweep frames, perturb and re-solve the rack, or modify the authored sweep.

Existing caster, KPI, steering-axis offset, scrub-radius, and mechanical-trail
metrics retain their physical steering-axis definitions. Rack-steered results
also report an additive, motion-derived family using the suffix `_virtual`:
`caster_virtual`, `kpi_virtual`, `steering_axis_offset_ground_virtual`,
`scrub_radius_virtual`, and `mechanical_trail_virtual`. Each is ordered beside
its physical counterpart and displayed with labels such as `Caster, Virtual`.
Here **virtual steering axis** means the isolated steering-response screw-axis
line above. The values use the same chassis, tyre, road-plane, and sign
conventions as their physical-axis counterparts. They are `None` when that
frame has no valid finite axis; selecting another published hold changes only
the virtual family and never the original physical metrics. Screw pitch and
angular rate remain separate axis properties
rather than being folded into these five line-based geometry values.

Holding wheel-centre height during the authored sweep is not the same as fixing
the wishbones: it adds the jounce needed to cancel vertical motion from steering
around an inclined axis. That remains the correct solved path, but it no longer
changes the virtual steering definition. At each of those states, the selected
suspension hold fixes current suspension travel and recovers the steering-only response.
For the ideal double-wishbone fixture, the default fitted virtual line
therefore agrees with the ball-joint line even while the authored internals move
between frames.

Internally, physical pivots and the motion fit each establish the same
source-agnostic `SteeringAxis` representation. One common geometry path then
computes its road intersection, caster, KPI, offset, scrub radius, and trail.

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
calculate analytical tangents once, then metrics, steering axes, and diagnostics
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

## License

From version 0.6.0 onward, Suspension Explorer Core is licensed under the
**GNU Affero General Public License v3.0 only (AGPL-3.0-only)**. Alternative
commercial licensing is available by separate agreement.

| Versions | License                                                        |
| -------- | -------------------------------------------------------------- |
| ≤ 0.5.1  | Apache License 2.0                                             |
| ≥ 0.6.0  | AGPL-3.0-only, with alternative commercial licensing available |

Releases up to and including v0.5.1 were published under Apache-2.0. That
license grant remains in force for those releases.

The AGPL version is free to use, including for commercial engineering,
research, education, motorsport, and hobby projects, provided that you comply
with its terms.

Alternative commercial licensing is intended for organisations that want to
incorporate Suspension Explorer Core into proprietary or closed-source
software or services where complying with the AGPL is not suitable.

For commercial licensing, contact
[contact@nickmccleery.com](mailto:contact@nickmccleery.com).

### Can I use it for confidential or commercial engineering work?

Yes.

The AGPL applies to the software, not simply to data processed by the
software. Using Suspension Explorer Core to analyse a commercial race car,
confidential vehicle programme, Formula Student car, or other proprietary
design does not by itself require you to disclose:

- hardpoint coordinates;
- vehicle geometry;
- simulation inputs;
- analysis results;
- setup data; or
- other confidential engineering information.

You can therefore use the solver internally for commercial or confidential
engineering work under the AGPL.

Additional obligations may arise if you modify Suspension Explorer Core,
distribute it or a work based on it, combine or incorporate it into other
software, or make a modified version available for users to interact with
over a network.

If you want to incorporate the solver into a proprietary application or
service without complying with the applicable AGPL obligations, contact us
about a commercial license.

This section is an informal summary intended to explain the project's
licensing model. The applicable license terms govern.

### Contributions

Contributions are welcome.

Because Suspension Explorer Core remains available under the AGPL while also
being offered under separate commercial terms, the project needs sufficient
rights to include accepted contributions under both licensing models.

Before contributing, please read [CONTRIBUTING.md](CONTRIBUTING.md).

Pull requests include the following contributor acknowledgement:

> - [ ] I have read and agree to the contributor terms in
>       `CONTRIBUTING.md`. I confirm that I own this contribution or have
>       authority to submit it, and I assign the copyright in my contribution
>       to Nick McCleery, as maintainer of Suspension Explorer Core, on the
>       terms stated there. By checking this box and submitting this pull
>       request, I intend this acknowledgement to constitute my electronic
>       signature and acceptance of those terms.

Accepted contributions remain available as part of the open-source project
under **AGPL-3.0-only**. The copyright assignment also allows those
contributions to be included in separately licensed commercial versions of
Suspension Explorer Core.
