# Ticket: axle-level ground line in chassis coordinates

## Status

Implemented. This is now the core geometry and export prerequisite for later
improvements to anti-dive, anti-lift, anti-squat, roll-center, ground-intersection,
and ride-height calculations.

## Implementation note

`kinematics.core.metrics.AxleGroundLine` derives the immutable chassis-frame `YZ`
road line once per solved axle state from the two wheel-plane road-tangent points. Its
optional road-plane form is the mathematical extrusion of that line parallel to
chassis `±X`, so longitudinal grade is deliberately assumed zero rather than inferred.
The shared instance is passed to both corner metric contexts and the axle metric row
exports:

```text
ground_line_normal_y
ground_line_normal_z
ground_line_offset
ground_line_angle
ground_z_centerline
```

The scalar row is the current structured/API representation; it carries explicit
axle scope and units through the existing export paths. `anti_geometry.py` has not
yet migrated to consume the shared axle datum.

## Summary

Introduce an explicit, state-derived ground representation for a solved two-corner
axle. For the current axle-only model, the canonical representation is a road line in
the axle's lateral/vertical (`YZ`) cross-section passing through the left and right
wheel-plane road-tangent points.

Two wheel-plane road-tangent points define an axle road line, but they do not define
longitudinal road grade. Where a plane representation is useful for APIs or exports,
define it as the axle road line extruded parallel to chassis `±X`. This is a deliberate
zero-grade assumption, not a full-vehicle road plane.

The road representation remains expressed in the existing chassis-fixed coordinate
system. It does not require a rear axle, a world/inertial frame, or a full-vehicle
pose.

## Why this is needed

The solver currently stores every authored and solved point in one Cartesian frame:

- `+X` is chassis forward.
- `+Y` is chassis left.
- `+Z` is chassis up.
- Fixed chassis hardpoints remain fixed in that frame.
- Suspension and tire points move relative to those hardpoints.

Although some documentation calls this a world frame, the implementation is
chassis-fixed. There is no separate road-frame transform, inertial vertical, chassis
pose, road surface, or explicit ground entity in `SuspensionState`.

Before this implementation, `MetricContext.ground_z` treated the selected corner's
wheel-plane road-tangent `Z` as a horizontal road plane. On an asymmetric axle state,
the left and right contexts therefore inferred different horizontal planes. A single
vehicle CG could consequently receive two different ground-height references. This was
sufficient for symmetric flat-road heave, but it was not a coherent axle datum for
roll, one-wheel bump, or asymmetric geometry.

The implemented axle path derives the line through both wheel-plane road-tangent
points once and shares it with both corner contexts, providing one axle-cross-section
datum without pretending that a full-car road plane is known. Standalone corners retain
the local wheel-plane road-tangent datum and assume a `+Z` road normal because they do
not contain the second point.

## Required model

### Ownership and lifetime

The ground line must:

- be derived for each solved **axle** state;
- use the current left and right `WHEEL_PLANE_ROAD_TANGENT` positions;
- be expressed in chassis coordinates;
- be immutable for the duration of metric evaluation;
- be calculated once and shared by all axle and corner metric consumers for that
  state; and
- remain derived output rather than authored configuration or a solver degree of
  freedom.

A standalone corner does not contain enough information to construct the axle road
line. It assumes a `+Z` road normal at its local wheel-plane road-tangent height and
must not be silently presented as an axle-derived road plane.

### Mathematical definition

Let the current wheel-plane road-tangent points, projected into the chassis `YZ`
cross-section, be:

```text
q_R = (y_R, z_R)
q_L = (y_L, z_L)
```

Orient the ground-line tangent from vehicle right to vehicle left:

```text
d = q_L - q_R = (dy, dz)
length = sqrt(dy^2 + dz^2)
t = d / length = (t_y, t_z)
```

Normally `dy > 0`. If necessary, reverse `t` so its lateral component is positive.
Define an upward-pointing unit normal in the cross-section:

```text
n = (n_y, n_z) = (-t_z, t_y)
```

Reverse `n` if required so `n_z >= 0`. The canonical Hessian-normal line equation is:

```text
n_y y + n_z z + c = 0
c = -(n_y y_R + n_z z_R)
```

The representation is valid when the two projected wheel-plane road-tangent points are
separated by more than the geometric tolerance. A collapsed track or coincident
projected points must produce an explicit undefined result or diagnostic, not NaN
coefficients.

Useful derived values are:

```text
ground angle = atan2(dz, dy)

z_ground(y) = -(n_y y + c) / n_z

signed normal distance(q) = n_y q_y + n_z q_z + c
```

`z_ground(y)` is undefined when `n_z` is within the geometric tolerance of zero. The
normal-form line can remain valid in that case.

### Wheel-plane road-tangent construction

For one wheel, let `C` be its centre, `R` its nominal radius, `a` its normalized spin
axis, and `n` the road-plane unit normal. Project the road normal into the wheel plane
and normalize it:

```text
q = n - (n · a) a
P = C - R q / ||q||
```

`P` is the canonical wheel-plane road-tangent point: the ideal rigid-tire proxy for
the road tangent in the wheel plane. It is undefined when `q` is degenerate. For an
axle, solve the shared road normal from the two points with:

```text
n = (0, n_y, n_z), n_z > 0
n · P_L = n · P_R
```

The `X = 0` normal component means the resulting road-plane height is independent of
chassis `X`: this is the zero-longitudinal-grade assumption. It does **not** mean that
`Z` is constant across `Y`; axle roll/bank remains represented by the `YZ` road line.

For a CG position expressed in the same chassis frame, two distinct quantities can be
calculated and must not be conflated:

```text
chassis-Z separation = cg_z - z_ground(cg_y)

normal separation = n_y cg_y + n_z cg_z + c
```

They are identical for a level axle line. A later anti-geometry ticket must choose the
quantity appropriate to its force model rather than reading a corner-local
wheel-plane road-tangent height.

### Optional plane representation

For a plane-shaped API or visualization primitive, extrude the axle road line parallel
to chassis `±X`:

```text
N = (0, n_y, n_z)
N_x x + N_y y + N_z z + c = 0
```

This is the only plane that this ticket should expose. Its `N_x = 0` encodes the
assumption that longitudinal grade/pitch is unknown and treated as zero. It must be
named or tagged as an axle-derived, X-extruded plane so consumers do not mistake it
for a measured or full-vehicle road plane.

## Integration requirements

1. Add a small typed value object for the axle ground line. It should own the unit
   tangent, upward unit normal, offset, validity checks, and evaluation helpers rather
   than distributing line arithmetic between metrics.
2. Compute it near the start of `compute_metrics_for_axle_state`, before the per-corner
   metric loop.
3. Make the same instance available to both corner metric contexts and axle-level
   metrics. An optional axle-ground field on `MetricContext`, or a dedicated
   `AxleMetricContext`, are both acceptable designs.
4. Do not store the ground line as hardpoints in `SuspensionState`; it is derived from
   state, just like instant centres and roll centre.
5. Keep existing point positions and solver constraints in chassis coordinates. No
   coordinate transformation should be applied to the solved state as part of this
   work.
6. Do not change anti-geometry formulas in this ticket. This work provides the common
   datum and API needed by a follow-up anti-geometry change.

## Export requirements

Ground geometry is axle-scoped. It must be capable of appearing in structured API
results, flat CSV output, metadata, and visualization data without reconstructing it
from left/right corner columns.

The canonical structured representation should include enough information to recreate
the line or its X-extruded plane:

```yaml
ground:
  model: axle_line_yz_extruded_x
  frame: chassis
  normal: {x: 0.0, y: <n_y>, z: <n_z>}
  offset_mm: <c>
  angle_deg: <atan2(dz, dy)>
```

The normal must be unit length and upward-oriented. `offset_mm` uses the equation
`normal dot position + offset_mm = 0`, with positions in millimetres.

For the existing scalar metric/export system, the minimum equivalent axle columns are:

```text
ground_line_normal_y       dimensionless
ground_line_normal_z       dimensionless
ground_line_offset         mm
ground_line_angle          deg
ground_z_centerline        mm
```

`ground_z_centerline` means `z_ground(0)`. It is a convenient display value, not the
canonical geometry. The normal and offset are the canonical export because they allow
evaluation at any lateral coordinate.

If both structured and flat representations are implemented, they must be generated
from the same typed ground-line object. Metric metadata must mark the scalar fields as
axle-scoped and provide the units above. Undefined geometry must serialize consistently
with other undefined metrics (`null` in structured results and the existing empty-value
policy in CSV).

## Documentation requirements

Update the public coordinate-system documentation to state explicitly that:

- authored and solved coordinates are chassis-fixed, despite historical references to
  a world frame;
- an authored design wheel-plane road-tangent point at `Z = 0` is a useful convention,
  not a permanent road constraint;
- wheel-plane road-tangent coordinates move through suspension sweeps;
- axle road geometry is derived from both current wheel-plane road-tangent points and
  is expressed in the chassis frame;
- `absolute` sweep targets are absolute in chassis coordinates, not an inertial/world
  frame; and
- the axle ground plane has no longitudinal-grade information and is an X extrusion of
  the axle ground line.

Internal docstrings should distinguish the standalone corner's local `+Z` road-normal
assumption from the shared axle road line.

## Acceptance criteria

- Equal left/right wheel-plane road-tangent `Z` produces a `YZ` normal of `(0, 1)`
  (and exported plane normal `(0, 0, 1)`), an angle of zero, and
  `ground_z_centerline` equal to that common `Z`.
- Equal and opposite wheel-plane road-tangent `Z` changes produce the expected signed
  ground-line angle while preserving the midpoint/centerline height.
- Equal vertical movement of both wheel-plane road-tangent points translates the line
  by the same amount without changing its angle or normal.
- Mirroring a symmetric axle preserves the deterministic right-to-left tangent and
  upward-normal orientation.
- Fore/aft differences between the two wheel-plane road-tangent points do not affect
  the axle `YZ` road line; this projection is intentional and documented.
- Degenerate projected wheel-plane road-tangent points return an undefined ground
  object or clear diagnostic without division by zero, infinity, or NaN.
- Both corner metric evaluations for one axle state receive the same ground-line
  instance or numerically identical immutable value.
- Structured and flat exports reproduce the line equation and carry explicit chassis
  frame/model metadata where the format supports it.
- Existing axle state and export tests remain green, with new unit tests for line
  construction, orientation, translation, roll, degeneracy, and export metadata.
- The public coordinate-system page no longer describes solved suspension coordinates
  as an inertial world frame.

## Non-goals

- A full-car model joining front and rear axles.
- Longitudinal road grade or chassis pitch relative to an inertial frame.
- Banked or curved road-surface authoring.
- Tire compliance, loaded radius, or tire-road force-distribution modelling.
- Enforcing contact constraints in the kinematic solver.
- Changing anti-dive, anti-lift, anti-squat, roll-center, scrub, or trail equations.
- Defining an axle ground line from a standalone corner.
- Renaming or transforming every existing chassis-coordinate point.

## Follow-up enabled by this ticket

Once a common axle ground datum is available, anti-geometry can stop calculating CG
height independently from each corner's wheel-plane road-tangent `Z`. A follow-up
should consume the shared ground line, select and document the appropriate CG-to-ground height
measure, add brake/drive reaction-topology inputs, and revalidate the anti calculations
for asymmetric axle states.
