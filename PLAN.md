# Axle Simulation & Pushrod/Rocker Architecture — Plan and Work Log

Working branch: `claude/suspension-axle-simulation-b71drh`

## Goal

1. **Phase A — Full axle (two-corner) simulation** for the existing double wishbone
   model: left and right corners solved together in a single constraint system,
   coupled through the steering rack, with per-side and axle-level metrics.
2. **Phase B — F1-style inboard actuation**: pushrod from the upright to an inboard
   rocker (rocker rotation axes parallel to the XZ plane), each rocker actuating a
   torsion bar (coaxial with the rocker pivot) and an inboard anti-roll bar via
   droplink.

Backward compatibility: the existing single-corner `double_wishbone` model, its YAML
schema, CLI, and e2e reference outputs must keep working unchanged.

## Coordinate system and sign conventions

The codebase uses **ISO 8855**: X forward, Y left, Z up (right-handed). This is
already asserted in `src/kinematics/metrics/angles.py` and
`steering_geometry.py`. Consequences we rely on:

- The existing test geometry (`tests/data/geometry.yaml`, all Y > 0) is a **left**
  corner.
- Mirroring left ↔ right is a reflection through the XZ plane: `y → -y` for points;
  for directions, the Y component negates.
- `MetricContext.side_sign` is `+1` for left (Y > 0), `-1` for right — existing
  corner metrics (camber, toe/roadwheel angle, scrub radius, KPI) already encode
  handedness through geometry and need no changes per side.
- Steering: the rack translates along world Y. Positive rack displacement (+Y,
  leftward) pushes the left trackrod outboard and pulls the right one inboard;
  with front-of-axle steering arms (trackrod outboard forward of the kingpin, as
  in the test geometry) this steers both wheels **right** (negative roadwheel
  angle per the existing `calculate_roadwheel_angle` convention: positive =
  steer left / toe-in for the left wheel).
- **Rotations about an axis** (rockers, ARB arms, torsion bars): signed by the
  right-hand rule about the axis' *defined direction* (first axis point → second
  axis point as given in the YAML). Angles are reported relative to the design
  condition (zero at design ride height).

## Existing architecture (survey summary)

- `PointID` (IntEnum, `core/enums.py`) is the single global point namespace. All
  machinery — `SuspensionState.positions: dict[PointID, Point3]`, `Constraint`
  subclasses, `ResidualComputer`, `DerivedPointsManager`, metrics, writers — keys on
  it. **Nothing in the runtime actually requires the key to be a `PointID`**: it
  needs hashability, ordering (state sorts `free_points`), and a `.name` for output
  columns. This is the lever for the axle refactor.
- `Suspension` ABC (`suspensions/base.py`) + `DoubleWishboneSuspension` build:
  initial state from hardpoints, 17 constraints (15 link-length `DistanceConstraint`s,
  1 upright `AngleConstraint`, 1 rack `PointOnLineConstraint` along Y), free points
  (6 → 18 vars), derived-point spec (wheel center, contact patch...), SVIC/FVIC.
- Solver (`solver.py`): scipy `least_squares` (LM) over the flattened free-point
  array; residuals = constraints + one row per sweep target; analytical Jacobians
  per constraint type (`jacobians.py`, SymPy-generated) plus dual-number autodiff
  for derived-point targets. LM requires residuals ≥ vars.
- Targets (`PointTarget`) drive the solve: project a point's position on a direction
  to a value. Sweep YAML (`io/sweep_loader.py`) expands targets per step.
- Metrics: per-corner catalog (`metrics/catalog.py`) computed via `MetricContext`.
- IO: geometry YAML → `load_geometry` → registry → suspension class; results writers
  take **string-keyed** flattened positions (side prefixes are trivial there).

## Phase A — architecture

### A1. Side-qualified point keys

New in `core/enums.py` / `core/point_ref.py`:

```python
class Side(IntEnum):
    LEFT = 0
    RIGHT = 1
    CENTER = 2   # shared chassis elements: rack, ARB axis

class PointRef(NamedTuple):
    side: Side
    point: PointID
    # .name property -> "left_lower_wishbone_outboard" etc. for output columns
```

Core machinery is **generalised over the key type, not rewritten**: introduce a
`PointKey` type alias (`PointID | PointRef`) and loosen annotations in `state.py`,
`constraints.py`, `solver.py`, `points/derived/manager.py`, `core/types.py`
(`PointTarget.point_id`). Runtime behaviour is unchanged — dict lookups, sorting,
and the Jacobian distribution logic are already key-agnostic. Single-corner models
keep plain `PointID` keys; no behavioural change, e2e refs stay byte-identical.

Constraints gain a `remap(fn: Callable[[PointKey], PointKey]) -> Constraint` method
(implemented per class over its point attributes) so corner-built constraints can be
re-keyed into a side namespace.

### A2. `DoubleWishboneAxleSuspension` (composition, not inheritance of behaviour)

New `suspensions/axle.py`. The axle **composes two `DoubleWishboneSuspension`
instances** (left/right geometry containers) and presents the same protocol surface
the solver/CLI need (`initial_state`, `free_points`, `constraints`, `derived_spec`,
`OUTPUT_POINTS`-equivalent, `get_visualization_links`, config). Key mechanisms:

- **State**: `{PointRef(side, pid): pos for pid, pos in corner.hardpoints}` for both
  sides; free points = each corner's free points, side-tagged.
- **Constraints**: each corner's `constraints()` remapped via
  `c.remap(lambda pid: PointRef(side, pid))`, plus **coupling constraints**:
  - Rigid steering rack: `DistanceConstraint(PointRef(L, TRACKROD_INBOARD),
    PointRef(R, TRACKROD_INBOARD), design separation)`. Both sides keep their
    per-corner rack `PointOnLineConstraint` along Y.
- **Derived points**: reuse each corner's spec unchanged through a lazy
  side-view mapping (`_SideView(Mapping)` whose `__getitem__(pid)` reads
  `positions[PointRef(side, pid)]`), wrapping each corner derived function. Works
  transparently for the dual-number autodiff path.
- **Metrics**: per side, strip the side tag to build a plain corner
  `SuspensionState` and reuse `compute_metrics_for_state` + the corner suspension
  unchanged; prefix columns `left_` / `right_`. Add **axle-level metrics**:
  - `roll_center_y_mm`, `roll_center_z_mm`: intersection of the two front-view
    lines contact-patch → FVIC (None if either FVIC undefined or lines parallel).
  - `total_toe_deg` (left + right toe-in), `track_mm` (contact patch ΔY),
    `rack_displacement_mm`.
- **Solvability** (steer + both wheel heights targeted): 36 vars;
  2×17 corner constraints + 1 rack + 3 targets = 38 residuals ≥ 36 ✓. The rack DOF
  must be pinned by a steering target (same requirement as the single-corner model).

### A3. YAML schema (axle)

```yaml
type: double_wishbone_axle
name: ...
units: MILLIMETERS

# Mirror mode: one side given, other generated by y -> -y.
hardpoints:
  side: left          # which side the points below describe (default: left)
  mirror: true        # default true when a flat block is given
  points: { lower_wishbone_inboard_front: {x,y,z}, ... }

# Explicit mode: both sides given.
hardpoints:
  left:  { ... }
  right: { ... }

config: <same corner config as today, shared by both sides>
```

Loader validates: explicit sides both complete; mirror source side consistent with
the sign of its Y coordinates (warn/error if a "left" block has Y < 0 outboard).
Camber shim config, when present, applies per side (mirrored for the generated
side: points y-negated, face normal Y-negated).

### A4. Sweep targets (side-qualified)

`TargetSpec` gains an optional `side: left | right` field (required when the
geometry is an axle and the point is per-side). `PointTarget.point_id` becomes a
`PointKey`. CLI resolves specs → keys through the loaded model so single-corner
files keep working with no `side` field.

Typical axle sweep: `wheel_center z` per side (equal = heave, opposite = roll) +
`trackrod_inboard y` steer target (side: left; the rack link keeps sides consistent).

### A5. Output & visualization

- CLI output columns: `left_<point>_x`, ..., metrics `left_camber_deg`, ...,
  axle metrics unprefixed.
- `get_visualization_links` returns links with `PointRef`s (viz plotting only ever
  indexes `state.positions`, so it is key-agnostic); both corners drawn.

## Phase B — pushrod / rocker / torsion bar / ARB

All new elements reduce to existing constraint types — **no new Jacobian codegen**.
A point rigidly attached to a body rotating about a fixed axis lies on a circle:
2 `DistanceConstraint`s to two fixed points on the axis; body rigidity between two
such points = 1 more `DistanceConstraint`.

### B1. New points (`PointID` additions)

Per corner (side-tagged): `ROCKER_AXIS_FRONT`, `ROCKER_AXIS_REAR` (chassis-fixed),
`ROCKER_DROPLINK` (free, on rocker), plus existing `PUSHROD_INBOARD` (free, on
rocker) and `PUSHROD_OUTBOARD` (free, upright-mounted). Shared (CENTER):
`ARB_AXIS_FRONT`, `ARB_AXIS_REAR` (chassis-fixed). Per side on the ARB:
`ARB_DROPLINK` (free, on the ARB arm; the droplink connects
`ROCKER_DROPLINK` ↔ `ARB_DROPLINK`).

Note: despite the FRONT/REAR naming (mirroring the wishbone convention), axis
points are just two distinct points defining the pivot line; direction of
positive rotation is first-point → second-point by right-hand rule.

### B2. Constraints per corner (12 new vars, 13 new residuals per side)

| element | constraints |
|---|---|
| pushrod outboard rigid to upright | 4 distances to UWB_OUT, LWB_OUT, AXLE_IN, AXLE_OUT |
| pushrod length | 1 distance PUSHROD_OUT ↔ PUSHROD_IN |
| pushrod inboard on rocker circle | 2 distances to ROCKER_AXIS_FRONT/REAR |
| rocker droplink on rocker circle | 2 distances to ROCKER_AXIS_FRONT/REAR |
| rocker rigidity | 1 distance PUSHROD_IN ↔ ROCKER_DROPLINK |
| droplink length | 1 distance ROCKER_DROPLINK ↔ ARB_DROPLINK |
| ARB arm end on ARB circle | 2 distances to ARB_AXIS_FRONT/REAR |

The ARB is modelled as **two independent arms** about a common axis (the bar's
torsional compliance is what an ARB *is*); its state is fully determined by wheel
positions, and the twist angle is reported as a metric. A rigid ARB would
over-constrain independent left/right wheel targets.

The torsion bar is coaxial with its rocker pivot: its twist equals the rocker's
rotation from design (inboard end assumed chassis-grounded).

### B3. Geometry validation

- Rocker axis parallel to XZ plane: `|y(ROCKER_AXIS_FRONT) - y(ROCKER_AXIS_REAR)|`
  ≤ tolerance, else load-time error.
- Axis points distinct; droplink/pushrod non-zero length; ARB axis distinct points.

### B4. New metrics (per side unless noted)

- `rocker_angle_deg`: signed rocker rotation from design about the rocker axis
  (right-hand rule, axis front → rear), measured via the `PUSHROD_INBOARD` radius
  vector projected into the plane ⟂ axis.
- `torsion_bar_twist_deg`: = rocker angle (kept as its own column for clarity).
- `arb_arm_angle_deg`: per-side ARB arm rotation from design about the ARB axis.
- `arb_twist_deg` (axle-level): left arm angle − right arm angle. Zero in pure
  heave for a symmetric car; non-zero in roll.

Motion/installation ratios are derivable from sweep output (d(angle)/d(wheel z));
not computed per-state.

### B5. YAML schema additions

```yaml
type: double_wishbone_axle
hardpoints:
  points:
    ...existing...
    pushrod_outboard: {..}
    pushrod_inboard: {..}
    rocker_axis_front: {..}
    rocker_axis_rear: {..}
    rocker_droplink: {..}
    arb_droplink: {..}
  center:                 # shared, not mirrored
    arb_axis_front: {..}
    arb_axis_rear: {..}
```

Rocker/ARB points are optional as a group: present → rocker constraint set and
metrics activate (validated all-or-nothing).

## Test plan (both phases)

Coordinate-system and sign-convention tests are first-class:

1. **Mirroring**: y-negation of points and direction normals; left/right corner
   states are exact XZ-plane reflections at design.
2. **Key generalisation**: `PointRef` ordering/stability; single-corner regression
   (existing suite + e2e reference files untouched).
3. **Axle equivalence**: axle solve with rack pinned ≡ two independent
   single-corner solves (positions match to solver tolerance) — the strongest
   correctness anchor.
4. **Symmetry**: pure heave on a mirrored axle → left/right metrics equal
   (camber L = camber R, toe L = toe R, roadwheel angles equal & opposite sign
   handled by convention), rack stays centred, roll centre on Y≈0.
5. **Steering**: rack +Y sweep → roadwheel angles change in the same direction on
   both wheels with the documented sign; total toe behaves per Ackermann geometry.
6. **Roll**: opposite wheel-height targets → roll-centre metrics finite and stable;
   camber signs opposite.
7. **Phase B signs**: constructed geometry where bump (wheel +Z) ⇒ rocker angle
   positive by right-hand rule; heave → `arb_twist_deg == 0`; roll → equal-and-
   opposite arm angles with documented `arb_twist` sign; droplink/pushrod lengths
   conserved across the sweep; axis-not-parallel-to-XZ rejected at load.
8. **Jacobian consistency**: analytic vs finite-difference Jacobian on the axle
   system (reusing the approach in `tests/test_jacobians.py`).

## Delivery stages

| stage | content | status |
|---|---|---|
| 0 | Survey, plan (this file) | done |
| 1 | Core key generalisation (`Side`, `PointRef`, annotations, `Constraint.remap`) | pending |
| 2 | Axle model + loader + sweep side-targets + metrics + writer/CLI/viz + tests | pending |
| 3 | Pushrod/rocker/torsion bar/ARB + validation + metrics + tests | pending |
| 4 | Docs (README/CHANGELOG), full verification, push | pending |

## Implementation notes (from survey)

- Tooling: `just test` (pytest+cov), `just lint` (ruff), `just type-check` (ty),
  `just check`. e2e golden files `tests/data/e2e/output.{csv,parquet}` must stay
  byte-identical (single-corner path untouched).
- `visualization/api.visualize_geometry` whitelists `TYPE_KEY` — the axle type
  must be added there.
- New constraint types would require SymPy Jacobian codegen
  (`tools/generate_jacobians.py`) kept in lockstep with residuals; the plan
  deliberately avoids this by expressing all Phase B elements with
  `DistanceConstraint`s.
- `PUSHROD_INBOARD`/`PUSHROD_OUTBOARD` already exist in `PointID` and are accepted
  by the loader but are kinematically inert today.

## Work log

- 2026-07-04: Surveyed codebase; confirmed ISO 8855 axes; identified `PointID`
  keying as the single-corner bottleneck; established that Phase B needs no new
  constraint types or Jacobian codegen. Wrote this plan.
