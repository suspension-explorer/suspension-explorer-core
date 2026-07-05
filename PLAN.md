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
`ARB_AXIS_A`, `ARB_AXIS_B` (chassis-fixed). Per side on the ARB:
`ARB_DROPLINK` (free, on the ARB arm; the droplink connects
`ROCKER_DROPLINK` ↔ `ARB_DROPLINK`).

Enum values (continuing after 23): `ROCKER_AXIS_FRONT = 24`,
`ROCKER_AXIS_REAR = 25`, `ROCKER_DROPLINK = 26`, `ARB_AXIS_A = 27`,
`ARB_AXIS_B = 28`, `ARB_DROPLINK = 29`.

Naming rationale: the rocker axis keeps FRONT/REAR because it must lie parallel
to the XZ plane (zero Y-extent), so it is naturally longitudinal. The ARB axis
uses A/B because it need not be longitudinal — it is typically transverse (along
Y). In both cases the two points just define the pivot line; the direction of
positive rotation is first-point → second-point (FRONT→REAR, A→B) by right-hand
rule.

### B2. Constraints per corner (12 new vars, 13 new residuals per side)

| element | constraints |
|---|---|
| pushrod outboard rigid to upright | 4 distances to UWB_OUT, LWB_OUT, AXLE_IN, AXLE_OUT |
| pushrod length | 1 distance PUSHROD_OUT ↔ PUSHROD_IN |
| pushrod inboard on rocker circle | 2 distances to ROCKER_AXIS_FRONT/REAR |
| rocker droplink on rocker circle | 2 distances to ROCKER_AXIS_FRONT/REAR |
| rocker rigidity | 1 distance PUSHROD_IN ↔ ROCKER_DROPLINK |
| droplink length | 1 distance ROCKER_DROPLINK ↔ ARB_DROPLINK |
| ARB arm end on ARB circle | 2 distances to ARB_AXIS_A/B |

The ARB is modelled as **two independent arms** about a common axis (the bar's
torsional compliance is what an ARB *is*); its state is fully determined by wheel
positions, and the twist angle is reported as a metric. A rigid ARB would
over-constrain independent left/right wheel targets.

The torsion bar is coaxial with its rocker pivot: its twist equals the rocker's
rotation from design (inboard end assumed chassis-grounded).

### B3. Geometry validation

- Rocker axis parallel to XZ plane: `|y(ROCKER_AXIS_FRONT) - y(ROCKER_AXIS_REAR)|`
  ≤ `EPS_GEOMETRIC`, else load-time error.
- Rocker group all-or-nothing ({PUSHROD_OUTBOARD, PUSHROD_INBOARD,
  ROCKER_AXIS_FRONT, ROCKER_AXIS_REAR}); ROCKER_DROPLINK requires that group.
- Rocker-axis points distinct; PUSHROD_INBOARD / ROCKER_DROPLINK off-axis
  (non-zero perpendicular distance).
- ARB group all-or-nothing across: center ARB_AXIS_A + ARB_AXIS_B, ARB_DROPLINK on
  both sides, ROCKER_DROPLINK on both sides. ARB axis points distinct; each
  ARB_DROPLINK off-axis. `arb_droplink` is an axle-only point — the corner class
  rejects it (single-corner YAML with `arb_droplink` fails as an unknown key).

### B4. New metrics (per side unless noted) — sign conventions as implemented

The signed-angle primitive is
`signed_angle_about_axis(p_design, p_current, axis_point, axis_direction)`
(`core/vector_utils/geometric.py`): projects both radius vectors into the plane ⟂
the axis and returns `atan2(d·(r0×r1), r0_perp·r1_perp)` (right-hand rule about
the axis direction).

- `rocker_angle_deg` (corner): raw signed angle of `PUSHROD_INBOARD` about the
  ROCKER_AXIS_FRONT → ROCKER_AXIS_REAR direction (design → current), **multiplied
  by `side_sign` (+1 left, −1 right)**. Rationale: reflection negates a signed
  angle about a mirrored axis, so the normalisation makes symmetric heave report
  EQUAL left/right rocker angles (and roll report equal-and-opposite). Emitted in
  `compute_metrics_for_state` when `suspension.has_rocker`, so it appears
  `left_`/`right_`-prefixed at axle level automatically.
- `torsion_bar_twist_deg` (corner): = `rocker_angle_deg` (the torsion bar is
  coaxial with the pivot and grounded at its far end); kept as its own column.
- `left_arb_arm_angle_deg` / `right_arb_arm_angle_deg` (axle): **raw** signed angle
  of that side's `ARB_DROPLINK` about the single shared ARB_AXIS_A → ARB_AXIS_B
  direction (design → current), with **no side normalisation** (both arms share
  one physical axis).
- `arb_twist_deg` (axle): `left_arb_arm_angle_deg − right_arb_arm_angle_deg` — the
  physical relative twist of the torsion element (RH rule about A→B). Symmetric
  heave → both raw arm angles equal → twist ≈ 0; roll → opposite → twist ≠ 0
  (left-wheel-up gives positive twist for the mirrored transverse-axis test ARB).

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
| 1 | Core key generalisation (`Side`, `PointRef`, annotations, `Constraint.remap`) | done |
| 2 | Axle model + loader + sweep side-targets + metrics + writer/CLI/viz + tests | done |
| 3 | Pushrod/rocker/torsion bar/ARB + validation + metrics + tests | done |
| 4 | Docs (README/CHANGELOG), full verification, push | done |

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

- 2026-07-05: Visual-verification rework of the rocker/ARB fixture per review:
  longer rocker levers (pushrod lever ~95 mm, droplink lever 90 mm directly
  inboard of the pivot so its tangent is pure Z), ARB moved BELOW the rocker
  axis (bar at z=300 vs rocker axis z=450), near-vertical ~170 mm droplinks,
  and wheel-up now demonstrably drives the droplinks down (-12.5 mm at +20 mm
  heave; asserted in TestAxleHeave). The ARB is drawn as a single series
  (left arm end -> bar end -> bar end -> right arm end), pairing each side with
  its nearer bar end at design. The roll ARB-twist sign flipped with the new
  layout (left-up => negative twist about the authored A->B (-Y) direction);
  the test's hand-derivation comment was updated accordingly.
- 2026-07-04: Stage 4 done. Updated README features (axle simulation, inboard
  actuation, axle metrics) and CHANGELOG (Unreleased section). Full-suite
  verification at each stage: 359 passed; the only 2 failures are pre-existing
  environmental mp4-animation e2e failures (no ffmpeg in the dev container,
  reproduced identically at the pre-change commit; CI installs without the viz
  extra and is unaffected). Ruff and `ty` clean. Design-condition renders of
  both the plain axle and the rocker/ARB axle visually verified. All stages
  pushed to `claude/suspension-axle-simulation-b71drh`.
- 2026-07-04: Stage 3 done. Added the F1-style inboard actuation (pushrod /
  rocker / torsion bar / inboard ARB), all expressed with existing
  `DistanceConstraint`s — no new constraint types or Jacobian codegen. New
  `PointID`s 24–29 (`ROCKER_AXIS_FRONT/REAR`, `ROCKER_DROPLINK`, `ARB_AXIS_A/B`,
  `ARB_DROPLINK`). Corner (`double_wishbone.py`): rocker group added to
  `OPTIONAL_POINTS`, `has_rocker`/`has_rocker_droplink`, all-or-nothing group
  validation with the rocker-axis-parallel-to-XZ and off-axis checks,
  instance-method `free_points()`/`output_points()` that include present rocker
  points, `_rocker_constraints` (pushrod-outboard rigid-to-upright ×4, pushrod
  length, pushrod-inboard rocker circle ×2, and droplink circle ×2 +
  rigidity ×1 when present), plus pushrod/rocker viz links. Axle (`axle.py`):
  optional `center:` block (`arb_axis_a/b`) and per-side `arb_droplink` parsed by
  the axle (popped before `parse_hardpoints` so the corner class stays clean and
  rejects `arb_droplink`), `has_arb`, all-or-nothing ARB validation,
  `initial_state`/`free_points`/`output_points` extended with the CENTER axis
  (fixed) and per-side ARB droplink (free), `_arb_constraints` (ARB-circle ×2 +
  droplink length ×1 per side), and ARB/droplink/axis viz links. Metrics:
  `signed_angle_about_axis` in `geometric.py`; `rocker_angle_deg` +
  `torsion_bar_twist_deg` (side-sign-normalised) appended in
  `compute_metrics_for_state` when `has_rocker`; raw `left/right_arb_arm_angle_deg`
  + `arb_twist_deg` appended in `compute_metrics_for_axle_state` when `has_arb`.
  `Suspension.output_points()` (base + overrides) added and wired into `cli.py`.
  New test data (`corner_rocker_geometry.yaml`, `axle_geometry_rocker.yaml`,
  `axle_rocker_sweep.yaml`) tuned so heave/roll/steer sweeps solve with residual
  ~0 (rocker droplink at the +Y circle extreme and ARB droplink at the +X arc
  extreme keep the droplink well-conditioned, avoiding a near-singular ARB arm).
  New `tests/test_rocker_arb.py` (22 tests): signed-angle unit tests, load
  validation, corner conservation/monotonicity + hand-verified bump sign, axle
  heave symmetry, axle roll antisymmetry + twist sign + droplink conservation,
  analytic-vs-FD Jacobian, CLI column smoke, and no-rocker regression. Full suite
  359 passed (2 pre-existing mp4-animation env failures only), ruff + ty green.
- 2026-07-04: Surveyed codebase; confirmed ISO 8855 axes; identified `PointID`
  keying as the single-corner bottleneck; established that Phase B needs no new
  constraint types or Jacobian codegen. Wrote this plan.
- 2026-07-04: Stage 2 done. Added `suspensions/axle.py`
  (`DoubleWishboneAxleSuspension`): composes two `DoubleWishboneSuspension`
  corners, side-tags state/free-points/constraints/derived-points via
  `PointRef`, couples them with a rigid-rack `DistanceConstraint`, and reuses
  the corner machinery through a value-type-agnostic `_SideView`. Registered the
  `double_wishbone_axle` type. Added a `Suspension.from_yaml_data` hook (base =
  existing `load_suspension`; axle parses the mirror/explicit side schema and
  mirrors camber-shim config for the RIGHT corner) and routed `load_geometry`
  through it. Added `Suspension.compute_state_metrics` (branch-free CLI
  dispatch) with `compute_metrics_for_axle_state` (per-side `left_`/`right_`
  blocks plus `roll_center_{y,z}_mm`, `total_toe_deg`, `track_mm`,
  `rack_displacement_mm`). Sweep loader gained `TargetSpec.side` and
  suspension-aware key resolution (`Suspension.resolve_target_key`). Generalised
  the visualizer over multiple `WheelAnchors` so the axle draws both wheels;
  added the axle to the `visualize_geometry` whitelist with per-side ground
  checks; kept matplotlib strictly optional. New tests in `tests/test_axle.py`
  (loading/validation, single-corner equivalence anchor, heave symmetry,
  steering monotonicity, roll, analytic-vs-FD Jacobian, CLI smoke) plus
  `tests/data/axle_geometry{,_explicit}.yaml` and `axle_sweep.yaml`. Toe/
  roadwheel sign convention: each corner's `roadwheel_angle_deg` (== toe) is
  centreline-relative, so pure heave gives equal left/right values while
  steering moves them with opposite sign (each monotonic in rack); `total_toe`
  is their sum. Full suite (336 passed, 3 skipped), ruff, and `ty` green; e2e
  goldens and single-corner path untouched; no new format drift.
- 2026-07-04: Stage 1 done. Added `core/point_ref.py` (`Side`, `PointRef`,
  `PointKey` alias); loosened `PointID` annotations to `PointKey` in
  `state.py`, `constraints.py`, `solver.py`, `points/derived/manager.py`,
  `core/types.py`, `core/dual.py`; added `Constraint.remap` (via per-class
  `_POINT_ATTRS`) for re-keying constraints into side namespaces; added
  `tests/test_point_ref.py`. No runtime/behavioural change; full suite, ruff,
  and `ty` all green (e2e golden files untouched).
