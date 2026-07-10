# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Curated public API: `kinematics/__init__.py` is now the single import surface for consumers (CLI, API servers, notebooks). It re-exports `analyze_sweep`, `initial_pose`, `build_suspension`, `build_sweep_config`, `parse_geometry_spec`, `compute_sweep_metrics`, `SweepSpec`, `GeometrySpec`, `MetricSpec`, `metric_display_for_keys`, `motion_ratio_specs`, and friends. Deeper module paths are internal and free to change; importing `kinematics` never pulls in matplotlib.
- New `kinematics.schema` package holding the transport-agnostic, validated input models and coercion helpers (no file IO):
  - `schema/coercion.py` -- case-insensitive enum/point/direction coercers and the Pydantic annotated types (`CIPointID`, `CIAxis`, `CISide`, `CIUnits`, `CITargetPositionMode`, `PydanticPoint3`, `PydanticDirection3`).
  - `schema/sweep.py` -- `SweepSpec` plus `build_sweep_config()` (and `TargetSpec`/`DirectionSpec`).
  - `schema/geometry.py` -- `GeometrySpec`, a discriminated union on `type` over `DoubleWishboneGeometrySpec` and `DoubleWishboneAxleGeometrySpec`, with `parse_geometry_spec()`. The axle hardpoints block is a real model (`AxleHardpointsSpec`) that validates mirror vs explicit mode.
  - `schema/config.py` -- `SuspensionConfig`, `WheelConfig`, `TireConfig`, `CamberShimConfig` (moved here from `suspensions/config/settings.py`).
- Declarative metric registry (`metrics/registry.py`): every emittable column family is declared exactly once as a `MetricSpec(key, label, unit, kind, scope, component, motion_ratio)`. Emitting code references the spec constants (`row[SPEC.key] = ...`) instead of repeating string literals, `metrics/metadata.py` derives all display metadata from the same table, and `motion_ratio_specs()` lists the damper / rocker / torsion-bar / ARB installation ratios with display names ready for front-ends.
- High-level front-end API so that adapters (API servers, notebooks) stay transport-thin and CLI users see identical data:
  - `kinematics.analysis.analyze_sweep(suspension, sweep_config)` returns the complete `SweepAnalysis` -- per-step frames (name-keyed display positions, full metric rows, solver info), metric display metadata, swept-parameter descriptors, the solved "setup" reference condition, advisory diagnostics, and the display topology. `initial_pose(suspension)` returns the static preview pose.
  - `kinematics.main.compute_sweep_metrics(suspension, sweep_config, states)` is the single metrics entry point for sweep consumers: it orchestrates the solution-manifold tangents internally (best-effort) so callers get derivative metrics without knowing about tangents or AD. The CLI now uses it.
  - `kinematics.metrics.metadata` resolves every exported column name to display metadata (label, unit, kind, component, motion-ratio membership, location), derived from the metric registry (`metrics/registry.py`); it maps both location-less columns and the axle's `_left`/`_right` suffixed variants (location folded into the label, e.g. "Left Camber Gain"), and a test guards that every emitted column resolves.
  - `kinematics.visualization.display` owns the interactive display topology: name-keyed links, the rocker fan replaced by axis + perpendicular lever arms (with synthetic `<pickup>_AXIS_FOOT` positions), display point sets extended with link-referenced points, wheel dimensions and anchors.
  - `SweepSpec.n_steps` reports the expanded step count of a sweep spec.
- Analytical motion ratios and kinematic rate metrics, computed exactly via the implicit function theorem plus forward-mode dual-number propagation -- no finite differencing across sweep steps.
  - New `sensitivity` module: at each converged sweep step the analytical residual Jacobian is reused to solve for the solution-manifold tangent d(position)/d(target) per sweep target; derived-point velocities follow from one dual-number Jacobian-vector-product pass. Constraints whose scalar residual is a norm (e.g. point-on-line) are pinned with equivalent smooth rows so the tangent system stays full rank at the solution.
  - Dual-number layer extended with `np.cross`, `sqrt`, `atan2`, and `degrees`, plus `seed_positions_with_tangent` for directional (tangent-field) seeding.
  - Dual-safe metric kernels (`metrics/kernels.py`) evaluate identically on floats and dual numbers; rate metrics (`metrics/rates.py`) are their exact derivatives along the tangents.
  - Corner rate columns (per mm of that corner's upward wheel travel): `camber_gain_deg_per_mm`, `roadwheel_angle_vs_bump_deg_per_mm`, `caster_gain_deg_per_mm`, `kpi_gain_deg_per_mm`, `half_track_rate_mm_per_mm`, `wheel_recession_rate_mm_per_mm`, `damper_motion_ratio` (installation ratio, d(damper compression)/d(wheel bump); wheel rate = spring rate * MR^2), `rocker_motion_ratio_deg_per_mm`, `torsion_bar_motion_ratio_deg_per_mm`, and per-side `arb_motion_ratio_deg_per_mm`; plus rack-driven `roadwheel_angle_vs_rack_deg_per_mm` and `camber_vs_rack_deg_per_mm`.
  - Axle modal rate columns built from linear combinations of the two bump tangents: per-corner `roadwheel_angle_vs_roll_deg_per_deg`, `camber_vs_roll_deg_per_deg`, `roadwheel_angle_vs_heave_deg_per_mm`, and `camber_vs_heave_deg_per_mm` (rendered `_left`/`_right` suffixed at flat export boundaries), plus the axle-level `arb_twist_vs_roll_deg_per_deg`. Roll is the wheel-pair rotation relative to the chassis, right-hand rule about +X (positive = left wheel in bump).
- Optional spring/damper (coilover) element on the double-wishbone corner via the previously inert `strut_top`/`strut_bottom` hardpoints (all-or-nothing group). Without a rocker the damper body mounts on the lower wishbone (outboard coilover); with a rocker it mounts on the rocker (inboard spring). Exports `damper_length_mm` and drives `damper_motion_ratio`.
- New per-state corner metrics: `wheel_travel_mm`, `half_track_change_mm`, `wheel_recession_mm`, `damper_length_mm`, `svsa_angle_deg`, and anti-pitch geometry percentages `anti_dive_pct`, `anti_lift_pct`, `anti_squat_pct` (side-view support geometry combined with CG height, wheelbase, brake bias, and driven axle; `None` when the required configuration is absent).
- New axle-level per-state metrics: `heave_mm`, `roll_deg`, `ride_height_change_mm`, and `ackermann_pct` (relative to the ideal cot-difference Ackermann condition, `None` below a 0.5 deg parallel-steer cutoff).
- New optional `SuspensionConfig` fields: `axle_position` (front/rear), `front_brake_bias` (0..1), `driven_axle` (front/rear).
- Full-axle (two-corner) simulation: new `double_wishbone_axle` geometry type composing left and right double-wishbone corners into a single coupled constraint system, linked by a rigid steering rack (fixed distance between the two inboard trackrod points).
  - Side-qualified point keys: `Side` (LEFT/RIGHT/CENTER) and `PointRef` (`core/point_ref.py`); core state/constraint/solver/derived-point machinery generalised over the `PointKey` type. Single-corner models are unchanged.
  - Axle geometry YAML supports mirror mode (one side given, the other generated by `y -> -y`, camber shim config mirrored) and explicit mode (both sides given).
  - Sweep targets accept an optional `side: left|right` field for axle geometries.
  - Per-corner metrics carried structurally per location and exported with `_left`/`_right` suffixes in flat outputs, plus axle-level metrics: `roll_center_y_mm`/`roll_center_z_mm` (front-view intersection of the two contact-patch-to-FVIC lines), `total_roadwheel_angle_deg`, `track_mm`, and `rack_displacement_mm`.
  - Design-condition plots and sweep animations render both corners and the steering rack.
- F1-style inboard actuation for the double-wishbone corner and axle: pushrod to an inboard rocker rotating about a chassis-fixed axis (validated parallel to the XZ plane), a torsion bar coaxial with each rocker pivot, and an inboard anti-roll bar actuated via droplinks from the rockers.
  - New hardpoints: `rocker_axis_front`/`rocker_axis_rear`, `droplink_rocker`, `arb_axis_a`/`arb_axis_b` (shared `center:` block), `droplink_arb`; the previously inert `pushrod_inboard`/`pushrod_outboard` points are now kinematically active when the rocker group is present.
  - All new elements are expressed with existing distance constraints (points rigid to a body rotating about a fixed axis are constrained by distances to two fixed axis points), so no new Jacobian machinery was needed.
  - New metrics: `rocker_angle_deg` and `torsion_bar_twist_deg` per corner (signed by right-hand rule about the rocker axis, side-normalised so symmetric heave reports equal values on both sides), per-corner `arb_arm_angle_deg` (raw angles about the shared ARB axis) and the axle-level `arb_twist_deg = left - right` (zero in heave, the physical bar twist in roll).
- `Constraint.remap` for re-keying constraints into side-qualified namespaces; `compute_state_metrics`, `resolve_target_key`, and instance-level `output_points()` extension hooks.
- Structured metric locations: locations ("left" / "right") are modelled structurally end-to-end and only rendered into column names at export boundaries.
  - `metrics/registry.py` owns the location vocabulary (`Location`, `LOCATIONS`) and the single flat renderer `flat_key(spec_key, location)` (suffix form: `camber_deg_left`) with its inverse `split_flat_key`. The vocabulary extends to a future full-car model (e.g. `front_left`) without renaming any metric.
  - `AxleMetricRows` (frozen): `compute_metrics_for_axle_state` returns one axle-level row plus one location-keyed row per corner; per-corner rates and ARB arm angles live in the corner rows with location-less keys. `flatten_metric_rows(metrics, corner_metrics)` / `AxleMetricRows.flat_row()` produce the canonical flat row -- used only by the CLI results writer, since result-file columns are the one truly flat surface. JSON adapters serialize the hierarchy as-is.
  - `AnalyzedFrame` and `ReferenceCondition` gained `corner_metrics: dict[str, MetricRow]` alongside the model-level `metrics` row; `SweepAnalysis` carries base-key lists (`metric_keys` for the model-level row, `corner_metric_keys` for the per-corner rows) plus the `locations` present, with `metric_display` covering every base key.
  - `MetricDisplay` gained `kind`, `component`, `motion_ratio`, and `location`, so front-ends can group metrics structurally instead of parsing key strings.
- Performance benchmark scaffold (baseline only): `tests/benchmarks/test_bench_sweep.py` benchmarks `solve_sweep` and `analyze_sweep` on the full rocker/ARB axle example under `pytest-benchmark` (excluded from the default suite; run with `just bench`), and `tools/profile_sweep.py` profiles the same sweep under cProfile (top-30 cumulative).

### Changed
- Strict package layering: `core <- schema <- suspensions <- io/cli`. Programmatic consumers use `kinematics.schema` and the top-level `kinematics` API directly; only `kinematics.io` touches the filesystem.
- `suspensions/build.py:build_suspension(spec)` is now the sole construction path: a validated `GeometrySpec` in, a live `Suspension` out.
- `kinematics.io` reduced to `loaders.py` (`load_geometry`, `load_sweep`) plus `results_writer.py`; the loaders are thin wrappers that read YAML, hand it to the schema layer, and build live objects.
- Torsion-bar columns (`torsion_bar_twist_deg`, `torsion_bar_motion_ratio_deg_per_mm`) are kept, but now declared explicitly in the registry as coaxial aliases of the rocker columns (`rocker_angle_deg` / `rocker_motion_ratio_deg_per_mm`), so spring-rate work can reference the bar by name.

### Removed
- `suspensions/config/settings.py` -- the config models moved to `schema/config.py`.
- `Suspension.from_yaml_data` / `from_yaml`, the `ALIASES` class attribute, and `matches_type` -- superseded by `build_suspension(spec)` and the discriminated `GeometrySpec`.

### Breaking changes
- Canonical naming, with zero backwards compatibility. "Toe" is gone everywhere in favour of "roadwheel angle":
  - `bump_steer_deg_per_mm` -> `roadwheel_angle_vs_bump_deg_per_mm`
  - `toe_vs_rack_deg_per_mm` -> `roadwheel_angle_vs_rack_deg_per_mm`
  - `left_`/`right_toe_vs_roll_deg_per_deg` -> `..._roadwheel_angle_vs_roll_deg_per_deg`
  - `left_`/`right_toe_vs_heave_deg_per_mm` -> `..._roadwheel_angle_vs_heave_deg_per_mm`
  - `total_toe_deg` -> `total_roadwheel_angle_deg`
- Point keys are now droplink-centric: `PointID.ROCKER_DROPLINK` -> `DROPLINK_ROCKER` and `PointID.ARB_DROPLINK` -> `DROPLINK_ARB` (YAML author keys `rocker_droplink`/`arb_droplink` -> `droplink_rocker`/`droplink_arb`; emitted axle position keys change accordingly, e.g. `LEFT_DROPLINK_ARB`). "Pickup" replaces "foot" for physical attachments; the synthetic visualization key `<pickup>_AXIS_FOOT` stays, as it names perpendicular-foot geometry rather than a hardpoint.
- Deleted IO modules: `io/validation.py`, `io/geometry_loader.py`, `io/sweep_loader.py` -- replaced by `schema/coercion.py`, `schema/geometry.py`, and `schema/sweep.py` respectively.
- `SweepFile` renamed to `SweepSpec` (attributes such as `SweepSpec.n_steps` unchanged otherwise).
- Suspension type aliases removed: `double_wishbone_front` and `double_wishbone_rear` no longer resolve; use the canonical `double_wishbone` (or `double_wishbone_axle`) type key with the `axle_position` config field to distinguish front/rear.
- Axle metric locations are no longer mangled into key strings anywhere programmatic. Result files (the one flat surface) moved from `left_`/`right_` prefixes to `_left`/`_right` suffixes: `left_camber_deg` -> `camber_deg_left`, and so on for every corner-scope column; axle-level and corner-model columns are unchanged. Every other consumer gets the hierarchy itself: `AnalyzedFrame`/`ReferenceCondition` carry the model-level `metrics` row plus `corner_metrics` (location -> base-keyed row), and API adapters serialize that structure directly rather than flattening.

## [0.3.0] - 2026-04-09

### Added
- Split-body camber shim assembly solver (`suspensions/config/shims.py`): solves for the outboard camber shim configuration using a least-squares formulation. The upper ball joint position, camber block rotation, and upright body rotation are solved simultaneously to satisfy wishbone arc constraints, shim face closure, normal alignment, and trackrod length preservation.

### Changed
- Relaxed `Vec3` type alias from `NDArray[np.float64]` to `NDArray[np.floating[Any]]` so that numpy arithmetic results satisfy the type checker without wrapping. `make_vec3` is retained at system boundaries (I/O, config loading, solver output extraction, dual-number passthrough) but removed from internal arithmetic call sites where it served only as type-checker appeasement.
- Adopted ISO/SAE wheel offset (ET) convention in `get_wheel_center`.
- Positive `wheel.offset` now places the wheel centerline inboard of the hub face (reduced track for larger positive ET).
- Updated wheel offset configuration docs to explicitly describe ET sign convention.
- Updated derived-point expectations to match ISO/SAE offset behavior for wheel center, wheel inboard, and wheel outboard.
- `ResidualComputer` now uses a fixed-size residual vector and Jacobian matrix, removing per-step trimming. The target count is validated once at each evaluation rather than allowing variable-length slicing.
- `ResidualComputer` internals are no longer private: `_n_vars` → `n_vars`, `_jac_buffer` → `jac_buffer`, `_jac_plans` → `jac_plan`, `_validate_target_count` → `validate_target_count`.
- Renamed Jacobian "scatter" operations to "distribute" in `ResidualComputer._build_jac_plan`.
- Moved underdetermined-system check out of the per-step loop in `solve_suspension_sweep` (both `n_vars` and `m_res` are constant across a sweep).
- Simplified `DoubleWishboneSuspension._apply_camber_shim` docstring.
- Default corner-metric exports now use `roadwheel_angle_deg` as the canonical steer column and no longer export duplicate `toe_deg` or placeholder anti-dive / anti-squat metrics.

### Added
- `ResidualComputer.validate_target_count` enforces that every evaluation receives the same number of targets configured at init time.
- Test for Jacobian shape consistency (`test_residual_computer_rejects_target_count_changes`).
- Front-view (Y-Z) comparison plot in `visualize_camber_shim.py` overlaying design and setup suspensions with distinct colours.
- Direct sign and known-value tests for `camber_deg`, `caster_deg`, and `roadwheel_angle_deg`, plus catalog coverage for the trusted corner-metric export set.
- Kingpin inclination metric (`kpi_deg`): steering axis angle in the front-view (YZ) plane.
- Scrub radius metric (`scrub_radius_mm`): lateral offset from steering axis ground intersection to contact patch centre.
- Mechanical trail metric (`mechanical_trail_mm`): longitudinal offset from steering axis ground intersection to contact patch centre.

### Removed
- `WHEEL_CENTER_ON_GROUND` point and `get_wheel_center_on_ground` derived point function. The Z=0 ground plane assumption was incorrect in a chassis-fixed frame; ground-plane intersections now use the contact patch Z via `MetricContext.ground_z`.

### Fixed
- Scrub radius now projects along the wheel axle direction in the ground plane instead of the global Y axis, giving correct values when the wheel is steered or cambered.
- Scrub radius and mechanical trail now intersect the steering axis at the contact patch Z rather than Z=0, giving correct values through bump travel.
- Clarified `get_contact_patch_center` docstring as the lowest point on an ideal tire circle in the wheel center plane.
- Dashboard plots now show KPI, mechanical trail, and scrub radius instead of swing arm lengths and FVIC height. Camber plot Y-axis tuned to [-2.5, -1.5] degrees.


