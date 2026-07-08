# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- High-level front-end API so that adapters (API servers, notebooks) stay transport-thin and CLI users see identical data:
  - `kinematics.analysis.analyze_sweep(suspension, sweep_config)` returns the complete `SweepAnalysis` -- per-step frames (name-keyed display positions, full metric rows, solver info), metric display metadata, swept-parameter descriptors, the solved "setup" reference condition, advisory diagnostics, and the display topology. `initial_pose(suspension)` returns the static preview pose.
  - `kinematics.main.compute_sweep_metrics(suspension, sweep_config, states)` is the single metrics entry point for sweep consumers: it orchestrates the solution-manifold tangents internally (best-effort) so callers get derivative metrics without knowing about tangents or AD. The CLI now uses it.
  - `kinematics.metrics.metadata` is the single source of truth for display labels/units of every exported column (catalog, rocker/ARB, axle, rate columns), with `left_`/`right_` prefix resolution; a test guards that every emitted column resolves.
  - `kinematics.visualization.display` owns the interactive display topology: name-keyed links, the rocker fan replaced by axis + perpendicular lever arms (with synthetic `*_AXIS_FOOT` positions), display point sets extended with link-referenced points, wheel dimensions and anchors.
  - `SweepFile.n_steps` reports the expanded step count of a sweep spec.
- Analytical motion ratios and kinematic rate metrics, computed exactly via the implicit function theorem plus forward-mode dual-number propagation -- no finite differencing across sweep steps.
  - New `sensitivity` module: at each converged sweep step the analytical residual Jacobian is reused to solve for the solution-manifold tangent d(position)/d(target) per sweep target; derived-point velocities follow from one dual-number Jacobian-vector-product pass. Constraints whose scalar residual is a norm (e.g. point-on-line) are pinned with equivalent smooth rows so the tangent system stays full rank at the solution.
  - Dual-number layer extended with `np.cross`, `sqrt`, `atan2`, and `degrees`, plus `seed_positions_with_tangent` for directional (tangent-field) seeding.
  - Dual-safe metric kernels (`metrics/kernels.py`) evaluate identically on floats and dual numbers; rate metrics (`metrics/rates.py`) are their exact derivatives along the tangents.
  - Corner rate columns (per mm of that corner's upward wheel travel): `camber_gain_deg_per_mm`, `bump_steer_deg_per_mm`, `caster_gain_deg_per_mm`, `kpi_gain_deg_per_mm`, `half_track_rate_mm_per_mm`, `wheel_recession_rate_mm_per_mm`, `damper_motion_ratio` (installation ratio, d(damper compression)/d(wheel bump); wheel rate = spring rate * MR^2), `rocker_motion_ratio_deg_per_mm`, `torsion_bar_motion_ratio_deg_per_mm`; plus rack-driven `toe_vs_rack_deg_per_mm` and `camber_vs_rack_deg_per_mm`.
  - Axle modal rate columns built from linear combinations of the two bump tangents: `left_`/`right_toe_vs_roll_deg_per_deg`, `left_`/`right_camber_vs_roll_deg_per_deg`, `left_`/`right_toe_vs_heave_deg_per_mm`, `left_`/`right_camber_vs_heave_deg_per_mm`, and `arb_twist_vs_roll_deg_per_deg`. Roll is the wheel-pair rotation relative to the chassis, right-hand rule about +X (positive = left wheel in bump).
- Optional spring/damper (coilover) element on the double-wishbone corner via the previously inert `strut_top`/`strut_bottom` hardpoints (all-or-nothing group). Without a rocker the damper body mounts on the lower wishbone (outboard coilover); with a rocker it mounts on the rocker (inboard spring). Exports `damper_length_mm` and drives `damper_motion_ratio`.
- New per-state corner metrics: `wheel_travel_mm`, `half_track_change_mm`, `wheel_recession_mm`, `damper_length_mm`, `svsa_angle_deg`, and anti-pitch geometry percentages `anti_dive_pct`, `anti_lift_pct`, `anti_squat_pct` (side-view support geometry combined with CG height, wheelbase, brake bias, and driven axle; `None` when the required configuration is absent).
- New axle-level per-state metrics: `heave_mm`, `roll_deg`, `ride_height_change_mm`, and `ackermann_pct` (relative to the ideal cot-difference Ackermann condition, `None` below a 0.5 deg parallel-steer cutoff).
- New optional `SuspensionConfig` fields: `axle_position` (front/rear), `front_brake_bias` (0..1), `driven_axle` (front/rear).
- Full-axle (two-corner) simulation: new `double_wishbone_axle` geometry type composing left and right double-wishbone corners into a single coupled constraint system, linked by a rigid steering rack (fixed distance between the two inboard trackrod points).
  - Side-qualified point keys: `Side` (LEFT/RIGHT/CENTER) and `PointRef` (`core/point_ref.py`); core state/constraint/solver/derived-point machinery generalised over the `PointKey` type. Single-corner models are unchanged.
  - Axle geometry YAML supports mirror mode (one side given, the other generated by `y -> -y`, camber shim config mirrored) and explicit mode (both sides given).
  - Sweep targets accept an optional `side: left|right` field for axle geometries.
  - Per-side corner metrics exported with `left_`/`right_` prefixes, plus axle-level metrics: `roll_center_y_mm`/`roll_center_z_mm` (front-view intersection of the two contact-patch-to-FVIC lines), `total_toe_deg`, `track_mm`, and `rack_displacement_mm`.
  - Design-condition plots and sweep animations render both corners and the steering rack.
- F1-style inboard actuation for the double-wishbone corner and axle: pushrod to an inboard rocker rotating about a chassis-fixed axis (validated parallel to the XZ plane), a torsion bar coaxial with each rocker pivot, and an inboard anti-roll bar actuated via droplinks from the rockers.
  - New hardpoints: `rocker_axis_front`/`rocker_axis_rear`, `rocker_droplink`, `arb_axis_a`/`arb_axis_b` (shared `center:` block), `arb_droplink`; the previously inert `pushrod_inboard`/`pushrod_outboard` points are now kinematically active when the rocker group is present.
  - All new elements are expressed with existing distance constraints (points rigid to a body rotating about a fixed axis are constrained by distances to two fixed axis points), so no new Jacobian machinery was needed.
  - New metrics: `rocker_angle_deg` and `torsion_bar_twist_deg` per corner (signed by right-hand rule about the rocker axis, side-normalised so symmetric heave reports equal values on both sides), `left_`/`right_arb_arm_angle_deg` (raw angles about the shared ARB axis) and `arb_twist_deg = left - right` (zero in heave, the physical bar twist in roll).
- `Constraint.remap` for re-keying constraints into side-qualified namespaces; `Suspension.from_yaml_data`, `compute_state_metrics`, `resolve_target_key`, and instance-level `output_points()` extension hooks.

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


