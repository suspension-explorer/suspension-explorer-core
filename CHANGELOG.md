# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Generic point references support ordinary corner points and side-qualified axle points throughout the constraint, state, solver, and derived-point systems.
- Validated geometry schemas and a shared registry select explicit double-wishbone corner, coilover, pushrod-rocker, axle, and shared anti-roll-bar topologies.
- Declarative derivative metrics use analytical solution-manifold tangents and forward-mode automatic differentiation for arbitrary scalar responses and drivers.
- Advisory sweep diagnostics report convergence, residual acceptance, branch continuity, derivative availability, rocker and anti-roll-bar chirality, and transmission margin.
- Coupled axle models solve left and right corners together and support either mirrored or independently authored geometry.
- The public `analyze_sweep()` and `initial_pose()` APIs return structured positions, metrics, locations, metadata, display topology, diagnostics, references, and solved frames.

### Changed

- Derived-point target Jacobians now evaluate only the target's transitive dependency chain and seed only relevant free points, substantially reducing solve time.
- Geometry parsing, validation, and construction now pass through `kinematics.schema` and the suspension registry; filesystem access remains in `kinematics.io`.
- Metric identities are lowercase, unit-free `snake_case`. Units use typed metadata and are written in CSV metadata or Parquet field metadata.
- Corner locations remain structural in the analysis API and are rendered as `_left` and `_right` suffixes only in flat result files.
- Steering metrics use `roadwheel_angle`; the concrete steering input is `trackrod_inboard`, and wheel-center longitudinal motion is expressed directly as `deriv_wheel_center_x_wrt_hub_z`.
- Half-track is exported as the absolute `half_track` state metric rather than a design-condition delta.

### Breaking changes

- Removed suspension type aliases `double_wishbone_front` and `double_wishbone_rear`; use an explicit canonical type and configuration.
- Removed legacy geometry construction and loader paths in favor of validated schemas, `build_suspension()`, `load_geometry()`, and `load_sweep()`.
- Renamed `SweepFile` to `SweepSpec`.
- Removed units from metric keys and changed flat axle corner columns from side prefixes to side suffixes, for example `left_camber_deg` to `camber_left`.

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
- Front-view (Y-Z) comparison plot in `visualize_camber_shim.py` overlaying design and setup suspensions with distinct colors.
- Direct sign and known-value tests for `camber_deg`, `caster_deg`, and `roadwheel_angle_deg`, plus catalog coverage for the trusted corner-metric export set.
- Kingpin inclination metric (`kpi_deg`): steering axis angle in the front-view (YZ) plane.
- Scrub radius metric (`scrub_radius_mm`): lateral offset from steering axis ground intersection to contact patch center.
- Mechanical trail metric (`mechanical_trail_mm`): longitudinal offset from steering axis ground intersection to contact patch center.

### Removed
- `WHEEL_CENTER_ON_GROUND` point and `get_wheel_center_on_ground` derived point function. The Z=0 ground plane assumption was incorrect in a chassis-fixed frame; ground-plane intersections now use the contact patch Z via `MetricContext.ground_z`.

### Fixed
- Scrub radius now projects along the wheel axle direction in the ground plane instead of the global Y axis, giving correct values when the wheel is steered or cambered.
- Scrub radius and mechanical trail now intersect the steering axis at the contact patch Z rather than Z=0, giving correct values through bump travel.
- Clarified `get_contact_patch_center` docstring as the lowest point on an ideal tire circle in the wheel center plane.
- Dashboard plots now show KPI, mechanical trail, and scrub radius instead of swing arm lengths and FVIC height. Camber plot Y-axis tuned to [-2.5, -1.5] degrees.
