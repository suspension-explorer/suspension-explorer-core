"""
Metric registry.

The single declarative table of every metric column the package can emit.
Each column family is described by a :class:`MetricSpec`; emitting code
references these specs (``row[SPEC.key] = ...``) instead of repeating string
literals, and display metadata (kinematics.metrics.metadata) is derived from
this table rather than mirrored by hand.

Scope semantics:

- ``scope="corner"``: emitted un-prefixed by corner models, and ``left_`` /
  ``right_`` prefixed by the axle model (one column per side).
- ``scope="axle"``: emitted un-prefixed, once per axle state.

Kind semantics:

- ``kind="state"``: a per-state quantity (an angle, a length, a position).
- ``kind="rate"``: a first derivative computed from solution-manifold
  tangents (motion ratios, gains, per-driver sensitivities).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kinematics.metrics.catalog import get_default_corner_metrics


@dataclass(frozen=True)
class MetricSpec:
    """
    Declarative description of one exported metric column family.

    Attributes:
        key: Stable base column key, un-prefixed (e.g. "camber_gain_deg_per_mm").
        label: Human-readable display name (e.g. "Camber Gain").
        unit: Physical unit symbol ("deg", "mm", "deg/mm", "mm/mm", "%", "-").
        kind: "state" for per-state quantities, "rate" for tangent-derived
            derivatives.
        scope: "corner" for per-corner columns (side-prefixed on axles),
            "axle" for axle-level columns.
        component: Optional physical component this metric belongs to
            ("damper", "rocker", "arb", ...). Used for grouping in front-ends.
        motion_ratio: True for installation-ratio metrics quoted relative to
            hub (wheel center) vertical travel.
    """

    key: str
    label: str
    unit: str
    kind: Literal["state", "rate"]
    scope: Literal["corner", "axle"]
    component: str | None = None
    motion_ratio: bool = False


# ----------------------------------------------------------------------
# Conditional corner state columns (emitted when the geometry has the
# relevant group; the unconditional corner state columns live in the
# catalog and are folded in by all_metric_specs()).
# ----------------------------------------------------------------------

ROCKER_ANGLE = MetricSpec(
    "rocker_angle_deg", "Rocker Angle", "deg", "state", "corner", component="rocker"
)
# The torsion bar (when fitted) is coaxial with the rocker pivot and grounded
# at its far end, so its twist IS the rocker rotation; the torsion-bar columns
# are declared aliases of the rocker columns, kept as their own keys so
# spring-rate work can reference the bar by name.
TORSION_BAR_TWIST = MetricSpec(
    "torsion_bar_twist_deg",
    "Torsion Bar Twist",
    "deg",
    "state",
    "corner",
    component="torsion_bar",
)
ARB_ARM_ANGLE = MetricSpec(
    "arb_arm_angle_deg", "ARB Arm Angle", "deg", "state", "corner", component="arb"
)

# ----------------------------------------------------------------------
# Corner rate columns (per-driver derivatives; see kinematics.metrics.rates).
# Motion ratios are quoted relative to hub (wheel center) vertical travel:
# per +1 mm of upward wheel-center Z motion of the same corner.
# ----------------------------------------------------------------------

CAMBER_GAIN = MetricSpec(
    "camber_gain_deg_per_mm", "Camber Gain", "deg/mm", "rate", "corner"
)
ROADWHEEL_ANGLE_VS_BUMP = MetricSpec(
    "roadwheel_angle_vs_bump_deg_per_mm",
    "Roadwheel Angle vs Bump",
    "deg/mm",
    "rate",
    "corner",
)
CASTER_GAIN = MetricSpec(
    "caster_gain_deg_per_mm", "Caster Gain", "deg/mm", "rate", "corner"
)
KPI_GAIN = MetricSpec("kpi_gain_deg_per_mm", "KPI Gain", "deg/mm", "rate", "corner")
HALF_TRACK_RATE = MetricSpec(
    "half_track_rate_mm_per_mm", "Half Track Rate", "mm/mm", "rate", "corner"
)
WHEEL_RECESSION_RATE = MetricSpec(
    "wheel_recession_rate_mm_per_mm",
    "Wheel Recession Rate",
    "mm/mm",
    "rate",
    "corner",
)
DAMPER_MOTION_RATIO = MetricSpec(
    "damper_motion_ratio",
    "Damper Motion Ratio",
    "mm/mm",
    "rate",
    "corner",
    component="damper",
    motion_ratio=True,
)
ROCKER_MOTION_RATIO = MetricSpec(
    "rocker_motion_ratio_deg_per_mm",
    "Rocker Motion Ratio",
    "deg/mm",
    "rate",
    "corner",
    component="rocker",
    motion_ratio=True,
)
# Coaxial alias of ROCKER_MOTION_RATIO; see TORSION_BAR_TWIST.
TORSION_BAR_MOTION_RATIO = MetricSpec(
    "torsion_bar_motion_ratio_deg_per_mm",
    "Torsion Bar Motion Ratio",
    "deg/mm",
    "rate",
    "corner",
    component="torsion_bar",
    motion_ratio=True,
)
ARB_MOTION_RATIO = MetricSpec(
    "arb_motion_ratio_deg_per_mm",
    "ARB Motion Ratio",
    "deg/mm",
    "rate",
    "corner",
    component="arb",
    motion_ratio=True,
)
ROADWHEEL_ANGLE_VS_RACK = MetricSpec(
    "roadwheel_angle_vs_rack_deg_per_mm",
    "Roadwheel Angle vs Rack",
    "deg/mm",
    "rate",
    "corner",
)
CAMBER_VS_RACK = MetricSpec(
    "camber_vs_rack_deg_per_mm", "Camber vs Rack", "deg/mm", "rate", "corner"
)
ROADWHEEL_ANGLE_VS_ROLL = MetricSpec(
    "roadwheel_angle_vs_roll_deg_per_deg",
    "Roadwheel Angle vs Roll",
    "deg/deg",
    "rate",
    "corner",
)
CAMBER_VS_ROLL = MetricSpec(
    "camber_vs_roll_deg_per_deg", "Camber vs Roll", "deg/deg", "rate", "corner"
)
ROADWHEEL_ANGLE_VS_HEAVE = MetricSpec(
    "roadwheel_angle_vs_heave_deg_per_mm",
    "Roadwheel Angle vs Heave",
    "deg/mm",
    "rate",
    "corner",
)
CAMBER_VS_HEAVE = MetricSpec(
    "camber_vs_heave_deg_per_mm", "Camber vs Heave", "deg/mm", "rate", "corner"
)

# ----------------------------------------------------------------------
# Axle-level columns.
# ----------------------------------------------------------------------

ROLL_CENTER_Y = MetricSpec("roll_center_y_mm", "Roll Center Y", "mm", "state", "axle")
ROLL_CENTER_Z = MetricSpec("roll_center_z_mm", "Roll Center Z", "mm", "state", "axle")
TOTAL_ROADWHEEL_ANGLE = MetricSpec(
    "total_roadwheel_angle_deg", "Total Roadwheel Angle", "deg", "state", "axle"
)
TRACK = MetricSpec("track_mm", "Track", "mm", "state", "axle")
RACK_DISPLACEMENT = MetricSpec(
    "rack_displacement_mm", "Rack Displacement", "mm", "state", "axle"
)
ARB_TWIST = MetricSpec(
    "arb_twist_deg", "ARB Twist", "deg", "state", "axle", component="arb"
)
HEAVE = MetricSpec("heave_mm", "Heave", "mm", "state", "axle")
ROLL = MetricSpec("roll_deg", "Roll", "deg", "state", "axle")
RIDE_HEIGHT_CHANGE = MetricSpec(
    "ride_height_change_mm", "Ride Height Change", "mm", "state", "axle"
)
ACKERMANN = MetricSpec("ackermann_pct", "Ackermann", "%", "state", "axle")
ARB_TWIST_VS_ROLL = MetricSpec(
    "arb_twist_vs_roll_deg_per_deg",
    "ARB Twist vs Roll",
    "deg/deg",
    "rate",
    "axle",
    component="arb",
)

# All specs declared outside the catalog, in stable display order.
_EXTRA_SPECS: tuple[MetricSpec, ...] = (
    ROCKER_ANGLE,
    TORSION_BAR_TWIST,
    ARB_ARM_ANGLE,
    CAMBER_GAIN,
    ROADWHEEL_ANGLE_VS_BUMP,
    CASTER_GAIN,
    KPI_GAIN,
    HALF_TRACK_RATE,
    WHEEL_RECESSION_RATE,
    DAMPER_MOTION_RATIO,
    ROCKER_MOTION_RATIO,
    TORSION_BAR_MOTION_RATIO,
    ARB_MOTION_RATIO,
    ROADWHEEL_ANGLE_VS_RACK,
    CAMBER_VS_RACK,
    ROADWHEEL_ANGLE_VS_ROLL,
    CAMBER_VS_ROLL,
    ROADWHEEL_ANGLE_VS_HEAVE,
    CAMBER_VS_HEAVE,
    ROLL_CENTER_Y,
    ROLL_CENTER_Z,
    TOTAL_ROADWHEEL_ANGLE,
    TRACK,
    RACK_DISPLACEMENT,
    ARB_TWIST,
    HEAVE,
    ROLL,
    RIDE_HEIGHT_CHANGE,
    ACKERMANN,
    ARB_TWIST_VS_ROLL,
)


def all_metric_specs() -> tuple[MetricSpec, ...]:
    """
    Every metric column family the package can emit.

    The unconditional corner state metrics come from the catalog (which also
    carries their compute functions); the conditional, rate, and axle-level
    families are declared above.
    """
    catalog_specs = tuple(
        MetricSpec(
            key=metric.column_name,
            label=metric.label,
            unit=metric.unit,
            kind="state",
            scope="corner",
        )
        for metric in get_default_corner_metrics()
    )
    return catalog_specs + _EXTRA_SPECS


def specs_by_key() -> dict[str, MetricSpec]:
    """All specs keyed by base column key."""
    return {spec.key: spec for spec in all_metric_specs()}


def motion_ratio_specs() -> tuple[MetricSpec, ...]:
    """
    The motion-ratio metrics, in display order.

    Each is a per-component installation ratio relative to hub (wheel center)
    vertical travel, with a display name and unit ready for front-ends.
    """
    return tuple(spec for spec in all_metric_specs() if spec.motion_ratio)
