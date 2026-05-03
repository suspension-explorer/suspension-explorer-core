"""
Suspension configuration models.

This module defines configuration structures for suspension systems, including units,
wheel parameters, and static alignment settings.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kinematics.core.constants import MM_PER_INCH
from kinematics.io.validation import PydanticPoint3


class TireConfig(BaseModel):
    """
    Configuration parameters for a tire.

    Attributes:
        aspect_ratio: Aspect ratio as a fraction in [0, 1], e.g., 0.55 for 55%.
        section_width: Section width in mm.
        rim_diameter: Rim diameter in inches.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    aspect_ratio: float
    section_width: float
    rim_diameter: float

    @field_validator("aspect_ratio")
    @classmethod
    def check_aspect_ratio(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"aspect_ratio must be in [0, 1], got {v}")
        return v

    @property
    def sidewall_height(self) -> float:
        """Calculate sidewall height in mm."""
        return self.aspect_ratio * self.section_width

    @property
    def rim_diameter_mm(self) -> float:
        """Convert rim diameter from inches to mm."""
        return self.rim_diameter * MM_PER_INCH

    @property
    def nominal_radius(self) -> float:
        """
        Calculate nominal tire radius in mm.

        This makes no consideration of vertical load or speed growth effects.
        """
        return (self.rim_diameter_mm + 2 * self.sidewall_height) / 2


class WheelConfig(BaseModel):
    """
    Configuration parameters for a wheel and tire assembly.

    Attributes:
        offset: Wheel offset (ET) from hub mounting face to wheel center plane
            in mm. Positive means wheel centerline is inboard of the hub face.
        tire: Tire configuration parameters.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    offset: float
    tire: TireConfig


class CamberShimConfig(BaseModel):
    """
    Configuration for a camber shim adjustment.

    This type of shim sits outboard of the top balljoint, effectively splitting the
    upright in two. A local assembly solve rotates the UBJ-side shim block about the
    upper ball joint and the lower upright body about the lower ball joint while the
    shim faces remain parallel at the requested setup thickness.

    The shim geometry is defined by two ordered dowel datum points (A, B) on the
    nominal mid-thickness plane, plus a shared face normal. The design upper and
    lower face positions are derived by offsetting +/- 0.5 * design_thickness
    along the normal from each datum point.

    Attributes:
        shim_face_point_a: First dowel datum on the design mid-thickness plane (mm).
        shim_face_point_b: Second dowel datum on the design mid-thickness plane (mm).
        shim_face_normal: Unit vector perpendicular to the design shim faces.
        design_thickness: Shim stack thickness in mm at design condition.
        setup_thickness: Actual shim stack thickness in mm for this configuration.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    shim_face_point_a: PydanticPoint3
    shim_face_point_b: PydanticPoint3
    shim_face_normal: PydanticPoint3
    design_thickness: float
    setup_thickness: float

    @model_validator(mode="after")
    def validate_face_definition(self) -> "CamberShimConfig":
        import numpy as np

        from kinematics.core.constants import EPS_GEOMETRIC
        from kinematics.core.geometry import extract_array

        normal_data = extract_array(self.shim_face_normal)
        magnitude = float(np.linalg.norm(normal_data))
        if magnitude < EPS_GEOMETRIC:
            raise ValueError("shim_face_normal vector is near-zero")

        point_a_data = extract_array(self.shim_face_point_a)
        point_b_data = extract_array(self.shim_face_point_b)
        datum_separation = float(np.linalg.norm(point_b_data - point_a_data))
        if datum_separation < EPS_GEOMETRIC:
            raise ValueError("shim_face_point_a and shim_face_point_b must be distinct")

        return self


class SuspensionConfig(BaseModel):
    """
    Complete configuration for a suspension system.

    Attributes:
        steered: Whether this suspension corner is steered.
        wheel: Wheel configuration parameters.
        cg_position: Center of gravity position in mm (required for anti-dive/squat).
        wheelbase: Wheelbase distance in mm.
        camber_shim: Optional camber shim configuration.
        upright_mounted_points: List of point names mounted to the upright that should
            move when camber shims are applied.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    steered: bool
    wheel: WheelConfig
    cg_position: PydanticPoint3
    wheelbase: float
    camber_shim: CamberShimConfig | None = None
    upright_mounted_points: list[str] = [
        "axle_inboard",
        "axle_outboard",
        "pushrod_outboard",
        "trackrod_outboard",
    ]
