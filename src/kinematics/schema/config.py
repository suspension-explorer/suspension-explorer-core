"""Suspension configuration schema models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kinematics.core.constants import MM_PER_INCH
from kinematics.schema.coercion import PydanticDirection3, PydanticPoint3


class TireConfig(BaseModel):
    """Tire dimensions used to derive the nominal unloaded radius."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    aspect_ratio: float
    section_width: float
    rim_diameter: float

    @field_validator("aspect_ratio")
    @classmethod
    def check_aspect_ratio(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError(f"aspect_ratio must be in [0, 1], got {value}")
        return value

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
        """Calculate nominal unloaded tire radius in mm."""
        return (self.rim_diameter_mm + 2 * self.sidewall_height) / 2


class WheelConfig(BaseModel):
    """Wheel offset and tire configuration."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    offset: float
    tire: TireConfig


class CamberShimConfig(BaseModel):
    """Geometry and design/setup thickness for an outboard camber shim."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    shim_face_point_a: PydanticPoint3
    shim_face_point_b: PydanticPoint3
    shim_face_normal: PydanticDirection3
    design_thickness: float
    setup_thickness: float

    @model_validator(mode="after")
    def validate_face_definition(self) -> "CamberShimConfig":
        from kinematics.core.constants import EPS_GEOMETRIC

        datum_separation = (self.shim_face_point_b - self.shim_face_point_a).norm()
        if datum_separation < EPS_GEOMETRIC:
            raise ValueError("shim_face_point_a and shim_face_point_b must be distinct")
        return self


class SuspensionConfig(BaseModel):
    """Configuration shared by the currently supported corner models."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    steered: bool
    wheel: WheelConfig
    cg_position: PydanticPoint3
    wheelbase: float
    camber_shim: CamberShimConfig | None = None
    upright_mounted_points: list[str] = Field(
        default_factory=lambda: [
            "axle_inboard",
            "axle_outboard",
            "pushrod_outboard",
            "trackrod_outboard",
        ]
    )
    axle_position: Literal["front", "rear"] | None = None
    front_brake_bias: float | None = None
    driven_axle: Literal["front", "rear"] | None = None

    @field_validator("front_brake_bias")
    @classmethod
    def check_front_brake_bias(cls, value: float | None) -> float | None:
        """Require front brake bias to be a fraction of total braking force."""
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"front_brake_bias must be in [0, 1], got {value}")
        return value
