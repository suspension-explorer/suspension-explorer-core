"""Suspension configuration schema models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kinematics.core.enums import ArbType, AxlePosition, HeaveLinkType
from kinematics.core.primitives.constants import MM_PER_INCH
from kinematics.core.primitives.geometry import Direction3, Point3


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

    shim_face_point_a: Point3
    shim_face_point_b: Point3
    shim_face_normal: Direction3
    design_thickness: float
    setup_thickness: float

    @model_validator(mode="after")
    def validate_face_definition(self) -> "CamberShimConfig":
        from kinematics.core.primitives.constants import EPS_GEOMETRIC

        datum_separation = (self.shim_face_point_b - self.shim_face_point_a).norm()
        if datum_separation < EPS_GEOMETRIC:
            raise ValueError("shim_face_point_a and shim_face_point_b must be distinct")
        return self


class VehicleConfig(BaseModel):
    """Vehicle-wide configuration shared across all axles."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    cg_position: Point3
    wheelbase: float
    front_brake_bias: float | None = None
    driven_axle: AxlePosition | None = None

    @field_validator("front_brake_bias")
    @classmethod
    def check_front_brake_bias(cls, value: float | None) -> float | None:
        """Require front brake bias to be a fraction of total braking force."""
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"front_brake_bias must be in [0, 1], got {value}")
        return value


class AntiRollConfig(BaseModel):
    """Selected axle anti-roll mechanism."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ArbType


class HeaveLinkConfig(BaseModel):
    """Selected axle heave-link mechanism."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: HeaveLinkType


class AxleConfig(BaseModel):
    """Configuration and shared mechanisms owned by one axle."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    axle_position: AxlePosition
    steered: bool
    wheel: WheelConfig
    anti_roll: AntiRollConfig
    heave_link: HeaveLinkConfig


class CornerConfig(BaseModel):
    """Side-local setup applied to one corner model."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    camber_shim: CamberShimConfig | None = None


class SuspensionConfig(VehicleConfig):
    """Complete runtime configuration for one built corner suspension."""

    steered: bool
    wheel: WheelConfig
    axle_position: AxlePosition | None = None
    camber_shim: CamberShimConfig | None = None

    @classmethod
    def from_parts(
        cls,
        vehicle: VehicleConfig,
        axle: AxleConfig,
        corner: CornerConfig,
    ) -> "SuspensionConfig":
        """Combine shared vehicle data with one corner's local setup."""
        return cls.model_validate(
            {
                **vehicle.model_dump(),
                "steered": axle.steered,
                "wheel": axle.wheel,
                "axle_position": axle.axle_position,
                "camber_shim": corner.camber_shim,
            }
        )
