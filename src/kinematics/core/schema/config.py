"""Suspension configuration schema models."""

from __future__ import annotations

from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kinematics.core.enums import ArbType, AxlePosition, HeaveLinkType, SteeringType
from kinematics.core.primitives.constants import EPS_GEOMETRIC, MM_PER_INCH
from kinematics.core.schema.decoding import Direction3Value, Point3Value


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

    shim_face_point_a: Point3Value
    shim_face_point_b: Point3Value
    shim_face_normal: Direction3Value
    design_thickness: float
    setup_thickness: float

    @model_validator(mode="after")
    def validate_face_definition(self) -> CamberShimConfig:
        datum_separation = (self.shim_face_point_b - self.shim_face_point_a).norm()
        if datum_separation < EPS_GEOMETRIC:
            raise ValueError("shim_face_point_a and shim_face_point_b must be distinct")
        return self


class LengthShimConfig(BaseModel):
    """Design and setup thickness for a shim that changes a link length.

    Lengths are in millimeters. The installed link-length adjustment is
    ``setup_thickness - design_thickness``: a positive value lengthens the
    affected pushrod, pullrod, toe link, or track rod.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    design_thickness: float
    setup_thickness: float

    @field_validator("design_thickness", "setup_thickness")
    @classmethod
    def check_finite_thickness(cls, value: float) -> float:
        """Require finite shim stack thicknesses."""
        if not isfinite(value):
            raise ValueError("Shim thicknesses must be finite")
        return value

    @property
    def length_adjustment(self) -> float:
        """Return the setup link-length change in millimeters."""
        return self.setup_thickness - self.design_thickness


class VehicleConfig(BaseModel):
    """Vehicle-wide configuration shared across all axles."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    cg_position: Point3Value
    wheelbase: float
    front_brake_bias: float | None = None
    driven_axle: AxlePosition | None = None

    @field_validator("wheelbase")
    @classmethod
    def check_positive_vehicle_length(cls, value: float) -> float:
        """Require a finite positive wheelbase for vehicle-level metrics."""
        if not isfinite(value) or value <= 0.0:
            raise ValueError("Vehicle length inputs must be finite and positive")
        return value

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


class SteeringConfig(BaseModel):
    """Selected steering actuator for one axle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: SteeringType


class CornerConfig(BaseModel):
    """Side-local setup applied to one corner model."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    camber_shim: CamberShimConfig | None = None
    pushrod_shim: LengthShimConfig | None = None
    toe_shim: LengthShimConfig | None = None

    @property
    def has_setup(self) -> bool:
        """Return whether this corner declares any side-local setup."""
        return any(
            shim is not None
            for shim in (self.camber_shim, self.pushrod_shim, self.toe_shim)
        )


class AxleConfig(BaseModel):
    """Shared axle mechanisms and optional side-local corner setup."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    axle_position: AxlePosition
    steering: SteeringConfig
    wheel: WheelConfig
    anti_roll: AntiRollConfig
    heave_link: HeaveLinkConfig
    left_setup: CornerConfig = Field(default_factory=CornerConfig)
    right_setup: CornerConfig | None = None


class SuspensionConfig(VehicleConfig):
    """Complete runtime configuration for one built corner suspension."""

    steering: SteeringConfig
    wheel: WheelConfig
    axle_position: AxlePosition | None = None
    camber_shim: CamberShimConfig | None = None
    pushrod_shim: LengthShimConfig | None = None
    toe_shim: LengthShimConfig | None = None

    @classmethod
    def from_parts(
        cls,
        vehicle: VehicleConfig,
        axle: AxleConfig,
        corner: CornerConfig,
    ) -> SuspensionConfig:
        """Combine shared vehicle data with one corner's local setup."""
        return cls.model_validate(
            {
                **vehicle.model_dump(),
                "steering": axle.steering,
                "wheel": axle.wheel,
                "axle_position": axle.axle_position,
                "camber_shim": corner.camber_shim,
                "pushrod_shim": corner.pushrod_shim,
                "toe_shim": corner.toe_shim,
            }
        )
