"""Structured geometry specifications for explicit corner models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kinematics.core.enums import (
    ActuationType,
    ArbType,
    CornerSpringType,
    HeaveLinkType,
    MountBody,
    Scope,
    SuspensionType,
    Units,
)
from kinematics.core.primitives.point_ref import Side
from kinematics.core.schema.config import (
    AxleConfig,
    CornerConfig,
    SuspensionConfig,
    VehicleConfig,
)
from kinematics.core.schema.decoding import Point3Value, PointIDValue, SideValue

HardpointMap = dict[PointIDValue, Point3Value]


class GeometrySpecBase(BaseModel):
    """Fields shared by every geometry specification."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    name: str = "unnamed"
    version: str = "0.0.0"
    units: Units = Units.MILLIMETERS
    type: SuspensionType
    scope: Scope


class CornerGeometrySpecBase(GeometrySpecBase):
    """Fields required by every explicitly sided corner geometry."""

    scope: Literal[Scope.CORNER] = Scope.CORNER
    side: SideValue = Side.LEFT
    config: SuspensionConfig

    @model_validator(mode="after")
    def check_physical_side(self) -> "CornerGeometrySpecBase":
        """A corner must be declared as the physical left or right side."""
        if self.side == Side.CENTER:
            raise ValueError("Corner geometry side must be 'left' or 'right'.")
        return self


class MechanismSpecBase(BaseModel):
    """Strict base for one explicitly selected suspension mechanism."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ActuationSpec(MechanismSpecBase):
    """Selected corner actuation mechanism."""

    type: ActuationType
    # The rigid corner body that carries the moving pickup: the spring pickup
    # for direct actuation or the outboard pushrod end for pushrod-rocker.
    mount: MountBody


class CornerSpringSpec(MechanismSpecBase):
    """Selected corner spring mechanism."""

    type: CornerSpringType


def check_double_wishbone_mechanism_combination(
    actuation: ActuationSpec,
    spring: CornerSpringSpec,
) -> None:
    """Reject combinations whose physical connection is not implemented."""
    if (
        actuation.type is ActuationType.DIRECT
        and spring.type is CornerSpringType.TORSION_BAR
    ):
        raise ValueError("Direct torsion-bar actuation is not implemented yet")


class DoubleWishboneGeometrySpec(CornerGeometrySpecBase):
    """Double-wishbone corner with composed actuation and spring mechanisms."""

    type: Literal[SuspensionType.DOUBLE_WISHBONE] = SuspensionType.DOUBLE_WISHBONE
    actuation: ActuationSpec
    spring: CornerSpringSpec
    hardpoints: HardpointMap

    @model_validator(mode="after")
    def check_mechanisms(self) -> "DoubleWishboneGeometrySpec":
        """Validate the selected corner mechanism combination."""
        check_double_wishbone_mechanism_combination(self.actuation, self.spring)
        return self


class MacPhersonGeometrySpec(CornerGeometrySpecBase):
    """MacPherson strut corner with the configured wheel-heading link."""

    type: Literal[SuspensionType.MACPHERSON] = SuspensionType.MACPHERSON
    hardpoints: HardpointMap


def check_trailing_arm_spring(spring: CornerSpringSpec) -> None:
    """Require one of the semi-trailing arm's implemented spring layouts."""
    if spring.type not in (CornerSpringType.COILOVER, CornerSpringType.TORSION_BAR):
        raise ValueError(
            "Semi-trailing arm spring must be 'coilover' or 'torsion_bar'"
        )


class TrailingArmGeometrySpec(CornerGeometrySpecBase):
    """Unsteered semi-trailing arm with coil or pivot-mounted torsion springing."""

    type: Literal[SuspensionType.TRAILING_ARM] = SuspensionType.TRAILING_ARM
    spring: CornerSpringSpec
    hardpoints: HardpointMap

    @model_validator(mode="after")
    def check_mechanisms(self) -> "TrailingArmGeometrySpec":
        """Reject spring choices without a trailing-arm implementation."""
        check_trailing_arm_spring(self.spring)
        if self.config.steering.type.value != "none":
            raise ValueError(
                "Semi-trailing arm geometry is unsteered; "
                "config.steering.type must be 'none'"
            )
        return self


class DoubleWishboneAxleConfig(AxleConfig):
    """Shared double-wishbone axle topology and optional side-local setup."""

    actuation: ActuationSpec
    spring: CornerSpringSpec
    left_setup: CornerConfig = Field(default_factory=CornerConfig)
    right_setup: CornerConfig | None = None

    @model_validator(mode="after")
    def check_mechanisms(self) -> "DoubleWishboneAxleConfig":
        """Validate the symmetric corner mechanisms and shared hardware."""
        check_double_wishbone_mechanism_combination(self.actuation, self.spring)
        has_rocker = self.actuation.type is ActuationType.PUSHROD_ROCKER
        if self.anti_roll.type in (ArbType.U_BAR, ArbType.T_BAR) and not has_rocker:
            raise ValueError(
                "The implemented anti-roll mechanism requires pushrod-rocker actuation"
            )
        if self.heave_link.type is HeaveLinkType.ROCKER_TO_ROCKER and not has_rocker:
            raise ValueError(
                "A rocker-to-rocker heave link requires pushrod-rocker actuation"
            )
        return self


class AxleHardpointsSpec(BaseModel):
    """Left, optional explicit right, and shared center axle hardpoints."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    left: HardpointMap
    right: HardpointMap | None = None
    center: HardpointMap = Field(default_factory=dict)


class AxleGeometrySpecBase(GeometrySpecBase):
    """Fields shared by every composed full-axle geometry."""

    scope: Literal[Scope.AXLE] = Scope.AXLE
    vehicle_config: VehicleConfig
    axle_config: AxleConfig
    hardpoints: AxleHardpointsSpec


class DoubleWishboneAxleGeometrySpec(AxleGeometrySpecBase):
    """Double-wishbone axle with corner mechanisms and shared hardware."""

    type: Literal[SuspensionType.DOUBLE_WISHBONE] = SuspensionType.DOUBLE_WISHBONE
    axle_config: DoubleWishboneAxleConfig

    @model_validator(mode="after")
    def check_right_setup(self) -> "DoubleWishboneAxleGeometrySpec":
        """Keep explicit asymmetric geometry and side-local setup paired."""
        if self.axle_config.right_setup is not None and self.hardpoints.right is None:
            raise ValueError(
                "axle_config.right_setup requires explicit hardpoints.right"
            )
        if (
            self.hardpoints.right is not None
            and self.axle_config.left_setup.camber_shim is not None
            and self.axle_config.right_setup is None
        ):
            raise ValueError(
                "Explicit hardpoints.right requires axle_config.right_setup when "
                "axle_config.left_setup contains side-local setup"
            )
        return self


class MacPhersonAxleGeometrySpec(AxleGeometrySpecBase):
    """MacPherson axle with a left and optional explicit right strut corner."""

    type: Literal[SuspensionType.MACPHERSON] = SuspensionType.MACPHERSON

    @model_validator(mode="after")
    def check_axle_mechanisms(self) -> "MacPhersonAxleGeometrySpec":
        """Reject shared hardware that needs a rocker corner."""
        if self.axle_config.anti_roll.type in (ArbType.U_BAR, ArbType.T_BAR):
            raise ValueError(
                "The implemented anti-roll mechanism requires pushrod-rocker "
                "actuation, which a MacPherson corner does not provide"
            )
        if self.axle_config.heave_link.type is HeaveLinkType.ROCKER_TO_ROCKER:
            raise ValueError(
                "A rocker-to-rocker heave link requires pushrod-rocker "
                "actuation, which a MacPherson corner does not provide"
            )
        return self


class TrailingArmAxleConfig(AxleConfig):
    """Shared configuration for two unsteered semi-trailing-arm corners."""

    spring: CornerSpringSpec

    @model_validator(mode="after")
    def check_mechanisms(self) -> "TrailingArmAxleConfig":
        """Limit the axle to hardware represented by this locating model."""
        check_trailing_arm_spring(self.spring)
        if self.steering.type.value != "none":
            raise ValueError(
                "Semi-trailing arm axle is unsteered; "
                "axle_config.steering.type must be 'none'"
            )
        if self.anti_roll.type is not ArbType.NONE:
            raise ValueError(
                "Semi-trailing arm axle does not support shared anti-roll hardware yet"
            )
        if self.heave_link.type is not HeaveLinkType.NONE:
            raise ValueError(
                "Semi-trailing arm axle does not support a shared heave link"
            )
        return self


class TrailingArmAxleGeometrySpec(AxleGeometrySpecBase):
    """Mirrored or explicit full axle of unsteered semi-trailing arms."""

    type: Literal[SuspensionType.TRAILING_ARM] = SuspensionType.TRAILING_ARM
    axle_config: TrailingArmAxleConfig


GeometrySpec = (
    DoubleWishboneGeometrySpec
    | MacPhersonGeometrySpec
    | TrailingArmGeometrySpec
    | DoubleWishboneAxleGeometrySpec
    | MacPhersonAxleGeometrySpec
    | TrailingArmAxleGeometrySpec
)
