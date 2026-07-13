"""
Tests for the suspension system.

Tests cover:
- Suspension base class
- DoubleWishboneSuspension
- Registry functions
- Validation utilities
- YAML loading
"""

from pathlib import Path

import numpy as np
import pytest

from kinematics.cli.io.loaders import load_geometry
from kinematics.core.elements import UprightElement
from kinematics.core.primitives.enums import PointID, ShimType, Units
from kinematics.core.primitives.geometry import Direction3, Point3
from kinematics.core.primitives.point_ref import Side
from kinematics.core.schema.config import (
    CamberShimConfig,
    SuspensionConfig,
    TireConfig,
    WheelConfig,
)
from kinematics.core.suspensions.base import Suspension
from kinematics.core.suspensions.corner import DoubleWishboneSuspension
from kinematics.core.suspensions.registry import (
    get_suspension_class,
    list_supported_types,
)

# Test fixtures


@pytest.fixture
def valid_hardpoints() -> dict[PointID, Point3]:
    """
    Valid hardpoints for double wishbone suspension.
    """
    return {
        PointID.LOWER_WISHBONE_INBOARD_FRONT: Point3([250, 400, 200]),
        PointID.LOWER_WISHBONE_INBOARD_REAR: Point3([-250, 450, 200]),
        PointID.LOWER_WISHBONE_OUTBOARD: Point3([0, 900, 200]),
        PointID.UPPER_WISHBONE_INBOARD_FRONT: Point3([225, 350, 500]),
        PointID.UPPER_WISHBONE_INBOARD_REAR: Point3([-275, 350, 500]),
        PointID.UPPER_WISHBONE_OUTBOARD: Point3([-25, 750, 500]),
        PointID.TRACKROD_INBOARD: Point3([50, 200, 250]),
        PointID.TRACKROD_OUTBOARD: Point3([150, 800, 275]),
        PointID.AXLE_INBOARD: Point3([-20, 800, 308.426]),
        PointID.AXLE_OUTBOARD: Point3([-20, 950, 313.426]),
    }


@pytest.fixture
def valid_config() -> SuspensionConfig:
    """
    Valid suspension configuration.
    """
    return SuspensionConfig(
        steered=True,
        wheel=WheelConfig(
            offset=0,
            tire=TireConfig(
                aspect_ratio=0.55,
                section_width=270,
                rim_diameter=13,
            ),
        ),
        cg_position=Point3([1250, 0, 450]),
        wheelbase=2500.0,
        camber_shim=CamberShimConfig(
            shim_face_point_a=Point3([-25.0, 750.0, 510.0]),
            shim_face_point_b=Point3([-25.0, 750.0, 490.0]),
            shim_face_normal=Direction3([0.0, 1.0, 0.0]),
            design_thickness=30.0,
            setup_thickness=30.0,
        ),
    )


# Test Suspension base class


class TestSuspensionBase:
    """
    Tests for Suspension base class.
    """

    def test_all_valid_points(self):
        """
        Test all_valid_points combines required and optional.
        """
        valid = DoubleWishboneSuspension.all_valid_points()
        # Check required points are included
        assert PointID.LOWER_WISHBONE_OUTBOARD in valid
        assert PointID.UPPER_WISHBONE_OUTBOARD in valid
        # Variant-specific points are excluded from the basic topology.
        assert PointID.PUSHROD_OUTBOARD not in valid
        assert PointID.STRUT_BOTTOM not in valid

    def test_matches_type(self):
        """
        Test exact, case-insensitive type matching.
        """
        assert DoubleWishboneSuspension.matches_type("double_wishbone")
        assert DoubleWishboneSuspension.matches_type("DOUBLE_WISHBONE")
        assert DoubleWishboneSuspension.matches_type("double_wishbone_front")
        assert not DoubleWishboneSuspension.matches_type("macpherson_strut")


# Test DoubleWishboneSuspension


class TestDoubleWishboneSuspension:
    """
    Tests for DoubleWishboneSuspension class.
    """

    def test_class_attributes(self):
        """
        Test class-level attributes are correctly defined.
        """
        assert DoubleWishboneSuspension.TYPE_KEY == "double_wishbone"
        assert "double_wishbone_front" in DoubleWishboneSuspension.ALIASES
        assert ShimType.OUTBOARD_CAMBER in DoubleWishboneSuspension.SUPPORTED_SHIMS

        # Check required points
        required = DoubleWishboneSuspension.REQUIRED_POINTS
        assert PointID.LOWER_WISHBONE_INBOARD_FRONT in required
        assert PointID.LOWER_WISHBONE_INBOARD_REAR in required
        assert PointID.LOWER_WISHBONE_OUTBOARD in required
        assert PointID.UPPER_WISHBONE_INBOARD_FRONT in required
        assert PointID.UPPER_WISHBONE_INBOARD_REAR in required
        assert PointID.UPPER_WISHBONE_OUTBOARD in required
        assert PointID.TRACKROD_INBOARD in required
        assert PointID.TRACKROD_OUTBOARD in required
        assert PointID.AXLE_INBOARD in required
        assert PointID.AXLE_OUTBOARD in required

    def test_create_suspension(self, valid_hardpoints, valid_config):
        """
        Test creating a suspension instance.
        """
        suspension = DoubleWishboneSuspension(
            name="test",
            version="1.0.0",
            units=Units.MILLIMETERS,
            side=Side.LEFT,
            hardpoints=valid_hardpoints,
            config=valid_config,
        )
        assert suspension.name == "test"
        assert len(suspension.hardpoints) == 10

    def test_rejects_missing_hardpoints(self, valid_hardpoints, valid_config):
        """
        Test that missing required hardpoints are rejected.
        """
        del valid_hardpoints[PointID.UPPER_WISHBONE_OUTBOARD]
        with pytest.raises(ValueError, match="Missing required hardpoints"):
            DoubleWishboneSuspension(
                name="test",
                units=Units.MILLIMETERS,
                side=Side.LEFT,
                hardpoints=valid_hardpoints,
                config=valid_config,
            )

    def test_initial_state(self, valid_hardpoints, valid_config):
        """
        Test generating initial state.
        """
        suspension = DoubleWishboneSuspension(
            name="test",
            units=Units.MILLIMETERS,
            side=Side.LEFT,
            hardpoints=valid_hardpoints,
            config=valid_config,
        )
        state = suspension.initial_state()
        assert state is not None
        assert PointID.UPPER_WISHBONE_OUTBOARD in state.positions
        assert PointID.WHEEL_CENTER in state.positions  # Derived point

    def test_free_points(self, valid_hardpoints, valid_config):
        """
        Test getting free points.
        """
        suspension = DoubleWishboneSuspension(
            name="test",
            units=Units.MILLIMETERS,
            side=Side.LEFT,
            hardpoints=valid_hardpoints,
            config=valid_config,
        )
        free = suspension.free_points()
        assert PointID.UPPER_WISHBONE_OUTBOARD in free
        assert PointID.LOWER_WISHBONE_OUTBOARD in free

    def test_constraints(self, valid_hardpoints, valid_config):
        """
        Test building constraints.
        """
        suspension = DoubleWishboneSuspension(
            name="test",
            units=Units.MILLIMETERS,
            side=Side.LEFT,
            hardpoints=valid_hardpoints,
            config=valid_config,
        )
        constraints = suspension.constraints()
        assert len(constraints) > 0

    def test_derived_spec(self, valid_hardpoints, valid_config):
        """
        Test derived point specification.
        """
        suspension = DoubleWishboneSuspension(
            name="test",
            units=Units.MILLIMETERS,
            side=Side.LEFT,
            hardpoints=valid_hardpoints,
            config=valid_config,
        )
        spec = suspension.derived_spec()
        assert PointID.WHEEL_CENTER in spec.functions
        assert PointID.CONTACT_PATCH_CENTER in spec.functions

    def test_suspension_elements(self, valid_hardpoints, valid_config):
        """
        Test physical suspension element generation.
        """
        suspension = DoubleWishboneSuspension(
            name="test",
            units=Units.MILLIMETERS,
            side=Side.LEFT,
            hardpoints=valid_hardpoints,
            config=valid_config,
        )
        elements = suspension.elements()
        assert len(elements) > 0
        # Check for expected element labels.
        labels = [element.label for element in elements]
        assert "Upper Wishbone Front Leg" in labels
        assert "Upper Wishbone Rear Leg" in labels
        assert "Lower Wishbone Front Leg" in labels
        assert "Lower Wishbone Rear Leg" in labels
        upright = next(
            element for element in elements if isinstance(element, UprightElement)
        )
        assert upright.hardpoints == (
            PointID.UPPER_WISHBONE_OUTBOARD,
            PointID.LOWER_WISHBONE_OUTBOARD,
            PointID.TRACKROD_OUTBOARD,
        )
        assert upright.attachments == (
            PointID.AXLE_INBOARD,
            PointID.AXLE_OUTBOARD,
        )
        assembly = suspension.assembly()
        assert PointID.UPPER_WISHBONE_INBOARD_FRONT in assembly.points.fixed
        assert PointID.UPPER_WISHBONE_OUTBOARD in assembly.points.free
        assert PointID.WHEEL_CENTER in assembly.points.derived


class TestCamberShimConfig:
    """
    Tests for camber shim configuration validation.
    """

    def test_rejects_coincident_face_datums(self):
        """
        Ordered A/B shim datums must be distinct so they carry interface clocking.
        """
        with pytest.raises(ValueError, match="must be distinct"):
            CamberShimConfig(
                shim_face_point_a=Point3([-25.0, 750.0, 500.0]),
                shim_face_point_b=Point3([-25.0, 750.0, 500.0]),
                shim_face_normal=Direction3([0.0, 1.0, 0.0]),
                design_thickness=30.0,
                setup_thickness=40.0,
            )


# Test registry


class TestRegistry:
    """
    Tests for suspension registry functions.
    """

    def test_list_supported_types(self):
        """
        Test listing supported types.
        """
        types = list_supported_types()
        assert "double_wishbone" in types
        assert "double_wishbone_coilover" in types
        assert "double_wishbone_front" in types

    def test_get_suspension_class(self):
        """
        Test getting a suspension class by key.
        """
        cls = get_suspension_class("double_wishbone")
        assert cls is not None
        assert cls == DoubleWishboneSuspension

        # Test case insensitivity
        cls2 = get_suspension_class("DOUBLE_WISHBONE")
        assert cls2 is not None
        assert cls2 == DoubleWishboneSuspension

    def test_get_suspension_class_not_found(self):
        """
        Test getting a non-existent suspension class.
        """
        assert get_suspension_class("nonexistent") is None


# Test YAML loading


class TestYAMLLoading:
    """
    Tests for loading geometry from YAML files.
    """

    def test_load_yaml(self, tmp_path):
        """
        Test loading YAML geometry file.
        """
        yaml_content = """
type: double_wishbone
side: LEFT
name: "Test"
version: "1.0.0"
units: MILLIMETERS

hardpoints:
  LOWER_WISHBONE_INBOARD_FRONT: [250, 400, 200]
  LOWER_WISHBONE_INBOARD_REAR: [-250, 450, 200]
  LOWER_WISHBONE_OUTBOARD: [0, 900, 200]
  UPPER_WISHBONE_INBOARD_FRONT: [225, 350, 500]
  UPPER_WISHBONE_INBOARD_REAR: [-275, 350, 500]
  UPPER_WISHBONE_OUTBOARD: [-25, 750, 500]
  TRACKROD_INBOARD: [50, 200, 250]
  TRACKROD_OUTBOARD: [150, 800, 275]
  AXLE_INBOARD: [-20, 800, 308.426]
  AXLE_OUTBOARD: [-20, 950, 313.426]

config:
  steered: true
  wheel:
    offset: 0
    tire:
      aspect_ratio: 0.55
      section_width: 270
      rim_diameter: 13
  cg_position: {x: 1250, y: 0, z: 450}
  wheelbase: 2500.0
"""
        yaml_file = tmp_path / "test_geometry.yaml"
        yaml_file.write_text(yaml_content)

        result = load_geometry(yaml_file)

        assert isinstance(result, Suspension)
        assert isinstance(result, DoubleWishboneSuspension)
        assert result.name == "Test"

    def test_load_with_camber_shim(self, tmp_path):
        """
        Test loading YAML with camber shim configuration.
        """
        yaml_content = """
type: double_wishbone
side: LEFT
name: "With Shim"
units: MILLIMETERS

hardpoints:
  LOWER_WISHBONE_INBOARD_FRONT: [250, 400, 200]
  LOWER_WISHBONE_INBOARD_REAR: [-250, 450, 200]
  LOWER_WISHBONE_OUTBOARD: [0, 900, 200]
  UPPER_WISHBONE_INBOARD_FRONT: [225, 350, 500]
  UPPER_WISHBONE_INBOARD_REAR: [-275, 350, 500]
  UPPER_WISHBONE_OUTBOARD: [-25, 750, 500]
  TRACKROD_INBOARD: [50, 200, 250]
  TRACKROD_OUTBOARD: [150, 800, 275]
  AXLE_INBOARD: [-20, 800, 308.426]
  AXLE_OUTBOARD: [-20, 950, 313.426]

config:
  steered: true
  wheel:
    offset: 0
    tire:
      aspect_ratio: 0.55
      section_width: 270
      rim_diameter: 13
  cg_position: {x: 1250, y: 0, z: 450}
  wheelbase: 2500.0
  camber_shim:
    shim_face_point_a: {x: -25.0, y: 750.0, z: 510.0}
    shim_face_point_b: {x: -25.0, y: 750.0, z: 490.0}
    shim_face_normal: {x: 0.0, y: 1.0, z: 0.0}
    design_thickness: 30.0
    setup_thickness: 35.0  # 5mm more than design
"""
        yaml_file = tmp_path / "test_shim.yaml"
        yaml_file.write_text(yaml_content)

        result = load_geometry(yaml_file)
        assert result.config is not None
        assert result.config.camber_shim is not None
        assert result.config.camber_shim.setup_thickness == 35.0

    def test_load_rejects_unknown_type(self, tmp_path):
        """
        Test that unknown geometry types are rejected.
        """
        yaml_content = """
type: unknown_suspension
hardpoints: {}
config:
  steered: true
  wheel:
    offset: 0
    tire:
      aspect_ratio: 0.55
      section_width: 270
      rim_diameter: 13
  cg_position: {x: 0, y: 0, z: 0}
  wheelbase: 2500.0
"""
        yaml_file = tmp_path / "unknown.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="Unsupported geometry type"):
            load_geometry(yaml_file)

    def test_load_rejects_missing_hardpoints(self, tmp_path):
        """
        Test that missing required hardpoints are rejected.
        """
        yaml_content = """
type: double_wishbone
side: LEFT
hardpoints:
  LOWER_WISHBONE_INBOARD_FRONT: [250, 400, 200]
  # Missing most required points!

config:
  steered: true
  wheel:
    offset: 0
    tire:
      aspect_ratio: 0.55
      section_width: 270
      rim_diameter: 13
  cg_position: {x: 0, y: 0, z: 0}
  wheelbase: 2500.0
"""
        yaml_file = tmp_path / "missing.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="Missing required hardpoints"):
            load_geometry(yaml_file)

    def test_load_file_not_found(self):
        """
        Test that FileNotFoundError is raised for missing files.
        """
        with pytest.raises(FileNotFoundError):
            load_geometry(Path("/nonexistent/path.yaml"))


# Integration tests


class TestIntegration:
    """
    Integration tests for the complete suspension system.
    """

    def test_full_workflow(self, valid_hardpoints, valid_config):
        """
        Test complete workflow from hardpoints to solved state.
        """
        # Create suspension
        suspension = DoubleWishboneSuspension(
            name="test",
            units=Units.MILLIMETERS,
            side=Side.LEFT,
            hardpoints=valid_hardpoints,
            config=valid_config,
        )

        # Get initial state
        state = suspension.initial_state()

        # Verify all required points present
        for point_id in DoubleWishboneSuspension.REQUIRED_POINTS:
            assert point_id in state.positions

        # Verify derived points calculated
        assert PointID.WHEEL_CENTER in state.positions
        assert PointID.CONTACT_PATCH_CENTER in state.positions

        # Verify constraints can be built
        constraints = suspension.constraints()
        assert len(constraints) > 0

    def test_shim_application_workflow(self, valid_hardpoints, valid_config):
        """
        Test workflow with camber shim application.
        """
        # Create config with modified shim thickness (5mm more than design).
        new_shim = valid_config.camber_shim.model_copy(update={"setup_thickness": 35.0})
        config = valid_config.model_copy(update={"camber_shim": new_shim})

        suspension = DoubleWishboneSuspension(
            name="test",
            units=Units.MILLIMETERS,
            side=Side.LEFT,
            hardpoints=valid_hardpoints,
            config=config,
        )
        state = suspension.initial_state()

        # Axle points should have moved due to shim
        original_axle = valid_hardpoints[PointID.AXLE_OUTBOARD]
        new_axle = state.positions[PointID.AXLE_OUTBOARD]

        # Should not be identical (shim rotates attachments)
        assert not np.allclose(original_axle.data, new_axle.data)
