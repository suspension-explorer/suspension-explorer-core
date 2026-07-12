from dataclasses import dataclass

import numpy as np

from kinematics.core.enums import PointID
from kinematics.core.geometry import Point3
from kinematics.core.point_ref import PointKey


@dataclass
class LinkVisualization:
    """Configuration for visualizing a suspension link."""

    points: list[PointKey]
    color: str
    label: str
    linewidth: float = 3.0
    linestyle: str = "-"
    marker: str = "o"
    markersize: float = 10.0


@dataclass(frozen=True)
class WheelAnchors:
    """Point keys used to draw one wheel."""

    center: PointKey
    inboard: PointKey
    outboard: PointKey
    axle_inboard: PointKey
    axle_outboard: PointKey

    @classmethod
    def for_corner(cls) -> "WheelAnchors":
        """Return the conventional single-corner wheel anchors."""
        return cls(
            center=PointID.WHEEL_CENTER,
            inboard=PointID.WHEEL_INBOARD,
            outboard=PointID.WHEEL_OUTBOARD,
            axle_inboard=PointID.AXLE_INBOARD,
            axle_outboard=PointID.AXLE_OUTBOARD,
        )


@dataclass
class WheelVisualization:
    """Configuration for visualizing the wheel."""

    diameter: float
    width: float
    num_points: int = 50
    color: str = "#444444"
    alpha: float = 0.75
    linestyle: str = "-"


class SuspensionVisualizer:
    """Renders suspension geometry to matplotlib 3D axes."""

    def draw_links(
        self,
        ax,
        positions: dict[PointKey, Point3],
    ) -> list:
        """
        Draws all links and returns a list of matplotlib line artists.
        """
        link_artists = []
        for link in self.links:
            pts = np.array([positions[pid].data for pid in link.points])
            (line,) = ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color=link.color,
                linewidth=link.linewidth,
                linestyle=link.linestyle,
                marker=link.marker,
                markersize=link.markersize,
                label=link.label,
            )
            link_artists.append(line)
        return link_artists

    def update_links(
        self,
        artists: list,
        positions: dict[PointKey, Point3],
    ) -> None:
        """
        Update all link artists with new geometry for animation.
        """
        for line, link in zip(artists, self.links):
            pts = np.array([positions[pid].data for pid in link.points])
            line.set_data(pts[:, 0], pts[:, 1])
            line.set_3d_properties(pts[:, 2])

    @staticmethod
    def get_band_endpoints(
        wheel_inboard: np.ndarray,
        wheel_outboard: np.ndarray,
        e2: np.ndarray,
        e3: np.ndarray,
        num_bands: int,
        radius: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the endpoints for cross-tire bands at true radial angles.

        Returns two arrays of shape (num_bands, 3): inboard and outboard endpoints.
        """
        thetas = np.linspace(0, 2 * np.pi, num_bands, endpoint=False)
        band_inboard = np.array(
            [
                wheel_inboard + radius * (np.cos(theta) * e2 + np.sin(theta) * e3)
                for theta in thetas
            ]
        )
        band_outboard = np.array(
            [
                wheel_outboard + radius * (np.cos(theta) * e2 + np.sin(theta) * e3)
                for theta in thetas
            ]
        )
        return band_inboard, band_outboard

    def __init__(
        self,
        links: list[LinkVisualization],
        wheel_config: WheelVisualization,
        wheel_anchors: list[WheelAnchors] | None = None,
    ):
        self.links = links
        self.wheel_config = wheel_config
        self.wheel_anchors = wheel_anchors or [WheelAnchors.for_corner()]

    def draw_wheel(
        self,
        ax,
        positions: dict[PointKey, Point3],
        num_bands: int = 48,
    ) -> list[dict]:
        """Draw every configured wheel and return their artists."""
        return [
            self._draw_single_wheel(ax, positions, anchors, num_bands)
            for anchors in self.wheel_anchors
        ]

    def _draw_single_wheel(
        self,
        ax,
        positions: dict[PointKey, Point3],
        anchors: WheelAnchors,
        num_bands: int,
    ) -> dict:
        """
        Draws a 3D wheel representation and returns the matplotlib artists.

        Returns dict with 'rims' (list of 3 lines) and 'bands' (list of lines).
        """
        # Extract raw arrays for matplotlib drawing math.
        wheel_center = positions[anchors.center].data
        wheel_inboard = positions[anchors.inboard].data
        wheel_outboard = positions[anchors.outboard].data
        axle_vector = (
            positions[anchors.axle_outboard].data - positions[anchors.axle_inboard].data
        )

        axle_vector = axle_vector / np.linalg.norm(axle_vector)

        e1 = axle_vector
        e2 = np.array([1, 0, 0])
        if np.abs(np.dot(e1, e2)) > 0.9:
            e2 = np.array([0, 1, 0])
        e2 = e2 - np.dot(e2, e1) * e1
        e2 = e2 / np.linalg.norm(e2)

        e3 = np.cross(e1, e2)

        theta = np.linspace(0, 2 * np.pi, self.wheel_config.num_points)
        radius = self.wheel_config.diameter / 2

        rim_points_center = np.zeros((self.wheel_config.num_points, 3))
        rim_points_inboard = np.zeros((self.wheel_config.num_points, 3))
        rim_points_outboard = np.zeros((self.wheel_config.num_points, 3))

        for i, angle in enumerate(theta):
            rim_points_center[i] = wheel_center + radius * (
                np.cos(angle) * e2 + np.sin(angle) * e3
            )
            rim_points_inboard[i] = wheel_inboard + radius * (
                np.cos(angle) * e2 + np.sin(angle) * e3
            )
            rim_points_outboard[i] = wheel_outboard + radius * (
                np.cos(angle) * e2 + np.sin(angle) * e3
            )

        rim_lines = []
        rim_lines.append(
            ax.plot(
                rim_points_center[:, 0],
                rim_points_center[:, 1],
                rim_points_center[:, 2],
                color=self.wheel_config.color,
                alpha=0.25,
                linestyle=self.wheel_config.linestyle,
            )[0]
        )
        rim_lines.append(
            ax.plot(
                rim_points_inboard[:, 0],
                rim_points_inboard[:, 1],
                rim_points_inboard[:, 2],
                color=self.wheel_config.color,
                alpha=self.wheel_config.alpha,
                linestyle=self.wheel_config.linestyle,
            )[0]
        )
        rim_lines.append(
            ax.plot(
                rim_points_outboard[:, 0],
                rim_points_outboard[:, 1],
                rim_points_outboard[:, 2],
                color=self.wheel_config.color,
                alpha=self.wheel_config.alpha,
                linestyle=self.wheel_config.linestyle,
            )[0]
        )

        band_inboard, band_outboard = self.get_band_endpoints(
            wheel_inboard, wheel_outboard, e2, e3, num_bands, radius
        )
        band_lines = []
        for i in range(num_bands):
            band_lines.append(
                ax.plot(
                    [band_inboard[i, 0], band_outboard[i, 0]],
                    [band_inboard[i, 1], band_outboard[i, 1]],
                    [band_inboard[i, 2], band_outboard[i, 2]],
                    color=self.wheel_config.color,
                    alpha=self.wheel_config.alpha,
                    linestyle=self.wheel_config.linestyle,
                )[0]
            )
        return {"rims": rim_lines, "bands": band_lines}

    def update_wheel(
        self,
        artists: list[dict],
        positions: dict[PointKey, Point3],
        num_bands: int = 36,
    ) -> None:
        """Update every configured wheel's artists."""
        for wheel_artists, anchors in zip(artists, self.wheel_anchors):
            self._update_single_wheel(wheel_artists, positions, anchors, num_bands)

    def _update_single_wheel(
        self,
        artists: dict,
        positions: dict[PointKey, Point3],
        anchors: WheelAnchors,
        num_bands: int,
    ) -> None:
        """
        Update the wheel artists with new geometry for animation.
        """
        # Extract raw arrays for matplotlib drawing math.
        wheel_center = positions[anchors.center].data
        wheel_inboard = positions[anchors.inboard].data
        wheel_outboard = positions[anchors.outboard].data
        axle_vector = (
            positions[anchors.axle_outboard].data - positions[anchors.axle_inboard].data
        )

        axle_vector = axle_vector / np.linalg.norm(axle_vector)

        e1 = axle_vector
        e2 = np.array([1, 0, 0])
        if np.abs(np.dot(e1, e2)) > 0.9:
            e2 = np.array([0, 1, 0])
        e2 = e2 - np.dot(e2, e1) * e1
        e2 = e2 / np.linalg.norm(e2)

        e3 = np.cross(e1, e2)

        theta = np.linspace(0, 2 * np.pi, self.wheel_config.num_points)
        radius = self.wheel_config.diameter / 2

        rim_points_center = np.zeros((self.wheel_config.num_points, 3))
        rim_points_inboard = np.zeros((self.wheel_config.num_points, 3))
        rim_points_outboard = np.zeros((self.wheel_config.num_points, 3))

        for i, angle in enumerate(theta):
            rim_points_center[i] = wheel_center + radius * (
                np.cos(angle) * e2 + np.sin(angle) * e3
            )
            rim_points_inboard[i] = wheel_inboard + radius * (
                np.cos(angle) * e2 + np.sin(angle) * e3
            )
            rim_points_outboard[i] = wheel_outboard + radius * (
                np.cos(angle) * e2 + np.sin(angle) * e3
            )

        # Update rim lines.
        artists["rims"][0].set_data(rim_points_center[:, 0], rim_points_center[:, 1])
        artists["rims"][0].set_3d_properties(rim_points_center[:, 2])
        artists["rims"][1].set_data(rim_points_inboard[:, 0], rim_points_inboard[:, 1])
        artists["rims"][1].set_3d_properties(rim_points_inboard[:, 2])
        artists["rims"][2].set_data(
            rim_points_outboard[:, 0], rim_points_outboard[:, 1]
        )
        artists["rims"][2].set_3d_properties(rim_points_outboard[:, 2])

        # Update band lines.
        band_inboard, band_outboard = self.get_band_endpoints(
            wheel_inboard, wheel_outboard, e2, e3, num_bands, radius
        )
        for i in range(num_bands):
            artists["bands"][i].set_data(
                [band_inboard[i, 0], band_outboard[i, 0]],
                [band_inboard[i, 1], band_outboard[i, 1]],
            )
            artists["bands"][i].set_3d_properties(
                [band_inboard[i, 2], band_outboard[i, 2]]
            )

    @staticmethod
    def get_band_indices(num_points: int, num_bands: int) -> np.ndarray:
        """
        Return indices for equally spaced cross-tire bands (not spokes).

        Ensures bands are radially spaced regardless of num_points.
        """
        step = max(1, num_points // num_bands)
        indices = np.arange(0, num_points, step)
        # Ensure exactly num_bands bands (may need to trim or pad)
        if len(indices) > num_bands:
            indices = indices[:num_bands]
        elif len(indices) < num_bands:
            # Pad by repeating last index if needed.
            indices = np.pad(indices, (0, num_bands - len(indices)), "edge")
        return indices
