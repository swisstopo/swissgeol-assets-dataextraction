"""Most of this code is from the swissgeol-boreholes-dataextraction repo.

(https://github.com/swisstopo/swissgeol-boreholes-dataextraction)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees

import numpy as np


@dataclass(frozen=True)
class Point:
    """Class to represent a point in 2D space.

    Code copied from swissgeol-boreholes-dataextraction repo.
    """

    x: float
    y: float

    @property
    def tuple(self) -> (float, float):
        return self.x, self.y

    def distance_to(self, point: Point) -> float:
        return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)


@dataclass(frozen=True)
class Line:
    """Class to represent a line in 2D space."""

    start: Point
    end: Point

    def __post_init__(self):
        object.__setattr__(self, "length", self.start.distance_to(self.end))

    @property
    def line_angle(self) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return degrees(atan2(dy, dx)) % 180

    def distance_to(self, point: Point) -> float:
        """Calculate the distance of a point to the (unbounded extension of the) line.

        Taken from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points

        Args:
            point (Point): The point to calculate the distance to.

        Returns:
            float: The distance of the point to the line.
        """
        if self.length == 0:
            return self.start.distance_to(point)
        else:
            return (
                np.abs(
                    (self.end.x - self.start.x) * (self.start.y - point.y)
                    - (self.start.x - point.x) * (self.end.y - self.start.y)
                )
                / self.length
            )

    # --------------------- copied from swissgeol-boreholes-dataextraction ------------------ #

    def distance_to_segment(self, point: Point) -> float:
        """Calculate the distance of a point to the line segment (bounded by endpoints).

        Uses vector projection with parametric line representation to find the closest
        point on the segment, then calculates Euclidean distance.

        Args:
            point (Point): The point to calculate the distance to.

        Returns:
            float: The distance of the point to the line segment.
        """
        # Vector from line start to point
        px, py = point.x - self.start.x, point.y - self.start.y
        # Line direction vector
        lx, ly = self.end.x - self.start.x, self.end.y - self.start.y

        line_length_sq = lx * lx + ly * ly
        if line_length_sq == 0:
            # Degenerate case: line is actually a point
            return ((point.x - self.start.x) ** 2 + (point.y - self.start.y) ** 2) ** 0.5

        # Project point onto line using dot product
        t = (px * lx + py * ly) / line_length_sq

        # Clamp t to [0, 1] to stay within segment bounds
        t = max(0, min(1, t))

        # Find closest point on segment
        proj_x = self.start.x + t * lx
        proj_y = self.start.y + t * ly

        # Calculate Euclidean distance
        return ((point.x - proj_x) ** 2 + (point.y - proj_y) ** 2) ** 0.5

    def point_near_segment(self, point: Point, threshold: float) -> bool:
        """Check if a point is within a threshold distance of the line segment.

        Args:
            point (Point): The point to check
            threshold (float): Distance threshold for proximity

        Returns:
            bool: True if the point is within threshold distance of the segment
        """
        return self.distance_to_segment(point) <= threshold

    def intersects_with(self, other: Line) -> bool:
        """Check if this line segment intersects with another line segment.

        Uses line-line intersection calculation with determinants.
        Reference: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

        Args:
            other (Line): The other line segment to check intersection with

        Returns:
            bool: True if line segments intersect, False otherwise
        """
        # Get line endpoints
        x1, y1 = self.start.x, self.start.y
        x2, y2 = self.end.x, self.end.y
        x3, y3 = other.start.x, other.start.y
        x4, y4 = other.end.x, other.end.y

        # Calculate line intersection using determinants
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        # Lines are parallel if denominator is 0
        if abs(denom) < 1e-10:
            return False

        # Calculate intersection parameters
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        # Lines intersect if both parameters are between 0 and 1
        return 0 <= t <= 1 and 0 <= u <= 1
