"""Most of this code is from the swissgeol-boreholes-dataextraction repo (https://github.com/swisstopo/swissgeol-boreholes-dataextraction)."""

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
    def tuple(self) -> tuple[float, float]:
        return self.x, self.y

    def distance_to(self, point: Point) -> float:
        return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)


@dataclass(frozen=True)
class Line:
    """Class to represent a line in 2D space."""

    start: Point
    end: Point

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)

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

    def is_horizontal(self, horizontal_slope_tolerance) -> bool:
        """Checks if a line is horizontal."""
        return abs(self.slope) <= horizontal_slope_tolerance

    @property
    def slope(self) -> float:
        """Calculate the slope of the line."""
        return (self.end.y - self.start.y) / (self.end.x - self.start.x) if self.end.x - self.start.x != 0 else np.inf
