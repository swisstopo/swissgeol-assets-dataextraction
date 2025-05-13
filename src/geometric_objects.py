"""
Most of this code is from the swissgeol-boreholes-dataextraction repo (https://github.com/swisstopo/swissgeol-boreholes-dataextraction)
"""
from __future__ import annotations

import numpy as np
import logging
from dataclasses import dataclass
from math import atan2, degrees

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Point:
    """Class to represent a point in 2D space.
        Code copied from swissgeol-boreholes-dataextraction repo. """

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
        object.__setattr__(self, 'length', self.start.distance_to(self.end))

    @property
    def line_angle(self) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return degrees(atan2(dy, dx)) % 180