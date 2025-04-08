from __future__ import annotations

import numpy as np
import cv2
import logging
from dataclasses import dataclass
from math import atan2, degrees
from typing import List, Set

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Point:
    """Class to represent a point in 2D space."""

    x: float
    y: float

    @property
    def tuple(self) -> (float, float):
        return self.x, self.y

    def distance_to(self, point: Point) -> float:
        return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)


@dataclass
class Line:
    """Class to represent a line in 2D space."""

    start: Point
    end: Point

    def __post_init__(self):
        if self.start.x > self.end.x:
            end = self.start
            self.start = self.end
            self.end = end

        self.length = self.start.distance_to(self.end)

    def distance_to(self, point: Point) -> float:
        return np.abs(
            (self.end.x - self.start.x) * (self.start.y - point.y)
            - (self.start.x - point.x) * (self.end.y - self.start.y)
        ) / np.sqrt((self.end.x - self.start.x) ** 2 + (self.end.y - self.start.y) ** 2)

    @property
    def line_angle(self) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return degrees(atan2(dy, dx)) % 180


@dataclass
class LineGroup:
    lines: List[Line]

    def __post_init__(self):
        self.endpoints: Set[Point] = set()
        for line in self.lines:
            self.endpoints.update([line.start, line.end])

    @property
    def average_angle(self) -> float:
        return np.mean([line.angle for line in self.lines])

    def is_connected_to(self, line: Line, angle_thresh=10, dist_thresh=10) -> bool:
        for pt in [line.start, line.end]:
            for ep in self.endpoints:
                if pt.distance_to(ep) < dist_thresh:
                    angle_diff = min(abs(line.line_angle - l.line_angle) % 180 for l in self.lines)
                    if angle_diff < angle_thresh or abs(angle_diff - 180) < angle_thresh:
                        return True
        return False

    def add_line(self, line: Line):
        self.lines.append(line)
        self.endpoints.update([line.start, line.end])

    def to_mask(self, shape: tuple) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        for line in self.lines:
            pt1 = tuple(map(int, line.start.tuple))
            pt2 = tuple(map(int, line.end.tuple))
            cv2.line(mask, pt1, pt2, 255, 1)
        # Connect endpoints to visually smooth the group
        endpoint_coords = [p.tuple for p in self.endpoints]
        for i in range(len(endpoint_coords)):
            for j in range(i + 1, len(endpoint_coords)):
                if np.linalg.norm(np.subtract(endpoint_coords[i], endpoint_coords[j])) < 10:
                    cv2.line(mask,
                             tuple(map(int, endpoint_coords[i])),
                             tuple(map(int, endpoint_coords[j])), 255, 1)
        return mask