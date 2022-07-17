import numpy as np
from typing import List
import math


class OcclusionLine:
    """
    Class that defines a straight line used in the occlusion detection algorithm.
    It is defined by 2 endpoints.
    """

    def __init__(self, p1: np.array, p2: np.array):

        if isinstance(p1, List):
            p1 = np.array(p1)
        if isinstance(p2, List):
            p2 = np.array(p2)

        assert p1.size == 2 and p2.size == 2

        self.points = (p1, p2)
        self.length = self.get_length()
        self.slope = self.get_slope()

    def get_direction(self):
        """
        Compute the direction of the line going from the first endpoint to the second.
        """

        return self.points[-1] - self.points[0]

    def get_length(self):
        """
        Compute the length of the line going from the first endpoint
        to the second.
        """
        (x1, y1), (x2, y2) = self.points

        delta_x = x2 - x1
        delta_y = y2 - y1
        return math.sqrt(delta_x ** 2 + delta_y ** 2)

    def angle_between(self, other_line: "OcclusionLine"):
        """
        Args:
            other_line: second line we want the angle from.
        """
        dot = np.dot(self.get_vector(), other_line.get_vector())
        return np.arccos(np.clip(dot / (self.length * other_line.length), -1, 1))

    def get_vector(self):
        (x1, y1), (x2, y2) = self.points
        return np.array([x2-x1, y2-y1])

    def get_slope(self):
        # Get the coordinates of the endpoints for the line.
        (x1, y1), (x2, y2) = self.points

        if x1 - x2 == 0:
            return math.inf

        return (y1 - y2) / (x1 - x2)

    def get_extended_point(self, distance: int, point: np.array):
        """
        Get the coordinates of a point as if it was at "distance"  meters from where it is now the current line.

        Args:
            distance:  how far from where it is now along the lane the point should be
            point:     the current position of the point we want to translate
        """

        # Use basic algebra formulas to find how much we need to translate the point for.
        delta_x = math.sqrt(distance ** 2 / (1 + self.slope ** 2))
        delta_y = math.sqrt(distance ** 2 - delta_x ** 2)

        delta_x = -delta_x if self.get_direction()[0] < 0 else delta_x
        delta_y = -delta_y if self.get_direction()[1] < 0 else delta_y

        # Find the new coordinates of the translated point.
        x, y = point
        x += delta_x
        y += delta_y

        return x, y
