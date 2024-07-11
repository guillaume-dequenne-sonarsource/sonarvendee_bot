import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass


@dataclass
class Location:
    x: float
    y: float

@dataclass
class Heading:
    angle: float  # in radians

@dataclass
class Speed:
    x: float
    y: float

@dataclass
class WindMagnitude:
    x: float
    y: float


class Instruction:
    def __init__(self, location: Location = None, heading: Heading = None, speed: Speed = None):
        self.location = location
        self.heading = heading
        self.speed = speed

class CheckPoint:
    def __init__(self, x: float, y: float, radius: float):
        self.location = Location(x, y)
        self.radius = radius
        self.was_reached = False

    def set_reached(self, x: float, y: float) -> bool:
        distance = np.sqrt((x - self.location.x)**2 + (y - self.location.y)**2)
        if distance <= self.radius:
            self.was_reached = True
            return True
        else:
            raise Exception("Coordinates are not within the checkpoint radius")
