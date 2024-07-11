import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass

from utils import Location, WindMagnitude, Instruction, Heading, Speed

class Player:
    def __init__(self):
        self.goal_checkpoints = []

    def step(self, location: Location, heading: Heading, speed: Speed, 
             wind: Callable[[Location], WindMagnitude], goal: Location, 
             is_obstacle: Callable[[Location], bool]) -> Instruction:
        return Instruction()  # Return empty instruction when goal is reached
