import numpy as np
from typing import Callable, List, Optional, Tuple
import random

from vendeeglobe import config
from vendeeglobe.core import Checkpoint, Heading, Instructions, Location
from vendeeglobe.utils import distance_on_surface


class Node:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent = None

def rrt_path(current_position: Tuple[float, float], goal_position: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    def distance(p1: Node, p2: Node) -> float:
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def random_point() -> Node:
        # Generate a random point within the map bounds
        # You may need to adjust these bounds based on your world map
        x = np.random.uniform(-180, 180)
        y = np.random.uniform(-90, 90)
        return Node(x, y)

    def nearest_node(nodes: List[Node], point: Node) -> Node:
        return min(nodes, key=lambda n: distance(n, point))

    def steer(from_node: Node, to_node: Node, max_distance: float) -> Node:
        d = distance(from_node, to_node)
        if d > max_distance:
            theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
            x = from_node.x + max_distance * np.cos(theta)
            y = from_node.y + max_distance * np.sin(theta)
            return Node(x, y)
        return to_node

    def is_collision_free(node1: Node, node2: Node) -> bool:
        # Implement collision checking using the world_map function
        # This is a placeholder and needs to be implemented
        return True

    start_node = Node(current_position[0], current_position[1])
    goal_node = Node(goal_position[0], goal_position[1])
    nodes = [start_node]
    max_iterations = 1000
    step_size = 1.0  # Adjust based on your map scale

    for _ in range(max_iterations):
        rand_point = random_point()
        nearest = nearest_node(nodes, rand_point)
        new_node = steer(nearest, rand_point, step_size)
        
        if is_collision_free(nearest, new_node):
            new_node.parent = nearest
            nodes.append(new_node)
            
            if distance(new_node, goal_node) < step_size:
                # Path found
                path = []
                current = new_node
                while current.parent:
                    path.append((current.x, current.y))
                    current = current.parent
                path.append((start_node.x, start_node.y))
                path.reverse()
                
                if len(path) > 1:
                    next_point = path[1]
                    direction = (next_point[0] - current_position[0], 
                                 next_point[1] - current_position[1])
                    return direction
    
    return None  # No path found


class Bot:
    def __init__(self):
        self.team = "RRT"
        self.course = [
                    Checkpoint(latitude=18.462766447612122, longitude=-68.10976042183249, radius=50),
                    Checkpoint(latitude=17.222395178619355, longitude=-68.3989848264081, radius=50),
                    Checkpoint(latitude=10.088911008694621, longitude=-80.29453551355418, radius=5),
                    Checkpoint(latitude=8.676374405720495, longitude=-79.36526262255794, radius=5),
                    Checkpoint(latitude=8.676374405720495, longitude=-79.36526262255794, radius=5),
                    # after panama
                    Checkpoint(latitude=7.107329354230183, longitude=-79.48386411964532, radius=5),
                    Checkpoint(latitude=6.583764969268285, longitude=-80.64925346079656, radius=5),
                    # Checkpoint 1?
                    Checkpoint(latitude=2.806318, longitude=-168.943864, radius=1990.0),
                    Checkpoint(latitude=9.0800, longitude=-79.6800, radius=50),
                    Checkpoint(
                        latitude=config.start.latitude,
                        longitude=config.start.longitude,
                        radius=5,
                    ),
                ]
        
        self.goal = []

    def run(
        self,
        t: float,
        dt: float,
        longitude: float,
        latitude: float,
        heading: float,
        speed: float,
        vector: np.ndarray,
        forecast: Callable,
        world_map: Callable,
    ) -> Instructions:
        instructions = Instructions()
        
        # Find the next unreached checkpoint
        next_checkpoint = next((ch for ch in self.course if not ch.reached), None)
        
        if next_checkpoint:
            current_pos = (longitude, latitude)
            goal_pos = (next_checkpoint.longitude, next_checkpoint.latitude)
            
            direction = rrt_path(current_pos, goal_pos)
            
            if direction:
                # Convert direction to heading
                new_heading = np.arctan2(direction[1], direction[0])
                instructions.heading = Heading(np.degrees(new_heading) % 360)
            else:
                # If no path found, fallback to direct route
                instructions.location = Location(longitude=next_checkpoint.longitude, 
                                                latitude=next_checkpoint.latitude)
        
        instructions.sail = 1.0  # Full sail by default
        
        return instructions