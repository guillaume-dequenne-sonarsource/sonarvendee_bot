import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Callable

from utils import Location, WindMagnitude, Instruction, Heading, Speed, CheckPoint

from player import Player

class Space2D:
    def __init__(self, x_limit: float = 1.0, y_limit: float = 1.0, dt: float = 0.01):
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.dt = dt
        self.obstacles: List[np.ndarray] = []
        self.players: Dict[str, Player] = {}
        self.player_states: Dict[str, Dict[str, Location | Heading | Speed]] = {}
        self.checkpoint: CheckPoint = None
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.wind_field = self._generate_wind_field()

    def _generate_wind_field(self):
        x = np.linspace(-self.x_limit, self.x_limit, 20)
        y = np.linspace(-self.y_limit, self.y_limit, 20)
        X, Y = np.meshgrid(x, y)
        
        # Generate wind field (mostly right to left, but with some variation)
        U = np.ones_like(X) * 0.8 + np.random.randn(*X.shape) * 0.02
        V = np.random.randn(*X.shape) * 0.02
        
        return X, Y, U, V

    def wind(self, loc: Location) -> WindMagnitude:
        # Interpolate wind at the given location
        x_idx = np.searchsorted(self.wind_field[0][0], loc.x) - 1
        y_idx = np.searchsorted(self.wind_field[1][:, 0], loc.y) - 1
        
        wind_x = self.wind_field[2][y_idx, x_idx]
        wind_y = self.wind_field[3][y_idx, x_idx]
        
        return WindMagnitude(wind_x, wind_y)

    def add_obstacle(self, boundary: List[Tuple[int, int]]):
        """Add an obstacle defined by its boundary points."""
        self.obstacles.append(np.array(boundary))

    def add_player(self, player_id: str, x: float, y: float, heading: float, speed: float):
        """Add a player to the space."""
        self.players[player_id] = Player()
        self.player_states[player_id] = {
            "location": Location(x, y),
            "heading": Heading(heading),
            "speed": Speed(speed * np.cos(heading), speed * np.sin(heading))
        }

    def set_checkpoint(self, checkpoint: CheckPoint):
        """Set the checkpoint in the space."""
        self.checkpoint = checkpoint

    def is_obstacle(self, point: np.ndarray) -> bool:
        """Check if a given point is within any obstacle."""
        for obstacle in self.obstacles:
            if self._point_in_polygon(point, obstacle):
                return True
        return False

    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm."""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if point[1] > min(p1y, p2y):
                if point[1] <= max(p1y, p2y):
                    if point[0] <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point[0] <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    # def step(self):
    #     """Step all players in the space."""
    #     for player_id, player in self.players.items():
    #         state = self.player_states[player_id]
    #         instruction = player.step(
    #             state["location"], state["heading"], state["speed"],
    #             self.wind, self.checkpoint.location, self.is_obstacle
    #         )
            
    #         if instruction.location:
    #             # Update player state based on instruction
    #             state["location"] = instruction.location
    #             state["heading"] = instruction.heading
    #             state["speed"] = instruction.speed
    #         else:
    #             # Apply wind to velocity
    #             wind = self.wind(state["location"])
                
    #             # Calculate the component of wind in the direction of player's heading
    #             heading_vector = np.array([np.cos(state["heading"].angle), np.sin(state["heading"].angle)])
    #             wind_vector = np.array([wind.x, wind.y])
    #             wind_component = np.dot(wind_vector, heading_vector) * heading_vector
                
    #             # Update velocity
    #             new_speed_x = state["speed"].x + wind_component[0]
    #             new_speed_y = state["speed"].y + wind_component[1]
    #             state["speed"] = Speed(new_speed_x, new_speed_y)
                
    #             # Update position based on new velocity
    #             new_x = state["location"].x + state["speed"].x * self.dt
    #             new_y = state["location"].y + state["speed"].y * self.dt
    #             new_location = Location(new_x, new_y)
                
    #             # Check if the new location is within bounds
    #             if abs(new_location.x) > self.x_limit or abs(new_location.y) > self.y_limit:
    #                 raise Exception(f"Player {player_id} has crossed the boundary of the Space!")
                
    #             state["location"] = new_location
            
    #         # Print player state for observability
    #         print(f"Player {player_id}:")
    #         print(f"  Position: ({state['location'].x:.4f}, {state['location'].y:.4f})")
    #         print(f"  Heading: {np.degrees(state['heading'].angle):.2f} degrees")
    #         print(f"  Speed: ({state['speed'].x:.4f}, {state['speed'].y:.4f})")
    #         print(f"  Wind: ({wind.x:.4f}, {wind.y:.4f})")
    #         print()
            
    #         # Check if player reached the checkpoint
    #         try:
    #             if self.checkpoint.set_reached(state["location"].x, state["location"].y):
    #                 print(f"Checkpoint at ({self.checkpoint.location.x:.4f}, {self.checkpoint.location.y:.4f}) reached!")
    #         except Exception:
    #             pass  # Not within radius, continue
    def _check_collision(self, current_location: Location, new_location: Location) -> Location:
        """Check if the path from current_location to new_location intersects any obstacle."""
        for obstacle in self.obstacles:
            for i in range(len(obstacle)):
                start = obstacle[i]
                end = obstacle[(i + 1) % len(obstacle)]
                intersection = self._line_intersection(
                    (current_location.x, current_location.y),
                    (new_location.x, new_location.y),
                    (start[0], start[1]),
                    (end[0], end[1])
                )
                if intersection:
                    # Return the point just before the intersection
                    return Location(intersection[0], intersection[1])
        return new_location

    def _line_intersection(self, line1_start, line1_end, line2_start, line2_end):
        """Helper function to find the intersection of two line segments."""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        A, B = line1_start, line1_end
        C, D = line2_start, line2_end
        
        if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
            t = ((A[0] - C[0]) * (C[1] - D[1]) - (A[1] - C[1]) * (C[0] - D[0])) / \
                ((A[0] - B[0]) * (C[1] - D[1]) - (A[1] - B[1]) * (C[0] - D[0]))
            return (A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1]))
        return None

    def step(self):
        """Step all players in the space."""
        for player_id, player in self.players.items():
            state = self.player_states[player_id]
            instruction = player.step(
                state["location"], state["heading"], state["speed"],
                self.wind, self.checkpoint.location, self.is_obstacle
            )
            
            if instruction.location:
                # Update player state based on instruction
                new_location = self._check_collision(state["location"], instruction.location)
                state["location"] = new_location
                state["heading"] = instruction.heading
                state["speed"] = instruction.speed
            else:
                # Apply wind to velocity
                wind = self.wind(state["location"])
                
                # Calculate the component of wind in the direction of player's heading
                heading_vector = np.array([np.cos(state["heading"].angle), np.sin(state["heading"].angle)])
                wind_vector = np.array([wind.x, wind.y])
                wind_component = np.dot(wind_vector, heading_vector) * heading_vector
                
                # Update velocity
                new_speed_x = state["speed"].x + wind_component[0]
                new_speed_y = state["speed"].y + wind_component[1]
                state["speed"] = Speed(new_speed_x, new_speed_y)
                
                # Update position based on new velocity
                new_x = state["location"].x + state["speed"].x * self.dt
                new_y = state["location"].y + state["speed"].y * self.dt
                new_location = Location(new_x, new_y)
                
                # Check for collisions
                new_location = self._check_collision(state["location"], new_location)
                
                # Check if the new location is within bounds
                if abs(new_location.x) > self.x_limit or abs(new_location.y) > self.y_limit:
                    raise Exception(f"Player {player_id} has crossed the boundary of the Space!")
                
                state["location"] = new_location
            
            # Print player state for observability
            print(f"Player {player_id}:")
            print(f"  Position: ({state['location'].x:.4f}, {state['location'].y:.4f})")
            print(f"  Heading: {np.degrees(state['heading'].angle):.2f} degrees")
            print(f"  Speed: ({state['speed'].x:.4f}, {state['speed'].y:.4f})")
            print(f"  Wind: ({wind.x:.4f}, {wind.y:.4f})")
            print()
            
            # Check if player reached the checkpoint
            try:
                if self.checkpoint.set_reached(state["location"].x, state["location"].y):
                    print(f"Checkpoint at ({self.checkpoint.location.x:.4f}, {self.checkpoint.location.y:.4f}) reached!")
            except Exception:
                pass  # Not within radius, continue

    def visualize(self):
        """Visualize the space with obstacles, players, checkpoint, wind, and current RRT."""
        self.ax.clear()
        sns.set_style("darkgrid")

        # Plot wind field
        self.ax.quiver(self.wind_field[0], self.wind_field[1], self.wind_field[2], self.wind_field[3], 
                       scale=1, color='lightblue', alpha=0.3)

        # Plot obstacles
        for obstacle in self.obstacles:
            polygon = plt.Polygon(obstacle, fill=True, facecolor='gray', edgecolor='black', alpha=0.5)
            self.ax.add_patch(polygon)

        # Plot checkpoint
        if self.checkpoint:
            color = 'green' if self.checkpoint.was_reached else 'red'
            circle = plt.Circle((self.checkpoint.location.x, self.checkpoint.location.y), self.checkpoint.radius, 
                                fill=False, edgecolor=color, linestyle='--', linewidth=2)
            self.ax.add_artist(circle)

        # Plot players
        for player_id, player in self.players.items():
            state = self.player_states[player_id]
            self.ax.plot(state["location"].x, state["location"].y, 'bo', markersize=10, label=f'Player {player_id}')
            
            # Plot player's heading
            heading_length = 0.1
            dx = heading_length * np.cos(state["heading"].angle)
            dy = heading_length * np.sin(state["heading"].angle)
            self.ax.arrow(state["location"].x, state["location"].y, dx, dy, 
                          head_width=0.02, head_length=0.02, fc='b', ec='b')

            # Plot player's speed
            speed_scale = 2.0  # Adjust this to make the speed arrow more visible
            self.ax.arrow(state["location"].x, state["location"].y, 
                          state["speed"].x * speed_scale, state["speed"].y * speed_scale, 
                          head_width=0.015, head_length=0.015, fc='r', ec='r', alpha=0.5)

        self.ax.set_xlim(-self.x_limit, self.x_limit)
        self.ax.set_ylim(-self.y_limit, self.y_limit)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_title("2D Space with Players, Obstacles, Checkpoint, Wind, and Optimized RRT")
        self.ax.legend()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)


def main():
    # Initialize the space
    space = Space2D(x_limit=1.0, y_limit=1.0, dt=0.01)

    # Add some sample obstacles (scaled down to fit in the new space)
    space.add_obstacle([(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)])
    space.add_obstacle([(-0.3, -0.3), (-0.1, -0.3), (-0.2, -0.1)])
    space.add_obstacle([(0.4, -0.2), (0.6, -0.4), (0.8, -0.2)])

    # Set the checkpoint
    checkpoint = CheckPoint(x=0.8, y=0.8, radius=0.05)
    space.set_checkpoint(checkpoint)

    # Add a player to the space
    space.add_player("player1", x=-0.8, y=-0.8, heading=np.pi/4, speed=0.1)

    # Set up interactive mode
    plt.ion()

    # Main loop
    try:
        for _ in range(1000):  # Keep the same number of steps
            space.step()
            space.visualize()
            
            # Check if checkpoint is reached
            if checkpoint.was_reached:
                print("Checkpoint reached! Simulation ending.")
                break
    except Exception as e:
        print(f"Simulation ended: {str(e)}")

    # Turn off interactive mode and show the final plot
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()