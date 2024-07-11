import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Callable

from utils import Location, WindMagnitude, Instruction, Heading, Speed, CheckPoint

from player import Player

class Space2D:
    def __init__(self, x_limit: int = 1000, y_limit: int = 1000, dt: float = 0.1):
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
        U = -np.ones_like(X) * 10 + np.random.randn(*X.shape) * 2
        V = np.random.randn(*X.shape) * 2
        
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
                state["location"] = instruction.location
                state["heading"] = instruction.heading
                state["speed"] = instruction.speed
            else:
                # No path found or goal reached, continue with current velocity affected by wind
                wind = self.wind(state["location"])
                new_x = state["location"].x + state["speed"].x * self.dt + wind.x * self.dt
                new_y = state["location"].y + state["speed"].y * self.dt + wind.y * self.dt
                state["location"] = Location(new_x, new_y)
            
            # Check if player reached the checkpoint
            try:
                if self.checkpoint.set_reached(state["location"].x, state["location"].y):
                    print(f"Checkpoint at ({self.checkpoint.location.x}, {self.checkpoint.location.y}) reached!")
            except Exception:
                pass  # Not within radius, continue

    def visualize(self):
        """Visualize the space with obstacles, players, checkpoint, wind, and current RRT."""
        self.ax.clear()
        sns.set_style("darkgrid")

        # Plot wind field
        self.ax.quiver(self.wind_field[0], self.wind_field[1], self.wind_field[2], self.wind_field[3], 
                       scale=500, color='lightblue', alpha=0.3)

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
            heading_length = 50
            dx = heading_length * np.cos(state["heading"].angle)
            dy = heading_length * np.sin(state["heading"].angle)
            self.ax.arrow(state["location"].x, state["location"].y, dx, dy, 
                          head_width=20, head_length=20, fc='b', ec='b')

            # Plot player's speed
            self.ax.arrow(state["location"].x, state["location"].y, state["speed"].x, state["speed"].y, 
                          head_width=15, head_length=15, fc='r', ec='r', alpha=0.5)

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
    space = Space2D(dt=0.1)

    # Add some sample obstacles
    space.add_obstacle([(100, 100), (200, 100), (200, 200), (100, 200)])
    space.add_obstacle([(-300, -300), (-100, -300), (-200, -100)])
    space.add_obstacle([(400, -200), (600, -400), (800, -200)])

    # Set the checkpoint
    checkpoint = CheckPoint(x=800, y=800, radius=50)
    space.set_checkpoint(checkpoint)

    # Add a player to the space
    space.add_player("player1", x=-800, y=-800, heading=np.pi/4, speed=50)

    # Set up interactive mode
    plt.ion()

    # Main loop
    for _ in range(1000):  # Increased steps to give more time to reach the checkpoint
        space.step()
        space.visualize()
        
        # Check if checkpoint is reached
        if checkpoint.was_reached:
            print("Checkpoint reached! Simulation ending.")
            break

    # Turn off interactive mode and show the final plot
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()