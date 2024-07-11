# SPDX-License-Identifier: BSD-3-Clause

# flake8: noqa F401
from collections.abc import Callable

import numpy as np

from vendeeglobe import (
    Checkpoint,
    Heading,
    Instructions,
    Location,
    Vector,
    config,
)
from vendeeglobe.utils import distance_on_surface, wind_force, goto


class Bot:
    """
    This is the ship-controlling bot that will be instantiated for the competition.
    """

    def __init__(self):
        self.team = "SonarVendee"  # This is your team name
        # This is the course that the ship has to follow
        self.course = [
            Checkpoint(latitude=18.462766447612122, longitude=-68.10976042183249, radius=50),
            Checkpoint(latitude=17.222395178619355, longitude=-68.3989848264081, radius=50),
            Checkpoint(latitude=10.088911008694621, longitude=-80.29453551355418, radius=5),
            Checkpoint(latitude=8.676374405720495, longitude=-79.36526262255794, radius=5),
            Checkpoint(latitude=8.676374405720495, longitude=-79.36526262255794, radius=5),
            # after Panama
            Checkpoint(latitude=7.107329354230183, longitude=-79.48386411964532, radius=5),
            Checkpoint(latitude=6.583764969268285, longitude=-80.64925346079656, radius=5),
            # Checkpoint 1
            Checkpoint(latitude=2.806318, longitude=-168.943864, radius=1990.0),
            # Pacific traverse
            # Samoa
            Checkpoint(latitude=-14.541275955922021, longitude=-167.85683625199763, radius=50),
            # Fidji
            #Checkpoint(latitude=-19.163364543766544, longitude=-176.28392251307213, radius=50),
            # Australia?
            #Checkpoint(latitude=-43.26040352398038, longitude=150.33252175615877, radius=50),
            # South Tasmania
            Checkpoint(latitude=-47.26099388314912, longitude=144.81303151983454, radius=50),
            # Checkpoint 2
            Checkpoint(latitude=-15.668984, longitude=77.674694, radius=1190.0),
            # British Indian Ocean
            Checkpoint(latitude=-7.438221836214018, longitude=70.27045362725342, radius=50),
            # Oman
            Checkpoint(latitude=14.6257185306258, longitude=54.44599959294399, radius=50),
            # Djibouti
            Checkpoint(latitude=11.810396214998173, longitude=44.0196685419259, radius=5),
            # Red sea
            Checkpoint(latitude=22.15816378327582, longitude=37.719390420875136, radius=5),
            # Suez 1
            Checkpoint(latitude=27.957840061198073, longitude=33.65266339253563, radius=5),
            # Suez 2
            Checkpoint(latitude=29.541432940288455, longitude=32.598315866479375, radius=5),
            # Out of Suez
            Checkpoint(latitude=32.600870077923386, longitude=32.31353749546453, radius=5),
            # Malta - Sicily
            Checkpoint(latitude=36.398679550371114, longitude=14.422794557812782, radius=5),
            # Sardinia
            Checkpoint(latitude=37.80578152405002, longitude=8.753946250569633, radius=5),
            # Before Gibraltar
            Checkpoint(latitude=36.031916008449734, longitude=-4.306423912720327, radius=5),
            # After Gibraltar
            Checkpoint(latitude=35.92765496409566, longitude=-6.373354034830671, radius=5),
            # Portugal
            Checkpoint(latitude=36.896350644999686, longitude=-9.467894938988914, radius=5),
            # Spain North
            Checkpoint(latitude=43.70603724998247, longitude=-9.794633652008832, radius=5),
            # Return
            Checkpoint(
                latitude=config.start.latitude,
                longitude=config.start.longitude,
                radius=5,
            ),
        ]

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
        """
        This is the method that will be called at every time step to get the
        instructions for the ship.

        Parameters
        ----------
        t:
            The current time in hours.
        dt:
            The time step in hours.
        longitude:
            The current longitude of the ship.
        latitude:
            The current latitude of the ship.
        heading:
            The current heading of the ship.
        speed:
            The current speed of the ship.
        vector:
            The current heading of the ship, expressed as a vector.
        forecast:
            Method to query the weather forecast for the next 5 days.
            Example:
            current_position_forecast = forecast(
                latitudes=latitude, longitudes=longitude, times=0
            )
        world_map:
            Method to query map of the world: 1 for sea, 0 for land.
            Example:
            current_position_terrain = world_map(
                latitudes=latitude, longitudes=longitude
            )

        Returns
        -------
        instructions:
            A set of instructions for the ship. This can be:
            - a Location to go to
            - a Heading to point to
            - a Vector to follow
            - a number of degrees to turn Left
            - a number of degrees to turn Right

            Optionally, a sail value between 0 and 1 can be set.
        """
        # Initialize the instructions
        instructions = Instructions()

        # TODO: Remove this, it's only for testing =================
        current_position_forecast = forecast(
            latitudes=latitude, longitudes=longitude, times=0
        )
        current_position_terrain = world_map(latitudes=latitude, longitudes=longitude)
        # ===========================================================
        target_checkpoint: Checkpoint = self.get_target_checkpoint(latitude, longitude)

        # Go through all checkpoints and find the next one to reach
        follow_the_wind = True
        search_grid = create_grid(
            start_lat=latitude,
            start_lon=longitude,
            end_lat=target_checkpoint.latitude,
            end_lon=target_checkpoint.longitude,
            deviation=5.0,
            num_points=100
        )
        #plot_grid(search_grid)
        def wind_speed_function(lat, lon):
            return forecast(latitudes=lat, longitudes=lon, times=0)
        wind_grid = create_wind_speed_grid(search_grid, wind_speed_function)
        if follow_the_wind:
            #instructions.vector = Vector(current_position_forecast[0], current_position_forecast[1])
            f1 = wind_force(vector, vector)
            f2 = wind_force(vector, np.array(current_position_forecast))
            heading_to_target = goto(Location(latitude=latitude, longitude=longitude), target_checkpoint)
            print(speed)
            vector_to_target = heading_to_vector(heading_to_target)
            instructions.sail = 1.0
            instructions.vector = Vector(vector_to_target[0], vector_to_target[1])

            return instructions

        for ch in self.course:
            # Compute the distance to the checkpoint
            dist = distance_on_surface(
                longitude1=longitude,
                latitude1=latitude,
                longitude2=ch.longitude,
                latitude2=ch.latitude,
            )
            # Consider slowing down if the checkpoint is close
            jump = dt * np.linalg.norm(speed)
            if dist < 2.0 * ch.radius + jump:
                instructions.sail = min(ch.radius / jump, 1)
            else:
                instructions.sail = 1.0
            # Check if the checkpoint has been reached
            if dist < ch.radius:
                ch.reached = True
            if not ch.reached:
                instructions.location = Location(
                    longitude=ch.longitude, latitude=ch.latitude
                )
                break

        return instructions

    def get_target_checkpoint(self, latitude, longitude):
        for ch in self.course:
            dist = distance_on_surface(
                longitude1=longitude,
                latitude1=latitude,
                longitude2=ch.longitude,
                latitude2=ch.latitude,
            )
            if dist < ch.radius:
                ch.reached = True
            if not ch.reached:
                return ch


def heading_to_vector(heading_degrees):
    # Convert heading from degrees to radians
    heading_radians = np.radians(heading_degrees)

    # Calculate the x and y components of the vector
    x = np.cos(heading_radians)
    y = np.sin(heading_radians)

    # Create a vector from the x and y components
    vector = np.array([x, y])

    return vector


def create_grid(start_lat, start_lon, end_lat, end_lon, num_points, deviation):
    # Create arrays of latitudes and longitudes between the start and end points
    latitudes = np.linspace(start_lat - deviation, end_lat + deviation, num_points)
    longitudes = np.linspace(start_lon - deviation, end_lon + deviation, num_points)

    # Create a grid of points around the straight line path
    grid_lat, grid_lon = np.meshgrid(latitudes, longitudes)
    grid = np.dstack((grid_lat, grid_lon))

    return grid


def create_wind_speed_grid(grid, wind_speed_function):
    # Get the shape of the grid
    shape = grid.shape

    # Create an empty grid of the same shape to store the wind speeds
    wind_speed_grid = np.empty(shape)

    # Iterate over the grid and calculate the wind speed at each point
    for i in range(shape[0]):
        for j in range(shape[1]):
            # Get the latitude and longitude of the current point
            lat, lon = grid[i, j]

            # Calculate the wind speed at the current point
            wind_speed = wind_speed_function(lat, lon)

            # Store the wind speed in the wind speed grid
            wind_speed_grid[i, j] = np.array(wind_speed)

    return wind_speed_grid


# Example usage:
def wind_speed_function(lat, lon):
    # This is a placeholder function. Replace it with your actual function.
    return lat + lon


def interpolate_points(point1, point2, num_steps):
    # Unpack the coordinates
    lat1, lon1 = point1
    lat2, lon2 = point2

    # Create arrays for the latitudes and longitudes
    latitudes = np.linspace(lat1, lat2, num_steps)
    longitudes = np.linspace(lon1, lon2, num_steps)

    return np.stack([latitudes, longitudes], axis=-1)


# def create_grid(path, num_steps):
#     # Compute the minimum and maximum latitudes and longitudes
#     min_lat = np.min(path[:, 0])
#     max_lat = np.max(path[:, 0])
#     min_lon = np.min(path[:, 1])
#     max_lon = np.max(path[:, 1])
#
#     # Compute the step sizes
#     lat_step = (max_lat - min_lat) / num_steps
#     lon_step = (max_lon - min_lon) / num_steps
#
#     # Create the grid
#     latitudes = np.arange(min_lat, max_lat + lat_step, lat_step)
#     longitudes = np.arange(min_lon, max_lon + lon_step, lon_step)
#     grid = np.meshgrid(latitudes, longitudes)
#
#     return grid