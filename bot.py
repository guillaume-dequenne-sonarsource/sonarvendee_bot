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
        self.course = [
            Checkpoint(latitude=47.197942594203916, longitude=-9.169403579504642, radius=50),

            Checkpoint(latitude=58.95165241742833, longitude=-52.89744360710141, radius=50),
            Checkpoint(latitude=69.8709884582048, longitude=-60.98338106650846, radius=50),
            Checkpoint(latitude=74.23674388960148, longitude=-75.57322474326467, radius=50),
            Checkpoint(latitude=74.17855297867567, longitude=-90.39165705628501, radius=50),
            # Part 2
            Checkpoint(latitude=74.38348380993342, longitude=-92.036196394819, radius=50),
            Checkpoint(latitude=74.33662921432835, longitude=-94.41891964096959, radius=50),
            Checkpoint(latitude=74.33483234173502, longitude=-97.88878428845919, radius=50),
            Checkpoint(latitude=74.56718603322253, longitude=-99.88243839278059, radius=50),

            Checkpoint(latitude=73.96792966946977, longitude=-113.92469720338893, radius=5),
            Checkpoint(latitude=75.24424400442808, longitude=-125.01981521992485, radius=5),

            Checkpoint(latitude=72.45254474089015, longitude=-128.32639874145934, radius=5),
            Checkpoint(latitude=70.76258454578803, longitude=-130.74896646950896, radius=5),
            # Beaufort
            Checkpoint(latitude=70.17782624628336, longitude=-140.5161495333721, radius=5),
            Checkpoint(latitude=71.53250684339804, longitude=-155.055755226562, radius=5),

            # Out of ice

            Checkpoint(latitude=70.99321268200681, longitude=-168.6924898862142, radius=50),
            # Checkpoint(latitude=66.109528309295,   longitude=-168.79539477667726, radius=50),
            # Diomede
            Checkpoint(latitude=65.7374390789708, longitude=-168.51149130625583, radius=50),
            # St Lawrence
            Checkpoint(latitude=63.11541756604175, longitude=-167.8427966130406, radius=50),
            Checkpoint(latitude=60.23898250534194, longitude=-171.29089811272792, radius=50),
            # Atka
            Checkpoint(latitude=52.41485604173362, longitude=-171.79266018799007, radius=20),

            # Checkpoint 1 (not needed anymore)
            # Checkpoint(latitude=2.806318, longitude=-168.943864, radius=1990.0),
            # Checkpoint(latitude=-3.0818639154250973, longitude=-148.41895620946707, radius=50),
            # Pacific traverse
            # Samoa
            Checkpoint(latitude=-14.541275955922021, longitude=-167.85683625199763, radius=50),
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
            # Portugal 1
            Checkpoint(latitude=36.896350644999686, longitude=-9.467894938988914, radius=5),
            # Portugal 2
            Checkpoint(latitude=38.67414010310592, longitude=-10.544215264829893, radius=5),
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
        # Go through all checkpoints and find the next one to reach
        for ch in self.course:
            # Compute the distance to the checkpoint
            dist = distance_on_surface(
                longitude1=longitude,
                latitude1=latitude,
                longitude2=ch.longitude,
                latitude2=ch.latitude,
            )
            # Consider slowing down if the checkpoint is close
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
