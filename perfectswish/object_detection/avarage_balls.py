import typing
from typing import List
import numpy as np

AVARGAE_FRANE = 10
RADIUS = 5


def average_locations(locations: np.array([]), last_locations: np.array([])) -> np.array([]):
    if len(last_locations) < 10:
        locations.append(last_locations)
    else:
        locations.pop(0)
        locations.append(last_locations)
    average_locations_balls = np.array([])
    return average_locations_balls
