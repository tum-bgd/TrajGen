from typing import Protocol
from ..trajectory import Trajectory
from ..config import Config


class SpatialStrategy(Protocol):
    def __init__(self, config: Config):
        pass

    def __call__(self, id) -> Trajectory:
        pass
