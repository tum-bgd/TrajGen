from typing import Protocol
from ..trajectory import Trajectory
from ..config import Config


class TemporalStrategy(Protocol):
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        raise NotImplementedError("TemporalStrategy is an abstract base class.")
