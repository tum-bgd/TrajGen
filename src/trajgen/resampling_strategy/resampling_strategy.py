from typing import Protocol
from ..config import Config
from ..trajectory import Trajectory


class ResamplingStrategy(Protocol):
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, trajectory: Trajectory):
        raise NotImplementedError("ResamplingStrategy is an abstract base class.")
