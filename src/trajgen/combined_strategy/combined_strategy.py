from ..config import Config
from ..trajectory import Trajectory


class CombinedStrategy:
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, id: int) -> Trajectory:
        raise NotImplementedError("CombinedStrategy is an abstract base class.")
