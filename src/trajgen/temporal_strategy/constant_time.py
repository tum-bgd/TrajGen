from ..trajectory import Trajectory
from ..config import Config


class ConstantTemporalStrategy:
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        length = len(trajectory.ls.coords)
        if length == 0:
            trajectory.set_time([])
            return trajectory

        t_min = self.config.get_next_tmin()
        times = [t_min] * length
        trajectory.set_time(times)
        return trajectory

    @staticmethod
    def get_requirements() -> dict:
        return {
            "get_next_tmin": {
                "short_name": "T Min",
                "type": "get_float_function",
                "default": 0.0,
                "default_mode": "fixed for dataset",
                "description": "Constant time value assigned to every point.",
                "optional": False,
            },
        }
