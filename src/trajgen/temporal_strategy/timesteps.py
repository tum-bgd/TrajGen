from ..trajectory import Trajectory
from ..config import Config


class ConstantTimeStepsTemporalStrategy:
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        length = len(trajectory.ls.coords)
        trajectory.set_time(
            list(
                range(
                    self.config.get_next_tmin(), length * self.time_step, self.time_step
                )
            )
        )
        return trajectory

    @staticmethod
    def get_requirements() -> dict:
        return {
            "get_next_tmin": {
                "short_name": "T Min",
                "type": "get_float_function",
                "default": 0.0,
                "default_mode": "fixed for dataset",
                "description": "Start time of the trajectory.",
                "optional": False,
            },
            "get_next_time_step": {
                "short_name": "Time Step",
                "type": "get_float_function",
                "default": 1.0,
                "default_mode": "fixed for dataset",
                "description": "Fixed time interval between consecutive points.",
                "optional": False,
            },
        }


class VariableTimeStepsTemporalStrategy:
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        length = len(trajectory.ls.coords)
        if length == 0:
            trajectory.set_time([])
            return trajectory

        time_steps = [self.config.get_next_time_step() for _ in range(length - 1)]
        t_min = self.config.get_next_tmin()
        times = [t_min]

        # Accumulate time steps
        for i in range(1, length):
            times.append(times[i - 1] + time_steps[i - 1])

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
                "description": "Start time of the trajectory.",
                "optional": False,
            },
            "get_next_time_step": {
                "short_name": "Time Step",
                "type": "get_float_function",
                "default": 1.0,
                "description": "Time interval between consecutive points.",
                "optional": False,
            },
        }
