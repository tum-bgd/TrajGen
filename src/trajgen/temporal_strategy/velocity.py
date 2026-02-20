from ..trajectory import Trajectory
from ..config import Config


class VelocityTemporalStrategy:
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        length = len(trajectory.ls.coords)
        if length == 0:
            trajectory.set_time([])
            return trajectory
        time_stamps = [self.config.get_next_tmin()]
        for i in range(length - 1):
            spatial_length = self.config.distance_function(
                trajectory.ls.coords[i], trajectory.ls.coords[i + 1]
            )
            velocity = self.config.get_next_velocity()
            if velocity == 0:
                time_stamps.append(
                    time_stamps[-1]
                )  # If velocity is zero, we can't move, so we keep the same timestamp
                continue
            t = spatial_length / velocity
            time_stamps.append(t + time_stamps[-1])
        trajectory.set_time(time_stamps)
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
            "get_next_velocity": {
                "short_name": "Velocity",
                "type": "get_float_function",
                "default": 1.0,
                "description": "Movement speed (spatial units per time unit).",
                "optional": False,
            },
        }
