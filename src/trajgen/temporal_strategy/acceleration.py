from ..config import Config
from ..trajectory import Trajectory


class AccelerationTemporalStrategy:
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        length = len(trajectory.ls.coords)
        time_stamps = [self.config.get_next_tmin()]
        v0 = self.config.get_next_velocity()
        for i in range(length - 1):
            spatial_length = self.config.distance_function(
                trajectory.ls.coords[i], trajectory.ls.coords[i + 1]
            )
            a = self.config.get_next_acceleration()
            if a == 0:
                if v0 == 0:
                    time_stamps.append(
                        time_stamps[-1]
                    )  # If velocity and acceleration are zero, we can't move, so we keep the same timestamp
                else:
                    t = spatial_length / v0
                    time_stamps.append(t + time_stamps[-1])
                continue
            t = (-v0 + (v0**2 + 2 * a * spatial_length) ** 0.5) / a
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
                "short_name": "Initial Velocity",
                "type": "get_float_function",
                "default": 1.0,
                "description": "Initial movement speed.",
                "optional": False,
            },
            "get_next_acceleration": {
                "short_name": "Acceleration",
                "type": "get_float_function",
                "default": 0.0,
                "description": "Rate of velocity change per time unit.",
                "optional": False,
            },
        }
