from ..trajectory import Trajectory
from shapely.geometry import LineString
from datetime import timedelta
import numpy as np


class ConstantTemporalStepResampling:
    def __init__(self, config):
        self.config = config

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        step = self.config.get_next_time_step_size()  # e.g., 60.0 seconds
        if step <= 0:
            raise ValueError("Time step size must be positive.")

        if not trajectory.t or len(trajectory.t) < 2:
            return trajectory  # No resampling needed for trajectories without timestamps or with only one timestamp

        def get_time_diff(t_end, t_start):
            """Get time difference, works for both datetime and numeric types."""
            diff = t_end - t_start
            try:
                return diff.total_seconds()  # datetime timedelta
            except AttributeError:
                return diff  # numeric

        def add_time_step(t_start, step_value, multiplier):
            """Add time step, works for both datetime and numeric types."""
            try:
                return t_start + timedelta(seconds=step_value * multiplier)  # datetime
            except TypeError:
                return t_start + step_value * multiplier  # numeric

        total_time = get_time_diff(trajectory.t[-1], trajectory.t[0])
        if total_time == 0:
            return trajectory  # No resampling needed for zero-duration trajectories

        num_steps = int(np.ceil(total_time / step)) + 1
        new_t = [add_time_step(trajectory.t[0], step, i) for i in range(num_steps)]

        # Calculate normalized time for interpolation
        normalized_times = [
            get_time_diff(t, trajectory.t[0]) / total_time for t in new_t
        ]

        new_ls = LineString(
            [
                trajectory.ls.interpolate(norm_t, normalized=True).coords[0]
                for norm_t in normalized_times
            ]
        )
        return Trajectory(id=trajectory.id, ls=new_ls, t=new_t)

    @staticmethod
    def get_requirements() -> dict:
        return {
            "get_next_time_step_size": {
                "short_name": "Time Step Size",
                "type": "get_float_function",
                "default": 0.1,
                "description": "Time interval (in seconds) between consecutive points in the resampled trajectory.",
                "optional": False,
                "default_mode": "fixed for dataset",
            }
        }
