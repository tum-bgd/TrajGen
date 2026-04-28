from ..trajectory import Trajectory
from shapely.geometry import LineString
import numpy as np


class TimeTeleportResampling:
    def __init__(self, config):
        self.config = config

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        """
        Randomly teleport segments of the trajectory forward or backward in time.

        Args:
            trajectory: Input trajectory with timestamps

        Returns:
            Trajectory with time-teleported segments
        """
        if not trajectory.t or len(trajectory.t) < 2:
            return trajectory  # No teleportation if no timestamps

        num_teleports = int(getattr(self.config, "num_time_teleports", 1))
        direction = getattr(self.config, "teleport_direction", "both")

        if num_teleports <= 0:
            return trajectory

        # Deterministic RNG for reproducibility
        base_seed = int(getattr(self.config, "seed", 42))
        traj_id = int(getattr(trajectory, "id", 0))
        rng = np.random.default_rng(base_seed + traj_id)

        coords = list(trajectory.ls.coords)
        new_t = list(trajectory.t)

        total_duration = trajectory.t[-1] - trajectory.t[0]

        # Handle both numeric and datetime types
        try:
            total_duration_secs = total_duration.total_seconds()
        except AttributeError:
            total_duration_secs = float(total_duration)

        if total_duration_secs <= 0:
            return trajectory  # No teleportation for zero-duration trajectories

        num_points = len(coords)

        for _ in range(num_teleports):
            # Randomly select a segment (start and end indices)
            if num_points < 2:
                break

            start_idx = rng.integers(0, num_points)
            end_idx = rng.integers(start_idx + 1, num_points + 1)

            # Generate random time shift within the trajectory duration
            max_shift = total_duration_secs * 0.5  # Shift up to 50% of total duration

            if direction == "backward":
                time_shift = -rng.uniform(0, max_shift)
            elif direction == "forward":
                time_shift = rng.uniform(0, max_shift)
            elif direction == "both":
                time_shift = rng.uniform(-max_shift, max_shift)
            else:
                time_shift = 0

            # Apply time shift to the selected segment
            for i in range(start_idx, end_idx):
                try:
                    # For datetime objects
                    from datetime import timedelta

                    new_t[i] = new_t[i] + timedelta(seconds=time_shift)
                except TypeError:
                    # For numeric timestamps
                    new_t[i] = new_t[i] + time_shift

        # Create new linestring (spatial coordinates unchanged)
        new_ls = LineString(coords)
        return Trajectory(trajectory.id, new_ls, new_t)

    @staticmethod
    def get_requirements() -> dict:
        return {
            "num_time_teleports": {
                "short_name": "Number of Teleports",
                "type": "get_int_function",
                "default": 1,
                "description": "Number of random time teleportations to apply to each trajectory.",
                "optional": False,
                "default_mode": "fixed for dataset",
            },
            "teleport_direction": {
                "short_name": "Teleport Direction",
                "type": "get_str_function",
                "default": "both",
                "description": "Direction of time teleportation: 'backward' (only past), 'forward' (only future), or 'both' (random direction).",
                "options": ["backward", "forward", "both"],
                "default_mode": "fixed for dataset",
            },
        }
