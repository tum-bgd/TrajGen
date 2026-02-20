from ..trajectory import Trajectory
from shapely.geometry import LineString, Point
import numpy as np


class ConstantSpatialStepResampling:
    def __init__(self, config):
        self.config = config

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        step = self.config.get_next_spatial_step_size()  # e.g., 10.0 units
        if step <= 0:
            raise ValueError("Spatial step size must be positive.")

        ls = trajectory.ls
        if not isinstance(ls, LineString):
            raise ValueError(
                "Trajectory geometry must be a LineString for spatial resampling."
            )

        total_length = ls.length
        if total_length == 0:
            return trajectory  # No resampling needed for zero-length trajectories

        distances = np.arange(0.0, total_length + step / 2.0, step)
        # Always include the endpoint if we have fractional coverage
        if len(distances) > 0 and distances[-1] < total_length:
            distances = np.append(distances, total_length)
        distances = np.clip(distances, 0, total_length)
        new_coords = [trajectory.ls.interpolate(d).coords[0] for d in distances]
        new_ls = LineString(new_coords)
        new_t = (
            self._interpolate_velocities(trajectory, len(new_coords))
            if trajectory.t
            else None
        )
        return Trajectory(id=trajectory.id, ls=new_ls, t=new_t)

    @staticmethod
    def get_requirements() -> dict:
        return {
            "get_next_spatial_step_size": {
                "short_name": "Spatial Step Size",
                "type": "get_float_function",
                "default": 0.1,
                "description": "Distance between consecutive points in the resampled trajectory.",
                "optional": False,
                "default_mode": "fixed for dataset",
            }
        }

    def _interpolate_time(self, trajectory: Trajectory, num_points: int) -> None | list:
        if not trajectory.t:
            return None
        if num_points <= 0:
            return []

        total_length = trajectory.ls.length
        if total_length == 0:
            return [
                trajectory.t[0]
            ] * num_points  # All points have the same timestamp, e.g., for zero-length trajectories
        #  bounding boxes
        t_0 = trajectory.t[0]
        t_1 = trajectory.t[-1]
        if t_1 == t_0:
            return [t_0] * num_points  # All points have the same timestamp

        distances = np.linspace(0, total_length, num_points)
        new_t = [
            trajectory.t[0] + (trajectory.t[-1] - trajectory.t[0]) * (d / total_length)
            for d in distances
        ]
        return new_t

    def _interpolate_velocities(
        self, trajectory: Trajectory, num_points: int
    ) -> list | None:
        if not trajectory.t:
            return None

        t = trajectory.t
        ls = trajectory.ls
        total_length = ls.length
        if total_length == 0 or len(t) < 2:
            return [t[0]] * num_points if t else None  # Fallback

        # Get distances between consecutive points along the trajectory
        coords = list(ls.coords)
        segment_distances = []
        for i in range(len(coords) - 1):
            p1 = Point(coords[i])
            p2 = Point(coords[i + 1])
            segment_distances.append(p1.distance(p2))

        # Calculate cumulative distances
        cum_distances = np.cumsum([0] + segment_distances)

        # Create the new sample points at regular spatial intervals
        step = total_length / (num_points - 1) if num_points > 1 else 0
        new_distances = [i * step for i in range(num_points)]
        new_distances[-1] = total_length  # Ensure last point is exactly at the end

        new_times = []
        for target_dist in new_distances:
            if target_dist == 0:
                new_times.append(t[0])
            elif target_dist >= total_length:
                new_times.append(t[-1])
            else:
                # Find which segment this distance falls into
                segment_idx = 0
                for i in range(len(cum_distances) - 1):
                    if cum_distances[i] <= target_dist < cum_distances[i + 1]:
                        segment_idx = i
                        break

                # Calculate the fraction along this segment
                segment_start_dist = cum_distances[segment_idx]
                segment_end_dist = cum_distances[segment_idx + 1]
                segment_length = segment_end_dist - segment_start_dist

                if segment_length > 0:
                    fraction_in_segment = (
                        target_dist - segment_start_dist
                    ) / segment_length
                else:
                    fraction_in_segment = 0

                # Interpolate time within this segment
                segment_start_time = t[segment_idx]
                segment_end_time = t[segment_idx + 1]
                interpolated_time = segment_start_time + fraction_in_segment * (
                    segment_end_time - segment_start_time
                )
                new_times.append(interpolated_time)

        return new_times

    def _linear_interpolate_time(
        self, trajectory: Trajectory, num_points: int
    ) -> list | None:
        """Linear interpolation of timestamps along the trajectory."""
        if not trajectory.t:
            return None
        if num_points <= 0:
            return []

        total_length = trajectory.ls.length
        if total_length == 0:
            return [trajectory.t[0]] * num_points

        t_0 = trajectory.t[0]
        t_1 = trajectory.t[-1]
        if t_1 == t_0:
            return [t_0] * num_points

        distances = np.linspace(0, total_length, num_points)
        new_t = [
            trajectory.t[0] + (trajectory.t[-1] - trajectory.t[0]) * (d / total_length)
            for d in distances
        ]
        return new_t
