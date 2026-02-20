from ..trajectory import Trajectory
from shapely.geometry import LineString


class ConstantLengthResampling:
    def __init__(self, config):
        self.config = config

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        """
        Resample trajectory to have a fixed number of points.

        Args:
            trajectory: Input trajectory to resample

        Returns:
            Resampled trajectory with the target number of points
        """
        # Get target length from config
        target_length = self.config.get_next_target_length()

        # Get original coordinates
        original_coords = list(trajectory.ls.coords)

        if len(original_coords) == target_length:
            return trajectory  # Already the right length

        if len(original_coords) < 2:
            # Can't resample a single point meaningfully
            if target_length == 1:
                return trajectory
            # Duplicate the single point
            new_coords = [original_coords[0]] * target_length
            new_ls = LineString(new_coords)
            if trajectory.t is not None:
                new_t = [trajectory.t[0]] * target_length
                return Trajectory(trajectory.id, new_ls, new_t)
            else:
                return Trajectory(trajectory.id, new_ls, None)

        # Calculate cumulative distances along the original trajectory
        distances = [0.0]
        for i in range(1, len(original_coords)):
            dist = self.config.distance(original_coords[i], original_coords[i - 1])
            distances.append(distances[-1] + dist)

        total_length = distances[-1]

        if total_length < 1e-10:  # Use small tolerance for zero-length detection
            # All points are effectively the same, just return copies of the first point
            new_coords = [original_coords[0]] * target_length
            # LineString requires at least 2 points
            if len(new_coords) == 1:
                new_coords.append(new_coords[0])
            new_ls = LineString(new_coords)

            # Handle timestamps
            if trajectory.t is not None:
                new_t = [trajectory.t[0]] * target_length
                return Trajectory(trajectory.id, new_ls, new_t)
            else:
                return Trajectory(trajectory.id, new_ls, None)

        # Calculate target distances for the new points
        if target_length == 1:
            target_distances = [total_length / 2.0]  # Middle point
        else:
            target_distances = [
                i * total_length / (target_length - 1) for i in range(target_length)
            ]

        # Interpolate coordinates at target distances
        new_coords = []
        new_timestamps = [] if trajectory.t is not None else None

        for target_dist in target_distances:
            # Find the segment that contains this distance
            segment_idx = 0
            for i in range(len(distances) - 1):
                if distances[i] <= target_dist <= distances[i + 1]:
                    segment_idx = i
                    break

            if target_dist <= distances[0]:
                # At the beginning
                new_coords.append(original_coords[0])
                if new_timestamps is not None:
                    new_timestamps.append(trajectory.t[0])
            elif target_dist >= distances[-1]:
                # At the end
                new_coords.append(original_coords[-1])
                if new_timestamps is not None:
                    new_timestamps.append(trajectory.t[-1])
            else:
                # Interpolate between segment_idx and segment_idx + 1
                start_dist = distances[segment_idx]
                end_dist = distances[segment_idx + 1]
                segment_length = end_dist - start_dist

                if segment_length == 0:
                    ratio = 0.0
                else:
                    ratio = (target_dist - start_dist) / segment_length

                start_coord = original_coords[segment_idx]
                end_coord = original_coords[segment_idx + 1]

                # Interpolate all coordinates (support N-dimensions)
                new_coord = tuple(
                    s + ratio * (e - s) for s, e in zip(start_coord, end_coord)
                )
                new_coords.append(new_coord)

                if new_timestamps is not None:
                    start_time = trajectory.t[segment_idx]
                    end_time = trajectory.t[segment_idx + 1]
                    new_time = start_time + ratio * (end_time - start_time)
                    new_timestamps.append(new_time)

        # LineString requires at least 2 points, duplicate the last point if needed
        if len(new_coords) == 1:
            new_coords.append(new_coords[0])
            if new_timestamps is not None:
                new_timestamps.append(new_timestamps[0])

        new_ls = LineString(new_coords)
        return Trajectory(trajectory.id, new_ls, new_timestamps)

    @staticmethod
    def get_requirements() -> dict:
        return {
            "get_next_target_length": {
                "short_name": "Target Length",
                "type": "get_int_function",
                "default": 10,
                "description": "Target number of points for the resampled trajectory.",
                "optional": False,
                "default_mode": "fixed for dataset",
            }
        }
