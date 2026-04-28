from ..trajectory import Trajectory
from shapely.geometry import LineString
import numpy as np


class NoiseResampling:
    def __init__(self, config):
        self.config = config

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        """
        Add spatial noise to the trajectory.

        Args:
            trajectory: Input trajectory

        Returns:
            Trajectory with added noise
        """
        noise_type = getattr(self.config, "noise_type", "random")
        noise_level = getattr(self.config, "noise_level", 1.0)

        coords = list(trajectory.ls.coords)
        if not coords:
            return trajectory

        # Deterministic RNG for noise, based on config seed + trajectory id
        base_seed = int(getattr(self.config, "seed", 42))
        traj_id = int(getattr(trajectory, "id", 0))
        rng = np.random.default_rng(base_seed + traj_id)

        # Convert to numpy array for easier manipulation
        coords_np = np.array(coords)
        num_points = len(coords_np)
        dim = coords_np.shape[1]

        if noise_type == "random":
            # Add independent Gaussian noise to each coordinate
            noise = rng.normal(0, noise_level, coords_np.shape)
            new_coords = coords_np + noise

        elif noise_type == "orthogonal":
            if num_points < 2:
                # Cannot calculate direction for single point, fallback to random
                noise = rng.normal(0, noise_level, coords_np.shape)
                new_coords = coords_np + noise
            else:
                new_coords = np.copy(coords_np)

                # Calculate tangents (directions)
                # For inner points, use average of incoming and outgoing vectors
                # For endpoints, use the single available segment vector
                tangents = np.zeros_like(coords_np)

                # Forward difference (for all except last)
                diffs = coords_np[1:] - coords_np[:-1]
                # Normalize
                lengths = np.linalg.norm(diffs, axis=1, keepdims=True)
                # Avoid division by zero
                lengths[lengths == 0] = 1.0
                normalized_diffs = diffs / lengths

                # Assign tangents
                tangents[:-1] += normalized_diffs
                tangents[1:] += normalized_diffs

                # Normalize average tangents
                tangent_lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
                tangent_lengths[tangent_lengths == 0] = 1.0
                tangents = tangents / tangent_lengths

                if dim == 2:
                    # In 2D, orthogonal is (-y, x)
                    normals = np.zeros_like(tangents)
                    normals[:, 0] = -tangents[:, 1]
                    normals[:, 1] = tangents[:, 0]

                    # Generate noise magnitudes
                    magnitudes = np.random.normal(0, noise_level, num_points)

                    # Apply noise along normal
                    new_coords += normals * magnitudes[:, np.newaxis]

                elif dim == 3:
                    # In 3D, we need two orthogonal vectors to the tangent to define the normal plane
                    # We can execute this by generating a random vector, taking cross product with
                    # tangent to get one normal

                    # Generate random reference vectors
                    random_vecs = rng.standard_normal((num_points, 3))

                    # Project out the component along the tangent to make it orthogonal
                    # v_orth = v - (v . t) * t
                    dot_products = np.sum(random_vecs * tangents, axis=1, keepdims=True)
                    orth_vecs = random_vecs - dot_products * tangents

                    # Normalize
                    orth_lengths = np.linalg.norm(orth_vecs, axis=1, keepdims=True)
                    # Handle case where random vector was parallel to tangent
                    orth_lengths[orth_lengths < 1e-6] = 1.0
                    orth_vecs = orth_vecs / orth_lengths

                    # Generate noise magnitudes
                    magnitudes = np.random.normal(0, noise_level, num_points)

                    new_coords += orth_vecs * magnitudes[:, np.newaxis]

        else:
            # Unknown type, return original
            return trajectory

        # Create new LineString
        points_list = [tuple(pt) for pt in new_coords]

        # Ensure at least 2 points for LineString if it was originally valid
        if len(points_list) < 2 and len(coords) >= 2:
            points_list.append(points_list[0])

        new_ls = LineString(points_list)
        return Trajectory(trajectory.id, new_ls, trajectory.t)

    @staticmethod
    def get_requirements() -> dict:
        return {
            "noise_type": {
                "short_name": "Noise Type",
                "type": "get_str_function",
                "default": "random",  # Default to random
                "description": "Type of noise to apply: 'random' (independent) or "
                "'orthogonal' (perpendicular to movement).",
                # Ideally this would be a select/dropdown, using string for now or custom type if supported
                "options": ["random", "orthogonal"],  # Hint for UI if supported
                "default_mode": "fixed for dataset",
            },
            "noise_level": {
                "short_name": "Noise Standard Deviation",
                "type": "get_float_function",
                "default": 1.0,
                "description": "Standard deviation of the Gaussian noise.",
                "optional": False,
                "default_mode": "fixed for dataset",
            },
        }
