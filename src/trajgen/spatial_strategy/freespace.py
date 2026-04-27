import random
from ..config import Config
from ..trajectory import Trajectory
from shapely.geometry import LineString, Polygon, Point
from dataclasses import dataclass
import numpy as np


from .requirements_helpers import bbox_requirements


@dataclass
class Obstacle:
    polygon: Polygon
    buffered: Polygon
    id: int


class FreespaceStrategy:
    def __init__(self, config: Config):
        self.config = config
        self.is_3d = getattr(self.config, 'spatial_dimension', '2D') == '3D'
        self.rng = random.Random(config.seed)
        self.start = (
            config.get_start_point()
            if hasattr(config, "get_start_point")
            else (Point(config.x_min, config.y_min, config.z_min) if self.is_3d else Point(config.x_min, config.y_min))
        )
        self.end = (
            config.get_end_point()
            if hasattr(config, "get_end_point")
            else (Point(config.x_max, config.y_max, config.z_max) if self.is_3d else Point(config.x_max, config.y_max))
        )
        self.obstacles = self._generate_obstacles(self.config.num_obstacles)

    def _generate_obstacles(self, num_obstacles: int) -> list[Obstacle]:
        """Generate random non-overlapping obstacles."""
        obstacles = []
        obstacle_size_min = float(getattr(self.config, "obstacle_size_min", 0.01))
        obstacle_size_max = float(getattr(self.config, "obstacle_size_max", 0.15))

        for i in range(num_obstacles):
            while True:
                # Random obstacle position/size
                location = self.config.point_generator()
                w = self.rng.uniform(obstacle_size_min, obstacle_size_max)
                h = self.rng.uniform(obstacle_size_min, obstacle_size_max)

                obs_poly = Polygon(
                    [
                        (location.x, location.y),
                        (location.x + w, location.y),
                        (location.x + w, location.y + h),
                        (location.x, location.y + h),
                    ]
                )

                # Check no overlap with existing
                if all(obs_poly.intersects(ob.polygon) is False for ob in obstacles):
                    obstacles.append(Obstacle(obs_poly, obs_poly.buffer(0.02), i))
                    break

        return obstacles

    def _plan_path_around_obstacles(
        self, rng: random.Random, num_points: int = 10, deviation_factor: float = 0.1
    ) -> list[Point]:
        """RANSAC-inspired path planning: sample waypoints avoiding obstacles."""
        path = [self.start]

        # current_pos = self.start
        for _ in range(num_points - 2):  # -2 for start/end
            attempts = 0
            while attempts < 50:  # Max collision avoidance attempts
                # Straight-line interpolation to target with deviation
                t = len(path) / num_points
                target_x = self.start.x + t * (self.end.x - self.start.x)
                target_y = self.start.y + t * (self.end.y - self.start.y)
                if self.is_3d:
                    target_z = self.start.z + t * (self.end.z - self.start.z)
                    deviation_z = rng.gauss(0, deviation_factor)
                else:
                    target_z = 0
                    deviation_z = 0

                # Add trajectory-specific deviation
                deviation_x = rng.gauss(0, deviation_factor)
                deviation_y = rng.gauss(0, deviation_factor)

                if self.is_3d:
                    candidate = Point(target_x + deviation_x, target_y + deviation_y, target_z + deviation_z)
                else:
                    candidate = Point(target_x + deviation_x, target_y + deviation_y)

                # Clamp to bounds
                if self.is_3d:
                    candidate = Point(
                        np.clip(candidate.x, self.config.x_min, self.config.x_max),
                        np.clip(candidate.y, self.config.y_min, self.config.y_max),
                        np.clip(candidate.z, self.config.z_min, self.config.z_max)
                    )
                else:
                    candidate = Point(
                        np.clip(candidate.x, self.config.x_min, self.config.x_max),
                        np.clip(candidate.y, self.config.y_min, self.config.y_max)
                    )

                # Check collision-free
                collision = any(
                    candidate.within(obstacle.buffered) for obstacle in self.obstacles
                )

                if not collision:
                    path.append(candidate)
                    break
                attempts += 1

        path.append(self.end)  # Always end at target
        return path

    def __call__(self, trajectory_id: int) -> Trajectory:
        """Generate single trajectory avoiding obstacles."""

        # Vary length slightly per trajectory
        length = self.config.get_next_length()

        rng = random.Random(self.config.seed + trajectory_id)
        points = self._plan_path_around_obstacles(
            rng, length, self.config.deviation_factor
        )

        # Smooth path (simple linear interpolation)
        smoothed_points = self._smooth_path(points)

        ls = LineString(smoothed_points)
        return Trajectory(id=trajectory_id, ls=ls)

    def _smooth_path(self, points: list[Point]) -> list[tuple[float, ...]]:
        """Simple smoothing via spline interpolation."""
        if len(points) < 3:
            if self.is_3d:
                return [(p.x, p.y, p.z) for p in points]
            else:
                return [(p.x, p.y) for p in points]

        # Catmull-Rom spline approximation (simple version)
        smoothed = []
        for i in range(len(points)):
            t = i / (len(points) - 1)

            # Interpolate between waypoints
            idx = min(int(t * (len(points) - 1)), len(points) - 2)
            frac = t * (len(points) - 1) - idx

            p1, p2 = points[idx], points[idx + 1]
            interp_x = p1.x + frac * (p2.x - p1.x)
            interp_y = p1.y + frac * (p2.y - p1.y)
            if self.is_3d:
                interp_z = p1.z + frac * (p2.z - p1.z)
                smoothed.append((interp_x, interp_y, interp_z))
            else:
                smoothed.append((interp_x, interp_y))

        return smoothed

    def visualize_obstacles(self):
        """Debug: plot obstacles + sample paths."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot obstacles
        for obstacle in self.obstacles:
            x, y = obstacle.polygon.exterior.xy
            ax.fill(x, y, alpha=0.6, label=f"Obstacle {obstacle.id}")

        # Sample 5 trajectories
        for i in range(5):
            traj = self(i)
            x, y = traj.ls.xy
            ax.plot(x, y, "o-", linewidth=3, markersize=4, label=f"Traj {i}")

        # Start/end
        ax.plot(self.start.x, self.start.y, "gs", markersize=15, label="Start")
        ax.plot(self.end.x, self.end.y, "r^", markersize=15, label="End")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.title("FreeSpacePlanner: Similar Start/End, Obstacle Avoidance")
        plt.show()

    @staticmethod
    def get_requirements(spatial_dim: str = "2D"):
        return {
            "get_next_length": {
                "short_name": "Trajectory Length",
                "type": "get_int_function",
                "default": 10,
                "description": "Number of waypoints per trajectory.",
                "optional": False,
            },
            "get_next_num_obstacles": {
                "short_name": "Number of Obstacles",
                "type": "get_int_function",
                "default": 2,
                "description": "Number of random obstacles in the space.",
                "optional": False,
            },
            "get_next_deviation_factor": {
                "short_name": "Deviation Factor",
                "type": "get_float_function",
                "default": 0.1,
                "description": "How much paths deviate to avoid obstacles.",
                "optional": False,
            },
            "get_next_obstacle_size_min": {
                "short_name": "Min Obstacle Size",
                "type": "get_float_function",
                "default": 0.01,
                "default_mode": "fixed for dataset",
                "description": "Minimum width/height of generated obstacles.",
                "optional": False,
            },
            "get_next_obstacle_size_max": {
                "short_name": "Max Obstacle Size",
                "type": "get_float_function",
                "default": 0.15,
                "default_mode": "fixed for dataset",
                "description": "Maximum width/height of generated obstacles.",
                "optional": False,
            },
            "get_start_point": {
                "short_name": "Start Point",
                "type": "get_point_function",
                "default": None,
                "description": "Trajectory start point.",
                "optional": False,
            },
            "get_end_point": {
                "short_name": "End Point",
                "type": "get_point_function",
                "default": None,
                "description": "Trajectory end point.",
                "optional": False,
            },
            **bbox_requirements(spatial_dim),
        }
