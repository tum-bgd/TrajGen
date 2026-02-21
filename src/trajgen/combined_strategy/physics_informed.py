from trajgen.spatial_strategy.requirements_helpers import bbox_requirements
from ..trajectory import Trajectory
from ..config import Config
from .combined_strategy import CombinedStrategy
from shapely.geometry import LineString
import random
import math


class PhysicsInformedCombinedStrategy(CombinedStrategy):
    def __init__(self, config: Config):
        super().__init__(config)
        self.gravity = getattr(config, "gravity", 9.81)  # m/s^2
        self.dt = getattr(config, "time_step", 0.01)  # time step in seconds
        self.simulation_time = getattr(
            config, "simulation_time", 10.0
        )  # total simulation time
        self.bounce_damping = getattr(
            config, "bounce_damping", 0.8
        )  # energy loss on bounce
        self.num_balls = getattr(config, "num_balls", 1)

    def __call__(self, trajectory_id: int):
        """Generate physics-informed trajectory by simulating bouncing balls."""
        # Get bounding box from config
        x_min = getattr(self.config, "x_min", 0.0)
        x_max = getattr(self.config, "x_max", 100.0)
        y_min = getattr(self.config, "y_min", 0.0)  # floor
        y_max = getattr(self.config, "y_max", 50.0)  # ceiling

        # Check if 3D simulation is requested
        dimension = getattr(self.config, "spatial_dimension", "2D")
        if dimension == "3D":
            z_min = getattr(self.config, "z_min", 0.0)
            z_max = getattr(self.config, "z_max", 50.0)
            return self._simulate_single_ball_3d(
                x_min, x_max, y_min, y_max, z_min, z_max, trajectory_id
            )
        else:
            return self._simulate_single_ball(x_min, x_max, y_min, y_max, trajectory_id)

    def _simulate_single_ball(self, x_min, x_max, y_min, y_max, trajectory_id):
        """Simulate physics of a single bouncing ball."""
        # Sample initial conditions
        point = self.config.get_next_point()
        x, y = point.x, point.y
        vx = self.config.get_next_velocity()
        vy = self.config.get_next_velocity()

        # Store trajectory points
        trajectory_points = [(x, y)]
        timestamps = [0.0]

        # Physics simulation loop
        current_time = 0.0
        while current_time < self.simulation_time:
            current_time += self.dt

            # Update velocity (gravity affects only y-component)
            vy -= self.gravity * self.dt  # gravity pulls downward

            # Update position
            x += vx * self.dt
            y += vy * self.dt

            # Handle collisions with walls and boundaries
            bounced = False

            # Left wall collision
            if x <= x_min:
                x = x_min
                vx = -vx * self.bounce_damping
                bounced = True

            # Right wall collision
            if x >= x_max:
                x = x_max
                vx = -vx * self.bounce_damping
                bounced = True

            # Floor collision
            if y <= y_min:
                y = y_min
                vy = -vy * self.bounce_damping
                bounced = True

            # Ceiling collision
            if y >= y_max:
                y = y_max
                vy = -vy * self.bounce_damping
                bounced = True

            # Add small random perturbation on bounce to prevent perfect repetition
            if bounced:
                vx += random.uniform(-0.5, 0.5)
                vy += random.uniform(-0.5, 0.5)

            # Store trajectory point
            trajectory_points.append((x, y))
            timestamps.append(current_time)

            # Stop if ball comes to rest (very low velocity)
            if abs(vx) < 0.1 and abs(vy) < 0.1 and y <= y_min + (y_max - y_min) * 0.01:
                break

        # Create LineString from trajectory points
        if len(trajectory_points) < 2:
            # If we only have one point, duplicate it to create a valid LineString
            trajectory_points.append(trajectory_points[0])
            timestamps.append(timestamps[0] + self.dt)

        linestring = LineString(trajectory_points)

        # Create and return Trajectory object
        return Trajectory(id=trajectory_id, ls=linestring, t=timestamps)

    def _simulate_single_ball_3d(
        self, x_min, x_max, y_min, y_max, z_min, z_max, trajectory_id
    ):
        """Simulate physics of a single bouncing ball in 3D space."""
        # Sample initial conditions
        point = self.config.get_next_point()
        x, y, z = point.x, point.y, getattr(point, "z", random.uniform(z_min, z_max))
        vx = self.config.get_next_velocity()
        vy = self.config.get_next_velocity()
        vz = self.config.get_next_velocity()

        # Store trajectory points
        trajectory_points = [(x, y, z)]
        timestamps = [0.0]

        # Physics simulation loop
        current_time = 0.0
        while current_time < self.simulation_time:
            current_time += self.dt

            # Update velocity (gravity affects only z-component in 3D)
            vz -= self.gravity * self.dt  # gravity pulls downward in -Z

            # Update position
            x += vx * self.dt
            y += vy * self.dt
            z += vz * self.dt

            # Handle collisions with walls and boundaries
            bounced = False

            # Left wall collision
            if x <= x_min:
                x = x_min
                vx = -vx * self.bounce_damping
                bounced = True

            # Right wall collision
            if x >= x_max:
                x = x_max
                vx = -vx * self.bounce_damping
                bounced = True

            # Front wall collision
            if y <= y_min:
                y = y_min
                vy = -vy * self.bounce_damping
                bounced = True

            # Back wall collision
            if y >= y_max:
                y = y_max
                vy = -vy * self.bounce_damping
                bounced = True

            # Floor collision (z_min is the floor)
            if z <= z_min:
                z = z_min
                vz = -vz * self.bounce_damping
                bounced = True

            # Ceiling collision (z_max is the ceiling)
            if z >= z_max:
                z = z_max
                vz = -vz * self.bounce_damping
                bounced = True

            # Add small random perturbation on bounce to prevent perfect repetition
            if bounced:
                vx += random.uniform(-0.5, 0.5)
                vy += random.uniform(-0.5, 0.5)
                vz += random.uniform(-0.5, 0.5)

            # Store trajectory point
            trajectory_points.append((x, y, z))
            timestamps.append(current_time)

            # Stop if ball comes to rest (very low velocity)
            if abs(vx) < 0.1 and abs(vy) < 0.1 and abs(vz) < 0.1 and z <= z_min + 1.0:
                break

        # Create LineString from trajectory points
        if len(trajectory_points) < 2:
            # If we only have one point, duplicate it to create a valid LineString
            trajectory_points.append(trajectory_points[0])
            timestamps.append(timestamps[0] + self.dt)

        linestring = LineString(trajectory_points)

        # Create and return Trajectory object
        return Trajectory(id=trajectory_id, ls=linestring, t=timestamps)

    def _calculate_trajectory_properties(self, trajectory_points):
        """Calculate physical properties of the trajectory."""
        if len(trajectory_points) < 2:
            return {}

        total_distance = 0.0
        max_height = max(point[1] for point in trajectory_points)
        min_height = min(point[1] for point in trajectory_points)

        # Calculate total path length
        for i in range(1, len(trajectory_points)):
            dx = trajectory_points[i][0] - trajectory_points[i - 1][0]
            dy = trajectory_points[i][1] - trajectory_points[i - 1][1]
            total_distance += math.sqrt(dx * dx + dy * dy)

        return {
            "total_distance": total_distance,
            "max_height": max_height,
            "min_height": min_height,
            "height_range": max_height - min_height,
        }

    @staticmethod
    def get_requirements(spatial_dim: str = "2D") -> dict:
        physics_requirements = {
            "gravity": {
                "short_name": "Gravity",
                "type": "get_float_function",
                "default": 9.81,
                "description": "Gravitational acceleration (m/s^2) affecting the trajectories.",
                "optional": True,
            },
            "time_step": {
                "short_name": "Time Step",
                "type": "get_float_function",
                "default": 0.01,
                "description": "Time step (in seconds) for the physics simulation.",
                "optional": True,
            },
            "simulation_time": {
                "short_name": "Simulation Time",
                "type": "get_float_function",
                "default": 1.0,
                "description": "Total time (in seconds) to simulate the trajectory.",
                "optional": True,
            },
            "bounce_damping": {
                "short_name": "Bounce Damping",
                "type": "get_float_function",
                "default": 0.8,
                "description": "Energy loss factor on bounce (0 to 1).",
                "optional": True,
            },
            "num_balls": {
                "short_name": "Number of Balls",
                "type": "get_int_function",
                "default": 1,
                "description": "Number of independent bouncing balls to simulate.",
                "optional": True,
            },
            "get_next_velocity": {
                "short_name": "Initial Velocity",
                "type": "get_float_function",
                "default": 8.0,
                "default_mode": "fixed for dataset",
                "description": "Initial speed applied to both vx and vy at the start of the simulation.",
                "optional": True,
            },
            **bbox_requirements(spatial_dim),
        }
        return physics_requirements
