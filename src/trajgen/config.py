import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Callable

from shapely.geometry import Point
from .point_generator import PointGenerator


@dataclass
class Config:
    # ------------------------------------------------------------------
    # General Parameters
    # ------------------------------------------------------------------
    seed: int = 43
    rng: random.Random = field(init=False)

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    # ------------------------------------------------------------------
    # Dataset Parameters
    # ------------------------------------------------------------------
    num_trajectories: int = 10
    spatial_dimension: str = "2D"  # "2D", "3D"
    spatial_dim_type: str = "continuous"  # "continuous", "discrete"
    temporal_dim_type: str = "continuous"  # "continuous", "discrete"

    # ------------------------------------------------------------------
    # Bounding Box
    # ------------------------------------------------------------------
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    z_min: float = 0.0
    z_max: float = 1.0

    # ------------------------------------------------------------------
    # Strategy Selection
    # ------------------------------------------------------------------
    spatial_strategy: str = "equal_distribution"
    resampling_strategy: Optional[str] = None
    temporal_strategy: str = "fixed_timesteps"

    # ------------------------------------------------------------------
    # Length Parameters
    # ------------------------------------------------------------------
    length_mode: str = (
        "undefined"  # "undefined", "fixed for dataset", "fixed for trajectory"
    )
    length: int = 10
    length_distribution: Optional[str] = None  # "uniform", "normal"
    length_min: int = 5
    length_max: int = 50
    length_mean: float = 25.0
    length_std: float = 5.0

    # ------------------------------------------------------------------
    # Spatial Step Size Parameters
    # ------------------------------------------------------------------
    spatial_step_size_mode: str = "undefined"
    spatial_step_size: float = 1.0
    spatial_step_size_distribution: Optional[str] = None
    spatial_step_size_min: float = 0.1
    spatial_step_size_max: float = 2.0
    spatial_step_size_mean: float = 1.0
    spatial_step_size_std: float = 0.2

    # ------------------------------------------------------------------
    # Smoothness Parameters
    # ------------------------------------------------------------------
    smoothness_mode: str = "undefined"
    smoothness: float = 1.0
    smoothness_distribution: Optional[str] = None
    smoothness_min: float = 0.5
    smoothness_max: float = 2.0
    smoothness_mean: float = 1.0
    smoothness_std: float = 0.2

    # ------------------------------------------------------------------
    # Temporal Parameters (Velocity, Acceleration, Time)
    # ------------------------------------------------------------------
    time_step: float = 1.0
    time_step_distribution: Optional[str] = None  # For resampling strategy requirement

    # Velocity
    velocity_mode: str = "fixed for dataset"
    velocity: float = 1.0
    velocity_distribution: Optional[str] = None
    velocity_min: float = 1.0
    velocity_max: float = 1.0
    velocity_mean: float = 1.0
    velocity_std: float = 0.1

    # Acceleration
    acceleration_mode: str = "fixed for dataset"
    acceleration: float = 0.0
    acceleration_distribution: Optional[str] = None
    acceleration_min: float = -1.0
    acceleration_max: float = 1.0
    acceleration_mean: float = 0.0
    acceleration_std: float = 0.1

    # Start Time & Temporal Extent
    start_time_mode: str = "fixed for dataset"
    start_time: float = 0.0
    start_time_distribution: Optional[str] = None
    start_time_min: float = 0.0
    start_time_max: float = 10.0
    start_time_mean: float = 0.0
    start_time_std: float = 1.0

    temporal_extent_mode: str = "undefined"
    temporal_extent: float = 10.0
    temporal_extent_distribution: Optional[str] = None
    temporal_extent_min: float = 1.0
    temporal_extent_max: float = 20.0
    temporal_extent_mean: float = 10.0
    temporal_extent_std: float = 2.0

    # ------------------------------------------------------------------
    # Point Generation (Start/End/Normal)
    # ------------------------------------------------------------------
    point_generator: Optional[PointGenerator] = None

    # Start Point
    start_point_mode: str = "undefined"
    # Note: we use flattened coordinates for start/end points
    start_point_x: float = 0.0
    start_point_y: float = 0.0
    start_point_z: float = 0.0

    # End Point
    end_point_mode: str = "undefined"
    end_point_x: float = 1.0
    end_point_y: float = 1.0
    end_point_z: float = 1.0

    # ------------------------------------------------------------------
    # Strategy Specific Parameters
    # ------------------------------------------------------------------

    # Equal Distribution
    grid_rows: int = 100
    grid_cols: int = 100

    # Polynomial
    num_control_points: Optional[int] = None
    interior_margin: Optional[float] = None
    closed_loop: bool = False

    # Freespace
    size_generator: Optional[PointGenerator] = None
    deviation_factor: float = 0.1
    num_obstacles: int = 2
    obstacle_size_min: float = 0.01
    obstacle_size_max: float = 0.15

    # OSM Sampling
    osm_place: Optional[str] = None
    osm_pbf_path: Optional[str] = None
    osm_max_hops: Optional[int] = None
    osm_max_meters: Optional[float] = None
    osm_jitter_std_m: Optional[float] = None
    osm_max_attempts_per_traj: int = 5

    # Resampling - Constant Length
    target_length: int = 10
    target_length_mode: str = "fixed for dataset"  # Inferred default

    # Resampling - Constant Time
    time_step_size: float = 0.1  # distinct from 'time_step' used in generation

    # Noise Resampling
    noise_type: str = "random"  # "random", "orthogonal"
    noise_level: float = 1.0

    # Physics
    gravity: float = 9.81
    simulation_time: float = 10.0
    bounce_damping: float = 0.8
    num_balls: int = 1

    # Distance Function
    distance_function_name: str = "Euclidean"  # "Euclidean", "Manhattan"

    # ------------------------------------------------------------------
    # I/O Parameters
    # ------------------------------------------------------------------
    output: str = "trajectory.txt"
    parameters: str = "params.json"

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------
    def _get_scalar_from_mode(self, prefix: str) -> Union[int, float]:
        """
        Generic helper to get a scalar value based on mode/distribution.
        prefix: e.g., 'length', 'velocity', 'spatial_step_size'
        """
        # Try to find specific mode attribute, generic fallback to "fixed for dataset" if not found but value exists
        mode_attr = f"{prefix}_mode"
        if not hasattr(self, mode_attr):
            # If no mode attribute, assume simple scalar access
            if hasattr(self, prefix):
                return getattr(self, prefix)
            return 0

        mode = getattr(self, mode_attr)

        if mode == "fixed for dataset" or mode is None:
            if hasattr(self, prefix):
                val = getattr(self, prefix)
                if val is not None:
                    return val
            # Fallback if specific prop is None/missing
            return 0

        elif mode == "fixed for trajectory":
            dist = getattr(self, f"{prefix}_distribution", None)

            if dist == "uniform":
                min_val = getattr(self, f"{prefix}_min", 0)
                max_val = getattr(self, f"{prefix}_max", 1)

                # Heuristic for int vs float
                is_float = isinstance(min_val, float) or isinstance(max_val, float)
                if not is_float:
                    # Check base property type hint or default value
                    base_val = getattr(self, prefix, None)
                    if isinstance(base_val, float):
                        is_float = True

                if is_float:
                    return self.rng.uniform(min_val, max_val)
                else:
                    return self.rng.randint(min_val, max_val)

            elif dist == "normal":
                mean_val = getattr(self, f"{prefix}_mean", 0)
                std_val = getattr(self, f"{prefix}_std", 1)
                val = self.rng.normalvariate(mean_val, std_val)

                # Heuristic for int/float
                if isinstance(mean_val, int) and isinstance(std_val, int):
                    return int(val)
                return val
            else:
                # If distribution is Undefined or None in trajectory mode, fallback to dataset value?
                # Or raise error. Let's return dataset value for robustness
                if hasattr(self, prefix):
                    return getattr(self, prefix)

        elif mode == "undefined":
            if hasattr(self, prefix):
                return getattr(self, prefix)
            return 0

        return 0

    def get_next_value(self, field_name: str):
        """
        Mimic app config generic getter.
        """
        # Strip 'get_next_' prefix if present to find the base property
        if field_name.startswith("get_next_"):
            base_name = field_name[9:]
        else:
            base_name = field_name

        # Specific overrides
        if base_name == "point":
            return self.get_next_point()
        if base_name == "bounding_box":
            return self.get_next_bounding_box()

        # General scalar lookup
        return self._get_scalar_from_mode(base_name)

    # --- Specific Getters used by Strategies ---

    def get_next_length(self) -> int:
        val = self._get_scalar_from_mode("length")
        return max(1, int(val))

    def get_next_spatial_step_size(self) -> float:
        val = self._get_scalar_from_mode("spatial_step_size")
        return max(0.0001, float(val))

    def get_next_velocity(self) -> float:
        # Backward compatibility for legacy files setting min/max directly without mode
        if self.velocity_mode == "fixed for dataset":
            # Check if legacy range is active (diff values)
            if hasattr(self, "velocity_min") and hasattr(self, "velocity_max"):
                if self.velocity_min != self.velocity_max:
                    # If defaults (1.0) are changed, likely user intent
                    if self.velocity_min != 1.0 or self.velocity_max != 1.0:
                        return self.rng.uniform(self.velocity_min, self.velocity_max)

        val = self._get_scalar_from_mode("velocity")
        return float(val)

    def get_next_acceleration(self) -> float:
        val = self._get_scalar_from_mode("acceleration")
        return float(val)

    def get_next_tmin(self) -> float:
        val = self._get_scalar_from_mode("start_time")
        return float(val)

    def get_next_temporal_extent(self) -> float:
        val = self._get_scalar_from_mode("temporal_extent")
        return float(val)

    def get_next_time_step(self) -> float:
        # Note: Physics strategy asks for 'time_step', FixedTime strategy asks for 'time_step'
        return float(self.time_step)

    def get_next_time_step_size(self) -> float:
        # Used by resampling
        return float(self.time_step_size)

    def get_next_target_length(self) -> int:
        val = self._get_scalar_from_mode("target_length")
        return int(val)

    def get_next_bounding_box(self) -> Tuple[float, ...]:
        if self.spatial_dimension == "2D":
            return (self.x_min, self.x_max, self.y_min, self.y_max)
        else:
            return (
                self.x_min,
                self.x_max,
                self.y_min,
                self.y_max,
                self.z_min,
                self.z_max,
            )

    # Properties for get_next_x_min etc used by bbox_requirements
    @property
    def get_next_x_min(self):
        return self.x_min

    @property
    def get_next_x_max(self):
        return self.x_max

    @property
    def get_next_y_min(self):
        return self.y_min

    @property
    def get_next_y_max(self):
        return self.y_max

    @property
    def get_next_z_min(self):
        return self.z_min

    @property
    def get_next_z_max(self):
        return self.z_max

    # Strategy specific lookups that might use distribution logic

    def get_next_grid_rows(self) -> int:
        return self.grid_rows

    def get_next_grid_cols(self) -> int:
        return self.grid_cols

    def get_next_num_control_points(self) -> int:
        if self.num_control_points is None:
            return 5
        return self.num_control_points

    def get_next_closed_loop(self) -> bool:
        return self.closed_loop

    def get_next_num_obstacles(self) -> int:
        return self.num_obstacles

    def get_next_obstacle_size_min(self) -> float:
        return self.obstacle_size_min

    def get_next_obstacle_size_max(self) -> float:
        return self.obstacle_size_max

    def get_next_osm_max_hops(self) -> int:
        return self.osm_max_hops if self.osm_max_hops else 10

    def get_next_osm_jitter_std_m(self) -> float:
        return self.osm_jitter_std_m if self.osm_jitter_std_m else 5.0

    def get_next_osm_max_meters(self) -> float:
        return self.osm_max_meters if self.osm_max_meters else 5000.0

    def get_next_osm_max_attempts_per_traj(self) -> int:
        return self.osm_max_attempts_per_traj

    def get_next_point(self) -> Point:
        if self.point_generator is not None:
            points = self.point_generator(1)
            return points[0]
        # Fallback to random in bbox
        return self._random_point_from_bbox()

    def _random_point_from_bbox(self) -> Point:
        x = self.rng.uniform(self.x_min, self.x_max)
        y = self.rng.uniform(self.y_min, self.y_max)
        if self.spatial_dimension == "3D":
            z = self.rng.uniform(self.z_min, self.z_max)
            return Point(x, y, z)
        return Point(x, y)

    def get_start_point(self) -> Optional[Point]:
        if self.start_point_mode == "fixed for dataset":
            if self.spatial_dimension == "3D":
                return Point(self.start_point_x, self.start_point_y, self.start_point_z)
            return Point(self.start_point_x, self.start_point_y)
        return None

    def get_end_point(self) -> Optional[Point]:
        if self.end_point_mode == "fixed for dataset":
            if self.spatial_dimension == "3D":
                return Point(self.end_point_x, self.end_point_y, self.end_point_z)
            return Point(self.end_point_x, self.end_point_y)
        return None

    @property
    def distance_function(self) -> Callable:
        dims = 3 if self.spatial_dimension == "3D" else 2
        name = self.distance_function_name

        if name == "Manhattan":
            return lambda a, b: sum(
                abs(a[i] - b[i]) for i in range(min(dims, len(a), len(b)))
            )
        # Euclidean (default)
        return (
            lambda a, b: sum(
                (a[i] - b[i]) ** 2 for i in range(min(dims, len(a), len(b)))
            )
            ** 0.5
        )

    def get_next_spatial_extent(self):
        raise NotImplementedError()

    def validate(self):
        assert self.num_trajectories > 0, "num_trajectories must be positive"
        # Add strategy validations as needed

    def write_to_file(self, filepath: str):
        # Exclude internal/complex objects for JSON serialization
        data = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["rng", "point_generator", "size_generator"]
        }
        import json

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    def read_from_file(self, filepath: str):
        import json

        with open(filepath, "r") as f:
            data = json.load(f)
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            # Re-init RNG only if seed is present, or just to be safe
            self.rng = random.Random(self.seed)
