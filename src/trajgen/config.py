import random

from shapely.geometry import Point
from trajgen.relevant_properties import Properties


class Config:
    """Config class that provides a unified interface for trajectory generation
    - ``get_next_value(field_name)`` – generic value retrieval that respects
      the mode / distribution configured in the UI.
    - Attribute access (``config.x_min``, ``config.seed``, …) for backward
      compatibility with ``trajgen.config.Config``.
    - Automatic ``get_*()`` method proxies so strategy code that calls e.g.
      ``config.get_next_length()`` keeps working.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, state: dict, seed: int | None = None):
        object.__setattr__(self, "_state", dict(state))
        _seed = seed if seed is not None else int(state.get("config_seed", 42))
        object.__setattr__(self, "_seed", _seed)
        object.__setattr__(self, "_rng", random.Random(_seed))
        # Lazily-built generators keyed by field_name
        object.__setattr__(self, "_generators", {})

    # ------------------------------------------------------------------
    # Attribute access – backward compatibility
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        # 1) get_* methods → always return a callable proxy
        if name.startswith("get_"):

            def _method_proxy(*_args, **_kwargs):
                return self.get_next_value(name)

            return _method_proxy

        config_key = f"config_{name}"

        # 2) Mode-based value (configured via universal_user_input_method)
        mode_key = f"{config_key}_mode"
        if mode_key in self._state:
            return self.get_next_value(name)

        # 3) Try with get_next_ prefix (e.g. closed_loop → get_next_closed_loop)
        get_next_mode_key = f"config_get_next_{name}_mode"
        if get_next_mode_key in self._state:
            return self.get_next_value(f"get_next_{name}")

        # 4) Direct value lookup
        if config_key in self._state:
            return self._state[config_key]

        # 5) Unprefixed lookup (e.g. selected_method)
        if name in self._state:
            return self._state[name]

        # 6) Built-in helpers
        if name == "rng":
            return self._rng
        if name == "seed":
            return self._seed

        raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._state[f"config_{name}"] = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_next_value(self, field_name: str):
        """Return the next value for *field_name* based on UI configuration.

        Depending on the type (int / float / point / bool) and the chosen
        mode (fixed for dataset / fixed for trajectory), this either returns
        a constant or draws from a lazily-built distribution.
        """
        value_type = self._detect_type(field_name)

        if value_type == "point":
            return self._get_next_point(field_name)
        elif value_type == "bool":
            return self._get_next_bool(field_name)
        else:
            return self._get_next_scalar(field_name, value_type)

    def get(self, key: str, default=None):
        """Dict-style access into the state snapshot."""
        config_key = f"config_{key}" if not key.startswith("config_") else key
        return self._state.get(config_key, default)

    # ------------------------------------------------------------------
    # Explicit backward-compat methods that DON'T follow the generic
    # mode / distribution pattern used by universal_user_input_method.
    # ------------------------------------------------------------------
    def get_next_bounding_box(self):
        spatial_dim = self._state.get("config_spatial_dimension", "2D")
        x_min = float(self._state.get("config_x_min", 0.0))
        x_max = float(self._state.get("config_x_max", 1.0))
        y_min = float(self._state.get("config_y_min", 0.0))
        y_max = float(self._state.get("config_y_max", 1.0))
        if spatial_dim == "3D":
            z_min = float(self._state.get("config_z_min", 0.0))
            z_max = float(self._state.get("config_z_max", 1.0))
            return (x_min, x_max, y_min, y_max, z_min, z_max)
        return (x_min, x_max, y_min, y_max)

    def get_next_tmin(self) -> float:
        return float(self._state.get("config_tmin", 0.0))

    def get_next_velocity(self) -> float:
        # Try generic UI configuration first
        mode_key = "config_get_next_velocity_mode"
        if mode_key in self._state:
            return self._get_next_scalar("get_next_velocity", "float")
        # Fallback to legacy min/max velocity
        min_v = float(self._state.get("config_min_velocity", 1.0))
        max_v = float(self._state.get("config_max_velocity", 1.0))
        return self._rng.uniform(min_v, max_v)

    def get_next_acceleration(self) -> float:
        mode_key = "config_get_next_acceleration_mode"
        if mode_key in self._state:
            return self._get_next_scalar("get_next_acceleration", "float")
        return float(self._state.get("config_acceleration", 0.0))

    def get_next_time_step(self) -> float:
        return float(self._state.get("config_time_step", 1.0))

    def get_next_point(self) -> Point:
        """Legacy: delegates to point_generator if set, else UI config."""
        pg = self._state.get("config_point_generator")
        if pg is not None:
            points = pg(1)
            return points[0]
        return self.get_next_value("get_next_point")

    def get_start_point(self) -> Point:
        return self.get_next_value("get_start_point")

    def get_end_point(self) -> Point:
        return self.get_next_value("get_end_point")

    def validate(self):
        """Minimal validation – extend as needed."""
        num_traj = self._state.get("config_num_trajectories", 0)
        assert num_traj > 0, "num_trajectories must be positive"

    # ------------------------------------------------------------------
    # Type detection
    # ------------------------------------------------------------------
    def _detect_type(self, field_name: str) -> str:
        """Infer the return type for *field_name*."""
        # 1) Check Properties metadata
        mode_attr = f"config_{field_name}_mode"
        prop = getattr(Properties, mode_attr, None)
        if prop is not None and isinstance(prop, dict):
            type_str = prop.get("type", "")
            if "int" in type_str:
                return "int"
            if "float" in type_str:
                return "float"
            if "str" in type_str or "string" in type_str:
                return "str"
            if "point" in type_str:
                return "point"
            if "bool" in type_str:
                return "bool"

        # 2) Point-specific mode key in state
        # Strip trailing _point before adding suffix to avoid double _point
        _base = field_name[:-6] if field_name.endswith("_point") else field_name
        point_mode_key = f"config_{_base}_point_mode"
        if point_mode_key in self._state:
            return "point"

        # 3) Bool-specific probability key
        prob_key = f"config_{field_name}_probability"
        if prob_key in self._state:
            return "bool"

        # 4) Infer from the stored value itself
        config_key = f"config_{field_name}"
        mode_key = f"{config_key}_mode"
        if mode_key in self._state:
            val = self._state.get(config_key)
            if isinstance(val, bool):
                return "bool"
            if isinstance(val, int):
                return "int"
            if isinstance(val, float):
                return "float"
            if isinstance(val, str):
                return "str"

        # 5) Infer from Properties for the value entry
        val_prop = getattr(Properties, config_key, None)
        if val_prop and isinstance(val_prop, dict):
            t = val_prop.get("type", "")
            if t == "int":
                return "int"
            if t == "float":
                return "float"
            if t == "str":
                return "str"

        return "float"  # ultimate default

    # ------------------------------------------------------------------
    # Scalar (int / float) handling
    # ------------------------------------------------------------------
    def _get_next_scalar(self, field_name: str, value_type: str):
        config_name = f"config_{field_name}"
        mode_key = f"{config_name}_mode"
        mode = self._state.get(mode_key)

        if mode == "fixed for dataset":
            val = self._state.get(config_name)
            if val is None:
                raise ValueError(
                    f"No value stored for '{field_name}' in "
                    "'fixed for dataset' mode."
                )
            if value_type == "str":
                return str(val)
            return int(val) if value_type == "int" else float(val)

        if mode == "fixed for trajectory":
            gen_key = f"_gen_{field_name}"
            if gen_key not in self._generators:
                self._generators[gen_key] = self._build_scalar_generator(
                    field_name, value_type
                )
            return self._generators[gen_key]()

        if mode is None or mode == "undefined":
            raise ValueError(
                f"Mode for '{field_name}' is undefined. "
                "Configure it in the UI first."
            )
        raise ValueError(f"Unknown mode '{mode}' for '{field_name}'")

    def _build_scalar_generator(self, field_name: str, value_type: str):
        config_name = f"config_{field_name}"
        dist_key = f"{config_name}_distribution"
        distribution = self._state.get(dist_key)

        if distribution == "uniform":
            min_val = self._find_param(field_name, "min")
            max_val = self._find_param(field_name, "max")
            if min_val is None or max_val is None:
                raise ValueError(
                    f"Uniform distribution for '{field_name}' "
                    "requires min and max parameters."
                )
            if value_type == "int":
                lo, hi = int(min_val), int(max_val)
                return lambda: self._rng.randint(lo, hi)
            lo, hi = float(min_val), float(max_val)
            return lambda: self._rng.uniform(lo, hi)

        if distribution == "normal":
            mean_val = self._find_param(field_name, "mean")
            std_val = self._find_param(field_name, "std")
            if mean_val is None or std_val is None:
                raise ValueError(
                    f"Normal distribution for '{field_name}' "
                    "requires mean and std parameters."
                )
            mu, sigma = float(mean_val), float(std_val)
            if value_type == "int":
                return lambda: int(self._rng.gauss(mu, sigma))
            return lambda: self._rng.gauss(mu, sigma)

        raise ValueError(f"Unknown distribution '{distribution}' for '{field_name}'")

    def _find_param(self, field_name: str, suffix: str):
        """Search session-state for a distribution parameter.

        Tries two naming patterns produced by universal_user_input_method:
        1. ``config_{field_name}_{suffix}``   (fallback pattern)
        2. Any key starting with ``config_{field_name}_`` and ending with
           ``_{suffix}``                       (Properties-based pattern)
        """
        config_name = f"config_{field_name}"

        # Pattern 1: direct
        key1 = f"{config_name}_{suffix}"
        val = self._state.get(key1)
        if val is not None:
            return val

        # Pattern 2: search
        for key, val in self._state.items():
            if (
                key.startswith(config_name + "_")
                and key.endswith(f"_{suffix}")
                and val is not None
            ):
                return val
        return None

    # ------------------------------------------------------------------
    # Bool handling
    # ------------------------------------------------------------------
    def _get_next_bool(self, field_name: str):
        config_name = f"config_{field_name}"
        mode_key = f"{config_name}_mode"
        mode = self._state.get(mode_key, "fixed for dataset")

        if mode == "fixed for dataset":
            return bool(self._state.get(config_name, False))

        if mode == "fixed for trajectory":
            prob = float(self._state.get(f"{config_name}_probability", 0.5))
            return self._rng.random() < prob

        raise ValueError(f"Unknown bool mode '{mode}' for '{field_name}'")

    # ------------------------------------------------------------------
    # Point handling
    # ------------------------------------------------------------------
    def _get_next_point(self, field_name: str) -> Point:
        # Strip trailing _point before adding suffix to avoid double _point
        _base = field_name[:-6] if field_name.endswith("_point") else field_name
        point_prefix = f"config_{_base}_point"
        mode_key = f"{point_prefix}_mode"
        dist_key = f"{point_prefix}_distribution"

        mode = self._state.get(mode_key, "undefined")
        spatial_dim = self._state.get("config_spatial_dimension", "2D")
        spatial_dim_type = self._state.get("config_spatial_dim_type", "continuous")
        is_3d = spatial_dim == "3D"

        if mode == "fixed for dataset":
            x = float(self._state.get(f"{point_prefix}_x", 0.0))
            y = float(self._state.get(f"{point_prefix}_y", 0.0))
            if is_3d:
                z = float(self._state.get(f"{point_prefix}_z", 0.0))
                return Point(x, y, z)
            return Point(x, y)

        if mode == "fixed for trajectory":
            distribution = self._state.get(dist_key)
            gen_key = f"_gen_{field_name}_point"
            if gen_key not in self._generators:
                self._generators[gen_key] = self._build_point_generator(
                    point_prefix, distribution, spatial_dim, spatial_dim_type
                )
            return self._generators[gen_key]()

        if mode == "undefined" or mode is None:
            # Not configured – return None so callers can treat it as
            # "no fixed point" and fall back to their own logic.
            return None
        raise ValueError(f"Unknown point mode '{mode}' for '{field_name}'")

    def _random_point_from_bbox(self) -> Point:
        """Return a uniformly random point inside the configured bounding box."""
        x = self._rng.uniform(
            float(self._state.get("config_get_next_x_min", 0.0)),
            float(self._state.get("config_get_next_x_max", 1.0)),
        )
        y = self._rng.uniform(
            float(self._state.get("config_get_next_y_min", 0.0)),
            float(self._state.get("config_get_next_y_max", 1.0)),
        )
        if self._state.get("config_spatial_dimension", "2D") == "3D":
            z = self._rng.uniform(
                float(self._state.get("config_get_next_z_min", 0.0)),
                float(self._state.get("config_get_next_z_max", 1.0)),
            )
            return Point(x, y, z)
        return Point(x, y)

    def _build_point_generator(
        self,
        point_prefix: str,
        distribution: str,
        spatial_dim: str,
        spatial_dim_type: str,
    ):
        is_3d = spatial_dim == "3D"
        discrete_type = self._state.get("config_discrete_dim_type", "grid-based")
        grid_res = float(self._state.get("config_grid_resolution", 1.0))

        if distribution == "uniform":
            return self._build_uniform_point_gen(
                point_prefix, is_3d, spatial_dim_type, discrete_type, grid_res
            )
        if distribution == "normal":
            return self._build_normal_point_gen(
                point_prefix, is_3d, spatial_dim_type, discrete_type, grid_res
            )
        if distribution == "discrete set":
            return self._build_discrete_point_gen(point_prefix, spatial_dim)

        raise ValueError(
            f"Unknown point distribution '{distribution}' " f"for '{point_prefix}'"
        )

    # -- Uniform point generator -----------------------------------------
    def _build_uniform_point_gen(self, pfx, is_3d, dim_type, disc_type, res):
        x0 = float(self._state.get(f"{pfx}_uniform_x_min", 0.0))
        x1 = float(self._state.get(f"{pfx}_uniform_x_max", 1.0))
        y0 = float(self._state.get(f"{pfx}_uniform_y_min", 0.0))
        y1 = float(self._state.get(f"{pfx}_uniform_y_max", 1.0))
        z0 = float(self._state.get(f"{pfx}_uniform_z_min", 0.0)) if is_3d else 0.0
        z1 = float(self._state.get(f"{pfx}_uniform_z_max", 1.0)) if is_3d else 0.0

        snap = dim_type == "discrete" and disc_type == "grid-based"

        def gen():
            x = self._rng.uniform(x0, x1)
            y = self._rng.uniform(y0, y1)
            if snap:
                x = self._snap_to_grid(x, res)
                y = self._snap_to_grid(y, res)
            if is_3d:
                z = self._rng.uniform(z0, z1)
                if snap:
                    z = self._snap_to_grid(z, res)
                return Point(x, y, z)
            return Point(x, y)

        return gen

    # -- Normal point generator ------------------------------------------
    def _build_normal_point_gen(self, pfx, is_3d, dim_type, disc_type, res):
        mx = float(self._state.get(f"{pfx}_normal_mean_x", 0.5))
        sx = float(self._state.get(f"{pfx}_normal_std_x", 0.1))
        my = float(self._state.get(f"{pfx}_normal_mean_y", 0.5))
        sy = float(self._state.get(f"{pfx}_normal_std_y", 0.1))
        mz = float(self._state.get(f"{pfx}_normal_mean_z", 0.5)) if is_3d else 0.0
        sz = float(self._state.get(f"{pfx}_normal_std_z", 0.1)) if is_3d else 0.0

        snap = dim_type == "discrete" and disc_type == "grid-based"

        def gen():
            x = self._rng.gauss(mx, sx)
            y = self._rng.gauss(my, sy)
            if snap:
                x = self._snap_to_grid(x, res)
                y = self._snap_to_grid(y, res)
            if is_3d:
                z = self._rng.gauss(mz, sz)
                if snap:
                    z = self._snap_to_grid(z, res)
                return Point(x, y, z)
            return Point(x, y)

        return gen

    # -- Discrete-set point generator ------------------------------------
    def _build_discrete_point_gen(self, pfx, spatial_dim):
        points = self._collect_discrete_points(pfx, spatial_dim)
        if not points:
            raise ValueError(f"No discrete points defined for '{pfx}'")

        def gen():
            return self._rng.choice(points)

        return gen

    def _collect_discrete_points(self, pfx, spatial_dim):
        is_3d = spatial_dim == "3D"
        count_key = f"{pfx}_discrete_count"
        pts_key = f"{pfx}_discrete_points"
        n = int(self._state.get(count_key, 0))
        points = []
        for i in range(n):
            pk = f"{pts_key}_point_{i}"
            x = self._state.get(f"{pk}_x")
            y = self._state.get(f"{pk}_y")
            if x is not None and y is not None:
                if is_3d:
                    z = self._state.get(f"{pk}_z", 0.0)
                    points.append(Point(float(x), float(y), float(z)))
                else:
                    points.append(Point(float(x), float(y)))
        return points

    @staticmethod
    def _snap_to_grid(value: float, resolution: float) -> float:
        if resolution <= 0:
            return value
        return round(value / resolution) * resolution

    # ------------------------------------------------------------------
    # Distance Functions
    # ------------------------------------------------------------------
    @property
    def distance_function(self):
        """Return the distance callable selected in the UI (dimension-aware)."""
        name = self._state.get("config_distance_function", "Euclidean")
        dims = 3 if self._state.get("config_spatial_dimension", "2D") == "3D" else 2

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
