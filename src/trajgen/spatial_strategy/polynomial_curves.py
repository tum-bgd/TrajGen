from shapely import LineString
from ..config import Config
from ..trajectory import Trajectory
import numpy as np
from scipy.interpolate import CubicSpline


from .requirements_helpers import bbox_requirements


class PolynomialCurvesStrategy:
    """Generate closed trajectories using cubic spline interpolation through random control points."""

    config: Config

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, id: int) -> Trajectory:
        # Generate control points
        control_points = self.config.point_generator(self.config.num_control_points)

        # First and last control point = SAME (closure)
        if self.config.closed_loop:
            control_points[-1] = control_points[0]

        # Parameter t for control points (roughly uniform)
        t_control = np.linspace(0.0, 1.0, self.config.num_control_points)

        # Cubic spline interpolation through control points
        spline_x = CubicSpline(
            t_control, [point.x for point in control_points], bc_type="clamped"
        )
        spline_y = CubicSpline(
            t_control, [point.y for point in control_points], bc_type="clamped"
        )

        n_points = self.config.get_next_length()
        # Evaluate at high resolution
        t = np.linspace(0.0, 1.0, n_points)
        x = spline_x(t)
        y = spline_y(t)

        # final clip (almost never triggers with small margin)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

        return Trajectory(id=id, ls=LineString(np.column_stack((x, y))))

    @staticmethod
    def get_requirements(spatial_dim: str = "2D") -> dict:
        return {
            "get_next_length": {
                "short_name": "Trajectory Length",
                "type": "get_int_function",
                "default": 100,
                "description": "Number of points evaluated along the spline.",
                "optional": False,
            },
            "get_next_num_control_points": {
                "short_name": "Control Points",
                "type": "get_int_function",
                "default": 5,
                "description": "Number of random control points for the spline.",
                "optional": False,
            },
            "get_next_closed_loop": {
                "short_name": "Closed Loop",
                "type": "get_bool_function",
                "default": False,
                "description": "Whether to close the curve by matching first and last control points.",
                "optional": True,
            },
            **bbox_requirements(spatial_dim),
        }
