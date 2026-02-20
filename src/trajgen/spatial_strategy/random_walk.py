from ..trajectory import Trajectory
from shapely.geometry import LineString
from ..config import Config
from ..point_generator import PointGenerator


from .requirements_helpers import bbox_requirements


class RandomWalkStrategy:
    seed: int
    config: Config
    point_generator: PointGenerator

    def __init__(self, config: Config):
        self.seed = config.seed
        self.config = config
        self.point_generator = config.point_generator

    def __call__(self, id: int) -> Trajectory:
        length = self.config.get_next_length()
        points = self.point_generator(length)

        # Apply configured start / end points (only when explicitly set)
        try:
            start = self.config.get_start_point()
            if start is not None and len(points) > 0:
                points[0] = start
        except Exception:
            pass

        try:
            end = self.config.get_end_point()
            if end is not None and len(points) > 0:
                points[-1] = end
        except Exception:
            pass

        if self.config.closed_loop and len(points) > 1:
            points[-1] = points[0]

        coords = [(p.x, p.y, p.z) if p.has_z else (p.x, p.y) for p in points]
        line = LineString(coords)
        return Trajectory(id=id, ls=line)

    @staticmethod
    def get_requirements(spatial_dim: str = "2D") -> dict:
        return {
            "get_next_length": {
                "short_name": "Trajectory Length",
                "type": "get_int_function",
                "default": 10,
                "description": "Number of points per trajectory.",
                "optional": False,
            },
            "get_next_closed_loop": {
                "short_name": "Closed Loop",
                "type": "get_bool_function",
                "default": False,
                "description": "Whether to snap the last point to the first.",
                "optional": True,
            },
            **bbox_requirements(spatial_dim),
            "get_start_point": {
                "short_name": "Start Point",
                "type": "get_point_function",
                "default": None,
                "description": "Optional fixed or sampled start point.",
                "optional": True,
            },
            "get_end_point": {
                "short_name": "End Point",
                "type": "get_point_function",
                "default": None,
                "description": "Optional fixed or sampled end point.",
                "optional": True,
            },
        }
