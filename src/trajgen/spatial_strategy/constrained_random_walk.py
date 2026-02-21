from ..trajectory import Trajectory
from shapely.geometry import LineString
from ..config import Config
from ..point_generator import PointGenerator


class ConstrainedRandomWalkStrategy:
    seed: int
    config: Config
    point_generator: PointGenerator

    def __init__(self, config: Config):
        self.seed = config.seed
        self.config = config
        self.point_generator = config.point_generator

    def __call__(self, id: int) -> Trajectory:
        points = self.point_generator(self.config.get_next_length())
        if self.config.get_next_closed_loop and len(points) > 1:
            start = points[0]
            points[-1] = start
        line = LineString([(point.x, point.y) for point in points])
        return Trajectory(id=id, ls=line)
