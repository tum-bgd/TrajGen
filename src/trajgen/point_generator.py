from abc import abstractmethod
from shapely.geometry import Point
import random


class PointGenerator:
    @abstractmethod
    def __call__(self, num_points: int | None = None) -> Point | list[Point]:
        raise NotImplementedError()


class RandomPointGenerator2D(PointGenerator):
    def __init__(
        self, row_range: tuple[float, float], col_range: tuple[float, float], seed: int
    ):
        self.row_range = row_range
        self.col_range = col_range
        self.seed = seed
        self.rng = random.Random(seed)

    def __call__(self, num_points: int | None = None) -> Point | list[Point]:
        if num_points is None:
            row = self.rng.uniform(*self.row_range)
            col = self.rng.uniform(*self.col_range)
            return Point(row, col)
        else:
            points = []
            for _ in range(num_points):
                row = self.rng.uniform(*self.row_range)
                col = self.rng.uniform(*self.col_range)
                points.append(Point(row, col))
            return points

    def __iter__(self):
        for _ in range(self.num_points):
            row = self.rng.uniform(*self.row_range)
            col = self.rng.uniform(*self.col_range)
            yield Point(row, col)


class GridPointGenerator2D(PointGenerator):
    def __init__(
        self,
        row_range: tuple[float, float],
        col_range: tuple[float, float],
        step: float,
    ):
        self.row_range = row_range
        self.col_range = col_range
        self.step = step

    def __call__(self, num_points: int | None = None) -> Point | list[Point]:
        points = []
        row = self.row_range[0]
        while row <= self.row_range[1]:
            col = self.col_range[0]
            while col <= self.col_range[1]:
                points.append(Point(row, col))
                col += self.step
            row += self.step

        if num_points is None:
            return points[0] if points else None
        else:
            if not points:
                return []
            # Cycle through available points with random shuffling for each complete cycle
            result = []
            remaining = num_points
            while remaining > 0:
                # Create a shuffled copy of points for this cycle
                cycle_points = points.copy()
                random.shuffle(cycle_points)

                # Take what we need from this shuffled cycle
                take = min(remaining, len(cycle_points))
                result.extend(cycle_points[:take])
                remaining -= take

            return result

    def __iter__(self):
        row = self.row_range[0]
        while row <= self.row_range[1]:
            col = self.col_range[0]
            while col <= self.col_range[1]:
                yield Point(row, col)
                col += self.step
            row += self.step


class RandomPointGenerator3D(PointGenerator):
    def __init__(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
        seed: int,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.seed = seed
        self.rng = random.Random(seed)

    def __call__(self, num_points: int | None = None) -> Point | list[Point]:
        if num_points is None:
            x = self.rng.uniform(*self.x_range)
            y = self.rng.uniform(*self.y_range)
            z = self.rng.uniform(*self.z_range)
            return Point(x, y, z)
        else:
            points = []
            for _ in range(num_points):
                x = self.rng.uniform(*self.x_range)
                y = self.rng.uniform(*self.y_range)
                z = self.rng.uniform(*self.z_range)
                points.append(Point(x, y, z))
            return points

    def __iter__(self):
        for _ in range(self.num_points):
            x = self.rng.uniform(*self.x_range)
            y = self.rng.uniform(*self.y_range)
            z = self.rng.uniform(*self.z_range)
            yield Point(x, y, z)


class GridPointGenerator3D(PointGenerator):
    def __init__(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
        step: float,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.step = step

    def __call__(self, num_points: int | None = None) -> Point | list[Point]:
        points = []
        x = self.x_range[0]
        while x <= self.x_range[1]:
            y = self.y_range[0]
            while y <= self.y_range[1]:
                z = self.z_range[0]
                while z <= self.z_range[1]:
                    points.append(Point(x, y, z))
                    z += self.step
                y += self.step
            x += self.step

        if num_points is None:
            return points[0] if points else None
        else:
            if not points:
                return []
            # Cycle through available points with random shuffling for each complete cycle
            result = []
            remaining = num_points
            while remaining > 0:
                # Create a shuffled copy of points for this cycle
                cycle_points = points.copy()
                random.shuffle(cycle_points)

                # Take what we need from this shuffled cycle
                take = min(remaining, len(cycle_points))
                result.extend(cycle_points[:take])
                remaining -= take

            return result

    def __iter__(self):
        x = self.x_range[0]
        while x <= self.x_range[1]:
            y = self.y_range[0]
            while y <= self.y_range[1]:
                z = self.z_range[0]
                while z <= self.z_range[1]:
                    yield Point(x, y, z)
                    z += self.step
                y += self.step
            x += self.step


class SmoothnessConstrainedPointGenerator(PointGenerator):
    def __init__(self, base_generator: PointGenerator, max_distance: float):
        self.base_generator = base_generator
        self.max_distance = max_distance

    def __call__(self, num_points: int | None = None) -> Point | list[Point]:
        if num_points is None:
            while True:
                point = self.base_generator()
                if (
                    not hasattr(self, "last_point")
                    or point.distance(self.last_point) <= self.max_distance
                ):
                    self.last_point = point
                    return point
        else:
            points = []
            for _ in range(num_points):
                while True:
                    point = self.base_generator()
                    if not points or point.distance(points[-1]) <= self.max_distance:
                        points.append(point)
                        break
            return points


class ConstrainedPointGenerator(PointGenerator):
    def __init__(self, base_generator: PointGenerator, constraint_func):
        self.base_generator = base_generator
        self.constraint_func = constraint_func

    def __call__(self, num_points: int | None = None) -> Point | list[Point]:
        if num_points is None:
            while True:
                point = self.base_generator()
                if self.constraint_func(point):
                    return point
        else:
            points = []
            while len(points) < num_points:
                point = self.base_generator()
                if self.constraint_func(point):
                    points.append(point)
            return points


class ConstrainFunction:
    def __call__(point: Point) -> bool:
        raise NotImplementedError()


class StartEndPointGenerator(PointGenerator):
    """Wrapper that pins the first and/or last generated point to fixed values.

    Parameters
    ----------
    inner : PointGenerator
        The underlying generator used for all non-pinned points.
    start : Point | None
        If given, ``points[0]`` is replaced with this value.
    end : Point | None
        If given, ``points[-1]`` is replaced with this value.
    """

    def __init__(
        self,
        inner: PointGenerator,
        start: Point | None = None,
        end: Point | None = None,
    ):
        self.inner = inner
        self.start = start
        self.end = end

    def __call__(self, num_points: int | None = None) -> Point | list[Point]:
        if num_points is None:
            # Single-point request – just delegate (nothing to pin)
            return self.inner(num_points)

        points = self.inner(num_points)

        if self.start is not None and len(points) > 0:
            points[0] = self.start
        if self.end is not None and len(points) > 0:
            points[-1] = self.end

        return points
