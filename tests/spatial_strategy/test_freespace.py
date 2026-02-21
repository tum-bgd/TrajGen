import pytest
from unittest.mock import Mock, patch
from shapely.geometry import Point, Polygon, LineString
from trajgen.spatial_strategy.freespace import FreespaceStrategy, Obstacle
from trajgen.config import Config
from trajgen.trajectory import Trajectory
from trajgen.point_generator import PointGenerator
import random


class TestObstacle:
    def test_obstacle_creation(self):
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        obstacle = Obstacle(polygon=polygon, id=1, buffered=polygon.buffer(0.1))

        assert obstacle.polygon == polygon
        assert obstacle.id == 1
        assert obstacle.buffered == polygon.buffer(0.1)


class TestFreespaceStrategy:
    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.x_min = 0.0
        config.y_min = 0.0
        config.x_max = 1.0
        config.y_max = 1.0
        config.seed = 42
        config.length_min = 5
        config.length_max = 15
        config.num_obstacles = 2
        config.obstacle_size_min = 0.01
        config.obstacle_size_max = 0.15
        config.point_generator = Mock(spec=PointGenerator)
        config.size_generator = Mock(spec=PointGenerator)
        config.point_generator.side_effect = [
            Point(0.2, 0.2),
            Point(0.5, 0.5),
            Point(0.8, 0.8),
            Point(0.95, 0.95),
            Point(0.8, 0.2),
            Point(0.2, 0.8),
            Point(0.5, 0.2),
            Point(0.2, 0.5),
            Point(0.5, 0.8),
            Point(0.8, 0.5),
        ]
        config.size_generator.return_value = Point(0.1, 0.1)
        config.deviation_factor = 0.1
        config.get_next_length.return_value = 10
        config.get_start_point.return_value = Point(0.0, 0.0)
        config.get_end_point.return_value = Point(1.0, 1.0)
        config.get_next_length.return_value = 10
        return config

    @pytest.fixture
    def strategy(self, mock_config):
        return FreespaceStrategy(mock_config)

    def test_init(self, mock_config):
        strategy = FreespaceStrategy(mock_config)

        assert strategy.config == mock_config
        assert strategy.start == Point(0.0, 0.0)
        assert strategy.end == Point(1.0, 1.0)

    def test_generate_obstacles_empty(self, strategy):
        obstacles = strategy._generate_obstacles(0)
        assert obstacles == []

    def test_generate_obstacles_single(self, strategy):
        obstacles = strategy._generate_obstacles(1)

        assert len(obstacles) == 1
        assert isinstance(obstacles[0], Obstacle)
        assert isinstance(obstacles[0].polygon, Polygon)
        assert obstacles[0].id == 0

    def test_generate_obstacles_multiple(self, strategy):
        obstacles = strategy._generate_obstacles(3)

        assert len(obstacles) == 3
        for i, obstacle in enumerate(obstacles):
            assert isinstance(obstacle, Obstacle)
            assert obstacle.id == i
            assert isinstance(obstacle.polygon, Polygon)

    def test_generate_obstacles_non_overlapping(self, strategy):
        obstacles = strategy._generate_obstacles(2)

        # Check obstacles don't overlap
        for i in range(len(obstacles)):
            for j in range(i + 1, len(obstacles)):
                assert not obstacles[i].polygon.intersects(obstacles[j].polygon)

    def test_plan_path_around_obstacles_no_obstacles(self, strategy):
        strategy.obstacles = []
        rng = random.Random(strategy.config.seed)
        path = strategy._plan_path_around_obstacles(rng, num_points=5)

        assert len(path) == 5
        assert path[0] == strategy.start
        assert path[-1] == strategy.end
        assert all(isinstance(point, Point) for point in path)

    def test_plan_path_around_obstacles_with_obstacles(self, strategy):
        # Create obstacles that don't block the path
        poly1 = Polygon([(0.8, 0.8), (0.9, 0.8), (0.9, 0.9), (0.8, 0.9)])
        obstacle1 = Obstacle(polygon=poly1, buffered=poly1.buffer(0.1), id=0)
        strategy.obstacles = [obstacle1]

        rng = random.Random(strategy.config.seed)
        path = strategy._plan_path_around_obstacles(rng, num_points=5)

        assert len(path) == 5
        assert path[0] == strategy.start
        assert path[-1] == strategy.end

    def test_plan_path_bounds_clamping(self, strategy):
        strategy.obstacles = []
        # Mock rng to return values outside bounds

        rng = Mock(random.Random)
        rng.gauss.side_effect = [2.0, 2.0]
        path = strategy._plan_path_around_obstacles(rng, num_points=3)

        # All points should be within bounds [0, 1]
        for point in path:
            assert 0 <= point.x <= 1
            assert 0 <= point.y <= 1

    def test_smooth_path_empty(self, strategy):
        result = strategy._smooth_path([])
        assert result == []

    def test_smooth_path_single_point(self, strategy):
        points = [Point(0.5, 0.5)]
        result = strategy._smooth_path(points)
        assert result == [(0.5, 0.5)]

    def test_smooth_path_two_points(self, strategy):
        points = [Point(0.0, 0.0), Point(1.0, 1.0)]
        result = strategy._smooth_path(points)
        assert len(result) == 2
        assert result[0] == (0.0, 0.0)
        assert result[1] == (1.0, 1.0)

    def test_smooth_path_multiple_points(self, strategy):
        points = [Point(0.0, 0.0), Point(0.5, 0.5), Point(1.0, 1.0)]
        result = strategy._smooth_path(points)

        assert len(result) == 3
        assert all(isinstance(coord, tuple) for coord in result)
        assert all(len(coord) == 2 for coord in result)

    def test_call_returns_trajectory(self, strategy):
        # Need to set obstacles attribute for __call__ to work
        strategy.obstacles = []

        # Mock the seed attribute that's used in __call__
        strategy.seed = 42

        trajectory = strategy(0)

        assert isinstance(trajectory, Trajectory)
        assert trajectory.id == 0
        assert isinstance(trajectory.ls, LineString)

    def test_call_different_trajectories(self, strategy):
        strategy.obstacles = []
        strategy.seed = 42

        traj1 = strategy(0)
        traj2 = strategy(1)

        assert traj1.id != traj2.id
        # Trajectories should be different due to different seeds
        assert traj1.ls.coords[:] != traj2.ls.coords[:]

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.subplots")
    def test_visualize_obstacles(self, mock_subplots, mock_show, strategy):
        # Setup mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Setup strategy with obstacles
        strategy.obstacles = [
            Obstacle(
                polygon=Polygon([(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)]),
                buffered=Polygon(
                    [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)]
                ).buffer(0.1),
                id=0,
            )
        ]
        strategy.seed = 42

        strategy.visualize_obstacles()

        # Verify matplotlib functions were called
        mock_subplots.assert_called_once_with(figsize=(10, 10))
        mock_show.assert_called_once()
        assert mock_ax.fill.called
        assert mock_ax.plot.called
        assert mock_ax.set_xlim.called
        assert mock_ax.set_ylim.called
