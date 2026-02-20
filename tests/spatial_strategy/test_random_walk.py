import pytest  # noqa F401
from unittest.mock import Mock
from shapely.geometry import LineString
from trajgen.spatial_strategy import RandomWalkStrategy
from trajgen.trajectory import Trajectory
from trajgen.config import Config
from trajgen.point_generator import PointGenerator


class TestRandomWalkStrategy:

    def test_init_stores_parameters(self):
        config = Mock(spec=Config)
        config.seed = 123
        point_generator = Mock(spec=PointGenerator)
        config.point_generator = point_generator
        strategy = RandomWalkStrategy(config=config)

        assert strategy.seed == 123
        assert strategy.config == config
        assert strategy.point_generator == point_generator

    def test_init_default_seed(self):
        config = Mock(spec=Config)
        config.seed = 42
        strategy = RandomWalkStrategy(config=config)

        assert strategy.seed == 42

    def test_call_returns_trajectory_with_correct_id(self):
        config = Mock(spec=Config)
        config.num_points = 5

        # Create real points instead of mocks to avoid attribution errors in shapely
        from shapely.geometry import Point

        mock_points = [Point(i, i + 1) for i in range(5)]

        point_generator = Mock(spec=PointGenerator)
        point_generator.return_value = mock_points
        config.point_generator = point_generator

        # Add required methods for config
        config.get_next_length.return_value = 5
        config.get_start_point.return_value = None

        strategy = RandomWalkStrategy(config=config)
        result = strategy(id=42)

        assert isinstance(result, Trajectory)
        assert result.id == 42

    def test_call_uses_config_num_points(self):
        config = Mock(spec=Config)
        config.num_points = 10

        from shapely.geometry import Point

        mock_points = [Point(i, i + 1) for i in range(10)]
        point_generator = Mock(spec=PointGenerator)
        point_generator.return_value = mock_points
        config.point_generator = point_generator
        config.get_start_point.return_value = None
        config.get_next_length.side_effect = [10, 20, 30]
        strategy = RandomWalkStrategy(config=config)
        strategy(id=1)

        # Uses get_next_length value
        config.point_generator.assert_called_once_with(10)

    def test_call_creates_linestring_from_points(self):
        config = Mock(spec=Config)
        config.get_next_length.return_value = 3
        config.closed_loop = False
        config.get_start_point.return_value = None

        from shapely.geometry import Point

        mock_points = [Point(1, 2), Point(3, 4), Point(5, 6)]
        point_generator = Mock(spec=PointGenerator)
        point_generator.return_value = mock_points
        config.point_generator = point_generator

        strategy = RandomWalkStrategy(config=config)
        result = strategy(id=1)

        expected_coords = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        assert isinstance(result.ls, LineString)
        assert list(result.ls.coords) == expected_coords

    def test_call_with_single_point(self):
        config = Mock(spec=Config)
        config.get_next_length.return_value = 1
        config.get_start_point.return_value = None

        from shapely.geometry import Point

        mock_points = [Point(10, 20)]
        point_generator = Mock(spec=PointGenerator)
        point_generator.return_value = mock_points
        config.point_generator = point_generator

        strategy = RandomWalkStrategy(config=config)
        result = strategy(id=99)

        assert result.id == 99
        assert isinstance(result.ls, LineString)
        # Shapely behavior for single point LineString might be just that point? No.
        # It's either empty or has 1 coord.
        assert len(result.ls.coords) <= 1

    def test_call_with_zero_points(self):
        config = Mock(spec=Config)
        config.get_next_length.return_value = 0
        config.get_start_point.return_value = None

        point_generator = Mock(spec=PointGenerator)
        point_generator.return_value = []
        config.point_generator = point_generator

        strategy = RandomWalkStrategy(config=config)
        result = strategy(id=1)

        assert isinstance(result.ls, LineString)
        # Empty LineString has 0 coords
        assert len(result.ls.coords) == 0

    def test_call_different_ids_produce_different_trajectories(self):
        config = Mock(spec=Config)
        config.get_next_length.return_value = 3
        config.get_start_point.return_value = None

        from shapely.geometry import Point

        mock_points1 = [Point(1, 2), Point(3, 4), Point(5, 6)]
        mock_points2 = [Point(7, 8), Point(9, 10), Point(11, 12)]

        point_generator = Mock(spec=PointGenerator)
        point_generator.side_effect = [mock_points1, mock_points2]
        config.point_generator = point_generator

        strategy = RandomWalkStrategy(config=config)

        traj1 = strategy(id=1)
        traj2 = strategy(id=2)

        assert traj1.id == 1
        assert traj2.id == 2
        assert list(traj1.ls.coords) != list(traj2.ls.coords)
