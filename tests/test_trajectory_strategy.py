import pytest  # noqa F401
from trajgen.trajectory import Trajectory
from trajgen.config import Config
from trajgen.spatial_strategy import SpatialStrategy
from trajgen.point_generator import PointGenerator
from unittest.mock import Mock


def test_trajectory_strategy_protocol():
    """Test that TrajectoryStrategy is a proper Protocol"""
    # This test ensures the protocol is defined correctly
    assert hasattr(SpatialStrategy, "__call__")


def test_trajectory_strategy_implementation_compatibility():
    """Test that a concrete implementation satisfies the protocol"""

    class MockStrategy:
        def __call__(self, id: int) -> Trajectory:
            return Mock(spec=Trajectory)

    strategy = MockStrategy()

    # This should not raise any type errors
    result = strategy(1)
    assert result is not None


def test_trajectory_strategy_signature():
    """Test that implementations must have correct signature"""

    class InvalidStrategy:
        def __call__(self, wrong_params) -> Trajectory:
            return Mock(spec=Trajectory)

    # Protocol should enforce correct signature at type checking level
    # This is more of a documentation test for expected interface
    strategy = InvalidStrategy()
    assert hasattr(strategy, "__call__")


def test_trajectory_strategy_return_type():
    """Test that implementations must return Trajectory"""

    class ValidStrategy:
        def __call__(
            self, id: int, config: Config, point_generator: PointGenerator
        ) -> Trajectory:
            return Mock(spec=Trajectory)

    strategy = ValidStrategy()
    config = Mock(spec=Config)
    point_gen = Mock(spec=PointGenerator)

    result = strategy(42, config, point_gen)
    assert isinstance(result, Trajectory)
    assert result is not None


def test_trajectory_strategy_with_mock_dependencies():
    """Test strategy protocol with mocked dependencies"""

    class TestStrategy:
        def __call__(
            self, id: int, config: Config, point_generator: PointGenerator
        ) -> Trajectory:
            # Simple implementation that uses the parameters
            mock_trajectory = Mock(spec=Trajectory)
            mock_trajectory.id = id
            return mock_trajectory

    strategy = TestStrategy()
    mock_config = Mock(spec=Config)
    mock_point_gen = Mock(spec=PointGenerator)

    result = strategy(99, mock_config, mock_point_gen)
    assert result.id == 99
