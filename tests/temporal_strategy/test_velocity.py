import pytest

from trajgen.temporal_strategy.velocity import VelocityTemporalStrategy


class MockConfig:
    def __init__(self, velocity):
        self.velocity = velocity

    def get_next_tmin(self):
        return 0

    def distance_function(self, coord1, coord2):
        # Simple Euclidean distance
        return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5

    def get_next_velocity(self):
        return self.velocity


# Mock trajectory with linestring coordinates
class MockTrajectory:
    def __init__(self, coords):
        self.ls = MockLineString(coords)
        self.time_stamps = None

    def set_time(self, time_stamps):
        self.time_stamps = time_stamps


class MockLineString:
    def __init__(self, coords):
        self.coords = coords


class TestVelocityTemporalStrategy:
    def test_velocity_temporal_strategy_basic(self):
        # Mock config with fixed velocity and simple distance function

        config = MockConfig(velocity=10.0)
        strategy = VelocityTemporalStrategy(config)

        # Test with simple coordinates
        coords = [(0, 0), (3, 4), (6, 8)]  # distances: 5, 5
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == 0.5  # 5/10
        assert result.time_stamps[2] == 1  # 5/10
        assert len(result.time_stamps) == 3

    def test_single_point_trajectory(self):
        """Test with trajectory containing only one point."""
        config = MockConfig(velocity=10.0)
        strategy = VelocityTemporalStrategy(config)

        coords = [(0, 0)]
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        assert len(result.time_stamps) == 1
        assert result.time_stamps[0] == 0

    def test_empty_trajectory(self):
        """Test with empty trajectory."""
        config = MockConfig(velocity=10.0)
        strategy = VelocityTemporalStrategy(config)

        coords = []
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # Empty trajectory results in empty timestamps
        assert len(result.time_stamps) == 0

    def test_two_point_trajectory(self):
        """Test with trajectory containing exactly two points."""
        config = MockConfig(velocity=5.0)
        strategy = VelocityTemporalStrategy(config)

        coords = [(0, 0), (10, 0)]  # distance: 10
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        assert len(result.time_stamps) == 2
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == 2.0  # 10/5

    def test_zero_distance_segments(self):
        """Test with trajectory containing duplicate points (zero distance)."""
        config = MockConfig(velocity=10.0)
        strategy = VelocityTemporalStrategy(config)

        coords = [(0, 0), (0, 0), (3, 4)]  # distances: 0, 5
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        assert len(result.time_stamps) == 3
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == 0  # No time passes for zero distance
        assert result.time_stamps[2] == 0.5  # 5/10

    def test_high_velocity(self):
        """Test with high velocity value."""
        config = MockConfig(velocity=100.0)
        strategy = VelocityTemporalStrategy(config)

        coords = [(0, 0), (30, 40), (60, 80)]  # distances: 50, 50
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        assert len(result.time_stamps) == 3
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(0.5, abs=1e-6)  # 50/100
        assert result.time_stamps[2] == pytest.approx(1.0, abs=1e-6)  # 50/100

    def test_low_velocity(self):
        """Test with low velocity value."""
        config = MockConfig(velocity=0.1)
        strategy = VelocityTemporalStrategy(config)

        coords = [(0, 0), (1, 0), (2, 0)]  # distances: 1, 1
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        assert len(result.time_stamps) == 3
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(10.0, abs=1e-6)  # 1/0.1
        assert result.time_stamps[2] == pytest.approx(20.0, abs=1e-6)  # 1/0.1

    def test_varying_distances(self):
        """Test with trajectory segments of varying distances."""
        config = MockConfig(velocity=2.0)
        strategy = VelocityTemporalStrategy(config)

        coords = [(0, 0), (2, 0), (2, 6), (8, 6)]  # distances: 2, 6, 6
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        assert len(result.time_stamps) == 4
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(1.0, abs=1e-6)  # 2/2
        assert result.time_stamps[2] == pytest.approx(4.0, abs=1e-6)  # 6/2
        assert result.time_stamps[3] == pytest.approx(7.0, abs=1e-6)  # 6/2

    def test_diagonal_movement(self):
        """Test with diagonal movement patterns."""
        config = MockConfig(velocity=5.0)
        strategy = VelocityTemporalStrategy(config)

        coords = [(0, 0), (3, 4), (0, 8)]  # distances: 5, 5
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        assert len(result.time_stamps) == 3
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(1.0, abs=1e-6)  # 5/5
        assert result.time_stamps[2] == pytest.approx(2.0, abs=1e-6)  # 5/5

    def test_many_points_trajectory(self):
        """Test with trajectory containing many points."""
        config = MockConfig(velocity=1.0)
        strategy = VelocityTemporalStrategy(config)

        # Create a trajectory with 10 points, each 1 unit apart
        coords = [(i, 0) for i in range(10)]
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        assert len(result.time_stamps) == 10
        # Each segment has distance 1, velocity 1, so time step is 1
        for i in range(10):
            assert result.time_stamps[i] == pytest.approx(float(i), abs=1e-6)

    def test_fractional_times(self):
        """Test that produces fractional timestamps."""
        config = MockConfig(velocity=3.0)
        strategy = VelocityTemporalStrategy(config)

        coords = [(0, 0), (1, 0), (3, 0)]  # distances: 1, 2
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        assert len(result.time_stamps) == 3
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(1 / 3, abs=1e-6)  # 1/3
        assert result.time_stamps[2] == pytest.approx(1.0, abs=1e-6)  # 2/3

    def test_cumulative_time_calculation(self):
        """Test that timestamps accumulate correctly."""
        config = MockConfig(velocity=4.0)
        strategy = VelocityTemporalStrategy(config)

        coords = [(0, 0), (4, 0), (4, 3), (8, 3)]  # distances: 4, 3, 4
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        assert len(result.time_stamps) == 4
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(1.0, abs=1e-6)  # 4/4 = 1
        assert result.time_stamps[2] == pytest.approx(1.75, abs=1e-6)  # 1 + 3/4 = 1.75
        assert result.time_stamps[3] == pytest.approx(
            2.75, abs=1e-6
        )  # 1.75 + 4/4 = 2.75

    def test_negative_coordinates(self):
        """Test with negative coordinate values."""
        config = MockConfig(velocity=2.0)
        strategy = VelocityTemporalStrategy(config)

        coords = [(-3, -4), (0, 0), (3, 4)]  # distances: 5, 5
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        assert len(result.time_stamps) == 3
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(2.5, abs=1e-6)  # 5/2
        assert result.time_stamps[2] == pytest.approx(5.0, abs=1e-6)  # 5/2

    def test_mixed_coordinate_patterns(self):
        """Test with mixed coordinate patterns including backtracking."""
        config = MockConfig(velocity=10.0)
        strategy = VelocityTemporalStrategy(config)

        coords = [(0, 0), (10, 0), (5, 0), (15, 0)]  # distances: 10, 5, 10
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        assert len(result.time_stamps) == 4
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(1.0, abs=1e-6)  # 10/10
        assert result.time_stamps[2] == pytest.approx(1.5, abs=1e-6)  # 1.0 + 5/10
        assert result.time_stamps[3] == pytest.approx(2.5, abs=1e-6)  # 1.5 + 10/10
