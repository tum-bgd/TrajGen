import pytest
import math

from trajgen.temporal_strategy.acceleration import AccelerationTemporalStrategy


class MockConfig:
    def __init__(self, initial_velocity, acceleration, tmin=0):
        self.initial_velocity = initial_velocity
        self.acceleration = acceleration
        self.tmin = tmin

    def get_next_tmin(self):
        return self.tmin

    def distance_function(self, coord1, coord2):
        # Simple Euclidean distance
        return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5

    def get_next_velocity(self):
        return self.initial_velocity

    def get_next_acceleration(self):
        return self.acceleration


class MockVariableConfig:
    """Mock config for tests requiring variable acceleration/velocity."""

    def __init__(self, initial_velocity, accelerations, tmin=0):
        self.initial_velocity = initial_velocity
        self.accelerations = accelerations
        self.acceleration_index = 0
        self.tmin = tmin

    def get_next_tmin(self):
        return self.tmin

    def distance_function(self, coord1, coord2):
        return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5

    def get_next_velocity(self):
        return self.initial_velocity

    def get_next_acceleration(self):
        if self.acceleration_index < len(self.accelerations):
            acc = self.accelerations[self.acceleration_index]
            self.acceleration_index += 1
            return acc
        return 0


class MockTrajectory:
    def __init__(self, coords):
        self.ls = MockLineString(coords)
        self.time_stamps = None

    def set_time(self, time_stamps):
        self.time_stamps = time_stamps


class MockLineString:
    def __init__(self, coords):
        self.coords = coords


class TestAccelerationTemporalStrategy:
    def test_basic_acceleration(self):
        """Test basic acceleration with constant positive acceleration."""
        config = MockConfig(initial_velocity=0, acceleration=2.0)
        strategy = AccelerationTemporalStrategy(config)

        # Test with simple coordinates: distance = 5
        coords = [(0, 0), (3, 4)]  # distance: 5
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # Using kinematic equation: s = v0*t + 0.5*a*t^2
        # For v0=0, a=2, s=5: 5 = 0.5*2*t^2 => t = sqrt(5)
        expected_time = math.sqrt(5)
        assert len(result.time_stamps) == 2
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(expected_time, abs=1e-6)

    def test_initial_velocity_with_acceleration(self):
        """Test with initial velocity and positive acceleration."""
        config = MockConfig(initial_velocity=3.0, acceleration=1.0)
        strategy = AccelerationTemporalStrategy(config)

        coords = [(0, 0), (10, 0)]  # distance: 10
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # Using kinematic equation: t = (-v0 + sqrt(v0^2 + 2*a*s)) / a
        # t = (-3 + sqrt(9 + 20)) / 1 = (-3 + sqrt(29)) / 1
        expected_time = (-3 + math.sqrt(29)) / 1
        assert len(result.time_stamps) == 2
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(expected_time, abs=1e-6)

    def test_zero_acceleration_with_velocity(self):
        """Test with zero acceleration but non-zero velocity (constant velocity)."""
        config = MockConfig(initial_velocity=5.0, acceleration=0.0)
        strategy = AccelerationTemporalStrategy(config)

        coords = [(0, 0), (15, 0)]  # distance: 15
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # With zero acceleration, it's constant velocity: t = s/v
        expected_time = 15.0 / 5.0  # 3.0
        assert len(result.time_stamps) == 2
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(expected_time, abs=1e-6)

    def test_zero_velocity_zero_acceleration(self):
        """Test edge case with both zero velocity and zero acceleration."""
        config = MockConfig(initial_velocity=0.0, acceleration=0.0)
        strategy = AccelerationTemporalStrategy(config)

        coords = [(0, 0), (5, 0)]  # distance: 5
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # Can't move with zero velocity and acceleration, time stays the same
        assert len(result.time_stamps) == 2
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == 0  # Same timestamp

    def test_multiple_segments_constant_acceleration(self):
        """Test with multiple segments and constant acceleration."""
        config = MockConfig(initial_velocity=0, acceleration=1.0)
        strategy = AccelerationTemporalStrategy(config)

        coords = [(0, 0), (2, 0), (6, 0)]  # distances: 2, 4
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # First segment: s=2, v0=0, a=1: t = sqrt(2*2/1) = 2
        # Second segment: s=4, v0=0, a=1: t = sqrt(2*4/1) = sqrt(8) = 2*sqrt(2)
        expected_time_1 = math.sqrt(4)  # 2.0
        expected_time_2 = expected_time_1 + math.sqrt(8)  # 2.0 + 2.828...

        assert len(result.time_stamps) == 3
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(expected_time_1, abs=1e-6)
        assert result.time_stamps[2] == pytest.approx(expected_time_2, abs=1e-6)

    def test_variable_acceleration(self):
        """Test with different acceleration values for each segment."""
        config = MockVariableConfig(initial_velocity=0, accelerations=[1.0, 4.0])
        strategy = AccelerationTemporalStrategy(config)

        coords = [(0, 0), (2, 0), (4, 0)]  # distances: 2, 2
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # First segment: s=2, v0=0, a=1: t = sqrt(4) = 2
        # Second segment: s=2, v0=0, a=4: t = sqrt(1) = 1
        expected_time_1 = 2.0
        expected_time_2 = expected_time_1 + 1.0  # 3.0

        assert len(result.time_stamps) == 3
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(expected_time_1, abs=1e-6)
        assert result.time_stamps[2] == pytest.approx(expected_time_2, abs=1e-6)

    def test_negative_acceleration_deceleration(self):
        """Test with negative acceleration (deceleration)."""
        config = MockConfig(initial_velocity=10.0, acceleration=-2.0)
        strategy = AccelerationTemporalStrategy(config)

        coords = [(0, 0), (20, 0)]  # distance: 20
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # t = (-v0 + sqrt(v0^2 + 2*a*s)) / a
        # t = (-10 + sqrt(100 + 2*(-2)*20)) / (-2)
        # t = (-10 + sqrt(100 - 80)) / (-2) = (-10 + sqrt(20)) / (-2)
        expected_time = (-10 + math.sqrt(20)) / (-2)

        assert len(result.time_stamps) == 2
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(expected_time, abs=1e-6)

    def test_single_point_trajectory(self):
        """Test with trajectory containing only one point."""
        config = MockConfig(initial_velocity=5.0, acceleration=2.0)
        strategy = AccelerationTemporalStrategy(config)

        coords = [(0, 0)]
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # Single point should have only one timestamp
        assert len(result.time_stamps) == 1
        assert result.time_stamps[0] == 0

    def test_empty_trajectory(self):
        """Test with empty trajectory."""
        config = MockConfig(initial_velocity=5.0, acceleration=2.0)
        strategy = AccelerationTemporalStrategy(config)

        coords = []
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # Empty trajectory should have only the initial timestamp
        assert len(result.time_stamps) == 1
        assert result.time_stamps[0] == 0

    def test_non_zero_starting_time(self):
        """Test with non-zero starting time (tmin)."""
        config = MockConfig(initial_velocity=0, acceleration=1.0, tmin=5.0)
        strategy = AccelerationTemporalStrategy(config)

        coords = [(0, 0), (2, 0)]  # distance: 2
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # Time calculation should be added to tmin
        segment_time = math.sqrt(4)  # 2.0
        expected_final_time = 5.0 + segment_time

        assert len(result.time_stamps) == 2
        assert result.time_stamps[0] == 5.0
        assert result.time_stamps[1] == pytest.approx(expected_final_time, abs=1e-6)

    def test_large_distances(self):
        """Test with large distance values."""
        config = MockConfig(initial_velocity=0, acceleration=0.5)
        strategy = AccelerationTemporalStrategy(config)

        coords = [(0, 0), (100, 0)]  # distance: 100
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # t = sqrt(2*s/a) = sqrt(2*100/0.5) = sqrt(400) = 20
        expected_time = 20.0

        assert len(result.time_stamps) == 2
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(expected_time, abs=1e-6)

    def test_small_distances(self):
        """Test with very small distance values."""
        config = MockConfig(initial_velocity=0, acceleration=10.0)
        strategy = AccelerationTemporalStrategy(config)

        coords = [(0, 0), (0.1, 0)]  # distance: 0.1
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # t = sqrt(2*s/a) = sqrt(2*0.1/10) = sqrt(0.02)
        expected_time = math.sqrt(0.02)

        assert len(result.time_stamps) == 2
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(expected_time, abs=1e-6)

    def test_diagonal_movement(self):
        """Test with diagonal movement (non-axis-aligned segments)."""
        config = MockConfig(initial_velocity=0, acceleration=1.0)
        strategy = AccelerationTemporalStrategy(config)

        coords = [(0, 0), (3, 4)]  # distance: 5 (3-4-5 triangle)
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # t = sqrt(2*s/a) = sqrt(2*5/1) = sqrt(10)
        expected_time = math.sqrt(10)

        assert len(result.time_stamps) == 2
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(expected_time, abs=1e-6)

    def test_complex_path(self):
        """Test with a complex multi-segment path."""
        config = MockVariableConfig(initial_velocity=1.0, accelerations=[2.0, 0.5, 1.0])
        strategy = AccelerationTemporalStrategy(config)

        # L-shaped path with different segment lengths
        coords = [(0, 0), (4, 0), (4, 3), (8, 3)]  # distances: 4, 3, 4
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # Segment 1: s=4, v0=1, a=2: t = (-1 + sqrt(1 + 16)) / 2 = (-1 + sqrt(17)) / 2
        t1 = (-1 + math.sqrt(17)) / 2
        # Segment 2: s=3, v0=1, a=0.5: t = (-1 + sqrt(1 + 3)) / 0.5 = (-1 + 2) / 0.5 = 2
        t2 = t1 + ((-1 + math.sqrt(4)) / 0.5)
        # Segment 3: s=4, v0=1, a=1: t = (-1 + sqrt(1 + 8)) / 1 = (-1 + 3) / 1 = 2
        t3 = t2 + ((-1 + math.sqrt(9)) / 1)

        assert len(result.time_stamps) == 4
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == pytest.approx(t1, abs=1e-6)
        assert result.time_stamps[2] == pytest.approx(t2, abs=1e-6)
        assert result.time_stamps[3] == pytest.approx(t3, abs=1e-6)

    def test_zero_distance_segment(self):
        """Test with a segment that has zero distance (duplicate points)."""
        config = MockConfig(initial_velocity=5.0, acceleration=1.0)
        strategy = AccelerationTemporalStrategy(config)

        coords = [(0, 0), (0, 0), (5, 0)]  # distances: 0, 5
        trajectory = MockTrajectory(coords)

        result = strategy(trajectory)

        # First segment has zero distance, so t = sqrt(0) = 0
        # Second segment: s=5, v0=5, a=1: t = (-5 + sqrt(25 + 10)) / 1
        t2 = (-5 + math.sqrt(35)) / 1

        assert len(result.time_stamps) == 3
        assert result.time_stamps[0] == 0
        assert result.time_stamps[1] == 0  # No time passes for zero distance
        assert result.time_stamps[2] == pytest.approx(t2, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
