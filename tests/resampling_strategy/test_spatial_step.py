import pytest
from datetime import datetime, timedelta
from shapely.geometry import LineString, Point
from trajgen.resampling_strategy.spatial_step import ConstantSpatialStepResampling
from trajgen.trajectory import Trajectory


class Config:
    """Mock config for testing."""

    def __init__(self, spatial_step_size):
        self.spatial_step_size = spatial_step_size

    def get_next_spatial_step_size(self):
        return self.spatial_step_size


class TestConstantSpatialStepResampling:
    """Test suite for FixedSpatialResampling class."""

    def test_basic_resampling(self):
        """Test basic resampling with simple straight line."""
        ls = LineString([Point(0, 0), Point(100, 0)])
        t = [datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 10, 10)]
        traj = Trajectory(id="test1", ls=ls, t=t)

        config = Config(spatial_step_size=10.0)
        resampler = ConstantSpatialStepResampling(config)
        result = resampler(traj)

        assert isinstance(result.ls, LineString)
        assert len(result.ls.coords) == 11  # 0, 10, 20, ..., 100
        assert result.ls.length == pytest.approx(100.0, abs=1e-6)
        assert result.id == "test1"

    def test_no_timestamps(self):
        """Test resampling trajectory without timestamps."""
        ls = LineString([Point(0, 0), Point(50, 0)])
        traj = Trajectory(id="test2", ls=ls, t=None)

        config = Config(spatial_step_size=10.0)
        resampler = ConstantSpatialStepResampling(config)
        result = resampler(traj)

        assert result.t is None
        assert len(result.ls.coords) == 6  # 0, 10, 20, 30, 40, 50

    def test_zero_length_trajectory(self):
        """Test trajectory with zero length (single point)."""
        ls = LineString([Point(5, 5), Point(5, 5)])
        t = [datetime(2026, 1, 1, 10, 0)]
        traj = Trajectory(id="test3", ls=ls, t=t)

        config = Config(spatial_step_size=10.0)
        resampler = ConstantSpatialStepResampling(config)
        result = resampler(traj)

        assert result == traj  # Should return unchanged

    def test_invalid_step_size(self):
        """Test that negative step size raises ValueError."""
        ls = LineString([Point(0, 0), Point(10, 0)])
        traj = Trajectory(id="test4", ls=ls, t=None)

        config = Config(spatial_step_size=-5.0)
        resampler = ConstantSpatialStepResampling(config)

        with pytest.raises(ValueError, match="Spatial step size must be positive"):
            resampler(traj)

    def test_zero_step_size(self):
        """Test that zero step size raises ValueError."""
        ls = LineString([Point(0, 0), Point(10, 0)])
        traj = Trajectory(id="test5", ls=ls, t=None)

        config = Config(spatial_step_size=0.0)
        resampler = ConstantSpatialStepResampling(config)

        with pytest.raises(ValueError, match="Spatial step size must be positive"):
            resampler(traj)

    def test_non_linestring_geometry(self):
        """Test that non-LineString geometry raises ValueError."""
        from shapely.geometry import Point

        point = Point(0, 0)
        traj = Trajectory(id="test6", ls=point, t=None)

        config = Config(spatial_step_size=10.0)
        resampler = ConstantSpatialStepResampling(config)

        with pytest.raises(ValueError, match="must be a LineString"):
            resampler(traj)

    def test_curved_trajectory(self):
        """Test resampling on curved (non-straight) trajectory."""
        # Create L-shaped path
        ls = LineString([Point(0, 0), Point(50, 0), Point(50, 50)])
        t = [
            datetime(2026, 1, 1, 10, 0),
            datetime(2026, 1, 1, 10, 5),
            datetime(2026, 1, 1, 10, 10),
        ]
        traj = Trajectory(id="test7", ls=ls, t=t)

        config = Config(spatial_step_size=25.0)
        resampler = ConstantSpatialStepResampling(config)
        result = resampler(traj)

        assert len(result.ls.coords) == 5  # 0, 25, 50, 75, 100
        assert result.ls.length == pytest.approx(100.0, abs=1e-6)

    def test_time_interpolation_uniform(self):
        """Test that timestamps are interpolated linearly."""
        ls = LineString([Point(0, 0), Point(100, 0)])
        t_start = datetime(2026, 1, 1, 10, 0)
        t_end = datetime(2026, 1, 1, 11, 0)  # 1 hour later
        traj = Trajectory(id="test8", ls=ls, t=[t_start, t_end])

        config = Config(spatial_step_size=25.0)
        resampler = ConstantSpatialStepResampling(config)
        result = resampler(traj)

        expected_times = [
            t_start,
            t_start + timedelta(minutes=15),
            t_start + timedelta(minutes=30),
            t_start + timedelta(minutes=45),
            t_end,
        ]

        # Compare timestamps (allowing small floating point differences)
        for res_t, exp_t in zip(result.t, expected_times):
            assert abs((res_t - exp_t).total_seconds()) < 1.0

    def test_identical_timestamps(self):
        """Test trajectory where start and end time are identical."""
        ls = LineString([Point(0, 0), Point(100, 0)])
        t_same = datetime(2026, 1, 1, 10, 0)
        traj = Trajectory(id="test9", ls=ls, t=[t_same, t_same])

        config = Config(spatial_step_size=50.0)
        resampler = ConstantSpatialStepResampling(config)
        result = resampler(traj)

        # All timestamps should be the same
        assert all(t == t_same for t in result.t)

    def test_step_larger_than_trajectory(self):
        """Test when step size is larger than trajectory length."""
        ls = LineString([Point(0, 0), Point(5, 0)])
        t = [datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 10, 1)]
        traj = Trajectory(id="test10", ls=ls, t=t)

        config = Config(spatial_step_size=100.0)
        resampler = ConstantSpatialStepResampling(config)
        result = resampler(traj)

        # Should have start and end points when step > length
        assert len(result.ls.coords) == 2
        coords = list(result.ls.coords)
        assert coords[0] == pytest.approx((0.0, 0.0), abs=1e-6)
        assert coords[1] == pytest.approx((5.0, 0.0), abs=1e-6)

    def test_fractional_step_coverage(self):
        """Test when trajectory length is not divisible by step size."""
        ls = LineString([Point(0, 0), Point(55, 0)])  # 55 units long
        traj = Trajectory(id="test11", ls=ls, t=None)

        config = Config(spatial_step_size=10.0)
        resampler = ConstantSpatialStepResampling(config)
        result = resampler(traj)

        # Should have points at 0, 10, 20, 30, 40, 50, and 55 (endpoint included)
        assert len(result.ls.coords) == 7
        coords = list(result.ls.coords)
        assert coords[-1][0] == pytest.approx(55.0, abs=1e-6)

    def test_3d_trajectory(self):
        """Test resampling 3D trajectory (with Z coordinates)."""
        ls = LineString([Point(0, 0, 0), Point(100, 0, 100)])
        t = [datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 10, 10)]
        traj = Trajectory(id="test12", ls=ls, t=t)

        config = Config(spatial_step_size=50.0)
        resampler = ConstantSpatialStepResampling(config)
        result = resampler(traj)

        assert len(result.ls.coords) == 3  # 0, 50, 100
        # Check Z coordinates are interpolated
        coords = list(result.ls.coords)
        assert coords[1][2] == pytest.approx(50.0, abs=1e-6)

    def test_multipoint_trajectory_with_times(self):
        """Test trajectory with multiple intermediate points and times."""
        ls = LineString([Point(0, 0), Point(10, 0), Point(20, 0), Point(30, 0)])
        times = [
            datetime(2026, 1, 1, 10, 0),
            datetime(2026, 1, 1, 10, 1),
            datetime(2026, 1, 1, 10, 2),
            datetime(2026, 1, 1, 10, 3),
        ]
        traj = Trajectory(id="test14", ls=ls, t=times)

        config = Config(spatial_step_size=5.0)
        resampler = ConstantSpatialStepResampling(config)
        result = resampler(traj)

        assert len(result.ls.coords) == 7  # 0, 5, 10, 15, 20, 25, 30
        assert result.t[0] == times[0]
        assert result.t[-1] == times[-1]

    def test_multipoint_trajectory_with_different_velocities(self):
        """Test trajectory with varying speeds along the path."""
        ls = LineString([Point(0, 0), Point(10, 0), Point(20, 0)])
        times = [
            datetime(2026, 1, 1, 10, 0),
            datetime(2026, 1, 1, 10, 1),  # Fast segment (10 units in 1 min)
            datetime(2026, 1, 1, 10, 3),  # Slow segment (10 units in 2 min)
        ]
        traj = Trajectory(id="test15", ls=ls, t=times)

        config = Config(spatial_step_size=5.0)
        resampler = ConstantSpatialStepResampling(config)
        result = resampler(traj)

        assert len(result.ls.coords) == 5  # 0, 5, 10, 15, 20
        expected_times = [
            times[0],
            times[0] + timedelta(seconds=30),  # Midpoint of first segment
            times[1],
            times[1] + timedelta(minutes=1),  # Midpoint of second segment
            times[2],
        ]
        for res_t, exp_t in zip(result.t, expected_times):
            assert abs((res_t - exp_t).total_seconds()) < 1.0

    def test_integer_timestamps(self):
        """Test trajectory with integer timestamps instead of datetime objects."""
        ls = LineString([Point(0, 0), Point(20, 0)])
        t = [0, 10]  # Simple integer timestamps
        traj = Trajectory(id="test16", ls=ls, t=t)

        config = Config(spatial_step_size=5.0)
        resampler = ConstantSpatialStepResampling(config)
        result = resampler(traj)

        assert len(result.ls.coords) == 5  # 0, 5, 10, 15, 20
        coords = list(result.ls.coords)
        assert coords[0] == pytest.approx((0.0, 0.0), abs=1e-6)
        assert coords[-1] == pytest.approx((20.0, 0.0), abs=1e-6)

        # Test that integer interpolation works correctly
        expected_times = [0, 2.5, 5, 7.5, 10]
        for res_t, exp_t in zip(result.t, expected_times):
            assert res_t == pytest.approx(exp_t, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
