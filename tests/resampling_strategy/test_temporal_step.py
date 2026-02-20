import pytest
from datetime import datetime, timedelta
from shapely.geometry import LineString, Point
from trajgen.resampling_strategy.temporal_step import ConstantTemporalStepResampling
from trajgen.trajectory import Trajectory


class Config:
    """Mock config for testing."""

    def __init__(self, time_step_size):
        self.time_step_size = time_step_size

    def get_next_time_step_size(self):
        return self.time_step_size


class TestConstantTemporalStepResampling:
    """Test suite for FixedTimeStepResampling class."""

    def test_basic_resampling_datetime(self):
        """Test basic resampling with datetime timestamps."""
        ls = LineString([Point(0, 0), Point(100, 0)])
        t = [datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 10, 2)]  # 2 minutes
        traj = Trajectory(id="test1", ls=ls, t=t)

        config = Config(time_step_size=30.0)  # 30 second steps
        resampler = ConstantTemporalStepResampling(config)
        result = resampler(traj)

        assert isinstance(result.ls, LineString)
        assert len(result.t) == 5  # 0, 30, 60, 90, 120 seconds
        assert result.id == "test1"

        # Check that timestamps are correct
        expected_times = [
            datetime(2026, 1, 1, 10, 0),
            datetime(2026, 1, 1, 10, 0, 30),
            datetime(2026, 1, 1, 10, 1),
            datetime(2026, 1, 1, 10, 1, 30),
            datetime(2026, 1, 1, 10, 2),
        ]
        for res_t, exp_t in zip(result.t, expected_times):
            assert res_t == exp_t

    def test_basic_resampling_integer(self):
        """Test basic resampling with integer timestamps."""
        ls = LineString([Point(0, 0), Point(100, 0)])
        t = [0, 120]  # 0 to 120 seconds
        traj = Trajectory(id="test2", ls=ls, t=t)

        config = Config(time_step_size=30.0)  # 30 second steps
        resampler = ConstantTemporalStepResampling(config)
        result = resampler(traj)

        assert isinstance(result.ls, LineString)
        assert len(result.t) == 5  # 0, 30, 60, 90, 120
        assert result.id == "test2"

        # Check that timestamps are correct
        expected_times = [0, 30, 60, 90, 120]
        for res_t, exp_t in zip(result.t, expected_times):
            assert res_t == exp_t

    def test_no_timestamps(self):
        """Test resampling trajectory without timestamps."""
        ls = LineString([Point(0, 0), Point(50, 0)])
        traj = Trajectory(id="test3", ls=ls, t=None)

        config = Config(time_step_size=10.0)
        resampler = ConstantTemporalStepResampling(config)
        result = resampler(traj)

        # Should return original trajectory unchanged
        assert result.t is None
        assert result.ls == traj.ls
        assert result.id == "test3"

    def test_single_timestamp(self):
        """Test trajectory with only one timestamp."""
        ls = LineString([Point(0, 0), Point(50, 0)])
        t = [datetime(2026, 1, 1, 10, 0)]
        traj = Trajectory(id="test4", ls=ls, t=t)

        config = Config(time_step_size=10.0)
        resampler = ConstantTemporalStepResampling(config)
        result = resampler(traj)

        # Should return original trajectory unchanged
        assert result.t == t
        assert result.ls == traj.ls
        assert result.id == "test4"

    def test_zero_duration_datetime(self):
        """Test trajectory with zero duration (identical timestamps)."""
        ls = LineString([Point(0, 0), Point(50, 0)])
        t_same = datetime(2026, 1, 1, 10, 0)
        t = [t_same, t_same]
        traj = Trajectory(id="test5", ls=ls, t=t)

        config = Config(time_step_size=10.0)
        resampler = ConstantTemporalStepResampling(config)
        result = resampler(traj)

        # Should return original trajectory unchanged
        assert result.t == t
        assert result.ls == traj.ls
        assert result.id == "test5"

    def test_zero_duration_integer(self):
        """Test trajectory with zero duration (identical integer timestamps)."""
        ls = LineString([Point(0, 0), Point(50, 0)])
        t = [10, 10]
        traj = Trajectory(id="test6", ls=ls, t=t)

        config = Config(time_step_size=5.0)
        resampler = ConstantTemporalStepResampling(config)
        result = resampler(traj)

        # Should return original trajectory unchanged
        assert result.t == t
        assert result.ls == traj.ls
        assert result.id == "test6"

    def test_invalid_step_size(self):
        """Test with invalid (negative) step size."""
        ls = LineString([Point(0, 0), Point(50, 0)])
        t = [0, 100]
        traj = Trajectory(id="test7", ls=ls, t=t)

        config = Config(time_step_size=-10.0)
        resampler = ConstantTemporalStepResampling(config)

        with pytest.raises(ValueError, match="Time step size must be positive"):
            resampler(traj)

    def test_zero_step_size(self):
        """Test with zero step size."""
        ls = LineString([Point(0, 0), Point(50, 0)])
        t = [0, 100]
        traj = Trajectory(id="test8", ls=ls, t=t)

        config = Config(time_step_size=0.0)
        resampler = ConstantTemporalStepResampling(config)

        with pytest.raises(ValueError, match="Time step size must be positive"):
            resampler(traj)

    def test_step_larger_than_duration_datetime(self):
        """Test when step size is larger than trajectory duration."""
        ls = LineString([Point(0, 0), Point(50, 0)])
        t = [datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 10, 0, 5)]  # 5 seconds
        traj = Trajectory(id="test9", ls=ls, t=t)

        config = Config(time_step_size=10.0)  # 10 seconds > 5 seconds duration
        resampler = ConstantTemporalStepResampling(config)
        result = resampler(traj)

        # Should have start and end points
        assert len(result.t) == 2
        assert result.t[0] == t[0]
        assert result.t[1] == t[0] + timedelta(seconds=10)

    def test_step_larger_than_duration_integer(self):
        """Test when step size is larger than trajectory duration with integers."""
        ls = LineString([Point(0, 0), Point(50, 0)])
        t = [0, 5]  # 5 second duration
        traj = Trajectory(id="test10", ls=ls, t=t)

        config = Config(time_step_size=10.0)  # 10 seconds > 5 seconds duration
        resampler = ConstantTemporalStepResampling(config)
        result = resampler(traj)

        # Should have start and one more point
        assert len(result.t) == 2
        assert result.t[0] == 0
        assert result.t[1] == 10

    def test_fractional_steps_datetime(self):
        """Test with fractional step sizes and datetime."""
        ls = LineString([Point(0, 0), Point(100, 0)])
        t = [datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 10, 0, 10)]  # 10 seconds
        traj = Trajectory(id="test11", ls=ls, t=t)

        config = Config(time_step_size=2.5)  # 2.5 second steps
        resampler = ConstantTemporalStepResampling(config)
        result = resampler(traj)

        assert len(result.t) == 5  # 0, 2.5, 5, 7.5, 10 seconds
        expected_times = [
            datetime(2026, 1, 1, 10, 0),
            datetime(2026, 1, 1, 10, 0, 2, 500000),  # 2.5 seconds
            datetime(2026, 1, 1, 10, 0, 5),
            datetime(2026, 1, 1, 10, 0, 7, 500000),  # 7.5 seconds
            datetime(2026, 1, 1, 10, 0, 10),
        ]
        for res_t, exp_t in zip(result.t, expected_times):
            assert res_t == exp_t

    def test_fractional_steps_integer(self):
        """Test with fractional step sizes and integer timestamps."""
        ls = LineString([Point(0, 0), Point(100, 0)])
        t = [0, 10]  # 10 second duration
        traj = Trajectory(id="test12", ls=ls, t=t)

        config = Config(time_step_size=2.5)  # 2.5 second steps
        resampler = ConstantTemporalStepResampling(config)
        result = resampler(traj)

        assert len(result.t) == 5  # 0, 2.5, 5, 7.5, 10
        expected_times = [0, 2.5, 5, 7.5, 10]
        for res_t, exp_t in zip(result.t, expected_times):
            assert res_t == pytest.approx(exp_t, abs=1e-6)

    def test_spatial_interpolation(self):
        """Test that spatial coordinates are correctly interpolated."""
        ls = LineString([Point(0, 0), Point(100, 0)])
        t = [0, 100]
        traj = Trajectory(id="test13", ls=ls, t=t)

        config = Config(time_step_size=25.0)  # 25 second steps
        resampler = ConstantTemporalStepResampling(config)
        result = resampler(traj)

        coords = list(result.ls.coords)
        expected_coords = [
            (0.0, 0.0),
            (25.0, 0.0),
            (50.0, 0.0),
            (75.0, 0.0),
            (100.0, 0.0),
        ]

        for res_coord, exp_coord in zip(coords, expected_coords):
            assert res_coord[0] == pytest.approx(exp_coord[0], abs=1e-6)
            assert res_coord[1] == pytest.approx(exp_coord[1], abs=1e-6)

    def test_curved_trajectory(self):
        """Test resampling on curved (non-straight) trajectory."""
        # Create L-shaped path
        ls = LineString([Point(0, 0), Point(50, 0), Point(50, 50)])
        t = [0, 50, 100]  # 50 seconds per segment
        traj = Trajectory(id="test14", ls=ls, t=t)

        config = Config(time_step_size=25.0)  # 25 second steps
        resampler = ConstantTemporalStepResampling(config)
        result = resampler(traj)

        assert len(result.t) == 5  # 0, 25, 50, 75, 100
        assert len(result.ls.coords) == 5
        coords = list(result.ls.coords)

        # At t=25, should be halfway along first segment
        assert coords[1][0] == pytest.approx(25.0, abs=1e-6)
        assert coords[1][1] == pytest.approx(0.0, abs=1e-6)

        # At t=75, should be halfway along second segment
        assert coords[3][0] == pytest.approx(50.0, abs=1e-6)
        assert coords[3][1] == pytest.approx(25.0, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
