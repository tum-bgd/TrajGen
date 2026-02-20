import unittest
from unittest.mock import Mock

from shapely import LineString
from trajgen.trajectory import Trajectory
from trajgen.config import Config
from trajgen.temporal_strategy import ConstantTemporalStrategy  # Adjust import path


class TestFixedTemporalStrategy(unittest.TestCase):

    def setUp(self):
        """Create fresh instances before each test."""
        self.length = 10
        self.config = Mock(spec=Config)
        self.config.get_next_tmin.return_value = 0
        self.config.get_next_time_step = 0.1
        real_ls = LineString()
        real_traj = Trajectory(id=0, ls=real_ls)
        self.trajectory = Mock(spec=real_traj)
        self.strategy = ConstantTemporalStrategy(self.config)

    def test_init_stores_config_reference(self):
        """Verify constructor stores config reference."""
        self.assertIs(self.strategy.config, self.config)

    def test_call_sets_correct_time_sequence(self):
        """Test time sequence generation and assignment."""
        # Setup mocks
        self.trajectory.ls.coords = [Mock()] * self.length  # Simulate trajectory length
        tmin = 10
        time_step = 0.1
        expected_times = [tmin for i in range(self.length)]

        self.config.get_next_tmin.return_value = tmin
        self.config.get_next_time_step = time_step

        # Execute
        result = self.strategy(self.trajectory)

        # Verify
        self.trajectory.set_time.assert_called_once_with(expected_times)
        self.assertIs(result, self.trajectory)  # Returns same trajectory
        self.config.get_next_tmin.assert_called_once()

    def test_call_works_with_length_zero(self):
        """Test empty trajectory handling."""
        self.trajectory.ls.coords = []
        self.config.get_next_tmin.return_value = 0
        self.config.time_step = 1.0

        result = self.strategy(self.trajectory)

        self.trajectory.set_time.assert_called_once_with([])
        self.assertIs(result, self.trajectory)
