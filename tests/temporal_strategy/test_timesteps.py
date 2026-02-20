import unittest
from unittest.mock import Mock
from trajgen.trajectory import Trajectory
from trajgen.config import Config
from trajgen.temporal_strategy import (
    ConstantTimeStepsTemporalStrategy,
    VariableTimeStepsTemporalStrategy,
)  # Adjust path


class TestFixedStepsTemporalStrategy(unittest.TestCase):

    def setUp(self):
        self.config = Mock(spec=Config)
        self.trajectory = Mock(spec=Trajectory)
        self.trajectory.ls = Mock()
        self.strategy = ConstantTimeStepsTemporalStrategy(self.config)

    def test_init_stores_config_reference(self):
        self.assertIs(self.strategy.config, self.config)

    def test_call_sets_correct_time_sequence(self):
        length = 5
        self.trajectory.ls.coords = [Mock()] * length
        tmin = 10
        time_step = 2
        expected_times = list(range(tmin, length * time_step, time_step))

        self.config.get_next_tmin.return_value = tmin
        self.strategy.time_step = time_step

        result = self.strategy(self.trajectory)

        self.trajectory.set_time.assert_called_once_with(expected_times)
        self.assertIs(result, self.trajectory)

    def test_call_empty_trajectory(self):
        self.trajectory.ls.coords = []
        self.config.get_next_tmin.return_value = 0
        self.strategy.time_step = 1

        result = self.strategy(self.trajectory)  # noqa
        self.trajectory.set_time.assert_called_once_with([])

    # def test_check_context_missing_time_step(self):
    #     delattr(self.config, "time_step")
    #     with self.assertRaises(ValueError):
    #         self.strategy.check_context()

    # def test_check_context_missing_get_next_tmin(self):
    #     delattr(self.config, "get_next_tmin")
    #     self.config.time_step = 1
    #     with self.assertRaises(ValueError):
    #         self.strategy.check_context()


class TestVariabledStepsTemporalStrategy(unittest.TestCase):

    def setUp(self):
        self.config = Mock(spec=Config)
        self.trajectory = Mock(spec=Trajectory)
        self.trajectory.ls = Mock()
        self.strategy = VariableTimeStepsTemporalStrategy(self.config)

    def test_init_stores_config_reference(self):
        self.assertIs(self.strategy.config, self.config)

    def test_call_single_point(self):
        """Length=1: no time_steps needed, just t_min."""
        self.trajectory.ls.coords = [Mock()]
        tmin = 5.0
        self.config.get_next_tmin.return_value = tmin

        result = self.strategy(self.trajectory)  # noqa
        self.trajectory.set_time.assert_called_once_with([tmin])

    def test_call_three_points(self):
        """Length=3: two time steps accumulated."""
        self.trajectory.ls.coords = [Mock()] * 3
        self.config.get_next_tmin.return_value = 0
        self.config.get_next_time_step = Mock(side_effect=[1.0, 2.5])

        result = self.strategy(self.trajectory)  # noqa
        self.trajectory.set_time.assert_called_once_with([0.0, 1.0, 3.5])
        self.assertEqual(self.config.get_next_time_step.call_count, 2)

    def test_call_empty_trajectory(self):
        self.trajectory.ls.coords = []
        self.config.get_next_tmin.return_value = 0

        result = self.strategy(self.trajectory)  # noqa
        self.trajectory.set_time.assert_called_once_with([])
