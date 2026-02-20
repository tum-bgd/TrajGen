import pytest
import numpy as np
from unittest.mock import Mock, call, patch
from shapely.geometry import LineString
from trajgen.point_generator import PointGenerator
from trajgen.trajectory import Trajectory
from trajgen.config import Config
from trajgen.spatial_strategy.polynomial_curves import PolynomialCurvesStrategy


class TestPolynomialCurvesStrategy:

    def setup_method(self):
        """Setup common mocks."""
        self.config = Mock(spec=Config)
        self.config.num_control_points = 5
        self.config.closed_loop = False
        self.config.point_generator = Mock(PointGenerator)
        self.config.get_next_length.return_value = 20

        # Mock control points with known coordinates
        self.mock_points = [
            Mock(x=0.1, y=0.1),
            Mock(x=0.4, y=0.8),
            Mock(x=0.7, y=0.3),
            Mock(x=0.2, y=0.6),
            Mock(x=0.9, y=0.9),
        ]
        self.mock_points2 = [
            Mock(x=0.2, y=0.2),
            Mock(x=0.5, y=0.7),
            Mock(x=0.8, y=0.4),
            Mock(x=0.3, y=0.5),
            Mock(x=0.6, y=0.8),
        ]
        self.config.point_generator.side_effect = [self.mock_points, self.mock_points2]

    def test_init_stores_config(self):
        """Test __init__ stores config correctly."""
        strategy = PolynomialCurvesStrategy(self.config)
        assert strategy.config == self.config

    def test_call_generates_trajectory(self):
        """Test generates Trajectory with correct ID."""
        strategy = PolynomialCurvesStrategy(self.config)
        result = strategy(id=42)

        assert isinstance(result, Trajectory)
        assert result.id == 42
        assert isinstance(result.ls, LineString)
        assert len(result.ls.coords) == 20  # get_next_length()

    def test_trajectory_points_in_bounds(self):
        """Test all generated points are clipped to [0,1] bounds."""
        strategy = PolynomialCurvesStrategy(self.config)
        result = strategy(id=1)

        x_coords, y_coords = result.ls.xy
        assert np.all(np.array(x_coords) >= 0.0)
        assert np.all(np.array(x_coords) <= 1.0)
        assert np.all(np.array(y_coords) >= 0.0)
        assert np.all(np.array(y_coords) <= 1.0)

    def test_calls_point_generator_correctly(self):
        """Test point_generator.generate called with num_control_points."""
        strategy = PolynomialCurvesStrategy(self.config)
        strategy(1)

        self.config.point_generator.assert_called_once_with(5)

    def test_calls_get_next_length(self):
        """Test config.get_next_length called once per trajectory."""
        strategy = PolynomialCurvesStrategy(self.config)
        strategy(1)

        self.config.get_next_length.assert_called_once()

    def test_spline_creation(self):
        """Test CubicSpline called with correct parameters."""
        strategy = PolynomialCurvesStrategy(self.config)
        with (
            patch(
                "trajgen.spatial_strategy.polynomial_curves.CubicSpline"
            ) as mock_spline,
            patch(
                "trajgen.spatial_strategy.polynomial_curves.np.linspace",
                return_value=np.linspace(0, 1, 5),
            ) as mock_linspace,
        ):
            strategy(1)  # Now mocks work!clear

            # Check t_control: linspace(0,1,num_control_points)
            mock_linspace.assert_has_calls(
                [
                    # calls for t_control
                    call(0.0, 1.0, 5),
                    # calls for t
                    call(0.0, 1.0, 20),
                ],
                any_order=False,
            )

            expected_t = np.linspace(0.0, 1.0, 5)
            x_coords = [0.1, 0.4, 0.7, 0.2, 0.9]
            y_coords = [0.1, 0.8, 0.3, 0.6, 0.9]
            # Check CubicSpline calls for x and y
            calls = mock_spline.call_args_list
            assert len(calls) == 2

            args_x, kwargs_x = calls[0]  # call = (args_tuple, kwargs_dict)
            t_x, coords_x = args_x[0], args_x[1]  # args_tuple = (t_array, coords_list)

            np.testing.assert_array_equal(t_x, expected_t)
            assert coords_x == x_coords
            assert kwargs_x == {"bc_type": "clamped"}

            args_y, kwargs_y = calls[1]
            t_y, coords_y = args_y[0], args_y[1]

            np.testing.assert_array_equal(t_y, expected_t)
            assert coords_y == y_coords
            assert kwargs_y == {"bc_type": "clamped"}

    def test_different_trajectory_ids_produce_different_paths(self):
        """Test different IDs produce different trajectories."""
        strategy = PolynomialCurvesStrategy(self.config)

        traj1 = strategy(1)
        traj2 = strategy(2)

        # Same length but different coordinates
        assert len(traj1.ls.coords) == len(traj2.ls.coords)
        assert not np.array_equal(traj1.ls.coords, traj2.ls.coords)

    def test_closed_loop_reuses_first_point(self):
        """Test closed_loop=True sets last control point = first."""
        self.config.closed_loop = True
        strategy = PolynomialCurvesStrategy(self.config)

        strategy(1)

        # Verify last point equals first
        first_point = self.mock_points[0]
        assert self.mock_points[-1] == first_point

    @pytest.mark.parametrize("length", [5, 15, 30])
    def test_variable_trajectory_lengths(self, length):
        """Test get_next_length() produces correct number of points."""
        self.config.get_next_length.return_value = length
        strategy = PolynomialCurvesStrategy(self.config)
        result = strategy(1)

        assert len(result.ls.coords) == length

    def test_spline_evaluation_correct_num_points(self):
        """Test spline produces exactly n_points."""
        self.config.get_next_length.return_value = 100
        strategy = PolynomialCurvesStrategy(self.config)
        result = strategy(1)
        assert len(result.ls.coords) == 100
